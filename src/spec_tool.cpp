// trilane-spec: Tri-Lane prefill (NPU) + CPU decode with MTP speculative decoding.
// The MTP head drafts tokens on the CPU; the CPU decode context verifies k+1 tokens per pass over the weights.
// Modes:
//   spec (default): NPU prefill -> state handoff -> speculative (or plain, --nmax 0) decode on the CPU
//   batchcost:      time one decode batch of B tokens on the CPU decode context (verification cost)

#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool g_verbose = false;
static void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static double now_ms() { return llama_time_us() / 1000.0; }

struct args {
    std::string model, prompt_file, mode = "spec", prefill_dev = "HTP0", alias = "auto";
    int n_gen = 128, n_ctx = 2048, threads = 8, nmax = 3, ubatch = 512;
    std::vector<int> batches = { 1, 2, 3, 4, 5, 8 };
    bool check = false;
    bool prefill_on_decode = false; // run the prompt through the CPU decode context (exercises batched CPU kernels)
};

static void usage() {
    fprintf(stderr, "usage: trilane-spec -m model.gguf -f prompt.txt [--mode spec|batchcost] [--prefill-dev HTP0|CPU] [--nmax K (0 = no spec)]\n"
                    "       [-n n_gen] [-c n_ctx] [-t threads] [-ub ubatch] [--alias auto|on|off] [--batches 1,2,4,8] [--check] [-v]\n");
    exit(1);
}

// ---- weight aliasing (same rules as trilane engine.cpp; copied so lane 1 can rewrite the engine freely) ----
typedef void * (*htptile_resolver_t)(const char * name, const ggml_tensor * t, void * user);
typedef void (*htptile_set_resolver_t)(htptile_resolver_t fn, void * user);
typedef ggml_backend_buffer_type_t (*htptile_buft_t)(void);

static void * resolve_from_model(const char * name, const ggml_tensor * t, void * user) {
    const ggml_tensor * src = llama_model_get_tensor((const llama_model *) user, name);
    if (!src || src->type != t->type || ggml_nelements(src) != ggml_nelements(t)) return nullptr;
    return src->data;
}

static bool has_experts(const llama_model * m) {
    char name[64];
    for (int i = 0; i < llama_model_n_layer(m); i++) {
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", i);
        if (llama_model_get_tensor(m, name)) return true;
    }
    return false;
}

static std::vector<std::string> alias_candidates(const std::string & path, const llama_model * mp, bool experts_only) {
    std::vector<std::string> out;
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return out;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const std::string name = gguf_get_tensor_name(g, i);
        if (name.rfind("token_embd", 0) == 0) continue;
        if (experts_only && name.find("_exps") == std::string::npos) continue;
        const ggml_tensor * t = llama_model_get_tensor(mp, name.c_str());
        if (!t || t->type != GGML_TYPE_Q4_0 || t->ne[3] != 1 || t->ne[0] % 32 || t->ne[1] % 32 || !ggml_is_contiguous(t)) continue;
        if (!t->buffer || !strstr(ggml_backend_buffer_name(t->buffer), "HTP")) continue;
        out.push_back(name);
    }
    gguf_free(g);
    return out;
}

static std::vector<std::string> exact_name_patterns(const std::vector<std::string> & names) {
    std::vector<std::string> pats;
    std::string cur;
    for (size_t i = 0; i < names.size(); i++) {
        std::string esc;
        for (char c : names[i]) { if (c == '.') esc += "\\."; else esc += c; }
        cur += (cur.empty() ? "" : "|") + esc;
        if ((i + 1) % 64 == 0 || i + 1 == names.size()) { pats.push_back("^(" + cur + ")$"); cur.clear(); }
    }
    return pats;
}

int main(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) usage(); return argv[++i]; };
        if      (k == "-m") a.model = next();
        else if (k == "-f") a.prompt_file = next();
        else if (k == "--mode") a.mode = next();
        else if (k == "--prefill-dev") a.prefill_dev = next();
        else if (k == "--alias") a.alias = next();
        else if (k == "--nmax") a.nmax = atoi(next());
        else if (k == "-n") a.n_gen = atoi(next());
        else if (k == "-c") a.n_ctx = atoi(next());
        else if (k == "-t") a.threads = atoi(next());
        else if (k == "-ub") a.ubatch = atoi(next());
        else if (k == "--batches") { a.batches.clear(); std::stringstream ss(next()); for (std::string s; std::getline(ss, s, ',');) a.batches.push_back(atoi(s.c_str())); }
        else if (k == "--check") a.check = true;
        else if (k == "--prefill-on-decode") a.prefill_on_decode = true;
        else if (k == "-v") g_verbose = true;
        else usage();
    }
    if (a.model.empty() || a.prompt_file.empty()) usage();

    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    ggml_backend_load_all();

    // ---- models: P (prefill, NPU) and D (decode, CPU, MTP layers loaded, weights aliased to P where needed) ----
    const bool use_npu = a.prefill_dev != "CPU";
    llama_model * mp = nullptr;
    std::vector<ggml_backend_dev_t> npu_devs;
    if (use_npu) {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(a.prefill_dev.c_str());
        if (!d) { fprintf(stderr, "device %s not found\n", a.prefill_dev.c_str()); return 1; }
        npu_devs = { d, nullptr };
        llama_model_params pp = llama_model_default_params();
        pp.load_mode = LLAMA_LOAD_MODE_NONE;
        pp.devices = npu_devs.data();
        pp.n_gpu_layers = -1;
        double t0 = now_ms();
        mp = llama_model_load_from_file(a.model.c_str(), pp);
        if (!mp) return 1;
        fprintf(stderr, "[spec] prefill model on %s loaded in %.0f ms\n", a.prefill_dev.c_str(), now_ms() - t0);
    }

    std::vector<ggml_backend_dev_t> no_devs = { nullptr };
    llama_model_params dp = llama_model_default_params();
    dp.load_mode = LLAMA_LOAD_MODE_NONE;
    dp.devices = no_devs.data();
    dp.n_gpu_layers = 0;
    dp.load_mtp = a.nmax > 0 && a.mode == "spec";
    std::vector<std::string> pats;
    std::vector<llama_model_tensor_buft_override> ovr;
    int n_alias = 0;
    if (use_npu && a.alias != "off") {
        ggml_backend_reg_t cpu = ggml_backend_reg_by_name("CPU");
        auto set_resolver = (htptile_set_resolver_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_htptile_set_resolver");
        auto get_buft = (htptile_buft_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_htptile_buffer_type");
        const bool moe = has_experts(mp);
        const bool dense = !moe && (a.alias == "on" || llama_model_size(mp) > (6ull << 30));
        const auto names = (moe || dense) ? alias_candidates(a.model, mp, moe) : std::vector<std::string>();
        if (set_resolver && get_buft && !names.empty()) {
            set_resolver(resolve_from_model, (void *) mp);
            pats = exact_name_patterns(names);
            for (auto & p : pats) ovr.push_back({ p.c_str(), get_buft() });
            ovr.push_back({ nullptr, nullptr });
            dp.tensor_buft_overrides = ovr.data();
            dp.load_mode = LLAMA_LOAD_MODE_MMAP;
            _putenv_s("LLAMA_MMAP_NO_PREFETCH", "1");
            n_alias = (int) names.size();
        }
    }
    double t0 = now_ms();
    llama_model * md = llama_model_load_from_file(a.model.c_str(), dp);
    if (!md) return 1;
    fprintf(stderr, "[spec] decode model on CPU loaded in %.0f ms (%d tensors aliased to NPU memory, mtp=%d)\n", now_ms() - t0, n_alias, (int) dp.load_mtp);
    if (!mp) mp = md;

    auto ctx_params = [&](int threads) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = a.n_ctx; cp.n_batch = std::max(2048, a.ubatch); cp.n_ubatch = a.ubatch; cp.n_seq_max = 1;
        cp.n_threads = threads; cp.n_threads_batch = threads; cp.no_perf = true;
        return cp;
    };
    llama_context_params cpp = ctx_params(4);
    cpp.op_offload = true;
    llama_context_params cpd = ctx_params(a.threads);
    cpd.op_offload = false;
    llama_context * cd = llama_init_from_model(md, cpd);
    llama_context * cp = mp != md ? llama_init_from_model(mp, cpp) : cd;
    if (!cd || !cp) return 1;
    if (a.prefill_on_decode) cp = cd;

    const llama_vocab * vocab = llama_model_get_vocab(md);
    std::string text;
    { std::ifstream f(a.prompt_file, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }
    std::vector<llama_token> inp = common_tokenize(cd, text, true, true);
    if ((int) inp.size() + a.n_gen + 16 > a.n_ctx) { fprintf(stderr, "prompt too long for -c\n"); return 1; }

    // ---- batchcost: cost of one decode call with B tokens on the CPU decode context ----
    if (a.mode == "batchcost") {
        // prefill the prompt (NPU when available) and hand the state over, like the engine does
        common_batch bp(cp);
        for (size_t i = 0; i < inp.size(); i++) bp.add(inp[i], (llama_pos) i, 0, i + 1 == inp.size());
        llama_process(cp, LLAMA_PROCESS_TYPE_DECODE, bp.get());
        if (cp != cd) {
            std::vector<uint8_t> st(llama_state_seq_get_size(cp, 0));
            llama_state_seq_get_data(cp, st.data(), st.size(), 0);
            llama_state_seq_set_data(cd, st.data(), st.size(), 0);
        }
        int pos = (int) inp.size();
        common_batch bb(cd);
        auto run = [&](int B) {
            bb.clear();
            for (int i = 0; i < B; i++) bb.add(inp[(pos + i) % inp.size()], pos + i, 0, true);
            double t = now_ms();
            llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, bb.get());
            llama_synchronize(cd);
            pos += B;
            return now_ms() - t;
        };
        run(1); run(4); // warm both graph shapes
        printf("batch | ms per call | ms per token | x single-token call\n");
        double base = 0;
        for (int B : a.batches) {
            double best = 1e30;
            for (int r = 0; r < 4; r++) best = std::min(best, run(B));
            if (B == 1) base = best;
            printf("%5d | %11.1f | %12.1f | %.2f\n", B, best, best / B, base > 0 ? best / base : 0.0);
            fflush(stdout);
        }
        return 0;
    }

    // ---- spec / plain decode ----
    llama_context * cm = nullptr;
    common_speculative * spec = nullptr;
    if (a.nmax > 0) {
        llama_context_params cpm = ctx_params(a.threads);
        cpm.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cpm.n_rs_seq = 0;
        cpm.ctx_other = cd;
        cpm.op_offload = false;
        cm = llama_init_from_model(md, cpm);
        if (!cm) { fprintf(stderr, "failed to create MTP context\n"); return 1; }
        common_params_speculative sp;
        sp.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        sp.draft.n_max = a.nmax;
        sp.draft.ctx_tgt = cp;   // prompt hidden states come from the prefill context
        sp.draft.ctx_dft = cm;
        sp.draft.backend_sampling = false;
        spec = common_speculative_init(sp, 1);
        if (!spec) { fprintf(stderr, "failed to init speculative decoding\n"); return 1; }
    }

    common_params_sampling sparams;
    sparams.temp = 0.0f;
    common_sampler_ptr smpl(common_sampler_init(md, sparams));

    // prompt phase: everything but the last token on the prefill context (NPU)
    double t_start = now_ms();
    common_batch batch_prompt(cp);
    for (size_t i = 0; i + 1 < inp.size(); i++) batch_prompt.add(inp[i], (llama_pos) i, 0, false);
    if (llama_process(cp, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0) { fprintf(stderr, "prompt decode failed\n"); return 1; }
    if (spec && !common_speculative_process(spec, batch_prompt)) { fprintf(stderr, "MTP prompt catch-up failed\n"); return 1; }
    double t_prefill = now_ms();
    if (cp != cd) {
        std::vector<uint8_t> st(llama_state_seq_get_size(cp, 0));
        llama_state_seq_get_data(cp, st.data(), st.size(), 0);
        if (llama_state_seq_set_data(cd, st.data(), st.size(), 0) == 0) { fprintf(stderr, "state handoff failed\n"); return 1; }
    }
    if (spec) common_speculative_set_ctx_tgt(spec, cd); // verification hidden states now come from the CPU context
    double t_handoff = now_ms();

    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    int n_past = (int) inp.size() - 1;
    int n_predict = 0, n_drafted = 0, n_accept = 0, n_steps = 0;
    double t_first = 0;
    std::string out;
    bool has_eos = false;

    if (spec) common_speculative_begin(spec, 0, prompt_tgt);
    const bool use_ckpt_tgt = common_context_can_seq_rm(cd) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = cm && common_context_can_seq_rm(cm) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    common_batch batch_tgt(cd);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    while (true) {
        if (spec && draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(), llama_memory_seq_pos_min(llama_get_memory(cd), 0), llama_memory_seq_pos_max(llama_get_memory(cd), 0));
            if (use_ckpt_dft) ckpt.update_dft(cm, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            int n_draft_max = std::max(0, std::min(a.n_ctx - n_past - 2, a.n_gen - n_predict - 1));
            common_speculative_get_draft_params(spec, 0) = { true, n_draft_max, n_past, id_last, &prompt_tgt, &draft };
            common_speculative_draft(spec);
            if (!draft.empty() && use_ckpt_tgt) ckpt.update_tgt(cd, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (use_ckpt_dft) ckpt.load_dft(cm, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(cm), 0, ckpt.pos_max + 1, -1);
        }

        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, 0, true);
        for (size_t i = 0; i < draft.size(); i++) batch_tgt.add(draft[i], n_past + (int) i, 0, true);
        if (llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0) { fprintf(stderr, "verify decode failed\n"); break; }
        n_steps++;
        if (spec && !common_speculative_process(spec, batch_tgt)) { fprintf(stderr, "MTP process failed\n"); break; }

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt && !draft.empty()) smpl_save.reset(common_sampler_clone(smpl.get()));
        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), cd, draft);
        if (t_first == 0) t_first = now_ms();

        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            // partial acceptance on a recurrent model: roll back to the checkpoint and redo the accepted part
            draft = std::move(ids);
            ckpt.load_tgt(cd, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(cd), 0, ckpt.pos_max + 1, -1);
            if (cm) { if (use_ckpt_dft) ckpt.load_dft(cm, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); llama_memory_seq_rm(llama_get_memory(cm), 0, ckpt.pos_max + 1, -1); }
            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }
        if (spec) common_speculative_accept(spec, 0, (uint16_t) (ids.size() - 1));
        n_past += (int) ids.size() - 1;
        n_drafted += (int) n_draft;
        n_accept += (int) ids.size() - 1;
        n_predict += (int) ids.size();
        for (size_t i = 0; i < ids.size(); i++) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(vocab, id_last)) { has_eos = true; break; }
            out += common_token_to_piece(cd, id_last);
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(cd), 0, n_past, -1);
        if (cm) llama_memory_seq_rm(llama_get_memory(cm), 0, n_past, -1);
        if (n_predict >= a.n_gen || has_eos) break;
    }
    double t_end = now_ms();

    printf("mode=%s prefill=%s nmax=%d alias=%d | prompt %zu tok | prefill %.0f ms (%.1f t/s) | handoff %.1f ms | TTFT %.0f ms | "
           "decode %d tok in %.0f ms = %.2f t/s | steps %d (%.2f tok/step) | drafted %d accepted %d (%.1f%%)\n",
           a.mode.c_str(), a.prefill_dev.c_str(), a.nmax, n_alias, inp.size(), t_prefill - t_start, (inp.size() - 1) / ((t_prefill - t_start) / 1000.0),
           t_handoff - t_prefill, t_first - t_start, n_predict, t_end - t_handoff, n_predict / ((t_end - t_handoff) / 1000.0),
           n_steps, n_steps ? (double) n_predict / n_steps : 0.0, n_drafted, n_accept, n_drafted ? 100.0 * n_accept / n_drafted : 0.0);
    if (a.check) printf("--- text ---\n%s\n--- end ---\n", out.c_str());
    fflush(stdout);
    if (spec) common_speculative_free(spec);
    return 0;
}
