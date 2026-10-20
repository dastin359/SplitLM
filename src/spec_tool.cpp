// trilane-spec: Tri-Lane prefill (NPU) + CPU decode with MTP speculative decoding.
// The MTP head drafts tokens on the CPU; the CPU decode context verifies k+1 tokens per pass over the weights.
// Modes:
//   spec (default): NPU prefill -> state handoff -> speculative (or plain, --nmax 0) decode on the CPU
//   batchcost:      time one decode batch of B tokens on the CPU decode context (verification cost)
// spec2 (two models in memory):
//   --draft-model F --spec-type draft-model   greedy drafting on a separate draft model (own loop, no re-evaluation),
//                                             draft context on --draft-dev CPU|HTP0|GPUOpenCL
//   --draft-model F --spec-type draft-simple|draft-dflash|draft-eagle3|draft-mtp   llama.cpp common drafters
//   --draft-head N[:T]   draft LM head over the first N token ids + last T ids (llama_set_draft_head); default for
//                        draft-mtp / draft-model with a vocabulary >= 128K: 65536:256 (--draft-head 0 = full head)
//   --spec-async         draft-model mode: keep drafting on the draft context while the target verifies
//   --adaptive           pick the draft length online (acceptance estimate x measured draft/verify times)

#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include "spec_draft.h"

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
    std::string model, prompt_file, mode = "spec", prefill_dev = "HTP0", alias = "auto", spec_type = "draft-mtp";
    int ngram_n = 3;
    int n_gen = 128, n_ctx = 2048, threads = 8, nmax = 3, ubatch = 512;
    double gb_per_tok = 0; // bytes read per plain decode token (GB), for GB/s-equivalent reporting
    std::vector<int> batches = { 1, 2, 3, 4, 5, 8 };
    bool check = false;
    bool share = true;              // decode model borrows the prefill model's CPU-resident weights (--no-share: own copies)
    std::string p_cpu = "none";     // --prefill-cpu none|output|nonq4: non-Q4_0 weights the prefill model keeps on CPU for sharing
    bool prefill_on_decode = false; // run the prompt through the CPU decode context (exercises batched CPU kernels)
    // spec2: second model in memory
    std::string draft_model;        // --draft-model: separate draft GGUF (small model, mtp-*.gguf, dflash-*.gguf)
    std::string draft_dev = "CPU";  // --draft-dev: device of the draft context
    int draft_threads = 0;          // --draft-threads: CPU threads of the draft context (0 = -t)
    int draft_head_n = -1, draft_head_tail = 256; // --draft-head N[:T]; -1 = auto (see apply_draft_head)
    bool spec_async = false;        // --spec-async
    bool adaptive = false;          // --adaptive
    bool draft_prefill_overlap = true; // --no-draft-overlap: draft prompt after (not during) the NPU prefill
};

static void usage() {
    fprintf(stderr, "usage: trilane-spec -m model.gguf -f prompt.txt [--mode spec|batchcost] [--prefill-dev HTP0|CPU] [--nmax K (0 = no spec)]\n"
                    "       [-n n_gen] [-c n_ctx] [-t threads] [-ub ubatch] [--alias auto|on|off] [--no-share] [--batches 1,2,4,8] [--check] [-v]\n"
                    "       [--spec-type draft-mtp|draft-model|draft-simple|draft-dflash|draft-eagle3|ngram-simple|...] [--draft-model F.gguf]\n"
                    "       [--draft-dev CPU|HTP0|GPUOpenCL] [--draft-threads N] [--draft-head N[:T]] [--spec-async] [--adaptive] [--no-draft-overlap]\n");
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

typedef const ggml_tensor * (*shared_resolver_t)(const char * name, const ggml_tensor * t, void * user);
typedef void (*shared_set_resolver_t)(shared_resolver_t fn, void * user);

static const ggml_tensor * resolve_tensor_from_model(const char * name, const ggml_tensor * t, void * user) {
    const ggml_tensor * src = llama_model_get_tensor((const llama_model *) user, name);
    if (!src || src->type != t->type || ggml_nbytes(src) != ggml_nbytes(t)) return nullptr;
    return src;
}

// weights the prefill model keeps in CPU memory (types the NPU cannot run, token embeddings): the decode model borrows them
static std::vector<std::string> shared_candidates(const std::string & path, const llama_model * mp, size_t * bytes) {
    std::vector<std::string> out;
    *bytes = 0;
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return out;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const std::string name = gguf_get_tensor_name(g, i);
        const ggml_tensor * t = llama_model_get_tensor(mp, name.c_str());
        if (!t || !t->buffer || ggml_nbytes(t) < (1u << 20)) continue;
        const std::string buf = ggml_backend_buffer_name(t->buffer);
        if (buf != "CPU" && buf != "CPU_REPACK") continue;
        out.push_back(name);
        *bytes += ggml_nbytes(t);
    }
    gguf_free(g);
    return out;
}

// weights the NPU would tile in formats the CPU kernels cannot read (Q6_K, Q5_K, Q4_1, ...): keeping them in the
// prefill model's CPU memory instead lets the decode model share them. "output" moves only the LM head, which costs
// prefill nothing (no prompt logits); "nonq4" also moves per-layer matmuls, which then run on the CPU during prefill.
static std::vector<std::string> prefill_cpu_names(const std::string & path, const std::string & mode) {
    std::vector<std::string> out;
    if (mode != "output" && mode != "nonq4") return out;
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return out;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const std::string name = gguf_get_tensor_name(g, i);
        const ggml_type type = gguf_get_tensor_type(g, i);
        if (name.rfind("token_embd", 0) == 0 || type == GGML_TYPE_Q4_0 || gguf_get_tensor_size(g, i) < (1u << 20)) continue;
        if (mode == "output" && name != "output.weight") continue;
        out.push_back(name);
    }
    gguf_free(g);
    return out;
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

// ---- spec2: draft-model speculation with our own loop (draft context on any device, optional draft-ahead) ----
// Pruned draft head (measured, identical text): 27B MTP k3 10.7 -> 12.2 t/s, 2B MTP k3 66.9 -> 83.0 t/s. Auto: large
// vocabularies (Qwen 248K) on the MTP / draft-model drafters; other drafters and small vocabularies keep the full head.
static void apply_draft_head(const args & a, llama_context * c) {
    int n = a.draft_head_n;
    if (n < 0) {
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c)));
        n = (nv >= 131072 && (a.spec_type == "draft-mtp" || a.spec_type == "draft-model")) ? 65536 : 0;
    }
    if (n > 0) llama_set_draft_head(c, n, a.draft_head_tail);
    fprintf(stderr, "[spec2] draft head: %s\n", n > 0 ? (std::to_string(n) + ":" + std::to_string(a.draft_head_tail)).c_str() : "full");
}

// Invariants: the target context cd holds committed[0..n_tgt-1]; committed.back() is id_last (not yet in cd).
// The draft context holds dr.dtoks (reconciled with committed before each fresh draft, see spec_draft.h).
static int run_draft_model(const args & a, llama_context * cp, llama_context * cd, llama_context * cdr,
                           const llama_model * md, const std::vector<llama_token> & inp, int n_alias) {
    const llama_vocab * vocab = llama_model_get_vocab(md);
    apply_draft_head(a, cdr);

    // rollback support of the target: bounded snapshots (n_rs_seq) or full checkpoints; probing clears the memory
    const bool use_ckpt_tgt = common_context_can_seq_rm(cd) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    if (cp != cd) llama_memory_clear(llama_get_memory(cp), true);
    llama_memory_clear(llama_get_memory(cdr), true);

    common_params_sampling sparams;
    sparams.temp = 0.0f;
    common_sampler_ptr smpl(common_sampler_init(md, sparams));

    draft_runner dr;
    dr.init(cdr);
    draft_ahead da;
    k_tuner tuner;
    tuner.init(a.nmax);

    // prompt: target on the prefill context (NPU), draft prompt on the draft context, overlapped when the draft
    // runs on another device than the prefill (e.g. NPU prefill + CPU draft)
    const std::vector<llama_token> prompt_wo_last(inp.begin(), inp.end() - 1);
    double t_start = now_ms(), t_draft_prompt = 0;
    const bool overlap = a.draft_prefill_overlap && a.draft_dev != a.prefill_dev;
    std::thread th_dp;
    auto draft_prompt = [&]() { const double t0 = now_ms(); dr.feed(prompt_wo_last, false); t_draft_prompt = now_ms() - t0; };
    if (overlap) th_dp = std::thread(draft_prompt);
    common_batch batch_prompt(cp);
    for (size_t i = 0; i < prompt_wo_last.size(); i++) batch_prompt.add(prompt_wo_last[i], (llama_pos) i, 0, false);
    if (llama_process(cp, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get()) != 0) { fprintf(stderr, "prompt decode failed\n"); return 1; }
    llama_synchronize(cp);
    double t_prefill = now_ms();
    if (cp != cd) {
        std::vector<uint8_t> st(llama_state_seq_get_size(cp, 0));
        llama_state_seq_get_data(cp, st.data(), st.size(), 0);
        if (llama_state_seq_set_data(cd, st.data(), st.size(), 0) == 0) { fprintf(stderr, "state handoff failed\n"); return 1; }
    }
    if (overlap) th_dp.join(); else draft_prompt();
    double t_handoff = now_ms();

    std::vector<llama_token> committed(inp.begin(), inp.end());
    int n_tgt = (int) inp.size() - 1;
    int n_predict = 0, n_drafted = 0, n_accept = 0, n_steps = 0, n_hit_steps = 0;
    std::vector<int> acc_by_pos(a.nmax + 2, 0), drafted_by_pos(a.nmax + 2, 0), k_hist(a.nmax + 2, 0);
    double t_draft = 0, t_verify = 0, t_first = 0, t_wait_ahead = 0;
    double t_verify_by_n[16] = { 0 };
    int    n_verify_by_n[16] = { 0 };
    std::string out;
    bool has_eos = false;
    common_batch batch_tgt(cd);
    std::vector<llama_token> draft, ahead;
    common_prompt_checkpoint ckpt;
    bool forced = false; // re-verification of an accepted prefix after a full-checkpoint rollback (no drafting)

    while (true) {
        const int room = std::max(0, std::min(a.n_ctx - n_tgt - 2, a.n_gen - n_predict - 1));
        int k = std::min(a.adaptive ? tuner.pick() : a.nmax, room);
        double t_step_draft = 0;
        bool can_launch = true;
        if (!forced) {
            const double td0 = now_ms();
            if (!ahead.empty()) {
                draft.swap(ahead);   // full draft-ahead hit: the next draft is already on the draft context
                ahead.clear();
                if ((int) draft.size() > room) {
                    draft.resize(room); // the draft context holds extra tokens; the next reconcile trims them
                    can_launch = false;
                }
            } else if (k > 0) {
                dr.draft(committed, k, draft);
            } else {
                draft.clear();
            }
            t_step_draft = now_ms() - td0;
            t_draft += t_step_draft;
            if (a.spec_async && can_launch && !draft.empty() && (int) draft.size() < room) da.launch(&dr, draft.back(), std::max(1, k));
            if (use_ckpt_tgt && !draft.empty()) {
                ckpt.update_pos((int64_t) committed.size() - 1, llama_memory_seq_pos_min(llama_get_memory(cd), 0), llama_memory_seq_pos_max(llama_get_memory(cd), 0));
                ckpt.update_tgt(cd, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }
        const bool was_forced = forced;   // re-verification: not a draft, keep it out of the statistics
        forced = false;
        const int n_draft = (int) draft.size();
        if (!was_forced) k_hist[std::min<int>(n_draft, (int) k_hist.size() - 1)]++;

        batch_tgt.clear();
        batch_tgt.add(committed.back(), n_tgt, 0, true);
        for (int i = 0; i < n_draft; i++) batch_tgt.add(draft[i], n_tgt + 1 + i, 0, true);
        const double tv0 = now_ms();
        if (llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0) { fprintf(stderr, "verify decode failed\n"); break; }
        llama_synchronize(cd);
        const double tv1 = now_ms();
        t_verify += tv1 - tv0;
        t_verify_by_n[std::min<int>(batch_tgt.size(), 15)] += tv1 - tv0;
        n_verify_by_n[std::min<int>(batch_tgt.size(), 15)]++;
        n_steps++;

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt && n_draft > 0) smpl_save.reset(common_sampler_clone(smpl.get()));
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), cd, draft);
        if (t_first == 0) t_first = now_ms();
        const int j = (int) ids.size() - 1;   // accepted draft tokens; ids.back() is the target's own next token

        // the draft-ahead thread must leave the draft context before anyone else touches it
        const double tw0 = now_ms();
        da.finish();
        t_wait_ahead += now_ms() - tw0;
        const bool hit = a.spec_async && n_draft > 0 && j == n_draft && da.guess.size() >= 2 && da.guess[0] == ids.back();

        if (use_ckpt_tgt && j < n_draft) {
            // target without partial rollback: restore the checkpoint and re-verify the accepted part plus the bonus
            ckpt.load_tgt(cd, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(cd), 0, n_tgt, -1);
            smpl = std::move(smpl_save);
            n_drafted += n_draft;
            n_accept += j;
            for (int i = 0; i < n_draft; i++) { drafted_by_pos[std::min(i, a.nmax)]++; if (i < j) acc_by_pos[std::min(i, a.nmax)]++; }
            draft = std::move(ids);
            forced = true;
            continue;
        }

        if (!was_forced) {
            n_drafted += n_draft;
            n_accept += j;
            for (int i = 0; i < n_draft; i++) { drafted_by_pos[std::min(i, a.nmax)]++; if (i < j) acc_by_pos[std::min(i, a.nmax)]++; }
            if (n_draft > 0) tuner.observe(n_draft, j, t_step_draft, tv1 - tv0);
        }
        n_predict += (int) ids.size();
        for (size_t i = 0; i < ids.size(); i++) {
            committed.push_back(ids[i]);
            if (llama_vocab_is_eog(vocab, ids[i])) { has_eos = true; break; }
            out += common_token_to_piece(cd, ids[i]);
        }
        n_tgt += j + 1;   // id_last + accepted drafts are now in the target context
        llama_memory_seq_rm(llama_get_memory(cd), 0, n_tgt, -1);   // rejected tail (bounded snapshot rollback)
        if (hit) {
            dr.promote_ahead();
            ahead.assign(da.guess.begin() + 1, da.guess.end());
            n_hit_steps++;
            da.n_hit++;
            da.n_tokens_used += (int) ahead.size();
        }
        draft.clear();
        if (n_predict >= a.n_gen || has_eos) break;
    }
    da.finish();
    double t_end = now_ms();

    const double tps = n_predict / ((t_end - t_handoff) / 1000.0);
    printf("mode=spec/draft-model draft=%s@%s prefill=%s nmax=%d%s%s alias=%d | prompt %zu tok | prefill %.0f ms (%.1f t/s) | handoff+draft prompt %.1f ms (draft prompt %.0f ms%s) | TTFT %.0f ms | "
           "decode %d tok in %.0f ms = %.2f t/s | steps %d (%.2f tok/step) | drafted %d accepted %d (%.1f%%)\n",
           a.draft_model.substr(a.draft_model.find_last_of("/\\") + 1).c_str(), a.draft_dev.c_str(), a.prefill_dev.c_str(), a.nmax,
           a.adaptive ? " adaptive" : "", a.spec_async ? " async" : "", n_alias, inp.size(), t_prefill - t_start,
           (inp.size() - 1) / ((t_prefill - t_start) / 1000.0), t_handoff - t_prefill, t_draft_prompt, overlap ? ", overlapped" : "",
           t_first - t_start, n_predict, t_end - t_handoff, tps, n_steps, n_steps ? (double) n_predict / n_steps : 0.0,
           n_drafted, n_accept, n_drafted ? 100.0 * n_accept / n_drafted : 0.0);
    printf("time split: draft %.0f ms (%d passes, %d tokens fed, %d checkpoint restores) | verify %.0f ms | wait for draft-ahead %.0f ms | other %.0f ms",
           t_draft, dr.n_passes, dr.n_fed, dr.n_restores, t_verify, t_wait_ahead, (t_end - t_handoff) - t_draft - t_verify - t_wait_ahead);
    if (a.gb_per_tok > 0) printf(" | %.1f GB/s-equivalent (%.3f GB/token)", tps * a.gb_per_tok, a.gb_per_tok);
    printf("\nverify ms by batch size:");
    for (int i = 1; i < 16; i++) if (n_verify_by_n[i]) printf("  %d:%.1f(x%d)", i, t_verify_by_n[i] / n_verify_by_n[i], n_verify_by_n[i]);
    printf("\nacceptance by draft position:");
    for (int i = 0; i <= a.nmax; i++) if (drafted_by_pos[i]) printf("  %d:%.0f%%(%d)", i + 1, 100.0 * acc_by_pos[i] / drafted_by_pos[i], drafted_by_pos[i]);
    printf("\ndraft length histogram:");
    for (int i = 0; i < (int) k_hist.size(); i++) if (k_hist[i]) printf("  %d:%d", i, k_hist[i]);
    if (a.adaptive) printf(" | alpha estimate %.3f", tuner.alpha());
    if (a.spec_async) printf("\ndraft-ahead: %d launches, %d full hits (%.1f%%, %d steps drafted for free), %d ahead tokens used", da.n_launch, da.n_hit,
                             da.n_launch ? 100.0 * da.n_hit / da.n_launch : 0.0, n_hit_steps, da.n_tokens_used);
    printf("\n");
    if (a.check) printf("--- text ---\n%s\n--- end ---\n", out.c_str());
    fflush(stdout);
    return 0;
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
        else if (k == "--no-share") a.share = false;
        else if (k == "--prefill-cpu") a.p_cpu = next();
        else if (k == "--nmax") a.nmax = atoi(next());
        else if (k == "--spec-type") a.spec_type = next();
        else if (k == "--ngram-n") a.ngram_n = atoi(next());
        else if (k == "-n") a.n_gen = atoi(next());
        else if (k == "-c") a.n_ctx = atoi(next());
        else if (k == "-t") a.threads = atoi(next());
        else if (k == "-ub") a.ubatch = atoi(next());
        else if (k == "--batches") { a.batches.clear(); std::stringstream ss(next()); for (std::string s; std::getline(ss, s, ',');) a.batches.push_back(atoi(s.c_str())); }
        else if (k == "--check") a.check = true;
        else if (k == "--gb-per-token") a.gb_per_tok = atof(next());
        else if (k == "--prefill-on-decode") a.prefill_on_decode = true;
        else if (k == "-v") g_verbose = true;
        else if (k == "--draft-model") a.draft_model = next();
        else if (k == "--draft-dev") a.draft_dev = next();
        else if (k == "--draft-threads") a.draft_threads = atoi(next());
        else if (k == "--draft-head") { std::string v = next(); a.draft_head_n = atoi(v.c_str()); size_t c = v.find(':'); if (c != std::string::npos) a.draft_head_tail = atoi(v.c_str() + c + 1); }
        else if (k == "--spec-async") a.spec_async = true;
        else if (k == "--adaptive") a.adaptive = true;
        else if (k == "--no-draft-overlap") a.draft_prefill_overlap = false;
        else usage();
    }
    if (a.model.empty() || a.prompt_file.empty()) usage();
    if (a.draft_threads <= 0) a.draft_threads = a.threads;
    if (a.spec_type == "draft-model" && a.draft_model.empty()) { fprintf(stderr, "--spec-type draft-model needs --draft-model\n"); return 1; }
    if (a.spec_async && a.spec_type != "draft-model") { fprintf(stderr, "--spec-async is implemented for --spec-type draft-model only\n"); return 1; }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    ggml_backend_load_all();

    // ---- models: P (prefill, NPU) and D (decode, CPU, MTP layers loaded, weights aliased to P where needed) ----
    const bool use_npu = a.prefill_dev != "CPU";
    llama_model * mp = nullptr;
    std::vector<ggml_backend_dev_t> npu_devs;
    std::vector<std::string> p_cpu_pats;
    std::vector<llama_model_tensor_buft_override> p_cpu_ovr;
    if (use_npu) {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(a.prefill_dev.c_str());
        if (!d) { fprintf(stderr, "device %s not found\n", a.prefill_dev.c_str()); return 1; }
        npu_devs = { d, nullptr };
        llama_model_params pp = llama_model_default_params();
        pp.load_mode = LLAMA_LOAD_MODE_NONE;
        pp.devices = npu_devs.data();
        pp.n_gpu_layers = -1;
        if (a.share && a.alias != "off") {
            p_cpu_pats = exact_name_patterns(prefill_cpu_names(a.model, a.p_cpu));
            for (auto & p : p_cpu_pats) p_cpu_ovr.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
            if (!p_cpu_ovr.empty()) {
                p_cpu_ovr.push_back({ nullptr, nullptr });
                pp.tensor_buft_overrides = p_cpu_ovr.data();
            }
        }
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
    dp.load_mtp = a.nmax > 0 && a.mode == "spec" && a.spec_type == "draft-mtp" && a.draft_model.empty();
    std::vector<std::string> pats;
    std::vector<llama_model_tensor_buft_override> ovr;
    int n_alias = 0, n_shared = 0;
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
            const size_t n_tile_pats = pats.size();
            n_alias = (int) names.size();

            auto set_shared = (shared_set_resolver_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_shared_set_resolver");
            auto get_shared = (htptile_buft_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_shared_buffer_type");
            if (a.share && set_shared && get_shared) {
                size_t bytes = 0;
                const auto shared = shared_candidates(a.model, mp, &bytes);
                if (!shared.empty()) {
                    set_shared(resolve_tensor_from_model, (void *) mp);
                    for (auto & p : exact_name_patterns(shared)) pats.push_back(p);
                    n_shared = (int) shared.size();
                    fprintf(stderr, "[spec] %d CPU-resident prefill-model tensors shared with the decode model (%.2f GiB)\n", n_shared, bytes / 1073741824.0);
                }
            }
            for (size_t i = 0; i < pats.size(); i++) ovr.push_back({ pats[i].c_str(), i < n_tile_pats ? get_buft() : get_shared() });
            ovr.push_back({ nullptr, nullptr });
            dp.tensor_buft_overrides = ovr.data();
            dp.load_mode = LLAMA_LOAD_MODE_MMAP;
            _putenv_s("LLAMA_MMAP_NO_PREFETCH", "1");
        }
    }
    double t0 = now_ms();
    llama_model * md = llama_model_load_from_file(a.model.c_str(), dp);
    if (!md) return 1;
    fprintf(stderr, "[spec] decode model on CPU loaded in %.0f ms (%d tensors aliased to NPU memory, mtp=%d)\n", now_ms() - t0, n_alias, (int) dp.load_mtp);
    if (!mp) mp = md;

    // ---- spec2: a second model in memory (draft), on its own device ----
    llama_model * mdr = nullptr;
    std::vector<ggml_backend_dev_t> draft_devs = { nullptr };
    if (!a.draft_model.empty() && a.mode == "spec" && a.nmax > 0) {
        llama_model_params mp_d = llama_model_default_params();
        mp_d.load_mode = LLAMA_LOAD_MODE_NONE;  // no mmap: CPU weights get repacked, one copy in memory
        if (a.draft_dev != "CPU") {
            ggml_backend_dev_t d = ggml_backend_dev_by_name(a.draft_dev.c_str());
            if (!d) { fprintf(stderr, "draft device %s not found\n", a.draft_dev.c_str()); return 1; }
            draft_devs = { d, nullptr };
            mp_d.n_gpu_layers = -1;
        } else {
            mp_d.n_gpu_layers = 0;
        }
        mp_d.devices = draft_devs.data();
        mp_d.load_mtp = a.spec_type == "draft-mtp"; // a separate mtp-*.gguf holds only the MTP block (+ embeddings, head)
        const double td0 = now_ms();
        mdr = llama_model_load_from_file(a.draft_model.c_str(), mp_d);
        if (!mdr) { fprintf(stderr, "failed to load draft model %s\n", a.draft_model.c_str()); return 1; }
        fprintf(stderr, "[spec2] draft model on %s loaded in %.0f ms (%.2f GiB)\n", a.draft_dev.c_str(), now_ms() - td0,
                llama_model_size(mdr) / 1073741824.0);
        if (a.spec_type != "draft-mtp" && a.spec_type != "draft-dflash" && a.spec_type != "draft-eagle3" &&
            !common_speculative_are_compatible(md, mdr)) {
            fprintf(stderr, "draft and target vocabularies are not compatible\n");
            return 1;
        }
    }

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
    cpd.n_outputs_max_per_seq = 0;                       // verify batches need k+1 outputs per sequence
    cpd.n_rs_seq = a.mode == "spec" ? (uint32_t) std::max(a.nmax, 0) : 0; // bounded recurrent rollback, as common does
    if (use_npu && !a.prefill_on_decode) {
        cpd.n_batch = cpd.n_ubatch = 64; // decode context only sees verify batches: a small compute buffer (~0.5 GB less on 27B)
    }
    llama_context * cd = llama_init_from_model(md, cpd);
    llama_context * cp = mp != md ? llama_init_from_model(mp, cpp) : cd;
    if (!cd || !cp) return 1;
    if (a.prefill_on_decode) cp = cd;

    llama_context * cdr = nullptr;
    if (mdr) {
        llama_context_params cpdr = ctx_params(a.draft_threads);
        cpdr.n_ctx = a.n_ctx + 64;      // the draft context also holds the drafted (and draft-ahead) tokens
        cpdr.n_batch = cpdr.n_ubatch = 512;
        cpdr.op_offload = a.draft_dev != "CPU";
        cpdr.n_rs_seq = 0;              // recurrent drafts roll back by checkpoint (single-token draft steps have no snapshots)
        if (a.spec_type == "draft-mtp") {
            cpdr.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cpdr.ctx_other = cd;
        } else if (a.spec_type == "draft-dflash" || a.spec_type == "draft-eagle3") {
            cpdr.ctx_other = cd;
        }
        cdr = llama_init_from_model(mdr, cpdr);
        if (!cdr) { fprintf(stderr, "failed to create the draft context\n"); return 1; }
    }

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

    if (a.spec_type == "draft-model" && a.nmax > 0) {
        return run_draft_model(a, cp, cd, cdr, md, inp, n_alias);
    }

    // ---- spec / plain decode ----
    llama_context * cm = nullptr;
    common_speculative * spec = nullptr;
    if (a.nmax > 0) {
        common_params_speculative sp;
        sp.types = common_speculative_types_from_names({ a.spec_type });
        sp.draft.n_max = a.nmax;
        sp.draft.ctx_tgt = cp;   // prompt hidden states come from the prefill context
        sp.draft.backend_sampling = false;
        sp.ngram_simple.size_n = (uint16_t) a.ngram_n;
        sp.ngram_simple.size_m = (uint16_t) a.nmax;
        if (cdr) {
            cm = cdr;           // separate draft model (draft-simple, draft-dflash, draft-eagle3, mtp-*.gguf)
            sp.draft.ctx_dft = cdr;
            const llama_model * mdm = llama_get_model(cdr);
            if (a.spec_type == "draft-simple" && (llama_model_is_recurrent(mdm) || llama_model_is_hybrid(mdm))) {
                fprintf(stderr, "[spec2] warning: draft-simple cannot roll a recurrent draft back after a rejection in this loop "
                                "(output stays exact, acceptance drops); use --spec-type draft-model\n");
            }
        } else if (a.spec_type == "draft-mtp") {
            llama_context_params cpm = ctx_params(a.threads);
            cpm.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cpm.n_rs_seq = 0;
            cpm.ctx_other = cd;
            cpm.op_offload = false;
            cm = llama_init_from_model(md, cpm);
            if (!cm) { fprintf(stderr, "failed to create MTP context\n"); return 1; }
            sp.draft.ctx_dft = cm;
        }
        if (cm) apply_draft_head(a, cm);
        spec = common_speculative_init(sp, 1);
        if (!spec) { fprintf(stderr, "failed to init speculative decoding\n"); return 1; }
    }
    k_tuner tuner;
    tuner.init(std::max(1, a.nmax));

    // probe rollback support before the prompt: common_context_can_seq_rm clears the context memory to test it
    const bool use_ckpt_tgt = common_context_can_seq_rm(cd) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = cm && common_context_can_seq_rm(cm) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    if (cp != cd) llama_memory_clear(llama_get_memory(cp), true);

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
    double t_draft = 0, t_verify = 0, t_mtp_proc = 0;
    double t_verify_by_n[16] = { 0 };
    int    n_verify_by_n[16] = { 0 };
    double t_first = 0;
    std::string out;
    bool has_eos = false;

    if (spec) common_speculative_begin(spec, 0, prompt_tgt);
    common_batch batch_tgt(cd);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    double t_step_draft = 0;

    while (true) {
        if (spec && draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(), llama_memory_seq_pos_min(llama_get_memory(cd), 0), llama_memory_seq_pos_max(llama_get_memory(cd), 0));
            if (use_ckpt_dft) ckpt.update_dft(cm, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            int n_draft_max = std::max(0, std::min(a.n_ctx - n_past - 2, a.n_gen - n_predict - 1));
            if (a.adaptive) n_draft_max = std::min(n_draft_max, tuner.pick());
            common_speculative_get_draft_params(spec, 0) = { true, n_draft_max, n_past, id_last, &prompt_tgt, &draft };
            const double td0 = now_ms();
            common_speculative_draft(spec);
            t_step_draft = now_ms() - td0;
            t_draft += t_step_draft;
            if (!draft.empty() && use_ckpt_tgt) ckpt.update_tgt(cd, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (cm) {
                if (use_ckpt_dft) ckpt.load_dft(cm, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(cm), 0, ckpt.pos_max + 1, -1);
            }
        }

        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, 0, true);
        for (size_t i = 0; i < draft.size(); i++) batch_tgt.add(draft[i], n_past + (int) i, 0, true);
        const double tv0 = now_ms();
        if (llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0) { fprintf(stderr, "verify decode failed\n"); break; }
        llama_synchronize(cd);
        const double tv1 = now_ms();
        t_verify += tv1 - tv0;
        t_verify_by_n[std::min<int>(batch_tgt.size(), 15)] += tv1 - tv0;
        n_verify_by_n[std::min<int>(batch_tgt.size(), 15)]++;
        n_steps++;
        if (spec && !common_speculative_process(spec, batch_tgt)) { fprintf(stderr, "MTP process failed\n"); break; }
        t_mtp_proc += now_ms() - tv1;

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
        if (spec && n_draft > 0) tuner.observe((int) n_draft, (int) ids.size() - 1, t_step_draft, tv1 - tv0);
        t_step_draft = 0;
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

    printf("mode=%s/%s prefill=%s nmax=%d alias=%d | prompt %zu tok | prefill %.0f ms (%.1f t/s) | handoff %.1f ms | TTFT %.0f ms | "
           "decode %d tok in %.0f ms = %.2f t/s | steps %d (%.2f tok/step) | drafted %d accepted %d (%.1f%%)\n",
           a.mode.c_str(), a.nmax > 0 ? a.spec_type.c_str() : "plain", a.prefill_dev.c_str(), a.nmax, n_alias, inp.size(), t_prefill - t_start, (inp.size() - 1) / ((t_prefill - t_start) / 1000.0),
           t_handoff - t_prefill, t_first - t_start, n_predict, t_end - t_handoff, n_predict / ((t_end - t_handoff) / 1000.0),
           n_steps, n_steps ? (double) n_predict / n_steps : 0.0, n_drafted, n_accept, n_drafted ? 100.0 * n_accept / n_drafted : 0.0);
    const double tps = n_predict / ((t_end - t_handoff) / 1000.0);
    printf("time split: draft %.0f ms | verify %.0f ms | mtp catch-up %.0f ms | other %.0f ms", t_draft, t_verify, t_mtp_proc,
           (t_end - t_handoff) - t_draft - t_verify - t_mtp_proc);
    if (a.gb_per_tok > 0) printf(" | %.1f GB/s-equivalent (%.3f GB/token)", tps * a.gb_per_tok, a.gb_per_tok);
    printf("\nverify ms by batch size:");
    for (int i = 1; i < 16; i++) if (n_verify_by_n[i]) printf("  %d:%.1f(x%d)", i, t_verify_by_n[i] / n_verify_by_n[i], n_verify_by_n[i]);
    printf("\n");
    if (a.check) printf("--- text ---\n%s\n--- end ---\n", out.c_str());
    fflush(stdout);
    if (spec) common_speculative_free(spec);
    return 0;
}
