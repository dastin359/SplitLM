// trilane-split: phase-split inference benchmark.
// Prefill on one device (e.g. HTP0), hand the sequence state over, decode on another (e.g. CPU).
// With --prefill-dev == --decode-dev it runs the single-device baseline through the same code path.

#include "llama.h"
#include "ggml-backend.h"

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

struct args {
    std::string model;
    std::string prefill_dev = "HTP0";
    std::string decode_dev  = "CPU";
    std::string prompt_file;
    int n_prompt   = 512;
    int n_gen      = 128;
    int n_ctx      = 4096;
    int n_batch    = 2048;
    int n_ubatch   = 512;
    int threads    = 4;   // decode threads
    int threads_b  = 4;   // prefill threads (host side)
    int reps       = 3;
    bool check     = false; // print generated text
};

static void usage() {
    fprintf(stderr,
        "usage: trilane-split -m model.gguf [--prefill-dev HTP0|GPUOpenCL|CPU] [--decode-dev CPU|HTP0|GPUOpenCL]\n"
        "       [-p n_prompt | -f prompt.txt] [-n n_gen] [-c n_ctx] [-b n_batch] [-ub n_ubatch]\n"
        "       [-t decode_threads] [-tb prefill_threads] [-r reps] [--check]\n");
    exit(1);
}

static args parse(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) usage(); return argv[++i]; };
        if      (k == "-m")            a.model = next();
        else if (k == "--prefill-dev") a.prefill_dev = next();
        else if (k == "--decode-dev")  a.decode_dev = next();
        else if (k == "-f")            a.prompt_file = next();
        else if (k == "-p")            a.n_prompt = atoi(next());
        else if (k == "-n")            a.n_gen = atoi(next());
        else if (k == "-c")            a.n_ctx = atoi(next());
        else if (k == "-b")            a.n_batch = atoi(next());
        else if (k == "-ub")           a.n_ubatch = atoi(next());
        else if (k == "-t")            a.threads = atoi(next());
        else if (k == "-tb")           a.threads_b = atoi(next());
        else if (k == "-r")            a.reps = atoi(next());
        else if (k == "--check")       a.check = true;
        else if (k == "-v")            g_verbose = true;
        else usage();
    }
    if (a.model.empty()) usage();
    return a;
}

static double now_ms() { return llama_time_us() / 1000.0; }

static llama_model * load_on(const std::string & path, const std::string & dev_name) {
    llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_NONE; // read once; mmap + repack keeps two copies of the weights
    static std::vector<std::vector<ggml_backend_dev_t>> keep; // devices list must outlive the load call
    keep.emplace_back();
    auto & devs = keep.back();
    if (dev_name == "CPU") {
        mp.n_gpu_layers = 0;
    } else {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(dev_name.c_str());
        if (!d) { fprintf(stderr, "device %s not found\n", dev_name.c_str()); exit(1); }
        devs.push_back(d);
        mp.n_gpu_layers = -1;
    }
    devs.push_back(nullptr);
    mp.devices = devs.data();
    double t0 = now_ms();
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    if (!m) { fprintf(stderr, "failed to load model on %s\n", dev_name.c_str()); exit(1); }
    fprintf(stderr, "[trilane] loaded model on %-9s in %.0f ms\n", dev_name.c_str(), now_ms() - t0);
    return m;
}

static llama_context * make_ctx(llama_model * m, const args & a, int threads, bool host_only) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = a.n_ctx;
    cp.n_batch         = a.n_batch;
    cp.n_ubatch        = a.n_ubatch;
    cp.n_seq_max       = 1;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    cp.no_perf         = true;
    cp.op_offload      = !host_only; // keep CPU decode context from bouncing ops to accelerators
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) { fprintf(stderr, "failed to create context\n"); exit(1); }
    return c;
}

static std::vector<llama_token> make_prompt(const llama_vocab * vocab, const args & a) {
    std::string text;
    if (!a.prompt_file.empty()) {
        std::ifstream f(a.prompt_file);
        std::stringstream ss; ss << f.rdbuf(); text = ss.str();
    } else {
        // deterministic natural-ish text so the check output is readable
        const char * para =
            "The history of computing is a story of moving work to where it runs best. Early machines did everything on one "
            "processor, then floating point moved to coprocessors, graphics moved to GPUs, and now neural networks move to NPUs. ";
        while ((int) text.size() < a.n_prompt * 8) text += para;
    }
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); exit(1); }
    toks.resize(n);
    if (a.prompt_file.empty() && (int) toks.size() > a.n_prompt) toks.resize(a.n_prompt);
    return toks;
}

// prefill in n_batch chunks, logits only for the last token
static void prefill(llama_context * c, const std::vector<llama_token> & toks, int n_batch) {
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    for (int i0 = 0; i0 < (int) toks.size(); i0 += n_batch) {
        int n = std::min(n_batch, (int) toks.size() - i0);
        b.n_tokens = n;
        for (int j = 0; j < n; j++) {
            b.token[j]     = toks[i0 + j];
            b.pos[j]       = i0 + j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = 0;
            b.logits[j]    = (i0 + j == (int) toks.size() - 1);
        }
        if (llama_decode(c, b) != 0) { fprintf(stderr, "prefill decode failed\n"); exit(1); }
    }
    llama_synchronize(c);
    llama_batch_free(b);
}

static llama_token greedy(llama_context * c, const llama_vocab * vocab) {
    const float * lg = llama_get_logits_ith(c, -1);
    int nv = llama_vocab_n_tokens(vocab);
    return (llama_token) (std::max_element(lg, lg + nv) - lg);
}

static std::string piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : std::string();
}

struct result {
    double prefill_ms = 0, xfer_ms = 0, decode_ms = 0;
    size_t xfer_bytes = 0;
    int n_prompt = 0, n_gen = 0;
    std::string text;
};

static result run_once(llama_context * cp, llama_context * cd, const llama_vocab * vocab,
                       const std::vector<llama_token> & toks, const args & a) {
    result r;
    r.n_prompt = (int) toks.size();
    llama_memory_clear(llama_get_memory(cp), true);
    if (cd != cp) llama_memory_clear(llama_get_memory(cd), true);

    double t0 = now_ms();
    prefill(cp, toks, a.n_batch);
    llama_token tok = greedy(cp, vocab);
    double t1 = now_ms();
    r.prefill_ms = t1 - t0;

    if (cd != cp) {
        size_t sz = llama_state_seq_get_size(cp, 0);
        std::vector<uint8_t> buf(sz);
        llama_state_seq_get_data(cp, buf.data(), sz, 0);
        if (llama_state_seq_set_data(cd, buf.data(), sz, 0) == 0) { fprintf(stderr, "state handoff failed\n"); exit(1); }
        r.xfer_bytes = sz;
    }
    double t2 = now_ms();
    r.xfer_ms = t2 - t1;

    llama_batch b = llama_batch_init(1, 0, 1);
    int pos = (int) toks.size();
    if (a.check) r.text += piece(vocab, tok);
    for (int i = 0; i < a.n_gen; i++) {
        if (llama_vocab_is_eog(vocab, tok)) break;
        b.n_tokens = 1; b.token[0] = tok; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        if (llama_decode(cd, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        tok = greedy(cd, vocab);
        r.n_gen++;
        if (a.check) r.text += piece(vocab, tok);
    }
    llama_synchronize(cd);
    r.decode_ms = now_ms() - t2;
    llama_batch_free(b);
    return r;
}

int main(int argc, char ** argv) {
    args a = parse(argc, argv);
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    ggml_backend_load_all();

    const bool split = a.prefill_dev != a.decode_dev;
    llama_model * mp = load_on(a.model, a.prefill_dev);
    llama_model * md = split ? load_on(a.model, a.decode_dev) : mp;
    llama_context * cp = make_ctx(mp, a, a.threads_b, a.prefill_dev == "CPU");
    llama_context * cd = split ? make_ctx(md, a, a.threads, a.decode_dev == "CPU") : cp;
    if (!split) llama_set_n_threads(cp, a.threads, a.threads_b);

    const llama_vocab * vocab = llama_model_get_vocab(mp);
    auto toks = make_prompt(vocab, a);

    // warmup: first graph build, NPU session setup, CPU repack touch
    { args w = a; w.n_gen = 4; w.check = false; run_once(cp, cd, vocab, std::vector<llama_token>(toks.begin(), toks.begin() + std::min<size_t>(64, toks.size())), w); }

    std::vector<result> rs;
    for (int i = 0; i < a.reps; i++) rs.push_back(run_once(cp, cd, vocab, toks, a));

    auto mean = [&](auto f) { double s = 0; for (auto & r : rs) s += f(r); return s / rs.size(); };
    double pf = mean([](const result & r) { return r.prefill_ms; });
    double xf = mean([](const result & r) { return r.xfer_ms; });
    double dc = mean([](const result & r) { return r.decode_ms; });
    double ng = mean([](const result & r) { return (double) r.n_gen; });
    printf("mode=%s prefill=%s decode=%s n_prompt=%d n_gen=%.0f | prefill %.1f ms (%.1f t/s) | handoff %.2f ms (%.2f MB) | TTFT %.1f ms | decode %.2f t/s\n",
        split ? "split" : "single", a.prefill_dev.c_str(), a.decode_dev.c_str(), rs[0].n_prompt, ng,
        pf, rs[0].n_prompt / (pf / 1000.0), xf, rs[0].xfer_bytes / 1048576.0, pf + xf, ng / (dc / 1000.0));
    if (a.check) printf("--- text ---\n%s\n", rs.back().text.c_str());
    // a backend teardown exits without flushing stdio, so flush before freeing anything
    fflush(stdout);

    if (split) { llama_free(cd); llama_model_free(md); }
    llama_free(cp);
    llama_model_free(mp);
    llama_backend_free();
    return 0;
}
