// trilane-bitcheck: is decoding k+1 tokens per call (speculative verification) bit-identical to one token per call?
// Run A decodes the greedy continuation of a prompt one token per call; run B feeds the same tokens C per call
// (optionally with several sequences in the batch, like the multi-slot server). The tool compares the logits of every
// position bitwise and, at the first mismatch, names the first graph tensor whose row for that position differs.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static uint64_t fnv(const void * p, size_t n) {
    const unsigned char * c = (const unsigned char *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

struct rec {
    int pos0 = 0;           // position of row 0 of the current ubatch (single sequence 0 runs only)
    int n_tok = 1;          // tokens of sequence 0 in the ubatch
    int n_seq = 1;
    bool on = false;
    std::map<std::string, std::map<int, uint64_t>> h;   // tensor name -> position -> row hash (sequence 0)
    std::vector<std::string> order;                        // first-seen order of names
    std::map<std::string, std::map<int, std::vector<float>>> v; // first values of each row
    std::map<std::string, std::string> shape;
    std::map<std::string, int> occ;   // occurrences of each name in the current graph (names repeat)
};

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    rec * r = (rec *) ud;
    if (ask) return r->on;
    const int occ = r->occ[t->name]++;
    if (strncmp(t->name, "cache_", 6) == 0) return true;
    if (!getenv("BITCHECK_ALL") && (strncmp(t->name, "node_", 5) == 0 || t->name[0] == ' ' || t->name[0] == 0)) return true; // unnamed: no stable key
    if (t->type != GGML_TYPE_F32 || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) || !ggml_is_contiguous(t)) return true;
    const int T = r->n_tok, S = r->n_seq;
    // [d, T*S] rows or [d, h, T, S] slices; sequence 0 comes first in both layouts
    int64_t row_bytes = 0;
    if (t->ne[1] == (int64_t) T * S && t->ne[2] == 1 && t->ne[3] == 1) row_bytes = t->nb[1];
    else if (t->ne[2] == T && t->ne[3] == S && t->ne[1] > 1) row_bytes = t->nb[2];
    else return true;
    const std::string name = std::string(t->name) + "#" + std::to_string(occ);
    if (!r->h.count(name)) r->order.push_back(name);
    for (int i = 0; i < T; i++) {
        const float * f = (const float *) ((const char *) t->data + i * row_bytes);
        r->h[name][r->pos0 + i] = fnv(f, row_bytes);
        r->v[name][r->pos0 + i] = std::vector<float>(f, f + std::min<int64_t>(4, row_bytes / 4));
    }
    char sh[128];
    snprintf(sh, sizeof(sh), "%s [%lld %lld %lld %lld]", ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    r->shape[name] = sh;
    return true;
}

int main(int argc, char ** argv) {
    std::string model, pfile;
    int n_gen = 64, chunk = 4, rs_a = 0, rs_b = 3, threads = 8, n_seq = 1, n_seq_a = 1, kvu = 1; bool trace = false;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (k == "-m") model = next();
        else if (k == "-f") pfile = next();
        else if (k == "-n") n_gen = atoi(next());
        else if (k == "--chunk") chunk = atoi(next());
        else if (k == "--rs-a") rs_a = atoi(next());
        else if (k == "--rs-b") rs_b = atoi(next());
        else if (k == "--seqs") n_seq = atoi(next());
        else if (k == "--seqs-a") n_seq_a = atoi(next());
        else if (k == "--kvu") kvu = atoi(next());
        else if (k == "-t") threads = atoi(next());
        else if (k == "--trace") trace = true;
    }
    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    static ggml_backend_dev_t nodev[] = { nullptr };
    mp.devices = nodev;
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int nv = llama_vocab_n_tokens(vocab);
    std::string text;
    { std::ifstream f(pfile, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }
    std::vector<llama_token> prompt(text.size() + 16);
    prompt.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), prompt.data(), (int) prompt.size(), true, true));

    auto make = [&](int rs, rec * r) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096 * std::max(n_seq, n_seq_a); cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_seq_max = std::max(n_seq, n_seq_a); cp.kv_unified = kvu != 0;
        cp.n_threads = threads; cp.n_threads_batch = threads; cp.no_perf = true; cp.op_offload = false;
        cp.n_rs_seq = rs; cp.n_outputs_max_per_seq = 0;
        cp.cb_eval = cb; cp.cb_eval_user_data = r;
        return llama_init_from_model(m, cp);
    };
    // prompt into every sequence, one sequence at a time
    auto do_prompt = [&](llama_context * c) {
        for (int s = 0; s < std::max(n_seq, n_seq_a); s++) {
            llama_batch b = llama_batch_init((int) prompt.size(), 0, 1);
            b.n_tokens = (int) prompt.size();
            for (int i = 0; i < b.n_tokens; i++) { b.token[i] = prompt[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = i + 1 == b.n_tokens; }
            if (llama_decode(c, b)) { fprintf(stderr, "prompt failed\n"); exit(1); }
            llama_batch_free(b);
        }
    };
    auto argmax = [&](const float * l) { int bi = 0; for (int i = 1; i < nv; i++) if (l[i] > l[bi]) bi = i; return bi; };

    // run A: one token per call, sequence 0 only... but with the same other sequences present
    rec ra, rb;
    llama_context * ca = make(rs_a, &ra);
    do_prompt(ca);
    std::vector<llama_token> toks;
    std::vector<uint64_t> la;
    llama_token cur = argmax(llama_get_logits_ith(ca, -1));
    ra.on = trace;
    for (int i = 0; i < n_gen; i++) {
        toks.push_back(cur);
        llama_batch b = llama_batch_init(n_seq_a, 0, 1);
        b.n_tokens = n_seq_a;
        for (int s = 0; s < n_seq_a; s++) { b.token[s] = cur; b.pos[s] = (int) prompt.size() + i; b.n_seq_id[s] = 1; b.seq_id[s][0] = s; b.logits[s] = 1; }
        ra.occ.clear(); ra.pos0 = (int) prompt.size() + i; ra.n_tok = 1; ra.n_seq = n_seq_a;
        if (llama_decode(ca, b)) { fprintf(stderr, "A decode failed\n"); return 1; }
        llama_batch_free(b);
        const float * l = llama_get_logits_ith(ca, 0);
        la.push_back(fnv(l, sizeof(float) * nv));
        cur = argmax(l);
    }
    llama_free(ca);

    // run B: the same tokens, chunk per call and sequence
    llama_context * cb2 = make(rs_b, &rb);
    do_prompt(cb2);
    std::vector<uint64_t> lb(n_gen);
    rb.on = trace;
    for (int i0 = 0; i0 < n_gen; i0 += chunk) {
        const int n = std::min(chunk, n_gen - i0);
        llama_batch b = llama_batch_init(n * n_seq, 0, 1);
        b.n_tokens = n * n_seq;
        for (int s = 0; s < n_seq; s++) for (int j = 0; j < n; j++) {
            const int q = s * n + j;
            b.token[q] = toks[i0 + j]; b.pos[q] = (int) prompt.size() + i0 + j; b.n_seq_id[q] = 1; b.seq_id[q][0] = s; b.logits[q] = 1;
        }
        rb.occ.clear(); rb.pos0 = (int) prompt.size() + i0; rb.n_tok = n; rb.n_seq = n_seq;
        if (llama_decode(cb2, b)) { fprintf(stderr, "B decode failed\n"); return 1; }
        llama_batch_free(b);
        for (int j = 0; j < n; j++) lb[i0 + j] = fnv(llama_get_logits_ith(cb2, j), sizeof(float) * nv);
    }
    int first = -1, ndiff = 0;
    for (int i = 0; i < n_gen; i++) if (la[i] != lb[i]) { ndiff++; if (first < 0) first = i; }
    printf("seqs_a %d chunk %d seqs %d rs_a %d rs_b %d: %d of %d positions with different logits", n_seq_a, chunk, n_seq, rs_a, rs_b, ndiff, n_gen);
    if (first < 0) { printf(" -> BIT-IDENTICAL\n"); fflush(stdout); return 0; }
    const int pos = (int) prompt.size() + first;
    printf(", first at step %d (pos %d)\n", first, pos);
    int shown = 0;
    if (getenv("BITCHECK_ORDER")) {
        for (int i = 0; i < 90 && i < (int) ra.order.size(); i++) printf("  A[%d] %-36s %-36s %04llx\n", i, ra.order[i].c_str(), ra.shape[ra.order[i]].c_str(), (unsigned long long) (ra.h[ra.order[i]][pos] & 0xffff));
        for (int i = 0; i < 90 && i < (int) rb.order.size(); i++) printf("  B[%d] %-36s %-36s %04llx\n", i, rb.order[i].c_str(), rb.shape[rb.order[i]].c_str(), (unsigned long long) (rb.h[rb.order[i]][pos] & 0xffff));
    }
    for (const auto & name : ra.order) {
        auto ia = ra.h[name].find(pos);
        if (ia == ra.h[name].end() || !rb.h.count(name)) continue;
        auto ib = rb.h[name].find(pos);
        if (ib == rb.h[name].end()) continue;
        if (ia->second != ib->second) {
            const auto & va = ra.v[name][pos], & vb = rb.v[name][pos];
            printf("  differing rows: %-28s A %s B %s | A %.9g %.9g B %.9g %.9g\n", name.c_str(), ra.shape[name].c_str(), rb.shape[name].c_str(),
                   va.size() > 0 ? va[0] : 0.f, va.size() > 1 ? va[1] : 0.f, vb.size() > 0 ? vb[0] : 0.f, vb.size() > 1 ? vb[1] : 0.f);
            if (++shown >= 12) break;
        }
    }
    // also: earliest position with any differing tensor
    int earliest = 1 << 30; std::string en;
    for (const auto & name : ra.order) {
        if (!rb.h.count(name)) continue;
        for (auto & kv : ra.h[name]) { auto ib = rb.h[name].find(kv.first); if (ib != rb.h[name].end() && ib->second != kv.second && kv.first < earliest) { earliest = kv.first; en = name; } }
    }
    printf("  earliest position with a differing tensor: %d (%s)\n", earliest, en.c_str());
    fflush(stdout);
    return 0;
}
