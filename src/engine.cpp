#include "engine.h"

#include "ggml-alloc.h"
#include "llama-ext.h"   // fork staging API: target hidden states for the MTP head (llama_set/get_embeddings_nextn)
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace trilane {

// The decode thread pool runs graph worker 0 on the thread that calls llama_decode (cpu_worker), but ggml sets
// priority and turns off Windows power throttling (EcoQoS) only for its own workers and for the thread that created
// the pool. Without this, worker 0 runs throttled at normal priority and the other workers wait for it at every
// barrier (and run all single-thread nodes slower). TRILANE_WORKER0_PRIO=0 restores the old behaviour (together with
// --prio 0, the old engine exactly).
static bool worker0_fix_on() {
    const char * e = getenv("TRILANE_WORKER0_PRIO");
    return !(e && atoi(e) == 0);
}

static void decode_thread_like_pool(int prio) {
#ifdef _WIN32
    if (!worker0_fix_on()) return;
    THREAD_POWER_THROTTLING_STATE t;
    ZeroMemory(&t, sizeof(t));
    t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    t.StateMask   = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t));
    int p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case 1: p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case 2: p = THREAD_PRIORITY_HIGHEST;       break;
        case 3: p = THREAD_PRIORITY_TIME_CRITICAL; break;
        default: break;
    }
    if (p != THREAD_PRIORITY_NORMAL) SetThreadPriority(GetCurrentThread(), p);
#else
    (void) prio;
#endif
}

// CPU_HTPTILE hooks exported by our ggml-cpu fork (see ggml/src/ggml-cpu/htp-tile.cpp)
typedef void * (*htptile_resolver_t)(const char * name, const ggml_tensor * t, void * user);
typedef void (*htptile_set_resolver_t)(htptile_resolver_t fn, void * user);
typedef ggml_backend_buffer_type_t (*htptile_buft_t)(void);

static void * resolve_from_model(const char * name, const ggml_tensor * t, void * user) {
    const ggml_tensor * src = llama_model_get_tensor((const llama_model *) user, name);
    if (!src || src->type != t->type || ggml_nelements(src) != ggml_nelements(t)) return nullptr;
    return src->data;
}

// Names of weights the decode model can read in place from the prefill model: Q4_0, tile-aligned, and
// actually resident (hence tiled) in an HTP buffer. MoE models alias experts only, dense models every match.
static std::vector<std::string> alias_candidates(const std::string & path, const llama_model * mp, bool experts_only) {
    std::vector<std::string> out;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return out;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const std::string name = gguf_get_tensor_name(g, i);
        if (name.rfind("token_embd", 0) == 0) continue;            // GET_ROWS needs the flat layout
        if (experts_only && name.find("_exps") == std::string::npos) continue;
        const ggml_tensor * t = llama_model_get_tensor(mp, name.c_str());
        if (!t || t->type != GGML_TYPE_Q4_0 || t->ne[3] != 1 || t->ne[0] % 32 || t->ne[1] % 32 || !ggml_is_contiguous(t)) continue;
        if (!t->buffer || !strstr(ggml_backend_buffer_name(t->buffer), "HTP")) continue;
        out.push_back(name);
    }
    gguf_free(g);
    return out;
}

// exact-name patterns, joined into a few alternations to keep the loader's regex pass cheap
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

// number of MTP (nextn) layers in a GGUF, 0 if none or unreadable
static int gguf_nextn_layers(const std::string & path) {
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return 0;
    int n = 0;
    const int64_t ka = gguf_find_key(g, "general.architecture");
    if (ka >= 0) {
        const std::string key = std::string(gguf_get_val_str(g, ka)) + ".nextn_predict_layers";
        const int64_t kn = gguf_find_key(g, key.c_str());
        if (kn >= 0) n = (int) gguf_get_val_u32(g, kn);
    }
    gguf_free(g);
    return n;
}

static bool has_experts(const llama_model * m) {
    char name[64];
    for (int i = 0; i < llama_model_n_layer(m); i++) {
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", i);
        if (llama_model_get_tensor(m, name)) return true;
    }
    return false;
}

// number of leading bytes of s that form complete UTF-8 sequences
static size_t utf8_complete(const std::string & s) {
    size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;            // continuation byte
        size_t len = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return back >= len ? n : n - back;           // drop a trailing partial sequence
    }
    return n;
}

static llama_sampler * make_sampler(const gen_params & gp) {
    llama_sampler * s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (gp.temperature <= 0.0f) {
        llama_sampler_chain_add(s, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(s, llama_sampler_init_top_k(64));
        llama_sampler_chain_add(s, llama_sampler_init_top_p(gp.top_p, 1));
        llama_sampler_chain_add(s, llama_sampler_init_min_p(gp.min_p, 1));
        llama_sampler_chain_add(s, llama_sampler_init_temp(gp.temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist(gp.seed));
    }
    return s;
}

// --- request ---

request::~request() {
    if (smpl) llama_sampler_free(smpl);
}

void request::push(std::string s) {
    { std::lock_guard<std::mutex> lk(mtx); pieces.push_back(std::move(s)); }
    cv.notify_one();
}

void request::finish(const std::string & reason) {
    { std::lock_guard<std::mutex> lk(mtx); if (st.stop_reason.empty()) st.stop_reason = reason; done = true; }
    cv.notify_all();
}

// --- engine ---

engine::~engine() {
    {
        std::lock_guard<std::mutex> lk(q_mtx_);
        stop_ = true;
    }
    q_cv_.notify_all();
    if (npu_thread_.joinable()) npu_thread_.join();
    if (cpu_thread_.joinable()) cpu_thread_.join();
    keepwarm_stop();
    // the aliased decode model points into the NPU model's buffers, so it must go first
    if (cm_) llama_free(cm_);
    if (cd_ && cd_ != cp_) llama_free(cd_);
    if (md_ && md_ != mp_) llama_model_free(md_);
    if (cp_) llama_free(cp_);
    if (mp_) llama_model_free(mp_);
    if (tp_decode_) {
        auto tp_free = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CPU"), "ggml_threadpool_free");
        if (tp_free) tp_free(tp_decode_);
    }
}

double engine::now_ms() const { return llama_time_us() / 1000.0; }

static llama_model_params base_mparams() {
    llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_NONE; // read once; mmap + CPU repack keeps two copies
    return mp;
}

bool engine::load(const engine_params & p) {
    params_ = p;
    params_.max_seqs = std::max(1, params_.max_seqs);
    static std::vector<ggml_backend_dev_t> npu_devs, no_devs;

    const bool use_npu = p.prefill_dev != "CPU";
    if (use_npu) {
        std::stringstream ss(p.prefill_dev);
        for (std::string name; std::getline(ss, name, ',');) {
            ggml_backend_dev_t d = ggml_backend_dev_by_name(name.c_str());
            if (!d) { fprintf(stderr, "[trilane] device %s not found (is ADSP_LIBRARY_PATH set?)\n", name.c_str()); return false; }
            npu_devs.push_back(d);
        }
        npu_devs.push_back(nullptr);
        llama_model_params mp = base_mparams();
        mp.devices = npu_devs.data();
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        double t0 = now_ms();
        const std::string & pm = p.prefill_model.empty() ? p.model : p.prefill_model;
        mp_ = llama_model_load_from_file(pm.c_str(), mp);
        if (!mp_) return false;
        fprintf(stderr, "[trilane] prefill model %s on %s loaded in %.0f ms\n", pm.c_str(), p.prefill_dev.c_str(), now_ms() - t0);
    }

    no_devs = { nullptr };
    llama_model_params dp = base_mparams();
    dp.devices = no_devs.data();
    dp.n_gpu_layers = 0;

    // speculative decoding: on by default for models with an MTP head; models without one take the old path exactly
    {
        int spec = p.spec;
        if (const char * e = getenv("TRILANE_SPEC")) spec = strcmp(e, "auto") == 0 ? -1 : atoi(e) != 0 ? 1 : 0;
        const int nextn = gguf_nextn_layers(p.model);
        // draft length per number of decoding slots; the rollback depth is the longest one
        // default table {nmax, 1, 0}: measured best on Qwen3.5-2B and Qwen3.8-27B (drafting loses from 3-4 decoding
        // slots on); spec_k = {-1} (--spec-k auto) lets the online tuner choose, seeded with the default table
        spec_auto_ = params_.spec_k.size() == 1 && params_.spec_k[0] < 0;
        if (params_.spec_k.empty() || spec_auto_) params_.spec_k = { p.spec_nmax, std::min(p.spec_nmax, 1), 0 };
        params_.spec_nmax = 0;
        for (int k : params_.spec_k) params_.spec_nmax = std::max(params_.spec_nmax, k);
        spec_ = spec != 0 && nextn > 0 && params_.spec_nmax > 0;
        // Batch-invariant CPU matmuls for MTP models (drafting on or off): the K-quant gemv and gemm kernels round
        // differently, so without this a token's greedy choice could depend on how many rows its decode call carried
        // (k + 1 when verifying drafts, one per slot when slots share a step). TRILANE_BATCH_INVARIANT=0 keeps the
        // faster mixed kernels; models without an MTP head are not touched.
        // On by default only while drafting is on: with --spec off / TRILANE_SPEC=0 the old kernels and the shared KV
        // pool run exactly as before (TRILANE_BATCH_INVARIANT=1 forces the invariant kernels without drafting, the
        // reference for text identity; =0 turns them off with drafting).
        const char * bi = getenv("TRILANE_BATCH_INVARIANT");
        batch_invariant_ = nextn > 0 && (bi ? atoi(bi) != 0 : spec_);
        if (batch_invariant_ && !getenv("GGML_CPU_BATCH_INVARIANT")) _putenv_s("GGML_CPU_BATCH_INVARIANT", "1");
        if (spec == 1 && nextn <= 0) fprintf(stderr, "[trilane] speculative decoding requested, but %s has no MTP head: off\n", p.model.c_str());
        dp.load_mtp = spec_;
    }
    static std::vector<std::string> alias_pats;
    static std::vector<llama_model_tensor_buft_override> alias_ovr;
    // a separate prefill file has its own tensor types, so the decode model cannot alias its weights
    if (use_npu && p.alias_experts && p.prefill_model.empty()) {
        ggml_backend_reg_t cpu = ggml_backend_reg_by_name("CPU");
        auto set_resolver = (htptile_set_resolver_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_htptile_set_resolver");
        auto get_buft     = (htptile_buft_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_htptile_buffer_type");
        // aliased decode is ~7% slower and its batched CPU prefill much slower, so only alias what memory forces:
        // MoE experts always, dense weights only for models too big to hold twice next to the OS
        const bool moe   = has_experts(mp_);
        const bool dense = !moe && llama_model_size(mp_) > (6ull << 30);
        const auto names = (moe || dense) ? alias_candidates(p.model, mp_, moe) : std::vector<std::string>();
        if (set_resolver && get_buft && !names.empty()) {
            set_resolver(resolve_from_model, (void *) mp_);
            alias_pats = exact_name_patterns(names);
            for (auto & pat : alias_pats) alias_ovr.push_back({ pat.c_str(), get_buft() });
            alias_ovr.push_back({ nullptr, nullptr });
            dp.tensor_buft_overrides = alias_ovr.data();
            dp.load_mode = LLAMA_LOAD_MODE_MMAP;          // aliased weights are never read from the file
            _putenv_s("LLAMA_MMAP_NO_PREFETCH", "1");     // nor prefetched into RAM
            aliased_ = true;
            dense_aliased_ = dense;
            n_aliased_ = (int) names.size();
            if (dense) {
                // the CPU lane only decodes: batched prefill over aliased dense weights is per-token GEMV
                params_.calibrate = false;
                params_.cpu_prefill_max = 0;
            }
            fprintf(stderr, "[trilane] aliasing %d %s tensors of the decode model to NPU memory\n", n_aliased_, moe ? "expert" : "dense");
        }
    }
    {
        double t0 = now_ms();
        md_ = llama_model_load_from_file(p.model.c_str(), dp);
        if (!md_) return false;
        fprintf(stderr, "[trilane] decode model on CPU%s loaded in %.0f ms\n", aliased_ ? " (weights aliased to NPU memory)" : "", now_ms() - t0);
    }
    if (!mp_) mp_ = md_;

    // models larger than an HTP session's ~3 GB window remap weights once per ubatch, so they want big ubatches;
    // small models fit, and big ubatches can exceed per-op NPU buffer limits (e.g. a 248K-vocab head).
    // 4096 rather than 2048: the HTP flash attention skips the fully masked (causal) KV blocks, so a bigger
    // ubatch no longer pays for the masked half of its attention square.
    if (params_.n_ubatch <= 0) params_.n_ubatch = llama_model_size(mp_) > (3ull << 30) ? 4096 : 512;
    params_.n_batch = std::max(params_.n_batch, params_.n_ubatch);
    if (mp_ == md_) params_.npu_decode_slots = 0;
    fprintf(stderr, "[trilane] n_ubatch = %d, decode slots = %d CPU + %d NPU\n", params_.n_ubatch, params_.max_seqs, params_.npu_decode_slots);

    // prefill_only: the context only ever asks for one output per sequence (the last prompt token, or one NPU
    // decode step), so cap n_outputs_max instead of reserving an n_batch x n_vocab logits buffer (2 GB at 2048)
    // verification batches (k + 1 tokens per slot) are compute-bound: two threads more than single-token steps measured
    // +9% (2B) and +7% (27B split route) with one slot, +5% / +9% with two (3 interleaved rounds each)
    const int tb = params_.threads_batch > 0 ? params_.threads_batch
                 : spec_ ? std::min(params_.threads_decode + 2, std::max(params_.threads_decode, (int) std::thread::hardware_concurrency()))
                         : params_.threads_decode;
    params_.threads_batch = tb;
    auto make_ctx = [&](llama_model * m, int threads, bool host_only, int n_seq, bool prefill_only, bool spec_tgt) {
        llama_context_params cp = llama_context_default_params();
        if (spec_tgt) {
            cp.n_rs_seq              = (uint32_t) params_.spec_nmax; // bounded recurrent-state rollback per slot
            cp.n_outputs_max_per_seq = 0;                            // verification needs k+1 outputs per slot
        }
        if (prefill_only) cp.n_outputs_max = (uint32_t) std::max(1, n_seq);
        cp.n_ctx           = params_.n_ctx * n_seq;
        cp.n_batch         = params_.n_batch;
        cp.n_ubatch        = params_.n_ubatch;
        if (spec_tgt && dense_aliased_) {
            // the CPU lane of an aliased dense model never prefills (cpu_prefill_max = 0), it only sees decode and
            // verification steps: a small compute buffer leaves room for the MTP context (~1 GB less on 27B at 4096)
            cp.n_batch = cp.n_ubatch = std::max(64, n_seq * (1 + params_.spec_nmax));
        }
        cp.n_seq_max       = n_seq;
        cp.kv_unified      = true;   // slots share one cell pool
        // batch-invariant decoding (MTP models): one KV stream per slot. In a shared pool a slot's cells can sit out
        // of position order (cells freed by rolled-back drafts are reused first), and attention sums cells in cache
        // order with f16 accumulation, so a token's logits depended on the other slots; a stream per slot keeps
        // every slot's cells in position order, as when it decodes alone (TRILANE_SPEC_KVU=1: shared pool)
        if (host_only && n_seq > 1 && batch_invariant_ && !getenv("TRILANE_SPEC_KVU")) cp.kv_unified = false;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        cp.no_perf         = true;
        cp.op_offload      = !host_only;
        return llama_init_from_model(m, cp);
    };
    cd_ = make_ctx(md_, params_.threads_decode, true, params_.max_seqs, false, spec_);
    cp_ = mp_ != md_ ? make_ctx(mp_, params_.threads_prefill, false, 1, true, false) : cd_;
    if (!cd_ || !cp_) return false;

    if (spec_) {
        // MTP draft context on the decode model: one sequence per slot, plain attention cache (no recurrent state)
        llama_context_params cp = llama_context_default_params();
        cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx           = params_.n_ctx * params_.max_seqs;
        cp.n_batch         = params_.n_batch;
        cp.n_ubatch        = std::min(512, params_.n_batch);
        cp.n_seq_max       = params_.max_seqs;
        cp.kv_unified      = true;
        cp.n_threads       = params_.threads_decode;
        cp.n_threads_batch = tb;
        cp.no_perf         = true;
        cp.op_offload      = false;
        cm_ = llama_init_from_model(md_, cp);
        if (!cm_) {
            fprintf(stderr, "[trilane] cannot create the MTP draft context: speculative decoding off\n");
            spec_ = false;
        } else {
            n_embd_  = llama_model_n_embd_out(md_);
            n_vocab_ = llama_vocab_n_tokens(llama_model_get_vocab(md_));
            int dh = params_.spec_draft_head < 0 ? (n_vocab_ >= 131072 ? 65536 : 0) : params_.spec_draft_head;
            if (dh + 256 >= n_vocab_) dh = 0;
            draft_head_ = dh;
            if (dh > 0) llama_set_draft_head(cm_, dh, 256);
            llama_set_embeddings_nextn(cd_, true, /*masked*/ false); // hidden rows of every verified token
            llama_set_embeddings_nextn(cm_, true, /*masked*/ true);  // the MTP head chains its own hidden state
            if (cp_ != cd_) llama_set_embeddings_nextn(cp_, true, /*masked*/ false); // prompt rows of the NPU prefill
            std::string ks;
            for (size_t i = 0; i < params_.spec_k.size(); i++) ks += (i ? "," : "") + std::to_string(params_.spec_k[i]);
            fprintf(stderr, "[trilane] speculative decoding: MTP head, drafted tokens per slot by decoding slots %s (last holds for more), draft head %s, verification threads %d\n",
                    ks.c_str(), dh > 0 ? (std::to_string(dh) + ":256").c_str() : "full", tb);
        }
    } else {
        fprintf(stderr, "[trilane] speculative decoding: off\n");
    }
    if (spec_ && spec_auto_) tuner_.init(params_.spec_nmax, params_.max_seqs, params_.spec_k);
    if (batch_invariant_) fprintf(stderr, "[trilane] batch-invariant CPU matmuls on (TRILANE_BATCH_INVARIANT=0 turns them off)\n");

    // Without an attached thread pool, ggml creates and joins a new pool of worker threads for every graph,
    // i.e. for every decode step. A persistent pool keeps them (polling between steps, like llama-bench).
    if (params_.threadpool) {
        ggml_backend_reg_t cpu = ggml_backend_reg_by_name("CPU");
        auto tp_new = cpu ? (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu, "ggml_threadpool_new") : nullptr;
        if (tp_new) {
            // with speculative decoding, verification batches may use more threads (threads_batch): one pool of the
            // larger size, every graph runs on its own thread count and the spare threads sleep
            ggml_threadpool_params tpp = ggml_threadpool_params_default(spec_ ? std::max(params_.threads_decode, tb) : params_.threads_decode);
            tpp.prio = (enum ggml_sched_priority) params_.prio;
#ifdef _WIN32
            // ggml_threadpool_new also applies the pool priority to the calling thread (this one, which later runs
            // the calibration, the HTTP server and the bench drivers). Only the decode threads should be raised.
            const int main_prio = GetThreadPriority(GetCurrentThread());
#endif
            tp_decode_ = tp_new(&tpp);
#ifdef _WIN32
            if (worker0_fix_on() && main_prio != THREAD_PRIORITY_ERROR_RETURN) SetThreadPriority(GetCurrentThread(), main_prio);
#endif
        }
        if (tp_decode_) llama_attach_threadpool(cd_, tp_decode_, nullptr);
        if (tp_decode_ && cm_) llama_attach_threadpool(cm_, tp_decode_, nullptr);
        fprintf(stderr, "[trilane] decode thread pool: %s (%d threads, prio %d)\n", tp_decode_ ? "persistent" : "FAILED, per-step", params_.threads_decode, params_.prio);
    }

    vocab_ = llama_model_get_vocab(mp_);
    char desc[256];
    llama_model_desc(mp_, desc, sizeof(desc));
    model_name_ = p.model.substr(p.model.find_last_of("/\\") + 1);
    const char * t = llama_model_chat_template(mp_, nullptr);
    tmpl_ = t ? t : "";
    fprintf(stderr, "[trilane] %s (%s), split=%d aliased=%d\n", model_name_.c_str(), desc, split(), aliased_);

    if (split() && params_.keepwarm_ms > 0) keepwarm_start();
    if (split() && params_.calibrate) calibrate();

    if (split()) npu_thread_ = std::thread([this] { npu_worker(); });
    cpu_thread_ = std::thread([this] { cpu_worker(); });
    return true;
}

// Times synthetic prefills of two lengths on each lane, fits t(n) = fixed + n/tps, and sets the routing threshold.
// Runs before the workers start, so it may use both contexts directly.
void engine::calibrate() {
    // prefill() chunks by n_batch, so the lengths need not follow n_ubatch
    const int n_short = 32, n_long = std::min(512, params_.n_ctx - 1);
    if (n_long <= n_short) return;
    std::vector<llama_token> toks(n_long);
    for (int i = 0; i < n_long; i++) toks[i] = (llama_token) (1000 + (i * 7919) % 20000);
    auto time_prefill = [&](llama_context * c, int n, bool npu) {
        std::vector<llama_token> t(toks.begin(), toks.begin() + n);
        double best = 1e30;
        for (int rep = 0; rep < 2; rep++) {
            if (npu) keepwarm_pause(true);
            llama_memory_clear(llama_get_memory(c), true);
            double t0 = now_ms();
            if (!prefill(c, t, 0, 0)) fprintf(stderr, "[trilane] calibration prefill failed\n");
            best = std::min(best, now_ms() - t0);
            if (npu) keepwarm_pause(false);
        }
        return best;
    };
    auto fit = [&](double ts, double tl, double & fixed_ms, double & tps) {
        const double per_tok = std::max((tl - ts) / (n_long - n_short), 1e-6); // ms per token
        tps      = 1000.0 / per_tok;
        fixed_ms = std::max(ts - n_short * per_tok, 0.0);
    };
    time_prefill(cp_, n_short, true); // warm both graphs
    time_prefill(cd_, n_short, false);
    fit(time_prefill(cd_, n_short, false), time_prefill(cd_, n_long, false), cost_.cpu_fixed_ms, cost_.cpu_tps);
    fit(time_prefill(cp_, n_short, true), time_prefill(cp_, n_long, true), cost_.npu_fixed_ms, cost_.npu_tps);
    llama_memory_clear(llama_get_memory(cp_), true);
    llama_memory_clear(llama_get_memory(cd_), true);

    const double gain_per_tok = 1000.0 / cost_.cpu_tps - 1000.0 / cost_.npu_tps;
    const double extra_fixed  = cost_.npu_fixed_ms - cost_.cpu_fixed_ms + 5.0;
    cost_.crossover = gain_per_tok > 0 ? (int) std::max(0.0, extra_fixed / gain_per_tok) : 1 << 30;
    params_.cpu_prefill_max = cost_.crossover;
    fprintf(stderr, "[trilane] lane cost model: CPU %.0f ms + n/%.0f t/s | NPU %.0f ms + n/%.0f t/s -> NPU prefill above %d new tokens\n",
            cost_.cpu_fixed_ms, cost_.cpu_tps, cost_.npu_fixed_ms, cost_.npu_tps, cost_.crossover);
}

std::string engine::apply_template(const std::vector<chat_msg> & msgs) const {
    std::vector<llama_chat_message> cm;
    for (auto & m : msgs) cm.push_back({ m.role.c_str(), m.content.c_str() });
    std::vector<char> buf(4096);
    int n = llama_chat_apply_template(tmpl_.empty() ? nullptr : tmpl_.c_str(), cm.data(), cm.size(), true, buf.data(), (int) buf.size());
    if (n > (int) buf.size()) {
        buf.resize(n + 1);
        n = llama_chat_apply_template(tmpl_.empty() ? nullptr : tmpl_.c_str(), cm.data(), cm.size(), true, buf.data(), (int) buf.size());
    }
    if (n >= 0) return std::string(buf.data(), n);
    // fallback: ChatML
    std::string s;
    for (auto & m : msgs) s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    return s + "<|im_start|>assistant\n";
}

std::vector<llama_token> engine::tokenize(const std::string & text) const {
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab_, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab_, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, true); }
    toks.resize(std::max(n, 0));
    return toks;
}

// The MTP head catches up on the most recent TRILANE_MTP_WINDOW (default 1024) positions only. A slot that does not
// draft for a long time (3+ decoding slots with the default table) or a long prompt would otherwise keep one hidden
// row per token and replay all of them through the MTP layer in one step, stalling every slot. Older positions are
// left out of the MTP cache: that only changes the drafts, never the text.
static size_t mtp_window() {
    static const size_t w = [] { const char * e = getenv("TRILANE_MTP_WINDOW"); return e ? (size_t) std::max(16, atoi(e)) : (size_t) 1024; }();
    return w;
}

// prefill toks[from..] at positions from.. into sequence seq, logits for the last token only; false on a failed decode
bool engine::prefill(llama_context * c, const std::vector<llama_token> & toks, int from, llama_seq_id seq, std::vector<float> * h_out) {
    // the context's own batch limit: a dense-aliased decode context with drafting has a small n_batch
    const int nb = std::min(params_.n_batch, (int) llama_n_batch(c));
    llama_batch b = llama_batch_init(nb, 0, 1);
    // the MTP head only catches up on the last mtp_window() prompt tokens (each paired with the hidden row before it):
    // h_out keeps the rows of the chunks that reach them. (Switching the hidden-row output off for the other chunks
    // cost ~180 ms of first-token time on the NPU context: its graph is rebuilt.)
    const int need_from = h_out ? std::max(from, (int) toks.size() - (int) mtp_window() - 1) : (int) toks.size();
    for (int i0 = from; i0 < (int) toks.size(); i0 += nb) {
        const int n = std::min(nb, (int) toks.size() - i0);
        const bool need = h_out && i0 + n > need_from;
        b.n_tokens = n;
        for (int j = 0; j < n; j++) {
            b.token[j] = toks[i0 + j]; b.pos[j] = i0 + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = seq;
            b.logits[j] = (i0 + j == (int) toks.size() - 1);
        }
        if (llama_decode(c, b) != 0) {
            fprintf(stderr, "[trilane] prefill decode failed at token %d\n", i0);
            llama_batch_free(b);
            return false;
        }
        if (need) {
            // target hidden states of this chunk, for the MTP head's prompt catch-up on the CPU lane
            llama_synchronize(c);
            const float * h = llama_get_embeddings_nextn(c);
            if (h) h_out->insert(h_out->end(), h, h + (size_t) n * n_embd_);
        }
    }
    llama_synchronize(c);
    llama_batch_free(b);
    return true;
}

// copies the longest cached exact prefix (shorter than toks) into state_out; returns its token count
int engine::cache_lookup(const std::vector<llama_token> & toks, std::vector<uint8_t> & state_out) {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    auto best = cache_.end();
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->toks.size() < toks.size() && std::equal(it->toks.begin(), it->toks.end(), toks.begin()) &&
            (best == cache_.end() || it->toks.size() > best->toks.size())) {
            best = it;
        }
    }
    if (best == cache_.end()) return 0;
    cache_.splice(cache_.begin(), cache_, best);
    state_out = cache_.front().state;
    return (int) cache_.front().toks.size();
}

void engine::cache_insert(const std::vector<llama_token> & toks, const std::vector<uint8_t> & state) {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->toks == toks) { cache_bytes_ -= it->state.size(); cache_.erase(it); break; }
    }
    cache_bytes_ += state.size();
    cache_.push_front({ toks, state });
    while (cache_bytes_ > params_.prefix_cache_mb * 1048576ull && cache_.size() > 1) {
        cache_bytes_ -= cache_.back().state.size();
        cache_.pop_back();
    }
}

size_t engine::cache_entries() const { std::lock_guard<std::mutex> lk(cache_mtx_); return cache_.size(); }
size_t engine::cache_bytes() const { std::lock_guard<std::mutex> lk(cache_mtx_); return cache_bytes_; }
std::vector<gen_stats> engine::history() const { std::lock_guard<std::mutex> lk(hist_mtx_); return history_; }
int engine::npu_queue_len() const { std::lock_guard<std::mutex> lk(q_mtx_); return (int) npu_queue_.size(); }

std::string engine::live_phase() const {
    std::lock_guard<std::mutex> lk(live_mtx_);
    return "NPU: " + npu_phase_ + " | CPU: " + cpu_phase_;
}

// learn how far the calibrated model is off for this lane (warm contexts, cached prefixes, thermal state)
void engine::learn(bool on_npu, int n, double ms) {
    std::lock_guard<std::mutex> lk(cost_mtx_);
    if (cost_.cpu_tps <= 0 || cost_.npu_tps <= 0 || n <= 0) return;
    const double base = on_npu ? cost_.npu_fixed_ms + 1000.0 * n / cost_.npu_tps
                               : cost_.cpu_fixed_ms + 1000.0 * n / cost_.cpu_tps;
    const double ratio = std::clamp(ms / std::max(base, 1.0), 0.25, 4.0);
    double & corr = on_npu ? cost_.npu_corr : cost_.cpu_corr;
    corr = 0.7 * corr + 0.3 * ratio;
}

void engine::set_live(const request_ptr & r, const std::string & lane, const char * phase) {
    std::lock_guard<std::mutex> lk(live_mtx_);
    live_[r->st.id] = { r->st.id, lane, phase, now_ms() };
}

void engine::clear_live(int64_t id) {
    std::lock_guard<std::mutex> lk(live_mtx_);
    live_.erase(id);
}

std::vector<live_seq> engine::live() const {
    std::lock_guard<std::mutex> lk(live_mtx_);
    std::vector<live_seq> v;
    for (auto & kv : live_) v.push_back(kv.second);
    return v;
}

// every finished request passes through here, errors included
void engine::record(const request_ptr & r) {
    clear_live(r->st.id);
    if (r->st.n_gen_npu > 0 && r->st.route != "npu") {
        const std::string npu = params_.prefill_dev.substr(0, params_.prefill_dev.find(','));
        r->st.decode_dev = r->st.n_gen_npu >= r->st.n_gen ? npu : npu + "+CPU";
    }
    std::lock_guard<std::mutex> lk(hist_mtx_);
    history_.push_back(r->st);
    if (history_.size() > 200) history_.erase(history_.begin());
}

request_ptr engine::submit(const std::vector<chat_msg> & msgs, const gen_params & gp) {
    auto r = std::make_shared<request>();
    r->gp = gp;
    r->st.t_submit_ms = now_ms();
    r->toks = tokenize(apply_template(msgs));
    r->st.n_prompt = (int) r->toks.size();
    if (r->toks.empty() || (int64_t) r->toks.size() >= params_.n_ctx) {
        r->finish("context_overflow");
        return r;
    }
    // max_tokens <= 0 means "as many as fit"; never more than the slot's context leaves room for
    const int64_t room = (int64_t) params_.n_ctx - (int64_t) r->toks.size();
    r->gp.max_tokens = (int) (gp.max_tokens <= 0 ? room : std::min<int64_t>(gp.max_tokens, room));
    r->st.n_cached  = cache_lookup(r->toks, r->restore);
    r->st.n_prefill = r->st.n_prompt - r->st.n_cached;
    r->smpl = make_sampler(gp);

    // lane choice: compare predicted prefill time per lane, counting the NPU queue ahead of us
    std::string route = gp.route;
    {
        std::lock_guard<std::mutex> lk(q_mtx_);
        r->st.id = next_id_++;
        if (!split()) route = "cpu";
        else if (route == "auto") {
            std::lock_guard<std::mutex> ck(cost_mtx_);
            if (cost_.cpu_tps > 0 && cost_.npu_tps > 0) {
                double npu_wait = 0;
                for (auto & q : npu_queue_) npu_wait += cost_.predict_npu(q->st.n_prefill);
                if (npu_busy_) npu_wait += cost_.predict_npu(256); // rough remainder of the prefill in flight
                // a CPU prefill pauses every sequence decoding on the CPU lane, so it costs once per stalled sequence too
                const int stalled = n_active_ + (int) cpu_inbox_.size();
                const double cpu_cost = cost_.predict_cpu(r->st.n_prefill) * (1 + stalled);
                route = npu_wait + cost_.predict_npu(r->st.n_prefill) < cpu_cost ? "split" : "cpu";
            } else {
                route = r->st.n_prefill > params_.cpu_prefill_max ? "split" : "cpu";
            }
        }
        r->st.route       = route;
        r->st.prefill_dev = route == "cpu" ? "CPU" : params_.prefill_dev;
        r->st.decode_dev  = route == "npu" ? params_.prefill_dev : "CPU";
        if (route == "cpu") cpu_inbox_.push_back(r); else npu_queue_.push_back(r);
    }
    q_cv_.notify_all();
    return r;
}

gen_stats engine::chat(const std::vector<chat_msg> & msgs, const gen_params & gp, const std::function<bool(const std::string &)> & on_piece) {
    request_ptr r = submit(msgs, gp);
    for (;;) {
        std::unique_lock<std::mutex> lk(r->mtx);
        r->cv.wait(lk, [&] { return !r->pieces.empty() || r->done; });
        while (!r->pieces.empty()) {
            std::string p = std::move(r->pieces.front());
            r->pieces.pop_front();
            lk.unlock();
            if (!on_piece(p)) r->cancelled = true;
            lk.lock();
        }
        if (r->done && r->pieces.empty()) break;
    }
    std::lock_guard<std::mutex> lk(r->mtx);
    return r->st;
}

bool engine::should_stop(const request_ptr & r) const {
    if (r->cancelled) { r->st.stop_reason = "client"; return true; }
    if (llama_vocab_is_eog(vocab_, r->cur)) { r->st.stop_reason = "stop"; return true; }
    if (r->st.n_gen >= r->gp.max_tokens) { r->st.stop_reason = "length"; return true; }
    return false;
}

// stream r->cur as text, holding back an incomplete UTF-8 tail
void engine::emit(const request_ptr & r) {
    char buf[256];
    int n = llama_token_to_piece(vocab_, r->cur, buf, sizeof(buf), 0, false);
    if (n > 0) r->pending.append(buf, n);
    size_t ok = utf8_complete(r->pending);
    if (ok > 0) { r->push(r->pending.substr(0, ok)); r->pending.erase(0, ok); }
    r->st.n_gen++;
}

// an NPU-prefilled sequence waits in the CPU queue behind a full CPU lane: the idle NPU can decode it meanwhile
// (call with q_mtx_ held) index of the oldest such sequence that will not get a CPU slot now, or -1
int engine::steal_index() const {
    if (params_.npu_decode_slots <= 0 || !split()) return -1;
    for (int i = std::max(0, params_.max_seqs - n_active_); i < (int) cpu_inbox_.size(); i++) {
        if (!cpu_inbox_[i]->blob.empty()) return i;
    }
    return -1;
}

// decode r on the NPU context (seq 0). As an overflow slot it yields as soon as the CPU lane has room for it
// or a prompt waits for prefill: the sequence state goes to the front of the CPU queue, pos and sampler intact.
// Returns false if the sequence moved to the CPU lane.
bool engine::npu_decode(const request_ptr & r, bool overflow) {
    if (!r->decoding) {
        r->decoding = true;
        r->pos = (int) r->toks.size();
        r->st.ttft_ms = now_ms() - r->st.t_submit_ms;
        r->t_decode0 = now_ms();
        r->st.t_decode_start_ms = r->t_decode0;
    }
    { std::lock_guard<std::mutex> lk(live_mtx_); npu_phase_ = "decode #" + std::to_string(r->st.id); }
    set_live(r, "NPU", "decode");
    const double t0 = now_ms();
    llama_batch b = llama_batch_init(1, 0, 1);
    bool yield = false;
    while (!should_stop(r)) {
        if (overflow) {
            std::lock_guard<std::mutex> lk(q_mtx_);
            yield = !npu_queue_.empty() || n_active_ + (int) cpu_inbox_.size() < params_.max_seqs;
            if (yield) break;
        }
        emit(r);
        r->st.n_gen_npu++;
        b.n_tokens = 1; b.token[0] = r->cur; b.pos[0] = r->pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        const double t_dec = now_ms();
        if (llama_decode(cp_, b) != 0) { r->st.stop_reason = "decode_error"; break; }
        r->st.decode_llama_ms += now_ms() - t_dec;
        r->cur = llama_sampler_sample(r->smpl, cp_, -1);
    }
    llama_batch_free(b);
    r->st.npu_decode_ms += now_ms() - t0;
    if (yield) {
        clear_live(r->st.id);
        r->blob.resize(llama_state_seq_get_size(cp_, 0));
        llama_state_seq_get_data(cp_, r->blob.data(), r->blob.size(), 0);
        {
            std::lock_guard<std::mutex> lk(q_mtx_);
            cpu_inbox_.push_front(r);
        }
        q_cv_.notify_all();
        return false;
    }
    if (!r->pending.empty()) { r->push(r->pending); r->pending.clear(); }
    r->st.decode_ms = now_ms() - r->t_decode0;
    r->st.batch_peak = std::max(r->st.batch_peak, 1);
    record(r);
    r->finish(r->st.stop_reason);
    return true;
}

// NPU decode slot: take a prefilled sequence the CPU lane has no slot for and decode it here until the CPU has room
void engine::npu_overflow(const request_ptr & r) {
    keepwarm_pause(true);
    const double t0 = now_ms();
    llama_memory_clear(llama_get_memory(cp_), true);
    if (llama_state_seq_set_data(cp_, r->blob.data(), r->blob.size(), 0) == 0) {
        // cannot restore here; leave it to the CPU lane
        {
            std::lock_guard<std::mutex> lk(q_mtx_);
            cpu_inbox_.push_front(r);
        }
        q_cv_.notify_all();
    } else {
        r->st.handoff_ms += now_ms() - t0;
        r->blob.clear();
        r->blob.shrink_to_fit();
        r->h_prompt.clear();          // decoded on the NPU: the CPU lane's MTP head cannot catch up with it
        r->h_prompt.shrink_to_fit();
        npu_decode(r, true);
    }
    keepwarm_pause(false);
    { std::lock_guard<std::mutex> lk(live_mtx_); npu_phase_ = "idle"; }
}

// NPU lane: prefill one request at a time, then hand its state to the CPU lane (or decode on the NPU for route "npu").
// With nothing to prefill it can serve as an extra decode slot (npu_decode_slots).
void engine::npu_worker() {
    for (;;) {
        request_ptr r;
        bool steal = false;
        {
            std::unique_lock<std::mutex> lk(q_mtx_);
            q_cv_.wait(lk, [&] { return stop_ || !npu_queue_.empty() || steal_index() >= 0; });
            if (stop_) return;
            if (!npu_queue_.empty()) {
                r = npu_queue_.front();
                npu_queue_.pop_front();
                npu_busy_ = true;
            } else {
                // the oldest sequence still waiting for a CPU slot
                const int i = steal_index();
                r = cpu_inbox_[i];
                cpu_inbox_.erase(cpu_inbox_.begin() + i);
                steal = true;
            }
        }
        if (steal) { npu_overflow(r); continue; }
        if (r->cancelled) { r->finish("client"); record(r); npu_busy_ = false; continue; }

        keepwarm_pause(true);
        { std::lock_guard<std::mutex> lk(live_mtx_); npu_phase_ = "prefill #" + std::to_string(r->st.id); }
        set_live(r, "NPU", "prefill");
        r->st.t_start_ms = now_ms();
        r->st.queue_ms   = r->st.t_start_ms - r->st.t_submit_ms;
        llama_memory_clear(llama_get_memory(cp_), true);
        const bool restored = r->restore.empty() || llama_state_seq_set_data(cp_, r->restore.data(), r->restore.size(), 0) > 0;
        if (!restored || !prefill(cp_, r->toks, r->st.n_cached, 0,spec_ && r->st.route != "npu" ? &r->h_prompt : nullptr)) {
            llama_memory_clear(llama_get_memory(cp_), true);
            keepwarm_pause(false);
            npu_busy_ = false;
            { std::lock_guard<std::mutex> lk(live_mtx_); npu_phase_ = "idle"; }
            r->st.stop_reason = "prefill_error";
            record(r);
            r->finish("prefill_error");
            continue;
        }
        r->cur = llama_sampler_sample(r->smpl, cp_, -1);
        r->st.prefill_ms = now_ms() - r->st.t_start_ms;
        learn(true, r->st.n_prefill, r->st.prefill_ms);

        // snapshot = handoff blob; it also becomes the prefix-cache entry for this prompt
        r->blob.resize(llama_state_seq_get_size(cp_, 0));
        llama_state_seq_get_data(cp_, r->blob.data(), r->blob.size(), 0);
        cache_insert(r->toks, r->blob);
        r->restore.clear();
        r->restore.shrink_to_fit();

        if (r->st.route == "npu") {
            // comparison mode: decode on the NPU too
            r->blob.clear();
            r->blob.shrink_to_fit();
            npu_decode(r, false);
        } else {
            clear_live(r->st.id);
            {
                std::lock_guard<std::mutex> lk(q_mtx_);
                cpu_inbox_.push_back(r);
            }
            q_cv_.notify_all();
        }
        keepwarm_pause(false);
        npu_busy_ = false;
        { std::lock_guard<std::mutex> lk(live_mtx_); npu_phase_ = "idle"; }
    }
}

// --- speculative decoding with the MTP head (CPU lane) ---
//
// Per step, every decoding slot drafts k tokens with the model's MTP head (all slots batched in each MTP pass), then
// the target model verifies cur + k drafts of every slot in ONE decode call (all slots have the same k + 1 tokens, so
// the hybrid memory keeps them in one ubatch; a slot that does not draft pads with copies of cur). Each slot keeps the
// longest drafted prefix that matches what its own sampler draws from the target logits at those positions
// (llama.cpp common's sample-and-match rule; greedy: identical text, sampled: the same distribution), and rolls its
// recurrent state back over the rest (n_rs_seq snapshots per slot, pending rollback index per sequence).

void engine::mtp_trim(const request_ptr & r, int slot) {
    if (r->mtp_tok.size() <= mtp_window()) return;
    const size_t drop = r->mtp_tok.size() - mtp_window();
    llama_memory_seq_rm(llama_get_memory(cm_), slot, r->mtp_pos0, -1);  // drafted cells of an earlier step
    r->mtp_tok.erase(r->mtp_tok.begin(), r->mtp_tok.begin() + drop);
    r->mtp_h.erase(r->mtp_h.begin(), r->mtp_h.begin() + drop * n_embd_);
    r->mtp_pos0 += (int) drop;
}

// the prompt hidden states of a freshly admitted request become its first MTP catch-up
void engine::mtp_setup(const request_ptr & r, int slot) {
    llama_memory_seq_rm(llama_get_memory(cm_), slot, -1, -1);
    r->mtp_ok = false;
    r->draft.clear();
    r->mtp_tok.clear();
    r->mtp_h.clear();
    const int n = (int) r->toks.size(), c = r->st.n_cached;
    const size_t rows = (size_t) (n - c);
    // h_prompt holds the hidden rows of the last `have` prefilled tokens (prefill skips chunks outside the MTP window)
    const size_t have = r->h_prompt.size() / n_embd_;
    // a sequence that already decoded on the NPU has positions the MTP head never saw: it does not draft
    if (!r->decoding && have > 0 && have <= rows && r->h_prompt.size() == have * n_embd_) {
        r->mtp_pos0 = n - (int) have;
        r->mtp_tok.assign(r->toks.end() - have, r->toks.end());
        r->mtp_h.assign(n_embd_, 0.0f);    // the token at mtp_pos0 pairs with the hidden state before it: unknown here
        r->mtp_h.insert(r->mtp_h.end(), r->h_prompt.begin(), r->h_prompt.end() - n_embd_);
        r->mtp_hlast.assign(r->h_prompt.end() - n_embd_, r->h_prompt.end());
        r->mtp_ok = true;
        mtp_trim(r, slot);
    }
    r->h_prompt.clear();
    r->h_prompt.shrink_to_fit();
}

static int argmax_draft(const float * lg, int n_vocab, int n_head) {
    int best = 0;
    float bv = lg[0];
    const int n0 = n_head > 0 ? n_head : n_vocab;
    for (int i = 1; i < n0; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
    if (n_head > 0) for (int i = std::max(n0, n_vocab - 256); i < n_vocab; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
    return best;
}

// Feeds every drafting slot's pending tokens into the MTP context, then drafts k_slot[s] tokens per slot: the first
// MTP pass carries the catch-up plus each slot's draft start (cur at pos, paired with the target hidden state of pos - 1),
// every further pass one token per slot. Fills r->draft; false if an MTP decode failed (no drafts this step).
bool engine::mtp_draft(std::vector<request_ptr> & slots, const std::vector<int> & in_batch, const std::vector<int> & k_slot) {
    struct ent { llama_token tok; llama_pos pos; int s; const float * h; };
    std::vector<ent> pend, starts;
    llama_memory_t mm = llama_get_memory(cm_);
    for (int s : in_batch) {
        request_ptr & r = slots[s];
        r->mtp_fed = false;
        r->draft.clear();
        if (!r->mtp_ok || k_slot[s] <= 0) continue;
        llama_memory_seq_rm(mm, s, r->mtp_pos0, -1);   // drafted positions of the previous step
        for (size_t i = 0; i < r->mtp_tok.size(); i++) pend.push_back({ r->mtp_tok[i], r->mtp_pos0 + (llama_pos) i, s, r->mtp_h.data() + i * n_embd_ });
        starts.push_back({ r->cur, r->pos, s, r->mtp_hlast.data() });
    }
    if (starts.empty()) return true;

    llama_batch_ext * mb = llama_batch_ext_init(cm_);
    const int nb = (int) llama_n_batch(cm_);
    auto add = [&](const ent & e, bool out) {
        const int32_t idx = llama_batch_ext_add_token(mb, e.s, e.tok);
        llama_batch_ext_set_pos(mb, idx, &e.pos);
        llama_batch_ext_set_embd_state(mb, idx, { e.h, 1, (size_t) n_embd_ });
        if (out) llama_batch_ext_set_output_logits(mb, idx, true);
        return idx;
    };
    bool ok = true;
    // long catch-ups (a new prompt) go first in n_batch chunks; the last call also carries the draft starts
    size_t i0 = 0;
    while (ok && pend.size() - i0 + starts.size() > (size_t) nb) {
        const size_t n = std::min((size_t) nb, pend.size() - i0);
        llama_batch_ext_clear(mb);
        for (size_t i = i0; i < i0 + n; i++) add(pend[i], false);
        ok = llama_process(cm_, LLAMA_PROCESS_TYPE_DECODE, mb) == 0;
        i0 += n;
    }
    std::vector<int32_t> out(starts.size());
    if (ok) {
        llama_batch_ext_clear(mb);
        for (size_t i = i0; i < pend.size(); i++) add(pend[i], false);
        for (size_t e = 0; e < starts.size(); e++) out[e] = add(starts[e], true);
        ok = llama_process(cm_, LLAMA_PROCESS_TYPE_DECODE, mb) == 0;
    }
    if (ok) {
        for (const ent & e : starts) {
            request_ptr & r = slots[e.s];
            r->mtp_fed = true;              // committed through pos; the drafted positions after it are dropped next step
            r->mtp_tok.clear();
            r->mtp_h.clear();
            r->mtp_pos0 = r->pos + 1;
        }
    }
    // draft passes: one token per slot that still needs one, chained on the MTP head's own hidden state
    std::vector<float> hbuf(starts.size() * n_embd_);
    while (ok) {
        std::vector<size_t> next;
        for (size_t e = 0; e < starts.size(); e++) {
            if (out[e] < 0) continue;
            request_ptr & r = slots[starts[e].s];
            const float * lg = llama_get_logits_ith(cm_, out[e]);
            const float * h  = llama_get_embeddings_nextn_ith(cm_, out[e]);
            if (!lg || !h) { ok = false; break; }
            r->draft.push_back(argmax_draft(lg, n_vocab_, draft_head_));
            if ((int) r->draft.size() < k_slot[starts[e].s]) {
                std::memcpy(hbuf.data() + e * n_embd_, h, sizeof(float) * n_embd_);
                next.push_back(e);
            }
            out[e] = -1;
        }
        if (!ok || next.empty()) break;
        llama_batch_ext_clear(mb);
        for (size_t e : next) {
            request_ptr & r = slots[starts[e].s];
            out[e] = add({ r->draft.back(), (llama_pos) (r->pos + r->draft.size()), starts[e].s, hbuf.data() + e * n_embd_ }, true);
        }
        ok = llama_process(cm_, LLAMA_PROCESS_TYPE_DECODE, mb) == 0;
    }
    llama_batch_ext_free(mb);
    if (!ok) {
        fprintf(stderr, "[trilane] MTP draft decode failed; this step decodes without drafts\n");
        for (const ent & e : starts) slots[e.s]->draft.clear();
    }
    return ok;
}

void spec_tuner::init(int k_max, int slots, const std::vector<int> & guess) {
    K = k_max; S = slots;
    t_ms.assign((size_t) (S + 1) * (K + 1), -1.0);
    n_obs.assign(t_ms.size(), 0);
    acc.assign(K + 1, 0.0);
    tried.assign(K + 1, 0.0);
    steps.assign(S + 1, 0);
    cur.assign(S + 1, 0);
    for (int a = 1; a <= S; a++) cur[a] = guess.empty() ? K : guess[std::min((size_t) a, guess.size()) - 1];
}

double spec_tuner::expected(int k) const {
    double e = 1.0, run = 1.0;
    for (int i = 1; i <= k; i++) {
        run *= (acc[i] + 1.0) / (tried[i] + 2.0);
        e += run;
    }
    return e;
}

int spec_tuner::pick(int a) {
    a = std::max(1, std::min(a, S));
    const int base = a * (K + 1);
    steps[a]++;
    // hill climb: measure the current length and its two neighbours, move to the fastest of the three
    for (int k : { cur[a], cur[a] - 1, cur[a] + 1 }) {
        if (k >= 0 && k <= K && n_obs[base + k] < 3) return k;
    }
    int best = cur[a];
    double best_tps = -1;
    for (int k = std::max(0, cur[a] - 1); k <= std::min(K, cur[a] + 1); k++) {
        const double tps = a * expected(k) / t_ms[base + k];
        if (tps > best_tps) { best_tps = tps; best = k; }
    }
    cur[a] = best;
    if (steps[a] % 128 == 0) {  // probe a neighbour now and then so its step time stays current
        int p = best + (((steps[a] / 128) & 1) ? 1 : -1);
        if (p < 0) p = best + 1;
        if (p > K) p = best - 1;
        return std::max(0, std::min(K, p));
    }
    return best;
}

void spec_tuner::observe(int a, int k, double ms) {
    a = std::max(1, std::min(a, S));
    const int i = a * (K + 1) + k;
    if (n_obs[i] >= 3 && ms > 3.0 * t_ms[i]) return;   // a stall (admission, CPU prefill, clock dip), not the step
    t_ms[i] = t_ms[i] < 0 ? ms : 0.85 * t_ms[i] + 0.15 * ms;
    n_obs[i]++;
}

void spec_tuner::dump() const {
    fprintf(stderr, "[spec-tuner] p:");
    for (int i = 1; i <= K; i++) fprintf(stderr, " %.2f", (acc[i] + 1.0) / (tried[i] + 2.0));
    for (int a = 1; a <= S; a++) {
        if (!steps[a]) continue;
        fprintf(stderr, " | %d slots: k=%d, ms", a, cur[a]);
        for (int k = 0; k <= K; k++) fprintf(stderr, " %.1f", t_ms[a * (K + 1) + k]);
    }
    fprintf(stderr, "\n");
}

void spec_tuner::observe_slot(int n_drafted, int n_accepted) {
    for (int i = 1; i <= std::min(n_drafted, K); i++) {
        if (i > n_accepted + 1) break;
        tried[i] = 0.998 * tried[i] + 1.0;
        acc[i]   = 0.998 * acc[i] + (i <= n_accepted ? 1.0 : 0.0);
    }
}

// CPU lane: continuous batching. Admit prefilled requests into free slots between steps, then decode
// one token for every active slot in a single batch (with speculative decoding: cur + k drafts per slot).
void engine::cpu_worker() {
    if (tp_decode_) decode_thread_like_pool(params_.prio); // per-step pools (--no-threadpool): unchanged
    const int S = params_.max_seqs;
    const int K = spec_ ? params_.spec_nmax : 0;
    std::vector<request_ptr> slots(S);
    llama_batch b = llama_batch_init(S * (1 + K), 0, 1);
    llama_memory_t mem = llama_get_memory(cd_);
    llama_memory_t mem_mtp = cm_ ? llama_get_memory(cm_) : nullptr;

    auto finish_slot = [&](int s) {
        request_ptr r = slots[s];
        if (!r->pending.empty()) { r->push(r->pending); r->pending.clear(); }
        r->st.decode_ms = now_ms() - r->t_decode0;
        llama_memory_seq_rm(mem, s, -1, -1);
        if (mem_mtp) llama_memory_seq_rm(mem_mtp, s, -1, -1);
        if (spec_auto_ && getenv("TRILANE_SPEC_DEBUG")) tuner_.dump();
        slots[s].reset();
        n_active_--;
        record(r);
        r->finish(r->st.stop_reason);
    };

    for (;;) {
        // admit
        for (;;) {
            int s = -1;
            for (int i = 0; i < S; i++) if (!slots[i]) { s = i; break; }
            request_ptr r;
            {
                std::unique_lock<std::mutex> lk(q_mtx_);
                if (n_active_ == 0) q_cv_.wait(lk, [&] { return stop_ || !cpu_inbox_.empty(); });
                if (stop_) { llama_batch_free(b); return; }
                if (s < 0 || cpu_inbox_.empty()) break;
                r = cpu_inbox_.front();
                cpu_inbox_.pop_front();
            }
            if (r->cancelled) { r->finish("client"); record(r); continue; }
            llama_memory_seq_rm(mem, s, -1, -1);
            const double t0 = now_ms();
            if (!r->blob.empty()) {
                // prefilled on the NPU: restore its state into this slot
                if (llama_state_seq_set_data(cd_, r->blob.data(), r->blob.size(), s) == 0) {
                    llama_memory_seq_rm(mem, s, -1, -1);
                    r->st.stop_reason = "handoff_error";
                    record(r);
                    r->finish("handoff_error");
                    continue;
                }
                r->st.handoff_mb = r->blob.size() / 1048576.0;
                r->st.handoff_ms += now_ms() - t0;
                r->blob.clear();
                r->blob.shrink_to_fit();
            } else {
                // short prompt: prefill on the CPU lane (the other slots pause for this one call)
                { std::lock_guard<std::mutex> lk(live_mtx_); cpu_phase_ = "prefill #" + std::to_string(r->st.id); }
                set_live(r, "CPU" + std::to_string(s), "prefill");
                r->st.t_start_ms = t0;
                r->st.queue_ms   = t0 - r->st.t_submit_ms;
                const bool restored = r->restore.empty() || llama_state_seq_set_data(cd_, r->restore.data(), r->restore.size(), s) > 0;
                if (!restored || !prefill(cd_, r->toks, restored ? r->st.n_cached : 0, s, spec_ ? &r->h_prompt : nullptr)) {
                    llama_memory_seq_rm(mem, s, -1, -1);
                    r->st.stop_reason = "prefill_error";
                    record(r);
                    r->finish("prefill_error");
                    continue;
                }
                r->cur = llama_sampler_sample(r->smpl, cd_, -1);
                r->st.prefill_ms = now_ms() - t0;
                learn(false, r->st.n_prefill, r->st.prefill_ms);
                std::vector<uint8_t> snap(llama_state_seq_get_size(cd_, s));
                llama_state_seq_get_data(cd_, snap.data(), snap.size(), s);
                cache_insert(r->toks, snap);
                r->restore.clear();
                r->restore.shrink_to_fit();
            }
            if (spec_) mtp_setup(r, s);
            r->st.slot = s;
            r->st.t_cpu_join_ms = now_ms();
            set_live(r, "CPU" + std::to_string(s), "decode");
            if (!r->decoding) {
                r->decoding = true;
                r->pos = (int) r->toks.size();
                r->st.ttft_ms = now_ms() - r->st.t_submit_ms;
                r->t_decode0 = now_ms();
                r->st.t_decode_start_ms = r->t_decode0;
            }
            slots[s] = r;
            n_active_++;
            if (params_.npu_decode_slots > 0) q_cv_.notify_all(); // a fuller CPU lane may leave work for the NPU
        }

        // step: emit each slot's pending token, stop finished ones, decode the rest together
        b.n_tokens = 0;
        std::vector<int> in_batch;
        for (int s = 0; s < S; s++) {
            request_ptr & r = slots[s];
            if (!r) continue;
            if (should_stop(r)) { finish_slot(s); continue; }
            emit(r);
            in_batch.push_back(s);
        }
        if (in_batch.empty()) continue;

        // drafts: k tokens per slot (0 = plain step), capped by what the request may still emit
        int k = 0;
        std::vector<int> k_slot(S, 0);
        const double t_step0 = now_ms();
        bool fresh = false;   // a slot's first step carries its prompt catch-up: not a typical step time
        for (int s : in_batch) fresh = fresh || slots[s]->st.n_steps == 0;
        const int k_step = !spec_ ? 0 : spec_auto_ ? tuner_.pick((int) in_batch.size())
                                                   : params_.spec_k[std::min(in_batch.size(), params_.spec_k.size()) - 1];
        if (k_step > 0) {
            // every slot of the step carries 1 + k tokens (non-drafting slots pad with cur), and each slot's KV stream
            // holds n_ctx cells: a slot near the end of its context caps the step's k for all slots
            int k_cap = k_step;
            for (int s : in_batch) k_cap = std::min(k_cap, params_.n_ctx - 1 - slots[s]->pos);
            for (int s : in_batch) {
                request_ptr & r = slots[s];
                if (!r->mtp_ok) continue;
                const int left = std::min(r->gp.max_tokens - r->st.n_gen, k_cap);
                k_slot[s] = std::max(0, std::min(k_step, left));
                k = std::max(k, k_slot[s]);
            }
        }
        if (spec_) {
            for (int s : in_batch) { slots[s]->draft.clear(); slots[s]->mtp_fed = false; }
            if (k > 0) {
                { std::lock_guard<std::mutex> lk(live_mtx_); cpu_phase_ = "draft x" + std::to_string(in_batch.size()); }
                if (!mtp_draft(slots, in_batch, k_slot)) k = 0;
            }
        }

        std::vector<int> base(S, -1);
        for (int s : in_batch) {
            request_ptr & r = slots[s];
            base[s] = b.n_tokens;
            const int nd = (int) r->draft.size();
            for (int i = 0; i <= k; i++) {
                const int j = b.n_tokens++;
                // a slot with fewer drafts pads with cur: equal token counts keep all slots in one ubatch
                b.token[j] = i == 0 ? r->cur : i <= nd ? r->draft[i - 1] : r->cur;
                b.pos[j] = r->pos + i; b.n_seq_id[j] = 1; b.seq_id[j][0] = s; b.logits[j] = 1;
            }
        }
        {
            std::lock_guard<std::mutex> lk(live_mtx_);
            cpu_phase_ = "decode x" + std::to_string(in_batch.size()) + (k > 0 ? " (+" + std::to_string(k) + " drafted each)" : "");
        }
        const bool more_threads = k > 0 && params_.threads_batch != params_.threads_decode;
        if (more_threads) llama_set_n_threads(cd_, params_.threads_decode, params_.threads_batch);
        const double t_dec = now_ms();
        const int rc = llama_decode(cd_, b);
        if (more_threads) llama_set_n_threads(cd_, params_.threads_decode, params_.threads_decode);
        if (rc != 0) {
            for (int s : in_batch) { slots[s]->st.stop_reason = "decode_error"; finish_slot(s); }
            continue;
        }
        const double dt_dec = now_ms() - t_dec;
        for (int s : in_batch) slots[s]->st.decode_llama_ms += dt_dec;
        if (!spec_) {
            for (int i = 0; i < (int) in_batch.size(); i++) {
                request_ptr & r = slots[in_batch[i]];
                r->pos++;
                r->cur = llama_sampler_sample(r->smpl, cd_, i);
                r->st.batch_peak = std::max(r->st.batch_peak, (int) in_batch.size());
            }
        } else {
            const float * H = llama_get_embeddings_nextn(cd_);   // target hidden state of every batch token
            for (int s : in_batch) {
                request_ptr & r = slots[s];
                r->st.batch_peak = std::max(r->st.batch_peak, (int) in_batch.size());
                r->st.n_steps++;
                const int nd = (int) r->draft.size(), b0 = base[s];
                // longest drafted prefix the slot's own sampler reproduces from the target logits
                int j = 0;
                llama_token t = llama_sampler_sample(r->smpl, cd_, b0);
                while (j < nd && t == r->draft[j] && !llama_vocab_is_eog(vocab_, t)) {
                    j++;
                    t = llama_sampler_sample(r->smpl, cd_, b0 + j);
                }
                r->st.n_drafted  += nd;
                r->st.n_accepted += j;
                if (spec_auto_ && nd > 0) tuner_.observe_slot(nd, j);
                const llama_token cur_old = r->cur;
                const int pos_old = r->pos;
                if (r->mtp_ok) {
                    if (!H) {
                        r->mtp_ok = false;
                    } else {
                        if (!r->mtp_fed) {   // cur was not fed to the MTP head this step: it waits with the rest
                            r->mtp_tok.push_back(cur_old);
                            r->mtp_h.insert(r->mtp_h.end(), r->mtp_hlast.begin(), r->mtp_hlast.end());
                        }
                        for (int i = 0; i < j; i++) {
                            r->mtp_tok.push_back(r->draft[i]);
                            r->mtp_h.insert(r->mtp_h.end(), H + (size_t) (b0 + i) * n_embd_, H + (size_t) (b0 + i + 1) * n_embd_);
                        }
                        mtp_trim(r, s);
                        r->mtp_hlast.assign(H + (size_t) (b0 + j) * n_embd_, H + (size_t) (b0 + j + 1) * n_embd_);
                    }
                }
                r->pos = pos_old + 1 + j;
                // drop the rejected drafts and pads: bounded recurrent-state rollback (at most k positions)
                if (j < k && !llama_memory_seq_rm(mem, s, r->pos, -1)) {
                    fprintf(stderr, "[trilane] rollback failed in slot %d\n", s);
                    r->st.stop_reason = "decode_error";
                    finish_slot(s);
                    continue;
                }
                bool stopped = false;
                for (int i = 0; i < j; i++) {
                    r->cur = r->draft[i];
                    if (should_stop(r)) { finish_slot(s); stopped = true; break; }
                    emit(r);
                }
                if (stopped) continue;
                r->cur = t;
                r->draft.clear();
            }
            if (spec_auto_ && k == k_step && !fresh) tuner_.observe((int) in_batch.size(), k, now_ms() - t_step0);
        }
        if (n_active_ == 0) { std::lock_guard<std::mutex> lk(live_mtx_); cpu_phase_ = "idle"; }
    }
}

// Keeps the NPU out of its idle low-power state: a 4 KB ADD on the prefill device every keepwarm_ms.
// Without it, a prefill after >1 s of NPU idle runs ~2.5x slower on big models.
void engine::keepwarm_start() {
    // setup happens here on the caller's thread, before any prefill can start: backend init and the
    // buffer upload touch the same (not thread-safe) HTP session as the prefill context
    std::string dev_name = params_.prefill_dev.substr(0, params_.prefill_dev.find(','));
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    ggml_backend_t be = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (!be) return;
    ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_tensor * y = ggml_add(ctx, x, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float> ones(1024, 1.0f);
    ggml_backend_tensor_set(x, ones.data(), 0, sizeof(float) * ones.size());

    kw_thread_ = std::thread([this, be, ctx, gf, buf]() {
        while (!kw_stop_) {
            {
                std::lock_guard<std::mutex> lk(kw_mtx_);
                if (!kw_paused_) { ggml_backend_graph_compute(be, gf); pings_++; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(params_.keepwarm_ms));
        }
        std::lock_guard<std::mutex> lk(kw_mtx_);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
    });
}

// the HTP session is not thread-safe, so pausing waits for an in-flight ping
void engine::keepwarm_pause(bool p) { std::lock_guard<std::mutex> lk(kw_mtx_); kw_paused_ = p; }

void engine::keepwarm_stop() {
    kw_stop_ = true;
    if (kw_thread_.joinable()) kw_thread_.join();
}

}  // namespace trilane
