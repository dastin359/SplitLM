#include "engine.h"

#include "ggml-alloc.h"
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
// barrier (and run all single-thread nodes slower). TRILANE_WORKER0_PRIO=0 restores the old behaviour.
static void decode_thread_like_pool(int prio) {
#ifdef _WIN32
    const char * e = getenv("TRILANE_WORKER0_PRIO");
    if (e && atoi(e) == 0) return;
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
    auto make_ctx = [&](llama_model * m, int threads, bool host_only, int n_seq, bool prefill_only) {
        llama_context_params cp = llama_context_default_params();
        if (prefill_only) cp.n_outputs_max = (uint32_t) std::max(1, n_seq);
        cp.n_ctx           = params_.n_ctx * n_seq;
        cp.n_batch         = params_.n_batch;
        cp.n_ubatch        = params_.n_ubatch;
        cp.n_seq_max       = n_seq;
        cp.kv_unified      = true;   // slots share one cell pool
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        cp.no_perf         = true;
        cp.op_offload      = !host_only;
        return llama_init_from_model(m, cp);
    };
    cd_ = make_ctx(md_, params_.threads_decode, true, params_.max_seqs, false);
    cp_ = mp_ != md_ ? make_ctx(mp_, params_.threads_prefill, false, 1, true) : cd_;
    if (!cd_ || !cp_) return false;

    // Without an attached thread pool, ggml creates and joins a new pool of worker threads for every graph,
    // i.e. for every decode step. A persistent pool keeps them (polling between steps, like llama-bench).
    if (params_.threadpool) {
        ggml_backend_reg_t cpu = ggml_backend_reg_by_name("CPU");
        auto tp_new = cpu ? (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu, "ggml_threadpool_new") : nullptr;
        if (tp_new) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(params_.threads_decode);
            tpp.prio = (enum ggml_sched_priority) params_.prio;
            tp_decode_ = tp_new(&tpp);
        }
        if (tp_decode_) llama_attach_threadpool(cd_, tp_decode_, nullptr);
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

// prefill toks[from..] at positions from.. into sequence seq, logits for the last token only; false on a failed decode
bool engine::prefill(llama_context * c, const std::vector<llama_token> & toks, int from, llama_seq_id seq) {
    const int nb = params_.n_batch;
    llama_batch b = llama_batch_init(nb, 0, 1);
    for (int i0 = from; i0 < (int) toks.size(); i0 += nb) {
        const int n = std::min(nb, (int) toks.size() - i0);
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
        if (!restored || !prefill(cp_, r->toks, restored ? r->st.n_cached : 0, 0)) {
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

// CPU lane: continuous batching. Admit prefilled requests into free slots between steps, then decode
// one token for every active slot in a single batch.
void engine::cpu_worker() {
    decode_thread_like_pool(tp_decode_ ? params_.prio : 0);
    const int S = params_.max_seqs;
    std::vector<request_ptr> slots(S);
    llama_batch b = llama_batch_init(S, 0, 1);
    llama_memory_t mem = llama_get_memory(cd_);

    auto finish_slot = [&](int s) {
        request_ptr r = slots[s];
        if (!r->pending.empty()) { r->push(r->pending); r->pending.clear(); }
        r->st.decode_ms = now_ms() - r->t_decode0;
        llama_memory_seq_rm(mem, s, -1, -1);
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
                if (!restored || !prefill(cd_, r->toks, restored ? r->st.n_cached : 0, s)) {
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
            const int i = b.n_tokens++;
            b.token[i] = r->cur; b.pos[i] = r->pos++; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = 1;
            in_batch.push_back(s);
        }
        if (b.n_tokens == 0) continue;
        {
            std::lock_guard<std::mutex> lk(live_mtx_);
            cpu_phase_ = "decode x" + std::to_string(b.n_tokens);
        }
        const double t_dec = now_ms();
        if (llama_decode(cd_, b) != 0) {
            for (int s : in_batch) { slots[s]->st.stop_reason = "decode_error"; finish_slot(s); }
            continue;
        }
        const double dt_dec = now_ms() - t_dec;
        for (int s : in_batch) slots[s]->st.decode_llama_ms += dt_dec;
        for (int i = 0; i < (int) in_batch.size(); i++) {
            request_ptr & r = slots[in_batch[i]];
            r->cur = llama_sampler_sample(r->smpl, cd_, i);
            r->st.batch_peak = std::max(r->st.batch_peak, (int) in_batch.size());
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
