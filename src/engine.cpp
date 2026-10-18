#include "engine.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace trilane {

// CPU_HTPTILE hooks exported by our ggml-cpu fork (see ggml/src/ggml-cpu/htp-tile.cpp)
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

engine::~engine() {
    keepwarm_stop();
    if (cd_ && cd_ != cp_) llama_free(cd_);
    if (md_ && md_ != mp_) llama_model_free(md_);
    if (cp_) llama_free(cp_);
    if (mp_) llama_model_free(mp_);
}

double engine::now_ms() const { return llama_time_us() / 1000.0; }

static llama_model_params base_mparams() {
    llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_NONE; // read once; mmap + CPU repack keeps two copies
    return mp;
}

bool engine::load(const engine_params & p) {
    params_ = p;
    static std::vector<ggml_backend_dev_t> npu_devs, no_devs;
    static llama_model_tensor_buft_override overrides[2] = {};

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
        mp_ = llama_model_load_from_file(p.model.c_str(), mp);
        if (!mp_) return false;
        fprintf(stderr, "[trilane] prefill model on %s loaded in %.0f ms\n", p.prefill_dev.c_str(), now_ms() - t0);
    }

    no_devs = { nullptr };
    llama_model_params dp = base_mparams();
    dp.devices = no_devs.data();
    dp.n_gpu_layers = 0;
    if (use_npu && p.alias_experts && has_experts(mp_)) {
        ggml_backend_reg_t cpu = ggml_backend_reg_by_name("CPU");
        auto set_resolver = (htptile_set_resolver_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_htptile_set_resolver");
        auto get_buft     = (htptile_buft_t) ggml_backend_reg_get_proc_address(cpu, "ggml_backend_cpu_htptile_buffer_type");
        if (set_resolver && get_buft) {
            set_resolver(resolve_from_model, (void *) mp_);
            overrides[0] = { "ffn_(gate|up|down)_exps", get_buft() };
            overrides[1] = { nullptr, nullptr };
            dp.tensor_buft_overrides = overrides;
            dp.load_mode = LLAMA_LOAD_MODE_MMAP;          // aliased experts are never read from the file
            _putenv_s("LLAMA_MMAP_NO_PREFETCH", "1");     // nor prefetched into RAM
            aliased_ = true;
        }
    }
    {
        double t0 = now_ms();
        md_ = llama_model_load_from_file(p.model.c_str(), dp);
        if (!md_) return false;
        fprintf(stderr, "[trilane] decode model on CPU%s loaded in %.0f ms\n", aliased_ ? " (experts aliased to NPU memory)" : "", now_ms() - t0);
    }
    if (!mp_) mp_ = md_;

    auto make_ctx = [&](llama_model * m, int threads, bool host_only) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = p.n_ctx;
        cp.n_batch         = p.n_batch;
        cp.n_ubatch        = p.n_ubatch;
        cp.n_seq_max       = 1;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        cp.no_perf         = true;
        cp.op_offload      = !host_only;
        return llama_init_from_model(m, cp);
    };
    cd_ = make_ctx(md_, p.threads_decode, true);
    cp_ = mp_ != md_ ? make_ctx(mp_, p.threads_prefill, false) : cd_;
    if (!cd_ || !cp_) return false;

    vocab_ = llama_model_get_vocab(mp_);
    char desc[256];
    llama_model_desc(mp_, desc, sizeof(desc));
    model_name_ = p.model.substr(p.model.find_last_of("/\\") + 1);
    const char * t = llama_model_chat_template(mp_, nullptr);
    tmpl_ = t ? t : "";
    fprintf(stderr, "[trilane] %s (%s), split=%d aliased=%d\n", model_name_.c_str(), desc, split(), aliased_);

    if (split() && p.keepwarm_ms > 0) keepwarm_start();
    return true;
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

// prefill toks[from..] at positions from.., logits for the last token only
void engine::prefill(llama_context * c, const std::vector<llama_token> & toks, int from) {
    const int nb = params_.n_batch;
    llama_batch b = llama_batch_init(nb, 0, 1);
    for (int i0 = from; i0 < (int) toks.size(); i0 += nb) {
        const int n = std::min(nb, (int) toks.size() - i0);
        b.n_tokens = n;
        for (int j = 0; j < n; j++) {
            b.token[j] = toks[i0 + j]; b.pos[j] = i0 + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0;
            b.logits[j] = (i0 + j == (int) toks.size() - 1);
        }
        if (llama_decode(c, b) != 0) fprintf(stderr, "[trilane] prefill decode failed\n");
    }
    llama_synchronize(c);
    llama_batch_free(b);
}

const engine::cache_entry * engine::cache_lookup(const std::vector<llama_token> & toks) {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    auto best = cache_.end();
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->toks.size() < toks.size() && std::equal(it->toks.begin(), it->toks.end(), toks.begin()) &&
            (best == cache_.end() || it->toks.size() > best->toks.size())) {
            best = it;
        }
    }
    if (best == cache_.end()) return nullptr;
    cache_.splice(cache_.begin(), cache_, best);
    return &cache_.front();
}

void engine::cache_insert(const std::vector<llama_token> & toks, std::vector<uint8_t> && state) {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        if (it->toks == toks) { cache_bytes_ -= it->state.size(); cache_.erase(it); break; }
    }
    cache_bytes_ += state.size();
    cache_.push_front({ toks, std::move(state) });
    while (cache_bytes_ > params_.prefix_cache_mb * 1048576ull && cache_.size() > 1) {
        cache_bytes_ -= cache_.back().state.size();
        cache_.pop_back();
    }
}

size_t engine::cache_entries() const { std::lock_guard<std::mutex> lk(cache_mtx_); return cache_.size(); }
size_t engine::cache_bytes() const { std::lock_guard<std::mutex> lk(cache_mtx_); return cache_bytes_; }

std::vector<gen_stats> engine::history() const { std::lock_guard<std::mutex> lk(hist_mtx_); return history_; }

void engine::set_phase(const std::string & p) { std::lock_guard<std::mutex> lk(live_mtx_); live_phase_ = p; }

gen_stats engine::chat(const std::vector<chat_msg> & msgs, const gen_params & gp, const std::function<bool(const std::string &)> & on_piece) {
    std::lock_guard<std::mutex> glk(gen_mtx_);
    gen_stats st;
    st.t_start_ms = now_ms();
    { std::lock_guard<std::mutex> lk(hist_mtx_); st.id = next_id_++; }

    const std::vector<llama_token> toks = tokenize(apply_template(msgs));
    st.n_prompt = (int) toks.size();
    if (toks.empty() || (int) toks.size() + gp.max_tokens > params_.n_ctx) {
        st.stop_reason = "context_overflow";
        return st;
    }

    // prefix cache: restore the longest cached exact prefix, prefill only the rest
    std::vector<uint8_t> restore;
    if (const cache_entry * e = cache_lookup(toks)) {
        st.n_cached = (int) e->toks.size();
        restore = e->state;
    }
    st.n_prefill = st.n_prompt - st.n_cached;

    // lane choice: long prefills on the NPU, short ones stay on the CPU (no NPU dispatch, no handoff)
    std::string route = gp.route;
    if (!split()) route = "cpu";
    else if (route == "auto") route = st.n_prefill > params_.cpu_prefill_max ? "split" : "cpu";
    st.route = route;
    llama_context * pre = route == "cpu" ? cd_ : cp_;
    llama_context * dec = route == "npu" ? cp_ : cd_;
    st.prefill_dev = pre == cp_ && split() ? params_.prefill_dev : "CPU";
    st.decode_dev  = dec == cp_ && split() ? params_.prefill_dev : "CPU";

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (gp.temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(64));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(gp.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(gp.min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(gp.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(gp.seed));
    }

    const bool uses_npu = pre == cp_ || dec == cp_;
    if (uses_npu && split()) keepwarm_pause(true);

    set_phase("prefill:" + st.prefill_dev);
    double t0 = now_ms();
    llama_memory_clear(llama_get_memory(pre), true);
    if (!restore.empty()) llama_state_seq_set_data(pre, restore.data(), restore.size(), 0);
    prefill(pre, toks, st.n_cached);
    llama_token tok = llama_sampler_sample(smpl, pre, -1);
    double t1 = now_ms();
    st.prefill_ms = t1 - t0;

    // snapshot = handoff blob; it also becomes the prefix-cache entry for this prompt
    std::vector<uint8_t> snap(llama_state_seq_get_size(pre, 0));
    llama_state_seq_get_data(pre, snap.data(), snap.size(), 0);
    if (dec != pre) {
        set_phase("handoff");
        llama_memory_clear(llama_get_memory(dec), true);
        llama_state_seq_set_data(dec, snap.data(), snap.size(), 0);
        st.handoff_mb = snap.size() / 1048576.0;
    }
    if (dec != cp_ && uses_npu && split()) keepwarm_pause(false);
    double t2 = now_ms();
    st.handoff_ms = t2 - t1;
    st.ttft_ms = t2 - t0;
    cache_insert(toks, std::move(snap));

    set_phase("decode:" + st.decode_dev);
    std::string pending;
    llama_batch b = llama_batch_init(1, 0, 1);
    int pos = (int) toks.size();
    st.stop_reason = "length";
    for (int i = 0; i < gp.max_tokens; i++) {
        if (llama_vocab_is_eog(vocab_, tok)) { st.stop_reason = "stop"; break; }
        char buf[256];
        int n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, false);
        if (n > 0) pending.append(buf, n);
        size_t ok = utf8_complete(pending);
        if (ok > 0) {
            if (!on_piece(pending.substr(0, ok))) { st.stop_reason = "client"; break; }
            pending.erase(0, ok);
        }
        st.n_gen++;
        if (i + 1 == gp.max_tokens) break;
        b.n_tokens = 1; b.token[0] = tok; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        if (llama_decode(dec, b) != 0) { st.stop_reason = "decode_error"; break; }
        tok = llama_sampler_sample(smpl, dec, -1);
    }
    if (!pending.empty()) on_piece(pending);
    llama_batch_free(b);
    llama_sampler_free(smpl);
    st.decode_ms = now_ms() - t2;
    if (dec == cp_ && split()) keepwarm_pause(false);
    set_phase("idle");

    {
        std::lock_guard<std::mutex> lk(hist_mtx_);
        history_.push_back(st);
        if (history_.size() > 200) history_.erase(history_.begin());
    }
    return st;
}

// Keeps the NPU out of its idle low-power state: a 4 KB ADD on the prefill device every keepwarm_ms.
// Without it, a prefill after >1 s of NPU idle runs ~2.5x slower on big models.
void engine::keepwarm_start() {
    std::string dev_name = params_.prefill_dev.substr(0, params_.prefill_dev.find(','));
    kw_thread_ = std::thread([this, dev_name]() {
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
        while (!kw_stop_) {
            {
                std::lock_guard<std::mutex> lk(kw_mtx_);
                if (!kw_paused_) { ggml_backend_graph_compute(be, gf); pings_++; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(params_.keepwarm_ms));
        }
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
