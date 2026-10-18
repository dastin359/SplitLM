// Tri-Lane engine: phase-split LLM inference on Snapdragon X (prefill on the NPU, decode on the CPU).
#pragma once

#include "llama.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace trilane {

struct engine_params {
    std::string model;
    std::string prefill_dev = "HTP0";  // NPU device(s) for prefill; "CPU" disables the NPU lane
    int    threads_decode   = 8;
    int    threads_prefill  = 4;       // host threads of the prefill context
    int    n_ctx            = 8192;
    int    n_batch          = 2048;
    int    n_ubatch         = 512;
    int    keepwarm_ms      = 20;      // tiny NPU op cadence between prefills (0 = off)
    size_t prefix_cache_mb  = 2048;    // budget for cached sequence-state snapshots
    int    cpu_prefill_max  = 24;      // prompts with at most this many new tokens prefill on the CPU
    bool   alias_experts    = true;    // CPU decode reads the NPU's expert weights in place (MoE models)
};

struct gen_params {
    int      max_tokens  = 512;
    float    temperature = 0.7f;
    float    top_p       = 0.95f;
    float    min_p       = 0.05f;
    uint32_t seed        = 0xC0FFEE;
    std::string route    = "auto";     // auto | npu | cpu  (per-request lane override for comparisons)
};

struct gen_stats {
    int64_t     id          = 0;
    double      t_start_ms  = 0;       // engine clock
    int         n_prompt    = 0;       // prompt tokens
    int         n_cached    = 0;       // prompt tokens restored from the prefix cache
    int         n_prefill   = 0;       // tokens actually prefilled
    std::string route;                 // effective route
    std::string prefill_dev;
    std::string decode_dev;
    double      prefill_ms  = 0;
    double      handoff_ms  = 0;
    double      handoff_mb  = 0;
    double      ttft_ms     = 0;
    double      decode_ms   = 0;
    int         n_gen       = 0;
    std::string stop_reason;
};

struct chat_msg {
    std::string role;
    std::string content;
};

class engine {
  public:
    engine() = default;
    ~engine();

    bool load(const engine_params & p);

    // Generates a reply; on_piece receives UTF-8 text pieces and returns false to stop early.
    gen_stats chat(const std::vector<chat_msg> & msgs, const gen_params & gp, const std::function<bool(const std::string &)> & on_piece);

    std::string model_name() const { return model_name_; }
    bool        split() const { return cd_ != cp_; }
    bool        aliased() const { return aliased_; }
    std::string prefill_dev() const { return params_.prefill_dev; }
    double      now_ms() const;

    // recent requests and live phase for the dashboard
    std::vector<gen_stats> history() const;
    std::string            live_phase() const { std::lock_guard<std::mutex> lk(live_mtx_); return live_phase_; }
    size_t                 cache_entries() const;
    size_t                 cache_bytes() const;
    int64_t                keepwarm_pings() const { return pings_.load(); }

  private:
    struct cache_entry {
        std::vector<llama_token> toks;
        std::vector<uint8_t>     state;
    };

    std::string apply_template(const std::vector<chat_msg> & msgs) const;
    std::vector<llama_token> tokenize(const std::string & text) const;
    void prefill(llama_context * c, const std::vector<llama_token> & toks, int from);
    const cache_entry * cache_lookup(const std::vector<llama_token> & toks);
    void cache_insert(const std::vector<llama_token> & toks, std::vector<uint8_t> && state);
    void set_phase(const std::string & p);

    void keepwarm_start();
    void keepwarm_pause(bool p);
    void keepwarm_stop();

    engine_params params_;
    llama_model *   mp_ = nullptr;  // prefill model (NPU), or the only model
    llama_model *   md_ = nullptr;  // decode model (CPU)
    llama_context * cp_ = nullptr;
    llama_context * cd_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    bool        aliased_ = false;
    std::string model_name_;
    std::string tmpl_;

    std::mutex gen_mtx_;                 // one generation at a time
    mutable std::mutex hist_mtx_;
    std::vector<gen_stats> history_;
    int64_t next_id_ = 1;

    mutable std::mutex live_mtx_;
    std::string live_phase_ = "idle";

    mutable std::mutex cache_mtx_;
    std::list<cache_entry> cache_;       // front = most recently used
    size_t cache_bytes_ = 0;

    std::thread       kw_thread_;
    std::mutex        kw_mtx_;
    bool              kw_paused_ = false;
    std::atomic<bool> kw_stop_{false};
    std::atomic<int64_t> pings_{0};
};

}  // namespace trilane
