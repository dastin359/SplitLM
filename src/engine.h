// Tri-Lane engine: phase-split LLM inference on Snapdragon X (prefill on the NPU, decode on the CPU).
// An NPU worker prefills queued requests one at a time while a CPU worker decodes up to max_seqs
// sequences in one batch, so both chips work at once.
#pragma once

#include "llama.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
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
    int    n_ctx            = 8192;    // per sequence
    int    n_batch          = 2048;
    int    n_ubatch         = 0;       // 0 = auto: 2048 for models beyond one NPU session window (fixed remap cost per ubatch), else 512
    int    max_seqs         = 4;       // concurrent decode slots on the CPU lane
    int    keepwarm_ms      = 20;      // tiny NPU op cadence between prefills (0 = off)
    size_t prefix_cache_mb  = 2048;    // budget for cached sequence-state snapshots
    int    cpu_prefill_max  = 96;      // prompts with at most this many new tokens prefill on the CPU
    bool   alias_experts    = true;    // CPU decode reads the NPU's copy of the weights in place when memory requires it
    bool   calibrate        = true;    // measure both lanes at startup and derive the routing threshold
};

// prefill cost model per lane: t(n) = fixed_ms + n / tps * 1000
struct lane_cost {
    double cpu_fixed_ms = 0, cpu_tps = 0;
    double npu_fixed_ms = 0, npu_tps = 0;
    int    crossover    = 0;           // NPU wins above this many new tokens (cold context)
    double cpu_corr     = 1.0;         // online actual/predicted ratios (EWMA), learned from real requests
    double npu_corr     = 1.0;

    double predict_cpu(int n) const { return cpu_corr * (cpu_fixed_ms + 1000.0 * n / cpu_tps); }
    double predict_npu(int n) const { return npu_corr * (npu_fixed_ms + 1000.0 * n / npu_tps) + 5.0; } // + handoff
};

struct gen_params {
    int      max_tokens  = 512;
    float    temperature = 0.7f;
    float    top_p       = 0.95f;
    float    min_p       = 0.05f;
    uint32_t seed        = 0xC0FFEE;
    std::string route    = "auto";     // auto | split | npu | cpu  (per-request lane override for comparisons)
};

struct gen_stats {
    int64_t     id          = 0;
    double      t_submit_ms = 0;       // engine clock
    double      t_start_ms  = 0;       // prefill start
    double      queue_ms    = 0;       // waiting for a lane
    int         n_prompt    = 0;
    int         n_cached    = 0;       // prompt tokens restored from the prefix cache
    int         n_prefill   = 0;       // tokens actually prefilled
    std::string route;                 // effective route
    std::string prefill_dev;
    std::string decode_dev;
    int         slot        = -1;      // CPU decode slot
    int         batch_peak  = 0;       // most sequences decoded together while this one ran
    double      prefill_ms  = 0;
    double      handoff_ms  = 0;
    double      handoff_mb  = 0;
    double      ttft_ms     = 0;       // from submit to first token
    double      t_decode_start_ms = 0; // engine clock when this sequence joined the CPU batch
    double      decode_ms   = 0;
    int         n_gen       = 0;
    std::string stop_reason;
};

struct chat_msg {
    std::string role;
    std::string content;
};

// one in-flight generation; the engine's workers fill it, a consumer drains pieces
struct request {
    gen_stats  st;
    gen_params gp;
    std::vector<llama_token> toks;
    std::vector<uint8_t>     restore;  // prefix-cache snapshot to start from (st.n_cached tokens)

    std::mutex                mtx;
    std::condition_variable   cv;
    std::deque<std::string>   pieces;
    bool                      done = false;
    std::atomic<bool>         cancelled{false};

    // worker-side state
    std::vector<uint8_t> blob;          // state after prefill, for the handoff
    llama_sampler *      smpl = nullptr;
    llama_token          cur  = 0;      // sampled, not yet emitted / fed
    int                  pos  = 0;
    std::string          pending;       // incomplete UTF-8 tail
    double               t_decode0 = 0;

    ~request();
    void push(std::string s);
    void finish(const std::string & reason);
};
using request_ptr = std::shared_ptr<request>;

class engine {
  public:
    engine() = default;
    ~engine();

    bool load(const engine_params & p);

    // queue a chat; returns immediately with a handle whose pieces stream in as tokens are generated
    request_ptr submit(const std::vector<chat_msg> & msgs, const gen_params & gp);

    // submit + drain; on_piece returns false to stop early
    gen_stats chat(const std::vector<chat_msg> & msgs, const gen_params & gp, const std::function<bool(const std::string &)> & on_piece);

    std::string model_name() const { return model_name_; }
    bool        split() const { return cd_ != cp_; }
    bool        aliased() const { return aliased_; }
    bool        dense_aliased() const { return dense_aliased_; }
    std::string prefill_dev() const { return params_.prefill_dev; }
    int         max_seqs() const { return params_.max_seqs; }
    double      now_ms() const;

    std::vector<gen_stats> history() const;
    std::string            live_phase() const;
    int                    active_slots() const { return n_active_.load(); }
    int                    npu_queue_len() const;
    size_t                 cache_entries() const;
    size_t                 cache_bytes() const;
    int64_t                keepwarm_pings() const { return pings_.load(); }
    lane_cost              cost() const { std::lock_guard<std::mutex> lk(cost_mtx_); return cost_; }
    int                    cpu_prefill_max() const { return params_.cpu_prefill_max; }

  private:
    struct cache_entry {
        std::vector<llama_token> toks;
        std::vector<uint8_t>     state;
    };

    std::string apply_template(const std::vector<chat_msg> & msgs) const;
    std::vector<llama_token> tokenize(const std::string & text) const;
    bool prefill(llama_context * c, const std::vector<llama_token> & toks, int from, llama_seq_id seq);
    int  cache_lookup(const std::vector<llama_token> & toks, std::vector<uint8_t> & state_out);
    void cache_insert(const std::vector<llama_token> & toks, const std::vector<uint8_t> & state);
    void learn(bool on_npu, int n, double ms);
    void record(const request_ptr & r);
    void calibrate();

    void npu_worker();
    void cpu_worker();
    void emit(const request_ptr & r);             // stream r->cur as text
    bool should_stop(const request_ptr & r) const;

    void keepwarm_start();
    void keepwarm_pause(bool p);
    void keepwarm_stop();

    engine_params params_;
    lane_cost     cost_;
    llama_model *   mp_ = nullptr;  // prefill model (NPU), or the only model
    llama_model *   md_ = nullptr;  // decode model (CPU)
    llama_context * cp_ = nullptr;  // owned by the NPU worker after load
    llama_context * cd_ = nullptr;  // owned by the CPU worker after load
    const llama_vocab * vocab_ = nullptr;
    bool        aliased_ = false;
    bool        dense_aliased_ = false;
    int         n_aliased_ = 0;
    std::string model_name_;
    std::string tmpl_;

    // scheduling
    mutable std::mutex      q_mtx_;
    std::condition_variable q_cv_;        // wakes both workers
    std::deque<request_ptr> npu_queue_;   // waiting for NPU prefill (or NPU-only generation)
    std::deque<request_ptr> cpu_inbox_;   // prefilled on the NPU, or short prompts to prefill on the CPU
    std::atomic<bool>       stop_{false};
    std::atomic<int>        n_active_{0};
    std::atomic<bool>       npu_busy_{false};
    std::thread             npu_thread_, cpu_thread_;
    int64_t                 next_id_ = 1;

    mutable std::mutex hist_mtx_;
    std::vector<gen_stats> history_;

    mutable std::mutex live_mtx_;
    std::string npu_phase_ = "idle", cpu_phase_ = "idle";

    mutable std::mutex cache_mtx_;
    std::list<cache_entry> cache_;        // front = most recently used
    size_t cache_bytes_ = 0;

    mutable std::mutex cost_mtx_;

    std::thread          kw_thread_;
    std::mutex           kw_mtx_;
    bool                 kw_paused_ = false;
    std::atomic<bool>    kw_stop_{false};
    std::atomic<int64_t> pings_{0};
};

}  // namespace trilane
