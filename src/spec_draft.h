// spec2: draft-model speculation helpers for trilane-spec (two models in memory: draft + target).
//
//   draft_runner   greedy drafting on a separate draft context (CPU, NPU or GPU), without the re-evaluation that
//                  common's draft-simple does: the draft context keeps the tokens it drafted and is reconciled with
//                  the committed text after verification (KV rollback by seq_rm; recurrent/hybrid drafts restore a
//                  state checkpoint taken while the context held only committed tokens).
//   draft_ahead    optional async mode: while the target verifies draft D, a thread keeps drafting past D on the
//                  draft context, guessing the target's bonus token. On a full hit (all of D accepted and the bonus
//                  equals the guess) the next draft is already there.
//   k_tuner        adaptive draft length: picks k that maximizes expected tokens per second from the online
//                  acceptance estimate and measured draft/verify times.
#pragma once

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <vector>

static inline double spec2_now_ms() { return llama_time_us() / 1000.0; }

struct draft_runner {
    llama_context * ctx = nullptr;
    int n_vocab = 0;
    bool recurrent = false;              // hybrid/recurrent draft: rollback by state checkpoint
    int n_batch = 512;

    std::vector<llama_token> dtoks;      // tokens currently held by the draft context (positions 0..)
    std::vector<uint8_t> ckpt;           // partial (recurrent) state while the context held dtoks[0..ckpt_pos-1]
    int ckpt_pos = -1;

    // async draft-ahead: checkpoint taken after feeding the last draft token (valid only on a full hit)
    std::vector<uint8_t> ckpt_ahead;
    int ckpt_ahead_pos = -1;

    // stats
    int n_passes = 0, n_restores = 0, n_fed = 0;
    double t_ms = 0;

    void init(llama_context * c) {
        ctx = c;
        const llama_model * m = llama_get_model(c);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
        recurrent = llama_model_is_recurrent(m) || llama_model_is_hybrid(m);
        n_batch = (int) llama_n_batch(c);
    }

    llama_token argmax_last() const {
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (lg[i] > lg[best]) best = i;
        return best;
    }

    // append tokens at positions dtoks.size().., return the greedy next token after the last one (or -1 on error)
    llama_token feed(const std::vector<llama_token> & toks, bool want_logits = true) {
        if (toks.empty()) return -1;
        const double t0 = spec2_now_ms();
        common_batch b(ctx);
        for (size_t off = 0; off < toks.size(); off += n_batch) {
            const size_t n = std::min(toks.size() - off, (size_t) n_batch);
            b.clear();
            for (size_t i = 0; i < n; i++) {
                const bool last = off + i + 1 == toks.size();
                b.add(toks[off + i], (llama_pos) dtoks.size() + (llama_pos) i, 0, last && want_logits);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
                fprintf(stderr, "[spec2] draft decode failed\n");
                return -1;
            }
            dtoks.insert(dtoks.end(), toks.begin() + off, toks.begin() + off + n);
            n_passes++;
        }
        n_fed += (int) toks.size();
        llama_token r = want_logits ? argmax_last() : -1;
        t_ms += spec2_now_ms() - t0;
        return r;
    }

    void save(std::vector<uint8_t> & buf, int & pos) {
        if (!recurrent) { pos = (int) dtoks.size(); return; }
        const size_t n = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        buf.resize(n);
        llama_state_seq_get_data_ext(ctx, buf.data(), n, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        pos = (int) dtoks.size();
    }

    // make dtoks a prefix of the committed tokens without the last one (committed = prompt + generated, ending
    // with id_last, which is always re-fed so that its logits are current)
    bool reconcile(const std::vector<llama_token> & committed) {
        const size_t n_keep_max = committed.empty() ? 0 : committed.size() - 1;
        size_t L = 0;
        while (L < dtoks.size() && L < n_keep_max && dtoks[L] == committed[L]) L++;
        if (L == dtoks.size()) return true;
        llama_memory_t mem = llama_get_memory(ctx);
        if (!recurrent) {
            llama_memory_seq_rm(mem, 0, (llama_pos) L, -1);
            dtoks.resize(L);
            return true;
        }
        // recurrent state cannot move back token by token: restore the newest checkpoint at or below L
        std::vector<uint8_t> * buf = nullptr;
        int pos = -1;
        if (ckpt_ahead_pos >= 0 && ckpt_ahead_pos <= (int) L && ckpt_ahead_pos > ckpt_pos) { buf = &ckpt_ahead; pos = ckpt_ahead_pos; }
        else if (ckpt_pos >= 0 && ckpt_pos <= (int) L) { buf = &ckpt; pos = ckpt_pos; }
        if (!buf) {
            fprintf(stderr, "[spec2] no draft checkpoint at or below %zu: re-feeding the draft context from scratch\n", L);
            llama_memory_clear(mem, true);
            dtoks.clear();
            ckpt_pos = ckpt_ahead_pos = -1;
            return false;
        }
        if (llama_state_seq_set_data_ext(ctx, buf->data(), buf->size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != buf->size()) {
            fprintf(stderr, "[spec2] draft checkpoint restore failed\n");
            llama_memory_clear(mem, true);
            dtoks.clear();
            ckpt_pos = ckpt_ahead_pos = -1;
            return false;
        }
        llama_memory_seq_rm(mem, 0, pos, -1);
        dtoks.resize(pos);
        n_restores++;
        return true;
    }

    // a full draft-ahead hit: the ahead checkpoint now holds committed tokens only, make it the rollback point
    void promote_ahead() {
        if (ckpt_ahead_pos < 0) return;
        std::swap(ckpt, ckpt_ahead);
        ckpt_pos = ckpt_ahead_pos;
        ckpt_ahead_pos = -1;
    }

    // draft k tokens after committed (which ends with id_last); out receives the draft. Leaves the context holding
    // committed + draft[0..k-2] (the last draft token is not fed).
    void draft(const std::vector<llama_token> & committed, int k, std::vector<llama_token> & out) {
        out.clear();
        if (k <= 0) return;
        reconcile(committed);
        save(ckpt, ckpt_pos);   // the context holds committed tokens only
        ckpt_ahead_pos = -1;
        std::vector<llama_token> pend(committed.begin() + dtoks.size(), committed.end());
        llama_token t = feed(pend);
        for (int i = 0; i < k && t >= 0; i++) {
            out.push_back(t);
            if (i + 1 < k) t = feed({ t });
        }
    }
};

// async draft-ahead on its own thread: feeds the last draft token, checkpoints, then keeps drafting guesses
struct draft_ahead {
    draft_runner * dr = nullptr;
    std::thread th;
    std::atomic<bool> stop { false };
    std::vector<llama_token> guess;      // guess[0] = expected bonus token, guess[1..] = next draft if it hits
    int n_max = 0;
    bool running = false;
    // stats
    int n_launch = 0, n_hit = 0, n_tokens_used = 0;

    void launch(draft_runner * r, llama_token last_draft, int n) {
        dr = r; n_max = n; stop = false; guess.clear(); running = true; n_launch++;
        th = std::thread([this, last_draft]() {
            llama_token t = dr->feed({ last_draft });
            if (t < 0) return;
            dr->save(dr->ckpt_ahead, dr->ckpt_ahead_pos); // holds committed + draft if every draft token is accepted
            guess.push_back(t);
            while ((int) guess.size() < n_max + 1 && !stop.load()) {
                t = dr->feed({ t });
                if (t < 0) break;
                guess.push_back(t);
            }
        });
    }

    void finish() {
        if (!running) return;
        stop = true;
        th.join();
        running = false;
    }
};

// adaptive draft length from the online acceptance estimate and measured times
struct k_tuner {
    int k_min = 1, k_max = 8;
    double acc_sum = 0, rej_sum = 0;     // geometric MLE: alpha = accepted / (accepted + early stops)
    double td_ms = -1;                   // EWMA ms per drafted token
    std::vector<double> tv_ms;           // EWMA ms per verify batch of n tokens (index n)
    int steps = 0;

    void init(int kmax) { k_max = std::max(1, kmax); tv_ms.assign(k_max + 2, -1.0); }

    void observe(int k, int n_accepted, double t_draft_ms, double t_verify_ms) {
        const double decay = 0.95;
        acc_sum = acc_sum * decay + n_accepted;
        rej_sum = rej_sum * decay + (n_accepted < k ? 1.0 : 0.0);
        if (k > 0 && t_draft_ms > 0) td_ms = td_ms < 0 ? t_draft_ms / k : 0.9 * td_ms + 0.1 * t_draft_ms / k;
        const int n = std::min(k + 1, (int) tv_ms.size() - 1);
        tv_ms[n] = tv_ms[n] < 0 ? t_verify_ms : 0.9 * tv_ms[n] + 0.1 * t_verify_ms;
        steps++;
    }

    double alpha() const { return std::min(0.98, (acc_sum + 1.0) / (acc_sum + rej_sum + 2.0)); }

    double tv(int n) const {
        if (n < (int) tv_ms.size() && tv_ms[n] > 0) return tv_ms[n];
        // unknown size: nearest known, scaled like the measured dense i8mm verify cost (~+15% per extra token)
        int best = -1;
        for (int i = 1; i < (int) tv_ms.size(); i++) if (tv_ms[i] > 0 && (best < 0 || std::abs(i - n) < std::abs(best - n))) best = i;
        if (best < 0) return 1.0;
        return tv_ms[best] * (1.0 + 0.15 * (n - 1)) / (1.0 + 0.15 * (best - 1));
    }

    int pick() const {
        if (steps < 4 || td_ms < 0) return k_max;               // warm up at the configured length
        if (steps % 16 == 0) return 1 + (steps / 16) % k_max;   // occasional probe keeps every tv(n) fresh
        const double a = alpha();
        int best_k = k_min;
        double best = -1;
        for (int k = k_min; k <= k_max; k++) {
            const double tok = 1.0 + a * (1.0 - std::pow(a, k)) / (1.0 - a);
            const double tps = tok / (k * td_ms + tv(k + 1));
            if (tps > best) { best = tps; best_k = k; }
        }
        return best_k;
    }
};
