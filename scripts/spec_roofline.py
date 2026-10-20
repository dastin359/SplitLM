#!/usr/bin/env python3
"""Roofline-style estimate of speculative decode speed for draft/target pairs on a Snapdragon X Elite
(one LPDDR5x bus, ~110 GB/s best read loop, ~82 GB/s sustained by single-token CPU decode).

Model (greedy decoding, one stream):
  tokens per step   = 1 + alpha * (1 - alpha^k) / (1 - alpha)        (geometric acceptance, k drafted tokens)
  step time (seq.)  = k * t_draft_tok + t_verify(k + 1) + t_step_overhead
  step time (async) = max(t_verify(k + 1) * (1 + contention), k * t_draft_tok_npu) on a hit,
                      plus k * t_draft_tok on a miss (hit = all k accepted and the draft guessed the bonus token)
  t_verify(n)       = dense_bytes / BW * f_dense(n) + expert_bytes(n) / BW
  expert_bytes(n)   = per_expert_all_layers * min(E, top_k * (1 + r * (n - 1)))
                      (each extra token brings r * top_k experts not yet loaded; r = 0.45 fits the measured LFM2.5
                      verify costs 4 tokens 1.85x and 8 tokens 3.4x; uniform routing would give r ~ 0.94)
All byte counts come from the GGUF tensor tables (bytes read per decoded token, embeddings excluded).

Usage: python spec_roofline.py [--bw 82] [--csv]
"""
import argparse

# measured CPU verify cost of an n-token batch relative to one token (Qwen3.8-27B, Q4_0, i8mm tile kernel)
F_DENSE = {1: 1.00, 2: 1.15, 3: 1.42, 4: 1.29, 5: 1.85, 6: 2.0, 7: 2.2, 8: 2.43, 9: 2.6, 12: 3.2, 16: 4.0}
# small dense models are overhead-bound: measured Qwen3.5-2B 5 tokens 1.8x, 9 tokens 2.6x
F_SMALL = {1: 1.00, 2: 1.25, 3: 1.45, 4: 1.6, 5: 1.8, 6: 1.95, 7: 2.15, 8: 2.4, 9: 2.6, 12: 3.2, 16: 4.0}


def interp(table, n):
    if n in table:
        return table[n]
    ks = sorted(table)
    lo = max(k for k in ks if k <= n)
    hi = min(k for k in ks if k >= n)
    return table[lo] + (table[hi] - table[lo]) * (n - lo) / (hi - lo)


TARGETS = {
    # name: dense GB (incl. head, excl. embeddings), per-expert GB summed over MoE layers, n_expert, top_k, r, small
    "Qwen3.5-2B":      dict(dense=1.204, exp=0.0, E=0, K=0, r=0.0, small=True),
    "Qwen3.8-27B":     dict(dense=15.065, exp=0.0, E=0, K=0, r=0.0, small=False),
    # 40 MoE layers x 1.511 MB per expert; r assumed equal to LFM2.5's
    "Qwen3.6-35B-A3B": dict(dense=2.015, exp=40 * 0.001511, E=256, K=8, r=0.45, small=False),
    # 38 MoE layers x 5.308 MB; r fitted so that 4 tokens cost 1.85x and 8 tokens 3.4x (measured)
    "LFM2.5-24B-A2B":  dict(dense=0.660, exp=38 * 0.005308, E=64, K=4, r=0.45, small=False),
}

# draft: GB per drafted token (sequential), fixed per-token overhead ms, alpha assumption (range)
DRAFTS = [
    # target, draft, GB/token, overhead ms, alpha (low, mid, high), note
    ("Qwen3.8-27B", "MTP (in-file nextn, Q6_K head)", 1.308, 0.5, (0.78, 0.81, 0.84), "measured alpha 0.81 (k=3, 3.0 tok/step)"),
    ("Qwen3.8-27B", "MTP + head pruned to 64K+tail", 0.540, 0.5, (0.76, 0.79, 0.82), "alpha x ~0.975 coverage"),
    ("Qwen3.8-27B", "Qwen3.5-0.8B Q4_0", 0.498, 1.3, (0.60, 0.68, 0.75), "assumed"),
    ("Qwen3.8-27B", "Qwen3.5-0.8B, head pruned 64K", 0.344, 1.3, (0.58, 0.66, 0.73), "assumed"),
    ("Qwen3.8-27B", "Qwen3.5-2B Q4_0", 1.204, 1.3, (0.65, 0.74, 0.82), "assumed"),
    ("Qwen3.6-35B-A3B", "MTP (mtp-Qwen3.6-35B-A3B-Q4_0)", 0.324, 0.5, (0.72, 0.78, 0.84), "assumed (MTP-like)"),
    ("Qwen3.6-35B-A3B", "MTP, head pruned 64K", 0.112, 0.5, (0.70, 0.76, 0.82), "assumed"),
    ("Qwen3.6-35B-A3B", "Qwen3.5-0.8B Q4_0", 0.498, 1.3, (0.58, 0.65, 0.72), "assumed"),
    ("Qwen3.6-35B-A3B", "Qwen3.5-2B Q4_0", 1.204, 1.3, (0.63, 0.71, 0.79), "assumed"),
    ("Qwen3.5-2B", "MTP (Qwen3.5-2B-MTP, Q6_K head)", 0.455, 0.5, (0.65, 0.72, 0.80), "assumed"),
    ("Qwen3.5-2B", "MTP, head pruned 64K", 0.148, 0.5, (0.63, 0.70, 0.78), "assumed"),
    ("Qwen3.5-2B", "Qwen3.5-0.8B Q4_0", 0.498, 1.3, (0.55, 0.62, 0.70), "assumed"),
    ("LFM2.5-24B-A2B", "LFM2.5-2.6B Q4_0", 1.590, 1.3, (0.60, 0.70, 0.78), "assumed; draft costs > target"),
    ("LFM2.5-24B-A2B", "n-gram (prompt lookup)", 0.0, 0.1, (0.40, 0.50, 0.75), "measured 40-50% on code edit"),
    ("Qwen3.5-2B", "n-gram (prompt lookup)", 0.0, 0.1, (0.52, 0.72, 0.80), "measured 52% summary, 72-74% code edit"),
]


def t_verify(tgt, n, bw):
    f = interp(F_SMALL if tgt["small"] else F_DENSE, n)
    t = tgt["dense"] / bw * f
    if tgt["E"]:
        t += tgt["exp"] * min(tgt["E"], tgt["K"] * (1 + tgt["r"] * (n - 1))) / bw
    return t * 1000.0  # ms


def tokens_per_step(alpha, k):
    if k == 0:
        return 1.0
    return 1.0 + alpha * (1 - alpha ** k) / (1 - alpha)


def best(tgt, gb_d, ovh_d, alpha, bw, kmax=8, step_ovh=1.0, async_mode=False, contention=0.17, npu_bw=35.0, npu_tok_ms=3.0):
    t1 = t_verify(tgt, 1, bw)
    res = (1000.0 / t1, 0, 1.0)
    for k in range(1, kmax + 1):
        td = gb_d / bw * 1000.0 + ovh_d
        tv = t_verify(tgt, k + 1, bw)
        if not async_mode:
            ts = k * td + tv + step_ovh
        else:
            # draft on the NPU, running ahead while the CPU verifies; contention = CPU slowdown while the NPU streams
            # (calibrated on the NPU decode-slot run: a 34 GB/s NPU stream cost the CPU batch ~17%)
            td_npu = gb_d / npu_bw * 1000.0 + npu_tok_ms
            overlap = min(1.0, k * td_npu / tv)
            c = contention * overlap
            hit = alpha ** (k + 1)
            ts = hit * max(tv * (1 + c), k * td_npu) + (1 - hit) * (tv * (1 + c) + k * td_npu) + step_ovh
        tps = tokens_per_step(alpha, k) * 1000.0 / ts
        if tps > res[0]:
            res = (tps, k, tps * t1 / 1000.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=82.0, help="sustained CPU decode GB/s")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    print(f"plain decode at {a.bw:.0f} GB/s: " + ", ".join(f"{n} {1000.0 / t_verify(t, 1, a.bw):.1f} t/s" for n, t in TARGETS.items()))
    print("verify cost f(n) = t(n)/t(1):")
    for n, t in TARGETS.items():
        print(f"  {n:16s} " + " ".join(f"{k}:{t_verify(t, k, a.bw) / t_verify(t, 1, a.bw):.2f}" for k in (2, 3, 4, 5, 8, 16)))
    hdr = "target | draft | alpha lo/mid/hi | best k (mid) | t/s lo / mid / hi | speedup mid | async mid (t/s) | note"
    print(hdr)
    for tname, dname, gb, ovh, (al, am, ah), note in DRAFTS:
        tgt = TARGETS[tname]
        lo = best(tgt, gb, ovh, al, a.bw)
        mid = best(tgt, gb, ovh, am, a.bw)
        hi = best(tgt, gb, ovh, ah, a.bw)
        asy = best(tgt, gb, ovh, am, a.bw, async_mode=True) if gb > 0 else mid
        sep = "," if a.csv else " | "
        print(sep.join([tname, dname, f"{al:.2f}/{am:.2f}/{ah:.2f}", str(mid[1]),
                        f"{lo[0]:.1f} / {mid[0]:.1f} / {hi[0]:.1f}", f"{mid[2]:.2f}x", f"{asy[0]:.1f}", note]))
    npu_table(a.bw)


# NPU verify: one pass costs about one NPU decode step (weights streamed at ~35-43 GB/s) plus n / prefill rate
NPU = {
    # name: (NPU decode t/s, NPU prefill t/s) measured (Qwen3.6-35B-A3B cannot run: IQ3_S experts)
    "Qwen3.5-2B": (29.0, 2000.0),
    "Qwen3.8-27B": (2.85, 105.0),
    "LFM2.5-24B-A2B": (20.6, 1268.0),
}


def npu_table(bw):
    print("verify of n tokens, CPU (i8mm model) vs NPU (decode step + n/prefill rate), ms:")
    for name, (tg, pp) in NPU.items():
        tgt = TARGETS[name]
        row = []
        for n in (1, 2, 4, 8, 12, 16, 32):
            cpu = t_verify(tgt, n, bw) if n <= 16 else t_verify(tgt, 16, bw) * n / 16
            npu = 1000.0 / tg + 1000.0 * n / pp
            row.append(f"{n}:{cpu:.0f}/{npu:.0f}")
        print(f"  {name:16s} " + "  ".join(row))


if __name__ == "__main__":
    main()
