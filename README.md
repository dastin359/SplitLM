# SplitLM: NPU prefill + CPU decode with custom kernels on Snapdragon X SoC

LLM inference on a Snapdragon X laptop, rebuilt around what each chip is good at, with each phase pushed toward its hardware ceiling. Two halves:

- **A phase-split engine.** Long prompts are prefilled on the Hexagon NPU, whose matrix unit prefills about 4× faster than the CPU or GPU here. Tokens are decoded on the Oryon CPU, which reads memory fastest. The model state moves between them through shared memory (about 7 ms for a 512-token prompt). For models too big to hold twice (MoE experts, dense models over 6 GiB), both chips work from one copy of the weights. Concurrent requests overlap: the NPU prefills while the CPU decodes a batch. OpenAI-compatible server and live dashboard.
- **Kernels pushed toward the hardware ceiling.** Fused NPU element-wise kernels, a faster MoE expert matmul and causal block skipping in NPU flash attention: Qwen3.5-2B NPU prefill +26% over stock llama.cpp kernels, LFM2.5-24B-A2B +21%. On the CPU: kernels that decode straight from the NPU's tiled weights, in-place recurrent state (+5% decode), and speculative decoding with the model's own MTP head (Qwen3.8-27B 1.7×).

**On Qwen3.5-2B, a 4K-token request is about 2× faster than llama.cpp's default device choice and 1.6× faster than its fastest single chip, from the same build. NPU prefill runs at 1.2× Qualcomm's own GenieX runtime on this laptop, with the same model file.**

Built in a day on a Dell Latitude 7455 (Snapdragon X Elite X1E80100, 32 GB LPDDR5x-8448, Windows 11 on Arm), on a fork of llama.cpp.

## Results

### What the split buys

Qwen3.5-2B Q4_0, one request with a 4096-token prompt and 128 generated tokens. Every row uses the same SplitLM build (same kernels), so the table isolates the effect of splitting the phases across chips:

| Setup (same build) | Prefill | Decode after 4096 tokens | First token | Whole request |
|---|---|---|---|---|
| llama.cpp default device choice | 725 t/s | 31.4 t/s | 5.65 s | 9.72 s |
| CPU only | 423 t/s | 55.4 t/s | 9.69 s | 12.0 s |
| NPU only | 1,614 t/s | 29.0 t/s | 2.54 s | 6.95 s |
| **SplitLM: NPU prefill, CPU decode** | **1,614 t/s** | **55.4 t/s** | **2.54 s** | **4.85 s** |

Rates are llama-bench at 4096 tokens (NPU with the engine's 1024-token batches); first-token and whole-request times are computed as prompt ÷ prefill + 128 ÷ decode. The split is 2.0× the default device choice, 2.5× CPU only and 1.4× NPU only. The SplitLM row assumes decode after the handoff runs at the CPU-only rate; in the engine's own end-to-end run on the current build (3,328-token prompt) it ran slower than a CPU-prefilled sequence, 43.1 vs 49.5 t/s, an open issue (an earlier build measured 51.2 on both). Measured end to end in that run: first token in 2.10 s vs 8.27 s CPU-only and 2.06 s NPU-only; whole request with 128 tokens about 5.1 s vs 10.9 s CPU-only and 6.4 s NPU-only.

### What the kernels buy

Each row is a same-session A/B against the build before the change, with greedy text or perplexity checked:

| Change | Model | Before → after | Check |
|---|---|---|---|
| Fused NPU element-wise kernels (single-pass sigmoid for SILU/SWIGLU, fused conv block, RMS_NORM+SCALE, residual add in the matmul output, batched short-row norms) | Qwen3.5-2B | NPU prefill at 4096 tokens 1,222 → 1,534 t/s (**+26%**); element-wise time per 512 tokens 167 → 80 ms | perplexity ratio 1.0008 |
| MoE expert matmul: next weight chunk's DMA issued during the activation gather and across experts; one gather for gate+up | LFM2.5-24B-A2B | prefill at 4096 tokens 579 → 641 t/s (**+11%**); expert gate+up 36.8 → 30.7 ms per layer | identical text and perplexity |
| Streamed, pipelined NPU expert kernel: the matrix unit multiplies chunk i while the vector units expand chunk i+1 and write out chunk i−1 | LFM2.5-24B-A2B | prefill 428 → 510 t/s at 512 tokens (**+19%**), 695 → 799 at 2048 (**+15%**), 685 → 749 at 4096 (**+9%**) | identical text and perplexity |
| Causal key-block skipping in NPU flash attention | LFM2.5-24B-A2B | attention −32 to −35%; prefill at 4096 tokens (4096-token batches) 603 → 654 t/s (**+8.5%**) | identical perplexity |
| All NPU fixes merged, vs stock NPU kernels at the same 2048-token batches | LFM2.5-24B-A2B | 537 → 651 t/s (**+21%**); 672 t/s with 4096-token batches | see Limitations |
| In-place recurrent state (no per-token state copy) | Qwen3.5-2B | CPU decode 60.3 → 63.4 t/s (**+5%**) | identical text |
| Single-token decode overhead: recurrent-state update fused with its copy, row-split CONCAT, 108 barriers per token elided between single-thread nodes | Qwen3.5-2B | CPU decode median 56.3 → 57.4 t/s (**+1.8%**), best 60.0 → 61.4 (+2.3%); state update 1.06 → 0.47 ms per token (interleaved runs on a throttling machine) | identical text and perplexity |
| Speculative decoding with the model's MTP head, 3 drafted tokens | Qwen3.8-27B | decode 5.0-5.3 → 8.7-8.9 t/s (**1.7×**), 67% accepted | identical text |


### Bigger models

| Model | Result |
|---|---|
| **LFM2.5-24B-A2B** (24B MoE, 64 experts) | Runs split in **14.7 GB** (measured with 512-token batches) instead of ~27 GB for two weight copies: the CPU reads the NPU's tiled experts in place. NPU prefill at 4096 tokens: 749 t/s (CPU: 167 t/s). End to end on a 3,686-token prompt: first token in 6.1 s vs 15.9 s CPU-only; decode 39.1 t/s vs 18.6 on the NPU alone. |
| **Qwen3.8-27B** (dense) | NPU prefill + CPU decode in one process in 17.5-18.4 GB (18.8-19.9 GB with the MTP head; two weight copies would need ~32 GB). MTP speculative decoding 8.7-8.9 t/s vs 5.0-5.3 plain (344-token prompt). |

### Against Qualcomm's own runtime

Qualcomm AI Hub's "optimized" Qwen3.5-2B is byte-identical to the GGUF used here (unsloth Q4_0, sha256 `cd70221b…`), deployed with GenieX, Qualcomm's llama.cpp-based runtime. Measured on this laptop with a 532-token prompt, `geniex infer` runs a model on one compute unit chosen with `--compute`: NPU 1,291 t/s prefill and 28.3 t/s decode, CPU 623 t/s prefill and 47.7 t/s decode. SplitLM's NPU prefills ~512 tokens at about 1,600 t/s (1.2× GenieX's NPU) and decodes on the CPU in the same request. AI Hub's published X Elite figures at a 4096-token context are 668 t/s prefill and 4.12 t/s decode; their test conditions are not stated, and GenieX was not measured here at 4096 tokens.

### How close to the hardware

| Phase | Model | Now | % of ceiling |
|---|---|---|---|
| NPU prefill, 4096 tokens | Qwen3.5-2B | 1,631 t/s = 4.5 TFLOPS (2048-token batches) | 39% of 11.48 TFLOPS |
| NPU prefill, 4096 tokens | LFM2.5-24B-A2B | 749 t/s = 3.3 TFLOPS | 29% (26% before the pipelined expert kernel) |
| NPU prefill, 4096 tokens | Qwen3.8-27B | 105 t/s = 5.2 TFLOPS (stock NPU kernels, 512-token batches; not re-measured) | 45% |
| CPU decode after 4096 tokens (llama-bench, CPU only) | Qwen3.5-2B | 55.4 t/s = 67 GB/s of weights | 61% of ~110 GB/s |

TFLOPS = t/s × 2 × active matmul weights per token (2.75 / 4.38 / 49.45 GFLOP; attention and output head excluded). The ceilings come from measurement, not datasheets: inside a matrix job the NPU sustains 3,809-3,982 FP16 MACs per cycle at 1.40 GHz, which we round up to a 4,096-MAC peak (11.48 TFLOPS); the best DRAM read loop reaches ~110 GB/s (135 GB/s theoretical).

What keeps Qwen3.5-2B below the prefill ceiling: every Q4_0 matmul expands weights to fp16 on the vector units first (the matmuls alone cap near 70%), and the recurrent-state scan and element-wise ops take about 40% of the time (125 of 317 ms per 512-token batch); its realistic FP16 ceiling is about 55%. LFM2.5 is held back mainly by its expert matmuls (the pipelined expert kernel above overlaps its DMA, dequant, matrix-unit and scatter steps; the matrix unit is still idle much of the time) and by attention. Decode's matmul kernels stream at 90-95 GB/s (profiled from an empty context); the rest of the gap is ~785 graph nodes per token with their thread barriers (about 1.3 ms) plus about 2.3 ms per token of attention over the 4,096-token cache (63.4 t/s from an empty context vs 55.4 at 4K).

Other measurements (Qwen3.5-2B, early build with stock NPU kernels): NPU prefill costs 8 mJ per token above a 15.5 W idle vs 90 mJ on the CPU (512-token prompts); CPU decode 0.81 J per token above idle vs 0.20 J on the NPU at half the speed (1.08 vs 0.73 J counting idle). Four simultaneous 400-550-token requests: 79 generated t/s aggregate with NPU prefill and CPU continuous batching; the same engine on the CPU alone reached 55 t/s in an earlier run (older build; not a same-session A/B).

## How it works

```
             prompt tokens                                   generated tokens
                  |                                                 ^
                  v                                                 |
   +-------------------------------+   state handoff    +-------------------------------+
   |  NPU lane (Hexagon v73)       |  KV + recurrent    |  CPU lane (Oryon, 8 threads)  |
   |  llama_context on HTP0        |  state, 7-16 ms    |  llama_context on CPU         |
   |  prefill, ubatch 512 / 4096   | -----------------> |  decode, continuous batching  |
   +-------------------------------+                    +-------------------------------+
                  |                                                 |
                  +---- one weight copy (MoE experts, dense > 6 GiB) ---+
                       Q4_0 tiles in NPU memory; the CPU reads them in place
```

- **State handoff.** After prefill, `llama_state_seq_get_data` snapshots the sequence (attention KV plus the recurrent state of hybrid models) and `llama_state_seq_set_data` restores it into the CPU context: 25 MB and ~7 ms for 512 tokens of Qwen3.5-2B, 16 ms for 1.7K tokens of LFM2.5.
- **One weight copy where memory needs it.** The NPU repacks Q4_0 weights into 32×32 tiles in shared memory. For MoE models (experts) and dense models over 6 GiB, a new CPU buffer type (`CPU_HTPTILE`) points the decode model's Q4_0 tensors at those tiles, and a NEON SDOT kernel reads them in place (CPU reads of NPU memory reach 91-99% of heap bandwidth; the expert kernel streams ~90 GB/s). Small models such as Qwen3.5-2B keep a second CPU copy, because aliased decode is about 7% slower. In `trilane-spec`, a second buffer type (`CPU_SHARED`) lets the decode model borrow the prefill model's CPU-resident head and embeddings (1.6 GiB saved on Qwen3.8-27B).
- **Concurrent engine.** An NPU worker prefills new prompts while the CPU decodes up to 4 sequences in one batch (`--slots N`). A stall-aware router sends a prompt to the CPU when the NPU's fixed cost would dominate. With `--npu-decode-slots 1`, the otherwise idle NPU decodes an overflow sequence and hands it back when a CPU slot frees up.
- **Prefix cache.** The handoff snapshot doubles as a cache entry keyed by the exact prompt tokens, so a follow-up turn prefills only what follows the cached prompt: the previous reply plus the new message (80 tokens and a 0.17 s first token on a 3.3K-token conversation).
- **NPU keep-warm.** After ~1-3 s idle the NPU drops into a low-power state, and the next prefill ran ~2.5× slower on LFM2.5; a 4 KB op every 20 ms keeps it ready.

### Changes to llama.cpp (`patches/`)

- `ggml-hexagon` (NPU): fused element-wise kernels (sigmoid, conv block, norms, residual add in the matmul output), a streamed, pipelined MoE expert matmul (weight DMA overlapped with the activation gather, one gather for gate+up, matrix-unit work overlapped with dequant and scatter), causal key-block skipping in flash attention, and a 64-bit mapping-mask fix. New `test-backend-ops` cases cover the fused kernels, LFM-shaped expert matmuls and causal-mask attention; the mask fix is checked with the existing tests.
- `ggml-cpu`: `CPU_HTPTILE` and `CPU_SHARED` buffer types, NEON kernels over NPU tiles, an i8mm kernel for 2-4-token speculative verification, single-token decode fusions (recurrent-state update + copy, row-split CONCAT) and barrier elision (`GGML_CPU_GDN_CPY_FUSION=0` / `GGML_CPU_BAR_ELIDE=0` disable them), a per-op and per-matmul-shape profiler (`GGML_CPU_PROF=N`), opt-in decode knobs (`GGML_GEMV_NOBAR`, `GGML_REPACK_CHUNKS`, `GGML_GEMV_PF`, default off).
- `llama`: recurrent states read in place instead of copied each step; `llama_model_get_tensor()`; `LLAMA_MMAP_NO_PREFETCH`. `common`: `common_speculative_set_ctx_tgt()`, so speculative decoding can prefill on one context and verify on another.

## The ExecuTorch path

SplitLM runs on llama.cpp's Hexagon backend because it builds natively for Windows on Arm today (ExecuTorch 1.5.1 publishes Windows wheels for x64 only, and its Qualcomm backend compiles models ahead of time with the QAIRT SDK). The design maps directly onto ExecuTorch, and that is the planned port:

- **Two methods, two delegates.** Export the model's prefill method (a 1024-4096-token batch) delegated to the Qualcomm QNN backend on the HTP, and its single-token decode method delegated to XNNPACK on the Oryon CPU. ExecuTorch's Qualcomm llama example already builds separate prefill and KV-cache decode graphs, but runs both on the NPU; our measurements say decode belongs on the CPU (55 t/s vs 29 t/s on the NPU for Qwen3.5-2B at 4096 tokens).
- **State handoff.** Keep the KV cache (and recurrent state for hybrid models) in a shared buffer that both methods use as input and output, the same snapshot/restore that SplitLM does with `llama_state_seq_get/set_data` today.
- **One weight copy.** Our `CPU_HTPTILE` result (CPU reads of NPU memory at 91-99% of heap bandwidth) says the decode delegate can read the NPU's weight buffers in place instead of loading a second copy.
- **Kernel lessons.** Fp16 matmuls with on-the-fly Q4_0 expansion cap near 70% of the HMX peak; QNN's integer (w4a16 / w8a16) matmuls avoid that expansion, which is the route past the ~55% FP16 ceiling measured here.

## How it was built

SplitLM was built in about 16 hours by Claude Code (Claude Opus 5.5) running autonomously overnight under my direction: targets set against measured hardware ceilings, then a fleet of agents in separate git worktrees (NPU kernels, speculative decoding, the MoE expert kernel, attention, decode overhead). Each kernel change was checked with `test-backend-ops`, a same-session A/B run against the previous build, and a greedy-text or perplexity comparison; a multi-agent code review covered the engine and server, and later kernel work also got an adversarial reviewer agent. One NPU and one memory bus were shared by everyone, so a hardware lock serialized the measurements; when the lock failed (a few times overnight), the overlapping runs were discarded and redone. Comparisons across builds or settings are labeled as such.

## Build and run

Requires Windows 11 on Snapdragon X, Visual Studio 2022 Build Tools (ARM64 + Clang), Hexagon SDK 6.6.0.0 (tools 19.0.07), Adreno OpenCL SDK 2.3.2, Windows SDK 10.0.26100 with the WDK (inf2cat, signtool), and test-signing with a trusted self-signed certificate (see `docs/backend/snapdragon/windows.md` in llama.cpp; put the .pfx at `%USERPROFILE%\Certs\ggml-htp-v1.pfx` or set `HEXAGON_HTP_CERT`). Clone both repos side by side in one folder, then run one command per line (PowerShell):

```
git clone https://github.com/dastin359/SplitLM
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 848bbda9041decd87904f7eb5a2990863ecce03b
git apply ..\SplitLM\patches\llama.cpp-splitlm.patch
..\SplitLM\scripts\build-llama.bat
cd ..\SplitLM
.\build.bat

cd ..\llama.cpp\pkg-wos\bin
$env:ADSP_LIBRARY_PATH = "$PWD\..\lib"
.\trilane-server.exe -m <model.gguf> --prefill-dev HTP0              # dashboard at http://127.0.0.1:8080
.\trilane-server.exe -m <model.gguf> --prefill-dev HTP0 -ub 1024 --bench 2600
```

`--bench N` runs a ~1.3N-token prompt through the split, NPU-only and CPU-only routes plus a cached follow-up, without HTTP (`-ub 1024` matches the Qwen3.5-2B runs; models over 3 GiB default to 4096-token batches). API: `POST /v1/chat/completions` (OpenAI format, `stream: true` for SSE); an optional `"trilane": {"route": "auto" | "split" | "npu" | "cpu"}` forces a route. `GET /stats` returns routing, cache, slots and per-request timings. The binaries keep the project's working name, `trilane`.

## Limitations

- In the current build, Qwen3.5-2B decode after an NPU handoff measured 43.1 t/s vs 49.5 t/s for a CPU-prefilled sequence at 3,328 tokens (an earlier build: 51.2 vs 51.2). Not yet explained.
- Single-copy aliasing covers Q4_0 weights only, because the CPU tile kernel reads Q4_0 tiles. Other formats the NPU tiles (Q4_1, Q8_0, K-quants, IQ4_NL, MXFP4) keep a separate CPU copy, and formats the NPU cannot run (e.g. IQ3_S) stay on the CPU.
- Speculative decoding runs in `trilane-spec` (one stream); the multi-slot server does not draft yet.
- The fork's NPU kernels shift LFM2.5's logits slightly against stock (KLD 0.006; perplexity +0.66% ± 0.3% on 8 wikitext chunks, +0.35% on 16). Disabling the new fusions one at a time does not remove the shift, so the cause is unresolved. Qwen3.5-2B is unchanged (perplexity ratio 1.0008). The causal key-block skip changes LFM2.5's greedy text after a few tokens (a no-op update that rounds fp16 differently; perplexity identical; `GGML_HEXAGON_FA_KV_SKIP=0` restores it).
- The NPU gated-delta-net kernel, unchanged from upstream, fails 16 of 43 `test-backend-ops` cases (NMSE up to 6e-2 against a 1e-7 threshold, varying run to run) in every build tested. Its effect on output quality vs the CPU has not been measured.
- A second process holding an NPU session, even an idle one, slowed another process's NPU work 2-4×, and overlapping NPU jobs have failed with dspqueue errors. Run one NPU process at a time.
