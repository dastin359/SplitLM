# SplitLM: NPU prefill + CPU decode for LLMs on Snapdragon X SoC

Every request is split by phase. The prompt (prefill) runs on the Hexagon NPU, whose matrix unit is the fastest arithmetic on the chip. Token generation (decode) runs on the Oryon CPU, which reads memory fastest. The model state moves between them through shared memory in about 7 ms, and both chips work from **one copy of the weights**. One engine, an OpenAI-compatible server, and a live dashboard.

**2.0× stock llama.cpp per 4K-token request on Qwen3.5-2B; 2.4× the NPU prefill that Qualcomm AI Hub publishes for the same model file.**

Built in a day on a Dell Latitude 7455 (Snapdragon X Elite X1E80100, 32 GB LPDDR5x-8448, Windows 11 on Arm), on a fork of llama.cpp.

## Results

### What the split buys over stock llama.cpp

Qwen3.5-2B Q4_0, one request with a 4096-token prompt and 128 generated tokens, same laptop:

| Setup | Prefill | Decode after 4096 tokens | First token | Whole request |
|---|---|---|---|---|
| Stock llama.cpp, default devices | 725 t/s | 31.4 t/s | 5.65 s | 9.73 s |
| Stock llama.cpp, CPU only | 423 t/s | 55.4 t/s | 9.69 s | 12.0 s |
| Stock llama.cpp, NPU only | 1,214 t/s | 29.0 t/s | 3.37 s | 7.79 s |
| **SplitLM** | **1,631 t/s** | **55.4 t/s** | **2.52 s** | **4.83 s** |

Every single-chip setup gives up one phase: the CPU prefills at a quarter of the NPU's speed, and the NPU decodes at half the CPU's speed. SplitLM takes the best chip for each phase, and its NPU kernels prefill 34% faster than stock. Prefill and decode are measured rates; first-token and whole-request times are computed from them (prompt ÷ prefill + 128 ÷ decode + 7 ms handoff). Measured end to end on a 3,328-token prompt: first token in 2.12 s vs 8.27 s on the CPU alone, at the same decode speed. (The CPU-only decode figure comes from the fork, whose in-place recurrent state makes decode about 5% faster than stock.)

### Bigger models

| Model | Result |
|---|---|
| **LFM2.5-24B-A2B** (24B MoE, 64 experts) | Runs split in **14.7 GB** instead of ~27 GB for two weight copies: the CPU reads the NPU's tiled experts in place. NPU prefill at 4096 tokens: 672 t/s (stock NPU kernels 537, CPU 167). End to end on a 3,686-token prompt: first token in 6.1 s vs 15.9 s CPU-only; decode 39.1 t/s vs 18.6 on the NPU alone. |
| **Qwen3.8-27B** (dense) | NPU prefill + CPU decode in one process in 19-20 GB (two copies would need ~30 GB). Speculative decoding with the model's own MTP head: 8.7-8.9 t/s vs 5.0 plain, identical greedy text. |

### Against Qualcomm's own runtime

Qualcomm AI Hub's "optimized" Qwen3.5-2B is byte-identical to the GGUF used here (unsloth Q4_0, sha256 `cd70221b…`), deployed with GenieX, Qualcomm's llama.cpp-based runtime. AI Hub publishes 668 t/s NPU prefill at a 4096-token context for the X Elite; SplitLM's NPU prefills at 1,631 t/s. On this laptop, GenieX makes you pick one chip per model (NPU: 28.3 t/s decode; CPU: 623 t/s prefill, measured with a 532-token prompt); SplitLM gets fast prefill and fast decode in one request.

### How close to the hardware

| Phase | Model | Now | % of ceiling |
|---|---|---|---|
| NPU prefill, 4096 tokens | Qwen3.5-2B | 1,631 t/s = 4.5 TFLOPS | 39% of 11.48 TFLOPS (FP16 matrix unit) |
| NPU prefill, 4096 tokens | LFM2.5-24B-A2B | 672 t/s = 2.9 TFLOPS | 26% |
| NPU prefill, 4096 tokens | Qwen3.8-27B | 105 t/s = 5.2 TFLOPS | 45% |
| CPU decode after 4096 tokens | Qwen3.5-2B | 55.4 t/s = 67 GB/s of weights | 61% of the ~110 GB/s we could read from DRAM |

The ceilings are measured, not datasheet guesses: the NPU's matrix unit sustains 3,809-3,982 of its 4,096 FP16 MACs per cycle at 1.40 GHz inside a single job, and the best DRAM read loop reaches ~110 GB/s (135 GB/s theoretical). What keeps prefill below the ceiling: every Q4_0 matmul expands weights to fp16 on the vector units first (matmuls cap near 70%), and the recurrent-state scan and element-wise ops are about half of the time. Realistic FP16 ceiling: about 55%. Decode's matmul kernels stream at 90-95 GB/s; the gap is ~785 small ops per token and their thread barriers.

Other measurements: NPU prefill costs 8 mJ per token vs 90 mJ on the CPU; CPU decode 0.81 J per token vs 0.21 J on the NPU at half the speed (whole-system power, 512-token prompts). Four simultaneous 400-550-token requests: 79 generated t/s aggregate with NPU prefill + CPU continuous batching, vs 55 for the same engine on the CPU alone.

## How it works

```
             prompt tokens                                   generated tokens
                  |                                                 ^
                  v                                                 |
   +-------------------------------+   state handoff    +-------------------------------+
   |  NPU lane (Hexagon v73)       |  KV + recurrent    |  CPU lane (Oryon, 8 threads)  |
   |  llama_context on HTP0        |  state, ~7 ms      |  llama_context on CPU         |
   |  prefill, ubatch 1024-4096    | -----------------> |  decode, continuous batching  |
   +-------------------------------+                    +-------------------------------+
                  |                                                 |
                  +------------------ one weight copy --------------+
                       Q4_0 tiles in NPU memory; the CPU reads them
                       in place with a NEON kernel (91-99% of heap bandwidth)
```

- **State handoff.** After prefill, `llama_state_seq_get_data` snapshots the sequence (attention KV plus the recurrent state of hybrid models) and `llama_state_seq_set_data` restores it into the CPU context: 25 MB and ~7 ms for 512 tokens of Qwen3.5-2B.
- **One weight copy.** The NPU repacks Q4_0 weights into 32×32 tiles in shared memory. A new CPU buffer type (`CPU_HTPTILE`) points the decode model's tensors at those tiles, and a NEON SDOT kernel reads them in place, for dense and MoE (MUL_MAT_ID) weights. A second buffer type (`CPU_SHARED`) lets the decode model borrow the prefill model's CPU-resident head and embeddings.
- **Concurrent engine.** An NPU worker prefills new prompts while the CPU decodes up to 4 sequences in one batch (`--slots N`). A stall-aware router sends tiny prompts to the CPU when the NPU's fixed cost would dominate. With `--npu-decode-slots 1`, the otherwise idle NPU decodes an overflow sequence and hands it back when a CPU slot frees up.
- **Prefix cache.** The handoff snapshot doubles as a cache entry keyed by the exact prompt tokens, so a follow-up turn prefills only the new text (0.17 s first token on a 3.3K-token conversation).
- **NPU keep-warm.** After ~1 s idle the NPU drops into a low-power state and the next prefill runs ~2.5× slower; a 4 KB op every 20 ms keeps it ready.

### Changes to llama.cpp (`patches/`)

- `ggml-cpu`: `CPU_HTPTILE` and `CPU_SHARED` buffer types, NEON kernels over NPU tiles, an i8mm kernel for 2-4-token speculative verification, a per-op and per-matmul-shape profiler (`GGML_CPU_PROF=N`).
- `llama`: recurrent states read in place instead of copied each step (+5% decode); `llama_model_get_tensor()`; `LLAMA_MMAP_NO_PREFETCH`.
- `ggml-hexagon` (NPU): fused element-wise kernels (sigmoid, conv block, norms, residual add in the matmul output), a faster MoE expert matmul (weight DMA overlapped with the activation gather, one gather for gate+up), causal key-block skipping in flash attention, and a 64-bit mapping-mask fix. New `test-backend-ops` cases cover each.

## How it was built

SplitLM was built in about 16 hours by Claude Code (Claude Opus 5.5) running autonomously overnight under my direction: targets set against measured hardware ceilings, then a fleet of agents in separate git worktrees (NPU kernels, speculative decoding, the MoE expert kernel, attention, decode overhead), each change attacked by an independent reviewer agent before merging. One NPU and one memory bus were shared by everyone, so a hardware lock serialized every measurement, and every speedup in this README is a same-session A/B against the previous build with identical greedy text or identical perplexity. Commits carry `Assisted-by` / `Co-Authored-By` trailers.

## Build and run

Requires Windows 11 on Snapdragon X, Visual Studio 2022 Build Tools (ARM64 + Clang), the Hexagon SDK 6.6 and Adreno OpenCL SDK, and test-signing with a trusted self-signed certificate (see `docs/backend/snapdragon/windows.md` in llama.cpp).

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 848bbda && git apply ..\SplitLM\patches\llama.cpp-splitlm.patch
..\SplitLM\scripts\build-llama.bat          # CPU + OpenCL + Hexagon, signed NPU libs -> llama.cpp\pkg-wos
cd ..\SplitLM && build.bat                  # engine binaries -> llama.cpp\pkg-wos\bin

cd ..\llama.cpp\pkg-wos\bin
$env:ADSP_LIBRARY_PATH = "<path>\llama.cpp\pkg-wos\lib"
.\trilane-server.exe -m <model.gguf> --prefill-dev HTP0                # dashboard at http://127.0.0.1:8080
.\trilane-server.exe -m <model.gguf> --prefill-dev HTP0 --bench 3200   # one request on each lane, no HTTP
```

API: `POST /v1/chat/completions` (OpenAI format, `stream: true` for SSE); an optional `"trilane": {"route": "auto" | "split" | "npu" | "cpu"}` forces a lane for comparisons. `GET /stats` returns routing, cache, slots and per-request timings. The binaries keep the project's working name, `trilane`.

## Limitations

- Single-copy aliasing covers Q4_0 weights; K-quants and IQ formats cannot be tiled for the NPU.
- Speculative decoding runs in `trilane-spec` (one stream); the multi-slot server does not draft yet.
- The NPU fusions shift LFM2.5's logits slightly against stock kernels (KLD 0.006, perplexity +0.66%); Qwen3.5-2B is unchanged (perplexity ratio 1.0008). The causal key-block skip changes LFM2.5's greedy text after a few tokens (a no-op update that rounds fp16 differently; perplexity identical; `GGML_HEXAGON_FA_KV_SKIP=0` restores it).
- The NPU gated-delta-net kernel fails 16 of 43 `test-backend-ops` cases (errors up to 6e-2) in every build including stock; Qwen3.5-2B perplexity is unaffected.
- Running two NPU processes at once slows both sharply; the engine owns the NPU while it runs.
