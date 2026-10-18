# Tri-Lane

Phase-split LLM inference for Snapdragon X laptops. Prefill runs on the Hexagon NPU, decode runs on the Oryon CPU, and the KV cache and recurrent state move between them through shared memory. One engine, one copy of the weights, an OpenAI-compatible server and a live dashboard.

Built in one night on a Dell Latitude 7455 (Snapdragon X Elite X1E80100, 32 GB LPDDR5x-8448) for a hackathon, on top of a fork of llama.cpp.

## Why split the phases

| Phase | What limits it | Best chip on this laptop (Qwen3.5-2B Q4_0) |
|---|---|---|
| Prefill (the prompt) | arithmetic: every weight is reused across hundreds of tokens | **NPU 1,279 t/s**, CPU 515, GPU 370 |
| Decode (each new token) | memory bandwidth and per-op overhead: every active weight is read once per token | **CPU 55 t/s**, GPU 31, NPU 30 |

No single chip wins both. vLLM and SGLang need CUDA/ROCm and Linux, and llama.cpp runs a model on one device per phase. Tri-Lane routes each phase to the chip that is good at it.

## Results

Full table with every chip, model and phase: https://claude.ai/artifact/WeygbBF4UgjEXvshPrEWPT

| Model | Tri-Lane TTFT | Tri-Lane decode | CPU only | NPU only |
|---|---|---|---|---|
| Qwen3.5-2B, 404-token prompt | **0.30 s** | **57.5 t/s** | 0.80 s, 58.2 t/s | 0.29 s, 29.4 t/s |
| Qwen3.5-2B, 3,328-token prompt | **2.12 s** | **51.2 t/s** | 8.27 s, 51.2 t/s | 2.11 s, 28.1 t/s |
| LFM2.5-24B-A2B (MoE), 388-token prompt | **1.28 s** | **49.5 t/s** | 1.84 s, 50.2 t/s | 1.27 s, 19.9 t/s |
| Qwen3.8-27B (dense) + MTP drafting, 344-token prompt | **4.8-5.4 s** | **8.7-8.9 t/s** | ~12.6 s, 5.6 t/s (10.4 with MTP) | 3.85 s, 2.85 t/s |

Tri-Lane gets NPU time to first token and CPU decode speed in the same request. On LFM2.5-24B-A2B the split fits in **14.7 GB**, and on Qwen3.8-27B in **19-20 GB**, because the CPU reads the NPU's copy of the weights in place (`CPU_HTPTILE`) and borrows its CPU-resident head and embeddings (`CPU_SHARED`). Two copies would need about 27 GB and 30 GB.

Against Qualcomm's own runtime (GenieX 0.9.0, llama.cpp-based) on the byte-identical Qwen3.5-2B GGUF that AI Hub ships: NPU 0.41 s TTFT / 28.3 t/s, CPU 0.85 s / 47.7 t/s. Energy: NPU prefill costs 8 mJ per token vs 90 mJ on the CPU; CPU decode 0.81 J per token vs 0.21 J on the NPU (half the speed).

How close to the hardware (details on the results page): NPU prefill reaches 38% of the 11.48 TFLOPS FP16 HMX peak on Qwen3.5-2B (was 29% with stock kernels); CPU decode streams 76-90 GB/s, 69-82% of the ~110 GB/s we could measure from DRAM.

Concurrency: the server prefills on the NPU while the CPU decodes up to four sequences in one batch: 79 generated t/s aggregate for four simultaneous 400-550-token requests on Qwen3.5-2B, vs 55 for the same engine on the CPU alone. An optional NPU decode slot (`--npu-decode-slots 1`) lets the idle NPU decode a fifth sequence.

## How it works

```
             prompt tokens                                   generated tokens
                  |                                                 ^
                  v                                                 |
   +-------------------------------+   state handoff    +-------------------------------+
   |  NPU lane (Hexagon v73, HTP)  |  KV + recurrent    |  CPU lane (Oryon, 8 threads)  |
   |  llama_context on HTP0        |  state, 7-15 ms    |  llama_context on CPU         |
   |  prefill, ubatch 512-2048     | -----------------> |  decode, 1 token per step     |
   +-------------------------------+                    +-------------------------------+
                  |                                                 |
                  +------------------ one weight copy --------------+
                       Q4_0 tiles in NPU rpcmem; the CPU reads them
                       with the CPU_HTPTILE NEON kernel (91-99% of heap bandwidth)
```

- **State handoff.** After prefill on the NPU context, `llama_state_seq_get_data` copies the sequence state (attention KV plus the hybrid models' recurrent state) and `llama_state_seq_set_data` restores it into the CPU context. Hybrid models (Gated DeltaNet, short-conv) keep this small: 25 MB for 512 tokens on Qwen3.5-2B.
- **Single weight copy (`CPU_HTPTILE`).** The NPU repacks Q4_0 weights into 32x32 tiles in rpcmem, which the CPU can map and read at 91-99% of normal heap bandwidth. A new CPU extra buffer type points the decode model's tensors at the NPU model's tensors by name. A NEON kernel un-transposes the tiles with zips and runs SDOT. The kernel handles MUL_MAT_ID (MoE experts) and MUL_MAT (dense weights). The engine picks the tensors to alias from where the NPU model actually placed them.
- **Lane router.** At startup the engine times short and long prefills on both lanes, fits `fixed_ms + n / tok_per_s` per lane, and routes each request to the cheaper one. An online EWMA correction learns from real requests. On LFM2.5 the NPU has a ~0.5 s fixed cost per ubatch (weights stream through a ~3 GB DSP window), so prompts under ~93 new tokens stay on the CPU.
- **Prefix cache.** The handoff snapshot doubles as a prefix-cache entry keyed by the exact prompt tokens. A multi-turn chat prefills only the new turn. Exact token boundaries make this safe for recurrent state.
- **NPU keep-warm.** After about 1 s idle the NPU drops into a low-power state, and the next big-model prefill runs ~2.5x slower. A background thread runs a 4 KB ADD on the NPU every 20 ms (paused while the NPU works), which keeps prefill at full speed at no decode cost.

## Build and run

Requires Windows 11 on Snapdragon X, Visual Studio 2022 Build Tools (ARM64 + Clang), the Hexagon SDK 6.6 and Adreno OpenCL SDK (`llama.cpp\scripts\snapdragon\setup-sdk.py`), and test-signing enabled with a trusted self-signed certificate (see `llama.cpp\docs\backend\snapdragon\windows.md`).

```
..\build-npu.bat                # fork of llama.cpp: CPU + OpenCL + Hexagon, signed HTP libs -> llama.cpp\pkg-wos
build.bat                       # trilane-server, trilane-split, trilane-rpcmem-bw -> pkg-wos\bin

cd ..\llama.cpp\pkg-wos\bin
$env:ADSP_LIBRARY_PATH = "C:\Users\NPU\hack\llama.cpp\pkg-wos\lib"
.\trilane-server.exe -m <model.gguf> --prefill-dev HTP0 -t 8          # http://127.0.0.1:8080
.\trilane-server.exe -m <model.gguf> --prefill-dev HTP0 --bench 300   # one-shot numbers, no HTTP
```

API: `POST /v1/chat/completions` (OpenAI format, `stream: true` for SSE). An optional `"trilane": {"route": "auto" | "npu" | "cpu"}` forces a lane for comparisons. `GET /stats` returns routing, cache and per-request timings.

## Fork changes (llama.cpp branch `trilane`)

- `ggml/src/ggml-cpu/htp-tile.cpp`: the `CPU_HTPTILE` buffer type and NEON kernels (SDOT default, USDOT and scalar variants via `GGML_HTPTILE_KERNEL`).
- `ggml/src/ggml-cpu/ggml-cpu.c`: per-op CPU profiler, enabled with `GGML_CPU_PROF=N`.
- `src/llama-model.cpp`, `include/llama.h`: `llama_model_get_tensor()` and `LLAMA_MMAP_NO_PREFETCH` (so an aliased model's mmap does not pull the whole file into RAM).
- `src/llama-graph.cpp`: recurrent states are read in place when every state comes from its own cell, instead of a GET_ROWS copy (+5% Qwen3.5 decode; `LLAMA_RS_NO_IDENT=1` disables).
- `ggml/src/ggml-cpu/htp-tile.cpp`: `CPU_SHARED` buffer type (decode model borrows another model's CPU-resident weights) and an i8mm multi-token tile kernel for speculative verification (lane 2).
- `ggml/src/ggml-hexagon/`: NPU prefill fusions: single-pass sigmoid for SILU/SWIGLU, fused conv block (CONCAT + SSM_CONV + SILU + state copy), RMS_NORM+SCALE, residual ADD inside the matmul output stage, batched short-row norms (lane 3). Tests in `tests/test-backend-ops.cpp`.

## Things we learned the hard way

- On battery the laptop throttles hard. Right after switching to AC we also saw decode flip between ~55 and ~30 t/s at the same reported CPU clock; a later 3-minute probe under sustained load showed no such state (memory bandwidth steady at ~104 GB/s, only brief CPU-compute dips), so the cause is unconfirmed. Benchmarks still get AC, cool-downs and temperature logging.
- `mmap` + CPU weight repacking keeps two copies of the weights. A 12.6 GB model then thrashes 32 GB. Read weights once (`-lm none`).
- Task Manager's NPU graph does show llama.cpp's Hexagon work (an early 0% reading was simply an idle moment, not an invisible path).
- The upstream Windows Snapdragon CMake preset is missing `-fno-finite-math-only`.
- Multi-session NPU layer split (`GGML_HEXAGON_DEVICES=4`) fails on Windows with `dspqueue_write 0x0E`.

## Limitations

- Speculative decoding runs in `trilane-spec` (single stream); the multi-slot server does not draft yet.
- Single-copy aliasing covers Q4_0 weights only. Qwen3.6-35B-A3B's IQ3_S experts cannot be tiled for the NPU.
- CPU-lane prefill over aliased weights is a per-token GEMV, so dense aliased models always prefill on the NPU.
- LFM2.5 on the fused NPU build shows a small logit shift against stock kernels (KLD 0.006, perplexity +0.66%); Qwen3.5-2B is unchanged (perplexity ratio 1.0008).
