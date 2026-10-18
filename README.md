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
| Qwen3.5-2B, 512-token prompt | **418 ms** | **55.9 t/s** | ~990 ms, 57 t/s | 401 ms, 28 t/s |
| LFM2.5-24B-A2B (MoE), 512-token prompt | **1.47 s** | **49.7 t/s** | 2.56 s, 56.7 t/s | 1.40 s, 20.6 t/s |
| Qwen3.8-27B (dense), 404-token prompt | **3.9 s** | **5.65 t/s** | ~12.6 s, 5.9 t/s | 3.85 s, 2.85 t/s |

Tri-Lane gets NPU-class time to first token and 88-98% of CPU decode speed. On LFM2.5-24B-A2B the split fits in **14.7 GB**, and on Qwen3.8-27B in **21 GB**, because the CPU reads the NPU's copy of the weights in place. Two copies would need about 27 GB and 30 GB.

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

## Things we learned the hard way

- On battery the laptop throttles hard. Right after switching to AC we also saw decode flip between ~55 and ~30 t/s at the same reported CPU clock; a later 3-minute probe under sustained load showed no such state (memory bandwidth steady at ~104 GB/s, only brief CPU-compute dips), so the cause is unconfirmed. Benchmarks still get AC, cool-downs and temperature logging.
- `mmap` + CPU weight repacking keeps two copies of the weights. A 12.6 GB model then thrashes 32 GB. Read weights once (`-lm none`).
- Task Manager's NPU graph does show llama.cpp's Hexagon work (an early 0% reading was simply an idle moment, not an invisible path).
- The upstream Windows Snapdragon CMake preset is missing `-fno-finite-math-only`.
- Multi-session NPU layer split (`GGML_HEXAGON_DEVICES=4`) fails on Windows with `dspqueue_write 0x0E`.

## Limitations

- One request at a time (no continuous batching yet).
- Single-copy aliasing covers Q4_0 weights only. Qwen3.6-35B-A3B's IQ3_S experts cannot be tiled for the NPU.
- CPU-lane prefill over aliased weights is a per-token GEMV, so dense aliased models always prefill on the NPU.
- MTP speculative decoding is not integrated yet.
