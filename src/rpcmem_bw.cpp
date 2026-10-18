// E1: can the CPU stream-read HTP0 (rpcmem) buffers as fast as ordinary heap memory?
// Gates the "tiled alias" single-copy design (CPU decode reads the NPU's expert weights in place).

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <malloc.h>

static volatile uint64_t g_sink;

static double read_gbs(const uint8_t * p, size_t n, int threads, int reps) {
    double best = 0;
    for (int r = 0; r < reps; r++) {
        std::vector<std::thread> th;
        std::vector<uint64_t> part(threads, 0);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < threads; i++) {
            th.emplace_back([&, i]() {
                const size_t chunk = n / threads / 64 * 64;
                const uint64_t * q = (const uint64_t *) (p + i * chunk);
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (size_t j = 0; j < chunk / 8; j += 4) { s0 += q[j]; s1 += q[j + 1]; s2 += q[j + 2]; s3 += q[j + 3]; }
                part[i] = s0 + s1 + s2 + s3;
            });
        }
        for (auto & t : th) t.join();
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t acc = 0; for (auto v : part) acc += v; g_sink = acc;
        best = std::max(best, n / s / 1e9);
    }
    return best;
}

int main(int argc, char ** argv) {
    size_t mib = argc > 1 ? (size_t) atoi(argv[1]) : 512;
    size_t n = mib << 20;
    ggml_backend_load_all();

    uint8_t * heap = (uint8_t *) _aligned_malloc(n, 4096);
    memset(heap, 1, n);

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    if (!dev) { fprintf(stderr, "HTP0 not found (set ADSP_LIBRARY_PATH)\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr); // opens the session
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    fprintf(stderr, "buft %s max_size %zu MiB\n", ggml_backend_buft_name(buft), ggml_backend_buft_get_max_size(buft) >> 20);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, n);
    if (!buf) { fprintf(stderr, "HTP0 alloc of %zu MiB failed\n", mib); return 1; }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    uint8_t * rp = (uint8_t *) ggml_backend_buffer_get_base(buf);
    memset(rp, 1, n);

    printf("buffer %zu MiB | threads | heap GB/s | rpcmem GB/s | ratio\n", mib);
    for (int t : {1, 4, 8, 12}) {
        double h = read_gbs(heap, n, t, 5);
        double r = read_gbs(rp, n, t, 5);
        printf("%27d | %9.1f | %11.1f | %5.2f\n", t, h, r, r / h);
    }
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    _aligned_free(heap);
    return 0;
}
