// trilane-probe: separate memory bandwidth from CPU compute throughput, to tell which one a throttle state cuts.
//   trilane-probe bw  [threads] [mib]  -> heap streaming read GB/s
//   trilane-probe alu [threads]        -> independent FMA chains, GFLOP/s (no memory traffic)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <malloc.h>

#include "energy.h"

static volatile double g_sink;

static double bw(int threads, size_t mib) {
    const size_t n = mib << 20;
    static uint8_t * p = nullptr;
    if (!p) { p = (uint8_t *) _aligned_malloc(n, 4096); memset(p, 1, n); }
    double best = 0;
    for (int r = 0; r < 3; r++) {
        std::vector<std::thread> th;
        std::vector<uint64_t> part(threads);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < threads; i++) th.emplace_back([&, i] {
            const size_t chunk = n / threads / 64 * 64;
            const uint64_t * q = (const uint64_t *) (p + i * chunk);
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t j = 0; j < chunk / 8; j += 4) { s0 += q[j]; s1 += q[j + 1]; s2 += q[j + 2]; s3 += q[j + 3]; }
            part[i] = s0 + s1 + s2 + s3;
        });
        for (auto & t : th) t.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t acc = 0; for (auto v : part) acc += v; g_sink = (double) acc;
        best = std::max(best, n / s / 1e9);
    }
    return best;
}

static double alu(int threads) {
    const int64_t iters = 40'000'000;
    std::vector<std::thread> th;
    std::vector<double> part(threads);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < threads; i++) th.emplace_back([&, i] {
        double a0 = 1.0 + i, a1 = 1.1, a2 = 1.2, a3 = 1.3, a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
        const double m = 0.999999, c = 1e-7;
        for (int64_t k = 0; k < iters; k++) {
            a0 = a0 * m + c; a1 = a1 * m + c; a2 = a2 * m + c; a3 = a3 * m + c;
            a4 = a4 * m + c; a5 = a5 * m + c; a6 = a6 * m + c; a7 = a7 * m + c;
        }
        part[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    });
    for (auto & t : th) t.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double acc = 0; for (auto v : part) acc += v; g_sink = acc;
    return 2.0 * 8 * iters * threads / s / 1e9;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "bw";
    const int threads = argc > 2 ? atoi(argv[2]) : 8;
    if (mode == "emi") {
        // how often do the energy meters update? poll every 2 ms for 2 s and report the change intervals
        energy_meter em;
        if (!em.init({ L"sys" })) { printf("no energy meter\n"); return 1; }
        double last = em.read()[0];
        auto t_last = std::chrono::steady_clock::now();
        const auto t_end = t_last + std::chrono::seconds(2);
        int updates = 0;
        while (std::chrono::steady_clock::now() < t_end) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const double v = em.read()[0];
            if (v != last) {
                const auto now = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(now - t_last).count();
                printf("  +%.4f J after %.1f ms (%.2f W)\n", v - last, ms, (v - last) / (ms / 1000.0));
                last = v; t_last = now; updates++;
            }
        }
        printf("updates: %d in 2 s\n", updates);
        return 0;
    }
    if (mode == "bw")  printf("%.1f\n", bw(threads, argc > 3 ? (size_t) atoi(argv[3]) : 512));
    else               printf("%.2f\n", alu(threads));
    return 0;
}
