// Reads the Snapdragon PMIC energy rails that Windows exposes as "\Energy Meter(<rail>)\Energy" (picowatt-hours).
#pragma once

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <pdh.h>

#include <string>
#include <vector>

struct energy_meter {
    struct rail { std::wstring name; PDH_HCOUNTER h = nullptr; };
    PDH_HQUERY q = nullptr;
    std::vector<rail> rails;

    bool init(const std::vector<std::wstring> & names = { L"sys", L"cpu_cluster_0", L"cpu_cluster_1", L"cpu_cluster_2", L"gpu" }) {
        if (PdhOpenQueryW(nullptr, 0, &q) != ERROR_SUCCESS) return false;
        for (auto & n : names) {
            rail r; r.name = n;
            const std::wstring path = L"\\Energy Meter(" + n + L")\\Energy";
            if (PdhAddEnglishCounterW(q, path.c_str(), 0, &r.h) == ERROR_SUCCESS) rails.push_back(r);
        }
        return !rails.empty();
    }

    // cumulative joules per rail
    std::vector<double> read() {
        std::vector<double> out(rails.size(), 0.0);
        if (!q || PdhCollectQueryData(q) != ERROR_SUCCESS) return out;
        for (size_t i = 0; i < rails.size(); i++) {
            PDH_RAW_COUNTER rc;
            if (PdhGetRawCounterValue(rails[i].h, nullptr, &rc) == ERROR_SUCCESS) out[i] = (double) rc.FirstValue * 3.6e-9; // pWh -> J
        }
        return out;
    }

    ~energy_meter() { if (q) PdhCloseQuery(q); }
};
