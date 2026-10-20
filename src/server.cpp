// trilane-server: OpenAI-compatible chat server + live dashboard on top of the Tri-Lane engine.

#include "engine.h"

#include "ggml-backend.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

using json = nlohmann::ordered_json;

static bool g_verbose = false;
static void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static json stats_json(const trilane::gen_stats & s) {
    return {
        { "id", s.id }, { "t_submit_ms", s.t_submit_ms }, { "t_start_ms", s.t_start_ms }, { "queue_ms", s.queue_ms },
        { "t_decode_start_ms", s.t_decode_start_ms }, { "t_cpu_join_ms", s.t_cpu_join_ms }, { "slot", s.slot }, { "batch_peak", s.batch_peak },
        { "n_prompt", s.n_prompt }, { "n_cached", s.n_cached }, { "n_prefill", s.n_prefill },
        { "route", s.route }, { "prefill_dev", s.prefill_dev }, { "decode_dev", s.decode_dev },
        { "prefill_ms", s.prefill_ms }, { "prefill_tps", s.prefill_ms > 0 ? s.n_prefill / (s.prefill_ms / 1000.0) : 0.0 },
        { "handoff_ms", s.handoff_ms }, { "handoff_mb", s.handoff_mb }, { "ttft_ms", s.ttft_ms },
        { "decode_ms", s.decode_ms }, { "n_gen", s.n_gen }, { "npu_decode_ms", s.npu_decode_ms }, { "n_gen_npu", s.n_gen_npu },
        { "decode_tps", s.decode_ms > 0 ? s.n_gen / (s.decode_ms / 1000.0) : 0.0 },
        { "stop", s.stop_reason },
    };
}

static std::string dump(const json & j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// value of an optional field; missing or null gives the default (OpenAI clients send nulls)
template <typename T> static T jv(const json & o, const char * key, T def) {
    if (!o.is_object() || !o.contains(key) || o[key].is_null()) return def;
    return o[key].get<T>();
}

static bool is_error(const std::string & stop) { return stop != "stop" && stop != "length" && stop != "client"; }

static std::string error_text(const std::string & stop) {
    if (stop == "context_overflow") return "the prompt does not fit in the context window";
    if (stop == "prefill_error") return "prompt processing failed";
    if (stop == "handoff_error") return "moving the prompt state to the decode lane failed";
    return "generation failed (" + stop + ")";
}

static void usage() {
    fprintf(stderr,
        "usage: trilane-server -m model.gguf [--prefill-model prefill.gguf] [--prefill-dev HTP0|CPU] [--host 127.0.0.1] [--port 8080]\n"
        "       [-t decode_threads] [-c n_ctx] [-b n_batch] [-ub n_ubatch] [--keepwarm-ms 20] [--cpu-prefill-max 24] [--no-alias]\n"
        "       [--cache-mb 2048] [--slots 4] [--npu-decode-slots 0|1] [--ui path/to/index.html] [-v]\n"
        "       [--prio 0-3] [--no-threadpool] [--bench-routes split,cpu,...]\n"
        "  --prio: decode thread priority 0 normal, 1 medium, 2 high (default), 3 realtime; 0 = old default\n"
        "  env TRILANE_WORKER0_PRIO=0: decode thread (graph worker 0) keeps normal priority and throttling (old behaviour)\n");
    exit(1);
}

int main(int argc, char ** argv) {
    trilane::engine_params ep;
    std::string host = "127.0.0.1", ui_path;
    int bench_words = 0, bench_gen = 64, bench_conc = 0, bench_words_conc = 200;
    std::string bench_routes; // --bench-routes: explicit route sequence (e.g. split,cpu,split,cpu), no follow-up turn
    int port = 8080;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) usage(); return argv[++i]; };
        if      (k == "-m")                ep.model = next();
        else if (k == "--prefill-model")   ep.prefill_model = next();
        else if (k == "--prefill-dev")     ep.prefill_dev = next();
        else if (k == "--host")            host = next();
        else if (k == "--port")            port = atoi(next());
        else if (k == "-t")                ep.threads_decode = atoi(next());
        else if (k == "-c")                ep.n_ctx = atoi(next());
        else if (k == "-ub")               ep.n_ubatch = atoi(next());
        else if (k == "-b")                ep.n_batch = atoi(next());
        else if (k == "--keepwarm-ms")     ep.keepwarm_ms = atoi(next());
        else if (k == "--cpu-prefill-max") ep.cpu_prefill_max = atoi(next());
        else if (k == "--no-alias")        ep.alias_experts = false;
        else if (k == "--no-calibrate")    ep.calibrate = false;
        else if (k == "--cache-mb")        ep.prefix_cache_mb = (size_t) atoll(next());
        else if (k == "--ui")              ui_path = next();
        else if (k == "--bench")           bench_words = atoi(next());
        else if (k == "--bench-gen")       bench_gen = atoi(next());
        else if (k == "--bench-concurrent") bench_conc = atoi(next());
        else if (k == "--bench-words")     bench_words_conc = atoi(next());
        else if (k == "--slots")           ep.max_seqs = atoi(next());
        else if (k == "--npu-decode-slots") ep.npu_decode_slots = atoi(next());
        else if (k == "--prio")            ep.prio = atoi(next());
        else if (k == "--no-threadpool")   ep.threadpool = false;
        else if (k == "--bench-routes")    bench_routes = next();
        else if (k == "-v")                g_verbose = true;
        else usage();
    }
    if (ep.model.empty()) usage();
    if (ui_path.empty()) {
        std::string exe = argv[0];
        ui_path = exe.substr(0, exe.find_last_of("/\\") + 1) + "trilane-ui.html";
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    ggml_backend_load_all();

    trilane::engine eng;
    if (!eng.load(ep)) { fprintf(stderr, "[trilane] failed to load %s\n", ep.model.c_str()); return 1; }

    // --bench N: one-shot run through the real engine (no HTTP), prompt of ~N words on each lane, then a cached follow-up
    if (bench_words > 0) {
        std::string doc = "Summarize the following notes in three bullet points.\n\n";
        const char * words[] = { "the", "npu", "runs", "prompt", "prefill", "while", "cpu", "cores", "stream", "tokens", "from", "shared", "memory" };
        for (int i = 0; i < bench_words; i++) doc += std::string(words[(i * 7 + i / 13) % 13]) + ((i % 17 == 16) ? ".\n" : " ");
        auto show = [&](const char * tag, const trilane::gen_stats & s, const std::string & text) {
            printf("%-14s prompt %5d cached %5d | prefill %5d tok on %-4s %7.1f ms (%6.1f t/s) | TTFT %7.1f ms | decode %5.2f t/s on %s (%d tok) | llama_decode %5.2f t/s\n",
                   tag, s.n_prompt, s.n_cached, s.n_prefill, s.prefill_dev.c_str(), s.prefill_ms, s.prefill_ms > 0 ? s.n_prefill / (s.prefill_ms / 1000.0) : 0.0,
                   s.ttft_ms, s.decode_ms > 0 ? s.n_gen / (s.decode_ms / 1000.0) : 0.0, s.decode_dev.c_str(), s.n_gen,
                   s.decode_llama_ms > 0 ? s.n_gen / (s.decode_llama_ms / 1000.0) : 0.0);
            printf("               > %.100s\n", text.c_str());
            fflush(stdout);
        };
        std::vector<std::string> routes = { "split", "npu", "cpu" };
        if (!bench_routes.empty()) {
            routes.clear();
            std::stringstream ss(bench_routes);
            for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) routes.push_back(x);
        }
        for (const std::string & route_s : routes) {
            const char * route = route_s.c_str();
            if (!eng.split() && strcmp(route, "cpu") != 0) continue;
            if (eng.dense_aliased() && strcmp(route, "cpu") == 0) {
                printf("cpu            skipped: aliased dense weights make CPU batch prefill a per-token GEMV\n");
                continue;
            }
            trilane::gen_params gp; gp.max_tokens = bench_gen; gp.temperature = 0; gp.route = route;
            std::string out;
            auto st = eng.chat({ { "user", std::string("[") + route + "] " + doc } }, gp, [&](const std::string & p) { out += p; return true; });
            show(route, st, out);
            if (strcmp(route, "split") == 0 && bench_routes.empty()) {
                gp.route = "auto";
                std::string out2;
                auto st2 = eng.chat({ { "user", std::string("[") + route + "] " + doc }, { "assistant", out }, { "user", "Now give it a title." } }, gp,
                                    [&](const std::string & p) { out2 += p; return true; });
                show("followup/auto", st2, out2);
            }
        }
        fflush(stdout);
        return 0;
    }

    // --bench-concurrent N: N different prompts, first one after another, then all at once
    if (bench_conc > 0) {
        const char * words[] = { "the", "npu", "runs", "prompt", "prefill", "while", "cpu", "cores", "stream", "tokens", "from", "shared", "memory" };
        auto prompt = [&](int k) {
            std::string doc = "Request " + std::to_string(k) + ": summarize these notes in three bullet points.\n\n";
            const int nw = bench_words_conc + 37 * k; // different lengths
            for (int i = 0; i < nw; i++) doc += std::string(words[(i * 7 + k * 3 + i / 13) % 13]) + ((i % 17 == 16) ? ".\n" : " ");
            return doc;
        };
        auto run = [&](bool together) {
            std::vector<trilane::gen_stats> st(bench_conc);
            std::vector<std::string> txt(bench_conc);
            const double t0 = eng.now_ms();
            auto one = [&](int k) {
                trilane::gen_params gp; gp.max_tokens = bench_gen; gp.temperature = 0;
                st[k] = eng.chat({ { "user", prompt(k) } }, gp, [&, k](const std::string & p) { txt[k] += p; return true; });
            };
            if (together) {
                std::vector<std::thread> th;
                for (int k = 0; k < bench_conc; k++) th.emplace_back(one, k);
                for (auto & t : th) t.join();
            } else {
                for (int k = 0; k < bench_conc; k++) one(k);
            }
            const double wall = eng.now_ms() - t0;
            int gen = 0, pre = 0;
            for (int k = 0; k < bench_conc; k++) {
                auto & s = st[k];
                gen += s.n_gen; pre += s.n_prefill;
                printf("  #%-3lld prompt %5d on %-5s queue %7.0f ms | TTFT %7.0f ms | decode %5.1f t/s on %-8s (batch up to %d, slot %d) | %d tok (%d on NPU)\n",
                       (long long) s.id, s.n_prompt, s.route.c_str(), s.queue_ms, s.ttft_ms,
                       s.decode_ms > 0 ? s.n_gen / (s.decode_ms / 1000.0) : 0.0, s.decode_dev.c_str(), s.batch_peak, s.slot, s.n_gen, s.n_gen_npu);
                std::string head = txt[k].substr(0, 70);
                for (auto & ch : head) if (ch == '\n') ch = ' ';
                printf("        text[%d] %s\n", k, head.c_str());
            }
            printf("  => %s: wall %.1f s, %d prompt + %d generated tokens, aggregate %.1f generated t/s\n\n",
                   together ? "concurrent" : "sequential", wall / 1000.0, pre, gen, gen / (wall / 1000.0));
            fflush(stdout);
        };
        printf("=== %d requests, one after another\n", bench_conc); run(false);
        printf("=== %d requests at once\n", bench_conc); run(true);
        return 0;
    }

    httplib::Server svr;
    // no CORS headers: the dashboard is same-origin, and other web pages must not drive the local engine
#ifdef _WIN32
    svr.set_socket_options([](socket_t sock) {
        int yes = 1; // fail loudly instead of silently sharing a busy port
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &yes, sizeof(yes));
    });
#endif

    svr.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        std::ifstream f(ui_path, std::ios::binary);
        if (!f) { res.status = 404; res.set_content("UI not found: " + ui_path, "text/plain"); return; }
        std::stringstream ss; ss << f.rdbuf();
        res.set_content(ss.str(), "text/html; charset=utf-8");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(dump({ { "object", "list" }, { "data", json::array({ { { "id", eng.model_name() }, { "object", "model" }, { "owned_by", "trilane" } } }) } }), "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request &, httplib::Response & res) {
        json hist = json::array();
        for (auto & s : eng.history()) hist.push_back(stats_json(s));
        json live = json::array();
        for (auto & l : eng.live()) live.push_back({ { "id", l.id }, { "lane", l.lane }, { "phase", l.phase }, { "t0_ms", l.t0_ms } });
        const trilane::lane_cost cost = eng.cost();
        res.set_content(dump({
            { "model", eng.model_name() }, { "split", eng.split() }, { "aliased", eng.aliased() },
            { "prefill_dev", eng.prefill_dev() }, { "phase", eng.live_phase() }, { "now_ms", eng.now_ms() },
            { "max_seqs", eng.max_seqs() }, { "npu_decode_slots", eng.npu_decode_slots() }, { "active", eng.active_slots() }, { "npu_queue", eng.npu_queue_len() },
            { "keepwarm_pings", eng.keepwarm_pings() },
            { "router", { { "cpu_prefill_max", eng.cpu_prefill_max() }, { "cpu_fixed_ms", cost.cpu_fixed_ms }, { "cpu_tps", cost.cpu_tps }, { "npu_fixed_ms", cost.npu_fixed_ms }, { "npu_tps", cost.npu_tps }, { "cpu_corr", cost.cpu_corr }, { "npu_corr", cost.npu_corr } } },
            { "cache", { { "entries", eng.cache_entries() }, { "mb", eng.cache_bytes() / 1048576.0 } } },
            { "history", hist }, { "live", live },
        }), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content(R"({"error":"bad json"})", "application/json"); return; }
        std::vector<trilane::chat_msg> msgs;
        trilane::gen_params gp;
        bool stream = false;
        try {
            if (!body.is_object()) throw std::runtime_error("body must be a JSON object");
            for (auto & m : jv(body, "messages", json::array())) {
                std::string content;
                const json c = m.is_object() && m.contains("content") ? m["content"] : json();
                if (c.is_string()) content = c.get<std::string>();
                else if (c.is_array()) for (auto & part : c) if (part.is_object() && jv(part, "type", std::string()) == "text") content += jv(part, "text", std::string());
                msgs.push_back({ m.is_object() ? jv(m, "role", std::string("user")) : std::string("user"), content });
            }
            gp.max_tokens  = jv(body, "max_tokens", jv(body, "max_completion_tokens", gp.max_tokens)); // <= 0 means "as many as fit"
            gp.temperature = jv(body, "temperature", gp.temperature);
            gp.top_p       = jv(body, "top_p", gp.top_p);
            gp.seed        = jv(body, "seed", gp.seed);
            if (body.contains("trilane") && body["trilane"].is_object()) gp.route = jv(body["trilane"], "route", gp.route);
            stream = jv(body, "stream", false);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(dump({ { "error", { { "message", std::string("bad request: ") + e.what() }, { "type", "invalid_request_error" } } } }), "application/json");
            return;
        }
        if (msgs.empty()) {
            res.status = 400;
            res.set_content(dump({ { "error", { { "message", "messages is empty" }, { "type", "invalid_request_error" } } } }), "application/json");
            return;
        }
        const std::string model = eng.model_name();

        if (!stream) {
            std::string out;
            auto st = eng.chat(msgs, gp, [&](const std::string & p) { out += p; return !req.is_connection_closed(); });
            if (is_error(st.stop_reason)) {
                res.status = st.stop_reason == "context_overflow" ? 400 : 500;
                res.set_content(dump({ { "error", { { "message", error_text(st.stop_reason) }, { "type", st.stop_reason } } }, { "trilane", stats_json(st) } }), "application/json");
                return;
            }
            res.set_content(dump({
                { "id", "chatcmpl-" + std::to_string(st.id) }, { "object", "chat.completion" }, { "model", model },
                { "choices", json::array({ { { "index", 0 }, { "message", { { "role", "assistant" }, { "content", out } } },
                                             { "finish_reason", st.stop_reason == "stop" ? "stop" : "length" } } }) },
                { "usage", { { "prompt_tokens", st.n_prompt }, { "completion_tokens", st.n_gen }, { "total_tokens", st.n_prompt + st.n_gen } } },
                { "trilane", stats_json(st) },
            }), "application/json");
            return;
        }

        res.set_chunked_content_provider("text/event-stream", [&eng, msgs, gp, model](size_t, httplib::DataSink & sink) {
            auto send = [&](const json & j) { std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            auto chunk = [&](const json & delta, const json & finish) {
                return json{ { "object", "chat.completion.chunk" }, { "model", model },
                             { "choices", json::array({ { { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
            };
            send(chunk({ { "role", "assistant" } }, nullptr));
            auto st = eng.chat(msgs, gp, [&](const std::string & p) { return send(chunk({ { "content", p } }, nullptr)); });
            static const char done[] = "data: [DONE]\n\n";
            if (is_error(st.stop_reason)) {
                send({ { "error", { { "message", error_text(st.stop_reason) }, { "type", st.stop_reason } } }, { "trilane", stats_json(st) } });
                sink.write(done, sizeof(done) - 1);
                sink.done();
                return true;
            }
            json last = chunk(json::object(), st.stop_reason == "stop" ? "stop" : "length");
            last["usage"] = { { "prompt_tokens", st.n_prompt }, { "completion_tokens", st.n_gen } };
            last["trilane"] = stats_json(st);
            send(last);
            sink.write(done, sizeof(done) - 1);
            sink.done();
            return true;
        });
    });

    if (!svr.bind_to_port(host, port)) {
        fprintf(stderr, "[trilane] cannot listen on %s:%d (port in use?)\n", host.c_str(), port);
        return 1;
    }
    fprintf(stderr, "[trilane] serving %s on http://%s:%d  (UI: %s)\n", eng.model_name().c_str(), host.c_str(), port, ui_path.c_str());
    return svr.listen_after_bind() ? 0 : 1;
}
