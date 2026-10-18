// trilane-server: OpenAI-compatible chat server + live dashboard on top of the Tri-Lane engine.

#include "engine.h"

#include "ggml-backend.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using json = nlohmann::ordered_json;

static bool g_verbose = false;
static void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static json stats_json(const trilane::gen_stats & s) {
    return {
        { "id", s.id }, { "t_start_ms", s.t_start_ms },
        { "n_prompt", s.n_prompt }, { "n_cached", s.n_cached }, { "n_prefill", s.n_prefill },
        { "route", s.route }, { "prefill_dev", s.prefill_dev }, { "decode_dev", s.decode_dev },
        { "prefill_ms", s.prefill_ms }, { "prefill_tps", s.prefill_ms > 0 ? s.n_prefill / (s.prefill_ms / 1000.0) : 0.0 },
        { "handoff_ms", s.handoff_ms }, { "handoff_mb", s.handoff_mb }, { "ttft_ms", s.ttft_ms },
        { "decode_ms", s.decode_ms }, { "n_gen", s.n_gen },
        { "decode_tps", s.decode_ms > 0 ? s.n_gen / (s.decode_ms / 1000.0) : 0.0 },
        { "stop", s.stop_reason },
    };
}

static std::string dump(const json & j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

static void usage() {
    fprintf(stderr,
        "usage: trilane-server -m model.gguf [--prefill-dev HTP0|CPU] [--host 127.0.0.1] [--port 8080]\n"
        "       [-t decode_threads] [-c n_ctx] [-b n_batch] [-ub n_ubatch] [--keepwarm-ms 20] [--cpu-prefill-max 24] [--no-alias]\n"
        "       [--cache-mb 2048] [--ui path/to/index.html] [-v]\n");
    exit(1);
}

int main(int argc, char ** argv) {
    trilane::engine_params ep;
    std::string host = "127.0.0.1", ui_path;
    int port = 8080;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) usage(); return argv[++i]; };
        if      (k == "-m")                ep.model = next();
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

    httplib::Server svr;
    svr.set_default_headers({ { "Access-Control-Allow-Origin", "*" } });
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    });

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
        res.set_content(dump({
            { "model", eng.model_name() }, { "split", eng.split() }, { "aliased", eng.aliased() },
            { "prefill_dev", eng.prefill_dev() }, { "phase", eng.live_phase() }, { "now_ms", eng.now_ms() },
            { "keepwarm_pings", eng.keepwarm_pings() },
            { "router", { { "cpu_prefill_max", eng.cpu_prefill_max() }, { "cpu_fixed_ms", eng.cost().cpu_fixed_ms }, { "cpu_tps", eng.cost().cpu_tps }, { "npu_fixed_ms", eng.cost().npu_fixed_ms }, { "npu_tps", eng.cost().npu_tps }, { "cpu_corr", eng.cost().cpu_corr }, { "npu_corr", eng.cost().npu_corr } } },
            { "cache", { { "entries", eng.cache_entries() }, { "mb", eng.cache_bytes() / 1048576.0 } } },
            { "history", hist },
        }), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content(R"({"error":"bad json"})", "application/json"); return; }
        std::vector<trilane::chat_msg> msgs;
        for (auto & m : body.value("messages", json::array())) {
            std::string content;
            if (m["content"].is_string()) content = m["content"].get<std::string>();
            else if (m["content"].is_array()) for (auto & part : m["content"]) if (part.value("type", "") == "text") content += part.value("text", "");
            msgs.push_back({ m.value("role", "user"), content });
        }
        trilane::gen_params gp;
        gp.max_tokens  = body.value("max_tokens", body.value("max_completion_tokens", gp.max_tokens));
        gp.temperature = body.value("temperature", gp.temperature);
        gp.top_p       = body.value("top_p", gp.top_p);
        gp.seed        = body.value("seed", gp.seed);
        if (body.contains("trilane")) gp.route = body["trilane"].value("route", gp.route);
        const bool stream = body.value("stream", false);
        const std::string model = eng.model_name();

        if (!stream) {
            std::string out;
            auto st = eng.chat(msgs, gp, [&](const std::string & p) { out += p; return true; });
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
            json last = chunk(json::object(), st.stop_reason == "stop" ? "stop" : "length");
            last["usage"] = { { "prompt_tokens", st.n_prompt }, { "completion_tokens", st.n_gen } };
            last["trilane"] = stats_json(st);
            send(last);
            static const char done[] = "data: [DONE]\n\n";
            sink.write(done, sizeof(done) - 1);
            sink.done();
            return true;
        });
    });

    fprintf(stderr, "[trilane] serving %s on http://%s:%d  (UI: %s)\n", eng.model_name().c_str(), host.c_str(), port, ui_path.c_str());
    svr.listen(host, port);
    return 0;
}
