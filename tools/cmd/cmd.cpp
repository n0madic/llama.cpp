#include "cmd.h"

#include "common.h"
#include "preset.h"
#include "download.h"
#include "hf-cache.h"
#include "http.h"
#include "console.h"

#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
using json   = nlohmann::json;

//
// logging: silence ggml/llama INFO chatter (e.g. backend/device init) for CLI commands
// that initialize the library, while still surfacing real warnings and errors
//

void quiet_log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

void quiet_backend_logs() {
    llama_log_set(quiet_log_cb, nullptr);
    ggml_log_set(quiet_log_cb, nullptr);
}

//
// configuration / endpoint resolution
//

struct run_config {
    std::string host = "127.0.0.1";
    int         port = 8080; // matches the default of `llama serve`
    std::string api_key;     // optional bearer token, for talking to an authenticated router

    std::string base_url() const {
        return "http://" + host + ":" + std::to_string(port);
    }
};

// parse a TCP port, validating the 1..65535 range
int parse_port(const std::string & s) {
    char * end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v < 1 || v > 65535) {
        throw std::runtime_error("invalid port: '" + s + "'");
    }
    return (int) v;
}

// resolve host/port from environment (flags override these, see parse_endpoint_flags)
run_config default_run_config() {
    run_config cfg;
    if (const char * h = std::getenv("LLAMA_RUN_HOST"); h && *h) {
        cfg.host = h;
    }
    if (const char * p = std::getenv("LLAMA_RUN_PORT"); p && *p) {
        try {
            cfg.port = parse_port(p);
        } catch (const std::exception & e) {
            fprintf(stderr, "warning: %s (LLAMA_RUN_PORT); using default %d\n", e.what(), cfg.port);
        }
    }
    if (const char * k = std::getenv("LLAMA_RUN_API_KEY"); k && *k) {
        cfg.api_key = k;
    } else if (const char * k2 = std::getenv("LLAMA_API_KEY"); k2 && *k2) {
        cfg.api_key = k2;
    }
    return cfg;
}

// extract --host/--port (and -p) endpoint flags from argv[1..], returning the positional args.
// argv[0] is the subcommand name and is skipped. Option parsing stops at the first positional
// (or an explicit "--"), so trailing arguments -- e.g. a `run` prompt -- are never mistaken for
// endpoint flags (`llama run model -p ...` keeps "-p" as part of the prompt).
std::vector<std::string> parse_endpoint_flags(int argc, char ** argv, run_config & cfg) {
    std::vector<std::string> pos;
    auto next_value = [&](int & i, const char * name) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + name);
        }
        return argv[++i];
    };
    int i = 1;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--host") {
            cfg.host = next_value(i, "--host");
        } else if (a == "--port" || a == "-p") {
            cfg.port = parse_port(next_value(i, a.c_str()));
        } else if (a == "--api-key") {
            cfg.api_key = next_value(i, "--api-key");
        } else if (a == "--") {
            ++i; // explicit end-of-options
            break;
        } else {
            break; // first positional: stop option parsing
        }
    }
    for (; i < argc; ++i) {
        pos.push_back(argv[i]);
    }
    return pos;
}

//
// cache / state paths (app cache, NOT the HF hub cache)
//

fs::path run_dir() {
    return fs::path(fs_get_cache_directory()) / "run";
}

fs::path default_preset_path() {
    return run_dir() / "models.ini";
}

fs::path daemon_log_path() {
    return run_dir() / "server.log";
}

fs::path daemon_state_path() {
    return run_dir() / "daemon.json";
}

void ensure_run_dir() {
    std::error_code ec;
    fs::create_directories(run_dir(), ec);
}

// the router crashes at startup if --models-preset points at a missing file, so make sure
// one exists. A comment-only file yields zero presets (no phantom models).
void ensure_default_preset() {
    ensure_run_dir();
    const fs::path path = default_preset_path();
    std::error_code ec;
    if (fs::exists(path, ec)) {
        return;
    }
    std::ofstream f(path);
    if (f) {
        f << "; llama.cpp model presets (managed by 'llama create')\n";
        f << "; each [section] is a model preset; see: llama serve --help (--models-preset)\n";
    }
}

//
// HTTP helpers (talk to the router daemon over plain HTTP on localhost)
//

httplib::Client make_client(const run_config & cfg, int read_timeout_s) {
    auto [cli, parts] = common_http_client(cfg.base_url());
    (void) parts;
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(read_timeout_s, 0);
    if (!cfg.api_key.empty()) {
        cli.set_bearer_token_auth(cfg.api_key); // authenticate against a router started with --api-key
    }
    return std::move(cli);
}

json http_get_json(const run_config & cfg, const std::string & path, int read_timeout_s = 10) {
    auto cli = make_client(cfg, read_timeout_s);
    auto res = cli.Get(path);
    if (!res) {
        throw std::runtime_error("request failed: " + httplib::to_string(res.error()));
    }
    if (res->status != 200) {
        throw std::runtime_error("HTTP " + std::to_string(res->status) + ": " + res->body);
    }
    return json::parse(res->body);
}

// extract a human-readable message from a parsed {"error": ...} object; empty string if absent
std::string error_message_from(const json & j) {
    if (j.is_object() && j.contains("error")) {
        return j["error"].is_object() ? j["error"].value("message", j["error"].dump()) : j["error"].dump();
    }
    return {};
}

// pull a human-readable message out of a llama-server JSON error body, falling back to `raw`
std::string error_message_from(const std::string & raw) {
    json j = json::parse(raw, nullptr, false);
    if (!j.is_discarded()) {
        const std::string msg = error_message_from(j);
        if (!msg.empty()) {
            return msg;
        }
    }
    return raw;
}

// refresh the router's model list (GET /models?reload=1 reloads when "reload" is non-empty).
// The reload runs synchronously in the router, so surface a failure (e.g. a malformed preset)
// instead of dropping it silently. Non-fatal: the command that triggered the reload still ran.
void reload_daemon_models(const run_config & cfg) {
    try {
        auto cli = make_client(cfg, 60); // a reload may (re)load a model, which can take a while
        auto res = cli.Get("/models?reload=1");
        if (!res) {
            fprintf(stderr, "warning: could not refresh daemon model list: %s\n",
                    httplib::to_string(res.error()).c_str());
            return;
        }
        if (res->status != 200) {
            fprintf(stderr, "warning: daemon failed to reload models: %s\n",
                    error_message_from(res->body).c_str());
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "warning: could not refresh daemon model list: %s\n", e.what());
    }
}

//
// daemon lifecycle
//

// returns true if a router-mode daemon is reachable at cfg
bool daemon_is_router(const run_config & cfg) {
    try {
        auto cli = make_client(cfg, 3);
        auto health = cli.Get("/health");
        if (!health || health->status != 200) {
            return false;
        }
        auto props = cli.Get("/props");
        if (!props || props->status != 200) {
            return false;
        }
        auto j = json::parse(props->body, nullptr, false);
        return j.is_object() && j.value("role", std::string()) == "router";
    } catch (...) {
        return false;
    }
}

std::vector<std::string> daemon_argv(const fs::path & self, const run_config & cfg) {
    return {
        self.string(),
        "serve",
        "--host",           cfg.host,
        "--port",           std::to_string(cfg.port),
        "--models-preset",  default_preset_path().string(),
        "--models-max",     "4",
        "--models-autoload",
        // enable Jinja chat templates so OpenAI-style tool/function calling works out of the box
        // (propagated to every model instance via the router's base preset)
        "--jinja",
    };
}

#if defined(_WIN32)
// append `arg` to a Windows command line, quoting/escaping per the CommandLineToArgvW rules:
// backslashes are literal unless they precede a double quote, and embedded quotes are escaped.
// A naive '"' + arg + '"' breaks on values containing quotes or a trailing backslash (e.g. a
// path like C:\dir\). ref: "Everyone quotes command line arguments the wrong way" (D. Colascione)
void win_append_quoted_arg(std::string & cmdline, const std::string & arg) {
    if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
        cmdline += arg; // no metacharacters: pass through unquoted
        return;
    }
    cmdline += '"';
    for (auto it = arg.begin(); ; ++it) {
        unsigned backslashes = 0;
        while (it != arg.end() && *it == '\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            // escape the backslashes so the closing quote stays a metacharacter
            cmdline.append(backslashes * 2, '\\');
            break;
        }
        if (*it == '"') {
            // escape the backslashes and the embedded quote
            cmdline.append(backslashes * 2 + 1, '\\');
            cmdline += '"';
        } else {
            cmdline.append(backslashes, '\\'); // backslashes aren't special here
            cmdline += *it;
        }
    }
    cmdline += '"';
}

// a spawned daemon process we can poll for early (startup) death
struct daemon_proc {
    HANDLE handle = INVALID_HANDLE_VALUE;
    bool valid() const { return handle != INVALID_HANDLE_VALUE; }
};

bool daemon_proc_exited(daemon_proc & p) {
    return p.valid() && WaitForSingleObject(p.handle, 0) == WAIT_OBJECT_0;
}

void daemon_proc_release(daemon_proc & p) {
    if (p.valid()) {
        CloseHandle(p.handle);
        p.handle = INVALID_HANDLE_VALUE;
    }
}

daemon_proc spawn_daemon(const run_config & cfg) {
    fs::path self = fs_get_exe_path();
    ensure_default_preset();

    // build a properly quoted command line (argv[0] included)
    std::string cmdline;
    for (const auto & a : daemon_argv(self, cfg)) {
        if (!cmdline.empty()) {
            cmdline += ' ';
        }
        win_append_quoted_arg(cmdline, a);
    }

    HANDLE log = CreateFileA(daemon_log_path().string().c_str(),
                             FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    if (log != INVALID_HANDLE_VALUE) {
        si.dwFlags    = STARTF_USESTDHANDLES;
        si.hStdOutput = log;
        si.hStdError  = log;
        SetHandleInformation(log, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    }

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    std::vector<char> mutable_cmd(cmdline.begin(), cmdline.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(self.string().c_str(), mutable_cmd.data(), nullptr, nullptr,
                             /*inherit*/ (log != INVALID_HANDLE_VALUE),
                             DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si, &pi);
    if (log != INVALID_HANDLE_VALUE) {
        CloseHandle(log);
    }
    if (!ok) {
        return {};
    }
    CloseHandle(pi.hThread);
    return daemon_proc{ pi.hProcess }; // keep the process handle so startup death is detectable
}
#else
// a spawned daemon process we can poll for early (startup) death
struct daemon_proc {
    pid_t pid = -1;
    bool valid() const { return pid > 0; }
};

bool daemon_proc_exited(daemon_proc & p) {
    if (!p.valid()) {
        return false;
    }
    int status = 0;
    return waitpid(p.pid, &status, WNOHANG) == p.pid; // reaps the child if it has already exited
}

void daemon_proc_release(daemon_proc & /*proc*/) {
    // the daemon is detached (setsid) and intentionally outlives this process; nothing to close
}

daemon_proc spawn_daemon(const run_config & cfg) {
    fs::path self = fs_get_exe_path();
    ensure_default_preset();

    // build argv (and copy the strings the child needs) BEFORE forking: between fork() and
    // execv() the child must avoid heap allocation, which is not async-signal-safe.
    const std::string self_str = self.string();
    const std::string log_str  = daemon_log_path().string();
    std::vector<std::string> args = daemon_argv(self, cfg);
    std::vector<char *>       cargv;
    cargv.reserve(args.size() + 1);
    for (auto & a : args) {
        cargv.push_back(const_cast<char *>(a.c_str()));
    }
    cargv.push_back(nullptr);

    // self-pipe: the child writes errno here if execv fails. The write end is close-on-exec, so a
    // successful exec closes it and the parent reads EOF (= success); a failed exec yields an errno.
    int pfd[2];
    if (pipe(pfd) != 0) {
        return {};
    }
    fcntl(pfd[1], F_SETFD, FD_CLOEXEC);

    pid_t pid = fork();
    if (pid < 0) {
        close(pfd[0]);
        close(pfd[1]);
        return {};
    }
    if (pid == 0) {
        // detached child: new session, redirect stdio to the log file
        close(pfd[0]);
        setsid();

        int fd = open(log_str.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            if (fd > STDERR_FILENO) {
                close(fd);
            }
        }
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            if (devnull > STDERR_FILENO) {
                close(devnull);
            }
        }

        execv(self_str.c_str(), cargv.data());
        // execv only returns on error: report errno to the parent, then exit
        const int err = errno;
        ssize_t w = write(pfd[1], &err, sizeof(err));
        (void) w;
        _exit(127);
    }

    // parent: EOF means exec succeeded; any bytes mean execv failed and reported its errno
    close(pfd[1]);
    int child_errno = 0;
    ssize_t n = read(pfd[0], &child_errno, sizeof(child_errno));
    close(pfd[0]);
    if (n > 0) {
        fprintf(stderr, "error: failed to exec daemon '%s': %s\n", self_str.c_str(), strerror(child_errno));
        return {};
    }
    return daemon_proc{ pid }; // parent: keep the child pid so startup death is detectable
}
#endif

void write_daemon_state(const run_config & cfg) {
    try {
        ensure_run_dir();
        json state = {
            {"host", cfg.host},
            {"port", cfg.port},
            {"url",  cfg.base_url()},
        };
        std::ofstream f(daemon_state_path());
        if (f) {
            f << state.dump(2) << "\n";
        }
    } catch (...) {
        // state file is informational only; /health is the source of truth
    }
}

// true if anything answers an HTTP request at cfg (used to tell "port free" from "port taken")
bool port_has_server(const run_config & cfg) {
    try {
        auto cli = make_client(cfg, 2);
        return (bool) cli.Get("/health");
    } catch (...) {
        return false;
    }
}

// ensure a router daemon is running; spawn one (detached) if needed and poll until ready
bool ensure_daemon(const run_config & cfg) {
    if (daemon_is_router(cfg)) {
        return true;
    }
    // something already answers here, but it is not our router: spawning a second serve on the
    // same host:port would only fail to bind, so report it instead of polling for ~30s.
    if (port_has_server(cfg)) {
        fprintf(stderr, "error: %s is already in use by a non-router server; "
                        "stop it or pick another port (--port / LLAMA_RUN_PORT)\n",
                cfg.base_url().c_str());
        return false;
    }
    fprintf(stderr, "starting llama daemon on %s ...\n", cfg.base_url().c_str());
    daemon_proc proc = spawn_daemon(cfg);
    if (!proc.valid()) {
        fprintf(stderr, "error: failed to spawn the llama daemon\n");
        return false;
    }

    // poll /health until the router is ready (cold start is fast: no model loaded yet), but bail
    // out early if the spawned process dies during startup instead of waiting the full timeout
    for (int i = 0; i < 300; ++i) { // up to ~30s
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (daemon_is_router(cfg)) {
            write_daemon_state(cfg); // record state only once the daemon is actually serving
            daemon_proc_release(proc);
            return true;
        }
        if (daemon_proc_exited(proc)) {
            fprintf(stderr, "error: the llama daemon exited during startup; see %s\n",
                    daemon_log_path().string().c_str());
            daemon_proc_release(proc);
            return false;
        }
    }
    daemon_proc_release(proc);
    fprintf(stderr, "error: daemon did not become ready; see %s\n", daemon_log_path().string().c_str());
    return false;
}

//
// formatting helpers
//

std::string format_size(uint64_t bytes) {
    static const char * units[] = {"B", "KB", "MB", "GB", "TB"};
    double size = (double) bytes;
    int    unit = 0;
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        ++unit;
    }
    char buf[32];
    if (unit == 0) {
        snprintf(buf, sizeof(buf), "%llu %s", (unsigned long long) bytes, units[unit]);
    } else {
        snprintf(buf, sizeof(buf), "%.1f %s", size, units[unit]);
    }
    return buf;
}

// split a "repo:tag" model id into its repo part (drops the trailing tag, if any)
std::string repo_of(const std::string & model_id) {
    const auto pos = model_id.rfind(':');
    if (pos == std::string::npos) {
        return model_id;
    }
    // a tag never contains '/', whereas a repo always does - guard against ports/schemes
    if (model_id.find('/', pos) != std::string::npos) {
        return model_id;
    }
    return model_id.substr(0, pos);
}

std::string tag_of(const std::string & model_id) {
    const std::string repo = repo_of(model_id);
    if (repo.size() < model_id.size()) {
        return model_id.substr(repo.size() + 1);
    }
    return {};
}

// format a unix timestamp (seconds) as a local "YYYY-MM-DD HH:MM"; empty string for 0/unknown
std::string format_mtime(int64_t unix_seconds) {
    if (unix_seconds <= 0) {
        return {};
    }
    const std::time_t t = (std::time_t) unix_seconds;
    std::tm tm = {};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", &tm);
    return buf;
}

std::string format_params(uint64_t n) {
    char buf[32];
    if (n >= 1000000000ULL) {
        snprintf(buf, sizeof(buf), "%.2f B", (double) n / 1e9);
    } else if (n >= 1000000ULL) {
        snprintf(buf, sizeof(buf), "%.2f M", (double) n / 1e6);
    } else if (n >= 1000ULL) {
        snprintf(buf, sizeof(buf), "%.2f K", (double) n / 1e3);
    } else {
        snprintf(buf, sizeof(buf), "%llu", (unsigned long long) n);
    }
    return buf;
}

// locate the cached GGUF file for a model id (repo:tag); empty if not cached.
// Uses the same resolver as `pull`, so `show` reads the same file the model would load.
std::string find_cached_gguf(const std::string & model_id) {
    const auto files = common_cached_model_files(model_id);
    return files.empty() ? std::string() : files.front();
}

// model metadata read directly from a GGUF file (offline, no backend init)
struct gguf_model_info {
    bool        ok = false;
    std::string arch;
    std::string name;
    std::string size_label;
    std::string license;
    uint64_t    n_params    = 0;
    uint32_t    n_ctx_train = 0;
    uint32_t    n_embd      = 0;
    uint32_t    n_layers    = 0;
    bool        has_chat_template = false;
    uint64_t    file_size   = 0;
    std::string path;
};

gguf_model_info read_gguf_meta(const std::string & path) {
    gguf_model_info info;
    info.path = path;

    struct ggml_context * meta = nullptr;
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    struct gguf_context * gg = gguf_init_from_file(path.c_str(), params);
    if (gg == nullptr) {
        return info; // ok stays false
    }

    auto get_str = [&](const std::string & key) -> std::string {
        const int64_t id = gguf_find_key(gg, key.c_str());
        if (id < 0 || gguf_get_kv_type(gg, id) != GGUF_TYPE_STRING) {
            return {};
        }
        const char * v = gguf_get_val_str(gg, id);
        return v ? std::string(v) : std::string();
    };
    auto get_u32 = [&](const std::string & key) -> uint32_t {
        const int64_t id = gguf_find_key(gg, key.c_str());
        if (id < 0) {
            return 0;
        }
        switch (gguf_get_kv_type(gg, id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(gg, id);
            case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(gg, id);
            case GGUF_TYPE_UINT64: { // saturate instead of silently wrapping a >32-bit value
                const uint64_t v = gguf_get_val_u64(gg, id);
                return v > UINT32_MAX ? UINT32_MAX : (uint32_t) v;
            }
            case GGUF_TYPE_UINT16: return gguf_get_val_u16(gg, id);
            default:               return 0;
        }
    };

    info.arch       = get_str("general.architecture");
    info.name       = get_str("general.name");
    info.size_label = get_str("general.size_label");
    info.license    = get_str("general.license");
    if (!info.arch.empty()) {
        info.n_ctx_train = get_u32(info.arch + ".context_length");
        info.n_embd      = get_u32(info.arch + ".embedding_length");
        info.n_layers    = get_u32(info.arch + ".block_count");
    }
    info.has_chat_template = gguf_find_key(gg, "tokenizer.chat_template") >= 0;

    if (meta != nullptr) {
        for (ggml_tensor * t = ggml_get_first_tensor(meta); t != nullptr; t = ggml_get_next_tensor(meta, t)) {
            info.n_params += (uint64_t) ggml_nelements(t);
        }
    }

    std::error_code ec;
    info.file_size = fs::exists(path, ec) ? (uint64_t) fs::file_size(path, ec) : 0;
    info.ok = true;

    gguf_free(gg);
    ggml_free(meta);
    return info;
}

//
// model resolution against the router's model list
//

struct model_entry {
    std::string id;
    std::string status;   // unloaded | loading | loaded | sleeping
    std::string size_repo; // repo:tag used to measure on-disk size (id itself, or the preset's hf-repo)
    std::vector<std::string> aliases; // user-facing names the router also routes this model by
};

std::vector<model_entry> fetch_models(const run_config & cfg) {
    std::vector<model_entry> out;
    json j = http_get_json(cfg, "/v1/models");
    if (!j.contains("data") || !j["data"].is_array()) {
        return out;
    }
    for (const auto & m : j["data"]) {
        model_entry e;
        e.id = m.value("id", std::string());
        if (m.contains("aliases") && m["aliases"].is_array()) {
            for (const auto & a : m["aliases"]) {
                if (a.is_string()) {
                    e.aliases.push_back(a.get<std::string>());
                }
            }
        }
        if (m.contains("status") && m["status"].is_object()) {
            e.status = m["status"].value("value", std::string());
            // a crashed child is reported as "unloaded" with failed=true: surface it as an error
            // so `llama list` does not show it as a healthy never-loaded model
            if (m["status"].value("failed", false)) {
                const int exit_code = m["status"].value("exit_code", 0);
                e.status = exit_code != 0 ? "error (exit " + std::to_string(exit_code) + ")" : "error";
            }
        }
        // size is measured against the HF repo. For cached models the id already is repo:tag;
        // for custom presets (a bare name) dig the hf-repo out of the rendered args.
        e.size_repo = e.id;
        if (e.id.find('/') == std::string::npos &&
            m.contains("status") && m["status"].contains("args") && m["status"]["args"].is_array()) {
            const auto & args = m["status"]["args"];
            for (size_t i = 0; i + 1 < args.size(); ++i) {
                if (args[i].is_string() && args[i].get<std::string>() == "--hf-repo" && args[i + 1].is_string()) {
                    e.size_repo = args[i + 1].get<std::string>();
                    break;
                }
            }
        }
        if (!e.id.empty()) {
            out.push_back(std::move(e));
        }
    }
    return out;
}

// resolve a user-supplied name (with or without tag) to a router model id.
// returns an empty string if not found.
std::string resolve_model_id(const run_config & cfg, const std::string & input) {
    const auto models = fetch_models(cfg);
    // pass 1: exact match on the canonical id or any alias the router routes this model by
    for (const auto & m : models) {
        if (m.id == input || std::find(m.aliases.begin(), m.aliases.end(), input) != m.aliases.end()) {
            return m.id;
        }
    }
    // pass 2: tag-less prefix (e.g. "owner/model" -> first cached quant "owner/model:Q4_K_M")
    if (input.find(':') == std::string::npos) {
        for (const auto & m : models) {
            if (m.id.rfind(input + ":", 0) == 0) {
                return m.id; // first matching quant
            }
        }
    }
    return {};
}

// fetch the full /v1/models entry for a user-supplied name in a single request, matching the
// canonical id, an alias, or a tag-less id prefix. empty json on failure / not found.
json fetch_model_entry(const run_config & cfg, const std::string & input) {
    try {
        json j = http_get_json(cfg, "/v1/models");
        if (!j.contains("data") || !j["data"].is_array()) {
            return json();
        }
        const auto & data = j["data"];
        // pass 1: exact match on the canonical id or any alias
        for (const auto & m : data) {
            if (m.value("id", std::string()) == input) {
                return m;
            }
            if (m.contains("aliases") && m["aliases"].is_array()) {
                for (const auto & a : m["aliases"]) {
                    if (a.is_string() && a.get<std::string>() == input) {
                        return m;
                    }
                }
            }
        }
        // pass 2: tag-less prefix (first matching quant)
        if (input.find(':') == std::string::npos) {
            for (const auto & m : data) {
                if (m.value("id", std::string()).rfind(input + ":", 0) == 0) {
                    return m;
                }
            }
        }
    } catch (const std::exception &) {
        // daemon unreachable / bad response - caller falls back to other sources
    }
    return json();
}

// poll the router until a model is no longer running, so we never delete a GGUF that a live
// instance still has mmap'd. Returns true once released (or daemon gone), false on timeout.
bool wait_until_unloaded(const run_config & cfg, const std::string & model_id, int timeout_s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    for (;;) {
        bool running = false;
        try {
            for (const auto & m : fetch_models(cfg)) {
                if (m.id == model_id) {
                    // loading/loaded/sleeping => child alive, may hold the file; anything else => released
                    running = (m.status == "loading" || m.status == "loaded" || m.status == "sleeping");
                    break;
                }
            }
        } catch (const std::exception &) {
            return true; // daemon unreachable: nothing holds the file, safe to proceed
        }
        if (!running) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// scan an INI file, invoking on_line(raw_line, in_target_section) for each line; in_target_section
// is true for the "[name]" header line and every line until the next section header. Returns true
// if the section was present. Deliberately lightweight/backend-free: the canonical parser in
// common/preset.cpp needs a preset context (arg-table + device init) that is too heavy for these
// read / edit-in-place paths, so the section scan is shared here instead.
template <typename F>
bool scan_ini(const fs::path & file, const std::string & name, F && on_line) {
    std::error_code ec;
    if (!fs::exists(file, ec)) {
        return false;
    }
    std::ifstream in(file);
    if (!in) {
        return false;
    }
    const std::string header = "[" + name + "]";
    std::string line;
    bool in_section = false;
    bool found = false;
    while (std::getline(in, line)) {
        const std::string t = string_strip(line);
        if (!t.empty() && t.front() == '[') {
            in_section = (t == header);
            if (in_section) {
                found = true;
            }
        }
        on_line(line, in_section);
    }
    return found;
}

// extract the raw "[name] ... " section text from an INI file
std::string read_preset_section(const fs::path & file, const std::string & name) {
    std::string out;
    scan_ini(file, name, [&](const std::string & line, bool in_section) {
        if (in_section) {
            out += line + "\n";
        }
    });
    return out;
}

// get the value of `key` from a preset INI block
std::string ini_value(const std::string & block, const std::string & key) {
    std::istringstream ss(block);
    std::string line;
    while (std::getline(ss, line)) {
        const std::string t = string_strip(line);
        const auto eq = t.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        if (string_strip(t.substr(0, eq)) == key) {
            return string_strip(t.substr(eq + 1));
        }
    }
    return {};
}

// remove the "[name]" section from an INI file in place (text-only, preserves other sections,
// the [*] global and comments). Returns true if the section existed and was removed.
bool remove_preset_section(const fs::path & file, const std::string & name) {
    std::vector<std::string> kept;
    const bool removed = scan_ini(file, name, [&](const std::string & line, bool in_section) {
        if (!in_section) {
            kept.push_back(line); // keep everything outside the target section
        }
    });
    if (!removed) {
        return false;
    }
    std::ofstream out(file, std::ios::trunc);
    if (!out) {
        return false;
    }
    for (const auto & l : kept) {
        out << l << "\n";
    }
    return true;
}

//
// download progress bar (used by `pull` and implicit pulls in `run`)
//

class pull_progress : public common_download_callback {
public:
    void on_start(const common_download_progress & p) override {
        last_file = short_name(p.url);
        if (p.cached) {
            fprintf(stderr, "  %s: already cached\n", last_file.c_str());
            printed_bar = false;
        } else {
            printed_bar = true;
        }
    }

    void on_update(const common_download_progress & p) override {
        if (!printed_bar) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_draw < std::chrono::milliseconds(100)) {
            return; // throttle redraws
        }
        last_draw = now;
        draw(p);
    }

    void on_done(const common_download_progress & p, bool ok) override {
        if (printed_bar) {
            draw(p);
            fprintf(stderr, "\n");
        }
        if (!ok) {
            fprintf(stderr, "  %s: download failed\n", last_file.c_str());
        }
    }

private:
    std::string                           last_file;
    bool                                  printed_bar = false;
    std::chrono::steady_clock::time_point last_draw;

    static std::string short_name(const std::string & url) {
        const auto pos = url.find_last_of('/');
        std::string name = pos == std::string::npos ? url : url.substr(pos + 1);
        const auto q = name.find('?');
        if (q != std::string::npos) {
            name = name.substr(0, q);
        }
        return name;
    }

    void draw(const common_download_progress & p) const {
        if (p.total > 0) {
            const double frac = std::min(1.0, (double) p.downloaded / (double) p.total);
            const int    pct  = (int) std::lround(frac * 100.0);
            const int    width = 24;
            const int    fill  = (int) std::lround(frac * width);
            std::string bar(fill, '=');
            bar.resize(width, ' ');
            fprintf(stderr, "\r  %s [%s] %3d%% (%s / %s)   ",
                    last_file.c_str(), bar.c_str(), pct,
                    format_size(p.downloaded).c_str(), format_size(p.total).c_str());
        } else {
            fprintf(stderr, "\r  %s %s   ", last_file.c_str(), format_size(p.downloaded).c_str());
        }
        fflush(stderr);
    }
};

// download a model by repo[:tag]; returns true on success
bool do_pull(const std::string & repo_with_tag) {
    common_params_model m;
    m.hf_repo = repo_with_tag;

    pull_progress cb;
    common_download_opts opts;
    opts.callback = &cb;

    fprintf(stderr, "pulling %s\n", repo_with_tag.c_str());
    common_download_model_result res;
    try {
        res = common_download_model(m, opts);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return false;
    }
    if (res.model_path.empty()) {
        fprintf(stderr, "error: failed to download '%s'\n", repo_with_tag.c_str());
        return false;
    }
    return true;
}

//
// chat streaming over /v1/chat/completions (stream:true)
//

// stream a chat completion; appends streamed content to out_reply. returns true on success.
bool stream_chat(const run_config & cfg, const json & body, std::string & out_reply) {
    auto cli = make_client(cfg, 600); // cold autoload can block the first response for a while

    std::string sse_buf;
    std::string error_msg;
    std::string err_body; // bounded copy of the body head (httplib leaves res->body empty here)
    bool        any_content = false;

    auto receiver = [&](const char * data, size_t len) -> bool {
        // recover an error message for a non-200 response: capture the head until real SSE
        // content starts (a successful stream then stops growing this buffer)
        if (!any_content && err_body.size() < 65536) {
            err_body.append(data, std::min(len, (size_t) 65536 - err_body.size()));
        }
        sse_buf.append(data, len);
        size_t nl;
        while ((nl = sse_buf.find('\n')) != std::string::npos) {
            std::string line = sse_buf.substr(0, nl);
            sse_buf.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.rfind("data: ", 0) != 0) {
                continue;
            }
            const std::string payload = line.substr(6);
            if (payload == "[DONE]") {
                return true;
            }
            json ev = json::parse(payload, nullptr, false);
            if (ev.is_discarded()) {
                continue;
            }
            if (ev.contains("error")) {
                error_msg = error_message_from(ev);
                continue;
            }
            if (ev.contains("choices") && ev["choices"].is_array() && !ev["choices"].empty()) {
                const auto & delta = ev["choices"][0]["delta"];
                if (delta.is_object() && delta.contains("content") && delta["content"].is_string()) {
                    const std::string chunk = delta["content"].get<std::string>();
                    if (!chunk.empty()) {
                        out_reply += chunk;
                        fputs(chunk.c_str(), stdout);
                        fflush(stdout);
                        any_content = true;
                    }
                }
            }
        }
        return true;
    };

    auto res = cli.Post("/v1/chat/completions", httplib::Headers{}, body.dump(), "application/json", receiver);
    if (!res) {
        fprintf(stderr, "\nerror: request failed: %s\n", httplib::to_string(res.error()).c_str());
        return false;
    }
    if (!error_msg.empty()) {
        // error delivered as a mid-stream SSE "error" event
        fprintf(stderr, "\nerror: %s\n", error_msg.c_str());
        return false;
    }
    if (res->status != 200) {
        // any non-200 is a failure (even if some content streamed before the error): httplib
        // routes the body to the receiver, so res->body is empty -- use the captured head
        const std::string raw = !res->body.empty() ? res->body : err_body;
        fprintf(stderr, "\nerror: %s\n", error_message_from(raw).c_str());
        return false;
    }
    return true;
}

int run_single_shot(const run_config & cfg, const std::string & model_id, const std::string & prompt) {
    json body = {
        {"model",    model_id},
        {"messages", json::array({ {{"role", "user"}, {"content", prompt}} })},
        {"stream",   true},
    };
    std::string reply;
    const bool ok = stream_chat(cfg, body, reply);
    fputs("\n", stdout);
    return ok ? 0 : 1;
}

int run_interactive(const run_config & cfg, const std::string & model_id) {
    printf("Chatting with %s. Type /exit to quit, /clear to reset the conversation.\n\n", model_id.c_str());

    // Use rich line editing only on an interactive TTY; fall back to plain line reads for
    // piped/redirected stdin (the advanced path puts the terminal in raw mode and would
    // otherwise misbehave on a non-tty).
#if defined(_WIN32)
    const bool stdin_is_tty = _isatty(_fileno(stdin)) != 0;
#else
    const bool stdin_is_tty = isatty(STDIN_FILENO) != 0;
#endif
    console::init(/*use_simple_io*/ !stdin_is_tty, /*use_advanced_display*/ false);
    struct console_guard {
        ~console_guard() { console::cleanup(); }
    } guard;

    // tell a real EOF (Ctrl-D / end of piped stdin) apart from console::readline returning
    // control on a lone '/', which also yields an empty buffer
    auto stdin_at_eof = []() -> bool {
#if defined(_WIN32)
        return std::cin.eof() || std::wcin.eof() || feof(stdin) != 0;
#else
        return std::cin.eof() || feof(stdin) != 0;
#endif
    };

    json messages = json::array();

    while (true) {
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        printf("> ");
        fflush(stdout);

        // console::readline returns the multiline-continuation flag (NOT a have-line/EOF
        // signal): false means "this line is complete", true means "keep appending". Each
        // completed line contributes at least a trailing '\n', so an entirely empty
        // accumulated buffer is how EOF (Ctrl-D / end of piped input) surfaces.
        std::string buffer;
        bool more = true;
        do {
            std::string line;
            more = console::readline(line, false);
            buffer += line;
        } while (more);
        console::set_display(DISPLAY_TYPE_RESET);

        if (buffer.empty()) {
            if (stdin_at_eof()) { // real end of input: end the session
                printf("\n");
                break;
            }
            continue; // empty input (e.g. a lone '/' returns control with no text): reprompt
        }
        while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r')) {
            buffer.pop_back();
        }

        const std::string trimmed = string_strip(buffer);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed == "/exit" || trimmed == "/quit" || trimmed == "/bye") {
            break;
        }
        if (trimmed == "/clear") {
            messages = json::array();
            printf("(conversation cleared)\n\n");
            continue;
        }

        messages.push_back({{"role", "user"}, {"content", trimmed}});

        json body = {
            {"model",    model_id},
            {"messages", messages},
            {"stream",   true},
        };

        std::string reply;
        stream_chat(cfg, body, reply);
        if (reply.empty()) {
            // nothing reached the screen (failed, or empty/tool-only response): drop the user turn
            messages.erase(messages.end() - 1);
            continue;
        }
        // content was printed (fully, or partially before an error): keep history in sync with it
        printf("\n\n");
        messages.push_back({{"role", "assistant"}, {"content", reply}});
    }

    return 0;
}

//
// preset (Modelfile) file read/write for `create` / `show`
//

void write_presets_file(const fs::path & path, const common_presets & presets, const common_preset & global) {
    ensure_run_dir();

    // write atomically (temp file + rename) so a crash mid-write can't truncate models.ini and
    // lose every existing preset
    std::ostringstream body;
    body << "; llama.cpp model presets (managed by 'llama create')\n";
    body << "; each [section] is a model preset; see: llama serve --help (--models-preset)\n\n";
    if (!global.options.empty()) {
        body << global.to_ini();
    }
    for (const auto & [name, preset] : presets) {
        (void) name;
        body << preset.to_ini();
    }

    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            throw std::runtime_error("failed to write preset file: " + tmp.string());
        }
        f << body.str();
        f.flush();
        if (!f) {
            throw std::runtime_error("failed to write preset file: " + tmp.string());
        }
    }

    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec); // best-effort cleanup; the original file is left intact
        throw std::runtime_error("failed to replace preset file: " + path.string());
    }
}

//
// model table rendering (shared by `list` and `ps`)
//

struct model_row {
    std::string name;
    std::string status;
    std::string size;
    std::string modified; // empty -> the MODIFIED column is omitted
};

// print NAME / SIZE / [MODIFIED] / STATUS columns with auto-sized widths. The MODIFIED column is
// shown only when at least one row carries a modified time (e.g. `list`, but not `ps`).
void print_model_table(const std::vector<model_row> & rows) {
    size_t w_name = 4; // strlen("NAME")
    size_t w_size = 4; // strlen("SIZE")
    size_t w_mod  = 8; // strlen("MODIFIED")
    bool   show_mod = false;
    for (const auto & r : rows) {
        w_name = std::max(w_name, r.name.size());
        w_size = std::max(w_size, r.size.size());
        if (!r.modified.empty()) {
            show_mod = true;
            w_mod   = std::max(w_mod, r.modified.size());
        }
    }
    if (show_mod) {
        printf("%-*s  %-*s  %-*s  %s\n",
               (int) w_name, "NAME", (int) w_size, "SIZE", (int) w_mod, "MODIFIED", "STATUS");
        for (const auto & r : rows) {
            printf("%-*s  %-*s  %-*s  %s\n",
                   (int) w_name, r.name.c_str(), (int) w_size, r.size.c_str(),
                   (int) w_mod, r.modified.c_str(), r.status.c_str());
        }
    } else {
        printf("%-*s  %-*s  %s\n", (int) w_name, "NAME", (int) w_size, "SIZE", "STATUS");
        for (const auto & r : rows) {
            printf("%-*s  %-*s  %s\n", (int) w_name, r.name.c_str(), (int) w_size, r.size.c_str(), r.status.c_str());
        }
    }
}

//
// commands
//

int cmd_pull(int argc, char ** argv) {
    run_config cfg = default_run_config();
    std::vector<std::string> pos = parse_endpoint_flags(argc, argv, cfg);
    if (pos.empty()) {
        fprintf(stderr, "usage: llama pull <repo[:tag]>\n");
        return 1;
    }

    if (!do_pull(pos[0])) {
        return 1;
    }
    fprintf(stderr, "done\n");

    // refresh the daemon's model list if it is running
    if (daemon_is_router(cfg)) {
        reload_daemon_models(cfg);
    }
    return 0;
}

int cmd_run(int argc, char ** argv) {
    run_config cfg = default_run_config();
    std::vector<std::string> pos = parse_endpoint_flags(argc, argv, cfg);
    if (pos.empty()) {
        fprintf(stderr, "usage: llama run <repo[:tag]|name> [prompt]\n");
        return 1;
    }

    const std::string model_arg = pos[0];
    std::string prompt;
    for (size_t i = 1; i < pos.size(); ++i) {
        if (!prompt.empty()) {
            prompt += ' ';
        }
        prompt += pos[i];
    }

    if (!ensure_daemon(cfg)) {
        return 1;
    }

    std::string model_id = resolve_model_id(cfg, model_arg);
    if (model_id.empty()) {
        // not known to the daemon: try an implicit pull (only meaningful for HF repos)
        if (model_arg.find('/') != std::string::npos) {
            fprintf(stderr, "model '%s' not found locally, pulling...\n", model_arg.c_str());
            if (!do_pull(model_arg)) {
                return 1;
            }
            reload_daemon_models(cfg);
            model_id = resolve_model_id(cfg, model_arg);
        }
        if (model_id.empty()) {
            fprintf(stderr, "error: model '%s' not found\n", model_arg.c_str());
            return 1;
        }
    }

    if (!prompt.empty()) {
        return run_single_shot(cfg, model_id, prompt);
    }
    return run_interactive(cfg, model_id);
}

int cmd_list(int argc, char ** argv) {
    run_config cfg = default_run_config();
    parse_endpoint_flags(argc, argv, cfg);

    std::vector<model_row> rows;

    if (daemon_is_router(cfg)) {
        try {
            for (const auto & m : fetch_models(cfg)) {
                const auto st = common_cached_model_disk_stats(m.size_repo);
                rows.push_back({m.id, m.status.empty() ? "unloaded" : m.status,
                                format_size(st.size), format_mtime(st.mtime)});
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    } else {
        // offline fallback: enumerate the HF cache directly. Request stats so size and mtime are
        // computed in the single cache walk, instead of re-resolving each model with a separate scan.
        for (const auto & m : common_list_cached_models(/* with_stats */ true)) {
            rows.push_back({m.to_string(), "-", format_size(m.size), format_mtime(m.mtime)});
        }
    }

    if (rows.empty()) {
        printf("No models found. Pull one with: llama pull <repo[:tag]>\n");
        return 0;
    }
    print_model_table(rows);
    return 0;
}

int cmd_ps(int argc, char ** argv) {
    run_config cfg = default_run_config();
    parse_endpoint_flags(argc, argv, cfg);

    if (!daemon_is_router(cfg)) {
        printf("No running models (daemon is not running).\n");
        return 0;
    }

    std::vector<model_row> rows;
    try {
        for (const auto & m : fetch_models(cfg)) {
            if (m.status == "loading" || m.status == "loaded" || m.status == "sleeping") {
                rows.push_back({m.id, m.status, format_size(common_cached_model_disk_stats(m.size_repo).size), ""});
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (rows.empty()) {
        printf("No running models.\n");
        return 0;
    }
    print_model_table(rows);
    return 0;
}

int cmd_stop(int argc, char ** argv) {
    run_config cfg = default_run_config();
    std::vector<std::string> pos = parse_endpoint_flags(argc, argv, cfg);
    if (pos.empty()) {
        fprintf(stderr, "usage: llama stop <repo[:tag]|name>\n");
        return 1;
    }

    if (!daemon_is_router(cfg)) {
        fprintf(stderr, "daemon is not running; nothing to stop\n");
        return 0;
    }

    std::string model_id = resolve_model_id(cfg, pos[0]);
    if (model_id.empty()) {
        model_id = pos[0]; // let the daemon report "not found"/"not running"
    }

    try {
        auto cli = make_client(cfg, 30);
        json body = {{"model", model_id}};
        auto res = cli.Post("/models/unload", body.dump(), "application/json");
        if (!res) {
            fprintf(stderr, "error: request failed: %s\n", httplib::to_string(res.error()).c_str());
            return 1;
        }
        json j = json::parse(res->body, nullptr, false);
        if (res->status == 200 && !j.is_discarded() && j.value("success", false)) {
            printf("stopped %s\n", model_id.c_str());
            return 0;
        }
        fprintf(stderr, "error: %s\n", error_message_from(res->body).c_str());
        return 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

int cmd_rm(int argc, char ** argv) {
    run_config cfg = default_run_config();
    std::vector<std::string> pos = parse_endpoint_flags(argc, argv, cfg);
    if (pos.empty()) {
        fprintf(stderr, "usage: llama rm <repo[:tag]|preset>\n");
        return 1;
    }
    const std::string name = pos[0];

    const bool daemon_up = daemon_is_router(cfg);

    // best-effort unload first, so we don't delete/redefine a model out from under a running instance
    std::string repo = repo_of(name);
    if (daemon_up) {
        const std::string model_id = resolve_model_id(cfg, name);
        if (!model_id.empty()) {
            repo = repo_of(model_id);
            try {
                auto cli = make_client(cfg, 30);
                json body = {{"model", model_id}};
                cli.Post("/models/unload", body.dump(), "application/json");
            } catch (...) {
                // ignore: model may not be running
            }
            // /models/unload only signals the stop and returns immediately, so wait for the model
            // to actually unload before deleting (20s covers the server's stop timeout + kill)
            if (!wait_until_unloaded(cfg, model_id, 20)) {
                fprintf(stderr, "warning: model '%s' did not unload in time; "
                                "deletion may fail if it is still in use\n", model_id.c_str());
            }
        }
    }

    // case 1: a custom preset (a [name] section in models.ini). Removing it only drops the
    // named config - the underlying model weights stay cached (delete those with `rm <repo>`).
    if (remove_preset_section(default_preset_path(), name)) {
        printf("removed preset '%s'\n", name.c_str());
        if (daemon_up) {
            reload_daemon_models(cfg);
        }
        return 0;
    }

    // case 2: a cached HF repo (always has a '/'). With a :tag remove only that quant's file(s)
    // and their blobs; without one drop the whole repo. A bare non-preset name hits the error below.
    if (repo.find('/') != std::string::npos) {
        if (!tag_of(name).empty()) {
            // a specific quant: delete only its file(s), leaving other quants and the mmproj intact
            const auto files = common_cached_model_files(name);
            if (!files.empty() && hf_cache::delete_cached_files(repo, files) > 0) {
                printf("removed %s\n", name.c_str());
                if (daemon_up) {
                    reload_daemon_models(cfg);
                }
                return 0;
            }
        } else if (hf_cache::delete_cached_repo(repo)) {
            // no tag: drop the whole repo
            printf("removed %s\n", repo.c_str());
            if (daemon_up) {
                reload_daemon_models(cfg);
            }
            return 0;
        }
    }

    fprintf(stderr, "error: nothing to remove for '%s' (no such preset or cached model)\n", name.c_str());
    return 1;
}

int cmd_create(int argc, char ** argv) {
    quiet_backend_logs(); // building the preset context initializes the arg parser (device probing)
    run_config cfg = default_run_config();

    // manual parse: <name> --from <repo[:tag]> [-k key=value ...] plus endpoint flags
    std::string name;
    std::string from;
    std::vector<std::pair<std::string, std::string>> kvs;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * opt) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + opt);
            }
            return argv[++i];
        };
        if (a == "--from" || a == "-f") {
            from = next(a.c_str());
        } else if (a == "-k" || a == "--set") {
            const std::string kv = next(a.c_str());
            const auto eq = kv.find('=');
            if (eq == std::string::npos) {
                fprintf(stderr, "error: -k expects key=value, got '%s'\n", kv.c_str());
                return 1;
            }
            kvs.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else if (a == "--host") {
            cfg.host = next("--host");
        } else if (a == "--port") {
            cfg.port = parse_port(next("--port"));
        } else if (a == "--api-key") {
            cfg.api_key = next("--api-key");
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return 1;
        } else if (name.empty()) {
            name = a;
        } else {
            fprintf(stderr, "error: unexpected argument '%s'\n", a.c_str());
            return 1;
        }
    }

    if (name.empty() || from.empty()) {
        fprintf(stderr, "usage: llama create <name> --from <repo[:tag]> [-k key=value ...]\n");
        return 1;
    }

    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);

    // load existing presets so we can update one in place
    common_presets presets;
    common_preset  global;
    const fs::path path = default_preset_path();
    std::error_code ec;
    if (fs::exists(path, ec)) {
        try {
            presets = ctx.load_from_ini(path.string(), global);
        } catch (const std::exception & e) {
            fprintf(stderr, "error: failed to read existing presets: %s\n", e.what());
            return 1;
        }
    }

    common_preset preset;
    preset.name = name;
    try {
        preset.set_option(ctx, "LLAMA_ARG_HF_REPO", from);
        for (const auto & [key, value] : kvs) {
            preset.set_option(ctx, key, value);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    presets[name] = preset;

    try {
        write_presets_file(path, presets, global);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    printf("created preset '%s' (from %s)\n", name.c_str(), from.c_str());

    if (daemon_is_router(cfg)) {
        reload_daemon_models(cfg);
    }
    return 0;
}

int cmd_show(int argc, char ** argv) {
    quiet_backend_logs();
    run_config cfg = default_run_config();
    std::vector<std::string> pos = parse_endpoint_flags(argc, argv, cfg);
    if (pos.empty()) {
        fprintf(stderr, "usage: llama show <name>\n");
        return 1;
    }
    const std::string name = pos[0];

    // gather what we can from three sources: the router (status + preset + runtime meta),
    // the local preset file (custom presets), and the GGUF file itself (model metadata).
    std::string model_id = name;
    std::string preset_ini;
    std::string status;
    json        daemon_entry;

    const bool daemon_up = daemon_is_router(cfg);
    if (daemon_up) {
        // single /v1/models round-trip: resolve the name and grab its full entry at once
        daemon_entry = fetch_model_entry(cfg, name);
        if (daemon_entry.is_object()) {
            model_id = daemon_entry.value("id", model_id);
            if (daemon_entry.contains("status") && daemon_entry["status"].is_object()) {
                status = daemon_entry["status"].value("value", std::string());
                if (daemon_entry["status"].contains("preset") && daemon_entry["status"]["preset"].is_string()) {
                    preset_ini = daemon_entry["status"]["preset"].get<std::string>();
                }
            }
        }
    }

    if (preset_ini.empty()) {
        preset_ini = read_preset_section(default_preset_path(), name);
    }

    // resolve a missing quant tag from the cache, preferring the same quant `pull` picks
    // (Q4_K_M -> Q8_0 -> first), so the reported metadata matches the file that would load
    if (model_id.find('/') != std::string::npos && tag_of(model_id).empty()) {
        const std::string repo = repo_of(model_id);
        std::vector<std::string> tags; // cm.tag is already upper-cased by the cache enumeration
        for (const auto & cm : common_list_cached_models()) {
            if (cm.repo == repo) {
                tags.push_back(cm.tag);
            }
        }
        if (!tags.empty()) {
            std::string chosen = tags.front();
            for (const char * pref : {"Q4_K_M", "Q8_0"}) {
                const auto it = std::find(tags.begin(), tags.end(), std::string(pref));
                if (it != tags.end()) {
                    chosen = *it;
                    break;
                }
            }
            model_id = repo + ":" + chosen;
        }
    }

    // resolve a repo:tag to locate the GGUF - for custom presets, dig the hf-repo out of the preset
    std::string lookup_id = model_id;
    if (lookup_id.find('/') == std::string::npos && !preset_ini.empty()) {
        std::string repo = ini_value(preset_ini, "hf-repo");
        if (repo.empty()) {
            repo = ini_value(preset_ini, "LLAMA_ARG_HF_REPO");
        }
        if (!repo.empty()) {
            lookup_id = repo;
        }
    }

    const std::string gguf_path = find_cached_gguf(lookup_id);
    gguf_model_info info;
    if (!gguf_path.empty()) {
        info = read_gguf_meta(gguf_path);
    }

    if (!info.ok && preset_ini.empty() && !daemon_entry.is_object()) {
        fprintf(stderr, "error: '%s' not found (not cached, no preset; is it pulled?)\n", name.c_str());
        return 1;
    }

    auto row = [](const char * key, const std::string & val) {
        if (!val.empty()) {
            printf("  %-18s %s\n", key, val.c_str());
        }
    };

    printf("%s\n", model_id.c_str());

    // Model section (from GGUF metadata)
    if (info.ok) {
        printf("\nModel\n");
        row("architecture",     info.arch);
        row("name",             info.name);
        row("parameters",       info.n_params ? format_params(info.n_params) : info.size_label);
        row("quantization",     tag_of(lookup_id));
        if (info.n_ctx_train) { row("context length",   std::to_string(info.n_ctx_train)); }
        if (info.n_embd)      { row("embedding length", std::to_string(info.n_embd)); }
        if (info.n_layers)    { row("layers",           std::to_string(info.n_layers)); }
        if (info.file_size)   { row("size",             format_size(info.file_size)); }
        if (info.has_chat_template) { row("chat template", "embedded"); }
        row("path", info.path);
    }

    // Status section (from the running daemon)
    if (daemon_up && !status.empty()) {
        printf("\nStatus\n");
        row("state", status);
        if (daemon_entry.contains("meta") && daemon_entry["meta"].is_object()) {
            const auto & meta = daemon_entry["meta"];
            if (meta.contains("n_ctx") && meta["n_ctx"].is_number_integer()) {
                row("context (loaded)", std::to_string(meta["n_ctx"].get<long long>()));
            }
        }
    }

    // License
    if (info.ok && !info.license.empty()) {
        printf("\nLicense\n  %s\n", info.license.c_str());
    }

    // Preset (the config applied when the model is loaded)
    if (!preset_ini.empty()) {
        printf("\nPreset\n");
        std::istringstream ss(preset_ini);
        std::string line;
        while (std::getline(ss, line)) {
            if (!string_strip(line).empty()) {
                printf("  %s\n", line.c_str());
            }
        }
    }

    return 0;
}

} // namespace

//
// public entry points
//

int llama_pull  (int argc, char ** argv) { return cmd_pull(argc, argv); }
int llama_run   (int argc, char ** argv) { return cmd_run(argc, argv); }
int llama_list  (int argc, char ** argv) { return cmd_list(argc, argv); }
int llama_ps    (int argc, char ** argv) { return cmd_ps(argc, argv); }
int llama_stop  (int argc, char ** argv) { return cmd_stop(argc, argv); }
int llama_rm    (int argc, char ** argv) { return cmd_rm(argc, argv); }
int llama_create(int argc, char ** argv) { return cmd_create(argc, argv); }
int llama_show  (int argc, char ** argv) { return cmd_show(argc, argv); }
