// Tuning sweep of the Aevum engine on a GPU. It times every FFT3161 shape of the transform
// size chosen for an exponent, then Aevum's kernel settings (-use KEY=VALUE, in the order and
// with the values of Aevum's own -tune), and prints the fastest configuration as prmers options.
// Every configuration is first checked against GMP on a random residue.
//
// usage: engine_tune_gpu <libaevum_engine> <device> <tune_dir> [exponent] [full|quick]
//   quick skips the NVIDIA cache-hint settings (LOADS/STORES) and the settings that only
//   apply when INPLACE=0.  AEVUM_TUNE_SECONDS sets the length of each timing (default 1).

#include "../src/EngineApi.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#if defined(_WIN32)
void* open_library(const char* path) { return reinterpret_cast<void*>(LoadLibraryA(path)); }
void* find_symbol(void* lib, const char* name) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name)); }
const char* library_error() { return "LoadLibrary failed"; }
#else
void* open_library(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void* find_symbol(void* lib, const char* name) { return dlsym(lib, name); }
const char* library_error() { return dlerror(); }
#endif

template <class T>
T load_symbol(void* lib, const char* name) {
    void* symbol = find_symbol(lib, name);
    if (!symbol) throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<T>(symbol);
}

struct Api {
    int (*resolve_fft)(uint32_t, const char*, char*, size_t);
    int (*set_use)(const char*);
    aevum_engine_handle (*create)(uint32_t, size_t, uint32_t, int, const char*, const char*);
    void (*destroy)(aevum_engine_handle);
    size_t (*word_count)(aevum_engine_handle);
    size_t (*transform_size)(aevum_engine_handle);
    int (*sync)(aevum_engine_handle);
    int (*set_words)(aevum_engine_handle, size_t, const uint32_t*, size_t);
    int (*get_words)(aevum_engine_handle, size_t, uint32_t*, size_t);
    int (*square_loop)(aevum_engine_handle, size_t, uint64_t, int);
    const char* (*last_error)();
};

using Settings = std::map<std::string, std::string>;

std::string lookup(const Settings& settings, const std::string& key) {
    const auto it = settings.find(key);
    return it == settings.end() ? "" : it->second;
}

std::string join(const Settings& settings) {
    std::string out;
    for (const auto& [key, value] : settings) out += (out.empty() ? "" : ",") + key + "=" + value;
    return out;
}

std::string size_name(uint32_t n) { return n % 1024 ? std::to_string(n) : std::to_string(n / 1024) + "K"; }

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// x := x mod 2^p - 1, for 0 <= x < 2^(2p).
void reduce(mpz_t x, mpz_t t, const mpz_t mp, uint32_t p) {
    while (mpz_sizeinbase(x, 2) > p) {
        mpz_tdiv_q_2exp(t, x, p);
        mpz_tdiv_r_2exp(x, x, p);
        mpz_add(x, x, t);
    }
    if (mpz_cmp(x, mp) >= 0) mpz_sub(x, x, mp);
}

class Tuner {
public:
    Tuner(const Api& api, uint32_t p, uint32_t device, std::string tune_dir, double seconds)
        : api_(api), p_(p), device_(device), tune_dir_(std::move(tune_dir)), seconds_(seconds) {
        words_ = (p + 31) / 32;
        std::mt19937_64 rng(0x7e57ULL + p);
        start_.resize(words_);
        for (auto& w : start_) w = static_cast<uint32_t>(rng());
        if (p % 32) start_.back() &= (uint32_t(1) << (p % 32)) - 1;

        mpz_inits(mp_, x_, t_, nullptr);
        mpz_setbit(mp_, p);
        mpz_sub_ui(mp_, mp_, 1);
        mpz_import(x_, words_, -1, sizeof(uint32_t), 0, 0, start_.data());
        reduce(x_, t_, mp_, p_);
        for (uint64_t i = 0; i < verify_steps; ++i) {
            mpz_mul(x_, x_, x_);
            reduce(x_, t_, mp_, p_);
        }
    }
    ~Tuner() { mpz_clears(mp_, x_, t_, nullptr); }

    std::string resolve(const std::string& fft) const {
        char buffer[128];
        return api_.resolve_fft(p_, fft.c_str(), buffer, sizeof(buffer)) ? buffer : fft;
    }

    // Milliseconds per squaring, or a negative value when the configuration fails.
    double time(const std::string& fft, const Settings& use, size_t* transform = nullptr) {
        const std::string settings = join(use);
        api_.set_use(settings.c_str());
        aevum_engine_handle h = api_.create(p_, 2, device_, 0, fft.c_str(), tune_dir_.c_str());
        double ms = -1.0;
        std::string status;
        if (!h) {
            status = std::string("FAILED to create: ") + api_.last_error();
        } else {
            try {
                if (transform) *transform = api_.transform_size(h);
                ms = measure(h);
                status = "ok";
            } catch (const std::exception& e) {
                status = std::string("FAILED: ") + e.what();
            }
            api_.destroy(h);
        }
        api_.set_use("");
        std::printf("tune: %9s ms/iter  %-18s %-40s %s\n", ms > 0 ? format(ms).c_str() : "-",
                    resolve(fft).c_str(), settings.empty() ? "(default settings)" : settings.c_str(), status.c_str());
        std::fflush(stdout);
        return ms;
    }

private:
    static constexpr uint64_t verify_steps = 12;

    static std::string format(double ms) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.4f", ms);
        return buffer;
    }

    void require(int ok, const char* what) const {
        if (!ok) throw std::runtime_error(std::string(what) + ": " + api_.last_error());
    }

    double measure(aevum_engine_handle h) {
        // Correctness first: verify_steps squarings of a random residue must match GMP.
        std::vector<uint32_t> out(words_);
        require(api_.word_count(h) == words_, "word_count");
        require(api_.set_words(h, 0, start_.data(), words_), "set_words");
        require(api_.square_loop(h, 0, verify_steps, 0), "square_loop");
        require(api_.get_words(h, 0, out.data(), words_), "get_words");
        mpz_import(t_, words_, -1, sizeof(uint32_t), 0, 0, out.data());
        mpz_t r;
        mpz_init(r);
        mpz_set(r, t_);
        reduce(r, t_, mp_, p_);
        const bool ok = mpz_cmp(r, x_) == 0;
        mpz_clear(r);
        if (!ok) throw std::runtime_error("WRONG RESULT against GMP");

        auto run = [&](uint64_t n) {
            const auto t0 = std::chrono::steady_clock::now();
            require(api_.square_loop(h, 0, n, 0), "square_loop");
            require(api_.sync(h), "sync");
            return seconds_since(t0) / double(n);
        };
        run(32);
        const double estimate = run(64);
        const uint64_t n = std::clamp<uint64_t>(uint64_t(seconds_ / std::max(estimate, 1e-7)), 64, 1000000);
        std::vector<double> t;
        for (int i = 0; i < 3; ++i) t.push_back(run(n));
        std::sort(t.begin(), t.end());
        return t[1] * 1e3;
    }

    const Api& api_;
    uint32_t p_, device_;
    std::string tune_dir_;
    double seconds_;
    size_t words_;
    std::vector<uint32_t> start_;
    mpz_t mp_, x_, t_;
};

// FFT3161 shapes of the given transform size, as Aevum enumerates them (power-of-two middles).
std::vector<std::string> shapes_of_size(size_t size) {
    std::vector<std::string> out;
    for (uint32_t w : {256u, 512u, 1024u, 4096u}) {
        for (uint32_t h : {256u, 512u, 1024u}) {
            if (w == 256 && h == 1024) continue;
            for (uint32_t m : {2u, 4u, 8u, 16u}) {
                if (size_t(2) * w * m * h == size) out.push_back("1:" + size_name(w) + ":" + std::to_string(m) + ":" + size_name(h) + ":202");
            }
        }
    }
    return out;
}

uint32_t shape_width(const std::string& fft) {
    const size_t a = fft.find(':') + 1;
    const std::string w = fft.substr(a, fft.find(':', a) - a);
    return uint32_t(std::stoul(w)) * (w.back() == 'K' ? 1024 : 1);
}

std::string digit_set(const std::string& value, uint32_t position, uint32_t digit) {
    uint32_t v = value.empty() ? 0 : uint32_t(std::stoul(value));
    uint32_t scale = 1;
    for (uint32_t i = 0; i < position; ++i) scale *= 10;
    v = v - (v / scale % 10) * scale + digit * scale;
    return std::to_string(v);
}

}  // namespace

int main(int argc, char** argv) {
    const char* library = argc > 1 ? argv[1] : "build-engine/libaevum_engine.so";
    const uint32_t device = argc > 2 ? uint32_t(std::strtoul(argv[2], nullptr, 10)) : 0;
    const std::string tune_dir = argc > 3 ? argv[3] : ".";
    const uint32_t p = argc > 4 ? uint32_t(std::strtoul(argv[4], nullptr, 10)) : 136279841;
    const bool quick = argc > 5 && std::string(argv[5]) == "quick";
    const char* seconds_env = std::getenv("AEVUM_TUNE_SECONDS");
    const double seconds = seconds_env ? std::max(0.1, std::atof(seconds_env)) : 1.0;

    void* lib = open_library(library);
    if (!lib) {
        std::fprintf(stderr, "%s: %s\n", library, library_error());
        return 2;
    }
    Api api{};
    try {
        api.resolve_fft = load_symbol<decltype(api.resolve_fft)>(lib, "aevum_engine_resolve_fft");
        api.set_use = load_symbol<decltype(api.set_use)>(lib, "aevum_engine_set_use");
        api.create = load_symbol<decltype(api.create)>(lib, "aevum_engine_create");
        api.destroy = load_symbol<decltype(api.destroy)>(lib, "aevum_engine_destroy");
        api.word_count = load_symbol<decltype(api.word_count)>(lib, "aevum_engine_word_count");
        api.transform_size = load_symbol<decltype(api.transform_size)>(lib, "aevum_engine_transform_size");
        api.sync = load_symbol<decltype(api.sync)>(lib, "aevum_engine_sync");
        api.set_words = load_symbol<decltype(api.set_words)>(lib, "aevum_engine_set_words");
        api.get_words = load_symbol<decltype(api.get_words)>(lib, "aevum_engine_get_words");
        api.square_loop = load_symbol<decltype(api.square_loop)>(lib, "aevum_engine_square_loop");
        api.last_error = load_symbol<decltype(api.last_error)>(lib, "aevum_engine_last_error");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", library, e.what());
        return 2;
    }

    std::printf("tune: M%u on device %u, %s mode, %.1f s per timing\n", p, device, quick ? "quick" : "full", seconds);
    std::fflush(stdout);
    Tuner tuner(api, p, device, tune_dir, seconds);

    // 1. The automatic choice with default settings (what prmers uses), then every shape of that size.
    size_t size = 0;
    const std::string auto_fft = tuner.resolve("");
    const double baseline = tuner.time("", {}, &size);
    if (baseline <= 0) {
        std::printf("tune: the default configuration fails on this device\n");
        return 1;
    }
    std::string best_fft;
    double best = 0;
    for (const std::string& fft : shapes_of_size(size)) {
        const double ms = tuner.time(fft, {});
        if (ms > 0 && (best <= 0 || ms < best)) { best = ms; best_fft = fft; }
    }
    if (best_fft.empty()) { best_fft = auto_fft; best = baseline; }

    // 2. Kernel settings, one at a time on the fastest shape; keep a value if it saves over 0.3%.
    //    Settings equal to Aevum's defaults are left out (PAD and UNROLL_* defaults depend on the GPU).
    const Settings defaults = {{"INPLACE", "0"}, {"IN_WG", "128"}, {"IN_SIZEX", "16"}, {"OUT_WG", "128"},
                               {"OUT_SIZEX", "16"}, {"MIDDLE_IN_LDS_TRANSPOSE", "1"}, {"MIDDLE_OUT_LDS_TRANSPOSE", "1"},
                               {"LOADS", "0"}, {"STORES", "0"}, {"FAST_BARRIER", "0"}, {"TAIL_KERNELS", "2"},
                               {"TAIL_TRIGS31", "0"}, {"TAIL_TRIGS61", "0"}, {"TABMUL_CHAIN31", "0"},
                               {"TABMUL_CHAIN61", "0"}, {"MODM31", "0"}, {"ZEROHACK_W", "1"}, {"ZEROHACK_H", "1"},
                               {"WMUL", "2"}};
    Settings use;
    auto tune = [&](const char* name, const std::vector<Settings>& choices) {
        Settings best_use = use;
        double best_ms = best;
        for (const Settings& choice : choices) {
            Settings trial = use;
            for (const auto& [key, value] : choice) {
                if (lookup(defaults, key) == value) trial.erase(key);
                else trial[key] = value;
            }
            if (trial == use) continue;
            const double ms = tuner.time(best_fft, trial);
            if (ms > 0 && ms < best_ms && ms < best * 0.997) { best_ms = ms; best_use = trial; }
        }
        if (best_use != use) {
            use = best_use;
            best = best_ms;
        }
        std::printf("tune: %s -> %s\n", name, use.empty() ? "(default settings)" : join(use).c_str());
        std::fflush(stdout);
    };
    auto values = [](const char* key, std::initializer_list<const char*> list) {
        std::vector<Settings> out;
        for (const char* v : list) out.push_back({{key, v}});
        return out;
    };

    tune("INPLACE", values("INPLACE", {"0", "1"}));
    if (!quick && lookup(use, "INPLACE") != "1") {
        std::vector<Settings> in, out;
        for (const char* wg : {"64", "128", "256"}) {
            for (const char* sizex : {"8", "16", "32"}) {
                in.push_back({{"IN_WG", wg}, {"IN_SIZEX", sizex}});
                out.push_back({{"OUT_WG", wg}, {"OUT_SIZEX", sizex}});
            }
        }
        tune("IN_WG/IN_SIZEX", in);
        tune("OUT_WG/OUT_SIZEX", out);
        tune("PAD", values("PAD", {"0", "64", "128", "256", "512"}));
        tune("MIDDLE_IN_LDS_TRANSPOSE", values("MIDDLE_IN_LDS_TRANSPOSE", {"0", "1"}));
        tune("MIDDLE_OUT_LDS_TRANSPOSE", values("MIDDLE_OUT_LDS_TRANSPOSE", {"0", "1"}));
    }
    if (!quick) {
        // Cache hints, as digits of LOADS and STORES. Values from 2 use NVIDIA PTX; on other GPUs
        // they fail to build and are skipped.
        auto digit_choices = [&](const char* key, uint32_t position, std::initializer_list<uint32_t> digits) {
            std::vector<Settings> out;
            for (uint32_t d : digits) out.push_back({{key, digit_set(lookup(use, key), position, d)}});
            return out;
        };
        tune("LOADS (FFT data)", digit_choices("LOADS", 0, {0, 1, 2, 3, 4}));
        tune("STORES (FFT data)", digit_choices("STORES", 0, {0, 1, 2, 3, 4}));
        std::vector<Settings> shuttle;
        for (auto [load, store] : {std::pair{0u, 0u}, std::pair{1u, 1u}, std::pair{4u, 2u}}) {
            shuttle.push_back({{"LOADS", digit_set(lookup(use, "LOADS"), 1, load)}, {"STORES", digit_set(lookup(use, "STORES"), 1, store)}});
        }
        tune("LOADS/STORES (carry shuttle)", shuttle);
        tune("LOADS (frequent trig)", digit_choices("LOADS", 2, {0, 5}));
        tune("LOADS (reused trig)", digit_choices("LOADS", 3, {0, 1, 2, 3, 4, 5}));
        tune("LOADS (single-use trig)", digit_choices("LOADS", 4, {0, 1, 2, 3, 4, 5}));
    }
    tune("FAST_BARRIER", values("FAST_BARRIER", {"0", "1"}));
    tune("TAIL_KERNELS", values("TAIL_KERNELS", {"0", "1", "2", "3"}));
    tune("TAIL_TRIGS31", values("TAIL_TRIGS31", {"0", "1"}));
    tune("TAIL_TRIGS61", values("TAIL_TRIGS61", {"0", "1"}));
    tune("TABMUL_CHAIN31", values("TABMUL_CHAIN31", {"0", "1"}));
    tune("TABMUL_CHAIN61", values("TABMUL_CHAIN61", {"0", "1"}));
    tune("MODM31", values("MODM31", {"0", "1", "2"}));
    tune("UNROLL_W", values("UNROLL_W", {"0", "1"}));
    tune("UNROLL_H", values("UNROLL_H", {"0", "1"}));
    tune("ZEROHACK_W", values("ZEROHACK_W", {"0", "1"}));
    tune("ZEROHACK_H", values("ZEROHACK_H", {"0", "1"}));
    if (shape_width(best_fft) != 4096) tune("WMUL", values("WMUL", {"1", "2", "4"}));

    // 3. The shapes again with the tuned settings, then the final comparison.
    std::string final_fft = best_fft;
    for (const std::string& fft : shapes_of_size(size)) {
        if (fft == best_fft) continue;
        const double ms = tuner.time(fft, use);
        if (ms > 0 && ms < best * 0.997) { best = ms; final_fft = fft; }
    }
    const double base_ms = tuner.time("", {});
    const double final_ms = tuner.time(final_fft, use);
    if (base_ms <= 0 || final_ms <= 0) {
        std::printf("tune: the final comparison failed\n");
        return 1;
    }
    std::printf("tune: default %s: %.4f ms/iter\n", auto_fft.c_str(), base_ms);
    std::printf("tune: tuned   %s%s%s: %.4f ms/iter (x%.3f)\n", final_fft.c_str(), use.empty() ? "" : " -use ",
                join(use).c_str(), final_ms, base_ms / final_ms);
    std::printf("tune: prmers options: -aevum-fft %s%s%s\n", final_fft.c_str(), use.empty() ? "" : " -aevum-use ",
                join(use).c_str());
    return 0;
}
