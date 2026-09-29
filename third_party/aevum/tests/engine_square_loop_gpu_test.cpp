// GPU check of aevum_engine_square_loop against GMP, and timing of the fused loop versus
// one aevum_engine_square_mul call per squaring.
//
// usage: engine_square_loop_gpu_test <libaevum_engine> <device> <tune_dir> [exponent[:iterations] ...]
//   AEVUM_TEST_FFT=<spec> forces an FFT shape, e.g. 1:1K:7:256:202

#include "../src/EngineApi.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#if defined(_WIN32)
void* open_library(const char* path) { return reinterpret_cast<void*>(LoadLibraryA(path)); }
void* find_symbol(void* lib, const char* name) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name)); }
void close_library(void* lib) { FreeLibrary(static_cast<HMODULE>(lib)); }
const char* library_error() { return "LoadLibrary failed"; }
#else
void* open_library(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void* find_symbol(void* lib, const char* name) { return dlsym(lib, name); }
void close_library(void* lib) { dlclose(lib); }
const char* library_error() { return dlerror(); }
#endif

template <class T>
T load_symbol(void* lib, const char* name) {
    void* symbol = find_symbol(lib, name);
    if (!symbol) throw std::runtime_error(std::string("missing symbol: ") + name);
    return reinterpret_cast<T>(symbol);
}

struct Api {
    aevum_engine_handle (*create)(uint32_t, size_t, uint32_t, int, const char*, const char*);
    void (*destroy)(aevum_engine_handle);
    size_t (*word_count)(aevum_engine_handle);
    size_t (*transform_size)(aevum_engine_handle);
    int (*sync)(aevum_engine_handle);
    int (*set_u32)(aevum_engine_handle, size_t, uint32_t);
    int (*get_words)(aevum_engine_handle, size_t, uint32_t*, size_t);
    int (*prepare)(aevum_engine_handle, size_t, size_t);
    int (*square_mul)(aevum_engine_handle, size_t, uint32_t);
    int (*square_loop)(aevum_engine_handle, size_t, uint64_t, int);
    int (*mul)(aevum_engine_handle, size_t, size_t, uint32_t);
    const char* (*last_error)();
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct Exponent {
    uint32_t p;
    uint64_t iterations;
};

Exponent parse(const std::string& text) {
    const size_t colon = text.find(':');
    const uint32_t p = static_cast<uint32_t>(std::strtoul(text.substr(0, colon).c_str(), nullptr, 10));
    const uint64_t k = colon == std::string::npos ? 100 : std::strtoull(text.substr(colon + 1).c_str(), nullptr, 10);
    return {p, k};
}

}  // namespace

int main(int argc, char** argv) {
    const char* library = argc > 1 ? argv[1] : "build-engine/libaevum_engine.so";
    const uint32_t device = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 0;
    const char* tune_dir = argc > 3 ? argv[3] : ".";
    std::vector<Exponent> exponents;
    for (int i = 4; i < argc; ++i) exponents.push_back(parse(argv[i]));
    // Long carry (1362763 at 256K words), then fused carry at several transform sizes.
    if (exponents.empty()) exponents = {{1362763, 100}, {2976221, 200}, {6972593, 200}, {13466917, 100}, {30402457, 50}};

    void* lib = open_library(library);
    if (!lib) {
        std::fprintf(stderr, "%s: %s\n", library, library_error());
        return 2;
    }
    Api api{};
    api.create = load_symbol<decltype(api.create)>(lib, "aevum_engine_create");
    api.destroy = load_symbol<decltype(api.destroy)>(lib, "aevum_engine_destroy");
    api.word_count = load_symbol<decltype(api.word_count)>(lib, "aevum_engine_word_count");
    api.transform_size = load_symbol<decltype(api.transform_size)>(lib, "aevum_engine_transform_size");
    api.sync = load_symbol<decltype(api.sync)>(lib, "aevum_engine_sync");
    api.set_u32 = load_symbol<decltype(api.set_u32)>(lib, "aevum_engine_set_u32");
    api.get_words = load_symbol<decltype(api.get_words)>(lib, "aevum_engine_get_words");
    api.prepare = load_symbol<decltype(api.prepare)>(lib, "aevum_engine_prepare");
    api.square_mul = load_symbol<decltype(api.square_mul)>(lib, "aevum_engine_square_mul");
    api.square_loop = load_symbol<decltype(api.square_loop)>(lib, "aevum_engine_square_loop");
    api.mul = load_symbol<decltype(api.mul)>(lib, "aevum_engine_mul");
    api.last_error = load_symbol<decltype(api.last_error)>(lib, "aevum_engine_last_error");

    int failures = 0;
    mpz_t mp, expected, actual, exponent, three;
    mpz_inits(mp, expected, actual, exponent, three, nullptr);
    mpz_set_ui(three, 3);

    for (const Exponent& e : exponents) {
        const char* spec = std::getenv("AEVUM_TEST_FFT");
        aevum_engine_handle h = api.create(e.p, 3, device, 0, spec ? spec : "", tune_dir);
        if (!h) {
            std::printf("M%u: cannot create engine: %s\n", e.p, api.last_error());
            ++failures;
            continue;
        }
        const size_t words = api.word_count(h);
        std::vector<uint32_t> buffer(words);
        mpz_set_ui(mp, 0);
        mpz_setbit(mp, e.p);
        mpz_sub_ui(mp, mp, 1);

        auto require = [&](int ok, const char* what) {
            if (!ok) throw std::runtime_error(std::string(what) + ": " + api.last_error());
        };
        auto check = [&](size_t reg, const char* label) {
            require(api.get_words(h, reg, buffer.data(), words), "get_words");
            mpz_import(actual, words, -1, sizeof(uint32_t), 0, 0, buffer.data());
            mpz_mod(actual, actual, mp);
            const bool ok = mpz_cmp(actual, expected) == 0;
            if (!ok) ++failures;
            std::printf("M%u %-34s %s\n", e.p, label, ok ? "ok" : "MISMATCH");
        };

        try {
            const uint64_t k = e.iterations;
            mpz_set_ui(exponent, 0);
            mpz_setbit(exponent, k);
            mpz_powm(expected, three, exponent, mp);   // 3^(2^k)

            // Warm up (kernel compilation) outside of the timings.
            require(api.set_u32(h, 0, 3), "set");
            require(api.square_loop(h, 0, 2, 0), "square_loop");
            require(api.square_mul(h, 0, 1), "square_mul");
            require(api.sync(h), "sync");

            require(api.set_u32(h, 0, 3), "set");
            auto t0 = std::chrono::steady_clock::now();
            for (uint64_t i = 0; i < k; ++i) require(api.square_mul(h, 0, 1), "square_mul");
            require(api.sync(h), "sync");
            const double single = seconds_since(t0) / double(k);
            check(0, "square_mul x k");

            require(api.set_u32(h, 1, 3), "set");
            t0 = std::chrono::steady_clock::now();
            require(api.square_loop(h, 1, k, 0), "square_loop");
            require(api.sync(h), "sync");
            const double fused = seconds_since(t0) / double(k);
            check(1, "square_loop");

            // The loop output must be a normal register: multiply it by a prepared operand.
            require(api.set_u32(h, 2, 5), "set");
            require(api.prepare(h, 2, 2), "prepare");
            require(api.mul(h, 1, 2, 1), "mul");
            mpz_mul_ui(expected, expected, 5);
            mpz_mod(expected, expected, mp);
            check(1, "square_loop then prepared mul");

            // Lucas-Lehmer steps x := x^2 - 2 from x = 4.
            mpz_set_ui(expected, 4);
            for (uint64_t i = 0; i < k; ++i) {
                mpz_mul(expected, expected, expected);
                mpz_sub_ui(expected, expected, 2);
                mpz_mod(expected, expected, mp);
            }
            require(api.set_u32(h, 0, 4), "set");
            require(api.square_loop(h, 0, k, 1), "square_loop LL");
            check(0, "square_loop Lucas-Lehmer");

            std::printf("M%u transform=%zu: %.3f ms/iter single, %.3f ms/iter fused loop (x%.2f)\n",
                        e.p, api.transform_size(h), single * 1e3, fused * 1e3, single / fused);
        } catch (const std::exception& ex) {
            std::printf("M%u: %s\n", e.p, ex.what());
            ++failures;
        }
        api.destroy(h);
    }

    mpz_clears(mp, expected, actual, exponent, three, nullptr);
    close_library(lib);
    std::printf("%s\n", failures ? "square_loop GPU test FAILED" : "square_loop GPU test passed");
    return failures ? 1 : 0;
}
