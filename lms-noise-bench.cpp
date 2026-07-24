// Runtime micro-benchmark for the lms-noise.mlir STREAMING kernel.
//
// Links the SAME --stream kernel object the CoreAudio host (lms-noise-macOS.cpp)
// and the correctness harness (lms-noise-check.cpp) drive, but plays no audio.
// It measures the only thing we optimize: one _mlir_ciface_run call renders a
// 128-sample block (~2.9 ms of audio at 44.1 kHz), so the meaningful metric is
// PER-BLOCK render latency vs that real-time budget -- NOT wall-clock over a long
// signal (the kernel is latency-bound, not throughput-bound; see samples/notes.md).
//
// Correctness lives in lms-noise-check; this file is perf only. It times a handful
// of REPRESENTATIVE configurations -- a config is a fixed (noise_kind, wet, held
// voices) setting -- because the per-block cost varies with which index_switch
// noise case runs, whether the LMS is actively cancelling (wet), and whether the
// voice bank is sounding. Each config is warmed up (LMS/caches/CPU clock settle)
// and then timed over many steady-state calls; we report min/median/p90/p99 and
// the real-time headroom, plus one BENCH_JSON line per config for autonomous A/B.
//
// Build + run via lms-noise-bench.sh, which also dumps the kernel's STATIC memory
// footprint (section sizes of the kernel object) for information.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

//===----------------------------------------------------------------------===//
// DSP-MLIR kernel boundary (same ABI as lms-noise-macOS.cpp / lms-noise-check.cpp)
//===----------------------------------------------------------------------===//
static constexpr size_t BLOCK = 128;   // samples per _mlir_ciface_run call
static constexpr double FS = 44100.0;  // sample rate
// Real-time budget for one block: how long 128 samples last as audio. A render
// slower than this cannot keep up; median/budget is the headroom (xRT).
static constexpr double BLOCK_BUDGET_MS = 1000.0 * BLOCK / FS; // ~2.9024 ms

// Each timed sample renders CHUNK blocks between two clock reads, so per-block
// timing stays well above the clock's overhead/resolution; we divide back out.
static constexpr size_t CHUNK = 64;

struct MemRefDescriptor1D {
    double *basePtr;
    double *data;
    int64_t offset;
    int64_t size;
    int64_t stride;
};

extern "C" void _mlir_ciface_run(MemRefDescriptor1D *out);
extern "C" void _mlir_ciface_set_note_event(int64_t voice, double freq,
                                            double gate, int64_t frame);
extern "C" double mu;          // @mu   : LMS step size
extern "C" double wet;         // @wet  : fraction of the noise estimate to subtract
extern "C" int64_t noise_kind; // @noise_kind : dsp.index_switch selector

enum { NK_WHITE = 0, NK_PINK = 1, NK_BROWN = 2, NK_OU = 3, NK_NONE = 4 };
constexpr int NUM_VOICES = 8; // MUST match the kernel's memref<8x...>

// Keep the optimizer from eliminating the kernel calls / output reads.
static volatile double gSink = 0.0;

//===----------------------------------------------------------------------===//
// Kernel invocation + interaction helpers
//===----------------------------------------------------------------------===//
static void renderInto(double *dst) {
    MemRefDescriptor1D desc{dst, dst, 0, static_cast<int64_t>(BLOCK), 1};
    _mlir_ciface_run(&desc);
}
static void renderDiscard(double *scratch, size_t blocks) {
    for (size_t i = 0; i < blocks; ++i) renderInto(scratch);
}
static double noteToHz(int note) {
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}
static void noteOn(int voice, int note) {
    _mlir_ciface_set_note_event(voice, noteToHz(note), 1.0, 0);
}
static void noteOff(int voice) {
    _mlir_ciface_set_note_event(voice, 0.0, 0.0, 0);
}
// Release every voice and let the one-pole gate smoother decay to ~0 so the next
// config starts from a clean synth state (matches lms-noise-check).
static void silenceAllVoices(double *scratch) {
    for (int v = 0; v < NUM_VOICES; ++v) noteOff(v);
    renderDiscard(scratch, 64);
}

// Percentile of an already-sorted ascending vector (p in [0,1]).
static double percentile(const std::vector<double> &sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double idx = p * (static_cast<double>(sorted.size()) - 1.0);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

//===----------------------------------------------------------------------===//
// Representative configurations
//===----------------------------------------------------------------------===//
struct Config {
    const char *name;
    int64_t noise;   // noise_kind
    double wet;      // @wet
    int voices;      // number of held voices (a chord); 0 = rest
    const char *note;
};

// Isolate the cost centres: a silent rest, the ANC hot path alone, the voice bank
// alone, and both together (the worst case).
static const Config kConfigs[] = {
    {"rest_silent", NK_NONE, 0.0, 0, "index_switch default (silence) + idle LMS"},
    {"anc_white", NK_WHITE, 1.0, 0, "white noise + active 32-tap LMS cancel"},
    {"synth_poly8", NK_NONE, 0.0, 8, "8-voice sawtooth bank + per-voice cutoff + reduce"},
    {"full_white_poly8", NK_WHITE, 1.0, 8, "synth bank + white noise + LMS cancel"},
};

// A spread-out chord so the 8 voices run at distinct pitches/phases.
static const int kChord[NUM_VOICES] = {36, 43, 48, 52, 55, 60, 64, 67};

static void setupConfig(const Config &c, double *scratch) {
    noise_kind = c.noise;
    wet = c.wet;
    silenceAllVoices(scratch);
    for (int v = 0; v < c.voices; ++v) noteOn(v, kChord[v]);
    renderDiscard(scratch, 32); // let gates ramp up before timing
}

//===----------------------------------------------------------------------===//
// Timing
//===----------------------------------------------------------------------===//
int main(int argc, char **argv) {
    size_t iterations = 3000; // timing samples per config (each renders CHUNK blocks)
    size_t warmup = 1000;     // blocks rendered before timing (LMS/cache/clock settle)

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = std::strtoull(argv[++i], nullptr, 10);
        else {
            std::fprintf(stderr, "usage: %s [--iterations N] [--warmup N]\n",
                         argv[0]);
            return 2;
        }
    }
    if (iterations == 0) {
        std::fprintf(stderr, "iterations must be > 0\n");
        return 2;
    }

    mu = 1.0e-3; // kernel default LMS step

    std::vector<double> buf(BLOCK);
    std::vector<double> perBlockMs;
    perBlockMs.reserve(iterations);
    using clock = std::chrono::steady_clock;

    std::printf("lms-noise.mlir runtime benchmark (block=%zu, fs=%.0f)\n", BLOCK,
                FS);
    std::printf("  real-time budget per block: %.4f ms "
                "(1.0x = just keeps up)\n",
                BLOCK_BUDGET_MS);
    std::printf("  timing: %zu samples x %zu blocks each, %zu-block warmup\n\n",
                iterations, CHUNK, warmup);

    for (const Config &c : kConfigs) {
        setupConfig(c, buf.data());
        renderDiscard(buf.data(), warmup);

        perBlockMs.clear();
        double checksum = 0.0; // accumulated over FINITE samples only
        size_t nonfinite = 0;  // blocks whose sampled output was inf/nan
        for (size_t i = 0; i < iterations; ++i) {
            const auto t0 = clock::now();
            for (size_t b = 0; b < CHUNK; ++b) renderInto(buf.data());
            const auto t1 = clock::now();
            const double chunkMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            perBlockMs.push_back(chunkMs / static_cast<double>(CHUNK));
            // Timing above is valid regardless of value stability; guard the
            // checksum so a divergent run stays comparable (see ports/bench.sh --
            // the f64 kernel is stable here, but the field keeps the BENCH_JSON
            // schema identical to the ports harness for A/B tooling).
            const double s = buf[i % BLOCK];
            if (std::isfinite(s)) checksum += s;
            else ++nonfinite;
        }
        const bool diverged = nonfinite > 0;
        gSink += checksum;

        std::vector<double> sorted = perBlockMs;
        std::sort(sorted.begin(), sorted.end());
        const double minMs = sorted.front();
        const double medMs = percentile(sorted, 0.50);
        const double p90Ms = percentile(sorted, 0.90);
        const double p99Ms = percentile(sorted, 0.99);
        double sum = 0.0;
        for (double v : perBlockMs) sum += v;
        const double meanMs = sum / static_cast<double>(perBlockMs.size());
        double var = 0.0;
        for (double v : perBlockMs) var += (v - meanMs) * (v - meanMs);
        const double stddevMs = std::sqrt(var / static_cast<double>(perBlockMs.size()));

        const double msamplePerS =
            (static_cast<double>(BLOCK) / (medMs / 1e3)) / 1e6;
        const double headroomX = BLOCK_BUDGET_MS / medMs;      // >1 = faster than RT
        const double budgetPct = 100.0 * medMs / BLOCK_BUDGET_MS;

        std::printf("[%s] %s\n", c.name, c.note);
        std::printf("  per block : min %.5f ms  median %.5f ms  mean %.5f ms\n",
                    minMs, medMs, meanMs);
        std::printf("              p90 %.5f ms  p99 %.5f ms  stddev %.5f ms\n",
                    p90Ms, p99Ms, stddevMs);
        std::printf("  real-time : %.1fx budget  (uses %.2f%% of the %.4f ms block)\n",
                    headroomX, budgetPct, BLOCK_BUDGET_MS);
        std::printf("  throughput: %.2f Msample/s (median)\n", msamplePerS);
        if (diverged)
            std::printf("  WARNING   : %zu/%zu sampled blocks non-finite.\n",
                        nonfinite, iterations);
        std::printf(
            "  BENCH_JSON {\"target\":\"kernel\",\"config\":\"%s\","
            "\"noise_kind\":%lld,\"wet\":%.1f,\"voices\":%d,\"iterations\":%zu,"
            "\"chunk\":%zu,\"warmup\":%zu,\"min_ms\":%.6f,\"median_ms\":%.6f,"
            "\"mean_ms\":%.6f,\"p90_ms\":%.6f,\"p99_ms\":%.6f,\"stddev_ms\":%.6f,"
            "\"realtime_x\":%.3f,\"budget_pct\":%.3f,"
            "\"msample_per_s_median\":%.3f,\"diverged\":%s,"
            "\"nonfinite_blocks\":%zu,\"checksum\":%.15e}\n\n",
            c.name, static_cast<long long>(c.noise), c.wet, c.voices, iterations,
            CHUNK, warmup, minMs, medMs, meanMs, p90Ms, p99Ms, stddevMs, headroomX,
            budgetPct, msamplePerS, diverged ? "true" : "false", nonfinite,
            checksum);
    }

    if (std::isnan(gSink)) std::fprintf(stderr, "unreachable %g\n", gSink);
    return 0;
}
