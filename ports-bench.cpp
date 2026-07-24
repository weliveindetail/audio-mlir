// Runtime micro-benchmark for the two lms-noise PORTS (Faust + Cmajor).
//
// This is the port-side twin of ../lms-noise-bench.cpp: same block size, same
// real-time budget, same representative configs, and the SAME statistics + one
// BENCH_JSON line per (target,config) -- so the ports and the DSP-MLIR kernel can
// be A/B'd apart from the language/toolchain. As with the kernel bench this is
// perf only (correctness is elsewhere): each timed unit renders one 128-sample
// block (~2.9 ms of audio at 44.1 kHz), so the meaningful metric is PER-BLOCK
// render latency vs that budget, NOT wall-clock over a long signal.
//
// Both ports are reduced to the same "one full pipeline per block" shape the
// kernel exposes:
//   * Faust: the port is polyphonic via [nvoices:8]; here we instantiate the
//     8 voice DSPs (the `process` class) + the mono `effect` DSP by hand, sum the
//     voices and feed the effect -- exactly what the faust2* poly architecture
//     does, and exactly the kernel's `d = sum_v tone_v + n0; out = d - wet*y`.
//     ALL 8 voices are always computed (gated by note), matching the kernel's
//     always-on tensor<8x128> bank rather than voice-stealing.
//   * Cmajor: the generated graph already contains the 8 voices + master, so we
//     just push MIDI note-ons, set the value endpoints and advance() a block.
//
// The f32 ports vs the f64 kernel is the dominant confound; see samples/notes.md.
//
// Built + run via ports/bench.sh (which generates the three headers first).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- Faust runtime base (dsp / UI / Meta) + label->zone binder --------------
#include <faust/dsp/dsp.h>
#include <faust/gui/meta.h>
#include <faust/gui/UI.h>
#include <faust/gui/MapUI.h>

// ---- the three generated port classes ---------------------------------------
#include "out/FaustVoice.h"   // struct FaustVoice  : the polyphonic `process` (one voice)
#include "out/FaustEffect.h"  // struct FaustEffect : the mono `effect` (noise + LMS)
#include "out/LMSNoiseCmaj.h" // struct LMSNoise    : the whole Cmajor graph

//===----------------------------------------------------------------------===//
// Shared constants (identical to ../lms-noise-bench.cpp)
//===----------------------------------------------------------------------===//
static constexpr size_t BLOCK = 128;   // samples per rendered block
static constexpr double FS = 44100.0;  // sample rate
static constexpr double BLOCK_BUDGET_MS = 1000.0 * BLOCK / FS; // ~2.9024 ms
static constexpr size_t CHUNK = 64;    // blocks rendered between two clock reads
static constexpr int NUM_VOICES = 8;   // MUST match [nvoices:8] / the kernel bank

enum { NK_WHITE = 0, NK_PINK = 1, NK_BROWN = 2, NK_OU = 3, NK_NONE = 4 };
static constexpr double MU_DEFAULT = 1.0e-3; // kernel default LMS step

static volatile double gSink = 0.0; // keep the optimizer honest

static double noteToHz(int note) {
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}

//===----------------------------------------------------------------------===//
// Representative configurations (identical set to the kernel bench)
//===----------------------------------------------------------------------===//
struct Config {
    const char *name;
    int64_t noise; // noise_kind
    double wet;
    int voices;    // held voices (a chord); 0 = rest
    const char *note;
};

static const Config kConfigs[] = {
    {"rest_silent", NK_NONE, 0.0, 0, "silence selector + idle LMS"},
    {"anc_white", NK_WHITE, 1.0, 0, "white noise + active 32-tap LMS cancel"},
    {"synth_poly8", NK_NONE, 0.0, 8, "8-voice sawtooth bank + per-voice cutoff + reduce"},
    {"full_white_poly8", NK_WHITE, 1.0, 8, "synth bank + white noise + LMS cancel"},
};

// A spread-out chord so the 8 voices run at distinct pitches/phases.
static const int kChord[NUM_VOICES] = {36, 43, 48, 52, 55, 60, 64, 67};

//===----------------------------------------------------------------------===//
// Engine interface: setup(config) then renderBlock() BLOCK samples at a time.
//===----------------------------------------------------------------------===//
struct Engine {
    virtual ~Engine() = default;
    virtual const char *target() const = 0;
    virtual void setup(const Config &c) = 0;
    virtual void renderBlock(double *out) = 0; // writes BLOCK samples
};

//===----------------------------------------------------------------------===//
// Faust engine: 8 hand-wired voice DSPs summed into the mono effect DSP.
//===----------------------------------------------------------------------===//
struct FaustEngine : Engine {
    FaustVoice voices[NUM_VOICES];
    FaustEffect effect;
    MapUI voiceUI[NUM_VOICES];
    MapUI effectUI;
    // per-voice scratch + mix/effect buffers (block-at-a-time compute)
    float voiceBuf[NUM_VOICES][BLOCK];
    float mixBuf[BLOCK];
    float effBuf[BLOCK];

    FaustEngine() {
        for (int v = 0; v < NUM_VOICES; ++v) {
            voices[v].init(static_cast<int>(FS));
            voices[v].buildUserInterface(&voiceUI[v]);
        }
        effect.init(static_cast<int>(FS));
        effect.buildUserInterface(&effectUI);
    }

    const char *target() const override { return "faust"; }

    void setup(const Config &c) override {
        // fresh state so a config never inherits the previous one's tails/weights
        for (int v = 0; v < NUM_VOICES; ++v) {
            voices[v].instanceClear();
            voiceUI[v].setParamValue("gain", 1.0f);
            voiceUI[v].setParamValue("freq",
                                     static_cast<float>(noteToHz(kChord[v])));
            voiceUI[v].setParamValue("gate", v < c.voices ? 1.0f : 0.0f);
        }
        effect.instanceClear();
        effectUI.setParamValue("Noise Cancel", static_cast<float>(c.wet));
        effectUI.setParamValue("LMS Rate", static_cast<float>(MU_DEFAULT));
        effectUI.setParamValue("Noise Color", static_cast<float>(c.noise));
    }

    void renderBlock(double *out) override {
        for (int v = 0; v < NUM_VOICES; ++v) {
            float *op = voiceBuf[v];
            voices[v].compute(static_cast<int>(BLOCK), nullptr, &op);
        }
        for (size_t i = 0; i < BLOCK; ++i) {
            float s = 0.0f;
            for (int v = 0; v < NUM_VOICES; ++v) s += voiceBuf[v][i];
            mixBuf[i] = s;
        }
        float *ip = mixBuf;
        float *op = effBuf;
        effect.compute(static_cast<int>(BLOCK), &ip, &op);
        for (size_t i = 0; i < BLOCK; ++i) out[i] = static_cast<double>(effBuf[i]);
    }
};

//===----------------------------------------------------------------------===//
// Cmajor engine: the generated graph does its own voice allocation + master.
//===----------------------------------------------------------------------===//
struct CmajEngine : Engine {
    LMSNoise proc;
    float outBuf[BLOCK];

    static constexpr uint32_t H_WET = 2, H_MU = 3, H_SWEEP = 4, H_COLOR = 5,
                              H_OUT = 6;

    CmajEngine() { proc.initialise(0, FS); }

    const char *target() const override { return "cmajor"; }

    static int32_t noteOnMsg(int note, int vel) {
        return 0x900000 | ((note & 0x7f) << 8) | (vel & 0x7f);
    }

    void setup(const Config &c) override {
        proc.reset(); // clear state (voices, LMS weights, smoothers)
        float wet = static_cast<float>(c.wet);
        float mu = static_cast<float>(MU_DEFAULT);
        float sweep = 3.0f; // port default Sweep Rate
        int32_t color = static_cast<int32_t>(c.noise);
        proc.setValue(H_WET, &wet, 0);
        proc.setValue(H_MU, &mu, 0);
        proc.setValue(H_SWEEP, &sweep, 0);
        proc.setValue(H_COLOR, &color, 0);
        for (int v = 0; v < c.voices; ++v) {
            LMSNoise::std_midi_Message m{noteOnMsg(kChord[v], 100)};
            proc.addEvent_midiIn(m);
        }
    }

    void renderBlock(double *out) override {
        proc.advance(static_cast<int32_t>(BLOCK));
        proc.copyOutputFrames(H_OUT, outBuf, static_cast<uint32_t>(BLOCK));
        for (size_t i = 0; i < BLOCK; ++i) out[i] = static_cast<double>(outBuf[i]);
    }
};

//===----------------------------------------------------------------------===//
// Timing (identical math to ../lms-noise-bench.cpp)
//===----------------------------------------------------------------------===//
static double percentile(const std::vector<double> &sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double idx = p * (static_cast<double>(sorted.size()) - 1.0);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static void benchOne(Engine &eng, const Config &c, size_t iterations,
                     size_t warmup) {
    eng.setup(c);

    std::vector<double> buf(BLOCK);
    // warm up (gates ramp, LMS/caches/CPU clock settle) -- discarded
    for (size_t i = 0; i < warmup; ++i) eng.renderBlock(buf.data());

    std::vector<double> perBlockMs;
    perBlockMs.reserve(iterations);
    double checksum = 0.0;   // accumulated over FINITE samples only
    size_t nonfinite = 0;    // blocks whose sampled output was inf/nan
    using clock = std::chrono::steady_clock;
    for (size_t i = 0; i < iterations; ++i) {
        const auto t0 = clock::now();
        for (size_t b = 0; b < CHUNK; ++b) eng.renderBlock(buf.data());
        const auto t1 = clock::now();
        const double chunkMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        perBlockMs.push_back(chunkMs / static_cast<double>(CHUNK));
        // The timing above is valid regardless of value stability. But some
        // ports' un-leaked LMS integrators random-walk to inf over millions of
        // wet=1 samples (see note in bench.sh); guard the checksum and flag it.
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
    const double stddevMs =
        std::sqrt(var / static_cast<double>(perBlockMs.size()));

    const double msamplePerS = (static_cast<double>(BLOCK) / (medMs / 1e3)) / 1e6;
    const double headroomX = BLOCK_BUDGET_MS / medMs;
    const double budgetPct = 100.0 * medMs / BLOCK_BUDGET_MS;

    std::printf("[%s/%s] %s\n", eng.target(), c.name, c.note);
    std::printf("  per block : min %.5f ms  median %.5f ms  mean %.5f ms\n",
                minMs, medMs, meanMs);
    std::printf("              p90 %.5f ms  p99 %.5f ms  stddev %.5f ms\n", p90Ms,
                p99Ms, stddevMs);
    std::printf("  real-time : %.1fx budget  (uses %.2f%% of the %.4f ms block)\n",
                headroomX, budgetPct, BLOCK_BUDGET_MS);
    std::printf("  throughput: %.2f Msample/s (median)\n", msamplePerS);
    if (diverged)
        std::printf("  WARNING   : %zu/%zu sampled blocks non-finite -- this "
                    "port's LMS drifts to inf here; timing is still valid.\n",
                    nonfinite, iterations);
    std::printf(
        "  BENCH_JSON {\"target\":\"%s\",\"config\":\"%s\",\"noise_kind\":%lld,"
        "\"wet\":%.1f,\"voices\":%d,\"iterations\":%zu,\"chunk\":%zu,"
        "\"warmup\":%zu,\"min_ms\":%.6f,\"median_ms\":%.6f,\"mean_ms\":%.6f,"
        "\"p90_ms\":%.6f,\"p99_ms\":%.6f,\"stddev_ms\":%.6f,\"realtime_x\":%.3f,"
        "\"budget_pct\":%.3f,\"msample_per_s_median\":%.3f,\"diverged\":%s,"
        "\"nonfinite_blocks\":%zu,\"checksum\":%.15e}\n\n",
        eng.target(), c.name, static_cast<long long>(c.noise), c.wet, c.voices,
        iterations, CHUNK, warmup, minMs, medMs, meanMs, p90Ms, p99Ms, stddevMs,
        headroomX, budgetPct, msamplePerS, diverged ? "true" : "false", nonfinite,
        checksum);
}

int main(int argc, char **argv) {
    size_t iterations = 3000;
    size_t warmup = 1000;
    bool doFaust = true, doCmaj = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--faust-only") == 0)
            doCmaj = false;
        else if (std::strcmp(argv[i], "--cmajor-only") == 0)
            doFaust = false;
        else {
            std::fprintf(stderr,
                         "usage: %s [--iterations N] [--warmup N] "
                         "[--faust-only|--cmajor-only]\n",
                         argv[0]);
            return 2;
        }
    }
    if (iterations == 0) {
        std::fprintf(stderr, "iterations must be > 0\n");
        return 2;
    }

    std::printf("lms-noise PORTS runtime benchmark (block=%zu, fs=%.0f)\n", BLOCK,
                FS);
    std::printf("  real-time budget per block: %.4f ms (1.0x = just keeps up)\n",
                BLOCK_BUDGET_MS);
    std::printf("  timing: %zu samples x %zu blocks each, %zu-block warmup\n\n",
                iterations, CHUNK, warmup);

    if (doFaust) {
        FaustEngine eng;
        for (const Config &c : kConfigs) benchOne(eng, c, iterations, warmup);
    }
    if (doCmaj) {
        CmajEngine eng;
        for (const Config &c : kConfigs) benchOne(eng, c, iterations, warmup);
    }

    if (std::isnan(gSink)) std::fprintf(stderr, "unreachable %g\n", gSink);
    return 0;
}
