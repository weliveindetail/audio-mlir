// Non-interactive correctness harness for the lms-noise.mlir kernel.
//
// Links the SAME --stream kernel object the CoreAudio host (lms-noise-macOS.cpp)
// drives, but plays no audio and takes no input. It runs a FIXED script of
// parameter/voice interactions through the kernel ABI and checks the rendered
// output against the kernel's expected behaviour:
//
//   * a rest with no noise  -> exact silence
//   * a rest with noise, wet=0 -> the noise is present (audible)
//   * a rest with noise, wet=1 -> the 3-tap path is nulled to ~machine zero
//     once the 32-tap LMS has converged (the "trivial" ANC case)
//   * a held note (noise=none) -> a tone at the expected fundamental
//   * a released note -> decays back to silence
//   * two held notes -> both fundamentals present (polyphony)
//   * each dsp.index_switch noise color -> active; "none" -> silence
//
// Everything is deterministic: the noise comes from a fixed in-kernel LCG seed
// and the script is fixed, so the output (and its checksum) reproduce exactly.
// Prints PASS/FAIL per case, a summary, and one machine-readable CHECK_JSON line
// (with a checksum fingerprint for exact A/B comparison across kernel builds).
//
// Build + run via lms-noise-check.sh.

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <vector>

//===----------------------------------------------------------------------===//
// DSP-MLIR kernel boundary (same ABI as lms-noise-macOS.cpp)
//===----------------------------------------------------------------------===//
static constexpr size_t BLOCK = 128;     // samples per _mlir_ciface_run call
static constexpr double FS = 44100.0;    // sample rate (for pitch analysis)

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

static constexpr int CUT_SHAPE_LEN = 8000;
extern "C" double voice_cut_shape[CUT_SHAPE_LEN]; // @voice_cut_shape : cutoff table
extern "C" double cut_lfo_step;                   // @cut_lfo_step : LFO speed for new voices

// noise_kind values: kernel has dsp.index_switch cases 0..3; anything else hits
// the default (silence). 4 is the canonical "none".
enum { NK_WHITE = 0, NK_PINK = 1, NK_BROWN = 2, NK_OU = 3, NK_NONE = 4 };
constexpr int NUM_VOICES = 8;             // MUST match the kernel's memref<8x...>

//===----------------------------------------------------------------------===//
// Kernel invocation + interaction helpers
//===----------------------------------------------------------------------===//
static long double gChecksum = 0.0L;      // running fingerprint over all captured audio

static void renderInto(double *dst) {
    MemRefDescriptor1D desc{dst, dst, 0, static_cast<int64_t>(BLOCK), 1};
    _mlir_ciface_run(&desc);
}

// Render `blocks` blocks and throw the output away (warm-up / settle / decay).
static void renderDiscard(size_t blocks) {
    double scratch[BLOCK];
    for (size_t i = 0; i < blocks; ++i) renderInto(scratch);
}

// Render `blocks` blocks into `out` (blocks*BLOCK samples) and fold the samples
// into the global checksum.
static void renderCapture(std::vector<double> &out, size_t blocks) {
    out.resize(blocks * BLOCK);
    for (size_t i = 0; i < blocks; ++i) renderInto(out.data() + i * BLOCK);
    for (double v : out) gChecksum += v;
}

static double noteToHz(int note) {
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}

// Stage a note-on/off for a voice; applied by @run at `frame` of the next block.
static void noteOn(int voice, int note, int64_t frame = 0) {
    _mlir_ciface_set_note_event(voice, noteToHz(note), 1.0, frame);
}
static void noteOff(int voice, int64_t frame = 0) {
    _mlir_ciface_set_note_event(voice, 0.0, 0.0, frame);
}
// Release every voice and let the one-pole gate smoother decay to ~0, so the
// next test starts from silence (the synth path has no other carried energy).
static void silenceAllVoices() {
    for (int v = 0; v < NUM_VOICES; ++v) noteOff(v);
    renderDiscard(64); // ~0.19 s: gate*0.99^8192 is ~1e-36, i.e. dead silence
}

//===----------------------------------------------------------------------===//
// Measurements
//===----------------------------------------------------------------------===//
static double rms(const std::vector<double> &b) {
    double s = 0.0;
    for (double v : b) s += v * v;
    return b.empty() ? 0.0 : std::sqrt(s / static_cast<double>(b.size()));
}

// Goertzel magnitude of frequency f over the whole buffer (normalised to a
// pure-tone amplitude, so a unit sine at f reads ~1.0).
static double goertzel(const std::vector<double> &b, double f) {
    const double w = 2.0 * M_PI * f / FS;
    const double coeff = 2.0 * std::cos(w);
    double s1 = 0.0, s2 = 0.0;
    for (double v : b) {
        double s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    const double real = s1 - s2 * std::cos(w);
    const double imag = s2 * std::sin(w);
    return 2.0 * std::sqrt(real * real + imag * imag) /
           static_cast<double>(b.size());
}

// One cycle of a triangle "wah" into the shared cutoff table (mirrors the native
// host's fillCutShape default): bright (0.35) at the wrap, muffled (0.02) mid.
// Exercises the @voice_cut_shape global-write path and gives the tone tests a
// real per-voice cutoff sweep instead of the flat default splat.
static void fillTriangleShape() {
    for (int i = 0; i < CUT_SHAPE_LEN; ++i) {
        double t = static_cast<double>(i) / CUT_SHAPE_LEN;
        double v = std::fabs(2.0 * t - 1.0);
        voice_cut_shape[i] = 0.02 + 0.33 * v;
    }
}

//===----------------------------------------------------------------------===//
// Tiny assertion framework
//===----------------------------------------------------------------------===//
static int gPass = 0, gFail = 0;

static void check(const char *name, bool cond, const char *fmt, ...) {
    char detail[160];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(detail, sizeof detail, fmt, ap);
    va_end(ap);
    std::printf("  [%s] %-26s %s\n", cond ? "PASS" : "FAIL", name, detail);
    if (cond) ++gPass; else ++gFail;
}

//===----------------------------------------------------------------------===//
// The fixed interaction script
//===----------------------------------------------------------------------===//
int main() {
    mu = 1.0e-3;              // small, stable LMS step (the kernel default)
    fillTriangleShape();      // shared per-voice cutoff shape
    cut_lfo_step = 1.0;       // ~5.5 Hz sweep, latched per voice at note-on

    std::printf("lms-noise.mlir correctness harness (block=%zu, fs=%.0f)\n\n",
                BLOCK, FS);

    std::vector<double> buf;

    // -- T1: a rest with no noise is exactly silent -------------------------
    // noise=none -> x=0 -> n0=0, no notes -> tone=0, LMS input 0 -> out=0.
    noise_kind = NK_NONE;
    wet = 0.0;
    silenceAllVoices();
    renderDiscard(8);
    renderCapture(buf, 64);
    double rmsSilent = rms(buf);
    check("rest_none_silence", rmsSilent < 1e-12,
          "rms=%.3e (want ~0)", rmsSilent);

    // -- T2: a rest with white noise, wet=0, is audibly noisy ---------------
    noise_kind = NK_WHITE;
    wet = 0.0;
    renderDiscard(8);
    renderCapture(buf, 128);
    double rmsNoisy = rms(buf);
    check("rest_white_wet0_noisy", rmsNoisy > 0.1,
          "rms=%.4f (want > 0.1)", rmsNoisy);

    // -- T3: a rest with white noise, wet=1, cancels to ~machine zero -------
    // Give the 32-tap LMS time to converge on the fixed 3-tap path, then the
    // residual should be a tiny fraction of the wet=0 level.
    noise_kind = NK_WHITE;
    wet = 1.0;
    renderDiscard(600);       // converge
    renderCapture(buf, 128);
    double rmsResidual = rms(buf);
    check("rest_white_wet1_cancelled",
          rmsResidual < 0.05 * rmsNoisy,
          "residual=%.3e vs noisy=%.4f (ratio %.2e, want < 0.05)",
          rmsResidual, rmsNoisy, rmsResidual / rmsNoisy);

    // -- T4: a held note (noise=none) yields its fundamental ----------------
    // A4 = 440 Hz on voice 0. Analyse a settled tail; the sawtooth's 440 Hz
    // fundamental must dominate an off-pitch probe (311 Hz, ~D#4).
    noise_kind = NK_NONE;
    wet = 0.0;
    silenceAllVoices();
    noteOn(0, 69);            // A4
    renderDiscard(32);        // attack + settle past the gate ramp
    renderCapture(buf, 64);
    double toneRms = rms(buf);
    double g440 = goertzel(buf, 440.0);
    double gOff = goertzel(buf, 311.0);
    check("note_A4_audible", toneRms > 0.01,
          "rms=%.4f (want > 0.01)", toneRms);
    check("note_A4_fundamental", g440 > 0.05 && g440 > 3.0 * gOff,
          "g440=%.4f  g311=%.4f (want g440>0.05 & >3x off)", g440, gOff);

    // -- T5: releasing the note returns to silence --------------------------
    noteOff(0);
    renderDiscard(64);        // let the gate decay
    renderCapture(buf, 32);
    double rmsAfterOff = rms(buf);
    check("note_off_silence", rmsAfterOff < 1e-6,
          "rms=%.3e (want ~0)", rmsAfterOff);

    // -- T6: polyphony -- two held notes, both fundamentals present ---------
    noise_kind = NK_NONE;
    wet = 0.0;
    silenceAllVoices();
    noteOn(0, 57);            // A3 = 220 Hz
    noteOn(1, 69);            // A4 = 440 Hz
    renderDiscard(32);
    renderCapture(buf, 64);
    double g220 = goertzel(buf, 220.0);
    double g440b = goertzel(buf, 440.0);
    check("poly_A3_present", g220 > 0.03, "g220=%.4f (want > 0.03)", g220);
    check("poly_A4_present", g440b > 0.03, "g440=%.4f (want > 0.03)", g440b);
    silenceAllVoices();

    // -- T7: every noise color is active; "none" is silent ------------------
    wet = 0.0;               // no notes, no cancellation: hear the raw color
    const int kinds[4] = {NK_WHITE, NK_PINK, NK_BROWN, NK_OU};
    const char *knames[4] = {"white", "pink", "brown", "ou"};
    for (int i = 0; i < 4; ++i) {
        noise_kind = kinds[i];
        renderDiscard(16);
        renderCapture(buf, 32);
        double r = rms(buf);
        char nm[32];
        std::snprintf(nm, sizeof nm, "noise_%s_active", knames[i]);
        check(nm, r > 0.005, "rms=%.4f (want > 0.005)", r);
    }
    noise_kind = NK_NONE;
    renderDiscard(16);
    renderCapture(buf, 32);
    double rmsNone = rms(buf);
    check("noise_none_silent", rmsNone < 1e-12, "rms=%.3e (want ~0)", rmsNone);

    // -- Summary + machine-readable line ------------------------------------
    double checksum = static_cast<double>(gChecksum);
    std::printf("\n%d passed, %d failed\n", gPass, gFail);
    std::printf(
        "CHECK_JSON {\"passed\":%d,\"failed\":%d,\"checksum\":%.15e,"
        "\"rms_noisy\":%.6f,\"rms_residual\":%.6e,\"tone_rms\":%.6f,"
        "\"g440\":%.6f}\n",
        gPass, gFail, checksum, rmsNoisy, rmsResidual, toneRms, g440);

    return gFail == 0 ? 0 : 1;
}
