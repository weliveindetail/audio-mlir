// Correctness / quality driver for the LMS hum-canceller kernel
// (`_mlir_ciface_run`, output length 44100). Links the same kernel object the
// CoreAudio host uses, but strips the audio machinery.
//
// It renders one buffer and reports:
//   * checksum   -- sum of samples, for exact A/B comparison of the built-in-op
//                   kernel (lms-hum.mlir) vs the spelled-out pure-MLIR kernel
//                   (lms-hum-pure.mlir). Same numbers => the hand-written LMS
//                   matches dsp.lmsFilterResponse.
//   * hum/tone   -- Goertzel magnitude at 60 Hz and 440 Hz on the *second half*
//                   of the buffer (after the weights have converged), so we can
//                   confirm the 60 Hz hum is suppressed while the 440 Hz tone
//                   survives.
//
// Build + run via lms-check.sh.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr size_t N = 44100;   // kernel output length (1 s @ 44.1 kHz)
static constexpr double FS = 44100.0;

struct MemRefDescriptor1D {
    double *basePtr;
    double *data;
    int64_t offset;
    int64_t size;
    int64_t stride;
};

extern "C" void _mlir_ciface_run(MemRefDescriptor1D *out);
extern "C" double mu; // the DSL's @mu global, read inside the kernel

static void invoke(double *buf) {
    MemRefDescriptor1D desc{buf, buf, 0, static_cast<int64_t>(N), 1};
    _mlir_ciface_run(&desc);
}

// Goertzel magnitude of frequency f over buf[start..start+len).
static double goertzel(const double *buf, size_t start, size_t len, double f) {
    const double w = 2.0 * M_PI * f / FS;
    const double coeff = 2.0 * std::cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = 0; i < len; ++i) {
        s0 = buf[start + i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    const double real = s1 - s2 * std::cos(w);
    const double imag = s2 * std::sin(w);
    return 2.0 * std::sqrt(real * real + imag * imag) / static_cast<double>(len);
}

int main() {
    std::vector<double> buf(N);
    invoke(buf.data());

    double checksum = 0.0;
    for (double v : buf) checksum += v;

    // Analyze the converged tail (second half of the buffer).
    const size_t half = N / 2;
    const double hum60 = goertzel(buf.data(), half, N - half, 60.0);
    const double tone440 = goertzel(buf.data(), half, N - half, 440.0);

    std::printf("LMS hum-canceller kernel check (mu=%g, %zu samples)\n", mu, N);
    std::printf("  checksum        : %.15e\n", checksum);
    std::printf("  tail 440 Hz mag : %.6f  (tone, want ~1.0 preserved)\n", tone440);
    std::printf("  tail  60 Hz mag : %.6f  (hum, want << 0.7 suppressed)\n", hum60);
    std::printf(
        "LMS_JSON {\"mu\":%.6f,\"samples\":%zu,\"checksum\":%.15e,"
        "\"tone440\":%.6f,\"hum60\":%.6f}\n",
        mu, N, checksum, tone440, hum60);
    return 0;
}
