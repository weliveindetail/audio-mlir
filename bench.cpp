// Micro-benchmark for the DSP-MLIR kernel (`_mlir_ciface_run`).
//
// This links the *same* compiled kernel object as the CoreAudio host
// (out/osc-low-pass-native.o) but strips away all the audio machinery, so it
// measures only the thing we actually optimize: one invocation of the generated
// `run` kernel. It is deliberately decoupled from HOW that object was built --
// rebuild the kernel through any different pipeline (fusion, vectorization,
// FFT convolution, ...) and rerun this unchanged binary for a clean A/B.
//
// Output is both human-readable and a single machine-readable `BENCH_JSON {...}`
// line so an agent can parse timings/checksum without scraping prose.
//
// Build + run via bench.sh.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Must match the kernel's memref<44200xf64> destination (44100 signal samples
// convolved with a 101-tap FIR -> 44100 + 101 - 1 output samples).
static constexpr size_t BATCH_SAMPLES = 44200; // kernel output length

// Approximate flop count for one kernel call, dominated by the 44100x101 FIR
// convolution (~8.9M mul-add) plus the ~9 elementwise 44100-sample passes.
// Only used to print a rough GFLOP/s; label it approximate.
static constexpr double APPROX_FLOPS_PER_CALL = 9.4e6;

// StridedMemRefType<double, 1>, same as the CoreAudio host.
struct MemRefDescriptor1D {
    double *basePtr;
    double *data;
    int64_t offset;
    int64_t size;
    int64_t stride;
};

extern "C" void _mlir_ciface_run(MemRefDescriptor1D *out);
extern "C" double cutoff; // the DSL's @cutoff global, read inside the kernel

// Keep the optimizer from eliminating the kernel calls / output reads.
static volatile double gSink = 0.0;

static void invoke(double *buf) {
    MemRefDescriptor1D desc{buf, buf, 0, static_cast<int64_t>(BATCH_SAMPLES), 1};
    _mlir_ciface_run(&desc);
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

int main(int argc, char **argv) {
    size_t iterations = 2000;
    size_t warmup = 200;

    // Minimal arg parsing: --iterations N, --warmup N.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = std::strtoull(argv[++i], nullptr, 10);
        else {
            std::fprintf(stderr,
                         "usage: %s [--iterations N] [--warmup N]\n", argv[0]);
            return 2;
        }
    }
    if (iterations == 0) {
        std::fprintf(stderr, "iterations must be > 0\n");
        return 2;
    }

    std::vector<double> buf(BATCH_SAMPLES);

    // Warm up: fault in pages, prime caches / branch predictors, let the CPU
    // ramp to a steady clock before we start recording.
    for (size_t i = 0; i < warmup; ++i) {
        cutoff = 200.0 + static_cast<double>(i % 100) * 78.0; // ~200..8000 Hz
        invoke(buf.data());
        gSink += buf[i % BATCH_SAMPLES];
    }

    // Timed loop: measure ONLY the kernel call. The cutoff write is a host store
    // done outside the timed region so we isolate compute from control.
    std::vector<double> samples_ms;
    samples_ms.reserve(iterations);
    using clock = std::chrono::steady_clock;
    for (size_t i = 0; i < iterations; ++i) {
        cutoff = 200.0 + static_cast<double>(i % 100) * 78.0; // sweep the sweepable range
        const auto t0 = clock::now();
        invoke(buf.data());
        const auto t1 = clock::now();
        samples_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        gSink += buf[i % BATCH_SAMPLES]; // defeat dead-code elimination
    }

    // Statistics.
    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    const double min_ms = sorted.front();
    const double median_ms = percentile(sorted, 0.50);
    const double p90_ms = percentile(sorted, 0.90);
    const double p99_ms = percentile(sorted, 0.99);
    double sum = 0.0;
    for (double v : samples_ms) sum += v;
    const double mean_ms = sum / static_cast<double>(samples_ms.size());
    double var = 0.0;
    for (double v : samples_ms) var += (v - mean_ms) * (v - mean_ms);
    const double stddev_ms = std::sqrt(var / static_cast<double>(samples_ms.size()));

    // Throughput reported off the median (robust to outliers).
    const double msample_per_s =
        (static_cast<double>(BATCH_SAMPLES) / (median_ms / 1e3)) / 1e6;
    const double gflops = (APPROX_FLOPS_PER_CALL / (median_ms / 1e3)) / 1e9;

    // Deterministic correctness fingerprint: raw kernel output at a fixed
    // cutoff. Pure codegen changes (Axis A) must keep this ~bit-stable; numeric
    // rewrites (Axis B, e.g. FFT convolution) should keep it within fp tolerance.
    cutoff = 1000.0;
    invoke(buf.data());
    double checksum = 0.0;
    for (double v : buf) checksum += v;

    std::printf("Kernel micro-benchmark (osc-low-pass _mlir_ciface_run)\n");
    std::printf("  iterations : %zu (warmup %zu)\n", iterations, warmup);
    std::printf("  per call   : min %.4f ms  median %.4f ms  mean %.4f ms\n",
                min_ms, median_ms, mean_ms);
    std::printf("               p90 %.4f ms  p99 %.4f ms  stddev %.4f ms\n",
                p90_ms, p99_ms, stddev_ms);
    std::printf("  throughput : %.2f Msample/s (median)  ~%.2f GFLOP/s (approx)\n",
                msample_per_s, gflops);
    std::printf("  checksum   : %.15e  (cutoff=1000Hz, %zu samples)\n",
                checksum, BATCH_SAMPLES);

    // Single machine-readable line for autonomous A/B comparison.
    std::printf(
        "BENCH_JSON {\"iterations\":%zu,\"warmup\":%zu,\"min_ms\":%.6f,"
        "\"median_ms\":%.6f,\"mean_ms\":%.6f,\"p90_ms\":%.6f,\"p99_ms\":%.6f,"
        "\"stddev_ms\":%.6f,\"samples_per_call\":%zu,"
        "\"msample_per_s_median\":%.6f,\"gflops_approx_median\":%.6f,"
        "\"checksum\":%.15e}\n",
        iterations, warmup, min_ms, median_ms, mean_ms, p90_ms, p99_ms,
        stddev_ms, BATCH_SAMPLES, msample_per_s, gflops, checksum);

    // Consume the sink so it cannot be optimized away.
    if (std::isnan(gSink)) std::fprintf(stderr, "unreachable %g\n", gSink);
    return 0;
}
