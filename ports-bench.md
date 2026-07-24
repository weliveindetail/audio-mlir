# lms-noise ports — benchmark summary

Runtime micro-benchmark comparing the two `lms-noise` ports (Faust, Cmajor)
against the DSP-MLIR kernel. Generated + run by
[`ports-bench.sh`](ports-bench.sh) (ports) and
[`lms-noise-bench.sh`](lms-noise-bench.sh) (kernel); both share the harness math
so the three targets are directly A/B-able apart from language/toolchain.

## Methodology

- **Unit of work:** one 128-sample block per render call. That block *lasts*
  `128 / 44100 ≈ 2.9025 ms` of audio — the real-time budget.
- **Metric:** per-block **compute latency** (`median_ms` and friends). `xRT`
  (`realtime_x`) is the derived headroom `budget / median_ms`; throughput is
  `msample_per_s`. See "xRT vs latency" below.
- **Configs:** `rest_silent`, `anc_white` (white noise + active 32-tap LMS),
  `synth_poly8` (8-voice sawtooth bank), `full_white_poly8` (both).
- **Run:** 3000 timing samples × 64 blocks each, 1000-block warmup.
- **Precision:** ports are **f32** (as shipped); the kernel is **f64**. This is
  the dominant confound in the comparison (see `samples/notes.md`).
- **Host:** Apple Silicon (Darwin arm64), `clang++ -O3 -std=c++17`.

## Results (median per-block)

| target | config | latency (ms) | xRT | throughput (Msample/s) |
|---|---|---:|---:|---:|
| kernel (f64) | rest_silent      | 0.01090 | 266× | 11.75 |
| kernel (f64) | anc_white        | 0.01115 | 260× | 11.49 |
| kernel (f64) | synth_poly8      | 0.01086 | 267× | 11.79 |
| kernel (f64) | full_white_poly8 | 0.01117 | 260× | 11.46 |
| cmajor (f32) | rest_silent      | 0.00795 | 365× | 16.10 |
| cmajor (f32) | anc_white        | 0.00825 | 352× | 15.52 |
| cmajor (f32) | synth_poly8      | 0.00795 | 365× | 16.11 |
| cmajor (f32) | full_white_poly8 | 0.00826 | 352× | 15.51 |
| faust (f32)  | rest_silent      | 0.01448 | 200× |  8.84 |
| faust (f32)  | anc_white        | 0.01447 | 201× |  8.84 |
| faust (f32)  | synth_poly8      | 0.01448 | 201× |  8.84 |
| faust (f32)  | full_white_poly8 | 0.01447 | 201× |  8.85 |

**Takeaways**

- All three are far inside real-time (200×–365× headroom); this is a hot-loop
  micro-benchmark, not an audio-dropout risk assessment.
- **Cmajor is fastest** (~16 Msample/s), ~40 % faster than the f64 kernel — most
  of which is the f32/f64 width difference.
- **Faust lands *below* the kernel** (~8.8 Msample/s) despite being f32. The
  hand-wired 8-voice + effect chain computes **all 8 voices every block**
  (matching the kernel's always-on `tensor<8x128>` bank) with no vectorization,
  which costs more than expected.
- Cost is **flat across configs** for every target — the LMS/synth hot loops
  dominate and the noise-color `index_switch`/`enable` selection is cheap.

## Static memory footprint

Per generated DSP object (`size` sections; all three compute paths are
malloc-free, so static size ≈ whole cost).

| object | dominant section | bytes | note |
|---|---|---:|---|
| kernel `.o`        | `__data` 65 608          | ~89 k | one shared `@voice_cut_shape` `8000×f64` = 64 000 B |
| Faust `FaustVoice` | `__bss` 32 000           | ~35 k | one **shared** `8000×f32` rdtable (all 8 voices read it) |
| Faust `FaustEffect`| `__text` 3 496           | ~5.8 k | LMS + noise, tiny state |
| Cmajor `LMSNoise`  | `__common` 259 056       | ~259 k | **8× copies** of the `8000×f32` cutoff table (256 000 B) |

- Faust and the kernel keep **one** shared cutoff-shape table; the Cmajor port
  stores it **per voice** (8 copies ⇒ ~256 kB), which explains its much larger
  footprint. Worth revisiting if memory matters — the table is read-only and
  could be a single shared node.

## Numerical fidelity note

The `wet=1` white-noise configs (`anc_white`, `full_white_poly8`) go
**non-finite** on the Faust port (flagged `"diverged":true`, WARNING line). Root
cause: the ports' LMS taps are **pure, un-leaked integrators**, so the weight
vector slowly random-walks and eventually overflows to `inf` — Faust at ~296 k
samples for `mu=1e-3`, scaling ~`1/mu`. It reproduces in **f64 too**, so it is
not a precision artifact. The f64 **kernel stays finite** over the full 24 M-sample
run, and the **Cmajor port** also stays finite here, so this is a **real
Faust-port fidelity gap**, not a bench bug. Timing is unaffected (identical
flop count per block), so the harness still reports valid latency and just flags
the run. A leakage term (`w ← (1−ε)·w + …`) on the Faust taps would close the gap.

##  vs latency

`xRT` is **not** latency — it is the dimensionless
`xRT = 2.9025 ms / median_ms`, i.e. how many times faster than real-time a block
renders. Latency is `median_ms` (and `min/p90/p99/stddev_ms`); the two are
inverses: `latency = 2.9025 / xRT` ms.

## Signal latency (todo)

Note that xRT is the headroom ratio, which depends on the **compute/render latency**.
This is the time to *produce* a block. Input→output delay is the **signal latency**
and we don't capture it yet. Signal latency is dominated by the
128-sample block buffering (~2.9 ms) plus the LMS/one-pole group delay, and
possibly differing between the kernel and each port. Worth adding a separate
measurement (e.g. impulse/chirp in, cross-correlate the output against the input
to recover the delay per target) so we can compare latency as *heard*, not just
compute time.
