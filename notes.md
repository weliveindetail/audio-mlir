# DSP-MLIR: Oscillator + Low-Pass Filter — Research Notes

## What DSP-MLIR Is

A compiler infrastructure extending LLVM/MLIR with a custom DSP dialect. The `.py` files in `test/Examples/DspExample/` are **not Python** — they are a custom DSL compiled by `build-relwithdebinfo/bin/dsp1`. The DSL processes static, fixed-size arrays and compiles to MLIR intermediate representation.

---

## Can the DSL Represent an Oscillator + LPF Pipeline?

**Yes — both are expressible.**

### Oscillator (sine wave)
No dedicated `oscillator()` primitive exists, but one can be composed from three ops:

```dsp
var pi   = 3.14159265359;
var freq = 440.0;
var dt   = 0.000022675736;              # 1 / 44100
var time = getRangeOfVector(0, 44100, dt);
var phase   = gain(time, 2 * pi * freq);
var osc     = sin(phase);
```

- `getRangeOfVector(first, N, step)` — generates a time/phase array
- `gain(tensor, scalar)` — scales every element
- `sin(tensor)` — element-wise sine (`SinOp`, `Ops.td` line 1125, `MLIRGen.cpp` line 694)

### Low-Pass FIR Filter
Exact match to the DSP-MLIR idiom shown in `dsp_biomedical.py`:

```dsp
var Fs  = 44100;
var fc  = 1000.0;
var N   = 101;
var wc  = 2 * pi * fc / Fs;    # normalized cutoff, range 0..pi
var lpf   = lowPassFIRFilter(wc, N);
var lpf_w = lpf * hamming(N);
var filtered = FIRFilterResponse(osc, lpf_w);
print(filtered);
```

- `lowPassFIRFilter(wc, N)` — designs FIR coefficients (`LowPassFIRFilterOp`, `Ops.td` line 1482)
- `hamming(N)` — Hamming window to smooth the filter
- `FIRFilterResponse(signal, coeffs)` — convolves signal with filter (`FIRFilterResponseOp`, `Ops.td` line 673)

**Critical:** `wc` must be normalized as `2 * pi * fc / Fs` (range 0 to pi, as commented in `dsp_biomedical.py`). The oscillator's `dt` and the filter's `Fs` must use the same sample rate.

---

## JIT Execution — Key Facts

The compiler supports `--emit=jit` which compiles and runs the program, printing results to stdout.

| Signal size | JIT time |
|-------------|----------|
| 3 samples (gain only) | ~22 ms |
| 200 samples (osc + LPF) | ~234 ms |
| 4 410 samples (0.1 s at 44 100 Hz) | ~255 ms |
| 44 100 samples (1 s at 44 100 Hz) | **~89 ms** |

Larger signals are *faster* because JIT compilation overhead is constant and dominates small inputs. For 1-second audio segments the round-trip is ~89 ms — acceptable for interactive filter sweeps.

**Output format:** space-separated floats on stdout.
Output length = `input_length + filter_taps - 1` (linear convolution). For 44 100 input + 101 taps → 44 200 values.

---

## Constraints of the DSL

| Capability | Status |
|------------|--------|
| Static fixed-size signal processing | yes |
| Oscillator via `getRangeOfVector` + `sin` + `gain` | yes |
| FIR LPF via `lowPassFIRFilter` + `hamming` + `FIRFilterResponse` | yes |
| Scalar arithmetic (`2 * pi * fc / Fs`) | yes |
| Real-time streaming / callbacks | no |
| Runtime parameters (change cutoff without recompile) | no |
| Audio output (WAV, speakers) | no |

---

## Integration Strategy for an Interactive Python App

Because the DSL can't stream or react to runtime input, the split is:

```
DSL file    → formal specification of the signal processing graph
Python app  → real-time audio streaming + keyboard control
```

**Practical approach (~89 ms latency on cutoff change):**

1. Python writes a temporary DSL file with the current `fc` substituted in
2. Calls `dsp1 --emit=jit tmp.py` in a subprocess
3. Parses the float array from stdout into a numpy array
4. Feeds it into `sounddevice` as a looping audio buffer
5. On Up/Down keypress, spawns a background thread to recompile and swap buffers

---

## Relevant File Locations

| Path | Purpose |
|------|---------|
| `build-relwithdebinfo/bin/dsp1` | DSL compiler / JIT runner |
| `DSP_MLIR/mlir/examples/dsp/SimpleBlocks/include/toy/Ops.td` | All op definitions (TableGen) |
| `DSP_MLIR/mlir/examples/dsp/SimpleBlocks/mlir/MLIRGen.cpp` | Op name to MLIR mapping |
| `DSP_MLIR/mlir/test/Examples/DspExample/dsp_biomedical.py` | Best reference for LPF idiom |
| `DSP_MLIR/mlir/test/Examples/DspExample/dsp_abs_argmax.py` | `getRangeOfVector` usage example |
| `samples/osc-low-pass.py` | Existing working Python app (sounddevice + pygame, IIR filter) |
