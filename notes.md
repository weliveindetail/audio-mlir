# Audio-MLIR — Research Notes

## Introduction

This project builds an **MLIR dialect for real-time audio DSP**, adapting the
[DSP-MLIR](https://arxiv.org/abs/2408.11205) `dsp` dialect (a large block/vector library:
FFT/DFT, FIR/IIR, windowing, correlation, DTMF/QAM, …) into a toolkit for **continuous,
block-wise audio that carries state across calls**. The long-term entry point is a
**node-based textual representation** (an audio patch graph) that the compiler lowers to
`dsp`-dialect MLIR, optimizes, and emits for a range of targets (native, WebAssembly,
and — prospectively — SIMD/matrix and DSP-core hardware).

The central design tension the project explores: *how far can a real-time audio pipeline
stay in immutable tensor land* — profiting from loop fusion, vectorization and
matrix-optimized hardware — *given that some audio components require sample-accurate
feedback that cannot be vectorized over the sample axis.* The answer we build around is
**structural decoupling**: feed-forward, closed-form work is expressed as pure tensor
maps (fuse, vectorize, tile), while sample-accurate feedback is isolated into a small set
of stateful **scan** primitives. A patch-graph frontend can then route each sub-graph to
the right regime automatically, the way Faust/Max-style engines already segment their
signal chains.

The design is deliberately biased toward **predictable, hard-real-time execution** over
peak throughput (see the dedicated section below). Everything downstream — fixed block
size, block-rate control, malloc-free kernels, per-block worst-case latency as the
metric — follows from that priority.

The compiler and dialect live in the [DSP-MLIR LLVM fork](https://github.com/weliveindetail/dsp-mlir/) 
that implements ops in `mlir/examples/dsp/SimpleBlocks/include/toy/Ops.td`,
lowering + stream/fusion passes in `mlir/LowerToAffineLoops.cpp`
and the compiler driver in `toyc.cpp`.
The repo here holds the demo kernel `lms-noise.mlir`, a native and a WASM host, simple tests, benchmark and research notes.

## Goals

1. **A `dsp` dialect for audio DSP, generated from a node-based textual representation.**
   The frontend is an audio patch graph; the compiler analyses it, emits one monolithic
   `dsp`-dialect MLIR module, and optimizes it before lowering.
2. **Structural decoupling of feed-forward vs. sample-accurate feedback.** Feed-forward
   ops stay pure tensor math; feedback recurrences are isolated into stateful scan ops.
   Generated kernels keep the two regimes separate so the optimizable part is maximised.
3. **Loop fusion and cross-module/kernel optimization.** Fuse the pointwise chain into
   shared loops; optimize *across* patched modules at the MLIR (semantic) level, above
   what LLVM LTO can see.
4. **Parallelism on SIMD and matrix/tile hardware.** Vectorize the embarrassingly-parallel
   axes (voices, channels, taps) and route matmul/tile-shaped work (mixing, convolution,
   reductions) to matrix units.
5. **Fixed-size block-wise processing, no sub-block fragmentation.** Block size is the
   number samples a kernel renders per call. It's a compile-time power-of-two constant:
   128 samples by default, 32 minimum. A kernel renders exactly `N`. No event-splitting,
   no variable block sizes.
6. **Block-aligned timestamped messages.** Support timestamped (MIDI/automation) events,
   but align them on block boundaries. Avoid artifacts with **parameter slewing** (ramp
   old→new across the block) and, where needed, block-size reduction. High-frequency
   per-sample parameter automation is out of scope.
7. **FM/RM as signal-rate input, not parameter update.** Audio-rate modulation (frequency
   modulation, ring modulation) enters the fused pipeline as a signal tensor and is
   computed sample-by-sample inside a fast loop — not delivered as host parameter writes.
8. **Polyphony and multi-channel as tensor dimensions.** Voices and channels become
   batched tensor axes that profit from vectorization and loop tiling. Voice *stealing* /
   stripping is allowed and may be target-dependent (JIT, dynamic allocation, or
   fixed SIMD-width-limited banks).

## Non-goals

- **No sub-block event-splitting / dynamic block sizes.** The whole point of the fixed
  block is a static kernel; splitting at event frames would reintroduce the variable
  control paths we are avoiding. Events are quantized to block boundaries and their
  intra-block effect is synthesised as dense control tensors (slewing), not by cutting the
  block.
- **No per-sample host-parameter automation.** Control-rate knobs update at most once per
  block (then slew). Anything that must move at audio rate is modelled as a signal input
  (goal 7), not a parameter.
- **No general bufferization pass.** Every `dsp` op stays on `tensor`s in the frontend and
  bufferizes only inside its own conversion pattern during affine lowering. This is what
  keeps the domain-specific rewrites tractable.
- **No dynamic voice count in-kernel.** A dynamic loop bound loses the compile-time trip
  count. The bank is a fixed compile-time size; host-side voice stealing hides the cap.
- **Not throughput-optimized.** The kernel is not throughput-bound; the metric is per-block
  worst-case latency (real-time headroom), not wall-clock over a long signal.

## Priority: predictable real-time execution

Audio glitches are catastrophic and non-recoverable: a single missed block deadline is an
audible dropout. So the toolkit optimizes **worst-case per-block latency**, not average
throughput, and pays for that predictability with a maximally static kernel. Real-world
motivation: pro-audio infrastructure (Dante/AVB endpoints, digital mixers at sub-ms
latency), live software instruments where CPU-headroom guarantees prevent buffer
underruns under load, composable modular sythesis patches in lock-step.

Concretely, the fixed compile-time block size `N` buys four properties we refuse to
sacrifice when adding features:

- **Compile-time trip count.** `N` is a literal, so the backend fully unrolls, constant-
  folds all block-size arithmetic, and sizes vector remainders statically — none of which a
  runtime frame count allows.
- **One optimization scope, no barriers.** The whole kernel is a single `@run` function
  with no internal call/event boundaries, so nothing forces state back to memory mid-block:
  loop-invariant work hoists freely and carried state stays register-resident across the
  block.
- **Whole-block, cross-op fusion.** An entire block is processed at once in tensor land, so
  the pointwise chain across ops fuses into shared loops over one buffer rather than being
  pinned apart by per-sample or per-node boundaries.
- **Malloc-free.** No heap allocation in the render path (`AssertNoHeapAllocPass` enforces
  0 malloc/free at compile time); every buffer is a stack `memref.alloca` or a module-scope
  state global. Deterministic memory cost = static footprint.

The tax for this determinism is that control is **block-rate**: interactive parameters are
read once per block and slewed, and events are block-aligned (see goals 5–6). This is a
deliberate trade, standard in commercial real-time engines.

---

## Current implementation state

Implemented pieces are summarised tersely here; the *future direction* section carries the
detail on what remains.

### The `dsp` dialect and `dsp1` compiler

- **Tensor-based, bufferization-pass-free.** Ops operate on `tensor`s through
  inlining/shape-inference; each op bufferizes only inside its own `ConversionPattern`
  during affine lowering (`LowerToAffineLoops.cpp`).
- **`dsp1` driver** (SimpleBlocks example, target `dsp1`). Input is a `.mlir` file in the
  `dsp` dialect; entry is `dsp.func @run(%out: memref<Nxf64>)` with
  `llvm.emit_c_interface`. Interactive knobs are `memref.global "public"` symbols
  (`@cutoff`, `@wet`, `@noise_kind`, `@mu`, …) the host reads/writes between calls.
- **`--emit=` targets:** `ast`, `mlir`, `mlir-affine`, `mlir-linalg`, `mlir-llvm`, `llvm`,
  `llvm-hexagonv68`, `wasm`, `jit`.
- **Flags:** `--opt` (fusion + scalar-replacement pipeline), `--stream` (cross-call state),
  `-o`. Native flow: `--emit=llvm [--stream] [--opt]` → `llc` → `clang++ host.cpp k.o`; the
  host calls `_mlir_ciface_run` once per block. Browser flow: `--emit=wasm` → `wasm-ld`.
- **Loop fusion (`--opt`):** after affine lowering, canonicalize + CSE, an
  `AffineFusionLegalityPass` guard (rejects the scalar-load-inside-elementwise-loop pattern
  that used to crash fusion), then affine loop fusion + scalar replacement.

### Block-wise streaming state

`--stream` runs `StreamStateMaterialization` before affine lowering: every stateful op
instance gets its own module-scope `memref.global` state buffer, resumed each call. State
is **not** threaded through `@run` (signature stays `run(%out)`). Ops inside a
`dsp.index_switch`/`variant_switch` region get independent per-case state; ops outside share
one global each. Covered ops: noise generators (`dsp.noise_white/pink/brown/ou`, LCG +
colored-filter state), `dsp.delay` (K-sample history), `dsp.lmsFilterResponse` (adaptive
taps), and the three voice-bank scans (gate, cutoff-phase, low-pass).

*Known gap:* `dsp.FIRFilterResponse` is stateless (no overlap-save history), so it can't be
streamed block-by-block without corrupting the leading `N-1` samples. `lms-noise.mlir`
avoids it by building its acoustic path from stateful `dsp.delay` + LMS.

### The state model: feed-forward tensor ops vs. feedback scan ops

The dialect splits by whether an op's output feeds back into itself — the mental model for
all new ops:

- **Feed-forward / closed-form / bounded-history → pure tensor math.** Oscillators
  (sin/saw/chirp), envelopes, FIR/convolution, `delay`, gain, mixers, waveshapers. A block
  is an embarrassingly-parallel map; cross-call state is at most a scalar phase or the last
  `N-1` input samples. This class fuses and vectorizes.
- **Feedback / recurrence → stateful scan op.** IIR/biquad, one-pole smoothers,
  integrators, envelope followers, adaptive filters (LMS). `y[n]` depends on its own past
  within the block, so it must lower to a sequential scan (`scf.for` with `iter_args`).

**Direction:** push feed-forward logic into small composable tensor ops; consolidate the
feedback class onto **one shared `dsp.scan` linear-recurrence primitive** so biquads,
smoothers and LMS reuse it instead of each being a bespoke loop. Both regimes carry state
through the same `StreamStateMaterialization` mechanism, so streaming is uniform.

### The composable primitive set (all implemented)

Five primitives replace the old hand-rolled voice loop; four feed-forward maps + one
feedback scan:

1. **`dsp.reduce`** — sum/reduction over ONE axis (`tensor<VxN> → tensor<N>`); the batched
   voice mixer. Removes the `@voice_mix` + `dsp.fromGlobal` escape hatch.
2. **Rank-2 `dsp.getRangeOfVector`** — batched ramp `y[v,n] = first[v] + n*step[v]`
   (per-voice oscillator/cutoff phase).
3. **`dsp.wavetable`** — batched table gather with linear interpolation and mod-`L` wrap
   (the shared LFO / wavetable mod source).
4. **`dsp.scan`** — one batched first-order linear recurrence
   `y[.,n] = a[.,n]·y[.,n-1] + b[.,n]·x[.,n]` with per-row carried state; covers the
   per-voice one-pole low-pass, the gate smoother, and (in matrix form) IIR/colored-noise.
   State via `StreamStateMaterialization`. First-order (k=1) only today.
5. **`dsp.eventToSignal` / `dsp.eventToTrigger`** — expand sparse `(value, frame)` events
   into a dense step/hold control tensor + a per-sample reset pulse; sample-accurate control
   without block-splitting. Split into two single-result ops (frontend-op convention).

`dsp.wrap` (branchless `x - floor(x/period)·period`) supports sawtooth wrap and table
indexing. Filter-state **reset on note-on** is done by coefficient masking (force `a=0` at
the trigger frame), so no per-sample reset operand is needed.

### Polyphonic voice bank (batched, in tensor land)

A fixed bank of **V=8** MIDI-triggered voices, rebuilt wholesale as a rank-2
`tensor<8x128>` pipeline (the hand-rolled fused loop is gone):

```
tgt   = eventToSignal(ev_gate, efrF, tgt0, 128)      // gate target step/hold
trg   = eventToTrigger(trigfr, 128)                  // note-on reset pulse
gate  = scan(0.99, 0.01, tgt)                         // one-pole gate smoother
phase = getRangeOfVector(ph_first, 128, inc)          // per-voice saw ramp
saw   = 2*wrap(phase, 1.0) - 1                         // branchless sawtooth
cphase= scan(1-trg, 1-trg, stepB)                     // cutoff LFO phase (reset@trg)
alpha = wavetable(voice_cut_shape, cphase)            // per-voice cutoff
lp    = scan((1-alpha)*(1-trg), alpha, saw)           // per-voice low-pass (reset@trg)
tone  = gain(reduce(lp * gate, axis=0), 0.2)          // mix 8 voices -> tensor<128>
```

A small 8-iteration block-level **decode** loop resolves each voice's pending MIDI event
into flat control vectors and commits carried per-voice state. Host owns voice *allocation*
(note→slot, oldest-first stealing); the kernel renders a fixed bank. Fixed V (not dynamic)
keeps the block static and malloc-free. `PromoteBuffersToStack` (in `toyc.cpp`, before the
no-heap check) stack-promotes the tiny fused-slice scratch memrefs the rank-2 chains leave.

**Current result:** the rewrite meets the *structural* goals (tensor-land end-to-end,
composable primitives, one scan primitive, escape hatch dropped) but is a **~25–28%
latency regression** vs. the old fused loop, because the V=8 voice axis needs an actual
**vectorizer** to pay off and `createAffineVectorizePass` is currently commented out of
`--opt`. Without SIMD, the batched form just pays extra buffer traffic for its ~14 rank-2
intermediates. Still ~257× real-time. Enabling the vectorizer is the next lever (below).

*Prototype limits:* one pending event per voice per block; event setters must be called
only from the render thread; with the current ring buffer live events mostly land at
frame 0 (per-sample gate ramp is correct, but tight sample-accurate live scheduling needs
closer audio-clock coupling).

### Runtime switching: `dsp.index_switch`

Runtime noise-color selection is a first-class `dsp.index_switch` (syntactically like
`scf.index_switch`: index selector, integer `case` regions, mandatory `default`,
`dsp.yield`). Born and stays in tensor land; its `ConversionPattern` rewrites to a
memref-yielding `scf.index_switch`. One function, a runtime branch, only the selected case
runs. It accepts but **ignores** an optional `reset` attribute (reserved for the planned
`variant_switch`, below).

## Reference kernel and signal chain

`lms-noise.mlir` is the canonical demo and the benchmark/perf reference (built with
`--stream`). In kernel order it exercises the full concept chain:

- sample-accurate MIDI note events → polyphonic sawtooth bank (V=8);
- a click-free one-pole gate envelope per voice;
- a per-voice swept low-pass whose cutoff is a shared wavetable LFO read at each voice's
  trigger-anchored phase;
- a mixer summing voices into a tone;
- a runtime-selectable colored-noise source (white/pink/brown/ou/none) through delay lines
  (the acoustic path);
- a second mixer burying the tone in noise;
- a 32-tap `dsp.lmsFilterResponse` adaptive filter that learns and subtracts the noise
  (the hot path, ~8.2K of ~10K flops/block);
- a final wet/dry mix revealing the tone.

Flop budget ≈ 10K flops/block (128 samples ≈ 2.9 ms at 44.1 kHz), 0 malloc/free. The
kernel is latency-bound, not throughput-bound.

*(`osc-low-pass.mlir` was the earlier FIR-lowpass demo; it is deprecated and no longer the
reference — `lms-noise` supersedes it.)*

## Testing (WIP: `lms-noise-check`)

`lms-noise-check.sh` builds the kernel exactly like the CoreAudio host
(`--stream [--opt]`, `OPT=0` to disable) but links the headless checker
`lms-noise-check.cpp` instead of the audio driver. The checker drives a fixed interaction
script through the kernel ABI and asserts output behaviour — **13 checks** covering:
silence-at-rest, white-noise audibility, LMS cancellation (residual RMS drop), single-note
audibility + fundamental, note-off silence, polyphonic partials, per-noise-color activity,
and none-selected silence. It prints `PASS`/`FAIL` per case plus a machine-readable
`CHECK_JSON {... "checksum": ...}` line for exact A/B comparison across builds.

- The **checksum** is the correctness fingerprint. Pure codegen changes (Axis A) must keep
  it ~bit-stable; numeric rewrites (Axis B, interpolation, reassociation) keep it within fp
  tolerance — the 13 threshold checks are then authoritative, not bit-identity.
- Current golden checksum after the tensor-land voice-bank rewrite:
  `-2.749372436647042e+03` (a ~1e-10 shift from the pre-rewrite value, from wavetable lerp
  + scan-phase float rounding + fusion reassociation).

Usage: `./lms-noise-check.sh [kernel.mlir]` (`OPT=0` optional).

## Benchmarking (WIP: `lms-noise-bench`)

`lms-noise-bench.sh` builds the same `--stream [--opt]` kernel and links the headless
timing driver `lms-noise-bench.cpp` (perf only; correctness is the checker). It measures
**per-block render latency** — the real-time-relevant metric — across four representative
configs:

- `rest_silent` — index_switch default (silence) + idle LMS;
- `anc_white` — white noise + active 32-tap LMS cancel;
- `synth_poly8` — 8-voice sawtooth bank + per-voice cutoff + reduce;
- `full_white_poly8` — full bank + white noise + LMS cancel.

Each run prints a human summary plus one `BENCH_JSON {"config":..,"min_ms":..,
"median_ms":..,"stddev_ms":..,"msample_per_s_median":..,"checksum":..}` line per config for
autonomous A/B. Compare `min_ms` (most stable) or `median_ms`; lower is better. The script
also dumps the kernel object's **static memory footprint** (`size -m`) — since the kernel
is malloc-free, that static size is its entire memory cost (`@voice_cut_shape`, an
8000×f64 = 64 KB wavetable, dominates `__data`).

Usage: `./lms-noise-bench.sh [kernel.mlir] [-- --iterations N --warmup N]` (`OPT=0`
optional).

What `--opt` currently buys on `lms-noise` (vs. plain): `affine.for` 28→19 (some pointwise
fusion), `memref.alloca` 46→32. Both pipelines are already 0 malloc/free. `--opt` does
**not** yet vectorize (zero NEON fp ops) — the biggest open win.

## Future direction

Ordered roughly by leverage.

### 1. Enable the affine vectorizer (the primary lever)

`createAffineVectorizePass` is commented out of `--opt`. Enabling it is what makes the
batched voice axis (V=8) and the LMS tap loop actually SIMD — and turns the voice-bank
rewrite's structural win into a speed win (the ~25–28% regression is entirely the missing
vectorizer). After enabling, verify NEON (`fmla v*.2d`) in the disassembly.

- **A1 — vectorize the LMS 32-tap loops (NEON).** The dot-product and weight-update inner
  `0..32` tap loops are the target (the outer sample loop carries the adaptive recurrence
  and stays sequential). Now unblocked: **A2 is done** — `LMSFilterResponseOpLowering`
  splits the sample loop at `splitN = FilterLength-1 = 31` into a *head* loop (keeps the
  `affine.if (n-i>=0)` history-boundary guard, for the first taps and `--stream`
  continuity) and a *steady* loop (guard-free, pure current-block loads). Only enabling the
  vectorizer remains.
- **A4 — finish elementwise fusion.** Fusion collapsed part of the pointwise chain; several
  128-element loops (t, sawtooth, tone, delay taps, mix) remain unfused.
- **A3/A5 — heap/stack (done).** All static buffers ≤64 KB stack-promote to
  `memref.alloca`; `AssertNoHeapAllocPass` enforces 0 heap at compile time.

### 2. LMS decomposition roadmap (hot path → composable ops)

The LMS is ~82% of flops/block; a "substantial" win lives here, but only *paired with the
vectorizer* (decomposition without SIMD regressed the voice bank).

- **Layer 0 (done = A2).** Head/steady sample-loop split; tap bodies factored into
  `emitDotTap`/`emitUpdateTap` helpers. No new ops, no materialized window buffer,
  bit-identical.
- **Layer 1 (deferred).** Promote those helpers to real ops `dsp.windowedDot`
  (`y[n]=Σ w[i]·x[n-i]`) and `dsp.rank1Update` (`w += scale·x-window`), reusable by FIR /
  correlation / NLMS / RLS. *Do this only once a second consumer exists* — with LMS as sole
  user it is speculative op surface. (These are recurrence *bodies*, not standalone tensor
  ops: `w` changes every sample.)
- **Layer 2 (deferred, sets up B1).** Generalize `dsp.scan` to carry a **k-vector state**
  (`memref<Vxkf64>`). LMS is then one state-space instantiation
  `w[n] = (I - mu·xₙxₙᵀ)·w[n-1] + mu·d[n]·xₙ` (rank-1-perturbed-identity transition),
  sharing one carrier with biquads (k=2) and colored-noise/IIR. Keep the structured rank-1
  form (don't materialize the dense 32×32 A). Single place to derive parallel-scan blocking.

### 3. Axis B — domain-specific rewrites

Legal only because the compiler understands DSP semantics (a generic C compiler won't do
them). The DSP-MLIR value proposition; stacks on top of good codegen.

- **B1 — block/parallelize the adaptive-filter recurrence.** LMS *is* linear in `w`
  (state-space form above), so it can be reassociated into a transition-matrix scan. This
  is Layer 2. Caveat: products of `(I - mu·xxᵀ)` don't stay rank-1, so composing
  transitions for a parallel scan is the genuinely hard part.
- **B2 — hoist loop-invariant DSP work.** Coefficients/state that change only when a knob
  moves (e.g. the one-pole sweep alpha, recomputed per block from the LFO) can be lifted and
  refreshed only on change — a DSP-aware invariant-hoisting pass.
- **B3 — filter/stage fusion + algebraic identities.** Cascade/fuse consecutive filters,
  fold constant gains, simplify gain/delay compositions at the dialect level.

### 4. Target-portable `dsp.scan` (associative / parallel-scan form)

`dsp.scan` (`y[n]=a[n]·y[n-1]+b[n]·x[n]`) lowers sequentially today — optimal for narrow
SIMD/VLIW DSP cores where the voice axis V already fills the vector unit. For targets wider
than V (AVX-512 with V≤8, GPUs, TPUs, wide HVX) the N axis is the serialization and must be
broken.

Each step is an affine map `f_n(y) = a[n]·y + c[n]` (with `c[n]=b[n]·x[n]`); composition is
affine and associative, giving the monoid combine
`(A_hi,C_hi)⊕(A_lo,C_lo) = (A_hi·A_lo, A_hi·C_lo + C_hi)`, identity `(1,0)`. An inclusive
prefix scan over the N pairs yields cumulative `(A,C)`, then `y[n] = A₀..ₙ·y[-1] + C₀..ₙ`.
Cost `log₂N` parallel steps (N=128 → 7) instead of N sequential ones.

- **Schedules:** Hillis-Steele (`N·logN` work, `logN` depth — in-lane SIMD) or Blelloch
  (`2N` work, up/down-sweep — GPU/TPU).
- **Key the lowering on a target width** (pass option / op attribute): `sequential`
  (default) or `associative`. The stream-state global and reset-as-coefficient-masking both
  survive (`a[f]=0` at a reset frame zeroes `A_cum` from that frame on).
- One rewrite retargets *every* feedback op (per-voice low-pass, gate, pink/brown/ou noise,
  LMS). Time-varying coefficients are free (the affine-pair form needs no constant-coeff
  assumption). Correctness: fp-tolerance band, not bit-equality.

### 5. Target encodings for the voice/scan work

Beyond the M1 Pro (NEON, 2-wide f64):

- **Newer Apple arm64 (M4 / A17+).** Adds SVE2 + SME/SME2 (tile/matrix). SME does *not*
  help the sequential scan; its win is `dsp.reduce` over V and the voice mix-down
  (outer-product/matmul-shaped: `FMOPA` f64, ZA tile). Route reduce/mix through
  Linalg→Vector→ArmSME; keep the scan on NEON `vector<2xf64>`. (M2/M3 are still plain NEON.)
- **Hexagon (Qualcomm HVX).** 1024-bit but integer/fixed-point — needs a Q-format
  reformulation (Q15 coeffs/samples, Q31 accumulators) and a scale-analysis pass. `vrmpy`
  dot-products map onto FIR taps + reduce; hardware circular addressing matches `dsp.wrap`
  and delay lines. No upstream dialect — lower fixed-point to LLVM + HVX intrinsics. Changes
  the checksum model to fixed-point error.
- **Dedicated DSP cores (SHARC / TI C6000 / Tensilica HiFi).** The kernel's native habitat
  (f32, VLIW ILP across the 8 voice lanes, zero-overhead loops, hardware circular buffers).
  Keep the *sequential* scan (their loops/addressing make it cheap; the associative rewrite
  would only add work). No upstream MLIR path — emit portable LLVM IR/C for the vendor
  toolchain.

### 6. Cross-module / node-graph frontend

Model each synth module as an MLIR function/op inside one top-level `builtin.module`; emit
the user's whole patch as one monolithic MLIR file and run a fast optimization pipeline
over it — MLIR sees blocks/graphs/multi-dim iterations (semantic level), unlike LLVM LTO's
flattened loops. Structural decoupling holds *across* modules as long as the compiler
controls the top-level composition. **Risk:** a feedback loop that spans multiple modules —
on a detected cross-module cycle, either reject the patch or fall back to compiling that
sub-graph as a sample-by-sample vector loop (bypassing the pure tensor pipeline). This is
the natural home for goal 1's node-based frontend, which today is hand-written `.mlir`.

### 7. Timestamped events → per-parameter tensors

Generalize the `dsp.eventToSignal` pattern so every control param becomes a dense
value-over-time tensor computed from block-aligned `(value, frame)` events — sample-accurate
without block-splitting and without growing the `@run` ABI. This turns control-rate knobs
(gate, cutoff automation, `@wet`) into feed-forward tensor math. Requires moving parameters
from plain globals to setter-functions carrying the MIDI timestamp, while preserving the
four real-time optimization properties. FM/RM (goal 7) stay signal-rate inputs, not events.
