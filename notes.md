# Research Notes

## The `dsp` dialect (work in progress)

We are adapting the [DSP-MLIR](https://arxiv.org/abs/2408.11205) `dsp` dialect for
audio processing. Upstream `dsp` is a large library of block/vector DSP ops (FFT/DFT,
FIR/IIR filters, windowing, correlation, DTMF/QAM, ...); on top of it we added and
adjusted ops for **continuous, block-wise audio that carries state across calls**. The
audio-relevant stateful additions are the colored-noise generators `dsp.noise_white` /
`noise_pink` / `noise_brown` / `noise_ou`, the `dsp.delay` line, and the adaptive
`dsp.lmsFilterResponse`; the samples build pipelines from these plus stateless blocks
like `dsp.getRangeOfVector` (ramp/broadcast), `dsp.mul/add/sub/div/modulo`, `dsp.sin`,
`dsp.gain`, `dsp.hamming`, `dsp.lowPassFIRFilter`, `dsp.FIRFilterResponse`, and the
runtime selector `dsp.index_switch` / `dsp.yield`. Three properties define the WIP design:

- **Tensor-based, bufferization-pass-free.** Every `dsp` op operates on `tensor`s in the
  frontend and stays in tensor land through inlining/shape inference. There is no general
  bufferization pass by design: each op only bufferizes inside its own `ConversionPattern`
  during affine lowering (`LowerToAffineLoops.cpp`), allocating/deallocating its memrefs
  locally. Keeping the pipeline at the tensor level is what enables the domain-specific
  (Axis B) rewrites listed under Performance Improvements below.
- **Block-wise processing with carried state.** Under `--stream`, a
  `StreamStateMaterialization` pass runs just before affine lowering and gives every
  stateful op instance its own module-scope `memref.global` state buffer (noise gens: LCG
  seed + colored-filter accumulators; `dsp.delay`: a K-sample history line;
  `dsp.lmsFilterResponse`: the adaptive tap weights). The kernel's C entry point
  `_mlir_ciface_run` is then called once per block and each call resumes where the last
  left off. State is **not** threaded through the `@run` signature, which stays
  `run(%out: memref<Nxf64>)`. Ops inside a `dsp.index_switch` region get independent
  per-case state; ops outside share one global each.
- **Loop fusion after bufferization.** `--opt` runs, after affine lowering,
  canonicalize + CSE, an `AffineFusionLegalityPass` guard, then affine loop fusion and
  affine scalar replacement. The legality pass rejects (with a clear diagnostic) the one
  pattern that used to crash fusion: a 0-D/scalar memref loaded *inside* an elementwise
  loop, e.g. `dsp.gain`'s broadcast. The samples avoid it by broadcasting scalars to
  vectors with `getRangeOfVector` instead of `dsp.gain`.

Compiler-side code lives in `../DSP_MLIR/mlir/examples/dsp/SimpleBlocks` (ops in
`include/toy/Ops.td`; lowering + the stream/fusion passes in `mlir/LowerToAffineLoops.cpp`;
driver in `toyc.cpp`). This repo (`samples/`) holds only demos and docs.

## Using the `dsp1` compiler

The driver is the SimpleBlocks example built as the CMake target `dsp1` (source
`toyc.cpp`): `ninja -C ../build-relwithdebinfo dsp1` produces
`../build-relwithdebinfo/bin/dsp1`. The sample scripts prepend that `bin/` to `PATH` and
invoke `dsp1 <pipeline>.mlir --emit=<target> [flags] [-o out]`.

- **Input** is a `.mlir` file in the `dsp` dialect (see `osc-low-pass.mlir`,
  `lms-noise.mlir`). The pipeline entry is
  `dsp.func @run(%out: memref<Nxf64>) attributes {llvm.emit_c_interface}`; interactive
  knobs are `memref.global "public"` symbols (`@cutoff`, `@wet`, `@noise_kind`, `@mu`)
  that the host reads/writes between calls. Passing a `.dsp` source instead needs `-x dsp`.
- **`--emit=` targets** (verified via `dsp1 --help-hidden`): `ast`, `mlir` (dsp dialect as
  parsed), `mlir-affine` (after lowering -- inspect generated loops/allocs), `mlir-linalg`,
  `mlir-llvm`, `llvm` (LLVM IR text), `llvm-hexagonv68`, `wasm` (a `.wasm` object), `jit`
  (compile and run in-process via `main`).
- **Flags:** `--opt` enables the fusion/scalar-replacement pipeline above; `--stream`
  enables `StreamStateMaterialization` for cross-call state; `-o <file>` sets the output
  (default stdout, for `--emit=llvm/wasm`). `--affineIn`, `--affineOpt`, `--canonOpt`
  exist for partial pipelines.

Native build flow (what `bench.sh` and `*-macOS.sh` do):
`dsp1 pipeline.mlir --emit=llvm [--stream] [--opt] -o k.ll`
→ `llc k.ll -filetype=obj -o k.o`
→ `clang++ host.cpp k.o ... -o app`.
The host links the `_mlir_ciface_run` symbol and calls it (per block under `--stream`). For
the browser demo, `--emit=wasm` then `wasm-ld --export=_mlir_ciface_run --export=<knobs>`
(see `sample-wasm.sh`). To inspect a pipeline without a host:
`dsp1 pipeline.mlir --emit=mlir-affine --opt`.

## Uncertainty algebra (a deferred probabilistic-type track)

Context for a design we deliberately did **not** build, kept so the reasoning isn't lost.
Motivated by the `noise-lang` DSL (`../noise-lang`), where every value is a *random
variable* and `P(cond)` / `E(x)` are estimated by Monte Carlo, the idea was a first-class
`!dsp.rv<f64>` type: arithmetic ops lift over it, `dsp.sample_normal` introduces it, and
query ops (`dsp.prob`/`expect`/`variance`) collapse it back to a number by lowering to an
ordinary affine/scf sampling loop. MLIR is a surprisingly good host: SSA already encodes
noise-lang's "one name = one fixed draw" sharing rule; a query is a clean lowering, not a
new runtime; and RVs compose with the existing `dsp` ops (e.g. `E[residual power]` after
the LMS canceller).

Why it stays deferred: the algebra **cannot express stateful time recurrences**, which is
exactly what our audio ops are. noise-lang samples independent lanes with no state across a
time index, whereas the LMS update `w[n] = w[n-1] + mu*e[n]*x[n-i]` and the delay/noise
streams are precisely such recurrences. The RV type could only *wrap* the deterministic
kernel (run it over many seeds and estimate output statistics), not model its core. Given
that, plus the large ODS/verifier surface to lift every op and the seeding/cost-model work,
it isn't worth it now. What actually helps the audio pipeline -- first-class colored-noise
**signal generators** in the `dsp` dialect -- is already implemented
(`dsp.noise_white/pink/brown/ou`, `noise-kinds.mlir`).

## Block-based processing -- status

Block-wise streaming now works via module-scope state globals (`--stream` +
`StreamStateMaterialization`, above) rather than the extra state *parameters* this section
originally predicted: `@run` stays `run(%out)` and state lives in globals, not in a grown
`run(out, n, fc, phase_in_out, hist_in_out)` ABI. What that machinery covers and what it
still does **not**:

- **Covered:** per-op stream state for the stateful ops. `dsp.delay` keeps its K-sample
  history line, `dsp.lmsFilterResponse` keeps its adaptive weights, and the noise
  generators keep their LCG/colored-filter state. This is what makes the LMS canceller
  (`lms-noise.mlir`) seamless across calls.
- **Addressed (lms-noise) -- oscillator/tone phase continuity.** The tone now resumes from a
  persistent `@sample_offset` counter: the kernel reads it, starts the time base at
  `offset*dt`, and stores `offset+N` back each call -- "implicit" global state, no host
  involvement. This required generalizing `dsp.getRangeOfVector` to accept a *dynamic*
  `first`/`step`: it previously hard-required `dsp.constant` operands (and null-deref-crashed
  on anything else), and now constant-folds when it can, else loads the scalar at runtime and
  feeds it to the same iter_arg recurrence. The constant path is unchanged (osc-low-pass
  checksum bit-identical). Still open: `osc-low-pass.mlir`'s sawtooth uses the same
  `getRangeOfVector(0, N, dt)` idiom and hasn't been converted, so a streamed sawtooth there
  would still click.
- **Addressed (lms-noise) -- small block size + block-synchronous host.** The kernel block is
  now N=128 samples per `@run` call (was 512, originally 44100 = one second); only the
  tensor/memref shapes and the `%n`/`%cnt` count constants change. 128 samples ≈ 2.9 ms ≈ a
  344 Hz block rate, small enough that once-per-block ("control-rate") parameter automation
  stays smooth. The CoreAudio host (`lms-noise-macOS.cpp`) was reworked to match: the old
  double-buffer + 100 ms `regenerate` + wraparound loop is replaced by a background render
  thread that produces 128-sample blocks into a lock-free SPSC ring buffer, drained by the
  audio callback. No host-side loop remains -- `--stream` state and `@sample_offset` make each
  block the true next 128 samples. The host also pins the hardware buffer
  (`kAudioDevicePropertyBufferFrameSize` + `kAudioUnitProperty_MaximumFramesPerSlice`); the
  ring makes playback correct even if the OS renegotiates a different slice size.
- **Addressed (lms-noise) -- automated control-rate parameter.** The demo now ends with a
  one-pole low-pass (`dsp.lowPassFilter`, a stateful op with a `memref<1xf64>` `state` global
  holding `y[n-1]`) whose cutoff coefficient `alpha` is *automated* in-kernel: a pure-arith
  triangle LFO over `@sample_offset` (period 147000 samples ≈ 3.3 s) sweeps alpha in
  [0.02, 0.35]. alpha is read once per block (control-rate), matching the "read params before
  loops, not inside them" rule; the LFO is deterministic in the sample counter, so it is
  walltime-insensitive like the rest of the kernel. Demonstrates block-rate automation without
  any host knob.
- **Addressed (lms-noise) -- zero heap per call (RT-safe kernel).** `insertAllocAndDealloc`
  now stack-promotes any statically-shaped buffer ≤ 64 KiB to `memref.alloca` (hoisted to the
  function entry block, no dealloc) instead of `memref.alloc`+`memref.dealloc`. At N=128 every
  buffer (≤ 1 KiB for a 128×f64) qualifies, so the emitted `.ll` has **0 malloc / 0 free**
  (was ~16 pairs per call). The last holdout was the `scf.index_switch` (noise-color selector)
  result: its post-conversion fix-up in `ToyToAffineLoweringPass` used to unconditionally emit
  one `memref.dealloc` on the switch result at the end of the block. Now that the case regions
  yield stack allocas, that dealloc is both a stray `free` and undefined (freeing a stack
  pointer); the fix-up only emits it for results every region yields via a heap `memref.alloc`.
  The kernel is now safe to call from the audio callback, though the host still uses the ring
  for slice-size decoupling.
- **Addressed (lms-noise) -- compile-time malloc-free guarantee (on by default).** Zero heap
  *for this kernel at N=128* is not by itself a guarantee: buffers over the 64 KiB stack-
  promotion threshold, the handful of op lowerings that create `memref.alloc` directly
  (bypassing `insertAllocAndDealloc`), dynamically-shaped buffers, and heap results escaping an
  `scf.index_switch` would all still malloc. A verification pass (`AssertNoHeapAllocPass`,
  module-level) runs **by default** for any runnable target (`--emit=llvm` and beyond), after
  affine lowering but before memref->LLVM, and fails the compile (exit 4) if *any* `memref.alloc`
  survives -- checking the actual emitted IR rather than reasoning about sizes, so it catches
  every source. The error points at the offending op's source location. So a future edit that
  reintroduces a heap buffer (e.g. bumping N past 8192, or adding a non-promoted op) breaks the
  build instead of silently regressing real-time safety.
  `--allow-heap` lifts the requirement: it downgrades the hard error to a warning (same message,
  ` (allowed by --allow-heap)` suffix) so the kernel still builds, just no longer guaranteed
  RT-safe -- useful for offline/large kernels (e.g. `osc-low-pass.mlir`'s 44200-sample buffers).
  NB: toyc registers no MLIR diagnostic handler, so the default engine drops warnings and prints
  only errors; the pass therefore prints the `--allow-heap` warning to stderr itself, in the
  same `<loc>: warning: ...` shape. (Sanity-checked: lms-noise N=128 passes by default;
  osc-low-pass fails by default and builds-with-5-warnings under `--allow-heap`.)
- **Addressed (lms-noise) -- LMS input-history continuity (item 3).** `dsp.lmsFilterResponse`
  now carries a *second* per-instance stream-state global (`state_hist`, alongside the weights
  `state`): a `memref<(taps-1)xf64>` holding the previous block's last `taps-1` input samples.
  `StreamStateMaterializationPass` materializes it and the op's lowering wires it in: the two
  FIR loops gained an `else` branch on their `n - i >= 0` guard that, when `n - i < 0`, reads
  `history[(taps-1) + n - i]` instead of zero-padding, and a small tail loop after the sample
  loop saves `x[N-(taps-1) .. N-1]` back into the history for the next call. So the FIR sum
  reaches into the real prior block at boundaries -- the per-block edge transient (the ~86 Hz
  buzz at N=512) is gone, and even the old once-per-second glitch at N=44100 disappears. Zero-
  initialized, so the very first block still zero-pads (nothing precedes the stream); the
  non-streaming path is unchanged (no `state_hist` -> old behavior). This is exactly the
  overlap-save history the stateless FIR below still lacks.
- **Unaddressed -- FIR overlap-save history.** `dsp.FIRFilterResponse` is *not* a stateful
  op: it has no per-block history global and emits the full `len + N - 1` linear
  convolution. Streaming it block-by-block would corrupt the leading N-1 samples of each
  block. `lms-noise.mlir` sidesteps this because its acoustic path is built from stateful
  `dsp.delay` ops and the LMS filter, not `FIRFilterResponse`. Overlap-save history for the
  FIR is the biggest open piece for a truly block-based low-pass.
- **Unaddressed -- coefficient crossfade on cutoff change.** Windowed-sinc taps are cheap
  to recompute per call, but swapping them instantaneously causes a small discontinuity; a
  crossfade between old/new coefficient sets is not implemented (acceptable for the demo).

## Feedforward tensor ops vs. feedback scan ops (the state model that scales)

The dialect splits cleanly into two categories by whether an op's output feeds back into
itself. This split decides what can stay pure tensor math and what needs a carried-state
loop, and it's the mental model to design new ops around.

- **Feedforward / closed-form / bounded-history -> pure tensor math.** Oscillators
  (sin/saw/chirp), envelopes (exp), FIR/convolution, `delay`, gain, mixers, waveshapers.
  The output is a function of the time index or of the input with a *bounded* look-back, so
  a block is an embarrassingly-parallel map. Cross-call state is at most a single scalar (a
  phase / sample offset) or the last N-1 input samples (overlap-save). This is the class
  that fuses and vectorizes, and where the Axis-B rewrites apply. `dsp.delay` is the
  bounded-history example (carries K samples). A stateful oscillator/time base is the
  scalar-state example: `sin(2*pi*f*t[n])` is pure math *because sine has a closed form in
  n* -- the only thing carried across blocks is the phase, never a sequential loop.
- **Feedback / recurrence -> stateful scan op.** IIR/biquad `y[n] = a*y[n-1] + b*x[n]`,
  one-pole smoothers, integrators, envelope followers, PLLs, and adaptive filters (the LMS
  weight update `w[n] = w[n-1] + mu*e[n]*x[n-i]`). Here `y[n]` depends on its *own past
  within the same block*, so there is no elementwise expression; it must lower to a
  sequential scan (`scf.for` with `iter_args`) carrying the recurrence state. Today each
  such op is a bespoke hand-rolled loop: `dsp.noise_pink/brown/ou` carry 6 recurrence
  values, `dsp.lmsFilterResponse` carries its tap weights.

**Design direction.** Keep pushing feedforward logic into small composable pure-tensor ops.
For the feedback class, introduce a shared set of **stateful "scan" ops** -- a general
`dsp.scan` / linear-recurrence op (or a state-space / biquad op) -- so biquads, smoothers,
and LMS reuse *one* stateful-loop primitive instead of each being reimplemented. Both
categories carry their state through the same `StreamStateMaterialization` mechanism (a
module-scope global per op instance), so block-wise streaming is uniform across them.

Escape hatch (not needed now): feedback isn't fundamentally stuck in sequential land --
linear recurrences can be blocked in parallel via an associative scan or a state-space
transition-matrix form (`y_block = f(state, x_block)`) instead of a naive map. Worth knowing
the option exists before committing to always-sequential scan lowering.

## Performance Improvements

The `_run` object generated today (Toy-tutorial lowering) is one heap buffer + one full
loop per DSP op, every scalar constant heap-allocated, ~25 malloc/free per call, a scalar
bounds-checked convolution, and no fusion. It looks alarming but most of the bloat is
one-time setup/teardown that is NOT on the hot path.

Flop budget for one `regenerate()` (44100-sample signal, 101-tap FIR):
- ~9 elementwise loops (t, phase, ones, frac, saw2, saw, wc, coef, ...): ~9 x 44100 ~= 0.4M
- coefficient build (sinc + hamming, with sin/cos): 101 taps, negligible
- FIR convolution (44200 x up to 101): ~4.5M mul-add  <- ~90% of runtime
- ~25 malloc + ~25 free: one-time, a few microseconds (<0.1% vs a few ms of compute)

So the convolution dominates and its inner loop is already the same arithmetic C would emit.
The naive version is within ~2-4x of hand C, not 20x. There are two independent axes of
improvement; do not conflate them. Axis A is backend codegen quality (matching C). Axis B is
the domain-specific algebraic rewrites the DSP-MLIR paper measures.

### Measuring

`bench.sh` builds the kernel object (from `osc-low-pass.mlir`) and a standalone driver
(`bench.cpp`) that links the same `_mlir_ciface_run` as the CoreAudio host but strips the
audio machinery, so it times ONLY the generated kernel. It is decoupled from how the object
is built: change a pass/pipeline, rerun `./bench.sh`, diff the output. Usage:
`./bench.sh [--iterations N] [--warmup N]`.

Each run prints a human summary plus one machine-readable line an agent can parse:
`BENCH_JSON {"min_ms":..,"median_ms":..,"stddev_ms":..,"msample_per_s_median":..,"checksum":..}`.
- Compare `min_ms` (most stable) or `median_ms`; lower is better.
- `checksum` is the raw kernel output at cutoff=1000Hz. Axis A (pure codegen) must keep it
  ~bit-stable; Axis B (numeric rewrites like FFT convolution) should keep it within fp
  tolerance. A changed checksum on an Axis-A change means the optimization broke correctness.

`bench.sh` builds with `--opt` by default; pass `OPT=0 ./bench.sh` to build the plain
baseline for an A/B. Baselines on this machine (checksum 1.769572322574197e+01, identical
both ways):
- plain pipeline (`OPT=0`): min ~3.86 ms, median ~4.05 ms/call, ~11 Msample/s.
- `--opt` (default): min ~3.44 ms, median ~3.59 ms/call, ~12 Msample/s (~12% faster). This
  is the number to beat.

What `--opt` already bought (vs plain): affine.for loops 13->9, memref.alloc 24->13,
44100-elt intermediate buffers 7->4, 0-D scalar memrefs 49->21. It did NOT vectorize (zero
NEON fp ops), did NOT hoist the convolution bounds check (the `affine.if` zero-pad guard is
still in the tap loop), and left 16 malloc/free in the emitted IR. So the hot-path wins
(A1/A2) and all of Axis B remain open. See the per-task status below.

### Axis A -- backend codegen quality

- [ ] A1. Vectorize the FIR convolution (NEON). The inner loop at the convolution site is
  scalar (fmul/fadd per tap); SIMD is the single biggest win (~2-4x) and the only change
  that buys real speed here. Enable MLIR vectorization (e.g. affine/vector lowering, or
  `-affine-super-vectorize`) so the tap loop emits vector fmla. Verify NEON (`fmla v*.2d`)
  appears in the disassembly of the convolution loop.
- [ ] A2. Hoist bounds checks out of the convolution inner loop. The zero-padding
  `tbnz ...,#0x3f` tests fire every iteration (~4.5M times). Split the output range into
  head (partial overlap), steady-state (full 101-tap, no checks), and tail regions -- what a
  C programmer does by hand. This unblocks A1 (a clean steady-state loop vectorizes; a
  branchy one does not).
- [~] A3. Stop heap-allocating scalar constants. Each `tensor<f64>` constant (pi, fs, N, 2.0,
  1.0, freq=440, dt, cutoff-load, ...) lowers to a malloc(8) + store + free. Lower 0-D
  constants to SSA values (arith.constant / plain f64) instead of a memref. Deletes roughly
  half the `bl malloc`/`bl free` pairs and all the associated stores. Cleanup + a little
  speed, low risk.
  PARTIAL (--opt): affine scalar replacement + the sawtooth rewrite cut 0-D scalar memrefs
  49->21, but the emitted IR still has 16 malloc/free. Not fully lowered to SSA yet.
- [~] A4. Fuse the elementwise loops. The sawtooth is 9 separate passes over 9 separate
  44100-element buffers. Affine loop fusion (`-affine-loop-fusion`) collapses the pointwise
  chain (t -> phase -> frac -> saw2 -> saw) into one pass over one buffer, cutting memory
  traffic ~9x and removing the intermediate allocations. Confirm the intermediate buffers
  disappear from the object.
  PARTIAL (--opt): fusion is now enabled and collapsed part of the chain (loops 13->9,
  44100-elt buffers 7->4). Not a single pass yet -- 4 elementwise loops remain; the chain
  is not fully fused into one.
- [~] A5. Buffer hoisting / reuse / stack promotion. Add buffer-deallocation +
  buffer-loop-hoisting; promote small fixed-size buffers (the 101-tap arrays) to stack
  (memref.alloca) so they never hit malloc. Pairs with A3/A4 to eliminate the remaining
  one-time allocation overhead.
  PARTIAL (--opt): total memref.alloc 24->13, but only as a side effect of A4 removing
  intermediates. No stack promotion happened (memref.alloca count = 0); mallocs remain.

Status legend: [ ] = open, [~] = partially addressed by `--opt`, [x] = done.
A1 and A2 are untouched by `--opt` and are the real hot-path wins.

Order to attempt: A3 + A4 first (clean up the IR and remove allocations, low risk) -- now
partly done by `--opt`, finish them off, then A2 (makes the hot loop branch-free), then A1
(vectorize the now-clean loop -- the real win), then A5 (mop up remaining allocations).
Measure regenerate() wall-time before/after each.

### Axis B -- domain-specific optimizations

These are legal only because the compiler understands DSP semantics; a generic C compiler
(and often a C programmer) will not do them automatically. They are the DSP-MLIR value
proposition and stack on top of good codegen.

- [ ] B1. Time-domain FIR -> FFT convolution. Rewrite the 44100 x 101 convolution as
  O(N log N) via FFT (multiply spectra, inverse FFT). This is the classic order-of-magnitude
  DSP win and a pure high-level dialect rewrite -- the single most illustrative change to
  demonstrate the paper's optimizations, since the whole 44200 x 101 nested loop disappears.
- [ ] B2. Coefficient hoisting (loop-invariant DSP). The windowed-sinc taps only change when
  `cutoff` changes; recomputing sinc + hamming (with sin/cos) on every regenerate() is waste.
  A DSP-aware invariant-hoisting pass lifts coefficient computation out of the per-batch path
  and recomputes only when cutoff moves.
- [ ] B3. Exploit input periodicity (linear -> circular convolution). The 440 Hz tone is
  exactly periodic over 44100 samples, so the filtered output is periodic; folding the FIR
  tail onto the head (out[n] += out[44100+n]) reconstructs circular convolution. The C++ host
  already does this by hand -- a DSP dialect that recognizes a periodic input could emit it
  automatically and drop the 100-sample tail work.
- [ ] B4. Filter/stage fusion and algebraic identities. Cascade/fuse consecutive filters,
  fold constant gains, simplify gain/delay compositions at the dialect level before lowering.
  Minor here (single filter stage) but the general mechanism behind the paper's speedups.

## Runtime switching (`dsp.index_switch`) vs. compile-time specialization (`dsp.variant_switch`)

Runtime noise-color selection is implemented today by **`dsp.index_switch`**, a
first-class dsp op. It is syntactically a near-clone of `scf.index_switch` (an
index selector, integer `case` regions, a mandatory `default`, a common result
type, `dsp.yield` terminator) but is a distinct dsp op so it is born and stays in
tensor land through the frontend and only bufferizes during affine lowering --
exactly like every other dsp op in this (deliberately bufferization-pass-free)
toolchain. Its `ConversionPattern` in `LowerToAffineLoops.cpp`
(`IndexSwitchOpLowering` + `YieldOpLowering`) rewrites it to a memref-yielding
`scf.index_switch`, which then lowers through `cf` to LLVM. This is the
**intended, final** lowering of `dsp.index_switch`: one function, a runtime
branch, only the selected case runs. There is no "fallback" framing -- this is
the whole op.

`dsp.index_switch` accepts an optional `reset` attribute but **ignores** it; the
attribute is reserved for the sibling op below.

### Planned: `dsp.variant_switch` (compile-time specialization / unswitch)

`dsp.variant_switch` is a **separate, explicit** op offering the compile-time
specialization ("unswitch") alternative to the runtime branch. It shares
`dsp.index_switch`'s syntax (same selector / `case` / `default` / `dsp.yield`
shape, same optional `reset`) but has fundamentally different lowering
semantics and, crucially, **no fallback**: where it cannot be applied it raises a
compile-time error rather than silently degrading to a runtime branch. (If a
runtime branch is what you want, write `dsp.index_switch`.)

Semantics:

- **Unswitch the enclosing kernel.** Instead of emitting one function with a
  runtime branch, the specialization pass clones the enclosing `@run` once per
  case (plus the default), with that case's region inlined in place of the
  switch, and constant-folds/specializes each clone. The selector then drives a
  small **dispatcher** that calls the matching specialized clone. Each variant is
  a straight-line kernel with no switch overhead, which is what makes the ops
  inside it independently optimizable (per-case shape inference, fusion, etc.).
- **State model (the reason `reset` exists).** Ops **inside** the switch regions
  (the noise generators) get **independent** stream-state globals -- one per
  case -- so each color keeps its own stream. Ops **outside** the switch (the two
  delays, the LMS weights, the tone phase) share **one** state global each across
  all variants. `reset = "region_local"` means: when the selector changes between
  calls, cleanly restart **only** the noise (region-local) state, while the
  shared LMS/delay/tone state persists across the switch -- "reset noises only".
  This is a real behavioral difference from `dsp.index_switch`, where all streams
  simply persist per case.
- **Error where it can't apply (no fallback).** The pass must diagnose and reject
  cases it cannot specialize, e.g.: the selector is not a compile-time-tractable
  value it can build a dispatcher for; a region result type it cannot thread
  through the cloned signature; or a state-sharing pattern that violates the
  inside/outside partition. Each is a clear `emitError`, never a silent runtime
  branch.

### Implementation steps

1. **ODS.** Add `def VariantSwitchOp : Dsp_Op<"variant_switch", ...>` mirroring
   `IndexSwitchOp` (index `arg`, `DenseI64ArrayAttr` cases, optional `reset`,
   variadic results, `SizedRegion<1>` default + `VariadicRegion` cases,
   `dsp.yield` terminator, verifier). It can reuse the `parseIndexSwitchCases` /
   `printIndexSwitchCases` custom directive. Extend `dsp.yield`'s `ParentOneOf`
   to accept both `IndexSwitchOp` and `VariantSwitchOp`.
2. **KernelSpecialization pass** (module-level, runs before
   `StreamStateMaterialization` so per-clone noise ops each get their own state
   global via the existing identity walk):
   - Locate each `dsp.variant_switch` and its enclosing kernel function.
   - For each case + default, clone the function, inline that region's body in
     place of the switch (RAUW the switch result with the region's yielded
     value), and drop the other regions.
   - Materialize the independent per-variant noise state and the shared
     outside-state, honoring `reset`: region-local state globals are reset on a
     selector change; shared globals persist.
   - Build a dispatcher (the original entry) that reads the selector and calls
     the matching specialized clone; the default clone handles the fall-through.
   - `emitError` and fail the pass on any non-specializable construct (see the
     error list above) -- no fallback to `scf.index_switch`.
3. **Pipeline wiring.** Add `createKernelSpecializationPass()` to `Passes.h` and
   run it in `toyc.cpp` under `isLoweringToAffine`, before
   `StreamStateMaterialization` and `LowerToAffine`. The specialized clones then
   lower through the ordinary per-op affine patterns; there is no residual
   `dsp.variant_switch` for `LowerToAffine` to see.
4. **Tests.** A variant of `lms-noise.mlir` that swaps `dsp.index_switch` for
   `dsp.variant_switch`, checking (a) N+1 specialized clones + a dispatcher in
   the lowered IR, (b) independent noise state per clone, (c) reset-on-switch of
   region-local state only, and (d) a clean diagnostic on a deliberately
   non-specializable selector.

### Next steps (beyond the initial version)

- **Arbitrary SSA selectors.** v1 assumes the selector is a plain load of a named
  global (`@noise_kind`). Generalize the dispatcher to build from an arbitrary
  `index` SSA value, falling back to `emitError` only when it truly cannot form a
  dispatch (rather than assuming the global shape).
- **Multiple `dsp.variant_switch` in one kernel.** Specializing N switches
  independently is a cross product of clones; decide whether to nest dispatchers
  (product blow-up) or restrict to one switch per kernel initially and diagnose
  the rest. Document the chosen policy when implemented.

DSP session:
claude --resume 48701461-7c6a-4dd0-a967-120d220de8e7

WASM session:
claude --resume 9a199408-3c7a-4e8e-ae36-733eb5ad685c
