# Research Notes

## The `dsp` dialect for audio (work in progress)

We are adapting the [DSP-MLIR](https://arxiv.org/abs/2408.11205) `dsp` dialect for
audio processing. Upstream `dsp` is a large library of block/vector DSP ops (FFT/DFT,
FIR/IIR filters, windowing, correlation, DTMF/QAM, ...); on top of it we added and
adjusted ops for **continuous, block-wise audio that carries state across calls**. The
audio-relevant stateful additions are the colored-noise generators `dsp.noise_white` /
`noise_pink` / `noise_brown` / `noise_ou`, the `dsp.delay` line, and the adaptive
`dsp.lmsFilterResponse`; the samples build pipelines from these plus stateless blocks
like `dsp.getRangeOfVector` (ramp/broadcast), `dsp.mul/add/sub/div/modulo`, `dsp.sin`,
`dsp.gain`, `dsp.hamming`, `dsp.lowPassFIRFilter`, `dsp.FIRFilterResponse`, and the
runtime selector `dsp.index_switch` / `dsp.yield`. Four properties define the WIP design:

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
- **Fixed block size, block-rate control.** N is a compile-time constant (128 in the
  samples): `@run` renders exactly N samples per call. We don't plan to change our
  approach towards event-splitting. Right now, interactive parameters
  (`@noise_kind`, `@lfo_period`, etc.) are read once at the top of the call and held
  constant for the whole block. That means they are applied in control-rate and not
  sample-accurate. This is a deliberate trade-off that keeps the kernel maximally
  static. In the future, we want to calculate a tensor for each parameter that
  represents its value over time. For that, we have to change parameters from global
  variables to setter-functions that provide the MIDI timestamp of the event. If
  possible, we don't want to sacrifice any of the following optimizations:
    - **Compile-time trip count.** The N=128 loop bound is a literal, so the backend can
      fully unroll, constant-fold all block-size arithmetic, and size vector remainders
      statically -- none of which is possible with a runtime frame count.
    - **One optimization scope, no barriers.** The whole kernel is a single `@run` function
      with no internal call/event boundaries, so nothing forces state back to memory or
      blocks reordering mid-block: loop-invariant work hoists freely and carried state can
      stay register-resident across the block.
    - **Whole-block, cross-op fusion.** Because an entire block is processed at once in
      tensor land, the pointwise chain across ops (`t -> frac -> saw2 -> saw`, etc.) fuses
      into shared loops over one buffer (the Axis-A A4 work) rather than being pinned apart
      by per-sample or per-node boundaries.

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

The reference kernel for this section is `lms-noise.mlir` (the streaming adaptive
noise-canceller), built with `--stream`: `_mlir_ciface_run(%out: memref<128xf64>)` is called
once per 128-sample block and resumes from module-scope state globals. Its per-block op chain
is:
* a colored-noise generator: one of four, behind `dsp.index_switch`
* a sawtooth tone through a swept one-pole low-pass
* two `dsp.delay` lines
* 32-tap `dsp.lmsFilterResponse` adaptive filter subtracts the correlated noise (hot path)

Flop budget for one block (128 samples ~= 2.9 ms of audio at 44.1 kHz):
- LMS filter output `y[n] = sum_k w[k]*x[n-k]`: 128 x 32 ~= 4.1K mul-add
- LMS weight update `w[k] += mu*e[n]*x[n-k]`: 128 x 32 ~= 4.1K mul-add  <- these two are the
  hot path (the outer sample loop is a sequential adaptive recurrence; the 32-tap inner loop
  is parallel)
- active noise-color scan (only the selected `index_switch` case runs): 128 x ~6 ~= 0.8K
- sawtooth + one-pole sweep + two delays + wet/dry mix (~8 elementwise 128-passes): ~1.5K
- 0 malloc / 0 free

So ~10K flops/block, i.e. ~3.4 Mflop/s to keep up with real time. lms-noise is not throughput-bound; the meaningful metric is per-block worst-case latency (real-time headroom), not wall-clock over a long signal.

The two per-tap `affine.if (n - k >= 0)` guards in the LMS loops (the `0 to 32` tap loops in
the `--emit=mlir-affine` dump) are lms-noise's analog of the FIR zero-pad bounds check: they
select the current-block sample `x[n-k]` vs the 31-sample carried history line, firing on
every tap iteration. Splitting the sample loop at n=32 removes them from the steady state (A2),
which in turn unblocks vectorizing the 32-tap inner loop (A1).

There are two independent axes of improvement; do not conflate them. Axis A is backend codegen
quality (matching what C would emit). Axis B is the domain-specific algebraic rewrites the
DSP-MLIR paper measures.

### Measuring

`bench.sh` + `bench.cpp` time the generated kernel in isolation: the driver links the same
`_mlir_ciface_run` as the CoreAudio host but strips the audio machinery. It is decoupled from
how the object is built: change a pass/pipeline, rerun `./bench.sh`, diff the output. Usage:
`./bench.sh [--iterations N] [--warmup N]`.

Each run prints a human summary plus one machine-readable line an agent can parse:
`BENCH_JSON {"min_ms":..,"median_ms":..,"stddev_ms":..,"msample_per_s_median":..,"checksum":..}`.
- Compare `min_ms` (most stable) or `median_ms`; lower is better.
- `checksum` guards correctness. Axis A (pure codegen) must keep it ~bit-stable; Axis B
  (numeric rewrites) should keep it within fp tolerance. A changed checksum on an Axis-A change
  means the optimization broke correctness.

Open harness task: `bench.sh` today still builds the one-shot `osc-low-pass.mlir` throughput
kernel (a 44100-sample FIR regenerate, where the older `min ~3.44 ms/call @ --opt` baselines
came from). Now that `lms-noise.mlir` is the reference, port the driver to loop
`_mlir_ciface_run` over 128-sample blocks under `--stream` and checksum the accumulated output
-- and report **per-block** time / real-time headroom, since the LMS kernel's cost is per
block, not a single regenerate() wall-time.

What `--opt` bought on lms-noise (vs plain): `affine.for` 28->19 (some pointwise chains fused),
`memref.alloca` 46->32 (fewer live stack buffers). Both pipelines are already **0 malloc / 0
free**. `--opt` did NOT vectorize (zero NEON fp ops) and did NOT split the LMS history-boundary
`affine.if` out of the tap loop. So the heap work (A3/A5) is done; the hot-path wins (A1/A2)
remain open. See the per-task status below.

### Axis A -- backend codegen quality

- [ ] A1. Vectorize the LMS 32-tap loops (NEON). Both the filter dot-product and the weight
  update are scalar (fmul/fadd per tap). The inner `0 to 32` loop over the tap weights is the
  vectorization target (the outer sample loop carries the adaptive recurrence and stays
  sequential). Needs A2 first: the per-tap `affine.if` history guard must be gone for a clean
  vectorizable body. Verify NEON (`fmla v*.2d`) appears in the tap-loop disassembly.
- [ ] A2. Hoist the history-boundary branch out of the tap loop. The `affine.if (n-k>=0)`
  selects current-block vs carried-history sample on every one of the 128 x 32 x 2 tap
  iterations. Split the sample loop into a head (n < 32, needs the history line) and a
  steady state (n >= 32, pure current-block, no branch) -- what a C programmer writes by hand.
  This unblocks A1 (a clean steady-state loop vectorizes; a branchy one does not).
- [x] A3. Stop heap-allocating buffers/scalars. DONE: all statically-shaped buffers ≤64 KiB
  stack-promote to `memref.alloca`; the emitted IR has 0 malloc/free in both pipelines, and
  `AssertNoHeapAllocPass` (default-on) enforces it at compile time. `--opt` additionally
  lowers most 0-D scalar constants to SSA (`arith.constant`).
- [~] A4. Fuse the elementwise loops. Affine loop fusion collapsed part of the pointwise chain
  (`affine.for` 28 -> 19 plain->opt; stack buffers 46 -> 32). Not a single pass yet -- several
  128-element elementwise loops (t, sawtooth, tone, delay taps, mix) remain unfused. Confirm
  intermediate buffers disappear from the affine dump.
- [x] A5. Buffer stack promotion. DONE together with A3: no buffer hits malloc; the small
  fixed-size arrays (128-sample block buffers, 32-tap weights, 31-sample history) are all
  `memref.alloca`. Further buffer *reuse*/hoisting to shrink the 32 live allocas is optional
  polish, not correctness.

Status legend: [ ] = open, [~] = partially addressed by `--opt`, [x] = done.
A3/A5 are done (malloc-free, compile-time-guaranteed). A1 and A2 are the remaining hot-path
wins; A2 unblocks A1.

Order to attempt: A2 first (makes the LMS tap loop branch-free), then A1 (vectorize the now-
clean loop -- the real win), then finish A4 (fuse the remaining elementwise chain). Measure
per-block time before/after each.

### Axis B -- domain-specific optimizations

These are legal only because the compiler understands DSP semantics; a generic C compiler
(and often a C programmer) will not do them automatically. They are the DSP-MLIR value
proposition and stack on top of good codegen. (Several were originally framed around the FIR
low-pass demo `osc-low-pass.mlir`; noted below where they do or do not carry over to the LMS
reference kernel, which has no large linear FIR.)

- [ ] B1. Block/parallelize the adaptive-filter recurrence. The LMS update is a feedback scan
  (`w[n]=w[n-1]+mu*e[n]*x[n-i]`), sequential in n today. The escape hatch from the state-model
  section applies: a linear recurrence can be reassociated / cast to a state-space transition
  form (`state_block = f(state, x_block)`) to expose parallelism instead of a naive per-sample
  scan. (This replaces the osc-only "FIR -> FFT convolution" item; lms-noise has no large FIR
  to FFT.)
- [ ] B2. Hoist loop-invariant DSP work. Coefficients/state that only change when a knob moves
  can be lifted and refreshed only on change -- e.g. the one-pole sweep alpha is recomputed per
  block from the LFO phase, but between blocks where `@lfo_period` is unchanged the per-sample
  work is invariant. A DSP-aware invariant-hoisting pass is the general mechanism from the
  paper. (For the FIR demo the same idea hoisted the windowed-sinc tap recompute past a static
  `cutoff`.)
- [ ] B3. Filter/stage fusion and algebraic identities. Cascade/fuse consecutive filters, fold
  constant gains, simplify gain/delay compositions at the dialect level before lowering (e.g.
  the tone's one-pole low-pass and the wet/dry mix). Minor here but the general mechanism
  behind the paper's speedups.

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
