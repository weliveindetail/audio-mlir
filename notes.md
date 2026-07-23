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

## Polyphonic MIDI voices (WIP prototype -- under review)

`lms-noise.mlir` runs a **fixed bank of V=8 MIDI-triggered voices** whose summed output
feeds the swept low-pass -> LMS chain. `@run`'s signature is unchanged
(`run(%out: memref<128xf64>)`), so every existing host/build path still links; the change
is additive globals + one new setter. The macOS host (`lms-noise-macOS.cpp`) opens
CoreMIDI, and the render thread does voice allocation and dispatches events to the kernel.
Verified: silent until note-on, one-pole-smoothed attack, chords sum, release to silence,
sample-accurate frame gating -- all malloc-free (default `AssertNoHeapAllocPass` passes).

**How the open design questions were answered in this prototype:**

- **Arbitrary trigger instant inside a fixed 128-sample block.** Reuse the LFO's
  timestamped-`(value, frame)` idea, one record per voice (`@voice_ev_frame/_gate/_freq`).
  `@run` steps the voice's gate *target* at exactly `ev_frame` and a one-pole smoother
  chases it, so a note that starts/stops mid-block is click-free with **no block
  splitting**. This is the "prefilled vector" reuse -- the per-sample gate is synthesised
  from the sparse event rather than being handed a dense buffer (cf. Mode B breakpoints,
  which stream a dense buffer; both are valid, the sparse-event form matches live MIDI).
- **Fixed vs. dynamic voice count.** Fixed (8), on purpose: a compile-time `0..V` loop
  keeps the block static and malloc-free (the notes.md optimisation properties). Dynamic
  polyphony would need a runtime loop bound and lose compile-time trip counts. Host-side
  voice *stealing* (oldest-first) hides the cap.
- **Voice state.** Each voice carries phase / smoothed-gate / target / freq in a
  `memref<8xf64>` slot -- the `@sample_offset` "implicit state" pattern generalised to a
  bank, persisted across blocks like every other `--stream` global.
- **Where the DSP/host line is drawn.** Voice *allocation* (note->slot, stealing) is
  sequential event bookkeeping and lives in the host; the kernel only renders a fixed bank.

**Design options for the voice bank itself (the sub-topic to discuss).** The prototype
renders the bank as a **hand-written affine loop nest** (voice-outer, sample-inner) summed
into `@voice_mix` and bridged back with `dsp.fromGlobal` -- the same escape hatch the LFO
breakpoint interpolator uses. It is the most direct reuse of validated machinery, but it is
**opaque to tensor-level fusion and the Axis-B rewrites**. Alternatives, ordered by how much
they preserve the optimiser:
  1. *Unrolled tensor lanes* -- write each voice as rank-1 `dsp` tensor ops (getRangeOfVector
     ramp / modulo / mul-gate) and `dsp.add` them. Stays in tensor land (fuses, vectorises),
     but V-fold source blow-up and the per-voice phase/gate recurrence still needs a scan
     (phase is closed-form per block, the smoothed gate is not -> would need a gate op).
  2. *Voices as `dsp.func` + `dsp.generic_call`, inlined.* The toyc pipeline inlines all
     calls into `@run` before lowering (`createInlinerPass`, then deletes the callee), so a
     `@voice(...)` helper called V times is **flattened to one scope** -- source clarity with
     no residual call barrier. After inlining it is identical to (1)/unrolling, so functions
     here are an organisational choice, not a different optimisation regime. (Exported
     `@set_*`/`@run` funcs survive; only internally-called helpers are inlined away.)
  3. *A batched `tensor<VxNxf64>` voice dimension* -- one op over all voices; needs dialect
     work (rank-2 variants of the oscillator/gate ops) but is the cleanest long-term.
  4. *A first-class stateful `dsp.osc`/`dsp.voice` scan op* -- folds the phase, gate AND
     per-voice cutoff-filter recurrences into one op that `StreamStateMaterialization` gives
     per-instance state (the "shared scan primitive" direction in the feedforward/feedback
     section below). See "Per-voice, trigger-anchored cutoff" below for what it must own.

**Per-voice, trigger-anchored cutoff (the requirement that tips this to option 4).** The
original low-pass sat on the *summed* saw (rank-1, one `dsp.lowPassFilter` with a single
`memref<1xf64>` carry). Moving the cutoff *per voice* -- each note filtered by its own
envelope, anchored sample-accurately at that note's trigger frame -- pushes the filter
*before* the sum and reshapes the problem:
  * The one-pole becomes a **rank-2 (batched) scan** `y[v,n] = (1-a[v,n])*y[v,n-1] +
    a[v,n]*x[v,n]`, with **per-voice filter state** (`memref<Vxf64>`, not `memref<1xf64>`)
    that must be **reset on note-on / voice-steal** so the previous note's filter tail does
    not bleed into a stolen voice.
  * The cutoff coefficient `a` stops being a global LFO and becomes a **per-voice envelope
    whose time origin is that voice's note-on** -- structurally the same object as the
    per-voice oscillator phase (a batched, trigger-anchored time base). It is closed-form
    (feedforward), so it needs a batched ramp/envelope plus one more per-voice state slot
    (`@voice_cut_phase`, the samples-since-trigger counter) carried across blocks and reset
    to 0 at the exact event frame.
  * **Sample-tight** anchoring means the envelope and the filter-state reset key off
    `ev_frame`, not block start -- so the gate, the cutoff envelope, and the filter reset are
    all sample-accurate, while the oscillator phase is still block-granular; a fully
    consistent voice would move phase to `ev_frame` too.

`samples/lms-noise.mlir` now prototypes this as **option A** (hand-rolled): the per-voice
one-pole is an extra `iter_arg` in the sample loop, the cutoff envelope another per-voice
phase global (`@voice_cut_phase` / `@voice_lp_state`), and the in-kernel + host LFO are
dropped (the `@lfo_*` globals and `@set_value_lfo_*` setters are retained inert only so the
existing host still links). It validates the behaviour but stays opaque to fusion/Axis-B.

**Missing pieces for option 4 (a first-class stateful `dsp.voice` scan op).** The per-voice
cutoff requirement makes option 4 the natural home, because every per-voice, trigger-anchored,
block-carried quantity is then one op's state:
  1. **ODS.** `def VoiceOp : Dsp_Op<"voice", ...>` taking rank-1 per-voice control vectors
     (`freq`, and the timestamped event triple `ev_frame`/`ev_gate`/`ev_freq` as
     `tensor<Vxi64>`/`tensor<Vxf64>`) plus scalar params (`dt`, gate rate, cutoff span/floor/
     length), producing `tensor<VxNxf64>` (or a pre-summed `tensor<Nxf64>`). `V`/`N` come from
     the operand/result types.
  2. **State.** Four per-voice recurrences -- oscillator phase, smoothed gate, cutoff-envelope
     phase, and the one-pole filter -- so the op carries a `memref<Vx4xf64>` (or four
     `memref<Vxf64>`) state block. `StreamStateMaterialization` already gives each stateful op
     instance its own global; the new work is sizing it `V`-wide and threading it via the op's
     `state` attr the way `dsp.lowPassFilter`/`dsp.lmsFilterResponse` do today.
  3. **Lowering.** A `ConversionPattern` emitting the voice-outer / sample-inner nest that is
     exactly the option-A prototype loop, but reading per-voice params from operands and state
     from the materialized global. The trigger anchor is a compare against the op's own
     `ev_frame` operand -- the op owns the event, so sample-tight cutoff is intrinsic.
  4. **Reduction.** Either fold the V-sum into the lowering (emit `tensor<Nxf64>` directly) or
     pair the `tensor<VxNxf64>` result with a new **reduce-over-axis** op (`dsp.sum` collapses
     *all* elements to a scalar, so it cannot collapse just `V`).
  5. **Tests.** State globals materialized per op, sample-accurate note-on (gate + cutoff both
     key off `ev_frame`), filter-state reset on steal, and continuity of all four recurrences
     across `--stream` blocks.

Option 3 (batched tensors) and option 4 share primitives: option 3 needs a rank-2
`getRangeOfVector` (per-row `first`/`step`) and a rank-2 `lowPassFilter`; option 4 folds both,
plus the gate and trigger logic, into one op. The reduce-over-axis op is needed by both.

**Prototype limitations (v1).** One pending event per voice per block (the host allocator
must not target a voice twice in one block); `@set_note_event` must be called only from the
render thread (same no-race discipline as the LFO breakpoint setter); and with the ring
buffer between render and playback, live events mostly land at `frame 0` -- the per-sample
gate ramp is in place and correct, but true sample-accurate live scheduling needs tighter
audio-clock coupling than the current ring gives.

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

## What ops to add (the composable primitive set)

The demo now exercises the full concept chain the readme enumerates -- sample-accurate MIDI
events -> polyphonic sawtooth bank -> per-voice gate envelope -> per-voice swept low-pass
(cutoff from a shared wavetable LFO at each voice's trigger-anchored phase) -> voice mixer ->
runtime-selectable colored noise through a delay-line acoustic path -> tone+noise mixer ->
32-tap LMS -> wet/dry mix. Today most of that (the voice bank, its gate, its per-voice
cutoff filter) is a **hand-rolled affine loop nest** bridged back to tensor land with
`dsp.fromGlobal` (`lms-noise.mlir`), which is opaque to fusion, vectorization and the Axis-B
rewrites. The plan is to express the whole chain as a **small set of composable tensor/scan
primitives** so it stays in tensor land end-to-end.

This is deliberately **option 3** from the voice-bank discussion above (shared batched
primitives), **not option 4** (a monolithic `dsp.osc`/`dsp.voice` scan op). Folding the four
per-voice recurrences into one op is the opposite of small/reusable, and the primitives below
are needed regardless -- biquads, smoothers, LFOs, mixers and the noise/LMS scans all reuse
them. If the `dsp.voice` sugar is ever wanted, it becomes a fusion/macro that lowers to these,
not a separate optimization regime.

The set splits cleanly along the feedforward/feedback line from the previous section: four
feedforward tensor maps (ramp, wavetable, reduce, event-expansion) plus **one** feedback scan
primitive that every recurrence in the toolkit shares.

Listed in implementation order (all **unaddressed**):

- [x] 1. **`dsp.reduce` -- sum (or other reduction) over ONE axis.** Collapses
  `tensor<VxNxf64> -> tensor<Nxf64>` (V = voices), the batched-voice mixer. Smallest,
  stateless, self-contained, and unblocks every batched result below, so it goes first.
  `dsp.sum` collapses *all* elements to a scalar and cannot collapse just V; this is the
  missing per-axis form. Feedforward. **Replaces** the `@voice_mix` store + `dsp.fromGlobal`
  bridge -- the summed tone becomes an ordinary tensor that fuses with the downstream
  `d = tone + n0`.
  **Done.** Op added (`Ops.td` `ReduceOp`, `axis` attr default 0, rank-2 -> rank-1),
  shape-inference + verifier in `Dialect.cpp`, `ReduceOpLowering` (nested affine loop with
  an inner iter_arg accumulator) in `LowerToAffineLoops.cpp`. The kernel's `@voice_mix` is
  now `memref<8x128>` (one row per voice, no in-loop cross-voice add, no pre-clear); the
  bank is lifted with `dsp.fromGlobal` to `tensor<8x128>` and summed with
  `dsp.reduce {axis=0}`. `lms-noise-check.sh` passes 13/13 with a **bit-identical checksum**
  (`-2.749372436647139e+03`) under both `--opt` and `OPT=0`, i.e. a pure structural refactor.
- [x] 2. **Rank-2 `dsp.getRangeOfVector` -- batched ramp (per-row `first`/`step`).** Produces
  `tensor<VxNxf64>` from a per-voice start and increment: the per-voice oscillator phase AND
  the cutoff-envelope phase are both this. Feedforward, closed-form in `n`; the only carried
  state is one scalar phase per row (the `@sample_offset` "implicit state" pattern generalised
  to a bank -- the feedforward scalar-state case), advanced across `--stream` blocks. Testable
  standalone before any scan exists.
  **Op done; kernel integration deferred to op 4.** The op now accepts rank-1 `first`/`step`
  (`tensor<Vxf64>`) and emits a `tensor<VxNxf64>` batched ramp `y[v,n] = first[v] + n*step[v]`,
  one independent arange per row; rank-0 `first` still gives the original rank-1 behaviour.
  `inferShapes` prepends `V` when `first` is ranked >=1 (`Dialect.cpp`); the lowering branches
  on `rank==2` for an outer per-row loop that loads `first[v]`/`step[v]` and runs the inner
  1..N iter_arg recurrence (`LowerToAffineLoops.cpp`). Verified standalone via `--emit=jit`:
  `first=[0,10], N=4, step=[1,2] => [[0 1 2 3][10 12 14 16]]`; rank-1 path unregressed
  (`getRangeOfVector(0,5,2) => [0 2 4 6 8]`). **Not yet wired into `lms-noise.mlir`:** the only
  candidate consumer is the sawtooth phase, but the kernel wraps phase with a per-sample
  single-subtract (`if phase>=1: phase-=1`), valid only because `inc<1`. A closed-form ramp
  `first+n*inc` grows unbounded over the block and needs a batched `frac`/`floor` (multi-wrap)
  to become a sawtooth -- `dsp.modulo` has no scalar-broadcast form, so that shaping op is
  out of scope here. Forcing an interim integration means a two-pass split with a saw `8x128`
  round-trip plus duplicated event-decode -- a pessimization with no structural cleanup. The
  batched ramp lands cleanly once op 4 (`dsp.scan`) replaces the fused per-sample voice loop;
  ops 2/3/4 integrate into the voice bank together. No `02-` benchmark taken (kernel unchanged).
- [x] 3. **`dsp.wavetable` / `dsp.lookup` -- batched table read.** Gathers a shared table
  (`tensor<Lxf64>`) at a batched, optionally fractional/wrapping index (`tensor<VxNxf64>`),
  with linear interpolation and index wrap. This is the generalised LFO / wavetable modulation
  source -- the `@voice_cut_shape` read at each voice's phase -- reusable for any mod source.
  Feedforward (a gather). **Replaces** the `memref.load %vcsM[%cpIdx]` per-voice table lookup.
  **Op done; kernel integration deferred to the voice-bank rewrite.** `dsp.wavetable(table,
  index)` gathers a rank-1 `table` (`tensor<Lxf64>`) at a fractional `index`: rank-1
  `tensor<Nxf64>` gives one row, rank-2 `tensor<VxNxf64>` gathers one row per voice. Each
  element is `pw = index mod L` (positive modulo via `p - floor(p/L)*L`, so negative phases
  wrap too), `i0=floor(pw)`, `frac=pw-i0`, `i1=(i0+1) mod L`, and
  `y = table[i0]*(1-frac) + table[i1]*frac` -- linear interpolation with table wrap-around, a
  strict superset of the kernel's current truncate-and-single-subtract lookup. Wired
  end-to-end: `Ops.td` `WavetableOp`, `Dialect.cpp` (build/inferShapes=index shape/verify:
  table rank-1, index rank 1-2), `WavetableOpLowering` in `LowerToAffineLoops.cpp` (affine
  iteration over the index, data-dependent `memref.load` for the two table taps), and a
  `wavetable(table,index)` frontend builtin in `MLIRGen.cpp`. Stateless -- no
  `StreamStateMaterialization` entry. Verified standalone via `--emit=jit` with table
  `[10,20,30,40]`: rank-1 `idx=[0,0.5,1,2.5,3.5,4] => 10 15 20 35 25 10` (the 4.0 wraps to
  table[0]); rank-2 batched matches; negative wrap `idx=[-0.5,-4] => 25 10`; compiles clean
  under `--stream --opt`. **Not yet wired into `lms-noise.mlir`:** same fused-loop blocker as
  ops 2/4 -- the cutoff lookup lives inside the per-sample voice loop alongside four
  recurrences, so replacing just `memref.load %vcsM[%cpIdx]` needs the loop split and an
  `8x128` cutoff-phase intermediate materialised (that phase in turn is the batched
  `getRangeOfVector` ramp from op 2, gated by op 5's trigger reset). Lands wholesale in the
  voice-bank rewrite. No `03-` benchmark taken (kernel unchanged); `lms-noise-check.sh` stays
  13/13, checksum `-2.749372436647139e+03`.
- [x] 4. **`dsp.scan` -- ONE batched linear-recurrence op (the shared feedback primitive).**
  Expresses `y[.,n] = a[.,n]*y[.,n-1] + b[.,n]*x[.,n]` with per-row carried state
  (`memref<Vxkf64>`), generalising to biquad / state-space (k-th order). Coefficients may be
  per-sample tensors so the cutoff can sweep. **This single op covers the per-voice one-pole
  low-pass, the click-free gate smoother (also a one-pole), and -- in matrix form -- the
  colored-noise filters and any IIR.** State flows through `StreamStateMaterialization` (a
  module-scope global per op instance), sized `V`-wide, exactly like `dsp.lowPassFilter` /
  `dsp.lmsFilterResponse` today. Needs an optional **per-sample reset input** (a gate/trigger
  signal) so a voice's filter state clears at its note-on frame instead of inheriting a stolen
  voice's tail -- the one wrinkle the batched form must own; that reset signal is produced by
  op 5.
  **Op done (first-order, k=1); kernel integration + reset input deferred.** `dsp.scan(a,b,x)`
  takes three same-shaped operands and runs `y[n] = a[n]*y[n-1] + b[n]*x[n]`: rank-1
  `tensor<Nxf64>` is a single recurrence, rank-2 `tensor<VxNxf64>` runs V independent rows.
  `y[.,-1]` is carried across `--stream` blocks from a zero-init state global
  (`__stream_state_scan_*`, `memref<Vxf64>`) materialised in `StreamStateMaterializationPass`
  and read/written exactly like the one-pole; the inner affine loop carries `y[n-1]` in an
  iter_arg so `n==0` consumes the seed. Wired end-to-end: `Ops.td` `ScanOp`, `Dialect.cpp`
  (build/inferShapes/verify -- operands must share shape, rank 1 or 2), `ScanOpLowering` in
  `LowerToAffineLoops.cpp`, and a `scan(a,b,x)` frontend builtin in `MLIRGen.cpp`. Verified
  standalone via `--emit=jit`: rank-1 one-pole (`a=b=0.5, x=1s => 0.5 0.75 0.875 0.9375`) and
  rank-2 batched (`row0` one-pole, `row1` a=0 => `b*x`); compiles clean under `--stream --opt`
  (no heap). **k>=2 (biquad/state-space) and the per-sample reset input are NOT implemented** --
  the reset input has no producer until op 5 (`dsp.eventToSignal`), so adding it now would be a
  half-finished operand with no consumer. **Not yet wired into `lms-noise.mlir`:** same
  structural blocker as op 2 -- the four voice recurrences (gate one-pole, saw ramp, cutoff
  ramp+reset, low-pass one-pole+reset) live in a single fused per-sample loop; pulling one
  recurrence into a scan means splitting that loop, materialising several `8x128` intermediates
  and duplicating the event decode (a pessimization), and the two reset-bearing recurrences
  additionally need op 5's trigger signal. The batched voice bank lands wholesale once ops 3
  (`wavetable`) and 5 (`eventToSignal`) exist -- plus a batched saw-shaping (`frac`) step -- so
  ramp/wavetable/scan/eventToSignal replace the fused loop together. No `03-` benchmark taken
  (kernel unchanged); `lms-noise-check.sh` stays 13/13, checksum `-2.749372436647139e+03`.
- [x] 5. **`dsp.eventToSignal` -- expand sparse timestamped events into a dense control
  tensor.** Takes the per-row `(value, frame)` event record (`tensor<Vxf64>` value +
  `tensor<Vxi64>` frame, one pending per row per block -- the shape the kernel already
  prototypes) plus a previous-value carry, and emits a dense `tensor<VxNxf64>` step/ramp
  control signal, together with the per-sample **trigger/reset** signal `dsp.scan` consumes.
  This is the notes.md "compute a tensor per parameter over its value-over-time" direction: it
  turns the gate target, the cutoff automation and (generalised) any knob like `@wet` into
  feedforward tensor math that is **sample-accurate without block splitting**. Feedforward (a
  scatter + hold/interpolate). Built last because it ties the MIDI/automation front end into
  the scan/ramp pipeline the first four ops establish.
  **Ops done (split into two single-result ops); kernel integration deferred to the voice-bank
  rewrite.** Because the toy frontend ops are single-result (same reason FFT is split into
  real/img ops), this landed as a pair: `dsp.eventToSignal(value, frame, prev, N)` emits the
  step/hold control `out[.,n] = (n < frame) ? prev : value`, and `dsp.eventToTrigger(frame, N)`
  emits the reset pulse `trig[.,n] = (n == frame) ? 1 : 0`. Both take scalar events (-> rank-1
  `tensor<Nxf64>`) or rank-1 per-row events `tensor<Vxf64>` (-> rank-2 `tensor<VxNxf64>`); a
  sentinel `frame >= N` leaves a row all-`prev` / all-zero (the "no event this block" case,
  exactly the kernel's `%hasEv = efrI < 128` guard). `frame` is carried as an f64 whole number
  and `N` is the scalar constant length (the `getRangeOfVector` convention), so both are
  frontend-testable. Whether an event is a note-on (fire the filter reset) stays a consumer-side
  AND with a `value > 0.5` mask, matching the kernel's `%isOn`. Wired end-to-end: `Ops.td`
  (`EventToSignalOp`, `EventToTriggerOp`), `Dialect.cpp` (build/inferShapes via shared
  `inferEventShape` -> `[V,N]` or `[N]` / verify: operands share shape, scalar-or-rank-1),
  `EventToSignalOpLowering` + `EventToTriggerOpLowering` in `LowerToAffineLoops.cpp`, and both
  `eventToSignal`/`eventToTrigger` frontend builtins in `MLIRGen.cpp`. Stateless -- no
  `StreamStateMaterialization` entry (the `prev` carry is passed in explicitly; under a full
  rewrite it would come from the persisted gate/cutoff-target globals). Verified standalone via
  `--emit=jit`: scalar `val=1,frame=3,prev=0,N=6 => sig 0 0 0 1 1 1 / trg 0 0 0 1 0 0`; batched
  `val=[1,9],frame=[2,99],prev=[0,5],N=4 => sig [[0 0 1 1][5 5 5 5]] / trg [[0 0 1 0][0 0 0 0]]`
  (row1's out-of-range frame 99 -> all-prev, no trigger); compiles clean under `--stream --opt`.
  **Not yet wired into `lms-noise.mlir`:** produces exactly the two signals (`%tgt` step/hold and
  `%atTrg` reset) the fused voice loop computes inline today, so this is the last missing piece
  for the wholesale rewrite -- see the integration note below. No `04-` benchmark taken (kernel
  unchanged); `lms-noise-check.sh` stays 13/13, checksum `-2.749372436647139e+03`.

Ops 1 and 4 (`dsp.reduce`, `dsp.scan`) are the load-bearing ones; 2, 3, 5 are thin. Rank-2
`getRangeOfVector`, `dsp.reduce` and `dsp.scan` are exactly the option-3 primitives called out
above; `dsp.wavetable` and `dsp.eventToSignal` complete the chain so the *whole* demo -- not
just the voice waveform -- lives in tensor land.

**Status: all five primitives implemented AND the voice bank has been rebuilt wholesale in
tensor land** (`lms-noise.mlir`, `bench-out/02-all-applied.log`). The hand-rolled fused per-
sample voice loop is gone; the batched rank-2 `tensor<8x128>` pipeline is:

    // per block, all batched over the 8 voices (rank-2 tensor<8x128>):
    tgt   = eventToSignal(ev_gate, vb_efrF, vb_tgt0, 128)          // gate target step/hold
    trg   = eventToTrigger(vb_trigfr, 128)   // frame sentinel-masked to note-ons only
    gate  = scan(0.99, 0.01, tgt)                                  // one-pole gate smoother
    phase = getRangeOfVector(vb_ph_first, 128, vb_inc)             // per-voice saw ramp
    saw   = 2*wrap(phase, 1.0) - 1                                 // branchless frac sawtooth
    cphase= scan(1-trg, 1-trg, stepB)                              // cutoff LFO phase (reset@trg)
    alpha = wavetable(voice_cut_shape, cphase)                     // per-voice cutoff (wraps modL)
    lp    = scan((1-alpha)*(1-trg), alpha, saw)                    // per-voice low-pass (reset@trg)
    tone  = gain(reduce(lp * gate, axis=0), 0.2)                   // mix 8 voices -> tensor<128>

The two prerequisites flagged earlier were resolved without new op surface: (a) **reset via
coefficient masking** (Q1) -- forcing the scan feedback coeff `a[.,f]=0` at the trigger frame
drops the carried state, so neither reset-bearing scan needs a per-sample reset operand; the
cutoff phase resets to 0 with `a=b=(1-trg)` and the low-pass masks its `a` with `(1-trg)`; and
(b) **`dsp.wrap`** (Q2) -- a branchless floor-subtract (`x - floor(x/period)*period`, HW
`frintm`, vectorizes) replaces the magnitude-unsafe single-subtract saw wrap, and the cutoff
`wavetable` wraps mod-L=8000 internally so the unwrapped growing scan phase indexes correctly.
The saw phase resets **block-level** (its pre-trigger portion is gated to ~0), so it stays a
plain per-voice ramp; only the cutoff phase and low-pass state reset sample-accurately.

A small 8-iteration block-level DECODE loop still resolves each voice's pending MIDI event into
the flat control vectors (`@vb_*` scratch globals) the batched ops consume, and commits the
carried per-voice state (freq, saw phase, gate target, cutoff speed; consume the event). The
three scans each carry their own `--stream` state global (gate value, cutoff phase, low-pass
state), replacing the old `@voice_gate` / `@voice_cut_phase` / `@voice_lp_state` manual carries
(those globals are now vestigial). `lms-noise-check.sh` stays **13/13**; the golden checksum
moves to `-2.749372436647042e+03` (was `...139e+03`) -- a ~1e-10 relative shift, expected from
(i) `wavetable`'s linear interpolation vs the old truncated table load, (ii) the cutoff scan's
unwrapped-vs-wrapped state carry (identical mod-L index, differs only in float rounding), and
(iii) fusion reassociation. This is a genuine numerical change, so the bit-identity checksum no
longer holds; the 13 threshold checks are authoritative.

**Toolchain gap surfaced + fixed.** The affine loop-fusion pass (`--opt`) materializes tiny
private scratch memrefs for fused producer/consumer slices; for the rank-1 chains
`AffineScalarReplacement` forwards them away, but the rank-2 batched chains leave ~8
`memref<1x1xf64>` **heap** allocs that `AssertNoHeapAlloc` (rightly) rejects. Fixed in
`toyc.cpp` by running `bufferization::PromoteBuffersToStack` (64 KiB cap, rank<=4,
non-escaping) just before the no-heap check -- exactly the remedy that pass's own error message
prescribes; it needs `memref::registerAllocationOpInterfaceExternalModels(registry)`.

**Result (`bench-out/02-all-applied.log`, vs `00-baseline.log`): a ~25-28% latency REGRESSION**
(full_white_poly8 median 0.01128 ms vs 0.00885 ms; regression is uniform across all four
configs, including idle `rest_silent` 0.01104 vs 0.00859 ms). Static footprint is flat (no heap;
`__data` 65608 vs 65400 B). *Why it regressed here:* the batched form's Axis-A1 win (the V=8
voice axis is embarrassingly parallel) needs an actual **vectorizer** to pay off, and
`createAffineVectorizePass` is commented out of the `--opt` pipeline, so the V axis lowers to
scalar loops. Without SIMD, the batched pipeline just pays extra buffer-materialization/traffic
for its ~14 rank-2 intermediates and the `wavetable` lerp (2 loads + interp vs 1 truncated load),
which the old single tight voice-outer/sample-inner nest (everything in scalar iter_args, zero
intermediate buffers) avoided. Still **~257x real-time** (0.39% of the 2.9025 ms block), so the
regression is far from the budget. The rewrite delivers the *structural* goals -- tensor-land
end-to-end, composable/reusable primitives, one `dsp.scan` feedback primitive, the `@voice_mix`+
`fromGlobal` escape hatch dropped -- and sets up the real speedup lever: **enable the affine
vectorizer** (or a hand `--emit` NEON path) so the V axis becomes actual SIMD; that is the next
data point to capture (`bench-out/03-vectorized.log` or similar).

### Why we choose these

- **Small and composable over monolithic.** Each op is independently specifiable, testable and
  reusable across the toolkit (biquads, smoothers, LFOs, mixers, the noise/LMS scans), whereas
  a single `dsp.voice` folding four recurrences is a one-off. The `dsp.voice` sugar, if ever
  wanted, lowers to these -- an organisational choice, not a different optimiser regime.
- **Stay in tensor land / drop the escape hatch.** `dsp.reduce` removes the `@voice_mix` +
  `dsp.fromGlobal` bridge, so the bank is tensor-typed end-to-end and fuses with the downstream
  add/LMS/mix instead of being pinned apart by a hand-written loop.
- **Vectorization (Axis A1).** A batched `tensor<VxNxf64>` voice dimension makes the V axis
  embarrassingly parallel -- the oscillator/gate/lookup maps vectorize across voices (NEON),
  which the current voice-outer/sample-inner nest is opaque to.
- **One recurrence to optimize (Axis B1).** A single `dsp.scan` is the *one* place to apply the
  state-space blocking / parallel-scan escape hatch, instead of re-deriving it for each bespoke
  loop (per-voice low-pass, gate, pink/brown/ou, LMS). Consolidating the feedback class is the
  leverage.
- **Automation as tensors (`dsp.eventToSignal`).** Expanding `(value, frame)` events into a
  dense control tensor makes control-rate params sample-accurate while keeping them feedforward
  -- no block splitting, no grown `@run` ABI -- and generalises the timestamped-setter pattern
  the kernel already prototypes to every knob.
- **Clean feedforward/feedback split.** Ramp, wavetable, reduce and event-expansion are
  feedforward tensor maps; `dsp.scan` is the single feedback primitive. That is exactly the
  state model that scales from the previous section, now made concrete as an op set.

## Target-portable `dsp.scan`: the associative / parallel-scan form

`dsp.scan` is a first-order linear recurrence

```
y[n] = a[n]*y[n-1] + b[n]*x[n]      (state carried across blocks in y[-1])
```

The current lowering (`ScanOpLowering`, LowerToAffineLoops.cpp) emits the obvious
sequential inner loop over N. That is optimal for narrow SIMD and VLIW DSP cores
(2--8 lanes; the sequential dependency is not the bottleneck when the voice axis V
already fills the vector unit). It is the *wrong* shape for any target wider than
the voice count -- AVX-512 with V<=8, GPUs, TPUs, wide HVX -- because there the N
axis is the serialization and it must be broken.

### Why it parallelizes

Fold the input into the running state: let `c[n] = b[n]*x[n]`. Then each timestep is
an **affine map** on the carried scalar `y`:

```
f_n(y) = a[n]*y + c[n]
```

Composition of two such maps is again affine and is **associative**:

```
(f_hi . f_lo)(y) = a_hi*(a_lo*y + c_lo) + c_hi
                 = (a_hi*a_lo)*y + (a_hi*c_lo + c_hi)
```

So represent each step by the pair `(A, C) = (a[n], c[n])` and define the combine

```
(A_hi, C_hi) (+) (A_lo, C_lo) = (A_hi*A_lo,  A_hi*C_lo + C_hi)   identity (1, 0)
```

This is an associative monoid, so an **inclusive prefix scan** over the N pairs
yields the cumulative `(A_0..n, C_0..n)`, and the output is recovered in one map:

```
y[n] = A_0..n * y[-1] + C_0..n         (y[-1] = carried block state)
```

The scan itself costs `log2(N)` parallel steps (N=128 -> 7 steps) instead of N
sequential ones. Two standard schedules:
- **Hillis-Steele** (inclusive, `N*log N` work, `log N` depth, no down-sweep):
  easiest to emit in-register / in-lane for wide SIMD.
- **Blelloch** (work-efficient, `2N` work, up- + down-sweep): the right choice for
  GPU/TPU where total work, not depth, is the cost.

### Optimization opportunities this unlocks

- **One rewrite, every feedback op.** Per the "one recurrence to optimize (Axis B1)"
  note above, every stateful block in the kernel (per-voice low-pass, envelope gate,
  pink/brown/OU noise, and structurally the LMS adaptive recurrence) is an instance
  of this scan. Implement the associative form once and all of them retarget.
- **Wide SIMD (AVX-512).** With V<=8 filling one ZMM on the voice axis, an additional
  in-lane Hillis-Steele over N lets a wide machine process a *time* block in `log N`
  vector steps rather than 128 scalar ones -- the escape from N-boundedness.
- **GPU / TPU.** The Blelloch scan maps the N axis onto many lanes; the reduce and
  wavetable gather (warp shuffle / texture fetch) already fit. This is the *only*
  route to occupancy on those targets given V=8 is too small alone.
- **Time-varying coefficients are free.** `a[n]` and `c[n]` are per-sample already,
  so the affine-pair formulation needs no constant-coefficient assumption -- the
  swept one-pole and event-driven gate keep working unchanged.

### Implementation sketch

Key the lowering on a target width (pass option or a `dsp.scan` target attribute);
`ScanOpLowering` picks a strategy:

1. **`sequential` (default, narrow SIMD / DSP cores).** Today's inner N loop.
2. **`associative` (wide SIMD / GPU / TPU).** Lower to:
   - a feedforward map computing `C[n] = b[n]*x[n]` (parallel over N and V),
   - a prefix scan over pairs `(a[n], C[n])` with the combine above -- Hillis-Steele
     emitted as `log2(N)` shifted vector passes for SIMD, or delegated to a
     `scf.parallel` / GPU-dialect Blelloch for accelerators,
   - a final `y[n] = A_cum[n]*y[-1] + C_cum[n]` map (parallel), where `y[-1]` is the
     module-scope stream-state global (unchanged plumbing).
   The carried-state global and the reset-as-coefficient-masking trick both survive:
   forcing `a[f]=0` at a reset frame zeroes `A_cum` from that frame on, exactly as in
   the sequential form.

Correctness note: the associative form reassociates the sums, so the checksum moves
within fp tolerance (an Axis-B numeric rewrite, per the checksum guard below), not
bit-stable. Guard it with the fp-tolerance band, not bit-equality.

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

## Target encodings for the voice/scan work

Notes on how the batched voice pipeline (`tensor<8x128xf64>`, V=8 parallel voices,
N=128 sequential scans) maps onto targets beyond the M1 Pro (NEON, 2-wide f64). 

### Newer Apple arm64 (M4 / A17+): SME / SME2

M2/M3 are still plain NEON (128-bit, `vector<2xf64>`) -- no change from M1. The jump
is **M4 / A17 Pro and later**, which add **SVE2** and **SME/SME2** (Scalable Matrix
Extension). SME is *tile*-oriented, not lane-oriented, so it does not accelerate the
sequential scan directly; its win is the **`dsp.reduce` over V** and the voice
mix-down, which are outer-product / matmul-shaped.

- **Encoding surface.** Streaming SVE mode entered/exited with `SMSTART` / `SMSTOP`
  (the `PSTATE.SM` and `PSTATE.ZA` bits); the accumulator lives in the **ZA tile**
  register. FP64 outer-product accumulate is `FMOPA` (f64 variant, gated on the
  `FEAT_SME_F64F64` feature); tile row/col moves are `MOVA`; predication uses the SVE
  governing predicate registers `p0`--`p15`. Streaming vector length `SVL` is
  implementation-defined (Apple exposes a fixed SVL; query via `RDSVL`).
- **MLIR path.** The ArmSME dialect lowers to these; route the reduce/mix through
  Linalg -> Vector -> ArmSME rather than the affine scalar path. This is the same
  machinery Accelerate/AMX already use under the hood.
- **Caveat.** Keep the scan on NEON `vector<2xf64>`; only the reduction and any
  FIR/convolution-shaped tap loop (the LMS 32-tap) are worth moving to SME.

### Hexagon (Qualcomm HVX): fixed-point reformulation

HVX is a **1024-bit** vector unit but **integer / fixed-point** -- its habitat is
int8/int16/int32 audio, not f64. To exploit it the kernel must move to **Q-format
fixed point** (Q15 for coefficients/samples, Q31 for accumulators), which is what
production mobile audio does anyway.

- **Encoding surface.** VLIW packets of up to 4 instructions (`{ ... }`). Vector
  MAC/multiply: `V6_vmpyhv` / `V6_vmpyiwh` and the `vrmpy` reduce-multiply family
  (dot products -> maps onto both the FIR taps and the `dsp.reduce`); saturating
  fixed-point ops carry the Q-format. State/history and the wavetable mod-L wrap map
  onto **hardware circular addressing** -- the `M0/M1` modifier registers plus the
  `CS0/CS1` circular-start registers, addressed with `.circ` post-increment modes --
  which is a near-exact match for `dsp.wrap` and the delay lines. Zero-overhead
  hardware loops via `loop0/loop1` + `endloop`.
- **MLIR path.** No upstream Hexagon dialect; lower the fixed-point form to LLVM IR
  and rely on the Hexagon LLVM backend + HVX intrinsics, or emit intrinsics directly.
  The scan's `a[n]*y[n-1]` becomes a Q31 saturating MAC.
- **Cost.** Requires a fixed-point pass (scale analysis / Q-format assignment) ahead
  of lowering -- a real reformulation, high payoff, but changes the checksum model
  entirely (fixed-point error, not fp tolerance).

### Dedicated DSP cores (SHARC / TI C6000 / Tensilica HiFi)

The kernel's native habitat: designed for exactly this class of IIR/FIR + wavetable
audio, favoring f32 (SHARC also 40-bit and native f64) with VLIW ILP across the 8
independent voice lanes even without wide SIMD.

- **ADI SHARC.** Single-cycle MAC; **zero-overhead loops** (`LCNTR = N, DO end UNTIL
  LCE`); **circular buffers** via the DAG address-generator register files
  (`I`ndex / `M`odify / `L`ength / `B`ase) -- set `L` non-zero and the pointer wraps
  in hardware, matching `dsp.wrap`, the delay lines, and the wavetable read with zero
  branch overhead. Native 32/40-bit float keeps the scan in floating point (no
  fixed-point detour).
- **TI C6000.** 8-way VLIW (two `.M` multiply units etc.); software-pipelined loops
  the compiler schedules across the parallel voices; `SPLOOP` buffer for tight
  kernels. f32 throughout.
- **Cadence Tensilica HiFi.** Configurable SIMD MACs (2/4-way) tuned for audio; codec
  vendors' default target. Intrinsics-driven, no MLIR dialect.
- **MLIR path.** None upstream for these; the realistic route is emit portable
  LLVM IR / C from the affine or scalar form and hand it to the vendor toolchain,
  keeping the `sequential` scan lowering (their zero-overhead loops + circular
  addressing make the sequential recurrence cheap -- the associative rewrite is not
  needed and would only add work).
