# Research Notes

## Uncertainty algebra (a probabilistic MLIR type -- opportunities & challenges)

Motivated by the `noise-lang` DSL (in `../noise-lang`), where every value is a *random
variable* (a probability distribution), operators lift over random variables, and `P(cond)` /
`E(x)` estimate probabilities and expectations by Monte Carlo. The question: could we bring that
"uncertainty algebra" into DSP-MLIR as a first-class type or dialect? This section records the
design of that ambitious track ("B") so we don't lose the reasoning. The concrete, shippable
piece ("A" -- noise *signal* generators) is separate and is implemented in the `dsp` dialect;
see `dsp.noise_white/pink/brown/ou` and `noise-kinds.mlir`.

### The idea

A new type, e.g. `!dsp.rv<f64>`, denoting a random variable rather than a value. Arithmetic ops
lift over it (`add(rv, rv) -> rv`), distribution ops introduce it (`dsp.sample_normal %mu, %sigma
: !dsp.rv<f64>`), and *query* ops collapse it back to a number by simulation:
`dsp.prob %cond -> f64`, `dsp.expect %x -> f64`, `dsp.variance %x -> f64`. A query lowers to a
Monte-Carlo loop: draw N samples, evaluate the random variable's def-use subgraph with fresh
draws at the sample sites, and reduce (mean / fraction-true / variance).

### Opportunities (why MLIR is a surprisingly good host)

- **SSA *is* noise-lang's sharing rule.** noise-lang's load-bearing rule is "one name = one
  fixed draw; every mention reuses it" (`X - X == 0`, `X + X == 2X`). In MLIR an SSA value is
  exactly one node shared by all its uses. So the def-use graph already encodes the sharing
  semantics for free; independence = two distinct `sample` ops (mirrors noise-lang's two `~`
  bindings), and `~[n]` maps to a vector/shaped result.
- **The query is a clean lowering, not a new runtime.** `dsp.prob`/`dsp.expect` lower to an
  ordinary affine/scf loop over the backward slice of the queried value, RNG at the sample ops,
  reduction at the end -- the same machinery the dialect already uses. It JITs/AOTs like any
  other kernel; no interpreter.
- **Distributions compose with the existing DSP ops.** Because the RV lifts through arithmetic,
  you can push a random noise source through a filter and query the *output* statistics -- e.g.
  `E[residual power]` after the LMS canceller, or `P(SNR > threshold)` -- reusing `dsp.filter`,
  `dsp.fft`, etc. unchanged.
- **Analysis passes have real semantics to exploit.** Constant sub-expressions fold eagerly
  (a point mass), independent-sum variance is additive, affine transforms of a Gaussian stay
  Gaussian -- a dialect could rewrite some queries to closed form instead of sampling.

### Challenges (and the honest scope limit)

- **Sequential/stateful processes are out of scope -- which is exactly what LMS is.** noise-lang
  is explicit: it samples independent lanes that *cannot carry state across a time index* (no
  random walks, Markov chains, queues). The adaptive filter in `lms-noise.mlir` is a per-sample
  weight recurrence `w[n] = w[n-1] + mu*e[n]*x[n-i]` -- precisely a stateful time recurrence. So
  the RV algebra **cannot express the canceller's core**; it can only *wrap* it (run the
  deterministic kernel over an RV noise source across many seeds and estimate output stats).
  This is the key reason A and B are different layers: A generates the samples, B characterizes
  the pipeline built on them.
- **Type-system surface.** `!dsp.rv<T>` needs lifting rules for every arithmetic/comparison op,
  a bool-RV for conditions, and a conditioning form (`X | C`). That is a lot of ODS + verifier
  work orthogonal to the current deterministic tensor DSP.
- **Cost model & seeding.** Each query is an N-sample pass (default 1e6 in noise-lang). Nested
  queries multiply. Reproducibility needs a threaded, well-defined seed per sample site. Shared
  vs. independent draws must be tracked precisely or the estimates are silently wrong.
- **Cross-query consistency isn't free.** As in noise-lang, `P(A)`, `P(B)`, `P(A&&B)` estimated
  in separate passes need not satisfy `P(A&&B) <= P(A)`; a serious implementation shares one
  sampling pass per query and rejection-conditions within it.
- **No scaling inference.** Forward Monte Carlo + rejection conditioning only; importance
  sampling / MCMC (needed to condition on continuous data) is a further track, same as
  noise-lang's own roadmap.

### Verdict

B is a legitimate and elegant MLIR dialect (SSA fits the sharing model beautifully), but it
*characterizes* stochastic pipelines rather than *generating* them, and it structurally can't
model the LMS recurrence. So it is deferred. What actually helps `lms-noise.mlir` today is A:
first-class colored-noise **signal generators** in the `dsp` dialect, which is what the rest of
this work implements.

## Batch-and-stream C++ implementation

- f64→f32 on copy: AudioCallback reads double samples from the published batch and casts to
float per-sample into the CoreAudio buffers.
- Batch-and-stream: regenerate() renders a full 1-second batch into the back buffer via the
kernel, then atomically swaps it in. The audio thread streams from gActive with
wraparound; the keyboard thread triggers a regenerate on each arrow press.
- Thread safety: single std::atomic<double*> pointer swap (acq/rel) publishes a
fully-rendered buffer; no torn reads.

## Moving to block-based processing later

The current design recomputes a whole second of audio on every cutoff change and loops it.
Block-based means the kernel produces exactly inNumberFrames of fresh audio per audio
callback, with continuity across calls. That requires carrying state the current run(out,
fc) kernel doesn't have:

1. Oscillator phase continuity. Right now getRangeOfVector(0, 44100, dt) always starts at
t=0. Across blocks the sine must not restart, or you get a click every block. The kernel
needs a phase (or sample-index) input/output: take phase_in, generate n samples starting
there, and write back phase_out. In the DSL that's a start offset added to the time vector;
at the ABI level it's an extra double* in/out parameter (or a returned scalar).

2. FIR filter state (the hard part). A 101-tap FIR convolution needs the previous N−1 = 100
input samples to compute the first outputs of the new block — otherwise each block's
leading 100 samples are wrong and you hear edge artifacts at every boundary. The standard
fix is overlap-save: keep a 100-sample history of the oscillator signal, prepend it to the
new block, convolve, and discard the transient. So the kernel needs a second caller-owned
state buffer (hist[100]) that it reads at entry and updates at exit.

3. Cutoff changes mid-stream. Recomputing the windowed-sinc coefficients every block
(cheap, 101 taps) is fine, but changing coefficients instantaneously causes a small
discontinuity. Acceptable for a demo; a crossfade between old/new coefficient sets removes
it.

4. Kernel signature growth. run(out, fc) becomes roughly:
run(out, n, fc, phase_in_out, hist_in_out)
i.e. the block length n, plus two extra caller-owned state buffers the kernel
reads-then-writes. On the compiler side this is the same out-param/destination-passing
machinery you're already building — just more memref arguments threaded through
FuncOpLowering, no new mechanism.

5. C++ side. AudioCallback calls run(...) directly with out = ioData (after an f64 scratch
buffer + convert, since CoreAudio wants f32), n = inNumberFrames, and the persistent
gPhase/gHist state. The double-buffer/batch scaffolding goes away; the callback does real
work, so the kernel must stay within the real-time budget (101-tap FIR × ~512 frames is
trivially fine).

The biggest lift is #2 — the FIR history buffer is what makes blocks seamless, and it's the
one piece the static DSL pipeline has no concept of today.

## Performance Improvements

The `_run` object we generate today (Toy-tutorial lowering) is one heap buffer + one full
loop per DSP op, every scalar constant heap-allocated, ~25 malloc/free per call, a scalar
bounds-checked convolution, and no fusion. It looks alarming but most of the bloat is
one-time setup/teardown that is NOT on the hot path.

Flop budget for one regenerate() (44100-sample signal, 101-tap FIR):
- ~9 elementwise loops (t, phase, ones, frac, saw2, saw, wc, coef, ...): ~9 x 44100 ~= 0.4M
- coefficient build (sinc + hamming, with sin/cos): 101 taps, negligible
- FIR convolution (44200 x up to 101): ~4.5M mul-add  <- ~90% of runtime
- ~25 malloc + ~25 free: one-time, a few microseconds (<0.1% vs a few ms of compute)

So the convolution dominates and its inner loop is already the same arithmetic C would emit.
The naive version is within ~2-4x of hand C, not 20x. There are two independent axes of
improvement; do not conflate them. Axis A is backend codegen quality (matching C). Axis B is
the domain-specific algebraic rewrites the DSP-MLIR paper measures.

### Measuring

`bench.sh` builds the kernel object and a standalone driver (`bench.cpp`) that links the same
`_mlir_ciface_run` as the CoreAudio host but strips the audio machinery, so it times ONLY the
generated kernel. It is decoupled from how the object is built: change a pass/pipeline, rerun
`./bench.sh`, diff the output. Usage: `./bench.sh [--iterations N] [--warmup N]`.

Each run prints a human summary plus one machine-readable line an agent can parse:
`BENCH_JSON {"min_ms":..,"median_ms":..,"stddev_ms":..,"msample_per_s_median":..,"checksum":..}`.
- Compare `min_ms` (most stable) or `median_ms`; lower is better.
- `checksum` is the raw kernel output at cutoff=1000Hz. Axis A (pure codegen) must keep it
  ~bit-stable; Axis B (numeric rewrites like FFT convolution) should keep it within fp
  tolerance. A changed checksum on an Axis-A change means the optimization broke correctness.

`bench.sh` now builds with `--opt` by default (loop fusion + affine scalar
replacement); pass `OPT=0 ./bench.sh` to build the plain baseline for an A/B.
`--opt` used to assert in the affine loop-fusion pass on this kernel; that is
now fixed -- the sawtooth was rewritten to avoid `dsp.gain`'s 0-D-memref-in-loop
broadcast, and a new `AffineFusionLegalityPass` (runs right before fusion under
`--opt`) rejects that pattern with a clear diagnostic instead of crashing.

Baselines on this machine (checksum 1.769572322574197e+01, identical both ways):
- plain pipeline (`OPT=0`): min ~3.86 ms, median ~4.05 ms/call, ~11 Msample/s.
- `--opt` (current default): min ~3.44 ms, median ~3.59 ms/call, ~12 Msample/s
  (~12% faster). This is the number to beat.

What `--opt` already bought (vs plain): affine.for loops 13->9, memref.alloc
24->13, 44100-elt intermediate buffers 7->4, 0-D scalar memrefs 49->21. It did
NOT vectorize (zero NEON fp ops), did NOT hoist the convolution bounds check
(the `affine.if` zero-pad guard is still in the tap loop), and left 16
malloc/free in the emitted IR. So the hot-path wins (A1/A2) and all of Axis B
remain open. See the per-task status below.

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
