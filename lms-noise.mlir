// Adaptive noise canceller (Widrow ANC) with a RUNTIME-selectable noise color
// driven by the @noise_kind knob via the dsp.index_switch op. dsp.index_switch
// is a first-class dsp op: a runtime index selector picks one of several regions
// and only the selected case runs.
//
//   x[n]  = noise color chosen at run time (white/pink/brown/ou, or none=silence)
//   n0[n] = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2]   (an "acoustic path", via delay)
//   saw   = sum of MIDI-triggered sawtooth voices (0 during rests; voice bank
//                                                  in @run, not a fixed 440 Hz)
//   tone  = lowPass(saw, alpha(t))               (swept 1st-order IIR, automated)
//   d[n]  = tone[n] + n0[n]                       (tone buried in noise)
//   y     = LMS(x -> d), a 32-tap adaptive FIR   (dsp.lmsFilterResponse)
//   out   = d - wet*y                            (noise removed, swept tone revealed)
//
// The low-pass sits on the *signal* path (the tone), not the final mix: a swept
// cutoff on a pure sine only tremolos its amplitude, but on a sawtooth it carves
// harmonics -- the classic, clearly audible filter sweep. The tone survives the
// adaptive canceller (it is uncorrelated with the noise reference x), so what you
// hear after cancellation is the swept-filtered sawtooth.
//
// WHAT THE DEMO ACTUALLY SHOWS (read before judging the canceller). The goal is
// pulling the noise OUT OF (tone + noise) -- the hard, realistic case -- NOT out
// of silence. While a note sounds, d = tone + n0: the LMS nulls the correlated
// n0, but the tone leaks into its gradient (w += mu*e*x, and the error e still
// carries the tone), so a small noise residual always survives -- measured about
// -25 dB below the tone at mu=1e-3. THAT residual is the real ANC behaviour and
// is the interesting part; it is exactly the "signal leaks into the adaptive
// filter" effect, and it scales with mu. Because the tone is now MIDI-GATED, a
// REST has d = n0 (pure noise): a 32-tap filter models the exact 3-tap FIR path
// perfectly and nulls it to ~machine zero -- dead silence. That "perfect"
// cancellation during rests is the TRIVIAL case (nothing but the reference to
// cancel), not a success metric. Always evaluate the canceller with a note held.
//
// dsp.index_switch mirrors scf.index_switch's syntax -- an index selector,
// integer `case` regions, a mandatory `default`, regions yielding a common
// result type (dsp.yield terminator). It is a distinct dsp op (not scf.index_-
// switch directly) so it travels through the tensor-land frontend and only
// bufferizes during affine lowering, where its own ConversionPattern rewrites it
// to a memref-yielding scf.index_switch (one function, a runtime branch). This
// is the intended, final lowering of dsp.index_switch.
//
// State model:
//   * ops INSIDE the switch regions (the noise generators) get INDEPENDENT
//     stream-state globals -- one per case, materialized by the stream-state
//     pass, which walks into the switch regions before lowering.
//   * ops OUTSIDE the switch (delay x2, LMS, tone phase) share ONE state global
//     each across all variants.
//   * the optional `reset = "region_local"` attribute is parsed but IGNORED by
//     this runtime-branch lowering; it is reserved for the planned, separate
//     dsp.variant_switch specialization/unswitch op (see samples/notes.md),
//     which would clone the enclosing @run per case, build a dispatcher, and
//     restart the noise state only on a switch.
//
// Cross-call persistence (block streaming) -- compile with --stream -- is
// unchanged from lms-noise.mlir; the @run signature is identical, so the
// existing interactive host drives this kernel as-is (Up/Down = @wet,
// Left/Right = @noise_kind, +/- = @lfo_period cutoff-sweep speed).
module {
  // LMS adaptation rate (interactive, read at render time inside the kernel).
  memref.global "public" @mu : memref<f64> = dense<1.000000e-03>
  // Wet mix (interactive): fraction of the estimated noise to subtract.
  memref.global "public" @wet : memref<f64> = dense<0.000000e+00>
  // Noise color select (interactive): 0=white 1=pink 2=brown 3=ou, and any
  // other value ("none") = silence. Read once per @run call as the
  // dsp.index_switch selector; v1 assumes exactly this shape (a plain load of
  // a named global), mid-term an arbitrary SSA selector.
  memref.global "public" @noise_kind : memref<i64> = dense<0>
  // --- polyphonic MIDI voice bank (fixed V = 8) -----------------------------
  // A fixed number of voices, on purpose: a compile-time voice count keeps the
  // block-render loop's trip count a literal (0..8) and every buffer statically
  // shaped, so the kernel stays malloc-free and the static-optimisation
  // properties in notes.md hold. The bank REPLACES the old continuously-running
  // 440 Hz @sample_offset tone: nothing sounds until a MIDI note triggers a
  // voice. Each voice keeps its own persistent state across blocks (one slot per
  // voice in these arrays), the same "implicit state" idea @sample_offset used:
  //   @voice_freq  : current oscillator frequency (Hz), held while the note sounds
  //   @voice_phase : sawtooth phase accumulator in [0,1), continuous across blocks
  //   @voice_gate  : one-pole-SMOOTHED amplitude in [0,1] (click-free attack/rel.)
  //   @voice_tgt   : the gate's step TARGET (0 = released, 1 = held) it chases
  memref.global "public" @voice_freq  : memref<8xf64> = dense<0.000000e+00>
  memref.global "public" @voice_phase : memref<8xf64> = dense<0.000000e+00>
  memref.global "public" @voice_gate  : memref<8xf64> = dense<0.000000e+00>
  memref.global "public" @voice_tgt   : memref<8xf64> = dense<0.000000e+00>
  // Pending MIDI event per voice -- the (value, frame) timestamped-setter record,
  // exactly like @lfo_period_pending/@lfo_period_frame but one per voice. The host
  // stages a note on/off through @note_event; @run consumes it, applies the gate
  // step at frame `ev_frame` (0..N; N=128 means "no pending event") and, for a
  // note-on, retunes to `ev_freq`. So an event at any sample inside the 128-frame
  // block is rendered at that exact frame -- the prefilled-vector idea from the
  // LFO, realised per voice.
  memref.global "public" @voice_ev_frame : memref<8xi64> = dense<128>
  memref.global "public" @voice_ev_gate  : memref<8xf64> = dense<0.000000e+00>
  memref.global "public" @voice_ev_freq  : memref<8xf64> = dense<0.000000e+00>
  // Per-block scratch the voice loop sums into, then bridges to tensor land via
  // dsp.fromGlobal (no to_tensor hook in this toolchain). Private.
  memref.global "private" @voice_mix : memref<128xf64> = dense<0.000000e+00>
  // Low-pass cutoff-LFO period in samples (interactive): the automation speed.
  // A smaller period = faster sweep (LFO Hz = 44100 / period; default 147000 ≈
  // 0.3 Hz). This is now the *held/current* value of the parameter: the host no
  // longer writes it directly but schedules changes via @set_value_lfo_period,
  // and @run commits the pending value into this global at the end of the block.
  memref.global "public" @lfo_period : memref<i64> = dense<147000>
  // Pending @lfo_period value + the frame (0..N) at which it takes effect within
  // the next block -- the timestamped-setter state record (phase-mode param).
  // frame == N (=128) means "no pending change". @set_value_lfo_period writes
  // these; @run reads them to advance the LFO phase sample-accurately, then
  // commits (@lfo_period := pending, frame := N).
  memref.global "public" @lfo_period_pending : memref<i64> = dense<147000>
  memref.global "public" @lfo_period_frame : memref<i64> = dense<128>
  // Cutoff-LFO phase accumulator, one cycle = [0,1). Each @run advances it by
  // N/@lfo_period and stores it back, so changing @lfo_period only changes the
  // per-block *increment*, never the accumulated phase -- rate changes are then
  // click-free (unlike `offset mod period`, which jumps the phase when period
  // changes). Store 0 to restart the sweep from its muffled edge.
  memref.global "public" @lfo_phase : memref<f64> = dense<0.000000e+00>

  // --- Mode B: host-side LFO (breakpoint-envelope) state --------------------
  // @lfo_mode selects who computes the cutoff-alpha sweep, read once per block:
  //   0 = KERNEL-side  (the phase-accumulator triangle above; @lfo_period),
  //   1 = HOST-side    (arbitrary shape streamed as breakpoints; below).
  // Both alphas are always computed; @run blends by mode (m in {0,1}) as
  // alpha = (1-m)*alphaKernel + m*alphaHost, so switching is branch-free.
  memref.global "public" @lfo_mode : memref<i64> = dense<0>
  // Breakpoint array (Mode B input): ONE slot per frame in the upcoming block,
  // slot index == frame. @set_value_lfo_breakpoint(value, frame) writes
  // @lfo_bp[frame] = value; @run linearly interpolates between the occupied
  // slots to fill a per-sample alpha. Empty slots hold the SENTINEL -1.0 (out of
  // the valid alpha range [0.02,0.35]); @run resets every slot back to -1.0 as
  // it consumes them, so a slot left untouched next block reads as "no anchor".
  memref.global "public" @lfo_bp : memref<128xf64> = dense<-1.000000e+00>
  // Per-sample host alpha buffer, filled by the interpolation loop in @run and
  // lifted into tensor land via dsp.fromGlobal (this toolchain has no
  // bufferization/to_tensor hook, so a hand-written loop result reaches the
  // tensor-typed low-pass only through that bridge op). Private scratch.
  memref.global "private" @lfo_alpha_host : memref<128xf64> = dense<2.000000e-02>
  // Scratch for the interpolator's backward pass: per sample, the value/frame of
  // the nearest breakpoint at-or-after it (rf == 128.0 means "none ahead").
  memref.global "private" @lfo_rv : memref<128xf64> = dense<0.000000e+00>
  memref.global "private" @lfo_rf : memref<128xf64> = dense<1.280000e+02>
  // Mode-B continuity: alpha at the end of the previous block, so the first
  // segment of the next block ramps from where the last one ended (no click).
  memref.global "public" @lfo_alpha_carry : memref<f64> = dense<2.000000e-02>

  // BLOCK SIZE: N = 128 samples per @run call (~2.9 ms at 44100). Small on
  // purpose: parameters are read once per call (control-rate == block-rate), so
  // a small N keeps automation smooth -- e.g. the low-pass cutoff LFO below
  // updates every ~2.9 ms (~344 Hz), fast enough to sweep without audible steps;
  // a large N would step it. dt below stays 1/44100 -- it is the sample period,
  // not 1/N -- so the tone frequency is unchanged. Noise/LMS/delay AND the IIR
  // low-pass stay continuous across calls via --stream state, and the tone's
  // time base is continuous via @sample_offset, so there is no per-block click.
  // To change N: replace the tensor/memref shapes, the two count constants
  // (%n, %cnt), and the @sample_offset advance (arith.constant 128).
  dsp.func @run(%out: memref<128xf64>) attributes {llvm.emit_c_interface} {
    // --- shared noise parameters (defined outside the switch; the regions
    //     reference them -- dsp.index_switch is NOT isolated-from-above) ---
    %seed  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %sigma = dsp.constant dense<1.000000e+00> : tensor<f64>
    %n     = dsp.constant dense<1.280000e+02> : tensor<f64>

    // --- selector: load @noise_kind once, interpret as an index ---
    %kmem  = memref.get_global @noise_kind : memref<i64>
    %kval  = memref.load %kmem[] : memref<i64>
    %ksel  = arith.index_cast %kval : i64 to index

    // --- runtime noise-color reference x[n] ---
    // Cases 0..3 each hold a stateful noise generator with its OWN persistent
    // stream-state global; the default ("none") case is silence -- a stateless
    // zero vector (getRangeOfVector with step 0), so it carries no state global.
    // The reset="region_local" attribute (reserved for dsp.variant_switch) is
    // ignored here: under the runtime-branch lowering the noise streams simply
    // persist per case, so switching color resumes that case's stream.
    %x = dsp.index_switch %ksel {reset = "region_local"} -> tensor<128xf64>
      case 0 {
        %w = "dsp.noise_white"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
        dsp.yield %w : tensor<128xf64>
      }
      case 1 {
        %p = "dsp.noise_pink"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
        dsp.yield %p : tensor<128xf64>
      }
      case 2 {
        %b = "dsp.noise_brown"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
        dsp.yield %b : tensor<128xf64>
      }
      case 3 {
        %o = "dsp.noise_ou"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
        dsp.yield %o : tensor<128xf64>
      }
      default {
        // "none": silence. Different signature from the noise ops (no
        // seed/sigma, no stream state) -- exactly why cases are regions, not an
        // op-rename. Zeros = getRangeOfVector(first=0, N, step=0).
        %z0 = dsp.constant dense<0.000000e+00> : tensor<f64>
        %s  = "dsp.getRangeOfVector"(%z0, %n, %z0) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
        dsp.yield %s : tensor<128xf64>
      }

    // --- acoustic path n0[n] = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2] ---
    // Delays turn noise into a 3-tap impulse response with real time structure,
    // so the adaptive filter has a genuine multi-tap path to converge on
    %zero = dsp.constant dense<0.000000e+00> : tensor<f64>
    %cnt  = dsp.constant dense<1.280000e+02> : tensor<f64>
    %d1c  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %d2c  = dsp.constant dense<2.000000e+00> : tensor<f64>
    %x1 = "dsp.delay"(%x, %d1c) : (tensor<128xf64>, tensor<f64>) -> tensor<128xf64>
    %x2 = "dsp.delay"(%x, %d2c) : (tensor<128xf64>, tensor<f64>) -> tensor<128xf64>

    %c07 = dsp.constant dense<7.000000e-01> : tensor<f64>
    %c05 = dsp.constant dense<5.000000e-01> : tensor<f64>
    %c03 = dsp.constant dense<3.000000e-01> : tensor<f64>
    %g07 = "dsp.getRangeOfVector"(%c07, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %g05 = "dsp.getRangeOfVector"(%c05, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %g03 = "dsp.getRangeOfVector"(%c03, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %p0 = "dsp.mul"(%x, %g07)  : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %p1 = "dsp.mul"(%x1, %g05) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %p2 = "dsp.mul"(%x2, %g03) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %n01 = "dsp.add"(%p0, %p1) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %n0  = "dsp.add"(%n01, %p2) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>

    // %ones (a constant-1 vector) is used by the LFO section further down; the
    // tone that used to be here is now the polyphonic voice bank below.
    %one1  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %ones  = "dsp.getRangeOfVector"(%one1, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>

    // --- polyphonic MIDI-triggered sawtooth voices (replaces the continuous
    //     440 Hz tone) -------------------------------------------------------
    // Design in one place: a FIXED bank of 8 voices, each an independent naive
    // sawtooth. Why the choices below:
    //  * Fixed count (compile-time 0..8 loop): keeps the block static and
    //    malloc-free (notes.md). Dynamic polyphony would need a runtime bound.
    //  * Per-voice state carried across blocks in the @voice_* globals (phase,
    //    smoothed gate, target, freq) -- one slot per voice, the @sample_offset
    //    "implicit state" pattern generalised to a bank.
    //  * Arbitrary trigger frame handled the LFO way: the pending (value, frame)
    //    event steps the gate TARGET at exactly `ev_frame` inside the block; a
    //    one-pole smoother chases it, so the note starts/stops sample-accurately
    //    and click-free without splitting the block.
    //  * Written as a hand-rolled affine loop nest (voice-outer, sample-inner)
    //    summed into @voice_mix and bridged to tensor land by dsp.fromGlobal --
    //    the same escape hatch the LFO breakpoint interpolator uses (a per-voice
    //    phase/gate recurrence is a scan, not expressible as pure tensor ops).
    //    Trade-off: this bank is opaque to the tensor-level fusion/Axis-B
    //    rewrites; the alternatives (unrolled tensor lanes, or voices as inlined
    //    dsp.func calls) are noted in samples/notes.md for later discussion.
    %vgain = arith.constant 2.000000e-01 : f64            // per-voice gain: 8*0.2 headroom
    %kc    = arith.constant 1.000000e-02 : f64            // gate smoother rate (~2 ms t.c.)
    %vdt   = arith.constant 2.2675736961451248E-5 : f64   // 1/44100
    %vone  = arith.constant 1.000000e+00 : f64
    %vtwo  = arith.constant 2.000000e+00 : f64
    %vzero = arith.constant 0.000000e+00 : f64
    %vhalf = arith.constant 5.000000e-01 : f64
    %vN    = arith.constant 128 : i64
    %vfM   = memref.get_global @voice_freq     : memref<8xf64>
    %vpM   = memref.get_global @voice_phase    : memref<8xf64>
    %vgM   = memref.get_global @voice_gate     : memref<8xf64>
    %vtM   = memref.get_global @voice_tgt      : memref<8xf64>
    %efM   = memref.get_global @voice_ev_frame : memref<8xi64>
    %egM   = memref.get_global @voice_ev_gate  : memref<8xf64>
    %erM   = memref.get_global @voice_ev_freq  : memref<8xf64>
    %mixM  = memref.get_global @voice_mix      : memref<128xf64>
    // clear the mix accumulator for this block
    affine.for %z = 0 to 128 {
      memref.store %vzero, %mixM[%z] : memref<128xf64>
    }
    // sum the 8 voices into @voice_mix
    affine.for %v = 0 to 8 {
      %freq0 = memref.load %vfM[%v] : memref<8xf64>
      %ph0   = memref.load %vpM[%v] : memref<8xf64>
      %g0    = memref.load %vgM[%v] : memref<8xf64>
      %tgt0  = memref.load %vtM[%v] : memref<8xf64>
      %efrI  = memref.load %efM[%v] : memref<8xi64>
      %eg    = memref.load %egM[%v] : memref<8xf64>
      %ef    = memref.load %erM[%v] : memref<8xf64>
      %efrF  = arith.sitofp %efrI : i64 to f64
      // event present this block? note-on == event whose gate target > 0.5
      %hasEv = arith.cmpi slt, %efrI, %vN : i64
      %egOn  = arith.cmpf ogt, %eg, %vhalf : f64
      %isOn  = arith.andi %hasEv, %egOn : i1
      // note-on retunes the voice and restarts its phase; else hold current
      %freqE = arith.select %isOn, %ef, %freq0 : f64
      %phSt  = arith.select %isOn, %vzero, %ph0 : f64
      %inc   = arith.mulf %freqE, %vdt : f64
      %res:2 = affine.for %sn = 0 to 128 iter_args(%gp = %g0, %pp = %phSt) -> (f64, f64) {
        %ni  = arith.index_cast %sn : index to i64
        %nf  = arith.sitofp %ni : i64 to f64
        // gate target: old target before the event frame, new target from it on
        %pre = arith.cmpf olt, %nf, %efrF : f64
        %tgt = arith.select %pre, %tgt0, %eg : f64
        // one-pole gate smoother: g += kc*(target - g)
        %gd  = arith.subf %tgt, %gp : f64
        %gst = arith.mulf %kc, %gd : f64
        %gn  = arith.addf %gp, %gst : f64
        // advance + wrap phase, build 2*frac(phase)-1 sawtooth
        %pr  = arith.addf %pp, %inc : f64
        %pge = arith.cmpf oge, %pr, %vone : f64
        %ps  = arith.subf %pr, %vone : f64
        %pw  = arith.select %pge, %ps, %pr : f64
        %s2  = arith.mulf %vtwo, %pw : f64
        %saw = arith.subf %s2, %vone : f64
        // mix += saw * gate * per-voice gain
        %sg  = arith.mulf %saw, %gn : f64
        %sgv = arith.mulf %sg, %vgain : f64
        %cur = memref.load %mixM[%sn] : memref<128xf64>
        %acc = arith.addf %cur, %sgv : f64
        memref.store %acc, %mixM[%sn] : memref<128xf64>
        affine.yield %gn, %pw : f64, f64
      }
      // persist per-voice state; commit the pending target; consume the event
      memref.store %res#0, %vgM[%v] : memref<8xf64>
      memref.store %res#1, %vpM[%v] : memref<8xf64>
      memref.store %freqE, %vfM[%v] : memref<8xf64>
      %tgtN = arith.select %hasEv, %eg, %tgt0 : f64
      memref.store %tgtN, %vtM[%v] : memref<8xf64>
      memref.store %vN, %efM[%v] : memref<8xi64>
    }
    %saw = "dsp.fromGlobal"() {global = @voice_mix} : () -> tensor<128xf64>

    // --- automated low-pass on the tone: cutoff swept by a slow in-kernel LFO ---
    // The cutoff coefficient alpha is now materialised PER SAMPLE as a
    // tensor<128xf64> (a smooth in-block sweep) rather than one control-rate
    // value held for the whole block. The LFO is a phase accumulator -- the
    // "phase-mode interpolation" of the @lfo_period parameter: from the running
    // phase phase0 and the held period P0 we build phase[n] = phase0 + n/P0,
    // wrap to [0,1), and map a triangle to alpha[n] = 0.02 + 0.33*tri[n].
    // dsp.lowPassFilter now consumes this per-sample (rank-1) alpha; it still
    // carries --stream state (its previous output), so the tone is continuous
    // across calls. Applied to the sawtooth (the signal path), so the sweep is
    // an audible timbral change. Built entirely from tensor ops (getRangeOfVector
    // ramp, modulo wrap, abs for |2f-1|), so it stays in tensor land.
    %one1f  = arith.constant 1.000000e+00 : f64
    %blkf   = arith.constant 1.280000e+02 : f64          // N (block size)
    // held/current period P0 and the running LFO phase at block start.
    %lpPmem = memref.get_global @lfo_period : memref<i64>
    %lpP    = memref.load %lpPmem[] : memref<i64>
    %lpPf   = arith.sitofp %lpP : i64 to f64
    %phmem  = memref.get_global @lfo_phase : memref<f64>
    %phase0 = memref.load %phmem[] : memref<f64>         // phase at start of block
    // per-sample phase increment incHeld = 1/P0 (cycles per sample), then
    // phase[n] = phase0 + n*incHeld as a tensor, wrapped to [0,1).
    %incH   = arith.divf %one1f, %lpPf : f64
    %incHt  = tensor.from_elements %incH   : tensor<f64>
    %ph0t   = tensor.from_elements %phase0 : tensor<f64>
    %phRamp = "dsp.getRangeOfVector"(%ph0t, %cnt, %incHt) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %phW    = "dsp.modulo"(%phRamp, %ones) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    // triangle tri[n] = 1 - |2*phase - 1|  (0 at edges, 1 at centre), per sample.
    %ph2    = "dsp.add"(%phW, %phW) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %ph2m1  = "dsp.sub"(%ph2, %ones) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %phAbs  = "dsp.abs"(%ph2m1) : (tensor<128xf64>) -> tensor<128xf64>
    %triV   = "dsp.sub"(%ones, %phAbs) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    // alphaKernel[n] = 0.02 + 0.33 * tri[n]  -> in [0.02, 0.35], per sample. This
    // is Mode A (kernel-side LFO). Mode B replaces it with a host-streamed shape.
    %c033   = dsp.constant dense<3.300000e-01> : tensor<f64>   // span -> up to ~0.35 (bright)
    %c002   = dsp.constant dense<2.000000e-02> : tensor<f64>   // muffled floor (~140 Hz)
    %spanV  = "dsp.getRangeOfVector"(%c033, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %aminV  = "dsp.getRangeOfVector"(%c002, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %amodV  = "dsp.mul"(%triV, %spanV) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %alphaKV = "dsp.add"(%amodV, %aminV) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>

    // --- Mode B: HOST-side LFO (breakpoint-envelope interpolation) ------------
    // The host streams a sparse shape into @lfo_bp (slot == frame, empty == -1.0
    // sentinel). We fill a per-sample alpha by piecewise-LINEAR interpolation
    // between the occupied slots -- a scan that pure tensor ops can't express (no
    // elementwise select / prefix), so it is a hand-written affine.for pair
    // writing memrefs, bridged back to tensor land by dsp.fromGlobal. For a
    // triangle this reproduces Mode A exactly (a triangle IS piecewise-linear);
    // for any other breakpoint set it draws an arbitrary shape.
    %sent   = arith.constant -1.000000e+00 : f64                // "no anchor here"
    %c127i  = arith.constant 127 : index
    %bpMem  = memref.get_global @lfo_bp : memref<128xf64>
    %rvMem  = memref.get_global @lfo_rv : memref<128xf64>
    %rfMem  = memref.get_global @lfo_rf : memref<128xf64>
    %ahMem  = memref.get_global @lfo_alpha_host : memref<128xf64>
    %acMem  = memref.get_global @lfo_alpha_carry : memref<f64>
    %carry0 = memref.load %acMem[] : memref<f64>                 // alpha at end of prev block
    // Backward pass: for each sample n (127..0) record the nearest anchor
    // at-or-after it -- value into @lfo_rv, frame into @lfo_rf (128.0 == none).
    %rv0 = arith.constant 0.000000e+00 : f64
    %rf0 = arith.constant 1.280000e+02 : f64
    affine.for %i = 0 to 128 iter_args(%rvAcc = %rv0, %rfAcc = %rf0) -> (f64, f64) {
      %nb    = arith.subi %c127i, %i : index
      %nib   = arith.index_cast %nb : index to i64
      %nfb   = arith.sitofp %nib : i64 to f64
      %bvb   = memref.load %bpMem[%nb] : memref<128xf64>
      %isbpb = arith.cmpf one, %bvb, %sent : f64
      %rvN   = arith.select %isbpb, %bvb, %rvAcc : f64
      %rfN   = arith.select %isbpb, %nfb, %rfAcc : f64
      memref.store %rvN, %rvMem[%nb] : memref<128xf64>
      memref.store %rfN, %rfMem[%nb] : memref<128xf64>
      affine.yield %rvN, %rfN : f64, f64
    }
    // Forward pass: carry the nearest anchor at-or-before (pv,pf); interpolate to
    // the right anchor (rv,rf); hold when there is none ahead (rf==128) or we sit
    // exactly on an anchor (denom==0). Reset each consumed slot to the sentinel.
    // The 3rd iter_arg carries the last alpha so we can store the block-end value.
    %pf0 = arith.constant 0.000000e+00 : f64
    %maxfr = arith.constant 1.280000e+02 : f64
    %la:3 = affine.for %m = 0 to 128 iter_args(%pv = %carry0, %pf = %pf0, %laAcc = %carry0) -> (f64, f64, f64) {
      %mi    = arith.index_cast %m : index to i64
      %mf    = arith.sitofp %mi : i64 to f64
      %bvf   = memref.load %bpMem[%m] : memref<128xf64>
      %isbpf = arith.cmpf one, %bvf, %sent : f64
      %pvN   = arith.select %isbpf, %bvf, %pv : f64
      %pfN   = arith.select %isbpf, %mf, %pf : f64
      %rv    = memref.load %rvMem[%m] : memref<128xf64>
      %rf    = memref.load %rfMem[%m] : memref<128xf64>
      %denom = arith.subf %rf, %pfN : f64
      %none  = arith.cmpf oge, %rf, %maxfr : f64                 // no anchor ahead
      %zeroD = arith.cmpf oeq, %denom, %pf0 : f64                // denom == 0
      %hold  = arith.ori %none, %zeroD : i1
      %safeD = arith.select %hold, %one1f, %denom : f64          // avoid div-by-zero
      %dv    = arith.subf %rv, %pvN : f64
      %slope = arith.divf %dv, %safeD : f64
      %dn    = arith.subf %mf, %pfN : f64
      %step  = arith.mulf %dn, %slope : f64
      %interp = arith.addf %pvN, %step : f64
      %alpha = arith.select %hold, %pvN, %interp : f64
      memref.store %alpha, %ahMem[%m] : memref<128xf64>
      memref.store %sent, %bpMem[%m] : memref<128xf64>           // consume: reset slot
      affine.yield %pvN, %pfN, %alpha : f64, f64, f64
    }
    memref.store %la#2, %acMem[] : memref<f64>                   // carry block-end alpha
    %alphaHV = "dsp.fromGlobal"() {global = @lfo_alpha_host} : () -> tensor<128xf64>

    // --- blend by mode: alpha = (1-m)*alphaKernel + m*alphaHost, m in {0,1} ---
    %mdMem  = memref.get_global @lfo_mode : memref<i64>
    %mdI    = memref.load %mdMem[] : memref<i64>
    %mdF    = arith.sitofp %mdI : i64 to f64
    %omF    = arith.subf %one1f, %mdF : f64
    %mdT    = tensor.from_elements %mdF : tensor<f64>
    %omT    = tensor.from_elements %omF : tensor<f64>
    %mdV    = "dsp.getRangeOfVector"(%mdT, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %omV    = "dsp.getRangeOfVector"(%omT, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %aKw    = "dsp.mul"(%alphaKV, %omV) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %aHw    = "dsp.mul"(%alphaHV, %mdV) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %alphaV = "dsp.add"(%aKw, %aHw) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %tone   = "dsp.lowPassFilter"(%saw, %alphaV) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>

    // --- advance the LFO phase, honouring the timestamped @lfo_period change ---
    // A pending change (scheduled by @set_value_lfo_period with a frame
    // timestamp) takes effect at frame f within THIS block: the phase advances at
    // the held rate 1/P0 for f samples and the pending rate 1/Pp for the
    // remaining N-f, so the exact event frame is reflected in the carried phase
    // (sample-accurate), even though the within-block alpha ramp above uses the
    // held period (the pending period becomes the ramp slope from next block).
    // f == N (=128) means "no pending change" (advance = N/P0, as before).
    // Advancing only the increment (not re-deriving phase from a counter) keeps
    // sweep-speed changes click-free.
    %ppMem  = memref.get_global @lfo_period_pending : memref<i64>
    %ppI    = memref.load %ppMem[] : memref<i64>
    %ppF    = arith.sitofp %ppI : i64 to f64
    %frMem  = memref.get_global @lfo_period_frame : memref<i64>
    %frI    = memref.load %frMem[] : memref<i64>
    %frF    = arith.sitofp %frI : i64 to f64
    %incP   = arith.divf %one1f, %ppF : f64              // 1/pending
    %nMinF  = arith.subf %blkf, %frF : f64               // N - f
    %advH   = arith.mulf %frF, %incH : f64               // f/P0
    %advP   = arith.mulf %nMinF, %incP : f64             // (N-f)/pending
    %adv    = arith.addf %advH, %advP : f64
    %phraw  = arith.addf %phase0, %adv : f64
    %phge1  = arith.cmpf oge, %phraw, %one1f : f64       // inc<1 so one subtract suffices
    %phsub  = arith.subf %phraw, %one1f : f64
    %phnext = arith.select %phge1, %phsub, %phraw : f64  // wrap to [0,1)
    memref.store %phnext, %phmem[] : memref<f64>
    // commit: held := pending, and mark the event consumed (frame := N).
    memref.store %ppI, %lpPmem[] : memref<i64>
    %blkI   = arith.constant 128 : i64
    memref.store %blkI, %frMem[] : memref<i64>

    // --- desired d[n] = tone + colored noise ---
    %d = "dsp.add"(%tone, %n0) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>

    // --- 32-tap LMS: learn the noise path, weights persist across calls ---
    %mumem = memref.get_global @mu : memref<f64>
    %muval = memref.load %mumem[] : memref<f64>
    %mu    = tensor.from_elements %muval : tensor<f64>
    %flen  = dsp.constant dense<3.200000e+01> : tensor<f64>
    %y = "dsp.lmsFilterResponse"(%x, %d, %mu, %flen) : (tensor<128xf64>, tensor<128xf64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>

    // --- out = d - wet*y  (interactive wet mix; wet=1 -> full cancellation) ---
    %wetmem = memref.get_global @wet : memref<f64>
    %wetval = memref.load %wetmem[] : memref<f64>
    %wet    = tensor.from_elements %wetval : tensor<f64>
    %wy     = "dsp.gain"(%y, %wet) : (tensor<128xf64>, tensor<f64>) -> tensor<128xf64>
    %outt   = "dsp.sub"(%d, %wy) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    dsp.return %outt : tensor<128xf64>
  }

  // MIDI voice event setter (out-of-band ABI, exported as
  // _mlir_ciface_set_note_event). The host does voice ALLOCATION (assign a note
  // to one of the 8 voices, steal on overflow) and then schedules the resulting
  // gate change here as a (voice, freq, gate, frame) tuple:
  //   note-on  -> note_event(v, freqHz, 1.0, frame)
  //   note-off -> note_event(v, 0.0,    0.0, frame)
  // It records the event into the voice's pending slots; @run's voice loop above
  // applies it at `frame` and consumes it. `frame` is 0..N (N=128 == "no event").
  // Limitation (v1): one pending event per voice per block -- the host allocator
  // must not target the same voice twice in one block (it allocates a different
  // free voice instead), and must call this only from the render thread so the
  // write and @run's consume of the pending slots stay sequential (no race), the
  // same discipline the LFO breakpoint setter uses.
  dsp.func @set_note_event(%voice: i64, %freq: f64, %gate: f64, %frame: i64) attributes {llvm.emit_c_interface} {
    %vi  = arith.index_cast %voice : i64 to index
    %frm = memref.get_global @voice_ev_frame : memref<8xi64>
    memref.store %frame, %frm[%vi] : memref<8xi64>
    %gm  = memref.get_global @voice_ev_gate : memref<8xf64>
    memref.store %gate, %gm[%vi] : memref<8xf64>
    %fm  = memref.get_global @voice_ev_freq : memref<8xf64>
    memref.store %freq, %fm[%vi] : memref<8xf64>
    dsp.return
  }

  // Timestamped parameter setter (out-of-band ABI, exported as
  // _mlir_ciface_set_value_lfo_period). The host calls this -- instead of
  // writing @lfo_period directly -- to schedule a new sweep-speed value that
  // takes effect at frame `frame` (0..N) within the next block. It just records
  // the pending value and frame into the parameter's state globals; @run above
  // consumes them (phase-mode interpolation) and commits. This is the first
  // prototype of the notes.md "parameters become setter-functions carrying a
  // MIDI timestamp" direction, kept in raw MLIR (no new dsp op yet).
  dsp.func @set_value_lfo_period(%value: i64, %frame: i64) attributes {llvm.emit_c_interface} {
    %pmem = memref.get_global @lfo_period_pending : memref<i64>
    memref.store %value, %pmem[] : memref<i64>
    %fmem = memref.get_global @lfo_period_frame : memref<i64>
    memref.store %frame, %fmem[] : memref<i64>
    dsp.return
  }

  // Mode-B breakpoint setter (exported as _mlir_ciface_set_value_lfo_breakpoint).
  // The host calls it -- typically several times per block -- to place a shape
  // breakpoint: @lfo_bp[frame] = value, i.e. "alpha should reach `value` at frame
  // `frame` within the next block". @run interpolates linearly between the
  // occupied slots and resets them to the -1.0 sentinel as it consumes them.
  // NOTE: for now the host must not call this while @run is consuming the array
  // (no cross-thread guard yet); a future revision rejects writes during
  // consumption. `frame` is assumed in [0,128).
  dsp.func @set_value_lfo_breakpoint(%value: f64, %frame: i64) attributes {llvm.emit_c_interface} {
    %bpmem = memref.get_global @lfo_bp : memref<128xf64>
    %idx   = arith.index_cast %frame : i64 to index
    memref.store %value, %bpmem[%idx] : memref<128xf64>
    dsp.return
  }
}
