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
  // Per-voice CUTOFF envelope + FILTER state (the per-voice, trigger-anchored
  // low-pass -- see samples/notes.md "Per-voice, trigger-anchored cutoff").
  //   @voice_cut_phase : samples since this voice's last note-on, one slot per
  //     voice, carried across blocks. @run resets it to 0 at the EXACT trigger
  //     frame (sample-tight) and counts up, WRAPPING at the table length L, so the
  //     shape LOOPS as a continuous per-voice cutoff LFO (not a one-shot attack).
  //     The wrap is anchored at each voice's own trigger, so the voices' relative
  //     phases are preserved. Init 0 (an untriggered voice is silent anyway).
  //   @voice_lp_state  : the per-voice one-pole low-pass state y[n-1], continuous
  //     across blocks, reset to 0 at a note-on so a stolen voice starts a fresh
  //     filter instead of inheriting the previous note's tail.
  memref.global "public" @voice_cut_phase : memref<8xf64> = dense<0.000000e+00>
  memref.global "public" @voice_lp_state  : memref<8xf64> = dense<0.000000e+00>
  //   @voice_cut_step  : the cutoff-LFO speed LATCHED per voice at its note-on,
  //     one slot per voice, carried across blocks. @run copies the current global
  //     @cut_lfo_step into this slot only at a note-on, so a live speed change
  //     affects ONLY notes triggered after it -- already-sounding voices keep the
  //     speed they were born with. Init matches @cut_lfo_step's default.
  memref.global "public" @voice_cut_step  : memref<8xf64> = dense<1.000000e+00>
  // Shared cutoff-SHAPE table: the one-pole coefficient alpha as a function of
  // samples-since-trigger, ONE entry per sample of the sweep (table index == the
  // voice's WRAPPING @voice_cut_phase, so the shape repeats as a looping LFO).
  // Every voice reads THIS one table at its own phase, so (a) editing the table --
  // the host writes voice_cut_shape[] -- re-voices all live notes at once, while
  // (b) each voice keeps its own trigger-anchored phase and therefore its own
  // position in the cycle. The host fills a PERIODIC curve (equal at both ends) so
  // the wrap is click-free. Init is a splat; the host fills the real curve at
  // start-up (its SHAPE is a WebUI control for now).
  memref.global "public" @voice_cut_shape : memref<8000xf64> = dense<3.500000e-01>
  // Cutoff-LFO speed for NEW voices: how many table entries a voice's phase
  // advances per sample. LFO Hz = SAMPLE_RATE * step / L (L=8000): step 1 => ~5.5
  // Hz. The host sets this live from the +/- keys, but @run only LATCHES it into a
  // voice's @voice_cut_step at that voice's note-on, so changing it re-speeds only
  // notes triggered afterwards -- already-sounding voices keep their latched speed.
  memref.global "public" @cut_lfo_step : memref<f64> = dense<1.000000e+00>
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
    // per-voice cutoff-envelope constants (the LFO is gone; each voice reads its
    // alpha from the shared @voice_cut_shape table at its own trigger-anchored
    // phase, so all voices share one shape but keep independent positions).
    %vL     = arith.constant 8.000000e+03 : f64   // shape-table length; cp wraps here
    %vcpM   = memref.get_global @voice_cut_phase : memref<8xf64>
    %vlpM   = memref.get_global @voice_lp_state  : memref<8xf64>
    %vcsM   = memref.get_global @voice_cut_shape : memref<8000xf64>
    %vcstM  = memref.get_global @voice_cut_step  : memref<8xf64>
    // current cutoff-LFO speed for NEW voices, read once per block; only latched
    // into a voice's own @voice_cut_step at its note-on (see the loop below).
    %vstepM = memref.get_global @cut_lfo_step : memref<f64>
    %vstepNew = memref.load %vstepM[] : memref<f64>
    // clear the mix accumulator for this block
    affine.for %z = 0 to 128 {
      memref.store %vzero, %mixM[%z] : memref<128xf64>
    }
    // sum the 8 voices into @voice_mix, each low-passed by its OWN cutoff envelope
    // anchored sample-accurately at that voice's trigger frame (option A of the
    // notes.md voice-bank options: the per-voice one-pole is an extra iter_arg,
    // the cutoff envelope another per-voice phase carried across blocks).
    affine.for %v = 0 to 8 {
      %freq0 = memref.load %vfM[%v] : memref<8xf64>
      %ph0   = memref.load %vpM[%v] : memref<8xf64>
      %g0    = memref.load %vgM[%v] : memref<8xf64>
      %tgt0  = memref.load %vtM[%v] : memref<8xf64>
      %cph0  = memref.load %vcpM[%v] : memref<8xf64>
      %ylp0  = memref.load %vlpM[%v] : memref<8xf64>
      %vst0  = memref.load %vcstM[%v] : memref<8xf64>
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
      // note-on LATCHES the current global speed for this voice; else keep the
      // speed it was born with, so a live @cut_lfo_step change affects only new
      // notes. Block-level like %freqE: the gate mutes any pre-trigger portion.
      %stepE = arith.select %isOn, %vstepNew, %vst0 : f64
      %res:4 = affine.for %sn = 0 to 128
               iter_args(%gp = %g0, %pp = %phSt, %cp = %cph0, %yp = %ylp0)
               -> (f64, f64, f64, f64) {
        %ni  = arith.index_cast %sn : index to i64
        %nf  = arith.sitofp %ni : i64 to f64
        // is this the EXACT trigger frame of a note-on? (the sample-tight anchor
        // for both the cutoff envelope reset and the filter-state reset below)
        %onFr  = arith.cmpf oeq, %nf, %efrF : f64
        %atTrg = arith.andi %onFr, %isOn : i1
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
        // per-voice cutoff LFO phase: resets to 0 at the exact trigger frame, then
        // advances by this voice's latched speed (%stepE) each sample and WRAPS at
        // the table length L, so the shape loops continuously at that rate.
        %cpA   = arith.addf %cp, %stepE : f64
        %cpR   = arith.select %atTrg, %vzero, %cpA : f64
        %cpGeL = arith.cmpf oge, %cpR, %vL : f64
        %cpW   = arith.subf %cpR, %vL : f64
        %cpC   = arith.select %cpGeL, %cpW, %cpR : f64
        // alpha = shared shape table at this voice's trigger-anchored, wrapping
        // index. The table is host-filled and shared by every voice, so an edit
        // re-voices all live notes while each keeps its own cp (its own phase).
        %cpI   = arith.fptosi %cpC : f64 to i64
        %cpIdx = arith.index_cast %cpI : i64 to index
        %alpha = memref.load %vcsM[%cpIdx] : memref<8000xf64>
        // per-voice one-pole low-pass on the saw; state reset at the trigger:
        // y[n] = (1-alpha)*y[n-1] + alpha*saw
        %oma   = arith.subf %vone, %alpha : f64
        %ypR   = arith.select %atTrg, %vzero, %yp : f64
        %lpA   = arith.mulf %oma, %ypR : f64
        %lpB   = arith.mulf %alpha, %saw : f64
        %yn    = arith.addf %lpA, %lpB : f64
        // mix += filtered * gate * per-voice gain
        %sg  = arith.mulf %yn, %gn : f64
        %sgv = arith.mulf %sg, %vgain : f64
        %cur = memref.load %mixM[%sn] : memref<128xf64>
        %acc = arith.addf %cur, %sgv : f64
        memref.store %acc, %mixM[%sn] : memref<128xf64>
        affine.yield %gn, %pw, %cpC, %yn : f64, f64, f64, f64
      }
      // persist per-voice state; commit the pending target; consume the event
      memref.store %res#0, %vgM[%v] : memref<8xf64>
      memref.store %res#1, %vpM[%v] : memref<8xf64>
      memref.store %res#2, %vcpM[%v] : memref<8xf64>
      memref.store %res#3, %vlpM[%v] : memref<8xf64>
      memref.store %stepE, %vcstM[%v] : memref<8xf64>
      memref.store %freqE, %vfM[%v] : memref<8xf64>
      %tgtN = arith.select %hasEv, %eg, %tgt0 : f64
      memref.store %tgtN, %vtM[%v] : memref<8xf64>
      memref.store %vN, %efM[%v] : memref<8xi64>
    }
    // The voice mix now already holds the per-voice-FILTERED, gated sum: each
    // voice was low-passed inside the loop above by its own cutoff envelope,
    // anchored sample-accurately at that voice's trigger frame. So the summed
    // signal IS the tone -- there is no post-sum, global low-pass and no in-kernel
    // LFO any more (the old rank-1 dsp.lowPassFilter + phase-accumulator LFO +
    // Mode-B breakpoint interpolator are all gone). The @lfo_* globals and the
    // @set_value_lfo_* setters are retained, inert, only so the existing host
    // still links; @run ignores them in this per-voice-cutoff variant.
    %tone = "dsp.fromGlobal"() {global = @voice_mix} : () -> tensor<128xf64>

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
