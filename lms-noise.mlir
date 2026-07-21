// Adaptive noise canceller (Widrow ANC) with a RUNTIME-selectable noise color
// driven by the @noise_kind knob via the dsp.index_switch op. dsp.index_switch
// is a first-class dsp op: a runtime index selector picks one of several regions
// and only the selected case runs.
//
//   x[n]  = noise color chosen at run time (white/pink/brown/ou, or none=silence)
//   n0[n] = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2]   (an "acoustic path", via delay)
//   saw   = 2*frac(440*t) - 1                     (harmonically rich tone)
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
  // Running sample index for the tone's time base. The kernel reads it, offsets t
  // by offset*dt, and stores offset+N back each call, so the sine phase is
  // continuous across blocks with no host involvement ("implicit" state). Store 0
  // to restart the tone from phase 0. It grows unbounded (i64); at f64 precision
  // the sin argument stays accurate for many hours, so no wrap is needed (it could
  // be wrapped mod 44100, one exact tone period, for indefinite exactness).
  memref.global "public" @sample_offset : memref<i64> = dense<0>
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

    // --- 440 Hz sawtooth tone: 2*frac(440*t) - 1, continuous across blocks ---
    // Phase continuity: instead of restarting t at 0 each call, resume the time
    // base from the persistent @sample_offset counter, so t[n] = (offset+n)*dt.
    // Read offset, build t from offset*dt, then store offset+N back -- the host
    // never touches it ("implicit" global state). A sawtooth (harmonically rich),
    // not a sine, so the swept low-pass below has partials to carve. Naive
    // (aliasing) saw -- fine for a demo; frac(x) = x mod 1 via dsp.modulo by 1.
    %dt     = dsp.constant dense<2.2675736961451248E-5> : tensor<f64>  // 1/44100
    %offmem = memref.get_global @sample_offset : memref<i64>
    %offval = memref.load %offmem[] : memref<i64>
    %offf   = arith.sitofp %offval : i64 to f64
    %dtf    = arith.constant 2.2675736961451248E-5 : f64               // 1/44100
    %t0f    = arith.mulf %offf, %dtf : f64                             // offset*dt
    %tstart = tensor.from_elements %t0f : tensor<f64>
    %t      = "dsp.getRangeOfVector"(%tstart, %cnt, %dt) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    // advance the counter by one block (N) for the next call
    %blk    = arith.constant 128 : i64
    %offnew = arith.addi %offval, %blk : i64
    memref.store %offnew, %offmem[] : memref<i64>
    %f440  = dsp.constant dense<4.400000e+02> : tensor<f64>            // 440 Hz
    %f440v = "dsp.getRangeOfVector"(%f440, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %cyc   = "dsp.mul"(%t, %f440v) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>  // cycles = 440*t
    %one1  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %ones  = "dsp.getRangeOfVector"(%one1, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %frac  = "dsp.modulo"(%cyc, %ones) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>  // frac(440*t) in [0,1)
    %saw2  = "dsp.add"(%frac, %frac) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>     // 2*frac
    %saw   = "dsp.sub"(%saw2, %ones) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>     // 2*frac - 1

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
    // alpha[n] = 0.02 + 0.33 * tri[n]  -> in [0.02, 0.35], as a per-sample tensor.
    %c033   = dsp.constant dense<3.300000e-01> : tensor<f64>   // span -> up to ~0.35 (bright)
    %c002   = dsp.constant dense<2.000000e-02> : tensor<f64>   // muffled floor (~140 Hz)
    %spanV  = "dsp.getRangeOfVector"(%c033, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %aminV  = "dsp.getRangeOfVector"(%c002, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<128xf64>
    %amodV  = "dsp.mul"(%triV, %spanV) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
    %alphaV = "dsp.add"(%amodV, %aminV) : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xf64>
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
}
