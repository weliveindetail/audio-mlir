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
  memref.global "public" @wet : memref<f64> = dense<1.000000e+00>
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
  // The triangle sweep below is `@sample_offset mod @lfo_period`, so a smaller
  // period = faster sweep (LFO Hz = 44100 / period; default 147000 ≈ 0.3 Hz).
  // The host writes this from the +/- keys; must stay > 0 (host clamps it).
  memref.global "public" @lfo_period : memref<i64> = dense<147000>

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
    // alpha (the 1st-order IIR coefficient) is recomputed once per @run from a
    // ~0.3 Hz triangle over @sample_offset -- an "automated parameter" driven
    // entirely in-kernel, no host involvement. Read once and held for the block
    // (block-rate automation); N=128 (~2.9 ms) keeps the sweep step-free.
    // dsp.lowPassFilter carries --stream state (its previous output), so the
    // filtered tone is continuous across calls (no per-block click). Applied to
    // the sawtooth here (the signal path), so the sweep is audible as a timbral
    // change, not just amplitude. Triangle is pure arith (|2f-1| via cmp/select).
    %lpPmem = memref.get_global @lfo_period : memref<i64>  // interactive sweep speed
    %lpP    = memref.load %lpPmem[] : memref<i64>         // LFO period (samples)
    %lpPf   = arith.sitofp %lpP : i64 to f64
    %lpph   = arith.remsi %offval, %lpP : i64             // phase 0..P-1
    %lpphf  = arith.sitofp %lpph : i64 to f64
    %lpfrac = arith.divf %lpphf, %lpPf : f64              // 0..1
    %lp2    = arith.constant 2.000000e+00 : f64
    %lp1    = arith.constant 1.000000e+00 : f64
    %lp0    = arith.constant 0.000000e+00 : f64
    %lp2f   = arith.mulf %lpfrac, %lp2 : f64
    %lp2fm1 = arith.subf %lp2f, %lp1 : f64               // -1..1
    %lpneg  = arith.negf %lp2fm1 : f64
    %lpisn  = arith.cmpf olt, %lp2fm1, %lp0 : f64
    %lpabs  = arith.select %lpisn, %lpneg, %lp2fm1 : f64 // |2f-1|
    %lptri  = arith.subf %lp1, %lpabs : f64              // 0 at edges, 1 at center
    %lpamin = arith.constant 2.000000e-02 : f64          // muffled floor (~140 Hz)
    %lpaspn = arith.constant 3.300000e-01 : f64          // span -> up to ~0.35 (bright)
    %lpamod = arith.mulf %lptri, %lpaspn : f64
    %lpaf   = arith.addf %lpamod, %lpamin : f64          // alpha in [0.02, 0.35]
    %lpalpha = tensor.from_elements %lpaf : tensor<f64>
    %tone   = "dsp.lowPassFilter"(%saw, %lpalpha) : (tensor<128xf64>, tensor<f64>) -> tensor<128xf64>

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
}
