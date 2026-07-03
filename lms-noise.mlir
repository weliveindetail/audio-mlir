// Adaptive noise canceller (Widrow ANC), the textbook LMS use case, done in
// pure affine/arith/math/memref -- no dsp dialect, no RNG primitive.
//
//   x[n]  = white noise           (LCG pseudo-random reference)
//   n0[n] = FIR(x) = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2]   (an "acoustic path")
//   d[n]  = sin(2*pi*440*t) + n0[n]                        (tone buried in noise)
//   y     = LMS(x -> d), a 32-tap adaptive FIR             (learns the path)
//   out   = d - wet*y                                      (noise removed)
//
// White noise is the well-conditioned ideal for LMS: convergence is fast and
// uniform, and the tone's leakage into the weight update spreads out as a low
// broadband hiss instead of the tonal sidebands you get when the interferer is
// a single sinusoid. The wet knob sweeps how much of the estimate to subtract
// (0 = tone + full hiss, 1 = tone revealed).
//
// The LCG is deterministic (fixed seed), so every rendered buffer is identical
// and the persistent weights converge to a stable solution.
module {
  memref.global "public" @mu : memref<f64> = dense<1.000000e-03>

  // Wet/dry mix for the noise estimate: out = d - wet*y (interactive knob).
  memref.global "public" @wet : memref<f64> = dense<1.000000e+00>

  // Noise-source color selector (interactive, left/right arrow):
  //   0 = white, 1 = pink (1/f), 2 = brown/red (1/f^2), 3 = Ornstein-Uhlenbeck.
  // These are the same four recurrences the dsp.noise_white/pink/brown/ou ops
  // encode (see ../DSP_MLIR .../LowerToAffineLoops.cpp and noise-kinds.mlir);
  // they are inlined here so the color can be picked at *render time* from this
  // global while the hand-tuned stateful LMS below is left untouched.
  memref.global "public" @noise_kind : memref<f64> = dense<0.000000e+00>

  // Persistent LMS weights (feedback state carried across renders).
  memref.global "public" @lms_weights : memref<32xf64> = dense<0.000000e+00>

  func.func @run(%out: memref<44100xf64>) attributes {llvm.emit_c_interface} {
    %z     = arith.constant 0.000000e+00 : f64
    %one   = arith.constant 1.000000e+00 : f64
    %dt    = arith.constant 2.2675736961451248E-5 : f64   // 1/44100 s
    %twopi = arith.constant 6.283185307179586 : f64
    %f440  = arith.constant 4.400000e+02 : f64
    %c0    = arith.constant 0 : index

    // Linear congruential PRNG (Numerical Recipes constants, modulus 2^32).
    %seed  = arith.constant 1 : i64
    %la    = arith.constant 1664525 : i64
    %lc    = arith.constant 1013904223 : i64
    %lmask = arith.constant 4294967295 : i64              // 2^32 - 1
    %nsc   = arith.constant 4.6566128730773926E-10 : f64  // 2 / 2^32

    // Noise-path FIR gains (the correlation the LMS has to discover).
    %g0 = arith.constant 7.000000e-01 : f64
    %g1 = arith.constant 5.000000e-01 : f64
    %g2 = arith.constant 3.000000e-01 : f64

    %t = memref.alloc() : memref<44100xf64>   // time base (seconds)
    %x = memref.alloc() : memref<44100xf64>   // white-noise reference
    %d = memref.alloc() : memref<44100xf64>   // desired = tone + colored noise
    %y = memref.alloc() : memref<44100xf64>   // adaptive noise estimate

    // t[i] = i*dt via the running-sum recurrence.
    affine.store %z, %t[%c0] : memref<44100xf64>
    %tlast = affine.for %i = 1 to 44100 iter_args(%acc = %z) -> (f64) {
      %v = arith.addf %acc, %dt : f64
      affine.store %v, %t[%i] : memref<44100xf64>
      affine.yield %v : f64
    }

    // Noise reference x[n]: one LCG white-noise stream, optionally colored by
    // pink / brown / OU recurrences, with the color chosen at render time from
    // the @noise_kind global. A single pass carries all recurrence state in
    // iter_args and stores the selected color; the LMS below sees only x[].
    %nkmem = memref.get_global @noise_kind : memref<f64>
    %nkval = memref.load %nkmem[] : memref<f64>
    %h0 = arith.constant 5.000000e-01 : f64   // kind < 0.5 -> white
    %h1 = arith.constant 1.500000e+00 : f64   // kind < 1.5 -> pink
    %h2 = arith.constant 2.500000e+00 : f64   // kind < 2.5 -> brown, else ou
    %isWhite = arith.cmpf olt, %nkval, %h0 : f64
    %isPink  = arith.cmpf olt, %nkval, %h1 : f64
    %isBrown = arith.cmpf olt, %nkval, %h2 : f64

    // Pink (Paul Kellet economy 3-pole), brown (leaky integrator) and OU
    // (mean-reverting one-pole) coefficients.
    %pk0a = arith.constant 0.99765 : f64
    %pk0g = arith.constant 0.0990460 : f64
    %pk1a = arith.constant 0.96300 : f64
    %pk1g = arith.constant 0.2965164 : f64
    %pk2a = arith.constant 0.57000 : f64
    %pk2g = arith.constant 1.0526913 : f64
    %pktail = arith.constant 0.1848 : f64
    %pkscale = arith.constant 2.000000e-01 : f64
    %bra = arith.constant 0.997 : f64
    %brg = arith.constant 2.000000e-02 : f64
    %brscale = arith.constant 4.000000e+00 : f64
    %oua = arith.constant 9.500000e-01 : f64
    %oug = arith.constant 5.000000e-02 : f64
    %ouscale = arith.constant 4.000000e+00 : f64

    %r:6 = affine.for %n = 0 to 44100
        iter_args(%s = %seed, %b0 = %z, %b1 = %z, %b2 = %z, %brs = %z, %ous = %z)
        -> (i64, f64, f64, f64, f64, f64) {
      // White: advance the LCG, map new state to [-1, 1).
      %m1 = arith.muli %s, %la : i64
      %a1 = arith.addi %m1, %lc : i64
      %u  = arith.andi %a1, %lmask : i64
      %uf = arith.uitofp %u : i64 to f64
      %sc = arith.mulf %uf, %nsc : f64
      %white = arith.subf %sc, %one : f64
      // Pink: three leaky poles + a direct term, scaled down.
      %b0d = arith.mulf %b0, %pk0a : f64
      %b0i = arith.mulf %white, %pk0g : f64
      %b0n = arith.addf %b0d, %b0i : f64
      %b1d = arith.mulf %b1, %pk1a : f64
      %b1i = arith.mulf %white, %pk1g : f64
      %b1n = arith.addf %b1d, %b1i : f64
      %b2d = arith.mulf %b2, %pk2a : f64
      %b2i = arith.mulf %white, %pk2g : f64
      %b2n = arith.addf %b2d, %b2i : f64
      %ps0 = arith.addf %b0n, %b1n : f64
      %ps1 = arith.addf %ps0, %b2n : f64
      %pt  = arith.mulf %white, %pktail : f64
      %ps2 = arith.addf %ps1, %pt : f64
      %pink = arith.mulf %ps2, %pkscale : f64
      // Brown/red: leaky integral of white.
      %brd = arith.mulf %brs, %bra : f64
      %bri = arith.mulf %white, %brg : f64
      %brn = arith.addf %brd, %bri : f64
      %brown = arith.mulf %brn, %brscale : f64
      // Ornstein-Uhlenbeck: mean-reverting one-pole low-pass.
      %oud = arith.mulf %ous, %oua : f64
      %oui = arith.mulf %white, %oug : f64
      %oun = arith.addf %oud, %oui : f64
      %ou  = arith.mulf %oun, %ouscale : f64
      // Pick the requested color for this render.
      %selBO = arith.select %isBrown, %brown, %ou : f64
      %selPBO = arith.select %isPink, %pink, %selBO : f64
      %xval = arith.select %isWhite, %white, %selPBO : f64
      affine.store %xval, %x[%n] : memref<44100xf64>
      affine.yield %u, %b0n, %b1n, %b2n, %brn, %oun : i64, f64, f64, f64, f64, f64
    }

    // d[n] = sin(2*pi*440*t) + 0.7*x[n]   (tap 0 of the noise path + the tone)
    affine.for %n = 0 to 44100 {
      %ti = affine.load %t[%n] : memref<44100xf64>
      %cyc = arith.mulf %ti, %f440 : f64
      %arg = arith.mulf %cyc, %twopi : f64
      %s440 = math.sin %arg : f64
      %xn = affine.load %x[%n] : memref<44100xf64>
      %n0 = arith.mulf %xn, %g0 : f64
      %di = arith.addf %s440, %n0 : f64
      affine.store %di, %d[%n] : memref<44100xf64>
    }
    // d[n] += 0.5*x[n-1]   (shifted loop range avoids a boundary guard)
    affine.for %n = 1 to 44100 {
      %xn1 = affine.load %x[%n - 1] : memref<44100xf64>
      %c1 = arith.mulf %xn1, %g1 : f64
      %dn = affine.load %d[%n] : memref<44100xf64>
      %ds = arith.addf %dn, %c1 : f64
      affine.store %ds, %d[%n] : memref<44100xf64>
    }
    // d[n] += 0.3*x[n-2]
    affine.for %n = 2 to 44100 {
      %xn2 = affine.load %x[%n - 2] : memref<44100xf64>
      %c2 = arith.mulf %xn2, %g2 : f64
      %dn = affine.load %d[%n] : memref<44100xf64>
      %ds = arith.addf %dn, %c2 : f64
      affine.store %ds, %d[%n] : memref<44100xf64>
    }

    // LMS adaptive FIR, 32 taps, weights persist across renders.
    %mumem = memref.get_global @mu : memref<f64>
    %muval = memref.load %mumem[] : memref<f64>
    %w = memref.get_global @lms_weights : memref<32xf64>

    affine.for %n = 0 to 44100 {
      affine.store %z, %y[%n] : memref<44100xf64>
      // y[n] = sum_i w[i] * x[n-i]
      affine.for %i = 0 to 32 {
        affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%n, %i) {
          %xni = affine.load %x[%n - %i] : memref<44100xf64>
          %wi  = affine.load %w[%i] : memref<32xf64>
          %p   = arith.mulf %xni, %wi : f64
          %yac = affine.load %y[%n] : memref<44100xf64>
          %yn  = arith.addf %p, %yac : f64
          affine.store %yn, %y[%n] : memref<44100xf64>
        }
      }
      // e[n] = d[n] - y[n]
      %dn = affine.load %d[%n] : memref<44100xf64>
      %yn = affine.load %y[%n] : memref<44100xf64>
      %e  = arith.subf %dn, %yn : f64
      // w[i] += mu * e[n] * x[n-i]
      affine.for %i = 0 to 32 {
        affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%n, %i) {
          %xni = affine.load %x[%n - %i] : memref<44100xf64>
          %wi  = affine.load %w[%i] : memref<32xf64>
          %ex  = arith.mulf %e, %xni : f64
          %mex = arith.mulf %muval, %ex : f64
          %wn  = arith.addf %wi, %mex : f64
          affine.store %wn, %w[%i] : memref<32xf64>
        }
      }
    }

    // out[n] = d[n] - wet*y[n]   (wet is the noise-reduction knob)
    %wetmem = memref.get_global @wet : memref<f64>
    %wetval = memref.load %wetmem[] : memref<f64>
    affine.for %n = 0 to 44100 {
      %dn = affine.load %d[%n] : memref<44100xf64>
      %yn = affine.load %y[%n] : memref<44100xf64>
      %wy = arith.mulf %wetval, %yn : f64
      %o  = arith.subf %dn, %wy : f64
      affine.store %o, %out[%n] : memref<44100xf64>
    }

    memref.dealloc %t : memref<44100xf64>
    memref.dealloc %x : memref<44100xf64>
    memref.dealloc %d : memref<44100xf64>
    memref.dealloc %y : memref<44100xf64>
    return
  }
}
