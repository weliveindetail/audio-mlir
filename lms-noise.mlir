// Adaptive noise canceller (Widrow ANC) expressed ENTIRELY in the dsp dialect
// -- no hand-written affine, everything is tensor-valued and lowers through the
// standard bufferization. This is the tensor-based counterpart of the earlier
// spelled-out affine kernel.
//
//   x[n]  = white noise            (dsp.noise_white, an LCG stream)
//   n0[n] = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2]   (an "acoustic path", via delay)
//   d[n]  = sin(2*pi*440*t) + n0[n]              (tone buried in noise)
//   y     = LMS(x -> d), a 32-tap adaptive FIR   (dsp.lmsFilterResponse)
//   out   = d - y                                (noise removed, tone revealed)
//
// Cross-call persistence (block streaming) -- compile with --stream:
// the StreamStateMaterialization pass gives each stateful op its own auto,
// uniquely-named module-scope state global, so successive _mlir_ciface_run
// calls continue ONE uninterrupted stream instead of restarting each block:
//   * dsp.noise_white     -> memref<6xf64>  LCG + colored-filter state
//   * dsp.delay (x2)      -> memref<Kxf64>  the delay line (last K inputs)
//   * dsp.lmsFilterResponse -> memref<32xf64> the adaptive weights (converge)
// None of this state is threaded through the @run signature.
//
// Interactive knobs:
//   * @mu  (LMS learning rate) -- the LMS lowering loads it once, outside the
//     per-sample loop, so it stays runtime-tunable AND loop-fusion-safe.
//   * @wet (noise-cancellation depth) -- the terminal mix does out = d - wet*y
//     via dsp.gain
// Runtime noise-color select is NOT wired: the dsp.noise_* ops fix their color at
// compile time (by op identity), so the color is white here. Making it runtime
// would need a new noise op that selects the coloring from a scalar.
module {
  // LMS adaptation rate (interactive, read at render time inside the kernel).
  memref.global "public" @mu : memref<f64> = dense<1.000000e-03>
  // Wet mix (interactive): fraction of the estimated noise to subtract.
  // 0.0 = noise left in, 1.0 = full cancellation.
  memref.global "public" @wet : memref<f64> = dense<1.000000e+00>

  dsp.func @run(%out: memref<44100xf64>) attributes {llvm.emit_c_interface} {
    // --- white-noise reference x[n] (persistent LCG stream) ---
    %seed  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %sigma = dsp.constant dense<1.000000e+00> : tensor<f64>
    %n     = dsp.constant dense<4.410000e+04> : tensor<f64>
    %x = "dsp.noise_white"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>

    // --- acoustic path n0[n] = 0.7 x[n] + 0.5 x[n-1] + 0.3 x[n-2] ---
    // Delays carry their tail across calls (persistent), so the path is correct
    // at block boundaries. Constant gains are broadcast as constant vectors
    // (getRangeOfVector with step 0) and applied with vector*vector muls, which
    // keeps every access 1-D and loop-fusion-friendly.
    %zero = dsp.constant dense<0.000000e+00> : tensor<f64>
    %cnt  = dsp.constant dense<4.410000e+04> : tensor<f64>
    %d1c  = dsp.constant dense<1.000000e+00> : tensor<f64>
    %d2c  = dsp.constant dense<2.000000e+00> : tensor<f64>
    %x1 = "dsp.delay"(%x, %d1c) : (tensor<44100xf64>, tensor<f64>) -> tensor<44100xf64>
    %x2 = "dsp.delay"(%x, %d2c) : (tensor<44100xf64>, tensor<f64>) -> tensor<44100xf64>

    %c07 = dsp.constant dense<7.000000e-01> : tensor<f64>
    %c05 = dsp.constant dense<5.000000e-01> : tensor<f64>
    %c03 = dsp.constant dense<3.000000e-01> : tensor<f64>
    %g07 = "dsp.getRangeOfVector"(%c07, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %g05 = "dsp.getRangeOfVector"(%c05, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %g03 = "dsp.getRangeOfVector"(%c03, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %p0 = "dsp.mul"(%x, %g07)  : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %p1 = "dsp.mul"(%x1, %g05) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %p2 = "dsp.mul"(%x2, %g03) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %n01 = "dsp.add"(%p0, %p1) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %n0  = "dsp.add"(%n01, %p2) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>

    // --- 440 Hz tone: sin(2*pi*440 * t), t[n] = n*dt ---
    %dt    = dsp.constant dense<2.2675736961451248E-5> : tensor<f64>   // 1/44100
    %t     = "dsp.getRangeOfVector"(%zero, %cnt, %dt) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %w440  = dsp.constant dense<2.764601535159018E+3> : tensor<f64>    // 2*pi*440
    %w440v = "dsp.getRangeOfVector"(%w440, %cnt, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>
    %arg   = "dsp.mul"(%t, %w440v) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    %tone  = dsp.sin(%arg : tensor<44100xf64>) to tensor<44100xf64>

    // --- desired d[n] = tone + colored noise ---
    %d = "dsp.add"(%tone, %n0) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>

    // --- 32-tap LMS: learn the noise path, weights persist across calls ---
    %mumem = memref.get_global @mu : memref<f64>
    %muval = memref.load %mumem[] : memref<f64>
    %mu    = tensor.from_elements %muval : tensor<f64>
    %flen  = dsp.constant dense<3.200000e+01> : tensor<f64>
    %y = "dsp.lmsFilterResponse"(%x, %d, %mu, %flen) : (tensor<44100xf64>, tensor<44100xf64>, tensor<f64>, tensor<f64>) -> tensor<44100xf64>

    // --- out = d - wet*y  (interactive wet mix; wet=1 -> full cancellation) ---
    %wetmem = memref.get_global @wet : memref<f64>
    %wetval = memref.load %wetmem[] : memref<f64>
    %wet    = tensor.from_elements %wetval : tensor<f64>
    %wy     = "dsp.gain"(%y, %wet) : (tensor<44100xf64>, tensor<f64>) -> tensor<44100xf64>
    %outt   = "dsp.sub"(%d, %wy) : (tensor<44100xf64>, tensor<44100xf64>) -> tensor<44100xf64>
    dsp.return %outt : tensor<44100xf64>
  }
}
