module {
  // LMS step size: larger mu = faster hum cancellation but risks divergence
  memref.global "public" @mu : memref<f64> = dense<1.000000e-02>

  // Adaptive 60 Hz hum canceller (Widrow's classic LMS noise-cancellation demo).
  //
  //   d[n] = sin(2*pi*440*t) + 0.7*sin(2*pi*60*t)   desired: tone + mains hum
  //   x[n] =                       sin(2*pi*60*t)    reference correlated w/ hum
  //   y    = lmsFilterResponse(x, d, mu, L)          adapts to reproduce the hum
  //   out  = d - y                                   ~ clean 440 Hz tone
  //
  // The 60 Hz interferer is a deterministic sinusoid (the DSL has no RNG), which
  // is exactly the mains-hum scenario LMS was first used for. As the weights
  // converge over the 1-second buffer you hear the buzz fade out of the tone.
  dsp.func @run(%out: memref<44100xf64>) attributes {llvm.emit_c_interface} {
    %zero  = dsp.constant dense<0.000000e+00> : tensor<f64>
    %count = dsp.constant dense<4.410000e+04> : tensor<f64>       // N = 44100
    %dt    = dsp.constant dense<2.2675735999999999E-5> : tensor<f64> // 1/44100 s

    // Time base (seconds): t[i] = i*dt.
    %t = "dsp.getRangeOfVector"(%zero, %count, %dt) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>

    // Broadcast scalars to full-length vectors so every multiply is
    // vector*vector (1-D accesses) -- the scalar-broadcast form (dsp.gain) loads
    // a 0-D memref inside the loop, which the affine loop-fusion pass (--opt)
    // rejects. Same trick as osc-low-pass.
    %twopi  = dsp.constant dense<6.2831853071800001> : tensor<f64>
    %twopiv = "dsp.getRangeOfVector"(%twopi, %count, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>

    // 440 Hz tone: sin(2*pi*440*t).
    %f440  = dsp.constant dense<4.400000e+02> : tensor<f64>
    %f440v = "dsp.getRangeOfVector"(%f440, %count, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %cyc440 = "dsp.mul"(%t, %f440v) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %arg440 = "dsp.mul"(%cyc440, %twopiv) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %s440 = dsp.sin(%arg440 : tensor<*xf64>) to tensor<*xf64>

    // 60 Hz hum reference: sin(2*pi*60*t).
    %f60  = dsp.constant dense<6.000000e+01> : tensor<f64>
    %f60v = "dsp.getRangeOfVector"(%f60, %count, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %cyc60 = "dsp.mul"(%t, %f60v) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %arg60 = "dsp.mul"(%cyc60, %twopiv) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %x = dsp.sin(%arg60 : tensor<*xf64>) to tensor<*xf64>

    // Hum injected into the desired signal: 0.7 * reference.
    %amp  = dsp.constant dense<7.000000e-01> : tensor<f64>
    %ampv = "dsp.getRangeOfVector"(%amp, %count, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %hum = "dsp.mul"(%x, %ampv) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %d = "dsp.add"(%s440, %hum) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    // Adaptive filter: predict the hum in d from reference x.
    %mumem = memref.get_global @mu : memref<f64>
    %muval = memref.load %mumem[] : memref<f64>
    %muT = tensor.from_elements %muval : tensor<f64>
    %L = dsp.constant dense<3.200000e+01> : tensor<f64>            // 32 taps
    %y = "dsp.lmsFilterResponse"(%x, %d, %muT, %L) : (tensor<*xf64>, tensor<*xf64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>

    // Cleaned output: subtract the adaptive hum estimate.
    %clean = "dsp.sub"(%d, %y) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    dsp.return %clean : tensor<*xf64>
  }
}
