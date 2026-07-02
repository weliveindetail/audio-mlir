module {
  // Filter cut-off in Hz exposed as an external symbol
  memref.global "public" @cutoff : memref<f64> = dense<1.000000e+03>

  // Sawtooth oscillator with frequency parameter: saw(t) = 2*frac(t*freq) - 1
  // where `frac` is `phase mod 1`. Consider switching to phase when moving on
  // to block-based processing.
  dsp.func private @sawtooth(%freq: tensor<f64>) -> tensor<*xf64> {
    %zero = dsp.constant dense<0.000000e+00> : tensor<f64>
    %count = dsp.constant dense<4.410000e+04> : tensor<f64>
    %dt = dsp.constant dense<2.2675735999999999E-5> : tensor<f64>
    %t = "dsp.getRangeOfVector"(%zero, %count, %dt) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>

    // Broadcast freq to a full-length vector so the phase multiply is
    // vector*vector (1-D accesses). dsp.gain's scalar broadcast instead loads a
    // 0-D memref *inside* the loop, which makes affine loop fusion fail with an
    // assertion.
    %freqvec = "dsp.getRangeOfVector"(%freq, %count, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %phase = "dsp.mul"(%t, %freqvec) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %one = dsp.constant dense<1.000000e+00> : tensor<f64>
    %ones = "dsp.getRangeOfVector"(%one, %count, %zero) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %frac = "dsp.modulo"(%phase, %ones) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    // 2*frac as frac+frac (vector add) rather than dsp.gain by 2, same reason.
    %saw2 = "dsp.add"(%frac, %frac) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %saw = "dsp.sub"(%saw2, %ones) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    dsp.return %saw : tensor<*xf64>
  }

  // Windowed low-pass FIR with cut-off parameter: 
  // wc (rad/sample) = 2*pi*cutoff/Fs, an N-tap sinc, Hamming-windowed, then
  // convolved with the input. Linear convolution, so the result is
  // len(in) + N - 1 samples long.
  dsp.func private @lowpass(%in: tensor<*xf64>, %cutoff: tensor<f64>) -> tensor<*xf64> {
    %pi = dsp.constant dense<3.1415926535900001> : tensor<f64>
    %fs = dsp.constant dense<4.410000e+04> : tensor<f64>
    %N = dsp.constant dense<1.010000e+02> : tensor<f64>
    %two = dsp.constant dense<2.000000e+00> : tensor<f64>
    %twopi = dsp.mul %two, %pi : (tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %twopicut = dsp.mul %twopi, %cutoff : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %wc = dsp.div %twopicut, %fs : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %lp = "dsp.lowPassFIRFilter"(%wc, %N) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %win = dsp.hamming(%N : tensor<f64>) to tensor<*xf64>
    %coef = dsp.mul %lp, %win : tensor<*xf64>
    %y = "dsp.FIRFilterResponse"(%in, %coef) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    dsp.return %y : tensor<*xf64>
  }

  // Feed a 440 Hz sawtooth through a low-pass filter
  dsp.func @run(%out: memref<44200xf64>) attributes {llvm.emit_c_interface} {
    %freq = dsp.constant dense<4.400000e+02> : tensor<f64>
    %cutmem = memref.get_global @cutoff : memref<f64>
    %cutval = memref.load %cutmem[] : memref<f64>
    %cut = tensor.from_elements %cutval : tensor<f64>
    %osc = dsp.generic_call @sawtooth(%freq) : (tensor<f64>) -> tensor<*xf64>
    %y = dsp.generic_call @lowpass(%osc, %cut) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    dsp.return %y : tensor<*xf64>
  }
}
