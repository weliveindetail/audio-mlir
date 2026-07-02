module {
  dsp.func @run(%out: memref<44200xf64>, %cutoff: f64) attributes {llvm.emit_c_interface} {
    %0 = dsp.constant dense<3.1415926535900001> : tensor<f64>
    %1 = dsp.constant dense<4.410000e+04> : tensor<f64>
    %2 = dsp.constant dense<4.400000e+02> : tensor<f64>
    %3 = tensor.from_elements %cutoff : tensor<f64>
    %4 = dsp.constant dense<1.010000e+02> : tensor<f64>
    %5 = dsp.constant dense<2.2675735999999999E-5> : tensor<f64>
    %6 = dsp.constant dense<0.000000e+00> : tensor<f64>
    %7 = dsp.constant dense<4.410000e+04> : tensor<f64>
    %8 = "dsp.getRangeOfVector"(%6, %7, %5) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    // Sawtooth oscillator: saw(t) = 2*frac(t*freq) - 1, a rising ramp in [-1,1)
    // that resets every 1/freq seconds. `frac` is `phase mod 1` (arith.remf);
    // since phase = t*freq is non-negative here, the result lands in [0,1).
    // The divisor/`-1` term is a length-matched ones vector built cheaply with
    // getRangeOfVector(first=1, step=0) so we avoid a large dense constant
    // (which ConstantOp would unroll into one store per element).
    %phase = "dsp.gain"(%8, %2) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %one = dsp.constant dense<1.000000e+00> : tensor<f64>
    %ones = "dsp.getRangeOfVector"(%one, %7, %6) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %frac = "dsp.modulo"(%phase, %ones) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %two = dsp.constant dense<2.000000e+00> : tensor<f64>
    %saw2 = "dsp.gain"(%frac, %two) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %13 = "dsp.sub"(%saw2, %ones) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %14 = dsp.constant dense<2.000000e+00> : tensor<f64>
    %15 = dsp.mul %14, %0 : (tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %16 = dsp.mul %15, %3 : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %17 = dsp.div %16, %1 : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %18 = "dsp.lowPassFIRFilter"(%17, %4) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %19 = dsp.hamming(%4 : tensor<f64>) to tensor<*xf64>
    %20 = dsp.mul %18, %19 : tensor<*xf64>
    %21 = "dsp.FIRFilterResponse"(%13, %20) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    dsp.return %21 : tensor<*xf64>
  }
}
