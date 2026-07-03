// Standalone smoke test for the dsp-dialect noise-signal generators
// (dsp.noise_white / noise_pink / noise_brown / noise_ou). Each op takes
// (N, sigma, seed) and produces a length-N vector of samples. Run with:
//
//   dsp1 noise-kinds.mlir --emit=jit
//
// and eyeball the four printed rows: white is broadband/uncorrelated, pink and
// brown are increasingly smooth (low-frequency-weighted), ou hovers around 0.
module {
  dsp.func @main() {
    %n     = dsp.constant dense<1.600000e+01> : tensor<f64>   // 16 samples
    %sigma = dsp.constant dense<1.000000e+00> : tensor<f64>
    %seed  = dsp.constant dense<1.000000e+00> : tensor<f64>

    %white = "dsp.noise_white"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %pink  = "dsp.noise_pink"(%n, %sigma, %seed)  : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %brown = "dsp.noise_brown"(%n, %sigma, %seed) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %ou    = "dsp.noise_ou"(%n, %sigma, %seed)    : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>

    dsp.print %white : tensor<*xf64>
    dsp.print %pink  : tensor<*xf64>
    dsp.print %brown : tensor<*xf64>
    dsp.print %ou    : tensor<*xf64>
    dsp.return
  }
}
