module {
  dsp.func @run() {
    %0 = dsp.constant dense<3.1415926535900001> : tensor<f64>
    %1 = dsp.constant dense<4.410000e+04> : tensor<f64>
    %2 = dsp.constant dense<4.400000e+02> : tensor<f64>
    %3 = dsp.constant dense<1.000000e+03> : tensor<f64>
    %4 = dsp.constant dense<1.010000e+02> : tensor<f64>
    %5 = dsp.constant dense<2.2675735999999999E-5> : tensor<f64>
    %6 = dsp.constant dense<0.000000e+00> : tensor<f64>
    %7 = dsp.constant dense<4.410000e+04> : tensor<f64>
    %8 = "dsp.getRangeOfVector"(%6, %7, %5) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %9 = dsp.constant dense<2.000000e+00> : tensor<f64>
    %10 = dsp.mul %9, %0 : (tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %11 = dsp.mul %10, %2 : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %12 = "dsp.gain"(%8, %11) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %13 = dsp.sin(%12 : tensor<*xf64>) to tensor<*xf64>
    %14 = dsp.constant dense<2.000000e+00> : tensor<f64>
    %15 = dsp.mul %14, %0 : (tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %16 = dsp.mul %15, %3 : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %17 = dsp.div %16, %1 : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %18 = "dsp.lowPassFIRFilter"(%17, %4) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %19 = dsp.hamming(%4 : tensor<f64>) to tensor<*xf64>
    %20 = dsp.mul %18, %19 : tensor<*xf64>
    %21 = "dsp.FIRFilterResponse"(%13, %20) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    dsp.print %21 : tensor<*xf64>
    dsp.return
  }
}
