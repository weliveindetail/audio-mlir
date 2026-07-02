def main() {
    # --- Parameters ---
    var pi = 3.14159265359; # pi constant
    var Fs = 44100; # sampling frequency (Hz)
    var freq = 440.0; # oscillator frequency (A4, Hz)
    var fc = 1000.0; # low-pass cut-off frequency (Hz)
    var N = 101; # FIR filter order (taps)

    # Division on a scalar produces an unranked tensor<*xf64>, which
    # getRangeOfVector silently rejects as its step argument.
    # var dt = 1 / Fs; # sample period (s)
    var dt = 0.000022675736; # sample period 1/Fs (s)

    # --- Oscillator: generate one block of a sine wave ---
    var time = getRangeOfVector(0, 44100, dt); # time vector for 1 s block
    var wt = 2 * pi * freq; # angular frequency (rad/s)
    var phase = gain(time, wt); # phase = 2*pi*f*t
    var osc = sin(phase); # sine oscillator output

    # --- Low-pass FIR filter design (windowed-sinc) ---
    var wc = 2 * pi * fc / Fs; # normalized cut-off, varies from 0 to pi
    var lpf = lowPassFIRFilter(wc, N); # ideal low-pass filter coefficients
    var lpf_w = lpf * hamming(N); # apply Hamming window

    # --- Apply filter to oscillator block ---
    var filtered = FIRFilterResponse(osc, lpf_w);

    print(filtered);
}
