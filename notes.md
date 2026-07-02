# Research Notes

## Batch-and-stream C++ implementation

- f64→f32 on copy: AudioCallback reads double samples from the published batch and casts to
float per-sample into the CoreAudio buffers.
- Batch-and-stream: regenerate() renders a full 1-second batch into the back buffer via the
kernel, then atomically swaps it in. The audio thread streams from gActive with
wraparound; the keyboard thread triggers a regenerate on each arrow press.
- Thread safety: single std::atomic<double*> pointer swap (acq/rel) publishes a
fully-rendered buffer; no torn reads.

## Moving to block-based processing later

The current design recomputes a whole second of audio on every cutoff change and loops it.
Block-based means the kernel produces exactly inNumberFrames of fresh audio per audio
callback, with continuity across calls. That requires carrying state the current run(out,
fc) kernel doesn't have:

1. Oscillator phase continuity. Right now getRangeOfVector(0, 44100, dt) always starts at
t=0. Across blocks the sine must not restart, or you get a click every block. The kernel
needs a phase (or sample-index) input/output: take phase_in, generate n samples starting
there, and write back phase_out. In the DSL that's a start offset added to the time vector;
at the ABI level it's an extra double* in/out parameter (or a returned scalar).

2. FIR filter state (the hard part). A 101-tap FIR convolution needs the previous N−1 = 100
input samples to compute the first outputs of the new block — otherwise each block's
leading 100 samples are wrong and you hear edge artifacts at every boundary. The standard
fix is overlap-save: keep a 100-sample history of the oscillator signal, prepend it to the
new block, convolve, and discard the transient. So the kernel needs a second caller-owned
state buffer (hist[100]) that it reads at entry and updates at exit.

3. Cutoff changes mid-stream. Recomputing the windowed-sinc coefficients every block
(cheap, 101 taps) is fine, but changing coefficients instantaneously causes a small
discontinuity. Acceptable for a demo; a crossfade between old/new coefficient sets removes
it.

4. Kernel signature growth. run(out, fc) becomes roughly:
run(out, n, fc, phase_in_out, hist_in_out)
i.e. the block length n, plus two extra caller-owned state buffers the kernel
reads-then-writes. On the compiler side this is the same out-param/destination-passing
machinery you're already building — just more memref arguments threaded through
FuncOpLowering, no new mechanism.

5. C++ side. AudioCallback calls run(...) directly with out = ioData (after an f64 scratch
buffer + convert, since CoreAudio wants f32), n = inNumberFrames, and the persistent
gPhase/gHist state. The double-buffer/batch scaffolding goes away; the callback does real
work, so the kernel must stay within the real-time budget (101-tap FIR × ~512 frames is
trivially fine).

The biggest lift is #2 — the FIR history buffer is what makes blocks seamless, and it's the
one piece the static DSL pipeline has no concept of today.
