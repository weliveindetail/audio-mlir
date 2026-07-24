# audio-mlir

This is a research project to build an audio MLIR dialect on top of [DSP-MLIR](https://arxiv.org/abs/2408.11205).
Please consider it a very early-stage hack right now.

You can try an early live demo in your browser here:
https://weliveindetail.github.io/audio-mlir/sample-wasm/

## Build the DSP compiler

The audio-mlir dialect itself is (still) implemented in the [DSP-MLIR LLVM fork](https://github.com/weliveindetail/dsp-mlir/).
Building it will take some time.
Make sure to use sccache for development.
```
> git clone https://github.com/weliveindetail/dsp-mlir
> cmake -GNinja -Bbuild-relwithdebinfo -Sdsp-mlir/llvm \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
        -DLLVM_TARGETS_TO_BUILD="host;WebAssembly" \
        -DLLVM_BUILD_EXAMPLES=On \
        -DLLVM_ENABLE_ASSERTIONS=On
> ninja -C build-relwithdebinfo dsp1
```

## Sample kernel

Right now, this repo only contains one sample kernel, which contains a few representative audio components:
a swept-filtered polyphonic sawtooth buried in broadband noise,
and an 32-tap adaptive LMS noise canceller to reveal the tone.
The noise color (white/pink/brown/ou/none) is selectable at run time.
The shape of the cutoff is shared across all voices and can be customized in real-time, while sweep and tone is specific for each voice. What is still missing at this point is a multi-channel/stereo feature, but conceptionuall this is "just" one more dimension in the tensor representation. The entire audio pipeline is defined in the target-agnostic [lms-noise.mlir](lms-noise.mlir) file.

This setup exercises, in kernel order:
* sample-accurate MIDI note events triggering a polyphonic bank of sawtooth oscillators
* a click-free one-pole gate envelope shaping each voice's amplitude
* a per-voice swept low-pass filter whose cutoff is modulated by a shared wavetable LFO ...
* read at each voice's own trigger-anchored phase
* a mixer summing the voices into a tone
* a runtime-selectable colored-noise source (white/pink/brown/ou/none) ...
* shaped through delay lines into an acoustic path
* a second mixer burying the tone in that noise
* a 32-tap LMS adaptive filter that learns the noise path
* a final wet/dry mix that subtracts the estimate to reveal the tone

Before we build that, let's checkout the repo:
```
> git clone https://github.com/weliveindetail/audio-mlir samples
> cd samples
```

### WebAssembly

Once the LLVM build finished, we can use `dsp1` to compile our audio pipeline into a WebAssembly module.
[This sample HTML file](sample-wasm/index.html) illustrates how to load and use it. You can try the [live demo in your browser here](https://weliveindetail.github.io/audio-mlir/sample-wasm/).
```
> ./sample-wasm.sh 
+ dsp1 lms-noise.mlir --stream --opt --emit=wasm -o out/lms-noise.wasm
+ wasm-ld --no-entry --import-memory --allow-undefined --export=run --export=_mlir_ciface_run --export=mu --export=wet --export=noise_kind --export=lfo_period ... out/lms-noise.wasm -o sample-wasm/lms-noise.linked.wasm
+ set +x
Sample is serving at http://localhost:8000/
Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```

### C++ sample app

The `dsp1` compiler can emit a native object as well.
[lms-noise-macOS.cpp](lms-noise-macOS.cpp) implements a native CoreAudio driver for macOS hosts.
Use your C++ host toolchain to build it:
```
> ./lms-noise-macOS.sh
+ dsp1 lms-noise.mlir --stream --emit=llvm --opt -o out/lms-noise-native.ll
+ llc out/lms-noise-native.ll -filetype=obj -o out/lms-noise-native.o
+ clang++ -O3 lms-noise-macOS.cpp out/lms-noise-native.o -framework AudioToolbox -framework CoreAudio -framework CoreMIDI -framework CoreFoundation -o out/lms-noise-macOS
Built out/lms-noise-macOS -- run it to hear the demo.

> ./out/lms-noise-macOS
Initializing macOS Core Audio Engine...
 -> MIDI: no source found -- computer keyboard emulates one
 -> [A W S E D F T G Y H U J K] one-octave piano (C4..C5) -- press to start, press again to stop
 -> [UP/DOWN]     more / less cancellation (toward a clean tone)
 -> [LEFT/RIGHT]  cycle noise color (white/pink/brown/ou/none)
 -> [+ / -]       cutoff-LFO speed (faster / slower, in Hz)
 -> [Q] or Ctrl+C to stop the program safely
==============================================

Reduce  | Color  | Sweep    | Voices
100%    | white  | 5.51 Hz  | 0/8   
```
