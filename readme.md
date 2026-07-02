# audio-mlir

This is a research project to build an audio MLIR dialect on top of [DSP-MLIR](https://arxiv.org/abs/2408.11205). Please consider it a very early-stage hack right now.

## Build the DSP compiler

This will build the LLVM fork with the compiler. It's gonna take a while. Make sure to use sccache for development.
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

## Build the sample app

Now, use the `dsp1` compiler binary and your C++ host toolchain to build the sample app. I only tested on macOS so far:
```
> git clone https://github.com/weliveindetail/audio-mlir samples
> cd samples

> ./sample-macOS.sh
+ mkdir -p out
++ pwd
+ PATH='/Users/ez/Develop/DSP-MLIR/samples/../build-relwithdebinfo/bin:...'
+ dsp1 osc-low-pass.mlir --emit=llvm
+ llc out/osc-low-pass-native.ll -filetype=obj -o out/osc-low-pass-native.o
+ clang++ -O3 sample-macOS.cpp out/osc-low-pass-native.o -framework AudioToolbox -framework CoreAudio -o out/sample-macOS

> ./out/sample-macOS
Initializing macOS Core Audio Engine...

==============================================
   CoreAudio Low-Pass Filter Synth Running!   
==============================================
 -> Press [UP Arrow]   to raise the cut-off frequency
 -> Press [DOWN Arrow] to lower the cut-off frequency
 -> Press [Q] or Ctrl+C to stop the program safely
==============================================

Cut-off Frequency: 2100.0 Hz   

Stopping audio engine and cleaning up channels...
```

This plays a sawtooth oscillator through a low-pass filter. The filter's cut-off frequency can be adjusted for illustration. The entire audio processing is implemented in [osc-low-pass.mlir](osc-low-pass.mlir). [sample-macOS.cpp](sample-macOS.cpp) implements the native CoreAudio wrapper for macOS hosts.
