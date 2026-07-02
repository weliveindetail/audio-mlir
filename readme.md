# audio-mlir

This is a research project to build an audio MLIR dialect on top of [DSP-MLIR](https://arxiv.org/abs/2408.11205).
Please consider it a very early-stage hack right now.

You can try an early live demo in your browser here:
https://weliveindetail.github.io/audio-mlir/sample-wasm/

## Build the DSP compiler

This will build the LLVM fork with the compiler. It's gonna take a while.
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

## Build the sample

Right now, this repo only contains one sample app.
The audio-mlir dialect itself is (still) implemented in-tree in LLVM above.
The sample plays a sawtooth oscillator through a low-pass filter.
The filter's cut-off frequency can be adjusted for illustration.
The entire audio pipeline is defined in the target-agnostic [osc-low-pass.mlir](osc-low-pass.mlir) file.
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
+ dsp1 osc-low-pass.mlir --emit=wasm -o out/osc-low-pass.wasm
+ wasm-ld --export=_mlir_ciface_run --export=cutoff ... out/osc-low-pass.wasm -o sample-wasm/osc-low-pass.linked.wasm
+ set +x
Sample is serving at http://localhost:8000/
Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```

### C++ sample app

The `dsp1` compiler can emit a native object as well.
[sample-macOS.cpp](sample-macOS.cpp) implements the native CoreAudio wrapper for macOS hosts.
Use your C++ host toolchain to build it. I only tested on macOS so far:
```
> ./sample-macOS.sh
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
