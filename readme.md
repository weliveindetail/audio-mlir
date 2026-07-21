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
The audio-mlir dialect itself is (still) implemented in-tree in the LLVM fork above.
The sample is an LMS adaptive noise canceller: a swept-filtered sawtooth tone is
buried in broadband noise, and a 32-tap adaptive FIR removes the noise to reveal
the tone. The noise color (white/pink/brown/ou/none) is selectable at run time.
The entire audio pipeline is defined in the target-agnostic [lms-noise.mlir](lms-noise.mlir) file.
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
+ clang++ -O3 lms-noise-macOS.cpp out/lms-noise-native.o -framework AudioToolbox -framework CoreAudio -o out/lms-noise-macOS
Built out/lms-noise-macOS -- run it to hear the demo.

> ./out/lms-noise-macOS
 -> Up / Down arrows    to adjust @wet (how much noise to remove)
 -> Left / Right arrows to cycle @noise_kind (white/pink/brown/ou/none)
 -> + / - keys          to change @lfo_period (cutoff-sweep speed)
 -> Press [Q] or Ctrl+C to stop the program safely
```
