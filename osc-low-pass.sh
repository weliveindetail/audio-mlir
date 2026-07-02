#!/usr/bin/env bash
set -ex

# Build the DSP-MLIR kernel object first (dsp.mlir -> dsp.ll -> dsp.o).
./dsp.sh

# Link the CoreAudio host against the compiled kernel.
clang++ -O3 osc-low-pass.cpp dsp.o -framework AudioToolbox -framework CoreAudio -o osc-low-pass
