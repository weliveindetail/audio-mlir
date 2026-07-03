#!/usr/bin/env bash
set -ex
mkdir -p out

# DSP-MLIR kernel
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH
dsp1 osc-low-pass.mlir --emit=llvm --opt -o out/osc-low-pass-native.ll
llc out/osc-low-pass-native.ll -filetype=obj -o out/osc-low-pass-native.o
llvm-objdump --disassemble-symbols=_run out/osc-low-pass-native.o

# Link with CoreAudio host
clang++ -O3 sample-macOS.cpp out/osc-low-pass-native.o -framework AudioToolbox -framework CoreAudio -o out/sample-macOS
