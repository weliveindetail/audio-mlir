#!/usr/bin/env bash
# Build the DSP-MLIR kernel object and the micro-benchmark, then run it.
# Any args (e.g. --iterations 5000 --warmup 500) are forwarded to the binary.
#
# The benchmark links the same kernel object as sample-macOS. To A/B a compiler
# change, edit the pipeline / passes, rerun this script, and diff the BENCH_JSON
# line (min_ms / median_ms lower is better; checksum must stay stable).
set -e
mkdir -p out

PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

# DSP-MLIR kernel -> LLVM IR -> object. Pass OPT=0 to build the plain baseline
# pipeline; default enables --opt (affine loop fusion + scalar replacement),
# which is now safe (a legality pass guards the fusion crash).
OPT_FLAG=${OPT:+}
if [ "${OPT:-1}" != "0" ]; then OPT_FLAG=--opt; fi
dsp1 osc-low-pass.mlir --emit=llvm $OPT_FLAG 2> out/osc-low-pass-native.ll
llc out/osc-low-pass-native.ll -filetype=obj -o out/osc-low-pass-native.o

# Benchmark driver (no CoreAudio; portable).
clang++ -O3 -std=c++17 bench.cpp out/osc-low-pass-native.o -o out/bench

./out/bench "$@"
