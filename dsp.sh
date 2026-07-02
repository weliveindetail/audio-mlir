#!/usr/bin/env bash
set -ex

PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

dsp1 dsp.mlir --emit=llvm 2> dsp.ll
llc dsp.ll -filetype=obj -o dsp.o
