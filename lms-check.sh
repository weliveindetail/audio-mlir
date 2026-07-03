#!/usr/bin/env bash
# Build an LMS hum-canceller kernel + the correctness/quality driver, then run.
#
# Usage: ./lms-check.sh [kernel.mlir]   (default: lms-hum.mlir)
# Pass OPT=0 to disable --opt (default builds with --opt).
#
# Run it on both kernels and diff the LMS_JSON line: the `checksum` must match
# between lms-hum.mlir (built-in dsp.lmsFilterResponse) and lms-hum-pure.mlir
# (hand-written affine LMS) -- that is the correctness check.
set -e
mkdir -p out

KERNEL=${1:-lms-hum.mlir}
STEM=out/$(basename "${KERNEL%.mlir}")

PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

OPT_FLAG=
if [ "${OPT:-1}" != "0" ]; then OPT_FLAG=--opt; fi
dsp1 "$KERNEL" --emit=llvm $OPT_FLAG 2> "$STEM.ll"
llc "$STEM.ll" -filetype=obj -o "$STEM.o"

clang++ -O3 -std=c++17 lms-check.cpp "$STEM.o" -o "$STEM-check"
"$STEM-check"
