#!/usr/bin/env bash
# Build + run the non-interactive correctness harness for lms-noise.mlir.
#
# Usage: ./lms-noise-check.sh [kernel.mlir]   (default: lms-noise.mlir)
# Pass OPT=0 to disable --opt (default builds with --opt).
#
# Compiles the kernel exactly like the CoreAudio host (--stream [--opt]) but
# links the headless checker lms-noise-check.cpp instead of the audio driver.
# The checker runs a fixed interaction script and asserts the output behaviour;
# it exits non-zero (and prints FAIL lines) if any check fails.
set -ex
mkdir -p out

KERNEL=${1:-lms-noise.mlir}
STEM=out/$(basename "${KERNEL%.mlir}")

# DSP-MLIR compiler
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

OPT_FLAG=--opt
if [ "${OPT:-1}" = "0" ]; then OPT_FLAG=; fi

dsp1 "$KERNEL" --stream --emit=llvm $OPT_FLAG -o "$STEM-check.ll"
llc "$STEM-check.ll" -filetype=obj -o "$STEM-check.o"
clang++ -O3 -std=c++17 lms-noise-check.cpp "$STEM-check.o" -o "$STEM-check"

set +x
"$STEM-check"
