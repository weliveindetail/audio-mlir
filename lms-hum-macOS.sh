#!/usr/bin/env bash
# Build + run the interactive CoreAudio LMS hum-canceller demo.
#
# Usage: ./lms-hum-macOS.sh [kernel.mlir]   (default: lms-hum-live.mlir)
# Pass OPT=0 to disable --opt (default builds with --opt).
#
# The demo defaults to lms-hum-live.mlir, whose LMS weights persist across
# renders (a "public" global memref). That lets a small step size converge over
# several one-second buffers, which keeps the tone undistorted so the looped
# buffer joins seamlessly -- unlike the from-scratch kernels, whose per-buffer
# misadjustment residual clicks at the loop boundary.
set -ex
mkdir -p out

KERNEL=${1:-lms-hum-live.mlir}
STEM=out/$(basename "${KERNEL%.mlir}")

# DSP-MLIR kernel
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

OPT_FLAG=
if [ "${OPT:-1}" != "0" ]; then OPT_FLAG=--opt; fi
dsp1 "$KERNEL" --emit=llvm $OPT_FLAG -o "$STEM-native.ll"
llc "$STEM-native.ll" -filetype=obj -o "$STEM-native.o"

# Link with CoreAudio host
clang++ -O3 lms-hum-macOS.cpp "$STEM-native.o" -framework AudioToolbox -framework CoreAudio -o "$STEM-macOS"

echo "Built $STEM-macOS -- run it to hear the demo."
