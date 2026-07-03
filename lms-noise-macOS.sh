#!/usr/bin/env bash
# Build + run the interactive CoreAudio LMS adaptive-noise-canceller demo:
# a 440 Hz tone buried in broadband noise, with the noise adaptively removed.
#
# Usage: ./lms-noise-macOS.sh [kernel.mlir]   (default: lms-noise.mlir)
# Pass OPT=0 to disable --opt (default builds with --opt).
#
# Uses the dedicated host lms-noise-macOS.cpp: streams @run's output, drives the
# @wet knob with Up/Down, and cycles the @noise_kind color (white/pink/brown/ou)
# with Left/Right. The noise reference is generated in-kernel from an LCG stream
# (optionally colored), so no RNG primitive or host-fed data is needed.
set -ex
mkdir -p out

KERNEL=${1:-lms-noise.mlir}
STEM=out/$(basename "${KERNEL%.mlir}")

# DSP-MLIR kernel
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

OPT_FLAG=
if [ "${OPT:-1}" != "0" ]; then OPT_FLAG=--opt; fi
dsp1 "$KERNEL" --emit=llvm $OPT_FLAG -o "$STEM-native.ll"
llc "$STEM-native.ll" -filetype=obj -o "$STEM-native.o"

# Link with the dedicated CoreAudio host (Up/Down = wet, Left/Right = noise color)
clang++ -O3 lms-noise-macOS.cpp "$STEM-native.o" -framework AudioToolbox -framework CoreAudio -o "$STEM-macOS"

echo "Built $STEM-macOS -- run it to hear the demo."
