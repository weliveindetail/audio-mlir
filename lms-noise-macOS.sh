#!/usr/bin/env bash
# Build + run the interactive CoreAudio LMS adaptive-noise-canceller demo:
# a 440 Hz tone buried in broadband noise, with the noise adaptively removed.
#
# Usage: ./lms-noise-macOS.sh [kernel.mlir]   (default: lms-noise.mlir)
# OPT defaults to 0: the runtime @wet mix uses dsp.gain, whose per-sample 0-D
# load is rejected by the affine-fusion pass under --opt. Pass OPT=1 only for a
# kernel with no runtime scalar gain.
#
# Uses the dedicated host lms-noise-macOS.cpp: streams @run's output and drives
# the @wet knob with Up/Down. (The @noise_kind color knob is disabled -- the
# dsp.noise_* ops fix their color at compile time, so the kernel is white-only.)
# The noise reference is generated in-kernel from an LCG stream, so no RNG
# primitive or host-fed data is needed.
set -ex
mkdir -p out

KERNEL=${1:-lms-noise.mlir}
STEM=out/$(basename "${KERNEL%.mlir}")

# DSP-MLIR kernel
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

OPT_FLAG=
if [ "${OPT:-0}" != "0" ]; then OPT_FLAG=--opt; fi
dsp1 "$KERNEL" --stream --emit=llvm $OPT_FLAG -o "$STEM-native.ll"
llc "$STEM-native.ll" -filetype=obj -o "$STEM-native.o"

# Link with the dedicated CoreAudio host (Up/Down = wet, Left/Right = noise color)
clang++ -O3 lms-noise-macOS.cpp "$STEM-native.o" -framework AudioToolbox -framework CoreAudio -o "$STEM-macOS"

echo "Built $STEM-macOS -- run it to hear the demo."
