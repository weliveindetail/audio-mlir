#!/usr/bin/env bash
# Build + run the interactive CoreAudio LMS adaptive-noise-canceller demo:
# a 440 Hz tone buried in broadband noise, with the noise adaptively removed.
#
# Usage: ./lms-noise-macOS.sh [kernel.mlir]   (default: lms-noise.mlir)
#
# Uses the dedicated host lms-noise-macOS.cpp: streams @run's output and drives
# the @wet knob with Up/Down and the @noise_kind color knob (the
# dsp.index_switch selector) with Left/Right. The noise reference is generated
# in-kernel from an LCG stream, so no RNG primitive or host-fed data is needed.
set -ex
mkdir -p out

KERNEL=${1:-lms-noise.mlir}
STEM=out/$(basename "${KERNEL%.mlir}")

# DSP-MLIR kernel
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

OPT_FLAG=--opt
if [ "${OPT:-1}" = "0" ]; then OPT_FLAG=; fi
# The malloc-free guarantee is on by default: dsp1 fails the build if the kernel
# would malloc at runtime, so the streamed kernel is real-time safe (callable from
# the audio callback). Pass --allow-heap to downgrade that error to a warning.
dsp1 "$KERNEL" --stream --emit=llvm $OPT_FLAG -o "$STEM-native.ll"
llc "$STEM-native.ll" -filetype=obj -o "$STEM-native.o"

# Link with the dedicated CoreAudio host (Up/Down = wet, Left/Right = noise color).
clang++ -O3 lms-noise-macOS.cpp "$STEM-native.o" -framework AudioToolbox -framework CoreAudio -framework CoreMIDI -framework CoreFoundation -o "$STEM-macOS"

echo "Built $STEM-macOS -- run it to hear the demo."
