#!/usr/bin/env bash
# Build + run the runtime micro-benchmark for lms-noise.mlir.
#
# Usage: ./lms-noise-bench.sh [kernel.mlir] [-- <bench args>]
#   default kernel: lms-noise.mlir; pass OPT=0 to disable --opt (default: --opt).
#   Extra args after the kernel are forwarded to the benchmark binary, e.g.
#     ./lms-noise-bench.sh -- --iterations 5000 --warmup 2000
#
# Compiles the kernel exactly like the CoreAudio host (--stream [--opt]) but links
# the headless timing driver lms-noise-bench.cpp instead of the audio driver. It
# measures PER-BLOCK render latency in a few representative configs (perf only --
# correctness is lms-noise-check). After building it also dumps the kernel object's
# STATIC memory footprint (section sizes) for information -- the kernel is
# malloc-free, so this static size is its whole memory cost.
set -e
mkdir -p out

KERNEL=lms-noise.mlir
if [ $# -gt 0 ] && [ "$1" != "--" ]; then KERNEL=$1; shift; fi
if [ "${1:-}" = "--" ]; then shift; fi
STEM=out/$(basename "${KERNEL%.mlir}")

# DSP-MLIR compiler
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

OPT_FLAG=--opt
if [ "${OPT:-1}" = "0" ]; then OPT_FLAG=; fi

set -x
dsp1 "$KERNEL" --stream --emit=llvm $OPT_FLAG -o "$STEM-bench.ll"
llc "$STEM-bench.ll" -filetype=obj -o "$STEM-bench.o"
clang++ -O3 -std=c++17 lms-noise-bench.cpp "$STEM-bench.o" -o "$STEM-bench"
set +x

# --- static memory footprint of the kernel object (information only) ----------
# The kernel is malloc-free (AssertNoHeapAllocPass enforces 0 heap alloc), so its
# entire memory cost is static: code (__text) + initialised globals (__data) +
# zero-init state buffers (__bss/__common). @voice_cut_shape (8000xf64 = 64000 B)
# dominates __data; the per-voice state banks live in __bss/__common.
echo
echo "=== kernel static memory footprint ($STEM-bench.o) ==="
size -m "$STEM-bench.o" 2>/dev/null || size "$STEM-bench.o"
echo "======================================================="
echo

"$STEM-bench" "$@"
