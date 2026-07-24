#!/usr/bin/env bash
# Build + run the runtime micro-benchmark for BOTH lms-noise ports (Faust +
# Cmajor). This is the port-side twin of ../lms-noise-bench.sh: same block size,
# same real-time budget, same representative configs and the same per-block
# latency / throughput statistics -- so the ports and the DSP-MLIR kernel can be
# A/B'd apart from the language/toolchain. See ports-bench.cpp for the harness.
#
# The port sources live in ports/ (lms-noise.dsp, LMSNoise.cmajorpatch); all
# generated headers/objects and the built binary land in out/ (shared with the
# kernel bench).
#
# Usage: ./ports-bench.sh [-- <bench args>]
#   Extra args after `--` are forwarded to the binary, e.g.
#     ./ports-bench.sh -- --iterations 5000 --warmup 2000
#     ./ports-bench.sh -- --faust-only        # or --cmajor-only
#
# Pipeline:
#   1. Faust  -> two C++ classes: the polyphonic `process` (one voice) and the
#      mono `effect`. The harness instantiates 8 voices + 1 effect by hand (what
#      faust2* poly does) so the whole d = sum_v tone_v + n0; out = d - wet*y
#      pipeline runs per block, matching the kernel's tensor<8x128> bank.
#   2. Cmajor -> one C++ class for the whole graph (voices + master already
#      inside); the harness just pushes MIDI + advances a block.
#   3. Compile ports-bench.cpp against all three headers and run it.
#   4. Dump each generated DSP object's STATIC footprint (section sizes), the same
#      information lms-noise-bench.sh prints for the kernel object. Like the
#      kernel these compute paths are malloc-free, so static size ~= whole cost.
#
# NOTE on the wet=1 white-noise configs: the ports' LMS taps are pure (un-leaked)
# integrators, so their weight vector slowly random-walks and eventually overflows
# to inf over millions of samples (Faust ~296k at mu=1e-3, scaling as ~1/mu; the
# f64 kernel does not, so this is a real port-fidelity gap, not a timing artifact).
# The harness detects this, still reports the (valid) per-block timing, and flags
# the run with a WARNING line + "diverged":true in BENCH_JSON.
set -e
cd "$(dirname "$0")"
mkdir -p out

SRC=ports   # port sources (lms-noise.dsp, LMSNoise.cmajorpatch) live here

FAUST_INC=${FAUST_INC:-$(dirname "$(dirname "$(readlink -f "$(command -v faust)" 2>/dev/null || command -v faust)")")/include}
# Homebrew keg layout: .../Cellar/faust/<ver>/bin/faust -> ../include
if [ ! -f "$FAUST_INC/faust/dsp/dsp.h" ]; then
    FAUST_INC=$(brew --prefix faust 2>/dev/null)/include
fi
if [ ! -f "$FAUST_INC/faust/dsp/dsp.h" ]; then
    echo "!! Faust runtime headers not found (looked in '$FAUST_INC')." >&2
    echo "   Set FAUST_INC=/path/to/faust/include and retry." >&2
    exit 1
fi

# Cmajor CLI ships inside the VS Code extension; allow override via CMAJ=...
CMAJ=${CMAJ:-$(command -v cmaj || true)}
if [ -z "$CMAJ" ]; then
    CMAJ=$(ls -d "$HOME"/.vscode*/extensions/cmajorsoftware.cmajor-tools-*/bin/cmaj 2>/dev/null | sort | tail -1 || true)
fi
if [ -z "$CMAJ" ] || [ ! -x "$CMAJ" ]; then
    echo "!! cmaj CLI not found. Set CMAJ=/path/to/cmaj (from the Cmajor VS Code" >&2
    echo "   extension bin/ dir) and retry." >&2
    exit 1
fi

echo "=== generating port C++ (faust=$FAUST_INC, cmaj=$CMAJ) ==="
set -x
# Faust voice (`process`) and effect (reached via library() so the same .dsp
# defines both). Single precision == the shipped f32 ports. The effect wrapper
# resolves lms-noise.dsp through -I "$SRC".
faust -lang cpp -cn FaustVoice  -scal            -o out/FaustVoice.h  "$SRC/lms-noise.dsp"
printf 'process = library("lms-noise.dsp").effect;\n' > out/_effect.dsp
faust -lang cpp -cn FaustEffect -scal -I "$SRC"   -o out/FaustEffect.h out/_effect.dsp
# Cmajor whole-graph class.
"$CMAJ" generate --target=cpp --output=out/LMSNoiseCmaj.h "$SRC/LMSNoise.cmajorpatch"
set +x

echo
echo "=== building ports-bench ==="
set -x
clang++ -O3 -std=c++17 -I "$FAUST_INC" ports-bench.cpp -o out/ports-bench
set +x

# --- static footprint of each generated DSP object (information only) ---------
# Symmetric with ../lms-noise-bench.sh's `size -m` on the kernel object. We
# compile each generated class into its own object (a tiny TU that just forces
# the class to be emitted) and size it. Faust's shared cutoff-shape table and
# Cmajor's voice/LMS state live in these sections just like @voice_cut_shape does
# in the kernel object.
foot() { # $1 = header, $2 = class, $3 = need faust base?
    local tu=out/_foot_$2.cpp
    {
        [ "$3" = faust ] && printf '#include <faust/dsp/dsp.h>\n#include <faust/gui/meta.h>\n#include <faust/gui/UI.h>\n'
        printf '#include "%s"\n' "$(basename "$1")"
        printf '%s g_keep_%s;\n' "$2" "$2"
    } > "$tu"
    clang++ -O3 -std=c++17 -I "$FAUST_INC" -I out -c "$tu" -o "out/_foot_$2.o" 2>/dev/null
    echo "--- $2 ($1) ---"
    size -m "out/_foot_$2.o" 2>/dev/null || size "out/_foot_$2.o"
}
echo
echo "=== port static memory footprints (per generated DSP object) ==="
foot out/FaustVoice.h   FaustVoice  faust
foot out/FaustEffect.h  FaustEffect faust
foot out/LMSNoiseCmaj.h LMSNoise
echo "==============================================================="
echo

# forward args after `--`
if [ "${1:-}" = "--" ]; then shift; fi
./out/ports-bench "$@"
