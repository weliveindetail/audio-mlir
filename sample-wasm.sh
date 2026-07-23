#!/usr/bin/env bash
set -ex
mkdir -p out

# DSP-MLIR kernels
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

# LMS adaptive noise canceller: interactive @wet/@noise_kind globals plus the
# per-voice, trigger-anchored cutoff LFO. The shared @voice_cut_shape table (one
# cycle of the cutoff coefficient) is written directly from JS -- the browser fills
# it from the on-screen shape editor -- and @cut_lfo_step sets the per-voice LFO
# speed the kernel latches at each note-on. Both are plain memref globals, exported
# so JS can read/write them in linear memory (like @wet). --stream materializes
# per-op state (noise seeds, delay line, adaptive LMS weights, per-voice phases)
# into module-scope __stream_state_*/globals, which makes the kernel malloc-free
# (no memref.alloc survives lowering) and lets state persist across per-block calls
# in the browser. --opt runs loop fusion.
dsp1 lms-noise.mlir --stream --opt --emit=wasm -o out/lms-noise.wasm
wasm-ld --no-entry --import-memory --allow-undefined \
        --export=run --export=_mlir_ciface_run \
        --export=wet --export=noise_kind \
        --export=cut_lfo_step --export=voice_cut_shape --export=voice_cut_phase \
        --export=_mlir_ciface_set_note_event \
        --export=__wasm_call_ctors --export=__heap_base --export=__data_end \
        out/lms-noise.wasm -o sample-wasm/lms-noise.linked.wasm

# Run from local server
set +x
echo "Sample is serving at http://localhost:8000/"
cd sample-wasm && python3 -m http.server
