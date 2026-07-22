#!/usr/bin/env bash
set -ex
mkdir -p out

# DSP-MLIR kernels
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

# LMS adaptive noise canceller: interactive @wet/@noise_kind globals plus the
# Mode-B host-side LFO (@lfo_mode = 1 + the @set_value_lfo_breakpoint setter):
# the browser streams the cutoff-sweep shape as per-sample breakpoints each block.
# --stream materializes per-op state (noise seeds, delay line, adaptive LMS
# weights, LFO phase/breakpoints) into module-scope __stream_state_*/globals,
# which makes the kernel malloc-free (no memref.alloc survives lowering) and lets
# state persist across per-block calls in the browser. --opt runs loop fusion.
dsp1 lms-noise.mlir --stream --opt --emit=wasm -o out/lms-noise.wasm
wasm-ld --no-entry --import-memory --allow-undefined \
        --export=run --export=_mlir_ciface_run \
        --export=wet --export=noise_kind --export=lfo_mode \
        --export=_mlir_ciface_set_value_lfo_breakpoint \
        --export=_mlir_ciface_set_note_event \
        --export=__wasm_call_ctors --export=__heap_base --export=__data_end \
        out/lms-noise.wasm -o sample-wasm/lms-noise.linked.wasm

# Run from local server
set +x
echo "Sample is serving at http://localhost:8000/"
cd sample-wasm && python3 -m http.server
