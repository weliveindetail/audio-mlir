#!/usr/bin/env bash
set -ex
mkdir -p out

# DSP-MLIR kernels
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

# LMS adaptive noise canceller: interactive @mu/@wet/@noise_kind globals.
# --stream materializes per-op state (noise seeds, delay line, adaptive LMS
# weights) into module-scope __stream_state_* globals, which makes the kernel
# malloc-free (no memref.alloc survives lowering) and lets state persist across
# per-block calls in the browser. --opt runs the loop-fusion pipeline.
dsp1 lms-noise.mlir --stream --opt --emit=wasm -o out/lms-noise.wasm
wasm-ld --no-entry --import-memory --allow-undefined \
        --export=run --export=_mlir_ciface_run \
        --export=mu --export=wet --export=noise_kind --export=lfo_period \
        --export=__wasm_call_ctors --export=__heap_base --export=__data_end \
        out/lms-noise.wasm -o sample-wasm/lms-noise.linked.wasm

# Run from local server
set +x
echo "Sample is serving at http://localhost:8000/"
cd sample-wasm && python3 -m http.server
