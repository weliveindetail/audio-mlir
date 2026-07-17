#!/usr/bin/env bash
set -ex
mkdir -p out

# DSP-MLIR kernels
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH

# LMS adaptive noise canceller: interactive @mu/@wet/@noise_kind globals,
# plus @lms_weights (persistent adaptive-filter state).
dsp1 lms-noise.mlir --emit=wasm -o out/lms-noise.wasm
wasm-ld --no-entry --import-memory --allow-undefined \
        --export=run --export=_mlir_ciface_run \
        --export=mu --export=wet --export=noise_kind --export=lms_weights \
        --export=__wasm_call_ctors --export=__heap_base --export=__data_end \
        out/lms-noise.wasm -o sample-wasm/lms-noise.linked.wasm

# Run from local server
set +x
echo "Sample is serving at http://localhost:8000/"
cd sample-wasm && python3 -m http.server
