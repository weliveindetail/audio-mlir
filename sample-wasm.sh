#!/usr/bin/env bash
set -ex
mkdir -p out

# DSP-MLIR kernel
PATH=$(pwd)/../build-relwithdebinfo/bin:$PATH
dsp1 osc-low-pass.mlir --emit=wasm -o out/osc-low-pass.wasm
wasm-ld --no-entry --import-memory --allow-undefined \
        --export=run --export=_mlir_ciface_run --export=cutoff \
        --export=__wasm_call_ctors --export=__heap_base --export=__data_end \
        out/osc-low-pass.wasm -o sample-wasm/osc-low-pass.linked.wasm

# Run from local server
set +x
echo "Sample is serving at http://localhost:8000/"
cd out && python3 -m http.server
