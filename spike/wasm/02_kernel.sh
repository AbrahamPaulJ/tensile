#!/bin/bash
# Probe 2: the flattened tensile + model source (tools/gen_wasm.py) on the WebAssembly target.
export PATH="$HOME/.local/node/bin:$PATH"
FE="$HOME/fe-master/target/release/fe"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
python3 tools/gen_wasm.py
echo "--- fe check"; s=$(date +%s); $FE check --color never wasm/tensile_wasm.fe 2>&1 | head -60; echo "check $(( $(date +%s) - s )) s"
echo "--- fe test --backend wasm"; s=$(date +%s); timeout 1200 $FE test --backend wasm --color never wasm/tensile_wasm.fe 2>&1 | tail -40; echo "test $(( $(date +%s) - s )) s"
echo "--- fe build --backend wasm"; s=$(date +%s); mkdir -p wasm/out; timeout 1200 $FE build --backend wasm --color never --out-dir wasm/out wasm/tensile_wasm.fe 2>&1 | tail -20; echo "build $(( $(date +%s) - s )) s"; ls -la wasm/out
