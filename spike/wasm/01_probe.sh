#!/bin/bash
# Probe 1: u256 arithmetic inside functions on the WebAssembly target.
export PATH="$HOME/.local/node/bin:$PATH"
FE="$HOME/fe-master/target/release/fe"
cd "$(dirname "$0")"
$FE --version
echo "--- fe test --backend wasm p1_u256.fe"; time $FE test --backend wasm --color never p1_u256.fe 2>&1 | tail -15
echo "--- fe build --backend wasm"; mkdir -p out; $FE build --backend wasm --color never --out-dir out p1_u256.fe 2>&1 | tail -10; ls -la out/
