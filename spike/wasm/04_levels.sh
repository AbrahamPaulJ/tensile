#!/bin/bash
export PATH="$HOME/.local/node/bin:$PATH"
FE=$HOME/fe-master/target/release/fe
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge
for o in 0 2; do
  d=$(mktemp -d); s=$(date +%s)
  $FE build --backend wasm -O $o --color never --out-dir $d wasm/tensile_wasm.fe > /dev/null && echo "-O $o build $(( $(date +%s) - s )) s"
  node wasm/diff.mjs $d/tensile_wasm.wasm; rm -rf $d
done
