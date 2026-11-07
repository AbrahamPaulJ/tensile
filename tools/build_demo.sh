#!/bin/bash
# Builds what the demo page embeds: the Fe contracts, the real compiler diagnostics of a few
# compile-fail cases, and demo/generated/assets.json. The page itself is then built with
# `npm run build` in demo/ (Node 20 or later; no Fe needed for that step).
# usage: build_demo.sh
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.." || exit 2
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
bash tools/gen_model.sh || exit 1
fe build --color never --out-dir "$out/fe" --ingot onchain . > "$out/build.log" 2>&1 || { cat "$out/build.log"; exit 1; }
python3 tools/demo_diagnostics.py "$out/errors.json" || exit 1
python3 tools/export_demo.py --fe-out "$out/fe" --compiler-errors "$out/errors.json"
