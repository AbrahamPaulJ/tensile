#!/bin/bash
# Regenerates ingots/model/src from ref/out (model.json and the reference test vectors), then lets
# `fe fmt` give the generated files their final layout. Re-run it whenever ref/out changes.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
set -e
cd "$(dirname "$0")/.."
python3 tools/gen_fe.py --model ref/out/model.json --vectors ref/out --out ingots/model/src
python3 tools/gen_tree.py --model ref/out/model.json --out build/tree.json
fe fmt ingots/model > /dev/null
