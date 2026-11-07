#!/bin/bash
# Runs the Foundry suite of the Fe Arbiter against the Solidity twin instead, by putting the twin's
# creation code where the tests read the Fe artifact. If the suite passes, the twin is a fair baseline
# for docs/numbers.md: the same tests cannot tell the two contracts apart.
#   wsl -e bash tools/m4_twin_suite.sh
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root" || exit 2
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
fe build --color never --out-dir "$out" --ingot onchain . > /dev/null 2>&1 || exit 1
stage="$HOME/.cache/tensile-evm"
mkdir -p "$stage"
rm -rf "$stage/src" "$stage/test" "$stage/fe-out" "$stage/ref-out" "$stage/invariant-out"
cp -r evm/src evm/test evm/foundry.toml "$stage/"
mkdir -p "$stage/fe-out" "$stage/ref-out" "$stage/invariant-out"
cp "$out"/*.bin "$stage/fe-out/"
cp ref/out/inputs.bin ref/out/pred.bin ref/out/trace.jsonl "$stage/ref-out/"
cd "$stage" || exit 2
forge build --color never > /dev/null || exit 1
python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bytecode"]["object"][2:])' \
  out/ArbiterTwin.sol/ArbiterTwin.json > fe-out/Arbiter.bin
forge test --color never --match-path 'test/Arbiter*.t.sol' "$@"
