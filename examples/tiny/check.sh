#!/bin/bash
# Bring your own model, end to end, on a model the tools have never seen: examples/tiny/model.json
# (40 -> 12 -> 8 -> 4, made-up weights). These are the commands of docs/USAGE.md, in order, in a
# scratch directory (build/tiny, not committed). Exits non-zero if any step fails.
#
#   bash examples/tiny/check.sh        (Linux or WSL; needs Fe 26.5.1, Foundry, Python 3)
#
# Step 1 needs NumPy (the reference is NumPy). Where `python3` has none, set PYTHON_NUMPY to an
# interpreter that does, or VECTORS to a directory of vectors written earlier by step 1.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
set -e
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
ws=build/tiny
rm -rf "$ws"
mkdir -p "$ws"
start=$(date +%s)

echo "=== 1. reference vectors (NumPy)"
if [ -n "$VECTORS" ]; then
  cp -r "$VECTORS" "$ws/out"
else
  "${PYTHON_NUMPY:-python3}" tools/ref_vectors.py --model examples/tiny/model.json --random 200 --seed 1 --out "$ws/out"
fi

echo "=== 2. generate the workspace"
bash tools/gen_workspace.sh "$ws/out/model.json" "$ws/out" "$ws" 3 network | grep -v '^wrote'

echo "=== 3. test and build"
(cd "$ws" && fe test --color never 2>&1 | tr '\r' '\n' | grep -E 'FAIL|test result')
(cd "$ws" && fe test --color never > /dev/null 2>&1)
(cd "$ws" && fe build --color never --out-dir build/out . > /dev/null)
bash tools/check_size.sh "$ws/build/out"

echo "=== 4. native classifier against the reference"
(cd "$ws" && fe build --color never --backend native -O 1 --out-dir build/native --ingot cli . > /dev/null)
python3 tools/diff_native.py --ref "$ws/out" "$ws/build/native/cli"

echo "=== 5. deploy on anvil and send every call once"
bash tools/arbiter_gas_any.sh "$ws/build/out" "$ws/out" "$ws/build/tree.json" "$ws/build/arbiter-gas.json" 18613
bash tools/network_gas.sh "$ws/build/out" "$ws/out" "$ws/build/network-gas.json" 18614

echo "tiny: all steps passed in $(( $(date +%s) - start )) s"
