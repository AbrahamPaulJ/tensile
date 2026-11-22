#!/bin/bash
# Every gas and size figure the project quotes, from one command: builds the contracts as they stand,
# builds the variants with the abstractions removed (tests/abstraction/) and the Solidity twins, runs
# real transactions on anvil, and writes docs/numbers.md and demo/generated/numbers.json.
#   wsl -e bash tools/m4_numbers.sh [--no-forms]    (--no-forms skips the slow kernel forms table)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root" || exit 2
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
started=$(date +%s.%N)

build_variant() { # build_variant <name> <patch command...>: a scratch workspace, patched, built.
  local name="$1"
  shift
  local dir="$work/$name"
  mkdir -p "$dir"
  cp "$root/fe.toml" "$dir/"
  cp -r "$root/ingots" "$dir/ingots"
  ( cd "$dir" && "$@" ) || { echo "patching $name failed"; exit 1; }
  ( cd "$dir" && fe build --color never $FE_FLAGS --out-dir "$work/out-$name" --ingot onchain . ) > "$work/$name.log" 2>&1 \
    || { echo "building $name failed:"; cat "$work/$name.log"; exit 1; }
}

bash tools/gen_model.sh > /dev/null || exit 1
build_variant typed true
build_variant raw_neuron cp "$root/tests/abstraction/neuron_check_raw.fe" ingots/onchain/src/neuron_check.fe
build_variant inline_settle python3 "$root/tests/abstraction/inline_settle.py" ingots/onchain/src/arbiter.fe
# The same source at the other optimisation level, for section 6 (`-O 1` is the default and what ships).
FE_FLAGS="-O 2" build_variant o2 true

# The Solidity twins, compiled with the settings of evm/foundry.toml.
stage="$HOME/.cache/tensile-evm"
mkdir -p "$stage"
rm -rf "$stage/src" "$stage/test"
cp -r "$root/evm/src" "$root/evm/test" "$root/evm/foundry.toml" "$stage/"
( cd "$stage" && forge build --color never ) > "$work/forge.log" 2>&1 || { cat "$work/forge.log"; exit 1; }
twin="$stage/out/NeuronTwin.sol/NeuronTwin.json"
arbiter_twin="$stage/out/ArbiterTwin.sol/ArbiterTwin.json"
mkdir -p "$work/out-twin"
python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bytecode"]["object"][2:])' "$arbiter_twin" > "$work/out-twin/Arbiter.bin"
# The twins again through solc's legacy pipeline (no via-IR), same optimiser runs, for section 6.
# NeuronTwin compiles. ArbiterTwin does not ("Stack too deep"): that is recorded, and checked here, so
# the table cannot go on saying so if a later solc or a changed twin makes it compile.
legacy() { ( cd "$stage" && FOUNDRY_VIA_IR=false FOUNDRY_OUT=out-legacy FOUNDRY_CACHE_PATH=cache-legacy forge build --color never "$@" ); }
legacy src/NeuronTwin.sol > "$work/forge-legacy.log" 2>&1 || { cat "$work/forge-legacy.log"; exit 1; }
twin_legacy="$stage/out-legacy/NeuronTwin.sol/NeuronTwin.json"
if legacy src/ArbiterTwin.sol > "$work/forge-legacy-arbiter.log" 2>&1; then
  echo "ArbiterTwin now compiles without via-IR: measure it and update section 6 of tools/m4_numbers.py"; exit 1
fi
grep -q "Stack too deep" "$work/forge-legacy-arbiter.log" || { cat "$work/forge-legacy-arbiter.log"; exit 1; }

# Reports. Each script checks every result against the reference or the protocol and exits non-zero
# if one is wrong, so a figure below is never the gas of a wrong answer.
run() { # run <file> <command...>
  local file="$1"
  shift
  "$@" > "$work/$file" 2>&1 || { echo "FAILED: $*"; cat "$work/$file"; exit 1; }
}
run sizes.txt bash tools/check_size.sh "$work/out-typed"
run report-typed.txt bash tools/gas_report.sh "$work/out-typed" "$twin"
run arbiter-typed.txt bash tools/arbiter_gas.sh "$work/out-typed"
run report-raw.txt bash tools/gas_report.sh "$work/out-raw_neuron"
run arbiter-inline.txt bash tools/arbiter_gas.sh "$work/out-inline_settle"
run arbiter-twin.txt bash tools/arbiter_gas.sh "$work/out-twin"
run report-o2.txt bash tools/gas_report.sh "$work/out-o2" "$twin_legacy"
run arbiter-o2.txt bash tools/arbiter_gas.sh "$work/out-o2"
if [ "$1" != "--no-forms" ]; then
  run forms.txt bash tools/kernel_forms.sh
else
  [ -f build/kernel_forms.txt ] && cp build/kernel_forms.txt "$work/forms.txt"
fi

raw_size=$(( $(tr -d '\n' < "$work/out-raw_neuron/NeuronCheck.runtime.bin" | wc -c) / 2 ))
inline_size=$(( $(tr -d '\n' < "$work/out-inline_settle/Arbiter.runtime.bin" | wc -c) / 2 ))
twin_size=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["deployedBytecode"]["object"]) // 2 - 1)' "$arbiter_twin")
python3 tools/m4_numbers.py "$work" "$raw_size" "$inline_size" "$twin_size" "$(fe --version)" "$(forge --version | head -1)" \
  "$(printf '%.1f' "$(echo "$(date +%s.%N) - $started" | bc)")"
