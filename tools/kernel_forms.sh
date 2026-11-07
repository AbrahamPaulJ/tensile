#!/bin/bash
# Measures every form of the kernel's inner dot product: the one in ingots/tensile/src/dot.fe
# ("kept") and each alternative in tests/kernel_forms/. For each form, in a scratch copy of the
# workspace with that file as dot.fe:
#   - the tensile unit tests on both backends (the form must be bit-identical);
#   - gas of one layer 1 neuron (784 multiply-adds) inside the EVM: the gas() difference
#     reported by ingots/onchain/src/gas.fe;
#   - the same neuron as a real transaction on anvil (Osaka): `cast estimate` of
#     NeuronCheck.neuron1, which adds the 21,000 base cost, calldata and decoding;
#   - the native classifier at -O 1 on all 10,000 samples, checked against the reference.
# Prints one table. usage: kernel_forms.sh [form-name ...]   (default: all)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
port=8571
rpc="http://127.0.0.1:$port"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
anvil --hardfork osaka --port $port > "$work/anvil.log" 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null; rm -rf "$work"' EXIT

mapfile -t call < <(python3 "$root/tools/neuron_args.py" 1 0 0)
signature="neuron1(uint256[25],uint256[25],int32)"

# measure <name> <dot.fe>: prints one table row.
measure() {
  local name="$1" dir="$work/$1"
  mkdir -p "$dir"
  cp "$root/fe.toml" "$dir/"
  cp -r "$root/ingots" "$dir/ingots"
  cp "$2" "$dir/ingots/tensile/src/dot.fe"
  cd "$dir"
  local tests="ok"
  for backend in sonatina native; do
    fe test --color never --backend $backend ingots/tensile > "$dir/tensile-$backend.log" 2>&1 || tests="FAIL"
  done
  fe test --color never --show-logs ingots/onchain 2>&1 | tr '\r' '\n' > "$dir/gas.log"
  local inside=$(python3 "$root/tools/kernel_gas.py" 1 < "$dir/gas.log")
  if ! fe build --color never --out-dir "$dir/out" --ingot onchain . > "$dir/build.log" 2>&1; then
    printf '%-14s does not build: %s
' "$name" "$(grep -m 1 -A 6 '^error' "$dir/build.log" | tr '
' ' ')"
    return
  fi
  local size=$(( $(tr -d '\n' < "$dir/out/NeuronCheck.runtime.bin" | wc -c) / 2 ))
  local address=$(cast send --rpc-url $rpc --private-key $key --create "0x$(cat "$dir/out/NeuronCheck.bin")" --json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["contractAddress"])')
  local got=$(cast call --rpc-url $rpc "$address" "$signature(int8)" "${call[0]}" "${call[1]}" "${call[2]}")
  [ "$got" = "${call[3]}" ] || tests="FAIL"
  local estimate=$(cast estimate --rpc-url $rpc "$address" "$signature" "${call[0]}" "${call[1]}" "${call[2]}")
  fe build --color never --backend native -O 1 --out-dir "$dir/native" --ingot cli . > "$dir/native.log" 2>&1
  local native=$(python3 "$root/tools/diff_native.py" "$dir/native/cli" | grep 'classes' \
    | sed -E 's/.*classes +([0-9]+) samples, ([0-9]+) mismatches, ([0-9.]+) s/\3 s (\2 mismatches)/')
  printf '%-14s %-5s %9s %7s %9s %7s %s\n' "$name" "$tests" "$inside" \
    "$(echo "scale=1; $inside / 784" | bc)" "$estimate" "$size" "$native"
}

sleep 2
printf '%-14s %-5s %9s %7s %9s %7s %s\n' form tests "gas()" per-mac estimate bytes "native 10,000 samples"
if [ $# -eq 0 ]; then
  measure kept "$root/ingots/tensile/src/dot.fe"
  for form in "$root"/tests/kernel_forms/*.fe; do
    measure "$(basename "$form" .fe)" "$form"
  done
else
  for name in "$@"; do
    if [ "$name" = kept ]; then
      measure kept "$root/ingots/tensile/src/dot.fe"
    else
      measure "$name" "$root/tests/kernel_forms/$name.fe"
    fi
  done
fi
