#!/bin/bash
# Development: deploy the arbiter on anvil, post one claim, and dump the storage slots of
# `claims[0]` for each candidate base slot, to see how Fe laid the record out.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
out="$(mktemp -d)"
fe build --color never --out-dir "$out" --ingot onchain . > /dev/null || exit 1
port=8574
rpc="http://127.0.0.1:$port"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
anvil --hardfork osaka --port $port > /dev/null 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null; rm -rf "$out"' EXIT
sleep 2
root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$root/build/tree.json")
args=$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" 1000 3600)
arbiter=$(cast send --rpc-url $rpc --private-key $key --create "0x$(cat "$out/Arbiter.bin")${args#0x}" --json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["contractAddress"])')
mapfile -t honest < <(python3 tools/claim_args.py claim)
cast send --rpc-url $rpc --private-key $key --value 1000 "$arbiter" "claim(uint256[25],uint256[1],int32[10],uint8)" "${honest[@]}" --json > /dev/null
echo "plain slots 0..7:"
for s in 0 1 2 3 4 5 6 7; do echo "  $s: $(cast storage --rpc-url $rpc "$arbiter" $s)"; done
for base in 0 1 2 3 4 5; do
  entry=$(cast index uint256 0 $base)
  echo "claims base slot $base, entry $entry:"
  for i in 0 1 2 3 4 5 6 7; do
    v=$(cast storage --rpc-url $rpc "$arbiter" $(python3 -c "print(hex(int('$entry',16) + $i))"))
    [ "$v" != "0x0000000000000000000000000000000000000000000000000000000000000000" ] && echo "  +$i: $v"
  done
done
