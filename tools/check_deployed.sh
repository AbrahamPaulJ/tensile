#!/bin/bash
# Provenance: the code at each recorded Sepolia address (deploy/sepolia.jsonl) must be the runtime
# code `fe build` produces from this repository, byte for byte. Fe keeps constructor arguments as
# words after the code, so the arbiter (root, bond, window) and the network (its two weight contracts)
# are the built code followed by those words, which are printed; the weight contracts compare exactly.
# usage: check_deployed.sh [rpc-url]      needs fe 26.5.1 and cast; read-only
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
rpc="${1:-https://ethereum-sepolia-rpc.publicnode.com}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
( cd "$root" && fe build --color never --out-dir "$work/out" --ingot onchain . ) > "$work/build.log" 2>&1 \
  || { cat "$work/build.log"; exit 1; }
failures=0
for name in Arbiter Network RowsLow RowsHigh; do
  address=$(python3 -c '
import json, sys
for line in open(sys.argv[1]):
    row = json.loads(line)
    if row["label"] == sys.argv[2]:
        print(row["address"])' "$root/deploy/sepolia.jsonl" "$name")
  deployed=$(cast code --rpc-url "$rpc" "$address" | tr -d '\n' | sed 's/^0x//')
  built=$(tr -d '\n' < "$work/out/$name.runtime.bin")
  tail="${deployed#"$built"}"
  if [ -n "$built" ] && [ "$deployed" = "$built" ]; then
    echo "$name at $address: $(( ${#built} / 2 )) bytes, equal to fe build"
  elif [ -n "$built" ] && [ "$tail" != "$deployed" ] && [ $(( ${#tail} % 64 )) -eq 0 ] && [ ${#tail} -le 192 ]; then
    echo "$name at $address: $(( ${#built} / 2 )) bytes equal to fe build, then $(( ${#tail} / 64 )) constructor words:"
    echo "$tail" | fold -w 64 | sed "s/^/    0x/"
  else
    echo "DIFFERENT: $name at $address ($(( ${#deployed} / 2 )) bytes deployed, $(( ${#built} / 2 )) built)"
    failures=$((failures + 1))
  fi
done
echo "check_deployed: $failures difference(s)"
[ $failures -eq 0 ]
