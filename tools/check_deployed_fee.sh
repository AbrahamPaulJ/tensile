#!/bin/bash
# Provenance of the live fee-tier system: the code at each address recorded in
# deploy/sepolia-fee.jsonl must be the runtime code `fe build` (Fe 26.5.1) produces from this
# repository, byte for byte:
#   Arbiter, FeeOracle, Network, Rows0   from the workspace examples/amm-fee
#   TickRelay                            from the ingot examples/amm-fee/relay
# Fe keeps constructor arguments as words after the code, so a contract built with arguments is the
# built code followed by those words, which are printed and compared with the record (the model
# root, bond and window of the arbiter; the relay and arbiter of the oracle, and its interval 0;
# the weight contract of the network). Read-only.
# usage: check_deployed_fee.sh [rpc-url] [record]      needs fe 26.5.1, cast, python3
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
rpc="${1:-https://ethereum-sepolia-rpc.publicnode.com}"
record="${2:-$root/deploy/sepolia-fee.jsonl}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
( cd "$root/examples/amm-fee" && fe build --color never --out-dir "$work/out" . ) > "$work/build.log" 2>&1 \
  || { cat "$work/build.log"; exit 1; }
( cd "$root/examples/amm-fee/relay" && fe build --color never --out-dir "$work/out" . ) > "$work/relay.log" 2>&1 \
  || { cat "$work/relay.log"; exit 1; }
field() {
  python3 -c '
import json, sys
for line in open(sys.argv[1]):
    row = json.loads(line)
    if row["label"] == sys.argv[2]:
        print(row.get(sys.argv[3], ""))
        break' "$record" "$1" "$2"
}
word() { python3 -c 'import sys; v = sys.argv[1]; v = int(v, 16) if v.startswith("0x") else int(v); print("%064x" % v)' "$1"; }
echo "$(fe --version)"
failures=0
for name in Arbiter FeeOracle Network Rows0 TickRelay; do
  address=$(field "$name" address)
  case $name in
    Arbiter) expected="$(word "$(field Arbiter root)")$(word "$(field Arbiter bond)")$(word "$(field Arbiter window)")" ;;
    FeeOracle) expected="$(word "$(field FeeOracle pool)")$(word "$(field FeeOracle arbiter)")$(word 0)" ;;
    Network) expected="$(word "$(field Network rows0)")" ;;
    # The relay's one fixed field, the keeper, is set by `init` from the deployer, not passed.
    TickRelay) expected="$(word "$(field TickRelay keeper)")" ;;
    *) expected="" ;;
  esac
  deployed=$(cast code --rpc-url "$rpc" "$address" | tr -d '\n' | sed 's/^0x//' | tr 'A-F' 'a-f')
  built=$(tr -d '\n' < "$work/out/$name.runtime.bin" | tr 'A-F' 'a-f')
  tail="${deployed#"$built"}"
  if [ -n "$built" ] && [ "$tail" != "$deployed" ] && [ "$tail" = "$expected" ]; then
    if [ -z "$tail" ]; then
      echo "$name at $address: $(( ${#built} / 2 )) bytes, equal to fe build"
    else
      echo "$name at $address: $(( ${#built} / 2 )) bytes equal to fe build, then $(( ${#tail} / 64 )) constructor words, equal to the record:"
      echo "$tail" | fold -w 64 | sed "s/^/    0x/"
    fi
  else
    echo "DIFFERENT: $name at $address ($(( ${#deployed} / 2 )) bytes deployed, $(( ${#built} / 2 )) built)"
    failures=$((failures + 1))
  fi
done
echo "check_deployed_fee: $failures difference(s)"
[ $failures -eq 0 ]
