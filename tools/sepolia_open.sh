#!/bin/bash
# Open claims on Sepolia for anyone to check: a second Arbiter with a 30-day window, and a dozen
# claims about reference samples 1 to 12, some honest and some false in one value (which ones is in
# CLAIMS below and in the record). The false ones are left unchallenged on purpose: the watcher page
# (demo/dist/tensile-sepolia.html) and `node demo/test/sepolia.mjs` read them from the chain, re-run
# the network with the WebAssembly kernel, name the wrong value, and ask a node whether the challenge
# would succeed. Whoever sends that challenge is credited half the bond.
#
# Resumes like tools/sepolia_run.sh: a step already in deploy/sepolia-open.jsonl is skipped.
# usage: SEPOLIA_KEY_FILE=<file with a line private_key=0x...> sepolia_open.sh
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
rpc="${SEPOLIA_RPC:-https://ethereum-sepolia-rpc.publicnode.com}"
bond="${BOND:-500000000000000}"
window="${WINDOW:-2592000}"
record="$root/deploy/sepolia-open.jsonl"
# sample:mutation (claim number = position, from 0)
CLAIMS="1:honest 2:hidden:5 3:logit:3 4:class 5:hidden:17 6:honest 7:logit:8 8:hidden:30 9:class 10:hidden:0 11:logit:0 12:honest"
[ -f "$SEPOLIA_KEY_FILE" ] || { echo "set SEPOLIA_KEY_FILE" >&2; exit 2; }
key=$(grep '^private_key=' "$SEPOLIA_KEY_FILE" | cut -d= -f2 | tr -d '\r\n')
me=$(cast wallet address --private-key "$key")
[ "$(cast chain-id --rpc-url "$rpc")" = 11155111 ] || { echo "not Sepolia" >&2; exit 2; }
mkdir -p "$root/deploy" "$root/build"
touch "$record"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
( cd "$root" && fe build --color never --out-dir "$work/out" --ingot onchain . ) > "$work/build.log" 2>&1 \
  || { cat "$work/build.log"; exit 1; }
python3 "$root/tools/gen_tree.py" > /dev/null
root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$root/build/tree.json")

field() {
  python3 - "$record" "$1" "$2" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    row = json.loads(line)
    if row["label"] == sys.argv[2]:
        print(row.get(sys.argv[3], ""))
        break
EOF
}

save() {
  python3 - "$record" "$@" <<'EOF'
import json, sys
record, label, path = sys.argv[1:4]
r = json.load(open(path))
row = {"label": label, "hash": r["transactionHash"], "block": int(r["blockNumber"], 16),
       "gasUsed": int(r["gasUsed"], 16), "status": int(r["status"], 16)}
if r.get("contractAddress"):
    row["address"] = r["contractAddress"]
for extra in sys.argv[4:]:
    k, v = extra.split("=", 1)
    row[k] = v
open(record, "a").write(json.dumps(row) + "\n")
print("%-22s gas %9d  %s" % (label, row["gasUsed"], row["hash"]))
if row["status"] != 1:
    sys.exit("transaction failed: " + label)
EOF
}

if [ -z "$(field Arbiter address)" ]; then
  args=$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" "$bond" "$window")
  cast send --rpc-url "$rpc" --private-key "$key" --create "0x$(tr -d '\n' < "$work/out/Arbiter.bin")${args#0x}" --json > "$work/receipt.json" || exit 1
  save Arbiter "$work/receipt.json" "bond=$bond" "window=$window" || exit 1
fi
arbiter=$(field Arbiter address)
echo "Arbiter $arbiter (root $root_hash, bond $bond wei, window $window s)"

id=0
for entry in $CLAIMS; do
  sample=${entry%%:*}
  mutation=${entry#*:}
  label="claim $id"
  if [ -z "$(field "$label" hash)" ]; then
    if [ "$mutation" = honest ]; then
      mapfile -t c < <(python3 "$root/tools/claim_args.py" claim --sample "$sample")
    else
      mapfile -t c < <(python3 "$root/tools/claim_args.py" claim --sample "$sample" --mutate "$mutation")
    fi
    cast send --rpc-url "$rpc" --private-key "$key" --value "$bond" "$arbiter" \
      "claim(uint256[25],uint256[1],int32[10],uint8)" "${c[@]}" --json > "$work/receipt.json" || exit 1
    save "$label" "$work/receipt.json" "id=$id" "sample=$sample" "posted=$mutation" || exit 1
  fi
  id=$((id + 1))
done
count=$(cast call --rpc-url "$rpc" "$arbiter" "claimCount()(uint256)" | cut -d' ' -f1)
echo "sepolia_open: the arbiter holds $count claims, balance left $(cast balance --ether --rpc-url "$rpc" "$me") ETH"
