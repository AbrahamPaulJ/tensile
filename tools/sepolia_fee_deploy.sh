#!/bin/bash
# The fee-tier example as a live system on Sepolia: deploys, from `fe build` output at Fe 26.5.1,
#
#   Rows0, Network   the whole 144-32-16-3 network, for read-only cross-checks (`classify`)
#   Arbiter          the generated arbiter of examples/amm-fee (model root, BOND, WINDOW)
#   TickRelay        examples/amm-fee/relay: the stand-in for the mainnet pool (a TRUST ASSUMPTION:
#                    the keeper writes mainnet's ticks into it; see docs/sepolia-fee.md)
#   FeeOracle        examples/amm-fee/ingots/consumer, reading the relay, interval 0 blocks (the
#                    relay's grid of mainnet blocks is its clock)
#
# then calls setOracle on the relay. Every transaction is appended to deploy/sepolia-fee.jsonl in
# the style of deploy/sepolia.jsonl; a step already recorded is skipped, so the script resumes.
#
#   --estimate   sends nothing: prints the balance, the gas price and eth_estimateGas of each
#                creation, and writes them to deploy/sepolia-fee-budget.json
#   --fund <address> <wei>   also sends that much to the second account (the one that plants a
#                deliberately false claim), once
#
# usage: SEPOLIA_KEY_FILE=<file with a line private_key=0x...> sepolia_fee_deploy.sh [--estimate | --fund A W]
#        optional: SEPOLIA_RPC, BOND (wei, default 0.0005 ETH), WINDOW (seconds, default 2700),
#        RECORD (default deploy/sepolia-fee.jsonl), CHAIN_ID (default 11155111; the anvil test sets it)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
rpc="${SEPOLIA_RPC:-https://ethereum-sepolia-rpc.publicnode.com}"
bond="${BOND:-500000000000000}"
window="${WINDOW:-2700}"
record="${RECORD:-$root/deploy/sepolia-fee.jsonl}"
chain="${CHAIN_ID:-11155111}"
[ -f "$SEPOLIA_KEY_FILE" ] || { echo "set SEPOLIA_KEY_FILE" >&2; exit 2; }
key=$(grep '^private_key=' "$SEPOLIA_KEY_FILE" | cut -d= -f2 | tr -d '\r\n')
me=$(cast wallet address --private-key "$key")
[ "$(cast chain-id --rpc-url "$rpc")" = "$chain" ] || { echo "not chain $chain" >&2; exit 2; }
mkdir -p "$(dirname "$record")"
touch "$record"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
( cd "$root/examples/amm-fee" && fe build --color never --out-dir "$work/ex" . ) > "$work/build.log" 2>&1 \
  || { cat "$work/build.log"; exit 1; }
( cd "$root/examples/amm-fee/relay" && fe build --color never --out-dir "$work/relay" . ) > "$work/relay.log" 2>&1 \
  || { cat "$work/relay.log"; exit 1; }
python3 "$root/tools/gen_tree.py" --model "$root/examples/amm-fee/out/model.json" --out "$work/tree.json" > /dev/null
root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$work/tree.json")
code() { echo "0x$(tr -d '\n' < "$1")${2#0x}"; }
fe_version="$(fe --version)"
echo "$fe_version, account $me, balance $(cast balance --ether --rpc-url "$rpc" "$me") ETH, model root $root_hash"

if [ "$1" = --estimate ]; then
  # Constructor arguments that are addresses take this account's address: a creation's cost does
  # not depend on which address it stores.
  any="$me"
  names=(Rows0 Network Arbiter TickRelay FeeOracle)
  codes=(
    "$(code "$work/ex/Rows0.bin")"
    "$(code "$work/ex/Network.bin" "$(cast abi-encode "constructor(address)" "$any")")"
    "$(code "$work/ex/Arbiter.bin" "$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" "$bond" "$window")")"
    "$(code "$work/relay/TickRelay.bin")"
    "$(code "$work/ex/FeeOracle.bin" "$(cast abi-encode "constructor(address,address,uint256)" "$any" "$any" 0)")"
  )
  gases=()
  for i in "${!names[@]}"; do
    gas=$(cast estimate --rpc-url "$rpc" --from "$me" --create "${codes[$i]}") || exit 1
    gases+=("$gas")
    printf '%-10s %10d gas (eth_estimateGas of the creation)\n' "${names[$i]}" "$gas"
  done
  balance=$(cast balance --rpc-url "$rpc" "$me")
  price=$(cast gas-price --rpc-url "$rpc")
  base=$(cast block --rpc-url "$rpc" latest --field baseFeePerGas)
  block=$(cast block-number --rpc-url "$rpc")
  python3 - "$root/deploy/sepolia-fee-budget.json" "$balance" "$price" "$base" "$block" "$fe_version" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${names[*]}" "${gases[@]}" <<'EOF'
import json, sys
path, balance, price, base, block, fe, when, names = sys.argv[1:9]
gases = [int(g) for g in sys.argv[9:]]
row = {"measured": when, "block": int(block), "fe": fe, "balance_wei": int(balance), "gas_price_wei": int(price),
       "base_fee_wei": int(base), "creations": dict(zip(names.split(), gases)), "creations_total": sum(gases)}
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(row, f, indent=1)
    f.write("\n")
print("total %d gas for the five creations; balance %d wei, gas price %d wei, base fee %d wei at block %s"
      % (row["creations_total"], row["balance_wei"], row["gas_price_wei"], row["base_fee_wei"], block))
EOF
  exit $?
fi

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

# save <label> <receipt json file> [extra key=value ...]
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
print("%-12s gas %9d  %s" % (label, row["gasUsed"], row["hash"]))
if row["status"] != 1:
    sys.exit("transaction failed: " + label)
EOF
}

# deploy <label> <creation code> [extra key=value ...]
deploy() {
  local label="$1" creation="$2"
  shift 2
  if [ -z "$(field "$label" address)" ]; then
    cast send --rpc-url "$rpc" --private-key "$key" --create "$creation" --json > "$work/receipt.json" || exit 1
    save "$label" "$work/receipt.json" "fe=$fe_version" "$@" || exit 1
  fi
}

deploy Rows0 "$(code "$work/ex/Rows0.bin")" "source=examples/amm-fee/ingots/network"
deploy Network "$(code "$work/ex/Network.bin" "$(cast abi-encode "constructor(address)" "$(field Rows0 address)")")" \
  "source=examples/amm-fee/ingots/network" "rows0=$(field Rows0 address)"
deploy Arbiter "$(code "$work/ex/Arbiter.bin" "$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" "$bond" "$window")")" \
  "source=examples/amm-fee/ingots/arbiter" "root=$root_hash" "bond=$bond" "window=$window"
deploy TickRelay "$(code "$work/relay/TickRelay.bin")" "source=examples/amm-fee/relay" "keeper=$me"
relay=$(field TickRelay address)
arbiter=$(field Arbiter address)
deploy FeeOracle "$(code "$work/ex/FeeOracle.bin" "$(cast abi-encode "constructor(address,address,uint256)" "$relay" "$arbiter" 0)")" \
  "source=examples/amm-fee/ingots/consumer" "pool=$relay" "arbiter=$arbiter" "interval=0"
oracle=$(field FeeOracle address)
if [ -z "$(field setOracle hash)" ]; then
  cast send --rpc-url "$rpc" --private-key "$key" "$relay" "setOracle(address)" "$oracle" --json > "$work/receipt.json" || exit 1
  save setOracle "$work/receipt.json" "oracle=$oracle" || exit 1
fi
# --fund <address> <wei>: sends test ETH to the second account that plants a false claim
# (tools/fee_keeper.mjs --plant), once, recorded.
if [ "$1" = --fund ] && [ -z "$(field "fund adversary" hash)" ]; then
  cast send --rpc-url "$rpc" --private-key "$key" --value "$3" "$2" --json > "$work/receipt.json" || exit 1
  save "fund adversary" "$work/receipt.json" "to=$2" "value=$3" || exit 1
fi
set_oracle=$(cast call --rpc-url "$rpc" "$relay" "oracle()(address)")
model_root=$(cast call --rpc-url "$rpc" "$arbiter" "modelRoot()(uint256)" | cut -d' ' -f1)
failures=0
[ "$(echo "$set_oracle" | tr 'A-F' 'a-f')" = "$(echo "$oracle" | tr 'A-F' 'a-f')" ] || { echo "WRONG: the relay feeds $set_oracle"; failures=$((failures + 1)); }
[ "$(cast to-hex "$model_root")" = "$(cast to-hex "$root_hash")" ] || { echo "WRONG: the arbiter's root is $model_root"; failures=$((failures + 1)); }
echo "Arbiter $arbiter, TickRelay $relay, FeeOracle $oracle, Network $(field Network address)"
echo "sepolia_fee_deploy: $failures wrong result(s), balance left $(cast balance --ether --rpc-url "$rpc" "$me") ETH"
[ $failures -eq 0 ]
