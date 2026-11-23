#!/bin/bash
# The dispute on a public network (Sepolia): deploys the Fe contracts built by `fe build`, posts one
# honest claim and three false ones (reference sample 0, each false claim with one value wrong: a
# hidden neuron, an output, the class), slashes the three with the three kinds of challenge,
# finalises the honest claim after the window and withdraws. Every transaction's hash and gasUsed is
# appended to deploy/sepolia.jsonl, from which tools/sepolia_page.py writes docs/sepolia.md.
#
# The script resumes: a step whose label is already in the record is skipped, so it can be run again
# after the window has passed. The claim's state is read back after every step and the script fails
# if it is not what the protocol says.
#
# usage: SEPOLIA_KEY_FILE=<file with a line private_key=0x...> sepolia_run.sh
#        optional: SEPOLIA_RPC (default: a public node), BOND (wei), WINDOW (seconds)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
rpc="${SEPOLIA_RPC:-https://ethereum-sepolia-rpc.publicnode.com}"
bond="${BOND:-1000000000000000}"
window="${WINDOW:-600}"
record="$root/deploy/sepolia.jsonl"
[ -f "$SEPOLIA_KEY_FILE" ] || { echo "set SEPOLIA_KEY_FILE" >&2; exit 2; }
key=$(grep '^private_key=' "$SEPOLIA_KEY_FILE" | cut -d= -f2 | tr -d '\r\n')
me=$(cast wallet address --private-key "$key")
[ "$(cast chain-id --rpc-url "$rpc")" = 11155111 ] || { echo "not Sepolia" >&2; exit 2; }
mkdir -p "$root/deploy" "$root/build"
touch "$record"
failures=0

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
( cd "$root" && fe build --color never --out-dir "$work/out" --ingot onchain . ) > "$work/build.log" 2>&1 \
  || { cat "$work/build.log"; exit 1; }
python3 "$root/tools/gen_tree.py" > /dev/null
root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$root/build/tree.json")
echo "fe $(fe --version), account $me, balance $(cast balance --ether --rpc-url "$rpc" "$me") ETH"

# field <label> <name>: a field of the recorded step, or nothing.
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
print("%-22s gas %9d  %s" % (label, row["gasUsed"], row["hash"]))
if row["status"] != 1:
    sys.exit("transaction failed: " + label)
EOF
}

# deploy <label> <creation code file> [constructor args hex]: prints nothing, records the address.
deploy() {
  if [ -z "$(field "$1" address)" ]; then
    cast send --rpc-url "$rpc" --private-key "$key" --create "0x$(tr -d '\n' < "$2")${3#0x}" --json > "$work/receipt.json" || exit 1
    save "$1" "$work/receipt.json" || exit 1
  fi
}

# send <label> <value-wei> <address> <signature> <args...>
send() {
  local label="$1" value="$2" address="$3" signature="$4"
  shift 4
  if [ -z "$(field "$label" hash)" ]; then
    cast send --rpc-url "$rpc" --private-key "$key" --value "$value" "$address" "$signature" "$@" --json > "$work/receipt.json" || exit 1
    save "$label" "$work/receipt.json" || exit 1
  fi
}

# expect_status <claim id> <status code> <what>
expect_status() {
  local got
  got=$(cast call --rpc-url "$rpc" "$arbiter" "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)" "$1" | tail -1)
  if [ "$got" != "$2" ]; then
    echo "WRONG: claim $1 has status $got after $3, expected $2"
    failures=$((failures + 1))
  fi
}

deploy RowsLow "$work/out/RowsLow.bin"
deploy RowsHigh "$work/out/RowsHigh.bin"
deploy Network "$work/out/Network.bin" "$(cast abi-encode "constructor(address,address)" "$(field RowsLow address)" "$(field RowsHigh address)")"
deploy Arbiter "$work/out/Arbiter.bin" "$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" "$bond" "$window")"
network=$(field Network address)
arbiter=$(field Arbiter address)
echo "Network $network, Arbiter $arbiter (root $root_hash, bond $bond wei, window $window s)"

mapfile -t honest < <(python3 "$root/tools/claim_args.py" claim)
mapfile -t bad_hidden < <(python3 "$root/tools/claim_args.py" claim --mutate hidden:0)
mapfile -t bad_logit < <(python3 "$root/tools/claim_args.py" claim --mutate logit:0)
mapfile -t bad_class < <(python3 "$root/tools/claim_args.py" claim --mutate class)
mapfile -t n1 < <(python3 "$root/tools/claim_args.py" proof 1 0)
mapfile -t n2 < <(python3 "$root/tools/claim_args.py" proof 2 0)
claim_sig="claim(uint256[25],uint256[1],int32[10],uint8)"

# The whole network, run by the deployed bytecode, read-only: it must give the reference's class.
class=$(cast call --rpc-url "$rpc" "$network" "classify(uint256[25])(uint256)" "${honest[0]}" | cut -d' ' -f1)
if [ "$class" != "${honest[3]}" ]; then
  echo "WRONG: Network.classify on Sepolia says $class, the reference says ${honest[3]}"
  failures=$((failures + 1))
else
  echo "Network.classify(sample 0) on Sepolia: $class, as the reference"
fi
# What sending that call would cost on this chain: an estimate from the node, recorded as such.
if [ -z "$(field "estimate classify" gasUsed)" ]; then
  estimate=$(cast estimate --rpc-url "$rpc" --from "$me" "$network" "classify(uint256[25])" "${honest[0]}" | head -1)
  python3 -c 'import json, sys; open(sys.argv[1], "a").write(json.dumps({"label": "estimate classify", "gasUsed": int(sys.argv[2]), "estimate": True}) + "\n")' "$record" "$estimate"
  echo "estimate of Network.classify as a transaction: $estimate gas"
fi

send "claim honest" "$bond" "$arbiter" "$claim_sig" "${honest[@]}"
send "claim false hidden" "$bond" "$arbiter" "$claim_sig" "${bad_hidden[@]}"
send "claim false output" "$bond" "$arbiter" "$claim_sig" "${bad_logit[@]}"
send "claim false class" "$bond" "$arbiter" "$claim_sig" "${bad_class[@]}"

# A challenge of the honest claim must revert (NoFraud): shown by a call, nothing is sent.
if cast call --rpc-url "$rpc" --from "$me" "$arbiter" "challengeClass(uint256,int32[10])" 0 "${honest[2]}" > /dev/null 2>&1; then
  echo "WRONG: a class challenge of the honest claim did not revert"
  failures=$((failures + 1))
fi

send "challengeNeuron1" 0 "$arbiter" \
  "challengeNeuron1(uint256,uint256,uint256[25],uint256[1],uint256[25],int32,bytes32[])" \
  1 0 "${bad_hidden[0]}" "${bad_hidden[1]}" "${n1[0]}" "${n1[1]}" "${n1[2]}"
expect_status 1 2 "challengeNeuron1"
send "challengeNeuron2" 0 "$arbiter" \
  "challengeNeuron2(uint256,uint256,uint256[1],int32[10],uint256[1],int32,bytes32[])" \
  2 0 "${bad_logit[1]}" "${bad_logit[2]}" "${n2[0]}" "${n2[1]}" "${n2[2]}"
expect_status 2 2 "challengeNeuron2"
send "challengeClass" 0 "$arbiter" "challengeClass(uint256,int32[10])" 3 "${bad_class[2]}"
expect_status 3 2 "challengeClass"

deadline=$(cast call --rpc-url "$rpc" "$arbiter" "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)" 0 | sed -n 3p | cut -d' ' -f1)
now=$(cast block --rpc-url "$rpc" latest --field timestamp)
if [ -z "$(field finalize hash)" ] && [ "$now" -le "$deadline" ]; then
  echo "the honest claim's window ends in $((deadline - now)) s: run this script again after that"
  echo "sepolia_run: $failures wrong result(s) so far"
  exit $(( failures > 0 ? 1 : 3 ))
fi
send "finalize" 0 "$arbiter" "finalize(uint256)" 0
expect_status 0 3 "finalize"
result=$(cast call --rpc-url "$rpc" "$arbiter" "result(uint256)(uint8)" 0)
if [ "$result" != "${honest[3]}" ]; then
  echo "WRONG: result(0) is $result, expected ${honest[3]}"
  failures=$((failures + 1))
fi
half=$((bond / 2))
burnt=$((3 * (bond - half)))
if [ -z "$(field withdraw hash)" ]; then
  credit=$(cast call --rpc-url "$rpc" "$arbiter" "creditOf(address)(uint256)" "$me" | cut -d' ' -f1)
  if [ "$credit" != "$((3 * half + bond))" ]; then
    echo "WRONG: credit is $credit, expected $((3 * half + bond)) (half of three slashed bonds and one returned)"
    failures=$((failures + 1))
  fi
fi
send "withdraw" 0 "$arbiter" "withdraw()"
burned=$(cast call --rpc-url "$rpc" "$arbiter" "burned()(uint256)" | cut -d' ' -f1)
balance=$(cast balance --rpc-url "$rpc" "$arbiter")
if [ "$burned" != "$burnt" ] || [ "$balance" != "$burnt" ]; then
  echo "WRONG: burned() is $burned and the arbiter holds $balance wei, expected $burnt for both"
  failures=$((failures + 1))
fi
echo "sepolia_run: $failures wrong result(s)"
[ $failures -eq 0 ]
