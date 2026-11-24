#!/bin/bash
# The whole use on a local chain (anvil, Osaka): an operator scores two accounts through the
# arbiter and `RiskGate` lets one of them in.
#
#   deploy Arbiter and RiskGate; claim the first reference sample the model clears (class 0) and
#   the first it flags (class 1); try to score from the open claims (must revert); move the clock
#   past the window and finalize both; score account A from the clear claim and account B from the
#   flagged one; try to clear B with A's claim (must revert: other features); A enters; B and an
#   unscored account are refused.
#
# Every result is read back and the script fails if one is not what the contracts promise. Gas is
# the gasUsed of the mined transaction.
#
# The script of examples/address-risk with the tree-ensemble arbiter: only the claim arguments
# (tools/trees_args.py) and the commitment file (build/forest.json) differ.
#
# usage: gate_anvil.sh <fe-out-dir> <ref-dir> <forest.json> [record.json] [port]
#   <fe-out-dir>  a directory with Arbiter.bin and RiskGate.bin from `fe build`
#   <ref-dir>     model.json, inputs.bin, pred.bin, trace.jsonl of the forest
#
# The keys are anvil's published development keys: they hold nothing anywhere else.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
if [ $# -lt 3 ] || [ ! -f "$1/RiskGate.bin" ] || [ ! -f "$1/Arbiter.bin" ]; then
  echo "usage: gate_anvil.sh <fe-out-dir> <ref-dir> <forest.json> [record.json] [port]" >&2
  exit 2
fi
out="$1"
ref="$2"
tree="$3"
record="$4"
port="${5:-18633}"
rpc="http://127.0.0.1:$port"
operator_key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
operator=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
a_key=0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d
a=0x70997970C51812dc3A010C7d01b50e0d17dc79C8
b_key=0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a
b=0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC
bond=1000000000000000000
window=3600
rows="$(mktemp)"
anvil --hardfork osaka --port $port > /dev/null 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null; rm -f "$rows"' EXIT
sleep 2
failures=0
args() { python3 "$root/tools/trees_args.py" --ref "$ref" --forest "$tree" "$@"; }
wrong() {
  echo "WRONG: $1"
  failures=$((failures + 1))
}

# send <label> <key> <value-wei> <address> <signature> <args...>: mines the call, records its gas.
send() {
  local label="$1" key="$2" value="$3" address="$4" signature="$5"
  shift 5
  local used
  used=$(cast send --rpc-url $rpc --private-key "$key" --value "$value" "$address" "$signature" "$@" --json \
    | python3 -c 'import json, sys; r = json.load(sys.stdin); print(int(r["gasUsed"], 16) if r["status"] == "0x1" else "reverted")')
  local calldata
  calldata=$(cast calldata "$signature" "$@")
  printf '%-34s %9s gas %6d bytes of calldata\n' "$label" "$used" $(( (${#calldata} - 2) / 2 ))
  echo "$label|$signature|$used|$(( (${#calldata} - 2) / 2 ))" >> "$rows"
  case "$used" in
    ''|reverted) wrong "$label did not go through" ;;
  esac
}
# reverts <what> <from> <address> <signature> <args...>: the call must revert.
reverts() {
  local what="$1" from="$2" address="$3" signature="$4"
  shift 4
  if cast call --rpc-url $rpc --from "$from" "$address" "$signature" "$@" > /dev/null 2>&1; then
    wrong "$what did not revert"
  else
    echo "reverts, as it must: $what"
  fi
}
# expect <what> <want> <address> <signature> <args...>: a view must return <want>.
expect() {
  local what="$1" want="$2" address="$3" signature="$4"
  shift 4
  local got
  got=$(cast call --rpc-url $rpc "$address" "$signature" "$@" | cut -d' ' -f1)
  [ "$got" = "$want" ] || wrong "$what is $got, expected $want"
}

# The first reference sample of each class, from the reference's own predictions.
read -r clear flagged < <(python3 - "$ref" <<'EOF'
import sys
pred = open(sys.argv[1] + "/pred.bin", "rb").read()[:1000]
print(pred.index(0), pred.index(1))
EOF
)
root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$tree")
deploy() {
  cast send --rpc-url $rpc --private-key $operator_key --create "$1" --json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["contractAddress"])'
}
ctor=$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" $bond $window)
arbiter=$(deploy "0x$(cat "$out/Arbiter.bin")${ctor#0x}")
ctor=$(cast abi-encode "constructor(address,address)" "$arbiter" $operator)
gate=$(deploy "0x$(cat "$out/RiskGate.bin")${ctor#0x}")
gate_bytes=$(( ($(cast code --rpc-url $rpc "$gate" | wc -c) - 3) / 2 ))
echo "Arbiter at $arbiter, RiskGate at $gate ($gate_bytes bytes deployed), operator $operator"
echo "reference sample $clear is cleared by the forest, sample $flagged is flagged"

claim_sig="$(args signature claim)"
score_sig="score(address,uint256,uint256[1])"
mapfile -t claim_clear < <(args --sample "$clear" claim)
mapfile -t claim_flagged < <(args --sample "$flagged" claim)
x_clear="${claim_clear[0]}"
x_flagged="${claim_flagged[0]}"
send "claim (cleared sample)" $operator_key $bond "$arbiter" "$claim_sig" "${claim_clear[@]}"
send "claim (flagged sample)" $operator_key $bond "$arbiter" "$claim_sig" "${claim_flagged[@]}"
reverts "score from a claim that is still open" $operator "$gate" "$score_sig" $a 0 "$x_clear"

cast rpc --rpc-url $rpc evm_increaseTime $((window + 1)) > /dev/null
cast rpc --rpc-url $rpc evm_mine > /dev/null
send "finalize (cleared sample)" $operator_key 0 "$arbiter" "finalize(uint256)" 0
send "finalize (flagged sample)" $operator_key 0 "$arbiter" "finalize(uint256)" 1
expect "result(0)" 0 "$arbiter" "result(uint256)(uint8)" 0
expect "result(1)" 1 "$arbiter" "result(uint256)(uint8)" 1

reverts "score by an account that is not the operator" $a "$gate" "$score_sig" $a 0 "$x_clear"
reverts "clearing B with the claim about A's features" $operator "$gate" "$score_sig" $b 0 "$x_flagged"
send "score A (cleared)" $operator_key 0 "$gate" "$score_sig" $a 0 "$x_clear"
send "score B (flagged)" $operator_key 0 "$gate" "$score_sig" $b 1 "$x_flagged"
expect "standingOf(A)" 1 "$gate" "standingOf(address)(uint8)" $a
expect "standingOf(B)" 2 "$gate" "standingOf(address)(uint8)" $b
expect "standingOf(operator)" 0 "$gate" "standingOf(address)(uint8)" $operator

send "enter (A, cleared)" $a_key 0 "$gate" "enter()"
expect "entries()" 1 "$gate" "entries()(uint256)"
reverts "enter by B, flagged" $b "$gate" "enter()"
reverts "enter by an unscored account" $operator "$gate" "enter()"
expect "entries() after the refusals" 1 "$gate" "entries()(uint256)"

echo "gate_anvil: $failures wrong result(s)"
if [ -n "$record" ] && [ $failures -eq 0 ]; then
  python3 - "$rows" "$record" "$ref/model.json" "$gate_bytes" "$(fe --version)" "$clear" "$flagged" <<'EOF'
import json, sys
rows_path, record_path, model_path, gate_bytes, fe_version, clear, flagged = sys.argv[1:8]
model = json.load(open(model_path, encoding="utf-8"))
calls = []
for line in open(rows_path, encoding="utf-8"):
    label, signature, used, size = line.rstrip("\n").split("|")
    calls.append({"call": label, "signature": signature, "gas_used": int(used), "calldata_bytes": int(size)})
record = {"model": model["name"], "fe": fe_version, "chain": "anvil, Osaka",
          "cleared_sample": int(clear), "flagged_sample": int(flagged),
          "gate_deployed_bytes": int(gate_bytes), "calls": calls}
with open(record_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
EOF
fi
[ $failures -eq 0 ]
