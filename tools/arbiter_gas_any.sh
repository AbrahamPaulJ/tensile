#!/bin/bash
# Real transactions of a generated Arbiter on a local chain (anvil, Osaka), reference sample 0, for
# a model with any number of layers. The general form of tools/arbiter_gas.sh: on the shipped
# two-layer model it sends the same transactions in the same order (and must give the same gas).
#
#   claim (honest); then one more claim per layer with output 0 of that layer wrong, and one with
#   the class wrong; challengeNeuronK against each (it succeeds and slashes); challengeClass;
#   finalize of the honest claim after the window (anvil's clock is moved); withdraw.
#
# The first slash of a run also pays for two storage slots that go from zero to non-zero: the
# challenger's credit and the contract's `burned` total (17,100 gas more each than a later write).
# In the sequence above that falls on challengeNeuron1 alone, which is how docs/numbers.md reports
# it. To compare the layers with each other, every challenge is therefore measured a second time
# ("later ..."), against a second set of false claims, when both slots are already non-zero: the
# state every challenge but the very first one finds if the challenger has slashed before.
#
# For each call: `cast estimate`, the gasUsed of the mined `cast send`, the execution gas (gasUsed
# minus 21,000 and the calldata; tools/execution_gas.py) and the calldata size. The claim's state
# is read back after every step and the script fails if it is not what the protocol says.
#
# usage: arbiter_gas_any.sh <fe-out-dir> <ref-dir> <tree.json> [record.json] [port]
#   <fe-out-dir>  a directory with Arbiter.bin from `fe build`
#   <ref-dir>     model.json, inputs.bin, trace.jsonl of the model
#   record.json   if given, the measurements are also written there as JSON
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ $# -lt 3 ] || [ ! -f "$1/Arbiter.bin" ]; then
  echo "usage: arbiter_gas_any.sh <fe-out-dir> <ref-dir> <tree.json> [record.json] [port]" >&2
  exit 2
fi
out="$1"
ref="$2"
tree="$3"
record="$4"
port="${5:-18611}"
rpc="http://127.0.0.1:$port"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
me=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
bond=1000000000000000000
window=3600
rows="$(mktemp)"
anvil --hardfork osaka --port $port > /dev/null 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null; rm -f "$rows"' EXIT
sleep 2
failures=0
args() { python3 "$root/tools/model_args.py" --ref "$ref" --tree "$tree" "$@"; }

json_field() {
  python3 -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

# measure <label> <value-wei> <address> <signature> <args...>: estimate, send, print a row.
measure() {
  local label="$1" value="$2" address="$3" signature="$4"
  shift 4
  local estimate
  estimate=$(cast estimate --rpc-url $rpc --from $me --value "$value" "$address" "$signature" "$@" 2>&1 | head -1)
  local calldata
  calldata=$(cast calldata "$signature" "$@")
  local used
  used=$(cast send --rpc-url $rpc --private-key $key --value "$value" "$address" "$signature" "$@" --json \
    | python3 -c 'import json, sys; print(int(json.load(sys.stdin)["gasUsed"], 16))')
  local execution
  execution=$(python3 "$root/tools/execution_gas.py" "$calldata" "$used")
  local bytes=$(( (${#calldata} - 2) / 2 ))
  printf '%-34s %9s %9s %9s %6d bytes of calldata\n' "$label" "$estimate" "$used" "$execution" $bytes
  echo "$label|$signature|$estimate|$used|$execution|$bytes" >> "$rows"
}

# expect_status <claim id> <status code> <what>
expect_status() {
  local got
  got=$(cast call --rpc-url $rpc "$arbiter" "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)" "$1" | tail -1)
  if [ "$got" != "$2" ]; then
    echo "WRONG: claim $1 has status $got after $3, expected $2"
    failures=$((failures + 1))
  fi
}

layers=$(args layers)
root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$tree")
ctor=$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" $bond $window)
arbiter=$(cast send --rpc-url $rpc --private-key $key --create "0x$(cat "$out/Arbiter.bin")${ctor#0x}" --json | json_field contractAddress)
runtime=$(( ($(cast code --rpc-url $rpc "$arbiter" | wc -c) - 3) / 2 ))
echo "Arbiter at $arbiter, runtime $runtime bytes, $layers layers, root $root_hash, bond $bond wei, window $window s"

claim_sig="$(args signature claim)"
class_sig="$(args signature class)"
mapfile -t honest < <(args claim)
printf '%-34s %9s %9s %9s\n' call estimate gasUsed execution
measure "claim" $bond "$arbiter" "$claim_sig" "${honest[@]}"
expect_status 0 1 "claim"
# One more claim per layer, each with output 0 of that layer wrong, then one with a wrong class;
# their cost is the same as the honest claim's.
for k in $(seq 1 $layers); do
  mapfile -t bad < <(args claim --mutate out:$k:0)
  cast send --rpc-url $rpc --private-key $key --value $bond "$arbiter" "$claim_sig" "${bad[@]}" --json > /dev/null
done
mapfile -t bad_class < <(args claim --mutate class)
cast send --rpc-url $rpc --private-key $key --value $bond "$arbiter" "$claim_sig" "${bad_class[@]}" --json > /dev/null

for k in $(seq 1 $layers); do
  mapfile -t vectors < <(args challenge $k --mutate out:$k:0)
  mapfile -t neuron < <(args proof $k 0)
  inputs=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["layers"][int(sys.argv[2]) - 1]["in"])' "$ref/model.json" $k)
  measure "challengeNeuron$k ($inputs inputs)" 0 "$arbiter" "$(args signature challenge $k)" \
    $k 0 "${vectors[0]}" "${vectors[1]}" "${neuron[0]}" "${neuron[1]}" "${neuron[2]}"
  expect_status $k 2 "challengeNeuron$k"
done
class_id=$((layers + 1))
logits_index=$layers
measure "challengeClass" 0 "$arbiter" "$class_sig" $class_id "${bad_class[$logits_index]}"
expect_status $class_id 2 "challengeClass"

# The second set: the same false claims again (ids layers + 2 onwards), challenged the same way.
for k in $(seq 1 $layers); do
  mapfile -t bad < <(args claim --mutate out:$k:0)
  cast send --rpc-url $rpc --private-key $key --value $bond "$arbiter" "$claim_sig" "${bad[@]}" --json > /dev/null
done
cast send --rpc-url $rpc --private-key $key --value $bond "$arbiter" "$claim_sig" "${bad_class[@]}" --json > /dev/null
for k in $(seq 1 $layers); do
  mapfile -t vectors < <(args challenge $k --mutate out:$k:0)
  mapfile -t neuron < <(args proof $k 0)
  inputs=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["layers"][int(sys.argv[2]) - 1]["in"])' "$ref/model.json" $k)
  id=$((layers + 1 + k))
  measure "later challengeNeuron$k ($inputs inputs)" 0 "$arbiter" "$(args signature challenge $k)" \
    $id 0 "${vectors[0]}" "${vectors[1]}" "${neuron[0]}" "${neuron[1]}" "${neuron[2]}"
  expect_status $id 2 "the later challengeNeuron$k"
done
class_id=$((2 * layers + 2))
measure "later challengeClass" 0 "$arbiter" "$class_sig" $class_id "${bad_class[$logits_index]}"
expect_status $class_id 2 "the later challengeClass"

# A challenge that proves nothing reverts with NoFraud and costs only gas: shown, not measured.
if cast call --rpc-url $rpc "$arbiter" "$class_sig" 0 "${honest[$logits_index]}" > /dev/null 2>&1; then
  echo "WRONG: a class challenge of the honest claim did not revert"
  failures=$((failures + 1))
fi

cast rpc --rpc-url $rpc evm_increaseTime $((window + 1)) > /dev/null
cast rpc --rpc-url $rpc evm_mine > /dev/null
measure "finalize" 0 "$arbiter" "finalize(uint256)" 0
expect_status 0 3 "finalize"
result=$(cast call --rpc-url $rpc "$arbiter" "result(uint256)(uint8)" 0)
class_index=$((layers + 1))
if [ "$result" != "${honest[$class_index]}" ]; then
  echo "WRONG: result(0) is $result, expected ${honest[$class_index]}"
  failures=$((failures + 1))
fi
# A slash credits half the bond and burns the rest; finalising credits the whole bond.
slashes=$((2 * layers + 2))
credit=$(cast call --rpc-url $rpc "$arbiter" "creditOf(address)(uint256)" $me | cut -d' ' -f1)
expected_credit=$(python3 -c 'import sys; b, n = int(sys.argv[1]), int(sys.argv[2]); print(n * (b // 2) + b)' $bond $slashes)
expected_burn=$(python3 -c 'import sys; b, n = int(sys.argv[1]), int(sys.argv[2]); print(n * (b - b // 2))' $bond $slashes)
if [ "$credit" != "$expected_credit" ]; then
  echo "WRONG: credit is $credit, expected $expected_credit (half of $slashes slashed bonds and one returned)"
  failures=$((failures + 1))
fi
burned=$(cast call --rpc-url $rpc "$arbiter" "burned()(uint256)" | cut -d' ' -f1)
if [ "$burned" != "$expected_burn" ]; then
  echo "WRONG: burned is $burned, expected $expected_burn (half of $slashes slashed bonds)"
  failures=$((failures + 1))
fi
measure "withdraw ($((slashes + 1)) credits)" 0 "$arbiter" "withdraw()"
balance=$(cast balance --rpc-url $rpc "$arbiter")
if [ "$balance" != "$expected_burn" ]; then
  echo "WRONG: the arbiter holds $balance wei, expected only the burned $expected_burn"
  failures=$((failures + 1))
fi
echo "arbiter_gas_any: $failures wrong result(s)"
if [ -n "$record" ] && [ $failures -eq 0 ]; then
  python3 - "$rows" "$record" "$ref/model.json" "$runtime" "$(fe --version)" <<'EOF'
import json, sys
rows_path, record_path, model_path, runtime, fe_version = sys.argv[1:6]
model = json.load(open(model_path, encoding="utf-8"))
calls = []
for line in open(rows_path, encoding="utf-8"):
    label, signature, estimate, used, execution, size = line.rstrip("\n").split("|")
    calls.append({"call": label, "signature": signature, "estimate": int(estimate), "gas_used": int(used),
                  "execution": execution if execution == "floor" else int(execution),
                  "calldata_bytes": int(size)})
record = {"model": model["name"], "fe": fe_version, "chain": "anvil, Osaka", "sample": 0,
          "arbiter_runtime_bytes": int(runtime), "calls": calls}
with open(record_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
EOF
fi
[ $failures -eq 0 ]
