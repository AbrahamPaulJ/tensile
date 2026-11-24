#!/bin/bash
# Real transactions of a tree-ensemble Arbiter (docs/spec-trees.md) on a local chain (anvil, Osaka),
# reference sample 0. The tree counterpart of tools/arbiter_gas_any.sh, same measurements:
#
#   claim (honest); claims with the output of tree 5, of the last tree, and the class wrong;
#   challengeTree against the first two and challengeClass against the third (each succeeds and
#   slashes); the same three false claims again, challenged again ("later": the challenger's
#   credit and the `burned` total are already non-zero, the state every slash but the very first
#   finds); a tree and a class challenge of the honest claim must revert; finalize of the honest
#   claim after the window (anvil's clock is moved); withdraw.
#
# For each call: `cast estimate`, the gasUsed of the mined `cast send`, the execution gas (gasUsed
# minus 21,000 and the calldata; tools/execution_gas.py) and the calldata size. The claim's state
# is read back after every step and the script fails if it is not what the protocol says.
#
# usage: trees_gas.sh <fe-out-dir> <ref-dir> <forest.json> [record.json] [port]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ $# -lt 3 ] || [ ! -f "$1/Arbiter.bin" ]; then
  echo "usage: trees_gas.sh <fe-out-dir> <ref-dir> <forest.json> [record.json] [port]" >&2
  exit 2
fi
out="$1"
ref="$2"
forest="$3"
record="$4"
port="${5:-18631}"
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
args() { python3 "$root/tools/trees_args.py" --ref "$ref" --forest "$forest" "$@"; }
json_field() { python3 -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

measure() {
  local label="$1" value="$2" address="$3" signature="$4"
  shift 4
  local estimate calldata used execution
  estimate=$(cast estimate --rpc-url $rpc --from $me --value "$value" "$address" "$signature" "$@" 2>&1 | head -1)
  calldata=$(cast calldata "$signature" "$@")
  used=$(cast send --rpc-url $rpc --private-key $key --value "$value" "$address" "$signature" "$@" --json \
    | python3 -c 'import json, sys; print(int(json.load(sys.stdin)["gasUsed"], 16))')
  execution=$(python3 "$root/tools/execution_gas.py" "$calldata" "$used")
  local bytes=$(( (${#calldata} - 2) / 2 ))
  printf '%-40s %9s %9s %9s %6d bytes of calldata\n' "$label" "$estimate" "$used" "$execution" $bytes
  echo "$label|$signature|$estimate|$used|$execution|$bytes" >> "$rows"
}

expect_status() {
  local got
  got=$(cast call --rpc-url $rpc "$arbiter" "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)" "$1" | tail -1)
  if [ "$got" != "$2" ]; then
    echo "WRONG: claim $1 has status $got after $3, expected $2"
    failures=$((failures + 1))
  fi
}

root_hash=$(args root)
trees=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["shape"]["trees"])' "$forest")
last=$((trees - 1))
ctor=$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" $bond $window)
arbiter=$(cast send --rpc-url $rpc --private-key $key --create "0x$(cat "$out/Arbiter.bin")${ctor#0x}" --json | json_field contractAddress)
runtime=$(( ($(cast code --rpc-url $rpc "$arbiter" | wc -c) - 3) / 2 ))
echo "Arbiter at $arbiter, runtime $runtime bytes, $trees trees, root $root_hash, bond $bond wei, window $window s"

claim_sig="$(args signature claim)"
tree_sig="$(args signature tree)"
class_sig="$(args signature class)"
mapfile -t honest < <(args claim)
proof_len() { python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["trees"][int(sys.argv[2])]["proof"]))' "$forest" "$1"; }
printf '%-40s %9s %9s %9s\n' call estimate gasUsed execution
measure "claim" $bond "$arbiter" "$claim_sig" "${honest[@]}"
expect_status 0 1 "claim"

# post_false: the three false claims, in order tree 5, last tree, class.
post_false() {
  for mutation in tree:5 tree:$last class; do
    mapfile -t bad < <(args --mutate $mutation claim)
    cast send --rpc-url $rpc --private-key $key --value $bond "$arbiter" "$claim_sig" "${bad[@]}" --json > /dev/null
  done
}
# challenge_false <first id> <label prefix>
challenge_false() {
  local id=$1 prefix="$2"
  for t in 5 $last; do
    mapfile -t c < <(args --mutate tree:$t challenge $t)
    measure "${prefix}challengeTree (tree $t, proof $(proof_len $t))" 0 "$arbiter" "$tree_sig" $id "${c[@]}"
    expect_status $id 2 "${prefix}challengeTree $t"
    id=$((id + 1))
  done
  measure "${prefix}challengeClass" 0 "$arbiter" "$class_sig" $id "$(args --mutate class outputs)"
  expect_status $id 2 "${prefix}challengeClass"
}
post_false
challenge_false 1 ""
post_false
challenge_false 4 "later "

# Challenges that prove nothing revert (NoFraud) and cost only gas: shown, not measured.
mapfile -t c < <(args challenge 5)
if cast call --rpc-url $rpc "$arbiter" "$tree_sig" 0 "${c[@]}" > /dev/null 2>&1; then
  echo "WRONG: a tree challenge of the honest claim did not revert"
  failures=$((failures + 1))
fi
if cast call --rpc-url $rpc "$arbiter" "$class_sig" 0 "${honest[1]}" > /dev/null 2>&1; then
  echo "WRONG: a class challenge of the honest claim did not revert"
  failures=$((failures + 1))
fi

cast rpc --rpc-url $rpc evm_increaseTime $((window + 1)) > /dev/null
cast rpc --rpc-url $rpc evm_mine > /dev/null
measure "finalize" 0 "$arbiter" "finalize(uint256)" 0
expect_status 0 3 "finalize"
result=$(cast call --rpc-url $rpc "$arbiter" "result(uint256)(uint8)" 0)
if [ "$result" != "${honest[2]}" ]; then
  echo "WRONG: result(0) is $result, expected ${honest[2]}"
  failures=$((failures + 1))
fi
slashes=6
credit=$(cast call --rpc-url $rpc "$arbiter" "creditOf(address)(uint256)" $me | cut -d' ' -f1)
expected_credit=$(python3 -c 'import sys; b, n = int(sys.argv[1]), int(sys.argv[2]); print(n * (b // 2) + b)' $bond $slashes)
expected_burn=$(python3 -c 'import sys; b, n = int(sys.argv[1]), int(sys.argv[2]); print(n * (b - b // 2))' $bond $slashes)
if [ "$credit" != "$expected_credit" ]; then
  echo "WRONG: credit is $credit, expected $expected_credit"
  failures=$((failures + 1))
fi
burned=$(cast call --rpc-url $rpc "$arbiter" "burned()(uint256)" | cut -d' ' -f1)
if [ "$burned" != "$expected_burn" ]; then
  echo "WRONG: burned is $burned, expected $expected_burn"
  failures=$((failures + 1))
fi
measure "withdraw ($((slashes + 1)) credits)" 0 "$arbiter" "withdraw()"
balance=$(cast balance --rpc-url $rpc "$arbiter")
if [ "$balance" != "$expected_burn" ]; then
  echo "WRONG: the arbiter holds $balance wei, expected only the burned $expected_burn"
  failures=$((failures + 1))
fi
echo "trees_gas: $failures wrong result(s)"
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
