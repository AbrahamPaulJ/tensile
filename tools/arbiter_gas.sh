#!/bin/bash
# Real transactions of the Arbiter on a local chain (anvil, Osaka), reference sample 0:
#   claim (honest), a layer 1 neuron challenge, a layer 2 neuron challenge, a class challenge (each
#   against a claim with that one value wrong, so the challenge succeeds and the claim is slashed),
#   finalize after the window (anvil's clock is moved), withdraw.
# For each call: `cast estimate`, then the gasUsed of a mined `cast send`, then the execution gas
# (gasUsed minus 21,000 and the calldata; tools/execution_gas.py) and the deployed code size. The
# claim's state is read back after every step and the script fails if it is not what the protocol
# says (open, slashed, final, the credits).
# usage: arbiter_gas.sh <fe-out-dir>      (a directory with Arbiter.bin from `fe build`)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ $# -ne 1 ] || [ ! -f "$1/Arbiter.bin" ]; then
  echo "usage: arbiter_gas.sh <fe-out-dir>" >&2
  exit 2
fi
out="$1"
port=8573
rpc="http://127.0.0.1:$port"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
me=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
bond=1000000000000000000
window=3600
anvil --hardfork osaka --port $port > /dev/null 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null' EXIT
sleep 2
failures=0

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
  printf '%-30s %9s %9s %9s %6d bytes of calldata\n' "$label" "$estimate" "$used" "$execution" \
    $(( (${#calldata} - 2) / 2 ))
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

root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$root/build/tree.json")
args=$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" $bond $window)
arbiter=$(cast send --rpc-url $rpc --private-key $key --create "0x$(cat "$out/Arbiter.bin")${args#0x}" --json | json_field contractAddress)
echo "Arbiter at $arbiter, runtime $(( ($(cast code --rpc-url $rpc "$arbiter" | wc -c) - 3) / 2 )) bytes, root $root_hash, bond $bond wei, window $window s"

mapfile -t honest < <(python3 "$root/tools/claim_args.py" claim)
mapfile -t bad_hidden < <(python3 "$root/tools/claim_args.py" claim --mutate hidden:0)
mapfile -t bad_logit < <(python3 "$root/tools/claim_args.py" claim --mutate logit:0)
mapfile -t bad_class < <(python3 "$root/tools/claim_args.py" claim --mutate class)
mapfile -t n1 < <(python3 "$root/tools/claim_args.py" proof 1 0)
mapfile -t n2 < <(python3 "$root/tools/claim_args.py" proof 2 0)
claim_sig="claim(uint256[25],uint256[1],int32[10],uint8)"

printf '%-30s %9s %9s %9s\n' call estimate gasUsed execution
measure "claim (honest, sample 0)" $bond "$arbiter" "$claim_sig" "${honest[@]}"
expect_status 0 1 "claim"
# Three more claims, each with one wrong value; their cost is the same as the honest claim's.
for bad in bad_hidden bad_logit bad_class; do
  eval 'cast send --rpc-url $rpc --private-key $key --value $bond "$arbiter" "$claim_sig" "${'$bad'[@]}" --json > /dev/null'
done
measure "challengeNeuron1 (layer 1, j=0)" 0 "$arbiter" \
  "challengeNeuron1(uint256,uint256,uint256[25],uint256[1],uint256[25],int32,bytes32[])" \
  1 0 "${bad_hidden[0]}" "${bad_hidden[1]}" "${n1[0]}" "${n1[1]}" "${n1[2]}"
expect_status 1 2 "challengeNeuron1"
measure "challengeNeuron2 (layer 2, j=0)" 0 "$arbiter" \
  "challengeNeuron2(uint256,uint256,uint256[1],int32[10],uint256[1],int32,bytes32[])" \
  2 0 "${bad_logit[1]}" "${bad_logit[2]}" "${n2[0]}" "${n2[1]}" "${n2[2]}"
expect_status 2 2 "challengeNeuron2"
measure "challengeClass" 0 "$arbiter" "challengeClass(uint256,int32[10])" 3 "${bad_class[2]}"
expect_status 3 2 "challengeClass"

# A challenge that proves nothing reverts with NoFraud and costs only gas: shown, not measured.
if cast call --rpc-url $rpc "$arbiter" "challengeClass(uint256,int32[10])" 0 "${honest[2]}" > /dev/null 2>&1; then
  echo "WRONG: a class challenge of the honest claim did not revert"
  failures=$((failures + 1))
fi

cast rpc --rpc-url $rpc evm_increaseTime $((window + 1)) > /dev/null
cast rpc --rpc-url $rpc evm_mine > /dev/null
measure "finalize (after the window)" 0 "$arbiter" "finalize(uint256)" 0
expect_status 0 3 "finalize"
result=$(cast call --rpc-url $rpc "$arbiter" "result(uint256)(uint8)" 0)
if [ "$result" != "${honest[3]}" ]; then
  echo "WRONG: result(0) is $result, expected ${honest[3]}"
  failures=$((failures + 1))
fi
credit=$(cast call --rpc-url $rpc "$arbiter" "creditOf(address)(uint256)" $me | cut -d' ' -f1)
if [ "$credit" != "$((4 * bond))" ]; then
  echo "WRONG: credit is $credit, expected $((4 * bond)) (three slashed bonds and one returned)"
  failures=$((failures + 1))
fi
measure "withdraw (4 bonds)" 0 "$arbiter" "withdraw()"
balance=$(cast balance --rpc-url $rpc "$arbiter")
if [ "$balance" != "0" ]; then
  echo "WRONG: the arbiter still holds $balance wei"
  failures=$((failures + 1))
fi
echo "arbiter_gas: $failures wrong result(s)"
[ $failures -eq 0 ]
