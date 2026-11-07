#!/bin/bash
# Real transactions on a local chain: deploys the Fe contracts (and the Solidity twin, if its
# Foundry artifact is given) to anvil (Osaka) and measures, for reference sample 0:
#   - Network.classify and Network.trace,
#   - NeuronCheck.neuron1 (layer 1, neuron 0, 784 inputs) and neuron2 (layer 2, neuron 0, 32 inputs),
#   - NeuronTwin.neuron1, neuron1Unrolled, neuron2 and neuron2Unrolled with the same calldata.
# For each call: the result is checked against the reference, then `cast estimate` and the gasUsed
# of a mined `cast send` are printed, with the deployed code size. "execution" is gasUsed minus the
# intrinsic cost of the transaction (21,000 plus the calldata; see tools/execution_gas.py); "frame"
# is the gas of the called contract's frame as `cast call --trace` reports it.
# usage: gas_report.sh <fe-out-dir> [NeuronTwin.json]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ $# -lt 1 ] || [ ! -d "$1" ]; then
  echo "usage: gas_report.sh <fe-out-dir> [NeuronTwin.json]" >&2
  exit 2
fi
out="$1"
twin_json="$2"
port=8572
rpc="http://127.0.0.1:$port"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
anvil --hardfork osaka --port $port > /dev/null 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null' EXIT
sleep 2
failures=0

# deploy <creation code hex without 0x> [constructor args hex without 0x]: prints the address.
deploy() {
  cast send --rpc-url $rpc --private-key $key --create "0x$1$2" --json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["contractAddress"])'
}

code_size() {
  echo $(( ($(cast code --rpc-url $rpc "$1" | wc -c) - 3) / 2 ))
}

# measure <label> <address> <signature with return type> <expected or -> <args...>: checks the
# result against <expected> (the output of `cast call`), prints a row.
measure() {
  local label="$1" address="$2" signature="$3" expected="$4"
  shift 4
  local inputs="${signature%(*}"
  local got
  got=$(cast call --rpc-url $rpc "$address" "$signature" "$@" 2>&1)
  local check="ok"
  if [ "$expected" != "-" ] && [ "$got" != "$expected" ]; then
    check="WRONG ($got, expected $expected)"
    failures=$((failures + 1))
  fi
  local estimate
  estimate=$(cast estimate --rpc-url $rpc "$address" "$inputs" "$@" 2>&1 | head -1)
  local calldata
  calldata=$(cast calldata "$inputs" "$@")
  local used
  used=$(cast send --rpc-url $rpc --private-key $key "$address" "$inputs" "$@" --json \
    | python3 -c 'import json, sys; print(int(json.load(sys.stdin)["gasUsed"], 16))')
  local execution
  execution=$(python3 "$root/tools/execution_gas.py" "$calldata" "$used")
  # The gas of the call frame itself, from the trace of the same call: visible even when the
  # EIP-7623 calldata floor is what the transaction is charged.
  local frame
  frame=$(cast call --rpc-url $rpc --trace "$address" "$inputs" "$@" 2>&1 | grep -m 1 -oE '\[[0-9]+\]' | tr -d '[]')
  printf '%-26s %9s %9s %9s %9s %7s %s\n' "$label" "$estimate" "$used" "$execution" "$frame" \
    "$(code_size "$address")" "$check"
}

printf '%-26s %9s %9s %9s %9s %7s %s\n' call estimate gasUsed execution frame bytes result
low=$(deploy "$(cat "$out/RowsLow.bin")")
high=$(deploy "$(cat "$out/RowsHigh.bin")")
args=$(cast abi-encode "constructor(address,address)" "$low" "$high")
network=$(deploy "$(cat "$out/Network.bin")" "${args#0x}")
check=$(deploy "$(cat "$out/NeuronCheck.bin")")
echo "RowsLow $(code_size "$low") bytes, RowsHigh $(code_size "$high") bytes"

mapfile -t n1 < <(python3 "$root/tools/neuron_args.py" 1 0 0)
mapfile -t n2 < <(python3 "$root/tools/neuron_args.py" 2 0 0)
class=$(python3 -c 'print(open("'"$root"'/ref/out/pred.bin", "rb").read(1)[0])')
measure "Network.classify" "$network" "classify(uint256[25])(uint256)" "$class" "${n1[1]}"
measure "Network.trace" "$network" "trace(uint256[25])(uint256[1],int32[10])" "-" "${n1[1]}"
# The raw return of trace: the hidden word, then the ten logits as 256-bit two's complement.
python3 "$root/tools/check_trace.py" "$(cast call --rpc-url $rpc "$network" "trace(uint256[25])" "${n1[1]}")" \
  "$root/ref/out/trace.jsonl" || failures=$((failures + 1))
measure "NeuronCheck.neuron1 (Fe)" "$check" "neuron1(uint256[25],uint256[25],int32)(int8)" "${n1[3]}" "${n1[0]}" "${n1[1]}" "${n1[2]}"
measure "NeuronCheck.neuron2 (Fe)" "$check" "neuron2(uint256[1],uint256[1],int32)(int32)" "${n2[3]}" "${n2[0]}" "${n2[1]}" "${n2[2]}"

if [ -n "$twin_json" ] && [ -f "$twin_json" ]; then
  twin_code=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["bytecode"]["object"][2:])' "$twin_json")
  twin=$(deploy "$twin_code")
  measure "NeuronTwin.neuron1 (Sol)" "$twin" "neuron1(uint256[25],uint256[25],int32)(int8)" "${n1[3]}" "${n1[0]}" "${n1[1]}" "${n1[2]}"
  measure "NeuronTwin.neuron1Unrolled" "$twin" "neuron1Unrolled(uint256[25],uint256[25],int32)(int8)" "${n1[3]}" "${n1[0]}" "${n1[1]}" "${n1[2]}"
  measure "NeuronTwin.neuron2 (Sol)" "$twin" "neuron2(uint256[1],uint256[1],int32)(int32)" "${n2[3]}" "${n2[0]}" "${n2[1]}" "${n2[2]}"
  measure "NeuronTwin.neuron2Unrolled" "$twin" "neuron2Unrolled(uint256[1],uint256[1],int32)(int32)" "${n2[3]}" "${n2[0]}" "${n2[1]}" "${n2[2]}"
fi
echo "gas_report: $failures wrong result(s)"
[ $failures -eq 0 ]
