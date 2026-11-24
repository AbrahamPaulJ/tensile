#!/bin/bash
# The live fee-tier system end to end on a local chain (anvil), with the mainnet pool replaced by
# examples/amm-fee/data/ticks.csv (its line i is block 23,850,000 + 50 i, the relay's grid):
#
#   1. tools/sepolia_fee_deploy.sh deploys Rows0, Network, Arbiter, TickRelay and FeeOracle as on
#      Sepolia (window 600 s here);
#   2. the keeper (tools/fee_keeper.mjs) catches up 201 ticks over three runs of at most 72, and
#      claims the newest window; the oracle's input must equal the moves of ticks.csv;
#   3. a run right after does nothing (idempotent);
#   4. a second account plants a false claim; the next keeper run challenges it and it is slashed;
#   5. after the window: the keeper finalises the honest claim, applies it, and fee() becomes the
#      fee of the class `Network.classify` gives for that window; credits are withdrawn;
#   6. a skipped stretch: mainnet moves 30 ticks ahead and one run relays them all;
#   7. limits: a run with a gas budget too small for a batch stops before sending it and the next
#      run finishes; a run with the base fee above the cap sends nothing and exits 0;
#   8. the watcher (demo/test/tensile-fee-live.mjs) passes, and fails on a copy of ticks.csv with one
#      relayed tick changed (a relay that lied).
#
# usage: fee_live_anvil.sh [port]      needs fe 26.5.1, Foundry, Python 3 and Node 22 (with demo/node_modules)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$HOME/.local/node22/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
port="${1:-18631}"
rpc="http://127.0.0.1:$port"
work="$(mktemp -d)"
anvil --hardfork osaka --port "$port" > "$work/anvil.log" 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null; rm -rf "$work"' EXIT
sleep 2
failures=0
wrong() { echo "WRONG: $1"; failures=$((failures + 1)); }
ticks="$root/examples/amm-fee/data/ticks.csv"
first=23850000
# Anvil's first account is the keeper; the adversary is the second account of anvil's mnemonic.
printf 'private_key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80\n' > "$work/keeper.txt"
adversary_key=$(cast wallet private-key "test test test test test test test test test test test junk" 1)
printf 'private_key=%s\n' "$adversary_key" > "$work/adversary.txt"
export SEPOLIA_RPC="$rpc" CHAIN_ID=31337 RECORD_FILE="$work/record.jsonl" RECORD="$work/record.jsonl"
export MAINNET_FILE="$ticks" MAX_FEE_WEI=100000000000 MIN_BALANCE_WEI=0 GAS_BUDGET=30000000
echo "=== deploy"
SEPOLIA_KEY_FILE="$work/keeper.txt" WINDOW=600 bash "$root/tools/sepolia_fee_deploy.sh" > "$work/deploy.log" 2>&1 \
  || { cat "$work/deploy.log"; echo "deploy failed"; exit 1; }
tail -2 "$work/deploy.log" | sed 's/^/    /'
relay=$(python3 -c 'import json,sys; print(next(json.loads(l)["address"] for l in open(sys.argv[1]) if json.loads(l)["label"]=="TickRelay"))' "$RECORD_FILE")
oracle=$(python3 -c 'import json,sys; print(next(json.loads(l)["address"] for l in open(sys.argv[1]) if json.loads(l)["label"]=="FeeOracle"))' "$RECORD_FILE")
arbiter=$(python3 -c 'import json,sys; print(next(json.loads(l)["address"] for l in open(sys.argv[1]) if json.loads(l)["label"]=="Arbiter"))' "$RECORD_FILE")
network=$(python3 -c 'import json,sys; print(next(json.loads(l)["address"] for l in open(sys.argv[1]) if json.loads(l)["label"]=="Network"))' "$RECORD_FILE")

# keeper <head tick index> [extra env...]: one keeper run with mainnet's safe head at that line of ticks.csv.
keeper() {
  local head=$((first + 50 * $1))
  shift
  env SEPOLIA_KEY_FILE="$work/keeper.txt" MAINNET_HEAD="$head" START_BLOCK=$((first + 50 * 800)) "$@" \
    node "$root/tools/fee_keeper.mjs" > "$work/keeper.log" 2>&1
  local status=$?
  sed 's/^/    /' "$work/keeper.log"
  return $status
}
sent() { grep -o 'done: [0-9]* transaction' "$work/keeper.log" | grep -o '[0-9]*'; }
call() { cast call --rpc-url "$rpc" "$@" | cut -d' ' -f1; }

echo "=== catch-up: 201 ticks due (lines 800 to 1000), at most 72 per run"
for run in 1 2 3; do keeper 1000 || wrong "keeper run $run failed"; done
[ "$(call "$relay" 'relayed()(uint256)')" = 201 ] || wrong "relayed() is $(call "$relay" 'relayed()(uint256)'), expected 201"
[ "$(call "$oracle" 'moves()(uint256)')" = 200 ] || wrong "moves() is not 200"
[ "$(call "$arbiter" 'claimCount()(uint256)')" = 1 ] || wrong "expected one claim after the catch-up"
# The oracle's window ending at move 198 must be the moves of ticks.csv lines 800 + 54 .. 800 + 198.
want=$(python3 -c '
import sys
t = [int(x) for x in open(sys.argv[1])][800:1001]
m = [min(abs(b - a), 127) for a, b in zip(t, t[1:])]
w = bytes(m[198 - 144:198]) + bytes(16)
print(" ".join(str(int.from_bytes(w[i:i + 32], "big")) for i in range(0, 160, 32)))' "$ticks")
got=$(cast call --rpc-url "$rpc" "$oracle" "inputWords(uint256)(uint256[5])" 198 | python3 -c 'import re, sys; print(" ".join(re.findall(r"\d+", re.sub(r"\[[0-9.e]+\]", "", sys.stdin.read()))))')
[ "$got" = "$want" ] && echo "inputWords(198) equals the moves of ticks.csv lines 854 to 998" || wrong "inputWords(198) differs from ticks.csv"

echo "=== a second run at once sends nothing"
keeper 1000 || wrong "the repeated run failed"
[ "$(sent)" = 0 ] || wrong "the repeated run sent $(sent) transaction(s)"

echo "=== a false claim by a second account, then a keeper run"
env ADVERSARY_KEY_FILE="$work/adversary.txt" MAINNET_HEAD=$((first + 50 * 1000)) node "$root/tools/fee_keeper.mjs" --plant layer1:5 --record > "$work/plant.log" 2>&1 \
  || { cat "$work/plant.log"; wrong "planting failed"; }
keeper 1000 || wrong "the watching run failed"
status=$(cast call --rpc-url "$rpc" "$arbiter" "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)" 1 | tail -1)
[ "$status" = 2 ] && echo "claim 1 (planted) is slashed" || wrong "claim 1 has status $status, expected 2 (slashed)"

echo "=== after the window: finalize, apply, withdraw"
cast rpc --rpc-url "$rpc" evm_increaseTime 601 > /dev/null
cast rpc --rpc-url "$rpc" anvil_mine 1 > /dev/null
keeper 1000 || wrong "the finalising run failed"
status=$(cast call --rpc-url "$rpc" "$arbiter" "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)" 0 | tail -1)
[ "$status" = 3 ] || wrong "claim 0 has status $status, expected 3 (final)"
words=$(cast call --rpc-url "$rpc" "$oracle" "inputWords(uint256)(uint256[5])" 198 | tr -d '\n' | sed 's/ \[[^]]*\]//g')
class=$(call "$network" "classify(uint256[5])(uint256)" "$words")
fee=$(call "$oracle" 'fee()(uint32)')
want_fee=$(python3 -c 'import sys; print([500, 3000, 10000][int(sys.argv[1])])' "$class")
[ "$fee" = "$want_fee" ] && echo "fee() is $fee: Network.classify gives tier $class for window 198" || wrong "fee() is $fee, expected $want_fee (tier $class)"
credit=$(call "$arbiter" "creditOf(address)(uint256)" 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266)
[ "$credit" = 0 ] || wrong "the keeper's credit is $credit after its run, expected 0 (withdrawn)"

echo "=== a skipped stretch: 30 ticks due at once"
keeper 1030 || wrong "the catch-up run failed"
[ "$(call "$relay" 'relayed()(uint256)')" = 231 ] || wrong "relayed() is $(call "$relay" 'relayed()(uint256)'), expected 231"
grep -q 'claim window 228' "$work/keeper.log" || wrong "the newest window (228) was not claimed"

echo "=== limits"
keeper 1060 GAS_BUDGET=100000 || wrong "the run with a small budget failed"
grep -q "gas budget of 100000 is reached" "$work/keeper.log" || wrong "the small budget did not stop the run"
[ "$(call "$relay" 'relayed()(uint256)')" = 231 ] || wrong "a tick was relayed beyond the budget"
keeper 1060 MAX_FEE_WEI=1 || wrong "the run above the fee cap did not exit 0"
grep -q "above the cap" "$work/keeper.log" && [ "$(sent)" = 0 ] || wrong "the run above the fee cap sent something"
keeper 1060 || wrong "the run after the limits failed"
[ "$(call "$relay" 'relayed()(uint256)')" = 261 ] || wrong "relayed() is $(call "$relay" 'relayed()(uint256)') after the limits, expected 261"

echo "=== the watcher"
MAINNET_HEAD=$((first + 50 * 1060)) LAG_LIMIT=100000 node "$root/demo/test/tensile-fee-live.mjs" > "$work/watch.log" 2>&1 \
  || { sed 's/^/    /' "$work/watch.log"; wrong "the watcher failed on an honest relay"; }
sed 's/^/    /' "$work/watch.log" | grep -v '^    claim ' ; grep -c '^claim ' "$work/watch.log" | sed 's/^/    claims audited: /'
python3 -c '
import sys
t = open(sys.argv[1]).read().split("\n")
t[900] = str(int(t[900]) + 1)
open(sys.argv[2], "w").write("\n".join(t))' "$ticks" "$work/lied.csv"
if MAINNET_FILE="$work/lied.csv" MAINNET_HEAD=$((first + 50 * 1060)) LAG_LIMIT=100000 node "$root/demo/test/tensile-fee-live.mjs" > "$work/lied.log" 2>&1; then
  wrong "the watcher passed a relay whose tick at line 900 differs from mainnet"
else
  grep -q "relayed tick 100 (mainnet block $((first + 50 * 900)))" "$work/lied.log" && echo "the watcher names the changed tick: $(grep 'relayed tick 100' "$work/lied.log")" \
    || { cat "$work/lied.log"; wrong "the watcher failed for another reason"; }
fi
echo "fee_live_anvil: $failures wrong result(s)"
[ $failures -eq 0 ]
