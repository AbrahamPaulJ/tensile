#!/bin/bash
# `FeeOracle` (ingots/consumer) with real transactions on a local chain (anvil, Osaka), on the real
# pool ticks behind one test sample: the first whose reference tier is not the middle one, so that
# the fee visibly leaves its default.
#
#   1. deploys the generated Arbiter (window 3,600 s), the test pool and a FeeOracle with an
#      interval of 50 blocks;
#   2. feeds the 145 ticks of that sample from data/ticks.csv: setTick on the test pool, `observe()`,
#      then 49 mined blocks. One `observe()` inside an interval must revert;
#   3. `inputWords(144)` must equal the packed input of the sample in out/inputs.bin: the bytes the
#      contract built from ticks are the bytes the model was exported on;
#   4. posts the honest claim about that input. `apply` must revert while the claim is open; after
#      the window and `finalize`, `apply(0, 144)` must set the fee of the reference's class, and
#      `fee()` must return it;
#   5. one more observation and the next sample: `inputWords(145)`, a window that starts inside
#      a stored word, must equal that sample too; its claim cannot be applied to window 144 and
#      sets the fee when applied to window 145.
#
# The pool is a test double that answers `slot0()`; no real pool is called. Gas of `observe`,
# `apply` and `fee` is recorded.
#
# usage: oracle_check.sh <fe-out-dir> <ref-dir> <tree.json> [record.json] [port]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
if [ $# -lt 3 ] || [ ! -f "$1/FeeOracle.bin" ]; then
  echo "usage: oracle_check.sh <fe-out-dir> <ref-dir> <tree.json> [record.json] [port]" >&2
  exit 2
fi
out="$1"
ref="$2"
tree="$3"
record="$4"
port="${5:-18623}"
rpc="http://127.0.0.1:$port"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
me=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
bond=1000000000000000000
window=3600
interval=50
anvil --hardfork osaka --port $port > /dev/null 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null' EXIT
sleep 2
failures=0
args() { python3 "$root/tools/model_args.py" --ref "$ref" --tree "$tree" "$@"; }
wrong() { echo "WRONG: $1"; failures=$((failures + 1)); }
deploy() {
  cast send --rpc-url $rpc --private-key $key --create "$1" --json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["contractAddress"])'
}
# send <address> <signature> <args...>: prints the gas used.
send() {
  cast send --rpc-url $rpc --private-key $key "$@" --json \
    | python3 -c 'import json, sys; print(int(json.load(sys.stdin)["gasUsed"], 16))'
}

root_hash=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["root"])' "$tree")
ctor=$(cast abi-encode "constructor(uint256,uint256,uint256)" "$root_hash" $bond $window)
arbiter=$(deploy "0x$(cat "$out/Arbiter.bin")${ctor#0x}")
pool=$(deploy "0x$(cat "$out/TestPool.bin")")
ctor=$(cast abi-encode "constructor(address,address,uint256)" "$pool" "$arbiter" $interval)
oracle=$(deploy "0x$(cat "$out/FeeOracle.bin")${ctor#0x}")
echo "Arbiter $arbiter, TestPool $pool, FeeOracle $oracle (interval $interval blocks)"

sample=$(python3 -c 'import sys; print(next(i for i, v in enumerate(open(sys.argv[1], "rb").read()) if v != 1))' "$ref/pred.bin")
mapfile -t ticks < <(python3 "$here/dataset.py" --sample-ticks "$sample")
observe_first=0
observe_gas=()
for i in "${!ticks[@]}"; do
  send "$pool" "setTick(int32)" "${ticks[$i]}" > /dev/null
  gas=$(send "$oracle" "observe()")
  if [ "$i" -eq 0 ]; then observe_first=$gas; else observe_gas+=("$gas"); fi
  if [ "$i" -eq 1 ]; then
    # Inside the interval a second observation must be refused.
    if cast call --rpc-url $rpc "$oracle" "observe()" > /dev/null 2>&1; then
      wrong "observe() inside the interval did not revert"
    fi
  fi
  cast rpc --rpc-url $rpc anvil_mine $((interval - 1)) > /dev/null
done
moves=$(cast call --rpc-url $rpc "$oracle" "moves()(uint256)")
[ "$moves" = "144" ] || wrong "moves() is $moves after ${#ticks[@]} observations, expected 144"
echo "fed ${#ticks[@]} ticks (${ticks[0]} to ${ticks[144]}), $moves moves stored"

mapfile -t honest < <(args claim --sample "$sample")
want_words=$(python3 -c 'import sys; print(" ".join(str(int(w, 16)) for w in sys.argv[1].strip("[]").split(",")))' "${honest[0]}")
got_words=$(cast call --rpc-url $rpc "$oracle" "inputWords(uint256)(uint256[5])" 144 \
  | python3 -c 'import re, sys; print(" ".join(re.findall(r"\d+", re.sub(r"\[[0-9.e]+\]", "", sys.stdin.read()))))')
if [ "$got_words" = "$want_words" ]; then
  echo "inputWords(144) equals the packed input of sample $sample in inputs.bin (144 bytes)"
else
  wrong "inputWords(144) differs from sample $sample of inputs.bin"
fi

claim_gas=$(send --value $bond "$arbiter" "$(args signature claim)" "${honest[@]}")
if cast call --rpc-url $rpc --from $me "$oracle" "apply(uint256,uint256)" 0 144 > /dev/null 2>&1; then
  wrong "apply of an open claim did not revert"
fi
cast rpc --rpc-url $rpc evm_increaseTime $((window + 1)) > /dev/null
cast rpc --rpc-url $rpc anvil_mine 1 > /dev/null
finalize_gas=$(send "$arbiter" "finalize(uint256)" 0)
before=$(cast call --rpc-url $rpc "$oracle" "fee()(uint32)" | cut -d' ' -f1)
apply_gas=$(send "$oracle" "apply(uint256,uint256)" 0 144)
after=$(cast call --rpc-url $rpc "$oracle" "fee()(uint32)" | cut -d' ' -f1)
fee_gas=$(cast estimate --rpc-url $rpc --from $me "$oracle" "fee()")
class=$(python3 -c 'import sys; print(open(sys.argv[1], "rb").read()[int(sys.argv[2])])' "$ref/pred.bin" "$sample")
want_fee=$(python3 -c 'import sys; print([500, 3000, 10000][int(sys.argv[1])])' "$class")
[ "$after" != "$before" ] || wrong "apply did not change the fee"
[ "$before" = "3000" ] || wrong "fee() was $before before any claim was applied, expected the default 3000"
[ "$after" = "$want_fee" ] || wrong "fee() is $after after apply, expected $want_fee for tier $class"
if cast call --rpc-url $rpc --from $me "$oracle" "apply(uint256,uint256)" 0 144 > /dev/null 2>&1; then
  wrong "applying the same window twice did not revert"
fi

# The next window: one more observation, so its 144 moves start in the middle of a stored word.
# Its claim is applied when the oracle's three fee slots are already non-zero: the cost of every
# apply after the first.
next=$((sample + 1))
mapfile -t next_ticks < <(python3 "$here/dataset.py" --sample-ticks "$next")
send "$pool" "setTick(int32)" "${next_ticks[144]}" > /dev/null
observe_gas+=("$(send "$oracle" "observe()")")
mapfile -t honest_next < <(args claim --sample "$next")
want_words=$(python3 -c 'import sys; print(" ".join(str(int(w, 16)) for w in sys.argv[1].strip("[]").split(",")))' "${honest_next[0]}")
got_words=$(cast call --rpc-url $rpc "$oracle" "inputWords(uint256)(uint256[5])" 145 \
  | python3 -c 'import re, sys; print(" ".join(re.findall(r"\d+", re.sub(r"\[[0-9.e]+\]", "", sys.stdin.read()))))')
if [ "$got_words" = "$want_words" ]; then
  echo "inputWords(145) equals the packed input of sample $next in inputs.bin"
else
  wrong "inputWords(145) differs from sample $next of inputs.bin"
fi
send --value $bond "$arbiter" "$(args signature claim)" "${honest_next[@]}" > /dev/null
cast rpc --rpc-url $rpc evm_increaseTime $((window + 1)) > /dev/null
cast rpc --rpc-url $rpc anvil_mine 1 > /dev/null
send "$arbiter" "finalize(uint256)" 1 > /dev/null
if cast call --rpc-url $rpc --from $me "$oracle" "apply(uint256,uint256)" 1 144 > /dev/null 2>&1; then
  wrong "a claim about window 145 was applied to window 144"
fi
apply_later_gas=$(send "$oracle" "apply(uint256,uint256)" 1 145)
next_class=$(python3 -c 'import sys; print(open(sys.argv[1], "rb").read()[int(sys.argv[2])])' "$ref/pred.bin" "$next")
next_fee=$(cast call --rpc-url $rpc "$oracle" "fee()(uint32)" | cut -d' ' -f1)
want_fee=$(python3 -c 'import sys; print([500, 3000, 10000][int(sys.argv[1])])' "$next_class")
[ "$next_fee" = "$want_fee" ] || wrong "fee() is $next_fee after the second apply, expected $want_fee for tier $next_class"

typical=$(python3 -c 'import sys; v = sorted(int(a) for a in sys.argv[1:]); print(v[len(v) // 2], v[0], v[-1])' "${observe_gas[@]}")
echo "sample $sample: reference tier $class, fee() $before before and $after after apply; sample $next: tier $next_class, fee() $next_fee"
echo "gas: first observe $observe_first, later observes median/min/max $typical, claim $claim_gas, finalize $finalize_gas, first apply $apply_gas, later apply $apply_later_gas, fee() estimate $fee_gas"
echo "oracle_check: $failures wrong result(s)"
if [ -n "$record" ] && [ $failures -eq 0 ]; then
  python3 - "$record" "$sample" "$observe_first" "$claim_gas" "$finalize_gas" "$apply_gas" "$fee_gas" "$class" "$after" "$(fe --version)" "$apply_later_gas" "$next_class" "$next_fee" "${observe_gas[@]}" <<'EOF'
import json, sys
path, sample, first, claim, finalize, apply, fee, tier, fee_value, fe_version, apply_later, next_tier, next_fee = sys.argv[1:14]
later = sorted(int(v) for v in sys.argv[14:])
record = {"fe": fe_version, "chain": "anvil, Osaka", "sample": int(sample), "interval_blocks": 50,
          "observations": len(later) + 1, "tier": int(tier), "fee": int(fee_value),
          "observe_first": int(first), "observe_median": later[len(later) // 2],
          "observe_min": later[0], "observe_max": later[-1],
          "claim": int(claim), "finalize": int(finalize), "apply": int(apply), "apply_later": int(apply_later),
          "next_tier": int(next_tier), "next_fee": int(next_fee), "fee_estimate": int(fee)}
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
EOF
fi
[ $failures -eq 0 ]
