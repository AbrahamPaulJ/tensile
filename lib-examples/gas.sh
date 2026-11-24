#!/bin/bash
# Gas and size of the library examples against their plain twins, with real transactions on a
# local chain (anvil, Osaka). Every contract is called with the same inputs as its twin, the two
# results must be equal, and the gas of each transaction is recorded.
#
#   TickEma / PlainTickEma         update(int32): the second tick (a storage slot already written)
#   Scorecard / PlainScorecard     score(uint256): one packed feature word
#   RateCurve / PlainRateCurve     borrowRate(uint256,uint256) below and above the kink
#   WadRateCurve                   the same, with core::fixed::Wad (a result in 18 decimals)
#
# "gas" is the transaction's gasUsed; "execution" is the gas of the code alone, from the opcode
# trace (see `send` below). Sizes are runtime bytes.
#
# usage: gas.sh <fe-out-dir> [record.json] [port]
#   (fe-out-dir: `fe build --out-dir DIR` run in lib-examples). If the port is busy, the next free
#   one of the ten above it is used.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
if [ $# -lt 1 ] || [ ! -f "$1/TickEma.bin" ]; then
  echo "usage: gas.sh <fe-out-dir> [record.json] [port]" >&2
  exit 2
fi
out="$1"
record="$2"
port="${3:-18631}"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80

anvil_pid=
for try in $(seq 0 10); do
  p=$((port + try))
  if ! (echo > /dev/tcp/127.0.0.1/$p) 2> /dev/null; then
    anvil --steps-tracing --hardfork osaka --port $p > /dev/null 2>&1 &
    anvil_pid=$!
    port=$p
    break
  fi
done
[ -n "$anvil_pid" ] || { echo "gas.sh: no free port from $port" >&2; exit 2; }
trace="$(mktemp)"
trap 'kill $anvil_pid 2> /dev/null; rm -f "$trace"' EXIT
rpc="http://127.0.0.1:$port"
for i in $(seq 1 50); do
  cast block-number --rpc-url $rpc > /dev/null 2>&1 && break
  sleep 0.2
done

failures=0
wrong() { echo "WRONG: $1"; failures=$((failures + 1)); }
deploy() {
  cast send --rpc-url $rpc --private-key $key --create "0x$(cat "$out/$1.bin")" --json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["contractAddress"])'
}
# send <address> <signature> <args...>: prints "gasUsed execution". The execution gas is read from
# the transaction's opcode trace (gas at the first step minus gas after the last), because for a
# call with little work and 64 bytes of arguments gasUsed is the EIP-7623 calldata floor, which
# hides it.
send() {
  local receipt hash used
  receipt=$(cast send --rpc-url $rpc --private-key $key --gas-limit 1000000 "$@" --json)
  read -r hash used <<< "$(python3 -c 'import json, sys; r = json.loads(sys.argv[1]); assert r["status"] in ("0x1", 1); print(r["transactionHash"], int(r["gasUsed"], 16))' "$receipt")" \
    || { echo "0 0"; return; }
  cast rpc --rpc-url $rpc debug_traceTransaction "$hash" \
    '{"disableStorage": true, "disableMemory": true, "disableStack": true}' > "$trace"
  echo "$used $(python3 -c 'import json, sys
logs = json.load(open(sys.argv[1]))["structLogs"]
print(logs[0]["gas"] - logs[-1]["gas"] + logs[-1]["gasCost"])' "$trace")"
}
size() { echo $(( $(tr -d '\n' < "$out/$1.runtime.bin" | wc -c) / 2 )); }

declare -A gas exe
measure() { # measure <name> <address> <signature> <args...>
  local name=$1
  shift
  read -r g e <<< "$(send "$@")"
  gas[$name]=$g
  exe[$name]=$e
}

# The tick average: seed, then the measured second tick.
for c in TickEma PlainTickEma; do
  a=$(deploy $c)
  send "$a" "update(int32)" 194000 > /dev/null
  measure "$c.update" "$a" "update(int32)" 194123
  v=$(cast call --rpc-url $rpc "$a" "raw()(int32)" | cut -d' ' -f1)
  eval "raw_$c=\$v"
done
[ "$raw_TickEma" = "$raw_PlainTickEma" ] || wrong "TickEma raw $raw_TickEma, plain $raw_PlainTickEma"

# The scorecard: features 0..127 from a seed, packed into the top 16 bytes of one word.
word=$(python3 -c 'print("0x" + "".join("%02x" % ((5 + i * 37) % 128) for i in range(16)) + "00" * 16)')
for c in Scorecard PlainScorecard; do
  a=$(deploy $c)
  measure "$c.score" "$a" "score(uint256)" "$word"
  v=$(cast call --rpc-url $rpc "$a" "score(uint256)(int32)" "$word" | cut -d' ' -f1)
  eval "score_$c=\$v"
done
[ "$score_Scorecard" = "$score_PlainScorecard" ] || wrong "Scorecard $score_Scorecard, plain $score_PlainScorecard"

# The rate curve: utilisation 50% (below the kink) and 87.5% (above it), 18-decimal amounts.
wad=1000000000000000000
for c in RateCurve PlainRateCurve WadRateCurve; do
  a=$(deploy $c)
  measure "$c.below" "$a" "borrowRate(uint256,uint256)" "500${wad#1}" "500${wad#1}"
  measure "$c.above" "$a" "borrowRate(uint256,uint256)" "125${wad#1}" "875${wad#1}"
  if [ $c = WadRateCurve ]; then
    v=$(cast call --rpc-url $rpc "$a" "borrowRate(uint256,uint256)(uint256)" 125 875 | cut -d' ' -f1)
  else
    v=$(cast call --rpc-url $rpc "$a" "borrowRate(uint256,uint256)(int32)" 125 875 | cut -d' ' -f1)
  fi
  eval "rate_$c=\$v"
done
[ "$rate_RateCurve" = "$rate_PlainRateCurve" ] || wrong "RateCurve $rate_RateCurve, plain $rate_PlainRateCurve"
# 87.5% utilisation: 0.359375 a year, 23552 / 65536 and 359375000000000000 / 1e18.
[ "$rate_RateCurve" = 23552 ] || wrong "RateCurve at 87.5%: $rate_RateCurve, expected 23552"
[ "$rate_WadRateCurve" = 359375000000000000 ] || wrong "WadRateCurve at 87.5%: $rate_WadRateCurve"

for k in "${!gas[@]}"; do
  [ "${gas[$k]}" != 0 ] || wrong "$k reverted"
done

printf '%-22s %8s %10s %8s\n' "call" "gas" "execution" "runtime"
for k in TickEma.update PlainTickEma.update Scorecard.score PlainScorecard.score RateCurve.below \
         PlainRateCurve.below WadRateCurve.below RateCurve.above PlainRateCurve.above WadRateCurve.above; do
  printf '%-22s %8s %10s %8s\n' "$k" "${gas[$k]}" "${exe[$k]}" "$(size "${k%%.*}")"
done

if [ -n "$record" ]; then
  {
    echo "{"
    echo "  \"fe\": \"$(fe --version)\","
    echo "  \"chain\": \"anvil, Osaka\","
    echo "  \"calls\": {"
    first=1
    for k in TickEma.update PlainTickEma.update Scorecard.score PlainScorecard.score RateCurve.below \
             PlainRateCurve.below WadRateCurve.below RateCurve.above PlainRateCurve.above WadRateCurve.above; do
      [ $first = 1 ] || echo ","
      first=0
      printf '    "%s": {"gas": %s, "execution": %s}' "$k" "${gas[$k]}" "${exe[$k]}"
    done
    echo
    echo "  },"
    echo "  \"runtime_bytes\": {"
    first=1
    for c in TickEma PlainTickEma Scorecard PlainScorecard RateCurve PlainRateCurve WadRateCurve; do
      [ $first = 1 ] || echo ","
      first=0
      printf '    "%s": %s' "$c" "$(size $c)"
    done
    echo
    echo "  }"
    echo "}"
  } > "$record"
  python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$record" || wrong "record is not JSON"
fi
echo "gas.sh: $failures wrong"
[ $failures -eq 0 ]
