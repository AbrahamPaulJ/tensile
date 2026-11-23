#!/bin/bash
# `FeeOracle.observe()` against the real pool, on a local fork of mainnet (anvil --fork-url).
# Not part of check.sh or CI, because it needs a mainnet RPC endpoint; run by hand after check.sh
# has built build/out. It answers one question the test pool cannot: does the `slot0()` message
# of ingots/consumer decode what the deployed Uniswap v3 pool returns?
#
#   1. forks mainnet at its latest block and deploys a FeeOracle that observes the real
#      USDC/WETH 0.05% pool, with an interval of 50 blocks;
#   2. `observe()`, 50 mined blocks, `observe()`: the tick the contract decoded is read from the
#      second call's `Observed` event and compared with `cast call pool slot0()`; on a fork
#      nobody trades, so the stored move must be 0.
#
# usage: fork_check.sh [rpc-url] [port]      (default https://ethereum-rpc.publicnode.com, 18624)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
here="$(cd "$(dirname "$0")" && pwd)"
fork="${1:-https://ethereum-rpc.publicnode.com}"
port="${2:-18624}"
rpc="http://127.0.0.1:$port"
key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
pool=0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640
out="$here/build/out"
[ -f "$out/FeeOracle.bin" ] || { echo "run check.sh first: $out/FeeOracle.bin is missing" >&2; exit 2; }
anvil --fork-url "$fork" --port $port > /dev/null 2>&1 &
anvil_pid=$!
trap 'kill $anvil_pid 2> /dev/null' EXIT
sleep 8
failures=0
block=$(cast block-number --rpc-url $rpc) || { echo "the fork did not start"; exit 1; }
ctor=$(cast abi-encode "constructor(address,address,uint256)" $pool 0x0000000000000000000000000000000000000000 50)
oracle=$(cast send --rpc-url $rpc --private-key $key --create "0x$(cat "$out/FeeOracle.bin")${ctor#0x}" --json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["contractAddress"])')
want=$(cast call --rpc-url $rpc $pool "slot0()(uint160,int24,uint16,uint16,uint16,uint8,bool)" | sed -n 2p | cut -d' ' -f1)
cast send --rpc-url $rpc --private-key $key "$oracle" "observe()" --json > /dev/null || failures=$((failures + 1))
cast rpc --rpc-url $rpc anvil_mine 50 > /dev/null
# The second observation's event carries the tick the contract decoded: Observed(uint256 indexed, int32, uint8).
data=$(cast send --rpc-url $rpc --private-key $key "$oracle" "observe()" --json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["logs"][0]["data"])')
got=$(python3 -c 'import sys; v = int(sys.argv[1][2:66], 16); print(v - (1 << 256) if v >> 255 else v)' "$data")
distance=$(python3 -c 'import sys; print(int(sys.argv[1][66:130], 16))' "$data")
moves=$(cast call --rpc-url $rpc "$oracle" "moves()(uint256)")
echo "fork of $fork at block $block: pool slot0 tick $want, FeeOracle decoded $got, stored move $distance, moves() $moves"
[ "$got" = "$want" ] || { echo "WRONG: the contract decoded another tick"; failures=$((failures + 1)); }
[ "$distance" = "0" ] && [ "$moves" = "1" ] || { echo "WRONG: expected one stored move of 0"; failures=$((failures + 1)); }
echo "fork_check: $failures wrong result(s)"
[ $failures -eq 0 ]
