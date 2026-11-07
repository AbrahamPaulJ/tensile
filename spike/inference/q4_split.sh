#!/bin/bash
# Q4 alternative: split weight-holder contracts. usage: q4_split.sh <workspace dir> <multiply-adds>
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
WS=${1:-layer}
MACS=${2:-3136}
cd $BASE/$WS
echo "=== fe test onchain_split --show-logs ($WS)"
s=$(date +%s.%N)
fe test --color never --show-logs onchain_split > $BASE/gen/q4_split_$MACS.log 2>&1
e=$(date +%s.%N)
python3 $BASE/decode_logs.py $MACS < $BASE/gen/q4_split_$MACS.log
grep -B2 -A14 -E "^error|Failed|panicked" $BASE/gen/q4_split_$MACS.log | head -80
echo "test wall=$(echo "$e - $s" | bc) s"
echo "=== fe build onchain_split"
rm -rf out-split
s=$(date +%s.%N)
fe build --color never --out-dir out-split --ingot onchain_split . 2>&1 | head -40
e=$(date +%s.%N)
echo "build wall=$(echo "$e - $s" | bc) s"
for f in out-split/*.runtime.bin; do
  n=$(tr -d '\n' < $f | wc -c)
  echo "$f: $((n / 2)) bytes (EIP-170 limit 24576)"
done
