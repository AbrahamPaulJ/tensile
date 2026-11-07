#!/bin/bash
# Q4 step 2: contract + gas tests, build and bytecode size.
# usage: q4b.sh <opt level> <number of multiply-adds> [workspace dir relative to spike/inference]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
OPT=${1:-1}
MACS=${2:-3136}
WS=${3:-layer}
cd $BASE/$WS
echo "=== fe test onchain -O $OPT --show-logs"
s=$(date +%s.%N)
fe test --color never --show-logs -O $OPT onchain > $BASE/gen/q4_test_${MACS}_O$OPT.log 2>&1
e=$(date +%s.%N)
python3 $BASE/decode_logs.py $MACS < $BASE/gen/q4_test_${MACS}_O$OPT.log
grep -B2 -A12 -E "^error|Failed|panicked" $BASE/gen/q4_test_${MACS}_O$OPT.log | head -60
echo "test wall=$(echo "$e - $s" | bc) s"
echo "=== fe build onchain -O $OPT"
rm -rf out-$OPT
s=$(date +%s.%N)
fe build --color never -O $OPT --out-dir out-$OPT --ingot onchain . 2>&1 | head -40
e=$(date +%s.%N)
echo "build wall=$(echo "$e - $s" | bc) s"
for f in out-$OPT/*.runtime.bin; do
  n=$(tr -d '\n' < $f | wc -c)
  echo "$f: $((n / 2)) bytes (EIP-170 limit 24576)"
done
