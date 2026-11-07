#!/bin/bash
# Q6: EVM StackOverflow when calling the naive (unpacked) Layer contract at 784 x 32.
# Needs gen/layer784 (run q4d.sh first).
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE/gen/layer784
rm -rf $BASE/gen/q6_dbg
fe test --color never --filter gas_contract_call --trace-evm --trace-evm-keep 24 --trace-evm-stack-n 3 --call-trace --debug-dir $BASE/gen/q6_dbg onchain > $BASE/gen/q6_stackoverflow.log 2>&1
tr '\r' '\n' < $BASE/gen/q6_stackoverflow.log | cut -c1-200 | grep -v "^$" | head -40
ls -la $BASE/gen/q6_dbg 2>/dev/null | head
for f in $BASE/gen/q6_dbg/*; do echo "--- $f"; tail -30 "$f" | cut -c1-200; done 2>/dev/null | head -120
