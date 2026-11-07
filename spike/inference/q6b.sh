#!/bin/bash
# Q6: second repro attempt for the EVM StackOverflow (same big array used in two messages).
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE
s=$(date +%s.%N)
timeout 400 fe test --color never -O 1 q6_array_twice.fe > gen/q6_array_twice.log 2>&1
st=$?
e=$(date +%s.%N)
tr '\r' '\n' < gen/q6_array_twice.log | grep -v "^$" | cut -c1-200 | head -12
echo "exit status: $st (124 means killed by timeout), wall=$(echo "$e - $s" | bc) s"
