#!/bin/bash
# Q6: minimal repros for the surprises.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE
echo "=== (a) large array by value: EVM StackOverflow, and compile time"
for OPT in 1 0; do
  echo "--- fe test -O $OPT q6_array_copy_stack.fe (timeout 240 s)"
  s=$(date +%s.%N)
  timeout 240 fe test --color never -O $OPT q6_array_copy_stack.fe > gen/q6_array_copy_O$OPT.log 2>&1
  st=$?
  e=$(date +%s.%N)
  tr '\r' '\n' < gen/q6_array_copy_O$OPT.log | grep -v "^$" | cut -c1-200 | head -12
  echo "exit status: $st (124 means killed by timeout), wall=$(echo "$e - $s" | bc) s"
done
echo "=== (b) bytes of code per constant array element"
cd $BASE/q6_const_size
rm -rf out
fe build --color never --out-dir out . 2>&1 | grep -v Wrote
for f in out/*.runtime.bin; do
  n=$(tr -d '\n' < $f | wc -c)
  echo "$f: $((n / 2)) bytes"
done
cd $BASE
echo "=== (c) black_box on an array"
echo "--- fe check"
fe check --color never q6_black_box.fe 2>&1 | head -5
echo "--- fe test"
fe test --color never q6_black_box.fe 2>&1 | tr '\r' '\n' | grep -v "^$" | cut -c1-300 | head -8
