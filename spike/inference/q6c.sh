#!/bin/bash
# Q6 (b): bytes of contract code per constant array element.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE/q6_const_size
rm -rf out
fe build --color never --out-dir out . 2>&1 | grep -v -E "Wrote|Warning"
for f in out/*.runtime.bin; do
  n=$(tr -d '
' < $f | wc -c)
  echo "$f: $((n / 2)) bytes"
done
