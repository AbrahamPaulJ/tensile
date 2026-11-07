#!/bin/bash
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
for f in q1_shapes.fe q1_mismatch.fe q1_flat.fe; do
  echo "=== fe test $f"
  fe test --color never $f 2>&1
  echo "exit=$?"
done
