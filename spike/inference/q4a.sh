#!/bin/bash
# Q4 step 1: the pure layer in the nn ingot, tested in the embedded EVM.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference/layer
echo "=== fe test nn (EVM)"
time fe test --color never nn 2>&1 | head -60
