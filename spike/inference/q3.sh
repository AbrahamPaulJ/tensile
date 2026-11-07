#!/bin/bash
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
echo "=== fe test q3_tables.fe --show-logs"
time fe test --color never --show-logs q3_tables.fe 2>&1
echo "exit=$?"
