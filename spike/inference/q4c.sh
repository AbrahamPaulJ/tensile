#!/bin/bash
# Q4 larger shape: 784 x 32. Copies the workspace to gen/layer784, regenerates weights, tests the nn ingot.
# usage: q4c.sh [generator flags, e.g. ctpack or nounpacked]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE
rm -rf gen/layer784
mkdir -p gen
cp -r layer gen/layer784
rm -rf gen/layer784/out-* gen/layer784/cli/out
sed -i 's/int8\[196\]/int8[784]/g' gen/layer784/onchain/src/lib.fe gen/layer784/onchain_split/src/lib.fe
python3 gen_weights.py 784 32 11 gen/layer784/nn/src $1
ls -la gen/layer784/nn/src/weights.fe
cd gen/layer784
echo "=== fe test nn (784x32, generator flags: $1)"
s=$(date +%s.%N)
fe test --color never nn > $BASE/gen/q4c_nn.log 2>&1
e=$(date +%s.%N)
tr '\r' '\n' < $BASE/gen/q4c_nn.log | cut -c1-220 | head -40
echo "nn test wall=$(echo "$e - $s" | bc) s"
