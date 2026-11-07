#!/bin/bash
# Q4 larger shape, full run: 784 x 32 without compile-time packing (literal packed weights from the generator).
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cp $BASE/gen/q4c_nn.log $BASE/gen/q4c_nn_ctpack_fail.log 2>/dev/null
bash $BASE/q4c.sh
bash $BASE/q4b.sh 1 25088 gen/layer784
bash $BASE/q4_split.sh gen/layer784 25088
