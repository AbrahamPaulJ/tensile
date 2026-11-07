#!/bin/bash
# Q5 extras: native throughput on a 10,000-sample set (196 x 16), and a native build of the 784 x 32 shape.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE/layer
mkdir -p $BASE/gen/q5big
exec > $BASE/gen/q5b.log 2>&1
python3 $BASE/gen_samples.py nn/src/weights.fe $BASE/gen/q5big 10000
echo "=== native -O1, 10,000 samples of 196 bytes (each sample runs 3 kernels: packed-wide, unchecked, packed)"
s=$(date +%s.%N)
out-native-1/cli < $BASE/gen/q5big/samples.bin > $BASE/gen/q5big/native.txt
st=$?
e=$(date +%s.%N)
echo "exit status $st, wall=$(echo "$e - $s" | bc) s, lines=$(wc -l < $BASE/gen/q5big/native.txt)"
tail -n +2 $BASE/gen/q5big/native.txt > $BASE/gen/q5big/native_samples.txt
if cmp -s $BASE/gen/q5big/native_samples.txt $BASE/gen/q5big/samples_ref.txt; then echo "10,000 native outputs IDENTICAL to Python reference"; else echo "DIFFERS"; fi

echo "=== native build of the 784 x 32 workspace (gen/layer784)"
cd $BASE/gen/layer784
mkdir -p cli/src
cp $BASE/layer/cli/fe.toml cli/fe.toml
cp $BASE/layer/cli/src/lib.fe cli/src/lib.fe
rm -rf out-native-1
s=$(date +%s.%N)
fe build --color never --backend native -O 1 --out-dir out-native-1 --ingot cli . 2>&1 | head -20
e=$(date +%s.%N)
echo "native build wall=$(echo "$e - $s" | bc) s"
ls -la out-native-1 | tail -n +2
out-native-1/cli < /dev/null
echo "exit status $?"
echo "reference: $(cat nn/expected_784x32.txt)"
