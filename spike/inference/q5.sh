#!/bin/bash
# Q5: one source, two backends. Builds the cli ingot natively, runs it on the embedded input plus
# 20 random samples, and compares with the Python reference and with the EVM contract on anvil.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE/layer
mkdir -p $BASE/gen/q5
python3 $BASE/gen_samples.py nn/src/weights.fe $BASE/gen/q5 20

for OPT in 0 1 2; do
  echo "=== fe build --backend native -O $OPT --ingot cli"
  rm -rf out-native-$OPT
  s=$(date +%s.%N)
  fe build --color never --backend native -O $OPT --out-dir out-native-$OPT --ingot cli . 2>&1 | head -30
  e=$(date +%s.%N)
  echo "native build wall=$(echo "$e - $s" | bc) s"
  ls -la out-native-$OPT | tail -n +2
  if [ -x out-native-$OPT/cli ]; then
    file out-native-$OPT/cli
    out-native-$OPT/cli < $BASE/gen/q5/samples.bin > $BASE/gen/q5/native_O$OPT.txt
    echo "exit status: $?  lines: $(wc -l < $BASE/gen/q5/native_O$OPT.txt)"
    head -2 $BASE/gen/q5/native_O$OPT.txt
  fi
done

echo "=== native vs Python reference (embedded input line, then 20 samples)"
cat nn/expected_196x16.txt $BASE/gen/q5/samples_ref.txt > $BASE/gen/q5/ref_all.txt
for OPT in 0 1 2; do
  if cmp -s $BASE/gen/q5/ref_all.txt $BASE/gen/q5/native_O$OPT.txt; then echo "native -O$OPT: IDENTICAL to reference (21 lines)"; else echo "native -O$OPT: DIFFERS"; diff $BASE/gen/q5/ref_all.txt $BASE/gen/q5/native_O$OPT.txt | head; fi
done

echo "=== EVM: same 21 inputs through LayerPacked on anvil (cast call)"
RPC=http://127.0.0.1:8545
PK=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
anvil --hardfork osaka > /tmp/anvil_q5.log 2>&1 &
APID=$!
sleep 3
ADDR=$(cast send --rpc-url $RPC --private-key $PK --create 0x$(cat out-1/LayerPacked.bin) --json | python3 -c "import json,sys;print(json.load(sys.stdin)['contractAddress'])")
: > $BASE/gen/q5/evm.txt
( cat nn/x_196x16.txt; cat $BASE/gen/q5/samples_abi.txt ) | while read -r X; do
  cast call --rpc-url $RPC $ADDR "infer(int8[196])(int8[16])" "$X" | tr -d '[],' >> $BASE/gen/q5/evm.txt
done
kill $APID
head -2 $BASE/gen/q5/evm.txt
if cmp -s $BASE/gen/q5/evm.txt $BASE/gen/q5/native_O1.txt; then echo "EVM (anvil) vs native -O1: IDENTICAL (21 lines)"; else echo "EVM vs native: DIFFERS"; diff $BASE/gen/q5/evm.txt $BASE/gen/q5/native_O1.txt | head; fi
md5sum $BASE/gen/q5/ref_all.txt $BASE/gen/q5/native_O*.txt $BASE/gen/q5/evm.txt

echo "=== fe test --backend native (nn and cli ingots)"
for OPT in 0 1; do
  s=$(date +%s.%N)
  fe test --color never --backend native -O $OPT nn cli 2>&1 | tr '\r' '\n' | grep -E "PASS|FAIL|ERROR|error|test result" | head -20
  e=$(date +%s.%N)
  echo "native test -O$OPT wall=$(echo "$e - $s" | bc) s"
done
echo "=== fe test --backend native on the onchain ingot (expected to be rejected: EVM effects)"
fe test --color never --backend native onchain 2>&1 | tr '\r' '\n' | head -12
