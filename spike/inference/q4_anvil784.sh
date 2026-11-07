#!/bin/bash
# Q4: the 784 x 32 per-row design (RowsA + RowsB + LayerRows) deployed to a local anvil (Osaka).
# Needs gen/layer784/out-split (run q4d.sh / q4_split.sh gen/layer784 25088 first).
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
BASE=/mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
cd $BASE/gen/layer784
exec > $BASE/gen/q4_anvil784.log 2>&1
RPC=http://127.0.0.1:8545
PK=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
anvil --hardfork osaka > /tmp/anvil_q4b.log 2>&1 &
APID=$!
sleep 3
deploy() {
  cast send --rpc-url $RPC --private-key $PK --create "$1" --json > /tmp/dep.json
  python3 -c "import json;d=json.load(open('/tmp/dep.json'));print(d['contractAddress'], int(d['gasUsed'],16), d['status'])"
}
read A GA SA <<< "$(deploy 0x$(cat out-split/RowsA.bin))"
echo "RowsA at $A, deploy gas $GA, status $SA"
read B GB SB <<< "$(deploy 0x$(cat out-split/RowsB.bin))"
echo "RowsB at $B, deploy gas $GB, status $SB"
ARGS=$(cast abi-encode "constructor(address,address)" $A $B)
read L GL SL <<< "$(deploy 0x$(cat out-split/LayerRows.bin)${ARGS#0x})"
echo "LayerRows at $L, deploy gas $GL, status $SL"
echo "LayerRows deployed code bytes: $(( ($(cast code --rpc-url $RPC $L | wc -c) - 3) / 2 ))"
X=$(cat nn/x_784x32.txt)
echo "expected: $(cat nn/expected_784x32.txt)"
echo "=== cast call infer(int8[784])(int8[32])"
cast call --rpc-url $RPC $L "infer(int8[784])(int8[32])" "$X"
echo "=== cast estimate"
cast estimate --rpc-url $RPC $L "infer(int8[784])" "$X"
echo "=== cast send"
cast send --rpc-url $RPC --private-key $PK $L "infer(int8[784])" "$X" --json > /tmp/tx.json
python3 -c "import json;d=json.load(open('/tmp/tx.json'));print('tx gasUsed', int(d['gasUsed'],16), 'status', d['status'])"
kill $APID
