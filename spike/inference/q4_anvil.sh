#!/bin/bash
# Q4: deploy LayerPacked (196x16, -O1 build) to a local anvil (Osaka) and measure real transaction gas.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference/layer
RPC=http://127.0.0.1:8545
PK=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
anvil --hardfork osaka > /tmp/anvil_q4.log 2>&1 &
APID=$!
sleep 3
X=$(cat nn/x_196x16.txt)
echo "expected: $(cat nn/expected_196x16.txt)"
echo "=== deploy LayerPacked"
cast send --rpc-url $RPC --private-key $PK --create 0x$(cat out-1/LayerPacked.bin) --json > /tmp/deploy.json
ADDR=$(python3 -c "import json;d=json.load(open('/tmp/deploy.json'));print(d['contractAddress'])")
python3 -c "import json;d=json.load(open('/tmp/deploy.json'));print('deploy gasUsed', int(d['gasUsed'],16), 'status', d['status'])"
echo "address $ADDR, deployed code bytes: $(( ($(cast code --rpc-url $RPC $ADDR | wc -c) - 3) / 2 ))"
echo "=== cast call infer(int8[196])(int8[16])"
cast call --rpc-url $RPC $ADDR "infer(int8[196])(int8[16])" "$X"
echo "=== cast estimate (full transaction gas, includes 21000 base and calldata)"
cast estimate --rpc-url $RPC $ADDR "infer(int8[196])" "$X"
echo "=== cast send (gasUsed of a mined transaction)"
cast send --rpc-url $RPC --private-key $PK $ADDR "infer(int8[196])" "$X" --json > /tmp/tx.json
python3 -c "import json;d=json.load(open('/tmp/tx.json'));print('tx gasUsed', int(d['gasUsed'],16), 'status', d['status'])"
kill $APID
