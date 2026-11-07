#!/bin/bash
FE=$HOME/fe-master/target/release/fe
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/tests/defects
for b in sonatina native wasm; do echo "--- $b"; PATH=$HOME/.local/node/bin:$PATH $FE test --backend $b --color never native_word_endianness_master.fe 2>&1 | tail -6; done
