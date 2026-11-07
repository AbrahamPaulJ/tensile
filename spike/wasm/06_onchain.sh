#!/bin/bash
export PATH="$HOME/.fe/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge
fe check ingots/onchain 2>&1 | tail -30
fe fmt --check ingots 2>&1 | tail -10
