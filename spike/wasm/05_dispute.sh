#!/bin/bash
export PATH="$HOME/.fe/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge
fe --version; fe check ingots/dispute 2>&1 | tail -20; fe test ingots/dispute 2>&1 | tail -6; fe test --backend native ingots/dispute 2>&1 | tail -3
