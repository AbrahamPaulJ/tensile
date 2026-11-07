#!/bin/bash
export PATH="$HOME/.fe/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge
fe fmt ingots >/dev/null; fe fmt --check ingots && echo fmt-ok
s=$(date +%s); fe test 2>&1 | grep -E 'FAIL|test result|error' | head -30; echo "fe test $(( $(date +%s) - s )) s"
