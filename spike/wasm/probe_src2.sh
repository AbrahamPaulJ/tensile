#!/bin/bash
cd ~/fe-master
f=$(grep -rl "struct HostText" --include=*.fe . | head -1); echo "== $f"
grep -n -E "pub (fn|struct|const fn|enum)|impl" "$f" | head -40
tail -3 /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/wasm/build.log
