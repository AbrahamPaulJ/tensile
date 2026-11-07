#!/bin/bash
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
sed -i 's/== 56/== 54/g' q3_limit_4096.fe
mkdir -p gen
for n in 4096 16384 32768 49152; do
  last=$((n - 1))
  want=$(( (last * 7 + 3) % 251 ))
  sed -e "s/4096/$n/g" -e "s/4095/$last/g" -e "s/== 54/== $want/g" q3_limit_4096.fe > gen/q3_limit_$n.fe
  echo "=== fe test gen/q3_limit_$n.fe"
  s=$(date +%s.%N)
  timeout 300 fe test --color never gen/q3_limit_$n.fe 2>&1 | grep -E "PASS|FAIL|error|steps|ok" | head -8
  e=$(date +%s.%N)
  echo "wall=$(echo "$e - $s" | bc) s"
done
echo "=== is the step limit configurable?"
grep -rnE "step_limit" ~/fe-src/crates --include=*.rs | grep -v "self.config\|config.step_limit" | head -20
