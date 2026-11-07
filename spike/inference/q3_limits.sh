#!/bin/bash
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference
python3 - <<'EOF'
print("ref 4095:", (4095*7+3) % 251, " ref 65535:", (65535*7+3) % 251)
EOF
for f in q3_static_assert_fail.fe q3_limit_4096.fe q3_limit_65536.fe q3_limit_steps.fe; do
  echo "=== fe test $f"
  s=$(date +%s.%N)
  timeout 300 fe test --color never $f 2>&1 | head -40
  e=$(date +%s.%N)
  echo "wall=$(echo "$e - $s" | bc) s"
done
