#!/bin/bash
# Milestone 2 check: regenerates the model ingot, runs every Fe test on both backends, builds the
# contracts, checks their runtime sizes, runs the Foundry differential and fuzz tests, and measures
# the real transactions on anvil. Prints a summary with the wall time of each step and exits
# non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../fe-challenge/tools/m2_check.sh
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
log="$(mktemp)"
out="$(mktemp -d)"
trap 'rm -rf "$log" "$out"' EXIT
summary=()
failures=0

# step <name> <command...>: runs the command, shows its output, records result and wall time.
step() {
  local name="$1"
  shift
  echo "=== $name"
  local start=$(date +%s.%N)
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  local seconds=$(printf '%.1f' "$(echo "$(date +%s.%N) - $start" | bc)")
  local detail="$(tail -1 "$log")"
  if [ $status -eq 0 ]; then
    summary+=("$(printf 'ok    %7s s  %-30s %s' "$seconds" "$name" "$detail")")
  else
    summary+=("$(printf 'FAIL  %7s s  %-30s %s' "$seconds" "$name" "$detail")")
    failures=$((failures + 1))
  fi
}

format_check() {
  fe fmt --check ingots && echo "fe fmt --check: clean"
}

build_contracts() {
  fe build --color never --out-dir "$out" --ingot onchain . && echo "built $(ls "$out"/*.runtime.bin | wc -l) contracts"
}

total_start=$(date +%s.%N)
echo "$(fe --version), $(forge --version | head -1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "generate model ingot" bash tools/gen_model.sh
step "fe fmt --check" format_check
for ingot in tensile model; do
  for backend in sonatina native; do
    step "fe test $ingot ($backend)" bash tools/test_ingot.sh $ingot $backend
  done
done
step "fe test cli (native)" bash tools/test_ingot.sh cli native
step "fe test onchain (sonatina)" bash tools/test_ingot.sh onchain sonatina
step "fe build onchain" build_contracts
step "runtime sizes" bash tools/check_size.sh "$out"
step "forge: differential and fuzz" bash tools/evm_test.sh "$out" --no-match-path "test/Arbiter*"
step "anvil: real transactions" bash tools/gas_report.sh "$out" "$HOME/.cache/tensile-evm/out/NeuronTwin.sol/NeuronTwin.json"

echo
echo "=== summary ($(fe --version))"
printf '%s\n' "${summary[@]}"
printf 'total %.1f s, %d step(s) failed\n' "$(echo "$(date +%s.%N) - $total_start" | bc)" "$failures"
[ $failures -eq 0 ]
