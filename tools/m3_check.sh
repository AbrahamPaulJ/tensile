#!/bin/bash
# Milestone 3 check: regenerates the model ingot and the neuron tree, checks the Python Keccak
# against `cast keccak`, runs the Fe tests of the state machine (both backends), the contracts and
# the auditor, builds the contracts and checks their sizes, runs the auditor differential test and
# measures the arbiter's real transactions on anvil. Prints a summary with the wall time of each
# step and exits non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../fe-challenge/tools/m3_check.sh
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
    summary+=("$(printf 'ok    %7s s  %-32s %s' "$seconds" "$name" "$detail")")
  else
    summary+=("$(printf 'FAIL  %7s s  %-32s %s' "$seconds" "$name" "$detail")")
    failures=$((failures + 1))
  fi
}

format_check() {
  fe fmt --check ingots && echo "fe fmt --check: clean"
}

# The pure-Python Keccak-256 behind the tree against Foundry's, on the empty string, a text and
# the first leaf preimage's tag.
keccak_check() {
  local ok=0
  for input in "" "tensile.neuron.v0" "0x$(head -c 64 /dev/zero | tr '\0' 'a' | xxd -p | tr -d '\n')"; do
    local ours theirs
    ours=$(python3 tools/keccak.py "$input")
    if [[ "$input" == 0x* ]]; then
      theirs=$(cast keccak "$input")
    else
      theirs=$(cast keccak "$input")
    fi
    [ "$ours" == "$theirs" ] || { echo "keccak mismatch on $input: $ours vs $theirs"; ok=1; }
  done
  [ $ok -eq 0 ] && echo "keccak.py agrees with cast keccak on 3 inputs"
  return $ok
}

build_contracts() {
  fe build --color never --out-dir "$out" --ingot onchain . && echo "built $(ls "$out"/*.runtime.bin | wc -l) contracts"
}

defect_check() {
  # The defect example must fail with the recorded diagnostic.
  if fe check --color never tests/defects/event_fixed_array.fe 2>&1 | grep -q "unsupported ABI field type"; then
    echo "tests/defects/event_fixed_array.fe reproduces error[10-0007]"
  else
    echo "tests/defects/event_fixed_array.fe no longer reproduces the defect"
    return 1
  fi
}

total_start=$(date +%s.%N)
echo "$(fe --version), $(forge --version | head -1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "generate model ingot and tree" bash tools/gen_model.sh
step "keccak.py vs cast keccak" keccak_check
step "fe fmt --check" format_check
for backend in sonatina native; do
  step "fe test dispute ($backend)" bash tools/test_ingot.sh dispute $backend
done
step "fe test tensile (sonatina)" bash tools/test_ingot.sh tensile sonatina
step "fe test cli (sonatina)" bash tools/test_ingot.sh cli sonatina
step "fe test cli (native)" bash tools/test_ingot.sh cli native
step "fe test onchain (sonatina)" bash tools/test_ingot.sh onchain sonatina
step "defect example" defect_check
step "fe build onchain" build_contracts
step "runtime sizes" bash tools/check_size.sh "$out"
step "auditor differential (P7)" bash tools/audit_diff.sh 100
step "anvil: arbiter transactions" bash tools/arbiter_gas.sh "$out"
step "forge: arbiter fuzz and invariants" bash tools/m3_foundry.sh

echo
echo "=== summary ($(fe --version))"
printf '%s\n' "${summary[@]}"
printf 'total %.1f s, %d step(s) failed\n' "$(echo "$(date +%s.%N) - $total_start" | bc)" "$failures"
[ $failures -eq 0 ]
