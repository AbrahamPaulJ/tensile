#!/bin/bash
# Milestone 1 check: regenerates the model ingot from ref/out and runs every test. Prints a summary
# with the wall time of each step and exits non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../fe-challenge/tools/m1_check.sh
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
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
    summary+=("$(printf 'ok    %7s s  %-28s %s' "$seconds" "$name" "$detail")")
  else
    summary+=("$(printf 'FAIL  %7s s  %-28s %s' "$seconds" "$name" "$detail")")
    failures=$((failures + 1))
  fi
}

format_check() {
  fe fmt --check ingots && echo "fe fmt --check: clean"
}

total_start=$(date +%s.%N)
echo "$(fe --version), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "generate model ingot" bash tools/gen_model.sh
step "fe fmt --check" format_check
for ingot in tensile model cli; do
  for backend in sonatina native; do
    step "fe test $ingot ($backend)" bash tools/test_ingot.sh $ingot $backend
  done
done
step "compile-fail cases" bash tools/check_compile_fail.sh
step "generator on test models" bash tools/check_generator.sh
step "native vs reference" bash tools/diff_native.sh

echo
echo "=== summary ($(fe --version))"
printf '%s\n' "${summary[@]}"
printf 'total %.1f s, %d step(s) failed\n' "$(echo "$(date +%s.%N) - $total_start" | bc)" "$failures"
[ $failures -eq 0 ]
