#!/bin/bash
# Everything that is claimed about the library examples, in one run, with the wall time of each
# step. Exits non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../lib-examples/check.sh        (bash lib-examples/check.sh on Linux)
#
# It needs Fe (26.5.1, or the binary in $FE), Foundry (anvil, cast) and Python 3 (standard
# library only).
#
#  1. `fe fmt --check` and `fe check` of the workspace;
#  2. `fe test` on the EVM backend: the pure tests of qmath, ema, score and curve, and the
#     `deployed` tests (every contract against its pure functions and against its plain twin);
#  3. `fe test` on the native backend for the four ingots that do not deploy anything;
#  4. the compile-fail cases (check_compile_fail.sh): each scale mistake and failed overflow
#     proof must be the diagnostic its file states, and the control case must compile;
#  5. `fe build`, and gas.sh: real transactions on anvil, each typed contract and its twin on the
#     same inputs (results must be equal), gas and sizes written to runs/gas.json.
#
# Set FE to another binary (for example Fe 26.7.0, tools/check_fe267_lib.sh) to run steps 1 to 4
# and `fe build` with it; anvil is then not used and runs/gas.json is not written, because the
# committed figures are from 26.5.1. Port 18631, or the next free one.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
FE="${FE:-fe}"
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
log="$(mktemp)"
work="$(mktemp -d)"
trap 'rm -rf "$log" "$work"' EXIT
summary=()
failures=0

# step <name> <command...>: runs the command, shows its output, records result and wall time.
step() {
  local name="$1"
  shift
  echo "=== $name"
  local start=$(date +%s.%N)
  "$@" 2>&1 | tr '\r' '\n' | grep -v '^\s*$' | tee "$log"
  local status=${PIPESTATUS[0]}
  local seconds=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
  local detail="$(tail -1 "$log")"
  if [ $status -eq 0 ]; then
    summary+=("$(printf 'ok    %7s s  %-34s %s' "$seconds" "$name" "$detail")")
  else
    summary+=("$(printf 'FAIL  %7s s  %-34s %s' "$seconds" "$name" "$detail")")
    failures=$((failures + 1))
  fi
}

# A test run that reports no result line is a failure: it would pass without testing anything.
tests() {
  "$FE" test --color never "$@" > "$work/test.log" 2>&1
  local status=$?
  grep -v '^PASS\|^COMPILING\|^READY' "$work/test.log"
  [ $status -eq 0 ] && grep -q 'test result: ok' "$work/test.log"
}

record() {
  "$FE" build --color never --out-dir "$work/out" > /dev/null || return 1
  bash gas.sh "$work/out" runs/gas.json
}

build_only() {
  "$FE" build --color never --out-dir "$work/out" > /dev/null || return 1
  echo "fe build: $(ls "$work/out"/*.runtime.bin | wc -l) contracts"
}

echo "$("$FE" --version), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p runs
step "fe fmt --check" "$FE" fmt --check ingots
step "fe check" "$FE" check --color never
step "fe test (EVM)" tests
step "fe test (native)" tests --backend native ingots/qmath ingots/ema ingots/score ingots/curve
step "compile-fail cases" env FE="$FE" bash check_compile_fail.sh
if [ "$FE" = fe ]; then
  step "fe build, gas on anvil" record
else
  step "fe build" build_only
fi

echo
printf '%s\n' "${summary[@]}"
echo "lib-examples/check.sh: $failures step(s) failed"
[ $failures -eq 0 ]
