#!/bin/bash
# Everything that is measured about the GENERATED three-layer arbiters, in one run, with the wall
# time of each step:
#
#  1. the Foundry fuzz and invariant suite (evm/generated/GenArbiter*.t.sol) against the generated
#     arbiters of examples/amm-fee and examples/address-risk, at the run counts in the sources
#     (tools/gen_foundry.sh);
#  2. the same suite against examples/mnist-784-128-64-10 with GEN_QUICK=1 (64 fuzz cases per test,
#     32 invariant runs of depth 50): its 784-128 first layer makes the oracle the slow part, so the
#     full counts are not run there by default;
#  3. with --mutants, the mutation run against the amm-fee arbiter (tools/gen_mutants.py --fresh),
#     which rewrites docs/mutation-results-generated.md. It is long, so it is not part of CI, as
#     tools/m3_mutants.py is not.
#
#   wsl -e bash /mnt/c/.../tools/gen_checks.sh [--mutants]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
summary=()
failures=0
now() { date +%s.%N; }
since() { python3 -c 'import sys, time; print("%.1f" % (time.time() - float(sys.argv[1])))' "$1"; }

# step <name> <command...>: runs the command, shows its output, records result and wall time.
step() {
  local name="$1"
  shift
  echo "=== $name"
  local start=$(now)
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  local detail="$(grep -E '^(gen_foundry|gen_mutants):' "$log" | tail -1)"
  if [ $status -eq 0 ]; then
    summary+=("$(printf 'ok    %7s s  %-44s %s' "$(since $start)" "$name" "$detail")")
  else
    summary+=("$(printf 'FAIL  %7s s  %-44s %s' "$(since $start)" "$name" "$detail")")
    failures=$((failures + 1))
  fi
}

total_start=$(now)
echo "$(fe --version), $(forge --version | head -1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "suite: amm-fee and address-risk (full counts)" bash tools/gen_foundry.sh amm-fee address-risk
step "suite: mnist-784-128-64-10 (GEN_QUICK)" env GEN_QUICK=1 bash tools/gen_foundry.sh mnist-784-128-64-10
if [ "$1" = "--mutants" ]; then
  step "mutation run: amm-fee arbiter" python3 tools/gen_mutants.py --fresh
fi
echo
echo "=== gen_checks summary"
printf '%s\n' "${summary[@]}"
printf 'total %s s, %d step(s) failed\n' "$(since $total_start)" "$failures"
[ $failures -eq 0 ]
