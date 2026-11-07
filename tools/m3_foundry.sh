#!/bin/bash
# Milestone 3: the Foundry fuzz and invariant tests of the Arbiter (evm/test/Arbiter*.t.sol,
# properties P1 to P6 of docs/spec-dispute.md section 9; docs/fe-notes.md M3.9).
#
# Builds the Fe contracts into $HOME/.cache/tensile-fe-out when any .fe file is newer than the
# last build, regenerates the fixture file (tools/gen_arbiter_fixtures.py), then runs forge through
# tools/evm_test.sh (which stages evm/ under $HOME, where Foundry is fast). With no arguments it
# runs only the Arbiter tests; any arguments replace that filter and go to `forge test` as they
# are (for example `--match-test testFuzz_P3` while developing). Prints a summary line with the
# wall time and exits non-zero if anything failed.
#
#   wsl -e bash /mnt/c/.../fe-challenge/tools/m3_foundry.sh [forge test options...]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
start=$(date +%s.%N)
fe_out="$HOME/.cache/tensile-fe-out"
elapsed() { printf '%.1f' "$(echo "$(date +%s.%N) - $start" | bc)"; }

if [ ! -f build/tree.json ]; then
  echo "=== build/tree.json missing: tools/gen_model.sh"
  bash tools/gen_model.sh || { echo "m3_foundry: FAIL (gen_model.sh), $(elapsed) s"; exit 1; }
fi
if [ ! -f "$fe_out/Arbiter.bin" ] || [ -n "$(find ingots -name '*.fe' -newer "$fe_out/Arbiter.bin" -print -quit)" ]; then
  echo "=== fe build onchain -> $fe_out"
  rm -rf "$fe_out"
  mkdir -p "$fe_out"
  fe build --color never --out-dir "$fe_out" --ingot onchain . \
    || { echo "m3_foundry: FAIL (fe build), $(elapsed) s"; exit 1; }
fi
python3 tools/gen_arbiter_fixtures.py || { echo "m3_foundry: FAIL (fixtures), $(elapsed) s"; exit 1; }

if [ $# -eq 0 ]; then
  set -- --match-path "test/Arbiter*.t.sol"
fi
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
bash tools/evm_test.sh "$fe_out" "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

# What the invariant campaign did, one line per run (ArbiterInvariantTest.afterInvariant): an
# outcome that never happened means the invariant passed without testing it.
campaign="$HOME/.cache/tensile-evm/invariant-out/campaign.log"
if [ -s "$campaign" ]; then
  awk '{ for (i = 1; i <= 7; i++) t[i] += $i }
    END {
      printf "invariant campaign: %d runs, %d claims, slashes %d (layer 1) %d (layer 2) %d (class), %d finals, %d payouts, %d refused calls\n",
        NR, t[1], t[2], t[3], t[4], t[5], t[6], t[7]
      for (i = 1; i <= 7; i++) if (t[i] == 0) { print "invariant campaign: an outcome never happened"; exit 1 }
    }' "$campaign" || status=1
fi
result="$(grep -E '^Ran [0-9]+ test suites?' "$log" | tail -1)"
[ -n "$result" ] || result="no forge summary (build failure?)"
if [ $status -eq 0 ]; then
  echo "m3_foundry: ok, $result, wall $(elapsed) s"
else
  echo "m3_foundry: FAIL (forge exit $status), $result, wall $(elapsed) s"
fi
exit $status
