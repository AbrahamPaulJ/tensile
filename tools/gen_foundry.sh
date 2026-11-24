#!/bin/bash
# The Foundry fuzz and invariant suite of a GENERATED arbiter (evm/generated/GenArbiter*.t.sol),
# run once per example against the arbiter that tools/gen_arbiter.py wrote for it. The suite of the
# shipped two-layer arbiter (evm/test/Arbiter*.t.sol, tools/m3_foundry.sh) is not touched.
#
# For each example:
#   1. its workspace (examples/<name>/fe.toml and ingots/, with this repository's tensile and
#      dispute ingots at the same relative place) is copied under $HOME, where builds are fast,
#      and the arbiter ingot is built there with `fe build --ingot arbiter`;
#   2. the neuron tree is rebuilt from out/model.json (tools/gen_tree.py);
#   3. the fixture file is written from out/model.json, the reference trace and the tree
#      (tools/gen_arbiter_fixtures_any.py, which checks every sample against its own forward pass);
#   4. evm/generated/*.sol, the shared helpers of evm/test (ArbiterBase.sol, FeDeployer.sol,
#      SpecNeuron.sol, Vm.sol) and foundry.toml are staged with the bytecode and the fixture, and
#      `forge test` runs.
# Then the invariant campaign's counters are added up per example (an outcome that never happened
# fails the run, as in tools/m3_foundry.sh), and one line per example gives the result and the
# wall time.
#
#   wsl -e bash /mnt/c/.../tools/gen_foundry.sh [example...] [-- forge test options...]
#
# Examples default to `amm-fee address-risk`; `mnist-784-128-64-10` is also accepted. Any options
# after `--` go to `forge test` as they are (for example `--match-test testFuzz_P3`).
# GEN_QUICK=1 runs every fuzz test with 64 cases and the invariant campaign with 32 runs of depth
# 50 (used for the large MNIST model): the staged copies of the tests get those counts in their
# `forge-config` lines, which a Foundry profile cannot override.
# GEN_STAGE overrides the staging directory ($HOME/.cache/tensile-gen); tools/gen_mutants.py uses
# that to run the suite against a mutated build.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
examples=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  examples+=("$1")
  shift
done
[ "$1" = "--" ] && shift
[ ${#examples[@]} -eq 0 ] && examples=(amm-fee address-risk)
cache="${GEN_STAGE:-$HOME/.cache/tensile-gen}"
now() { date +%s.%N; }
since() { python3 -c 'import sys, time; print("%.1f" % (time.time() - float(sys.argv[1])))' "$1"; }
total_start=$(now)
summary=()
failures=0

# prepare <example> <work dir>: Arbiter.bin, tree.json and generated.bin in <work dir>.
prepare() {
  local ex="$1" work="$2" dir="examples/$1"
  mkdir -p "$work/ws/ingots" "$work/ws/examples/$ex" "$work/fe-out"
  # The workspace copy, rebuilt only when a source changed (a hash of every file that goes in).
  local stamp
  stamp="$(cat ingots/tensile/fe.toml ingots/dispute/fe.toml "$dir/fe.toml" \
    $(find ingots/tensile ingots/dispute "$dir/ingots" -name '*.fe' -o -name 'fe.toml' | sort) | sha256sum | cut -c1-16)"
  if [ ! -s "$work/fe-out/Arbiter.bin" ] || [ "$(cat "$work/stamp" 2> /dev/null)" != "$stamp" ]; then
    rm -rf "$work/ws" "$work/fe-out"
    mkdir -p "$work/ws/ingots" "$work/ws/examples/$ex" "$work/fe-out"
    cp -r ingots/tensile ingots/dispute "$work/ws/ingots/"
    cp -r "$dir/fe.toml" "$dir/ingots" "$work/ws/examples/$ex/"
    echo "=== fe build --ingot arbiter ($ex)"
    (cd "$work/ws/examples/$ex" && fe build --color never --out-dir "$work/fe-out" --ingot arbiter .) \
      > "$work/build.log" 2>&1 || { tail -20 "$work/build.log"; return 1; }
    echo "$stamp" > "$work/stamp"
  fi
  python3 tools/gen_tree.py --model "$dir/out/model.json" --out "$work/tree.json" > /dev/null || return 1
  local inputs=()
  case "$ex" in
    amm-fee)
      # The test inputs are rebuilt from the committed tick series (as check.sh step 1 does).
      [ -f "$dir/out/inputs.bin" ] || python3 -I "$dir/dataset.py" > /dev/null || return 1
      ;;
    mnist-784-128-64-10)
      # The MNIST test set: the same bytes as ref/out's (check.sh step 1).
      inputs=(--inputs ref/out/inputs.bin)
      ;;
  esac
  python3 tools/gen_arbiter_fixtures_any.py --ref "$dir/out" --tree "$work/tree.json" "${inputs[@]}" \
    --out "$work/generated.bin"
}

# stage <work dir> <stage dir>: the Foundry project.
stage() {
  local work="$1" st="$2"
  mkdir -p "$st"
  rm -rf "$st/src" "$st/test" "$st/fe-out" "$st/invariant-out"
  mkdir -p "$st/src" "$st/test/fixtures" "$st/fe-out" "$st/invariant-out"
  cp evm/foundry.toml "$st/"
  cp evm/test/ArbiterBase.sol evm/test/FeDeployer.sol evm/test/SpecNeuron.sol evm/test/Vm.sol "$st/test/"
  cp evm/generated/*.sol "$st/test/"
  if [ -n "$GEN_QUICK" ]; then
    sed -i -E -e 's/(forge-config: default\.fuzz\.runs = )[0-9]+/\164/' \
      -e 's/(forge-config: default\.invariant\.runs = )[0-9]+/\132/' \
      -e 's/(forge-config: default\.invariant\.depth = )[0-9]+/\150/' "$st"/test/GenArbiter*.t.sol
    sed -i -E 's/^runs = 256$/runs = 64/' "$st/foundry.toml"
  fi
  cp "$work/fe-out/Arbiter.bin" "$st/fe-out/"
  cp "$work/generated.bin" "$st/test/fixtures/generated.bin"
}

for ex in "${examples[@]}"; do
  start=$(now)
  if [ ! -f "examples/$ex/out/model.json" ]; then
    echo "gen_foundry: no example $ex"
    summary+=("FAIL  $ex: no such example")
    failures=$((failures + 1))
    continue
  fi
  work="$cache/$ex/work"
  st="$cache/$ex/stage"
  echo "=== $ex"
  if ! prepare "$ex" "$work"; then
    summary+=("$(printf 'FAIL  %-22s preparation failed, %s s' "$ex" "$(since $start)")")
    failures=$((failures + 1))
    continue
  fi
  stage "$work" "$st"
  log="$work/forge.log"
  (cd "$st" && forge test --color never "$@") 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  campaign="$st/invariant-out/campaign.log"
  counts=""
  if [ -s "$campaign" ]; then
    # Per run: claims, then slashes by a neuron of each layer and by the class, finals, payouts,
    # refused calls (GenArbiterInvariantTest.afterInvariant).
    counts="$(awk '{ for (i = 1; i <= NF; i++) t[i] += $i; n = NF }
      END {
        s = sprintf("%d runs, %d claims, slashes", NR, t[1])
        for (i = 2; i <= n - 4; i++) s = s sprintf(" %d (layer %d)", t[i], i - 1)
        s = s sprintf(" %d (class), %d finals, %d payouts, %d refused calls", t[n - 3], t[n - 2], t[n - 1], t[n])
        print s
        for (i = 1; i <= n; i++) if (t[i] == 0) { print "an outcome never happened"; exit 1 }
      }' "$campaign")" || status=1
    echo "invariant campaign ($ex): $counts"
  fi
  result="$(grep -E '^Ran [0-9]+ test suites?' "$log" | tail -1)"
  [ -n "$result" ] || result="no forge summary (build failure?)"
  if [ $status -eq 0 ]; then
    summary+=("$(printf 'ok    %-22s %s, %s s' "$ex" "$result" "$(since $start)")")
  else
    summary+=("$(printf 'FAIL  %-22s %s, %s s' "$ex" "$result" "$(since $start)")")
    failures=$((failures + 1))
  fi
  [ -n "$counts" ] && summary+=("      $(printf '%-22s' '') invariant campaign: ${counts//$'\n'/; }")
done

echo
echo "=== gen_foundry summary ($(fe --version), $(forge --version | head -1)$([ -n "$GEN_QUICK" ] && echo ', GEN_QUICK: 64 fuzz cases per test, invariant 32 runs x depth 50'))"
printf '%s\n' "${summary[@]}"
echo "gen_foundry: $failures failure(s), wall $(since $total_start) s"
[ $failures -eq 0 ]
