#!/bin/bash
# Everything that is claimed about the fee-tier example (144 -> 32 -> 16 -> 3), in one run, with the
# wall time of each step. Exits non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../examples/amm-fee/check.sh        (bash check.sh on Linux)
#
# It needs Fe 26.5.1, Foundry (anvil, cast) and Python 3 (standard library only: the tick series
# in data/, the model in model/ and its reference output in out/ are committed; fetch_data.py,
# train.py and export.py, which need the network or NumPy, are not run here). Where an interpreter
# with NumPy is at hand, set PYTHON_NUMPY to it and step 1 also re-runs export.py and requires
# out/ to come out byte for byte as committed.
#
#  1. the test inputs and labels are rebuilt from data/ticks.csv by dataset.py; the thresholds
#     must be the training split's quantiles; every score recorded in out/model.json is
#     recomputed from the committed predictions;
#  2. the workspace is regenerated from out/model.json and must come out as committed;
#  3. `fe test`: the kernel against SAMPLES reference vectors (EVM and native backends), the
#     generated arbiter's contract tests, the FeeOracle consumer;
#  4. `fe build` and the 24,576-byte limit for every contract;
#  5. a native build of the classifier against the reference on every test sample (classes) and
#     1,000 traces (every hidden activation and logit);
#  6. real transactions on anvil: the arbiter's calls, one whole forward pass on-chain, and the
#     FeeOracle fed the real ticks of sample 0 (oracle_check.sh), each result checked;
#  7. RESULTS.md is rewritten from the records in runs/.
#
# Ports 18621 to 18623. Set SAMPLES to change the number of reference vectors compiled into the
# model ingot's tests (the committed sources use 16).
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$here"
SAMPLES="${SAMPLES:-16}"
log="$(mktemp)"
work="$(mktemp -d)"
trap 'rm -rf "$log" "$work"' EXIT
mkdir -p runs build
steps="runs/steps.tsv"
: > "$steps"
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
  local seconds=$(python3 -c 'import sys, time; print("%.1f" % (time.time() - float(sys.argv[1])))' "$start")
  local detail="$(tail -1 "$log")"
  if [ $status -eq 0 ]; then
    summary+=("$(printf 'ok    %7s s  %-40s %s' "$seconds" "$name" "$detail")")
    printf 'ok\t%s\t%s\t%s\n' "$seconds" "$name" "$detail" >> "$steps"
  else
    summary+=("$(printf 'FAIL  %7s s  %-40s %s' "$seconds" "$name" "$detail")")
    printf 'FAIL\t%s\t%s\t%s\n' "$seconds" "$name" "$detail" >> "$steps"
    failures=$((failures + 1))
  fi
}

vectors() {
  if [ -n "$PYTHON_NUMPY" ]; then
    mkdir -p "$work/out"
    cp out/model.json out/pred.bin out/trace.jsonl "$work/out/"
    "$PYTHON_NUMPY" export.py > "$work/export.log" 2>&1 || { tail -20 "$work/export.log"; return 1; }
    for name in model.json pred.bin trace.jsonl; do
      cmp "out/$name" "$work/out/$name" || { echo "export.py did not reproduce the committed out/$name"; return 1; }
    done
    echo "export.py ($PYTHON_NUMPY) reproduced out/model.json, pred.bin and trace.jsonl byte for byte"
  fi
  python3 -I dataset.py --check
}

regenerate() {
  cp -r ingots "$work/before"
  cp fe.toml "$work/before.toml"
  bash "$root/tools/gen_workspace.sh" out/model.json out . "$SAMPLES" network | grep -v '^wrote' || return 1
  if [ "$SAMPLES" != 16 ]; then
    echo "generated with SAMPLES=$SAMPLES: not compared with the committed sources (16)"
  elif diff -r "$work/before" ingots > /dev/null && cmp -s "$work/before.toml" fe.toml; then
    echo "the regenerated workspace equals the committed one ($(find ingots -name '*.fe' | wc -l) Fe files)"
  else
    diff -rq "$work/before" ingots | head -10
    echo "the regenerated workspace differs from the committed one"
    return 1
  fi
}

format_check() {
  fe fmt --check ingots && echo "fe fmt --check: clean"
}

# test_ingot <ingot> <sonatina|native>
test_ingot() {
  local output status
  output="$(set -o pipefail; fe test --color never --backend "$2" "ingots/$1" 2>&1 | tr '\r' '\n')"
  status=$?
  grep -q 'test result: ok' <<< "$output" || status=1
  if [ $status -ne 0 ]; then
    grep -v '^\s*$' <<< "$output" | grep -v '^PASS' | head -60
  fi
  echo "$1 on $2: $(grep 'test result' <<< "$output" | tail -1)"
  return $status
}

build_contracts() {
  rm -rf build/out
  fe build --color never --out-dir build/out . > "$work/build.log" 2>&1 || { tail -20 "$work/build.log"; return 1; }
  echo "built $(ls build/out/*.runtime.bin | wc -l) contracts"
}

sizes() {
  bash "$root/tools/check_size.sh" build/out || return 1
  python3 - <<'EOF'
import glob, json, os
sizes = {os.path.basename(p)[:-len(".runtime.bin")]: len(open(p).read().strip()) // 2
         for p in sorted(glob.glob("build/out/*.runtime.bin"))}
with open("runs/sizes.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(sizes, f, indent=1)
    f.write("\n")
print("largest: %s, %d of 24576 bytes" % max(sizes.items(), key=lambda kv: kv[1]))
EOF
}

native() {
  rm -rf build/native
  local start=$(date +%s.%N)
  fe build --color never --backend native -O 1 --out-dir build/native --ingot cli . > "$work/native.log" 2>&1 \
    || { tail -20 "$work/native.log"; return 1; }
  echo "native build at -O 1: $(python3 -c 'import sys, time; print("%.1f" % (time.time() - float(sys.argv[1])))' "$start") s, $(stat -c %s build/native/cli) bytes"
  python3 "$root/tools/diff_native.py" --ref out build/native/cli | sed -E 's/, [0-9.]+ s$//' | tee runs/native.txt
  return ${PIPESTATUS[0]}
}

total_start=$(date +%s.%N)
echo "$(fe --version), $(forge --version | head -1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "data set, thresholds and scores" vectors
step "regenerate the workspace" regenerate
step "fe fmt --check" format_check
step "fe test model (sonatina)" test_ingot model sonatina
step "fe test model (native)" test_ingot model native
step "fe test arbiter (sonatina)" test_ingot arbiter sonatina
step "fe test consumer (sonatina)" test_ingot consumer sonatina
step "fe test cli (native)" test_ingot cli native
step "fe build (all contracts)" build_contracts
step "runtime sizes" sizes
step "native classifier vs reference" native
step "anvil: arbiter transactions" bash "$root/tools/arbiter_gas_any.sh" build/out out build/tree.json runs/arbiter-gas.json 18621
step "anvil: whole network on-chain" bash "$root/tools/network_gas.sh" build/out out runs/network-gas.json 18622
step "anvil: FeeOracle on real ticks" bash "$here/oracle_check.sh" build/out out build/tree.json runs/oracle-gas.json 18623
total=$(python3 -c 'import sys, time; print("%.1f" % (time.time() - float(sys.argv[1])))' "$total_start")
printf 'total\t%s\t%s\t%s\n' "$total" "$(fe --version)" "$SAMPLES" >> "$steps"
if [ $failures -eq 0 ]; then
  python3 results.py || failures=$((failures + 1))
fi

echo
echo "=== summary ($(fe --version))"
printf '%s\n' "${summary[@]}"
printf 'total %s s, %d step(s) failed\n' "$total" "$failures"
[ $failures -eq 0 ]
