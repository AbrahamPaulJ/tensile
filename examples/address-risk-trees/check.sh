#!/bin/bash
# Everything that is claimed about the address-risk forest (300 trees of depth 6), in one run, with
# the wall time of each step. Exits non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../examples/address-risk-trees/check.sh        (bash check.sh on Linux)
#
# It needs Fe 26.5.1, Foundry (anvil, cast) and Python 3, standard library only: the model and the
# reference vectors in model/ and out/ are committed; train.py, which needs the data set and NumPy,
# is not run here.
#
#  1. the test rows are the dense example's, byte for byte; the metrics in out/model.json are
#     recomputed from out/pred.bin and out/labels.bin;
#  2. the pure-Python reference (tools/trees_ref.py) runs again on all 7,054 test rows and must
#     reproduce out/pred.bin and out/trace.jsonl;
#  3. the workspace is regenerated from out/model.json and must come out as committed;
#  4. the `trees` kernel: `fe test` on both backends, and its compile-fail cases;
#  5. `fe test`: the model against SAMPLES reference vectors (EVM and native backends), the
#     generated arbiter's contract tests, the RiskGate consumer, the native classifier;
#  6. `fe build` and the 24,576-byte limit for every contract;
#  7. a native build of the classifier against the reference on all 7,054 test rows (classes) and
#     1,000 traces (every tree's output and both scores);
#  8. real transactions on anvil: the arbiter's calls, the whole forest on-chain, and the gate's
#     story end to end, each result checked;
#  9. RESULTS.md is rewritten from the records in runs/ (and, for the dense column of its tables,
#     from the records committed in examples/address-risk/runs/).
#
# Ports 18631 to 18633. Set SAMPLES to change the number of reference vectors compiled into the
# model ingot's tests (the committed sources use 8).
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$here"
SAMPLES="${SAMPLES:-8}"
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
  cmp out/inputs.bin ../address-risk/out/inputs.bin || return 1
  cmp out/labels.bin ../address-risk/out/labels.bin || return 1
  python3 - <<'EOF'
import json, sys
model = json.load(open("out/model.json", encoding="utf-8"))
n_in = model["in"]
inputs = open("out/inputs.bin", "rb").read()
labels = open("out/labels.bin", "rb").read()
pred = open("out/pred.bin", "rb").read()
if len(inputs) != n_in * len(labels) or len(pred) != len(labels) or max(inputs) > 127:
    sys.exit("out/: inputs, labels and predictions do not fit together")
tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
tn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 0)
n = len(labels)
got = {"samples": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / n,
       "precision": tp / (tp + fp), "recall": tp / (tp + fn),
       "f1": 2 * tp / (2 * tp + fp + fn), "majority_baseline": max(tp + fn, fp + tn) / n}
want = model["metrics"]
for key, value in got.items():
    if abs(value - want[key]) > 1e-12:
        sys.exit("out/model.json says %s is %r, the vectors say %r" % (key, want[key], value))
print("%s: %d trees of depth %d over %d features; the dense example's %d test rows: accuracy %.4f, "
      "precision %.4f, recall %.4f, F1 %.4f, majority baseline %.4f (recomputed from pred.bin and "
      "labels.bin)" % (model["name"], len(model["trees"]), model["depth"], n_in, n, got["accuracy"],
                       got["precision"], got["recall"], got["f1"], got["majority_baseline"]))
EOF
}

reference() {
  rm -rf build/reference
  python3 "$root/tools/trees_vectors.py" --model out/model.json --inputs out/inputs.bin --out build/reference > /dev/null || return 1
  for name in inputs.bin pred.bin trace.jsonl; do
    cmp "out/$name" "build/reference/$name" || return 1
  done
  echo "the pure-Python reference, run again on out/inputs.bin, reproduces out/pred.bin and out/trace.jsonl"
}

regenerate() {
  cp -r ingots "$work/before"
  cp fe.toml "$work/before.toml"
  bash "$root/tools/gen_trees_workspace.sh" out/model.json out . "$SAMPLES" | grep -v '^gen_trees' || return 1
  if [ "$SAMPLES" != 8 ]; then
    echo "generated with SAMPLES=$SAMPLES: not compared with the committed sources (8)"
  elif diff -r "$work/before" ingots > /dev/null && cmp -s "$work/before.toml" fe.toml; then
    echo "the regenerated workspace equals the committed one ($(find ingots -name '*.fe' | wc -l) Fe files)"
  else
    diff -rq "$work/before" ingots | head -10
    echo "the regenerated workspace differs from the committed one"
    return 1
  fi
}

format_check() {
  fe fmt --check ingots && fe fmt --check "$root/ingots/trees" && echo "fe fmt --check: clean (workspace and ingots/trees)"
}

# test_ingot <ingot dir> <sonatina|native>
test_ingot() {
  local output status
  output="$(set -o pipefail; fe test --color never --backend "$2" "$1" 2>&1 | tr '\r' '\n')"
  status=$?
  grep -q 'test result: ok' <<< "$output" || status=1
  if [ $status -ne 0 ]; then
    grep -v '^\s*$' <<< "$output" | grep -v '^PASS' | head -60
  fi
  echo "$(basename "$1") on $2: $(grep 'test result' <<< "$output" | tail -1)"
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
  python3 "$root/tools/diff_native_trees.py" --ref out build/native/cli | tee runs/native.txt
  return ${PIPESTATUS[0]}
}

total_start=$(date +%s.%N)
echo "$(fe --version), $(forge --version | head -1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "model file, vectors and metrics" vectors
step "pure-Python reference again" reference
step "regenerate the workspace" regenerate
step "fe fmt --check" format_check
step "fe test trees kernel (sonatina)" test_ingot "$root/ingots/trees" sonatina
step "fe test trees kernel (native)" test_ingot "$root/ingots/trees" native
step "trees compile-fail cases" bash "$root/tools/check_compile_fail_trees.sh"
step "fe test model (sonatina)" test_ingot ingots/model sonatina
step "fe test model (native)" test_ingot ingots/model native
step "fe test arbiter (sonatina)" test_ingot ingots/arbiter sonatina
step "fe test consumer (sonatina)" test_ingot ingots/consumer sonatina
step "fe test cli (native)" test_ingot ingots/cli native
step "fe build (all contracts)" build_contracts
step "runtime sizes" sizes
step "native classifier vs reference" native
step "anvil: arbiter transactions" bash "$root/tools/trees_gas.sh" build/out out build/forest.json runs/arbiter-gas.json 18631
step "anvil: whole forest on-chain" bash "$root/tools/ensemble_gas.sh" build/out out build/forest.json runs/ensemble-gas.json 18632
step "anvil: the gate, end to end" bash "$here/gate_anvil.sh" build/out out build/forest.json runs/gate-gas.json 18633
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
