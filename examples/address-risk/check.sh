#!/bin/bash
# Everything that is claimed about the address-risk example (14 -> 32 -> 16 -> 2), in one run, with
# the wall time of each step. Exits non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../examples/address-risk/check.sh        (bash check.sh on Linux)
#
# It needs Fe 26.5.1, Foundry (anvil, cast) and Python 3 (standard library only: the model, the
# feature thresholds and the reference vectors in model/, features.json and out/ are committed;
# prepare.py, train.py and export.py, which need the data set, NumPy and pyarrow, are not run here).
#
#  1. the metrics in out/model.json are recomputed from out/pred.bin and out/labels.bin; if an
#     interpreter with NumPy is at hand (python3, or the one named by PYTHON_NUMPY), the NumPy
#     reference is run again on out/inputs.bin and must reproduce out/pred.bin and out/trace.jsonl;
#  2. the workspace is regenerated from out/model.json and must come out as committed;
#  3. `fe test`: the kernel against SAMPLES reference vectors (EVM and native backends), the
#     generated arbiter's contract tests, the RiskGate consumer;
#  4. `fe build` and the 24,576-byte limit for every contract;
#  5. a native build of the classifier against the reference on all 7,054 test rows (classes) and
#     1,000 traces (every hidden activation and logit);
#  6. real transactions on anvil: the arbiter's calls, one whole forward pass on-chain, and the
#     gate's story end to end, each result checked;
#  7. RESULTS.md is rewritten from the records in runs/ (and, for the two MNIST columns of its
#     table, from the records committed in examples/mnist-784-128-64-10/runs/).
#
# Ports 18615 to 18617. Set SAMPLES to change the number of reference vectors compiled into the
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
  python3 - <<'EOF'
import json, sys
model = json.load(open("out/model.json", encoding="utf-8"))
n_in = model["layers"][0]["in"]
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
sizes = [n_in] + [layer["out"] for layer in model["layers"]]
weights = sum(layer["in"] * layer["out"] for layer in model["layers"])
print("%s: %s, %d weights and %d biases; %d test rows: accuracy %.4f, precision %.4f, recall %.4f, "
      "majority baseline %.4f (recomputed from pred.bin and labels.bin)"
      % (model["name"], " -> ".join(str(v) for v in sizes), weights, sum(sizes[1:]), n,
         got["accuracy"], got["precision"], got["recall"], got["majority_baseline"]))
EOF
}

reference() {
  local python="${PYTHON_NUMPY:-python3}"
  if ! "$python" -c 'import numpy' > /dev/null 2>&1; then
    echo "skipped: no NumPy in $python (set PYTHON_NUMPY to an interpreter that has it)"
    return 0
  fi
  # Through a relative path, so that a Windows interpreter called from WSL can open it too.
  rm -rf build/reference
  "$python" ../../tools/ref_vectors.py --model out/model.json --inputs out/inputs.bin --out build/reference > /dev/null || return 1
  for name in inputs.bin pred.bin trace.jsonl; do
    cmp "out/$name" "build/reference/$name" || return 1
  done
  echo "the NumPy reference, run again on out/inputs.bin, reproduces out/pred.bin and out/trace.jsonl"
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
  python3 "$root/tools/diff_native.py" --ref out build/native/cli | tee runs/native.txt
  return ${PIPESTATUS[0]}
}

total_start=$(date +%s.%N)
echo "$(fe --version), $(forge --version | head -1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "model file, vectors and metrics" vectors
step "reference again (needs NumPy)" reference
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
step "anvil: arbiter transactions" bash "$root/tools/arbiter_gas_any.sh" build/out out build/tree.json runs/arbiter-gas.json 18615
step "anvil: whole network on-chain" bash "$root/tools/network_gas.sh" build/out out runs/network-gas.json 18616
step "anvil: the gate, end to end" bash "$here/gate_anvil.sh" build/out out build/tree.json runs/gate-gas.json 18617
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
