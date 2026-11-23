#!/bin/bash
# Everything that is claimed about the 784 -> 128 -> 64 -> 10 example, in one run, with the wall
# time of each step. Exits non-zero if any step fails.
#
#   wsl -e bash /mnt/c/.../examples/mnist-784-128-64-10/check.sh        (bash check.sh on Linux)
#
# It needs Fe 26.5.1, Foundry (anvil, cast) and Python 3 (standard library only: the model and its
# reference vectors in model/ and out/ are committed; train.py and export.py, which need NumPy and
# MNIST, are not run here).
#
#  1. the workspace is regenerated from out/model.json and must come out as committed;
#  2. `fe test`: the kernel against SAMPLES reference vectors (EVM and native backends), the
#     generated arbiter's contract tests, the consumer example;
#  3. `fe build` and the 24,576-byte limit for every contract;
#  4. a native build of the classifier against the reference on all 10,000 test images (classes)
#     and 1,000 traces (every hidden activation and logit);
#  5. real transactions on anvil: the arbiter's calls and one whole forward pass on-chain, each
#     result checked; the same two scripts on the shipped 784 -> 32 -> 10 model, for the table;
#  6. RESULTS.md is rewritten from the records in runs/.
#
# Ports 18611 and 18612 (the repository's own scripts use others). Set SAMPLES to change the number
# of reference vectors compiled into the model ingot's tests (the committed sources use 16).
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
  # inputs.bin and labels.bin are the MNIST test set: the same bytes as ref/out's.
  for name in inputs.bin labels.bin; do
    [ -f "out/$name" ] || cp "$root/ref/out/$name" "out/$name"
    cmp "out/$name" "$root/ref/out/$name" || return 1
  done
  python3 - <<'EOF'
import json
model = json.load(open("out/model.json", encoding="utf-8"))
sizes = [model["layers"][0]["in"]] + [layer["out"] for layer in model["layers"]]
weights = sum(layer["in"] * layer["out"] for layer in model["layers"])
print("%s: %s, %d weights and %d biases, integer test accuracy %.4f (%d samples), float %.4f"
      % (model["name"], " -> ".join(str(n) for n in sizes), weights, sum(sizes[1:]),
         model["accuracy"]["integer"], model["accuracy"]["test_samples"], model["accuracy"]["float"]))
EOF
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

model1() {
  # The shipped two-layer model through the same two scripts, for the side-by-side table.
  (cd "$root" && fe build --color never --out-dir "$work/model1" --ingot onchain . > "$work/model1.log" 2>&1) \
    || { tail -20 "$work/model1.log"; return 1; }
  python3 "$root/tools/gen_tree.py" --model "$root/ref/out/model.json" --out "$work/model1-tree.json" > /dev/null
  bash "$root/tools/arbiter_gas_any.sh" "$work/model1" "$root/ref/out" "$work/model1-tree.json" runs/model1-arbiter-gas.json 18611 || return 1
  bash "$root/tools/network_gas.sh" "$work/model1" "$root/ref/out" runs/model1-network-gas.json 18612
}

total_start=$(date +%s.%N)
echo "$(fe --version), $(forge --version | head -1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "model file and vectors" vectors
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
step "anvil: arbiter transactions" bash "$root/tools/arbiter_gas_any.sh" build/out out build/tree.json runs/arbiter-gas.json 18611
step "anvil: whole network on-chain" bash "$root/tools/network_gas.sh" build/out out runs/network-gas.json 18612
step "the shipped model, same scripts" model1
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
