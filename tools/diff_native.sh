#!/bin/bash
# Builds the native classifier at -O 0, 1 and 2 (into a scratch directory) and compares each build
# with the NumPy reference vectors in ref/out (tools/diff_native.py). Zero mismatches is the pass
# condition.
cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
binaries=()
for level in 0 1 2; do
  start=$(date +%s.%N)
  bash tools/build_cli.sh $level "$work/O$level" > /dev/null || { echo "build at -O $level failed"; exit 1; }
  echo "built -O $level in $(echo "$(date +%s.%N) - $start" | bc) s ($(stat -c %s "$work/O$level/cli") bytes)"
  binaries+=("$work/O$level/cli")
done
python3 tools/diff_native.py "${binaries[@]}"
