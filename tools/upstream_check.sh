#!/bin/bash
# Reproduces each compiler defect listed in docs/upstream-reports.md on the installed Fe and prints the
# compiler's output and the time taken. Nothing is filed by this script.
#   wsl -e bash tools/upstream_check.sh
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
# FE=/path/to/fe re-runs everything on another build (for example a pinned master).
fe() { "${FE:-$HOME/.fe/bin/fe}" "$@"; }
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fe --version

timed() { # timed <limit-seconds> <command...>: output (first 25 lines), exit status and wall time.
  local limit="$1"
  shift
  local start=$(date +%s.%N)
  timeout "$limit" "$@" > "$work/o.txt" 2>&1
  local status=$?
  head -25 "$work/o.txt" | tr -d '\r' | cut -c1-200
  printf 'exit %d after %.1f s (limit %d s)\n' "$status" "$(echo "$(date +%s.%N) - $start" | bc)" "$limit"
}

cd "$root/spike/inference" || exit 2
echo; echo "=== 1. large array used in two messages: q6_array_twice.fe"
timed 300 fe test -O 1 q6_array_twice.fe
echo; echo "=== 2. compile time, large message arrays: q6_array_copy_stack.fe"
timed 300 fe test -O 1 q6_array_copy_stack.fe
echo; echo "=== 3. native read_word / write_word byte order"
cd "$root/tests/defects" || exit 2
echo "--- EVM"; timed 60 fe test --color never native_word_endianness.fe
echo "--- native"; timed 60 fe test --color never --backend native native_word_endianness.fe
cd "$root/spike/inference" || exit 2
echo; echo "=== 4. black_box on an array: q6_black_box.fe"
echo "--- fe check"; timed 60 fe check --color never q6_black_box.fe
echo "--- fe test"; timed 60 fe test --color never q6_black_box.fe
echo; echo "=== 5. no warning past 24,576 bytes: q6_const_size"
cp -r q6_const_size "$work/size"; rm -rf "$work/size/out"
( cd "$work/size" && fe build --color never --out-dir out . > build.log 2>&1; grep -c -i -E 'size|24,?576|limit' build.log; for f in out/*.runtime.bin; do echo "$f $(( $(tr -d '\n' < "$f" | wc -c) / 2 )) bytes"; done )
cd "$root" || exit 2
echo; echo "=== 6. internal error: tests/defects/store_array_row.fe"
timed 60 fe check --color never tests/defects/store_array_row.fe
echo; echo "=== 7. misleading diagnostics"
timed 60 fe check --color never tests/defects/event_fixed_array.fe
timed 60 fe check --color never tests/defects/param_named_like_const_fn.fe
