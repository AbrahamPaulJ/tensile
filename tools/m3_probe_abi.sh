#!/bin/bash
# What the Fe ABI decoder does with words a strict decoder refuses (docs/fe-notes.md M3.9): runs the
# testFuzz_ABI tests with full traces and prints every revert the Arbiter gives, with a count.
#   wsl -e bash /mnt/c/.../fe-challenge/tools/m3_probe_abi.sh
cd "$(dirname "$0")/.."
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
bash tools/m3_foundry.sh --color never --match-test 'testFuzz_ABI' -vvvvv > "$log" 2>&1
echo "exit $?, $(wc -l < "$log") log lines"
for name in bias_not_sign_extended class_word_with_high_bits logit_not_sign_extended_in_a_challenge logit_not_sign_extended_in_a_claim; do
  echo "--- $name"
  awk -v n="$name" 'index($0, "testFuzz_ABI_") { on = index($0, n) > 0 } on' "$log" \
    | grep -E 'Revert\]|Stop\]|Return\]|EvmError' | sed -E 's/^[ │├└─]+//' | cut -c1-90 | sort | uniq -c | sort -rn | head -8
done
grep -E '^(Suite result|Ran )' "$log"
