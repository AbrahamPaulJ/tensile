#!/bin/bash
# The identity check of tools/gen_arbiter.py.
#
# Generates the arbiter for the shipped model (ref/out/model.json), runs `fe fmt` on the result and
# compares it, byte for byte, with the hand-written sources in ingots/onchain/src:
#   arbiter.fe, record.fe, signals.fe, commitment.fe and arbiter_tests.fe.
# Any difference is printed and the script exits non-zero. Text identity of the sources implies
# identity of the compiled `Arbiter`; with --bytecode the script also builds both and compares the
# creation and runtime code, which checks that claim instead of assuming it.
#
# Because the texts are equal, everything measured on the hand-written arbiter (the Fe contract
# tests, the 45 Foundry tests, the Solidity twin, the mutation run) is evidence about the
# generator's output for a two-layer model. It is not evidence about models with more layers: those
# are covered by the generated contract tests of each example (examples/*/check.sh).
#
# Also checks that the generator is deterministic (two runs, identical bytes).
#
#   wsl -e bash /mnt/c/.../tools/check_arbiter_gen.sh [--bytecode]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
files="arbiter.fe record.fe signals.fe commitment.fe arbiter_tests.fe"
failed=0

python3 "$root/tools/gen_arbiter.py" --model "$root/ref/out/model.json" --vectors "$root/ref/out" \
  --out "$work/gen" > /dev/null || { echo "check_arbiter_gen: the generator failed"; exit 1; }
python3 "$root/tools/gen_arbiter.py" --model "$root/ref/out/model.json" --vectors "$root/ref/out" \
  --out "$work/again" > /dev/null
if diff -r "$work/gen" "$work/again" > /dev/null; then
  echo "ok    two runs give identical files"
else
  echo "FAIL  two runs differ"
  failed=1
fi
fe fmt "$work/gen" > /dev/null 2>&1
if ! fe fmt --check "$work/gen" > /dev/null 2>&1; then
  echo "FAIL  fe fmt does not settle on the generated files"
  failed=1
fi

for name in $files; do
  if cmp -s "$work/gen/$name" "$root/ingots/onchain/src/$name"; then
    printf 'ok    %-17s identical to ingots/onchain/src/%s (%d bytes)\n' "$name" "$name" "$(wc -c < "$work/gen/$name")"
  else
    echo "FAIL  $name differs from ingots/onchain/src/$name"
    diff "$work/gen/$name" "$root/ingots/onchain/src/$name" | head -${DIFF_LINES:-40}
    failed=1
  fi
done

if [ "$1" = "--bytecode" ]; then
  # The shipped workspace, built twice: once as it is, once with the generated files put in place
  # of the hand-written ones.
  mkdir -p "$work/ws"
  cp -r "$root/fe.toml" "$root/ingots" "$work/ws/"
  (cd "$work/ws" && fe build --color never --out-dir "$work/hand" --ingot onchain . > /dev/null 2>&1) \
    || { echo "FAIL  the shipped onchain ingot does not build"; failed=1; }
  for name in $files; do
    cp "$work/gen/$name" "$work/ws/ingots/onchain/src/$name"
  done
  (cd "$work/ws" && fe build --color never --out-dir "$work/built" --ingot onchain . > /dev/null 2>&1) \
    || { echo "FAIL  the generated arbiter does not build"; failed=1; }
  for artifact in Arbiter.bin Arbiter.runtime.bin; do
    if [ -s "$work/hand/$artifact" ] && cmp -s "$work/hand/$artifact" "$work/built/$artifact"; then
      printf 'ok    %-19s identical (%d bytes of code)\n' "$artifact" $(( $(tr -d '\n' < "$work/built/$artifact" | wc -c) / 2 ))
    else
      echo "FAIL  $artifact differs"
      failed=1
    fi
  done
fi

if [ $failed -eq 0 ]; then
  echo "check_arbiter_gen: the generated arbiter equals the hand-written one"
else
  echo "check_arbiter_gen: FAILED"
fi
exit $failed
