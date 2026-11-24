#!/bin/bash
# Compile-fail tests of the library examples, in the style of tools/check_compile_fail.sh. Every
# lib-examples/compile_fail/*.fe is checked as the only source file of a scratch ingot that depends
# on `tensile`, `qmath`, `ema`, `score` and `curve`. Its header states what must happen:
#
#   // expect: <text>     `fe check` must fail and print <text> (several lines: all of them)
#   // expect-success     `fe check` must pass (the control case)
#
# usage: check_compile_fail.sh [-v]     (-v prints every diagnostic)
#        FE=/path/to/fe bash check_compile_fail.sh     (default: fe on the PATH)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
FE="${FE:-fe}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
verbose=$1
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# The same layout as the repository, so the ingots' path dependencies resolve unchanged.
mkdir -p "$work/ingots" "$work/lib-examples/ingots/case/src"
cp -r "$root/ingots/tensile" "$work/ingots/tensile"
for ingot in qmath ema score curve; do
  cp -r "$here/ingots/$ingot" "$work/lib-examples/ingots/$ingot"
done
cat > "$work/lib-examples/fe.toml" <<'EOF'
[workspace]
name = "lib-compile-fail"
version = "0.1.0"
members = [{ path = "ingots/qmath", name = "qmath" }, { path = "ingots/ema", name = "ema" }, { path = "ingots/score", name = "score" }, { path = "ingots/curve", name = "curve" }, { path = "ingots/case", name = "case" }]
EOF
cat > "$work/lib-examples/ingots/case/fe.toml" <<'EOF'
[ingot]
name = "case"
version = "0.1.0"

[dependencies]
tensile = { path = "../../../ingots/tensile" }
qmath = true
ema = true
score = true
curve = true
EOF

passed=0
failed=0
for file in "$here"/compile_fail/*.fe; do
  name="$(basename "$file" .fe)"
  cp "$file" "$work/lib-examples/ingots/case/src/lib.fe"
  output="$(cd "$work/lib-examples" && "$FE" check --color never ingots/case 2>&1)"
  status=$?
  verdict=ok
  if grep -q '^// expect-success' "$file"; then
    [ $status -eq 0 ] || verdict="should compile, but fe check failed"
  elif ! grep -q '^// expect: ' "$file"; then
    verdict="no '// expect:' line"
  elif [ $status -eq 0 ]; then
    verdict="compiled, but must not"
  else
    while IFS= read -r expected; do
      if ! grep -qF -- "$expected" <<< "$output"; then
        verdict="diagnostic lacks: $expected"
      fi
    done < <(sed -n 's|^// expect: ||p' "$file")
  fi
  if [ "$verdict" = ok ]; then
    passed=$((passed + 1))
    echo "ok    $name"
  else
    failed=$((failed + 1))
    echo "FAIL  $name: $verdict"
  fi
  if [ -n "$verbose" ] || [ "$verdict" != ok ]; then
    sed 's/^/      /' <<< "$output"
  fi
done

echo "lib compile-fail: $passed passed, $failed failed"
[ $failed -eq 0 ]
