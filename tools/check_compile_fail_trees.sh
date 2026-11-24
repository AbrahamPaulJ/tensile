#!/bin/bash
# Compile-fail tests of the `trees` ingot: tools/check_compile_fail.sh for tests/compile_fail/trees.
# Every tests/compile_fail/trees/*.fe is checked as the only source file of a scratch ingot that
# depends on `tensile` and `trees`. Its header states what must happen:
#
#   // expect: <text>     `fe check` must fail and print <text> (several lines: all of them)
#   // expect-success     `fe check` must pass (the control case)
#
# A case that compiles when it should not, or fails with another diagnostic, fails the script.
# usage: check_compile_fail_trees.sh [-v] [fe binary]     (-v prints every diagnostic)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
verbose=""
if [ "$1" = -v ]; then
  verbose=1
  shift
fi
fe_bin="${1:-fe}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/ingots/case/src"
cp -r "$root/ingots/tensile" "$work/ingots/tensile"
cp -r "$root/ingots/trees" "$work/ingots/trees"
cat > "$work/fe.toml" <<'EOF'
[workspace]
name = "compile-fail-trees"
version = "0.1.0"
members = [{ path = "ingots/tensile", name = "tensile" }, { path = "ingots/trees", name = "trees" }, { path = "ingots/case", name = "case" }]
EOF
# The trees ingot names tensile by path; inside the scratch workspace that path is the same.
cat > "$work/ingots/case/fe.toml" <<'EOF'
[ingot]
name = "case"
version = "0.1.0"

[dependencies]
tensile = true
trees = true
EOF

passed=0
failed=0
for file in "$root"/tests/compile_fail/trees/*.fe; do
  name="$(basename "$file" .fe)"
  cp "$file" "$work/ingots/case/src/lib.fe"
  output="$(cd "$work" && "$fe_bin" check --color never ingots/case 2>&1)"
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

echo "compile-fail (trees): $passed passed, $failed failed"
[ $failed -eq 0 ]
