#!/bin/bash
# Compile-fail tests. Every tests/compile_fail/*.fe is checked as the only source file of a
# scratch ingot that depends on `tensile`. Its header states what must happen:
#
#   // expect: <text>     `fe check` must fail and print <text> (several lines: all of them)
#   // expect-success     `fe check` must pass (the control case)
#
# A case that compiles when it should not, or fails with another diagnostic, fails the script.
# usage: check_compile_fail.sh [-v]     (-v prints every diagnostic)
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
verbose=$1
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/ingots/case/src"
cp -r "$root/ingots/tensile" "$work/ingots/tensile"
cat > "$work/fe.toml" <<'EOF'
[workspace]
name = "compile-fail"
version = "0.1.0"
members = [{ path = "ingots/tensile", name = "tensile" }, { path = "ingots/case", name = "case" }]
EOF
cat > "$work/ingots/case/fe.toml" <<'EOF'
[ingot]
name = "case"
version = "0.1.0"

[dependencies]
tensile = true
EOF

passed=0
failed=0
for file in "$root"/tests/compile_fail/*.fe; do
  name="$(basename "$file" .fe)"
  cp "$file" "$work/ingots/case/src/lib.fe"
  output="$(cd "$work" && fe check --color never ingots/case 2>&1)"
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

echo "compile-fail: $passed passed, $failed failed"
[ $failed -eq 0 ]
