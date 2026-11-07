#!/bin/bash
# Tests of tools/gen_fe.py against the model files in tests/models.
#
# Each model is generated into a scratch workspace and checked with `fe check`. Its "expect" key
# states what must happen: null means the generated ingot compiles; a string means it must NOT
# compile, and the diagnostic must contain that string (a string that starts with "gen_fe:" is an
# error of the generator itself). This is how a network whose layers do not chain is caught: by
# the Fe type checker, in the generated code.
#
# Also checks that the generator is deterministic (two runs, identical bytes).
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/ingots/model"
cp -r "$root/ingots/tensile" "$work/ingots/tensile"
cp "$root/ingots/model/fe.toml" "$work/ingots/model/fe.toml"
cat > "$work/fe.toml" <<'EOF'
[workspace]
name = "generator-check"
version = "0.1.0"
members = [{ path = "ingots/tensile", name = "tensile" }, { path = "ingots/model", name = "model" }]
EOF

passed=0
failed=0
report() {
  if [ "$2" = ok ]; then
    passed=$((passed + 1))
    echo "ok    $1"
  else
    failed=$((failed + 1))
    echo "FAIL  $1: $2"
    [ -z "$3" ] || sed 's/^/      /' <<< "$3"
  fi
}

for file in "$root"/tests/models/*.json; do
  name="$(basename "$file" .json)"
  expected="$(python3 -c 'import json, sys; e = json.load(open(sys.argv[1]))["expect"]; print("" if e is None else e)' "$file")"
  rm -rf "$work/ingots/model/src"
  if ! output="$(python3 "$root/tools/gen_fe.py" --model "$file" --out "$work/ingots/model/src" 2>&1)"; then
    if [[ "$expected" == gen_fe:* ]] && grep -qF -- "$expected" <<< "$output"; then
      report "$name (rejected: $expected)" ok
    else
      report "$name" "the generator rejected the model" "$output"
    fi
    continue
  fi
  output="$(cd "$work" && fe check --color never ingots/model 2>&1)"
  status=$?
  if [ -z "$expected" ]; then
    [ $status -eq 0 ] && report "$name (compiles)" ok || report "$name" "should compile" "$output"
  elif [ $status -eq 0 ]; then
    report "$name" "compiled, but must not"
  elif grep -qF -- "$expected" <<< "$output"; then
    report "$name (rejected: $expected)" ok
  else
    report "$name" "diagnostic lacks: $expected" "$output"
  fi
done

python3 "$root/tools/gen_fe.py" --vectors "$root/ref/out" --out "$work/first" > /dev/null
python3 "$root/tools/gen_fe.py" --vectors "$root/ref/out" --out "$work/second" > /dev/null
if diff -r "$work/first" "$work/second" > /dev/null; then
  report "two runs on ref/out give identical files" ok
else
  report "determinism" "two runs on ref/out differ"
fi

echo "generator: $passed passed, $failed failed"
[ $failed -eq 0 ]
