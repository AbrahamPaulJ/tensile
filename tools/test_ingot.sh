#!/bin/bash
# Runs the unit tests of one ingot on one backend and prints the result line.
# usage: test_ingot.sh <ingot> <sonatina|native>
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
output="$(fe test --color never --backend "$2" "ingots/$1" 2>&1 | tr '\r' '\n')"
status=$?
if [ $status -ne 0 ]; then
  grep -v '^\s*$' <<< "$output" | grep -v '^PASS' | head -60
fi
echo "$1 on $2: $(grep 'test result' <<< "$output" | tail -1)"
exit $status
