#!/bin/bash
# Development loop for milestone 3: formats the model ingot, then runs the tests named on the
# command line as <ingot>:<backend> pairs, e.g.  m3_dev.sh dispute:native dispute:sonatina onchain:sonatina
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
fe fmt ingots/model > /dev/null
status=0
for pair in "$@"; do
  ingot="${pair%%:*}"
  backend="${pair##*:}"
  echo "=== fe test $ingot ($backend)"
  bash tools/test_ingot.sh "$ingot" "$backend" || status=1
done
exit $status
