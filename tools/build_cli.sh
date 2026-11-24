#!/bin/bash
# Builds the native classifier (the `cli` ingot) as <out-dir>/cli.
# usage: build_cli.sh <0|1|2> <out-dir>
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
if [ $# -ne 2 ]; then
  echo "usage: build_cli.sh <0|1|2> <out-dir>" >&2
  exit 2
fi
fe build --color never --backend native -O "$1" --out-dir "$2" ingots/cli || exit 1
test -x "$2/cli"
