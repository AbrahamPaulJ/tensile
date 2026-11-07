#!/bin/bash
# Builds the native CLI at -O 1 into a scratch directory and runs the auditor differential test
# (tools/audit_diff.py) on the first N traced reference samples (default 100).
# usage: audit_diff.sh [N]
cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
bash tools/build_cli.sh 1 "$work" > /dev/null || { echo "build failed"; exit 1; }
python3 tools/audit_diff.py "$work/cli" --samples "${1:-100}"
