#!/bin/bash
# Development: run the milestone 3, 1 and 2 checks one after another, keeping each full log in
# /tmp and printing only the summaries.
cd "$(dirname "$0")/.."
for m in 3 1 2; do
  log="/tmp/m${m}_check.log"
  bash "tools/m${m}_check.sh" > "$log" 2>&1
  echo "##### m${m}_check.sh exit $? (log $log)"
  sed -n '/^=== summary/,$p' "$log"
done
