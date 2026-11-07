#!/bin/bash
# Prints the runtime code size of every contract built by `fe build` and fails if one exceeds the
# EIP-170 limit of 24,576 bytes, which `fe build` itself does not check.
# usage: check_size.sh <out-dir>   (a directory with <Contract>.runtime.bin files)
if [ $# -ne 1 ] || [ ! -d "$1" ]; then
  echo "usage: check_size.sh <out-dir>" >&2
  exit 2
fi
limit=24576
status=0
found=0
for file in "$1"/*.runtime.bin; do
  [ -e "$file" ] || continue
  found=1
  bytes=$(( $(tr -d '\n' < "$file" | wc -c) / 2 ))
  name=$(basename "$file" .runtime.bin)
  if [ $bytes -le $limit ]; then
    printf 'ok    %-14s %6d bytes (%d under the limit)\n' "$name" $bytes $((limit - bytes))
  else
    printf 'FAIL  %-14s %6d bytes (%d over the limit)\n' "$name" $bytes $((bytes - limit))
    status=1
  fi
done
if [ $found -eq 0 ]; then
  echo "check_size: no *.runtime.bin in $1" >&2
  exit 2
fi
exit $status
