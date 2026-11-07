#!/bin/bash
# Runs the Foundry tests in evm/ (the differential test of the Fe contracts against the reference
# vectors, and the fuzz tests) against the Fe artifacts in <fe-out-dir>.
#
# Foundry is slow on /mnt/c, so the project is staged in a directory under $HOME: evm/ is copied
# there, the Fe artifacts into its fe-out/ and ref/out into its ref-out/, the two paths the tests
# read. The forge build cache in the staging directory is kept between runs.
# usage: evm_test.sh <fe-out-dir> [forge test options...]
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ $# -lt 1 ] || [ ! -d "$1" ]; then
  echo "usage: evm_test.sh <fe-out-dir> [forge test options...]" >&2
  exit 2
fi
fe_out="$(cd "$1" && pwd)"
shift
stage="$HOME/.cache/tensile-evm"
mkdir -p "$stage"
rm -rf "$stage/src" "$stage/test" "$stage/fe-out" "$stage/ref-out" "$stage/invariant-out"
cp -r "$root/evm/src" "$root/evm/test" "$root/evm/foundry.toml" "$stage/"
mkdir -p "$stage/fe-out" "$stage/ref-out" "$stage/invariant-out"
cp "$fe_out"/*.bin "$stage/fe-out/"
cp "$root"/ref/out/inputs.bin "$root"/ref/out/pred.bin "$root"/ref/out/trace.jsonl "$stage/ref-out/"
cd "$stage"
forge test "$@"
