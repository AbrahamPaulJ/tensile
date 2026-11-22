#!/bin/bash
# Builds the WebAssembly module of the kernel and checks it.
#
#   wsl -e bash tools/build_wasm.sh            (from PowerShell; bash tools/build_wasm.sh on Linux)
#
# The WebAssembly target was on Fe master only when this was built (argotorg/fe PR 1904, merged
# 2026-11-05; released in 26.7.0), so this uses a pinned master commit, which the organisers allow. Everything else in
# the project uses the release in `fe.toml`/README. Set FE to a fe binary built at that commit with
# `cargo build --release -p fe --features wasm` (spike/wasm/00_build_master.sh does that in ~8 min).
#
# Steps: flatten the ingots into wasm/tensile_wasm.fe (tools/gen_wasm.py), run every kernel and
# model unit test on the wasm backend, build the module into wasm/out/, build it a second time and
# require identical bytes, then the 10,000-image differential test against the NumPy reference, at
# -O 1 (the committed module) and again at -O 0 and -O 2.
set -u
PIN=89d07b9dbf7ea73d0ca6b9a5b4f14de7899d1908   # keep in step with FE_MASTER in .github/workflows/ci.yml
FE="${FE:-$HOME/fe-master/target/release/fe}"
[ -d "$HOME/.local/node/bin" ] && export PATH="$HOME/.local/node/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root" || exit 2
fail=0
step() { echo; echo "=== $*"; }

"$FE" --version
case "$("$FE" --version)" in *"${PIN:0:7}"*) ;; *) echo "FAIL fe is not the pinned commit ${PIN:0:7}"; exit 1 ;; esac

step "flatten the ingots"
python3 tools/gen_wasm.py || fail=1

step "unit tests on the wasm backend"
start=$(date +%s)
"$FE" test --backend wasm --color never wasm/tensile_wasm.fe | tail -3 || fail=1
"$FE" test --backend wasm --color never wasm/tensile_wasm.fe > /dev/null 2>&1 || fail=1
echo "$(( $(date +%s) - start )) s"

step "build, twice, and compare"
rm -rf wasm/out wasm/out-again
"$FE" build --backend wasm --color never --out-dir wasm/out wasm/tensile_wasm.fe || fail=1
"$FE" build --backend wasm --color never --out-dir wasm/out-again wasm/tensile_wasm.fe > /dev/null || fail=1
if cmp -s wasm/out/tensile_wasm.wasm wasm/out-again/tensile_wasm.wasm \
  && cmp -s wasm/out/tensile_wasm.manifest.json wasm/out-again/tensile_wasm.manifest.json; then
  echo "identical: $(sha256sum wasm/out/tensile_wasm.wasm | cut -c1-16)..., $(stat -c %s wasm/out/tensile_wasm.wasm) bytes"
else
  echo "FAIL the two builds differ"; fail=1
fi
rm -rf wasm/out-again

step "differential test against the NumPy reference"
node wasm/diff.mjs || fail=1

step "the same at -O 0 and -O 2"
for o in 0 2; do
  d="$(mktemp -d)"
  "$FE" build --backend wasm -O "$o" --color never --out-dir "$d" wasm/tensile_wasm.fe > /dev/null || fail=1
  node wasm/diff.mjs "$d/tensile_wasm.wasm" || fail=1
  rm -rf "$d"
done

echo
[ "$fail" = 0 ] && echo "build_wasm: all steps passed" || echo "build_wasm: FAILED"
exit "$fail"
