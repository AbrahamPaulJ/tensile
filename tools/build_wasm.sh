#!/bin/bash
# Builds the WebAssembly module of the kernel and checks it.
#
#   wsl -e bash tools/build_wasm.sh            (from PowerShell; bash tools/build_wasm.sh on Linux)
#
# The WebAssembly target is in the Fe release binaries from 26.7.0, and from that release it builds
# ingots, so the module is the ingot `wasm/` (its exports, `wasm/src/lib.fe`) on top of
# `ingots/tensile` and `ingots/model` as they stand. Everything else in the project uses Fe 26.5.1
# (README). Set FE to the 26.7.0 release binary (default ~/fe-267/fe):
#   https://github.com/argotorg/fe/releases/download/v26.7.0/fe_linux_amd64
#   sha256 4ef2b818e260f0ba0b0af4bd78468dda589dc4949503fffaef25d8188e5107d5
#
# Steps: run every kernel and model unit test on the wasm backend, build the module into wasm/out/,
# build it a second time and require identical bytes, then the 10,000-image differential test
# against the NumPy reference, at -O 1 (the committed module) and again at -O 0 and -O 2.
set -u
WANT="26.7.0"   # keep in step with the wasm job in .github/workflows/ci.yml
FE="${FE:-$HOME/fe-267/fe}"
[ -d "$HOME/.local/node/bin" ] && export PATH="$HOME/.local/node/bin:$PATH"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root" || exit 2
fail=0
step() { echo; echo "=== $*"; }

"$FE" --version || { echo "FAIL no fe binary at $FE (set FE to the Fe $WANT release binary)"; exit 1; }
case "$("$FE" --version)" in *"fe $WANT "*) ;; *) echo "FAIL fe is not the $WANT release"; exit 1 ;; esac

step "fe fmt --check wasm"
"$FE" fmt --check wasm && echo "clean" || fail=1

step "unit tests on the wasm backend"
start=$(date +%s)
for ingot in tensile model; do
  result="$("$FE" test --backend wasm --color never "ingots/$ingot" 2>&1 | tr '\r' '\n' | grep 'test result' | tail -1)"
  echo "$ingot: $result"
  case "$result" in *"test result: ok."*) ;; *) fail=1 ;; esac
done
echo "$(( $(date +%s) - start )) s"

step "build, twice, and compare"
rm -rf wasm/out wasm/out-again
"$FE" build --backend wasm --color never --out-dir wasm/out wasm || fail=1
"$FE" build --backend wasm --color never --out-dir wasm/out-again wasm > /dev/null || fail=1
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
  "$FE" build --backend wasm -O "$o" --color never --out-dir "$d" wasm > /dev/null || fail=1
  node wasm/diff.mjs "$d/tensile_wasm.wasm" || fail=1
  rm -rf "$d"
done

echo
[ "$fail" = 0 ] && echo "build_wasm: all steps passed" || echo "build_wasm: FAILED"
exit "$fail"
