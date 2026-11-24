#!/bin/bash
# Builds what the fee-tier page embeds: `fe build` of examples/amm-fee (Arbiter, Network, Rows0,
# FeeOracle, TestPool), then demo/generated/fee-assets.json (demo/fee_assets.py). The page itself is
# then built with `npm run build:fee` in demo/ (Node; no Fe needed for that step).
#
#   fee_assets.sh            writes demo/generated/fee-assets.json
#   fee_assets.sh --check    writes nothing: builds again and requires the committed file to be
#                            byte for byte what this compiler and these sources give, so the
#                            bytecode in the page is exactly `fe build`'s
#
# Needs Fe 26.5.1 and Python 3 (standard library). Nothing is written under examples/.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
demo="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$demo/.." && pwd)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
cd "$root/examples/amm-fee" || exit 2
start=$(date +%s)
fe build --color never --out-dir "$out/fe" . > "$out/build.log" 2>&1 || { tail -20 "$out/build.log"; exit 1; }
echo "$(fe --version): built $(ls "$out/fe"/*.runtime.bin | wc -l) contracts of examples/amm-fee in $(( $(date +%s) - start )) s"
python3 -I "$demo/fee_assets.py" --fe-out "$out/fe" "$@"
