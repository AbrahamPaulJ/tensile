#!/bin/bash
# How large a layer the compile-time overflow proof can cover: `static_assert(worst_case(..))` over
# OUT x 784 int8 weights, every row the same packed constant, on Fe 26.5.1 and on Fe 26.7.0. One
# `fe check` per size and release, in a scratch workspace with a copy of ingots/tensile.
#
#   bash lib-examples/step_limit.sh                 (sizes: 192 200 208 224 232 240 256)
#   OUTS="64 128" FE267=/path/to/fe bash lib-examples/step_limit.sh
#
# Measured 2026-11-24: 26.5.1 passes 232 (181,888 weights) and fails 240; 26.7.0 passes 200
# (156,800) and fails 208, both with `error[3-0026]: const evaluation exceeded the step limit`.
export PATH="$HOME/.fe/bin:$PATH"
FE267="${FE267:-$HOME/fe-267/fe}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/ingots/case/src"
cp -r "$root/ingots/tensile" "$work/ingots/tensile"
cat > "$work/fe.toml" <<'EOF'
[workspace]
name = "step-limit"
version = "0.1.0"
members = [{ path = "ingots/tensile", name = "tensile" }, { path = "ingots/case", name = "case" }]
EOF
cat > "$work/ingots/case/fe.toml" <<'EOF'
[ingot]
name = "case"
version = "0.1.0"

[dependencies]
tensile = true
EOF
cd "$work" || exit 2
probe() { # probe <fe> <release> <out>
  cat > ingots/case/src/lib.fe <<EOF
use tensile::{Bias, I32_MAX, Row, Weights, worst_case}

/// 127, -127 and 1 repeated: every lane in use, both signs.
const WORD: u256 = 0x7f81017f81017f81017f81017f81017f81017f81017f81017f81017f81017f81

/// 784 weights: 24 full words and a last one with its 16 padding bytes zero.
const fn words() -> [u256; 25] {
    var w: [u256; 25] = [WORD; 25]
    w[24] = (WORD >> 128) << 128
    w
}

const ROW: Row<784, 7> = Row::from_words(words())
const W: Weights<$3, 784, 7> = Weights::new([ROW; $3])
const B: Bias<$3, 14> = Bias::from_raw([0; $3])

static_assert(worst_case(weights: W, bias: B) <= I32_MAX)
EOF
  local start=$(date +%s.%N) result
  if "$1" check --color never ingots/case > "$work/log" 2>&1; then
    result=pass
  else
    result="fail: $(grep -o 'error\[[^]]*\]: [a-z ]*' "$work/log" | head -1)"
  fi
  printf '%-7s %3d x 784 = %7d weights  %-62s %5s s\n' "$2" "$3" $(( $3 * 784 )) "$result" \
    "$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')"
}
for out in ${OUTS:-192 200 208 224 232 240 256}; do
  probe fe 26.5.1 "$out"
  [ -x "$FE267" ] && probe "$FE267" 26.7.0 "$out"
done
