#!/bin/bash
# The same sources on the Fe 26.7.0 release: `fe check` and `fe test` of the four ingots that build
# on both releases (the kernel, the generated model, the dispute rule, the contracts). The entry is
# pinned to Fe 26.5.1 and every published figure comes from that release; this check shows that the
# source also compiles and passes its tests on 26.7.0.
#
# `ingots/cli` is left out: it is the native classifier, and the native I/O API it uses was replaced
# after 26.5.1. The workspace `fe.toml` lists it, so each ingot is given to `fe` by its own path
# instead of running `fe test` on the workspace.
#
#   wsl -e bash /mnt/c/.../fe-challenge/tools/check_fe267.sh
#   FE267=/path/to/fe bash tools/check_fe267.sh        (default: ~/fe-267/fe)
# Prints one line per step with its wall time and exits non-zero if a step fails. Exits 3 if the
# binary is absent or is not 26.7.0.
FE267="${FE267:-$HOME/fe-267/fe}"
EXPECTED="26.7.0"
cd "$(dirname "$0")/.." || exit 2
if [ ! -x "$FE267" ]; then
  echo "check_fe267: no Fe $EXPECTED binary at $FE267 (set FE267 to one)" >&2
  exit 3
fi
version="$("$FE267" --version)"
case "$version" in
  *"$EXPECTED"*) ;;
  *) echo "check_fe267: $FE267 is '$version', not Fe $EXPECTED" >&2; exit 3 ;;
esac
echo "$version, sha256 $(sha256sum "$FE267" | cut -d' ' -f1), $(date -u +%Y-%m-%dT%H:%M:%SZ)"

log="$(mktemp)"
trap 'rm -f "$log"' EXIT
failures=0
total_start=$(date +%s.%N)
elapsed() { awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }'; }

for ingot in tensile model dispute onchain; do
  start=$(date +%s.%N)
  if "$FE267" check --color never "ingots/$ingot" > "$log" 2>&1; then
    printf 'ok    %7s s  fe check %s\n' "$(elapsed "$start")" "$ingot"
  else
    printf 'FAIL  %7s s  fe check %s\n' "$(elapsed "$start")" "$ingot"
    head -60 "$log"
    failures=$((failures + 1))
  fi
  start=$(date +%s.%N)
  "$FE267" test --color never "ingots/$ingot" 2>&1 | tr '\r' '\n' > "$log"
  status=${PIPESTATUS[0]}
  result="$(grep 'test result' "$log" | tail -1)"
  # A run that reports no tests is a failure too: it would pass without testing anything.
  if [ $status -eq 0 ] && [ -n "$result" ]; then
    printf 'ok    %7s s  fe test %-8s %s\n' "$(elapsed "$start")" "$ingot" "$result"
  else
    printf 'FAIL  %7s s  fe test %-8s %s\n' "$(elapsed "$start")" "$ingot" "$result"
    grep -v '^\s*$' "$log" | grep -v '^PASS' | head -60
    failures=$((failures + 1))
  fi
done
printf 'check_fe267: total %s s, %d step(s) failed\n' "$(elapsed "$total_start")" "$failures"
[ $failures -eq 0 ]
