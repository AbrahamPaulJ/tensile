#!/bin/bash
# The library examples (lib-examples/) on the Fe 26.7.0 release, the companion of
# tools/check_fe267.sh (which covers the four shipped ingots and is unchanged). It runs
# lib-examples/check.sh with the 26.7.0 binary: `fe fmt --check`, `fe check`, `fe test` on the EVM
# and native backends, the compile-fail cases and `fe build`. No anvil: the recorded gas is 26.5.1's.
#
#   wsl -e bash /mnt/c/.../fe-challenge/tools/check_fe267_lib.sh
#   FE267=/path/to/fe bash tools/check_fe267_lib.sh        (default: ~/fe-267/fe)
# Exits non-zero if a step fails, 3 if the binary is absent or is not 26.7.0.
FE267="${FE267:-$HOME/fe-267/fe}"
EXPECTED="26.7.0"
cd "$(dirname "$0")/.." || exit 2
if [ ! -x "$FE267" ]; then
  echo "check_fe267_lib: no Fe $EXPECTED binary at $FE267 (set FE267 to one)" >&2
  exit 3
fi
version="$("$FE267" --version)"
case "$version" in
  *"$EXPECTED"*) ;;
  *) echo "check_fe267_lib: $FE267 is '$version', not Fe $EXPECTED" >&2; exit 3 ;;
esac
echo "$version, sha256 $(sha256sum "$FE267" | cut -d' ' -f1)"
FE="$FE267" bash lib-examples/check.sh
