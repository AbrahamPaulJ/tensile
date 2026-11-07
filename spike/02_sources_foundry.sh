#!/usr/bin/env bash
# Fe source at the installed tag (for reading the standard library) and Foundry (anvil, forge, cast).
set -eu
if [ ! -d "$HOME/fe-src" ]; then
  git clone --quiet --depth 1 --branch v26.5.1 https://github.com/argotorg/fe "$HOME/fe-src"
fi
echo "== fe-src"; git -C "$HOME/fe-src" log --oneline -1; ls "$HOME/fe-src/ingots"
echo "== std modules"; find "$HOME/fe-src/ingots" -name '*.fe' | sed "s#$HOME/fe-src/ingots/##" | sort | tr '\n' ' '
echo
if [ ! -x "$HOME/.foundry/bin/forge" ]; then
  curl -fsSL https://foundry.paradigm.xyz -o /tmp/foundry_install.sh
  bash /tmp/foundry_install.sh >/dev/null 2>&1 || true
  "$HOME/.foundry/bin/foundryup" 2>&1 | tail -3
fi
echo "== foundry"; "$HOME/.foundry/bin/forge" --version | head -1; "$HOME/.foundry/bin/anvil" --version | head -1
