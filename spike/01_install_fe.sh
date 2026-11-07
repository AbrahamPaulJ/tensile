#!/usr/bin/env bash
# Install Fe v26.5.1 into ~/.fe (no sudo), then prove it works on a fresh project.
set -eu
bash /tmp/feup.sh --version v26.5.1
export PATH="$HOME/.fe/bin:$PATH"
echo "== version"; fe --version
echo "== help"; fe --help
rm -rf /tmp/fe_hello && mkdir -p /tmp/fe_hello && cd /tmp/fe_hello
echo "== fe new"; fe new hello 2>&1 | tail -5
cd hello && find . -type f | sort && echo "== fe.toml" && cat fe.toml
echo "== src"; for f in src/*.fe; do echo "--- $f"; cat "$f"; done
echo "== fe test"; time fe test 2>&1 | tail -15
echo "== fe build"; fe build 2>&1 | tail -5; find . -path ./src -prune -o -type f -newer fe.toml -print | sort | head -20
