#!/usr/bin/env bash
# Report what the WSL environment has before installing Fe.
set -u
echo "== os"; . /etc/os-release; echo "$PRETTY_NAME"; uname -m
echo "== tools"
for t in curl git cc gcc python3 pip3 cargo forge anvil cast solc node fe; do
  if command -v "$t" >/dev/null 2>&1; then echo "$t: $(command -v "$t")"; else echo "$t: MISSING"; fi
done
echo "== python"; python3 --version 2>&1; python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | tail -1
echo "== disk"; df -h "$HOME" | tail -1
echo "== fe dir"; ls -la "$HOME/.fe" 2>/dev/null || echo "no ~/.fe"
echo "== feup script head"
curl -fsSL https://raw.githubusercontent.com/argotorg/fe/master/feup/feup.sh -o /tmp/feup.sh && wc -l /tmp/feup.sh && grep -nE "curl|wget|sudo|rm -rf|PATH|install_dir|FE_HOME|chmod|profile|bashrc" /tmp/feup.sh | head -60
