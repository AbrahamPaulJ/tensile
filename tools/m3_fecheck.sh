#!/bin/bash
# Development: `fe check` of one ingot, first N lines of diagnostics (default 80).
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
fe check --color never "ingots/$1" 2>&1 | head -${2:-80}
echo "exit ${PIPESTATUS[0]}"
