#!/bin/bash
# Development: raw `fe test` output of one ingot on one backend (default sonatina), unfiltered.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
fe test --color never --backend "${2:-sonatina}" "ingots/$1" 2>&1 | tr '\r' '\n' | grep -v '^PASS' | head -${3:-80}
