#!/bin/bash
# Development: format every ingot, then show what `fe fmt --check` still complains about.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
cd "$(dirname "$0")/.."
fe fmt ingots
fe fmt --check ingots && echo "fe fmt --check: clean"
