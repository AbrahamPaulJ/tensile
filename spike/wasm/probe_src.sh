#!/bin/bash
cd ~/fe-master
echo "--- inline mod files"; grep -rlE '^\s*(pub )?mod [a-z_]+ \{' --include=*.fe crates ingots library 2>/dev/null | head -5
grep -rhE '^\s*(pub )?mod [a-z_]+ \{' --include=*.fe crates 2>/dev/null | head -3
echo "--- examples/wasm"; ls examples/wasm
echo "--- build.rs wasm limits"; grep -n -iE 'standalone|ingot' crates/fe/src/build.rs | grep -i -E 'wasm|standalone' | head -10
