#!/bin/bash
cd ~/fe-src/ingots
f=$(grep -rl "pub struct Wei" . | head -1); echo "== $f"; grep -n "pub struct Wei\|impl.*for Wei\|pub fn\|pub const fn" "$f" | head -30
