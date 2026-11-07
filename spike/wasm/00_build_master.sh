#!/usr/bin/env bash
# Build Fe from a pinned master commit (for the WebAssembly target, argotorg/fe PR 1904), in WSL.
# Installs rustup and Node into the home folder (no root). Re-runnable: skips finished steps.
set -euo pipefail
PIN="${PIN:-89d07b9dbf7ea73d0ca6b9a5b4f14de7899d1908}"   # master, 2026-11-07, CI and WebAssembly workflows green
SRC="$HOME/fe-master"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if ! command -v cargo >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  log "installing rustup"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
. "$HOME/.cargo/env"

if [ ! -x "$HOME/.local/node/bin/node" ]; then
  log "installing node 22"
  mkdir -p "$HOME/.local/node"
  curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | tar -xJ -C "$HOME/.local/node" --strip-components=1
fi
export PATH="$HOME/.local/node/bin:$PATH"

if [ ! -d "$SRC/.git" ]; then
  log "cloning fe"
  git clone -q https://github.com/argotorg/fe.git "$SRC"
fi
cd "$SRC"
git fetch -q origin master
git checkout -q "$PIN"
git submodule update --init --recursive -q || true
log "building fe at $(git rev-parse --short HEAD)"
start=$(date +%s)
cargo build --release -p fe 2>&1 | tail -5
log "build took $(( $(date +%s) - start )) s"
"$SRC/target/release/fe" --version
node --version
