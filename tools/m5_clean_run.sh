#!/bin/bash
# Follows the README from nothing, as far as one machine allows: copies exactly the files git would
# publish (.gitignore applied, no .git left behind) to an empty directory, makes that directory's
# parent the new $HOME, installs Fe, Foundry and Node there by the README's commands, and runs the
# README's commands. It shares the operating system's packages (bash, curl, bc, xxd, python3, a C
# linker) with this machine, so it is a clean home, not a clean machine.
#   wsl -e bash tools/m5_clean_run.sh [keep]      (keep: leave the directory in place afterwards)
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
clean="$(mktemp -d "$HOME/.cache/tensile-clean.XXXXXX")"
keep="${1:-}"
[ "$keep" = keep ] || trap 'rm -rf "$clean"' EXIT
log="$root/build/clean_run.log"
mkdir -p "$root/build"
exec > >(tee "$log") 2>&1
started=$(date +%s)
failures=0

step() {
  local name="$1"
  shift
  local start=$(date +%s)
  echo "=== $name"
  if "$@"; then
    echo "ok    $name ($(( $(date +%s) - start )) s)"
  else
    echo "FAIL  $name ($(( $(date +%s) - start )) s)"
    failures=$((failures + 1))
  fi
}

git_dir="$clean/gitdir"
git --git-dir="$git_dir" --work-tree="$root" init -q
git --git-dir="$git_dir" --work-tree="$root" add -A
mkdir -p "$clean/home" "$clean/repo"
git --git-dir="$git_dir" --work-tree="$root" ls-files -z | ( cd "$root" && xargs -0 cp --parents -t "$clean/repo" )
echo "public tree: $(git --git-dir="$git_dir" --work-tree="$root" ls-files | wc -l) files, $(du -sh "$clean/repo" | cut -f1)"
for private in CLAUDE.md HANDOFF.md docs/ideas.md docs/decision.md docs/master-rubric.md docs/research private claude-yolo.bat; do
  [ -e "$clean/repo/$private" ] && { echo "FAIL: $private would be published"; failures=$((failures + 1)); }
done

# A new home, and a PATH with nothing of ours on it.
export HOME="$clean/home"
export PATH="/usr/local/bin:/usr/bin:/bin"
cd "$clean/repo" || exit 2

install_fe() {
  curl -fsSL https://raw.githubusercontent.com/argotorg/fe/master/feup/feup.sh | bash -s -- --version v26.5.1 --no-modify-path \
    && "$HOME/.fe/bin/fe" --version
}
install_foundry() {
  curl -fsSL https://foundry.paradigm.xyz | bash > /dev/null \
    && "$HOME/.foundry/bin/foundryup" --install 1.9.0 > /dev/null \
    && "$HOME/.foundry/bin/forge" --version | head -1
}
install_node() {
  mkdir -p "$HOME/node" \
    && curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz | tar -xJ --strip-components=1 -C "$HOME/node" \
    && "$HOME/node/bin/node" --version
}
step "install Fe 26.5.1" install_fe
step "install Foundry 1.9.0" install_foundry
step "install Node 22" install_node
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$HOME/node/bin:$PATH"

step "fe fmt --check ingots" fe fmt --check ingots
step "fe test" fe test --color never
step "fe build" fe build --color never
native_example() {
  fe build --color never --backend native -O 1 --out-dir out-native --ingot cli . > /dev/null \
    && [ "$(head -c 784 ref/out/inputs.bin | out-native/cli)" = "7" ] \
    && head -c 784 ref/out/inputs.bin | out-native/cli trace | tail -1
}
step "README's native example" native_example
step "tools/m1_check.sh" bash tools/m1_check.sh
step "tools/m2_check.sh" bash tools/m2_check.sh
step "tools/m3_check.sh" bash tools/m3_check.sh

demo() {
  cp demo/dist/tensile-demo.html "$clean/committed.html"
  ( cd demo && npm ci --silent && npm run build --silent ) || return 1
  cmp demo/dist/tensile-demo.html "$clean/committed.html" && echo "the committed page is byte for byte what these sources build"
}
step "demo: npm ci, build, compare" demo
step "demo: npm test" bash -c 'cd demo && npm test --silent'

echo
echo "clean run: $failures step(s) failed, $(( $(date +%s) - started )) s in total"
grep -E '^(ok|FAIL)  ' "$log"
[ "$failures" -eq 0 ]
