#!/bin/bash
# docs/LIBRARY.md section 1, followed to the letter in an empty directory outside the repository:
# a fresh ingot made by `fe new`, depending on a copy of ingots/tensile. The two files the guide
# shows are read from the guide itself (the fenced block after each `<!-- walkthrough: PATH -->`
# line), so that what is tested is what a reader copies.
#
#   1. copy ingots/tensile into the empty directory; `fe new quickstart`;
#   2. write quickstart/fe.toml and quickstart/src/lib.fe from the guide;
#   3. `fe check`, `fe test` (EVM), `fe test --backend native`;
#   4. the guide's deliberate mistake (an offset at the wrong scale) must not compile, with the
#      message the guide quotes;
#   5. dependency forms (section 2): a version-only dependency outside a workspace must be refused;
#      with GIT=1, a git dependency on the public repository at the commit in $GIT_REV must build
#      and pass the same tests (needs the network).
#
# usage: walkthrough.sh [empty-dir]        (default: a new temporary directory)
#        FE=/path/to/fe GIT=1 GIT_REV=<sha> bash walkthrough.sh
export PATH="$HOME/.fe/bin:$PATH"
FE="${FE:-fe}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
guide="$root/docs/LIBRARY.md"
dir="${1:-$(mktemp -d)}"
mkdir -p "$dir"
if [ -n "$(ls -A "$dir")" ]; then
  echo "walkthrough: $dir is not empty" >&2
  exit 2
fi
cd "$dir" || exit 2
echo "$("$FE" --version), in $dir"
failures=0
elapsed() { awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }'; }
# run <label> <expect: ok|fail> <command...>
run() {
  local label="$1" expect="$2"
  shift 2
  local start=$(date +%s.%N)
  "$@" > "$dir.log" 2>&1
  local status=$?
  local result="$(tr '\r' '\n' < "$dir.log" | grep 'test result' | tail -1)"
  if { [ "$expect" = ok ] && [ $status -eq 0 ]; } || { [ "$expect" = fail ] && [ $status -ne 0 ]; }; then
    printf 'ok    %6s s  %s  %s\n' "$(elapsed "$start")" "$label" "$result"
  else
    printf 'FAIL  %6s s  %s (exit %d)\n' "$(elapsed "$start")" "$label" $status
    tr '\r' '\n' < "$dir.log" | grep -v '^\s*$' | head -40
    failures=$((failures + 1))
  fi
}
# file <path>: writes the guide's block for <path>.
file() {
  python3 - "$guide" "$1" > "$1" <<'EOF' || { echo "no block for $1 in the guide" >&2; exit 1; }
import sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
marker = "<!-- walkthrough: %s -->" % sys.argv[2]
start = lines.index(marker)
fence = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("```"))
end = next(i for i in range(fence + 1, len(lines)) if lines[i].startswith("```"))
print("\n".join(lines[fence + 1:end]))
EOF
}
expect_message() { # expect_message <text>: the last command's output must contain it
  if grep -qF -- "$1" "$dir.log"; then
    echo "ok             message: $1"
  else
    echo "FAIL           message lacks: $1"
    failures=$((failures + 1))
  fi
}

run "cp -r ingots/tensile tensile" ok cp -r "$root/ingots/tensile" tensile
run "fe new quickstart" ok "$FE" new quickstart
file quickstart/fe.toml
file quickstart/src/lib.fe
run "fe check quickstart" ok "$FE" check --color never quickstart
run "fe test quickstart" ok "$FE" test --color never quickstart
run "fe test --backend native quickstart" ok "$FE" test --color never --backend native quickstart

# Step 4: the mistake the guide shows.
cp quickstart/src/lib.fe "$dir.lib.fe"
sed -i 's/pub const OFFSET: Q<13> = Q::from_raw(-1024)/pub const OFFSET: Q<7> = Q::from_raw(-16)/' quickstart/src/lib.fe
grep -q 'OFFSET: Q<7>' quickstart/src/lib.fe || { echo "FAIL  the guide's OFFSET line changed"; failures=$((failures + 1)); }
run "fe check, offset at the wrong scale" fail "$FE" check --color never quickstart
expect_message 'expected `Q<13>`, but `Q<7>` is given'
cp "$dir.lib.fe" quickstart/src/lib.fe

# Step 5: dependency forms.
mkdir -p by-version/src
cp quickstart/src/lib.fe by-version/src/lib.fe
sed 's|^tensile = .*|tensile = "0.1.0"|' quickstart/fe.toml > by-version/fe.toml
run "version-only dependency outside a workspace" fail "$FE" check --color never by-version
expect_message "uses name-only lookup outside a workspace"
if [ "$GIT" = 1 ]; then
  rev="${GIT_REV:?set GIT_REV to a commit of github.com/AbrahamPaulJ/tensile}"
  mkdir -p by-git/src
  cp quickstart/src/lib.fe by-git/src/lib.fe
  sed "s|^tensile = .*|tensile = { source = \"https://github.com/AbrahamPaulJ/tensile\", rev = \"$rev\", path = \"ingots/tensile\" }|" \
    quickstart/fe.toml > by-git/fe.toml
  run "git dependency: fe check (first: clones)" ok "$FE" check --color never by-git
  run "git dependency: fe check (again)" ok "$FE" check --color never by-git
  run "git dependency: fe test" ok "$FE" test --color never by-git
fi
rm -f "$dir.log" "$dir.lib.fe"
echo "walkthrough: $failures step(s) failed"
[ $failures -eq 0 ]
