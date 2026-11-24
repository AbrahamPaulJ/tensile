#!/bin/bash
# Makes (or refreshes) an Fe workspace for one tree ensemble (docs/spec-trees.md). Its ingots depend
# on this repository's `tensile`, `trees` and `dispute` ingots by relative path:
#
#   ingots/model     the trees as constants, the commitment, reference tests      tools/gen_trees.py
#   ingots/arbiter   the dispute contract and its contract tests                  tools/gen_trees.py
#   ingots/ensemble  the whole forest on-chain: tree holders and `Ensemble`       tools/gen_trees.py
#   ingots/cli       a native classifier (`cli`, `cli trace`) for differential tests against the
#                    reference: its entry point is generated, input.fe and output.fe are copies of
#                    ingots/cli
#   build/forest.json  the commitment: root, every tree's words, leaf and proof   tools/gen_trees.py
#
# usage: gen_trees_workspace.sh <model.json> <vectors-dir> <workspace-dir> [samples]
#   <vectors-dir> holds inputs.bin and trace.jsonl for the model (tools/trees_vectors.py); the
#   first [samples] (default 3, at least 2) become Fe unit tests of the model ingot.
#
# The files named above and the fe.toml files are overwritten. Other ingots found under
# <workspace-dir>/ingots (your own contracts) are kept and listed as workspace members.
export PATH="$HOME/.fe/bin:$HOME/.foundry/bin:$PATH"
set -e
if [ $# -lt 3 ]; then
  echo "usage: gen_trees_workspace.sh <model.json> <vectors-dir> <workspace-dir> [samples]" >&2
  exit 2
fi
root="$(cd "$(dirname "$0")/.." && pwd)"
model="$1"
vectors="$2"
dir="$3"
samples="${4:-3}"
for ingot_dir in model arbiter ensemble cli; do
  mkdir -p "$dir/ingots/$ingot_dir/src"
done
mkdir -p "$dir/build"
dir="$(cd "$dir" && pwd)"
name="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$model")"
# The path from an ingot directory of the workspace to this repository's ingots.
up="$(python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]).replace(os.sep, "/"))' "$root/ingots" "$dir/ingots/model")"

ingot() {
  # ingot <name> <dependency lines...>: writes ingots/<name>/fe.toml
  local ingot_name="$1"
  shift
  {
    printf '[ingot]\nname = "%s"\nversion = "0.1.0"\n\n[dependencies]\n' "$ingot_name"
    printf '%s\n' "$@"
  } > "$dir/ingots/$ingot_name/fe.toml"
}
tensile="tensile = { path = \"$up/tensile\" }"
trees="trees = { path = \"$up/trees\" }"
dispute="dispute = { path = \"$up/dispute\" }"
ingot model "$tensile" "$trees"
ingot arbiter "$tensile" "$trees" "$dispute" "model = true"
ingot ensemble "$tensile" "$trees" "model = true"
ingot cli "$tensile" "$trees" "model = true"

# Every ingot directory is a workspace member: the generated ones first, then any others.
members=""
for member in model arbiter ensemble cli $(cd "$dir/ingots" && ls); do
  [ -f "$dir/ingots/$member/fe.toml" ] || continue
  case ", $members" in *"name = \"$member\""*) continue ;; esac
  members="$members{ path = \"ingots/$member\", name = \"$member\" }, "
done
cat > "$dir/fe.toml" <<EOF
[workspace]
name = "$name"
version = "0.1.0"
members = [${members%, }]
EOF

python3 "$root/tools/gen_trees.py" --model "$model" --vectors "$vectors" --samples "$samples" --out "$dir"
cp "$root/ingots/cli/src/input.fe" "$root/ingots/cli/src/output.fe" "$dir/ingots/cli/src/"
fe fmt "$dir/ingots" > /dev/null
echo "workspace $dir: ingots $(cd "$dir/ingots" && ls | tr '\n' ' ')for $name"
