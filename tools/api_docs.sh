#!/bin/bash
# The API documentation of the kernel ingot, as a static site: `fe doc` on ingots/tensile, then the
# items that are not part of the public API (private constants, test helpers) are dropped from the
# index, because the viewer shows every item it is given and has no visibility filter.
#
#   bash tools/api_docs.sh <out-dir>          (FE=/path/to/fe to use another binary)
#
# The site is four files and docs.json (about 2 MB, most of it the viewer's script). It uses relative
# links and hash routing, so it works from any sub-path: the Pages workflow puts it under /api/.
export PATH="$HOME/.fe/bin:$PATH"
FE="${FE:-fe}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$1"
if [ -z "$out" ]; then
  echo "usage: api_docs.sh <out-dir>" >&2
  exit 2
fi
mkdir -p "$out"
out="$(cd "$out" && pwd)"
# fe doc writes nothing into the ingot, but run it on a copy all the same.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp -r "$root/ingots/tensile" "$work/tensile"
(cd "$work/tensile" && "$FE" doc --color never -o "$out" . static) || exit 1
python3 - "$out/docs.json" <<'EOF'
import json, sys

path = sys.argv[1]
docs = json.load(open(path))
index = docs["index"]
before = len(index["items"])
private = {item["path"] for item in index["items"] if item.get("visibility") == "private"}
index["items"] = [item for item in index["items"] if item["path"] not in private]


def prune(module):
    module["items"] = [item for item in module.get("items", []) if item["path"] not in private]
    for child in module.get("children", []):
        prune(child)


for module in index["modules"]:
    prune(module)
json.dump(docs, open(path, "w"), separators=(",", ":"))
print(f"api_docs: {len(index['items'])} of {before} items kept, {len(private)} private ones dropped")
EOF
