"""The reference for tree ensembles (docs/spec-trees.md): the oracle every Fe build must match.

Pure Python, standard library only, so CI and anyone without NumPy can run it. It holds:

  - the model file: load_model, check_model, model_to_json (docs/spec-trees.md section 5);
  - the arithmetic: leaf_index, run (per-tree leaf, per-tree value, class scores, class);
  - the packing: tree_words, outputs_words (16-bit lanes, docs/spec-trees.md section 6);
  - the commitment: tree_leaf, header_leaf, commitment (root and proofs, section 7).

    python tools/trees_ref.py MODEL.json INPUTS.bin    prints the class of every sample

The Merkle tree is tools/gen_tree.py's (sorted pairs, the odd last node promoted), over the leaves
of this file, and the hash is tools/keccak.py.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_tree import build_tree  # noqa: E402
from keccak import keccak256  # noqa: E402

FORMAT = "tensile.forest.v0"
TREE_TAG = b"tensile.tree.v0"
HEADER_TAG = b"tensile.forest.v0"
I16_MIN, I16_MAX = -32768, 32767
BASE_LIMIT = (1 << 30) - 1
INT32_MAX = (1 << 31) - 1
MAX_TREES = 32768  # trees * 2^15 <= 2^30: the bound of docs/spec-trees.md section 4


class ModelError(ValueError):
    pass


def node_count(depth):
    return (1 << depth) - 1


def leaf_count(depth):
    return 1 << depth


def lane_words(n):
    return (n + 15) // 16


def node_words(depth):
    return lane_words(node_count(depth))


def tree_word_count(depth):
    return node_words(depth) + lane_words(leaf_count(depth))


def target(t, classes):
    """The class score tree t adds to: class 1 for two classes, t % classes otherwise."""
    return 1 if classes == 2 else t % classes


def check_model(model):
    """Raises ModelError unless `model` is a well-formed forest of docs/spec-trees.md section 5."""
    if model.get("format") != FORMAT:
        raise ModelError("format must be %r" % FORMAT)
    n_in, depth, classes = model["in"], model["depth"], model["classes"]
    trees, base, lf = model["trees"], model["base"], model["leaf_frac_bits"]
    if not 1 <= n_in <= 256:
        raise ModelError("in must be 1..256 (a feature index is 8 bits)")
    if not 1 <= depth <= 8:
        raise ModelError("depth must be 1..8")
    if not 2 <= classes <= 255:
        raise ModelError("classes must be 2..255 (the class is a uint8)")
    if not 1 <= len(trees) <= MAX_TREES:
        raise ModelError("1 to %d trees" % MAX_TREES)
    if not 0 <= lf <= 30:
        raise ModelError("leaf_frac_bits must be 0..30")
    if model.get("input_frac_bits") != 7:
        raise ModelError("input_frac_bits must be 7 (the activations of docs/spec.md)")
    if len(base) != classes or any(not -BASE_LIMIT <= b <= BASE_LIMIT for b in base):
        raise ModelError("base: one int per class, each within +-(2^30 - 1)")
    for t, tree in enumerate(trees):
        if len(tree["feature"]) != node_count(depth) or len(tree["threshold"]) != node_count(depth):
            raise ModelError("tree %d: %d nodes expected" % (t, node_count(depth)))
        if len(tree["leaf"]) != leaf_count(depth):
            raise ModelError("tree %d: %d leaves expected" % (t, leaf_count(depth)))
        if any(not 0 <= f < n_in for f in tree["feature"]):
            raise ModelError("tree %d: a feature index is not below %d" % (t, n_in))
        if any(not 0 <= v <= 255 for v in tree["threshold"]):
            raise ModelError("tree %d: a threshold is not 0..255" % t)
        if any(not I16_MIN <= v <= I16_MAX for v in tree["leaf"]):
            raise ModelError("tree %d: a leaf is outside the int16 range" % t)
    if "worst_case_abs_score" in model and model["worst_case_abs_score"] != worst_case(model):
        raise ModelError("worst_case_abs_score is %d, the trees give %d"
                         % (model["worst_case_abs_score"], worst_case(model)))


def load_model(path):
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    check_model(model)
    return model


def worst_case(model):
    """The largest absolute class score over every input: per class, |base| plus the largest
    absolute leaf of every tree that adds to it; the largest over the classes."""
    classes = model["classes"]
    bounds = [abs(b) for b in model["base"]]
    for t, tree in enumerate(model["trees"]):
        bounds[target(t, classes)] += max(abs(v) for v in tree["leaf"])
    return max(bounds)


def leaf_index(tree, x, depth):
    """The leaf the walk on x ends at: left when x[feature] <= threshold (section 3)."""
    i = 0
    feature, threshold = tree["feature"], tree["threshold"]
    for _ in range(depth):
        i = 2 * i + 1 if x[feature[i]] <= threshold[i] else 2 * i + 2
    return i - node_count(depth)


def run(model, x):
    """One sample: (leaf indices, leaf values, class scores, class). The class is the highest
    score, the lowest index winning a tie."""
    depth, classes = model["depth"], model["classes"]
    if len(x) != model["in"] or any(not 0 <= v <= 127 for v in x):
        raise ValueError("an input is %d bytes in 0..127" % model["in"])
    indices, values = [], []
    scores = list(model["base"])
    for t, tree in enumerate(model["trees"]):
        k = leaf_index(tree, x, depth)
        indices.append(k)
        values.append(tree["leaf"][k])
        scores[target(t, classes)] += tree["leaf"][k]
    best = 0
    for c in range(1, classes):
        if scores[c] > scores[best]:
            best = c
    return indices, values, scores, best


def class_of(model, values):
    """The class a claim's per-tree values give: what the class challenge recomputes."""
    scores = list(model["base"])
    for t, v in enumerate(values):
        scores[target(t, model["classes"])] += v
    best = 0
    for c in range(1, model["classes"]):
        if scores[c] > scores[best]:
            best = c
    return scores, best


def pack_lanes(lanes, words):
    """16-bit lanes (unsigned 0..65535) into `words` 256-bit integers, lane i at position i % 16
    of word i // 16 from the most significant end; unused lanes zero."""
    out = [0] * words
    for i, lane in enumerate(lanes):
        out[i // 16] |= (lane & 0xFFFF) << ((15 - i % 16) * 16)
    return out


def tree_words(tree, depth):
    """A tree as packed words: the node lanes (feature << 8 | threshold), then the leaf lanes."""
    nodes = pack_lanes([(f << 8) | t for f, t in zip(tree["feature"], tree["threshold"])],
                       node_words(depth))
    leaves = pack_lanes(tree["leaf"], lane_words(leaf_count(depth)))
    return nodes + leaves


def outputs_words(values):
    """The per-tree values of a claim as packed words."""
    return pack_lanes(values, lane_words(len(values)))


def word_bytes(words):
    return b"".join(w.to_bytes(32, "big") for w in words)


def tree_leaf(t, words):
    """keccak256("tensile.tree.v0" || u256(t) || tree words), as 32 bytes."""
    return keccak256(TREE_TAG + t.to_bytes(32, "big") + word_bytes(words))


def header_leaf(model):
    """keccak256("tensile.forest.v0" || u256(in) || u256(depth) || u256(classes) ||
    u256(leaf_frac_bits) || u256(trees) || i256(base[0]) || ... ), as 32 bytes."""
    fields = [model["in"], model["depth"], model["classes"], model["leaf_frac_bits"],
              len(model["trees"])]
    data = b"".join(v.to_bytes(32, "big") for v in fields)
    data += b"".join((b % (1 << 256)).to_bytes(32, "big") for b in model["base"])
    return keccak256(HEADER_TAG + data)


def commitment(model):
    """The model commitment: one leaf per tree in tree order, then the header leaf.
    {"name", "root", "leaf_count", "header": {"leaf", "proof"}, "trees": [{"tree", "words",
    "leaf", "proof"}]}; words and hashes as 0x-hex."""
    depth = model["depth"]
    words = [tree_words(tree, depth) for tree in model["trees"]]
    hashes = [tree_leaf(t, w) for t, w in enumerate(words)] + [header_leaf(model)]
    root, proofs = build_tree(hashes)

    def hx(b):
        return "0x" + b.hex()

    return {
        "name": model["name"],
        "root": hx(root),
        "leaf_count": len(hashes),
        "header": {"leaf": hx(hashes[-1]), "proof": [hx(p) for p in proofs[-1]]},
        "trees": [{"tree": t, "words": ["0x%064x" % v for v in w], "leaf": hx(hashes[t]),
                   "proof": [hx(p) for p in proofs[t]]} for t, w in enumerate(words)],
    }


def model_to_json(model):
    """The model file, deterministic, one tree per line."""
    head = {k: v for k, v in model.items() if k != "trees"}
    lines = ["{"]
    keys = list(head.keys())
    for k in keys:
        lines.append(" %s: %s," % (json.dumps(k), json.dumps(head[k], sort_keys=True)))
    lines.append(' "trees": [')
    for t, tree in enumerate(model["trees"]):
        sep = "," if t + 1 < len(model["trees"]) else ""
        lines.append("  " + json.dumps({"feature": tree["feature"], "threshold": tree["threshold"],
                                        "leaf": tree["leaf"]}) + sep)
    lines.append(" ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model = load_model(sys.argv[1])
    data = open(sys.argv[2], "rb").read()
    n = model["in"]
    for s in range(len(data) // n):
        print(run(model, data[s * n:(s + 1) * n])[3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
