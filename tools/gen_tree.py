"""Build the model commitment: the Merkle tree with one leaf per neuron (docs/spec-dispute.md
section 2).

Usage:
    python tools/gen_tree.py [--model ref/out/model.json] [--out build/tree.json]

    leaf(l, j) = keccak256("tensile.neuron.v0" || u256(l) || u256(j) || i256(bias) || row words)

`l` is the layer (1-based), `j` the neuron (0-based), `bias` the int32 bias as a 256-bit two's
complement word, `row words` the packed weight row of docs/spec.md section 7. The leaves are in
order of layer, then neuron. Parents are `keccak256(min(a, b) || max(a, b))`, the sorted-pair
tree OpenZeppelin's `MerkleProof` and Fe's `std::evm::merkle::verify` check; the last node of an
odd level is promoted unchanged. A proof lists the siblings from the leaf's level up to the child
of the root, so it is one node shorter for every level at which the leaf's node was promoted.

The output JSON holds the root, the leaf count and, per leaf, its layer, neuron, hash and proof,
so any tool (the gas script, the Foundry tests) can send a challenge without rebuilding the tree.
Standard library only; the hash is tools/keccak.py.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from keccak import keccak256  # noqa: E402

TAG = b"tensile.neuron.v0"


def pack_words(values):
    """Packed words as bytes: value i is byte i % 32 of word i // 32, zero padding at the end."""
    out = bytearray(v & 0xFF for v in values)
    while len(out) % 32 != 0:
        out.append(0)
    return bytes(out)


def leaf(layer, neuron, bias, row):
    """The leaf hash of one neuron, as 32 bytes."""
    preimage = (TAG + layer.to_bytes(32, "big") + neuron.to_bytes(32, "big")
                + (bias % (1 << 256)).to_bytes(32, "big") + pack_words(row))
    return keccak256(preimage)


def hash_pair(a, b):
    return keccak256(min(a, b) + max(a, b))


def build_tree(leaves):
    """Return (root, proofs): proofs[i] is the list of sibling hashes for leaf i, bottom up."""
    proofs = [[] for _ in leaves]
    level = list(leaves)
    members = [[i] for i in range(len(leaves))]  # the leaves under each node of the level
    while len(level) > 1:
        next_level, next_members = [], []
        for k in range(0, len(level), 2):
            if k + 1 == len(level):
                next_level.append(level[k])
                next_members.append(members[k])
                continue
            for i in members[k]:
                proofs[i].append(level[k + 1])
            for i in members[k + 1]:
                proofs[i].append(level[k])
            next_level.append(hash_pair(level[k], level[k + 1]))
            next_members.append(members[k] + members[k + 1])
        level, members = next_level, next_members
    return level[0], proofs


# ---------------------------------------------------------------- word trees
# The commitment to one packed vector of a claim under the word-tree binding (docs/spec-dispute.md
# section 11): a leaf per 32-byte word, bound to its index, padded with zero nodes to a power of two.

WORD_TAG = b"tensile.word.v0".ljust(32, b"\0")
ZERO = bytes(32)


def word_leaf(index, word):
    """keccak256(WORD_TAG || u256(index) || word), `word` as 32 bytes."""
    return keccak256(WORD_TAG + index.to_bytes(32, "big") + word)


def word_width(count):
    """The number of leaves of the tree of `count` words: the least power of two >= count."""
    width = 1
    while width < count:
        width *= 2
    return width


def word_depth(count):
    return word_width(count).bit_length() - 1


def word_parent(a, b):
    """A node of a word tree: zero over two zero nodes, else the sorted-pair hash."""
    if a == ZERO and b == ZERO:
        return ZERO
    return hash_pair(a, b)


def word_levels(words):
    """Every level of the tree of `words` (a list of 32-byte values), leaves first, root last."""
    level = [word_leaf(i, w) for i, w in enumerate(words)]
    level += [ZERO] * (word_width(len(words)) - len(words))
    levels = [level]
    while len(level) > 1:
        level = [word_parent(level[k], level[k + 1]) for k in range(0, len(level), 2)]
        levels.append(level)
    return levels


def word_root(words):
    return word_levels(words)[-1][0]


def word_proof(words, index):
    """The siblings of word `index` from its leaf up to the child of the root: word_depth nodes."""
    proof = []
    for level in word_levels(words)[:-1]:
        proof.append(level[index ^ 1])
        index //= 2
    return proof


def word_verify(root, index, word, proof):
    node = word_leaf(index, word)
    for sibling in proof:
        node = hash_pair(node, sibling)
    return node == root


def commitment(model):
    """The tree of a model file: {"root", "leaf_count", "leaves": [{layer, neuron, leaf, proof}]}."""
    positions, hashes = [], []
    for index, layer in enumerate(model["layers"], start=1):
        for j, (row, bias) in enumerate(zip(layer["weights"], layer["bias"])):
            positions.append((index, j))
            hashes.append(leaf(index, j, bias, row))
    root, proofs = build_tree(hashes)
    return {
        "name": model["name"],
        "root": "0x" + root.hex(),
        "leaf_count": len(hashes),
        "leaves": [
            {"layer": l, "neuron": j, "leaf": "0x" + h.hex(), "proof": ["0x" + p.hex() for p in proof]}
            for (l, j), h, proof in zip(positions, hashes, proofs)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.path.join(ROOT, "ref", "out", "model.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "build", "tree.json"))
    args = parser.parse_args()
    with open(args.model, "r", encoding="utf-8") as f:
        model = json.load(f)
    tree = commitment(model)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(tree, f, indent=1)
        f.write("\n")
    print("root %s, %d leaves, proofs of %d to %d nodes, wrote %s" % (
        tree["root"], tree["leaf_count"], min(len(x["proof"]) for x in tree["leaves"]),
        max(len(x["proof"]) for x in tree["leaves"]), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
