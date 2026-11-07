"""Write the fixture file of the Foundry tests of the Arbiter (evm/test/Arbiter*.t.sol).

Usage:
    python3 tools/gen_arbiter_fixtures.py [--ref ref/out] [--tree build/tree.json]
                                          [--out evm/test/fixtures/arbiter.bin] [--samples 16]

The file is a flat sequence of 32-byte big-endian words, read in Solidity by
evm/test/ArbiterBase.sol (the offsets below are repeated there as constants):

    word 0                 the model root (build/tree.json)
    words 1 .. 800         layer 1 rows, 32 rows of 25 packed words (docs/spec.md section 7)
    words 801 .. 832       layer 1 biases, int32 sign-extended to 256 bits
    words 833 .. 842       layer 2 rows, 10 rows of 1 packed word
    words 843 .. 852       layer 2 biases
    words 853 .. 1146      42 proofs (layer 1 neuron j at index j, layer 2 neuron j at 32 + j),
                           7 words each: the length, then up to 6 siblings, zero filled
    word 1147              the number of samples S
    words 1148 ..          S honest claims of 38 words each: the sample's index in ref/out, x0
                           (25 packed words), hidden (1 packed word), logits (10 words, int32
                           sign-extended), class

The samples are the first traced sample of each class 0..9, then the lowest unused indices, so
every class is claimed at least once.

Before writing, the script checks its own inputs: every leaf rebuilt from the model's row and bias
equals build/tree.json, every proof climbs to the root, and every sample's hidden vector, logits
and class equal an integer forward pass of the model in plain Python (docs/spec.md section 3).
Deterministic: the same inputs give the same bytes. Standard library only.
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
PROOF_SLOTS = 6
SAMPLE_WORDS = 38


def word(value):
    """A 256-bit two's complement word."""
    return (value % (1 << 256)).to_bytes(32, "big")


def packed(values):
    out = bytearray(v & 0xFF for v in values)
    while len(out) % 32 != 0:
        out.append(0)
    return bytes(out)


def leaf(layer, neuron, bias, row):
    return keccak256(TAG + word(layer) + word(neuron) + word(bias) + packed(row))


def climb(node, proof):
    for sibling in proof:
        a, b = sorted([node, sibling])
        node = keccak256(a + b)
    return node


def forward(model, x0):
    """The integer network of docs/spec.md section 3."""
    l1, l2 = model["layers"]
    hidden = []
    for j in range(l1["out"]):
        acc = l1["bias"][j] + sum(w * x for w, x in zip(l1["weights"][j], x0))
        hidden.append(min(max(acc >> l1["shift"], 0), 127))
    logits = [l2["bias"][j] + sum(w * h for w, h in zip(l2["weights"][j], hidden)) for j in range(l2["out"])]
    cls = max(range(len(logits)), key=lambda k: (logits[k], -k))
    return hidden, logits, cls


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default=os.path.join(ROOT, "ref", "out"))
    parser.add_argument("--tree", default=os.path.join(ROOT, "build", "tree.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evm", "test", "fixtures", "arbiter.bin"))
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()

    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        model = json.load(f)
    with open(args.tree, "r", encoding="utf-8") as f:
        tree = json.load(f)
    l1, l2 = model["layers"]
    assert (l1["in"], l1["out"], l2["in"], l2["out"]) == (784, 32, 32, 10), "unexpected model shape"
    root = bytes.fromhex(tree["root"][2:])

    out = bytearray(root)
    for j in range(32):
        out += packed(l1["weights"][j])
    for j in range(32):
        out += word(l1["bias"][j])
    for j in range(10):
        out += packed(l2["weights"][j])
    for j in range(10):
        out += word(l2["bias"][j])

    entries = {(e["layer"], e["neuron"]): e for e in tree["leaves"]}
    assert len(entries) == 42, "the tree does not have 42 leaves"
    for layer, count in ((1, 32), (2, 10)):
        spec = model["layers"][layer - 1]
        for j in range(count):
            e = entries[(layer, j)]
            mine = leaf(layer, j, spec["bias"][j], spec["weights"][j])
            assert "0x" + mine.hex() == e["leaf"], "leaf %d.%d differs from build/tree.json" % (layer, j)
            proof = [bytes.fromhex(p[2:]) for p in e["proof"]]
            assert climb(mine, proof) == root, "proof %d.%d does not reach the root" % (layer, j)
            assert len(proof) <= PROOF_SLOTS
            out += word(len(proof))
            for p in proof:
                out += p
            out += bytes(32 * (PROOF_SLOTS - len(proof)))

    out += word(args.samples)
    with open(os.path.join(args.ref, "inputs.bin"), "rb") as f:
        inputs = f.read()
    with open(os.path.join(args.ref, "trace.jsonl"), "r", encoding="utf-8") as f:
        every = [json.loads(line) for line in f.readlines()]
    chosen = []
    for c in range(10):
        chosen.append(next(t["i"] for t in every if t["class"] == c))
    chosen += [i for i in range(len(every)) if i not in chosen][: args.samples - len(chosen)]
    chosen = chosen[: args.samples]
    traces = [every[i] for i in chosen]
    for n, t in zip(chosen, traces):
        assert t["i"] == n
        x0 = list(inputs[n * 784 : (n + 1) * 784])
        hidden, logits, cls = forward(model, x0)
        assert hidden == t["hidden"][0] and logits == t["logits"] and cls == t["class"], "sample %d" % n
        out += word(n) + packed(x0) + packed(hidden)
        for z in logits:
            out += word(z)
        out += word(cls)

    assert len(out) == 32 * (1148 + SAMPLE_WORDS * args.samples)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(bytes(out))
    print("wrote %s: %d words, samples %s, root %s" % (
        os.path.relpath(args.out, ROOT), len(out) // 32, chosen, tree["root"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
