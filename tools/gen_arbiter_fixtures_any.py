"""Write the fixture file of the Foundry tests of a GENERATED arbiter (evm/generated/*.t.sol), for a
dense model with any number of layers. The general form of tools/gen_arbiter_fixtures.py, which is
fixed to the shipped 784-32-10 model and is left as it is.

Usage:
    python3 tools/gen_arbiter_fixtures_any.py --ref DIR --tree FILE --out FILE
                                              [--inputs FILE] [--samples 16]

DIR holds model.json and trace.jsonl (and inputs.bin, unless --inputs names it); FILE is the tree
tools/gen_tree.py wrote for the same model.json.

The file is a flat sequence of 32-byte big-endian words, read in Solidity by
evm/generated/GenBase.sol (library `Gx`). A header says where everything is, so the Solidity side
holds no sizes of its own:

    word 0                 the tag "tensile.generated.fixtures.v1" (left-aligned)
    word 1                 the model root
    word 2                 L, the number of layers
    word 3                 P, the number of sibling slots of a proof
    word 4                 S, the number of samples
    word 5                 the word index of the first sample
    word 6                 the number of words of one sample
    words 7 + 6 (l - 1) .. for layer l = 1..L: inputs, outputs, shift (0 for the last layer, which
                           has none), then the word indices of its rows, its biases and its proofs

    rows of layer l        `outputs` rows of ceil(inputs / 32) packed words (docs/spec.md section 7)
    biases of layer l      `outputs` words, int32 sign-extended to 256 bits
    proofs of layer l      `outputs` proofs of 1 + P words: the length, then the siblings, zero filled
    a sample               its index in the reference vectors, x0 (packed words), every hidden
                           vector in layer order (packed words), the logits (one word each, int32
                           sign-extended), the class

The samples are the first traced sample of each class, then the lowest unused indices, so every
class is claimed at least once (a class the trace never predicts is reported and skipped).

This script knows nothing of the Fe code or of tools/gen_arbiter.py: it reads the model file, the
reference trace and the tree. Before writing, it checks its own inputs: every leaf rebuilt from the
model's row and bias equals the tree's, every proof climbs to the root, and every sample's hidden
vectors, logits and class equal an integer forward pass of the model in plain Python (docs/spec.md
section 3). Deterministic: the same inputs give the same bytes. Standard library only.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from keccak import keccak256  # noqa: E402

TAG = b"tensile.neuron.v0"
FILE_TAG = b"tensile.generated.fixtures.v1"
HEADER = 7
PER_LAYER = 6


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
    """The integer network of docs/spec.md section 3: every hidden vector, the logits, the class."""
    layers = model["layers"]
    hidden = []
    x = x0
    for spec in layers[:-1]:
        x = [
            min(max((spec["bias"][j] + sum(w * v for w, v in zip(spec["weights"][j], x))) >> spec["shift"], 0), 127)
            for j in range(spec["out"])
        ]
        hidden.append(x)
    last = layers[-1]
    logits = [last["bias"][j] + sum(w * v for w, v in zip(last["weights"][j], x)) for j in range(last["out"])]
    cls = max(range(len(logits)), key=lambda k: (logits[k], -k))
    return hidden, logits, cls


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs")
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()

    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        model = json.load(f)
    with open(args.tree, "r", encoding="utf-8") as f:
        tree = json.load(f)
    layers = model["layers"]
    count = len(layers)
    for a, b in zip(layers, layers[1:]):
        assert a["out"] == b["in"], "the layer sizes do not chain"
    root = bytes.fromhex(tree["root"][2:])
    entries = {(e["layer"], e["neuron"]): e for e in tree["leaves"]}
    assert len(entries) == sum(spec["out"] for spec in layers), "the tree has another number of leaves"
    slots = max(len(e["proof"]) for e in entries.values())

    # The data sections, each remembered with its word index.
    body = bytearray()
    base = HEADER + PER_LAYER * count
    table = []
    for number, spec in enumerate(layers, start=1):
        rows_at = base + len(body) // 32
        for j in range(spec["out"]):
            assert len(spec["weights"][j]) == spec["in"]
            body += packed(spec["weights"][j])
        bias_at = base + len(body) // 32
        for j in range(spec["out"]):
            body += word(spec["bias"][j])
        proofs_at = base + len(body) // 32
        for j in range(spec["out"]):
            e = entries[(number, j)]
            mine = leaf(number, j, spec["bias"][j], spec["weights"][j])
            assert "0x" + mine.hex() == e["leaf"], "leaf %d.%d differs from the tree" % (number, j)
            proof = [bytes.fromhex(p[2:]) for p in e["proof"]]
            assert climb(mine, proof) == root, "proof %d.%d does not reach the root" % (number, j)
            body += word(len(proof))
            for p in proof:
                body += p
            body += bytes(32 * (slots - len(proof)))
        shift = spec["shift"] if number < count else 0
        assert number == count or isinstance(shift, int), "a hidden layer has no shift"
        table.append((spec["in"], spec["out"], shift, rows_at, bias_at, proofs_at))

    n_in = layers[0]["in"]
    classes = layers[-1]["out"]
    with open(args.inputs or os.path.join(args.ref, "inputs.bin"), "rb") as f:
        inputs = f.read()
    with open(os.path.join(args.ref, "trace.jsonl"), "r", encoding="utf-8") as f:
        every = [json.loads(line) for line in f.readlines()]
    chosen = []
    missing = []
    for c in range(classes):
        first = next((t["i"] for t in every if t["class"] == c), None)
        if first is None:
            missing.append(c)
        else:
            chosen.append(first)
    chosen += [i for i in range(len(every)) if i not in chosen][: max(0, args.samples - len(chosen))]
    chosen = chosen[: args.samples]
    samples_at = base + len(body) // 32
    sample_words = None
    for n in chosen:
        t = every[n]
        assert t["i"] == n
        x0 = list(inputs[n * n_in : (n + 1) * n_in])
        assert len(x0) == n_in and max(x0) <= 127, "sample %d is not an input" % n
        hidden, logits, cls = forward(model, x0)
        assert hidden == t["hidden"] and logits == t["logits"] and cls == t["class"], "sample %d" % n
        start = len(body)
        body += word(n) + packed(x0)
        for h in hidden:
            body += packed(h)
        for z in logits:
            body += word(z)
        body += word(cls)
        sample_words = (len(body) - start) // 32

    out = bytearray(FILE_TAG.ljust(32, b"\0"))
    out += root + word(count) + word(slots) + word(len(chosen)) + word(samples_at) + word(sample_words)
    for row in table:
        for value in row:
            out += word(value)
    assert len(out) == 32 * base
    out += body
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(bytes(out))
    print("wrote %s: %s, %d words, samples %s%s, root %s" % (
        args.out, model["name"], len(out) // 32, chosen,
        (", no sample of class %s" % missing) if missing else "", tree["root"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
