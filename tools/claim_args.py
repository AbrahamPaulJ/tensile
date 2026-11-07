"""Print the arguments of Arbiter calls, for `cast`.

Usage:
    python3 tools/claim_args.py claim [--sample N] [--mutate hidden:J | logit:J | class]
        prints four lines: x0 words, hidden words, logits, class (the claim's arguments), for the
        honest claim of traced sample N or with one value changed by one (the class by one modulo
        10).
    python3 tools/claim_args.py proof LAYER NEURON
        prints three lines: the neuron's row as packed words, its bias, its Merkle proof as a
        `bytes32[]`, from ref/out/model.json and build/tree.json (tools/gen_tree.py).

Packed words are docs/spec.md section 7. Standard library only.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pack_words(values):
    words = []
    for start in range(0, len(values), 32):
        chunk = bytes(v & 0xFF for v in values[start:start + 32])
        words.append("0x" + chunk.ljust(32, b"\0").hex())
    return "[" + ",".join(words) + "]"


def claim(args):
    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        n_in = json.load(f)["layers"][0]["in"]
    with open(os.path.join(args.ref, "inputs.bin"), "rb") as f:
        f.seek(args.sample * n_in)
        x0 = list(f.read(n_in))
    with open(os.path.join(args.ref, "trace.jsonl"), "r", encoding="utf-8") as f:
        trace = json.loads(f.readlines()[args.sample])
    hidden, logits, cls = list(trace["hidden"][0]), list(trace["logits"]), trace["class"]
    if args.mutate:
        kind, _, index = args.mutate.partition(":")
        if kind == "hidden":
            j = int(index)
            hidden[j] = hidden[j] - 1 if hidden[j] == 127 else hidden[j] + 1
        elif kind == "logit":
            logits[int(index)] += 1
        elif kind == "class":
            cls = (cls + 1) % 10
        else:
            raise SystemExit("claim_args: unknown mutation %r" % args.mutate)
    print(pack_words(x0))
    print(pack_words(hidden))
    print("[" + ",".join(str(v) for v in logits) + "]")
    print(cls)


def proof(args):
    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        layer = json.load(f)["layers"][args.layer - 1]
    with open(args.tree, "r", encoding="utf-8") as f:
        leaves = json.load(f)["leaves"]
    entry = next(x for x in leaves if x["layer"] == args.layer and x["neuron"] == args.neuron)
    print(pack_words(layer["weights"][args.neuron]))
    print(layer["bias"][args.neuron])
    print("[" + ",".join(entry["proof"]) + "]")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default=os.path.join(ROOT, "ref", "out"))
    parser.add_argument("--tree", default=os.path.join(ROOT, "build", "tree.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("claim")
    c.add_argument("--sample", type=int, default=0)
    c.add_argument("--mutate", default=None)
    c.set_defaults(run=claim)
    p = commands.add_parser("proof")
    p.add_argument("layer", type=int)
    p.add_argument("neuron", type=int)
    p.set_defaults(run=proof)
    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
