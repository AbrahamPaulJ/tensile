"""Print the signatures and arguments of the generated Arbiter's calls, for `cast`, for any model.

Usage:
    python3 tools/model_args.py --ref DIR [--tree FILE] [--binding flat|tree] COMMAND

    layers
        the number of layers
    kind K
        dense or conv1d
    signature claim | challenge K | class
        the Solidity signature of the call (what `cast send` and `cast calldata` take):
        `challengeNeuronK` for a dense layer, `challengeFilterK` for a convolution
    claim [--sample N] [--mutate out:K:J | class]
        one line per argument of `claim`: the input words, the words of every hidden vector, the
        logits, the class; for the honest claim of traced sample N, or with output J of layer K
        changed by one (a hidden activation, or a logit when K is the last layer), or with the
        class changed by one modulo the number of classes.
    challenge K [--sample N] [--mutate ...]
        the two vector arguments of `challengeNeuronK` for that claim: the layer's input, then its
        output.
    filter K --position P --filter J [--sample N] [--mutate ...]
        the vector arguments of `challengeFilterK` (a convolution) for output P * filters + J of
        that claim: with the flat binding the layer's input and output; with the word-tree binding
        the input words the window reads, their proofs, the output word that holds the value and
        its proof (docs/spec-dispute.md section 11).
    proof K J
        three lines: row J of layer K as packed words, its bias, its Merkle proof as `bytes32[]`.

DIR holds model.json, inputs.bin and trace.jsonl (docs/spec.md sections 5 and 6); FILE is the
tree tools/gen_tree.py wrote (default DIR/../build/tree.json). The binding defaults to the one
tools/gen_arbiter.py chooses: tree for a model with a convolution, flat otherwise. The general form of
tools/claim_args.py, which is fixed to the shipped two-layer model. Standard library only.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_arbiter import Shape, challenge_signature, claim_signature, conv_signature  # noqa: E402
import gen_tree  # noqa: E402


def pack_words(values):
    words = []
    for start in range(0, len(values), 32):
        chunk = bytes(v & 0xFF for v in values[start:start + 32])
        words.append("0x" + chunk.ljust(32, b"\0").hex())
    return "[" + ",".join(words) + "]"


def load_model(args):
    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def claim_vectors(args, model):
    """[x0, hidden 1, .., logits] as lists of integers, and the class, after the mutation."""
    n_in = model["layers"][0]["in"]
    count = len(model["layers"])
    with open(os.path.join(args.ref, "inputs.bin"), "rb") as f:
        f.seek(args.sample * n_in)
        x0 = list(f.read(n_in))
    with open(os.path.join(args.ref, "trace.jsonl"), "r", encoding="utf-8") as f:
        trace = json.loads(f.readlines()[args.sample])
    vectors = [x0] + [list(h) for h in trace["hidden"]] + [list(trace["logits"])]
    cls = trace["class"]
    if args.mutate:
        parts = args.mutate.split(":")
        if parts[0] == "class" and len(parts) == 1:
            cls = (cls + 1) % model["layers"][-1]["out"]
        elif parts[0] == "out" and len(parts) == 3:
            k, j = int(parts[1]), int(parts[2])
            if k == count:
                vectors[k][j] += 1
            else:
                vectors[k][j] += -1 if vectors[k][j] == 127 else 1
        else:
            raise SystemExit("model_args: unknown mutation %r" % args.mutate)
    return vectors, cls


def word_bytes(values):
    """The packed words of `values` as a list of 32-byte strings."""
    out = []
    for start in range(0, len(values), 32):
        out.append(bytes(v & 0xFF for v in values[start:start + 32]).ljust(32, b"\0"))
    return out


def hex_list(items):
    return "[" + ",".join("0x" + item.hex() for item in items) + "]"


def filter_arguments(shape, vectors, k, position, j):
    """The vector arguments of challengeFilterK, one per line, as gen_arbiter.conv_fields orders them."""
    if not shape.tree:
        return [as_argument(vectors, k - 1), as_argument(vectors, k)]
    c = shape.conv[k]
    x, y = word_bytes(vectors[k - 1]), word_bytes(vectors[k])
    first = min(position * c["S"] * c["C"] // 32, c["in_words"] - c["span"])
    span = x[first:first + c["span"]]
    proofs = [node for i in range(c["span"]) for node in gen_tree.word_proof(x, first + i)]
    at = (position * c["F"] + j) // 32
    return [hex_list(span), hex_list(proofs), "0x" + y[at].hex(), hex_list(gen_tree.word_proof(y, at))]


def as_argument(vectors, index):
    if index == len(vectors) - 1:
        return "[" + ",".join(str(v) for v in vectors[index]) + "]"
    return pack_words(vectors[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True)
    parser.add_argument("--tree", default=None)
    parser.add_argument("--binding", choices=["flat", "tree"], default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("layers")
    commands.add_parser("kind").add_argument("layer", type=int)
    s = commands.add_parser("signature")
    s.add_argument("what", choices=["claim", "challenge", "class"])
    s.add_argument("layer", type=int, nargs="?")
    for name in ("claim", "challenge", "filter"):
        c = commands.add_parser(name)
        if name in ("challenge", "filter"):
            c.add_argument("layer", type=int)
        if name == "filter":
            c.add_argument("--position", type=int, required=True)
            c.add_argument("--filter", type=int, required=True)
        c.add_argument("--sample", type=int, default=0)
        c.add_argument("--mutate", default=None)
    p = commands.add_parser("proof")
    p.add_argument("layer", type=int)
    p.add_argument("neuron", type=int)
    args = parser.parse_args()

    model = load_model(args)
    binding = args.binding
    if binding is None:
        binding = "tree" if any(layer.get("kind") == "conv1d" for layer in model["layers"]) else "flat"
    if args.command == "layers":
        print(len(model["layers"]))
    elif args.command == "kind":
        print(model["layers"][args.layer - 1].get("kind", "dense"))
    elif args.command == "signature":
        shape = Shape(model, binding)
        if args.what == "claim":
            print(claim_signature(shape))
        elif args.what == "challenge" and shape.is_conv(args.layer):
            print(conv_signature(shape, args.layer))
        elif args.what == "challenge":
            print(challenge_signature(shape, args.layer))
        else:
            print("challengeClass(uint256,int32[%d])" % shape.classes)
    elif args.command == "claim":
        vectors, cls = claim_vectors(args, model)
        for index in range(len(vectors)):
            print(as_argument(vectors, index))
        print(cls)
    elif args.command == "challenge":
        vectors, _ = claim_vectors(args, model)
        print(as_argument(vectors, args.layer - 1))
        print(as_argument(vectors, args.layer))
    elif args.command == "filter":
        shape = Shape(model, binding)
        vectors, _ = claim_vectors(args, model)
        for line in filter_arguments(shape, vectors, args.layer, args.position, args.filter):
            print(line)
    else:
        tree_path = args.tree or os.path.join(args.ref, "..", "build", "tree.json")
        layer = model["layers"][args.layer - 1]
        with open(tree_path, "r", encoding="utf-8") as f:
            leaves = json.load(f)["leaves"]
        entry = next(x for x in leaves if x["layer"] == args.layer and x["neuron"] == args.neuron)
        print(pack_words(layer["weights"][args.neuron]))
        print(layer["bias"][args.neuron])
        print("[" + ",".join(entry["proof"]) + "]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
