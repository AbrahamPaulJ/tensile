"""Arguments for `cast` calls to a tree-ensemble Arbiter (docs/spec-trees.md), one per line.

    python3 tools/trees_args.py --ref DIR --forest FOREST.json [--sample N] COMMAND ...

DIR holds model.json, inputs.bin and trace.jsonl; FOREST.json is the commitment that
tools/gen_trees.py writes to build/forest.json. Commands:

    signature claim|tree|class|ensemble   the function signature
    claim [--mutate tree:T|class]         x0, outputs, class of sample N, honest or with the
                                          output of tree T (or the class) changed by one step
    challenge T [--mutate ...]            t, x0, outputs, tree words, proof: a challenge of tree T
                                          against that claim
    outputs [--mutate ...]                the outputs words of that claim (for challengeClass)
    x0                                    the input words of sample N
    root                                  the model root

Arrays are printed in cast's form, `[0x..,0x..]`. Standard library only.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trees_ref as ref  # noqa: E402


def array(words):
    return "[" + ",".join("0x%064x" % w for w in words) + "]"


def input_words(x):
    words = [0] * ((len(x) + 31) // 32)
    for i, v in enumerate(x):
        words[i // 32] |= v << ((31 - i % 32) * 8)
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True)
    parser.add_argument("--forest", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--mutate", default="")
    parser.add_argument("command")
    parser.add_argument("rest", nargs="*")
    args = parser.parse_args()
    model = ref.load_model(os.path.join(args.ref, "model.json"))
    with open(args.forest, encoding="utf-8") as f:
        forest = json.load(f)
    shape = forest["shape"]
    xw, lw = shape["input_words"], shape["output_words"]

    if args.command == "signature":
        kind = args.rest[0]
        print({
            "claim": "claim(uint256[%d],uint256[%d],uint8)" % (xw, lw),
            "tree": "challengeTree(uint256,uint256,uint256[%d],uint256[%d],uint256[%d],bytes32[])"
                    % (xw, lw, shape["tree_words"]),
            "class": "challengeClass(uint256,uint256[%d])" % lw,
            "ensemble": "classify(uint256[%d])" % xw,
        }[kind])
        return 0
    if args.command == "root":
        print(forest["root"])
        return 0

    n = model["in"]
    with open(os.path.join(args.ref, "inputs.bin"), "rb") as f:
        f.seek(args.sample * n)
        x = list(f.read(n))
    _, values, _, cls = ref.run(model, x)
    values = list(values)
    if args.mutate.startswith("tree:"):
        t = int(args.mutate[5:])
        values[t] = 32766 if values[t] == 32767 else values[t] + 1
    elif args.mutate == "class":
        cls = (cls + 1) % model["classes"]
    elif args.mutate:
        raise SystemExit("trees_args: unknown mutation %r" % args.mutate)
    x0 = array(input_words(x))
    outputs = array(ref.outputs_words(values))

    if args.command == "x0":
        print(x0)
    elif args.command == "claim":
        print(x0)
        print(outputs)
        print(cls)
    elif args.command == "outputs":
        print(outputs)
    elif args.command == "challenge":
        t = int(args.rest[0])
        entry = forest["trees"][t]
        print(t)
        print(x0)
        print(outputs)
        print("[" + ",".join(entry["words"]) + "]")
        print("[" + ",".join(entry["proof"]) + "]")
    else:
        raise SystemExit("trees_args: unknown command %r" % args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
