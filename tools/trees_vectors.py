"""Reference vectors of a tree ensemble (docs/spec-trees.md), from the pure-Python reference.

    python tools/trees_vectors.py --model MODEL.json --inputs INPUTS.bin --out DIR [--trace N]

Writes, under DIR:

    inputs.bin    the inputs, copied (one byte per feature, 0..127)
    pred.bin      the class of every sample, one byte each
    trace.jsonl   the first N samples (default 1,000), one JSON object per line:
                  {"i", "leaves", "outputs", "scores", "class"}: the leaf every tree ends at, its
                  value, the class scores, the class

These are what the native Fe classifier (tools/diff_native_trees.py) and the generated Fe unit
tests (tools/gen_trees.py) must reproduce. Standard library only.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trees_ref as ref  # noqa: E402


def write_vectors(model, inputs, out, trace=1000):
    n = model["in"]
    if len(inputs) % n:
        raise SystemExit("trees_vectors: %d input bytes are not whole samples of %d" % (len(inputs), n))
    os.makedirs(out, exist_ok=True)
    pred = bytearray()
    with open(os.path.join(out, "trace.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for s in range(len(inputs) // n):
            leaves, values, scores, cls = ref.run(model, inputs[s * n:(s + 1) * n])
            pred.append(cls)
            if s < trace:
                f.write(json.dumps({"i": s, "leaves": leaves, "outputs": values, "scores": scores,
                                    "class": cls}, separators=(",", ":")) + "\n")
    with open(os.path.join(out, "inputs.bin"), "wb") as f:
        f.write(inputs)
    with open(os.path.join(out, "pred.bin"), "wb") as f:
        f.write(bytes(pred))
    return bytes(pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", type=int, default=1000)
    args = parser.parse_args()
    model = ref.load_model(args.model)
    with open(args.inputs, "rb") as f:
        inputs = f.read()
    pred = write_vectors(model, inputs, args.out, args.trace)
    print("trees_vectors: %d samples, %d traced, into %s" % (len(pred), min(args.trace, len(pred)), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
