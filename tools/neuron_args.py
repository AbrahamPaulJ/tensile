"""Print the arguments of one single-neuron call, for `cast`.

Usage:
    python3 tools/neuron_args.py LAYER NEURON SAMPLE [--ref ref/out]

LAYER counts from 1, NEURON and SAMPLE from 0; SAMPLE must be one of the traced samples. Prints four
lines: the row of weights as packed words, the layer's input for that sample as packed words, the
bias, and the result the reference expects (the activation for a hidden layer, the logit for the
last one). Packed words are `docs/spec.md` section 7. Standard library only.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pack_words(values):
    """Value i is byte i % 32 of word i // 32, most significant byte first, zero padding."""
    words = []
    for start in range(0, len(values), 32):
        chunk = bytes(v & 0xFF for v in values[start:start + 32])
        words.append("0x" + chunk.ljust(32, b"\0").hex())
    return "[" + ",".join(words) + "]"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("layer", type=int)
    parser.add_argument("neuron", type=int)
    parser.add_argument("sample", type=int)
    parser.add_argument("--ref", default=os.path.join(ROOT, "ref", "out"))
    args = parser.parse_args()

    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        layers = json.load(f)["layers"]
    with open(os.path.join(args.ref, "trace.jsonl"), "r", encoding="utf-8") as f:
        trace = json.loads(f.readlines()[args.sample])
    layer = layers[args.layer - 1]
    if args.layer == 1:
        with open(os.path.join(args.ref, "inputs.bin"), "rb") as f:
            f.seek(args.sample * layer["in"])
            x = list(f.read(layer["in"]))
    else:
        x = trace["hidden"][args.layer - 2]
    if args.layer == len(layers):
        expected = trace["logits"][args.neuron]
    else:
        expected = trace["hidden"][args.layer - 1][args.neuron]

    print(pack_words(layer["weights"][args.neuron]))
    print(pack_words(x))
    print(layer["bias"][args.neuron])
    print(expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
