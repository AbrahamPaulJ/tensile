"""Write model.json: a small dense model with made-up weights, 40 -> 12 -> 8 -> 4.

Usage:  python3 examples/tiny/make_model.py

It stands in for "a model you trained and quantised yourself" in docs/USAGE.md and in the
bring-your-own check (check.sh in this folder). The weights are random integers from a fixed seed:
the network computes nothing useful, and that is the point. Nothing in the tools knows this
model, its sizes (an input that is not a multiple of 32, three layers, four classes) or its
scales. Standard library only; the output is the same on every machine.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7
SIZES = [40, 12, 8, 4]
# (weight fractional bits, shift) per layer; null shift on the last. With the input at 7
# fractional bits the hidden vectors have 5 and 4, the logits 10.
SCALES = [(7, 9), (6, 7), (6, None)]


def main():
    rng = random.Random(SEED)
    layers = []
    for k, (fw, shift) in enumerate(SCALES):
        n_in, n_out = SIZES[k], SIZES[k + 1]
        layers.append({
            "in": n_in,
            "out": n_out,
            "weight_frac_bits": fw,
            "shift": shift,
            "weights": [[rng.randint(-127, 127) for _ in range(n_in)] for _ in range(n_out)],
            "bias": [rng.randint(-20000, 60000) for _ in range(n_out)],
        })
    model = {"name": "tiny-40-12-8-4", "input_frac_bits": 7, "layers": layers}
    with open(os.path.join(HERE, "model.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(model, f)
        f.write("\n")
    print("wrote model.json: %s" % " -> ".join(str(n) for n in SIZES))


if __name__ == "__main__":
    main()
