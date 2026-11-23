"""Run the NumPy reference on any dense model and write what the generators need.

Usage:
    python tools/ref_vectors.py --model MODEL.json --out DIR [--inputs INPUTS.bin | --random N]
                                [--seed S] [--trace N]

MODEL.json is a model in the format of docs/spec.md section 5: "name", "input_frac_bits" (7),
and per layer "in", "out", "weight_frac_bits", "shift" (null on the last layer), "weights"
(OUT rows of IN integers in -127..127) and "bias" (OUT int32). "worst_case_abs_acc" may be
missing: it is computed here.

Writes into DIR, in the formats of docs/spec.md section 6:
    model.json    the model with "worst_case_abs_acc" per layer (what tools/gen_fe.py reads)
    inputs.bin    the inputs, one byte per activation, each 0..127
    pred.bin      the class of every input
    trace.jsonl   hidden activations, logits and class of the first --trace inputs (default 1000)

The inputs are the file given with --inputs, or --random N vectors drawn with --seed (default 1)
when there is no data set at hand. This is ref/tensile_ref.py, the oracle, and nothing else: the
MNIST exporters (ref/export.py and the examples) add only the data set and the accuracy figures.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))

import tensile_ref as ref  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--random", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace", type=int, default=1000)
    args = parser.parse_args()

    model = ref.load_model(args.model)
    n_in = model["layers"][0]["in"]
    if args.inputs is not None:
        raw = np.fromfile(args.inputs, dtype=np.uint8)
        if raw.size == 0 or raw.size % n_in != 0:
            raise SystemExit("ref_vectors: %s is not a whole number of %d-byte inputs" % (args.inputs, n_in))
        x0 = raw.reshape(-1, n_in).astype(np.int64)
    elif args.random is not None:
        x0 = np.random.default_rng(args.seed).integers(0, 128, (args.random, n_in), dtype=np.int64)
    else:
        raise SystemExit("ref_vectors: give --inputs FILE or --random N")
    hidden, logits, cls = ref.run(model, x0)
    if model["layers"][-1]["out"] > 255:
        raise SystemExit("ref_vectors: pred.bin holds one byte per class: at most 255 outputs")

    layers = []
    for lay in model["layers"]:
        bound = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        layers.append({"in": lay["in"], "out": lay["out"], "weight_frac_bits": lay["weight_frac_bits"],
                       "shift": lay["shift"], "worst_case_abs_acc": bound,
                       "weights": lay["weights"], "bias": lay["bias"]})
    out_model = {"name": model["name"], "input_frac_bits": model["input_frac_bits"], "layers": layers}

    os.makedirs(args.out, exist_ok=True)

    def write(name, data):
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(data)

    write("model.json", ref.model_to_json(out_model).encode("utf-8"))
    write("inputs.bin", x0.astype(np.uint8).tobytes(order="C"))
    write("pred.bin", cls.astype(np.uint8).tobytes())
    lines = []
    for i in range(min(args.trace, x0.shape[0])):
        lines.append(json.dumps({"i": i, "hidden": [h[i].tolist() for h in hidden],
                                 "logits": logits[i].tolist(), "class": int(cls[i])}))
    write("trace.jsonl", ("\n".join(lines) + "\n").encode("utf-8"))
    print("%s: %d inputs, %d traced, worst-case |acc| per layer %s, wrote %s"
          % (model["name"], x0.shape[0], len(lines), [lay["worst_case_abs_acc"] for lay in layers], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
