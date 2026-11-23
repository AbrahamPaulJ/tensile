"""Export the quantised address-risk model and its test vectors to out/.

Usage:  python examples/address-risk/export.py [--data DIR]      (after train.py)

Writes, in the formats of docs/spec.md sections 5 and 6:
    out/model.json    the model, the worst-case accumulator of every layer, and the test metrics
    out/inputs.bin    the 7,054 test rows as activations, 14 bytes each (features.json's rule)
    out/labels.bin    their labels (1 = flagged in the data set)
    out/pred.bin      the class the integer reference predicts
    out/trace.jsonl   hidden activations, logits and class of the first 1,000

The output is a pure function of the model file, features.json and the data set: running it twice
gives byte-identical files. The metrics are recomputed here from the reference's predictions and
must equal the ones train.py recorded.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))

import features  # noqa: E402
import tensile_ref as ref  # noqa: E402

MODEL_PATH = os.path.join(HERE, "model", "address-risk-14-32-16-2.json")
OUT_DIR = os.path.join(HERE, "out")
TRACE_SAMPLES = 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=features.DEFAULT_DATA)
    args = parser.parse_args()
    t0 = time.perf_counter()
    trained = ref.load_model(MODEL_PATH)
    addresses, x, y = features.load(args.data)
    _, _, test_idx = features.split(addresses)
    x0 = features.quantise(x[test_idx], features.load_thresholds()).astype(np.int64)
    labels = y[test_idx]
    hidden, logits, cls = ref.run(trained, x0)
    measured = features.metrics(cls, labels)
    if measured != trained["training"]["test"]["integer"]:
        raise SystemExit("export: the test metrics differ from the ones train.py recorded")

    layers = []
    for lay in trained["layers"]:
        bound = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        layers.append({"in": lay["in"], "out": lay["out"], "weight_frac_bits": lay["weight_frac_bits"],
                       "shift": lay["shift"], "worst_case_abs_acc": bound,
                       "weights": lay["weights"], "bias": lay["bias"]})
    model = {
        "name": trained["name"],
        "input_frac_bits": trained["input_frac_bits"],
        "layers": layers,
        "accuracy": {"float": trained["float_accuracy"], "integer": measured["accuracy"],
                     "test_samples": measured["samples"]},
        "metrics": measured,
    }

    os.makedirs(OUT_DIR, exist_ok=True)

    def write(name, data):
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(data)

    write("model.json", ref.model_to_json(model).encode("utf-8"))
    write("inputs.bin", x0.astype(np.uint8).tobytes(order="C"))
    write("labels.bin", labels.astype(np.uint8).tobytes())
    write("pred.bin", cls.astype(np.uint8).tobytes())
    lines = []
    for i in range(TRACE_SAMPLES):
        lines.append(json.dumps({"i": i, "hidden": [h[i].tolist() for h in hidden],
                                 "logits": logits[i].tolist(), "class": int(cls[i])}))
    write("trace.jsonl", ("\n".join(lines) + "\n").encode("utf-8"))

    reloaded = ref.load_model(os.path.join(OUT_DIR, "model.json"))
    _, logits2, cls2 = ref.run(reloaded, x0)
    assert np.array_equal(logits2, logits) and np.array_equal(cls2, cls)
    print("%s: %d test rows, %d traced; accuracy %.4f, precision %.4f, recall %.4f, majority baseline %.4f"
          % (model["name"], x0.shape[0], len(lines), measured["accuracy"], measured["precision"],
             measured["recall"], measured["majority_baseline"]))
    print("export time %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
