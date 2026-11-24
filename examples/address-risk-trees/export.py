"""The reference vectors of the forest, and its test metrics, from the pure-Python reference.

    python -I export.py

Reads the trained forest (model/*.json, written by train.py) and the 7,054 test rows of the dense
address-risk example (../address-risk/out/inputs.bin and labels.bin: train.py checked that its own
quantised test rows are these bytes), and writes out/:

    model.json    the forest, with "metrics": the integer forest's test metrics recomputed here
    inputs.bin    the test rows, one byte per feature (a copy)
    labels.bin    their labels, 1 = flagged (a copy)
    pred.bin      the forest's class for every row
    trace.jsonl   the first 1,000 rows: every tree's leaf and value, the scores, the class

The metrics must equal the ones train.py recorded for the integer forest. Standard library only,
so check.sh runs it again in CI and compares.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import trees_ref as ref  # noqa: E402
from trees_vectors import write_vectors  # noqa: E402

DENSE_OUT = os.path.join(HERE, "..", "address-risk", "out")


def metrics(pred, labels):
    tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 0)
    n = len(labels)
    return {"samples": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / n,
            "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
            "majority_baseline": max(tp + fn, fp + tn) / n}


def main():
    paths = sorted(glob.glob(os.path.join(HERE, "model", "*.json")))
    if len(paths) != 1:
        raise SystemExit("export: expected one model in model/, found %d" % len(paths))
    model = ref.load_model(paths[0])
    with open(os.path.join(DENSE_OUT, "inputs.bin"), "rb") as f:
        inputs = f.read()
    with open(os.path.join(DENSE_OUT, "labels.bin"), "rb") as f:
        labels = f.read()
    if len(inputs) != model["in"] * len(labels):
        raise SystemExit("export: the inputs and labels do not fit the model")
    out = os.path.join(HERE, "out")
    pred = write_vectors(model, inputs, out, trace=1000)
    with open(os.path.join(out, "labels.bin"), "wb") as f:
        f.write(labels)
    got = metrics(pred, labels)
    want = model["training"]["test"]["integer"]
    for key, value in got.items():
        if abs(value - want[key]) > 1e-12:
            raise SystemExit("export: train.py recorded %s = %r, the reference gives %r"
                             % (key, want[key], value))
    model = dict(model)
    trees = model.pop("trees")
    model["metrics"] = got
    model["trees"] = trees
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(ref.model_to_json(model))
    print("export: %s, %d test rows: accuracy %.4f, precision %.4f, recall %.4f, F1 %.4f, "
          "majority baseline %.4f (pure-Python reference, equal to train.py's record)"
          % (model["name"], got["samples"], got["accuracy"], got["precision"], got["recall"],
             got["f1"], got["majority_baseline"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
