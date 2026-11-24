"""Export the quantised convolutional fee-tier model and its test vectors to out/, and score it on
the test split.

Usage:  python examples/amm-fee-conv/export.py        (NumPy; run after train.py)

Writes, in the formats of docs/spec.md sections 5, 6 and 8:
    out/model.json    the model with worst_case_abs_acc per layer and a "metrics" record
    out/inputs.bin    every sample of the test split, 288 bytes each (dataset.py's rule)
    out/labels.bin    the tier that followed each sample
    out/past.bin      the persistence baseline's answer for each sample
    out/pred.bin      the class the integer reference predicts
    out/trace.jsonl   hidden activations, logits and class of the first 100 samples (5,888 hidden values each)

inputs.bin, labels.bin and past.bin are not committed: check.sh rebuilds them from
../amm-fee/data/ticks.csv with dataset.py (standard library) and recomputes every figure of
"metrics" from the committed pred.bin, so the scores in the README can be checked without NumPy.
"metrics" also holds the dense 144-32-16-3 model's score on the same test windows, from its
committed predictions.

This is the only place the test split is scored, and nothing is chosen by that score.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))
sys.path.insert(0, HERE)

import dataset  # noqa: E402
import tensile_ref as ref  # noqa: E402

TRACE_SAMPLES = 100
CHUNK = 1000


def run_chunked(model, x0):
    """tensile_ref.run in chunks of CHUNK samples (a convolution's patches are large)."""
    hidden, logits, cls = None, [], []
    for start in range(0, x0.shape[0], CHUNK):
        h, lg, c = ref.run(model, x0[start:start + CHUNK])
        if hidden is None:
            hidden = [[] for _ in h]
        for k, v in enumerate(h):
            hidden[k].append(v)
        logits.append(lg)
        cls.append(c)
    return [np.concatenate(v) for v in hidden], np.concatenate(logits), np.concatenate(cls)


def main():
    t0 = time.perf_counter()
    dataset.main()
    out_dir = os.path.join(HERE, "out")
    trained = ref.load_model(dataset.MODEL_PATH)
    raw = np.fromfile(os.path.join(out_dir, "inputs.bin"), dtype=np.uint8)
    x0 = raw.reshape(-1, dataset.N_OBS).astype(np.int64)
    hidden, logits, cls = run_chunked(trained, x0)

    def write(name, data):
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)

    write("pred.bin", cls.astype(np.uint8).tobytes())
    lines = []
    for i in range(min(TRACE_SAMPLES, x0.shape[0])):
        lines.append(json.dumps({"i": i, "hidden": [h[i].tolist() for h in hidden],
                                 "logits": logits[i].tolist(), "class": int(cls[i])}))
    write("trace.jsonl", ("\n".join(lines) + "\n").encode("utf-8"))

    layers = []
    for lay in trained["layers"]:
        bound = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        assert bound <= ref.INT32_MAX
        entry = {}
        if lay.get("kind") == "conv1d":
            entry.update({"kind": "conv1d", "channels": lay["channels"], "kernel": lay["kernel"],
                          "stride": lay["stride"]})
        entry.update({"in": lay["in"], "out": lay["out"], "weight_frac_bits": lay["weight_frac_bits"],
                      "shift": lay["shift"], "worst_case_abs_acc": bound,
                      "weights": lay["weights"], "bias": lay["bias"]})
        layers.append(entry)
    metrics = dataset.metrics(out_dir)
    dense_correct, _ = dataset.dense_on_the_same_windows(out_dir)
    metrics["dense_144_32_16_3_correct"] = dense_correct
    model = {"name": trained["name"], "input_frac_bits": trained["input_frac_bits"], "layers": layers,
             "metrics": metrics}
    write("model.json", ref.model_to_json(model).encode("utf-8"))

    reloaded = ref.load_model(os.path.join(out_dir, "model.json"))
    _, logits2, cls2 = run_chunked(reloaded, x0)
    assert np.array_equal(logits2, logits) and np.array_equal(cls2, cls)
    print(dataset.describe(metrics) + "; the dense 144-32-16-3 model on the same windows %d" % dense_correct)
    for name in ("model.json", "inputs.bin", "labels.bin", "past.bin", "pred.bin", "trace.jsonl"):
        path = os.path.join(out_dir, name)
        print("%s  %9d bytes  %s" % (dataset.sha256(path), os.path.getsize(path), name))
    print("export time %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
