"""Export the quantised model and the test vectors to ref/out/ (docs/spec.md, sections 5 and 6).

Usage:  python ref/export.py

Reads  ref/model/mnist-784-32-10.json (written by train.py) and the MNIST test set.
Writes ref/out/model.json, inputs.bin, labels.bin, pred.bin, trace.jsonl.

The output is a pure function of the model file and the MNIST test set: running it twice gives
byte-identical files (LF line endings, fixed key order, no timestamps).
"""

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import get_mnist  # noqa: E402
import tensile_ref as ref  # noqa: E402

MODEL_PATH = os.path.join(HERE, "model", "mnist-784-32-10.json")
OUT_DIR = os.path.join(HERE, "out")
TRACE_SAMPLES = 1000
OUT_FILES = ["model.json", "inputs.bin", "labels.bin", "pred.bin", "trace.jsonl"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(out_dir=OUT_DIR, verbose=True):
    """Write the five output files into out_dir. Returns a dict of facts about the export."""
    trained = ref.load_model(MODEL_PATH)
    get_mnist.ensure(verbose=False)
    x0 = ref.quantise_input(get_mnist.load_images("t10k"))  # x0 = pixel >> 1, values 0..127
    labels = get_mnist.load_labels("t10k").astype(np.int64)
    n = x0.shape[0]
    if n != 10000 or labels.shape != (n,):
        raise ValueError("expected the 10,000 image MNIST test set")

    hidden, logits, cls = ref.run(trained, x0)
    integer_acc = int((cls == labels).sum()) / n

    # Worst-case |accumulator| over every input in 0..127, per layer (the largest neuron bound).
    layers_out = []
    worst = []
    for lay in trained["layers"]:
        bound = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        assert bound <= ref.INT32_MAX, "worst-case accumulator %d does not fit int32" % bound
        worst.append(bound)
        layers_out.append({
            "in": lay["in"],
            "out": lay["out"],
            "weight_frac_bits": lay["weight_frac_bits"],
            "shift": lay["shift"],
            "worst_case_abs_acc": bound,
            "weights": lay["weights"],
            "bias": lay["bias"],
        })
    model = {
        "name": trained["name"],
        "input_frac_bits": trained["input_frac_bits"],
        "layers": layers_out,
        "accuracy": {
            "float": trained["float_accuracy"],
            "integer": integer_acc,
            "test_samples": n,
        },
    }

    os.makedirs(out_dir, exist_ok=True)

    def write(name, data):
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)

    write("model.json", ref.model_to_json(model).encode("utf-8"))
    write("inputs.bin", x0.astype(np.uint8).tobytes(order="C"))
    write("labels.bin", labels.astype(np.uint8).tobytes())
    write("pred.bin", cls.astype(np.uint8).tobytes())
    lines = []
    for i in range(TRACE_SAMPLES):
        lines.append(json.dumps({
            "i": i,
            "hidden": [h[i].tolist() for h in hidden],
            "logits": logits[i].tolist(),
            "class": int(cls[i]),
        }))
    write("trace.jsonl", ("\n".join(lines) + "\n").encode("utf-8"))

    # The written model must reproduce the predictions (guards the serialiser).
    reloaded = ref.load_model(os.path.join(out_dir, "model.json"))
    _, logits2, cls2 = ref.run(reloaded, x0)
    assert np.array_equal(logits2, logits) and np.array_equal(cls2, cls)

    # Observed accumulator extremes on the test set, against the worst-case bound.
    observed = []
    x = x0
    for li, lay in enumerate(trained["layers"]):
        acc = ref.layer_acc(lay["weights"], lay["bias"], x)
        observed.append(int(np.abs(acc).max()))
        assert observed[-1] <= worst[li]
        if lay["shift"] is not None:
            x = hidden[li]

    info = {
        "float_accuracy": trained["float_accuracy"],
        "integer_accuracy": integer_acc,
        "worst_case_abs_acc": worst,
        "observed_abs_acc": observed,
        "files": {name: (os.path.getsize(os.path.join(out_dir, name)),
                         sha256_file(os.path.join(out_dir, name))) for name in OUT_FILES},
    }
    if verbose:
        acc_bits, out_bits = ref.frac_bits(trained)
        for li, lay in enumerate(trained["layers"]):
            print("layer %d: %d -> %d, weight_frac_bits %d, shift %s, acc frac bits %d, "
                  "out frac bits %s" % (li + 1, lay["in"], lay["out"], lay["weight_frac_bits"],
                                        lay["shift"], acc_bits[li], out_bits[li]))
            print("  worst-case |acc| %d = %.4f%% of int32 max %d (largest seen on the test set: %d)"
                  % (worst[li], 100.0 * worst[li] / ref.INT32_MAX, ref.INT32_MAX, observed[li]))
        print("float accuracy   %.4f" % trained["float_accuracy"])
        print("integer accuracy %.4f (%d of %d)" % (integer_acc, round(integer_acc * n), n))
        for name in OUT_FILES:
            size, digest = info["files"][name]
            print("%s  %9d bytes  %s" % (digest, size, name))
    return info


def main():
    t0 = time.perf_counter()
    build()
    print("export time %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
