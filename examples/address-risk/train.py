"""Train the address-risk network (14 -> 32 -> 16 -> 2) in plain NumPy and quantise it to docs/spec.md.

Usage:  python examples/address-risk/train.py [--data DIR]      (after prepare.py)

  1. Split the 35,272 addresses into training, validation and test rows (features.py, fixed
     seed) and fit the feature thresholds on the training rows: features.json.
  2. Float training (ReLU hidden layers, Adam with decoupled weight decay, cross-entropy with a
     weight on the flagged class), once per candidate class weight in CLASS_WEIGHTS.
  3. Post-training quantisation of each candidate: a grid search over the power-of-two scales of
     every layer. Every score comes from tensile_ref.run, the integer reference itself.
  4. The candidate whose integer model has the best F1 on the validation rows is written to
     model/address-risk-14-32-16-2.json. Only then are the test rows scored, once.

Every choice (class weight, scales) is made on the validation rows. The test rows are read by one
call at the end of main() and by export.py, and by nothing that feeds back into the model.

The optimiser, the loss gradient and the learning-rate schedule are ref/train.py's. Deterministic
for the fixed seeds on one machine (floating point: another CPU or BLAS build may round
differently). export.py in this folder turns the saved model into out/.
"""

import argparse
import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))

import features  # noqa: E402
import tensile_ref as ref  # noqa: E402
import train as base  # noqa: E402  (ref/train.py)

MODEL_NAME = "address-risk-14-32-16-2"
MODEL_PATH = os.path.join(HERE, "model", MODEL_NAME + ".json")
SIZES = [14, 32, 16, 2]
SEED = 20261124
F0 = ref.INPUT_FRAC_BITS
EPOCHS = 60
BATCH = 100
LR = 2e-3
WEIGHT_DECAY = 0.01
# Candidate weights of the flagged class in the loss (the other class has weight 1). The data is
# 16% flagged, so weight 1 favours the majority; the choice is made on validation F1.
CLASS_WEIGHTS = [1.0, 1.5, 2.0, 3.0]


def float_forward(params, x_real):
    """params = [w1, b1, w2, b2, ...]. Returns (pre-activations, activations, logits)."""
    zs, hs = [], []
    h = x_real
    count = len(params) // 2
    for k in range(count):
        z = h @ params[2 * k].T + params[2 * k + 1]
        if k == count - 1:
            return zs, hs, z
        zs.append(z)
        h = np.maximum(z, 0.0)
        hs.append(h)


def weighted_xent(logits, labels, class_weight):
    """Cross-entropy with weight `class_weight` on rows of class 1, and its gradient."""
    n = logits.shape[0]
    loss, g = base.softmax_xent(logits, labels)  # g is the mean gradient, already divided by n
    w = np.where(labels == 1, class_weight, 1.0)
    g *= (w * (n / w.sum()))[:, None]
    return loss, g


def train_float(x0, labels, class_weight, rng, sizes=None, epochs=EPOCHS):
    sizes = sizes or SIZES
    count = len(sizes) - 1
    params = []
    for k in range(count):
        n_in, n_out = sizes[k], sizes[k + 1]
        gain = 1.0 if k == count - 1 else 2.0
        params += [rng.normal(0.0, np.sqrt(gain / n_in), (n_out, n_in)), np.zeros(n_out)]
    opt = base.Adam(params, WEIGHT_DECAY, [True, False] * count)
    x_real = x0.astype(np.float64) / 128.0
    n = x0.shape[0]
    steps_per_epoch = n // BATCH
    total = epochs * steps_per_epoch
    step = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            acts = [x_real[idx]]
            zs = []
            for k in range(count):
                z = acts[-1] @ params[2 * k].T + params[2 * k + 1]
                zs.append(z)
                if k < count - 1:
                    acts.append(np.maximum(z, 0.0))
            _, g = weighted_xent(zs[-1], labels[idx], class_weight)
            grads = [None] * (2 * count)
            for k in range(count - 1, -1, -1):
                grads[2 * k] = g.T @ acts[k]
                grads[2 * k + 1] = g.sum(axis=0)
                if k > 0:
                    g = (g @ params[2 * k]) * (zs[k - 1] > 0.0)
            opt.step(grads, base.cosine_lr(LR, step, total))
            step += 1
    return params


def quantise(params, cfg, sizes=None):
    """Float parameters to an integer model dict. cfg = (fw_1, f_1, fw_2, f_2, ..., fw_L): the
    weight fractional bits of each layer and the fractional bits of each hidden activation. The
    shift of a hidden layer follows from the spec: shift_k = f_{k-1} + fw_k - f_k. A hidden bias
    gets half of the shift divisor added, which turns the floor of the shift into
    round-to-nearest (as ref/train.py and the MNIST example)."""
    sizes = sizes or SIZES
    count = len(params) // 2
    layers = []
    f_prev = F0
    for k in range(count):
        w, b = params[2 * k], params[2 * k + 1]
        fw = cfg[2 * k]
        last = k == count - 1
        wq = np.clip(np.rint(w * 2.0 ** fw), -127, 127).astype(np.int64)
        bq = np.rint(b * 2.0 ** (f_prev + fw)).astype(np.int64)
        if last:
            shift = None
        else:
            f_out = cfg[2 * k + 1]
            shift = f_prev + fw - f_out
            if shift < 0:
                raise ValueError("negative shift")
            bq = bq + ((1 << (shift - 1)) if shift > 0 else 0)
        layers.append({"in": sizes[k], "out": sizes[k + 1], "weight_frac_bits": int(fw),
                       "shift": None if last else int(shift), "weights": wq, "bias": bq})
        if not last:
            f_prev = f_out
    return {"name": MODEL_NAME, "input_frac_bits": F0, "layers": layers}


def int_metrics(model, x0, labels):
    _, _, cls = ref.run(model, x0.astype(np.int64))
    return features.metrics(cls, labels)


def float_metrics(params, x0, labels):
    _, _, logits = float_forward(params, x0.astype(np.float64) / 128.0)
    return features.metrics(np.argmax(logits, axis=1), labels)


def score(m):
    """Sort key of a metrics dict: F1 of the flagged class, then accuracy."""
    return (m["f1"], m["accuracy"])


def ptq_search(params, x0_cal, x0_val, y_val, sizes=None):
    """Pick the scales by validation score over a grid: for every layer the largest weight scale
    with no clipped weight or one bit more, for every hidden layer the largest activation scale
    with no clipped activation (on the calibration rows) or up to three bits more."""
    count = len(params) // 2
    _, hs, _ = float_forward(params, x0_cal.astype(np.float64) / 128.0)
    axes = []
    for k in range(count):
        m = int(np.floor(np.log2(127.0 / np.abs(params[2 * k]).max())))
        axes.append((m, m + 1))
        if k < count - 1:
            g = int(np.floor(np.log2(127.0 / float(hs[k].max()))))
            axes.append((g, g + 1, g + 2, g + 3))
    best = None
    for cfg in itertools.product(*axes):
        try:
            model = quantise(params, cfg, sizes)
            ref.check_model(model)
        except (ValueError, OverflowError):
            continue
        res = int_metrics(model, x0_val, y_val)
        if best is None or score(res) > score(best[1]):
            best = (cfg, res)
    return best


def show(label, m):
    print("  %-28s accuracy %.4f  precision %.4f  recall %.4f  F1 %.4f  (tp %d fp %d fn %d tn %d)"
          % (label, m["accuracy"], m["precision"], m["recall"], m["f1"],
             m["tp"], m["fp"], m["fn"], m["tn"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=features.DEFAULT_DATA)
    args = parser.parse_args()
    t_start = time.perf_counter()
    addresses, x, y = features.load(args.data)
    train_idx, val_idx, test_idx = features.split(addresses)
    thresholds = features.fit_thresholds(x[train_idx])
    features.save_thresholds(thresholds, train_idx.shape[0])
    x0_train, y_train = features.quantise(x[train_idx], thresholds), y[train_idx]
    x0_val, y_val = features.quantise(x[val_idx], thresholds), y[val_idx]
    print("seed %d, numpy %s, sizes %s, train %d (%d flagged), validation %d (%d flagged), test %d"
          % (SEED, np.__version__, SIZES, train_idx.shape[0], int(y_train.sum()),
             val_idx.shape[0], int(y_val.sum()), test_idx.shape[0]))
    print("validation majority-class baseline: accuracy %.4f" % (1.0 - y_val.mean()))

    print("[1/3] float training and post-training quantisation, one candidate per class weight")
    best = None
    for index, class_weight in enumerate(CLASS_WEIGHTS):
        rng = np.random.default_rng(SEED + index)
        params = train_float(x0_train, y_train, class_weight, rng)
        cfg, val_int = ptq_search(params, x0_train, x0_val, y_val)
        val_float = float_metrics(params, x0_val, y_val)
        show("weight %.1f float, validation" % class_weight, val_float)
        show("weight %.1f integer, validation" % class_weight, val_int)
        if best is None or score(val_int) > score(best["val_int"]):
            best = {"class_weight": class_weight, "params": params, "cfg": cfg,
                    "val_int": val_int, "val_float": val_float}
    chosen = quantise(best["params"], best["cfg"])
    ref.check_model(chosen)
    print("  chosen: class weight %.1f, scales %s" % (best["class_weight"], list(best["cfg"])))

    print("[2/3] the test rows, scored once")
    x0_test, y_test = features.quantise(x[test_idx], thresholds), y[test_idx]
    test_float = float_metrics(best["params"], x0_test, y_test)
    test_int = int_metrics(chosen, x0_test, y_test)
    show("float, test", test_float)
    show("integer, test", test_int)
    print("  test majority-class baseline: accuracy %.4f" % test_int["majority_baseline"])

    print("[3/3] writing %s" % os.path.relpath(MODEL_PATH, os.path.join(HERE, "..", "..")))
    out = dict(chosen)
    out["float_accuracy"] = test_float["accuracy"]
    out["training"] = {
        "seed": SEED,
        "numpy": np.__version__,
        "method": "ptq",
        "float_epochs": EPOCHS,
        "class_weights_tried": CLASS_WEIGHTS,
        "class_weight": best["class_weight"],
        "selection": "integer F1 of the flagged class on the validation rows",
        "rows": {"train": int(train_idx.shape[0]), "validation": int(val_idx.shape[0]),
                 "test": int(test_idx.shape[0])},
        "validation": {"float": best["val_float"], "integer": best["val_int"]},
        "test": {"float": test_float, "integer": test_int},
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(ref.model_to_json(out).encode("utf-8"))
    acc_bits, out_bits = ref.frac_bits(chosen)
    for li, lay in enumerate(chosen["layers"]):
        wc = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        print("  layer %d: weight_frac_bits %d, shift %s, acc frac bits %d, out frac bits %s, "
              "worst-case |acc| %d (int32 max %d)"
              % (li + 1, lay["weight_frac_bits"], lay["shift"], acc_bits[li], out_bits[li],
                 wc, ref.INT32_MAX))
    print("RESULT integer test accuracy %.4f, precision %.4f, recall %.4f, baseline %.4f"
          % (test_int["accuracy"], test_int["precision"], test_int["recall"],
             test_int["majority_baseline"]))
    print("RESULT time: %.1f s" % (time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
