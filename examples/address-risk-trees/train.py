"""Train gradient-boosted trees on the address-risk data and quantise them to docs/spec-trees.md.

Usage:  python -I examples/address-risk-trees/train.py --data DIR      (after address-risk/prepare.py)

The data, the split, the feature rule and its thresholds are the dense example's, unchanged:
examples/address-risk/features.py and features.json. The test rows this script quantises must be,
byte for byte, examples/address-risk/out/inputs.bin and labels.bin; it stops if they are not.

  1. Gradient boosting in NumPy, written out here (no scikit-learn or LightGBM): logistic loss,
     complete trees of a fixed depth grown level by level on the 128 input levels, split rule
     "x[feature] <= threshold goes left", second-order leaf values, a fixed learning rate. There
     is no randomness: no row or feature sampling, ties broken by the lowest feature, then the
     lowest threshold.
  2. For every depth in DEPTHS and class weight in CLASS_WEIGHTS, one ensemble of the largest
     size in TREE_COUNTS; every prefix in TREE_COUNTS is a candidate.
  3. Each candidate is quantised (leaf values to int16 with the most fractional bits that fit)
     and scored as integers on the validation rows. The best F1 of the flagged class (then
     accuracy, then fewer trees, then smaller depth) is written to model/. Only then are the test
     rows scored, once.

Every choice is made on validation rows. Deterministic on one machine (floating point sums).
"""

import argparse
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DENSE = os.path.join(ROOT, "examples", "address-risk")
sys.path.insert(0, DENSE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import features  # noqa: E402  (examples/address-risk/features.py)
import trees_ref  # noqa: E402

MODEL_DIR = os.path.join(HERE, "model")
DEPTHS = [4, 5, 6]
TREE_COUNTS = [100, 200, 300]
CLASS_WEIGHTS = [1.0, 1.5, 2.0, 3.0]
LEARNING_RATE = 0.1
L2 = 1.0  # lambda in the leaf value -G / (H + lambda) and in the gain
MIN_CHILD_HESSIAN = 1.0  # a split needs at least this hessian on both sides
LEVELS = 128  # inputs are 0..127
MAX_LEAF_FRAC_BITS = 14


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def grow_tree(x, g, h, depth):
    """One complete tree of `depth` levels. Returns (features, thresholds, leaf_of_row, leaf_g,
    leaf_h): nodes in heap order, the leaf each training row reaches, and the gradient and hessian
    sums per leaf."""
    n, n_features = x.shape
    nodes = (1 << depth) - 1
    feat = np.zeros(nodes, dtype=np.int64)
    thr = np.full(nodes, LEVELS - 1, dtype=np.int64)  # a node that does not split sends all left
    pos = np.zeros(n, dtype=np.int64)  # position of each row within its level
    for level in range(depth):
        width = 1 << level
        first = width - 1
        best_gain = np.zeros(width)
        best_feat = np.zeros(width, dtype=np.int64)
        best_thr = np.full(width, LEVELS - 1, dtype=np.int64)
        g_node = np.bincount(pos, weights=g, minlength=width)
        h_node = np.bincount(pos, weights=h, minlength=width)
        parent = g_node ** 2 / (h_node + L2)
        found = np.zeros(width, dtype=bool)
        for f in range(n_features):
            key = pos * LEVELS + x[:, f]
            gh = np.bincount(key, weights=g, minlength=width * LEVELS).reshape(width, LEVELS)
            hh = np.bincount(key, weights=h, minlength=width * LEVELS).reshape(width, LEVELS)
            gl = np.cumsum(gh, axis=1)[:, :LEVELS - 1]  # bins <= t, t = 0..126
            hl = np.cumsum(hh, axis=1)[:, :LEVELS - 1]
            gr = g_node[:, None] - gl
            hr = h_node[:, None] - hl
            gain = gl ** 2 / (hl + L2) + gr ** 2 / (hr + L2) - parent[:, None]
            valid = (hl >= MIN_CHILD_HESSIAN) & (hr >= MIN_CHILD_HESSIAN)
            gain = np.where(valid, gain, -np.inf)
            t = np.argmax(gain, axis=1)  # first maximum: the lowest threshold
            g_best = gain[np.arange(width), t]
            better = g_best > best_gain + 1e-12 * (1.0 + np.abs(best_gain))  # a later feature must win outright
            better &= g_best > 0
            best_gain = np.where(better, g_best, best_gain)
            best_feat = np.where(better, f, best_feat)
            best_thr = np.where(better, t, best_thr)
            found |= better
        feat[first:first + width] = np.where(found, best_feat, 0)
        thr[first:first + width] = np.where(found, best_thr, LEVELS - 1)
        go_right = x[np.arange(n), feat[first + pos]] > thr[first + pos]
        pos = 2 * pos + go_right.astype(np.int64)
    leaves = 1 << depth
    leaf_g = np.bincount(pos, weights=g, minlength=leaves)
    leaf_h = np.bincount(pos, weights=h, minlength=leaves)
    return feat, thr, pos, leaf_g, leaf_h


def train(x, y, depth, rounds, class_weight):
    """Gradient boosting with logistic loss and weight `class_weight` on the flagged rows.
    Returns (base, trees) with trees = [(features, thresholds, leaf values)], values already
    multiplied by the learning rate."""
    w = np.where(y == 1, class_weight, 1.0)
    prior = float((w * y).sum() / w.sum())
    base = float(np.log(prior / (1.0 - prior)))
    margin = np.full(y.shape[0], base)
    trees = []
    for _ in range(rounds):
        p = sigmoid(margin)
        g = w * (p - y)
        h = w * p * (1.0 - p)
        feat, thr, leaf, leaf_g, leaf_h = grow_tree(x, g, h, depth)
        values = -leaf_g / (leaf_h + L2) * LEARNING_RATE
        trees.append((feat, thr, values))
        margin += values[leaf]
    return base, trees


def leaves_of(trees, x, depth):
    """(rows, trees) matrix of the leaf each row reaches in each tree."""
    n = x.shape[0]
    out = np.empty((n, len(trees)), dtype=np.int64)
    rows = np.arange(n)
    for t, (feat, thr, _) in enumerate(trees):
        i = np.zeros(n, dtype=np.int64)
        for _ in range(depth):
            i = 2 * i + 1 + (x[rows, feat[i]] > thr[i]).astype(np.int64)
        out[:, t] = i - ((1 << depth) - 1)
    return out


def quantise(base, trees):
    """(leaf_frac_bits, integer base, integer leaf arrays): the most fractional bits, up to
    MAX_LEAF_FRAC_BITS, at which every leaf fits an int16 and the base fits BASE_LIMIT."""
    largest = max(float(np.abs(v).max()) for _, _, v in trees)
    lf = MAX_LEAF_FRAC_BITS
    while lf > 0 and (largest * 2.0 ** lf > trees_ref.I16_MAX
                      or abs(base) * 2.0 ** lf > trees_ref.BASE_LIMIT):
        lf -= 1
    scale = 2.0 ** lf
    leaves = [np.rint(v * scale).astype(np.int64) for _, _, v in trees]
    return lf, int(np.rint(base * scale)), leaves


def int_classes(base_int, leaves, leaf_matrix):
    """The integer model's class: 1 if base + sum of leaves > 0 (two classes, docs/spec-trees.md
    section 4: class 0's score is 0 and the lowest index wins a tie)."""
    margin = np.full(leaf_matrix.shape[0], base_int, dtype=np.int64)
    for t, values in enumerate(leaves):
        margin += values[leaf_matrix[:, t]]
    return (margin > 0).astype(np.int64)


def float_classes(base, trees, leaf_matrix):
    margin = np.full(leaf_matrix.shape[0], base)
    for t, (_, _, values) in enumerate(trees):
        margin += values[leaf_matrix[:, t]]
    return (margin > 0.0).astype(np.int64)


def score(m):
    return (m["f1"], m["accuracy"])


def show(label, m):
    print("  %-36s accuracy %.4f  precision %.4f  recall %.4f  F1 %.4f  (tp %d fp %d fn %d tn %d)"
          % (label, m["accuracy"], m["precision"], m["recall"], m["f1"],
             m["tp"], m["fp"], m["fn"], m["tn"]), flush=True)


def to_model(name, depth, lf, base_int, trees, leaves):
    return {
        "name": name,
        "format": trees_ref.FORMAT,
        "in": len(features.FEATURES),
        "input_frac_bits": 7,
        "depth": depth,
        "classes": 2,
        "leaf_frac_bits": lf,
        "base": [0, base_int],
        "trees": [{"feature": [int(v) for v in feat], "threshold": [int(v) for v in thr],
                   "leaf": [int(v) for v in values]}
                  for (feat, thr, _), values in zip(trees, leaves)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=features.DEFAULT_DATA)
    args = parser.parse_args()
    t_start = time.perf_counter()
    addresses, x, y = features.load(args.data)
    train_idx, val_idx, test_idx = features.split(addresses)
    thresholds = features.load_thresholds()
    if not np.array_equal(thresholds, features.fit_thresholds(x[train_idx])):
        raise SystemExit("train: features.json is not the thresholds of these training rows")
    x_train, y_train = features.quantise(x[train_idx], thresholds).astype(np.int64), y[train_idx]
    x_val, y_val = features.quantise(x[val_idx], thresholds).astype(np.int64), y[val_idx]
    print("numpy %s; train %d (%d flagged), validation %d (%d flagged), test %d"
          % (np.__version__, train_idx.shape[0], int(y_train.sum()), val_idx.shape[0],
             int(y_val.sum()), test_idx.shape[0]))
    print("learning rate %g, lambda %g, min child hessian %g; depths %s, trees %s, class weights %s"
          % (LEARNING_RATE, L2, MIN_CHILD_HESSIAN, DEPTHS, TREE_COUNTS, CLASS_WEIGHTS))

    print("[1/3] boosting, then every candidate quantised and scored as integers on validation")
    candidates = []
    best = None
    for depth in DEPTHS:
        for class_weight in CLASS_WEIGHTS:
            t0 = time.perf_counter()
            base, trees = train(x_train, y_train, depth, max(TREE_COUNTS), class_weight)
            val_leaves = leaves_of(trees, x_val, depth)
            for count in TREE_COUNTS:
                lf, base_int, leaves = quantise(base, trees[:count])
                m_int = features.metrics(int_classes(base_int, leaves, val_leaves[:, :count]), y_val)
                m_float = features.metrics(float_classes(base, trees[:count], val_leaves[:, :count]), y_val)
                row = {"depth": depth, "trees": count, "class_weight": class_weight,
                       "leaf_frac_bits": lf, "validation_integer": m_int, "validation_float": m_float}
                candidates.append(row)
                show("depth %d, %3d trees, weight %.1f, int" % (depth, count, class_weight), m_int)
                key = (score(m_int), -count, -depth)
                if best is None or key > best[0]:
                    best = (key, row, base, trees[:count])
            print("    (%.1f s)" % (time.perf_counter() - t0), flush=True)
    _, row, base, trees = best
    depth, count = row["depth"], row["trees"]
    lf, base_int, leaves = quantise(base, trees)
    name = "address-risk-trees-%dx%d" % (count, depth)
    print("  chosen: %s, class weight %.1f, leaf_frac_bits %d" % (name, row["class_weight"], lf))

    print("[2/3] the test rows, scored once")
    x_test, y_test = features.quantise(x[test_idx], thresholds), y[test_idx]
    with open(os.path.join(DENSE, "out", "inputs.bin"), "rb") as f:
        dense_inputs = f.read()
    with open(os.path.join(DENSE, "out", "labels.bin"), "rb") as f:
        dense_labels = f.read()
    if x_test.astype(np.uint8).tobytes() != dense_inputs or y_test.astype(np.uint8).tobytes() != dense_labels:
        raise SystemExit("train: the test rows differ from examples/address-risk/out/")
    x_test = x_test.astype(np.int64)
    test_leaves = leaves_of(trees, x_test, depth)
    m_test_int = features.metrics(int_classes(base_int, leaves, test_leaves), y_test)
    m_test_float = features.metrics(float_classes(base, trees, test_leaves), y_test)
    show("float, test", m_test_float)
    show("integer, test", m_test_int)

    print("[3/3] writing the model")
    model = to_model(name, depth, lf, base_int, trees, leaves)
    model["worst_case_abs_score"] = trees_ref.worst_case(model)
    trees_ref.check_model(model)
    model["training"] = {
        "method": "gradient boosting, logistic loss, NumPy (train.py)",
        "numpy": np.__version__,
        "learning_rate": LEARNING_RATE, "lambda": L2, "min_child_hessian": MIN_CHILD_HESSIAN,
        "depths_tried": DEPTHS, "tree_counts_tried": TREE_COUNTS, "class_weights_tried": CLASS_WEIGHTS,
        "class_weight": row["class_weight"],
        "selection": "integer F1 of the flagged class on the validation rows",
        "rows": {"train": int(train_idx.shape[0]), "validation": int(val_idx.shape[0]),
                 "test": int(test_idx.shape[0])},
        "validation": {"float": row["validation_float"], "integer": row["validation_integer"]},
        "test": {"float": m_test_float, "integer": m_test_int},
        "candidates": [{k: c[k] for k in ("depth", "trees", "class_weight", "leaf_frac_bits")}
                       | {"validation_f1": round(c["validation_integer"]["f1"], 6),
                          "validation_accuracy": round(c["validation_integer"]["accuracy"], 6)}
                       for c in candidates],
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    for old in os.listdir(MODEL_DIR):
        if old.endswith(".json"):
            os.remove(os.path.join(MODEL_DIR, old))
    path = os.path.join(MODEL_DIR, name + ".json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(trees_ref.model_to_json(model))
    print("  wrote %s: %d trees of depth %d, %d parameters, worst-case |score| %d"
          % (os.path.relpath(path, ROOT), count, depth,
             count * ((1 << depth) - 1) * 2 + count * (1 << depth), model["worst_case_abs_score"]))
    print("RESULT integer test accuracy %.4f, precision %.4f, recall %.4f, F1 %.4f, baseline %.4f"
          % (m_test_int["accuracy"], m_test_int["precision"], m_test_int["recall"], m_test_int["f1"],
             m_test_int["majority_baseline"]))
    print("RESULT time: %.1f s" % (time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
