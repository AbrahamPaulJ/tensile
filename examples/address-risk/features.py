"""The public-data side of the address-risk example: which columns are used, how the rows are
split, and the rule that turns a feature value into the 0..127 activation the network reads.
NumPy only. Imported by train.py and export.py.

The rule: every feature has 127 thresholds t_1 <= ... <= t_127, and its activation is the number
of thresholds that are <= the value, an integer 0..127. The thresholds are the k/128 quantiles
(k = 1..127, the "lower" rule, so each is a value that occurs in the data) of that feature over
the training split only. They are fixed numbers in features.json: anyone who holds an address's 14
feature values computes the same 14 bytes.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(HERE, "features.json")
DEFAULT_DATA = os.path.join(HERE, "data")

# The 14 on-chain columns of the data set, in the order the network reads them. The other 17
# columns (market, Reddit, Twitter) are not used: they are not properties of an address.
FEATURES = [
    "active_days",
    "normal_sent_cnt_mean",
    "normal_sent_cnt_max",
    "normal_recv_cnt_mean",
    "normal_recv_cnt_max",
    "normal_total_cnt_mean",
    "normal_total_cnt_max",
    "eth_sent_sum_mean",
    "eth_net_flow_mean",
    "eth_net_flow_max",
    "sessions_cnt_mean",
    "burst_max_tx_5m_mean",
    "burst_max_tx_5m_max",
    "days_since_last_activity_mean",
]
LABEL = "is_scam"
SPLIT_SEED = 20261124
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
LEVELS = 128  # activations are 0..127 (docs/spec.md section 2)


def load(data_dir):
    """The table prepare.py wrote: (addresses, x float64 (N, 14), y int64 (N,))."""
    with np.load(os.path.join(data_dir, "onchain.npz"), allow_pickle=False) as z:
        if [str(v) for v in z["features"]] != FEATURES:
            raise SystemExit("onchain.npz has other columns than features.py expects")
        return z["address"], z["x"].astype(np.float64), z["y"].astype(np.int64)


def split(addresses):
    """Row indices (train, validation, test). The rows are put in address order first, so the
    split does not depend on the order of the file, then shuffled with a fixed seed."""
    order = np.argsort(addresses, kind="stable")
    perm = order[np.random.default_rng(SPLIT_SEED).permutation(order.shape[0])]
    n = perm.shape[0]
    n_test = int(n * TEST_FRACTION)
    n_val = int(n * VALIDATION_FRACTION)
    return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]


def fit_thresholds(x_train):
    """(14, 127) thresholds from the training rows."""
    q = np.arange(1, LEVELS) / LEVELS
    return np.quantile(x_train, q, axis=0, method="lower").T.copy()


def quantise(x, thresholds):
    """Feature values (N, 14) to activations (N, 14), uint8 in 0..127."""
    out = np.empty(x.shape, dtype=np.uint8)
    for f in range(x.shape[1]):
        out[:, f] = np.searchsorted(thresholds[f], x[:, f], side="right")
    return out


def save_thresholds(thresholds, rows):
    doc = {
        "feature_rule": "activation = number of thresholds <= value (0..127)",
        "thresholds_from": "the k/128 quantiles, k = 1..127, numpy.quantile method 'lower', of the "
                           "%d training rows" % rows,
        "split_seed": SPLIT_SEED,
        "features": [{"name": name, "thresholds": [float(v) for v in thresholds[f]]}
                     for f, name in enumerate(FEATURES)],
    }
    with open(FEATURES_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def load_thresholds():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if [entry["name"] for entry in doc["features"]] != FEATURES:
        raise SystemExit("features.json has other features than features.py")
    return np.array([entry["thresholds"] for entry in doc["features"]], dtype=np.float64)


def metrics(pred, labels):
    """Counts and rates with class 1 (flagged) as the positive class."""
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"samples": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / n, "precision": precision, "recall": recall, "f1": f1,
            "majority_baseline": max(tp + fn, fp + tn) / n}
