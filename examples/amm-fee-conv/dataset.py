"""The data set of the convolutional fee-tier example: examples/amm-fee's task with a 48-hour input.

Standard library only, integers only. check.sh runs this file to rebuild out/inputs.bin,
out/labels.bin and out/past.bin from ../amm-fee/data/ticks.csv; train.py and export.py import it.

Everything but the input length is examples/amm-fee/dataset.py, which this file loads and calls:
the tick series, the input rule (move[i] = min(|tick[i] - tick[i - 1]|, 127), one byte each), the
target (the sum of the HORIZON = 24 moves that start GAP = 6 intervals after the window), the
tiers and their thresholds, the time-ordered 70/15/15 split of the moves, and the scores.

What differs:

    input = the last N_OBS = 288 moves, oldest first (48 hours)      288 bytes

The windows are the dense example's windows. A sample is identified by `end`, the index one past
its last input move, and its label is the tier of moves[end + GAP : end + GAP + HORIZON]. The
samples of a split are exactly the dense example's samples of that split (the same `end`s, so the
same labels and the same persistence answers): its labels lie inside the split. The longer input
reaches up to 144 moves further back, which for the first 144 windows of the validation and test
splits means into the end of the previous split. Those moves are in the past at decision time,
the same moves a contract would have stored, and no label is taken from outside the split. In the
training split there is nothing before move 0, so its first 144 windows (whose input would start
before the series) are dropped: 36,475 training samples instead of 36,619.

The thresholds are the dense example's, read from its model file: the terciles of the target over
the dense example's training split, computed by its train.py from the training split alone. They
are reused, not recomputed, so that a window has the same label in both examples and the two
models are scored on the same 7,711 test windows with the same labels. `--check` also recomputes
the terciles over this example's own (slightly smaller) training split and prints them.

    python3 examples/amm-fee-conv/dataset.py            writes out/inputs.bin, out/labels.bin, out/past.bin
    python3 examples/amm-fee-conv/dataset.py --check    what check.sh runs
"""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DENSE = os.path.join(HERE, "..", "amm-fee")

_spec = importlib.util.spec_from_file_location("amm_fee_dataset", os.path.join(DENSE, "dataset.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

MODEL_NAME = "amm-fee-conv-288"
MODEL_PATH = os.path.join(HERE, "model", MODEL_NAME + ".json")
DENSE_MODEL_PATH = os.path.join(DENSE, "model", "amm-fee-144-32-16-3.json")
TICKS_PATH = base.TICKS_PATH

N_OBS = 288
DENSE_N_OBS = base.N_OBS
GAP, HORIZON, TIERS, CLIP = base.GAP, base.HORIZON, base.TIERS, base.CLIP

load_ticks = base.load_ticks
moves_of = base.moves_of
split_ranges = base.split_ranges
quantile_thresholds = base.quantile_thresholds
tier = base.tier
sha256 = base.sha256
metrics = base.metrics
describe = base.describe


def samples(moves, lo, hi):
    """The dense example's samples of moves[lo:hi] whose 288-move input starts at move 0 or later:
    a list of (end, past, future) (see examples/amm-fee/dataset.py `samples`)."""
    return [row for row in base.samples(moves, lo, hi) if row[0] >= N_OBS]


def input_bytes(moves, end):
    return bytes(moves[end - N_OBS:end])


def dense_thresholds():
    with open(DENSE_MODEL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["task"]["thresholds"]


def main():
    ticks = load_ticks()
    moves = moves_of(ticks)
    thresholds = dense_thresholds()
    lo, hi = split_ranges(len(moves))["test"]
    rows = samples(moves, lo, hi)
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    files = {
        "inputs.bin": b"".join(input_bytes(moves, end) for end, _, _ in rows),
        "labels.bin": bytes(tier(future, thresholds) for _, _, future in rows),
        "past.bin": bytes(tier(past, thresholds) for _, past, _ in rows),
    }
    for name, data in files.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
    print("%d ticks, %d moves, test split moves %d to %d: %d samples of %d bytes (the first reaches back "
          "to move %d), thresholds %s (the dense example's)"
          % (len(ticks), len(moves), lo, hi, len(rows), N_OBS, rows[0][0] - N_OBS, thresholds))
    return 0


def dense_on_the_same_windows(out_dir):
    """(correct, samples) of the dense 144-32-16-3 model of examples/amm-fee on this example's test
    windows, from its committed predictions and the labels here. The windows are the same, so this
    must be its own recorded score."""
    with open(os.path.join(DENSE, "out", "pred.bin"), "rb") as f:
        dense_pred = f.read()
    with open(os.path.join(out_dir, "labels.bin"), "rb") as f:
        labels = f.read()
    if len(dense_pred) != len(labels):
        raise SystemExit("dataset: the dense example's predictions do not line up with the samples here")
    return sum(1 for a, b in zip(dense_pred, labels) if a == b), len(labels)


def check():
    """check.sh: rebuild the files, require the labels and the persistence answers of the dense
    example's windows, and recompute every recorded score from the committed pred.bin."""
    main()
    out = os.path.join(HERE, "out")
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        task = json.load(f)["task"]
    if task["ticks_sha256"] != sha256(TICKS_PATH):
        print("WRONG: ../amm-fee/data/ticks.csv is not the file the model was trained on")
        return 1
    if task["thresholds"] != dense_thresholds():
        print("WRONG: the model file's thresholds are not the dense example's")
        return 1
    moves = moves_of(load_ticks())
    lo, hi = split_ranges(len(moves))["train"]
    dense_train = quantile_thresholds([future for _, _, future in base.samples(moves, lo, hi)])
    own_train = quantile_thresholds([future for _, _, future in samples(moves, lo, hi)])
    if dense_train != task["thresholds"]:
        print("WRONG: the dense example's training split quantiles are %s, the model file says %s"
              % (dense_train, task["thresholds"]))
        return 1
    # The dense example's labels and persistence answers, rebuilt by its own rules.
    lo, hi = split_ranges(len(moves))["test"]
    dense_rows = base.samples(moves, lo, hi)
    rows = samples(moves, lo, hi)
    if [r[0] for r in rows] != [r[0] for r in dense_rows]:
        print("WRONG: the test windows are not the dense example's")
        return 1
    with open(os.path.join(out, "model.json"), "r", encoding="utf-8") as f:
        recorded = json.load(f)["metrics"]
    computed = metrics(out)
    dense_correct, count = dense_on_the_same_windows(out)
    computed["dense_144_32_16_3_correct"] = dense_correct
    if computed != recorded:
        print("WRONG: the scores recomputed from pred.bin differ from out/model.json")
        return 1
    print("thresholds %s are the dense example's (its training split's terciles; this example's own "
          "training split gives %s); the %d test windows are the dense example's; the recorded scores "
          "equal the recomputed ones" % (task["thresholds"], own_train, count))
    print(describe(computed) + "; the dense 144-32-16-3 model on the same windows %d (%.2f%%)"
          % (dense_correct, 100.0 * dense_correct / count))
    return 0


if __name__ == "__main__":
    sys.exit(check() if sys.argv[1:] == ["--check"] else main())
