"""The data set of the fee-tier example: from the pool's tick series to inputs and labels.

Standard library only, integers only. check.sh runs this file to rebuild out/inputs.bin and
out/labels.bin from data/ticks.csv; the committed predictions (out/pred.bin) are then checked
against those inputs by the native Fe classifier, and the recorded scores against those labels.
train.py and export.py import it. This file is the definition of the task: change a rule here and the
model, the vectors and the README must change with it.

    python3 examples/amm-fee/dataset.py            writes out/inputs.bin, out/labels.bin, out/past.bin

The series. data/ticks.csv has one line per observation: the pool's current tick (slot0) at the
end of block FIRST_BLOCK + STEP * i. STEP = 50 blocks is 10 minutes when no slot is missed.

The input rule (what a contract can compute from its own storage; ingots/consumer does):

    move[i] = min(|tick[i] - tick[i - 1]|, 127)          one byte, 0..127
    input   = the last N_OBS moves, oldest first          N_OBS bytes

The target. A window ending at observation `end` (its last move is move[end]) is used to choose
the fee for the HORIZON intervals that start GAP intervals later: the gap is the time the claim
spends in its dispute window. The realised activity of a stretch of moves is their sum (a sum of
absolute tick changes; the same clipped moves as the input, so a contract could label a past
window too):

    future = move[end + GAP + 1] + ... + move[end + GAP + HORIZON]
    past   = move[end - HORIZON + 1] + ... + move[end]

    tier(v) = number of thresholds t with v >= t         0 .. TIERS - 1

The thresholds are quantiles of `future` over the training split only (train.py computes them
once and stores them in the model file). The label is tier(future). The persistence baseline is
tier(past): "the next window looks like the window just seen".

The split is by time: the first 70% of the moves train, the next 15% validate, the last 15% test.
A sample belongs to a split only if all of it, from its first input move to its last future
move, lies inside the split, so no sample straddles a boundary.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS_PATH = os.path.join(HERE, "data", "ticks.csv")
MODEL_NAME = "amm-fee-144-32-16-3"
MODEL_PATH = os.path.join(HERE, "model", MODEL_NAME + ".json")

FIRST_BLOCK = 23_850_000
STEP = 50          # blocks between observations
CLIP = 127         # the largest input activation of the kernel (docs/spec.md section 2)
N_OBS = 144        # input moves: 24 hours, one daily cycle
GAP = 6            # intervals between the last input move and the first target move: 1 hour
HORIZON = 24       # target moves: 4 hours
TIERS = 3
TRAIN_SHARE, VAL_SHARE = 70, 15   # percent of the moves; the rest is the test split


def load_ticks(path=TICKS_PATH):
    """The ticks, in block order."""
    with open(path, "r", encoding="ascii") as f:
        return [int(line) for line in f if line.strip()]


def moves_of(ticks):
    """move[i] for i = 1 .. len(ticks) - 1, as a list indexed from 0 (entry k is move[k + 1])."""
    return [min(abs(ticks[i] - ticks[i - 1]), CLIP) for i in range(1, len(ticks))]


def split_ranges(count):
    """{"train": (lo, hi), ...}: half-open ranges of move indices."""
    a = count * TRAIN_SHARE // 100
    b = count * (TRAIN_SHARE + VAL_SHARE) // 100
    return {"train": (0, a), "validation": (a, b), "test": (b, count)}


def samples(moves, lo, hi):
    """Every sample inside moves[lo:hi]: a list of (end, past, future), where the input is
    moves[end - N_OBS:end], `past` the sum of its last HORIZON moves and `future` the sum of
    moves[end + GAP:end + GAP + HORIZON]. `end` is an index into `moves` (one past the input)."""
    prefix = [0]
    for value in moves:
        prefix.append(prefix[-1] + value)
    out = []
    for end in range(lo + N_OBS, hi - GAP - HORIZON + 1):
        past = prefix[end] - prefix[end - HORIZON]
        future = prefix[end + GAP + HORIZON] - prefix[end + GAP]
        out.append((end, past, future))
    return out


def quantile_thresholds(values, tiers=TIERS):
    """Integer thresholds that cut `values` into `tiers` parts of equal count (nearest rank)."""
    ordered = sorted(values)
    return [ordered[len(ordered) * k // tiers] for k in range(1, tiers)]


def tier(value, thresholds):
    return sum(1 for t in thresholds if value >= t)


def input_bytes(moves, end):
    return bytes(moves[end - N_OBS:end])


def load_task():
    """The thresholds train.py fixed, from the model file."""
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["task"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def metrics(out_dir):
    """Every score of the test split, as counts, from labels.bin, pred.bin and past.bin.

    "majority" is the best constant answer: the tier that is most common among the test labels
    themselves (chosen with knowledge of the test labels, so it is the hardest constant to beat).
    """
    def read(name):
        with open(os.path.join(out_dir, name), "rb") as f:
            return f.read()
    labels, pred, past = read("labels.bin"), read("pred.bin"), read("past.bin")
    if not (len(labels) == len(pred) == len(past)):
        raise SystemExit("dataset: labels.bin, pred.bin and past.bin differ in length")
    per_tier = []
    for t in range(TIERS):
        per_tier.append({
            "tier": t,
            "labels": sum(1 for v in labels if v == t),
            "model_correct": sum(1 for v, p in zip(labels, pred) if v == t and p == t),
            "persistence_correct": sum(1 for v, p in zip(labels, past) if v == t and p == t),
            "model_answers": sum(1 for p in pred if p == t),
        })
    confusion = [[sum(1 for v, p in zip(labels, pred) if v == a and p == b) for b in range(TIERS)]
                 for a in range(TIERS)]
    adjacent = sum(1 for v, p in zip(labels, pred) if abs(v - p) <= 1)
    return {
        "test_samples": len(labels),
        "model_correct": sum(row["model_correct"] for row in per_tier),
        "persistence_correct": sum(row["persistence_correct"] for row in per_tier),
        "majority_correct": max(row["labels"] for row in per_tier),
        "majority_tier": max(range(TIERS), key=lambda t: per_tier[t]["labels"]),
        "model_within_one_tier": adjacent,
        "persistence_within_one_tier": sum(1 for v, p in zip(labels, past) if abs(v - p) <= 1),
        "per_tier": per_tier,
        "confusion_label_by_answer": confusion,
    }


def describe(m):
    n = m["test_samples"]
    return ("test split, %d samples: model %d correct (%.2f%%), persistence %d (%.2f%%), majority tier %d (%.2f%%); "
            "recall per tier, model %s, persistence %s"
            % (n, m["model_correct"], 100.0 * m["model_correct"] / n,
               m["persistence_correct"], 100.0 * m["persistence_correct"] / n,
               m["majority_correct"], 100.0 * m["majority_correct"] / n,
               ["%d/%d" % (r["model_correct"], r["labels"]) for r in m["per_tier"]],
               ["%d/%d" % (r["persistence_correct"], r["labels"]) for r in m["per_tier"]]))


def check():
    """check.sh: rebuild the inputs and labels from the ticks, recompute the scores from the
    committed pred.bin, and require the record in out/model.json to say the same."""
    main()
    out = os.path.join(HERE, "out")
    with open(os.path.join(out, "model.json"), "r", encoding="utf-8") as f:
        recorded = json.load(f)["metrics"]
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        task = json.load(f)["task"]
    if task["ticks_sha256"] != sha256(TICKS_PATH):
        print("WRONG: data/ticks.csv is not the file the model was trained on")
        return 1
    moves = moves_of(load_ticks())
    lo, hi = split_ranges(len(moves))["train"]
    thresholds = quantile_thresholds([future for _, _, future in samples(moves, lo, hi)])
    if thresholds != task["thresholds"]:
        print("WRONG: the training split's quantiles are %s, the model file says %s" % (thresholds, task["thresholds"]))
        return 1
    computed = metrics(out)
    if computed != recorded:
        print("WRONG: the scores recomputed from pred.bin differ from out/model.json")
        return 1
    print("thresholds %s are the training split's quantiles; the recorded scores equal the recomputed ones" % thresholds)
    print(describe(computed))
    return 0


def main():
    ticks = load_ticks()
    moves = moves_of(ticks)
    task = load_task()
    thresholds = task["thresholds"]
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
    print("%d ticks (blocks %d to %d), %d moves, test split moves %d to %d: %d samples of %d bytes, "
          "thresholds %s, sha256 of data/ticks.csv %s"
          % (len(ticks), FIRST_BLOCK, FIRST_BLOCK + STEP * (len(ticks) - 1), len(moves), lo, hi,
             len(rows), N_OBS, thresholds, sha256(TICKS_PATH)[:16]))
    return 0


def sample_ticks(number):
    """The N_OBS + 1 ticks whose moves are the input of test sample `number`, oldest first."""
    ticks = load_ticks()
    lo, _ = split_ranges(len(ticks) - 1)["test"]
    return ticks[lo + number:lo + number + N_OBS + 1]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sample-ticks"]:
        print("\n".join(str(t) for t in sample_ticks(int(sys.argv[2]))))
        sys.exit(0)
    sys.exit(check() if sys.argv[1:] == ["--check"] else main())
