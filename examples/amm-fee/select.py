"""The choices behind the fee-tier model, scored on the validation split only.

Usage:  python examples/amm-fee/select.py > examples/amm-fee/runs/select.txt      (NumPy, about a minute)

Float training of each candidate (input length, hidden sizes, learning rate) with train.py's own
loop, the best validation epoch of each. The test split is never built here. The candidate chosen
for train.py is the first line; the rule was "highest validation accuracy, and the smaller network
when two are within half a point".

Not recorded here: while the series was still downloading, the same kind of run was made on the
part that had arrived (validation split of that part only), also with inputs of 48 and 288 moves,
horizons of 12 and 36 moves, hidden layers 16-8 and two weight decays. Those runs set the list
below and the horizon of 24; their numbers were on other data and are not quoted anywhere.
"""

import contextlib
import importlib.util
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fee_train", os.path.join(HERE, "train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
ds = T.dataset

# (input moves, hidden sizes, learning rate)
CANDIDATES = [
    (144, [32, 16], 5e-4),
    (96, [32, 16], 5e-4),
    (192, [32, 16], 5e-4),
    (144, [64, 32], 5e-4),
    (144, [32, 16], 2e-3),
    (96, [32, 16], 2e-3),
]


def main():
    moves = ds.moves_of(ds.load_ticks())
    ranges = ds.split_ranges(len(moves))
    print("validation split only; seed %d, %d epochs, numpy %s" % (T.SEED, T.EPOCHS, np.__version__))
    print("%-6s %-10s %-7s %-9s %-22s %-22s %s" % ("input", "hidden", "lr", "samples", "float model correct",
                                                   "persistence correct", "kept epoch"))
    for n_obs, hidden, lr in CANDIDATES:
        ds.N_OBS, T.LR = n_obs, lr
        T.SIZES = [n_obs] + hidden + [ds.TIERS]
        rows = {name: ds.samples(moves, lo, hi) for name in ("train", "validation") for lo, hi in [ranges[name]]}
        thresholds = ds.quantile_thresholds([future for _, _, future in rows["train"]])
        x, y, _ = T.split_arrays(moves, rows["train"], thresholds)
        xv, yv, pv = T.split_arrays(moves, rows["validation"], thresholds)
        with contextlib.redirect_stdout(io.StringIO()):
            epoch, res, _ = T.train_float(x, y, xv, yv, np.random.default_rng(T.SEED))
        correct, persistence = int(round(res[0] * len(yv))), int((pv == yv).sum())
        print("%-6d %-10s %-7g %-9d %-22s %-22s %d"
              % (n_obs, "-".join(str(h) for h in hidden), lr, len(yv),
                 "%d (%.2f%%)" % (correct, 100.0 * correct / len(yv)),
                 "%d (%.2f%%)" % (persistence, 100.0 * persistence / len(yv)), epoch), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
