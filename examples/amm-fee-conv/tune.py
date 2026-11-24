"""The training settings of the convolutional fee-tier model, scored on the validation split only.

Usage:  python examples/amm-fee-conv/tune.py > examples/amm-fee-conv/runs/tune.txt
        (NumPy; the candidates run in parallel, one process each, about 10 minutes)

The network's shape is fixed by train.py (ARCH): it was chosen for what this example demonstrates,
a convolution whose whole run on-chain does not fit one transaction, not searched for accuracy.
What is chosen here is how to train it: learning rate, weight decay and number of epochs. Float
training with train.py's own loop, the best validation epoch of each; the test split is never
built. The rule: the highest validation accuracy.

A first run of train.py with a learning rate of 2e-3 and a weight decay of 0.05 (16 of 30 epochs
seen, then stopped, not recorded) overfitted: training loss fell from 0.90 to 0.30 while
validation accuracy fell from 56% to 49%. That is why the candidates below are slower and more
regularised.
"""

import contextlib
import importlib.util
import io
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))

# (learning rate, weight decay, epochs)
CANDIDATES = [
    (5e-4, 0.05, 12),
    (2e-4, 0.05, 12),
    (5e-4, 0.5, 12),
    (2e-4, 0.5, 12),
    (1e-4, 0.05, 12),
    (1e-3, 0.5, 12),
]


def load_train():
    spec = importlib.util.spec_from_file_location("conv_train", os.path.join(HERE, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(candidate):
    import numpy as np
    T = load_train()
    lr, wd, epochs = candidate
    T.WEIGHT_DECAY = wd
    _, _, _, (x, y), (xv, yv, pv) = T.load_splits()
    with contextlib.redirect_stdout(io.StringIO()):
        epoch, res, _ = T.train_float(x, y, xv, yv, np.random.default_rng(T.SEED), epochs=epochs, lr=lr, log=False)
    return candidate, epoch, int(round(res[0] * len(yv))), res[1], int((pv == yv).sum()), len(yv)


def main():
    T = load_train()
    import numpy as np
    print("validation split only; seed %d, numpy %s, network %s" % (T.SEED, np.__version__, T.ARCH))
    print("%-8s %-6s %-7s %-22s %-10s %-22s %s" % ("lr", "decay", "epochs", "float model correct", "val loss",
                                                  "persistence correct", "kept epoch"))
    with Pool(len(CANDIDATES)) as pool:
        for (lr, wd, epochs), kept, correct, loss, persistence, n in pool.imap(run, CANDIDATES):
            print("%-8g %-6g %-7d %-22s %-10.4f %-22s %d"
                  % (lr, wd, epochs, "%d (%.2f%%)" % (correct, 100.0 * correct / n), loss,
                     "%d (%.2f%%)" % (persistence, 100.0 * persistence / n), kept), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
