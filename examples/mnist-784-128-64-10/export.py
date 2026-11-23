"""Export the quantised 784 -> 128 -> 64 -> 10 model and its test vectors to out/.

Usage:  python examples/mnist-784-128-64-10/export.py

This is ref/export.py run on this folder's model file: the same code writes model.json,
inputs.bin, labels.bin, pred.bin and trace.jsonl (docs/spec.md sections 5 and 6). inputs.bin and
labels.bin depend only on the MNIST test set, so they are byte for byte ref/out's and are not
committed a second time (.gitignore); check.sh copies them from ref/out when they are missing.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))

import export as base  # noqa: E402  (ref/export.py)


def main():
    t0 = time.perf_counter()
    base.MODEL_PATH = os.path.join(HERE, "model", "mnist-784-128-64-10.json")
    base.build(out_dir=os.path.join(HERE, "out"))
    print("export time %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
