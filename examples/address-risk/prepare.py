"""Fetch the data set and write its 14 on-chain columns as data/onchain.npz for train.py.

Usage:  python examples/address-risk/prepare.py [--data DIR] [--no-download]

The only script here that needs more than NumPy: it reads a parquet file, so it needs pyarrow.
The file is pinned by revision and checked by SHA-256 before it is opened. Nothing it writes is
committed (data/ is in .gitignore).

Source: FuseChain Multimodal Ethereum Fraud Dataset, Nileshka Fernando, 2026, MIT licence,
https://huggingface.co/datasets/Nileshka/fusechain-data (see README.md in this folder).
"""

import argparse
import hashlib
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import features  # noqa: E402

REVISION = "ef382e837540182c483ea2c2fbb6259c42f0633a"
FILE = "address_level_fused.parquet"
URL = "https://huggingface.co/datasets/Nileshka/fusechain-data/resolve/%s/%s" % (REVISION, FILE)
SHA256 = "294bf2a663773e56d50cb90296cfb0b83f5067255ff61a97e850b0fbd58454c9"
SIZE = 7247469
ROWS = 35272


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=features.DEFAULT_DATA)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.data, exist_ok=True)
    path = os.path.join(args.data, FILE)
    if not os.path.exists(path):
        if args.no_download:
            raise SystemExit("prepare: %s is missing" % path)
        print("downloading %s" % URL)
        with urllib.request.urlopen(URL) as response, open(path + ".part", "wb") as f:
            f.write(response.read())
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != SIZE or digest != SHA256:
        raise SystemExit("prepare: %s is %d bytes with SHA-256 %s, expected %d and %s"
                         % (path, len(raw), digest, SIZE, SHA256))

    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["address", features.LABEL] + features.FEATURES)
    address = np.array(table.column("address").to_pylist(), dtype="U42")
    y = table.column(features.LABEL).to_numpy(zero_copy_only=False).astype(np.uint8)
    x = np.stack([table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
                  for name in features.FEATURES], axis=1)
    if x.shape != (ROWS, len(features.FEATURES)) or np.unique(address).shape[0] != ROWS:
        raise SystemExit("prepare: expected %d rows with distinct addresses" % ROWS)
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise SystemExit("prepare: a feature is not finite or a label is not 0 or 1")
    np.savez(os.path.join(args.data, "onchain.npz"), address=address, x=x, y=y,
             features=np.array(features.FEATURES))
    print("%s: SHA-256 ok, %d addresses, %d flagged (%.1f%%), %d features, wrote onchain.npz"
          % (FILE, ROWS, int(y.sum()), 100.0 * y.mean(), x.shape[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
