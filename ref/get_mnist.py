"""Download and verify the four MNIST idx files into ref/data/.

Usage:  python ref/get_mnist.py

The files are fetched gzip-compressed from one of two public mirrors, decompressed in memory,
checked against a recorded size and SHA-256 (of the decompressed idx content), and only then
written to disk. Files that are already present and pass the check are skipped.

The module is also imported by train.py and export.py for `load_images` / `load_labels`.
Standard library only, except NumPy for the two loaders.
"""

import gzip
import hashlib
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

# name -> (size in bytes of the decompressed idx file, SHA-256 of the decompressed idx file)
# Recorded 2026-11-05 from both mirrors (identical content; the .gz MD5s also equal the values
# torchvision ships for MNIST).
FILES = {
    "train-images-idx3-ubyte": (
        47040016,
        "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    ),
    "train-labels-idx1-ubyte": (
        60008,
        "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    ),
    "t10k-images-idx3-ubyte": (
        7840016,
        "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    ),
    "t10k-labels-idx1-ubyte": (
        10008,
        "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
    ),
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _check(name, data):
    """Return None if `data` is the expected content of `name`, else a reason string."""
    size, digest = FILES[name]
    if len(data) != size:
        return "size %d, expected %d" % (len(data), size)
    got = _sha256(data)
    if got != digest:
        return "sha256 %s, expected %s" % (got, digest)
    return None


def _present(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        return _check(name, f.read()) is None


def _download(name):
    errors = []
    for mirror in MIRRORS:
        url = mirror + name + ".gz"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tensile-ref/0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = gzip.decompress(resp.read())
            reason = _check(name, data)
            if reason is not None:
                raise ValueError("verification failed: " + reason)
            return data, url
        except Exception as exc:  # try the next mirror
            errors.append("%s: %r" % (url, exc))
    raise RuntimeError("could not fetch %s:\n  %s" % (name, "\n  ".join(errors)))


def ensure(verbose=True):
    """Make sure all four verified idx files exist in ref/data/. Returns the data directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        if _present(name):
            if verbose:
                print("ok       %s (present, verified)" % name)
            continue
        data, url = _download(name)
        path = os.path.join(DATA_DIR, name)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        if verbose:
            print("fetched  %s (%d bytes, sha256 %s) from %s" % (name, len(data), _sha256(data), url))
    return DATA_DIR


def _read_verified(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("%s is missing: run `python ref/get_mnist.py` first" % path)
    with open(path, "rb") as f:
        data = f.read()
    reason = _check(name, data)
    if reason is not None:
        raise ValueError("%s is corrupt (%s): delete it and re-run get_mnist.py" % (path, reason))
    return data


def load_images(split):
    """Return the raw pixels of `split` ("train" or "t10k") as a uint8 array of shape (N, 784)."""
    import numpy as np

    data = _read_verified("%s-images-idx3-ubyte" % split)
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    if magic != 2051 or rows != 28 or cols != 28 or len(data) != 16 + n * 784:
        raise ValueError("unexpected idx3 header in %s images" % split)
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, 784).copy()


def load_labels(split):
    """Return the labels of `split` ("train" or "t10k") as a uint8 array of shape (N,)."""
    import numpy as np

    data = _read_verified("%s-labels-idx1-ubyte" % split)
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    if magic != 2049 or len(data) != 8 + n:
        raise ValueError("unexpected idx1 header in %s labels" % split)
    return np.frombuffer(data, dtype=np.uint8, offset=8).copy()


def main():
    t0 = time.perf_counter()
    ensure(verbose=True)
    print("MNIST ready in ref/data (%.1f s)" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
