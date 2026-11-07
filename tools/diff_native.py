"""Differential test: the native classifier against the NumPy reference vectors.

Usage:
    python3 tools/diff_native.py [--ref ref/out] BINARY [BINARY ...]

For every binary (a build of the `cli` ingot):
  1. class mode on all of inputs.bin: every printed class must equal pred.bin;
  2. trace mode on the samples covered by trace.jsonl: every hidden activation, every logit
     and the class must equal the reference line.

Prints counts and wall times, and exits non-zero on any mismatch. Standard library only. It runs
the binaries, so it needs the platform they were built for (Linux x86-64, in WSL here).
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(binary, arguments, data):
    """Run the binary with `data` on stdin. Returns (stdout text, wall seconds)."""
    start = time.perf_counter()
    done = subprocess.run([binary] + arguments, input=data, stdout=subprocess.PIPE)
    wall = time.perf_counter() - start
    if done.returncode != 0:
        raise SystemExit("%s %s: exit status %d" % (binary, " ".join(arguments), done.returncode))
    return done.stdout.decode("ascii"), wall


def check_classes(binary, inputs, pred):
    """Returns (samples, mismatches, wall seconds)."""
    text, wall = run(binary, [], inputs)
    classes = [int(line) for line in text.splitlines()]
    if len(classes) != len(pred):
        raise SystemExit("%s: printed %d classes for %d samples" % (binary, len(classes), len(pred)))
    wrong = [n for n, (got, want) in enumerate(zip(classes, pred)) if got != want]
    for n in wrong[:5]:
        print("  sample %d: class %d, reference %d" % (n, classes[n], pred[n]))
    return len(classes), len(wrong), wall


def parse_trace(text):
    """Trace mode output as a list of {"hidden": [[..]], "logits": [..], "class": c}."""
    samples = []
    hidden = []
    logits = None
    for line in text.splitlines():
        fields = line.split()
        if fields[0] == "hidden":
            if int(fields[1]) != len(hidden) + 1:
                raise SystemExit("trace: hidden layer %s out of order" % fields[1])
            hidden.append([int(v) for v in fields[2:]])
        elif fields[0] == "logits":
            logits = [int(v) for v in fields[1:]]
        elif fields[0] == "class":
            samples.append({"hidden": hidden, "logits": logits, "class": int(fields[1])})
            hidden = []
            logits = None
        else:
            raise SystemExit("trace: unexpected line %r" % line)
    return samples


def check_trace(binary, inputs, sample_bytes, reference):
    """Returns (samples, values compared, mismatching samples, wall seconds)."""
    text, wall = run(binary, ["trace"], inputs[:sample_bytes * len(reference)])
    samples = parse_trace(text)
    if len(samples) != len(reference):
        raise SystemExit("%s: traced %d samples, expected %d" % (binary, len(samples), len(reference)))
    wrong = 0
    values = 0
    for n, (got, want) in enumerate(zip(samples, reference)):
        values += sum(len(h) for h in want["hidden"]) + len(want["logits"]) + 1
        if any(got[key] != want[key] for key in ("hidden", "logits", "class")):
            wrong += 1
            if wrong <= 5:
                print("  sample %d differs from trace.jsonl" % n)
    return len(samples), values, wrong, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default=os.path.join(ROOT, "ref", "out"))
    parser.add_argument("binaries", nargs="+", metavar="BINARY")
    args = parser.parse_args()

    def load(name):
        with open(os.path.join(args.ref, name), "rb") as f:
            return f.read()

    inputs, pred, labels = load("inputs.bin"), load("pred.bin"), load("labels.bin")
    reference = [json.loads(line) for line in load("trace.jsonl").decode("utf-8").splitlines()]
    if any(line["i"] != n for n, line in enumerate(reference)):
        raise SystemExit("trace.jsonl: samples are not numbered from 0 in order")
    sample_bytes = len(inputs) // len(pred)
    correct = sum(1 for p, t in zip(pred, labels) if p == t)
    print("reference: %d samples of %d bytes, %d traced, accuracy %d of %d" % (
        len(pred), sample_bytes, len(reference), correct, len(labels)))

    failures = 0
    for binary in args.binaries:
        label = os.sep.join(os.path.normpath(binary).split(os.sep)[-2:])
        count, wrong, wall = check_classes(binary, inputs, pred)
        print("%s: classes %5d samples, %d mismatches, %.2f s" % (label, count, wrong, wall))
        failures += wrong
        count, values, wrong, wall = check_trace(binary, inputs, sample_bytes, reference)
        print("%s: trace   %5d samples (%d values), %d mismatching samples, %.2f s" % (
            label, count, values, wrong, wall))
        failures += wrong
    print("diff_native: %s" % ("PASS, zero mismatches" if failures == 0 else "FAIL"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
