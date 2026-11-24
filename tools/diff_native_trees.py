"""Differential test: a native tree-ensemble classifier against the reference vectors.

Usage:
    python3 tools/diff_native_trees.py --ref DIR BINARY

The binary is a build of a tree workspace's `cli` ingot (tools/gen_trees_workspace.sh); DIR holds
inputs.bin, pred.bin and trace.jsonl from the pure-Python reference (tools/trees_vectors.py).
  1. class mode on all of inputs.bin: every printed class must equal pred.bin;
  2. trace mode on the samples covered by trace.jsonl: every tree's output, every class score and
     the class must equal the reference line.

Prints counts and wall times, and exits non-zero on any mismatch. Standard library only; it runs
the binary, so it needs the platform it was built for (Linux x86-64, in WSL here).
"""

import argparse
import json
import os
import subprocess
import sys
import time


def run(binary, arguments, data):
    start = time.perf_counter()
    done = subprocess.run([binary] + arguments, input=data, stdout=subprocess.PIPE)
    wall = time.perf_counter() - start
    if done.returncode != 0:
        raise SystemExit("%s %s: exit status %d" % (binary, " ".join(arguments), done.returncode))
    return done.stdout.decode("ascii"), wall


def parse_trace(text):
    """Trace mode output as a list of {"outputs": [..], "scores": [..], "class": c}."""
    samples = []
    current = {}
    for line in text.splitlines():
        fields = line.split()
        if fields[0] == "trees":
            current["outputs"] = [int(v) for v in fields[1:]]
        elif fields[0] == "scores":
            current["scores"] = [int(v) for v in fields[1:]]
        elif fields[0] == "class":
            current["class"] = int(fields[1])
            samples.append(current)
            current = {}
        else:
            raise SystemExit("trace: unexpected line %r" % line[:80])
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True)
    parser.add_argument("binary")
    args = parser.parse_args()

    def load(name):
        with open(os.path.join(args.ref, name), "rb") as f:
            return f.read()

    inputs, pred = load("inputs.bin"), load("pred.bin")
    reference = [json.loads(line) for line in load("trace.jsonl").decode("utf-8").splitlines()]
    if any(line["i"] != n for n, line in enumerate(reference)):
        raise SystemExit("trace.jsonl: samples are not numbered from 0 in order")
    sample_bytes = len(inputs) // len(pred)
    trees = len(reference[0]["outputs"])
    print("reference: %d samples of %d bytes, %d traced, %d trees" % (
        len(pred), sample_bytes, len(reference), trees))
    failures = 0

    text, wall = run(args.binary, [], inputs)
    classes = [int(line) for line in text.splitlines()]
    if len(classes) != len(pred):
        raise SystemExit("printed %d classes for %d samples" % (len(classes), len(pred)))
    wrong = [n for n, (got, want) in enumerate(zip(classes, pred)) if got != want]
    for n in wrong[:5]:
        print("  sample %d: class %d, reference %d" % (n, classes[n], pred[n]))
    print("classes: %d samples, %d mismatches, %.2f s" % (len(classes), len(wrong), wall))
    failures += len(wrong)

    text, wall = run(args.binary, ["trace"], inputs[:sample_bytes * len(reference)])
    samples = parse_trace(text)
    if len(samples) != len(reference):
        raise SystemExit("traced %d samples, expected %d" % (len(samples), len(reference)))
    mismatching = 0
    values = 0
    for n, (got, want) in enumerate(zip(samples, reference)):
        values += len(want["outputs"]) + len(want["scores"]) + 1
        if any(got.get(key) != want[key] for key in ("outputs", "scores", "class")):
            mismatching += 1
            if mismatching <= 5:
                print("  sample %d differs from trace.jsonl" % n)
    print("trace: %d samples (%d values), %d mismatching samples, %.2f s" % (
        len(samples), values, mismatching, wall))
    failures += mismatching
    print("diff_native_trees: %s" % ("PASS, zero mismatches" if failures == 0 else "FAIL"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
