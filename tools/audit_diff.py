"""Differential test of the native auditor (P7 of docs/spec-dispute.md).

Usage:
    python3 tools/audit_diff.py CLI_BINARY [--samples 100] [--ref ref/out]

For each of the first SAMPLES traced reference samples, takes the honest claim (input, hidden
activations, logits, class from trace.jsonl) and derives every single-value mutation: each hidden
activation changed by one (staying in 0..127), each logit changed by one, and every other class.
Together with the honest claim itself, that is 1 + 32 + 10 + 9 claims per sample.

For every claim the expected verdict is computed here, in plain Python with the model file and the
rule of docs/spec-dispute.md section 6 (layer 1 on the input against the claimed hidden vector,
layer 2 on the CLAIMED hidden vector against the claimed logits, then the argmax of the CLAIMED
logits against the class). All claims are then piped through `cli audit` in one run and its lines
are compared with the expected ones. Prints the counts and every mismatch; exit status 1 on any.
Standard library only.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def neuron(row, x, bias):
    return bias + sum(w * v for w, v in zip(row, x))


def hidden_layer(layer, x):
    out = []
    for row, bias in zip(layer["weights"], layer["bias"]):
        acc = neuron(row, x, bias) >> layer["shift"]
        out.append(0 if acc < 0 else 127 if acc > 127 else acc)
    return out


def logits_layer(layer, x):
    return [neuron(row, x, bias) for row, bias in zip(layer["weights"], layer["bias"])]


def argmax(values):
    best = 0
    for j, v in enumerate(values):
        if v > values[best]:
            best = j
    return best


def verdict(model, x0, hidden, logits, cls):
    """The first failing check, in the auditor's output format."""
    recomputed = hidden_layer(model["layers"][0], x0)
    for j, (e, c) in enumerate(zip(recomputed, hidden)):
        if e != c:
            return "neuron 1 %d %d %d" % (j, e, c)
    recomputed = logits_layer(model["layers"][1], hidden)
    for j, (e, c) in enumerate(zip(recomputed, logits)):
        if e != c:
            return "neuron 2 %d %d %d" % (j, e, c)
    expected = argmax(logits)
    if expected != cls:
        return "class %d %d" % (expected, cls)
    return "ok"


def encode(x0, hidden, logits, cls):
    out = bytes(x0) + bytes(hidden)
    for v in logits:
        out += (v % (1 << 32)).to_bytes(4, "big")
    return out + bytes([cls])


def mutations(x0, hidden, logits, cls):
    yield x0, hidden, logits, cls
    for j in range(len(hidden)):
        changed = list(hidden)
        changed[j] = changed[j] - 1 if changed[j] == 127 else changed[j] + 1
        yield x0, changed, logits, cls
    for j in range(len(logits)):
        changed = list(logits)
        changed[j] += 1
        yield x0, hidden, changed, cls
    for other in range(10):
        if other != cls:
            yield x0, hidden, logits, other


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cli")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--ref", default=os.path.join(ROOT, "ref", "out"))
    args = parser.parse_args()

    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        model = json.load(f)
    n_in = model["layers"][0]["in"]
    with open(os.path.join(args.ref, "inputs.bin"), "rb") as f:
        inputs = f.read(n_in * args.samples)
    with open(os.path.join(args.ref, "trace.jsonl"), "r", encoding="utf-8") as f:
        traces = [json.loads(f.readline()) for _ in range(args.samples)]

    expected, stream = [], bytearray()
    for n, trace in enumerate(traces):
        x0 = list(inputs[n * n_in:(n + 1) * n_in])
        for claim in mutations(x0, trace["hidden"][0], trace["logits"], trace["class"]):
            expected.append(verdict(model, *claim))
            stream += encode(*claim)

    run = subprocess.run([args.cli, "audit"], input=bytes(stream), stdout=subprocess.PIPE, check=False)
    got = run.stdout.decode("ascii").splitlines()
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    if len(got) != len(expected):
        mismatches.append((len(got), "%d lines" % len(expected), "%d lines" % len(got)))
    for i, e, g in mismatches[:20]:
        print("claim %d: python says %r, auditor says %r" % (i, e, g))
    kinds = {k: sum(1 for e in expected if e.split()[0] == k) for k in ("ok", "neuron", "class")}
    print("audit_diff: %d claims over %d samples (%d ok, %d neuron, %d class), exit status %d, %d mismatch(es)"
          % (len(expected), args.samples, kinds["ok"], kinds["neuron"], kinds["class"], run.returncode,
             len(mismatches)))
    return 1 if mismatches or run.returncode != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
