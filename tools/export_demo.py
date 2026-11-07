"""Write demo/generated/assets.json: everything the demo page needs, from real build outputs.

Usage:
    python3 tools/export_demo.py --fe-out DIR [--compiler-errors FILE] [--out FILE]

    DIR holds the `fe build` artifacts (<Contract>.bin). FILE (optional) is the JSON written by
    tools/build_demo.sh with the compiler's real diagnostics for a few compile-fail cases.

The page holds no copy of the network's logic: it runs the Fe bytecode on an EVM. What it needs from
here is the bytecode, the committed weights and Merkle proofs a challenger must present (the same
data as tools/claim_args.py), and some test images. Standard library only.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gen_tree import commitment  # noqa: E402

CONTRACTS = ["Arbiter", "Network", "NeuronCheck", "RowsLow", "RowsHigh"]
PER_DIGIT = 3
WRONG_SAMPLES = 2
N_IN = 784


def contract(fe_out, name):
    with open(os.path.join(fe_out, name + ".bin"), "r", encoding="ascii") as f:
        creation = f.read().strip()
    with open(os.path.join(fe_out, name + ".runtime.bin"), "r", encoding="ascii") as f:
        runtime = f.read().strip()
    return {
        "creation": creation,
        "runtime_bytes": len(runtime) // 2,
        "sha256": hashlib.sha256(bytes.fromhex(creation)).hexdigest(),
    }


def samples(ref):
    """The first few correctly classified images of each digit, and the first misclassified ones."""
    with open(os.path.join(ref, "inputs.bin"), "rb") as f:
        inputs = f.read()
    with open(os.path.join(ref, "labels.bin"), "rb") as f:
        labels = f.read()
    with open(os.path.join(ref, "pred.bin"), "rb") as f:
        pred = f.read()
    count = {d: 0 for d in range(10)}
    wrong = 0
    out = []
    for i, (label, predicted) in enumerate(zip(labels, pred)):
        if label == predicted:
            if count[label] >= PER_DIGIT:
                continue
            count[label] += 1
        else:
            if wrong >= WRONG_SAMPLES:
                continue
            wrong += 1
        out.append({
            "i": i,
            "label": label,
            "pred": predicted,
            "x0": inputs[i * N_IN:(i + 1) * N_IN].hex(),
        })
        if wrong >= WRONG_SAMPLES and all(c >= PER_DIGIT for c in count.values()):
            break
    return out


def excerpt(path, start, end):
    """The lines of `path` from the first one containing `start` through the next containing `end`,
    with their common indentation removed, so that the page quotes the source as it is."""
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    first = next(i for i, line in enumerate(lines) if start in line)
    last = next(i for i in range(first, len(lines)) if end in lines[i])
    chosen = lines[first:last + 1]
    indent = min(len(l) - len(l.lstrip()) for l in chosen if l.strip())
    return {"file": path, "line": first + 1, "code": "\n".join(l[indent:] for l in chosen)}


EXCERPTS = {
    "effects_claim": ("ingots/onchain/src/arbiter.fe", "Claim { x0, hidden, logits, class } -> u256", "uses (ctx, bond, window"),
    "effects_finalize": ("ingots/onchain/src/arbiter.fe", "Finalize { id } uses", "Finalize { id } uses"),
    "effects_withdraw": ("ingots/onchain/src/arbiter.fe", "Withdraw -> Wei uses", "Withdraw -> Wei uses"),
    "neuron": ("ingots/tensile/src/dense.fe", "pub fn neuron<", "where dot_fits_i32(IN)"),
    "step_assert": ("ingots/dispute/src/step.fe", "static_assert(rejects(", "))"),
}


def fe_version():
    try:
        return subprocess.check_output(
            [os.path.expanduser("~/.fe/bin/fe"), "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fe-out", required=True)
    parser.add_argument("--ref", default=os.path.join(ROOT, "ref", "out"))
    parser.add_argument("--compiler-errors", default=None)
    parser.add_argument("--out", default=os.path.join(ROOT, "demo", "generated", "assets.json"))
    args = parser.parse_args()

    with open(os.path.join(args.ref, "model.json"), "r", encoding="utf-8") as f:
        model = json.load(f)
    tree = commitment(model)
    by_position = {(x["layer"], x["neuron"]): x for x in tree["leaves"]}

    layers = []
    for index, layer in enumerate(model["layers"], start=1):
        layers.append({
            "in": layer["in"],
            "out": layer["out"],
            "shift": layer["shift"],
            "neurons": [
                {
                    "row": bytes(w & 0xFF for w in row).hex(),
                    "bias": bias,
                    "proof": by_position[(index, j)]["proof"],
                    "leaf": by_position[(index, j)]["leaf"],
                }
                for j, (row, bias) in enumerate(zip(layer["weights"], layer["bias"]))
            ],
        })

    errors = []
    if args.compiler_errors:
        with open(args.compiler_errors, "r", encoding="utf-8") as f:
            errors = json.load(f)

    assets = {
        "fe_version": fe_version(),
        "model": {
            "name": model["name"],
            "root": tree["root"],
            "leaf_count": tree["leaf_count"],
            "accuracy": model["accuracy"],
            "layers": layers,
        },
        "contracts": {name: contract(args.fe_out, name) for name in CONTRACTS},
        "samples": samples(args.ref),
        "compiler_errors": errors,
        "excerpts": {key: excerpt(*spec) for key, spec in EXCERPTS.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(assets, f, separators=(",", ":"))
        f.write("\n")
    print("wrote %s: %d bytes, %d samples, %d compiler diagnostics, root %s" % (
        args.out, os.path.getsize(args.out), len(assets["samples"]), len(errors), tree["root"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
