"""Write demo/generated/fee-assets.json: everything the fee-tier page needs, from real outputs.

Usage:
    python3 demo/fee_assets.py --fe-out DIR [--check] [--out FILE]

    DIR holds the `fe build` artifacts of examples/amm-fee (<Contract>.bin, <Contract>.runtime.bin).
    With --check nothing is written: the file is built in memory and must equal the committed one
    byte for byte, which proves that the bytecode the page embeds is what `fe build` produced.

What goes in, and where each part comes from:

    contracts   the creation bytecode of Arbiter, Network, Rows0, FeeOracle and TestPool    fe build
    model       shifts, rows, biases, Merkle proofs, root          examples/amm-fee/out/model.json, tools/gen_tree.py
    task        the input rule's constants, the tier thresholds    examples/amm-fee/dataset.py and model/*.json
    metrics     the recorded scores of the test split              examples/amm-fee/out/model.json
    ticks       the test split of the pool's tick series           examples/amm-fee/data/ticks.csv
    excerpts    quoted lines of the consumer's source              examples/amm-fee/ingots/consumer/src/lib.fe
    anvil       the gas check.sh recorded on anvil                 examples/amm-fee/runs/*.json

The page holds its own copy of the integer arithmetic (demo/src/fee-kernel.js) for the keeper and
the watcher; demo/test/fee.mjs checks it against examples/amm-fee/out and against the Fe bytecode.
Standard library only.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLE = os.path.join(ROOT, "examples", "amm-fee")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, EXAMPLE)

import dataset  # noqa: E402  (examples/amm-fee/dataset.py: the definition of the task)
from gen_tree import commitment  # noqa: E402

CONTRACTS = ["Arbiter", "Network", "Rows0", "FeeOracle", "TestPool"]
CONSUMER = "examples/amm-fee/ingots/consumer/src/lib.fe"


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
    "contract": (CONSUMER, "pub contract FeeOracle uses", "pub contract FeeOracle uses"),
    "apply": (CONSUMER, "let words = with (u256 = count)", "let tier: u8 = source.static"),
    "move": (CONSUMER, "pub fn clipped_move(", "^}"),
}


def fe_version():
    try:
        return subprocess.check_output(
            [os.path.expanduser("~/.fe/bin/fe"), "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(fe_out):
    with open(os.path.join(EXAMPLE, "out", "model.json"), "r", encoding="utf-8") as f:
        model = json.load(f)
    task = dataset.load_task()
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
                }
                for j, (row, bias) in enumerate(zip(layer["weights"], layer["bias"]))
            ],
        })

    # The test split of the series, whole: every tick a test sample's input or target touches.
    ticks = dataset.load_ticks()
    if dataset.sha256(dataset.TICKS_PATH) != task["ticks_sha256"]:
        raise SystemExit("fee_assets: data/ticks.csv is not the file the model was trained on")
    moves = dataset.moves_of(ticks)
    lo, hi = dataset.split_ranges(len(moves))["test"]
    windows = len(dataset.samples(moves, lo, hi))
    part = ticks[lo:hi + 1]

    excerpts = {}
    for key, (path, start, end) in EXCERPTS.items():
        if end == "^}":
            with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            first = next(i for i, line in enumerate(lines) if start in line)
            last = next(i for i in range(first, len(lines)) if lines[i] == "}")
            excerpts[key] = {"file": path, "line": first + 1, "code": "\n".join(lines[first:last + 1])}
        else:
            excerpts[key] = excerpt(path, start, end)

    return {
        "fe_version": fe_version(),
        "model": {
            "name": model["name"],
            "root": tree["root"],
            "leaf_count": tree["leaf_count"],
            "layers": layers,
        },
        "task": {
            "pool": task["pool"],
            "n_obs": dataset.N_OBS,
            "gap": dataset.GAP,
            "horizon": dataset.HORIZON,
            "clip": dataset.CLIP,
            "tiers": dataset.TIERS,
            "thresholds": task["thresholds"],
            "step_blocks": dataset.STEP,
            "ticks_sha256": task["ticks_sha256"],
            "series_ticks": len(ticks),
        },
        "metrics": model["metrics"],
        "ticks": {
            # Tick k of this slice is line `first_index + k` of data/ticks.csv, observed at the end
            # of block `first_block + step_blocks * k`. Stored as the first tick and the differences.
            "first_index": lo,
            "first_block": dataset.FIRST_BLOCK + dataset.STEP * lo,
            "count": len(part),
            "windows": windows,
            "start": part[0],
            "deltas": [part[i] - part[i - 1] for i in range(1, len(part))],
        },
        "contracts": {name: contract(fe_out, name) for name in CONTRACTS},
        "excerpts": excerpts,
        "anvil": anvil_records(),
        "fees": fees(),
    }


def fees():
    """The fee of each tier and the default, from the constants of the consumer's source."""
    with open(os.path.join(ROOT, CONSUMER), "r", encoding="utf-8") as f:
        source = f.read()
    value = {}
    for name in ("FEE_QUIET", "FEE_MIDDLE", "FEE_ACTIVE", "DEFAULT_FEE"):
        found = re.search(r"pub const %s: u32 = (\w+)" % name, source)
        if not found:
            raise SystemExit("fee_assets: no constant %s in %s" % (name, CONSUMER))
        value[name] = found.group(1)
    by_tier = [int(value[n]) for n in ("FEE_QUIET", "FEE_MIDDLE", "FEE_ACTIVE")]
    default = value["DEFAULT_FEE"]
    return {"by_tier": by_tier, "default": int(default) if default.isdigit() else int(value[default])}


def anvil_records():
    """What examples/amm-fee/check.sh measured on anvil (runs/*.json), for the page to set beside
    its own in-tab figures. demo/test/fee.mjs requires the tab to reproduce them exactly."""
    def read(name):
        with open(os.path.join(EXAMPLE, "runs", name), "r", encoding="utf-8") as f:
            return json.load(f)
    arbiter = read("arbiter-gas.json")
    oracle = read("oracle-gas.json")
    return {
        "fe": arbiter["fe"],
        "chain": arbiter["chain"],
        "classify": read("network-gas.json")["classify"]["gas_used"],
        "calls": {c["call"]: c["gas_used"] for c in arbiter["calls"]},
        "oracle": {k: oracle[k] for k in ("observe_first", "observe_median", "claim", "finalize", "apply", "apply_later")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fe-out", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", default=os.path.join(HERE, "generated", "fee-assets.json"))
    args = parser.parse_args()

    assets = build(args.fe_out)
    text = json.dumps(assets, separators=(",", ":")) + "\n"
    summary = ", ".join("%s %s" % (name, c["sha256"][:12]) for name, c in assets["contracts"].items())
    if args.check:
        with open(args.out, "r", encoding="utf-8", newline="") as f:
            committed = f.read()
        if committed != text:
            old = json.loads(committed)
            for name in CONTRACTS:
                if old["contracts"].get(name, {}).get("creation") != assets["contracts"][name]["creation"]:
                    print("WRONG: the embedded creation bytecode of %s is not what `fe build` produced" % name)
            print("WRONG: %s differs from what these sources and this compiler produce" % args.out)
            return 1
        print("%s equals what `fe build` (%s) and the committed data produce, byte for byte: %s"
              % (os.path.relpath(args.out, ROOT), assets["fe_version"], summary))
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote %s: %d bytes, %d ticks (%d windows), root %s, %s; %s" % (
        os.path.relpath(args.out, ROOT), len(text.encode("utf-8")), assets["ticks"]["count"],
        assets["ticks"]["windows"], assets["model"]["root"], assets["fe_version"], summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
