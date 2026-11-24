"""Write RESULTS.md from the records check.sh leaves in runs/. Standard library only.

Usage:  python3 examples/amm-fee-conv/results.py      (check.sh runs it last)

Every figure in RESULTS.md is read from a file a script wrote:
    out/model.json ("metrics"), model/*.json ("task", "training")      export.py, train.py
    runs/arbiter-gas.json, runs/arbiter-gas-flat.json                  this run, on anvil (two bindings)
    runs/network-gas.json                                              this run: Osaka's refusal, then Prague
    runs/sizes.json, runs/native.txt, runs/steps.tsv                   sizes, native differential, timings
    ../amm-fee/runs/*.json, ../amm-fee/out/model.json                  the dense model of the same task, as committed
Nothing is typed in by hand; the sentences are templates filled from the same numbers, and the
comparisons ("beats", "more", "less") are computed, not assumed.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DENSE = os.path.join(ROOT, "examples", "amm-fee")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import dataset  # noqa: E402
from gen_tree import commitment  # noqa: E402

CAP = 16777216  # the Osaka limit of one transaction (EIP-7825)


def load(*parts):
    with open(os.path.join(*parts), "r", encoding="utf-8") as f:
        return json.load(f)


def n(value):
    return "{:,}".format(value)


def pct(part, whole):
    return "%.1f%%" % (100.0 * part / whole)


def share(part, whole):
    return "%s of %s (%.2f%%)" % (n(part), n(whole), 100.0 * part / whole)


def calls_by_name(record):
    return {call["call"]: call for call in record["calls"]}


def find(record, prefix):
    """The first call whose name starts with `prefix`."""
    return next(call for call in record["calls"] if call["call"].startswith(prefix))


def challenges(record, later):
    """The challenge rows of one pass (the first slashes, or the later ones), in layer order."""
    rows = []
    for call in record["calls"]:
        name = call["call"]
        if later != name.startswith("later "):
            continue
        bare = name[len("later "):] if later else name
        if bare.startswith("challenge"):
            rows.append((bare, call))
    return rows


def params(model):
    total = 0
    for layer in model["layers"]:
        total += sum(len(row) for row in layer["weights"]) + len(layer["bias"])
    return total


def macs(model):
    """Multiply-accumulates of one forward pass."""
    total = 0
    for layer in model["layers"]:
        if layer.get("kind") == "conv1d":
            total += layer["out"] * layer["kernel"] * layer["channels"]
        else:
            total += layer["in"] * layer["out"]
    return total


def describe_layer(layer):
    if layer.get("kind") == "conv1d":
        c, k, s = layer["channels"], layer["kernel"], layer["stride"]
        length = layer["in"] // c
        positions = (length - k) // s + 1
        filters = len(layer["weights"])
        return ("conv1d: %d positions x %d channels, %d filters of %d x %d, stride %d -> %d positions x %d"
                % (length, c, filters, k, c, s, positions, filters))
    return "dense: %d -> %d" % (layer["in"], layer["out"])


def main():
    model = load(HERE, "out", "model.json")
    trained = load(dataset.MODEL_PATH)
    task, training, m = trained["task"], trained["training"], model["metrics"]
    tree_rec = load(HERE, "runs", "arbiter-gas.json")
    flat_rec = load(HERE, "runs", "arbiter-gas-flat.json")
    network = load(HERE, "runs", "network-gas.json")
    sizes = load(HERE, "runs", "sizes.json")
    dense_model = load(DENSE, "out", "model.json")
    dense_arbiter = load(DENSE, "runs", "arbiter-gas.json")
    dense_network = load(DENSE, "runs", "network-gas.json")
    tree = commitment(model)
    with open(os.path.join(HERE, "runs", "steps.tsv"), "r", encoding="utf-8") as f:
        steps = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    with open(os.path.join(HERE, "runs", "native.txt"), "r", encoding="utf-8") as f:
        native = [line.rstrip("\n") for line in f if line.strip()]
    total = next(row for row in steps if row[0] == "total")
    tests = m["test_samples"]
    dense_correct = m["dense_144_32_16_3_correct"]
    tc = calls_by_name(tree_rec)
    fc = calls_by_name(flat_rec)
    whole = network["classify"]["gas_used"]
    claim = tc["claim"]["gas_used"]
    finalize = tc["finalize"]["gas_used"]
    honest = claim + finalize
    first = challenges(tree_rec, later=False)
    later = challenges(tree_rec, later=True)
    dearest = max(call["gas_used"] for _, call in first)
    dearest_later = max(call["gas_used"] for _, call in later)
    dense_later = challenges(dense_arbiter, later=True)
    dense_cheapest = min(call["gas_used"] for _, call in dense_later)
    dense_cheapest_name = min(dense_later, key=lambda row: row[1]["gas_used"])[0]
    dense_honest = calls_by_name(dense_arbiter)["claim"]["gas_used"] + calls_by_name(dense_arbiter)["finalize"]["gas_used"]

    out = []
    add = out.append
    add("# Results: the fee tier with a convolutional front end, %d moves in" % model["layers"][0]["in"])
    add("")
    add("Generated by `check.sh` (its last step, `results.py`) from the records in `runs/`, `out/` and `model/`.")
    add("Change the scripts or the code, never this file. Gas is from real transactions on anvil, test sample 0,")
    add("each result checked against the NumPy reference or the protocol before its gas is recorded.")
    add("%s; whole `check.sh` run: %s s." % (total[2], total[1]))
    add("")
    add("## 1. The task and the model")
    add("")
    add("The task, the data, the labels, the thresholds and the test windows are those of `examples/amm-fee`")
    add("(`dataset.py` here loads that example's `dataset.py` and changes only the input length). Only the input is")
    add("longer: the last %d moves (48 hours) instead of %d (24 hours)." % (task["n_obs"], dense_model["layers"][0]["in"]))
    add("")
    add("| | |")
    add("|---|---|")
    add("| Data | %s, `../amm-fee/data/ticks.csv` (%s ticks, sha256 `%s`) |" % (task["pool"], n(task["ticks"]), task["ticks_sha256"]))
    add("| Input | the last %d moves, each `min(\\|tick - previous tick\\|, %d)`, one byte each, oldest first |" % (task["n_obs"], task["clip"]))
    add("| Target | the tier of the sum of the %d moves that start %d moves after the input ends |" % (task["horizon"], task["gap"]))
    add("| Tiers | sum below %s: tier 0; below %s: tier 1; else tier 2 (%s) |" % (n(task["thresholds"][0]), n(task["thresholds"][1]), task["thresholds_from"]))
    add("| Split, in time order | training moves %s, validation %s, test %s |" % tuple(
        "%s to %s" % (n(lo), n(hi)) for lo, hi in (task["split_moves"][k] for k in ("train", "validation", "test"))))
    add("| Training | `train.py`: seed %d, NumPy %s, %d float epochs (epoch %d kept, by validation), post-training "
        "quantisation with a scale search on validation, %s training samples |" % (
            training["seed"], training["numpy"], training["float_epochs"], training["kept_epoch"], n(training["train_samples"])))
    add("| Parameters | %s (int8 weights and int32 biases) |" % n(params(model)))
    add("| Model root (what the arbiter is deployed with) | `%s`, %d leaves: one per filter and per dense neuron |" % (tree["root"], tree["leaf_count"]))
    add("")
    add("| Layer | Shape | Outputs | Weight fractional bits | Shift | Worst-case accumulator, proved by a `static_assert` |")
    add("|---|---|---|---|---|---|")
    for index, layer in enumerate(model["layers"], start=1):
        add("| %d | %s | %s | %d | %s | %s |" % (
            index, describe_layer(layer), n(layer["out"]), layer["weight_frac_bits"],
            "none (logits)" if layer["shift"] is None else layer["shift"], n(layer["worst_case_abs_acc"])))
    add("")
    add("Training split: the dense example's training windows whose %d-move input starts at move 0 or later (the"
        % task["n_obs"])
    add("first 144 need moves before the series). Validation and test windows are exactly the dense example's: the")
    add("longer input of the first 144 reaches into the previous split's moves, which are in the past at decision")
    add("time; no label comes from outside its split.")
    add("")
    add("## 2. Scores on the held-out test split")
    add("")
    add("Scored once, by `export.py`; nothing was chosen on it. `check.sh` (step 1) rebuilds the labels from the ticks")
    add("and recomputes every count from the committed predictions, including the dense model's, from")
    add("`../amm-fee/out/pred.bin`. Consecutive windows overlap, so differences of a few tens of samples are noise.")
    add("")
    add("| Predictor, same %s test windows | Correct tier |" % n(tests))
    add("|---|---|")
    add("| **This model** (convolutional, %d moves, what the arbiter here enforces) | %s |" % (task["n_obs"], share(m["model_correct"], tests)))
    add("| The dense 144-32-16-3 model of `examples/amm-fee` (144 moves) | %s |" % share(dense_correct, tests))
    add("| Persistence: the tier of the last %d moves | %s |" % (task["horizon"], share(m["persistence_correct"], tests)))
    add("| Majority: always tier %d, the most common tier of the test split itself | %s |" % (m["majority_tier"], share(m["majority_correct"], tests)))
    add("")
    add("| Tier | Test samples | Model recall | Persistence recall | Times the model answered it |")
    add("|---|---|---|---|---|")
    for row in m["per_tier"]:
        add("| %d | %s | %s | %s | %s |" % (row["tier"], n(row["labels"]), share(row["model_correct"], row["labels"]),
                                            share(row["persistence_correct"], row["labels"]), n(row["model_answers"])))
    add("")
    diff = m["model_correct"] - dense_correct
    v = training["validation"]
    verdict = ("**It beats the dense model**" if diff > 0 else
               "**It does not beat the dense model**" if diff < 0 else "**It ties the dense model**")
    add("- %s: %+d samples (%+.2f percentage points) on the same windows%s"
        % (verdict, diff, 100.0 * diff / tests,
           ", a difference of the size this test cannot tell from noise." if abs(diff) <= 50 else "."))
    if diff <= 0:
        add("  Twice the history and three convolutions did not help on this task. This example is the demonstration")
        add("  of the mechanism (a convolutional model, its arbiter and its dispute), not a better fee model.")
    add("- It beats persistence by %+d samples and the best constant by %+d."
        % (m["model_correct"] - m["persistence_correct"], m["model_correct"] - m["majority_correct"]))
    add("- On the validation split, which chose the epoch and the scales: model %s, persistence %s, the"
        % (share(v["integer_correct"], v["samples"]), share(v["persistence_correct"], v["samples"])))
    add("  training split's majority tier %s; the float model before quantisation %s."
        % (share(v["majority_correct"], v["samples"]), share(v["float_correct"], v["samples"])))
    weakest = min(m["per_tier"], key=lambda row: row["model_correct"] / max(row["labels"], 1))
    add("- Unflattering: the model's weakest tier is tier %d, right for %s."
        % (weakest["tier"], share(weakest["model_correct"], weakest["labels"])))
    add("")
    add("## 3. Gas: the whole run does not fit a transaction; a dispute is one filter on one window")
    add("")
    if "osaka" in network:
        add("**The whole forward pass cannot run in one Osaka transaction.** `tools/network_gas.sh` deployed the")
        add("generated `Network` (every layer in one contract, %s bytes) on anvil at the Osaka hardfork and asked for" % n(network["network_deployed_bytes"]))
        add("`classify` of test sample 0:")
        add("")
        add("```")
        add("cast estimate:                         %s" % network["osaka"]["estimate"])
        add("cast send, 30,000,000 gas limit:       %s" % network["osaka"]["send_30m_gas_limit"])
        add("```")
        add("")
        add("It then deployed the same bytecode on a fresh anvil at Prague (no per-transaction cap, block gas limit")
        add("1,000,000,000), checked `classify` on %d samples against `pred.bin` and `trace` of sample 0 against"
            % network["classify_samples_checked"])
        add("`trace.jsonl` (every hidden value and logit), and mined `classify`: **%s gas, %.2f times the %s gas an"
            % (n(whole), whole / CAP, n(CAP)))
        add("Osaka transaction may use.**")
    else:
        add("The whole forward pass ran in one Osaka transaction: %s gas (%s of the %s cap)." % (n(whole), pct(whole, CAP), n(CAP)))
    add("")
    over = network.get("network_over_size_limit")
    if over:
        add("**Nor can it be deployed.** The `Network` contract is %s bytes of runtime code, %.2f times the %s bytes"
            % (n(over["runtime_bytes"]), over["runtime_bytes"] / over["eip170_runtime_limit"], n(over["eip170_runtime_limit"])))
        add("a contract may have (EIP-170), and its %s bytes of init code exceed the %s that EIP-3860 allows. It"
            % (n(over["init_code_bytes"]), n(over["eip3860_initcode_limit"])))
        add("was measured on anvil started with `%s`, deployed through a raw `eth_sendTransaction`." % over["anvil"])
        add("")
    mnist = load(ROOT, "examples", "mnist-784-128-64-10", "runs", "network-gas.json")
    mnist_model = load(ROOT, "examples", "mnist-784-128-64-10", "out", "model.json")
    mnist_macs = sum(layer["in"] * layer["out"] for layer in mnist_model["layers"])
    rate = mnist["classify"]["gas_used"] / mnist_macs
    add("**Unflattering, and a projection, not a measurement:** the dense kernel's whole run of 784 -> 128 -> 64 -> 10")
    add("costs %s gas for %s multiply-accumulates, %.1f gas each. At that rate this model's %s would cost about"
        % (n(mnist["classify"]["gas_used"]), n(mnist_macs), rate, n(macs(model))))
    add("%s gas, %s of the cap. The generated convolution costs %.1f gas per multiply-accumulate instead: layer 1 is"
        % (n(int(round(rate * macs(model)))), pct(rate * macs(model), CAP), whole / macs(model)))
    add("%s separate `neuron` calls on windows of %d values, so the fixed cost of a call dominates. \"Over the cap\""
        % (n(model["layers"][0]["out"]), model["layers"][0]["kernel"] * model["layers"][0]["channels"]))
    add("and \"over the size limit\" are facts about this generated contract, not bounds on every implementation of")
    add("this model; a contract written for gas might fit. The dispute does not depend on either.")
    add("")
    add("| | This model (conv, %s parameters) | Dense 144-32-16-3 (`examples/amm-fee`, %s parameters) |"
        % (n(params(model)), n(params(dense_model))))
    add("|---|---|---|")
    add("| Whole forward pass on-chain, gas | %s (%s) | %s |" % (n(whole), network["chain"], n(dense_network["classify"]["gas_used"])))
    add("| `claim`, gas | %s | %s |" % (n(claim), n(calls_by_name(dense_arbiter)["claim"]["gas_used"])))
    add("| Honest path, `claim` + `finalize`, gas | %s | %s |" % (n(honest), n(dense_honest)))
    add("| the same, as a share of one whole forward pass | %s | %s |" % (pct(honest, whole), pct(dense_honest, dense_network["classify"]["gas_used"])))
    add("| Dearest dispute (first slash of the run), gas | %s | %s |" % (n(dearest), n(find(dense_arbiter, "challengeNeuron1")["gas_used"])))
    add("| Dearest later dispute, gas | %s | %s |" % (n(dearest_later), n(max(call["gas_used"] for _, call in dense_later))))
    add("| Cheapest later dispute, gas | %s | %s |" % (n(min(call["gas_used"] for _, call in later)), n(dense_cheapest)))
    add("| `Arbiter` deployed bytes (limit 24,576) | %s | %s |" % (n(tree_rec["arbiter_runtime_bytes"]), n(dense_arbiter["arbiter_runtime_bytes"])))
    add("")
    add("Every challenge of this model, word-tree binding (the arbiter `check.sh` builds and tests), against the")
    add("same challenge on an arbiter generated with the flat binding (whole vectors in every challenge, the dense")
    add("rule), both measured in this run on anvil (Osaka). \"Later\" is the same challenge once the challenger's credit")
    add("and `burned` are already non-zero.")
    add("")
    add("| Call | Word tree: gas | calldata bytes | Flat: gas | calldata bytes |")
    add("|---|---|---|---|---|")
    for call in tree_rec["calls"]:
        other = fc.get(call["call"])
        add("| `%s` | %s | %s | %s | %s |" % (call["call"], n(call["gas_used"]), n(call["calldata_bytes"]),
                                            n(other["gas_used"]) if other else "", n(other["calldata_bytes"]) if other else ""))
    add("| `Arbiter` deployed bytes | %s | | %s | |" % (n(tree_rec["arbiter_runtime_bytes"]), n(flat_rec["arbiter_runtime_bytes"])))
    add("")
    tree_filters = [(name, call) for name, call in later if name.startswith("challengeFilter")]
    worst_tree = max(call["gas_used"] for _, call in tree_filters)
    worst_flat_name, worst_flat = max(((name, fc["later " + name]["gas_used"]) for name, _ in tree_filters),
                                      key=lambda row: row[1])
    cheapest = min(call["gas_used"] for _, call in later)
    dense_neurons = [(name, call) for name, call in dense_later if name.startswith("challengeNeuron")]
    dense_small_name, dense_small = min(((name, call["gas_used"]) for name, call in dense_neurons), key=lambda row: row[1])
    dense_big_name, dense_big = max(((name, call["gas_used"]) for name, call in dense_neurons), key=lambda row: row[1])
    flat_size = flat_rec["arbiter_runtime_bytes"]
    add("### What the numbers show")
    add("")
    add("- **A wrong value anywhere in this network is disputed for %s to %s gas** (later challenges), %s to %s"
        % (n(cheapest), n(dearest_later), pct(cheapest, whole), pct(dearest_later, whole)))
    add("  of the whole run%s. A filter challenge costs %s to %s gas. For scale, the dense fee model's neuron"
        % (" that one Osaka transaction cannot hold" if "osaka" in network else "",
           n(min(call["gas_used"] for _, call in tree_filters)), n(worst_tree)))
    add("  challenges cost %s (`%s`, the smallest) to %s (`%s`), so the dearest filter challenge here is %.2f"
        % (n(dense_small), dense_small_name, n(dense_big), dense_big_name, worst_tree / dense_small))
    add("  times the smallest dense one: a dispute stays the size of one neuron while the whole run grows %.1f times"
        % (whole / dense_network["classify"]["gas_used"]))
    add("  between the two fee models.")
    add("- **The binding was chosen by these rows.** With the flat binding the dearest later filter challenge,")
    add("  `%s`, costs %s gas: it carries the layer's whole input and output as calldata, and is priced by"
        % (worst_flat_name, n(worst_flat)))
    add("  the calldata floor of EIP-7623 (\"floor\" in `runs/arbiter-gas-flat.json`). With the word tree it costs")
    add("  %s. The flat claim is cheaper (%s gas against %s: the word tree computes a Merkle root of every"
        % (n(worst_tree), n(fc["claim"]["gas_used"]), n(claim)))
    add("  vector on-chain), so the flat binding is cheaper on the honest path by %s gas, and the word tree is"
        % n(claim - fc["claim"]["gas_used"]))
    add("  cheaper per dispute by up to %s. What decides it: **the flat `Arbiter` is %s deployed bytes, %s over the"
        % (n(worst_flat - worst_tree), n(flat_size), "%s bytes" % n(flat_size - 24576) if flat_size > 24576 else "not"))
    add("  24,576-byte limit of EIP-170, and could not be deployed** (it was measured on anvil with")
    add("  `--disable-code-size-limit`); the word-tree `Arbiter` is %s. A dispute that stays small however"
        % n(tree_rec["arbiter_runtime_bytes"]))
    add("  long the sequence grows is also what the scheme is for. `tools/gen_arbiter.py --binding flat` gives the other.")
    add("- **Unflattering: the claim is the dear part.** The honest path costs %s gas, %.2f times the dense fee"
        % (n(honest), honest / dense_honest))
    add("  model's, because the claim's calldata holds every hidden vector (%s bytes of calldata) and the claim"
        % n(tc["claim"]["calldata_bytes"]))
    add("  hashes them into word trees. It is still %s of a whole run." % pct(honest, whole))
    add("- **%s** (section 2). What this example shows is the"
        % ("And the model is no better than the dense one" if diff <= 0 else "The model is also a little better than the dense one"))
    add("  mechanism: a network whose full evaluation %s is still settled by one"
        % ("does not fit a transaction" if "osaka" in network else "is costly"))
    add("  transaction that re-runs %d to %d multiplications." % (
        min(layer["kernel"] * layer["channels"] for layer in model["layers"] if layer.get("kind") == "conv1d"),
        max(layer["in"] for layer in model["layers"])))
    add("")
    add("## 4. The checks behind this file (`check.sh`, one run)")
    add("")
    add("| Step | Result | Wall time | Last line of its output |")
    add("|---|---|---|---|")
    for status, seconds, name, detail in (row for row in steps if row[0] != "total"):
        add("| %s | %s | %s s | %s |" % (name, status, seconds, detail.replace("|", "\\|")))
    add("")
    add("The model ingot's tests are the first %s test samples (layer by layer, and value by value through the" % total[3])
    add("functions the arbiter calls), on the EVM backend and again natively. The arbiter's contract tests cover an")
    add("honest claim surviving challenges at the first, middle and last positions of every filter, each kind of")
    add("wrong value slashed by its own challenge only, wrong challenges refused, malformed claims rejected, and")
    add("the lifecycle. The native classifier against the NumPy reference on the whole test split:")
    add("")
    add("```")
    out.extend(native)
    add("```")
    add("")
    add("## 5. Runtime sizes (`fe build`, limit 24,576 bytes)")
    add("")
    add("| Contract | Runtime bytes | Under the limit by |")
    add("|---|---|---|")
    for name, size in sizes.items():
        add("| `%s` | %s | %s |" % (name, n(size), n(24576 - size)))
    add("")
    add("`Wallet` is a test double from the arbiter's tests; `Network` is the whole forward pass, all weights in")
    add("its own code (no weight contracts: a convolution's filters are few).")
    add("")
    with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print("wrote RESULTS.md (%d lines)" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
