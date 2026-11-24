#!/usr/bin/env python3
"""Mutation test of the generated-arbiter Foundry suite: does it notice when a GENERATED arbiter is
wrong? The sibling of tools/m3_mutants.py (the shipped two-layer arbiter), for the three-layer
arbiter that tools/gen_arbiter.py wrote for examples/amm-fee (144 -> 32 -> 16 -> 3).

Each mutant is one deliberate fault in a COPY of the example's generated sources (arbiter.fe,
record.fe, commitment.fe, and the model ingot written by tools/gen_fe.py); the project's files are
never touched. The copy is built with `fe build --ingot arbiter`, then evm/generated/GenArbiter*.t.sol
runs against that bytecode with --fail-fast. Outcomes, as in tools/m3_mutants.py:

  killed     a Foundry test failed
  compiler   `fe build` refused the mutant (the message is recorded)
  SURVIVED   every test passed: a gap in the suite, or an equivalent mutant (argued in the report)
  no-change  the bytecode is identical to the baseline, so the mutant says nothing
  timeout    forge did not finish in the time limit (counted as a kill, and worth a look)

Most mutants exist only because the arbiter has more than two layers: a middle handler whose input
and output are both hidden vectors, so a hash, a leaf's layer number, a kernel, a range bound or a
vector can be taken from the wrong layer and still type-check.

The baseline is the whole suite run by tools/gen_foundry.sh (staged under $WORK/gen) and must pass.
Results go to build/gen-mutants.jsonl as they happen, so an interrupted run resumes where it
stopped; docs/mutation-results-generated.md is written at the end.

  wsl -e python3 /mnt/c/.../fe-challenge/tools/gen_mutants.py [--fresh] [--only NAME[,NAME...]] [--report]

--report only rewrites the report from build/gen-mutants.jsonl (for example after editing NOTES).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path.home() / ".cache" / "tensile-gen-mutants"
RESULTS = ROOT / "build" / "gen-mutants.jsonl"
PASS = ROOT / "build" / "gen-mutants-pass.txt"  # the header line of the last full pass
REPORT = ROOT / "docs" / "mutation-results-generated.md"
EXAMPLE = "amm-fee"
EX = f"examples/{EXAMPLE}"
FORGE_LIMIT = 900

ARB = f"{EX}/ingots/arbiter/src/arbiter.fe"
REC = f"{EX}/ingots/arbiter/src/record.fe"
COM = f"{EX}/ingots/arbiter/src/commitment.fe"
L1 = f"{EX}/ingots/model/src/layer1.fe"
L2 = f"{EX}/ingots/model/src/layer2.fe"
L3 = f"{EX}/ingots/model/src/layer3.fe"

H1 = "if keccak_words(hidden1) != record.hidden1_hash {"  # 0: layer 1's output, 1: layer 2's input
H2 = "if keccak_words(hidden2) != record.hidden2_hash {"  # 0: layer 2's output, 1: layer 3's input
LOGH = "if logits_hash(logits) != record.logits_hash {"  # 0: layer 3's output, 1: the class challenge
FRAUD = "Action::Challenge(recomputed != claimed)"  # 0, 1, 2: layers 1, 2, 3
LEAF1 = "leaf(layer: 1, neuron: j, bias, row)"
LEAF2 = "leaf(layer: 2, neuron: j, bias, row)"
LEAF3 = "leaf(layer: 3, neuron: j, bias, row)"
CLAIMED2 = "let claimed = Hidden2::from_words(hidden2).get(j.downcast_truncate())"
MULTI = "multi-layer"  # marks a mutant a two-layer arbiter cannot have

# (name, file, old text, new text, which occurrence of the old text, what it breaks, kind)
MUTANTS = [
    # Claim: what is checked and what is committed
    ("claim_no_bond_check", ARB, "if paid < bond {", "if false {", 0, "any bond is accepted", ""),
    ("claim_no_input_check", ARB, "if !Input::well_formed(x0) {", "if false {", 0, "malformed input is stored, and traps a challenge", ""),
    ("claim_no_hidden1_check", ARB, "if !Hidden1::well_formed(hidden1) {", "if false {", 0, "a malformed first hidden vector is stored", ""),
    ("claim_no_hidden2_check", ARB, "if !Hidden2::well_formed(hidden2) {", "if false {", 0, "a malformed second hidden vector is stored", MULTI),
    ("claim_hidden2_checked_as_32_values", ARB, "if !Hidden2::well_formed(hidden2) {", "if !Hidden1::well_formed(hidden2) {", 0, "the 16 padding bytes of the second hidden vector go unchecked", MULTI),
    ("claim_class_off_by_one", ARB, "if class >= CLASSES {", "if class > CLASSES {", 0, "class 3 is accepted", ""),
    ("claim_stores_minimum_bond", ARB, "bond: paid,", "bond: bond,", 0, "an excess payment is stuck", ""),
    ("claim_event_wrong_bond", ARB, "bond: paid,", "bond: bond,", 1, "the event misreports the bond", ""),
    ("claim_id_not_incremented", ARB, "count = id + 1", "count = id", 0, "claims overwrite each other", ""),
    ("claim_commits_x0_as_hidden1", ARB, "hidden1_hash: keccak_words(hidden1),", "hidden1_hash: keccak_words(x0),", 0, "the first hidden vector is not committed", ""),
    ("claim_commits_hidden1_as_hidden2", ARB, "hidden2_hash: keccak_words(hidden2),", "hidden2_hash: keccak_words(hidden1),", 0, "the second hidden vector is not committed; the first is, twice", MULTI),
    ("claim_event_hidden2_is_hidden1", ARB, "hidden2: dynamic(hidden2),", "hidden2: dynamic(hidden1),", 0, "the event carries the wrong second hidden vector", MULTI),
    ("claim_hashes_view_swapped", ARB, "(record.x0_hash, record.hidden1_hash, record.hidden2_hash, record.logits_hash)", "(record.x0_hash, record.hidden2_hash, record.hidden1_hash, record.logits_hash)", 0, "claimHashes returns the hidden hashes out of layer order", MULTI),
    # Challenges: range checks
    ("n1_no_range_check", ARB, "if j >= HIDDEN1 {", "if false {", 0, "neuron index unchecked", ""),
    ("n2_no_range_check", ARB, "if j >= HIDDEN2 {", "if false {", 0, "neuron index of the middle layer unchecked", MULTI),
    ("n2_range_of_layer_1", ARB, "if j >= HIDDEN2 {", "if j >= HIDDEN1 {", 0, "the middle layer accepts indices 16 to 31", MULTI),
    ("n3_no_range_check", ARB, "if j >= CLASSES as u256 {", "if false {", 0, "neuron index unchecked", ""),
    ("n3_range_off_by_one", ARB, "if j >= CLASSES as u256 {", "if j > CLASSES as u256 {", 0, "logit index 3 accepted", ""),
    # Challenges: the hashes that bind a handler to the claim's vectors
    ("n1_no_input_hash", ARB, "if keccak_words(x0) != record.x0_hash {", "if false {", 0, "a challenger can use another input", ""),
    ("n1_no_output_hash", ARB, H1, "if false {", 0, "a challenger can use another first hidden vector", ""),
    ("n2_no_input_hash", ARB, H1, "if false {", 1, "the middle layer's input is not bound to the claim", MULTI),
    ("n2_no_output_hash", ARB, H2, "if false {", 0, "the middle layer's output is not bound to the claim", MULTI),
    ("n3_no_input_hash", ARB, H2, "if false {", 1, "the last layer's input is not bound to the claim", ""),
    ("n3_no_logits_hash", ARB, LOGH, "if false {", 0, "a challenger can use other logits", ""),
    ("class_no_logits_hash", ARB, LOGH, "if false {", 1, "a challenger can use other logits", ""),
    ("n2_input_checked_against_x0_hash", ARB, H1, "if keccak_words(hidden1) != record.x0_hash {", 1, "the middle layer checks its input against the input layer's hash", MULTI),
    ("n2_input_checked_against_hidden2_hash", ARB, H1, "if keccak_words(hidden1) != record.hidden2_hash {", 1, "the middle layer checks its input against its own output's hash", MULTI),
    ("n2_output_checked_against_hidden1_hash", ARB, H2, "if keccak_words(hidden2) != record.hidden1_hash {", 0, "the middle layer checks its output against its input's hash", MULTI),
    ("n3_input_checked_against_hidden1_hash", ARB, H2, "if keccak_words(hidden2) != record.hidden1_hash {", 1, "the last layer checks its input against the first hidden vector's hash", MULTI),
    # Challenges: the Merkle leaf
    ("n1_no_merkle", ARB, f"if !merkle::verify(proof, root, leaf: {LEAF1}) {{", "if false {", 0, "a challenger can use any weights", ""),
    ("n2_no_merkle", ARB, f"if !merkle::verify(proof, root, leaf: {LEAF2}) {{", "if false {", 0, "a challenger can use any weights in the middle layer", MULTI),
    ("n3_no_merkle", ARB, f"if !merkle::verify(proof, root, leaf: {LEAF3}) {{", "if false {", 0, "a challenger can use any weights", ""),
    ("n1_leaf_says_layer_2", ARB, LEAF1, "leaf(layer: 2, neuron: j, bias, row)", 0, "layer 1 builds its leaf with layer number 2", ""),
    ("n2_leaf_says_layer_1", ARB, LEAF2, "leaf(layer: 1, neuron: j, bias, row)", 0, "the middle layer's leaf is off by one layer, down", MULTI),
    ("n2_leaf_says_layer_3", ARB, LEAF2, "leaf(layer: 3, neuron: j, bias, row)", 0, "the middle layer's leaf is off by one layer, up: rows of layers 2 and 3 have the same width", MULTI),
    ("n3_leaf_says_layer_2", ARB, LEAF3, "leaf(layer: 2, neuron: j, bias, row)", 0, "the last layer accepts the middle layer's leaves, which have the same width", MULTI),
    ("n2_leaf_neuron_off_by_one", ARB, LEAF2, "leaf(layer: 2, neuron: j + 1, bias, row)", 0, "the middle layer's leaf names the next neuron", MULTI),
    # Challenges: the recomputation and what it is compared with
    ("n1_always_fraud", ARB, FRAUD, "Action::Challenge(true)", 0, "an honest claim can be slashed", ""),
    ("n2_always_fraud", ARB, FRAUD, "Action::Challenge(true)", 1, "an honest claim can be slashed at the middle layer", MULTI),
    ("n3_always_fraud", ARB, FRAUD, "Action::Challenge(true)", 2, "an honest claim can be slashed", ""),
    ("n1_never_fraud", ARB, FRAUD, "Action::Challenge(false)", 0, "a false first hidden value stands", ""),
    ("n2_never_fraud", ARB, FRAUD, "Action::Challenge(false)", 1, "a false second hidden value stands", MULTI),
    ("n3_never_fraud", ARB, FRAUD, "Action::Challenge(false)", 2, "a false logit stands", ""),
    ("n2_recomputes_on_its_output", ARB, "layer2::neuron_from_words(row, x: hidden1, bias)", "layer2::neuron_from_words(row, x: hidden2, bias)", 0, "the middle layer runs on the wrong vector (same width)", MULTI),
    ("n2_uses_layer3_kernel", ARB, "layer2::neuron_from_words(row, x: hidden1, bias)", "layer3::neuron_from_words(row, x: hidden1, bias)", 0, "the middle layer runs the last layer's kernel (no shift, no clamp)", MULTI),
    ("n3_uses_layer2_kernel", ARB, "layer3::neuron_from_words(row, x: hidden2, bias)", "layer2::neuron_from_words(row, x: hidden2, bias)", 0, "the last layer runs the middle layer's kernel", MULTI),
    ("n2_compares_with_hidden1", ARB, CLAIMED2, "let claimed = Hidden1::from_words(hidden1).get(j.downcast_truncate())", 0, "the middle layer compares with its input instead of its output", MULTI),
    ("n2_reads_output_as_layer_1_type", ARB, CLAIMED2, "let claimed = Hidden1::from_words(hidden2).get(j.downcast_truncate())", 0, "the middle layer reads its output with the first hidden layer's type", MULTI),
    ("n1_reads_neuron_0", ARB, "Hidden1::from_words(hidden1).get(j.downcast_truncate())", "Hidden1::from_words(hidden1).get(0)", 0, "compares with the wrong hidden value", ""),
    ("n2_reads_neuron_0", ARB, CLAIMED2, "let claimed = Hidden2::from_words(hidden2).get(0)", 0, "the middle layer compares with the wrong value", MULTI),
    ("n3_reads_logit_0", ARB, "let claimed = logits[j.downcast_truncate()]", "let claimed = logits[0]", 0, "compares with the wrong logit", ""),
    ("class_never_fraud", ARB, "let fraud = Logits::from_raw(logits).argmax() != record.class as usize", "let fraud = false", 0, "a false class stands", ""),
    ("class_always_fraud", ARB, "let fraud = Logits::from_raw(logits).argmax() != record.class as usize", "let fraud = true", 0, "an honest class can be slashed", ""),
    # Settlement
    ("slash_pays_claimant", ARB, "Payee::Challenger => challenger,", "Payee::Challenger => record.claimant,", 0, "the challenger is not paid", ""),
    ("finalize_pays_caller", ARB, "Payee::Claimant => record.claimant,", "Payee::Claimant => challenger,", 0, "anyone can take a bond by finalising", ""),
    ("credit_overwrites", ARB, "credits[payee] = credits[payee] + wei(paid)", "credits[payee] = wei(paid)", 0, "an earlier credit is lost", ""),
    ("slash_credits_whole_bond", ARB, "credits[payee] = credits[payee] + wei(paid)", "credits[payee] = credits[payee] + record.bond", 0, "a slash pays the whole bond and burns half as well", ""),
    ("burn_not_recorded", ARB, "if burnt != 0 {", "if false {", 0, "the burned half is held but not accounted", ""),
    ("burn_overwrites", ARB, "burned = burned + wei(burnt)", "burned = wei(burnt)", 0, "an earlier burn is forgotten", ""),
    ("status_not_stored", ARB, "claims[id].status = transition.status.code()", "", 0, "a bond can be credited again and again", ""),
    ("clock_plus_one", ARB, "now: now.as_seconds()", "now: now.as_seconds() + 1", 0, "the deadline is off by one", ""),
    ("deadline_plus_one", ARB, "deadline: record.deadline.as_seconds(),", "deadline: record.deadline.as_seconds() + 1,", 0, "the deadline is off by one", ""),
    # Withdraw
    ("withdraw_keeps_credit", ARB, "credits[account] = Wei::zero()", "", 0, "a credit can be withdrawn again and again", ""),
    (
        "withdraw_zeroes_after_transfer",
        ARB,
        "credits[account] = Wei::zero()\n"
        "            if !call.send_value(addr: account, gas: ctx.gas(), value: amount).success() {\n"
        "                revert_error(TransferFailed {})\n"
        "            }\n",
        "if !call.send_value(addr: account, gas: ctx.gas(), value: amount).success() {\n"
        "                revert_error(TransferFailed {})\n"
        "            }\n"
        "            credits[account] = Wei::zero()\n",
        0,
        "the classic re-entrancy bug",
        "",
    ),
    ("withdraw_ignores_failure", ARB, "value: amount).success() {", "value: amount).success() && false {", 0, "a refused transfer loses the credit", ""),
    # Views and events
    ("unknown_claim_not_refused", ARB, "if id >= count {", "if false {", 0, "an unused number reads as a claim", ""),
    ("result_before_final", ARB, "if record.status != Status::Final.code() {", "if false {", 0, "an open or slashed claim's class reads as a result", ""),
    ("event_n1_wrong_layer", ARB, "layer: 1, neuron: j })", "layer: 2, neuron: j })", 0, "the slash event misreports the layer", ""),
    ("event_n2_wrong_layer", ARB, "layer: 2, neuron: j })", "layer: 3, neuron: j })", 0, "the middle layer's slash event says layer 3", MULTI),
    ("event_n3_wrong_layer", ARB, "layer: 3, neuron: j })", "layer: 2, neuron: j })", 0, "the last layer's slash event says layer 2", MULTI),
    ("event_n2_wrong_neuron", ARB, "layer: 2, neuron: j })", "layer: 2, neuron: 0 })", 0, "the slash event misreports the neuron", MULTI),
    ("event_class_slashed_missing", ARB, "log.emit(ClassSlashed { id, challenger })", "", 0, "no event for a class slash", ""),
    ("event_finalized_wrong_class", ARB, "log.emit(Finalized { id, class: record.class })", "log.emit(Finalized { id, class: 0 })", 0, "the event misreports the class", ""),
    ("event_withdrawn_missing", ARB, "log.emit(Withdrawn { account, amount })", "", 0, "no event for a withdrawal", ""),
    # Commitments
    ("logits_hash_first_only", REC, "words[j] = logits[j].to_word()", "words[j] = logits[0].to_word()", 0, "the commitment ignores two of the three logits", ""),
    ("leaf_without_layer", COM, "    preimage.u256(layer)\n", "    preimage.u256(1)\n", 0, "every leaf says layer 1", ""),
    ("leaf_without_neuron", COM, "    preimage.u256(neuron)\n", "    preimage.u256(0)\n", 0, "every leaf says neuron 0", ""),
    # The generated model ingot (tools/gen_fe.py), as the arbiter calls it
    ("model_layer1_shift_off_by_one", L1, "tensile::neuron(row, x, bias).shifted<9>().clamped()", "tensile::neuron(row, x, bias).shifted<8>().clamped()", 0, "the first hidden layer requantises by 8 bits instead of 9", ""),
    ("model_layer2_shift_off_by_one", L2, "tensile::neuron(row, x, bias).shifted<8>().clamped()", "tensile::neuron(row, x, bias).shifted<9>().clamped()", 0, "the middle layer requantises by 9 bits instead of 8", MULTI),
    ("model_layer2_no_clamp", L2, "tensile::neuron(row, x, bias).shifted<8>().clamped()", "tensile::neuron(row, x, bias).shifted<8>().raw().downcast_truncate()", 0, "the middle layer does not clamp to 0..127", MULTI),
    ("model_layer3_ignores_bias", L3, "        bias: Q::from_raw(bias),\n", "        bias: Q::from_raw(0),\n", 0, "the last layer's challenge drops the bias", ""),
]


def env():
    e = dict(os.environ)
    home = str(Path.home())
    e["PATH"] = f"{home}/.fe/bin:{home}/.foundry/bin:" + e["PATH"]
    e["NO_COLOR"] = "1"
    return e


def run(cmd, cwd, limit=None, extra=None):
    e = env()
    if extra:
        e.update(extra)
    return subprocess.run(cmd, cwd=cwd, env=e, text=True, errors="replace", capture_output=True, timeout=limit)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_nth(text, old, new, n):
    if "\r\n" in text:
        old = old.replace("\n", "\r\n")
        new = new.replace("\n", "\r\n")
    pos = -1
    for _ in range(n + 1):
        pos = text.find(old, pos + 1)
        if pos < 0:
            return None
    return text[:pos] + new + text[pos + len(old) :]


class Bench:
    def __init__(self):
        self.repo = WORK / "repo"
        self.fe_out = WORK / "fe-out"
        self.gen = WORK / "gen"
        self.stage = self.gen / EXAMPLE / "stage"
        shutil.rmtree(self.repo, ignore_errors=True)
        (self.repo / "ingots").mkdir(parents=True)
        for ingot in ("tensile", "dispute"):
            shutil.copytree(ROOT / "ingots" / ingot, self.repo / "ingots" / ingot)
        (self.repo / EX).mkdir(parents=True)
        shutil.copytree(ROOT / EX / "ingots", self.repo / EX / "ingots")
        shutil.copy(ROOT / EX / "fe.toml", self.repo / EX / "fe.toml")

    def build(self):
        shutil.rmtree(self.fe_out, ignore_errors=True)
        self.fe_out.mkdir(parents=True)
        return run(
            ["fe", "build", "--color", "never", "--out-dir", str(self.fe_out), "--ingot", "arbiter", "."],
            self.repo / EX,
        )

    def baseline(self):
        """The whole suite through tools/gen_foundry.sh, which also stages the Foundry project."""
        return run(["bash", str(ROOT / "tools" / "gen_foundry.sh"), EXAMPLE], ROOT, extra={"GEN_STAGE": str(self.gen)})

    def forge(self):
        shutil.copy(self.fe_out / "Arbiter.bin", self.stage / "fe-out" / "Arbiter.bin")
        log = self.stage / "invariant-out" / "campaign.log"
        if log.exists():
            log.unlink()
        return run(["forge", "test", "--color", "never", "--fail-fast"], self.stage, FORGE_LIMIT)


def first_failure(output):
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("[FAIL"):
            continue
        reason = re.match(r"\[FAIL:?\s*([^;\]]*)", line).group(1).strip()[:110]
        m = re.match(r"\[FAIL.*\]\s+(\w+)\(", line)
        if m:
            return m.group(1), reason
        for later in lines[i + 1 :]:
            m = re.match(r"\s*(invariant_\w+)\(", later)
            if m:
                return m.group(1), reason
        return "?", reason
    for line in lines:
        if "FAIL" in line or "Error" in line:
            return "?", line.strip()[:110]
    return "?", ""


def compile_message(output):
    for line in output.splitlines():
        if line.startswith("error") or "static_assert" in line or "assertion" in line:
            return line.strip()[:150]
    return output.strip().splitlines()[0][:150] if output.strip() else ""


def load_done():
    done = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            r = json.loads(line)
            done[r["name"]] = r
    return done


# What each mutant that survived or changed nothing means: why it is equivalent, or which test was
# added to kill it. Written after a run, by hand; printed at the end of the report.
NOTES = {
    "n2_reads_output_as_layer_1_type": (
        "SURVIVED, and equivalent: no test can kill it. The mutant reads the middle layer's claimed value "
        "as `Hidden1::from_words(hidden2).get(j)` instead of `Hidden2::from_words(hidden2).get(j)`. In "
        "144-32-16-3 both types are one packed word (`[u256; 1]`, 32 and 16 values), so it type-checks. "
        "The two reads differ in three ways, and none is reachable: `get` asserts `j < 32` instead of "
        "`j < 16`, but the handler has already refused `j >= HIDDEN2` with `NeuronOutOfRange` (which "
        "`n2_no_range_check` and `n2_range_of_layer_1` show is tested); `from_words` checks that bytes 16 "
        "to 31 are at most 127 instead of zero, but the vector has just been matched against the claim's "
        "`hidden2_hash`, and the claim stored that hash only after `Hidden2::well_formed` accepted it "
        "(`claim_hidden2_checked_as_32_values` is killed); and byte j of the word is the same byte for "
        "both types. So for every call that reaches the line, both versions return the same value and "
        "neither assertion fires (barring a keccak collision). The types separate layers only where the "
        "Fe signatures differ: the kernel swaps (`n2_uses_layer3_kernel`, `n3_uses_layer2_kernel`) are "
        "refused by the compiler because a hidden neuron returns an `i8` activation and the last layer an "
        "`i32` logit, but every handler takes its vectors as plain word arrays and `from_words` accepts any "
        "array of the right word count, so two layers whose vectors fit in the same number of words (here "
        "both in one) can be read as each other. That is what the rest of the multi-layer rows test, and "
        "the suite kills every such mutant that changes a value."
    ),
}


def write_report(done, header):
    notes = NOTES
    counts = {}
    multi_counts = {}
    rows = []
    for name, file, _, _, _, breaks, kind in MUTANTS:
        r = done.get(name)
        if not r:
            continue
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        if kind == MULTI:
            multi_counts[r["outcome"]] = multi_counts.get(r["outcome"], 0) + 1
        by = r.get("by", "")
        detail = f"`{by}`" if by else (r.get("message", "").replace("|", "\\|") if r["outcome"] == "compiler" else "")
        rows.append(
            f"| `{name}` | `{Path(file).parent.parent.name}/{Path(file).name}` | {breaks} | {'yes' if kind == MULTI else ''} "
            f"| {r['outcome']} | {detail} | {r['seconds']:.0f} |"
        )
    total = sum(counts.values())
    explained = [f"- `{k}`: {v}" for k, v in notes.items() if k in done]
    head = [
        "# Mutation test of the generated three-layer arbiter's Foundry suite",
        "",
        f"Generated by `tools/gen_mutants.py`. Each row is one deliberate fault in a copy of the sources that",
        f"the generator wrote for `{EX}` (144 -> 32 -> 16 -> 3): the arbiter ingot (`tools/gen_arbiter.py`)",
        "and, in the last rows, the model ingot (`tools/gen_fe.py`). Each mutant is built with `fe build`",
        "and run against `evm/generated/GenArbiter*.t.sol` with `--fail-fast`, so the test named is the",
        "first to fail, not the only one. The shipped two-layer arbiter's run is `docs/mutation-results.md`.",
        "",
        "\"Multi-layer\" marks a fault a two-layer arbiter cannot have: the middle handler, whose input and",
        "output are both hidden vectors, takes a hash, a leaf's layer number, a kernel, a range bound or a",
        "vector from the wrong layer. Most of these still type-check; the compiler refuses only the kernel",
        "swaps. A survivor is either given the test that kills it or argued equivalent at the end.",
        "",
        "The suite under test is the shipped arbiter's (`evm/test/Arbiter*.t.sol`, `docs/fe-notes.md` M3.9)",
        "with the sizes taken out: shapes, rows, biases, proofs and reference samples are read from a fixture",
        "that `tools/gen_arbiter_fixtures_any.py` writes from the example's `out/model.json`, its reference",
        "trace and its neuron tree, and the calldata is built from signatures derived from the spec. Its",
        "oracle for what the network computes is `GModel` (`evm/generated/GenBase.sol`): the integer network",
        "in plain Solidity over the fixture's rows, with the byte loop of `evm/test/SpecNeuron.sol`. It shares",
        "no code with the Fe kernel, the generated model ingot or the generator, and",
        "`test_oracle_reproduces_the_reference_samples` checks it against the committed reference trace (the",
        "fixture writer also checks every sample against its own forward pass in Python). Run it with",
        "`tools/gen_foundry.sh`; this file with `tools/gen_mutants.py`.",
        "",
        "Totals: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + f" (of {total}); multi-layer only: "
        + ", ".join(f"{v} {k}" for k, v in sorted(multi_counts.items())) + f" (of {sum(multi_counts.values())}).",
    ]
    if header:
        head += ["", header]
    head += [
        "",
        "| Mutant | File | What it breaks | Multi-layer | Outcome | First failing test, or the compiler's message | s |",
        "|---|---|---|---|---|---|---|",
    ]
    tail = []
    if explained:
        tail = ["", "## Survivors and mutants that changed nothing", ""] + explained
    REPORT.write_text("\n".join(head + rows + tail) + "\n", encoding="utf-8", newline="\n")


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        only = set(args[args.index("--only") + 1].split(","))
    if "--report" in args:
        # Only rewrite the report from the recorded results (after editing NOTES, for example).
        write_report(load_done(), PASS.read_text().strip() if PASS.exists() else "")
        print(f"rewrote {REPORT.relative_to(ROOT)} from {RESULTS.relative_to(ROOT)}")
        return 0
    if "--fresh" in args and RESULTS.exists():
        RESULTS.unlink()
    RESULTS.parent.mkdir(exist_ok=True)
    done = load_done()
    start = time.time()
    bench = Bench()

    t = time.time()
    f = bench.baseline()
    summary = [l for l in f.stdout.splitlines() if l.startswith("ok ") or l.startswith("FAIL ")]
    print(f"baseline (tools/gen_foundry.sh {EXAMPLE}): exit {f.returncode}, {summary[-1].strip() if summary else 'no summary'}, "
          f"{time.time() - t:.1f} s", flush=True)
    if f.returncode != 0:
        print(f.stdout[-4000:] + f.stderr[-2000:])
        return 2
    t = time.time()
    b = bench.build()
    if b.returncode != 0:
        print("the copy does not build:\n" + b.stdout + b.stderr)
        return 2
    base_hash = sha(bench.fe_out / "Arbiter.runtime.bin")
    staged = (bench.stage / "fe-out" / "Arbiter.bin").read_text().strip()
    if (bench.fe_out / "Arbiter.bin").read_text().strip() != staged:
        print("the copy's bytecode differs from the baseline's")
        return 2
    print(f"copy built, identical to the baseline's bytecode, {time.time() - t:.1f} s", flush=True)

    for name, file, old, new, nth, _, _ in MUTANTS:
        if only is not None and name not in only:
            continue
        if only is None and name in done:
            continue
        t = time.time()
        path = bench.repo / file
        original = path.read_bytes().decode()
        mutated = replace_nth(original, old, new, nth)
        if mutated is None:
            r = {"name": name, "outcome": "BAD-MUTANT", "message": "old text not found", "seconds": 0.0}
        else:
            path.write_bytes(mutated.encode())
            b = bench.build()
            if b.returncode != 0:
                r = {"name": name, "outcome": "compiler", "message": compile_message(b.stdout + b.stderr)}
            elif sha(bench.fe_out / "Arbiter.runtime.bin") == base_hash:
                r = {"name": name, "outcome": "no-change"}
            else:
                try:
                    f = bench.forge()
                    if f.returncode == 0:
                        r = {"name": name, "outcome": "SURVIVED"}
                    else:
                        by, why = first_failure(f.stdout + f.stderr)
                        r = {"name": name, "outcome": "killed", "by": by, "message": why}
                except subprocess.TimeoutExpired:
                    r = {"name": name, "outcome": "timeout"}
            path.write_bytes(original.encode())
            r["seconds"] = round(time.time() - t, 1)
        done[name] = r
        with RESULTS.open("a") as out:
            out.write(json.dumps(r) + "\n")
        print(f"{name}: {r['outcome']} {r.get('by', '')} {r.get('message', '')} ({r['seconds']} s)", flush=True)
        write_report(done, "")

    counts = {}
    for name, *_ in MUTANTS:
        if name in done:
            counts[done[name]["outcome"]] = counts.get(done[name]["outcome"], 0) + 1
    wall = time.time() - start
    header = ""
    if only is None:
        header = f"Last full pass: {len(MUTANTS)} mutants, wall {wall:.1f} s including the baseline."
        PASS.write_text(header + "\n")
    write_report(done, header)
    print(f"gen_mutants: {counts}, wall {wall:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
