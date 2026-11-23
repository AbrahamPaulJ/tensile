#!/usr/bin/env python3
"""Mutation test of the Arbiter's Foundry suite: does it notice when the contract is wrong?

Each mutant is one deliberate fault in a COPY of the Fe source (the project's own files are never
touched). The copy is built with `fe build`, then evm/test/Arbiter*.t.sol runs against that bytecode
with --fail-fast. Outcomes:

  killed     a Foundry test failed
  compiler   `fe build` refused the mutant (a static_assert or a type error; the message is recorded)
  SURVIVED   every test passed: a gap in the suite, or an equivalent mutant (explain it in M3.9)
  no-change  the bytecode is identical to the baseline, so the mutant says nothing
  timeout    forge did not finish in the time limit (counted as a kill, and worth a look)

An unmutated baseline runs first and must pass. Results go to build/mutants.jsonl as they happen, so
an interrupted run resumes where it stopped, and docs/mutation-results.md is written at the end.

  wsl -e python3 /mnt/c/.../fe-challenge/tools/m3_mutants.py [--fresh] [--only NAME[,NAME...]]
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
WORK = Path.home() / ".cache" / "tensile-mutants"
RESULTS = ROOT / "build" / "mutants.jsonl"
REPORT = ROOT / "docs" / "mutation-results.md"
FORGE_LIMIT = 600

ARB = "ingots/onchain/src/arbiter.fe"
STEP = "ingots/dispute/src/step.fe"
REC = "ingots/onchain/src/record.fe"
ACC = "ingots/tensile/src/accumulators.fe"
ACT = "ingots/tensile/src/activations.fe"
PAK = "ingots/tensile/src/packed.fe"
DOT = "ingots/tensile/src/dot.fe"

HASH1 = "if keccak_words(hidden) != record.hidden_hash {"
LOGH = "if logits_hash(logits) != record.logits_hash {"
FRAUD = "Action::Challenge(recomputed != claimed)"

# (name, file, old text, new text, which occurrence of the old text, what it breaks)
MUTANTS = [
    # Claim
    ("claim_no_bond_check", ARB, "if paid < bond {", "if false {", 0, "any bond is accepted"),
    ("claim_no_input_check", ARB, "if !Input::well_formed(x0) {", "if false {", 0, "malformed input is stored, and traps a challenge"),
    ("claim_no_hidden_check", ARB, "if !Hidden::well_formed(hidden) {", "if false {", 0, "malformed hidden is stored, and traps a challenge"),
    ("claim_class_off_by_one", ARB, "if class >= CLASSES {", "if class > CLASSES {", 0, "class 10 is accepted"),
    ("claim_stores_minimum_bond", ARB, "bond: paid,", "bond: bond,", 0, "an excess payment is stuck"),
    ("claim_event_wrong_bond", ARB, "bond: paid,", "bond: bond,", 1, "the event misreports the bond"),
    ("claim_id_not_incremented", ARB, "count = id + 1", "count = id", 0, "claims overwrite each other"),
    # Challenges: the checks that make a challenge binding
    ("n1_no_range_check", ARB, "if j >= HIDDEN {", "if false {", 0, "neuron index unchecked"),
    ("n2_no_range_check", ARB, "if j >= CLASSES as u256 {", "if false {", 0, "neuron index unchecked"),
    ("n1_no_input_hash", ARB, "if keccak_words(x0) != record.x0_hash {", "if false {", 0, "a challenger can use another input"),
    ("n1_no_hidden_hash", ARB, HASH1, "if false {", 0, "a challenger can use another hidden vector"),
    ("n2_no_hidden_hash", ARB, HASH1, "if false {", 1, "a challenger can use another hidden vector"),
    ("n2_no_logits_hash", ARB, LOGH, "if false {", 0, "a challenger can use other logits"),
    ("class_no_logits_hash", ARB, LOGH, "if false {", 1, "a challenger can use other logits"),
    ("n1_no_merkle", ARB, "if !merkle::verify(proof, root, leaf: leaf(layer: 1, neuron: j, bias, row)) {", "if false {", 0, "a challenger can use any weights"),
    ("n2_no_merkle", ARB, "if !merkle::verify(proof, root, leaf: leaf(layer: 2, neuron: j, bias, row)) {", "if false {", 0, "a challenger can use any weights"),
    # Challenges: the judgement
    ("n1_always_fraud", ARB, FRAUD, "Action::Challenge(true)", 0, "an honest claim can be slashed"),
    ("n2_always_fraud", ARB, FRAUD, "Action::Challenge(true)", 1, "an honest claim can be slashed"),
    ("n1_never_fraud", ARB, FRAUD, "Action::Challenge(false)", 0, "a false hidden value stands"),
    ("n2_never_fraud", ARB, FRAUD, "Action::Challenge(false)", 1, "a false logit stands"),
    ("n1_reads_neuron_0", ARB, "Hidden::from_words(hidden).get(j.downcast_truncate())", "Hidden::from_words(hidden).get(0)", 0, "compares with the wrong hidden value"),
    ("n2_reads_logit_0", ARB, "let claimed = logits[j.downcast_truncate()]", "let claimed = logits[0]", 0, "compares with the wrong logit"),
    ("class_never_fraud", ARB, "let fraud = Logits::from_raw(logits).argmax() != record.class as usize", "let fraud = false", 0, "a false class stands"),
    ("class_always_fraud", ARB, "let fraud = Logits::from_raw(logits).argmax() != record.class as usize", "let fraud = true", 0, "an honest class can be slashed"),
    # Settlement
    ("slash_pays_claimant", ARB, "Payee::Challenger => challenger,", "Payee::Challenger => record.claimant,", 0, "the challenger is not paid"),
    ("finalize_pays_caller", ARB, "Payee::Claimant => record.claimant,", "Payee::Claimant => challenger,", 0, "anyone can take a bond by finalising"),
    ("credit_overwrites", ARB, "credits[payee] = credits[payee] + wei(paid)", "credits[payee] = wei(paid)", 0, "an earlier credit is lost"),
    ("slash_credits_whole_bond", ARB, "credits[payee] = credits[payee] + wei(paid)", "credits[payee] = credits[payee] + record.bond", 0, "a slash pays the whole bond and burns half as well: more is owed than held"),
    ("burn_not_recorded", ARB, "if burnt != 0 {", "if false {", 0, "the burned half is held but not accounted"),
    ("burn_overwrites", ARB, "burned = burned + wei(burnt)", "burned = wei(burnt)", 0, "an earlier burn is forgotten"),
    ("status_not_stored", ARB, "claims[id].status = transition.status.code()", "", 0, "a bond can be credited again and again"),
    ("clock_plus_one", ARB, "now: now.as_seconds()", "now: now.as_seconds() + 1", 0, "the deadline is off by one"),
    ("deadline_plus_one", ARB, "deadline: record.deadline.as_seconds(),", "deadline: record.deadline.as_seconds() + 1,", 0, "the deadline is off by one"),
    # Withdraw
    ("withdraw_keeps_credit", ARB, "credits[account] = Wei::zero()", "", 0, "a credit can be withdrawn again and again"),
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
    ),
    ("withdraw_ignores_failure", ARB, "value: amount).success() {", "value: amount).success() && false {", 0, "a refused transfer loses the credit"),
    # Views and events
    ("unknown_claim_not_refused", ARB, "if id >= count {", "if false {", 0, "an unused number reads as a claim"),
    ("result_before_final", ARB, "if record.status != Status::Final.code() {", "if false {", 0, "an open or slashed claim's class reads as a result"),
    ("event_n1_wrong_layer", ARB, "layer: 1, neuron: j })", "layer: 2, neuron: j })", 0, "the slash event misreports the layer"),
    ("event_n2_wrong_neuron", ARB, "layer: 2, neuron: j })", "layer: 2, neuron: 0 })", 0, "the slash event misreports the neuron"),
    ("event_class_slashed_missing", ARB, "log.emit(ClassSlashed { id, challenger })", "", 0, "no event for a class slash"),
    ("event_finalized_wrong_class", ARB, "log.emit(Finalized { id, class: record.class })", "log.emit(Finalized { id, class: 0 })", 0, "the event misreports the class"),
    ("event_withdrawn_missing", ARB, "log.emit(Withdrawn { account, amount })", "", 0, "no event for a withdrawal"),
    # The lifecycle rule (pure ingot)
    ("step_terminal_not_checked", STEP, "if !claim.status.is_open() {", "if false {", 0, "a terminal claim moves again"),
    ("step_fraud_not_checked", STEP, "} else if !fraud {", "} else if false {", 0, "every challenge slashes"),
    ("step_challenge_boundary", STEP, "if now >= claim.deadline {", "if now > claim.deadline {", 0, "a challenge at the deadline succeeds"),
    ("step_finalize_boundary", STEP, "if now < claim.deadline {", "if now <= claim.deadline {", 0, "a finalisation at the deadline fails"),
    ("step_slash_pays_claimant", STEP, "bond_to: Payee::Challenger }", "bond_to: Payee::Claimant }", 0, "the challenger is not paid"),
    ("shares_slash_pays_all", STEP, "let paid = bond / 2", "let paid = bond", 0, "a slash burns nothing: a claimant slashing itself loses nothing"),
    ("shares_lose_odd_wei", STEP, "(paid, bond - paid)", "(paid, paid)", 0, "an odd bond loses a wei"),
    ("shares_finalize_burns", STEP, "Payee::Claimant => (bond, 0),", "Payee::Claimant => (bond / 2, bond - bond / 2),", 0, "an honest claimant loses half"),
    # Commitments
    ("logits_hash_first_only", REC, "words[j] = logits[j].to_word()", "words[j] = logits[0].to_word()", 0, "the commitment ignores nine logits"),
    # The kernel
    ("argmax_highest_index_wins", ACC, "if self.raw[j] > self.raw[best] {", "if self.raw[j] >= self.raw[best] {", 0, "ties go to the wrong class"),
    ("clamp_top_is_126", ACC, "            127\n", "            126\n", 0, "activations saturate one low"),
    ("relu_lets_minus_one_through", ACC, "if self.raw < 0 {", "if self.raw < -1 {", 0, "the ReLU misses -1"),
    ("shift_truncates", ACC, "Q::from_raw(self.raw >> SHIFT.downcast_truncate())", "Q::from_raw(self.raw / (1 << SHIFT.downcast_truncate()))", 0, "rounds toward zero instead of down"),
    ("dot_wrong_offset", DOT, "let offset: i64 = (128 * inputs).downcast_truncate()", "let offset: i64 = (127 * inputs).downcast_truncate()", 0, "every dot product is wrong"),
    ("well_formed_no_sign_check", ACT, "if words[k] & SIGN_BITS != 0 {", "if false {", 0, "bytes above 127 pass"),
    ("well_formed_no_padding_check", PAK, "words[N / 32] << ((N % 32) * 8) as u256 == 0", "true", 0, "non-zero padding passes"),
]


def env():
    e = dict(os.environ)
    home = str(Path.home())
    e["PATH"] = f"{home}/.fe/bin:{home}/.foundry/bin:" + e["PATH"]
    e["NO_COLOR"] = "1"
    return e


def run(cmd, cwd, limit=None):
    return subprocess.run(cmd, cwd=cwd, env=env(), text=True, errors="replace", capture_output=True, timeout=limit)


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
        self.stage = WORK / "stage"
        self.fe_out = WORK / "fe-out"
        shutil.rmtree(self.repo, ignore_errors=True)
        self.repo.mkdir(parents=True)
        shutil.copytree(ROOT / "ingots", self.repo / "ingots")
        shutil.copy(ROOT / "fe.toml", self.repo / "fe.toml")
        for sub in ("src", "test", "ref-out", "invariant-out", "fe-out"):
            shutil.rmtree(self.stage / sub, ignore_errors=True)
        self.stage.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "evm" / "src", self.stage / "src")
        shutil.copytree(ROOT / "evm" / "test", self.stage / "test")
        shutil.copy(ROOT / "evm" / "foundry.toml", self.stage / "foundry.toml")
        (self.stage / "ref-out").mkdir()
        (self.stage / "fe-out").mkdir()
        (self.stage / "invariant-out").mkdir()
        for name in ("inputs.bin", "pred.bin", "trace.jsonl"):
            shutil.copy(ROOT / "ref" / "out" / name, self.stage / "ref-out" / name)

    def build(self):
        shutil.rmtree(self.fe_out, ignore_errors=True)
        self.fe_out.mkdir(parents=True)
        return run(["fe", "build", "--color", "never", "--out-dir", str(self.fe_out), "--ingot", "onchain", "."], self.repo)

    def forge(self):
        for f in (self.stage / "fe-out").glob("*"):
            f.unlink()
        for f in self.fe_out.glob("*.bin"):
            shutil.copy(f, self.stage / "fe-out" / f.name)
        log = self.stage / "invariant-out" / "campaign.log"
        if log.exists():
            log.unlink()
        return run(
            ["forge", "test", "--color", "never", "--fail-fast", "--match-path", "test/Arbiter*.t.sol"],
            self.stage,
            FORGE_LIMIT,
        )


def first_failure(output):
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("[FAIL"):
            continue
        reason = re.match(r"\[FAIL:?\s*([^;\]]*)", line).group(1).strip()[:110]
        m = re.match(r"\[FAIL.*\]\s+(\w+)\(", line)
        if m:
            return m.group(1), reason
        # An invariant failure prints its call sequence first and the test's name after it.
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


def write_report(done):
    counts = {}
    rows = []
    for name, file, _, _, _, breaks in MUTANTS:
        r = done.get(name)
        if not r:
            continue
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        by = r.get("by", "")
        rows.append(f"| `{name}` | `{Path(file).name}` | {breaks} | {r['outcome']} | {('`' + by + '`') if by else ''} | {r['seconds']:.0f} |")
    head = [
        "# Mutation test of the Arbiter's Foundry suite",
        "",
        "Generated by `tools/m3_mutants.py`. Each row is one deliberate fault in a copy of the Fe source,",
        "built and run against `evm/test/Arbiter*.t.sol` with `--fail-fast`, so the test named is the",
        "first to fail, not the only one. Discussion: `docs/fe-notes.md` M3.9.",
        "",
        "Totals: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + f" (of {sum(counts.values())}).",
        "",
        "| Mutant | File | What it breaks | Outcome | First failing test | s |",
        "|---|---|---|---|---|---|",
    ]
    REPORT.write_text("\n".join(head + rows) + "\n")


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        only = set(args[args.index("--only") + 1].split(","))
    if "--fresh" in args and RESULTS.exists():
        RESULTS.unlink()
    RESULTS.parent.mkdir(exist_ok=True)
    done = load_done()
    start = time.time()
    bench = Bench()

    t = time.time()
    b = bench.build()
    if b.returncode != 0:
        print("baseline does not build:\n" + b.stdout + b.stderr)
        return 2
    base_hash = sha(bench.fe_out / "Arbiter.runtime.bin")
    f = bench.forge()
    summary = [l for l in f.stdout.splitlines() if l.startswith("Ran ")]
    print(f"baseline: exit {f.returncode}, {summary[-1] if summary else 'no summary'}, {time.time() - t:.1f} s", flush=True)
    if f.returncode != 0:
        print(f.stdout[-3000:])
        return 2

    for name, file, old, new, nth, _ in MUTANTS:
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
        write_report(done)

    write_report(done)
    counts = {}
    for name, *_ in MUTANTS:
        if name in done:
            counts[done[name]["outcome"]] = counts.get(done[name]["outcome"], 0) + 1
    print(f"m3_mutants: {counts}, wall {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
