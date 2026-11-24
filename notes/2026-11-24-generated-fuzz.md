# 2026-11-24: fuzz, invariant and mutation evidence for the generated three-layer arbiters

Before this, only the shipped two-layer arbiter was under the Foundry suite and the mutation run;
the three-layer arbiters that `tools/gen_arbiter.py` writes for the examples had their generated
Fe tests and anvil scripts only.

## What was added

- `evm/generated/`: the shipped suite (`evm/test/Arbiter*.t.sol`) with the sizes taken out, plus
  tests only a middle layer needs. 50 tests per example: 22 in `GenArbiterFuzz.t.sol` (P2, P3,
  Merkle, and the "Layers" tests: a committed vector of another layer, a leaf of another layer,
  a lie carried forward, every index just past each layer), 24 in `GenArbiterBounds.t.sol` (P5
  deadlines, P6 well-formedness, ABI edge cases, events, withdraw, re-entrancy, forced ETH), 4 in
  `GenArbiterInvariant.t.sol` (P1 and P4 under a 256 x 100 campaign, a scripted run that reaches
  every outcome, two negative controls).
- The oracle is `GModel` in `GenBase.sol`: the integer network in plain Solidity over the rows of
  a fixture file that `tools/gen_arbiter_fixtures_any.py` writes from the example's `model.json`,
  its committed `trace.jsonl` and its neuron tree. It shares no code with the Fe kernel, the model
  ingot or the generator; the fixture writer checks every sample against its own Python forward
  pass, and `test_oracle_reproduces_the_reference_samples` checks GModel against the trace.
- `tools/gen_foundry.sh` (per example), `tools/gen_mutants.py` (amm-fee, 81 mutants),
  `tools/gen_checks.sh` (all of it, with timings), and the `generated-arbiter-fuzz` job in
  `.github/workflows/examples.yml` (amm-fee and address-risk; the mutation run stays local, as
  `tools/m3_mutants.py` does).

## Runs (WSL2, Fe 26.5.1, Foundry 1.9.0)

| Command | Result | Wall |
|---|---|---|
| `tools/gen_checks.sh` | amm-fee 50/50, address-risk 50/50 (full counts), mnist-784-128-64-10 50/50 (GEN_QUICK: 64 fuzz cases, invariant 32 x 50) | 313.3 s |
| `tools/gen_mutants.py --fresh` | 81 mutants: 78 killed, 2 compiler, 1 survived (equivalent, argued in `docs/mutation-results-generated.md`) | 1587.4 s |
| `tools/gen_foundry.sh amm-fee address-risk` on a clean `git archive` export (the CI job's command) | 100/100 | 215.3 s |
| `tools/check_arbiter_gen.sh --bytecode` | source and bytecode identical | 28.2 s |
| `tools/m3_foundry.sh` | the shipped arbiter's 45 tests pass | 105.0 s |

No defect was found in the generator or the generated contracts. The one survivor reads the middle
layer's output with the first hidden layer's type; both are one packed word and the index and
padding are checked before the read, so no input tells them apart.

Not verified: the MNIST arbiter at the full fuzz and invariant counts; the mutation run on any
example other than amm-fee; the CI job on GitHub itself (simulated locally from a clean export).
