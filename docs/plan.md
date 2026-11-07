# Tensile: project plan (2026-11-05)

"Tensile" is a working name. Deadline: 2026-11-24 23:59 UTC. Feature freeze: 2026-11-20.

## Pitch

One integer inference kernel, written once in Fe. Tensor shapes and fixed-point scales are checked
by the compiler. Built natively it classifies a test set; built for the EVM it re-runs a single
neuron to settle a dispute about a result posted on-chain. Native, EVM and a NumPy reference agree
bit for bit.

## Parts

| Part | Language | What it is |
|---|---|---|
| `tensile` ingot | Fe (pure, no effects) | `Vector<N>`, `Matrix<R, C>`, scale-typed `Q<F>`, packed int8 rows, dense / ReLU / requantise / argmax kernels, compile-time tables |
| `model` ingot | Fe (generated) | Weights as packed literals plus the typed model definition |
| `cli` ingot | Fe (native) | Classifies samples from stdin, prints classes and activation commitments |
| `arbiter` ingot | Fe (contracts) | Optimistic claims with a bond; a challenger names one neuron, the contract recomputes it from calldata checked against commitments and a Merkle root of the weights |
| `ref/` | Python, NumPy | Training, quantisation, the integer reference, test vectors |
| `tools/` | Python, shell | Model-to-Fe generator, differential test drivers, size and gas checks |
| `test/foundry/` | Solidity | Fuzz and invariant tests against the built bytecode; a hand-written Solidity twin as the gas baseline |
| `demo/` | HTML, JS | Draw a digit, classify it, post a wrong claim and watch the dispute |

Arithmetic and file formats: `docs/spec.md`. Commitments and the dispute protocol: `docs/spec-dispute.md`. What the build has learned about Fe: `docs/fe-notes.md`.

## Milestones

| # | Dates | Deliverable | Done when |
|---|---|---|---|
| M1 | Done 2026-11-05 | Reference model, typed kernel library, native CLI | **Met.** Native equals the NumPy reference on all 10,000 test images at `-O 0`, `-O 1` and `-O 2` (0 mismatches); 14 compile-fail cases pass; integer accuracy 97.44%; `tools/m1_check.sh` takes 124.7 s |
| M2 | Done 2026-11-06 | EVM build of the same kernel, gas and size measurements | **Met.** On-chain `Network` equals the reference on all 10,000 classes and 1,000 traces (0 mismatches); largest contract 12,994 bytes; `classify` 1,896,043 gas; one layer-1 neuron 92,608 gas (Fe) against 74,673 (best hand-written Solidity), both as of `docs/numbers.md`; `tools/m2_check.sh` 183.0 s |
| M3 | Done 2026-11-06 | Arbiter contract and commitment scheme | **Met.** 42-leaf neuron Merkle tree; pure claim state machine (9 tests, both backends); `Arbiter` 11,550 bytes, 26 contract tests; native auditor names the same first failing check as Python on 5,200 mutated claims (0 mismatches); a layer-1 neuron challenge costs 156,380 gas; 45 Foundry fuzz and invariant tests for P1 to P6 pass, no contract bug found; 53 of 53 deliberate faults caught (45 by Foundry, 8 by `static_assert`), `docs/mutation-results.md`; `tools/m3_check.sh` 156.1 s |
| M4 | Done 2026-11-06, except filing | Demo page, Solidity baseline, abstraction-cost table, upstream bug reports | **Met, filing aside.** One HTML file, no install, tested in Chrome offline from `file://`; its chain equals anvil's gas exactly; `tools/m4_numbers.sh` writes `docs/numbers.md`; Solidity arbiter twin passes the same 45 tests; 7 reports drafted in `docs/upstream-reports.md`, waiting for approval to file |
| M5 | Done 2026-11-06, except publishing | Hardening, CI, README on the starter template, limitations, video | **Met as far as one machine allows.** `tools/m5_clean_run.sh` follows the README from an empty home directory: 0 steps failed, 545 s. Not yet a clean machine, and CI has not run on GitHub, because the repository does not exist yet |
| Submit | Nov 21 to 22 | First valid submission | Confirmation email says the automated checks passed |
| Buffer | Nov 23 to 24 | Fixes only, resubmit if needed | |

## Rules for the build

- The NumPy reference is the oracle. Any disagreement is a bug until proven otherwise.
- Never pass a large array by value in a message or twice in one function (compiler defects found
  in the spike: `spike/inference/RESULTS.md`, section 6).
- Weights are packed 32 per `u256` word. Plain `[i8; N]` constants cost 32 bytes of code each.
- Check runtime bytecode size after every build: `fe build` and `fe test` do not.
- Measure gas on a local chain with `cast estimate`; in-test figures overstate large calls.
- Time every full test run and record it.
- Every compiler defect gets a minimal example and goes on the list for upstream reports.

## Known compiler defects to report upstream

Drafts, examples and what was withdrawn: `docs/upstream-reports.md`. Re-check with `tools/upstream_check.sh`.
