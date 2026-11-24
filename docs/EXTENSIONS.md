# Extensions: beyond the dense digit model

The README covers the core: one typed kernel and a one-neuron dispute, shown on a digit classifier. This
page holds what was built on that core afterwards. Each part states its own evidence and its own limits;
the convolution and tree arbiters are newer than the dense one and are not under the fuzz and mutation
runs.

- [Convolutions](#convolutions-a-model-that-cannot-run-in-one-transaction-still-disputed-in-one)
- [Tree ensembles](#tree-ensembles-the-same-dispute-for-the-models-people-use-on-tabular-data)
- [The kernel as a library](#use-the-kernel-as-a-library)
- The fee tier running on Sepolia on mainnet data: [`sepolia-fee.md`](sepolia-fee.md)

## Convolutions: a model that cannot run in one transaction, still disputed in one

`ingots/tensile/src/conv.fe` adds 1D convolution in the same typed style. A filter is a `Row<K*C, FW>`,
one output is the dense `neuron` applied to a window of the packed input, and the output length is a
const generic tied to the shape by a `where` clause, so a result declared with the wrong length does not
compile. The overflow proof is the dense one over the filter rows.

A dispute re-runs one filter at one position. To do that without the challenger resending a whole layer,
each vector in a conv claim is committed as a Merkle tree over its 32-byte words, built by the contract
itself from words it has range-checked; a challenge opens only the few input words the window covers and
the one output word (`docs/spec-dispute.md` section 11, with the soundness argument). The flat design was
built and measured too: its filter challenges cost up to 208,130 gas against 107,674, and its arbiter is
27,394 bytes, over the contract size limit, so it cannot be deployed.

[`examples/amm-fee-conv`](../examples/amm-fee-conv/README.md) is the fee-tier task with a convolutional front
end over 48 hours of ticks (three conv layers, then dense 272-32-3; 11,059 parameters):

| | Conv model | Dense 144-32-16-3 |
|---|---|---|
| Correct on the 7,711 held-out windows | 4,374 (56.72%) | 4,399 (57.05%) |
| Whole forward pass on-chain, gas | 17,966,815 | 466,408 |
| `claim`, gas | 617,131 | 239,460 |
| Honest path (`claim` + `finalize`), gas | 672,404 | 290,511 |
| Honest path against one whole run | 3.7% | 62.3% |
| Dearest dispute (first slash), gas | 126,032 | 116,791 |

The whole run needs 1.07 times the 16,777,216 gas one Osaka transaction may use, and anvil with the cap
enforced refuses it (the figure is measured under Prague's rules); the generated `Network` is also 70,261
bytes, too large to deploy as one contract. So this model cannot be run on-chain at all as it stands, and
every value in it can still be disputed for 62,940 to 126,032 gas. What it costs: the claim is 2.6 times
the dense one, because the contract builds the word trees. And the conv model is not better: 25 windows
worse than the dense one, within noise. It is here as the demonstration of the mechanism, not as a better
fee policy. Not covered by the fuzz and mutation runs; its evidence is 22 generated contract tests, the
anvil runs, and 0 mismatches between the native build and the NumPy reference on 7,711 samples and
592,400 traced values. No padding, no pooling, no 2D.

## Tree ensembles: the same dispute for the models people use on tabular data

`ingots/trees` is a second kernel on the same types, for gradient-boosted trees and random forests:
`Tree<DEPTH, IN, LF>` and `Forest<TREES, DEPTH, IN, CLASSES, LF>`, with the no-overflow proof of the score
sum in a `where` clause. A claim commits to the input and to every tree's output. One challenge names a
tree, and the contract re-walks it (6 comparisons at depth 6) against a Merkle proof of that tree, with the
same `Tree::eval` the off-chain forest runs; the other re-adds the committed outputs and compares the
class. If any tree's output is wrong the first succeeds; if all are right the class follows from them, so
a wrong class falls to the second. `ingots/dispute`, the claim state machine, is reused unchanged. Spec:
[`docs/spec-trees.md`](spec-trees.md).

On the address-risk data ([`examples/address-risk-trees`](../examples/address-risk-trees/README.md)), a
forest of 300 trees of depth 6, chosen on validation and scored once on the same 7,054 held-out accounts:

| | Forest 300x6 | Dense 14-32-16-2 | Always "not flagged" |
|---|---|---|---|
| Accuracy | 91.22% | 89.35% | 83.37% |
| Precision | 79.59% | 72.79% | |
| Recall | 63.51% | 57.46% | |
| F1 | 0.7065 | 0.6422 | |
| Whole model on-chain, gas | 1,301,202 | 154,748 | |
| Honest path (`claim` + `finalize`), gas | 251,287 | 281,361 | |
| Honest path against one whole run | 19.3% | 181.8% | |
| Dearest dispute, gas | 118,880 | 101,727 | |
| Dearest dispute against one whole run | 9.1% | 65.7% | |

So on this data the better model is also the one the dispute pays for: the forest is too dear to run on
every call and cheap to claim, where the small dense network was the reverse. The unflattering rows: a
tree dispute costs more gas than a small neuron's (1,316 bytes of calldata against 452), the native
classifier is slow (20 to 25 s for 7,054 rows), and the tree arbiter has its generated Fe tests and anvil
scripts but no fuzz suite or mutation run. Choosing a whole tree per Merkle leaf over one path rests on a
calculation, not on having built both (spec section 10). One compiler limit was met and recorded
(`tests/defects/where_after_substitution.fe`).

## Use the kernel as a library

Under the inference sits a small fixed-point library for Fe: values whose binary scale is part of the
type (`Q<F>`), int8 vectors packed 32 to a word, a dot product whose overflow is ruled out by a `where`
clause, and bounds the compiler checks. `ingots/tensile` has no dependencies and works as a path or git
dependency:

```toml
tensile = { source = "https://github.com/AbrahamPaulJ/tensile", rev = "<commit>", path = "ingots/tensile" }
```

[`lib-examples/`](../lib-examples/README.md) builds three contracts on it with no neural network in sight:
`TickEma` (a moving average of a pool's tick), `Scorecard` (a weighted score over 16 features) and
`RateCurve` (a kinked borrow rate). Six mistakes that compile in Solidity are compile errors there, with
the compiler's message recorded for each (`lib-examples/compile_fail/`): a missing shift after a
fixed-point multiply is "expected `Q<16>`, but `Q<28>` is given", and a rate curve that would overflow at
100% utilisation fails a `static_assert`. Measured against plain-integer twins written in Fe, the types
cost no gas where the code has the same shape (execution gas: `RateCurve` 472 against 472, `TickEma` 7,634
against 7,751), and the kernel's packed dot product runs the scorecard for 1,559 against 4,691 for a
byte-by-byte loop (`lib-examples/runs/gas.json`).
The guide, [`docs/LIBRARY.md`](LIBRARY.md), starts from an empty directory, and CI runs that
walkthrough on Fe 26.5.1 and 26.7.0. API reference, generated by `fe doc`:
[abrahampaulj.github.io/tensile/api](https://abrahampaulj.github.io/tensile/api/). The example parameters
are made up; the library is not audited.
