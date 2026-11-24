# A fee tier from 48 hours of ticks, with 1D convolutions

The task of [`examples/amm-fee`](../amm-fee/README.md), the same windows, labels and test split,
with a convolutional network over twice the history. It is here to show one thing: **a network
whose whole run on-chain does not fit in a transaction, and whose every value can still be
disputed in one small transaction**, because a convolution's output depends on one filter and
one short window of its input. The dispute names `(layer, filter, position)` and re-runs that
filter on that window with `tensile::filter_at`, the same function the forward pass runs at every
position.

Everything below is measured. The generated tables are in `RESULTS.md`; `bash check.sh` rebuilds
them and checks every figure (441.8 s in its last run here; no download, no NumPy).

## The model

| | |
|---|---|
| Input | the last 288 moves of the mainnet USDC/WETH 0.05% pool, each `min(\|tick - previous tick\|, 127)`: 288 bytes, 48 hours (`../amm-fee/data/ticks.csv`) |
| Target, tiers, split | `examples/amm-fee`'s, unchanged: the same 7,711 test windows with the same labels and the same thresholds (the terciles of that example's training split) |
| Layer 1 | conv1d, 1 channel -> 16 filters of kernel 8, stride 1: 281 positions x 16 = 4,496 values |
| Layer 2 | conv1d, 16 -> 16 filters of kernel 4, stride 4: 70 positions x 16 = 1,120 values |
| Layer 3 | conv1d, 16 -> 16 filters of kernel 4, stride 4: 17 positions x 16 = 272 values |
| Layers 4, 5 | dense 272 -> 32 -> 3 |
| Parameters | 11,059 int8 weights and int32 biases; integer arithmetic only (`docs/spec.md` sections 3 and 8) |

ReLU (the clamp to 0..127) after every layer but the last; no padding and no pooling, the strides
downsample. The shape was fixed for what this example demonstrates, not searched for accuracy;
`tune.py` chose only the learning rate and weight decay, on the validation split
(`runs/tune.txt`). Training is `train.py` (NumPy, seed 20261124, 12 float epochs, post-training
quantisation with a scale search on validation; `runs/train.txt`).

## How well it predicts: no better than the dense model

On the test split, scored once by `export.py` after every choice was made:

| Predictor, the same 7,711 windows | Correct tier |
|---|---|
| **This model** (288 moves, convolutional) | 4,374 (56.72%) |
| The dense 144-32-16-3 model of `examples/amm-fee` (144 moves) | 4,399 (57.05%) |
| Persistence: the tier of the last 4 hours | 3,779 (49.01%) |
| Majority: always the test split's most common tier | 2,802 (36.34%) |

**It does not beat the dense model**: 25 windows fewer, within the noise of overlapping windows.
Twice the history and three convolutions bought nothing on this task. It still beats persistence by
595 windows. Keep it as the demonstration of the mechanism, not as a better fee model.

## What the dispute costs

Real transactions on anvil, sample 0, each result checked. From `RESULTS.md`:

| | This model, 11,059 parameters | Dense 144-32-16-3, 5,219 |
|---|---|---|
| Whole forward pass on-chain (`Network.classify`), gas | 17,966,815 | 466,408 |
| `Network` runtime code (EIP-170 limit 24,576 bytes) | 70,261 bytes | 7,832 bytes |
| `claim`, gas | 617,131 | 239,460 |
| Honest path, `claim` + `finalize`, gas | 672,404 (3.7% of a whole run) | 290,511 (62.3%) |
| A filter challenge (later slash), gas | 91,832 to 107,674 | |
| Any challenge (later slash), gas | 62,940 to 110,306 (0.4% to 0.6% of a whole run) | 58,503 to 82,591 |
| `Arbiter` deployed bytes | 22,565 | 10,814 |

**The whole run does not fit.** It needs 17,966,815 gas, 1.07 times the 16,777,216 an Osaka
transaction may use (EIP-7825): anvil started with `--enable-tx-gas-limit` refused a
`classify` transaction with a 30,000,000 gas limit, and the figure was measured on Prague, where
`classify` was also checked against the reference on 20 samples. Its contract, 70,261 bytes, is
also 2.86 times the size a contract may have, so it could only be measured on an anvil with the
size limit turned off. **Unflattering:** both are facts about the generated contract, not about
the model. At the dense kernel's measured 75.3 gas per multiply-accumulate (784-128-64-10), this
model's 133,856 would cost about 10.1 million gas, a projection under the cap; the generated
convolution pays 134.2 per multiply-accumulate because layer 1 is 4,496 calls of 8 values each.

**Every value is still disputed in one small transaction.** The dearest filter challenge (107,674
gas) costs 1.61 times the smallest neuron challenge of the dense fee model (66,995), while the
whole run costs 38.5 times as much. The claim is the dear part: it carries every hidden vector
(6,372 bytes of calldata) and builds a word tree of each.

## How a claim commits to a long vector: the word tree

A dense challenge supplies the whole layer input and the contract hashes it against the claim.
For a convolution that means shipping 4,496 values to re-run a filter that reads 64 of them. So a
claim about a model with a convolution commits to each vector as a Merkle tree over its 32-byte
words (a leaf binds the word to its index), and a filter challenge opens only the 2 or 3 words
its window covers and the one output word that holds the value. Both rules were generated and
measured on this model in the same run (`RESULTS.md`, section 3); the soundness argument is
`docs/spec-dispute.md` section 11.

## Run it

```
bash examples/amm-fee-conv/check.sh                       # everything; Fe 26.5.1, Foundry, Python 3
PYTHON_NUMPY=python3 bash examples/amm-fee-conv/check.sh  # also re-runs export.py and compares out/ byte for byte
python examples/amm-fee-conv/train.py && python examples/amm-fee-conv/export.py   # retrain (NumPy)
```

| File | What |
|---|---|
| `dataset.py` | the task: `../amm-fee/dataset.py` with a 288-move input. Standard library |
| `tune.py`, `train.py`, `export.py` | settings on validation, training, export and the one scoring of the test split |
| `model/`, `out/` | the trained integer model; its reference output on the test split (inputs and labels are rebuilt from the ticks) |
| `ingots/model`, `arbiter`, `cli`, `network` | generated by `tools/gen_workspace.sh`, never edited |
| `check.sh`, `results.py` | the checks, and the generator of `RESULTS.md` |

## Limits

- **1D only.** No 2D convolution, no padding, no pooling (`docs/spec.md` section 8).
- **The claim is the dear part**: it carries every hidden vector, 5,888 values here, and computes a
  word-tree root of each on-chain. A longer sequence makes the claim, not the dispute, grow.
- **No consumer contract.** Unlike `examples/amm-fee` there is no `FeeOracle` here: the
  contract-side input would be 288 stored moves instead of 144, and nothing else about it changes.
- **The model is not better** than the dense one (above), and everything said about the task's
  limits in `examples/amm-fee/README.md` applies.
- **The word-tree binding is covered by this example's generated contract tests and anvil run**,
  not by the Foundry fuzz suite or the mutation test, which cover the dense two-layer arbiter.
- The model, the data handling and this text were written with AI assistance (see the repository
  README's disclosure).
