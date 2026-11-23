# A fee tier from the pool's own ticks

A liquidity pool wants its swap fee to follow expected volatility: low when the next hours look
quiet, high when they look active. A keeper runs a small network on the pool's last 24 hours of
tick moves and posts "for this window the model says tier K" with a bond; anyone who re-runs the
model and finds a wrong value slashes the bond by re-running one neuron on-chain. The input is not
an off-chain feature: the `FeeOracle` contract here builds it byte for byte from ticks it read from
the pool, and applies a claim only if the claim's committed input hash is the hash of those bytes.

Everything below is measured on one year of the mainnet Uniswap v3 USDC/WETH 0.05% pool
(`data/SOURCE.md`). The generated tables are in `RESULTS.md`; `bash check.sh` rebuilds them and
checks every figure (248 s here, no download, no NumPy).

## What it predicts, and how well

| | |
|---|---|
| Observation | the pool's tick (`slot0`) every 50 blocks, about 10 minutes |
| Input | the last 144 moves, each `min(\|tick - previous tick\|, 127)`: 144 bytes, 24 hours |
| Target | the activity of the 4 hours that start 1 hour after the window ends: the sum of those 24 moves, cut into 3 tiers |
| Why the 1 hour gap | a claim spends its dispute window open; the fee it sets can only apply afterwards |
| Tiers | sum below 261: tier 0 (quiet); below 418: tier 1; else tier 2 (active). The two thresholds are the terciles of the training split, and of nothing else |
| Network | 144 -> 32 -> 16 -> 3, dense, 5,219 parameters, int8 weights, integer arithmetic only (`docs/spec.md`) |
| Split, in time order | first 70% of the year trains, next 15% validates, last 15% (about 55 days) is the test |

On the test split, 7,711 windows, scored once and after every choice was made:

| Predictor | Correct tier | |
|---|---|---|
| **The integer model**, the exact arithmetic the arbiter enforces | 4,399 of 7,711 | 57.05% |
| Persistence: "the next 4 hours are in the tier of the last 4 hours" | 3,779 of 7,711 | 49.01% |
| Majority: always the most common tier of the test split itself | 2,802 of 7,711 | 36.34% |

| Tier | Test windows | Model recall | Persistence recall |
|---|---|---|---|
| 0, quiet | 2,758 | 1,976 (71.65%) | 1,697 (61.53%) |
| 1, middle | 2,802 | 1,359 (48.50%) | 1,155 (41.22%) |
| 2, active | 2,151 | 1,064 (49.47%) | 927 (43.10%) |

**The model beats persistence**, by 620 windows (8.04 percentage points) on the test split and by
370 of 7,711 (4.80 points) on the validation split, and it beats it in every tier. Read that with
these in mind:

- It is still wrong for 43% of the windows, and for half of the active ones. It is off by two
  tiers for 494 windows (6.41%); persistence for 652.
- The windows overlap (a new one starts every 10 minutes and each spans 29 hours), so 7,711
  windows are far fewer independent trials. No confidence interval was computed.
- Validation accuracy (61.57%) is higher than test accuracy (57.05%): the validation split chose
  the epoch and the scales, and the two periods differ (persistence also drops, from 56.78% to
  49.01%).
- One pool, one year, one seed. Persistence is a weak baseline for a target with a daily rhythm;
  a 24-hour input can see "this time yesterday" and persistence cannot. A stronger baseline
  (for example an exponentially weighted average with a time-of-day term) was not tried.
- Whether a fee set this way earns liquidity providers more than a fixed fee is not measured
  anywhere here. The tiers' fees (0.05%, 0.30%, 1.00%) are illustrative.

How the shape was chosen (validation only, `select.py`, `runs/select.txt`): input lengths of 96,
144 and 192 moves, hidden layers 32-16 and 64-32, two learning rates. 144 moves with 32-16 was
kept; 64-32 scored 30 windows more on validation and was not taken, as within noise. Earlier runs
on the part of the series that had been downloaded at the time (validation only, not recorded)
also tried other horizons and input lengths and set that list. Training is
`train.py` (NumPy, seed 20261124, 30 float epochs, post-training quantisation; `runs/train.txt`);
quantisation changed validation accuracy from 4,738 to 4,748 correct.

## What the dispute costs at this size

Real transactions on anvil (Osaka), generated arbiter, each result checked. From `RESULTS.md`:

| | This model, 5,219 parameters | 784 -> 32 -> 10, 25,450 | 784 -> 128 -> 64 -> 10, 109,386 |
|---|---|---|---|
| Whole forward pass on-chain, gas | 466,408 | 1,896,043 | 8,219,318 |
| `claim`, gas | 239,460 | 246,113 | 274,284 |
| Honest path, `claim` + `finalize`, gas | 290,511 | 295,147 | 325,255 |
| the same, share of a whole run | 62.3% | 15.6% | 4.0% |
| Dearest dispute (one layer 1 neuron, first slash), gas | 116,791 | 178,662 | 185,003 |
| the same, share of a whole run | 25.0% | 9.4% | 2.3% |

**Unflattering, and the main thing to know: a model this small does not need the dispute.** The
honest path costs 62.3% of simply running the network on-chain, and once the consumer's `apply`
(167,967 gas the first time, 113,934 after) is added the claim route costs 404,445 gas per
decision against 466,408 for calling `Network.classify` directly: 13% less, in exchange for a
bond, a dispute window and a watcher. (The direct route would also have to store the fee; that
was not measured.) The mechanism pays when the model is larger (the two columns on the right),
because a claim grows with the layer widths and a whole run with the number of weights. This example is here for the other half of the argument: a use where the input is
on-chain data, the answer is consumed by a contract, and the model is better than the obvious
rule. It does not show that a larger fee model would predict better.

## The consumer: `FeeOracle` (`ingots/consumer`, hand-written Fe, 8 tests)

- `observe()` reads `slot0()` of the pool at most once per `interval` blocks and stores one byte,
  the clipped move, 32 to a storage word. Anyone may call it.
- `inputWords(end)` returns the packed 144 moves that end at move `end`: what a keeper runs the
  model on.
- `apply(id, end)` hashes that window, requires `claimHashes(id)` of the arbiter to carry the same
  input hash, and reads `result(id)`, which reverts unless the claim is final. Then `fee()`
  returns the tier's fee for the 30 intervals the claim covers (the gap and the horizon), and the
  default fee after that, so a keeper that stops only costs the pool its dynamic fee.
- Its effects are `ctx: Ctx, call: Call, log: mut Log`: it can read other contracts and write its
  own storage, and cannot send ETH or change the pool or the arbiter.

`oracle_check.sh` (part of `check.sh`) runs it on anvil with an interval of 50 mined blocks: a
test pool answers `slot0()` with the 145 real ticks behind test sample 11, and the packed
input the contract builds must equal that sample in `out/inputs.bin`, the bytes the model was
exported on. The honest claim is then finalised and applied, and `fee()` goes from 3,000 to 500;
the next window, one observation later, is checked the same way. Gas: `observe()` 51,685 (median
of 145), so the 24 observations of one decision cost about 1,240,440 gas, three quarters of
everything a decision costs and four times the claim. Unflattering: `observe` is written to be
read, not to be cheap (five storage writes and a call to the pool each time); what a hook that
already holds the tick would pay was not measured.

`fork_check.sh` (by hand, not in CI, needs a mainnet RPC endpoint) deploys the same contract on a
fork of mainnet and calls `observe()` on the real pool. Run once, on 2026-11-24, on a fork at
block 26,479,143: the pool's `slot0()` tick was 201,576 and the contract decoded 201,576.

## Run it

```
bash examples/amm-fee/check.sh                       # everything, about 4 minutes; Fe 26.5.1, Foundry, Python 3
PYTHON_NUMPY=python3 bash examples/amm-fee/check.sh  # also re-runs export.py and compares out/ byte for byte
python examples/amm-fee/train.py && python examples/amm-fee/export.py     # retrain (NumPy); writes model/ and out/
python -I examples/amm-fee/fetch_data.py --out DIR   # download the series again (about 30 minutes, public RPC)
```

| File | What |
|---|---|
| `data/ticks.csv`, `data/SOURCE.md` | the series (52,561 ticks, 368 kB) and where it came from |
| `dataset.py` | the task: input rule, target, tiers, split, scores. Standard library, integers only |
| `fetch_data.py`, `select.py`, `train.py`, `export.py` | download, shape selection on validation, training, export and the one scoring of the test split |
| `model/`, `out/` | the trained integer model; its reference output on the test split (inputs and labels are rebuilt from the ticks) |
| `ingots/model`, `arbiter`, `cli`, `network` | generated by `tools/gen_workspace.sh`, never edited |
| `ingots/consumer` | `FeeOracle` and its tests, hand-written |
| `check.sh`, `oracle_check.sh`, `fork_check.sh`, `results.py` | the checks, and the generator of `RESULTS.md` |

## Limits

- **The arbiter proves the computation, not the forecast.** A finalised claim says "the committed
  network gives tier K for this input". Whether tier K was the right fee is the model's business;
  the scores above are all there is on that.
- **One honest watcher is needed**, awake inside every dispute window, who recomputes each claim
  (`docs/USAGE.md` section 5). An unwatched wrong claim becomes the fee.
- **The fee is late by design.** The window is observed, then the claim waits out its dispute
  window; the model was trained with a gap of 6 intervals (one hour) for that. A longer window
  needs a model trained with a longer gap.
- **Keeper liveness.** Somebody must call `observe()` every interval and post a claim every
  decision. A late `observe()` stretches an interval and the move it records is no longer a
  10-minute move; nothing in the contract corrects for that. With no fresh claim the fee is the
  default.
- **The observation is a spot tick.** One trade in the observed block moves it, and whoever can
  move it can push the input, and so the tier. A time-weighted tick (`observe()` of the v3
  oracle) would be the sturdier input; the model was not trained on one.
- **Who posts and who may apply is open.** Any finalised claim about the right window is accepted;
  there is no keeper registry, no reward for the keeper, and the bond size is the arbiter's.
- **Not a Uniswap v4 hook.** No hook is written or tested. `FeeOracle` reads the v3 `slot0()`
  view from an address it is given; the real pool was called once, on a fork, by hand.
- **The generated code's limits** are in `docs/USAGE.md` section 6; the dispute design's in
  `docs/spec-dispute.md` section 7. Not audited.
- The model, the data handling and this text were written with AI assistance (see the
  repository README's disclosure).
