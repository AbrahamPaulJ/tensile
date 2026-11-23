# Address risk: a scam-address score a contract can act on

A third model through the pipeline of `docs/USAGE.md`, and the first that is not an image
classifier: 14 -> 32 -> 16 -> 2, trained on a public data set of Ethereum accounts labelled as scam
addresses or not.

**The user story.** A pool wants to refuse accounts that behave like known scam addresses, and it
wants the rule to be a published model anyone can re-run, not a private blocklist. An operator
computes 14 features from an account's public transaction history, runs the model, and posts the
answer to the arbiter with a bond; anyone who re-runs the same integer network and gets another
answer proves it with one transaction. `RiskGate` (`ingots/consumer`) reads the finalised answer
and lets only cleared accounts in.

**What was measured, in one paragraph** (all of it from `RESULTS.md`, which `check.sh` generates).
On 7,054 held-out accounts the integer model is right on 89.35%, where answering "not flagged" every
time is right on 83.37%; of the accounts it flags, 72.79% are flagged in the data (precision), and
it finds 57.46% of the flagged ones (recall). The native build and the contracts agree with the NumPy
reference on every one of those rows. And the unflattering part: this network is so small that
running it whole on-chain costs 154,748 gas, while claiming and finalising an answer costs 281,361.
**At this size the dispute protocol does not pay; call the generated `Network` contract instead.**
The arbiter starts to pay at a few thousand parameters (a projection, `RESULTS.md` section 2).

## The data

| | |
|---|---|
| Data set | FuseChain Multimodal Ethereum Fraud Dataset, Nileshka Fernando, 2026 |
| Source | https://huggingface.co/datasets/Nileshka/fusechain-data, revision `ef382e837540182c483ea2c2fbb6259c42f0633a`, file `address_level_fused.parquet` |
| Licence | MIT (the data set card's `license: mit`) |
| Built on | Ethereum Fraud Dataset by Activity, Ishmukhamedov, 2025, https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity, CC BY 4.0, itself from the public BigQuery Ethereum tables, Etherscan labels and curated scam lists |
| Downloaded | 2026-11-23 (UTC), without a login |
| SHA-256 | `294bf2a663773e56d50cb90296cfb0b83f5067255ff61a97e850b0fbd58454c9` (7,247,469 bytes) |
| Content | 35,272 externally owned accounts, 5,742 (16.3%) labelled scam; activity between 2017-01-01 and 2021-02-10 |
| Used here | the label and the 14 on-chain columns. The 17 market, Reddit and Twitter columns are not used: they are not properties of an account |

The raw file is not committed. What is committed is derived from it: the feature thresholds
(`features.json`), the trained model (`model/`), and the test split as the network sees it, 14 bytes
per account with its label (`out/inputs.bin`, `out/labels.bin`), without the addresses.

## How an account becomes 14 bytes

The columns, as the data set names them, in the order the network reads them:

`active_days`, `normal_sent_cnt_mean`, `normal_sent_cnt_max`, `normal_recv_cnt_mean`,
`normal_recv_cnt_max`, `normal_total_cnt_mean`, `normal_total_cnt_max`, `eth_sent_sum_mean`,
`eth_net_flow_mean`, `eth_net_flow_max`, `sessions_cnt_mean`, `burst_max_tx_5m_mean`,
`burst_max_tx_5m_max`, `days_since_last_activity_mean`.

By the data set's card these are per-day figures of an account's ordinary transactions (counts sent
and received, ETH sent, net ETH flow, sessions, the most transactions in five minutes, days since the
previous activity), aggregated over the account's active days as a mean or a maximum. We took the
columns as published and did not re-derive them from chain data; the pipeline that made them is the
data set author's (linked from its card).

Each value becomes an activation 0..127 by one rule (`features.py`): **the number of thresholds that
are less than or equal to the value.** Every feature has 127 thresholds, the 1/128 to 127/128
quantiles of that column over the 21,164 training rows, written out as numbers in `features.json`.
So the rule is a table lookup that anyone can repeat, and it needs nothing from the validation or
test rows. Heavy-tailed columns (a few accounts with hundreds of thousands of transactions) and
signed ones (net flow) need no special case.

## Run it

```
bash examples/address-risk/check.sh        # needs Fe 26.5.1, Foundry, Python 3; about two and a half minutes
```

`check.sh` uses only what is committed and the standard library, and `RESULTS.md` records the wall
time of each of its steps. Where `python3` has NumPy, or `PYTHON_NUMPY` names an interpreter that
has, it also re-runs the reference on `out/inputs.bin`. To rebuild the
committed files from the data set (NumPy; `prepare.py` alone also needs pyarrow):

```
python examples/address-risk/prepare.py     # downloads the parquet file, checks its SHA-256, writes data/onchain.npz
python examples/address-risk/train.py       # features.json and model/address-risk-14-32-16-2.json, 31 s
python examples/address-risk/export.py      # out/: model.json, inputs.bin, labels.bin, pred.bin, trace.jsonl
bash tools/gen_workspace.sh examples/address-risk/out/model.json examples/address-risk/out examples/address-risk 16 network
```

Training is deterministic for its seeds on one machine; another CPU or BLAS build may round
differently and give a slightly different model. The committed model is the one every figure here
belongs to.

| File | What |
|---|---|
| `features.py`, `features.json` | the columns, the split, the quantisation rule and its thresholds |
| `prepare.py`, `train.py`, `export.py` | data set to table, table to integer model, model to reference vectors |
| `model/`, `out/` | the trained model with its training record; the reference vectors and metrics |
| `ingots/model`, `arbiter`, `cli`, `network` | generated by `tools/gen_workspace.sh`, never edited |
| `ingots/consumer` | `RiskGate`, hand-written, 7 tests |
| `gate_anvil.sh` | the use on a local chain: claim, finalize, score, enter, and every refusal |
| `check.sh`, `results.py`, `runs/`, `RESULTS.md` | the checks, their records, and the report made from them |

## Limits

- **The arbiter proves a computation, not a verdict.** A final claim says: the committed integer
  network, run on these 14 bytes, gives this class. It says nothing about whether the class is right.
  The model is wrong on 751 of 7,054 test accounts. The demonstration itself shows it: reference
  sample 6, the account `gate_anvil.sh` has refused, is flagged by the model and not by the data set,
  one of its 252 false positives.
- **Nobody checks that the 14 bytes belong to the account.** Computing features from transaction
  history happens off-chain. `RiskGate` checks that a claim is final and is about the features it is
  shown, and takes the operator's word that those are the account's features; it publishes account
  and features together in its `Scored` event so that anyone can recompute them, and punishes nothing.
  To be trustless that step needs its own check (features proved from chain data, or a second dispute
  about the derivation). That is not built here.
- **The input must be public,** as in every Tensile dispute: a claim publishes its features. A risk
  model over private data (a credit file) cannot use this as it stands.
- **At this size, do not use the arbiter.** See the paragraph at the top and `RESULTS.md` section 2.
- **The data is old and its labels are noisy.** Activity ends in February 2021; scam patterns since
  (token approvals, NFT and bridge fraud) are not in it. Labels come from Etherscan tags and
  community lists, so unreported scam accounts count as clean.
- **The 16.3% scam rate is the data set's, not the chain's.** Its source is a list assembled around
  labelled scam addresses, and Ethereum has hundreds of millions of accounts, so among all accounts
  the rate is far lower. Precision falls with the base rate: the share of flagged accounts that are
  truly scams would be well below the 72.79% measured here. We did not measure it.
- **How the clean accounts were chosen is not ours to vouch for.** If they differ from scam accounts
  in ways that come from the sampling (for example in when they were active), the model may have
  learned the sampling. We did not test for that.
- **A gate like this refuses real people.** Of the 926 test accounts the model flags, 252 are not scam
  accounts in the data. Anything built on it needs an appeal path; this example has none.
- **One operator, one model.** `RiskGate` names its operator and arbiter at deployment; nothing is
  upgradable, and a retrained model is a new arbiter and a new gate.
- **Evidence for a three-layer arbiter** is its generated contract tests and the anvil scripts, not the
  Foundry fuzz suite or the mutation run, which cover the two-layer arbiter (`docs/USAGE.md`, section 6).

## Credits and AI assistance

Data: Nileshka Fernando, *FuseChain: Ethereum Fraud Detection via Multimodal Signal Fusion*, 2026
(MIT), built on Ishmukhamedov's *Ethereum Fraud Dataset by Activity*, 2025 (CC BY 4.0). The scripts,
the contract and this page were written with Claude (Anthropic) through Claude Code, as the rest of
the repository (`README.md`, Declaration).
