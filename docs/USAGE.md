# Using Tensile with your own model

From a model file to a tested, deployed dispute arbiter in five commands. Every command below was
run as written, on `examples/tiny/model.json` (40 -> 12 -> 8 -> 4, made-up weights, a model none of
the tools knows): `bash examples/tiny/check.sh` runs sections 2 and 3 in 96 s. A real second model,
MNIST 784 -> 128 -> 64 -> 10, is in `examples/mnist-784-128-64-10/` with its measurements in
`RESULTS.md` there.

Needs Fe 26.5.1, Foundry and Python 3. Only step 1 needs NumPy. Run from the repository root, on
Linux or in WSL.

## 1. The model file

JSON, as in `docs/spec.md` section 5: integers only, already quantised.

```json
{"name": "tiny-40-12-8-4", "input_frac_bits": 7, "layers": [
  {"in": 40, "out": 12, "weight_frac_bits": 7, "shift": 9, "weights": [[..40 ints..], ..12 rows..], "bias": [..12 ints..]},
  {"in": 12, "out": 8,  "weight_frac_bits": 6, "shift": 7, "weights": [..], "bias": [..]},
  {"in": 8,  "out": 4,  "weight_frac_bits": 6, "shift": null, "weights": [..], "bias": [..]}]}
```

Weights are int8 in -127..127, biases int32 at the scale of a weight times an input, inputs and
hidden activations 0..127. A hidden layer is `clamp(acc >> shift, 0, 127)`; the last layer has
`"shift": null` and its accumulators are the logits. `examples/mnist-784-128-64-10/train.py` shows
one way to get such a file from float training (post-training quantisation, any number of layers).

Any layer but the last may be a 1D convolution without padding (`docs/spec.md` section 8):

```json
{"kind": "conv1d", "in": 288, "out": 4496, "channels": 1, "kernel": 8, "stride": 1,
 "weight_frac_bits": 7, "shift": 8, "weights": [..16 filters of kernel * channels ints..], "bias": [..16 ints..]}
```

`in` is positions times channels, `out` is output positions times filters, and the values run
position by position. The generated arbiter then has a `challengeFilterK` for that layer, which
names a filter and a position and re-runs that filter on that one window, and the claim commits to
its vectors with word trees (`docs/spec-dispute.md` section 11). `examples/amm-fee-conv` is a
worked example, from training to the dispute's gas.

## 2. Generate, test, build

```
python3 tools/ref_vectors.py --model examples/tiny/model.json --random 200 --seed 1 --out build/tiny/out
bash tools/gen_workspace.sh build/tiny/out/model.json build/tiny/out build/tiny 3 network
cd build/tiny
fe test && fe build --out-dir build/out && bash ../../tools/check_size.sh build/out
fe build --backend native -O 1 --out-dir build/native --ingot cli . && python3 ../../tools/diff_native.py --ref out build/native/cli
```

1. `ref_vectors.py` runs the NumPy reference (`ref/tensile_ref.py`, the oracle) on your inputs
   (`--inputs file.bin`, one byte per activation) or on random ones, and writes `model.json` with the
   worst-case accumulator of every layer, `inputs.bin`, `pred.bin` and `trace.jsonl`.
2. `gen_workspace.sh` writes an Fe workspace that depends on this repository's `tensile` and
   `dispute` ingots by relative path:

   | Ingot | From | Holds |
   |---|---|---|
   | `model` | `tools/gen_fe.py` | packed weights, the typed layer chain, one overflow proof per layer, the first 3 reference vectors as unit tests |
   | `arbiter` | `tools/gen_arbiter.py` | the `Arbiter` contract, its claim record, events and errors, and contract tests |
   | `cli` | copied from `ingots/cli` | a native classifier, `cli` and `cli trace`, for step 4 |
   | `network` | `tools/gen_network.py` (the optional last argument) | the whole forward pass on-chain, to measure what a dispute saves |

   and `build/tree.json`: the model's Merkle root and the proof of every neuron.
3. `fe test` (26 tests for the tiny model) and `fe build`. The Fe compiler, not the generator,
   rejects a model whose layers do not chain, whose scales disagree, or whose int32 accumulator
   could overflow for any input: those are type errors and failed `static_assert`s in the generated
   code (`tools/check_generator.sh` has the cases). `check_size.sh` checks the 24,576-byte limit,
   which `fe build` does not.
4. The native build against the reference on every input: classes, and every hidden activation and
   logit of the traced ones. Expect `diff_native: PASS, zero mismatches`.

The arbiter is generated, not one generic contract, because a `msg` variant needs concrete ABI
types and a selector string, so the number of layers and their widths cannot be const generics of a
handler. For the shipped model the generator's output equals the hand-written
`ingots/onchain/src/arbiter.fe`, `record.fe`, `signals.fe`, `commitment.fe` and `arbiter_tests.fe`
byte for byte after `fe fmt`, and the built `Arbiter` is the same bytecode:
`bash tools/check_arbiter_gen.sh --bytecode`.

## 3. Deploy and dispute

`bash tools/arbiter_gas_any.sh build/tiny/build/out build/tiny/out build/tiny/build/tree.json`
does all of this on a local anvil and checks every result. By hand, with `ETH_RPC_URL` and `KEY` set:

```
A="python3 tools/model_args.py --ref build/tiny/out --tree build/tiny/build/tree.json"
ROOT=$(python3 -c 'import json; print(json.load(open("build/tiny/build/tree.json"))["root"])')
ARGS=$(cast abi-encode "constructor(uint256,uint256,uint256)" $ROOT 1000000000000000000 3600)
cast send --private-key $KEY --create 0x$(cat build/tiny/build/out/Arbiter.bin)${ARGS#0x}   # root, bond in wei, window in seconds

mapfile -t CLAIM < <($A claim --sample 0 --mutate out:2:3)      # a false claim: output 3 of layer 2 is off by one
cast send --private-key $KEY --value 1ether $ARBITER "$($A signature claim)" "${CLAIM[@]}"

mapfile -t V < <($A challenge 2 --sample 0 --mutate out:2:3)    # the claim's input and output of layer 2
mapfile -t N < <($A proof 2 3)                                  # row 3 of layer 2, its bias, its Merkle proof
cast send --private-key $KEY $ARBITER "$($A signature challenge 2)" 0 3 "${V[@]}" "${N[@]}"
cast call $ARBITER "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)" 0   # last value 2: slashed
```

For the tiny model the signatures are `claim(uint256[2],uint256[1],uint256[1],int32[4],uint8)` and
`challengeNeuron2(uint256,uint256,uint256[1],uint256[1],uint256[1],int32,bytes32[])`. The claim
carries the input and every layer's output as packed words (the logits as `int32`); the record
keeps one hash per vector; there is one `challengeNeuronK` per layer and one `challengeClass`. An
honest claim is closed with `finalize(id)` after the window, and `result(id)` then returns its
class. With one hidden layer the vector is named `hidden`, with more `hidden1`, `hidden2`, ...

## 4. Use a result from another contract

`examples/mnist-784-128-64-10/ingots/consumer/src/lib.fe` (hand-written, 60 lines, 2 tests) counts
one vote per finalised claim. It imports the arbiter's message type and asks for the result:

```fe
use arbiter::ArbiterMsg
...
let class: u8 = source.static(ArbiterMsg::ResultOf { id })
```

`result` reverts with `NotFinal` unless the claim is final, and the revert propagates, so an open or
slashed claim cannot be counted. The contract's effects are `call: Call` (not `mut Call`) and its
own storage: the signature shows it cannot move ETH or change the arbiter.

## 5. Watching claims off-chain

A watcher recomputes the honest vectors for a claim's input and compares them with the `Claimed`
event; the first difference names the challenge to send. For a generated model the honest vectors
come from the reference (`ref_vectors.py`) or from the workspace's native `cli trace`. The shipped
native auditor (`cli audit`), which does the comparison in Fe, and the shipped WebAssembly module
are fixed to the shipped 784-32-10 model; neither is generated for other models.

## 6. Limits

- **Dense layers and 1D convolutions only** (no 2D convolution, no padding, no pooling: stride
  instead), int8 weights, power-of-two scales, ReLU as a clamp to 0..127,
  `input_frac_bits` 7. At least two layers, the last one dense, at most 255 classes; the generated
  tests need three or more neurons in every hidden layer (three or more positions and filters in a
  convolution) and one of the first three reference samples to differ from sample 0 in every
  vector.
- **Sizes are fixed at compile time.** A new model is a new workspace, a new root and a new
  deployment. Nothing is upgradable.
- **One overflow proof visits at most about 150,000 weights**: a layer of 192 x 784 compiled and
  256 x 784 did not (`docs/fe-notes.md` section 4, measured on Fe 26.5.1 for one-layer models). The
  largest layer built through this path is 128 x 784 = 100,352 weights. A neuron's fan-in is at
  most 131,071 (the `where` clause on the dot product).
- **Contract size.** The `Arbiter` grows by one handler per layer: 11,649 bytes of runtime for two
  layers, 14,361 for three, of 24,576. The optional `Network` keeps every layer after the first in
  one contract, which fits for 128 -> 64 -> 10 (22,247 bytes) and will not for much wider layers.
- **The claim grows with the model's width**: its calldata is every layer's output. Measured:
  `examples/mnist-784-128-64-10/RESULTS.md`.
- **What the evidence covers.** The identity check ties the Foundry fuzz and invariant suite, the
  Solidity twin and the mutation run to the generator for two-layer models only. A deeper model is
  covered by its generated Fe contract tests (22 for the MNIST example), the reference
  differentials and the anvil script, not by fuzzing or mutation testing.
- **Tests of large models are split.** The Fe test EVM gives each test 1,000,000,000 gas and never
  frees memory, so above 64 neurons the honest-claim test is generated as several tests of at most
  32 neurons each.
- Not audited. The limits of the dispute design itself are in `docs/spec-dispute.md` section 7.
