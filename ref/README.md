# ref: the NumPy reference for Tensile

The integer oracle that the Fe code (native and EVM) must match bit for bit, plus the scripts
that train the model and write the test vectors. The arithmetic and file formats are defined in
`docs/spec.md`; this folder implements them and nothing else.

Dependencies: Python 3 with NumPy, and the standard library. No PyTorch, no pip installs.
Measured with Python 3.11.9 and NumPy 2.3.3 on Windows 11 (Intel i5-13420H).

## Files

| File | What it does |
|---|---|
| `get_mnist.py` | Downloads the four MNIST idx files into `ref/data/` (two mirrors), verifies size and SHA-256, skips files already present. Also holds the idx loaders. |
| `train.py` | Trains the 784 -> 32 -> 10 perceptron in plain NumPy (fixed seed), quantises it to the spec, writes `ref/model/mnist-784-32-10.json`. |
| `tensile_ref.py` | The integer reference, importable: `neuron_acc` (one neuron), `layer_acc` and `layer` (one layer), `run` (whole network: hidden activations, logits, class), `shift_floor`, `clamp_act`, `argmax_low`, `worst_case_abs_acc`, `load_model`. |
| `export.py` | Runs the reference on the 10,000 MNIST test images and writes `ref/out/`: `model.json`, `inputs.bin`, `labels.bin`, `pred.bin`, `trace.jsonl`. |
| `test_ref.py` | Plain-assert tests of the arithmetic rules and of the exported files. |
| `model/mnist-784-32-10.json` | The trained integer model (output of `train.py`, input of `export.py`). Small enough to commit. |
| `data/` | Downloaded MNIST, gitignored. |
| `out/` | The exported files. Tracked, so that the differential tests need no download. |

## Commands, in order

Run from the repository root (any working directory works: paths are resolved from the scripts).

```
python ref/get_mnist.py     # about 8 s, 55 MB download
python ref/train.py         # 77 to 123 s measured on a busy laptop (CPU, one thread)
python ref/export.py        # about 2 s
python ref/test_ref.py      # about 5 s
```

`train.py` can be skipped if `ref/model/mnist-784-32-10.json` is already present: `export.py`
needs only that file and the MNIST test set.

## Expected results

| Quantity | Value |
|---|---|
| Float test accuracy | 97.50 % (9,750 of 10,000) |
| Integer test accuracy | 97.44 % (9,744 of 10,000) |
| Gap (float minus integer) | 0.06 percentage points |
| Layer 1 (784 -> 32) | `weight_frac_bits` 7, `shift` 11, accumulator at 14 fractional bits, hidden output at 3 |
| Layer 2 (32 -> 10) | `weight_frac_bits` 6, `shift` null, logits at 9 fractional bits |
| Worst-case accumulator, layer 1 | 1,481,397 (0.069 % of the int32 maximum 2,147,483,647) |
| Worst-case accumulator, layer 2 | 66,144 (0.003 % of the int32 maximum) |
| `test_ref.py` | `PASS: 7781 checks in 11 groups` |

Expected SHA-256 of the files in `ref/out/` (printed by `export.py`):

```
a0ddf492df8c600de94d18e99bfd49c704ef9a2744d68cbbbbb419cc9a933c8b     103882 bytes  model.json
102f936c1607d58e4e81044005738a8ebd2d52f0175fcd95f9b1e5a2825a61e4    7840000 bytes  inputs.bin
ddeff807876a9661a1110d45c266c86239a3a1b7d37da0c3716a7a683c852ff5      10000 bytes  labels.bin
2a3b7bcc2fc4baba67de4bbb2246a7916ca20fbe01b31cc7372e461b0b30cc34      10000 bytes  pred.bin
81a696e6194cfed7a311cbdcb46ef880bf058e8554f17b6684bd812f560e6811     226427 bytes  trace.jsonl
```

`ref/model/mnist-784-32-10.json`: `698a030823a54b5ebb7b9fa152549fd829835d20a3d907078f01cbbe2739ff98`.

## Reproducibility

- `export.py` is a pure function of the model file and the MNIST test set: no timestamps, fixed
  key order, LF line endings. Two runs give identical hashes, and `test_ref.py` re-exports into
  a temporary folder and compares.
- `inputs.bin` and `labels.bin` depend only on MNIST, so their hashes hold on every machine.
- `train.py` is deterministic on one machine (three runs here gave the same model hash). It uses
  floating point, so a different CPU or BLAS build may round differently and produce a
  different, equally valid model. To reproduce the hashes above exactly on another machine, keep
  the committed `ref/model/mnist-784-32-10.json` and run only `get_mnist.py`, `export.py` and
  `test_ref.py`. After a retrain, the model, `pred.bin` and `trace.jsonl` hashes change together
  and the tests still have to pass.

## How the model is quantised

- Input: `x0 = pixel >> 1` (0..127). The float model is trained on exactly `x0 / 128`.
- Weights: `round(w * 2^weight_frac_bits)`, clipped to -127..127. Biases: rounded at the
  accumulator scale (spec section 4).
- The hidden bias gets `2^(shift - 1)` added, which turns the floor of the arithmetic shift into
  round-to-nearest at no cost at inference time. It is still an ordinary int32 bias.
- The three scales are chosen by a small grid search scored on 5,000 training images held out
  for validation. The test set is never used for a decision. The chosen hidden scale clips
  activations above 127 / 8 = 15.875 (the 99.9th percentile is about 16.5).
- Post-training quantisation loses 0.08 points on validation, so quantisation-aware fine-tuning
  does not run by default (the trigger is 0.5 points). `python ref/train.py --qat` forces it
  (straight-through estimator, exact integer forward pass): measured once, it reached 97.58 %
  integer test accuracy in 53 more seconds. The numbers and hashes above are without it.
- Training uses random one-pixel shifts and decoupled weight decay. Weights of pixels that are
  never lit start at 0 and stay 0.

## Details the spec leaves open (as implemented)

- `model.json` has one extra key per layer, `worst_case_abs_acc`: the largest value over the
  layer's neurons of `|bias| + max(sum of max(127 * w, 0), sum of max(-127 * w, 0))`. It bounds
  the accumulator and every partial sum, in any summation order, for all inputs in 0..127.
- `accuracy.float` and `accuracy.integer` are fractions (0.9744), not percentages.
- `"i"` in `trace.jsonl` is zero-based; line `n` describes the image at byte offset `784 * n` of
  `inputs.bin`.
- JSON text is UTF-8 with LF line endings; each weight row is on its own line.

## What the tests cover

The shift rounds toward negative infinity; clamp boundaries (0, 127, 128, -1); the argmax tie
picks the lowest index; int32 overflow is detected (accumulator, bias, and the worst case of a
model); operand ranges are enforced; the exported files have the right sizes and structure;
`inputs.bin` and `labels.bin` equal the MNIST test set; a separate pure-Python integer
implementation (no NumPy, no shared code) matches every hidden activation, logit and class on
500 images; `layer_acc` equals `neuron_acc` applied per output; `pred.bin` equals the reference
on all 10,000 images and the classes in `trace.jsonl` on the first 1,000; a fresh export is
byte-identical to `ref/out/`.
