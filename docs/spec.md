# Tensile: arithmetic and data specification (v0, 2026-11-05)

"Tensile" is the working name. This file is the single source of truth for the integer arithmetic.
Three implementations must agree bit for bit: the NumPy reference (`ref/`), the Fe code built
natively, and the Fe code built for the EVM. If an implementation and this file disagree, fix one
of them in the same change; never leave them apart.

## 1. Network

A multilayer perceptron with `L` dense layers. The reference model is MNIST, 784 -> 32 -> 10.

- Layer `l` (1-based) has `IN_l` inputs and `OUT_l` outputs, with `IN_1 = 784` and
  `IN_{l+1} = OUT_l`.
- All values are integers. There is no floating point anywhere at inference time.

## 2. Value ranges

| Quantity | Type | Range |
|---|---|---|
| Input activation `x0[i]` | int8 | 0..127, computed as `pixel >> 1` from the 0..255 MNIST pixel |
| Weight `w[j][i]` | int8 | -127..127 (-128 is never produced) |
| Bias `b[j]` | int32 | any int32 |
| Accumulator `acc[j]` | int32 | must fit without overflow; the exporter checks the worst case |
| Hidden activation | int8 | 0..127 |
| Output logits (last layer) | int32 | the raw accumulator |

**Range rules for untrusted input** (anything that arrives in calldata or on stdin):

- An activation vector (input or hidden) with any byte above 127 is malformed and must be
  rejected. The compile-time overflow proof assumes activations in 0..127.
- A weight row is never trusted on its own: on-chain it is accepted only through a Merkle proof
  against the committed model (`docs/spec-dispute.md`), so it is always one of the exported rows.
  A weight of -128 therefore cannot occur, and the kernel would still be safe if it did
  (`dot_fits_i32` allows for `128 * 128`).

## 3. Layer arithmetic

For every layer and every output `j`:

```
acc[j] = b[j] + sum over i of ( w[j][i] * x[i] )
```

Hidden layers (all but the last), with a per-layer non-negative integer `shift`:

```
y[j] = clamp( acc[j] >> shift, 0, 127 )
```

- `>>` is an arithmetic shift: floor division by `2^shift`, rounding toward negative infinity.
  `-1 >> 1` is `-1`.
- `clamp(v, 0, 127)` is `0` if `v < 0`, `127` if `v > 127`, else `v`. The lower clamp is the ReLU.

Last layer: no shift and no clamp. `logits[j] = acc[j]`.

Class: `argmax(logits)`, the lowest index winning ties.

## 4. Scales (what the types track)

Every quantity has a number of fractional bits `F`: real value = integer / `2^F`. All scales are
powers of two, so changing scale is a shift and nothing else.

- Input: `F_0 = 7` (pixel / 255 is approximated by `x0 / 128`).
- Weights of layer `l`: `FW_l` fractional bits.
- Accumulator and bias of layer `l`: `F_{l-1} + FW_l` fractional bits. A bias at any other scale
  is a bug, and in the Fe code it must not compile.
- Hidden output: `F_l = F_{l-1} + FW_l - shift_l`.

## 5. Model file: `ref/out/model.json`

```json
{
  "name": "mnist-784-32-10",
  "input_frac_bits": 7,
  "layers": [
    {
      "in": 784, "out": 32,
      "weight_frac_bits": 7,
      "shift": 11,
      "worst_case_abs_acc": 1481397,
      "weights": [[ "out rows, each of in integers in -127..127" ]],
      "bias": [ "out integers, int32" ]
    }
  ],
  "accuracy": { "float": 0.975, "integer": 0.9744, "test_samples": 10000 }
}
```

- The last layer has `"shift": null`.
- `worst_case_abs_acc` is the largest absolute accumulator any neuron of the layer can reach over
  every input in 0..127: per neuron, `|bias|` plus the larger of the positive-weight sum and the
  negative-weight sum, each times 127. It also bounds every partial sum. It must fit int32; the Fe
  code re-proves this at compile time.
- `accuracy` values are fractions (0.9744 means 97.44%).
- A hidden layer's bias may include a rounding offset of `2^(shift-1)`, so that the floor shift
  rounds to nearest. This needs nothing from the Fe side: it is just part of the bias.

## 6. Test vectors: `ref/out/`

| File | Content |
|---|---|
| `inputs.bin` | `N x 784` bytes: `x0` for each test image, row-major, one byte per value (0..127) |
| `labels.bin` | `N` bytes: the true digit |
| `pred.bin` | `N` bytes: the class the integer reference predicts |
| `trace.jsonl` | For the first 1,000 samples, one JSON object per line: `{"i": n, "hidden": [[...]], "logits": [...], "class": c}` where `hidden` lists every hidden layer's activations |

`N` is 10,000 (the MNIST test set). `"i"` in the trace is zero-based.

The exported vectors cannot tell a floor shift from a truncating one, because every negative
accumulator clamps to 0 either way. Each implementation therefore needs its own unit test of the
shift rule on a negative value (`-1 >> 1` must be `-1`).

## 7. Byte encoding and packed words

- An int8 vector (activations, or one row of weights) is encoded one byte per element, two's
  complement, index order.
- An int32 logit vector is encoded as 32 bytes per element, two's complement sign-extended to 256
  bits, index order: the ABI encoding of `int32[]` elements, which is what the contracts return.
  (An earlier draft said four bytes per element; nothing was built on it.)
- Hash function: Keccak-256.

**Packed words.** Wherever an int8 vector of length `N` is held or passed as 256-bit words, it is
the byte encoding above, cut into 32-byte big-endian words: value `i` is byte `i % 32` of word
`i / 32`, counting from the most significant byte. There are `ceil(N / 32)` words and the unused
low bytes of the last word are zero. A packed vector with a non-zero padding byte is malformed and
must be rejected. Hashing the words in order is therefore hashing the byte encoding followed by
zero padding to a multiple of 32 bytes; commitments hash the padded form.

The commitment and dispute rules are in `docs/spec-dispute.md` (milestone 3).
