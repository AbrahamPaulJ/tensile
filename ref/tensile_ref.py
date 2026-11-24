"""Tensile integer reference (the oracle). Implements docs/spec.md, sections 1 to 4.

Everything here is integer arithmetic. NumPy int64 is used internally and every accumulator is
checked to fit int32, because the Fe implementations (native and EVM) work in int32.

The unit of arithmetic is one neuron:

    acc[j] = b[j] + sum over i of ( w[j][i] * x[i] )

`neuron_acc` computes exactly that for one output index. `layer_acc` is the same arithmetic
(one shared function, `_neuron_arith`) applied to every output index, so a dispute that re-runs
a single neuron on-chain re-runs the arithmetic the whole network was built from. test_ref.py
checks that `layer_acc` equals `neuron_acc` per output.

All functions accept either one input vector, shape (IN,), or a batch, shape (N, IN).
"""

import json

import numpy as np

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1
ACT_MIN = 0  # lower clamp of a hidden activation (this is the ReLU)
ACT_MAX = 127  # upper clamp of a hidden activation, and the largest input activation
WEIGHT_MAX = 127  # weights are -127..127; -128 is never produced
INPUT_FRAC_BITS = 7  # x0 / 128 approximates pixel / 255


class Int32OverflowError(OverflowError):
    """An accumulator (or a bias) does not fit in int32."""


def _as_int64(a, what):
    arr = np.asarray(a)
    if arr.dtype.kind not in "iu":
        raise TypeError("%s must be integers, got dtype %s" % (what, arr.dtype))
    return arr.astype(np.int64, copy=False)


def check_int32(values, what="accumulator"):
    """Raise Int32OverflowError unless every value fits int32. Returns the values as int64."""
    arr = _as_int64(values, what)
    if arr.size and (int(arr.min()) < INT32_MIN or int(arr.max()) > INT32_MAX):
        raise Int32OverflowError(
            "%s out of int32 range: min %d, max %d" % (what, int(arr.min()), int(arr.max()))
        )
    return arr


def quantise_input(pixels):
    """MNIST pixel (0..255) to input activation x0 (0..127): x0 = pixel >> 1."""
    arr = _as_int64(pixels, "pixels")
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 255):
        raise ValueError("pixels must be in 0..255")
    return arr >> 1


def _check_operands(w, bias, xv):
    """Range and shape checks shared by neuron_acc and layer_acc (w is (IN,) or (OUT, IN))."""
    if xv.ndim not in (1, 2) or xv.shape[-1] != w.shape[-1]:
        raise ValueError("input shape %s does not match %d weights" % (xv.shape, w.shape[-1]))
    if w.size and (int(w.min()) < -WEIGHT_MAX or int(w.max()) > WEIGHT_MAX):
        raise ValueError("weight outside -127..127")
    if xv.size and (int(xv.min()) < ACT_MIN or int(xv.max()) > ACT_MAX):
        raise ValueError("activation outside 0..127")
    check_int32(bias, "bias")


def _neuron_arith(w, b, xv):
    """The arithmetic of one neuron on already validated int64 operands.

    This is the single place where an accumulator is computed: neuron_acc and layer_acc both
    call it, once per output index.
    """
    # int64 cannot overflow here: |w * x| <= 127 * 127 and |b| < 2^31.
    acc = np.int64(b) + xv @ w
    check_int32(acc, "accumulator")
    return acc


def neuron_acc(w_row, bias, x):
    """Accumulator of ONE neuron: bias + sum_i w_row[i] * x[i].

    w_row: (IN,) integers in -127..127.   bias: one int32.
    x: (IN,) or (N, IN) integers in 0..127.
    Returns an int64 scalar, or an int64 array of shape (N,). Raises Int32OverflowError if the
    result does not fit int32.
    """
    w = _as_int64(w_row, "weights")
    xv = _as_int64(x, "activations")
    if w.ndim != 1:
        raise ValueError("w_row must be one-dimensional")
    _check_operands(w, bias, xv)
    return _neuron_arith(w, bias, xv)


def shift_floor(acc, shift):
    """Arithmetic right shift: floor(acc / 2^shift), rounding toward negative infinity."""
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)) or not 0 <= shift <= 62:
        raise ValueError("shift must be an integer in 0..62")
    return np.floor_divide(_as_int64(acc, "accumulator"), np.int64(1) << np.int64(shift))


def clamp_act(v):
    """clamp(v, 0, 127): 0 if v < 0, 127 if v > 127, else v."""
    return np.clip(_as_int64(v, "value"), ACT_MIN, ACT_MAX)


def argmax_low(logits):
    """Index of the largest logit; the LOWEST index wins ties. Works on the last axis."""
    arr = _as_int64(logits, "logits")
    # numpy.argmax returns the first occurrence of the maximum, which is the lowest index.
    out = np.argmax(arr, axis=-1)
    return int(out) if arr.ndim == 1 else out.astype(np.int64)


def layer_acc(weights, bias, x):
    """Accumulators of a whole layer: the neuron arithmetic for every output index j, stacked on
    the last axis. Equal to [neuron_acc(weights[j], bias[j], x) for j in range(OUT)]; the operands
    are validated once for the layer instead of once per neuron.

    weights: (OUT, IN).  bias: (OUT,).  x: (IN,) or (N, IN).  Returns (OUT,) or (N, OUT) int64.
    """
    w = _as_int64(weights, "weights")
    b = _as_int64(bias, "bias")
    xv = _as_int64(x, "activations")
    if w.ndim != 2 or b.shape != (w.shape[0],):
        raise ValueError("weights must be (OUT, IN) and bias (OUT,)")
    _check_operands(w, b, xv)
    return np.stack([_neuron_arith(w[j], b[j], xv) for j in range(w.shape[0])], axis=-1)


def layer(weights, bias, x, shift):
    """Run one dense layer.

    Hidden layer (shift is an int): y = clamp(acc >> shift, 0, 127).
    Last layer (shift is None): no shift and no clamp, y = acc (the logits).
    """
    acc = layer_acc(weights, bias, x)
    if shift is None:
        return acc
    return clamp_act(shift_floor(acc, shift))


def conv_positions(length, kernel, stride):
    """Number of positions of a convolution without padding (spec section 8)."""
    return (length - kernel) // stride + 1


def conv_shape(lay):
    """(L, C, K, S, P, filters) of a "conv1d" layer dict; raises ValueError if its fields disagree."""
    c, k, s = lay["channels"], lay["kernel"], lay["stride"]
    for name, value in (("channels", c), ("kernel", k), ("stride", s)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("conv1d: %s must be a positive integer" % name)
    if lay["in"] % c != 0:
        raise ValueError("conv1d: in is not a whole number of positions of %d channels" % c)
    length = lay["in"] // c
    if length < k:
        raise ValueError("conv1d: the kernel is longer than the sequence")
    filters = len(lay["weights"])
    p = conv_positions(length, k, s)
    if lay["out"] != p * filters:
        raise ValueError("conv1d: out must be positions * filters = %d * %d" % (p, filters))
    return length, c, k, s, p, filters


def conv1d_acc(weights, bias, x, channels, stride):
    """Accumulators of a 1D convolution without padding (spec section 8).

    weights: (FILTERS, K * C), filter j as K positions of C channels.  bias: (FILTERS,).
    x: (L * C,) or (N, L * C), position by position.  Returns (P * FILTERS,) or (N, P * FILTERS):
    value p * FILTERS + j is `_neuron_arith` (the arithmetic of a dense neuron) of filter j on the
    K * C values of x that start at value p * stride * C.
    """
    w = _as_int64(weights, "weights")
    b = _as_int64(bias, "bias")
    xv = _as_int64(x, "activations")
    if w.ndim != 2 or b.shape != (w.shape[0],) or w.shape[1] % channels != 0:
        raise ValueError("weights must be (FILTERS, K * C) and bias (FILTERS,)")
    if xv.ndim not in (1, 2) or xv.shape[-1] % channels != 0 or xv.shape[-1] < w.shape[1]:
        raise ValueError("input shape %s does not hold positions of %d channels" % (xv.shape, channels))
    span = w.shape[1]
    _check_operands(w, b, xv[..., :span])
    if xv.size and (int(xv.min()) < ACT_MIN or int(xv.max()) > ACT_MAX):
        raise ValueError("activation outside 0..127")
    count = conv_positions(xv.shape[-1] // channels, span // channels, stride)
    starts = np.arange(count) * (stride * channels)
    # patches[..., p, :] is the window of position p.
    patches = xv[..., starts[:, None] + np.arange(span)[None, :]]
    acc = np.stack([_neuron_arith(w[j], b[j], patches) for j in range(w.shape[0])], axis=-1)
    return acc.reshape(xv.shape[:-1] + (count * w.shape[0],))


def layer_of(lay, x):
    """Run one layer given as a model dict entry: a dense layer, or `"kind": "conv1d"`."""
    if lay.get("kind", "dense") == "conv1d":
        acc = conv1d_acc(lay["weights"], lay["bias"], x, lay["channels"], lay["stride"])
        if lay["shift"] is None:
            return acc
        return clamp_act(shift_floor(acc, lay["shift"]))
    return layer(lay["weights"], lay["bias"], x, lay["shift"])


def run(model, x0):
    """Run the whole network on x0, shape (IN_1,) or (N, IN_1), values 0..127.

    Returns (hidden, logits, cls):
      hidden: list with one int64 array per hidden layer, (OUT_l,) or (N, OUT_l), values 0..127
      logits: int64 array (OUT_L,) or (N, OUT_L), each value fits int32
      cls:    int, or int64 array (N,): argmax of the logits, lowest index winning ties
    """
    layers = model["layers"]
    x = _as_int64(x0, "activations")
    hidden = []
    for idx, lay in enumerate(layers):
        last = idx == len(layers) - 1
        if last != (lay["shift"] is None):
            raise ValueError("exactly the last layer must have shift null")
        x = layer_of(lay, x)
        if not last:
            hidden.append(x)
    return hidden, x, argmax_low(x)


def worst_case_abs_acc(weights, bias):
    """Per-neuron upper bound on |acc| over ALL inputs in 0..127.

    bound[j] = |b[j]| + max( sum_i max(127 * w[j][i], 0), sum_i max(-127 * w[j][i], 0) )

    The same value also bounds every partial sum, in any summation order and with the bias added
    first or last, so an int32 accumulator can never overflow part-way through.
    Returns a list of Python ints, one per neuron.
    """
    w = _as_int64(weights, "weights")
    b = _as_int64(bias, "bias")
    pos = ACT_MAX * np.where(w > 0, w, 0).sum(axis=1)
    neg = ACT_MAX * np.where(w < 0, -w, 0).sum(axis=1)
    return [abs(int(b[j])) + max(int(pos[j]), int(neg[j])) for j in range(w.shape[0])]


def check_model(model):
    """Validate structure and value ranges of a model dict (spec sections 1, 2 and 5)."""
    layers = model["layers"]
    if model["input_frac_bits"] != INPUT_FRAC_BITS:
        raise ValueError("input_frac_bits must be %d" % INPUT_FRAC_BITS)
    if not layers:
        raise ValueError("a model needs at least one layer")
    prev_out = None
    for idx, lay in enumerate(layers):
        w = _as_int64(lay["weights"], "weights")
        b = _as_int64(lay["bias"], "bias")
        kind = lay.get("kind", "dense")
        if kind == "conv1d":
            _, c, k, _, _, filters = conv_shape(lay)
            if w.shape != (filters, k * c) or b.shape != (filters,):
                raise ValueError("layer %d: weights or bias shape does not match the filters" % (idx + 1))
            if idx == len(layers) - 1:
                raise ValueError("the last layer must be dense")
        elif kind != "dense":
            raise ValueError("layer %d: unknown kind %r" % (idx + 1, kind))
        elif w.shape != (lay["out"], lay["in"]) or b.shape != (lay["out"],):
            raise ValueError("layer %d: weights or bias shape does not match in/out" % (idx + 1))
        if prev_out is not None and lay["in"] != prev_out:
            raise ValueError("layer %d: in != out of the previous layer" % (idx + 1))
        prev_out = lay["out"]
        if int(w.min()) < -WEIGHT_MAX or int(w.max()) > WEIGHT_MAX:
            raise ValueError("layer %d: weight outside -127..127" % (idx + 1))
        check_int32(b, "bias")
        last = idx == len(layers) - 1
        shift = lay["shift"]
        if last:
            if shift is not None:
                raise ValueError("the last layer must have shift null")
        elif isinstance(shift, bool) or not isinstance(shift, int) or shift < 0:
            raise ValueError("layer %d: shift must be a non-negative integer" % (idx + 1))
        if isinstance(lay["weight_frac_bits"], bool) or not isinstance(lay["weight_frac_bits"], int):
            raise ValueError("layer %d: weight_frac_bits must be an integer" % (idx + 1))
        if max(worst_case_abs_acc(w, b)) > INT32_MAX:
            raise Int32OverflowError("layer %d: worst-case accumulator exceeds int32" % (idx + 1))
    return model


def frac_bits(model):
    """Fractional bits per stage (spec section 4).

    Returns (acc_bits, out_bits): acc_bits[l] is the scale of layer l's accumulator and bias,
    F_{l-1} + FW_l; out_bits[l] is F_l for hidden layers and None for the last layer.
    """
    f = model["input_frac_bits"]
    acc_bits, out_bits = [], []
    for lay in model["layers"]:
        a = f + lay["weight_frac_bits"]
        acc_bits.append(a)
        if lay["shift"] is None:
            out_bits.append(None)
        else:
            f = a - lay["shift"]
            out_bits.append(f)
    return acc_bits, out_bits


def load_model(path):
    """Read a model.json (spec section 5); weights and biases become int64 arrays."""
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    for lay in model["layers"]:
        lay["weights"] = np.array(lay["weights"], dtype=np.int64)
        lay["bias"] = np.array(lay["bias"], dtype=np.int64)
    return check_model(model)


def _plain(value):
    """NumPy arrays and scalars to plain Python lists and ints, for JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    return value


def model_to_json(model):
    """Serialise a model dict to the text of model.json (spec section 5), deterministically.

    Key order is the insertion order of the dicts. Each weight row is written on its own line.
    Line endings are always LF; write the result in binary mode as UTF-8.
    """
    def scalar(v):
        return json.dumps(_plain(v), separators=(", ", ": "))

    def layer_lines(lay):
        keys = list(lay)
        lines = ["    {"]
        for ki, k in enumerate(keys):
            comma = "," if ki < len(keys) - 1 else ""
            if k == "weights":
                rows = _plain(np.asarray(lay[k]))
                lines.append('      "weights": [')
                for ri, row in enumerate(rows):
                    lines.append("        %s%s" % (scalar(row), "," if ri < len(rows) - 1 else ""))
                lines.append("      ]" + comma)
            else:
                lines.append("      %s: %s%s" % (json.dumps(k), scalar(lay[k]), comma))
        lines.append("    }")
        return lines

    out = ["{"]
    keys = list(model)
    for ki, k in enumerate(keys):
        comma = "," if ki < len(keys) - 1 else ""
        if k == "layers":
            out.append('  "layers": [')
            for li, lay in enumerate(model[k]):
                lines = layer_lines(lay)
                if li < len(model[k]) - 1:
                    lines[-1] += ","
                out.extend(lines)
            out.append("  ]" + comma)
        else:
            out.append("  %s: %s%s" % (json.dumps(k), scalar(model[k]), comma))
    out.append("}")
    return "\n".join(out) + "\n"
