"""Tests for the Tensile integer reference and the exported files.

Usage:  python ref/test_ref.py        (after get_mnist.py, train.py and export.py)

Plain checks, no test framework. Exits non-zero on the first failure. Prints the number of
checks and the wall time.
"""

import hashlib
import json
import operator
import os
import random
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export  # noqa: E402
import get_mnist  # noqa: E402
import tensile_ref as ref  # noqa: E402

OUT = os.path.join(HERE, "out")
N = 10000
N_TRACE = 1000
N_PURE = 500  # images recomputed with pure Python integers

CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


def raises(exc_type, fn, msg):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except exc_type:
        return
    raise AssertionError("expected %s: %s" % (exc_type.__name__, msg))


def read_bytes(name):
    with open(os.path.join(OUT, name), "rb") as f:
        return f.read()


# ------------------------------------------------------------------ arithmetic rules


def test_shift_floor():
    # Spec section 3: arithmetic shift, floor division by 2^shift, toward negative infinity.
    cases = [(-1, 1, -1), (-1, 7, -1), (-2, 1, -1), (-3, 1, -2), (-128, 7, -1), (-129, 7, -2),
             (-127, 7, -1), (127, 7, 0), (128, 7, 1), (255, 7, 1), (0, 5, 0), (5, 0, 5),
             (-5, 0, -5), (-(2 ** 31), 31, -1), (2 ** 31 - 1, 31, 0), (-(2 ** 31), 4, -(2 ** 27))]
    for acc, shift, want in cases:
        check(int(ref.shift_floor(acc, shift)) == want, "shift_floor(%d, %d) != %d" % (acc, shift, want))
    # Against Python's own >> (arithmetic on negative ints) and floor division, on a sweep.
    rng = random.Random(1)
    values = list(range(-300, 301)) + [rng.randrange(-(2 ** 31), 2 ** 31) for _ in range(2000)]
    arr = np.array(values, dtype=np.int64)
    for shift in (0, 1, 2, 7, 8, 13, 30):
        got = ref.shift_floor(arr, shift).tolist()
        check(got == [v >> shift for v in values], "shift %d differs from Python >>" % shift)
        check(got == [v // (1 << shift) for v in values], "shift %d differs from floor division" % shift)
        check(got == np.right_shift(arr, shift).tolist(), "shift %d differs from numpy >>" % shift)
    # Truncation toward zero would give 0 here; the spec requires -1.
    check(int(ref.shift_floor(-1, 1)) != int(-1 / 2), "shift must not truncate toward zero")
    raises(ValueError, lambda: ref.shift_floor(4, -1), "negative shift")
    raises(ValueError, lambda: ref.shift_floor(4, 63), "shift too large for int64")


def test_clamp():
    for v, want in [(0, 0), (127, 127), (128, 127), (-1, 0), (1, 1), (126, 126),
                    (2 ** 31 - 1, 127), (-(2 ** 31), 0)]:
        check(int(ref.clamp_act(v)) == want, "clamp(%d) != %d" % (v, want))
    # The same boundaries through a real hidden layer: one neuron, weight 1, input 0, shift 3,
    # so acc = bias and y = clamp(bias >> 3).
    for bias, want in [(0, 0), (7, 0), (8, 1), (127 * 8, 127), (127 * 8 + 7, 127), (128 * 8, 127),
                       (-1, 0), (-8, 0), (-9, 0), (10 ** 6, 127)]:
        y = ref.layer([[1]], [bias], [0], 3)
        check(y.tolist() == [want], "layer clamp for bias %d: %s != %d" % (bias, y.tolist(), want))
    # Last layer: no shift and no clamp.
    check(ref.layer([[1]], [-5], [3], None).tolist() == [-2], "last layer must return the raw accumulator")
    check(ref.layer([[127]], [1000], [127], None).tolist() == [17129], "last layer must not clamp")


def test_argmax():
    cases = [([5, 5, 1], 0), ([1, 7, 7], 1), ([3, 3, 3, 3], 0), ([-4, -2, -2, -9], 1), ([0], 0),
             ([1, 2, 3], 2), ([-(2 ** 31), -(2 ** 31)], 0), ([2 ** 31 - 1, 0, 2 ** 31 - 1], 0),
             ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 9), ([9, 0, 0, 0, 0, 0, 0, 0, 0, 9], 0)]
    for logits, want in cases:
        check(ref.argmax_low(logits) == want, "argmax_low(%s) != %d" % (logits, want))
    batch = ref.argmax_low(np.array([[1, 1, 0], [0, 2, 2], [4, 3, 4]]))
    check(batch.tolist() == [0, 1, 0], "batched argmax tie rule")
    # A tie through the whole network: two identical output neurons.
    model = {"input_frac_bits": 7, "layers": [
        {"in": 2, "out": 2, "weight_frac_bits": 0, "shift": 0,
         "weights": [[1, 0], [0, 1]], "bias": [0, 0]},
        {"in": 2, "out": 3, "weight_frac_bits": 0, "shift": None,
         "weights": [[1, 1], [2, 2], [2, 2]], "bias": [0, 0, 0]}]}
    hidden, logits, cls = ref.run(model, [3, 4])
    check(hidden[0].tolist() == [3, 4] and logits.tolist() == [7, 14, 14] and cls == 1,
          "network tie must pick the lowest index")


def test_overflow():
    top, bottom = ref.INT32_MAX, ref.INT32_MIN
    check(top == 2147483647 and bottom == -2147483648, "int32 limits")
    # Exactly at the limits is fine.
    check(int(ref.neuron_acc([1], top - 127, [127])) == top, "acc == INT32_MAX must pass")
    check(int(ref.neuron_acc([-1], bottom + 127, [127])) == bottom, "acc == INT32_MIN must pass")
    # One past the limits is an error.
    raises(ref.Int32OverflowError, lambda: ref.neuron_acc([1], top - 126, [127]), "acc = INT32_MAX + 1")
    raises(ref.Int32OverflowError, lambda: ref.neuron_acc([-1], bottom + 126, [127]), "acc = INT32_MIN - 1")
    raises(ref.Int32OverflowError, lambda: ref.neuron_acc([1], top + 1, [0]), "bias above int32")
    raises(ref.Int32OverflowError, lambda: ref.neuron_acc([1], bottom - 1, [0]), "bias below int32")
    raises(ref.Int32OverflowError, lambda: ref.layer([[1], [0]], [top, 0], [1], None), "layer overflow")
    raises(ref.Int32OverflowError, lambda: ref.layer([[127] * 4], [top - 10], [[0] * 4, [1] * 4], 3),
           "overflow in one row of a batch")
    raises(ref.Int32OverflowError, lambda: ref.check_int32([0, top + 1]), "check_int32")
    check(ref.check_int32([bottom, top]).tolist() == [bottom, top], "check_int32 accepts the limits")
    # Int32OverflowError is an OverflowError, so generic handlers catch it too.
    check(issubclass(ref.Int32OverflowError, OverflowError), "exception hierarchy")
    # The worst-case bound: |b| + max(positive side, negative side), with inputs up to 127.
    check(ref.worst_case_abs_acc([[1, -2, 3]], [10]) == [10 + 127 * 4], "worst case, positive side")
    check(ref.worst_case_abs_acc([[1, -5, 3]], [-10]) == [10 + 127 * 5], "worst case, negative side")
    check(ref.worst_case_abs_acc([[0, 0]], [-7]) == [7], "worst case, zero weights")
    # A model whose worst case does not fit int32 is rejected, even if no test input reaches it.
    bad = {"input_frac_bits": 7, "layers": [
        {"in": 2, "out": 1, "weight_frac_bits": 0, "shift": None,
         "weights": [[127, 127]], "bias": [top - 100]}]}
    raises(ref.Int32OverflowError, lambda: ref.check_model(bad), "worst-case accumulator above int32")
    bad["layers"][0]["bias"] = [top - 2 * 127 * 127]
    check(ref.check_model(bad) is bad, "worst case exactly INT32_MAX is accepted")


def test_operand_checks():
    raises(ValueError, lambda: ref.neuron_acc([1], 0, [128]), "activation 128")
    raises(ValueError, lambda: ref.neuron_acc([1], 0, [-1]), "activation -1")
    raises(ValueError, lambda: ref.neuron_acc([-128], 0, [1]), "weight -128")
    raises(ValueError, lambda: ref.neuron_acc([128], 0, [1]), "weight 128")
    raises(ValueError, lambda: ref.neuron_acc([1, 2], 0, [1]), "length mismatch")
    raises(TypeError, lambda: ref.neuron_acc([1.0], 0, [1]), "float weight")
    raises(TypeError, lambda: ref.neuron_acc([1], 0, [0.5]), "float activation")
    raises(ValueError, lambda: ref.layer([[1]], [0], [200], 0), "layer activation 200")
    raises(ValueError, lambda: ref.quantise_input([256]), "pixel 256")
    check(ref.quantise_input([0, 1, 2, 254, 255]).tolist() == [0, 0, 1, 127, 127], "x0 = pixel >> 1")
    check(int(ref.neuron_acc([127, -127], -3, [127, 0])) == 127 * 127 - 3, "neuron value")
    check(int(ref.neuron_acc([0, 0], 42, [127, 127])) == 42, "zero weights leave the bias")


# ------------------------------------------------------------------ exported files


def test_file_sizes(model):
    inputs, labels, pred = read_bytes("inputs.bin"), read_bytes("labels.bin"), read_bytes("pred.bin")
    check(len(inputs) == N * 784, "inputs.bin must be %d bytes, is %d" % (N * 784, len(inputs)))
    check(len(labels) == N, "labels.bin must be %d bytes" % N)
    check(len(pred) == N, "pred.bin must be %d bytes" % N)
    check(max(inputs) <= 127, "inputs.bin values must be 0..127")
    check(max(labels) <= 9 and max(pred) <= 9, "labels and predictions must be digits")
    trace = read_bytes("trace.jsonl")
    check(b"\r" not in trace and trace.endswith(b"\n"), "trace.jsonl must use LF line endings")
    check(trace.count(b"\n") == N_TRACE, "trace.jsonl must have %d lines" % N_TRACE)
    check(b"\r" not in read_bytes("model.json"), "model.json must use LF line endings")
    for name in export.OUT_FILES:
        check(chr(0x2014) not in read_bytes(name).decode("utf-8", errors="replace"), "em dash in " + name)

    # model.json structure (spec section 5).
    check(list(model)[:4] == ["name", "input_frac_bits", "layers", "accuracy"], "model.json top-level keys")
    check(model["name"] == "mnist-784-32-10" and model["input_frac_bits"] == 7, "name and input scale")
    shapes = [(lay["in"], lay["out"]) for lay in model["layers"]]
    check(shapes == [(784, 32), (32, 10)], "layer shapes %s" % shapes)
    for li, lay in enumerate(model["layers"]):
        last = li == len(model["layers"]) - 1
        check(len(lay["weights"]) == lay["out"] and all(len(r) == lay["in"] for r in lay["weights"]),
              "weights must be out rows of in values")
        check(len(lay["bias"]) == lay["out"], "bias length")
        flat = [w for row in lay["weights"] for w in row]
        check(all(type(w) is int and -127 <= w <= 127 for w in flat), "weights must be ints in -127..127")
        check(all(type(b) is int and -(2 ** 31) <= b < 2 ** 31 for b in lay["bias"]), "bias must be int32")
        check(type(lay["weight_frac_bits"]) is int, "weight_frac_bits must be an int")
        if last:
            check(lay["shift"] is None, "last layer shift must be null")
        else:
            check(type(lay["shift"]) is int and lay["shift"] >= 0, "hidden shift must be a non-negative int")
        # Worst-case accumulator, recomputed here with plain Python integers.
        worst = 0
        for row, b in zip(lay["weights"], lay["bias"]):
            pos = sum(127 * w for w in row if w > 0)
            neg = sum(-127 * w for w in row if w < 0)
            worst = max(worst, abs(b) + max(pos, neg))
        check(lay["worst_case_abs_acc"] == worst, "layer %d worst_case_abs_acc" % (li + 1))
        check(worst <= 2 ** 31 - 1, "layer %d worst case must fit int32" % (li + 1))
    acc = model["accuracy"]
    check(acc["test_samples"] == N, "test_samples")
    correct = sum(1 for p, t in zip(pred, labels) if p == t)
    check(acc["integer"] == correct / N, "accuracy.integer must equal pred.bin against labels.bin")
    check(0.0 <= acc["float"] <= 1.0, "accuracy.float must be a fraction")
    check(acc["integer"] >= 0.96, "integer accuracy %.4f is below the 96 percent target" % acc["integer"])


def test_inputs_are_mnist():
    pixels = get_mnist.load_images("t10k")
    check(read_bytes("inputs.bin") == (pixels >> 1).tobytes(), "inputs.bin must be MNIST test pixels >> 1")
    check(read_bytes("labels.bin") == get_mnist.load_labels("t10k").tobytes(), "labels.bin must be MNIST test labels")


def pure_python_forward(model, x):
    """The network of spec section 3 on Python ints only. Deliberately shares no code with
    tensile_ref: its own dot product, Python's >> for the shift, its own clamp and argmax."""
    hidden = []
    for lay in model["layers"]:
        accs = []
        for row, b in zip(lay["weights"], lay["bias"]):
            acc = b + sum(map(operator.mul, row, x))
            if not -(2 ** 31) <= acc <= 2 ** 31 - 1:
                raise OverflowError("accumulator out of int32")
            accs.append(acc)
        if lay["shift"] is None:
            logits = accs
        else:
            x = []
            for acc in accs:
                v = acc >> lay["shift"]
                x.append(0 if v < 0 else (127 if v > 127 else v))
            hidden.append(x)
    best = 0
    for j in range(1, len(logits)):
        if logits[j] > logits[best]:  # strict: the lowest index keeps a tie
            best = j
    return hidden, logits, best


def test_pure_python(model, np_model, x0):
    hidden, logits, cls = ref.run(np_model, x0[:N_PURE])
    raw = read_bytes("inputs.bin")
    for n in range(N_PURE):
        x = list(raw[n * 784:(n + 1) * 784])
        p_hidden, p_logits, p_cls = pure_python_forward(model, x)
        check(p_hidden == [h[n].tolist() for h in hidden], "sample %d: hidden activations differ" % n)
        check(p_logits == logits[n].tolist(), "sample %d: logits differ" % n)
        check(p_cls == int(cls[n]), "sample %d: class differs" % n)
    # One sample through the single-vector code path of the module as well.
    h1, l1, c1 = ref.run(np_model, x0[0])
    check(h1[0].tolist() == hidden[0][0].tolist() and l1.tolist() == logits[0].tolist()
          and c1 == int(cls[0]), "single-vector run must equal the batched run")


def test_layer_equals_neuron(np_model, x0):
    rng = np.random.default_rng(7)
    hidden, _, _ = ref.run(np_model, x0[:64])
    layer_inputs = [
        np.concatenate([x0[:64], rng.integers(0, 128, (16, 784)), np.full((1, 784), 127), np.zeros((1, 784), np.int64)]),
        np.concatenate([hidden[0], rng.integers(0, 128, (16, 32)), np.full((1, 32), 127), np.zeros((1, 32), np.int64)]),
    ]
    for lay, xs in zip(np_model["layers"], layer_inputs):
        w, b = lay["weights"], lay["bias"]
        acc = ref.layer_acc(w, b, xs)
        y = ref.layer(w, b, xs, lay["shift"])
        for j in range(lay["out"]):
            col = ref.neuron_acc(w[j], b[j], xs)  # batched neuron
            check(np.array_equal(acc[:, j], col), "layer_acc column %d != neuron_acc" % j)
            one = int(ref.neuron_acc(w[j], b[j], xs[3]))  # single-vector neuron
            check(one == int(acc[3, j]), "single neuron %d != layer_acc" % j)
            want = col if lay["shift"] is None else np.clip(col >> lay["shift"], 0, 127)
            check(np.array_equal(y[:, j], want), "layer output %d != shift and clamp of neuron_acc" % j)
        check(np.array_equal(ref.layer_acc(w, b, xs[5]), acc[5]), "single-vector layer_acc")


def test_trace_and_pred(np_model, x0):
    pred = read_bytes("pred.bin")
    lines = read_bytes("trace.jsonl").decode("utf-8").split("\n")[:-1]
    hidden, logits, cls = ref.run(np_model, x0)
    check(cls.astype(np.uint8).tobytes() == pred, "pred.bin must equal the reference on all %d inputs" % N)
    for n, line in enumerate(lines):
        rec = json.loads(line)
        check(list(rec) == ["i", "hidden", "logits", "class"], "trace line %d keys" % n)
        check(rec["i"] == n, "trace line %d has i = %r" % (n, rec["i"]))
        check(rec["class"] == pred[n], "trace line %d: class != pred.bin" % n)
        check(rec["hidden"] == [h[n].tolist() for h in hidden], "trace line %d: hidden != reference" % n)
        check(rec["logits"] == logits[n].tolist(), "trace line %d: logits != reference" % n)
        top = max(rec["logits"])
        check(rec["class"] == rec["logits"].index(top), "trace line %d: class is not the lowest argmax" % n)
    check(len(lines) == N_TRACE, "trace line count")
    check(all(0 <= v <= 127 for h in hidden for v in (int(h.min()), int(h.max()))), "hidden range")


def test_export_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        info = export.build(out_dir=tmp, verbose=False)
        for name in export.OUT_FILES:
            with open(os.path.join(tmp, name), "rb") as f:
                again = hashlib.sha256(f.read()).hexdigest()
            current = hashlib.sha256(read_bytes(name)).hexdigest()
            check(again == current, "%s: a fresh export differs from ref/out (re-run export.py)" % name)
            check(info["files"][name][1] == again, "%s: reported hash" % name)


def main():
    t0 = time.perf_counter()
    for name in export.OUT_FILES:
        if not os.path.isfile(os.path.join(OUT, name)):
            print("missing ref/out/%s: run get_mnist.py, train.py and export.py first" % name)
            return 1
    with open(os.path.join(OUT, "model.json"), "r", encoding="utf-8") as f:
        model = json.load(f)  # plain lists and ints, for the pure-Python checks
    np_model = ref.load_model(os.path.join(OUT, "model.json"))
    x0 = np.frombuffer(read_bytes("inputs.bin"), dtype=np.uint8).reshape(N, 784)

    tests = [
        ("shift rounds toward negative infinity", test_shift_floor),
        ("clamp boundaries", test_clamp),
        ("argmax tie rule", test_argmax),
        ("int32 overflow detection", test_overflow),
        ("operand range checks", test_operand_checks),
        ("exported file sizes and model.json structure", lambda: test_file_sizes(model)),
        ("inputs.bin and labels.bin are the MNIST test set", test_inputs_are_mnist),
        ("pure-Python recomputation on %d images" % N_PURE, lambda: test_pure_python(model, np_model, x0)),
        ("layer equals neuron applied per output", lambda: test_layer_equals_neuron(np_model, x0)),
        ("pred.bin and trace.jsonl against the reference", lambda: test_trace_and_pred(np_model, x0)),
        ("export is deterministic and ref/out is current", test_export_is_deterministic),
    ]
    for label, fn in tests:
        before, t1 = CHECKS, time.perf_counter()
        fn()
        print("ok  %-52s %5d checks  %5.2f s" % (label, CHECKS - before, time.perf_counter() - t1))
    print("PASS: %d checks in %d groups, %.1f s" % (CHECKS, len(tests), time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
