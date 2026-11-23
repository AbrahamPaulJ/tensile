"""Train the MNIST 784 -> 128 -> 64 -> 10 perceptron in plain NumPy and quantise it to docs/spec.md.

Usage:  python examples/mnist-784-128-64-10/train.py

The same flow as ref/train.py, written for any number of dense layers (ref/train.py is fixed to
one hidden layer and is not changed). Shared pieces are imported from it: the optimiser, the loss,
the shift augmentation, the dead-pixel rule and every hyperparameter except the seed.

  1. Float training (ReLU hidden layers, Adam with decoupled weight decay, cross-entropy, random
     one-pixel shifts), on x0 = pixel >> 1 as the real value x0 / 128.
  2. Post-training quantisation: a grid search over the power-of-two scales of every layer, scored
     on a held-out validation split of the training set (never on the test set). Every score comes
     from tensile_ref.run, the integer reference itself.
  3. The chosen integer model is written to model/mnist-784-128-64-10.json next to this script.

There is no quantisation-aware fine-tuning here: ref/train.py has it for one hidden layer only.
The script prints the gap between float and integer accuracy so the cost of that is visible.

Deterministic for the fixed seed on one machine (floating point: another CPU or BLAS build may
round differently). export.py in this folder turns the saved model into out/.
"""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "..", "ref")
sys.path.insert(0, REF)

import get_mnist  # noqa: E402
import tensile_ref as ref  # noqa: E402
import train as base  # noqa: E402  (ref/train.py)

MODEL_NAME = "mnist-784-128-64-10"
MODEL_PATH = os.path.join(HERE, "model", MODEL_NAME + ".json")
SIZES = [784, 128, 64, 10]
SEED = 20261124
F0 = ref.INPUT_FRAC_BITS


def float_forward(params, x_real):
    """params = [w1, b1, w2, b2, ...]. Returns (pre-activations, activations, logits)."""
    zs, hs = [], []
    h = x_real
    count = len(params) // 2
    for k in range(count):
        z = h @ params[2 * k].T + params[2 * k + 1]
        if k == count - 1:
            return zs, hs, z
        zs.append(z)
        h = np.maximum(z, 0.0)
        hs.append(h)


def float_eval(params, x0, labels, chunk=5000):
    correct, loss_sum = 0, 0.0
    for s in range(0, x0.shape[0], chunk):
        xb = x0[s:s + chunk].astype(np.float64) / 128.0
        yb = labels[s:s + chunk]
        _, _, logits = float_forward(params, xb)
        loss, _ = base.softmax_xent(logits, yb)
        loss_sum += loss * xb.shape[0]
        correct += int((np.argmax(logits, axis=1) == yb).sum())
    return correct / x0.shape[0], loss_sum / x0.shape[0]


def train_float(x0, labels, x0_val, y_val, rng):
    count = len(SIZES) - 1
    params = []
    for k in range(count):
        n_in, n_out = SIZES[k], SIZES[k + 1]
        # He initialisation before a ReLU, 1 / fan-in before the logits (as ref/train.py).
        gain = 1.0 if k == count - 1 else 2.0
        w = rng.normal(0.0, np.sqrt(gain / n_in), (n_out, n_in))
        if k == 0:
            w[:, base.dead_pixels(x0, base.MAX_SHIFT)] = 0.0
        params += [w, np.zeros(n_out)]
    opt = base.Adam(params, base.WEIGHT_DECAY, [True, False] * count)
    xb = np.empty((base.BATCH, SIZES[0]))
    n = x0.shape[0]
    steps_per_epoch = n // base.BATCH
    total = base.FLOAT_EPOCHS * steps_per_epoch
    step = 0
    for epoch in range(base.FLOAT_EPOCHS):
        perm = rng.permutation(n)
        shifts = rng.integers(-base.MAX_SHIFT, base.MAX_SHIFT + 1, (steps_per_epoch, 2))
        loss_sum = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * base.BATCH:(s + 1) * base.BATCH]
            base.fill_batch(xb, x0, idx, int(shifts[s, 0]), int(shifts[s, 1]))
            yb = labels[idx]
            # Forward. The input stays as the integers x0; the division by 128 is applied to
            # the first layer's products, as in ref/train.py.
            acts = [xb]
            zs = []
            for k in range(count):
                z = acts[-1] @ params[2 * k].T
                if k == 0:
                    z = z / 128.0
                z = z + params[2 * k + 1]
                zs.append(z)
                if k < count - 1:
                    acts.append(np.maximum(z, 0.0))
            loss, g = base.softmax_xent(zs[-1], yb)
            loss_sum += loss
            # Backward.
            grads = [None] * (2 * count)
            for k in range(count - 1, -1, -1):
                dw = g.T @ acts[k]
                if k == 0:
                    dw = dw / 128.0
                grads[2 * k] = dw
                grads[2 * k + 1] = g.sum(axis=0)
                if k > 0:
                    g = (g @ params[2 * k]) * (zs[k - 1] > 0.0)
            opt.step(grads, base.cosine_lr(base.FLOAT_LR, step, total))
            step += 1
        if epoch % 5 == 4 or epoch == base.FLOAT_EPOCHS - 1:
            acc, vloss = float_eval(params, x0_val, y_val)
            print("  float epoch %2d  train loss %.4f  val acc %.4f  val loss %.4f"
                  % (epoch + 1, loss_sum / steps_per_epoch, acc, vloss), flush=True)
    return params


def quantise(params, cfg):
    """Float parameters to an integer model dict. cfg = (fw_1, f_1, fw_2, f_2, ..., fw_L): the
    weight fractional bits of each layer and the fractional bits of each hidden activation. The
    shift of a hidden layer follows from the spec: shift_k = f_{k-1} + fw_k - f_k. As in
    ref/train.py, a hidden bias gets half of the shift divisor added, which turns the floor of the
    arithmetic shift into round-to-nearest."""
    count = len(params) // 2
    layers = []
    f_prev = F0
    for k in range(count):
        w, b = params[2 * k], params[2 * k + 1]
        fw = cfg[2 * k]
        last = k == count - 1
        wq = np.clip(np.rint(w * 2.0 ** fw), -127, 127).astype(np.int64)
        bq = np.rint(b * 2.0 ** (f_prev + fw)).astype(np.int64)
        if last:
            shift = None
        else:
            f_out = cfg[2 * k + 1]
            shift = f_prev + fw - f_out
            if shift < 0:
                raise ValueError("negative shift")
            bq = bq + ((1 << (shift - 1)) if shift > 0 else 0)
        layers.append({"in": SIZES[k], "out": SIZES[k + 1], "weight_frac_bits": int(fw),
                       "shift": None if last else int(shift), "weights": wq, "bias": bq})
        if not last:
            f_prev = f_out
    return {"name": MODEL_NAME, "input_frac_bits": F0, "layers": layers}


def int_eval(model, x0, labels):
    _, logits, cls = ref.run(model, x0)
    acc_bits, _ = ref.frac_bits(model)
    loss, _ = base.softmax_xent(logits.astype(np.float64) / 2.0 ** acc_bits[-1], labels)
    return float((cls == labels).mean()), float(loss)


def ptq_search(params, x0_cal, x0_val, y_val):
    """Pick the scales by validation score over a grid: for every layer the largest weight scale
    with no clipped weight or one bit more, for every hidden layer the largest activation scale
    with no clipped activation or up to three bits more."""
    count = len(params) // 2
    _, hs, _ = float_forward(params, x0_cal.astype(np.float64) / 128.0)
    axes = []
    for k in range(count):
        m = int(np.floor(np.log2(127.0 / np.abs(params[2 * k]).max())))
        axes.append((m, m + 1))
        if k < count - 1:
            g = int(np.floor(np.log2(127.0 / float(hs[k].max()))))
            axes.append((g, g + 1, g + 2, g + 3))
            print("  layer %d: max|w| %.4f, max output %.4f (99.9th percentile %.4f)"
                  % (k + 1, np.abs(params[2 * k]).max(), hs[k].max(), np.percentile(hs[k], 99.9)))
        else:
            print("  layer %d: max|w| %.4f" % (k + 1, np.abs(params[2 * k]).max()))
    best = None
    tried = 0
    for cfg in itertools.product(*axes):
        try:
            model = quantise(params, cfg)
            ref.check_model(model)
        except (ValueError, OverflowError):
            continue
        res = int_eval(model, x0_val, y_val)
        tried += 1
        if best is None or base.score(res) > base.score(best[1]):
            best = (cfg, res)
            print("    %s  val acc %.4f  val loss %.4f  (best so far)" % (list(cfg), res[0], res[1]),
                  flush=True)
    print("  %d configurations scored" % tried)
    return best


def main():
    t_start = time.perf_counter()
    get_mnist.ensure(verbose=False)
    x0_all = (get_mnist.load_images("train") >> 1).astype(np.uint8)
    y_all = get_mnist.load_labels("train").astype(np.int64)
    x0_test = (get_mnist.load_images("t10k") >> 1).astype(np.uint8)
    y_test = get_mnist.load_labels("t10k").astype(np.int64)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(x0_all.shape[0])
    val_idx, train_idx = np.sort(perm[:base.N_VAL]), np.sort(perm[base.N_VAL:])
    x0_train, y_train = x0_all[train_idx], y_all[train_idx]
    x0_val, y_val = x0_all[val_idx], y_all[val_idx]
    print("seed %d, numpy %s, sizes %s, train %d, validation %d, test %d"
          % (SEED, np.__version__, SIZES, x0_train.shape[0], x0_val.shape[0], x0_test.shape[0]))

    print("[1/3] float training (%d epochs)" % base.FLOAT_EPOCHS)
    t0 = time.perf_counter()
    params = train_float(x0_train, y_train, x0_val, y_val, rng)
    float_val = float_eval(params, x0_val, y_val)
    float_test = float_eval(params, x0_test, y_test)
    t_float = time.perf_counter() - t0
    print("  float: val acc %.4f, test acc %.4f (%.1f s)" % (float_val[0], float_test[0], t_float))

    print("[2/3] post-training quantisation, scale search on validation")
    t0 = time.perf_counter()
    cfg, ptq_val = ptq_search(params, x0_train[:10000], x0_val, y_val)
    chosen = quantise(params, cfg)
    ptq_test = int_eval(chosen, x0_test, y_test)
    t_ptq = time.perf_counter() - t0
    print("  chosen %s: PTQ val acc %.4f, test acc %.4f (%.1f s)"
          % (list(cfg), ptq_val[0], ptq_test[0], t_ptq))

    print("[3/3] writing %s" % os.path.relpath(MODEL_PATH, os.path.join(HERE, "..", "..")))
    ref.check_model(chosen)
    acc_bits, out_bits = ref.frac_bits(chosen)
    out = dict(chosen)
    out["float_accuracy"] = float_test[0]
    out["training"] = {
        "seed": SEED,
        "numpy": np.__version__,
        "method": "ptq",
        "float_epochs": base.FLOAT_EPOCHS,
        "validation_samples": base.N_VAL,
        "validation_accuracy": {"float": float_val[0], "ptq": ptq_val[0]},
        "test_accuracy": {"float": float_test[0], "ptq": ptq_test[0]},
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(ref.model_to_json(out).encode("utf-8"))
    for li, lay in enumerate(chosen["layers"]):
        wc = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        print("  layer %d: weight_frac_bits %d, shift %s, acc frac bits %d, out frac bits %s, "
              "worst-case |acc| %d (int32 max %d)"
              % (li + 1, lay["weight_frac_bits"], lay["shift"], acc_bits[li], out_bits[li],
                 wc, ref.INT32_MAX))
    print("RESULT method ptq")
    print("RESULT float test accuracy   %.4f" % float_test[0])
    print("RESULT integer test accuracy %.4f" % ptq_test[0])
    print("RESULT gap (float - integer) %+.2f percentage points"
          % (100.0 * (float_test[0] - ptq_test[0])))
    print("RESULT time: float %.1f s, ptq %.1f s, total %.1f s"
          % (t_float, t_ptq, time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
