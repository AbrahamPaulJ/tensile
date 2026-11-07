"""Train the MNIST 784 -> 32 -> 10 perceptron in plain NumPy and quantise it to docs/spec.md.

Usage:  python ref/train.py          (add --qat to force stage 3)

Stages (all deterministic for a fixed seed on one machine):
  1. Float training (ReLU hidden layer, Adam with decoupled weight decay, cross-entropy, random
     one-pixel shifts). The float model sees exactly what the integer model sees:
     x0 = pixel >> 1, as the real value x0 / 128.
  2. Post-training quantisation (PTQ): a small search over the power-of-two scales, scored on a
     held-out validation split of the training set (never on the test set).
  3. Quantisation-aware fine-tuning (QAT) with a straight-through estimator: the forward pass
     is the exact integer network, the backward pass treats rounding as the identity. It runs
     only if PTQ loses more than 0.5 percentage points on validation (or with --qat), and its
     result is kept only if it beats PTQ on validation.
  4. The chosen integer model is written to ref/model/mnist-784-32-10.json.

The integer forward pass used for every score is tensile_ref.run, the same code the test
vectors come from. export.py turns the saved model into ref/out/.
"""

import os
import sys

# One BLAS thread: the matrices are small, and it removes one source of run-to-run variation.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import get_mnist  # noqa: E402
import tensile_ref as ref  # noqa: E402

MODEL_DIR = os.path.join(HERE, "model")
MODEL_NAME = "mnist-784-32-10"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME + ".json")

SEED = 20261105
N_IN, N_HID, N_OUT = 784, 32, 10
F0 = ref.INPUT_FRAC_BITS  # input fractional bits: real input = x0 / 2^7
N_VAL = 5000  # held out from the 60,000 training images for every quantisation decision

FLOAT_EPOCHS = 30
FLOAT_LR = 2e-3
QAT_EPOCHS = 10
QAT_LR = 5e-4
BATCH = 100
WEIGHT_DECAY = 0.05  # decoupled (AdamW style), weights only
MAX_SHIFT = 1  # training augmentation: random whole-pixel shift of up to this many pixels
QAT_GAP_TRIGGER = 0.005  # run QAT if PTQ loses more than this (validation accuracy, fraction)


class Adam:
    """Adam with optional decoupled weight decay. All updates are in place, with one scratch
    array per parameter, so a step allocates nothing."""

    def __init__(self, params, weight_decay=0.0, decay_mask=None):
        self.params = params
        self.wd = weight_decay
        self.decay_mask = decay_mask or [False] * len(params)
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.tmp = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, lr):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1 = 1.0 - b1 ** self.t
        c2 = 1.0 - b2 ** self.t
        for p, g, m, v, tmp, dec in zip(self.params, grads, self.m, self.v, self.tmp,
                                        self.decay_mask):
            m *= b1
            np.multiply(g, 1.0 - b1, out=tmp)
            m += tmp
            v *= b2
            np.multiply(g, g, out=tmp)
            tmp *= 1.0 - b2
            v += tmp
            np.divide(v, c2, out=tmp)
            np.sqrt(tmp, out=tmp)
            tmp += eps
            np.divide(m, tmp, out=tmp)
            tmp *= lr / c1
            p -= tmp
            if dec and self.wd:
                np.multiply(p, lr * self.wd, out=tmp)
                p -= tmp


def cosine_lr(base, step, total):
    return base * 0.5 * (1.0 + np.cos(np.pi * step / total))


def softmax_xent(logits, labels):
    """Mean cross-entropy and its gradient with respect to the logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), labels]).mean()
    g = p
    g[np.arange(n), labels] -= 1.0
    g /= n
    return loss, g


# ---------------------------------------------------------------- float model


def float_forward(params, x_real):
    w1, b1, w2, b2 = params
    z1 = x_real @ w1.T + b1
    h = np.maximum(z1, 0.0)
    return z1, h, h @ w2.T + b2


def float_eval(params, x0, labels, chunk=5000):
    """Accuracy and mean cross-entropy of the float model on integer inputs x0 (0..127)."""
    correct, loss_sum = 0, 0.0
    for s in range(0, x0.shape[0], chunk):
        xb = x0[s:s + chunk].astype(np.float64) / 128.0
        yb = labels[s:s + chunk]
        _, _, logits = float_forward(params, xb)
        loss, _ = softmax_xent(logits, yb)
        loss_sum += loss * xb.shape[0]
        correct += int((np.argmax(logits, axis=1) == yb).sum())
    return correct / x0.shape[0], loss_sum / x0.shape[0]


def fill_batch(buf, x0, idx, dy=0, dx=0):
    """Copy the images x0[idx] into buf (float64, integer-valued), shifted by (dy, dx) pixels
    with zero fill. buf has shape (len(idx), 784)."""
    src = x0[idx].reshape(-1, 28, 28)
    dst = buf.reshape(-1, 28, 28)
    if dy == 0 and dx == 0:
        dst[...] = src
        return
    dst.fill(0.0)
    dst[:, max(dy, 0):28 + min(dy, 0), max(dx, 0):28 + min(dx, 0)] =         src[:, max(-dy, 0):28 + min(-dy, 0), max(-dx, 0):28 + min(-dx, 0)]


def dead_pixels(x0, max_shift):
    """Boolean mask (784,) of the pixels that are 0 in every image of x0 under every shift of
    up to max_shift pixels."""
    lit = (x0.max(axis=0) > 0).reshape(28, 28)
    reach = np.zeros((28, 28), dtype=bool)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            reach[max(dy, 0):28 + min(dy, 0), max(dx, 0):28 + min(dx, 0)] |=                 lit[max(-dy, 0):28 + min(-dy, 0), max(-dx, 0):28 + min(-dx, 0)]
    return ~reach.reshape(784)


def train_float(x0, labels, x0_val, y_val, rng, epochs=None, lr=None, weight_decay=None,
                max_shift=None):
    epochs = FLOAT_EPOCHS if epochs is None else epochs
    lr = FLOAT_LR if lr is None else lr
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    max_shift = MAX_SHIFT if max_shift is None else max_shift
    w1 = rng.normal(0.0, np.sqrt(2.0 / N_IN), (N_HID, N_IN))
    # A pixel that is 0 in every (shifted) training image never produces a gradient, so its
    # weights would keep their random start. Start them at 0 instead: they then stay exactly 0.
    w1[:, dead_pixels(x0, max_shift)] = 0.0
    b1 = np.zeros(N_HID)
    w2 = rng.normal(0.0, np.sqrt(1.0 / N_HID), (N_OUT, N_HID))
    b2 = np.zeros(N_OUT)
    params = [w1, b1, w2, b2]
    opt = Adam(params, weight_decay, [True, False, True, False])
    xb = np.empty((BATCH, N_IN))  # reused batch buffer, holds the integers x0 as float64
    n = x0.shape[0]
    steps_per_epoch = n // BATCH
    total = epochs * steps_per_epoch
    step = 0
    for epoch in range(epochs):
        perm = rng.permutation(n)
        shifts = rng.integers(-max_shift, max_shift + 1, (steps_per_epoch, 2))
        loss_sum = 0.0
        for k in range(steps_per_epoch):
            idx = perm[k * BATCH:(k + 1) * BATCH]
            fill_batch(xb, x0, idx, int(shifts[k, 0]), int(shifts[k, 1]))
            yb = labels[idx]
            # real input = x0 / 128; the division is applied to the small products instead
            z1 = (xb @ w1.T) / 128.0 + b1
            h = np.maximum(z1, 0.0)
            logits = h @ w2.T + b2
            loss, g = softmax_xent(logits, yb)
            loss_sum += loss
            dw2 = g.T @ h
            db2 = g.sum(axis=0)
            dz1 = (g @ w2) * (z1 > 0.0)
            dw1 = (dz1.T @ xb) / 128.0
            db1 = dz1.sum(axis=0)
            opt.step([dw1, db1, dw2, db2], cosine_lr(lr, step, total))
            step += 1
        if epoch % 5 == 4 or epoch == epochs - 1:
            acc, vloss = float_eval(params, x0_val, y_val)
            print("  float epoch %2d  train loss %.4f  val acc %.4f  val loss %.4f"
                  % (epoch + 1, loss_sum / steps_per_epoch, acc, vloss), flush=True)
    return params


# ---------------------------------------------------------------- quantisation


def quantise(params, cfg):
    """Float parameters to an integer model dict (spec sections 2, 4 and 5).

    cfg = (fw1, f1, fw2): weight fractional bits of layer 1, fractional bits of the hidden
    activation, weight fractional bits of layer 2. The hidden shift follows from the spec:
    shift1 = F0 + fw1 - f1.

    The hidden bias gets half of the shift divisor added, which turns the floor of the
    arithmetic shift into round-to-nearest at no cost at inference time.
    """
    w1, b1, w2, b2 = params
    fw1, f1, fw2 = cfg
    shift1 = F0 + fw1 - f1
    if shift1 < 0:
        raise ValueError("negative shift")
    half = (1 << (shift1 - 1)) if shift1 > 0 else 0
    w1q = np.clip(np.rint(w1 * 2.0 ** fw1), -127, 127).astype(np.int64)
    b1q = np.rint(b1 * 2.0 ** (F0 + fw1)).astype(np.int64) + half
    w2q = np.clip(np.rint(w2 * 2.0 ** fw2), -127, 127).astype(np.int64)
    b2q = np.rint(b2 * 2.0 ** (f1 + fw2)).astype(np.int64)
    return {
        "name": MODEL_NAME,
        "input_frac_bits": F0,
        "layers": [
            {"in": N_IN, "out": N_HID, "weight_frac_bits": int(fw1), "shift": int(shift1),
             "weights": w1q, "bias": b1q},
            {"in": N_HID, "out": N_OUT, "weight_frac_bits": int(fw2), "shift": None,
             "weights": w2q, "bias": b2q},
        ],
    }


def int_eval(model, x0, labels):
    """Accuracy and mean cross-entropy of the integer model, through the reference itself."""
    _, logits, cls = ref.run(model, x0)
    acc_bits, _ = ref.frac_bits(model)
    loss, _ = softmax_xent(logits.astype(np.float64) / 2.0 ** acc_bits[-1], labels)
    return float((cls == labels).mean()), float(loss)


def score(acc_loss):
    """Sort key: higher accuracy first, then lower loss."""
    return (acc_loss[0], -acc_loss[1])


def ptq_search(params, x0_cal, x0_val, y_val):
    """Pick (fw1, f1, fw2) by validation score over a small grid of power-of-two scales."""
    w1, _, w2, _ = params
    _, h, _ = float_forward(params, x0_cal.astype(np.float64) / 128.0)
    h_max = float(h.max())
    m1 = int(np.floor(np.log2(127.0 / np.abs(w1).max())))  # largest scale with no clipped weight
    m2 = int(np.floor(np.log2(127.0 / np.abs(w2).max())))
    g1 = int(np.floor(np.log2(127.0 / h_max)))  # largest hidden scale with no clipped activation
    print("  max|w1| %.4f  max|w2| %.4f  max hidden %.4f (99.9th percentile %.4f)"
          % (np.abs(w1).max(), np.abs(w2).max(), h_max, np.percentile(h, 99.9)))
    best = None
    for fw1 in (m1, m1 + 1):
        for f1 in (g1, g1 + 1, g1 + 2, g1 + 3):
            if F0 + fw1 - f1 < 0:
                continue
            for fw2 in (m2, m2 + 1):
                cfg = (fw1, f1, fw2)
                res = int_eval(quantise(params, cfg), x0_val, y_val)
                print("    fw1 %d  hidden bits %d  fw2 %d  shift %2d  val acc %.4f  val loss %.4f"
                      % (fw1, f1, fw2, F0 + fw1 - f1, res[0], res[1]))
                if best is None or score(res) > score(best[1]):
                    best = (cfg, res)
    return best


def train_qat(params, cfg, x0, labels, x0_val, y_val, rng):
    """Fine-tune with the exact integer forward pass and straight-through gradients.

    float64 holds every integer here exactly (accumulators are far below 2^53), so this
    forward pass is bit-identical to tensile_ref.run on the quantised weights.
    """
    fw1, f1, fw2 = cfg
    shift1 = F0 + fw1 - f1
    half = (1 << (shift1 - 1)) if shift1 > 0 else 0
    lim1 = 127.0 / 2.0 ** fw1
    lim2 = 127.0 / 2.0 ** fw2
    w1, b1, w2, b2 = [p.copy() for p in params]
    np.clip(w1, -lim1, lim1, out=w1)
    np.clip(w2, -lim2, lim2, out=w2)
    latent = [w1, b1, w2, b2]
    opt = Adam(latent)
    xi = np.empty((BATCH, N_IN))  # reused batch buffer, holds the integers x0 as float64
    w1q = np.empty_like(w1)
    n = x0.shape[0]
    steps_per_epoch = n // BATCH
    total = QAT_EPOCHS * steps_per_epoch
    step = 0
    best = None
    for epoch in range(QAT_EPOCHS):
        perm = rng.permutation(n)
        shifts = rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, (steps_per_epoch, 2))
        for k in range(steps_per_epoch):
            idx = perm[k * BATCH:(k + 1) * BATCH]
            fill_batch(xi, x0, idx, int(shifts[k, 0]), int(shifts[k, 1]))
            yb = labels[idx]
            np.multiply(w1, 2.0 ** fw1, out=w1q)
            np.rint(w1q, out=w1q)
            np.clip(w1q, -127.0, 127.0, out=w1q)
            b1q = np.rint(b1 * 2.0 ** (F0 + fw1)) + half
            w2q = np.clip(np.rint(w2 * 2.0 ** fw2), -127, 127)
            b2q = np.rint(b2 * 2.0 ** (f1 + fw2))
            acc1 = xi @ w1q.T + b1q
            hq = np.clip(np.floor(acc1 / 2.0 ** shift1), 0.0, 127.0)
            acc2 = hq @ w2q.T + b2q
            _, g = softmax_xent(acc2 / 2.0 ** (f1 + fw2), yb)
            dw2 = g.T @ (hq / 2.0 ** f1)
            db2 = g.sum(axis=0)
            open_mask = (acc1 >= 0.0) & (acc1 < 128.0 * 2.0 ** shift1)  # neither clamp is active
            dz1 = (g @ (w2q / 2.0 ** fw2)) * open_mask
            dw1 = (dz1.T @ xi) / 128.0
            db1 = dz1.sum(axis=0)
            opt.step([dw1, db1, dw2, db2], cosine_lr(QAT_LR, step, total))
            np.clip(w1, -lim1, lim1, out=w1)
            np.clip(w2, -lim2, lim2, out=w2)
            step += 1
        res = int_eval(quantise(latent, cfg), x0_val, y_val)
        mark = ""
        if best is None or score(res) > score(best[1]):
            best = ([p.copy() for p in latent], res, epoch + 1)
            mark = "  (best so far)"
        print("  qat epoch %2d  val acc %.4f  val loss %.4f%s" % (epoch + 1, res[0], res[1], mark),
              flush=True)
    return best


def main():
    force_qat = "--qat" in sys.argv[1:]
    t_start = time.perf_counter()
    get_mnist.ensure(verbose=False)
    x0_all = (get_mnist.load_images("train") >> 1).astype(np.uint8)
    y_all = get_mnist.load_labels("train").astype(np.int64)
    x0_test = (get_mnist.load_images("t10k") >> 1).astype(np.uint8)
    y_test = get_mnist.load_labels("t10k").astype(np.int64)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(x0_all.shape[0])
    val_idx, train_idx = np.sort(perm[:N_VAL]), np.sort(perm[N_VAL:])
    x0_train, y_train = x0_all[train_idx], y_all[train_idx]
    x0_val, y_val = x0_all[val_idx], y_all[val_idx]
    print("seed %d, numpy %s, train %d, validation %d, test %d"
          % (SEED, np.__version__, x0_train.shape[0], x0_val.shape[0], x0_test.shape[0]))

    print("[1/4] float training (%d epochs)" % FLOAT_EPOCHS)
    t0 = time.perf_counter()
    params = train_float(x0_train, y_train, x0_val, y_val, rng)
    float_val = float_eval(params, x0_val, y_val)
    float_test = float_eval(params, x0_test, y_test)
    t_float = time.perf_counter() - t0
    print("  float: val acc %.4f, test acc %.4f (%.1f s)" % (float_val[0], float_test[0], t_float))

    print("[2/4] post-training quantisation, scale search on validation")
    t0 = time.perf_counter()
    cfg, ptq_val = ptq_search(params, x0_train[:10000], x0_val, y_val)
    ptq_model = quantise(params, cfg)
    ptq_test = int_eval(ptq_model, x0_test, y_test)
    t_ptq = time.perf_counter() - t0
    print("  chosen fw1 %d, hidden bits %d, fw2 %d: PTQ val acc %.4f, test acc %.4f (%.1f s)"
          % (cfg[0], cfg[1], cfg[2], ptq_val[0], ptq_test[0], t_ptq))

    ptq_gap = float_val[0] - ptq_val[0]
    chosen, chosen_val, method = ptq_model, ptq_val, "ptq"
    qat_val = qat_test = None
    qat_epoch = None
    t_qat = 0.0
    if force_qat or ptq_gap > QAT_GAP_TRIGGER:
        print("[3/4] quantisation-aware fine-tuning (%d epochs, PTQ validation gap %.2f points)"
              % (QAT_EPOCHS, 100.0 * ptq_gap))
        t0 = time.perf_counter()
        latent, qat_val, qat_epoch = train_qat(params, cfg, x0_train, y_train, x0_val, y_val, rng)
        qat_model = quantise(latent, cfg)
        qat_test = int_eval(qat_model, x0_test, y_test)
        t_qat = time.perf_counter() - t0
        print("  QAT (epoch %d): val acc %.4f, test acc %.4f (%.1f s)"
              % (qat_epoch, qat_val[0], qat_test[0], t_qat))
        if score(qat_val) > score(ptq_val):
            chosen, chosen_val, method = qat_model, qat_val, "qat"
    else:
        print("[3/4] QAT not needed: PTQ validation gap %.2f points is within %.2f (force it with --qat)"
              % (100.0 * ptq_gap, 100.0 * QAT_GAP_TRIGGER))

    print("[4/4] writing ref/model/%s.json" % MODEL_NAME)
    ref.check_model(chosen)
    int_test = int_eval(chosen, x0_test, y_test)
    acc_bits, out_bits = ref.frac_bits(chosen)
    out = dict(chosen)
    out["float_accuracy"] = float_test[0]
    out["training"] = {
        "seed": SEED,
        "numpy": np.__version__,
        "method": method,
        "float_epochs": FLOAT_EPOCHS,
        "qat_epochs": QAT_EPOCHS,
        "qat_best_epoch": qat_epoch,
        "validation_samples": N_VAL,
        "validation_accuracy": {"float": float_val[0], "ptq": ptq_val[0],
                                "qat": None if qat_val is None else qat_val[0]},
        "test_accuracy": {"float": float_test[0], "ptq": ptq_test[0],
                          "qat": None if qat_test is None else qat_test[0]},
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(ref.model_to_json(out).encode("utf-8"))

    for li, lay in enumerate(chosen["layers"]):
        wc = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        print("  layer %d: weight_frac_bits %d, shift %s, acc frac bits %d, out frac bits %s, "
              "worst-case |acc| %d (int32 max %d)"
              % (li + 1, lay["weight_frac_bits"], lay["shift"], acc_bits[li], out_bits[li],
                 wc, ref.INT32_MAX))
    gap = float_test[0] - int_test[0]
    print("RESULT method %s" % method)
    print("RESULT float test accuracy   %.4f" % float_test[0])
    print("RESULT integer test accuracy %.4f" % int_test[0])
    print("RESULT gap (float - integer) %+.2f percentage points" % (100.0 * gap))
    print("RESULT time: float %.1f s, ptq %.1f s, qat %.1f s, total %.1f s"
          % (t_float, t_ptq, t_qat, time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
