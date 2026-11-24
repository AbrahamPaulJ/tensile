"""Train the convolutional fee-tier network in plain NumPy and quantise it to docs/spec.md.

Usage:  python examples/amm-fee-conv/train.py        (NumPy; about 10 minutes on one core; settings chosen by tune.py)

The task, the input rule and the split are in dataset.py (read that first). The network, on 288
moves (one channel):

    conv1d  1 -> 16 channels, kernel 8, stride 1    281 positions, 4,496 values
    conv1d 16 -> 16 channels, kernel 4, stride 4     70 positions, 1,120 values
    conv1d 16 -> 16 channels, kernel 4, stride 4     17 positions,   272 values
    dense  272 -> 32
    dense   32 -> 3

ReLU after every layer but the last, no padding, no pooling (the strides downsample). The layout
of a sequence is position by position (docs/spec.md section 8): value p * C + c is channel c of
position p, and a filter is K positions of C channels, weight k * C + c.

  1. Float training (Adam with decoupled weight decay, cross-entropy, the cosine schedule of
     ref/train.py) on the training split, inputs as move / 128. The epoch with the best validation
     accuracy is kept.
  2. Post-training quantisation: the power-of-two scales of every layer, chosen on the validation
     split by tensile_ref.run (the integer reference itself), by a greedy search: start from the
     largest scales that clip nothing, then for each scale in layer order try the alternatives and
     keep the best, twice over.
  3. The integer model is written to model/amm-fee-conv-288.json with the thresholds and the
     validation scores.

The test split is never built here: export.py scores it once. Deterministic for the fixed seed on
one machine (floating point; that is why the model file is committed and check.sh does not
retrain).
"""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))  # first: `train` below is ref/train.py

import dataset  # noqa: E402
import tensile_ref as ref  # noqa: E402
import train as base  # noqa: E402  (ref/train.py: Adam, softmax_xent, cosine_lr, score)

# ("conv", channels in, kernel, stride, filters) or ("dense", in, out)
ARCH = [
    ("conv", 1, 8, 1, 16),
    ("conv", 16, 4, 4, 16),
    ("conv", 16, 4, 4, 16),
    ("dense", None, 32),
    ("dense", 32, dataset.TIERS),
]
SEED = 20261124
EPOCHS = 12
BATCH = 100
LR = 1e-3
WEIGHT_DECAY = 0.5
F0 = ref.INPUT_FRAC_BITS
EVAL_CHUNK = 1000


def shapes():
    """Per layer: dict with kind, and for conv: L, C, K, S, P, F; for dense: n_in, n_out."""
    out = []
    length, channels = dataset.N_OBS, 1
    for spec in ARCH:
        if spec[0] == "conv":
            _, c, k, s, f = spec
            assert c == channels
            p = ref.conv_positions(length, k, s)
            out.append({"kind": "conv", "L": length, "C": c, "K": k, "S": s, "P": p, "F": f,
                        "n_in": length * c, "n_out": p * f})
            length, channels = p, f
        else:
            _, n_in, n_out = spec
            n_in = length * channels if n_in is None else n_in
            out.append({"kind": "dense", "n_in": n_in, "n_out": n_out})
            length, channels = 1, n_out
    return out


SHAPES = shapes()


def windows(x, sh):
    """(N, L, C) -> the patches (N, P, K * C), weight order k * C + c."""
    w = sliding_window_view(x, sh["K"], axis=1)[:, ::sh["S"]][:, :sh["P"]]  # (N, P, C, K)
    return np.ascontiguousarray(w.transpose(0, 1, 3, 2)).reshape(x.shape[0], sh["P"], sh["K"] * sh["C"])


def forward(params, x_real, keep=False):
    """x_real (N, L) floats. Returns logits, and with keep=True the cache for backward."""
    n = x_real.shape[0]
    h = x_real.reshape(n, -1, 1)
    cache = []
    count = len(SHAPES)
    for k, sh in enumerate(SHAPES):
        w, b = params[2 * k], params[2 * k + 1]
        if sh["kind"] == "conv":
            cols = windows(h, sh)
            z = cols @ w.T + b  # (N, P, F)
            entry = (cols, z)
        else:
            flat = h.reshape(n, -1)
            z = flat @ w.T + b
            entry = (flat, z)
        if keep:
            cache.append(entry)
        if k == count - 1:
            return z, cache
        h = np.maximum(z, 0.0)
        if sh["kind"] == "dense":
            h = h.reshape(n, -1, 1)


def backward(params, cache, g):
    """Gradients of every parameter, from the gradient of the loss with respect to the logits."""
    grads = [None] * len(params)
    for k in range(len(SHAPES) - 1, -1, -1):
        sh = SHAPES[k]
        w = params[2 * k]
        inp, _ = cache[k]
        if sh["kind"] == "conv":
            grads[2 * k] = np.einsum("npf,npk->fk", g, inp)
            grads[2 * k + 1] = g.sum(axis=(0, 1))
        else:
            grads[2 * k] = g.T @ inp
            grads[2 * k + 1] = g.sum(axis=0)
        if k == 0:
            break
        dinp = g @ w  # conv: (N, P, K * C); dense: (N, n_in)
        prev = SHAPES[k - 1]
        n = g.shape[0]
        if sh["kind"] == "conv":
            d = dinp.reshape(n, sh["P"], sh["K"], sh["C"])
            dx = np.zeros((n, sh["L"], sh["C"]))
            span = sh["S"] * (sh["P"] - 1) + 1
            for j in range(sh["K"]):
                dx[:, j:j + span:sh["S"], :] += d[:, :, j, :]
        else:
            dx = dinp
        _, z_prev = cache[k - 1]
        g = dx.reshape(z_prev.shape) * (z_prev > 0.0)
    return grads


def init_params(rng):
    params = []
    for k, sh in enumerate(SHAPES):
        last = k == len(SHAPES) - 1
        gain = 1.0 if last else 2.0
        if sh["kind"] == "conv":
            fan_in = sh["K"] * sh["C"]
            params += [rng.normal(0.0, np.sqrt(gain / fan_in), (sh["F"], fan_in)), np.zeros(sh["F"])]
        else:
            params += [rng.normal(0.0, np.sqrt(gain / sh["n_in"]), (sh["n_out"], sh["n_in"])), np.zeros(sh["n_out"])]
    return params


def float_eval(params, x, y):
    logits = np.concatenate([forward(params, x[i:i + EVAL_CHUNK].astype(np.float64) / 128.0)[0]
                             for i in range(0, len(x), EVAL_CHUNK)])
    loss, _ = base.softmax_xent(logits, y)
    return float((np.argmax(logits, axis=1) == y).mean()), float(loss)


def train_float(x, y, x_val, y_val, rng, epochs=EPOCHS, lr=LR, log=True):
    params = init_params(rng)
    opt = base.Adam(params, WEIGHT_DECAY, [True, False] * len(SHAPES))
    xr = x.astype(np.float64) / 128.0
    n = x.shape[0]
    steps_per_epoch = n // BATCH
    total = epochs * steps_per_epoch
    step = 0
    best = None
    for epoch in range(epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        loss_sum = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            logits, cache = forward(params, xr[idx], keep=True)
            loss, g = base.softmax_xent(logits, y[idx])
            loss_sum += loss
            opt.step(backward(params, cache, g), base.cosine_lr(lr, step, total))
            step += 1
        res = float_eval(params, x_val, y_val)
        mark = ""
        if best is None or base.score(res) > base.score(best[1]):
            best = (epoch + 1, res, [p.copy() for p in params])
            mark = "  (kept)"
        if log:
            print("  float epoch %2d  train loss %.4f  val acc %.4f  val loss %.4f  %.1f s%s"
                  % (epoch + 1, loss_sum / steps_per_epoch, res[0], res[1], time.perf_counter() - t0, mark),
                  flush=True)
    return best


def quantise(params, cfg):
    """Float parameters to an integer model dict. cfg = [fw_1, f_1, fw_2, f_2, ..., fw_L]: the
    weight fractional bits of each layer and the fractional bits of each hidden activation; the
    shift of a hidden layer is f_{k-1} + fw_k - f_k. A hidden bias gets half the shift divisor
    added, which turns the floor of the shift into round-to-nearest (docs/spec.md section 5)."""
    layers = []
    f_prev = F0
    count = len(SHAPES)
    for k, sh in enumerate(SHAPES):
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
        lay = {}
        if sh["kind"] == "conv":
            lay.update({"kind": "conv1d", "channels": sh["C"], "kernel": sh["K"], "stride": sh["S"]})
        lay.update({"in": sh["n_in"], "out": sh["n_out"], "weight_frac_bits": int(fw),
                    "shift": None if last else int(shift), "weights": wq, "bias": bq})
        layers.append(lay)
        if not last:
            f_prev = f_out
    return {"name": dataset.MODEL_NAME, "input_frac_bits": F0, "layers": layers}


def int_run(model, x):
    """Classes and logits of the integer reference, in chunks (the conv patches are large)."""
    logits, cls = [], []
    for i in range(0, len(x), EVAL_CHUNK):
        _, lg, c = ref.run(model, x[i:i + EVAL_CHUNK].astype(np.int64))
        logits.append(lg)
        cls.append(c)
    return np.concatenate(logits), np.concatenate(cls)


def int_eval(model, x, y):
    logits, cls = int_run(model, x)
    acc_bits, _ = ref.frac_bits(model)
    loss, _ = base.softmax_xent(logits.astype(np.float64) / 2.0 ** acc_bits[-1], y)
    return float((cls == y).mean()), float(loss)


def float_hidden_max(params, x):
    """Largest float activation of every hidden layer over x."""
    n = x.shape[0]
    h = x.astype(np.float64).reshape(n, -1, 1) / 128.0
    maxima = []
    for k, sh in enumerate(SHAPES[:-1]):
        w, b = params[2 * k], params[2 * k + 1]
        z = windows(h, sh) @ w.T + b if sh["kind"] == "conv" else h.reshape(n, -1) @ w.T + b
        h = np.maximum(z, 0.0)
        maxima.append(float(h.max()))
        if sh["kind"] == "dense":
            h = h.reshape(n, -1, 1)
    return maxima


def ptq_search(params, x_cal, x_val, y_val):
    """Greedy search over the power-of-two scales: start from the largest that clip nothing, then
    for each scale in order try the alternatives (weights: one bit more; activations: up to three
    bits more), keeping the best by validation score. Two passes."""
    count = len(SHAPES)
    maxima = float_hidden_max(params, x_cal)
    axes = []
    for k in range(count):
        m = int(np.floor(np.log2(127.0 / np.abs(params[2 * k]).max())))
        axes.append([m, m + 1])
        if k < count - 1:
            g = int(np.floor(np.log2(127.0 / maxima[k])))
            axes.append([g, g + 1, g + 2, g + 3])
    cfg = [axis[0] for axis in axes]
    scored = {}

    def evaluate(c):
        key = tuple(c)
        if key not in scored:
            try:
                model = quantise(params, c)
                ref.check_model(model)
                scored[key] = int_eval(model, x_val, y_val)
            except (ValueError, OverflowError):
                scored[key] = None
        return scored[key]

    best = evaluate(cfg)
    print("    start %s  val acc %.4f  val loss %.4f" % (cfg, best[0], best[1]), flush=True)
    for _ in range(2):
        for i, axis in enumerate(axes):
            for option in axis:
                trial = list(cfg)
                trial[i] = option
                res = evaluate(trial)
                if res is not None and base.score(res) > base.score(best):
                    cfg, best = trial, res
                    print("    %s  val acc %.4f  val loss %.4f  (best so far)" % (cfg, res[0], res[1]), flush=True)
    print("  %d configurations scored" % len(scored))
    return cfg, best


def split_arrays(moves, rows, thresholds):
    """(inputs as uint8 (n, N_OBS), labels, persistence tiers) of a list of dataset.samples rows."""
    x = np.frombuffer(b"".join(dataset.input_bytes(moves, end) for end, _, _ in rows), dtype=np.uint8)
    x = x.reshape(len(rows), dataset.N_OBS)
    y = np.array([dataset.tier(future, thresholds) for _, _, future in rows], dtype=np.int64)
    p = np.array([dataset.tier(past, thresholds) for _, past, _ in rows], dtype=np.int64)
    return x, y, p


def load_splits():
    ticks = dataset.load_ticks()
    moves = dataset.moves_of(ticks)
    ranges = dataset.split_ranges(len(moves))
    rows = {name: dataset.samples(moves, *ranges[name]) for name in ("train", "validation")}
    thresholds = dataset.dense_thresholds()
    x, y, _ = split_arrays(moves, rows["train"], thresholds)
    xv, yv, pv = split_arrays(moves, rows["validation"], thresholds)
    return ticks, ranges, thresholds, (x, y), (xv, yv, pv)


def main():
    t_start = time.perf_counter()
    ticks, ranges, thresholds, (x_train, y_train), (x_val, y_val, p_val) = load_splits()
    params_count = sum((sh["F"] * (sh["K"] * sh["C"] + 1)) if sh["kind"] == "conv" else sh["n_out"] * (sh["n_in"] + 1)
                       for sh in SHAPES)
    print("seed %d, numpy %s, %d ticks, train %d, validation %d (test not read here)"
          % (SEED, np.__version__, len(ticks), len(y_train), len(y_val)))
    for sh in SHAPES:
        if sh["kind"] == "conv":
            print("  conv1d L %d C %d K %d S %d -> %d positions x %d filters = %d values"
                  % (sh["L"], sh["C"], sh["K"], sh["S"], sh["P"], sh["F"], sh["n_out"]))
        else:
            print("  dense %d -> %d" % (sh["n_in"], sh["n_out"]))
    print("  %d parameters" % params_count)
    print("thresholds (the dense example's, its training split's terciles): %s" % thresholds)
    print("training labels per tier: %s; validation labels per tier: %s"
          % (np.bincount(y_train, minlength=dataset.TIERS).tolist(),
             np.bincount(y_val, minlength=dataset.TIERS).tolist()))
    val_majority = int(np.bincount(y_val, minlength=dataset.TIERS)[np.bincount(y_train).argmax()])
    val_persist = int((p_val == y_val).sum())
    print("validation baselines: majority tier of training %d of %d, persistence %d of %d"
          % (val_majority, len(y_val), val_persist, len(y_val)))

    print("[1/3] float training (%d epochs, lr %g, the best validation epoch is kept)" % (EPOCHS, LR))
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    epoch, float_val, params = train_float(x_train, y_train, x_val, y_val, rng)
    t_float = time.perf_counter() - t0
    print("  float: epoch %d kept, val acc %.4f (%.1f s)" % (epoch, float_val[0], t_float))

    print("[2/3] post-training quantisation, scale search on validation")
    t0 = time.perf_counter()
    cfg, ptq_val = ptq_search(params, x_train[:10000], x_val, y_val)
    chosen = quantise(params, cfg)
    t_ptq = time.perf_counter() - t0
    _, cls_val = int_run(chosen, x_val)
    val_correct = int((cls_val == y_val).sum())
    print("  chosen %s: integer val acc %.4f (%d of %d) (%.1f s)"
          % (cfg, ptq_val[0], val_correct, len(y_val), t_ptq))

    print("[3/3] writing model/%s.json" % dataset.MODEL_NAME)
    ref.check_model(chosen)
    out = dict(chosen)
    out["task"] = {
        "pool": "Uniswap v3 USDC/WETH 0.05%, Ethereum mainnet, 0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        "ticks": len(ticks), "ticks_sha256": dataset.sha256(dataset.TICKS_PATH),
        "n_obs": dataset.N_OBS, "gap": dataset.GAP, "horizon": dataset.HORIZON, "clip": dataset.CLIP,
        "thresholds": thresholds, "thresholds_from": "examples/amm-fee (training split terciles)",
        "split_moves": {name: list(span) for name, span in ranges.items()},
    }
    out["training"] = {
        "seed": SEED, "numpy": np.__version__, "method": "ptq", "float_epochs": EPOCHS, "lr": LR,
        "kept_epoch": epoch, "train_samples": len(y_train), "parameters": params_count,
        "validation": {"samples": len(y_val), "float_correct": int(round(float_val[0] * len(y_val))),
                       "integer_correct": val_correct, "majority_correct": val_majority,
                       "persistence_correct": val_persist},
    }
    os.makedirs(os.path.dirname(dataset.MODEL_PATH), exist_ok=True)
    with open(dataset.MODEL_PATH, "wb") as f:
        f.write(ref.model_to_json(out).encode("utf-8"))
    acc_bits, out_bits = ref.frac_bits(chosen)
    for li, lay in enumerate(chosen["layers"]):
        wc = max(ref.worst_case_abs_acc(lay["weights"], lay["bias"]))
        print("  layer %d (%s): weight_frac_bits %d, shift %s, acc frac bits %d, out frac bits %s, "
              "worst-case |acc| %d (int32 max %d)"
              % (li + 1, lay.get("kind", "dense"), lay["weight_frac_bits"], lay["shift"], acc_bits[li],
                 out_bits[li], wc, ref.INT32_MAX))
    print("RESULT validation: integer model %d, float model %d, persistence %d, majority %d, of %d"
          % (val_correct, out["training"]["validation"]["float_correct"], val_persist, val_majority, len(y_val)))
    print("RESULT time: float %.1f s, ptq %.1f s, total %.1f s" % (t_float, t_ptq, time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
