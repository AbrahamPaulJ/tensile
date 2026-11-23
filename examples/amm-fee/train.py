"""Train the fee-tier perceptron 144 -> 32 -> 16 -> 3 in plain NumPy and quantise it to docs/spec.md.

Usage:  python examples/amm-fee/train.py

The task, the input rule and the time-ordered split are in dataset.py (read that first). Here:

  1. The tier thresholds are fixed as quantiles of the target over the training split only.
  2. Float training (ReLU hidden layers, Adam with decoupled weight decay, cross-entropy) on the
     training split, inputs as the real value move / 128. The epoch with the best validation
     accuracy is kept. The optimiser, the loss and the learning-rate schedule are ref/train.py's.
  3. Post-training quantisation: a grid search over the power-of-two scales of every layer, scored
     on the validation split by tensile_ref.run, the integer reference itself.
  4. The chosen integer model is written to model/amm-fee-144-32-16-3.json, with the thresholds and
     the validation scores of the model and of the two baselines.

The test split is not read by any decision here: it is scored once, by export.py. Deterministic
for the fixed seed on one machine (floating point: another CPU or BLAS build may round
differently, which is why the model file is committed and check.sh does not retrain).
"""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))  # first: `train` below is ref/train.py

import dataset  # noqa: E402
import tensile_ref as ref  # noqa: E402
import train as base  # noqa: E402  (ref/train.py: Adam, softmax_xent, cosine_lr, score)

SIZES = [dataset.N_OBS, 32, 16, dataset.TIERS]
SEED = 20261124
EPOCHS = 30
BATCH = 100
LR = 5e-4
WEIGHT_DECAY = 0.05
F0 = ref.INPUT_FRAC_BITS


def split_arrays(moves, rows, thresholds):
    """(inputs as uint8 (n, N_OBS), labels, persistence tiers) of a list of dataset.samples rows."""
    x = np.frombuffer(b"".join(dataset.input_bytes(moves, end) for end, _, _ in rows), dtype=np.uint8)
    x = x.reshape(len(rows), dataset.N_OBS)
    y = np.array([dataset.tier(future, thresholds) for _, _, future in rows], dtype=np.int64)
    p = np.array([dataset.tier(past, thresholds) for _, past, _ in rows], dtype=np.int64)
    return x, y, p


def float_forward(params, x_real):
    h = x_real
    count = len(params) // 2
    hs = []
    for k in range(count):
        z = h @ params[2 * k].T + params[2 * k + 1]
        if k == count - 1:
            return hs, z
        h = np.maximum(z, 0.0)
        hs.append(h)


def float_eval(params, x, y):
    _, logits = float_forward(params, x.astype(np.float64) / 128.0)
    loss, _ = base.softmax_xent(logits, y)
    return float((np.argmax(logits, axis=1) == y).mean()), float(loss)


def train_float(x, y, x_val, y_val, rng):
    count = len(SIZES) - 1
    params = []
    for k in range(count):
        gain = 1.0 if k == count - 1 else 2.0
        params += [rng.normal(0.0, np.sqrt(gain / SIZES[k]), (SIZES[k + 1], SIZES[k])), np.zeros(SIZES[k + 1])]
    opt = base.Adam(params, WEIGHT_DECAY, [True, False] * count)
    xr = x.astype(np.float64) / 128.0
    n = x.shape[0]
    steps_per_epoch = n // BATCH
    total = EPOCHS * steps_per_epoch
    step = 0
    best = None
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        loss_sum = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            acts = [xr[idx]]
            zs = []
            for k in range(count):
                z = acts[-1] @ params[2 * k].T + params[2 * k + 1]
                zs.append(z)
                if k < count - 1:
                    acts.append(np.maximum(z, 0.0))
            loss, g = base.softmax_xent(zs[-1], y[idx])
            loss_sum += loss
            grads = [None] * (2 * count)
            for k in range(count - 1, -1, -1):
                grads[2 * k] = g.T @ acts[k]
                grads[2 * k + 1] = g.sum(axis=0)
                if k > 0:
                    g = (g @ params[2 * k]) * (zs[k - 1] > 0.0)
            opt.step(grads, base.cosine_lr(LR, step, total))
            step += 1
        res = float_eval(params, x_val, y_val)
        mark = ""
        if best is None or base.score(res) > base.score(best[1]):
            best = (epoch + 1, res, [p.copy() for p in params])
            mark = "  (kept)"
        print("  float epoch %2d  train loss %.4f  val acc %.4f  val loss %.4f%s"
              % (epoch + 1, loss_sum / steps_per_epoch, res[0], res[1], mark), flush=True)
    return best


def quantise(params, cfg):
    """Float parameters to an integer model dict. cfg = (fw_1, f_1, fw_2, f_2, ..., fw_L): the
    weight fractional bits of each layer and the fractional bits of each hidden activation. The
    shift of a hidden layer follows from the spec: shift_k = f_{k-1} + fw_k - f_k. A hidden bias
    gets half of the shift divisor added, which turns the floor of the shift into round-to-nearest
    (docs/spec.md section 5)."""
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
    return {"name": dataset.MODEL_NAME, "input_frac_bits": F0, "layers": layers}


def int_eval(model, x, y):
    _, logits, cls = ref.run(model, x.astype(np.int64))
    acc_bits, _ = ref.frac_bits(model)
    loss, _ = base.softmax_xent(logits.astype(np.float64) / 2.0 ** acc_bits[-1], y)
    return float((cls == y).mean()), float(loss)


def ptq_search(params, x_cal, x_val, y_val):
    """Pick the scales by validation score over a grid: for every layer the largest weight scale
    with no clipped weight or one bit more, for every hidden layer the largest activation scale
    with no clipped activation or up to three bits more."""
    count = len(params) // 2
    hs, _ = float_forward(params, x_cal.astype(np.float64) / 128.0)
    axes = []
    for k in range(count):
        m = int(np.floor(np.log2(127.0 / np.abs(params[2 * k]).max())))
        axes.append((m, m + 1))
        if k < count - 1:
            g = int(np.floor(np.log2(127.0 / float(hs[k].max()))))
            axes.append((g, g + 1, g + 2, g + 3))
    best = None
    tried = 0
    for cfg in itertools.product(*axes):
        try:
            model = quantise(params, cfg)
            ref.check_model(model)
        except (ValueError, OverflowError):
            continue
        res = int_eval(model, x_val, y_val)
        tried += 1
        if best is None or base.score(res) > base.score(best[1]):
            best = (cfg, res)
            print("    %s  val acc %.4f  val loss %.4f  (best so far)" % (list(cfg), res[0], res[1]), flush=True)
    print("  %d configurations scored" % tried)
    return best


def main():
    t_start = time.perf_counter()
    ticks = dataset.load_ticks()
    moves = dataset.moves_of(ticks)
    ranges = dataset.split_ranges(len(moves))
    rows = {name: dataset.samples(moves, lo, hi) for name, (lo, hi) in ranges.items()}
    thresholds = dataset.quantile_thresholds([future for _, _, future in rows["train"]])
    x_train, y_train, _ = split_arrays(moves, rows["train"], thresholds)
    x_val, y_val, p_val = split_arrays(moves, rows["validation"], thresholds)
    print("seed %d, numpy %s, sizes %s, %d ticks, train %d, validation %d (test %d, not read here)"
          % (SEED, np.__version__, SIZES, len(ticks), len(rows["train"]), len(rows["validation"]),
             len(rows["test"])))
    print("thresholds (training split quantiles of the next-window sum): %s" % thresholds)
    print("training labels per tier: %s; validation labels per tier: %s"
          % (np.bincount(y_train, minlength=dataset.TIERS).tolist(),
             np.bincount(y_val, minlength=dataset.TIERS).tolist()))
    val_majority = int(np.bincount(y_val, minlength=dataset.TIERS)[np.bincount(y_train).argmax()])
    val_persist = int((p_val == y_val).sum())
    print("validation baselines: majority tier of training %d of %d, persistence %d of %d"
          % (val_majority, len(y_val), val_persist, len(y_val)))

    print("[1/3] float training (%d epochs, the best validation epoch is kept)" % EPOCHS)
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
    _, _, cls_val = ref.run(chosen, x_val.astype(np.int64))
    val_correct = int((cls_val == y_val).sum())
    print("  chosen %s: integer val acc %.4f (%d of %d) (%.1f s)"
          % (list(cfg), ptq_val[0], val_correct, len(y_val), t_ptq))

    print("[3/3] writing model/%s.json" % dataset.MODEL_NAME)
    ref.check_model(chosen)
    out = dict(chosen)
    out["task"] = {
        "pool": "Uniswap v3 USDC/WETH 0.05%, Ethereum mainnet, 0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        "first_block": dataset.FIRST_BLOCK, "step_blocks": dataset.STEP, "ticks": len(ticks),
        "ticks_sha256": dataset.sha256(dataset.TICKS_PATH),
        "n_obs": dataset.N_OBS, "gap": dataset.GAP, "horizon": dataset.HORIZON, "clip": dataset.CLIP,
        "thresholds": thresholds,
        "split_moves": {name: list(span) for name, span in ranges.items()},
    }
    out["training"] = {
        "seed": SEED, "numpy": np.__version__, "method": "ptq", "float_epochs": EPOCHS,
        "kept_epoch": epoch, "train_samples": len(rows["train"]),
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
        print("  layer %d: weight_frac_bits %d, shift %s, acc frac bits %d, out frac bits %s, "
              "worst-case |acc| %d (int32 max %d)"
              % (li + 1, lay["weight_frac_bits"], lay["shift"], acc_bits[li], out_bits[li], wc, ref.INT32_MAX))
    print("RESULT validation: integer model %d, float model %d, persistence %d, majority %d, of %d"
          % (val_correct, out["training"]["validation"]["float_correct"], val_persist, val_majority, len(y_val)))
    print("RESULT time: float %.1f s, ptq %.1f s, total %.1f s" % (t_float, t_ptq, time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
