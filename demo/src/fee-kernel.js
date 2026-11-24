// The off-chain arithmetic of the fee-tier page: the data rule of examples/amm-fee/dataset.py and
// the integer network of docs/spec.md section 3, in plain JavaScript. The keeper uses it to compute
// what it posts and the watcher to recompute it. Nothing here decides a dispute: the contracts do.
// demo/test/fee.mjs checks every function against examples/amm-fee/out (all 7,711 inputs, labels
// and classes, 1,000 traces) and against the Fe bytecode of `Network` on the EVM.
import { fromHex } from './abi.js';

// --- the data rule ----------------------------------------------------------------------------------

/** The embedded slice of the tick series: tick k is line `first_index + k` of data/ticks.csv. */
export function ticksOf(slice) {
  const ticks = new Int32Array(slice.count);
  ticks[0] = slice.start;
  for (let k = 1; k < slice.count; k++) ticks[k] = ticks[k - 1] + slice.deltas[k - 1];
  return ticks;
}

/** One input byte: min(|tick - previous tick|, clip). */
export const moveOf = (tick, previous, clip) => Math.min(Math.abs(tick - previous), clip);

/** tier(v): the number of thresholds t with v >= t. */
export const tierOf = (value, thresholds) => thresholds.reduce((n, t) => n + (value >= t ? 1 : 0), 0);

/** How many windows the slice holds: each needs its input, the gap and the whole target. */
export const windowCount = (ticks, task) => ticks.length - 1 - task.n_obs - task.gap - task.horizon + 1;

/**
 * Window `i` of the slice (test sample `i` of the example): the n_obs + 1 ticks behind its input,
 * the input bytes, and the label, which is the tier of the sum of the `horizon` moves that start
 * `gap` moves after the input ends. `past` is the same sum over the input's last `horizon` moves:
 * the persistence baseline.
 */
export function windowOf(ticks, task, i) {
  const move = (k) => moveOf(ticks[k + 1], ticks[k], task.clip); // move k of the slice
  const x0 = [];
  for (let k = i; k < i + task.n_obs; k++) x0.push(move(k));
  const end = i + task.n_obs;
  let future = 0;
  for (let k = end + task.gap; k < end + task.gap + task.horizon; k++) future += move(k);
  let past = 0;
  for (let k = end - task.horizon; k < end; k++) past += move(k);
  return {
    i,
    ticks: Array.from(ticks.subarray(i, end + 1)),
    x0,
    future,
    past,
    label: tierOf(future, task.thresholds),
    persistence: tierOf(past, task.thresholds),
  };
}

// --- the network ------------------------------------------------------------------------------------

/** The layers of the embedded model as typed arrays. */
export function loadNet(model) {
  return model.layers.map((layer) => ({
    in: layer.in,
    out: layer.out,
    shift: layer.shift,
    rows: layer.neurons.map((n) => new Int8Array(fromHex(n.row).buffer)),
    bias: layer.neurons.map((n) => n.bias),
  }));
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * Neuron `j` of layer `k` (1-based) on the vector `x`: bias + sum of weight * input in int32; a
 * hidden layer then shifts right arithmetically and clamps to 0..127, the last layer does neither.
 */
export function neuron(net, k, j, x) {
  const layer = net[k - 1];
  const row = layer.rows[j];
  let acc = layer.bias[j];
  for (let i = 0; i < layer.in; i++) acc += row[i] * x[i];
  if (layer.shift === null) return acc;
  const y = acc >> layer.shift;
  return y < 0 ? 0 : y > 127 ? 127 : y;
}

const layerOf = (net, k, x) => Array.from({ length: net[k - 1].out }, (_, j) => neuron(net, k, j, x));

/** Every vector after `vectors[k]`, computed from it: vectors[0] is the input, the last the logits. */
function downstream(net, vectors, k) {
  const out = vectors.slice(0, k + 1).map((v) => v.slice());
  for (let l = k + 1; l <= net.length; l++) out.push(layerOf(net, l, out[l - 1]));
  return out;
}

const asClaim = (vectors, cls = argmax(vectors[vectors.length - 1])) => ({
  x0: vectors[0],
  hidden: vectors.slice(1, -1),
  logits: vectors[vectors.length - 1],
  cls,
});
const vectorsOf = (claim) => [claim.x0, ...claim.hidden, claim.logits];

/** The whole forward pass: { x0, hidden: [layer 1, layer 2], logits, cls }. */
export const forward = (net, x0) => asClaim(downstream(net, [Array.from(x0)], 0));

// --- what a watcher does ----------------------------------------------------------------------------

/**
 * Walks a claim as docs/spec-dispute.md section 6 says: layer by layer, each neuron recomputed from
 * the claim's own vector one layer down, then the argmax of the claimed logits. Returns the first
 * check that fails, which is the challenge to send, or null if the claim is consistent throughout.
 */
export function audit(net, claim) {
  const vectors = vectorsOf(claim);
  for (let k = 1; k <= net.length; k++) {
    for (let j = 0; j < net[k - 1].out; j++) {
      const recomputed = neuron(net, k, j, vectors[k - 1]);
      if (recomputed !== vectors[k][j]) return { kind: 'neuron', layer: k, j, claimed: vectors[k][j], recomputed };
    }
  }
  const best = argmax(claim.logits);
  if (best !== claim.cls) return { kind: 'class', claimed: claim.cls, recomputed: best };
  return null;
}

// --- the lies ---------------------------------------------------------------------------------------

export const honestClaim = (net, x0) => forward(net, x0);

/** The claimed tier is wrong; every vector is honest. Caught by the argmax. */
export const lieAboutClass = (net, x0, target) => ({ ...forward(net, x0), cls: target });

/**
 * Values of layer `k` are forged until the network's answer is `target`, and everything after layer
 * `k` is computed correctly from the forgery, so every later check passes and only re-running a
 * forged neuron of layer `k` on the layer below can catch it. For a hidden layer: greedily, the one
 * activation set to 0 or 127 that most favours `target`, until the answer flips. For the last layer
 * the target's logit is raised above the others. Returns null if no forgery in that layer reaches
 * the target.
 */
export function lieInLayer(net, x0, k, target) {
  const honest = vectorsOf(forward(net, x0));
  if (k === net.length) {
    const logits = honest[k].slice();
    logits[target] = Math.max(...logits) + 1;
    return { ...asClaim([...honest.slice(0, k), logits]), forged: { layer: k, neurons: [target] } };
  }
  const margin = (vectors) => {
    const logits = vectors[vectors.length - 1];
    let rival = -Infinity;
    for (let c = 0; c < logits.length; c++) if (c !== target && logits[c] > rival) rival = logits[c];
    return logits[target] - rival;
  };
  let current = honest;
  const touched = new Set();
  while (argmax(current[net.length]) !== target && touched.size < net[k - 1].out) {
    let best = null;
    for (let i = 0; i < net[k - 1].out; i++) {
      if (touched.has(i)) continue;
      for (const value of [0, 127]) {
        if (current[k][i] === value) continue;
        const trial = current.slice(0, k + 1).map((v) => v.slice());
        trial[k][i] = value;
        const next = downstream(net, trial, k);
        const m = margin(next);
        if (best === null || m > best.m) best = { m, i, next };
      }
    }
    if (best === null) break;
    touched.add(best.i);
    current = best.next;
  }
  if (argmax(current[net.length]) !== target || touched.size === 0) return null;
  return { ...asClaim(current), forged: { layer: k, neurons: Array.from(touched).sort((a, b) => a - b) } };
}

/**
 * Correct arithmetic about an input the pool never produced: every move of the window halved, as if
 * the pool had been half as active, and the network's true output for that. No challenge can slash
 * it, because nothing in it is miscomputed. Only the consumer can refuse it, by the input hash.
 */
export function claimOnHalvedInput(net, x0) {
  return { ...forward(net, x0.map((v) => v >> 1)), doctored: true };
}

/**
 * Correct arithmetic about another window of the same series: the network's true output for the
 * window nearest to `i - day` (searching outwards, then anywhere) whose input differs from window
 * `i`'s and whose answer is `target`. A replayed or misdirected claim: no challenge can slash it,
 * and the consumer refuses it by the input hash. Returns null if no window of the slice answers
 * `target`.
 */
export function claimAboutOtherWindow(net, ticks, task, i, target, day = task.n_obs) {
  const count = windowCount(ticks, task);
  const own = windowOf(ticks, task, i).x0.join(',');
  const start = Math.max(0, i - day);
  for (let step = 0; step < count; step++) {
    for (const j of step === 0 ? [start] : [start - step, start + step]) {
      if (j < 0 || j >= count || j === i) continue;
      const w = windowOf(ticks, task, j);
      if (w.x0.join(',') === own) continue;
      const claim = forward(net, w.x0);
      if (claim.cls === target) return { ...claim, doctored: true, other: j };
    }
  }
  return null;
}
