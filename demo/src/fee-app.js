// The fee-tier page. It draws, encodes and displays. Every verdict, refusal, fee and gas figure it
// shows came back from the Fe bytecode of examples/amm-fee on the EVM in chain.js; the arithmetic in
// fee-kernel.js is only what the keeper posts and the watcher checks, and decides nothing. No button
// is gated by what the page expects a contract to say: a refused call is shown with its own error.
import { hex, keccak, pack } from './abi.js';
import {
  argmax, audit, claimAboutOtherWindow, forward, honestClaim, lieAboutClass, lieInLayer, loadNet, ticksOf, windowCount, windowOf,
} from './fee-kernel.js';
import { BOND, FeeDeployment, STATUS, WINDOW, hashOfInput } from './fee-protocol.js';

const assets = window.TENSILE_FEE_ASSETS;
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString('en-US');
const eth = (wei) => `${(Number(wei / 10n ** 16n) / 100).toLocaleString('en-US', { maximumFractionDigits: 2 })} ETH`;
const paint = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const pct = (part, whole) => `${((Number(part) / Number(whole)) * 100).toFixed(1)}%`;
const median = (values) => values.map(Number).sort((a, b) => a - b)[values.length >> 1];
const esc = (text) => String(text).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

const task = assets.task;
const N = task.n_obs;
const net = loadNet(assets.model);
const ticks = ticksOf(assets.ticks);
const WINDOWS = windowCount(ticks, task);
const SPAN = N + task.gap + task.horizon; // moves from a window's first input move to its last target move
const FEES = assets.fees.by_tier;
const feeText = (fee) => `${fmt(fee)} (${(fee / 10000).toFixed(2)}%)`;
const sizes = [net[0].in, ...net.map((l) => l.out)];
const PARAMETERS = net.reduce((n, l) => n + l.in * l.out + l.out, 0);
const LAYER_NAME = (k, j) => (k === net.length ? `Output ${j} (tier ${j})` : `Layer ${k} neuron ${j}`);

const S = {
  d: null,
  i: 0,
  w: null,
  honest: null,
  evm: null,
  oracle: null,
  prepared: null,
  target: null,
  current: null,
  busy: false,
};

// --- acts -----------------------------------------------------------------------------------------
function show(act) {
  document.querySelectorAll('.act').forEach((el) => el.classList.toggle('on', el.id === `act-${act}`));
  document.querySelectorAll('#steps button').forEach((b) => b.classList.toggle('on', b.dataset.act === act));
  window.scrollTo({ top: 0 });
  if (act === 'claim') prepare();
  if (act === 'apply') renderApply();
}
document.querySelectorAll('#steps button').forEach((b) => b.addEventListener('click', () => show(b.dataset.act)));

/** Runs one user action at a time, and shows a failure instead of swallowing it. */
function action(fn) {
  return async (...args) => {
    if (S.busy) return;
    S.busy = true;
    S.locked = true;
    document.body.style.cursor = 'progress';
    try {
      await fn(...args);
    } catch (e) {
      console.error(e);
      $('l-engine').textContent = `Page error: ${e.message}`;
    } finally {
      S.busy = false;
      S.locked = false;
      document.body.style.cursor = '';
    }
  };
}

// --- the ledger strip -----------------------------------------------------------------------------
async function refreshLedger() {
  const { d } = S;
  const { accounts } = d.chain;
  $('l-keeper').textContent = eth(await d.chain.balance(accounts.Keeper.address));
  $('l-watcher').textContent = eth(await d.chain.balance(accounts.Watcher.address));
  $('l-arbiter').textContent = eth(await d.chain.balance(d.arbiter));
  const burned = await d.burned();
  $('l-arbiter-burned').textContent = burned > 0n ? `${eth(burned)} of it burned` : '';
  for (const [name, id] of [['Keeper', 'l-keeper-credit'], ['Watcher', 'l-watcher-credit']]) {
    const c = await d.creditOf(name);
    $(id).textContent = c > 0n ? `+${eth(c)} to withdraw` : '';
  }
  if (S.current) {
    const info = await d.claimInfo(S.current.id);
    S.current.status = info.status;
    const name = STATUS[info.status];
    $('l-claim').innerHTML = `#${S.current.id} says tier ${info.cls}: <span class="st-${name}">${name}</span>`;
    const left = info.deadline - d.chain.now;
    $('l-deadline').textContent = info.status !== 1 ? '' : left > 0n ? `${left / 60n}m ${String(left % 60n).padStart(2, '0')}s left` : 'window closed';
  } else {
    $('l-claim').textContent = 'none';
    $('l-deadline').textContent = '';
  }
  if (S.oracle) {
    S.oracle.fee = await d.fee();
    S.oracle.moves = await d.moves();
    $('l-fee').textContent = feeText(S.oracle.fee);
    $('l-moves').textContent = `${fmt(S.oracle.moves)} moves stored`;
  } else {
    $('l-fee').textContent = 'no window observed';
    $('l-moves').textContent = '';
  }
}

function logTx(who, call, tx) {
  const row = document.createElement('tr');
  const outcome = tx.reverted
    ? `<span class="rev">refused: ${tx.why?.name ?? 'reverted'}</span>`
    : `<span class="okk">${(tx.events ?? []).join(', ') || 'ok'}</span>`;
  row.innerHTML = `<td>${who}</td><td class="mono">${call}</td><td class="n">${fmt(tx.gasUsed ?? tx.gas)}</td><td>${outcome}</td>`;
  $('log').tBodies[0].prepend(row);
}

// --- 1: the window --------------------------------------------------------------------------------
const SVG = 'http://www.w3.org/2000/svg';
let tickMin = Infinity;
let tickMax = -Infinity;
for (const t of ticks) {
  if (t < tickMin) tickMin = t;
  if (t > tickMax) tickMax = t;
}
const blockOf = (k) => assets.ticks.first_block + task.step_blocks * k;

/** The whole slice as a band of each bucket's lowest and highest tick, and the picked window. */
function drawOverview() {
  const svg = $('overview');
  const W = 1000;
  const H = 150;
  const buckets = W / 2;
  const per = ticks.length / buckets;
  const y = (t) => (4 + ((tickMax - t) / (tickMax - tickMin)) * (H - 8)).toFixed(1);
  const top = [];
  const bottom = [];
  for (let b = 0; b < buckets; b++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = Math.floor(b * per); k < Math.min(ticks.length, Math.floor((b + 1) * per) + 1); k++) {
      if (ticks[k] < lo) lo = ticks[k];
      if (ticks[k] > hi) hi = ticks[k];
    }
    const x = ((b + 0.5) * (W / buckets)).toFixed(1);
    top.push(`${x},${y(hi)}`);
    bottom.unshift(`${x},${y(lo)}`);
  }
  const x = (k) => ((k / (ticks.length - 1)) * W).toFixed(1);
  const i = S.i;
  svg.innerHTML = `<polygon class="ov-band" points="${top.join(' ')} ${bottom.join(' ')}"/>`
    + `<rect class="ov-input" x="${x(i)}" y="0" width="${Math.max(2, x(i + N) - x(i)).toFixed(1)}" height="${H}"/>`
    + `<rect class="ov-target" x="${x(i + N + task.gap)}" y="0" width="${Math.max(1.5, x(i + SPAN) - x(i + N + task.gap)).toFixed(1)}" height="${H}"/>`;
  $('ov-left').textContent = `block ${fmt(blockOf(0))}`;
  $('ov-right').textContent = `block ${fmt(blockOf(ticks.length - 1))}`;
}

$('overview').addEventListener('click', (e) => {
  const r = $('overview').getBoundingClientRect();
  const k = ((e.clientX - r.left) / r.width) * (ticks.length - 1);
  selectWindow(Math.round(k - SPAN / 2));
});

/** The window close up: its ticks, then every move as a bar, input, gap and target shaded. */
function drawZoom() {
  const svg = $('zoom');
  const w = S.w;
  const span = ticks.subarray(S.i, S.i + SPAN + 1);
  let lo = Infinity;
  let hi = -Infinity;
  for (const t of span) {
    if (t < lo) lo = t;
    if (t > hi) hi = t;
  }
  if (hi === lo) hi = lo + 1;
  const x = (k) => 10 + (k * 680) / SPAN;
  const y = (t) => 14 + ((hi - t) / (hi - lo)) * 136;
  const region = (a, b, cls, y0, h) => `<rect class="${cls}" x="${x(a).toFixed(1)}" y="${y0}" width="${(x(b) - x(a)).toFixed(1)}" height="${h}"/>`;
  const parts = [
    region(0, N, 'z-input', 6, 238),
    region(N, N + task.gap, 'z-gap', 6, 238),
    region(N + task.gap, SPAN, 'z-target', 6, 238),
    `<polyline class="z-line" points="${Array.from(span, (t, k) => `${x(k).toFixed(1)},${y(t).toFixed(1)}`).join(' ')}"/>`,
  ];
  const move = (k) => Math.min(Math.abs(ticks[S.i + k + 1] - ticks[S.i + k]), task.clip);
  for (let k = 0; k < SPAN; k++) {
    const v = move(k);
    const h = (v / task.clip) * 78;
    const cls = k < N ? 'z-bar' : k < N + task.gap ? 'z-bar gap' : 'z-bar target';
    parts.push(`<rect class="${cls}" x="${(x(k) + 0.4).toFixed(1)}" y="${(242 - h).toFixed(1)}" width="${Math.max(0.8, x(k + 1) - x(k) - 0.8).toFixed(1)}" height="${Math.max(0.6, h).toFixed(1)}"/>`);
  }
  parts.push(`<line class="z-axis" x1="10" x2="690" y1="242.5" y2="242.5"/>`);
  svg.innerHTML = parts.join('');
  $('zoom-label').innerHTML = `Window ${fmt(S.i)} of ${fmt(WINDOWS)}: blocks ${fmt(blockOf(S.i))} to ${fmt(blockOf(S.i + SPAN))}, ticks ${fmt(lo)} to ${fmt(hi)}. `
    + '<span class="key k-input">input, 144 moves (24 h)</span> <span class="key k-gap">gap, 6 (1 h, the dispute window)</span> <span class="key k-target">target, 24 (4 h)</span>';
  $('bytes').innerHTML = w.x0.map((v) => `<span${v === task.clip ? ' class="clip"' : ''}>${v}</span>`).join('');
  $('bytes-label').textContent = `The 144 input bytes, min(|tick - previous tick|, ${task.clip}), oldest first`;
}

function renderTiers() {
  const w = S.w;
  const [t1, t2] = task.thresholds;
  const tier = (t) => `<b>tier ${t}</b>, fee ${feeText(FEES[t])}`;
  const model = S.evm
    ? `${tier(S.evm.cls)}${S.evm.cls === w.label ? '' : ' <span class="false">(wrong)</span>'}`
    : 'running...';
  $('tiers').innerHTML = `<tbody>
    <tr><th>What happened in the 4 hours after the gap (from the data)</th><td>the 24 target moves sum to ${fmt(w.future)}: ${tier(w.label)}</td></tr>
    <tr><th>The model, <code>Network.classify</code> on the EVM in this tab</th><td>${model}</td></tr>
    <tr><th>Persistence: the last 4 hours of the input</th><td>sum ${fmt(w.past)}: tier ${w.persistence}</td></tr>
  </tbody>`;
  $('tiers-label').textContent = `Tiers of this window: a sum below ${fmt(t1)} is tier 0, below ${fmt(t2)} tier 1, else tier 2 (thresholds from the training split)`;
}

let evmTimer = null;
function selectWindow(i) {
  if (S.locked) {
    $('w-range').value = String(S.i);
    return;
  }
  S.i =Math.max(0, Math.min(WINDOWS - 1, i));
  S.w = windowOf(ticks, task, S.i);
  S.honest = forward(net, S.w.x0);
  S.evm = null;
  S.oracle = null;
  S.prepared = null;
  S.current = null;
  S.target = null;
  $('w-range').value = String(S.i);
  $('to-observe').disabled = true;
  $('to-claim').disabled = true;
  drawOverview();
  drawZoom();
  renderTiers();
  resetObserve();
  $('window-note').textContent = 'Running the network on the EVM in this tab...';
  clearTimeout(evmTimer);
  evmTimer = setTimeout(runEvm, 180);
}

/** The network on this window, as Fe bytecode: Network.trace as a call, Network.classify as a transaction. */
async function runEvm() {
  if (S.busy || !S.d) {
    evmTimer = setTimeout(runEvm, 150);
    return;
  }
  S.busy = true;
  try {
    const i = S.i;
    const x0 = S.w.x0;
    const t = await S.d.trace(x0);
    const c = await S.d.classifyTx(x0);
    if (S.i !== i) return;
    const js = S.honest;
    const agree = c.cls === js.cls
      && t.logits.every((v, j) => v === js.logits[j])
      && t.hidden.every((h, k) => h.every((v, j) => v === js.hidden[k][j]));
    S.evm = { cls: c.cls, gas: c.gas, agree };
    logTx('Anyone', `Network.classify(window ${fmt(i)})`, c);
    renderTiers();
    $('window-note').innerHTML = `<b>${fmt(c.gas)} gas</b> to run the whole network as one transaction in this tab: tier ${c.cls}. `
      + (agree
        ? `The keeper's and the watcher's arithmetic in this page gives the same 32 + 16 hidden values and ${js.logits.length} outputs as <code>Network.trace</code>, to the last bit.`
        : '<b>The page\'s arithmetic differs from the EVM\'s</b>: please report this.');
    $('to-observe').disabled = false;
    await refreshLedger();
  } catch (e) {
    console.error(e);
    $('l-engine').textContent = `Page error: ${e.message}`;
  } finally {
    S.busy = false;
  }
}

$('w-range').max = String(WINDOWS - 1);
$('w-range').addEventListener('input', (e) => selectWindow(Number(e.target.value)));
const stepBy = (delta) => () => { if (!S.busy) selectWindow(S.i + delta); };
$('w-back').addEventListener('click', stepBy(-1));
$('w-fwd').addEventListener('click', stepBy(1));
$('w-day-back').addEventListener('click', stepBy(-N));
$('w-day-fwd').addEventListener('click', stepBy(N));
const seek = (test) => {
  for (let step = 1; step < WINDOWS; step++) {
    const j = (S.i + step) % WINDOWS;
    if (test(j)) return selectWindow(j);
  }
  return null;
};
document.querySelectorAll('[data-next-tier]').forEach((b) => b.addEventListener('click', () => {
  if (S.busy) return;
  const t = Number(b.dataset.nextTier);
  seek((j) => windowOf(ticks, task, j).label === t);
}));
$('w-wrong').addEventListener('click', () => {
  if (S.busy) return;
  seek((j) => {
    const w = windowOf(ticks, task, j);
    return forward(net, w.x0).cls !== w.label;
  });
});
$('to-observe').addEventListener('click', () => show('observe'));

// --- 2: observe -----------------------------------------------------------------------------------
function resetObserve() {
  $('feed').disabled = false;
  $('feed-count').textContent = '';
  $('feed-bar').style.width = '0%';
  $('observe-out').className = 'finding small';
  $('observe-out').textContent = `A fresh FeeOracle is deployed for window ${fmt(S.i)}, so it has seen nothing yet.`;
  $('words').innerHTML = '';
  $('words-out').className = 'finding small';
  $('words-out').textContent = 'Nothing observed yet.';
  $('observe-note').textContent = '';
}

$('feed').addEventListener('click', action(async () => {
  const { d, w } = S;
  const i = S.i;
  $('feed').disabled = true;
  await d.newOracle();
  S.oracle = null;
  const gas = [];
  let first = null;
  let last = null;
  for (let k = 0; k < w.ticks.length; k++) {
    const tx = await d.observeTick(w.ticks[k], 'Keeper');
    if (tx.reverted) throw new Error(`observe() refused: ${tx.why?.name}`);
    if (k === 0) {
      first = tx;
      logTx('Keeper', 'observe() #1, the starting tick', tx);
    } else {
      gas.push(tx.gasUsed);
    }
    last = tx;
    if (k % 5 === 4 || k === w.ticks.length - 1) {
      $('feed-count').textContent = `${k + 1} of ${w.ticks.length} ticks`;
      $('feed-bar').style.width = `${(((k + 1) / w.ticks.length) * 100).toFixed(1)}%`;
      await paint();
    }
  }
  logTx('Keeper', `observe() #${w.ticks.length}`, last);
  const words = await d.inputWords(N);
  const page = pack(w.x0);
  const got = words.bytes ?? new Uint8Array(0);
  const equal = hex(got) === hex(page);
  S.oracle = { i, hash: hex(keccak(got)), next: i + w.ticks.length, equal, observed: w.ticks.length };
  $('observe-out').className = 'finding small good';
  $('observe-out').innerHTML = `${w.ticks.length} real ticks went through <code>setTick</code> on the test pool and <code>observe()</code>, one every 50 blocks: `
    + `<b>${fmt(await d.moves())} moves stored</b>. Gas of <code>observe()</code>: ${fmt(first.gasUsed)} the first time (it stores only the starting tick), `
    + `then ${fmt(median(gas))} typically (lowest ${fmt(Math.min(...gas.map(Number)))}, highest ${fmt(Math.max(...gas.map(Number)))}).`;
  const rows = [];
  for (let k = 0; k * 32 < got.length; k++) {
    const mine = hex(got.subarray(k * 32, k * 32 + 32));
    const ours = hex(page.subarray(k * 32, k * 32 + 32));
    rows.push(`<tr><th>word ${k}</th><td class="mono hexw">0x${mine}</td><td>${mine === ours ? '<span class="okk">equal</span>' : '<span class="false">differs</span>'}</td></tr>`);
  }
  $('words').innerHTML = `<thead><tr><th></th><th><code>inputWords(144)</code>, from the contract</th><th>page</th></tr></thead><tbody>${rows.join('')}</tbody>`;
  $('words-out').className = `finding small ${equal ? 'good' : 'bad'}`;
  $('words-out').innerHTML = equal
    ? `<b>The contract's packed input equals the page's 144 bytes</b>, byte for byte. Its hash, <code class="hexw">0x${S.oracle.hash}</code>, is what <code>apply</code> will require a claim to have committed to.`
    : '<b>The contract built other bytes than the page</b>: please report this.';
  $('observe-note').textContent = 'The rule is the one the model was trained with (examples/amm-fee/dataset.py), so what a keeper computes from the chain is what the model saw.';
  $('to-claim').disabled = false;
  await refreshLedger();
}));
$('to-claim').addEventListener('click', () => show('claim'));

// --- 3: the claim ---------------------------------------------------------------------------------
const lieKind = () => document.querySelector('input[name="lie"]:checked').value;

function renderTargets() {
  const box = $('targets');
  box.replaceChildren();
  const kind = lieKind();
  const truth = S.honest.cls;
  for (let c = 0; c < FEES.length; c++) {
    const b = document.createElement('button');
    b.innerHTML = `${c}<i>fee ${(FEES[c] / 10000).toFixed(2)}%</i>`;
    b.disabled = kind === 'honest' || c === truth;
    if (kind === 'honest' ? c === truth : c === S.target) b.classList.add('on');
    b.addEventListener('click', () => {
      S.target = c;
      prepare();
    });
    box.append(b);
  }
}

function renderCells(id, values, forged = [], named = null) {
  const grid = $(id);
  grid.replaceChildren();
  values.forEach((v, j) => {
    const cell = document.createElement('div');
    cell.textContent = v;
    cell.title = `neuron ${j}: ${v}`;
    cell.style.backgroundColor = `rgba(var(--cell), ${(v / 127).toFixed(3)})`;
    cell.style.color = v > 72 ? 'var(--accent-ink)' : 'var(--ink)';
    if (forged.includes(j)) cell.classList.add('forged');
    if (named === j) cell.classList.add('named');
    grid.append(cell);
  });
}

function renderLogits(values, changed = []) {
  const box = $('logits');
  box.replaceChildren();
  const low = Math.min(...values, 0);
  const high = Math.max(...values);
  const top = argmax(values);
  values.forEach((v, j) => {
    const row = document.createElement('div');
    row.className = `logit${j === top ? ' best' : ''}${changed.includes(j) ? ' forged' : ''}`;
    const width = high === low ? 0 : (((v - low) / (high - low)) * 100).toFixed(1);
    row.innerHTML = `<span>${j}</span><span class="track"><span class="fill" style="display:block;width:${width}%"></span></span><span class="v">${fmt(v)}</span>`;
    box.append(row);
  });
}

function renderNetwork(claim, finding = null) {
  const forged = claim.forged ?? { layer: 0, neurons: [] };
  const named = (k) => (finding?.kind === 'neuron' && finding.layer === k ? finding.j : null);
  renderCells('h1', claim.hidden[0], forged.layer === 1 ? forged.neurons : [], named(1));
  renderCells('h2', claim.hidden[1], forged.layer === 2 ? forged.neurons : [], named(2));
  renderLogits(claim.logits, forged.layer === 3 ? forged.neurons : []);
}

const changed = (a, b) => a.flatMap((v, j) => (v !== b[j] ? [j] : []));

function prepare() {
  S.prepared = null;
  $('post').disabled = true;
  if (!S.w) return;
  const kind = lieKind();
  if (kind !== 'honest' && (S.target === null || S.target === S.honest.cls)) S.target = (S.honest.cls + 1) % FEES.length;
  renderTargets();
  const { w, honest, target } = S;
  let claim = null;
  if (kind === 'honest') claim = honestClaim(net, w.x0);
  if (kind === 'class') claim = lieAboutClass(net, w.x0, target);
  if (kind === 'layer1') claim = lieInLayer(net, w.x0, 1, target);
  if (kind === 'layer2') claim = lieInLayer(net, w.x0, 2, target);
  if (kind === 'layer3') claim = lieInLayer(net, w.x0, 3, target);
  if (kind === 'input') claim = claimAboutOtherWindow(net, ticks, task, S.i, target);
  $('c-true-class').textContent = `tier ${honest.cls}`;
  $('claim-title').textContent = S.evm
    ? `Running it on-chain cost ${fmt(S.evm.gas)} gas. A keeper claims the tier instead, with a bond.`
    : 'A keeper claims the tier instead, with a bond.';
  if (!claim) {
    $('c-claim-class').textContent = `tier ${target}`;
    for (const id of ['c-claim-h1', 'c-claim-h2', 'c-claim-logits', 'c-claim-input']) $(id).textContent = '';
    $('c-claim-h1').textContent = 'no such claim found that reaches this tier';
    return;
  }
  const lying = claim.cls !== honest.cls;
  $('c-claim-class').innerHTML = `<span class="${lying ? 'false' : ''}">tier ${claim.cls}</span>`;
  const otherInput = Boolean(claim.doctored);
  const say = (diff, total) => (diff.length
    ? `<span class="false">${diff.length} ${otherInput ? 'differ' : 'forged'}</span>${diff.length < total ? ` (${diff.join(', ')})` : ''}`
    : `the same ${total}`);
  const d1 = changed(claim.hidden[0], honest.hidden[0]);
  const d2 = changed(claim.hidden[1], honest.hidden[1]);
  const d3 = changed(claim.logits, honest.logits);
  $('c-claim-h1').innerHTML = say(d1, 32);
  $('c-claim-h2').innerHTML = say(d2, 16);
  $('c-claim-logits').innerHTML = say(d3, 3);
  if (claim.forged && claim.forged.layer < net.length) {
    const after = claim.forged.layer === 1 ? 'Layer 2 and the outputs' : 'The outputs';
    $(claim.forged.layer === 1 ? 'c-claim-h2' : 'c-claim-logits').innerHTML += `. ${after} are correct for the forgery`;
  }
  if (otherInput) {
    const gap = claim.other - S.i;
    $('c-claim-input').innerHTML = `<span class="false">window ${fmt(claim.other)}'s bytes</span>, ${fmt(Math.abs(gap))} windows ${gap < 0 ? 'earlier' : 'later'}; every value after it is right for them`;
  } else {
    $('c-claim-input').textContent = 'the same 144 bytes';
  }
  renderNetwork(claim);
  S.prepared = { kind, claim };
  const ready = S.oracle && S.oracle.i === S.i;
  $('post').disabled = !ready;
  $('claim-note').textContent = ready
    ? 'The bond is 1 ETH of play money. The challenge window is one hour of chain time: the gap before the fee would apply.'
    : 'Feed the window through the oracle first (step 2), so that there is something to apply the claim to.';
}

document.querySelectorAll('input[name="lie"]').forEach((input) => input.addEventListener('change', () => prepare()));

$('post').addEventListener('click', action(async () => {
  const { d } = S;
  const { kind, claim } = S.prepared;
  $('claim-note').textContent = 'Posting...';
  await paint();
  const tx = await d.postClaim('Keeper', claim);
  logTx('Keeper', 'claim(...) with 1 ETH', tx);
  if (tx.reverted) {
    $('claim-note').textContent = `The contract refused the claim: ${tx.why.name}, ${tx.why.meaning}.`;
    return;
  }
  const id = (await d.claimCount()) - 1n;
  S.current = {
    id,
    kind,
    claim,
    finding: audit(net, claim),
    postGas: tx.gasUsed,
    inputHash: hashOfInput(claim.x0),
    applied: null,
  };
  $('claim-note').textContent = `Claim #${id} is on-chain. It cost ${fmt(tx.gasUsed)} gas, ${pct(tx.gasUsed, S.evm.gas)} of running the network.`;
  await refreshLedger();
  renderDispute();
  show('dispute');
}));

// --- 4: the dispute -------------------------------------------------------------------------------
function describe(c) {
  const f = c.finding;
  if (!f && c.inputHash !== S.oracle.hash) {
    return {
      html: `<b>Every value is right for the input the claim names</b>, so the arbiter has nothing to slash. But that input is not the oracle's: `
        + `the claim committed to <code class="hexw">0x${c.inputHash.slice(0, 16)}...</code>, the oracle's window hashes to <code class="hexw">0x${S.oracle.hash.slice(0, 16)}...</code>. `
        + 'Step 5 shows what FeeOracle does with it.',
      tone: 'bad',
    };
  }
  if (!f) return { html: 'The watcher recomputed every value and <b>every value matches</b>. There is nothing to challenge.', tone: 'good' };
  if (f.kind === 'class') {
    return { html: `<b>The tier is wrong.</b> The posted outputs give tier ${f.recomputed}, the claim says ${f.claimed}.`, tone: 'bad' };
  }
  return { html: `<b>${LAYER_NAME(f.layer, f.j)} is wrong.</b> The claim says ${fmt(f.claimed)}, re-running it gives ${fmt(f.recomputed)}.`, tone: 'bad' };
}

function checksFor(f) {
  if (f?.kind === 'neuron') {
    const k = f.layer;
    const out = k === net.length ? `${sizes[k]} outputs` : `${sizes[k]} values of layer ${k}`;
    const input = k === 1 ? `${sizes[0]} input bytes` : `${sizes[k - 1]} values of layer ${k - 1}`;
    return [
      ['range', `Claim exists, and neuron ${f.j} is in layer ${k}`],
      ['InputMismatch', `The ${input} presented hash to the claim's commitment`],
      ['OutputMismatch', `The ${out} presented hash to the commitment`],
      ['NotCommitted', 'The neuron\'s weights verify against the model root'],
      ['rerun', `Re-run this one neuron: ${sizes[k - 1]} multiply-adds, the same Fe code as the whole network`],
      ['settle', 'The claim is open, within its window, and the values differ: slash it'],
    ];
  }
  return [
    ['range', 'Claim exists'],
    ['OutputMismatch', `The ${sizes[net.length]} outputs presented hash to the commitment`],
    ['rerun', 'Take the largest output, the lowest index winning a tie'],
    ['settle', 'The claim is open, within its window, and the tiers differ: slash it'],
  ];
}

const WHERE = {
  UnknownClaim: 'range',
  NeuronOutOfRange: 'range',
  InputMismatch: 'InputMismatch',
  OutputMismatch: 'OutputMismatch',
  NotCommitted: 'NotCommitted',
  ClaimNotOpen: 'settle',
  TooLate: 'settle',
  NoFraud: 'settle',
};

function renderChecks(list) {
  $('checks').replaceChildren(...list.map(([, text]) => {
    const li = document.createElement('li');
    li.textContent = text;
    return li;
  }));
}

/** Lights the list up to where the contract stopped, from the error the contract returned. */
async function replayChecks(list, tx) {
  const items = [...$('checks').children];
  const stopAt = tx.reverted ? WHERE[tx.why.name] ?? null : null;
  for (let i = 0; i < list.length; i++) {
    await pause(150);
    if (stopAt === list[i][0]) {
      items[i].classList.add('no');
      items[i].textContent = `${list[i][1]}. Refused: ${tx.why.name}, ${tx.why.meaning}.`;
      for (let k = i + 1; k < list.length; k++) items[k].classList.add('skip');
      return;
    }
    items[i].classList.add('ok');
  }
}

const LEAVES = sizes.slice(1).reduce((a, b) => a + b, 0);
const leafOf = (f) => (f?.kind === 'neuron' ? sizes.slice(1, f.layer).reduce((a, b) => a + b, 0) + f.j : null);

/** The model's Merkle tree, an odd node carried up a level, with the path of one leaf. */
function drawTree(leaf = null) {
  const counts = [LEAVES];
  while (counts[counts.length - 1] > 1) counts.push(Math.ceil(counts[counts.length - 1] / 2));
  const width = 560;
  const height = 104;
  const levelY = (l) => height - 7 - (l * (height - 14)) / (counts.length - 1);
  const xs = [Array.from({ length: LEAVES }, (_, i) => 8 + (i * (width - 16)) / (LEAVES - 1))];
  for (let l = 1; l < counts.length; l++) {
    const below = xs[l - 1];
    xs.push(Array.from({ length: counts[l] }, (_, i) => (below[i * 2 + 1] === undefined ? below[i * 2] : (below[i * 2] + below[i * 2 + 1]) / 2)));
  }
  const parts = [];
  for (let l = 0; l + 1 < counts.length; l++) {
    for (let i = 0; i < counts[l]; i++) {
      const onPath = leaf !== null && leaf >> l === i;
      parts.push(`<line class="te${onPath ? ' path' : ''}" x1="${xs[l][i].toFixed(1)}" y1="${levelY(l).toFixed(1)}" x2="${xs[l + 1][i >> 1].toFixed(1)}" y2="${levelY(l + 1).toFixed(1)}"/>`);
    }
  }
  for (let l = 0; l < counts.length; l++) {
    for (let i = 0; i < counts[l]; i++) {
      let cls = 'tn';
      if (leaf !== null) {
        const here = leaf >> l;
        if (i === here) cls += ' path';
        else if (i === (here ^ 1)) cls += ' sib';
      }
      if (l === counts.length - 1) cls += ' root';
      const r = l === counts.length - 1 ? 6 : l === 0 ? 3 : 3.6;
      parts.push(`<circle class="${cls}" cx="${xs[l][i].toFixed(1)}" cy="${levelY(l).toFixed(1)}" r="${r}"/>`);
    }
  }
  $('tree').innerHTML = parts.join('');
  $('tree-label').textContent = `Merkle proof of the neuron's weights, ${LEAVES} neurons in one root`;
}

function renderGasBars(hitGas) {
  const rows = [['Whole network', S.evm.gas, false], ['Posting the claim', S.current.postGas, false]];
  if (hitGas) rows.push(['This challenge', hitGas, true]);
  const most = Math.max(...rows.map((r) => Number(r[1])));
  $('gas-bars').replaceChildren(...rows.map(([label, gas, hit]) => {
    const row = document.createElement('div');
    row.className = `bar${hit ? ' hit' : ''}`;
    row.innerHTML = `<span>${label}</span><span class="track"><span class="fill" style="display:block;width:${((Number(gas) / most) * 68).toFixed(2)}%"></span><span class="v">${fmt(gas)} gas</span></span>`;
    return row;
  }));
  const A = assets.anvil;
  const honestPath = A.calls.claim + A.calls.finalize;
  $('gas-honest').innerHTML = `At this size the dispute saves little. On anvil, <code>claim</code> plus <code>finalize</code> cost ${fmt(honestPath)} gas, `
    + `${pct(honestPath, A.classify)} of the ${fmt(A.classify)} a whole run costs, and the consumer pays for <code>apply</code> on top `
    + `(examples/amm-fee/RESULTS.md, sections 3 and 4). A claim has a fixed cost that a network of ${fmt(PARAMETERS)} weights and biases hardly outweighs; the saving comes with size.`;
}

function renderDispute() {
  const c = S.current;
  if (!c) return;
  const text = describe(c);
  $('finding').innerHTML = text.html;
  $('finding').className = `finding ${text.tone}`;
  $('dispute-title').textContent = c.finding
    ? 'Anyone can check the claim. One neuron proves it false.'
    : c.inputHash === S.oracle.hash ? 'An honest claim. Try to break it anyway.' : 'Nothing to slash: the arithmetic is right. The input is not.';
  $('send-named').disabled = !c.finding;
  $('send-named').textContent = c.finding ? 'Send this challenge' : 'Nothing to send';
  $('tries').open = !c.finding;
  $('verdict').className = 'verdict';
  $('verdict').textContent = '';
  $('dispute-note').textContent = '';
  renderChecks(checksFor(c.finding));
  drawTree(leafOf(c.finding));
  renderGasBars(null);
  renderNetwork(c.claim, c.finding);
}

async function sendChallenge(finding, label) {
  const c = S.current;
  const list = checksFor(finding);
  renderChecks(list);
  drawTree(leafOf(finding));
  $('verdict').className = 'verdict';
  $('dispute-note').textContent = 'Mining the challenge...';
  await paint();
  const tx = await S.d.challenge('Watcher', c.id, c.claim, finding);
  logTx('Watcher', label, tx);
  await replayChecks(list, tx);
  const v = $('verdict');
  if (tx.reverted) {
    if (tx.why.name === 'NoFraud') {
      v.textContent = 'CLAIM STANDS';
      v.className = 'verdict stands show';
    }
    $('dispute-note').innerHTML = `The contract refused: <b>${tx.why.name}</b>, ${tx.why.meaning}. The failed challenge cost the watcher ${fmt(tx.gasUsed)} gas and changed nothing.`;
  } else {
    renderGasBars(tx.gasUsed);
    v.textContent = 'SLASHED';
    v.className = 'verdict slashed show';
    $('dispute-note').innerHTML = `<b>${fmt(tx.gasUsed)} gas settled it, ${pct(tx.gasUsed, S.evm.gas)} of the ${fmt(S.evm.gas)} the whole run cost in this tab.</b> `
      + 'Half the 1 ETH bond is now the watcher\'s to withdraw; the other half is burned. A slashed claim can never set the fee.';
  }
  await refreshLedger();
}

$('send-named').addEventListener('click', action(async () => {
  const f = S.current.finding;
  await sendChallenge(f, f.kind === 'class' ? 'challengeClass(...)' : `challengeNeuron${f.layer}(j=${f.j})`);
}));
$('try-class').addEventListener('click', action(async () => {
  if (S.current) await sendChallenge({ kind: 'class' }, 'challengeClass(...)');
}));
$('pick-layer').addEventListener('change', () => {
  $('pick-j').max = String(sizes[Number($('pick-layer').value)] - 1);
});
$('try-neuron').addEventListener('click', action(async () => {
  const k = Number($('pick-layer').value);
  const j = Number($('pick-j').value);
  if (S.current) await sendChallenge({ kind: 'neuron', layer: k, j }, `challengeNeuron${k}(j=${j})`);
}));
$('to-apply').addEventListener('click', () => show('apply'));

// --- 5: the fee -----------------------------------------------------------------------------------
function applySay(html, tone = '') {
  $('apply-out').innerHTML = html;
  $('apply-out').className = `finding small ${tone}`;
}

async function renderFee() {
  if (!S.oracle) {
    $('fee-value').textContent = '?';
    $('fee-detail').textContent = 'No window observed yet (step 2).';
    return;
  }
  await refreshLedger();
  const { fee, moves } = S.oracle;
  $('fee-value').textContent = fmt(fee);
  const a = S.oracle.applied;
  const until = N + task.gap + task.horizon;
  $('fee-detail').innerHTML = a
    ? `${feeText(fee)} in hundredths of a basis point. Claim #${a.id} set tier ${a.tier}; it is in force while <code>moves()</code>, now ${fmt(moves)}, is below ${until}: the ${task.horizon} intervals after the ${task.gap}-interval gap. `
      + `What the pool actually did in those 4 hours, from the data: <b>tier ${S.w.label}</b> (the moves sum to ${fmt(S.w.future)})${a.tier === S.w.label ? '' : ', so the model was wrong here'}.`
    : `${feeText(fee)}: the default, because no claim has been applied. <code>moves()</code> is ${fmt(moves)}.`;
}

function renderApply() {
  const c = S.current;
  if (!c) applySay('No claim yet.');
  else {
    const name = STATUS[c.status] ?? 'open';
    applySay(`Claim #${c.id}, tier ${c.claim.cls}, is <b class="st-${name}">${name}</b>.`, name === 'slashed' ? 'bad' : name === 'final' ? 'good' : '');
  }
  $('apply').textContent = c ? `apply(${c.id}, ${N})` : `apply(id, ${N})`;
  renderFee();
}

$('forward').addEventListener('click', action(async () => {
  if (!S.oracle) {
    S.d.chain.advance(WINDOW);
    await refreshLedger();
    applySay('The clock moved one hour.');
    return;
  }
  const from = S.oracle.next;
  const clock = S.d.chain.now;
  let real = 0;
  for (let k = 0; k < task.gap; k++) {
    if (S.oracle.next < ticks.length) {
      const tx = await S.d.observeTick(ticks[S.oracle.next], 'Keeper');
      if (tx.reverted) throw new Error(`observe() refused: ${tx.why?.name}`);
      S.oracle.next++;
      real++;
    } else {
      S.d.mine(50n);
    }
  }
  await renderFee();
  applySay(`The oracle observed the next ${real} real ticks of the series (data rows ${fmt(assets.ticks.first_index + from)} to ${fmt(assets.ticks.first_index + from + real - 1)}): `
    + `${task.gap * task.step_blocks} blocks and ${(S.d.chain.now - clock) / 60n} minutes of chain time passed. <code>moves()</code> is now ${fmt(S.oracle.moves)}.`);
}));

$('finalize').addEventListener('click', action(async () => {
  const c = S.current;
  if (!c) return;
  const tx = await S.d.finalize('Keeper', c.id);
  logTx('Keeper', `finalize(${c.id})`, tx);
  await refreshLedger();
  if (tx.reverted) {
    applySay(`The arbiter refused: <b>${tx.why.name}</b>, ${tx.why.meaning}.`, 'bad');
    return;
  }
  c.finalizeGas = tx.gasUsed;
  applySay(`Claim #${c.id} is <b class="st-final">final</b> and the bond is back with the keeper. <code>result(${c.id})</code> now answers tier ${(await S.d.result(c.id)).cls}.`, 'good');
}));

$('apply').addEventListener('click', action(async () => {
  const c = S.current;
  if (!c || !S.oracle) return;
  const tx = await S.d.apply('Keeper', c.id, BigInt(N));
  logTx('Keeper', `apply(${c.id}, ${N})`, tx);
  $('apply-code').classList.remove('lit');
  if (tx.reverted) {
    const why = tx.why;
    let more = '';
    if (why.name === 'NotThisWindow') {
      more = ` The claim committed to the hash of window ${fmt(c.claim.other ?? '?')}'s bytes, not of the ${N} moves this oracle observed, so it can never set this pool's fee, final or not.`;
      $('apply-code').classList.add('lit');
    } else if (why.name === 'NotFinal' || why.name === 'empty revert') {
      more = ' The input matched, but the arbiter has no result for this claim: it is open or slashed.';
    }
    applySay(`FeeOracle refused: <b>${why.name}</b>, ${why.meaning}.${more} <code>fee()</code> is unchanged.`, 'bad');
    await renderFee();
    return;
  }
  S.oracle.applied = { id: c.id, tier: c.claim.cls };
  c.applyGas = tx.gasUsed;
  await renderFee();
  applySay(`<code>apply</code> rebuilt the window from storage, found the claim's input hash equal to it, read <code>result(${c.id})</code> and set the fee: <b>${feeText(tx.fee)}</b>.`, 'good');
  if (c.finalizeGas) {
    const total = c.postGas + c.finalizeGas + tx.gasUsed;
    $('route-note').innerHTML = `This decision by the claim route, from the receipts in this tab: claim ${fmt(c.postGas)} + finalize ${fmt(c.finalizeGas)} + apply ${fmt(tx.gasUsed)} = <b>${fmt(total)} gas</b>, `
      + `against ${fmt(S.evm.gas)} for one whole run of <code>Network.classify</code> (which would still need a transaction to store the fee). At this size the two are close; the observations cost the same either way.`;
  }
}));

$('again').addEventListener('click', () => show('claim'));
$('restart').addEventListener('click', () => show('window'));

// --- start ----------------------------------------------------------------------------------------
function renderStatic() {
  const t = assets.ticks;
  $('window-lede').innerHTML = `${esc(task.pool)}. Its tick (one tick is a 0.01% price step) at the end of every ${task.step_blocks}th block, about every 10 minutes: `
    + `the test split, which the model never trained on, ${fmt(t.count)} ticks and ${fmt(WINDOWS)} windows. The model reads 144 moves (24 hours) and picks the fee tier for the 4 hours that start an hour later. `
    + `On all ${fmt(assets.metrics.test_samples)} it is right ${fmt(assets.metrics.model_correct)} times, persistence ${fmt(assets.metrics.persistence_correct)}.`;
  $('overview-label').textContent = `The pool's tick, ${fmt(t.count)} observations (data rows ${fmt(t.first_index)} to ${fmt(t.first_index + t.count - 1)}). Click to pick a window.`;
  $('move-code').textContent = `// ${assets.excerpts.move.file}, line ${assets.excerpts.move.line}\n${assets.excerpts.move.code}`;
  $('apply-code').textContent = `// ${assets.excerpts.apply.file}, line ${assets.excerpts.apply.line}\n${assets.excerpts.apply.code}`;
  if (window.location.protocol === 'file:') $('to-digits').href = 'tensile-demo.html';
}

async function boot() {
  renderStatic();
  // The window oracle_check.sh uses: the first test sample whose reference tier is not the middle one.
  let first = 0;
  while (first < WINDOWS - 1 && forward(net, windowOf(ticks, task, first).x0).cls === 1) first++;
  selectWindow(first);
  drawTree();
  renderChecks(checksFor({ kind: 'neuron', layer: 1, j: 0 }));
  await paint();
  const started = performance.now();
  S.d = await FeeDeployment.create(assets);
  const ms = performance.now() - started;
  const root = await S.d.modelRoot();
  if (root !== assets.model.root) throw new Error('the deployed root differs from the exported one');
  $('l-engine').textContent = `EVM (Osaka), ${S.d.deployed} Fe contracts in ${fmt(Math.round(ms))} ms. No wallet, no network, play money.`;
  $('l-engine').title = 'Arbiter, Network, Rows0, TestPool and FeeOracle, from fe build of examples/amm-fee';
  await refreshLedger();
  window.tensileFee = { S, BOND, ready: true };
}
boot().catch((e) => {
  console.error(e);
  $('l-engine').textContent = `The EVM failed to start: ${e.message}`;
});
