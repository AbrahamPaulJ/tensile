// The page. It draws, encodes and displays; everything it reports as computed or decided came back
// from the Fe bytecode running on the EVM in chain.js. No button is gated by what the page expects
// the contract to say: a refused call is shown with the contract's own error.
import { BOND, Deployment, STATUS, WINDOW, argmax } from './protocol.js';

const assets = window.TENSILE_ASSETS;
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString('en-US');
const eth = (wei) => {
  const value = Number(wei / 10n ** 16n) / 100;
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })} ETH`;
};
const paint = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const S = {
  d: null,
  x0: null,
  honest: null,
  runGas: null,
  label: null,
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
}
document.querySelectorAll('#steps button').forEach((b) => b.addEventListener('click', () => show(b.dataset.act)));

/** Runs one user action at a time, and shows a failure instead of swallowing it. */
function action(fn) {
  return async (...args) => {
    if (S.busy) return;
    S.busy = true;
    document.body.style.cursor = 'progress';
    try {
      await fn(...args);
    } catch (e) {
      console.error(e);
      $('l-engine').textContent = `Page error: ${e.message}`;
    } finally {
      S.busy = false;
      document.body.style.cursor = '';
    }
  };
}

// --- the ledger strip -----------------------------------------------------------------------------
async function refreshLedger() {
  const { d } = S;
  const { accounts } = d.chain;
  $('l-claimant').textContent = eth(await d.chain.balance(accounts.Claimant.address));
  $('l-watcher').textContent = eth(await d.chain.balance(accounts.Watcher.address));
  $('l-arbiter').textContent = eth(await d.chain.balance(d.arbiter));
  const credit = async (name, id) => {
    const c = await d.creditOf(name);
    $(id).textContent = c > 0n ? `+${eth(c)} to withdraw` : '';
  };
  await credit('Claimant', 'l-claimant-credit');
  await credit('Watcher', 'l-watcher-credit');
  if (S.current) {
    const info = await d.claimInfo(S.current.id);
    S.current.status = info.status;
    const name = STATUS[info.status];
    $('l-claim').innerHTML = `#${S.current.id} says ${info.cls}: <span class="st-${name}">${name}</span>`;
    const left = info.deadline - d.chain.now;
    if (info.status === 1) {
      if (left > 0n) {
        const m = left / 60n;
        const s = left % 60n;
        $('l-deadline').textContent = `${m}m ${String(s).padStart(2, '0')}s left`;
      } else {
        $('l-deadline').textContent = 'window closed';
      }
    } else {
      $('l-deadline').textContent = '';
    }
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

// --- 1: drawing ------------------------------------------------------------------------------------
const pad = $('pad');
const pctx = pad.getContext('2d');
pctx.lineCap = 'round';
pctx.lineJoin = 'round';

function clearPad() {
  pctx.fillStyle = '#000';
  pctx.fillRect(0, 0, pad.width, pad.height);
}
clearPad();

let drawing = false;
let last = null;
function point(e) {
  const r = pad.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * pad.width, y: ((e.clientY - r.top) / r.height) * pad.height };
}
pad.addEventListener('pointerdown', (e) => {
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  last = point(e);
  pctx.fillStyle = '#fff';
  pctx.beginPath();
  pctx.arc(last.x, last.y, 10, 0, 2 * Math.PI);
  pctx.fill();
});
pad.addEventListener('pointermove', (e) => {
  if (!drawing) return;
  const p = point(e);
  pctx.strokeStyle = '#fff';
  pctx.lineWidth = 20;
  pctx.beginPath();
  pctx.moveTo(last.x, last.y);
  pctx.lineTo(p.x, p.y);
  pctx.stroke();
  last = p;
});
let pending = null;
const stop = () => {
  if (!drawing) return;
  drawing = false;
  clearTimeout(pending);
  pending = setTimeout(() => runDrawn(), 220);
};
pad.addEventListener('pointerup', stop);
pad.addEventListener('pointercancel', stop);

/**
 * The drawing, processed as MNIST images were: the ink's bounding box scaled to fit 20 x 20 and
 * centred on its centre of mass in 28 x 28. Returns x0: pixel >> 1 (docs/spec.md section 2).
 */
function preprocess() {
  const { data } = pctx.getImageData(0, 0, pad.width, pad.height);
  let minX = pad.width;
  let minY = pad.height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < pad.height; y++) {
    for (let x = 0; x < pad.width; x++) {
      if (data[(y * pad.width + x) * 4] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const scale = 20 / Math.max(w, h);
  const small = document.createElement('canvas');
  small.width = 28;
  small.height = 28;
  const sctx = small.getContext('2d', { willReadFrequently: true });
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = 'high';
  sctx.fillStyle = '#000';
  sctx.fillRect(0, 0, 28, 28);
  const dw = w * scale;
  const dh = h * scale;
  sctx.drawImage(pad, minX, minY, w, h, (28 - dw) / 2, (28 - dh) / 2, dw, dh);
  const first = sctx.getImageData(0, 0, 28, 28).data;
  let mass = 0;
  let mx = 0;
  let my = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = first[(y * 28 + x) * 4];
      mass += v;
      mx += v * x;
      my += v * y;
    }
  }
  const dx = Math.round(13.5 - mx / mass);
  const dy = Math.round(13.5 - my / mass);
  sctx.fillRect(0, 0, 28, 28);
  sctx.drawImage(pad, minX, minY, w, h, (28 - dw) / 2 + dx, (28 - dh) / 2 + dy, dw, dh);
  const final = sctx.getImageData(0, 0, 28, 28).data;
  const x0 = new Array(784);
  for (let i = 0; i < 784; i++) x0[i] = final[i * 4] >> 1;
  return x0;
}

function drawInput(canvas, x0) {
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = x0[i] * 2;
    image.data[i * 4] = v;
    image.data[i * 4 + 1] = v;
    image.data[i * 4 + 2] = v;
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function renderHidden(values, forged = [], named = null) {
  const grid = $('hidden');
  grid.replaceChildren();
  values.forEach((v, j) => {
    const cell = document.createElement('div');
    cell.textContent = v;
    cell.title = `Hidden neuron ${j}: ${v}`;
    cell.style.backgroundColor = `rgba(var(--cell), ${(v / 127).toFixed(3)})`;
    cell.style.color = v > 72 ? 'var(--accent-ink)' : 'var(--ink)';
    if (forged.includes(j)) cell.classList.add('forged');
    if (named === j) cell.classList.add('named');
    grid.append(cell);
  });
}

function renderLogits(values) {
  const box = $('logits');
  box.replaceChildren();
  const low = Math.min(...values);
  const high = Math.max(...values);
  const top = argmax(values);
  values.forEach((v, j) => {
    const row = document.createElement('div');
    row.className = `logit${j === top ? ' best' : ''}`;
    const width = high === low ? 0 : (((v - low) / (high - low)) * 100).toFixed(1);
    row.innerHTML = `<span>${j}</span><span class="track"><span class="fill" style="display:block;width:${width}%"></span></span><span class="v">${fmt(v)}</span>`;
    box.append(row);
  });
}

async function runNetwork(x0, label) {
  S.x0 = x0;
  S.label = label;
  drawInput($('input'), x0);
  $('run-note').textContent = 'Running the network on the EVM in this tab...';
  await paint();
  const started = performance.now();
  const t = await S.d.traceTx(x0);
  const wall = performance.now() - started;
  S.honest = t;
  S.runGas = t.gas;
  renderHidden(t.hidden);
  renderLogits(t.logits);
  $('answer').textContent = t.cls;
  $('run-note').innerHTML = `<b>${fmt(t.gas)} gas</b>, Ethereum's unit of computing cost, to run the whole network as one transaction. `
    + `It took ${fmt(Math.round(wall))} ms in this tab. On a real chain that fee is paid for every single answer.`;
  $('to-claim').disabled = false;
  logTx('Anyone', 'Network.trace(image)', t);
  S.prepared = null;
  S.target = null;
  await refreshLedger();
}

const runDrawn = action(async () => {
  const x0 = preprocess();
  if (x0) await runNetwork(x0, 'your drawing');
});

$('clear').addEventListener('click', clearPad);

function renderSamples() {
  const strip = $('samples');
  assets.samples.forEach((sample, k) => {
    if (k >= 30) return; // the first 30: three of each digit; the misclassified are shown after them
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const x0 = Array.from(sample.x0.match(/../g), (b) => parseInt(b, 16));
    drawInput(canvas, x0);
    canvas.title = `MNIST test image ${sample.i}, labelled ${sample.label}`;
    canvas.setAttribute('role', 'button');
    canvas.addEventListener('click', action(async () => {
      // Put the image on the pad too, so it is clear what was picked.
      clearPad();
      pctx.imageSmoothingEnabled = false;
      pctx.drawImage(canvas, 0, 0, pad.width, pad.height);
      await runNetwork(x0, `test image ${sample.i}`);
    }));
    strip.append(canvas);
  });
}

// --- 2: the claim ----------------------------------------------------------------------------------
const lieKind = () => document.querySelector('input[name="lie"]:checked').value;

function renderTargets() {
  const box = $('targets');
  box.replaceChildren();
  const kind = lieKind();
  const truth = S.honest ? S.honest.cls : null;
  for (let c = 0; c < 10; c++) {
    const b = document.createElement('button');
    b.textContent = c;
    b.disabled = kind === 'honest' || c === truth;
    if (kind === 'honest' ? c === truth : c === S.target) b.classList.add('on');
    b.addEventListener('click', action(async () => {
      S.target = c;
      await prepare();
    }));
    box.append(b);
  }
}

const prepare = async () => {
  S.prepared = null;
  $('post').disabled = true;
  if (!S.honest) {
    renderTargets();
    return;
  }
  const kind = lieKind();
  if (kind !== 'honest' && (S.target === null || S.target === S.honest.cls)) {
    S.target = (S.honest.cls + 7) % 10;
    if (S.target === S.honest.cls) S.target = (S.target + 1) % 10;
  }
  renderTargets();
  const { d, x0, honest, target } = S;
  let claim;
  if (kind === 'honest') claim = d.honestClaim(x0, honest);
  if (kind === 'class') claim = d.lieAboutClass(x0, honest, target);
  if (kind === 'logit') claim = d.lieAboutLogit(x0, honest, target);
  if (kind === 'hidden') claim = await d.lieAboutHidden(x0, honest, target);
  $('c-true-class').textContent = honest.cls;
  if (!claim) {
    $('c-claim-class').textContent = target;
    $('c-claim-hidden').textContent = 'no forgery found that reaches this answer';
    $('c-claim-logits').textContent = '';
    return;
  }
  S.prepared = { kind, claim };
  const lying = claim.cls !== honest.cls;
  $('c-claim-class').innerHTML = `<span class="${lying ? 'false' : ''}">${claim.cls}</span>`;
  const changedHidden = claim.hidden.flatMap((v, j) => (v !== honest.hidden[j] ? [j] : []));
  const changedLogits = claim.logits.flatMap((v, j) => (v !== honest.logits[j] ? [j] : []));
  $('c-claim-hidden').innerHTML = changedHidden.length
    ? `<span class="false">${changedHidden.length} forged</span> (neurons ${changedHidden.join(', ')})`
    : 'the same 32';
  if (kind === 'hidden') {
    $('c-claim-logits').innerHTML = '<span class="false">all 10 changed</span>, but each is correct for the forged neurons';
  } else if (changedLogits.length) {
    $('c-claim-logits').innerHTML = `<span class="false">output ${changedLogits[0]} raised</span> from ${fmt(honest.logits[changedLogits[0]])} to ${fmt(claim.logits[changedLogits[0]])}`;
  } else {
    $('c-claim-logits').textContent = 'the same 10';
  }
  $('post').disabled = false;
};

$('to-claim').addEventListener('click', action(async () => {
  show('claim');
  await prepare();
}));
document.querySelectorAll('input[name="lie"]').forEach((input) => {
  input.addEventListener('change', action(prepare));
});

$('post').addEventListener('click', action(async () => {
  const { d } = S;
  const { kind, claim } = S.prepared;
  const honest = S.honest;
  $('claim-note').textContent = 'Posting...';
  await paint();
  const tx = await d.postClaim('Claimant', claim);
  logTx('Claimant', 'claim(...) with 1 ETH', tx);
  if (tx.reverted) {
    $('claim-note').textContent = `The contract refused the claim: ${tx.why.name}, ${tx.why.meaning}.`;
    return;
  }
  const id = (await d.claimCount()) - 1n;
  S.current = {
    id, kind, claim, honest, finding: d.audit(claim, honest), postGas: tx.gasUsed, challengeGas: null,
  };
  $('claim-note').textContent = `Claim #${id} is on-chain. It cost ${fmt(tx.gasUsed)} gas, about ${pct(tx.gasUsed, S.runGas)} of running the network.`;
  await refreshLedger();
  renderDispute();
  show('dispute');
}));

const pct = (part, whole) => `${((Number(part) / Number(whole)) * 100).toFixed(1)}%`;

// --- 3: the dispute --------------------------------------------------------------------------------
function describe(finding) {
  if (!finding) {
    return { html: 'The watcher re-ran the network and <b>every value matches</b>. There is nothing to challenge.', tone: 'good' };
  }
  if (finding.kind === 'class') {
    return {
      html: `<b>The answer is wrong.</b> The posted outputs give ${finding.recomputed}, the claim says ${finding.claimed}.`,
      tone: 'bad',
    };
  }
  const what = finding.layer === 1 ? `Hidden neuron ${finding.j}` : `Output ${finding.j}`;
  return {
    html: `<b>${what} is wrong.</b> The claim says ${fmt(finding.claimed)}, re-running it gives ${fmt(finding.recomputed)}.`,
    tone: 'bad',
  };
}

const CHECKS = {
  1: (j) => [
    ['range', `Claim exists, and neuron ${j} is in layer 1`],
    ['InputMismatch', 'The 784 inputs presented hash to the claim\'s commitment'],
    ['OutputMismatch', 'The 32 hidden values presented hash to the commitment'],
    ['NotCommitted', 'The neuron\'s weights verify against the model root'],
    ['rerun', 'Re-run this one neuron: 784 multiply-adds, the same Fe code as the full network'],
    ['settle', 'The claim is open, within its window, and the values differ: slash it'],
  ],
  2: (j) => [
    ['range', `Claim exists, and output ${j} is in layer 2`],
    ['InputMismatch', 'The 32 hidden values presented hash to the commitment'],
    ['OutputMismatch', 'The 10 outputs presented hash to the commitment'],
    ['NotCommitted', 'The neuron\'s weights verify against the model root'],
    ['rerun', 'Re-run this one neuron: 32 multiply-adds, the same Fe code as the full network'],
    ['settle', 'The claim is open, within its window, and the values differ: slash it'],
  ],
  class: () => [
    ['range', 'Claim exists'],
    ['OutputMismatch', 'The 10 outputs presented hash to the commitment'],
    ['rerun', 'Take the largest output, the lowest index winning a tie'],
    ['settle', 'The claim is open, within its window, and the answers differ: slash it'],
  ],
};

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
  const box = $('checks');
  box.replaceChildren();
  list.forEach(([, text]) => {
    const li = document.createElement('li');
    li.textContent = text;
    box.append(li);
  });
}

/** Lights the list up to where the contract stopped, from the error the contract returned. */
async function replayChecks(list, tx) {
  const items = [...$('checks').children];
  const stopAt = tx.reverted ? WHERE[tx.why.name] ?? null : null;
  for (let i = 0; i < list.length; i++) {
    await pause(170);
    if (stopAt === list[i][0]) {
      items[i].classList.add('no');
      items[i].textContent = `${list[i][1]}. Refused: ${tx.why.name}, ${tx.why.meaning}.`;
      for (let k = i + 1; k < list.length; k++) items[k].classList.add('skip');
      return;
    }
    items[i].classList.add('ok');
  }
}

function drawTree(leaf = null) {
  const svg = $('tree');
  const sizes = [42];
  while (sizes[sizes.length - 1] > 1) sizes.push(Math.ceil(sizes[sizes.length - 1] / 2));
  const width = 560;
  const height = 104;
  const levelY = (l) => height - 7 - (l * (height - 14)) / (sizes.length - 1);
  // x of each node: leaves evenly spaced, parents over their children.
  const xs = [Array.from({ length: 42 }, (_, i) => 8 + (i * (width - 16)) / 41)];
  for (let l = 1; l < sizes.length; l++) {
    const below = xs[l - 1];
    xs.push(Array.from({ length: sizes[l] }, (_, i) => {
      const a = below[i * 2];
      const b = below[i * 2 + 1];
      return b === undefined ? a : (a + b) / 2;
    }));
  }
  const parts = [];
  for (let l = 0; l + 1 < sizes.length; l++) {
    for (let i = 0; i < sizes[l]; i++) {
      const parent = i >> 1;
      const onPath = leaf !== null && leaf >> l === i;
      parts.push(`<line class="te${onPath ? ' path' : ''}" x1="${xs[l][i].toFixed(1)}" y1="${levelY(l).toFixed(1)}" x2="${xs[l + 1][parent].toFixed(1)}" y2="${levelY(l + 1).toFixed(1)}"/>`);
    }
  }
  for (let l = 0; l < sizes.length; l++) {
    for (let i = 0; i < sizes[l]; i++) {
      let cls = 'tn';
      if (leaf !== null) {
        const here = leaf >> l;
        if (i === here) cls += ' path';
        else if (i === (here ^ 1)) cls += ' sib';
      }
      if (l === sizes.length - 1) cls += ' root';
      const r = l === sizes.length - 1 ? 6 : l === 0 ? 3 : 3.6;
      parts.push(`<circle class="${cls}" cx="${xs[l][i].toFixed(1)}" cy="${levelY(l).toFixed(1)}" r="${r}"/>`);
    }
  }
  svg.innerHTML = parts.join('');
}

function renderGasBars(hitGas, hitLabel) {
  const box = $('gas-bars');
  const rows = [['Whole network', S.current?.runGas ?? S.runGas, false]];
  if (S.current) rows.push(['Posting the claim', S.current.postGas, false]);
  if (hitGas) rows.push([hitLabel, hitGas, true]);
  const most = Math.max(...rows.map((r) => Number(r[1])));
  box.replaceChildren();
  rows.forEach(([label, gas, hit]) => {
    const row = document.createElement('div');
    row.className = `bar${hit ? ' hit' : ''}`;
    const width = ((Number(gas) / most) * 68).toFixed(2);
    row.innerHTML = `<span>${label}</span><span class="track"><span class="fill" style="display:block;width:${width}%"></span><span class="v">${fmt(gas)} gas</span></span>`;
    box.append(row);
  });
}

function checksFor(finding) {
  if (finding?.kind === 'neuron') return CHECKS[finding.layer](finding.j);
  return CHECKS.class();
}

function renderDispute() {
  const c = S.current;
  if (!c) return;
  c.runGas = c.runGas ?? S.runGas;
  const text = describe(c.finding);
  const box = $('finding');
  box.innerHTML = text.html;
  box.className = `finding ${text.tone}`;
  $('dispute-title').textContent = c.finding
    ? 'Anyone can check the claim. It takes one neuron to prove it false.'
    : 'An honest claim. Try to break it anyway.';
  $('send-named').disabled = !c.finding;
  $('send-named').textContent = c.finding ? 'Send this challenge' : 'Nothing to send';
  $('tries').open = !c.finding;
  $('verdict').className = 'verdict';
  $('verdict').textContent = '';
  $('dispute-note').textContent = '';
  renderChecks(checksFor(c.finding));
  drawTree(leafOf(c.finding));
  renderGasBars(null);
  // Show the claim on the network view too, so going back to step 1 shows what was forged.
  renderHidden(c.claim.hidden, c.claim.forged ?? [], c.finding?.layer === 1 ? c.finding.j : null);
}

const leafOf = (f) => (f?.kind === 'neuron' ? (f.layer === 1 ? f.j : 32 + f.j) : null);

async function sendChallenge(finding, label) {
  const c = S.current;
  const { d } = S;
  const list = checksFor(finding);
  renderChecks(list);
  drawTree(leafOf(finding));
  $('verdict').className = 'verdict';
  $('dispute-note').textContent = 'Mining the challenge...';
  await paint();
  const tx = await d.challenge('Watcher', c.id, c.claim, finding);
  logTx('Watcher', label, tx);
  await replayChecks(list, tx);
  const v = $('verdict');
  if (tx.reverted) {
    if (tx.why.name === 'NoFraud') {
      v.textContent = 'CLAIM STANDS';
      v.className = 'verdict stands show';
    }
    $('dispute-note').innerHTML = `The contract refused: <b>${tx.why.name}</b>, ${tx.why.meaning}. `
      + `The failed challenge cost the watcher ${fmt(tx.gasUsed)} gas and changed nothing.`;
  } else {
    c.challengeGas = tx.gasUsed;
    renderGasBars(tx.gasUsed, 'This challenge');
    v.textContent = 'SLASHED';
    v.className = 'verdict slashed show';
    $('dispute-note').innerHTML = `<b>${fmt(tx.gasUsed)} gas settled it: ${pct(tx.gasUsed, c.runGas)} of the ${fmt(c.runGas)} a full run costs.</b> `
      + 'The whole 1 ETH bond is now the watcher\'s to withdraw.';
  }
  await refreshLedger();
  renderSettle();
}

$('send-named').addEventListener('click', action(async () => {
  const f = S.current.finding;
  const label = f.kind === 'class' ? 'challengeClass(...)' : `challengeNeuron${f.layer}(j=${f.j})`;
  await sendChallenge(f, label);
}));
$('try-class').addEventListener('click', action(async () => {
  if (S.current) await sendChallenge({ kind: 'class' }, 'challengeClass(...)');
}));
$('try-l2').addEventListener('click', action(async () => {
  const j = Number($('pick-l2').value);
  if (S.current) await sendChallenge({ kind: 'neuron', layer: 2, j }, `challengeNeuron2(j=${j})`);
}));
$('try-l1').addEventListener('click', action(async () => {
  const j = Number($('pick-l1').value);
  if (S.current) await sendChallenge({ kind: 'neuron', layer: 1, j }, `challengeNeuron1(j=${j})`);
}));
$('to-settle').addEventListener('click', () => {
  renderSettle();
  show('settle');
});

// --- 4: settlement ---------------------------------------------------------------------------------
function settleSay(html, tone = '') {
  const box = $('settle-out');
  box.innerHTML = html;
  box.className = `finding small ${tone}`;
}

function renderSettle() {
  const c = S.current;
  if (!c) return;
  const name = STATUS[c.status] ?? 'open';
  if (name === 'slashed') {
    settleSay(`Claim #${c.id} is <b class="st-slashed">slashed</b>. It can never become a result, and the watcher has the bond.`, 'bad');
  } else if (name === 'final') {
    settleSay(`Claim #${c.id} is <b class="st-final">final</b>.`, 'good');
  } else {
    settleSay(`Claim #${c.id} is <b class="st-open">open</b>. Nothing reads it as true yet.`);
  }
}

$('forward').addEventListener('click', action(async () => {
  S.d.chain.advance(WINDOW);
  await refreshLedger();
  settleSay('The clock moved one hour. Blocks in this tab use this clock, as a block timestamp would.');
}));
$('finalize').addEventListener('click', action(async () => {
  const c = S.current;
  if (!c) return;
  const tx = await S.d.finalize('Claimant', c.id);
  logTx('Claimant', `finalize(${c.id})`, tx);
  await refreshLedger();
  if (tx.reverted) {
    settleSay(`The contract refused: <b>${tx.why.name}</b>, ${tx.why.meaning}.`, 'bad');
    return;
  }
  const lie = c.finding !== null;
  settleSay(
    `Claim #${c.id} is <b class="st-final">final</b> and its bond is back with the claimant. ${
      lie ? '<b>This claim was false and nobody challenged it in time, so it now stands.</b> That is the assumption the design rests on: at least one honest watcher.' : ''}`,
    lie ? 'bad' : 'good',
  );
}));
$('read-result').addEventListener('click', action(async () => {
  const c = S.current;
  if (!c) return;
  const r = await S.d.result(c.id);
  if (r.why) settleSay(`<code>result(${c.id})</code> reverts with <b>${r.why.name}</b>: ${r.why.meaning}.`, 'bad');
  else settleSay(`<code>result(${c.id})</code> returns <b>${r.cls}</b>.`, 'good');
}));
const withdraw = (name, id) => $(id).addEventListener('click', action(async () => {
  const before = await S.d.creditOf(name);
  const tx = await S.d.withdraw(name);
  logTx(name, 'withdraw()', tx);
  await refreshLedger();
  settleSay(tx.reverted
    ? `The contract refused: <b>${tx.why.name}</b>, ${tx.why.meaning}.`
    : `${name} pulled ${eth(before)}. This is the only handler in the contract that can send ETH.`, tx.reverted ? 'bad' : 'good');
}));
withdraw('Claimant', 'w-claimant');
withdraw('Watcher', 'w-watcher');
$('again').addEventListener('click', action(async () => {
  show('claim');
  await prepare();
}));

// --- 5: Fe ------------------------------------------------------------------------------------------
function esc(text) {
  return text.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

function renderFe() {
  const tabs = $('fe-tabs');
  const pick = (entry, button) => {
    tabs.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === button));
    $('fe-diag').textContent = `$ fe check   # ${entry.file}\n\n${entry.diagnostic}`;
  };
  assets.compiler_errors.forEach((entry, k) => {
    const b = document.createElement('button');
    b.textContent = entry.title;
    b.addEventListener('click', () => pick(entry, b));
    tabs.append(b);
    if (k === 0) pick(entry, b);
  });
  const e = assets.excerpts;
  const mark = (code) => esc(code).replace(/ \{$/, '').replace(/mut call/g, '<span class="hl">mut call</span>');
  $('fe-effects').innerHTML = [
    `// ${e.effects_claim.file}`,
    mark(e.effects_claim.code),
    '',
    mark(e.effects_finalize.code),
    '',
    mark(e.effects_withdraw.code),
  ].join('\n');
  $('fe-neuron').textContent = `// ${e.neuron.file}\n${e.neuron.code}`;
  const N = assets.numbers;
  const up = (a, b) => `${(((a - b) / b) * 100).toFixed(1)}% more`;
  $('fe-costs').innerHTML = `<tr><th>Gas</th><th class="n">As built</th><th class="n">Baseline</th><th class="n">Cost</th></tr>
    <tr><td>Typed vectors and scales, against plain words: one neuron's execution</td><td class="n">${fmt(N.typed_exec)}</td><td class="n">${fmt(N.raw_exec)}</td><td class="n">${up(N.typed_exec, N.raw_exec)}</td></tr>
    <tr><td>Pure state machine, against inlined checks: <code>finalize</code></td><td class="n">${fmt(N.step_finalize)}</td><td class="n">${fmt(N.inline_finalize)}</td><td class="n">${up(N.step_finalize, N.inline_finalize)}</td></tr>
    <tr><td>Fe, against a hand-written Solidity twin: <code>claim</code></td><td class="n">${fmt(N.calls.claim.fe)}</td><td class="n">${fmt(N.calls.claim.sol)}</td><td class="n">${up(N.calls.claim.fe, N.calls.claim.sol)}</td></tr>`;
  $('fe-build').textContent = `Built with ${assets.fe_version}. Arbiter ${fmt(assets.contracts.Arbiter.runtime_bytes)} bytes, `
    + `bytecode sha256 ${assets.contracts.Arbiter.sha256.slice(0, 16)}... Model accuracy ${(assets.model.accuracy.integer * 100).toFixed(2)}% on the 10,000 MNIST test images.`;
}

// --- start --------------------------------------------------------------------------------------------
async function boot() {
  renderSamples();
  renderFe();
  drawTree();
  renderChecks(CHECKS[1](0));
  await paint();
  const started = performance.now();
  S.d = await Deployment.create(assets);
  const ms = performance.now() - started;
  const root = await S.d.modelRoot();
  if (root !== assets.model.root) throw new Error('the deployed root differs from the exported one');
  $('l-engine').textContent = `EVM (Osaka), 5 Fe contracts deployed in ${fmt(Math.round(ms))} ms. No wallet, no network, play money.`;
  await refreshLedger();
  window.tensileDemo = { S, BOND, ready: true };
}
boot().catch((e) => {
  console.error(e);
  $('l-engine').textContent = `The EVM failed to start: ${e.message}`;
});
