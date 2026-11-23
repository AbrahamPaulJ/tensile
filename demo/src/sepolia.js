// The page around the Sepolia watcher (sepolia-core.js). It shows what the chain and this browser's
// own run say; it gates nothing and decides nothing.
import { hex } from './abi.js';
import { CHAIN_ID, Node, STATUS, Watcher, to0x } from './sepolia-core.js';
import { fromBase64, loadWasm } from './wasm.js';

const data = window.TENSILE;
const $ = (id) => document.getElementById(id);
const SCAN = 'https://sepolia.etherscan.io';
const eth = (wei) => `${Number(wei) / 1e18} ETH`;
const short = (text) => `${text.slice(0, 8)}...${text.slice(-6)}`;
const state = { rows: new Map(), selected: null };

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function describe(finding) {
  if (finding === null) return 'nothing wrong';
  if (finding.kind === 'class') return 'wrong answer';
  return finding.layer === 1 ? `hidden neuron ${finding.j}` : `output ${finding.j}`;
}

function show(id) {
  const row = state.rows.get(id);
  if (!row?.claim) return;
  state.selected = id;
  for (const li of $('claims').children) li.classList.toggle('on', Number(li.dataset.id) === id);
  const { claim, honest, finding, answer, info, onChain, challenge } = row;
  $('d-title').textContent = `Claim ${id}, ${STATUS[info.status]}`;
  $('d-body').hidden = false;
  const ctx = $('d-input').getContext('2d');
  const image = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.min(255, claim.x0[i] * 2);
    image.data.set([v, v, v, 255], i * 4);
  }
  ctx.putImageData(image, 0, 0);
  $('d-claimed').textContent = claim.cls;
  $('d-wasm').textContent = honest.cls;
  $('d-chain').textContent = onChain ?? '?';
  $('d-hidden').replaceChildren(...claim.hidden.map((v, j) => {
    const cell = el('div', v, v !== honest.hidden[j] ? 'bad' : '');
    cell.title = `hidden neuron ${j}: claimed ${v}, this browser computed ${honest.hidden[j]}`;
    return cell;
  }));
  $('d-logits').replaceChildren(...claim.logits.map((v, j) => {
    const cell = el('div', v, v !== honest.logits[j] ? 'bad' : '');
    cell.title = `output ${j}: claimed ${v}, this browser computed ${honest.logits[j]}`;
    return cell;
  }));
  const verdict = $('d-verdict');
  verdict.className = `verdict ${finding ? 'false' : 'true'}`;
  const head = el('b');
  const body = el('p');
  body.style.margin = '6px 0 0';
  if (finding === null) {
    head.textContent = 'This claim is what the network computes.';
    body.textContent = answer.accepted
      ? 'Unexpected: the node says a challenge would be accepted.'
      : `Asked anyway: a challenge of its answer is refused by the contract with ${answer.why.name} (${answer.why.meaning}).`;
  } else {
    const what = finding.kind === 'class'
      ? `The outputs it committed to make the answer ${finding.recomputed}, and it claims ${finding.claimed}.`
      : `${finding.layer === 1 ? 'Hidden neuron' : 'Output'} ${finding.j} is claimed as ${finding.claimed}. This browser computed ${finding.recomputed}.`;
    head.textContent = 'This claim is false.';
    body.textContent = info.status === 2
      ? `${what} Someone has already sent the challenge: the claim is slashed.`
      : answer.accepted
        ? `${what} Asked as a call, the arbiter on Sepolia accepts the challenge below: it would re-run that one ${finding.kind === 'class' ? 'argmax' : 'neuron'}, slash the claim and credit the sender ${eth(info.bond / 2n)}.`
        : `${what} The arbiter refuses the challenge: ${answer.why.name} (${answer.why.meaning}).`;
  }
  verdict.replaceChildren(head, body);
  $('d-data').textContent = to0x(challenge);
  $('d-cast').textContent = `cast send --rpc-url https://ethereum-sepolia-rpc.publicnode.com --private-key <yours> ${data.config.arbiter} ${to0x(challenge)}`;
  const can = finding !== null && info.status === 1 && answer.accepted;
  $('d-send').disabled = !can || !window.ethereum;
  $('d-sendnote').textContent = !window.ethereum
    ? 'No browser wallet found here. The command above does the same with any Sepolia key.'
    : can ? 'Needs a wallet on Sepolia with a little test ETH for gas. The page never asks for a key.'
      : 'There is nothing to send for this claim.';
  $('d-tx').textContent = '';
}

async function send() {
  const row = state.rows.get(state.selected);
  const note = $('d-sendnote');
  try {
    const [from] = await window.ethereum.request({ method: 'eth_requestAccounts' });
    await window.ethereum.request({ method: 'wallet_switchEthereumChain', params: [{ chainId: '0x' + CHAIN_ID.toString(16) }] });
    note.textContent = 'Confirm the transaction in the wallet.';
    const hash = await window.ethereum.request({
      method: 'eth_sendTransaction',
      params: [{ from, to: data.config.arbiter, data: to0x(row.challenge) }],
    });
    const link = $('d-tx');
    link.href = `${SCAN}/tx/${hash}`;
    link.textContent = `Sent: ${short(hash)}`;
    note.textContent = 'When it is mined the claim is slashed and withdraw() pays the credit. Reload to see the new status.';
  } catch (error) {
    note.textContent = `Not sent: ${error.message ?? error}`;
  }
}

function drawX0(canvas, x0) {
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.min(255, x0[i] * 2);
    image.data.set([v, v, v, 255], i * 4);
  }
  ctx.putImageData(image, 0, 0);
}

/**
 * The drawing, processed as MNIST images were (the same steps as the offline demo): the ink's bounding
 * box scaled to fit 20 x 20 and centred on its centre of mass in 28 x 28. Returns x0: pixel >> 1.
 */
function preprocess(pad) {
  const pctx = pad.getContext('2d');
  const { data: px } = pctx.getImageData(0, 0, pad.width, pad.height);
  let [minX, minY, maxX, maxY] = [pad.width, pad.height, -1, -1];
  for (let y = 0; y < pad.height; y++) {
    for (let x = 0; x < pad.width; x++) {
      if (px[(y * pad.width + x) * 4] > 20) {
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
  let [mass, mx, my] = [0, 0, 0];
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

function setUpPad(watcher, wasm) {
  const pad = $('pad');
  const pctx = pad.getContext('2d', { willReadFrequently: true });
  pctx.lineCap = 'round';
  pctx.lineJoin = 'round';
  const clear = () => {
    pctx.fillStyle = '#000';
    pctx.fillRect(0, 0, pad.width, pad.height);
    $('pad-wasm').textContent = '?';
    $('pad-chain').textContent = '?';
  };
  clear();
  $('pad-clear').addEventListener('click', clear);
  let drawing = false;
  let last = null;
  let pending = null;
  let turn = 0;
  const point = (e) => {
    const r = pad.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * pad.width, y: ((e.clientY - r.top) / r.height) * pad.height };
  };
  const run = async () => {
    const x0 = preprocess(pad);
    if (!x0) return;
    const mine = ++turn;
    drawX0($('pad-input'), x0);
    const local = wasm.run(Uint8Array.from(x0));
    $('pad-wasm').textContent = local.cls;
    $('pad-chain').textContent = '...';
    const started = performance.now();
    try {
      const cls = await watcher.classifyOnChain(x0);
      if (mine !== turn) return;
      $('pad-chain').textContent = cls;
      $('pad-note').textContent = cls === local.cls
        ? `Both say ${cls}: the WebAssembly build in this tab (${local.ms.toFixed(2)} ms) and the EVM bytecode on Sepolia (${Math.round(performance.now() - started)} ms through a public node), from the same Fe source. Nothing was sent as a transaction.`
        : `They disagree (${local.cls} here, ${cls} on Sepolia). That should never happen: please report it.`;
    } catch (error) {
      if (mine === turn) $('pad-chain').textContent = '?';
      $('pad-note').textContent = `Sepolia did not answer: ${error.message}`;
    }
  };
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
  const stop = () => {
    if (!drawing) return;
    drawing = false;
    clearTimeout(pending);
    pending = setTimeout(run, 250);
  };
  pad.addEventListener('pointerup', stop);
  pad.addEventListener('pointercancel', stop);
}

async function main() {
  $('s-arbiter').href = `${SCAN}/address/${data.config.arbiter}`;
  $('s-arbiter').textContent = short(data.config.arbiter);
  $('d-send').addEventListener('click', send);
  const wasm = await loadWasm(fromBase64(data.wasm.base64));
  const node = new Node(data.config.rpc);
  const watcher = new Watcher(node, data.config, data.model, (x0) => wasm.run(x0));
  setUpPad(watcher, wasm);
  const summary = await watcher.summary();
  $('s-count').textContent = `${summary.count}, root ${summary.root === data.model.root ? 'matches the model' : 'DIFFERS'}`;
  $('s-bond').textContent = eth(summary.bond);
  $('s-node').textContent = new URL(node.used).host;
  const list = $('claims');
  for (let id = 0; id < summary.count; id++) {
    const li = el('li');
    li.dataset.id = id;
    li.append(el('b', `#${id}`), el('i', 'reading'), el('span', 'checking', 'tag wait'));
    li.addEventListener('click', () => show(id));
    list.append(li);
  }
  for (let id = 0; id < summary.count; id++) {
    const li = list.children[id];
    try {
      const [info, claim] = [await watcher.info(id), await watcher.claim(id)];
      const { honest, finding } = watcher.audit(claim);
      const challenge = watcher.challengeData(id, claim, finding);
      const answer = await watcher.ask(challenge);
      const row = { info, claim, honest, finding, challenge, answer, onChain: null };
      state.rows.set(id, row);
      li.children[1].textContent = `says ${claim.cls}, ${STATUS[info.status]}: ${describe(finding)}`;
      const tag = li.children[2];
      if (finding === null) {
        tag.textContent = answer.accepted ? 'unexpected' : 'honest';
        tag.className = 'tag true';
      } else {
        tag.textContent = info.status === 2 ? 'slashed' : answer.accepted ? 'can be slashed' : 'refused';
        tag.className = 'tag false';
      }
      if (state.selected === null && finding !== null && info.status === 1) show(id);
      watcher.classifyOnChain(claim.x0).then((cls) => {
        row.onChain = cls;
        if (state.selected === id) $('d-chain').textContent = cls;
      }).catch(() => {});
    } catch (error) {
      li.children[1].textContent = `could not be read: ${error.message}`;
    }
  }
  if (state.selected === null && state.rows.size) show(state.rows.keys().next().value);
}

main().catch((error) => {
  $('error').textContent = `Sepolia could not be reached from this browser (${error.message}). The offline demo needs no network.`;
  $('s-count').textContent = 'unavailable';
});
// `hex` is kept in the bundle for the console: hex(bytes) of anything above.
window.tensileHex = hex;
