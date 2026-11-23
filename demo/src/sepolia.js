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

async function main() {
  $('s-arbiter').href = `${SCAN}/address/${data.config.arbiter}`;
  $('s-arbiter').textContent = short(data.config.arbiter);
  $('d-send').addEventListener('click', send);
  const wasm = await loadWasm(fromBase64(data.wasm.base64));
  const node = new Node(data.config.rpc);
  const watcher = new Watcher(node, data.config, data.model, (x0) => wasm.run(x0));
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
