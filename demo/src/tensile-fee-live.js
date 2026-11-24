// The live status page of the fee-tier system on Sepolia (tensile-fee-live-core.js does the work).
// It shows what the chains and this browser's own run say; it decides nothing.
import {
  FEES, FeeSystem, GAP, HORIZON, Mainnet, N_OBS, Node, STATUS, STEP, TIER_NAMES,
  checkGrid, describeFinding, inputHash, movesOf, to0x, windowOf, windowsByHash,
} from './tensile-fee-live-core.js';

const data = window.TENSILE_FEE;
const $ = (id) => document.getElementById(id);
const SCAN = 'https://sepolia.etherscan.io';
const MAINSCAN = 'https://etherscan.io';
const CHAIN_ID = 11155111;
const short = (text) => `${text.slice(0, 8)}...${text.slice(-6)}`;
const state = { rows: new Map(), selected: null, system: null };

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function link(href, text) {
  const a = el('a', text);
  a.href = href;
  a.target = '_blank';
  a.rel = 'noopener';
  return a;
}
function td(content, cls) {
  const cell = el('td', null, cls);
  if (typeof content === 'object' && content !== null) cell.append(content);
  else cell.textContent = content;
  return cell;
}
const utc = (seconds) => new Date(seconds * 1000).toISOString().slice(0, 16).replace('T', ' ');

function role(address) {
  const a = address.toLowerCase();
  if (a === data.config.keeper.toLowerCase()) return 'keeper';
  if (data.config.adversaries.map((x) => x.toLowerCase()).includes(a)) return 'adversary (planted)';
  return short(address);
}

function drawMoves(moves) {
  const svg = $('moves');
  const ns = 'http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const w = 576 / moves.length;
  moves.forEach((m, i) => {
    const r = document.createElementNS(ns, 'rect');
    const h = Math.max(1, (m / 127) * 116);
    r.setAttribute('x', (i * w + 0.5).toFixed(2));
    r.setAttribute('width', Math.max(1, w - 1).toFixed(2));
    r.setAttribute('y', (120 - h).toFixed(2));
    r.setAttribute('height', h.toFixed(2));
    r.setAttribute('fill', 'var(--bar)');
    const t = document.createElementNS(ns, 'title');
    t.textContent = `move ${i + 1} of 144: ${m} ticks`;
    r.append(t);
    svg.append(r);
  });
}

function show(id) {
  const row = state.rows.get(id);
  if (!row) return;
  state.selected = id;
  for (const li of $('claims').children) li.classList.toggle('on', Number(li.dataset.id) === id);
  const { claim, honest, finding, status, answer, challenge, end } = row;
  $('d-title').textContent = `Claim ${id}, ${STATUS[status]}`;
  $('d-body').hidden = false;
  $('d-meta').replaceChildren(
    `Posted by ${role(claim.claimant)} in `, link(`${SCAN}/tx/${claim.hash}`, short(claim.hash)),
    end === undefined ? ', about an input that is not one of the oracle\'s windows.' : `, about the window ending at move ${end} (mainnet block ${data.firstBlock + STEP * end}). `,
    `Window closes ${utc(claim.deadline)} UTC.`,
  );
  const cells = (target, claimed, mine, name) => target.replaceChildren(...claimed.map((v, j) => {
    const cell = el('div', v, v !== mine[j] ? 'bad' : '');
    cell.title = `${name} ${j}: claimed ${v}, this browser computed ${mine[j]}`;
    return cell;
  }));
  cells($('d-h1'), claim.hidden[0], honest.hidden[0], 'layer 1 neuron');
  cells($('d-h2'), claim.hidden[1], honest.hidden[1], 'layer 2 neuron');
  cells($('d-logits'), claim.logits, honest.logits, 'logit');
  $('d-logits').append(el('div', `tier ${claim.cls}`, claim.cls !== honest.cls || finding?.kind === 'class' ? 'bad' : ''));
  const verdict = $('d-verdict');
  verdict.className = `verdict ${finding ? 'false' : 'true'}`;
  const head = el('b');
  const body = el('p');
  body.style.margin = '6px 0 0';
  if (finding === null) {
    head.textContent = `This claim is what the network computes: tier ${honest.cls}, ${TIER_NAMES[honest.cls]}, fee ${FEES[honest.cls] / 10000}%.`;
    body.textContent = status === 1
      ? (answer?.accepted ? 'Unexpected: the node says a challenge would be accepted.' : `Asked anyway: a challenge is refused by the arbiter with ${answer?.why?.name} (${answer?.why?.meaning}).`)
      : status === 3 ? 'Its window passed unchallenged; it is final.' : 'Unexpected: it was slashed.';
  } else {
    head.textContent = 'This claim is false.';
    const what = `${describeFinding(finding)}.`;
    body.textContent = status === 2
      ? `${what} It has been slashed: the arbiter re-ran that one ${finding.kind === 'class' ? 'argmax' : 'neuron'} and took the bond.`
      : status === 3 ? `${what} It became final unchallenged: the watcher failed.`
        : answer?.accepted ? `${what} Asked as a call, the arbiter accepts the challenge below.` : `${what} The arbiter refuses the challenge: ${answer?.why?.name}.`;
  }
  verdict.replaceChildren(head, body);
  $('d-data').textContent = to0x(challenge);
  const can = finding !== null && status === 1 && answer?.accepted;
  $('d-send').disabled = !can || !window.ethereum;
  $('d-sendnote').textContent = can
    ? (window.ethereum ? 'Needs a wallet on Sepolia with a little test ETH for gas. The page never asks for a key.' : 'No browser wallet here; `cast send <arbiter> <calldata>` with any Sepolia key does the same.')
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
    const hash = await window.ethereum.request({ method: 'eth_sendTransaction', params: [{ from, to: data.config.arbiter, data: to0x(row.challenge) }] });
    const a = $('d-tx');
    a.href = `${SCAN}/tx/${hash}`;
    a.textContent = `Sent: ${short(hash)}`;
    note.textContent = 'When it is mined the claim is slashed. Reload to see it.';
  } catch (error) {
    note.textContent = `Not sent: ${error.message ?? error}`;
  }
}

async function checkRelay(relayed, safe) {
  const mainnet = new Mainnet(data.mainnetRpc, { agree: 2 });
  const rows = new Map();
  const body = $('ticks');
  const recent = relayed.slice(-48).reverse();
  for (const r of recent) {
    const tr = el('tr');
    tr.append(
      td(String(r.number)), td(link(`${MAINSCAN}/block/${r.block}`, r.block.toLocaleString('en'))), td('', 'wait'),
      td(String(r.tick)), td('reading', 'wait'), td('', 'wait'), td(link(`${SCAN}/tx/${r.hash}`, r.sepoliaBlock.toLocaleString('en'))),
    );
    body.append(tr);
    rows.set(r.number, tr);
  }
  const checkable = relayed.filter((r) => r.block <= safe);
  const order = checkable.slice().reverse();
  let equal = 0;
  let wrong = 0;
  let done = 0;
  for (let i = 0; i < order.length; i += 50) {
    const part = order.slice(i, i + 50);
    try {
      const got = await mainnet.ticksAt(part.map((r) => r.block));
      got.forEach((m, k) => {
        const r = part[k];
        const same = m.tick === r.tick;
        if (same) equal++;
        else wrong++;
        done++;
        const tr = rows.get(r.number);
        if (tr) {
          tr.children[2].textContent = utc(m.timestamp);
          tr.children[2].className = '';
          tr.children[4].textContent = String(m.tick);
          tr.children[4].className = '';
          tr.children[5].textContent = same ? 'equal' : 'DIFFERENT';
          tr.children[5].className = same ? 'ok' : 'no';
        }
      });
    } catch (error) {
      $('relay-note').textContent = `Mainnet nodes did not answer from this browser (${error.message}); the scheduled watcher does the same check.`;
      break;
    }
    $('s-relay').textContent = wrong ? `${wrong} DIFFERENT of ${done}` : `${equal} of ${checkable.length} equal`;
    $('s-relay').className = wrong ? 'no' : done === checkable.length ? 'ok' : '';
  }
  if (done) {
    $('relay-note').textContent = `${relayed.length} ticks relayed, of mainnet blocks ${relayed[0].block.toLocaleString('en')} to ${relayed.at(-1).block.toLocaleString('en')} (every 50th block). This browser read the pool's slot0() at each of those blocks from two public mainnet nodes that had to agree: ${equal} of ${done} equal${wrong ? `, ${wrong} DIFFERENT` : ''}. The newest 48 are below.`;
  }
}

async function main() {
  $('d-send').addEventListener('click', send);
  const node = new Node(data.sepoliaRpc);
  const system = new FeeSystem(node, data.config, data.model);
  state.system = system;
  const head = await node.value('eth_getBlockByNumber', ['latest', false]);
  const latest = parseInt(head.number, 16);
  const now = parseInt(head.timestamp, 16);
  const [relayState, oracleState, arbiterState, relayed, claims, slashes, finalized, feeSets] = await Promise.all([
    system.relayState(), system.oracleState(), system.arbiterState(), system.relayedTicks(latest),
    system.claims(latest), system.slashes(latest), system.finalized(latest), system.feeSets(latest),
  ]);
  data.firstBlock = relayState.firstBlock;
  const moves = movesOf(relayed.map((r) => r.tick));
  const grid = checkGrid(relayed, relayState.firstBlock);
  const last = feeSets.reduce((a, f) => (!a || f.end > a.end ? f : a), null);
  const tier = FEES.indexOf(oracleState.fee);
  $('s-fee').textContent = `${oracleState.fee / 10000}% (${last && oracleState.moves < last.end + GAP + HORIZON ? `tier ${tier}, ${TIER_NAMES[tier]}` : 'the default'})`;
  $('oracle-note').textContent = `${oracleState.moves} moves stored${grid.length ? `; the relay's events break its own grid: ${grid[0]}` : ', one per relayed tick, each 50 mainnet blocks apart'}. `
    + (last ? `The fee comes from claim ${last.id}, about the window ending at move ${last.end}; it holds until move ${last.end + GAP + HORIZON}.` : 'No claim has been applied yet: the fee is the default, 0.30%.');
  if (moves.length >= N_OBS) drawMoves(windowOf(moves, moves.length));
  const fees = $('fees');
  for (const f of feeSets.slice().reverse().slice(0, 12)) {
    const tr = el('tr');
    tr.append(td(String(f.id)), td(String(f.end)), td(`${f.tier}, ${TIER_NAMES[f.tier]}`), td(`${f.fee / 10000}%`), td(link(`${SCAN}/tx/${f.hash}`, short(f.hash))));
    fees.append(tr);
  }
  if (!feeSets.length) {
    const tr = el('tr');
    tr.append(td('none yet'));
    fees.append(tr);
  }

  // The relay against mainnet runs alongside the claims.
  const mainnetHead = new Mainnet(data.mainnetRpc, { agree: 1 });
  mainnetHead.safeHead().then((safe) => {
    const lag = safe - relayState.latestBlock;
    $('s-lag').textContent = relayState.relayed ? `${lag} blocks, about ${Math.round((lag * 12) / 60)} min` : 'not started';
    return checkRelay(relayed, safe);
  }).catch((error) => {
    $('s-lag').textContent = 'mainnet unreachable';
    $('relay-note').textContent = `Mainnet nodes did not answer from this browser: ${error.message}`;
  });

  const windows = windowsByHash(moves);
  const counts = { 1: 0, 2: 0, 3: 0 };
  const list = $('claims');
  const ids = [...claims.keys()].sort((a, b) => b - a);
  for (const id of ids) {
    const li = el('li');
    li.dataset.id = id;
    li.append(el('b', `#${id}`), el('i', 'reading'), el('span', 'checking', 'tag wait'));
    li.addEventListener('click', () => show(id));
    list.append(li);
  }
  for (const id of ids) {
    const claim = claims.get(id);
    const li = list.querySelector(`li[data-id="${id}"]`);
    const { honest, finding } = system.audit(claim);
    const status = slashes[id] ? 2 : finalized.has(id) ? 3 : (await system.info(id)).status;
    counts[status]++;
    const challenge = system.challengeData(id, claim, finding);
    const answer = status === 1 ? await system.ask(challenge) : null;
    const end = windows.get(inputHash(claim.x0));
    state.rows.set(id, { claim, honest, finding, status, answer, challenge, end });
    li.children[1].textContent = `tier ${claim.cls}, ${STATUS[status]}, ${end === undefined ? 'not a window' : `window ${end}`}, by ${role(claim.claimant)}`;
    const tag = li.children[2];
    if (finding === null) {
      tag.textContent = 'honest';
      tag.className = 'tag true';
    } else {
      tag.textContent = status === 2 ? 'false, slashed' : status === 1 ? (answer?.accepted ? 'false, can be slashed' : 'false') : 'false, FINAL';
      tag.className = 'tag false';
    }
    if (state.selected === null) show(id);
    $('s-claims').textContent = `${claims.size}: ${counts[1]} open, ${counts[3]} final, ${counts[2]} slashed`;
  }
  if (!claims.size) $('s-claims').textContent = `none yet (${moves.length} of ${N_OBS} moves)`;
  const open = [...state.rows.entries()].find(([, r]) => r.finding && r.status === 1);
  if (open) show(open[0]);
  void now;
  void arbiterState;
}

main().catch((error) => {
  $('error').textContent = `Sepolia could not be read from this browser (${error.message}).`;
});
