// The live fee-tier system on Sepolia, with no page around it: read the relay, the oracle and the
// arbiter over JSON-RPC, read the real pool on mainnet, re-run the 144-32-16-3 network on a claim's
// committed input, and build the calldata of a claim or a challenge. Used by the status page
// (tensile-fee-live.js), the keeper (tools/fee_keeper.mjs) and the watcher (demo/test/tensile-fee-live.mjs).
//
// Nothing here decides anything about a claim: a challenge is put to the deployed arbiter, and the
// arbiter's answer is the verdict.
//
// The byte and ABI helpers below are a copy of the few that this file needs from demo/src/abi.js
// (copied, not imported, so that this module does not depend on the shared demo modules).
import { keccak_256 } from '@noble/hashes/sha3.js';

const encoder = new TextEncoder();

export const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
export const to0x = (bytes) => '0x' + hex(bytes);

export function fromHex(text) {
  const clean = text.startsWith('0x') ? text.slice(2) : text;
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(clean.substr(i * 2, 2), 16);
  return out;
}

export const keccak = (bytes) => keccak_256(bytes);
export const selector = (signature) => keccak_256(encoder.encode(signature)).slice(0, 4);
export const topic = (signature) => to0x(keccak_256(encoder.encode(signature)));

export function concat(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** A 32-byte word: an integer, two's complement if negative. */
export function word(value) {
  let v = BigInt(value);
  if (v < 0n) v += 1n << 256n;
  return fromHex(v.toString(16).padStart(64, '0'));
}

export const readUint = (bytes, at = 0) => BigInt('0x' + (hex(bytes.subarray(at, at + 32)) || '0'));
export function readInt(bytes, at = 0) {
  const v = readUint(bytes, at);
  return v >= 1n << 255n ? v - (1n << 256n) : v;
}

/** Int8 activations as packed words (docs/spec.md section 7): zero padding to a multiple of 32. */
export function pack(values) {
  const out = new Uint8Array(Math.ceil(values.length / 32) * 32);
  for (let i = 0; i < values.length; i++) out[i] = values[i] & 0xff;
  return out;
}

/** selector || static head || one trailing bytes32[] if given. */
export function encodeCall(signature, parts, dynamicWords = null) {
  const head = concat(parts);
  if (dynamicWords === null) return concat([selector(signature), head]);
  return concat([selector(signature), head, word(head.length + 32), word(dynamicWords.length), ...dynamicWords.map((w) => fromHex(w))]);
}

// ---------------------------------------------------------------------------------------------
// The model: the integer arithmetic of docs/spec.md section 3, and its commitment (tools/gen_tree.py)
// ---------------------------------------------------------------------------------------------

/** Mainnet blocks between two observations, and the clip of a move: examples/amm-fee/dataset.py. */
export const STEP = 50;
export const CLIP = 127;
export const N_OBS = 144;
export const GAP = 6;
export const HORIZON = 24;
export const FEES = [500, 3000, 10000];
export const DEFAULT_FEE = 3000;
export const TIER_NAMES = ['quiet', 'middle', 'active'];

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * The forward pass. Hidden layers: clamp((b + w.x) >> shift, 0, 127), with an arithmetic (floor)
 * shift; the last layer returns the accumulators. Every accumulator of this model fits in int32
 * (model.json records the worst case), so JavaScript numbers are exact here.
 */
export function forward(model, x0) {
  let x = Array.from(x0);
  const hidden = [];
  const layers = model.layers;
  for (let l = 0; l < layers.length; l++) {
    const { weights, bias, shift } = layers[l];
    const out = new Array(weights.length);
    for (let j = 0; j < weights.length; j++) {
      const row = weights[j];
      let acc = bias[j];
      for (let i = 0; i < row.length; i++) acc += row[i] * x[i];
      if (l < layers.length - 1) {
        const shifted = Math.floor(acc / 2 ** shift);
        out[j] = shifted < 0 ? 0 : shifted > CLIP ? CLIP : shifted;
      } else {
        out[j] = acc;
      }
    }
    if (l < layers.length - 1) hidden.push(out);
    x = out;
  }
  return { hidden, logits: x, cls: argmax(x) };
}

const TAG = encoder.encode('tensile.neuron.v0');

/** keccak256(TAG || u256(layer) || u256(neuron) || i256(bias) || packed row): one leaf per neuron. */
export function leafHash(layer, neuron, bias, row) {
  return keccak(concat([TAG, word(layer), word(neuron), word(bias), pack(row)]));
}

/** The sorted-pair Merkle tree of tools/gen_tree.py: { root, proofs: { 'l:j': [0x..] } }. */
export function commitment(model) {
  const positions = [];
  const leaves = [];
  model.layers.forEach((layer, index) => {
    layer.weights.forEach((row, j) => {
      positions.push(`${index + 1}:${j}`);
      leaves.push(leafHash(index + 1, j, layer.bias[j], row));
    });
  });
  const proofs = leaves.map(() => []);
  let level = leaves;
  let members = leaves.map((_, i) => [i]);
  while (level.length > 1) {
    const nextLevel = [];
    const nextMembers = [];
    for (let k = 0; k < level.length; k += 2) {
      if (k + 1 === level.length) {
        nextLevel.push(level[k]);
        nextMembers.push(members[k]);
        continue;
      }
      for (const i of members[k]) proofs[i].push(level[k + 1]);
      for (const i of members[k + 1]) proofs[i].push(level[k]);
      const [a, b] = hex(level[k]) < hex(level[k + 1]) ? [level[k], level[k + 1]] : [level[k + 1], level[k]];
      nextLevel.push(keccak(concat([a, b])));
      nextMembers.push(members[k].concat(members[k + 1]));
    }
    level = nextLevel;
    members = nextMembers;
  }
  const byPosition = {};
  positions.forEach((p, i) => {
    byPosition[p] = proofs[i].map(to0x);
  });
  return { root: to0x(level[0]), proofs: byPosition };
}

// ---------------------------------------------------------------------------------------------
// The input rule of FeeOracle and dataset.py
// ---------------------------------------------------------------------------------------------

/** move[k] = min(|tick[k+1] - tick[k]|, 127): the bytes FeeOracle stores. */
export const movesOf = (ticks) => ticks.slice(1).map((t, k) => Math.min(Math.abs(t - ticks[k]), CLIP));

/** The input of the window that ends at move `end`: moves end-144 .. end-1, oldest first. */
export function windowOf(moves, end) {
  if (end < N_OBS || end > moves.length) throw new Error(`no window ends at move ${end} (${moves.length} moves)`);
  return moves.slice(end - N_OBS, end);
}

/** keccak of the packed words: the arbiter's committed input hash (`keccak_words`). */
export const inputHash = (x0) => to0x(keccak(pack(x0)));

// ---------------------------------------------------------------------------------------------
// Calls and events of the three contracts
// ---------------------------------------------------------------------------------------------

export const SIG = {
  claim: 'claim(uint256[5],uint256[1],uint256[1],int32[3],uint8)',
  challenge1: 'challengeNeuron1(uint256,uint256,uint256[5],uint256[1],uint256[5],int32,bytes32[])',
  challenge2: 'challengeNeuron2(uint256,uint256,uint256[1],uint256[1],uint256[1],int32,bytes32[])',
  challenge3: 'challengeNeuron3(uint256,uint256,uint256[1],int32[3],uint256[1],int32,bytes32[])',
  challengeClass: 'challengeClass(uint256,int32[3])',
  finalize: 'finalize(uint256)',
  withdraw: 'withdraw()',
  result: 'result(uint256)',
  claimCount: 'claimCount()',
  claimInfo: 'claimInfo(uint256)',
  claimHashes: 'claimHashes(uint256)',
  creditOf: 'creditOf(address)',
  modelRoot: 'modelRoot()',
  bond: 'bond()',
  burned: 'burned()',
  window: 'window()',
  observe: 'observe()',
  apply: 'apply(uint256,uint256)',
  fee: 'fee()',
  inputWords: 'inputWords(uint256)',
  moves: 'moves()',
  relay: 'relay(uint256,int32[24],uint256)',
  setOracle: 'setOracle(address)',
  relayed: 'relayed()',
  firstBlock: 'firstBlock()',
  nextBlock: 'nextBlock()',
  latest: 'latest()',
  keeper: 'keeper()',
  oracle: 'oracle()',
  classify: 'classify(uint256[5])',
};

export const EVENTS = {
  Claimed: topic('Claimed(uint256,address,uint256[],uint256[],uint256[],int32[],uint8,uint256,uint256)'),
  NeuronSlashed: topic('NeuronSlashed(uint256,address,uint256,uint256)'),
  ClassSlashed: topic('ClassSlashed(uint256,address)'),
  Finalized: topic('Finalized(uint256,uint8)'),
  Relayed: topic('Relayed(uint256,uint256,int32,uint256)'),
  FeeSet: topic('FeeSet(uint256,uint256,uint8,uint32)'),
};

export const STATUS = { 1: 'open', 2: 'slashed', 3: 'final' };
export const BATCH = 24;

/** Errors the three contracts revert with, by selector, so that an answer can be named. */
const ERRORS = {};
for (const [name, types, meaning] of [
  ['UnknownClaim', 'uint256', 'no claim has this number'],
  ['ClaimNotOpen', 'uint256', 'the claim is already slashed or final'],
  ['TooLate', 'uint256', 'the challenge window has closed'],
  ['TooEarly', 'uint256', 'the challenge window is still open'],
  ['NoFraud', '', 'the contract recomputed exactly the committed value, so the claim stands'],
  ['NeuronOutOfRange', 'uint256', 'that neuron is not in the layer'],
  ['InputMismatch', '', 'the vector presented does not hash to the committed one'],
  ['OutputMismatch', '', 'the vector presented does not hash to the committed one'],
  ['NotCommitted', '', 'the weights do not verify against the model root'],
  ['NotFinal', 'uint256', 'the claim is not final'],
  ['NotNewer', 'uint256', 'a window at least as new is already applied'],
  ['NotThisWindow', 'uint256,uint256', 'the claim is about another input'],
  ['NoSuchWindow', 'uint256', 'there is no such window of 144 moves'],
  ['NotRelaying', '', 'the relay answers slot0() only inside relay()'],
  ['NotNext', 'uint256,uint256', 'the batch does not start at the next block of the grid'],
  ['NotKeeper', 'address', 'only the keeper relays'],
  ['NothingToWithdraw', '', 'no credit'],
  ['BondTooSmall', 'uint256,uint256', 'less than the bond was attached'],
]) {
  ERRORS[hex(selector(`${name}(${types})`))] = { name, meaning };
}

export function decodeRevert(bytes) {
  if (!bytes || bytes.length < 4) return { name: 'revert', meaning: 'the call reverted with no reason' };
  return ERRORS[hex(bytes.subarray(0, 4))] ?? { name: 'unknown error', meaning: hex(bytes) };
}

/** The calldata of an honest (or deliberately altered) claim about input x0. */
export function claimData(x0, run) {
  return encodeCall(SIG.claim, [pack(x0), pack(run.hidden[0]), pack(run.hidden[1]), ...run.logits.map((v) => word(v)), word(run.cls)]);
}

/** The claim's vectors, decoded from its `Claimed` event. */
export function decodeClaimed(log) {
  const data = fromHex(log.data);
  const array = (headIndex) => {
    const at = Number(readUint(data, headIndex * 32));
    const n = Number(readUint(data, at));
    return Array.from({ length: n }, (_, i) => data.subarray(at + 32 + i * 32, at + 64 + i * 32));
  };
  const bytesOf = (words, length) => Array.from(concat(words).subarray(0, length));
  return {
    id: Number(BigInt(log.topics[1])),
    claimant: '0x' + log.topics[2].slice(26),
    x0: bytesOf(array(0), N_OBS),
    hidden: [bytesOf(array(1), 32), bytesOf(array(2), 16)],
    logits: array(3).map((w) => Number(readInt(w))),
    cls: Number(readUint(data, 4 * 32)),
    bond: readUint(data, 5 * 32),
    deadline: Number(readUint(data, 6 * 32)),
    hash: log.transactionHash,
    block: parseInt(log.blockNumber, 16),
  };
}

/**
 * Walks a claim in the order of docs/spec-dispute.md section 6: hidden layer 1, hidden layer 2,
 * logits, then the class. The first value that differs from this machine's run is the finding;
 * null when the claim is what the network computes.
 */
export function audit(model, claim) {
  const honest = forward(model, claim.x0);
  const claimedLayers = [claim.hidden[0], claim.hidden[1], claim.logits];
  const honestLayers = [honest.hidden[0], honest.hidden[1], honest.logits];
  for (let l = 0; l < 3; l++) {
    for (let j = 0; j < honestLayers[l].length; j++) {
      if (claimedLayers[l][j] !== honestLayers[l][j]) {
        return { honest, finding: { kind: 'neuron', layer: l + 1, j, claimed: claimedLayers[l][j], recomputed: honestLayers[l][j] } };
      }
    }
  }
  const best = argmax(claim.logits);
  if (best !== claim.cls) return { honest, finding: { kind: 'class', claimed: claim.cls, recomputed: best } };
  return { honest, finding: null };
}

export function describeFinding(finding) {
  if (finding === null) return 'nothing wrong';
  if (finding.kind === 'class') return `class ${finding.claimed}, the logits say ${finding.recomputed}`;
  const where = finding.layer === 3 ? `logit ${finding.j}` : `layer ${finding.layer} neuron ${finding.j}`;
  return `${where}: claimed ${finding.claimed}, recomputed ${finding.recomputed}`;
}

/** The challenge a finding names; with no finding, a class challenge (which an honest claim refuses). */
export function challengeData(model, tree, id, claim, finding) {
  const neuron = (layer, j) => {
    const l = model.layers[layer - 1];
    return { row: pack(l.weights[j]), bias: word(l.bias[j]), proof: tree.proofs[`${layer}:${j}`] };
  };
  const logits = claim.logits.map((v) => word(v));
  if (finding?.kind === 'neuron') {
    const n = neuron(finding.layer, finding.j);
    const vectors = finding.layer === 1
      ? [pack(claim.x0), pack(claim.hidden[0])]
      : finding.layer === 2
        ? [pack(claim.hidden[0]), pack(claim.hidden[1])]
        : [pack(claim.hidden[1]), ...logits];
    const signature = [SIG.challenge1, SIG.challenge2, SIG.challenge3][finding.layer - 1];
    return encodeCall(signature, [word(id), word(finding.j), ...vectors, n.row, n.bias], n.proof);
  }
  return encodeCall(SIG.challengeClass, [word(id), ...logits]);
}

// ---------------------------------------------------------------------------------------------
// JSON-RPC
// ---------------------------------------------------------------------------------------------

/** JSON-RPC over fetch, trying each node in turn; an execution revert is an answer, not a failure. */
export class Node {
  constructor(urls, { fetcher = globalThis.fetch.bind(globalThis), timeout = 20000 } = {}) {
    this.urls = urls;
    this.fetch = fetcher;
    this.timeout = timeout;
    this.used = null;
    this.n = 0;
  }

  async post(url, body) {
    const response = await this.fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status} from ${new URL(url).host}`);
    return response.json();
  }

  async rpc(method, params) {
    let last = null;
    for (const url of this.urls) {
      try {
        const body = await this.post(url, { jsonrpc: '2.0', id: ++this.n, method, params });
        if (body.error && !isRevert(body.error)) throw new Error(`${method}: ${body.error.message}`);
        this.used = url;
        return body.error ? { error: body.error } : { result: body.result };
      } catch (error) {
        last = error;
      }
    }
    throw new Error(`no node answered ${method} (${last?.message ?? 'unknown error'})`);
  }

  async value(method, params) {
    const out = await this.rpc(method, params);
    if (out.error) throw new Error(`${method}: ${out.error.message}`);
    return out.result;
  }

  /** eth_call: { reverted, returnValue }. */
  async call(to, data, { from = null, block = 'latest' } = {}) {
    const tx = { to, data: typeof data === 'string' ? data : to0x(data) };
    if (from) tx.from = from;
    const out = await this.rpc('eth_call', [tx, block]);
    if (out.error) {
      const raw = typeof out.error.data === 'string' ? out.error.data : out.error.data?.data;
      return { reverted: true, returnValue: raw ? fromHex(raw) : new Uint8Array(0), message: out.error.message };
    }
    return { reverted: false, returnValue: fromHex(out.result) };
  }

  /** The return bytes of a view that must not revert. */
  async view(to, signature, parts = []) {
    const out = await this.call(to, encodeCall(signature, parts));
    if (out.reverted) throw new Error(`${signature} reverted: ${decodeRevert(out.returnValue).name}`);
    return out.returnValue;
  }

  async blockNumber() {
    return parseInt(await this.value('eth_blockNumber', []), 16);
  }

  /** eth_getLogs over [from, to] in chunks, halving the chunk when a node refuses the range. */
  async logs(filter, from, to, chunk = 10000) {
    const out = [];
    let start = from;
    let size = chunk;
    while (start <= to) {
      const end = Math.min(start + size - 1, to);
      try {
        const got = await this.value('eth_getLogs', [{ ...filter, fromBlock: '0x' + start.toString(16), toBlock: '0x' + end.toString(16) }]);
        out.push(...got);
        start = end + 1;
      } catch (error) {
        if (size <= 500) throw error;
        size = Math.floor(size / 2);
      }
    }
    return out;
  }
}

function isRevert(error) {
  const text = `${error.message ?? ''}`.toLowerCase();
  return error.code === 3 || text.includes('revert') || (error.data !== undefined && error.data !== null);
}

// ---------------------------------------------------------------------------------------------
// Mainnet: the pool's tick at a block, read from more than one public node
// ---------------------------------------------------------------------------------------------

export const POOL = '0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640';
export const MULTICALL3 = '0xcA11bde05977b3631167028862bE2a173976CA11';
export const MAINNET_RPC = ['https://eth.drpc.org', 'https://rpc.mevblocker.io', 'https://eth-mainnet.public.blastapi.io'];
export const SEPOLIA_RPC = ['https://ethereum-sepolia-rpc.publicnode.com', 'https://sepolia.gateway.tenderly.co', 'https://rpc.sepolia.ethpandaops.io'];

/**
 * Multicall3.aggregate([(multicall3, getCurrentBlockTimestamp()), (pool, slot0())]): the block
 * number, its timestamp and the pool's tick in one answer, as examples/amm-fee/fetch_data.py asks.
 */
const AGGREGATE = (() => {
  const w = (v) => v.toString(16).padStart(64, '0');
  const element = (address, sel) => address.slice(2).toLowerCase().padStart(64, '0') + w(0x40) + w(4) + sel.padEnd(64, '0');
  const first = element(MULTICALL3, '0f28c97d');
  const second = element(POOL, '3850c7bd');
  return '0x252dba42' + w(0x20) + w(2) + w(0x40) + w(0x40 + first.length / 2) + first + second;
})();

export function decodeAggregate(result, block) {
  const raw = fromHex(result);
  const words = [];
  for (let i = 0; i < raw.length; i += 32) words.push(readUint(raw, i));
  if (Number(words[0]) !== block) throw new Error(`asked for block ${block}, the node answered for ${words[0]}`);
  const base = Number(words[1]) / 32 + 1;
  if (Number(words[base - 1]) !== 2) throw new Error(`block ${block}: expected two results`);
  const first = base + Number(words[base]) / 32;
  const second = base + Number(words[base + 1]) / 32;
  if (Number(words[first]) !== 32 || Number(words[second]) !== 224) throw new Error(`block ${block}: unexpected result sizes`);
  const timestamp = Number(words[first + 1]);
  let tick = Number(words[second + 2] & ((1n << 24n) - 1n));
  if (tick >= 1 << 23) tick -= 1 << 24;
  return { block, timestamp, tick };
}

/**
 * The pool on mainnet, read from several public nodes. `ticksAt` asks each block of every node
 * until `agree` nodes have answered (and checks that each answered for the block asked); any
 * disagreement is an error, never a choice.
 */
export class Mainnet {
  constructor(urls = MAINNET_RPC, { agree = 2, fetcher, timeout = 20000 } = {}) {
    this.nodes = urls.map((u) => new Node([u], { fetcher, timeout }));
    this.agree = Math.min(agree, urls.length);
  }

  /** The newest block the nodes call `safe`: the smallest of the answers. */
  async safeHead() {
    const answers = await Promise.allSettled(this.nodes.map((n) => n.value('eth_getBlockByNumber', ['safe', false])));
    const heads = answers.filter((a) => a.status === 'fulfilled' && a.value).map((a) => parseInt(a.value.number, 16));
    if (heads.length < this.agree) throw new Error(`only ${heads.length} mainnet node(s) answered`);
    return Math.min(...heads);
  }

  async ticksFrom(node, blocks) {
    const payload = blocks.map((b, i) => ({ jsonrpc: '2.0', id: i, method: 'eth_call', params: [{ to: MULTICALL3, data: AGGREGATE }, '0x' + b.toString(16)] }));
    let answer = await node.post(node.urls[0], payload);
    if (!Array.isArray(answer)) answer = await Promise.all(payload.map((p) => node.post(node.urls[0], p)));
    const byId = new Map(answer.map((a) => [a.id, a]));
    return blocks.map((b, i) => {
      const a = byId.get(i);
      if (!a || !a.result) throw new Error(`block ${b}: ${a?.error?.message ?? 'no answer'}`);
      return decodeAggregate(a.result, b);
    });
  }

  /** [{ block, timestamp, tick, nodes }] for the given blocks, in batches of 25. */
  async ticksAt(blocks) {
    const out = [];
    for (let i = 0; i < blocks.length; i += 25) {
      const part = blocks.slice(i, i + 25);
      const answers = [];
      const hosts = [];
      for (const node of this.nodes) {
        if (answers.length >= this.agree) break;
        try {
          answers.push(await this.ticksFrom(node, part));
          hosts.push(new URL(node.urls[0]).host);
        } catch {
          // the next node
        }
      }
      if (answers.length < this.agree) throw new Error(`fewer than ${this.agree} mainnet nodes answered for blocks ${part[0]}..${part.at(-1)}`);
      part.forEach((b, k) => {
        const seen = answers.map((a) => a[k]);
        if (seen.some((s) => s.tick !== seen[0].tick || s.timestamp !== seen[0].timestamp)) {
          throw new Error(`mainnet nodes disagree about block ${b}: ${seen.map((s) => s.tick).join(' vs ')}`);
        }
        out.push({ ...seen[0], nodes: hosts });
      });
    }
    return out;
  }
}

/** A stand-in for Mainnet in tests: the ticks of data/ticks.csv (block 23,850,000 + 50 i). */
export class FileMainnet {
  constructor(ticks, { firstBlock = 23850000, head }) {
    this.ticks = ticks;
    this.firstBlock = firstBlock;
    this.head = head;
  }

  async safeHead() {
    return this.head;
  }

  async ticksAt(blocks) {
    return blocks.map((block) => {
      const i = (block - this.firstBlock) / STEP;
      if (!Number.isInteger(i) || i < 0 || i >= this.ticks.length || block > this.head) throw new Error(`no tick for block ${block}`);
      return { block, timestamp: 0, tick: this.ticks[i], nodes: ['file'] };
    });
  }
}

// ---------------------------------------------------------------------------------------------
// The deployed system, read back
// ---------------------------------------------------------------------------------------------

/** `config`: { relay, oracle, arbiter, network, fromBlock } from deploy/sepolia-fee.jsonl. */
export class FeeSystem {
  constructor(node, config, model, tree = null) {
    this.node = node;
    this.config = config;
    this.model = model;
    this.tree = tree ?? commitment(model);
  }

  async relayState() {
    const r = this.config.relay;
    const [count, first, latest] = await Promise.all([
      this.node.view(r, SIG.relayed),
      this.node.view(r, SIG.firstBlock),
      this.node.view(r, SIG.latest),
    ]);
    return {
      relayed: Number(readUint(count)),
      firstBlock: Number(readUint(first)),
      latestBlock: Number(readUint(latest, 0)),
      latestTick: Number(readInt(latest, 32)),
    };
  }

  /** Every relayed tick, from the relay's events: [{ number, block, tick, moves, sepoliaBlock, hash }]. */
  async relayedTicks(latest = null) {
    const to = latest ?? await this.node.blockNumber();
    const logs = await this.node.logs({ address: this.config.relay, topics: [EVENTS.Relayed] }, this.config.fromBlock, to);
    return logs.map((log) => {
      const data = fromHex(log.data);
      return {
        number: Number(BigInt(log.topics[1])),
        block: Number(BigInt(log.topics[2])),
        tick: Number(readInt(data, 0)),
        moves: Number(readUint(data, 32)),
        sepoliaBlock: parseInt(log.blockNumber, 16),
        hash: log.transactionHash,
      };
    }).sort((a, b) => a.number - b.number);
  }

  async oracleState() {
    const o = this.config.oracle;
    const [moves, fee] = await Promise.all([this.node.view(o, SIG.moves), this.node.view(o, SIG.fee)]);
    return { moves: Number(readUint(moves)), fee: Number(readUint(fee)) };
  }

  async inputWords(end) {
    const raw = await this.node.view(this.config.oracle, SIG.inputWords, [word(end)]);
    return Array.from(raw.subarray(0, N_OBS));
  }

  async feeSets(latest = null) {
    const to = latest ?? await this.node.blockNumber();
    const logs = await this.node.logs({ address: this.config.oracle, topics: [EVENTS.FeeSet] }, this.config.fromBlock, to);
    return logs.map((log) => {
      const data = fromHex(log.data);
      return {
        id: Number(BigInt(log.topics[1])),
        end: Number(BigInt(log.topics[2])),
        tier: Number(readUint(data, 0)),
        fee: Number(readUint(data, 32)),
        sepoliaBlock: parseInt(log.blockNumber, 16),
        hash: log.transactionHash,
      };
    });
  }

  async arbiterState() {
    const a = this.config.arbiter;
    const [count, root, bond, burned, window] = await Promise.all(
      [SIG.claimCount, SIG.modelRoot, SIG.bond, SIG.burned, SIG.window].map((s) => this.node.view(a, s)),
    );
    return {
      count: Number(readUint(count)),
      root: to0x(root),
      bond: readUint(bond),
      burned: readUint(burned),
      window: Number(readUint(window)),
    };
  }

  async info(id) {
    const r = await this.node.view(this.config.arbiter, SIG.claimInfo, [word(id)]);
    return {
      claimant: to0x(r.subarray(12, 32)),
      bond: readUint(r, 32),
      deadline: Number(readUint(r, 64)),
      cls: Number(readUint(r, 96)),
      status: Number(readUint(r, 128)),
    };
  }

  /** Every claim with its vectors (from its `Claimed` event), keyed by number. */
  async claims(latest = null) {
    const to = latest ?? await this.node.blockNumber();
    const logs = await this.node.logs({ address: this.config.arbiter, topics: [EVENTS.Claimed] }, this.config.fromBlock, to);
    const out = new Map();
    for (const log of logs) {
      const c = decodeClaimed(log);
      out.set(c.id, c);
    }
    return out;
  }

  /** How each slashed claim was slashed: { id: { hash, challenger, layer, neuron } }. */
  async slashes(latest = null) {
    const to = latest ?? await this.node.blockNumber();
    const logs = await this.node.logs({ address: this.config.arbiter, topics: [[EVENTS.NeuronSlashed, EVENTS.ClassSlashed]] }, this.config.fromBlock, to);
    const out = {};
    for (const log of logs) {
      const data = fromHex(log.data);
      const neuron = log.topics[0] === EVENTS.NeuronSlashed;
      out[Number(BigInt(log.topics[1]))] = {
        hash: log.transactionHash,
        challenger: '0x' + log.topics[2].slice(26),
        layer: neuron ? Number(readUint(data, 0)) : null,
        neuron: neuron ? Number(readUint(data, 32)) : null,
      };
    }
    return out;
  }

  /** The numbers of the finalised claims, from their `Finalized` events. */
  async finalized(latest = null) {
    const to = latest ?? await this.node.blockNumber();
    const logs = await this.node.logs({ address: this.config.arbiter, topics: [EVENTS.Finalized] }, this.config.fromBlock, to);
    return new Map(logs.map((log) => [Number(BigInt(log.topics[1])), { hash: log.transactionHash, sepoliaBlock: parseInt(log.blockNumber, 16) }]));
  }

  audit(claim) {
    return audit(this.model, claim);
  }

  challengeData(id, claim, finding) {
    return challengeData(this.model, this.tree, id, claim, finding);
  }

  /** Puts a call to the arbiter as an eth_call: would it be accepted right now? */
  async ask(data, from = '0x000000000000000000000000000000000000dEaD') {
    const out = await this.node.call(this.config.arbiter, data, { from });
    return out.reverted ? { accepted: false, why: decodeRevert(out.returnValue), message: out.message } : { accepted: true };
  }

  /** The whole network run by the deployed `Network` bytecode, read-only. */
  async classifyOnChain(x0) {
    if (!this.config.network) return null;
    const out = await this.node.call(this.config.network, encodeCall(SIG.classify, [pack(x0)]));
    if (out.reverted) throw new Error('Network.classify reverted');
    return Number(readUint(out.returnValue));
  }
}

/**
 * Compares relayed ticks with mainnet's ticks at the same blocks:
 * { checked, equal, wrong: [{ number, block, relayed, mainnet }] }.
 */
export function compareTicks(relayed, mainnet) {
  const byBlock = new Map(mainnet.map((m) => [m.block, m]));
  const wrong = [];
  let checked = 0;
  for (const r of relayed) {
    const m = byBlock.get(r.block);
    if (!m) continue;
    checked++;
    if (m.tick !== r.tick) wrong.push({ number: r.number, block: r.block, relayed: r.tick, mainnet: m.tick });
  }
  return { checked, equal: checked - wrong.length, wrong };
}

/**
 * The relay's own rules, checked from its events: tick n is of block first + 50 n, with no gap,
 * and the oracle held exactly n moves after tick n. Returns a list of problems (empty: none).
 */
export function checkGrid(relayed, firstBlock) {
  const problems = [];
  relayed.forEach((r, n) => {
    if (r.number !== n) problems.push(`event ${n} is numbered ${r.number}`);
    if (r.block !== firstBlock + STEP * n) problems.push(`tick ${n} is of block ${r.block}, the grid says ${firstBlock + STEP * n}`);
    if (r.moves !== n) problems.push(`after tick ${n} the oracle held ${r.moves} moves, expected ${n}`);
  });
  return problems;
}

/** Map from input hash to window end, for every window of the relayed ticks. */
export function windowsByHash(moves) {
  const out = new Map();
  for (let end = N_OBS; end <= moves.length; end++) out.set(inputHash(windowOf(moves, end)), end);
  return out;
}
