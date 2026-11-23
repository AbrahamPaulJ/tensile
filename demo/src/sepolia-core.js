// A watcher for the arbiter on Sepolia, with no page around it: read a claim from the chain, re-run
// the network with the WebAssembly build of the Fe kernel, name the first wrong value, and build the
// challenge that proves it. Nothing here decides anything about a claim: `ask` puts the challenge to
// a public node as a call, and the answer is the deployed contract's.
import { concat, encodeCall, fromHex, hex, keccak, pack, readInt, readUint, selector, word } from './abi.js';

export const CHAIN_ID = 11155111;
export const STATUS = { 1: 'open', 2: 'slashed', 3: 'final' };

const SIG = {
  classify: 'classify(uint256[25])',
  claim: 'claim(uint256[25],uint256[1],int32[10],uint8)',
  challengeNeuron1: 'challengeNeuron1(uint256,uint256,uint256[25],uint256[1],uint256[25],int32,bytes32[])',
  challengeNeuron2: 'challengeNeuron2(uint256,uint256,uint256[1],int32[10],uint256[1],int32,bytes32[])',
  challengeClass: 'challengeClass(uint256,int32[10])',
  claimInfo: 'claimInfo(uint256)',
  claimCount: 'claimCount()',
  modelRoot: 'modelRoot()',
  bond: 'bond()',
  burned: 'burned()',
};
const CLAIMED = '0x' + hex(keccak(new TextEncoder().encode('Claimed(uint256,address,uint256[],uint256[],int32[],uint8,uint256,uint256)')));

// The arbiter's errors (ingots/onchain/src/signals.fe) a challenge can meet, by selector.
const ERRORS = {};
for (const [name, types, meaning] of [
  ['UnknownClaim', 'uint256', 'no claim has this number'],
  ['ClaimNotOpen', 'uint256', 'the claim is already slashed or final'],
  ['TooLate', 'uint256', 'the challenge window has closed'],
  ['NoFraud', '', 'the contract recomputed exactly the committed value, so the claim stands'],
  ['NeuronOutOfRange', 'uint256', 'that neuron is not in the layer'],
  ['InputMismatch', '', 'the vector presented does not hash to the committed one'],
  ['OutputMismatch', '', 'the vector presented does not hash to the committed one'],
  ['NotCommitted', '', 'the weights do not verify against the model root'],
]) {
  ERRORS[hex(selector(`${name}(${types})`))] = { name, meaning };
}

const to0x = (bytes) => '0x' + hex(bytes);
const logitsWords = (logits) => concat(logits.map((v) => word(v)));
const packedRow = (neuron) => pack(Array.from(fromHex(neuron.row)));

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** JSON-RPC over fetch, trying each node in turn. */
export class Node {
  constructor(urls, fetcher = globalThis.fetch.bind(globalThis)) {
    this.urls = urls;
    this.fetch = fetcher;
    this.used = null;
    this.n = 0;
  }

  async rpc(method, params) {
    let last = null;
    for (const url of this.urls) {
      try {
        const response = await this.fetch(url, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ jsonrpc: '2.0', id: ++this.n, method, params }),
        });
        const body = await response.json();
        this.used = url;
        // An execution revert is an answer, not a failure of the node.
        if (body.error) return { error: body.error };
        return { result: body.result };
      } catch (error) {
        last = error;
      }
    }
    throw new Error(`no Sepolia node answered (${last?.message ?? 'unknown error'})`);
  }

  /** eth_call: the return bytes, or the revert's bytes. */
  async call(to, data, from = null) {
    const tx = { to, data: to0x(data) };
    if (from) tx.from = from;
    const out = await this.rpc('eth_call', [tx, 'latest']);
    if (out.error) {
      const raw = typeof out.error.data === 'string' ? out.error.data : out.error.data?.data;
      return { reverted: true, returnValue: raw ? fromHex(raw) : new Uint8Array(0), message: out.error.message };
    }
    return { reverted: false, returnValue: fromHex(out.result) };
  }
}

export function decodeRevert(returnValue) {
  if (returnValue.length < 4) return { name: 'revert', meaning: 'the call reverted with no reason' };
  return ERRORS[hex(returnValue.subarray(0, 4))] ?? { name: 'unknown error', meaning: hex(returnValue) };
}

/** The arguments of a `claim` transaction, from its input data. */
export function decodeClaim(input) {
  const data = fromHex(input);
  if (hex(data.subarray(0, 4)) !== hex(selector(SIG.claim))) throw new Error('not a claim transaction');
  const body = data.subarray(4);
  const x0 = Array.from(body.subarray(0, 784));
  const hidden = Array.from(body.subarray(25 * 32, 25 * 32 + 32));
  const logits = [];
  for (let j = 0; j < 10; j++) logits.push(Number(readInt(body, 26 * 32 + j * 32)));
  const cls = Number(readUint(body, 36 * 32));
  return { x0, hidden, logits, cls };
}

export class Watcher {
  /** `config`: { arbiter, network, fromBlock, claims: [{ id, hash }] }; `model`: the neurons with proofs; `run`: the wasm kernel. */
  constructor(node, config, model, run) {
    this.node = node;
    this.config = config;
    this.model = model;
    this.run = run;
  }

  neuron(layer, j) {
    return this.model.layers[layer - 1].neurons[j];
  }

  async summary() {
    const [count, root, bond, burned] = await Promise.all(
      [SIG.claimCount, SIG.modelRoot, SIG.bond, SIG.burned].map((s) => this.node.call(this.config.arbiter, encodeCall(s, []))),
    );
    return {
      count: Number(readUint(count.returnValue)),
      root: to0x(root.returnValue),
      bond: readUint(bond.returnValue),
      burned: readUint(burned.returnValue),
    };
  }

  async info(id) {
    const r = (await this.node.call(this.config.arbiter, encodeCall(SIG.claimInfo, [word(id)]))).returnValue;
    return {
      claimant: to0x(r.subarray(12, 32)),
      bond: readUint(r, 32),
      deadline: Number(readUint(r, 64)),
      cls: Number(readUint(r, 96)),
      status: Number(readUint(r, 128)),
    };
  }

  /** The transaction that posted claim `id`: recorded with the page, or found by its `Claimed` event. */
  async claimHash(id) {
    const known = this.config.claims.find((c) => c.id === id);
    if (known) return known.hash;
    const latest = parseInt((await this.node.rpc('eth_blockNumber', [])).result, 16);
    const topic = to0x(word(id));
    for (let from = this.config.fromBlock; from <= latest; from += 40000) {
      const out = await this.node.rpc('eth_getLogs', [{
        address: this.config.arbiter,
        topics: [CLAIMED, topic],
        fromBlock: '0x' + from.toString(16),
        toBlock: '0x' + Math.min(from + 39999, latest).toString(16),
      }]);
      if (out.result?.length) return out.result[0].transactionHash;
    }
    throw new Error(`the Claimed event of claim ${id} was not found`);
  }

  /** What was claimed, read from the chain. */
  async claim(id) {
    const hash = await this.claimHash(id);
    const tx = (await this.node.rpc('eth_getTransactionByHash', [hash])).result;
    if (!tx || tx.to?.toLowerCase() !== this.config.arbiter.toLowerCase()) throw new Error(`transaction ${hash} is not a call of this arbiter`);
    return { ...decodeClaim(tx.input), hash, from: tx.from };
  }

  /**
   * Walks a claim in the order of docs/spec-dispute.md section 6, as the native auditor does: hidden
   * neurons, outputs, then the answer. The first value that differs from this machine's own run is
   * the challenge. Null: the claim is what the network computes.
   */
  audit(claim) {
    const honest = this.run(Uint8Array.from(claim.x0));
    for (let j = 0; j < honest.hidden.length; j++) {
      if (claim.hidden[j] !== honest.hidden[j]) {
        return { honest, finding: { kind: 'neuron', layer: 1, j, claimed: claim.hidden[j], recomputed: honest.hidden[j] } };
      }
    }
    for (let j = 0; j < honest.logits.length; j++) {
      if (claim.logits[j] !== honest.logits[j]) {
        return { honest, finding: { kind: 'neuron', layer: 2, j, claimed: claim.logits[j], recomputed: honest.logits[j] } };
      }
    }
    const best = argmax(claim.logits);
    if (best !== claim.cls) return { honest, finding: { kind: 'class', claimed: claim.cls, recomputed: best } };
    return { honest, finding: null };
  }

  /** The calldata of the challenge a finding names. With no finding: a class challenge, which an honest claim must refuse. */
  challengeData(id, claim, finding) {
    if (finding?.kind === 'neuron' && finding.layer === 1) {
      const n = this.neuron(1, finding.j);
      return encodeCall(
        SIG.challengeNeuron1,
        [word(id), word(finding.j), pack(claim.x0), pack(claim.hidden), packedRow(n), word(n.bias)],
        n.proof,
      );
    }
    if (finding?.kind === 'neuron') {
      const n = this.neuron(2, finding.j);
      return encodeCall(
        SIG.challengeNeuron2,
        [word(id), word(finding.j), pack(claim.hidden), logitsWords(claim.logits), packedRow(n), word(n.bias)],
        n.proof,
      );
    }
    return encodeCall(SIG.challengeClass, [word(id), logitsWords(claim.logits)]);
  }

  /** Puts a challenge to the node as a call: would the deployed arbiter accept it right now? */
  async ask(data, from = '0x000000000000000000000000000000000000dEaD') {
    const out = await this.node.call(this.config.arbiter, data, from);
    return out.reverted ? { accepted: false, why: decodeRevert(out.returnValue), message: out.message } : { accepted: true };
  }

  /** The whole network run by the bytecode on Sepolia, read-only. */
  async classifyOnChain(x0) {
    const out = await this.node.call(this.config.network, encodeCall(SIG.classify, [pack(x0)]));
    if (out.reverted) throw new Error('Network.classify reverted on Sepolia');
    return Number(readUint(out.returnValue));
  }
}

export const signatures = SIG;
export { to0x };
