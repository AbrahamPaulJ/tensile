// The off-chain side of the dispute protocol (docs/spec-dispute.md). Nothing here decides anything:
// the network runs as Fe bytecode through `Network`, and every judgement comes from `Arbiter`. This
// file only encodes calls, decodes results, crafts the lies, and walks a claim to name the challenge.
import { concat, encodeCall, fromHex, hex, keccak, pack, readInt, readUint, selector, word } from './abi.js';
import { Chain } from './chain.js';

export const BOND = 10n ** 18n;
export const WINDOW = 3600n;
export const STATUS = { 1: 'open', 2: 'slashed', 3: 'final' };

const SIG = {
  classify: 'classify(uint256[25])',
  trace: 'trace(uint256[25])',
  neuron2: 'neuron2(uint256[1],uint256[1],int32)',
  claim: 'claim(uint256[25],uint256[1],int32[10],uint8)',
  challengeNeuron1: 'challengeNeuron1(uint256,uint256,uint256[25],uint256[1],uint256[25],int32,bytes32[])',
  challengeNeuron2: 'challengeNeuron2(uint256,uint256,uint256[1],int32[10],uint256[1],int32,bytes32[])',
  challengeClass: 'challengeClass(uint256,int32[10])',
  finalize: 'finalize(uint256)',
  withdraw: 'withdraw()',
  result: 'result(uint256)',
  claimInfo: 'claimInfo(uint256)',
  claimHashes: 'claimHashes(uint256)',
  creditOf: 'creditOf(address)',
  modelRoot: 'modelRoot()',
  bond: 'bond()',
  window: 'window()',
  claimCount: 'claimCount()',
};

// The errors of ingots/onchain/src/signals.fe, by selector.
const ERRORS = {};
for (const [name, types, meaning] of [
  ['BondTooSmall', 'uint256,uint256', 'less ETH was attached than the bond'],
  ['MalformedInput', '', 'the input has a byte above 127 or dirty padding'],
  ['MalformedHidden', '', 'the hidden vector has a byte above 127 or dirty padding'],
  ['ClassOutOfRange', 'uint8', 'the class is not an output of the network'],
  ['UnknownClaim', 'uint256', 'no claim has this number'],
  ['ClaimNotOpen', 'uint256', 'the claim is already slashed or final'],
  ['TooLate', 'uint256', 'the challenge window has closed'],
  ['TooEarly', 'uint256', 'the challenge window is still open'],
  ['NoFraud', '', 'the contract recomputed exactly the committed value, so the claim stands'],
  ['NeuronOutOfRange', 'uint256', 'that neuron is not in the layer'],
  ['InputMismatch', '', 'the vector presented does not hash to the committed one'],
  ['OutputMismatch', '', 'the vector presented does not hash to the committed one'],
  ['NotCommitted', '', 'the weights do not verify against the model root'],
  ['NotFinal', 'uint256', 'the claim is not final, so it has no result'],
  ['NothingToWithdraw', '', 'this account has no credit'],
  ['TransferFailed', '', 'the ETH transfer failed'],
]) {
  ERRORS[hex(selector(`${name}(${types})`))] = { name, meaning };
}

// The events of ingots/onchain/src/signals.fe, by topic.
const EVENTS = {};
for (const [name, signature] of [
  ['Claimed', 'Claimed(uint256,address,uint256[],uint256[],int32[],uint8,uint256,uint256)'],
  ['NeuronSlashed', 'NeuronSlashed(uint256,address,uint256,uint256)'],
  ['ClassSlashed', 'ClassSlashed(uint256,address)'],
  ['Finalized', 'Finalized(uint256,uint8)'],
  ['Withdrawn', 'Withdrawn(address,uint256)'],
]) {
  EVENTS[hex(keccak(new TextEncoder().encode(signature)))] = name;
}

export function decodeRevert(returnValue) {
  if (returnValue.length < 4) return { name: 'empty revert', meaning: 'the call reverted with no reason' };
  return ERRORS[hex(returnValue.subarray(0, 4))] ?? { name: 'unknown error', meaning: hex(returnValue) };
}

export function eventNames(logs) {
  return logs.map((log) => EVENTS[hex(log[1][0])] ?? 'unknown event');
}

const logitsWords = (logits) => concat(logits.map((v) => word(v)));
const packedRow = (neuron) => pack(Array.from(fromHex(neuron.row)));

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** Everything deployed: the Fe bytecode from `fe build`, unchanged. */
export class Deployment {
  static async create(assets, { names = null, onStep = () => {} } = {}) {
    const self = new Deployment();
    self.assets = assets;
    self.chain = await Chain.create(
      names ?? [
        { name: 'Deployer' },
        { name: 'Claimant' },
        { name: 'Watcher' },
      ],
    );
    self.deployer = Object.keys(self.chain.accounts)[0];
    const deploy = async (name, args = []) => {
      const data = concat([fromHex(assets.contracts[name].creation), ...args.map((a) => word(a))]);
      const tx = await self.chain.send({ from: self.deployer, data });
      if (tx.reverted) throw new Error(`deploying ${name}: ${tx.error}`);
      onStep({ name, gas: tx.gasUsed, ms: tx.ms, address: tx.address });
      return tx.address;
    };
    const asWord = (address) => BigInt(address.toString());
    self.low = await deploy('RowsLow');
    self.high = await deploy('RowsHigh');
    self.network = await deploy('Network', [asWord(self.low), asWord(self.high)]);
    self.check = await deploy('NeuronCheck');
    self.arbiter = await deploy('Arbiter', [BigInt(assets.model.root), BOND, WINDOW]);
    return self;
  }

  neuron(layer, j) {
    return this.assets.model.layers[layer - 1].neurons[j];
  }

  // --- the network, as Fe bytecode -----------------------------------------------------------

  /** `Network.trace` as a call: the honest hidden activations, logits and class. */
  async trace(x0) {
    const out = await this.chain.call({
      to: this.network,
      data: encodeCall(SIG.trace, [pack(x0)]),
    });
    if (out.reverted) throw new Error('Network.trace reverted');
    const hidden = Array.from(out.returnValue.subarray(0, 32));
    const logits = [];
    for (let j = 0; j < 10; j++) logits.push(Number(readInt(out.returnValue, 32 + j * 32)));
    return { hidden, logits, cls: argmax(logits), ms: out.ms, executionGas: out.executionGas };
  }

  /** `Network.classify` as a real transaction, for its gas. */
  async classifyTx(x0, from = this.deployer) {
    const tx = await this.chain.send({
      from,
      to: this.network,
      data: encodeCall(SIG.classify, [pack(x0)]),
    });
    if (tx.reverted) throw new Error('Network.classify reverted');
    return { cls: Number(readUint(tx.returnValue)), gas: tx.gasUsed, ms: tx.ms };
  }

  /** `Network.trace` as a real transaction: the same vectors, and what the whole run costs. */
  async traceTx(x0, from = this.deployer) {
    const tx = await this.chain.send({ from, to: this.network, data: encodeCall(SIG.trace, [pack(x0)]) });
    if (tx.reverted) throw new Error('Network.trace reverted');
    const hidden = Array.from(tx.returnValue.subarray(0, 32));
    const logits = [];
    for (let j = 0; j < 10; j++) logits.push(Number(readInt(tx.returnValue, 32 + j * 32)));
    return { hidden, logits, cls: argmax(logits), gas: tx.gasUsed, ms: tx.ms };
  }

  /** `NeuronCheck.neuron2`: the Fe code recomputing one logit from a hidden vector. */
  async logit(j, hidden) {
    const n = this.neuron(2, j);
    const out = await this.chain.call({
      to: this.check,
      data: encodeCall(SIG.neuron2, [packedRow(n), pack(hidden), word(n.bias)]),
    });
    if (out.reverted) throw new Error('NeuronCheck.neuron2 reverted');
    return Number(readInt(out.returnValue));
  }

  // --- the arbiter ---------------------------------------------------------------------------

  async sendArbiter(from, data, value = 0n) {
    const tx = await this.chain.send({ from, to: this.arbiter, data, value });
    return { ...tx, why: tx.reverted ? decodeRevert(tx.returnValue) : null, events: eventNames(tx.logs) };
  }

  postClaim(from, claim, value = BOND) {
    return this.sendArbiter(
      from,
      encodeCall(SIG.claim, [pack(claim.x0), pack(claim.hidden), logitsWords(claim.logits), word(claim.cls)]),
      value,
    );
  }

  challengeNeuron1(from, id, j, claim, neuron = this.neuron(1, j)) {
    return this.sendArbiter(
      from,
      encodeCall(
        SIG.challengeNeuron1,
        [word(id), word(j), pack(claim.x0), pack(claim.hidden), packedRow(neuron), word(neuron.bias)],
        neuron.proof,
      ),
    );
  }

  challengeNeuron2(from, id, j, claim, neuron = this.neuron(2, j)) {
    return this.sendArbiter(
      from,
      encodeCall(
        SIG.challengeNeuron2,
        [word(id), word(j), pack(claim.hidden), logitsWords(claim.logits), packedRow(neuron), word(neuron.bias)],
        neuron.proof,
      ),
    );
  }

  challengeClass(from, id, claim) {
    return this.sendArbiter(from, encodeCall(SIG.challengeClass, [word(id), logitsWords(claim.logits)]));
  }

  /** Sends the challenge a walk of the claim named. */
  challenge(from, id, claim, finding) {
    if (finding.kind === 'neuron' && finding.layer === 1) return this.challengeNeuron1(from, id, finding.j, claim);
    if (finding.kind === 'neuron') return this.challengeNeuron2(from, id, finding.j, claim);
    return this.challengeClass(from, id, claim);
  }

  finalize(from, id) {
    return this.sendArbiter(from, encodeCall(SIG.finalize, [word(id)]));
  }

  withdraw(from) {
    return this.sendArbiter(from, encodeCall(SIG.withdraw, []));
  }

  async view(signature, parts = []) {
    return this.chain.call({ to: this.arbiter, data: encodeCall(signature, parts) });
  }

  async claimInfo(id) {
    const out = await this.view(SIG.claimInfo, [word(id)]);
    const r = out.returnValue;
    return {
      bond: readUint(r, 32),
      deadline: readUint(r, 64),
      cls: Number(readUint(r, 96)),
      status: Number(readUint(r, 128)),
    };
  }

  /** `result(id)`: the class of a final claim, or the reason it has none. */
  async result(id) {
    const out = await this.view(SIG.result, [word(id)]);
    return out.reverted ? { why: decodeRevert(out.returnValue) } : { cls: Number(readUint(out.returnValue)) };
  }

  async creditOf(name) {
    const out = await this.view(SIG.creditOf, [word(BigInt(this.chain.accounts[name].address.toString()))]);
    return readUint(out.returnValue);
  }

  async modelRoot() {
    return `0x${hex((await this.view(SIG.modelRoot)).returnValue)}`;
  }

  async claimCount() {
    return readUint((await this.view(SIG.claimCount)).returnValue);
  }

  /** The vectors of a claim as committed: the claim's three hashes. */
  async claimHashes(id) {
    const r = (await this.view(SIG.claimHashes, [word(id)])).returnValue;
    return { x0: hex(r.subarray(0, 32)), hidden: hex(r.subarray(32, 64)), logits: hex(r.subarray(64, 96)) };
  }

  // --- what a watcher does -------------------------------------------------------------------

  /**
   * Walks a claim in the order of docs/spec-dispute.md section 6, as the native auditor does, and
   * names the first failing check. `honest` is the network's own output on the claim's input.
   */
  audit(claim, honest) {
    for (let j = 0; j < honest.hidden.length; j++) {
      if (claim.hidden[j] !== honest.hidden[j]) {
        return { kind: 'neuron', layer: 1, j, claimed: claim.hidden[j], recomputed: honest.hidden[j] };
      }
    }
    for (let j = 0; j < honest.logits.length; j++) {
      if (claim.logits[j] !== honest.logits[j]) {
        return { kind: 'neuron', layer: 2, j, claimed: claim.logits[j], recomputed: honest.logits[j] };
      }
    }
    const best = argmax(claim.logits);
    if (best !== claim.cls) return { kind: 'class', claimed: claim.cls, recomputed: best };
    return null;
  }

  // --- the lies ------------------------------------------------------------------------------

  honestClaim(x0, honest) {
    return { x0: Array.from(x0), hidden: honest.hidden.slice(), logits: honest.logits.slice(), cls: honest.cls };
  }

  /** The claimed class is wrong; the vectors are honest. Caught by the argmax. */
  lieAboutClass(x0, honest, target) {
    return { ...this.honestClaim(x0, honest), cls: target };
  }

  /** One logit is raised so that the argmax agrees with the false class. Caught at layer 2. */
  lieAboutLogit(x0, honest, target) {
    const logits = honest.logits.slice();
    logits[target] = Math.max(...logits) + 1;
    return { ...this.honestClaim(x0, honest), logits, cls: target };
  }

  /**
   * The strongest lie: a few hidden activations are forged, and the logits and the class are the
   * true consequences of the forged vector. Layer 2 and the argmax check out, so only re-running a
   * forged neuron of layer 1 on the committed input can catch it.
   */
  async lieAboutHidden(x0, honest, target) {
    const row = (j) => Array.from(fromHex(this.neuron(2, j).row), (b) => (b > 127 ? b - 256 : b));
    const rows = Array.from({ length: 10 }, (_, j) => row(j));
    const biases = Array.from({ length: 10 }, (_, j) => this.neuron(2, j).bias);
    const forward = (h) => rows.map((w, j) => biases[j] + w.reduce((sum, v, i) => sum + v * h[i], 0));
    const hidden = honest.hidden.slice();
    let logits = forward(hidden);
    const touched = new Set();
    while (argmax(logits) !== target && touched.size < hidden.length) {
      const rival = argmax(logits);
      let best = -1;
      let bestGain = -Infinity;
      let bestValue = 0;
      for (let i = 0; i < hidden.length; i++) {
        if (touched.has(i)) continue;
        const slope = rows[target][i] - rows[rival][i];
        const value = slope > 0 ? 127 : 0;
        const gain = slope * (value - hidden[i]);
        if (gain > bestGain) {
          bestGain = gain;
          best = i;
          bestValue = value;
        }
      }
      touched.add(best);
      hidden[best] = bestValue;
      logits = forward(hidden);
    }
    if (argmax(logits) !== target) return null;
    // Logits from the Fe code, so what is posted is exactly what layer 2 computes on the forgery.
    const posted = [];
    for (let j = 0; j < 10; j++) posted.push(await this.logit(j, hidden));
    return { x0: Array.from(x0), hidden, logits: posted, cls: argmax(posted), forged: Array.from(touched).sort((a, b) => a - b) };
  }
}
