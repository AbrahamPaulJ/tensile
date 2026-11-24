// The chain side of the fee-tier page: the contracts `fe build` produced from examples/amm-fee
// (Arbiter, Network, Rows0, FeeOracle and the test pool), deployed unchanged on the EVM of chain.js.
// This file encodes calls and decodes results. It decides nothing: every verdict, refusal and gas
// figure it returns came back from the bytecode.
import { concat, encodeCall, fromHex, hex, keccak, pack, readInt, readUint, selector, word } from './abi.js';
import { Chain } from './chain.js';
import { BOND, STATUS, WINDOW, decodeRevert as decodeArbiterRevert } from './protocol.js';

export { BOND, STATUS, WINDOW };
/** Blocks between two observations, as in examples/amm-fee/oracle_check.sh: 10 minutes of slots. */
export const INTERVAL = 50n;
const SLOT_SECONDS = 12n;

const words = (n) => Math.ceil(n / 32);

/** The signatures of the generated arbiter and network, from the layer sizes (tools/gen_arbiter.py). */
function signatures(layers) {
  const [input, ...rest] = [layers[0].in, ...layers.map((l) => l.out)];
  const classes = rest[rest.length - 1];
  const hidden = rest.slice(0, -1);
  const vector = (n) => `uint256[${words(n)}]`;
  const sig = {
    classify: `classify(${vector(input)})`,
    trace: `trace(${vector(input)})`,
    claim: `claim(${[vector(input), ...hidden.map(vector), `int32[${classes}]`, 'uint8'].join(',')})`,
    challengeClass: `challengeClass(uint256,int32[${classes}])`,
    challengeNeuron: [],
  };
  const sizes = [input, ...rest];
  for (let k = 1; k <= layers.length; k++) {
    const out = k === layers.length ? `int32[${classes}]` : vector(sizes[k]);
    sig.challengeNeuron[k] = `challengeNeuron${k}(uint256,uint256,${vector(sizes[k - 1])},${out},${vector(sizes[k - 1])},int32,bytes32[])`;
  }
  return sig;
}

const VIEW = {
  finalize: 'finalize(uint256)',
  withdraw: 'withdraw()',
  result: 'result(uint256)',
  claimInfo: 'claimInfo(uint256)',
  claimHashes: 'claimHashes(uint256)',
  creditOf: 'creditOf(address)',
  burned: 'burned()',
  modelRoot: 'modelRoot()',
  claimCount: 'claimCount()',
  setTick: 'setTick(int32)',
  slot0: 'slot0()',
  observe: 'observe()',
  apply: 'apply(uint256,uint256)',
  fee: 'fee()',
  inputWords: 'inputWords(uint256)',
  moves: 'moves()',
};

// The errors of examples/amm-fee/ingots/consumer/src/lib.fe, by selector. The arbiter's are in protocol.js.
const ORACLE_ERRORS = {};
for (const [name, types, meaning] of [
  ['TooSoon', 'uint256', 'the interval since the last observation has not passed'],
  ['NoSuchWindow', 'uint256', 'the oracle has not stored 144 moves ending there'],
  ['NotNewer', 'uint256', 'that window is not newer than the one already applied'],
  ['NotThisWindow', 'uint256,uint256', "the claim's committed input is not the hash of the ticks this contract observed"],
]) {
  ORACLE_ERRORS[hex(selector(`${name}(${types})`))] = { name, meaning };
}

export function decodeRevert(returnValue) {
  if (returnValue.length >= 4) {
    const own = ORACLE_ERRORS[hex(returnValue.subarray(0, 4))];
    if (own) return own;
  }
  return decodeArbiterRevert(returnValue);
}

// The events of the generated arbiter's signals.fe and of the consumer, by topic.
const EVENTS = {};
for (const [name, signature] of [
  ['Claimed', 'Claimed(uint256,address,uint256[],uint256[],uint256[],int32[],uint8,uint256,uint256)'],
  ['NeuronSlashed', 'NeuronSlashed(uint256,address,uint256,uint256)'],
  ['ClassSlashed', 'ClassSlashed(uint256,address)'],
  ['Finalized', 'Finalized(uint256,uint8)'],
  ['Withdrawn', 'Withdrawn(address,uint256)'],
  ['Observed', 'Observed(uint256,int32,uint8)'],
  ['FeeSet', 'FeeSet(uint256,uint256,uint8,uint32)'],
]) {
  EVENTS[hex(keccak(new TextEncoder().encode(signature)))] = name;
}
const eventNames = (logs) => logs.map((log) => EVENTS[hex(log[1][0])] ?? 'unknown event');

const logitsWords = (logits) => concat(logits.map((v) => word(v)));
const asWord = (address) => BigInt(address.toString());

/** Everything deployed: the Fe bytecode from `fe build`, unchanged. */
export class FeeDeployment {
  static async create(assets, { names = null, onStep = () => {} } = {}) {
    const self = new FeeDeployment();
    self.assets = assets;
    self.layers = assets.model.layers;
    self.sig = signatures(self.layers);
    self.chain = await Chain.create(names ?? [{ name: 'Deployer' }, { name: 'Keeper' }, { name: 'Watcher' }]);
    self.deployer = Object.keys(self.chain.accounts)[0];
    self.onStep = onStep;
    self.deployed = 0;
    self.rows = await self.deploy('Rows0');
    self.network = await self.deploy('Network', [asWord(self.rows)]);
    self.arbiter = await self.deploy('Arbiter', [BigInt(assets.model.root), BOND, WINDOW]);
    self.pool = await self.deploy('TestPool');
    await self.newOracle();
    return self;
  }

  async deploy(name, args = []) {
    const data = concat([fromHex(this.assets.contracts[name].creation), ...args.map((a) => word(a))]);
    const tx = await this.chain.send({ from: this.deployer, data });
    if (tx.reverted) throw new Error(`deploying ${name}: ${tx.error}`);
    this.deployed++;
    this.onStep({ name, gas: tx.gasUsed, ms: tx.ms, address: tx.address });
    return tx.address;
  }

  /** A fresh FeeOracle on the same pool and arbiter: it has observed nothing and applied nothing. */
  async newOracle() {
    this.oracle = await this.deploy('FeeOracle', [asWord(this.pool), asWord(this.arbiter), INTERVAL]);
    return this.oracle;
  }

  neuron(layer, j) {
    return this.layers[layer - 1].neurons[j];
  }

  async send(from, to, data, value = 0n) {
    const tx = await this.chain.send({ from, to, data, value });
    return { ...tx, why: tx.reverted ? decodeRevert(tx.returnValue) : null, events: eventNames(tx.logs) };
  }

  // --- the network, as Fe bytecode ---------------------------------------------------------------

  decodeTrace(r) {
    const sizes = this.layers.map((l) => l.out);
    const hidden = sizes.slice(0, -1).map((n, k) => Array.from(r.subarray(k * 32, k * 32 + n)));
    const at = (sizes.length - 1) * 32;
    const logits = [];
    for (let j = 0; j < sizes[sizes.length - 1]; j++) logits.push(Number(readInt(r, at + j * 32)));
    return { hidden, logits };
  }

  /** `Network.trace` as a call: every hidden vector and the logits, from the Fe code. */
  async trace(x0) {
    const out = await this.chain.call({ to: this.network, data: encodeCall(this.sig.trace, [pack(x0)]) });
    if (out.reverted) throw new Error('Network.trace reverted');
    return { ...this.decodeTrace(out.returnValue), ms: out.ms };
  }

  /** `Network.classify` as a real transaction: the tier, and what one whole run costs. */
  async classifyTx(x0, from = this.deployer) {
    const tx = await this.send(from, this.network, encodeCall(this.sig.classify, [pack(x0)]));
    if (tx.reverted) throw new Error('Network.classify reverted');
    return { ...tx, cls: Number(readUint(tx.returnValue)), gas: tx.gasUsed };
  }

  // --- the pool and the oracle --------------------------------------------------------------------

  /**
   * One observation, as oracle_check.sh makes it: the test pool is given the tick, `observe()` is
   * mined, and the rest of the interval passes with no transaction (anvil_mine there, the block
   * number and the clock here). Returns the `observe()` transaction.
   */
  async observeTick(tick, from = this.deployer) {
    const set = await this.send(this.deployer, this.pool, encodeCall(VIEW.setTick, [word(tick)]));
    if (set.reverted) throw new Error('TestPool.setTick reverted');
    const tx = await this.observe(from);
    this.mine(INTERVAL - 2n);
    return tx;
  }

  observe(from = this.deployer) {
    return this.send(from, this.oracle, encodeCall(VIEW.observe, []));
  }

  /** Empty blocks: only the block number and the clock move. */
  mine(blocks) {
    this.chain.number += blocks;
    this.chain.now += blocks * SLOT_SECONDS;
  }

  async oracleView(signature, parts = []) {
    return this.chain.call({ to: this.oracle, data: encodeCall(signature, parts) });
  }

  async moves() {
    return Number(readUint((await this.oracleView(VIEW.moves)).returnValue));
  }

  /** `inputWords(end)`: the packed input the contract built, as bytes, or the reason it has none. */
  async inputWords(end) {
    const out = await this.oracleView(VIEW.inputWords, [word(end)]);
    return out.reverted ? { why: decodeRevert(out.returnValue) } : { bytes: out.returnValue };
  }

  async fee() {
    return Number(readUint((await this.oracleView(VIEW.fee)).returnValue));
  }

  /** `apply(id, end)`: sets the fee from a finalised claim about the window ending at move `end`. */
  async apply(from, id, end) {
    const tx = await this.send(from, this.oracle, encodeCall(VIEW.apply, [word(id), word(end)]));
    return { ...tx, fee: tx.reverted ? null : Number(readUint(tx.returnValue)) };
  }

  // --- the arbiter --------------------------------------------------------------------------------

  claimData(claim) {
    return encodeCall(this.sig.claim, [pack(claim.x0), ...claim.hidden.map((h) => pack(h)), logitsWords(claim.logits), word(claim.cls)]);
  }

  postClaim(from, claim, value = BOND) {
    return this.send(from, this.arbiter, this.claimData(claim), value);
  }

  /** `challengeNeuronK(id, j, layer input, layer output, row, bias, proof)` for layer `k`. */
  challengeNeuron(from, id, k, j, claim) {
    const vectors = [claim.x0, ...claim.hidden, claim.logits];
    const last = k === this.layers.length;
    const n = this.neuron(k, j) ?? this.neuron(k, 0); // an index outside the layer: the contract refuses it
    return this.send(
      from,
      this.arbiter,
      encodeCall(
        this.sig.challengeNeuron[k],
        [word(id), word(j), pack(vectors[k - 1]), last ? logitsWords(vectors[k]) : pack(vectors[k]), pack(Array.from(fromHex(n.row))), word(n.bias)],
        n.proof,
      ),
    );
  }

  challengeClass(from, id, claim) {
    return this.send(from, this.arbiter, encodeCall(this.sig.challengeClass, [word(id), logitsWords(claim.logits)]));
  }

  /** Sends the challenge a walk of the claim named. */
  challenge(from, id, claim, finding) {
    if (finding.kind === 'neuron') return this.challengeNeuron(from, id, finding.layer, finding.j, claim);
    return this.challengeClass(from, id, claim);
  }

  finalize(from, id) {
    return this.send(from, this.arbiter, encodeCall(VIEW.finalize, [word(id)]));
  }

  withdraw(from) {
    return this.send(from, this.arbiter, encodeCall(VIEW.withdraw, []));
  }

  async view(signature, parts = []) {
    return this.chain.call({ to: this.arbiter, data: encodeCall(signature, parts) });
  }

  async claimInfo(id) {
    const r = (await this.view(VIEW.claimInfo, [word(id)])).returnValue;
    return { bond: readUint(r, 32), deadline: readUint(r, 64), cls: Number(readUint(r, 96)), status: Number(readUint(r, 128)) };
  }

  /** `result(id)`: the tier of a final claim, or the reason it has none. */
  async result(id) {
    const out = await this.view(VIEW.result, [word(id)]);
    return out.reverted ? { why: decodeRevert(out.returnValue) } : { cls: Number(readUint(out.returnValue)) };
  }

  /** The committed input hash of claim `id`, as the consumer reads it. */
  async claimInputHash(id) {
    return hex((await this.view(VIEW.claimHashes, [word(id)])).returnValue.subarray(0, 32));
  }

  async creditOf(name) {
    const out = await this.view(VIEW.creditOf, [word(asWord(this.chain.accounts[name].address))]);
    return readUint(out.returnValue);
  }

  async burned() {
    return readUint((await this.view(VIEW.burned)).returnValue);
  }

  async modelRoot() {
    return `0x${hex((await this.view(VIEW.modelRoot)).returnValue)}`;
  }

  async claimCount() {
    return readUint((await this.view(VIEW.claimCount)).returnValue);
  }
}

/** keccak256 of a vector's packed words: what the arbiter commits to and the oracle compares. */
export const hashOfInput = (x0) => hex(keccak(pack(x0)));
