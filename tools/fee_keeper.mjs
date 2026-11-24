// The keeper of the live fee-tier system on Sepolia (docs/sepolia-fee.md). One run reads everything
// it needs from the chains, does what is due, and stops; it keeps no state of its own, so a run can
// be repeated, skipped or late, and two runs in a row do nothing twice. In order of urgency:
//
//   1. watch: re-runs the network on every open claim (anyone's) and challenges a false one while
//      its window is open; the challenge is first put to the arbiter as a call;
//   2. finalize every open claim whose window has passed and that this run audited as honest;
//   3. apply the newest finalised honest claim about one of the oracle's windows to FeeOracle;
//   4. relay: reads the mainnet pool's tick at the next blocks of the relay's grid (every 50th
//      block, up to the newest block the nodes call `safe`), from two public nodes that must agree,
//      and relays them in batches of 24; a late or skipped run catches up here, each tick from its
//      own block (TRUST ASSUMPTION: the relay holds what this keeper writes; the watcher re-reads
//      every relayed tick from mainnet);
//   5. claims the newest window that ends on a multiple of CLAIM_EVERY moves, unless an honest
//      claim about it stands, with the arbiter's bond;
//   6. withdraws the keeper's credit (returned bonds, half of slashed ones).
//
// Spending is bounded: no transaction is sent when the base fee is above MAX_FEE_WEI, none pays more
// than MAX_FEE_WEI per gas, the gas limits of one run add up to at most GAS_BUDGET, and nothing is
// sent when the balance is under MIN_BALANCE_WEI. Every transaction is estimated first.
//
// usage: node tools/fee_keeper.mjs [--record] [--dry-run]
//        node tools/fee_keeper.mjs --plant <layer1:J | layer2:J | logit:J | class> --record
//   key:      SEPOLIA_KEEPER_KEY (0x...) or SEPOLIA_KEY_FILE (a file with a line private_key=0x...);
//             --plant uses ADVERSARY_KEY_FILE instead: a second account that posts one deliberately
//             false claim about the newest window, labelled as such in the record
//   chains:   SEPOLIA_RPC (comma-separated), MAINNET_RPC (comma-separated); for tests MAINNET_FILE
//             (a ticks.csv on the grid from block 23,850,000) with MAINNET_HEAD
//   limits:   GAS_BUDGET (6000000), MAX_FEE_WEI (50000000, 0.05 gwei), MIN_BALANCE_WEI (3e15),
//             MAX_TICKS (72 per run), CLAIM_EVERY (6 moves), START_BLOCK (the grid's first block,
//             only before the first relay; default 145 ticks before the safe head)
//   record:   RECORD_FILE (deploy/sepolia-fee.jsonl): the addresses, and with --record every
//             transaction this run sends is appended to it
import { createRequire } from 'node:module';
import { appendFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  BATCH, CLIP, FEES, FeeSystem, FileMainnet, Mainnet, N_OBS, Node, SEPOLIA_RPC, MAINNET_RPC, SIG, STATUS, STEP,
  checkGrid, claimData, decodeRevert, describeFinding, encodeCall, forward, fromHex, hex, inputHash, movesOf,
  readUint, to0x, windowOf, windowsByHash, word,
} from '../demo/src/tensile-fee-live-core.js';

const require = createRequire(new URL('../demo/package.json', import.meta.url));
const { createFeeMarket1559Tx } = require('@ethereumjs/tx');
const { Mainnet: MainnetChain, createCustomCommon } = require('@ethereumjs/common');

const at = (path) => fileURLToPath(new URL(path, import.meta.url));
const env = process.env;
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : null);
const started = Date.now();
const log = (text) => console.log(`[${((Date.now() - started) / 1000).toFixed(1).padStart(5)} s] ${text}`);

const LIMITS = {
  gasBudget: BigInt(env.GAS_BUDGET ?? 6000000),
  maxFee: BigInt(env.MAX_FEE_WEI ?? 50000000),
  minBalance: BigInt(env.MIN_BALANCE_WEI ?? 3000000000000000),
  maxTicks: Number(env.MAX_TICKS ?? 72),
  claimEvery: Number(env.CLAIM_EVERY ?? 6),
};
const RECORD_FILE = env.RECORD_FILE ?? at('../deploy/sepolia-fee.jsonl');
const record = readFileSync(RECORD_FILE, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
const by = (label) => record.find((r) => r.label === label);
for (const name of ['Arbiter', 'TickRelay', 'FeeOracle']) if (!by(name)?.address) throw new Error(`${RECORD_FILE} has no ${name}`);
const config = {
  arbiter: by('Arbiter').address,
  relay: by('TickRelay').address,
  oracle: by('FeeOracle').address,
  network: by('Network')?.address ?? null,
  fromBlock: Math.min(...record.filter((r) => r.block).map((r) => r.block)),
};
const model = JSON.parse(readFileSync(at('../examples/amm-fee/out/model.json'), 'utf8'));

function readKey() {
  if (flag('--plant')) {
    if (!env.ADVERSARY_KEY_FILE) throw new Error('--plant needs ADVERSARY_KEY_FILE');
    return keyFromFile(env.ADVERSARY_KEY_FILE);
  }
  if (env.SEPOLIA_KEEPER_KEY) return env.SEPOLIA_KEEPER_KEY.trim();
  if (env.SEPOLIA_KEY_FILE) return keyFromFile(env.SEPOLIA_KEY_FILE);
  throw new Error('set SEPOLIA_KEEPER_KEY or SEPOLIA_KEY_FILE');
}
function keyFromFile(path) {
  const line = readFileSync(path, 'utf8').split(/\r?\n/).find((l) => l.startsWith('private_key='));
  if (!line) throw new Error(`no private_key= line in the key file`);
  return line.slice('private_key='.length).trim();
}

/** Signs and sends transactions within the run's limits, one at a time, waiting for each receipt. */
class Sender {
  constructor(node, key, chainId) {
    this.node = node;
    this.key = fromHex(key.startsWith('0x') ? key : '0x' + key);
    this.common = createCustomCommon({ chainId: Number(chainId) }, MainnetChain, { hardfork: 'prague' });
    const probe = createFeeMarket1559Tx({ gasLimit: 21000n, maxFeePerGas: 1n }, { common: this.common }).sign(this.key);
    this.address = probe.getSenderAddress().toString();
    this.spent = 0n;
    this.sent = [];
    this.stopped = null;
  }

  async prepare() {
    const [latest, pending, balance, block] = await Promise.all([
      this.node.value('eth_getTransactionCount', [this.address, 'latest']),
      this.node.value('eth_getTransactionCount', [this.address, 'pending']),
      this.node.value('eth_getBalance', [this.address, 'latest']),
      this.node.value('eth_getBlockByNumber', ['latest', false]),
    ]);
    this.balance = BigInt(balance);
    this.baseFee = BigInt(block.baseFeePerGas ?? 0);
    this.nonce = BigInt(latest);
    if (BigInt(pending) > this.nonce) this.stopped = `a transaction of this account is still pending (nonce ${latest}): nothing is sent this run`;
    else if (this.baseFee > LIMITS.maxFee) this.stopped = `the base fee is ${this.baseFee} wei, above the cap of ${LIMITS.maxFee}: nothing is sent this run`;
    else if (this.balance < LIMITS.minBalance) this.stopped = `the balance is ${this.balance} wei, under the floor of ${LIMITS.minBalance}: nothing is sent this run`;
    let tip = 1n;
    try {
      tip = BigInt(await this.node.value('eth_maxPriorityFeePerGas', []));
    } catch {
      // keep 1 wei
    }
    this.tip = tip < 1n ? 1n : tip;
    this.maxFeePerGas = this.baseFee * 2n + this.tip;
    if (this.maxFeePerGas > LIMITS.maxFee) this.maxFeePerGas = LIMITS.maxFee;
    if (this.tip > this.maxFeePerGas - this.baseFee) this.tip = this.maxFeePerGas - this.baseFee;
  }

  /** Estimates, checks the limits, signs, sends and waits. Returns the receipt, or null if not sent. */
  async send(label, to, data, value = 0n, extra = {}) {
    if (this.stopped) return null;
    const call = { from: this.address, to, data: to0x(data), value: '0x' + value.toString(16) };
    let estimate = await this.node.rpc('eth_estimateGas', [call]);
    // After a transaction of this run, a public node may still answer from the block before it
    // (a claim just finalised reads as open): ask again for up to a minute.
    for (let attempt = 0; estimate.error && this.sent.length && attempt < 6; attempt++) {
      await new Promise((done) => setTimeout(done, 10000));
      estimate = await this.node.rpc('eth_estimateGas', [call]);
    }
    if (estimate.error) {
      log(`${label}: not sent, the node's estimate reverts (${estimate.error.message})`);
      this.failed = true;
      return null;
    }
    const gasLimit = (BigInt(estimate.result) * 6n) / 5n + 5000n;
    if (this.spent + gasLimit > LIMITS.gasBudget) {
      this.stopped = `the run's gas budget of ${LIMITS.gasBudget} is reached (${this.spent} used, ${label} needs ${gasLimit})`;
      log(this.stopped);
      return null;
    }
    if (flag('--dry-run')) {
      log(`${label}: dry run, would send (${BigInt(estimate.result)} gas estimated)`);
      return null;
    }
    const tx = createFeeMarket1559Tx({
      nonce: this.nonce, to, value, data, gasLimit, maxFeePerGas: this.maxFeePerGas, maxPriorityFeePerGas: this.tip,
    }, { common: this.common }).sign(this.key);
    const hash = await this.node.value('eth_sendRawTransaction', [to0x(tx.serialize())]);
    this.nonce += 1n;
    this.spent += gasLimit;
    let receipt = null;
    for (let i = 0; i < 120 && !receipt; i++) {
      receipt = await this.node.value('eth_getTransactionReceipt', [hash]).catch(() => null);
      if (!receipt) await new Promise((r) => setTimeout(r, i < 5 ? 500 : 3000));
    }
    if (!receipt) {
      this.stopped = `${label}: ${hash} not mined after waiting; the next run will see it`;
      log(this.stopped);
      return null;
    }
    const row = { label, hash, block: parseInt(receipt.blockNumber, 16), gasUsed: parseInt(receipt.gasUsed, 16), status: parseInt(receipt.status, 16), ...extra };
    this.sent.push(row);
    log(`${label}: ${row.status === 1 ? 'mined' : 'FAILED'}, gas ${row.gasUsed}, ${hash}`);
    if (row.status !== 1) this.failed = true;
    if (flag('--record')) appendFileSync(RECORD_FILE, JSON.stringify(row) + '\n');
    return receipt;
  }
}

async function main() {
  const node = new Node(env.SEPOLIA_RPC ? env.SEPOLIA_RPC.split(',') : SEPOLIA_RPC);
  const chainId = parseInt(await node.value('eth_chainId', []), 16);
  if (chainId !== Number(env.CHAIN_ID ?? 11155111)) throw new Error(`chain ${chainId} is not the expected chain`);
  const mainnet = env.MAINNET_FILE
    ? new FileMainnet(readFileSync(env.MAINNET_FILE, 'utf8').trim().split('\n').map(Number), { head: Number(env.MAINNET_HEAD) })
    : new Mainnet(env.MAINNET_RPC ? env.MAINNET_RPC.split(',') : MAINNET_RPC);
  const system = new FeeSystem(node, config, model);
  const sender = new Sender(node, readKey(), chainId);
  await sender.prepare();
  log(`account ${sender.address}, balance ${sender.balance} wei, base fee ${sender.baseFee} wei, max fee ${sender.maxFeePerGas} wei/gas, budget ${LIMITS.gasBudget} gas`);
  if (sender.stopped) log(sender.stopped);

  // The state of the three contracts, from views and events.
  const head = await node.value('eth_getBlockByNumber', ['latest', false]);
  const latest = parseInt(head.number, 16);
  const now = parseInt(head.timestamp, 16);
  const [relayState, oracleState, arbiterState, relayed, claims, slashes, feeSets, finalized] = await Promise.all([
    system.relayState(), system.oracleState(), system.arbiterState(),
    system.relayedTicks(latest), system.claims(latest), system.slashes(latest), system.feeSets(latest), system.finalized(latest),
  ]);
  const problems = checkGrid(relayed, relayState.firstBlock);
  if (relayed.length !== relayState.relayed) problems.push(`${relayed.length} Relayed events, relayed() says ${relayState.relayed}`);
  const moves = movesOf(relayed.map((r) => r.tick));
  if (relayed.length && moves.length !== oracleState.moves) problems.push(`${moves.length} moves from the relayed ticks, the oracle holds ${oracleState.moves}`);
  if (problems.length) {
    for (const p of problems) log(`WRONG: ${p}`);
    throw new Error('the relay and the oracle do not agree with their own events; nothing is sent');
  }
  const windows = windowsByHash(moves);
  log(`relay: ${relayState.relayed} ticks, next block ${relayState.relayed ? relayState.firstBlock + STEP * relayState.relayed : 'any'}; oracle: ${oracleState.moves} moves, fee ${oracleState.fee}; arbiter: ${arbiterState.count} claims`);

  if (flag('--plant')) return plant(sender, system, moves, arbiterState);

  // 1 and 2: watch and finalize. Status from events: slashed, finalised (claimInfo), or open.
  const audits = new Map();
  for (const [id, claim] of [...claims].sort((a, b) => a[0] - b[0])) {
    const { finding, honest } = system.audit(claim);
    audits.set(id, { finding, honest });
    if (slashes[id] || finalized.has(id)) {
      claim.status = slashes[id] ? 2 : 3;
      continue;
    }
    const info = await system.info(id);
    claim.status = info.status;
    if (info.status !== 1) continue;
    if (finding && now < claim.deadline) {
      const data = system.challengeData(id, claim, finding);
      const answer = await system.ask(data, sender.address);
      log(`claim ${id} by ${claim.claimant} is false (${describeFinding(finding)}); the arbiter ${answer.accepted ? 'accepts' : `refuses (${answer.why.name})`} the challenge`);
      if (answer.accepted) await sender.send(`challenge claim ${id}`, config.arbiter, data, 0n, { id: String(id), finding: describeFinding(finding) });
    } else if (finding) {
      log(`claim ${id} is false (${describeFinding(finding)}) and its window has closed: it is not finalised by this keeper`);
    } else if (now >= claim.deadline) {
      const data = encodeCall(SIG.finalize, [word(id)]);
      const answer = await system.ask(data, sender.address);
      if (answer.accepted) await sender.send(`finalize claim ${id}`, config.arbiter, data, 0n, { id: String(id) });
      else log(`finalize ${id}: refused (${answer.why.name})`);
    }
  }

  // 3: apply the newest finalised honest claim about one of the oracle's windows.
  const appliedEnd = feeSets.reduce((m, f) => Math.max(m, f.end), -1);
  let best = null;
  for (const [id, claim] of claims) {
    const end = windows.get(inputHash(claim.x0));
    const final = (await finalNow(system, id, claim)) && !audits.get(id).finding;
    if (end !== undefined && final && end > appliedEnd && (!best || end > best.end)) best = { id, end };
  }
  if (best && oracleState.moves < best.end + 6 + 24) {
    const data = encodeCall(SIG.apply, [word(best.id), word(best.end)]);
    const answer = await node.call(config.oracle, data, { from: sender.address });
    if (!answer.reverted) await sender.send(`apply claim ${best.id} (window ending at move ${best.end})`, config.oracle, data, 0n, { id: String(best.id), end: String(best.end) });
    else log(`apply ${best.id}: refused (${decodeRevert(answer.returnValue).name})`);
  }

  // 4: relay the ticks that are due, from the relay's next grid block to the safe head.
  const safe = await mainnet.safeHead();
  let next = relayState.relayed ? relayState.firstBlock + STEP * relayState.relayed : Number(env.START_BLOCK ?? Math.floor((safe - STEP * (N_OBS + 1)) / STEP) * STEP);
  if (next % STEP !== 0) throw new Error(`START_BLOCK ${next} is not a multiple of ${STEP}`);
  const gridStart = relayState.relayed ? relayState.firstBlock : next;
  const due = [];
  for (let b = next; b <= safe && due.length < LIMITS.maxTicks; b += STEP) due.push(b);
  const behind = Math.max(0, Math.floor((safe - next) / STEP) + 1);
  log(`mainnet safe head ${safe}: ${behind} tick(s) due, relaying up to ${due.length} this run`);
  const read = due.length ? await mainnet.ticksAt(due) : [];
  let relayedNow = 0;
  for (let i = 0; i < read.length && !sender.stopped; i += BATCH) {
    const part = read.slice(i, i + BATCH);
    const ticks = part.map((p) => p.tick).concat(new Array(BATCH - part.length).fill(0));
    const data = encodeCall(SIG.relay, [word(part[0].block), ...ticks.map((t) => word(t)), word(part.length)]);
    const receipt = await sender.send(`relay ticks ${relayState.relayed + i}..${relayState.relayed + i + part.length - 1} (mainnet blocks ${part[0].block}..${part.at(-1).block})`,
      config.relay, data, 0n, { first: String(part[0].block), count: String(part.length), nodes: part[0].nodes.join(' ') });
    if (!receipt || parseInt(receipt.status, 16) !== 1) break;
    for (const p of part) relayed.push({ tick: p.tick });
    relayedNow += part.length;
  }
  const allMoves = relayedNow ? movesOf(relayed.map((r) => r.tick)) : moves;

  // 5: claim the newest window that ends on a multiple of CLAIM_EVERY.
  const end = Math.floor(allMoves.length / LIMITS.claimEvery) * LIMITS.claimEvery;
  if (end >= N_OBS) {
    const x0 = windowOf(allMoves, end);
    const hash = inputHash(x0);
    const standing = [...claims].some(([id, c]) => inputHash(c.x0) === hash && !audits.get(id).finding && !slashes[id]);
    if (!standing) {
      // A public node may answer from a block before this run's own relay transaction: ask again
      // for up to a minute before treating a missing window as an error.
      let onChain;
      for (let attempt = 0; ; attempt++) {
        try {
          onChain = await system.inputWords(end);
          break;
        } catch (error) {
          if (!relayedNow || attempt >= 6 || !String(error.message).includes('NoSuchWindow')) throw error;
          await new Promise((done) => setTimeout(done, 10000));
        }
      }
      if (onChain.join() !== x0.join()) throw new Error(`the oracle's window ending at move ${end} differs from the relayed ticks`);
      const run = forward(model, x0);
      const sent = await sender.send(`claim window ${end}: tier ${run.cls} (${FEES[run.cls]})`, config.arbiter, claimData(x0, run), arbiterState.bond, { end: String(end), tier: String(run.cls) });
      if (sent) log(`claimed the window ending at move ${end} (its last tick is of mainnet block ${gridStart + STEP * end}): logits ${run.logits.join(' ')}, tier ${run.cls}`);
    } else {
      log(`window ${end} already has an honest claim standing`);
    }
  } else {
    log(`no window yet: ${allMoves.length} of ${N_OBS} moves`);
  }

  // 6: withdraw what the arbiter owes this account.
  const credit = readUint(await node.view(config.arbiter, SIG.creditOf, [word(sender.address)]));
  if (credit > 0n) await sender.send(`withdraw ${credit} wei`, config.arbiter, encodeCall(SIG.withdraw, []), 0n);

  summary(sender, relayedNow);
  return sender.failed ? 1 : 0;
}

/** Whether claim `id` is final now (its claimInfo status). */
async function finalNow(system, id, claim) {
  if (claim.status === 3) return true;
  if (claim.status === 2) return false;
  const info = await system.info(id);
  claim.status = info.status;
  return info.status === 3;
}

/** --plant: a second account posts one claim about the newest window with one value changed. */
async function plant(sender, system, moves, arbiterState) {
  const mutation = option('--plant');
  const end = moves.length;
  if (end < N_OBS) throw new Error(`no window yet (${end} moves)`);
  const x0 = windowOf(moves, end);
  const run = forward(model, x0);
  const bad = structuredClone(run);
  const [what, j] = mutation.split(':');
  if (what === 'layer1') bad.hidden[0][Number(j)] += bad.hidden[0][Number(j)] === CLIP ? -1 : 1;
  else if (what === 'layer2') bad.hidden[1][Number(j)] += bad.hidden[1][Number(j)] === CLIP ? -1 : 1;
  else if (what === 'logit') bad.logits[Number(j)] += 1;
  else if (what === 'class') bad.cls = (bad.cls + 1) % 3;
  else throw new Error(`unknown mutation ${mutation}`);
  const id = arbiterState.count;
  await sender.send(`planted false claim ${id}`, system.config.arbiter, claimData(x0, bad), arbiterState.bond,
    { id: String(id), end: String(end), posted: mutation, role: 'adversary (deliberately false, to show the dispute on the live system)', claimant: sender.address });
  summary(sender, 0);
  return sender.failed ? 1 : 0;
}

function summary(sender, relayedNow) {
  const used = sender.sent.reduce((n, r) => n + r.gasUsed, 0);
  log(`done: ${sender.sent.length} transaction(s), ${used} gas used, ${relayedNow} tick(s) relayed${sender.stopped ? `; stopped: ${sender.stopped}` : ''}`);
}

main().then((code) => process.exit(code)).catch((error) => {
  log(`keeper failed: ${error.message}`);
  if (env.DEBUG) console.log(error.stack);
  process.exit(1);
});
// `hex` and STATUS are part of the shared core; kept for debugging from a REPL.
void hex;
void STATUS;
