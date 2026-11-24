// The watcher of the live fee-tier system on Sepolia: read-only, no key. It checks, from public
// nodes, everything the keeper is trusted for or obliged to do (docs/sepolia-fee.md):
//   - every tick the relay holds equals the mainnet pool's tick at the block the relay names for it,
//     read from two public mainnet nodes that must agree (the relay's TRUST ASSUMPTION, checked);
//     where the grid overlaps examples/amm-fee/data/ticks.csv, the relayed ticks equal that file;
//   - the relay kept its grid (tick n of block first + 50 n, no gap) and the oracle stored exactly
//     one move per relayed tick, each the clipped distance of the relayed ticks;
//   - the arbiter's model root is the committed model's;
//   - every claim is re-run here (the integer network in JavaScript, tested against the example's
//     reference vectors by tensile-fee-live-core.mjs): an honest claim must be refused a challenge
//     (NoFraud) while open; a false claim must be slashed or still challengeable; a false claim
//     that became final, or is past its window unslashed, fails the run; claims recorded as
//     planted must be found false at the value that was changed;
//   - `Network.classify` (the whole network as EVM bytecode on Sepolia) agrees with this run on
//     the newest claims, and the fee the oracle reports follows from its last applied claim;
//   - the relay is no more than LAG_LIMIT mainnet blocks behind mainnet's safe head (the keeper is
//     alive).
// usage: node demo/test/tensile-fee-live.mjs
//   env: SEPOLIA_RPC, MAINNET_RPC (comma-separated), RECORD_FILE (deploy/sepolia-fee.jsonl),
//        LAG_LIMIT (900 blocks, three hours), MAINNET_FILE + MAINNET_HEAD (tests: a ticks.csv as
//        mainnet), CHAIN_ID (11155111)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_FEE, FeeSystem, FileMainnet, GAP, HORIZON, MAINNET_RPC, Mainnet, Node, SEPOLIA_RPC, STATUS, STEP,
  checkGrid, compareTicks, describeFinding, inputHash, movesOf, windowsByHash,
} from '../src/tensile-fee-live-core.js';

const started = Date.now();
const env = process.env;
const at = (path) => fileURLToPath(new URL(path, import.meta.url));
const recordFile = env.RECORD_FILE ?? at('../../deploy/sepolia-fee.jsonl');
const record = readFileSync(recordFile, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
const by = (label) => record.find((r) => r.label === label);
const config = {
  arbiter: by('Arbiter').address,
  relay: by('TickRelay').address,
  oracle: by('FeeOracle').address,
  network: by('Network')?.address ?? null,
  fromBlock: Math.min(...record.filter((r) => r.block).map((r) => r.block)),
};
const model = JSON.parse(readFileSync(at('../../examples/amm-fee/out/model.json'), 'utf8'));
const dataset = readFileSync(at('../../examples/amm-fee/data/ticks.csv'), 'utf8').trim().split('\n').map(Number);
const DATASET_FIRST = 23850000;
const lagLimit = Number(env.LAG_LIMIT ?? 900);

let failures = 0;
const fail = (text) => {
  failures++;
  console.log(`WRONG: ${text}`);
};

const node = new Node(env.SEPOLIA_RPC ? env.SEPOLIA_RPC.split(',') : SEPOLIA_RPC);
const chainId = parseInt(await node.value('eth_chainId', []), 16);
if (chainId !== Number(env.CHAIN_ID ?? 11155111)) throw new Error(`chain ${chainId} is not the expected chain`);
const mainnet = env.MAINNET_FILE
  ? new FileMainnet(readFileSync(env.MAINNET_FILE, 'utf8').trim().split('\n').map(Number), { head: Number(env.MAINNET_HEAD) })
  : new Mainnet(env.MAINNET_RPC ? env.MAINNET_RPC.split(',') : MAINNET_RPC);
const system = new FeeSystem(node, config, model);

const head = await node.value('eth_getBlockByNumber', ['latest', false]);
const latest = parseInt(head.number, 16);
const now = parseInt(head.timestamp, 16);
const [relayState, oracleState, arbiterState, relayed, claims, slashes, finalized, feeSets] = await Promise.all([
  system.relayState(), system.oracleState(), system.arbiterState(), system.relayedTicks(latest),
  system.claims(latest), system.slashes(latest), system.finalized(latest), system.feeSets(latest),
]);

// The arbiter's model.
if (arbiterState.root !== system.tree.root) fail(`the arbiter's model root is ${arbiterState.root}, the model's is ${system.tree.root}`);
console.log(`Arbiter ${config.arbiter}: ${arbiterState.count} claims, bond ${arbiterState.bond} wei, window ${arbiterState.window} s, root as the model's`);

// The relay's grid and the oracle's moves.
for (const p of checkGrid(relayed, relayState.firstBlock)) fail(`relay: ${p}`);
if (relayed.length !== relayState.relayed) fail(`${relayed.length} Relayed events, relayed() says ${relayState.relayed}`);
const moves = movesOf(relayed.map((r) => r.tick));
if (relayed.length && moves.length !== oracleState.moves) fail(`${moves.length} moves from the relayed ticks, the oracle holds ${oracleState.moves}`);
console.log(`TickRelay ${config.relay}: ${relayed.length} ticks, mainnet blocks ${relayState.firstBlock} to ${relayState.latestBlock}; FeeOracle ${config.oracle}: ${oracleState.moves} moves, fee ${oracleState.fee}`);

// The relay against mainnet, every tick.
const safe = await mainnet.safeHead();
const checkable = relayed.filter((r) => r.block <= safe);
const real = await mainnet.ticksAt(checkable.map((r) => r.block));
const compared = compareTicks(checkable, real);
for (const w of compared.wrong) fail(`relayed tick ${w.number} (mainnet block ${w.block}) is ${w.relayed}; mainnet says ${w.mainnet}`);
const nodes = [...new Set(real.flatMap((r) => r.nodes))].join(', ');
console.log(`relay against mainnet: ${compared.equal} of ${compared.checked} ticks equal the pool's slot0() at their block (nodes: ${nodes || 'none'})`);
const overlap = relayed.filter((r) => r.block >= DATASET_FIRST && (r.block - DATASET_FIRST) / STEP < dataset.length && (r.block - DATASET_FIRST) % STEP === 0);
const overlapWrong = overlap.filter((r) => dataset[(r.block - DATASET_FIRST) / STEP] !== r.tick);
for (const r of overlapWrong) fail(`relayed tick ${r.number} (block ${r.block}) differs from data/ticks.csv`);
if (overlap.length) console.log(`relay against data/ticks.csv: ${overlap.length - overlapWrong.length} of ${overlap.length} overlapping ticks equal`);

// Liveness: how far the relay is behind mainnet's safe head.
const next = relayState.relayed ? relayState.firstBlock + STEP * relayState.relayed : null;
const lag = next === null ? null : Math.max(0, safe - next + STEP);
if (lag !== null && lag > lagLimit) fail(`the relay is ${lag} mainnet blocks behind the safe head (limit ${lagLimit}): the keeper is not running`);
console.log(`mainnet safe head ${safe}: the relay is ${lag ?? 'not started,'} ${lag === null ? '' : 'blocks behind'}`.trim());

// Every claim.
const windows = windowsByHash(moves);
const planted = new Map(record.filter((r) => r.label.startsWith('planted false claim')).map((r) => [Number(r.id), r]));
const sorted = [...claims.values()].sort((a, b) => a.id - b.id);
let onChainChecked = 0;
for (const claim of sorted) {
  const { honest, finding } = system.audit(claim);
  const status = slashes[claim.id] ? 2 : finalized.has(claim.id) ? 3 : (await system.info(claim.id)).status;
  const end = windows.get(inputHash(claim.x0));
  let verdict;
  if (finding === null) {
    if (status === 2) fail(`claim ${claim.id} is honest and was slashed`);
    if (status === 1) {
      const answer = await system.ask(system.challengeData(claim.id, claim, null));
      if (answer.accepted || answer.why.name !== 'NoFraud') fail(`claim ${claim.id} is honest and a challenge was not refused with NoFraud`);
      verdict = `honest, ${answer.why?.name ?? 'accepted'} to a challenge`;
    } else verdict = 'honest';
  } else if (status === 3) {
    fail(`claim ${claim.id} is false (${describeFinding(finding)}) and became final`);
    verdict = 'FALSE AND FINAL';
  } else if (status === 2) {
    verdict = `false, slashed in ${slashes[claim.id]?.hash.slice(0, 12)}...`;
  } else if (now >= claim.deadline) {
    fail(`claim ${claim.id} is false (${describeFinding(finding)}), open and past its window: nobody challenged it`);
    verdict = 'FALSE, UNCHALLENGED, WINDOW CLOSED';
  } else {
    const answer = await system.ask(system.challengeData(claim.id, claim, finding));
    if (!answer.accepted) fail(`claim ${claim.id} is false and open, and the challenge is refused: ${answer.why.name}`);
    verdict = `false and open: the arbiter would accept the challenge (${claim.deadline - now} s left)`;
  }
  const mark = planted.get(claim.id);
  if (mark) {
    const [what, j] = mark.posted.split(':');
    const layer = { layer1: 1, layer2: 2, logit: 3 }[what];
    const ok = what === 'class' ? finding?.kind === 'class' : finding?.kind === 'neuron' && finding.layer === layer && finding.j === Number(j);
    if (!ok) fail(`claim ${claim.id} was planted false at ${mark.posted} and the audit says ${describeFinding(finding)}`);
  }
  if (onChainChecked < 3 && claim.id >= sorted.length - 3 && config.network) {
    const cls = await system.classifyOnChain(claim.x0);
    if (cls !== honest.cls) fail(`claim ${claim.id}: Network.classify on chain says ${cls}, this run ${honest.cls}`);
    onChainChecked++;
  }
  const where = end === undefined ? 'not an oracle window' : `window ending at move ${end}`;
  console.log(`claim ${String(claim.id).padStart(3)} (${STATUS[status]}) by ${claim.claimant.slice(0, 10)}..., ${where}, tier ${claim.cls}: ${mark ? 'PLANTED, ' : ''}${verdict}`);
}
if (onChainChecked) console.log(`Network.classify on chain agrees with this run on the ${onChainChecked} newest claim(s)`);

// The fee the oracle reports follows from the last applied claim.
const last = feeSets.reduce((a, f) => (!a || f.end > a.end ? f : a), null);
const expected = last && oracleState.moves < last.end + GAP + HORIZON ? last.fee : DEFAULT_FEE;
if (oracleState.fee !== expected) fail(`fee() is ${oracleState.fee}, the last FeeSet (${last ? `claim ${last.id}, window ${last.end}` : 'none'}) gives ${expected}`);
for (const f of feeSets) {
  const claim = claims.get(f.id);
  if (!claim || system.audit(claim).finding !== null) fail(`FeeSet from claim ${f.id}, which is not an honest claim`);
}
console.log(`fee: ${oracleState.fee}${last ? ` from claim ${last.id} (window ${last.end}, applied ${feeSets.length} time(s) in all)` : ' (the default: nothing applied yet)'}`);
console.log(`tensile-fee-live watcher: ${failures} wrong result(s), ${((Date.now() - started) / 1000).toFixed(1)} s`);
process.exit(failures ? 1 : 0);
