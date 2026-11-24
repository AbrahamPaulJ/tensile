// Offline tests of demo/src/tensile-fee-live-core.js, the arithmetic that the keeper, the watcher
// and the live status page share. No network.
//   - the JavaScript forward pass against the example's reference: every hidden activation, logit
//     and class of the 1,000 traced test samples (out/trace.jsonl), and the class of all 7,711 test
//     samples (out/pred.bin), on inputs rebuilt here from data/ticks.csv by the rule of dataset.py;
//   - the JavaScript Merkle tree against the model root compiled into the generated arbiter, and
//     (when Python is at hand) every proof against tools/gen_tree.py;
//   - calldata sizes against the measured calls of runs/arbiter-gas.json;
//   - the relay checks: a tampered tick and a gap in the grid are found.
// usage: node demo/test/tensile-fee-live-core.mjs
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  FileMainnet, N_OBS, audit, challengeData, checkGrid, claimData, commitment, compareTicks, decodeAggregate,
  forward, inputHash, movesOf, windowOf, windowsByHash,
} from '../src/tensile-fee-live-core.js';

const started = Date.now();
const at = (path) => fileURLToPath(new URL(path, import.meta.url));
const example = (path) => at(`../../examples/amm-fee/${path}`);
let failures = 0;
let checks = 0;
const check = (ok, text) => {
  checks++;
  if (!ok) {
    failures++;
    console.log(`WRONG: ${text}`);
  }
};

const model = JSON.parse(readFileSync(example('out/model.json'), 'utf8'));
const ticks = readFileSync(example('data/ticks.csv'), 'utf8').trim().split('\n').map(Number);
const moves = movesOf(ticks);
// dataset.py: the test split is the last 15% of the moves; sample k ends at move lo + 144 + k.
const lo = Math.floor((moves.length * 85) / 100);
const hi = moves.length;
const ends = [];
for (let end = lo + N_OBS; end <= hi - 6 - 24; end++) ends.push(end);
const pred = readFileSync(example('out/pred.bin'));
check(ends.length === pred.length, `${ends.length} test samples rebuilt, pred.bin has ${pred.length}`);

const traces = readFileSync(example('out/trace.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
let classes = 0;
let traced = 0;
ends.forEach((end, k) => {
  const run = forward(model, windowOf(moves, end));
  if (run.cls === pred[k]) classes++;
  if (k < traces.length) {
    const t = traces[k];
    const same = JSON.stringify(run.hidden) === JSON.stringify(t.hidden) && JSON.stringify(run.logits) === JSON.stringify(t.logits) && run.cls === t.class;
    if (same) traced++;
  }
});
check(classes === pred.length, `forward: ${classes} of ${pred.length} classes equal pred.bin`);
check(traced === traces.length, `forward: ${traced} of ${traces.length} traces equal trace.jsonl in every value`);
console.log(`forward pass: ${classes} of ${pred.length} classes equal pred.bin; ${traced} of ${traces.length} traces equal in every hidden activation and logit`);

// The floor shift on a negative accumulator (docs/spec.md section 6 asks every implementation to
// test it): a 1-1-1 model whose hidden accumulator is -1 with shift 1 gives floor(-0.5) = -1, so 0.
const tiny = { layers: [{ weights: [[1]], bias: [-2], shift: 1 }, { weights: [[1]], bias: [5], shift: null }] };
check(forward(tiny, [1]).hidden[0][0] === 0 && forward(tiny, [127]).hidden[0][0] === 62, 'the shift and clamp of a tiny model');
check(Math.floor(-1 / 2) === -1, 'floor(-1 / 2) is -1');

// The commitment: the root compiled into the generated model ingot.
const tree = commitment(model);
const fe = readFileSync(example('ingots/model/src/commitment.fe'), 'utf8');
const compiled = fe.match(/MODEL_ROOT: u256 = (0x[0-9a-fA-F]+)/)?.[1]?.toLowerCase();
check(compiled && BigInt(compiled) === BigInt(tree.root), `commitment root ${tree.root}, MODEL_ROOT in commitment.fe ${compiled}`);
let pythonChecked = 'not run (no python3)';
for (const python of ['python3', 'python']) {
  try {
    const dir = mkdtempSync(join(tmpdir(), 'fee-tree-'));
    execFileSync(python, ['-I', at('../../tools/gen_tree.py'), '--model', example('out/model.json'), '--out', join(dir, 'tree.json')], { stdio: 'pipe' });
    const py = JSON.parse(readFileSync(join(dir, 'tree.json'), 'utf8'));
    rmSync(dir, { recursive: true, force: true });
    const same = py.leaves.every((leaf) => JSON.stringify(leaf.proof) === JSON.stringify(tree.proofs[`${leaf.layer}:${leaf.neuron}`]));
    check(py.root === tree.root && same, 'the JavaScript tree differs from tools/gen_tree.py');
    pythonChecked = `${py.leaves.length} proofs equal tools/gen_tree.py`;
    break;
  } catch (error) {
    if (error.code !== 'ENOENT') {
      check(false, `tools/gen_tree.py failed: ${error.message}`);
      break;
    }
  }
}
console.log(`commitment: root ${tree.root} equals MODEL_ROOT; ${pythonChecked}`);

// Calldata sizes: the arbiter calls measured on anvil (runs/arbiter-gas.json).
const measured = Object.fromEntries(JSON.parse(readFileSync(example('runs/arbiter-gas.json'), 'utf8')).calls.map((c) => [c.call, c.calldata_bytes]));
const x0 = windowOf(moves, ends[0]);
const honest = forward(model, x0);
const claim = { x0, hidden: honest.hidden, logits: honest.logits, cls: honest.cls };
check(claimData(x0, honest).length === measured.claim, `claim calldata ${claimData(x0, honest).length} bytes, measured ${measured.claim}`);
const sizes = [
  [{ kind: 'neuron', layer: 1, j: 3 }, 'challengeNeuron1 (144 inputs)'],
  [{ kind: 'neuron', layer: 2, j: 3 }, 'challengeNeuron2 (32 inputs)'],
  [{ kind: 'neuron', layer: 3, j: 1 }, 'challengeNeuron3 (16 inputs)'],
  [{ kind: 'class' }, 'challengeClass'],
];
for (const [finding, name] of sizes) {
  const data = challengeData(model, tree, 0, claim, finding);
  // A proof's length depends on the neuron's place in the tree; the measured call used neuron 0.
  const proofLength = finding.kind === 'class' ? 0 : tree.proofs[`${finding.layer}:${finding.j}`].length - tree.proofs[`${finding.layer}:0`].length;
  check(data.length - 32 * proofLength === measured[name], `${name}: ${data.length} bytes, measured ${measured[name]}`);
}

// The audit names the value that was changed, and nothing in an honest claim.
check(audit(model, claim).finding === null, 'an honest claim audits clean');
const changed = (layer, j) => {
  const c = structuredClone(claim);
  if (layer === 3) c.logits[j] += 1;
  else c.hidden[layer - 1][j] += c.hidden[layer - 1][j] === 127 ? -1 : 1;
  return c;
};
for (const [layer, j] of [[1, 5], [2, 9], [3, 2]]) {
  const f = audit(model, changed(layer, j)).finding;
  check(f?.kind === 'neuron' && f.layer === layer && f.j === j, `a claim changed at layer ${layer} neuron ${j} audits as ${JSON.stringify(f)}`);
}
const wrongClass = { ...claim, cls: (claim.cls + 1) % 3 };
check(audit(model, wrongClass).finding?.kind === 'class', 'a wrong class is found');

// Input hashes: one per window, and the window of a sample is found by its hash.
const byHash = windowsByHash(moves.slice(0, 400));
check(byHash.get(inputHash(windowOf(moves, 300))) === 300, 'the window ending at move 300 is found by its hash');

// The relay checks.
const relayed = ticks.slice(0, 10).map((tick, n) => ({ number: n, block: 23850000 + 50 * n, tick, moves: n }));
const file = new FileMainnet(ticks, { head: 23850000 + 50 * 9 });
const mainnet = await file.ticksAt(relayed.map((r) => r.block));
const same = compareTicks(relayed, mainnet);
check(same.checked === 10 && same.wrong.length === 0, 'ten ticks of the file compare equal to themselves');
const tampered = relayed.map((r) => (r.number === 4 ? { ...r, tick: r.tick + 1 } : r));
const found = compareTicks(tampered, mainnet);
check(found.wrong.length === 1 && found.wrong[0].number === 4, 'a tampered tick is found');
check(checkGrid(relayed, 23850000).length === 0, 'the grid of ten ticks is clean');
check(checkGrid(relayed.filter((r) => r.number !== 6), 23850000).length > 0, 'a gap in the grid is found');

// decodeAggregate: the answer of Multicall3.aggregate for block 26479000, tick -201234 (built here).
const w = (v) => BigInt.asUintN(256, BigInt(v)).toString(16).padStart(64, '0');
const answer = '0x' + w(26479000) + w(0x40) + w(2) + w(0x40) + w(0x80) + w(32) + w(1795400000) + w(224)
  + w(123) + w(-201234) + w(0) + w(0) + w(0) + w(0) + w(1);
const decoded = decodeAggregate(answer, 26479000);
check(decoded.tick === -201234 && decoded.timestamp === 1795400000, `decodeAggregate: ${JSON.stringify(decoded)}`);
let refused = false;
try {
  decodeAggregate(answer, 26479050);
} catch {
  refused = true;
}
check(refused, 'an answer for another block is refused');

console.log(`tensile-fee-live core: ${checks} checks, ${failures} wrong, ${((Date.now() - started) / 1000).toFixed(1)} s`);
process.exit(failures ? 1 : 0);
