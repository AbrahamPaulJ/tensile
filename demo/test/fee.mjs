// Tests of the fee-tier page's engine in Node, against examples/amm-fee:
//   1. the page's data rule and integer arithmetic (src/fee-kernel.js) against the example's
//      reference output: every one of the test split's inputs, labels and classes, 1,000 traces;
//   2. the chain in the page reproduces, gas unit for gas unit, what the example's check.sh measured
//      on anvil (examples/amm-fee/runs/*.json), with the same transactions in the same order;
//   3. the Fe bytecode of `Network` agrees with the page's arithmetic and the reference;
//   4. every lie the page can tell is caught by the challenge the walk names and by no other, and an
//      honest claim survives every challenge;
//   5. FeeOracle builds the page's bytes from ticks, applies a final claim about them, and refuses a
//      claim about any other input, an open claim, a second apply and a slashed claim;
//   6. the deadline rules.
// The reference inputs and labels are not committed: `python3 -I examples/amm-fee/dataset.py`
// rebuilds them from data/ticks.csv (standard library, as examples/amm-fee/check.sh step 1 does).
// usage: node test/fee.mjs [evm-samples]        (default 40 forward passes on the EVM)
import { existsSync, readFileSync } from 'node:fs';
import { pack, hex } from '../src/abi.js';
import {
  argmax, audit, claimAboutOtherWindow, claimOnHalvedInput, forward, honestClaim, lieAboutClass, lieInLayer, loadNet, ticksOf, windowCount, windowOf,
} from '../src/fee-kernel.js';
import { BOND, FeeDeployment, INTERVAL, WINDOW, hashOfInput } from '../src/fee-protocol.js';

const here = new URL('.', import.meta.url);
const example = new URL('../../examples/amm-fee/', here);
const assets = JSON.parse(readFileSync(new URL('../generated/fee-assets.json', here), 'utf8'));
const EVM_SAMPLES = Number(process.argv[2] ?? 40);
const started = performance.now();

let failures = 0;
let checks = 0;
function check(condition, what) {
  checks++;
  if (!condition) {
    failures++;
    console.log(`FAIL  ${what}`);
  }
}
const show = (v) => JSON.stringify(v, (_, x) => (typeof x === 'bigint' ? x.toString() : x));
function equal(got, expected, what) {
  check(show(got) === show(expected), `${what}: got ${show(got)}, expected ${show(expected)}`);
}
const section = (title) => console.log(`--- ${title}`);
const record = (name) => JSON.parse(readFileSync(new URL(`runs/${name}`, example), 'utf8'));

for (const name of ['inputs.bin', 'labels.bin', 'past.bin']) {
  if (!existsSync(new URL(`out/${name}`, example))) {
    console.log(`examples/amm-fee/out/${name} is missing: run  python3 -I examples/amm-fee/dataset.py  first`);
    process.exit(2);
  }
}
const ref = (name) => readFileSync(new URL(`out/${name}`, example));
const inputs = ref('inputs.bin');
const labels = ref('labels.bin');
const past = ref('past.bin');
const pred = ref('pred.bin');
const traces = ref('trace.jsonl').toString('utf8').trim().split('\n').map((l) => JSON.parse(l));

const task = assets.task;
const N = task.n_obs;
const net = loadNet(assets.model);
const ticks = ticksOf(assets.ticks);
const windows = windowCount(ticks, task);
// The page's fee table, read from the consumer's constants by fee_assets.py; the contract must agree
// (every apply below), and so must oracle_check.sh, which writes the same table out.
const FEES = assets.fees.by_tier;
const DEFAULT_FEE = assets.fees.default;
equal([FEES, DEFAULT_FEE], [[500, 3000, 10000], 3000], 'the fee table against oracle_check.sh');

// ------------------------------------------------------------------------------------------------
section('the embedded ticks, the data rule and the arithmetic, against examples/amm-fee/out');
{
  const csv = readFileSync(new URL('data/ticks.csv', example), 'utf8').trim().split('\n').map(Number);
  equal(csv.length, task.series_ticks, 'ticks in data/ticks.csv');
  equal(Array.from(ticks), csv.slice(assets.ticks.first_index, assets.ticks.first_index + assets.ticks.count), 'the embedded slice against data/ticks.csv');
  equal(assets.ticks.first_index + assets.ticks.count, csv.length, 'the slice runs to the end of the series');
  equal(windows, pred.length, 'windows in the slice against the reference');
  equal(windows, assets.ticks.windows, 'windows against the count the exporter recorded');
  equal(inputs.length, windows * N, 'inputs.bin length');
  let badInput = 0;
  let badLabel = 0;
  let badPast = 0;
  let badClass = 0;
  let correct = 0;
  let persistence = 0;
  for (let i = 0; i < windows; i++) {
    const w = windowOf(ticks, task, i);
    if (Buffer.compare(Buffer.from(w.x0), inputs.subarray(i * N, (i + 1) * N)) !== 0) badInput++;
    if (w.label !== labels[i]) badLabel++;
    if (w.persistence !== past[i]) badPast++;
    const cls = forward(net, w.x0).cls;
    if (cls !== pred[i]) badClass++;
    if (cls === w.label) correct++;
    if (w.persistence === w.label) persistence++;
  }
  equal(badInput, 0, 'windows whose 144 bytes differ from inputs.bin');
  equal(badLabel, 0, 'windows whose label differs from labels.bin');
  equal(badPast, 0, 'windows whose persistence tier differs from past.bin');
  equal(badClass, 0, 'windows whose class differs from pred.bin');
  equal(correct, assets.metrics.model_correct, 'correct tiers against the recorded score');
  equal(persistence, assets.metrics.persistence_correct, 'persistence against the recorded score');
  let badTrace = 0;
  for (const t of traces) {
    const f = forward(net, windowOf(ticks, task, t.i).x0);
    if (show(f.hidden) !== show(t.hidden) || show(f.logits) !== show(t.logits) || f.cls !== t.class) badTrace++;
  }
  equal(badTrace, 0, 'traces that differ from trace.jsonl');
  // The shift rule on a negative accumulator (docs/spec.md section 6): floor, then the clamp.
  equal(-1 >> 1, -1, 'arithmetic shift of a negative value');
  console.log(`${windows} windows: inputs, labels, persistence and classes equal the reference (${correct} correct tiers); ${traces.length} traces (${traces.length * (32 + 16 + 3)} values) equal`);
}

// anvil's default account, the one the example's scripts send everything from.
const ANVIL = '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80';
const FUNDS = 1_000_000n * BOND;
const names = () => [
  { name: 'Deployer', key: ANVIL, balance: FUNDS },
  { name: 'Keeper', balance: FUNDS },
  { name: 'Watcher', balance: FUNDS },
  { name: 'Bystander', balance: FUNDS },
];
const me = 'Deployer';
const sample = (i) => windowOf(ticks, task, i);
const refClaim = (i) => ({ x0: sample(i).x0, hidden: traces[i].hidden.map((h) => h.slice()), logits: traces[i].logits.slice(), cls: traces[i].class });
const gasRows = [];
function measured(label, got, onAnvil) {
  gasRows.push([label, got, onAnvil]);
  equal(got, BigInt(onAnvil), `gas of ${label}`);
}

// ------------------------------------------------------------------------------------------------
section('deployment');
const deployStart = performance.now();
const d = await FeeDeployment.create(assets, { names: names() });
console.log(`deployed ${d.deployed} contracts in ${(performance.now() - deployStart).toFixed(0)} ms`);
equal(await d.modelRoot(), assets.model.root, 'model root read back');
equal(await d.fee(), DEFAULT_FEE, 'fee() of a fresh oracle');

// ------------------------------------------------------------------------------------------------
section('gas against anvil: tools/arbiter_gas_any.sh and tools/network_gas.sh, reference sample 0');
{
  const arbiterGas = Object.fromEntries(record('arbiter-gas.json').calls.map((c) => [c.call, c.gas_used]));
  const classified = await d.classifyTx(sample(0).x0);
  equal(classified.cls, pred[0], 'classify of sample 0');
  measured('Network.classify', classified.gas, record('network-gas.json').classify.gas_used);

  const honest = refClaim(0);
  const L = net.length;
  // model_args.py --mutate out:K:0 and --mutate class: one value changed by one, nothing recomputed.
  const mutated = (k) => {
    const c = refClaim(0);
    if (k === L) c.logits[0] += 1;
    else c.hidden[k - 1][0] += c.hidden[k - 1][0] === 127 ? -1 : 1;
    return c;
  };
  const badClass = { ...refClaim(0), cls: (honest.cls + 1) % net[L - 1].out };
  const r0 = await d.postClaim(me, honest);
  check(!r0.reverted, 'honest claim accepted');
  equal(r0.events, ['Claimed'], 'claim event');
  measured('claim', r0.gasUsed, arbiterGas.claim);
  for (const prefix of ['', 'later ']) {
    const first = Number(await d.claimCount());
    for (let k = 1; k <= L; k++) check(!(await d.postClaim(me, mutated(k))).reverted, 'false claim accepted');
    check(!(await d.postClaim(me, badClass)).reverted, 'false class claim accepted');
    for (let k = 1; k <= L; k++) {
      const tx = await d.challengeNeuron(me, BigInt(first + k - 1), k, 0, mutated(k));
      check(!tx.reverted, `${prefix}layer ${k} challenge succeeds (${tx.why?.name})`);
      equal(tx.events, ['NeuronSlashed'], `${prefix}layer ${k} challenge event`);
      measured(`${prefix}challengeNeuron${k}`, tx.gasUsed, arbiterGas[`${prefix}challengeNeuron${k} (${net[k - 1].in} inputs)`]);
      equal((await d.claimInfo(BigInt(first + k - 1))).status, 2, `slashed after the ${prefix}layer ${k} challenge`);
    }
    const tx = await d.challengeClass(me, BigInt(first + L), badClass);
    check(!tx.reverted, `${prefix}class challenge succeeds (${tx.why?.name})`);
    measured(`${prefix}challengeClass`, tx.gasUsed, arbiterGas[`${prefix}challengeClass`]);
  }
  equal((await d.challengeClass(me, 0n, honest)).why?.name, 'NoFraud', 'class challenge of the honest claim');
  d.chain.advance(WINDOW + 1n);
  const f0 = await d.finalize(me, 0n);
  check(!f0.reverted, 'finalize succeeds');
  measured('finalize', f0.gasUsed, arbiterGas.finalize);
  equal((await d.result(0n)).cls, honest.cls, 'result of the final claim');
  const slashes = BigInt(2 * L + 2);
  equal(await d.creditOf(me), slashes * (BOND / 2n) + BOND, 'credit: half of every slashed bond, and one returned');
  equal(await d.burned(), slashes * (BOND - BOND / 2n), 'burned: the other half of each slashed bond');
  const w0 = await d.withdraw(me);
  check(!w0.reverted, 'withdraw succeeds');
  measured(`withdraw (${slashes + 1n} credits)`, w0.gasUsed, arbiterGas[`withdraw (${slashes + 1n} credits)`]);
  equal(await d.chain.balance(d.arbiter), await d.burned(), 'the arbiter holds only the burned halves after withdraw');
}

// ------------------------------------------------------------------------------------------------
section('gas and behaviour against anvil: examples/amm-fee/oracle_check.sh, the same transactions');
{
  const anvil = record('oracle-gas.json');
  const o = await FeeDeployment.create(assets, { names: names() });
  const s = pred.findIndex((v) => v !== 1);
  equal(s, anvil.sample, 'the sample oracle_check.sh picks');
  const w = sample(s);
  const later = [];
  let first = null;
  for (let i = 0; i < w.ticks.length; i++) {
    const tx = await o.observeTick(w.ticks[i]);
    check(!tx.reverted, `observe ${i} mined`);
    if (i === 0) first = tx.gasUsed;
    else later.push(tx.gasUsed);
    if (i === 1) {
      // The interval has passed here (observeTick mines it), so go back inside it for this one call.
      o.mine(-(INTERVAL - 2n));
      const again = await o.observe();
      equal(again.why?.name, 'TooSoon', 'observe() inside the interval');
      o.mine(INTERVAL - 3n);
    }
  }
  equal(await o.moves(), N, 'moves() after 145 observations');
  equal(hex((await o.inputWords(N)).bytes), hex(pack(Array.from(inputs.subarray(s * N, (s + 1) * N)))), `inputWords(144) against sample ${s} of inputs.bin`);
  const c0 = await o.postClaim(me, refClaim(s));
  measured('oracle run: claim', c0.gasUsed, anvil.claim);
  check((await o.apply(me, 0n, N)).reverted, 'apply of an open claim reverts');
  o.chain.advance(WINDOW + 1n);
  const f0 = await o.finalize(me, 0n);
  measured('oracle run: finalize', f0.gasUsed, anvil.finalize);
  equal(await o.fee(), DEFAULT_FEE, 'fee() before any claim is applied');
  const a0 = await o.apply(me, 0n, N);
  check(!a0.reverted, `apply succeeds (${a0.why?.name})`);
  equal(a0.events, ['FeeSet'], 'apply event');
  measured('apply, the first', a0.gasUsed, anvil.apply);
  equal(await o.fee(), FEES[pred[s]], 'fee() after apply');
  equal(await o.fee(), anvil.fee, 'fee() against the anvil record');
  equal((await o.apply(me, 0n, N)).why?.name, 'NotNewer', 'applying the same window twice');
  // The next window: one more observation, so its 144 moves start inside a stored word.
  const next = sample(s + 1);
  later.push((await o.observeTick(next.ticks[N])).gasUsed);
  equal(hex((await o.inputWords(N + 1)).bytes), hex(pack(next.x0)), 'inputWords(145) against the next sample');
  check(!(await o.postClaim(me, refClaim(s + 1))).reverted, 'the next claim');
  o.chain.advance(WINDOW + 1n);
  check(!(await o.finalize(me, 1n)).reverted, 'the next claim finalised');
  check((await o.apply(me, 1n, N)).reverted, 'a claim about window 145 is not applied to window 144');
  const a1 = await o.apply(me, 1n, N + 1);
  measured('apply, the next one', a1.gasUsed, anvil.apply_later);
  equal(await o.fee(), FEES[pred[s + 1]], 'fee() after the second apply');
  const sorted = later.map(Number).sort((a, b) => a - b);
  measured('observe(), the first', first, anvil.observe_first);
  measured('observe(), median of the later', BigInt(sorted[sorted.length >> 1]), anvil.observe_median);
  measured('observe(), lowest', BigInt(sorted[0]), anvil.observe_min);
  measured('observe(), highest', BigInt(sorted[sorted.length - 1]), anvil.observe_max);
  equal(later.length + 1, anvil.observations, 'observations made');
}
for (const [label, got, onAnvil] of gasRows) console.log(`${label.padEnd(32)} page ${String(got).padStart(8)}   anvil ${String(onAnvil).padStart(8)}`);

// ------------------------------------------------------------------------------------------------
section(`the Fe bytecode of Network against the page's arithmetic and the reference, ${EVM_SAMPLES} samples`);
{
  const diffStart = performance.now();
  let mismatches = 0;
  for (let n = 0; n < EVM_SAMPLES; n++) {
    // The first half are traced reference samples, the rest spread over the whole split.
    const i = n < EVM_SAMPLES / 2 ? n : Math.floor((n * (windows - 1)) / (EVM_SAMPLES - 1));
    const x0 = sample(i).x0;
    const evm = await d.trace(x0);
    const js = forward(net, x0);
    let same = show(evm.hidden) === show(js.hidden) && show(evm.logits) === show(js.logits) && argmax(evm.logits) === pred[i];
    if (i < traces.length) same = same && show(evm.hidden) === show(traces[i].hidden) && show(evm.logits) === show(traces[i].logits);
    if (!same) mismatches++;
  }
  equal(mismatches, 0, 'EVM traces that differ from the page or the reference');
  console.log(`${EVM_SAMPLES} traces, ${mismatches} mismatches, ${((performance.now() - diffStart) / EVM_SAMPLES).toFixed(0)} ms per forward pass on the EVM`);
}

// ------------------------------------------------------------------------------------------------
section('every lie, on windows spread over the test split');
{
  const f = await FeeDeployment.create(assets, { names: names() });
  const L = net.length;
  let id = 0n;
  let lies = 0;
  let unreachable = 0;
  const lieGas = {};
  const picks = Array.from({ length: 12 }, (_, n) => Math.floor((n * (windows - 1)) / 11));
  for (const i of picks) {
    const w = sample(i);
    const honest = honestClaim(net, w.x0);
    equal(honest.cls, pred[i], `class of window ${i}`);
    equal(audit(net, honest), null, 'audit of an honest claim');
    check(!(await f.postClaim('Keeper', honest)).reverted, 'honest claim posted');
    const honestId = id++;
    equal(await f.claimInputHash(honestId), hashOfInput(w.x0), 'the committed input hash is the hash of the packed window');
    equal((await f.challengeClass('Watcher', honestId, honest)).why?.name, 'NoFraud', 'class challenge of honest');
    for (let k = 1; k <= L; k++) {
      equal((await f.challengeNeuron('Watcher', honestId, k, i % net[k - 1].out, honest)).why?.name, 'NoFraud', `layer ${k} challenge of honest`);
    }
    equal((await f.challengeNeuron('Watcher', honestId, 1, net[0].out, honest)).why?.name, 'NeuronOutOfRange', 'a neuron outside the layer');
    // A challenge that presents other vectors than the committed ones is refused before any arithmetic.
    const other = honestClaim(net, sample((i + 1) % windows).x0);
    equal((await f.challengeNeuron('Watcher', honestId, 1, 0, other)).why?.name, 'InputMismatch', 'a challenge with another input');

    // Correct arithmetic about an input the pool never produced: nothing to slash.
    const halved = claimOnHalvedInput(net, w.x0);
    equal(audit(net, halved), null, 'audit of the halved-input claim');
    check(!(await f.postClaim('Keeper', halved)).reverted, 'halved-input claim posted');
    const halvedId = id++;
    equal((await f.challengeClass('Watcher', halvedId, halved)).why?.name, 'NoFraud', 'halved input: class holds');
    for (let k = 1; k <= L; k++) {
      equal((await f.challengeNeuron('Watcher', halvedId, k, 0, halved)).why?.name, 'NoFraud', `halved input: layer ${k} holds`);
    }

    for (const target of [0, 1, 2].filter((c) => c !== honest.cls)) {
      const lying = [['class', lieAboutClass(net, w.x0, target)]];
      for (let k = 1; k <= L; k++) lying.push([`layer${k}`, lieInLayer(net, w.x0, k, target)]);
      for (const [kind, claim] of lying) {
        if (claim === null) {
          unreachable++;
          continue;
        }
        lies++;
        equal(claim.cls, target, `${kind} lie claims the target`);
        equal(claim.x0, w.x0, `${kind} lie keeps the input`);
        const finding = audit(net, claim);
        check(finding !== null, `${kind} lie is found`);
        equal(finding.kind === 'class' ? 'class' : `layer${finding.layer}`, kind, `${kind} lie is named at its layer`);
        const post = await f.postClaim('Keeper', claim);
        check(!post.reverted, `${kind} lie posted (${post.why?.name})`);
        const lieId = id++;
        if (finding.kind === 'neuron') {
          // Everything after the forged layer is true of the forgery, so those checks all hold, and
          // so does everything before it and every untouched neuron of the layer itself.
          equal(finding.j, claim.forged.neurons[0], `${kind} lie: the first forged neuron is named`);
          equal((await f.challengeClass('Watcher', lieId, claim)).why?.name, 'NoFraud', `${kind} lie: class check holds`);
          for (let k = 1; k <= L; k++) {
            for (let j = 0; j < net[k - 1].out; j++) {
              if (k === finding.layer && claim.forged.neurons.includes(j)) continue;
              if (k === 1 && j % 8 !== i % 8) continue; // layer 1 is the dearest: an eighth of its neurons per claim
              equal((await f.challengeNeuron('Watcher', lieId, k, j, claim)).why?.name, 'NoFraud', `${kind} lie: layer ${k} neuron ${j} holds`);
            }
          }
        } else {
          for (let k = 1; k <= L; k++) equal((await f.challengeNeuron('Watcher', lieId, k, 0, claim)).why?.name, 'NoFraud', 'class lie: neurons hold');
        }
        const before = await f.creditOf('Watcher');
        const burnedBefore = await f.burned();
        const hit = await f.challenge('Watcher', lieId, claim, finding);
        check(!hit.reverted, `${kind} lie slashed (${hit.why?.name})`);
        (lieGas[kind] ??= []).push(hit.gasUsed);
        equal((await f.claimInfo(lieId)).status, 2, `${kind} lie status`);
        equal(await f.creditOf('Watcher'), before + BOND / 2n, `${kind} lie pays the watcher half the bond`);
        equal(await f.burned(), burnedBefore + (BOND - BOND / 2n), `${kind} lie burns the other half`);
        equal((await f.challenge('Bystander', lieId, claim, finding)).why?.name, 'ClaimNotOpen', 'a second challenge');
        equal((await f.result(lieId)).why?.name, 'NotFinal', 'a slashed claim has no result');
      }
    }
  }
  check(lies >= picks.length * 2 * 3, `at least three kinds of lie per target found (${lies})`);
  console.log(`${lies} lies caught on ${picks.length} windows; ${unreachable} forgeries not found (no change in that layer reaches the target)`);
  for (const [kind, gases] of Object.entries(lieGas)) {
    const sorted = gases.map(Number).sort((a, b) => a - b);
    console.log(`  ${kind.padEnd(7)} ${String(gases.length).padStart(3)} lies, challenge gas ${sorted[0]} to ${sorted[sorted.length - 1]}`);
  }
}

// ------------------------------------------------------------------------------------------------
section('FeeOracle: the page\'s bytes from ticks, the fee from a final claim, every refusal');
{
  const o = await FeeDeployment.create(assets, { names: names() });
  let id = 0n;
  for (const [n, i] of [0, 3333, windows - 1].entries()) {
    if (n > 0) await o.newOracle();
    const w = sample(i);
    equal((await o.inputWords(N)).why?.name, 'NoSuchWindow', 'inputWords before the window is observed');
    for (const tick of w.ticks) check(!(await o.observeTick(tick, 'Keeper')).reverted, 'observe');
    equal(hex((await o.inputWords(N)).bytes), hex(pack(w.x0)), `window ${i}: the contract's packed input equals the page's bytes`);
    const honest = honestClaim(net, w.x0);
    const halved = claimOnHalvedInput(net, w.x0);
    const lie = lieAboutClass(net, w.x0, (honest.cls + 1) % 3);
    // The page's "right arithmetic, another window" claim: the true output of another window.
    const other = claimAboutOtherWindow(net, ticks, task, i, (honest.cls + 1) % 3);
    check(other !== null && other.other !== i, `window ${i}: another window answering tier ${(honest.cls + 1) % 3} is found`);
    equal(audit(net, other), null, 'audit of the other-window claim: nothing miscomputed');
    equal(other.x0, sample(other.other).x0, "the other-window claim carries that window's bytes");
    const [halvedId, honestId, lieId, otherId] = [id, id + 1n, id + 2n, id + 3n];
    id += 4n;
    for (const c of [halved, honest, lie, other]) check(!(await o.postClaim('Keeper', c)).reverted, 'claim posted');
    for (let k = 1; k <= net.length; k++) {
      equal((await o.challengeNeuron('Watcher', otherId, k, k - 1, other)).why?.name, 'NoFraud', `other window: layer ${k} holds`);
    }
    equal((await o.challengeClass('Watcher', otherId, other)).why?.name, 'NoFraud', 'other window: the tier holds');
    check(!(await o.challengeClass('Watcher', lieId, lie)).reverted, 'the lie is slashed');
    equal((await o.apply('Keeper', halvedId, N)).why?.name, 'NotThisWindow', 'open claim about another input');
    const early = await o.apply('Keeper', honestId, N);
    check(early.reverted, 'apply of an open claim reverts');
    equal((await o.result(honestId)).why?.name, 'NotFinal', 'the arbiter has no result for the open claim');
    check((await o.apply('Keeper', lieId, N)).reverted, 'apply of a slashed claim reverts');
    equal((await o.finalize('Keeper', honestId)).why?.name, 'TooEarly', 'finalize inside the window');
    // The dispute window is the gap: six more observations of the real series are one hour.
    if (i + N + task.gap < ticks.length) {
      for (let k = 1; k <= task.gap; k++) check(!(await o.observeTick(ticks[i + N + k], 'Keeper')).reverted, 'observe in the gap');
    } else {
      o.mine(BigInt(task.gap) * INTERVAL);
    }
    check(!(await o.finalize('Keeper', halvedId)).reverted, 'the halved-input claim finalises: its arithmetic is right');
    check(!(await o.finalize('Keeper', honestId)).reverted, 'the honest claim finalises after six intervals');
    equal((await o.finalize('Keeper', lieId)).why?.name, 'ClaimNotOpen', 'a slashed claim cannot be finalised');
    equal((await o.apply('Keeper', halvedId, N)).why?.name, 'NotThisWindow', 'final claim about another input is refused');
    check(!(await o.finalize('Keeper', otherId)).reverted, 'the other-window claim finalises: its arithmetic is right');
    equal((await o.result(otherId)).cls, other.cls, 'the arbiter answers for the other-window claim');
    equal((await o.apply('Keeper', otherId, N)).why?.name, 'NotThisWindow', 'final claim about another window is refused');
    equal(await o.fee(), DEFAULT_FEE, 'fee() unchanged by the refused claim');
    check((await o.apply('Keeper', lieId, N)).reverted, 'a slashed claim is never applied');
    equal((await o.apply('Keeper', honestId, N + 1)).why?.name, 'NotThisWindow', 'the honest claim applied to the next window');
    equal((await o.apply('Keeper', honestId, N + task.gap + 1)).why?.name, 'NoSuchWindow', 'a window not observed yet');
    const applied = await o.apply('Bystander', honestId, N);
    check(!applied.reverted, `the honest claim is applied, by anyone (${applied.why?.name})`);
    equal(applied.fee, FEES[honest.cls], 'apply returns the fee of the tier');
    equal(await o.fee(), FEES[honest.cls], `window ${i}: fee() is the fee of tier ${honest.cls}`);
    equal((await o.apply('Keeper', honestId, N)).why?.name, 'NotNewer', 'a second apply of the window');
    // After the gap and the horizon the fee falls back to the default.
    const moves = await o.moves();
    for (let k = moves; k < N + task.gap + task.horizon; k++) {
      check(!(await o.observeTick(ticks[Math.min(i + k + 1, ticks.length - 1)], 'Keeper')).reverted, 'observe in the horizon');
      if (k + 1 < N + task.gap + task.horizon) equal(await o.fee(), FEES[honest.cls], 'fee() in force during the horizon');
    }
    equal(await o.fee(), DEFAULT_FEE, 'fee() back to the default after the horizon');
  }
  equal(await o.creditOf('Keeper'), 9n * BOND, 'nine bonds returned to the keeper');
  equal(await o.burned(), 3n * (BOND - BOND / 2n), 'half of each of the three slashed bonds burned');
}

// ------------------------------------------------------------------------------------------------
section('deadlines and money');
{
  const m = await FeeDeployment.create(assets, { names: names() });
  const w = sample(1);
  const claim = honestClaim(net, w.x0);
  equal((await m.postClaim('Keeper', claim, BOND - 1n)).why?.name, 'BondTooSmall', 'short bond');
  equal((await m.postClaim('Keeper', { ...claim, x0: w.x0.map((v, k) => (k === 0 ? 128 : v)) })).why?.name, 'MalformedInput', 'an input byte above 127');
  equal((await m.postClaim('Keeper', { ...claim, cls: 3 })).why?.name, 'ClassOutOfRange', 'a tier the network does not have');
  check(!(await m.postClaim('Keeper', claim)).reverted, 'claim');
  equal((await m.finalize('Bystander', 0n)).why?.name, 'TooEarly', 'finalize before the deadline');
  equal(await m.chain.balance(m.arbiter), BOND, 'arbiter holds the bond');
  const lie = lieAboutClass(net, w.x0, (claim.cls + 1) % 3);
  check(!(await m.postClaim('Keeper', lie)).reverted, 'false claim');
  m.chain.advance(WINDOW + 1n);
  equal((await m.challengeClass('Watcher', 1n, lie)).why?.name, 'TooLate', 'challenge after the deadline');
  check(!(await m.finalize('Bystander', 0n)).reverted, 'anyone can finalize');
  equal(await m.burned(), 0n, 'finalising burns nothing');
  equal(await m.creditOf('Keeper'), BOND, 'bond credited to the keeper, not the caller');
  equal((await m.withdraw('Bystander')).why?.name, 'NothingToWithdraw', 'withdraw with no credit');
  const balance = await m.chain.balance(m.chain.accounts.Keeper.address);
  check(!(await m.withdraw('Keeper')).reverted, 'withdraw');
  equal(await m.chain.balance(m.chain.accounts.Keeper.address), balance + BOND, 'bond back in the account');
}

console.log(`\nfee demo tests: ${checks} checks, ${failures} failed, ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(failures === 0 ? 0 : 1);
