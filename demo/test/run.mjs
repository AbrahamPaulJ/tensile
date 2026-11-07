// Tests of the demo's engine in Node, against the repository's reference and measurements:
//   1. the chain in the page reproduces, gas unit for gas unit, the figures tools/m4_numbers.sh measured
//      on anvil (generated/numbers.json), so the page's numbers are the project's numbers;
//   2. the Fe bytecode on it agrees with the NumPy reference (classes and traces);
//   3. every lie the page can tell is caught by the challenge the walk names, and only by it;
//   4. honest claims survive every challenge, and the deadline rules hold.
// usage: node test/run.mjs [samples]       (default 40 samples for the differential)
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { BOND, Deployment, WINDOW, argmax } from '../src/protocol.js';
import { loadWasm } from '../src/wasm.js';

const here = new URL('.', import.meta.url);
const repo = new URL('../../', here);
const assets = JSON.parse(readFileSync(new URL('../generated/assets.json', here), 'utf8'));
const anvil = JSON.parse(readFileSync(new URL('../generated/numbers.json', here), 'utf8'));
const anvilGas = (name) => anvil.calls[name].fe;
const SAMPLES = Number(process.argv[2] ?? 40);
const N_IN = 784;

let failures = 0;
let checks = 0;
function check(condition, what) {
  checks++;
  if (!condition) {
    failures++;
    console.log(`FAIL  ${what}`);
  }
}
function equal(got, expected, what) {
  const same = JSON.stringify(got, (_, v) => (typeof v === 'bigint' ? v.toString() : v))
    === JSON.stringify(expected, (_, v) => (typeof v === 'bigint' ? v.toString() : v));
  check(same, `${what}: got ${JSON.stringify(got, (_, v) => (typeof v === 'bigint' ? v.toString() : v))}, expected ${JSON.stringify(expected, (_, v) => (typeof v === 'bigint' ? v.toString() : v))}`);
}

const inputs = readFileSync(new URL('ref/out/inputs.bin', repo));
const pred = readFileSync(new URL('ref/out/pred.bin', repo));
const traces = readFileSync(new URL('ref/out/trace.jsonl', repo), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const image = (i) => Array.from(inputs.subarray(i * N_IN, (i + 1) * N_IN));

// anvil's default account, the one tools/arbiter_gas.sh sends everything from.
const ANVIL = '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80';
const FUNDS = 1_000_000n * BOND;
const names = () => [
  { name: 'Deployer', key: ANVIL, balance: FUNDS },
  { name: 'Claimant', balance: FUNDS },
  { name: 'Watcher', balance: FUNDS },
  { name: 'Bystander', balance: FUNDS },
];

function section(title) {
  console.log(`--- ${title}`);
}

// ------------------------------------------------------------------------------------------------
section('deployment');
const started = performance.now();
const d = await Deployment.create(assets, { names: names() });
console.log(`deployed 5 contracts in ${(performance.now() - started).toFixed(0)} ms`);
equal(await d.modelRoot(), assets.model.root, 'model root read back');

// ------------------------------------------------------------------------------------------------
section('gas against anvil (tools/m4_numbers.sh, reference sample 0)');
const x0 = image(0);
const t0 = traces[0];
const honest0 = { hidden: t0.hidden[0], logits: t0.logits, cls: t0.class };
const me = 'Deployer';
const gas = [];
function measured(label, got, onAnvil) {
  gas.push([label, got, onAnvil]);
  equal(got, BigInt(onAnvil), `gas of ${label}`);
}
const classified = await d.classifyTx(x0);
equal(classified.cls, pred[0], 'classify of sample 0');
measured('Network.classify', classified.gas, anvil.classify);

const honestClaim = d.honestClaim(x0, honest0);
const r0 = await d.postClaim(me, honestClaim);
check(!r0.reverted, 'honest claim accepted');
equal(r0.events, ['Claimed'], 'claim event');
measured('claim', r0.gasUsed, anvilGas('claim'));
const flip = (v) => (v === 127 ? v - 1 : v + 1);
const badHidden = { ...honestClaim, hidden: honestClaim.hidden.map((v, i) => (i === 0 ? flip(v) : v)) };
const badLogit = { ...honestClaim, logits: honestClaim.logits.map((v, i) => (i === 0 ? v + 1 : v)) };
const badClass = { ...honestClaim, cls: (honestClaim.cls + 1) % 10 };
for (const bad of [badHidden, badLogit, badClass]) check(!(await d.postClaim(me, bad)).reverted, 'false claim accepted');
const c1 = await d.challengeNeuron1(me, 1n, 0, badHidden);
check(!c1.reverted, `layer 1 challenge succeeds (${c1.why?.name})`);
measured('challengeNeuron1', c1.gasUsed, anvilGas('challengeNeuron1 (784 inputs)'));
equal((await d.claimInfo(1n)).status, 2, 'slashed after layer 1 challenge');
const c2 = await d.challengeNeuron2(me, 2n, 0, badLogit);
check(!c2.reverted, `layer 2 challenge succeeds (${c2.why?.name})`);
measured('challengeNeuron2', c2.gasUsed, anvilGas('challengeNeuron2 (32 inputs)'));
const c3 = await d.challengeClass(me, 3n, badClass);
check(!c3.reverted, `class challenge succeeds (${c3.why?.name})`);
measured('challengeClass', c3.gasUsed, anvilGas('challengeClass'));
const refused = await d.challengeClass(me, 0n, honestClaim);
equal(refused.why?.name, 'NoFraud', 'class challenge of the honest claim');
d.chain.advance(WINDOW + 1n);
const f0 = await d.finalize(me, 0n);
check(!f0.reverted, 'finalize succeeds');
measured('finalize', f0.gasUsed, anvilGas('finalize'));
equal((await d.result(0n)).cls, honestClaim.cls, 'result of the final claim');
equal(await d.creditOf(me), 4n * BOND, 'credit of three slashed bonds and one returned');
const w0 = await d.withdraw(me);
check(!w0.reverted, 'withdraw succeeds');
measured('withdraw (4 bonds)', w0.gasUsed, anvilGas('withdraw (4 bonds)'));
equal(await d.chain.balance(d.arbiter), 0n, 'arbiter empty after withdraw');
for (const [label, got, onAnvil] of gas) console.log(`${label.padEnd(22)} page ${String(got).padStart(9)}   anvil ${String(onAnvil).padStart(9)}`);

// ------------------------------------------------------------------------------------------------
section(`differential against the NumPy reference, ${SAMPLES} samples`);
const diffStart = performance.now();
let mismatches = 0;
for (let i = 0; i < SAMPLES; i++) {
  const t = await d.trace(image(i));
  const same = JSON.stringify(t.hidden) === JSON.stringify(traces[i].hidden[0])
    && JSON.stringify(t.logits) === JSON.stringify(traces[i].logits)
    && t.cls === traces[i].class && t.cls === pred[i];
  if (!same) mismatches++;
}
equal(mismatches, 0, 'mismatches against the reference');
const diffMs = performance.now() - diffStart;
console.log(`${SAMPLES} traces, ${mismatches} mismatches, ${(diffMs / SAMPLES).toFixed(0)} ms per forward pass`);

// ------------------------------------------------------------------------------------------------
section('the WebAssembly build of the same kernel (wasm/out), against the EVM and the reference');
{
  const wasmBytes = readFileSync(new URL('wasm/out/tensile_wasm.wasm', repo));
  const manifest = JSON.parse(readFileSync(new URL('wasm/out/tensile_wasm.manifest.json', repo), 'utf8'));
  equal(createHash('sha256').update(wasmBytes).digest('hex'), manifest.module.sha256, 'wasm module against its manifest');
  const w = await loadWasm(wasmBytes);
  let wasmMismatches = 0;
  for (let i = 0; i < SAMPLES; i++) {
    const evm = await d.trace(image(i));
    const got = w.run(image(i));
    const same = JSON.stringify(got.hidden) === JSON.stringify(evm.hidden)
      && JSON.stringify(got.hidden) === JSON.stringify(traces[i].hidden[0])
      && JSON.stringify(got.logits) === JSON.stringify(evm.logits)
      && JSON.stringify(got.logits) === JSON.stringify(traces[i].logits)
      && got.cls === evm.cls && got.cls === pred[i];
    if (!same) wasmMismatches++;
  }
  equal(wasmMismatches, 0, 'wasm against the EVM and the reference');
  let classMismatches = 0;
  for (let i = 0; i < pred.length; i++) if (w.run(image(i)).cls !== pred[i]) classMismatches++;
  equal(classMismatches, 0, 'wasm classes on every reference image');
  console.log(`${SAMPLES} samples equal on the EVM, in wasm and in NumPy; ${pred.length} wasm classes, ${classMismatches} mismatches`);
}

// ------------------------------------------------------------------------------------------------
section('every lie, on every embedded sample');
const fresh = await Deployment.create(assets, { names: names() });
let id = 0n;
let lies = 0;
const lieGas = { class: [], logit: [], hidden: [] };
for (const sample of assets.samples) {
  const x = Array.from(Buffer.from(sample.x0, 'hex'));
  const honest = await fresh.trace(x);
  equal(honest.cls, sample.pred, `class of sample ${sample.i}`);
  const target = (honest.cls + 3) % 10;

  // The honest claim: the walk finds nothing, and no challenge of any kind works.
  const hc = fresh.honestClaim(x, honest);
  equal(fresh.audit(hc, honest), null, 'audit of an honest claim');
  check(!(await fresh.postClaim('Claimant', hc)).reverted, 'honest claim posted');
  const honestId = id++;
  equal((await fresh.challengeClass('Watcher', honestId, hc)).why?.name, 'NoFraud', 'class challenge of honest');
  equal((await fresh.challengeNeuron1('Watcher', honestId, sample.i % 32, hc)).why?.name, 'NoFraud', 'layer 1 challenge of honest');
  equal((await fresh.challengeNeuron2('Watcher', honestId, sample.i % 10, hc)).why?.name, 'NoFraud', 'layer 2 challenge of honest');

  const lying = [
    ['class', fresh.lieAboutClass(x, honest, target)],
    ['logit', fresh.lieAboutLogit(x, honest, target)],
    ['hidden', await fresh.lieAboutHidden(x, honest, target)],
  ];
  for (const [kind, claim] of lying) {
    if (claim === null) {
      check(false, `no forgery found for sample ${sample.i}`);
      continue;
    }
    lies++;
    equal(claim.cls, target, `${kind} lie claims the target`);
    const finding = fresh.audit(claim, honest);
    check(finding !== null, `${kind} lie is found`);
    equal(finding.kind === 'class' ? 'class' : finding.layer === 1 ? 'hidden' : 'logit', kind, `${kind} lie is named at its layer`);
    const post = await fresh.postClaim('Claimant', claim);
    check(!post.reverted, `${kind} lie posted (${post.why?.name})`);
    const lieId = id++;
    if (kind === 'hidden') {
      // The strongest lie stands up to layer 2 and the argmax: the logits are true of the forgery.
      equal((await fresh.challengeClass('Watcher', lieId, claim)).why?.name, 'NoFraud', 'forged hidden: class check holds');
      for (let j = 0; j < 10; j++) {
        equal((await fresh.challengeNeuron2('Watcher', lieId, j, claim)).why?.name, 'NoFraud', `forged hidden: logit ${j} holds`);
      }
      const clean = claim.hidden.findIndex((v, j) => v === honest.hidden[j]);
      if (clean >= 0) equal((await fresh.challengeNeuron1('Watcher', lieId, clean, claim)).why?.name, 'NoFraud', 'forged hidden: untouched neuron holds');
    }
    const before = await fresh.creditOf('Watcher');
    const hit = await fresh.challenge('Watcher', lieId, claim, finding);
    check(!hit.reverted, `${kind} lie slashed (${hit.why?.name})`);
    lieGas[kind].push(hit.gasUsed);
    equal((await fresh.claimInfo(lieId)).status, 2, `${kind} lie status`);
    equal(await fresh.creditOf('Watcher'), before + BOND, `${kind} lie pays the watcher`);
    equal((await fresh.challenge('Bystander', lieId, claim, finding)).why?.name, 'ClaimNotOpen', 'a second challenge');
    equal((await fresh.result(lieId)).why?.name, 'NotFinal', 'a slashed claim has no result');
  }
  equal(argmax(honest.logits), honest.cls, 'trace class is the argmax');
}
console.log(`${lies} lies caught on ${assets.samples.length} samples`);
for (const [kind, gases] of Object.entries(lieGas)) {
  const sorted = gases.map(Number).sort((a, b) => a - b);
  console.log(`  ${kind.padEnd(7)} challenge gas ${sorted[0]} to ${sorted[sorted.length - 1]}`);
}

// ------------------------------------------------------------------------------------------------
section('deadlines and money');
const m = await Deployment.create(assets, { names: names() });
const hx = image(1);
const hh = await m.trace(hx);
const claim = m.honestClaim(hx, hh);
equal((await m.postClaim('Claimant', claim, BOND - 1n)).why?.name, 'BondTooSmall', 'short bond');
check(!(await m.postClaim('Claimant', claim)).reverted, 'claim');
equal((await m.finalize('Bystander', 0n)).why?.name, 'TooEarly', 'finalize before the deadline');
equal((await m.result(0n)).why?.name, 'NotFinal', 'an open claim has no result');
equal(await m.chain.balance(m.arbiter), BOND, 'arbiter holds the bond');
const lie = m.lieAboutClass(hx, hh, (hh.cls + 1) % 10);
check(!(await m.postClaim('Claimant', lie)).reverted, 'false claim');
m.chain.advance(WINDOW + 1n);
equal((await m.challengeClass('Watcher', 1n, lie)).why?.name, 'TooLate', 'challenge after the deadline');
check(!(await m.finalize('Bystander', 0n)).reverted, 'anyone can finalize');
equal(await m.creditOf('Claimant'), BOND, 'bond credited to the claimant, not the caller');
equal(await m.creditOf('Bystander'), 0n, 'nothing credited to the caller');
equal((await m.withdraw('Bystander')).why?.name, 'NothingToWithdraw', 'withdraw with no credit');
const balance = await m.chain.balance(m.chain.accounts.Claimant.address);
check(!(await m.withdraw('Claimant')).reverted, 'withdraw');
equal(await m.chain.balance(m.chain.accounts.Claimant.address), balance + BOND, 'bond back in the account');

console.log(`\ndemo tests: ${checks} checks, ${failures} failed, ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(failures === 0 ? 0 : 1);
