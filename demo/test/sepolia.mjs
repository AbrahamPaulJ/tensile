// The watcher against the live arbiter on Sepolia (read-only, needs the network, no key):
//   for every claim recorded in deploy/sepolia-open.jsonl, read it back from the chain, re-run the
//   network with the committed WebAssembly module, and put the challenge to a public node.
//   - a claim posted honest must audit clean, and the arbiter must refuse a challenge (NoFraud);
//   - a claim posted false must be named at the value that was changed, and the arbiter must accept
//     the challenge, unless someone has already sent it (then the claim's status is slashed);
//   - `Network.classify` on Sepolia must give the WebAssembly module's class for every input.
// usage: node test/sepolia.mjs [rpc-url]
import { readFileSync } from 'node:fs';
import { Node, STATUS, Watcher } from '../src/sepolia-core.js';
import { loadWasm } from '../src/wasm.js';

const at = (path) => new URL(path, import.meta.url);
const lines = (path) => readFileSync(at(path), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const open = lines('../../deploy/sepolia-open.jsonl');
const first = lines('../../deploy/sepolia.jsonl');
const arbiter = open.find((r) => r.label === 'Arbiter');
const posted = open.filter((r) => r.label.startsWith('claim '));
const config = {
  arbiter: arbiter.address,
  network: first.find((r) => r.label === 'Network').address,
  fromBlock: arbiter.block,
  claims: posted.map((r) => ({ id: Number(r.id), hash: r.hash })),
};
const model = JSON.parse(readFileSync(at('../generated/assets.json'), 'utf8')).model;
const wasm = await loadWasm(readFileSync(at('../../wasm/out/tensile_wasm.wasm')));
const urls = process.argv[2] ? [process.argv[2]] : ['https://ethereum-sepolia-rpc.publicnode.com', 'https://sepolia.gateway.tenderly.co'];
const watcher = new Watcher(new Node(urls), config, model, (x0) => wasm.run(x0));

const started = Date.now();
const summary = await watcher.summary();
let failures = 0;
const fail = (text) => {
  failures++;
  console.log(`WRONG: ${text}`);
};
if (summary.root !== model.root) fail(`the arbiter's model root is ${summary.root}, the model's is ${model.root}`);
if (summary.count < posted.length) fail(`the arbiter has ${summary.count} claims, ${posted.length} were recorded`);
console.log(`Arbiter ${config.arbiter} on Sepolia: ${summary.count} claims, bond ${summary.bond} wei, root as the model's`);

const kinds = { hidden: 'layer 1', logit: 'layer 2', class: 'class' };
for (const row of posted) {
  const id = Number(row.id);
  const [info, claim] = [await watcher.info(id), await watcher.claim(id)];
  const { honest, finding } = watcher.audit(claim);
  const onChain = await watcher.classifyOnChain(claim.x0);
  if (onChain !== honest.cls) fail(`claim ${id}: Network.classify on Sepolia says ${onChain}, the WebAssembly module ${honest.cls}`);
  const answer = await watcher.ask(watcher.challengeData(id, claim, finding));
  const found = finding === null ? 'nothing wrong' : finding.kind === 'class' ? 'class' : `layer ${finding.layer} neuron ${finding.j}`;
  let verdict;
  if (row.posted === 'honest') {
    if (finding !== null) fail(`claim ${id} was posted honest and the audit names ${found}`);
    if (answer.accepted || answer.why.name !== 'NoFraud') fail(`claim ${id} is honest and a challenge was not refused with NoFraud`);
    verdict = `refused: ${answer.why?.name}`;
  } else {
    const [what, j] = row.posted.split(':');
    const expected = what === 'class' ? 'class' : `${kinds[what]} neuron ${j}`;
    if (found !== expected) fail(`claim ${id} was posted false at ${expected} and the audit names ${found}`);
    if (info.status === 2) verdict = 'already slashed by someone';
    else if (answer.accepted) verdict = 'the arbiter would accept this challenge';
    else {
      fail(`claim ${id} is false and open, and the challenge is refused: ${answer.why.name}`);
      verdict = `refused: ${answer.why.name}`;
    }
  }
  console.log(`claim ${String(id).padStart(2)} (${STATUS[info.status]}): posted ${row.posted.padEnd(9)} audit: ${found.padEnd(18)} Sepolia: ${verdict}`);
}
console.log(`sepolia watcher: ${failures} wrong result(s), ${((Date.now() - started) / 1000).toFixed(1)} s`);
process.exit(failures ? 1 : 0);
