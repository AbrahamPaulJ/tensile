// Prints the calldata of the challenge the watcher names for one claim on the Sepolia arbiter, for
// `cast send <arbiter> <calldata>`. Read-only. usage: node test/sepolia_calldata.mjs <claim id>
import { readFileSync } from 'node:fs';
import { Node, Watcher, to0x } from '../src/sepolia-core.js';
import { loadWasm } from '../src/wasm.js';

const at = (path) => new URL(path, import.meta.url);
const lines = (path) => readFileSync(at(path), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const open = lines('../../deploy/sepolia-open.jsonl');
const arbiter = open.find((r) => r.label === 'Arbiter');
const config = {
  arbiter: arbiter.address,
  network: lines('../../deploy/sepolia.jsonl').find((r) => r.label === 'Network').address,
  fromBlock: arbiter.block,
  claims: open.filter((r) => r.label.startsWith('claim ')).map((r) => ({ id: Number(r.id), hash: r.hash })),
};
const model = JSON.parse(readFileSync(at('../generated/assets.json'), 'utf8')).model;
const wasm = await loadWasm(readFileSync(at('../../wasm/out/tensile_wasm.wasm')));
const watcher = new Watcher(new Node(['https://ethereum-sepolia-rpc.publicnode.com']), config, model, (x0) => wasm.run(x0));
const id = Number(process.argv[2]);
const claim = await watcher.claim(id);
const { finding } = watcher.audit(claim);
if (finding === null) {
  console.error(`claim ${id} is honest: there is nothing to challenge`);
  process.exit(1);
}
console.log(to0x(watcher.challengeData(id, claim, finding)));
