// Builds demo/dist/tensile-fee-live.html: the live status page of the fee-tier system on Sepolia,
// one self-contained file. It embeds the integer model (examples/amm-fee/out/model.json: the page
// re-runs it and rebuilds its Merkle tree for the challenge calldata) and the addresses recorded in
// deploy/sepolia-fee.jsonl. It needs the network: it reads Sepolia and mainnet through public nodes.
// usage: node tools/build_fee_live.mjs      (after `npm ci` in demo/; esbuild comes from there)
import { createRequire } from 'node:module';
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { MAINNET_RPC, SEPOLIA_RPC } from '../demo/src/tensile-fee-live-core.js';

const require = createRequire(new URL('../demo/package.json', import.meta.url));
const { build } = require('esbuild');
const at = (path) => fileURLToPath(new URL(path, import.meta.url));
const result = await build({
  entryPoints: [at('../demo/src/tensile-fee-live.js')],
  bundle: true,
  minify: true,
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  write: false,
  legalComments: 'none',
  logLevel: 'warning',
  nodePaths: [at('../demo/node_modules')],
});
const inline = (text) => text.replace(/<\/(script)/gi, '<\\/$1');
const record = readFileSync(at('../deploy/sepolia-fee.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const by = (label) => record.find((r) => r.label === label);
const model = JSON.parse(readFileSync(at('../examples/amm-fee/out/model.json'), 'utf8'));
const data = {
  config: {
    arbiter: by('Arbiter').address,
    relay: by('TickRelay').address,
    oracle: by('FeeOracle').address,
    network: by('Network').address,
    fromBlock: Math.min(...record.filter((r) => r.block).map((r) => r.block)),
    keeper: by('TickRelay').keeper,
    adversaries: [...new Set(record.filter((r) => r.label.startsWith('planted false claim')).map((r) => r.claimant))],
  },
  // Ticks relayed at the start, before the first claim (read after the fact from mainnet).
  backfill: record.filter((r) => r.label.startsWith('relay ticks') && r.block <= record.find((c) => c.label.startsWith('claim window')).block)
    .reduce((n, r) => n + Number(r.count), 0),
  sepoliaRpc: SEPOLIA_RPC,
  mainnetRpc: MAINNET_RPC,
  model: { name: model.name, layers: model.layers.map(({ weights, bias, shift }) => ({ weights, bias, shift })) },
};
const page = readFileSync(at('../demo/src/tensile-fee-live.html'), 'utf8')
  .replace('/*ASSETS*/null', () => inline(JSON.stringify(data)))
  .replace('/*SCRIPT*/', () => inline(result.outputFiles[0].text));
mkdirSync(at('../demo/dist/'), { recursive: true });
const target = at('../demo/dist/tensile-fee-live.html');
writeFileSync(target, page);
console.log(`demo/dist/tensile-fee-live.html: ${statSync(target).size} bytes`);
