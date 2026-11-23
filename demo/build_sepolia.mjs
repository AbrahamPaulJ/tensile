// Builds dist/tensile-sepolia.html: the watcher page for the arbiter on Sepolia, one self-contained
// file. It embeds the model's neurons with their Merkle proofs (generated/assets.json), the WebAssembly
// build of the kernel, and the addresses and claim transactions recorded in ../deploy/. Unlike the
// demo it needs the network: it reads Sepolia through public nodes. usage: node build_sepolia.mjs
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';

const at = (path) => new URL(path, import.meta.url);
const result = await build({
  entryPoints: [at('src/sepolia.js').pathname.replace(/^\/([A-Za-z]:)/, '$1')],
  bundle: true,
  minify: true,
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  write: false,
  legalComments: 'none',
  logLevel: 'warning',
});
const inline = (text) => text.replace(/<\/(script)/gi, '<\\/$1');
const lines = (path) => readFileSync(at(path), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const open = lines('../deploy/sepolia-open.jsonl');
const arbiter = open.find((r) => r.label === 'Arbiter');
const manifest = JSON.parse(readFileSync(at('../wasm/out/tensile_wasm.manifest.json'), 'utf8'));
const wasm = readFileSync(at('../wasm/out/tensile_wasm.wasm'));
if (createHash('sha256').update(wasm).digest('hex') !== manifest.module.sha256) throw new Error('wasm/out/tensile_wasm.wasm does not match its manifest');
const data = {
  config: {
    arbiter: arbiter.address,
    network: lines('../deploy/sepolia.jsonl').find((r) => r.label === 'Network').address,
    fromBlock: arbiter.block,
    claims: open.filter((r) => r.label.startsWith('claim ')).map((r) => ({ id: Number(r.id), hash: r.hash })),
    rpc: ['https://ethereum-sepolia-rpc.publicnode.com', 'https://sepolia.gateway.tenderly.co', 'https://rpc.sepolia.ethpandaops.io'],
  },
  model: JSON.parse(readFileSync(at('generated/assets.json'), 'utf8')).model,
  wasm: { base64: wasm.toString('base64'), sha256: manifest.module.sha256 },
};
const page = readFileSync(at('src/sepolia.html'), 'utf8')
  .replace('/*ASSETS*/null', () => inline(JSON.stringify(data)))
  .replace('/*SCRIPT*/', () => inline(result.outputFiles[0].text));
mkdirSync(at('dist/'), { recursive: true });
const target = at('dist/tensile-sepolia.html');
writeFileSync(target, page);
console.log(`dist/tensile-sepolia.html: ${statSync(target).size} bytes`);
