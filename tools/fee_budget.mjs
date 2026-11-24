// Reads the running budget of the live fee-tier system from Sepolia and writes it to
// deploy/sepolia-fee-running.json, which tools/sepolia_fee_page.py turns into the budget section of
// docs/sepolia-fee.md: the keeper's balance and nonce, the arbiter's credit owed to it, the bond, the
// base fee and the node's gas price, at one block. Read-only, no key.
// usage: node tools/fee_budget.mjs      env: SEPOLIA_RPC (comma-separated)
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Node, SEPOLIA_RPC, SIG, readUint, word } from '../demo/src/tensile-fee-live-core.js';

const at = (path) => fileURLToPath(new URL(path, import.meta.url));
const record = readFileSync(at('../deploy/sepolia-fee.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const by = (label) => record.find((r) => r.label === label);
const keeper = by('TickRelay').keeper;
const adversary = by('fund adversary')?.to ?? null;
const arbiter = by('Arbiter').address;
const node = new Node(process.env.SEPOLIA_RPC ? process.env.SEPOLIA_RPC.split(',') : SEPOLIA_RPC);
const head = await node.value('eth_getBlockByNumber', ['latest', false]);
const block = head.number;
const big = async (method, params) => BigInt(await node.value(method, params));
const out = {
  measured: new Date(parseInt(head.timestamp, 16) * 1000).toISOString().replace('.000', ''),
  block: parseInt(block, 16),
  keeper,
  keeper_balance_wei: String(await big('eth_getBalance', [keeper, block])),
  keeper_nonce: Number(await big('eth_getTransactionCount', [keeper, block])),
  keeper_credit_wei: String(readUint(await node.view(arbiter, SIG.creditOf, [word(keeper)]))),
  adversary,
  adversary_balance_wei: adversary ? String(await big('eth_getBalance', [adversary, block])) : null,
  bond_wei: String(readUint(await node.view(arbiter, SIG.bond))),
  base_fee_wei: Number(BigInt(head.baseFeePerGas)),
  gas_price_wei: Number(await big('eth_gasPrice', [])),
};
writeFileSync(at('../deploy/sepolia-fee-running.json'), JSON.stringify(out, null, 1) + '\n');
console.log(JSON.stringify(out));
