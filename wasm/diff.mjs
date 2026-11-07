// Differential test of the WebAssembly build against the NumPy reference.
//
//   node wasm/diff.mjs [wasm/out/tensile_wasm.wasm]
//
// Every test image in ref/out/inputs.bin is classified by the module and compared with
// ref/out/pred.bin; for the first 1,000 (ref/out/trace.jsonl) all 32 hidden activations and all ten
// logits are compared too.
// The module is checked against its manifest (size and SHA-256) first. No dependencies.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modulePath = process.argv[2] || path.join(root, "wasm/out/tensile_wasm.wasm");
const manifest = JSON.parse(fs.readFileSync(modulePath.replace(/\.wasm$/, ".manifest.json"), "utf8"));
const bytes = fs.readFileSync(modulePath);
const sha = crypto.createHash("sha256").update(bytes).digest("hex");
if (sha !== manifest.module.sha256 || bytes.length !== manifest.module.size) {
  throw new Error(`module does not match its manifest: sha256 ${sha}, ${bytes.length} bytes`);
}

const inputs = fs.readFileSync(path.join(root, "ref/out/inputs.bin"));
const pred = fs.readFileSync(path.join(root, "ref/out/pred.bin"));
const trace = fs.readFileSync(path.join(root, "ref/out/trace.jsonl"), "utf8").trim().split("\n").map(JSON.parse);
const n = pred.length;
if (inputs.length !== n * 784) throw new Error("inputs.bin and pred.bin disagree on the sample count");

let image = 0;
const { instance } = await WebAssembly.instantiate(bytes, {
  env: { pixel: (i) => inputs[image * 784 + i] },
});
const { classify, hidden, logit } = instance.exports;

let classMismatches = 0;
let logitMismatches = 0;
let hiddenMismatches = 0;
const start = process.hrtime.bigint();
for (image = 0; image < n; image++) {
  if (classify() !== pred[image]) classMismatches++;
}
const classifySeconds = Number(process.hrtime.bigint() - start) / 1e9;
for (const t of trace) {
  image = t.i;
  for (let j = 0; j < 10; j++) {
    if (logit(j) !== t.logits[j]) logitMismatches++;
  }
  for (let j = 0; j < 32; j++) {
    if (hidden(j) !== t.hidden[0][j]) hiddenMismatches++;
  }
}
console.log(`module ${path.basename(modulePath)}: ${bytes.length} bytes, ${manifest.compiler}, -O ${manifest.optimization}`);
console.log(`classes: ${n - classMismatches} of ${n} equal the reference (${classMismatches} mismatches), ${classifySeconds.toFixed(2)} s, ${(1e6 * classifySeconds / n).toFixed(0)} us per image`);
console.log(`logits:  ${trace.length * 10 - logitMismatches} of ${trace.length * 10} equal the reference (${logitMismatches} mismatches)`);
console.log(`hidden:  ${trace.length * 32 - hiddenMismatches} of ${trace.length * 32} equal the reference (${hiddenMismatches} mismatches)`);
process.exit(classMismatches || logitMismatches || hiddenMismatches ? 1 : 0);
