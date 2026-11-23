// Differential test of the WebAssembly build against the NumPy reference.
//
//   node wasm/diff.mjs [wasm/out/tensile_wasm.wasm]
//
// Every test image in ref/out/inputs.bin is classified by the module and compared with
// ref/out/pred.bin; for the first 1,000 (ref/out/trace.jsonl) all 32 hidden activations and all ten
// logits are compared too. Then the three refusals: an index outside the layer and a pixel above 127
// must end the call through the import `__fe_revert`, which this host turns into an exception.
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

// The module's imports: `pixel`, and `__fe_revert(address, length)`, through which a failed `assert!`
// or any other panic hands over its Solidity `Panic(uint256)` data instead of returning.
class Revert extends Error {}
const instantiate = async (pixel) => {
  let memory = null;
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {
      pixel,
      __fe_revert: (address, length) => {
        const data = new Uint8Array(memory.buffer, address >>> 0, length >>> 0);
        throw new Revert(Buffer.from(data).toString("hex"));
      },
    },
  });
  memory = instance.exports.memory;
  return instance.exports;
};

let image = 0;
const { classify, hidden, logit } = await instantiate((i) => inputs[image * 784 + i]);

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
// Refusals. A call that reverts leaves the instance in an unknown state, so each one gets its own.
const PANIC_ASSERT = "4e487b71" + "01".padStart(64, "0");
const refusals = [
  ["hidden(32)", (m) => m.hidden(32), () => 0],
  ["logit(10)", (m) => m.logit(10), () => 0],
  ["a pixel of 128", (m) => m.classify(), (i) => (i === 783 ? 128 : 0)],
];
let notRefused = 0;
for (const [what, call, pixel] of refusals) {
  let data = null;
  try {
    call(await instantiate(pixel));
  } catch (error) {
    if (!(error instanceof Revert)) throw error;
    data = error.message;
  }
  if (data !== PANIC_ASSERT) {
    notRefused++;
    console.log(`NOT REFUSED as expected: ${what} gave ${data === null ? "a result" : "revert data " + data}`);
  }
}
console.log(`module ${path.basename(modulePath)}: ${bytes.length} bytes, ${manifest.compiler}, -O ${manifest.optimization}`);
console.log(`classes: ${n - classMismatches} of ${n} equal the reference (${classMismatches} mismatches), ${classifySeconds.toFixed(2)} s, ${(1e6 * classifySeconds / n).toFixed(0)} us per image`);
console.log(`logits:  ${trace.length * 10 - logitMismatches} of ${trace.length * 10} equal the reference (${logitMismatches} mismatches)`);
console.log(`hidden:  ${trace.length * 32 - hiddenMismatches} of ${trace.length * 32} equal the reference (${hiddenMismatches} mismatches)`);
console.log(`refusals: ${refusals.length - notRefused} of ${refusals.length} calls outside the model's range revert with Panic(1)`);
process.exit(classMismatches || logitMismatches || hiddenMismatches || notRefused ? 1 : 0);
