// The WebAssembly build of the same Fe kernel and weights (the ingot `wasm/` on top of
// `ingots/tensile` and `ingots/model`, built by `fe build --backend wasm`). The module exports
// `classify()`, `hidden(j)` and `logit(j)` and imports `env.pixel(i)`, which the page answers from
// the image being run, and `env.__fe_revert(address, length)`, which it calls instead of returning
// when a check fails (a pixel above 127, an index outside a layer). The page never sends either.

/** Instantiates the module from its bytes and returns `run(x0)`. */
export async function loadWasm(bytes) {
  let current = null;
  let memory = null;
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {
      pixel: (i) => current[i],
      __fe_revert: (address, length) => {
        const data = new Uint8Array(memory.buffer, address >>> 0, length >>> 0);
        throw new Error(`the WebAssembly kernel refused the call: 0x${[...data].map((b) => b.toString(16).padStart(2, '0')).join('')}`);
      },
    },
  });
  memory = instance.exports.memory;
  const { classify, hidden: hiddenAt, logit } = instance.exports;
  return {
    /** The class, the 32 hidden activations and the ten logits of `x0` (784 values in 0..127), and the time taken. */
    run(x0) {
      current = x0;
      const started = performance.now();
      const cls = classify();
      const logits = [];
      for (let j = 0; j < 10; j++) logits.push(logit(j));
      const ms = performance.now() - started;
      const hidden = [];
      for (let j = 0; j < 32; j++) hidden.push(hiddenAt(j));
      current = null;
      return { cls, hidden, logits, ms };
    },
  };
}

/** Base64 to bytes, in the browser and in Node. */
export function fromBase64(text) {
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(text, 'base64'));
  const raw = atob(text);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}
