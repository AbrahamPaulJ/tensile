// The small part of the Solidity ABI this demo needs, written out so that nothing is hidden behind a
// library: every argument of the arbiter is a word, a fixed array of words, or one bytes32[].
import { keccak_256 } from '@noble/hashes/sha3.js';

const encoder = new TextEncoder();

export const hex = (bytes) =>
  Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');

export function fromHex(text) {
  const clean = text.startsWith('0x') ? text.slice(2) : text;
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(clean.substr(i * 2, 2), 16);
  return out;
}

export const keccak = (bytes) => keccak_256(bytes);
export const selector = (signature) => keccak_256(encoder.encode(signature)).slice(0, 4);

export function concat(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** A 32-byte word: an integer, two's complement if negative. */
export function word(value) {
  let v = BigInt(value);
  if (v < 0n) v += 1n << 256n;
  return fromHex(v.toString(16).padStart(64, '0'));
}

export function readUint(bytes, at = 0) {
  return BigInt('0x' + hex(bytes.subarray(at, at + 32)));
}

export function readInt(bytes, at = 0) {
  const v = readUint(bytes, at);
  return v >= 1n << 255n ? v - (1n << 256n) : v;
}

/** Int8 values as packed words (docs/spec.md section 7): padded with zeros to a multiple of 32. */
export function pack(values) {
  const out = new Uint8Array(Math.ceil(values.length / 32) * 32);
  for (let i = 0; i < values.length; i++) out[i] = values[i] & 0xff;
  return out;
}

/** selector || static head || one trailing bytes32[] if given. */
export function encodeCall(signature, parts, dynamicWords = null) {
  const head = concat(parts);
  if (dynamicWords === null) return concat([selector(signature), head]);
  const offset = word(head.length + 32);
  const length = word(dynamicWords.length);
  return concat([selector(signature), head, offset, length, ...dynamicWords.map((w) => fromHex(w))]);
}

export const signatureSelector = (signature) => hex(selector(signature));
