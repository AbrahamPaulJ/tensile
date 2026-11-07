# Compiler defects found while building Tensile

**Status, 2026-11-07.** Every example below was re-run on Fe master `89d07b9` (`tools/upstream_check.sh` with
`FE=` set to that build). Three are already fixed there, so they were not filed; the other four were filed, as
five issues:

| # | Defect | On master `89d07b9` | Issue |
|---|---|---|---|
| 1 | Stack overflow, one large array in two messages | fixed (passes in 14.5 s) | not filed |
| 2 | Minutes of compile time for large message arrays | fixed (1.9 s, was about 100 s) | not filed |
| 3 | Native `read_word`/`write_word` little-endian | fixed (big-endian on EVM, native and wasm; `tests/defects/native_word_endianness_master.fe`) | not filed |
| 4 | `black_box` on an array passes `fe check` | still present | [#1950](https://github.com/argotorg/fe/issues/1950) |
| 5 | No warning past 24,576 bytes | still present | [#1951](https://github.com/argotorg/fe/issues/1951) |
| 6 | Internal error, array stored into a nested array | still present | [#1952](https://github.com/argotorg/fe/issues/1952) |
| 7 | Misleading diagnostics: event field; name clash | both still present | [#1953](https://github.com/argotorg/fe/issues/1953), [#1954](https://github.com/argotorg/fe/issues/1954) |

The sections below are the original drafts, reproduced on `fe 26.5.1 (149c758)` on 2026-11-06.

Withdrawn from the earlier list: "`fe test --backend native` fails on an ingot that contains a contract".
A minimal file with a contract and a pure test passes on the native backend. The real limits are narrower:
a native test cannot deploy a contract or use `should_revert` or `balance`, and the second of those has a
clear message (`native test ... cannot use EVM should_revert or balance attributes`). Not a defect.

---

## 1. EVM stack overflow at run time when one large array is used in two messages

**Severity:** high. It compiles without a diagnostic and halts at run time.

**Example:** `spike/inference/q6_array_twice.fe`, about 30 lines, standalone.

```
$ fe test -O 1 q6_array_twice.fe
FAIL  same_array_in_two_calls
    Test halted: StackOverflow (gas: 1000000000)
```

**Expected:** the test passes, as it does with the array used in one message, or a compile error if this
form cannot be supported.

**Cause, from an opcode trace:** the array copy is emitted as straight-line code, one
`MLOAD, DUP1, PUSH3 addr, MSTORE` per element, and every loaded element stays on the EVM stack for the
second copy, so the stack climbs by one per element until it passes 1,024:

```
pc=27572 op=0x51 stack=1022 ...
pc=27578 op=0x52 stack=1024 ...
trace result: Ok(Halt { reason: StackOverflow, gas_used: 1000000000 })
```

**Workaround:** pack the data into words, and never pass a large array by value twice in one function.

---

## 2. Compile time of minutes for message arrays past about 1,000 elements

**Related:** [#835](https://github.com/argotorg/fe/issues/835) "Compilation time increases too much with
array's size", open. Add this as a data point there rather than a new issue.

**Example:** `spike/inference/q6_array_copy_stack.fe`, 37 lines, arrays of 600 and 1,100 `u8` in two
messages: 104.7 s and 115.3 s on the two runs at `-O 1` (i5-13420H, WSL2). `q6_array_twice.fe`, one
784-element array: 42 to 46 s. The spike's run at `-O 0` had not finished at 240 s.

---

## 3. Native `read_word` and `write_word` are little-endian, though documented big-endian

**Severity:** high for shared code: it is silent and differs between backends.

**Example:** `tests/defects/native_word_endianness.fe`.

```
$ fe test native_word_endianness.fe
PASS  read_word_is_big_endian
PASS  write_word_is_big_endian
$ fe test --backend native native_word_endianness.fe
FAIL  write_word_is_big_endian    native test failed (signal: 4 (SIGILL) (core dumped))
FAIL  read_word_is_big_endian     native test failed (signal: 4 (SIGILL) (core dumped))
```

**Expected:** both backends do what `core/src/ptr.fe` documents: "as a big-endian word", "Writes `value`
big-endian". Effect in practice: hashes read out of trie nodes came out byte-reversed natively, so a valid
proof was rejected, and finding it took a debug binary because native failures carry no message.

---

## 4. `black_box` on an array is accepted by `fe check` and fails only in code generation

**Example:** `spike/inference/q6_black_box.fe`.

```
$ fe check q6_black_box.fe        (exits 0, no output)
$ fe test q6_black_box.fe
Failed to emit test Sonatina: MIR lowering failed: unsupported while lowering `t`:
`core::hint::black_box` requires a scalar argument, such as an integer or `bool`, but found `[i8; 4]`
```

**Expected:** the type checker refuses it, with a source location. The message itself is good and has
none. Either a trait bound on `black_box` or support for arrays would do.

---

## 5. No warning when a contract exceeds the 24,576-byte limit

**Example:** `spike/inference/q6_const_size/`. Two of its contracts come out at 32,867 and 32,848 bytes of
runtime code. `fe build` says nothing about size, and `fe test` runs with a 1 MiB limit, so the tests pass.
The first sign of the problem is a failed deployment.

**Expected:** a warning from `fe build` past EIP-170's 24,576 bytes (and perhaps EIP-3860's initcode limit).
This project works around it with `tools/check_size.sh`.

---

## 6. Internal error: storing a whole array into a nested array with a computed inner length

**Example:** `tests/defects/store_array_row.fe`.

```
error[16-0004]: internal access checking error in `fn double_rows`
   │ pub const fn double_rows<const OUT: usize, const N: usize>() -> [[u256; half(N)]; OUT] {
   ...
   │ normalized body verification failed: StoreType { value: NValueId(10), destination: Root(NRootId(0)) }
```

**Expected:** this compiles, or gets an ordinary diagnostic. "Internal" errors should not reach a user.

---

## 7. Two misleading diagnostics

**Related:** [#1624](https://github.com/argotorg/fe/issues/1624) "Report all `#[error]` and `#[event]` field
problems in record field analysis" touches the first.

**Event field.** `tests/defects/event_fixed_array.fe`: a fixed array is a fine message field and return
type, but not an event field, and the note talks about tuples, which the code does not use:

```
error[10-0007]: unsupported ABI field type
10 │ struct Posted { words: [u256; 2] }
   │                        ^^^^^^^^^ `[u256; 2]` is not supported as an event field
   = tuples are not supported in event or custom error fields; use a struct type for grouped data
```

Either support `uint256[2]`, as Solidity events do, or change the note and suggest `DynArray`, which is
what works.

**Name clash.** `tests/defects/param_named_like_const_fn.fe`: a parameter named like a const fn used in its
own type makes the type unresolvable, and the message does not mention the clash:

```
error[8-0064]: const value must be resolvable during type checking
14 │ pub fn first<const N: usize>(_ half: [u256; half(N)]) -> u256 {
   │                                             ^^^^ requires fully-resolved const value
```
