# Compiler findings re-checked on Fe 26.7.0

**2026-11-24.** Every compiler, standard-library and tooling finding in `docs/fe-notes.md` and
`docs/upstream-reports.md` was run again on the two releases side by side. The five drafts in section 3
were then filed as written: A is [argotorg/fe#2071](https://github.com/argotorg/fe/issues/2071), B
[#2072](https://github.com/argotorg/fe/issues/2072), C [#2073](https://github.com/argotorg/fe/issues/2073),
D [#2074](https://github.com/argotorg/fe/issues/2074) and E [#2075](https://github.com/argotorg/fe/issues/2075).

| | |
|---|---|
| Old release | `fe 26.5.1 (149c758)`, `~/.fe/bin/fe` (what the entry is built with) |
| New release | `fe 26.7.0 (e6e6342)`, `~/fe-267/fe` (released 2026-11-18) |
| Also used | `fe 26.6.0 (5b3f96a)`, the release binary, downloaded to `~/recheck-267/fe-266/fe`, to date two changes; master `89d07b9` for one wasm check |
| Machine | i5-13420H, WSL2 Ubuntu; Foundry 1.9.0 (`cast`, `anvil`) |
| Sources | `~/recheck-267/` in WSL: `orig/` (copies of `tests/defects/` and the spike files), `new/`, `speed/`, `native/`, `mask/`, `arr/`, `wasm/` |

Times are wall time of the whole command, one run unless a range is given. "Parallel" marks a time taken
while other compilations ran, so only its result counts.

## 1. Status of every finding

Counts: **15 fixed, 10 still present, 3 changed, 2 could not reproduce, 1 not re-tested** (31 findings).
Eight more entries of the notes are language design, not defects; they are listed in section 4.

| # | Finding (where recorded) | 26.5.1 | 26.7.0 | Status | Upstream |
|---|---|---|---|---|---|
| 1 | EVM stack overflow at run time for a large array in messages (reports 1) | 784 elements, two arms, `-O 1`: `StackOverflow` | 784 passes at `-O 1`; 1,100 still fails at `-O 1`; 512 fails at `-O 0` in one arm | **still present**, threshold moved | not reported: draft A |
| 2 | Minutes of compile time for message arrays (reports 2) | `q6_array_copy_stack.fe` 118.6 s | 2.7 s; but two arms of the same array type are still superlinear (13.0 s at 784, 247 s at 1,568) | **changed** | #835 (open, 2023); in draft A |
| 3 | Native `read_word`/`write_word` little-endian (reports 3) | both tests die with SIGILL | pass | fixed in 26.6.0 | PR #1870 |
| 4 | `black_box` on an array passes `fe check` (reports 4) | `fe check` exits 0 | `error[8-0157]` with a location | fixed in 26.6.0 | #1950 |
| 5 | No warning past 24,576 bytes (reports 5) | silent | `Warning: contract ConstI8 has 32884 bytes of runtime code, exceeding the EIP-170 limit` | fixed in 26.6.0 | #1951 |
| 6 | Internal error storing an array row (reports 6, notes 3.1) | `error[16-0004]` | compiles | fixed in 26.6.0 | #1952 |
| 7 | Event field `[u256; 2]`: note talks about tuples (reports 7, M3.3 19) | tuple note | "fixed-size arrays are not supported in event or custom error fields yet; use `std::abi::DynArray`" | fixed in 26.6.0 (the note; arrays are still unsupported) | #1953 |
| 8 | Parameter named like a `const fn` (reports 7, notes 3.3) | `error[8-0064]`, no mention of the clash | `error[8-0158]` naming the parameter | fixed in 26.6.0 | #1954 |
| 9 | `for` cannot be used in a `const fn` (notes 3.2) | `error[8-0062]` | the same | **still present** | not reported: draft D |
| 10 | `==` on arrays is not usable in a constant expression (notes 3.4) | `error[3-0024]` | the same | **still present** | not reported; see section 4 |
| 11 | No `#[test(should_revert)]` on the native backend (notes 3.5) | "cannot use EVM `should_revert`" | passes | fixed in 26.6.0 | changelog 26.6.0 |
| 12 | No `loop` keyword; it parses as a struct literal (notes 3.6) | `error[2-0002]: loop is not found` | passes | fixed in 26.6.0 | #1937 |
| 13 | A function ending in `while true { .. return .. }` does not type-check (notes 3.7) | "expected `i32`, but `()` is given" | passes | fixed in 26.6.0 | #1937 |
| 14 | One wrong argument gives follow-on errors (notes 3.9) | 4 errors for one mistake | 4 errors; the last now says "cannot infer the const parameter `F` of `dense`" | **still present**, reworded | not reported; see section 4 |
| 15 | `fe fmt` wraps well before 100 columns (notes 3.10) | a 100-column line is left alone | the same | **could not reproduce** | |
| 16 | Compile-time evaluator: fixed limit of 1,000,000 steps (notes 4) | 30,302 simple loop iterations | 24,389: the same limit buys 20% less | **still present**, tighter since 26.6.0 | not reported: draft C |
| 17 | A constant read inside another evaluation is counted again (notes 4) | fails | fails | **still present** | not reported: in draft C |
| 18 | 256-bit shifts slow natively (notes 5) | 9.7 to 10.0 s | 0.76 to 0.80 s | fixed (12x) | changelog 26.6.0 |
| 19 | `downcast_truncate` to `i8` slow natively (notes 5) | 3.56 s, shift pair 0.39 s | 0.30 s, shift pair 0.30 s | fixed | |
| 20 | `isize` slower than `i32` natively (notes 5) | 21.0 to 23.2 s against 0.43 s | 0.40 s against 0.45 s | fixed in 26.6.0 | changelog 26.6.0 |
| 21 | `usize` arithmetic at `u256` speed natively (M2.10 13) | 19.4 to 22.6 s, `u64` 0.37 to 0.40 s | 0.37 to 0.38 s, `u64` 0.38 to 0.40 s | fixed in 26.6.0 | changelog 26.6.0 |
| 22 | Result masks after every `u32`/`u64` operation on the EVM (M2.10 14) | every operation masked | none where the range is known, except after `BYTE`; 26.6.0 had none at all | **changed**: regression from 26.6.0 to 26.7.0 | not reported: draft B |
| 23 | The test runner's address cannot receive ETH (M3.3 21) | | | not re-tested (harness behaviour) | |
| 24 | No time travel in `fe test` (M3.3 22) | timestamp fixed at 1 | the same | **still present in 26.7.0** | #2039, closed by PR #2048 on master after the release |
| 25 | `Address` takes a whole storage word (M3.3 25, M4.4) | `Address, u8, u8, u64`: 2 slots | 1 slot | fixed in 26.6.0 | #1826 |
| 26 | No way to mark a contract as test-only (M3.3 27) | `#[cfg(test)]` accepted without a diagnostic | `error[12-0002]`: `cfg` has no `test` | **still present** | #2038 (open) |
| 27 | `PackedBytes` has no constructor from words (notes 2, README) | field private, only `from_array` | the same | **still present** | not reported: draft E |
| 28 | `PackedBytes::from_array` cannot be evaluated for large arrays (notes 2) | up to 16,871 bytes | up to 15,370 bytes | **still present**, lower | not reported: in drafts C and E |
| 29 | `fe build --backend wasm` refuses an ingot (wasm section) | (no wasm) master `89d07b9`: "only supports standalone `.fe` files" | builds `ing.wasm` | fixed in 26.7.0 | #1981 |
| 30 | `//!` not allowed inside an inline `mod` (wasm section) | `fe check` passes | passes | **could not reproduce** | |
| 31 | `HostText` gives no byte access from safe code (wasm section) | | `HostText::copy()` returns a `DynString` (read in the source, not run) | **changed** | |

## 2. Commands and results

All commands were run in the directory named, with `NO_COLOR=1`. `F5` is the 26.5.1 binary, `F7` the 26.7.0
binary.

| # | Command (directory under `~/recheck-267/`) | 26.5.1 | 26.7.0 |
|---|---|---|---|
| 1 | `fe test -O 1 q6_array_twice.fe` (`orig/`) | `FAIL Test halted: StackOverflow`, 40.7 s | `PASS`, 13.5 s |
| 1 | `fe test -O 0 single_512.fe` (`arr/`, draft A) | `StackOverflow`, 20.7 s | `StackOverflow`, 4.9 s |
| 1 | `fe test -O 1 single_512.fe` | pass, 5.1 s | pass, 1.1 s |
| 1 | `fe test -O 1 twice_1100.fe` (`q6_array_twice.fe` with 1,100) | `StackOverflow`, 229 s (parallel) | `Test reverted: 0x`, 71 s (parallel) |
| 1 | `fe test -O 1 callA_1100.fe` (two arms, one call, draft A; `drafts/two_arms.fe` is the same file as the draft words it, also run at `-O 2`) | | `Test reverted: 0x`, 54.2 s; with 1,000 elements pass, 36.9 s |
| 2 | `fe test -O 1 q6_array_copy_stack.fe` (`orig/`) | 2 pass, 118.6 s | 2 pass, 2.7 s |
| 2 | `fe test -O 1 twice_N.fe`, N = 98, 196, 392, 784, 1,568 (`arr/`) | | 1.4, 1.5, 2.6, 13.0, 247.1 s |
| 3 | `fe test --backend native native_word_endianness.fe` (`orig/`); on 26.7.0 the `_master` copy, because `MemBuffer::alloc` became `MemBuffer::new` | 2 fail, `signal: 4 (SIGILL)`, 0.7 s | 2 pass, 1.0 s |
| 4 | `fe check q6_black_box.fe` | exit 0, 0.2 s | `error[8-0157]: core::hint::black_box requires a scalar argument`, 0.3 s |
| 5 | `fe build --out-dir out .` in `orig/q6_const_size` | no warning, 1.7 s | two EIP-170 warnings, 1.4 s |
| 6 | `fe check store_array_row.fe` | `error[16-0004]: internal access checking error`, 0.3 s | exit 0, 0.3 s |
| 7 | `fe check event_fixed_array.fe` | tuple note, 0.2 s | array note, 0.3 s |
| 8 | `fe check param_named_like_const_fn.fe` | `error[8-0064]`, 0.2 s | `error[8-0158] ... half resolves to a function parameter here`, 0.3 s |
| 9 | `fe check for_in_const_fn.fe`; `fe check const_for.fe` (`new/`) | `error[8-0062]: non-const call in const fn ... start is not a const fn`, 0.2 s | the same, 0.3 s |
| 10 | `fe check array_eq_in_const.fe` | `error[3-0024]: non-const function call in const context`, 0.2 s | the same, 0.3 s |
| 11 | `fe test --backend native should_revert_native.fe` (`new/`) | "native test `rejects` cannot use EVM `should_revert` or `balance` attributes", 0.2 s | `PASS rejects`, 0.9 s |
| 12 | `fe test loop_kw.fe` | `error[2-0002]: loop is not found`, 0.3 s | pass, 0.3 s |
| 13 | `fe test while_true_return.fe` | "expected `i32`, but `()` is given", 0.2 s | pass, 0.3 s |
| 14 | `fe check ingots/case` (`cf/`: `tests/compile_fail/shape_mismatch.fe` against `tensile`) | 3 type mismatches and "type annotation is needed", 0.3 s | 3 type mismatches and "cannot infer the const parameter `F` of `dense`", 0.3 s |
| 15 | `fe fmt fmtw.fe` (one 100-column signature) | unchanged | unchanged |
| 16 | `fe check loop25000.fe` (draft C); bisection of the loop count | exit 0, 1.0 s; largest passing 30,302 | `error[3-0073]: const evaluation exceeded the step limit`, 0.5 s; largest passing 24,389 (26.6.0 and master `89d07b9`: between 24,320 and 24,417) |
| 16 | `fe check q3_limit_steps.fe` (2,000,000 iterations) | step limit, 1.6 s | step limit, 0.7 s |
| 16 | bisection of N in a `[u8; N]` table filled by a `const fn` loop | between 13,631 and 13,728 | between 12,270 and 12,367 |
| 17 | `fe check ctfe_reeval.fe` (draft C) | step limit, 3.4 s; without the last line exit 0, 2.5 s | step limit, 1.2 s; without the last line exit 0, 1.0 s |
| 18 | `fe build --backend native -O 1 --out-dir out shift256.fe && time out/shift256` (`native/`, 256 million iterations) | 10.02, 9.68 s | 0.80, 0.76 s |
| 19 | the same with `trunc.fe` (`(i >> 3).downcast_truncate()` to `i8`) and `shiftpair.fe` | 3.56, 3.58 s; 0.39, 0.38 s | 0.30, 0.29 s; 0.29, 0.31 s |
| 20, 21 | the same with `sum_T.fe` (`speed/`: the loop of `tests/defects/usize_native_speed.fe` in each type) | `u32` 0.41, `u64` 0.37, `i32` 0.45, `usize` 22.6 and 19.4, `isize` 23.2 and 21.0, `u256` 22.1 and 21.4 s | `u32` 0.48, `u64` 0.40, `i32` 0.46, `usize` 0.38, `isize` 0.40, `u256` 0.91 and 0.84 s |
| 22 | `fe build -O 2 --out-dir o lanes_u64.fe`, `cast disassemble`, and `debug_traceCall` on anvil (`mask/`, draft B) | 585 gas, 7 masks | 382 gas, 6 masks (26.6.0: 334 gas, none) |
| 24 | `fe test timestamp.fe` (the example of #2039) | `timestamp_argument` fails, 0.3 s | the same, 0.4 s |
| 25 | `fe test storage_pack.fe` (reads slots 0 and 1 with `RawStorage::sload`) | `record_takes_two_slots` passes, 0.9 s | `record_takes_one_slot` passes, 0.9 s |
| 26 | `fe check cfg_test.fe` (`#[cfg(test)] pub contract Double {}`) | exit 0, no diagnostic, 0.3 s | `error[12-0002]: invalid #[cfg] attribute form on item`, 0.3 s |
| 27 | `fe check packed_words.fe` | `error[2-0003]: words is not visible`, 0.2 s | the same, 0.3 s |
| 28 | `fe check packed_16384.fe`; bisection of N | exit 0, 2.3 s; largest passing 16,871 | step limit in `src/bytes.fe:122`, 0.9 s; largest passing 15,370 |
| 29 | `fe build --backend wasm --out-dir out ing` (`wasm/`, a one-function ingot) | master `89d07b9`: "WebAssembly output only supports standalone `.fe` files" | `Wrote out7/ing.wasm`, 0.3 s |
| 30 | `fe check inner_doc.fe` (`mod inner { //! .. }`) | exit 0 | exit 0 |

## 3. Issue drafts

In order of value to the compiler team. Each was searched for in `argotorg/fe` (issues and pull requests,
all states) and, for B, in `fe-lang/sonatina`, on 2026-11-24, with the keywords listed in section 5.
None was found.

---

### Draft A

**Title:** EVM `StackOverflow` at run time for a large fixed-size array in a message: 512 elements at `-O 0`, 1,100 at `-O 1`

**Version:** `fe 26.7.0 (e6e6342)`. Also on `fe 26.5.1 (149c758)`.

Code that passes `fe check` and builds without a diagnostic halts in the EVM with a stack overflow once
a message field is a large enough array. There are two cases.

**Case 1, `-O 0`, one arm.** `one.fe`:

```fe
const BIG: [i8; 512] = [7; 512]

msg M {
    #[selector = 0x01]
    A { v: [i8; 512] } -> u256,
}

pub contract C {
    recv M {
        A { v } -> u256 {
            if v[0] == 7 && v[511] == 7 { 1 } else { 0 }
        }
    }
}

#[test]
fn array_in_one_call() uses (evm: mut Evm) {
    let c = evm.create2<C>(value: Wei::zero(), args: (), salt: 0)
    let a: u256 = evm.call(addr: c, gas: 100000000, value: Wei::zero(), message: M::A { v: BIG })
    assert!(a == 1)
}
```

```
$ fe test -O 0 one.fe
FAIL  [0.0006s] array_in_one_call
    Test halted: StackOverflow (gas: 1000000000)
```

With `--call-trace --trace-evm` the call trace has no call to `C` (`CREATE2 ok -> $1`, then the test's own
frame ends), and the trace ends in the test function at a full stack:

```
TRACE (last 14 of 11764 steps)
pc=29247 op=0x91 stack=1024 gas_rem=996083171 top=0x20924,0x4000,0x7244
pc=29248 op=0x61 stack=1024 gas_rem=996083168 top=0x7244,0x4000,0x20924
trace result: Ok(Halt { reason: StackOverflow, gas_used: 1000000000 })
```

So the overflow is in the code that builds the message. It passes with 450 elements, fails with 512, and
`[u256; 512]` fails the same way. Copying the same constant into a local, or passing it to a function,
passes at `-O 0`. At `-O 1` and `-O 2` the file passes.

**Case 2, `-O 1` and `-O 2`, two arms of the same array type.** Change `512` to `1100` and `511` to
`1099` in `one.fe`, and add a second arm, which the test never calls:

```fe
    #[selector = 0x02]
    B { v: [i8; 1100] } -> u256,
```
```fe
        B { v } -> u256 {
            if v[0] == 7 && v[1099] == 7 { 2 } else { 0 }
        }
```

```
$ fe test -O 1 two.fe          (about a minute to compile; -O 2 gives the same result)
FAIL  [0.0024s] array_in_one_call
    Test reverted: 0x
```

Here the callee halts. The trace shows `C` at `stack=1024` while it still has 99.8 million of its
100 million gas, and the test's frame then reports all of the forwarded gas as used:

```
pc=13519 op=0x80 stack=1023 gas_rem=99795172 top=0x7,0x7,0x7
pc=13520 op=0x61 stack=1024 gas_rem=99795169 top=0x7,0x7,0x7
...
trace result: Ok(Revert { gas_used: 106252153, output: 0x })
```

What changes the result, all at `-O 1` on 26.7.0:

| Contract | Result | `fe test` wall time |
|---|---|---|
| one arm, `[i8; 6000]` | pass | 4.3 s |
| two arms, `[u8; 600]` and `[u8; 1100]` | pass | 2.7 s |
| two arms, both `[i8; 784]`, each called once | pass | 13.0 s |
| two arms, both `[i8; 1000]`, one called | pass | 36.9 s |
| two arms, both `[i8; 1100]`, one called | callee halts | 54.2 s |
| two arms, both `[i8; 1568]`, each called once (also with `u8`) | callee halts | 247.1 s |

Compile time for the two-arm form grows much faster than the array: 1.4, 1.5, 2.6, 13.0 and 247.1 s for
98, 196, 392, 784 and 1,568 elements, where the one-arm form takes 4.3 s for 6,000. `fe build` of the
784-element file alone takes 16 s, so the time goes into the contract, not the test. This may be the
same cause as the overflow, and it is the current form of #835.

On 26.5.1 the two-arm form already overflowed at 784 elements at `-O 1`; 26.7.0 moved the threshold, it
did not remove it.

**Expected:** the tests pass, or the compiler refuses a function whose code cannot keep within the
1,024-slot stack. A stack overflow should not be reachable from safe code that type-checks.

**Why it matters:** nothing warns before deployment, and in case 2 the contract's arm can never be
called. `-O 0` is the level one reaches for when debugging, and there the limit is 512 elements in the
simplest possible contract.

Found while building https://github.com/AbrahamPaulJ/tensile

---

### Draft B

**Title:** 26.7.0: `u32`/`u64` truncation masks return after `BYTE` is selected for `(x >> 8k) & 0xff`; 26.6.0 emitted none

**Version:** `fe 26.7.0 (e6e6342)`. Compared with `fe 26.6.0 (5b3f96a)`.

26.6.0 removes the truncation mask after a `u32` or `u64` operation when the operands' ranges show that
the result fits. 26.7.0 selects `BYTE` for a byte lane, and after that the masks are back, so code that
multiplies bytes held in narrow integers costs more than on 26.6.0.

`lanes.fe`:

```fe
msg LanesMsg {
    #[selector = 1]
    Dot4 { a: u64, b: u64 } -> u64,
}

pub contract Lanes {
    recv LanesMsg {
        Dot4 { a, b } -> u64 {
            dot4(a, b)
        }
    }
}

// Four 8-bit lanes: every product is below 2**16 and the sum below 2**18.
#[arithmetic(unchecked)]
fn dot4(_ a: u64, _ b: u64) -> u64 {
    ((a >> 24) & 0xff) * ((b >> 24) & 0xff)
        + ((a >> 16) & 0xff) * ((b >> 16) & 0xff)
        + ((a >> 8) & 0xff) * ((b >> 8) & 0xff)
        + (a & 0xff) * (b & 0xff)
}
```

```
$ fe build -O 2 --out-dir out lanes.fe
$ cast disassemble 0x$(cat out/Lanes.runtime.bin)
```

The body of `dot4` on 26.6.0, no mask:

```
DUP2 PUSH1 0x18 SHR PUSH1 0xff AND  DUP2 PUSH1 0x18 SHR PUSH1 0xff AND  MUL
DUP3 PUSH1 0x10 SHR PUSH1 0xff AND  DUP3 PUSH1 0x10 SHR PUSH1 0xff AND  MUL ADD
DUP3 PUSH1 0x08 SHR PUSH1 0xff AND  DUP3 PUSH1 0x08 SHR PUSH1 0xff AND  MUL ADD
SWAP2 PUSH1 0xff AND  SWAP1 PUSH1 0xff AND  MUL ADD
```

On 26.7.0, six masks of 14 gas each (`PUSH0 NOT PUSH1 0xc0 SHR AND`), one after each product of two
`BYTE` results and one after each addition:

```
DUP2 PUSH1 0x1c BYTE  DUP2 PUSH1 0x1c BYTE  MUL  PUSH0 NOT PUSH1 0xc0 SHR AND
DUP3 PUSH1 0x1d BYTE  DUP3 PUSH1 0x1d BYTE  MUL  PUSH0 NOT PUSH1 0xc0 SHR AND  ADD  PUSH0 NOT PUSH1 0xc0 SHR AND
DUP3 PUSH1 0x1e BYTE  DUP3 PUSH1 0x1e BYTE  MUL  PUSH0 NOT PUSH1 0xc0 SHR AND  ADD  PUSH0 NOT PUSH1 0xc0 SHR AND
SWAP2 PUSH1 0xff AND  SWAP1 PUSH1 0xff AND  MUL  ADD  PUSH0 NOT PUSH1 0xc0 SHR AND
```

The product of the two lanes that stay `x & 0xff` has no mask, so the range analysis is still at work; it
is only behind `BYTE` that the range is lost. With 7-bit lanes (`& 0x7f`, which cannot become `BYTE`)
26.7.0 emits no mask at all.

Execution gas of one call with `a = 0x01020304`, `b = 0x05060708`, from `debug_traceCall` on anvil 1.9.0
(Osaka), `-O 2` (`-O 1` gives the same code):

| Function | 26.5.1 | 26.6.0 | 26.7.0 | 26.7.0 without the masks |
|---|---|---|---|---|
| `dot4` in `u64` (above) | 585 gas, 207 bytes, 7 masks | 334 gas, 136 bytes, 0 masks | 382 gas, 154 bytes, 6 masks | 298 gas |
| `dot4` in `u32` | 501, 201 bytes, 7 masks | 306, 130 bytes, 0 | 312, 148 bytes, 5 | 282 |
| two lanes in `u32` | 413, 151 bytes, 3 masks | 242, 104 bytes, 0 | 242, 110 bytes, 2 | 230 |
| two lanes in `u32`, `& 0x7f` | 413, 151 bytes | 242, 104 bytes, 0 | 242, 104 bytes, 0 | |
| `dot4` in `u256` | 278, 117 bytes | 278, 117 bytes | 242, 99 bytes | |

**Expected:** `BYTE` and no masks: 298 gas for the `u64` function, not 382. The `u256` row shows what
`BYTE` selection is worth when no mask interferes.

**Actual:** the `u64` function costs 48 gas (14%) more than on 26.6.0 and is 18 bytes larger.

I have not looked for the cause. fe-lang/sonatina#572, which added the `BYTE` selection, says that known
bits were taught about `BYTE` and mentions a definedness guard on mask removal, so this may be that guard
or the order of the two passes. Not checked against a commit later than `e6e6342`.

**Why it matters:** bytes packed into `u32` or `u64` limbs are the form that is also fast in native code,
so portable kernels are written this way. In this project's contract for one 784-input neuron (a dot
product over `u32` limbs with `(a >> shift) & 0xff` lanes, same source on all three releases, `-O 1`),
execution gas is 54,588 on 26.5.1, 44,470 on 26.6.0 and 46,035 on 26.7.0; the trace runs 1,904, 325 and
1,500 `PUSH4 0xffffffff AND` masks. The 1,175 masks that came back cost 7,050 gas, 15% of the call.

Found while building https://github.com/AbrahamPaulJ/tensile

---

### Draft C

**Title:** Const evaluation: the step limit buys 20% less than in 26.5.1 (24,389 loop iterations, was 30,302), cannot be raised, and counts a constant again wherever it is read

**Version:** `fe 26.7.0 (e6e6342)`. The same on `fe 26.6.0 (5b3f96a)`; compared with `fe 26.5.1 (149c758)`.

**1. Code that 26.5.1 evaluates now exceeds the limit.** `loop.fe`:

```fe
const fn work(_ seed: u256) -> u256 {
    var acc: u256 = seed
    var i: u256 = 0
    while i < 25000 {
        acc += i & 3
        i += 1
    }
    acc
}

const A: u256 = work(1)
```

```
$ fe check loop.fe        # 26.5.1: exit 0, no output
$ fe check loop.fe        # 26.6.0 and 26.7.0
error[3-0073]: const evaluation exceeded the step limit
   ┌─ loop.fe:11:17
   │
 6 │         i += 1
   │         ------ const evaluation takes too many steps
   ·
11 │ const A: u256 = work(1)
   │                 ^^^^^^^ while evaluating this
```

The limit is 1,000,000 steps in both (`ctfe/machine.rs`, `step_limit: 1_000_000`), so the same source now
costs more steps. Largest size that still evaluates, found by bisection:

| Computation | 26.5.1 | 26.7.0 |
|---|---|---|
| iterations of the loop above | 30,302 | 24,389 |
| `core::bytes::PackedBytes<N>::from_array([7; N])`, N | 16,871 | 15,370 |
| a `[u8; N]` table filled by `t[i] = ((i * 7 + 3) % 251).downcast_truncate()`, N | 13,631 to 13,728 | 12,270 to 12,367 |

**2. A constant is counted again by every evaluation that reads it.** `two.fe`: the function above with
`20000` in place of `25000`, then

```fe
const A: u256 = work(1)
const B: u256 = work(2)

static_assert(A == 30001)
static_assert(B == 30002)
static_assert(A + B == 60003)
```

```
$ fe check two.fe
error[3-0026]: const evaluation exceeded the step limit
  ┌─ two.fe:1:37
  │
1 │   const fn work(_ seed: u256) -> u256 {
  ...
  │ ╰─^ const evaluation takes too many steps
```

Without the last line the file passes, so `A` and `B` are each within the limit and only the expression
that reads both is not. The wall time (1.2 s with the line, 1.0 s without) suggests the values are not
computed again, only charged again. Splitting a large computation into named constants therefore does
not help: the evaluation that combines them pays for all of them.

**3. The limit cannot be changed.** I found no `fe.toml` key, attribute or command-line option for it.

**Expected:** any of these would do: the cost of the same source not rising between releases; a finished
constant costing its reader one step; a way to raise the limit per project (`fe.toml`) or per item.

**Why it matters:** 1,000,000 steps is about 24,000 iterations of a three-line loop, and the standard
library's own `PackedBytes::from_array` stops at 15,370 bytes. Checking or packing a 25,088-byte table at
compile time is out of reach, and a project that sat between the two thresholds on 26.5.1 stops
compiling on upgrade with no change of its own.

Found while building https://github.com/AbrahamPaulJ/tensile

---

### Draft D

**Title:** `for i in 0..N` and `for x in array` cannot be used in a `const fn`: the `Collection` impls of `Range` and arrays are not `const fn`

**Version:** `fe 26.7.0 (e6e6342)`. The same on 26.5.1.

`const_for.fe`:

```fe
const fn total<const N: usize>(_ values: [u8; N]) -> u256 {
    var sum: u256 = 0
    for i in 0..N {
        sum += values[i] as u256
    }
    sum
}

const T: u256 = total([1, 2, 3])
static_assert(T == 6)
```

```
$ fe check const_for.fe
error[3-0024]: non-const function call in const context
  ┌─ const_for.fe:3:14
  │
3 │     for i in 0..N {
  │              ^^^^ calls in const context must be `const fn`

error[8-0062]: non-const call in `const fn`
   ┌─ const_for.fe:3:5
   │
 3 │ ╭     for i in 0..N {
 4 │ │         sum += values[i] as u256
 5 │ │     }
   │ ╰─────^ `start` is not a `const fn`
   │
   ┌─ src/iter.fe:23:8
   │
23 │     fn start(self) -> Self::Cursor uses E
   │        ----- callee defined here
```

and the same for `has`, `next` and `at`. `for v in values` gives the same errors.

The loop itself is not the obstacle. A user type whose `Collection` impl has `const fn` methods works in
a `const fn` today (`fe check` passes, and the `static_assert` holds):

```fe
use core::iter::Collection

struct Upto { n: usize }

impl Collection for Upto {
    type Cursor = usize
    type Item = usize

    const fn start(self) -> usize { 0 }
    const fn has(self, _ c: usize) -> bool { c < self.n }
    const fn next(self, _ c: usize) -> usize { c + 1 }
    const fn at(self, _ c: usize) -> ref usize { ref c }
}

const fn total(_ n: usize) -> usize {
    var sum: usize = 0
    let range = Upto { n }
    for i in range {
        sum += i
    }
    sum
}

const T: usize = total(5)
static_assert(T == 10)
```

**Expected:** `for` over a range or an array in a `const fn`. For ranges this looks like marking the four
methods of each `impl Collection for Range<..>` in `core/src/range.fe` as `const fn` (their bodies are
one line of arithmetic each), and the same for `impl<T, const N: usize> Collection for [T; N]` in
`core/src/iter.fe`.

**Why it matters:** every loop in a `const fn` has to be a `while` with a hand-kept counter, which is
where off-by-one mistakes in compile-time tables come from. Related: #2065 and #1349 ask for other parts
of `core` to be usable in `const fn`.

Found while building https://github.com/AbrahamPaulJ/tensile

---

### Draft E

**Title:** `core::bytes::PackedBytes`: no constructor from words, and `from_array` is limited by const evaluation to 15,370 bytes

**Version:** `fe 26.7.0 (e6e6342)`. The same on 26.5.1, where the limit was 16,871 bytes.

`PackedBytes<N>` is the standard type for a fixed byte sequence packed 32 to a word. Its field is
private and the only public constructor is `from_array`, so a large constant can only be made by letting
the compiler pack it a byte at a time, which runs into the step limit of const evaluation. `packed.fe`:

```fe
use core::PackedBytes

const BYTES: [u8; 16384] = [7; 16384]
const PACKED: PackedBytes<16384> = PackedBytes<16384>::from_array(BYTES)
```

```
$ fe check packed.fe
error[3-0073]: const evaluation exceeded the step limit
    ┌─ packed.fe:4:36
    │
  4 │ const PACKED: PackedBytes<16384> = PackedBytes<16384>::from_array(BYTES)
    │                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ while evaluating this
    │
    ┌─ src/bytes.fe:122:9
    │
122 │         self.len += width
    │         -------- const evaluation takes too many steps
```

The largest N that evaluates is 15,370. Writing the words directly is refused:

```fe
const PACKED: PackedBytes<64> = PackedBytes<64> { words: [1, 2] }
```
```
error[2-0003]: `words` is not visible
```

**Expected:** a public `const fn` that takes the words, for example
`PackedBytes<N>::from_words(_ words: [u256; word_len(N)]) -> Self`, which asserts that the unused low
bytes of the last word are zero. A generator can then emit the packed literal and the compiler does no
per-byte work. `word(index)` already reads a word back, so only the way in is missing.

**Why it matters:** a table of 25,088 bytes (784 words) could not be a `PackedBytes` constant, so the
project carries its own copy of the type with the same layout, built from words.

Found while building https://github.com/AbrahamPaulJ/tensile

## 4. Not filed, and why

**Already reported upstream**

- **No time travel in `fe test` (24).** Reported by another entrant as #2039 on 2026-11-19 and closed by
  PR #2048 on 2026-11-20, which adds `#[test(timestamp = N, block_number = N)]`. That is on master, not
  in 26.7.0, where the example of #2039 still fails. The pull request says moving time forward inside one
  test is left out on purpose ("a new API rather than a small change"). Our contract tests need exactly
  that to cross a deadline in one test, so the constant-window workaround stays.
- **No test-only contracts (26).** #2038 (open) asks for `#[cfg(test)]` and the other predicate forms.
  Note that 26.5.1 accepts `#[cfg(test)]` without a diagnostic, while 26.7.0 rejects it.
- **Compile time of large message arrays (2).** #835 is open since 2023. The measurements are part of
  draft A because the slow case and the overflowing case have the same trigger. If draft A is filed, one
  line on #835 pointing to it is enough.

**Arguably intended, or a language feature rather than a defect**

- **`==` on arrays in a constant expression (10).** `impl<T: Eq, const N: usize> Eq for [T; N]` calls
  `T::eq`, so making it `const fn` needs const trait bounds, a language feature, not the one-keyword
  change of draft D. Draft D mentions the family (#2065, #1349).
- **Follow-on errors after one wrong argument (14).** The first error is exact; the others follow from an
  uninferred const parameter. 26.7.0 already words the last one better. Cosmetic.
- **The test runner's address cannot receive ETH (23).** A property of the harness; a `Wallet` contract
  in the test is the normal answer. Not re-tested.
- **Design choices, not re-run:** requantising upward is rejected by the type before any `where` clause
  (notes 3.8); `#[inline(always)]` moves gas both ways (M2.10 15); returning an array element at a dynamic
  index needs `Copy` (M2.10 16); a shift amount has the type of the shifted value (M2.10 17); a `msg` is a
  module of variant types (M3.3 20); narrowing `as` casts are rejected (M3.3 24, and stricter per target
  since 26.6.0); the ABI decoder refuses non-canonical words (M3.9 7); label rules (notes 3.11, M3.3 23).

**Could not reproduce**

- **`fe fmt` wrapping before 100 columns (15).** A 100-column function signature is left alone by both
  releases. The notes record no example, so there is nothing to file.
- **`//!` inside an inline `mod` (30).** `mod inner { //! doc .. }` passes `fe check` on both releases.
  The original was seen in a generated file on master `89d07b9`, which the entry no longer has.

**Fixed, nothing to do:** rows 3 to 8, 11 to 13, 18 to 21, 25 and 29 of the table.

## 5. Notes for the entry's own documents

The entry is measured on 26.5.1 and its documents describe that release; the first two points below
were then added to `docs/upstream-reports.md` and `docs/gas-gap.md`.

- **`docs/upstream-reports.md` says defect 1 (stack overflow) is "fixed" on master.** Only the recorded
  size is: 784 elements pass on 26.7.0 at `-O 1`, 1,100 do not, and 512 fail at `-O 0` on both releases.
  This is draft A and was never filed.
- **`docs/gas-gap.md` and the README put 68.7% of the neuron's gas gap on masks "on values that provably
  fit" and name range analysis as the fix.** That analysis exists since 26.6.0. On 26.7.0 part of it is
  undone for byte lanes, which is exactly the kernel's `lane` function (`(a >> shift) & 0xff`). Measured
  here on a scratch copy of `ingots/` (`~/recheck-267/entry`, `fe build -O 1 --ingot onchain`, then
  `debug_traceCall` of `neuron1` on reference sample 0, result 36 on all three):

  | Release | Execution gas | `u32` masks executed | `NeuronCheck` bytes |
  |---|---|---|---|
  | 26.5.1 | 54,588 | 1,904 | 4,301 |
  | 26.6.0 | 44,470 | 325 | 3,369 |
  | 26.7.0 | 46,035 | 1,500 | 3,263 |

  The 26.5.1 and 26.7.0 figures equal `docs/numbers.md` sections 3 and 7. So the best release for this
  kernel so far is 26.6.0, and 26.7.0 without the returned masks would be about 38,985 gas, 6% above the
  Solidity twin's 36,653 (a projection: 46,035 minus 1,175 masks at 6 gas).
- **`ingots/onchain/src/record.fe`, `docs/fe-notes.md` M3.3 25 and M4.4 say `Address` is a whole word.**
  True on 26.5.1, not since 26.6.0: `Address, u8, u8, u64` is one slot on 26.7.0. The README already says
  that 26.6.0 packs storage.
- **`tests/defects/usize_native_speed.fe` and M2.10 13.** Fixed in 26.6.0: `usize` 0.37 s against
  19 to 23 s. `u256` arithmetic in native code is also about 25 times faster (0.9 s against 22 s), and
  256-bit shifts 12 times, so the native times of the `i256` kernel forms in M2.7 no longer describe the
  current compiler.
- **Notes 3.5, 3.6, 3.7** (`should_revert` on native, `loop`, a function ending in `while true`) are
  all fixed in 26.6.0.
- **The README's reason for not using `PackedBytes` still holds,** and the limit is lower now (15,370
  bytes, was 16,871).

Search terms used on the trackers, none of which found an existing report for drafts A to E:
`StackOverflow`, `stack overflow`, `stack too deep`, `large array`, `array message`, `mask`, `u32 mask`,
`redundant mask`, `truncation mask`, `narrow integer`, `range analysis`, `known bits`, `BYTE`,
`step limit`, `const evaluation exceeded`, `CTFE`, `static_assert`, `for loop in const fn`, `const fn for`,
`array == const`, `PackedBytes`, `from_array`, `from_words`, `usize native`, `storage packing Address`,
`timestamp fe test`, `cfg test`.
