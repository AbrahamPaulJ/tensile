# Feasibility spike: quantised inference in Fe 26.5.1

Date: 2026-11-05. Compiler: `fe 26.5.1 (149c758)` in WSL2 Ubuntu 24.04, Foundry 1.9.0 (anvil `--hardfork osaka`).
Everything below was run, unless a line is labelled "projection". All paths are relative to
`C:\Users\abrah\Desktop\CC\competitions\fe-challenge\spike\inference\`. Scripts are run as
`wsl -e bash /mnt/c/Users/abrah/Desktop/CC/competitions/fe-challenge/spike/inference/<script>.sh`.

| Question | Verdict |
| --- | --- |
| 1. Const-generic shapes | WORKS |
| 2. Scale in the type | WORKS |
| 3. Compile-time tables | WORKS (hard limit: 1,000,000 CTFE steps per evaluation, about 12,000 table entries) |
| 4. A real layer on the EVM | WORKS WITH WORKAROUND (weights must be bit-packed; 784 x 32 needs separate weight contracts) |
| 5. One source, two backends | WORKS (bit-identical on 10,021 inputs) |
| 6. Blockers and surprises | 2 compiler defects around large arrays, 1 size trap, several small ones; none fatal |

Files: `q1_*.fe`, `q2_*.fe`, `q3_*.fe`, `q6_*.fe` are standalone test files. `layer/` is a workspace with
four ingots: `nn` (the shared pure layer code plus generated `weights.fe`), `onchain` and `onchain_split`
(contracts and gas tests), `cli` (native executable). `gen_weights.py` writes the weights, the test input
and the Python integer reference output. `gen/` holds generated copies and logs (`gen/layer784` is the
784 x 32 copy of the workspace).

---

## 1. Const-generic shapes: WORKS

Run: `q1.sh`, `q2.sh` (the second also runs `q1_mismatch2.fe`).

`q1_shapes.fe` (3 tests pass):

```fe
pub struct Matrix<const R: usize, const C: usize> {
    pub data: [[i32; C]; R],
}

impl<const R: usize, const C: usize> Matrix<R, C> {
    pub fn matmul<const K: usize>(self, _ other: Matrix<C, K>) -> Matrix<R, K> { ... }
}

impl<const R: usize, const K: usize, const C: usize> Mul<Matrix<K, C>> for Matrix<R, K> {
    type Output = Matrix<R, C>
    fn mul(own self, _ other: own Matrix<K, C>) -> Matrix<R, C> { self.matmul<C>(other) }
}
```

`let c = a * b` with `a: Matrix<2, 3>`, `b: Matrix<3, 2>` infers `Matrix<2, 2>` and computes the right
values. Output:

```
PASS  [0.0023s] signed_types_in_arrays
PASS  [0.0024s] operator_mul_infers_shape
PASS  [0.0025s] matmul_2x3_times_3x2
```

Shape mismatch is a compile error (`q1_mismatch.fe`, `q1_mismatch2.fe`), and the messages are readable:

```
error[8-0000]: type mismatch
21 |     let c = a.matmul(b)
   |                      ^ expected `Matrix<3, _>`, but `Matrix<4, 2>` is given

error[8-0000]: type mismatch
30 |     let c = a * b
   |                 ^ expected `Matrix<3, _>`, but `Matrix<4, 2>` is given

error[8-0000]: type mismatch
45 |     let c: Matrix<2, 4> = a * b
   |                           ^^^^^ expected `Matrix<2, 4>`, but `Matrix<2, 5>` is given
```

Arithmetic on const generics in type position works (`q1_flat.fe`, 1 test passes):

```fe
pub struct Flat<const R: usize, const C: usize> {
    pub data: [i32; { R * C }],
}
let a: Flat<2, 3> = Flat { data: [1, 2, 3, 4, 5, 6] }     // accepted: 6 == 2 * 3
```

A wrong literal length is rejected: ``expected `[i32; 6]`, but `[i32; 5]` is given``. So open issue
#1372 (const arithmetic in array types not evaluated during unification) did not reproduce for this
pattern on 26.5.1. Calling a const fn in type position also works (`[[u256; words(IN)]; OUT]` in
`layer/nn/src/lib.fe`, used on both backends).

Signed integers: `i8`, `i16`, `i32`, `i64`, `i128`, `i256` all work as const array element types with
their minimum and maximum literals (`signed_types_in_arrays`). `i8 as i32` widening works with `as`;
narrowing or sign-changing needs `.downcast_truncate()` (or `.downcast()`, `.downcast_saturate()`).
`>>` on a negative `i32` is an arithmetic shift (`-1000 >> 3 == -125`). `isize` exists in the fixtures
but was not tested here.

## 2. Scale in the type: WORKS

Run: `q2.sh`. Files: `q2_scale.fe` (3 tests pass plus a `static_assert`), `q2_mismatch.fe` (fails on purpose).

```fe
pub struct Q<const F: usize> { pub raw: i32 }

impl<const F: usize> Add for Q<F> {
    const fn add(own self, _ other: own Self) -> Self { Q { raw: self.raw + other.raw } }
}

// Q<A> * Q<B> -> Q<{A + B}>
impl<const A: usize, const B: usize> Mul<Q<B>> for Q<A> {
    type Output = Q<{ A + B }>
    const fn mul(own self, _ other: own Q<B>) -> Q<{ A + B }> { Q { raw: self.raw * other.raw } }
}

pub const fn qmul<const A: usize, const B: usize>(_ a: Q<A>, _ b: Q<B>) -> Q<{ A + B }> { ... }

// explicit requantise, only downwards
pub const fn requant<const G: usize>(self) -> Q<G> where G <= F {
    Q { raw: self.raw >> (F - G).downcast_truncate() }
}

static_assert(ct_demo() == -960)   // Q8 * Q4 -> Q12 -> requant<8>, evaluated by the compiler
```

`let c: Q<12> = a * b` (a: `Q<8>`, b: `Q<4>`) compiles and `(a * b).requant<8>()` gives the right raw
value at run time and at compile time. So `Q<{A + B}>` works both as an associated `Output` type and as a
plain return type. This is the same mechanism as `core::fixed::UFixed<const D: u256>` (decimal scale,
operators keep the scale, `rescale<E>()` converts), except that the standard library never adds scales
in a type; that part is new here and it works.

Compile errors from `q2_mismatch.fe`:

```
32 |     let c = a + b                    expected `Q<8>`, but `Q<16>` is given
40 |     let c: Q<8> = qmul(a, b)         expected `Q<8>`, but `Q<12>` is given
54 |     let c = a + 5                    expected `Q<8>`, but `{integer}` is given

error[8-0093]: const requirement is not satisfied
13 |     pub const fn requant<const G: usize>(self) -> Q<G> where G <= F {
   |                                                              ------ required by this predicate
47 |     let c: Q<12> = a.requant<12>()
   |                    ^^^^^^^^^^^^^^^ condition evaluated to `false`
```

Two things needed fixing on the way (both with clear diagnostics): a struct is not `Copy` unless you
write `impl<const F: usize> Copy for Q<F> {}` (otherwise `a * b` moves `a`), and `(F - G) as i32` is
rejected ("cast is not provably lossless"), use `.downcast_truncate()`.

## 3. Compile-time tables: WORKS, with a hard step limit

Run: `q3.sh`, `q3_limits.sh`, `q3_limits2.sh`. Reference values: `python q3_ref.py`.

`q3_tables.fe` builds a 256-entry sigmoid table (`[u8; 256]`) and a clamped-ReLU table (`[i8; 256]`) with
`const fn`. Even the constant e^(1/16) is computed at compile time by a Taylor series loop, and each
sigmoid entry runs a loop of up to 128 multiplications:

```fe
const fn sigmoid_table() -> [u8; 256] {
    var t: [u8; 256] = [0; 256]
    var i: usize = 0
    while i < 256 {
        t[i] = sigmoid_entry(i)
        i += 1
    }
    t
}
pub const SIGMOID: [u8; 256] = sigmoid_table()
static_assert(SIGMOID[144] == 186)
```

How compile-time evaluation was confirmed:

1. `static_assert` on entries passes, and a wrong one fails the build with the computed value
   (`q3_static_assert_fail.fe`):
   ```
   error[8-0081]: static assertion failed
   15 | static_assert(SQ[4] == 17)
      |               ^^^^^^^^^^^ `static_assert` evaluated to `false`
      = left: `16`
      = right: `17`
   ```
2. The asserted entries equal the Python integer reference (`q3_ref.py`: 0 -> 0, 128 -> 128, 144 -> 186,
   255 -> 255), and a runtime test compares all 256 table entries with the same function run in the EVM.
3. Gas, measured with `ctx.gas()` deltas and reported through an event (`fe test --show-logs`): reading
   `SIGMOID[250]` costs 68 gas (0x44), computing the same entry at run time costs 10,993 gas (0x2af1).

Cost: `fe test q3_tables.fe` takes 4.6 s wall, against roughly 0.5 to 1.3 s for the other small files,
so the CTFE work for this table is about 3.5 s.

Limits:

- Loops (`while`, `break`, `continue`), nested const fn calls, array writes and structs all work in
  const fn.
- The step limit is 1,000,000 per evaluation, hard-coded (`crates/hir/src/analysis/semantic/ctfe/machine.rs:70`,
  `step_limit: 1_000_000`). I found no CLI flag or `fe.toml` key that changes it.
- Measured with a trivial loop body (`t[i] = ((i * 7 + 3) % 251).downcast_truncate()`): tables of 4,096,
  8,192 and 12,288 entries build (1.3 s for 4,096); 16,384, 32,768, 49,152 and 65,536 entries fail:
  ```
  error[3-0026]: const evaluation exceeded the step limit
  6 |         t[i] = ((i * 7 + 3) % 251).downcast_truncate()
    |                  ^^^^^ const evaluation takes too many steps
  ```
  So the budget is about 60 to 80 steps per simple loop iteration. A 2,000,000-iteration scalar loop
  fails the same way (`q3_limit_steps.fe`).
- Consequence seen in question 4: packing 3,136 weights with a const fn works, packing 25,088 does not.
- A constant array costs 32 bytes of contract code per element whatever the element type (see 6b), so a
  256-entry `u8` table is about 8 KB of code. Pack tables into `u256` words if size matters.

## 4. A real layer on the EVM: WORKS WITH WORKAROUND

Run: `q4a.sh`, `q4b.sh <opt> <macs> [workspace]`, `q4_split.sh`, `q4_anvil.sh`, `q4d.sh` (784 x 32),
`q4_anvil784.sh`.

The layer (`layer/nn/src/lib.fe`): int8 weights, int8 inputs, int32 bias and accumulator, ReLU,
requantise by arithmetic shift, clamp to int8. Generic over shape:

```fe
pub fn dense<const IN: usize, const OUT: usize>(
    w: [[i8; IN]; OUT], b: [i32; OUT], x: [i8; IN], shift: i32,
) -> [i8; OUT]
```

Weights are deterministic pseudo-random int8 values from `gen_weights.py`, which also computes the
expected output with Python integers. Every variant below returns exactly the Python output
(196 x 16: `127 2 0 0 24 3 0 0 0 62 36 27 87 53 0 0`).

Kernel variants (all in the same file, all bit-identical):

- `dense`: weights as `[[i8; IN]; OUT]`, checked arithmetic (Fe default).
- `dense_unchecked`: same with `#[arithmetic(unchecked)]`.
- `dense_packed`: weights packed 32 per `u256` word as `(w + 128)` bytes, i32 accumulator, unchecked.
- `dense_packed_wide`: packed weights, `i256` accumulator (no sign extension after each operation).
- `pack`: a `const fn` that produces the packed form from the int8 matrix at compile time. For 196 x 16
  the generated file contains `pub const WP_CT: [[u256; 7]; 16] = ingot::pack(W)` and `static_assert`s
  that it equals the generator's packed literal; this compiles.

### 196 x 16 (3,136 multiply-adds), `-O 1` (the default)

| Variant | Runtime bytecode | Gas, pure function | Gas per multiply-add | Gas through CALL (in `fe test`) |
| --- | --- | --- | --- | --- |
| `Layer`, int8 constants, checked | 107,259 bytes | 719,173 | 229.3 | 1,279,837 |
| `Layer`, int8 constants, unchecked | (same contract) | 481,216 | 153.4 | 1,038,987 |
| `LayerPacked`, packed, i32 accumulator | 5,986 bytes | 653,776 | 208.5 | 853,464 |
| packed, i256 accumulator | (kernel only) | 475,488 | 151.6 | |
| `LayerSplit` + 2 weight contracts | 7,012 + 2 x 2,082 bytes | | | 738,374 |
| `LayerRows` + 2 row contracts | 5,632 + 2 x 1,957 bytes | | | 725,359 |

"Gas, pure function" is a `ctx.gas()` delta around the call inside a test contract. "Through CALL"
includes the caller's ABI encoding, the CALL, decoding, the layer and the return.

Real transaction on anvil (Osaka), `LayerPacked` (`q4_anvil.sh`): deployment 1,336,075 gas, deployed
code 5,986 bytes, `cast call` returns `[127, 2, 0, 0, 24, 3, 0, 0, 0, 62, 36, 27, 87, 53, 0, 0]`,
`cast estimate` and a mined `cast send` both report **749,311 gas** for one inference (includes the
21,000 base cost and the calldata for `int8[196]`). That is 239 gas per multiply-add all in.

The headline problem: **the straightforward version does not fit in a contract.** A constant `[i8; N]`
array costs 32 bytes of code per element, so 3,136 int8 weights become 100,352 bytes and the contract is
107,259 bytes, 4.4 times the 24,576 byte limit. `fe build` gives no warning and `fe test` passes (its
EVM allows 1 MiB of code). Packing the weights into `u256` words brings it to 5,986 bytes.

Optimisation levels (196 x 16): `-O 2` gives the same gas as `-O 1` to within 20 gas and identical
sizes. `-O 0` roughly doubles gas (packed 1,317,492; unchecked int8 946,748; checked 1,342,284) and
sizes are 108,842 and 8,396 bytes. Outputs were correct at all three levels; the spurious overflow
panic of issue #1853 did not appear.

Compile time (196 x 16, wall clock, includes process start): `fe test nn` 0.9 to 1.9 s,
`fe test onchain` (6 tests) 5.3 to 6.5 s, `fe build` of the two contracts 4.4 to 6.1 s,
`fe build` of the six split contracts 4.8 s.

### 784 x 32 (25,088 multiply-adds), `-O 1`

It compiles and computes the right output (`0 10 127 20 0 31 120 58 0 0 0 0 88 12 0 55 98 0 0 0 101 6 0 0 0 0 0 65 0 0 0 84`).

| Variant | Runtime bytecode | Gas, pure function | Gas per multiply-add | Gas through CALL (in `fe test`) |
| --- | --- | --- | --- | --- |
| `Layer`, int8 constants, checked | 838,620 bytes | 5,728,301 | 228.3 | test halts: EVM StackOverflow (see 6a) |
| int8 constants, unchecked | (same) | 3,822,423 | 152.4 | |
| `LayerPacked` | 31,944 bytes (over the limit) | 5,191,671 | 206.9 | 7,369,978 |
| packed, i256 accumulator | (kernel only) | 3,770,455 | 150.3 | |
| `LayerSplit` + `WeightsA`/`WeightsB` | 29,936 (over the limit) + 2 x 13,570 bytes | | | 6,630,427 |
| `LayerRows` + `RowsA`/`RowsB` | 20,791 + 2 x 12,971 bytes | | | 6,307,193 |

- Embedding in one contract is impossible in principle: 25,088 weights are 25,088 bytes even at one
  byte each, more than the 24,576 byte limit. `LayerPacked` is 31,944 bytes.
- Compile-time packing fails at this size: `pub const WP_CT: [[u256; 25]; 32] = ingot::pack(W)` gives
  `error[3-0026]: const evaluation exceeded the step limit` (log: `gen/q4c_nn_ctpack_fail.log`,
  reproduce with `q4c.sh ctpack`). The
  generator has to emit the packed literal.
- Alternative tested: weights in separate contracts (`layer/onchain_split/src/lib.fe`). Two designs:
  - `LayerSplit`: two weight contracts each return half the matrix (`[[u256; 25]; 16]`) from one
    STATICCALL. Correct, but `LayerSplit` itself is 29,936 bytes, over the limit, because the compiler
    emits unrolled word-by-word copy code for the large arrays (3,017 MSTORE instructions).
  - `LayerRows`: one STATICCALL per output neuron returning one packed row (`[u256; 25]`), then the
    shared `row_packed_wide` kernel. 20,791 bytes, fits (3.7 KB of headroom once deployed).
- Real transactions on anvil (Osaka) for the `LayerRows` design (`q4_anvil784.sh`): deployment gas
  2,854,883 (`RowsA`) + 2,854,967 (`RowsB`) + 4,547,782 (`LayerRows`, 20,855 bytes deployed including two
  immutable addresses); `cast call` returns the expected 32 values; `cast estimate` and a mined
  `cast send` both report **4,451,886 gas** for one 784 x 32 layer, 177 gas per multiply-add all in.
  That is under the Osaka per-transaction cap of 16,777,216 gas.
- The in-test CALL figures (6.3M to 7.4M) are higher than the real transaction because the test
  contract's own ABI encoding of a 784-element array is very expensive (same unrolled copy code).
- Compile time (wall): `fe test nn` 2.5 s; `fe test onchain` 47.8 s and `fe build` 44.6 s (dominated by
  the 838 KB contract); `fe test onchain_split` 26 s and `fe build` of its six contracts 26.5 s.

Not measured: weights passed as calldata (projection: 25,088 non-zero calldata bytes cost about
400,000 gas at 16 gas per byte, plus a hash check, so it would also work for one layer); EXTCODECOPY
from a raw data contract (the API exists: `create_raw`, `ctx.extcodecopy` into a `MemBuffer`, but it
needs unsafe memory code and a different weight accessor on the native side).

## 5. One source, two backends: WORKS

Run: `q5.sh`, `q5b.sh`.

Layout: workspace `layer/` with `nn` (pure code and weights, no EVM imports), `onchain` (depends on `nn`,
contracts), `cli` (depends on `nn`, `pub fn main() -> i32`). Dependencies inside a workspace are
`[dependencies] nn = true`.

Native `main` (`layer/cli/src/lib.fe`) prints the outputs for the embedded input and then for every
196-byte sample on stdin. Printing is one character at a time (`std::io::{Read, Write, host, read_char,
write_char}` inside `with (Read = host(), Write = host()) { ... }`); there is no integer formatting at
26.5.1, so `print_int` is hand-written (20 lines).

```
$ fe build --backend native -O 1 --out-dir out-native-1 --ingot cli .
Wrote out-native-1/cli                      (2.4 s wall; ELF 64-bit x86-64, 53,104 bytes, links with cc)
$ out-native-1/cli < gen/q5/samples.bin
127 2 0 0 24 3 0 0 0 62 36 27 87 53 0 0
0 37 0 0 28 0 0 85 0 99 75 19 67 0 0 0
...
```

Bit-identical check on 21 inputs (the embedded one plus 20 random int8 vectors, negative inputs
included): Python integer reference, native at `-O 0`, `-O 1`, `-O 2`, and the EVM contract
(`LayerPacked` on anvil, queried with `cast call`) all produce the same 21 lines:

```
bdbc7fc707355d753a3012352f2b9597  gen/q5/ref_all.txt
bdbc7fc707355d753a3012352f2b9597  gen/q5/native_O0.txt
bdbc7fc707355d753a3012352f2b9597  gen/q5/native_O1.txt
bdbc7fc707355d753a3012352f2b9597  gen/q5/native_O2.txt
bdbc7fc707355d753a3012352f2b9597  gen/q5/evm.txt
```

The native binary also cross-checks three kernels against each other per sample (exit status 3 on any
disagreement; it exited 0). A 10,000-sample set: all 10,000 native outputs identical to the Python
reference, 5.8 s wall for 30,000 layer evaluations (three kernels per sample, measured while another
compile was running on the machine). The 784 x 32 workspace also builds natively (3.6 s, 110,448 bytes)
and prints the expected 32 values. `u256` and `i256` arithmetic works natively, and `usize` being 64-bit
natively and 256-bit on the EVM caused no difference.

`fe test --backend native` exists and works on pure ingots:

```
$ fe test --backend native -O 1 nn cli
PASS  nn  infer_matches_python_reference        (and 3 more nn tests)
PASS  cli cli_kernels_agree
test result: ok. 5 passed; 0 failed             (2.2 s wall; same at -O 0)
```

It does not work on an ingot that contains a contract:

```
$ fe test --backend native onchain
Failed to emit test native: unsupported: while lowering `contract_init_abi_Layer` at bb0[1]:
target `x86_64-unknown-native` does not support instruction `sonatina_ir::inst::evm::EvmCallValue`
```

So shared code must live in its own ingot, which is the layout used here. Native limits that touch this
idea: no file I/O and no argument-to-integer parsing at 26.5.1, so the test set has to arrive on stdin
as raw bytes; a failed assertion is a bare trap.

## 6. Blockers and surprises

Run: `q6.sh`, `q6b.sh`, `q6c.sh`, `q6_stackoverflow.sh`. No compiler crash (Rust panic) was hit anywhere in
the spike, and issue #1853 did not show up. Timings in this section were taken while other jobs ran on
the machine at times; read them as plus or minus 20 percent.

### 6a. Miscompile: EVM StackOverflow when one large array is used in two messages

Repro: `q6_array_twice.fe` (about 30 lines, standalone). A `[i8; 784]` constant is passed by value in two
`evm.call` messages inside one test function.

```
$ fe test -O 1 q6_array_twice.fe
FAIL  [0.0016s] same_array_in_two_calls
    Test halted: StackOverflow (gas: 1000000000)
```

It compiles without any diagnostic and fails at run time. The opcode trace of the original occurrence
(the 784 x 32 `gas_contract_call` test, `gen/q6_dbg/onchain__gas_contract_call.evm_trace.txt`) shows the
cause: the array copy is emitted as straight-line code, one `MLOAD, DUP1, PUSH3 addr, MSTORE` group per
element, and every loaded element stays on the EVM stack for the second copy, so the stack climbs by one
per element until it passes 1,024:

```
pc=27572 op=0x51 stack=1022 ...
pc=27578 op=0x52 stack=1024 ...
pc=27586 op=0x62 stack=1024 ...
trace result: Ok(Halt { reason: StackOverflow, gas_used: 1000000000 })
```

A single use of a 600 or 1,100 element array in a message passes (`q6_array_copy_stack.fe`). Avoid by
never passing a large array by value twice in a function, or by keeping message arrays small (pack the
input into `u256` words).

### 6b. Compile time explodes with large fixed arrays in messages

- `q6_array_copy_stack.fe` (37 lines, one contract, arrays of 600 and 1,100 `u8` in two messages):
  `fe test -O 1` takes **110.5 s**; `fe test -O 0` did **not finish in 240 s** (killed by `timeout`; an
  earlier unbounded run was at 100 percent CPU and 1.1 GB resident memory after 218 s).
- `q6_array_twice.fe` (one 784-element array): 49 s at `-O 1`.
- The 784 x 32 workspace: 26 to 48 s per `fe test` or `fe build`, against 4 to 6 s for 196 x 16.

The same code with 196-element arrays compiles in seconds, so the cost grows much faster than linearly
in the array length.

### 6c. Code size traps (no warning from the compiler)

- A constant array costs **32 bytes of code per element, whatever the element type**
  (`q6_const_size/`, `q6c.sh`): 1,024 x `i8` gives 32,867 bytes of runtime code, 1,024 x `u8` gives 32,848,
  32 x `u256` (the same 1,024 bytes of information) gives 1,100, the same contract without a table is 55.
- By-value copies and ABI decoding of large arrays are unrolled, about 9 bytes of code per element per
  copy. This is why `LayerSplit` is 29,936 bytes at 784 x 32 and why `LayerRows`, whose logic is tiny, is
  still 20,791 bytes with an `int8[784]` argument.
- `fe build` does not check the 24,576 byte limit and `fe test` runs with a 1 MiB limit, so an
  undeployable contract passes every test. Check `out/<Contract>.runtime.bin` (hex length / 2) in CI.

### 6d. CTFE step limit

1,000,000 steps per evaluation, not configurable (question 3). Hit twice: a 16,384-entry table, and
compile-time packing of 25,088 weights. Error text: `error[3-0026]: const evaluation exceeded the step limit`.
Splitting the work across several constants was not tested.

### 6e. `black_box` only accepts scalars, and says so late

Repro: `q6_black_box.fe`. `fe check` passes with no output; `fe test` then fails in code generation with
no source location:

```
Failed to emit test Sonatina: MIR lowering failed: unsupported while lowering `t`:
`core::hint::black_box` requires a scalar argument, such as an integer or `bool`, but found `[i8; 4]`
```

Workaround used in the gas tests: copy the array and launder one element (`x[0] = black_box(x[0])`).

### 6f. Smaller surprises

- The optimiser can replace a regular constant table by its formula: a table filled with
  `(i * 37 + 11) % 256` and read at a runtime index compiled to `i * 0x25 + 0x0b & 0xff` with no table in
  the bytecode (72 bytes of runtime code). Harmless, but it makes size experiments with regular test data
  misleading; use irregular data.
- Structs are not `Copy` unless you implement it; operators take `own self`, so `a * b` moves `a`
  (`error[16-0002]: move conflict`, clear message).
- `as` only does provably lossless casts; `usize as i32` is an error that names the alternatives.
- A raw integer selector (`#[selector = 0x01]`) leaves the arm out of the ABI JSON with a warning
  ("selector signature is unknown"); use `sol("name(types)")` for anything called with cast.
- In-test gas through `evm.call` overstates real cost for large array arguments (6.3M in the test
  against 4.45M for the real transaction at 784 x 32), because the test contract's own encoding is
  expensive. Use anvil plus `cast estimate` for numbers that matter.
- `fe test` progress output uses carriage returns; captured logs need `tr '\r' '\n'` or lines look cut.
- Native: no integer formatting, no file I/O, contracts cannot share an ingot with native tests (question 5).

---

## Overall verdict

**Feasible on Fe 26.5.1.** Every element of the idea was demonstrated with running code: shapes and
fixed-point scale checked by the type system (including `Q<{A + B}>` and `[T; { R * C }]`), lookup tables
and weight packing computed by `const fn`, a dense layer running in the EVM with the exact Python
reference output, and the same source built as a native executable that agrees with the EVM
bit for bit on every input tried (21 inputs across both backends, 10,000 more native against Python).

What has to change in the design:

1. **Weights cannot be plain `[[i8; IN]; OUT]` constants on-chain.** They cost 32 bytes of code each
   (107 KB for 196 x 16). Pack 32 weights per `u256` word. For layers up to a few thousand weights the
   packing can itself be a `const fn` (a good showcase); for larger layers the CTFE step limit forces
   the generator script to emit packed literals.
2. **A layer with more than about 17,000 weights cannot live in one contract** (24,576 bytes minus
   roughly 7 KB of code, at one byte per weight). 784 x 32 needs separate
   weight contracts and one STATICCALL per output row (measured: three contracts of 12,971, 12,971 and
   20,855 bytes, 4,451,886 gas per layer execution on anvil). A 196 x 16 layer fits in one 5,986 byte
   contract at 749,311 gas.
3. **Keep large arrays out of message signatures and out of by-value copies.** They inflate code size
   (about 9 bytes per element per copy), compile time (minutes past roughly 1,000 elements) and can
   trigger the StackOverflow miscompile. Projection, not measured: passing the input as packed words
   (`[u256; 25]` for 784 bytes) instead of `int8[784]` should shrink `LayerRows` well below 20 KB and cut
   calldata and decoding gas.
4. **Budget about 150 to 230 gas per multiply-add** with the kernels written here (150 with an `i256`
   accumulator and unchecked arithmetic, 229 with default checked `i32`). "Re-run one layer on-chain"
   is affordable for MNIST-sized layers (0.75M to 4.5M gas); re-running a whole network of that size in
   one transaction is not attractive, so the on-chain part should stay a single layer or a single
   neuron (784 multiply-adds, about 120,000 to 140,000 gas by the same rate: projection).
5. **Put the shared code in its own ingot** (`nn`), with the contract and the native CLI as separate
   ingots depending on it. The native side reads its test set from stdin as raw bytes.

Biggest remaining risk: **the compiler's handling of large arrays**. The three problems found (unrolled
copies, the StackOverflow miscompile, compile times of minutes) all sit exactly where a neural network
puts its data, the 784-input layer contract has only 3.7 KB of headroom under the size limit, and the
workarounds were found by experiment, not from documentation. A second or third layer, a bigger input,
or the next compiler release could move any of these. Second risk: gas per multiply-add is several
times what hand-written EVM code would need (projection, no hand-written baseline was measured), which
weakens any "efficient on-chain inference" claim; the entry should be pitched on type safety and
bit-identical dual-target execution, with gas reported honestly.

Not tested: convolution layers, more than one layer on-chain, weights as calldata, EXTCODECOPY data
contracts, real MNIST weights and accuracy, `isize`, splitting CTFE work across several constants,
deployment to a public testnet (needs Osaka opcodes).
