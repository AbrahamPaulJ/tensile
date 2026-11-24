# The Tensile kernel as a library

`ingots/tensile` was written for one job, integer inference with a dispute, but most of it is
general numeric machinery that any Fe project doing fixed-point arithmetic on the EVM can use:
values whose binary scale is a const generic, int8 vectors packed 32 to a word, a dot product over
packed limbs, a compile-time overflow proof, and an effect for instrumentation. This guide is for
an Fe builder who wants those pieces without the neural network.

Everything below was run on Fe 26.5.1 (the entry's release) and again on 26.7.0, in WSL2 Ubuntu.
Wall times were taken while other builds ran on the same machine, so read them as upper bounds.

| Section | What |
|---|---|
| [1](#1-from-an-empty-directory) | From an empty directory to a tested ingot that depends on the kernel |
| [2](#2-how-to-depend-on-it) | Path, git and version dependencies: what works on 26.5.1 |
| [3](#3-the-types-and-functions) | Every public type and function, with the smallest use of each |
| [4](#4-the-scale-arithmetic) | The rules the types enforce, and `qmath` for the rest |
| [5](#5-the-overflow-proof) | How a compile-time overflow proof is written, and its limits |
| [6](#6-worked-examples-that-are-not-neural-networks) | Three DeFi contracts built on it, their compile errors and their gas |
| [7](#7-fe-features-it-leans-on) | The Fe features the library depends on |
| [8](#8-relation-to-fe-core-and-std) | What it takes from Fe's `core`, and when `core::fixed` is the better tool |
| [9](#9-what-this-library-is-not) | Honest limits |
| [10](#10-api-documentation) | The generated API reference |
| [11](#11-pitfalls-met-while-writing-the-examples) | Compiler behaviour that surprised us |

## 1. From an empty directory

`lib-examples/walkthrough.sh` does exactly this in a new temporary directory and reads the two
files below from this page, so the text and the test cannot drift apart.

```
mkdir quickstart-demo && cd quickstart-demo
cp -r <tensile checkout>/ingots/tensile tensile     # any copy of the ingot will do
fe new quickstart                                     # an ingot with a sample Counter contract
# replace quickstart/fe.toml and quickstart/src/lib.fe with the two files below
fe check quickstart
fe test quickstart
fe test --backend native quickstart
```

<!-- walkthrough: quickstart/fe.toml -->
```toml
[ingot]
name = "quickstart"
version = "0.1.0"

[dependencies]
tensile = { path = "../tensile" }
```

<!-- walkthrough: quickstart/src/lib.fe -->
```fe
//! A weighted index of four readings, each in [0, 1).
use tensile::{Activations, I32_MAX, Q, Row, neuron, neuron_bound}

/// Four readings in [0, 1): bytes 0..127 with 7 fractional bits.
pub type Readings = Activations<4, 7>

/// The weight of each reading, 6 fractional bits (64 is 1.0).
pub const WEIGHTS: Row<4, 6> = Row::pack([64, 32, -16, 8])

/// The offset, at the scale of a weight times a reading: 6 + 7 = 13 bits.
pub const OFFSET: Q<13> = Q::from_raw(-1024)

// No input can overflow the index: the compiler checks it.
static_assert(neuron_bound(row: WEIGHTS, bias: OFFSET) <= I32_MAX)

/// `OFFSET + sum of WEIGHTS[i] * x[i]`, with 13 fractional bits.
pub fn index(_ x: Readings) -> Q<13> {
    neuron(row: WEIGHTS, x, bias: OFFSET)
}

#[test]
fn every_reading_at_one_half() {
    // (1 + 0.5 - 0.25 + 0.125) * 0.5 - 0.125 = 0.5625, which is 4608 / 8192.
    let x = Readings::pack([64, 64, 64, 64])
    assert!(index(x).raw() == 4608)
}
```

Now make the classic mistake: write the offset in the units of a reading,
`pub const OFFSET: Q<7> = Q::from_raw(-16)`. `fe check quickstart` refuses it:

```
error[8-0000]: type mismatch
   │     neuron(row: WEIGHTS, x, bias: OFFSET)
   │                                   ^^^^^^ expected `Q<13>`, but `Q<7>` is given
```

Measured (`bash lib-examples/walkthrough.sh`, 2026-11-24):

| Step | Fe 26.5.1 | Fe 26.7.0 |
|---|---|---|
| `cp -r` the ingot, `fe new quickstart` | 0.7 s | 0.3 s |
| `fe check quickstart` | 3.1 s | 3.6 s |
| `fe test quickstart` (EVM): 1 passed | 2.1 s | 4.4 s |
| `fe test --backend native quickstart`: 1 passed | 2.4 s | 5.0 s |
| the offset at `Q<7>`: refused with the message above | 0.4 s | 1.0 s |
| `tensile = "0.1.0"` instead of the path: refused (section 2) | 0.0 s | 0.1 s |
| `tensile` as a git dependency on github.com/AbrahamPaulJ/tensile at `67d6ee4`: `fe check` (clones), again, `fe test`: 1 passed | 11.5 s, 2.5 s, 2.5 s | not run |

Nothing else is needed: no package manager, no network for the path form, no file outside the
two directories.

## 2. How to depend on it

| Form | `fe.toml` | Fe 26.5.1 |
|---|---|---|
| Path | `tensile = { path = "../tensile" }` | Works. What every ingot in this repository uses, and section 1 |
| Git | `tensile = { source = "https://github.com/AbrahamPaulJ/tensile", rev = "<full commit>", path = "ingots/tensile" }` | Works. Tried with `rev = "67d6ee4dd0853bd3a442815fc5da99abd88ea02b"` (the public `main` of 2026-11-23): the first `fe check` cloned it in 11.5 s, later ones took 2.5 s, and the quickstart's test passed (`GIT=1 GIT_REV=.. bash lib-examples/walkthrough.sh`). `rev` must be given; a branch name was not tried. Pin a full commit |
| Version only | `tensile = "0.1.0"` | Refused outside a workspace: `Dependency 'tensile' ... uses name-only lookup outside a workspace`. Inside a workspace it means "the member of that name" (`qmath = true` in `lib-examples/`) |
| Registry | none | Fe 26.5.1 has no package registry and no `fe add` or `fe publish`; `fe --help` lists `build, check, dev, doc, tree, fmt, test, bench, new, completion, lsif, root, lsp, scip` |

The git keys are `source` (a URL), `rev` (required) and `path` (the ingot inside the repository),
read from Fe's own manifest parser (`crates/common/src/config/dependency.rs` at v26.5.1). The ingot
has no dependencies of its own beyond Fe's built-in `core`, so nothing else has to be fetched.

## 3. The types and functions

All of these are exported from `tensile`'s root (`ingots/tensile/src/lib.fe`). The ingot imports
nothing from `std`, so everything runs in a contract, in a native executable and (Fe 26.7.0) in
WebAssembly.

### `Q<F>`: one int32 at a binary scale

The real value is `raw / 2^F`. `F` is a `usize` const generic, so `Q<8>` and `Q<0>` are different
types and cannot be mixed by accident. `Copy`.

| Item | What |
|---|---|
| `Q::from_raw(raw: i32) -> Q<F>` | the value whose representation is `raw` (`const fn`) |
| `q.raw() -> i32` | the representation |
| `q.shifted<SHIFT>() -> Q<{ F - SHIFT }>` | drop `SHIFT` fractional bits, an arithmetic shift (floor). `SHIFT > F` does not compile: the scale `F - SHIFT` would be negative |
| `q.clamped() -> i8` | `clamp(raw, 0, 127)`, at the same scale (the ReLU of the kernel) |

```fe
let price: Q<8> = Q::from_raw(51200)          // 200.0
let whole: Q<0> = price.shifted<8>()           // 200
```

`Q` also has `+`, `-`, `*`, `==` and `<` as operators, with the same scale rules (section 4).

### `Packed<N>`: N int8 values, 32 per word

Value `i` is byte `i % 32` of word `i / 32`, most significant byte first, two's complement; the
unused bytes of the last word are zero. A `Packed<N>` constant costs one byte of contract code per
value, where an `[i8; N]` constant costs 32. `Copy`, `Eq`.

| Item | What |
|---|---|
| `Packed::pack(values: [i8; N])` | packs; the compiler does it for a constant |
| `Packed::from_words(words: [u256; word_count(N)])` | wraps words already packed; fails if a padding byte is not zero |
| `Packed::padded_with_zeros(words) -> bool` | the same check as a value, to revert with your own error |
| `p.get(i) -> i8` | value `i` |
| `p.words() -> [u256; word_count(N)]` | the packed words, the form a message or a storage slot carries |
| `word_count(n) -> usize` | `ceil(n / 32)` |

```fe
const TAPS: Packed<5> = Packed::pack([1, -1, 127, -128, 0])   // one word in the code
```


### `Activations<N, F>`: N values in 0..127 at scale F

The input of the dot product: `N` non-negative bytes with `F` fractional bits. Outside a network it
is any vector of small non-negative readings: features in [0, 1) at 7 bits, a histogram, a window
of quantised samples. `Eq`.

| Item | What |
|---|---|
| `Activations::pack(values: [i8; N])` | fails if a value is negative (at compile time for a constant) |
| `Activations::well_formed(words) -> bool` | every byte in 0..127 and zero padding: check untrusted calldata with this |
| `Activations::from_words(words)` | wraps checked words; fails like `pack` |
| `a.words() -> [u256; word_count(N)]` | the form a message carries |
| `a.get(i) -> i8` | value `i` |

```fe
if !Activations<16, 7>::well_formed([word]) { revert_error(Malformed { word }) }
let x = Activations<16, 7>::from_words([word])
```

### `Row<IN, FW>` and `Weights<OUT, IN, FW>`: signed int8 coefficients at scale FW

A `Row` is `IN` int8 weights (any sign) with `FW` fractional bits, packed; `Weights` is `OUT` rows.
Outside a network a `Row` is a coefficient vector: scorecard points, filter taps, index weights.
`Row` is `Copy` and `Eq`.

| Item | What |
|---|---|
| `Row::pack(weights: [i8; IN])`, `Row::from_words(words)` | as for `Packed` |
| `row.words()`, `row.get(i)` | the packed words, weight `i` |
| `Weights::new(rows: [Row<IN, FW>; OUT])`, `w.row(j)` | a matrix from rows, row `j` |
| `w.is_packed_from(weights: [[i8; IN]; OUT]) -> bool` | in a `static_assert`, the compiler repeats the packing and compares |

### `Bias<N, F>` and `Accumulators<N, F>`: int32 vectors at scale F

`Bias::from_raw([i32; N])` and `b.at(j) -> Q<F>`. `Accumulators` adds `values()`, `argmax()`
(lowest index wins a tie; `N > 0` is a `where` clause), `==`, and `requantise<SHIFT>()`, which
shifts and clamps every value into `Activations<N, { F - SHIFT }>`.

### The dot product: `neuron` and `dense`

```fe
pub fn neuron<const IN: usize, const FW: usize, const F: usize>(
    row: Row<IN, FW>, x: Activations<IN, F>, bias: Q<{ F + FW }>,
) -> Q<{ F + FW }> where dot_fits_i32(IN)
```

`bias + sum of row[i] * x[i]`. The two vectors must have the same length, the bias must be at the
scale of the products (`F + FW`), and the result is at that scale: all three are type checks. The
sum of products is computed 32 lanes per word in unchecked arithmetic, and the `where` clause
`dot_fits_i32(IN)` (`IN * 128 * 128 <= 2^31 - 1`, so `IN <= 131,071`) is the proof that it cannot
overflow for any values. Adding the bias is checked. `dense(weights, x, bias)` is `neuron` once per
row and returns `Accumulators<OUT, { F + FW }>`.

The kernel's lower-level `dot` (two `Packed<N>`, any sign on one side) is `pub(ingot)`, not
exported: a signed-by-signed product is not available outside the ingot today.

### The overflow proof: `neuron_bound`, `worst_case`, `I32_MAX`

`neuron_bound(row, bias) -> u256` is the largest absolute value `neuron` can reach for any input
in 0..127: `|bias| + 127 * max(P, N)`, with `P` and `N` the sums of the positive and of the
negative weights. It also bounds every partial sum. `worst_case(weights, bias)` is the largest over
a layer. Both are `const fn`, computed a word at a time, and meant for `static_assert`:

```fe
static_assert(neuron_bound(row: POINTS, bias: BASE) <= I32_MAX)
```

### `Probe` and `Blind`: observation as an effect

`Probe` is a trait with one generic method, `hidden<N, F>(mut self, layer, activations)`. A
function that declares `uses (probe: mut Probe)` can be run with any observer the caller provides
(`var blind = Blind {}` then `with (Probe = blind) { forward(x) }`, as `ingots/model` does);
`Blind` observes nothing. In the kernel it
lets one forward pass serve the classifier, the tracer and the auditor. It is specific to
`Activations`, so for other data it is a pattern to copy rather than a trait to implement.

## 4. The scale arithmetic

Write `Q<F>` for a value with `F` fractional bits. The kernel's signatures enforce:

| Operation | Rule | Where |
|---|---|---|
| product of a weight and an activation | `Q<FW>` times `Q<F>` is `Q<{ F + FW }>` | `neuron`, `dense` |
| bias of a product | must be `Q<{ F + FW }>` exactly | `neuron`'s `bias` parameter |
| drop precision | `Q<F>.shifted<S>()` is `Q<{ F - S }>`, floor; `S > F` does not compile | `Q::shifted` |
| requantise | `Accumulators<N, F>.requantise<S>()` is `Activations<N, { F - S }>` | `requantise` |
| compare | `==` only between equal types | `Eq` impls |

That is all a network needs. A price filter or a rate curve also adds, subtracts, multiplies and
compares single values, and those are in a companion ingot, `lib-examples/ingots/qmath` (about 100
lines), with the same rules written as signatures:

| `qmath` | Signature | Rule |
|---|---|---|
| `add`, `sub` | `(Q<F>, Q<F>) -> Q<F>` | same scale in, same scale out |
| `mul` | `(Q<FA>, Q<FB>) -> Q<{ FA + FB }>` | scales add; nothing is dropped until you call `shifted` |
| `scaled_by` | `(Q<FA>, Q<FB>) -> Q<FA>` | `mul` then `shifted<FB>`, for generic code (section 11) |
| `lt`, `le`, `gt`, `ge`, `same`, `min`, `max` | `(Q<F>, Q<F>)` | same scale only |
| `from_int<F>(i32)` | `-> Q<F>` | `whole * 2^F`, checked |
| `product_fits_i32(max_a, max_b)` | `-> bool` | for overflow proofs |

```fe
use qmath::{add, mul, sub}
// average: Q<8>, tick: Q<0>, ALPHA: Q<8>
let next: Q<8> = add(average, mul(sub(tick, average.shifted<8>()), ALPHA))
```

The kernel itself also implements the operators for `Q`: `a + b` and `a - b` at one scale,
`a * b` with `Q<FA> * Q<FB>` giving `Q<{ FA + FB }>`, `==` and the comparisons at one scale, all
checked. They live in the kernel because only the ingot that defines `Q` may implement
`core::ops::Add` for it: from another ingot, Fe 26.5.1 and 26.7.0 both say
`error[5-0000]: external trait cannot be implemented for external type`. The example contracts were
written before the operators were added and use the `qmath` functions; both forms obey the same rules.

```fe
let p: Q<10> = Q<4>::from_raw(24) * Q<6>::from_raw(-144)   // scales add
```

All arithmetic on `Q` is ordinary checked `i32` arithmetic: an overflow reverts (on the EVM) or
traps. The `static_assert`s of section 5 are how you show that it cannot happen.

## 5. The overflow proof

Three patterns, all checked by the compiler:

1. **A `where` clause on a const fn of the sizes.** `neuron` requires `dot_fits_i32(IN)`, so a dot
   product that could overflow for some input does not compile, whatever the values. This is what
   lets the inner loop run unchecked. Use it when the bound depends only on sizes.
2. **`static_assert` of a bound computed from the constants.** `neuron_bound` and `worst_case`
   read the actual weights. The scorecard in section 6 proves its score fits this way.
3. **A closed-form bound of your own, as a `const fn`.** For a recurrence (an average) or a
   formula (a rate curve), write the worst case as arithmetic on the parameters and assert it:
   `ema_fits(max_sample, alpha_bits)` and `curve_fits(base, slope, jump, kink)` in
   `lib-examples/`. The argument for why the bound is a bound lives in the doc comment; the
   compiler checks the arithmetic, not the argument.

A failed `assert!` inside a `const fn` evaluated for a constant is also a compile error, so
constructors validate constants for free (`Activations::pack([-1])` does not compile).

**Limits.** Compile-time evaluation has a fixed budget of 1,000,000 steps per evaluation
(`step_limit: 1_000_000` in the compiler; no `fe.toml` key, attribute or flag changes it), and Fe
26.7.0 spends about 20% more steps on the same code than 26.5.1:

| Computation | Fe 26.5.1 | Fe 26.7.0 | Source |
|---|---|---|---|
| Iterations of a three-line `while` loop | 30,302 | 24,389 | `docs/upstream-recheck-26.7.0.md` |
| `worst_case` over `OUT x 784` weights, one repeated row (`lib-examples/step_limit.sh`) | 232 x 784 = 181,888 pass, 240 x 784 fail | 200 x 784 = 156,800 pass, 208 x 784 fail | measured 2026-11-24 |
| `worst_case` through the model generator (one literal per row) | 192 x 784 = 150,528 pass, 256 x 784 fail | not measured | `docs/fe-notes.md` section 4 |
| The same bound visiting weights one at a time | 8 x 784 = 6,272 pass | not measured | `docs/fe-notes.md` section 4 |
| `is_packed_from` (pack and compare) | 12 x 784 = 9,408 pass | not measured | `docs/fe-notes.md` section 4 |
| `core::bytes::PackedBytes::from_array` | 16,871 bytes | 15,370 bytes | `docs/upstream-recheck-26.7.0.md` |

Two more rules from the same notes: constants referenced from one evaluation are evaluated again
inside it, so their steps add up (split a large proof into one `static_assert` per piece), and
`for` cannot be used in a `const fn` (use `while`). The proofs of the examples in section 6 are
closed-form and cost a handful of steps.

## 6. Worked examples that are not neural networks

`lib-examples/` is a separate Fe workspace (so the shipped workspace and its bytecode are
untouched). Each example is its own ingot that depends on `tensile` by path and on `qmath`:

| Ingot | Contract | What it computes | Kernel pieces |
|---|---|---|---|
| `ema` | `TickEma` | exponential moving average of a Uniswap v3 tick, `Q<8>` | `Q`, `shifted`, `I32_MAX`, `qmath` |
| `score` | `Scorecard` | linear credit score: 16 features in [0, 1) times int8 points packed in the code | `Activations`, `Row`, `neuron`, `neuron_bound`, `qmath::ge` |
| `curve` | `RateCurve` | kinked borrow rate of a lending pool (Compound `JumpRateModel` shape) | `Q`, `shifted`, `qmath` |
| `plain` | `PlainTickEma`, `PlainScorecard`, `PlainRateCurve`, `WadRateCurve` | the same three with plain integers, and the curve with `core::fixed::Wad` | none |
| `deployed` | (tests) | every contract against its pure functions and against its twin | |

### What each one refuses to compile

`lib-examples/compile_fail/`, run by `lib-examples/check_compile_fail.sh` (7 cases: 6 that must
fail with the stated message, 1 control that must compile). Each is a mistake that compiles in
Solidity:

| Case | The mistake | Fe's message | What Solidity does with it |
|---|---|---|---|
| `ema_scale_mix.fe` | subtract the stored average (8 bits) from a tick (0 bits) | `expected Q<0>, but Q<8> is given` | compiles; the average drifts to a wrong value |
| `ema_too_many_bits.fe` | keep 12 fractional bits instead of 8 | `static assertion failed` (`ema_fits`) | 0.8: reverts at the extreme tick, so the oracle sticks when the price moves most; `unchecked` or `int32(..)`: wraps silently |
| `score_base_wrong_scale.fe` | the base score in points (4 bits), not points times features (11) | `expected Q<11>, but Q<4> is given` | compiles; every score is off by a constant |
| `score_threshold_wrong_scale.fe` | compare the score with a threshold in points | `expected Q<11>, but Q<4> is given` | compiles; accepts almost everyone |
| `curve_missing_shift.fe` | `base + slope * u` without the shift back | `expected Q<16>, but Q<28> is given` | compiles; a rate 4,096 times too large (the missing `/ 1e18`) |
| `curve_jump_overflow.fe` | a jump slope of 800% a year | `static assertion failed` (`curve_fits`) | 0.8: reverts at 100% utilisation, and every call that accrues interest with it |

### Tests

| Ingot | Tests | Backends |
|---|---|---|
| `qmath` | 4, plus 13 `static_assert`s | EVM, native |
| `ema` | 3 (the step against the plain formula over extreme ticks, 800 steps between the two extreme ticks without an overflow, the bound) | EVM, native |
| `score` | 4 (40 feature vectors against a plain loop, every packed point, both extremes of the bound, the threshold) | EVM, native |
| `curve` | 3 (every one of the 4,097 utilisations against the plain formula, in both of its forms; monotone with the jump at the kink; utilisation from token amounts) | EVM, native |
| `deployed` | 9 (each contract against its pure functions, its errors, and its plain twin on the same calls; `WadRateCurve` equal to `RateCurve` where utilisation is a whole number of 1/4096 and at most 34 units above elsewhere) | EVM only: a test that deploys a contract cannot run natively |

Fe 26.5.1 and 26.7.0 both pass all 23 EVM tests, all 14 native ones and all 7 compile-fail cases
(`lib-examples/check.sh`, `tools/check_fe267_lib.sh`).

### Gas: the typed version against plain Fe

`lib-examples/gas.sh`: real transactions on anvil (Osaka), each contract and its twin on the same
input, results required equal. "Execution" is the gas of the code alone, read from the opcode
trace, because a call with two `uint256` arguments pays the EIP-7623 calldata floor and its gasUsed
hides the code. Recorded in `lib-examples/runs/gas.json`.

| Call | Input | Typed (kernel) | Plain Fe | `core::fixed::Wad` |
|---|---|---|---|---|
| `TickEma.update(int32)` | second tick, 194,123 (slot already written) | 7,634 (28,862 tx) | 7,751 (28,979 tx) | |
| `Scorecard.score(uint256)` | one packed word of 16 features | 1,559 (22,943 tx) | 4,691 (26,075 tx) | |
| `RateCurve.borrowRate(uint256,uint256)` | 500 and 500 tokens: 50% | 472 (22,220 tx) | 472 (22,220 tx) | 939 (22,427 tx) |
| `RateCurve.borrowRate(uint256,uint256)` | 125 and 875 tokens: 87.5% | 494 (22,220 tx) | 494 (22,220 tx) | 944 (22,432 tx) |
| Runtime bytes, average / scorecard / curve | | 407 / 1,146 / 355 | 406 / 926 / 340 | 477 |

Execution gas, with the transaction's gasUsed in brackets; Fe 26.5.1, one run. Gas on anvil is
deterministic, and CI checks that a fresh run writes the same `runs/gas.json`.

What the numbers say:

- **The types cost nothing where the code is the same.** `RateCurve` and `PlainRateCurve` take the
  same execution gas to the unit, and the average costs 117 gas less typed than plain. `Q<F>` is a
  struct around an `i32`, and Sonatina compiles it away.
- **The formula matters more than the types.** A first version of `rate` computed both products
  every time (`rate_by_formula`, still in the ingot and tested): 706 and 701 gas in an earlier run
  of the same script, against the plain version's branch on the kink. The gap was the formula; with
  the same branch the typed version is equal.
- **The packed kernel is faster than the obvious plain loop**: the scorecard's 16 multiply-adds
  cost 1,559 gas through `neuron`, 4,691 as a byte-by-byte loop over the word. That is the kernel's
  limb arithmetic (`docs/fe-notes.md` M2.7), not the type system; a hand-written packed loop in
  plain Fe could match it. Its code is 220 bytes larger.
- **`core::fixed::Wad` costs about twice the gas** of the `Q` curve (939 against 472), for 18
  decimals of precision instead of 1/4,096 of utilisation. For token amounts that precision is
  usually what you want (section 8).

## 7. Fe features it leans on

| Feature | Where | Notes |
|---|---|---|
| Const generic structs | `Q<F>`, `Packed<N>`, `Row<IN, FW>`, ... | the scale and the length are part of the type |
| Arithmetic in const generic arguments | `Q<{ F + FW }>` in a parameter, `Q<{ F - SHIFT }>` in a return type | an underflowing `F - SHIFT` is a compile error, which is how `shifted` refuses to add precision |
| Struct fields sized by a `const fn` | `words: [u256; word_count(N)]` | |
| `where` clauses that call a `const fn` | `where dot_fits_i32(IN)`, `where SHIFT < 32`, `where N > 0` | turns a size condition into a type error |
| `const fn` and `static_assert` | every proof and every packed constant | the compile-time evaluator, with its step limit (section 5) |
| `assert!` in a `const fn` | `Activations::pack`, `Packed::from_words` | a bad constant fails to compile |
| `#[arithmetic(unchecked)]` and `#[inline(always)]` | the dot product | unchecked only where a `where` clause proves it safe |
| Effects with a generic trait method | `Probe`, `uses (probe: mut Probe)` | observation without a global |
| `Eq` with `const fn eq` and `ne` | `Packed`, `Row`, `Activations`, `Accumulators` | so `==` works in a `static_assert` |
| `pub(ingot)` | the packed words, `dot`, `lane_sum` | the representation stays private to the kernel |
| Ingots and path dependencies, workspaces | everything | section 2 |
| `msg`, `#[selector = sol(..)]`, `#[error]`, `revert_error`, `uses (mut field)` | the example contracts, not the kernel | the kernel has no EVM code at all |

## 8. Relation to Fe core and std

The kernel uses Fe's `core` where it has what is needed, and nothing from `std`:

| From Fe | Used for |
|---|---|
| `core::marker::Copy` | `Q`, `Packed`, `Row` |
| `core::ops::Eq` | equality of the vector types |
| `core::ops::WrappingMul` | summing the bytes of a word in one multiplication (`lane_sum`) |
| `core::num::IntWord` (`to_word`) | an `i8` as a `u256` while packing |
| `core::num::UnsignedAbs` | the magnitudes in `neuron_bound` |
| `downcast_truncate` (core's integer conversions) | narrowing inside the kernel |

What it does not use, and why:

- **`core::bytes::PackedBytes`** has the same byte layout as `Packed`, but its field is private and
  its only constructor, `from_array`, packs a byte at a time at compile time, which runs out of
  steps at 16,871 bytes (26.5.1) and 15,370 (26.7.0). `Packed` takes words directly and keeps a
  sign. See `docs/upstream-recheck-26.7.0.md`.
- **`core::fixed`** (`UFixed<D>`, `Fixed<D>`, `Wad`, `Ray`, `SWad`) is Fe's own scale-typed
  arithmetic, and it is in both 26.5.1 and 26.7.0. It is decimal (`raw / 10^D`), 256 bits wide,
  with operators that keep the scale and a 512-bit intermediate product, plus `exp`, `ln`, `sqrt`.
  Use it for token amounts, prices and rates in 18 or 27 decimals. Use `tensile`'s `Q<F>` when you
  want what `core::fixed` does not do: binary scales that **add** through a product so that the type
  shows where precision is dropped, int32 values that pack into int8 vectors, the packed dot
  product, and a compile-time bound on every intermediate. `WadRateCurve` in section 6 is the same
  curve both ways.

## 9. What this library is not

- **Not audited.** It has unit tests against plain loops at every word boundary, a NumPy reference
  on three backends, a Foundry fuzz suite and a mutation run for the dispute contracts
  (`docs/mutation-results.md`), but no external review.
- **int8 and uint8-oriented.** Packed values are int8; `Activations` are 0..127; `Q` is an int32.
  There is no packed int16, no 64-bit `Q`, and no signed-by-signed packed product in the public
  API. A value that does not fit 31 bits does not belong in a `Q`.
- **Dense arithmetic only.** Dot products of whole vectors; no sparse forms, no division, no
  square root or exponential (`core::fixed` has those for 18 decimals).
- **Binary scales only.** `F` is a number of bits, so 0.1 is not exact. Decimal amounts belong in
  `core::fixed`.
- **Some of it is specific to the model generator.** `Weights`, `Bias`, `Accumulators`,
  `requantise`, `argmax`, `worst_case` and `Probe` are shaped for layers; `clamped` is the ReLU.
  They work for other uses (a `Weights` is a matrix), but their names say "network". The model
  generator (`tools/gen_*.py`), the dispute ingot and the arbiter are not part of the library.
- **Sizes are fixed at compile time.** A vector length is a const generic; there is no dynamic
  length.
- **The proof is only as good as its statement.** A `static_assert` proves the arithmetic of the
  bound you wrote. If the bound's argument is wrong, the compiler will not notice; each example
  states its argument in a doc comment so that a reviewer can check it.

## 10. API documentation

`fe doc` generates a static site for the ingot: `bash tools/api_docs.sh <dir>` runs
`fe doc -o <dir> . static` on a copy of `ingots/tensile` (1.5 s) and drops the items whose
visibility is private (24 of 53: test constants and helpers), because the viewer shows every item it
is given. The result is five files, about 2 MB, most of it the viewer's JavaScript; links are
relative and routing is by hash, so it works under any path. The Pages workflow builds it with Fe
26.5.1 and publishes it at `/api/` (https://abrahampaulj.github.io/tensile/api/); nothing generated
is committed. `fe doc` on 26.5.1 and 26.7.0 has the same options (`static`, `json`, `bundle`,
`pages`, `serve`; `--self-contained` for one HTML file).

Two things the generated pages do not show, as of 26.5.1: the ingot's own `//!` documentation (the
root page lists the modules only), and the difference between `pub` and `pub(ingot)` (both are
"public" in `docs.json`, so `dot`, `lane_sum` and `sequence` appear though no other ingot can call
them).

## 11. Pitfalls met while writing the examples

| What | Message | Work-around |
|---|---|---|
| A generic scale expression is not simplified: `Q<{ R + U }>.shifted<U>()` is `Q<{ R + U - U }>`, not `Q<R>`, when `R` and `U` are parameters | `expected Q<const R: usize>, but Q<((const R: usize + const U: usize) - const U: usize)> is given` | write generic code with `qmath::scaled_by`, or use concrete scales, where `{ 16 + 12 - 12 }` is `16` |
| Operators cannot be added to `Q` outside the kernel | `error[5-0000]: external trait cannot be implemented for external type` | `qmath`'s free functions |
| A local variable or function named like a dependency ingot resolves to the ingot (`let curve = ..` with a `curve` dependency; `use score::score`) | `error[8-0030]: value is expected ... mod cannot be used as a value` | rename (`score_of`), or call through the path (`curve::rate(..)`) |
| An ingot whose tests deploy a contract cannot be tested natively | `Failed to emit test native: ... does not support instruction EvmSload` | keep pure tests in the ingot that defines the logic, and contract tests in a separate ingot (`deployed`), as the shipped workspace does with `tensile` and `onchain` |
| A two-argument call's gasUsed is the calldata floor | (none) | measure execution gas from the trace (`lib-examples/gas.sh`) |
