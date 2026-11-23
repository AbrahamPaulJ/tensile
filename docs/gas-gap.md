# Where the 1.24x goes

One layer 1 neuron costs 92,608 gas in Fe and 74,673 in the best Solidity found: 17,935 more (`docs/numbers.md` section 3).
Opcode traces of both put 68.7% of that on narrow-integer masks, 27.4% on moving two array arguments into memory, 3.9% elsewhere.
In the arbiter the same two causes return, joined by storage layout: two more slots per claim record.

Every figure below is either quoted from `docs/numbers.md` (section named) or printed by
`tools/gas_gap.py`, which traces the same transactions with `debug_traceTransaction` and asserts that
the opcodes of each trace sum to its execution gas. Anything else is marked as an estimate.

## What is compared, and what is fair to say

- The Solidity side is the best version found, not the natural one: 32 lanes written out in assembly
  (`evm/src/NeuronTwin.sol`, `neuron1Unrolled`). Against the natural Solidity loop Fe costs 0.78x
  (`docs/numbers.md` section 3).
- Fe has a form that costs 1.06x (`i256_unrolled`, 79,467 gas, section 5). It is 8x slower natively
  (38.03 s against 4.68 s for 10,000 samples, section 5), and the entry ships one kernel for the EVM,
  native code and wasm on purpose (`docs/fe-notes.md` M2.6). So 13,141 of the 17,935 is the price of
  the portable `u32` form, and this page shows that almost all of that price is masks.
- No setting closes the gap: Fe `-O 2` moves no gas figure by more than 0.03% (section 6).

- This page is about Fe 26.5.1, the release the entry is measured on. Later releases move the mask
  figures: the same neuron runs 1,904 `u32` masks and costs 54,588 gas on 26.5.1, 325 masks and 44,470
  gas on 26.6.0, whose range analysis removes most of them, and 1,500 masks and 46,035 gas on 26.7.0,
  where they return behind `BYTE` (`docs/upstream-recheck-26.7.0.md`, filed as argotorg/fe#2072).

## Causes

Layer 1 neuron, `NeuronCheck.neuron1` against `NeuronTwin.neuron1Unrolled`, gap 17,935 gas:

| # | Cause | Where it shows | Measured | Share | What would remove it |
|---|---|---|---|---|---|
| 1a | `u32` mask after each multiplication | `dot4`, 800 times | 4,800 | 26.8% | compiler: range analysis (both operands are below 256) |
| 1b | `u32` mask after each addition | `dot4`, `dot32`, 776 times | 4,656 | 26.0% | compiler: the same analysis carried through sums |
| 1c | other `u32` masks | 355 times, 350 of them `limb` truncations | 2,130 | 11.9% | compiler: demanded-bits analysis; or source (not tried) |
| 1d | `u64` masks at 14 gas each | the two accumulators, 54 times | 756 | 4.2% | compiler: a 6 gas encoding; removal needs the loop bound |
| | masks the twin runs (`int32` sign extension, twice) | | -16 | -0.1% | |
| 2 | two `[u256; 25]` arguments decoded and copied into memory | before the first multiplication | 4,911 | 27.4% | compiler: `CALLDATACOPY`, no fill, fewer copies |
| 3 | everything else | dot product +508, after it +190 | 698 | 3.9% | not broken down |
| | **total** | | **17,935** | **100%** | |

The arbiter (`Arbiter` against `evm/src/ArbiterTwin.sol`), each share against that call's own gap:

| Call | Gap | Storage layout | Masks | The rest |
|---|---|---|---|---|
| `claim` | 50,958 | 44,200 (86.7%): two more fresh slots | -166 | 6,924 (13.6%) |
| `challengeNeuron1` | 35,220 | 4,200 (11.9%): two more cold loads | 12,334 (35.0%) | 18,686 (53.1%), see cause 5 |
| `challengeNeuron2` | 8,541 | 4,200 (49.2%) | 510 (6.0%) | 3,831 (44.9%) |
| `challengeClass` | 4,323 | 4,200 (97.2%) | -160 | 283 (6.5%) |
| `finalize` | 4,723 | 4,200 (88.9%) | 0 | 523 (11.1%) |

## 1. Narrow-integer masks: 12,326 gas, 68.7%

The kernel (`ingots/tensile/src/dot.fe`) multiplies bytes held in `u32` limbs under
`#[arithmetic(unchecked)]`. After every `u32` multiplication or addition the compiler emits
`PUSH4 0xffffffff AND` (6 gas); after every `u64` one, `PUSH0 NOT PUSH1 0xc0 SHR AND` (14 gas). This is
finding 14 of `docs/fe-notes.md` M2.10, now counted in a real transaction: 1,985 masks executed for
12,342 gas, rows 1a to 1d above (`gas_gap.py`, "Layer 1 neuron: Fe as shipped", the "masks executed"
line). The twin runs 16 gas of masks, so the difference is 12,326.

- **1a and 1b are provably redundant.** Every operand of the 800 multiplications is either `x & 0xff`
  or `a >> 24` with `a: u32`, so it is below 256, a product is at most 65,025, and the 32 products of a
  word sum to at most 2,080,800, far below 2^32. An interval analysis over `AND`, `SHR`, `MUL` and `ADD`
  proves all 1,576 masks dead: 9,456 gas, 52.7% of the gap.
- **1c is mostly asked for by the source.** `limb` truncates a word to `u32` 16 times per word of
  input; the compiler already omits the two that follow `>> 224` and emits the other 14 (350 in 25
  words). Three of a limb's four lanes are then cut down by `& 0xff` anyway, so only the top lane
  (`a >> 24`) needs the truncation. Writing that lane as `(a >> 24) & 0xff` would make every truncation
  mask dead for a demanded-bits pass. Not tried.
- **1d cannot be removed locally.** The accumulators stay below 2^64 only because the loop runs 25
  times, which the source proves with a `where` clause the optimiser does not read. But the encoding
  could be cheaper: `PUSH8 0xffffffffffffffff AND` is 6 gas and 10 bytes where the emitted sequence is
  14 gas and 6 bytes. At 54 masks that is 432 gas (estimate from opcode prices: 54 x 8).

With the masks set aside the portable kernel is level with the assembly. From the first to the last
`MUL` executed, Fe spends 45,207 gas, 12,236 of it on masks, which leaves 32,971 against the twin's
32,463 (`gas_gap.py`, "Layer 1 neuron by phase"). That 32,971 includes the `128 * sum(x)` correction
the sign-flip form needs. The `i256_unrolled` form, which has no masks, spends 32,239 there: 224 gas
less than the twin.

## 2. Array arguments: 4,911 gas, 27.4%

Before the first multiplication Fe spends 8,889 gas outside masks and the twin 3,978. The twin reads
its two `uint256[25] calldata` arguments in place. Fe moves them into memory, in three steps that the
trace shows (`gas_gap.py`, "arrays" line and "Loops before the first MUL"):

| Step | Opcode shape | Count | Gas |
|---|---|---|---|
| fill the array with a first value | `DUP1 PUSH MSTORE` | 107 words | 1,280 |
| decode loop, one word per turn, with a checked add on the offset | 27 opcodes around `CALLDATALOAD .. MSTORE` | 2 loops | 4,360 (2,180 each, about 87 per word) |
| copy a word from memory to memory | `PUSH MLOAD PUSH MSTORE` | 59 words | 706 |

That is 6,346 gas for 50 words. One `CALLDATACOPY` per array is 3 + 3 x 25 = 78 gas (estimate from the
opcode price; memory expansion is paid either way). The same 2,180 gas loop appears once in `claim`
and twice in `challengeNeuron1`. It is the whole residue of Fe's best form too: with `i256_unrolled`
the gap of 4,794 is +4,881 before the first multiplication, -224 in the dot product and +137 after.

To be fair to Fe: its validation loop is the cheaper one, 1,650 gas against the twin's 3,348, because
the twin pays a bounds check on every `x[k]`.

## 3. Everything else in the neuron: 698 gas, 3.9%

Outside masks, the dot product costs Fe 508 gas more than the twin and the code after it 190 more.
Not broken down further. By opcode class Fe spends 5,560 more on `PUSH`, `DUP`, `SWAP` and `POP` and
1,668 less on jumps than the twin over the whole transaction (`gas_gap.py`, class table).

## 4. Storage layout: 44,200 gas of `claim`, 4,200 of every call that loads a claim

`claim` executes 8 `SSTORE` and 1 `SLOAD` in Fe (176,800 gas) and 6 `SSTORE` and 2 `SLOAD` in the twin
(132,600 gas): the record takes seven slots in Fe and five in the twin (`docs/fe-notes.md` M3.3
finding 25). Each further fresh slot is 22,100 gas to write and 2,100 to load cold, and the 4,200
appears unchanged in all four calls that load a record. The two slots have different owners:

- **`Address` is a whole word** (`std::evm::Address` wraps a `u256`), so the claimant cannot share a
  slot with the class and status bytes as it does in Solidity. This one is the library's.
- **The deadline is a whole word** because `Timestamp` wraps a `u256`; the twin stores a `uint64`.
  This one the source could take back today by storing the seconds as a `u64` next to `class` and
  `status`: one slot fewer, so 22,100 gas less on `claim` and 2,100 less per load (estimate, not built).

## 5. The rest of `challengeNeuron1`: 18,686 gas

Before the first multiplication, outside masks and the two extra loads, Fe spends 18,764 gas more than
the twin; the dot product is 692 less (the twin validates inside its loop) and the code after it 614
more. What Fe does in that first part, from the trace:

| Work | Gas |
|---|---|
| `leaf`: appending the 25 row words to `std::evm::packed::Packed`, one `u256` call per word | 9,012 (about 360 per word) |
| the two decode loops of cause 2 | 4,360 |
| word fills (112) and memory-to-memory word copies (111) | 2,288 + 1,332 |
| validation loop | 1,750 |

The twin builds the same preimage with `abi.encodePacked`: three `CALLDATACOPY` in its whole
transaction and no loop of 1,000 gas or more before its first multiplication. The twin's side of each
row was not isolated, so these rows are Fe's spend (18,742 gas in all), not differences, and the
18,764 is not split one to one.

## Bytecode size: 11,649 against 4,940 bytes

Measured by building the arbiter on other kernels (`gas_gap.py`, "Runtime bytecode, static"):

| Arbiter built with | Runtime bytes | Against the twin's 4,940 |
|---|---|---|
| the shipped `u32` kernel | 11,649 | +6,709 |
| `i256_unrolled` | 10,284 | +5,344 |
| `i256_loop`, the smallest kernel | 9,405 | +4,465 |

- **Kernel form: 2,244 bytes, 33.4% of the gap.** The kernel is `#[inline(always)]` and generic over
  the layer size, so the shipped arbiter holds it twice (66 `MUL` in the code, 32 in the twin, which
  shares one `dotWord`). That is the source's choice, taken for gas (`docs/fe-notes.md` M2.4).
- **Masks: 1,014 bytes inside that**, 169 sequences of 6 bytes. Cause 1a and 1b would remove most.
- **`dispute::step` as a pure function: 376 bytes** (`docs/numbers.md` section 2).
- **Unexplained: about 4,089 bytes, 61% of the gap.** With the smallest kernel the Fe arbiter is still
  1.90x the twin. It has 231 `JUMPDEST` against 143 and 5,420 bytes of `PUSH` instructions against
  2,642, which points at inlined decoding and encoding per handler, but that was not measured.

`NeuronCheck` (4,301 against 1,229 bytes): 2,242 of the 3,072 bytes go with the kernel form (2,059
bytes on `i256_loop`), 984 of them masks; 830 bytes remain.

## What this suggests for the compiler

1. **Drop masks that a range proves dead.** `((a & 0xff) * (b & 0xff)) + ((a >> 24) * (b >> 24))` on
   `u32` under unchecked arithmetic should compile with no `PUSH4 0xffffffff AND`. Here: 9,456 gas per
   layer 1 neuron, and up to 966 bytes of the arbiter (its 161 static `u32` masks, 6 bytes each; how
   many of them a range proves dead was not counted).
2. **Pick the mask encoding by cost.** `u64` masks: 6 gas and 10 bytes, or 14 gas and 6 bytes. `-O 2`
   could take the first.
3. **Decode a fixed array of words with one copy.** A handler taking `[u256; 25]` should show one
   `CALLDATACOPY` and no per-word loop or fill. Here: about 2,100 gas per array argument.
4. **Do not copy an array that is only read.** 59 word copies in `neuron1`, 111 in `challengeNeuron1`.
5. **Give `packed::Packed` a bulk append** for a word array, or make `u256` cheap in a loop. Here:
   9,012 gas for 25 words.
6. **Let `Address` take 20 bytes in a storage struct**, so it packs with small fields. Here: 22,100 gas
   per claim and 2,100 per load.

Projection, not a build: without the 1,576 range-dead masks the neuron would cost about 83,152 gas
(1.11x); with one `CALLDATACOPY` per array and no fills or copies as well, about 76,962 (1.03x:
83,152 - 6,346 + 156).

## Reproduce

```
bash tools/gen_model.sh                 # once, writes build/tree.json
python3 tools/gas_gap.py                # in WSL; 54 s here; its own anvil on port 18731 (--port N)
```

The script builds in a temporary directory and writes nothing to the repository. Versions of the
run quoted here: fe 26.5.1 (149c758), forge 1.9.0, solc 0.8.38 via-IR with 200 optimiser runs, anvil
at Osaka with `--steps-tracing`, reference sample 0. Its totals equal `docs/numbers.md`: execution gas
54,588 and 36,653 for the neuron, and the transaction gaps of section 3 for every arbiter call.

| Figure here | Section of the script's output |
|---|---|
| causes 1a to 1d, the class shares | "Layer 1 neuron: Fe as shipped against the best Solidity" |
| cause 2, cause 3, the phase split | "Layer 1 neuron by phase", "Loops before the first MUL", the "arrays" lines |
| the `i256_unrolled` residue | "Layer 1 neuron: Fe with the i256_unrolled kernel against the best Solidity" |
| cause 4 and the arbiter table | "Arbiter.claim" to "Arbiter.finalize", the storage and mask rows |
| cause 5 | "Arbiter.challengeNeuron1 by phase", "Loops before the first MUL" |
| sizes | "Runtime bytecode, static" |

How the script counts: a mask is `PUSHn` of n bytes of `ff` then `AND` (with at most one `DUP`
between), or `PUSH0 NOT PUSH1 k SHR AND`, or `PUSH1 k SIGNEXTEND`; `PUSH1 0xff AND` is not counted,
because in the kernel it is the source's own byte extraction. The phase split is at the first and the
last `MUL` executed, so the last word's closing additions fall in "after" (94 gas of masks).
