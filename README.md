# Tensile

**Pitch:** One integer neural-network kernel, written once in Fe with shapes and fixed-point scales in
its types, runs natively and in WebAssembly to classify, and on the EVM to settle a dispute: a false
answer posted on-chain is proven false by re-running a single neuron, at 9.4% of the gas of running the
whole network.

**[Try it: `demo/dist/tensile-demo.html`](demo/dist/tensile-demo.html)** is one self-contained file. Download it and open
it in a browser: no wallet, no install, no network. **[Watch it: `demo/video/tensile-demo.mp4`](demo/video/tensile-demo.mp4)**, about
three minutes.

**For a judge with five minutes**, from the repository root:

| Command | Needs | What it shows | Time |
|---|---|---|---|
| open `demo/dist/tensile-demo.html` | a browser | draw a digit, post a lie, watch one neuron disprove it | 2 min |
| `fe test` | Fe 26.5.1 | 72 tests: kernel, model against the reference, dispute rule, contracts | 25 s |
| `node wasm/diff.mjs` | Node 20+ | the committed WebAssembly module against the NumPy reference: 10,000 images, 0 mismatches | 4 s |
| `bash tools/check_compile_fail.sh -v` | Fe 26.5.1 | 14 wrong models that must not compile (a bias at the wrong scale, mismatched layers...), each with the compiler's message | 17 s |

## The problem

Running a neural network inside a smart contract is possible and costly: this project's small MNIST
network costs 1,896,043 gas per answer. Paying that for every answer rules out almost any use.

The known way out is optimistic: post the answer with a bond, and let anyone prove it wrong within a
time window. Then only a dispute costs anything, and it need only re-run one step. That has a catch.
The dispute is worth nothing unless the step re-run on-chain computes *exactly* what the off-chain run
computed, down to the rounding of a negative number. Usually these are two code bases in two languages,
kept in step by hope.

Tensile is for people who want to build on verifiable inference, and for Fe builders who want a library
of typed numeric code that compiles to both backends.

## Approach

- **One kernel, three targets.** `ingots/tensile` is pure Fe with no effects. The native classifier,
  the on-chain arbiter and a WebAssembly module call the same `neuron` function. The NumPy reference
  and the native, EVM and WebAssembly builds agree bit for bit: 0 mismatches on all 10,000 MNIST test
  classes and 1,000 full traces, on every backend, at every optimisation level tested. In the demo,
  the WebAssembly build is the watcher's own off-chain run.
- **The types carry the model.** `Row<IN, FW>`, `Activations<N, F>` and `Q<F>` hold shape and scale as
  const generics. Wiring layers of the wrong shape, mixing scales, or giving a bias the wrong scale does
  not compile. A `static_assert` proves that no accumulator can overflow, for every possible input.
- **A dispute re-runs one neuron.** A claim commits to the input, every hidden activation, every output
  and the answer. A challenger names one wrong value and presents the neuron's weights with a Merkle
  proof against the model's root. The contract holds no weights and never runs a layer.
- **The rule is a pure function.** `dispute::step(claim, now, action)` is the whole lifecycle, with no
  storage or clock, unit-tested on both backends, and `dispute::shares` splits the bond. Its deadline
  rules, and that every bond is shared out to the last wei, are `static_assert`s.
- **Why not `core::fixed`.** Fe's standard fixed point is unsigned decimal on 256-bit words
  (`UFixed<D>`, a power-of-ten scale). A network needs signed values with a binary point: 8-bit
  weights and activations packed 32 to a word, 32-bit accumulators, and requantisation as an
  arithmetic shift. `Q<F>` is that, an `i32` with `F` fractional bits, so it is its own type.

Everything was built during the contest window. Integer accuracy is 97.44% on the MNIST test set; the
floating-point model it came from reaches 97.50%.

## Fe highlight

Many Fe libraries put a scale in a type. Tensile computes with scales in types: the neuron's signature
states the rule of fixed-point multiplication, so a bias must have the scale of a weight times an
activation, `F + FW` fractional bits, and a compile-time function in the `where` clause proves that
the accumulator cannot overflow for any input of that length:

```rust
// ingots/tensile/src/dense.fe
pub fn neuron<const IN: usize, const FW: usize, const F: usize>(
    row: Row<IN, FW>,
    x: Activations<IN, F>,
    bias: Q<{ F + FW }>,
) -> Q<{ F + FW }>
where dot_fits_i32(IN)
```

Get it wrong and the compiler says so (`tests/compile_fail/bias_wrong_scale.fe`, one of 14 such cases):

```
error[8-0000]: type mismatch
  │     dense(weights, x, bias)
  │                       ^^^^ expected `Bias<2, 14>`, but `Bias<2, 15>` is given
```

All of that disappears at compile time. Measured against the same contract written over plain words,
the types cost **2.1%** of a layer 1 neuron's execution gas, and the pure state machine under 1% of a
call (`docs/numbers.md`, section 2).

Fe's effects also make the security argument readable. Of the arbiter's handlers, exactly one declares
that it can send ETH:

```rust
// ingots/onchain/src/arbiter.fe
Finalize { id } uses (ctx, count, mut claims, mut credits, mut burned, mut log) {
Withdraw -> Wei uses (ctx, mut call, mut credits, mut log) {
```

## Architecture

| Path | Language | What it is |
|---|---|---|
| `ingots/tensile` | Fe, pure | `Packed`, `Row`, `Weights`, `Activations`, `Q`, `Bias`; `neuron`, `dense`, requantise, argmax; the compile-time overflow bound |
| `ingots/model` | Fe, generated | The weights as packed literals and the typed layer chain, from `tools/gen_fe.py` |
| `ingots/dispute` | Fe, pure | The claim state machine |
| `ingots/cli` | Fe, native | The classifier and auditor: reads samples or claims on stdin, names the first failing check |
| `ingots/onchain` | Fe, contracts | `Arbiter`, `Network`, `NeuronCheck`, `RowsLow`, `RowsHigh`, the leaf commitment |
| `wasm/` | Fe, WebAssembly | `entry.fe`, the module's exports; `tensile_wasm.fe`, `tensile` and `model` flattened into one file by `tools/gen_wasm.py` (the wasm target takes only standalone files); the built module in `wasm/out/` |
| `ref/` | Python, NumPy | Training, quantisation, and the integer reference: the oracle every other part must match |
| `evm/` | Solidity | Foundry fuzz and invariant tests of the Fe bytecode; Solidity twins as gas baselines |
| `demo/` | JavaScript | The page: an EVM in the browser running the `fe build` output |
| `tools/` | Python, shell | Generator, differential tests, check scripts, mutation test, numbers |

```
                     ingots/tensile  (pure kernel, types)
                       │         │
               ingots/model   ingots/dispute (pure lifecycle)
                 │       │       │
   ingots/cli (native)  ingots/onchain (EVM)       wasm/ (WebAssembly)
        │                        │                     │
   names the challenge  ──►  Arbiter re-runs     the demo's watcher
                             that one neuron
```

The arithmetic is specified in `docs/spec.md`, the protocol in `docs/spec-dispute.md`, what the build
learned about Fe in `docs/fe-notes.md`.

### What a dispute does

1. A claimant posts input, 32 hidden values, 10 outputs and the answer, with a bond (246,113 gas).
2. A watcher re-runs the network off-chain with the native build. The first value that differs names
   the challenge.
3. The arbiter checks the vectors against the claim's hashes, checks the neuron's weights against the
   model root, re-runs that one neuron, and on a mismatch credits the challenger half the bond and
   burns the rest, so a false claim costs at least half its bond even if its own claimant slashes it
   (178,662 gas at layer 1, 73,142 at layer 2, 63,540 for the answer).
4. A claim nobody disproves becomes final after the window (49,034 gas), and only then does
   `result(id)` answer. Bonds are pulled with `withdraw()`, the only handler that sends ETH.

## Build and test

The exact versions used: **fe 26.5.1 (149c758)**, Foundry 1.9.0, solc 0.8.38, Python 3.11, NumPy 2.3,
Node 22. Linux or WSL2 for Fe and Foundry; run the commands from the repository root.

```sh
# Fe, pinned
curl -fsSL https://raw.githubusercontent.com/argotorg/fe/master/feup/feup.sh | bash -s -- --version v26.5.1

fe --version          # fe 26.5.1 (149c758)
fe fmt --check ingots
fe test               # 72 tests on the EVM backend, about 25 s
fe build              # the contracts, into ./out
```

The full checks need Foundry (`curl -L https://foundry.paradigm.xyz | bash && foundryup --install 1.9.0`),
`bc`, `xxd` and `python3`:

| Command | What it establishes | Time |
|---|---|---|
| `bash tools/m1_check.sh` | Both backends' unit tests, 14 compile-fail cases, native against the reference on all 10,000 samples at `-O 0`, `-O 1`, `-O 2` | 68 s |
| `bash tools/m2_check.sh` | The contracts against the reference on all 10,000 classes and 1,000 traces; fuzz against a plain Solidity spec; real gas on anvil | 199 s |
| `bash tools/m3_check.sh` | The arbiter: 26 Fe contract tests, 45 Foundry fuzz and invariant tests for properties P1 to P6, auditor against Python on 5,200 mutated claims | 176 s |
| `bash tools/m4_numbers.sh` | Regenerates `docs/numbers.md`: every gas and size figure, from real transactions | about 6 min, or 78 s with `--no-forms` |
| `python3 tools/m3_mutants.py --fresh` | 59 deliberate faults in the contract; all are caught, 11 of them by the compiler. Results in `docs/mutation-results.md` | 977 s |
| `python3 ref/get_mnist.py && python3 ref/test_ref.py` (needs NumPy; the first downloads MNIST and checks its SHA-256) | The reference's arithmetic and exported files | 5 s |
| `bash tools/m5_clean_run.sh` | Copies only what git would publish to an empty directory, makes it `$HOME`, installs the tools, runs all the above except the mutation test and the numbers, and rebuilds the demo | 9 min |

The demo needs only Node to rebuild and test (the built page is committed):

```sh
cd demo && npm ci
npm run build         # dist/tensile-demo.html
npm test              # the page's chain against anvil's gas and the reference; every lie, on every sample
npm run test:ui       # the page itself in Chrome, offline, from file://
```

`bash tools/build_demo.sh` regenerates what the page embeds from a fresh `fe build`.

### The WebAssembly build: a pinned master commit

Fe's WebAssembly target was merged on 2026-11-05 ([argotorg/fe#1904](https://github.com/argotorg/fe/pull/1904)) and
is not in a release yet, so this one part uses **fe 26.5.0 (89d07b9db)**, master commit
`89d07b9dbf7ea73d0ca6b9a5b4f14de7899d1908`, which the organisers allow. Everything else uses 26.5.1. The
built module is committed, so the demo needs no Rust.

```sh
git clone https://github.com/argotorg/fe ~/fe-master && cd ~/fe-master && git checkout 89d07b9dbf7ea73d0ca6b9a5b4f14de7899d1908
(cd crates/tree-sitter-fe && npm ci) && PATH="$PWD/crates/tree-sitter-fe/node_modules/.bin:$PATH" cargo build --release -p fe --features wasm
cd - && FE=~/fe-master/target/release/fe bash tools/build_wasm.sh
```

`tools/build_wasm.sh` flattens the ingots, runs all 26 kernel and model unit tests on the wasm backend
(12 s), builds the module twice and requires identical bytes (37,471 bytes), then checks it against the
reference: 10,000 of 10,000 classes, 10,000 of 10,000 logits and 32,000 of 32,000 hidden activations,
about 380 microseconds per image in Node, and the same again for modules built at `-O 0` and `-O 2`. CI
does all of this from a fresh build of that commit.

Times are from one laptop (i5-13420H, WSL2 Ubuntu 24.04), with the repository on the Linux file system.

### A reproducible example without a browser

```sh
fe build --backend native -O 1 --out-dir out-native --ingot cli .
head -c 784 ref/out/inputs.bin | out-native/cli          # 7
head -c 784 ref/out/inputs.bin | out-native/cli trace    # every hidden value, every output, the class
```

## Demo

Open [`demo/dist/tensile-demo.html`](demo/dist/tensile-demo.html). Draw a digit and the bytecode from `fe build` classifies it,
on an EVM running in the tab, and the WebAssembly build of the same source classifies it again, in well
under a millisecond, with every hidden value and output equal. Post a claim, honest or forged, and watch the watcher name the wrong value
and the contract settle it. Every refusal you see is the contract's own error, and no button is gated by
the page. The page's chain reproduces this project's anvil gas figures exactly (`demo/test/run.mjs`).

The [video](demo/video/tensile-demo.mp4) walks through the same page. Its narration is a synthetic voice, and the script is
`demo/video/script.json`.

## Limitations

- **One honest watcher.** The scheme is sound only if someone checks every claim and can get a
  transaction in before the deadline, as with every optimistic protocol. A false claim nobody challenges
  becomes final. The demo shows this.
- **Half the bond is burned on every slash.** That closes the claimant's own escape (slashing its false
  claim before anyone else to get the bond back): it now costs half the bond, not just gas. The price is
  that an honest challenger earns half the bond, not all of it, and the burned half stays in the arbiter
  for good (`burned()`). Pinned by tests and `static_assert`s (`docs/spec-dispute.md`, sections 4 and 7).
- **Challenges can be front-run,** so the reward may go to whoever copies a challenge from the mempool.
  The fraud is still caught.
- **Deployment parameters are not checked.** A bond or window of zero is accepted, so a consumer must read
  `modelRoot()`, `bond()` and `window()` before trusting an arbiter.
- **The model root is trusted as deployed.** Anyone can rebuild it from the model file and compare.
- **It proves a computation, not a good model.** The arbiter says a claim matches the committed integer
  network, nothing about whether the network is right about the digit: it is wrong on 2.56% of the test
  set, and hand-drawn digits are harder still.
- **Small dense networks only.** No convolution. Layer sizes are compile-time. Layer 1's weights do not fit
  one contract, so `Network` reads them from two others.
- **Not audited, and not deployed.** Nothing has been deployed to a public network. The mutation test's 53
  faults were chosen by hand, so "59 of 59" is not a random sample. Do not hold real value with this.
- **Fe costs more gas than the best hand-written Solidity here:** 1.24x per neuron transaction and 1.26x
  per claim (`docs/numbers.md`, section 3), and 2.39x the code. It is cheaper than the natural Solidity
  assembly loop (0.78x).
- **The demo's chain is EthereumJS, not a real network.** Its watcher runs the WebAssembly build of the
  kernel, not the native auditor (`ingots/cli`), which does not run in a browser.
- **The WebAssembly build rests on an unreleased compiler commit,** and its source is generated by
  flattening two ingots into one file, because the target does not take ingots yet. Only the kernel and
  the model are built for it; the dispute logic and the auditor are not.

Compiler defects found along the way, with minimal examples: `docs/upstream-reports.md`.

## Declaration

- **Pre-existing work:** none. The project started in the build window, on 2026-11-05, from the
  [contest starter](https://github.com/fe-lang/builders-challenge-starter)'s template and CI.
- **Third-party code and data:**
  EthereumJS (`@ethereumjs/*`, MPL-2.0 and MIT) and noble-hashes and noble-curves (MIT) are bundled into the demo page,
  listed in `demo/THIRD_PARTY.txt`, unmodified. esbuild (MIT) and Playwright (Apache-2.0) are development tools.
  OpenZeppelin's `MerkleProof` is the model for the sorted-pair tree format, via Fe's `std::evm::merkle`; none of its code is used.
  The MNIST database (LeCun, Cortes, Burges) is the data: the test images are in `ref/out/`.
  The video's narration uses the Kokoro text-to-speech model (Apache-2.0).
- **AI assistance:** extensive. Claude (Anthropic), through Claude Code, did the research, wrote most of
  the Fe, Solidity, Python and JavaScript, the tests and these documents, and ran the measurements, under
  the direction of the human author, who chose the project and its scope and made the decisions on it.
  Every figure here comes from a recorded run of a script in this repository.

## Team

[AbrahamPaulJ](https://github.com/AbrahamPaulJ).

## License

Apache-2.0, see [`LICENSE`](LICENSE). `demo/dist/tensile-demo.html` bundles the third-party code listed above
under its own licenses.
