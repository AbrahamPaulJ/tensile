# Tensile: tree ensembles, arithmetic, commitment and dispute (v0, 2026-11-24)

A second model family beside the dense networks of `docs/spec.md` and `docs/spec-dispute.md`:
decision-tree ensembles (gradient-boosted trees, random forests), the models practitioners use for
tabular risk, pricing and scoring. Same inputs, same claim lifecycle (`ingots/dispute`), same
one-transaction dispute; the unit a challenge re-runs is one tree instead of one neuron.

| Part | Where |
|---|---|
| Kernel: `Tree`, `Outputs`, `Base`, `Forest`, `TreeProbe` | `ingots/trees` (pure Fe, no `std`) |
| Reference (the oracle) | `tools/trees_ref.py` (pure Python, standard library only) |
| Generator | `tools/gen_trees.py`, `tools/gen_trees_workspace.sh` |
| Example | `examples/address-risk-trees` (the address-risk data, 300 trees of depth 6) |

## 1. What it is for

An operator runs a forest off-chain and posts the answer with a bond. The answer stands after a
window unless someone proves it wrong with one transaction that re-walks **one tree** (`DEPTH`
comparisons) or re-adds the committed per-tree outputs. The contract holds no trees: a challenger
brings the tree's words with a Merkle proof against the model root.

## 2. Types

| Type | Meaning | Why the parameters are in the type |
|---|---|---|
| `Tree<DEPTH, IN, LF>` | one complete binary tree of depth `DEPTH` over `IN` input bytes, leaves int16 with `LF` fractional bits | a tree of another depth or feature count does not fit where this one is expected; its leaves cannot be added to scores at another scale |
| `Outputs<TREES, LF>` | one int16 value per tree, packed | what a claim commits to; its length is the forest's |
| `Base<CLASSES, LF>` | one int32 base score per class, `\|base\| <= 2^30 - 1` | the base's scale must match the leaves' |
| `Forest<TREES, DEPTH, IN, CLASSES, LF>` | `TREES` trees and a base | every size is checked by the type checker |

Inputs are `tensile::Activations<IN, 7>` (bytes 0..127, `docs/spec.md` section 6) and scores
`tensile::Accumulators<CLASSES, LF>`, the dense kernel's own types, so a forest and a network over
the same features take the same input value.

**Complete trees of a fixed depth.** Every tree has exactly `2^DEPTH - 1` internal nodes and
`2^DEPTH` leaves, in heap order. A trained tree that is not complete is made complete: a node that
does not split gets a threshold of 127 or more, which sends every input left, and the leaves no input
can reach hold any value (`examples/address-risk-trees/train.py` grows complete trees this way and
writes 0 there). This costs storage (a depth-6 tree is always 8 words) and buys a fixed shape: the tree is an array whose length is a const expression of
`DEPTH`, so a walk is `DEPTH` steps with no bounds to check at run time and the words a challenger
sends have a size fixed by the ABI.

**What the compiler did not allow, and what replaced it.** `Tree::build` and `from_words` were to
require `where tree_fits(DEPTH, IN)` (depth 1 to 8, at most 256 features, so a feature index fits a
byte). A generic caller that itself requires the same predicate with a literal argument is rejected
after substitution (`tests/defects/where_after_substitution.fe`, Fe 26.5.1, `error[8-0093]`). The
requirement became an `assert!` in the function body: still a compile error for every constant tree
(`tests/compile_fail/trees/feature_out_of_range.fe`), a run-time check otherwise. The overflow bound
of section 4 is a real `where` clause, because its callers are concrete.

## 3. The walk

Node `i` (root 0, children `2i + 1` and `2i + 2`) holds a feature index `f` and a threshold `t`
(0..255). The walk goes **left when `x[f] <= t`**, right otherwise, `DEPTH` times, and ends at leaf
`i - (2^DEPTH - 1)`. The tree's output is that leaf's value. An input byte is at most 127, so a
threshold of 127 or more sends every input left.

`Tree::leaf_index` and `Tree::eval` are the only walk. `Forest::walk` (with a probe),
`Forest::outputs`, the whole-forest contract and the tree challenge all call them.

## 4. Scores, class, and why no sum overflows

- `target(t, CLASSES)`: with two classes every tree adds to class 1 (one margin; class 0 scores its
  base alone, as a binary GBDT's logit against 0). With more classes, tree `t` adds to class
  `t % CLASSES` (the layout of a multi-class GBDT that builds one tree per class per round).
- Score of class `c` = `base[c]` + the outputs of the trees whose target is `c`, in int32.
- Class = the highest score, **the lowest index winning a tie**, `Accumulators::argmax`.

**Overflow, proved by the type.** Every output is an int16, at least `-2^15` and at most `2^15 - 1`,
and every base is within `2^30 - 1`. So a score is at most `2^30 - 1 + TREES * 2^15` in size, which
fits an int32 when `TREES * 2^15 <= 2^30`, that is up to 32,768 trees. `Outputs::scores` carries
`where sum_fits_i32(TREES)` and is `#[arithmetic(unchecked)]`: a forest of more trees does not
compile (`tests/compile_fail/trees/too_many_trees.fe`), and for every forest that does, no input and
no claimed output vector can overflow. This holds for *any* outputs a claim states, which matters:
the class challenge adds whatever the claimant committed.

The generated model ingot also recomputes the model's actual range (`|base|` plus each tree's
largest absolute leaf, per class) with `static_assert`s and requires the value the model file
records. For hundreds of trees one evaluation exceeds the compiler's limit of 1,000,000 steps per
constant (`docs/fe-notes.md`), so each tree's largest leaf is a constant of its own and
`trees::score_bound` adds them.

## 5. The model file (`tensile.forest.v0`)

```
{"name", "format": "tensile.forest.v0", "in", "input_frac_bits": 7, "depth", "classes",
 "leaf_frac_bits", "base": [int per class], "worst_case_abs_score" (optional, checked),
 "trees": [{"feature": [2^depth - 1 ints], "threshold": [...], "leaf": [2^depth ints]}, ...]}
```

`tools/trees_ref.py` `check_model` states the limits: `in` 1..256, `depth` 1..8, `classes` 2..255,
1 to 32,768 trees, `leaf_frac_bits` 0..30, features below `in`, thresholds 0..255, leaves int16,
bases within `2^30 - 1`. The file may carry a `training` record and test `metrics`; they are not
part of the commitment.

## 6. Packing

Everything is 16-bit lanes in 256-bit words, lane `i` at position `i % 16` of word `i / 16` from the
most significant end, unused lanes zero (`ingots/trees/src/lanes.fe`).

- A tree: the node lanes `feature << 8 | threshold` (`node_words(DEPTH)` words), then the leaf
  lanes as int16 two's complement (`lane_words(2^DEPTH)` words). Depth 6: 4 + 4 = 8 words.
  `Tree::well_formed`: every feature below `IN`, every unused lane zero; checked a word at a time.
- The outputs of a claim: one int16 lane per tree, in tree order, `lane_words(TREES)` words (300
  trees: 19 words). Well formed when the unused lanes of the last word are zero; every 16-bit value
  is a possible output.
- The input: `docs/spec.md` section 7 (32 bytes per word, each 0..127, zero padding).

## 7. The commitment and a claim

**Model root.** A Merkle tree over `TREES + 1` leaves, the sorted-pair Keccak-256 tree of
`docs/spec-dispute.md` section 2 (`std::evm::merkle::verify`, odd last node promoted):

```
tree leaf t = keccak256("tensile.tree.v0"   || u256(t) || the tree's words)
header leaf = keccak256("tensile.forest.v0" || u256(in) || u256(depth) || u256(classes)
                        || u256(leaf_frac_bits) || u256(trees) || i256(base[0]) || ...)
```

The tree index is inside the leaf, so a tree cannot be presented at another position (sorted pairs
do not bind positions). No leaf preimage is 64 bytes long (303 bytes for a depth-6 tree), so no
internal node passes as a leaf, and the two tags differ, so no tree leaf passes as the header. The
header leaf commits the shape and the base scores; the arbiter takes them from the typed model
ingot, and the generated tests rebuild the root from that ingot and require `MODEL_ROOT`, so an
arbiter compiled with other base scores fails its tests.

**Claim.** `claim(x0 words, outputs words, class)` with at least the bond attached:

1. Rejects a malformed input (a byte above 127 or non-zero padding), malformed outputs (non-zero
   padding lanes) and a class outside `0..CLASSES-1`. The input check is security-critical: a tree
   challenge decodes the committed input with `Input::from_words`, which would revert on a
   malformed one and leave a false claim unchallengeable.
2. Stores claimant, class, status, bond, deadline (`now + window`), `keccak256(x0 words)` and
   `keccak256(outputs words)`: six slots (section 8).
3. Emits `Claimed` with both vectors, so a watcher needs nothing but the log.

## 8. What the arbiter stores

`Record { claimant, class: u8, status: u8, bond, deadline, x0_hash, outputs_hash }`: the class and
status share a slot. Two commitments, where the dense arbiter keeps one per layer; the claim's size
grows with `TREES` only through its calldata and event (two bytes per tree).

## 9. Challenges and finalising

A challenge is allowed while `now < deadline` and the claim is open. One that proves nothing
reverts (`NoFraud`) and changes nothing. No challenger bond.

**(a) Tree challenge**, `challengeTree(id, t, x0, outputs, tree words, proof)`:

1. `t < TREES` (`TreeOutOfRange`);
2. `keccak256(x0) == x0_hash` (`InputMismatch`) and `keccak256(outputs) == outputs_hash`
   (`OutputMismatch`);
3. `tree leaf t` built from the words verifies against the root with `proof` (`NotCommitted`);
4. `Tree::from_words(words).eval(Input::from_words(x0))`, compared with lane `t` of `outputs`.
   Different: fraud.

**(b) Class challenge**, `challengeClass(id, outputs)`: `keccak256(outputs) == outputs_hash`; the
contract adds the committed outputs to the base with `Outputs::scores` (the function the off-chain
forest and the whole-forest contract use) and takes the argmax. Different from the claimed class:
fraud.

**On fraud** the claim is `Slashed`; half the bond is credited to the challenger and the rest burned
(`dispute::shares`). **Finalising** at or after the deadline makes it `Final` and credits the bond
back; `result(id)` answers only for a `Final` claim. Credits are pulled with `withdraw`. The rule is
`dispute::step`, unchanged: the arbiter calls the same pure state machine as the dense one.

## 10. Why this is sound

Let `F` be the committed forest: trees `T_0 .. T_{n-1}` and base `b`. A claim commits to `x0`, a
vector `o` of `n` int16 values, and a class `c`. It is true when `o_t = T_t(x0)` for every `t` and
`c = argmax(b + sum_t o_t e_{target(t)})`, which then equals `F(x0)`.

- **Every false claim can be slashed by one of the two challenges.** If some `o_t != T_t(x0)`, the
  tree challenge of `t` succeeds: the challenger presents the committed `x0` and `o` (they hash to
  the stored commitments) and the committed words of `T_t` (they verify at position `t`), and the
  contract recomputes `T_t(x0)` with the same `Tree::eval`. Otherwise every `o_t` is right, so the
  class that `o` adds up to is `F(x0)`; if `c` differs, the class challenge succeeds. There is no
  third case. A claim whose class *follows from* a forged output (the class challenge passes) is
  caught by the tree challenge of the forged tree: the generated test
  `a_forged_output_with_a_consistent_class_is_caught_only_by_its_tree` posts exactly that lie.
  Nothing a claimant can commit makes a challenge revert for another reason than "no fraud": the
  input and outputs were checked well formed at claim time, the walk and the sum cannot trap
  (section 4), and the challenger supplies the tree.
- **No true claim can be slashed.** Every tree recomputes to its committed output and the sum to the
  committed class, the arithmetic is deterministic integer arithmetic, and a challenger cannot
  substitute another tree (Merkle proof, index in the leaf), another position (index in the leaf),
  another input or other outputs (hashes) without breaking Keccak-256 collision resistance.

**Why a whole tree per Merkle leaf, not a path.** Calculated, not measured as an alternative build:
a depth-6 tree is 8 words (256 bytes of calldata, at most 4,096 gas at 16 gas per byte) under one
proof of 9 siblings for 301 leaves. Committing paths instead (one leaf per tree and leaf index, the
6 nodes and the value in one word) saves 7 words of calldata but makes the tree 64 times as many
leaves, adding 6 siblings (192 bytes) and 6 hashes to every proof, and needs a second argument that
the path is the one the input takes (each node's direction). Per-node leaves with a proof each
would cost about 7 proofs. The whole-tree leaf keeps the challenge to one `merkle::verify` and runs
the forest's own `Tree::eval` on the forest's own packed type, so the dispute is the off-chain
computation, not a re-statement of it. The measured cost of the chosen design is section 13.

## 11. Assumptions and limits

The assumptions of `docs/spec-dispute.md` section 7 carry over unchanged: one honest watcher able
to get a transaction in before the deadline; a correct model root (anyone can rebuild it from the
published model file with `tools/trees_ref.py`); the bond is the only penalty, and a claimant who
slashes its own false claim still loses the burned half; deployment parameters (bond, window) are
not checked; forced ETH; Keccak-256 collision resistance. Specific to trees:

- **The arbiter proves the computation, not the quality of the model.** The forest is a scorer
  trained on public data; a final claim says what it outputs, not that the output is right.
- **The input must be public**, as for every Tensile dispute.
- **Not covered by the repository's Foundry fuzz suite or mutation run**, which are tied to the dense
  arbiter. The tree arbiter's evidence is its generated Fe contract tests and the anvil scripts.
- **Shape limits**: depth 1 to 8 (a depth-8 tree is already 32 words of calldata in a challenge),
  up to 256 features (a feature index is one byte of a node lane), up to 32,768 trees, int16 leaves. Unbalanced trees are padded to
  complete ones (section 2).

## 12. Structure of the code

- `ingots/trees`: the kernel. `lanes.fe` (packing), `tree.fe` (`Tree`, the walk, well-formedness),
  `scores.fe` (`Outputs`, `Base`, `target`, the overflow bound), `forest.fe` (`Forest`, `walk` with
  `uses (probe: mut TreeProbe)`, `score_bound`), `probe.fe` (`TreeProbe`, `Unobserved`), `tests.fe`.
  Pure: one source for the EVM and native code (both backends run its tests); a WebAssembly
  build was not tried.
- The generated workspace (`tools/gen_trees.py`): `model` (every tree a checked constant, the
  forest, the commitment, reference tests), `arbiter` (the contract, its record, events and errors,
  the leaf functions, contract tests), `ensemble` (the whole forest on-chain: holders `Trees0..k`,
  each a slice of at most 64 trees as code constants, and `Ensemble`, which calls them, joins their
  packed outputs and adds them with `Outputs::scores`), `cli` (native classifier; `cli trace` prints
  every tree's output through the `TreeProbe` effect).
- `ingots/dispute` is reused as is. The arbiter's `record.fe` and `signals.fe` are written for
  trees (two commitments, tree errors), in the dense arbiter's style; nothing of `ingots/onchain`
  is copied.

## 13. Properties the tests establish

| # | Property | How |
|---|---|---|
| T1 | The Fe walk is the specification's walk, for every depth 1 to 8 | `trees_of_every_depth_walk_as_the_spec_says`: `Tree::eval` against a plain re-implementation on generated trees, both backends |
| T2 | Native Fe and EVM Fe equal the reference on every vector | `examples/address-risk-trees/check.sh`: native CLI on all 7,054 test rows and 1,000 traces (every tree's output, both scores); `Ensemble` on the EVM on 20 rows and the full trace of sample 0; 8 samples as Fe unit tests, whole forest and tree by tree, on both backends |
| T3 | Mismatched shapes do not compile | 10 cases in `tests/compile_fail/trees` (depth, feature count, class count, leaf scale, too many trees, leaf, base and feature out of range, malformed words) and a control |
| T4 | A true claim survives every challenge | generated test: the honest claim against all 300 tree challenges and the class challenge |
| T5 | A false claim is slashed by the challenge that names its error, and by no other | generated tests: one wrong tree output (only its tree), a wrong class (only the class challenge), a forged output with a consistent class (only its tree) |
| T6 | Only the committed tree, at its position, can be used | generated test: another tree's proof, another tree's words, one flipped bit, an index beyond the forest: each `NotCommitted` or `TreeOutOfRange` |
| T7 | Malformed claims and substituted vectors are refused with their own errors | generated tests, each failed call checked for its error selector |
| T8 | Bonds move exactly once: slash (half credited, half burned), finalise (all back), withdraw | generated tests and `tools/trees_gas.sh` on anvil (credits, `burned` and the balance read back) |

The measured costs (anvil, Osaka) are in `examples/address-risk-trees/RESULTS.md`, generated by its
`check.sh`.

## 14. Bring your own forest

1. Write the model file of section 5 (any trainer: export each tree as a complete tree of one depth,
   inputs quantised to 0..127, leaves scaled to int16 with `leaf_frac_bits`).
2. `python3 tools/trees_vectors.py --model MODEL.json --inputs INPUTS.bin --out DIR` writes the
   reference vectors (`inputs.bin`, `pred.bin`, `trace.jsonl`). Copy the model file to
   `DIR/model.json`: the anvil scripts read it there.
3. `bash tools/gen_trees_workspace.sh MODEL.json DIR WORKSPACE [samples]` writes the Fe workspace
   (section 12); `fe test` and `fe build` in it; `tools/diff_native_trees.py`,
   `tools/trees_gas.sh` and `tools/ensemble_gas.sh` check and measure it, as
   `examples/address-risk-trees/check.sh` does.
