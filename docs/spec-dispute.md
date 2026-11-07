# Tensile: commitment and dispute specification (v1, 2026-11-06)

Milestone 3. Builds on `docs/spec.md` (arithmetic, byte encoding, packed words). The reference
model is MNIST 784 -> 32 -> 10, but nothing here depends on its sizes except where stated.

## 1. What the arbiter is for

Anyone can run the network off-chain and post the result on-chain with a bond. The result is
accepted after a waiting period unless someone proves it wrong. Proving it wrong costs one
transaction that re-runs **one neuron** (or one argmax) on-chain, with the same `tensile::neuron`
code the off-chain run used. The contract never runs a whole layer and holds no weights.

## 2. Model commitment

The arbiter is deployed for one model and stores one value for it, `model_root`: the root of a
Merkle tree with one leaf per neuron.

```
leaf(l, j) = keccak256( "tensile.neuron.v0" || u256(l) || u256(j) || i256(bias) || row words )
```

- `l` is the layer (1-based), `j` the neuron index (0-based), `bias` the neuron's int32 bias
  sign-extended to 256 bits, `row words` the packed weight row (`docs/spec.md` section 7).
- The layer and index are inside the leaf, so a leaf cannot be replayed at another position.
- The tree is a sorted-pair (commutative) Keccak-256 tree, the kind OpenZeppelin's `MerkleProof`
  verifies: a parent is `keccak256(min(a, b) || max(a, b))` over two 32-byte children. Proofs are
  checked with the standard library's `std::evm::merkle::verify(proof, root, leaf)`, where `proof`
  lists siblings from the leaf's level up to the child of the root. The off-chain tree builder
  must produce exactly this.
- The leaf is hashed once. That is safe here because a leaf's preimage is never 64 bytes long (it
  is the 17-byte tag, three words and at least one row word), so no internal node can be passed
  off as a leaf.
- With an odd number of nodes at a level, the last node is promoted unchanged to the next level.

The model's shape (layer sizes, scales, shifts) is not committed in the root: it is compiled into
the arbiter through the typed model ingot.

## 3. A claim

A claim states: "for input `x0`, the hidden activations are `h`, the logits are `z`, and the class
is `c`." For a network with more hidden layers, `h` is one vector per hidden layer.

`claim(x0 words, hidden words, logits, class)` with at least `BOND` of ETH attached:

1. Rejects malformed vectors: non-zero padding, an input or hidden activation outside 0..127, a
   class outside `0..OUT_last` (the class is a `uint8`, so `class < 10` for the reference model). This is security-critical, not hygiene: the compile-time overflow
   proof holds only for activations in 0..127, so an out-of-range vector could make the neuron
   recomputation trap and leave a false claim impossible to challenge.
2. Stores the claimant, the bond, the deadline (`now + WINDOW`), the class, and a hash of each
   vector: `keccak256(x0 words)`, `keccak256(hidden words)`, `keccak256(logit bytes)`.
3. Emits an event carrying the full vectors, so anyone can audit the claim without a trace of the
   transaction. The vectors are dynamic arrays in the event: Fe 26.5.1 does not allow fixed-size
   arrays in events (`tests/defects/event_fixed_array.fe`).

The bond, the challenge window and the model root are constructor arguments.

Claims are numbered in order of arrival. Several claims about the same input may coexist.

## 4. Challenges

A challenge is allowed while `now < deadline` and the claim is open. A challenge that does not
prove fraud reverts and changes nothing, so a failed challenge costs only the challenger's gas and
cannot delay or damage a claim. No challenger bond is needed.

**Neuron challenge.** One handler per layer, `challengeNeuron1`, `challengeNeuron2` and so on, each
taking `(claim, j, input words, output vector, row words, bias, proof)`. A handler per layer is
forced by the typed design: each layer has its own input length, scale and shift in its types, so
one handler with a runtime layer number would have to give those types up. Each handler:

1. `input` must hash to the committed vector feeding layer `l` (`x0` for layer 1, the hidden
   vector of layer `l - 1` otherwise).
2. `output` must hash to the committed vector produced by layer `l` (the hidden vector, or the
   logits for the last layer).
3. `leaf(l, j)` built from `row words` and `bias` must verify against `model_root` with `proof`.
4. The contract recomputes neuron `j` of layer `l` on `input` (accumulate, then shift and clamp
   for a hidden layer; the raw accumulator for the last layer).
5. If the result differs from `output[j]`, the claim is fraudulent. Otherwise the call reverts.

**Class challenge.** `challenge_class(claim, logits)`: `logits` must hash to the committed logits;
if `argmax(logits)` (lowest index wins a tie) differs from the claimed class, the claim is
fraudulent. Otherwise the call reverts.

**On fraud.** The claim becomes `Slashed` (terminal). Half the bond, rounded down, is credited to
the challenger; the rest is burned: added to `burned()`, held by the contract for good and owed to
nobody. The burn is what makes a false claim cost its claimant something even when the claimant
slashes it itself (section 7). The split is `dispute::shares`, a pure function whose
`static_assert`s state that the two shares always add up to the bond.

## 5. Finalising

`finalize(claim)` is allowed when `now >= deadline` and the claim is open. The claim becomes
`Final` (terminal) and the bond is credited back to the claimant. `result(claim)` returns the
class only for a `Final` claim.

Credits are withdrawn with `withdraw()` (pull payments): state is updated before any ETH moves,
and no handler sends ETH to an address that is not the caller.

## 6. Why this is sound

Let `f` be the network of `docs/spec.md` with the committed weights.

- **A false claim can always be slashed.** If `c != f(x0)`, at least one committed step is wrong.
  Walk the claim in order: if some hidden value differs from the neuron recomputed on the
  committed input, that neuron challenge succeeds. Otherwise the hidden vector is right; if some
  logit differs from its neuron recomputed on the committed hidden vector, that challenge
  succeeds. Otherwise the logits are right, so `c != argmax(logits)` and the class challenge
  succeeds. Each check compares two *committed* vectors, so a later vector that is "consistent
  with" an earlier wrong one does not hide the earlier error.
- **A true claim can never be slashed.** Every neuron and the argmax recompute to the committed
  values, the arithmetic is deterministic, and a challenger cannot substitute other weights
  without breaking the Merkle proof or other vectors without breaking a hash.

## 7. Assumptions and limits

- **One honest watcher.** At least one party re-runs claims off-chain and can get a transaction
  included before the deadline. `WINDOW` must be long enough for that. Censorship of the
  challenger for the whole window defeats the scheme, as in every optimistic protocol.
- **The model root is right.** The deployer commits the root. Anyone can rebuild it from the
  published model file and compare.
- **The bond is the only penalty.** A false claim costs its claimant at least half its bond: all of
  it when someone else slashes it, half when the claimant slashes it first (below). The scheme does
  not make false claims impossible; it makes them unprofitable when half the bond exceeds the gain
  from a false result standing for at most `WINDOW`. Consumers must read results only from `Final`
  claims.
- **Challenge front-running, and the claimant's own slash.** A successful challenge can be copied
  from the public mempool by someone who then collects the challenger's half. The fraud is still
  caught; the reward may go to the copier. The copier can be the claimant: nothing stops a claimant
  from challenging its own false claim, so the claim ends `Slashed` (the false result is never
  usable) and the claimant gets back the challenger's half. The other half is burned whoever
  slashes, so this escape costs half the bond, not just gas. (Before the burn it cost only gas.)
  The cost is that an honest challenger earns half the bond rather than all of it. Pinned by
  `test_claimant_slashing_its_own_false_claim_loses_half_the_bond`.
- **Deployment parameters are not checked.** The constructor accepts a bond of zero (no penalty) and
  a window of zero (every claim can be finalised at once and never challenged). A consumer must read
  `modelRoot()`, `bond()` and `window()` before trusting an arbiter. Pinned by
  `test_zero_bond_deployment_has_no_penalty` and `testFuzz_P5_zero_window_allows_only_finalize`.
- **Forced ETH.** ETH can reach any contract without calling it (`SELFDESTRUCT`, a block reward or
  withdrawal recipient). It stays in the arbiter for good and changes nothing else, so P1 is an
  equality only without forced ETH, and `>=` in general. Pinned by
  `test_P1_forced_eth_breaks_only_the_equality`.
- **Data availability.** All vectors of a claim are in its calldata and event, so the watcher needs
  nothing from the claimant.
- **Hash and Merkle soundness.** Keccak-256 collision resistance.
- **Scope.** Dense layers only. The arbiter says a claim matches the committed integer network; it
  says nothing about whether that network is a good model.

## 8. Structure of the code

- The claim lifecycle is a pure state machine in its own ingot: `(claim, now, action) -> result`,
  with no storage, no clock and no ETH. The contract is a thin shell that reads the clock, loads
  and stores the claim, and moves credits. The state machine is unit-tested on both backends
  without any time-travel facility, because time is an argument.
- The off-chain auditor is the native CLI: given a claim's vectors it re-runs the network with the
  same kernel and prints either `ok` or the first failing check (`neuron l j expected claimed` or
  `class expected claimed`). That line is exactly the challenge to send.

## 9. Properties the tests must establish

| # | Property | How |
|---|---|---|
| P1 | The arbiter's ETH balance equals the bonds of open claims plus unwithdrawn credits plus `burned()` (plus any forced ETH, section 7) | Foundry invariant test |
| P2 | A claim equal to the reference output cannot be slashed by any challenge | Foundry fuzz: random challenges against honest claims all revert |
| P3 | A claim with any single wrong value can be slashed, by the challenge the auditor names | Foundry fuzz: mutate one value of an honest claim, send the named challenge, expect `Slashed` |
| P4 | A claim only moves `Open -> Slashed` or `Open -> Final`; terminal states never change; each bond is shared out exactly once (all to the claimant, or half to the challenger and the rest burned) | State machine unit tests and `static_assert`s, plus Foundry invariant |
| P5 | Challenges succeed only before the deadline; `finalize` only at or after it | State machine unit tests at the boundary; Foundry with time travel |
| P6 | Malformed vectors are rejected at claim time | Unit and fuzz tests: padding, range, class bound |
| P7 | The native auditor and the Python reference name the same first failing check | Differential test over mutated claims |

## 10. Costs

Measured 2026-11-07 on anvil (Osaka), after the half burn, real transactions for reference sample 0 (`tools/arbiter_gas.sh`):

| Call | Gas | Calldata |
|---|---|---|
| claim | 246,113 | storage-bound: 7 record slots plus the counter |
| challengeNeuron1 (784 inputs) | 178,662 | |
| challengeNeuron2 (32 inputs) | 73,142 | 708 bytes |
| challengeClass | 63,540 | 356 bytes |
| finalize | 49,034 | 36 bytes |
| withdraw (4 credits) | 30,096 | 4 bytes |

A slash costs about 22,000 gas more than before the burn: the first write of `burned` to a zero slot. A failed
challenge reverts, so `cast estimate` cannot measure it; the demo page mines one and shows its gas. All current
figures: `docs/numbers.md`.
