// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {IArbiter} from "./ArbiterBase.sol";
import {GClaim, GModel, GenUser, Gx} from "./GenBase.sol";

/// Fuzz tests of a generated Fe `Arbiter` with any number of layers, for properties P2 (an honest
/// claim cannot be slashed) and P3 (any wrong value is slashed by the challenge the spec's walk
/// names), in every layer, and of the Merkle commitment (docs/spec-dispute.md sections 2, 4, 6
/// and 9). The expected behaviour comes from the spec, through `GModel` (the network in plain
/// Solidity over the fixture rows), never from the Fe code or the generator. The tests follow
/// evm/test/ArbiterFuzz.t.sol one for one; those marked "Layers" exist only because a network
/// with three or more layers has a middle layer whose input and output are both hidden vectors.
contract GenArbiterFuzzTest is GenUser {
    uint256 internal constant BOND = 1 ether;
    uint256 internal constant WINDOW = 1 days;
    uint256 internal constant T0 = 1_000_000;
    address internal constant CLAIMANT = address(0xC1A1);
    address internal constant CHALLENGER = address(0xC4A1);

    function setUp() public {
        Gx.install();
        initShape();
        arbiter = deployArbiter(BOND, WINDOW);
        vm.warp(T0);
    }

    // ----------------------------------------------------------------------------------------
    // Helpers

    function pick(uint256 seed) internal view returns (GClaim memory) {
        return Gx.sample(seed % Gx.sampleCount());
    }

    /// `n` words from the fuzzer: the first eight as it gave them, the rest hashed from them.
    function expand(uint256[8] memory raw, uint256 n) internal pure returns (uint256[] memory x) {
        x = new uint256[](n);
        for (uint256 k = 0; k < n; k++) {
            x[k] = k < 8 ? raw[k] : uint256(keccak256(abi.encode(raw, k)));
        }
    }

    /// A well-formed input from the fuzzer. `amp` keeps only the low bits of every byte in seven
    /// runs of eight (activations up to 63, 31, .. 0), so that small inputs, where hidden neurons
    /// are neither all zero nor all saturated, are as frequent as large ones. With bit 0x40 of
    /// `amp` set, every value is an extreme instead (0 or 127, from the top bit of the fuzzed
    /// byte): on a model whose neurons are small on average (the 14-32-16-2 one never reaches 127
    /// on 64 uniform inputs) these are the inputs that saturate hidden neurons.
    function inputFrom(uint256[8] memory raw, uint8 amp) internal view returns (uint256[] memory x) {
        x = expand(raw, Gx.wordsOf(OUT[0]));
        if ((amp & 0x40) != 0) {
            for (uint256 k = 0; k < x.length; k++) {
                x[k] = ((x[k] & GModel.SIGN_BITS) >> 7) * 0x7f;
            }
        }
        x = GModel.sanitise(x, OUT[0]);
        uint256 mask = (0x7f >> (amp % 8)) * (GModel.SIGN_BITS >> 7);
        for (uint256 k = 0; k < x.length; k++) {
            x[k] &= mask;
        }
    }

    bytes4 internal constant NO_FRAUD = IArbiter.NoFraud.selector;

    /// The claim is still open and no ETH or credit moved.
    function assertUntouched(uint256 id, uint256 balance) internal view {
        require(statusOf(id) == 1, "a refused challenge changed the claim's status");
        require(address(arbiter).balance == balance, "a refused challenge moved ETH");
        require(arbiter.creditOf(CHALLENGER) == 0, "a refused challenge credited the challenger");
        require(arbiter.creditOf(CLAIMANT) == 0, "a refused challenge credited the claimant");
        require(arbiter.burned() == 0, "a refused challenge burned something");
    }

    /// The claim is slashed, half its bond credited once to `who` and the rest burned (the only
    /// slash of the test), and every further action on it is refused; the credit can be withdrawn
    /// exactly once and the burned half stays in the arbiter.
    function assertSlashed(uint256 id, address who, uint256 bond, GClaim memory c) internal {
        uint256 reward = bond / 2;
        require(statusOf(id) == 2, "a proven challenge did not slash");
        require(arbiter.creditOf(who) == reward, "half the bond was not credited to the challenger");
        require(arbiter.burned() == bond - reward, "the other half was not burned");
        require(arbiter.creditOf(CLAIMANT) == 0, "the claimant was credited for a slashed claim");
        bytes memory notOpen = abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, id);
        for (uint256 kind = 1; kind <= L + 1; kind++) {
            mustFail(who, checkCall(id, kind, 0, c), notOpen, "a slashed claim was challenged again");
        }
        vm.warp(deadlineOf(id));
        vm.expectRevert(notOpen);
        arbiter.finalize(id);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.NotFinal.selector, id));
        arbiter.result(id);
        uint256 before = who.balance;
        vm.prank(who);
        require(arbiter.withdraw() == reward, "withdraw returned another amount");
        require(who.balance == before + reward, "withdraw paid another amount");
        require(arbiter.creditOf(who) == 0, "the credit survived the withdrawal");
        require(address(arbiter).balance == bond - reward, "the burned half left the arbiter");
        vm.prank(who);
        vm.expectRevert(IArbiter.NothingToWithdraw.selector);
        arbiter.withdraw();
    }

    // ----------------------------------------------------------------------------------------
    // The oracle itself

    /// The fixtures and GModel agree with the reference trace: every hidden activation of every
    /// layer, every logit and the class of every fixture sample is what GModel computes from the
    /// sample's input. Without this the other tests would test the oracle, not the arbiter.
    function test_oracle_reproduces_the_reference_samples() public view {
        uint256 n = Gx.sampleCount();
        require(n >= 8 && L >= 2, "too few fixture samples or layers");
        bool[] memory seen = new bool[](CLASSES);
        for (uint256 s = 0; s < n; s++) {
            GClaim memory c = Gx.sample(s);
            GClaim memory m = GModel.honest(c.v[0]);
            for (uint256 l = 1; l < L; l++) {
                require(GModel.same(m.v[l], c.v[l]), "GModel's hidden vector differs from trace.jsonl");
            }
            for (uint256 j = 0; j < CLASSES; j++) {
                require(m.z[j] == c.z[j], "GModel's logits differ from trace.jsonl");
            }
            require(m.cls == c.cls, "GModel's class differs from trace.jsonl");
            (uint256 kind,) = walk(c);
            require(kind == 0, "the walk finds a failing check in a reference sample");
            seen[c.cls] = true;
        }
        for (uint256 k = 0; k < CLASSES; k++) {
            require(seen[k], "a class is never claimed by a fixture sample");
        }
    }

    /// The fuzzed inputs reach the middle of every hidden layer: over 256 inputs made the way the
    /// fuzz tests make them (one per value of `amp`), every hidden layer shows activations at 0,
    /// strictly between 0 and 127, and (somewhere in the network) at 127, and every class is
    /// predicted. Otherwise "an honest claim on a random input" would only ever test dead or
    /// saturated neurons, or one class.
    function test_fuzzed_inputs_exercise_every_layer() public view {
        uint256[] memory zero = new uint256[](L);
        uint256[] memory mid = new uint256[](L);
        uint256 top = 0;
        bool[] memory seen = new bool[](CLASSES);
        for (uint256 s = 0; s < 256; s++) {
            uint256[8] memory raw;
            for (uint256 k = 0; k < 8; k++) {
                raw[k] = uint256(keccak256(abi.encode("exercise", s, k)));
            }
            // Every value of `amp`: every amplitude, with and without extremes, as often as the
            // fuzzer picks them.
            GClaim memory c = GModel.honest(inputFrom(raw, uint8(s)));
            for (uint256 l = 1; l < L; l++) {
                for (uint256 j = 0; j < OUT[l]; j++) {
                    uint8 h = GModel.byteAt(c.v[l], j);
                    if (h == 0) zero[l]++;
                    else if (h == 127) top++;
                    else mid[l]++;
                }
            }
            seen[c.cls] = true;
        }
        for (uint256 l = 1; l < L; l++) {
            require(zero[l] > 0, "a hidden layer never outputs 0 on the fuzzed inputs");
            require(mid[l] >= 64, "a hidden layer is almost never strictly between 0 and 127 on the fuzzed inputs");
        }
        require(top > 0, "no hidden neuron ever saturates on the fuzzed inputs");
        for (uint256 k = 0; k < CLASSES; k++) {
            require(seen[k], "a class is never predicted on the fuzzed inputs");
        }
    }

    // ----------------------------------------------------------------------------------------
    // P2: a claim equal to the reference output cannot be slashed by any challenge

    /// Every neuron of every layer with its committed row, and the class challenge, against the
    /// honest claim of every fixture sample: all `NoFraud`, nothing moves.
    function test_P2_every_correct_challenge_of_every_honest_sample_reverts() public {
        for (uint256 s = 0; s < Gx.sampleCount(); s++) {
            GClaim memory c = Gx.sample(s);
            uint256 id = post(c, CLAIMANT, BOND);
            for (uint256 l = 1; l <= L; l++) {
                for (uint256 j = 0; j < OUT[l]; j++) {
                    mustFail(CHALLENGER, neuronCall(id, l, j, c), NO_FRAUD, "a correct neuron was not NoFraud");
                }
            }
            mustFail(CHALLENGER, classCall(id, c), NO_FRAUD, "a correct class was not NoFraud");
            assertUntouched(id, BOND * (s + 1));
        }
    }

    /// The honest claim for a random well-formed input (computed by GModel) survives a random
    /// neuron challenge of every layer and the class challenge.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P2_honest_claim_on_a_random_input_cannot_be_slashed(
        uint256[8] memory raw,
        uint8 amp,
        uint256 neurons
    ) public {
        GClaim memory c = GModel.honest(inputFrom(raw, amp));
        uint256 id = post(c, CLAIMANT, BOND);
        for (uint256 l = 1; l <= L; l++) {
            uint256 j = uint256(keccak256(abi.encode(neurons, l))) % OUT[l];
            mustFail(CHALLENGER, neuronCall(id, l, j, c), NO_FRAUD, "an honest neuron was not NoFraud");
        }
        mustFail(CHALLENGER, classCall(id, c), NO_FRAUD, "an honest class was not NoFraud");
        assertUntouched(id, BOND);
    }

    /// A neuron index past the layer is refused in every layer.
    function testFuzz_P2_neuron_index_out_of_range_is_refused(uint256 seed, uint256 j) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        for (uint256 l = 1; l <= L; l++) {
            uint256 n = OUT[l];
            uint256 jl = j < n ? j + n : j;
            bytes memory data = neuronData(
                l, id, jl, inputOf(c, l), outputOf(c, l), Gx.row(l, jl % n), Gx.bias(l, jl % n), Gx.proof(l, jl % n)
            );
            mustFail(
                CHALLENGER,
                data,
                abi.encodeWithSelector(IArbiter.NeuronOutOfRange.selector, jl),
                "a neuron index past the layer was not NeuronOutOfRange"
            );
        }
        assertUntouched(id, BOND);
    }

    /// Vectors that differ from the committed ones in one bit (the input or the output of any
    /// layer's handler, or the logits of the class challenge) are refused with the matching error,
    /// even though they would change the recomputation.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P2_substituted_vectors_are_refused(uint256 seed, uint256 j, uint256 bit, uint8 which) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        GClaim memory d = GModel.copy(c);
        uint256 w = which % (2 * L + 1);
        if (w == 2 * L) {
            uint256 k = (bit >> 8) % CLASSES;
            d.z[k] = int32(uint32(d.z[k]) ^ (uint32(1) << uint32(bit % 32)));
            mustFail(CHALLENGER, classCall(id, d), IArbiter.OutputMismatch.selector, "other logits in the class challenge");
        } else {
            uint256 l = 1 + w / 2;
            bool output = w % 2 == 1;
            uint256 vec = output ? l : l - 1; // which vector of the claim
            if (vec == L) {
                uint256 k = (bit >> 8) % CLASSES;
                d.z[k] = int32(uint32(d.z[k]) ^ (uint32(1) << uint32(bit % 32)));
            } else {
                d.v[vec][(bit >> 8) % d.v[vec].length] ^= uint256(1) << (bit % 256);
            }
            mustFail(
                CHALLENGER,
                neuronCall(id, l, j % OUT[l], d),
                output ? IArbiter.OutputMismatch.selector : IArbiter.InputMismatch.selector,
                "a substituted vector was not refused with its error"
            );
        }
        assertUntouched(id, BOND);
    }

    /// Layers: a handler given ANOTHER committed vector of the same claim in place of its own
    /// input or output (cut or zero-filled to the handler's width) refuses it. In a three-layer
    /// arbiter the middle handler takes two hidden vectors, and both of them, as well as the
    /// input, are committed by the same claim: only the hash it checks tells them apart.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_Layers_a_committed_vector_of_another_layer_is_refused(
        uint256[8] memory raw,
        uint8 amp,
        uint256 seed,
        uint8 lSeed,
        uint8 kSeed,
        uint256 j,
        bool output
    ) public {
        // A fixture sample, or the honest claim of a fuzzed input.
        GClaim memory c = (amp & 0x80) != 0 ? pick(seed) : GModel.honest(inputFrom(raw, amp));
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 l = 1 + lSeed % L;
        // The vector to replace: the handler's input (v[l - 1]) or its output (v[l], hidden layers only).
        if (l == L) output = false;
        uint256 vec = output ? l : l - 1;
        uint256 other = kSeed % L;
        if (other == vec) other = (other + 1) % L;
        GClaim memory d = GModel.copy(c);
        d.v[vec] = GModel.resized(c.v[other], c.v[vec].length);
        vm.assume(!GModel.same(d.v[vec], c.v[vec]));
        mustFail(
            CHALLENGER,
            neuronCall(id, l, j % OUT[l], d),
            output ? IArbiter.OutputMismatch.selector : IArbiter.InputMismatch.selector,
            "another layer's committed vector was not refused with its error"
        );
        assertUntouched(id, BOND);
    }

    /// Any mix of correct and fuzzed arguments, in any layer's handler: `keep` chooses, argument
    /// by argument, the correct value or the fuzzed one, so the fuzzer explores near misses (one
    /// argument wrong) as well as random calls. No such challenge can slash an honest claim; with
    /// every argument correct the reason is `NoFraud`.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P2_mixed_correct_and_fuzzed_arguments_never_slash(
        uint256 seed,
        uint256 j,
        uint8 keep,
        uint8 lSeed,
        uint256[8] memory input,
        uint256[8] memory output,
        uint256[8] memory row,
        int32 bias,
        bytes32[] memory proof
    ) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 l = 1 + lSeed % L;
        uint256 n = OUT[l];
        uint256 jj = (keep & 0x40) != 0 ? j : j % n;
        uint256 rj = jj % n;
        uint256 w = Gx.wordsOf(OUT[l - 1]);
        bytes memory out;
        if ((keep & 2) == 0) {
            out = outputOf(c, l);
        } else if (l < L) {
            out = abi.encodePacked(expand(output, Gx.wordsOf(n)));
        } else {
            int32[] memory z = new int32[](CLASSES);
            uint256[] memory words = expand(output, CLASSES);
            for (uint256 k = 0; k < CLASSES; k++) {
                z[k] = int32(uint32(words[k]));
            }
            out = logitWords(z);
        }
        bytes memory data = neuronData(
            l,
            id,
            jj,
            (keep & 1) == 0 ? inputOf(c, l) : abi.encodePacked(expand(input, w)),
            out,
            (keep & 8) == 0 ? Gx.row(l, rj) : expand(row, w),
            (keep & 16) == 0 ? Gx.bias(l, rj) : bias,
            (keep & 32) == 0 ? Gx.proof(l, rj) : proof
        );
        (bool ok, bytes memory reason) = send(CHALLENGER, data);
        require(!ok, "a challenge with fuzzed arguments slashed an honest claim");
        if ((keep & 0x3b) == 0 && jj < n) {
            // Every argument the handler reads is the correct one.
            require(bytes4(reason) == NO_FRAUD, "an all-correct challenge failed for another reason");
        }
        assertUntouched(id, BOND);
    }

    /// Raw calldata mutation: a correct challenge of an honest claim (any layer's handler or the
    /// class challenge), then up to four bytes changed anywhere (selector, ABI offsets and
    /// lengths included), and optionally the calldata cut short or extended with junk. Whatever
    /// the Fe ABI decoder makes of it, an honest claim is never slashed and nothing moves.
    /// forge-config: default.fuzz.runs = 2048
    function testFuzz_P2_mutated_calldata_never_slashes_an_honest_claim(
        uint256 seed,
        uint8 kind,
        uint256 j,
        uint256[4] memory where,
        uint256[4] memory what,
        uint16 cut,
        uint8 mode,
        bytes32 junk
    ) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 check = 1 + kind % (L + 1);
        bytes memory data = checkCall(id, check, check <= L ? j % OUT[check] : 0, c);
        uint256 changes = 1 + mode % 4;
        for (uint256 i = 0; i < changes; i++) {
            uint256 p = where[i] % data.length;
            data[p] = data[p] ^ bytes1(uint8(what[i] | 1));
        }
        if ((mode >> 2) % 3 == 1) {
            uint256 len = cut % data.length;
            assembly ("memory-safe") {
                mstore(data, len)
            }
        } else if ((mode >> 2) % 3 == 2) {
            data = bytes.concat(data, junk, junk);
        }
        (bool ok,) = send(CHALLENGER, data);
        require(!ok, "mutated calldata slashed an honest claim");
        assertUntouched(id, BOND);
    }

    // ----------------------------------------------------------------------------------------
    // P3: a claim with any single wrong value is slashed by the named challenge

    /// A wrong hidden activation of any hidden layer (any other value in 0..127, so both
    /// directions) is named by the walk as neuron l j and slashed by `challengeNeuron<l>(j)`; the
    /// other neurons of that layer still recompute to their committed values.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_wrong_hidden_activation_is_slashed_by_its_neuron(
        uint256 seed,
        uint8 lSeed,
        uint256 j,
        uint256 salt,
        uint256 k
    ) public {
        GClaim memory honest = pick(seed);
        uint256 l = 1 + lSeed % (L - 1);
        j %= OUT[l];
        GClaim memory c = GModel.copy(honest);
        lie(c, l, j, salt);
        (uint256 kind, uint256 named) = walk(c);
        require(kind == l && named == j, "the walk does not name the mutated hidden neuron");
        uint256 id = post(c, CLAIMANT, BOND);
        k %= OUT[l];
        if (k != j) mustFail(CHALLENGER, neuronCall(id, l, k, c), NO_FRAUD, "a correct neuron next to the lie");
        mustPass(CHALLENGER, neuronCall(id, l, j, c), "the named neuron challenge was refused");
        assertSlashed(id, CHALLENGER, BOND, c);
    }

    /// Layers: a lie carried forward. Hidden neuron j of layer l is wrong and every later vector
    /// and the class are recomputed FROM the wrong vector, so the claim is consistent from there
    /// on (what a claimant who wants a false result to look right would post). Only the check at
    /// the lie fails: a neuron of every other layer, another neuron of the same layer and the
    /// class are all `NoFraud` (each handler compares two committed vectors and no more), and
    /// `challengeNeuron<l>(j)` slashes.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_Layers_a_lie_carried_forward_is_slashed_only_at_its_origin(
        uint256[8] memory raw,
        uint8 amp,
        uint256 seed,
        uint8 lSeed,
        uint256 j,
        uint256 salt,
        uint256 probes
    ) public {
        GClaim memory c = (amp & 0x80) != 0 ? GModel.copy(pick(seed)) : GModel.honest(inputFrom(raw, amp));
        uint256 l = 1 + lSeed % (L - 1);
        j %= OUT[l];
        lie(c, l, j, salt);
        GModel.carry(c, l);
        uint256 id = post(c, CLAIMANT, BOND);
        for (uint256 m = 1; m <= L; m++) {
            uint256 k = uint256(keccak256(abi.encode(probes, m))) % OUT[m];
            if (m == l && k == j) continue;
            mustFail(CHALLENGER, neuronCall(id, m, k, c), NO_FRAUD, "a consistent neuron of a carried lie was not NoFraud");
        }
        mustFail(CHALLENGER, classCall(id, c), NO_FRAUD, "the consistent class of a carried lie was not NoFraud");
        (uint256 kind, uint256 named) = walk(c);
        require(kind == l && named == j, "the walk does not name the origin of the lie");
        mustPass(CHALLENGER, neuronCall(id, l, j, c), "the challenge at the origin of the lie was refused");
        assertSlashed(id, CHALLENGER, BOND, c);
    }

    /// A wrong logit (any other int32, extremes included) is slashed by the last layer's handler.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_wrong_logit_is_slashed_by_its_neuron(uint256 seed, uint256 j, int32 v) public {
        GClaim memory honest = pick(seed);
        j %= CLASSES;
        vm.assume(v != honest.z[j]);
        GClaim memory c = GModel.copy(honest);
        c.z[j] = v;
        (uint256 kind, uint256 named) = walk(c);
        require(kind == L && named == j, "the walk does not name the mutated logit");
        uint256 id = post(c, CLAIMANT, BOND);
        mustPass(CHALLENGER, neuronCall(id, L, j, c), "the named logit challenge was refused");
        assertSlashed(id, CHALLENGER, BOND, c);
    }

    /// A wrong class is slashed by `challengeClass`.
    function testFuzz_P3_wrong_class_is_slashed_by_the_class_challenge(uint256 seed, uint256 salt) public {
        GClaim memory honest = pick(seed);
        GClaim memory c = GModel.copy(honest);
        lie(c, L + 1, 0, salt);
        (uint256 kind,) = walk(c);
        require(kind == L + 1 && c.cls != honest.cls && c.cls < CLASSES, "the walk does not name the class");
        uint256 id = post(c, CLAIMANT, BOND);
        mustPass(CHALLENGER, classCall(id, c), "the class challenge was refused");
        assertSlashed(id, CHALLENGER, BOND, c);
    }

    /// Every hidden activation of every hidden layer and every logit of four samples, one up and
    /// one down (where the range allows), and the class one up and one down: each claim is
    /// slashed by the challenge the walk names. Every neuron of every layer, both directions.
    function test_P3_every_value_off_by_one_is_slashed_by_the_named_challenge() public {
        uint256 claims = 0;
        for (uint256 s = 0; s < 4; s++) {
            GClaim memory honest = Gx.sample(s);
            for (uint256 l = 1; l <= L + 1; l++) {
                for (uint256 j = 0; j < (l <= L ? OUT[l] : 1); j++) {
                    for (uint256 up = 0; up < 2; up++) {
                        GClaim memory c = GModel.copy(honest);
                        if (!nudge(c, l, j, up == 1)) continue;
                        (uint256 kind, uint256 named) = firstDifference(c, honest);
                        require(kind == l && named == j, "the walk names another check");
                        uint256 id = post(c, CLAIMANT, BOND);
                        uint256 credit = arbiter.creditOf(CHALLENGER);
                        uint256 burnt = arbiter.burned();
                        mustPass(CHALLENGER, checkCall(id, kind, named, c), "the named challenge was refused");
                        require(statusOf(id) == 2, "the named challenge did not slash");
                        require(arbiter.creditOf(CHALLENGER) == credit + BOND / 2, "half the bond was not credited once");
                        require(arbiter.burned() == burnt + (BOND - BOND / 2), "the other half was not burned once");
                        claims++;
                    }
                }
            }
        }
        // At least one direction for every neuron of every layer of every sample, and the class.
        uint256 neurons = 0;
        for (uint256 l = 1; l <= L; l++) {
            neurons += OUT[l];
        }
        require(claims >= 4 * (neurons + 1), "fewer off-by-one claims than values");
    }

    /// The whole judgement on random claims: a random input, then for every layer either the
    /// vector the spec computes from the vector committed before it or the honest claim's vector
    /// (which stops fitting once an earlier vector lies), with lies in any number of its values;
    /// the class the argmax of the committed logits, or any class. The challenge the walk names
    /// slashes; on copies of the claim, a random neuron of every layer and the class challenge
    /// succeed exactly when GModel says that check fails, and revert with `NoFraud` otherwise. A
    /// false claim can always be slashed and only false checks slash.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_any_lie_on_any_input_is_judged_as_the_spec_says(
        uint256[8] memory raw,
        uint8 amp,
        uint256 lies,
        uint8 cls,
        uint256 probes
    ) public {
        GClaim memory honest = GModel.honest(inputFrom(raw, amp));
        GClaim memory c = GModel.copy(honest);
        for (uint256 l = 1; l <= L; l++) {
            uint256 r = uint256(keccak256(abi.encode(lies, l)));
            // Bit 0: recompute this layer from the committed vector before it. Bits 1 and 2: how
            // many values of it lie (0, 0, 1 or 3).
            if (r & 1 == 1) {
                if (l < L) c.v[l] = GModel.hiddenLayer(l, c.v[l - 1]);
                else c.z = GModel.logits(c.v[L - 1]);
            }
            uint256 count = ((r >> 1) & 3) == 3 ? 3 : ((r >> 1) & 3) == 2 ? 1 : 0;
            for (uint256 i = 0; i < count; i++) {
                lie(c, l, (r >> (8 + 16 * i)) % OUT[l], r >> (64 + 32 * i));
            }
        }
        // Without a lie in the logits the class is often the only lie: keep it honest half the time.
        c.cls = (lies >> 255) == 1 ? cls % uint8(CLASSES) : GModel.argmax(c.z);
        (uint256 kind, uint256 named) = walk(c);

        uint256 a = post(c, CLAIMANT, BOND);
        if (kind == 0) {
            mustFail(CHALLENGER, classCall(a, c), NO_FRAUD, "a claim the walk finds honest was not NoFraud");
        } else {
            mustPass(CHALLENGER, checkCall(a, kind, named, c), "the challenge the walk names was refused");
            require(statusOf(a) == 2, "the challenge the walk names did not slash");
        }
        for (uint256 check = 1; check <= L + 1; check++) {
            uint256 j = check <= L ? uint256(keccak256(abi.encode(probes, check))) % OUT[check] : 0;
            judge(c, check, j);
        }
    }

    /// Posts `c` again and sends the correct challenge of check `kind`, `j`: it must slash exactly
    /// when the spec says the check fails.
    function judge(GClaim memory c, uint256 kind, uint256 j) internal {
        bool wrong = fails(c, kind, j);
        uint256 id = post(c, CLAIMANT, BOND);
        (bool ok, bytes memory ret) = send(CHALLENGER, checkCall(id, kind, j, c));
        require(ok == wrong, "the arbiter judged a check unlike the spec");
        if (!ok) require(bytes4(ret) == NO_FRAUD && ret.length == 4, "a check that holds was refused for another reason");
        require(statusOf(id) == (wrong ? 2 : 1), "the status does not follow the judgement");
    }

    /// The class challenge follows argmax with the lowest index winning ties: random logits, with
    /// `ties` copying the largest value into other positions.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_class_challenge_follows_argmax_lowest_index(
        uint256[8] memory raw,
        uint256 ties,
        uint8 cls,
        uint256 seed
    ) public {
        GClaim memory c = pick(seed);
        uint256[] memory words = expand(raw, CLASSES);
        for (uint256 k = 0; k < CLASSES; k++) {
            c.z[k] = int32(uint32(words[k]));
        }
        int32 top = c.z[GModel.argmax(c.z)];
        for (uint256 k = 0; k < CLASSES; k++) {
            if ((ties >> (k % 256)) & 1 == 1) c.z[k] = top;
        }
        c.cls = cls % uint8(CLASSES);
        uint256 id = post(c, CLAIMANT, BOND);
        bool wrong = c.cls != GModel.argmax(c.z);
        (bool ok, bytes memory ret) = send(CHALLENGER, classCall(id, c));
        require(ok == wrong, "the class challenge disagrees with argmax");
        if (!ok) require(bytes4(ret) == NO_FRAUD, "a right class was refused for another reason");
        require(statusOf(id) == (wrong ? 2 : 1), "the status does not follow the class challenge");
    }

    /// The worst cases of the overflow proof (spec section 3, item 1: an input that made the
    /// recomputation trap would leave a false claim unchallengeable), for EVERY layer. For every
    /// neuron j of layer l, the input vectors of that layer that maximise and minimise its
    /// accumulator (127 where the weight is positive, resp. negative, else 0), and for neuron 0
    /// also the all-0 and all-127 vectors. The claim commits that vector as layer l's input (for
    /// l > 1 a hidden vector no honest run need ever produce, which a claimant may still post)
    /// and everything after it computed from it: neuron j's challenge is `NoFraud`, and with its
    /// output one off it slashes.
    function test_P2_P3_worst_case_accumulators_never_trap() public {
        GClaim memory base = Gx.sample(0);
        for (uint256 l = 1; l <= L; l++) {
            uint256 n = OUT[l - 1];
            for (uint256 j = 0; j < OUT[l]; j++) {
                uint256[] memory row = Gx.row(l, j);
                for (uint256 mode = 0; mode < (j == 0 ? 4 : 2); mode++) {
                    GClaim memory c = GModel.copy(base);
                    c.v[l - 1] = new uint256[](Gx.wordsOf(n));
                    for (uint256 i = 0; i < n; i++) {
                        int8 w = int8(GModel.byteAt(row, i));
                        bool on = mode == 0 ? w > 0 : (mode == 1 ? w < 0 : mode == 3);
                        if (on) GModel.setByte(c.v[l - 1], i, 127);
                    }
                    GModel.carry(c, l - 1);
                    uint256 id = post(c, CLAIMANT, BOND);
                    mustFail(CHALLENGER, neuronCall(id, l, j, c), NO_FRAUD, "a worst-case neuron was not NoFraud");
                    require(nudge(c, l, j, l < L ? GModel.byteAt(c.v[l], j) == 0 : mode % 2 == 1), "no room for a lie");
                    id = post(c, CLAIMANT, BOND);
                    mustPass(CHALLENGER, neuronCall(id, l, j, c), "a worst-case lie was not slashed");
                    require(statusOf(id) == 2, "a worst-case lie was not slashed");
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // Merkle: only a committed row, at its own position, can be used

    bytes4 internal constant NOT_COMMITTED = IArbiter.NotCommitted.selector;

    /// Neuron k's row, bias and proof presented as neuron j (j != k), and neuron j's row and bias
    /// with neuron k's proof: `NotCommitted` in every layer, against an honest claim (where a
    /// foreign row would otherwise recompute to another value and slash it).
    function testFuzz_Merkle_another_neurons_leaf_or_proof_is_refused(uint256 seed, uint256 j, uint256 k) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        for (uint256 l = 1; l <= L; l++) {
            uint256 n = OUT[l];
            if (n < 2) continue;
            uint256 jl = j % n;
            uint256 kl = k % n;
            if (kl == jl) kl = (kl + 1) % n;
            bytes memory input = inputOf(c, l);
            bytes memory output = outputOf(c, l);
            mustFail(
                CHALLENGER,
                neuronData(l, id, jl, input, output, Gx.row(l, kl), Gx.bias(l, kl), Gx.proof(l, kl)),
                NOT_COMMITTED,
                "another neuron's leaf and proof"
            );
            mustFail(
                CHALLENGER,
                neuronData(l, id, jl, input, output, Gx.row(l, jl), Gx.bias(l, jl), Gx.proof(l, kl)),
                NOT_COMMITTED,
                "another neuron's proof"
            );
            mustFail(
                CHALLENGER,
                neuronData(l, id, jl, input, output, Gx.row(l, kl), Gx.bias(l, kl), Gx.proof(l, jl)),
                NOT_COMMITTED,
                "another neuron's row and bias"
            );
            mustFail(
                CHALLENGER,
                neuronData(l, id, jl, input, output, Gx.row(l, jl), Gx.bias(l, kl), Gx.proof(l, jl)),
                Gx.bias(l, kl) == Gx.bias(l, jl) ? NO_FRAUD : NOT_COMMITTED,
                "another neuron's bias"
            );
        }
        assertUntouched(id, BOND);
    }

    /// Layers: a leaf of ANOTHER layer. Into the handler of layer l go, for every other layer m:
    /// the right row and bias with a proof of layer m; and a whole committed neuron of layer m
    /// (row, bias, proof), its row cut or zero-filled to the handler's width, at the same neuron
    /// index wherever layer m has one. Where two layers have rows of the same number of words (the
    /// second and third layers of a 144-32-16-3 network both have one), that last call differs
    /// from a valid challenge of layer m only in the handler it is sent to: the layer number
    /// inside the leaf is the only thing that refuses it.
    function testFuzz_Layers_a_leaf_of_another_layer_is_refused(uint256 seed, uint256 j, uint256 k) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        for (uint256 l = 1; l <= L; l++) {
            uint256 jl = j % OUT[l];
            bytes memory input = inputOf(c, l);
            bytes memory output = outputOf(c, l);
            uint256 w = Gx.wordsOf(OUT[l - 1]);
            for (uint256 m = 1; m <= L; m++) {
                if (m == l) continue;
                uint256 km = k % OUT[m];
                mustFail(
                    CHALLENGER,
                    neuronData(l, id, jl, input, output, Gx.row(l, jl), Gx.bias(l, jl), Gx.proof(m, km)),
                    NOT_COMMITTED,
                    "a proof of another layer"
                );
                // The same index in layer m when it exists there, so only the layer number differs.
                uint256 jm = jl < OUT[m] ? jl : km;
                mustFail(
                    CHALLENGER,
                    neuronData(
                        l, id, jl, input, output, GModel.resized(Gx.row(m, jm), w), Gx.bias(m, jm), Gx.proof(m, jm)
                    ),
                    NOT_COMMITTED,
                    "a committed neuron of another layer"
                );
            }
        }
        assertUntouched(id, BOND);
    }

    /// The correct row with a proof cut short (any element removed), extended (a fuzzed node
    /// inserted anywhere, or the leaf's own sibling repeated), with two siblings swapped, or with
    /// one bit of one sibling flipped, in any layer.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_Merkle_a_damaged_proof_is_refused(
        uint256 seed,
        uint256 j,
        uint8 how,
        uint8 lSeed,
        uint8 pos,
        bytes32 extra,
        uint8 bit
    ) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 l = 1 + lSeed % L;
        uint256 jj = j % OUT[l];
        bytes32[] memory p = Gx.proof(l, jj);
        require(p.length >= 2, "the tree is too small for this test");
        bytes32[] memory q;
        uint256 mode = how % 5;
        if (mode == 0) {
            // Remove element `pos`.
            q = new bytes32[](p.length - 1);
            uint256 cut = pos % p.length;
            for (uint256 i = 0; i + 1 < p.length; i++) {
                q[i] = p[i < cut ? i : i + 1];
            }
        } else if (mode == 1 || mode == 2) {
            // Insert a fuzzed node, or repeat the first sibling, at position `pos`.
            q = new bytes32[](p.length + 1);
            uint256 put = pos % (p.length + 1);
            for (uint256 i = 0; i < q.length; i++) {
                q[i] = i < put ? p[i] : (i == put ? (mode == 1 ? extra : p[0]) : p[i - 1]);
            }
        } else if (mode == 3) {
            // Swap two different siblings.
            q = p;
            uint256 a = pos % p.length;
            uint256 b = (a + 1 + uint256(bit) % (p.length - 1)) % p.length;
            (q[a], q[b]) = (q[b], q[a]);
        } else {
            q = p;
            q[pos % p.length] ^= bytes32(uint256(1) << bit);
        }
        mustFail(
            CHALLENGER,
            neuronData(l, id, jj, inputOf(c, l), outputOf(c, l), Gx.row(l, jj), Gx.bias(l, jj), q),
            NOT_COMMITTED,
            "a damaged proof was not NotCommitted"
        );
        assertUntouched(id, BOND);
    }

    /// One flipped bit anywhere in the row (padding bits included) or the bias, with the correct
    /// proof: `NotCommitted`, in every layer.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_Merkle_a_flipped_bit_in_the_row_or_bias_is_refused(
        uint256 seed,
        uint256 j,
        uint256 bit,
        uint8 lSeed
    ) public {
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 l = 1 + lSeed % L;
        uint256 jj = j % OUT[l];
        uint256[] memory row = Gx.row(l, jj);
        int32 bias = Gx.bias(l, jj);
        uint256 b = bit % (row.length * 256 + 32);
        if (b < row.length * 256) row[b / 256] ^= uint256(1) << (b % 256);
        else bias = int32(uint32(uint32(bias) ^ (uint32(1) << uint32(b - row.length * 256))));
        mustFail(
            CHALLENGER,
            neuronData(l, id, jj, inputOf(c, l), outputOf(c, l), row, bias, Gx.proof(l, jj)),
            NOT_COMMITTED,
            "a flipped bit in the row or bias was not NotCommitted"
        );
        assertUntouched(id, BOND);
    }
}
