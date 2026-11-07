// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {ArbiterUser, Claim, Fx, IArbiter, Model} from "./ArbiterBase.sol";

/// Fuzz tests of the Fe `Arbiter` for properties P2 (an honest claim cannot be slashed) and P3
/// (any single wrong value is slashed by the challenge the spec's walk names), and of the Merkle
/// commitment (docs/spec-dispute.md sections 2, 4, 6 and 9). The expected behaviour comes from
/// the spec, through `Model` (the network in plain Solidity over the fixture rows), never from
/// the Fe code. docs/fe-notes.md M3.9 lists what each test covers.
contract ArbiterFuzzTest is ArbiterUser {
    uint256 internal constant BOND = 1 ether;
    uint256 internal constant WINDOW = 1 days;
    uint256 internal constant T0 = 1_000_000;
    address internal constant CLAIMANT = address(0xC1A1);
    address internal constant CHALLENGER = address(0xC4A1);

    IArbiter internal arbiter;

    function setUp() public {
        Fx.install();
        arbiter = deployArbiter(BOND, WINDOW);
        vm.warp(T0);
    }

    // ----------------------------------------------------------------------------------------
    // Helpers

    function pick(uint256 seed) internal view returns (Claim memory) {
        return Fx.sample(seed % Fx.sampleCount());
    }

    /// The first failing check of the spec's walk (section 6) for `claimed`, given the honest
    /// claim for the same input: 1 and j (hidden neuron j), 2 and j (logit j), 3 (the class), or
    /// 0 (no check fails: the claim is honest).
    function walk(Claim memory claimed, Claim memory honest) internal view returns (uint256 kind, uint256 j) {
        for (j = 0; j < 32; j++) {
            if (Model.hiddenAt(claimed.hidden, j) != Model.hiddenAt(honest.hidden, j)) return (1, j);
        }
        for (j = 0; j < 10; j++) {
            if (claimed.logits[j] != Model.neuron2(j, claimed.hidden)) return (2, j);
        }
        if (claimed.cls != Model.argmax(claimed.logits)) return (3, 0);
        return (0, 0);
    }

    /// Sends the challenge `kind`, `j` names (with the committed vectors and the correct row).
    function send(uint256 id, uint256 kind, uint256 j, Claim memory c, address who) internal {
        if (kind == 1) challenge1(arbiter, id, j, c, who);
        else if (kind == 2) challenge2(arbiter, id, j, c, who);
        else challengeC(arbiter, id, c, who);
    }

    /// The claim is still open and no ETH or credit moved.
    function assertUntouched(uint256 id, uint256 balance) internal view {
        require(statusOf(arbiter, id) == 1, "a refused challenge changed the claim's status");
        require(address(arbiter).balance == balance, "a refused challenge moved ETH");
        require(arbiter.creditOf(CHALLENGER) == 0, "a refused challenge credited the challenger");
        require(arbiter.creditOf(CLAIMANT) == 0, "a refused challenge credited the claimant");
    }

    /// The claim is slashed, its bond credited once to `who`, and every further action on it is
    /// refused; the credit can be withdrawn exactly once.
    function assertSlashed(uint256 id, address who, uint256 bond, Claim memory c) internal {
        require(statusOf(arbiter, id) == 2, "a proven challenge did not slash");
        require(arbiter.creditOf(who) == bond, "the bond was not credited to the challenger");
        require(arbiter.creditOf(CLAIMANT) == 0, "the claimant was credited for a slashed claim");
        vm.expectRevert(abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, id));
        challengeC(arbiter, id, c, who);
        vm.warp(deadlineOf(arbiter, id));
        vm.expectRevert(abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, id));
        arbiter.finalize(id);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.NotFinal.selector, id));
        arbiter.result(id);
        uint256 before = who.balance;
        vm.prank(who);
        require(arbiter.withdraw() == bond, "withdraw returned another amount");
        require(who.balance == before + bond, "withdraw paid another amount");
        require(arbiter.creditOf(who) == 0, "the credit survived the withdrawal");
        vm.prank(who);
        vm.expectRevert(IArbiter.NothingToWithdraw.selector);
        arbiter.withdraw();
    }

    // ----------------------------------------------------------------------------------------
    // The oracle itself

    /// The fixtures and Model agree with the reference trace: every hidden activation, logit and
    /// class of every fixture sample is what Model computes from its input. Without this the other
    /// tests would test the oracle, not the arbiter.
    function test_oracle_reproduces_the_reference_samples() public view {
        uint256 n = Fx.sampleCount();
        require(n >= 10, "too few fixture samples");
        for (uint256 s = 0; s < n; s++) {
            Claim memory c = Fx.sample(s);
            Claim memory m = Model.honest(c.x0);
            require(m.hidden[0] == c.hidden[0], "Model's hidden vector differs from trace.jsonl");
            for (uint256 j = 0; j < 10; j++) {
                require(m.logits[j] == c.logits[j], "Model's logits differ from trace.jsonl");
            }
            require(m.cls == c.cls, "Model's class differs from trace.jsonl");
        }
    }

    // ----------------------------------------------------------------------------------------
    // P2: a claim equal to the reference output cannot be slashed by any challenge

    /// Every neuron of both layers with its committed row, and the class challenge, against the
    /// honest claim of every fixture sample: all `NoFraud`, nothing moves.
    function test_P2_every_correct_challenge_of_every_honest_sample_reverts() public {
        for (uint256 s = 0; s < Fx.sampleCount(); s++) {
            Claim memory c = Fx.sample(s);
            uint256 id = post(arbiter, c, CLAIMANT, BOND);
            for (uint256 j = 0; j < 32; j++) {
                vm.expectRevert(IArbiter.NoFraud.selector);
                challenge1(arbiter, id, j, c, CHALLENGER);
            }
            for (uint256 j = 0; j < 10; j++) {
                vm.expectRevert(IArbiter.NoFraud.selector);
                challenge2(arbiter, id, j, c, CHALLENGER);
            }
            vm.expectRevert(IArbiter.NoFraud.selector);
            challengeC(arbiter, id, c, CHALLENGER);
            assertUntouched(id, BOND * (s + 1));
        }
    }

    /// The honest claim for a random well-formed input (computed by Model) survives a random
    /// neuron challenge of each layer and the class challenge.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P2_honest_claim_on_a_random_input_cannot_be_slashed(uint256[25] memory x0, uint8 j1, uint8 j2)
        public
    {
        Claim memory c = Model.honest(Model.sanitise(x0));
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        vm.expectRevert(IArbiter.NoFraud.selector);
        challenge1(arbiter, id, j1 % 32, c, CHALLENGER);
        vm.expectRevert(IArbiter.NoFraud.selector);
        challenge2(arbiter, id, j2 % 10, c, CHALLENGER);
        vm.expectRevert(IArbiter.NoFraud.selector);
        challengeC(arbiter, id, c, CHALLENGER);
        assertUntouched(id, BOND);
    }

    /// A neuron index past the layer is refused in both layers.
    function testFuzz_P2_neuron_index_out_of_range_is_refused(uint256 seed, uint256 j) public {
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        uint256 j1 = j < 32 ? j + 32 : j;
        uint256 j2 = j < 10 ? j + 10 : j;
        vm.prank(CHALLENGER);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.NeuronOutOfRange.selector, j1));
        arbiter.challengeNeuron1(id, j1, c.x0, c.hidden, Fx.row1(j1 % 32), Fx.bias1(j1 % 32), Fx.proof(1, j1 % 32));
        vm.prank(CHALLENGER);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.NeuronOutOfRange.selector, j2));
        arbiter.challengeNeuron2(id, j2, c.hidden, c.logits, Fx.row2(j2 % 10), Fx.bias2(j2 % 10), Fx.proof(2, j2 % 10));
        assertUntouched(id, BOND);
    }

    /// Vectors that differ from the committed ones in one bit (input, hidden or logits, in each
    /// handler that takes them) are refused with the matching error, even though they would
    /// change the recomputation.
    function testFuzz_P2_substituted_vectors_are_refused(uint256 seed, uint256 j, uint256 bit, uint8 which)
        public
    {
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        Claim memory d = Model.copy(c);
        uint256 w = which % 5;
        if (w == 0) {
            uint256 b = bit % 6272; // a value bit of the 784 input bytes
            d.x0[b / 256] ^= uint256(1) << (255 - b % 256);
            vm.expectRevert(IArbiter.InputMismatch.selector);
            challenge1(arbiter, id, j % 32, d, CHALLENGER);
        } else if (w == 1 || w == 2) {
            d.hidden[0] ^= uint256(1) << (bit % 256);
            vm.expectRevert(w == 1 ? IArbiter.OutputMismatch.selector : IArbiter.InputMismatch.selector);
            if (w == 1) challenge1(arbiter, id, j % 32, d, CHALLENGER);
            else challenge2(arbiter, id, j % 10, d, CHALLENGER);
        } else {
            uint256 k = (bit >> 8) % 10;
            d.logits[k] = int32(uint32(d.logits[k]) ^ (uint32(1) << uint32(bit % 32)));
            vm.expectRevert(IArbiter.OutputMismatch.selector);
            if (w == 3) challenge2(arbiter, id, j % 10, d, CHALLENGER);
            else challengeC(arbiter, id, d, CHALLENGER);
        }
        assertUntouched(id, BOND);
    }

    /// Any mix of correct and fuzzed arguments: `keep` chooses, argument by argument, the correct
    /// value or the fuzzed one, so the fuzzer explores near misses (one argument wrong) as well as
    /// random calls. No such challenge can slash an honest claim; with every argument correct the
    /// reason is `NoFraud`.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P2_mixed_correct_and_fuzzed_arguments_never_slash(
        uint256 seed,
        uint256 j,
        uint8 keep,
        uint256[25] memory x0,
        uint256 hidden,
        int32[10] memory logits,
        uint256[25] memory row,
        int32 bias,
        bytes32[] memory proof
    ) public {
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        bool layer2 = (keep & 0x80) != 0;
        uint256 jj = (keep & 0x40) != 0 ? j : j % (layer2 ? 10 : 32);
        uint256 rj = jj % (layer2 ? 10 : 32);
        bytes memory data;
        if (!layer2) {
            data = abi.encodeCall(
                IArbiter.challengeNeuron1,
                (
                    id,
                    jj,
                    (keep & 1) == 0 ? c.x0 : x0,
                    (keep & 2) == 0 ? c.hidden : [hidden],
                    (keep & 8) == 0 ? Fx.row1(rj) : row,
                    (keep & 16) == 0 ? Fx.bias1(rj) : bias,
                    (keep & 32) == 0 ? Fx.proof(1, rj) : proof
                )
            );
        } else {
            data = abi.encodeCall(
                IArbiter.challengeNeuron2,
                (
                    id,
                    jj,
                    (keep & 2) == 0 ? c.hidden : [hidden],
                    (keep & 4) == 0 ? c.logits : logits,
                    (keep & 8) == 0 ? Fx.row2(rj) : [row[0]],
                    (keep & 16) == 0 ? Fx.bias2(rj) : bias,
                    (keep & 32) == 0 ? Fx.proof(2, rj) : proof
                )
            );
        }
        vm.prank(CHALLENGER);
        (bool ok, bytes memory reason) = address(arbiter).call(data);
        require(!ok, "a challenge with fuzzed arguments slashed an honest claim");
        if ((keep & (layer2 ? 0x3e : 0x3b)) == 0 && jj < (layer2 ? 10 : 32)) {
            // Every argument the handler reads is the correct one.
            require(bytes4(reason) == IArbiter.NoFraud.selector, "an all-correct challenge failed for another reason");
        }
        assertUntouched(id, BOND);
    }

    /// Raw calldata mutation: a correct challenge of an honest claim (any of the three handlers),
    /// then up to four bytes changed anywhere (selector, ABI offsets and lengths included), and
    /// optionally the calldata cut short or extended with junk. Whatever the Fe ABI decoder makes
    /// of it, an honest claim is never slashed and nothing moves.
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
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        bytes memory data;
        kind %= 3;
        if (kind == 0) {
            j %= 32;
            data = abi.encodeCall(
                IArbiter.challengeNeuron1, (id, j, c.x0, c.hidden, Fx.row1(j), Fx.bias1(j), Fx.proof(1, j))
            );
        } else if (kind == 1) {
            j %= 10;
            data = abi.encodeCall(
                IArbiter.challengeNeuron2, (id, j, c.hidden, c.logits, Fx.row2(j), Fx.bias2(j), Fx.proof(2, j))
            );
        } else {
            data = abi.encodeCall(IArbiter.challengeClass, (id, c.logits));
        }
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
        vm.prank(CHALLENGER);
        (bool ok,) = address(arbiter).call(data);
        require(!ok, "mutated calldata slashed an honest claim");
        assertUntouched(id, BOND);
    }

    // ----------------------------------------------------------------------------------------
    // P3: a claim with any single wrong value is slashed by the named challenge

    /// A wrong hidden activation (any other value in 0..127, so both directions) is named by the
    /// walk as neuron 1 j and slashed by `challengeNeuron1(j)`; the other hidden neurons still
    /// recompute to their committed values.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_wrong_hidden_activation_is_slashed_by_its_neuron(uint256 seed, uint8 j, uint8 v, uint8 k)
        public
    {
        Claim memory honest = pick(seed);
        j %= 32;
        v %= 128;
        vm.assume(v != Model.hiddenAt(honest.hidden, j));
        Claim memory c = Model.copy(honest);
        Model.setByte(c.hidden, j, v);
        (uint256 kind, uint256 named) = walk(c, honest);
        require(kind == 1 && named == j, "the walk does not name the mutated hidden neuron");
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        k %= 32;
        if (k != j) {
            vm.expectRevert(IArbiter.NoFraud.selector);
            challenge1(arbiter, id, k, c, CHALLENGER);
        }
        challenge1(arbiter, id, j, c, CHALLENGER);
        assertSlashed(id, CHALLENGER, BOND, c);
    }

    /// A wrong logit (any other int32, extremes included) is slashed by `challengeNeuron2(j)`.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_wrong_logit_is_slashed_by_its_neuron(uint256 seed, uint8 j, int32 v) public {
        Claim memory honest = pick(seed);
        j %= 10;
        vm.assume(v != honest.logits[j]);
        Claim memory c = Model.copy(honest);
        c.logits[j] = v;
        (uint256 kind, uint256 named) = walk(c, honest);
        require(kind == 2 && named == j, "the walk does not name the mutated logit");
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        challenge2(arbiter, id, j, c, CHALLENGER);
        assertSlashed(id, CHALLENGER, BOND, c);
    }

    /// A wrong class is slashed by `challengeClass`.
    function testFuzz_P3_wrong_class_is_slashed_by_the_class_challenge(uint256 seed, uint8 cls) public {
        Claim memory honest = pick(seed);
        cls %= 10;
        vm.assume(cls != honest.cls);
        Claim memory c = Model.copy(honest);
        c.cls = cls;
        (uint256 kind,) = walk(c, honest);
        require(kind == 3, "the walk does not name the class");
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        challengeC(arbiter, id, c, CHALLENGER);
        assertSlashed(id, CHALLENGER, BOND, c);
    }

    /// Every hidden activation and every logit of four samples, one up and one down (where the
    /// range allows), and the class one up and one down: each claim is slashed by the challenge
    /// the walk names. 344 claims, every layer and both directions.
    function test_P3_every_value_off_by_one_is_slashed_by_the_named_challenge() public {
        for (uint256 s = 0; s < 4; s++) {
            Claim memory honest = Fx.sample(s);
            for (uint256 j = 0; j < 32; j++) {
                uint8 h = Model.hiddenAt(honest.hidden, j);
                for (uint256 up = 0; up < 2; up++) {
                    if ((up == 1 && h == 127) || (up == 0 && h == 0)) continue;
                    Claim memory c = Model.copy(honest);
                    Model.setByte(c.hidden, j, up == 1 ? h + 1 : h - 1);
                    sendNamedAndCheck(c, honest, 1, j);
                }
            }
            for (uint256 j = 0; j < 10; j++) {
                for (uint256 up = 0; up < 2; up++) {
                    Claim memory c = Model.copy(honest);
                    c.logits[j] = up == 1 ? honest.logits[j] + 1 : honest.logits[j] - 1;
                    sendNamedAndCheck(c, honest, 2, j);
                }
            }
            for (uint256 up = 0; up < 2; up++) {
                Claim memory c = Model.copy(honest);
                c.cls = uint8((honest.cls + (up == 1 ? 1 : 9)) % 10);
                sendNamedAndCheck(c, honest, 3, 0);
            }
        }
    }

    function sendNamedAndCheck(Claim memory c, Claim memory honest, uint256 wantKind, uint256 wantJ) internal {
        (uint256 kind, uint256 j) = walk(c, honest);
        require(kind == wantKind && j == wantJ, "the walk names another check");
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        uint256 credit = arbiter.creditOf(CHALLENGER);
        send(id, kind, j, c, CHALLENGER);
        require(statusOf(arbiter, id) == 2, "the named challenge did not slash");
        require(arbiter.creditOf(CHALLENGER) == credit + BOND, "the bond was not credited once");
    }

    /// The whole judgement on random claims: a random input, its honest claim from Model, then
    /// lies in any number of hidden activations and logits and the class (`lies` picks which).
    /// The challenge the walk names slashes; on copies of the claim, a random neuron of each layer
    /// and the class challenge succeed exactly when Model says that check fails, and revert with
    /// `NoFraud` otherwise. A false claim can always be slashed and only false checks slash.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_any_lie_on_any_input_is_judged_as_the_spec_says(
        uint256[25] memory x0,
        uint256 lieWord,
        uint64 lies,
        int32[10] memory fakeLogits,
        uint8 cls,
        uint8 r1,
        uint8 r2
    ) public {
        Claim memory honest = Model.honest(Model.sanitise(x0));
        Claim memory c = Model.copy(honest);
        for (uint256 j = 0; j < 32; j++) {
            if ((lies >> j) & 1 == 1) Model.setByte(c.hidden, j, Model.byteOf(lieWord, j) & 0x7f);
        }
        for (uint256 j = 0; j < 10; j++) {
            if ((lies >> (32 + j)) & 1 == 1) c.logits[j] = fakeLogits[j];
        }
        // Without a lie in the logits the class is often the only lie: keep it honest half the time.
        c.cls = (lies >> 42) & 1 == 1 ? cls % 10 : Model.argmax(c.logits);
        (uint256 kind, uint256 named) = walk(c, honest);

        uint256 a = post(arbiter, c, CLAIMANT, BOND);
        if (kind == 0) {
            vm.expectRevert(IArbiter.NoFraud.selector);
            challengeC(arbiter, a, c, CHALLENGER);
        } else {
            send(a, kind, named, c, CHALLENGER);
            require(statusOf(arbiter, a) == 2, "the challenge the walk names did not slash");
        }

        r1 %= 32;
        r2 %= 10;
        bool wrong1 = Model.hiddenAt(c.hidden, r1) != Model.hiddenAt(honest.hidden, r1);
        bool wrong2 = c.logits[r2] != Model.neuron2(r2, c.hidden);
        bool wrongC = c.cls != Model.argmax(c.logits);
        judge(c, 1, r1, wrong1);
        judge(c, 2, r2, wrong2);
        judge(c, 3, 0, wrongC);
    }

    function judge(Claim memory c, uint256 kind, uint256 j, bool wrong) internal {
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        if (!wrong) vm.expectRevert(IArbiter.NoFraud.selector);
        send(id, kind, j, c, CHALLENGER);
        require(statusOf(arbiter, id) == (wrong ? 2 : 1), "the arbiter judged a check unlike the spec");
    }

    /// The class challenge follows argmax with the lowest index winning ties: random logits, with
    /// `ties` copying the largest value into other positions.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P3_class_challenge_follows_argmax_lowest_index(
        int32[10] memory z,
        uint16 ties,
        uint8 cls,
        uint256 seed
    ) public {
        int32 top = z[Model.argmax(z)];
        for (uint256 k = 0; k < 10; k++) {
            if ((ties >> k) & 1 == 1) z[k] = top;
        }
        Claim memory c = pick(seed);
        c.logits = z;
        c.cls = cls % 10;
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        bool wrong = c.cls != Model.argmax(z);
        if (!wrong) vm.expectRevert(IArbiter.NoFraud.selector);
        challengeC(arbiter, id, c, CHALLENGER);
        require(statusOf(arbiter, id) == (wrong ? 2 : 1), "the class challenge disagrees with argmax");
    }

    /// The worst cases of the overflow proof (spec section 3, item 1: an input that made the
    /// recomputation trap would leave a false claim unchallengeable). For every hidden neuron j,
    /// the inputs that maximise and minimise its accumulator (127 where the weight is positive,
    /// resp. negative, else 0), plus the all-0 and all-127 inputs: the honest claim survives
    /// neuron j's challenge, and a claim with hidden j one off is slashed by it. For every logit
    /// k, the hidden vectors that maximise and minimise it: a claim with logit k one off is
    /// slashed by `challengeNeuron2(k)` and the correct one survives.
    function test_P2_P3_worst_case_accumulators_never_trap() public {
        for (uint256 j = 0; j < 34; j++) {
            for (uint256 sign = 0; sign < 2; sign++) {
                if (j >= 32 && sign == 1) continue;
                uint256[25] memory x0;
                uint256[25] memory row = Fx.row1(j % 32);
                for (uint256 i = 0; i < 784; i++) {
                    int8 w = int8(Model.inputAt(row, i));
                    bool on = j == 32 ? false : (j == 33 ? true : (sign == 0 ? w > 0 : w < 0));
                    if (on) Model.setInput(x0, i, 127);
                }
                uint256 n = j % 32;
                Claim memory honest = Model.honest(x0);
                uint256 id = post(arbiter, honest, CLAIMANT, BOND);
                vm.expectRevert(IArbiter.NoFraud.selector);
                challenge1(arbiter, id, n, honest, CHALLENGER);
                Claim memory c = Model.copy(honest);
                Model.setByte(c.hidden, n, Model.hiddenAt(honest.hidden, n) ^ 1);
                id = post(arbiter, c, CLAIMANT, BOND);
                challenge1(arbiter, id, n, c, CHALLENGER);
                require(statusOf(arbiter, id) == 2, "a worst-case layer 1 lie was not slashed");
            }
        }
        Claim memory base = Fx.sample(0);
        for (uint256 k = 0; k < 10; k++) {
            for (uint256 sign = 0; sign < 2; sign++) {
                Claim memory c = Model.copy(base);
                uint256 row = Fx.row2(k)[0];
                c.hidden[0] = 0;
                for (uint256 i = 0; i < 32; i++) {
                    int8 w = int8(Model.byteOf(row, i));
                    if (sign == 0 ? w > 0 : w < 0) Model.setByte(c.hidden, i, 127);
                }
                for (uint256 m = 0; m < 10; m++) {
                    c.logits[m] = Model.neuron2(m, c.hidden);
                }
                c.cls = Model.argmax(c.logits);
                uint256 id = post(arbiter, c, CLAIMANT, BOND);
                vm.expectRevert(IArbiter.NoFraud.selector);
                challenge2(arbiter, id, k, c, CHALLENGER);
                c.logits[k] += sign == 0 ? int32(-1) : int32(1);
                c.cls = Model.argmax(c.logits);
                id = post(arbiter, c, CLAIMANT, BOND);
                challenge2(arbiter, id, k, c, CHALLENGER);
                require(statusOf(arbiter, id) == 2, "a worst-case layer 2 lie was not slashed");
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // Merkle: only a committed row, at its own position, can be used

    /// Neuron k's row, bias and proof presented as neuron j (j != k), and neuron j's row and bias
    /// with neuron k's proof: `NotCommitted` in both layers, against an honest claim (where a
    /// foreign row would otherwise recompute to another value and slash it).
    function testFuzz_Merkle_another_neurons_leaf_or_proof_is_refused(uint256 seed, uint8 j, uint8 k) public {
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        uint256 j1 = j % 32;
        uint256 k1 = k % 32;
        if (k1 == j1) k1 = (k1 + 1) % 32;
        uint256 j2 = j % 10;
        uint256 k2 = k % 10;
        if (k2 == j2) k2 = (k2 + 1) % 10;
        vm.startPrank(CHALLENGER);
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron1(id, j1, c.x0, c.hidden, Fx.row1(k1), Fx.bias1(k1), Fx.proof(1, k1));
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron1(id, j1, c.x0, c.hidden, Fx.row1(j1), Fx.bias1(j1), Fx.proof(1, k1));
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron1(id, j1, c.x0, c.hidden, Fx.row1(k1), Fx.bias1(k1), Fx.proof(1, j1));
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron2(id, j2, c.hidden, c.logits, Fx.row2(k2), Fx.bias2(k2), Fx.proof(2, k2));
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron2(id, j2, c.hidden, c.logits, Fx.row2(j2), Fx.bias2(j2), Fx.proof(2, k2));
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron2(id, j2, c.hidden, c.logits, Fx.row2(k2), Fx.bias2(k2), Fx.proof(2, j2));
        vm.stopPrank();
        assertUntouched(id, BOND);
    }

    /// A proof from the other layer: layer 2 neuron j's proof for layer 1 neuron j (and the
    /// reverse), and a layer 1 row's first word with its own proof as a layer 2 row.
    function testFuzz_Merkle_a_proof_from_the_other_layer_is_refused(uint256 seed, uint8 j, uint8 k) public {
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        uint256 j1 = j % 32;
        uint256 j2 = k % 10;
        vm.startPrank(CHALLENGER);
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron1(id, j1, c.x0, c.hidden, Fx.row1(j1), Fx.bias1(j1), Fx.proof(2, j2));
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron2(id, j2, c.hidden, c.logits, Fx.row2(j2), Fx.bias2(j2), Fx.proof(1, j1));
        vm.expectRevert(IArbiter.NotCommitted.selector);
        arbiter.challengeNeuron2(id, j2, c.hidden, c.logits, [Fx.row1(j1)[0]], Fx.bias1(j1), Fx.proof(1, j1));
        vm.stopPrank();
        assertUntouched(id, BOND);
    }

    /// The correct row with a proof cut short (any element removed), extended (a fuzzed node
    /// inserted anywhere, or the leaf's own sibling repeated), with two siblings swapped, or with
    /// one bit of one sibling flipped.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_Merkle_a_damaged_proof_is_refused(uint256 seed, uint8 j, uint8 how, uint8 pos, bytes32 extra, uint8 bit)
        public
    {
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        bool layer2 = (how & 0x80) != 0;
        uint256 jj = j % (layer2 ? 10 : 32);
        bytes32[] memory p = Fx.proof(layer2 ? 2 : 1, jj);
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
        vm.prank(CHALLENGER);
        vm.expectRevert(IArbiter.NotCommitted.selector);
        if (!layer2) arbiter.challengeNeuron1(id, jj, c.x0, c.hidden, Fx.row1(jj), Fx.bias1(jj), q);
        else arbiter.challengeNeuron2(id, jj, c.hidden, c.logits, Fx.row2(jj), Fx.bias2(jj), q);
        assertUntouched(id, BOND);
    }

    /// One flipped bit anywhere in the row (padding bits included) or the bias, with the correct
    /// proof: `NotCommitted`, in both layers.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_Merkle_a_flipped_bit_in_the_row_or_bias_is_refused(uint256 seed, uint8 j, uint16 bit, bool layer2)
        public
    {
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        vm.prank(CHALLENGER);
        vm.expectRevert(IArbiter.NotCommitted.selector);
        if (!layer2) {
            uint256 jj = j % 32;
            uint256[25] memory row = Fx.row1(jj);
            int32 bias = Fx.bias1(jj);
            uint256 b = bit % (25 * 256 + 32);
            if (b < 25 * 256) row[b / 256] ^= uint256(1) << (b % 256);
            else bias = int32(uint32(uint32(bias) ^ (uint32(1) << uint32(b - 25 * 256))));
            arbiter.challengeNeuron1(id, jj, c.x0, c.hidden, row, bias, Fx.proof(1, jj));
        } else {
            uint256 jj = j % 10;
            uint256[1] memory row = Fx.row2(jj);
            int32 bias = Fx.bias2(jj);
            uint256 b = bit % (256 + 32);
            if (b < 256) row[0] ^= uint256(1) << b;
            else bias = int32(uint32(uint32(bias) ^ (uint32(1) << uint32(b - 256))));
            arbiter.challengeNeuron2(id, jj, c.hidden, c.logits, row, bias, Fx.proof(2, jj));
        }
        assertUntouched(id, BOND);
    }
}
