// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {ArbiterUser, Claim, CheatsX, Fx, IArbiter, Model, VmX} from "./ArbiterBase.sol";
import {SpecNeuron} from "./SpecNeuron.sol";

/// A challenger or claimant that re-enters the arbiter from its `receive`: a second `withdraw`
/// (must find nothing) and a fresh `claim` paid from the ETH just received.
contract Reenterer {
    IArbiter internal immutable arbiter;
    bool internal armed;
    bool public secondWithdrawFoundNothing;
    bool public claimedInside;
    Claim internal claim;

    constructor(IArbiter a) {
        arbiter = a;
    }

    function load(Claim memory c) external {
        claim = c;
    }

    function challengeClass(uint256 id, int32[10] calldata logits) external {
        arbiter.challengeClass(id, logits);
    }

    function withdraw() external returns (uint256 amount) {
        armed = true;
        amount = arbiter.withdraw();
        armed = false;
    }

    receive() external payable {
        if (!armed) return;
        armed = false;
        try arbiter.withdraw() {
            secondWithdrawFoundNothing = false;
        } catch (bytes memory why) {
            secondWithdrawFoundNothing = bytes4(why) == IArbiter.NothingToWithdraw.selector;
        }
        // Everything it holds: the half bond just paid plus the top-up the test gave it, one bond.
        arbiter.claim{value: address(this).balance}(claim.x0, claim.hidden, claim.logits, claim.cls);
        claimedInside = true;
    }
}

/// Sends its ETH to `target` with SELFDESTRUCT, which no contract can refuse (still true after
/// EIP-6780 for a contract destroyed in the transaction that created it).
contract Forcer {
    constructor(address target) payable {
        assembly {
            selfdestruct(target)
        }
    }
}

/// A contract that refuses ETH.
contract Refuser {
    function challengeClass(IArbiter a, uint256 id, int32[10] calldata logits) external {
        a.challengeClass(id, logits);
    }

    function withdraw(IArbiter a) external returns (uint256) {
        return a.withdraw();
    }
}

/// Fuzz tests of the Fe `Arbiter` for properties P5 (the deadline) and P6 (malformed claims), and
/// for what the spec implies but section 9 does not list: non-canonical ABI encodings, the
/// `Claimed` event as the data a watcher needs, unknown claim numbers, re-entrancy and a payee
/// that refuses ETH. docs/fe-notes.md M3.9 lists what each test covers.
contract ArbiterBoundsTest is ArbiterUser {
    uint256 internal constant BOND = 1 ether;
    uint256 internal constant WINDOW = 1 days;
    uint256 internal constant T0 = 1_000_000;
    address internal constant CLAIMANT = address(0xC1A1);
    address internal constant CHALLENGER = address(0xC4A1);

    // Byte offsets of ABI words in the calldata of each call (4-byte selector included).
    uint256 internal constant CLAIM_LOGITS = 836;
    uint256 internal constant CLAIM_CLASS = 1156;
    uint256 internal constant N1_BIAS = 1700;
    uint256 internal constant N2_LOGITS = 100;
    uint256 internal constant N2_BIAS = 452;
    uint256 internal constant CLASS_LOGITS = 36;

    IArbiter internal arbiter;

    function setUp() public {
        Fx.install();
        arbiter = deployArbiter(BOND, WINDOW);
        vm.warp(T0);
    }

    function pick(uint256 seed) internal view returns (Claim memory) {
        return Fx.sample(seed % Fx.sampleCount());
    }

    function wrongClass(Claim memory honest) internal pure returns (Claim memory c) {
        c = Model.copy(honest);
        c.cls = uint8((honest.cls + 1) % 10);
    }

    function wrongHidden(Claim memory honest, uint256 j) internal pure returns (Claim memory c) {
        c = Model.copy(honest);
        Model.setByte(c.hidden, j, Model.hiddenAt(honest.hidden, j) ^ 1);
    }

    function wrongLogit(Claim memory honest, uint256 j) internal pure returns (Claim memory c) {
        c = Model.copy(honest);
        c.logits[j] = honest.logits[j] + 1;
    }

    function patch(bytes memory data, uint256 offset, uint256 w) internal pure {
        assembly ("memory-safe") {
            mstore(add(add(data, 32), offset), w)
        }
    }

    /// `v`'s low 32 bits under fuzzed high bits: an int32 word that is not sign-extended.
    function dirty32(int32 v, uint256 high) internal pure returns (uint256 w) {
        w = (uint256(uint32(v))) | (high << 32);
    }

    function rawCall(address who, bytes memory data) internal returns (bool ok, bytes memory ret) {
        vm.prank(who);
        (ok, ret) = address(arbiter).call(data);
    }

    function claimData(Claim memory c) internal pure returns (bytes memory) {
        return abi.encodeCall(IArbiter.claim, (c.x0, c.hidden, c.logits, c.cls));
    }

    function nothingPosted() internal view {
        require(arbiter.claimCount() == 0, "a malformed claim was stored");
        require(address(arbiter).balance == 0, "a malformed claim kept ETH");
    }

    // ----------------------------------------------------------------------------------------
    // P5: challenges succeed only before the deadline, finalize only at or after it

    /// Fuzzed window (1 s to 10 years) and claim time (up to 2^40 s). The deadline is claim time
    /// plus window. At deadline - 1 every kind of challenge slashes and finalize is `TooEarly`; at
    /// the deadline every kind of challenge is `TooLate` and finalize returns the bond; later the
    /// result is readable and terminal claims refuse everything.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_P5_challenge_before_the_deadline_finalize_from_it(
        uint256 windowSeed,
        uint256 t0Seed,
        uint256 seed,
        uint8 j,
        uint256 later
    ) public {
        uint256 window = 1 + windowSeed % (10 * 365 days);
        uint256 t0 = 1 + t0Seed % (1 << 40);
        arbiter = deployArbiter(BOND, window);
        vm.warp(t0);
        Claim memory honest = pick(seed);
        Claim[3] memory bad;
        bad[0] = wrongClass(honest);
        bad[1] = wrongHidden(honest, j % 32);
        bad[2] = wrongLogit(honest, j % 10);
        uint256[3] memory early;
        uint256[3] memory late;
        for (uint256 k = 0; k < 3; k++) {
            early[k] = post(arbiter, bad[k], CLAIMANT, BOND);
            late[k] = post(arbiter, bad[k], CLAIMANT, BOND);
        }
        uint256 deadline = t0 + window;
        for (uint256 k = 0; k < 3; k++) {
            require(deadlineOf(arbiter, early[k]) == deadline, "deadline is not claim time plus window");
            require(deadlineOf(arbiter, late[k]) == deadline, "deadline is not claim time plus window");
        }

        vm.warp(deadline - 1);
        for (uint256 k = 0; k < 3; k++) {
            vm.expectRevert(abi.encodeWithSelector(IArbiter.TooEarly.selector, deadline));
            arbiter.finalize(late[k]);
        }
        challengeC(arbiter, early[0], bad[0], CHALLENGER);
        challenge1(arbiter, early[1], j % 32, bad[1], CHALLENGER);
        challenge2(arbiter, early[2], j % 10, bad[2], CHALLENGER);
        for (uint256 k = 0; k < 3; k++) {
            require(statusOf(arbiter, early[k]) == 2, "a proven challenge at deadline - 1 did not slash");
        }
        require(arbiter.creditOf(CHALLENGER) == 3 * (BOND / 2), "challenger credit");
        require(arbiter.burned() == 3 * (BOND - BOND / 2), "burned");

        vm.warp(deadline);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.TooLate.selector, deadline));
        challengeC(arbiter, late[0], bad[0], CHALLENGER);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.TooLate.selector, deadline));
        challenge1(arbiter, late[1], j % 32, bad[1], CHALLENGER);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.TooLate.selector, deadline));
        challenge2(arbiter, late[2], j % 10, bad[2], CHALLENGER);
        for (uint256 k = 0; k < 3; k++) {
            vm.expectRevert(abi.encodeWithSelector(IArbiter.NotFinal.selector, late[k]));
            arbiter.result(late[k]);
            arbiter.finalize(late[k]);
            require(statusOf(arbiter, late[k]) == 3, "finalize at the deadline did not finalise");
            vm.expectRevert(abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, early[k]));
            arbiter.finalize(early[k]);
        }
        require(arbiter.creditOf(CLAIMANT) == 3 * BOND, "claimant credit");
        require(arbiter.creditOf(CHALLENGER) == 3 * (BOND / 2), "challenger credit changed");
        require(arbiter.burned() == 3 * (BOND - BOND / 2), "finalising burned something");

        vm.warp(deadline + later % (1 << 64));
        for (uint256 k = 0; k < 3; k++) {
            require(arbiter.result(late[k]) == bad[k].cls, "result is not the claimed class");
            vm.expectRevert(abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, late[k]));
            arbiter.finalize(late[k]);
        }
        vm.expectRevert(abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, late[0]));
        challengeC(arbiter, late[0], bad[0], CHALLENGER);
        require(address(arbiter).balance == 6 * BOND, "balance");
    }

    /// A window of zero: the deadline is the claim's own block time, so even a proven challenge
    /// in the same block is `TooLate` and finalize succeeds at once (spec section 7: the window
    /// must be long enough; this pins the boundary at zero).
    function testFuzz_P5_zero_window_allows_only_finalize(uint256 seed, uint256 t0Seed) public {
        uint256 t0 = 1 + t0Seed % (1 << 40);
        arbiter = deployArbiter(BOND, 0);
        vm.warp(t0);
        Claim memory c = wrongClass(pick(seed));
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        require(deadlineOf(arbiter, id) == t0, "deadline");
        vm.expectRevert(abi.encodeWithSelector(IArbiter.TooLate.selector, t0));
        challengeC(arbiter, id, c, CHALLENGER);
        arbiter.finalize(id);
        require(statusOf(arbiter, id) == 3 && arbiter.creditOf(CLAIMANT) == BOND, "finalize");
    }

    /// A window so long that claim time plus window passes 2^256 must not wrap to a deadline in
    /// the past (which would make every claim finalisable at once and unchallengeable): the claim
    /// reverts.
    function testFuzz_P5_deadline_cannot_wrap_around(uint256 t0Seed, uint256 windowSeed) public {
        uint256 t0 = 1 + t0Seed % (1 << 40);
        uint256 window = type(uint256).max - windowSeed % t0; // t0 + window >= 2^256
        arbiter = deployArbiter(BOND, window);
        vm.warp(t0);
        Claim memory c = pick(0);
        vm.deal(CLAIMANT, BOND);
        vm.prank(CLAIMANT);
        (bool ok,) = address(arbiter).call{value: BOND}(claimData(c));
        if (ok) {
            require(deadlineOf(arbiter, 0) >= t0, "the deadline wrapped around to the past");
        }
        require(!ok, "a claim whose deadline overflows was accepted");
    }

    // ----------------------------------------------------------------------------------------
    // P6: malformed claims are rejected

    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_input_byte_above_127_is_rejected(uint256 seed, uint16 pos, uint8 v) public {
        Claim memory c = pick(seed);
        Model.setInput(c.x0, pos % 784, v | 0x80);
        vm.expectRevert(IArbiter.MalformedInput.selector);
        post(arbiter, c, CLAIMANT, BOND);
        nothingPosted();
    }

    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_nonzero_input_padding_is_rejected(uint256 seed, uint8 pos, uint8 v, uint128 more) public {
        Claim memory c = pick(seed);
        // One padding byte (any of the 16, any non-zero value), or random padding.
        uint128 dirt = (pos & 0x80) != 0 ? more : uint128(uint256(v == 0 ? 1 : v) << (8 * (pos % 16)));
        vm.assume(dirt != 0);
        c.x0[24] |= uint256(dirt);
        vm.expectRevert(IArbiter.MalformedInput.selector);
        post(arbiter, c, CLAIMANT, BOND);
        nothingPosted();
    }

    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_hidden_byte_above_127_is_rejected(uint256 seed, uint8 pos, uint8 v) public {
        Claim memory c = pick(seed);
        Model.setByte(c.hidden, pos % 32, v | 0x80);
        vm.expectRevert(IArbiter.MalformedHidden.selector);
        post(arbiter, c, CLAIMANT, BOND);
        nothingPosted();
    }

    function testFuzz_P6_class_out_of_range_is_rejected(uint256 seed, uint8 cls) public {
        Claim memory c = pick(seed);
        c.cls = uint8(10 + cls % 246);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.ClassOutOfRange.selector, c.cls));
        post(arbiter, c, CLAIMANT, BOND);
        nothingPosted();
    }

    /// A fuzzed required bond: anything below is `BondTooSmall(sent, required)`; the required
    /// amount or more is accepted and the claim's bond is what was sent.
    function testFuzz_P6_bond_below_the_required_amount_is_rejected(uint256 required, uint256 sent, uint256 extra)
        public
    {
        required = 1 + required % 1e30;
        sent = sent % required;
        extra = extra % 1e30;
        arbiter = deployArbiter(required, WINDOW);
        Claim memory c = pick(required);
        vm.expectRevert(abi.encodeWithSelector(IArbiter.BondTooSmall.selector, sent, required));
        post(arbiter, c, CLAIMANT, sent);
        nothingPosted();
        uint256 id = post(arbiter, c, CLAIMANT, required + extra);
        (, uint256 bond,,,) = arbiter.claimInfo(id);
        require(bond == required + extra, "the stored bond is not the amount sent");
        require(address(arbiter).balance == required + extra, "balance");
    }

    /// Random vectors with `raw` choosing which input words, the hidden word and the class stay
    /// unsanitised: `claim` accepts exactly the claims the spec calls well formed (every input and
    /// hidden byte 0..127, zero input padding, class < 10).
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_claim_accepts_exactly_the_well_formed(
        uint256[25] memory x0,
        uint256 hidden,
        uint8 cls,
        uint32 raw
    ) public {
        Claim memory c;
        for (uint256 k = 0; k < 25; k++) {
            c.x0[k] = (raw >> k) & 1 == 1 ? x0[k] : x0[k] & ~Model.SIGN_BITS;
        }
        if ((raw >> 25) & 1 == 0) c.x0[24] &= ~Model.PADDING_1;
        c.hidden[0] = (raw >> 26) & 1 == 1 ? hidden : hidden & ~Model.SIGN_BITS;
        c.cls = (raw >> 27) & 1 == 1 ? cls : cls % 10;
        bytes memory xb = abi.encodePacked(c.x0);
        bool inputOk = SpecNeuron.allActivations(xb, 784) && SpecNeuron.paddingIsZero(xb, 784);
        bool hiddenOk = SpecNeuron.allActivations(abi.encodePacked(c.hidden), 32);
        bool good = inputOk && hiddenOk && c.cls < 10;
        vm.deal(CLAIMANT, BOND);
        vm.prank(CLAIMANT);
        (bool ok, bytes memory reason) = address(arbiter).call{value: BOND}(claimData(c));
        require(ok == good, ok ? "a malformed claim was accepted" : "a well-formed claim was rejected");
        if (!ok) {
            bytes4 why = bytes4(reason);
            require(
                why == IArbiter.MalformedInput.selector || why == IArbiter.MalformedHidden.selector
                    || why == IArbiter.ClassOutOfRange.selector,
                "rejected for another reason"
            );
            if (!inputOk) require(why == IArbiter.MalformedInput.selector, "the input is checked first");
            nothingPosted();
        }
    }

    // ----------------------------------------------------------------------------------------
    // Non-canonical ABI encodings (raw calldata). Solidity would refuse these words; the spec is
    // silent. Whatever the Fe decoder does, they must not slash an honest claim, corrupt a stored
    // claim, or make a false claim unchallengeable.

    /// A challenge whose `bias` word holds the right int32 in its low 32 bits but is not
    /// sign-extended. If the leaf were built from the cleaned bias and the kernel used the raw
    /// word (or the reverse), an honest claim could be slashed.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_bias_not_sign_extended_cannot_slash(uint256 seed, uint8 j, uint256 high, bool layer2)
        public
    {
        high %= 1 << 224;
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        bytes memory data;
        uint256 w;
        if (!layer2) {
            uint256 jj = j % 32;
            data = abi.encodeCall(
                IArbiter.challengeNeuron1, (id, jj, c.x0, c.hidden, Fx.row1(jj), Fx.bias1(jj), Fx.proof(1, jj))
            );
            w = dirty32(Fx.bias1(jj), high);
            vm.assume(w != uint256(int256(Fx.bias1(jj))));
            patch(data, N1_BIAS, w);
        } else {
            uint256 jj = j % 10;
            data = abi.encodeCall(
                IArbiter.challengeNeuron2, (id, jj, c.hidden, c.logits, Fx.row2(jj), Fx.bias2(jj), Fx.proof(2, jj))
            );
            w = dirty32(Fx.bias2(jj), high);
            vm.assume(w != uint256(int256(Fx.bias2(jj))));
            patch(data, N2_BIAS, w);
        }
        (bool ok,) = rawCall(CHALLENGER, data);
        require(!ok, "a non-canonical bias slashed an honest claim");
        require(statusOf(arbiter, id) == 1, "status");
    }

    /// A challenge whose logits hold a non-sign-extended word at position k: in
    /// `challengeNeuron2` (compared with the recomputed logit) and in `challengeClass` (argmax).
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_logit_not_sign_extended_in_a_challenge_cannot_slash(
        uint256 seed,
        uint8 k,
        uint8 j,
        uint256 high,
        bool classChallenge
    ) public {
        high %= 1 << 224;
        Claim memory c = pick(seed);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        k %= 10;
        uint256 w = dirty32(c.logits[k], high);
        vm.assume(w != uint256(int256(c.logits[k])));
        bytes memory data;
        if (classChallenge) {
            data = abi.encodeCall(IArbiter.challengeClass, (id, c.logits));
            patch(data, CLASS_LOGITS + 32 * uint256(k), w);
        } else {
            uint256 jj = j % 10;
            data = abi.encodeCall(
                IArbiter.challengeNeuron2, (id, jj, c.hidden, c.logits, Fx.row2(jj), Fx.bias2(jj), Fx.proof(2, jj))
            );
            patch(data, N2_LOGITS + 32 * uint256(k), w);
        }
        (bool ok,) = rawCall(CHALLENGER, data);
        require(!ok, "non-canonical logits slashed an honest claim");
        require(statusOf(arbiter, id) == 1, "status");
    }

    /// A claim whose class word has bits above the low byte. Accepted or not, it must not write
    /// into the status byte that shares the class's storage slot, and an accepted claim must be
    /// open, with a class below 10, and judged on that class.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_class_word_with_high_bits_cannot_corrupt_the_claim(uint256 seed, uint256 high, uint8 low)
        public
    {
        high %= 1 << 248;
        vm.assume(high != 0);
        Claim memory honest = pick(seed);
        bytes memory data = claimData(honest);
        uint256 w = uint256(low % 10) | (high << 8);
        patch(data, CLAIM_CLASS, w);
        vm.deal(CLAIMANT, BOND);
        vm.prank(CLAIMANT);
        (bool ok,) = address(arbiter).call{value: BOND}(data);
        if (!ok) {
            nothingPosted();
            return;
        }
        (address claimant, uint256 bond,, uint8 cls, uint8 status) = arbiter.claimInfo(0);
        require(status == 1, "a dirty class word changed the stored status");
        require(claimant == CLAIMANT && bond == BOND, "a dirty class word changed the record");
        require(cls < 10, "a class of 10 or more was stored");
        Claim memory c = Model.copy(honest);
        c.cls = cls;
        bool wrong = cls != honest.cls;
        if (!wrong) vm.expectRevert(IArbiter.NoFraud.selector);
        challengeC(arbiter, 0, c, CHALLENGER);
        require(statusOf(arbiter, 0) == (wrong ? 2 : 1), "the stored class is judged unlike the spec");
    }

    /// A false claim (wrong class) posted with one logit word not sign-extended. If it is
    /// accepted, a watcher who reads the `Claimed` event must be able to slash it: the event's
    /// logits must hash to the commitment, and the class challenge with them must succeed.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_logit_not_sign_extended_in_a_claim_stays_challengeable(uint256 seed, uint8 k, uint256 high)
        public
    {
        high %= 1 << 224;
        Claim memory c = wrongClass(pick(seed));
        k %= 10;
        uint256 w = dirty32(c.logits[k], high);
        vm.assume(w != uint256(int256(c.logits[k])));
        bytes memory data = claimData(c);
        patch(data, CLAIM_LOGITS + 32 * uint256(k), w);
        vm.recordLogs();
        vm.deal(CLAIMANT, BOND);
        vm.prank(CLAIMANT);
        (bool ok,) = address(arbiter).call{value: BOND}(data);
        if (!ok) {
            nothingPosted();
            return;
        }
        Claim memory seen = claimFromEvent(0);
        (,, uint256 logitsHash) = arbiter.claimHashes(0);
        require(logitsHash == uint256(keccak256(abi.encode(seen.logits))), "the event's logits miss the commitment");
        bool wrong = seen.cls != Model.argmax(seen.logits);
        if (!wrong) vm.expectRevert(IArbiter.NoFraud.selector);
        challengeC(arbiter, 0, seen, CHALLENGER);
        require(statusOf(arbiter, 0) == (wrong ? 2 : 1), "the claim as seen in the event is judged unlike the spec");
    }

    // ----------------------------------------------------------------------------------------
    // The Claimed event carries everything a watcher needs (spec section 3, item 3)

    /// Decodes the `Claimed` event of claim `id` from the recorded logs (strict ABI decoding: an
    /// int32 that is not sign-extended reverts here).
    function claimFromEvent(uint256 id) internal returns (Claim memory c) {
        VmX.Log[] memory logs = vm.getRecordedLogs();
        uint256 found = 0;
        for (uint256 i = 0; i < logs.length; i++) {
            if (logs[i].emitter != address(arbiter) || logs[i].topics[0] != IArbiter.Claimed.selector) continue;
            if (uint256(logs[i].topics[1]) != id) continue;
            found++;
            (uint256[] memory x0, uint256[] memory hidden, int32[] memory logits, uint8 cls, uint256 bond,) =
                abi.decode(logs[i].data, (uint256[], uint256[], int32[], uint8, uint256, uint256));
            require(x0.length == 25 && hidden.length == 1 && logits.length == 10, "event array lengths");
            for (uint256 k = 0; k < 25; k++) {
                c.x0[k] = x0[k];
            }
            c.hidden[0] = hidden[0];
            for (uint256 k = 0; k < 10; k++) {
                c.logits[k] = logits[k];
            }
            c.cls = cls;
            (, uint256 stored,,,) = arbiter.claimInfo(id);
            require(bond == stored, "event bond");
        }
        require(found == 1, "not exactly one Claimed event for the claim");
    }

    /// Random well-formed vectors: the event's vectors, class, bond, deadline and indexed fields
    /// equal the claim's, and the stored hashes are the spec's (M3.2).
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_event_and_hashes_carry_the_claim(
        uint256[25] memory x0,
        uint256 hidden,
        int32[10] memory logits,
        uint8 cls,
        uint256 extra
    ) public {
        Claim memory c;
        c.x0 = Model.sanitise(x0);
        c.hidden[0] = hidden & ~Model.SIGN_BITS;
        c.logits = logits;
        c.cls = cls % 10;
        extra %= 1e24;
        vm.recordLogs();
        uint256 id = post(arbiter, c, CLAIMANT, BOND + extra);
        VmX.Log[] memory logs = vm.getRecordedLogs();
        require(logs.length == 1, "claim emitted another number of events");
        require(logs[0].topics.length == 3, "Claimed has two indexed fields");
        require(uint256(logs[0].topics[2]) == uint256(uint160(CLAIMANT)), "indexed claimant");
        (
            uint256[] memory ex0,
            uint256[] memory eh,
            int32[] memory ez,
            uint8 ecls,
            uint256 ebond,
            uint256 edeadline
        ) = abi.decode(logs[0].data, (uint256[], uint256[], int32[], uint8, uint256, uint256));
        require(logs[0].topics[0] == IArbiter.Claimed.selector && uint256(logs[0].topics[1]) == id, "topics");
        require(keccak256(abi.encodePacked(ex0)) == keccak256(abi.encodePacked(c.x0)), "event x0");
        require(eh.length == 1 && eh[0] == c.hidden[0], "event hidden");
        require(ex0.length == 25 && ez.length == 10, "event array lengths");
        for (uint256 k = 0; k < 10; k++) {
            require(ez[k] == c.logits[k], "event logits");
        }
        require(ecls == c.cls && ebond == BOND + extra && edeadline == T0 + WINDOW, "event scalars");
        (uint256 hx, uint256 hh, uint256 hz) = arbiter.claimHashes(id);
        require(hx == uint256(keccak256(abi.encodePacked(c.x0))), "x0 hash");
        require(hh == uint256(keccak256(abi.encodePacked(c.hidden))), "hidden hash");
        require(hz == uint256(keccak256(abi.encode(c.logits))), "logits hash");
        (address who, uint256 bond, uint256 deadline, uint8 scls, uint8 status) = arbiter.claimInfo(id);
        require(who == CLAIMANT && bond == BOND + extra && deadline == T0 + WINDOW, "claimInfo");
        require(scls == c.cls && status == 1, "claimInfo class or status");
    }

    /// The one event of the calls recorded since `vm.recordLogs`, from the arbiter.
    function oneLog() internal returns (VmX.Log memory) {
        VmX.Log[] memory logs = vm.getRecordedLogs();
        require(logs.length == 1, "not exactly one event");
        require(logs[0].emitter == address(arbiter), "the event is not the arbiter's");
        return logs[0];
    }

    /// Every way a claim ends, and a withdrawal, emits its event with the right fields: a watcher
    /// follows the contract through these (the mutation test found that nothing checked them).
    function testFuzz_every_outcome_emits_its_event(uint256 seed, uint8 j, uint8 cls) public {
        Claim memory honest = pick(seed);
        uint256 j1 = j % 32;
        uint256 j2 = j % 10;

        Claim memory c = wrongHidden(honest, j1);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        vm.recordLogs();
        challenge1(arbiter, id, j1, c, CHALLENGER);
        VmX.Log memory log = oneLog();
        require(log.topics.length == 3 && log.topics[0] == IArbiter.NeuronSlashed.selector, "layer 1 topic");
        require(uint256(log.topics[1]) == id && address(uint160(uint256(log.topics[2]))) == CHALLENGER, "layer 1 who");
        require(keccak256(log.data) == keccak256(abi.encode(uint256(1), j1)), "layer 1 layer and neuron");

        c = wrongLogit(honest, j2);
        id = post(arbiter, c, CLAIMANT, BOND);
        vm.recordLogs();
        challenge2(arbiter, id, j2, c, CHALLENGER);
        log = oneLog();
        require(log.topics.length == 3 && log.topics[0] == IArbiter.NeuronSlashed.selector, "layer 2 topic");
        require(uint256(log.topics[1]) == id && address(uint160(uint256(log.topics[2]))) == CHALLENGER, "layer 2 who");
        require(keccak256(log.data) == keccak256(abi.encode(uint256(2), j2)), "layer 2 layer and neuron");

        c = wrongClass(honest);
        id = post(arbiter, c, CLAIMANT, BOND);
        vm.recordLogs();
        challengeC(arbiter, id, c, CHALLENGER);
        log = oneLog();
        require(log.topics.length == 3 && log.topics[0] == IArbiter.ClassSlashed.selector, "class topic");
        require(uint256(log.topics[1]) == id && address(uint160(uint256(log.topics[2]))) == CHALLENGER, "class who");
        require(log.data.length == 0, "class data");

        c = Model.copy(honest);
        c.cls = cls % 10;
        id = post(arbiter, c, CLAIMANT, BOND);
        vm.warp(deadlineOf(arbiter, id));
        vm.recordLogs();
        arbiter.finalize(id);
        log = oneLog();
        require(log.topics.length == 2 && log.topics[0] == IArbiter.Finalized.selector, "final topic");
        require(uint256(log.topics[1]) == id, "final id");
        require(keccak256(log.data) == keccak256(abi.encode(uint256(c.cls))), "final class");

        vm.recordLogs();
        vm.prank(CHALLENGER);
        arbiter.withdraw();
        log = oneLog();
        require(log.topics.length == 2 && log.topics[0] == IArbiter.Withdrawn.selector, "withdraw topic");
        require(address(uint160(uint256(log.topics[1]))) == CHALLENGER, "withdraw account");
        require(keccak256(log.data) == keccak256(abi.encode(3 * (BOND / 2))), "withdraw amount");
    }

    // ----------------------------------------------------------------------------------------
    // Unknown claims, re-entrancy, a payee that refuses ETH

    /// Any number at or past `claimCount()` is `UnknownClaim` in every handler and view.
    function testFuzz_unknown_claim_numbers_are_refused(uint256 seed, uint256 id) public {
        Claim memory c = pick(seed);
        post(arbiter, c, CLAIMANT, BOND);
        id = id == 0 ? 1 : id;
        bytes memory unknown = abi.encodeWithSelector(IArbiter.UnknownClaim.selector, id);
        vm.expectRevert(unknown);
        arbiter.finalize(id);
        vm.expectRevert(unknown);
        challengeC(arbiter, id, c, CHALLENGER);
        vm.expectRevert(unknown);
        challenge1(arbiter, id, 0, c, CHALLENGER);
        vm.expectRevert(unknown);
        challenge2(arbiter, id, 0, c, CHALLENGER);
        vm.expectRevert(unknown);
        arbiter.result(id);
        vm.expectRevert(unknown);
        arbiter.claimInfo(id);
        vm.expectRevert(unknown);
        arbiter.claimHashes(id);
        require(statusOf(arbiter, 0) == 1 && address(arbiter).balance == BOND, "claim 0 changed");
    }

    /// A challenger contract whose `receive` calls `withdraw` again and posts a new claim with
    /// the ETH it just received (half a bond, topped up to one): it is paid its half once, the
    /// inner withdraw finds nothing, the new claim is stored, and the balance still equals open
    /// bonds plus credits plus the burned half (P1). A bystander's
    /// open bond is in the arbiter throughout: without it a contract that zeroes the credit after
    /// the transfer passes this test, because its inner withdraw fails for lack of funds, not for
    /// lack of credit (found by the mutant `withdraw_zeroes_after_transfer`, docs/fe-notes.md M3.9).
    function test_reentrant_withdraw_pays_once() public {
        Reenterer r = new Reenterer(arbiter);
        Claim memory honest = pick(1);
        r.load(honest);
        Claim memory c = wrongClass(honest);
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        uint256 bystander = post(arbiter, pick(3), address(0xB1), BOND);
        r.challengeClass(id, c.logits);
        require(arbiter.creditOf(address(r)) == BOND / 2, "credit");
        vm.deal(address(r), BOND - BOND / 2);
        require(r.withdraw() == BOND / 2, "withdraw amount");
        require(r.secondWithdrawFoundNothing(), "the inner withdraw was not refused for lack of credit");
        require(r.claimedInside(), "the re-entrant claim did not run");
        require(arbiter.creditOf(address(r)) == 0, "credit after");
        require(statusOf(arbiter, bystander) == 1, "the bystander's claim changed");
        require(arbiter.claimCount() == 3 && statusOf(arbiter, 2) == 1, "the re-entrant claim");
        require(
            address(arbiter).balance == 2 * BOND + arbiter.burned() && arbiter.burned() == BOND - BOND / 2,
            "balance: two open bonds and the burned half, no credit"
        );
    }

    /// A payee that refuses ETH: `withdraw` reverts with `TransferFailed` and the credit stays.
    function test_refused_transfer_keeps_the_credit() public {
        Refuser r = new Refuser();
        Claim memory c = wrongClass(pick(2));
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        r.challengeClass(arbiter, id, c.logits);
        vm.expectRevert(IArbiter.TransferFailed.selector);
        r.withdraw(arbiter);
        require(arbiter.creditOf(address(r)) == BOND / 2, "the credit was lost");
        require(address(arbiter).balance == BOND, "balance");
    }

    /// The arbiter has no fallback: a plain transfer, or ETH sent with an unknown selector, is
    /// refused, so ETH enters only through `claim` (what P1 relies on).
    function test_plain_eth_transfer_is_refused() public {
        vm.deal(address(this), 2 ether);
        (bool ok,) = address(arbiter).call{value: 1 ether}("");
        require(!ok, "a plain transfer was accepted");
        (ok,) = address(arbiter).call{value: 1 ether}(hex"deadbeef");
        require(!ok, "ETH with an unknown selector was accepted");
        (ok,) = address(arbiter).call{value: 1 ether}(abi.encodeCall(IArbiter.withdraw, ()));
        require(!ok, "ETH sent to withdraw was accepted");
        require(address(arbiter).balance == 0, "balance");
    }

    /// P1 as an equality cannot hold against ETH forced in by SELFDESTRUCT (or as a block reward
    /// or withdrawal recipient): the balance then exceeds open bonds plus credits. This pins the
    /// limit of P1 (it is `>=` in general, `==` only without forced ETH) and checks that nothing
    /// else changes: the forced ETH is never paid out and every credit is still paid in full.
    function test_P1_forced_eth_breaks_only_the_equality() public {
        Claim memory c = wrongClass(pick(4));
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        new Forcer{value: 5 wei}(address(arbiter));
        require(address(arbiter).balance == BOND + 5 wei, "the forced ETH did not arrive");
        challengeC(arbiter, id, c, CHALLENGER);
        vm.prank(CHALLENGER);
        require(arbiter.withdraw() == BOND / 2, "withdraw paid another amount");
        require(address(arbiter).balance == BOND - BOND / 2 + 5 wei, "only the burned half and the forced ETH stay");
        require(arbiter.burned() == BOND - BOND / 2, "burned");
        require(arbiter.creditOf(CHALLENGER) == 0 && arbiter.creditOf(CLAIMANT) == 0, "credits");
    }

    /// Nothing stops a claimant from challenging its own false claim (docs/spec-dispute.md section
    /// 7). The claim is still slashed and never final, so the false result cannot be used, and the
    /// slash burns half the bond: the claimant gets back only the other half, so a false claim costs
    /// at least half its bond whoever slashes it. (Before the burn, a claimant that won the race
    /// against a real challenger lost only gas.)
    function test_claimant_slashing_its_own_false_claim_loses_half_the_bond() public {
        Claim memory c = wrongClass(pick(0));
        uint256 id = post(arbiter, c, CLAIMANT, BOND);
        uint256 before = CLAIMANT.balance;
        challengeC(arbiter, id, c, CLAIMANT);
        require(statusOf(arbiter, id) == 2, "the claim is not slashed");
        require(arbiter.creditOf(CLAIMANT) == BOND / 2, "the claimant was credited more than half");
        require(arbiter.burned() == BOND - BOND / 2, "the other half was not burned");
        vm.prank(CLAIMANT);
        arbiter.withdraw();
        require(CLAIMANT.balance == before + BOND / 2, "the claimant got back more than half");
        require(address(arbiter).balance == BOND - BOND / 2, "the burned half left the arbiter");
        vm.expectRevert(abi.encodeWithSelector(IArbiter.NotFinal.selector, id));
        arbiter.result(id);
    }

    /// The constructor does not check its parameters (a limit, spec section 7): with a required
    /// bond of zero a claim needs no ETH and a slash pays nothing. Anyone can read `bond()`.
    function test_zero_bond_deployment_has_no_penalty() public {
        arbiter = deployArbiter(0, WINDOW);
        Claim memory c = wrongClass(pick(0));
        uint256 id = post(arbiter, c, CLAIMANT, 0);
        challengeC(arbiter, id, c, CHALLENGER);
        require(statusOf(arbiter, id) == 2 && arbiter.creditOf(CHALLENGER) == 0, "slash without a bond");
        vm.prank(CHALLENGER);
        vm.expectRevert(IArbiter.NothingToWithdraw.selector);
        arbiter.withdraw();
    }

    /// `withdraw` with no credit is `NothingToWithdraw` and changes nothing.
    function testFuzz_withdraw_without_credit_is_refused(address who) public {
        vm.assume(who != address(0) && uint160(who) > 0xffff);
        post(arbiter, pick(3), CLAIMANT, BOND);
        uint256 before = who.balance;
        vm.prank(who);
        vm.expectRevert(IArbiter.NothingToWithdraw.selector);
        arbiter.withdraw();
        require(who.balance == before && address(arbiter).balance == BOND, "balances changed");
    }
}
