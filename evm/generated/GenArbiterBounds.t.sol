// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {IArbiter, VmX} from "./ArbiterBase.sol";
import {GClaim, GModel, GenUser, Gx} from "./GenBase.sol";
import {SpecNeuron} from "./SpecNeuron.sol";

/// A challenger that re-enters the arbiter from its `receive`: a second `withdraw` (must find
/// nothing) and a fresh claim paid from the ETH just received. The calls are prepared by the test
/// as calldata, because their signatures depend on the model.
contract GenReenterer {
    address internal immutable arbiter;
    bool internal armed;
    bool public secondWithdrawFoundNothing;
    bool public claimedInside;
    bytes internal claimCall;

    constructor(address a) {
        arbiter = a;
    }

    function load(bytes memory data) external {
        claimCall = data;
    }

    function send(bytes memory data) external {
        (bool ok, bytes memory ret) = arbiter.call(data);
        if (!ok) {
            assembly ("memory-safe") {
                revert(add(ret, 32), mload(ret))
            }
        }
    }

    function withdraw() external returns (uint256 amount) {
        armed = true;
        amount = IArbiter(arbiter).withdraw();
        armed = false;
    }

    receive() external payable {
        if (!armed) return;
        armed = false;
        try IArbiter(arbiter).withdraw() {
            secondWithdrawFoundNothing = false;
        } catch (bytes memory why) {
            secondWithdrawFoundNothing = bytes4(why) == IArbiter.NothingToWithdraw.selector;
        }
        (bool ok,) = arbiter.call{value: address(this).balance}(claimCall);
        require(ok, "the re-entrant claim was refused");
        claimedInside = true;
    }
}

/// A contract that refuses ETH.
contract GenRefuser {
    function send(address a, bytes memory data) external {
        (bool ok,) = a.call(data);
        require(ok, "the call was refused");
    }

    function withdraw(IArbiter a) external returns (uint256) {
        return a.withdraw();
    }
}

/// Sends its ETH to `target` with SELFDESTRUCT, which no contract can refuse.
contract GenForcer {
    constructor(address target) payable {
        assembly {
            selfdestruct(target)
        }
    }
}

/// Fuzz tests of a generated Fe `Arbiter` with any number of layers for properties P5 (the
/// deadline) and P6 (malformed claims), and for what the spec implies but section 9 does not
/// list: non-canonical ABI encodings, the `Claimed` event and `claimHashes` as the data a watcher
/// needs, the slash events, unknown claim numbers, re-entrancy and a payee that refuses ETH. They
/// follow evm/test/ArbiterBounds.t.sol one for one, with every check that names a layer made for
/// every layer: each hidden vector of the claim (its range and its padding), the deadline of each
/// layer's handler, the layer number in each `NeuronSlashed` event.
contract GenArbiterBoundsTest is GenUser {
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

    function pick(uint256 seed) internal view returns (GClaim memory) {
        return Gx.sample(seed % Gx.sampleCount());
    }

    /// A copy of `honest` with one lie at check `kind` (a neuron of layer 1 .. L, or the class).
    function wrongAt(GClaim memory honest, uint256 kind, uint256 j, uint256 salt)
        internal
        view
        returns (GClaim memory c)
    {
        c = GModel.copy(honest);
        lie(c, kind, j, salt);
    }

    function nothingPosted() internal view {
        require(arbiter.claimCount() == 0, "a malformed claim was stored");
        require(address(arbiter).balance == 0, "a malformed claim kept ETH");
    }

    /// The claim must be refused with exactly `err` and nothing stored.
    function mustRefuseClaim(GClaim memory c, bytes memory err, string memory what) internal {
        (bool ok, bytes memory ret) = postRaw(claimData(c), CLAIMANT, BOND);
        require(!ok && keccak256(ret) == keccak256(err), what);
        nothingPosted();
    }

    function patch(bytes memory data, uint256 offset, uint256 w) internal pure {
        assembly ("memory-safe") {
            mstore(add(add(data, 32), offset), w)
        }
    }

    /// `v`'s low 32 bits under fuzzed high bits: an int32 word that is not sign-extended.
    function dirty32(int32 v, uint256 high) internal pure returns (uint256) {
        return uint256(uint32(v)) | (high << 32);
    }

    /// Words of all the vectors before the logits in `claim(...)`.
    function vectorWords() internal view returns (uint256 n) {
        for (uint256 k = 0; k < L; k++) {
            n += Gx.wordsOf(OUT[k]);
        }
    }

    /// Byte offset (selector included) of the first logit in `claim(...)`.
    function claimLogitsAt() internal view returns (uint256) {
        return 4 + 32 * vectorWords();
    }

    /// Byte offset of the first logit in `challengeNeuron<L>(...)`: past id, j and the input.
    function lastLayerLogitsAt() internal view returns (uint256) {
        return 4 + 64 + 32 * Gx.wordsOf(OUT[L - 1]);
    }

    function wordIn(bytes memory b, uint256 i) internal pure returns (uint256 w) {
        assembly ("memory-safe") {
            w := mload(add(add(b, 32), mul(i, 32)))
        }
    }

    // ----------------------------------------------------------------------------------------
    // P5: challenges succeed only before the deadline, finalize only at or after it

    /// Fuzzed window (1 s to 10 years) and claim time (up to 2^40 s). For every check (a neuron of
    /// every layer, the class) one false claim is challenged at deadline - 1 (slashed; finalize is
    /// `TooEarly`) and one at the deadline (`TooLate`; finalize returns the bond). Later the result
    /// is readable, and both kinds of terminal claim refuse every handler with `ClaimNotOpen`.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_P5_challenge_before_the_deadline_finalize_from_it(
        uint256 windowSeed,
        uint256 t0Seed,
        uint256 seed,
        uint256 j,
        uint256 salt,
        uint256 later
    ) public {
        uint256 window = 1 + windowSeed % (10 * 365 days);
        uint256 t0 = 1 + t0Seed % (1 << 40);
        arbiter = deployArbiter(BOND, window);
        vm.warp(t0);
        GClaim memory honest = pick(seed);
        uint256 kinds = L + 1;
        GClaim[] memory bad = new GClaim[](kinds);
        uint256[] memory js = new uint256[](kinds);
        uint256[] memory early = new uint256[](kinds);
        uint256[] memory late = new uint256[](kinds);
        for (uint256 k = 0; k < kinds; k++) {
            js[k] = k < L ? uint256(keccak256(abi.encode(j, k))) % OUT[k + 1] : 0;
            bad[k] = wrongAt(honest, k + 1, js[k], salt);
            early[k] = post(bad[k], CLAIMANT, BOND);
            late[k] = post(bad[k], CLAIMANT, BOND);
        }
        uint256 deadline = t0 + window;
        for (uint256 k = 0; k < kinds; k++) {
            require(deadlineOf(early[k]) == deadline, "deadline is not claim time plus window");
            require(deadlineOf(late[k]) == deadline, "deadline is not claim time plus window");
        }

        vm.warp(deadline - 1);
        for (uint256 k = 0; k < kinds; k++) {
            vm.expectRevert(abi.encodeWithSelector(IArbiter.TooEarly.selector, deadline));
            arbiter.finalize(late[k]);
        }
        for (uint256 k = 0; k < kinds; k++) {
            mustPass(CHALLENGER, checkCall(early[k], k + 1, js[k], bad[k]), "a proven challenge at deadline - 1");
            require(statusOf(early[k]) == 2, "a proven challenge at deadline - 1 did not slash");
        }
        require(arbiter.creditOf(CHALLENGER) == kinds * (BOND / 2), "challenger credit");
        require(arbiter.burned() == kinds * (BOND - BOND / 2), "burned");

        vm.warp(deadline);
        bytes memory tooLate = abi.encodeWithSelector(IArbiter.TooLate.selector, deadline);
        for (uint256 k = 0; k < kinds; k++) {
            mustFail(CHALLENGER, checkCall(late[k], k + 1, js[k], bad[k]), tooLate, "a challenge at the deadline");
        }
        for (uint256 k = 0; k < kinds; k++) {
            vm.expectRevert(abi.encodeWithSelector(IArbiter.NotFinal.selector, late[k]));
            arbiter.result(late[k]);
            arbiter.finalize(late[k]);
            require(statusOf(late[k]) == 3, "finalize at the deadline did not finalise");
            vm.expectRevert(abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, early[k]));
            arbiter.finalize(early[k]);
        }
        require(arbiter.creditOf(CLAIMANT) == kinds * BOND, "claimant credit");
        require(arbiter.creditOf(CHALLENGER) == kinds * (BOND / 2), "challenger credit changed");
        require(arbiter.burned() == kinds * (BOND - BOND / 2), "finalising burned something");

        vm.warp(deadline + later % (1 << 64));
        for (uint256 k = 0; k < kinds; k++) {
            require(arbiter.result(late[k]) == bad[k].cls, "result is not the claimed class");
            vm.expectRevert(abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, late[k]));
            arbiter.finalize(late[k]);
            // Every handler, on a final and on a slashed claim.
            for (uint256 m = 1; m <= L + 1; m++) {
                mustFail(
                    CHALLENGER,
                    checkCall(late[k], m, 0, bad[k]),
                    abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, late[k]),
                    "a final claim was challenged"
                );
                mustFail(
                    CHALLENGER,
                    checkCall(early[k], m, 0, bad[k]),
                    abi.encodeWithSelector(IArbiter.ClaimNotOpen.selector, early[k]),
                    "a slashed claim was challenged"
                );
            }
        }
        require(address(arbiter).balance == 2 * kinds * BOND, "balance");
    }

    /// A window of zero: the deadline is the claim's own block time, so even a proven challenge
    /// of any layer in the same block is `TooLate` and finalize succeeds at once.
    function testFuzz_P5_zero_window_allows_only_finalize(uint256 seed, uint256 t0Seed, uint256 kSeed, uint256 j)
        public
    {
        uint256 t0 = 1 + t0Seed % (1 << 40);
        arbiter = deployArbiter(BOND, 0);
        vm.warp(t0);
        uint256 kind = 1 + kSeed % (L + 1);
        j = kind <= L ? j % OUT[kind] : 0;
        GClaim memory c = wrongAt(pick(seed), kind, j, seed);
        uint256 id = post(c, CLAIMANT, BOND);
        require(deadlineOf(id) == t0, "deadline");
        mustFail(
            CHALLENGER, checkCall(id, kind, j, c), abi.encodeWithSelector(IArbiter.TooLate.selector, t0), "TooLate"
        );
        arbiter.finalize(id);
        require(statusOf(id) == 3 && arbiter.creditOf(CLAIMANT) == BOND, "finalize");
    }

    /// A window so long that claim time plus window passes 2^256 must not wrap to a deadline in
    /// the past: the claim reverts.
    function testFuzz_P5_deadline_cannot_wrap_around(uint256 t0Seed, uint256 windowSeed) public {
        uint256 t0 = 1 + t0Seed % (1 << 40);
        uint256 window = type(uint256).max - windowSeed % t0; // t0 + window >= 2^256
        arbiter = deployArbiter(BOND, window);
        vm.warp(t0);
        (bool ok,) = postRaw(claimData(pick(0)), CLAIMANT, BOND);
        if (ok) {
            require(deadlineOf(0) >= t0, "the deadline wrapped around to the past");
        }
        require(!ok, "a claim whose deadline overflows was accepted");
    }

    // ----------------------------------------------------------------------------------------
    // P6: malformed claims are rejected

    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_input_byte_above_127_is_rejected(uint256 seed, uint256 pos, uint8 v) public {
        GClaim memory c = pick(seed);
        GModel.setByte(c.v[0], pos % OUT[0], v | 0x80);
        mustRefuseClaim(c, abi.encodePacked(IArbiter.MalformedInput.selector), "an input byte above 127");
    }

    /// One padding byte of the input (any of them, any non-zero value), or random padding.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_nonzero_input_padding_is_rejected(uint256 seed, uint8 pos, uint8 v, uint256 more) public {
        uint256 mask = GModel.paddingMask(OUT[0]);
        if (mask == 0) return; // an input of a whole number of words has no padding
        GClaim memory c = pick(seed);
        uint256 padBytes = 32 - OUT[0] % 32;
        uint256 dirt = (pos & 0x80) != 0 ? more & mask : uint256(v == 0 ? 1 : v) << (8 * (pos % padBytes));
        vm.assume(dirt != 0);
        c.v[0][c.v[0].length - 1] |= dirt;
        mustRefuseClaim(c, abi.encodePacked(IArbiter.MalformedInput.selector), "non-zero input padding");
    }

    /// Layers: a byte above 127 in ANY hidden vector is `MalformedHidden`.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_hidden_byte_above_127_is_rejected(uint256 seed, uint8 lSeed, uint256 pos, uint8 v) public {
        GClaim memory c = pick(seed);
        uint256 l = 1 + lSeed % (L - 1);
        GModel.setByte(c.v[l], pos % OUT[l], v | 0x80);
        mustRefuseClaim(c, abi.encodePacked(IArbiter.MalformedHidden.selector), "a hidden byte above 127");
    }

    /// Layers: non-zero padding in any hidden vector whose length is not a multiple of 32 (in
    /// a 144-32-16-3 network the second hidden vector has 16 padding bytes) is `MalformedHidden`.
    /// The layer is drawn from the hidden layers that have padding; a model with none (the
    /// 784-128-64-10 one) has nothing to test here, and the test then passes without a claim.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_nonzero_hidden_padding_is_rejected(uint256 seed, uint8 lSeed, uint8 pos, uint8 v, uint256 more)
        public
    {
        uint256[] memory padded = new uint256[](L);
        uint256 count = 0;
        for (uint256 k = 1; k < L; k++) {
            if (GModel.paddingMask(OUT[k]) != 0) padded[count++] = k;
        }
        if (count == 0) return;
        GClaim memory c = pick(seed);
        uint256 l = padded[lSeed % count];
        uint256 mask = GModel.paddingMask(OUT[l]);
        uint256 padBytes = 32 - OUT[l] % 32;
        uint256 dirt = (pos & 0x80) != 0 ? more & mask : uint256(v == 0 ? 1 : v) << (8 * (pos % padBytes));
        vm.assume(dirt != 0);
        c.v[l][c.v[l].length - 1] |= dirt;
        mustRefuseClaim(c, abi.encodePacked(IArbiter.MalformedHidden.selector), "non-zero hidden padding");
    }

    function testFuzz_P6_class_out_of_range_is_rejected(uint256 seed, uint8 cls) public {
        GClaim memory c = pick(seed);
        c.cls = uint8(CLASSES + cls % (256 - CLASSES));
        mustRefuseClaim(
            c, abi.encodeWithSelector(IArbiter.ClassOutOfRange.selector, c.cls), "a class out of range was accepted"
        );
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
        GClaim memory c = pick(required);
        (bool ok, bytes memory ret) = postRaw(claimData(c), CLAIMANT, sent);
        require(
            !ok && keccak256(ret) == keccak256(abi.encodeWithSelector(IArbiter.BondTooSmall.selector, sent, required)),
            "BondTooSmall"
        );
        nothingPosted();
        uint256 id = post(c, CLAIMANT, required + extra);
        (, uint256 bond,,,) = arbiter.claimInfo(id);
        require(bond == required + extra, "the stored bond is not the amount sent");
        require(address(arbiter).balance == required + extra, "balance");
    }

    /// Random vectors with `raw` choosing which words of which vectors, and the class, stay
    /// unsanitised: `claim` accepts exactly the claims the spec calls well formed (every input and
    /// hidden byte 0..127, zero padding in every vector, class < CLASSES), and the input is checked
    /// before the hidden vectors.
    /// forge-config: default.fuzz.runs = 1024
    function testFuzz_P6_claim_accepts_exactly_the_well_formed(uint256[8] memory words, uint8 cls, uint256 raw)
        public
    {
        GClaim memory c;
        c.v = new uint256[][](L);
        uint256 bit = 0;
        bool inputOk = true;
        bool hiddenOk = true;
        for (uint256 k = 0; k < L; k++) {
            uint256 n = Gx.wordsOf(OUT[k]);
            c.v[k] = new uint256[](n);
            for (uint256 w = 0; w < n; w++) {
                uint256 x = uint256(keccak256(abi.encode(words, k, w)));
                // Each word: fuzzed bytes kept as they are with probability 1/8, else sanitised.
                if ((raw >> (bit++ % 256)) & 7 != 0 || (raw >> 200) & 1 == 1) x &= ~GModel.SIGN_BITS;
                if (w == n - 1 && (raw >> ((bit++ + 64) % 256)) & 3 != 0) x &= ~GModel.paddingMask(OUT[k]);
                c.v[k][w] = x;
            }
            bytes memory b = abi.encodePacked(c.v[k]);
            bool ok = SpecNeuron.allActivations(b, OUT[k]) && SpecNeuron.paddingIsZero(b, OUT[k]);
            if (k == 0) inputOk = ok;
            else hiddenOk = hiddenOk && ok;
        }
        c.z = new int32[](CLASSES);
        for (uint256 k = 0; k < CLASSES; k++) {
            c.z[k] = int32(uint32(words[k % 8] >> (32 * (k / 8))));
        }
        c.cls = (raw >> 255) == 1 ? cls : uint8(cls % CLASSES);
        bool good = inputOk && hiddenOk && c.cls < CLASSES;
        (bool ok, bytes memory reason) = postRaw(claimData(c), CLAIMANT, BOND);
        require(ok == good, ok ? "a malformed claim was accepted" : "a well-formed claim was rejected");
        if (!ok) {
            bytes4 why = bytes4(reason);
            if (!inputOk) require(why == IArbiter.MalformedInput.selector, "the input is checked first");
            else if (!hiddenOk) require(why == IArbiter.MalformedHidden.selector, "a hidden vector, then the class");
            else require(why == IArbiter.ClassOutOfRange.selector, "the class");
            nothingPosted();
        }
    }

    // ----------------------------------------------------------------------------------------
    // Non-canonical ABI encodings (raw calldata). Whatever the Fe decoder does with them, they
    // must not slash an honest claim, corrupt a stored claim, or make a false claim unchallengeable.

    /// A challenge of any layer whose `bias` word holds the right int32 in its low 32 bits but is
    /// not sign-extended.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_bias_not_sign_extended_cannot_slash(uint256 seed, uint256 j, uint256 high, uint8 lSeed)
        public
    {
        high %= 1 << 224;
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 l = 1 + lSeed % L;
        uint256 jj = j % OUT[l];
        int32 bias = Gx.bias(l, jj);
        uint256 w = dirty32(bias, high);
        vm.assume(w != uint256(int256(bias)));
        bytes memory data =
            neuronDataRaw(l, id, jj, inputOf(c, l), outputOf(c, l), Gx.row(l, jj), w, Gx.proof(l, jj));
        (bool ok,) = send(CHALLENGER, data);
        require(!ok, "a non-canonical bias slashed an honest claim");
        require(statusOf(id) == 1, "status");
    }

    /// Logits with a non-sign-extended word at position k: in the last layer's handler (compared
    /// with the recomputed logit) and in `challengeClass` (argmax).
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_logit_not_sign_extended_in_a_challenge_cannot_slash(
        uint256 seed,
        uint256 k,
        uint256 j,
        uint256 high,
        bool classChallenge
    ) public {
        high %= 1 << 224;
        GClaim memory c = pick(seed);
        uint256 id = post(c, CLAIMANT, BOND);
        k %= CLASSES;
        uint256 w = dirty32(c.z[k], high);
        vm.assume(w != uint256(int256(c.z[k])));
        bytes memory data;
        if (classChallenge) {
            data = classCall(id, c);
            patch(data, 36 + 32 * k, w);
        } else {
            data = neuronCall(id, L, j % CLASSES, c);
            patch(data, lastLayerLogitsAt() + 32 * k, w);
        }
        (bool ok,) = send(CHALLENGER, data);
        require(!ok, "non-canonical logits slashed an honest claim");
        require(statusOf(id) == 1, "status");
    }

    /// A claim whose class word has bits above the low byte. Accepted or not, it must not write
    /// into the status byte that shares the class's storage slot, and an accepted claim must be
    /// open, with a class below CLASSES, and judged on that class.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_class_word_with_high_bits_cannot_corrupt_the_claim(uint256 seed, uint256 high, uint8 low)
        public
    {
        high %= 1 << 248;
        vm.assume(high != 0);
        GClaim memory honest = pick(seed);
        bytes memory data = claimData(honest);
        patch(data, claimLogitsAt() + 32 * CLASSES, uint256(low % CLASSES) | (high << 8));
        (bool ok,) = postRaw(data, CLAIMANT, BOND);
        if (!ok) {
            nothingPosted();
            return;
        }
        (address claimant, uint256 bond,, uint8 cls, uint8 status) = arbiter.claimInfo(0);
        require(status == 1, "a dirty class word changed the stored status");
        require(claimant == CLAIMANT && bond == BOND, "a dirty class word changed the record");
        require(cls < CLASSES, "a class out of range was stored");
        GClaim memory c = GModel.copy(honest);
        c.cls = cls;
        bool wrong = cls != honest.cls;
        (bool slashed, bytes memory ret) = send(CHALLENGER, classCall(0, c));
        require(slashed == wrong, "the stored class is judged unlike the spec");
        if (!slashed) require(bytes4(ret) == IArbiter.NoFraud.selector, "NoFraud");
    }

    /// A false claim (wrong class) posted with one logit word not sign-extended. If it is
    /// accepted, a watcher who reads the `Claimed` event must be able to slash it.
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_ABI_logit_not_sign_extended_in_a_claim_stays_challengeable(uint256 seed, uint256 k, uint256 high)
        public
    {
        high %= 1 << 224;
        GClaim memory c = wrongAt(pick(seed), L + 1, 0, seed);
        k %= CLASSES;
        uint256 w = dirty32(c.z[k], high);
        vm.assume(w != uint256(int256(c.z[k])));
        bytes memory data = claimData(c);
        patch(data, claimLogitsAt() + 32 * k, w);
        vm.recordLogs();
        (bool ok,) = postRaw(data, CLAIMANT, BOND);
        if (!ok) {
            nothingPosted();
            return;
        }
        (GClaim memory seen,,) = claimFromEvent(0);
        (bool read, uint256[] memory h) = hashesOf(0);
        require(read && h[L] == uint256(keccak256(abi.encodePacked(seen.z))), "the event's logits miss the commitment");
        bool wrong = seen.cls != GModel.argmax(seen.z);
        (bool slashed,) = send(CHALLENGER, classCall(0, seen));
        require(slashed == wrong, "the claim as seen in the event is judged unlike the spec");
    }

    // ----------------------------------------------------------------------------------------
    // The Claimed event and claimHashes carry everything a watcher needs (spec section 3)

    /// Decodes the one `Claimed` event of claim `id` from the recorded logs, strictly: every
    /// array has the length the model fixes and every logit is a sign-extended int32.
    function claimFromEvent(uint256 id) internal returns (GClaim memory c, uint256 bond, uint256 deadline) {
        VmX.Log[] memory logs = vm.getRecordedLogs();
        uint256 found = 0;
        for (uint256 i = 0; i < logs.length; i++) {
            if (logs[i].emitter != address(arbiter) || logs[i].topics[0] != claimedTopic) continue;
            if (uint256(logs[i].topics[1]) != id) continue;
            found++;
            require(logs[i].topics.length == 3, "Claimed has two indexed fields");
            (c, bond, deadline) = decodeClaimed(logs[i].data);
        }
        require(found == 1, "not exactly one Claimed event for the claim");
    }

    /// The data of a `Claimed` event: L vectors and the logits as dynamic arrays, then the class,
    /// the bond and the deadline (spec section 3, item 3).
    function decodeClaimed(bytes memory data) internal view returns (GClaim memory c, uint256 bond, uint256 deadline) {
        c.v = new uint256[][](L);
        for (uint256 k = 0; k < L; k++) {
            uint256 at_ = wordIn(data, k) / 32;
            uint256 n = wordIn(data, at_);
            require(n == Gx.wordsOf(OUT[k]), "an event vector has another length");
            c.v[k] = new uint256[](n);
            for (uint256 i = 0; i < n; i++) {
                c.v[k][i] = wordIn(data, at_ + 1 + i);
            }
        }
        uint256 zat = wordIn(data, L) / 32;
        require(wordIn(data, zat) == CLASSES, "the event's logits have another length");
        c.z = new int32[](CLASSES);
        for (uint256 i = 0; i < CLASSES; i++) {
            uint256 w = wordIn(data, zat + 1 + i);
            require(w == uint256(int256(int32(int256(w)))), "an event logit is not a sign-extended int32");
            c.z[i] = int32(int256(w));
        }
        uint256 cls = wordIn(data, L + 1);
        require(cls < 256, "the event's class is not a uint8");
        c.cls = uint8(cls);
        bond = wordIn(data, L + 2);
        deadline = wordIn(data, L + 3);
    }

    /// Random well-formed vectors: the event's vectors (every hidden vector in its place), class,
    /// bond, deadline and indexed fields equal the claim's, and `claimHashes` returns, in layer
    /// order, the hashes the spec prescribes (section 3, item 2).
    /// forge-config: default.fuzz.runs = 512
    function testFuzz_event_and_hashes_carry_the_claim(uint256[8] memory words, uint8 cls, uint256 extra) public {
        GClaim memory c;
        c.v = new uint256[][](L);
        for (uint256 k = 0; k < L; k++) {
            uint256 n = Gx.wordsOf(OUT[k]);
            c.v[k] = new uint256[](n);
            for (uint256 w = 0; w < n; w++) {
                c.v[k][w] = uint256(keccak256(abi.encode(words, k, w)));
            }
            GModel.sanitise(c.v[k], OUT[k]);
        }
        c.z = new int32[](CLASSES);
        for (uint256 k = 0; k < CLASSES; k++) {
            c.z[k] = int32(uint32(uint256(keccak256(abi.encode(words, "z", k)))));
        }
        c.cls = cls % uint8(CLASSES);
        extra %= 1e24;
        vm.recordLogs();
        uint256 id = post(c, CLAIMANT, BOND + extra);
        VmX.Log[] memory logs = vm.getRecordedLogs();
        require(logs.length == 1, "claim emitted another number of events");
        require(logs[0].topics.length == 3, "Claimed has two indexed fields");
        require(logs[0].topics[0] == claimedTopic && uint256(logs[0].topics[1]) == id, "topics");
        require(uint256(logs[0].topics[2]) == uint256(uint160(CLAIMANT)), "indexed claimant");
        (GClaim memory e, uint256 ebond, uint256 edeadline) = decodeClaimed(logs[0].data);
        for (uint256 k = 0; k < L; k++) {
            require(GModel.same(e.v[k], c.v[k]), "an event vector differs from the claim's");
        }
        for (uint256 k = 0; k < CLASSES; k++) {
            require(e.z[k] == c.z[k], "event logits");
        }
        require(e.cls == c.cls && ebond == BOND + extra && edeadline == T0 + WINDOW, "event scalars");
        (bool ok, uint256[] memory h) = hashesOf(id);
        require(ok, "claimHashes");
        uint256[] memory spec = specHashes(c);
        for (uint256 k = 0; k <= L; k++) {
            require(h[k] == spec[k], "a stored hash is not the spec's, or not in layer order");
        }
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

    /// Every way a claim ends, and a withdrawal, emits its event with the right fields. Layers: a
    /// slash by the handler of layer l says layer l and the neuron, for every layer.
    function testFuzz_every_outcome_emits_its_event(uint256 seed, uint256 j, uint256 salt, uint8 cls) public {
        GClaim memory honest = pick(seed);
        for (uint256 l = 1; l <= L; l++) {
            uint256 jl = uint256(keccak256(abi.encode(j, l))) % OUT[l];
            GClaim memory c = wrongAt(honest, l, jl, salt);
            uint256 id = post(c, CLAIMANT, BOND);
            vm.recordLogs();
            mustPass(CHALLENGER, neuronCall(id, l, jl, c), "the slash");
            VmX.Log memory log = oneLog();
            require(log.topics.length == 3 && log.topics[0] == IArbiter.NeuronSlashed.selector, "neuron topic");
            require(uint256(log.topics[1]) == id && address(uint160(uint256(log.topics[2]))) == CHALLENGER, "who");
            require(keccak256(log.data) == keccak256(abi.encode(l, jl)), "NeuronSlashed: layer and neuron");
        }

        GClaim memory w = wrongAt(honest, L + 1, 0, salt);
        uint256 cid = post(w, CLAIMANT, BOND);
        vm.recordLogs();
        mustPass(CHALLENGER, classCall(cid, w), "the class slash");
        VmX.Log memory clog = oneLog();
        require(clog.topics.length == 3 && clog.topics[0] == IArbiter.ClassSlashed.selector, "class topic");
        require(uint256(clog.topics[1]) == cid && address(uint160(uint256(clog.topics[2]))) == CHALLENGER, "class who");
        require(clog.data.length == 0, "class data");

        GClaim memory f = GModel.copy(honest);
        f.cls = cls % uint8(CLASSES);
        uint256 fid = post(f, CLAIMANT, BOND);
        vm.warp(deadlineOf(fid));
        vm.recordLogs();
        arbiter.finalize(fid);
        VmX.Log memory flog = oneLog();
        require(flog.topics.length == 2 && flog.topics[0] == IArbiter.Finalized.selector, "final topic");
        require(uint256(flog.topics[1]) == fid, "final id");
        require(keccak256(flog.data) == keccak256(abi.encode(uint256(f.cls))), "final class");

        vm.recordLogs();
        vm.prank(CHALLENGER);
        arbiter.withdraw();
        VmX.Log memory wlog = oneLog();
        require(wlog.topics.length == 2 && wlog.topics[0] == IArbiter.Withdrawn.selector, "withdraw topic");
        require(address(uint160(uint256(wlog.topics[1]))) == CHALLENGER, "withdraw account");
        require(keccak256(wlog.data) == keccak256(abi.encode((L + 1) * (BOND / 2))), "withdraw amount");
    }

    // ----------------------------------------------------------------------------------------
    // Unknown claims, re-entrancy, a payee that refuses ETH

    /// Any number at or past `claimCount()` is `UnknownClaim` in every handler (each layer's
    /// included) and every view.
    function testFuzz_unknown_claim_numbers_are_refused(uint256 seed, uint256 id) public {
        GClaim memory c = pick(seed);
        post(c, CLAIMANT, BOND);
        id = id == 0 ? 1 : id;
        bytes memory unknown = abi.encodeWithSelector(IArbiter.UnknownClaim.selector, id);
        vm.expectRevert(unknown);
        arbiter.finalize(id);
        for (uint256 kind = 1; kind <= L + 1; kind++) {
            mustFail(CHALLENGER, checkCall(id, kind, 0, c), unknown, "a challenge of an unknown claim");
        }
        vm.expectRevert(unknown);
        arbiter.result(id);
        vm.expectRevert(unknown);
        arbiter.claimInfo(id);
        (bool ok, bytes memory ret) = address(arbiter).staticcall(abi.encodeWithSignature("claimHashes(uint256)", id));
        require(!ok && keccak256(ret) == keccak256(unknown), "claimHashes of an unknown claim");
        require(statusOf(0) == 1 && address(arbiter).balance == BOND, "claim 0 changed");
    }

    /// A challenger contract whose `receive` calls `withdraw` again and posts a new claim with the
    /// ETH it just received: it is paid its half once, the inner withdraw finds nothing, the new
    /// claim is stored, and a bystander's open bond is untouched (see the shipped suite's test of
    /// the same name for why the bystander matters).
    function test_reentrant_withdraw_pays_once() public {
        GenReenterer r = new GenReenterer(address(arbiter));
        GClaim memory honest = pick(1);
        r.load(claimData(honest));
        GClaim memory c = wrongAt(honest, L + 1, 0, 0);
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 bystander = post(pick(3), address(0xB1), BOND);
        r.send(classCall(id, c));
        require(arbiter.creditOf(address(r)) == BOND / 2, "credit");
        vm.deal(address(r), BOND - BOND / 2);
        require(r.withdraw() == BOND / 2, "withdraw amount");
        require(r.secondWithdrawFoundNothing(), "the inner withdraw was not refused for lack of credit");
        require(r.claimedInside(), "the re-entrant claim did not run");
        require(arbiter.creditOf(address(r)) == 0, "credit after");
        require(statusOf(bystander) == 1, "the bystander's claim changed");
        require(arbiter.claimCount() == 3 && statusOf(2) == 1, "the re-entrant claim");
        require(
            address(arbiter).balance == 2 * BOND + arbiter.burned() && arbiter.burned() == BOND - BOND / 2,
            "balance: two open bonds and the burned half, no credit"
        );
    }

    /// A payee that refuses ETH: `withdraw` reverts with `TransferFailed` and the credit stays.
    function test_refused_transfer_keeps_the_credit() public {
        GenRefuser r = new GenRefuser();
        GClaim memory c = wrongAt(pick(2), L + 1, 0, 0);
        uint256 id = post(c, CLAIMANT, BOND);
        r.send(address(arbiter), classCall(id, c));
        vm.expectRevert(IArbiter.TransferFailed.selector);
        r.withdraw(arbiter);
        require(arbiter.creditOf(address(r)) == BOND / 2, "the credit was lost");
        require(address(arbiter).balance == BOND, "balance");
    }

    /// The arbiter has no fallback: ETH enters only through `claim` (what P1 relies on).
    function test_plain_eth_transfer_is_refused() public {
        vm.deal(address(this), 2 ether);
        (bool ok,) = address(arbiter).call{value: 1 ether}("");
        require(!ok, "a plain transfer was accepted");
        (ok,) = address(arbiter).call{value: 1 ether}(hex"deadbeef");
        require(!ok, "ETH with an unknown selector was accepted");
        (ok,) = address(arbiter).call{value: 1 ether}(abi.encodeCall(IArbiter.withdraw, ()));
        require(!ok, "ETH sent to withdraw was accepted");
        (ok,) = address(arbiter).call{value: 1 ether}(classCall(0, pick(0)));
        require(!ok, "ETH sent to a challenge was accepted");
        require(address(arbiter).balance == 0, "balance");
    }

    /// Forced ETH breaks only the equality of P1; it is never paid out.
    function test_P1_forced_eth_breaks_only_the_equality() public {
        GClaim memory c = wrongAt(pick(4), L + 1, 0, 0);
        uint256 id = post(c, CLAIMANT, BOND);
        new GenForcer{value: 5 wei}(address(arbiter));
        require(address(arbiter).balance == BOND + 5 wei, "the forced ETH did not arrive");
        mustPass(CHALLENGER, classCall(id, c), "the class slash");
        vm.prank(CHALLENGER);
        require(arbiter.withdraw() == BOND / 2, "withdraw paid another amount");
        require(address(arbiter).balance == BOND - BOND / 2 + 5 wei, "only the burned half and the forced ETH stay");
        require(arbiter.burned() == BOND - BOND / 2, "burned");
    }

    /// A claimant that slashes its own false claim (here at the middle layer when there is one)
    /// loses half the bond, and the claim is never final.
    function test_claimant_slashing_its_own_false_claim_loses_half_the_bond() public {
        uint256 l = L >= 3 ? 2 : 1;
        GClaim memory c = wrongAt(pick(0), l, 0, 7);
        uint256 id = post(c, CLAIMANT, BOND);
        uint256 before = CLAIMANT.balance;
        mustPass(CLAIMANT, neuronCall(id, l, 0, c), "the claimant's own slash");
        require(statusOf(id) == 2, "the claim is not slashed");
        require(arbiter.creditOf(CLAIMANT) == BOND / 2, "the claimant was credited more than half");
        require(arbiter.burned() == BOND - BOND / 2, "the other half was not burned");
        vm.prank(CLAIMANT);
        arbiter.withdraw();
        require(CLAIMANT.balance == before + BOND / 2, "the claimant got back more than half");
        require(address(arbiter).balance == BOND - BOND / 2, "the burned half left the arbiter");
        vm.expectRevert(abi.encodeWithSelector(IArbiter.NotFinal.selector, id));
        arbiter.result(id);
    }

    /// With a required bond of zero a claim needs no ETH and a slash pays nothing.
    function test_zero_bond_deployment_has_no_penalty() public {
        arbiter = deployArbiter(0, WINDOW);
        GClaim memory c = wrongAt(pick(0), L + 1, 0, 0);
        uint256 id = post(c, CLAIMANT, 0);
        mustPass(CHALLENGER, classCall(id, c), "the class slash");
        require(statusOf(id) == 2 && arbiter.creditOf(CHALLENGER) == 0, "slash without a bond");
        vm.prank(CHALLENGER);
        vm.expectRevert(IArbiter.NothingToWithdraw.selector);
        arbiter.withdraw();
    }

    /// `withdraw` with no credit is `NothingToWithdraw` and changes nothing.
    function testFuzz_withdraw_without_credit_is_refused(address who) public {
        vm.assume(who != address(0) && uint160(who) > 0xffff);
        post(pick(3), CLAIMANT, BOND);
        uint256 before = who.balance;
        vm.prank(who);
        vm.expectRevert(IArbiter.NothingToWithdraw.selector);
        arbiter.withdraw();
        require(who.balance == before && address(arbiter).balance == BOND, "balances changed");
    }
}
