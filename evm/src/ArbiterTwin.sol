// SPDX-License-Identifier: Apache-2.0
pragma solidity 0.8.38;

/// The Solidity twin of the Fe `Arbiter` (ingots/onchain/src/arbiter.fe): the same selectors, events,
/// errors, checks and order of checks, written the way a Solidity engineer would write it, and the
/// gas baseline for the whole contract (docs/numbers.md). The Foundry suite that tests the Fe contract
/// also runs against this one (tools/m4_twin_suite.sh), which is the evidence the two behave alike.
///
/// Its storage record is packed: the address, deadline, class and status share a slot, where Fe gives
/// an `Address` a whole word. It uses `uint64` for the deadline, which a window near 2^64 seconds would
/// overflow; the Fe contract keeps a whole word for it, and its checked add has the same effect.
contract ArbiterTwin {
    uint256 private constant SIGN_BITS = 0x8080808080808080808080808080808080808080808080808080808080808080;
    uint256 private constant SHIFT_1 = 11;
    uint8 private constant OPEN = 1;
    uint8 private constant SLASHED = 2;
    uint8 private constant FINAL = 3;

    struct Record {
        address claimant;
        uint64 deadline;
        uint8 class_;
        uint8 status;
        uint256 bond;
        uint256 x0Hash;
        uint256 hiddenHash;
        uint256 logitsHash;
    }

    uint256 public immutable modelRoot;
    uint256 public immutable bond;
    uint256 public immutable window;
    uint256 public claimCount;
    mapping(uint256 => Record) private claims;
    mapping(address => uint256) public creditOf;
    /// Half of every slashed bond, rounded up: held for good, owed to nobody.
    uint256 public burned;

    event Claimed(
        uint256 indexed id,
        address indexed claimant,
        uint256[] x0,
        uint256[] hidden,
        int32[] logits,
        uint8 class_,
        uint256 bond,
        uint256 deadline
    );
    event NeuronSlashed(uint256 indexed id, address indexed challenger, uint256 layer, uint256 neuron);
    event ClassSlashed(uint256 indexed id, address indexed challenger);
    event Finalized(uint256 indexed id, uint8 class_);
    event Withdrawn(address indexed account, uint256 amount);

    error BondTooSmall(uint256 sent, uint256 required);
    error MalformedInput();
    error MalformedHidden();
    error ClassOutOfRange(uint8 class_);
    error UnknownClaim(uint256 id);
    error ClaimNotOpen(uint256 id);
    error TooLate(uint256 deadline);
    error TooEarly(uint256 deadline);
    error NoFraud();
    error NeuronOutOfRange(uint256 neuron);
    error InputMismatch();
    error OutputMismatch();
    error NotCommitted();
    error NotFinal(uint256 id);
    error NothingToWithdraw();
    error TransferFailed();

    constructor(uint256 root_, uint256 bond_, uint256 window_) {
        modelRoot = root_;
        bond = bond_;
        window = window_;
    }

    function claim(uint256[25] calldata x0, uint256[1] calldata hidden, int32[10] calldata logits, uint8 class_)
        external
        payable
        returns (uint256 id)
    {
        if (msg.value < bond) revert BondTooSmall(msg.value, bond);
        if (x0[24] << 128 != 0) revert MalformedInput();
        for (uint256 k = 0; k < 25; k++) {
            if (x0[k] & SIGN_BITS != 0) revert MalformedInput();
        }
        if (hidden[0] & SIGN_BITS != 0) revert MalformedHidden();
        if (class_ >= 10) revert ClassOutOfRange(class_);
        id = claimCount++;
        uint256 deadline = block.timestamp + window;
        claims[id] = Record({
            claimant: msg.sender,
            deadline: uint64(deadline),
            class_: class_,
            status: OPEN,
            bond: msg.value,
            x0Hash: uint256(keccak256(abi.encode(x0))),
            hiddenHash: uint256(keccak256(abi.encode(hidden))),
            logitsHash: uint256(keccak256(abi.encode(logits)))
        });
        uint256[] memory x0Dynamic = new uint256[](25);
        for (uint256 k = 0; k < 25; k++) x0Dynamic[k] = x0[k];
        uint256[] memory hiddenDynamic = new uint256[](1);
        hiddenDynamic[0] = hidden[0];
        int32[] memory logitsDynamic = new int32[](10);
        for (uint256 k = 0; k < 10; k++) logitsDynamic[k] = logits[k];
        emit Claimed(id, msg.sender, x0Dynamic, hiddenDynamic, logitsDynamic, class_, msg.value, deadline);
    }

    function challengeNeuron1(
        uint256 id,
        uint256 j,
        uint256[25] calldata x0,
        uint256[1] calldata hidden,
        uint256[25] calldata row,
        int32 bias,
        bytes32[] calldata proof
    ) external {
        Record memory record = load(id);
        if (j >= 32) revert NeuronOutOfRange(j);
        if (uint256(keccak256(abi.encode(x0))) != record.x0Hash) revert InputMismatch();
        if (uint256(keccak256(abi.encode(hidden))) != record.hiddenHash) revert OutputMismatch();
        bytes32 leaf = keccak256(abi.encodePacked("tensile.neuron.v0", uint256(1), j, int256(bias), row));
        if (!verify(proof, leaf)) revert NotCommitted();
        if (row[24] << 128 != 0) revert();
        if (x0[24] << 128 != 0) revert();
        int256 acc = bias;
        for (uint256 k = 0; k < 25; k++) {
            if (x0[k] & SIGN_BITS != 0) revert();
            unchecked {
                acc += dotWord(row[k], x0[k]);
            }
        }
        int256 shifted = acc >> SHIFT_1;
        uint8 recomputed = shifted < 0 ? 0 : shifted > 127 ? 127 : uint8(uint256(shifted));
        uint8 claimed = uint8(hidden[0] >> (248 - j * 8));
        settleChallenge(id, record, recomputed != claimed);
        emit NeuronSlashed(id, msg.sender, 1, j);
    }

    function challengeNeuron2(
        uint256 id,
        uint256 j,
        uint256[1] calldata hidden,
        int32[10] calldata logits,
        uint256[1] calldata row,
        int32 bias,
        bytes32[] calldata proof
    ) external {
        Record memory record = load(id);
        if (j >= 10) revert NeuronOutOfRange(j);
        if (uint256(keccak256(abi.encode(hidden))) != record.hiddenHash) revert InputMismatch();
        if (uint256(keccak256(abi.encode(logits))) != record.logitsHash) revert OutputMismatch();
        bytes32 leaf = keccak256(abi.encodePacked("tensile.neuron.v0", uint256(2), j, int256(bias), row));
        if (!verify(proof, leaf)) revert NotCommitted();
        if (hidden[0] & SIGN_BITS != 0) revert();
        int256 recomputed;
        unchecked {
            recomputed = bias + dotWord(row[0], hidden[0]);
        }
        settleChallenge(id, record, recomputed != logits[j]);
        emit NeuronSlashed(id, msg.sender, 2, j);
    }

    function challengeClass(uint256 id, int32[10] calldata logits) external {
        Record memory record = load(id);
        if (uint256(keccak256(abi.encode(logits))) != record.logitsHash) revert OutputMismatch();
        uint256 best = 0;
        for (uint256 j = 1; j < 10; j++) {
            if (logits[j] > logits[best]) best = j;
        }
        settleChallenge(id, record, best != record.class_);
        emit ClassSlashed(id, msg.sender);
    }

    function finalize(uint256 id) external {
        Record memory record = load(id);
        if (record.status != OPEN) revert ClaimNotOpen(id);
        if (block.timestamp < record.deadline) revert TooEarly(record.deadline);
        claims[id].status = FINAL;
        creditOf[record.claimant] += record.bond;
        emit Finalized(id, record.class_);
    }

    function withdraw() external returns (uint256 amount) {
        amount = creditOf[msg.sender];
        if (amount == 0) revert NothingToWithdraw();
        creditOf[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amount}("");
        if (!ok) revert TransferFailed();
        emit Withdrawn(msg.sender, amount);
    }

    function result(uint256 id) external view returns (uint8) {
        Record memory record = load(id);
        if (record.status != FINAL) revert NotFinal(id);
        return record.class_;
    }

    function claimInfo(uint256 id) external view returns (address, uint256, uint256, uint8, uint8) {
        Record memory record = load(id);
        return (record.claimant, record.bond, record.deadline, record.class_, record.status);
    }

    function claimHashes(uint256 id) external view returns (uint256, uint256, uint256) {
        Record memory record = load(id);
        return (record.x0Hash, record.hiddenHash, record.logitsHash);
    }

    function load(uint256 id) private view returns (Record memory) {
        if (id >= claimCount) revert UnknownClaim(id);
        return claims[id];
    }

    function settleChallenge(uint256 id, Record memory record, bool fraud) private {
        if (record.status != OPEN) revert ClaimNotOpen(id);
        if (block.timestamp >= record.deadline) revert TooLate(record.deadline);
        if (!fraud) revert NoFraud();
        claims[id].status = SLASHED;
        uint256 paid = record.bond / 2;
        creditOf[msg.sender] += paid;
        if (record.bond - paid != 0) burned += record.bond - paid;
    }

    function verify(bytes32[] calldata proof, bytes32 leaf) private view returns (bool) {
        bytes32 node = leaf;
        for (uint256 k = 0; k < proof.length; k++) {
            bytes32 sibling = proof[k];
            node = node < sibling
                ? keccak256(abi.encodePacked(node, sibling))
                : keccak256(abi.encodePacked(sibling, node));
        }
        return uint256(node) == modelRoot;
    }

    /// The dot product of the 32 signed weight bytes of `w` and the 32 activation bytes of `a`,
    /// written out lane by lane (as `NeuronTwin.neuron1Unrolled`, the cheaper of the two shapes).
    function dotWord(uint256 w, uint256 a) private pure returns (int256 s) {
        assembly ("memory-safe") {
            function lane(w_, a_, i) -> p {
                p := mul(signextend(0, byte(i, w_)), byte(i, a_))
            }
            s := add(add(add(lane(w, a, 0), lane(w, a, 1)), add(lane(w, a, 2), lane(w, a, 3))),
                     add(add(lane(w, a, 4), lane(w, a, 5)), add(lane(w, a, 6), lane(w, a, 7))))
            s := add(s, add(add(add(lane(w, a, 8), lane(w, a, 9)), add(lane(w, a, 10), lane(w, a, 11))),
                            add(add(lane(w, a, 12), lane(w, a, 13)), add(lane(w, a, 14), lane(w, a, 15)))))
            s := add(s, add(add(add(lane(w, a, 16), lane(w, a, 17)), add(lane(w, a, 18), lane(w, a, 19))),
                            add(add(lane(w, a, 20), lane(w, a, 21)), add(lane(w, a, 22), lane(w, a, 23)))))
            s := add(s, add(add(add(lane(w, a, 24), lane(w, a, 25)), add(lane(w, a, 26), lane(w, a, 27))),
                            add(add(lane(w, a, 28), lane(w, a, 29)), add(lane(w, a, 30), lane(w, a, 31)))))
        }
    }
}
