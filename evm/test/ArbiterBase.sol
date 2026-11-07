// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {FeDeployer} from "./FeDeployer.sol";
import {SpecNeuron} from "./SpecNeuron.sol";

/// The Foundry cheatcodes the Arbiter tests use (Vm.sol holds the smaller set of the milestone 2
/// tests and is left as it is). Same cheatcode address.
interface VmX {
    struct Log {
        bytes32[] topics;
        bytes data;
        address emitter;
    }

    function warp(uint256 timestamp) external;
    function deal(address account, uint256 balance) external;
    function prank(address sender) external;
    function startPrank(address sender) external;
    function stopPrank() external;
    function expectRevert() external;
    function expectRevert(bytes4 selector) external;
    function expectRevert(bytes calldata data) external;
    function etch(address target, bytes calldata code) external;
    function readFileBinary(string calldata path) external view returns (bytes memory data);
    function assume(bool condition) external pure;
    function recordLogs() external;
    function getRecordedLogs() external returns (Log[] memory logs);
    function label(address account, string calldata name) external;
    function writeLine(string calldata path, string calldata data) external;
    function toString(uint256 value) external pure returns (string memory);
}

library CheatsX {
    VmX internal constant vm = VmX(address(uint160(uint256(keccak256("hevm cheat code")))));
}

/// The Fe `Arbiter` (ingots/onchain/src/arbiter.fe) as Solidity sees it: docs/fe-notes.md M3.5.
interface IArbiter {
    error BondTooSmall(uint256 sent, uint256 required);
    error MalformedInput();
    error MalformedHidden();
    error ClassOutOfRange(uint8 class);
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

    event Claimed(
        uint256 indexed id,
        address indexed claimant,
        uint256[] x0,
        uint256[] hidden,
        int32[] logits,
        uint8 class,
        uint256 bond,
        uint256 deadline
    );
    event NeuronSlashed(uint256 indexed id, address indexed challenger, uint256 layer, uint256 neuron);
    event ClassSlashed(uint256 indexed id, address indexed challenger);
    event Finalized(uint256 indexed id, uint8 class);
    event Withdrawn(address indexed account, uint256 amount);

    function claim(uint256[25] calldata x0, uint256[1] calldata hidden, int32[10] calldata logits, uint8 class)
        external
        payable
        returns (uint256 id);
    function challengeNeuron1(
        uint256 id,
        uint256 j,
        uint256[25] calldata x0,
        uint256[1] calldata hidden,
        uint256[25] calldata row,
        int32 bias,
        bytes32[] calldata proof
    ) external;
    function challengeNeuron2(
        uint256 id,
        uint256 j,
        uint256[1] calldata hidden,
        int32[10] calldata logits,
        uint256[1] calldata row,
        int32 bias,
        bytes32[] calldata proof
    ) external;
    function challengeClass(uint256 id, int32[10] calldata logits) external;
    function finalize(uint256 id) external;
    function withdraw() external returns (uint256);
    function result(uint256 id) external view returns (uint8);
    function claimCount() external view returns (uint256);
    function claimInfo(uint256 id)
        external
        view
        returns (address claimant, uint256 bond, uint256 deadline, uint8 class, uint8 status);
    function claimHashes(uint256 id) external view returns (uint256 x0Hash, uint256 hiddenHash, uint256 logitsHash);
    function creditOf(address account) external view returns (uint256);
    function modelRoot() external view returns (uint256);
    function bond() external view returns (uint256);
    function window() external view returns (uint256);
}

/// One claim's vectors (docs/spec-dispute.md section 3).
struct Claim {
    uint256[25] x0;
    uint256[1] hidden;
    int32[10] logits;
    uint8 cls;
}

/// The fixtures written by tools/gen_arbiter_fixtures.py (its docstring has the layout): the
/// model root, every neuron's row, bias and Merkle proof, and honest claims for reference samples.
/// `install` puts the file into the code of a fixed address, where any contract (the tests and the
/// invariant handler) reads it with EXTCODECOPY.
library Fx {
    VmX private constant vm = CheatsX.vm;
    address internal constant DATA = address(uint160(uint256(keccak256("tensile.arbiter.fixtures"))));
    uint256 private constant ROWS1 = 1;
    uint256 private constant BIAS1 = 801;
    uint256 private constant ROWS2 = 833;
    uint256 private constant BIAS2 = 843;
    uint256 private constant PROOFS = 853;
    uint256 private constant PROOF_WORDS = 7;
    uint256 private constant COUNT = 1147;
    uint256 private constant SAMPLES = 1148;
    uint256 private constant SAMPLE_WORDS = 38;

    function install() internal {
        bytes memory blob = vm.readFileBinary("test/fixtures/arbiter.bin");
        require(blob.length == 32 * (SAMPLES + SAMPLE_WORDS * wordAt(blob, COUNT)), "arbiter.bin has the wrong size");
        // A leading STOP byte, so the data is never run as code.
        vm.etch(DATA, bytes.concat(hex"00", blob));
    }

    function wordAt(bytes memory blob, uint256 i) private pure returns (uint256 w) {
        assembly ("memory-safe") {
            w := mload(add(add(blob, 32), mul(i, 32)))
        }
    }

    function word(uint256 i) internal view returns (uint256 w) {
        address data = DATA;
        assembly ("memory-safe") {
            let p := mload(0x40)
            extcodecopy(data, p, add(1, mul(i, 32)), 32)
            w := mload(p)
        }
    }

    function words(uint256 start, uint256 n) private view returns (uint256[] memory out) {
        out = new uint256[](n);
        address data = DATA;
        assembly ("memory-safe") {
            extcodecopy(data, add(out, 32), add(1, mul(start, 32)), mul(n, 32))
        }
    }

    function root() internal view returns (uint256) {
        return word(0);
    }

    function row1(uint256 j) internal view returns (uint256[25] memory row) {
        uint256[] memory w = words(ROWS1 + 25 * j, 25);
        for (uint256 k = 0; k < 25; k++) {
            row[k] = w[k];
        }
    }

    function bias1(uint256 j) internal view returns (int32) {
        return int32(int256(word(BIAS1 + j)));
    }

    function row2(uint256 j) internal view returns (uint256[1] memory) {
        return [word(ROWS2 + j)];
    }

    function bias2(uint256 j) internal view returns (int32) {
        return int32(int256(word(BIAS2 + j)));
    }

    /// The Merkle proof of neuron `j` of layer `layer` (1 or 2).
    function proof(uint256 layer, uint256 j) internal view returns (bytes32[] memory p) {
        uint256 base = PROOFS + PROOF_WORDS * (layer == 1 ? j : 32 + j);
        uint256 n = word(base);
        p = new bytes32[](n);
        for (uint256 k = 0; k < n; k++) {
            p[k] = bytes32(word(base + 1 + k));
        }
    }

    function sampleCount() internal view returns (uint256) {
        return word(COUNT);
    }

    /// The index in ref/out of fixture sample `n`.
    function sampleIndex(uint256 n) internal view returns (uint256) {
        return word(SAMPLES + SAMPLE_WORDS * n);
    }

    /// The honest claim of fixture sample `n`.
    function sample(uint256 n) internal view returns (Claim memory c) {
        uint256[] memory w = words(SAMPLES + SAMPLE_WORDS * n + 1, SAMPLE_WORDS - 1);
        for (uint256 k = 0; k < 25; k++) {
            c.x0[k] = w[k];
        }
        c.hidden[0] = w[25];
        for (uint256 k = 0; k < 10; k++) {
            c.logits[k] = int32(int256(w[26 + k]));
        }
        c.cls = uint8(w[36]);
    }
}

/// The network of docs/spec.md sections 3 and 7, computed in Solidity from the fixture rows with
/// SpecNeuron (the plain byte loop): the tests' own oracle, independent of the Fe kernel.
library Model {
    uint256 internal constant SHIFT_1 = 11;
    /// 0x80 in every byte.
    uint256 internal constant SIGN_BITS = 0x8080808080808080808080808080808080808080808080808080808080808080;
    /// The low 16 bytes of word 24 of an input: padding (784 = 24 * 32 + 16).
    uint256 internal constant PADDING_1 = (1 << 128) - 1;

    /// Neuron `j` of layer 1 on input `x0`.
    function neuron1(uint256 j, uint256[25] memory x0) internal view returns (int8) {
        int256 acc = SpecNeuron.accumulate(abi.encodePacked(Fx.row1(j)), abi.encodePacked(x0), 784, Fx.bias1(j));
        return SpecNeuron.requantise(acc, SHIFT_1);
    }

    /// Neuron `j` of layer 2 (a logit) on the hidden vector `hidden`.
    function neuron2(uint256 j, uint256[1] memory hidden) internal view returns (int32) {
        int256 acc = SpecNeuron.accumulate(abi.encodePacked(Fx.row2(j)), abi.encodePacked(hidden), 32, Fx.bias2(j));
        require(acc == int32(acc), "a layer 2 accumulator left int32");
        return int32(acc);
    }

    /// The lowest index of the largest logit.
    function argmax(int32[10] memory z) internal pure returns (uint8 best) {
        for (uint8 k = 1; k < 10; k++) {
            if (z[k] > z[best]) best = k;
        }
    }

    /// The honest claim for input `x0`.
    function honest(uint256[25] memory x0) internal view returns (Claim memory c) {
        c.x0 = x0;
        for (uint256 j = 0; j < 32; j++) {
            c.hidden = setByte(c.hidden, j, uint8(neuron1(j, x0)));
        }
        for (uint256 j = 0; j < 10; j++) {
            c.logits[j] = neuron2(j, c.hidden);
        }
        c.cls = argmax(c.logits);
    }

    /// Makes `x` a well-formed input: every byte 0..127, zero padding.
    function sanitise(uint256[25] memory x) internal pure returns (uint256[25] memory) {
        for (uint256 k = 0; k < 25; k++) {
            x[k] &= ~SIGN_BITS;
        }
        x[24] &= ~PADDING_1;
        return x;
    }

    function byteOf(uint256 w, uint256 i) internal pure returns (uint8) {
        return uint8(w >> (8 * (31 - i)));
    }

    function hiddenAt(uint256[1] memory h, uint256 j) internal pure returns (uint8) {
        return byteOf(h[0], j);
    }

    function inputAt(uint256[25] memory x, uint256 i) internal pure returns (uint8) {
        return byteOf(x[i / 32], i % 32);
    }

    function setByte(uint256[1] memory h, uint256 j, uint8 v) internal pure returns (uint256[1] memory) {
        uint256 shift = 8 * (31 - j);
        h[0] = (h[0] & ~(uint256(0xff) << shift)) | (uint256(v) << shift);
        return h;
    }

    function setInput(uint256[25] memory x, uint256 i, uint8 v) internal pure returns (uint256[25] memory) {
        uint256 shift = 8 * (31 - i % 32);
        x[i / 32] = (x[i / 32] & ~(uint256(0xff) << shift)) | (uint256(v) << shift);
        return x;
    }

    /// A deep copy (assigning a memory array in Solidity copies only the reference).
    function copy(Claim memory c) internal pure returns (Claim memory d) {
        for (uint256 k = 0; k < 25; k++) {
            d.x0[k] = c.x0[k];
        }
        d.hidden[0] = c.hidden[0];
        for (uint256 k = 0; k < 10; k++) {
            d.logits[k] = c.logits[k];
        }
        d.cls = c.cls;
    }
}

/// Deployment, posting and the correct challenge data, shared by the fuzz tests and the handler.
abstract contract ArbiterUser {
    VmX internal constant vm = CheatsX.vm;

    function deployArbiter(uint256 bondRequired, uint256 windowSeconds) internal returns (IArbiter) {
        return IArbiter(FeDeployer.deploy("Arbiter", abi.encode(Fx.root(), bondRequired, windowSeconds)));
    }

    function post(IArbiter a, Claim memory c, address who, uint256 value) internal returns (uint256) {
        vm.deal(who, who.balance + value);
        vm.prank(who);
        return a.claim{value: value}(c.x0, c.hidden, c.logits, c.cls);
    }

    function challenge1(IArbiter a, uint256 id, uint256 j, Claim memory c, address who) internal {
        vm.prank(who);
        a.challengeNeuron1(id, j, c.x0, c.hidden, Fx.row1(j), Fx.bias1(j), Fx.proof(1, j));
    }

    function challenge2(IArbiter a, uint256 id, uint256 j, Claim memory c, address who) internal {
        vm.prank(who);
        a.challengeNeuron2(id, j, c.hidden, c.logits, Fx.row2(j), Fx.bias2(j), Fx.proof(2, j));
    }

    function challengeC(IArbiter a, uint256 id, Claim memory c, address who) internal {
        vm.prank(who);
        a.challengeClass(id, c.logits);
    }

    function statusOf(IArbiter a, uint256 id) internal view returns (uint8 status) {
        (,,,, status) = a.claimInfo(id);
    }

    function deadlineOf(IArbiter a, uint256 id) internal view returns (uint256 deadline) {
        (,, deadline,,) = a.claimInfo(id);
    }
}
