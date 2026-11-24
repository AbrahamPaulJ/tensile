// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {CheatsX, IArbiter, VmX} from "./ArbiterBase.sol";
import {FeDeployer} from "./FeDeployer.sol";
import {SpecNeuron} from "./SpecNeuron.sol";

// The Foundry suite of a GENERATED arbiter (tools/gen_arbiter.py) for a dense model with any
// number of layers. It is the suite of evm/test/Arbiter*.t.sol (the shipped two-layer arbiter)
// with the sizes taken out: nothing here knows a layer count or a vector length. They are read
// from the fixture file tools/gen_arbiter_fixtures_any.py writes from the model file, and the
// calls are encoded by hand from them, because a Solidity interface fixes its array lengths.
//
// tools/gen_foundry.sh stages these files next to evm/test/ArbiterBase.sol, FeDeployer.sol and
// SpecNeuron.sol (imported, not copied), the example's `Arbiter.bin` and its fixture, and runs
// forge once per example. From ArbiterBase.sol only the cheatcodes and the parts of `IArbiter`
// that do not depend on the model are used: the errors, the events without vectors, and the calls
// without vectors.

/// One claim (docs/spec-dispute.md section 3) of a network with L layers: `v[0]` is the input as
/// packed words, `v[k]` the output of hidden layer k (k = 1 .. L - 1) as packed words, `z` the
/// logits, `cls` the class.
struct GClaim {
    uint256[][] v;
    int32[] z;
    uint8 cls;
}

/// The fixture file (tools/gen_arbiter_fixtures_any.py; its docstring has the layout): the shape
/// of the model, its root, every neuron's row, bias and Merkle proof, and honest claims for
/// reference samples. `install` puts the file into the code of a fixed address, where any
/// contract (the tests and the invariant handler) reads it with EXTCODECOPY. Layers are numbered
/// from 1, as in the spec.
library Gx {
    VmX private constant vm = CheatsX.vm;
    address internal constant DATA = address(uint160(uint256(keccak256("tensile.generated.fixtures"))));
    bytes32 private constant TAG = "tensile.generated.fixtures.v1";
    uint256 private constant HEADER = 7;
    uint256 private constant PER_LAYER = 6;

    function install() internal {
        bytes memory blob = vm.readFileBinary("test/fixtures/generated.bin");
        require(blob.length >= 32 * HEADER && blob.length % 32 == 0, "generated.bin is not a fixture file");
        // A leading STOP byte, so the data is never run as code.
        vm.etch(DATA, bytes.concat(hex"00", blob));
        require(bytes32(word(0)) == TAG, "generated.bin has another tag");
        require(blob.length == 32 * (word(5) + word(6) * word(4)), "generated.bin has the wrong size");
    }

    function word(uint256 i) internal view returns (uint256 w) {
        address data = DATA;
        assembly ("memory-safe") {
            let p := mload(0x40)
            extcodecopy(data, p, add(1, mul(i, 32)), 32)
            w := mload(p)
        }
    }

    function words(uint256 start, uint256 n) internal view returns (uint256[] memory out) {
        out = new uint256[](n);
        address data = DATA;
        assembly ("memory-safe") {
            extcodecopy(data, add(out, 32), add(1, mul(start, 32)), mul(n, 32))
        }
    }

    function root() internal view returns (uint256) {
        return word(1);
    }

    function layers() internal view returns (uint256) {
        return word(2);
    }

    function sampleCount() internal view returns (uint256) {
        return word(4);
    }

    function field(uint256 layer, uint256 k) private view returns (uint256) {
        return word(HEADER + PER_LAYER * (layer - 1) + k);
    }

    /// Number of inputs of layer `layer`.
    function nin(uint256 layer) internal view returns (uint256) {
        return field(layer, 0);
    }

    /// Number of neurons of layer `layer`.
    function nout(uint256 layer) internal view returns (uint256) {
        return field(layer, 1);
    }

    /// The requantisation shift of hidden layer `layer`.
    function shift(uint256 layer) internal view returns (uint256) {
        return field(layer, 2);
    }

    /// Packed words of a vector of `n` values (docs/spec.md section 7).
    function wordsOf(uint256 n) internal pure returns (uint256) {
        return (n + 31) / 32;
    }

    function row(uint256 layer, uint256 j) internal view returns (uint256[] memory) {
        uint256 w = wordsOf(nin(layer));
        return words(field(layer, 3) + w * j, w);
    }

    function bias(uint256 layer, uint256 j) internal view returns (int32) {
        return int32(int256(word(field(layer, 4) + j)));
    }

    /// The Merkle proof of neuron `j` of layer `layer`.
    function proof(uint256 layer, uint256 j) internal view returns (bytes32[] memory p) {
        uint256 base = field(layer, 5) + (1 + word(3)) * j;
        uint256 n = word(base);
        p = new bytes32[](n);
        for (uint256 k = 0; k < n; k++) {
            p[k] = bytes32(word(base + 1 + k));
        }
    }

    /// The index in the reference vectors of fixture sample `n`.
    function sampleIndex(uint256 n) internal view returns (uint256) {
        return word(word(5) + word(6) * n);
    }

    /// The honest claim of fixture sample `n`, as the reference trace has it.
    function sample(uint256 n) internal view returns (GClaim memory c) {
        uint256 count = layers();
        uint256 p = word(5) + word(6) * n + 1;
        c.v = new uint256[][](count);
        for (uint256 k = 0; k < count; k++) {
            uint256 w = wordsOf(k == 0 ? nin(1) : nout(k));
            c.v[k] = words(p, w);
            p += w;
        }
        uint256 classes = nout(count);
        uint256[] memory z = words(p, classes);
        c.z = new int32[](classes);
        for (uint256 k = 0; k < classes; k++) {
            c.z[k] = int32(int256(z[k]));
        }
        c.cls = uint8(word(p + classes));
    }
}

/// The network of docs/spec.md sections 3 and 7, computed in Solidity from the fixture rows with
/// SpecNeuron (the plain byte loop): the tests' own oracle. It shares no code with the Fe kernel,
/// the model ingot or the generator.
library GModel {
    /// 0x80 in every byte.
    uint256 internal constant SIGN_BITS = 0x8080808080808080808080808080808080808080808080808080808080808080;

    /// The accumulator of neuron `j` of layer `layer` on the packed vector `input`.
    function accumulator(uint256 layer, uint256 j, uint256[] memory input) internal view returns (int256) {
        return SpecNeuron.accumulate(
            abi.encodePacked(Gx.row(layer, j)), abi.encodePacked(input), Gx.nin(layer), Gx.bias(layer, j)
        );
    }

    /// Neuron `j` of hidden layer `layer` on `input`: an activation, 0..127.
    function hiddenNeuron(uint256 layer, uint256 j, uint256[] memory input) internal view returns (uint8) {
        return uint8(SpecNeuron.requantise(accumulator(layer, j, input), Gx.shift(layer)));
    }

    /// Neuron `j` of the last layer (a logit) on `input`.
    function logit(uint256 j, uint256[] memory input) internal view returns (int32) {
        int256 acc = accumulator(Gx.layers(), j, input);
        require(acc == int32(acc), "a last-layer accumulator left int32");
        return int32(acc);
    }

    /// The whole output of hidden layer `layer` on `input`, as packed words.
    function hiddenLayer(uint256 layer, uint256[] memory input) internal view returns (uint256[] memory out) {
        uint256 n = Gx.nout(layer);
        out = new uint256[](Gx.wordsOf(n));
        for (uint256 j = 0; j < n; j++) {
            setByte(out, j, hiddenNeuron(layer, j, input));
        }
    }

    function logits(uint256[] memory input) internal view returns (int32[] memory z) {
        uint256 n = Gx.nout(Gx.layers());
        z = new int32[](n);
        for (uint256 j = 0; j < n; j++) {
            z[j] = logit(j, input);
        }
    }

    /// The lowest index of the largest logit.
    function argmax(int32[] memory z) internal pure returns (uint8 best) {
        for (uint256 k = 1; k < z.length; k++) {
            if (z[k] > z[best]) best = uint8(k);
        }
    }

    /// Recomputes everything after vector `from` (0 = the input) from `c.v[from]` as it stands:
    /// the later hidden vectors, the logits and the class. The claim is then consistent from
    /// there on, whatever `c.v[from]` holds.
    function carry(GClaim memory c, uint256 from) internal view {
        uint256 count = Gx.layers();
        for (uint256 l = from + 1; l < count; l++) {
            c.v[l] = hiddenLayer(l, c.v[l - 1]);
        }
        c.z = logits(c.v[count - 1]);
        c.cls = argmax(c.z);
    }

    /// The honest claim for input `x0`.
    function honest(uint256[] memory x0) internal view returns (GClaim memory c) {
        c.v = new uint256[][](Gx.layers());
        c.v[0] = x0;
        carry(c, 0);
    }

    /// Makes `x` a well-formed packed vector of `n` values: every byte 0..127, zero padding.
    function sanitise(uint256[] memory x, uint256 n) internal pure returns (uint256[] memory) {
        for (uint256 k = 0; k < x.length; k++) {
            x[k] &= ~SIGN_BITS;
        }
        if (n % 32 != 0) x[x.length - 1] &= ~paddingMask(n);
        return x;
    }

    /// The padding bytes of the last word of a packed vector of `n` values (zero if there are none).
    function paddingMask(uint256 n) internal pure returns (uint256) {
        return n % 32 == 0 ? 0 : (uint256(1) << (8 * (32 - n % 32))) - 1;
    }

    function byteAt(uint256[] memory v, uint256 i) internal pure returns (uint8) {
        return uint8(v[i / 32] >> (8 * (31 - i % 32)));
    }

    function setByte(uint256[] memory v, uint256 i, uint8 b) internal pure {
        uint256 s = 8 * (31 - i % 32);
        v[i / 32] = (v[i / 32] & ~(uint256(0xff) << s)) | (uint256(b) << s);
    }

    function copyWords(uint256[] memory a) internal pure returns (uint256[] memory b) {
        b = new uint256[](a.length);
        for (uint256 k = 0; k < a.length; k++) {
            b[k] = a[k];
        }
    }

    /// `a` cut or zero-filled to `n` words.
    function resized(uint256[] memory a, uint256 n) internal pure returns (uint256[] memory b) {
        b = new uint256[](n);
        for (uint256 k = 0; k < n && k < a.length; k++) {
            b[k] = a[k];
        }
    }

    function same(uint256[] memory a, uint256[] memory b) internal pure returns (bool) {
        return keccak256(abi.encodePacked(a)) == keccak256(abi.encodePacked(b));
    }

    /// A deep copy (assigning a memory array in Solidity copies only the reference).
    function copy(GClaim memory c) internal pure returns (GClaim memory d) {
        d.v = new uint256[][](c.v.length);
        for (uint256 k = 0; k < c.v.length; k++) {
            d.v[k] = copyWords(c.v[k]);
        }
        d.z = new int32[](c.z.length);
        for (uint256 k = 0; k < c.z.length; k++) {
            d.z[k] = c.z[k];
        }
        d.cls = c.cls;
    }
}

/// The shape, the hand-made calldata of every call that carries a vector, deployment, posting and
/// the correct challenge data: shared by the tests and the invariant handler.
abstract contract GenUser {
    VmX internal constant vm = CheatsX.vm;

    IArbiter internal arbiter;
    /// Number of layers; the class challenge is check kind `L + 1`.
    uint256 internal L;
    /// `OUT[l]` is the number of neurons of layer l (1-based); `OUT[0]` the number of inputs.
    uint256[] internal OUT;
    uint256 internal CLASSES;
    bytes4 internal claimSel;
    bytes4 internal classSel;
    /// `neuronSel[l]` is the selector of `challengeNeuron<l>`.
    bytes4[] internal neuronSel;
    bytes32 internal claimedTopic;

    /// Reads the shape from the installed fixture and derives every signature from it, as
    /// docs/spec-dispute.md sections 3 and 4 give them: vectors as fixed arrays of packed words,
    /// logits as `int32[classes]`.
    function initShape() internal {
        L = Gx.layers();
        OUT.push(Gx.nin(1));
        string memory vectors = "";
        string memory dynamic = "";
        for (uint256 l = 1; l <= L; l++) {
            require(Gx.nin(l) == OUT[l - 1], "the fixture's layers do not chain");
            OUT.push(Gx.nout(l));
            vectors = string.concat(vectors, wordsType(OUT[l - 1]), ",");
            dynamic = string.concat(dynamic, "uint256[],");
        }
        CLASSES = OUT[L];
        string memory logitsType = string.concat("int32[", vm.toString(CLASSES), "]");
        claimSel = bytes4(keccak256(bytes(string.concat("claim(", vectors, logitsType, ",uint8)"))));
        classSel = bytes4(keccak256(bytes(string.concat("challengeClass(uint256,", logitsType, ")"))));
        claimedTopic =
            keccak256(bytes(string.concat("Claimed(uint256,address,", dynamic, "int32[],uint8,uint256,uint256)")));
        neuronSel.push(bytes4(0));
        for (uint256 l = 1; l <= L; l++) {
            string memory input = wordsType(OUT[l - 1]);
            neuronSel.push(
                bytes4(
                    keccak256(
                        bytes(
                            string.concat(
                                "challengeNeuron",
                                vm.toString(l),
                                "(uint256,uint256,",
                                input,
                                ",",
                                l < L ? wordsType(OUT[l]) : logitsType,
                                ",",
                                input,
                                ",int32,bytes32[])"
                            )
                        )
                    )
                )
            );
        }
    }

    function wordsType(uint256 n) private pure returns (string memory) {
        return string.concat("uint256[", vm.toString(Gx.wordsOf(n)), "]");
    }

    function deployArbiter(uint256 bondRequired, uint256 windowSeconds) internal returns (IArbiter) {
        return deployArbiterWithRoot(Gx.root(), bondRequired, windowSeconds);
    }

    function deployArbiterWithRoot(uint256 root, uint256 bondRequired, uint256 windowSeconds)
        internal
        returns (IArbiter)
    {
        return IArbiter(FeDeployer.deploy("Arbiter", abi.encode(root, bondRequired, windowSeconds)));
    }

    // ----------------------------------------------------------------------------------------
    // Calldata

    function logitWords(int32[] memory z) internal pure returns (bytes memory) {
        // Array elements are padded to 32 bytes (sign-extended) in packed mode.
        return abi.encodePacked(z);
    }

    function claimData(GClaim memory c) internal view returns (bytes memory data) {
        data = abi.encodePacked(claimSel);
        for (uint256 k = 0; k < c.v.length; k++) {
            data = bytes.concat(data, abi.encodePacked(c.v[k]));
        }
        data = bytes.concat(data, logitWords(c.z), abi.encode(uint256(c.cls)));
    }

    /// `challengeNeuron<l>` from its parts: the two vectors as words, the row, the bias as a
    /// whole ABI word (so a test can send one that is not sign-extended) and the proof.
    function neuronDataRaw(
        uint256 l,
        uint256 id,
        uint256 j,
        bytes memory input,
        bytes memory output,
        uint256[] memory row,
        uint256 biasWord,
        bytes32[] memory proof
    ) internal view returns (bytes memory) {
        bytes memory head = abi.encodePacked(neuronSel[l], id, j, input, output, row, biasWord);
        // The offset of the proof, from the start of the arguments: past this head and its own word.
        return abi.encodePacked(head, head.length - 4 + 32, proof.length, proof);
    }

    function neuronData(
        uint256 l,
        uint256 id,
        uint256 j,
        bytes memory input,
        bytes memory output,
        uint256[] memory row,
        int32 bias,
        bytes32[] memory proof
    ) internal view returns (bytes memory) {
        return neuronDataRaw(l, id, j, input, output, row, uint256(int256(bias)), proof);
    }

    /// The vector layer `l` reads, as the claim commits it.
    function inputOf(GClaim memory c, uint256 l) internal pure returns (bytes memory) {
        return abi.encodePacked(c.v[l - 1]);
    }

    /// The vector layer `l` writes, as the claim commits it.
    function outputOf(GClaim memory c, uint256 l) internal pure returns (bytes memory) {
        return l < c.v.length ? abi.encodePacked(c.v[l]) : logitWords(c.z);
    }

    /// The correct challenge of neuron `j` of layer `l`: the committed vectors, the committed
    /// row, bias and proof.
    function neuronCall(uint256 id, uint256 l, uint256 j, GClaim memory c) internal view returns (bytes memory) {
        return neuronData(l, id, j, inputOf(c, l), outputOf(c, l), Gx.row(l, j), Gx.bias(l, j), Gx.proof(l, j));
    }

    function classCall(uint256 id, GClaim memory c) internal view returns (bytes memory) {
        return abi.encodePacked(classSel, id, logitWords(c.z));
    }

    /// The correct challenge of check `kind`: a neuron of layer `kind` (1 .. L) or the class (L + 1).
    function checkCall(uint256 id, uint256 kind, uint256 j, GClaim memory c) internal view returns (bytes memory) {
        return kind <= L ? neuronCall(id, kind, j, c) : classCall(id, c);
    }

    // ----------------------------------------------------------------------------------------
    // Calls

    function send(address who, bytes memory data) internal returns (bool ok, bytes memory ret) {
        vm.prank(who);
        (ok, ret) = address(arbiter).call(data);
    }

    function mustPass(address who, bytes memory data, string memory what) internal {
        (bool ok,) = send(who, data);
        require(ok, what);
    }

    /// The call must revert with exactly `err` (an encoded custom error).
    function mustFail(address who, bytes memory data, bytes memory err, string memory what) internal {
        (bool ok, bytes memory ret) = send(who, data);
        require(!ok && keccak256(ret) == keccak256(err), what);
    }

    function mustFail(address who, bytes memory data, bytes4 err, string memory what) internal {
        mustFail(who, data, abi.encodePacked(err), what);
    }

    function postRaw(bytes memory data, address who, uint256 value) internal returns (bool ok, bytes memory ret) {
        vm.deal(who, who.balance + value);
        vm.prank(who);
        (ok, ret) = address(arbiter).call{value: value}(data);
        // A refused claim returns the ETH; take it out again so balances stay easy to follow.
        if (!ok) vm.deal(who, who.balance - value);
    }

    function post(GClaim memory c, address who, uint256 value) internal returns (uint256) {
        (bool ok, bytes memory ret) = postRaw(claimData(c), who, value);
        require(ok, "a well-formed claim was refused");
        return abi.decode(ret, (uint256));
    }

    function statusOf(uint256 id) internal view returns (uint8 status) {
        (,,,, status) = arbiter.claimInfo(id);
    }

    function deadlineOf(uint256 id) internal view returns (uint256 deadline) {
        (,, deadline,,) = arbiter.claimInfo(id);
    }

    /// `claimHashes(id)`: one word per committed vector (input, hidden vectors, logits).
    function hashesOf(uint256 id) internal view returns (bool ok, uint256[] memory h) {
        bytes memory ret;
        (ok, ret) = address(arbiter).staticcall(abi.encodeWithSignature("claimHashes(uint256)", id));
        if (!ok) return (false, h);
        require(ret.length == 32 * (L + 1), "claimHashes returned another number of words");
        h = new uint256[](L + 1);
        for (uint256 k = 0; k <= L; k++) {
            uint256 w;
            assembly ("memory-safe") {
                w := mload(add(add(ret, 32), mul(k, 32)))
            }
            h[k] = w;
        }
    }

    /// The commitments the spec prescribes for claim `c` (section 3, item 2).
    function specHashes(GClaim memory c) internal pure returns (uint256[] memory h) {
        h = new uint256[](c.v.length + 1);
        for (uint256 k = 0; k < c.v.length; k++) {
            h[k] = uint256(keccak256(abi.encodePacked(c.v[k])));
        }
        h[c.v.length] = uint256(keccak256(abi.encodePacked(c.z)));
    }

    // ----------------------------------------------------------------------------------------
    // The spec's judgement

    /// Whether check `kind`, `j` fails for claim `c` by the spec (section 4): the neuron
    /// recomputed by GModel on the COMMITTED input of its layer differs from the committed
    /// output, or the class is not the argmax of the committed logits.
    function fails(GClaim memory c, uint256 kind, uint256 j) internal view returns (bool) {
        if (kind < L) return GModel.byteAt(c.v[kind], j) != GModel.hiddenNeuron(kind, j, c.v[kind - 1]);
        if (kind == L) return c.z[j] != GModel.logit(j, c.v[L - 1]);
        return c.cls != GModel.argmax(c.z);
    }

    /// The first failing check of the spec's walk (section 6): layer by layer, neuron by neuron,
    /// then the class. `kind` 0 means no check fails: the claim is honest.
    function walk(GClaim memory c) internal view returns (uint256 kind, uint256 j) {
        for (kind = 1; kind <= L; kind++) {
            for (j = 0; j < OUT[kind]; j++) {
                if (fails(c, kind, j)) return (kind, j);
            }
        }
        if (fails(c, L + 1, 0)) return (L + 1, 0);
        return (0, 0);
    }

    /// The first position at which `c` differs from `honest`, in the order of the walk. For a
    /// claim on the same input as an honest claim this IS the walk's answer (every check before
    /// it compares honest vectors, and the check at it recomputes from an honest input to the
    /// honest value), at a fraction of the cost.
    function firstDifference(GClaim memory c, GClaim memory honest) internal view returns (uint256 kind, uint256 j) {
        for (kind = 1; kind < L; kind++) {
            for (j = 0; j < OUT[kind]; j++) {
                if (GModel.byteAt(c.v[kind], j) != GModel.byteAt(honest.v[kind], j)) return (kind, j);
            }
        }
        for (j = 0; j < CLASSES; j++) {
            if (c.z[j] != honest.z[j]) return (L, j);
        }
        if (c.cls != honest.cls) return (L + 1, 0);
        return (0, 0);
    }

    /// A lie in one committed value of `c`, in place: output `j` of layer `l` becomes another
    /// value picked by `salt` (any other activation 0..127 for a hidden layer, any other int32 for
    /// a logit), or for `l == L + 1` the class becomes another class.
    function lie(GClaim memory c, uint256 l, uint256 j, uint256 salt) internal view {
        if (l < L) {
            uint256 h = GModel.byteAt(c.v[l], j);
            GModel.setByte(c.v[l], j, uint8((h + 1 + salt % 127) % 128));
        } else if (l == L) {
            int32 v = int32(uint32(salt));
            c.z[j] = v == c.z[j] ? v ^ 1 : v;
        } else {
            c.cls = uint8((c.cls + 1 + salt % (CLASSES - 1)) % CLASSES);
        }
    }

    /// The smallest lie: one up or one down where the range allows (`false` if it does not).
    function nudge(GClaim memory c, uint256 l, uint256 j, bool up) internal view returns (bool) {
        if (l < L) {
            uint8 h = GModel.byteAt(c.v[l], j);
            if ((up && h == 127) || (!up && h == 0)) return false;
            GModel.setByte(c.v[l], j, up ? h + 1 : h - 1);
        } else if (l == L) {
            if ((up && c.z[j] == type(int32).max) || (!up && c.z[j] == type(int32).min)) return false;
            c.z[j] = up ? c.z[j] + 1 : c.z[j] - 1;
        } else {
            if (CLASSES < 3 && !up) return false;
            c.cls = uint8((c.cls + (up ? 1 : CLASSES - 1)) % CLASSES);
        }
        return true;
    }
}
