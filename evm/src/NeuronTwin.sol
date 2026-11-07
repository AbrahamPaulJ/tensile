// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

/// The Solidity twin of Tensile's `NeuronCheck`: the same two functions, the same calldata layout
/// (docs/spec.md section 7), the same validation, the same results, written for gas the way a
/// Solidity engineer would write them: operands read straight from calldata, the lanes taken with
/// `byte` and `signextend`, an unchecked loop, and the accumulator a plain 256-bit word.
///
/// It is the gas baseline the Fe contract is compared with (docs/fe-notes.md, milestone 2). It is
/// not a strawman: `neuron1` and `neuron2` are the natural assembly loop; the `Unrolled` variants
/// write the 32 lanes of a word out, as the Fe kernel does, so both shapes are measured and the
/// cheaper one is the baseline.
contract NeuronTwin {
    /// The sign bit of each byte of a word.
    uint256 private constant SIGN_BITS = 0x8080808080808080808080808080808080808080808080808080808080808080;
    /// The shift of layer 1 (model.json).
    uint256 private constant SHIFT_1 = 11;

    error Malformed();

    /// The activation of a layer 1 neuron: 784 weights, 784 inputs in 0..127, 25 words each.
    function neuron1(uint256[25] calldata row, uint256[25] calldata x, int32 bias) external pure returns (int8) {
        validate(row, x);
        int256 acc = bias;
        assembly ("memory-safe") {
            for { let k := 0 } lt(k, 25) { k := add(k, 1) } {
                let w := calldataload(add(row, shl(5, k)))
                let a := calldataload(add(x, shl(5, k)))
                for { let i := 0 } lt(i, 32) { i := add(i, 1) } {
                    acc := add(acc, mul(signextend(0, byte(i, w)), byte(i, a)))
                }
            }
        }
        return requantise(acc, SHIFT_1);
    }

    /// `neuron1` with the 32 lanes of a word written out.
    function neuron1Unrolled(uint256[25] calldata row, uint256[25] calldata x, int32 bias)
        external
        pure
        returns (int8)
    {
        validate(row, x);
        int256 acc = bias;
        for (uint256 k = 0; k < 25; k++) {
            unchecked {
                acc += dotWord(row[k], x[k]);
            }
        }
        return requantise(acc, SHIFT_1);
    }

    /// The logit of a layer 2 neuron: 32 weights, 32 inputs, one word each.
    function neuron2(uint256[1] calldata row, uint256[1] calldata x, int32 bias) external pure returns (int32) {
        uint256 w = row[0];
        uint256 a = x[0];
        if (a & SIGN_BITS != 0) revert Malformed();
        int256 acc = bias;
        assembly ("memory-safe") {
            for { let i := 0 } lt(i, 32) { i := add(i, 1) } {
                acc := add(acc, mul(signextend(0, byte(i, w)), byte(i, a)))
            }
        }
        return int32(acc);
    }

    /// `neuron2` with the 32 lanes written out.
    function neuron2Unrolled(uint256[1] calldata row, uint256[1] calldata x, int32 bias) external pure returns (int32) {
        uint256 a = x[0];
        if (a & SIGN_BITS != 0) revert Malformed();
        unchecked {
            return int32(bias + dotWord(row[0], a));
        }
    }

    /// The dot product of the 32 signed weight bytes of `w` and the 32 activation bytes of `a`,
    /// written out lane by lane.
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

    /// The packed-word rule for 784 values in 25 words: the low 16 bytes of the last word are
    /// padding and must be zero; every input byte must be 0..127.
    function validate(uint256[25] calldata row, uint256[25] calldata x) private pure {
        if (row[24] << 128 != 0 || x[24] << 128 != 0) revert Malformed();
        for (uint256 k = 0; k < 25; k++) {
            if (x[k] & SIGN_BITS != 0) revert Malformed();
        }
    }

    /// `clamp(acc >> shift, 0, 127)`; the shift rounds toward negative infinity.
    function requantise(int256 acc, uint256 shift) private pure returns (int8) {
        int256 shifted = acc >> shift;
        if (shifted < 0) return 0;
        if (shifted > 127) return 127;
        return int8(shifted);
    }
}
