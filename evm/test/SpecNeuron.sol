// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

/// The neuron arithmetic of docs/spec.md, written as plainly as possible: the independent second
/// implementation the Fe kernel is fuzzed against. Nothing here is optimised.
library SpecNeuron {
    /// `acc = bias + sum over i of w[i] * x[i]` (spec section 3) for `n` values, each operand given
    /// as its byte encoding (spec section 7: one byte per int8 value, two's complement).
    function accumulate(bytes memory weights, bytes memory inputs, uint256 n, int32 bias)
        internal
        pure
        returns (int256 acc)
    {
        acc = bias;
        for (uint256 i = 0; i < n; i++) {
            int256 w = int8(uint8(weights[i]));
            int256 x = int8(uint8(inputs[i]));
            acc += w * x;
        }
    }

    /// A hidden layer's output: `clamp(acc >> shift, 0, 127)`, the shift rounding toward negative
    /// infinity (spec section 3). Solidity's `>>` on a signed integer is that shift.
    function requantise(int256 acc, uint256 shift) internal pure returns (int8) {
        int256 shifted = acc >> shift;
        if (shifted < 0) return 0;
        if (shifted > 127) return 127;
        return int8(shifted);
    }

    /// Whether `words` is a well-formed packed vector of `n` values (spec section 7): every byte
    /// past value `n - 1` is zero.
    function paddingIsZero(bytes memory words, uint256 n) internal pure returns (bool) {
        for (uint256 i = n; i < words.length; i++) {
            if (words[i] != 0) return false;
        }
        return true;
    }

    /// Whether every one of the `n` values is an activation, 0..127 (spec section 2).
    function allActivations(bytes memory words, uint256 n) internal pure returns (bool) {
        for (uint256 i = 0; i < n; i++) {
            if (uint8(words[i]) > 127) return false;
        }
        return true;
    }
}
