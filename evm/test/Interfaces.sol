// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

/// The Fe contracts of ingots/onchain, as Solidity sees them.
interface INetwork {
    function classify(uint256[25] calldata x) external view returns (uint256);
    function trace(uint256[25] calldata x) external view returns (uint256[1] memory hidden, int32[10] memory logits);
}

interface INeuronCheck {
    function neuron1(uint256[25] calldata row, uint256[25] calldata x, int32 bias) external view returns (int8);
    function neuron2(uint256[1] calldata row, uint256[1] calldata x, int32 bias) external view returns (int32);
}
