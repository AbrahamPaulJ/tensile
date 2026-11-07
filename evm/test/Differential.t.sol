// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {Cheats, Vm} from "./Vm.sol";
import {FeDeployer} from "./FeDeployer.sol";
import {INetwork} from "./Interfaces.sol";

/// The differential test: the Fe `Network` deployed from its built bytecode against the NumPy
/// reference vectors in ref-out/ (docs/spec.md section 6). Every class of every sample in
/// inputs.bin must equal pred.bin, and for the traced samples every hidden activation, every logit
/// and the class must equal trace.jsonl. Zero mismatches is the pass condition.
contract DifferentialTest {
    Vm private constant vm = Cheats.vm;
    uint256 private constant INPUTS = 784;
    uint256 private constant WORDS = 25;

    INetwork private network;

    function setUp() public {
        address low = FeDeployer.deploy("RowsLow");
        address high = FeDeployer.deploy("RowsHigh");
        network = INetwork(FeDeployer.deploy("Network", abi.encode(low, high)));
    }

    /// Sample `n` of inputs.bin as 25 packed words.
    function packedInput(bytes memory inputs, uint256 n) internal pure returns (uint256[25] memory x) {
        uint256 base = n * INPUTS;
        for (uint256 i = 0; i < INPUTS; i++) {
            x[i / 32] |= uint256(uint8(inputs[base + i])) << (8 * (31 - i % 32));
        }
    }

    function test_every_class_matches_pred_bin() public view {
        bytes memory inputs = vm.readFileBinary("ref-out/inputs.bin");
        bytes memory pred = vm.readFileBinary("ref-out/pred.bin");
        require(inputs.length == pred.length * INPUTS, "inputs.bin and pred.bin disagree on N");
        uint256 mismatches = 0;
        for (uint256 n = 0; n < pred.length; n++) {
            if (network.classify(packedInput(inputs, n)) != uint8(pred[n])) mismatches++;
        }
        require(mismatches == 0, "a class differs from the reference");
    }

    function test_every_trace_matches_trace_jsonl() public {
        bytes memory inputs = vm.readFileBinary("ref-out/inputs.bin");
        uint256 traced = 0;
        uint256 mismatches = 0;
        while (true) {
            string memory line = vm.readLine("ref-out/trace.jsonl");
            if (bytes(line).length == 0) break;
            uint256 n = vm.parseJsonUint(line, ".i");
            require(n == traced, "trace.jsonl is not numbered from 0 in order");
            int256[] memory hidden = vm.parseJsonIntArray(line, ".hidden[0]");
            int256[] memory logits = vm.parseJsonIntArray(line, ".logits");
            uint256 class = vm.parseJsonUint(line, ".class");
            if (!traceMatches(packedInput(inputs, n), hidden, logits, class)) mismatches++;
            traced++;
        }
        vm.closeFile("ref-out/trace.jsonl");
        require(traced >= 1000, "fewer than 1,000 traced samples");
        require(mismatches == 0, "a trace differs from the reference");
    }

    function traceMatches(uint256[25] memory x, int256[] memory hidden, int256[] memory logits, uint256 class)
        internal
        view
        returns (bool)
    {
        (uint256[1] memory gotHidden, int32[10] memory gotLogits) = network.trace(x);
        uint256 wantHidden = 0;
        for (uint256 j = 0; j < 32; j++) {
            wantHidden |= uint256(hidden[j]) << (8 * (31 - j));
        }
        if (gotHidden[0] != wantHidden) return false;
        for (uint256 j = 0; j < 10; j++) {
            if (gotLogits[j] != logits[j]) return false;
        }
        return network.classify(x) == class;
    }
}
