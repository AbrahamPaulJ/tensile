// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {Cheats, Vm} from "./Vm.sol";
import {FeDeployer} from "./FeDeployer.sol";
import {INeuronCheck} from "./Interfaces.sol";
import {SpecNeuron} from "./SpecNeuron.sol";
import {NeuronTwin} from "../src/NeuronTwin.sol";

/// Fuzz tests: for random packed rows, inputs and biases, the Fe `NeuronCheck` (and the Solidity
/// twin) equal the plain implementation of the spec in SpecNeuron.sol, and malformed words are
/// rejected.
contract NeuronFuzzTest {
    Vm private constant vm = Cheats.vm;

    /// 0x80 in every byte: cleared from random inputs to make them activations.
    uint256 private constant SIGN_BITS = 0x8080808080808080808080808080808080808080808080808080808080808080;
    /// 784 = 24 * 32 + 16: the low 16 bytes of word 24 are padding.
    uint256 private constant PADDING_1 = (1 << 128) - 1;
    /// The largest |sum of 784 products|: the bias is kept so that the int32 accumulator the
    /// kernel uses cannot overflow (the spec requires the model to guarantee this).
    int32 private constant BIAS_MARGIN_1 = 784 * 128 * 127;
    int32 private constant BIAS_MARGIN_2 = 32 * 128 * 127;
    uint256 private constant SHIFT_1 = 11;

    INeuronCheck private check;
    NeuronTwin private twin;

    function setUp() public {
        check = INeuronCheck(FeDeployer.deploy("NeuronCheck"));
        twin = new NeuronTwin();
    }

    function wellFormed1(uint256[25] memory row, uint256[25] memory x) internal pure {
        for (uint256 k = 0; k < 25; k++) {
            x[k] &= ~SIGN_BITS;
        }
        row[24] &= ~PADDING_1;
        x[24] &= ~PADDING_1;
    }

    function testFuzz_layer_1_neuron_equals_the_spec(uint256[25] memory row, uint256[25] memory x, int32 bias)
        public
        view
    {
        vm.assume(bias > type(int32).min + BIAS_MARGIN_1 && bias < type(int32).max - BIAS_MARGIN_1);
        wellFormed1(row, x);
        int256 acc = SpecNeuron.accumulate(abi.encodePacked(row), abi.encodePacked(x), 784, bias);
        int8 want = SpecNeuron.requantise(acc, SHIFT_1);
        require(check.neuron1(row, x, bias) == want, "Fe neuron1 differs from the spec");
        require(twin.neuron1(row, x, bias) == want, "Solidity neuron1 differs from the spec");
        require(twin.neuron1Unrolled(row, x, bias) == want, "Solidity neuron1Unrolled differs from the spec");
    }

    function testFuzz_layer_2_neuron_equals_the_spec(uint256 row, uint256 x, int32 bias) public view {
        vm.assume(bias > type(int32).min + BIAS_MARGIN_2 && bias < type(int32).max - BIAS_MARGIN_2);
        x &= ~SIGN_BITS;
        int256 acc = SpecNeuron.accumulate(abi.encodePacked(row), abi.encodePacked(x), 32, bias);
        require(acc == int32(acc), "the test's own bound is wrong");
        require(check.neuron2([row], [x], bias) == int32(acc), "Fe neuron2 differs from the spec");
        require(twin.neuron2([row], [x], bias) == int32(acc), "Solidity neuron2 differs from the spec");
        require(twin.neuron2Unrolled([row], [x], bias) == int32(acc), "Solidity neuron2Unrolled differs from the spec");
    }

    function testFuzz_an_input_above_127_is_rejected(uint256[25] memory row, uint256[25] memory x, uint16 where)
        public
    {
        wellFormed1(row, x);
        uint256 i = where % 784;
        x[i / 32] |= uint256(0x80) << (8 * (31 - i % 32));
        vm.expectRevert();
        check.neuron1(row, x, 0);
        vm.expectRevert();
        twin.neuron1(row, x, 0);
        vm.expectRevert();
        twin.neuron1Unrolled(row, x, 0);
    }

    function testFuzz_input_padding_that_is_not_zero_is_rejected(uint256[25] memory row, uint256[25] memory x, uint128 dirt)
        public
    {
        vm.assume(dirt != 0);
        wellFormed1(row, x);
        x[24] |= uint256(dirt);
        // The padding is checked on its own, so a padding byte with its sign bit set is also a
        // rejection for the right reason in the Fe kernel: both checks run before any arithmetic.
        vm.expectRevert();
        check.neuron1(row, x, 0);
        vm.expectRevert();
        twin.neuron1(row, x, 0);
        vm.expectRevert();
        twin.neuron1Unrolled(row, x, 0);
    }

    function testFuzz_row_padding_that_is_not_zero_is_rejected(uint256[25] memory row, uint256[25] memory x, uint128 dirt)
        public
    {
        vm.assume(dirt != 0);
        wellFormed1(row, x);
        row[24] |= uint256(dirt);
        vm.expectRevert();
        check.neuron1(row, x, 0);
        vm.expectRevert();
        twin.neuron1(row, x, 0);
        vm.expectRevert();
        twin.neuron1Unrolled(row, x, 0);
    }

    function testFuzz_a_layer_2_input_above_127_is_rejected(uint256 row, uint256 x, uint8 where) public {
        x &= ~SIGN_BITS;
        x |= uint256(0x80) << (8 * (31 - where % 32));
        vm.expectRevert();
        check.neuron2([row], [x], 0);
        vm.expectRevert();
        twin.neuron2([row], [x], 0);
        vm.expectRevert();
        twin.neuron2Unrolled([row], [x], 0);
    }
}
