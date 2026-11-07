// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {Cheats, Vm} from "./Vm.sol";

/// Deploys contracts built by `fe build`: the creation code is read from `fe-out/<Name>.bin`
/// (hex, no 0x prefix) and the ABI-encoded constructor arguments are appended to it, which is
/// how Fe's `init` receives them.
library FeDeployer {
    function deploy(string memory name, bytes memory args) internal returns (address deployed) {
        bytes memory code = bytes.concat(creationCode(name), args);
        assembly ("memory-safe") {
            deployed := create(0, add(code, 32), mload(code))
        }
        require(deployed != address(0), string.concat("deploying ", name, " failed"));
    }

    function deploy(string memory name) internal returns (address) {
        return deploy(name, "");
    }

    function creationCode(string memory name) internal view returns (bytes memory) {
        bytes memory hex_ = Cheats.vm.readFileBinary(string.concat("fe-out/", name, ".bin"));
        if (hex_.length > 0 && hex_[hex_.length - 1] == 0x0a) {
            // A trailing newline, if the file has one.
            assembly ("memory-safe") {
                mstore(hex_, sub(mload(hex_), 1))
            }
        }
        require(hex_.length % 2 == 0, "odd hex length");
        bytes memory code = new bytes(hex_.length / 2);
        for (uint256 i = 0; i < code.length; i++) {
            code[i] = bytes1(nibble(hex_[2 * i]) * 16 + nibble(hex_[2 * i + 1]));
        }
        return code;
    }

    function nibble(bytes1 c) private pure returns (uint8) {
        uint8 v = uint8(c);
        if (v >= 0x30 && v <= 0x39) return v - 0x30;
        if (v >= 0x61 && v <= 0x66) return v - 0x61 + 10;
        if (v >= 0x41 && v <= 0x46) return v - 0x41 + 10;
        revert("not a hex digit");
    }
}
