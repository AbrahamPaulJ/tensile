// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

/// The Foundry cheatcodes these tests use, declared here so the project has no dependency.
/// The cheatcode contract lives at address(uint160(uint256(keccak256("hevm cheat code")))).
interface Vm {
    function readFileBinary(string calldata path) external view returns (bytes memory data);
    function readLine(string calldata path) external returns (string memory line);
    function closeFile(string calldata path) external;
    function parseJsonUint(string calldata json, string calldata key) external pure returns (uint256);
    function parseJsonIntArray(string calldata json, string calldata key) external pure returns (int256[] memory);
    function expectRevert() external;
    function assume(bool condition) external pure;
}

library Cheats {
    Vm internal constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
}
