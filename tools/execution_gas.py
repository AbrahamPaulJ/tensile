"""Gas of a transaction's execution: its gasUsed minus the intrinsic cost of the transaction.

Usage:
    python3 tools/execution_gas.py CALLDATA_HEX GAS_USED

The intrinsic cost of a call (Osaka) is 21,000 plus 4 gas per calldata token, a zero byte being one
token and any other byte four (EIP-2028, EIP-7623). EIP-7623 also sets a floor of 21,000 plus 10 gas
per token; when the floor is what was charged, the execution cost is not visible in gasUsed and
"floor" is printed instead. Standard library only.
"""

import sys


def main():
    data = bytes.fromhex(sys.argv[1].strip()[2:])
    used = int(sys.argv[2])
    zeros = data.count(0)
    tokens = zeros + 4 * (len(data) - zeros)
    if used == 21000 + 10 * tokens:
        print("floor")
    else:
        print(used - 21000 - 4 * tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
