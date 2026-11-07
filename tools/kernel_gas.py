"""Read a `KernelGas` report from the output of `fe test --show-logs ingots/onchain`.

Usage:
    fe test --show-logs ingots/onchain | tr '\r' '\n' | python3 tools/kernel_gas.py LAYER

Prints the gas that one neuron of layer LAYER used inside the EVM (the `gas()` difference measured
by `ingots/onchain/src/gas.fe`). The report is an event: the layer is its second topic and the gas
its data. Standard library only.
"""

import re
import sys


def main():
    layer = int(sys.argv[1])
    pattern = re.compile(r"topics: \[0x[0-9a-f]{64}, 0x([0-9a-f]{64})\], data: 0x([0-9a-f]{64})")
    for line in sys.stdin:
        match = pattern.search(line)
        if match and int(match.group(1), 16) == layer:
            print(int(match.group(2), 16))
            return 0
    print("kernel_gas: no report for layer %d" % layer, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
