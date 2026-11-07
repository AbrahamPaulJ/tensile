"""Check the raw return of `Network.trace` for reference sample 0 against trace.jsonl.

Usage:
    python3 tools/check_trace.py RAW_HEX ref/out/trace.jsonl

RAW_HEX is the ABI-encoded return value of `trace(uint256[25]) -> (uint256[1], int32[10])`: the
packed hidden activations as one word, then the ten logits as 256-bit two's complement words.
Exits non-zero if either differs from the first line of trace.jsonl. Standard library only.
"""

import json
import sys


def main():
    raw = bytes.fromhex(sys.argv[1].strip()[2:])
    with open(sys.argv[2], "r", encoding="utf-8") as f:
        reference = json.loads(f.readline())
    hidden = bytes(reference["hidden"][0]).ljust(32, b"\0")
    logits = [int.from_bytes(raw[32 * (k + 1):32 * (k + 2)], "big", signed=True) for k in range(10)]
    if raw[:32] != hidden or logits != reference["logits"]:
        print("trace: differs from the reference: %s %s" % (raw[:32].hex(), logits))
        return 1
    print("Network.trace returns the reference hidden activations and logits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
