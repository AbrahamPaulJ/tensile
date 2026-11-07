"""Keccak-256 in pure Python (the Ethereum hash, not NIST SHA3-256: padding byte 0x01).

`hashlib` has no Keccak-256, and the tooling is limited to the standard library, so this is the
reference implementation of FIPS 202 section 3 written out plainly. It is slow (about 1 ms per
block) and only hashes the few kilobytes a model commitment needs. `tools/m3_check.sh` compares it
with `cast keccak`, and the Fe tests verify the proofs it produces with `std::evm::merkle`.

    python tools/keccak.py <hex or text>   prints keccak256 of the argument (0x-prefixed hex is bytes)
"""

import sys

ROTATIONS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

ROUND_CONSTANTS = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

MASK = (1 << 64) - 1
RATE = 136  # bytes absorbed per block for a 256-bit output


def _rotl(value, bits):
    return ((value << bits) | (value >> (64 - bits))) & MASK


def _keccak_f(state):
    """The permutation on 25 lanes of 64 bits, state[x][y]."""
    for constant in ROUND_CONSTANTS:
        # theta
        parity = [state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4] for x in range(5)]
        delta = [parity[(x - 1) % 5] ^ _rotl(parity[(x + 1) % 5], 1) for x in range(5)]
        state = [[state[x][y] ^ delta[x] for y in range(5)] for x in range(5)]
        # rho and pi
        moved = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                moved[y][(2 * x + 3 * y) % 5] = _rotl(state[x][y], ROTATIONS[x][y])
        # chi
        state = [[moved[x][y] ^ ((~moved[(x + 1) % 5][y]) & moved[(x + 2) % 5][y]) for y in range(5)]
                 for x in range(5)]
        # iota
        state[0][0] ^= constant
    return state


def keccak256(data):
    """Keccak-256 digest of `data` (bytes), as 32 bytes."""
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % RATE != 0:
        padded.append(0x00)
    padded[-1] |= 0x80
    state = [[0] * 5 for _ in range(5)]
    for start in range(0, len(padded), RATE):
        block = padded[start:start + RATE]
        for i in range(RATE // 8):
            lane = int.from_bytes(block[8 * i:8 * i + 8], "little")
            state[i % 5][i // 5] ^= lane
        state = _keccak_f(state)
    out = b"".join(state[i % 5][i // 5].to_bytes(8, "little") for i in range(4))
    return out


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    data = bytes.fromhex(text[2:]) if text.startswith("0x") else text.encode("utf-8")
    print("0x" + keccak256(data).hex())
    return 0


if __name__ == "__main__":
    sys.exit(main())
