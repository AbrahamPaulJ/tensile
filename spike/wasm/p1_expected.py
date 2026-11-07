M = (1 << 256) - 1
SIGN = int("80" * 32, 16)
B = int("0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20", 16)
def probe(x, i):
    r = ((((x * B) & M) + (x >> 7)) & M) ^ ((B << 200) & M) ^ SIGN
    return (r >> (8 * i)) & 0xFF
print(probe(3, 0), probe(3, 31))
