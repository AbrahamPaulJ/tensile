# Integer reference for q3_tables.fe (same algorithm, Python big ints).
ONE = 1 << 64
s = ONE
term = ONE
for n in range(1, 16):
    term = term // (16 * n)
    s += term
E16 = s


def entry(i):
    k = i - 128 if i >= 128 else 128 - i
    p = ONE
    for _ in range(k):
        p = (p * E16) >> 64
    num = p if i >= 128 else ONE
    den = ONE + p
    return ((num * 255 + den // 2) // den) & 0xFF


t = [entry(i) for i in range(256)]
print("E16", E16)
for i in (0, 64, 100, 128, 144, 200, 250, 255):
    print(i, t[i])
import math
worst = max(abs(t[i] - 255 / (1 + math.exp(-(i - 128) / 16))) for i in range(256))
print("max abs error vs float sigmoid*255:", worst)
