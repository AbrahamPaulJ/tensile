# Generates N random int8 input samples for the layer in <weights dir>, writes:
#   samples.bin  (N * IN raw bytes, stdin for the native executable)
#   samples_ref.txt (reference outputs, one line per sample, computed here with Python ints)
#   samples_abi.txt (one "[a,b,...]" int8[IN] literal per line, for cast call)
# usage: python3 gen_samples.py <path to weights.fe> <out dir> <N>
import sys, re, os, random

src = open(sys.argv[1]).read()
outdir, N = sys.argv[2], int(sys.argv[3])


def const(name):
    m = re.search(r"pub const %s: [^=]+= (.*?)(?=\npub const|\nstatic_assert|\Z)" % name, src, re.S)
    return eval(m.group(1))


IN, OUT, SHIFT = const("IN"), const("OUT"), const("SHIFT")
W, B = const("W"), const("B")
random.seed(20261105)
os.makedirs(outdir, exist_ok=True)
raw = bytearray()
ref, abi = [], []
for n in range(N):
    x = [random.randint(-128, 127) for _ in range(IN)]
    raw += bytes(v & 0xFF for v in x)
    y = []
    for o in range(OUT):
        acc = B[o] + sum(W[o][i] * x[i] for i in range(IN))
        acc = max(acc, 0) >> SHIFT
        y.append(min(acc, 127))
    ref.append(" ".join(map(str, y)))
    abi.append("[" + ",".join(map(str, x)) + "]")
open(os.path.join(outdir, "samples.bin"), "wb").write(raw)
open(os.path.join(outdir, "samples_ref.txt"), "w", newline="\n").write("\n".join(ref) + "\n")
open(os.path.join(outdir, "samples_abi.txt"), "w", newline="\n").write("\n".join(abi) + "\n")
print("wrote", N, "samples of", IN, "bytes")
