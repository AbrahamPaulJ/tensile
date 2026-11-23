"""Where the gas gap between the Fe contracts and their Solidity twins goes, opcode by opcode.

Usage (WSL):
    python3 tools/gas_gap.py [--port 18731] [--keep DIR]

Builds, in a scratch copy (nothing in the repository is written):
  - the `onchain` ingot as shipped, and again with `tests/kernel_forms/i256_unrolled.fe` as the kernel
    (it needs `build/tree.json`, which `tools/gen_model.sh` writes);
  - the Solidity twins of `evm/src` with the settings of `evm/foundry.toml`.
Then, on its own anvil (Osaka), it sends the transactions of `docs/numbers.md` section 3 (reference
sample 0, same calldata to both sides, every result checked), takes `debug_traceTransaction` of each,
and prints for each pair:
  - the gas of every executed opcode, grouped into classes, Fe against Solidity, and the difference;
  - the narrow-integer masks executed (`PUSHn ff..ff AND`, `PUSH0 NOT PUSH1 k SHR AND`,
    `PUSH1 k SIGNEXTEND`), by width, with their gas;
  - for the neuron, the same split into "before the first MUL", "first to last MUL" (the dot product)
    and "after the last MUL";
  - how array arguments reach memory (fills, calldata decode loops, word copies) and the loops that
    run before the first MUL.
The classes of one trace sum to its execution gas (gasUsed minus 21,000 and the calldata), which the
script asserts, so each table adds up to the gap with nothing left over.
It also prints a static breakdown of the runtime bytecode of each contract, with the Fe contracts
built a third time on `tests/kernel_forms/i256_loop.fe`, the smallest kernel.
`--keep DIR` saves the traces and the deployed code there. Standard library only; needs `fe`, `forge`,
`cast` and `anvil` on PATH or in ~/.fe/bin and ~/.foundry/bin. About one minute.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
BOND = 10**18
WINDOW = 3600

os.environ["PATH"] = os.pathsep.join(
    [os.path.expanduser("~/.fe/bin"), os.path.expanduser("~/.foundry/bin"), os.environ["PATH"]]
)

CLASSES = [
    ("narrow-integer masks", None),  # filled from matched sequences, not from single opcodes
    ("storage (SLOAD, SSTORE)", {"SLOAD", "SSTORE"}),
    ("hashing (KECCAK256)", {"KECCAK256", "SHA3"}),
    ("events (LOG)", {"LOG0", "LOG1", "LOG2", "LOG3", "LOG4"}),
    ("memory (MLOAD, MSTORE, copies)", {"MLOAD", "MSTORE", "MSTORE8", "CALLDATACOPY", "MCOPY", "CODECOPY",
                                        "RETURNDATACOPY", "MSIZE"}),
    ("calldata reads", {"CALLDATALOAD", "CALLDATASIZE"}),
    ("arithmetic and bit operations", {"ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "ADDMOD", "MULMOD",
                                       "EXP", "SIGNEXTEND", "AND", "OR", "XOR", "NOT", "BYTE", "SHL", "SHR",
                                       "SAR"}),
    ("comparisons", {"LT", "GT", "SLT", "SGT", "EQ", "ISZERO"}),
    ("stack (PUSH, DUP, SWAP, POP)", None),  # by prefix
    ("control flow (JUMP, JUMPI, JUMPDEST)", {"JUMP", "JUMPI", "JUMPDEST", "PC"}),
    ("other", None),
]


def run(*command, cwd=None):
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit("FAILED: %s\n%s\n%s" % (" ".join(command)[:300], done.stdout[-3000:], done.stderr[-3000:]))
    return done.stdout.strip()


class Chain:
    def __init__(self, port):
        self.url = "http://127.0.0.1:%d" % port
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                sys.exit("port %d is in use: pass another with --port" % port)
        self.process = subprocess.Popen(["anvil", "--hardfork", "osaka", "--steps-tracing", "--port", str(port)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            time.sleep(0.2)
            try:
                self.rpc("eth_chainId")
                return
            except OSError:
                pass
        sys.exit("anvil did not start")

    def stop(self):
        self.process.terminate()

    def rpc(self, method, *params):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.load(response)
        if "error" in answer:
            sys.exit("%s: %s" % (method, answer["error"]))
        return answer["result"]

    def deploy(self, creation_hex, constructor_args=""):
        out = run("cast", "send", "--rpc-url", self.url, "--private-key", KEY, "--json", "--create",
                  "0x" + creation_hex + constructor_args)
        return json.loads(out)["contractAddress"]

    def call(self, address, signature, *args):
        return run("cast", "call", "--rpc-url", self.url, address, signature, *args)

    def send(self, address, signature, *args, value=0):
        """Mines the call; returns (transaction hash, gasUsed, execution gas)."""
        inputs = signature[:signature.index(")") + 1] if not signature.endswith(")") else signature
        out = json.loads(run("cast", "send", "--rpc-url", self.url, "--private-key", KEY, "--json", "--value",
                             str(value), address, inputs, *args))
        if int(out["status"], 16) != 1:
            sys.exit("reverted: %s" % signature)
        data = bytes.fromhex(run("cast", "calldata", inputs, *args)[2:])
        zeros = data.count(0)
        tokens = zeros + 4 * (len(data) - zeros)
        used = int(out["gasUsed"], 16)
        if used == 21000 + 10 * tokens:
            sys.exit("%s was charged the calldata floor: its execution gas is not visible" % signature)
        return out["transactionHash"], used, used - 21000 - 4 * tokens

    def code(self, address):
        return bytes.fromhex(self.rpc("eth_getCode", address, "latest")[2:])

    def trace(self, tx_hash):
        result = self.rpc("debug_traceTransaction", tx_hash,
                          {"disableStorage": True, "disableMemory": True, "disableStack": True})
        return [(step["pc"], step["op"], int(step["gasCost"])) for step in result["structLogs"]]


def immediates(code):
    """pc -> (width, value) of every PUSH in the code, read linearly."""
    found = {}
    pc = 0
    while pc < len(code):
        op = code[pc]
        if 0x60 <= op <= 0x7F:
            width = op - 0x5F
            found[pc] = (width, int.from_bytes(code[pc + 1:pc + 1 + width], "big"))
            pc += width
        pc += 1
    return found


def find_masks(steps, pushes):
    """The narrow-integer masks among the executed steps: (kind -> [count, gas], set of step indices).

    Kinds: 'uN mask' for `PUSHn <n bytes of ff> AND` (also with one DUP between the two) with N = 8n
    above 8 bits, and for `PUSH0 NOT PUSH1 k SHR AND` with N = 256 - k, each named with the opcode
    executed just before it (MUL, ADD, or anything else: a truncation or a loaded value);
    'iN sign extension' for `PUSH1 k SIGNEXTEND` with N = 8(k + 1). `PUSH1 0xff AND` is counted as
    'low byte (& 0xff)' but its steps are not in the set: in the kernel it is the source's own lane
    extraction, not a mask the compiler added.
    """
    kinds, masked = {}, set()
    ops = [step[1] for step in steps]
    i = 0
    while i < len(steps):
        pc, op, _ = steps[i]
        kind, members = None, ()
        before = ops[i - 1] if i and ops[i - 1] in ("MUL", "ADD") else "other"
        if op.startswith("PUSH") and op != "PUSH0" and i + 2 < len(steps):
            width, value = pushes.get(pc, (0, -1))
            direct = ops[i + 1] == "AND"
            if value == (1 << (8 * width)) - 1 and (direct or (ops[i + 1].startswith("DUP") and ops[i + 2] == "AND")):
                members = (i, i + 1) if direct else (i, i + 2)
                kind = "low byte (& 0xff)" if width == 1 else "u%d mask after %s" % (8 * width, before)
            elif ops[i + 1] == "SIGNEXTEND" and width == 1 and value < 31:
                kind, members = "i%d sign extension" % (8 * (value + 1)), (i, i + 1)
        elif op == "PUSH0" and ops[i + 1:i + 5] == ["NOT", "PUSH1", "SHR", "AND"]:
            kind = "u%d mask after %s" % (256 - pushes[steps[i + 2][0]][1], before)
            members = tuple(range(i, i + 5))
        if kind:
            entry = kinds.setdefault(kind, [0, 0])
            entry[0] += 1
            entry[1] += sum(steps[j][2] for j in members)
            if not kind.startswith("low byte"):
                masked.update(members)
            i = members[-1] + 1
        else:
            i += 1
    return kinds, masked


def classify(steps, masked):
    """class name -> [steps, gas]; the mask class takes the steps of every compiler-added mask."""
    totals = {name: [0, 0] for name, _ in CLASSES}
    for i, (_, op, gas) in enumerate(steps):
        if i in masked:
            name = "narrow-integer masks"
        elif op.startswith(("PUSH", "DUP", "SWAP")) or op == "POP":
            name = "stack (PUSH, DUP, SWAP, POP)"
        else:
            name = next((n for n, ops in CLASSES if ops and op in ops), "other")
        totals[name][0] += 1
        totals[name][1] += gas
    return totals


def compare(title, left_name, left, right_name, right):
    """left and right are (steps, code, execution gas). Prints the class table and the masks."""
    print("\n## %s" % title)
    rows = []
    for name, (steps, code, execution) in ((left_name, left), (right_name, right)):
        total = sum(step[2] for step in steps)
        if total != execution:
            sys.exit("%s: the trace sums to %d, the receipt says %d of execution" % (name, total, execution))
        kinds, masked = find_masks(steps, immediates(code))
        rows.append((name, steps, execution, kinds, classify(steps, masked)))
    gap = rows[0][2] - rows[1][2]
    print("%s: %d steps, %d gas of execution. %s: %d steps, %d gas. Gap %d gas."
          % (rows[0][0], len(rows[0][1]), rows[0][2], rows[1][0], len(rows[1][1]), rows[1][2], gap))
    print("%-38s %8s %8s %8s %8s %8s %7s" % ("class", "Fe steps", "Fe gas", "Sol stps", "Sol gas", "diff", "share"))
    for name, _ in CLASSES:
        a, b = rows[0][4][name], rows[1][4][name]
        if a[0] or b[0]:
            print("%-38s %8d %8d %8d %8d %+8d %6.1f%%"
                  % (name, a[0], a[1], b[0], b[1], a[1] - b[1], 100.0 * (a[1] - b[1]) / gap if gap else 0))
    for name, steps, _, kinds, _ in rows:
        print("%s, masks executed: %s" % (name, "; ".join(
            "%s x %d = %d gas" % (k, kinds[k][0], kinds[k][1]) for k in sorted(kinds)) or "none"))
        top = Counter()
        for _, op, gas in steps:
            top[op] += gas
        print("%s, top opcodes by gas: %s" % (name, ", ".join("%s %d" % pair for pair in top.most_common(12))))
    for name, steps, _, _, _ in rows:
        arrays(name, steps)


def arrays(name, steps):
    """How array arguments reach memory: the three shapes the Fe output uses, counted in the trace.

    copy: `PUSH MLOAD PUSH MSTORE`, one word moved from memory to memory; fill: `DUP1 PUSH MSTORE`,
    one word of an array initialised; decode loop: a loop whose body starts
    `JUMPDEST DUP1 CALLDATALOAD .. MSTORE`, one word of calldata per turn, costed from one visit of
    its head to the next (so the last turn of each loop is not counted).
    """
    ops = [step[1] for step in steps]
    n = len(steps)
    taken = set()
    found = []
    for label, shape in (("word copies memory to memory", ("PUSH", "MLOAD", "PUSH", "MSTORE")),
                         ("word fills", ("DUP1", "PUSH", "MSTORE"))):
        count = gas = 0
        i = 0
        while i + len(shape) <= n:
            if i not in taken and all(ops[i + k].startswith(shape[k]) for k in range(len(shape))):
                count += 1
                gas += sum(step[2] for step in steps[i:i + len(shape)])
                taken.update(range(i, i + len(shape)))
                i += len(shape)
            else:
                i += 1
        found.append("%s x %d = %d gas" % (label, count, gas))
    heads = sorted({steps[i - 2][0] for i in range(2, n - 2) if ops[i] == "CALLDATALOAD" and ops[i - 1] == "DUP1"
                    and ops[i - 2] == "JUMPDEST" and "MSTORE" in ops[i + 1:i + 3]})
    turns = gas = 0
    for head in heads:
        visits = [i for i in range(n) if steps[i][0] == head]
        costs = [sum(step[2] for step in steps[a:b]) for a, b in zip(visits, visits[1:]) if b - a < 60]
        turns += len(costs)
        gas += sum(costs)
    found.append("calldata decode loops: %d, %d turns = %d gas" % (len(heads), turns, gas))
    count = Counter(ops)
    print("%s, arrays: %s; MSTORE %d, MLOAD %d, CALLDATALOAD %d, CALLDATACOPY %d, MCOPY %d"
          % (name, "; ".join(found), count["MSTORE"], count["MLOAD"], count["CALLDATALOAD"],
             count["CALLDATACOPY"], count["MCOPY"]))


def phases(name, steps, code):
    """Gas before the first MUL, from the first to the last MUL, and after; masks apart in each."""
    _, masked = find_masks(steps, immediates(code))
    first = next(i for i, step in enumerate(steps) if step[1] == "MUL")
    last = max(i for i, step in enumerate(steps) if step[1] == "MUL")
    parts = []
    for a, b in ((0, first), (first, last + 1), (last + 1, len(steps))):
        total = sum(steps[i][2] for i in range(a, b))
        mask = sum(steps[i][2] for i in range(a, b) if i in masked)
        parts.append("%6d (masks %5d, the rest %6d)" % (total, mask, total - mask))
    print("%-20s before the first MUL %s; first to last MUL %s; after %s; MUL executed %d times"
          % (name, parts[0], parts[1], parts[2], sum(1 for s in steps if s[1] == "MUL")))


def loops(name, steps):
    """The loops that run before the first MUL: each maximal run of steps whose pc is executed six
    times or more in that part of the trace and that costs 1,000 gas or more, with the first opcodes
    of its body."""
    first = next(i for i, step in enumerate(steps) if step[1] == "MUL")
    before = steps[:first]
    visits = Counter(step[0] for step in before)
    straight = sum(step[2] for step in before if visits[step[0]] < 6)
    print("%s: %d gas before the first MUL, %d of it in code that runs fewer than six times" % (name, len(before) and sum(
        step[2] for step in before), straight))
    i = 0
    while i < len(before):
        if visits[before[i][0]] < 6:
            i += 1
            continue
        j = i
        while j < len(before) and visits[before[j][0]] >= 6:
            j += 1
        body, seen = [], set()
        for pc, op, _ in before[i:j]:
            if pc in seen:
                break
            seen.add(pc)
            body.append(op)
        gas = sum(step[2] for step in before[i:j])
        if gas >= 1000:  # the smaller ones are single calls of a helper that is called six times
            print("    loop: %5d gas, body of %3d opcodes starting %s" % (gas, len(body), " ".join(body[:8])))
        i = j


def static_breakdown(name, code):
    """Bytes of runtime code by kind, and how many of each mask sequence it contains."""
    pushes = immediates(code)
    push_bytes = sum(1 + width for width, _ in pushes.values())
    ops = Counter()
    pc = 0
    listing = []
    while pc < len(code):
        listing.append((pc, code[pc]))
        ops[code[pc]] += 1
        pc += 1 + (pushes[pc][0] if pc in pushes else 0)
    masks = Counter()
    for i, (pc, op) in enumerate(listing):
        following = listing[i + 1][1] if i + 1 < len(listing) else None
        if pc in pushes:
            width, value = pushes[pc]
            if following == 0x16 and value == (1 << (8 * width)) - 1 and width > 1:
                masks["u%d mask, %d bytes each" % (8 * width, width + 2)] += 1
            if following == 0x0B and width == 1 and value < 31:
                masks["i%d sign extension, 3 bytes each" % (8 * (value + 1))] += 1
        if op == 0x5F and [o for _, o in listing[i + 1:i + 5]] == [0x19, 0x60, 0x1C, 0x16]:
            masks["u%d mask, 6 bytes each" % (256 - pushes[listing[i + 2][0]][1])] += 1
    mask_bytes = sum(count * int(kind.split(", ")[1].split()[0]) for kind, count in masks.items())
    print("%-34s %6d bytes: PUSH with data %5d, DUP/SWAP/POP %5d, JUMPDEST %4d, JUMP/JUMPI %4d, MUL %3d, "
          "SLOAD %2d, SSTORE %2d; masks %d bytes (%s)"
          % (name, len(code), push_bytes, sum(ops[o] for o in range(0x80, 0xA0)) + ops[0x50], ops[0x5B],
             ops[0x56] + ops[0x57], ops[0x02], ops[0x54], ops[0x55], mask_bytes,
             "; ".join("%s x %d" % pair for pair in sorted(masks.items())) or "none"))


def build_fe(work, name, kernel=None):
    source = os.path.join(work, name)
    os.makedirs(source)
    shutil.copy(os.path.join(ROOT, "fe.toml"), source)
    shutil.copytree(os.path.join(ROOT, "ingots"), os.path.join(source, "ingots"))
    if kernel:
        shutil.copy(kernel, os.path.join(source, "ingots", "tensile", "src", "dot.fe"))
    out = os.path.join(work, "out-" + name)
    run("fe", "build", "--color", "never", "--out-dir", out, "--ingot", "onchain", ".", cwd=source)
    return out


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def lines(*command):
    return run(sys.executable, *command).splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18731)
    parser.add_argument("--keep", help="directory to save the traces and build outputs in")
    args = parser.parse_args()
    started = time.time()
    work = tempfile.mkdtemp(prefix="gas-gap-")
    chain = None
    try:
        kept = build_fe(work, "kept")
        unrolled = build_fe(work, "unrolled", os.path.join(ROOT, "tests", "kernel_forms", "i256_unrolled.fe"))
        loop = build_fe(work, "loop", os.path.join(ROOT, "tests", "kernel_forms", "i256_loop.fe"))
        stage = os.path.join(work, "evm")
        os.makedirs(stage)
        shutil.copytree(os.path.join(ROOT, "evm", "src"), os.path.join(stage, "src"))
        shutil.copy(os.path.join(ROOT, "evm", "foundry.toml"), stage)
        run("forge", "build", "--color", "never", cwd=stage)

        def twin(contract):
            with open(os.path.join(stage, "out", contract + ".sol", contract + ".json"), encoding="utf-8") as f:
                return json.load(f)["bytecode"]["object"][2:]

        print("# gas_gap: %s; %s; solc and settings of evm/foundry.toml"
              % (run("fe", "--version"), run("forge", "--version").splitlines()[0]))
        chain = Chain(args.port)
        traces = {}

        def traced(label, address, signature, *call_args, value=0):
            tx, used, execution = chain.send(address, signature, *call_args, value=value)
            steps = chain.trace(tx)
            traces[label] = {"gasUsed": used, "execution": execution, "steps": steps}
            return steps, chain.code(address), execution

        # One layer 1 neuron: docs/numbers.md section 3, first table, and section 5.
        n1 = lines(os.path.join(HERE, "neuron_args.py"), "1", "0", "0")
        neuron_sig = "neuron1(uint256[25],uint256[25],int32)"
        sides = {}
        for label, creation, sig in (("Fe, as shipped", read(os.path.join(kept, "NeuronCheck.bin")), neuron_sig),
                                     ("Fe, i256_unrolled", read(os.path.join(unrolled, "NeuronCheck.bin")), neuron_sig),
                                     ("Solidity, 32 lanes", twin("NeuronTwin"),
                                      "neuron1Unrolled(uint256[25],uint256[25],int32)")):
            address = chain.deploy(creation)
            got = chain.call(address, sig + "(int8)", n1[0], n1[1], n1[2])
            if got != n1[3]:
                sys.exit("%s returned %s, the reference says %s" % (label, got, n1[3]))
            sides[label] = traced("neuron1: " + label, address, sig, n1[0], n1[1], n1[2])
        compare("Layer 1 neuron: Fe as shipped against the best Solidity", "Fe", sides["Fe, as shipped"],
                "Solidity", sides["Solidity, 32 lanes"])
        compare("Layer 1 neuron: Fe with the i256_unrolled kernel against the best Solidity", "Fe",
                sides["Fe, i256_unrolled"], "Solidity", sides["Solidity, 32 lanes"])
        print("\n## Layer 1 neuron by phase (split at the first and the last MUL executed)")
        for label in sides:
            phases(label, sides[label][0], sides[label][1])

        # The arbiter: docs/numbers.md section 3, second table. The sequence of tools/arbiter_gas.sh.
        tree = json.loads(read(os.path.join(ROOT, "build", "tree.json")))
        constructor = run("cast", "abi-encode", "constructor(uint256,uint256,uint256)", tree["root"], str(BOND),
                          str(WINDOW))[2:]
        claims = {kind: lines(os.path.join(HERE, "claim_args.py"), "claim", *extra)
                  for kind, extra in (("honest", ()), ("hidden", ("--mutate", "hidden:0")),
                                      ("logit", ("--mutate", "logit:0")), ("class", ("--mutate", "class")))}
        p1 = lines(os.path.join(HERE, "claim_args.py"), "proof", "1", "0")
        p2 = lines(os.path.join(HERE, "claim_args.py"), "proof", "2", "0")
        claim_sig = "claim(uint256[25],uint256[1],int32[10],uint8)"
        arbiters = {"Fe": chain.deploy(read(os.path.join(kept, "Arbiter.bin")), constructor),
                    "Solidity": chain.deploy(twin("ArbiterTwin"), constructor)}
        calls = {}
        for side, address in arbiters.items():
            calls[side] = {}
            calls[side]["claim"] = traced(side + " claim", address, claim_sig, *claims["honest"], value=BOND)
            for kind in ("hidden", "logit", "class"):
                chain.send(address, claim_sig, *claims[kind], value=BOND)
            calls[side]["challengeNeuron1"] = traced(
                side + " challengeNeuron1", address,
                "challengeNeuron1(uint256,uint256,uint256[25],uint256[1],uint256[25],int32,bytes32[])",
                "1", "0", claims["hidden"][0], claims["hidden"][1], p1[0], p1[1], p1[2])
            calls[side]["challengeNeuron2"] = traced(
                side + " challengeNeuron2", address,
                "challengeNeuron2(uint256,uint256,uint256[1],int32[10],uint256[1],int32,bytes32[])",
                "2", "0", claims["logit"][1], claims["logit"][2], p2[0], p2[1], p2[2])
            calls[side]["challengeClass"] = traced(side + " challengeClass", address,
                                                   "challengeClass(uint256,int32[10])", "3", claims["class"][2])
        chain.rpc("evm_increaseTime", WINDOW + 1)
        chain.rpc("evm_mine")
        for side, address in arbiters.items():
            calls[side]["finalize"] = traced(side + " finalize", address, "finalize(uint256)", "0")
            statuses = [chain.call(address, "claimInfo(uint256)(address,uint256,uint256,uint8,uint8)",
                                   str(i)).splitlines()[-1] for i in range(4)]
            if statuses != ["3", "2", "2", "2"]:
                sys.exit("%s arbiter: claim statuses %s, expected final, slashed, slashed, slashed" % (side, statuses))
        for call in ("claim", "challengeNeuron1", "challengeNeuron2", "challengeClass", "finalize"):
            compare("Arbiter.%s" % call, "Fe", calls["Fe"][call], "Solidity", calls["Solidity"][call])
        print("\n## Arbiter.challengeNeuron1 by phase (split at the first and the last MUL executed)")
        for side in calls:
            phases(side, calls[side]["challengeNeuron1"][0], calls[side]["challengeNeuron1"][1])
        print("\n## Loops before the first MUL")
        loops("NeuronCheck.neuron1, Fe as shipped", sides["Fe, as shipped"][0])
        loops("NeuronTwin.neuron1Unrolled", sides["Solidity, 32 lanes"][0])
        for side in calls:
            loops("Arbiter.challengeNeuron1, " + side, calls[side]["challengeNeuron1"][0])

        print("\n## Runtime bytecode, static (the Fe files are `fe build` output, before immutables are appended)")
        for form, out in (("as shipped", kept), ("i256_unrolled", unrolled), ("i256_loop", loop)):
            for contract in ("NeuronCheck", "Arbiter"):
                static_breakdown("%s (Fe, %s)" % (contract, form),
                                 bytes.fromhex(read(os.path.join(out, contract + ".runtime.bin"))))
        static_breakdown("NeuronTwin (Solidity)", sides["Solidity, 32 lanes"][1])
        static_breakdown("ArbiterTwin (Solidity)", calls["Solidity"]["claim"][1])
        print("\ngas_gap: every trace summed to its execution gas; %.1f s" % (time.time() - started))
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
            with open(os.path.join(args.keep, "traces.json"), "w", encoding="utf-8") as f:
                json.dump(traces, f)
            for label, (_, code, _) in list(sides.items()) + [("Arbiter " + s, calls[s]["claim"]) for s in calls]:
                with open(os.path.join(args.keep, label.replace(" ", "_").replace(",", "") + ".runtime.hex"), "w") as f:
                    f.write(code.hex())
    finally:
        if chain:
            chain.stop()
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
