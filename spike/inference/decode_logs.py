# Reads fe test --show-logs output on stdin, prints "tag N: gas" for GasReport events.
import sys, re
names = {1: "pure checked (unpacked consts)", 2: "pure unchecked (unpacked consts)", 3: "CALL Layer.infer checked",
         4: "CALL Layer.inferUnchecked", 5: "pure packed unchecked", 6: "CALL LayerPacked.infer",
         7: "pure packed, i256 accumulator", 8: "CALL LayerSplit.infer (2 weight contracts)", 9: "CALL LayerRows.infer (row per STATICCALL)"}
macs = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for line in sys.stdin:
    m = re.search(r"topics: \[0x[0-9a-f]+, 0x([0-9a-f]+)\], data: 0x([0-9a-f]+)", line)
    if m:
        tag, gas = int(m.group(1), 16), int(m.group(2)[:64], 16)
        per = (" (%.1f gas per multiply-add over %d)" % (gas / macs, macs)) if macs else ""
        print("tag %d %-40s %10d gas%s" % (tag, names.get(tag, "?"), gas, per))
    elif re.search(r"PASS|FAIL|ERROR|error|test result", line):
        print(line.rstrip())
