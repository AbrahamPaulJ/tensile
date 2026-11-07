"""Capture the compiler's real diagnostics for a few compile-fail cases, for the demo page.

Usage: python3 tools/demo_diagnostics.py OUT.json

Each case is checked the way tools/check_compile_fail.sh does it: as the only source file of a
scratch ingot that depends on `tensile`. The source of the case and the diagnostic text are kept. A
case that compiles is an error: the page must never show a refusal the compiler did not make.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.expanduser("~/.fe/bin/fe")

CASES = [
    ("Mixing fixed-point scales", "scale_mix.fe"),
    ("Wiring layers of the wrong shape", "layer_chain_mismatch.fe"),
    ("A bias at the wrong scale", "bias_wrong_scale.fe"),
    ("A model whose sum could overflow", "overflow_proof_fails.fe"),
]

WORKSPACE = """[workspace]
name = "compile-fail"
version = "0.1.0"
members = [{ path = "ingots/tensile", name = "tensile" }, { path = "ingots/case", name = "case" }]
"""

INGOT = """[ingot]
name = "case"
version = "0.1.0"

[dependencies]
tensile = true
"""


def main():
    out = []
    with tempfile.TemporaryDirectory() as work:
        shutil.copytree(os.path.join(ROOT, "ingots", "tensile"),
                        os.path.join(work, "ingots", "tensile"))
        os.makedirs(os.path.join(work, "ingots", "case", "src"))
        with open(os.path.join(work, "fe.toml"), "w", encoding="utf-8") as f:
            f.write(WORKSPACE)
        with open(os.path.join(work, "ingots", "case", "fe.toml"), "w", encoding="utf-8") as f:
            f.write(INGOT)
        target = os.path.join(work, "ingots", "case", "src", "lib.fe")
        for title, name in CASES:
            rel = "tests/compile_fail/" + name
            shutil.copyfile(os.path.join(ROOT, rel), target)
            run = subprocess.run([FE, "check", "--color", "never", "ingots/case"], cwd=work,
                                 capture_output=True, text=True)
            if run.returncode == 0:
                print("demo_diagnostics: %s compiled, expected a refusal" % rel, file=sys.stderr)
                return 1
            text = (run.stdout + run.stderr).replace(work + "/", "").strip()
            with open(target, "r", encoding="utf-8") as f:
                source = f.read()
            out.append({"title": title, "file": rel, "source": source, "diagnostic": text})
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f)
    print("captured %d diagnostics" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
