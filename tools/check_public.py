"""Checks the files git would publish, before anything is pushed. Prints paths and counts only.

usage: python tools/check_public.py [SECRETS_DIR]

- No value from SECRETS_DIR (default ../../.secrets) appears in a public file. Each secret file is cut into
  tokens: e-mail addresses, URLs, keys, long opaque strings. A match prints the file, never the value.
- No private planning file is in the list.
- No em dash in a text file a person reads.
"""

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIVATE = ["CLAUDE.md", "HANDOFF.md", "claude-yolo.bat", "docs/ideas.md", "docs/decision.md",
           "docs/master-rubric.md", "docs/research/", "private/"]
SKIP_BINARY = (".bin", ".png", ".jpg", ".mp4", ".wav", ".onnx")


def public_files():
    with tempfile.TemporaryDirectory() as git_dir:
        base = ["git", "--git-dir", git_dir, "--work-tree", ROOT]
        subprocess.run(base + ["init", "-q"], check=True)
        subprocess.run(base + ["add", "-A"], check=True, capture_output=True)
        out = subprocess.run(base + ["ls-files"], check=True, capture_output=True, text=True).stdout
    return out.split("\n")[:-1]


def tokens(folder):
    found = set()
    for name in os.listdir(folder):
        if name not in ("fe-challenge-registration.md", "fe-zuliprc", "fe-sepolia-key.txt"):
            continue
        with open(os.path.join(folder, name), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # The testnet account's address is public (deploy/sepolia.jsonl); its key is not.
        text = "\n".join(line for line in text.split("\n") if not line.startswith(("address=", "#")))
        found |= set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text))
        found |= set(re.findall(r"https?://[^\s)>\]]+", text))
        found |= set(re.findall(r"\b[A-Za-z0-9_-]{16,}\b", text))
    # Public addresses that may appear in the secret file's prose are not secret.
    return {t for t in found if not re.match(r"https://(fe-lang|github\.com|docs\.google\.com/forms/d/e/1FAIpQLSehDU)", t)}


def main():
    secrets = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "..", ".secrets")
    files = public_files()
    problems = 0
    for path in files:
        if any(path == p or path.startswith(p) for p in PRIVATE if p.endswith("/") or p == path):
            print("PRIVATE FILE IN THE LIST:", path)
            problems += 1
    values = tokens(secrets) if os.path.isdir(secrets) else set()
    for path in files:
        if path.endswith(SKIP_BINARY):
            continue
        with open(os.path.join(ROOT, path), "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="ignore")
        hits = sum(1 for v in values if v in text)
        if hits:
            print("SECRET VALUE in %s: %d match(es)" % (path, hits))
            problems += 1
        if path.endswith((".md", ".txt", ".html", ".yml", ".fe", ".sol", ".py", ".sh", ".mjs", ".js", ".css", ".json")) \
                and chr(0x2014) in text and not path.endswith("dist/tensile-demo.html"):
            print("em dash in", path)
            problems += 1
    size = sum(os.path.getsize(os.path.join(ROOT, p)) for p in files)
    print("%d public files, %.1f MB, %d secret values checked, %d problem(s)" % (len(files), size / 1e6, len(values), problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
