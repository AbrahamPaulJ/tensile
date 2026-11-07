# 2026-11-06: demo page, numbers, reports, README, video

Abraham asked for the remaining plan items and a demo video, overnight, while asleep. The two open
decisions (burn half; a second entry) were left unanswered, so neither was acted on: the contract did not
change, and nothing was registered, filed, committed or pushed.

What was built, in order: the page (`demo/`), one numbers script (`tools/m4_numbers.sh`, `docs/numbers.md`), a
Solidity arbiter twin, compiler report drafts, README, CI, a clean-home run, the video.

Worth remembering:

- **An in-browser EVM was the right call.** The page has no model in JavaScript. It also gave a second EVM
  implementation to compare with anvil, and that comparison found the next item within minutes.
- **Our own docs were wrong.** `classify` was 1,896,043 on anvil and 1,903,649 in HANDOFF and the plan, which
  had just been "re-verified". The check scripts measure gas without asserting it, so milestone 3's cheaper
  validation changed it silently. I first suspected EthereumJS; an opcode-by-opcode comparison cleared it.
  The figures are now generated, and the demo's test fails on drift.
- **A defect came off the upstream list when actually re-tested.** "Native tests fail on an ingot with a
  contract" did not reproduce on a minimal file. Reports should always be re-run before they are sent.
- **`.gitignore` had a trap:** `out/` also ignored `ref/out`, which every differential test reads, so a fresh
  clone could not have run the checks. Now `/out/`, and `ref/out` is tracked.
- **`fe fmt --check` from the root, as in the starter's CI, fails with "No .fe files found"** on this workspace.
  It needs `ingots` as an argument.

Details: `docs/fe-notes.md` milestone 4.
