# The kernel as a library: three contracts that are not neural networks

A separate Fe workspace that uses `ingots/tensile` the way an outside project would, by path. The
guide is `docs/LIBRARY.md`; this folder is its evidence.

| Ingot | What |
|---|---|
| `qmath` | add, subtract, multiply (scales add) and compare `tensile::Q<F>` values |
| `ema` | `TickEma`: an exponential moving average of a Uniswap v3 tick, overflow proved for every tick |
| `score` | `Scorecard`: a linear credit score, 16 int8 points packed into the code, through `tensile::neuron` |
| `curve` | `RateCurve`: a kinked borrow-rate curve, the scale of every product in its type |
| `plain` | the same three over plain integers, and the curve with `core::fixed::Wad`, for the gas comparison |
| `deployed` | EVM tests: every contract against its pure functions and against its twin |

| File | What |
|---|---|
| `check.sh` | everything: fmt, check, tests on both backends, compile-fail cases, build, gas on anvil |
| `compile_fail/`, `check_compile_fail.sh` | six scale and overflow mistakes that compile in Solidity and not here, and a control |
| `gas.sh`, `runs/gas.json` | gas of each contract and its twin on the same calls |
| `walkthrough.sh` | `docs/LIBRARY.md` section 1, from an empty directory |
| `step_limit.sh` | how many weights one compile-time overflow proof can cover, on 26.5.1 and 26.7.0 |

```
wsl -e bash /mnt/c/.../lib-examples/check.sh                  (Fe 26.5.1, Foundry, Python 3)
wsl -e bash /mnt/c/.../tools/check_fe267_lib.sh               (the same on Fe 26.7.0, no anvil)
```

The points, rates and smoothing factor are made up for the examples. Not audited.
