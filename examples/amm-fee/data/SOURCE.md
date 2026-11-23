# Where `ticks.csv` comes from

| | |
|---|---|
| Pool | Uniswap v3 USDC/WETH 0.05% on Ethereum mainnet, `0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640` |
| Value | the pool's current tick: the second return value of `slot0()`, read from the state at the end of a block |
| Blocks | 23,850,000 to 26,478,000, every 50th: 52,561 observations |
| Time covered | 2025-11-21 21:18:59 UTC to 2026-11-23 13:01:23 UTC (block timestamps), about 367 days |
| Spacing | 600 seconds between observations when no slot is missed; the longest gap in the series is 900 seconds |
| File | one tick per line, in block order: line `i` (from 0) is block 23,850,000 + 50 `i`. 367,927 bytes |
| sha256 | `00af771601a2365266adbb2266f940df96b2275f6b038e113520e3771d70515c` |
| Retrieved | 2026-11-24, with `../fetch_data.py` |

## How it was obtained

`fetch_data.py` (standard library, no API key, no login, nothing sent but the JSON-RPC request)
made one archive `eth_call` per block to Multicall3
(`0xcA11bde05977b3631167028862bE2a173976CA11`), which returned the block number, the block
timestamp and the pool's `slot0()` in one answer. The block number in every answer was checked
against the block asked for. Three public endpoints were used in turn:

- `https://eth.drpc.org`
- `https://rpc.mevblocker.io`
- `https://eth-mainnet.public.blastapi.io`

The first run stalled after 40,475 rows when the endpoints began to refuse or time out; a second
run with smaller batches fetched the remaining 12,086. Which endpoint served which row was not
recorded.

Afterwards 200 evenly spaced rows were fetched again from each endpoint by itself
(`fetch_data.py --verify 200`): all 200 from `rpc.mevblocker.io` and all 200 from
`eth-mainnet.public.blastapi.io` were identical to the stored rows (block, timestamp, tick);
`eth.drpc.org` answered HTTP 400 at that time and was not compared. That is a spot check of 200
rows against two providers, not a check of every row, and the endpoints are not independent
sources of truth: they are nodes of the same chain.

The timestamps are not committed (only their range and spacing above); the model does not use
them. Chain data is public; this file is a derived series of 52,561 integers.

## What the series is not

- It is the tick at the end of every 50th block, not a time-weighted average. One trade in the
  observed block moves it.
- 0.39% of the 10-minute moves are larger than 127 ticks and are clipped by the input rule
  (`../dataset.py`); the largest is 1,036 ticks. 4.1% are zero.
- It is one pool over one year. Nothing here says the model carries over to another pool or
  another year.
