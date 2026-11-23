"""Fetch the price history of one Uniswap v3 pool from public Ethereum RPC endpoints.

Usage:  python -I examples/amm-fee/fetch_data.py --out DIR [--start B] [--end B] [--step N]

For every block B = start, start + step, ..., end it makes one archive `eth_call` at block B to
Multicall3 (0xcA11bde05977b3631167028862bE2a173976CA11), which returns, from the state at the end
of that block:

    the block number          (checked against B)
    the block timestamp       Multicall3.getCurrentBlockTimestamp()
    the pool's current tick   pool.slot0(), second return value

Standard library only. No API key, no login; the only thing sent is the JSON-RPC request. The
rows are appended to DIR/samples.csv as they arrive (block,timestamp,tick), so an interrupted run
continues where it stopped. `--verify N` re-fetches N evenly spaced rows from each endpoint by
itself and compares. `--series FILE` writes the ticks alone, one per line in block order: that is
data/ticks.csv. This script is not part of check.sh: the series it produced is committed in data/
(see data/SOURCE.md) and CI never downloads anything.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request

POOL = "88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"       # Uniswap v3 USDC/WETH 0.05%, mainnet
MULTICALL3 = "ca11bde05977b3631167028862be2a173976ca11"
ENDPOINTS = ["https://eth.drpc.org", "https://rpc.mevblocker.io", "https://eth-mainnet.public.blastapi.io"]


def word(value):
    return "%064x" % value


def call_data():
    """Multicall3.aggregate([(multicall3, getCurrentBlockTimestamp()), (pool, slot0())])."""
    def element(address, selector):
        return address.rjust(64, "0") + word(0x40) + word(4) + selector.ljust(64, "0")
    first = element(MULTICALL3, "0f28c97d")
    second = element(POOL, "3850c7bd")
    return "0x252dba42" + word(0x20) + word(2) + word(0x40) + word(0x40 + len(first) // 2) + first + second


def signed(value, bits):
    return value - (1 << bits) if value >> (bits - 1) else value


def decode(result, block):
    """(timestamp, tick) from the return data of aggregate: (uint256 blockNumber, bytes[] returnData)."""
    raw = bytes.fromhex(result[2:])
    words = [int.from_bytes(raw[i:i + 32], "big") for i in range(0, len(raw), 32)]
    if words[0] != block:
        raise ValueError("block %d: the node answered for block %d" % (block, words[0]))
    base = words[1] // 32 + 1                      # first word after the array length
    if words[base - 1] != 2:
        raise ValueError("block %d: expected two results" % block)
    first = base + words[base] // 32               # bytes element: length, data
    second = base + words[base + 1] // 32
    if words[first] != 32 or words[second] != 224:
        raise ValueError("block %d: unexpected result sizes" % block)
    timestamp = words[first + 1]
    tick = signed(words[second + 2] & ((1 << 24) - 1), 24)
    return timestamp, tick


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json", "user-agent": "fetch-data/1"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read())


def fetch(blocks, data, first_endpoint, only=False):
    """One JSON-RPC batch; tries the endpoints in turn, with a pause after each failure. With
    `only`, every attempt goes to the one endpoint (for --verify)."""
    payload = [{"jsonrpc": "2.0", "id": i, "method": "eth_call",
                "params": [{"to": "0x" + MULTICALL3, "data": data}, hex(block)]} for i, block in enumerate(blocks)]
    last = None
    for attempt in range(3 if only else 12):
        url = ENDPOINTS[(first_endpoint + (0 if only else attempt)) % len(ENDPOINTS)]
        try:
            answer = post(url, payload)
            if isinstance(answer, dict) and len(blocks) > 1:
                # The endpoint refuses batches (or this one): ask block by block.
                answer = [post(url, item) for item in payload]
            if not isinstance(answer, list) or len(answer) != len(blocks):
                raise ValueError("batch answer: %s" % str(answer)[:200])
            by_id = {item["id"]: item for item in answer}
            rows = []
            for i, block in enumerate(blocks):
                if "result" not in by_id[i]:
                    raise ValueError(str(by_id[i])[:200])
                rows.append((block,) + decode(by_id[i]["result"], block))
            return rows
        except Exception as error:  # noqa: BLE001  (any failure: next endpoint)
            last = error
            time.sleep(min(2 + attempt * 2, 15))
    if only:
        raise ConnectionError(str(last)[:120])
    raise SystemExit("fetch_data: blocks %d..%d failed on every endpoint: %s" % (blocks[0], blocks[-1], last))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=23_850_000)
    parser.add_argument("--end", type=int, default=26_478_000)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--verify", type=int, default=0)
    parser.add_argument("--series", default=None)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "samples.csv")
    data = call_data()

    have = {}
    if os.path.exists(path):
        for line in open(path, encoding="ascii"):
            block, timestamp, tick = line.split(",")
            have[int(block)] = (int(timestamp), int(tick))

    if args.series:
        blocks = list(range(args.start, args.end + 1, args.step))
        missing = [b for b in blocks if b not in have]
        if missing:
            raise SystemExit("fetch_data: %d blocks are missing, the first is %d" % (len(missing), missing[0]))
        text = "".join("%d\n" % have[b][1] for b in blocks)
        with open(args.series, "wb") as f:
            f.write(text.encode("ascii"))
        stamps = [have[b][0] for b in blocks]
        if any(later <= earlier for earlier, later in zip(stamps, stamps[1:])):
            raise SystemExit("fetch_data: the timestamps do not rise with the block number")
        gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
        print("%d ticks, blocks %d to %d, timestamps %d to %d, seconds between observations: least %d, most %d"
              % (len(blocks), blocks[0], blocks[-1], stamps[0], stamps[-1], min(gaps), max(gaps)))
        print("sha256 %s  %s (%d bytes)" % (hashlib.sha256(text.encode("ascii")).hexdigest(), args.series, len(text)))
        return 0

    if args.verify:
        blocks = sorted(have)
        picked = [blocks[i * (len(blocks) - 1) // (args.verify - 1)] for i in range(args.verify)]
        bad = 0
        for endpoint in range(len(ENDPOINTS)):
            rows = []
            try:
                for i in range(0, len(picked), args.batch):
                    rows += fetch(picked[i:i + args.batch], data, endpoint, only=True)
            except ConnectionError as error:
                print("verify: %s did not answer (%s)" % (ENDPOINTS[endpoint], error))
                continue
            wrong = sum(1 for block, timestamp, tick in rows if have[block] != (timestamp, tick))
            bad += wrong
            print("verify: %d rows re-fetched from %s, %d differ" % (len(rows), ENDPOINTS[endpoint], wrong))
        return 1 if bad else 0

    todo = [b for b in range(args.start, args.end + 1, args.step) if b not in have]
    print("%d rows on disk, %d to fetch" % (len(have), len(todo)))
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    t0 = time.time()
    done = 0
    with open(path, "a", encoding="ascii", newline="\n") as out, \
            concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        futures = [pool.submit(fetch, batch, data, i % len(ENDPOINTS)) for i, batch in enumerate(batches)]
        for future in concurrent.futures.as_completed(futures):
            for block, timestamp, tick in future.result():
                out.write("%d,%d,%d\n" % (block, timestamp, tick))
            out.flush()
            done += 1
            if done % 40 == 0:
                print("%d of %d batches, %.0f s" % (done, len(batches), time.time() - t0), flush=True)
    print("done: %d rows fetched in %.0f s" % (len(todo), time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
