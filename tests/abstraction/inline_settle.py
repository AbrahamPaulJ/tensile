"""Rewrites a scratch copy of ingots/onchain/src/arbiter.fe with the lifecycle rule inlined, for
tools/m4_numbers.sh: no `dispute::step`, no `Action`, `Result`, `Transition`, `Payee` or `Status`
values, just the same checks in the same order, written into `settle` over status codes. The gas
difference from the real Arbiter is what the pure state machine costs at run time.

usage: python3 inline_settle.py <path to the scratch arbiter.fe>
"""

import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    source = f.read()

# Every call site passes two plain flags instead of an `Action`.
source, n = re.subn(r"action: Action::Challenge\((.+?)\),\n", r"fraud: \1,\n                finalizing: false,\n", source)
assert n == 3, n
source, n = re.subn(r"action: Action::Finalize,\n", "fraud: false,\n                finalizing: true,\n", source)
assert n == 1, n

start = source.index("fn settle(")
source = source[:start] + """fn settle(
    _ id: u256,
    _ record: Record,
    now: Timestamp,
    fraud: bool,
    finalizing: bool,
    challenger: Address,
)
uses (claims: mut StorageMap<u256, Record>, credits: mut StorageMap<Address, Wei>)
{
    if record.status != 1 {
        revert_error(ClaimNotOpen { id })
    }
    let late = now.as_seconds() >= record.deadline.as_seconds()
    if finalizing {
        if !late {
            revert_error(TooEarly { deadline: record.deadline })
        }
        claims[id].status = 3
        credits[record.claimant] = credits[record.claimant] + record.bond
    } else {
        if late {
            revert_error(TooLate { deadline: record.deadline })
        }
        if !fraud {
            revert_error(NoFraud {})
        }
        claims[id].status = 2
        credits[challenger] = credits[challenger] + record.bond
    }
}
"""
source = source.replace("use core::result::Result\n", "")
source = source.replace("use dispute::{Action, Payee, Rejection, Status, Transition, step}\n", "")
source = source.replace("status: Status::Open.code(),", "status: 1,")
source = source.replace("if record.status != Status::Final.code() {", "if record.status != 3 {")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(source)
