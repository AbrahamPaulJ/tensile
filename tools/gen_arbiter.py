"""Generate the Fe source of the dispute arbiter for any dense Tensile model.

Usage:
    python tools/gen_arbiter.py [--model ref/out/model.json] [--vectors ref/out] --out DIR

Writes into DIR, the `src` directory of a contract ingot that depends on `tensile`, `dispute` and
the model ingot `tools/gen_fe.py` writes from the same model file:

    arbiter.fe        the `Arbiter` contract and its message
    record.fe         the claim record: one hash per committed vector
    signals.fe        events and errors
    commitment.fe     the neuron leaf, with tests against the generated root
    arbiter_tests.fe  contract tests on reference sample 0 (only with --vectors)
    lib.fe            the ingot root (only with --lib; the shipped ingot has more contracts)

Why a generator and not one generic contract: a `msg` variant needs concrete ABI types and a
selector string, so the number of layers and their widths cannot be const generics of a handler.
The claim carries one packed vector per layer boundary, the record one hash per vector, and there
is one `challengeNeuronK` handler per layer plus `challengeClass`.

For the shipped 784-32-10 model the output, after `fe fmt`, is byte for byte the hand-written
`ingots/onchain/src/{arbiter,record,signals,commitment,arbiter_tests}.fe`;
`tools/check_arbiter_gen.sh` checks it. That is what ties the Foundry suite, the Solidity twin
and the mutation run of the hand-written arbiter to this generator. With one hidden layer the
vector is called `hidden`; with more they are `hidden1`, `hidden2`, ... in layer order.

The output is a pure function of the input files. It is laid out simply: run `fe fmt` on the
ingot afterwards. Standard library only.
"""

import argparse
import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gen_tree import commitment  # noqa: E402

I1 = "    "
I2 = I1 * 2
I3 = I1 * 3

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve"]


class ModelError(ValueError):
    pass


def words(n):
    """Number of 256-bit words that hold `n` packed int8 values."""
    return (n + 31) // 32


def doc(text, indent="", prefix="/// ", width=80):
    """A doc comment wrapped at `width` columns. Two comments of the hand-written sources are
    wrapped at 78, so their callers say so: the identity check compares comments too."""
    lines = textwrap.wrap(text, width - len(indent) - len(prefix), break_long_words=False,
                          break_on_hyphens=False)
    return [indent + prefix + line for line in lines]


def and_list(items):
    """"a", "a and b", "a, b and c"."""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


class Shape:
    """What the templates need to know about a model: sizes, scales and names."""

    def __init__(self, model):
        self.model = model
        layers = model["layers"]
        self.count = len(layers)
        if self.count < 2:
            raise ModelError("the arbiter needs at least two layers (one hidden vector)")
        self.n_in = layers[0]["in"]
        self.outs = [layer["out"] for layer in layers]
        self.classes = self.outs[-1]
        if self.classes > 255:
            raise ModelError("the class is a uint8: at most 255 outputs")
        if self.classes < 2:
            raise ModelError("the last layer needs at least two outputs")
        # Fractional bits of every hidden vector (the type parameter of its `Activations`).
        self.frac = []
        f = model["input_frac_bits"]
        for layer in layers[:-1]:
            f = f + layer["weight_frac_bits"] - layer["shift"]
            self.frac.append(f)
        self.hidden_count = self.count - 1
        self.single = self.hidden_count == 1

    # Names. `k` is the 1-based layer whose output the vector is.
    def hid(self, k):
        return "hidden" if self.single else "hidden%d" % k

    def hid_type(self, k):
        return "Hidden" if self.single else "Hidden%d" % k

    def hid_const(self, k):
        return "HIDDEN" if self.single else "HIDDEN%d" % k

    def hid_words(self, k):
        return words(self.outs[k - 1])

    def hiddens(self):
        return range(1, self.count)

    def layers(self):
        return range(1, self.count + 1)

    def layer_in(self, k):
        """Name of the vector layer `k` reads."""
        return "x0" if k == 1 else self.hid(k - 1)

    def layer_in_words(self, k):
        return words(self.n_in) if k == 1 else self.hid_words(k - 1)

    def layer_out(self, k):
        return "logits" if k == self.count else self.hid(k)

    def layer_out_fe(self, k):
        if k == self.count:
            return "[i32; %d]" % self.classes
        return "[u256; %d]" % self.hid_words(k)

    def layer_out_sol(self, k):
        if k == self.count:
            return "int32[%d]" % self.classes
        return "uint256[%d]" % self.hid_words(k)

    def vector_fields(self):
        """(name, Fe type, Solidity type) of the claim's vectors, in order."""
        out = [("x0", "[u256; %d]" % words(self.n_in), "uint256[%d]" % words(self.n_in))]
        for k in self.hiddens():
            out.append((self.hid(k), self.layer_out_fe(k), self.layer_out_sol(k)))
        out.append(("logits", "[i32; %d]" % self.classes, "int32[%d]" % self.classes))
        return out

    def model_use(self, extra_first=(), extra_last=()):
        names = list(extra_first) + ["layer%d" % k for k in self.layers()] + list(extra_last)
        return "use model::{%s}" % ", ".join(names)


# ------------------------------------------------------------------ arbiter.fe


ARBITER_HEAD = """\
//! `Arbiter`: optimistic verification of inference claims
//! (`docs/spec-dispute.md`).
//!
//! Anyone posts a claimed inference with a bond; anyone can prove it wrong
//! before the deadline by re-running one neuron or the argmax on-chain, with
//! the model ingot's `neuron_from_words`: the code the forward pass runs. The
//! lifecycle rule is `dispute::step`, a pure function, and `dispute::shares`
//! says how a bond is shared out: all of it back on finalising; on a slash,
//! half to the challenger and the rest burned, so that a false claim costs
//! at least half its bond even when its own claimant slashes it. This contract
//! reads the clock, loads and stores the claim, and moves credits. Credits are pulled
//! with `withdraw`, the only handler that sends ETH, which its effect
//! signature shows (`mut call`).
use core::result::Result
use dispute::{Action, Payee, Rejection, Status, Transition, shares, step}
use ingot::commitment::leaf
use ingot::record::{Record, dynamic, logits_hash}
use ingot::signals::{BondTooSmall, ClaimNotOpen, ClassOutOfRange, ClassSlashed, Claimed, Finalized, InputMismatch, MalformedHidden, MalformedInput, NeuronOutOfRange, NeuronSlashed, NoFraud, NotCommitted, NotFinal, NothingToWithdraw, OutputMismatch, TooEarly, TooLate, TransferFailed, UnknownClaim, Withdrawn}
@MODEL_USE@
use std::abi::{Bytes32, DynArray}
use std::evm::{Address, Call, Ctx, Duration, Log, StorageMap, Timestamp, Wei, keccak_words, merkle, wei}
use tensile::Activations
"""

ARBITER_MSG_TAIL = """\
    /// Proves claim `id` wrong at the class: the argmax of the committed
    /// logits (lowest index on a tie) is not the claimed class.
    #[selector = sol("challengeClass(uint256,int32[@CLASSES@])")]
    ChallengeClass { id: u256, logits: [i32; @CLASSES@] },
    /// Closes claim `id` at or after its deadline and credits the whole bond
    /// back to the claimant.
    #[selector = sol("finalize(uint256)")]
    Finalize { id: u256 },
    /// Sends the caller's credits to the caller. Returns the amount.
    #[selector = sol("withdraw()")]
    Withdraw -> Wei,
    /// The class of a `Final` claim. Reverts for any other claim.
    #[selector = sol("result(uint256)")]
    ResultOf { id: u256 } -> u8,
    /// Number of claims posted so far; claims are numbered from 0.
    #[selector = sol("claimCount()")]
    ClaimCount -> u256,
    /// Claimant, bond, deadline, class and status code (1 open, 2 slashed,
    /// 3 final) of claim `id`.
    #[selector = sol("claimInfo(uint256)")]
    ClaimInfo { id: u256 } -> (Address, Wei, Timestamp, u8, u8),
@HASHES_DOC@
    #[selector = sol("claimHashes(uint256)")]
    ClaimHashes { id: u256 } -> (@HASH_TUPLE@),
    /// What `account` can withdraw.
    #[selector = sol("creditOf(address)")]
    CreditOf { account: Address } -> Wei,
    /// The model root the arbiter was deployed with.
    #[selector = sol("modelRoot()")]
    ModelRoot -> u256,
    /// The bond a claim must attach.
    #[selector = sol("bond()")]
    Bond -> Wei,
    /// The total burned by slashes: half of every slashed bond, rounded up.
    /// It stays in the contract and nobody can withdraw it.
    #[selector = sol("burned()")]
    Burned -> Wei,
    /// How long a claim stays challengeable.
    #[selector = sol("window()")]
    Window -> Duration,
}

/// The arbiter for one model: deployed with the model root, the bond and the
/// challenge window. Each handler names the effects it needs: only `Withdraw`
/// can move ETH (`mut call`); the views touch nothing but storage.
pub contract Arbiter uses (ctx: Ctx, call: mut Call, log: mut Log) {
    root: u256,
    bond: Wei,
    window: Duration,
    mut count: u256,
    mut claims: StorageMap<u256, Record>,
    mut credits: StorageMap<Address, Wei>,
    mut burned: Wei,

    init(model_root: u256, bond_required: Wei, challenge_window: Duration) uses (mut root, mut bond, mut window) {
        root = model_root
        bond = bond_required
        window = challenge_window
    }

    recv ArbiterMsg {
"""

ARBITER_HANDLERS_TAIL = """\
        ChallengeClass { id, logits } uses (ctx, count, mut claims, mut credits, mut burned, mut log) {
            let record = load(id)
            if logits_hash(logits) != record.logits_hash {
                revert_error(OutputMismatch {})
            }
            let fraud = Logits::from_raw(logits).argmax() != record.class as usize
            let challenger = ctx.caller()
            settle(id, record, now: ctx.timestamp(), action: Action::Challenge(fraud), challenger)
            log.emit(ClassSlashed { id, challenger })
        }

        Finalize { id } uses (ctx, count, mut claims, mut credits, mut burned, mut log) {
            let record = load(id)
            settle(id, record, now: ctx.timestamp(), action: Action::Finalize, challenger: ctx.caller())
            log.emit(Finalized { id, class: record.class })
        }

        Withdraw -> Wei uses (ctx, mut call, mut credits, mut log) {
            let account = ctx.caller()
            let amount = credits[account]
            if amount == Wei::zero() {
                revert_error(NothingToWithdraw {})
            }
            // State first, then the transfer; a failed transfer reverts both.
            credits[account] = Wei::zero()
            if !call.send_value(addr: account, gas: ctx.gas(), value: amount).success() {
                revert_error(TransferFailed {})
            }
            log.emit(Withdrawn { account, amount })
            amount
        }

        ResultOf { id } -> u8 uses (count, claims) {
            let record = load(id)
            if record.status != Status::Final.code() {
                revert_error(NotFinal { id })
            }
            record.class
        }

        ClaimCount -> u256 uses (count) {
            count
        }

        ClaimInfo { id } -> (Address, Wei, Timestamp, u8, u8) uses (count, claims) {
            let record = load(id)
            (record.claimant, record.bond, record.deadline, record.class, record.status)
        }

        ClaimHashes { id } -> (@HASH_TUPLE@) uses (count, claims) {
            let record = load(id)
            (@HASH_VALUES@)
        }

        CreditOf { account } -> Wei uses (credits) {
            credits[account]
        }

        ModelRoot -> u256 uses (root) {
            root
        }

        Bond -> Wei uses (bond) {
            bond
        }

        Burned -> Wei uses (burned) {
            burned
        }

        Window -> Duration uses (window) {
            window
        }
    }
}

/// Claim `id`, or `UnknownClaim`.
fn load(_ id: u256) -> Record uses (count: u256, claims: StorageMap<u256, Record>) {
    if id >= count {
        revert_error(UnknownClaim { id })
    }
    claims[id]
}

/// Applies the lifecycle rule to claim `id`: stores the new status, credits
/// the payee its share of the bond and adds the rest to `burned`, or reverts
/// with the rule's reason. `challenger` is the caller; it is paid only when
/// the rule says so. `burned` grows only on a slash; finalising burns nothing.
fn settle(_ id: u256, _ record: Record, now: Timestamp, action: Action, challenger: Address) uses (claims: mut StorageMap<u256, Record>, credits: mut StorageMap<Address, Wei>, burned: mut Wei) {
    let claim = dispute::Claim {
        status: Status::from_code(record.status),
        deadline: record.deadline.as_seconds(),
    }
    match step(claim, now: now.as_seconds(), action) {
        Result::Err(why) => {
            match why {
                Rejection::NotOpen => revert_error(ClaimNotOpen { id }),
                Rejection::TooLate => revert_error(TooLate { deadline: record.deadline }),
                Rejection::TooEarly => revert_error(TooEarly { deadline: record.deadline }),
                Rejection::NoFraud => revert_error(NoFraud {}),
            }
        },
        Result::Ok(transition) => {
            claims[id].status = transition.status.code()
            let payee = match transition.bond_to {
                Payee::Challenger => challenger,
                Payee::Claimant => record.claimant,
            }
            let (paid, burnt) = shares(record.bond.as_wei(), transition)
            credits[payee] = credits[payee] + wei(paid)
            if burnt != 0 {
                burned = burned + wei(burnt)
            }
        },
    }
}
"""


def fill(template, **values):
    for key, value in values.items():
        template = template.replace("@%s@" % key, str(value))
    if "@" in template:
        raise AssertionError("unfilled placeholder near: %s" % template[template.index("@"):][:40])
    return template


def challenge_fields(shape, k):
    """(name, Fe type, Solidity type) of the fields of `ChallengeNeuronK`."""
    row = shape.layer_in_words(k)
    return [
        ("id", "u256", "uint256"),
        ("j", "u256", "uint256"),
        (shape.layer_in(k), "[u256; %d]" % row, "uint256[%d]" % row),
        (shape.layer_out(k), shape.layer_out_fe(k), shape.layer_out_sol(k)),
        ("row", "[u256; %d]" % row, "uint256[%d]" % row),
        ("bias", "i32", "int32"),
        ("proof", "DynArray<Bytes32>", "bytes32[]"),
    ]


def challenge_signature(shape, k):
    return "challengeNeuron%d(%s)" % (k, ",".join(sol for _, _, sol in challenge_fields(shape, k)))


def claim_signature(shape):
    return "claim(%s,uint8)" % ",".join(sol for _, _, sol in shape.vector_fields())


def arbiter_source(shape):
    s = shape
    out = [fill(ARBITER_HEAD, MODEL_USE=s.model_use(["Input", "Logits"])).rstrip("\n"), ""]
    for k in s.hiddens():
        out += ["/// The output of layer %d." % k,
                "type %s = Activations<%d, %d>" % (s.hid_type(k), s.outs[k - 1], s.frac[k - 1]), ""]
    for k in s.hiddens():
        out += ["/// Number of hidden neurons (layer %d outputs)." % k,
                "const %s: u256 = %d" % (s.hid_const(k), s.outs[k - 1]), ""]
    out += ["/// Number of classes (layer %d outputs)." % s.count,
            "const CLASSES: u8 = %d" % s.classes, ""]

    # The message.
    hidden_names = ["`%s`" % s.hid(k) for k in s.hiddens()]
    out += [
        "/// The arbiter's interface. Vectors are packed words (`docs/spec.md` section",
        "/// 7); `uint8` for a class, `int32` for a bias or logit, `bytes32[]` for a",
        "/// Merkle proof (siblings from the leaf up, as OpenZeppelin's `MerkleProof`).",
        "pub msg ArbiterMsg {",
    ]
    out += doc("Posts a claim with at least `bond()` attached: for input `x0`, the hidden "
               "activations are %s, the logits `logits`, the class `class`. Returns the claim "
               "number." % and_list(hidden_names), I1, width=78)
    out += [
        I1 + '#[selector = sol("%s")]' % claim_signature(s),
        I1 + "Claim { %s, class: u8 } -> u256," % ", ".join(
            "%s: %s" % (name, fe) for name, fe, _ in s.vector_fields()),
    ]
    for k in s.layers():
        if k == 1 and s.single:
            text = ("Proves claim `id` wrong at hidden neuron `j`: `x0` and `hidden` must be the "
                    "committed vectors, `row`, `bias` and `proof` a committed neuron of layer 1. "
                    "Reverts unless the neuron recomputes to another value.")
        elif k == s.count and s.single:
            text = "Proves claim `id` wrong at logit `j`, from the committed hidden vector."
        elif k == s.count:
            text = ("Proves claim `id` wrong at logit `j`, from the committed vector `%s`."
                    % s.layer_in(k))
        else:
            text = ("Proves claim `id` wrong at neuron `j` of layer %d: `%s` and `%s` must be the "
                    "committed vectors, `row`, `bias` and `proof` a committed neuron of layer %d. "
                    "Reverts unless the neuron recomputes to another value."
                    % (k, s.layer_in(k), s.layer_out(k), k))
        out += doc(text, I1)
        out += [
            I1 + '#[selector = sol("%s")]' % challenge_signature(s, k),
            I1 + "ChallengeNeuron%d { %s }," % (k, ", ".join(
                "%s: %s" % (name, fe) for name, fe, _ in challenge_fields(s, k))),
        ]
    if s.single:
        hashes_doc = [I1 + "/// The committed hashes of claim `id`: input words, hidden words, logits."]
    else:
        hashes_doc = doc("The committed hashes of claim `id`: input words, the words of each "
                         "hidden vector in layer order, logits.", I1)
    hash_tuple = ", ".join(["u256"] * (s.count + 1))
    hash_values = ", ".join(["record.x0_hash"] + ["record.%s_hash" % s.hid(k) for k in s.hiddens()]
                            + ["record.logits_hash"])
    out.append(fill(ARBITER_MSG_TAIL, CLASSES=s.classes, HASHES_DOC="\n".join(hashes_doc),
                    HASH_TUPLE=hash_tuple).rstrip("\n"))

    # The handlers.
    names = ", ".join(name for name, _, _ in s.vector_fields())
    out += [
        I2 + "#[payable]",
        I2 + "Claim { %s, class } -> u256 uses (ctx, bond, window, mut count, mut claims, mut log) {" % names,
        I3 + "let paid = ctx.value()",
        I3 + "if paid < bond {",
        I3 + I1 + "revert_error(BondTooSmall { sent: paid, required: bond })",
        I3 + "}",
        I3 + "if !Input::well_formed(x0) {",
        I3 + I1 + "revert_error(MalformedInput {})",
        I3 + "}",
    ]
    for k in s.hiddens():
        out += [
            I3 + "if !%s::well_formed(%s) {" % (s.hid_type(k), s.hid(k)),
            I3 + I1 + "revert_error(MalformedHidden {})",
            I3 + "}",
        ]
    out += [
        I3 + "if class >= CLASSES {",
        I3 + I1 + "revert_error(ClassOutOfRange { class })",
        I3 + "}",
        I3 + "let id = count",
        I3 + "count = id + 1",
        I3 + "let claimant = ctx.caller()",
        I3 + "let deadline = ctx.timestamp() + window",
        I3 + "claims[id] = Record {",
        I3 + I1 + "claimant,",
        I3 + I1 + "class,",
        I3 + I1 + "status: Status::Open.code(),",
        I3 + I1 + "bond: paid,",
        I3 + I1 + "deadline,",
        I3 + I1 + "x0_hash: keccak_words(x0),",
    ]
    for k in s.hiddens():
        out.append(I3 + I1 + "%s_hash: keccak_words(%s)," % (s.hid(k), s.hid(k)))
    out += [
        I3 + I1 + "logits_hash: logits_hash(logits),",
        I3 + "}",
        I3 + "log.emit(Claimed { id, claimant, %s, class, bond: paid, deadline })" % ", ".join(
            "%s: dynamic(%s)" % (name, name) for name, _, _ in s.vector_fields()),
        I3 + "id",
        I2 + "}",
        "",
    ]
    for k in s.layers():
        last = k == s.count
        vin, vout = s.layer_in(k), s.layer_out(k)
        limit = "CLASSES as u256" if last else s.hid_const(k)
        out_hash = "logits_hash(logits)" if last else "keccak_words(%s)" % vout
        claimed = "logits[j as usize]" if last else "%s::from_words(%s).get(j as usize)" % (
            s.hid_type(k), vout)
        out += [
            I2 + "ChallengeNeuron%d { id, j, %s, %s, row, bias, proof } uses (ctx, root, count, mut claims, "
            "mut credits, mut burned, mut log) {" % (k, vin, vout),
            I3 + "let record = load(id)",
            I3 + "if j >= %s {" % limit,
            I3 + I1 + "revert_error(NeuronOutOfRange { neuron: j })",
            I3 + "}",
            I3 + "if keccak_words(%s) != record.%s_hash {" % (vin, vin),
            I3 + I1 + "revert_error(InputMismatch {})",
            I3 + "}",
            I3 + "if %s != record.%s_hash {" % (out_hash, vout),
            I3 + I1 + "revert_error(OutputMismatch {})",
            I3 + "}",
            I3 + "if !merkle::verify(proof, root, leaf: leaf(layer: %d, neuron: j, bias, row)) {" % k,
            I3 + I1 + "revert_error(NotCommitted {})",
            I3 + "}",
            I3 + "let recomputed = layer%d::neuron_from_words(row, x: %s, bias)" % (k, vin),
            I3 + "let claimed = %s" % claimed,
            I3 + "let challenger = ctx.caller()",
            I3 + "settle(id, record, now: ctx.timestamp(), action: Action::Challenge(recomputed != claimed), "
            "challenger)",
            I3 + "log.emit(NeuronSlashed { id, challenger, layer: %d, neuron: j })" % k,
            I2 + "}",
            "",
        ]
    out.append(fill(ARBITER_HANDLERS_TAIL, HASH_TUPLE=hash_tuple, HASH_VALUES=hash_values))
    return "\n".join(out)


# ------------------------------------------------------------------- record.fe


RECORD_TAIL = """\
impl Copy for Record {}

/// The commitment to an int32 vector: `keccak256` of its elements as 32-byte
/// two's complement words, the ABI encoding of `int32[N]` (`docs/spec.md`
/// section 7). Solidity: `keccak256(abi.encode(logits))`.
pub fn logits_hash<const N: usize>(_ logits: [i32; N]) -> u256 {
    var words: [u256; N] = [0; N]
    for j in 0 .. N {
        words[j] = logits[j].to_word()
    }
    keccak_words(words)
}

/// `values` as a dynamic ABI array, the form an event field takes.
pub fn dynamic<T, const N: usize>(_ values: [T; N]) -> DynArray<T> where T: Encode<Sol> + AbiSize + Copy {
    var array = unsafe {
        ptr::alloc_array<T>(N)
    }
    for i in 0 .. N {
        array[i] = values[i]
    }
    DynArray::from_slice(array.as_slice())
}

#[test]
fn logits_hash_sign_extends_every_element() {
    // keccak256(abi.encode(int32(-1), int32(2))): the first word is all ones.
    let expected = keccak_words([u256::max(), 2])
    assert!(logits_hash([-1, 2]) == expected)
    assert!(logits_hash([1, 2]) != expected)
}
"""


def record_source(shape):
    s = shape
    slots = 4 + s.count + 1
    slot_word = NUMBER_WORDS[slots] if slots < len(NUMBER_WORDS) else str(slots)
    out = [
        "//! What the arbiter stores about a claim, and the commitments it stores.",
        "use core::abi::{AbiSize, Encode}",
        "use core::marker::Copy",
        "use core::num::IntWord",
        "use core::ptr",
        "use std::abi::{DynArray, Sol}",
        "use std::evm::{Address, Timestamp, Wei, keccak_words}",
        "",
        "/// One claim in storage (`docs/spec-dispute.md` section 3). The vectors",
        "/// themselves are only in the `Claimed` event and the calldata; a challenge",
        "/// must present vectors that hash to these commitments.",
        "///",
    ]
    if s.single:
        out += [
            "/// Fields narrower than a word share a slot, in declaration order: the class",
            "/// and status bytes do. `Address` is a whole word in Fe, so a record takes",
            "/// seven slots (measured on anvil, `docs/fe-notes.md` M3.6).",
        ]
    else:
        out += doc("Fields narrower than a word share a slot, in declaration order: the class and "
                   "status bytes do. `Address` is a whole word in Fe, so a record takes %s slots: "
                   "four for the claimant, the two bytes, the bond and the deadline, and one per "
                   "committed vector (`docs/fe-notes.md` M3.6 measured the seven slots of a "
                   "two-layer model on anvil)." % slot_word)
    out += [
        "pub struct Record {",
        I1 + "/// Who posted the claim and receives the bond back on `Final`.",
        I1 + "pub claimant: Address,",
        I1 + "/// The claimed class.",
        I1 + "pub class: u8,",
        I1 + "/// `dispute::Status::code()`: 1 open, 2 slashed, 3 final. 0 means no claim.",
        I1 + "pub status: u8,",
        I1 + "/// The ETH attached to the claim, credited exactly once when it ends.",
        I1 + "pub bond: Wei,",
        I1 + "/// From this time on the claim can be finalised and no longer challenged.",
        I1 + "pub deadline: Timestamp,",
        I1 + "/// `keccak256` of the packed input words.",
        I1 + "pub x0_hash: u256,",
    ]
    for k in s.hiddens():
        if s.single:
            out.append(I1 + "/// `keccak256` of the packed hidden activation words.")
        else:
            out.append(I1 + "/// `keccak256` of the packed activation words of hidden layer %d." % k)
        out.append(I1 + "pub %s_hash: u256," % s.hid(k))
    out += [
        I1 + "/// `keccak256` of the logits, each sign-extended to a 32-byte word.",
        I1 + "pub logits_hash: u256,",
        "}",
        "",
        RECORD_TAIL,
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ signals.fe


SIGNALS_TAIL = """\
    pub class: u8,
    pub bond: Wei,
    pub deadline: Timestamp,
}

/// A neuron challenge proved a claim wrong: neuron `neuron` of layer `layer`
/// recomputed to another value. Half the bond is credited to `challenger`;
/// the rest is burned (`dispute::shares`).
#[event]
pub struct NeuronSlashed {
    #[indexed]
    pub id: u256,
    #[indexed]
    pub challenger: Address,
    pub layer: u256,
    pub neuron: u256,
}

/// A class challenge proved a claim wrong: the claimed class is not the
/// argmax of the committed logits. Half the bond is credited to `challenger`;
/// the rest is burned (`dispute::shares`).
#[event]
pub struct ClassSlashed {
    #[indexed]
    pub id: u256,
    #[indexed]
    pub challenger: Address,
}

/// A claim passed its deadline unchallenged. The bond is credited back to the
/// claimant and `result(id)` now answers.
#[event]
pub struct Finalized {
    #[indexed]
    pub id: u256,
    pub class: u8,
}

/// `account` withdrew its credits.
#[event]
pub struct Withdrawn {
    #[indexed]
    pub account: Address,
    pub amount: Wei,
}

/// Less ETH was attached to a claim than the bond.
#[error]
pub struct BondTooSmall { pub sent: Wei, pub required: Wei }

/// The input vector of a claim has a byte above 127 or non-zero padding.
#[error]
pub struct MalformedInput {}

/// @MALFORMED_HIDDEN@
#[error]
pub struct MalformedHidden {}

/// The claimed class is not an output of the network.
#[error]
pub struct ClassOutOfRange { pub class: u8 }

/// No claim has this number.
#[error]
pub struct UnknownClaim { pub id: u256 }

/// The claim is already `Slashed` or `Final`.
#[error]
pub struct ClaimNotOpen { pub id: u256 }

/// A challenge at or after the deadline.
#[error]
pub struct TooLate { pub deadline: Timestamp }

/// A finalisation before the deadline.
#[error]
pub struct TooEarly { pub deadline: Timestamp }

/// The challenge recomputed exactly the committed value: the claim stands.
#[error]
pub struct NoFraud {}

/// The neuron index is not in the layer.
#[error]
pub struct NeuronOutOfRange { pub neuron: u256 }

/// The vector presented as the layer's input does not hash to the committed one.
#[error]
pub struct InputMismatch {}

/// The vector presented as the layer's output does not hash to the committed one.
#[error]
pub struct OutputMismatch {}

/// The row and bias do not verify against the model root at this position.
#[error]
pub struct NotCommitted {}

/// `result` was asked for a claim that is not `Final`.
#[error]
pub struct NotFinal { pub id: u256 }

/// `withdraw` with no credit.
#[error]
pub struct NothingToWithdraw {}

/// The ETH transfer of a withdrawal failed; the credit is left in place.
#[error]
pub struct TransferFailed {}
"""


def signals_source(shape):
    s = shape
    fields = s.vector_fields()
    sol_types = ["`uint256[]`"] * (len(fields) - 1) + ["`int32[]`"]
    lengths = [str(words(s.n_in))] + [str(s.hid_words(k)) for k in s.hiddens()] + [str(s.classes)]
    out = [
        "//! The events the arbiter emits and the errors it reverts with.",
        "//!",
        "//! Every error is a Solidity custom error (`revert_error`), so a caller can",
        "//! tell a rejected challenge (`NoFraud`) from a malformed one (`InputMismatch`,",
        "//! `NotCommitted`) and a client can decode the reason.",
        "use std::abi::DynArray",
        "use std::evm::{Address, Timestamp, Wei}",
        "",
    ]
    out += doc("A claim was posted. Carries the full vectors, so a watcher needs nothing but the "
               "log to audit it (`docs/spec-dispute.md` section 3). The vectors are dynamic arrays "
               "(%s) of the lengths the `claim` message fixes (%s), because an event field cannot "
               "be a fixed-size array (`tests/defects/event_fixed_array.fe`)."
               % (", ".join(sol_types), and_list(lengths)), width=78)
    out += [
        "#[event]",
        "pub struct Claimed {",
        I1 + "#[indexed]",
        I1 + "pub id: u256,",
        I1 + "#[indexed]",
        I1 + "pub claimant: Address,",
    ]
    for name, _, _ in fields:
        out.append(I1 + "pub %s: DynArray<%s>," % (name, "i32" if name == "logits" else "u256"))
    if s.single:
        malformed = "The hidden vector of a claim has a byte above 127 or non-zero padding."
    else:
        malformed = "A hidden vector of a claim has a byte above 127 or non-zero padding."
    out.append(fill(SIGNALS_TAIL, MALFORMED_HIDDEN=malformed))
    return "\n".join(out)


# --------------------------------------------------------------- commitment.fe


COMMITMENT_HEAD = """\
//! The neuron leaf of the model commitment (`docs/spec-dispute.md` section 2).
//!
//! `leaf(l, j) = keccak256("tensile.neuron.v0" || u256(l) || u256(j) || i256(bias) || row words)`.
//! The arbiter rebuilds the leaf from the row and bias a challenger sends and
//! verifies it against the committed root with `std::evm::merkle::verify`, so
//! only a committed row can be used in a challenge, and only at its own
//! position. `tools/gen_tree.py` is the off-chain twin of this function.
use core::num::IntWord
use core::ptr::{self, MemSlice}
@MODEL_USE@
use std::evm::{merkle, packed}

/// The domain tag at the start of every leaf preimage.
pub const TAG: String<17> = "tensile.neuron.v0"

/// The leaf of neuron `neuron` of layer `layer` (1-based) with `bias` and the
/// packed weight row `row` (`N` words). The bias is sign-extended to 256 bits.
pub fn leaf<const N: usize>(layer: u256, neuron: u256, bias: i32, row: [u256; N]) -> u256 {
    var preimage = packed::Packed::with_capacity(bytes: 17 + 3 * 32 + 32 * N)
    preimage.string(TAG)
    preimage.u256(layer)
    preimage.u256(neuron)
    preimage.u256(bias.to_word())
    for word in row {
        preimage.u256(word)
    }
    preimage.keccak256()
}

/// A proof given as an array, in the form `merkle::verify` reads.
fn nodes<const N: usize>(_ proof: [u256; N]) -> ref MemSlice<u256> {
    var array = unsafe {
        ptr::alloc_array<u256>(N)
    }
    for i in 0 .. N {
        array[i] = proof[i]
    }
    array.as_slice()
}
"""

COMMITMENT_TAIL = """\
#[test]
fn a_row_does_not_verify_at_another_position() {
    let root = commitment::MODEL_ROOT
    // Row 0 of layer 1 presented as neuron 1, as layer 2, and with another bias.
    let moved = leaf(layer: 1, neuron: 1, bias: layer1::bias(0).raw(), row: layer1::row(0).words())
    let other_layer = leaf(layer: 2, neuron: 0, bias: layer1::bias(0).raw(), row: layer1::row(0).words())
    let other_bias = leaf(layer: 1, neuron: 0, bias: layer1::bias(0).raw() + 1, row: layer1::row(0).words())
    assert!(!merkle::verify(proof: nodes(commitment::PROOF_1_0), root, leaf: moved))
    assert!(!merkle::verify(proof: nodes(commitment::PROOF_1_0), root, leaf: other_layer))
    assert!(!merkle::verify(proof: nodes(commitment::PROOF_1_0), root, leaf: other_bias))
    // The right leaf with the proof of another neuron.
    assert!(!merkle::verify(proof: nodes(commitment::PROOF_1_@LAST1@), root, leaf: leaf1(0)))
}
"""


def commitment_fe_source(shape):
    s = shape
    if s.outs[0] < 2:
        raise ModelError("layer 1 needs at least two neurons")
    out = [fill(COMMITMENT_HEAD, MODEL_USE=s.model_use(["commitment"])).rstrip("\n"), ""]
    for k in s.layers():
        if k == 1:
            out.append("/// The leaf of layer 1, neuron `j`, from the model ingot's own row and bias.")
        else:
            out.append("/// The leaf of layer %d, neuron `j`." % k)
        out += [
            "fn leaf%d(_ j: usize) -> u256 {" % k,
            I1 + "leaf(layer: %d, neuron: j as u256, bias: layer%d::bias(j).raw(), row: layer%d::row(j).words())"
            % (k, k, k),
            "}",
            "",
        ]
    ends = [(k, j) for k in s.layers() for j in sorted({0, s.outs[k - 1] - 1})]
    out += ["#[test]", "fn leaves_equal_the_generator() {"]
    for k, j in ends:
        out.append(I1 + "assert!(leaf%d(%d) == commitment::LEAF_%d_%d)" % (k, j, k, j))
    out += ["}", "", "#[test]", "fn proofs_verify_against_the_root() {",
            I1 + "let root = commitment::MODEL_ROOT"]
    for k, j in ends:
        out.append(I1 + "assert!(merkle::verify(proof: nodes(commitment::PROOF_%d_%d), root, leaf: leaf%d(%d)))"
                   % (k, j, k, j))
    out += ["}", "", fill(COMMITMENT_TAIL, LAST1=s.outs[0] - 1)]
    return "\n".join(out)


# ------------------------------------------------------------ arbiter_tests.fe


TESTS_HEAD = """\
//! Unit tests of `Arbiter` on reference sample 0: an honest claim survives
//! every challenge; a claim with one wrong hidden value, one wrong logit or a
//! wrong class is slashed by the challenge the auditor would name and by no
//! other; malformed claims, wrong proofs and vectors that do not match the
//! commitments are rejected with their own errors; finalising and
//! withdrawing move the bond. Each failed call is checked for its error
//! selector, so a revert for the wrong reason fails the test.
//!
//! The proofs are built here from the model's own rows, by the same rule as
//! `tools/gen_tree.py`; the root must come out as the generated `MODEL_ROOT`.
use core::abi::Encode
use core::error::ErrorVariant
use core::marker::Copy
use core::message::MsgVariant
use core::ptr::MemVec
use ingot::{Arbiter, ArbiterMsg, leaf}
use ingot::signals::{BondTooSmall, ClaimNotOpen, ClassOutOfRange, InputMismatch, MalformedHidden, MalformedInput, NoFraud, NotCommitted, NotFinal, NothingToWithdraw, OutputMismatch, TooEarly, TooLate, TransferFailed, UnknownClaim}
@MODEL_USE@
use std::abi::{Bytes32, DynArray, FixedBytes, Sol}
use std::evm::{Address, Call, CallOutcome, Ctx, Duration, Evm, Timestamp, Wei, is_error, seconds, wei}
use tensile::Activations

/// Gas given to every call: the Osaka limit of one transaction.
const GAS: u256 = 16777216

/// The bond the test arbiters require.
const BOND: Wei = wei(1_000_000)

/// What a slash credits the challenger: half the bond. The rest is burned.
const REWARD: Wei = wei(500_000)
const BURN: Wei = wei(500_000)

/// A window no test can reach the end of.
const LONG: Duration = seconds(1_000_000)

/// Status codes (`dispute::Status::code`).
const OPEN: u8 = 1
const SLASHED: u8 = 2
const FINAL: u8 = 3

/// Number of leaves of the neuron tree: @LEAVES_DOC@.
const LEAVES: usize = @LEAVES@

/// The longest proof in a tree of @LEAVES@ leaves.
const DEPTH: usize = @DEPTH@

/// The vectors of one claim.
struct Vectors { @VECTOR_FIELDS@, class: u8 }

impl Copy for Vectors {}

/// The honest claim for reference sample 0.
fn honest() -> Vectors {
    Vectors {
        x0: reference::X0.words(),
@HONEST_HIDDEN@
        logits: reference::LOGITS0.values(),
        class: reference::CLASS0.downcast_truncate(),
    }
}
"""

TESTS_MIDDLE = """\
/// The honest claim with logit `j` changed by one.
fn wrong_logit(_ j: usize) -> Vectors {
    var vectors = honest()
    vectors.logits[j] += 1
    vectors
}

/// The honest claim with another class.
fn wrong_class() -> Vectors {
    var vectors = honest()
    vectors.class = (vectors.class + 1) % @CLASSES@
    vectors
}

fn deploy(window: Duration) -> Address uses (evm: mut Evm) {
    evm.create2<Arbiter>(value: Wei::zero(), args: (commitment::MODEL_ROOT, BOND, window), salt: 0)
}

/// Posts `vectors` with the bond and returns the claim number.
fn post(arbiter: Address, _ vectors: Vectors) -> u256 uses (evm: mut Evm) {
    evm.call(addr: arbiter, gas: GAS, value: BOND, message: claim_message(vectors))
}

fn claim_message(_ vectors: Vectors) -> ArbiterMsg::Claim {
    ArbiterMsg::Claim { @VECTOR_COPY@, class: vectors.class }
}

/// Every leaf of the neuron tree, in tree order.
fn leaves() -> [u256; LEAVES] {
    var out: [u256; LEAVES] = [0; LEAVES]
@LEAF_LOOPS@
    out
}

/// The proof of leaf `index` and its length, by the rule of
/// `docs/spec-dispute.md` section 2 (sorted pairs, the odd last node promoted).
/// The root the walk reaches must be the generated one.
fn proof_for(_ index: usize) -> ([u256; DEPTH], usize) {
    var level = leaves()
    var count = LEAVES
    var position = index
    var proof: [u256; DEPTH] = [0; DEPTH]
    var len: usize = 0
    while count > 1 {
        var next: usize = 0
        var k: usize = 0
        while k < count {
            if k + 1 == count {
                level[next] = level[k]
            } else {
                if position == k {
                    proof[len] = level[k + 1]
                    len += 1
                } else if position == k + 1 {
                    proof[len] = level[k]
                    len += 1
                }
                level[next] = std::evm::merkle::hash_pair_sorted(level[k], level[k + 1])
            }
            if position == k || position == k + 1 {
                position = next
            }
            next += 1
            k += 2
        }
        count = next
    }
    assert!(level[0] == commitment::MODEL_ROOT)
    (proof, len)
}

/// The first `len` nodes of `proof` as the `bytes32[]` a challenge carries.
fn as_bytes32(_ proof: [u256; DEPTH], len: usize) -> DynArray<Bytes32> {
    var nodes = MemVec::with_capacity(len)
    for i in 0 .. len {
        nodes.push(FixedBytes<32> { val: proof[i] })
    }
    DynArray::from_slice(nodes.as_slice())
}
"""

TESTS_HELPERS = """\
fn challenge_class(arbiter: Address, id: u256, _ vectors: Vectors) -> CallOutcome uses (evm: mut Evm) {
    attempt(arbiter, ArbiterMsg::ChallengeClass { id, logits: vectors.logits })
}

fn finalize(arbiter: Address, id: u256) -> CallOutcome uses (evm: mut Evm) {
    attempt(arbiter, ArbiterMsg::Finalize { id })
}

fn status(arbiter: Address, id: u256) -> u8 uses (evm: mut Evm) {
    let info: (Address, Wei, Timestamp, u8, u8) = evm.call(addr: arbiter, gas: GAS, value: Wei::zero(), message: ArbiterMsg::ClaimInfo { id })
    info.4
}

fn credit(arbiter: Address, account: Address) -> Wei uses (evm: mut Evm) {
    evm.call(addr: arbiter, gas: GAS, value: Wei::zero(), message: ArbiterMsg::CreditOf { account })
}

fn burned(arbiter: Address) -> Wei uses (evm: mut Evm) {
    evm.call(addr: arbiter, gas: GAS, value: Wei::zero(), message: ArbiterMsg::Burned {})
}

/// After one slash by this test's account: half the bond is its credit, the
/// other half is burned, and the arbiter still holds all of it.
fn slashed_once(arbiter: Address) -> bool uses (evm: mut Evm) {
    credit(arbiter, account: evm.address()) == REWARD && burned(arbiter) == BURN && evm.balance(arbiter) == BOND
}

fn claim_count(arbiter: Address) -> u256 uses (evm: mut Evm) {
    evm.call(addr: arbiter, gas: GAS, value: Wei::zero(), message: ArbiterMsg::ClaimCount {})
}

/// Whether the call reverted with error `E`.
fn failed_with<E>(_ outcome: CallOutcome) -> bool where E: ErrorVariant<Sol> {
    !outcome.success() && is_error<E>(outcome.returndata())
}
"""

TESTS_TAIL = """\
#[test(balance = 10_000_000)]
fn finalising_after_the_window_returns_the_bond() uses (evm: mut Evm) {
    // A window of zero: the deadline is now, so a challenge is already too
    // late and finalising is allowed.
    let arbiter = deploy(window: seconds(0))
    let id = post(arbiter, wrong_class())
    assert!(failed_with<TooLate>(challenge_class(arbiter, id, wrong_class())))
    assert!(finalize(arbiter, id).success())
    assert!(status(arbiter, id) == FINAL)
    let result: u8 = evm.call(addr: arbiter, gas: GAS, value: Wei::zero(), message: ArbiterMsg::ResultOf { id })
    assert!(result == wrong_class().class)
    assert!(failed_with<ClaimNotOpen>(finalize(arbiter, id)))
    assert!(failed_with<ClaimNotOpen>(challenge_class(arbiter, id, wrong_class())))
    assert!(credit(arbiter, account: evm.address()) == BOND)
    assert!(burned(arbiter) == Wei::zero())
    assert!(evm.balance(arbiter) == BOND)
    // The test runner's own address cannot receive ETH, so this withdrawal
    // fails in the transfer: the credit and the balance must be untouched.
    assert!(failed_with<TransferFailed>(attempt(arbiter, ArbiterMsg::Withdraw {})))
    assert!(credit(arbiter, account: evm.address()) == BOND)
    assert!(evm.balance(arbiter) == BOND)
}

#[test(balance = 10_000_000)]
fn withdrawing_pays_the_credit_once() uses (evm: mut Evm) {
    let arbiter = deploy(window: seconds(0))
    let wallet = evm.create2<Wallet>(value: Wei::zero(), args: (), salt: 1)
    let vectors = honest()
    let id: u256 = evm.call(addr: wallet, gas: GAS, value: BOND, message: WalletMsg::Post { arbiter, @VECTOR_COPY@, class: vectors.class })
    assert!(finalize(arbiter, id).success())
    assert!(credit(arbiter, account: wallet) == BOND)
    assert!(evm.balance(wallet) == Wei::zero())

    let paid: Wei = evm.call(addr: wallet, gas: GAS, value: Wei::zero(), message: WalletMsg::Collect { arbiter })
    assert!(paid == BOND)
    assert!(evm.balance(wallet) == BOND)
    assert!(evm.balance(arbiter) == Wei::zero())
    assert!(credit(arbiter, account: wallet) == Wei::zero())
    assert!(!attempt(arbiter: wallet, WalletMsg::Collect { arbiter }).success())
    assert!(evm.balance(wallet) == BOND)
}

/// What the test wallet does: post a claim with the ETH it is sent, collect
/// its credits.
msg WalletMsg {
    #[selector = sol("post(address,@CLAIM_SOL@,uint8)")]
    Post { arbiter: Address, @VECTOR_FIELDS@, class: u8 } -> u256,
    #[selector = sol("collect(address)")]
    Collect { arbiter: Address } -> Wei,
}

/// A claimant that can receive ETH, which the test runner's address cannot.
contract Wallet uses (ctx: Ctx, call: mut Call) {
    recv {
        #[payable]
        WalletMsg::Post { arbiter, @VECTOR_NAMES@, class } -> u256 uses (ctx, mut call) {
            call.call(addr: arbiter, gas: GAS, value: ctx.value(), message: ArbiterMsg::Claim { @VECTOR_NAMES@, class })
        }

        WalletMsg::Collect { arbiter } -> Wei uses (mut call) {
            call.call(addr: arbiter, gas: GAS, value: Wei::zero(), message: ArbiterMsg::Withdraw {})
        }

        /// Plain ETH transfers (the arbiter's `withdraw`) are accepted.
        #[payable]
        _ {}
    }
}

#[test(balance = 10_000_000)]
fn finalising_before_the_deadline_is_too_early() uses (evm: mut Evm) {
    let arbiter = deploy(window: LONG)
    let id = post(arbiter, honest())
    assert!(failed_with<TooEarly>(finalize(arbiter, id)))
    assert!(failed_with<NotFinal>(attempt(arbiter, ArbiterMsg::ResultOf { id })))
    assert!(failed_with<NothingToWithdraw>(attempt(arbiter, ArbiterMsg::Withdraw {})))
    assert!(status(arbiter, id) == OPEN)
}
"""


def argmax_low(values):
    best = 0
    for i, v in enumerate(values):
        if v > values[best]:
            best = i
    return best


def tests_source(shape, vectors_dir):
    s = shape
    with open(os.path.join(vectors_dir, "trace.jsonl"), "r", encoding="utf-8") as f:
        trace = json.loads(f.readline())
    if trace["i"] != 0:
        raise ModelError("trace.jsonl does not start with sample 0")
    for k in s.layers():
        if s.outs[k - 1] < 3:
            raise ModelError("the generated tests need at least three neurons in every layer")
    # The neuron of each layer whose value the tests change, with a neighbour on each side.
    probe = {k: (5 if s.outs[k - 1] >= 7 else 1) for k in s.layers()}
    # The logit the tests change by one: the first of 3, 4, .., 1, 2 whose change leaves the argmax
    # alone, so that only the neuron challenge can catch it.
    logits = trace["logits"]
    wrong_logit = None
    for t in list(range(3, s.classes)) + [1, 2]:
        if t >= s.classes:
            continue
        mutated = list(logits)
        mutated[t] += 1
        if argmax_low(mutated) == trace["class"]:
            wrong_logit = t
            break
    if wrong_logit is None:
        raise ModelError("no logit of sample 0 can change by one without changing the class")

    tree = commitment(s.model)
    leaf_count = tree["leaf_count"]
    depth = max(len(entry["proof"]) for entry in tree["leaves"])
    offsets = [sum(s.outs[:k - 1]) for k in s.layers()]
    leaves_doc = ", then ".join("%d of layer %d" % (s.outs[k - 1], k) for k in s.layers())

    fields = s.vector_fields()
    vector_fields = ", ".join("%s: %s" % (name, fe) for name, fe, _ in fields)
    vector_copy = ", ".join("%s: vectors.%s" % (name, name) for name, _, _ in fields)
    vector_names = ", ".join(name for name, _, _ in fields)
    claim_sol = ",".join(sol for _, _, sol in fields)

    out = [fill(
        TESTS_HEAD,
        MODEL_USE=s.model_use(["commitment"], ["reference"]),
        LEAVES_DOC=leaves_doc, LEAVES=leaf_count, DEPTH=depth, VECTOR_FIELDS=vector_fields,
        HONEST_HIDDEN="\n".join(I2 + "%s: reference::HIDDEN%d_0.words()," % (s.hid(k), k)
                                for k in s.hiddens()),
    )]
    for k in s.hiddens():
        n = s.outs[k - 1]
        if s.single:
            out.append("/// The honest claim with hidden activation `j` changed by one.")
        else:
            out.append("/// The honest claim with activation `j` of hidden layer %d changed by one." % k)
        out += [
            "fn wrong_%s(_ j: usize) -> Vectors {" % s.hid(k),
            I1 + "var values: [i8; %d] = [0; %d]" % (n, n),
            I1 + "for i in 0 .. %d {" % n,
            I2 + "values[i] = reference::HIDDEN%d_0.get(i)" % k,
            I1 + "}",
            I1 + "values[j] = if values[j] == 127 { 126 } else { values[j] + 1 }",
            I1 + "var vectors = honest()",
            I1 + "vectors.%s = Activations<%d, %d>::pack(values).words()" % (s.hid(k), n, s.frac[k - 1]),
            I1 + "vectors",
            "}",
            "",
        ]
    loops = []
    for k in s.layers():
        index = "j" if offsets[k - 1] == 0 else "%d + j" % offsets[k - 1]
        loops += [
            I1 + "for j in 0 .. %d {" % s.outs[k - 1],
            I2 + "out[%s] = leaf(layer: %d, neuron: j as u256, bias: layer%d::bias(j).raw(), "
            "row: layer%d::row(j).words())" % (index, k, k, k),
            I1 + "}",
        ]
    out.append(fill(TESTS_MIDDLE, CLASSES=s.classes, VECTOR_COPY=vector_copy,
                    LEAF_LOOPS="\n".join(loops)))
    for k in s.layers():
        index = "j" if offsets[k - 1] == 0 else "%d + j" % offsets[k - 1]
        out += [
            "/// The proof of neuron `j` of layer %d." % k,
            "fn proof%d(_ j: usize) -> DynArray<Bytes32> {" % k,
            I1 + "let found = proof_for(%s)" % index,
            I1 + "as_bytes32(found.0, len: found.1)",
            "}",
            "",
        ]
    out += [
        "fn attempt<M>(arbiter: Address, _ message: own M) -> CallOutcome uses (evm: mut Evm) "
        "where M: MsgVariant<Sol> + Encode<Sol> {",
        I1 + "evm.try_call(addr: arbiter, gas: GAS, value: Wei::zero(), message)",
        "}",
        "",
    ]
    for k in s.layers():
        vin, vout = s.layer_in(k), s.layer_out(k)
        out += [
            "/// A neuron challenge of layer %d with the committed row, bias and proof of `j`." % k,
            "fn challenge%d(arbiter: Address, id: u256, _ j: usize, _ vectors: Vectors) -> CallOutcome "
            "uses (evm: mut Evm) {" % k,
            I1 + "attempt(arbiter, ArbiterMsg::ChallengeNeuron%d { id, j: j as u256, %s: vectors.%s, "
            "%s: vectors.%s, row: layer%d::row(j).words(), bias: layer%d::bias(j).raw(), proof: proof%d(j) })"
            % (k, vin, vin, vout, vout, k, k, k),
            "}",
            "",
        ]
    out.append(TESTS_HELPERS)

    test = "#[test(balance = 10_000_000)]"

    def no_fraud(call):
        return I1 + "assert!(failed_with<NoFraud>(%s))" % call

    # An honest claim.
    out += [test, "fn an_honest_claim_survives_every_challenge() uses (evm: mut Evm) {",
            I1 + "let arbiter = deploy(window: LONG)",
            I1 + "let id = post(arbiter, honest())"]
    for k in s.layers():
        out += [I1 + "for j in 0 .. %d {" % s.outs[k - 1],
                I1 + no_fraud("challenge%d(arbiter, id, j, honest())" % k),
                I1 + "}"]
    out += [no_fraud("challenge_class(arbiter, id, honest())"),
            I1 + "assert!(status(arbiter, id) == OPEN)",
            I1 + "assert!(credit(arbiter, account: evm.address()) == Wei::zero())",
            I1 + "assert!(burned(arbiter) == Wei::zero())",
            "}", ""]

    # One wrong hidden value per hidden layer.
    for k in s.hiddens():
        j = probe[k]
        name = "a_wrong_hidden_value" if s.single else "a_wrong_value_of_hidden_layer_%d" % k
        out += [test, "fn %s_is_slashed_by_its_neuron_only() uses (evm: mut Evm) {" % name,
                I1 + "let arbiter = deploy(window: LONG)",
                I1 + "let claimed = wrong_%s(%d)" % (s.hid(k), j),
                I1 + "let id = post(arbiter, claimed)",
                I1 + "// The other neurons of layer %d recompute to the committed values." % k,
                no_fraud("challenge%d(arbiter, id, %d, claimed)" % (k, j - 1)),
                no_fraud("challenge%d(arbiter, id, %d, claimed)" % (k, j + 1))]
        if s.single:
            out.append(I1 + "// Layer 2 and the class are consistent with the (wrong) hidden vector.")
        else:
            out.append(I1 + "// The class is still the argmax of the committed logits.")
        out += [no_fraud("challenge_class(arbiter, id, claimed)"),
                I1 + "assert!(status(arbiter, id) == OPEN)",
                "",
                I1 + "assert!(challenge%d(arbiter, id, %d, claimed).success())" % (k, j),
                I1 + "assert!(status(arbiter, id) == SLASHED)",
                I1 + "assert!(slashed_once(arbiter))",
                I1 + "// Terminal: nothing moves it again, and it has no result.",
                I1 + "assert!(failed_with<ClaimNotOpen>(challenge%d(arbiter, id, %d, claimed)))" % (k, j),
                I1 + "assert!(failed_with<ClaimNotOpen>(finalize(arbiter, id)))",
                I1 + "assert!(failed_with<NotFinal>(attempt(arbiter, ArbiterMsg::ResultOf { id })))",
                I1 + "assert!(slashed_once(arbiter))",
                "}", ""]

    # A wrong logit.
    t = wrong_logit
    last = s.count
    out += [test, "fn a_wrong_logit_is_slashed_by_its_neuron_only() uses (evm: mut Evm) {",
            I1 + "let arbiter = deploy(window: LONG)",
            I1 + "let claimed = wrong_logit(%d)" % t,
            I1 + "let id = post(arbiter, claimed)"]
    for k in s.hiddens():
        out.append(no_fraud("challenge%d(arbiter, id, 0, claimed)" % k))
    out += [no_fraud("challenge%d(arbiter, id, %d, claimed)" % (last, t - 1)),
            I1 + "// Logit %d plus one does not change the argmax, so the class is consistent" % t,
            I1 + "// with the wrong logits: only the neuron challenge catches this claim.",
            no_fraud("challenge_class(arbiter, id, claimed)"),
            I1 + "assert!(challenge%d(arbiter, id, %d, claimed).success())" % (last, t),
            I1 + "assert!(status(arbiter, id) == SLASHED)",
            I1 + "assert!(slashed_once(arbiter))",
            "}", ""]

    # A wrong class.
    out += [test, "fn a_wrong_class_is_slashed_by_the_class_challenge_only() uses (evm: mut Evm) {",
            I1 + "let arbiter = deploy(window: LONG)",
            I1 + "let claimed = wrong_class()",
            I1 + "let id = post(arbiter, claimed)"]
    for k in s.layers():
        out.append(no_fraud("challenge%d(arbiter, id, 0, claimed)" % k))
    out += [I1 + "assert!(challenge_class(arbiter, id, claimed).success())",
            I1 + "assert!(status(arbiter, id) == SLASHED)",
            I1 + "assert!(slashed_once(arbiter))",
            "}", ""]

    # Malformed claims.
    def rejected(error, variable, value="BOND"):
        return (I1 + "assert!(failed_with<%s>(evm.try_call(addr: arbiter, gas: GAS, value: %s, "
                "message: claim_message(%s))))" % (error, value, variable))

    has_padding = s.n_in % 32 != 0
    out += [test, "fn malformed_claims_are_rejected() uses (evm: mut Evm) {",
            I1 + "let arbiter = deploy(window: LONG)"]
    if has_padding:
        out += [I1 + "var padding = honest()",
                I1 + "padding.x0[%d] |= 1" % (words(s.n_in) - 1)]
    out += [I1 + "var high_input = honest()",
            I1 + "high_input.x0[0] |= 0x80 << 248"]
    for k in s.hiddens():
        out += [I1 + "var high_%s = honest()" % s.hid(k),
                I1 + "high_%s.%s[0] |= 0x80 << 248" % (s.hid(k), s.hid(k))]
    out += [I1 + "var hidden_padding = honest()",
            I1 + "hidden_padding.%s[0] = 0" % s.hid(1),
            I1 + "var class = honest()",
            I1 + "class.class = %d" % s.classes]
    if has_padding:
        out.append(rejected("MalformedInput", "padding"))
    out.append(rejected("MalformedInput", "high_input"))
    for k in s.hiddens():
        out.append(rejected("MalformedHidden", "high_%s" % s.hid(k)))
    out += [rejected("ClassOutOfRange", "class"),
            rejected("BondTooSmall", "honest()", "BOND - wei(1)"),
            I1 + "assert!(claim_count(arbiter) == 0)"]
    if s.single:
        out.append(I1 + "// A hidden vector of all zeros is well formed, whatever it claims.")
    else:
        out.append(I1 + "// A hidden vector with a word of zeros is well formed, whatever it claims.")
    out += [I1 + "assert!(evm.try_call(addr: arbiter, gas: GAS, value: BOND, "
            "message: claim_message(hidden_padding)).success())",
            I1 + "assert!(claim_count(arbiter) == 1)",
            "}", ""]

    # A wrong proof (layer 1).
    j = probe[1]

    def challenge1_literal(row, bias, proof):
        return ("ArbiterMsg::ChallengeNeuron1 { id, j: %d, x0: claimed.x0, %s: claimed.%s, "
                "row: layer1::row(%d).words(), bias: layer1::bias(%d).raw()%s, proof: proof1(%d) }"
                % (j, s.hid(1), s.hid(1), row, j, bias, proof))

    out += [test, "fn a_wrong_proof_is_rejected() uses (evm: mut Evm) {",
            I1 + "let arbiter = deploy(window: LONG)",
            I1 + "let claimed = wrong_%s(%d)" % (s.hid(1), j),
            I1 + "let id = post(arbiter, claimed)",
            I1 + "// The proof of another neuron, another bias, another row: none verifies.",
            I1 + "let other_proof = " + challenge1_literal(j, "", j + 1),
            I1 + "let other_bias = " + challenge1_literal(j, " + 1", j),
            I1 + "let other_row = " + challenge1_literal(j + 1, "", j),
            I1 + "assert!(failed_with<NotCommitted>(attempt(arbiter, other_proof)))",
            I1 + "assert!(failed_with<NotCommitted>(attempt(arbiter, other_bias)))",
            I1 + "assert!(failed_with<NotCommitted>(attempt(arbiter, other_row)))",
            I1 + "assert!(status(arbiter, id) == OPEN)",
            I1 + "assert!(challenge1(arbiter, id, %d, claimed).success())" % j,
            "}", ""]

    # Vectors that do not match the commitments.
    out += [test, "fn vectors_that_do_not_match_the_commitments_are_rejected() uses (evm: mut Evm) {",
            I1 + "let arbiter = deploy(window: LONG)",
            I1 + "let claimed = wrong_%s(%d)" % (s.hid(1), j),
            I1 + "let id = post(arbiter, claimed)",
            I1 + "var other_input = claimed",
            I1 + "other_input.x0 = reference::X1.words()"]
    for k in s.hiddens():
        out += [I1 + "var other_%s = claimed" % s.hid(k),
                I1 + "other_%s.%s = reference::HIDDEN%d_1.words()" % (s.hid(k), s.hid(k), k)]
    out += [I1 + "var other_logits = claimed",
            I1 + "other_logits.logits = reference::LOGITS1.values()",
            I1 + "assert!(failed_with<InputMismatch>(challenge1(arbiter, id, %d, other_input)))" % j]
    for k in s.hiddens():
        out += [
            I1 + "assert!(failed_with<OutputMismatch>(challenge%d(arbiter, id, %d, other_%s)))"
            % (k, probe[k] if k == 1 else 0, s.hid(k)),
            I1 + "assert!(failed_with<InputMismatch>(challenge%d(arbiter, id, 0, other_%s)))"
            % (k + 1, s.hid(k)),
        ]
    out += [I1 + "assert!(failed_with<OutputMismatch>(challenge%d(arbiter, id, 0, other_logits)))" % last,
            I1 + "assert!(failed_with<OutputMismatch>(challenge_class(arbiter, id, other_logits)))",
            I1 + "assert!(failed_with<UnknownClaim>(challenge_class(arbiter, id: 7, claimed)))",
            I1 + "assert!(status(arbiter, id) == OPEN)",
            "}", ""]

    out.append(fill(TESTS_TAIL, VECTOR_COPY=vector_copy, VECTOR_FIELDS=vector_fields,
                    VECTOR_NAMES=vector_names, CLAIM_SOL=claim_sol))
    return "\n".join(out)


LIB = """\
//! GENERATED by tools/gen_arbiter.py from the model "@NAME@". Do not edit.
//!
//! The dispute arbiter of the network @SIZES@: `Arbiter` settles disputes about
//! claimed inferences by re-running one neuron, or the argmax, against the
//! committed model (`docs/spec-dispute.md`). It calls the `model` ingot, which
//! calls `tensile`.
pub use arbiter::{Arbiter, ArbiterMsg}
pub use commitment::leaf
"""


def generate(model, vectors_dir=None, lib=False):
    """Return {file name: source} for the arbiter ingot's src directory."""
    shape = Shape(model)
    files = {
        "arbiter.fe": arbiter_source(shape),
        "record.fe": record_source(shape),
        "signals.fe": signals_source(shape),
        "commitment.fe": commitment_fe_source(shape),
    }
    if vectors_dir is not None:
        files["arbiter_tests.fe"] = tests_source(shape, vectors_dir)
    if lib:
        sizes = " -> ".join([str(shape.n_in)] + [str(n) for n in shape.outs])
        files["lib.fe"] = fill(LIB, NAME=model["name"], SIZES=sizes)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.path.join(ROOT, "ref", "out", "model.json"))
    parser.add_argument("--vectors", default=None,
                        help="directory with trace.jsonl, to generate arbiter_tests.fe (the model "
                             "ingot must then hold at least two reference samples)")
    parser.add_argument("--lib", action="store_true", help="also write lib.fe")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        model = json.load(f)
    try:
        files = generate(model, args.vectors, args.lib)
    except ModelError as error:
        print("gen_arbiter: %s" % error, file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    for name, source in sorted(files.items()):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(source)
        print("wrote %s (%d bytes)" % (os.path.join(args.out, name), len(source)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
