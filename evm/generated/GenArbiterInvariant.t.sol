// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {IArbiter} from "./ArbiterBase.sol";
import {FeDeployer} from "./FeDeployer.sol";
import {GClaim, GModel, GenUser, Gx} from "./GenBase.sol";

/// An actor that refuses ETH: its `withdraw` always fails with `TransferFailed`, so its credits
/// stay in the arbiter for good (they must still be counted by P1).
contract GenStuckActor {}

/// Drives a generated arbiter with random sequences of honest and dishonest claims (a lie at any
/// neuron of any layer, a lie in a hidden layer carried consistently through every later layer, a
/// wrong class), malformed claims (in the input, in any hidden vector, the class, the bond),
/// correct and corrupted challenges of every layer, time steps, finalisations and withdrawals
/// from five actors (four accounts and one contract that refuses ETH). It predicts the outcome of
/// every call from the spec (docs/spec-dispute.md sections 3 to 5, with `GModel` as the network)
/// and keeps ghost accounts; a call whose outcome differs from the prediction is recorded in
/// `violations`, which the invariant reads. The generalisation of evm/test/ArbiterInvariant.t.sol
/// to any number of layers.
contract GenArbiterHandler is GenUser {
    struct Rec {
        uint256 sample;
        uint256 kind; // 0 honest, 1 one value of layer l wrong, 2 hidden l j wrong and carried on, 3 class
        uint256 l;
        uint256 j;
        uint256 salt;
        address claimant;
        uint256 bond;
        uint256 deadline;
        uint8 status; // what the spec says the status is now
        bytes32 fixedFields; // claimant, bond, deadline, class and every hash at claim time
    }

    uint256 public immutable bondRequired;
    uint256 public immutable window;
    address[5] internal actors;

    Rec[] internal recs;
    mapping(address => uint256) public ghostCredit;
    mapping(address => uint256) public ghostCreditedEver;
    mapping(address => uint256) public ghostWithdrawn;
    uint256 public ghostOpenBonds;
    uint256 public ghostBurned;
    uint256 public ghostTotalBonded;
    uint256 public ghostTotalCredited;
    uint256 public ghostTotalWithdrawn;
    uint256 public violations;
    string public firstViolation;

    // What a run did. `slashesBy[k]` counts slashes by check k: a neuron of layer k (1 .. L) or
    // the class (L + 1).
    uint256[] public slashesBy;
    uint256 public slashes;
    uint256 public finals;
    uint256 public payouts;
    uint256 public refused;

    constructor(IArbiter a, uint256 bondRequired_, uint256 window_) {
        arbiter = a;
        initShape();
        bondRequired = bondRequired_;
        window = window_;
        actors[0] = address(0xA11CE);
        actors[1] = address(0xB0B);
        actors[2] = address(0xCA201);
        actors[3] = address(0xD0D0);
        actors[4] = address(new GenStuckActor());
        for (uint256 k = 0; k <= L + 1; k++) {
            slashesBy.push(0);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Views for the invariant

    function layerCount() external view returns (uint256) {
        return L;
    }

    function claimCount() external view returns (uint256) {
        return recs.length;
    }

    function rec(uint256 id) external view returns (Rec memory) {
        return recs[id];
    }

    function actor(uint256 k) external view returns (address) {
        return actors[k];
    }

    function actorCount() external pure returns (uint256) {
        return 5;
    }

    function fixedFields(uint256 id) public view returns (bytes32) {
        (address claimant, uint256 bond, uint256 deadline, uint8 cls,) = arbiter.claimInfo(id);
        (bool ok, uint256[] memory h) = hashesOf(id);
        require(ok, "claimHashes");
        return keccak256(abi.encode(claimant, bond, deadline, cls, h));
    }

    // ----------------------------------------------------------------------------------------
    // Helpers

    function violation(string memory what) internal {
        if (violations == 0) firstViolation = what;
        violations++;
    }

    function build(Rec memory r) internal view returns (GClaim memory c) {
        c = Gx.sample(r.sample);
        if (r.kind == 1 || r.kind == 2) lie(c, r.l, r.j, r.salt);
        if (r.kind == 2) GModel.carry(c, r.l);
        if (r.kind == 3) lie(c, L + 1, 0, r.salt);
    }

    function callAs(address who, bytes memory data) internal returns (bool ok, bytes memory ret) {
        vm.prank(who);
        (ok, ret) = address(arbiter).call(data);
    }

    /// Sends the correct challenge of check `kind`, `j` and checks its outcome against the spec.
    function challengeChecked(uint256 id, address who, uint256 kind, uint256 j) internal {
        Rec storage r = recs[id];
        GClaim memory c = build(r);
        bool expect = r.status == 1 && block.timestamp < r.deadline && fails(c, kind, j);
        (bool ok,) = callAs(who, checkCall(id, kind, j, c));
        if (ok != expect) {
            violation(ok ? "a challenge succeeded that the spec refuses" : "a challenge failed that the spec accepts");
            return;
        }
        if (ok) {
            r.status = 2;
            credit(who, r.bond / 2);
            ghostOpenBonds -= r.bond - r.bond / 2;
            ghostBurned += r.bond - r.bond / 2;
            slashes++;
            slashesBy[kind]++;
        } else {
            refused++;
        }
    }

    function credit(address who, uint256 amount) internal {
        ghostOpenBonds -= amount;
        ghostCredit[who] += amount;
        ghostCreditedEver[who] += amount;
        ghostTotalCredited += amount;
    }

    /// The first open claim at or after `id` (cyclically), or `id` if none is open.
    function openFrom(uint256 id) internal view returns (uint256) {
        for (uint256 k = 0; k < recs.length; k++) {
            uint256 i = (id + k) % recs.length;
            if (recs[i].status == 1) return i;
        }
        return id;
    }

    // ----------------------------------------------------------------------------------------
    // Actions

    /// An honest or dishonest well-formed claim with at least the required bond.
    function claim(uint256 actorSeed, uint256 sampleSeed, uint256 mutation, uint256 extra) external {
        Rec memory r;
        r.sample = sampleSeed % Gx.sampleCount();
        r.kind = (mutation & 0xff) % 4;
        if (r.kind == 1) {
            r.l = 1 + (mutation >> 8) % L;
        } else if (r.kind == 2) {
            r.l = 1 + (mutation >> 8) % (L - 1);
        }
        if (r.l != 0) r.j = (mutation >> 16) % OUT[r.l];
        r.salt = mutation >> 32;
        r.claimant = actors[actorSeed % 5];
        r.bond = bondRequired + extra % 1 ether;
        GClaim memory c = build(r);
        (bool ok, bytes memory ret) = postRaw(claimData(c), r.claimant, r.bond);
        if (!ok) {
            violation("a well-formed claim was refused");
            return;
        }
        uint256 id = abi.decode(ret, (uint256));
        if (id != recs.length) violation("claims are not numbered in order from 0");
        r.deadline = block.timestamp + window;
        r.status = 1;
        r.fixedFields = fixedFields(id);
        recs.push(r);
        ghostOpenBonds += r.bond;
        ghostTotalBonded += r.bond;
    }

    /// A malformed claim: an input byte above 127, non-zero input padding, a byte above 127 or
    /// non-zero padding in any hidden vector, a class out of range, or a bond too small. Refused,
    /// nothing stored. (Padding falls back to a byte above 127 where a vector has none.)
    function claimMalformed(uint256 actorSeed, uint256 sampleSeed, uint256 what, uint256 pos, uint8 v) external {
        GClaim memory c = Gx.sample(sampleSeed % Gx.sampleCount());
        uint256 value = bondRequired;
        uint256 w = what % 4;
        uint256 l = (what >> 8) % L; // the vector: 0 the input, else a hidden layer
        if (w == 0) {
            GModel.setByte(c.v[l], pos % OUT[l], v | 0x80);
        } else if (w == 1) {
            uint256 mask = GModel.paddingMask(OUT[l]);
            if (mask == 0) GModel.setByte(c.v[l], pos % OUT[l], v | 0x80);
            else c.v[l][c.v[l].length - 1] |= (uint256(v == 0 ? 1 : v) << (8 * (pos % 32))) & mask | (mask & 1);
        } else if (w == 2) {
            c.cls = uint8(CLASSES + v % (256 - CLASSES));
        } else {
            value = pos % bondRequired;
        }
        (bool ok,) = postRaw(claimData(c), actors[actorSeed % 5], value);
        if (ok) violation("a malformed claim was accepted");
        else refused++;
    }

    /// A correct challenge (committed vectors, the right row, bias and proof) of a random neuron
    /// of a random layer, or of the class, of a random claim, open or not, early or late.
    function challenge(uint256 actorSeed, uint256 claimSeed, uint256 kindSeed, uint256 j) external {
        if (recs.length == 0) return;
        uint256 kind = 1 + kindSeed % (L + 1);
        challengeChecked(claimSeed % recs.length, actors[actorSeed % 5], kind, kind <= L ? j % OUT[kind] : 0);
    }

    /// The challenge the spec's walk names for the lie of the first open claim from a random one
    /// (the class challenge for an honest claim), so that slashes are frequent.
    function challengeNamed(uint256 actorSeed, uint256 claimSeed) external {
        if (recs.length == 0) return;
        uint256 id = openFrom(claimSeed % recs.length);
        Rec memory r = recs[id];
        if (r.kind == 1 || r.kind == 2) challengeChecked(id, actors[actorSeed % 5], r.l, r.j);
        else challengeChecked(id, actors[actorSeed % 5], L + 1, 0);
    }

    /// A challenge of any layer with one corrupted piece: another neuron's leaf, a flipped bit in
    /// the row or the bias, a cut proof, a proof of another layer, a substituted input or output
    /// vector, or other logits in the class challenge. Always refused.
    function challengeCorrupt(uint256 actorSeed, uint256 claimSeed, uint256 how, uint256 j, uint256 bit) external {
        if (recs.length == 0) return;
        uint256 id = claimSeed % recs.length;
        GClaim memory c = build(recs[id]);
        uint256 l = 1 + (how >> 8) % L;
        uint256 n = OUT[l];
        j %= n;
        uint256 k = (j + 1 + bit % (n - 1)) % n; // another neuron
        uint256[] memory row = Gx.row(l, j);
        int32 bias = Gx.bias(l, j);
        bytes32[] memory proof = Gx.proof(l, j);
        bytes memory input = inputOf(c, l);
        bytes memory output = outputOf(c, l);
        uint256 h = how % 8;
        if (h == 0) {
            row = Gx.row(l, k);
            bias = Gx.bias(l, k);
            proof = Gx.proof(l, k);
        } else if (h == 1) {
            row[(bit >> 8) % row.length] ^= uint256(1) << (bit % 256);
        } else if (h == 2) {
            bias = int32(uint32(bias) ^ (uint32(1) << uint32(bit % 32)));
        } else if (h == 3) {
            assembly ("memory-safe") {
                mstore(proof, sub(mload(proof), 1))
            }
        } else if (h == 4) {
            uint256 m = 1 + (l + bit % (L - 1)) % L; // another layer
            proof = Gx.proof(m, j % OUT[m]);
        } else if (h == 5) {
            input[(bit >> 3) % input.length] ^= bytes1(uint8(1) << uint8(bit % 8));
        } else if (h == 6) {
            output[(bit >> 3) % output.length] ^= bytes1(uint8(1) << uint8(bit % 8));
        }
        bytes memory data;
        if (h == 7) {
            GClaim memory d = GModel.copy(c);
            d.z[bit % CLASSES] ^= int32(uint32(1) << uint32((bit >> 8) % 32));
            data = classCall(id, d);
        } else {
            data = neuronData(l, id, j, input, output, row, bias, proof);
        }
        (bool ok,) = callAs(actors[actorSeed % 5], data);
        if (ok) violation("a corrupted challenge succeeded");
        else refused++;
    }

    /// Time moves forward by up to two thirds of the window, or to just before or exactly at a
    /// claim's deadline.
    function warp(uint256 dt, uint256 claimSeed, uint8 mode) external {
        uint256 target = block.timestamp + dt % (2 * window / 3 + 1);
        if (mode % 3 != 0 && recs.length > 0) {
            uint256 d = recs[openFrom(claimSeed % recs.length)].deadline;
            target = mode % 3 == 1 ? d - 1 : d;
        }
        if (target > block.timestamp) vm.warp(target);
    }

    /// Finalises a random claim (`anyClaim`) or the first open claim from a random one.
    function finalize(uint256 actorSeed, uint256 claimSeed, bool anyClaim) external {
        if (recs.length == 0) return;
        uint256 id = anyClaim ? claimSeed % recs.length : openFrom(claimSeed % recs.length);
        Rec storage r = recs[id];
        bool expect = r.status == 1 && block.timestamp >= r.deadline;
        (bool ok,) = callAs(actors[actorSeed % 5], abi.encodeCall(IArbiter.finalize, (id)));
        if (ok != expect) {
            violation(ok ? "finalize succeeded where the spec refuses" : "finalize failed where the spec accepts");
            return;
        }
        if (ok) {
            r.status = 3;
            credit(r.claimant, r.bond);
            finals++;
            (bool readable, bytes memory ret) = address(arbiter).staticcall(abi.encodeCall(IArbiter.result, (id)));
            if (!readable || abi.decode(ret, (uint8)) != build(r).cls) violation("result of a final claim");
        } else {
            refused++;
        }
    }

    function withdraw(uint256 actorSeed) external {
        address who = actors[actorSeed % 5];
        uint256 owed = ghostCredit[who];
        uint256 before = who.balance;
        (bool ok, bytes memory ret) = callAs(who, abi.encodeCall(IArbiter.withdraw, ()));
        bool expect = owed > 0 && who != actors[4];
        if (ok != expect) {
            violation(ok ? "withdraw paid where the spec refuses" : "withdraw failed with a credit");
            return;
        }
        if (!ok) {
            bytes4 why = bytes4(ret);
            bytes4 want = owed == 0 ? IArbiter.NothingToWithdraw.selector : IArbiter.TransferFailed.selector;
            if (why != want) violation("withdraw refused for the wrong reason");
            refused++;
            return;
        }
        uint256 amount = abi.decode(ret, (uint256));
        if (amount != owed || who.balance != before + owed) violation("withdraw paid another amount than the credit");
        ghostCredit[who] = 0;
        payouts++;
        ghostWithdrawn[who] += amount;
        ghostTotalWithdrawn += amount;
    }
}

/// Invariant tests P1 and P4 of docs/spec-dispute.md section 9, and the withdrawal rule, for a
/// generated arbiter, over the handler above: evm/test/ArbiterInvariant.t.sol with any number of
/// layers.
contract GenArbiterInvariantTest is GenUser {
    struct FuzzSelector {
        address addr;
        bytes4[] selectors;
    }

    GenArbiterHandler internal handler;

    function setUp() public {
        Gx.install();
        initShape();
        vm.warp(1_000_000);
        arbiter = deployArbiter(1 ether, 1 hours);
        handler = new GenArbiterHandler(arbiter, 1 ether, 1 hours);
    }

    function targetContracts() public view returns (address[] memory t) {
        t = new address[](1);
        t[0] = address(handler);
    }

    function targetSelectors() public view returns (FuzzSelector[] memory t) {
        bytes4[] memory s = new bytes4[](8);
        s[0] = GenArbiterHandler.claim.selector;
        s[1] = GenArbiterHandler.claimMalformed.selector;
        s[2] = GenArbiterHandler.challenge.selector;
        s[3] = GenArbiterHandler.challengeNamed.selector;
        s[4] = GenArbiterHandler.challengeCorrupt.selector;
        s[5] = GenArbiterHandler.warp.selector;
        s[6] = GenArbiterHandler.finalize.selector;
        s[7] = GenArbiterHandler.withdraw.selector;
        t = new FuzzSelector[](1);
        t[0] = FuzzSelector(address(handler), s);
    }

    /// P1: the balance equals the open bonds plus the unwithdrawn credits plus the burned halves
    /// of slashed bonds, as the ghost accounts say and as the arbiter's own views say. P4: every
    /// claim's status is the one the spec predicts, its other fields (every committed hash
    /// included) never changed, and each bond was shared out exactly once. Withdrawals: nobody
    /// withdrew more than they were credited, and the arbiter's credits equal the ghost credits.
    /// Every call the handler made had the outcome the spec predicts (`violations`).
    /// forge-config: default.invariant.runs = 256
    /// forge-config: default.invariant.depth = 100
    /// forge-config: default.invariant.show-metrics = true
    function invariant_P1_P4_ledger_lifecycle_and_withdrawals() public view {
        require(handler.violations() == 0, handler.firstViolation());
        uint256 n = handler.claimCount();
        require(arbiter.claimCount() == n, "claim count");
        uint256 openBonds = 0;
        uint256 closedBonds = 0;
        for (uint256 id = 0; id < n; id++) {
            GenArbiterHandler.Rec memory r = handler.rec(id);
            uint8 status = statusOf(id);
            require(status == r.status, "P4: a status differs from the spec's lifecycle");
            require(handler.fixedFields(id) == r.fixedFields, "P4: a claim's fixed fields changed");
            if (status == 1) openBonds += r.bond;
            else closedBonds += r.bond;
        }
        uint256 credits = 0;
        for (uint256 k = 0; k < handler.actorCount(); k++) {
            address a = handler.actor(k);
            require(arbiter.creditOf(a) == handler.ghostCredit(a), "credit differs from the ghost");
            require(handler.ghostWithdrawn(a) <= handler.ghostCreditedEver(a), "withdrew more than credited");
            credits += arbiter.creditOf(a);
        }
        uint256 burned = arbiter.burned();
        require(openBonds == handler.ghostOpenBonds(), "P1: open bonds differ from the ghost");
        require(burned == handler.ghostBurned(), "P1: burned differs from the ghost");
        require(
            address(arbiter).balance == openBonds + credits + burned, "P1: balance != open bonds + credits + burned"
        );
        require(handler.ghostTotalCredited() + burned == closedBonds, "P4: credited + burned != bonds of closed claims");
        require(handler.ghostTotalBonded() == openBonds + closedBonds, "P4: bonded != open + closed");
        require(
            credits + handler.ghostTotalWithdrawn() == handler.ghostTotalCredited(),
            "P4: outstanding + withdrawn != credited"
        );
        require(handler.ghostTotalCredited() <= handler.ghostTotalBonded(), "P4: credited more than bonded");
    }

    /// What this run of the campaign did, one line per run: claims, slashes by a neuron of each
    /// layer and by the class, finalisations, payouts, refused calls (tools/gen_foundry.sh adds
    /// them up and fails if an outcome never happened in the whole campaign).
    function afterInvariant() public {
        string memory line = vm.toString(handler.claimCount());
        for (uint256 k = 1; k <= L + 1; k++) {
            line = string.concat(line, " ", vm.toString(handler.slashesBy(k)));
        }
        line = string.concat(
            line,
            " ",
            vm.toString(handler.finals()),
            " ",
            vm.toString(handler.payouts()),
            " ",
            vm.toString(handler.refused())
        );
        vm.writeLine("invariant-out/campaign.log", line);
    }

    /// The handler itself, on a scripted sequence: it reaches every outcome (a slash by each
    /// layer's handler, by a carried lie and by the class challenge, finalisation, paid and
    /// refused withdrawals, refused corrupt and malformed calls) without a violation, so the
    /// campaign above is not vacuous.
    function test_handler_reaches_every_outcome() public {
        // Claims 0 .. L - 1: one value of layer id + 1 wrong.
        for (uint256 l = 1; l <= L; l++) {
            handler.claim(l - 1, 0, 1 | ((l - 1) << 8) | (1 << 16) | (5 << 32), 0);
        }
        handler.claim(0, 1, 2 | (2 << 16) | (9 << 32), 0); // L: hidden 1 2 wrong, the rest consistent with it
        handler.claim(1, 2, 3 | (1 << 32), 0); // L + 1: class wrong
        handler.claim(4, 4, 0, 0); // L + 2: honest, claimed by the actor that refuses ETH
        handler.claim(0, 5, 0, 5); // L + 3: honest
        for (uint256 m = 2; m <= L + 1; m++) {
            handler.challenge(1, L, m - 1, 0); // claim L at layer m or the class: consistent, refused
        }
        for (uint256 id = 0; id < L; id++) {
            handler.challengeNamed(1 + id % 3, id);
        }
        handler.challengeNamed(4, L); // the actor that refuses ETH collects claim L's half bond
        handler.challengeNamed(2, L + 1);
        handler.challengeCorrupt(0, L + 2, 0, 1, 2);
        handler.challengeCorrupt(0, L + 3, 4 | (1 << 8), 1, 2);
        handler.claimMalformed(0, 0, 0, 10, 1);
        handler.claimMalformed(0, 0, 3, 10, 1);
        handler.withdraw(0); // no credit yet
        handler.warp(0, L + 2, 2); // to claim L + 2's deadline
        handler.challenge(0, L + 2, L, 0); // the class, too late (and honest)
        handler.finalize(0, L + 2, true);
        handler.finalize(0, L + 3, true);
        handler.finalize(0, 0, true); // already slashed
        handler.withdraw(4); // refuses ETH: TransferFailed, credit kept
        handler.withdraw(1);
        handler.withdraw(0); // claim L + 3's bond
        handler.withdraw(0); // nothing left
        require(handler.violations() == 0, handler.firstViolation());
        require(handler.slashes() == L + 2 && handler.finals() == 2 && handler.payouts() == 2, "outcomes");
        for (uint256 k = 1; k <= L + 1; k++) {
            require(handler.slashesBy(k) > 0, "a check never slashed");
        }
        require(handler.refused() == L + 9, "refusals");
        invariant_P1_P4_ledger_lifecycle_and_withdrawals();
        require(arbiter.burned() == (L + 2) * 0.5 ether, "every slash burned half a bond");
    }

    /// Negative control: against an arbiter deployed with another model root, the handler's
    /// predictions fail (a proven challenge of the middle layer is refused) and the invariant
    /// reports it.
    function test_negative_control_wrong_root_is_detected() public {
        IArbiter other =
            IArbiter(FeDeployer.deploy("Arbiter", abi.encode(uint256(keccak256("another model")), 1 ether, 1 hours)));
        arbiter = other;
        handler = new GenArbiterHandler(other, 1 ether, 1 hours);
        handler.claim(0, 0, 1 | ((L >= 3 ? 1 : 0) << 8) | (3 << 16), 0);
        handler.challengeNamed(1, 0);
        require(handler.violations() == 1, "the handler missed a refused proven challenge");
        (bool ok,) = address(this).staticcall(abi.encodeCall(this.invariant_P1_P4_ledger_lifecycle_and_withdrawals, ()));
        require(!ok, "the invariant passed against a broken arbiter");
    }

    /// Negative control: one wei more than the ledger (as forced ETH would add) fails P1.
    function test_negative_control_unbalanced_ledger_is_detected() public {
        handler.claim(0, 0, 0, 0);
        vm.deal(address(arbiter), address(arbiter).balance + 1);
        (bool ok, bytes memory reason) =
            address(this).staticcall(abi.encodeCall(this.invariant_P1_P4_ledger_lifecycle_and_withdrawals, ()));
        require(
            !ok
                && keccak256(reason)
                    == keccak256(abi.encodeWithSignature("Error(string)", "P1: balance != open bonds + credits + burned")),
            "P1 did not fail"
        );
    }
}
