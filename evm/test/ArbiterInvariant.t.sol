// SPDX-License-Identifier: MIT
pragma solidity 0.8.38;

import {ArbiterUser, Claim, Fx, IArbiter, Model} from "./ArbiterBase.sol";
import {FeDeployer} from "./FeDeployer.sol";

/// An actor that refuses ETH: its `withdraw` always fails with `TransferFailed`, so its credits
/// stay in the arbiter for good (they must still be counted by P1).
contract StuckActor {}

/// Drives the arbiter with random sequences of honest and dishonest claims, malformed claims,
/// correct and corrupted challenges, time steps, finalisations and withdrawals from five actors
/// (four accounts and one contract that refuses ETH). It predicts the outcome of every call from
/// the spec (docs/spec-dispute.md sections 3 to 5, with `Model` as the network) and keeps ghost
/// accounts; a call whose outcome differs from the prediction is recorded in `violations`, which
/// the invariant reads (a revert inside a handler would only be counted as a failed call).
contract ArbiterHandler is ArbiterUser {
    struct Rec {
        uint256 sample;
        uint256 kind; // 0 honest, 1 hidden j, 2 logit j, 3 class, 4 hidden j with consistent logits and class
        uint256 j;
        int256 value;
        address claimant;
        uint256 bond;
        uint256 deadline;
        uint8 status; // what the spec says the status is now
        bytes32 fixedFields; // claimant, bond, deadline, class and the three hashes at claim time
    }

    IArbiter public immutable arbiter;
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

    // Counters of what a run did (forge's own metrics are only per selector). The test writes them
    // to a file after each run; tools/m3_foundry.sh adds them up and fails if an outcome never
    // happened in the whole campaign.
    uint256[3] public slashesBy; // by challenge kind: neuron of layer 1, of layer 2, class
    uint256 public slashes;
    uint256 public finals;
    uint256 public payouts;
    uint256 public refused;

    constructor(IArbiter a, uint256 bondRequired_, uint256 window_) {
        arbiter = a;
        bondRequired = bondRequired_;
        window = window_;
        actors[0] = address(0xA11CE);
        actors[1] = address(0xB0B);
        actors[2] = address(0xCA201);
        actors[3] = address(0xD0D0);
        actors[4] = address(new StuckActor());
    }

    // ----------------------------------------------------------------------------------------
    // Views for the invariant

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
        (uint256 hx, uint256 hh, uint256 hz) = arbiter.claimHashes(id);
        return keccak256(abi.encode(claimant, bond, deadline, cls, hx, hh, hz));
    }

    // ----------------------------------------------------------------------------------------
    // Helpers

    function violation(string memory what) internal {
        if (violations == 0) firstViolation = what;
        violations++;
    }

    function build(Rec memory r) internal view returns (Claim memory c) {
        c = Fx.sample(r.sample);
        if (r.kind == 1 || r.kind == 4) Model.setByte(c.hidden, r.j, uint8(uint256(r.value)));
        if (r.kind == 2) c.logits[r.j] = int32(r.value);
        if (r.kind == 3) c.cls = uint8(uint256(r.value));
        if (r.kind == 4) {
            for (uint256 k = 0; k < 10; k++) {
                c.logits[k] = Model.neuron2(k, c.hidden);
            }
            c.cls = Model.argmax(c.logits);
        }
    }

    /// Whether a correct challenge of kind `kind` (0 neuron of layer 1, 1 of layer 2, 2 class)
    /// at `j` proves claim `c` wrong, by the spec.
    function proves(Claim memory c, uint256 sample, uint256 kind, uint256 j) internal view returns (bool) {
        if (kind == 0) return Model.hiddenAt(c.hidden, j) != Model.hiddenAt(Fx.sample(sample).hidden, j);
        if (kind == 1) return c.logits[j] != Model.neuron2(j, c.hidden);
        return c.cls != Model.argmax(c.logits);
    }

    function challengeData(uint256 id, Claim memory c, uint256 kind, uint256 j) internal view returns (bytes memory) {
        if (kind == 0) {
            return abi.encodeCall(
                IArbiter.challengeNeuron1, (id, j, c.x0, c.hidden, Fx.row1(j), Fx.bias1(j), Fx.proof(1, j))
            );
        }
        if (kind == 1) {
            return abi.encodeCall(
                IArbiter.challengeNeuron2, (id, j, c.hidden, c.logits, Fx.row2(j), Fx.bias2(j), Fx.proof(2, j))
            );
        }
        return abi.encodeCall(IArbiter.challengeClass, (id, c.logits));
    }

    function callAs(address who, bytes memory data) internal returns (bool ok, bytes memory ret) {
        vm.prank(who);
        (ok, ret) = address(arbiter).call(data);
    }

    /// Sends a correct challenge and checks its outcome against the spec.
    function challengeChecked(uint256 id, address who, uint256 kind, uint256 j) internal {
        Rec storage r = recs[id];
        Claim memory c = build(r);
        bool expect = r.status == 1 && block.timestamp < r.deadline && proves(c, r.sample, kind, j);
        (bool ok,) = callAs(who, challengeData(id, c, kind, j));
        if (ok != expect) {
            violation(ok ? "a challenge succeeded that the spec refuses" : "a challenge failed that the spec accepts");
            return;
        }
        if (ok) {
            r.status = 2;
            // A slash credits the challenger half the bond, rounded down, and burns the rest.
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

    // ----------------------------------------------------------------------------------------
    // Actions

    /// An honest or dishonest well-formed claim with at least the required bond.
    function claim(uint256 actorSeed, uint256 sampleSeed, uint256 mutation, uint256 extra) external {
        Rec memory r;
        r.sample = sampleSeed % Fx.sampleCount();
        r.kind = (mutation & 0xff) % 5;
        Claim memory honest = Fx.sample(r.sample);
        if (r.kind == 1 || r.kind == 4) {
            r.j = (mutation >> 8) % 32;
            uint256 h = Model.hiddenAt(honest.hidden, r.j);
            r.value = int256(((mutation >> 16) % 127 + 1 + h) % 128); // any other value in 0..127
        } else if (r.kind == 2) {
            r.j = (mutation >> 8) % 10;
            int32 v = int32(uint32(mutation >> 16));
            if (v == honest.logits[r.j]) v ^= 1;
            r.value = v;
        } else if (r.kind == 3) {
            r.value = int256(((mutation >> 16) % 9 + 1 + honest.cls) % 10);
        }
        r.claimant = actors[actorSeed % 5];
        r.bond = bondRequired + extra % 1 ether;
        Claim memory c = build(r);
        vm.deal(r.claimant, r.claimant.balance + r.bond);
        vm.prank(r.claimant);
        try arbiter.claim{value: r.bond}(c.x0, c.hidden, c.logits, c.cls) returns (uint256 id) {
            if (id != recs.length) violation("claims are not numbered in order from 0");
            r.deadline = block.timestamp + window;
            r.status = 1;
            r.fixedFields = fixedFields(id);
            recs.push(r);
            ghostOpenBonds += r.bond;
            ghostTotalBonded += r.bond;
        } catch {
            violation("a well-formed claim was refused");
        }
    }

    /// A malformed claim (input byte above 127, input padding, hidden byte above 127, class of 10
    /// or more, bond too small): refused, nothing stored.
    function claimMalformed(uint256 actorSeed, uint256 sampleSeed, uint256 what, uint256 pos, uint8 v) external {
        Claim memory c = Fx.sample(sampleSeed % Fx.sampleCount());
        uint256 value = bondRequired;
        uint256 w = what % 5;
        if (w == 0) Model.setInput(c.x0, pos % 784, v | 0x80);
        else if (w == 1) c.x0[24] |= uint256(v == 0 ? 1 : v) << (8 * (pos % 16));
        else if (w == 2) Model.setByte(c.hidden, pos % 32, v | 0x80);
        else if (w == 3) c.cls = uint8(10 + v % 246);
        else value = pos % bondRequired;
        address who = actors[actorSeed % 5];
        vm.deal(who, who.balance + value);
        vm.prank(who);
        try arbiter.claim{value: value}(c.x0, c.hidden, c.logits, c.cls) {
            violation("a malformed claim was accepted");
        } catch {
            refused++;
        }
    }

    /// A correct challenge (committed vectors, the right row, bias and proof) of a random neuron
    /// or the class of a random claim, open or not, early or late.
    function challenge(uint256 actorSeed, uint256 claimSeed, uint256 kind, uint256 j) external {
        if (recs.length == 0) return;
        kind %= 3;
        challengeChecked(claimSeed % recs.length, actors[actorSeed % 5], kind, j % (kind == 0 ? 32 : 10));
    }

    /// The challenge the spec's walk names for the lie of the first open claim from a random one
    /// (the class challenge for an honest claim), so that slashes are frequent.
    function challengeNamed(uint256 actorSeed, uint256 claimSeed) external {
        if (recs.length == 0) return;
        uint256 id = claimSeed % recs.length;
        Rec memory r = recs[openFrom(id)];
        id = openFrom(id);
        uint256 kind = r.kind == 1 || r.kind == 4 ? 0 : (r.kind == 2 ? 1 : 2);
        challengeChecked(id, actors[actorSeed % 5], kind, r.j);
    }

    /// A challenge with one corrupted piece: another neuron's leaf, a flipped bit in the row or
    /// the bias, a cut proof, a substituted input, hidden vector or logits. Always refused.
    function challengeCorrupt(uint256 actorSeed, uint256 claimSeed, uint256 how, uint256 j, uint256 bit) external {
        if (recs.length == 0) return;
        uint256 id = claimSeed % recs.length;
        Claim memory c = build(recs[id]);
        bool layer2 = (how >> 8) & 1 == 1;
        uint256 n = layer2 ? 10 : 32;
        j %= n;
        uint256 k = (j + 1 + bit % (n - 1)) % n; // another neuron
        uint256[25] memory row1 = Fx.row1(layer2 ? 0 : j);
        uint256[1] memory row2 = Fx.row2(layer2 ? j : 0);
        int32 bias = layer2 ? Fx.bias2(j) : Fx.bias1(j);
        bytes32[] memory proof = Fx.proof(layer2 ? 2 : 1, j);
        uint256 h = how % 6;
        if (h == 0) {
            if (layer2) row2 = Fx.row2(k);
            else row1 = Fx.row1(k);
            bias = layer2 ? Fx.bias2(k) : Fx.bias1(k);
            proof = Fx.proof(layer2 ? 2 : 1, k);
        } else if (h == 1) {
            if (layer2) row2[0] ^= uint256(1) << (bit % 256);
            else row1[(bit >> 8) % 25] ^= uint256(1) << (bit % 256);
        } else if (h == 2) {
            bias = int32(uint32(bias) ^ (uint32(1) << uint32(bit % 32)));
        } else if (h == 3) {
            assembly ("memory-safe") {
                mstore(proof, sub(mload(proof), 1))
            }
        } else if (h == 4) {
            if (layer2) c.hidden[0] ^= uint256(1) << (bit % 256);
            else c.x0[(bit >> 8) % 24] ^= uint256(1) << (bit % 256);
        } else {
            c.logits[bit % 10] ^= int32(uint32(1) << uint32((bit >> 8) % 32));
        }
        bytes memory data = h == 5
            ? abi.encodeCall(IArbiter.challengeClass, (id, c.logits))
            : layer2
                ? abi.encodeCall(IArbiter.challengeNeuron2, (id, j, c.hidden, c.logits, row2, bias, proof))
                : abi.encodeCall(IArbiter.challengeNeuron1, (id, j, c.x0, c.hidden, row1, bias, proof));
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

    /// The first open claim at or after `id` (cyclically), or `id` if none is open.
    function openFrom(uint256 id) internal view returns (uint256) {
        for (uint256 k = 0; k < recs.length; k++) {
            uint256 i = (id + k) % recs.length;
            if (recs[i].status == 1) return i;
        }
        return id;
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

/// Invariant tests P1 and P4 of docs/spec-dispute.md section 9, and the withdrawal rule, over the
/// handler above. One campaign checks every property after every call.
contract ArbiterInvariantTest is ArbiterUser {
    struct FuzzSelector {
        address addr;
        bytes4[] selectors;
    }

    ArbiterHandler internal handler;
    IArbiter internal arbiter;

    function setUp() public {
        Fx.install();
        vm.warp(1_000_000);
        arbiter = deployArbiter(1 ether, 1 hours);
        handler = new ArbiterHandler(arbiter, 1 ether, 1 hours);
    }

    function targetContracts() public view returns (address[] memory t) {
        t = new address[](1);
        t[0] = address(handler);
    }

    function targetSelectors() public view returns (FuzzSelector[] memory t) {
        bytes4[] memory s = new bytes4[](8);
        s[0] = ArbiterHandler.claim.selector;
        s[1] = ArbiterHandler.claimMalformed.selector;
        s[2] = ArbiterHandler.challenge.selector;
        s[3] = ArbiterHandler.challengeNamed.selector;
        s[4] = ArbiterHandler.challengeCorrupt.selector;
        s[5] = ArbiterHandler.warp.selector;
        s[6] = ArbiterHandler.finalize.selector;
        s[7] = ArbiterHandler.withdraw.selector;
        t = new FuzzSelector[](1);
        t[0] = FuzzSelector(address(handler), s);
    }

    /// P1: the balance equals the open bonds plus the unwithdrawn credits plus the burned halves
    /// of slashed bonds, both as the ghost accounts say and as the arbiter's own views say. P4:
    /// every claim's status is the one the spec predicts (so it only moved Open -> Slashed or
    /// Open -> Final and never again), its other fields never changed, and each bond was shared
    /// out exactly once (bonded = open + credited + burned, credited = outstanding + withdrawn). Withdrawals: nobody withdrew more than
    /// they were credited, and the arbiter's credits equal the ghost credits. Every call the
    /// handler made had the outcome the spec predicts (`violations`).
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
            ArbiterHandler.Rec memory r = handler.rec(id);
            uint8 status = statusOf(arbiter, id);
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

    /// What this run of the campaign did, one line per run: claims, slashes by a neuron of layer 1,
    /// by a neuron of layer 2 and by the class, finalisations, payouts, refused calls. A campaign
    /// that never slashes or never pays would pass the invariant without testing it.
    function afterInvariant() public {
        vm.writeLine(
            "invariant-out/campaign.log",
            string.concat(
                vm.toString(handler.claimCount()),
                " ",
                vm.toString(handler.slashesBy(0)),
                " ",
                vm.toString(handler.slashesBy(1)),
                " ",
                vm.toString(handler.slashesBy(2)),
                " ",
                vm.toString(handler.finals()),
                " ",
                vm.toString(handler.payouts()),
                " ",
                vm.toString(handler.refused())
            )
        );
    }

    /// The handler itself, on a scripted sequence: it reaches every outcome (slash by each kind of
    /// challenge, finalisation, paid and refused withdrawals, refused corrupt and malformed
    /// calls) without a violation, so the campaign above is not vacuous.
    function test_handler_reaches_every_outcome() public {
        handler.claim(0, 0, 1 | (5 << 8), 0); // 0: hidden 5 wrong
        handler.claim(1, 1, 2 | (3 << 8) | (77 << 16), 0); // 1: logit 3 wrong
        handler.claim(2, 2, 3, 0); // 2: class wrong
        handler.claim(3, 3, 4 | (7 << 8), 0); // 3: hidden 7 wrong, logits and class consistent with it
        handler.claim(4, 4, 0, 0); // 4: honest, claimed by the actor that refuses ETH
        handler.claim(0, 5, 0, 5); // 5: honest
        handler.challenge(1, 3, 1, 0); // layer 2 on claim 3: consistent, refused
        handler.challenge(1, 3, 2, 0); // class on claim 3: consistent, refused
        handler.challengeNamed(1, 0);
        handler.challengeNamed(2, 1);
        handler.challengeNamed(3, 2);
        handler.challengeNamed(4, 3); // the actor that refuses ETH collects claim 3's bond
        handler.challengeCorrupt(0, 4, 0, 1, 2);
        handler.challengeCorrupt(0, 5, 3 | (1 << 8), 1, 2);
        handler.claimMalformed(0, 0, 0, 10, 1);
        handler.claimMalformed(0, 0, 4, 10, 1);
        handler.withdraw(0); // no credit yet
        handler.warp(0, 4, 2); // to claim 4's deadline
        handler.challenge(0, 4, 2, 0); // too late (and honest)
        handler.finalize(0, 4, true);
        handler.finalize(0, 5, true);
        handler.finalize(0, 0, true); // already slashed
        handler.withdraw(4); // refuses ETH: TransferFailed, credit kept
        handler.withdraw(1);
        handler.withdraw(0); // claim 5's bond
        handler.withdraw(0); // nothing left
        require(handler.violations() == 0, handler.firstViolation());
        require(handler.slashes() == 4 && handler.finals() == 2 && handler.payouts() == 2, "outcomes");
        require(handler.refused() == 11, "refusals");
        invariant_P1_P4_ledger_lifecycle_and_withdrawals();
        // Four slashes burned 0.5 ether each; two challengers hold 0.5 ether each, and the actor that
        // refuses ETH holds its 0.5 ether reward plus claim 4's whole bond.
        require(arbiter.burned() == 2 ether, "four half bonds burned");
        require(
            address(arbiter).balance == 4.5 ether, "2 ether burned, two credits of 0.5 ether, the stuck actor's 1.5 ether"
        );
    }

    /// Negative control: against an arbiter deployed with another model root, the handler's
    /// predictions fail (a proven challenge is refused) and the invariant reports it.
    function test_negative_control_wrong_root_is_detected() public {
        IArbiter other =
            IArbiter(FeDeployer.deploy("Arbiter", abi.encode(uint256(keccak256("another model")), 1 ether, 1 hours)));
        arbiter = other;
        handler = new ArbiterHandler(other, 1 ether, 1 hours);
        handler.claim(0, 0, 1 | (5 << 8), 0);
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
        require(!ok && keccak256(reason) == keccak256(abi.encodeWithSignature("Error(string)", "P1: balance != open bonds + credits + burned")), "P1 did not fail");
    }
}
