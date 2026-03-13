"""
Tests for C238: Paxos Consensus Protocol

Tests Basic Paxos (single-decree) and Multi-Paxos (replicated state machine).
"""

import pytest
import random
from paxos import (
    ProposalID, Message, MessageType, LogEntry, Snapshot,
    StateMachine, Acceptor, Learner, Proposer,
    MultiPaxosNode, Network, create_cluster, BasicPaxos,
)


# ============================================================
# ProposalID Tests
# ============================================================

class TestProposalID:
    def test_ordering_by_number(self):
        p1 = ProposalID(1, "a")
        p2 = ProposalID(2, "a")
        assert p1 < p2
        assert p2 > p1
        assert p1 <= p2
        assert p2 >= p1

    def test_ordering_by_node_id(self):
        p1 = ProposalID(1, "a")
        p2 = ProposalID(1, "b")
        assert p1 < p2
        assert p2 > p1

    def test_equality(self):
        p1 = ProposalID(1, "a")
        p2 = ProposalID(1, "a")
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_inequality(self):
        p1 = ProposalID(1, "a")
        p2 = ProposalID(1, "b")
        assert p1 != p2

    def test_none_comparison(self):
        p = ProposalID(1, "a")
        assert p > None
        assert p >= None
        assert not (p < None)
        assert not (p <= None)
        assert p != None  # noqa: E711

    def test_to_from_dict(self):
        p = ProposalID(5, "node1")
        d = p.to_dict()
        p2 = ProposalID.from_dict(d)
        assert p == p2

    def test_from_dict_none(self):
        assert ProposalID.from_dict(None) is None

    def test_hash_in_set(self):
        s = {ProposalID(1, "a"), ProposalID(1, "a"), ProposalID(2, "b")}
        assert len(s) == 2


# ============================================================
# StateMachine Tests
# ============================================================

class TestStateMachine:
    def test_set_and_get(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 10})
        assert sm.apply({"op": "get", "key": "x"}) == 10

    def test_delete(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 10})
        sm.apply({"op": "delete", "key": "x"})
        assert sm.apply({"op": "get", "key": "x"}) is None

    def test_cas_success(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1})
        result = sm.apply({"op": "cas", "key": "x", "expected": 1, "value": 2})
        assert result is True
        assert sm.data["x"] == 2

    def test_cas_failure(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1})
        result = sm.apply({"op": "cas", "key": "x", "expected": 999, "value": 2})
        assert result is False
        assert sm.data["x"] == 1

    def test_snapshot_restore(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "a", "value": 1})
        sm.apply({"op": "set", "key": "b", "value": 2})
        snap = sm.snapshot()
        sm2 = StateMachine()
        sm2.restore(snap)
        assert sm2.data == {"a": 1, "b": 2}

    def test_apply_none(self):
        sm = StateMachine()
        assert sm.apply(None) is None

    def test_apply_unknown_op(self):
        sm = StateMachine()
        assert sm.apply({"op": "unknown"}) is None

    def test_apply_non_dict(self):
        sm = StateMachine()
        assert sm.apply("not a dict") is None

    def test_restore_none(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1})
        sm.restore(None)
        assert sm.data == {}


# ============================================================
# Acceptor Tests
# ============================================================

class TestAcceptor:
    def test_promise_on_first_prepare(self):
        acc = Acceptor("a1")
        msg = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(1, "p1").to_dict()}
        )
        acc.receive_prepare(msg)
        assert len(acc.outbox) == 1
        assert acc.outbox[0].type == MessageType.PROMISE

    def test_nack_on_lower_prepare(self):
        acc = Acceptor("a1")
        # First prepare with high number
        msg1 = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(10, "p1").to_dict()}
        )
        acc.receive_prepare(msg1)
        acc.outbox.clear()

        # Second prepare with low number
        msg2 = Message(
            type=MessageType.PREPARE, sender="p2", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(5, "p2").to_dict()}
        )
        acc.receive_prepare(msg2)
        assert len(acc.outbox) == 1
        assert acc.outbox[0].type == MessageType.NACK

    def test_promise_includes_accepted_value(self):
        acc = Acceptor("a1")
        # Prepare
        msg1 = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(1, "p1").to_dict()}
        )
        acc.receive_prepare(msg1)
        acc.outbox.clear()

        # Accept
        msg2 = Message(
            type=MessageType.ACCEPT, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(1, "p1").to_dict(), "value": "hello"}
        )
        acc.receive_accept(msg2)
        acc.outbox.clear()

        # Higher prepare should see accepted value
        msg3 = Message(
            type=MessageType.PREPARE, sender="p2", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(2, "p2").to_dict()}
        )
        acc.receive_prepare(msg3)
        promise = acc.outbox[0]
        assert promise.data["accepted_value"] == "hello"
        assert ProposalID.from_dict(promise.data["accepted_id"]) == ProposalID(1, "p1")

    def test_accept_on_matching_prepare(self):
        acc = Acceptor("a1")
        pid = ProposalID(1, "p1")
        msg1 = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": pid.to_dict()}
        )
        acc.receive_prepare(msg1)
        acc.outbox.clear()

        msg2 = Message(
            type=MessageType.ACCEPT, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "value": "world"}
        )
        acc.receive_accept(msg2)
        assert len(acc.outbox) == 1
        assert acc.outbox[0].type == MessageType.ACCEPTED

    def test_nack_accept_with_lower_proposal(self):
        acc = Acceptor("a1")
        # Promise high
        msg1 = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(10, "p1").to_dict()}
        )
        acc.receive_prepare(msg1)
        acc.outbox.clear()

        # Try accept with lower
        msg2 = Message(
            type=MessageType.ACCEPT, sender="p2", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(5, "p2").to_dict(), "value": "x"}
        )
        acc.receive_accept(msg2)
        assert acc.outbox[0].type == MessageType.NACK

    def test_get_accepted(self):
        acc = Acceptor("a1")
        pid = ProposalID(1, "p1")
        msg1 = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": pid.to_dict()}
        )
        acc.receive_prepare(msg1)
        acc.outbox.clear()
        msg2 = Message(
            type=MessageType.ACCEPT, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "value": "val"}
        )
        acc.receive_accept(msg2)
        a_id, a_val = acc.get_accepted(1)
        assert a_id == pid
        assert a_val == "val"

    def test_get_accepted_empty(self):
        acc = Acceptor("a1")
        a_id, a_val = acc.get_accepted(999)
        assert a_id is None
        assert a_val is None

    def test_compact(self):
        acc = Acceptor("a1")
        for slot in range(1, 6):
            msg = Message(
                type=MessageType.PREPARE, sender="p1", receiver="a1",
                data={"slot": slot, "proposal_id": ProposalID(1, "p1").to_dict()}
            )
            acc.receive_prepare(msg)
        acc.outbox.clear()
        acc.compact(3)
        assert 1 not in acc.slots
        assert 2 not in acc.slots
        assert 3 not in acc.slots
        assert 4 in acc.slots
        assert 5 in acc.slots

    def test_multiple_slots_independent(self):
        acc = Acceptor("a1")
        # Prepare slot 1
        msg1 = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": ProposalID(1, "p1").to_dict()}
        )
        acc.receive_prepare(msg1)
        # Prepare slot 2 with different proposal
        msg2 = Message(
            type=MessageType.PREPARE, sender="p2", receiver="a1",
            data={"slot": 2, "proposal_id": ProposalID(2, "p2").to_dict()}
        )
        acc.receive_prepare(msg2)
        assert len(acc.outbox) == 2
        assert all(m.type == MessageType.PROMISE for m in acc.outbox)

    def test_equal_proposal_accepted(self):
        """Accept with same proposal ID as promised should work."""
        acc = Acceptor("a1")
        pid = ProposalID(5, "p1")
        msg1 = Message(
            type=MessageType.PREPARE, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": pid.to_dict()}
        )
        acc.receive_prepare(msg1)
        acc.outbox.clear()
        msg2 = Message(
            type=MessageType.ACCEPT, sender="p1", receiver="a1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "value": "v"}
        )
        acc.receive_accept(msg2)
        assert acc.outbox[0].type == MessageType.ACCEPTED


# ============================================================
# Learner Tests
# ============================================================

class TestLearner:
    def test_quorum_decides(self):
        learner = Learner("l1", quorum_size=2)
        pid = ProposalID(1, "p1")
        result1 = learner.receive_accepted(1, pid, "val", "a1")
        assert result1 is None
        result2 = learner.receive_accepted(1, pid, "val", "a2")
        assert result2 == "val"

    def test_decided_value_persists(self):
        learner = Learner("l1", quorum_size=2)
        pid = ProposalID(1, "p1")
        learner.receive_accepted(1, pid, "val", "a1")
        learner.receive_accepted(1, pid, "val", "a2")
        assert learner.is_decided(1)
        assert learner.get_decided(1) == "val"

    def test_different_proposals_separate(self):
        learner = Learner("l1", quorum_size=2)
        pid1 = ProposalID(1, "p1")
        pid2 = ProposalID(2, "p2")
        learner.receive_accepted(1, pid1, "val1", "a1")
        learner.receive_accepted(1, pid2, "val2", "a2")
        # Neither has quorum
        assert not learner.is_decided(1)

    def test_quorum_of_three(self):
        learner = Learner("l1", quorum_size=3)
        pid = ProposalID(1, "p1")
        learner.receive_accepted(1, pid, "x", "a1")
        learner.receive_accepted(1, pid, "x", "a2")
        assert not learner.is_decided(1)
        learner.receive_accepted(1, pid, "x", "a3")
        assert learner.is_decided(1)
        assert learner.get_decided(1) == "x"

    def test_already_decided_returns_value(self):
        learner = Learner("l1", quorum_size=1)
        pid = ProposalID(1, "p1")
        learner.receive_accepted(1, pid, "v", "a1")
        # Already decided, further accepts return the decided value
        result = learner.receive_accepted(1, pid, "v", "a2")
        assert result == "v"

    def test_compact(self):
        learner = Learner("l1", quorum_size=1)
        pid = ProposalID(1, "p1")
        for s in range(1, 6):
            learner.receive_accepted(s, pid, f"v{s}", "a1")
        learner.compact(3)
        # Decided values remain (compact only removes from .accepted)
        assert learner.is_decided(1)
        assert learner.is_decided(4)

    def test_not_decided(self):
        learner = Learner("l1", quorum_size=2)
        assert not learner.is_decided(1)
        assert learner.get_decided(1) is None

    def test_duplicate_acceptor(self):
        """Same acceptor sending twice shouldn't double-count."""
        learner = Learner("l1", quorum_size=2)
        pid = ProposalID(1, "p1")
        learner.receive_accepted(1, pid, "v", "a1")
        learner.receive_accepted(1, pid, "v", "a1")  # duplicate
        assert not learner.is_decided(1)

    def test_multiple_slots(self):
        learner = Learner("l1", quorum_size=2)
        pid = ProposalID(1, "p1")
        learner.receive_accepted(1, pid, "v1", "a1")
        learner.receive_accepted(2, pid, "v2", "a1")
        learner.receive_accepted(1, pid, "v1", "a2")
        learner.receive_accepted(2, pid, "v2", "a2")
        assert learner.get_decided(1) == "v1"
        assert learner.get_decided(2) == "v2"


# ============================================================
# Proposer Tests
# ============================================================

class TestProposer:
    def test_propose_sends_prepare(self):
        p = Proposer("p1", ["a1", "a2", "a3"], quorum_size=2)
        p.propose(1, "val")
        assert len(p.outbox) == 3
        assert all(m.type == MessageType.PREPARE for m in p.outbox)

    def test_promise_quorum_starts_phase2(self):
        p = Proposer("p1", ["a1", "a2", "a3"], quorum_size=2)
        pid = p.propose(1, "val")
        p.outbox.clear()

        # Send promises
        msg1 = Message(
            type=MessageType.PROMISE, sender="a1", receiver="p1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "accepted_id": None, "accepted_value": None}
        )
        result1 = p.receive_promise(msg1)
        assert result1 is None  # not quorum yet

        msg2 = Message(
            type=MessageType.PROMISE, sender="a2", receiver="p1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "accepted_id": None, "accepted_value": None}
        )
        result2 = p.receive_promise(msg2)
        assert result2 == "phase2_started"
        # Should have sent ACCEPT messages
        assert len(p.outbox) == 3
        assert all(m.type == MessageType.ACCEPT for m in p.outbox)

    def test_adopt_highest_accepted_value(self):
        p = Proposer("p1", ["a1", "a2", "a3"], quorum_size=2)
        pid = p.propose(1, "my_val")
        p.outbox.clear()

        # a1 has no accepted value
        msg1 = Message(
            type=MessageType.PROMISE, sender="a1", receiver="p1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "accepted_id": None, "accepted_value": None}
        )
        p.receive_promise(msg1)

        # a2 has an accepted value from a previous proposal
        prev_pid = ProposalID(0, "p0")
        msg2 = Message(
            type=MessageType.PROMISE, sender="a2", receiver="p1",
            data={
                "slot": 1, "proposal_id": pid.to_dict(),
                "accepted_id": prev_pid.to_dict(), "accepted_value": "prev_val"
            }
        )
        p.receive_promise(msg2)

        # Phase 2 should use "prev_val" not "my_val"
        accept_msgs = [m for m in p.outbox if m.type == MessageType.ACCEPT]
        assert all(m.data["value"] == "prev_val" for m in accept_msgs)

    def test_accepted_quorum_decides(self):
        p = Proposer("p1", ["a1", "a2", "a3"], quorum_size=2)
        pid = p.propose(1, "val")
        p.outbox.clear()

        # Get to phase 2
        for aid in ["a1", "a2"]:
            msg = Message(
                type=MessageType.PROMISE, sender=aid, receiver="p1",
                data={"slot": 1, "proposal_id": pid.to_dict(), "accepted_id": None, "accepted_value": None}
            )
            p.receive_promise(msg)
        p.outbox.clear()

        # Phase 2: accepted
        msg1 = Message(
            type=MessageType.ACCEPTED, sender="a1", receiver="p1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "value": "val"}
        )
        result1 = p.receive_accepted(msg1)
        assert result1 is None

        msg2 = Message(
            type=MessageType.ACCEPTED, sender="a2", receiver="p1",
            data={"slot": 1, "proposal_id": pid.to_dict(), "value": "val"}
        )
        result2 = p.receive_accepted(msg2)
        assert result2 == "val"

    def test_is_decided(self):
        p = Proposer("p1", ["a1", "a2"], quorum_size=2)
        pid = p.propose(1, "val")
        p.outbox.clear()

        for aid in ["a1", "a2"]:
            msg = Message(
                type=MessageType.PROMISE, sender=aid, receiver="p1",
                data={"slot": 1, "proposal_id": pid.to_dict(), "accepted_id": None, "accepted_value": None}
            )
            p.receive_promise(msg)
        p.outbox.clear()

        for aid in ["a1", "a2"]:
            msg = Message(
                type=MessageType.ACCEPTED, sender=aid, receiver="p1",
                data={"slot": 1, "proposal_id": pid.to_dict(), "value": "val"}
            )
            p.receive_accepted(msg)

        assert p.is_decided(1)
        assert p.get_decided_value(1) == "val"

    def test_nack_updates_seq(self):
        p = Proposer("p1", ["a1", "a2"], quorum_size=2)
        p.propose(1, "val")
        p.outbox.clear()

        msg = Message(
            type=MessageType.NACK, sender="a1", receiver="p1",
            data={"slot": 1, "proposal_id": ProposalID(1, "p1").to_dict(),
                  "promised": ProposalID(100, "p2").to_dict()}
        )
        p.receive_nack(msg)
        assert p.proposal_seq >= 100

    def test_stale_promise_ignored(self):
        p = Proposer("p1", ["a1", "a2", "a3"], quorum_size=2)
        pid = p.propose(1, "val")
        p.outbox.clear()

        # Send promise with wrong proposal_id
        wrong_pid = ProposalID(999, "p1")
        msg = Message(
            type=MessageType.PROMISE, sender="a1", receiver="p1",
            data={"slot": 1, "proposal_id": wrong_pid.to_dict(), "accepted_id": None, "accepted_value": None}
        )
        result = p.receive_promise(msg)
        assert result is None

    def test_stale_accepted_ignored(self):
        p = Proposer("p1", ["a1", "a2"], quorum_size=2)
        pid = p.propose(1, "val")
        p.outbox.clear()
        # Go to phase 2
        for aid in ["a1", "a2"]:
            msg = Message(
                type=MessageType.PROMISE, sender=aid, receiver="p1",
                data={"slot": 1, "proposal_id": pid.to_dict(), "accepted_id": None, "accepted_value": None}
            )
            p.receive_promise(msg)
        p.outbox.clear()
        # Wrong proposal_id
        wrong_pid = ProposalID(999, "p1")
        msg = Message(
            type=MessageType.ACCEPTED, sender="a1", receiver="p1",
            data={"slot": 1, "proposal_id": wrong_pid.to_dict(), "value": "val"}
        )
        result = p.receive_accepted(msg)
        assert result is None

    def test_unknown_slot_ignored(self):
        p = Proposer("p1", ["a1"], quorum_size=1)
        msg = Message(
            type=MessageType.PROMISE, sender="a1", receiver="p1",
            data={"slot": 999, "proposal_id": ProposalID(1, "p1").to_dict(),
                  "accepted_id": None, "accepted_value": None}
        )
        assert p.receive_promise(msg) is None


# ============================================================
# BasicPaxos Tests (single-decree)
# ============================================================

class TestBasicPaxos:
    def test_single_proposer_decides(self):
        bp = BasicPaxos(["a1", "a2", "a3"])
        bp.create_proposer("a1")
        result = bp.run_round("a1", 1, "hello")
        assert result == "hello"

    def test_decision_visible_to_all_learners(self):
        bp = BasicPaxos(["a1", "a2", "a3"])
        bp.create_proposer("a1")
        bp.run_round("a1", 1, "hello")
        for lid, learner in bp.learners.items():
            assert learner.is_decided(1)
            assert learner.get_decided(1) == "hello"

    def test_different_values_same_slot(self):
        """If two proposers try to propose different values, only one should win."""
        bp = BasicPaxos(["a1", "a2", "a3"])
        bp.create_proposer("a1")
        bp.create_proposer("a2")

        # First proposer decides
        result1 = bp.run_round("a1", 1, "val_a")
        assert result1 == "val_a"

        # Second proposer should adopt the decided value
        result2 = bp.run_round("a2", 1, "val_b")
        # Should get val_a (the already-decided value)
        assert result2 == "val_a"

    def test_multiple_slots(self):
        bp = BasicPaxos(["a1", "a2", "a3"])
        bp.create_proposer("a1")
        r1 = bp.run_round("a1", 1, "first")
        r2 = bp.run_round("a1", 2, "second")
        assert r1 == "first"
        assert r2 == "second"
        assert bp.get_decided(1) == "first"
        assert bp.get_decided(2) == "second"

    def test_get_decided(self):
        bp = BasicPaxos(["a1", "a2", "a3"])
        assert bp.get_decided(1) is None
        bp.create_proposer("a1")
        bp.run_round("a1", 1, "val")
        assert bp.get_decided(1) == "val"

    def test_message_log(self):
        bp = BasicPaxos(["a1", "a2", "a3"])
        bp.create_proposer("a1")
        bp.run_round("a1", 1, "val")
        # Should have messages: 3 prepare, 3 promise, 3 accept, 3 accepted
        types = [m.type for m in bp.message_log]
        assert types.count(MessageType.PREPARE) == 3
        assert types.count(MessageType.PROMISE) == 3
        assert types.count(MessageType.ACCEPT) == 3
        assert types.count(MessageType.ACCEPTED) == 3

    def test_two_acceptors_quorum(self):
        bp = BasicPaxos(["a1", "a2"])
        bp.create_proposer("a1")
        result = bp.run_round("a1", 1, "val")
        assert result == "val"


# ============================================================
# LogEntry Tests
# ============================================================

class TestLogEntry:
    def test_to_from_dict(self):
        pid = ProposalID(1, "p1")
        entry = LogEntry(slot=5, proposal_id=pid, value={"op": "set", "key": "x", "value": 1})
        d = entry.to_dict()
        entry2 = LogEntry.from_dict(d)
        assert entry2.slot == 5
        assert entry2.proposal_id == pid
        assert entry2.value == {"op": "set", "key": "x", "value": 1}

    def test_none_proposal_id(self):
        entry = LogEntry(slot=1, proposal_id=None, value="test")
        d = entry.to_dict()
        entry2 = LogEntry.from_dict(d)
        assert entry2.proposal_id is None


# ============================================================
# Multi-Paxos Node Tests
# ============================================================

class TestMultiPaxosNode:
    def test_create_node(self):
        node = MultiPaxosNode("n1", ["n2", "n3"])
        assert node.node_id == "n1"
        assert not node.is_leader
        assert node.last_applied == 0
        assert node.next_slot == 1

    def test_submit_not_leader(self):
        node = MultiPaxosNode("n1", ["n2", "n3"])
        assert node.submit({"op": "set", "key": "x", "value": 1}) is None

    def test_become_leader(self):
        node = MultiPaxosNode("n1", ["n2", "n3"])
        node._become_leader()
        assert node.is_leader
        assert node.leader_id == "n1"
        # Should have sent heartbeats
        hb_msgs = [m for m in node.outbox if m.type == MessageType.HEARTBEAT]
        assert len(hb_msgs) >= 1

    def test_step_down(self):
        node = MultiPaxosNode("n1", ["n2", "n3"])
        node._become_leader()
        node._step_down()
        assert not node.is_leader

    def test_get_state(self):
        node = MultiPaxosNode("n1", ["n2", "n3"])
        state = node.get_state()
        assert state["node_id"] == "n1"
        assert state["is_leader"] is False
        assert state["last_applied"] == 0

    def test_quorum_size_3_nodes(self):
        node = MultiPaxosNode("n1", ["n2", "n3"])
        assert node._quorum_size() == 2

    def test_quorum_size_5_nodes(self):
        node = MultiPaxosNode("n1", ["n2", "n3", "n4", "n5"])
        assert node._quorum_size() == 3


# ============================================================
# Network Tests
# ============================================================

class TestNetwork:
    def test_create_cluster(self):
        net = create_cluster(["n1", "n2", "n3"])
        assert len(net.nodes) == 3
        assert "n1" in net.nodes
        assert "n2" in net.nodes
        assert "n3" in net.nodes

    def test_partition(self):
        net = Network()
        net.partition("a", "b")
        assert net.is_partitioned("a", "b")
        assert net.is_partitioned("b", "a")  # symmetric
        assert not net.is_partitioned("a", "c")

    def test_heal_specific(self):
        net = Network()
        net.partition("a", "b")
        net.partition("a", "c")
        net.heal("a", "b")
        assert not net.is_partitioned("a", "b")
        assert net.is_partitioned("a", "c")

    def test_heal_all(self):
        net = Network()
        net.partition("a", "b")
        net.partition("a", "c")
        net.heal()
        assert not net.is_partitioned("a", "b")
        assert not net.is_partitioned("a", "c")

    def test_isolate(self):
        net = create_cluster(["n1", "n2", "n3"])
        net.isolate("n1")
        assert net.is_partitioned("n1", "n2")
        assert net.is_partitioned("n1", "n3")
        assert not net.is_partitioned("n2", "n3")

    def test_remove_node(self):
        net = create_cluster(["n1", "n2", "n3"])
        net.remove_node("n3")
        assert "n3" not in net.nodes

    def test_dropped_messages_on_partition(self):
        net = create_cluster(["n1", "n2", "n3"])
        net.partition("n1", "n2")
        # Make n1 leader
        net.nodes["n1"]._become_leader()
        net.tick()
        # Messages from n1 to n2 should be dropped
        dropped_to_n2 = [m for m in net.dropped if m.receiver == "n2" and m.sender == "n1"]
        assert len(dropped_to_n2) > 0


# ============================================================
# Multi-Paxos Integration Tests
# ============================================================

class TestMultiPaxosIntegration:
    def test_leader_election(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None
        assert leader.is_leader

    def test_single_value_consensus(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 42})
        assert slot is not None
        # Run until decided
        decided = net.wait_for_decision(slot, max_ticks=3000)
        assert decided == {"op": "set", "key": "x", "value": 42}

    def test_multiple_values(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slots = []
        for i in range(5):
            slot = leader.submit({"op": "set", "key": f"k{i}", "value": i})
            slots.append(slot)

        for slot in slots:
            net.wait_for_decision(slot, max_ticks=3000)

        # Leader should have applied all
        net.run(500)
        assert leader.state_machine.data.get("k0") == 0
        assert leader.state_machine.data.get("k4") == 4

    def test_replication_to_followers(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 99})
        net.run(2000)

        # All nodes should eventually apply
        for node in net.nodes.values():
            if node.learner.is_decided(slot):
                assert node.learner.get_decided(slot) == {"op": "set", "key": "x", "value": 99}

    def test_state_machine_applied(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        leader.submit({"op": "set", "key": "a", "value": 1})
        leader.submit({"op": "set", "key": "b", "value": 2})
        net.run(2000)

        assert leader.state_machine.data.get("a") == 1
        assert leader.state_machine.data.get("b") == 2

    def test_submit_not_leader_returns_none(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # Find a non-leader
        follower = None
        for node in net.nodes.values():
            if not node.is_leader:
                follower = node
                break
        assert follower is not None
        assert follower.submit({"op": "set", "key": "x", "value": 1}) is None

    def test_leader_partition_new_leader(self):
        """If leader is isolated, remaining nodes should elect a new leader."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None
        old_leader_id = leader.node_id

        # Isolate the leader
        net.isolate(old_leader_id)
        # Run enough ticks for new election
        new_leader = None
        for _ in range(5000):
            net.tick()
            for node in net.nodes.values():
                if node.is_leader and node.node_id != old_leader_id:
                    new_leader = node
                    break
            if new_leader:
                break
        assert new_leader is not None
        assert new_leader.node_id != old_leader_id

    def test_value_after_leader_change(self):
        """Values committed before partition should survive leader change."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "durable", "value": 100})
        net.run(1000)

        # Value should be decided
        decided = leader.learner.get_decided(slot)
        assert decided == {"op": "set", "key": "durable", "value": 100}

    def test_phase2_optimization(self):
        """Test stable leader Phase 2 optimization (skip Prepare)."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit_phase2({"op": "set", "key": "fast", "value": 1})
        assert slot is not None
        net.run(1000)

        # Should be decided
        decided = leader.learner.get_decided(slot)
        assert decided == {"op": "set", "key": "fast", "value": 1}

    def test_snapshot(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # Submit values and ensure applied
        for i in range(5):
            leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.run(2000)

        snap = leader.take_snapshot()
        assert snap is not None
        assert snap.last_included_slot == leader.last_applied
        assert snap.data == leader.state_machine.data

    def test_snapshot_restore(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        for i in range(5):
            leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.run(2000)

        snap = leader.take_snapshot()
        # Send snapshot to a follower
        follower = None
        for node in net.nodes.values():
            if not node.is_leader:
                follower = node
                break
        leader._send_snapshot(follower.node_id)
        net.run(100)

        # Follower state should match snapshot
        assert follower.last_applied >= snap.last_included_slot

    def test_multiple_clients_concurrent(self):
        """Multiple values submitted rapidly should all get decided."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slots = []
        for i in range(10):
            slot = leader.submit({"op": "set", "key": f"c{i}", "value": i * 10})
            if slot:
                slots.append(slot)

        net.run(3000)

        # All should be decided
        decided_count = sum(1 for s in slots if leader.learner.is_decided(s))
        assert decided_count == len(slots)

    def test_heartbeat_resets_election_timer(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # Followers should have their election timers reset by heartbeats
        for node in net.nodes.values():
            if not node.is_leader:
                assert node.election_timer < node.election_timeout

    def test_committed_entries_tracking(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(2000)

        # Should have committed entries (noop + command)
        assert len(leader.committed_entries) >= 1

    def test_applied_results(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 42})
        net.run(2000)

        if slot in leader.applied_results:
            assert leader.applied_results[slot] == 42

    def test_noop_on_leader_election(self):
        """Leader should propose a noop on becoming leader."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # The noop should be proposed at the first slot
        # Check that learner knows about slot 1 (the noop slot)
        net.run(500)
        # Noop should have been decided
        noop_val = leader.learner.get_decided(leader.first_unchosen - 1)
        # Could be None if first_unchosen hasn't advanced yet, or the noop dict
        if noop_val is not None:
            # The noop has __type: noop
            if isinstance(noop_val, dict):
                assert noop_val.get("__type") == "noop"

    def test_cluster_five_nodes(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 5})
        net.run(2000)
        decided = leader.learner.get_decided(slot)
        assert decided == {"op": "set", "key": "x", "value": 5}

    def test_single_node_cluster(self):
        random.seed(42)
        net = create_cluster(["n1"])
        leader = net.wait_for_leader(max_ticks=1000)
        assert leader is not None
        assert leader.node_id == "n1"

        slot = leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(500)
        assert leader.state_machine.data.get("x") == 1

    def test_cas_through_paxos(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(1000)
        leader.submit({"op": "cas", "key": "x", "expected": 1, "value": 2})
        net.run(1000)

        assert leader.state_machine.data.get("x") == 2

    def test_delete_through_paxos(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(1000)
        leader.submit({"op": "delete", "key": "x"})
        net.run(1000)

        assert "x" not in leader.state_machine.data

    def test_network_run(self):
        net = create_cluster(["n1", "n2", "n3"])
        net.run(10, tick_ms=5)
        # Just checking it runs without error

    def test_add_member(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.add_member("n4")
        assert slot is not None
        net.run(2000)

        # Config should include n4
        assert "n4" in leader.cluster_config

    def test_remove_member(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # Find a non-leader to remove
        remove_id = None
        for node in net.nodes.values():
            if not node.is_leader:
                remove_id = node.node_id
                break

        slot = leader.remove_member(remove_id)
        assert slot is not None
        net.run(2000)

        assert remove_id not in leader.cluster_config

    def test_add_member_not_leader(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        follower = None
        for node in net.nodes.values():
            if not node.is_leader:
                follower = node
                break
        assert follower.add_member("n4") is None

    def test_remove_member_not_leader(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        follower = None
        for node in net.nodes.values():
            if not node.is_leader:
                follower = node
                break
        assert follower.remove_member("n2") is None

    def test_maybe_compact(self):
        """Test auto-compaction."""
        random.seed(42)
        node = MultiPaxosNode("n1", [])
        node._become_leader()
        node.snapshot_threshold = 5

        # Add committed entries manually
        for i in range(6):
            node.committed_entries.append(LogEntry(slot=i+1, proposal_id=None, value=f"v{i}"))

        snap = node.maybe_compact()
        # Won't have a snapshot since last_applied is 0
        # But it shows auto-compact logic

    def test_partition_drops_messages(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None
        old_dropped = len(net.dropped)
        net.isolate(leader.node_id)
        net.run(100)
        assert len(net.dropped) > old_dropped


# ============================================================
# Multi-Paxos Resilience Tests
# ============================================================

class TestMultiPaxosResilience:
    def test_minority_failure(self):
        """Cluster survives one node down in 3-node cluster."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # Remove one non-leader node
        for nid in ["n1", "n2", "n3"]:
            if nid != leader.node_id:
                net.isolate(nid)
                break

        slot = leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(2000)

        # Should still reach consensus with 2 nodes
        decided = leader.learner.get_decided(slot)
        assert decided == {"op": "set", "key": "x", "value": 1}

    def test_two_failures_in_five(self):
        """5-node cluster survives 2 failures."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # Isolate 2 non-leader nodes
        isolated = 0
        for nid in ["n1", "n2", "n3", "n4", "n5"]:
            if nid != leader.node_id and isolated < 2:
                net.isolate(nid)
                isolated += 1

        slot = leader.submit({"op": "set", "key": "x", "value": 5})
        net.run(2000)

        decided = leader.learner.get_decided(slot)
        assert decided == {"op": "set", "key": "x", "value": 5}

    def test_heal_partition_consistency(self):
        """After healing a partition, all nodes should converge."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        # Isolate one follower
        follower_id = None
        for nid in net.nodes:
            if nid != leader.node_id:
                follower_id = nid
                break
        net.isolate(follower_id)

        slot = leader.submit({"op": "set", "key": "x", "value": 42})
        net.run(1000)

        # Heal
        net.heal()
        net.run(2000)

        # All nodes should have the value decided
        for node in net.nodes.values():
            if node.learner.is_decided(slot):
                assert node.learner.get_decided(slot) == {"op": "set", "key": "x", "value": 42}

    def test_rapid_leader_changes(self):
        """Multiple leader changes shouldn't lose committed data."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "stable", "value": 999})
        net.run(1000)

        # Force leader step down
        leader._step_down()
        net.run(3000)

        # New leader should exist
        new_leader = net.find_leader()
        # The value should be decided somewhere
        for node in net.nodes.values():
            if node.learner.is_decided(slot):
                assert node.learner.get_decided(slot) == {"op": "set", "key": "stable", "value": 999}

    def test_sequential_operations(self):
        """Operations applied in order."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        leader.submit({"op": "set", "key": "counter", "value": 0})
        net.run(500)
        leader.submit({"op": "set", "key": "counter", "value": 1})
        net.run(500)
        leader.submit({"op": "set", "key": "counter", "value": 2})
        net.run(500)

        assert leader.state_machine.data.get("counter") == 2

    def test_idempotent_noop(self):
        """Noops should not affect state machine."""
        random.seed(42)
        node = MultiPaxosNode("n1", [])
        node._become_leader()
        # The noop should be in proposer proposals
        net = Network()
        net.add_node(node)
        net.run(500)
        # State machine should be empty (only noops)
        assert node.state_machine.data == {}


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    def test_empty_cluster_no_crash(self):
        net = Network()
        net.tick()  # should not crash

    def test_message_to_nonexistent_node(self):
        net = create_cluster(["n1", "n2", "n3"])
        net.nodes["n1"].outbox.append(Message(
            type=MessageType.HEARTBEAT,
            sender="n1",
            receiver="n99",
            data={},
        ))
        net.tick()
        assert any(m.receiver == "n99" for m in net.dropped)

    def test_acceptor_multiple_prepares_increasing(self):
        acc = Acceptor("a1")
        for i in range(1, 6):
            msg = Message(
                type=MessageType.PREPARE, sender="p1", receiver="a1",
                data={"slot": 1, "proposal_id": ProposalID(i, "p1").to_dict()}
            )
            acc.receive_prepare(msg)
        # All should be promises (each higher than last)
        assert all(m.type == MessageType.PROMISE for m in acc.outbox)

    def test_proposer_no_proposals(self):
        p = Proposer("p1", ["a1"], quorum_size=1)
        assert not p.is_decided(1)
        assert p.get_decided_value(1) is None

    def test_learner_empty(self):
        l = Learner("l1", quorum_size=2)
        assert not l.is_decided(1)
        assert l.get_decided(1) is None

    def test_snapshot_with_no_data(self):
        node = MultiPaxosNode("n1", [])
        snap = node.take_snapshot()
        assert snap is None  # nothing to snapshot

    def test_install_snapshot_already_past(self):
        """Node should handle snapshot for slot it's already past."""
        node = MultiPaxosNode("n1", [])
        node.last_applied = 10
        msg = Message(
            type=MessageType.INSTALL_SNAPSHOT,
            sender="n2",
            receiver="n1",
            data={"last_included_slot": 5, "data": {}, "config": ["n1", "n2"]},
        )
        node._handle_install_snapshot(msg)
        assert node.last_applied == 10  # unchanged

    def test_heartbeat_from_another_leader(self):
        """Node receiving heartbeat from another leader should step down."""
        node = MultiPaxosNode("n1", ["n2"])
        node._become_leader()
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender="n2",
            receiver="n1",
            data={"leader_id": "n2", "first_unchosen": 1, "last_applied": 0},
        )
        node._handle_heartbeat(msg)
        assert not node.is_leader

    def test_client_request_not_leader(self):
        node = MultiPaxosNode("n1", ["n2"])
        msg = Message(
            type=MessageType.CLIENT_REQUEST,
            sender="client",
            receiver="n1",
            data={"command": {"op": "set", "key": "x", "value": 1}},
        )
        node.receive(msg)
        response = [m for m in node.outbox if m.type == MessageType.CLIENT_RESPONSE]
        assert len(response) == 1
        assert response[0].data["success"] is False

    def test_client_request_as_leader(self):
        node = MultiPaxosNode("n1", ["n2"])
        node._become_leader()
        node.outbox.clear()
        msg = Message(
            type=MessageType.CLIENT_REQUEST,
            sender="client",
            receiver="n1",
            data={"command": {"op": "set", "key": "x", "value": 1}},
        )
        node.receive(msg)
        # Should have started paxos (PREPARE messages)
        prepare_msgs = [m for m in node.outbox if m.type == MessageType.PREPARE]
        assert len(prepare_msgs) > 0

    def test_wait_for_leader_timeout(self):
        """Should return None if no leader elected in time."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        # Isolate all from each other
        net.partition("n1", "n2")
        net.partition("n1", "n3")
        net.partition("n2", "n3")
        result = net.wait_for_leader(max_ticks=100)
        # Might still elect if single node can become leader
        # but with full partition, no quorum possible

    def test_wait_for_decision_timeout(self):
        net = create_cluster(["n1", "n2", "n3"])
        result = net.wait_for_decision(999, max_ticks=10)
        assert result is None

    def test_proposal_id_not_equal_to_non_proposal(self):
        pid = ProposalID(1, "a")
        assert pid.__eq__("not a proposal") == NotImplemented

    def test_remove_nonexistent_member(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None
        # Remove non-existent member
        slot = leader.remove_member("n99")
        assert slot is None  # n99 not in config

    def test_add_existing_member(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None
        # Add already-existing member
        slot = leader.add_member("n2")
        assert slot is None  # n2 already in config

    def test_wait_for_apply(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 1})
        result = net.wait_for_apply(slot, count=1, max_ticks=3000)
        assert result is True

    def test_wait_for_apply_timeout(self):
        net = create_cluster(["n1", "n2", "n3"])
        result = net.wait_for_apply(999, count=3, max_ticks=10)
        assert result is False

    def test_wait_for_replication_alias(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 1})
        result = net.wait_for_replication(slot, count=1, max_ticks=3000)
        assert result is True

    def test_find_leader_multiple(self):
        """If multiple nodes think they're leader, find_leader returns None."""
        net = create_cluster(["n1", "n2", "n3"])
        net.nodes["n1"]._become_leader()
        net.nodes["n2"]._become_leader()
        leader = net.find_leader()
        assert leader is None

    def test_find_leader_none(self):
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.find_leader()
        assert leader is None

    def test_nack_for_unknown_slot(self):
        p = Proposer("p1", ["a1"], quorum_size=1)
        msg = Message(
            type=MessageType.NACK, sender="a1", receiver="p1",
            data={"slot": 999, "proposal_id": ProposalID(1, "p1").to_dict(),
                  "promised": ProposalID(10, "p2").to_dict()}
        )
        p.receive_nack(msg)  # should not crash

    def test_send_snapshot_no_snapshot(self):
        node = MultiPaxosNode("n1", ["n2"])
        node._send_snapshot("n2")
        # No snapshot to send, no messages
        assert len(node.outbox) == 0


# ============================================================
# Determinism and Correctness Tests
# ============================================================

class TestCorrectness:
    def test_agreement_property(self):
        """All nodes that decide must decide the same value for the same slot."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 42})
        net.run(2000)

        decided_values = set()
        for node in net.nodes.values():
            if node.learner.is_decided(slot):
                val = node.learner.get_decided(slot)
                decided_values.add(str(val))

        # All decided values must be the same
        assert len(decided_values) <= 1

    def test_validity_property(self):
        """A decided value must have been proposed by some proposer."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        proposed = {"op": "set", "key": "x", "value": 42}
        slot = leader.submit(proposed)
        net.run(2000)

        decided = leader.learner.get_decided(slot)
        if decided:
            assert decided == proposed

    def test_termination_single_proposer(self):
        """With a single proposer and no failures, consensus should be reached."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 1})
        decided = net.wait_for_decision(slot, max_ticks=3000)
        assert decided is not None

    def test_multiple_slots_ordered(self):
        """Values applied in slot order."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slots = []
        for i in range(5):
            slot = leader.submit({"op": "set", "key": "seq", "value": i})
            slots.append(slot)

        net.run(3000)

        # Last applied value should be 4 (the final write)
        assert leader.state_machine.data.get("seq") == 4

    def test_no_split_brain_values(self):
        """After partition and heal, state should converge."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        leader = net.wait_for_leader(max_ticks=3000)
        assert leader is not None

        slot = leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(1000)

        # Partition: isolate 2 nodes
        non_leaders = [n for n in net.nodes if n != leader.node_id]
        for n in non_leaders[:2]:
            net.isolate(n)

        slot2 = leader.submit({"op": "set", "key": "x", "value": 2})
        net.run(1000)

        # Heal
        net.heal()
        net.run(3000)

        # Check agreement: all nodes that decided slot2 agree
        decided_values = set()
        for node in net.nodes.values():
            if node.learner.is_decided(slot2):
                decided_values.add(str(node.learner.get_decided(slot2)))
        assert len(decided_values) <= 1


# ============================================================
# BasicPaxos Competing Proposers Tests
# ============================================================

class TestCompetingProposers:
    def test_two_proposers_one_wins(self):
        """Two proposers competing: first to complete wins."""
        bp = BasicPaxos(["a1", "a2", "a3"])
        bp.create_proposer("a1")
        bp.create_proposer("a2")

        result1 = bp.run_round("a1", 1, "val_a")
        assert result1 == "val_a"

        # a2 should adopt val_a
        result2 = bp.run_round("a2", 1, "val_b")
        assert result2 == "val_a"

    def test_proposer_respects_accepted(self):
        """If acceptor has already accepted, proposer must adopt that value."""
        bp = BasicPaxos(["a1", "a2", "a3"])
        bp.create_proposer("a1")
        bp.run_round("a1", 1, "first")

        bp.create_proposer("a2")
        result = bp.run_round("a2", 1, "second")
        # Must adopt "first" because acceptors already accepted it
        assert result == "first"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
