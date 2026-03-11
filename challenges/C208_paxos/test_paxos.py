"""
Tests for C208: Paxos / Multi-Paxos Consensus Protocol
"""

import pytest
import json
from paxos import (
    BallotNumber, Proposal, Message, MessageType, ProposalStatus,
    AcceptorState, SingleDecreePaxos, MultiPaxosNode, PaxosCluster,
    FlexiblePaxos, CatchUpProtocol, PaxosStats,
)


# ============================================================
# BallotNumber Tests
# ============================================================

class TestBallotNumber:
    def test_creation(self):
        b = BallotNumber(1, "node1")
        assert b.round == 1
        assert b.proposer_id == "node1"

    def test_ordering_by_round(self):
        b1 = BallotNumber(1, "a")
        b2 = BallotNumber(2, "a")
        assert b1 < b2
        assert b2 > b1
        assert not b1 > b2

    def test_ordering_by_proposer_id(self):
        b1 = BallotNumber(1, "a")
        b2 = BallotNumber(1, "b")
        assert b1 < b2

    def test_equality(self):
        b1 = BallotNumber(1, "a")
        b2 = BallotNumber(1, "a")
        assert b1 == b2
        assert not (b1 != b2)

    def test_inequality(self):
        b1 = BallotNumber(1, "a")
        b2 = BallotNumber(1, "b")
        assert b1 != b2

    def test_le_ge(self):
        b1 = BallotNumber(1, "a")
        b2 = BallotNumber(2, "a")
        assert b1 <= b2
        assert b2 >= b1
        assert b1 <= BallotNumber(1, "a")
        assert b1 >= BallotNumber(1, "a")

    def test_hash(self):
        b1 = BallotNumber(1, "a")
        b2 = BallotNumber(1, "a")
        assert hash(b1) == hash(b2)
        s = {b1, b2}
        assert len(s) == 1

    def test_serialization(self):
        b = BallotNumber(3, "node2")
        d = b.to_dict()
        assert d == {"round": 3, "proposer_id": "node2"}
        b2 = BallotNumber.from_dict(d)
        assert b == b2

    def test_not_equal_to_non_ballot(self):
        b = BallotNumber(1, "a")
        assert b != "not a ballot"
        assert not (b == 42)


# ============================================================
# Proposal Tests
# ============================================================

class TestProposal:
    def test_creation(self):
        b = BallotNumber(1, "n1")
        p = Proposal(b, "value1")
        assert p.ballot == b
        assert p.value == "value1"
        assert p.slot == 0

    def test_with_slot(self):
        b = BallotNumber(1, "n1")
        p = Proposal(b, "v", slot=5)
        assert p.slot == 5

    def test_serialization(self):
        b = BallotNumber(2, "n2")
        p = Proposal(b, {"key": "val"}, slot=3)
        d = p.to_dict()
        p2 = Proposal.from_dict(d)
        assert p2.ballot == b
        assert p2.value == {"key": "val"}
        assert p2.slot == 3


# ============================================================
# AcceptorState Tests
# ============================================================

class TestAcceptorState:
    def test_initial_state(self):
        a = AcceptorState("node1")
        assert a.promised is None
        assert a.accepted_ballot is None
        assert a.accepted_value is None

    def test_prepare_first_time(self):
        a = AcceptorState("node1")
        result = a.handle_prepare(BallotNumber(1, "p1"))
        assert result["ok"] is True
        assert result["accepted_ballot"] is None
        assert result["accepted_value"] is None

    def test_prepare_higher_ballot(self):
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"))
        result = a.handle_prepare(BallotNumber(2, "p1"))
        assert result["ok"] is True

    def test_prepare_lower_ballot_rejected(self):
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(2, "p1"))
        result = a.handle_prepare(BallotNumber(1, "p1"))
        assert result["ok"] is False

    def test_accept_after_promise(self):
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"))
        result = a.handle_accept(BallotNumber(1, "p1"), "value1")
        assert result["ok"] is True
        assert a.accepted_value == "value1"

    def test_accept_lower_ballot_rejected(self):
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(2, "p1"))
        result = a.handle_accept(BallotNumber(1, "p1"), "value1")
        assert result["ok"] is False

    def test_prepare_returns_accepted(self):
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"))
        a.handle_accept(BallotNumber(1, "p1"), "val1")
        result = a.handle_prepare(BallotNumber(2, "p2"))
        assert result["ok"] is True
        assert result["accepted_ballot"] == BallotNumber(1, "p1")
        assert result["accepted_value"] == "val1"

    def test_slot_isolation(self):
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"), slot=1)
        a.handle_accept(BallotNumber(1, "p1"), "val1", slot=1)
        a.handle_prepare(BallotNumber(1, "p1"), slot=2)
        a.handle_accept(BallotNumber(1, "p1"), "val2", slot=2)
        assert a.slots[1]["accepted_value"] == "val1"
        assert a.slots[2]["accepted_value"] == "val2"

    def test_to_dict(self):
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"))
        a.handle_accept(BallotNumber(1, "p1"), "v")
        d = a.to_dict()
        assert d["node_id"] == "node1"
        assert d["accepted_value"] == "v"


# ============================================================
# SingleDecreePaxos Tests
# ============================================================

class TestSingleDecreePaxos:
    def make_cluster(self, n=3):
        ids = [f"n{i}" for i in range(n)]
        nodes = {}
        for nid in ids:
            peers = [p for p in ids if p != nid]
            nodes[nid] = SingleDecreePaxos(nid, peers)
        return nodes

    def deliver_all(self, nodes, max_rounds=50):
        for _ in range(max_rounds):
            msgs = []
            for node in nodes.values():
                while node.outbox:
                    msgs.append(node.outbox.pop(0))
            if not msgs:
                break
            for msg in msgs:
                if msg.dst in nodes:
                    nodes[msg.dst].receive(msg)

    def test_basic_consensus(self):
        nodes = self.make_cluster(3)
        nodes["n0"].propose("hello")
        self.deliver_all(nodes)
        assert nodes["n0"].is_decided
        assert nodes["n0"].decided_value == "hello"

    def test_all_nodes_learn(self):
        nodes = self.make_cluster(3)
        nodes["n0"].propose("agreed")
        self.deliver_all(nodes)
        for node in nodes.values():
            assert node.decided_value == "agreed"

    def test_five_node_consensus(self):
        nodes = self.make_cluster(5)
        nodes["n0"].propose(42)
        self.deliver_all(nodes)
        for node in nodes.values():
            assert node.decided_value == 42

    def test_competing_proposals_same_round(self):
        """Two proposers compete -- one must win."""
        nodes = self.make_cluster(3)
        nodes["n0"].propose("A")
        nodes["n1"].propose("B")
        self.deliver_all(nodes)
        # At least one must decide
        decided = [n for n in nodes.values() if n.is_decided]
        assert len(decided) >= 1
        # All decided nodes must agree
        values = set(n.decided_value for n in decided)
        assert len(values) == 1

    def test_proposal_status_transitions(self):
        nodes = self.make_cluster(3)
        n = nodes["n0"]
        assert n.proposal_status == ProposalStatus.PENDING
        n.propose("test")
        self.deliver_all(nodes)
        assert n.proposal_status == ProposalStatus.DECIDED

    def test_quorum_size(self):
        nodes = self.make_cluster(3)
        assert nodes["n0"].quorum_size == 2

    def test_quorum_size_5(self):
        nodes = self.make_cluster(5)
        assert nodes["n0"].quorum_size == 3

    def test_dict_value(self):
        nodes = self.make_cluster(3)
        val = {"key": "value", "num": 42}
        nodes["n0"].propose(val)
        self.deliver_all(nodes)
        assert nodes["n0"].decided_value == val

    def test_previously_accepted_value_wins(self):
        """If an acceptor has already accepted a value, new proposer must use it."""
        nodes = self.make_cluster(3)
        # n0 proposes "first"
        nodes["n0"].propose("first")
        self.deliver_all(nodes)
        assert nodes["n0"].decided_value == "first"

        # Create new nodes for a "second round" scenario
        nodes2 = self.make_cluster(3)
        # Manually set n1's acceptor to have accepted "first"
        nodes2["n1"].acceptor.promised = BallotNumber(1, "n0")
        nodes2["n1"].acceptor.accepted_ballot = BallotNumber(1, "n0")
        nodes2["n1"].acceptor.accepted_value = "first"
        # n2 proposes "second" but should adopt "first"
        nodes2["n2"].propose("second")
        self.deliver_all(nodes2)
        if nodes2["n2"].is_decided:
            assert nodes2["n2"].decided_value == "first"

    def test_multiple_sequential_proposals(self):
        """Second proposal after first decides should also decide."""
        nodes = self.make_cluster(3)
        nodes["n0"].propose("v1")
        self.deliver_all(nodes)
        assert nodes["n0"].decided_value == "v1"
        # Already decided, new proposal won't change it
        nodes["n1"].propose("v2")
        self.deliver_all(nodes)
        # Original value stays
        assert nodes["n0"].decided_value == "v1"

    def test_none_value(self):
        nodes = self.make_cluster(3)
        nodes["n0"].propose(None)
        self.deliver_all(nodes)
        assert nodes["n0"].is_decided
        assert nodes["n0"].decided_value is None

    def test_list_value(self):
        nodes = self.make_cluster(3)
        nodes["n0"].propose([1, 2, 3])
        self.deliver_all(nodes)
        assert nodes["n0"].decided_value == [1, 2, 3]


# ============================================================
# PaxosCluster Tests (Single-Decree)
# ============================================================

class TestPaxosClusterSingle:
    def test_create_cluster(self):
        c = PaxosCluster(["a", "b", "c"])
        assert len(c.nodes) == 3
        assert not c.multi

    def test_propose_single(self):
        c = PaxosCluster(["a", "b", "c"])
        assert c.propose_single("a", "hello")
        assert c.get_decided_value("a") == "hello"

    def test_all_decide(self):
        c = PaxosCluster(["a", "b", "c"])
        c.propose_single("a", 42)
        for nid in ["a", "b", "c"]:
            assert c.get_decided_value(nid) == 42

    def test_all_decided_same(self):
        c = PaxosCluster(["a", "b", "c"])
        c.propose_single("a", "consensus")
        assert c.all_decided_same()

    def test_five_nodes(self):
        c = PaxosCluster(["a", "b", "c", "d", "e"])
        c.propose_single("c", "five")
        assert c.all_decided_same()
        assert c.get_decided_value("a") == "five"

    def test_partition_blocks_consensus(self):
        """Partitioned proposer can't reach quorum."""
        c = PaxosCluster(["a", "b", "c"])
        c.partition(["a"], ["b", "c"])
        c.propose_single("a", "blocked")
        # a can only reach itself (1/3), not quorum
        assert not c.nodes["a"].is_decided

    def test_majority_partition_succeeds(self):
        """Proposer with majority can still decide."""
        c = PaxosCluster(["a", "b", "c", "d", "e"])
        c.partition(["d", "e"], ["a", "b", "c"])
        c.propose_single("a", "majority")
        assert c.nodes["a"].is_decided
        assert c.nodes["a"].decided_value == "majority"

    def test_heal_partition(self):
        c = PaxosCluster(["a", "b", "c"])
        c.partition(["a"], ["b", "c"])
        c.heal_partition()
        c.propose_single("a", "healed")
        assert c.nodes["a"].is_decided

    def test_message_counting(self):
        c = PaxosCluster(["a", "b", "c"])
        c.propose_single("a", "count")
        assert c.delivered_messages > 0

    def test_message_log(self):
        c = PaxosCluster(["a", "b", "c"])
        c.propose_single("a", "logged")
        assert len(c.message_log) > 0
        assert all("type" in m and "src" in m for m in c.message_log)

    def test_dropped_messages_counted(self):
        c = PaxosCluster(["a", "b", "c"])
        c.partition(["a"], ["b", "c"])
        c.propose_single("a", "drop")
        assert c.dropped_messages > 0


# ============================================================
# MultiPaxosNode Tests
# ============================================================

class TestMultiPaxosNode:
    def test_creation(self):
        n = MultiPaxosNode("n0", ["n1", "n2"])
        assert n.node_id == "n0"
        assert not n.is_leader
        assert n.commit_index == 0
        assert n.next_slot == 1

    def test_quorum_3(self):
        n = MultiPaxosNode("n0", ["n1", "n2"])
        assert n.quorum_size == 2

    def test_quorum_5(self):
        n = MultiPaxosNode("n0", ["n1", "n2", "n3", "n4"])
        assert n.quorum_size == 3

    def test_start_election(self):
        n = MultiPaxosNode("n0", ["n1", "n2"])
        ballot = n.start_election()
        assert ballot.round == 1
        assert ballot.proposer_id == "n0"
        assert len(n.outbox) == 3  # Prepare to all

    def test_submit_not_leader(self):
        n = MultiPaxosNode("n0", ["n1", "n2"])
        result = n.submit_command({"type": "set", "key": "k", "value": "v"})
        assert result is None

    def test_get_decided_values_empty(self):
        n = MultiPaxosNode("n0", ["n1", "n2"])
        assert n.get_decided_values() == []

    def test_state_machine_empty(self):
        n = MultiPaxosNode("n0", ["n1", "n2"])
        assert n.state_machine == {}


# ============================================================
# PaxosCluster Multi-Paxos Tests
# ============================================================

class TestPaxosClusterMulti:
    def test_create_multi_cluster(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        assert c.multi
        assert len(c.nodes) == 3

    def test_election(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        assert c.run_election("a")
        assert c.nodes["a"].is_leader

    def test_submit_single_command(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        slot = c.submit("a", {"type": "set", "key": "x", "value": 1})
        assert slot == 1
        assert c.all_logs_consistent()

    def test_submit_multiple_commands(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "x", "value": 1})
        c.submit("a", {"type": "set", "key": "y", "value": 2})
        c.submit("a", {"type": "set", "key": "z", "value": 3})
        assert c.all_logs_consistent()
        sm = c.get_state_machine("a")
        assert sm.get("x") == 1
        assert sm.get("y") == 2
        assert sm.get("z") == 3

    def test_state_machine_replication(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "k", "value": 42})
        # Followers learn via Decide messages
        assert 1 in c.get_log("b")
        assert c.get_log("b")[1] == {"type": "set", "key": "k", "value": 42}

    def test_delete_command(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "k", "value": 1})
        c.submit("a", {"type": "delete", "key": "k"})
        sm = c.get_state_machine("a")
        assert "k" not in sm

    def test_noop_is_harmless(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        node = c.nodes["a"]
        node._apply_command({"type": "noop"})
        assert node.state_machine == {}

    def test_get_leader(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("b")
        assert c.get_leader() == "b"

    def test_no_leader_initially(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        assert c.get_leader() is None

    def test_five_node_multi(self):
        c = PaxosCluster(["a", "b", "c", "d", "e"], multi=True)
        c.run_election("c")
        c.submit("c", {"type": "set", "key": "big", "value": "cluster"})
        assert c.all_logs_consistent()
        for nid in ["a", "b", "c", "d", "e"]:
            assert 1 in c.get_log(nid)

    def test_heartbeat(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.nodes["a"].send_heartbeat()
        c.deliver_messages()
        # Followers should recognize leader
        assert c.nodes["b"].leader_id == "a"
        assert c.nodes["c"].leader_id == "a"

    def test_heartbeat_updates_commit_index(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "k", "value": 1})
        # Leader has commit_index advanced
        leader = c.nodes["a"]
        assert leader.commit_index >= 1
        # Send heartbeat with commit index
        leader.send_heartbeat()
        c.deliver_messages()
        # Followers should update commit_index
        assert c.nodes["b"].commit_index >= 1

    def test_partition_blocks_multi(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.partition(["a"], ["b", "c"])
        slot = c.submit("a", {"type": "set", "key": "x", "value": 1})
        # Leader can't reach quorum, log entry won't be committed
        assert c.nodes["a"].commit_index == 0

    def test_leader_election_after_partition(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.partition(["a"], ["b", "c"])
        # b takes over
        assert c.run_election("b")
        assert c.nodes["b"].is_leader

    def test_multiple_elections(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        assert c.nodes["a"].is_leader
        c.run_election("b")
        assert c.nodes["b"].is_leader

    def test_ten_commands(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        for i in range(10):
            c.submit("a", {"type": "set", "key": f"k{i}", "value": i})
        assert c.all_logs_consistent()
        sm = c.get_state_machine("a")
        assert len(sm) == 10
        for i in range(10):
            assert sm[f"k{i}"] == i

    def test_log_consistency_after_leader_change(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "before", "value": 1})
        # Leader change
        c.run_election("b")
        c.submit("b", {"type": "set", "key": "after", "value": 2})
        assert c.all_logs_consistent()

    def test_submit_returns_none_for_non_leader(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        result = c.submit("b", {"type": "set", "key": "k", "value": 1})
        assert result is None

    def test_submit_returns_none_for_single(self):
        c = PaxosCluster(["a", "b", "c"])
        result = c.submit("a", {"type": "set", "key": "k", "value": 1})
        assert result is None

    def test_run_election_returns_false_for_single(self):
        c = PaxosCluster(["a", "b", "c"])
        assert not c.run_election("a")


# ============================================================
# FlexiblePaxos Tests
# ============================================================

class TestFlexiblePaxos:
    def make_cluster(self, n=5, q1=None, q2=None):
        ids = [f"n{i}" for i in range(n)]
        nodes = {}
        for nid in ids:
            peers = [p for p in ids if p != nid]
            nodes[nid] = FlexiblePaxos(nid, peers, q1, q2)
        return nodes

    def deliver_all(self, nodes, max_rounds=50):
        for _ in range(max_rounds):
            msgs = []
            for node in nodes.values():
                while node.outbox:
                    msgs.append(node.outbox.pop(0))
            if not msgs:
                break
            for msg in msgs:
                if msg.dst in nodes:
                    nodes[msg.dst].receive(msg)

    def test_default_quorums(self):
        n = FlexiblePaxos("n0", ["n1", "n2"])
        assert n.phase1_quorum == 2
        assert n.phase2_quorum == 2

    def test_custom_quorums(self):
        n = FlexiblePaxos("n0", ["n1", "n2", "n3", "n4"],
                          phase1_quorum=4, phase2_quorum=2)
        assert n.phase1_quorum == 4
        assert n.phase2_quorum == 2

    def test_invalid_quorums_rejected(self):
        with pytest.raises(ValueError):
            FlexiblePaxos("n0", ["n1", "n2", "n3", "n4"],
                          phase1_quorum=2, phase2_quorum=2)  # 2+2 = 4, not > 5

    def test_consensus_default(self):
        nodes = self.make_cluster(3)
        nodes["n0"].propose("flex")
        self.deliver_all(nodes)
        assert nodes["n0"].decided_value == "flex"

    def test_consensus_aggressive_phase2(self):
        """Phase 2 quorum of 2, Phase 1 quorum of 4 (N=5)."""
        nodes = self.make_cluster(5, q1=4, q2=2)
        nodes["n0"].propose("fast_accept")
        self.deliver_all(nodes)
        assert nodes["n0"].decided_value == "fast_accept"

    def test_consensus_aggressive_phase1(self):
        """Phase 1 quorum of 2, Phase 2 quorum of 4 (N=5)."""
        nodes = self.make_cluster(5, q1=2, q2=4)
        nodes["n0"].propose("fast_prepare")
        self.deliver_all(nodes)
        assert nodes["n0"].decided_value == "fast_prepare"

    def test_all_nodes_decide(self):
        nodes = self.make_cluster(5, q1=4, q2=2)
        nodes["n0"].propose("all_learn")
        self.deliver_all(nodes)
        for node in nodes.values():
            assert node.decided_value == "all_learn"

    def test_serialization_roundtrip(self):
        b = BallotNumber(5, "test")
        d = b.to_dict()
        b2 = BallotNumber.from_dict(d)
        assert b == b2


# ============================================================
# CatchUpProtocol Tests
# ============================================================

class TestCatchUp:
    def test_needs_catchup(self):
        node = MultiPaxosNode("n0", ["n1", "n2"])
        cu = CatchUpProtocol(node)
        assert cu.needs_catchup(5)
        assert not cu.needs_catchup(0)

    def test_serve_catchup(self):
        leader = MultiPaxosNode("leader", ["f1"])
        leader.log = {1: "a", 2: "b", 3: "c"}
        leader.commit_index = 3
        entries = CatchUpProtocol.serve_catchup(leader, 2)
        assert entries == {2: "b", 3: "c"}

    def test_handle_catchup_response(self):
        node = MultiPaxosNode("n0", ["n1"])
        cu = CatchUpProtocol(node)
        node.log[1] = {"type": "set", "key": "a", "value": 1}
        node.commit_index = 1
        cu.handle_catchup_response({2: {"type": "set", "key": "b", "value": 2}})
        assert 2 in node.log
        assert not cu.pending_catchup

    def test_request_catchup_sends_message(self):
        node = MultiPaxosNode("n0", ["n1"])
        cu = CatchUpProtocol(node)
        cu.request_catchup("n1", 1)
        assert cu.pending_catchup
        assert len(node.outbox) == 1
        assert node.outbox[0].type == MessageType.CLIENT_REQUEST

    def test_full_catchup_flow(self):
        leader = MultiPaxosNode("leader", ["follower"])
        leader.log = {1: {"type": "set", "key": "x", "value": 10},
                      2: {"type": "set", "key": "y", "value": 20}}
        leader.commit_index = 2

        follower = MultiPaxosNode("follower", ["leader"])
        cu = CatchUpProtocol(follower)
        assert cu.needs_catchup(leader.commit_index)

        entries = CatchUpProtocol.serve_catchup(leader, 1)
        cu.handle_catchup_response(entries)
        assert follower.log == leader.log


# ============================================================
# PaxosStats Tests
# ============================================================

class TestPaxosStats:
    def test_initial(self):
        s = PaxosStats()
        assert s.proposals_started == 0
        assert s.success_rate == 0.0
        assert s.average_rounds == 0.0

    def test_record_proposal(self):
        s = PaxosStats()
        s.record_proposal(True, 1)
        assert s.proposals_decided == 1
        assert s.success_rate == 1.0

    def test_record_rejected(self):
        s = PaxosStats()
        s.record_proposal(False)
        assert s.proposals_rejected == 1
        assert s.success_rate == 0.0

    def test_mixed_proposals(self):
        s = PaxosStats()
        s.record_proposal(True, 1)
        s.record_proposal(True, 2)
        s.record_proposal(False)
        assert s.proposals_started == 3
        assert s.proposals_decided == 2
        assert abs(s.success_rate - 2/3) < 0.01

    def test_average_rounds(self):
        s = PaxosStats()
        s.record_proposal(True, 1)
        s.record_proposal(True, 3)
        assert s.average_rounds == 2.0

    def test_election_tracking(self):
        s = PaxosStats()
        s.record_election(True)
        s.record_election(False)
        assert s.elections_held == 2
        assert s.leader_changes == 1

    def test_summary(self):
        s = PaxosStats()
        s.record_proposal(True, 1)
        s.record_election(True)
        d = s.summary()
        assert d["proposals"] == 1
        assert d["decided"] == 1
        assert d["success_rate"] == 1.0


# ============================================================
# Edge Cases and Stress Tests
# ============================================================

class TestEdgeCases:
    def test_two_node_cluster(self):
        """Two nodes need both for quorum (no fault tolerance)."""
        c = PaxosCluster(["a", "b"])
        c.propose_single("a", "two")
        assert c.nodes["a"].decided_value == "two"

    def test_single_node_cluster(self):
        """Single node is always quorum."""
        c = PaxosCluster(["a"])
        c.propose_single("a", "solo")
        assert c.nodes["a"].decided_value == "solo"

    def test_single_node_multi(self):
        c = PaxosCluster(["a"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "solo", "value": 1})
        assert c.get_state_machine("a") == {"solo": 1}

    def test_seven_node_cluster(self):
        ids = [f"n{i}" for i in range(7)]
        c = PaxosCluster(ids)
        c.propose_single("n3", "seven")
        assert c.all_decided_same()

    def test_empty_string_value(self):
        c = PaxosCluster(["a", "b", "c"])
        c.propose_single("a", "")
        assert c.nodes["a"].decided_value == ""

    def test_integer_value(self):
        c = PaxosCluster(["a", "b", "c"])
        c.propose_single("a", 999)
        assert c.nodes["a"].decided_value == 999

    def test_nested_dict_value(self):
        val = {"a": {"b": {"c": [1, 2, 3]}}}
        c = PaxosCluster(["a", "b", "c"])
        c.propose_single("a", val)
        assert c.nodes["a"].decided_value == val

    def test_all_logs_consistent_empty(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        assert c.all_logs_consistent()

    def test_deliver_no_messages(self):
        c = PaxosCluster(["a", "b", "c"])
        count = c.deliver_messages()
        assert count == 0

    def test_is_partitioned_check(self):
        c = PaxosCluster(["a", "b", "c"])
        assert not c.is_partitioned("a", "b")
        c.partition(["a"], ["b"])
        assert c.is_partitioned("a", "b")
        assert c.is_partitioned("b", "a")

    def test_multi_partition_minority_leader(self):
        """Leader in minority partition can't commit."""
        c = PaxosCluster(["a", "b", "c", "d", "e"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "before", "value": 1})
        # Partition leader into minority
        c.partition(["a", "b"], ["c", "d", "e"])
        slot = c.submit("a", {"type": "set", "key": "blocked", "value": 2})
        # Can't reach quorum (need 3, only have a+b=2)
        # "blocked" should not be committed
        assert "blocked" not in c.get_state_machine("a")

    def test_heal_and_continue(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "k1", "value": 1})
        c.partition(["a"], ["b", "c"])
        c.run_election("b")
        c.submit("b", {"type": "set", "key": "k2", "value": 2})
        c.heal_partition()
        c.deliver_messages()
        assert c.all_logs_consistent()


# ============================================================
# Message Type Tests
# ============================================================

class TestMessageTypes:
    def test_message_type_values(self):
        assert MessageType.PREPARE.value == "prepare"
        assert MessageType.PROMISE.value == "promise"
        assert MessageType.ACCEPT.value == "accept"
        assert MessageType.ACCEPTED.value == "accepted"
        assert MessageType.DECIDE.value == "decide"

    def test_proposal_status_values(self):
        assert ProposalStatus.PENDING.value == "pending"
        assert ProposalStatus.DECIDED.value == "decided"

    def test_message_creation(self):
        m = Message(
            type=MessageType.PREPARE,
            src="a",
            dst="b",
            ballot=BallotNumber(1, "a"),
        )
        assert m.type == MessageType.PREPARE
        assert m.src == "a"
        assert m.dst == "b"
        assert m.data == {}


# ============================================================
# Gap Filling Tests
# ============================================================

class TestGapFilling:
    def test_noop_gaps_filled_on_election(self):
        """New leader fills gaps with noops."""
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "k1", "value": 1})
        c.submit("a", {"type": "set", "key": "k2", "value": 2})

        # Simulate a gap: remove slot 1 from b's log
        if 1 in c.nodes["b"].log:
            del c.nodes["b"].log[1]

        # New leader takes over - should fill gaps
        c.run_election("b")
        c.deliver_messages()
        # After re-election, logs should be consistent
        assert c.all_logs_consistent()

    def test_decided_values_in_order(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        c.submit("a", {"type": "set", "key": "first", "value": 1})
        c.submit("a", {"type": "set", "key": "second", "value": 2})
        vals = c.nodes["a"].get_decided_values()
        assert len(vals) == 2


# ============================================================
# Durability and Recovery Tests
# ============================================================

class TestDurability:
    def test_acceptor_state_persists(self):
        """Acceptor state survives across proposals."""
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"))
        a.handle_accept(BallotNumber(1, "p1"), "val1")

        # New prepare with higher ballot
        result = a.handle_prepare(BallotNumber(2, "p2"))
        assert result["ok"] is True
        assert result["accepted_ballot"] == BallotNumber(1, "p1")
        assert result["accepted_value"] == "val1"

    def test_acceptor_rejects_stale_accept(self):
        """Acceptor promised higher ballot rejects lower accept."""
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(2, "p2"))
        result = a.handle_accept(BallotNumber(1, "p1"), "stale")
        assert result["ok"] is False

    def test_acceptor_updates_promise_on_accept(self):
        """Accepting also updates promise."""
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"))
        a.handle_accept(BallotNumber(1, "p1"), "val")
        assert a.promised == BallotNumber(1, "p1")

    def test_multi_slot_acceptor_independence(self):
        """Different slots have independent state."""
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"), slot=1)
        a.handle_accept(BallotNumber(1, "p1"), "v1", slot=1)
        a.handle_prepare(BallotNumber(2, "p2"), slot=2)
        a.handle_accept(BallotNumber(2, "p2"), "v2", slot=2)

        # Slot 1 has ballot 1, slot 2 has ballot 2
        assert a.slots[1]["accepted_ballot"] == BallotNumber(1, "p1")
        assert a.slots[2]["accepted_ballot"] == BallotNumber(2, "p2")

    def test_acceptor_slot0_vs_slots(self):
        """Slot 0 uses single-decree state."""
        a = AcceptorState("node1")
        a.handle_prepare(BallotNumber(1, "p1"), slot=0)
        a.handle_accept(BallotNumber(1, "p1"), "slot0_val", slot=0)
        assert a.accepted_value == "slot0_val"  # Stored in main fields
        assert 0 not in a.slots  # Not in slots dict


# ============================================================
# Stress Tests
# ============================================================

class TestStress:
    def test_many_commands(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        c.run_election("a")
        for i in range(50):
            c.submit("a", {"type": "set", "key": f"k{i}", "value": i})
        assert c.all_logs_consistent()
        sm = c.get_state_machine("a")
        assert len(sm) == 50

    def test_rapid_elections(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        for candidate in ["a", "b", "c", "a", "b"]:
            c.run_election(candidate)
        leader = c.get_leader()
        assert leader is not None
        c.submit(leader, {"type": "set", "key": "after_chaos", "value": 1})
        assert c.all_logs_consistent()

    def test_alternating_leaders(self):
        c = PaxosCluster(["a", "b", "c"], multi=True)
        for i in range(5):
            leader = ["a", "b"][i % 2]
            c.run_election(leader)
            c.submit(leader, {"type": "set", "key": f"k{i}", "value": i})
        assert c.all_logs_consistent()

    def test_large_cluster(self):
        ids = [f"n{i}" for i in range(9)]
        c = PaxosCluster(ids, multi=True)
        c.run_election("n0")
        for i in range(20):
            c.submit("n0", {"type": "set", "key": f"k{i}", "value": i})
        assert c.all_logs_consistent()
        assert len(c.get_state_machine("n0")) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
