"""Tests for C230: Raft Consensus Protocol."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import unittest
import random
from raft import (
    RaftNode, Role, MessageType, Message, LogEntry, StateMachine,
    Network, Snapshot, create_cluster
)


class TestStateMachine(unittest.TestCase):
    """Tests for the key-value state machine."""

    def test_set_and_get(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 42})
        assert sm.apply({"op": "get", "key": "x"}) == 42

    def test_delete(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1})
        result = sm.apply({"op": "delete", "key": "x"})
        assert result == 1
        assert sm.apply({"op": "get", "key": "x"}) is None

    def test_cas_success(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1})
        assert sm.apply({"op": "cas", "key": "x", "expected": 1, "value": 2}) is True
        assert sm.data["x"] == 2

    def test_cas_failure(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1})
        assert sm.apply({"op": "cas", "key": "x", "expected": 99, "value": 2}) is False
        assert sm.data["x"] == 1

    def test_snapshot_and_restore(self):
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

    def test_delete_nonexistent(self):
        sm = StateMachine()
        assert sm.apply({"op": "delete", "key": "x"}) is None

    def test_restore_empty(self):
        sm = StateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1})
        sm.restore(None)
        assert sm.data == {}


class TestLogEntry(unittest.TestCase):
    """Tests for LogEntry serialization."""

    def test_to_dict_and_back(self):
        e = LogEntry(term=3, index=5, command={"op": "set", "key": "x", "value": 1})
        d = e.to_dict()
        e2 = LogEntry.from_dict(d)
        assert e2.term == 3
        assert e2.index == 5
        assert e2.command == {"op": "set", "key": "x", "value": 1}

    def test_entry_type_preserved(self):
        e = LogEntry(term=1, index=1, command=None, entry_type="noop")
        d = e.to_dict()
        e2 = LogEntry.from_dict(d)
        assert e2.entry_type == "noop"

    def test_default_entry_type(self):
        d = {"term": 1, "index": 1, "command": None}
        e = LogEntry.from_dict(d)
        assert e.entry_type == "command"


class TestRaftNodeBasic(unittest.TestCase):
    """Basic node state tests."""

    def test_initial_state(self):
        node = RaftNode("n1", ["n2", "n3"])
        assert node.role == Role.FOLLOWER
        assert node.current_term == 0
        assert node.voted_for is None
        assert node.commit_index == 0
        assert node.last_applied == 0

    def test_last_log_index_empty(self):
        node = RaftNode("n1", [])
        assert node._last_log_index() == 0
        assert node._last_log_term() == 0

    def test_last_log_index_with_entries(self):
        node = RaftNode("n1", [])
        node.log.append(LogEntry(term=1, index=1, command=None))
        node.log.append(LogEntry(term=2, index=2, command=None))
        assert node._last_log_index() == 2
        assert node._last_log_term() == 2

    def test_get_log_entry(self):
        node = RaftNode("n1", [])
        node.log.append(LogEntry(term=1, index=1, command="a"))
        node.log.append(LogEntry(term=1, index=2, command="b"))
        assert node._get_log_entry(1).command == "a"
        assert node._get_log_entry(2).command == "b"
        assert node._get_log_entry(3) is None
        assert node._get_log_entry(0) is None

    def test_quorum_3_nodes(self):
        node = RaftNode("n1", ["n2", "n3"])
        assert node._quorum_size() == 2

    def test_quorum_5_nodes(self):
        node = RaftNode("n1", ["n2", "n3", "n4", "n5"])
        assert node._quorum_size() == 3

    def test_voting_members(self):
        node = RaftNode("n1", ["n2", "n3"])
        assert set(node._voting_members()) == {"n1", "n2", "n3"}

    def test_voting_excludes_learners(self):
        node = RaftNode("n1", ["n2", "n3"])
        node.learners.add("n3")
        members = node._voting_members()
        assert "n3" not in members
        assert set(members) == {"n1", "n2"}


class TestSingleNodeCluster(unittest.TestCase):
    """Single node should self-elect immediately."""

    def test_single_node_becomes_leader(self):
        net = create_cluster(["n1"])
        leader = net.wait_for_leader(max_ticks=500)
        assert leader is not None
        assert leader.node_id == "n1"
        assert leader.role == Role.LEADER

    def test_single_node_commits(self):
        net = create_cluster(["n1"])
        net.wait_for_leader(max_ticks=500)
        leader = net.find_leader()
        idx = leader.submit({"op": "set", "key": "x", "value": 42})
        assert idx is not None
        net.run(ticks=100)
        assert leader.state_machine.data.get("x") == 42


class TestLeaderElection(unittest.TestCase):
    """Leader election tests."""

    def test_three_node_election(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader(max_ticks=2000)
        assert leader is not None
        assert leader.role == Role.LEADER

    def test_five_node_election(self):
        random.seed(123)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        leader = net.wait_for_leader(max_ticks=2000)
        assert leader is not None
        followers = [n for n in net.nodes.values() if n.role == Role.FOLLOWER]
        assert len(followers) >= 3

    def test_only_one_leader_per_term(self):
        random.seed(7)
        net = create_cluster(["n1", "n2", "n3"])
        net.run(ticks=2000)
        leaders = [n for n in net.nodes.values() if n.role == Role.LEADER]
        if len(leaders) > 1:
            terms = [l.current_term for l in leaders]
            assert len(set(terms)) == len(terms), "Multiple leaders in same term!"

    def test_higher_term_causes_step_down(self):
        node = RaftNode("n1", ["n2"])
        node.role = Role.LEADER
        node.current_term = 1
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=5,
            data={"leader_id": "n2", "prev_log_index": 0, "prev_log_term": 0,
                  "entries": [], "leader_commit": 0}
        )
        node.receive(msg)
        assert node.role == Role.FOLLOWER
        assert node.current_term == 5

    def test_vote_denied_for_stale_term(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 5
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            sender="n2", receiver="n1",
            term=3,
            data={"candidate_id": "n2", "last_log_index": 0, "last_log_term": 0}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["vote_granted"] is False

    def test_vote_denied_for_stale_log(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 1
        node.log.append(LogEntry(term=1, index=1, command="x"))
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            sender="n2", receiver="n1",
            term=1,
            data={"candidate_id": "n2", "last_log_index": 0, "last_log_term": 0}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["vote_granted"] is False

    def test_vote_granted_for_up_to_date_log(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 1
        node.log.append(LogEntry(term=1, index=1, command="x"))
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            sender="n2", receiver="n1",
            term=2,
            data={"candidate_id": "n2", "last_log_index": 1, "last_log_term": 1}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["vote_granted"] is True
        assert node.voted_for == "n2"

    def test_election_resets_timer(self):
        node = RaftNode("n1", ["n2"])
        node.election_timer = 100
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            sender="n2", receiver="n1",
            term=1,
            data={"candidate_id": "n2", "last_log_index": 0, "last_log_term": 0}
        )
        node.receive(msg)
        assert node.election_timer == 0

    def test_leader_noop_on_election(self):
        """Leader should append a noop entry on election."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        assert leader is not None
        # The last entry should be a noop
        noop_found = any(e.entry_type == "noop" for e in leader.log)
        assert noop_found, "Leader should append noop on election"


class TestLogReplication(unittest.TestCase):
    """Log replication tests."""

    def test_basic_replication(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        idx = leader.submit({"op": "set", "key": "x", "value": 1})
        net.wait_for_commit(idx)
        net.run(ticks=500)
        for node in net.nodes.values():
            assert node.state_machine.data.get("x") == 1, \
                f"Node {node.node_id} missing x=1"

    def test_multiple_entries(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        for i in range(10):
            leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.run(ticks=1000)
        for node in net.nodes.values():
            for i in range(10):
                assert node.state_machine.data.get(f"k{i}") == i

    def test_follower_rejects_stale_term(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 5
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=3,
            data={"leader_id": "n2", "prev_log_index": 0, "prev_log_term": 0,
                  "entries": [], "leader_commit": 0}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["success"] is False

    def test_follower_rejects_inconsistent_log(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 1
        # Node has entry at index 1 with term 1
        node.log.append(LogEntry(term=1, index=1, command="a"))
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=1,
            data={"leader_id": "n2", "prev_log_index": 2, "prev_log_term": 1,
                  "entries": [], "leader_commit": 0}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["success"] is False

    def test_leader_decrements_next_index_on_reject(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        # Add entries only to leader
        for i in range(5):
            idx = leader._last_log_index() + 1
            leader.log.append(LogEntry(term=leader.current_term, index=idx, command=f"cmd{i}"))
        # Next round should trigger catch-up
        net.run(ticks=500)
        # All nodes should eventually converge
        for node in net.nodes.values():
            assert node._last_log_index() >= 5

    def test_commit_requires_quorum(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        # Isolate one follower
        followers = [n for n in net.nodes.values() if n.role == Role.FOLLOWER]
        isolated = followers[0]
        net.isolate(isolated.node_id)
        idx = leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(ticks=500)
        # Leader + 1 follower = quorum of 2 (out of 3)
        assert leader.commit_index >= idx
        assert leader.state_machine.data.get("x") == 1

    def test_no_commit_without_quorum(self):
        """With 5 nodes, isolating 3 leaves leader with only 1 follower -- no quorum."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        leader = net.wait_for_leader()
        # Isolate 3 nodes from leader
        follower_ids = [n.node_id for n in net.nodes.values() if n.node_id != leader.node_id]
        for fid in follower_ids[:3]:
            net.partition(leader.node_id, fid)
        idx = leader.submit({"op": "set", "key": "x", "value": 1})
        net.run(ticks=500)
        # noop is at some index, our entry is at idx
        # With only 1 follower reachable out of 5, can't reach quorum of 3
        assert leader.state_machine.data.get("x") is None

    def test_conflict_resolution(self):
        """Conflicting entries should be overwritten by leader's log."""
        node = RaftNode("n1", ["n2"])
        node.current_term = 2
        node.log.append(LogEntry(term=1, index=1, command="old"))
        # Leader sends entry at index 1 with term 2
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=2,
            data={
                "leader_id": "n2",
                "prev_log_index": 0, "prev_log_term": 0,
                "entries": [{"term": 2, "index": 1, "command": "new", "entry_type": "command"}],
                "leader_commit": 0,
            }
        )
        node.receive(msg)
        assert node.log[0].command == "new"
        assert node.log[0].term == 2


class TestNetworkPartition(unittest.TestCase):
    """Tests for network partitions."""

    def test_partition_blocks_messages(self):
        net = Network()
        n1 = RaftNode("n1", ["n2"])
        n2 = RaftNode("n2", ["n1"])
        net.add_node(n1)
        net.add_node(n2)
        net.partition("n1", "n2")
        n1.outbox.append(Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n1", receiver="n2",
            term=1, data={}
        ))
        net.tick()
        assert len(net.dropped) == 1
        assert len(net.delivered) == 0

    def test_heal_restores_messages(self):
        net = Network()
        n1 = RaftNode("n1", ["n2"])
        n2 = RaftNode("n2", ["n1"])
        net.add_node(n1)
        net.add_node(n2)
        net.partition("n1", "n2")
        net.heal("n1", "n2")
        assert not net.is_partitioned("n1", "n2")

    def test_heal_all(self):
        net = Network()
        net.partition("a", "b")
        net.partition("a", "c")
        net.heal()
        assert len(net.partitions) == 0

    def test_leader_reelection_after_partition_heals(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        old_leader_id = leader.node_id
        # Isolate the leader
        net.isolate(old_leader_id)
        # Run long enough for new election
        net.run(ticks=2000)
        # Heal
        net.heal()
        net.run(ticks=2000)
        # Should have exactly one leader
        leaders = [n for n in net.nodes.values() if n.role == Role.LEADER]
        assert len(leaders) == 1

    def test_split_brain_resolved(self):
        """After partition heals, cluster converges to single leader."""
        random.seed(99)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        leader = net.wait_for_leader()
        # Partition into {n1,n2} vs {n3,n4,n5}
        for a in ["n1", "n2"]:
            for b in ["n3", "n4", "n5"]:
                net.partition(a, b)
        net.run(ticks=3000)
        net.heal()
        net.run(ticks=3000)
        leaders = [n for n in net.nodes.values() if n.role == Role.LEADER]
        assert len(leaders) == 1

    def test_data_survives_partition(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        idx = leader.submit({"op": "set", "key": "x", "value": 42})
        net.wait_for_commit(idx)
        net.run(ticks=200)
        # Partition and heal
        old_id = leader.node_id
        net.isolate(old_id)
        net.run(ticks=2000)
        net.heal()
        net.run(ticks=2000)
        # All nodes should still have x=42
        for node in net.nodes.values():
            assert node.state_machine.data.get("x") == 42

    def test_isolate_node(self):
        net = Network()
        for nid in ["a", "b", "c"]:
            net.add_node(RaftNode(nid, [x for x in ["a", "b", "c"] if x != nid]))
        net.isolate("a")
        assert net.is_partitioned("a", "b")
        assert net.is_partitioned("a", "c")
        assert not net.is_partitioned("b", "c")


class TestSnapshot(unittest.TestCase):
    """Snapshot / log compaction tests."""

    def test_take_snapshot(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        for i in range(10):
            leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.run(ticks=1000)
        snap = leader.take_snapshot()
        assert snap is not None
        assert snap.last_included_index == leader.last_applied
        # Log should be compacted
        old_len = len(leader.log)
        assert leader._last_log_index() >= snap.last_included_index

    def test_snapshot_preserves_state(self):
        random.seed(42)
        net = create_cluster(["n1"])
        leader = net.wait_for_leader()
        for i in range(5):
            leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.run(ticks=500)
        snap = leader.take_snapshot()
        assert snap.data == leader.state_machine.data

    def test_install_snapshot_on_slow_follower(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        # Isolate one follower
        followers = [n for n in net.nodes.values() if n.role == Role.FOLLOWER]
        slow = followers[0]
        net.isolate(slow.node_id)
        # Add many entries and compact
        for i in range(20):
            leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.run(ticks=1000)
        leader.take_snapshot()
        # Heal -- slow follower should catch up via snapshot
        net.heal()
        net.run(ticks=2000)
        # Slow follower should have all data
        for i in range(20):
            assert slow.state_machine.data.get(f"k{i}") == i, \
                f"Slow follower missing k{i}"

    def test_snapshot_log_trimmed(self):
        node = RaftNode("n1", [])
        node.role = Role.LEADER
        node.current_term = 1
        for i in range(1, 11):
            node.log.append(LogEntry(term=1, index=i, command={"op": "set", "key": f"k{i}", "value": i}))
            node.state_machine.apply({"op": "set", "key": f"k{i}", "value": i})
        node.last_applied = 10
        node.commit_index = 10
        snap = node.take_snapshot()
        assert len(node.log) == 0
        assert node._last_log_index() == 10  # via snapshot
        assert node._last_log_term() == 1

    def test_snapshot_config_preserved(self):
        node = RaftNode("n1", ["n2", "n3"])
        node.role = Role.LEADER
        node.current_term = 1
        node.log.append(LogEntry(term=1, index=1, command=None, entry_type="noop"))
        node.state_machine.apply(None)
        node.last_applied = 1
        node.commit_index = 1
        snap = node.take_snapshot()
        assert set(snap.config) == {"n1", "n2", "n3"}


class TestMembershipChange(unittest.TestCase):
    """Cluster membership change tests."""

    def test_add_member(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        # Add n4
        n4 = RaftNode("n4", ["n1", "n2", "n3"])
        net.add_node(n4)
        idx = leader.add_member("n4")
        assert idx is not None
        net.run(ticks=2000)
        # Config should include n4
        assert "n4" in leader.cluster_config

    def test_add_member_not_leader(self):
        node = RaftNode("n1", ["n2"])
        assert node.add_member("n3") is None

    def test_add_existing_member(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        assert leader.add_member("n2") is None

    def test_remove_member(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        # Remove a follower
        followers = [n for n in net.nodes.values() if n.role == Role.FOLLOWER]
        target = followers[0].node_id
        idx = leader.remove_member(target)
        assert idx is not None
        net.run(ticks=2000)
        assert target not in leader.cluster_config

    def test_remove_member_not_leader(self):
        node = RaftNode("n1", ["n2"])
        assert node.remove_member("n2") is None

    def test_remove_nonexistent(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        assert leader.remove_member("n99") is None

    def test_learner_does_not_vote(self):
        node = RaftNode("n1", ["n2", "n3"])
        node.learners.add("n3")
        assert node._quorum_size() == 2  # {n1, n2} -> quorum = 2
        assert "n3" not in node._voting_members()


class TestPreVote(unittest.TestCase):
    """Pre-vote protocol tests."""

    def test_pre_vote_election(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        for node in net.nodes.values():
            node.pre_vote_enabled = True
        leader = net.wait_for_leader()
        assert leader is not None
        assert leader.role == Role.LEADER

    def test_pre_vote_prevents_term_inflation(self):
        """Isolated node with pre-vote shouldn't inflate its term."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        for node in net.nodes.values():
            node.pre_vote_enabled = True
        leader = net.wait_for_leader()
        # Isolate one follower
        followers = [n for n in net.nodes.values() if n.role == Role.FOLLOWER]
        isolated = followers[0]
        initial_term = isolated.current_term
        net.isolate(isolated.node_id)
        net.run(ticks=2000)
        # Term should not have inflated much (pre-vote prevents unnecessary increments)
        # Without pre-vote, it would increment each election attempt
        # With pre-vote, it stays at initial_term since pre-votes fail
        assert isolated.current_term == initial_term

    def test_pre_vote_granted(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 1
        msg = Message(
            type=MessageType.PRE_VOTE,
            sender="n2", receiver="n1",
            term=1,
            data={"candidate_id": "n2", "last_log_index": 0, "last_log_term": 0, "next_term": 2}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["vote_granted"] is True

    def test_pre_vote_denied_stale(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 5
        msg = Message(
            type=MessageType.PRE_VOTE,
            sender="n2", receiver="n1",
            term=1,
            data={"candidate_id": "n2", "last_log_index": 0, "last_log_term": 0, "next_term": 2}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["vote_granted"] is False


class TestClientRequest(unittest.TestCase):
    """Client request handling."""

    def test_submit_as_leader(self):
        random.seed(42)
        net = create_cluster(["n1"])
        leader = net.wait_for_leader()
        idx = leader.submit({"op": "set", "key": "x", "value": 1})
        assert idx is not None
        net.run(ticks=100)
        assert leader.state_machine.data.get("x") == 1

    def test_submit_as_follower_returns_none(self):
        node = RaftNode("n1", ["n2"])
        assert node.submit({"op": "set", "key": "x", "value": 1}) is None

    def test_client_request_to_follower(self):
        node = RaftNode("n1", ["n2"])
        node.leader_id = "n2"
        msg = Message(
            type=MessageType.CLIENT_REQUEST,
            sender="client", receiver="n1",
            term=0,
            data={"command": {"op": "set", "key": "x", "value": 1}}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["success"] is False
        assert resp.data["leader_id"] == "n2"

    def test_many_concurrent_writes(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        last_idx = None
        for i in range(50):
            last_idx = leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.wait_for_commit(last_idx, max_ticks=5000)
        net.run(ticks=500)
        for i in range(50):
            assert leader.state_machine.data.get(f"k{i}") == i

    def test_cas_through_raft(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        idx1 = leader.submit({"op": "set", "key": "x", "value": 1})
        net.wait_for_commit(idx1)
        net.run(ticks=100)
        idx2 = leader.submit({"op": "cas", "key": "x", "expected": 1, "value": 2})
        net.wait_for_commit(idx2)
        net.run(ticks=100)
        assert leader.state_machine.data["x"] == 2


class TestGetState(unittest.TestCase):
    """State inspection."""

    def test_get_state(self):
        node = RaftNode("n1", ["n2", "n3"])
        state = node.get_state()
        assert state["node_id"] == "n1"
        assert state["role"] == "follower"
        assert state["term"] == 0
        assert state["log_length"] == 0

    def test_get_state_after_election(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        state = leader.get_state()
        assert state["role"] == "leader"
        assert state["term"] >= 1


class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_message_to_unknown_node(self):
        net = Network()
        n1 = RaftNode("n1", ["n2"])
        net.add_node(n1)
        n1.outbox.append(Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n1", receiver="n99",
            term=1, data={}
        ))
        net.tick()
        assert len(net.dropped) == 1

    def test_empty_append_entries_heartbeat(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 1
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=1,
            data={"leader_id": "n2", "prev_log_index": 0, "prev_log_term": 0,
                  "entries": [], "leader_commit": 0}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["success"] is True
        assert node.leader_id == "n2"

    def test_double_vote_same_candidate(self):
        node = RaftNode("n1", ["n2", "n3"])
        node.current_term = 1
        node.voted_for = "n2"
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            sender="n2", receiver="n1",
            term=1,
            data={"candidate_id": "n2", "last_log_index": 0, "last_log_term": 0}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["vote_granted"] is True  # same candidate, ok

    def test_double_vote_different_candidate(self):
        node = RaftNode("n1", ["n2", "n3"])
        node.current_term = 1
        node.voted_for = "n2"
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            sender="n3", receiver="n1",
            term=1,
            data={"candidate_id": "n3", "last_log_index": 0, "last_log_term": 0}
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["vote_granted"] is False  # already voted for n2

    def test_removed_node_handling(self):
        net = Network()
        n1 = RaftNode("n1", ["n2"])
        net.add_node(n1)
        net.remove_node("n1")
        assert "n1" not in net.nodes

    def test_leader_commit_advances_with_replication(self):
        """Commit index should advance as followers replicate."""
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        idx = leader.submit({"op": "set", "key": "x", "value": 1})
        old_commit = leader.commit_index
        net.run(ticks=500)
        assert leader.commit_index >= idx

    def test_applied_results_tracked(self):
        random.seed(42)
        net = create_cluster(["n1"])
        leader = net.wait_for_leader()
        idx = leader.submit({"op": "set", "key": "x", "value": 42})
        net.run(ticks=200)
        assert leader.applied_results.get(idx) == 42

    def test_get_term_at_zero(self):
        node = RaftNode("n1", [])
        assert node._get_term_at(0) == 0

    def test_get_term_at_snapshot_boundary(self):
        node = RaftNode("n1", [])
        node.snapshot = Snapshot(last_included_index=5, last_included_term=3, data={})
        assert node._get_term_at(5) == 3

    def test_log_base_no_snapshot(self):
        node = RaftNode("n1", [])
        assert node._log_base_index() == 0
        assert node._log_base_term() == 0


class TestNetworkUtilities(unittest.TestCase):
    """Network utility method tests."""

    def test_create_cluster(self):
        net = create_cluster(["a", "b", "c"])
        assert len(net.nodes) == 3
        assert "a" in net.nodes
        assert set(net.nodes["a"].peers) == {"b", "c"}

    def test_wait_for_leader_timeout(self):
        """If no messages flow, no election can complete for multi-node."""
        net = Network()
        # All nodes partitioned from each other
        for nid in ["n1", "n2", "n3"]:
            peers = [x for x in ["n1", "n2", "n3"] if x != nid]
            net.add_node(RaftNode(nid, peers))
        net.isolate("n1")
        net.isolate("n2")
        net.isolate("n3")
        result = net.wait_for_leader(max_ticks=500)
        assert result is None

    def test_wait_for_replication(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        idx = leader.submit({"op": "set", "key": "x", "value": 1})
        result = net.wait_for_replication(idx, count=3, max_ticks=3000)
        assert result is True

    def test_wait_for_replication_timeout(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        # Isolate all followers
        for n in net.nodes.values():
            if n.node_id != leader.node_id:
                net.isolate(n.node_id)
        idx = leader.submit({"op": "set", "key": "x", "value": 1})
        result = net.wait_for_replication(idx, count=3, max_ticks=200)
        assert result is False


class TestStressAndConvergence(unittest.TestCase):
    """Stress tests for convergence."""

    def test_rapid_leader_changes(self):
        """Multiple elections should still converge."""
        random.seed(77)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        for _ in range(3):
            leader = net.wait_for_leader()
            if leader:
                net.isolate(leader.node_id)
                net.run(ticks=1000)
                net.heal()
                net.run(ticks=500)
        net.heal()
        leader = net.wait_for_leader(max_ticks=5000)
        assert leader is not None

    def test_all_data_consistent_after_churn(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3"])
        leader = net.wait_for_leader()
        # Write some data
        for i in range(5):
            leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.run(ticks=500)
        # Partition and heal
        net.isolate(leader.node_id)
        net.run(ticks=2000)
        net.heal()
        net.run(ticks=2000)
        # Find new leader and write more
        leader = net.find_leader()
        if leader:
            for i in range(5, 10):
                leader.submit({"op": "set", "key": f"k{i}", "value": i})
            net.run(ticks=1000)
        # Check consistency
        leader = net.find_leader()
        if leader:
            for i in range(5):
                assert leader.state_machine.data.get(f"k{i}") == i

    def test_log_convergence_five_nodes(self):
        random.seed(42)
        net = create_cluster(["n1", "n2", "n3", "n4", "n5"])
        leader = net.wait_for_leader()
        last = None
        for i in range(20):
            last = leader.submit({"op": "set", "key": f"k{i}", "value": i})
        net.wait_for_replication(last, count=5, max_ticks=5000)
        # All nodes should agree
        for node in net.nodes.values():
            for i in range(20):
                assert node.state_machine.data.get(f"k{i}") == i


class TestAutoCompaction(unittest.TestCase):
    """Auto-compaction tests."""

    def test_maybe_compact_below_threshold(self):
        node = RaftNode("n1", [])
        node.snapshot_threshold = 100
        for i in range(50):
            node.log.append(LogEntry(term=1, index=i + 1, command=None))
        result = node.maybe_compact()
        assert result is None

    def test_maybe_compact_above_threshold(self):
        node = RaftNode("n1", [])
        node.snapshot_threshold = 10
        node.current_term = 1
        node.role = Role.LEADER
        for i in range(1, 16):
            node.log.append(LogEntry(term=1, index=i, command={"op": "set", "key": f"k{i}", "value": i}))
            node.state_machine.apply({"op": "set", "key": f"k{i}", "value": i})
        node.last_applied = 15
        node.commit_index = 15
        result = node.maybe_compact()
        assert result is not None
        assert len(node.log) < 15

    def test_snapshot_no_double_compact(self):
        """Taking snapshot when last_applied <= base should return None."""
        node = RaftNode("n1", [])
        node.snapshot = Snapshot(last_included_index=10, last_included_term=1, data={})
        node.last_applied = 10
        result = node.take_snapshot()
        assert result is None


class TestAppendEntriesEdgeCases(unittest.TestCase):
    """Detailed append entries protocol tests."""

    def test_append_entries_updates_commit(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 1
        node.log.append(LogEntry(term=1, index=1, command="a"))
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=1,
            data={
                "leader_id": "n2", "prev_log_index": 1, "prev_log_term": 1,
                "entries": [], "leader_commit": 1,
            }
        )
        node.receive(msg)
        assert node.commit_index == 1

    def test_candidate_steps_down_on_append(self):
        node = RaftNode("n1", ["n2"])
        node.role = Role.CANDIDATE
        node.current_term = 2
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=2,
            data={
                "leader_id": "n2", "prev_log_index": 0, "prev_log_term": 0,
                "entries": [], "leader_commit": 0,
            }
        )
        node.receive(msg)
        assert node.role == Role.FOLLOWER

    def test_append_entries_at_snapshot_boundary(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 2
        node.snapshot = Snapshot(last_included_index=5, last_included_term=1, data={"a": 1})
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            sender="n2", receiver="n1",
            term=2,
            data={
                "leader_id": "n2", "prev_log_index": 5, "prev_log_term": 1,
                "entries": [{"term": 2, "index": 6, "command": "b", "entry_type": "command"}],
                "leader_commit": 5,
            }
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["success"] is True
        assert node._get_log_entry(6) is not None

    def test_install_snapshot_stale_term_rejected(self):
        node = RaftNode("n1", ["n2"])
        node.current_term = 5
        msg = Message(
            type=MessageType.INSTALL_SNAPSHOT,
            sender="n2", receiver="n1",
            term=3,
            data={
                "leader_id": "n2",
                "last_included_index": 10,
                "last_included_term": 2,
                "data": {},
            }
        )
        node.receive(msg)
        resp = node.outbox[-1]
        assert resp.data["success"] is False

    def test_install_snapshot_response_updates_leader(self):
        node = RaftNode("n1", ["n2"])
        node.role = Role.LEADER
        node.current_term = 3
        node.next_index["n2"] = 1
        node.match_index["n2"] = 0
        msg = Message(
            type=MessageType.INSTALL_SNAPSHOT_RESPONSE,
            sender="n2", receiver="n1",
            term=3,
            data={"success": True, "match_index": 10}
        )
        node.receive(msg)
        assert node.match_index["n2"] == 10
        assert node.next_index["n2"] == 11


class TestRoleEnum(unittest.TestCase):
    def test_role_values(self):
        assert Role.FOLLOWER.value == "follower"
        assert Role.CANDIDATE.value == "candidate"
        assert Role.LEADER.value == "leader"
        assert Role.LEARNER.value == "learner"


class TestMessageType(unittest.TestCase):
    def test_message_types(self):
        assert MessageType.REQUEST_VOTE.value == "request_vote"
        assert MessageType.APPEND_ENTRIES.value == "append_entries"
        assert MessageType.INSTALL_SNAPSHOT.value == "install_snapshot"
        assert MessageType.CLIENT_REQUEST.value == "client_request"


class TestLeaderSelfRemoval(unittest.TestCase):
    def test_leader_steps_down_on_self_removal(self):
        node = RaftNode("n1", ["n2", "n3"])
        node.role = Role.LEADER
        node.current_term = 1
        node._apply_config_change({"op": "remove_member", "node_id": "n1", "config": ["n2", "n3"]})
        assert node.role == Role.FOLLOWER


if __name__ == "__main__":
    unittest.main()
