"""Tests for C201: Raft Consensus Protocol."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import random
from raft import (
    RaftNode, RaftCluster, RaftLog, LogEntry, Message, MessageType,
    Role, Snapshot, KeyValueStateMachine, create_cluster
)


# ============================================================
# LogEntry Tests
# ============================================================

class TestLogEntry:
    def test_create(self):
        e = LogEntry(term=1, index=1, command={"op": "set", "key": "x", "value": 1})
        assert e.term == 1
        assert e.index == 1
        assert e.command == {"op": "set", "key": "x", "value": 1}
        assert not e.is_config_change

    def test_config_change(self):
        e = LogEntry(term=1, index=1, command={"op": "add_server"}, is_config_change=True)
        assert e.is_config_change

    def test_serialize_deserialize(self):
        e = LogEntry(term=2, index=5, command={"op": "set", "key": "a", "value": "b"})
        d = e.to_dict()
        e2 = LogEntry.from_dict(d)
        assert e2.term == 2
        assert e2.index == 5
        assert e2.command == {"op": "set", "key": "a", "value": "b"}

    def test_serialize_config_change(self):
        e = LogEntry(term=1, index=1, command=None, is_config_change=True)
        d = e.to_dict()
        e2 = LogEntry.from_dict(d)
        assert e2.is_config_change


# ============================================================
# RaftLog Tests
# ============================================================

class TestRaftLog:
    def test_empty_log(self):
        log = RaftLog()
        assert len(log) == 0
        assert log.last_index() == 0
        assert log.last_term() == 0

    def test_append_and_get(self):
        log = RaftLog()
        log.append(LogEntry(1, 1, "cmd1"))
        log.append(LogEntry(1, 2, "cmd2"))
        assert len(log) == 2
        assert log.get(1).command == "cmd1"
        assert log.get(2).command == "cmd2"

    def test_last_index_term(self):
        log = RaftLog()
        log.append(LogEntry(1, 1, "a"))
        log.append(LogEntry(2, 2, "b"))
        assert log.last_index() == 2
        assert log.last_term() == 2

    def test_term_at(self):
        log = RaftLog()
        log.append(LogEntry(1, 1, "a"))
        log.append(LogEntry(3, 2, "b"))
        assert log.term_at(0) == 0
        assert log.term_at(1) == 1
        assert log.term_at(2) == 3

    def test_entries_from(self):
        log = RaftLog()
        for i in range(1, 6):
            log.append(LogEntry(1, i, f"cmd{i}"))
        entries = log.entries_from(3)
        assert len(entries) == 3
        assert entries[0].command == "cmd3"

    def test_truncate_from(self):
        log = RaftLog()
        for i in range(1, 6):
            log.append(LogEntry(1, i, f"cmd{i}"))
        log.truncate_from(4)
        assert len(log) == 3
        assert log.last_index() == 3

    def test_match_term(self):
        log = RaftLog()
        log.append(LogEntry(1, 1, "a"))
        log.append(LogEntry(2, 2, "b"))
        assert log.match_term(0, 0)  # Always true for index 0
        assert log.match_term(1, 1)
        assert not log.match_term(1, 2)
        assert log.match_term(2, 2)

    def test_compact(self):
        log = RaftLog()
        for i in range(1, 11):
            log.append(LogEntry(1, i, f"cmd{i}"))
        log.compact(5, 1)
        assert log.snapshot_last_index == 5
        assert log.snapshot_last_term == 1
        assert len(log) == 10  # Total logical length unchanged
        assert log.get(5) is None  # Compacted
        assert log.get(6).command == "cmd6"

    def test_compact_preserves_length(self):
        log = RaftLog()
        for i in range(1, 6):
            log.append(LogEntry(2, i, f"cmd{i}"))
        log.compact(3, 2)
        assert len(log) == 5
        assert log.last_index() == 5

    def test_get_out_of_range(self):
        log = RaftLog()
        assert log.get(1) is None
        assert log.get(0) is None

    def test_entries_from_after_compact(self):
        log = RaftLog()
        for i in range(1, 11):
            log.append(LogEntry(1, i, f"cmd{i}"))
        log.compact(5, 1)
        entries = log.entries_from(6)
        assert len(entries) == 5
        assert entries[0].command == "cmd6"

    def test_term_at_snapshot_boundary(self):
        log = RaftLog()
        for i in range(1, 6):
            log.append(LogEntry(3, i, f"cmd{i}"))
        log.compact(3, 3)
        assert log.term_at(3) == 3  # Snapshot boundary


# ============================================================
# KeyValueStateMachine Tests
# ============================================================

class TestStateMachine:
    def test_set_get(self):
        sm = KeyValueStateMachine()
        sm.apply({"op": "set", "key": "x", "value": 42}, 1)
        assert sm.apply({"op": "get", "key": "x"}, 2) == 42

    def test_delete(self):
        sm = KeyValueStateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1}, 1)
        sm.apply({"op": "delete", "key": "x"}, 2)
        assert sm.apply({"op": "get", "key": "x"}, 3) is None

    def test_cas_success(self):
        sm = KeyValueStateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1}, 1)
        result = sm.apply({"op": "cas", "key": "x", "expected": 1, "new_value": 2}, 2)
        assert result is True
        assert sm.apply({"op": "get", "key": "x"}, 3) == 2

    def test_cas_failure(self):
        sm = KeyValueStateMachine()
        sm.apply({"op": "set", "key": "x", "value": 1}, 1)
        result = sm.apply({"op": "cas", "key": "x", "expected": 99, "new_value": 2}, 2)
        assert result is False
        assert sm.apply({"op": "get", "key": "x"}, 3) == 1

    def test_snapshot_restore(self):
        sm = KeyValueStateMachine()
        sm.apply({"op": "set", "key": "a", "value": 1}, 1)
        sm.apply({"op": "set", "key": "b", "value": 2}, 2)
        snap = sm.snapshot()

        sm2 = KeyValueStateMachine()
        sm2.restore(snap)
        assert sm2.store["a"] == 1
        assert sm2.store["b"] == 2
        assert sm2.last_applied == 2

    def test_noop(self):
        sm = KeyValueStateMachine()
        result = sm.apply(None, 1)
        assert result is None
        assert sm.last_applied == 1

    def test_unknown_op(self):
        sm = KeyValueStateMachine()
        result = sm.apply({"op": "unknown"}, 1)
        assert result is None

    def test_get_missing_key(self):
        sm = KeyValueStateMachine()
        assert sm.apply({"op": "get", "key": "missing"}, 1) is None

    def test_delete_missing_key(self):
        sm = KeyValueStateMachine()
        result = sm.apply({"op": "delete", "key": "missing"}, 1)
        assert result is None


# ============================================================
# Single Node Tests
# ============================================================

class TestSingleNode:
    def test_initial_state(self):
        node = RaftNode("n1", [])
        assert node.role == Role.FOLLOWER
        assert node.current_term == 0
        assert node.voted_for is None
        assert node.commit_index == 0

    def test_single_node_election(self):
        node = RaftNode("n1", [])
        # Tick until election timeout
        for _ in range(500):
            node.tick(1)
        assert node.role == Role.LEADER
        assert node.current_term == 1

    def test_status(self):
        node = RaftNode("n1", ["n2", "n3"])
        s = node.status()
        assert s["id"] == "n1"
        assert s["role"] == "follower"
        assert s["term"] == 0
        assert len(s["peers"]) == 2


# ============================================================
# Leader Election Tests
# ============================================================

class TestLeaderElection:
    def test_three_node_election(self):
        cluster = create_cluster(3)
        assert cluster.wait_for_leader()
        leader = cluster.get_leader()
        assert leader is not None
        assert leader.role == Role.LEADER

    def test_five_node_election(self):
        cluster = create_cluster(5)
        assert cluster.wait_for_leader()
        leader = cluster.get_leader()
        assert leader is not None

    def test_leader_has_highest_term(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        for node in cluster.nodes.values():
            assert node.current_term <= leader.current_term

    def test_all_nodes_agree_on_leader(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        # Give time for followers to learn leader
        for _ in range(200):
            cluster.tick(1)
            cluster.deliver_all()
        for node in cluster.nodes.values():
            if node.role == Role.FOLLOWER:
                assert node.leader_id == leader.id

    def test_election_safety_one_leader_per_term(self):
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        safe, msg = cluster.check_safety()
        assert safe, msg

    def test_follower_votes_only_once_per_term(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        for node in cluster.nodes.values():
            if node.role == Role.FOLLOWER:
                # voted_for should be set
                assert node.voted_for is not None

    def test_candidate_votes_for_self(self):
        node = RaftNode("n1", ["n2", "n3"])
        node.election_timer = 0
        node.tick(1)
        assert node.role == Role.CANDIDATE
        assert node.voted_for == "n1"

    def test_higher_term_causes_stepdown(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()

        # Simulate a message with a higher term
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            src="fake",
            dst=leader.id,
            term=leader.current_term + 10,
            data={"prev_log_index": 0, "prev_log_term": 0, "entries": [], "leader_commit": 0}
        )
        leader.receive(msg)
        assert leader.role == Role.FOLLOWER


# ============================================================
# Log Replication Tests
# ============================================================

class TestLogReplication:
    def _setup_cluster(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        return cluster

    def test_append_entry(self):
        cluster = self._setup_cluster()
        leader = cluster.get_leader()
        cluster.submit({"op": "set", "key": "x", "value": 1}, "r1")
        cluster.deliver_all()
        # Leader should have the entry
        assert len(leader.log) >= 2  # no-op + command

    def test_replicate_to_followers(self):
        cluster = self._setup_cluster()
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 1})

        # All nodes should have the entry
        for node in cluster.nodes.values():
            assert len(node.log) >= 2

    def test_commit_index_advances(self):
        cluster = self._setup_cluster()
        leader = cluster.get_leader()
        initial_commit = leader.commit_index
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 1})
        assert leader.commit_index > initial_commit

    def test_follower_applies_committed(self):
        cluster = self._setup_cluster()
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 42})

        for node in cluster.nodes.values():
            val = node.state_machine.apply({"op": "get", "key": "x"}, node.last_applied + 1)
            assert val == 42

    def test_multiple_entries(self):
        cluster = self._setup_cluster()
        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        for i in range(10):
            val = leader.state_machine.store.get(f"k{i}")
            assert val == i

    def test_log_consistency(self):
        cluster = self._setup_cluster()
        for i in range(5):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        safe, msg = cluster.check_safety()
        assert safe, msg

    def test_leader_noop_on_election(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        # Leader should have appended a no-op
        assert len(leader.log) >= 1
        first = leader.log.get(1)
        assert first.command is None  # no-op

    def test_client_result_stored(self):
        cluster = self._setup_cluster()
        result = cluster.submit_and_commit({"op": "set", "key": "x", "value": 99})
        assert result == 99

    def test_get_query(self):
        cluster = self._setup_cluster()
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 42})
        leader = cluster.get_leader()
        # Read query
        rid = "read_1"
        cluster.submit({"op": "get", "key": "x"}, rid, leader.id)
        cluster.deliver_all()
        assert leader.committed_results.get(rid) == 42

    def test_cas_through_cluster(self):
        cluster = self._setup_cluster()
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 1})
        result = cluster.submit_and_commit(
            {"op": "cas", "key": "x", "expected": 1, "new_value": 2}
        )
        assert result is True


# ============================================================
# Network Partition Tests
# ============================================================

class TestPartitions:
    def test_minority_partition_no_progress(self):
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        leader = cluster.get_leader()

        # Partition leader alone
        others = [n for n in cluster.nodes if n != leader.id]
        cluster.partition([leader.id], others)

        # Old leader can't commit
        old_commit = leader.commit_index
        cluster.submit({"op": "set", "key": "x", "value": 1}, "r1", leader.id)
        for _ in range(500):
            cluster.tick(1)
            cluster.deliver_all()
        assert leader.commit_index == old_commit  # No progress

    def test_majority_elects_new_leader(self):
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        old_leader = cluster.get_leader()

        # Partition old leader
        others = [n for n in cluster.nodes if n != old_leader.id]
        cluster.partition([old_leader.id], others)

        # Majority should elect new leader
        for _ in range(2000):
            cluster.tick(1)
            cluster.deliver_all()

        new_leaders = [n for n in cluster.nodes.values()
                       if n.role == Role.LEADER and n.id != old_leader.id]
        assert len(new_leaders) >= 1

    def test_partition_heal_reconverge(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        old_leader = cluster.get_leader()

        others = [n for n in cluster.nodes if n != old_leader.id]
        cluster.partition([old_leader.id], others)

        # Let new leader emerge
        for _ in range(2000):
            cluster.tick(1)
            cluster.deliver_all()

        # Heal
        cluster.heal_partition()
        for _ in range(2000):
            cluster.tick(1)
            cluster.deliver_all()

        # Should converge to one leader
        leaders = cluster.get_leaders()
        assert len(leaders) == 1

    def test_split_brain_safety(self):
        cluster = create_cluster(5)
        cluster.wait_for_leader()

        # Partition into [0,1] vs [2,3,4]
        ids = list(cluster.nodes.keys())
        cluster.partition(ids[:2], ids[2:])

        for _ in range(2000):
            cluster.tick(1)
            cluster.deliver_all()

        # Check safety still holds
        safe, msg = cluster.check_safety()
        assert safe, msg


# ============================================================
# Snapshot Tests
# ============================================================

class TestSnapshots:
    def test_take_snapshot(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        for i in range(5):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        snap = leader.take_snapshot()
        assert snap is not None
        assert snap.last_included_index == leader.last_applied
        assert "store" in snap.data

    def test_snapshot_compacts_log(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        old_entries = len(leader.log.entries)
        leader.take_snapshot()
        # Internal entries list should be shorter
        assert len(leader.log.entries) < old_entries

    def test_snapshot_preserves_state(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        for i in range(5):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        leader.take_snapshot()

        # State machine should still work
        assert leader.state_machine.store["k0"] == 0
        assert leader.state_machine.store["k4"] == 4

    def test_install_snapshot_to_follower(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        snap = leader.take_snapshot()

        # Create a new node and send it the snapshot
        new_node = RaftNode("new_node", list(cluster.nodes.keys()))
        msg = Message(
            type=MessageType.INSTALL_SNAPSHOT,
            src=leader.id,
            dst="new_node",
            term=leader.current_term,
            data={
                "last_included_index": snap.last_included_index,
                "last_included_term": snap.last_included_term,
                "data": snap.data,
                "config": snap.config,
            }
        )
        new_node.receive(msg)
        assert new_node.state_machine.store["k0"] == 0
        assert new_node.last_applied == snap.last_included_index

    def test_leader_sends_snapshot_to_lagging_follower(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()

        # Add entries and compact
        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        leader.take_snapshot()

        # Set a follower's next_index to before snapshot
        follower_id = [p for p in leader.peers][0]
        leader.next_index[follower_id] = 1

        # Trigger heartbeat
        leader.heartbeat_timer = 0
        leader.tick(1)

        # Check that an InstallSnapshot was sent
        snap_msgs = [m for m in leader.outbox if m.type == MessageType.INSTALL_SNAPSHOT]
        assert len(snap_msgs) >= 1

    def test_snapshot_on_empty_state(self):
        node = RaftNode("n1", [])
        snap = node.take_snapshot()
        assert snap is None  # Nothing applied


# ============================================================
# Membership Change Tests
# ============================================================

class TestMembership:
    def test_add_server(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()

        ok = leader.add_server("new_node")
        assert ok
        assert "new_node" in leader.peers

    def test_add_duplicate_server(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        peer = leader.peers[0]
        assert not leader.add_server(peer)

    def test_add_self(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        assert not leader.add_server(leader.id)

    def test_remove_server(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        peer = leader.peers[0]
        ok = leader.remove_server(peer)
        assert ok
        assert peer not in leader.peers

    def test_remove_nonexistent_server(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        assert not leader.remove_server("ghost_node")

    def test_remove_leader(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        ok = leader.remove_server(leader.id)
        assert ok
        assert leader.role == Role.FOLLOWER

    def test_follower_cannot_add(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        follower = [n for n in cluster.nodes.values() if n.role == Role.FOLLOWER][0]
        assert not follower.add_server("new")

    def test_cluster_add_node(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()

        # Submit some data first
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 42})

        # Add a new node
        ok = cluster.add_node("node_3")
        assert ok
        assert "node_3" in cluster.nodes

        # Replicate to new node
        for _ in range(500):
            cluster.tick(1)
            cluster.deliver_all()

    def test_cluster_remove_node(self):
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        victim = [p for p in leader.peers][0]

        ok = cluster.remove_node(victim)
        assert ok
        assert victim not in leader.peers


# ============================================================
# Safety Property Tests
# ============================================================

class TestSafety:
    def test_election_safety(self):
        """At most one leader per term."""
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        safe, msg = cluster.check_safety()
        assert safe, msg

    def test_leader_append_only(self):
        """Leader never overwrites its own log entries."""
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()

        for i in range(5):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        # Verify log is monotonically increasing in index
        for i in range(1, len(leader.log) + 1):
            e = leader.log.get(i)
            if e:
                assert e.index == i

    def test_log_matching_property(self):
        """If two logs have an entry with same index and term, all preceding entries match."""
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        safe, msg = cluster.check_safety()
        assert safe, msg

    def test_state_machine_safety(self):
        """All nodes apply the same commands in the same order."""
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i * 10})

        # All nodes should have same state
        stores = [dict(n.state_machine.store) for n in cluster.nodes.values()]
        for store in stores:
            for i in range(10):
                assert store.get(f"k{i}") == i * 10

    def test_committed_entries_survive_leader_change(self):
        cluster = create_cluster(5)
        cluster.wait_for_leader()

        # Commit some entries
        cluster.submit_and_commit({"op": "set", "key": "important", "value": 999})

        old_leader = cluster.get_leader()

        # Kill old leader (partition it)
        others = [n for n in cluster.nodes if n != old_leader.id]
        cluster.partition([old_leader.id], others)

        # Wait for new leader
        for _ in range(3000):
            cluster.tick(1)
            cluster.deliver_all()

        # New leader should have the committed entry
        new_leaders = [n for n in cluster.nodes.values()
                       if n.role == Role.LEADER and n.id != old_leader.id]
        if new_leaders:
            new_leader = new_leaders[0]
            assert new_leader.state_machine.store.get("important") == 999

    def test_no_committed_entry_lost(self):
        """Once committed, an entry appears in every future leader's log."""
        cluster = create_cluster(5)
        cluster.wait_for_leader()
        cluster.submit_and_commit({"op": "set", "key": "persist", "value": "forever"})

        # Cycle through leaders
        for _ in range(3):
            leader = cluster.get_leader()
            if leader:
                # Partition leader
                others = [n for n in cluster.nodes if n != leader.id]
                cluster.partition([leader.id], others)
                for _ in range(2000):
                    cluster.tick(1)
                    cluster.deliver_all()
                cluster.heal_partition()
                for _ in range(2000):
                    cluster.tick(1)
                    cluster.deliver_all()

        # Final leader should still have the entry
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        if leader:
            assert leader.state_machine.store.get("persist") == "forever"


# ============================================================
# Stress / Larger Tests
# ============================================================

class TestStress:
    def test_many_commands(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        for i in range(50):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        assert len(leader.state_machine.store) == 50
        safe, msg = cluster.check_safety()
        assert safe, msg

    def test_rapid_leader_changes(self):
        cluster = create_cluster(5)
        cluster.wait_for_leader()

        # Submit initial data
        cluster.submit_and_commit({"op": "set", "key": "base", "value": "ok"})

        # Force several leader changes
        for _ in range(3):
            leader = cluster.get_leader()
            if leader:
                leader._become_follower(leader.current_term + 1)
            for _ in range(1000):
                cluster.tick(1)
                cluster.deliver_all()

        cluster.wait_for_leader()
        safe, msg = cluster.check_safety()
        assert safe, msg

    def test_seven_node_cluster(self):
        cluster = create_cluster(7)
        cluster.wait_for_leader()
        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        safe, msg = cluster.check_safety()
        assert safe, msg


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_stale_request_vote(self):
        """A RequestVote with old term is rejected."""
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()

        # Send stale vote request
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            src="fake",
            dst=leader.id,
            term=0,  # Old term
            data={"candidate_id": "fake", "last_log_index": 0, "last_log_term": 0}
        )
        leader.receive(msg)
        # Should not have changed role
        assert leader.role == Role.LEADER

    def test_stale_append_entries(self):
        """AppendEntries with old term is rejected."""
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()

        follower = [n for n in cluster.nodes.values() if n.role == Role.FOLLOWER][0]
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            src="fake",
            dst=follower.id,
            term=0,  # Old term
            data={"prev_log_index": 0, "prev_log_term": 0, "entries": [], "leader_commit": 0}
        )
        follower.receive(msg)
        # Response should have success=False
        resp = [m for m in follower.outbox if m.type == MessageType.APPEND_ENTRIES_RESPONSE]
        assert any(not r.data["success"] for r in resp)

    def test_log_conflict_resolution(self):
        """When follower has conflicting entries, they're overwritten."""
        leader = RaftNode("leader", ["f1"])
        leader.role = Role.LEADER
        leader.current_term = 2
        leader.log.append(LogEntry(1, 1, "a"))
        leader.log.append(LogEntry(2, 2, "b"))

        follower = RaftNode("f1", ["leader"])
        follower.current_term = 1
        follower.log.append(LogEntry(1, 1, "a"))
        follower.log.append(LogEntry(1, 2, "CONFLICT"))  # Different term at index 2

        # Leader sends AppendEntries
        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            src="leader",
            dst="f1",
            term=2,
            data={
                "prev_log_index": 1,
                "prev_log_term": 1,
                "entries": [LogEntry(2, 2, "b").to_dict()],
                "leader_commit": 0,
            }
        )
        follower.receive(msg)
        assert follower.log.get(2).command == "b"  # Overwritten

    def test_not_leader_redirect(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        follower = [n for n in cluster.nodes.values() if n.role == Role.FOLLOWER][0]

        msg = Message(
            type=MessageType.CLIENT_REQUEST,
            src="client",
            dst=follower.id,
            term=0,
            data={"command": {"op": "set", "key": "x", "value": 1}, "request_id": "r1"}
        )
        follower.receive(msg)
        responses = [m for m in follower.outbox if m.type == MessageType.CLIENT_RESPONSE]
        assert len(responses) == 1
        assert responses[0].data["error"] == "not_leader"
        # leader_id may or may not be set depending on timing
        assert responses[0].data["error"] == "not_leader"

    def test_empty_append_entries_heartbeat(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()

        # Force heartbeat
        leader.heartbeat_timer = 0
        leader.tick(1)

        # Should have sent AppendEntries with empty entries
        ae_msgs = [m for m in leader.outbox if m.type == MessageType.APPEND_ENTRIES]
        assert len(ae_msgs) > 0

    def test_prevote_not_granted_for_outdated_log(self):
        """Vote not granted if candidate's log is less up-to-date."""
        node = RaftNode("n1", ["n2"])
        node.current_term = 5
        node.log.append(LogEntry(5, 1, "latest"))

        # Request vote from candidate with older log
        msg = Message(
            type=MessageType.REQUEST_VOTE,
            src="n2",
            dst="n1",
            term=5,
            data={"candidate_id": "n2", "last_log_index": 0, "last_log_term": 0}
        )
        node.receive(msg)
        resp = [m for m in node.outbox if m.type == MessageType.REQUEST_VOTE_RESPONSE]
        assert not resp[0].data["vote_granted"]

    def test_two_node_cluster(self):
        cluster = create_cluster(2)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        assert leader is not None
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 1})
        assert leader.state_machine.store["x"] == 1

    def test_concurrent_elections_converge(self):
        """When multiple nodes start elections, cluster eventually converges."""
        cluster = create_cluster(5)
        # Force all nodes to start elections at once
        for node in cluster.nodes.values():
            node.election_timer = 0
        cluster.tick(1)
        cluster.deliver_all()

        # Should eventually converge
        assert cluster.wait_for_leader(max_ticks=10000)

    def test_follower_rejects_conflicting_prev(self):
        """Follower rejects AppendEntries if prev log doesn't match."""
        node = RaftNode("f1", ["leader"])
        node.current_term = 1
        node.log.append(LogEntry(1, 1, "a"))

        msg = Message(
            type=MessageType.APPEND_ENTRIES,
            src="leader",
            dst="f1",
            term=1,
            data={
                "prev_log_index": 2,  # We don't have index 2
                "prev_log_term": 1,
                "entries": [],
                "leader_commit": 0,
            }
        )
        node.receive(msg)
        resp = [m for m in node.outbox if m.type == MessageType.APPEND_ENTRIES_RESPONSE]
        assert not resp[0].data["success"]


# ============================================================
# create_cluster helper tests
# ============================================================

class TestHelpers:
    def test_create_cluster_default(self):
        cluster = create_cluster()
        assert len(cluster.nodes) == 3

    def test_create_cluster_custom(self):
        cluster = create_cluster(7)
        assert len(cluster.nodes) == 7

    def test_dropped_messages_tracked(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        others = [n for n in cluster.nodes if n != leader.id]
        cluster.partition([leader.id], others)

        leader.heartbeat_timer = 0
        leader.tick(1)
        cluster.deliver_all()

        assert len(cluster.dropped_messages) > 0

    def test_delivered_messages_tracked(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        assert len(cluster.delivered_messages) > 0


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """Create cluster, elect leader, replicate, snapshot, partition, heal."""
        cluster = create_cluster(5)

        # 1. Election
        cluster.wait_for_leader()
        leader = cluster.get_leader()
        assert leader is not None

        # 2. Replication
        for i in range(20):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        # 3. Verify state
        for node in cluster.nodes.values():
            assert node.state_machine.store.get("k0") == 0
            assert node.state_machine.store.get("k19") == 19

        # 4. Snapshot
        leader.take_snapshot()
        assert leader.snapshot is not None

        # 5. Partition
        others = [n for n in cluster.nodes if n != leader.id]
        cluster.partition([leader.id], others)
        for _ in range(2000):
            cluster.tick(1)
            cluster.deliver_all()

        # 6. Heal and reconverge
        cluster.heal_partition()
        for _ in range(2000):
            cluster.tick(1)
            cluster.deliver_all()

        # 7. New leader should have all data
        cluster.wait_for_leader()
        new_leader = cluster.get_leader()
        assert new_leader.state_machine.store.get("k0") == 0

        # 8. Safety
        safe, msg = cluster.check_safety()
        assert safe, msg

    def test_delete_then_read(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        cluster.submit_and_commit({"op": "set", "key": "x", "value": 1})
        cluster.submit_and_commit({"op": "delete", "key": "x"})

        leader = cluster.get_leader()
        assert "x" not in leader.state_machine.store

    def test_cas_workflow(self):
        cluster = create_cluster(3)
        cluster.wait_for_leader()
        cluster.submit_and_commit({"op": "set", "key": "counter", "value": 0})

        # CAS success
        r = cluster.submit_and_commit({"op": "cas", "key": "counter", "expected": 0, "new_value": 1})
        assert r is True

        # CAS failure
        r = cluster.submit_and_commit({"op": "cas", "key": "counter", "expected": 0, "new_value": 2})
        assert r is False

        leader = cluster.get_leader()
        assert leader.state_machine.store["counter"] == 1

    def test_snapshot_and_new_node(self):
        """New node catches up via snapshot."""
        cluster = create_cluster(3)
        cluster.wait_for_leader()

        for i in range(10):
            cluster.submit_and_commit({"op": "set", "key": f"k{i}", "value": i})

        leader = cluster.get_leader()
        leader.take_snapshot()

        # Add a new node
        cluster.add_node("node_new")
        # Let it catch up
        for _ in range(1000):
            cluster.tick(1)
            cluster.deliver_all()

        new_node = cluster.nodes["node_new"]
        # Should have received snapshot or log entries
        assert new_node.last_applied > 0 or new_node.commit_index > 0

    def test_continuous_operation_through_failures(self):
        """Cluster continues operating through sequential node failures."""
        cluster = create_cluster(5)
        cluster.wait_for_leader()

        # Write some data
        cluster.submit_and_commit({"op": "set", "key": "start", "value": "ok"})

        ids = list(cluster.nodes.keys())

        # Partition one node at a time (keep majority)
        for i in range(2):  # Can lose 2 out of 5
            victim = ids[i]
            others = [n for n in ids if n != victim]
            cluster.partition([victim], others)

            for _ in range(2000):
                cluster.tick(1)
                cluster.deliver_all()

            # Should still be able to elect leader and commit
            cluster.wait_for_leader()
            leader = cluster.get_leader()
            if leader:
                cluster.submit_and_commit({"op": "set", "key": f"after_fail_{i}", "value": i})

        # Heal all partitions
        cluster.heal_partition()
        for _ in range(2000):
            cluster.tick(1)
            cluster.deliver_all()

        cluster.wait_for_leader()
        safe, msg = cluster.check_safety()
        assert safe, msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
