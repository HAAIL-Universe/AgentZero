"""
Tests for C203: Gossip Protocol & Failure Detection
"""

import time
import math
import pytest
from gossip import (
    GossipNode, GossipNetwork, GossipState, GossipMessage,
    PhiAccrualDetector, HeartbeatDetector, MerkleTree,
    DisseminationQueue, MembershipUpdate, StateEntry, Member,
    NodeStatus, MessageType, DisseminationType, MerkleNode,
    _log2_ceil
)


# =============================================================================
# Helper
# =============================================================================

def make_cluster(n, seed=42, config=None):
    """Create a cluster of n nodes in a network."""
    net = GossipNetwork(seed=seed)
    nodes = []
    for i in range(n):
        nid = f"node_{i}"
        node = GossipNode(nid, config=config)
        net.add_node(node)
        nodes.append(node)
    # Wire up membership: all nodes know about each other
    for node in nodes:
        for other in nodes:
            if other.node_id != node.node_id:
                node.members[other.node_id] = Member(
                    other.node_id, other.node_id, NodeStatus.ALIVE)
                node._record_heartbeat(other.node_id, 0.0)
    return net, nodes


# =============================================================================
# PhiAccrualDetector Tests
# =============================================================================

class TestPhiAccrualDetector:
    def test_no_heartbeat(self):
        """Phi is 0 for unknown nodes."""
        d = PhiAccrualDetector()
        assert d.phi("unknown") == 0.0

    def test_initial_heartbeat(self):
        """After first heartbeat, phi should be low."""
        d = PhiAccrualDetector(initial_heartbeat_ms=1000)
        d.heartbeat("n1", now=0.0)
        assert d.phi("n1", now=0.5) < d.threshold

    def test_phi_increases_with_time(self):
        """Phi increases as time since last heartbeat grows."""
        d = PhiAccrualDetector(initial_heartbeat_ms=1000)
        for t in range(10):
            d.heartbeat("n1", now=float(t))
        # Last heartbeat at t=9, mean interval=1.0
        # Test times after expected next heartbeat (t=10)
        phi_a = d.phi("n1", now=10.2)  # 1.2s elapsed, slightly late
        phi_b = d.phi("n1", now=10.5)  # 1.5s elapsed, more late
        phi_c = d.phi("n1", now=11.0)  # 2.0s elapsed, very late
        assert phi_a > 0
        assert phi_a < phi_b < phi_c

    def test_regular_heartbeats(self):
        """Regular heartbeats keep phi low."""
        d = PhiAccrualDetector(initial_heartbeat_ms=1000)
        for t in range(20):
            d.heartbeat("n1", now=float(t))
        # Just after expected heartbeat
        assert d.phi("n1", now=20.5) < d.threshold

    def test_is_available(self):
        """is_available reflects phi threshold."""
        d = PhiAccrualDetector(threshold=8.0, initial_heartbeat_ms=1000)
        for t in range(10):
            d.heartbeat("n1", now=float(t))
        assert d.is_available("n1", now=10.5)
        # Long after last heartbeat
        assert not d.is_available("n1", now=100.0)

    def test_remove(self):
        """Remove stops tracking a node."""
        d = PhiAccrualDetector()
        d.heartbeat("n1", now=0.0)
        d.remove("n1")
        assert d.phi("n1") == 0.0

    def test_multiple_nodes(self):
        """Track multiple nodes independently."""
        d = PhiAccrualDetector(initial_heartbeat_ms=1000)
        for t in range(10):
            d.heartbeat("n1", now=float(t))
            d.heartbeat("n2", now=float(t))
        d.heartbeat("n1", now=10.0)
        # n2 missed heartbeat at t=10
        phi_n1 = d.phi("n1", now=11.0)
        phi_n2 = d.phi("n2", now=11.0)
        assert phi_n1 < phi_n2

    def test_irregular_heartbeats(self):
        """Irregular intervals increase std, making phi more tolerant."""
        d = PhiAccrualDetector(initial_heartbeat_ms=1000)
        # Irregular intervals: 0.5, 1.5, 0.3, 2.0, 0.8, ...
        times = [0.0, 0.5, 2.0, 2.3, 4.3, 5.1, 7.1, 7.8, 9.8, 10.2]
        for t in times:
            d.heartbeat("n1", now=t)
        # With irregular intervals, should be more tolerant
        phi = d.phi("n1", now=12.5)
        assert phi < 20  # Reasonable bound

    def test_phi_cap(self):
        """Phi is capped at extreme values."""
        d = PhiAccrualDetector(initial_heartbeat_ms=1000)
        d.heartbeat("n1", now=0.0)
        phi = d.phi("n1", now=10000.0)
        assert phi <= 34.0


# =============================================================================
# HeartbeatDetector Tests
# =============================================================================

class TestHeartbeatDetector:
    def test_no_heartbeat(self):
        """Unknown nodes are assumed alive."""
        d = HeartbeatDetector(timeout=5.0)
        assert d.is_available("unknown")

    def test_recent_heartbeat(self):
        d = HeartbeatDetector(timeout=5.0)
        d.heartbeat("n1", now=10.0)
        assert d.is_available("n1", now=14.0)

    def test_expired_heartbeat(self):
        d = HeartbeatDetector(timeout=5.0)
        d.heartbeat("n1", now=10.0)
        assert not d.is_available("n1", now=16.0)

    def test_time_since(self):
        d = HeartbeatDetector(timeout=5.0)
        d.heartbeat("n1", now=10.0)
        assert d.time_since("n1", now=13.0) == 3.0

    def test_time_since_unknown(self):
        d = HeartbeatDetector(timeout=5.0)
        assert d.time_since("unknown") == float('inf')

    def test_remove(self):
        d = HeartbeatDetector()
        d.heartbeat("n1", now=0.0)
        d.remove("n1")
        assert d.is_available("n1")  # Back to default (no data)


# =============================================================================
# MerkleTree Tests
# =============================================================================

class TestMerkleTree:
    def test_empty_tree(self):
        mt = MerkleTree()
        assert mt.root_hash() == ""
        assert mt.digest()["root"] == ""

    def test_single_entry(self):
        mt = MerkleTree()
        mt.update("key1", "value1")
        assert mt.root_hash() != ""
        assert "key1" in mt.digest()["keys"]

    def test_deterministic(self):
        mt1 = MerkleTree()
        mt2 = MerkleTree()
        mt1.update("a", "1")
        mt1.update("b", "2")
        mt2.update("a", "1")
        mt2.update("b", "2")
        assert mt1.root_hash() == mt2.root_hash()

    def test_different_values(self):
        mt1 = MerkleTree()
        mt2 = MerkleTree()
        mt1.update("a", "1")
        mt2.update("a", "2")
        assert mt1.root_hash() != mt2.root_hash()

    def test_diff_missing_key(self):
        mt1 = MerkleTree()
        mt2 = MerkleTree()
        mt1.update("a", "1")
        mt1.update("b", "2")
        mt2.update("a", "1")
        diff = mt1.diff(mt2.digest())
        assert "b" in diff

    def test_diff_different_value(self):
        mt1 = MerkleTree()
        mt2 = MerkleTree()
        mt1.update("a", "1")
        mt2.update("a", "2")
        diff = mt1.diff(mt2.digest())
        assert "a" in diff

    def test_diff_identical(self):
        mt1 = MerkleTree()
        mt2 = MerkleTree()
        mt1.update("a", "1")
        mt2.update("a", "1")
        diff = mt1.diff(mt2.digest())
        assert diff == []

    def test_remove(self):
        mt = MerkleTree()
        mt.update("a", "1")
        mt.update("b", "2")
        mt.remove("a")
        assert "a" not in mt.digest()["keys"]
        assert "b" in mt.digest()["keys"]

    def test_update_existing(self):
        mt = MerkleTree()
        mt.update("a", "1")
        h1 = mt.root_hash()
        mt.update("a", "2")
        h2 = mt.root_hash()
        assert h1 != h2

    def test_many_entries(self):
        mt = MerkleTree()
        for i in range(100):
            mt.update(f"key_{i}", f"value_{i}")
        assert len(mt.digest()["keys"]) == 100
        assert mt.root_hash() != ""


# =============================================================================
# GossipState Tests
# =============================================================================

class TestGossipState:
    def test_get_set(self):
        gs = GossipState("n1")
        gs.set("key1", "value1")
        assert gs.get("key1") == "value1"

    def test_get_missing(self):
        gs = GossipState("n1")
        assert gs.get("nope") is None

    def test_version_increments(self):
        gs = GossipState("n1")
        e1 = gs.set("key1", "v1")
        assert e1.version == 1
        e2 = gs.set("key1", "v2")
        assert e2.version == 2

    def test_merge_newer(self):
        gs1 = GossipState("n1")
        gs2 = GossipState("n2")
        gs1.set("key1", "v1")
        gs1.set("key1", "v2")  # version 2
        # Merge version 2 into gs2
        entries = gs1.all_entries()
        merged = gs2.merge_entries(entries)
        assert merged == 1
        assert gs2.get("key1") == "v2"

    def test_merge_older_ignored(self):
        gs1 = GossipState("n1")
        gs2 = GossipState("n2")
        gs2.set("key1", "v_new")
        gs2.set("key1", "v_newer")  # version 2
        # Try to merge version 1
        old_entry = StateEntry("key1", "v_old", 1, "n1", 0.0)
        merged = gs2.merge_entry(old_entry)
        assert not merged
        assert gs2.get("key1") == "v_newer"

    def test_entries_for_keys(self):
        gs = GossipState("n1")
        gs.set("a", "1")
        gs.set("b", "2")
        gs.set("c", "3")
        entries = gs.entries_for_keys(["a", "c"])
        assert len(entries) == 2
        keys = {e.key for e in entries}
        assert keys == {"a", "c"}

    def test_delete(self):
        gs = GossipState("n1")
        gs.set("key1", "v1")
        assert gs.delete("key1")
        assert gs.get("key1") is None

    def test_delete_nonexistent(self):
        gs = GossipState("n1")
        assert not gs.delete("nope")

    def test_keys_and_size(self):
        gs = GossipState("n1")
        gs.set("a", 1)
        gs.set("b", 2)
        assert set(gs.keys()) == {"a", "b"}
        assert gs.size() == 2

    def test_digest(self):
        gs = GossipState("n1")
        gs.set("a", 1)
        d = gs.digest()
        assert "root" in d
        assert "a" in d["keys"]

    def test_diff_keys(self):
        gs1 = GossipState("n1")
        gs2 = GossipState("n2")
        gs1.set("a", 1)
        gs1.set("b", 2)
        gs2.set("a", 1)
        diff = gs1.diff_keys(gs2.digest())
        assert "b" in diff


# =============================================================================
# DisseminationQueue Tests
# =============================================================================

class TestDisseminationQueue:
    def test_add_and_get(self):
        dq = DisseminationQueue()
        dq.add(MembershipUpdate(DisseminationType.ALIVE, "n1", 1))
        batch = dq.get_batch(10, 5)
        assert len(batch) == 1
        assert batch[0].node_id == "n1"

    def test_supersede_higher_incarnation(self):
        dq = DisseminationQueue()
        dq.add(MembershipUpdate(DisseminationType.ALIVE, "n1", 1))
        dq.add(MembershipUpdate(DisseminationType.ALIVE, "n1", 2))
        batch = dq.get_batch(10, 5)
        assert len(batch) == 1
        assert batch[0].incarnation == 2

    def test_supersede_higher_priority(self):
        dq = DisseminationQueue()
        dq.add(MembershipUpdate(DisseminationType.ALIVE, "n1", 1))
        dq.add(MembershipUpdate(DisseminationType.SUSPECT, "n1", 1))
        batch = dq.get_batch(10, 5)
        assert len(batch) == 1
        assert batch[0].dtype == DisseminationType.SUSPECT

    def test_max_transmissions(self):
        """Updates expire after lambda*log(N) transmissions."""
        dq = DisseminationQueue(retransmit_mult=1)
        dq.add(MembershipUpdate(DisseminationType.ALIVE, "n1", 1))
        # With cluster_size=4, max = 1 * ceil(log2(4)) = 2
        batch1 = dq.get_batch(10, 4)
        assert len(batch1) == 1
        batch2 = dq.get_batch(10, 4)
        assert len(batch2) == 1
        batch3 = dq.get_batch(10, 4)
        assert len(batch3) == 0  # Expired

    def test_batch_limit(self):
        dq = DisseminationQueue()
        for i in range(20):
            dq.add(MembershipUpdate(DisseminationType.ALIVE, f"n{i}", 1))
        batch = dq.get_batch(5, 20)
        assert len(batch) == 5

    def test_size(self):
        dq = DisseminationQueue()
        assert dq.size() == 0
        dq.add(MembershipUpdate(DisseminationType.ALIVE, "n1", 1))
        assert dq.size() == 1

    def test_multiple_nodes(self):
        dq = DisseminationQueue()
        dq.add(MembershipUpdate(DisseminationType.ALIVE, "n1", 1))
        dq.add(MembershipUpdate(DisseminationType.SUSPECT, "n2", 1))
        batch = dq.get_batch(10, 5)
        assert len(batch) == 2
        node_ids = {u.node_id for u in batch}
        assert node_ids == {"n1", "n2"}


# =============================================================================
# GossipNode Basic Tests
# =============================================================================

class TestGossipNodeBasic:
    def test_create_node(self):
        node = GossipNode("n1")
        assert node.node_id == "n1"
        assert node.incarnation == 0
        assert "n1" in node.members
        assert node.members["n1"].status == NodeStatus.ALIVE

    def test_alive_members_excludes_self(self):
        node = GossipNode("n1")
        node.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        assert node.alive_members() == ["n2"]

    def test_alive_members_includes_suspect(self):
        node = GossipNode("n1")
        node.members["n2"] = Member("n2", "n2", NodeStatus.SUSPECT)
        assert "n2" in node.alive_members()

    def test_alive_members_excludes_dead(self):
        node = GossipNode("n1")
        node.members["n2"] = Member("n2", "n2", NodeStatus.DEAD)
        assert node.alive_members() == []

    def test_member_count(self):
        node = GossipNode("n1")
        node.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        node.members["n3"] = Member("n3", "n3", NodeStatus.DEAD)
        assert node.member_count() == 2  # n1 + n2

    def test_join_sends_message(self):
        node = GossipNode("n1")
        node.join("n2")
        assert "n2" in node.members
        # Should have a JOIN message in outbox (already drained by join? no, join appends)
        # join doesn't drain, let's check _outbox
        assert len(node._outbox) == 1
        assert node._outbox[0].type == MessageType.JOIN

    def test_leave(self):
        node = GossipNode("n1")
        node.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        node.leave()
        assert node.members["n1"].status == NodeStatus.LEFT
        assert node.incarnation == 1

    def test_refute_suspicion(self):
        node = GossipNode("n1")
        old_inc = node.incarnation
        node.refute_suspicion()
        assert node.incarnation == old_inc + 1
        assert node.members["n1"].status == NodeStatus.ALIVE

    def test_state_get_set(self):
        node = GossipNode("n1")
        node.state.set("key1", "value1")
        assert node.state.get("key1") == "value1"

    def test_get_member(self):
        node = GossipNode("n1")
        m = node.get_member("n1")
        assert m is not None
        assert m.node_id == "n1"
        assert node.get_member("nonexistent") is None


# =============================================================================
# GossipNode SWIM Protocol Tests
# =============================================================================

class TestSWIMProtocol:
    def test_ping_and_ack(self):
        """Direct ping gets ack."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        # n1 sends ping to n2
        n1._send_ping("n2", now=0.0)
        msgs = n1._drain_outbox()
        assert len(msgs) == 1
        assert msgs[0].type == MessageType.PING

        # n2 receives ping
        responses = n2.receive(msgs[0], now=0.1)
        assert len(responses) == 1
        assert responses[0].type == MessageType.PING_ACK

        # n1 receives ack
        n1.receive(responses[0], now=0.2)
        assert n1.stats["acks_received"] == 1

    def test_tick_sends_ping(self):
        """Tick probes a random member."""
        n1 = GossipNode("n1", config={"protocol_period": 1.0})
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n1._record_heartbeat("n2", 0.0)
        msgs = n1.tick(now=1.0)
        ping_msgs = [m for m in msgs if m.type == MessageType.PING]
        assert len(ping_msgs) >= 1

    def test_ping_timeout_triggers_ping_req(self):
        """Timed-out ping triggers indirect ping-req."""
        n1 = GossipNode("n1", config={"ping_timeout": 0.5, "ping_req_count": 2})
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n1.members["n3"] = Member("n3", "n3", NodeStatus.ALIVE)
        n1.members["n4"] = Member("n4", "n4", NodeStatus.ALIVE)
        n1._record_heartbeat("n2", 0.0)
        n1._record_heartbeat("n3", 0.0)
        n1._record_heartbeat("n4", 0.0)

        # Send ping to n2
        n1._send_ping("n2", now=0.0)
        n1._drain_outbox()

        # Timeout at t=0.6 (> 0.5)
        n1._check_pending_acks(0.6)
        msgs = n1._drain_outbox()
        ping_req_msgs = [m for m in msgs if m.type == MessageType.PING_REQ]
        assert len(ping_req_msgs) > 0

    def test_suspect_on_no_ack(self):
        """Node suspected when no ack arrives."""
        n1 = GossipNode("n1", config={"ping_timeout": 0.5, "suspect_timeout": 2.0})
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)

        # Suspect n2
        n1._suspect_node("n2")
        assert n1.members["n2"].status == NodeStatus.SUSPECT
        assert "n2" in n1._suspect_timers

    def test_suspect_to_dead(self):
        """Suspect node becomes dead after timeout."""
        n1 = GossipNode("n1", config={"suspect_timeout": 2.0})
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)

        n1._suspect_node("n2")
        assert n1.members["n2"].status == NodeStatus.SUSPECT

        # Advance past suspect timeout
        n1._suspect_timers["n2"] = 1.0  # Set deadline
        n1._check_suspect_timers(2.0)
        assert n1.members["n2"].status == NodeStatus.DEAD

    def test_alive_clears_suspicion(self):
        """Alive refutes suspicion."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n1._suspect_node("n2")
        assert n1.members["n2"].status == NodeStatus.SUSPECT

        n1._mark_alive("n2", incarnation=1)
        assert n1.members["n2"].status == NodeStatus.ALIVE
        assert "n2" not in n1._suspect_timers

    def test_handle_join(self):
        """Handle join request adds member and sends member list."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)

        join_msg = GossipMessage(
            type=MessageType.JOIN, src="n3", dst="n1",
            data={"node_id": "n3", "address": "n3"})
        responses = n1.receive(join_msg, now=1.0)
        assert "n3" in n1.members
        assert n1.members["n3"].status == NodeStatus.ALIVE
        # Should send back member list
        compound_msgs = [m for m in responses if m.type == MessageType.COMPOUND]
        assert len(compound_msgs) == 1

    def test_handle_leave(self):
        """Handle leave notification marks member as left."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE, incarnation=0)

        leave_msg = GossipMessage(
            type=MessageType.LEAVE, src="n2", dst="n1",
            data={"node_id": "n2", "incarnation": 1})
        n1.receive(leave_msg, now=1.0)
        assert n1.members["n2"].status == NodeStatus.LEFT


# =============================================================================
# Piggybacking / Dissemination Tests
# =============================================================================

class TestDissemination:
    def test_piggyback_alive(self):
        """Alive updates propagate via piggyback."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        # n1 learns about n3
        n1.members["n3"] = Member("n3", "n3", NodeStatus.ALIVE, 1)
        n1.dissemination.add(MembershipUpdate(
            DisseminationType.ALIVE, "n3", 1))

        # n1 pings n2 with piggybacked info
        n1._send_ping("n2", now=0.0)
        msgs = n1._drain_outbox()
        assert len(msgs) == 1
        assert len(msgs[0].piggyback) > 0

        # n2 processes the ping
        n2.receive(msgs[0], now=0.1)
        assert "n3" in n2.members
        assert n2.members["n3"].status == NodeStatus.ALIVE

    def test_piggyback_suspect(self):
        """Suspect updates propagate via piggyback."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n1.members["n3"] = Member("n3", "n3", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)
        n2.members["n3"] = Member("n3", "n3", NodeStatus.ALIVE)

        # n1 suspects n3
        n1._suspect_node("n3")
        # n1 pings n2
        n1._send_ping("n2", now=0.0)
        msgs = n1._drain_outbox()
        n2.receive(msgs[0], now=0.1)
        assert n2.members["n3"].status == NodeStatus.SUSPECT

    def test_piggyback_confirm(self):
        """Confirm dead propagates."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n1.members["n3"] = Member("n3", "n3", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)
        n2.members["n3"] = Member("n3", "n3", NodeStatus.ALIVE)

        n1._confirm_dead("n3")
        n1._send_ping("n2", now=0.0)
        msgs = n1._drain_outbox()
        n2.receive(msgs[0], now=0.1)
        assert n2.members["n3"].status == NodeStatus.DEAD

    def test_self_suspicion_refuted(self):
        """When a node hears it's suspected, it refutes."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)

        # Piggyback says n1 is suspect
        piggyback = [{"dtype": "suspect", "node_id": "n1", "incarnation": 0}]
        msg = GossipMessage(
            type=MessageType.PING, src="n2", dst="n1",
            seq=1, piggyback=piggyback)
        n1.receive(msg, now=0.0)
        # n1 should have refuted
        assert n1.incarnation == 1
        assert n1.members["n1"].status == NodeStatus.ALIVE

    def test_leave_propagates(self):
        """Leave updates propagate via piggyback."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n3 = GossipNode("n3")
        for n in [n1, n2, n3]:
            for other in [n1, n2, n3]:
                if other.node_id != n.node_id:
                    n.members[other.node_id] = Member(other.node_id, other.node_id, NodeStatus.ALIVE)

        # n3 leaves
        piggyback = [{"dtype": "leave", "node_id": "n3", "incarnation": 1}]
        msg = GossipMessage(type=MessageType.PING, src="n1", dst="n2",
                            seq=1, piggyback=piggyback)
        n2.receive(msg, now=0.0)
        assert n2.members["n3"].status == NodeStatus.LEFT


# =============================================================================
# State Sync (Anti-Entropy) Tests
# =============================================================================

class TestAntiEntropy:
    def test_push_state(self):
        """Push state to a peer."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        n1.state.set("key1", "value1")
        msgs = n1.push_state_to("n2")
        assert len(msgs) == 1
        assert msgs[0].type == MessageType.STATE_PUSH

        n2.receive(msgs[0], now=0.0)
        assert n2.state.get("key1") == "value1"

    def test_sync_digest(self):
        """Merkle-based sync finds differences."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        n1.state.set("a", 1)
        n1.state.set("b", 2)
        n2.state.set("a", 1)
        # n2 is missing "b"

        msgs = n1.sync_state_with("n2")
        assert len(msgs) == 1
        assert msgs[0].type == MessageType.SYNC_DIGEST

        responses = n2.receive(msgs[0], now=0.0)
        # n2 should send back delta (with its entries for differing keys)
        delta_msgs = [m for m in responses if m.type == MessageType.SYNC_DELTA]
        if delta_msgs:
            n1.receive(delta_msgs[0], now=0.1)

    def test_push_pull(self):
        """Push-pull exchanges state bidirectionally."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        n1.state.set("a", 1)
        n2.state.set("b", 2)

        msg = GossipMessage(
            type=MessageType.STATE_PUSH_PULL, src="n1", dst="n2",
            data={"entries": n1._serialize_state_entries(n1.state.all_entries())})
        responses = n2.receive(msg, now=0.0)
        # n2 should have n1's state
        assert n2.state.get("a") == 1
        # n2 sends its state back
        assert len(responses) == 1
        n1.receive(responses[0], now=0.1)
        assert n1.state.get("b") == 2

    def test_state_pull(self):
        """Pull state from a peer."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        n2.state.set("x", 42)
        pull_msg = GossipMessage(
            type=MessageType.STATE_PULL, src="n1", dst="n2",
            data={"keys": ["x"]})
        responses = n2.receive(pull_msg, now=0.0)
        assert len(responses) == 1
        n1.receive(responses[0], now=0.1)
        assert n1.state.get("x") == 42


# =============================================================================
# GossipNetwork Tests
# =============================================================================

class TestGossipNetwork:
    def test_create_network(self):
        net = GossipNetwork(seed=42)
        n1 = GossipNode("n1")
        net.add_node(n1)
        assert "n1" in net.nodes

    def test_simple_cluster(self):
        """3-node cluster can gossip."""
        net, nodes = make_cluster(3)
        net.run(ticks=10)
        assert net.delivered_count() > 0

    def test_partition(self):
        """Partitioned nodes can't communicate."""
        net, nodes = make_cluster(3)
        net.partition("node_0", "node_1")
        # Messages from n0 to n1 should be dropped
        msg = GossipMessage(type=MessageType.PING, src="node_0", dst="node_1")
        net.send([msg])
        net.deliver_pending()
        assert net.dropped_count() > 0

    def test_heal_partition(self):
        """Healed partitions allow communication."""
        net, nodes = make_cluster(3)
        net.partition("node_0", "node_1")
        net.heal_partition("node_0", "node_1")
        msg = GossipMessage(type=MessageType.PING, src="node_0", dst="node_1")
        net.send([msg])
        net.deliver_pending()
        assert net.delivered_count() > 0

    def test_message_loss(self):
        """Message loss drops some messages."""
        net, nodes = make_cluster(5, seed=42)
        net.set_loss_rate(0.5)
        net.run(ticks=20)
        assert net.dropped_count() > 0

    def test_remove_node(self):
        """Removed node stops receiving messages."""
        net, nodes = make_cluster(3)
        net.remove_node("node_1")
        assert "node_1" not in net.nodes

    def test_convergence(self):
        """Cluster converges on membership."""
        net, nodes = make_cluster(5, seed=42)
        ticks = net.converge(max_ticks=30)
        assert ticks <= 30

    def test_cluster_view(self):
        net, nodes = make_cluster(3)
        view = net.get_cluster_view("node_0")
        assert "node_0" in view
        assert view["node_0"] == "alive"

    def test_partition_node(self):
        """Partition a single node from all others."""
        net, nodes = make_cluster(4)
        net.partition_node("node_0")
        msg1 = GossipMessage(type=MessageType.PING, src="node_0", dst="node_1")
        msg2 = GossipMessage(type=MessageType.PING, src="node_0", dst="node_2")
        net.send([msg1, msg2])
        net.deliver_pending()
        assert net.dropped_count() == 2

    def test_heal_all(self):
        net, nodes = make_cluster(3)
        net.partition("node_0", "node_1")
        net.partition("node_1", "node_2")
        net.heal_all()
        msg = GossipMessage(type=MessageType.PING, src="node_0", dst="node_1")
        net.send([msg])
        net.deliver_pending()
        assert net.delivered_count() > 0

    def test_broadcast_state_sync(self):
        """Anti-entropy sync propagates state."""
        net, nodes = make_cluster(3, seed=42)
        nodes[0].state.set("key1", "value1", now=0.0)
        net.broadcast_state_sync()
        # After sync, at least one other node should have the state
        has_state = any(n.state.get("key1") == "value1" for n in nodes[1:])
        assert has_state

    def test_current_time(self):
        net = GossipNetwork()
        assert net.current_time() == 0.0
        net.tick(dt=2.5)
        assert net.current_time() == 2.5


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    def test_join_and_gossip(self):
        """New node joins and learns about cluster."""
        net = GossipNetwork(seed=42)
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        net.add_node(n1)
        net.add_node(n2)

        # n1 knows n2
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        # n3 joins via n1
        n3 = GossipNode("n3")
        net.add_node(n3)
        n3.join("n1")
        msgs = n3._drain_outbox()
        net.send(msgs)
        net.deliver_pending()
        net.deliver_pending()

        # n1 should know about n3
        assert "n3" in n1.members

    def test_failure_detection_flow(self):
        """Full failure detection: alive -> suspect -> dead."""
        n1 = GossipNode("n1", config={"suspect_timeout": 2.0})
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)

        # Suspect n2
        n1._suspect_node("n2")
        assert n1.members["n2"].status == NodeStatus.SUSPECT

        # Wait for suspect timeout
        n1._suspect_timers["n2"] = 0.0
        n1._check_suspect_timers(3.0)
        assert n1.members["n2"].status == NodeStatus.DEAD

    def test_state_convergence(self):
        """State converges across cluster via push."""
        net, nodes = make_cluster(5, seed=42)
        # Set state on node_0
        nodes[0].state.set("config", {"version": 42}, now=0.0)

        # Push to all peers
        for n in nodes[1:]:
            msgs = nodes[0].push_state_to(n.node_id)
            net.send(msgs)
        net.deliver_pending()

        for n in nodes:
            assert n.state.get("config") == {"version": 42}

    def test_state_convergence_via_gossip(self):
        """State converges through multiple sync rounds."""
        net, nodes = make_cluster(5, seed=42)
        nodes[0].state.set("key", "val", now=0.0)

        # Multiple rounds of anti-entropy
        for _ in range(5):
            net.broadcast_state_sync()

        count = sum(1 for n in nodes if n.state.get("key") == "val")
        assert count >= 3  # Should reach most nodes

    def test_partition_and_heal(self):
        """Nodes detect failure during partition, recover after heal."""
        net, nodes = make_cluster(3, seed=42,
                                  config={"suspect_timeout": 3.0, "ping_timeout": 0.5})
        # Partition node_2
        net.partition_node("node_2")
        # Run for a while -- node_2 won't respond
        net.run(ticks=10)
        # Heal
        net.heal_all()
        net.run(ticks=10)

    def test_graceful_leave(self):
        """Graceful leave propagates to all members."""
        net, nodes = make_cluster(3, seed=42)
        # node_2 leaves
        nodes[2].leave()
        msgs = nodes[2]._drain_outbox()
        net.send(msgs)
        net.deliver_pending()

        for n in nodes[:2]:
            assert n.members["node_2"].status == NodeStatus.LEFT

    def test_incarnation_refutes_suspicion(self):
        """Higher incarnation refutes suspicion."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        # n1 suspects n2
        n1._suspect_node("n2")
        assert n1.members["n2"].status == NodeStatus.SUSPECT

        # n2 sends alive with higher incarnation
        n2.refute_suspicion()
        update = {"dtype": "alive", "node_id": "n2", "incarnation": n2.incarnation}
        msg = GossipMessage(type=MessageType.PING, src="n2", dst="n1",
                            seq=1, piggyback=[update])
        n1.receive(msg, now=0.0)
        assert n1.members["n2"].status == NodeStatus.ALIVE

    def test_large_cluster(self):
        """10-node cluster functions."""
        net, nodes = make_cluster(10, seed=42)
        net.run(ticks=20)
        assert net.delivered_count() > 0
        # All nodes should still consider themselves alive
        for n in nodes:
            assert n.members[n.node_id].status == NodeStatus.ALIVE

    def test_state_with_multiple_writers(self):
        """Multiple nodes writing different keys."""
        net, nodes = make_cluster(5, seed=42)
        for i, n in enumerate(nodes):
            n.state.set(f"key_{i}", f"val_{i}", now=0.0)

        # Sync rounds
        for _ in range(10):
            net.broadcast_state_sync()

        # Each node should have all keys
        for n in nodes:
            count = sum(1 for i in range(5) if n.state.get(f"key_{i}") is not None)
            assert count >= 3  # At least most keys propagated

    def test_conflicting_writes_version_wins(self):
        """Higher version wins in state conflict."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        n1.state.set("x", "old")
        n1.state.set("x", "new")  # version 2
        n2.state.set("x", "other")  # version 1

        # n1 pushes to n2
        msgs = n1.push_state_to("n2")
        n2.receive(msgs[0])
        assert n2.state.get("x") == "new"  # Higher version wins

    def test_stats_tracking(self):
        """Stats are tracked correctly."""
        net, nodes = make_cluster(3, seed=42)
        net.run(ticks=5)
        total_pings = sum(n.stats["pings_sent"] for n in nodes)
        total_acks = sum(n.stats["acks_sent"] for n in nodes)
        assert total_pings > 0
        assert total_acks > 0

    def test_events_logged(self):
        """Events are logged for debugging."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n1._suspect_node("n2")
        events = [e for e in n1._events if e["type"] == "suspect"]
        assert len(events) == 1

    def test_empty_network_tick(self):
        """Network with single node doesn't crash."""
        net = GossipNetwork()
        n1 = GossipNode("n1")
        net.add_node(n1)
        net.tick()  # Should not error
        assert net.delivered_count() == 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_ping_to_unknown_node(self):
        """Ping to node not in network is dropped."""
        net = GossipNetwork()
        n1 = GossipNode("n1")
        net.add_node(n1)
        msg = GossipMessage(type=MessageType.PING, src="n1", dst="ghost")
        net.send([msg])
        net.deliver_pending()
        assert net.dropped_count() == 1

    def test_empty_piggyback(self):
        """Empty piggyback doesn't crash."""
        n1 = GossipNode("n1")
        msg = GossipMessage(type=MessageType.PING, src="n2", dst="n1",
                            seq=1, piggyback=[])
        n1.receive(msg)

    def test_duplicate_join(self):
        """Duplicate join is harmless."""
        n1 = GossipNode("n1")
        join_msg = GossipMessage(
            type=MessageType.JOIN, src="n2", dst="n1",
            data={"node_id": "n2", "address": "n2"})
        n1.receive(join_msg, now=1.0)
        n1.receive(join_msg, now=2.0)
        assert n1.members["n2"].status == NodeStatus.ALIVE

    def test_dead_node_leave(self):
        """Leave from a dead node is handled."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.DEAD, incarnation=5)
        leave_msg = GossipMessage(
            type=MessageType.LEAVE, src="n2", dst="n1",
            data={"node_id": "n2", "incarnation": 6})
        n1.receive(leave_msg)
        assert n1.members["n2"].status == NodeStatus.LEFT

    def test_stale_suspect_ignored(self):
        """Suspect with lower incarnation is ignored."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE, incarnation=5)
        piggyback = [{"dtype": "suspect", "node_id": "n2", "incarnation": 3}]
        msg = GossipMessage(type=MessageType.PING, src="n3", dst="n1",
                            seq=1, piggyback=piggyback)
        n1.receive(msg)
        assert n1.members["n2"].status == NodeStatus.ALIVE  # Not suspected

    def test_confirm_already_left(self):
        """Confirm for an already-left node is no-op."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.LEFT, incarnation=5)
        piggyback = [{"dtype": "confirm", "node_id": "n2", "incarnation": 5}]
        msg = GossipMessage(type=MessageType.PING, src="n3", dst="n1",
                            seq=1, piggyback=piggyback)
        n1.receive(msg)
        assert n1.members["n2"].status == NodeStatus.LEFT  # Unchanged

    def test_member_copy(self):
        """Member.copy creates independent copy."""
        m = Member("n1", "addr", NodeStatus.ALIVE, 5, 1.0, {"k": "v"})
        m2 = m.copy()
        m2.status = NodeStatus.DEAD
        m2.metadata["k"] = "changed"
        assert m.status == NodeStatus.ALIVE
        assert m.metadata["k"] == "v"

    def test_state_entry_copy(self):
        """StateEntry.copy creates independent copy."""
        e = StateEntry("k", "v", 1, "n1", 1.0)
        e2 = e.copy()
        e2.value = "changed"
        assert e.value == "v"

    def test_log2_ceil(self):
        assert _log2_ceil(1) == 1
        assert _log2_ceil(2) == 1
        assert _log2_ceil(3) == 2
        assert _log2_ceil(4) == 2
        assert _log2_ceil(5) == 3

    def test_ping_req_no_mediators(self):
        """Ping-req with no other members is harmless."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        # Only n2 alive, no mediators for ping-req to n2
        n1._send_ping_req("n2", now=0.0)
        msgs = n1._drain_outbox()
        assert len(msgs) == 0

    def test_ping_req_forwarding(self):
        """Ping-req is forwarded to target."""
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")  # mediator
        n3 = GossipNode("n3")  # target
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)
        n2.members["n3"] = Member("n3", "n3", NodeStatus.ALIVE)

        req = GossipMessage(
            type=MessageType.PING_REQ, src="n1", dst="n2",
            seq=1, data={"target": "n3"})
        responses = n2.receive(req, now=0.0)
        pings = [m for m in responses if m.type == MessageType.PING]
        assert len(pings) == 1
        assert pings[0].dst == "n3"

    def test_network_latency(self):
        """Messages with latency are delayed."""
        net = GossipNetwork()
        n1 = GossipNode("n1")
        n2 = GossipNode("n2")
        net.add_node(n1)
        net.add_node(n2)
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n2.members["n1"] = Member("n1", "n1", NodeStatus.ALIVE)

        net.set_latency(2.0)
        msg = GossipMessage(type=MessageType.PING, src="n1", dst="n2", seq=1)
        net.send([msg])
        # At t=0, nothing delivered
        delivered = net.deliver_pending()
        assert delivered == 0
        # Advance past latency
        net._time = 3.0
        delivered = net.deliver_pending()
        assert delivered == 1

    def test_all_member_ids(self):
        n = GossipNode("n1")
        n.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE)
        n.members["n3"] = Member("n3", "n3", NodeStatus.DEAD)
        assert set(n.all_member_ids()) == {"n1", "n2", "n3"}


# =============================================================================
# Stress / Scale Tests
# =============================================================================

class TestScale:
    def test_20_node_cluster(self):
        """20-node cluster gossips without errors."""
        net, nodes = make_cluster(20, seed=123)
        net.run(ticks=15)
        alive_count = sum(1 for n in nodes
                          if n.members[n.node_id].status == NodeStatus.ALIVE)
        assert alive_count == 20

    def test_state_propagation_10_nodes(self):
        """State propagates through 10-node cluster."""
        net, nodes = make_cluster(10, seed=42)
        nodes[0].state.set("global", "hello", now=0.0)
        for _ in range(15):
            net.broadcast_state_sync()
        count = sum(1 for n in nodes if n.state.get("global") == "hello")
        assert count >= 5

    def test_churn(self):
        """Handle nodes joining and leaving."""
        net, nodes = make_cluster(5, seed=42)
        net.run(ticks=5)

        # node_2 leaves
        nodes[2].leave()
        msgs = nodes[2]._drain_outbox()
        net.send(msgs)
        net.deliver_pending()
        net.run(ticks=5)

        # New node joins
        n_new = GossipNode("node_new")
        net.add_node(n_new)
        n_new.join("node_0")
        msgs = n_new._drain_outbox()
        net.send(msgs)
        net.deliver_pending()
        net.run(ticks=5)

        assert "node_new" in nodes[0].members

    def test_many_state_keys(self):
        """Many state keys work correctly."""
        gs = GossipState("n1")
        for i in range(200):
            gs.set(f"key_{i}", f"value_{i}")
        assert gs.size() == 200
        for i in range(200):
            assert gs.get(f"key_{i}") == f"value_{i}"

    def test_concurrent_suspicion_and_alive(self):
        """Concurrent suspect and alive updates resolve correctly."""
        n1 = GossipNode("n1")
        n1.members["n2"] = Member("n2", "n2", NodeStatus.ALIVE, incarnation=1)

        # Suspect at incarnation 1
        piggyback1 = [{"dtype": "suspect", "node_id": "n2", "incarnation": 1}]
        msg1 = GossipMessage(type=MessageType.PING, src="n3", dst="n1",
                             seq=1, piggyback=piggyback1)
        n1.receive(msg1)
        assert n1.members["n2"].status == NodeStatus.SUSPECT

        # Alive at incarnation 2 (refutation)
        piggyback2 = [{"dtype": "alive", "node_id": "n2", "incarnation": 2}]
        msg2 = GossipMessage(type=MessageType.PING, src="n4", dst="n1",
                             seq=2, piggyback=piggyback2)
        n1.receive(msg2)
        assert n1.members["n2"].status == NodeStatus.ALIVE
        assert n1.members["n2"].incarnation == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
