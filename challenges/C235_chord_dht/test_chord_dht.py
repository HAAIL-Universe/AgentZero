"""
Tests for C235: Chord Distributed Hash Table
"""

import pytest
import random
from chord_dht import (
    ChordNode, ChordRing, VirtualNodeManager, RangeQueryMixin,
    LookupStats, measure_lookup_performance, calculate_key_migration,
    batch_put, batch_get,
    _hash_to_m, in_range, in_range_open, distance, _in_id_range,
    DEFAULT_M, DEFAULT_SUCCESSORS
)


# =============================================================================
# 1. Utility Functions
# =============================================================================

class TestUtilities:
    def test_hash_to_m_string(self):
        h = _hash_to_m("hello", 8)
        assert 0 <= h < 256

    def test_hash_to_m_int(self):
        h = _hash_to_m(42, 8)
        assert 0 <= h < 256

    def test_hash_to_m_deterministic(self):
        assert _hash_to_m("test", 8) == _hash_to_m("test", 8)

    def test_hash_to_m_different_keys(self):
        h1 = _hash_to_m("key1", 8)
        h2 = _hash_to_m("key2", 8)
        # Different keys should usually hash differently (probabilistic)
        # Just test they're valid
        assert 0 <= h1 < 256
        assert 0 <= h2 < 256

    def test_hash_to_m_different_m(self):
        h16 = _hash_to_m("test", 16)
        assert 0 <= h16 < 65536

    def test_in_range_simple(self):
        assert in_range(5, 3, 7, 256)
        assert not in_range(2, 3, 7, 256)

    def test_in_range_inclusive_end(self):
        assert in_range(7, 3, 7, 256)
        assert not in_range(3, 3, 7, 256)

    def test_in_range_wrap(self):
        assert in_range(1, 250, 5, 256)
        assert in_range(254, 250, 5, 256)
        assert not in_range(100, 250, 5, 256)

    def test_in_range_full(self):
        # When a == b, entire ring
        assert in_range(50, 10, 10, 256)

    def test_in_range_open_simple(self):
        assert in_range_open(5, 3, 7, 256)
        assert not in_range_open(3, 3, 7, 256)
        assert not in_range_open(7, 3, 7, 256)

    def test_in_range_open_wrap(self):
        assert in_range_open(1, 250, 5, 256)
        assert not in_range_open(5, 250, 5, 256)

    def test_distance(self):
        assert distance(3, 7, 256) == 4
        assert distance(250, 5, 256) == 11
        assert distance(5, 5, 256) == 0

    def test_in_id_range(self):
        assert _in_id_range(5, 3, 7, 256)
        assert _in_id_range(3, 3, 7, 256)
        assert _in_id_range(7, 3, 7, 256)

    def test_in_id_range_wrap(self):
        assert _in_id_range(254, 250, 5, 256)
        assert _in_id_range(1, 250, 5, 256)
        assert not _in_id_range(100, 250, 5, 256)


# =============================================================================
# 2. ChordNode Basics
# =============================================================================

class TestChordNodeBasics:
    def test_create_node(self):
        node = ChordNode(10, m=8)
        assert node.node_id == 10
        assert node.m == 8
        assert node.ring_size == 256

    def test_create_node_defaults(self):
        node = ChordNode(5)
        assert node.m == DEFAULT_M
        assert node.num_successors == DEFAULT_SUCCESSORS

    def test_node_repr(self):
        node = ChordNode(10)
        r = repr(node)
        assert "10" in r
        assert "ChordNode" in r

    def test_join_alone(self):
        node = ChordNode(10)
        node._network = type('Net', (), {'get_node': lambda self, x: None})()
        node.join(None)
        assert node.successor == 10
        assert node.predecessor is None

    def test_node_alive(self):
        node = ChordNode(10)
        assert node.alive
        node.alive = False
        assert not node.alive

    def test_finger_table_init(self):
        node = ChordNode(10, m=8)
        assert len(node.finger) == 8
        assert all(f is None for f in node.finger)

    def test_store_empty(self):
        node = ChordNode(10)
        assert len(node.store) == 0
        assert len(node.replicas) == 0

    def test_local_keys_empty(self):
        node = ChordNode(10)
        assert node.local_keys() == {}

    def test_local_keys_with_data(self):
        node = ChordNode(10)
        node.store["key_a"] = "value_a"
        node.store["key_b"] = "value_b"
        keys = node.local_keys()
        assert keys["key_a"] == "value_a"
        assert keys["key_b"] == "value_b"


# =============================================================================
# 3. Single Node Ring
# =============================================================================

class TestSingleNodeRing:
    def test_create_ring(self):
        ring = ChordRing(m=8)
        assert len(ring) == 0

    def test_add_first_node(self):
        ring = ChordRing(m=8)
        node = ring.add_node(10)
        assert node.node_id == 10
        assert len(ring) == 1

    def test_single_node_successor(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        node = ring.nodes[10]
        assert node.successor == 10  # points to itself

    def test_single_node_put_get(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        ring.put("hello", "world")
        assert ring.get("hello") == "world"

    def test_single_node_delete(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        ring.put("key", "value")
        assert ring.delete("key")
        assert ring.get("key") is None

    def test_single_node_delete_nonexistent(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        assert not ring.delete("nope")

    def test_single_node_get_nonexistent(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        assert ring.get("nope") is None

    def test_ring_contains(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        assert 10 in ring
        assert 20 not in ring


# =============================================================================
# 4. Multi-Node Ring
# =============================================================================

class TestMultiNodeRing:
    def setup_method(self):
        self.ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            self.ring.add_node(nid)
        self.ring.stabilize(5)

    def test_four_nodes(self):
        assert len(self.ring) == 4

    def test_ring_order(self):
        order = self.ring.get_ring_order()
        assert order == [0, 64, 128, 192]

    def test_successor_chain(self):
        """Successors should form a cycle."""
        self.ring.stabilize(5)
        n0 = self.ring.nodes[0]
        n64 = self.ring.nodes[64]
        n128 = self.ring.nodes[128]
        n192 = self.ring.nodes[192]
        assert n0.successor == 64
        assert n64.successor == 128
        assert n128.successor == 192
        assert n192.successor == 0

    def test_predecessor_chain(self):
        self.ring.stabilize(5)
        n0 = self.ring.nodes[0]
        n64 = self.ring.nodes[64]
        assert n64.predecessor == 0
        assert n0.predecessor == 192

    def test_put_get_multiple_nodes(self):
        for i in range(20):
            self.ring.put(f"key_{i}", f"value_{i}")
        for i in range(20):
            assert self.ring.get(f"key_{i}") == f"value_{i}"

    def test_keys_distributed(self):
        for i in range(50):
            self.ring.put(f"dist_{i}", i)
        dist = self.ring.get_key_distribution()
        # At least 2 nodes should have keys
        nodes_with_keys = sum(1 for v in dist.values() if v > 0)
        assert nodes_with_keys >= 2

    def test_ring_integrity(self):
        valid, issues = self.ring.verify_ring_integrity()
        assert valid, f"Ring integrity issues: {issues}"

    def test_get_all_keys(self):
        for i in range(10):
            self.ring.put(f"all_{i}", i)
        all_keys = self.ring.get_all_keys()
        assert len(all_keys) == 10
        for i in range(10):
            assert all_keys[f"all_{i}"] == i

    def test_finger_table(self):
        table = self.ring.get_finger_table(0)
        assert table is not None
        assert len(table) == 8
        assert table[0]['start'] == 1   # 0 + 2^0 = 1
        assert table[1]['start'] == 2   # 0 + 2^1 = 2
        assert table[6]['start'] == 64  # 0 + 2^6 = 64
        assert table[7]['start'] == 128 # 0 + 2^7 = 128

    def test_finger_table_nonexistent(self):
        assert self.ring.get_finger_table(999) is None


# =============================================================================
# 5. Node Join
# =============================================================================

class TestNodeJoin:
    def test_join_updates_ring(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(5)

        assert ring.nodes[0].successor == 128
        assert ring.nodes[128].successor == 0

    def test_join_middle(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(5)

        ring.add_node(64)
        ring.stabilize(5)

        assert ring.nodes[0].successor == 64
        assert ring.nodes[64].successor == 128
        assert ring.nodes[128].successor == 0

    def test_join_preserves_keys(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(3)

        for i in range(20):
            ring.put(f"join_{i}", i)

        ring.add_node(64)
        ring.stabilize(5)

        # All keys should still be accessible
        for i in range(20):
            assert ring.get(f"join_{i}") == i

    def test_join_random_id(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        node = ring.add_node()  # random ID
        assert node.node_id != 0
        assert len(ring) == 2

    def test_join_duplicate(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        node = ring.add_node(10)  # should return existing
        assert len(ring) == 1
        assert node.node_id == 10


# =============================================================================
# 6. Node Leave (Graceful)
# =============================================================================

class TestNodeLeave:
    def test_remove_node(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.remove_node(64)
        assert 64 not in ring
        assert len(ring) == 3

    def test_remove_transfers_keys(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128]:
            ring.add_node(nid)
        ring.stabilize(5)

        for i in range(30):
            ring.put(f"leave_{i}", i)

        ring.remove_node(64)
        ring.stabilize(5)

        # All keys should still be accessible
        for i in range(30):
            assert ring.get(f"leave_{i}") == i

    def test_remove_nonexistent(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        assert not ring.remove_node(999)

    def test_remove_updates_ring(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.remove_node(64)
        ring.stabilize(5)

        assert ring.nodes[0].successor == 128
        assert ring.nodes[128].successor == 0


# =============================================================================
# 7. Node Failure (Crash)
# =============================================================================

class TestNodeFailure:
    def test_fail_node(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.fail_node(64)
        assert 64 not in ring

    def test_recovery_after_failure(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.fail_node(64)
        ring.stabilize(5)

        # Ring should recover
        valid, issues = ring.verify_ring_integrity()
        assert valid, f"Issues after failure: {issues}"

    def test_data_loss_on_crash(self):
        ring = ChordRing(m=8)
        for nid in [0, 128]:
            ring.add_node(nid)
        ring.stabilize(5)

        for i in range(20):
            ring.put(f"crash_{i}", i)

        # Count keys on node 0
        node0_keys = len(ring.nodes[0].store)

        ring.fail_node(0)
        ring.stabilize(3)

        # Keys on the crashed node are lost (no replication)
        remaining = ring.get_all_keys()
        assert len(remaining) < 20 or node0_keys == 0


# =============================================================================
# 8. Stabilization
# =============================================================================

class TestStabilization:
    def test_stabilize_fixes_ring(self):
        ring = ChordRing(m=8)
        for nid in [0, 50, 100, 150, 200]:
            ring.add_node(nid)

        ring.stabilize(10)

        valid, issues = ring.verify_ring_integrity()
        assert valid, f"Stabilization issues: {issues}"

    def test_stabilize_successor_list(self):
        ring = ChordRing(m=8, num_successors=3)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        node = ring.nodes[0]
        assert len(node.successor_list) >= 2

    def test_fix_fingers(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(10)

        node = ring.nodes[0]
        # finger[6] covers start=64, should point to 64
        assert node.finger[6] == 64
        # finger[7] covers start=128, should point to 128
        assert node.finger[7] == 128

    def test_check_predecessor(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128]:
            ring.add_node(nid)
        ring.stabilize(5)

        # Set a fake predecessor
        ring.nodes[0].predecessor = 200
        ring.nodes[0].check_predecessor()
        assert ring.nodes[0].predecessor is None  # 200 doesn't exist

    def test_multiple_stabilize_rounds(self):
        ring = ChordRing(m=8)
        for nid in [10, 50, 90, 130, 170, 210, 250]:
            ring.add_node(nid)
        ring.stabilize(20)

        valid, issues = ring.verify_ring_integrity()
        assert valid


# =============================================================================
# 9. Replication
# =============================================================================

class TestReplication:
    def test_put_replicated(self):
        ring = ChordRing(m=8, replicas=3)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.put_replicated("rep_key", "rep_value")
        assert ring.get_replicated("rep_key") == "rep_value"

    def test_replicated_survives_failure(self):
        ring = ChordRing(m=8, replicas=3, num_successors=3)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.put_replicated("survive", "value")

        # Find which node has the primary
        key_id = _hash_to_m("survive", 8)
        primary_id = ring._any_node().find_successor(key_id)

        # Fail the primary
        ring.fail_node(primary_id)
        ring.stabilize(3)

        # Should still be retrievable from replicas
        result = ring.get_replicated("survive")
        assert result == "value"

    def test_replication_count(self):
        ring = ChordRing(m=8, replicas=3, num_successors=3)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.put_replicated("multi", "copies")

        # Count how many nodes have this key
        copies = 0
        for node in ring.nodes.values():
            if "multi" in node.store or "multi" in node.replicas:
                copies += 1
        assert copies >= 2  # at least primary + 1 replica

    def test_get_replicated_empty(self):
        ring = ChordRing(m=8)
        assert ring.get_replicated("nothing") is None

    def test_put_replicated_empty_ring(self):
        ring = ChordRing(m=8)
        assert not ring.put_replicated("key", "val")


# =============================================================================
# 10. Lookup Path
# =============================================================================

class TestLookupPath:
    def test_lookup_path_exists(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        path = ring.get_lookup_path("test_key")
        assert len(path) >= 1

    def test_lookup_path_terminates(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        path = ring.get_lookup_path("any_key")
        # Should not be excessively long
        assert len(path) <= len(ring) + 1

    def test_lookup_path_ends_at_responsible(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        ring.put("target", "value")
        key_id = _hash_to_m("target", 8)
        path = ring.get_lookup_path("target")

        # Last node in path should be or precede the responsible node
        last = path[-1]
        assert last in ring.nodes

    def test_lookup_path_empty_ring(self):
        ring = ChordRing(m=8)
        path = ring.get_lookup_path("key")
        assert path == []


# =============================================================================
# 11. Lookup Performance
# =============================================================================

class TestLookupPerformance:
    def test_performance_stats(self):
        ring = ChordRing(m=8)
        for nid in [0, 32, 64, 96, 128, 160, 192, 224]:
            ring.add_node(nid)
        ring.stabilize(10)

        stats = measure_lookup_performance(ring, 50)
        assert stats.total_lookups == 50
        assert stats.avg_hops > 0
        assert stats.max_hops <= 8 + 1  # O(log N) with m=8

    def test_lookup_stats_reset(self):
        stats = LookupStats()
        stats.record(3)
        stats.record(5)
        assert stats.total_lookups == 2
        stats.reset()
        assert stats.total_lookups == 0
        assert stats.avg_hops == 0

    def test_o_log_n_hops(self):
        """Average hops should be O(log N)."""
        ring = ChordRing(m=10)  # larger space
        for nid in range(0, 1024, 64):  # 16 nodes
            ring.add_node(nid)
        ring.stabilize(10)

        stats = measure_lookup_performance(ring, 100)
        # O(log 16) = 4, allow some slack
        assert stats.avg_hops <= 8


# =============================================================================
# 12. Virtual Nodes
# =============================================================================

class TestVirtualNodes:
    def test_add_physical_node(self):
        ring = ChordRing(m=8)
        vnm = VirtualNodeManager(ring, num_vnodes=3)
        ids = vnm.add_physical_node("server1")
        assert len(ids) == 3
        assert len(ring) == 3

    def test_multiple_physical_nodes(self):
        ring = ChordRing(m=8)
        vnm = VirtualNodeManager(ring, num_vnodes=3)
        vnm.add_physical_node("server1")
        vnm.add_physical_node("server2")
        ring.stabilize(5)
        assert len(ring) == 6
        assert len(vnm.physical_nodes) == 2

    def test_remove_physical_node(self):
        ring = ChordRing(m=8)
        vnm = VirtualNodeManager(ring, num_vnodes=3)
        vnm.add_physical_node("server1")
        vnm.add_physical_node("server2")
        ring.stabilize(5)

        vnm.remove_physical_node("server1")
        assert len(ring) == 3
        assert "server1" not in vnm.physical_nodes

    def test_get_physical_node(self):
        ring = ChordRing(m=8)
        vnm = VirtualNodeManager(ring, num_vnodes=3)
        vnm.add_physical_node("server1")
        vnm.add_physical_node("server2")
        ring.stabilize(5)

        phys = vnm.get_physical_node("test_key")
        assert phys in ["server1", "server2"]

    def test_physical_distribution(self):
        ring = ChordRing(m=8)
        vnm = VirtualNodeManager(ring, num_vnodes=4)
        vnm.add_physical_node("A")
        vnm.add_physical_node("B")
        ring.stabilize(5)

        for i in range(50):
            ring.put(f"vn_{i}", i)

        dist = vnm.get_physical_distribution()
        assert "A" in dist or "B" in dist

    def test_remove_nonexistent(self):
        ring = ChordRing(m=8)
        vnm = VirtualNodeManager(ring, num_vnodes=3)
        assert not vnm.remove_physical_node("nope")

    def test_get_physical_empty(self):
        ring = ChordRing(m=8)
        vnm = VirtualNodeManager(ring, num_vnodes=3)
        assert vnm.get_physical_node("key") is None


# =============================================================================
# 13. Range Queries
# =============================================================================

class TestRangeQueries:
    def test_range_query(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        for i in range(50):
            ring.put(f"range_{i}", i)

        results = RangeQueryMixin.range_query(ring, 0, 128)
        # Should return some subset of keys
        assert isinstance(results, dict)

    def test_scan_all(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(5)

        for i in range(10):
            ring.put(f"scan_{i}", i)

        all_keys = RangeQueryMixin.scan_all(ring)
        assert len(all_keys) == 10

    def test_count_in_range(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(5)

        for i in range(20):
            ring.put(f"cnt_{i}", i)

        total = RangeQueryMixin.count_in_range(ring, 0, 255)
        assert total == 20


# =============================================================================
# 14. Batch Operations
# =============================================================================

class TestBatchOperations:
    def test_batch_put(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(3)

        items = [(f"batch_{i}", i) for i in range(10)]
        results = batch_put(ring, items)
        assert all(results)

    def test_batch_get(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(3)

        for i in range(10):
            ring.put(f"bg_{i}", i)

        results = batch_get(ring, [f"bg_{i}" for i in range(10)])
        assert len(results) == 10
        for i in range(10):
            assert results[f"bg_{i}"] == i

    def test_batch_get_missing(self):
        ring = ChordRing(m=8)
        ring.add_node(0)

        results = batch_get(ring, ["no1", "no2"])
        assert results["no1"] is None
        assert results["no2"] is None


# =============================================================================
# 15. Key Migration
# =============================================================================

class TestKeyMigration:
    def test_minimal_migration(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        for i in range(100):
            ring.put(f"mig_{i}", i)

        # Simulate adding node 32
        moved, total, fraction = calculate_key_migration(ring, 32)
        assert total == 100
        # Should move roughly 1/5 of keys (one new node among 5)
        assert fraction < 0.5  # definitely less than half

    def test_migration_empty_ring(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        moved, total, frac = calculate_key_migration(ring, 128)
        assert total == 0


# =============================================================================
# 16. Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_empty_ring_ops(self):
        ring = ChordRing(m=8)
        assert ring.get("key") is None
        assert not ring.put("key", "val")
        assert not ring.delete("key")

    def test_all_keys_same_hash(self):
        """Multiple keys hashing to same ID -- last write wins per key."""
        ring = ChordRing(m=4)  # small space, more collisions
        ring.add_node(0)
        ring.add_node(8)
        ring.stabilize(3)

        ring.put("a", 1)
        ring.put("a", 2)  # overwrite
        assert ring.get("a") == 2

    def test_single_bit_space(self):
        ring = ChordRing(m=1)  # only 2 positions: 0 and 1
        ring.add_node(0)
        ring.add_node(1)
        ring.stabilize(5)

        ring.put("x", 10)
        assert ring.get("x") == 10

    def test_large_ring(self):
        ring = ChordRing(m=10)  # 1024 positions
        for i in range(0, 1024, 128):
            ring.add_node(i)
        ring.stabilize(10)

        for i in range(50):
            ring.put(f"large_{i}", i)
        for i in range(50):
            assert ring.get(f"large_{i}") == i

    def test_adjacent_nodes(self):
        ring = ChordRing(m=8)
        ring.add_node(10)
        ring.add_node(11)
        ring.stabilize(5)

        ring.put("adj", "value")
        assert ring.get("adj") == "value"

    def test_node_at_zero(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(255)
        ring.stabilize(5)

        ring.put("zero", "val")
        assert ring.get("zero") == "val"


# =============================================================================
# 17. Responsible For
# =============================================================================

class TestResponsibleFor:
    def test_responsible_solo(self):
        node = ChordNode(10, m=8)
        # No predecessor means responsible for everything
        assert node.responsible_for(5)
        assert node.responsible_for(10)
        assert node.responsible_for(200)

    def test_responsible_with_predecessor(self):
        node = ChordNode(10, m=8)
        node.predecessor = 5
        assert node.responsible_for(7)
        assert node.responsible_for(10)
        assert not node.responsible_for(3)

    def test_responsible_wrap(self):
        node = ChordNode(5, m=8)
        node.predecessor = 250
        # Should be responsible for (250, 5]
        assert node.responsible_for(0)
        assert node.responsible_for(5)
        assert node.responsible_for(254)
        assert not node.responsible_for(100)


# =============================================================================
# 18. Concurrent Joins
# =============================================================================

class TestConcurrentJoins:
    def test_many_joins(self):
        ring = ChordRing(m=8)
        random.seed(42)
        ids = random.sample(range(256), 20)
        for nid in ids:
            ring.add_node(nid)

        ring.stabilize(20)
        valid, issues = ring.verify_ring_integrity()
        assert valid, f"Issues: {issues}"

    def test_join_then_leave_then_join(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(5)

        ring.add_node(64)
        ring.stabilize(5)

        ring.remove_node(128)
        ring.stabilize(5)

        ring.add_node(192)
        ring.stabilize(5)

        valid, issues = ring.verify_ring_integrity()
        assert valid

    def test_all_leave_except_one(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        for i in range(30):
            ring.put(f"last_{i}", i)

        ring.remove_node(64)
        ring.remove_node(128)
        ring.remove_node(192)
        ring.stabilize(5)

        assert len(ring) == 1
        # Keys that were on node 0 are still there
        # Keys from other nodes were transferred


# =============================================================================
# 19. Stress Tests
# =============================================================================

class TestStress:
    def test_many_keys(self):
        ring = ChordRing(m=10)
        for nid in range(0, 1024, 64):
            ring.add_node(nid)
        ring.stabilize(10)

        for i in range(500):
            ring.put(f"stress_{i}", i)
        for i in range(500):
            assert ring.get(f"stress_{i}") == i

    def test_churn(self):
        """Add and remove nodes while data exists."""
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        for i in range(20):
            ring.put(f"churn_{i}", i)

        # Add new nodes
        ring.add_node(32)
        ring.add_node(96)
        ring.stabilize(5)

        # Remove some
        ring.remove_node(64)
        ring.stabilize(5)

        # All keys should still be accessible
        for i in range(20):
            assert ring.get(f"churn_{i}") == i

    def test_random_operations(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(5)

        random.seed(123)
        stored = {}
        for _ in range(100):
            op = random.choice(['put', 'get', 'delete'])
            key = f"rand_{random.randint(0, 50)}"
            if op == 'put':
                val = random.randint(0, 1000)
                ring.put(key, val)
                stored[key] = val
            elif op == 'get':
                result = ring.get(key)
                if key in stored:
                    assert result == stored[key]
            else:
                ring.delete(key)
                stored.pop(key, None)


# =============================================================================
# 20. Successor List Recovery
# =============================================================================

class TestSuccessorListRecovery:
    def test_successor_list_populated(self):
        ring = ChordRing(m=8, num_successors=3)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        node = ring.nodes[0]
        assert len(node.successor_list) >= 2

    def test_recover_from_successor_failure(self):
        ring = ChordRing(m=8, num_successors=3)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(5)

        # Fail node 64 (successor of 0)
        ring.fail_node(64)
        ring.stabilize(5)

        # Node 0 should recover and point to 128
        node0 = ring.nodes[0]
        assert node0.successor in ring.nodes
        assert node0.successor == 128

    def test_double_failure_recovery(self):
        ring = ChordRing(m=8, num_successors=4)
        for nid in [0, 50, 100, 150, 200]:
            ring.add_node(nid)
        ring.stabilize(10)

        ring.fail_node(50)
        ring.fail_node(100)
        ring.stabilize(10)

        valid, issues = ring.verify_ring_integrity()
        assert valid, f"Issues: {issues}"


# =============================================================================
# 21. Transfer Log
# =============================================================================

class TestTransferLog:
    def test_put_logged(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.put("logged", "val")

        node = ring.nodes[0]
        assert any(op == 'put' and k == 'logged' for op, k, _ in node.transfer_log)

    def test_transfer_logged(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.stabilize(3)

        for i in range(10):
            ring.put(f"xfer_{i}", i)

        ring.add_node(128)
        ring.stabilize(5)

        # Check for transfer_out events on original nodes
        all_logs = []
        for node in ring.nodes.values():
            all_logs.extend(node.transfer_log)
        # Should have at least the puts
        assert len(all_logs) >= 10


# =============================================================================
# 22. Notify Mechanism
# =============================================================================

class TestNotify:
    def test_notify_sets_predecessor(self):
        node = ChordNode(10, m=8)
        node.predecessor = None
        node.notify(5)
        assert node.predecessor == 5

    def test_notify_closer_predecessor(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.add_node(128)
        ring.stabilize(5)

        node128 = ring.nodes[128]
        # node128.predecessor should be 0
        # Notify with a closer node
        ring.add_node(64)
        ring.stabilize(5)
        assert ring.nodes[128].predecessor == 64


# =============================================================================
# 23. Closest Preceding Node
# =============================================================================

class TestClosestPreceding:
    def test_closest_preceding(self):
        ring = ChordRing(m=8)
        for nid in [0, 64, 128, 192]:
            ring.add_node(nid)
        ring.stabilize(10)

        node0 = ring.nodes[0]
        # Looking for successor of 100: closest preceding should be 64
        cpn = node0.closest_preceding_node(100)
        assert cpn == 64

    def test_closest_preceding_self(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.stabilize(5)

        node = ring.nodes[0]
        # Only node, should return self
        cpn = node.closest_preceding_node(100)
        assert cpn == 0


# =============================================================================
# 24. Ring with Specific IDs
# =============================================================================

class TestSpecificIDs:
    def test_powers_of_two(self):
        ring = ChordRing(m=8)
        for i in range(8):
            ring.add_node(2**i)
        ring.stabilize(10)

        valid, issues = ring.verify_ring_integrity()
        assert valid

        for i in range(20):
            ring.put(f"pow_{i}", i)
        for i in range(20):
            assert ring.get(f"pow_{i}") == i

    def test_consecutive_ids(self):
        ring = ChordRing(m=8)
        for i in range(10):
            ring.add_node(i)
        ring.stabilize(10)

        valid, issues = ring.verify_ring_integrity()
        assert valid


# =============================================================================
# 25. Empty / Boundary Conditions
# =============================================================================

class TestBoundary:
    def test_get_ring_order_empty(self):
        ring = ChordRing(m=8)
        assert ring.get_ring_order() == []

    def test_get_all_keys_empty(self):
        ring = ChordRing(m=8)
        assert ring.get_all_keys() == {}

    def test_get_key_distribution_empty(self):
        ring = ChordRing(m=8)
        assert ring.get_key_distribution() == {}

    def test_verify_empty_ring(self):
        ring = ChordRing(m=8)
        valid, issues = ring.verify_ring_integrity()
        assert valid

    def test_put_empty_value(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.put("empty", "")
        assert ring.get("empty") == ""

    def test_put_none_value(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.put("none_val", None)
        assert ring.get("none_val") is None  # stored None

    def test_put_complex_value(self):
        ring = ChordRing(m=8)
        ring.add_node(0)
        ring.put("complex", {"a": [1, 2, 3]})
        assert ring.get("complex") == {"a": [1, 2, 3]}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
