"""Tests for C205: Consistent Hashing."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from consistent_hashing import (
    HashRing, WeightedHashRing, BoundedLoadHashRing, RendezvousHashing,
    JumpConsistentHash, MultiProbeHashRing, MaglevHashRing, ReplicatedHashRing,
    md5_hash, sha256_hash, RING_SIZE
)
from collections import defaultdict
import math


# ============================================================
# HashRing tests
# ============================================================

class TestHashRing:
    """Tests for the basic consistent hash ring."""

    def test_empty_ring(self):
        ring = HashRing()
        assert ring.get_node("key") is None
        assert len(ring) == 0

    def test_single_node(self):
        ring = HashRing(["node1"])
        assert ring.get_node("any_key") == "node1"
        assert len(ring) == 1

    def test_add_remove_node(self):
        ring = HashRing()
        ring.add_node("A")
        assert ring.get_node("key") == "A"
        ring.add_node("B")
        assert ring.get_node("key") in ("A", "B")
        ring.remove_node("A")
        ring.remove_node("B")
        assert ring.get_node("key") is None

    def test_contains(self):
        ring = HashRing(["A", "B"])
        assert "A" in ring
        assert "C" not in ring

    def test_consistent_mapping(self):
        """Same key always maps to same node."""
        ring = HashRing(["A", "B", "C"])
        results = [ring.get_node("test_key") for _ in range(100)]
        assert len(set(results)) == 1

    def test_add_node_minimal_disruption(self):
        """Adding a node should only remap ~1/n keys."""
        nodes = [f"node_{i}" for i in range(5)]
        ring = HashRing(nodes, num_replicas=150)

        keys = [f"key_{i}" for i in range(1000)]
        before = {k: ring.get_node(k) for k in keys}

        ring.add_node("node_new")
        after = {k: ring.get_node(k) for k in keys}

        changed = sum(1 for k in keys if before[k] != after[k])
        # Should be roughly 1/6 of keys (adding 6th node)
        assert changed < 400  # generous bound

    def test_remove_node_minimal_disruption(self):
        """Removing a node should only remap that node's keys."""
        nodes = [f"node_{i}" for i in range(6)]
        ring = HashRing(nodes, num_replicas=150)

        keys = [f"key_{i}" for i in range(1000)]
        before = {k: ring.get_node(k) for k in keys}

        ring.remove_node("node_3")
        after = {k: ring.get_node(k) for k in keys}

        # Only keys that were on node_3 should move
        for k in keys:
            if before[k] != "node_3":
                assert after[k] == before[k], f"Key {k} moved from {before[k]} to {after[k]}"

    def test_distribution_balance(self):
        """Keys should be roughly evenly distributed."""
        ring = HashRing([f"node_{i}" for i in range(5)], num_replicas=150)
        dist = ring.get_distribution(10000)
        expected = 10000 / 5
        for node, count in dist.items():
            # Each node should get 10-30% of keys (generous)
            assert 500 < count < 4000, f"{node}: {count}"

    def test_get_nodes_multiple(self):
        ring = HashRing(["A", "B", "C", "D"])
        nodes = ring.get_nodes("key", count=3)
        assert len(nodes) == 3
        assert len(set(nodes)) == 3  # all distinct

    def test_get_nodes_more_than_available(self):
        ring = HashRing(["A", "B"])
        nodes = ring.get_nodes("key", count=5)
        assert len(nodes) == 2

    def test_get_nodes_empty(self):
        ring = HashRing()
        assert ring.get_nodes("key", count=3) == []
        ring2 = HashRing(["A"])
        assert ring2.get_nodes("key", count=0) == []

    def test_duplicate_add(self):
        ring = HashRing(["A"])
        ring.add_node("A")
        assert len(ring) == 1

    def test_remove_nonexistent(self):
        ring = HashRing(["A"])
        ring.remove_node("Z")  # should not error
        assert len(ring) == 1

    def test_nodes_property(self):
        ring = HashRing(["A", "B", "C"])
        assert ring.nodes == frozenset({"A", "B", "C"})

    def test_many_nodes(self):
        nodes = [f"n{i}" for i in range(100)]
        ring = HashRing(nodes, num_replicas=50)
        assert len(ring) == 100
        node = ring.get_node("any_key")
        assert node in nodes

    def test_custom_hash_fn(self):
        ring = HashRing(["A", "B"], hash_fn=sha256_hash)
        assert ring.get_node("key") in ("A", "B")

    def test_string_and_numeric_keys(self):
        ring = HashRing(["A", "B", "C"])
        assert ring.get_node("hello") is not None
        assert ring.get_node("12345") is not None


# ============================================================
# WeightedHashRing tests
# ============================================================

class TestWeightedHashRing:

    def test_empty(self):
        ring = WeightedHashRing()
        assert ring.get_node("key") is None

    def test_single_node(self):
        ring = WeightedHashRing()
        ring.add_node("A", weight=1.0)
        assert ring.get_node("key") == "A"

    def test_weighted_distribution(self):
        """Higher weight node should get more keys."""
        ring = WeightedHashRing(base_replicas=100)
        ring.add_node("heavy", weight=3.0)
        ring.add_node("light", weight=1.0)
        dist = ring.get_distribution(10000)
        # Heavy should get roughly 3x of light
        assert dist.get("heavy", 0) > dist.get("light", 0)

    def test_remove_weighted_node(self):
        ring = WeightedHashRing()
        ring.add_node("A", weight=2.0)
        ring.add_node("B", weight=1.0)
        assert ring.get_node("key") in ("A", "B")
        ring.remove_node("A")
        assert ring.get_node("key") == "B"

    def test_reweight_node(self):
        """Adding a node that already exists with new weight re-adds it."""
        ring = WeightedHashRing()
        ring.add_node("A", weight=1.0)
        ring.add_node("A", weight=5.0)
        assert ring.nodes == {"A": 5.0}

    def test_nodes_property(self):
        ring = WeightedHashRing()
        ring.add_node("A", weight=2.0)
        ring.add_node("B", weight=0.5)
        assert ring.nodes == {"A": 2.0, "B": 0.5}

    def test_remove_nonexistent(self):
        ring = WeightedHashRing()
        ring.remove_node("Z")  # no error


# ============================================================
# BoundedLoadHashRing tests
# ============================================================

class TestBoundedLoadHashRing:

    def test_empty(self):
        ring = BoundedLoadHashRing()
        assert ring.assign("key") is None

    def test_single_node(self):
        ring = BoundedLoadHashRing(["A"])
        assert ring.assign("k1") == "A"
        assert ring.assign("k2") == "A"

    def test_load_bounded(self):
        """No node should exceed the bounded load."""
        ring = BoundedLoadHashRing([f"n{i}" for i in range(5)], epsilon=0.25)
        for i in range(100):
            ring.assign(f"key_{i}")

        loads = ring.get_all_loads()
        avg = 100 / 5
        cap = math.ceil(avg * 1.25)
        for node, load in loads.items():
            assert load <= cap + 1, f"{node} has load {load}, cap is {cap}"

    def test_release(self):
        ring = BoundedLoadHashRing(["A", "B"])
        ring.assign("k1")
        load_before = ring.get_load("A") + ring.get_load("B")
        ring.release("k1")
        load_after = ring.get_load("A") + ring.get_load("B")
        assert load_after == load_before - 1

    def test_idempotent_assign(self):
        """Assigning same key twice returns same node."""
        ring = BoundedLoadHashRing(["A", "B", "C"])
        n1 = ring.assign("key")
        n2 = ring.assign("key")
        assert n1 == n2

    def test_release_nonexistent(self):
        ring = BoundedLoadHashRing(["A"])
        ring.release("nonexistent")  # no error

    def test_add_remove_node(self):
        ring = BoundedLoadHashRing(["A", "B"])
        ring.assign("k1")
        ring.add_node("C")
        ring.assign("k2")
        ring.remove_node("A")
        # Orphaned keys from A are released
        assert "A" not in ring.nodes

    def test_epsilon_effect(self):
        """Smaller epsilon means tighter bound."""
        ring = BoundedLoadHashRing([f"n{i}" for i in range(5)], epsilon=0.1)
        for i in range(50):
            ring.assign(f"key_{i}")
        loads = ring.get_all_loads()
        max_load = max(loads.values()) if loads else 0
        min_load = min(loads.values()) if loads else 0
        # With epsilon=0.1, difference should be small
        assert max_load - min_load <= 10


# ============================================================
# RendezvousHashing tests
# ============================================================

class TestRendezvousHashing:

    def test_empty(self):
        rh = RendezvousHashing()
        assert rh.get_node("key") is None

    def test_single_node(self):
        rh = RendezvousHashing(["A"])
        assert rh.get_node("key") == "A"

    def test_consistent(self):
        rh = RendezvousHashing(["A", "B", "C"])
        results = [rh.get_node("test") for _ in range(50)]
        assert len(set(results)) == 1

    def test_minimal_disruption_add(self):
        rh = RendezvousHashing([f"n{i}" for i in range(5)])
        keys = [f"k{i}" for i in range(1000)]
        before = {k: rh.get_node(k) for k in keys}
        rh.add_node("n_new")
        after = {k: rh.get_node(k) for k in keys}
        changed = sum(1 for k in keys if before[k] != after[k])
        assert changed < 400

    def test_minimal_disruption_remove(self):
        rh = RendezvousHashing([f"n{i}" for i in range(5)])
        keys = [f"k{i}" for i in range(1000)]
        before = {k: rh.get_node(k) for k in keys}
        rh.remove_node("n2")
        after = {k: rh.get_node(k) for k in keys}
        for k in keys:
            if before[k] != "n2":
                assert after[k] == before[k]

    def test_get_nodes_ranked(self):
        rh = RendezvousHashing(["A", "B", "C", "D"])
        nodes = rh.get_nodes("key", count=3)
        assert len(nodes) == 3
        assert len(set(nodes)) == 3

    def test_distribution(self):
        rh = RendezvousHashing([f"n{i}" for i in range(4)])
        dist = rh.get_distribution(10000)
        expected = 2500
        for count in dist.values():
            assert 1000 < count < 4000

    def test_nodes_property(self):
        rh = RendezvousHashing(["X", "Y"])
        assert rh.nodes == frozenset({"X", "Y"})


# ============================================================
# JumpConsistentHash tests
# ============================================================

class TestJumpConsistentHash:

    def test_single_bucket(self):
        assert JumpConsistentHash.hash("key", 1) == 0

    def test_range(self):
        for i in range(100):
            bucket = JumpConsistentHash.hash(f"key_{i}", 10)
            assert 0 <= bucket < 10

    def test_consistent(self):
        results = [JumpConsistentHash.hash("stable_key", 5) for _ in range(100)]
        assert len(set(results)) == 1

    def test_minimal_movement_add_bucket(self):
        """Adding a bucket should move roughly 1/n keys."""
        keys = [f"k{i}" for i in range(1000)]
        before = {k: JumpConsistentHash.hash(k, 10) for k in keys}
        after = {k: JumpConsistentHash.hash(k, 11) for k in keys}
        moved = sum(1 for k in keys if before[k] != after[k])
        # ~1/11 should move, so about 91. Allow generous bound.
        assert moved < 250

    def test_distribution(self):
        dist = JumpConsistentHash.distribute(
            [f"key_{i}" for i in range(10000)], 5
        )
        for bucket, keys in dist.items():
            assert 1000 < len(keys) < 3500

    def test_zero_buckets_raises(self):
        try:
            JumpConsistentHash.hash("key", 0)
            assert False, "Should raise"
        except ValueError:
            pass

    def test_string_key(self):
        b = JumpConsistentHash.hash("hello", 100)
        assert 0 <= b < 100

    def test_integer_key(self):
        b = JumpConsistentHash.hash(42, 10)
        assert 0 <= b < 10

    def test_monotonicity(self):
        """Key should either stay in same bucket or move to new one when growing."""
        key = "monotone_test"
        prev = JumpConsistentHash.hash(key, 1)
        for n in range(2, 20):
            curr = JumpConsistentHash.hash(key, n)
            assert curr == prev or curr == n - 1
            prev = curr


# ============================================================
# MultiProbeHashRing tests
# ============================================================

class TestMultiProbeHashRing:

    def test_empty(self):
        ring = MultiProbeHashRing()
        assert ring.get_node("key") is None

    def test_single_node(self):
        ring = MultiProbeHashRing(["A"])
        assert ring.get_node("key") == "A"

    def test_consistent(self):
        ring = MultiProbeHashRing(["A", "B", "C"])
        results = [ring.get_node("key") for _ in range(50)]
        assert len(set(results)) == 1

    def test_distribution(self):
        """Multi-probe should achieve decent balance even with 1 vnode per node."""
        ring = MultiProbeHashRing([f"n{i}" for i in range(10)], num_probes=21)
        dist = ring.get_distribution(10000)
        assert len(dist) == 10
        for count in dist.values():
            # With 10 nodes and 21 probes, distribution is reasonable
            assert 200 < count < 3000

    def test_add_remove(self):
        ring = MultiProbeHashRing(["A", "B"])
        n1 = ring.get_node("key")
        ring.add_node("C")
        ring.remove_node("A")
        ring.remove_node("B")
        assert ring.get_node("key") == "C"

    def test_minimal_disruption(self):
        ring = MultiProbeHashRing([f"n{i}" for i in range(5)])
        keys = [f"k{i}" for i in range(500)]
        before = {k: ring.get_node(k) for k in keys}
        ring.add_node("n_new")
        after = {k: ring.get_node(k) for k in keys}
        changed = sum(1 for k in keys if before[k] != after[k])
        assert changed < 300

    def test_probes_parameter(self):
        ring = MultiProbeHashRing(["A", "B"], num_probes=1)
        assert ring.get_node("key") in ("A", "B")

    def test_nodes_property(self):
        ring = MultiProbeHashRing(["X", "Y", "Z"])
        assert ring.nodes == frozenset({"X", "Y", "Z"})


# ============================================================
# MaglevHashRing tests
# ============================================================

class TestMaglevHashRing:

    def test_empty(self):
        ring = MaglevHashRing()
        assert ring.get_node("key") is None

    def test_single_node(self):
        ring = MaglevHashRing(["A"], table_size=17)
        assert ring.get_node("key") == "A"

    def test_consistent(self):
        ring = MaglevHashRing(["A", "B", "C"], table_size=101)
        results = [ring.get_node("key") for _ in range(50)]
        assert len(set(results)) == 1

    def test_distribution(self):
        ring = MaglevHashRing([f"n{i}" for i in range(5)], table_size=1009)
        dist = ring.get_distribution(10000)
        assert len(dist) == 5
        for count in dist.values():
            assert 1000 < count < 4000

    def test_add_node_minimal_disruption(self):
        ring = MaglevHashRing([f"n{i}" for i in range(5)], table_size=1009)
        keys = [f"k{i}" for i in range(1000)]
        before = {k: ring.get_node(k) for k in keys}
        ring.add_node("n_new")
        after = {k: ring.get_node(k) for k in keys}
        changed = sum(1 for k in keys if before[k] != after[k])
        assert changed < 500

    def test_remove_node(self):
        ring = MaglevHashRing(["A", "B", "C"], table_size=17)
        ring.remove_node("B")
        for i in range(20):
            assert ring.get_node(f"k{i}") in ("A", "C")

    def test_table_fills_completely(self):
        ring = MaglevHashRing(["A", "B", "C"], table_size=17)
        assert None not in ring._table

    def test_prime_table_size(self):
        ring = MaglevHashRing(["A"], table_size=10)
        assert ring._table_size == 11  # next prime >= 10

    def test_nodes_property(self):
        ring = MaglevHashRing(["A", "B"])
        assert ring.nodes == frozenset({"A", "B"})

    def test_duplicate_add(self):
        ring = MaglevHashRing(["A"], table_size=17)
        ring.add_node("A")  # no error
        assert len(ring.nodes) == 1


# ============================================================
# ReplicatedHashRing tests
# ============================================================

class TestReplicatedHashRing:

    def test_empty(self):
        ring = ReplicatedHashRing()
        assert ring.get_coordinator("key") is None

    def test_single_node(self):
        ring = ReplicatedHashRing(["A"], replication_factor=3)
        assert ring.get_coordinator("key") == "A"
        assert ring.get_preference_list("key") == ["A"]

    def test_preference_list_size(self):
        ring = ReplicatedHashRing([f"n{i}" for i in range(5)], replication_factor=3)
        pref = ring.get_preference_list("key")
        assert len(pref) == 3
        assert len(set(pref)) == 3

    def test_preference_list_with_failures(self):
        ring = ReplicatedHashRing([f"n{i}" for i in range(5)], replication_factor=3)
        pref_before = ring.get_preference_list("key")
        coordinator = pref_before[0]
        ring.mark_failed(coordinator)
        pref_after = ring.get_preference_list("key")
        assert coordinator not in pref_after
        assert len(pref_after) == 3

    def test_coordinator_skips_failed(self):
        ring = ReplicatedHashRing(["A", "B", "C"], replication_factor=1)
        coord = ring.get_coordinator("key")
        ring.mark_failed(coord)
        new_coord = ring.get_coordinator("key")
        assert new_coord != coord
        assert new_coord is not None

    def test_mark_healthy(self):
        ring = ReplicatedHashRing(["A", "B", "C"], replication_factor=1)
        ring.mark_failed("A")
        ring.mark_healthy("A")
        assert "A" in ring.healthy_nodes

    def test_handoff_node(self):
        ring = ReplicatedHashRing([f"n{i}" for i in range(6)], replication_factor=3)
        pref = ring.get_preference_list("key")
        failed = pref[0]
        ring.mark_failed(failed)
        handoff = ring.get_handoff_node("key", failed)
        assert handoff is not None
        assert handoff not in pref

    def test_handoff_all_failed(self):
        ring = ReplicatedHashRing(["A", "B", "C"], replication_factor=3)
        for n in ["A", "B", "C"]:
            ring.mark_failed(n)
        assert ring.get_handoff_node("key", "A") is None

    def test_add_remove_node(self):
        ring = ReplicatedHashRing(["A", "B", "C"])
        ring.add_node("D")
        assert "D" in ring.nodes
        ring.remove_node("D")
        assert "D" not in ring.nodes

    def test_key_ranges(self):
        ring = ReplicatedHashRing(["A", "B", "C"], num_replicas=10)
        ranges = ring.get_key_ranges("A")
        assert len(ranges) == 10
        for start, end in ranges:
            assert isinstance(start, int)
            assert isinstance(end, int)

    def test_replication_factor_property(self):
        ring = ReplicatedHashRing(replication_factor=5)
        assert ring.replication_factor == 5

    def test_healthy_nodes_property(self):
        ring = ReplicatedHashRing(["A", "B", "C"])
        ring.mark_failed("B")
        assert ring.healthy_nodes == frozenset({"A", "C"})


# ============================================================
# Cross-cutting / Integration tests
# ============================================================

class TestCrossCutting:

    def test_all_algorithms_agree_on_coverage(self):
        """All algorithms should assign every key to some node."""
        nodes = ["A", "B", "C", "D"]
        algorithms = [
            HashRing(nodes),
            WeightedHashRing(),
            RendezvousHashing(nodes),
            MultiProbeHashRing(nodes),
            MaglevHashRing(nodes, table_size=17),
        ]
        # Add nodes to weighted ring
        for n in nodes:
            algorithms[1].add_node(n, weight=1.0)

        for algo in algorithms:
            for i in range(100):
                node = algo.get_node(f"key_{i}")
                assert node is not None, f"{algo.__class__.__name__} returned None"
                assert node in nodes, f"{algo.__class__.__name__} returned unknown node {node}"

    def test_hash_functions_deterministic(self):
        assert md5_hash("test") == md5_hash("test")
        assert sha256_hash("test") == sha256_hash("test")

    def test_hash_functions_different(self):
        """Different hash functions should produce different values."""
        assert md5_hash("test") != sha256_hash("test")

    def test_large_cluster_simulation(self):
        """Simulate a 50-node cluster with consistent hashing."""
        nodes = [f"server_{i:03d}" for i in range(50)]
        ring = HashRing(nodes, num_replicas=100)
        dist = ring.get_distribution(50000)
        expected = 50000 / 50

        max_load = max(dist.values())
        min_load = min(dist.values())
        # Max/min ratio should be reasonable
        assert max_load / max(min_load, 1) < 5.0

    def test_node_addition_cascade(self):
        """Add nodes one by one, verify distribution improves."""
        ring = HashRing(num_replicas=150)
        ring.add_node("n0")
        dist0 = ring.get_distribution(1000)
        assert dist0["n0"] == 1000  # single node gets all

        ring.add_node("n1")
        dist1 = ring.get_distribution(1000)
        assert len(dist1) == 2

    def test_jump_hash_vs_ring_coverage(self):
        """Both should cover all keys."""
        keys = [f"k{i}" for i in range(100)]
        ring = HashRing(["A", "B", "C"])
        for k in keys:
            assert ring.get_node(k) in ("A", "B", "C")
            assert 0 <= JumpConsistentHash.hash(k, 3) < 3

    def test_replicated_ring_under_stress(self):
        """Replicated ring with many failures."""
        nodes = [f"n{i}" for i in range(10)]
        ring = ReplicatedHashRing(nodes, replication_factor=3)
        # Fail half the nodes
        for i in range(5):
            ring.mark_failed(f"n{i}")
        # Should still find coordinators
        for i in range(50):
            coord = ring.get_coordinator(f"key_{i}")
            assert coord is not None
            assert coord not in ring._failed_nodes

    def test_bounded_load_under_stress(self):
        """Bounded load ring with high contention."""
        ring = BoundedLoadHashRing([f"n{i}" for i in range(3)], epsilon=0.5)
        for i in range(30):
            node = ring.assign(f"key_{i}")
            assert node is not None

        loads = ring.get_all_loads()
        total = sum(loads.values())
        assert total == 30

    def test_maglev_table_all_nodes_present(self):
        """Every node should appear in the Maglev table."""
        nodes = ["A", "B", "C", "D", "E"]
        ring = MaglevHashRing(nodes, table_size=101)
        present = set(ring._table)
        assert present == set(nodes)


# ============================================================
# Runner
# ============================================================

def run_tests():
    test_classes = [
        TestHashRing,
        TestWeightedHashRing,
        TestBoundedLoadHashRing,
        TestRendezvousHashing,
        TestJumpConsistentHash,
        TestMultiProbeHashRing,
        TestMaglevHashRing,
        TestReplicatedHashRing,
        TestCrossCutting,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS: {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e))
                print(f"  FAIL: {cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"  C205 Consistent Hashing: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailures:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}: {err}")

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
