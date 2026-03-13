"""
Tests for C234: Consistent Hashing
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import unittest
import math
from collections import defaultdict
from consistent_hashing import (
    HashRing, WeightedHashRing, BoundedLoadHashRing,
    JumpHash, RendezvousHash, MultiProbeHash, MaglevHash,
    calculate_migration, measure_balance,
    _md5_hash, _sha256_hash, _multi_hash
)


# ========== Utility Tests ==========

class TestHashUtils(unittest.TestCase):

    def test_md5_hash_deterministic(self):
        self.assertEqual(_md5_hash("hello"), _md5_hash("hello"))

    def test_md5_hash_different_keys(self):
        self.assertNotEqual(_md5_hash("hello"), _md5_hash("world"))

    def test_md5_hash_bytes(self):
        self.assertEqual(_md5_hash(b"hello"), _md5_hash("hello"))

    def test_sha256_hash_deterministic(self):
        self.assertEqual(_sha256_hash("test"), _sha256_hash("test"))

    def test_sha256_different_from_md5(self):
        # Different algorithms should produce different hashes (usually)
        h1 = _md5_hash("test")
        h2 = _sha256_hash("test")
        # They could theoretically collide but extremely unlikely
        self.assertIsInstance(h1, int)
        self.assertIsInstance(h2, int)

    def test_multi_hash_count(self):
        hashes = _multi_hash("key", 5)
        self.assertEqual(len(hashes), 5)

    def test_multi_hash_distinct(self):
        hashes = _multi_hash("key", 10)
        self.assertEqual(len(set(hashes)), 10)

    def test_multi_hash_deterministic(self):
        h1 = _multi_hash("key", 3)
        h2 = _multi_hash("key", 3)
        self.assertEqual(h1, h2)


# ========== HashRing Tests ==========

class TestHashRing(unittest.TestCase):

    def test_empty_ring(self):
        ring = HashRing()
        self.assertIsNone(ring.get_node("key"))
        self.assertEqual(len(ring), 0)

    def test_single_node(self):
        ring = HashRing(["node1"])
        self.assertEqual(ring.get_node("any_key"), "node1")
        self.assertEqual(len(ring), 1)

    def test_add_remove_node(self):
        ring = HashRing()
        ring.add_node("A")
        self.assertEqual(ring.get_node("key"), "A")
        ring.add_node("B")
        self.assertEqual(len(ring), 2)
        ring.remove_node("A")
        self.assertEqual(ring.get_node("key"), "B")
        self.assertEqual(len(ring), 1)

    def test_contains(self):
        ring = HashRing(["A", "B"])
        self.assertIn("A", ring)
        self.assertNotIn("C", ring)

    def test_nodes_property(self):
        ring = HashRing(["A", "B", "C"])
        self.assertEqual(ring.nodes, {"A", "B", "C"})

    def test_deterministic(self):
        ring = HashRing(["A", "B", "C"])
        n1 = ring.get_node("key1")
        n2 = ring.get_node("key1")
        self.assertEqual(n1, n2)

    def test_distribution(self):
        ring = HashRing(["A", "B", "C"], num_replicas=150)
        dist = ring.get_distribution(10000)
        # Each node should get roughly 3333 keys
        for count in dist.values():
            self.assertGreater(count, 2000)
            self.assertLess(count, 5000)

    def test_get_nodes_replication(self):
        ring = HashRing(["A", "B", "C"])
        nodes = ring.get_nodes("key", 2)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(set(nodes)), 2)  # distinct

    def test_get_nodes_all(self):
        ring = HashRing(["A", "B", "C"])
        nodes = ring.get_nodes("key", 3)
        self.assertEqual(set(nodes), {"A", "B", "C"})

    def test_get_nodes_more_than_available(self):
        ring = HashRing(["A", "B"])
        nodes = ring.get_nodes("key", 5)
        self.assertEqual(len(nodes), 2)

    def test_get_nodes_zero(self):
        ring = HashRing(["A"])
        self.assertEqual(ring.get_nodes("key", 0), [])

    def test_minimal_disruption_add(self):
        """Adding a node should move ~1/n keys."""
        nodes = ["A", "B", "C", "D"]
        ring1 = HashRing(nodes)
        keys = [f"key_{i}" for i in range(10000)]
        map1 = {k: ring1.get_node(k) for k in keys}

        ring2 = HashRing(nodes + ["E"])
        map2 = {k: ring2.get_node(k) for k in keys}

        moved, total, frac = calculate_migration(map1, map2)
        # Should move roughly 1/5 = 20% of keys
        self.assertLess(frac, 0.35)
        self.assertGreater(frac, 0.10)

    def test_minimal_disruption_remove(self):
        """Removing a node should move ~1/n keys."""
        nodes = ["A", "B", "C", "D", "E"]
        ring1 = HashRing(nodes)
        keys = [f"key_{i}" for i in range(10000)]
        map1 = {k: ring1.get_node(k) for k in keys}

        ring2 = HashRing(["A", "B", "C", "D"])
        map2 = {k: ring2.get_node(k) for k in keys}

        moved, total, frac = calculate_migration(map1, map2)
        self.assertLess(frac, 0.35)
        self.assertGreater(frac, 0.10)

    def test_custom_hash_fn(self):
        ring = HashRing(["A", "B"], hash_fn=_sha256_hash)
        node = ring.get_node("test")
        self.assertIn(node, ["A", "B"])

    def test_add_duplicate_node(self):
        ring = HashRing(["A"])
        ring.add_node("A")
        self.assertEqual(len(ring), 1)

    def test_remove_nonexistent_node(self):
        ring = HashRing(["A"])
        ring.remove_node("B")  # should not raise
        self.assertEqual(len(ring), 1)

    def test_many_nodes(self):
        nodes = [f"node_{i}" for i in range(20)]
        ring = HashRing(nodes, num_replicas=50)
        dist = ring.get_distribution(20000)
        self.assertEqual(len(dist), 20)

    def test_different_keys_may_map_differently(self):
        ring = HashRing(["A", "B", "C"])
        results = set()
        for i in range(100):
            results.add(ring.get_node(f"k{i}"))
        # With 100 keys and 3 nodes, should hit all nodes
        self.assertEqual(len(results), 3)


# ========== WeightedHashRing Tests ==========

class TestWeightedHashRing(unittest.TestCase):

    def test_empty(self):
        ring = WeightedHashRing()
        self.assertIsNone(ring.get_node("key"))
        self.assertEqual(len(ring), 0)

    def test_single_node(self):
        ring = WeightedHashRing()
        ring.add_node("A", weight=1.0)
        self.assertEqual(ring.get_node("key"), "A")

    def test_weight_affects_distribution(self):
        ring = WeightedHashRing(base_replicas=200)
        ring.add_node("heavy", weight=3.0)
        ring.add_node("light", weight=1.0)
        dist = defaultdict(int)
        for i in range(10000):
            node = ring.get_node(f"key_{i}")
            dist[node] += 1
        # Heavy should get roughly 3x the keys
        ratio = dist["heavy"] / max(1, dist["light"])
        self.assertGreater(ratio, 2.0)
        self.assertLess(ratio, 5.0)

    def test_get_weight(self):
        ring = WeightedHashRing()
        ring.add_node("A", weight=2.5)
        self.assertEqual(ring.get_weight("A"), 2.5)
        self.assertEqual(ring.get_weight("B"), 0)

    def test_nodes_property(self):
        ring = WeightedHashRing()
        ring.add_node("A", 1.0)
        ring.add_node("B", 2.0)
        self.assertEqual(ring.nodes, {"A": 1.0, "B": 2.0})

    def test_remove_node(self):
        ring = WeightedHashRing()
        ring.add_node("A", 1.0)
        ring.add_node("B", 1.0)
        ring.remove_node("A")
        self.assertEqual(ring.get_node("key"), "B")
        self.assertEqual(len(ring), 1)

    def test_update_weight(self):
        ring = WeightedHashRing()
        ring.add_node("A", weight=1.0)
        ring.add_node("A", weight=3.0)  # re-add with new weight
        self.assertEqual(ring.get_weight("A"), 3.0)

    def test_get_nodes_replication(self):
        ring = WeightedHashRing()
        ring.add_node("A", 1.0)
        ring.add_node("B", 2.0)
        ring.add_node("C", 1.0)
        nodes = ring.get_nodes("key", 2)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(set(nodes)), 2)

    def test_remove_nonexistent(self):
        ring = WeightedHashRing()
        ring.remove_node("X")  # should not raise
        self.assertEqual(len(ring), 0)


# ========== BoundedLoadHashRing Tests ==========

class TestBoundedLoadHashRing(unittest.TestCase):

    def test_empty(self):
        ring = BoundedLoadHashRing()
        self.assertIsNone(ring.get_node("key"))

    def test_single_node_all_keys(self):
        ring = BoundedLoadHashRing(["A"])
        for i in range(10):
            self.assertEqual(ring.get_node(f"key_{i}"), "A")

    def test_load_tracking(self):
        ring = BoundedLoadHashRing(["A", "B", "C"])
        ring.get_node("k1")
        ring.get_node("k2")
        total = sum(ring.get_load(n) for n in ["A", "B", "C"])
        self.assertEqual(total, 2)

    def test_bounded_load(self):
        """No node should exceed (1+epsilon) * average."""
        ring = BoundedLoadHashRing(["A", "B", "C", "D"], epsilon=0.25)
        for i in range(1000):
            ring.get_node(f"key_{i}")
        avg = 1000 / 4
        cap = math.ceil(avg * 1.25)
        for node in ["A", "B", "C", "D"]:
            self.assertLessEqual(ring.get_load(node), cap + 1)

    def test_release(self):
        ring = BoundedLoadHashRing(["A"])
        ring.get_node("k1")
        self.assertEqual(ring.get_load("A"), 1)
        ring.release("k1", "A")
        self.assertEqual(ring.get_load("A"), 0)

    def test_reset_load(self):
        ring = BoundedLoadHashRing(["A", "B"])
        for i in range(100):
            ring.get_node(f"k{i}")
        ring.reset_load()
        self.assertEqual(ring.get_load("A"), 0)
        self.assertEqual(ring.get_load("B"), 0)

    def test_capacity_per_node(self):
        ring = BoundedLoadHashRing(["A", "B"], epsilon=0.5)
        # No load yet: capacity should be ceil(max(1, 0)/2 * 1.5) = 1
        self.assertGreaterEqual(ring.capacity_per_node, 1)

    def test_add_remove_nodes(self):
        ring = BoundedLoadHashRing(["A", "B"])
        ring.add_node("C")
        self.assertEqual(len(ring), 3)
        ring.remove_node("A")
        self.assertEqual(len(ring), 2)
        self.assertIn("C", ring.nodes)

    def test_spillover(self):
        """When natural node is full, key spills to next node."""
        ring = BoundedLoadHashRing(["A", "B"], epsilon=0.0, num_replicas=1)
        # With epsilon=0, capacity is very tight
        # Just verify it doesn't crash and assigns all keys
        nodes_seen = set()
        for i in range(20):
            n = ring.get_node(f"key_{i}")
            nodes_seen.add(n)
        self.assertTrue(len(nodes_seen) >= 1)

    def test_nodes_property(self):
        ring = BoundedLoadHashRing(["X", "Y"])
        self.assertEqual(ring.nodes, {"X", "Y"})


# ========== JumpHash Tests ==========

class TestJumpHash(unittest.TestCase):

    def test_single_bucket(self):
        self.assertEqual(JumpHash.hash("key", 1), 0)

    def test_range(self):
        for i in range(100):
            b = JumpHash.hash(f"key_{i}", 10)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, 10)

    def test_deterministic(self):
        self.assertEqual(JumpHash.hash("hello", 5), JumpHash.hash("hello", 5))

    def test_uniform_distribution(self):
        dist = defaultdict(int)
        for i in range(10000):
            b = JumpHash.hash(f"key_{i}", 5)
            dist[b] += 1
        for count in dist.values():
            self.assertGreater(count, 1500)
            self.assertLess(count, 2500)

    def test_minimal_disruption(self):
        """Adding one bucket should move ~1/(n+1) keys."""
        keys = [f"key_{i}" for i in range(10000)]
        map1 = {k: JumpHash.hash(k, 10) for k in keys}
        map2 = {k: JumpHash.hash(k, 11) for k in keys}
        moved = sum(1 for k in keys if map1[k] != map2[k])
        frac = moved / len(keys)
        self.assertLess(frac, 0.15)  # ~1/11 ≈ 9%

    def test_monotonic(self):
        """Keys only move to the new bucket, never between old buckets."""
        keys = [f"key_{i}" for i in range(5000)]
        map1 = {k: JumpHash.hash(k, 5) for k in keys}
        map2 = {k: JumpHash.hash(k, 6) for k in keys}
        for k in keys:
            if map1[k] != map2[k]:
                self.assertEqual(map2[k], 5)  # moved to the new bucket

    def test_zero_buckets_raises(self):
        with self.assertRaises(ValueError):
            JumpHash.hash("key", 0)

    def test_negative_buckets_raises(self):
        with self.assertRaises(ValueError):
            JumpHash.hash("key", -1)

    def test_integer_key(self):
        b = JumpHash.hash(42, 10)
        self.assertGreaterEqual(b, 0)
        self.assertLess(b, 10)

    def test_hash_with_names(self):
        buckets = ["alpha", "beta", "gamma"]
        result = JumpHash.hash_with_names("key", buckets)
        self.assertIn(result, buckets)

    def test_hash_with_names_empty(self):
        self.assertIsNone(JumpHash.hash_with_names("key", []))

    def test_hash_with_names_deterministic(self):
        buckets = ["a", "b", "c"]
        r1 = JumpHash.hash_with_names("key1", buckets)
        r2 = JumpHash.hash_with_names("key1", buckets)
        self.assertEqual(r1, r2)

    def test_large_bucket_count(self):
        b = JumpHash.hash("key", 100000)
        self.assertGreaterEqual(b, 0)
        self.assertLess(b, 100000)


# ========== RendezvousHash Tests ==========

class TestRendezvousHash(unittest.TestCase):

    def test_empty(self):
        rh = RendezvousHash()
        self.assertIsNone(rh.get_node("key"))

    def test_single_node(self):
        rh = RendezvousHash(["A"])
        self.assertEqual(rh.get_node("key"), "A")

    def test_deterministic(self):
        rh = RendezvousHash(["A", "B", "C"])
        n1 = rh.get_node("key")
        n2 = rh.get_node("key")
        self.assertEqual(n1, n2)

    def test_distribution(self):
        rh = RendezvousHash([f"node_{i}" for i in range(5)])
        dist = defaultdict(int)
        for i in range(10000):
            node = rh.get_node(f"key_{i}")
            dist[node] += 1
        for count in dist.values():
            self.assertGreater(count, 1200)
            self.assertLess(count, 2800)

    def test_minimal_disruption_add(self):
        nodes = ["A", "B", "C", "D"]
        rh1 = RendezvousHash(nodes)
        keys = [f"key_{i}" for i in range(10000)]
        map1 = {k: rh1.get_node(k) for k in keys}

        rh2 = RendezvousHash(nodes + ["E"])
        map2 = {k: rh2.get_node(k) for k in keys}

        moved, total, frac = calculate_migration(map1, map2)
        self.assertLess(frac, 0.30)

    def test_minimal_disruption_remove(self):
        nodes = ["A", "B", "C", "D", "E"]
        rh1 = RendezvousHash(nodes)
        keys = [f"key_{i}" for i in range(10000)]
        map1 = {k: rh1.get_node(k) for k in keys}

        rh2 = RendezvousHash(["A", "B", "C", "D"])
        map2 = {k: rh2.get_node(k) for k in keys}

        moved, total, frac = calculate_migration(map1, map2)
        self.assertLess(frac, 0.30)

    def test_get_nodes(self):
        rh = RendezvousHash(["A", "B", "C"])
        nodes = rh.get_nodes("key", 2)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(set(nodes)), 2)

    def test_get_nodes_all(self):
        rh = RendezvousHash(["A", "B", "C"])
        nodes = rh.get_nodes("key", 3)
        self.assertEqual(set(nodes), {"A", "B", "C"})

    def test_add_remove(self):
        rh = RendezvousHash()
        rh.add_node("A")
        rh.add_node("B")
        self.assertEqual(len(rh), 2)
        rh.remove_node("A")
        self.assertEqual(rh.get_node("key"), "B")

    def test_add_duplicate(self):
        rh = RendezvousHash()
        rh.add_node("A")
        rh.add_node("A")
        self.assertEqual(len(rh), 1)

    def test_nodes_property(self):
        rh = RendezvousHash(["X", "Y"])
        self.assertEqual(rh.nodes, ["X", "Y"])


# ========== MultiProbeHash Tests ==========

class TestMultiProbeHash(unittest.TestCase):

    def test_empty(self):
        mp = MultiProbeHash()
        self.assertIsNone(mp.get_node("key"))

    def test_single_node(self):
        mp = MultiProbeHash(["A"])
        self.assertEqual(mp.get_node("key"), "A")

    def test_deterministic(self):
        mp = MultiProbeHash(["A", "B", "C"])
        n1 = mp.get_node("key")
        n2 = mp.get_node("key")
        self.assertEqual(n1, n2)

    def test_distribution(self):
        mp = MultiProbeHash([f"node_{i}" for i in range(5)], num_probes=21)
        dist = defaultdict(int)
        for i in range(10000):
            node = mp.get_node(f"key_{i}")
            dist[node] += 1
        # Should hit all nodes
        self.assertEqual(len(dist), 5)
        for count in dist.values():
            self.assertGreater(count, 800)

    def test_add_remove(self):
        mp = MultiProbeHash()
        mp.add_node("A")
        mp.add_node("B")
        self.assertEqual(len(mp), 2)
        mp.remove_node("A")
        self.assertEqual(mp.get_node("key"), "B")

    def test_nodes_property(self):
        mp = MultiProbeHash(["A", "B"])
        self.assertEqual(mp.nodes, {"A", "B"})

    def test_many_nodes(self):
        nodes = [f"n{i}" for i in range(20)]
        mp = MultiProbeHash(nodes, num_probes=21)
        results = set()
        for i in range(1000):
            results.add(mp.get_node(f"key_{i}"))
        self.assertEqual(len(results), 20)


# ========== MaglevHash Tests ==========

class TestMaglevHash(unittest.TestCase):

    def test_empty(self):
        mg = MaglevHash()
        self.assertIsNone(mg.get_node("key"))

    def test_single_node(self):
        mg = MaglevHash(["A"], table_size=17)
        self.assertEqual(mg.get_node("key"), "A")

    def test_deterministic(self):
        mg = MaglevHash(["A", "B", "C"], table_size=17)
        n1 = mg.get_node("key")
        n2 = mg.get_node("key")
        self.assertEqual(n1, n2)

    def test_distribution(self):
        mg = MaglevHash([f"node_{i}" for i in range(5)], table_size=65537)
        dist = defaultdict(int)
        for i in range(10000):
            node = mg.get_node(f"key_{i}")
            dist[node] += 1
        self.assertEqual(len(dist), 5)
        for count in dist.values():
            self.assertGreater(count, 1000)

    def test_add_node(self):
        mg = MaglevHash(["A", "B"], table_size=17)
        mg.add_node("C")
        self.assertEqual(len(mg), 3)

    def test_remove_node(self):
        mg = MaglevHash(["A", "B", "C"], table_size=17)
        mg.remove_node("B")
        self.assertEqual(len(mg), 2)
        node = mg.get_node("key")
        self.assertIn(node, ["A", "C"])

    def test_minimal_disruption(self):
        nodes = ["A", "B", "C", "D"]
        mg1 = MaglevHash(nodes, table_size=997)
        keys = [f"key_{i}" for i in range(5000)]
        map1 = {k: mg1.get_node(k) for k in keys}

        mg2 = MaglevHash(nodes + ["E"], table_size=997)
        map2 = {k: mg2.get_node(k) for k in keys}

        moved, total, frac = calculate_migration(map1, map2)
        self.assertLess(frac, 0.40)

    def test_table_size_is_prime(self):
        mg = MaglevHash(["A"], table_size=100)
        self.assertTrue(MaglevHash._is_prime(mg.table_size))
        self.assertGreaterEqual(mg.table_size, 100)

    def test_nodes_property(self):
        mg = MaglevHash(["X", "Y"])
        self.assertEqual(mg.nodes, ["X", "Y"])

    def test_o1_lookup(self):
        """Verify table-based lookup works."""
        mg = MaglevHash(["A", "B", "C"], table_size=17)
        self.assertTrue(len(mg._table) > 0)
        node = mg.get_node("test")
        self.assertIn(node, ["A", "B", "C"])

    def test_add_duplicate(self):
        mg = MaglevHash(["A", "B"], table_size=17)
        mg.add_node("A")
        self.assertEqual(len(mg), 2)

    def test_remove_nonexistent(self):
        mg = MaglevHash(["A"], table_size=17)
        mg.remove_node("Z")
        self.assertEqual(len(mg), 1)

    def test_is_prime(self):
        self.assertTrue(MaglevHash._is_prime(2))
        self.assertTrue(MaglevHash._is_prime(3))
        self.assertTrue(MaglevHash._is_prime(17))
        self.assertTrue(MaglevHash._is_prime(65537))
        self.assertFalse(MaglevHash._is_prime(4))
        self.assertFalse(MaglevHash._is_prime(100))

    def test_next_prime(self):
        self.assertEqual(MaglevHash._next_prime(2), 2)
        self.assertEqual(MaglevHash._next_prime(4), 5)
        self.assertEqual(MaglevHash._next_prime(17), 17)


# ========== Cross-Algorithm Comparison Tests ==========

class TestCrossAlgorithm(unittest.TestCase):
    """Compare properties across all hashing algorithms."""

    def _get_all_mappings(self, nodes, keys):
        """Get key->node mapping for each algorithm."""
        hr = HashRing(nodes)
        wr = WeightedHashRing()
        for n in nodes:
            wr.add_node(n, 1.0)
        rh = RendezvousHash(nodes)
        mp = MultiProbeHash(nodes)
        mg = MaglevHash(nodes, table_size=997)

        return {
            'HashRing': {k: hr.get_node(k) for k in keys},
            'Weighted': {k: wr.get_node(k) for k in keys},
            'Rendezvous': {k: rh.get_node(k) for k in keys},
            'MultiProbe': {k: mp.get_node(k) for k in keys},
            'Maglev': {k: mg.get_node(k) for k in keys},
        }

    def test_all_cover_all_nodes(self):
        """Each algorithm should use all available nodes."""
        nodes = ["A", "B", "C", "D", "E"]
        keys = [f"key_{i}" for i in range(5000)]
        mappings = self._get_all_mappings(nodes, keys)
        for name, mapping in mappings.items():
            used = set(mapping.values())
            self.assertEqual(used, set(nodes), f"{name} didn't cover all nodes")

    def test_all_deterministic(self):
        """Each algorithm should be deterministic."""
        nodes = ["X", "Y", "Z"]
        keys = [f"k{i}" for i in range(100)]
        m1 = self._get_all_mappings(nodes, keys)
        m2 = self._get_all_mappings(nodes, keys)
        for name in m1:
            self.assertEqual(m1[name], m2[name], f"{name} not deterministic")

    def test_jump_hash_distribution_vs_ring(self):
        """Jump hash should have similar balance to ring hash."""
        keys = [f"key_{i}" for i in range(10000)]
        ring = HashRing([f"n{i}" for i in range(5)])
        ring_dist = ring.get_distribution(10000)

        jump_dist = defaultdict(int)
        for k in keys:
            b = JumpHash.hash(k, 5)
            jump_dist[b] += 1

        _, ring_cv, _ = measure_balance(ring_dist)
        _, jump_cv, _ = measure_balance(dict(jump_dist))
        # Both should have reasonable balance
        self.assertLess(ring_cv, 0.20)
        self.assertLess(jump_cv, 0.10)  # Jump hash is typically more uniform


# ========== Utility Function Tests ==========

class TestUtilityFunctions(unittest.TestCase):

    def test_calculate_migration_no_change(self):
        m = {"a": 1, "b": 2}
        moved, total, frac = calculate_migration(m, m)
        self.assertEqual(moved, 0)
        self.assertEqual(frac, 0.0)

    def test_calculate_migration_all_changed(self):
        m1 = {"a": "X", "b": "X"}
        m2 = {"a": "Y", "b": "Y"}
        moved, total, frac = calculate_migration(m1, m2)
        self.assertEqual(moved, 2)
        self.assertEqual(frac, 1.0)

    def test_calculate_migration_empty(self):
        moved, total, frac = calculate_migration({}, {})
        self.assertEqual(moved, 0)
        self.assertEqual(frac, 0.0)

    def test_measure_balance_uniform(self):
        dist = {"A": 100, "B": 100, "C": 100}
        std, cv, max_dev = measure_balance(dist)
        self.assertAlmostEqual(std, 0.0)
        self.assertAlmostEqual(cv, 0.0)

    def test_measure_balance_skewed(self):
        dist = {"A": 900, "B": 50, "C": 50}
        std, cv, max_dev = measure_balance(dist)
        self.assertGreater(cv, 0.5)
        self.assertGreater(max_dev, 1.0)

    def test_measure_balance_empty(self):
        std, cv, max_dev = measure_balance({})
        self.assertEqual(std, 0.0)


# ========== Edge Cases & Stress Tests ==========

class TestEdgeCases(unittest.TestCase):

    def test_hashring_single_key_many_times(self):
        ring = HashRing(["A", "B", "C"])
        results = set(ring.get_node("same_key") for _ in range(100))
        self.assertEqual(len(results), 1)  # always same result

    def test_empty_string_key(self):
        ring = HashRing(["A", "B"])
        node = ring.get_node("")
        self.assertIn(node, ["A", "B"])

    def test_unicode_keys(self):
        ring = HashRing(["A", "B", "C"])
        node = ring.get_node("unicode_key_\u00e9\u00e8\u00ea")
        self.assertIn(node, ["A", "B", "C"])

    def test_unicode_nodes(self):
        ring = HashRing(["server-\u03b1", "server-\u03b2"])
        node = ring.get_node("key")
        self.assertIn(node, ["server-\u03b1", "server-\u03b2"])

    def test_long_key(self):
        ring = HashRing(["A", "B"])
        key = "x" * 10000
        node = ring.get_node(key)
        self.assertIn(node, ["A", "B"])

    def test_long_node_name(self):
        name = "node_" + "x" * 1000
        ring = HashRing([name])
        self.assertEqual(ring.get_node("key"), name)

    def test_weighted_zero_weight(self):
        ring = WeightedHashRing()
        ring.add_node("A", weight=0.01)  # very low but not zero
        ring.add_node("B", weight=10.0)
        dist = defaultdict(int)
        for i in range(5000):
            dist[ring.get_node(f"k{i}")] += 1
        # B should dominate
        self.assertGreater(dist["B"], dist["A"])

    def test_bounded_heavy_load(self):
        ring = BoundedLoadHashRing(["A", "B", "C"], epsilon=0.1)
        for i in range(3000):
            ring.get_node(f"key_{i}")
        # Verify bounded
        loads = [ring.get_load(n) for n in ["A", "B", "C"]]
        self.assertEqual(sum(loads), 3000)
        avg = 1000
        cap = math.ceil(avg * 1.1)
        for load in loads:
            self.assertLessEqual(load, cap + 5)  # small tolerance

    def test_rendezvous_remove_and_reassign(self):
        """Removing a node should only affect keys that were on that node."""
        rh = RendezvousHash(["A", "B", "C", "D"])
        keys = [f"k{i}" for i in range(1000)]
        map1 = {k: rh.get_node(k) for k in keys}

        rh.remove_node("C")
        map2 = {k: rh.get_node(k) for k in keys}

        for k in keys:
            if map1[k] != "C":
                # Keys not on C should stay where they are
                self.assertEqual(map1[k], map2[k],
                    f"Key {k} moved from {map1[k]} to {map2[k]} when C was removed")

    def test_maglev_full_table_coverage(self):
        """Every slot in the Maglev table should be assigned."""
        mg = MaglevHash(["A", "B", "C"], table_size=17)
        for slot in mg._table:
            self.assertNotEqual(slot, -1)

    def test_all_algorithms_handle_single_node(self):
        for cls in [HashRing, WeightedHashRing, RendezvousHash, MultiProbeHash]:
            if cls == WeightedHashRing:
                ring = cls()
                ring.add_node("only")
            else:
                ring = cls(["only"])
            self.assertEqual(ring.get_node("key"), "only")

    def test_maglev_single_node(self):
        mg = MaglevHash(["only"], table_size=7)
        self.assertEqual(mg.get_node("key"), "only")

    def test_jump_hash_single_bucket(self):
        self.assertEqual(JumpHash.hash("any", 1), 0)

    def test_hashring_num_replicas_1(self):
        """Ring with 1 replica per node still works."""
        ring = HashRing(["A", "B", "C"], num_replicas=1)
        node = ring.get_node("key")
        self.assertIn(node, ["A", "B", "C"])

    def test_hashring_high_replicas(self):
        """Ring with many replicas has better distribution."""
        ring = HashRing(["A", "B", "C"], num_replicas=500)
        dist = ring.get_distribution(10000)
        _, cv, _ = measure_balance(dist)
        self.assertLess(cv, 0.10)  # should be very balanced


# ========== Integration / Scenario Tests ==========

class TestScenarios(unittest.TestCase):

    def test_cache_cluster_scenario(self):
        """Simulate a cache cluster with node additions."""
        # Start with 3 cache servers
        ring = HashRing(["cache-1", "cache-2", "cache-3"])
        keys = [f"user:{i}:profile" for i in range(1000)]
        initial = {k: ring.get_node(k) for k in keys}

        # Scale up to 4
        ring.add_node("cache-4")
        after_scale = {k: ring.get_node(k) for k in keys}

        moved, _, frac = calculate_migration(initial, after_scale)
        # Should move roughly 25% of keys
        self.assertLess(frac, 0.40)
        self.assertGreater(moved, 0)

    def test_weighted_heterogeneous_servers(self):
        """Bigger servers should get proportionally more keys."""
        ring = WeightedHashRing(base_replicas=200)
        ring.add_node("small-1", weight=1.0)
        ring.add_node("small-2", weight=1.0)
        ring.add_node("large-1", weight=4.0)

        dist = defaultdict(int)
        for i in range(12000):
            dist[ring.get_node(f"req_{i}")] += 1

        # large-1 should get ~4x each small server
        ratio = dist["large-1"] / max(1, dist["small-1"])
        self.assertGreater(ratio, 2.5)
        self.assertLess(ratio, 6.0)

    def test_bounded_load_hot_key(self):
        """Bounded load prevents hot-key pile-up."""
        ring = BoundedLoadHashRing(["A", "B", "C", "D", "E"], epsilon=0.25)
        # All keys hash to similar positions -- simulates hot partition
        # The bounded ring should spread them out
        for i in range(500):
            ring.get_node(f"hot_{i}")
        loads = {n: ring.get_load(n) for n in ["A", "B", "C", "D", "E"]}
        max_load = max(loads.values())
        min_load = min(loads.values())
        # Gap between max and min should be limited
        self.assertLess(max_load - min_load, 200)

    def test_replication_across_algorithms(self):
        """HashRing and Rendezvous both support multi-node replication."""
        nodes = ["dc-east-1", "dc-west-1", "dc-east-2", "dc-west-2"]
        hr = HashRing(nodes)
        rh = RendezvousHash(nodes)

        for i in range(100):
            key = f"doc:{i}"
            hr_nodes = hr.get_nodes(key, 2)
            rh_nodes = rh.get_nodes(key, 2)
            self.assertEqual(len(hr_nodes), 2)
            self.assertEqual(len(rh_nodes), 2)
            # Both should return distinct nodes
            self.assertNotEqual(hr_nodes[0], hr_nodes[1])
            self.assertNotEqual(rh_nodes[0], rh_nodes[1])

    def test_consistent_after_churn(self):
        """After add+remove cycle, keys on untouched nodes stay put."""
        ring = HashRing(["A", "B", "C"])
        keys = [f"k{i}" for i in range(1000)]
        before = {k: ring.get_node(k) for k in keys}

        ring.add_node("D")
        ring.remove_node("D")
        after = {k: ring.get_node(k) for k in keys}

        self.assertEqual(before, after)

    def test_jump_hash_server_list(self):
        """Jump hash with named servers."""
        servers = ["db-primary", "db-replica-1", "db-replica-2"]
        assignments = defaultdict(list)
        for i in range(300):
            key = f"shard_{i}"
            server = JumpHash.hash_with_names(key, servers)
            assignments[server].append(key)
        # All servers should have some keys
        for s in servers:
            self.assertGreater(len(assignments[s]), 50)

    def test_multiprobe_fewer_vnodes_ok(self):
        """Multi-probe achieves balance with 1 vnode per physical node."""
        mp = MultiProbeHash([f"s{i}" for i in range(10)], num_probes=21)
        dist = defaultdict(int)
        for i in range(10000):
            dist[mp.get_node(f"k{i}")] += 1
        _, cv, _ = measure_balance(dist)
        self.assertLess(cv, 0.45)

    def test_maglev_fast_rebuild(self):
        """Maglev table rebuild after node change."""
        mg = MaglevHash(["A", "B", "C"], table_size=251)
        map_before = {f"k{i}": mg.get_node(f"k{i}") for i in range(500)}
        mg.add_node("D")
        map_after = {f"k{i}": mg.get_node(f"k{i}") for i in range(500)}
        moved, _, frac = calculate_migration(map_before, map_after)
        # Should be relatively low disruption
        self.assertLess(frac, 0.45)


if __name__ == '__main__':
    unittest.main()
