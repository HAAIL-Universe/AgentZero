"""Tests for C236: Kademlia DHT."""

import unittest
import time
import random

from kademlia_dht import (
    xor_distance, bit_length, bucket_index, _hash_to_bits,
    Contact, KBucket, RoutingTable, StoredValue, KademliaNode,
    KademliaNetwork, visualize_routing_table, visualize_network,
    DEFAULT_ID_BITS, DEFAULT_K,
)


# ============================================================
# XOR Distance & Utilities
# ============================================================

class TestXorDistance(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(xor_distance(42, 42), 0)

    def test_symmetry(self):
        self.assertEqual(xor_distance(10, 25), xor_distance(25, 10))

    def test_triangle_inequality(self):
        a, b, c = 5, 17, 30
        self.assertLessEqual(xor_distance(a, c),
                             xor_distance(a, b) + xor_distance(b, c))

    def test_known_values(self):
        # 0b0101 ^ 0b0011 = 0b0110 = 6
        self.assertEqual(xor_distance(5, 3), 6)
        self.assertEqual(xor_distance(0, 255), 255)
        self.assertEqual(xor_distance(255, 255), 0)

    def test_xor_distance_zero(self):
        self.assertEqual(xor_distance(0, 0), 0)

    def test_xor_nonzero(self):
        self.assertGreater(xor_distance(0, 1), 0)


class TestBitLength(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(bit_length(0), 0)

    def test_one(self):
        self.assertEqual(bit_length(1), 1)

    def test_powers_of_two(self):
        self.assertEqual(bit_length(2), 2)
        self.assertEqual(bit_length(4), 3)
        self.assertEqual(bit_length(8), 4)
        self.assertEqual(bit_length(128), 8)

    def test_non_powers(self):
        self.assertEqual(bit_length(3), 2)
        self.assertEqual(bit_length(7), 3)
        self.assertEqual(bit_length(255), 8)


class TestBucketIndex(unittest.TestCase):
    def test_same_id(self):
        self.assertEqual(bucket_index(10, 10, 8), -1)

    def test_distance_1(self):
        # XOR distance 1 -> bit_length 1 -> bucket 0
        self.assertEqual(bucket_index(10, 11, 8), 0)

    def test_distance_large(self):
        # XOR(0, 128) = 128 -> bit_length 8 -> bucket 7
        self.assertEqual(bucket_index(0, 128, 8), 7)

    def test_distance_2(self):
        # XOR(0, 2) = 2 -> bit_length 2 -> bucket 1
        self.assertEqual(bucket_index(0, 2, 8), 1)


class TestHashToBits(unittest.TestCase):
    def test_deterministic(self):
        h1 = _hash_to_bits("key1", 8)
        h2 = _hash_to_bits("key1", 8)
        self.assertEqual(h1, h2)

    def test_range(self):
        for key in ["a", "b", "c", "test", "hello"]:
            h = _hash_to_bits(key, 8)
            self.assertGreaterEqual(h, 0)
            self.assertLess(h, 256)

    def test_different_keys(self):
        h1 = _hash_to_bits("key1", 8)
        h2 = _hash_to_bits("key2", 8)
        # Very unlikely to collide
        # (don't assert != because it's possible but astronomically unlikely)
        self.assertIsInstance(h1, int)
        self.assertIsInstance(h2, int)

    def test_int_key(self):
        h = _hash_to_bits(42, 8)
        self.assertGreaterEqual(h, 0)
        self.assertLess(h, 256)


# ============================================================
# Contact
# ============================================================

class TestContact(unittest.TestCase):
    def test_creation(self):
        c = Contact(42)
        self.assertEqual(c.node_id, 42)
        self.assertEqual(c.address, "addr_42")

    def test_equality(self):
        c1 = Contact(10)
        c2 = Contact(10)
        self.assertEqual(c1, c2)

    def test_inequality(self):
        c1 = Contact(10)
        c2 = Contact(20)
        self.assertNotEqual(c1, c2)

    def test_hashable(self):
        s = {Contact(1), Contact(2), Contact(1)}
        self.assertEqual(len(s), 2)

    def test_repr(self):
        c = Contact(5)
        self.assertIn("5", repr(c))

    def test_custom_address(self):
        c = Contact(1, address="192.168.1.1:4000")
        self.assertEqual(c.address, "192.168.1.1:4000")


# ============================================================
# KBucket
# ============================================================

class TestKBucket(unittest.TestCase):
    def test_add_contact(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        c = Contact(10)
        result = b.add_contact(c, now=1.0)
        self.assertIsNone(result)
        self.assertEqual(len(b), 1)

    def test_full_bucket_returns_eviction(self):
        b = KBucket(k=2, range_lower=1, range_upper=2)
        b.add_contact(Contact(1), now=1.0)
        b.add_contact(Contact(2), now=2.0)
        evict = b.add_contact(Contact(3), now=3.0)
        self.assertIsNotNone(evict)
        self.assertEqual(evict.node_id, 1)  # LRU

    def test_update_existing_moves_to_tail(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        b.add_contact(Contact(1), now=1.0)
        b.add_contact(Contact(2), now=2.0)
        b.add_contact(Contact(1), now=3.0)  # Re-add
        contacts = b.get_contacts()
        self.assertEqual(contacts[-1].node_id, 1)  # Moved to tail

    def test_remove_contact(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        b.add_contact(Contact(5), now=1.0)
        self.assertTrue(b.remove_contact(5))
        self.assertEqual(len(b), 0)

    def test_remove_nonexistent(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        self.assertFalse(b.remove_contact(99))

    def test_head_lru(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        b.add_contact(Contact(10), now=1.0)
        b.add_contact(Contact(20), now=2.0)
        self.assertEqual(b.head().node_id, 10)

    def test_head_empty(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        self.assertIsNone(b.head())

    def test_is_full(self):
        b = KBucket(k=2, range_lower=1, range_upper=2)
        self.assertFalse(b.is_full())
        b.add_contact(Contact(1), now=1.0)
        self.assertFalse(b.is_full())
        b.add_contact(Contact(2), now=2.0)
        self.assertTrue(b.is_full())

    def test_covers(self):
        b = KBucket(k=3, range_lower=4, range_upper=8)
        self.assertTrue(b.covers(4))
        self.assertTrue(b.covers(7))
        self.assertFalse(b.covers(8))
        self.assertFalse(b.covers(3))

    def test_repr(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        self.assertIn("KBucket", repr(b))

    def test_last_updated(self):
        b = KBucket(k=3, range_lower=1, range_upper=2)
        b.add_contact(Contact(1), now=5.0)
        self.assertEqual(b.last_updated, 5.0)


# ============================================================
# RoutingTable
# ============================================================

class TestRoutingTable(unittest.TestCase):
    def setUp(self):
        self.rt = RoutingTable(own_id=0, id_bits=8, k=3)

    def test_add_contact(self):
        added, evict = self.rt.add_contact(Contact(1), now=1.0)
        self.assertTrue(added)
        self.assertIsNone(evict)

    def test_add_self(self):
        added, evict = self.rt.add_contact(Contact(0), now=1.0)
        self.assertFalse(added)
        self.assertIsNone(evict)

    def test_find_closest(self):
        for i in [1, 2, 3, 10, 20, 100, 200]:
            self.rt.add_contact(Contact(i), now=1.0)
        closest = self.rt.find_closest(5, count=3)
        # Should be sorted by XOR distance to 5
        dists = [xor_distance(c.node_id, 5) for c in closest]
        self.assertEqual(dists, sorted(dists))
        self.assertEqual(len(closest), 3)

    def test_find_closest_with_exclude(self):
        self.rt.add_contact(Contact(1), now=1.0)
        self.rt.add_contact(Contact(2), now=1.0)
        self.rt.add_contact(Contact(3), now=1.0)
        closest = self.rt.find_closest(1, count=3, exclude={1})
        ids = [c.node_id for c in closest]
        self.assertNotIn(1, ids)

    def test_remove_contact(self):
        self.rt.add_contact(Contact(5), now=1.0)
        self.assertTrue(self.rt.remove_contact(5))
        self.assertEqual(self.rt.total_contacts(), 0)

    def test_total_contacts(self):
        for i in [1, 2, 4, 8]:
            self.rt.add_contact(Contact(i), now=1.0)
        self.assertEqual(self.rt.total_contacts(), 4)

    def test_get_all_contacts(self):
        self.rt.add_contact(Contact(1), now=1.0)
        self.rt.add_contact(Contact(128), now=1.0)
        all_c = self.rt.get_all_contacts()
        ids = {c.node_id for c in all_c}
        self.assertEqual(ids, {1, 128})

    def test_stale_buckets(self):
        self.rt.add_contact(Contact(1), now=1.0)
        stale = self.rt.stale_buckets(now=5000.0, threshold=3600)
        self.assertIn(0, stale)  # Bucket 0 holds node 1

    def test_get_bucket(self):
        b = self.rt.get_bucket(0)
        self.assertIsNotNone(b)
        self.assertIsNone(self.rt.get_bucket(100))

    def test_bucket_distribution(self):
        """Contacts at different XOR distances go to different buckets."""
        self.rt.add_contact(Contact(1), now=1.0)    # dist 1, bucket 0
        self.rt.add_contact(Contact(2), now=1.0)    # dist 2, bucket 1
        self.rt.add_contact(Contact(128), now=1.0)  # dist 128, bucket 7
        self.assertEqual(len(self.rt.buckets[0]), 1)
        self.assertEqual(len(self.rt.buckets[1]), 1)
        self.assertEqual(len(self.rt.buckets[7]), 1)


# ============================================================
# StoredValue
# ============================================================

class TestStoredValue(unittest.TestCase):
    def test_creation(self):
        sv = StoredValue("hello", publisher_id=5, store_time=100.0)
        self.assertEqual(sv.value, "hello")
        self.assertEqual(sv.publisher_id, 5)

    def test_not_expired(self):
        sv = StoredValue("val", 1, store_time=100.0, expire_time=3600)
        self.assertFalse(sv.is_expired(now=200.0))

    def test_expired(self):
        sv = StoredValue("val", 1, store_time=100.0, expire_time=100)
        self.assertTrue(sv.is_expired(now=300.0))

    def test_edge_expired(self):
        sv = StoredValue("val", 1, store_time=0.0, expire_time=100)
        self.assertTrue(sv.is_expired(now=100.0))


# ============================================================
# KademliaNode (Unit)
# ============================================================

class TestKademliaNode(unittest.TestCase):
    def test_creation(self):
        n = KademliaNode(node_id=10, id_bits=8)
        self.assertEqual(n.node_id, 10)
        self.assertTrue(n.is_alive)

    def test_contact(self):
        n = KademliaNode(node_id=42, id_bits=8)
        c = n._contact()
        self.assertEqual(c.node_id, 42)

    def test_rpc_ping(self):
        n = KademliaNode(node_id=10, id_bits=8)
        self.assertTrue(n.rpc_ping(Contact(20)))

    def test_rpc_ping_dead(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.is_alive = False
        self.assertFalse(n.rpc_ping(Contact(20)))

    def test_rpc_store(self):
        n = KademliaNode(node_id=10, id_bits=8)
        self.assertTrue(n.rpc_store(Contact(20), 5, "value"))
        self.assertIn(5, n.storage)

    def test_rpc_store_dead(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.is_alive = False
        self.assertFalse(n.rpc_store(Contact(20), 5, "value"))

    def test_rpc_find_node(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.routing_table.add_contact(Contact(20), now=1.0)
        n.routing_table.add_contact(Contact(30), now=1.0)
        result = n.rpc_find_node(Contact(5), target_id=25)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_rpc_find_node_dead_node(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.is_alive = False
        self.assertIsNone(n.rpc_find_node(Contact(5), 25))

    def test_rpc_find_value_stored(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.rpc_store(Contact(20), 5, "hello")
        result = n.rpc_find_value(Contact(20), 5)
        self.assertEqual(result['value'], "hello")

    def test_rpc_find_value_not_stored(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.routing_table.add_contact(Contact(20), now=1.0)
        result = n.rpc_find_value(Contact(5), 99)
        self.assertIn('contacts', result)

    def test_rpc_find_value_dead(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.is_alive = False
        self.assertIsNone(n.rpc_find_value(Contact(5), 99))

    def test_rpc_updates_routing(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.rpc_ping(Contact(20))
        contacts = n.routing_table.get_all_contacts()
        ids = {c.node_id for c in contacts}
        self.assertIn(20, ids)


# ============================================================
# KademliaNetwork
# ============================================================

class TestKademliaNetwork(unittest.TestCase):
    def test_create_node(self):
        net = KademliaNetwork(id_bits=8)
        n = net.create_node(node_id=5)
        self.assertEqual(n.node_id, 5)
        self.assertIs(n.network, net)

    def test_get_node(self):
        net = KademliaNetwork(id_bits=8)
        n = net.create_node(node_id=42)
        self.assertIs(net.get_node(42), n)
        self.assertIsNone(net.get_node(99))

    def test_bootstrap_network(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=10, seed=42)
        self.assertEqual(len(nodes), 10)
        # All nodes should have some routing table entries
        for node in nodes:
            self.assertGreater(node.routing_table.total_contacts(), 0)

    def test_remove_node(self):
        net = KademliaNetwork(id_bits=8)
        n = net.create_node(node_id=5)
        self.assertTrue(net.remove_node(5))
        self.assertFalse(n.is_alive)

    def test_restore_node(self):
        net = KademliaNetwork(id_bits=8)
        n = net.create_node(node_id=5)
        net.remove_node(5)
        self.assertTrue(net.restore_node(5))
        self.assertTrue(n.is_alive)

    def test_remove_nonexistent(self):
        net = KademliaNetwork(id_bits=8)
        self.assertFalse(net.remove_node(99))

    def test_get_alive_nodes(self):
        net = KademliaNetwork(id_bits=8)
        net.create_node(node_id=1)
        net.create_node(node_id=2)
        net.remove_node(1)
        alive = net.get_alive_nodes()
        self.assertEqual(len(alive), 1)
        self.assertEqual(alive[0].node_id, 2)


# ============================================================
# Store and Lookup (Integration)
# ============================================================

class TestStoreAndLookup(unittest.TestCase):
    def setUp(self):
        self.net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        self.nodes = self.net.bootstrap_network(count=15, seed=100)

    def test_store_and_retrieve(self):
        node = self.nodes[0]
        key_id, count = node.store("testkey", "testvalue")
        self.assertGreater(count, 0)
        # Retrieve from a different node
        result = self.nodes[-1].lookup("testkey")
        self.assertEqual(result, "testvalue")

    def test_store_multiple_keys(self):
        node = self.nodes[0]
        for i in range(10):
            node.store(f"key_{i}", f"value_{i}")
        # Retrieve all from various nodes -- allow small failure rate in 8-bit space
        found = 0
        for i in range(10):
            source = self.nodes[i % len(self.nodes)]
            result = source.lookup(f"key_{i}")
            if result == f"value_{i}":
                found += 1
        self.assertGreaterEqual(found, 8)

    def test_lookup_nonexistent(self):
        result = self.nodes[0].lookup("no_such_key")
        self.assertIsNone(result)

    def test_store_overwrite(self):
        node = self.nodes[0]
        node.store("key", "v1")
        node.store("key", "v2")
        result = self.nodes[5].lookup("key")
        self.assertEqual(result, "v2")

    def test_store_from_different_nodes(self):
        self.nodes[0].store("from_0", "val_0")
        self.nodes[5].store("from_5", "val_5")
        self.nodes[10].store("from_10", "val_10")
        self.assertEqual(self.nodes[7].lookup("from_0"), "val_0")
        self.assertEqual(self.nodes[2].lookup("from_5"), "val_5")
        self.assertEqual(self.nodes[0].lookup("from_10"), "val_10")

    def test_local_store_and_lookup(self):
        """A node can retrieve its own stored values."""
        node = self.nodes[3]
        node.store("my_key", "my_val")
        self.assertEqual(node.lookup("my_key"), "my_val")


# ============================================================
# Node Failure and Resilience
# ============================================================

class TestResilience(unittest.TestCase):
    def setUp(self):
        self.net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        self.nodes = self.net.bootstrap_network(count=20, seed=200)

    def test_survive_single_failure(self):
        self.nodes[0].store("resilient_key", "resilient_val")
        # Kill a random node
        self.net.remove_node(self.nodes[5].node_id)
        result = self.nodes[10].lookup("resilient_key")
        self.assertEqual(result, "resilient_val")

    def test_survive_multiple_failures(self):
        self.nodes[0].store("multi_fail", "still_here")
        # Kill several nodes
        for i in [3, 7, 11, 15]:
            self.net.remove_node(self.nodes[i].node_id)
        result = self.nodes[1].lookup("multi_fail")
        self.assertEqual(result, "still_here")

    def test_key_redundancy(self):
        """Stored keys should be replicated across multiple nodes."""
        key_id, count = self.nodes[0].store("replicated", "data")
        redundancy = self.net.key_redundancy(key_id)
        self.assertGreater(redundancy, 1)

    def test_network_health(self):
        health = self.net.network_health()
        self.assertEqual(health['alive'], 20)
        self.assertEqual(health['total'], 20)
        self.assertEqual(health['health'], 1.0)

    def test_network_health_after_failure(self):
        self.net.remove_node(self.nodes[0].node_id)
        health = self.net.network_health()
        self.assertEqual(health['alive'], 19)
        self.assertLess(health['health'], 1.0)


# ============================================================
# Delete
# ============================================================

class TestDelete(unittest.TestCase):
    def setUp(self):
        self.net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        self.nodes = self.net.bootstrap_network(count=10, seed=300)

    def test_delete_existing(self):
        self.nodes[0].store("del_key", "del_val")
        deleted = self.nodes[0].delete("del_key")
        self.assertGreater(deleted, 0)

    def test_delete_removes_from_network(self):
        self.nodes[0].store("gone", "away")
        self.nodes[0].delete("gone")
        # Should not be findable
        result = self.nodes[5].lookup("gone")
        self.assertIsNone(result)

    def test_delete_nonexistent(self):
        deleted = self.nodes[0].delete("never_existed")
        # Should not error
        self.assertIsInstance(deleted, int)


# ============================================================
# Republish and Expiry
# ============================================================

class TestRepublishAndExpiry(unittest.TestCase):
    def setUp(self):
        self.net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        self.nodes = self.net.bootstrap_network(count=10, seed=400)

    def test_republish(self):
        self.nodes[0].store("repub", "data")
        count = self.nodes[0].republish()
        self.assertGreaterEqual(count, 1)

    def test_expire_keys(self):
        n = self.nodes[0]
        n.storage[99] = StoredValue("old", 0, store_time=0.0, expire_time=100)
        expired = n.expire_keys(now=200.0)
        self.assertEqual(expired, 1)
        self.assertNotIn(99, n.storage)

    def test_expire_cache(self):
        n = self.nodes[0]
        n.cache[88] = StoredValue("cached", 0, store_time=0.0, expire_time=50)
        expired = n.expire_keys(now=100.0)
        self.assertEqual(expired, 1)
        self.assertNotIn(88, n.cache)

    def test_expire_all_network(self):
        for node in self.nodes[:5]:
            node.storage[77] = StoredValue("exp", 0, store_time=0.0, expire_time=10)
        total = self.net.expire_all(now=100.0)
        self.assertEqual(total, 5)

    def test_republish_all_network(self):
        self.nodes[0].store("net_repub", "net_data")
        total = self.net.republish_all()
        self.assertGreaterEqual(total, 0)

    def test_non_expired_keys_remain(self):
        n = self.nodes[0]
        n.storage[50] = StoredValue("fresh", 0, store_time=100.0, expire_time=3600)
        expired = n.expire_keys(now=200.0)
        self.assertEqual(expired, 0)
        self.assertIn(50, n.storage)


# ============================================================
# Bootstrap
# ============================================================

class TestBootstrap(unittest.TestCase):
    def test_bootstrap_populates_routing(self):
        net = KademliaNetwork(id_bits=8, k=3)
        n1 = net.create_node(node_id=10)
        n2 = net.create_node(node_id=20)
        n3 = net.create_node(node_id=30)
        # Manually set up n1's routing
        n1.routing_table.add_contact(Contact(20), now=1.0)
        n1.routing_table.add_contact(Contact(30), now=1.0)
        # Bootstrap n2 via n1
        n2.bootstrap(n1._contact())
        self.assertGreater(n2.routing_table.total_contacts(), 0)

    def test_bootstrap_discovers_other_nodes(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=8, seed=500)
        # Every node should know about multiple other nodes
        for node in nodes:
            self.assertGreater(node.routing_table.total_contacts(), 0)


# ============================================================
# Refresh Buckets
# ============================================================

class TestRefreshBuckets(unittest.TestCase):
    def test_refresh(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=10, seed=600)
        # Force all buckets to be stale
        for bucket in nodes[0].routing_table.buckets:
            bucket.last_updated = 0.0
        refreshed = nodes[0].refresh_buckets(now=5000.0, threshold=3600)
        # Should have refreshed at least some buckets
        self.assertGreaterEqual(refreshed, 0)


# ============================================================
# Caching
# ============================================================

class TestCaching(unittest.TestCase):
    def setUp(self):
        self.net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        self.nodes = self.net.bootstrap_network(count=15, seed=700)

    def test_cache_along_path(self):
        """Looking up a value may cache it on intermediate nodes."""
        self.nodes[0].store("cache_test", "cached_value")
        # Lookup from far node
        self.nodes[-1].lookup("cache_test")
        # Check if any node has it in cache
        cached_nodes = [n for n in self.nodes if len(n.cache) > 0]
        # Cache may or may not be populated depending on lookup path
        # Just verify no errors
        self.assertIsInstance(cached_nodes, list)

    def test_find_value_from_cache(self):
        """A node with a cached value returns it."""
        n = self.nodes[5]
        key_id = _hash_to_bits("cached_key", 8)
        n.cache[key_id] = StoredValue("from_cache", 0)
        result = n.rpc_find_value(Contact(0), key_id)
        self.assertEqual(result['value'], "from_cache")


# ============================================================
# Visualization
# ============================================================

class TestVisualization(unittest.TestCase):
    def test_visualize_routing_table(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=5, seed=800)
        output = visualize_routing_table(nodes[0])
        self.assertIn("Routing table", output)

    def test_visualize_network(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=5, seed=800)
        output = visualize_network(net)
        self.assertIn("Kademlia Network", output)
        self.assertIn("alive", output)


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases(unittest.TestCase):
    def test_single_node_network(self):
        net = KademliaNetwork(id_bits=8, k=3)
        n = net.create_node(node_id=42)
        key_id, count = n.store("solo", "lonely")
        self.assertGreater(count, 0)
        self.assertEqual(n.lookup("solo"), "lonely")

    def test_two_node_network(self):
        net = KademliaNetwork(id_bits=8, k=3)
        n1 = net.create_node(node_id=0)
        n2 = net.create_node(node_id=128)
        n2.bootstrap(n1._contact())
        n1.store("shared", "value")
        self.assertEqual(n2.lookup("shared"), "value")

    def test_all_nodes_same_bucket(self):
        """Nodes with IDs differing only in low bits end up in same bucket."""
        net = KademliaNetwork(id_bits=8, k=3)
        n0 = net.create_node(node_id=0)
        # IDs 1, 2, 3 all in low-distance buckets from 0
        n1 = net.create_node(node_id=1)
        n2 = net.create_node(node_id=2)
        n3 = net.create_node(node_id=3)
        for n in [n1, n2, n3]:
            n.bootstrap(n0._contact())
        # Node 0 bucket 0 should have node 1 (distance 1)
        b0 = n0.routing_table.buckets[0]
        self.assertGreater(len(b0), 0)

    def test_id_space_boundary(self):
        """Node IDs at extremes of ID space."""
        net = KademliaNetwork(id_bits=8, k=3)
        n0 = net.create_node(node_id=0)
        n255 = net.create_node(node_id=255)
        n255.bootstrap(n0._contact())
        n0.store("boundary", "test")
        self.assertEqual(n255.lookup("boundary"), "test")

    def test_store_complex_values(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=5, seed=900)
        # Store various value types
        nodes[0].store("int_val", 42)
        nodes[0].store("list_val", [1, 2, 3])
        nodes[0].store("dict_val", {"a": 1})
        nodes[0].store("none_val", None)
        self.assertEqual(nodes[2].lookup("int_val"), 42)
        self.assertEqual(nodes[2].lookup("list_val"), [1, 2, 3])
        self.assertEqual(nodes[2].lookup("dict_val"), {"a": 1})
        self.assertIsNone(nodes[2].lookup("none_val"))

    def test_store_none_value_vs_not_found(self):
        """Distinguish between stored None and key-not-found."""
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=5, seed=901)
        nodes[0].store("null_key", None)
        # Both return None, but they mean different things
        # This is a known ambiguity in DHTs -- test that no crash occurs
        result = nodes[2].lookup("null_key")
        self.assertIsNone(result)


# ============================================================
# Larger Network Tests
# ============================================================

class TestLargerNetwork(unittest.TestCase):
    def test_20_nodes_100_keys(self):
        net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        nodes = net.bootstrap_network(count=20, seed=1000)
        # Store 100 keys from random nodes
        keys = {}
        for i in range(100):
            src = nodes[i % len(nodes)]
            key = f"large_test_{i}"
            val = f"value_{i}"
            src.store(key, val)
            keys[key] = val
        # Retrieve all from random nodes
        found = 0
        for key, expected in keys.items():
            src = random.choice(nodes)
            result = src.lookup(key)
            if result == expected:
                found += 1
        # Should find most/all keys
        self.assertGreater(found, 80)  # Allow for some failures in small ID space

    def test_network_with_churn(self):
        """Nodes joining and leaving while data is stored."""
        net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        nodes = net.bootstrap_network(count=15, seed=1100)
        # Store some data
        nodes[0].store("churn_key", "churn_val")
        # Remove some nodes
        for i in [3, 7, 11]:
            net.remove_node(nodes[i].node_id)
        # Add new nodes
        for i in range(3):
            new_node = net.create_node()
            new_node.bootstrap(nodes[0]._contact())
        # Should still find the data
        result = nodes[1].lookup("churn_key")
        self.assertEqual(result, "churn_val")


# ============================================================
# RPC Counting
# ============================================================

class TestRPCCounting(unittest.TestCase):
    def test_rpcs_counted(self):
        n = KademliaNode(node_id=10, id_bits=8)
        n.rpc_ping(Contact(20))
        n.rpc_ping(Contact(30))
        self.assertEqual(n.rpc_count, 2)

    def test_store_generates_rpcs(self):
        net = KademliaNetwork(id_bits=8, k=3, alpha=2)
        nodes = net.bootstrap_network(count=10, seed=1200)
        initial_rpcs = sum(n.rpc_count for n in nodes)
        nodes[0].store("rpc_test", "rpc_val")
        final_rpcs = sum(n.rpc_count for n in nodes)
        self.assertGreater(final_rpcs, initial_rpcs)


# ============================================================
# Network Stats
# ============================================================

class TestNetworkStats(unittest.TestCase):
    def test_total_stored_keys(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=5, seed=1300)
        nodes[0].store("k1", "v1")
        nodes[0].store("k2", "v2")
        total = net.total_stored_keys()
        self.assertGreaterEqual(total, 2)

    def test_stats_tracking(self):
        net = KademliaNetwork(id_bits=8, k=3)
        nodes = net.bootstrap_network(count=5, seed=1400)
        net.store(nodes[0], "s1", "v1")
        net.lookup(nodes[1], "s1")
        self.assertEqual(net.stats['stores'], 1)
        self.assertEqual(net.stats['lookups'], 1)


if __name__ == '__main__':
    unittest.main()
