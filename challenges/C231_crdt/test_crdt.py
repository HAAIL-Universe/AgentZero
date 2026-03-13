"""Tests for C231: Conflict-free Replicated Data Types"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import unittest
from crdt import (
    GCounter, PNCounter, LWWRegister, MVRegister,
    GSet, TwoPSet, ORSet, LWWElementSet,
    EWFlag, DWFlag,
    RGASequence, ORMap, VectorClock, CRDTNetwork,
)


# =============================================================================
# GCounter
# =============================================================================

class TestGCounter(unittest.TestCase):
    def test_basic_increment(self):
        c = GCounter("A")
        self.assertEqual(c.value, 0)
        c.increment()
        self.assertEqual(c.value, 1)
        c.increment(5)
        self.assertEqual(c.value, 6)

    def test_negative_increment_rejected(self):
        c = GCounter("A")
        with self.assertRaises(ValueError):
            c.increment(-1)

    def test_merge_two_replicas(self):
        a = GCounter("A")
        b = GCounter("B")
        a.increment(3)
        b.increment(5)
        merged = a.merge(b)
        self.assertEqual(merged.value, 8)

    def test_merge_commutativity(self):
        a = GCounter("A"); a.increment(3)
        b = GCounter("B"); b.increment(5)
        self.assertEqual(a.merge(b).value, b.merge(a).value)

    def test_merge_idempotency(self):
        a = GCounter("A"); a.increment(3)
        self.assertEqual(a.merge(a).value, a.value)

    def test_merge_associativity(self):
        a = GCounter("A"); a.increment(2)
        b = GCounter("B"); b.increment(3)
        c = GCounter("C"); c.increment(4)
        ab_c = a.merge(b).merge(c)
        a_bc = a.merge(b.merge(c))
        self.assertEqual(ab_c.value, a_bc.value)
        self.assertEqual(ab_c.value, 9)

    def test_merge_with_overlapping_updates(self):
        a = GCounter("A"); a.increment(3)
        b = GCounter("A"); b.increment(5)  # same replica ID
        merged = a.merge(b)
        self.assertEqual(merged.value, 5)  # max, not sum

    def test_three_replicas(self):
        a = GCounter("A"); a.increment(1)
        b = GCounter("B"); b.increment(2)
        c = GCounter("C"); c.increment(3)
        merged = a.merge(b).merge(c)
        self.assertEqual(merged.value, 6)

    def test_repr(self):
        c = GCounter("A"); c.increment(42)
        self.assertEqual(repr(c), "GCounter(42)")


# =============================================================================
# PNCounter
# =============================================================================

class TestPNCounter(unittest.TestCase):
    def test_increment_and_decrement(self):
        c = PNCounter("A")
        c.increment(5)
        c.decrement(3)
        self.assertEqual(c.value, 2)

    def test_negative_value(self):
        c = PNCounter("A")
        c.decrement(10)
        self.assertEqual(c.value, -10)

    def test_merge(self):
        a = PNCounter("A"); a.increment(5)
        b = PNCounter("B"); b.decrement(3)
        merged = a.merge(b)
        self.assertEqual(merged.value, 2)

    def test_merge_commutativity(self):
        a = PNCounter("A"); a.increment(5)
        b = PNCounter("B"); b.decrement(3)
        self.assertEqual(a.merge(b).value, b.merge(a).value)

    def test_merge_idempotency(self):
        a = PNCounter("A"); a.increment(5); a.decrement(2)
        self.assertEqual(a.merge(a).value, a.value)

    def test_merge_associativity(self):
        a = PNCounter("A"); a.increment(5)
        b = PNCounter("B"); b.decrement(3)
        c = PNCounter("C"); c.increment(1)
        self.assertEqual(a.merge(b).merge(c).value, a.merge(b.merge(c)).value)

    def test_concurrent_inc_dec(self):
        a = PNCounter("A"); a.increment(10)
        b = PNCounter("B"); b.decrement(5)
        merged = a.merge(b)
        self.assertEqual(merged.value, 5)

    def test_repr(self):
        c = PNCounter("A"); c.increment(7); c.decrement(2)
        self.assertEqual(repr(c), "PNCounter(5)")


# =============================================================================
# LWWRegister
# =============================================================================

class TestLWWRegister(unittest.TestCase):
    def test_basic_set_get(self):
        r = LWWRegister("A")
        r.set("hello", timestamp=1)
        self.assertEqual(r.value, "hello")

    def test_later_write_wins(self):
        r = LWWRegister("A")
        r.set("old", timestamp=1)
        r.set("new", timestamp=2)
        self.assertEqual(r.value, "new")

    def test_earlier_write_ignored(self):
        r = LWWRegister("A")
        r.set("new", timestamp=2)
        r.set("old", timestamp=1)
        self.assertEqual(r.value, "new")

    def test_merge_later_wins(self):
        a = LWWRegister("A"); a.set("a_val", timestamp=1)
        b = LWWRegister("B"); b.set("b_val", timestamp=2)
        merged = a.merge(b)
        self.assertEqual(merged.value, "b_val")

    def test_merge_commutativity(self):
        a = LWWRegister("A"); a.set("a_val", timestamp=1)
        b = LWWRegister("B"); b.set("b_val", timestamp=2)
        self.assertEqual(a.merge(b).value, b.merge(a).value)

    def test_merge_tie_break_by_replica(self):
        a = LWWRegister("A"); a.set("a_val", timestamp=1)
        b = LWWRegister("B"); b.set("b_val", timestamp=1)
        merged = a.merge(b)
        # Higher replica ID wins on tie
        self.assertEqual(merged.value, "b_val")

    def test_merge_idempotency(self):
        a = LWWRegister("A"); a.set("val", timestamp=1)
        merged = a.merge(a)
        self.assertEqual(merged.value, a.value)

    def test_none_initial(self):
        r = LWWRegister("A")
        self.assertIsNone(r.value)


# =============================================================================
# MVRegister
# =============================================================================

class TestMVRegister(unittest.TestCase):
    def test_basic_set(self):
        r = MVRegister("A")
        r.set("hello")
        self.assertEqual(r.value, "hello")

    def test_sequential_writes(self):
        r = MVRegister("A")
        r.set("first")
        r.set("second")
        self.assertEqual(r.value, "second")

    def test_concurrent_writes_produce_multiple(self):
        a = MVRegister("A"); a.set("a_val")
        b = MVRegister("B"); b.set("b_val")
        merged = a.merge(b)
        vals = sorted(merged.values)
        self.assertEqual(vals, ["a_val", "b_val"])

    def test_merge_commutativity(self):
        a = MVRegister("A"); a.set("a_val")
        b = MVRegister("B"); b.set("b_val")
        m1 = a.merge(b)
        m2 = b.merge(a)
        self.assertEqual(sorted(m1.values), sorted(m2.values))

    def test_resolve_conflict(self):
        a = MVRegister("A"); a.set("a_val")
        b = MVRegister("B"); b.set("b_val")
        merged = a.merge(b)
        # Resolve by setting new value
        merged.set("resolved")
        self.assertEqual(merged.value, "resolved")

    def test_empty_register(self):
        r = MVRegister("A")
        self.assertEqual(r.values, [])


# =============================================================================
# GSet
# =============================================================================

class TestGSet(unittest.TestCase):
    def test_add_and_contains(self):
        s = GSet("A")
        s.add(1)
        self.assertTrue(s.contains(1))
        self.assertFalse(s.contains(2))

    def test_len(self):
        s = GSet("A")
        s.add(1); s.add(2); s.add(3)
        self.assertEqual(len(s), 3)

    def test_duplicate_add(self):
        s = GSet("A")
        s.add(1); s.add(1)
        self.assertEqual(len(s), 1)

    def test_merge(self):
        a = GSet("A"); a.add(1); a.add(2)
        b = GSet("B"); b.add(2); b.add(3)
        merged = a.merge(b)
        self.assertEqual(merged.elements, frozenset({1, 2, 3}))

    def test_merge_commutativity(self):
        a = GSet("A"); a.add(1)
        b = GSet("B"); b.add(2)
        self.assertEqual(a.merge(b).elements, b.merge(a).elements)

    def test_merge_idempotency(self):
        a = GSet("A"); a.add(1); a.add(2)
        self.assertEqual(a.merge(a).elements, a.elements)

    def test_merge_associativity(self):
        a = GSet("A"); a.add(1)
        b = GSet("B"); b.add(2)
        c = GSet("C"); c.add(3)
        self.assertEqual(a.merge(b).merge(c).elements, a.merge(b.merge(c)).elements)


# =============================================================================
# TwoPSet
# =============================================================================

class TestTwoPSet(unittest.TestCase):
    def test_add_remove(self):
        s = TwoPSet("A")
        s.add(1)
        self.assertTrue(s.contains(1))
        s.remove(1)
        self.assertFalse(s.contains(1))

    def test_remove_is_permanent(self):
        s = TwoPSet("A")
        s.add(1)
        s.remove(1)
        s.add(1)  # re-add after remove
        self.assertFalse(s.contains(1))  # tombstoned permanently

    def test_remove_unknown_element(self):
        s = TwoPSet("A")
        s.remove(1)  # not in added set
        s.add(1)
        self.assertTrue(s.contains(1))  # only tombstoned if was in added

    def test_merge(self):
        a = TwoPSet("A"); a.add(1); a.add(2)
        b = TwoPSet("B"); b.add(2); b.add(3); b.remove(2)
        merged = a.merge(b)
        self.assertTrue(merged.contains(1))
        self.assertFalse(merged.contains(2))  # removed in b
        self.assertTrue(merged.contains(3))

    def test_len(self):
        s = TwoPSet("A")
        s.add(1); s.add(2); s.add(3); s.remove(2)
        self.assertEqual(len(s), 2)

    def test_merge_commutativity(self):
        a = TwoPSet("A"); a.add(1); a.add(2)
        b = TwoPSet("B"); b.add(2); b.remove(2)
        self.assertEqual(a.merge(b).elements, b.merge(a).elements)


# =============================================================================
# ORSet
# =============================================================================

class TestORSet(unittest.TestCase):
    def test_add_contains(self):
        s = ORSet("A")
        s.add(1)
        self.assertTrue(s.contains(1))

    def test_remove(self):
        s = ORSet("A")
        s.add(1)
        s.remove(1)
        self.assertFalse(s.contains(1))

    def test_add_remove_add(self):
        s = ORSet("A")
        s.add(1)
        s.remove(1)
        s.add(1)  # new unique tag
        self.assertTrue(s.contains(1))  # unlike TwoPSet, this works

    def test_concurrent_add_remove(self):
        """Concurrent add on A and remove on B -- add wins."""
        a = ORSet("A"); a.add("x")
        b = ORSet("B")
        # Sync a -> b
        b = b.merge(a)
        # Concurrent: a adds again, b removes
        a.add("x")
        b.remove("x")
        merged = a.merge(b)
        self.assertTrue(merged.contains("x"))  # new tag from a survives

    def test_merge_commutativity(self):
        a = ORSet("A"); a.add(1); a.add(2)
        b = ORSet("B"); b.add(2); b.add(3)
        m1 = a.merge(b)
        m2 = b.merge(a)
        self.assertEqual(m1.elements, m2.elements)

    def test_len(self):
        s = ORSet("A")
        s.add(1); s.add(2); s.add(3); s.remove(2)
        self.assertEqual(len(s), 2)

    def test_elements(self):
        s = ORSet("A")
        s.add("a"); s.add("b"); s.add("c")
        self.assertEqual(s.elements, frozenset({"a", "b", "c"}))

    def test_merge_idempotency(self):
        a = ORSet("A"); a.add(1); a.add(2)
        self.assertEqual(a.merge(a).elements, a.elements)


# =============================================================================
# LWWElementSet
# =============================================================================

class TestLWWElementSet(unittest.TestCase):
    def test_add_contains(self):
        s = LWWElementSet("A")
        s.add("x", timestamp=1)
        self.assertTrue(s.contains("x"))

    def test_remove(self):
        s = LWWElementSet("A")
        s.add("x", timestamp=1)
        s.remove("x", timestamp=2)
        self.assertFalse(s.contains("x"))

    def test_add_after_remove(self):
        s = LWWElementSet("A")
        s.add("x", timestamp=1)
        s.remove("x", timestamp=2)
        s.add("x", timestamp=3)
        self.assertTrue(s.contains("x"))

    def test_add_bias_on_tie(self):
        s = LWWElementSet("A")
        s.add("x", timestamp=1)
        s.remove("x", timestamp=1)
        self.assertTrue(s.contains("x"))  # add bias

    def test_merge(self):
        a = LWWElementSet("A")
        a.add("x", timestamp=1)
        b = LWWElementSet("B")
        b.add("x", timestamp=1)
        b.remove("x", timestamp=2)
        merged = a.merge(b)
        self.assertFalse(merged.contains("x"))

    def test_merge_commutativity(self):
        a = LWWElementSet("A")
        a.add("x", timestamp=1)
        b = LWWElementSet("B")
        b.remove("x", timestamp=2)
        b.add("x", timestamp=1)
        self.assertEqual(a.merge(b).elements, b.merge(a).elements)

    def test_elements(self):
        s = LWWElementSet("A")
        s.add("a", timestamp=1)
        s.add("b", timestamp=2)
        s.add("c", timestamp=3)
        s.remove("b", timestamp=4)
        self.assertEqual(s.elements, frozenset({"a", "c"}))

    def test_len(self):
        s = LWWElementSet("A")
        s.add("a", timestamp=1)
        s.add("b", timestamp=2)
        self.assertEqual(len(s), 2)


# =============================================================================
# EWFlag
# =============================================================================

class TestEWFlag(unittest.TestCase):
    def test_initial_disabled(self):
        f = EWFlag("A")
        self.assertFalse(f.value)

    def test_enable(self):
        f = EWFlag("A")
        f.enable()
        self.assertTrue(f.value)

    def test_disable(self):
        f = EWFlag("A")
        f.enable()
        f.disable()
        self.assertFalse(f.value)

    def test_concurrent_enable_wins(self):
        a = EWFlag("A"); a.enable()
        b = EWFlag("B"); b.enable(); b.disable()
        merged = a.merge(b)
        self.assertTrue(merged.value)  # enable wins

    def test_merge_commutativity(self):
        a = EWFlag("A"); a.enable()
        b = EWFlag("B"); b.disable()
        self.assertEqual(a.merge(b).value, b.merge(a).value)


# =============================================================================
# DWFlag
# =============================================================================

class TestDWFlag(unittest.TestCase):
    def test_initial_disabled(self):
        f = DWFlag("A")
        self.assertFalse(f.value)

    def test_enable_disable(self):
        f = DWFlag("A")
        f.enable()
        self.assertTrue(f.value)
        f.disable()
        self.assertFalse(f.value)

    def test_concurrent_disable_wins(self):
        a = DWFlag("A"); a.enable()
        b = DWFlag("B"); b.enable(); b.disable()
        merged = a.merge(b)
        self.assertFalse(merged.value)  # disable wins

    def test_merge_commutativity(self):
        a = DWFlag("A"); a.enable()
        b = DWFlag("B"); b.disable()
        self.assertEqual(a.merge(b).value, b.merge(a).value)


# =============================================================================
# RGASequence
# =============================================================================

class TestRGASequence(unittest.TestCase):
    def test_append(self):
        s = RGASequence("A")
        s.append("a"); s.append("b"); s.append("c")
        self.assertEqual(s.elements, ["a", "b", "c"])

    def test_insert_at_beginning(self):
        s = RGASequence("A")
        s.append("b"); s.append("c")
        s.insert(0, "a")
        self.assertEqual(s.elements, ["a", "b", "c"])

    def test_insert_in_middle(self):
        s = RGASequence("A")
        s.append("a"); s.append("c")
        s.insert(1, "b")
        self.assertEqual(s.elements, ["a", "b", "c"])

    def test_delete(self):
        s = RGASequence("A")
        s.append("a"); s.append("b"); s.append("c")
        s.delete(1)
        self.assertEqual(s.elements, ["a", "c"])

    def test_len(self):
        s = RGASequence("A")
        s.append("a"); s.append("b")
        self.assertEqual(len(s), 2)
        s.delete(0)
        self.assertEqual(len(s), 1)

    def test_getitem(self):
        s = RGASequence("A")
        s.append("x"); s.append("y")
        self.assertEqual(s[0], "x")
        self.assertEqual(s[1], "y")

    def test_merge_concurrent_inserts(self):
        a = RGASequence("A"); a.append("x")
        b = RGASequence("B"); b.append("y")
        merged = a.merge(b)
        # Both elements present
        self.assertEqual(len(merged), 2)
        elems = merged.elements
        self.assertIn("x", elems)
        self.assertIn("y", elems)

    def test_merge_preserves_order(self):
        a = RGASequence("A")
        a.append("a"); a.append("b"); a.append("c")
        b = RGASequence("B")
        # b has same structure (merge with self copy)
        merged = a.merge(a)
        self.assertEqual(merged.elements, ["a", "b", "c"])

    def test_merge_delete(self):
        a = RGASequence("A")
        a.append("a"); a.append("b")
        # Clone state to b
        b = a.merge(a)  # identical copy
        a.delete(0)  # a deletes "a"
        merged = a.merge(b)
        # Delete wins (tombstone)
        self.assertEqual(len(merged), 1)

    def test_index_error(self):
        s = RGASequence("A")
        with self.assertRaises(IndexError):
            s.insert(1, "x")  # can only insert at 0 when empty

    def test_delete_index_error(self):
        s = RGASequence("A")
        with self.assertRaises(IndexError):
            s.delete(0)


# =============================================================================
# ORMap
# =============================================================================

class TestORMap(unittest.TestCase):
    def test_put_get(self):
        m = ORMap("A")
        c = GCounter("A"); c.increment(5)
        m.put("key1", c)
        result = m.get("key1")
        self.assertEqual(result.value, 5)

    def test_contains(self):
        m = ORMap("A")
        m.put("key1", GCounter("A"))
        self.assertTrue(m.contains("key1"))
        self.assertFalse(m.contains("key2"))

    def test_remove(self):
        m = ORMap("A")
        m.put("key1", GCounter("A"))
        m.remove("key1")
        self.assertFalse(m.contains("key1"))

    def test_keys(self):
        m = ORMap("A")
        m.put("a", GCounter("A"))
        m.put("b", GCounter("A"))
        self.assertEqual(m.keys, frozenset({"a", "b"}))

    def test_merge_values(self):
        a = ORMap("A")
        ca = GCounter("A"); ca.increment(3)
        a.put("key", ca)

        b = ORMap("B")
        cb = GCounter("B"); cb.increment(5)
        b.put("key", cb)

        merged = a.merge(b)
        self.assertEqual(merged.get("key").value, 8)

    def test_merge_keys(self):
        a = ORMap("A"); a.put("x", GCounter("A"))
        b = ORMap("B"); b.put("y", GCounter("B"))
        merged = a.merge(b)
        self.assertEqual(merged.keys, frozenset({"x", "y"}))

    def test_merge_commutativity(self):
        a = ORMap("A"); a.put("k", GCounter("A"))
        b = ORMap("B"); b.put("k", GCounter("B"))
        m1 = a.merge(b)
        m2 = b.merge(a)
        self.assertEqual(m1.keys, m2.keys)


# =============================================================================
# VectorClock
# =============================================================================

class TestVectorClock(unittest.TestCase):
    def test_increment(self):
        vc = VectorClock("A")
        vc.increment()
        self.assertEqual(vc.get("A"), 1)

    def test_merge(self):
        a = VectorClock("A"); a.increment(); a.increment()
        b = VectorClock("B"); b.increment()
        merged = a.merge(b)
        self.assertEqual(merged.get("A"), 2)
        self.assertEqual(merged.get("B"), 1)

    def test_dominates(self):
        a = VectorClock("A")
        a.set("A", 2); a.set("B", 1)
        b = VectorClock("B")
        b.set("A", 1); b.set("B", 1)
        self.assertTrue(a.dominates(b))
        self.assertFalse(b.dominates(a))

    def test_concurrent(self):
        a = VectorClock("A"); a.set("A", 2); a.set("B", 0)
        b = VectorClock("B"); b.set("A", 0); b.set("B", 2)
        self.assertTrue(a.concurrent(b))

    def test_le(self):
        a = VectorClock("A"); a.set("A", 1)
        b = VectorClock("B"); b.set("A", 2); b.set("B", 1)
        self.assertTrue(a <= b)
        self.assertFalse(b <= a)

    def test_equality(self):
        a = VectorClock("A"); a.set("A", 1); a.set("B", 2)
        b = VectorClock("B"); b.set("A", 1); b.set("B", 2)
        self.assertEqual(a, b)


# =============================================================================
# CRDTNetwork
# =============================================================================

class TestCRDTNetwork(unittest.TestCase):
    def test_gcounter_convergence(self):
        net = CRDTNetwork()
        a = GCounter("A"); a.increment(3)
        b = GCounter("B"); b.increment(5)
        c = GCounter("C"); c.increment(7)
        net.add_replica("A", a)
        net.add_replica("B", b)
        net.add_replica("C", c)
        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.get("A").value, 15)
        self.assertEqual(net.get("B").value, 15)
        self.assertEqual(net.get("C").value, 15)

    def test_orset_convergence(self):
        net = CRDTNetwork()
        a = ORSet("A"); a.add("x"); a.add("y")
        b = ORSet("B"); b.add("y"); b.add("z")
        net.add_replica("A", a)
        net.add_replica("B", b)
        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.get("A").elements, frozenset({"x", "y", "z"}))

    def test_pncounter_convergence(self):
        net = CRDTNetwork()
        a = PNCounter("A"); a.increment(10)
        b = PNCounter("B"); b.decrement(3)
        c = PNCounter("C"); c.increment(5); c.decrement(2)
        net.add_replica("A", a)
        net.add_replica("B", b)
        net.add_replica("C", c)
        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.get("A").value, 10)  # 10 + 0 - 3 + 5 - 2 = 10

    def test_sync_pair(self):
        net = CRDTNetwork()
        a = GCounter("A"); a.increment(3)
        b = GCounter("B"); b.increment(5)
        net.add_replica("A", a)
        net.add_replica("B", b)
        net.sync("A", "B")  # only A -> B
        self.assertEqual(net.get("B").value, 8)
        self.assertEqual(net.get("A").value, 3)  # A not updated

    def test_partial_sync(self):
        net = CRDTNetwork()
        a = GSet("A"); a.add(1)
        b = GSet("B"); b.add(2)
        c = GSet("C"); c.add(3)
        net.add_replica("A", a)
        net.add_replica("B", b)
        net.add_replica("C", c)
        net.sync("A", "B")
        self.assertEqual(net.get("B").elements, frozenset({1, 2}))
        self.assertEqual(net.get("C").elements, frozenset({3}))  # C isolated

    def test_single_replica_converged(self):
        net = CRDTNetwork()
        a = GCounter("A"); a.increment(1)
        net.add_replica("A", a)
        self.assertTrue(net.converged())


# =============================================================================
# Integration / Scenarios
# =============================================================================

class TestCRDTScenarios(unittest.TestCase):
    def test_shopping_cart_scenario(self):
        """Two users editing a shared shopping cart."""
        # User A's cart
        cart_a = ORMap("A")
        apples_a = PNCounter("A"); apples_a.increment(3)
        cart_a.put("apples", apples_a)

        # User B's cart
        cart_b = ORMap("B")
        bananas_b = PNCounter("B"); bananas_b.increment(2)
        cart_b.put("bananas", bananas_b)

        # Merge carts
        merged = cart_a.merge(cart_b)
        self.assertTrue(merged.contains("apples"))
        self.assertTrue(merged.contains("bananas"))

    def test_collaborative_editing_scenario(self):
        """Two editors inserting into a shared document."""
        doc_a = RGASequence("A")
        doc_a.append("H"); doc_a.append("i")

        doc_b = RGASequence("B")
        doc_b.append("!"); doc_b.append("!")

        merged = doc_a.merge(doc_b)
        self.assertEqual(len(merged), 4)

    def test_distributed_counter_scenario(self):
        """Page view counter across 5 servers."""
        counters = []
        for i in range(5):
            c = GCounter(f"server_{i}")
            c.increment(100)
            counters.append(c)

        # Merge all
        result = counters[0]
        for c in counters[1:]:
            result = result.merge(c)
        self.assertEqual(result.value, 500)

    def test_feature_flag_scenario(self):
        """Feature flag with enable-wins semantics."""
        flag_a = EWFlag("A")
        flag_b = EWFlag("B")

        # A enables the feature
        flag_a.enable()
        # B disables it (concurrent)
        flag_b.enable()
        flag_b.disable()

        merged = flag_a.merge(flag_b)
        self.assertTrue(merged.value)  # enable wins

    def test_lww_register_config_scenario(self):
        """Distributed config with last-writer-wins."""
        config_a = LWWRegister("A")
        config_b = LWWRegister("B")

        config_a.set({"timeout": 30}, timestamp=100)
        config_b.set({"timeout": 60}, timestamp=200)

        merged = config_a.merge(config_b)
        self.assertEqual(merged.value, {"timeout": 60})

    def test_network_partition_recovery(self):
        """Three replicas, one partitioned, then all merge."""
        a = GCounter("A"); a.increment(10)
        b = GCounter("B"); b.increment(20)
        c = GCounter("C"); c.increment(30)

        # A and B sync during partition (C isolated)
        ab = a.merge(b)

        # After partition heals, merge C
        final = ab.merge(c)
        self.assertEqual(final.value, 60)

    def test_conflict_resolution_mvregister(self):
        """MVRegister shows conflicts, user resolves."""
        a = MVRegister("A"); a.set("price: $10")
        b = MVRegister("B"); b.set("price: $20")
        merged = a.merge(b)
        # Both values visible
        self.assertEqual(len(merged.values), 2)
        # User resolves
        merged.set("price: $15")
        self.assertEqual(merged.value, "price: $15")

    def test_twopset_user_ban(self):
        """User ban list with permanent removal semantics."""
        bans = TwoPSet("A")
        bans.add("user123")
        bans.add("user456")
        bans.remove("user123")  # unban
        bans.add("user123")  # try to re-ban -- TwoPSet says no
        self.assertFalse(bans.contains("user123"))

    def test_orset_user_ban_allows_reban(self):
        """ORSet allows re-adding after removal."""
        bans = ORSet("A")
        bans.add("user123")
        bans.remove("user123")
        bans.add("user123")  # re-ban works
        self.assertTrue(bans.contains("user123"))


if __name__ == "__main__":
    unittest.main()
