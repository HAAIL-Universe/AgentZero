"""Tests for C232: Vector Clocks -- Causal Ordering for Distributed Systems."""

import unittest
from vector_clocks import (
    VectorClock, VersionVector, Dot, DottedVersionVector,
    ITCStamp, BloomClock, Event, CausalHistory,
    CausalBroadcast, CausalConsistencyChecker,
)


# ===================================================================
# 1. VectorClock Tests
# ===================================================================

class TestVectorClock(unittest.TestCase):

    def test_empty_clock(self):
        vc = VectorClock()
        self.assertEqual(vc.clock, {})
        self.assertEqual(vc.get("a"), 0)

    def test_increment(self):
        vc = VectorClock()
        vc2 = vc.increment("a")
        self.assertEqual(vc2.get("a"), 1)
        self.assertEqual(vc.get("a"), 0)  # Immutable

    def test_multiple_increments(self):
        vc = VectorClock()
        vc = vc.increment("a").increment("a").increment("b")
        self.assertEqual(vc.get("a"), 2)
        self.assertEqual(vc.get("b"), 1)

    def test_merge(self):
        vc1 = VectorClock({"a": 2, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 3, "c": 1})
        merged = vc1.merge(vc2)
        self.assertEqual(merged.get("a"), 2)
        self.assertEqual(merged.get("b"), 3)
        self.assertEqual(merged.get("c"), 1)

    def test_merge_commutative(self):
        vc1 = VectorClock({"a": 2, "b": 1})
        vc2 = VectorClock({"b": 3, "c": 1})
        self.assertEqual(vc1.merge(vc2), vc2.merge(vc1))

    def test_merge_idempotent(self):
        vc = VectorClock({"a": 2, "b": 1})
        self.assertEqual(vc.merge(vc), vc)

    def test_merge_associative(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"b": 2})
        vc3 = VectorClock({"c": 3})
        self.assertEqual(vc1.merge(vc2).merge(vc3), vc1.merge(vc2.merge(vc3)))

    def test_equality(self):
        vc1 = VectorClock({"a": 1, "b": 2})
        vc2 = VectorClock({"a": 1, "b": 2})
        self.assertEqual(vc1, vc2)

    def test_equality_with_zero(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 1, "b": 0})
        self.assertEqual(vc1, vc2)

    def test_happens_before(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 2})
        self.assertTrue(vc1 < vc2)
        self.assertFalse(vc2 < vc1)

    def test_happens_before_multinode(self):
        vc1 = VectorClock({"a": 1, "b": 1})
        vc2 = VectorClock({"a": 2, "b": 1})
        self.assertTrue(vc1 < vc2)

    def test_concurrent(self):
        vc1 = VectorClock({"a": 2, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 2})
        self.assertTrue(vc1.concurrent(vc2))
        self.assertFalse(vc1 < vc2)
        self.assertFalse(vc2 < vc1)

    def test_compare_equal(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 1})
        self.assertEqual(vc1.compare(vc2), "equal")

    def test_compare_before(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 2})
        self.assertEqual(vc1.compare(vc2), "before")

    def test_compare_after(self):
        vc1 = VectorClock({"a": 2})
        vc2 = VectorClock({"a": 1})
        self.assertEqual(vc1.compare(vc2), "after")

    def test_compare_concurrent(self):
        vc1 = VectorClock({"a": 2, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 2})
        self.assertEqual(vc1.compare(vc2), "concurrent")

    def test_dominates(self):
        vc1 = VectorClock({"a": 2, "b": 2})
        vc2 = VectorClock({"a": 1, "b": 1})
        self.assertTrue(vc1.dominates(vc2))
        self.assertFalse(vc2.dominates(vc1))

    def test_descends(self):
        vc1 = VectorClock({"a": 2, "b": 1})
        vc2 = VectorClock({"a": 1})
        self.assertTrue(vc1.descends(vc2))

    def test_nodes(self):
        vc = VectorClock({"a": 1, "b": 0, "c": 3})
        self.assertEqual(vc.nodes(), {"a", "c"})

    def test_copy(self):
        vc = VectorClock({"a": 1})
        vc2 = vc.copy()
        vc3 = vc2.increment("a")
        self.assertEqual(vc.get("a"), 1)  # Original unchanged
        self.assertEqual(vc3.get("a"), 2)

    def test_hash(self):
        vc1 = VectorClock({"a": 1, "b": 2})
        vc2 = VectorClock({"a": 1, "b": 2})
        self.assertEqual(hash(vc1), hash(vc2))
        s = {vc1}
        self.assertIn(vc2, s)

    def test_le_ge(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 2})
        self.assertTrue(vc1 <= vc2)
        self.assertTrue(vc2 >= vc1)
        self.assertTrue(vc1 <= vc1)

    def test_repr(self):
        vc = VectorClock({"a": 1, "b": 2})
        r = repr(vc)
        self.assertIn("VC(", r)


# ===================================================================
# 2. VersionVector Tests
# ===================================================================

class TestVersionVector(unittest.TestCase):

    def test_empty(self):
        vv = VersionVector()
        self.assertEqual(vv.vector, {})

    def test_increment(self):
        vv = VersionVector().increment("r1")
        self.assertEqual(vv.vector, {"r1": 1})

    def test_merge(self):
        vv1 = VersionVector({"r1": 2, "r2": 1})
        vv2 = VersionVector({"r1": 1, "r2": 3})
        merged = vv1.merge(vv2)
        self.assertEqual(merged.vector["r1"], 2)
        self.assertEqual(merged.vector["r2"], 3)

    def test_dominates(self):
        vv1 = VersionVector({"r1": 2, "r2": 2})
        vv2 = VersionVector({"r1": 1, "r2": 1})
        self.assertTrue(vv1.dominates(vv2))
        self.assertFalse(vv2.dominates(vv1))

    def test_not_dominates_equal(self):
        vv = VersionVector({"r1": 1})
        self.assertFalse(vv.dominates(vv))

    def test_conflicts(self):
        vv1 = VersionVector({"r1": 2, "r2": 1})
        vv2 = VersionVector({"r1": 1, "r2": 2})
        self.assertTrue(vv1.conflicts(vv2))

    def test_no_conflict_domination(self):
        vv1 = VersionVector({"r1": 2, "r2": 2})
        vv2 = VersionVector({"r1": 1, "r2": 1})
        self.assertFalse(vv1.conflicts(vv2))

    def test_no_conflict_equal(self):
        vv = VersionVector({"r1": 1})
        self.assertFalse(vv.conflicts(vv))

    def test_equality(self):
        vv1 = VersionVector({"r1": 1})
        vv2 = VersionVector({"r1": 1})
        self.assertEqual(vv1, vv2)

    def test_copy(self):
        vv = VersionVector({"r1": 1})
        vv2 = vv.copy()
        self.assertEqual(vv, vv2)

    def test_hash(self):
        vv1 = VersionVector({"r1": 1})
        vv2 = VersionVector({"r1": 1})
        self.assertEqual(hash(vv1), hash(vv2))

    def test_repr(self):
        vv = VersionVector({"r1": 1})
        self.assertIn("VV(", repr(vv))

    def test_three_way_conflict(self):
        """Three replicas, all concurrent."""
        vv1 = VersionVector({"r1": 2, "r2": 1, "r3": 1})
        vv2 = VersionVector({"r1": 1, "r2": 2, "r3": 1})
        vv3 = VersionVector({"r1": 1, "r2": 1, "r3": 2})
        self.assertTrue(vv1.conflicts(vv2))
        self.assertTrue(vv2.conflicts(vv3))
        self.assertTrue(vv1.conflicts(vv3))


# ===================================================================
# 3. DottedVersionVector Tests
# ===================================================================

class TestDottedVersionVector(unittest.TestCase):

    def test_empty(self):
        dvv = DottedVersionVector()
        self.assertIsNone(dvv.dot)
        self.assertEqual(dvv.version_vector, {})

    def test_new_event(self):
        dvv = DottedVersionVector()
        dvv2 = dvv.new_event("r1")
        self.assertEqual(dvv2.dot, Dot("r1", 1))

    def test_sequential_events(self):
        dvv = DottedVersionVector()
        dvv = dvv.new_event("r1")
        dvv = dvv.new_event("r1")
        self.assertEqual(dvv.dot, Dot("r1", 2))

    def test_descends(self):
        dvv1 = DottedVersionVector()
        dvv1 = dvv1.new_event("r1")
        dvv2 = dvv1.new_event("r1")
        self.assertTrue(dvv2.descends(dvv1))
        self.assertFalse(dvv1.descends(dvv2))

    def test_concurrent(self):
        dvv = DottedVersionVector()
        dvv1 = dvv.new_event("r1")
        dvv2 = dvv.new_event("r2")
        self.assertTrue(dvv1.concurrent(dvv2))

    def test_sync(self):
        dvv1 = DottedVersionVector(Dot("r1", 1))
        dvv2 = DottedVersionVector(Dot("r2", 1))
        synced = dvv1.sync(dvv2)
        full = synced._full_vv()
        self.assertEqual(full.get("r1", 0), 1)
        self.assertEqual(full.get("r2", 0), 1)

    def test_dot_equality(self):
        d1 = Dot("r1", 1)
        d2 = Dot("r1", 1)
        self.assertEqual(d1, d2)
        self.assertEqual(hash(d1), hash(d2))

    def test_dot_inequality(self):
        d1 = Dot("r1", 1)
        d2 = Dot("r1", 2)
        self.assertNotEqual(d1, d2)

    def test_dvv_equality(self):
        dvv1 = DottedVersionVector(Dot("r1", 1), {"r1": 0})
        dvv2 = DottedVersionVector(Dot("r1", 1), {})
        self.assertEqual(dvv1, dvv2)

    def test_multi_replica_events(self):
        dvv = DottedVersionVector()
        dvv = dvv.new_event("r1")
        dvv = dvv.new_event("r2")
        dvv = dvv.new_event("r1")
        full = dvv._full_vv()
        self.assertEqual(full.get("r1", 0), 2)

    def test_repr(self):
        dvv = DottedVersionVector(Dot("r1", 1))
        self.assertIn("DVV(", repr(dvv))


# ===================================================================
# 4. IntervalTreeClock Tests
# ===================================================================

class TestIntervalTreeClock(unittest.TestCase):

    def test_seed(self):
        s = ITCStamp.seed()
        self.assertEqual(s.id, 1)
        self.assertEqual(s.event, 0)

    def test_fork(self):
        s = ITCStamp.seed()
        a, b = s.fork()
        self.assertNotEqual(a.id, b.id)
        # IDs are complementary
        merged_id = ITCStamp._sum_id(a.id, b.id)
        self.assertEqual(merged_id, 1)

    def test_event_tick(self):
        s = ITCStamp.seed()
        s2 = s.event_tick()
        self.assertTrue(s.leq(s2))
        self.assertFalse(s2.leq(s))

    def test_fork_then_tick(self):
        s = ITCStamp.seed()
        a, b = s.fork()
        a2 = a.event_tick()
        b2 = b.event_tick()
        self.assertTrue(a2.concurrent(b2))

    def test_join_after_fork(self):
        s = ITCStamp.seed()
        a, b = s.fork()
        a2 = a.event_tick()
        b2 = b.event_tick()
        joined = a2.join(b2)
        self.assertTrue(a2.leq(joined))
        self.assertTrue(b2.leq(joined))

    def test_multiple_forks(self):
        s = ITCStamp.seed()
        a, b = s.fork()
        a1, a2 = a.fork()
        # Now we have 3 participants
        a1t = a1.event_tick()
        a2t = a2.event_tick()
        bt = b.event_tick()
        self.assertTrue(a1t.concurrent(a2t))
        self.assertTrue(a1t.concurrent(bt))

    def test_sequential_events(self):
        s = ITCStamp.seed()
        s1 = s.event_tick()
        s2 = s1.event_tick()
        s3 = s2.event_tick()
        self.assertTrue(s.leq(s1))
        self.assertTrue(s1.leq(s2))
        self.assertTrue(s2.leq(s3))
        self.assertFalse(s3.leq(s))

    def test_fork_join_fork(self):
        """Dynamic process creation/retirement."""
        s = ITCStamp.seed()
        a, b = s.fork()
        a = a.event_tick()
        b = b.event_tick()
        # b retires, merges back into a
        ab = a.join(b)
        # Fork again from merged
        c, d = ab.fork()
        c = c.event_tick()
        # c should see both a and b's events
        self.assertTrue(a.leq(c))
        self.assertTrue(b.leq(c))

    def test_leq_reflexive(self):
        s = ITCStamp.seed().event_tick()
        self.assertTrue(s.leq(s))

    def test_repr(self):
        s = ITCStamp.seed()
        self.assertIn("ITC(", repr(s))

    def test_deep_fork_tree(self):
        """Fork multiple levels deep."""
        s = ITCStamp.seed()
        stamps = [s]
        for _ in range(4):
            a, b = stamps[-1].fork()
            stamps.append(a)
            stamps.append(b)
        # All leaf stamps should be concurrent after ticking
        ticked = [st.event_tick() for st in stamps[1:]]
        for i in range(len(ticked)):
            for j in range(i + 1, len(ticked)):
                # Not necessarily all concurrent (some share ancestry)
                # but they should all be leq to their join
                joined = ticked[i].join(ticked[j])
                self.assertTrue(ticked[i].leq(joined))
                self.assertTrue(ticked[j].leq(joined))


# ===================================================================
# 5. BloomClock Tests
# ===================================================================

class TestBloomClock(unittest.TestCase):

    def test_empty(self):
        bc = BloomClock()
        self.assertEqual(bc.count, 0)
        self.assertAlmostEqual(bc.fill_ratio(), 0.0)

    def test_record(self):
        bc = BloomClock()
        bc2 = bc.record("node1")
        self.assertEqual(bc2.count, 1)
        self.assertEqual(bc.count, 0)  # Immutable

    def test_multiple_records(self):
        bc = BloomClock()
        for i in range(10):
            bc = bc.record("node1")
        self.assertEqual(bc.count, 10)
        self.assertGreater(bc.fill_ratio(), 0.0)

    def test_merge(self):
        bc1 = BloomClock().record("a").record("a")
        bc2 = BloomClock().record("b").record("b")
        merged = bc1.merge(bc2)
        # Merged should contain all bits from both
        self.assertTrue(merged.contains(bc1))
        self.assertTrue(merged.contains(bc2))

    def test_contains_self(self):
        bc = BloomClock().record("a")
        self.assertTrue(bc.contains(bc))

    def test_empty_contains_empty(self):
        bc = BloomClock()
        self.assertTrue(bc.contains(bc))

    def test_maybe_before(self):
        bc1 = BloomClock(size=256, num_hashes=5)
        bc1 = bc1.record("a")
        bc2 = bc1.record("a")  # bc2 is "after" bc1
        self.assertTrue(bc1.maybe_before(bc2))

    def test_definitely_concurrent(self):
        # With large enough clocks and distinct events
        bc1 = BloomClock(size=1024, num_hashes=7)
        bc2 = BloomClock(size=1024, num_hashes=7)
        # Record many distinct events to fill different bits
        for i in range(50):
            bc1 = bc1.record(f"nodeA_{i}")
        for i in range(50):
            bc2 = bc2.record(f"nodeB_{i}")
        # High probability of being detected as concurrent
        # (different bits set in each)
        if not bc1.contains(bc2) and not bc2.contains(bc1):
            self.assertTrue(bc1.definitely_concurrent(bc2))

    def test_copy(self):
        bc = BloomClock().record("a")
        bc2 = bc.copy()
        self.assertEqual(bc.count, bc2.count)
        self.assertTrue(bc.contains(bc2))

    def test_fill_ratio_increases(self):
        bc = BloomClock(size=128)
        ratios = []
        for i in range(20):
            bc = bc.record(f"node_{i}")
            ratios.append(bc.fill_ratio())
        # Fill ratio should generally increase (monotonic with noise from collisions)
        self.assertGreater(ratios[-1], ratios[0])

    def test_false_positive_rate(self):
        bc = BloomClock(size=128, num_hashes=3)
        for i in range(10):
            bc = bc.record(f"n{i}")
        fpr = bc.estimated_false_positive_rate()
        self.assertGreaterEqual(fpr, 0.0)
        self.assertLessEqual(fpr, 1.0)

    def test_repr(self):
        bc = BloomClock().record("a")
        self.assertIn("BloomClock(", repr(bc))

    def test_size_property(self):
        bc = BloomClock(size=256)
        self.assertEqual(bc.size, 256)


# ===================================================================
# 6. CausalHistory Tests
# ===================================================================

class TestCausalHistory(unittest.TestCase):

    def test_empty(self):
        ch = CausalHistory()
        self.assertEqual(len(ch), 0)

    def test_add_event(self):
        ch = CausalHistory()
        e = ch.add_event("e1", "node1")
        self.assertEqual(e.event_id, "e1")
        self.assertEqual(e.node_id, "node1")
        self.assertEqual(len(ch), 1)

    def test_causal_chain(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "a", causes=["e1"])
        ch.add_event("e3", "a", causes=["e2"])
        self.assertTrue(ch.happens_before("e1", "e2"))
        self.assertTrue(ch.happens_before("e2", "e3"))
        self.assertTrue(ch.happens_before("e1", "e3"))

    def test_concurrent_events(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        self.assertTrue(ch.concurrent("e1", "e2"))

    def test_send_receive(self):
        ch = CausalHistory()
        ch.send_event("s1", "a", "hello")
        ch.receive_event("r1", "b", "s1", "got hello")
        self.assertTrue(ch.happens_before("s1", "r1"))

    def test_causal_path(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b", causes=["e1"])
        ch.add_event("e3", "c", causes=["e2"])
        path = ch.causal_path("e1", "e3")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "e1")
        self.assertEqual(path[-1], "e3")

    def test_no_causal_path(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        # No path from concurrent events (unless e1 -> e2 via causes)
        path = ch.causal_path("e1", "e2")
        self.assertIsNone(path)

    def test_causal_path_self(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        self.assertEqual(ch.causal_path("e1", "e1"), ["e1"])

    def test_frontier(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        ch.add_event("e3", "c", causes=["e1", "e2"])
        frontier = ch.frontier()
        self.assertEqual(frontier, ["e3"])

    def test_frontier_multiple(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        frontier = ch.frontier()
        self.assertEqual(sorted(frontier), ["e1", "e2"])

    def test_concurrent_set(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        ch.add_event("e3", "c")
        conc = ch.concurrent_set("e1")
        self.assertIn("e2", conc)
        self.assertIn("e3", conc)

    def test_causal_cut(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        ch.add_event("e3", "c", causes=["e1", "e2"])
        cut = ch.causal_cut("e3")
        self.assertIn("e1", cut)
        self.assertIn("e2", cut)

    def test_linearize(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        ch.add_event("e3", "c", causes=["e1", "e2"])
        order = ch.linearize()
        self.assertEqual(len(order), 3)
        self.assertLess(order.index("e1"), order.index("e3"))
        self.assertLess(order.index("e2"), order.index("e3"))

    def test_local_event(self):
        ch = CausalHistory()
        ch.local_event("e1", "a", "first")
        ch.local_event("e2", "a", "second")
        self.assertTrue(ch.happens_before("e1", "e2"))

    def test_duplicate_event_error(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        with self.assertRaises(ValueError):
            ch.add_event("e1", "b")

    def test_missing_cause_error(self):
        ch = CausalHistory()
        with self.assertRaises(ValueError):
            ch.add_event("e1", "a", causes=["nonexistent"])

    def test_complex_dag(self):
        """
        a: e1 -> e4 -> e6
        b: e2 -> e5 -> e6
        c: e3 --------> e6
        """
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b")
        ch.add_event("e3", "c")
        ch.add_event("e4", "a", causes=["e1"])
        ch.add_event("e5", "b", causes=["e2"])
        ch.add_event("e6", "a", causes=["e4", "e5", "e3"])

        self.assertTrue(ch.happens_before("e1", "e6"))
        self.assertTrue(ch.happens_before("e2", "e6"))
        self.assertTrue(ch.happens_before("e3", "e6"))
        self.assertTrue(ch.concurrent("e1", "e2"))
        self.assertTrue(ch.concurrent("e4", "e5"))
        self.assertEqual(ch.frontier(), ["e6"])

    def test_repr(self):
        ch = CausalHistory()
        ch.add_event("e1", "a")
        self.assertIn("CausalHistory(", repr(ch))

    def test_diamond_dag(self):
        """
        e1 -> e2 -> e4
        e1 -> e3 -> e4
        """
        ch = CausalHistory()
        ch.add_event("e1", "a")
        ch.add_event("e2", "b", causes=["e1"])
        ch.add_event("e3", "c", causes=["e1"])
        ch.add_event("e4", "a", causes=["e2", "e3"])
        self.assertTrue(ch.happens_before("e1", "e4"))
        self.assertTrue(ch.concurrent("e2", "e3"))
        cut = ch.causal_cut("e4")
        self.assertEqual(cut, {"e1", "e2", "e3"})


# ===================================================================
# 7. CausalBroadcast Tests
# ===================================================================

class TestCausalBroadcast(unittest.TestCase):

    def test_basic_broadcast(self):
        a = CausalBroadcast("a", ["b"])
        clock, data = a.broadcast("hello")
        self.assertEqual(data, "hello")
        self.assertEqual(clock.get("a"), 1)

    def test_delivery(self):
        a = CausalBroadcast("a", ["b"])
        b = CausalBroadcast("b", ["a"])

        clock, data = a.broadcast("msg1")
        delivered = b.receive("a", clock, data)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0], ("a", "msg1"))

    def test_causal_order_delivery(self):
        """Messages delivered in causal order even if received out of order."""
        a = CausalBroadcast("a", ["b", "c"])
        b = CausalBroadcast("b", ["a", "c"])
        c = CausalBroadcast("c", ["a", "b"])

        # a sends msg1
        clock1, data1 = a.broadcast("msg1")

        # b receives msg1, then sends msg2
        b.receive("a", clock1, data1)
        clock2, data2 = b.broadcast("msg2")

        # c receives msg2 BEFORE msg1 (out of order)
        delivered = c.receive("b", clock2, data2)
        self.assertEqual(len(delivered), 0)  # msg2 buffered (depends on msg1)
        self.assertEqual(c.pending_count, 1)

        # c now receives msg1
        delivered = c.receive("a", clock1, data1)
        # Should deliver both: msg1 then msg2
        self.assertEqual(len(delivered), 2)
        self.assertEqual(delivered[0], ("a", "msg1"))
        self.assertEqual(delivered[1], ("b", "msg2"))

    def test_concurrent_messages(self):
        a = CausalBroadcast("a", ["b", "c"])
        b = CausalBroadcast("b", ["a", "c"])
        c = CausalBroadcast("c", ["a", "b"])

        clock_a, data_a = a.broadcast("from_a")
        clock_b, data_b = b.broadcast("from_b")

        # c receives both -- both should deliver (concurrent)
        d1 = c.receive("a", clock_a, data_a)
        d2 = c.receive("b", clock_b, data_b)
        self.assertEqual(len(d1), 1)
        self.assertEqual(len(d2), 1)

    def test_multiple_messages_same_sender(self):
        a = CausalBroadcast("a", ["b"])
        b = CausalBroadcast("b", ["a"])

        c1, d1 = a.broadcast("first")
        c2, d2 = a.broadcast("second")

        # Deliver out of order
        delivered = b.receive("a", c2, d2)
        self.assertEqual(len(delivered), 0)  # second buffered

        delivered = b.receive("a", c1, d1)
        self.assertEqual(len(delivered), 2)
        self.assertEqual(delivered[0][1], "first")
        self.assertEqual(delivered[1][1], "second")

    def test_delivered_property(self):
        a = CausalBroadcast("a", ["b"])
        a.broadcast("m1")
        a.broadcast("m2")
        self.assertEqual(len(a.delivered), 2)

    def test_repr(self):
        a = CausalBroadcast("a", ["b"])
        self.assertIn("CausalBroadcast(", repr(a))

    def test_three_node_chain(self):
        """a -> b -> c: each node forwards with correct causal context."""
        a = CausalBroadcast("a", ["b", "c"])
        b = CausalBroadcast("b", ["a", "c"])
        c = CausalBroadcast("c", ["a", "b"])

        ca, da = a.broadcast("start")
        b.receive("a", ca, da)
        cb, db = b.broadcast("relay")

        # c receives b's relay first
        delivered = c.receive("b", cb, db)
        self.assertEqual(len(delivered), 0)  # Needs a's message first

        delivered = c.receive("a", ca, da)
        self.assertEqual(len(delivered), 2)


# ===================================================================
# 8. CausalConsistencyChecker Tests
# ===================================================================

class TestCausalConsistencyChecker(unittest.TestCase):

    def test_empty_log(self):
        checker = CausalConsistencyChecker()
        violations = checker.check_consistency()
        self.assertEqual(violations, [])

    def test_consistent_log(self):
        checker = CausalConsistencyChecker()
        checker.add_entry("a", VectorClock({"a": 1}))
        checker.add_entry("a", VectorClock({"a": 2}))
        checker.add_entry("b", VectorClock({"b": 1}))
        violations = checker.check_consistency()
        self.assertEqual(violations, [])

    def test_non_monotonic(self):
        checker = CausalConsistencyChecker()
        checker.add_entry("a", VectorClock({"a": 2}))
        checker.add_entry("a", VectorClock({"a": 1}))  # Goes backward
        violations = checker.check_consistency()
        self.assertTrue(any("Non-monotonic" in v for v in violations))

    def test_gap_detection(self):
        checker = CausalConsistencyChecker()
        checker.add_entry("a", VectorClock({"a": 1}))
        checker.add_entry("a", VectorClock({"a": 5}))  # Gap: 2,3,4 missing
        violations = checker.check_consistency()
        self.assertTrue(any("Gap" in v for v in violations))

    def test_find_anomalies_empty(self):
        checker = CausalConsistencyChecker()
        anomalies = checker.find_anomalies()
        self.assertEqual(anomalies["gaps"], [])
        self.assertEqual(anomalies["non_monotonic"], [])

    def test_find_anomalies_gap(self):
        checker = CausalConsistencyChecker()
        checker.add_entry("a", VectorClock({"a": 1}))
        checker.add_entry("a", VectorClock({"a": 4}))
        anomalies = checker.find_anomalies()
        self.assertEqual(anomalies["gaps"], [1])

    def test_find_anomalies_non_monotonic(self):
        checker = CausalConsistencyChecker()
        checker.add_entry("a", VectorClock({"a": 2}))
        checker.add_entry("a", VectorClock({"a": 1}))
        anomalies = checker.find_anomalies()
        self.assertIn(1, anomalies["non_monotonic"])

    def test_len(self):
        checker = CausalConsistencyChecker()
        checker.add_entry("a", VectorClock({"a": 1}))
        checker.add_entry("b", VectorClock({"b": 1}))
        self.assertEqual(len(checker), 2)

    def test_multi_node_consistent(self):
        checker = CausalConsistencyChecker()
        checker.add_entry("a", VectorClock({"a": 1}))
        checker.add_entry("b", VectorClock({"a": 1, "b": 1}))
        checker.add_entry("a", VectorClock({"a": 2, "b": 1}))
        violations = checker.check_consistency()
        self.assertEqual(violations, [])


# ===================================================================
# Integration / Scenario Tests
# ===================================================================

class TestIntegrationScenarios(unittest.TestCase):

    def test_distributed_kv_store_scenario(self):
        """Simulate a distributed KV store with version vectors."""
        # Three replicas
        r1_vv = VersionVector()
        r2_vv = VersionVector()
        r3_vv = VersionVector()

        # r1 writes key=A
        r1_vv = r1_vv.increment("r1")
        # r2 receives r1's write
        r2_vv = r2_vv.merge(r1_vv).increment("r2")
        # r3 independently writes key=A
        r3_vv = r3_vv.increment("r3")

        # r2 and r3 are in conflict
        self.assertTrue(r2_vv.conflicts(r3_vv))
        # r1 is dominated by r2
        self.assertTrue(r2_vv.dominates(r1_vv))

    def test_chat_application_scenario(self):
        """Simulate a chat where messages must be causally ordered."""
        alice = CausalBroadcast("alice", ["bob", "carol"])
        bob = CausalBroadcast("bob", ["alice", "carol"])
        carol = CausalBroadcast("carol", ["alice", "bob"])

        # Alice: "Let's meet at 5pm"
        c1, d1 = alice.broadcast("Let's meet at 5pm")

        # Bob reads and replies: "Sounds good"
        bob.receive("alice", c1, d1)
        c2, d2 = bob.broadcast("Sounds good")

        # Carol gets Bob's reply before Alice's message
        delivered = carol.receive("bob", c2, d2)
        self.assertEqual(len(delivered), 0)  # Buffered

        # Carol gets Alice's message
        delivered = carol.receive("alice", c1, d1)
        # Both should deliver in order
        self.assertEqual(len(delivered), 2)
        self.assertEqual(delivered[0][1], "Let's meet at 5pm")
        self.assertEqual(delivered[1][1], "Sounds good")

    def test_dynamic_cluster_with_itc(self):
        """Simulate dynamic cluster membership with ITCs."""
        # Start with single node
        master = ITCStamp.seed()
        master = master.event_tick()  # Event 1

        # Node joins: fork
        node1, node2 = master.fork()
        node1 = node1.event_tick()
        node2 = node2.event_tick()

        # Both see master's event
        self.assertTrue(master.leq(node1))
        self.assertTrue(master.leq(node2))

        # But concurrent with each other
        self.assertTrue(node1.concurrent(node2))

        # Node2 retires
        combined = node1.join(node2)
        combined = combined.event_tick()

        # Combined sees everything
        self.assertTrue(node1.leq(combined))
        self.assertTrue(node2.leq(combined))

    def test_causal_history_message_flow(self):
        """Build a message flow DAG and verify properties."""
        ch = CausalHistory()

        # Alice sends to Bob
        ch.send_event("a_send1", "alice", "hello")
        ch.receive_event("b_recv1", "bob", "a_send1", "got hello")

        # Bob replies to Alice
        ch.send_event("b_send1", "bob", "reply")
        ch.receive_event("a_recv1", "alice", "b_send1", "got reply")

        # Carol sends independently
        ch.send_event("c_send1", "carol", "hey all")

        # Verify causality
        self.assertTrue(ch.happens_before("a_send1", "b_recv1"))
        self.assertTrue(ch.happens_before("b_send1", "a_recv1"))
        self.assertTrue(ch.concurrent("a_send1", "c_send1"))

        # Verify path exists
        path = ch.causal_path("a_send1", "a_recv1")
        self.assertIsNotNone(path)

        # Carol's event is concurrent with everything
        for eid in ["a_send1", "b_recv1", "b_send1", "a_recv1"]:
            self.assertTrue(ch.concurrent("c_send1", eid))

    def test_version_vector_merge_resolution(self):
        """Simulate conflict detection and resolution with VVs."""
        # Two replicas independently update
        v1 = VersionVector({"r1": 1})
        v2 = VersionVector({"r2": 1})

        # Conflict detected
        self.assertTrue(v1.conflicts(v2))

        # Resolve by merging
        resolved = v1.merge(v2).increment("r1")

        # Resolved dominates both
        self.assertTrue(resolved.dominates(v1))
        self.assertTrue(resolved.dominates(v2))

    def test_bloom_clock_scalability(self):
        """Bloom clocks scale to many processes without growing."""
        bc = BloomClock(size=256, num_hashes=4)
        for i in range(100):
            bc = bc.record(f"process_{i}")
        # Size stays fixed
        self.assertEqual(bc.size, 256)
        self.assertEqual(bc.count, 100)

    def test_consistency_checker_scenario(self):
        """Check a full distributed exchange for consistency."""
        checker = CausalConsistencyChecker()

        # Node A: local events
        checker.add_entry("a", VectorClock({"a": 1}))
        checker.add_entry("a", VectorClock({"a": 2}))

        # Node B: receives from A, then does local
        checker.add_entry("b", VectorClock({"a": 2, "b": 1}))
        checker.add_entry("b", VectorClock({"a": 2, "b": 2}))

        # Node A: receives from B
        checker.add_entry("a", VectorClock({"a": 3, "b": 2}))

        violations = checker.check_consistency()
        self.assertEqual(violations, [])

    def test_vector_clock_lamport_simulation(self):
        """Classic Lamport diagram: 3 processes, messages crossing."""
        ch = CausalHistory()

        # Process 1: e1 -> e4
        ch.add_event("e1", "p1")
        # Process 2: e2 -> e5
        ch.add_event("e2", "p2")
        # Process 3: e3 -> e6
        ch.add_event("e3", "p3")

        # p1 sends to p2 (e1 -> e5)
        ch.add_event("e4", "p1", causes=["e1"])
        ch.add_event("e5", "p2", causes=["e2", "e4"])

        # p2 sends to p3 (e5 -> e6)
        ch.add_event("e6", "p3", causes=["e3", "e5"])

        # Transitive: e1 -> e6
        self.assertTrue(ch.happens_before("e1", "e6"))
        # e2 -> e5 -> e6
        self.assertTrue(ch.happens_before("e2", "e6"))
        # e3 || e4 (concurrent)
        self.assertTrue(ch.concurrent("e3", "e4"))

        # Linearization respects causality
        order = ch.linearize()
        self.assertLess(order.index("e1"), order.index("e4"))
        self.assertLess(order.index("e4"), order.index("e5"))
        self.assertLess(order.index("e5"), order.index("e6"))

    def test_dvv_no_false_conflicts(self):
        """DVVs avoid false conflicts that basic VVs would report."""
        # Coordinated writes through a coordinator
        coord = DottedVersionVector()
        v1 = coord.new_event("r1")
        # v1 syncs with coordinator
        v2 = v1.new_event("r1")
        # v2 should descend from v1
        self.assertTrue(v2.descends(v1))
        self.assertFalse(v1.concurrent(v2))


if __name__ == "__main__":
    unittest.main()
