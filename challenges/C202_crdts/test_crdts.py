"""
Tests for C202: CRDTs (Conflict-free Replicated Data Types)
"""

import unittest
import copy
import time
import random
from crdts import (
    VectorClock, CausalContext,
    GCounter, PNCounter, GSet, TwoPSet, ORSet,
    LWWRegister, MVRegister, LWWElementSet, RGA,
    OpCounter, OpSet,
    DeltaGCounter, DeltaPNCounter,
    CRDTMap, CRDTNetwork,
    create_crdt,
)


# =============================================================================
# Vector Clock Tests
# =============================================================================

class TestVectorClock(unittest.TestCase):

    def test_empty_clock(self):
        vc = VectorClock()
        self.assertEqual(vc.get('a'), 0)

    def test_increment(self):
        vc = VectorClock()
        vc.increment('a')
        self.assertEqual(vc.get('a'), 1)
        vc.increment('a')
        self.assertEqual(vc.get('a'), 2)

    def test_merge(self):
        vc1 = VectorClock({'a': 3, 'b': 1})
        vc2 = VectorClock({'a': 1, 'b': 5, 'c': 2})
        merged = vc1.merge(vc2)
        self.assertEqual(merged.get('a'), 3)
        self.assertEqual(merged.get('b'), 5)
        self.assertEqual(merged.get('c'), 2)

    def test_happens_before(self):
        vc1 = VectorClock({'a': 1, 'b': 1})
        vc2 = VectorClock({'a': 2, 'b': 1})
        self.assertTrue(vc1 < vc2)
        self.assertFalse(vc2 < vc1)

    def test_concurrent(self):
        vc1 = VectorClock({'a': 2, 'b': 1})
        vc2 = VectorClock({'a': 1, 'b': 2})
        self.assertTrue(vc1.concurrent(vc2))
        self.assertTrue(vc2.concurrent(vc1))
        self.assertFalse(vc1 < vc2)
        self.assertFalse(vc2 < vc1)

    def test_equality(self):
        vc1 = VectorClock({'a': 1, 'b': 2})
        vc2 = VectorClock({'a': 1, 'b': 2})
        self.assertEqual(vc1, vc2)

    def test_le(self):
        vc1 = VectorClock({'a': 1})
        vc2 = VectorClock({'a': 1, 'b': 1})
        self.assertTrue(vc1 <= vc2)

    def test_copy(self):
        vc = VectorClock({'a': 1})
        vc2 = vc.copy()
        vc2.increment('a')
        self.assertEqual(vc.get('a'), 1)
        self.assertEqual(vc2.get('a'), 2)


# =============================================================================
# Causal Context Tests
# =============================================================================

class TestCausalContext(unittest.TestCase):

    def test_next_dot(self):
        ctx = CausalContext()
        d1 = ctx.next_dot('a')
        self.assertEqual(d1, ('a', 1))
        d2 = ctx.next_dot('a')
        self.assertEqual(d2, ('a', 2))

    def test_has_dot(self):
        ctx = CausalContext()
        ctx.next_dot('a')
        ctx.next_dot('a')
        self.assertTrue(ctx.has_dot(('a', 1)))
        self.assertTrue(ctx.has_dot(('a', 2)))
        self.assertFalse(ctx.has_dot(('a', 3)))
        self.assertFalse(ctx.has_dot(('b', 1)))

    def test_compact(self):
        ctx = CausalContext()
        ctx.dots.add(('a', 1))
        ctx.dots.add(('a', 2))
        ctx.dots.add(('a', 4))  # gap at 3
        ctx.compact()
        self.assertEqual(ctx.cc.get('a'), 2)
        self.assertIn(('a', 4), ctx.dots)

    def test_merge(self):
        ctx1 = CausalContext()
        ctx1.next_dot('a')
        ctx1.next_dot('a')
        ctx2 = CausalContext()
        ctx2.next_dot('b')
        merged = ctx1.merge(ctx2)
        self.assertTrue(merged.has_dot(('a', 1)))
        self.assertTrue(merged.has_dot(('a', 2)))
        self.assertTrue(merged.has_dot(('b', 1)))


# =============================================================================
# GCounter Tests
# =============================================================================

class TestGCounter(unittest.TestCase):

    def test_initial_value(self):
        gc = GCounter('a')
        self.assertEqual(gc.value, 0)

    def test_increment(self):
        gc = GCounter('a')
        gc.increment()
        self.assertEqual(gc.value, 1)
        gc.increment(5)
        self.assertEqual(gc.value, 6)

    def test_negative_increment_raises(self):
        gc = GCounter('a')
        with self.assertRaises(ValueError):
            gc.increment(-1)

    def test_merge_two_nodes(self):
        gc1 = GCounter('a')
        gc2 = GCounter('b')
        gc1.increment(3)
        gc2.increment(5)
        merged = gc1.merge(gc2)
        self.assertEqual(merged.value, 8)

    def test_merge_idempotent(self):
        gc1 = GCounter('a')
        gc1.increment(3)
        merged = gc1.merge(gc1)
        self.assertEqual(merged.value, 3)

    def test_merge_commutative(self):
        gc1 = GCounter('a')
        gc2 = GCounter('b')
        gc1.increment(3)
        gc2.increment(5)
        self.assertEqual(gc1.merge(gc2).value, gc2.merge(gc1).value)

    def test_merge_associative(self):
        gc1 = GCounter('a')
        gc2 = GCounter('b')
        gc3 = GCounter('c')
        gc1.increment(1)
        gc2.increment(2)
        gc3.increment(3)
        m1 = gc1.merge(gc2).merge(gc3)
        m2 = gc1.merge(gc2.merge(gc3))
        self.assertEqual(m1.value, m2.value)

    def test_merge_preserves_max(self):
        gc1 = GCounter('a')
        gc2 = GCounter('a')  # same node_id, different replicas
        gc1.increment(5)
        gc2.increment(3)
        merged = gc1.merge(gc2)
        self.assertEqual(merged.value, 5)  # max, not sum


# =============================================================================
# PNCounter Tests
# =============================================================================

class TestPNCounter(unittest.TestCase):

    def test_initial_value(self):
        pn = PNCounter('a')
        self.assertEqual(pn.value, 0)

    def test_increment_decrement(self):
        pn = PNCounter('a')
        pn.increment(5)
        pn.decrement(3)
        self.assertEqual(pn.value, 2)

    def test_negative_value(self):
        pn = PNCounter('a')
        pn.decrement(10)
        self.assertEqual(pn.value, -10)

    def test_merge(self):
        pn1 = PNCounter('a')
        pn2 = PNCounter('b')
        pn1.increment(10)
        pn2.decrement(3)
        merged = pn1.merge(pn2)
        self.assertEqual(merged.value, 7)

    def test_merge_commutative(self):
        pn1 = PNCounter('a')
        pn2 = PNCounter('b')
        pn1.increment(5)
        pn2.decrement(2)
        self.assertEqual(pn1.merge(pn2).value, pn2.merge(pn1).value)

    def test_merge_idempotent(self):
        pn = PNCounter('a')
        pn.increment(7)
        pn.decrement(2)
        self.assertEqual(pn.merge(pn).value, 5)


# =============================================================================
# GSet Tests
# =============================================================================

class TestGSet(unittest.TestCase):

    def test_empty(self):
        gs = GSet()
        self.assertEqual(gs.value, frozenset())

    def test_add_lookup(self):
        gs = GSet()
        gs.add('a')
        self.assertTrue(gs.lookup('a'))
        self.assertFalse(gs.lookup('b'))

    def test_merge(self):
        gs1 = GSet()
        gs2 = GSet()
        gs1.add('a')
        gs2.add('b')
        merged = gs1.merge(gs2)
        self.assertEqual(merged.value, frozenset(['a', 'b']))

    def test_merge_idempotent(self):
        gs = GSet()
        gs.add('a')
        self.assertEqual(gs.merge(gs).value, frozenset(['a']))


# =============================================================================
# TwoPSet Tests
# =============================================================================

class TestTwoPSet(unittest.TestCase):

    def test_add_remove(self):
        tps = TwoPSet()
        tps.add('a')
        self.assertTrue(tps.lookup('a'))
        tps.remove('a')
        self.assertFalse(tps.lookup('a'))

    def test_remove_permanent(self):
        tps = TwoPSet()
        tps.add('a')
        tps.remove('a')
        tps.add('a')  # re-add doesn't work
        self.assertFalse(tps.lookup('a'))

    def test_remove_nonexistent(self):
        tps = TwoPSet()
        tps.remove('a')  # no-op
        tps.add('a')
        self.assertTrue(tps.lookup('a'))

    def test_merge(self):
        tps1 = TwoPSet()
        tps2 = TwoPSet()
        tps1.add('a')
        tps1.add('b')
        tps2.add('b')
        tps2.remove('b')
        merged = tps1.merge(tps2)
        self.assertTrue(merged.lookup('a'))
        self.assertFalse(merged.lookup('b'))  # remove wins


# =============================================================================
# ORSet Tests
# =============================================================================

class TestORSet(unittest.TestCase):

    def test_add_lookup(self):
        ors = ORSet('a')
        ors.add('x')
        self.assertTrue(ors.lookup('x'))

    def test_remove(self):
        ors = ORSet('a')
        ors.add('x')
        ors.remove('x')
        self.assertFalse(ors.lookup('x'))

    def test_add_after_remove(self):
        ors = ORSet('a')
        ors.add('x')
        ors.remove('x')
        ors.add('x')
        self.assertTrue(ors.lookup('x'))  # new tag survives

    def test_concurrent_add_remove(self):
        """Add on one replica, remove on another -- add wins (observed-remove)."""
        ors1 = ORSet('a')
        ors1.add('x')
        ors2 = ORSet('b')
        ors2.elements = copy.deepcopy(ors1.elements)
        ors2.tombstones = copy.copy(ors1.tombstones)

        # Node a adds again (new tag)
        ors1.add('x')
        # Node b removes (tombstones old tags only)
        ors2.remove('x')

        merged = ors1.merge(ors2)
        self.assertTrue(merged.lookup('x'))  # new tag from ors1 survives

    def test_merge_commutative(self):
        ors1 = ORSet('a')
        ors2 = ORSet('b')
        ors1.add('x')
        ors2.add('y')
        m1 = ors1.merge(ors2)
        m2 = ors2.merge(ors1)
        self.assertEqual(m1.value, m2.value)

    def test_value(self):
        ors = ORSet('a')
        ors.add('x')
        ors.add('y')
        ors.remove('x')
        self.assertEqual(ors.value, frozenset(['y']))


# =============================================================================
# LWWRegister Tests
# =============================================================================

class TestLWWRegister(unittest.TestCase):

    def test_initial_value(self):
        reg = LWWRegister('a')
        self.assertIsNone(reg.value)

    def test_set_get(self):
        reg = LWWRegister('a')
        reg.set('hello', timestamp=1)
        self.assertEqual(reg.value, 'hello')

    def test_last_writer_wins(self):
        reg1 = LWWRegister('a')
        reg2 = LWWRegister('b')
        reg1.set('first', timestamp=1)
        reg2.set('second', timestamp=2)
        merged = reg1.merge(reg2)
        self.assertEqual(merged.value, 'second')

    def test_merge_commutative(self):
        reg1 = LWWRegister('a')
        reg2 = LWWRegister('b')
        reg1.set('x', timestamp=1)
        reg2.set('y', timestamp=2)
        self.assertEqual(reg1.merge(reg2).value, reg2.merge(reg1).value)

    def test_tiebreak_by_node_id(self):
        reg1 = LWWRegister('a')
        reg2 = LWWRegister('b')
        reg1.set('x', timestamp=5)
        reg2.set('y', timestamp=5)
        merged = reg1.merge(reg2)
        # 'b' > 'a' in node_id, so b wins
        self.assertEqual(merged.value, 'y')


# =============================================================================
# MVRegister Tests
# =============================================================================

class TestMVRegister(unittest.TestCase):

    def test_single_value(self):
        mv = MVRegister('a')
        mv.set('hello')
        self.assertEqual(mv.value, frozenset(['hello']))

    def test_sequential_writes(self):
        mv = MVRegister('a')
        mv.set('first')
        mv.set('second')
        self.assertEqual(mv.value, frozenset(['second']))

    def test_concurrent_writes_preserved(self):
        mv1 = MVRegister('a')
        mv2 = MVRegister('b')
        mv1.set('x')
        mv2.set('y')
        merged = mv1.merge(mv2)
        self.assertEqual(merged.value, frozenset(['x', 'y']))

    def test_merge_after_resolve(self):
        mv1 = MVRegister('a')
        mv2 = MVRegister('b')
        mv1.set('x')
        mv2.set('y')
        merged = mv1.merge(mv2)
        # Now someone resolves the conflict
        merged.set('resolved')
        self.assertEqual(merged.value, frozenset(['resolved']))


# =============================================================================
# LWWElementSet Tests
# =============================================================================

class TestLWWElementSet(unittest.TestCase):

    def test_add_lookup(self):
        lww = LWWElementSet()
        lww.add('x', timestamp=1)
        self.assertTrue(lww.lookup('x'))

    def test_remove(self):
        lww = LWWElementSet()
        lww.add('x', timestamp=1)
        lww.remove('x', timestamp=2)
        self.assertFalse(lww.lookup('x'))

    def test_add_wins_on_same_timestamp(self):
        lww = LWWElementSet()
        lww.add('x', timestamp=5)
        lww.remove('x', timestamp=5)
        # add timestamp NOT > remove timestamp, so lookup is false
        self.assertFalse(lww.lookup('x'))

    def test_readd_after_remove(self):
        lww = LWWElementSet()
        lww.add('x', timestamp=1)
        lww.remove('x', timestamp=2)
        lww.add('x', timestamp=3)
        self.assertTrue(lww.lookup('x'))

    def test_merge(self):
        lww1 = LWWElementSet()
        lww2 = LWWElementSet()
        lww1.add('x', timestamp=1)
        lww2.remove('x', timestamp=2)
        merged = lww1.merge(lww2)
        self.assertFalse(merged.lookup('x'))

    def test_merge_concurrent(self):
        lww1 = LWWElementSet()
        lww2 = LWWElementSet()
        lww1.add('x', timestamp=3)
        lww2.remove('x', timestamp=2)
        merged = lww1.merge(lww2)
        self.assertTrue(merged.lookup('x'))  # add at 3 > remove at 2


# =============================================================================
# RGA Tests
# =============================================================================

class TestRGA(unittest.TestCase):

    def test_empty(self):
        rga = RGA('a')
        self.assertEqual(rga.value, [])

    def test_append(self):
        rga = RGA('a')
        rga.append('x')
        rga.append('y')
        rga.append('z')
        self.assertEqual(rga.value, ['x', 'y', 'z'])

    def test_insert_at_position(self):
        rga = RGA('a')
        rga.append('a')
        rga.append('c')
        rga.insert(1, 'b')
        self.assertEqual(rga.value, ['a', 'b', 'c'])

    def test_insert_at_start(self):
        rga = RGA('a')
        rga.append('b')
        rga.insert(0, 'a')
        self.assertEqual(rga.value, ['a', 'b'])

    def test_delete(self):
        rga = RGA('a')
        rga.append('a')
        rga.append('b')
        rga.append('c')
        rga.delete(1)  # delete 'b'
        self.assertEqual(rga.value, ['a', 'c'])

    def test_delete_out_of_range(self):
        rga = RGA('a')
        rga.append('a')
        with self.assertRaises(IndexError):
            rga.delete(5)

    def test_merge_disjoint(self):
        rga1 = RGA('a')
        rga2 = RGA('b')
        rga1.append('x')
        rga2.append('y')
        merged = rga1.merge(rga2)
        self.assertEqual(len(merged.value), 2)
        self.assertIn('x', merged.value)
        self.assertIn('y', merged.value)

    def test_merge_with_tombstone(self):
        rga1 = RGA('a')
        rga1.append('x')
        rga1.append('y')
        rga2 = copy.deepcopy(rga1)
        rga2.delete(0)  # delete 'x'
        merged = rga1.merge(rga2)
        self.assertEqual(merged.value, ['y'])


# =============================================================================
# OpCounter Tests
# =============================================================================

class TestOpCounter(unittest.TestCase):

    def test_increment(self):
        oc = OpCounter('a')
        oc.increment(5)
        self.assertEqual(oc.value, 5)

    def test_decrement(self):
        oc = OpCounter('a')
        oc.increment(10)
        oc.decrement(3)
        self.assertEqual(oc.value, 7)

    def test_apply_remote_op(self):
        oc1 = OpCounter('a')
        oc2 = OpCounter('b')
        op = oc1.increment(5)
        oc2.apply_op(op)
        self.assertEqual(oc2.value, 5)

    def test_bidirectional_ops(self):
        oc1 = OpCounter('a')
        oc2 = OpCounter('b')
        op1 = oc1.increment(3)
        op2 = oc2.increment(7)
        oc1.apply_op(op2)
        oc2.apply_op(op1)
        self.assertEqual(oc1.value, 10)
        self.assertEqual(oc2.value, 10)


# =============================================================================
# OpSet Tests
# =============================================================================

class TestOpSet(unittest.TestCase):

    def test_add(self):
        os1 = OpSet('a')
        os1.add('x')
        self.assertEqual(os1.value, frozenset(['x']))

    def test_remove(self):
        os1 = OpSet('a')
        os1.add('x')
        os1.remove('x')
        self.assertEqual(os1.value, frozenset())

    def test_apply_remote_add(self):
        os1 = OpSet('a')
        os2 = OpSet('b')
        op = os1.add('x')
        os2.apply_op(op)
        self.assertEqual(os2.value, frozenset(['x']))

    def test_apply_remote_remove(self):
        os1 = OpSet('a')
        os2 = OpSet('b')
        op_add = os1.add('x')
        os2.apply_op(op_add)
        op_rem = os1.remove('x')
        os2.apply_op(op_rem)
        self.assertEqual(os2.value, frozenset())


# =============================================================================
# DeltaGCounter Tests
# =============================================================================

class TestDeltaGCounter(unittest.TestCase):

    def test_increment_returns_delta(self):
        dgc = DeltaGCounter('a')
        delta = dgc.increment(5)
        self.assertEqual(delta, {'a': 5})

    def test_apply_delta(self):
        dgc1 = DeltaGCounter('a')
        dgc2 = DeltaGCounter('b')
        delta = dgc1.increment(3)
        dgc2.apply_delta(delta)
        self.assertEqual(dgc2.value, 3)

    def test_multiple_deltas(self):
        dgc1 = DeltaGCounter('a')
        dgc2 = DeltaGCounter('b')
        d1 = dgc1.increment(3)
        d2 = dgc1.increment(2)
        dgc2.apply_delta(d1)
        dgc2.apply_delta(d2)
        self.assertEqual(dgc2.value, 5)

    def test_idempotent_delta(self):
        dgc1 = DeltaGCounter('a')
        dgc2 = DeltaGCounter('b')
        delta = dgc1.increment(5)
        dgc2.apply_delta(delta)
        dgc2.apply_delta(delta)  # duplicate
        self.assertEqual(dgc2.value, 5)  # idempotent

    def test_merge(self):
        dgc1 = DeltaGCounter('a')
        dgc2 = DeltaGCounter('b')
        dgc1.increment(3)
        dgc2.increment(7)
        merged = dgc1.merge(dgc2)
        self.assertEqual(merged.value, 10)


# =============================================================================
# DeltaPNCounter Tests
# =============================================================================

class TestDeltaPNCounter(unittest.TestCase):

    def test_increment_decrement(self):
        dpn = DeltaPNCounter('a')
        dpn.increment(10)
        dpn.decrement(3)
        self.assertEqual(dpn.value, 7)

    def test_apply_delta(self):
        dpn1 = DeltaPNCounter('a')
        dpn2 = DeltaPNCounter('b')
        d1 = dpn1.increment(5)
        d2 = dpn1.decrement(2)
        dpn2.apply_delta(d1)
        dpn2.apply_delta(d2)
        self.assertEqual(dpn2.value, 3)

    def test_merge(self):
        dpn1 = DeltaPNCounter('a')
        dpn2 = DeltaPNCounter('b')
        dpn1.increment(10)
        dpn2.decrement(4)
        merged = dpn1.merge(dpn2)
        self.assertEqual(merged.value, 6)


# =============================================================================
# CRDTMap Tests
# =============================================================================

class TestCRDTMap(unittest.TestCase):

    def test_set_get(self):
        m = CRDTMap('a')
        gc = GCounter('a')
        gc.increment(5)
        m.set('counter', gc)
        retrieved = m.get('counter')
        self.assertEqual(retrieved.value, 5)

    def test_remove(self):
        m = CRDTMap('a')
        gc = GCounter('a')
        m.set('counter', gc)
        m.remove('counter')
        self.assertIsNone(m.get('counter'))

    def test_keys(self):
        m = CRDTMap('a')
        m.set('a', GCounter('a'))
        m.set('b', GCounter('a'))
        m.remove('b')
        self.assertEqual(m.keys(), frozenset(['a']))

    def test_merge(self):
        m1 = CRDTMap('a')
        m2 = CRDTMap('b')
        gc1 = GCounter('a')
        gc1.increment(3)
        gc2 = GCounter('b')
        gc2.increment(5)
        m1.set('counter', gc1)
        m2.set('counter', gc2)
        merged = m1.merge(m2)
        self.assertEqual(merged.get('counter').value, 8)

    def test_merge_different_keys(self):
        m1 = CRDTMap('a')
        m2 = CRDTMap('b')
        gc1 = GCounter('a')
        gc1.increment(1)
        gc2 = GCounter('b')
        gc2.increment(2)
        m1.set('x', gc1)
        m2.set('y', gc2)
        merged = m1.merge(m2)
        self.assertEqual(merged.keys(), frozenset(['x', 'y']))

    def test_value(self):
        m = CRDTMap('a')
        gc = GCounter('a')
        gc.increment(10)
        m.set('count', gc)
        self.assertEqual(m.value, {'count': 10})


# =============================================================================
# Network Simulation Tests
# =============================================================================

class TestCRDTNetwork(unittest.TestCase):

    def test_gcounter_convergence(self):
        net = CRDTNetwork()
        net.add_node('a', GCounter('a'))
        net.add_node('b', GCounter('b'))
        net.add_node('c', GCounter('c'))

        net.nodes['a'].increment(3)
        net.nodes['b'].increment(5)
        net.nodes['c'].increment(7)

        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.nodes['a'].value, 15)

    def test_pncounter_convergence(self):
        net = CRDTNetwork()
        net.add_node('a', PNCounter('a'))
        net.add_node('b', PNCounter('b'))

        net.nodes['a'].increment(10)
        net.nodes['b'].decrement(3)

        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.nodes['a'].value, 7)

    def test_partition_and_heal(self):
        net = CRDTNetwork()
        net.add_node('a', GCounter('a'))
        net.add_node('b', GCounter('b'))

        # Partition
        net.partition('a', 'b')
        net.nodes['a'].increment(5)
        net.nodes['b'].increment(3)

        net.sync_all()
        # Still partitioned -- no convergence
        self.assertEqual(net.nodes['a'].value, 5)
        self.assertEqual(net.nodes['b'].value, 3)

        # Heal
        net.heal('a', 'b')
        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.nodes['a'].value, 8)

    def test_orset_convergence(self):
        net = CRDTNetwork()
        net.add_node('a', ORSet('a'))
        net.add_node('b', ORSet('b'))

        net.nodes['a'].add('x')
        net.nodes['b'].add('y')

        net.sync_all()
        self.assertTrue(net.converged())
        vals = net.nodes['a'].value
        self.assertIn('x', vals)
        self.assertIn('y', vals)

    def test_three_node_convergence(self):
        net = CRDTNetwork()
        net.add_node('a', GSet())
        net.add_node('b', GSet())
        net.add_node('c', GSet())

        net.nodes['a'].add('x')
        net.nodes['b'].add('y')
        net.nodes['c'].add('z')

        net.sync_all()
        self.assertTrue(net.converged())
        expected = frozenset(['x', 'y', 'z'])
        self.assertEqual(net.nodes['a'].value, expected)

    def test_delta_broadcast(self):
        net = CRDTNetwork()
        net.add_node('a', DeltaGCounter('a'))
        net.add_node('b', DeltaGCounter('b'))

        delta = net.nodes['a'].increment(5)
        net.broadcast_delta('a', delta)
        net.deliver_all()

        self.assertEqual(net.nodes['b'].value, 5)

    def test_op_broadcast(self):
        net = CRDTNetwork()
        net.add_node('a', OpCounter('a'))
        net.add_node('b', OpCounter('b'))

        op = net.nodes['a'].increment(7)
        net.broadcast_op('a', op)
        net.deliver_all()

        self.assertEqual(net.nodes['b'].value, 7)

    def test_partition_three_nodes(self):
        """A-B partitioned, B-C connected, A-C connected."""
        net = CRDTNetwork()
        net.add_node('a', GCounter('a'))
        net.add_node('b', GCounter('b'))
        net.add_node('c', GCounter('c'))

        net.partition('a', 'b')
        net.nodes['a'].increment(1)
        net.nodes['b'].increment(2)
        net.nodes['c'].increment(3)

        # First sync: a can't reach b, b can't reach a
        net.sync_all()
        # c should have all three (connected to both a and b)
        self.assertEqual(net.nodes['c'].value, 6)

        # Heal and sync again
        net.heal()
        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.nodes['a'].value, 6)


# =============================================================================
# Factory Tests
# =============================================================================

class TestFactory(unittest.TestCase):

    def test_create_gcounter(self):
        gc = create_crdt('gcounter', 'a')
        gc.increment(5)
        self.assertEqual(gc.value, 5)

    def test_create_pncounter(self):
        pn = create_crdt('pncounter', 'a')
        pn.increment(3)
        self.assertEqual(pn.value, 3)

    def test_create_gset(self):
        gs = create_crdt('gset')
        gs.add('x')
        self.assertTrue(gs.lookup('x'))

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            create_crdt('nonexistent')


# =============================================================================
# Convergence Property Tests
# =============================================================================

class TestConvergenceProperties(unittest.TestCase):
    """Test mathematical properties that CRDTs must satisfy."""

    def test_gcounter_semilattice(self):
        """LUB (merge) must be commutative, associative, idempotent."""
        a = GCounter('a')
        b = GCounter('b')
        c = GCounter('c')
        a.increment(1)
        b.increment(2)
        c.increment(3)

        # Commutative
        self.assertEqual(a.merge(b).value, b.merge(a).value)
        # Associative
        self.assertEqual(a.merge(b).merge(c).value, a.merge(b.merge(c)).value)
        # Idempotent
        self.assertEqual(a.merge(a).value, a.value)

    def test_gset_semilattice(self):
        a = GSet()
        b = GSet()
        c = GSet()
        a.add('x')
        b.add('y')
        c.add('z')

        self.assertEqual(a.merge(b).value, b.merge(a).value)
        self.assertEqual(a.merge(b).merge(c).value, a.merge(b.merge(c)).value)
        self.assertEqual(a.merge(a).value, a.value)

    def test_pncounter_semilattice(self):
        a = PNCounter('a')
        b = PNCounter('b')
        c = PNCounter('c')
        a.increment(5)
        b.decrement(3)
        c.increment(1)
        c.decrement(2)

        self.assertEqual(a.merge(b).value, b.merge(a).value)
        self.assertEqual(a.merge(b).merge(c).value, a.merge(b.merge(c)).value)
        self.assertEqual(a.merge(a).value, a.value)

    def test_orset_concurrent_add_remove_add_wins(self):
        """The add-wins semantics of ORSet under concurrent operations."""
        # Replica 1 and 2 both see element x
        r1 = ORSet('r1')
        r1.add('x')
        r2 = copy.deepcopy(r1)
        r2.node_id = 'r2'

        # r1 removes x, r2 adds x again concurrently
        r1.remove('x')
        r2.add('x')

        merged = r1.merge(r2)
        # Add wins: the new tag from r2 survives r1's tombstone
        self.assertTrue(merged.lookup('x'))

    def test_lww_register_total_order(self):
        """LWW register resolves to most recent write."""
        r1 = LWWRegister('a')
        r2 = LWWRegister('b')
        r3 = LWWRegister('c')
        r1.set('first', timestamp=1)
        r2.set('second', timestamp=3)
        r3.set('third', timestamp=2)
        merged = r1.merge(r2).merge(r3)
        self.assertEqual(merged.value, 'second')

    def test_mvregister_conflict_detection(self):
        """MVRegister detects and preserves concurrent writes."""
        r1 = MVRegister('a')
        r2 = MVRegister('b')
        r1.set('x')
        r2.set('y')
        merged = r1.merge(r2)
        self.assertEqual(len(merged.value), 2)  # Both preserved

    def test_eventual_convergence_stress(self):
        """Multiple nodes with random operations all converge after sync."""
        net = CRDTNetwork()
        for i in range(5):
            net.add_node(str(i), GCounter(str(i)))

        # Random increments
        for _ in range(20):
            node = str(random.randint(0, 4))
            net.nodes[node].increment(random.randint(1, 10))

        # Full sync
        net.sync_all()
        net.sync_all()  # second pass for transitivity
        self.assertTrue(net.converged())

    def test_partition_convergence_stress(self):
        """Nodes converge even after partitions are healed."""
        net = CRDTNetwork()
        for i in range(4):
            net.add_node(str(i), PNCounter(str(i)))

        # Partition 0-1 from 2-3
        net.partition('0', '2')
        net.partition('0', '3')
        net.partition('1', '2')
        net.partition('1', '3')

        net.nodes['0'].increment(10)
        net.nodes['1'].decrement(3)
        net.nodes['2'].increment(5)
        net.nodes['3'].decrement(2)

        net.sync_all()
        # Partitioned -- should NOT converge
        self.assertFalse(net.converged())

        # Heal all
        net.heal()
        net.sync_all()
        net.sync_all()
        self.assertTrue(net.converged())
        self.assertEqual(net.nodes['0'].value, 10)  # 10-3+5-2


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases(unittest.TestCase):

    def test_gcounter_zero_increment(self):
        gc = GCounter('a')
        gc.increment(0)
        self.assertEqual(gc.value, 0)

    def test_empty_merge(self):
        gc1 = GCounter('a')
        gc2 = GCounter('b')
        merged = gc1.merge(gc2)
        self.assertEqual(merged.value, 0)

    def test_orset_empty_remove(self):
        ors = ORSet('a')
        ors.remove('x')  # no-op on nonexistent
        self.assertEqual(ors.value, frozenset())

    def test_rga_empty_merge(self):
        rga1 = RGA('a')
        rga2 = RGA('b')
        merged = rga1.merge(rga2)
        self.assertEqual(merged.value, [])

    def test_twopset_value(self):
        tps = TwoPSet()
        tps.add('a')
        tps.add('b')
        tps.add('c')
        tps.remove('b')
        self.assertEqual(tps.value, frozenset(['a', 'c']))

    def test_lww_element_set_multiple_elements(self):
        lww = LWWElementSet()
        lww.add('a', timestamp=1)
        lww.add('b', timestamp=2)
        lww.add('c', timestamp=3)
        lww.remove('b', timestamp=4)
        self.assertEqual(lww.value, frozenset(['a', 'c']))

    def test_crdt_map_nested(self):
        """Nested CRDTMap with counters inside."""
        outer1 = CRDTMap('a')
        outer2 = CRDTMap('b')
        gc1 = GCounter('a')
        gc1.increment(3)
        gc2 = GCounter('b')
        gc2.increment(7)
        outer1.set('count', gc1)
        outer2.set('count', gc2)
        merged = outer1.merge(outer2)
        self.assertEqual(merged.get('count').value, 10)

    def test_network_heal_all(self):
        net = CRDTNetwork()
        net.add_node('a', GCounter('a'))
        net.add_node('b', GCounter('b'))
        net.partition('a', 'b')
        net.heal()  # heal all
        self.assertTrue(net.can_communicate('a', 'b'))

    def test_vector_clock_empty_merge(self):
        vc1 = VectorClock()
        vc2 = VectorClock()
        merged = vc1.merge(vc2)
        self.assertEqual(merged.clock, {})

    def test_delta_counter_bidirectional(self):
        d1 = DeltaGCounter('a')
        d2 = DeltaGCounter('b')
        delta_a = d1.increment(5)
        delta_b = d2.increment(3)
        d1.apply_delta(delta_b)
        d2.apply_delta(delta_a)
        self.assertEqual(d1.value, 8)
        self.assertEqual(d2.value, 8)


# =============================================================================
# Integration / Scenario Tests
# =============================================================================

class TestScenarios(unittest.TestCase):

    def test_distributed_counter_scenario(self):
        """Three data centers independently counting, then syncing."""
        dc1 = GCounter('dc1')
        dc2 = GCounter('dc2')
        dc3 = GCounter('dc3')

        # Each DC counts independently
        for _ in range(100):
            dc1.increment()
        for _ in range(200):
            dc2.increment()
        for _ in range(150):
            dc3.increment()

        # Merge pairwise (simulating gossip)
        dc1 = dc1.merge(dc2)
        dc2 = dc2.merge(dc1)
        dc3 = dc3.merge(dc1)
        dc1 = dc1.merge(dc3)
        dc2 = dc2.merge(dc3)

        self.assertEqual(dc1.value, 450)
        self.assertEqual(dc2.value, 450)
        self.assertEqual(dc3.value, 450)

    def test_collaborative_editing_scenario(self):
        """Two users editing a shared document (RGA)."""
        doc1 = RGA('alice')

        # Alice writes 'hello'
        for ch in 'hello':
            doc1.append(ch)
        self.assertEqual(doc1.value, list('hello'))

        # Sync to Bob via deepcopy (preserves node structure)
        doc2 = copy.deepcopy(doc1)
        doc2.node_id = 'bob'

        # Alice appends ' world'
        for ch in ' world':
            doc1.append(ch)
        self.assertEqual(doc1.value, list('hello world'))

        # Bob deletes 'h' (pos 0) and appends '!'
        doc2.delete(0)
        doc2.append('!')
        self.assertEqual(doc2.value, list('ello!'))

        # Merge
        merged = doc1.merge(doc2)
        val = merged.value
        # 'h' should be deleted (tombstoned in doc2)
        self.assertNotIn('h', val)
        # Both ' world' chars and '!' should be present
        self.assertIn('!', val)
        self.assertIn('w', val)

    def test_shopping_cart_scenario(self):
        """Concurrent shopping cart modifications using ORSet."""
        cart1 = ORSet('user_device1')
        cart2 = ORSet('user_device2')

        # Device 1 adds items
        cart1.add('apple')
        cart1.add('banana')

        # Sync to device 2
        cart2 = cart1.merge(cart2)

        # Device 1 removes banana
        cart1.remove('banana')
        # Device 2 adds cherry (concurrent with remove)
        cart2.add('cherry')

        merged = cart1.merge(cart2)
        self.assertIn('apple', merged.value)
        self.assertNotIn('banana', merged.value)
        self.assertIn('cherry', merged.value)

    def test_last_writer_wins_config(self):
        """Config settings using LWW registers."""
        config1 = CRDTMap('server1')
        config2 = CRDTMap('server2')

        reg1 = LWWRegister('server1')
        reg1.set('dark', timestamp=1)
        config1.set('theme', reg1)

        reg2 = LWWRegister('server2')
        reg2.set('light', timestamp=2)
        config2.set('theme', reg2)

        merged = config1.merge(config2)
        self.assertEqual(merged.get('theme').value, 'light')

    def test_multi_value_conflict_tracking(self):
        """Track conflicting edits with MVRegister."""
        doc1 = MVRegister('editor1')
        doc2 = MVRegister('editor2')

        doc1.set("version A")
        doc2.set("version B")

        merged = doc1.merge(doc2)
        self.assertEqual(len(merged.value), 2)
        self.assertIn("version A", merged.value)
        self.assertIn("version B", merged.value)

    def test_network_partition_and_recovery(self):
        """Full scenario: work during partition, converge after heal."""
        net = CRDTNetwork()
        for i in range(3):
            net.add_node(str(i), ORSet(str(i)))

        # All add 'shared'
        for i in range(3):
            net.nodes[str(i)].add('shared')
        net.sync_all()

        # Partition node 0
        net.partition('0', '1')
        net.partition('0', '2')

        # Node 0 adds 'isolated_item'
        net.nodes['0'].add('isolated_item')
        # Nodes 1,2 add 'majority_item'
        net.nodes['1'].add('majority_item')
        net.nodes['2'].add('majority_item')

        net.sync_all()
        # Node 0 shouldn't have majority_item
        self.assertNotIn('majority_item', net.nodes['0'].value)
        # Nodes 1,2 shouldn't have isolated_item
        self.assertNotIn('isolated_item', net.nodes['1'].value)

        # Heal
        net.heal()
        net.sync_all()
        self.assertTrue(net.converged())
        all_expected = frozenset(['shared', 'isolated_item', 'majority_item'])
        self.assertEqual(net.nodes['0'].value, all_expected)


if __name__ == '__main__':
    unittest.main()
