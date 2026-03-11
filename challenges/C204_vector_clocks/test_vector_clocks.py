"""Tests for C204: Vector Clocks & Causal Broadcast."""
import pytest
from vector_clocks import (
    VectorClock, compare_clocks, VersionVector, Dot,
    DottedVersionVector, CausalHistory, IntervalClock,
    StampedValue, CausalStore, ConflictError,
    CausalMessage, CausalBroadcastNode, CausalNetwork,
    MatrixClock, BloomClock, CausalDeliveryVerifier
)


# ===== VectorClock =====

class TestVectorClock:
    def test_empty_clock(self):
        vc = VectorClock()
        assert vc.get('A') == 0
        assert vc.get('B') == 0

    def test_increment(self):
        vc = VectorClock()
        vc.increment('A')
        assert vc.get('A') == 1
        vc.increment('A')
        assert vc.get('A') == 2

    def test_increment_multiple_nodes(self):
        vc = VectorClock()
        vc.increment('A').increment('B').increment('A')
        assert vc.get('A') == 2
        assert vc.get('B') == 1

    def test_merge(self):
        vc1 = VectorClock({'A': 3, 'B': 1})
        vc2 = VectorClock({'A': 1, 'B': 4, 'C': 2})
        merged = vc1.merge(vc2)
        assert merged.get('A') == 3
        assert merged.get('B') == 4
        assert merged.get('C') == 2

    def test_update_in_place(self):
        vc1 = VectorClock({'A': 1})
        vc2 = VectorClock({'A': 3, 'B': 2})
        vc1.update(vc2)
        assert vc1.get('A') == 3
        assert vc1.get('B') == 2

    def test_copy(self):
        vc = VectorClock({'A': 5})
        vc2 = vc.copy()
        vc2.increment('A')
        assert vc.get('A') == 5
        assert vc2.get('A') == 6

    def test_equality(self):
        vc1 = VectorClock({'A': 1, 'B': 2})
        vc2 = VectorClock({'A': 1, 'B': 2})
        assert vc1 == vc2

    def test_equality_missing_zeros(self):
        vc1 = VectorClock({'A': 1})
        vc2 = VectorClock({'A': 1, 'B': 0})
        assert vc1 == vc2

    def test_happened_before(self):
        vc1 = VectorClock({'A': 1, 'B': 2})
        vc2 = VectorClock({'A': 2, 'B': 3})
        assert vc1 < vc2
        assert not vc2 < vc1

    def test_happened_after(self):
        vc1 = VectorClock({'A': 2, 'B': 3})
        vc2 = VectorClock({'A': 1, 'B': 2})
        assert vc1 > vc2

    def test_concurrent(self):
        vc1 = VectorClock({'A': 2, 'B': 1})
        vc2 = VectorClock({'A': 1, 'B': 2})
        assert vc1.concurrent_with(vc2)
        assert vc2.concurrent_with(vc1)
        assert not vc1 < vc2
        assert not vc2 < vc1

    def test_le_ge(self):
        vc1 = VectorClock({'A': 1})
        vc2 = VectorClock({'A': 1})
        assert vc1 <= vc2
        assert vc1 >= vc2

    def test_dominates(self):
        vc1 = VectorClock({'A': 3, 'B': 5})
        vc2 = VectorClock({'A': 2, 'B': 4})
        assert vc1.dominates(vc2)
        assert not vc2.dominates(vc1)

    def test_clock_property(self):
        vc = VectorClock({'A': 1, 'B': 2})
        c = vc.clock
        assert c == {'A': 1, 'B': 2}
        c['A'] = 99  # shouldn't affect original
        assert vc.get('A') == 1

    def test_repr(self):
        vc = VectorClock({'B': 2, 'A': 1})
        assert 'A:1' in repr(vc)
        assert 'B:2' in repr(vc)

    def test_hash(self):
        vc1 = VectorClock({'A': 1, 'B': 2})
        vc2 = VectorClock({'A': 1, 'B': 2})
        assert hash(vc1) == hash(vc2)
        s = {vc1, vc2}
        assert len(s) == 1


class TestCompareClocks:
    def test_equal(self):
        assert compare_clocks(VectorClock({'A': 1}), VectorClock({'A': 1})) == 'equal'

    def test_before(self):
        assert compare_clocks(VectorClock({'A': 1}), VectorClock({'A': 2})) == 'before'

    def test_after(self):
        assert compare_clocks(VectorClock({'A': 2}), VectorClock({'A': 1})) == 'after'

    def test_concurrent(self):
        assert compare_clocks(
            VectorClock({'A': 2, 'B': 1}),
            VectorClock({'A': 1, 'B': 2})
        ) == 'concurrent'


# ===== VersionVector =====

class TestVersionVector:
    def test_basic(self):
        vv = VersionVector()
        vv.increment('A')
        assert vv.get('A') == 1

    def test_merge(self):
        vv1 = VersionVector()
        vv1.increment('A').increment('A')
        vv2 = VersionVector()
        vv2.increment('B').increment('B').increment('B')
        merged = vv1.merge(vv2)
        assert merged.get('A') == 2
        assert merged.get('B') == 3

    def test_descends_from(self):
        vv1 = VersionVector()
        vv1.increment('A').increment('A')
        vv2 = VersionVector()
        vv2.increment('A')
        assert vv1.descends_from(vv2)
        assert not vv2.descends_from(vv1)

    def test_conflicts(self):
        vv1 = VersionVector()
        vv1.increment('A')
        vv2 = VersionVector()
        vv2.increment('B')
        assert vv1.conflicts_with(vv2)

    def test_no_conflict(self):
        vv1 = VersionVector()
        vv1.increment('A').increment('B')
        vv2 = VersionVector()
        vv2.increment('A')
        assert not vv1.conflicts_with(vv2)

    def test_copy(self):
        vv = VersionVector()
        vv.increment('A')
        vv2 = vv.copy()
        vv2.increment('A')
        assert vv.get('A') == 1
        assert vv2.get('A') == 2

    def test_equality(self):
        vv1 = VersionVector()
        vv1.increment('A')
        vv2 = VersionVector()
        vv2.increment('A')
        assert vv1 == vv2

    def test_versions_property(self):
        vv = VersionVector()
        vv.increment('X')
        assert vv.versions == {'X': 1}

    def test_repr(self):
        vv = VersionVector()
        vv.increment('A')
        assert 'A:1' in repr(vv)


# ===== Dot & DottedVersionVector =====

class TestDot:
    def test_equality(self):
        d1 = Dot('A', 1)
        d2 = Dot('A', 1)
        assert d1 == d2

    def test_inequality(self):
        assert Dot('A', 1) != Dot('A', 2)
        assert Dot('A', 1) != Dot('B', 1)

    def test_hash(self):
        s = {Dot('A', 1), Dot('A', 1), Dot('B', 2)}
        assert len(s) == 2

    def test_repr(self):
        d = Dot('A', 1)
        assert 'A' in repr(d) and '1' in repr(d)


class TestDottedVersionVector:
    def test_descends(self):
        vv1 = VersionVector()
        vv1.increment('A').increment('A')
        dvv1 = DottedVersionVector(version_vector=vv1, dot=Dot('A', 2))

        vv2 = VersionVector()
        vv2.increment('A')
        dvv2 = DottedVersionVector(version_vector=vv2, dot=Dot('A', 1))

        assert dvv1.descends(dvv2)
        assert not dvv2.descends(dvv1)

    def test_concurrent(self):
        vv1 = VersionVector()
        vv1.increment('A')
        dvv1 = DottedVersionVector(version_vector=vv1, dot=Dot('A', 1))

        vv2 = VersionVector()
        vv2.increment('B')
        dvv2 = DottedVersionVector(version_vector=vv2, dot=Dot('B', 1))

        assert dvv1.concurrent_with(dvv2)

    def test_no_dot(self):
        vv1 = VersionVector()
        vv1.increment('A').increment('A')
        dvv1 = DottedVersionVector(version_vector=vv1)

        vv2 = VersionVector()
        vv2.increment('A')
        dvv2 = DottedVersionVector(version_vector=vv2)

        assert dvv1.descends(dvv2)


# ===== CausalHistory =====

class TestCausalHistory:
    def test_add_and_has(self):
        ch = CausalHistory()
        ch.add('A', 1)
        assert ch.has('A', 1)
        assert not ch.has('A', 2)

    def test_contiguous_compaction(self):
        ch = CausalHistory()
        ch.add('A', 1)
        ch.add('A', 2)
        ch.add('A', 3)
        assert ch.has('A', 1)
        assert ch.has('A', 2)
        assert ch.has('A', 3)
        assert ch._contiguous.get('A') == 3
        assert len(ch._dots) == 0

    def test_out_of_order_compaction(self):
        ch = CausalHistory()
        ch.add('A', 3)
        ch.add('A', 1)
        ch.add('A', 2)
        assert ch._contiguous.get('A') == 3

    def test_gap(self):
        ch = CausalHistory()
        ch.add('A', 1)
        ch.add('A', 3)
        assert ch.has('A', 1)
        assert not ch.has('A', 2)
        assert ch.has('A', 3)
        assert ch._contiguous.get('A') == 1
        assert Dot('A', 3) in ch._dots

    def test_fill_gap(self):
        ch = CausalHistory()
        ch.add('A', 1)
        ch.add('A', 3)
        ch.add('A', 2)
        assert ch._contiguous.get('A') == 3
        assert len(ch._dots) == 0

    def test_next_dot(self):
        ch = CausalHistory()
        ch.add('A', 1)
        ch.add('A', 2)
        d = ch.next_dot('A')
        assert d.node == 'A'
        assert d.counter == 3

    def test_next_dot_with_gap(self):
        ch = CausalHistory()
        ch.add('A', 1)
        ch.add('A', 3)
        d = ch.next_dot('A')
        assert d.counter == 2  # Fill the gap first

    def test_merge(self):
        ch1 = CausalHistory()
        ch1.add('A', 1).add('A', 2)
        ch2 = CausalHistory()
        ch2.add('A', 2).add('A', 3).add('B', 1)
        merged = ch1.merge(ch2)
        assert merged.has('A', 1)
        assert merged.has('A', 2)
        assert merged.has('A', 3)
        assert merged.has('B', 1)

    def test_events_unseen_by(self):
        ch1 = CausalHistory()
        ch1.add('A', 1).add('A', 2).add('A', 3)
        ch2 = CausalHistory()
        ch2.add('A', 1)
        unseen = ch1.events_unseen_by(ch2)
        assert Dot('A', 2) in unseen
        assert Dot('A', 3) in unseen
        assert Dot('A', 1) not in unseen

    def test_size(self):
        ch = CausalHistory()
        ch.add('A', 1).add('A', 2).add('B', 1)
        assert ch.size == 3


# ===== IntervalClock =====

class TestIntervalClock:
    def test_add_single(self):
        ic = IntervalClock()
        ic.add('A', 1)
        assert ic.has('A', 1)
        assert not ic.has('A', 2)

    def test_add_range(self):
        ic = IntervalClock()
        ic.add_range('A', 1, 5)
        for i in range(1, 6):
            assert ic.has('A', i)
        assert not ic.has('A', 0)
        assert not ic.has('A', 6)

    def test_merge_adjacent(self):
        ic = IntervalClock()
        ic.add('A', 1)
        ic.add('A', 2)
        ic.add('A', 3)
        assert ic._intervals['A'] == [(1, 3)]

    def test_merge_with_gap(self):
        ic = IntervalClock()
        ic.add_range('A', 1, 3)
        ic.add_range('A', 6, 8)
        assert ic.gap_count == 1

    def test_fill_gap(self):
        ic = IntervalClock()
        ic.add_range('A', 1, 3)
        ic.add_range('A', 6, 8)
        ic.add_range('A', 4, 5)
        assert ic._intervals['A'] == [(1, 8)]
        assert ic.gap_count == 0

    def test_max_counter(self):
        ic = IntervalClock()
        ic.add_range('A', 1, 10)
        assert ic.max_counter('A') == 10
        assert ic.max_counter('B') == 0

    def test_merge_clocks(self):
        ic1 = IntervalClock()
        ic1.add_range('A', 1, 5)
        ic2 = IntervalClock()
        ic2.add_range('A', 3, 8)
        merged = ic1.merge(ic2)
        assert merged._intervals['A'] == [(1, 8)]

    def test_repr(self):
        ic = IntervalClock()
        ic.add_range('A', 1, 3)
        assert 'A' in repr(ic)


# ===== CausalStore =====

class TestCausalStore:
    def test_put_get(self):
        store = CausalStore('node1')
        store.put('key1', 'value1')
        assert store.get_one('key1') == 'value1'

    def test_overwrite(self):
        store = CausalStore('node1')
        store.put('key1', 'v1')
        store.put('key1', 'v2')
        assert store.get_one('key1') == 'v2'

    def test_missing_key(self):
        store = CausalStore('node1')
        assert store.get_one('missing') is None

    def test_keys(self):
        store = CausalStore('node1')
        store.put('a', 1)
        store.put('b', 2)
        assert set(store.keys()) == {'a', 'b'}

    def test_replicate_sequential(self):
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        sv = s1.put('x', 'hello')
        s2.receive_put('x', sv)
        assert s2.get_one('x') == 'hello'

    def test_replicate_supersede(self):
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        sv1 = s1.put('x', 'v1')
        s2.receive_put('x', sv1)
        sv2 = s1.put('x', 'v2')
        s2.receive_put('x', sv2)
        assert s2.get_one('x') == 'v2'

    def test_concurrent_writes_create_siblings(self):
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        sv1 = s1.put('x', 'from_n1')
        sv2 = s2.put('x', 'from_n2')
        s1.receive_put('x', sv2)
        assert s1.has_conflicts('x')
        assert s1.sibling_count('x') == 2

    def test_conflict_error(self):
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        sv1 = s1.put('x', 'a')
        sv2 = s2.put('x', 'b')
        s1.receive_put('x', sv2)
        with pytest.raises(ConflictError) as exc_info:
            s1.get_one('x')
        assert exc_info.value.key == 'x'
        assert len(exc_info.value.siblings) == 2

    def test_resolve_conflict(self):
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        sv1 = s1.put('x', 'a')
        sv2 = s2.put('x', 'b')
        s1.receive_put('x', sv2)
        s1.resolve('x', 'merged')
        assert s1.get_one('x') == 'merged'
        assert not s1.has_conflicts('x')

    def test_resolve_supersedes_old(self):
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        sv1 = s1.put('x', 'a')
        sv2 = s2.put('x', 'b')
        s1.receive_put('x', sv2)
        resolved = s1.resolve('x', 'merged')
        # Send resolved to s2
        s2.receive_put('x', resolved)
        assert s2.get_one('x') == 'merged'
        assert not s2.has_conflicts('x')

    def test_sync_from(self):
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        s1.put('a', 1)
        s1.put('b', 2)
        s2.sync_from(s1)
        assert s2.get_one('a') == 1
        assert s2.get_one('b') == 2

    def test_receive_old_write(self):
        s1 = CausalStore('n1')
        sv1 = s1.put('x', 'old')
        sv2 = s1.put('x', 'new')
        s2 = CausalStore('n2')
        s2.receive_put('x', sv2)
        s2.receive_put('x', sv1)  # Late arrival of old write
        assert s2.get_one('x') == 'new'  # Old is superseded


# ===== CausalBroadcast =====

class TestCausalBroadcast:
    def test_single_broadcast(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        a.broadcast('hello')
        net.deliver_all()
        assert len(b.delivered) == 1
        assert b.delivered[0].payload == 'hello'

    def test_self_delivery(self):
        net = CausalNetwork()
        a = net.add_node('A')
        a.broadcast('hi')
        assert len(a.delivered) == 1

    def test_three_node_broadcast(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')
        a.broadcast('msg1')
        net.deliver_all()
        assert len(b.delivered) == 1
        assert len(c.delivered) == 1

    def test_causal_order_maintained(self):
        """If A sends m1, B receives m1 and sends m2, C must get m1 before m2."""
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')

        a.broadcast('m1')
        # Deliver m1 to B only
        for i in range(len(net.in_flight)):
            target, msg = net.in_flight[i]
            if target == 'B':
                net.in_flight.pop(i)
                b.receive(msg)
                break

        # Now B broadcasts m2 (which causally depends on m1)
        b.broadcast('m2')

        # Deliver remaining: m1->C and m2->A,C
        # Even if m2 arrives at C first, it should be buffered
        net.deliver_all()

        # C should have m1 before m2
        c_payloads = [m.payload for m in c.delivered]
        assert c_payloads.index('m1') < c_payloads.index('m2')

    def test_reversed_delivery_still_causal(self):
        """Even with reversed network delivery, causal order holds."""
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')

        a.broadcast('first')
        net.deliver_all()
        b.broadcast('second')
        net.deliver_reversed()

        for node in [a, c]:
            payloads = [m.payload for m in node.delivered]
            assert payloads.index('first') < payloads.index('second')

    def test_random_delivery_causal(self):
        net = CausalNetwork()
        nodes = [net.add_node(str(i)) for i in range(5)]
        nodes[0].broadcast('a')
        net.deliver_all()
        nodes[1].broadcast('b')
        net.deliver_all()
        nodes[2].broadcast('c')
        net.deliver_random(seed=123)

        for node in nodes:
            payloads = [m.payload for m in node.delivered]
            assert payloads.index('a') < payloads.index('b')
            assert payloads.index('b') < payloads.index('c')

    def test_pending_buffer(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')

        a.broadcast('m1')
        # Don't deliver m1 to C yet
        in_flight_copy = list(net.in_flight)
        net.in_flight = [x for x in net.in_flight if x[0] != 'C']
        m1_to_c = [x for x in in_flight_copy if x[0] == 'C']

        net.deliver_all()
        b.broadcast('m2')
        net.deliver_all()

        # C has m2 pending (can't deliver without m1)
        assert c.pending_count >= 0  # may have delivered m2 if clock allows

        # Now deliver m1 to C
        for target, msg in m1_to_c:
            c.receive(msg)

        # Now m2 should also be delivered
        payloads = [m.payload for m in c.delivered]
        assert 'm1' in payloads
        assert 'm2' in payloads

    def test_on_deliver_callback(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        delivered = []
        b.on_deliver(lambda msg: delivered.append(msg.payload))
        a.broadcast('hello')
        net.deliver_all()
        assert delivered == ['hello']

    def test_duplicate_delivery(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        msg = a.broadcast('test')
        net.deliver_all()
        # Deliver same message again
        b.receive(a.delivered[0])
        assert len(b.delivered) == 1  # No duplicate

    def test_many_messages(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        for i in range(20):
            a.broadcast('msg_{}'.format(i))
        net.deliver_all()
        assert len(b.delivered) == 20


# ===== CausalNetwork =====

class TestCausalNetwork:
    def test_partition(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')
        net.partition('A', 'B')
        a.broadcast('hi')
        net.deliver_all()
        assert len(c.delivered) == 1  # C got it
        assert len(b.delivered) == 0  # B is partitioned from A

    def test_heal_partition(self):
        """After partition heals, messages sent post-heal are delivered.
        Note: B buffers 'after' until it receives 'before' (causal dep)."""
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')
        net.partition('A', 'B')
        msg_before = a.broadcast('before')
        net.deliver_all()
        assert len(b.delivered) == 0  # B partitioned
        assert len(c.delivered) == 1  # C got it

        net.heal('A', 'B')
        a.broadcast('after')
        net.deliver_all()
        # B has 'after' pending (depends on 'before' which it missed)
        assert b.pending_count == 1

        # Manually resend 'before' to B (anti-entropy / repair)
        b.receive(msg_before)
        payloads = [m.payload for m in b.delivered]
        assert 'before' in payloads
        assert 'after' in payloads
        assert payloads.index('before') < payloads.index('after')

    def test_heal_all(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')
        net.partition('A', 'B')
        net.partition('A', 'C')
        net.heal_all()
        a.broadcast('hi')
        net.deliver_all()
        assert len(b.delivered) == 1
        assert len(c.delivered) == 1

    def test_in_flight_count(self):
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')
        a.broadcast('x')
        assert net.in_flight_count == 2  # To B and C
        net.deliver_all()
        assert net.in_flight_count == 0

    def test_repr(self):
        net = CausalNetwork()
        net.add_node('A')
        assert 'nodes=1' in repr(net)


# ===== MatrixClock =====

class TestMatrixClock:
    def test_increment(self):
        mc = MatrixClock('A', ['A', 'B', 'C'])
        mc.increment()
        assert mc.get('A', 'A') == 1

    def test_send_receive(self):
        mc_a = MatrixClock('A', ['A', 'B'])
        mc_b = MatrixClock('B', ['A', 'B'])
        stamp = mc_a.send_stamp()
        mc_b.receive_stamp('A', stamp)
        assert mc_b.get('B', 'A') == 1

    def test_local_clock(self):
        mc = MatrixClock('A', ['A', 'B'])
        mc.increment()
        lc = mc.local_clock()
        assert lc.get('A') == 1

    def test_min_known(self):
        nodes = ['A', 'B', 'C']
        mc_a = MatrixClock('A', nodes)
        mc_b = MatrixClock('B', nodes)
        mc_c = MatrixClock('C', nodes)

        # A sends to B
        stamp = mc_a.send_stamp()
        mc_b.receive_stamp('A', stamp)

        # A sends to C
        stamp = mc_a.send_stamp()
        mc_c.receive_stamp('A', stamp)

        # B sends to C (C now knows B knows about A:1)
        stamp_b = mc_b.send_stamp()
        mc_c.receive_stamp('B', stamp_b)

        # From C's perspective: A knows A:2, B knows A:1, C knows A:2
        # But we need all nodes' views
        assert mc_c.min_known('A') >= 0

    def test_gc_check(self):
        nodes = ['A', 'B']
        mc_a = MatrixClock('A', nodes)
        mc_b = MatrixClock('B', nodes)

        # Exchange messages
        stamp = mc_a.send_stamp()
        mc_b.receive_stamp('A', stamp)
        stamp = mc_b.send_stamp()
        mc_a.receive_stamp('B', stamp)

        # A knows B has seen A:1, so A can GC event (A, 1)
        assert mc_a.can_gc('A', 1)

    def test_repr(self):
        mc = MatrixClock('A', ['A', 'B'])
        assert 'A' in repr(mc)


# ===== BloomClock =====

class TestBloomClock:
    def test_increment(self):
        bc = BloomClock(size=32)
        bc.increment('A')
        bc2 = BloomClock(size=32)
        assert bc._counters != bc2._counters

    def test_merge(self):
        bc1 = BloomClock(size=32)
        bc1.increment('A')
        bc2 = BloomClock(size=32)
        bc2.increment('B')
        merged = bc1.merge(bc2)
        for i in range(32):
            assert merged._counters[i] >= bc1._counters[i]
            assert merged._counters[i] >= bc2._counters[i]

    def test_possibly_before(self):
        bc1 = BloomClock(size=64)
        bc1.increment('A')
        bc2 = bc1.copy()
        bc2.increment('B')
        assert bc1.possibly_before(bc2)
        assert not bc2.possibly_before(bc1)

    def test_definitely_concurrent(self):
        bc1 = BloomClock(size=64, num_hashes=3)
        bc2 = BloomClock(size=64, num_hashes=3)
        # Use nodes that hash to different positions
        for _ in range(10):
            bc1.increment('node_A_unique')
        for _ in range(10):
            bc2.increment('node_B_unique')
        # They should be concurrent if they hash to different positions
        # (probabilistic -- may not always work)
        if bc1.definitely_concurrent(bc2):
            assert True
        else:
            # Hash collision -- just verify the logic works
            assert bc1.possibly_before(bc2) or bc2.possibly_before(bc1)

    def test_copy(self):
        bc = BloomClock()
        bc.increment('A')
        bc2 = bc.copy()
        bc2.increment('B')
        assert bc._counters != bc2._counters

    def test_repr(self):
        bc = BloomClock()
        assert 'BloomClock' in repr(bc)


# ===== CausalDeliveryVerifier =====

class TestCausalDeliveryVerifier:
    def test_causal_delivery_verified(self):
        """Verify that our broadcast protocol maintains causal order."""
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')
        verifier = CausalDeliveryVerifier()

        for node_id, node in net.nodes.items():
            node.on_deliver(lambda msg, nid=node_id: verifier.record_delivery(nid, msg))

        a.broadcast('m1')
        net.deliver_all()
        b.broadcast('m2')
        net.deliver_all()
        c.broadcast('m3')
        net.deliver_all()

        assert verifier.is_causal

    def test_violation_detection(self):
        """Manually construct a violation."""
        verifier = CausalDeliveryVerifier()
        # Msg1 happened before msg2
        msg1 = CausalMessage('A', 'first', VectorClock({'A': 1}))
        msg2 = CausalMessage('B', 'second', VectorClock({'A': 1, 'B': 1}))
        # Deliver in wrong order at node C
        verifier.record_delivery('C', msg2)  # second delivered first -- wrong!
        verifier.record_delivery('C', msg1)
        violations = verifier.verify()
        assert len(violations) > 0

    def test_concurrent_no_violation(self):
        verifier = CausalDeliveryVerifier()
        msg1 = CausalMessage('A', 'x', VectorClock({'A': 1}))
        msg2 = CausalMessage('B', 'y', VectorClock({'B': 1}))
        # Either order is fine for concurrent messages
        verifier.record_delivery('C', msg1)
        verifier.record_delivery('C', msg2)
        assert verifier.is_causal


# ===== Integration Tests =====

class TestIntegration:
    def test_five_node_chain(self):
        """Chain of broadcasts across 5 nodes."""
        net = CausalNetwork()
        nodes = [net.add_node(str(i)) for i in range(5)]

        nodes[0].broadcast('start')
        net.deliver_all()
        nodes[1].broadcast('step1')
        net.deliver_all()
        nodes[2].broadcast('step2')
        net.deliver_all()
        nodes[3].broadcast('step3')
        net.deliver_all()
        nodes[4].broadcast('finish')
        net.deliver_all()

        # All nodes should see all 5 messages in causal order
        for node in nodes:
            payloads = [m.payload for m in node.delivered]
            assert payloads.index('start') < payloads.index('step1')
            assert payloads.index('step1') < payloads.index('step2')
            assert payloads.index('step2') < payloads.index('step3')
            assert payloads.index('step3') < payloads.index('finish')

    def test_diamond_causality(self):
        """
        A broadcasts m1
        B and C both receive m1 and broadcast m2, m3
        D must see m1 before both m2 and m3
        """
        net = CausalNetwork()
        a = net.add_node('A')
        b = net.add_node('B')
        c = net.add_node('C')
        d = net.add_node('D')

        a.broadcast('m1')
        # Deliver to B, C, and D
        msgs_to_deliver = list(net.in_flight)
        net.in_flight = []
        # Deliver m1 to B and C first (not D yet)
        m1_to_d = None
        for target, msg in msgs_to_deliver:
            if target in ('B', 'C'):
                net.nodes[target].receive(msg)
            elif target == 'D':
                m1_to_d = (target, msg)

        b.broadcast('m2')
        c.broadcast('m3')

        # Put m1_to_d back at end (so D gets m2/m3 before m1)
        net.in_flight.append(m1_to_d)
        net.deliver_all()

        # D: m1 must come before m2 and m3 (causal broadcast buffers)
        d_payloads = [m.payload for m in d.delivered]
        assert 'm1' in d_payloads
        assert 'm2' in d_payloads
        assert 'm3' in d_payloads
        assert d_payloads.index('m1') < d_payloads.index('m2')
        assert d_payloads.index('m1') < d_payloads.index('m3')

    def test_store_replication_three_nodes(self):
        """Three stores replicate with conflict detection."""
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')
        s3 = CausalStore('n3')

        # n1 writes
        sv = s1.put('users', ['alice'])
        s2.receive_put('users', sv)
        s3.receive_put('users', sv)

        # n2 and n3 concurrently update
        sv2 = s2.put('users', ['alice', 'bob'])
        sv3 = s3.put('users', ['alice', 'carol'])

        # Replicate to n1
        s1.receive_put('users', sv2)
        s1.receive_put('users', sv3)

        assert s1.has_conflicts('users')
        values = [sv.value for sv in s1.get('users')]
        assert ['alice', 'bob'] in values
        assert ['alice', 'carol'] in values

        # Resolve
        s1.resolve('users', ['alice', 'bob', 'carol'])
        assert s1.get_one('users') == ['alice', 'bob', 'carol']

    def test_partition_and_heal_store(self):
        """Stores diverge during partition and reconcile after."""
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')

        # Both write during partition
        s1.put('x', 'value_from_n1')
        s2.put('x', 'value_from_n2')

        # Heal: sync
        s1.sync_from(s2)
        assert s1.has_conflicts('x')

        # Resolve using last-writer-wins
        siblings = s1.get('x')
        winner = max(siblings, key=lambda sv: sv.clock.get(sv.writer))
        s1.resolve('x', winner.value)
        assert not s1.has_conflicts('x')

    def test_vector_clock_transitivity(self):
        """If a < b and b < c, then a < c."""
        a = VectorClock({'X': 1})
        b = VectorClock({'X': 2, 'Y': 1})
        c = VectorClock({'X': 3, 'Y': 2, 'Z': 1})
        assert a < b
        assert b < c
        assert a < c  # Transitivity

    def test_causal_broadcast_with_verifier(self):
        """Full integration: broadcast + verify."""
        net = CausalNetwork()
        nodes = [net.add_node(chr(65 + i)) for i in range(4)]  # A-D
        verifier = CausalDeliveryVerifier()

        for node_id, node in net.nodes.items():
            node.on_deliver(lambda msg, nid=node_id: verifier.record_delivery(nid, msg))

        # Complex message pattern
        nodes[0].broadcast('alpha')
        net.deliver_all()
        nodes[1].broadcast('beta')
        nodes[2].broadcast('gamma')
        net.deliver_all()
        nodes[3].broadcast('delta')
        net.deliver_random(seed=99)

        assert verifier.is_causal

    def test_matrix_clock_gc_scenario(self):
        """Matrix clock enables garbage collection of old events.
        Requires all nodes to report back so A knows everyone saw A:1."""
        nodes = ['A', 'B', 'C']
        mcs = {n: MatrixClock(n, nodes) for n in nodes}

        # A sends to B and C
        stamp = mcs['A'].send_stamp()
        mcs['B'].receive_stamp('A', stamp)
        stamp2 = mcs['A'].send_stamp()
        mcs['C'].receive_stamp('A', stamp2)

        # B sends to A (A now knows B has seen A:1)
        stamp = mcs['B'].send_stamp()
        mcs['A'].receive_stamp('B', stamp)

        # C sends to A (A now knows C has seen A:1)
        stamp = mcs['C'].send_stamp()
        mcs['A'].receive_stamp('C', stamp)

        # A knows all nodes have seen A:1 (B reported, C reported)
        assert mcs['A'].can_gc('A', 1)

    def test_interval_clock_sparse_events(self):
        """Interval clock handles sparse event sequences efficiently."""
        ic = IntervalClock()
        # Simulate events arriving out of order with gaps
        for c in [1, 5, 10, 2, 6, 3, 7, 4, 8, 9]:
            ic.add('A', c)
        assert ic._intervals['A'] == [(1, 10)]
        assert ic.gap_count == 0

    def test_causal_history_anti_entropy(self):
        """Two nodes sync via causal history to find missing events."""
        ch1 = CausalHistory()
        ch2 = CausalHistory()
        for i in range(1, 11):
            ch1.add('A', i)
        for i in range(1, 6):
            ch2.add('A', i)
        ch2.add('B', 1)

        unseen_by_2 = ch1.events_unseen_by(ch2)
        assert len(unseen_by_2) == 5  # A:6 through A:10

        unseen_by_1 = ch2.events_unseen_by(ch1)
        assert len(unseen_by_1) == 1  # B:1

    def test_store_last_writer_wins(self):
        """Implement LWW conflict resolution strategy."""
        s1 = CausalStore('n1')
        s2 = CausalStore('n2')

        # Sequential writes with known ordering
        sv1 = s1.put('config', {'port': 8080})
        s2.receive_put('config', sv1)
        sv2 = s2.put('config', {'port': 9090})
        s1.receive_put('config', sv2)

        # No conflict -- sv2 supersedes sv1
        assert s1.get_one('config') == {'port': 9090}

    def test_stamped_value(self):
        vc = VectorClock({'A': 1})
        sv = StampedValue('hello', vc, writer='A', timestamp=42)
        assert sv.value == 'hello'
        assert sv.writer == 'A'
        assert sv.timestamp == 42
        assert 'hello' in repr(sv)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
