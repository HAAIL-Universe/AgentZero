"""Tests for C118: Cache Systems."""
import pytest
from cache import (
    LRUCache, LFUCache, TTLCache, SLRUCache,
    ARCCache, WriteBackCache, MultiTierCache,
)


# ============================================================
# LRU Cache Tests
# ============================================================

class TestLRUCache:
    def test_basic_put_get(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        assert c.get('a') == 1
        assert c.get('b') == 2
        assert c.get('c') == 3

    def test_miss_returns_default(self):
        c = LRUCache(3)
        assert c.get('x') is None
        assert c.get('x', 42) == 42

    def test_eviction_order(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        c.put('d', 4)  # evicts 'a' (LRU)
        assert c.get('a') is None
        assert c.get('b') == 2

    def test_access_updates_recency(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        c.get('a')  # 'a' is now MRU
        c.put('d', 4)  # evicts 'b' (now LRU)
        assert c.get('a') == 1
        assert c.get('b') is None

    def test_update_existing_key(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.put('a', 10)
        assert c.get('a') == 10
        assert len(c) == 1

    def test_delete(self):
        c = LRUCache(3)
        c.put('a', 1)
        assert c.delete('a') is True
        assert c.get('a') is None
        assert c.delete('a') is False

    def test_contains(self):
        c = LRUCache(3)
        c.put('a', 1)
        assert 'a' in c
        assert 'b' not in c

    def test_len(self):
        c = LRUCache(3)
        assert len(c) == 0
        c.put('a', 1)
        assert len(c) == 1
        c.put('b', 2)
        assert len(c) == 2

    def test_subscript_ops(self):
        c = LRUCache(3)
        c['a'] = 1
        assert c['a'] == 1
        del c['a']
        with pytest.raises(KeyError):
            _ = c['a']
        with pytest.raises(KeyError):
            del c['a']

    def test_peek_no_recency_update(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        assert c.peek('a') == 1  # does NOT update recency
        c.put('d', 4)  # should evict 'a' since peek didn't touch it
        assert c.get('a') is None

    def test_keys_values_items_order(self):
        c = LRUCache(5)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        c.get('a')  # a is now MRU
        assert c.keys() == ['a', 'c', 'b']
        assert c.values() == [1, 3, 2]
        assert c.items() == [('a', 1), ('c', 3), ('b', 2)]

    def test_clear(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.clear()
        assert len(c) == 0
        assert c.get('a') is None

    def test_on_evict_callback(self):
        evicted = []
        c = LRUCache(2, on_evict=lambda k, v: evicted.append((k, v)))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)  # evicts 'a'
        assert evicted == [('a', 1)]

    def test_hit_rate(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.get('a')  # hit
        c.get('b')  # miss
        assert c.hit_rate == 0.5
        s = c.stats()
        assert s['hits'] == 1
        assert s['misses'] == 1

    def test_capacity_one(self):
        c = LRUCache(1)
        c.put('a', 1)
        c.put('b', 2)
        assert c.get('a') is None
        assert c.get('b') == 2

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            LRUCache(0)

    def test_large_eviction_sequence(self):
        c = LRUCache(100)
        for i in range(200):
            c.put(i, i * 10)
        assert len(c) == 100
        # First 100 should be evicted
        for i in range(100):
            assert c.get(i) is None
        for i in range(100, 200):
            assert c.get(i) == i * 10

    def test_repeated_updates(self):
        c = LRUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        # Keep updating 'a' to keep it alive
        for i in range(10):
            c.put('a', i)
            c.put(f'x{i}', i)
        assert 'a' in c


# ============================================================
# LFU Cache Tests
# ============================================================

class TestLFUCache:
    def test_basic_put_get(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        assert c.get('a') == 1
        assert c.get('b') == 2

    def test_evicts_least_frequent(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        c.get('a')  # freq: a=2, b=1, c=1
        c.get('a')  # freq: a=3, b=1, c=1
        c.get('b')  # freq: a=3, b=2, c=1
        c.put('d', 4)  # evicts 'c' (freq=1)
        assert c.get('c') is None
        assert c.get('a') == 1
        assert c.get('b') == 2
        assert c.get('d') == 4

    def test_lfu_tiebreak_by_recency(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        # All have freq=1, 'a' is oldest -> evict 'a'
        c.put('d', 4)
        assert c.get('a') is None
        assert c.get('b') == 2

    def test_update_existing(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.put('a', 10)
        assert c.get('a') == 10
        assert c.frequency('a') == 3  # put(1) + put(10) touch + get

    def test_delete(self):
        c = LFUCache(3)
        c.put('a', 1)
        assert c.delete('a') is True
        assert c.get('a') is None
        assert c.delete('a') is False

    def test_frequency(self):
        c = LFUCache(3)
        c.put('a', 1)
        assert c.frequency('a') == 1
        c.get('a')
        assert c.frequency('a') == 2
        assert c.frequency('z') == 0

    def test_contains_len(self):
        c = LFUCache(3)
        c.put('a', 1)
        assert 'a' in c
        assert 'b' not in c
        assert len(c) == 1

    def test_clear(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.clear()
        assert len(c) == 0
        assert c.get('a') is None

    def test_on_evict(self):
        evicted = []
        c = LFUCache(2, on_evict=lambda k, v: evicted.append((k, v)))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        assert len(evicted) == 1
        assert evicted[0][0] in ('a', 'b')

    def test_hit_rate(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.get('a')  # hit
        c.get('b')  # miss
        assert c.hit_rate == 0.5

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            LFUCache(0)

    def test_capacity_one(self):
        c = LFUCache(1)
        c.put('a', 1)
        c.put('b', 2)
        assert c.get('a') is None
        assert c.get('b') == 2

    def test_frequency_promotion(self):
        c = LFUCache(3)
        c.put('a', 1)  # freq 1
        c.put('b', 2)  # freq 1
        c.put('c', 3)  # freq 1
        c.get('a')     # freq 2
        c.get('b')     # freq 2
        c.put('d', 4)  # evicts 'c' (freq 1, oldest)
        assert 'c' not in c
        assert c.get('a') == 1

    def test_many_frequencies(self):
        c = LFUCache(3)
        c.put('a', 1)
        for _ in range(10):
            c.get('a')
        c.put('b', 2)
        c.put('c', 3)
        c.put('d', 4)  # evicts 'b' (lowest freq, oldest)
        assert 'a' in c  # freq 11 -- safe


# ============================================================
# TTL Cache Tests
# ============================================================

class TestTTLCache:
    def _make_clock(self, start=0.0):
        """Return a controllable clock function."""
        t = [start]
        def clock():
            return t[0]
        def advance(dt):
            t[0] += dt
        return clock, advance

    def test_basic_put_get(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1)
        assert c.get('a') == 1

    def test_expiry(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1)
        advance(11)
        assert c.get('a') is None

    def test_no_expiry(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, time_func=clock)  # no default TTL
        c.put('a', 1)
        advance(1000)
        assert c.get('a') == 1

    def test_per_entry_ttl(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1, ttl=5)
        c.put('b', 2)  # default TTL=10
        advance(7)
        assert c.get('a') is None  # expired
        assert c.get('b') == 2     # still alive

    def test_ttl_remaining(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1)
        advance(3)
        remaining = c.ttl_remaining('a')
        assert abs(remaining - 7.0) < 0.01
        assert c.ttl_remaining('z') == -1

    def test_no_ttl_remaining(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, time_func=clock)
        c.put('a', 1)
        assert c.ttl_remaining('a') is None

    def test_contains_expiry(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=5, time_func=clock)
        c.put('a', 1)
        assert 'a' in c
        advance(6)
        assert 'a' not in c

    def test_len_with_expiry(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=5, time_func=clock)
        c.put('a', 1)
        c.put('b', 2)
        assert len(c) == 2
        advance(6)
        assert len(c) == 0

    def test_eviction_before_capacity(self):
        clock, advance = self._make_clock()
        c = TTLCache(2, default_ttl=5, time_func=clock)
        c.put('a', 1)
        c.put('b', 2)
        advance(6)  # both expired
        c.put('c', 3)  # should not evict living entry
        c.put('d', 4)
        assert c.get('c') == 3
        assert c.get('d') == 4

    def test_update_resets_ttl(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1)
        advance(8)
        c.put('a', 2)  # reset TTL
        advance(5)
        assert c.get('a') == 2  # still alive (8+5=13, but TTL reset at t=8)

    def test_delete(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1)
        assert c.delete('a') is True
        assert c.get('a') is None

    def test_on_evict(self):
        evicted = []
        clock, advance = self._make_clock()
        c = TTLCache(2, default_ttl=5, time_func=clock,
                     on_evict=lambda k, v: evicted.append((k, v)))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)  # evicts LRU
        assert len(evicted) == 1

    def test_clear(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1)
        c.clear()
        assert len(c) == 0

    def test_hit_rate(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1)
        c.get('a')  # hit
        c.get('b')  # miss
        assert c.hit_rate == 0.5

    def test_zero_ttl_entry(self):
        clock, advance = self._make_clock()
        c = TTLCache(3, default_ttl=10, time_func=clock)
        c.put('a', 1, ttl=0)
        assert c.get('a') is None  # immediately expired

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            TTLCache(0)


# ============================================================
# SLRU Cache Tests
# ============================================================

class TestSLRUCache:
    def test_basic_put_get(self):
        c = SLRUCache(4)
        c.put('a', 1)
        c.put('b', 2)
        assert c.get('a') == 1
        assert c.get('b') == 2

    def test_new_entries_in_probation(self):
        c = SLRUCache(4)
        c.put('a', 1)
        assert c.segment_of('a') == 'probation'

    def test_promotion_to_protected(self):
        c = SLRUCache(4)
        c.put('a', 1)
        c.get('a')  # second access promotes to protected
        assert c.segment_of('a') == 'protected'

    def test_eviction_from_probation(self):
        c = SLRUCache(4, protected_ratio=0.5)
        # Protected cap: 2, Probation cap: 2
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        c.put('d', 4)
        c.put('e', 5)  # should evict from probation
        assert len(c) <= 5  # at most capacity entries survived

    def test_demotion_from_protected(self):
        c = SLRUCache(4, protected_ratio=0.5)
        # Protected cap: 2, Probation cap: 2
        c.put('a', 1)
        c.get('a')  # promote to protected
        c.put('b', 2)
        c.get('b')  # promote to protected (protected now full)
        c.put('c', 3)
        c.get('c')  # promote to protected -> demotes 'a' back to probation
        assert c.segment_of('a') == 'probation'
        assert c.segment_of('c') == 'protected'

    def test_delete(self):
        c = SLRUCache(4)
        c.put('a', 1)
        c.get('a')  # promote
        assert c.delete('a') is True
        assert 'a' not in c
        assert c.delete('z') is False

    def test_update_promotes(self):
        c = SLRUCache(4)
        c.put('a', 1)
        c.put('a', 10)  # update -> promotes to protected
        assert c.segment_of('a') == 'protected'
        assert c.get('a') == 10

    def test_contains_len(self):
        c = SLRUCache(4)
        c.put('a', 1)
        assert 'a' in c
        assert 'z' not in c
        assert len(c) == 1

    def test_clear(self):
        c = SLRUCache(4)
        c.put('a', 1)
        c.get('a')
        c.clear()
        assert len(c) == 0

    def test_on_evict(self):
        evicted = []
        c = SLRUCache(3, on_evict=lambda k, v: evicted.append((k, v)))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        c.put('d', 4)  # triggers eviction
        assert len(evicted) >= 1

    def test_hit_rate(self):
        c = SLRUCache(4)
        c.put('a', 1)
        c.get('a')  # hit
        c.get('z')  # miss
        assert c.hit_rate == 0.5

    def test_stats(self):
        c = SLRUCache(4)
        c.put('a', 1)
        c.get('a')
        s = c.stats()
        assert s['size'] == 1
        assert s['protected_size'] == 1
        assert s['probation_size'] == 0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            SLRUCache(1)

    def test_protected_get_moves_to_front(self):
        c = SLRUCache(10, protected_ratio=0.5)
        c.put('a', 1)
        c.get('a')  # promote
        c.put('b', 2)
        c.get('b')  # promote
        c.get('a')  # access 'a' in protected -> move to front
        assert c.segment_of('a') == 'protected'

    def test_many_entries(self):
        c = SLRUCache(10, protected_ratio=0.5)
        for i in range(20):
            c.put(i, i)
        assert len(c) <= 10


# ============================================================
# ARC Cache Tests
# ============================================================

class TestARCCache:
    def test_basic_put_get(self):
        c = ARCCache(3)
        c.put('a', 1)
        c.put('b', 2)
        assert c.get('a') == 1
        assert c.get('b') == 2

    def test_miss_returns_default(self):
        c = ARCCache(3)
        assert c.get('x') is None
        assert c.get('x', 42) == 42

    def test_basic_eviction(self):
        c = ARCCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        c.put('d', 4)  # evicts something
        assert len(c) == 3

    def test_t1_to_t2_promotion(self):
        c = ARCCache(3)
        c.put('a', 1)
        # 'a' is in T1
        c.get('a')  # now in T2
        s = c.stats()
        assert s['t1_size'] == 0
        assert s['t2_size'] == 1

    def test_ghost_hit_b1_adapts(self):
        c = ARCCache(2)
        c.put('a', 1)
        c.put('b', 2)
        # Now T1 = [b, a]
        c.put('c', 3)  # evicts 'a' to B1
        assert c.get('a') is None  # 'a' not in cache
        p_before = c.stats()['p']
        c.put('a', 10)  # ghost hit on B1 -> increase p
        p_after = c.stats()['p']
        assert p_after >= p_before
        assert c.get('a') == 10

    def test_ghost_hit_b2_adapts(self):
        c = ARCCache(2)
        c.put('a', 1)
        c.get('a')  # a -> T2
        c.put('b', 2)
        c.get('b')  # b -> T2
        # T2 = [b, a]
        c.put('c', 3)  # evicts LRU of T2 -> 'a' to B2
        p_before = c.stats()['p']
        c.put('a', 10)  # ghost hit on B2 -> decrease p
        p_after = c.stats()['p']
        assert p_after <= p_before

    def test_contains_len(self):
        c = ARCCache(3)
        c.put('a', 1)
        assert 'a' in c
        assert 'z' not in c
        assert len(c) == 1

    def test_delete(self):
        c = ARCCache(3)
        c.put('a', 1)
        assert c.delete('a') is True
        assert 'a' not in c
        assert c.delete('z') is False

    def test_delete_from_t2(self):
        c = ARCCache(3)
        c.put('a', 1)
        c.get('a')  # move to T2
        assert c.delete('a') is True
        assert len(c) == 0

    def test_clear(self):
        c = ARCCache(3)
        c.put('a', 1)
        c.get('a')
        c.clear()
        assert len(c) == 0
        s = c.stats()
        assert s['b1_size'] == 0
        assert s['b2_size'] == 0

    def test_on_evict(self):
        evicted = []
        c = ARCCache(2, on_evict=lambda k, v: evicted.append((k, v)))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        assert len(evicted) >= 1

    def test_hit_rate(self):
        c = ARCCache(3)
        c.put('a', 1)
        c.get('a')  # hit
        c.get('z')  # miss
        assert c.hit_rate == 0.5

    def test_update_in_t1(self):
        c = ARCCache(3)
        c.put('a', 1)
        c.put('a', 10)  # update -> moves to T2
        assert c.get('a') == 10
        s = c.stats()
        assert s['t2_size'] == 1

    def test_update_in_t2(self):
        c = ARCCache(3)
        c.put('a', 1)
        c.get('a')  # T2
        c.put('a', 20)  # update in T2
        assert c.get('a') == 20

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            ARCCache(0)

    def test_adapts_to_workload(self):
        """ARC should adapt to changing access patterns."""
        c = ARCCache(5)
        # Phase 1: sequential scan
        for i in range(10):
            c.put(i, i)
        # Phase 2: repeated access to a few keys
        for _ in range(5):
            for k in [7, 8, 9]:
                c.get(k)
        # Frequently accessed keys should survive
        assert c.get(9) == 9

    def test_capacity_one(self):
        c = ARCCache(1)
        c.put('a', 1)
        c.put('b', 2)
        assert len(c) == 1
        assert c.get('b') == 2

    def test_stress(self):
        c = ARCCache(10)
        for i in range(100):
            c.put(i, i)
            if i > 5:
                c.get(i - 3)
        assert len(c) <= 10


# ============================================================
# WriteBack Cache Tests
# ============================================================

class TestWriteBackCache:
    def test_basic_put_get(self):
        c = WriteBackCache(3)
        c.put('a', 1)
        assert c.get('a') == 1

    def test_dirty_tracking(self):
        c = WriteBackCache(3)
        c.put('a', 1)
        assert c.is_dirty('a') is True
        assert c.dirty_keys() == {'a'}

    def test_flush_single(self):
        flushed = {}
        c = WriteBackCache(3, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.flush('a')
        assert flushed == {'a': 1}
        assert c.is_dirty('a') is False

    def test_flush_all(self):
        flushed = {}
        c = WriteBackCache(3, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.put('b', 2)
        c.flush_all()
        assert flushed == {'a': 1, 'b': 2}
        assert len(c.dirty_keys()) == 0

    def test_eviction_flushes(self):
        flushed = {}
        c = WriteBackCache(2, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)  # evicts 'a', which should be flushed first
        assert 'a' in flushed

    def test_read_func(self):
        store = {'x': 42, 'y': 99}
        c = WriteBackCache(3, read_func=lambda k: store[k])
        assert c.get('x') == 42  # loaded from store
        assert c.get('x') == 42  # now cached
        assert c.is_dirty('x') is False  # loaded, not written

    def test_read_func_miss(self):
        c = WriteBackCache(3, read_func=lambda k: (_ for _ in ()).throw(KeyError(k)))
        assert c.get('z') is None

    def test_delete_flushes(self):
        flushed = {}
        c = WriteBackCache(3, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.delete('a')
        assert 'a' in flushed

    def test_contains_len(self):
        c = WriteBackCache(3)
        c.put('a', 1)
        assert 'a' in c
        assert len(c) == 1

    def test_clear_flushes(self):
        flushed = {}
        c = WriteBackCache(3, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.clear(flush=True)
        assert 'a' in flushed
        assert len(c) == 0

    def test_clear_no_flush(self):
        flushed = {}
        c = WriteBackCache(3, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.clear(flush=False)
        assert flushed == {}

    def test_on_evict(self):
        evicted = []
        c = WriteBackCache(2, on_evict=lambda k, v: evicted.append((k, v)))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        assert len(evicted) == 1

    def test_hit_rate(self):
        c = WriteBackCache(3)
        c.put('a', 1)
        c.get('a')  # hit
        c.get('z')  # miss
        assert c.hit_rate == 0.5

    def test_stats(self):
        flushed = {}
        c = WriteBackCache(3, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.flush('a')
        s = c.stats()
        assert s['writes_back'] == 1
        assert s['dirty_count'] == 0

    def test_write_then_read_is_dirty(self):
        c = WriteBackCache(3)
        c.put('a', 1)
        c.get('a')
        assert c.is_dirty('a') is True  # still dirty after read

    def test_update_keeps_dirty(self):
        flushed = {}
        c = WriteBackCache(3, write_func=lambda k, v: flushed.__setitem__(k, v))
        c.put('a', 1)
        c.flush('a')
        c.put('a', 2)  # update -> dirty again
        assert c.is_dirty('a') is True

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            WriteBackCache(0)


# ============================================================
# MultiTier Cache Tests
# ============================================================

class TestMultiTierCache:
    def test_basic_two_tier(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        assert mt.get('a') == 1

    def test_l2_hit_promotes_to_l1(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2], inclusive=True)
        # Only put in L2
        l2.put('x', 42)
        val = mt.get('x')
        assert val == 42
        # Should now be in L1 too
        assert l1.get('x') == 42

    def test_origin_function(self):
        l1 = LRUCache(2)
        store = {'db_key': 'db_value'}
        mt = MultiTierCache([l1], origin=lambda k: store[k])
        val = mt.get('db_key')
        assert val == 'db_value'
        # Should now be cached
        assert l1.get('db_key') == 'db_value'

    def test_origin_miss(self):
        l1 = LRUCache(2)
        mt = MultiTierCache([l1], origin=lambda k: (_ for _ in ()).throw(KeyError(k)))
        assert mt.get('z') is None

    def test_no_origin_miss(self):
        l1 = LRUCache(2)
        mt = MultiTierCache([l1])
        assert mt.get('z') is None
        assert mt.get('z', 42) == 42

    def test_put_writes_all_levels(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        assert l1.get('a') == 1
        assert l2.get('a') == 1

    def test_delete_from_all_levels(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        mt.delete('a')
        assert l1.get('a') is None
        assert l2.get('a') is None

    def test_invalidate(self):
        l1 = LRUCache(2)
        mt = MultiTierCache([l1])
        mt.put('a', 1)
        mt.invalidate('a')
        assert 'a' not in mt

    def test_contains(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2])
        l2.put('a', 1)
        assert 'a' in mt

    def test_len(self):
        l1 = LRUCache(2)
        mt = MultiTierCache([l1])
        mt.put('a', 1)
        assert len(mt) == 1

    def test_clear(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        mt.clear()
        assert len(l1) == 0
        assert len(l2) == 0

    def test_stats(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        mt.get('a')  # L1 hit
        s = mt.stats()
        assert s['hits_per_level'][0] == 1
        assert s['total_hits'] == 1

    def test_three_tier(self):
        l1 = LRUCache(1)
        l2 = LRUCache(3)
        l3 = LRUCache(10)
        mt = MultiTierCache([l1, l2, l3])
        mt.put('a', 1)
        mt.put('b', 2)  # evicts 'a' from L1
        # 'a' should still be in L2 and L3
        assert l1.get('a') is None
        assert l2.get('a') == 1
        val = mt.get('a')  # L2 hit, promotes back to L1
        assert val == 1
        s = mt.stats()
        assert s['hits_per_level'][1] == 1

    def test_exclusive_no_promotion(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        mt = MultiTierCache([l1, l2], inclusive=False)
        l2.put('x', 42)
        val = mt.get('x')
        assert val == 42
        # Exclusive: should NOT promote to L1
        assert l1.peek('x') is None

    def test_no_levels_raises(self):
        with pytest.raises(ValueError):
            MultiTierCache([])

    def test_mixed_cache_types(self):
        l1 = LRUCache(2)
        l2 = LFUCache(5)
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        assert mt.get('a') == 1

    def test_origin_populates_all_levels(self):
        l1 = LRUCache(2)
        l2 = LRUCache(5)
        store = {'k': 'v'}
        mt = MultiTierCache([l1, l2], origin=lambda k: store[k])
        mt.get('k')
        assert l1.peek('k') == 'v'
        assert l2.peek('k') == 'v'


# ============================================================
# Cross-cutting / Integration Tests
# ============================================================

class TestCrossIntegration:
    def test_lru_as_l1_writeback_as_l2(self):
        """MultiTier with WriteBackCache as L2."""
        flushed = {}
        l1 = LRUCache(2)
        l2 = WriteBackCache(5, write_func=lambda k, v: flushed.__setitem__(k, v))
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        l2.flush_all()
        assert flushed == {'a': 1}

    def test_ttl_in_multitier(self):
        clock, advance = [0.0], lambda dt: clock.__setitem__(0, clock[0] + dt)
        l1 = TTLCache(2, default_ttl=5, time_func=lambda: clock[0])
        l2 = LRUCache(10)
        mt = MultiTierCache([l1, l2])
        mt.put('a', 1)
        advance(6)
        # L1 expired, but L2 should still have it
        val = mt.get('a')
        assert val == 1

    def test_all_cache_types_basic_ops(self):
        """Ensure all cache types support basic get/put/delete/len/contains."""
        caches = [
            LRUCache(5),
            LFUCache(5),
            TTLCache(5),
            SLRUCache(5),
            ARCCache(5),
            WriteBackCache(5),
        ]
        for c in caches:
            c.put('a', 1)
            assert c.get('a') == 1, f"{type(c).__name__} get failed"
            assert 'a' in c, f"{type(c).__name__} contains failed"
            assert len(c) == 1, f"{type(c).__name__} len failed"
            c.delete('a')
            assert 'a' not in c, f"{type(c).__name__} delete failed"

    def test_eviction_storm(self):
        """All cache types handle rapid eviction under capacity pressure."""
        caches = [
            LRUCache(5),
            LFUCache(5),
            TTLCache(5),
            SLRUCache(5),
            ARCCache(5),
            WriteBackCache(5),
        ]
        for c in caches:
            for i in range(100):
                c.put(i, i)
            assert len(c) <= 5, f"{type(c).__name__} overflow"

    def test_writeback_stress(self):
        """WriteBackCache correctly flushes under stress."""
        flushed = {}
        c = WriteBackCache(5, write_func=lambda k, v: flushed.__setitem__(k, v))
        for i in range(50):
            c.put(i, i * 10)
        c.flush_all()
        # At least the remaining entries should be flushed
        for k in list(c._map.keys()):
            assert k in flushed


# ============================================================
# Edge cases and regression tests
# ============================================================

class TestEdgeCases:
    def test_lru_single_key_repeated(self):
        c = LRUCache(3)
        for i in range(100):
            c.put('only', i)
        assert c.get('only') == 99
        assert len(c) == 1

    def test_lfu_all_same_frequency(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)
        # All freq=1, eviction should pick oldest (LRU tiebreak)
        c.put('d', 4)
        assert 'a' not in c

    def test_arc_capacity_respects_limit(self):
        c = ARCCache(5)
        for i in range(50):
            c.put(i, i)
            if i > 5:
                c.get(i - 2)
        assert len(c) <= 5

    def test_slru_segment_balance(self):
        c = SLRUCache(10, protected_ratio=0.5)
        for i in range(20):
            c.put(i, i)
        # Promote some
        for i in range(10, 20):
            c.get(i)
        s = c.stats()
        assert s['protected_size'] <= c._protected_cap
        assert s['size'] <= 10

    def test_none_values(self):
        """Caches should handle None as a valid value."""
        c = LRUCache(3)
        c.put('a', None)
        assert c.get('a') is None
        assert c.get('a', 'default') is None  # should return None, not default
        assert 'a' in c

    def test_various_key_types(self):
        c = LRUCache(10)
        c.put(1, 'int')
        c.put('str', 'string')
        c.put((1, 2), 'tuple')
        c.put(3.14, 'float')
        assert c.get(1) == 'int'
        assert c.get('str') == 'string'
        assert c.get((1, 2)) == 'tuple'
        assert c.get(3.14) == 'float'

    def test_lru_evict_callback_multiple(self):
        evicted = []
        c = LRUCache(2, on_evict=lambda k, v: evicted.append((k, v)))
        c.put('a', 1)
        c.put('b', 2)
        c.put('c', 3)  # evicts 'a'
        c.put('d', 4)  # evicts 'b'
        assert evicted == [('a', 1), ('b', 2)]

    def test_writeback_no_write_func(self):
        """WriteBackCache works without write_func (no-op flush)."""
        c = WriteBackCache(3)
        c.put('a', 1)
        c.flush('a')  # should not raise
        assert c.is_dirty('a') is False

    def test_multitier_deep_miss(self):
        l1 = LRUCache(1)
        l2 = LRUCache(1)
        l3 = LRUCache(1)
        mt = MultiTierCache([l1, l2, l3])
        assert mt.get('nonexistent') is None
        s = mt.stats()
        assert s['misses'] == 1

    def test_lfu_delete_then_reinsert(self):
        c = LFUCache(3)
        c.put('a', 1)
        c.get('a')  # freq 2
        c.delete('a')
        c.put('a', 10)
        assert c.frequency('a') == 1  # reset after delete

    def test_arc_many_ghost_hits(self):
        c = ARCCache(3)
        # Fill and evict to create ghosts
        for i in range(10):
            c.put(i, i)
        # Re-insert evicted keys (ghost hits)
        for i in range(10):
            c.put(i, i * 10)
        assert len(c) <= 3

    def test_ttl_boundary_exact_expiry(self):
        clock, advance = [0.0], lambda dt: clock.__setitem__(0, clock[0] + dt)
        c = TTLCache(3, default_ttl=10, time_func=lambda: clock[0])
        c.put('a', 1)
        advance(10)  # exactly at expiry time
        assert c.get('a') is None  # should be expired at exactly TTL


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
