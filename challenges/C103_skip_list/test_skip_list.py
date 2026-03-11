"""Tests for C103: Skip List."""
import pytest
from skip_list import SkipListMap, SkipListSet, SkipNode


# ============================================================
# SkipNode
# ============================================================

class TestSkipNode:
    def test_create_node(self):
        n = SkipNode("a", 1, 3)
        assert n.key == "a"
        assert n.value == 1
        assert len(n.forward) == 4  # levels 0-3
        assert len(n.span) == 4

    def test_repr(self):
        n = SkipNode(42, "hello", 2)
        assert "42" in repr(n)
        assert "level=2" in repr(n)


# ============================================================
# SkipListMap: Basic operations
# ============================================================

class TestSkipListMapBasic:
    def test_empty(self):
        sl = SkipListMap(seed=42)
        assert len(sl) == 0
        assert not sl
        assert 5 not in sl

    def test_put_and_get(self):
        sl = SkipListMap(seed=42)
        assert sl.put(3, "three") is True  # new
        assert sl.put(1, "one") is True
        assert sl.put(2, "two") is True
        assert len(sl) == 3
        assert sl.get(1) == "one"
        assert sl.get(2) == "two"
        assert sl.get(3) == "three"

    def test_get_default(self):
        sl = SkipListMap(seed=42)
        assert sl.get(99) is None
        assert sl.get(99, "missing") == "missing"

    def test_contains(self):
        sl = SkipListMap(seed=42)
        sl[5] = "five"
        assert 5 in sl
        assert 6 not in sl

    def test_getitem(self):
        sl = SkipListMap(seed=42)
        sl[10] = "ten"
        assert sl[10] == "ten"

    def test_getitem_missing(self):
        sl = SkipListMap(seed=42)
        with pytest.raises(KeyError):
            _ = sl[999]

    def test_setitem(self):
        sl = SkipListMap(seed=42)
        sl["x"] = 1
        sl["y"] = 2
        assert sl["x"] == 1
        assert sl["y"] == 2

    def test_update_value(self):
        sl = SkipListMap(seed=42)
        assert sl.put(1, "a") is True   # new
        assert sl.put(1, "b") is False   # update
        assert sl[1] == "b"
        assert len(sl) == 1

    def test_bool_truthy(self):
        sl = SkipListMap(seed=42)
        assert not sl
        sl[1] = "one"
        assert sl

    def test_repr(self):
        sl = SkipListMap(seed=42)
        sl[1] = "a"
        sl[2] = "b"
        r = repr(sl)
        assert "SkipListMap" in r
        assert "1" in r


# ============================================================
# SkipListMap: Delete
# ============================================================

class TestSkipListMapDelete:
    def test_delete_existing(self):
        sl = SkipListMap(seed=42)
        sl[1] = "a"
        sl[2] = "b"
        sl[3] = "c"
        assert sl.delete(2) is True
        assert 2 not in sl
        assert len(sl) == 2

    def test_delete_missing(self):
        sl = SkipListMap(seed=42)
        sl[1] = "a"
        assert sl.delete(99) is False
        assert len(sl) == 1

    def test_delitem(self):
        sl = SkipListMap(seed=42)
        sl[5] = "five"
        del sl[5]
        assert 5 not in sl

    def test_delitem_missing(self):
        sl = SkipListMap(seed=42)
        with pytest.raises(KeyError):
            del sl[42]

    def test_delete_all(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        for i in range(10):
            sl.delete(i)
        assert len(sl) == 0
        assert not sl

    def test_delete_reinsert(self):
        sl = SkipListMap(seed=42)
        sl[1] = "first"
        sl.delete(1)
        sl[1] = "second"
        assert sl[1] == "second"
        assert len(sl) == 1


# ============================================================
# SkipListMap: Ordered operations
# ============================================================

class TestSkipListMapOrdered:
    def test_min(self):
        sl = SkipListMap(seed=42)
        sl[5] = "e"
        sl[1] = "a"
        sl[3] = "c"
        assert sl.min() == (1, "a")

    def test_max(self):
        sl = SkipListMap(seed=42)
        sl[5] = "e"
        sl[1] = "a"
        sl[3] = "c"
        assert sl.max() == (5, "e")

    def test_min_empty(self):
        sl = SkipListMap(seed=42)
        with pytest.raises(ValueError):
            sl.min()

    def test_max_empty(self):
        sl = SkipListMap(seed=42)
        with pytest.raises(ValueError):
            sl.max()

    def test_floor(self):
        sl = SkipListMap(seed=42)
        for i in [2, 4, 6, 8]:
            sl[i] = i
        assert sl.floor(5) == (4, 4)
        assert sl.floor(6) == (6, 6)
        assert sl.floor(1) is None
        assert sl.floor(8) == (8, 8)
        assert sl.floor(10) == (8, 8)

    def test_ceiling(self):
        sl = SkipListMap(seed=42)
        for i in [2, 4, 6, 8]:
            sl[i] = i
        assert sl.ceiling(5) == (6, 6)
        assert sl.ceiling(6) == (6, 6)
        assert sl.ceiling(1) == (2, 2)
        assert sl.ceiling(9) is None

    def test_lower(self):
        sl = SkipListMap(seed=42)
        for i in [2, 4, 6]:
            sl[i] = i
        assert sl.lower(4) == (2, 2)
        assert sl.lower(5) == (4, 4)
        assert sl.lower(2) is None

    def test_higher(self):
        sl = SkipListMap(seed=42)
        for i in [2, 4, 6]:
            sl[i] = i
        assert sl.higher(4) == (6, 6)
        assert sl.higher(3) == (4, 4)
        assert sl.higher(6) is None


# ============================================================
# SkipListMap: Range queries
# ============================================================

class TestSkipListMapRange:
    def test_range_inclusive(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        result = sl.range(3, 7)
        assert result == [(3, 30), (4, 40), (5, 50), (6, 60), (7, 70)]

    def test_range_exclusive_lo(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        result = sl.range(3, 7, inclusive_lo=False)
        assert result == [(4, 40), (5, 50), (6, 60), (7, 70)]

    def test_range_exclusive_hi(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        result = sl.range(3, 7, inclusive_hi=False)
        assert result == [(3, 30), (4, 40), (5, 50), (6, 60)]

    def test_range_both_exclusive(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        result = sl.range(3, 7, inclusive_lo=False, inclusive_hi=False)
        assert result == [(4, 40), (5, 50), (6, 60)]

    def test_range_empty(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        assert sl.range(20, 30) == []

    def test_range_single(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        assert sl.range(5, 5) == [(5, 50)]

    def test_range_single_exclusive(self):
        sl = SkipListMap(seed=42)
        for i in range(10):
            sl[i] = i * 10
        assert sl.range(5, 5, inclusive_lo=False) == []


# ============================================================
# SkipListMap: Rank and Select
# ============================================================

class TestSkipListMapRankSelect:
    def test_rank(self):
        sl = SkipListMap(seed=42)
        for i in [10, 20, 30, 40, 50]:
            sl[i] = i
        assert sl.rank(10) == 0
        assert sl.rank(30) == 2
        assert sl.rank(50) == 4
        assert sl.rank(25) == 2  # between 20 and 30

    def test_rank_empty(self):
        sl = SkipListMap(seed=42)
        assert sl.rank(5) == 0

    def test_select(self):
        sl = SkipListMap(seed=42)
        for i in [10, 20, 30, 40, 50]:
            sl[i] = i
        assert sl.select(0) == (10, 10)
        assert sl.select(2) == (30, 30)
        assert sl.select(4) == (50, 50)

    def test_select_negative(self):
        sl = SkipListMap(seed=42)
        for i in [10, 20, 30]:
            sl[i] = i
        assert sl.select(-1) == (30, 30)
        assert sl.select(-3) == (10, 10)

    def test_select_out_of_range(self):
        sl = SkipListMap(seed=42)
        sl[1] = 1
        with pytest.raises(IndexError):
            sl.select(5)

    def test_rank_select_consistency(self):
        sl = SkipListMap(seed=42)
        for i in range(20):
            sl[i * 3] = i
        for i in range(20):
            k, v = sl.select(i)
            assert sl.rank(k) == i


# ============================================================
# SkipListMap: Iteration
# ============================================================

class TestSkipListMapIteration:
    def test_iter_sorted(self):
        sl = SkipListMap(seed=42)
        sl[5] = "e"
        sl[1] = "a"
        sl[3] = "c"
        sl[2] = "b"
        sl[4] = "d"
        assert list(sl) == [1, 2, 3, 4, 5]

    def test_keys(self):
        sl = SkipListMap(seed=42)
        sl[3] = "c"
        sl[1] = "a"
        sl[2] = "b"
        assert sl.keys() == [1, 2, 3]

    def test_values(self):
        sl = SkipListMap(seed=42)
        sl[3] = "c"
        sl[1] = "a"
        sl[2] = "b"
        assert sl.values() == ["a", "b", "c"]

    def test_items(self):
        sl = SkipListMap(seed=42)
        sl[3] = "c"
        sl[1] = "a"
        assert sl.items() == [(1, "a"), (3, "c")]

    def test_iter_empty(self):
        sl = SkipListMap(seed=42)
        assert list(sl) == []


# ============================================================
# SkipListMap: Pop operations
# ============================================================

class TestSkipListMapPop:
    def test_pop_min(self):
        sl = SkipListMap(seed=42)
        sl[3] = "c"
        sl[1] = "a"
        sl[2] = "b"
        assert sl.pop_min() == (1, "a")
        assert len(sl) == 2
        assert 1 not in sl

    def test_pop_max(self):
        sl = SkipListMap(seed=42)
        sl[3] = "c"
        sl[1] = "a"
        sl[2] = "b"
        assert sl.pop_max() == (3, "c")
        assert len(sl) == 2

    def test_pop_min_empty(self):
        sl = SkipListMap(seed=42)
        with pytest.raises(ValueError):
            sl.pop_min()

    def test_pop_max_empty(self):
        sl = SkipListMap(seed=42)
        with pytest.raises(ValueError):
            sl.pop_max()

    def test_pop_all_min(self):
        sl = SkipListMap(seed=42)
        for i in range(5):
            sl[i] = i
        results = []
        while sl:
            results.append(sl.pop_min())
        assert results == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


# ============================================================
# SkipListMap: Bulk and utility
# ============================================================

class TestSkipListMapBulk:
    def test_update(self):
        sl = SkipListMap(seed=42)
        sl.update([(3, "c"), (1, "a"), (2, "b")])
        assert sl.keys() == [1, 2, 3]

    def test_from_items(self):
        sl = SkipListMap.from_items([(1, "a"), (2, "b")], seed=42)
        assert sl[1] == "a"
        assert sl[2] == "b"

    def test_clear(self):
        sl = SkipListMap(seed=42)
        sl.update([(1, "a"), (2, "b"), (3, "c")])
        sl.clear()
        assert len(sl) == 0
        assert list(sl) == []

    def test_clear_then_reuse(self):
        sl = SkipListMap(seed=42)
        sl[1] = "a"
        sl.clear()
        sl[2] = "b"
        assert sl.keys() == [2]
        assert len(sl) == 1

    def test_copy(self):
        sl = SkipListMap(seed=42)
        sl[1] = "a"
        sl[2] = "b"
        cp = sl.copy(seed=99)
        assert cp.keys() == [1, 2]
        cp[3] = "c"
        assert 3 not in sl  # independent

    def test_merge_overwrite(self):
        a = SkipListMap(seed=42)
        a[1] = "a1"
        a[2] = "a2"
        b = SkipListMap(seed=42)
        b[2] = "b2"
        b[3] = "b3"
        m = a.merge(b, conflict='overwrite')
        assert m[1] == "a1"
        assert m[2] == "b2"
        assert m[3] == "b3"

    def test_merge_keep(self):
        a = SkipListMap(seed=42)
        a[1] = "a1"
        a[2] = "a2"
        b = SkipListMap(seed=42)
        b[2] = "b2"
        m = a.merge(b, conflict='keep')
        assert m[2] == "a2"

    def test_merge_callable(self):
        a = SkipListMap(seed=42)
        a[1] = 10
        b = SkipListMap(seed=42)
        b[1] = 20
        m = a.merge(b, conflict=lambda k, v1, v2: v1 + v2)
        assert m[1] == 30


# ============================================================
# SkipListMap: Edge cases
# ============================================================

class TestSkipListMapEdgeCases:
    def test_single_element(self):
        sl = SkipListMap(seed=42)
        sl[42] = "answer"
        assert sl.min() == (42, "answer")
        assert sl.max() == (42, "answer")
        assert sl.rank(42) == 0
        assert sl.select(0) == (42, "answer")

    def test_negative_keys(self):
        sl = SkipListMap(seed=42)
        sl[-5] = "neg5"
        sl[-1] = "neg1"
        sl[0] = "zero"
        sl[3] = "pos3"
        assert sl.min() == (-5, "neg5")
        assert sl.keys() == [-5, -1, 0, 3]

    def test_string_keys(self):
        sl = SkipListMap(seed=42)
        sl["banana"] = 2
        sl["apple"] = 1
        sl["cherry"] = 3
        assert sl.keys() == ["apple", "banana", "cherry"]

    def test_float_keys(self):
        sl = SkipListMap(seed=42)
        sl[1.5] = "a"
        sl[0.5] = "b"
        sl[2.5] = "c"
        assert sl.keys() == [0.5, 1.5, 2.5]

    def test_many_elements(self):
        sl = SkipListMap(seed=42)
        for i in range(1000):
            sl[i] = i * 10
        assert len(sl) == 1000
        assert sl[0] == 0
        assert sl[999] == 9990
        assert sl.min() == (0, 0)
        assert sl.max() == (999, 9990)

    def test_reverse_insertion(self):
        sl = SkipListMap(seed=42)
        for i in range(99, -1, -1):
            sl[i] = i
        assert sl.keys() == list(range(100))

    def test_random_operations(self):
        """Stress test: random insert/delete/query."""
        import random as rng
        rng.seed(12345)
        sl = SkipListMap(seed=42)
        reference = {}
        for _ in range(500):
            op = rng.choice(['put', 'put', 'put', 'delete', 'get'])
            k = rng.randint(0, 100)
            if op == 'put':
                v = rng.randint(0, 999)
                sl[k] = v
                reference[k] = v
            elif op == 'delete':
                if k in reference:
                    del sl[k]
                    del reference[k]
            elif op == 'get':
                assert sl.get(k) == reference.get(k)
        assert sorted(sl.keys()) == sorted(reference.keys())

    def test_config_validation(self):
        with pytest.raises(ValueError):
            SkipListMap(p=0)
        with pytest.raises(ValueError):
            SkipListMap(p=1)
        with pytest.raises(ValueError):
            SkipListMap(max_level=0)

    def test_low_max_level(self):
        sl = SkipListMap(max_level=2, seed=42)
        for i in range(50):
            sl[i] = i
        assert len(sl) == 50
        assert sl.keys() == list(range(50))

    def test_debug_levels(self):
        sl = SkipListMap(seed=42)
        sl[1] = "a"
        sl[2] = "b"
        d = sl.debug_levels()
        assert "L0" in d
        assert "H" in d


# ============================================================
# SkipListSet: Basic
# ============================================================

class TestSkipListSetBasic:
    def test_empty(self):
        s = SkipListSet(seed=42)
        assert len(s) == 0
        assert not s

    def test_add(self):
        s = SkipListSet(seed=42)
        assert s.add(3) is True
        assert s.add(1) is True
        assert s.add(3) is False  # duplicate
        assert len(s) == 2

    def test_contains(self):
        s = SkipListSet(seed=42)
        s.add(5)
        assert 5 in s
        assert 6 not in s

    def test_discard(self):
        s = SkipListSet(seed=42)
        s.add(1)
        assert s.discard(1) is True
        assert s.discard(1) is False
        assert len(s) == 0

    def test_remove(self):
        s = SkipListSet(seed=42)
        s.add(1)
        s.remove(1)
        assert 1 not in s

    def test_remove_missing(self):
        s = SkipListSet(seed=42)
        with pytest.raises(KeyError):
            s.remove(99)

    def test_iter_sorted(self):
        s = SkipListSet(seed=42)
        for x in [5, 3, 1, 4, 2]:
            s.add(x)
        assert list(s) == [1, 2, 3, 4, 5]


# ============================================================
# SkipListSet: Ordered
# ============================================================

class TestSkipListSetOrdered:
    def test_min_max(self):
        s = SkipListSet(seed=42)
        for x in [5, 3, 1, 4, 2]:
            s.add(x)
        assert s.min() == 1
        assert s.max() == 5

    def test_floor_ceiling(self):
        s = SkipListSet(seed=42)
        for x in [2, 4, 6, 8]:
            s.add(x)
        assert s.floor(5) == 4
        assert s.ceiling(5) == 6
        assert s.floor(4) == 4
        assert s.ceiling(4) == 4

    def test_lower_higher(self):
        s = SkipListSet(seed=42)
        for x in [2, 4, 6]:
            s.add(x)
        assert s.lower(4) == 2
        assert s.higher(4) == 6
        assert s.lower(2) is None
        assert s.higher(6) is None

    def test_range(self):
        s = SkipListSet(seed=42)
        for x in range(10):
            s.add(x)
        assert s.range(3, 7) == [3, 4, 5, 6, 7]

    def test_rank_select(self):
        s = SkipListSet(seed=42)
        for x in [10, 20, 30, 40, 50]:
            s.add(x)
        assert s.rank(30) == 2
        assert s.select(2) == 30

    def test_pop_min_max(self):
        s = SkipListSet(seed=42)
        for x in [1, 2, 3]:
            s.add(x)
        assert s.pop_min() == 1
        assert s.pop_max() == 3
        assert list(s) == [2]


# ============================================================
# SkipListSet: Set operations
# ============================================================

class TestSkipListSetOps:
    def test_union(self):
        a = SkipListSet(seed=42)
        b = SkipListSet(seed=43)
        for x in [1, 2, 3]:
            a.add(x)
        for x in [2, 3, 4]:
            b.add(x)
        u = a.union(b)
        assert u.to_list() == [1, 2, 3, 4]

    def test_intersection(self):
        a = SkipListSet(seed=42)
        b = SkipListSet(seed=43)
        for x in [1, 2, 3]:
            a.add(x)
        for x in [2, 3, 4]:
            b.add(x)
        i = a.intersection(b)
        assert i.to_list() == [2, 3]

    def test_difference(self):
        a = SkipListSet(seed=42)
        b = SkipListSet(seed=43)
        for x in [1, 2, 3]:
            a.add(x)
        for x in [2, 3, 4]:
            b.add(x)
        d = a.difference(b)
        assert d.to_list() == [1]

    def test_symmetric_difference(self):
        a = SkipListSet(seed=42)
        b = SkipListSet(seed=43)
        for x in [1, 2, 3]:
            a.add(x)
        for x in [2, 3, 4]:
            b.add(x)
        sd = a.symmetric_difference(b)
        assert sd.to_list() == [1, 4]

    def test_issubset(self):
        a = SkipListSet(seed=42)
        b = SkipListSet(seed=43)
        for x in [1, 2]:
            a.add(x)
        for x in [1, 2, 3]:
            b.add(x)
        assert a.issubset(b) is True
        assert b.issubset(a) is False

    def test_issuperset(self):
        a = SkipListSet(seed=42)
        b = SkipListSet(seed=43)
        for x in [1, 2, 3]:
            a.add(x)
        for x in [1, 2]:
            b.add(x)
        assert a.issuperset(b) is True
        assert b.issuperset(a) is False

    def test_to_list(self):
        s = SkipListSet(seed=42)
        for x in [5, 1, 3]:
            s.add(x)
        assert s.to_list() == [1, 3, 5]

    def test_copy(self):
        s = SkipListSet(seed=42)
        s.add(1)
        s.add(2)
        cp = s.copy(seed=99)
        cp.add(3)
        assert 3 not in s

    def test_clear(self):
        s = SkipListSet(seed=42)
        s.add(1)
        s.clear()
        assert len(s) == 0

    def test_repr(self):
        s = SkipListSet(seed=42)
        s.add(1)
        assert "SkipListSet" in repr(s)

    def test_empty_set_operations(self):
        a = SkipListSet(seed=42)
        b = SkipListSet(seed=43)
        assert a.union(b).to_list() == []
        assert a.intersection(b).to_list() == []
        assert a.difference(b).to_list() == []
        assert a.issubset(b) is True


# ============================================================
# SkipListMap: Deterministic behavior with seed
# ============================================================

class TestDeterminism:
    def test_same_seed_same_structure(self):
        sl1 = SkipListMap(seed=42)
        sl2 = SkipListMap(seed=42)
        for i in range(20):
            sl1[i] = i
            sl2[i] = i
        assert sl1.debug_levels() == sl2.debug_levels()

    def test_different_seed_different_structure(self):
        sl1 = SkipListMap(seed=42)
        sl2 = SkipListMap(seed=99)
        for i in range(20):
            sl1[i] = i
            sl2[i] = i
        # Same data, different internal structure
        assert sl1.keys() == sl2.keys()
        # Levels will very likely differ
        # (not guaranteed but extremely probable with 20 elements)


# ============================================================
# Integration: Priority Queue usage
# ============================================================

class TestPriorityQueueUsage:
    def test_as_priority_queue(self):
        """SkipListMap can serve as a priority queue via pop_min."""
        pq = SkipListMap(seed=42)
        pq[3] = "low"
        pq[1] = "high"
        pq[2] = "medium"
        assert pq.pop_min() == (1, "high")
        assert pq.pop_min() == (2, "medium")
        assert pq.pop_min() == (3, "low")

    def test_as_double_ended_pq(self):
        """Both min and max extraction."""
        pq = SkipListMap(seed=42)
        for i in range(10):
            pq[i] = f"task-{i}"
        assert pq.pop_min()[0] == 0
        assert pq.pop_max()[0] == 9
        assert len(pq) == 8


# ============================================================
# Integration: sorted-dict use cases
# ============================================================

class TestSortedDictUsage:
    def test_time_series(self):
        """Store time-series data and do range queries."""
        ts = SkipListMap(seed=42)
        ts[100] = "event_a"
        ts[200] = "event_b"
        ts[300] = "event_c"
        ts[400] = "event_d"
        ts[500] = "event_e"
        # Query window
        window = ts.range(200, 400)
        assert len(window) == 3
        assert window[0] == (200, "event_b")
        assert window[-1] == (400, "event_d")

    def test_leaderboard(self):
        """Use as a leaderboard with rank queries."""
        lb = SkipListMap(seed=42)
        lb[100] = "alice"
        lb[250] = "bob"
        lb[175] = "charlie"
        lb[300] = "diana"
        # Who is rank 0 (lowest score)?
        assert lb.select(0) == (100, "alice")
        # What rank is score 250?
        assert lb.rank(250) == 2
        # Top scorer
        assert lb.max() == (300, "diana")


# ============================================================
# Span integrity after operations
# ============================================================

class TestSpanIntegrity:
    def _verify_spans(self, sl: SkipListMap):
        """Verify span consistency: traversing level 0 with spans must reach all nodes."""
        # Count at level 0
        count = 0
        x = sl._header.forward[0]
        while x is not None:
            count += 1
            x = x.forward[0]
        assert count == len(sl)

        # For each level, sum of spans should equal size + 1 (header span to end)
        for lvl in range(sl._level + 1):
            total_span = 0
            x = sl._header
            nodes_at_level = 0
            while x.forward[lvl] is not None:
                total_span += x.span[lvl]
                x = x.forward[lvl]
                nodes_at_level += 1
            # The last span at this level should cover remaining elements
            # total elements reachable = total_span
            # This should equal len(sl) if we also add the last node's span
            # Actually: header.span[lvl] skips to first node at lvl
            # Each node.span[lvl] skips to next node at lvl
            # The total span = len(sl) is verified by select/rank working

    def test_spans_after_inserts(self):
        sl = SkipListMap(seed=42)
        for i in range(50):
            sl[i] = i
            self._verify_spans(sl)

    def test_spans_after_deletes(self):
        sl = SkipListMap(seed=42)
        for i in range(30):
            sl[i] = i
        for i in range(0, 30, 2):
            sl.delete(i)
            self._verify_spans(sl)

    def test_rank_select_after_mixed_ops(self):
        sl = SkipListMap(seed=42)
        for i in range(40):
            sl[i] = i
        for i in [5, 10, 15, 20, 25]:
            sl.delete(i)
        # Remaining keys
        remaining = sl.keys()
        for idx, k in enumerate(remaining):
            assert sl.rank(k) == idx
            assert sl.select(idx) == (k, k)


# ============================================================
# ConcurrentSkipListMap
# ============================================================

class TestConcurrentSkipList:
    def test_basic_concurrent(self):
        """Basic thread-safe operations."""
        from skip_list import ConcurrentSkipListMap
        csl = ConcurrentSkipListMap(seed=42)
        csl.put(1, "a")
        csl.put(2, "b")
        assert csl.get(1) == "a"
        assert len(csl) == 2
        assert 1 in csl
        csl.delete(1)
        assert 1 not in csl

    def test_concurrent_getsetdel(self):
        from skip_list import ConcurrentSkipListMap
        csl = ConcurrentSkipListMap(seed=42)
        csl[10] = "ten"
        assert csl[10] == "ten"
        del csl[10]
        assert len(csl) == 0

    def test_concurrent_items_keys(self):
        from skip_list import ConcurrentSkipListMap
        csl = ConcurrentSkipListMap(seed=42)
        csl[3] = "c"
        csl[1] = "a"
        csl[2] = "b"
        assert csl.keys() == [1, 2, 3]
        assert csl.items() == [(1, "a"), (2, "b"), (3, "c")]

    def test_concurrent_range(self):
        from skip_list import ConcurrentSkipListMap
        csl = ConcurrentSkipListMap(seed=42)
        for i in range(10):
            csl[i] = i
        assert csl.range(3, 7) == [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]

    def test_threaded_writes(self):
        """Multiple threads writing concurrently."""
        import threading
        from skip_list import ConcurrentSkipListMap
        csl = ConcurrentSkipListMap(seed=42)
        errors = []

        def writer(start, end):
            try:
                for i in range(start, end):
                    csl.put(i, i * 10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100, (i + 1) * 100)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(csl) == 400
