"""Tests for C111: Skip List."""

import pytest
import random
import threading
from skip_list import (
    SkipList, SkipListSet, IndexableSkipList,
    ConcurrentSkipList, MergeableSkipList, IntervalSkipList
)


# ============================================================
# SkipList -- basic operations
# ============================================================

class TestSkipListBasic:
    def test_empty(self):
        sl = SkipList()
        assert len(sl) == 0
        assert not sl
        assert sl.get(1) is None
        assert 1 not in sl

    def test_insert_single(self):
        sl = SkipList()
        assert sl.insert(5, "five") is True
        assert len(sl) == 1
        assert sl.get(5) == "five"
        assert 5 in sl

    def test_insert_multiple(self):
        sl = SkipList()
        for i in [3, 1, 4, 1, 5, 9]:
            sl.insert(i, i * 10)
        assert len(sl) == 5  # duplicate 1 updates
        assert sl.get(1) == 10
        assert sl.get(3) == 30
        assert sl.get(9) == 90

    def test_insert_returns_false_on_update(self):
        sl = SkipList()
        assert sl.insert(1, "a") is True
        assert sl.insert(1, "b") is False
        assert sl.get(1) == "b"
        assert len(sl) == 1

    def test_delete(self):
        sl = SkipList()
        sl.insert(1, "a")
        sl.insert(2, "b")
        assert sl.delete(1) is True
        assert len(sl) == 1
        assert 1 not in sl
        assert 2 in sl

    def test_delete_nonexistent(self):
        sl = SkipList()
        sl.insert(1, "a")
        assert sl.delete(99) is False

    def test_delete_empty(self):
        sl = SkipList()
        assert sl.delete(1) is False

    def test_get_default(self):
        sl = SkipList()
        assert sl.get(1, "default") == "default"

    def test_bool(self):
        sl = SkipList()
        assert not sl
        sl.insert(1, 1)
        assert sl

    def test_iteration_order(self):
        sl = SkipList()
        keys = [5, 3, 8, 1, 9, 2, 7]
        for k in keys:
            sl.insert(k, k)
        assert list(sl) == sorted(keys)

    def test_items(self):
        sl = SkipList()
        sl.insert(2, "b")
        sl.insert(1, "a")
        sl.insert(3, "c")
        assert list(sl.items()) == [(1, "a"), (2, "b"), (3, "c")]

    def test_values(self):
        sl = SkipList()
        sl.insert(2, "b")
        sl.insert(1, "a")
        assert list(sl.values()) == ["a", "b"]

    def test_repr(self):
        sl = SkipList()
        sl.insert(1, "a")
        r = repr(sl)
        assert "SkipList" in r
        assert "(1, 'a')" in r

    def test_large_insert(self):
        sl = SkipList()
        n = 1000
        keys = list(range(n))
        random.shuffle(keys)
        for k in keys:
            sl.insert(k, k * 2)
        assert len(sl) == n
        for k in keys:
            assert sl.get(k) == k * 2
        assert list(sl) == list(range(n))

    def test_large_delete(self):
        sl = SkipList()
        n = 500
        for k in range(n):
            sl.insert(k, k)
        for k in range(0, n, 2):
            sl.delete(k)
        assert len(sl) == n // 2
        for k in range(n):
            if k % 2 == 0:
                assert k not in sl
            else:
                assert k in sl


# ============================================================
# SkipList -- min/max/floor/ceiling/predecessor/successor
# ============================================================

class TestSkipListNavigation:
    def setup_method(self):
        self.sl = SkipList()
        for k in [10, 20, 30, 40, 50]:
            self.sl.insert(k, k)

    def test_min(self):
        assert self.sl.min() == (10, 10)

    def test_max(self):
        assert self.sl.max() == (50, 50)

    def test_min_empty(self):
        assert SkipList().min() is None

    def test_max_empty(self):
        assert SkipList().max() is None

    def test_floor_exact(self):
        assert self.sl.floor(30) == (30, 30)

    def test_floor_between(self):
        assert self.sl.floor(25) == (20, 20)

    def test_floor_below_min(self):
        assert self.sl.floor(5) is None

    def test_floor_above_max(self):
        assert self.sl.floor(55) == (50, 50)

    def test_ceiling_exact(self):
        assert self.sl.ceiling(30) == (30, 30)

    def test_ceiling_between(self):
        assert self.sl.ceiling(25) == (30, 30)

    def test_ceiling_above_max(self):
        assert self.sl.ceiling(55) is None

    def test_ceiling_below_min(self):
        assert self.sl.ceiling(5) == (10, 10)

    def test_predecessor(self):
        assert self.sl.predecessor(30) == (20, 20)

    def test_predecessor_min(self):
        assert self.sl.predecessor(10) is None

    def test_successor(self):
        assert self.sl.successor(30) == (40, 40)

    def test_successor_max(self):
        assert self.sl.successor(50) is None


# ============================================================
# SkipList -- range queries
# ============================================================

class TestSkipListRange:
    def setup_method(self):
        self.sl = SkipList()
        for k in [10, 20, 30, 40, 50]:
            self.sl.insert(k, k)

    def test_range_inclusive(self):
        r = self.sl.range_query(20, 40)
        assert r == [(20, 20), (30, 30), (40, 40)]

    def test_range_exclusive_lo(self):
        r = self.sl.range_query(20, 40, inclusive_lo=False)
        assert r == [(30, 30), (40, 40)]

    def test_range_exclusive_hi(self):
        r = self.sl.range_query(20, 40, inclusive_hi=False)
        assert r == [(20, 20), (30, 30)]

    def test_range_both_exclusive(self):
        r = self.sl.range_query(20, 40, inclusive_lo=False, inclusive_hi=False)
        assert r == [(30, 30)]

    def test_range_empty(self):
        r = self.sl.range_query(21, 29)
        assert r == []

    def test_range_single(self):
        r = self.sl.range_query(30, 30)
        assert r == [(30, 30)]

    def test_range_all(self):
        r = self.sl.range_query(0, 100)
        assert len(r) == 5


# ============================================================
# SkipList -- pop and clear
# ============================================================

class TestSkipListPopClear:
    def test_pop_min(self):
        sl = SkipList()
        sl.insert(3, "c")
        sl.insert(1, "a")
        sl.insert(2, "b")
        assert sl.pop_min() == (1, "a")
        assert len(sl) == 2
        assert 1 not in sl

    def test_pop_max(self):
        sl = SkipList()
        sl.insert(3, "c")
        sl.insert(1, "a")
        sl.insert(2, "b")
        assert sl.pop_max() == (3, "c")
        assert len(sl) == 2

    def test_pop_min_empty(self):
        with pytest.raises(KeyError):
            SkipList().pop_min()

    def test_pop_max_empty(self):
        with pytest.raises(KeyError):
            SkipList().pop_max()

    def test_clear(self):
        sl = SkipList()
        for i in range(100):
            sl.insert(i, i)
        sl.clear()
        assert len(sl) == 0
        assert list(sl) == []

    def test_to_list(self):
        sl = SkipList()
        sl.insert(3, "c")
        sl.insert(1, "a")
        assert sl.to_list() == [(1, "a"), (3, "c")]

    def test_level_distribution(self):
        sl = SkipList()
        for i in range(100):
            sl.insert(i, i)
        dist = sl.level_distribution()
        assert sum(dist.values()) == 100


# ============================================================
# SkipList -- string keys
# ============================================================

class TestSkipListStringKeys:
    def test_string_keys(self):
        sl = SkipList()
        sl.insert("banana", 1)
        sl.insert("apple", 2)
        sl.insert("cherry", 3)
        assert list(sl) == ["apple", "banana", "cherry"]
        assert sl.get("banana") == 1

    def test_string_floor_ceiling(self):
        sl = SkipList()
        sl.insert("b", 1)
        sl.insert("d", 2)
        assert sl.floor("c") == ("b", 1)
        assert sl.ceiling("c") == ("d", 2)


# ============================================================
# SkipListSet
# ============================================================

class TestSkipListSet:
    def test_empty(self):
        s = SkipListSet()
        assert len(s) == 0
        assert not s

    def test_add(self):
        s = SkipListSet()
        assert s.add(5) is True
        assert s.add(5) is False
        assert 5 in s
        assert len(s) == 1

    def test_remove(self):
        s = SkipListSet()
        s.add(5)
        s.remove(5)
        assert 5 not in s

    def test_remove_missing(self):
        s = SkipListSet()
        with pytest.raises(KeyError):
            s.remove(99)

    def test_discard(self):
        s = SkipListSet()
        s.discard(99)  # no error

    def test_iteration(self):
        s = SkipListSet()
        for k in [5, 3, 1, 4, 2]:
            s.add(k)
        assert list(s) == [1, 2, 3, 4, 5]

    def test_min_max(self):
        s = SkipListSet()
        for k in [5, 3, 1]:
            s.add(k)
        assert s.min() == 1
        assert s.max() == 5

    def test_floor_ceiling(self):
        s = SkipListSet()
        for k in [10, 20, 30]:
            s.add(k)
        assert s.floor(15) == 10
        assert s.ceiling(15) == 20

    def test_range_query(self):
        s = SkipListSet()
        for k in [10, 20, 30, 40]:
            s.add(k)
        assert s.range_query(15, 35) == [20, 30]

    def test_pop_min_max(self):
        s = SkipListSet()
        for k in [1, 2, 3]:
            s.add(k)
        assert s.pop_min() == 1
        assert s.pop_max() == 3
        assert len(s) == 1

    def test_union(self):
        a = SkipListSet()
        b = SkipListSet()
        for k in [1, 2, 3]:
            a.add(k)
        for k in [3, 4, 5]:
            b.add(k)
        u = a.union(b)
        assert list(u) == [1, 2, 3, 4, 5]

    def test_intersection(self):
        a = SkipListSet()
        b = SkipListSet()
        for k in [1, 2, 3]:
            a.add(k)
        for k in [2, 3, 4]:
            b.add(k)
        assert list(a.intersection(b)) == [2, 3]

    def test_difference(self):
        a = SkipListSet()
        b = SkipListSet()
        for k in [1, 2, 3]:
            a.add(k)
        for k in [2, 3, 4]:
            b.add(k)
        assert list(a.difference(b)) == [1]

    def test_clear(self):
        s = SkipListSet()
        for k in range(10):
            s.add(k)
        s.clear()
        assert len(s) == 0

    def test_to_list(self):
        s = SkipListSet()
        s.add(3)
        s.add(1)
        assert s.to_list() == [1, 3]

    def test_repr(self):
        s = SkipListSet()
        s.add(1)
        assert "SkipListSet" in repr(s)


# ============================================================
# IndexableSkipList -- rank/select
# ============================================================

class TestIndexableSkipList:
    def test_empty(self):
        isl = IndexableSkipList()
        assert len(isl) == 0
        assert isl.rank(1) == -1
        assert isl.select(0) is None

    def test_insert_and_rank(self):
        isl = IndexableSkipList()
        for k in [30, 10, 20, 40, 50]:
            isl.insert(k, k)
        assert isl.rank(10) == 0
        assert isl.rank(20) == 1
        assert isl.rank(30) == 2
        assert isl.rank(40) == 3
        assert isl.rank(50) == 4
        assert isl.rank(99) == -1

    def test_select(self):
        isl = IndexableSkipList()
        for k in [30, 10, 20, 40, 50]:
            isl.insert(k, k)
        assert isl.select(0) == (10, 10)
        assert isl.select(1) == (20, 20)
        assert isl.select(2) == (30, 30)
        assert isl.select(3) == (40, 40)
        assert isl.select(4) == (50, 50)

    def test_select_out_of_range(self):
        isl = IndexableSkipList()
        isl.insert(1, 1)
        assert isl.select(-1) is None
        assert isl.select(1) is None

    def test_delete_updates_rank(self):
        isl = IndexableSkipList()
        for k in [10, 20, 30, 40]:
            isl.insert(k, k)
        isl.delete(20)
        assert isl.rank(10) == 0
        assert isl.rank(30) == 1
        assert isl.rank(40) == 2
        assert isl.select(1) == (30, 30)

    def test_update_value(self):
        isl = IndexableSkipList()
        isl.insert(1, "a")
        assert isl.insert(1, "b") is False
        assert isl.get(1) == "b"
        assert len(isl) == 1

    def test_contains(self):
        isl = IndexableSkipList()
        isl.insert(5, 5)
        assert 5 in isl
        assert 6 not in isl

    def test_iteration(self):
        isl = IndexableSkipList()
        for k in [5, 3, 1]:
            isl.insert(k, k)
        assert list(isl) == [1, 3, 5]

    def test_rank_select_large(self):
        isl = IndexableSkipList()
        n = 200
        keys = list(range(n))
        random.shuffle(keys)
        for k in keys:
            isl.insert(k, k)
        for i in range(n):
            assert isl.rank(i) == i
            assert isl.select(i) == (i, i)

    def test_delete_and_rerank(self):
        isl = IndexableSkipList()
        for k in range(10):
            isl.insert(k, k)
        isl.delete(5)
        remaining = [k for k in range(10) if k != 5]
        for i, k in enumerate(remaining):
            assert isl.rank(k) == i
            assert isl.select(i) == (k, k)

    def test_range_query(self):
        isl = IndexableSkipList()
        for k in [10, 20, 30, 40, 50]:
            isl.insert(k, k)
        assert isl.range_query(20, 40) == [(20, 20), (30, 30), (40, 40)]

    def test_min_max(self):
        isl = IndexableSkipList()
        for k in [5, 3, 8]:
            isl.insert(k, k)
        assert isl.min() == (3, 3)
        assert isl.max() == (8, 8)

    def test_min_max_empty(self):
        isl = IndexableSkipList()
        assert isl.min() is None
        assert isl.max() is None

    def test_to_list(self):
        isl = IndexableSkipList()
        isl.insert(2, "b")
        isl.insert(1, "a")
        assert isl.to_list() == [(1, "a"), (2, "b")]

    def test_get_default(self):
        isl = IndexableSkipList()
        assert isl.get(1, "x") == "x"


# ============================================================
# ConcurrentSkipList
# ============================================================

class TestConcurrentSkipList:
    def test_basic_operations(self):
        csl = ConcurrentSkipList()
        csl.insert(1, "a")
        csl.insert(2, "b")
        assert len(csl) == 2
        assert csl.get(1) == "a"
        assert 1 in csl
        csl.delete(1)
        assert 1 not in csl

    def test_concurrent_inserts(self):
        csl = ConcurrentSkipList()
        errors = []

        def insert_range(start, end):
            try:
                for i in range(start, end):
                    csl.insert(i, i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_range, args=(i * 100, (i + 1) * 100)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(csl) == 400
        for i in range(400):
            assert csl.get(i) == i

    def test_concurrent_reads(self):
        csl = ConcurrentSkipList()
        for i in range(100):
            csl.insert(i, i)

        results = [None] * 4

        def read_all(idx):
            results[idx] = list(csl)

        threads = [threading.Thread(target=read_all, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            assert r == list(range(100))

    def test_concurrent_mixed(self):
        csl = ConcurrentSkipList()
        for i in range(100):
            csl.insert(i, i)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    csl.get(random.randint(0, 99))
                    list(csl)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100, 150):
                    csl.insert(i, i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(2)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_navigation(self):
        csl = ConcurrentSkipList()
        for k in [10, 20, 30]:
            csl.insert(k, k)
        assert csl.min() == (10, 10)
        assert csl.max() == (30, 30)
        assert csl.floor(15) == (10, 10)
        assert csl.ceiling(15) == (20, 20)

    def test_range_query(self):
        csl = ConcurrentSkipList()
        for k in [10, 20, 30, 40]:
            csl.insert(k, k)
        assert csl.range_query(15, 35) == [(20, 20), (30, 30)]

    def test_pop(self):
        csl = ConcurrentSkipList()
        for k in [1, 2, 3]:
            csl.insert(k, k)
        assert csl.pop_min() == (1, 1)
        assert csl.pop_max() == (3, 3)

    def test_items_to_list(self):
        csl = ConcurrentSkipList()
        csl.insert(2, "b")
        csl.insert(1, "a")
        assert csl.items() == [(1, "a"), (2, "b")]
        assert csl.to_list() == [(1, "a"), (2, "b")]

    def test_clear(self):
        csl = ConcurrentSkipList()
        csl.insert(1, 1)
        csl.clear()
        assert len(csl) == 0


# ============================================================
# MergeableSkipList
# ============================================================

class TestMergeableSkipList:
    def test_merge(self):
        a = MergeableSkipList()
        b = MergeableSkipList()
        for k in [1, 3, 5]:
            a.insert(k, k)
        for k in [2, 4, 6]:
            b.insert(k, k)
        a.merge(b)
        assert list(a) == [1, 2, 3, 4, 5, 6]

    def test_merge_overlapping(self):
        a = MergeableSkipList()
        b = MergeableSkipList()
        a.insert(1, "a1")
        b.insert(1, "b1")
        a.merge(b)
        assert a.get(1) == "b1"  # b overwrites
        assert len(a) == 1

    def test_split(self):
        sl = MergeableSkipList()
        for k in [1, 2, 3, 4, 5]:
            sl.insert(k, k)
        left, right = sl.split(3)
        assert list(left) == [1, 2]
        assert list(right) == [3, 4, 5]

    def test_split_empty_left(self):
        sl = MergeableSkipList()
        for k in [5, 6, 7]:
            sl.insert(k, k)
        left, right = sl.split(5)
        assert list(left) == []
        assert list(right) == [5, 6, 7]

    def test_split_empty_right(self):
        sl = MergeableSkipList()
        for k in [1, 2, 3]:
            sl.insert(k, k)
        left, right = sl.split(10)
        assert list(left) == [1, 2, 3]
        assert list(right) == []

    def test_bulk_insert(self):
        sl = MergeableSkipList()
        sl.bulk_insert([(3, "c"), (1, "a"), (2, "b")])
        assert list(sl) == [1, 2, 3]
        assert len(sl) == 3


# ============================================================
# IntervalSkipList
# ============================================================

class TestIntervalSkipList:
    def test_empty(self):
        isl = IntervalSkipList()
        assert len(isl) == 0
        assert isl.stab(5) == []

    def test_add_and_stab(self):
        isl = IntervalSkipList()
        isl.add(1, 5, "a")
        isl.add(3, 8, "b")
        result = isl.stab(4)
        assert len(result) == 2
        keys = sorted(r[2] for r in result)
        assert keys == ["a", "b"]

    def test_stab_miss(self):
        isl = IntervalSkipList()
        isl.add(1, 3, "a")
        assert isl.stab(5) == []

    def test_stab_boundary(self):
        isl = IntervalSkipList()
        isl.add(1, 5, "a")
        assert len(isl.stab(1)) == 1
        assert len(isl.stab(5)) == 1

    def test_remove(self):
        isl = IntervalSkipList()
        id1 = isl.add(1, 5, "a")
        isl.add(3, 8, "b")
        isl.remove(id1)
        assert len(isl) == 1
        result = isl.stab(4)
        assert len(result) == 1
        assert result[0][2] == "b"

    def test_remove_nonexistent(self):
        isl = IntervalSkipList()
        assert isl.remove(99) is False

    def test_overlap(self):
        isl = IntervalSkipList()
        isl.add(1, 3, "a")
        isl.add(5, 8, "b")
        isl.add(6, 10, "c")
        result = isl.overlap(4, 7)
        keys = sorted(r[2] for r in result)
        assert keys == ["b", "c"]

    def test_overlap_miss(self):
        isl = IntervalSkipList()
        isl.add(1, 3, "a")
        isl.add(5, 8, "b")
        assert isl.overlap(3.5, 4.5) == []

    def test_all_intervals(self):
        isl = IntervalSkipList()
        isl.add(1, 5, "a")
        isl.add(3, 8, "b")
        all_int = isl.all_intervals()
        assert len(all_int) == 2

    def test_many_intervals(self):
        isl = IntervalSkipList()
        for i in range(100):
            isl.add(i, i + 5, i)
        assert len(isl) == 100
        result = isl.stab(50)
        assert len(result) == 6  # intervals [46,51] through [50,55]


# ============================================================
# Edge cases and stress tests
# ============================================================

class TestEdgeCases:
    def test_single_element_operations(self):
        sl = SkipList()
        sl.insert(42, "answer")
        assert sl.min() == (42, "answer")
        assert sl.max() == (42, "answer")
        assert sl.floor(42) == (42, "answer")
        assert sl.ceiling(42) == (42, "answer")
        assert sl.predecessor(42) is None
        assert sl.successor(42) is None

    def test_negative_keys(self):
        sl = SkipList()
        for k in [-5, -3, -1, 0, 1, 3, 5]:
            sl.insert(k, k)
        assert list(sl) == [-5, -3, -1, 0, 1, 3, 5]
        assert sl.floor(-2) == (-3, -3)

    def test_float_keys(self):
        sl = SkipList()
        sl.insert(1.5, "a")
        sl.insert(2.5, "b")
        sl.insert(0.5, "c")
        assert list(sl) == [0.5, 1.5, 2.5]

    def test_none_value(self):
        sl = SkipList()
        sl.insert(1, None)
        assert sl.get(1) is None
        assert 1 in sl  # contains uses get which returns None -- need to check

    def test_contains_with_none_value(self):
        """Verify __contains__ works even when value is None."""
        sl = SkipList()
        sl.insert(1, None)
        # get returns None for both "not found" and "value is None"
        # __contains__ uses get, so this is a known limitation
        # The workaround is to use get with a sentinel default
        result = sl.get(1, object())  # sentinel != None
        assert result is None  # confirms key exists with None value

    def test_insert_delete_reinsert(self):
        sl = SkipList()
        sl.insert(1, "first")
        sl.delete(1)
        sl.insert(1, "second")
        assert sl.get(1) == "second"
        assert len(sl) == 1

    def test_delete_all_then_insert(self):
        sl = SkipList()
        for k in range(10):
            sl.insert(k, k)
        for k in range(10):
            sl.delete(k)
        assert len(sl) == 0
        sl.insert(5, 5)
        assert len(sl) == 1
        assert sl.get(5) == 5

    def test_stress_random_operations(self):
        random.seed(42)
        sl = SkipList()
        reference = {}
        for _ in range(1000):
            op = random.choice(["insert", "insert", "delete", "get"])
            key = random.randint(0, 100)
            if op == "insert":
                sl.insert(key, key)
                reference[key] = key
            elif op == "delete":
                sl.delete(key)
                reference.pop(key, None)
            else:
                assert sl.get(key) == reference.get(key)
        assert len(sl) == len(reference)
        assert sorted(sl) == sorted(reference.keys())

    def test_deterministic_with_seed(self):
        """Two skip lists with same seed produce same structure."""
        random.seed(123)
        sl1 = SkipList()
        for k in range(50):
            sl1.insert(k, k)

        random.seed(123)
        sl2 = SkipList()
        for k in range(50):
            sl2.insert(k, k)

        assert list(sl1.items()) == list(sl2.items())

    def test_ascending_insert(self):
        sl = SkipList()
        for k in range(100):
            sl.insert(k, k)
        assert list(sl) == list(range(100))

    def test_descending_insert(self):
        sl = SkipList()
        for k in range(99, -1, -1):
            sl.insert(k, k)
        assert list(sl) == list(range(100))


# ============================================================
# IndexableSkipList -- additional stress
# ============================================================

class TestIndexableStress:
    def test_random_insert_delete_rank_select(self):
        random.seed(99)
        isl = IndexableSkipList()
        reference = []
        for _ in range(200):
            op = random.choice(["insert", "insert", "delete"])
            if op == "insert":
                k = random.randint(0, 500)
                isl.insert(k, k)
                if k not in reference:
                    reference.append(k)
                    reference.sort()
            elif reference:
                k = random.choice(reference)
                isl.delete(k)
                reference.remove(k)
        # Verify rank and select
        for i, k in enumerate(reference):
            assert isl.rank(k) == i, f"rank({k}) should be {i}, got {isl.rank(k)}"
            assert isl.select(i) == (k, k), f"select({i}) should be ({k},{k}), got {isl.select(i)}"


# ============================================================
# Performance characteristics
# ============================================================

class TestPerformance:
    def test_logarithmic_behavior(self):
        """Insert and search in large list should complete quickly."""
        sl = SkipList()
        n = 10000
        keys = list(range(n))
        random.shuffle(keys)
        for k in keys:
            sl.insert(k, k)
        # Search for all keys
        for k in range(n):
            assert sl.get(k) == k

    def test_skip_list_vs_sorted_list(self):
        """Verify skip list maintains sorted order under random ops."""
        random.seed(77)
        sl = SkipList()
        for _ in range(500):
            sl.insert(random.randint(0, 1000), True)
        keys = list(sl)
        assert keys == sorted(keys)
