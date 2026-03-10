"""
Tests for C079: Skip List

Comprehensive tests covering:
- Basic operations (insert, get, delete)
- Order statistics (rank, select)
- Range queries
- Floor/ceiling
- Min/max, pop_min/pop_max
- Iteration
- Bulk operations (map, filter, reduce, slice)
- Set operations (merge, intersection, difference, diff)
- Persistent skip list
- Edge cases and stress tests
"""

import pytest
from skip_list import (
    SkipList, SkipNode, PersistentSkipList, PersistentSkipNode,
    merge, diff, intersection, difference,
)


# ===== Basic Construction =====

class TestConstruction:
    def test_empty_skip_list(self):
        sl = SkipList(seed=42)
        assert len(sl) == 0
        assert not sl

    def test_single_insert(self):
        sl = SkipList(seed=42)
        assert sl.insert(5, "five") is True
        assert len(sl) == 1
        assert sl

    def test_insert_returns_false_on_update(self):
        sl = SkipList(seed=42)
        sl.insert(5, "five")
        assert sl.insert(5, "FIVE") is False
        assert len(sl) == 1
        assert sl[5] == "FIVE"

    def test_multiple_inserts(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i * 10)
        assert len(sl) == 10

    def test_custom_max_level(self):
        sl = SkipList(max_level=4, seed=42)
        for i in range(100):
            sl.insert(i, i)
        assert len(sl) == 100

    def test_custom_probability(self):
        sl = SkipList(p=0.25, seed=42)
        for i in range(50):
            sl.insert(i, i)
        assert len(sl) == 50


# ===== Get / Contains =====

class TestGetContains:
    def test_get_existing(self):
        sl = SkipList(seed=42)
        sl.insert(10, "ten")
        assert sl.get(10) == "ten"

    def test_get_missing_default(self):
        sl = SkipList(seed=42)
        assert sl.get(10) is None
        assert sl.get(10, "default") == "default"

    def test_getitem(self):
        sl = SkipList(seed=42)
        sl.insert(3, "three")
        assert sl[3] == "three"

    def test_getitem_missing_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(KeyError):
            _ = sl[99]

    def test_contains(self):
        sl = SkipList(seed=42)
        sl.insert(7, "seven")
        assert 7 in sl
        assert 8 not in sl

    def test_setitem(self):
        sl = SkipList(seed=42)
        sl[5] = "five"
        assert sl[5] == "five"

    def test_none_value(self):
        sl = SkipList(seed=42)
        sl.insert(1)  # default value is None
        assert 1 in sl
        assert sl[1] is None


# ===== Delete =====

class TestDelete:
    def test_delete_existing(self):
        sl = SkipList(seed=42)
        sl.insert(5, "five")
        assert sl.delete(5) is True
        assert 5 not in sl
        assert len(sl) == 0

    def test_delete_missing(self):
        sl = SkipList(seed=42)
        assert sl.delete(5) is False

    def test_delitem(self):
        sl = SkipList(seed=42)
        sl.insert(3, "three")
        del sl[3]
        assert 3 not in sl

    def test_delitem_missing_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(KeyError):
            del sl[99]

    def test_delete_head(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        sl.delete(0)
        assert sl.keys() == [1, 2, 3, 4]

    def test_delete_tail(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        sl.delete(4)
        assert sl.keys() == [0, 1, 2, 3]

    def test_delete_middle(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        sl.delete(2)
        assert sl.keys() == [0, 1, 3, 4]

    def test_delete_all(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        for i in range(10):
            sl.delete(i)
        assert len(sl) == 0
        assert sl.keys() == []

    def test_reinsert_after_delete(self):
        sl = SkipList(seed=42)
        sl.insert(5, "first")
        sl.delete(5)
        sl.insert(5, "second")
        assert sl[5] == "second"


# ===== Ordering =====

class TestOrdering:
    def test_sorted_keys(self):
        sl = SkipList(seed=42)
        for x in [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]:
            sl.insert(x, x)
        assert sl.keys() == list(range(10))

    def test_string_keys(self):
        sl = SkipList(seed=42)
        for w in ["banana", "apple", "cherry", "date"]:
            sl.insert(w, len(w))
        assert sl.keys() == ["apple", "banana", "cherry", "date"]

    def test_negative_keys(self):
        sl = SkipList(seed=42)
        for x in [-5, 3, -1, 0, 7, -3]:
            sl.insert(x, x)
        assert sl.keys() == [-5, -3, -1, 0, 3, 7]

    def test_float_keys(self):
        sl = SkipList(seed=42)
        for x in [3.14, 1.41, 2.72, 0.57]:
            sl.insert(x, str(x))
        assert sl.keys() == [0.57, 1.41, 2.72, 3.14]


# ===== Min / Max =====

class TestMinMax:
    def test_min(self):
        sl = SkipList(seed=42)
        sl.insert(5, "five")
        sl.insert(2, "two")
        sl.insert(8, "eight")
        assert sl.min() == (2, "two")

    def test_max(self):
        sl = SkipList(seed=42)
        sl.insert(5, "five")
        sl.insert(2, "two")
        sl.insert(8, "eight")
        assert sl.max() == (8, "eight")

    def test_min_empty_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(ValueError):
            sl.min()

    def test_max_empty_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(ValueError):
            sl.max()

    def test_pop_min(self):
        sl = SkipList(seed=42)
        sl.insert(3, "three")
        sl.insert(1, "one")
        sl.insert(5, "five")
        assert sl.pop_min() == (1, "one")
        assert len(sl) == 2
        assert sl.min() == (3, "three")

    def test_pop_max(self):
        sl = SkipList(seed=42)
        sl.insert(3, "three")
        sl.insert(1, "one")
        sl.insert(5, "five")
        assert sl.pop_max() == (5, "five")
        assert len(sl) == 2
        assert sl.max() == (3, "three")

    def test_pop_min_empty_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(ValueError):
            sl.pop_min()

    def test_pop_max_empty_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(ValueError):
            sl.pop_max()


# ===== Floor / Ceiling =====

class TestFloorCeiling:
    def test_floor_exact(self):
        sl = SkipList(seed=42)
        for i in [2, 4, 6, 8]:
            sl.insert(i, i)
        assert sl.floor(4) == (4, 4)

    def test_floor_between(self):
        sl = SkipList(seed=42)
        for i in [2, 4, 6, 8]:
            sl.insert(i, i)
        assert sl.floor(5) == (4, 4)

    def test_floor_below_min(self):
        sl = SkipList(seed=42)
        sl.insert(5, 5)
        assert sl.floor(3) is None

    def test_ceiling_exact(self):
        sl = SkipList(seed=42)
        for i in [2, 4, 6, 8]:
            sl.insert(i, i)
        assert sl.ceiling(4) == (4, 4)

    def test_ceiling_between(self):
        sl = SkipList(seed=42)
        for i in [2, 4, 6, 8]:
            sl.insert(i, i)
        assert sl.ceiling(5) == (6, 6)

    def test_ceiling_above_max(self):
        sl = SkipList(seed=42)
        sl.insert(5, 5)
        assert sl.ceiling(7) is None

    def test_floor_at_max(self):
        sl = SkipList(seed=42)
        for i in [1, 3, 5]:
            sl.insert(i, i)
        assert sl.floor(10) == (5, 5)

    def test_ceiling_at_min(self):
        sl = SkipList(seed=42)
        for i in [5, 10, 15]:
            sl.insert(i, i)
        assert sl.ceiling(1) == (5, 5)


# ===== Rank / Select =====

class TestRankSelect:
    def test_rank_of_elements(self):
        sl = SkipList(seed=42)
        for i in [10, 20, 30, 40, 50]:
            sl.insert(i, i)
        assert sl.rank(10) == 0
        assert sl.rank(30) == 2
        assert sl.rank(50) == 4

    def test_rank_of_missing(self):
        sl = SkipList(seed=42)
        for i in [10, 20, 30]:
            sl.insert(i, i)
        assert sl.rank(15) == 1  # 1 element < 15
        assert sl.rank(25) == 2

    def test_rank_below_all(self):
        sl = SkipList(seed=42)
        sl.insert(10, 10)
        assert sl.rank(5) == 0

    def test_rank_above_all(self):
        sl = SkipList(seed=42)
        sl.insert(10, 10)
        assert sl.rank(15) == 1

    def test_select_valid(self):
        sl = SkipList(seed=42)
        for i in [10, 20, 30, 40, 50]:
            sl.insert(i, i)
        assert sl.select(0) == (10, 10)
        assert sl.select(2) == (30, 30)
        assert sl.select(4) == (50, 50)

    def test_select_negative_index(self):
        sl = SkipList(seed=42)
        for i in [10, 20, 30]:
            sl.insert(i, i)
        assert sl.select(-1) == (30, 30)
        assert sl.select(-3) == (10, 10)

    def test_select_out_of_range(self):
        sl = SkipList(seed=42)
        sl.insert(1, 1)
        with pytest.raises(IndexError):
            sl.select(5)
        with pytest.raises(IndexError):
            sl.select(-5)

    def test_rank_select_inverse(self):
        sl = SkipList(seed=42)
        for i in range(20):
            sl.insert(i * 3, i)
        for r in range(20):
            k, v = sl.select(r)
            assert sl.rank(k) == r


# ===== Range Queries =====

class TestRange:
    def test_full_range(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        assert sl.range() == [(i, i) for i in range(5)]

    def test_inclusive_range(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(3, 7)
        assert result == [(i, i) for i in range(3, 8)]

    def test_exclusive_lo(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(3, 7, lo_inclusive=False)
        assert result == [(i, i) for i in range(4, 8)]

    def test_exclusive_hi(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(3, 7, hi_inclusive=False)
        assert result == [(i, i) for i in range(3, 7)]

    def test_both_exclusive(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(3, 7, lo_inclusive=False, hi_inclusive=False)
        assert result == [(i, i) for i in range(4, 7)]

    def test_unbounded_lo(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(hi=3)
        assert result == [(i, i) for i in range(4)]

    def test_unbounded_hi(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(lo=7)
        assert result == [(i, i) for i in range(7, 10)]

    def test_empty_range(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(20, 30)
        assert result == []

    def test_single_element_range(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = sl.range(5, 5)
        assert result == [(5, 5)]


# ===== Iteration =====

class TestIteration:
    def test_iter_keys(self):
        sl = SkipList(seed=42)
        for i in [5, 3, 1, 4, 2]:
            sl.insert(i, i)
        assert list(sl) == [1, 2, 3, 4, 5]

    def test_iter_items(self):
        sl = SkipList(seed=42)
        sl.insert(1, "a")
        sl.insert(2, "b")
        assert sl.items() == [(1, "a"), (2, "b")]

    def test_iter_values(self):
        sl = SkipList(seed=42)
        sl.insert(1, "a")
        sl.insert(2, "b")
        assert sl.values() == ["a", "b"]

    def test_iter_from(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        result = list(sl.iter_from(5))
        assert result == [(i, i) for i in range(5, 10)]

    def test_iter_from_missing_key(self):
        sl = SkipList(seed=42)
        for i in [0, 2, 4, 6, 8]:
            sl.insert(i, i)
        result = list(sl.iter_from(3))
        assert result == [(4, 4), (6, 6), (8, 8)]

    def test_iter_empty(self):
        sl = SkipList(seed=42)
        assert list(sl) == []


# ===== Clear / Copy / Update =====

class TestClearCopyUpdate:
    def test_clear(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        sl.clear()
        assert len(sl) == 0
        assert sl.keys() == []

    def test_copy(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        copy = sl.copy()
        assert copy.items() == sl.items()
        copy.insert(10, 10)
        assert 10 not in sl
        assert 10 in copy

    def test_update(self):
        sl = SkipList(seed=42)
        sl.update([(1, "a"), (2, "b"), (3, "c")])
        assert sl.items() == [(1, "a"), (2, "b"), (3, "c")]

    def test_update_overwrites(self):
        sl = SkipList(seed=42)
        sl.insert(1, "old")
        sl.update([(1, "new"), (2, "two")])
        assert sl[1] == "new"


# ===== Equality / Repr =====

class TestEqualityRepr:
    def test_equality(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        for i in range(5):
            sl1.insert(i, i)
            sl2.insert(i, i)
        assert sl1 == sl2

    def test_inequality_size(self):
        sl1 = SkipList(seed=42)
        sl2 = SkipList(seed=42)
        sl1.insert(1, 1)
        assert sl1 != sl2

    def test_inequality_values(self):
        sl1 = SkipList(seed=42)
        sl2 = SkipList(seed=42)
        sl1.insert(1, "a")
        sl2.insert(1, "b")
        assert sl1 != sl2

    def test_repr(self):
        sl = SkipList(seed=42)
        sl.insert(1, "one")
        r = repr(sl)
        assert "SkipList" in r
        assert "(1, 'one')" in r

    def test_not_equal_to_other_types(self):
        sl = SkipList(seed=42)
        assert sl != "not a skip list"


# ===== Bulk Operations =====

class TestBulkOperations:
    def test_map(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        doubled = sl.map(lambda k, v: v * 2)
        assert doubled.items() == [(i, i * 2) for i in range(5)]

    def test_filter(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        evens = sl.filter(lambda k, v: k % 2 == 0)
        assert evens.keys() == [0, 2, 4, 6, 8]

    def test_reduce(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        total = sl.reduce(lambda acc, item: acc + item[1], 0)
        assert total == 10

    def test_reduce_no_initial(self):
        sl = SkipList(seed=42)
        sl.insert(1, 10)
        sl.insert(2, 20)
        result = sl.reduce(lambda acc, item: (acc[0] + item[0], acc[1] + item[1]))
        assert result == (3, 30)

    def test_reduce_empty_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(ValueError):
            sl.reduce(lambda acc, item: acc)

    def test_slice(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i * 10)
        s = sl.slice(3, 7)
        assert s.keys() == [3, 4, 5, 6]

    def test_slice_negative(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        s = sl.slice(-3, 100)
        assert s.keys() == [7, 8, 9]

    def test_nearest(self):
        sl = SkipList(seed=42)
        for i in [10, 20, 30]:
            sl.insert(i, i)
        assert sl.nearest(15) == (10, 10)  # tie goes to smaller
        assert sl.nearest(16) == (20, 20)  # 20 is closer
        assert sl.nearest(10) == (10, 10)
        assert sl.nearest(5) == (10, 10)
        assert sl.nearest(35) == (30, 30)

    def test_nearest_empty_raises(self):
        sl = SkipList(seed=42)
        with pytest.raises(ValueError):
            sl.nearest(5)


# ===== Merge =====

class TestMerge:
    def test_merge_disjoint(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        for i in range(5):
            sl1.insert(i, i)
        for i in range(5, 10):
            sl2.insert(i, i)
        result = merge(sl1, sl2)
        assert result.keys() == list(range(10))

    def test_merge_overlapping_last(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, "a")
        sl2.insert(1, "b")
        result = merge(sl1, sl2, conflict='last')
        assert result[1] == "b"

    def test_merge_overlapping_first(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, "a")
        sl2.insert(1, "b")
        result = merge(sl1, sl2, conflict='first')
        assert result[1] == "a"

    def test_merge_overlapping_callable(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, 10)
        sl2.insert(1, 20)
        result = merge(sl1, sl2, conflict=lambda k, v1, v2: v1 + v2)
        assert result[1] == 30

    def test_merge_empty(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, 1)
        result = merge(sl1, sl2)
        assert result.keys() == [1]


# ===== Diff =====

class TestDiff:
    def test_diff_identical(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        for i in range(5):
            sl1.insert(i, i)
            sl2.insert(i, i)
        only1, only2, differ = diff(sl1, sl2)
        assert only1 == []
        assert only2 == []
        assert differ == []

    def test_diff_disjoint(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, 1)
        sl2.insert(2, 2)
        only1, only2, differ = diff(sl1, sl2)
        assert only1 == [1]
        assert only2 == [2]
        assert differ == []

    def test_diff_value_differences(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, "a")
        sl2.insert(1, "b")
        only1, only2, differ = diff(sl1, sl2)
        assert only1 == []
        assert only2 == []
        assert differ == [1]


# ===== Intersection / Difference =====

class TestSetOperations:
    def test_intersection(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        for i in range(5):
            sl1.insert(i, i)
        for i in range(3, 8):
            sl2.insert(i, i * 10)
        result = intersection(sl1, sl2)
        assert result.keys() == [3, 4]
        assert result[3] == 3  # takes value from sl1

    def test_intersection_empty(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, 1)
        sl2.insert(2, 2)
        result = intersection(sl1, sl2)
        assert len(result) == 0

    def test_difference(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        for i in range(5):
            sl1.insert(i, i)
        for i in range(3, 8):
            sl2.insert(i, i)
        result = difference(sl1, sl2)
        assert result.keys() == [0, 1, 2]

    def test_difference_nothing_removed(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        sl1.insert(1, 1)
        sl2.insert(2, 2)
        result = difference(sl1, sl2)
        assert result.keys() == [1]


# ===== SkipNode =====

class TestSkipNode:
    def test_node_creation(self):
        node = SkipNode(5, "five", 3)
        assert node.key == 5
        assert node.value == "five"
        assert node.level == 3
        assert len(node.forward) == 4

    def test_node_repr(self):
        node = SkipNode(5, "five", 2)
        r = repr(node)
        assert "5" in r
        assert "five" in r


# ===== Stress Tests =====

class TestStress:
    def test_large_insert_delete(self):
        sl = SkipList(seed=42)
        n = 1000
        for i in range(n):
            sl.insert(i, i)
        assert len(sl) == n
        assert sl.keys() == list(range(n))
        for i in range(0, n, 2):
            sl.delete(i)
        assert len(sl) == n // 2
        assert sl.keys() == list(range(1, n, 2))

    def test_random_operations(self):
        import random
        rng = random.Random(123)
        sl = SkipList(seed=42)
        reference = {}
        for _ in range(500):
            op = rng.choice(['insert', 'insert', 'delete', 'get'])
            key = rng.randint(0, 100)
            if op == 'insert':
                val = rng.randint(0, 1000)
                sl.insert(key, val)
                reference[key] = val
            elif op == 'delete':
                sl.delete(key)
                reference.pop(key, None)
            elif op == 'get':
                if key in reference:
                    assert sl[key] == reference[key]
                else:
                    assert key not in sl
        assert sorted(reference.keys()) == sl.keys()

    def test_rank_select_consistency(self):
        sl = SkipList(seed=42)
        for i in range(200):
            sl.insert(i * 7, i)
        for r in range(200):
            k, v = sl.select(r)
            assert sl.rank(k) == r

    def test_insert_reverse_order(self):
        sl = SkipList(seed=42)
        for i in range(100, -1, -1):
            sl.insert(i, i)
        assert sl.keys() == list(range(101))

    def test_delete_all_reinsert(self):
        sl = SkipList(seed=42)
        for i in range(50):
            sl.insert(i, i)
        for i in range(50):
            sl.delete(i)
        assert len(sl) == 0
        for i in range(50):
            sl.insert(i, i * 100)
        assert len(sl) == 50
        assert sl[25] == 2500


# ===== Persistent Skip List =====

class TestPersistentConstruction:
    def test_empty(self):
        psl = PersistentSkipList(seed=42)
        assert len(psl) == 0
        assert not psl

    def test_single_insert(self):
        psl = PersistentSkipList(seed=42)
        psl2 = psl.insert(5, "five")
        assert len(psl) == 0  # original unchanged
        assert len(psl2) == 1
        assert psl2[5] == "five"

    def test_multiple_inserts(self):
        psl = PersistentSkipList(seed=42)
        versions = [psl]
        for i in range(5):
            versions.append(versions[-1].insert(i, i * 10))
        # Check all versions are intact
        for i, v in enumerate(versions):
            assert len(v) == i

    def test_from_items(self):
        psl = PersistentSkipList.from_items([(i, i) for i in range(10)], seed=42)
        assert len(psl) == 10
        assert psl.keys() == list(range(10))


class TestPersistentGet:
    def test_get_existing(self):
        psl = PersistentSkipList(seed=42)
        psl = psl.insert(10, "ten")
        assert psl.get(10) == "ten"

    def test_get_missing(self):
        psl = PersistentSkipList(seed=42)
        assert psl.get(10) is None
        assert psl.get(10, "default") == "default"

    def test_getitem_missing_raises(self):
        psl = PersistentSkipList(seed=42)
        with pytest.raises(KeyError):
            _ = psl[99]

    def test_contains(self):
        psl = PersistentSkipList(seed=42).insert(7, 7)
        assert 7 in psl
        assert 8 not in psl


class TestPersistentUpdate:
    def test_update_preserves_old(self):
        psl1 = PersistentSkipList(seed=42).insert(1, "old")
        psl2 = psl1.insert(1, "new")
        assert psl1[1] == "old"
        assert psl2[1] == "new"

    def test_update_size_unchanged(self):
        psl = PersistentSkipList(seed=42).insert(1, "a")
        psl2 = psl.insert(1, "b")
        assert len(psl2) == 1


class TestPersistentDelete:
    def test_delete(self):
        psl = PersistentSkipList.from_items([(i, i) for i in range(5)], seed=42)
        psl2 = psl.delete(2)
        assert len(psl) == 5  # original unchanged
        assert len(psl2) == 4
        assert 2 in psl
        assert 2 not in psl2

    def test_delete_missing_raises(self):
        psl = PersistentSkipList(seed=42)
        with pytest.raises(KeyError):
            psl.delete(99)

    def test_delete_all(self):
        psl = PersistentSkipList.from_items([(i, i) for i in range(5)], seed=42)
        for i in range(5):
            psl = psl.delete(i)
        assert len(psl) == 0


class TestPersistentOrdering:
    def test_sorted_keys(self):
        psl = PersistentSkipList(seed=42)
        for x in [5, 3, 8, 1, 9]:
            psl = psl.insert(x, x)
        assert psl.keys() == [1, 3, 5, 8, 9]

    def test_items(self):
        psl = PersistentSkipList(seed=42)
        for i in range(5):
            psl = psl.insert(i, i * 10)
        assert psl.items() == [(i, i * 10) for i in range(5)]

    def test_values(self):
        psl = PersistentSkipList(seed=42)
        for i in range(3):
            psl = psl.insert(i, chr(65 + i))
        assert psl.values() == ['A', 'B', 'C']


class TestPersistentMinMax:
    def test_min(self):
        psl = PersistentSkipList.from_items([(3, "c"), (1, "a"), (5, "e")], seed=42)
        assert psl.min() == (1, "a")

    def test_max(self):
        psl = PersistentSkipList.from_items([(3, "c"), (1, "a"), (5, "e")], seed=42)
        assert psl.max() == (5, "e")

    def test_min_empty_raises(self):
        psl = PersistentSkipList(seed=42)
        with pytest.raises(ValueError):
            psl.min()

    def test_max_empty_raises(self):
        psl = PersistentSkipList(seed=42)
        with pytest.raises(ValueError):
            psl.max()


class TestPersistentFloorCeiling:
    def test_floor(self):
        psl = PersistentSkipList.from_items([(2, 2), (4, 4), (6, 6)], seed=42)
        assert psl.floor(3) == (2, 2)
        assert psl.floor(4) == (4, 4)

    def test_floor_below_min(self):
        psl = PersistentSkipList.from_items([(5, 5)], seed=42)
        assert psl.floor(3) is None

    def test_ceiling(self):
        psl = PersistentSkipList.from_items([(2, 2), (4, 4), (6, 6)], seed=42)
        assert psl.ceiling(3) == (4, 4)
        assert psl.ceiling(4) == (4, 4)

    def test_ceiling_above_max(self):
        psl = PersistentSkipList.from_items([(5, 5)], seed=42)
        assert psl.ceiling(7) is None


class TestPersistentIteration:
    def test_iter(self):
        psl = PersistentSkipList.from_items([(3, 3), (1, 1), (2, 2)], seed=42)
        assert list(psl) == [1, 2, 3]


class TestPersistentEquality:
    def test_equal(self):
        psl1 = PersistentSkipList.from_items([(1, 1), (2, 2)], seed=1)
        psl2 = PersistentSkipList.from_items([(1, 1), (2, 2)], seed=2)
        assert psl1 == psl2

    def test_not_equal(self):
        psl1 = PersistentSkipList.from_items([(1, 1)], seed=42)
        psl2 = PersistentSkipList.from_items([(1, 2)], seed=42)
        assert psl1 != psl2

    def test_repr(self):
        psl = PersistentSkipList.from_items([(1, "a")], seed=42)
        assert "PersistentSkipList" in repr(psl)


class TestPersistentVersioning:
    def test_branching_versions(self):
        """Create two branches from same base."""
        base = PersistentSkipList(seed=42)
        base = base.insert(1, "one")
        base = base.insert(2, "two")

        branch_a = base.insert(3, "three_a")
        branch_b = base.insert(3, "three_b")

        assert branch_a[3] == "three_a"
        assert branch_b[3] == "three_b"
        assert len(base) == 2

    def test_version_chain(self):
        """Build a chain of versions and verify all."""
        versions = [PersistentSkipList(seed=42)]
        for i in range(10):
            versions.append(versions[-1].insert(i, i))
        for i, v in enumerate(versions):
            assert len(v) == i
            for j in range(i):
                assert v[j] == j

    def test_delete_preserves_original(self):
        psl = PersistentSkipList.from_items([(1, 1), (2, 2), (3, 3)], seed=42)
        psl2 = psl.delete(2)
        assert 2 in psl
        assert 2 not in psl2
        assert len(psl) == 3
        assert len(psl2) == 2


class TestPersistentStress:
    def test_many_versions(self):
        """Create 100 versions and spot-check."""
        psl = PersistentSkipList(seed=42)
        versions = [psl]
        for i in range(100):
            versions.append(versions[-1].insert(i, i))

        assert len(versions[0]) == 0
        assert len(versions[50]) == 50
        assert len(versions[100]) == 100
        assert versions[100].keys() == list(range(100))

    def test_interleaved_insert_delete(self):
        psl = PersistentSkipList(seed=42)
        for i in range(20):
            psl = psl.insert(i, i)
        for i in range(0, 20, 2):
            psl = psl.delete(i)
        assert psl.keys() == list(range(1, 20, 2))
        assert len(psl) == 10


# ===== Additional Edge Cases =====

class TestEdgeCases:
    def test_single_element_min_max(self):
        sl = SkipList(seed=42)
        sl.insert(42, "answer")
        assert sl.min() == (42, "answer")
        assert sl.max() == (42, "answer")

    def test_duplicate_insert_keeps_size(self):
        sl = SkipList(seed=42)
        sl.insert(1, "a")
        sl.insert(1, "b")
        sl.insert(1, "c")
        assert len(sl) == 1
        assert sl[1] == "c"

    def test_floor_ceiling_single_element(self):
        sl = SkipList(seed=42)
        sl.insert(5, 5)
        assert sl.floor(5) == (5, 5)
        assert sl.ceiling(5) == (5, 5)
        assert sl.floor(10) == (5, 5)
        assert sl.ceiling(1) == (5, 5)

    def test_range_with_gaps(self):
        sl = SkipList(seed=42)
        for i in [1, 5, 10, 15, 20]:
            sl.insert(i, i)
        result = sl.range(3, 12)
        assert result == [(5, 5), (10, 10)]

    def test_select_all_positions(self):
        sl = SkipList(seed=42)
        data = [10, 20, 30, 40, 50]
        for d in data:
            sl.insert(d, d)
        for i, d in enumerate(data):
            assert sl.select(i) == (d, d)

    def test_iter_from_past_end(self):
        sl = SkipList(seed=42)
        sl.insert(1, 1)
        assert list(sl.iter_from(100)) == []

    def test_iter_from_before_start(self):
        sl = SkipList(seed=42)
        for i in [5, 10, 15]:
            sl.insert(i, i)
        result = list(sl.iter_from(0))
        assert result == [(5, 5), (10, 10), (15, 15)]

    def test_copy_independence(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        c = sl.copy()
        sl.delete(0)
        assert 0 not in sl
        assert 0 in c

    def test_update_from_dict(self):
        sl = SkipList(seed=42)
        sl.update({"a": 1, "b": 2, "c": 3}.items())
        assert sl.keys() == ["a", "b", "c"]

    def test_map_preserves_keys(self):
        sl = SkipList(seed=42)
        for i in range(3):
            sl.insert(i, str(i))
        mapped = sl.map(lambda k, v: v + "!")
        assert mapped.keys() == [0, 1, 2]
        assert mapped[0] == "0!"

    def test_filter_empty_result(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        result = sl.filter(lambda k, v: k > 100)
        assert len(result) == 0

    def test_slice_empty(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        s = sl.slice(2, 2)
        assert len(s) == 0

    def test_slice_full(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        s = sl.slice(0, 5)
        assert s.keys() == list(range(5))

    def test_merge_both_empty(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        result = merge(sl1, sl2)
        assert len(result) == 0

    def test_diff_both_empty(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        only1, only2, differ = diff(sl1, sl2)
        assert only1 == []
        assert only2 == []
        assert differ == []

    def test_intersection_with_self(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        result = intersection(sl, sl)
        assert result.keys() == list(range(5))

    def test_difference_with_self(self):
        sl = SkipList(seed=42)
        for i in range(5):
            sl.insert(i, i)
        result = difference(sl, sl)
        assert len(result) == 0

    def test_persistent_from_items_empty(self):
        psl = PersistentSkipList.from_items([], seed=42)
        assert len(psl) == 0

    def test_persistent_insert_sorted_order(self):
        psl = PersistentSkipList(seed=42)
        for i in [50, 30, 10, 40, 20]:
            psl = psl.insert(i, str(i))
        assert psl.keys() == [10, 20, 30, 40, 50]

    def test_persistent_floor_ceiling_edge(self):
        psl = PersistentSkipList.from_items([(10, 10), (20, 20), (30, 30)], seed=42)
        assert psl.floor(0) is None
        assert psl.floor(10) == (10, 10)
        assert psl.ceiling(30) == (30, 30)
        assert psl.ceiling(31) is None

    def test_rank_empty(self):
        sl = SkipList(seed=42)
        assert sl.rank(5) == 0

    def test_nearest_tie_goes_to_smaller(self):
        sl = SkipList(seed=42)
        sl.insert(10, 10)
        sl.insert(20, 20)
        # Distance to both is 5, should return smaller
        assert sl.nearest(15) == (10, 10)

    def test_pop_min_until_empty(self):
        sl = SkipList(seed=42)
        for i in [3, 1, 4, 1, 5]:
            sl.insert(i, i)
        # Note: duplicate 1 was overwritten
        results = []
        while sl:
            results.append(sl.pop_min())
        assert [r[0] for r in results] == [1, 3, 4, 5]

    def test_pop_max_until_empty(self):
        sl = SkipList(seed=42)
        for i in [3, 1, 4, 5]:
            sl.insert(i, i)
        results = []
        while sl:
            results.append(sl.pop_max())
        assert [r[0] for r in results] == [5, 4, 3, 1]

    def test_clear_and_reuse(self):
        sl = SkipList(seed=42)
        for i in range(10):
            sl.insert(i, i)
        sl.clear()
        assert len(sl) == 0
        sl.insert(99, 99)
        assert sl[99] == 99
        assert len(sl) == 1

    def test_range_exclusive_single_match(self):
        sl = SkipList(seed=42)
        sl.insert(5, 5)
        assert sl.range(5, 5, lo_inclusive=False) == []
        assert sl.range(5, 5, hi_inclusive=False) == []

    def test_large_merge(self):
        sl1 = SkipList(seed=1)
        sl2 = SkipList(seed=2)
        for i in range(0, 100, 2):
            sl1.insert(i, i)
        for i in range(1, 100, 2):
            sl2.insert(i, i)
        result = merge(sl1, sl2)
        assert result.keys() == list(range(100))

    def test_persistent_node_with_forward(self):
        node = PersistentSkipNode(5, "five", [None, None], [1, 1])
        node2 = node.with_forward(0, PersistentSkipNode(10, "ten", [None], [1]))
        assert node.forward[0] is None  # original unchanged
        assert node2.forward[0].key == 10

    def test_persistent_node_with_value(self):
        node = PersistentSkipNode(5, "five", [None], [1])
        node2 = node.with_value("FIVE")
        assert node.value == "five"
        assert node2.value == "FIVE"
        assert node2.key == 5
