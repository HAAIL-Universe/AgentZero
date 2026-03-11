"""Tests for C108: Red-Black Tree."""
import pytest
import random
from red_black_tree import (
    RedBlackTree, RedBlackMap, RedBlackMultiMap,
    OrderStatisticTree, IntervalMap, NIL, RED, BLACK
)


# ============================================================
# RedBlackTree (Ordered Set)
# ============================================================

class TestRedBlackTreeBasic:
    def test_empty_tree(self):
        t = RedBlackTree()
        assert len(t) == 0
        assert not t
        assert 5 not in t
        assert t.min() is None
        assert t.max() is None

    def test_single_insert(self):
        t = RedBlackTree()
        assert t.insert(10) is True
        assert len(t) == 1
        assert 10 in t
        assert t.min() == 10
        assert t.max() == 10

    def test_duplicate_insert(self):
        t = RedBlackTree()
        assert t.insert(5) is True
        assert t.insert(5) is False
        assert len(t) == 1

    def test_multiple_inserts(self):
        t = RedBlackTree()
        for x in [5, 3, 7, 1, 4, 6, 8]:
            t.insert(x)
        assert len(t) == 7
        assert list(t) == [1, 3, 4, 5, 6, 7, 8]

    def test_constructor_with_items(self):
        t = RedBlackTree([3, 1, 4, 1, 5, 9])
        assert len(t) == 5  # set semantics, no dupes
        assert list(t) == [1, 3, 4, 5, 9]

    def test_bool(self):
        t = RedBlackTree()
        assert not t
        t.insert(1)
        assert t

    def test_contains(self):
        t = RedBlackTree([10, 20, 30])
        assert 10 in t
        assert 20 in t
        assert 15 not in t

    def test_repr(self):
        t = RedBlackTree([3, 1, 2])
        assert repr(t) == "RedBlackTree([1, 2, 3])"


class TestRedBlackTreeDelete:
    def test_delete_leaf(self):
        t = RedBlackTree([5, 3, 7])
        assert t.delete(3) is True
        assert 3 not in t
        assert len(t) == 2

    def test_delete_nonexistent(self):
        t = RedBlackTree([5, 3, 7])
        assert t.delete(99) is False
        assert len(t) == 3

    def test_delete_root(self):
        t = RedBlackTree([5])
        assert t.delete(5) is True
        assert len(t) == 0

    def test_delete_with_children(self):
        t = RedBlackTree([5, 3, 7, 1, 4, 6, 8])
        t.delete(3)
        assert sorted(list(t)) == [1, 4, 5, 6, 7, 8]
        t.delete(7)
        assert sorted(list(t)) == [1, 4, 5, 6, 8]

    def test_delete_all(self):
        t = RedBlackTree([5, 3, 7, 1, 4])
        for x in [5, 3, 7, 1, 4]:
            t.delete(x)
        assert len(t) == 0

    def test_delete_maintains_invariants(self):
        t = RedBlackTree()
        keys = list(range(20))
        random.seed(42)
        random.shuffle(keys)
        for k in keys:
            t.insert(k)
        random.shuffle(keys)
        for k in keys:
            t.delete(k)
            assert t.is_valid()

    def test_pop_min(self):
        t = RedBlackTree([5, 3, 7, 1])
        assert t.pop_min() == 1
        assert t.min() == 3
        assert len(t) == 3

    def test_pop_max(self):
        t = RedBlackTree([5, 3, 7, 9])
        assert t.pop_max() == 9
        assert t.max() == 7

    def test_pop_empty_raises(self):
        t = RedBlackTree()
        with pytest.raises(ValueError):
            t.pop_min()
        with pytest.raises(ValueError):
            t.pop_max()


class TestRedBlackTreeQueries:
    def test_min_max(self):
        t = RedBlackTree([5, 3, 7, 1, 9])
        assert t.min() == 1
        assert t.max() == 9

    def test_find(self):
        t = RedBlackTree([5, 3, 7])
        assert t.find(5) == 5
        assert t.find(3) == 3
        assert t.find(99) is None

    def test_floor(self):
        t = RedBlackTree([10, 20, 30, 40])
        assert t.floor(25) == 20
        assert t.floor(20) == 20
        assert t.floor(5) is None
        assert t.floor(40) == 40

    def test_ceiling(self):
        t = RedBlackTree([10, 20, 30, 40])
        assert t.ceiling(25) == 30
        assert t.ceiling(20) == 20
        assert t.ceiling(45) is None
        assert t.ceiling(10) == 10

    def test_successor(self):
        t = RedBlackTree([10, 20, 30])
        assert t.successor(10) == 20
        assert t.successor(20) == 30
        assert t.successor(30) is None
        assert t.successor(15) == 20

    def test_predecessor(self):
        t = RedBlackTree([10, 20, 30])
        assert t.predecessor(30) == 20
        assert t.predecessor(20) == 10
        assert t.predecessor(10) is None
        assert t.predecessor(25) == 20

    def test_range(self):
        t = RedBlackTree([1, 3, 5, 7, 9, 11])
        assert t.range(3, 9) == [3, 5, 7, 9]
        assert t.range(4, 8) == [5, 7]
        assert t.range(0, 2) == [1]
        assert t.range(12, 20) == []

    def test_reversed(self):
        t = RedBlackTree([1, 2, 3, 4, 5])
        assert list(reversed(t)) == [5, 4, 3, 2, 1]

    def test_to_sorted_list(self):
        t = RedBlackTree([5, 3, 7, 1, 4])
        assert t.to_sorted_list() == [1, 3, 4, 5, 7]

    def test_height(self):
        t = RedBlackTree(range(15))
        h = t.height()
        assert h >= 4  # log2(16) = 4
        assert h <= 10  # RB tree height <= 2*log2(n+1)

    def test_clear(self):
        t = RedBlackTree([1, 2, 3])
        t.clear()
        assert len(t) == 0
        assert t.root is NIL


class TestRedBlackTreeInvariants:
    def test_root_is_black(self):
        t = RedBlackTree([5, 3, 7])
        assert t.root.color == BLACK

    def test_no_red_red(self):
        t = RedBlackTree(range(20))
        assert t.is_valid()

    def test_black_height_consistent(self):
        t = RedBlackTree(range(20))
        assert t.is_valid()
        assert t.black_height() >= 1

    def test_valid_after_sequential_inserts(self):
        t = RedBlackTree()
        for i in range(50):
            t.insert(i)
            assert t.is_valid(), f"Invalid after inserting {i}"

    def test_valid_after_random_inserts(self):
        random.seed(123)
        t = RedBlackTree()
        for _ in range(100):
            t.insert(random.randint(0, 1000))
            assert t.is_valid()

    def test_valid_after_mixed_ops(self):
        random.seed(456)
        t = RedBlackTree()
        keys = set()
        for _ in range(200):
            if random.random() < 0.7 or not keys:
                k = random.randint(0, 100)
                t.insert(k)
                keys.add(k)
            else:
                k = random.choice(list(keys))
                t.delete(k)
                keys.discard(k)
            assert t.is_valid()
        assert sorted(keys) == list(t)

    def test_valid_empty(self):
        assert RedBlackTree().is_valid()


class TestRedBlackTreeStress:
    def test_large_sequential(self):
        t = RedBlackTree()
        n = 1000
        for i in range(n):
            t.insert(i)
        assert len(t) == n
        assert list(t) == list(range(n))
        assert t.is_valid()

    def test_large_random_insert_delete(self):
        random.seed(789)
        t = RedBlackTree()
        inserted = set()
        for _ in range(500):
            k = random.randint(0, 200)
            t.insert(k)
            inserted.add(k)
        for k in list(inserted)[:250]:
            t.delete(k)
            inserted.discard(k)
        assert sorted(inserted) == list(t)
        assert t.is_valid()

    def test_reverse_sequential(self):
        t = RedBlackTree()
        for i in range(100, 0, -1):
            t.insert(i)
        assert list(t) == list(range(1, 101))
        assert t.is_valid()


# ============================================================
# RedBlackMap (Ordered Key-Value Map)
# ============================================================

class TestRedBlackMap:
    def test_empty(self):
        m = RedBlackMap()
        assert len(m) == 0
        assert not m

    def test_put_get(self):
        m = RedBlackMap()
        assert m.put("a", 1) is True
        assert m["a"] == 1
        assert m.get("a") == 1

    def test_update_value(self):
        m = RedBlackMap()
        m.put("a", 1)
        assert m.put("a", 2) is False  # update, not new
        assert m["a"] == 2
        assert len(m) == 1

    def test_getitem_missing_raises(self):
        m = RedBlackMap()
        with pytest.raises(KeyError):
            _ = m["x"]

    def test_get_default(self):
        m = RedBlackMap()
        assert m.get("x") is None
        assert m.get("x", 42) == 42

    def test_setitem_delitem(self):
        m = RedBlackMap()
        m["a"] = 1
        m["b"] = 2
        assert "a" in m
        del m["a"]
        assert "a" not in m
        with pytest.raises(KeyError):
            del m["z"]

    def test_contains(self):
        m = RedBlackMap([("a", 1), ("b", 2)])
        assert "a" in m
        assert "c" not in m

    def test_iter_sorted(self):
        m = RedBlackMap([("c", 3), ("a", 1), ("b", 2)])
        assert list(m) == ["a", "b", "c"]

    def test_keys_values_items(self):
        m = RedBlackMap([("b", 2), ("a", 1), ("c", 3)])
        assert m.keys() == ["a", "b", "c"]
        assert m.values() == [1, 2, 3]
        assert m.items() == [("a", 1), ("b", 2), ("c", 3)]

    def test_min_max(self):
        m = RedBlackMap([("b", 2), ("a", 1), ("c", 3)])
        assert m.min_key() == "a"
        assert m.max_key() == "c"
        assert m.min_item() == ("a", 1)
        assert m.max_item() == ("c", 3)

    def test_floor_ceiling(self):
        m = RedBlackMap([(10, "a"), (20, "b"), (30, "c")])
        assert m.floor(25) == 20
        assert m.ceiling(25) == 30

    def test_range(self):
        m = RedBlackMap([(1, "a"), (2, "b"), (3, "c"), (4, "d")])
        assert m.range(2, 3) == [(2, "b"), (3, "c")]

    def test_delete(self):
        m = RedBlackMap([(1, "a"), (2, "b")])
        assert m.delete(1) is True
        assert m.delete(99) is False
        assert len(m) == 1

    def test_clear(self):
        m = RedBlackMap([(1, "a"), (2, "b")])
        m.clear()
        assert len(m) == 0

    def test_from_dict(self):
        m = RedBlackMap({"x": 1, "y": 2})
        assert "x" in m
        assert m["y"] == 2

    def test_repr(self):
        m = RedBlackMap([(1, "a")])
        assert "RedBlackMap" in repr(m)

    def test_empty_min_max(self):
        m = RedBlackMap()
        assert m.min_key() is None
        assert m.max_key() is None
        assert m.min_item() is None
        assert m.max_item() is None


# ============================================================
# RedBlackMultiMap
# ============================================================

class TestRedBlackMultiMap:
    def test_empty(self):
        mm = RedBlackMultiMap()
        assert len(mm) == 0
        assert not mm

    def test_insert_duplicates(self):
        mm = RedBlackMultiMap()
        mm.insert(5, "a")
        mm.insert(5, "b")
        mm.insert(5, "c")
        assert len(mm) == 3
        assert 5 in mm
        vals = mm.get_all(5)
        assert sorted(vals) == ["a", "b", "c"]

    def test_insert_mixed(self):
        mm = RedBlackMultiMap([(3, "x"), (1, "y"), (3, "z")])
        assert len(mm) == 3
        assert mm.count(3) == 2
        assert mm.count(1) == 1

    def test_delete_one(self):
        mm = RedBlackMultiMap([(5, "a"), (5, "b"), (5, "c")])
        assert mm.delete_one(5) is True
        assert len(mm) == 2
        assert mm.count(5) == 2

    def test_delete_all(self):
        mm = RedBlackMultiMap([(5, "a"), (5, "b"), (3, "c")])
        assert mm.delete_all(5) == 2
        assert mm.count(5) == 0
        assert len(mm) == 1

    def test_delete_nonexistent(self):
        mm = RedBlackMultiMap([(1, "a")])
        assert mm.delete_one(99) is False
        assert mm.delete_all(99) == 0

    def test_items_sorted(self):
        mm = RedBlackMultiMap([(3, "c"), (1, "a"), (2, "b"), (1, "d")])
        items = mm.items()
        keys = [k for k, v in items]
        assert keys == sorted(keys)

    def test_iter(self):
        mm = RedBlackMultiMap([(2, "b"), (1, "a")])
        items = list(mm)
        assert items == [(1, "a"), (2, "b")]

    def test_contains(self):
        mm = RedBlackMultiMap([(5, "x")])
        assert 5 in mm
        assert 6 not in mm

    def test_clear(self):
        mm = RedBlackMultiMap([(1, "a"), (2, "b")])
        mm.clear()
        assert len(mm) == 0

    def test_get_all_empty(self):
        mm = RedBlackMultiMap()
        assert mm.get_all(5) == []


# ============================================================
# OrderStatisticTree
# ============================================================

class TestOrderStatisticTree:
    def test_rank_select_basic(self):
        ost = OrderStatisticTree([10, 20, 30, 40, 50])
        assert ost.rank(10) == 0
        assert ost.rank(30) == 2
        assert ost.rank(50) == 4
        assert ost.select(0) == 10
        assert ost.select(2) == 30
        assert ost.select(4) == 50

    def test_rank_missing_raises(self):
        ost = OrderStatisticTree([10, 20, 30])
        with pytest.raises(KeyError):
            ost.rank(99)

    def test_select_out_of_range(self):
        ost = OrderStatisticTree([10, 20])
        with pytest.raises(IndexError):
            ost.select(5)
        with pytest.raises(IndexError):
            ost.select(-1)

    def test_insert_updates_rank(self):
        ost = OrderStatisticTree([10, 30])
        ost.insert(20)
        assert ost.rank(20) == 1
        assert ost.rank(30) == 2

    def test_delete_updates_rank(self):
        ost = OrderStatisticTree([10, 20, 30])
        ost.delete(10)
        assert ost.rank(20) == 0
        assert ost.rank(30) == 1

    def test_count_less(self):
        ost = OrderStatisticTree([10, 20, 30, 40, 50])
        assert ost.count_less(30) == 2
        assert ost.count_less(10) == 0
        assert ost.count_less(55) == 5
        assert ost.count_less(5) == 0

    def test_count_range(self):
        ost = OrderStatisticTree([10, 20, 30, 40, 50])
        assert ost.count_range(20, 40) == 3
        assert ost.count_range(10, 50) == 5
        assert ost.count_range(25, 35) == 1

    def test_contains(self):
        ost = OrderStatisticTree([1, 2, 3])
        assert 1 in ost
        assert 4 not in ost

    def test_len(self):
        ost = OrderStatisticTree([1, 2, 3])
        assert len(ost) == 3

    def test_iter(self):
        ost = OrderStatisticTree([3, 1, 2])
        assert list(ost) == [1, 2, 3]

    def test_min_max(self):
        ost = OrderStatisticTree([5, 3, 7])
        assert ost.min() == 3
        assert ost.max() == 7

    def test_to_sorted_list(self):
        ost = OrderStatisticTree([5, 1, 3])
        assert ost.to_sorted_list() == [1, 3, 5]

    def test_stress_rank_select(self):
        random.seed(101)
        ost = OrderStatisticTree()
        keys = random.sample(range(1000), 100)
        for k in keys:
            ost.insert(k)
        sorted_keys = sorted(keys)
        for i, k in enumerate(sorted_keys):
            assert ost.rank(k) == i
            assert ost.select(i) == k


# ============================================================
# IntervalMap
# ============================================================

class TestIntervalMap:
    def test_empty(self):
        im = IntervalMap()
        assert len(im) == 0
        assert not im
        assert im.stab(5) == []

    def test_single_interval(self):
        im = IntervalMap()
        im.insert(1, 10, "A")
        assert len(im) == 1
        assert im

    def test_stab_query(self):
        im = IntervalMap()
        im.insert(1, 10, "A")
        im.insert(5, 15, "B")
        im.insert(20, 30, "C")
        results = im.stab(7)
        labels = sorted([r[2] for r in results])
        assert labels == ["A", "B"]

    def test_stab_at_boundary(self):
        im = IntervalMap()
        im.insert(1, 10, "A")
        results = im.stab(1)
        assert len(results) == 1
        results = im.stab(10)
        assert len(results) == 1
        results = im.stab(0)
        assert len(results) == 0
        results = im.stab(11)
        assert len(results) == 0

    def test_overlap_query(self):
        im = IntervalMap()
        im.insert(1, 5, "A")
        im.insert(3, 8, "B")
        im.insert(10, 15, "C")
        results = im.overlap(4, 6)
        labels = sorted([r[2] for r in results])
        assert labels == ["A", "B"]

    def test_overlap_no_match(self):
        im = IntervalMap()
        im.insert(1, 5, "A")
        im.insert(10, 15, "B")
        assert im.overlap(6, 9) == []

    def test_delete_interval(self):
        im = IntervalMap()
        im.insert(1, 10, "A")
        im.insert(5, 15, "B")
        assert im.delete(1, 10) is True
        assert len(im) == 1
        results = im.stab(7)
        assert len(results) == 1

    def test_delete_nonexistent(self):
        im = IntervalMap()
        im.insert(1, 10, "A")
        assert im.delete(2, 10) is False

    def test_all_intervals(self):
        im = IntervalMap()
        im.insert(5, 10, "B")
        im.insert(1, 3, "A")
        im.insert(15, 20, "C")
        intervals = im.all_intervals()
        assert len(intervals) == 3
        # Should be sorted by lo
        los = [i[0] for i in intervals]
        assert los == sorted(los)

    def test_clear(self):
        im = IntervalMap()
        im.insert(1, 10, "A")
        im.clear()
        assert len(im) == 0

    def test_overlapping_intervals(self):
        im = IntervalMap()
        im.insert(1, 10, "A")
        im.insert(2, 5, "B")
        im.insert(3, 7, "C")
        results = im.stab(4)
        assert len(results) == 3

    def test_no_value(self):
        im = IntervalMap()
        im.insert(1, 10)
        results = im.stab(5)
        assert len(results) == 1
        assert results[0][2] is None

    def test_many_intervals(self):
        im = IntervalMap()
        for i in range(50):
            im.insert(i * 2, i * 2 + 5, f"I{i}")
        assert len(im) == 50
        # Point 10 should be covered by several intervals
        results = im.stab(10)
        assert len(results) >= 3

    def test_stab_point_outside(self):
        im = IntervalMap()
        im.insert(10, 20, "A")
        assert im.stab(5) == []
        assert im.stab(25) == []


# ============================================================
# Integration & Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_element_tree(self):
        t = RedBlackTree([42])
        assert t.is_valid()
        assert t.root.color == BLACK
        assert len(t) == 1
        t.delete(42)
        assert len(t) == 0

    def test_two_elements(self):
        t = RedBlackTree([1, 2])
        assert t.is_valid()
        assert list(t) == [1, 2]
        t.delete(1)
        assert list(t) == [2]
        assert t.is_valid()

    def test_negative_keys(self):
        t = RedBlackTree([-5, -3, -1, 0, 1, 3, 5])
        assert list(t) == [-5, -3, -1, 0, 1, 3, 5]
        assert t.min() == -5
        assert t.max() == 5

    def test_string_keys(self):
        t = RedBlackTree(["banana", "apple", "cherry"])
        assert list(t) == ["apple", "banana", "cherry"]
        assert t.min() == "apple"
        assert t.max() == "cherry"

    def test_float_keys(self):
        t = RedBlackTree([1.5, 2.5, 0.5])
        assert list(t) == [0.5, 1.5, 2.5]

    def test_map_with_none_value(self):
        m = RedBlackMap()
        m.put(1, None)
        assert 1 in m
        assert m[1] is None

    def test_ost_empty(self):
        ost = OrderStatisticTree()
        assert len(ost) == 0
        assert ost.min() is None

    def test_multimap_many_dupes(self):
        mm = RedBlackMultiMap()
        for i in range(50):
            mm.insert(1, f"v{i}")
        assert mm.count(1) == 50
        assert len(mm) == 50

    def test_interval_touching(self):
        """Intervals that share endpoints overlap."""
        im = IntervalMap()
        im.insert(1, 5, "A")
        im.insert(5, 10, "B")
        results = im.stab(5)
        assert len(results) == 2

    def test_tree_insert_delete_reinsert(self):
        t = RedBlackTree()
        t.insert(10)
        t.delete(10)
        assert 10 not in t
        t.insert(10)
        assert 10 in t
        assert t.is_valid()

    def test_map_update_preserves_order(self):
        m = RedBlackMap()
        m.put(3, "c")
        m.put(1, "a")
        m.put(2, "b")
        m.put(2, "B")  # update
        assert m.items() == [(1, "a"), (2, "B"), (3, "c")]

    def test_black_height_grows(self):
        t = RedBlackTree()
        prev_bh = 0
        for i in range(100):
            t.insert(i)
            bh = t.black_height()
            assert bh >= prev_bh  # black height should be non-decreasing
            prev_bh = bh


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
