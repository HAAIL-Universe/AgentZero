"""Tests for C116: B+ Tree."""

import pytest
import random
from bplus_tree import (
    BPlusTree, BPlusTreeMap, BPlusTreeSet,
    BulkLoader, merge_trees, diff_trees,
    LeafNode, InternalNode,
)


# ===== BPlusTree core =====

class TestBPlusTreeBasic:
    """Basic operations: insert, get, contains, len."""

    def test_empty_tree(self):
        t = BPlusTree(order=4)
        assert len(t) == 0
        assert not t
        assert t.get(1) is None
        assert 1 not in t

    def test_single_insert(self):
        t = BPlusTree(order=4)
        t.insert(5, "five")
        assert len(t) == 1
        assert t
        assert t.get(5) == "five"
        assert 5 in t
        assert 3 not in t

    def test_multiple_inserts(self):
        t = BPlusTree(order=4)
        for i in range(10):
            t.insert(i, i * 10)
        assert len(t) == 10
        for i in range(10):
            assert t[i] == i * 10

    def test_update_existing_key(self):
        t = BPlusTree(order=4)
        t.insert(1, "a")
        t.insert(1, "b")
        assert len(t) == 1
        assert t[1] == "b"

    def test_getitem_missing_raises(self):
        t = BPlusTree(order=4)
        with pytest.raises(KeyError):
            _ = t[99]

    def test_setitem(self):
        t = BPlusTree(order=4)
        t[10] = "ten"
        assert t[10] == "ten"

    def test_get_default(self):
        t = BPlusTree(order=4)
        assert t.get(1, "default") == "default"

    def test_order_min(self):
        with pytest.raises(ValueError):
            BPlusTree(order=2)

    def test_repr(self):
        t = BPlusTree(order=4)
        assert "BPlusTree" in repr(t)
        assert "order=4" in repr(t)


class TestBPlusTreeInsertSplit:
    """Tests that exercise node splitting."""

    def test_leaf_split(self):
        t = BPlusTree(order=4)
        # Order 4 => max 3 keys per leaf, splits at 4
        for i in range(4):
            t.insert(i, i)
        assert len(t) == 4
        assert t.height() >= 2
        for i in range(4):
            assert t[i] == i

    def test_multiple_splits(self):
        t = BPlusTree(order=4)
        for i in range(20):
            t.insert(i, i)
        assert len(t) == 20
        for i in range(20):
            assert t[i] == i

    def test_reverse_order_insert(self):
        t = BPlusTree(order=4)
        for i in range(19, -1, -1):
            t.insert(i, i * 2)
        assert len(t) == 20
        for i in range(20):
            assert t[i] == i * 2

    def test_random_order_insert(self):
        t = BPlusTree(order=4)
        keys = list(range(50))
        random.seed(42)
        random.shuffle(keys)
        for k in keys:
            t.insert(k, k)
        assert len(t) == 50
        for k in keys:
            assert t[k] == k

    def test_internal_split(self):
        t = BPlusTree(order=3)
        # Order 3 => max 2 keys per node, splits at 3
        for i in range(15):
            t.insert(i, i)
        assert len(t) == 15
        assert t.height() >= 3
        for i in range(15):
            assert t[i] == i

    def test_large_order(self):
        t = BPlusTree(order=64)
        for i in range(200):
            t.insert(i, str(i))
        assert len(t) == 200
        for i in range(200):
            assert t[i] == str(i)


class TestBPlusTreeDelete:
    """Deletion with borrowing and merging."""

    def test_delete_from_root_leaf(self):
        t = BPlusTree(order=4)
        t.insert(1, "a")
        t.insert(2, "b")
        assert t.delete(1) is True
        assert len(t) == 1
        assert 1 not in t
        assert t[2] == "b"

    def test_delete_nonexistent(self):
        t = BPlusTree(order=4)
        t.insert(1, "a")
        assert t.delete(99) is False
        assert len(t) == 1

    def test_delete_all(self):
        t = BPlusTree(order=4)
        for i in range(10):
            t.insert(i, i)
        for i in range(10):
            assert t.delete(i) is True
        assert len(t) == 0
        assert not t

    def test_delete_triggers_borrow_right(self):
        t = BPlusTree(order=4)
        for i in range(8):
            t.insert(i, i)
        # Delete from leftmost to trigger borrow from right
        t.delete(0)
        assert 0 not in t
        for i in range(1, 8):
            assert t[i] == i

    def test_delete_triggers_borrow_left(self):
        t = BPlusTree(order=4)
        for i in range(8):
            t.insert(i, i)
        # Delete from rightmost to trigger borrow from left
        t.delete(7)
        assert 7 not in t
        for i in range(7):
            assert t[i] == i

    def test_delete_triggers_merge(self):
        t = BPlusTree(order=4)
        for i in range(6):
            t.insert(i, i)
        # Delete enough to trigger merge
        t.delete(0)
        t.delete(1)
        t.delete(2)
        for i in range(3, 6):
            assert t[i] == i

    def test_delitem(self):
        t = BPlusTree(order=4)
        t[1] = "a"
        del t[1]
        assert 1 not in t

    def test_delitem_missing_raises(self):
        t = BPlusTree(order=4)
        with pytest.raises(KeyError):
            del t[99]

    def test_pop_existing(self):
        t = BPlusTree(order=4)
        t.insert(1, "hello")
        assert t.pop(1) == "hello"
        assert 1 not in t

    def test_pop_missing_default(self):
        t = BPlusTree(order=4)
        assert t.pop(99, "nope") == "nope"

    def test_pop_missing_raises(self):
        t = BPlusTree(order=4)
        with pytest.raises(KeyError):
            t.pop(99)

    def test_delete_random_order(self):
        t = BPlusTree(order=4)
        keys = list(range(50))
        for k in keys:
            t.insert(k, k)
        random.seed(123)
        random.shuffle(keys)
        for k in keys:
            t.delete(k)
        assert len(t) == 0

    def test_delete_and_reinsert(self):
        t = BPlusTree(order=4)
        for i in range(20):
            t.insert(i, i)
        for i in range(10):
            t.delete(i)
        for i in range(10):
            t.insert(i, i * 100)
        assert len(t) == 20
        for i in range(10):
            assert t[i] == i * 100
        for i in range(10, 20):
            assert t[i] == i

    def test_heavy_delete_maintains_invariants(self):
        t = BPlusTree(order=4)
        for i in range(100):
            t.insert(i, i)
        random.seed(456)
        delete_order = list(range(100))
        random.shuffle(delete_order)
        for k in delete_order[:80]:
            t.delete(k)
        assert len(t) == 20
        t._verify()


class TestBPlusTreeQueries:
    """Min, max, floor, ceiling, rank, select."""

    def setup_method(self):
        self.t = BPlusTree(order=4)
        for i in [10, 20, 30, 40, 50]:
            self.t.insert(i, i * 10)

    def test_min_key(self):
        assert self.t.min_key() == 10

    def test_max_key(self):
        assert self.t.max_key() == 50

    def test_min_item(self):
        assert self.t.min_item() == (10, 100)

    def test_max_item(self):
        assert self.t.max_item() == (50, 500)

    def test_min_max_empty(self):
        t = BPlusTree(order=4)
        assert t.min_key() is None
        assert t.max_key() is None
        assert t.min_item() is None
        assert t.max_item() is None

    def test_floor_exact(self):
        assert self.t.floor(30) == 30

    def test_floor_between(self):
        assert self.t.floor(25) == 20

    def test_floor_below_min(self):
        assert self.t.floor(5) is None

    def test_ceiling_exact(self):
        assert self.t.ceiling(30) == 30

    def test_ceiling_between(self):
        assert self.t.ceiling(25) == 30

    def test_ceiling_above_max(self):
        assert self.t.ceiling(55) is None

    def test_floor_item(self):
        assert self.t.floor_item(25) == (20, 200)

    def test_ceiling_item(self):
        assert self.t.ceiling_item(25) == (30, 300)

    def test_rank(self):
        assert self.t.rank(10) == 0
        assert self.t.rank(30) == 2
        assert self.t.rank(50) == 4
        assert self.t.rank(55) == 5
        assert self.t.rank(5) == 0

    def test_select(self):
        assert self.t.select(0) == 10
        assert self.t.select(2) == 30
        assert self.t.select(4) == 50

    def test_select_negative(self):
        assert self.t.select(-1) == 50
        assert self.t.select(-5) == 10

    def test_select_out_of_range(self):
        with pytest.raises(IndexError):
            self.t.select(5)
        with pytest.raises(IndexError):
            self.t.select(-6)


class TestBPlusTreeRangeQuery:
    """Range queries via leaf chain."""

    def setup_method(self):
        self.t = BPlusTree(order=4)
        for i in range(1, 11):
            self.t.insert(i * 10, i * 100)

    def test_full_range(self):
        items = self.t.range_query()
        assert len(items) == 10
        assert items[0] == (10, 100)
        assert items[-1] == (100, 1000)

    def test_bounded_range(self):
        items = self.t.range_query(30, 70)
        assert items == [(30, 300), (40, 400), (50, 500), (60, 600), (70, 700)]

    def test_exclusive_bounds(self):
        items = self.t.range_query(30, 70, include_low=False, include_high=False)
        assert items == [(40, 400), (50, 500), (60, 600)]

    def test_range_keys(self):
        keys = self.t.range_keys(20, 50)
        assert keys == [20, 30, 40, 50]

    def test_range_values(self):
        vals = self.t.range_values(20, 50)
        assert vals == [200, 300, 400, 500]

    def test_count_range(self):
        assert self.t.count_range(20, 50) == 4
        assert self.t.count_range(25, 55) == 3
        assert self.t.count_range() == 10

    def test_range_empty_result(self):
        items = self.t.range_query(15, 19)
        assert items == []

    def test_range_single_item(self):
        items = self.t.range_query(30, 30)
        assert items == [(30, 300)]

    def test_range_unbounded_low(self):
        items = self.t.range_query(high=30)
        assert items == [(10, 100), (20, 200), (30, 300)]

    def test_range_unbounded_high(self):
        items = self.t.range_query(low=80)
        assert items == [(80, 800), (90, 900), (100, 1000)]


class TestBPlusTreeIteration:
    """Iteration via leaf chain."""

    def setup_method(self):
        self.t = BPlusTree(order=4)
        for i in range(10):
            self.t.insert(i, i * 10)

    def test_iter_keys(self):
        assert list(self.t) == list(range(10))

    def test_iter_items(self):
        assert list(self.t.items()) == [(i, i * 10) for i in range(10)]

    def test_iter_values(self):
        assert list(self.t.values()) == [i * 10 for i in range(10)]

    def test_reversed_keys(self):
        assert list(self.t.reversed_keys()) == list(range(9, -1, -1))

    def test_reversed_items(self):
        assert list(self.t.reversed_items()) == [(i, i * 10) for i in range(9, -1, -1)]

    def test_iter_empty(self):
        t = BPlusTree(order=4)
        assert list(t) == []
        assert list(t.items()) == []
        assert list(t.values()) == []


class TestBPlusTreeMisc:
    """Height, leaf_count, clear, update, verify."""

    def test_height_empty(self):
        t = BPlusTree(order=4)
        assert t.height() == 0

    def test_height_single(self):
        t = BPlusTree(order=4)
        t.insert(1, 1)
        assert t.height() == 1

    def test_height_grows(self):
        t = BPlusTree(order=4)
        for i in range(20):
            t.insert(i, i)
        assert t.height() >= 3

    def test_leaf_count(self):
        t = BPlusTree(order=4)
        for i in range(10):
            t.insert(i, i)
        assert t.leaf_count() >= 4

    def test_clear(self):
        t = BPlusTree(order=4)
        for i in range(10):
            t.insert(i, i)
        t.clear()
        assert len(t) == 0
        assert t.height() == 0

    def test_update(self):
        t = BPlusTree(order=4)
        t.update([(1, "a"), (2, "b"), (3, "c")])
        assert len(t) == 3
        assert t[2] == "b"

    def test_verify_valid(self):
        t = BPlusTree(order=4)
        for i in range(30):
            t.insert(i, i)
        assert t._verify() is True

    def test_verify_after_deletes(self):
        t = BPlusTree(order=4)
        for i in range(30):
            t.insert(i, i)
        for i in range(0, 30, 3):
            t.delete(i)
        assert t._verify() is True


class TestBPlusTreeStress:
    """Stress tests with larger data sets."""

    def test_sequential_1000(self):
        t = BPlusTree(order=8)
        for i in range(1000):
            t.insert(i, i)
        assert len(t) == 1000
        assert t[500] == 500
        t._verify()

    def test_random_1000(self):
        t = BPlusTree(order=8)
        random.seed(789)
        keys = list(range(1000))
        random.shuffle(keys)
        for k in keys:
            t.insert(k, k * 2)
        assert len(t) == 1000
        for k in keys:
            assert t[k] == k * 2
        t._verify()

    def test_insert_delete_cycle(self):
        t = BPlusTree(order=5)
        for i in range(200):
            t.insert(i, i)
        random.seed(321)
        for _ in range(500):
            if random.random() < 0.5 and len(t) > 0:
                k = random.choice(list(t))
                t.delete(k)
            else:
                k = random.randint(0, 999)
                t.insert(k, k)
        t._verify()

    def test_small_order_stress(self):
        """Order 3 exercises splitting/merging more aggressively."""
        t = BPlusTree(order=3)
        for i in range(100):
            t.insert(i, i)
        assert len(t) == 100
        t._verify()
        for i in range(0, 100, 2):
            t.delete(i)
        assert len(t) == 50
        t._verify()

    def test_string_keys(self):
        t = BPlusTree(order=4)
        words = ["apple", "banana", "cherry", "date", "elderberry",
                 "fig", "grape", "honeydew", "kiwi", "lemon"]
        for w in words:
            t.insert(w, len(w))
        assert len(t) == 10
        assert list(t) == sorted(words)
        assert t["cherry"] == 6

    def test_leaf_chain_integrity(self):
        """Verify doubly-linked leaf chain after many operations."""
        t = BPlusTree(order=4)
        for i in range(50):
            t.insert(i, i)
        for i in range(0, 50, 3):
            t.delete(i)
        # Walk forward
        forward = []
        leaf = t._head
        while leaf:
            forward.extend(leaf.keys)
            leaf = leaf.next_leaf
        # Walk backward
        backward = []
        leaf = t._tail
        while leaf:
            backward.extend(reversed(leaf.keys))
            leaf = leaf.prev_leaf
        backward.reverse()
        assert forward == backward
        assert forward == sorted(forward)


# ===== BPlusTreeMap =====

class TestBPlusTreeMap:
    """Dict-like ordered map interface."""

    def test_create_empty(self):
        m = BPlusTreeMap(order=4)
        assert len(m) == 0
        assert not m

    def test_create_from_items(self):
        m = BPlusTreeMap(order=4, items=[(1, "a"), (2, "b")])
        assert len(m) == 2
        assert m[1] == "a"

    def test_create_from_dict(self):
        m = BPlusTreeMap(order=4, items={"x": 1, "y": 2})
        assert m["x"] == 1

    def test_dict_operations(self):
        m = BPlusTreeMap(order=4)
        m[1] = "one"
        m[2] = "two"
        assert m[1] == "one"
        assert m.get(3, "nope") == "nope"
        del m[1]
        assert 1 not in m

    def test_ordered_operations(self):
        m = BPlusTreeMap(order=4, items=[(i, i) for i in [30, 10, 50, 20, 40]])
        assert m.min_key() == 10
        assert m.max_key() == 50
        assert m.floor(25) == 20
        assert m.ceiling(25) == 30

    def test_range_query(self):
        m = BPlusTreeMap(order=4, items=[(i, i) for i in range(10)])
        assert m.range_keys(3, 7) == [3, 4, 5, 6, 7]

    def test_iteration(self):
        m = BPlusTreeMap(order=4, items=[(3, "c"), (1, "a"), (2, "b")])
        assert list(m) == [1, 2, 3]
        assert list(m.items()) == [(1, "a"), (2, "b"), (3, "c")]

    def test_update(self):
        m = BPlusTreeMap(order=4)
        m.update({"a": 1, "b": 2})
        assert m["a"] == 1

    def test_pop(self):
        m = BPlusTreeMap(order=4, items=[(1, "a")])
        assert m.pop(1) == "a"
        assert len(m) == 0

    def test_clear(self):
        m = BPlusTreeMap(order=4, items=[(i, i) for i in range(10)])
        m.clear()
        assert len(m) == 0

    def test_rank_select(self):
        m = BPlusTreeMap(order=4, items=[(i * 10, i) for i in range(5)])
        assert m.rank(20) == 2
        assert m.select(3) == 30

    def test_reversed(self):
        m = BPlusTreeMap(order=4, items=[(i, i) for i in range(5)])
        assert list(m.reversed_keys()) == [4, 3, 2, 1, 0]

    def test_repr(self):
        m = BPlusTreeMap(order=4, items=[(1, "a")])
        assert "BPlusTreeMap" in repr(m)

    def test_height(self):
        m = BPlusTreeMap(order=4, items=[(i, i) for i in range(20)])
        assert m.height() >= 2


# ===== BPlusTreeSet =====

class TestBPlusTreeSet:
    """Ordered set interface."""

    def test_create_empty(self):
        s = BPlusTreeSet(order=4)
        assert len(s) == 0

    def test_create_from_items(self):
        s = BPlusTreeSet(order=4, items=[3, 1, 4, 1, 5])
        assert len(s) == 4  # deduplicated
        assert 3 in s
        assert 2 not in s

    def test_add_remove(self):
        s = BPlusTreeSet(order=4)
        s.add(10)
        s.add(20)
        assert 10 in s
        s.remove(10)
        assert 10 not in s

    def test_discard(self):
        s = BPlusTreeSet(order=4, items=[1, 2, 3])
        s.discard(2)
        s.discard(99)  # no error
        assert len(s) == 2

    def test_remove_missing_raises(self):
        s = BPlusTreeSet(order=4)
        with pytest.raises(KeyError):
            s.remove(99)

    def test_pop_min_max(self):
        s = BPlusTreeSet(order=4, items=[5, 3, 7, 1, 9])
        assert s.pop_min() == 1
        assert s.pop_max() == 9
        assert len(s) == 3

    def test_pop_empty_raises(self):
        s = BPlusTreeSet(order=4)
        with pytest.raises(KeyError):
            s.pop_min()
        with pytest.raises(KeyError):
            s.pop_max()

    def test_min_max(self):
        s = BPlusTreeSet(order=4, items=[30, 10, 50])
        assert s.min() == 10
        assert s.max() == 50

    def test_floor_ceiling(self):
        s = BPlusTreeSet(order=4, items=[10, 20, 30, 40, 50])
        assert s.floor(25) == 20
        assert s.ceiling(25) == 30

    def test_range(self):
        s = BPlusTreeSet(order=4, items=range(10))
        assert s.range(3, 7) == [3, 4, 5, 6, 7]

    def test_rank_select(self):
        s = BPlusTreeSet(order=4, items=[10, 20, 30])
        assert s.rank(20) == 1
        assert s.select(2) == 30

    def test_iteration(self):
        s = BPlusTreeSet(order=4, items=[3, 1, 2])
        assert list(s) == [1, 2, 3]

    def test_reversed(self):
        s = BPlusTreeSet(order=4, items=[1, 2, 3])
        assert list(s.reversed()) == [3, 2, 1]

    def test_union(self):
        s1 = BPlusTreeSet(order=4, items=[1, 2, 3])
        s2 = BPlusTreeSet(order=4, items=[3, 4, 5])
        u = s1.union(s2)
        assert list(u) == [1, 2, 3, 4, 5]

    def test_intersection(self):
        s1 = BPlusTreeSet(order=4, items=[1, 2, 3, 4])
        s2 = BPlusTreeSet(order=4, items=[3, 4, 5, 6])
        i = s1.intersection(s2)
        assert list(i) == [3, 4]

    def test_difference(self):
        s1 = BPlusTreeSet(order=4, items=[1, 2, 3, 4])
        s2 = BPlusTreeSet(order=4, items=[3, 4, 5, 6])
        d = s1.difference(s2)
        assert list(d) == [1, 2]

    def test_repr(self):
        s = BPlusTreeSet(order=4, items=[1, 2])
        assert "BPlusTreeSet" in repr(s)

    def test_count_range(self):
        s = BPlusTreeSet(order=4, items=range(20))
        assert s.count_range(5, 15) == 11


# ===== BulkLoader =====

class TestBulkLoader:
    """Efficient bottom-up construction."""

    def test_empty(self):
        t = BulkLoader.load([], order=4)
        assert len(t) == 0

    def test_single_item(self):
        t = BulkLoader.load([(1, "a")], order=4)
        assert len(t) == 1
        assert t[1] == "a"

    def test_sorted_data(self):
        data = [(i, i * 10) for i in range(100)]
        t = BulkLoader.load(data, order=8, sorted_data=True)
        assert len(t) == 100
        for i in range(100):
            assert t[i] == i * 10
        t._verify()

    def test_unsorted_data(self):
        random.seed(555)
        data = [(i, i) for i in range(50)]
        random.shuffle(data)
        t = BulkLoader.load(data, order=8)
        assert len(t) == 50
        for i in range(50):
            assert t[i] == i
        t._verify()

    def test_deduplication(self):
        data = [(1, "a"), (2, "b"), (1, "c"), (3, "d")]
        t = BulkLoader.load(data, order=4)
        assert len(t) == 3
        assert t[1] == "c"  # last value wins

    def test_bulk_then_modify(self):
        data = [(i, i) for i in range(50)]
        t = BulkLoader.load(data, order=8, sorted_data=True)
        t.insert(100, 100)
        t.delete(25)
        assert len(t) == 50
        assert t[100] == 100
        assert 25 not in t
        t._verify()

    def test_bulk_leaf_chain(self):
        data = [(i, i) for i in range(100)]
        t = BulkLoader.load(data, order=8, sorted_data=True)
        # Verify leaf chain
        forward = list(t)
        assert forward == list(range(100))
        backward = list(t.reversed_keys())
        assert backward == list(range(99, -1, -1))

    def test_bulk_large(self):
        data = [(i, i) for i in range(5000)]
        t = BulkLoader.load(data, order=32, sorted_data=True)
        assert len(t) == 5000
        assert t[2500] == 2500
        t._verify()

    def test_bulk_small_order(self):
        data = [(i, i) for i in range(20)]
        t = BulkLoader.load(data, order=3, sorted_data=True)
        assert len(t) == 20
        for i in range(20):
            assert t[i] == i
        t._verify()


# ===== Merge / Diff =====

class TestMergeDiff:
    """Tree merge and diff operations."""

    def test_merge_disjoint(self):
        t1 = BPlusTree(order=4)
        t2 = BPlusTree(order=4)
        for i in range(5):
            t1.insert(i, i)
        for i in range(5, 10):
            t2.insert(i, i)
        merged = merge_trees(t1, t2)
        assert len(merged) == 10
        for i in range(10):
            assert merged[i] == i

    def test_merge_overlapping(self):
        t1 = BPlusTree(order=4)
        t2 = BPlusTree(order=4)
        for i in range(5):
            t1.insert(i, "t1")
        for i in range(3, 8):
            t2.insert(i, "t2")
        merged = merge_trees(t1, t2)
        assert len(merged) == 8
        assert merged[0] == "t1"
        assert merged[3] == "t2"  # tree2 wins on overlap
        assert merged[7] == "t2"

    def test_merge_empty(self):
        t1 = BPlusTree(order=4)
        t2 = BPlusTree(order=4)
        for i in range(5):
            t1.insert(i, i)
        merged = merge_trees(t1, t2)
        assert len(merged) == 5

    def test_merge_both_empty(self):
        t1 = BPlusTree(order=4)
        t2 = BPlusTree(order=4)
        merged = merge_trees(t1, t2)
        assert len(merged) == 0

    def test_diff_identical(self):
        t1 = BPlusTree(order=4)
        t2 = BPlusTree(order=4)
        for i in range(5):
            t1.insert(i, i)
            t2.insert(i, i)
        d = diff_trees(t1, t2)
        assert d['same'] == 5
        assert d['only_in_first'] == []
        assert d['only_in_second'] == []
        assert d['different'] == []

    def test_diff_disjoint(self):
        t1 = BPlusTree(order=4)
        t2 = BPlusTree(order=4)
        for i in range(3):
            t1.insert(i, i)
        for i in range(3, 6):
            t2.insert(i, i)
        d = diff_trees(t1, t2)
        assert d['same'] == 0
        assert len(d['only_in_first']) == 3
        assert len(d['only_in_second']) == 3

    def test_diff_different_values(self):
        t1 = BPlusTree(order=4)
        t2 = BPlusTree(order=4)
        t1.insert(1, "a")
        t2.insert(1, "b")
        d = diff_trees(t1, t2)
        assert d['different'] == [(1, "a", "b")]
        assert d['same'] == 0


# ===== Verification =====

class TestVerification:
    """Tree invariant verification."""

    def test_verify_after_many_ops(self):
        t = BPlusTree(order=4)
        random.seed(999)
        keys = list(range(200))
        random.shuffle(keys)
        for k in keys[:150]:
            t.insert(k, k)
        random.shuffle(keys)
        for k in keys[:100]:
            t.delete(k)
        t._verify()

    def test_verify_all_orders(self):
        for order in [3, 4, 5, 8, 16]:
            t = BPlusTree(order=order)
            for i in range(50):
                t.insert(i, i)
            t._verify()
            for i in range(0, 50, 2):
                t.delete(i)
            t._verify()

    def test_parent_pointers(self):
        t = BPlusTree(order=4)
        for i in range(30):
            t.insert(i, i)
        # Verify all parent pointers
        def check_parents(node, expected_parent):
            assert node.parent is expected_parent
            if not node.is_leaf:
                for child in node.children:
                    check_parents(child, node)
        check_parents(t._root, None)

    def test_leaf_chain_after_splits(self):
        t = BPlusTree(order=3)
        for i in range(30):
            t.insert(i, i)
        keys_forward = list(t)
        keys_backward = list(t.reversed_keys())
        keys_backward.reverse()
        assert keys_forward == keys_backward
        assert keys_forward == list(range(30))


# ===== Edge cases =====

class TestEdgeCases:
    """Corner cases and special scenarios."""

    def test_duplicate_inserts(self):
        t = BPlusTree(order=4)
        for _ in range(10):
            t.insert(1, "same")
        assert len(t) == 1

    def test_negative_keys(self):
        t = BPlusTree(order=4)
        for i in range(-10, 11):
            t.insert(i, abs(i))
        assert len(t) == 21
        assert t[-5] == 5
        assert t[0] == 0

    def test_float_keys(self):
        t = BPlusTree(order=4)
        for x in [1.5, 2.7, 0.3, 4.1, 3.9]:
            t.insert(x, x * 2)
        assert t[1.5] == 3.0
        assert list(t) == [0.3, 1.5, 2.7, 3.9, 4.1]

    def test_tuple_keys(self):
        t = BPlusTree(order=4)
        t.insert((1, "a"), "first")
        t.insert((1, "b"), "second")
        t.insert((2, "a"), "third")
        assert t[(1, "b")] == "second"

    def test_none_values(self):
        t = BPlusTree(order=4)
        t.insert(1, None)
        assert t[1] is None
        assert t.get(1) is None  # same as missing but key exists
        assert 1 in t

    def test_mixed_operations_order3(self):
        """Order 3 is the minimum, exercises all edge cases."""
        t = BPlusTree(order=3)
        ops = []
        random.seed(42)
        for _ in range(200):
            if random.random() < 0.6:
                k = random.randint(0, 50)
                t.insert(k, k)
                ops.append(('i', k))
            else:
                k = random.randint(0, 50)
                t.delete(k)
                ops.append(('d', k))
        t._verify()

    def test_alternating_insert_delete(self):
        t = BPlusTree(order=4)
        for i in range(100):
            t.insert(i, i)
            if i > 0 and i % 3 == 0:
                t.delete(i - 1)
        t._verify()
