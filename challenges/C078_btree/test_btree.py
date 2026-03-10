"""Tests for C078: Persistent B-Tree"""
import pytest
from btree import BTree, BTreeNode, _search_index


# ============================================================
# Construction & basics
# ============================================================

class TestConstruction:
    def test_empty_tree(self):
        t = BTree()
        assert len(t) == 0
        assert not t
        assert t.height() == 0

    def test_default_order(self):
        t = BTree()
        assert t.order == 32

    def test_custom_order(self):
        t = BTree(order=4)
        assert t.order == 4

    def test_order_too_small(self):
        with pytest.raises(ValueError):
            BTree(order=2)

    def test_repr_empty(self):
        assert "size=0" in repr(BTree())

    def test_repr_nonempty(self):
        t = BTree().insert(1, "a")
        assert "size=1" in repr(t)

    def test_bool_empty(self):
        assert not BTree()

    def test_bool_nonempty(self):
        assert BTree().insert(1, "a")


# ============================================================
# Single insert/get
# ============================================================

class TestSingleOps:
    def test_insert_and_get(self):
        t = BTree().insert(5, "five")
        assert t.get(5) == "five"

    def test_get_missing_default(self):
        t = BTree().insert(5, "five")
        assert t.get(10) is None

    def test_get_missing_custom_default(self):
        t = BTree().insert(5, "five")
        assert t.get(10, "nope") == "nope"

    def test_contains(self):
        t = BTree().insert(5, "five")
        assert 5 in t
        assert 10 not in t

    def test_getitem(self):
        t = BTree().insert(5, "five")
        assert t[5] == "five"

    def test_getitem_missing(self):
        t = BTree()
        with pytest.raises(KeyError):
            t[5]

    def test_height_single(self):
        t = BTree().insert(1, "a")
        assert t.height() == 1


# ============================================================
# Multiple inserts
# ============================================================

class TestMultiInsert:
    def test_ascending(self):
        t = BTree(order=4)
        for i in range(20):
            t = t.insert(i, str(i))
        assert len(t) == 20
        for i in range(20):
            assert t[i] == str(i)

    def test_descending(self):
        t = BTree(order=4)
        for i in range(19, -1, -1):
            t = t.insert(i, str(i))
        assert len(t) == 20
        for i in range(20):
            assert t[i] == str(i)

    def test_random_order(self):
        import random
        rng = random.Random(42)
        keys = list(range(100))
        rng.shuffle(keys)
        t = BTree(order=5)
        for k in keys:
            t = t.insert(k, k * 10)
        assert len(t) == 100
        for k in keys:
            assert t[k] == k * 10

    def test_update_existing(self):
        t = BTree().insert(5, "old")
        t2 = t.insert(5, "new")
        assert t[5] == "old"  # Persistence
        assert t2[5] == "new"
        assert len(t2) == 1  # No duplicate

    def test_many_updates(self):
        t = BTree(order=4)
        for i in range(10):
            t = t.insert(i, i)
        for i in range(10):
            t = t.insert(i, i * 100)
        assert len(t) == 10
        for i in range(10):
            assert t[i] == i * 100

    def test_height_grows(self):
        t = BTree(order=4)
        for i in range(50):
            t = t.insert(i, i)
        assert t.height() >= 2


# ============================================================
# Persistence
# ============================================================

class TestPersistence:
    def test_insert_preserves_original(self):
        t1 = BTree().insert(1, "a")
        t2 = t1.insert(2, "b")
        assert len(t1) == 1
        assert len(t2) == 2
        assert 2 not in t1

    def test_delete_preserves_original(self):
        t1 = BTree().insert(1, "a").insert(2, "b")
        t2 = t1.delete(1)
        assert len(t1) == 2
        assert len(t2) == 1
        assert 1 in t1
        assert 1 not in t2

    def test_chain_operations(self):
        t = BTree()
        versions = [t]
        for i in range(10):
            t = t.insert(i, i)
            versions.append(t)
        for i, v in enumerate(versions):
            assert len(v) == i

    def test_structural_sharing_different_objects(self):
        t1 = BTree(order=4)
        for i in range(20):
            t1 = t1.insert(i, i)
        t2 = t1.insert(100, 100)
        # Both should work independently
        assert 100 in t2
        assert 100 not in t1


# ============================================================
# Delete
# ============================================================

class TestDelete:
    def test_delete_from_leaf(self):
        t = BTree().insert(1, "a").insert(2, "b").insert(3, "c")
        t2 = t.delete(2)
        assert len(t2) == 2
        assert 2 not in t2
        assert 1 in t2
        assert 3 in t2

    def test_delete_missing_raises(self):
        t = BTree().insert(1, "a")
        with pytest.raises(KeyError):
            t.delete(99)

    def test_delete_from_empty_raises(self):
        with pytest.raises(KeyError):
            BTree().delete(1)

    def test_delete_last_item(self):
        t = BTree().insert(1, "a").delete(1)
        assert len(t) == 0
        assert not t

    def test_delete_all_ascending(self):
        t = BTree(order=4)
        for i in range(20):
            t = t.insert(i, i)
        for i in range(20):
            t = t.delete(i)
        assert len(t) == 0

    def test_delete_all_descending(self):
        t = BTree(order=4)
        for i in range(20):
            t = t.insert(i, i)
        for i in range(19, -1, -1):
            t = t.delete(i)
        assert len(t) == 0

    def test_delete_random_order(self):
        import random
        rng = random.Random(123)
        keys = list(range(50))
        rng.shuffle(keys)
        t = BTree(order=4)
        for k in keys:
            t = t.insert(k, k)
        rng.shuffle(keys)
        for k in keys:
            t = t.delete(k)
            assert k not in t
        assert len(t) == 0

    def test_delete_triggers_borrow_left(self):
        # Small order to force rebalancing
        t = BTree(order=4)
        for i in range(10):
            t = t.insert(i, i)
        t = t.delete(0)
        assert 0 not in t
        assert all(i in t for i in range(1, 10))

    def test_delete_triggers_borrow_right(self):
        t = BTree(order=4)
        for i in range(10):
            t = t.insert(i, i)
        t = t.delete(9)
        assert 9 not in t

    def test_delete_triggers_merge(self):
        t = BTree(order=4)
        for i in range(8):
            t = t.insert(i, i)
        # Delete multiple to force merges
        for i in [0, 1, 2, 3]:
            t = t.delete(i)
        assert len(t) == 4

    def test_delete_internal_node_key(self):
        # Force a split then delete the internal key
        t = BTree(order=4)
        for i in range(12):
            t = t.insert(i, i)
        # Internal nodes should exist at height > 1
        for i in range(12):
            if i in t:
                t2 = t.delete(i)
                assert i not in t2
                assert len(t2) == len(t) - 1

    def test_discard_existing(self):
        t = BTree().insert(1, "a").insert(2, "b")
        t2 = t.discard(1)
        assert len(t2) == 1

    def test_discard_missing(self):
        t = BTree().insert(1, "a")
        t2 = t.discard(99)
        assert t2 is t  # Same object


# ============================================================
# Min / Max
# ============================================================

class TestMinMax:
    def test_min(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(5, "e")
        assert t.min() == (1, "a")

    def test_max(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(5, "e")
        assert t.max() == (5, "e")

    def test_min_empty_raises(self):
        with pytest.raises(ValueError):
            BTree().min()

    def test_max_empty_raises(self):
        with pytest.raises(ValueError):
            BTree().max()

    def test_min_single(self):
        t = BTree().insert(42, "x")
        assert t.min() == (42, "x")

    def test_max_single(self):
        t = BTree().insert(42, "x")
        assert t.max() == (42, "x")


# ============================================================
# Iteration
# ============================================================

class TestIteration:
    def test_iter_empty(self):
        assert list(BTree()) == []

    def test_iter_sorted(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(5, "e").insert(2, "b")
        assert list(t) == [1, 2, 3, 5]

    def test_items(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(5, "e")
        assert list(t.items()) == [(1, "a"), (3, "c"), (5, "e")]

    def test_keys(self):
        t = BTree().insert(2, "b").insert(1, "a")
        assert list(t.keys()) == [1, 2]

    def test_values(self):
        t = BTree().insert(2, "b").insert(1, "a")
        assert list(t.values()) == ["a", "b"]

    def test_iter_large(self):
        t = BTree(order=4)
        for i in range(100):
            t = t.insert(i, i)
        assert list(t) == list(range(100))

    def test_reversed(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(5, "e")
        assert list(reversed(t)) == [5, 3, 1]

    def test_reversed_large(self):
        t = BTree(order=4)
        for i in range(50):
            t = t.insert(i, i)
        assert list(reversed(t)) == list(range(49, -1, -1))

    def test_values_order(self):
        t = BTree(order=4)
        for i in [5, 3, 8, 1, 9, 2]:
            t = t.insert(i, i * 10)
        assert list(t.values()) == [10, 20, 30, 50, 80, 90]


# ============================================================
# Range queries
# ============================================================

class TestRange:
    def test_range_all(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        assert list(t.range()) == [(i, i) for i in range(10)]

    def test_range_lower_bound(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        assert list(t.range(lo=5)) == [(i, i) for i in range(5, 10)]

    def test_range_upper_bound(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        assert list(t.range(hi=5)) == [(i, i) for i in range(5)]

    def test_range_both_bounds(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        assert list(t.range(lo=3, hi=7)) == [(i, i) for i in range(3, 7)]

    def test_range_inclusive_hi(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        result = list(t.range(lo=3, hi=7, inclusive_hi=True))
        assert result == [(i, i) for i in range(3, 8)]

    def test_range_exclusive_lo(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        result = list(t.range(lo=3, inclusive_lo=False))
        assert result == [(i, i) for i in range(4, 10)]

    def test_range_empty_result(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        assert list(t.range(lo=20)) == []

    def test_range_on_empty_tree(self):
        assert list(BTree().range(lo=0, hi=10)) == []

    def test_range_with_small_order(self):
        t = BTree(order=4)
        for i in range(50):
            t = t.insert(i, i)
        result = list(t.range(lo=10, hi=20))
        assert result == [(i, i) for i in range(10, 20)]

    def test_count_range(self):
        t = BTree()
        for i in range(100):
            t = t.insert(i, i)
        assert t.count_range(lo=10, hi=20) == 10

    def test_count_range_all(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        assert t.count_range() == 10


# ============================================================
# Floor / Ceiling
# ============================================================

class TestFloorCeiling:
    def test_floor_exact(self):
        t = BTree()
        for i in range(0, 20, 2):
            t = t.insert(i, i)
        assert t.floor(10) == (10, 10)

    def test_floor_between(self):
        t = BTree()
        for i in range(0, 20, 2):
            t = t.insert(i, i)
        assert t.floor(11) == (10, 10)

    def test_floor_below_min(self):
        t = BTree().insert(5, 5)
        assert t.floor(3) is None

    def test_ceiling_exact(self):
        t = BTree()
        for i in range(0, 20, 2):
            t = t.insert(i, i)
        assert t.ceiling(10) == (10, 10)

    def test_ceiling_between(self):
        t = BTree()
        for i in range(0, 20, 2):
            t = t.insert(i, i)
        assert t.ceiling(11) == (12, 12)

    def test_ceiling_above_max(self):
        t = BTree().insert(5, 5)
        assert t.ceiling(10) is None

    def test_floor_empty(self):
        assert BTree().floor(5) is None

    def test_ceiling_empty(self):
        assert BTree().ceiling(5) is None


# ============================================================
# Rank / Select
# ============================================================

class TestRankSelect:
    def test_rank(self):
        t = BTree(order=4)
        for i in range(10):
            t = t.insert(i * 2, i)
        assert t.rank(5) == 3  # Keys 0, 2, 4 < 5
        assert t.rank(0) == 0
        assert t.rank(20) == 10

    def test_rank_empty(self):
        assert BTree().rank(5) == 0

    def test_select(self):
        t = BTree(order=4)
        for i in range(10):
            t = t.insert(i, i * 10)
        assert t.select(0) == (0, 0)
        assert t.select(5) == (5, 50)
        assert t.select(9) == (9, 90)

    def test_select_out_of_range(self):
        t = BTree().insert(1, 1)
        with pytest.raises(IndexError):
            t.select(1)
        with pytest.raises(IndexError):
            t.select(-1)

    def test_kth_largest(self):
        t = BTree(order=4)
        for i in range(10):
            t = t.insert(i, i)
        assert t.kth_largest(0) == (9, 9)
        assert t.kth_largest(9) == (0, 0)

    def test_rank_select_roundtrip(self):
        t = BTree(order=4)
        for i in range(20):
            t = t.insert(i * 3, i)
        for i in range(20):
            k, v = t.select(i)
            assert t.rank(k) == i


# ============================================================
# Bulk construction
# ============================================================

class TestBulk:
    def test_from_sorted(self):
        items = [(i, i * 10) for i in range(20)]
        t = BTree.from_sorted(items, order=4)
        assert len(t) == 20
        for i in range(20):
            assert t[i] == i * 10

    def test_from_items(self):
        items = [(i, i) for i in [5, 3, 8, 1, 9, 2]]
        t = BTree.from_items(items, order=4)
        assert list(t) == [1, 2, 3, 5, 8, 9]

    def test_from_dict(self):
        d = {3: "c", 1: "a", 2: "b"}
        t = BTree.from_dict(d)
        assert t[1] == "a"
        assert t[2] == "b"
        assert t[3] == "c"

    def test_from_sorted_empty(self):
        t = BTree.from_sorted([])
        assert len(t) == 0

    def test_from_items_duplicates(self):
        # Last value wins for duplicate keys after sorting
        items = [(1, "first"), (2, "b"), (1, "second")]
        t = BTree.from_items(items, order=4)
        assert len(t) == 2


# ============================================================
# Merge / Diff
# ============================================================

class TestMergeDiff:
    def test_merge_disjoint(self):
        t1 = BTree().insert(1, "a").insert(3, "c")
        t2 = BTree().insert(2, "b").insert(4, "d")
        merged = t1.merge(t2)
        assert len(merged) == 4
        assert list(merged) == [1, 2, 3, 4]

    def test_merge_overlap_default(self):
        t1 = BTree().insert(1, "old")
        t2 = BTree().insert(1, "new")
        merged = t1.merge(t2)
        assert merged[1] == "new"  # Other wins by default

    def test_merge_overlap_custom(self):
        t1 = BTree().insert(1, 10)
        t2 = BTree().insert(1, 20)
        merged = t1.merge(t2, conflict=lambda k, a, b: a + b)
        assert merged[1] == 30

    def test_merge_empty(self):
        t1 = BTree().insert(1, "a")
        t2 = BTree()
        assert t1.merge(t2) == t1

    def test_diff_identical(self):
        t = BTree().insert(1, "a").insert(2, "b")
        d = t.diff(t)
        assert d == {'added': {}, 'removed': {}, 'changed': {}}

    def test_diff_added(self):
        t1 = BTree().insert(1, "a")
        t2 = BTree().insert(1, "a").insert(2, "b")
        d = t1.diff(t2)
        assert d['added'] == {2: "b"}
        assert d['removed'] == {}
        assert d['changed'] == {}

    def test_diff_removed(self):
        t1 = BTree().insert(1, "a").insert(2, "b")
        t2 = BTree().insert(1, "a")
        d = t1.diff(t2)
        assert d['removed'] == {2: "b"}

    def test_diff_changed(self):
        t1 = BTree().insert(1, "old")
        t2 = BTree().insert(1, "new")
        d = t1.diff(t2)
        assert d['changed'] == {1: ("old", "new")}


# ============================================================
# Map / Filter / Reduce
# ============================================================

class TestFunctional:
    def test_map_values(self):
        t = BTree().insert(1, 10).insert(2, 20)
        t2 = t.map_values(lambda v: v * 2)
        assert t2[1] == 20
        assert t2[2] == 40
        assert t[1] == 10  # Original unchanged

    def test_map_empty(self):
        t = BTree()
        t2 = t.map_values(lambda v: v)
        assert len(t2) == 0

    def test_filter(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        t2 = t.filter(lambda k, v: k % 2 == 0)
        assert list(t2) == [0, 2, 4, 6, 8]

    def test_filter_all(self):
        t = BTree().insert(1, 1)
        t2 = t.filter(lambda k, v: False)
        assert len(t2) == 0

    def test_reduce(self):
        t = BTree().insert(1, 10).insert(2, 20).insert(3, 30)
        total = t.reduce(lambda acc, kv: acc + kv[1], 0)
        assert total == 60

    def test_reduce_no_initial(self):
        t = BTree().insert(1, 10).insert(2, 20)
        result = t.reduce(lambda acc, kv: (acc[0] + kv[0], acc[1] + kv[1]))
        assert result == (3, 30)

    def test_reduce_empty_raises(self):
        with pytest.raises(ValueError):
            BTree().reduce(lambda acc, kv: acc)


# ============================================================
# Equality
# ============================================================

class TestEquality:
    def test_equal_trees(self):
        t1 = BTree().insert(1, "a").insert(2, "b")
        t2 = BTree().insert(2, "b").insert(1, "a")
        assert t1 == t2

    def test_unequal_size(self):
        t1 = BTree().insert(1, "a")
        t2 = BTree().insert(1, "a").insert(2, "b")
        assert t1 != t2

    def test_unequal_values(self):
        t1 = BTree().insert(1, "a")
        t2 = BTree().insert(1, "b")
        assert t1 != t2

    def test_not_equal_to_other_type(self):
        t = BTree().insert(1, "a")
        assert t != "not a tree"

    def test_empty_trees_equal(self):
        assert BTree() == BTree()


# ============================================================
# Pop min/max
# ============================================================

class TestPopMinMax:
    def test_pop_min(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(5, "e")
        (k, v), t2 = t.pop_min()
        assert (k, v) == (1, "a")
        assert len(t2) == 2
        assert 1 not in t2

    def test_pop_max(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(5, "e")
        (k, v), t2 = t.pop_max()
        assert (k, v) == (5, "e")
        assert len(t2) == 2

    def test_pop_min_empty_raises(self):
        with pytest.raises(ValueError):
            BTree().pop_min()

    def test_pop_max_empty_raises(self):
        with pytest.raises(ValueError):
            BTree().pop_max()

    def test_pop_all_min(self):
        t = BTree(order=4)
        for i in range(10):
            t = t.insert(i, i)
        items = []
        while t:
            (k, v), t = t.pop_min()
            items.append(k)
        assert items == list(range(10))


# ============================================================
# Slice
# ============================================================

class TestSlice:
    def test_slice(self):
        t = BTree()
        for i in range(20):
            t = t.insert(i, i)
        s = t.slice(5, 10)
        assert list(s) == [5, 6, 7, 8, 9]

    def test_slice_no_bounds(self):
        t = BTree().insert(1, 1).insert(2, 2)
        s = t.slice()
        assert list(s) == [1, 2]

    def test_slice_empty(self):
        t = BTree()
        for i in range(10):
            t = t.insert(i, i)
        s = t.slice(100, 200)
        assert len(s) == 0


# ============================================================
# to_dict
# ============================================================

class TestToDict:
    def test_to_dict(self):
        t = BTree().insert(3, "c").insert(1, "a").insert(2, "b")
        assert t.to_dict() == {1: "a", 2: "b", 3: "c"}

    def test_to_dict_empty(self):
        assert BTree().to_dict() == {}


# ============================================================
# Nearest
# ============================================================

class TestNearest:
    def test_nearest_exact(self):
        t = BTree().insert(1, 1).insert(5, 5).insert(10, 10)
        assert t.nearest(5) == (5, 5)

    def test_nearest_between(self):
        t = BTree().insert(1, 1).insert(10, 10)
        assert t.nearest(8) == (10, 10)
        assert t.nearest(3) == (1, 1)

    def test_nearest_below_min(self):
        t = BTree().insert(5, 5)
        assert t.nearest(1) == (5, 5)

    def test_nearest_above_max(self):
        t = BTree().insert(5, 5)
        assert t.nearest(10) == (5, 5)

    def test_nearest_empty(self):
        assert BTree().nearest(5) is None

    def test_nearest_tie(self):
        t = BTree().insert(0, 0).insert(10, 10)
        # Equal distance: floor wins
        assert t.nearest(5) == (0, 0)


# ============================================================
# Different orders
# ============================================================

class TestDifferentOrders:
    def test_order_3(self):
        t = BTree(order=3)
        for i in range(30):
            t = t.insert(i, i)
        assert len(t) == 30
        assert list(t) == list(range(30))

    def test_order_5(self):
        t = BTree(order=5)
        for i in range(100):
            t = t.insert(i, i)
        assert len(t) == 100

    def test_order_128(self):
        t = BTree(order=128)
        for i in range(200):
            t = t.insert(i, i)
        assert len(t) == 200

    def test_order_3_delete_all(self):
        t = BTree(order=3)
        for i in range(20):
            t = t.insert(i, i)
        for i in range(20):
            t = t.delete(i)
        assert len(t) == 0

    def test_order_5_delete_random(self):
        import random
        rng = random.Random(999)
        t = BTree(order=5)
        keys = list(range(40))
        for k in keys:
            t = t.insert(k, k)
        rng.shuffle(keys)
        for k in keys:
            t = t.delete(k)
        assert len(t) == 0


# ============================================================
# Stress tests
# ============================================================

class TestStress:
    def test_large_insert_delete(self):
        import random
        rng = random.Random(777)
        t = BTree(order=6)
        keys = list(range(500))
        rng.shuffle(keys)
        for k in keys:
            t = t.insert(k, k)
        assert len(t) == 500
        assert list(t) == list(range(500))
        rng.shuffle(keys)
        for k in keys[:250]:
            t = t.delete(k)
        assert len(t) == 250
        remaining = sorted(keys[250:])
        assert list(t) == remaining

    def test_mixed_insert_delete(self):
        import random
        rng = random.Random(555)
        t = BTree(order=4)
        present = set()
        for _ in range(300):
            op = rng.choice(['insert', 'insert', 'delete'])
            k = rng.randint(0, 99)
            if op == 'insert':
                t = t.insert(k, k)
                present.add(k)
            else:
                if k in present:
                    t = t.delete(k)
                    present.discard(k)
            assert len(t) == len(present)
        assert sorted(present) == list(t)

    def test_many_versions(self):
        versions = [BTree(order=4)]
        for i in range(50):
            versions.append(versions[-1].insert(i, i))
        # All versions should be valid
        for i, v in enumerate(versions):
            assert len(v) == i


# ============================================================
# _search_index helper
# ============================================================

class TestSearchIndex:
    def test_empty(self):
        assert _search_index((), 5) == 0

    def test_found(self):
        assert _search_index((1, 3, 5, 7), 5) == 2

    def test_between(self):
        assert _search_index((1, 3, 5, 7), 4) == 2

    def test_before_all(self):
        assert _search_index((1, 3, 5), 0) == 0

    def test_after_all(self):
        assert _search_index((1, 3, 5), 10) == 3


# ============================================================
# BTreeNode
# ============================================================

class TestBTreeNode:
    def test_leaf_node(self):
        n = BTreeNode((1, 2), ("a", "b"))
        assert n.leaf is True
        assert n.children is None
        assert "Leaf" in repr(n)

    def test_internal_node(self):
        left = BTreeNode((1,), ("a",))
        right = BTreeNode((3,), ("c",))
        n = BTreeNode((2,), ("b",), (left, right))
        assert n.leaf is False
        assert "Internal" in repr(n)


# ============================================================
# String keys
# ============================================================

class TestStringKeys:
    def test_string_keys(self):
        t = BTree()
        t = t.insert("banana", 1).insert("apple", 2).insert("cherry", 3)
        assert list(t) == ["apple", "banana", "cherry"]
        assert t["banana"] == 1

    def test_string_range(self):
        t = BTree()
        for w in ["alpha", "beta", "gamma", "delta", "epsilon"]:
            t = t.insert(w, w)
        result = list(t.range(lo="beta", hi="epsilon"))
        assert result == [("beta", "beta"), ("delta", "delta")]


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_none_values(self):
        t = BTree().insert(1, None)
        assert t[1] is None
        assert 1 in t

    def test_zero_key(self):
        t = BTree().insert(0, "zero")
        assert t[0] == "zero"

    def test_negative_keys(self):
        t = BTree().insert(-5, "neg").insert(5, "pos")
        assert t.min() == (-5, "neg")
        assert t.max() == (5, "pos")

    def test_float_keys(self):
        t = BTree().insert(1.5, "a").insert(2.7, "b").insert(0.3, "c")
        assert list(t) == [0.3, 1.5, 2.7]

    def test_insert_same_key_many_times(self):
        t = BTree(order=4)
        for i in range(50):
            t = t.insert(1, i)
        assert len(t) == 1
        assert t[1] == 49

    def test_contains_empty(self):
        assert 5 not in BTree()

    def test_get_empty(self):
        assert BTree().get(5) is None

    def test_height_empty(self):
        assert BTree().height() == 0

    def test_large_values(self):
        t = BTree()
        big = "x" * 10000
        t = t.insert(1, big)
        assert t[1] == big
        assert len(t[1]) == 10000
