"""Tests for C107: Splay Tree -- 5 variants."""
import pytest
from splay_tree import (
    SplayTree, SplayTreeMap, SplayTreeMultiSet,
    ImplicitSplayTree, LinkCutTree
)


# ============================================================
# SplayTree (core)
# ============================================================

class TestSplayTreeBasic:
    def test_empty(self):
        t = SplayTree()
        assert len(t) == 0
        assert not t
        assert t.root is None

    def test_insert_single(self):
        t = SplayTree()
        assert t.insert(5) is True
        assert len(t) == 1
        assert t.root.key == 5

    def test_insert_duplicate(self):
        t = SplayTree()
        t.insert(5)
        assert t.insert(5) is False
        assert len(t) == 1

    def test_insert_multiple(self):
        t = SplayTree()
        for x in [5, 3, 7, 1, 4]:
            t.insert(x)
        assert len(t) == 5
        assert sorted(t) == [1, 3, 4, 5, 7]

    def test_find_existing(self):
        t = SplayTree()
        for x in [10, 5, 15, 3, 7]:
            t.insert(x)
        assert t.find(7) == 7
        assert t.root.key == 7  # splayed to root

    def test_find_missing(self):
        t = SplayTree()
        for x in [10, 5, 15]:
            t.insert(x)
        assert t.find(8) is None

    def test_find_empty(self):
        t = SplayTree()
        assert t.find(5) is None

    def test_contains(self):
        t = SplayTree()
        t.insert(3)
        t.insert(7)
        assert 3 in t
        assert 7 in t
        assert 5 not in t

    def test_delete_leaf(self):
        t = SplayTree()
        for x in [5, 3, 7]:
            t.insert(x)
        assert t.delete(3) is True
        assert len(t) == 2
        assert 3 not in t

    def test_delete_root(self):
        t = SplayTree()
        t.insert(5)
        assert t.delete(5) is True
        assert len(t) == 0

    def test_delete_nonexistent(self):
        t = SplayTree()
        t.insert(5)
        assert t.delete(10) is False
        assert len(t) == 1

    def test_delete_empty(self):
        t = SplayTree()
        assert t.delete(5) is False

    def test_delete_all(self):
        t = SplayTree()
        for x in [1, 2, 3, 4, 5]:
            t.insert(x)
        for x in [3, 1, 5, 2, 4]:
            assert t.delete(x) is True
        assert len(t) == 0


class TestSplayTreeMinMax:
    def test_minimum(self):
        t = SplayTree()
        for x in [5, 3, 7, 1, 9]:
            t.insert(x)
        assert t.minimum() == 1

    def test_maximum(self):
        t = SplayTree()
        for x in [5, 3, 7, 1, 9]:
            t.insert(x)
        assert t.maximum() == 9

    def test_min_empty(self):
        assert SplayTree().minimum() is None

    def test_max_empty(self):
        assert SplayTree().maximum() is None

    def test_min_single(self):
        t = SplayTree()
        t.insert(42)
        assert t.minimum() == 42

    def test_max_single(self):
        t = SplayTree()
        t.insert(42)
        assert t.maximum() == 42


class TestSplayTreePredSucc:
    def test_predecessor(self):
        t = SplayTree()
        for x in [10, 5, 15, 3, 7, 12, 20]:
            t.insert(x)
        assert t.predecessor(10) == 7
        assert t.predecessor(15) == 12
        assert t.predecessor(3) is None

    def test_successor(self):
        t = SplayTree()
        for x in [10, 5, 15, 3, 7, 12, 20]:
            t.insert(x)
        assert t.successor(10) == 12
        assert t.successor(5) == 7
        assert t.successor(20) is None

    def test_pred_succ_missing_key(self):
        t = SplayTree()
        for x in [10, 20, 30]:
            t.insert(x)
        assert t.predecessor(15) == 10
        assert t.successor(15) == 20

    def test_pred_succ_empty(self):
        t = SplayTree()
        assert t.predecessor(5) is None
        assert t.successor(5) is None


class TestSplayTreeKthRank:
    def test_kth(self):
        t = SplayTree()
        for x in [5, 3, 7, 1, 9]:
            t.insert(x)
        assert t.kth(0) == 1
        assert t.kth(1) == 3
        assert t.kth(2) == 5
        assert t.kth(3) == 7
        assert t.kth(4) == 9

    def test_kth_out_of_range(self):
        t = SplayTree()
        t.insert(5)
        assert t.kth(-1) is None
        assert t.kth(1) is None

    def test_rank(self):
        t = SplayTree()
        for x in [5, 3, 7, 1, 9]:
            t.insert(x)
        assert t.rank(1) == 0
        assert t.rank(3) == 1
        assert t.rank(5) == 2
        assert t.rank(7) == 3
        assert t.rank(9) == 4

    def test_rank_missing(self):
        t = SplayTree()
        for x in [10, 20, 30]:
            t.insert(x)
        assert t.rank(15) == 1
        assert t.rank(5) == 0
        assert t.rank(35) == 3


class TestSplayTreeRange:
    def test_range_query(self):
        t = SplayTree()
        for x in range(1, 11):
            t.insert(x)
        assert t.range_query(3, 7) == [3, 4, 5, 6, 7]

    def test_range_empty(self):
        t = SplayTree()
        for x in [1, 5, 10]:
            t.insert(x)
        assert t.range_query(6, 9) == []

    def test_range_single(self):
        t = SplayTree()
        for x in [1, 5, 10]:
            t.insert(x)
        assert t.range_query(5, 5) == [5]


class TestSplayTreeSplitMerge:
    def test_split(self):
        t = SplayTree()
        for x in [1, 3, 5, 7, 9]:
            t.insert(x)
        left, right = t.split(5)
        assert left.to_sorted_list() == [1, 3, 5]
        assert right.to_sorted_list() == [7, 9]
        assert len(t) == 0  # original emptied

    def test_split_empty(self):
        t = SplayTree()
        left, right = t.split(5)
        assert len(left) == 0
        assert len(right) == 0

    def test_merge(self):
        t1 = SplayTree()
        t2 = SplayTree()
        for x in [1, 3, 5]:
            t1.insert(x)
        for x in [7, 9, 11]:
            t2.insert(x)
        merged = SplayTree.merge(t1, t2)
        assert merged.to_sorted_list() == [1, 3, 5, 7, 9, 11]

    def test_merge_empty_left(self):
        t1 = SplayTree()
        t2 = SplayTree()
        t2.insert(5)
        merged = SplayTree.merge(t1, t2)
        assert merged.to_sorted_list() == [5]

    def test_merge_empty_right(self):
        t1 = SplayTree()
        t2 = SplayTree()
        t1.insert(5)
        merged = SplayTree.merge(t1, t2)
        assert merged.to_sorted_list() == [5]

    def test_split_merge_roundtrip(self):
        t = SplayTree()
        for x in range(1, 20):
            t.insert(x)
        left, right = t.split(10)
        merged = SplayTree.merge(left, right)
        assert merged.to_sorted_list() == list(range(1, 20))


class TestSplayTreeMisc:
    def test_clear(self):
        t = SplayTree()
        for x in range(10):
            t.insert(x)
        t.clear()
        assert len(t) == 0
        assert t.root is None

    def test_iter(self):
        t = SplayTree()
        for x in [5, 2, 8, 1, 3]:
            t.insert(x)
        assert list(t) == [1, 2, 3, 5, 8]

    def test_large_sequential(self):
        t = SplayTree()
        for x in range(100):
            t.insert(x)
        assert len(t) == 100
        assert t.to_sorted_list() == list(range(100))

    def test_large_reverse(self):
        t = SplayTree()
        for x in range(99, -1, -1):
            t.insert(x)
        assert len(t) == 100
        assert t.minimum() == 0
        assert t.maximum() == 99

    def test_splay_property(self):
        """After access, element should be at root."""
        t = SplayTree()
        for x in [10, 5, 15, 3, 7]:
            t.insert(x)
        t.find(3)
        assert t.root.key == 3
        t.find(15)
        assert t.root.key == 15

    def test_string_keys(self):
        t = SplayTree()
        for s in ["banana", "apple", "cherry"]:
            t.insert(s)
        assert t.to_sorted_list() == ["apple", "banana", "cherry"]
        assert t.find("banana") == "banana"


# ============================================================
# SplayTreeMap
# ============================================================

class TestSplayTreeMap:
    def test_put_get(self):
        m = SplayTreeMap()
        m.put("a", 1)
        m.put("b", 2)
        assert m.get("a") == 1
        assert m.get("b") == 2

    def test_update_value(self):
        m = SplayTreeMap()
        m.put("a", 1)
        m.put("a", 10)
        assert m.get("a") == 10
        assert len(m) == 1

    def test_get_default(self):
        m = SplayTreeMap()
        assert m.get("x", 42) == 42

    def test_delete(self):
        m = SplayTreeMap()
        m.put("a", 1)
        m.put("b", 2)
        assert m.delete("a") is True
        assert m.get("a") is None
        assert len(m) == 1

    def test_delete_missing(self):
        m = SplayTreeMap()
        m.put("a", 1)
        assert m.delete("x") is False

    def test_contains(self):
        m = SplayTreeMap()
        m.put("a", 1)
        assert "a" in m
        assert "b" not in m

    def test_getitem_setitem(self):
        m = SplayTreeMap()
        m["x"] = 42
        assert m["x"] == 42

    def test_getitem_missing(self):
        m = SplayTreeMap()
        with pytest.raises(KeyError):
            _ = m["x"]

    def test_delitem(self):
        m = SplayTreeMap()
        m["a"] = 1
        del m["a"]
        assert "a" not in m

    def test_delitem_missing(self):
        m = SplayTreeMap()
        with pytest.raises(KeyError):
            del m["x"]

    def test_floor(self):
        m = SplayTreeMap()
        for i in [10, 20, 30]:
            m.put(i, i * 10)
        assert m.floor(25) == (20, 200)
        assert m.floor(10) == (10, 100)
        assert m.floor(5) is None

    def test_ceiling(self):
        m = SplayTreeMap()
        for i in [10, 20, 30]:
            m.put(i, i * 10)
        assert m.ceiling(15) == (20, 200)
        assert m.ceiling(30) == (30, 300)
        assert m.ceiling(35) is None

    def test_range_query(self):
        m = SplayTreeMap()
        for i in range(1, 11):
            m.put(i, i * 100)
        r = m.range_query(3, 6)
        assert r == [(3, 300), (4, 400), (5, 500), (6, 600)]

    def test_items_keys_values(self):
        m = SplayTreeMap()
        m.put(3, "c")
        m.put(1, "a")
        m.put(2, "b")
        assert m.keys() == [1, 2, 3]
        assert m.values() == ["a", "b", "c"]
        assert m.items() == [(1, "a"), (2, "b"), (3, "c")]

    def test_min_max_entry(self):
        m = SplayTreeMap()
        m.put(5, "e")
        m.put(1, "a")
        m.put(9, "i")
        assert m.min_entry() == (1, "a")
        assert m.max_entry() == (9, "i")

    def test_min_max_empty(self):
        m = SplayTreeMap()
        assert m.min_entry() is None
        assert m.max_entry() is None

    def test_iter(self):
        m = SplayTreeMap()
        m.put(3, "c")
        m.put(1, "a")
        assert list(m) == [(1, "a"), (3, "c")]

    def test_bool(self):
        m = SplayTreeMap()
        assert not m
        m.put(1, 1)
        assert m

    def test_put_with_none_value(self):
        """Ensure get distinguishes None value from missing key."""
        m = SplayTreeMap()
        m.put("x", None)
        assert m.get("x") is None
        assert "x" in m
        assert m.get("y") is None
        assert "y" not in m


# ============================================================
# SplayTreeMultiSet
# ============================================================

class TestSplayTreeMultiSet:
    def test_add_count(self):
        ms = SplayTreeMultiSet()
        ms.add(5)
        ms.add(5)
        ms.add(5)
        assert ms.count(5) == 3
        assert len(ms) == 3
        assert ms.distinct_count == 1

    def test_add_with_count(self):
        ms = SplayTreeMultiSet()
        ms.add(3, 10)
        assert ms.count(3) == 10
        assert len(ms) == 10

    def test_remove(self):
        ms = SplayTreeMultiSet()
        ms.add(5, 5)
        removed = ms.remove(5, 3)
        assert removed == 3
        assert ms.count(5) == 2

    def test_remove_all(self):
        ms = SplayTreeMultiSet()
        ms.add(5, 3)
        removed = ms.remove(5, 5)
        assert removed == 3
        assert ms.count(5) == 0
        assert 5 not in ms

    def test_remove_nonexistent(self):
        ms = SplayTreeMultiSet()
        assert ms.remove(5) == 0

    def test_contains(self):
        ms = SplayTreeMultiSet()
        ms.add(3)
        assert 3 in ms
        assert 5 not in ms

    def test_iter_with_duplicates(self):
        ms = SplayTreeMultiSet()
        ms.add(1, 2)
        ms.add(3, 3)
        assert list(ms) == [1, 1, 3, 3, 3]

    def test_distinct_keys(self):
        ms = SplayTreeMultiSet()
        ms.add(5, 3)
        ms.add(3, 2)
        ms.add(7, 1)
        assert ms.distinct_keys() == [3, 5, 7]

    def test_most_common(self):
        ms = SplayTreeMultiSet()
        ms.add(1, 5)
        ms.add(2, 10)
        ms.add(3, 3)
        result = ms.most_common(2)
        assert result == [(2, 10), (1, 5)]

    def test_most_common_all(self):
        ms = SplayTreeMultiSet()
        ms.add(1, 1)
        ms.add(2, 2)
        result = ms.most_common()
        assert result == [(2, 2), (1, 1)]

    def test_add_zero_ignored(self):
        ms = SplayTreeMultiSet()
        ms.add(5, 0)
        assert len(ms) == 0

    def test_multiple_keys(self):
        ms = SplayTreeMultiSet()
        for x in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]:
            ms.add(x)
        assert ms.count(5) == 3
        assert ms.count(1) == 2
        assert ms.count(3) == 2
        assert ms.distinct_count == 7


# ============================================================
# ImplicitSplayTree
# ============================================================

class TestImplicitSplayTree:
    def test_append_get(self):
        t = ImplicitSplayTree()
        t.append(10)
        t.append(20)
        t.append(30)
        assert t.get(0) == 10
        assert t.get(1) == 20
        assert t.get(2) == 30

    def test_insert_at_index(self):
        t = ImplicitSplayTree()
        t.append(1)
        t.append(3)
        t.insert(1, 2)  # insert 2 at index 1
        assert t.to_list() == [1, 2, 3]

    def test_insert_at_start(self):
        t = ImplicitSplayTree()
        t.append(2)
        t.append(3)
        t.insert(0, 1)
        assert t.to_list() == [1, 2, 3]

    def test_delete(self):
        t = ImplicitSplayTree()
        for x in [1, 2, 3, 4, 5]:
            t.append(x)
        val = t.delete(2)  # delete index 2 (value 3)
        assert val == 3
        assert t.to_list() == [1, 2, 4, 5]

    def test_delete_first(self):
        t = ImplicitSplayTree()
        for x in [1, 2, 3]:
            t.append(x)
        assert t.delete(0) == 1
        assert t.to_list() == [2, 3]

    def test_delete_last(self):
        t = ImplicitSplayTree()
        for x in [1, 2, 3]:
            t.append(x)
        assert t.delete(2) == 3
        assert t.to_list() == [1, 2]

    def test_set(self):
        t = ImplicitSplayTree()
        t.append(1)
        t.append(2)
        t.set(1, 42)
        assert t.get(1) == 42

    def test_index_out_of_range(self):
        t = ImplicitSplayTree()
        t.append(1)
        with pytest.raises(IndexError):
            t.get(5)
        with pytest.raises(IndexError):
            t.set(5, 1)
        with pytest.raises(IndexError):
            t.delete(5)
        with pytest.raises(IndexError):
            t.insert(5, 1)

    def test_reverse_range(self):
        t = ImplicitSplayTree.from_list([1, 2, 3, 4, 5])
        t.reverse_range(1, 3)  # reverse indices 1-3
        assert t.to_list() == [1, 4, 3, 2, 5]

    def test_reverse_entire(self):
        t = ImplicitSplayTree.from_list([1, 2, 3, 4, 5])
        t.reverse_range(0, 4)
        assert t.to_list() == [5, 4, 3, 2, 1]

    def test_reverse_prefix(self):
        t = ImplicitSplayTree.from_list([1, 2, 3, 4, 5])
        t.reverse_range(0, 2)
        assert t.to_list() == [3, 2, 1, 4, 5]

    def test_reverse_suffix(self):
        t = ImplicitSplayTree.from_list([1, 2, 3, 4, 5])
        t.reverse_range(2, 4)
        assert t.to_list() == [1, 2, 5, 4, 3]

    def test_reverse_single(self):
        t = ImplicitSplayTree.from_list([1, 2, 3])
        t.reverse_range(1, 1)  # no-op
        assert t.to_list() == [1, 2, 3]

    def test_from_list(self):
        t = ImplicitSplayTree.from_list([10, 20, 30])
        assert len(t) == 3
        assert t.to_list() == [10, 20, 30]

    def test_len(self):
        t = ImplicitSplayTree()
        assert len(t) == 0
        t.append(1)
        assert len(t) == 1

    def test_multiple_reverses(self):
        t = ImplicitSplayTree.from_list([1, 2, 3, 4, 5, 6])
        t.reverse_range(0, 5)
        t.reverse_range(0, 2)
        assert t.to_list() == [4, 5, 6, 3, 2, 1]

    def test_large_sequence(self):
        data = list(range(100))
        t = ImplicitSplayTree.from_list(data)
        assert t.to_list() == data
        assert t.get(50) == 50

    def test_insert_delete_interleaved(self):
        t = ImplicitSplayTree()
        t.append(1)
        t.append(2)
        t.append(3)
        t.delete(1)  # remove 2
        t.insert(1, 4)  # insert 4 at index 1
        assert t.to_list() == [1, 4, 3]


# ============================================================
# LinkCutTree
# ============================================================

class TestLinkCutTree:
    def test_link_connected(self):
        lct = LinkCutTree(5)
        assert not lct.connected(0, 1)
        lct.link(0, 1)
        assert lct.connected(0, 1)

    def test_link_transitive(self):
        lct = LinkCutTree(5)
        lct.link(0, 1)
        lct.link(1, 2)
        assert lct.connected(0, 2)

    def test_cut(self):
        lct = LinkCutTree(5)
        lct.link(0, 1)
        lct.link(1, 2)
        assert lct.connected(0, 2)
        lct.cut(0, 1)
        assert not lct.connected(0, 2)
        assert lct.connected(1, 2)

    def test_link_already_connected(self):
        lct = LinkCutTree(3)
        lct.link(0, 1)
        assert lct.link(0, 1) is False

    def test_path_aggregate(self):
        lct = LinkCutTree(4)
        for i in range(4):
            lct.set_value(i, i + 1)  # values: 1, 2, 3, 4
        lct.link(0, 1)
        lct.link(1, 2)
        lct.link(2, 3)
        # Path 0->3: 1+2+3+4 = 10
        assert lct.path_aggregate(0, 3) == 10

    def test_set_get_value(self):
        lct = LinkCutTree(3)
        lct.set_value(0, 100)
        assert lct.get_value(0) == 100

    def test_path_length(self):
        lct = LinkCutTree(5)
        lct.link(0, 1)
        lct.link(1, 2)
        lct.link(2, 3)
        assert lct.path_length(0, 3) == 4

    def test_lca(self):
        lct = LinkCutTree(5)
        lct.link(1, 0)
        lct.link(2, 0)
        lct.link(3, 1)
        lct.link(4, 2)
        # Tree: 0 is root, 1 and 2 are children, 3 under 1, 4 under 2
        assert lct.lca(3, 4) == 0

    def test_dynamic_forest(self):
        """Build, query, cut, rebuild."""
        lct = LinkCutTree(6)
        # Build path: 0-1-2-3-4-5
        for i in range(5):
            lct.link(i, i + 1)
        assert lct.connected(0, 5)

        # Cut middle
        lct.cut(2, 3)
        assert not lct.connected(0, 5)
        assert lct.connected(0, 2)
        assert lct.connected(3, 5)

        # Relink differently
        lct.link(2, 4)
        assert lct.connected(0, 5)

    def test_star_graph(self):
        n = 10
        lct = LinkCutTree(n)
        for i in range(1, n):
            lct.link(i, 0)
        for i in range(1, n):
            assert lct.connected(i, 0)
        for i in range(1, n):
            for j in range(i + 1, n):
                assert lct.connected(i, j)

    def test_isolated_nodes(self):
        lct = LinkCutTree(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert not lct.connected(i, j)

    def test_path_aggregate_after_update(self):
        lct = LinkCutTree(3)
        lct.set_value(0, 1)
        lct.set_value(1, 2)
        lct.set_value(2, 3)
        lct.link(0, 1)
        lct.link(1, 2)
        assert lct.path_aggregate(0, 2) == 6
        lct.set_value(1, 10)
        assert lct.path_aggregate(0, 2) == 14

    def test_self_connected(self):
        lct = LinkCutTree(3)
        assert lct.connected(0, 0)


# ============================================================
# Stress / property tests
# ============================================================

class TestSplayTreeStress:
    def test_insert_delete_stress(self):
        import random
        random.seed(42)
        t = SplayTree()
        ref = set()
        for _ in range(200):
            op = random.randint(0, 2)
            val = random.randint(0, 50)
            if op <= 1:
                t.insert(val)
                ref.add(val)
            else:
                t.delete(val)
                ref.discard(val)
        assert t.to_sorted_list() == sorted(ref)
        assert len(t) == len(ref)

    def test_kth_rank_consistency(self):
        t = SplayTree()
        for x in [10, 30, 50, 70, 90]:
            t.insert(x)
        for i in range(5):
            key = t.kth(i)
            assert t.rank(key) == i

    def test_split_merge_stress(self):
        import random
        random.seed(123)
        keys = random.sample(range(1000), 50)
        t = SplayTree()
        for k in keys:
            t.insert(k)
        pivot = sorted(keys)[25]
        left, right = t.split(pivot)
        for k in left:
            assert k <= pivot
        for k in right:
            assert k > pivot
        merged = SplayTree.merge(left, right)
        assert merged.to_sorted_list() == sorted(keys)

    def test_implicit_reverse_stress(self):
        import random
        random.seed(456)
        data = list(range(50))
        t = ImplicitSplayTree.from_list(data)
        for _ in range(20):
            lo = random.randint(0, 48)
            hi = random.randint(lo, 49)
            t.reverse_range(lo, hi)
            data[lo:hi+1] = data[lo:hi+1][::-1]
        assert t.to_list() == data

    def test_map_floor_ceiling_stress(self):
        import random
        random.seed(789)
        m = SplayTreeMap()
        keys = sorted(random.sample(range(1000), 50))
        for k in keys:
            m.put(k, k * 10)
        for _ in range(30):
            q = random.randint(0, 999)
            f = m.floor(q)
            c = m.ceiling(q)
            if f is not None:
                assert f[0] <= q
            if c is not None:
                assert c[0] >= q

    def test_link_cut_chain(self):
        """Build and tear down a long chain."""
        n = 50
        lct = LinkCutTree(n)
        for i in range(n - 1):
            lct.link(i, i + 1)
        assert lct.connected(0, n - 1)
        assert lct.path_length(0, n - 1) == n
        # Cut every other edge
        for i in range(0, n - 1, 2):
            lct.cut(i, i + 1)
        assert not lct.connected(0, 1)

    def test_multiset_stress(self):
        import random
        random.seed(321)
        ms = SplayTreeMultiSet()
        counts = {}
        for _ in range(200):
            k = random.randint(0, 20)
            if random.random() < 0.7:
                c = random.randint(1, 5)
                ms.add(k, c)
                counts[k] = counts.get(k, 0) + c
            else:
                c = random.randint(1, 3)
                removed = ms.remove(k, c)
                if k in counts:
                    actual = min(c, counts[k])
                    assert removed == actual
                    counts[k] -= actual
                    if counts[k] <= 0:
                        del counts[k]
                else:
                    assert removed == 0
        for k, v in counts.items():
            assert ms.count(k) == v


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
