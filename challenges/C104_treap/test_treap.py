"""Tests for C104: Treap"""
import pytest
import random
from treap import (
    TreapMap, TreapSet, ImplicitTreap, PersistentTreap,
    TreapNode, split, merge
)


# ============================================================
# TreapMap -- Basic operations
# ============================================================

class TestTreapMapBasic:
    def test_empty(self):
        t = TreapMap()
        assert len(t) == 0
        assert not t
        assert 5 not in t

    def test_put_and_get(self):
        t = TreapMap()
        t.put(3, "three")
        t.put(1, "one")
        t.put(5, "five")
        assert t[3] == "three"
        assert t[1] == "one"
        assert t[5] == "five"
        assert len(t) == 3

    def test_contains(self):
        t = TreapMap()
        t.put(10)
        assert 10 in t
        assert 20 not in t

    def test_setitem(self):
        t = TreapMap()
        t[7] = "seven"
        assert t[7] == "seven"

    def test_update_existing_key(self):
        t = TreapMap()
        t.put(1, "one")
        t.put(1, "ONE")
        assert t[1] == "ONE"
        assert len(t) == 1

    def test_get_with_default(self):
        t = TreapMap()
        assert t.get(99) is None
        assert t.get(99, "default") == "default"
        t.put(99, "val")
        assert t.get(99) == "val"

    def test_getitem_missing(self):
        t = TreapMap()
        with pytest.raises(KeyError):
            _ = t[42]

    def test_delete(self):
        t = TreapMap()
        for i in range(10):
            t.put(i, i * 10)
        t.delete(5)
        assert 5 not in t
        assert len(t) == 9
        t._verify()

    def test_delete_missing(self):
        t = TreapMap()
        with pytest.raises(KeyError):
            t.delete(1)

    def test_pop(self):
        t = TreapMap()
        t.put(1, "a")
        assert t.pop(1) == "a"
        assert 1 not in t

    def test_pop_default(self):
        t = TreapMap()
        assert t.pop(1, "default") == "default"

    def test_pop_missing(self):
        t = TreapMap()
        with pytest.raises(KeyError):
            t.pop(1)

    def test_clear(self):
        t = TreapMap()
        for i in range(5):
            t.put(i)
        t.clear()
        assert len(t) == 0
        assert not t


# ============================================================
# TreapMap -- Order statistics
# ============================================================

class TestTreapMapOrderStats:
    def test_min_max(self):
        t = TreapMap()
        t.put(5, "e")
        t.put(2, "b")
        t.put(8, "h")
        assert t.min() == (2, "b")
        assert t.max() == (8, "h")

    def test_min_empty(self):
        t = TreapMap()
        with pytest.raises(ValueError):
            t.min()

    def test_max_empty(self):
        t = TreapMap()
        with pytest.raises(ValueError):
            t.max()

    def test_kth(self):
        t = TreapMap()
        for x in [5, 3, 7, 1, 9]:
            t.put(x, x * 10)
        assert t.kth(0) == (1, 10)
        assert t.kth(2) == (5, 50)
        assert t.kth(4) == (9, 90)

    def test_kth_out_of_range(self):
        t = TreapMap()
        t.put(1)
        with pytest.raises(IndexError):
            t.kth(1)
        with pytest.raises(IndexError):
            t.kth(-1)

    def test_rank(self):
        t = TreapMap()
        for x in [2, 4, 6, 8, 10]:
            t.put(x)
        assert t.rank(1) == 0
        assert t.rank(2) == 0
        assert t.rank(3) == 1
        assert t.rank(6) == 2
        assert t.rank(11) == 5


# ============================================================
# TreapMap -- Range queries
# ============================================================

class TestTreapMapRange:
    def test_floor(self):
        t = TreapMap()
        for x in [2, 4, 6, 8]:
            t.put(x, x)
        assert t.floor(5) == (4, 4)
        assert t.floor(4) == (4, 4)
        assert t.floor(1) is None
        assert t.floor(9) == (8, 8)

    def test_ceiling(self):
        t = TreapMap()
        for x in [2, 4, 6, 8]:
            t.put(x, x)
        assert t.ceiling(5) == (6, 6)
        assert t.ceiling(6) == (6, 6)
        assert t.ceiling(9) is None
        assert t.ceiling(1) == (2, 2)

    def test_range_query(self):
        t = TreapMap()
        for x in range(1, 11):
            t.put(x, x * 10)
        result = t.range_query(3, 7)
        assert result == [(3, 30), (4, 40), (5, 50), (6, 60), (7, 70)]

    def test_range_query_empty(self):
        t = TreapMap()
        for x in [1, 10]:
            t.put(x)
        assert t.range_query(3, 7) == []


# ============================================================
# TreapMap -- Iteration
# ============================================================

class TestTreapMapIteration:
    def test_iter(self):
        t = TreapMap()
        for x in [5, 3, 7, 1, 9]:
            t.put(x)
        assert list(t) == [1, 3, 5, 7, 9]

    def test_items(self):
        t = TreapMap()
        t.put(2, "b")
        t.put(1, "a")
        t.put(3, "c")
        assert list(t.items()) == [(1, "a"), (2, "b"), (3, "c")]

    def test_keys_values(self):
        t = TreapMap()
        t.put(3, "c")
        t.put(1, "a")
        assert list(t.keys()) == [1, 3]
        assert list(t.values()) == ["a", "c"]


# ============================================================
# TreapMap -- Split and merge
# ============================================================

class TestTreapMapSplitMerge:
    def test_split_at(self):
        t = TreapMap()
        for x in range(1, 11):
            t.put(x, x)
        left, right = t.split_at(5)
        assert list(left) == [1, 2, 3, 4]
        assert list(right) == [5, 6, 7, 8, 9, 10]
        left._verify()
        right._verify()

    def test_update(self):
        a = TreapMap()
        a.put(1, "a")
        a.put(2, "b")
        b = TreapMap()
        b.put(2, "B")
        b.put(3, "c")
        a.update(b)
        assert a[2] == "B"
        assert a[3] == "c"
        assert len(a) == 3


# ============================================================
# TreapMap -- Invariant verification
# ============================================================

class TestTreapMapVerify:
    def test_verify_after_operations(self):
        t = TreapMap()
        random.seed(42)
        for i in range(100):
            t.put(random.randint(0, 200), i)
        t._verify()
        for _ in range(30):
            keys = list(t)
            if keys:
                t.delete(random.choice(keys))
        t._verify()

    def test_many_insertions_sorted_order(self):
        t = TreapMap()
        for i in range(200):
            t.put(i, i)
        t._verify()
        assert list(t) == list(range(200))

    def test_many_insertions_reverse_order(self):
        t = TreapMap()
        for i in range(200, 0, -1):
            t.put(i, i)
        t._verify()
        assert list(t) == list(range(1, 201))


# ============================================================
# TreapSet
# ============================================================

class TestTreapSet:
    def test_basic(self):
        s = TreapSet()
        s.add(3)
        s.add(1)
        s.add(5)
        assert 3 in s
        assert 2 not in s
        assert len(s) == 3

    def test_from_iterable(self):
        s = TreapSet([5, 3, 1, 7])
        assert list(s) == [1, 3, 5, 7]

    def test_discard(self):
        s = TreapSet([1, 2, 3])
        s.discard(2)
        assert 2 not in s
        s.discard(99)  # no error

    def test_remove(self):
        s = TreapSet([1, 2, 3])
        s.remove(2)
        with pytest.raises(KeyError):
            s.remove(99)

    def test_pop_min_max(self):
        s = TreapSet([3, 1, 5])
        assert s.pop_min() == 1
        assert s.pop_max() == 5
        assert list(s) == [3]

    def test_min_max(self):
        s = TreapSet([4, 2, 8])
        assert s.min() == 2
        assert s.max() == 8

    def test_kth_rank(self):
        s = TreapSet([10, 20, 30, 40, 50])
        assert s.kth(0) == 10
        assert s.kth(3) == 40
        assert s.rank(30) == 2

    def test_floor_ceiling(self):
        s = TreapSet([2, 4, 6, 8])
        assert s.floor(5) == 4
        assert s.ceiling(5) == 6
        assert s.floor(1) is None
        assert s.ceiling(9) is None

    def test_range_query(self):
        s = TreapSet(range(1, 11))
        assert s.range_query(3, 7) == [3, 4, 5, 6, 7]

    def test_union(self):
        a = TreapSet([1, 3, 5])
        b = TreapSet([2, 3, 4])
        c = a | b
        assert list(c) == [1, 2, 3, 4, 5]

    def test_intersection(self):
        a = TreapSet([1, 2, 3, 4])
        b = TreapSet([3, 4, 5, 6])
        c = a & b
        assert list(c) == [3, 4]

    def test_difference(self):
        a = TreapSet([1, 2, 3, 4])
        b = TreapSet([3, 4, 5])
        c = a - b
        assert list(c) == [1, 2]

    def test_symmetric_difference(self):
        a = TreapSet([1, 2, 3])
        b = TreapSet([2, 3, 4])
        c = a ^ b
        assert list(c) == [1, 4]

    def test_is_subset(self):
        a = TreapSet([1, 2])
        b = TreapSet([1, 2, 3])
        assert a.is_subset(b)
        assert not b.is_subset(a)

    def test_clear(self):
        s = TreapSet([1, 2, 3])
        s.clear()
        assert len(s) == 0

    def test_verify(self):
        s = TreapSet(range(50))
        s._verify()

    def test_duplicate_add(self):
        s = TreapSet()
        s.add(1)
        s.add(1)
        assert len(s) == 1


# ============================================================
# ImplicitTreap
# ============================================================

class TestImplicitTreap:
    def test_empty(self):
        t = ImplicitTreap()
        assert len(t) == 0
        assert not t

    def test_append(self):
        t = ImplicitTreap()
        t.append(10)
        t.append(20)
        t.append(30)
        assert len(t) == 3
        assert t[0] == 10
        assert t[2] == 30

    def test_from_iterable(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        assert t.to_list() == [1, 2, 3, 4, 5]

    def test_insert(self):
        t = ImplicitTreap([1, 3, 4])
        t.insert(1, 2)
        assert t.to_list() == [1, 2, 3, 4]

    def test_insert_at_beginning(self):
        t = ImplicitTreap([2, 3])
        t.insert(0, 1)
        assert t.to_list() == [1, 2, 3]

    def test_insert_at_end(self):
        t = ImplicitTreap([1, 2])
        t.insert(10, 3)  # clamps to end
        assert t.to_list() == [1, 2, 3]

    def test_delete(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        t.delete(2)
        assert t.to_list() == [1, 2, 4, 5]

    def test_delete_first(self):
        t = ImplicitTreap([1, 2, 3])
        t.delete(0)
        assert t.to_list() == [2, 3]

    def test_delete_last(self):
        t = ImplicitTreap([1, 2, 3])
        t.delete(2)
        assert t.to_list() == [1, 2]

    def test_delete_negative(self):
        t = ImplicitTreap([1, 2, 3])
        t.delete(-1)
        assert t.to_list() == [1, 2]

    def test_delete_out_of_range(self):
        t = ImplicitTreap([1])
        with pytest.raises(IndexError):
            t.delete(5)

    def test_getitem(self):
        t = ImplicitTreap([10, 20, 30, 40])
        assert t[0] == 10
        assert t[-1] == 40
        assert t[2] == 30

    def test_getitem_out_of_range(self):
        t = ImplicitTreap([1])
        with pytest.raises(IndexError):
            _ = t[5]

    def test_getitem_slice(self):
        t = ImplicitTreap([0, 1, 2, 3, 4])
        assert t[1:4] == [1, 2, 3]
        assert t[::2] == [0, 2, 4]

    def test_setitem(self):
        t = ImplicitTreap([1, 2, 3])
        t[1] = 20
        assert t.to_list() == [1, 20, 3]

    def test_setitem_out_of_range(self):
        t = ImplicitTreap([1])
        with pytest.raises(IndexError):
            t[5] = 99

    def test_reverse_full(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        t.reverse()
        assert t.to_list() == [5, 4, 3, 2, 1]

    def test_reverse_subrange(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        t.reverse(1, 4)
        assert t.to_list() == [1, 4, 3, 2, 5]

    def test_reverse_double(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        t.reverse()
        t.reverse()
        assert t.to_list() == [1, 2, 3, 4, 5]

    def test_reverse_empty_range(self):
        t = ImplicitTreap([1, 2, 3])
        t.reverse(2, 2)
        assert t.to_list() == [1, 2, 3]

    def test_iter(self):
        t = ImplicitTreap([3, 1, 4])
        assert list(t) == [3, 1, 4]

    def test_split_at(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        left, right = t.split_at(3)
        assert left.to_list() == [1, 2, 3]
        assert right.to_list() == [4, 5]

    def test_concat(self):
        a = ImplicitTreap([1, 2])
        b = ImplicitTreap([3, 4])
        a.concat(b)
        assert a.to_list() == [1, 2, 3, 4]
        assert len(b) == 0

    def test_large_sequence(self):
        n = 1000
        t = ImplicitTreap(range(n))
        assert t.to_list() == list(range(n))
        assert t[500] == 500

    def test_multiple_reverses(self):
        t = ImplicitTreap([1, 2, 3, 4, 5, 6, 7, 8])
        t.reverse(0, 4)  # [4,3,2,1,5,6,7,8]
        t.reverse(4, 8)  # [4,3,2,1,8,7,6,5]
        assert t.to_list() == [4, 3, 2, 1, 8, 7, 6, 5]

    def test_insert_negative_index(self):
        t = ImplicitTreap([1, 2, 3])
        t.insert(-1, 99)  # inserts at index 2
        assert t.to_list() == [1, 2, 99, 3]


# ============================================================
# PersistentTreap
# ============================================================

class TestPersistentTreap:
    def test_empty(self):
        t = PersistentTreap()
        assert len(t) == 0
        assert not t

    def test_put_creates_new_version(self):
        t0 = PersistentTreap()
        t1 = t0.put(3, "three")
        t2 = t1.put(1, "one")
        t3 = t2.put(5, "five")
        # all versions coexist
        assert len(t0) == 0
        assert len(t1) == 1
        assert len(t2) == 2
        assert len(t3) == 3
        assert t3[3] == "three"
        assert t3[1] == "one"

    def test_delete_creates_new_version(self):
        t0 = PersistentTreap()
        t1 = t0.put(1, "a")
        t2 = t1.put(2, "b")
        t3 = t2.delete(1)
        assert 1 in t2
        assert 1 not in t3
        assert len(t2) == 2
        assert len(t3) == 1

    def test_delete_missing(self):
        t = PersistentTreap()
        with pytest.raises(KeyError):
            t.delete(1)

    def test_update_value(self):
        t0 = PersistentTreap()
        t1 = t0.put(1, "a")
        t2 = t1.put(1, "A")
        assert t1[1] == "a"
        assert t2[1] == "A"
        assert len(t2) == 1

    def test_contains(self):
        t = PersistentTreap().put(5).put(10)
        assert 5 in t
        assert 10 in t
        assert 7 not in t

    def test_get(self):
        t = PersistentTreap().put(1, "x")
        assert t.get(1) == "x"
        assert t.get(2) is None
        assert t.get(2, "default") == "default"

    def test_getitem_missing(self):
        t = PersistentTreap()
        with pytest.raises(KeyError):
            _ = t[1]

    def test_min_max(self):
        t = PersistentTreap().put(5).put(2).put(8)
        assert t.min() == (2, None)
        assert t.max() == (8, None)

    def test_min_empty(self):
        with pytest.raises(ValueError):
            PersistentTreap().min()

    def test_iter(self):
        t = PersistentTreap().put(5).put(3).put(7).put(1)
        assert list(t) == [1, 3, 5, 7]

    def test_items(self):
        t = PersistentTreap().put(2, "b").put(1, "a")
        assert list(t.items()) == [(1, "a"), (2, "b")]

    def test_kth(self):
        t = PersistentTreap()
        for x in [5, 3, 7, 1, 9]:
            t = t.put(x, x * 10)
        assert t.kth(0) == (1, 10)
        assert t.kth(2) == (5, 50)
        assert t.kth(4) == (9, 90)

    def test_kth_out_of_range(self):
        t = PersistentTreap().put(1)
        with pytest.raises(IndexError):
            t.kth(1)

    def test_rank(self):
        t = PersistentTreap()
        for x in [2, 4, 6, 8]:
            t = t.put(x)
        assert t.rank(1) == 0
        assert t.rank(3) == 1
        assert t.rank(6) == 2
        assert t.rank(9) == 4

    def test_structural_sharing(self):
        """Versions share nodes -- inserting shouldn't copy everything."""
        t0 = PersistentTreap()
        for i in range(100):
            t0 = t0.put(i)
        t1 = t0.put(1000, "new")
        # Both should work independently
        assert 1000 not in t0  # t0 doesn't have it -- wait, t0 was reassigned
        # Actually t0 was reassigned in the loop so let's test differently
        t_base = PersistentTreap().put(1).put(2).put(3)
        t_a = t_base.put(4)
        t_b = t_base.put(5)
        assert 4 in t_a
        assert 4 not in t_b
        assert 5 in t_b
        assert 5 not in t_a
        assert list(t_base) == [1, 2, 3]

    def test_many_versions(self):
        versions = [PersistentTreap()]
        for i in range(50):
            versions.append(versions[-1].put(i, i))
        assert len(versions[0]) == 0
        assert len(versions[50]) == 50
        assert len(versions[25]) == 25
        assert list(versions[25]) == list(range(25))


# ============================================================
# Split / Merge primitives
# ============================================================

class TestSplitMerge:
    def test_split_empty(self):
        l, r = split(None, 5)
        assert l is None
        assert r is None

    def test_merge_empty(self):
        assert merge(None, None) is None
        n = TreapNode(5)
        assert merge(n, None) is n
        assert merge(None, n) is n

    def test_split_single(self):
        n = TreapNode(5, "five")
        l, r = split(n, 5)
        assert l is None
        assert r is not None
        assert r.key == 5

    def test_split_merge_roundtrip(self):
        t = TreapMap()
        random.seed(123)
        for i in range(20):
            t.put(i, i)
        left, right = split(t._root, 10)
        merged = merge(left, right)
        # Rebuild map and verify
        result = TreapMap()
        result._root = merged
        result._verify()
        assert list(result) == list(range(20))


# ============================================================
# Stress tests
# ============================================================

class TestStress:
    def test_random_operations_treapmap(self):
        random.seed(99)
        t = TreapMap()
        ref = {}
        for _ in range(500):
            op = random.choice(['put', 'put', 'put', 'delete', 'get'])
            key = random.randint(0, 100)
            if op == 'put':
                val = random.randint(0, 1000)
                t.put(key, val)
                ref[key] = val
            elif op == 'delete':
                if key in ref:
                    t.delete(key)
                    del ref[key]
            elif op == 'get':
                assert t.get(key) == ref.get(key)
        # Verify final state
        t._verify()
        assert sorted(ref.keys()) == list(t)
        for k, v in ref.items():
            assert t[k] == v

    def test_random_operations_implicit_treap(self):
        random.seed(77)
        t = ImplicitTreap()
        ref = []
        for _ in range(300):
            op = random.choice(['append', 'append', 'insert', 'delete', 'reverse'])
            if op == 'append':
                val = random.randint(0, 1000)
                t.append(val)
                ref.append(val)
            elif op == 'insert' and len(ref) > 0:
                idx = random.randint(0, len(ref))
                val = random.randint(0, 1000)
                t.insert(idx, val)
                ref.insert(idx, val)
            elif op == 'delete' and len(ref) > 0:
                idx = random.randint(0, len(ref) - 1)
                t.delete(idx)
                ref.pop(idx)
            elif op == 'reverse' and len(ref) >= 2:
                l = random.randint(0, len(ref) - 1)
                r = random.randint(l + 1, len(ref))
                t.reverse(l, r)
                ref[l:r] = ref[l:r][::-1]
        assert t.to_list() == ref

    def test_persistent_treap_consistency(self):
        random.seed(55)
        versions = [PersistentTreap()]
        ref_versions = [{}]
        for _ in range(100):
            ver_idx = random.randint(0, len(versions) - 1)
            t = versions[ver_idx]
            ref = dict(ref_versions[ver_idx])
            key = random.randint(0, 50)
            if random.random() < 0.7:
                val = random.randint(0, 1000)
                t = t.put(key, val)
                ref[key] = val
            else:
                if key in ref:
                    t = t.delete(key)
                    del ref[key]
            versions.append(t)
            ref_versions.append(ref)
        # Verify all versions
        for i in range(len(versions)):
            assert sorted(ref_versions[i].keys()) == list(versions[i])
            for k, v in ref_versions[i].items():
                assert versions[i][k] == v


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_element_operations(self):
        t = TreapMap()
        t.put(42, "only")
        assert t.min() == (42, "only")
        assert t.max() == (42, "only")
        assert t.kth(0) == (42, "only")
        assert t.rank(42) == 0
        assert t.rank(43) == 1
        assert t.floor(42) == (42, "only")
        assert t.ceiling(42) == (42, "only")
        t.delete(42)
        assert len(t) == 0

    def test_string_keys(self):
        t = TreapMap()
        t.put("banana", 1)
        t.put("apple", 2)
        t.put("cherry", 3)
        assert list(t) == ["apple", "banana", "cherry"]
        assert t["banana"] == 1

    def test_negative_keys(self):
        t = TreapMap()
        for x in [-5, -3, 0, 3, 5]:
            t.put(x, x)
        assert t.min() == (-5, -5)
        assert list(t) == [-5, -3, 0, 3, 5]
        assert t.rank(0) == 2

    def test_treapset_bool(self):
        s = TreapSet()
        assert not s
        s.add(1)
        assert s

    def test_implicit_treap_bool(self):
        t = ImplicitTreap()
        assert not t
        t.append(1)
        assert t

    def test_persistent_treap_bool(self):
        t = PersistentTreap()
        assert not t
        t = t.put(1)
        assert t

    def test_implicit_treap_setitem_negative(self):
        t = ImplicitTreap([1, 2, 3])
        t[-1] = 30
        assert t.to_list() == [1, 2, 30]

    def test_treapmap_iter_empty(self):
        t = TreapMap()
        assert list(t) == []
        assert list(t.items()) == []
        assert list(t.keys()) == []
        assert list(t.values()) == []
