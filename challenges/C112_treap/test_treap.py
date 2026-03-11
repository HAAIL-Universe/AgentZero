"""Tests for C112: Treap -- 5 variants."""

import random
import pytest
from treap import (
    Treap, TreapNode,
    ImplicitTreap,
    PersistentTreap,
    MergeableTreap,
    IntervalTreap,
)


# ============================================================
# Variant 1: Treap
# ============================================================

class TestTreap:
    def test_empty(self):
        t = Treap()
        assert len(t) == 0
        assert not t
        assert t.verify()

    def test_insert_single(self):
        t = Treap()
        t.insert(5, "five")
        assert len(t) == 1
        assert t.search(5) == "five"
        assert 5 in t
        assert t.verify()

    def test_insert_multiple(self):
        t = Treap()
        for i in [3, 1, 4, 1, 5, 9, 2, 6]:
            t.insert(i, i * 10)
        assert len(t) == 7  # duplicate 1 counted once
        assert t.search(3) == 30
        assert t.verify()

    def test_insert_update(self):
        t = Treap()
        t.insert(5, "old")
        t.insert(5, "new")
        assert len(t) == 1
        assert t.search(5) == "new"

    def test_delete(self):
        t = Treap()
        for i in range(10):
            t.insert(i)
        assert t.delete(5)
        assert 5 not in t
        assert len(t) == 9
        assert t.verify()

    def test_delete_nonexistent(self):
        t = Treap()
        t.insert(1)
        assert not t.delete(99)
        assert len(t) == 1

    def test_delete_all(self):
        t = Treap()
        keys = list(range(20))
        for k in keys:
            t.insert(k)
        random.shuffle(keys)
        for k in keys:
            assert t.delete(k)
        assert len(t) == 0
        assert t.verify()

    def test_search_not_found(self):
        t = Treap()
        t.insert(1)
        assert t.search(99) is None
        assert 99 not in t

    def test_min_max(self):
        t = Treap()
        for i in [5, 3, 8, 1, 9]:
            t.insert(i)
        assert t.min() == 1
        assert t.max() == 9

    def test_min_max_empty(self):
        t = Treap()
        with pytest.raises(ValueError):
            t.min()
        with pytest.raises(ValueError):
            t.max()

    def test_floor_ceiling(self):
        t = Treap()
        for i in [2, 4, 6, 8]:
            t.insert(i)
        assert t.floor(5) == 4
        assert t.floor(6) == 6
        assert t.floor(1) is None
        assert t.ceiling(5) == 6
        assert t.ceiling(6) == 6
        assert t.ceiling(9) is None

    def test_rank(self):
        t = Treap()
        for i in [1, 3, 5, 7, 9]:
            t.insert(i)
        assert t.rank(1) == 0
        assert t.rank(5) == 2
        assert t.rank(6) == 3
        assert t.rank(10) == 5

    def test_select(self):
        t = Treap()
        for i in [5, 3, 1, 4, 2]:
            t.insert(i)
        assert t.select(0) == 1
        assert t.select(2) == 3
        assert t.select(4) == 5

    def test_select_out_of_range(self):
        t = Treap()
        t.insert(1)
        with pytest.raises(IndexError):
            t.select(1)
        with pytest.raises(IndexError):
            t.select(-1)

    def test_range_query(self):
        t = Treap()
        for i in range(1, 11):
            t.insert(i)
        assert t.range_query(3, 7) == [3, 4, 5, 6, 7]
        assert t.range_query(0, 0) == []
        assert t.range_query(10, 10) == [10]

    def test_inorder(self):
        t = Treap()
        vals = [5, 2, 8, 1, 3]
        for v in vals:
            t.insert(v)
        assert t.inorder() == sorted(vals)

    def test_iter(self):
        t = Treap()
        for i in [3, 1, 2]:
            t.insert(i)
        assert list(t) == [1, 2, 3]

    def test_split_merge(self):
        t = Treap()
        for i in range(1, 11):
            t.insert(i)
        left, right = t.split(6)
        assert left.inorder() == [1, 2, 3, 4, 5]
        assert right.inorder() == [6, 7, 8, 9, 10]
        merged = Treap.merge(left, right)
        assert merged.inorder() == list(range(1, 11))
        assert merged.verify()

    def test_large_random(self):
        random.seed(42)
        t = Treap()
        keys = random.sample(range(10000), 500)
        for k in keys:
            t.insert(k, k)
        assert len(t) == 500
        assert t.verify()
        for k in keys[:100]:
            assert t.search(k) == k
        for k in keys[:50]:
            assert t.delete(k)
        assert len(t) == 450
        assert t.verify()

    def test_sorted_insert(self):
        """Worst case for BST, but treap handles it via random priorities."""
        t = Treap()
        for i in range(100):
            t.insert(i)
        assert len(t) == 100
        assert t.inorder() == list(range(100))
        assert t.verify()

    def test_bool(self):
        t = Treap()
        assert not t
        t.insert(1)
        assert t

    def test_verify_valid(self):
        t = Treap()
        for i in [5, 3, 7, 1, 9]:
            t.insert(i)
        assert t.verify()


# ============================================================
# Variant 2: Implicit Treap
# ============================================================

class TestImplicitTreap:
    def test_empty(self):
        t = ImplicitTreap()
        assert len(t) == 0
        assert not t

    def test_append_and_access(self):
        t = ImplicitTreap()
        for i in range(5):
            t.append(i * 10)
        assert len(t) == 5
        assert t[0] == 0
        assert t[2] == 20
        assert t[4] == 40

    def test_init_with_values(self):
        t = ImplicitTreap([10, 20, 30])
        assert len(t) == 3
        assert t.to_list() == [10, 20, 30]

    def test_negative_index(self):
        t = ImplicitTreap([1, 2, 3])
        assert t[-1] == 3
        assert t[-2] == 2

    def test_setitem(self):
        t = ImplicitTreap([1, 2, 3])
        t[1] = 99
        assert t[1] == 99
        assert t.to_list() == [1, 99, 3]

    def test_setitem_negative(self):
        t = ImplicitTreap([1, 2, 3])
        t[-1] = 99
        assert t.to_list() == [1, 2, 99]

    def test_insert_at(self):
        t = ImplicitTreap([1, 2, 3])
        t.insert_at(1, 99)
        assert t.to_list() == [1, 99, 2, 3]
        t.insert_at(0, 0)
        assert t.to_list() == [0, 1, 99, 2, 3]
        t.insert_at(5, 100)
        assert t.to_list() == [0, 1, 99, 2, 3, 100]

    def test_insert_at_invalid(self):
        t = ImplicitTreap([1, 2, 3])
        with pytest.raises(IndexError):
            t.insert_at(-1, 0)
        with pytest.raises(IndexError):
            t.insert_at(4, 0)

    def test_delete_at(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        val = t.delete_at(2)
        assert val == 3
        assert t.to_list() == [1, 2, 4, 5]

    def test_delete_at_invalid(self):
        t = ImplicitTreap([1, 2, 3])
        with pytest.raises(IndexError):
            t.delete_at(-1)
        with pytest.raises(IndexError):
            t.delete_at(3)

    def test_reverse_range(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        t.reverse_range(1, 3)
        assert t.to_list() == [1, 4, 3, 2, 5]

    def test_reverse_full(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        t.reverse_range(0, 4)
        assert t.to_list() == [5, 4, 3, 2, 1]

    def test_reverse_single(self):
        t = ImplicitTreap([1, 2, 3])
        t.reverse_range(1, 1)
        assert t.to_list() == [1, 2, 3]

    def test_reverse_invalid(self):
        t = ImplicitTreap([1, 2, 3])
        with pytest.raises(IndexError):
            t.reverse_range(-1, 2)
        with pytest.raises(IndexError):
            t.reverse_range(0, 3)
        with pytest.raises(IndexError):
            t.reverse_range(2, 1)

    def test_double_reverse(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        t.reverse_range(0, 4)
        t.reverse_range(0, 4)
        assert t.to_list() == [1, 2, 3, 4, 5]

    def test_split_at(self):
        t = ImplicitTreap([1, 2, 3, 4, 5])
        left, right = t.split_at(3)
        assert left.to_list() == [1, 2, 3]
        assert right.to_list() == [4, 5]

    def test_merge_implicit(self):
        a = ImplicitTreap([1, 2, 3])
        b = ImplicitTreap([4, 5, 6])
        c = ImplicitTreap.merge(a, b)
        assert c.to_list() == [1, 2, 3, 4, 5, 6]

    def test_iter(self):
        t = ImplicitTreap([10, 20, 30])
        assert list(t) == [10, 20, 30]

    def test_large_operations(self):
        random.seed(42)
        t = ImplicitTreap(list(range(200)))
        # Do several random reverses
        for _ in range(50):
            a = random.randint(0, 198)
            b = random.randint(a, 199)
            t.reverse_range(a, b)
        assert len(t) == 200
        # All elements still present
        assert sorted(t.to_list()) == list(range(200))

    def test_getitem_out_of_range(self):
        t = ImplicitTreap([1, 2])
        with pytest.raises(IndexError):
            t[2]
        with pytest.raises(IndexError):
            t[-3]

    def test_setitem_out_of_range(self):
        t = ImplicitTreap([1, 2])
        with pytest.raises(IndexError):
            t[2] = 99
        with pytest.raises(IndexError):
            t[-3] = 99

    def test_bool(self):
        t = ImplicitTreap()
        assert not t
        t.append(1)
        assert t


# ============================================================
# Variant 3: Persistent Treap
# ============================================================

class TestPersistentTreap:
    def test_empty(self):
        t = PersistentTreap()
        assert len(t) == 0
        assert not t

    def test_insert_returns_new(self):
        t0 = PersistentTreap()
        t1 = t0.insert(5, "five")
        assert len(t0) == 0
        assert len(t1) == 1
        assert t1.search(5) == "five"

    def test_immutability(self):
        t0 = PersistentTreap()
        t1 = t0.insert(1)
        t2 = t1.insert(2)
        t3 = t2.insert(3)
        assert len(t0) == 0
        assert len(t1) == 1
        assert len(t2) == 2
        assert len(t3) == 3
        assert list(t3) == [1, 2, 3]

    def test_delete_returns_new(self):
        t0 = PersistentTreap()
        t1 = t0.insert(1).insert(2).insert(3)
        t2 = t1.delete(2)
        assert len(t1) == 3
        assert len(t2) == 2
        assert 2 in t1
        assert 2 not in t2

    def test_delete_nonexistent(self):
        t = PersistentTreap().insert(1)
        t2 = t.delete(99)
        assert t2 is t  # Same object returned

    def test_version_history(self):
        t = PersistentTreap()
        t = t.insert(1)
        t = t.insert(2)
        t = t.insert(3)
        assert t.version_count() == 3  # 3 previous versions (empty, {1}, {1,2})
        v0 = t.get_version(0)
        assert len(v0) == 0
        v1 = t.get_version(1)
        assert list(v1) == [1]

    def test_version_out_of_range(self):
        t = PersistentTreap()
        with pytest.raises(IndexError):
            t.get_version(0)

    def test_update_value(self):
        t = PersistentTreap().insert(5, "old")
        t2 = t.insert(5, "new")
        assert t.search(5) == "old"
        assert t2.search(5) == "new"

    def test_contains(self):
        t = PersistentTreap().insert(1).insert(2)
        assert 1 in t
        assert 3 not in t

    def test_large_persistent(self):
        random.seed(42)
        t = PersistentTreap()
        versions = [t]
        for i in range(50):
            t = t.insert(i)
            versions.append(t)
        assert len(versions[-1]) == 50
        assert len(versions[25]) == 25
        # Old versions untouched
        assert len(versions[0]) == 0

    def test_bool(self):
        t = PersistentTreap()
        assert not t
        t = t.insert(1)
        assert t


# ============================================================
# Variant 4: Mergeable Treap
# ============================================================

class TestMergeableTreap:
    def test_basic_ops(self):
        t = MergeableTreap()
        t.insert(5, "five")
        assert t.search(5) == "five"
        assert t.verify()

    def test_union(self):
        a = MergeableTreap()
        for i in [1, 3, 5, 7]:
            a.insert(i)
        b = MergeableTreap()
        for i in [2, 4, 6, 8]:
            b.insert(i)
        c = a.union(b)
        assert c.inorder() == [1, 2, 3, 4, 5, 6, 7, 8]
        assert c.verify()

    def test_union_overlapping(self):
        a = MergeableTreap()
        for i in [1, 2, 3, 4, 5]:
            a.insert(i)
        b = MergeableTreap()
        for i in [3, 4, 5, 6, 7]:
            b.insert(i)
        c = a.union(b)
        assert c.inorder() == [1, 2, 3, 4, 5, 6, 7]
        assert c.verify()

    def test_union_empty(self):
        a = MergeableTreap()
        a.insert(1)
        b = MergeableTreap()
        c = a.union(b)
        assert c.inorder() == [1]
        d = b.union(a)
        assert d.inorder() == [1]

    def test_intersection(self):
        a = MergeableTreap()
        for i in [1, 2, 3, 4, 5]:
            a.insert(i)
        b = MergeableTreap()
        for i in [3, 4, 5, 6, 7]:
            b.insert(i)
        c = a.intersection(b)
        assert c.inorder() == [3, 4, 5]
        assert c.verify()

    def test_intersection_disjoint(self):
        a = MergeableTreap()
        for i in [1, 2, 3]:
            a.insert(i)
        b = MergeableTreap()
        for i in [4, 5, 6]:
            b.insert(i)
        c = a.intersection(b)
        assert c.inorder() == []

    def test_difference(self):
        a = MergeableTreap()
        for i in [1, 2, 3, 4, 5]:
            a.insert(i)
        b = MergeableTreap()
        for i in [3, 4, 5, 6, 7]:
            b.insert(i)
        c = a.difference(b)
        assert c.inorder() == [1, 2]
        assert c.verify()

    def test_difference_all(self):
        a = MergeableTreap()
        for i in [1, 2, 3]:
            a.insert(i)
        b = MergeableTreap()
        for i in [1, 2, 3]:
            b.insert(i)
        c = a.difference(b)
        assert c.inorder() == []

    def test_set_operations_large(self):
        random.seed(42)
        a = MergeableTreap()
        b = MergeableTreap()
        set_a = set(random.sample(range(200), 100))
        set_b = set(random.sample(range(200), 100))
        for x in set_a:
            a.insert(x)
        for x in set_b:
            b.insert(x)
        assert a.union(b).inorder() == sorted(set_a | set_b)
        assert a.intersection(b).inorder() == sorted(set_a & set_b)
        assert a.difference(b).inorder() == sorted(set_a - set_b)

    def test_inherits_treap(self):
        """MergeableTreap has all Treap operations."""
        t = MergeableTreap()
        for i in [5, 3, 7, 1, 9]:
            t.insert(i)
        assert t.min() == 1
        assert t.max() == 9
        assert t.rank(5) == 2
        assert t.select(2) == 5


# ============================================================
# Variant 5: Interval Treap
# ============================================================

class TestIntervalTreap:
    def test_empty(self):
        t = IntervalTreap()
        assert len(t) == 0
        assert not t

    def test_insert_and_stab(self):
        t = IntervalTreap()
        t.insert(1, 5, "a")
        t.insert(3, 8, "b")
        t.insert(7, 10, "c")
        result = t.stab(4)
        labels = sorted(r[2] for r in result)
        assert labels == ["a", "b"]

    def test_stab_none(self):
        t = IntervalTreap()
        t.insert(1, 3)
        t.insert(5, 7)
        assert t.stab(4) == []

    def test_stab_boundary(self):
        t = IntervalTreap()
        t.insert(1, 5, "a")
        result = t.stab(1)
        assert len(result) == 1
        result = t.stab(5)
        assert len(result) == 1

    def test_overlap(self):
        t = IntervalTreap()
        t.insert(1, 5, "a")
        t.insert(3, 8, "b")
        t.insert(10, 15, "c")
        result = t.overlap(4, 6)
        labels = sorted(r[2] for r in result)
        assert labels == ["a", "b"]

    def test_overlap_all(self):
        t = IntervalTreap()
        t.insert(1, 10, "a")
        t.insert(2, 9, "b")
        t.insert(3, 8, "c")
        result = t.overlap(0, 100)
        assert len(result) == 3

    def test_overlap_none(self):
        t = IntervalTreap()
        t.insert(1, 3)
        t.insert(5, 7)
        assert t.overlap(8, 10) == []

    def test_delete_interval(self):
        t = IntervalTreap()
        t.insert(1, 5, "a")
        t.insert(3, 8, "b")
        assert len(t) == 2
        assert t.delete(1, 5)
        assert len(t) == 1
        result = t.stab(4)
        assert len(result) == 1
        assert result[0][2] == "b"

    def test_delete_nonexistent(self):
        t = IntervalTreap()
        t.insert(1, 5)
        assert not t.delete(1, 6)
        assert len(t) == 1

    def test_all_intervals(self):
        t = IntervalTreap()
        t.insert(5, 10, "c")
        t.insert(1, 3, "a")
        t.insert(3, 7, "b")
        intervals = t.all_intervals()
        # Sorted by lo
        assert [iv[0] for iv in intervals] == [1, 3, 5]

    def test_min_interval(self):
        t = IntervalTreap()
        t.insert(5, 10)
        t.insert(1, 3)
        t.insert(3, 7)
        lo, hi, _ = t.min_interval()
        assert lo == 1
        assert hi == 3

    def test_min_interval_empty(self):
        t = IntervalTreap()
        with pytest.raises(ValueError):
            t.min_interval()

    def test_enclosing(self):
        t = IntervalTreap()
        t.insert(1, 10, "big")
        t.insert(3, 7, "medium")
        t.insert(4, 5, "small")
        result = t.enclosing(4, 5)
        labels = sorted(r[2] for r in result)
        assert labels == ["big", "medium", "small"]

    def test_enclosing_none(self):
        t = IntervalTreap()
        t.insert(2, 4)
        t.insert(6, 8)
        assert t.enclosing(1, 9) == []

    def test_large_interval(self):
        random.seed(42)
        t = IntervalTreap()
        intervals = []
        for _ in range(200):
            a = random.randint(0, 1000)
            b = a + random.randint(1, 50)
            intervals.append((a, b))
            t.insert(a, b)
        assert len(t) == 200
        # Stab test: brute force verify
        point = 500
        expected = [(a, b) for a, b in intervals if a <= point <= b]
        result = t.stab(point)
        assert len(result) == len(expected)

    def test_bool(self):
        t = IntervalTreap()
        assert not t
        t.insert(1, 2)
        assert t

    def test_overlapping_intervals_same_lo(self):
        t = IntervalTreap()
        t.insert(1, 5, "a")
        t.insert(1, 10, "b")
        result = t.stab(3)
        assert len(result) == 2


# ============================================================
# Cross-variant and stress tests
# ============================================================

class TestStress:
    def test_treap_insert_delete_cycle(self):
        random.seed(123)
        t = Treap()
        keys = list(range(100))
        random.shuffle(keys)
        for k in keys:
            t.insert(k)
        assert t.verify()
        random.shuffle(keys)
        for k in keys:
            t.delete(k)
        assert len(t) == 0

    def test_implicit_treap_rope_like(self):
        """Use implicit treap as a rope: insert chars, reverse sections."""
        t = ImplicitTreap(list("hello world"))
        assert "".join(t.to_list()) == "hello world"
        t.reverse_range(0, 4)
        assert "".join(t.to_list()) == "olleh world"

    def test_persistent_fork(self):
        """Fork from a historical version and build differently."""
        t = PersistentTreap()
        t = t.insert(1)
        t = t.insert(2)
        # Fork from version with just {1}
        fork = t.get_version(1)
        fork = fork.insert(10)
        assert list(fork) == [1, 10]
        assert list(t) == [1, 2]

    def test_treap_deterministic_priority(self):
        """With controlled priorities, verify heap property."""
        t = Treap()
        n1 = TreapNode(5, priority=0.9)
        n2 = TreapNode(3, priority=0.5)
        n3 = TreapNode(7, priority=0.3)
        # Manual construction to test merge
        from treap import _merge
        root = _merge(_merge(n2, n1), n3)
        assert root.key == 5  # Highest priority
        assert root.priority == 0.9

    def test_treap_split_empty(self):
        t = Treap()
        left, right = t.split(5)
        assert len(left) == 0
        assert len(right) == 0

    def test_implicit_treap_empty_ops(self):
        a = ImplicitTreap()
        b = ImplicitTreap()
        c = ImplicitTreap.merge(a, b)
        assert len(c) == 0

    def test_mergeable_union_self(self):
        t = MergeableTreap()
        for i in [1, 2, 3]:
            t.insert(i)
        u = t.union(t)
        assert u.inorder() == [1, 2, 3]

    def test_interval_many_stabs(self):
        """Many intervals, many stab queries."""
        random.seed(77)
        t = IntervalTreap()
        intervals = []
        for _ in range(100):
            a = random.randint(0, 100)
            b = a + random.randint(1, 20)
            intervals.append((a, b))
            t.insert(a, b)
        for point in range(0, 120, 5):
            result = t.stab(point)
            expected = sum(1 for a, b in intervals if a <= point <= b)
            assert len(result) == expected

    def test_treap_rank_select_consistency(self):
        t = Treap()
        for i in [10, 20, 30, 40, 50]:
            t.insert(i)
        for i in range(5):
            key = t.select(i)
            assert t.rank(key) == i

    def test_treap_floor_ceiling_all(self):
        t = Treap()
        for i in range(0, 100, 10):
            t.insert(i)
        for i in range(100):
            f = t.floor(i)
            c = t.ceiling(i)
            assert f is None or f <= i
            assert c is None or c >= i

    def test_implicit_multiple_reverses(self):
        """Multiple overlapping reverses."""
        t = ImplicitTreap(list(range(10)))
        t.reverse_range(2, 7)  # [0,1,7,6,5,4,3,2,8,9]
        t.reverse_range(0, 4)  # [5,6,7,1,0,4,3,2,8,9]
        assert sorted(t.to_list()) == list(range(10))

    def test_persistent_delete_chain(self):
        t = PersistentTreap()
        for i in range(5):
            t = t.insert(i)
        t2 = t.delete(2).delete(3)
        assert list(t2) == [0, 1, 4]
        assert list(t) == [0, 1, 2, 3, 4]

    def test_mergeable_difference_empty_result(self):
        a = MergeableTreap()
        b = MergeableTreap()
        c = a.difference(b)
        assert c.inorder() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
