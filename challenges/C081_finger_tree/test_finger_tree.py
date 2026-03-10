"""
Tests for C081: Finger Tree
"""

import pytest
import random
from finger_tree import (
    EMPTY, Empty, Single, Deep, Elem, Node2, Node3,
    SizeMonoid, PriorityMonoid, MaxPriorityMonoid, KeyMonoid,
    cons, snoc, head, last, tail, init,
    concat, split, to_list, from_list, size,
    lookup, update, insert_at, delete_at, take, drop, slice_tree,
    fold_left, fold_right,
    pq_insert, pq_find_min, pq_delete_min,
    ordered_insert, ordered_search, ordered_delete, ordered_merge,
    FingerTreeSeq, FingerTreePQ, FingerTreeOrdSeq,
)


# ============================================================
# Basic construction tests
# ============================================================

class TestBasicConstruction:
    def test_empty(self):
        assert isinstance(EMPTY, Empty)
        assert EMPTY.is_empty()
        assert to_list(EMPTY) == []

    def test_singleton(self):
        t = snoc(EMPTY, 1, SizeMonoid)
        assert isinstance(t, Single)
        assert not t.is_empty()
        assert to_list(t) == [1]

    def test_two_elements(self):
        t = from_list([1, 2], SizeMonoid)
        assert to_list(t) == [1, 2]

    def test_three_elements(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(t) == [1, 2, 3]

    def test_four_elements(self):
        t = from_list([1, 2, 3, 4], SizeMonoid)
        assert to_list(t) == [1, 2, 3, 4]

    def test_five_elements(self):
        t = from_list([1, 2, 3, 4, 5], SizeMonoid)
        assert to_list(t) == [1, 2, 3, 4, 5]

    def test_many_elements(self):
        items = list(range(100))
        t = from_list(items, SizeMonoid)
        assert to_list(t) == items

    def test_large_tree(self):
        items = list(range(1000))
        t = from_list(items, SizeMonoid)
        assert to_list(t) == items

    def test_empty_singleton(self):
        """Empty is a singleton."""
        assert Empty() is Empty()
        assert EMPTY is Empty()


# ============================================================
# Cons (prepend) tests
# ============================================================

class TestCons:
    def test_cons_to_empty(self):
        t = cons(EMPTY, 42, SizeMonoid)
        assert to_list(t) == [42]

    def test_cons_to_single(self):
        t = cons(Single(Elem(2)), 1, SizeMonoid)
        assert to_list(t) == [1, 2]

    def test_cons_multiple(self):
        t = EMPTY
        for i in range(10, 0, -1):
            t = cons(t, i, SizeMonoid)
        assert to_list(t) == list(range(1, 11))

    def test_cons_triggers_overflow(self):
        """Cons when left digit is full (4 elements) pushes to spine."""
        t = from_list(list(range(5)), SizeMonoid)
        t = cons(t, -1, SizeMonoid)
        assert to_list(t) == [-1] + list(range(5))

    def test_cons_many(self):
        t = EMPTY
        for i in range(50, 0, -1):
            t = cons(t, i, SizeMonoid)
        assert to_list(t) == list(range(1, 51))


# ============================================================
# Snoc (append) tests
# ============================================================

class TestSnoc:
    def test_snoc_to_empty(self):
        t = snoc(EMPTY, 42, SizeMonoid)
        assert to_list(t) == [42]

    def test_snoc_to_single(self):
        t = snoc(Single(Elem(1)), 2, SizeMonoid)
        assert to_list(t) == [1, 2]

    def test_snoc_multiple(self):
        t = EMPTY
        for i in range(1, 11):
            t = snoc(t, i, SizeMonoid)
        assert to_list(t) == list(range(1, 11))

    def test_snoc_triggers_overflow(self):
        """Snoc when right digit is full (4 elements) pushes to spine."""
        t = from_list(list(range(5)), SizeMonoid)
        t = snoc(t, 99, SizeMonoid)
        assert to_list(t) == list(range(5)) + [99]


# ============================================================
# Head / Last tests
# ============================================================

class TestHeadLast:
    def test_head_single(self):
        assert head(Single(Elem(42))) == 42

    def test_head_deep(self):
        t = from_list([10, 20, 30], SizeMonoid)
        assert head(t) == 10

    def test_head_empty_raises(self):
        with pytest.raises(IndexError):
            head(EMPTY)

    def test_last_single(self):
        assert last(Single(Elem(42))) == 42

    def test_last_deep(self):
        t = from_list([10, 20, 30], SizeMonoid)
        assert last(t) == 30

    def test_last_empty_raises(self):
        with pytest.raises(IndexError):
            last(EMPTY)

    def test_head_large(self):
        t = from_list(list(range(100)), SizeMonoid)
        assert head(t) == 0

    def test_last_large(self):
        t = from_list(list(range(100)), SizeMonoid)
        assert last(t) == 99


# ============================================================
# Tail / Init tests
# ============================================================

class TestTailInit:
    def test_tail_single(self):
        t = tail(Single(Elem(42)), SizeMonoid)
        assert t.is_empty()

    def test_tail_two(self):
        t = from_list([1, 2], SizeMonoid)
        t2 = tail(t, SizeMonoid)
        assert to_list(t2) == [2]

    def test_tail_many(self):
        t = from_list(list(range(10)), SizeMonoid)
        t2 = tail(t, SizeMonoid)
        assert to_list(t2) == list(range(1, 10))

    def test_tail_empty_raises(self):
        with pytest.raises(IndexError):
            tail(EMPTY, SizeMonoid)

    def test_init_single(self):
        t = init(Single(Elem(42)), SizeMonoid)
        assert t.is_empty()

    def test_init_two(self):
        t = from_list([1, 2], SizeMonoid)
        t2 = init(t, SizeMonoid)
        assert to_list(t2) == [1]

    def test_init_many(self):
        t = from_list(list(range(10)), SizeMonoid)
        t2 = init(t, SizeMonoid)
        assert to_list(t2) == list(range(9))

    def test_init_empty_raises(self):
        with pytest.raises(IndexError):
            init(EMPTY, SizeMonoid)

    def test_repeated_tail(self):
        """Repeatedly taking tail drains the tree."""
        t = from_list(list(range(20)), SizeMonoid)
        for i in range(20):
            assert head(t) == i
            t = tail(t, SizeMonoid)
        assert t.is_empty()

    def test_repeated_init(self):
        """Repeatedly taking init drains the tree."""
        t = from_list(list(range(20)), SizeMonoid)
        for i in range(19, -1, -1):
            assert last(t) == i
            t = init(t, SizeMonoid)
        assert t.is_empty()

    def test_tail_borrows_from_middle(self):
        """Tail on a Deep with 1-element left digit borrows from middle."""
        t = from_list(list(range(10)), SizeMonoid)
        # Repeatedly tail until we trigger borrowing
        for _ in range(3):
            t = tail(t, SizeMonoid)
        assert to_list(t) == list(range(3, 10))


# ============================================================
# Concatenation tests
# ============================================================

class TestConcat:
    def test_concat_empty_empty(self):
        assert to_list(concat(EMPTY, EMPTY, SizeMonoid)) == []

    def test_concat_empty_nonempty(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(concat(EMPTY, t, SizeMonoid)) == [1, 2, 3]

    def test_concat_nonempty_empty(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(concat(t, EMPTY, SizeMonoid)) == [1, 2, 3]

    def test_concat_single_single(self):
        a = from_list([1], SizeMonoid)
        b = from_list([2], SizeMonoid)
        assert to_list(concat(a, b, SizeMonoid)) == [1, 2]

    def test_concat_small(self):
        a = from_list([1, 2, 3], SizeMonoid)
        b = from_list([4, 5, 6], SizeMonoid)
        assert to_list(concat(a, b, SizeMonoid)) == [1, 2, 3, 4, 5, 6]

    def test_concat_large(self):
        a = from_list(list(range(50)), SizeMonoid)
        b = from_list(list(range(50, 100)), SizeMonoid)
        assert to_list(concat(a, b, SizeMonoid)) == list(range(100))

    def test_concat_asymmetric(self):
        a = from_list([1], SizeMonoid)
        b = from_list(list(range(2, 20)), SizeMonoid)
        assert to_list(concat(a, b, SizeMonoid)) == [1] + list(range(2, 20))

    def test_concat_multiple(self):
        trees = [from_list(list(range(i*10, (i+1)*10)), SizeMonoid) for i in range(5)]
        result = trees[0]
        for t in trees[1:]:
            result = concat(result, t, SizeMonoid)
        assert to_list(result) == list(range(50))


# ============================================================
# Size / Measure tests
# ============================================================

class TestMeasure:
    def test_empty_measure(self):
        assert EMPTY.measure(SizeMonoid) == 0

    def test_single_measure(self):
        t = from_list([42], SizeMonoid)
        assert t.measure(SizeMonoid) == 1

    def test_deep_measure(self):
        t = from_list(list(range(10)), SizeMonoid)
        assert t.measure(SizeMonoid) == 10

    def test_measure_after_cons(self):
        t = from_list(list(range(10)), SizeMonoid)
        t = cons(t, -1, SizeMonoid)
        assert t.measure(SizeMonoid) == 11

    def test_measure_after_snoc(self):
        t = from_list(list(range(10)), SizeMonoid)
        t = snoc(t, 99, SizeMonoid)
        assert t.measure(SizeMonoid) == 11

    def test_measure_after_tail(self):
        t = from_list(list(range(10)), SizeMonoid)
        t = tail(t, SizeMonoid)
        assert t.measure(SizeMonoid) == 9

    def test_measure_after_concat(self):
        a = from_list(list(range(5)), SizeMonoid)
        b = from_list(list(range(5, 15)), SizeMonoid)
        c = concat(a, b, SizeMonoid)
        assert c.measure(SizeMonoid) == 15


# ============================================================
# Split tests
# ============================================================

class TestSplit:
    def test_split_single(self):
        t = from_list([42], SizeMonoid)
        l, x, r = split(t, lambda m: m > 0, SizeMonoid)
        assert to_list(l) == []
        assert x == 42
        assert to_list(r) == []

    def test_split_at_first(self):
        t = from_list([1, 2, 3, 4, 5], SizeMonoid)
        l, x, r = split(t, lambda m: m > 0, SizeMonoid)
        assert to_list(l) == []
        assert x == 1
        assert to_list(r) == [2, 3, 4, 5]

    def test_split_at_middle(self):
        t = from_list([1, 2, 3, 4, 5], SizeMonoid)
        l, x, r = split(t, lambda m: m > 2, SizeMonoid)
        assert to_list(l) == [1, 2]
        assert x == 3
        assert to_list(r) == [4, 5]

    def test_split_at_last(self):
        t = from_list([1, 2, 3, 4, 5], SizeMonoid)
        l, x, r = split(t, lambda m: m > 4, SizeMonoid)
        assert to_list(l) == [1, 2, 3, 4]
        assert x == 5
        assert to_list(r) == []

    def test_split_never_true_raises(self):
        t = from_list([1, 2, 3], SizeMonoid)
        with pytest.raises(ValueError):
            split(t, lambda m: False, SizeMonoid)

    def test_split_empty_raises(self):
        with pytest.raises(ValueError):
            split(EMPTY, lambda m: True, SizeMonoid)

    def test_split_large(self):
        items = list(range(100))
        t = from_list(items, SizeMonoid)
        for idx in [0, 25, 50, 75, 99]:
            l, x, r = split(t, lambda m, i=idx: m > i, SizeMonoid)
            assert to_list(l) == items[:idx]
            assert x == items[idx]
            assert to_list(r) == items[idx+1:]

    def test_split_preserves_tree(self):
        """Split is non-destructive (persistent)."""
        t = from_list([1, 2, 3, 4, 5], SizeMonoid)
        l1, x1, r1 = split(t, lambda m: m > 2, SizeMonoid)
        assert to_list(t) == [1, 2, 3, 4, 5]  # Original unchanged


# ============================================================
# Random access tests (lookup, update, insert, delete)
# ============================================================

class TestRandomAccess:
    def test_lookup_each(self):
        items = list(range(20))
        t = from_list(items, SizeMonoid)
        for i, v in enumerate(items):
            assert lookup(t, i) == v

    def test_lookup_out_of_range(self):
        t = from_list([1, 2, 3], SizeMonoid)
        with pytest.raises(IndexError):
            lookup(t, 3)
        with pytest.raises(IndexError):
            lookup(t, -1)

    def test_update(self):
        t = from_list([10, 20, 30, 40, 50], SizeMonoid)
        t2 = update(t, 2, 99)
        assert to_list(t2) == [10, 20, 99, 40, 50]
        assert to_list(t) == [10, 20, 30, 40, 50]  # Persistent

    def test_update_first(self):
        t = from_list([1, 2, 3], SizeMonoid)
        t2 = update(t, 0, 99)
        assert to_list(t2) == [99, 2, 3]

    def test_update_last(self):
        t = from_list([1, 2, 3], SizeMonoid)
        t2 = update(t, 2, 99)
        assert to_list(t2) == [1, 2, 99]

    def test_insert_at_beginning(self):
        t = from_list([1, 2, 3], SizeMonoid)
        t2 = insert_at(t, 0, 0)
        assert to_list(t2) == [0, 1, 2, 3]

    def test_insert_at_end(self):
        t = from_list([1, 2, 3], SizeMonoid)
        t2 = insert_at(t, 3, 4)
        assert to_list(t2) == [1, 2, 3, 4]

    def test_insert_at_middle(self):
        t = from_list([1, 2, 4, 5], SizeMonoid)
        t2 = insert_at(t, 2, 3)
        assert to_list(t2) == [1, 2, 3, 4, 5]

    def test_delete_at_beginning(self):
        t = from_list([1, 2, 3, 4], SizeMonoid)
        t2 = delete_at(t, 0)
        assert to_list(t2) == [2, 3, 4]

    def test_delete_at_end(self):
        t = from_list([1, 2, 3, 4], SizeMonoid)
        t2 = delete_at(t, 3)
        assert to_list(t2) == [1, 2, 3]

    def test_delete_at_middle(self):
        t = from_list([1, 2, 3, 4, 5], SizeMonoid)
        t2 = delete_at(t, 2)
        assert to_list(t2) == [1, 2, 4, 5]

    def test_insert_delete_roundtrip(self):
        t = from_list([1, 2, 3], SizeMonoid)
        t2 = insert_at(t, 1, 99)
        t3 = delete_at(t2, 1)
        assert to_list(t3) == [1, 2, 3]


# ============================================================
# Take / Drop / Slice tests
# ============================================================

class TestTakeDropSlice:
    def test_take_zero(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(take(t, 0)) == []

    def test_take_all(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(take(t, 3)) == [1, 2, 3]

    def test_take_some(self):
        t = from_list(list(range(10)), SizeMonoid)
        assert to_list(take(t, 5)) == [0, 1, 2, 3, 4]

    def test_take_more_than_size(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(take(t, 10)) == [1, 2, 3]

    def test_drop_zero(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(drop(t, 0)) == [1, 2, 3]

    def test_drop_all(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(drop(t, 3)) == []

    def test_drop_some(self):
        t = from_list(list(range(10)), SizeMonoid)
        assert to_list(drop(t, 5)) == [5, 6, 7, 8, 9]

    def test_drop_more_than_size(self):
        t = from_list([1, 2, 3], SizeMonoid)
        assert to_list(drop(t, 10)) == []

    def test_slice(self):
        t = from_list(list(range(20)), SizeMonoid)
        assert to_list(slice_tree(t, 5, 15)) == list(range(5, 15))

    def test_slice_beginning(self):
        t = from_list(list(range(10)), SizeMonoid)
        assert to_list(slice_tree(t, 0, 5)) == [0, 1, 2, 3, 4]

    def test_slice_end(self):
        t = from_list(list(range(10)), SizeMonoid)
        assert to_list(slice_tree(t, 5, 10)) == [5, 6, 7, 8, 9]

    def test_slice_empty(self):
        t = from_list(list(range(10)), SizeMonoid)
        assert to_list(slice_tree(t, 5, 5)) == []

    def test_take_drop_identity(self):
        """take(n) ++ drop(n) == original."""
        items = list(range(20))
        t = from_list(items, SizeMonoid)
        for n in [0, 1, 5, 10, 19, 20]:
            left = take(t, n, SizeMonoid)
            right = drop(t, n, SizeMonoid)
            combined = concat(left, right, SizeMonoid)
            assert to_list(combined) == items


# ============================================================
# Fold tests
# ============================================================

class TestFold:
    def test_fold_left_sum(self):
        t = from_list([1, 2, 3, 4, 5], SizeMonoid)
        assert fold_left(t, lambda acc, x: acc + x, 0) == 15

    def test_fold_right_list(self):
        t = from_list([1, 2, 3], SizeMonoid)
        result = fold_right(t, lambda x, acc: [x] + acc, [])
        assert result == [1, 2, 3]

    def test_fold_left_empty(self):
        assert fold_left(EMPTY, lambda acc, x: acc + x, 0) == 0

    def test_fold_right_empty(self):
        assert fold_right(EMPTY, lambda x, acc: [x] + acc, []) == []


# ============================================================
# Persistence tests
# ============================================================

class TestPersistence:
    def test_cons_preserves_original(self):
        t1 = from_list([1, 2, 3], SizeMonoid)
        t2 = cons(t1, 0, SizeMonoid)
        assert to_list(t1) == [1, 2, 3]
        assert to_list(t2) == [0, 1, 2, 3]

    def test_snoc_preserves_original(self):
        t1 = from_list([1, 2, 3], SizeMonoid)
        t2 = snoc(t1, 4, SizeMonoid)
        assert to_list(t1) == [1, 2, 3]
        assert to_list(t2) == [1, 2, 3, 4]

    def test_tail_preserves_original(self):
        t1 = from_list([1, 2, 3], SizeMonoid)
        t2 = tail(t1, SizeMonoid)
        assert to_list(t1) == [1, 2, 3]
        assert to_list(t2) == [2, 3]

    def test_concat_preserves_originals(self):
        a = from_list([1, 2, 3], SizeMonoid)
        b = from_list([4, 5, 6], SizeMonoid)
        c = concat(a, b, SizeMonoid)
        assert to_list(a) == [1, 2, 3]
        assert to_list(b) == [4, 5, 6]
        assert to_list(c) == [1, 2, 3, 4, 5, 6]

    def test_multiple_versions(self):
        """Create multiple versions from same base."""
        base = from_list([1, 2, 3], SizeMonoid)
        v1 = cons(base, 0, SizeMonoid)
        v2 = snoc(base, 4, SizeMonoid)
        v3 = tail(base, SizeMonoid)
        assert to_list(base) == [1, 2, 3]
        assert to_list(v1) == [0, 1, 2, 3]
        assert to_list(v2) == [1, 2, 3, 4]
        assert to_list(v3) == [2, 3]

    def test_deep_versioning(self):
        """Multiple operations on large tree preserve all versions."""
        t = from_list(list(range(50)), SizeMonoid)
        versions = [t]
        for i in range(10):
            t = snoc(t, 50 + i, SizeMonoid)
            versions.append(t)
        for i, v in enumerate(versions):
            assert len(to_list(v)) == 50 + i


# ============================================================
# Node tests
# ============================================================

class TestNodes:
    def test_node2_to_list(self):
        n = Node2(Elem(1), Elem(2))
        assert n.to_list() == [1, 2]

    def test_node3_to_list(self):
        n = Node3(Elem(1), Elem(2), Elem(3))
        assert n.to_list() == [1, 2, 3]

    def test_node2_measure(self):
        n = Node2(Elem(1), Elem(2))
        assert n.measure(SizeMonoid) == 2

    def test_node3_measure(self):
        n = Node3(Elem(1), Elem(2), Elem(3))
        assert n.measure(SizeMonoid) == 3

    def test_node2_to_digit(self):
        n = Node2(Elem(1), Elem(2))
        assert len(n.to_digit()) == 2

    def test_node3_to_digit(self):
        n = Node3(Elem(1), Elem(2), Elem(3))
        assert len(n.to_digit()) == 3


# ============================================================
# FingerTreeSeq (high-level API) tests
# ============================================================

class TestFingerTreeSeq:
    def test_empty_seq(self):
        s = FingerTreeSeq()
        assert len(s) == 0
        assert s.is_empty()
        assert s.to_list() == []

    def test_from_list(self):
        s = FingerTreeSeq.from_list([1, 2, 3])
        assert s.to_list() == [1, 2, 3]
        assert len(s) == 3

    def test_prepend(self):
        s = FingerTreeSeq.from_list([2, 3])
        s2 = s.prepend(1)
        assert s2.to_list() == [1, 2, 3]

    def test_append(self):
        s = FingerTreeSeq.from_list([1, 2])
        s2 = s.append(3)
        assert s2.to_list() == [1, 2, 3]

    def test_head_last(self):
        s = FingerTreeSeq.from_list([10, 20, 30])
        assert s.head() == 10
        assert s.last() == 30

    def test_tail_init(self):
        s = FingerTreeSeq.from_list([10, 20, 30])
        assert s.tail().to_list() == [20, 30]
        assert s.init().to_list() == [10, 20]

    def test_getitem(self):
        s = FingerTreeSeq.from_list([10, 20, 30, 40, 50])
        assert s[0] == 10
        assert s[2] == 30
        assert s[4] == 50

    def test_getitem_negative(self):
        s = FingerTreeSeq.from_list([10, 20, 30])
        assert s[-1] == 30
        assert s[-2] == 20

    def test_getitem_slice(self):
        s = FingerTreeSeq.from_list(list(range(10)))
        result = s[2:7]
        assert result.to_list() == [2, 3, 4, 5, 6]

    def test_concat(self):
        a = FingerTreeSeq.from_list([1, 2, 3])
        b = FingerTreeSeq.from_list([4, 5, 6])
        c = a.concat(b)
        assert c.to_list() == [1, 2, 3, 4, 5, 6]

    def test_insert(self):
        s = FingerTreeSeq.from_list([1, 2, 4])
        s2 = s.insert(2, 3)
        assert s2.to_list() == [1, 2, 3, 4]

    def test_delete(self):
        s = FingerTreeSeq.from_list([1, 2, 3, 4])
        s2 = s.delete(2)
        assert s2.to_list() == [1, 2, 4]

    def test_update(self):
        s = FingerTreeSeq.from_list([1, 2, 3])
        s2 = s.update(1, 99)
        assert s2.to_list() == [1, 99, 3]

    def test_take_drop(self):
        s = FingerTreeSeq.from_list(list(range(10)))
        assert s.take(5).to_list() == [0, 1, 2, 3, 4]
        assert s.drop(5).to_list() == [5, 6, 7, 8, 9]

    def test_split_at(self):
        s = FingerTreeSeq.from_list([1, 2, 3, 4, 5])
        left, right = s.split_at(3)
        assert left.to_list() == [1, 2, 3]
        assert right.to_list() == [4, 5]

    def test_reverse(self):
        s = FingerTreeSeq.from_list([1, 2, 3])
        assert s.reverse().to_list() == [3, 2, 1]

    def test_map(self):
        s = FingerTreeSeq.from_list([1, 2, 3])
        s2 = s.map(lambda x: x * 2)
        assert s2.to_list() == [2, 4, 6]

    def test_filter(self):
        s = FingerTreeSeq.from_list([1, 2, 3, 4, 5])
        s2 = s.filter(lambda x: x % 2 == 0)
        assert s2.to_list() == [2, 4]

    def test_fold_left(self):
        s = FingerTreeSeq.from_list([1, 2, 3, 4])
        assert s.fold_left(lambda a, b: a + b, 0) == 10

    def test_fold_right(self):
        s = FingerTreeSeq.from_list(['a', 'b', 'c'])
        assert s.fold_right(lambda x, acc: x + acc, '') == 'abc'

    def test_iter(self):
        s = FingerTreeSeq.from_list([1, 2, 3])
        assert list(s) == [1, 2, 3]

    def test_equality(self):
        a = FingerTreeSeq.from_list([1, 2, 3])
        b = FingerTreeSeq.from_list([1, 2, 3])
        assert a == b

    def test_inequality(self):
        a = FingerTreeSeq.from_list([1, 2, 3])
        b = FingerTreeSeq.from_list([1, 2, 4])
        assert a != b

    def test_repr(self):
        s = FingerTreeSeq.from_list([1, 2, 3])
        assert 'FingerTreeSeq' in repr(s)

    def test_large_seq(self):
        items = list(range(500))
        s = FingerTreeSeq.from_list(items)
        assert len(s) == 500
        assert s[0] == 0
        assert s[499] == 499
        assert s[250] == 250


# ============================================================
# Priority Queue tests
# ============================================================

class TestPriorityQueue:
    def test_empty_pq(self):
        pq = FingerTreePQ()
        assert pq.is_empty()

    def test_insert_and_find_min(self):
        pq = FingerTreePQ()
        pq = pq.insert(5, 'five')
        pq = pq.insert(3, 'three')
        pq = pq.insert(7, 'seven')
        assert pq.find_min() == (3, 'three')

    def test_delete_min(self):
        pq = FingerTreePQ()
        pq = pq.insert(5, 'five')
        pq = pq.insert(3, 'three')
        pq = pq.insert(7, 'seven')
        elem, pq2 = pq.delete_min()
        assert elem == (3, 'three')
        assert pq2.find_min() == (5, 'five')

    def test_sorted_extraction(self):
        pq = FingerTreePQ()
        values = [5, 3, 8, 1, 9, 2, 7, 4, 6]
        for v in values:
            pq = pq.insert(v, str(v))
        result = pq.to_sorted_list()
        priorities = [x[0] for x in result]
        assert priorities == sorted(values)

    def test_merge_pqs(self):
        pq1 = FingerTreePQ()
        pq2 = FingerTreePQ()
        for v in [5, 3, 1]:
            pq1 = pq1.insert(v, str(v))
        for v in [4, 2, 6]:
            pq2 = pq2.insert(v, str(v))
        merged = pq1.merge(pq2)
        result = merged.to_sorted_list()
        priorities = [x[0] for x in result]
        assert priorities == [1, 2, 3, 4, 5, 6]

    def test_duplicate_priorities(self):
        pq = FingerTreePQ()
        pq = pq.insert(3, 'a')
        pq = pq.insert(3, 'b')
        pq = pq.insert(1, 'c')
        elem, pq2 = pq.delete_min()
        assert elem == (1, 'c')

    def test_pq_len(self):
        pq = FingerTreePQ()
        for i in range(10):
            pq = pq.insert(i, str(i))
        assert len(pq) == 10

    def test_pq_persistent(self):
        pq1 = FingerTreePQ()
        pq1 = pq1.insert(5, 'five')
        pq1 = pq1.insert(3, 'three')
        _, pq2 = pq1.delete_min()
        assert pq1.find_min() == (3, 'three')
        assert pq2.find_min() == (5, 'five')


# ============================================================
# Ordered Sequence tests
# ============================================================

class TestOrderedSequence:
    def test_empty_ord(self):
        os = FingerTreeOrdSeq()
        assert os.is_empty()
        assert os.to_sorted_list() == []

    def test_insert_maintains_order(self):
        os = FingerTreeOrdSeq()
        for v in [5, 3, 8, 1, 4]:
            os = os.insert(v)
        assert os.to_sorted_list() == [1, 3, 4, 5, 8]

    def test_insert_with_values(self):
        os = FingerTreeOrdSeq()
        os = os.insert(3, 'three')
        os = os.insert(1, 'one')
        os = os.insert(5, 'five')
        result = os.to_sorted_list()
        assert result == [(1, 'one'), (3, 'three'), (5, 'five')]

    def test_search_found(self):
        os = FingerTreeOrdSeq()
        for v in [5, 3, 8, 1, 4]:
            os = os.insert(v)
        assert os.search(3) == 3
        assert os.search(8) == 8

    def test_search_not_found(self):
        os = FingerTreeOrdSeq()
        for v in [5, 3, 8]:
            os = os.insert(v)
        assert os.search(4) is None
        assert os.search(0) is None

    def test_delete(self):
        os = FingerTreeOrdSeq()
        for v in [5, 3, 8, 1, 4]:
            os = os.insert(v)
        os2 = os.delete(3)
        assert os2.to_sorted_list() == [1, 4, 5, 8]

    def test_delete_nonexistent(self):
        os = FingerTreeOrdSeq()
        for v in [1, 2, 3]:
            os = os.insert(v)
        os2 = os.delete(99)
        assert os2.to_sorted_list() == [1, 2, 3]

    def test_min_max(self):
        os = FingerTreeOrdSeq()
        for v in [5, 3, 8, 1, 4]:
            os = os.insert(v)
        assert os.min() == 1
        assert os.max() == 8

    def test_merge(self):
        os1 = FingerTreeOrdSeq()
        os2 = FingerTreeOrdSeq()
        for v in [1, 3, 5]:
            os1 = os1.insert(v)
        for v in [2, 4, 6]:
            os2 = os2.insert(v)
        merged = os1.merge(os2)
        assert merged.to_sorted_list() == [1, 2, 3, 4, 5, 6]

    def test_range_query(self):
        os = FingerTreeOrdSeq()
        for v in range(1, 11):
            os = os.insert(v)
        result = os.range_query(3, 7)
        assert result == [3, 4, 5, 6, 7]

    def test_range_query_empty(self):
        os = FingerTreeOrdSeq()
        for v in [1, 5, 10]:
            os = os.insert(v)
        assert os.range_query(6, 9) == []

    def test_ord_iter(self):
        os = FingerTreeOrdSeq()
        for v in [3, 1, 2]:
            os = os.insert(v)
        assert list(os) == [1, 2, 3]

    def test_ord_len(self):
        os = FingerTreeOrdSeq()
        for v in range(10):
            os = os.insert(v)
        assert len(os) == 10

    def test_ord_persistent(self):
        os1 = FingerTreeOrdSeq()
        for v in [1, 2, 3]:
            os1 = os1.insert(v)
        os2 = os1.insert(4)
        os3 = os1.delete(2)
        assert os1.to_sorted_list() == [1, 2, 3]
        assert os2.to_sorted_list() == [1, 2, 3, 4]
        assert os3.to_sorted_list() == [1, 3]


# ============================================================
# MaxPriorityMonoid tests
# ============================================================

class TestMaxPriority:
    def test_max_priority_queue(self):
        pq = FingerTreePQ(monoid=MaxPriorityMonoid)
        for v in [5, 3, 8, 1, 9]:
            pq = pq.insert(v, str(v))
        assert pq.find_min() == (9, '9')  # "min" with MaxPriority is max

    def test_max_extraction(self):
        pq = FingerTreePQ(monoid=MaxPriorityMonoid)
        for v in [5, 3, 8, 1, 9]:
            pq = pq.insert(v, str(v))
        result = pq.to_sorted_list()
        priorities = [x[0] for x in result]
        assert priorities == [9, 8, 5, 3, 1]


# ============================================================
# Stress / randomized tests
# ============================================================

class TestStress:
    def test_random_cons_snoc(self):
        """Randomized cons/snoc operations."""
        random.seed(42)
        t = EMPTY
        expected = []
        for _ in range(200):
            v = random.randint(0, 999)
            if random.random() < 0.5:
                t = cons(t, v, SizeMonoid)
                expected.insert(0, v)
            else:
                t = snoc(t, v, SizeMonoid)
                expected.append(v)
        assert to_list(t) == expected

    def test_random_operations(self):
        """Random mix of cons, snoc, head, last, tail, init."""
        random.seed(123)
        t = EMPTY
        expected = []
        for _ in range(300):
            op = random.choice(['cons', 'snoc', 'tail', 'init'])
            if op == 'cons':
                v = random.randint(0, 999)
                t = cons(t, v, SizeMonoid)
                expected.insert(0, v)
            elif op == 'snoc':
                v = random.randint(0, 999)
                t = snoc(t, v, SizeMonoid)
                expected.append(v)
            elif op == 'tail' and expected:
                t = tail(t, SizeMonoid)
                expected.pop(0)
            elif op == 'init' and expected:
                t = init(t, SizeMonoid)
                expected.pop()
        assert to_list(t) == expected

    def test_random_concat(self):
        """Random concatenations."""
        random.seed(456)
        trees = []
        for _ in range(10):
            items = [random.randint(0, 99) for _ in range(random.randint(5, 30))]
            trees.append((from_list(items, SizeMonoid), items))

        while len(trees) > 1:
            i = random.randint(0, len(trees) - 2)
            t1, l1 = trees[i]
            t2, l2 = trees[i + 1]
            trees[i] = (concat(t1, t2, SizeMonoid), l1 + l2)
            trees.pop(i + 1)

        assert to_list(trees[0][0]) == trees[0][1]

    def test_random_split(self):
        """Random splits verify l ++ [x] ++ r == original."""
        random.seed(789)
        items = list(range(50))
        t = from_list(items, SizeMonoid)
        for _ in range(20):
            idx = random.randint(0, 49)
            l, x, r = split(t, lambda m, i=idx: m > i, SizeMonoid)
            reconstructed = to_list(l) + [x] + to_list(r)
            assert reconstructed == items

    def test_random_lookup(self):
        """Random lookups on a large tree."""
        random.seed(101)
        items = list(range(200))
        t = from_list(items, SizeMonoid)
        for _ in range(100):
            idx = random.randint(0, 199)
            assert lookup(t, idx) == items[idx]


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:
    def test_single_element_operations(self):
        t = from_list([42], SizeMonoid)
        assert head(t) == 42
        assert last(t) == 42
        assert tail(t, SizeMonoid).is_empty()
        assert init(t, SizeMonoid).is_empty()

    def test_two_element_symmetry(self):
        t = from_list([1, 2], SizeMonoid)
        assert head(t) == 1
        assert last(t) == 2
        assert to_list(tail(t, SizeMonoid)) == [2]
        assert to_list(init(t, SizeMonoid)) == [1]

    def test_concat_single_elements(self):
        a = from_list([1], SizeMonoid)
        b = from_list([2], SizeMonoid)
        assert to_list(concat(a, b, SizeMonoid)) == [1, 2]

    def test_split_two_elements(self):
        t = from_list([1, 2], SizeMonoid)
        l, x, r = split(t, lambda m: m > 0, SizeMonoid)
        assert x == 1
        assert to_list(r) == [2]

    def test_string_elements(self):
        t = from_list(['hello', 'world', 'foo'], SizeMonoid)
        assert to_list(t) == ['hello', 'world', 'foo']
        assert head(t) == 'hello'
        assert last(t) == 'foo'

    def test_mixed_type_elements(self):
        t = from_list([1, 'two', 3.0, None, (4, 5)], SizeMonoid)
        assert to_list(t) == [1, 'two', 3.0, None, (4, 5)]

    def test_nested_tuples(self):
        t = from_list([(1, 'a'), (2, 'b'), (3, 'c')], SizeMonoid)
        assert lookup(t, 1) == (2, 'b')

    def test_empty_concat_chains(self):
        t = EMPTY
        for _ in range(10):
            t = concat(t, EMPTY, SizeMonoid)
        assert t.is_empty()

    def test_alternating_cons_tail(self):
        """Alternating cons/tail stays at size 1."""
        t = from_list([1], SizeMonoid)
        for i in range(100):
            t = cons(t, i, SizeMonoid)
            t = tail(t, SizeMonoid)
        assert to_list(t) == [1]


# ============================================================
# Deque behavior tests (double-ended queue via finger tree)
# ============================================================

class TestDeque:
    def test_push_front_pop_front(self):
        """Stack-like LIFO behavior from front."""
        t = EMPTY
        for i in range(10):
            t = cons(t, i, SizeMonoid)
        for i in range(9, -1, -1):
            assert head(t) == i
            t = tail(t, SizeMonoid)
        assert t.is_empty()

    def test_push_back_pop_back(self):
        """Stack-like LIFO behavior from back."""
        t = EMPTY
        for i in range(10):
            t = snoc(t, i, SizeMonoid)
        for i in range(9, -1, -1):
            assert last(t) == i
            t = init(t, SizeMonoid)
        assert t.is_empty()

    def test_push_front_pop_back(self):
        """Queue-like FIFO behavior."""
        t = EMPTY
        for i in range(10):
            t = cons(t, i, SizeMonoid)
        for i in range(10):
            assert last(t) == i
            t = init(t, SizeMonoid)
        assert t.is_empty()

    def test_push_back_pop_front(self):
        """Queue-like FIFO behavior (other direction)."""
        t = EMPTY
        for i in range(10):
            t = snoc(t, i, SizeMonoid)
        for i in range(10):
            assert head(t) == i
            t = tail(t, SizeMonoid)
        assert t.is_empty()
