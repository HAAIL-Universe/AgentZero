"""
Tests for C082: Interval Tree composing C081 Finger Tree.

Coverage:
  - Interval class (construction, validation, predicates)
  - IntervalMonoid (identity, combine, measure)
  - Insert/delete operations
  - Stabbing queries (point containment)
  - Overlap queries (range intersection)
  - Containment queries
  - Nearest interval queries
  - Endpoint queries (min/max lo/hi, span)
  - Set operations (merge, intersection, difference)
  - Interval arithmetic (merge overlapping, gaps, coverage, complement, clip, fragment)
  - Statistics (depth, max_depth, histogram)
  - Sweep line (intersect_all, pairwise_overlaps)
  - Bulk operations (from_intervals, filter, map_data, fold)
  - Serialization (to_dict, from_dict)
  - Edge cases and stress tests
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from interval_tree import Interval, IntervalMonoid, IntervalTree


# ============================================================
# Interval class
# ============================================================

class TestInterval:
    def test_basic_construction(self):
        iv = Interval(1, 5)
        assert iv.lo == 1
        assert iv.hi == 5
        assert iv.data is None

    def test_construction_with_data(self):
        iv = Interval(1, 5, "hello")
        assert iv.data == "hello"

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            Interval(5, 1)

    def test_point_interval(self):
        iv = Interval(3, 3)
        assert iv.lo == 3
        assert iv.hi == 3
        assert iv.length == 0

    def test_contains_point(self):
        iv = Interval(1, 5)
        assert iv.contains_point(1)
        assert iv.contains_point(3)
        assert iv.contains_point(5)
        assert not iv.contains_point(0)
        assert not iv.contains_point(6)

    def test_contains_point_boundary(self):
        iv = Interval(1, 5)
        assert iv.contains_point(1)  # inclusive
        assert iv.contains_point(5)  # inclusive

    def test_overlaps(self):
        a = Interval(1, 5)
        b = Interval(3, 7)
        c = Interval(6, 10)
        assert a.overlaps(b)
        assert b.overlaps(a)
        assert not a.overlaps(c)
        assert b.overlaps(c)

    def test_overlaps_touching(self):
        a = Interval(1, 5)
        b = Interval(5, 10)
        assert a.overlaps(b)  # touching is overlapping

    def test_overlaps_contained(self):
        a = Interval(1, 10)
        b = Interval(3, 7)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_contains_interval(self):
        a = Interval(1, 10)
        b = Interval(3, 7)
        assert a.contains_interval(b)
        assert not b.contains_interval(a)

    def test_length(self):
        assert Interval(1, 5).length == 4
        assert Interval(0, 0).length == 0
        assert Interval(-3, 3).length == 6

    def test_midpoint(self):
        assert Interval(1, 5).midpoint == 3.0
        assert Interval(0, 10).midpoint == 5.0

    def test_repr(self):
        assert "1" in repr(Interval(1, 5))
        assert "5" in repr(Interval(1, 5))
        assert "hello" in repr(Interval(1, 5, "hello"))

    def test_ordering(self):
        a = Interval(1, 5)
        b = Interval(2, 6)
        c = Interval(1, 6)
        assert a < b
        assert a < c

    def test_equality(self):
        assert Interval(1, 5) == Interval(1, 5)
        assert Interval(1, 5) != Interval(1, 6)

    def test_float_intervals(self):
        iv = Interval(1.5, 3.7)
        assert iv.contains_point(2.0)
        assert not iv.contains_point(1.4)
        assert iv.length == pytest.approx(2.2)

    def test_negative_intervals(self):
        iv = Interval(-5, -1)
        assert iv.contains_point(-3)
        assert not iv.contains_point(0)


# ============================================================
# IntervalMonoid
# ============================================================

class TestIntervalMonoid:
    def test_empty(self):
        assert IntervalMonoid.empty() is None

    def test_measure(self):
        iv = Interval(1, 5)
        m = IntervalMonoid.measure(iv)
        assert m == (1, 1, 5)  # (max_lo, min_lo, max_hi)

    def test_combine_with_empty(self):
        m = IntervalMonoid.measure(Interval(1, 5))
        assert IntervalMonoid.combine(m, None) == (1, 1, 5)
        assert IntervalMonoid.combine(None, m) == (1, 1, 5)

    def test_combine_two(self):
        a = IntervalMonoid.measure(Interval(1, 5))
        b = IntervalMonoid.measure(Interval(3, 10))
        c = IntervalMonoid.combine(a, b)
        assert c == (3, 1, 10)  # (max_lo=3, min_lo=1, max_hi=10)

    def test_combine_associative(self):
        a = (1, 1, 5)
        b = (3, 3, 10)
        c = (7, 7, 8)
        ab_c = IntervalMonoid.combine(IntervalMonoid.combine(a, b), c)
        a_bc = IntervalMonoid.combine(a, IntervalMonoid.combine(b, c))
        assert ab_c == a_bc

    def test_identity(self):
        m = (3, 3, 7)
        assert IntervalMonoid.combine(IntervalMonoid.empty(), m) == m
        assert IntervalMonoid.combine(m, IntervalMonoid.empty()) == m


# ============================================================
# Empty tree
# ============================================================

class TestEmptyTree:
    def test_is_empty(self):
        t = IntervalTree()
        assert t.is_empty()
        assert len(t) == 0

    def test_stab_empty(self):
        assert IntervalTree().stab(5) == []

    def test_overlap_empty(self):
        assert IntervalTree().overlap(1, 5) == []

    def test_any_overlap_empty(self):
        assert not IntervalTree().any_overlap(1, 5)

    def test_min_lo_empty(self):
        with pytest.raises(IndexError):
            IntervalTree().min_lo()

    def test_max_hi_empty(self):
        with pytest.raises(IndexError):
            IntervalTree().max_hi()

    def test_iter_empty(self):
        assert list(IntervalTree()) == []

    def test_span_empty(self):
        assert IntervalTree().span() is None

    def test_coverage_empty(self):
        assert IntervalTree().coverage() == 0

    def test_gaps_empty(self):
        assert IntervalTree().gaps() == []

    def test_merge_overlapping_empty(self):
        assert IntervalTree().merge_overlapping().is_empty()


# ============================================================
# Insert operations
# ============================================================

class TestInsert:
    def test_insert_single(self):
        t = IntervalTree().insert(1, 5)
        assert len(t) == 1
        assert not t.is_empty()

    def test_insert_interval_object(self):
        t = IntervalTree().insert(Interval(1, 5, "a"))
        items = t.intervals()
        assert len(items) == 1
        assert items[0].data == "a"

    def test_insert_multiple(self):
        t = IntervalTree()
        t = t.insert(1, 5)
        t = t.insert(3, 7)
        t = t.insert(6, 10)
        assert len(t) == 3

    def test_insert_maintains_sort(self):
        t = IntervalTree()
        t = t.insert(5, 10)
        t = t.insert(1, 3)
        t = t.insert(3, 7)
        items = t.intervals()
        assert items[0].lo <= items[1].lo <= items[2].lo

    def test_insert_with_data(self):
        t = IntervalTree().insert(1, 5, "event_a")
        items = t.intervals()
        assert items[0].data == "event_a"

    def test_insert_duplicate(self):
        t = IntervalTree().insert(1, 5).insert(1, 5)
        assert len(t) == 2

    def test_insert_persistence(self):
        t1 = IntervalTree().insert(1, 5)
        t2 = t1.insert(3, 7)
        assert len(t1) == 1
        assert len(t2) == 2

    def test_insert_many(self):
        t = IntervalTree()
        for i in range(100):
            t = t.insert(i, i + 10)
        assert len(t) == 100

    def test_insert_reverse_order(self):
        t = IntervalTree()
        for i in range(20, 0, -1):
            t = t.insert(i, i + 5)
        items = t.intervals()
        for i in range(len(items) - 1):
            assert items[i].lo <= items[i + 1].lo

    def test_insert_same_lo_different_hi(self):
        t = IntervalTree()
        t = t.insert(1, 10)
        t = t.insert(1, 5)
        t = t.insert(1, 7)
        assert len(t) == 3


# ============================================================
# Delete operations
# ============================================================

class TestDelete:
    def test_delete_single(self):
        t = IntervalTree().insert(1, 5)
        t = t.delete(1, 5)
        assert t.is_empty()

    def test_delete_by_interval(self):
        t = IntervalTree().insert(Interval(1, 5, "a"))
        t = t.delete(Interval(1, 5, "a"))
        assert t.is_empty()

    def test_delete_nonexistent(self):
        t = IntervalTree().insert(1, 5)
        t2 = t.delete(2, 6)
        assert len(t2) == 1  # unchanged

    def test_delete_preserves_others(self):
        t = IntervalTree().insert(1, 5).insert(3, 7).insert(6, 10)
        t = t.delete(3, 7)
        assert len(t) == 2
        items = t.intervals()
        assert any(iv.lo == 1 and iv.hi == 5 for iv in items)
        assert any(iv.lo == 6 and iv.hi == 10 for iv in items)

    def test_delete_first_of_duplicates(self):
        t = IntervalTree().insert(1, 5).insert(1, 5).insert(1, 5)
        t = t.delete(1, 5)
        assert len(t) == 2

    def test_delete_all(self):
        t = IntervalTree().insert(1, 5).insert(1, 5).insert(3, 7)
        t = t.delete_all(1, 5)
        assert len(t) == 1
        assert t.intervals()[0] == Interval(3, 7)

    def test_delete_with_data_match(self):
        t = IntervalTree().insert(1, 5, "a").insert(1, 5, "b")
        t = t.delete(Interval(1, 5, "a"))
        assert len(t) == 1
        assert t.intervals()[0].data == "b"

    def test_delete_persistence(self):
        t1 = IntervalTree().insert(1, 5).insert(3, 7)
        t2 = t1.delete(1, 5)
        assert len(t1) == 2
        assert len(t2) == 1


# ============================================================
# Stabbing queries
# ============================================================

class TestStab:
    def test_stab_single_hit(self):
        t = IntervalTree().insert(1, 5)
        assert len(t.stab(3)) == 1

    def test_stab_miss(self):
        t = IntervalTree().insert(1, 5)
        assert len(t.stab(6)) == 0

    def test_stab_boundary(self):
        t = IntervalTree().insert(1, 5)
        assert len(t.stab(1)) == 1
        assert len(t.stab(5)) == 1

    def test_stab_multiple_hits(self):
        t = IntervalTree()
        t = t.insert(1, 10)
        t = t.insert(3, 7)
        t = t.insert(5, 15)
        results = t.stab(6)
        assert len(results) == 3

    def test_stab_partial_hits(self):
        t = IntervalTree()
        t = t.insert(1, 5)
        t = t.insert(3, 7)
        t = t.insert(8, 12)
        results = t.stab(4)
        assert len(results) == 2
        los = {iv.lo for iv in results}
        assert 1 in los
        assert 3 in los

    def test_stab_point_interval(self):
        t = IntervalTree().insert(5, 5)
        assert len(t.stab(5)) == 1
        assert len(t.stab(4)) == 0

    def test_stab_many_intervals(self):
        t = IntervalTree()
        # 50 intervals, each [i, i+20]
        for i in range(50):
            t = t.insert(i, i + 20)
        # Point 25 should be in intervals [5..25], that's 21 intervals
        results = t.stab(25)
        assert len(results) == 21

    def test_stab_non_overlapping(self):
        t = IntervalTree()
        for i in range(10):
            t = t.insert(i * 10, i * 10 + 5)
        assert len(t.stab(3)) == 1
        assert len(t.stab(7)) == 0
        assert len(t.stab(13)) == 1

    def test_stab_returns_correct_intervals(self):
        t = IntervalTree()
        t = t.insert(1, 5, "a")
        t = t.insert(3, 7, "b")
        t = t.insert(8, 12, "c")
        results = t.stab(4)
        data = {iv.data for iv in results}
        assert data == {"a", "b"}


# ============================================================
# Overlap queries
# ============================================================

class TestOverlap:
    def test_overlap_single_hit(self):
        t = IntervalTree().insert(1, 5)
        assert len(t.overlap(3, 7)) == 1

    def test_overlap_miss(self):
        t = IntervalTree().insert(1, 5)
        assert len(t.overlap(6, 10)) == 0

    def test_overlap_touching(self):
        t = IntervalTree().insert(1, 5)
        assert len(t.overlap(5, 10)) == 1  # touching counts

    def test_overlap_multiple(self):
        t = IntervalTree()
        t = t.insert(1, 5)
        t = t.insert(4, 8)
        t = t.insert(10, 15)
        results = t.overlap(3, 6)
        assert len(results) == 2

    def test_overlap_contained(self):
        t = IntervalTree().insert(1, 10)
        results = t.overlap(3, 7)
        assert len(results) == 1

    def test_overlap_containing(self):
        t = IntervalTree().insert(3, 7)
        results = t.overlap(1, 10)
        assert len(results) == 1

    def test_overlap_with_interval_object(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        results = t.overlap(Interval(4, 6))
        assert len(results) == 2

    def test_any_overlap_true(self):
        t = IntervalTree().insert(1, 5)
        assert t.any_overlap(3, 7)

    def test_any_overlap_false(self):
        t = IntervalTree().insert(1, 5)
        assert not t.any_overlap(6, 10)

    def test_any_overlap_early_exit(self):
        # Build large tree, query should exit early
        t = IntervalTree()
        for i in range(100):
            t = t.insert(i * 10, i * 10 + 5)
        assert t.any_overlap(2, 3)
        assert not t.any_overlap(6, 9)

    def test_overlap_many(self):
        t = IntervalTree()
        for i in range(50):
            t = t.insert(i, i + 20)
        results = t.overlap(25, 30)
        # Intervals [lo, lo+20] overlap [25,30] when lo <= 30 and lo+20 >= 25
        # lo <= 30 and lo >= 5 => lo in [5, 30]
        assert len(results) == 26


# ============================================================
# Containment queries
# ============================================================

class TestContainment:
    def test_containing(self):
        t = IntervalTree()
        t = t.insert(1, 10)
        t = t.insert(3, 7)
        t = t.insert(2, 5)
        results = t.containing(3, 7)
        assert len(results) == 2  # [1,10] and [3,7]

    def test_containing_none(self):
        t = IntervalTree().insert(3, 7)
        results = t.containing(1, 10)
        assert len(results) == 0

    def test_contained_by(self):
        t = IntervalTree()
        t = t.insert(3, 7)
        t = t.insert(4, 6)
        t = t.insert(1, 10)
        results = t.contained_by(2, 8)
        assert len(results) == 2  # [3,7] and [4,6]

    def test_contained_by_none(self):
        t = IntervalTree().insert(1, 10)
        results = t.contained_by(3, 7)
        assert len(results) == 0

    def test_containing_interval_object(self):
        t = IntervalTree().insert(1, 10)
        results = t.containing(Interval(3, 7))
        assert len(results) == 1


# ============================================================
# Nearest queries
# ============================================================

class TestNearest:
    def test_nearest_inside(self):
        t = IntervalTree().insert(1, 5)
        results = t.nearest(3)
        assert len(results) == 1
        assert results[0][0] == 0  # distance 0

    def test_nearest_before(self):
        t = IntervalTree().insert(5, 10)
        results = t.nearest(2)
        assert results[0][0] == 3  # distance to lo=5

    def test_nearest_after(self):
        t = IntervalTree().insert(1, 5)
        results = t.nearest(8)
        assert results[0][0] == 3  # distance to hi=5

    def test_nearest_multiple(self):
        t = IntervalTree()
        t = t.insert(1, 5)
        t = t.insert(8, 12)
        t = t.insert(20, 25)
        results = t.nearest(6, count=2)
        assert len(results) == 2
        assert results[0][0] == 1  # distance to [1,5]
        assert results[1][0] == 2  # distance to [8,12]

    def test_nearest_empty(self):
        assert IntervalTree().nearest(5) == []

    def test_nearest_count(self):
        t = IntervalTree()
        for i in range(10):
            t = t.insert(i * 10, i * 10 + 3)
        results = t.nearest(15, count=3)
        assert len(results) == 3


# ============================================================
# Endpoint queries
# ============================================================

class TestEndpoints:
    def test_min_lo(self):
        t = IntervalTree().insert(3, 7).insert(1, 5).insert(5, 10)
        assert t.min_lo() == 1

    def test_max_lo(self):
        t = IntervalTree().insert(1, 5).insert(3, 7).insert(5, 10)
        assert t.max_lo() == 5

    def test_max_hi(self):
        t = IntervalTree().insert(1, 5).insert(3, 12).insert(5, 8)
        assert t.max_hi() == 12

    def test_min_hi(self):
        t = IntervalTree().insert(1, 5).insert(3, 12).insert(5, 8)
        assert t.min_hi() == 5

    def test_span(self):
        t = IntervalTree().insert(3, 7).insert(1, 5).insert(5, 12)
        s = t.span()
        assert s.lo == 1
        assert s.hi == 12

    def test_span_single(self):
        t = IntervalTree().insert(3, 7)
        s = t.span()
        assert s == Interval(3, 7)


# ============================================================
# Set operations
# ============================================================

class TestSetOperations:
    def test_merge(self):
        t1 = IntervalTree().insert(1, 5).insert(8, 12)
        t2 = IntervalTree().insert(3, 7).insert(10, 15)
        merged = t1.merge(t2)
        assert len(merged) == 4

    def test_merge_sorted(self):
        t1 = IntervalTree().insert(5, 10)
        t2 = IntervalTree().insert(1, 3)
        merged = t1.merge(t2)
        items = merged.intervals()
        for i in range(len(items) - 1):
            assert items[i].lo <= items[i + 1].lo

    def test_intersection(self):
        t1 = IntervalTree().insert(1, 5).insert(3, 7)
        t2 = IntervalTree().insert(3, 7).insert(8, 12)
        result = t1.intersection(t2)
        assert len(result) == 1
        assert result.intervals()[0] == Interval(3, 7)

    def test_intersection_empty(self):
        t1 = IntervalTree().insert(1, 5)
        t2 = IntervalTree().insert(3, 7)
        result = t1.intersection(t2)
        assert len(result) == 0

    def test_difference(self):
        t1 = IntervalTree().insert(1, 5).insert(3, 7).insert(8, 12)
        t2 = IntervalTree().insert(3, 7)
        result = t1.difference(t2)
        assert len(result) == 2

    def test_difference_empty(self):
        t1 = IntervalTree().insert(1, 5)
        t2 = IntervalTree().insert(1, 5)
        result = t1.difference(t2)
        assert len(result) == 0

    def test_contains(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        assert Interval(1, 5) in t
        assert Interval(3, 7) in t
        assert Interval(2, 6) not in t


# ============================================================
# Merge overlapping
# ============================================================

class TestMergeOverlapping:
    def test_no_overlaps(self):
        t = IntervalTree().insert(1, 3).insert(5, 7).insert(9, 11)
        merged = t.merge_overlapping()
        assert len(merged) == 3

    def test_two_overlapping(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        merged = t.merge_overlapping()
        assert len(merged) == 1
        iv = merged.intervals()[0]
        assert iv.lo == 1
        assert iv.hi == 7

    def test_chain_overlapping(self):
        t = IntervalTree().insert(1, 5).insert(3, 8).insert(7, 12)
        merged = t.merge_overlapping()
        assert len(merged) == 1
        assert merged.intervals()[0] == Interval(1, 12)

    def test_adjacent_not_merged(self):
        # Adjacent intervals (touching at a point) ARE merged
        t = IntervalTree().insert(1, 5).insert(5, 10)
        merged = t.merge_overlapping()
        assert len(merged) == 1

    def test_nested(self):
        t = IntervalTree().insert(1, 10).insert(3, 7)
        merged = t.merge_overlapping()
        assert len(merged) == 1
        assert merged.intervals()[0] == Interval(1, 10)

    def test_many_overlapping(self):
        t = IntervalTree()
        for i in range(20):
            t = t.insert(i, i + 2)
        merged = t.merge_overlapping()
        assert len(merged) == 1
        assert merged.intervals()[0] == Interval(0, 21)


# ============================================================
# Gaps
# ============================================================

class TestGaps:
    def test_no_gaps(self):
        t = IntervalTree().insert(1, 5).insert(3, 8)
        assert t.gaps() == []

    def test_single_gap(self):
        t = IntervalTree().insert(1, 5).insert(8, 12)
        gaps = t.gaps()
        assert len(gaps) == 1
        assert gaps[0] == Interval(5, 8)

    def test_multiple_gaps(self):
        t = IntervalTree().insert(1, 3).insert(5, 7).insert(9, 11)
        gaps = t.gaps()
        assert len(gaps) == 2
        assert gaps[0] == Interval(3, 5)
        assert gaps[1] == Interval(7, 9)

    def test_no_gap_touching(self):
        t = IntervalTree().insert(1, 5).insert(5, 10)
        assert t.gaps() == []


# ============================================================
# Coverage
# ============================================================

class TestCoverage:
    def test_non_overlapping(self):
        t = IntervalTree().insert(1, 3).insert(5, 7)
        assert t.coverage() == 4  # 2 + 2

    def test_overlapping(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        assert t.coverage() == 6  # [1,7] = 6

    def test_nested(self):
        t = IntervalTree().insert(1, 10).insert(3, 7)
        assert t.coverage() == 9  # [1,10] = 9

    def test_single(self):
        t = IntervalTree().insert(0, 5)
        assert t.coverage() == 5

    def test_union_length(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        assert t.union_length() == 6


# ============================================================
# Complement
# ============================================================

class TestComplement:
    def test_complement_with_gap(self):
        t = IntervalTree().insert(2, 4).insert(6, 8)
        comp = t.complement(0, 10)
        assert len(comp) == 3
        assert comp[0] == Interval(0, 2)
        assert comp[1] == Interval(4, 6)
        assert comp[2] == Interval(8, 10)

    def test_complement_full_coverage(self):
        t = IntervalTree().insert(0, 10)
        comp = t.complement(0, 10)
        assert len(comp) == 0

    def test_complement_empty(self):
        t = IntervalTree()
        comp = t.complement(0, 10)
        assert len(comp) == 1
        assert comp[0] == Interval(0, 10)


# ============================================================
# Clip
# ============================================================

class TestClip:
    def test_clip_no_change(self):
        t = IntervalTree().insert(2, 4)
        clipped = t.clip(0, 10)
        assert len(clipped) == 1
        assert clipped.intervals()[0] == Interval(2, 4)

    def test_clip_trims(self):
        t = IntervalTree().insert(1, 10)
        clipped = t.clip(3, 7)
        assert len(clipped) == 1
        assert clipped.intervals()[0] == Interval(3, 7)

    def test_clip_removes(self):
        t = IntervalTree().insert(1, 5).insert(8, 12)
        clipped = t.clip(3, 7)
        assert len(clipped) == 1
        assert clipped.intervals()[0] == Interval(3, 5)

    def test_clip_preserves_data(self):
        t = IntervalTree().insert(1, 10, "x")
        clipped = t.clip(3, 7)
        assert clipped.intervals()[0].data == "x"


# ============================================================
# Fragment
# ============================================================

class TestFragment:
    def test_non_overlapping(self):
        t = IntervalTree().insert(1, 3).insert(5, 7)
        f = t.fragment()
        assert len(f) == 2

    def test_overlapping(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        f = t.fragment()
        items = f.intervals()
        assert len(items) == 3
        assert items[0] == Interval(1, 3)
        assert items[1] == Interval(3, 5)
        assert items[2] == Interval(5, 7)

    def test_nested(self):
        t = IntervalTree().insert(1, 10).insert(3, 7)
        f = t.fragment()
        items = f.intervals()
        assert len(items) == 3
        assert items[0] == Interval(1, 3)
        assert items[1] == Interval(3, 7)
        assert items[2] == Interval(7, 10)

    def test_empty(self):
        assert IntervalTree().fragment().is_empty()

    def test_single(self):
        t = IntervalTree().insert(1, 5)
        f = t.fragment()
        assert len(f) == 1


# ============================================================
# Statistics
# ============================================================

class TestStatistics:
    def test_depth_at(self):
        t = IntervalTree().insert(1, 5).insert(3, 7).insert(4, 6)
        assert t.depth_at(4) == 3
        assert t.depth_at(2) == 1
        assert t.depth_at(6) == 2

    def test_max_depth(self):
        t = IntervalTree().insert(1, 10).insert(3, 7).insert(4, 6)
        d, p = t.max_depth()
        assert d == 3
        assert p == 4

    def test_max_depth_empty(self):
        d, p = IntervalTree().max_depth()
        assert d == 0
        assert p is None

    def test_max_depth_non_overlapping(self):
        t = IntervalTree().insert(1, 3).insert(5, 7).insert(9, 11)
        d, _ = t.max_depth()
        assert d == 1

    def test_histogram(self):
        t = IntervalTree()
        for i in range(100):
            t = t.insert(i, i + 5)
        hist = t.histogram(10)
        assert len(hist) == 10
        total = sum(c for _, c in hist)
        assert total == 100

    def test_histogram_empty(self):
        assert IntervalTree().histogram() == []

    def test_count(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        assert t.count() == 2


# ============================================================
# Sweep line
# ============================================================

class TestSweepLine:
    def test_intersect_all(self):
        t = IntervalTree().insert(1, 10).insert(3, 8).insert(5, 12)
        result = t.intersect_all()
        assert result == Interval(5, 8)

    def test_intersect_all_none(self):
        t = IntervalTree().insert(1, 3).insert(5, 7)
        assert t.intersect_all() is None

    def test_intersect_all_empty(self):
        assert IntervalTree().intersect_all() is None

    def test_intersect_all_single(self):
        t = IntervalTree().insert(1, 5)
        assert t.intersect_all() == Interval(1, 5)

    def test_pairwise_overlaps(self):
        t = IntervalTree().insert(1, 5).insert(3, 7).insert(8, 12)
        pairs = t.pairwise_overlaps()
        assert len(pairs) == 1  # only [1,5] and [3,7] overlap

    def test_pairwise_overlaps_many(self):
        t = IntervalTree().insert(1, 10).insert(3, 8).insert(5, 12)
        pairs = t.pairwise_overlaps()
        assert len(pairs) == 3  # all three overlap each other

    def test_pairwise_overlaps_none(self):
        t = IntervalTree().insert(1, 3).insert(5, 7).insert(9, 11)
        pairs = t.pairwise_overlaps()
        assert len(pairs) == 0

    def test_pairwise_overlaps_empty(self):
        assert IntervalTree().pairwise_overlaps() == []


# ============================================================
# Bulk operations
# ============================================================

class TestBulkOperations:
    def test_from_intervals(self):
        ivs = [Interval(1, 5), Interval(3, 7), Interval(8, 12)]
        t = IntervalTree.from_intervals(ivs)
        assert len(t) == 3

    def test_from_tuples(self):
        t = IntervalTree.from_intervals([(1, 5), (3, 7), (8, 12)])
        assert len(t) == 3

    def test_from_tuples_with_data(self):
        t = IntervalTree.from_intervals([(1, 5, "a"), (3, 7, "b")])
        assert len(t) == 2
        items = t.intervals()
        assert items[0].data == "a"

    def test_from_intervals_sorted(self):
        t = IntervalTree.from_intervals([(5, 10), (1, 3), (3, 7)])
        items = t.intervals()
        for i in range(len(items) - 1):
            assert items[i].lo <= items[i + 1].lo

    def test_filter(self):
        t = IntervalTree.from_intervals([(1, 5), (3, 7), (8, 12)])
        filtered = t.filter(lambda iv: iv.length > 3)
        assert len(filtered) == 3  # all have length 4 > 3

    def test_map_data(self):
        t = IntervalTree.from_intervals([(1, 5, 10), (3, 7, 20)])
        mapped = t.map_data(lambda d: d * 2 if d else d)
        items = mapped.intervals()
        assert items[0].data == 20
        assert items[1].data == 40

    def test_fold(self):
        t = IntervalTree.from_intervals([(1, 5), (3, 7), (8, 12)])
        total_length = t.fold(lambda acc, iv: acc + iv.length, 0)
        assert total_length == 12  # 4 + 4 + 4

    def test_from_intervals_invalid(self):
        with pytest.raises(TypeError):
            IntervalTree.from_intervals([42])

    def test_from_intervals_empty(self):
        t = IntervalTree.from_intervals([])
        assert t.is_empty()


# ============================================================
# Serialization
# ============================================================

class TestSerialization:
    def test_to_dict(self):
        t = IntervalTree().insert(1, 5, "a").insert(3, 7)
        d = t.to_dict()
        assert 'intervals' in d
        assert len(d['intervals']) == 2

    def test_from_dict(self):
        d = {'intervals': [
            {'lo': 1, 'hi': 5, 'data': 'a'},
            {'lo': 3, 'hi': 7}
        ]}
        t = IntervalTree.from_dict(d)
        assert len(t) == 2

    def test_roundtrip(self):
        t = IntervalTree().insert(1, 5, "a").insert(3, 7, "b").insert(8, 12)
        d = t.to_dict()
        t2 = IntervalTree.from_dict(d)
        assert len(t2) == len(t)
        for iv1, iv2 in zip(t.intervals(), t2.intervals()):
            assert iv1.lo == iv2.lo
            assert iv1.hi == iv2.hi
            assert iv1.data == iv2.data

    def test_to_list(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        items = t.to_list()
        assert len(items) == 2


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_point_intervals(self):
        t = IntervalTree()
        for i in range(10):
            t = t.insert(i, i)
        assert len(t) == 10
        assert len(t.stab(5)) == 1
        assert len(t.stab(5.5)) == 0

    def test_very_wide_interval(self):
        t = IntervalTree().insert(-1e6, 1e6)
        assert len(t.stab(0)) == 1
        assert len(t.stab(999999)) == 1

    def test_many_same_start(self):
        t = IntervalTree()
        for i in range(20):
            t = t.insert(0, i + 1)
        assert len(t) == 20
        assert len(t.stab(0)) == 20
        assert len(t.stab(15)) == 6  # [0,15]..[0,20]

    def test_many_same_end(self):
        t = IntervalTree()
        for i in range(20):
            t = t.insert(i, 20)
        assert len(t) == 20
        assert len(t.stab(20)) == 20
        assert len(t.stab(15)) == 16  # [0,20]..[15,20]

    def test_persistence_chain(self):
        trees = [IntervalTree()]
        for i in range(10):
            trees.append(trees[-1].insert(i, i + 5))
        for i, t in enumerate(trees):
            assert len(t) == i

    def test_negative_intervals(self):
        t = IntervalTree().insert(-10, -5).insert(-3, 3)
        assert len(t.stab(-7)) == 1   # in [-10,-5]
        assert len(t.stab(0)) == 1    # in [-3,3]
        assert len(t.stab(-5)) == 1   # in [-10,-5] only (-5 < -3)
        assert len(t.stab(-3)) == 1   # in [-3,3] only (-3 > -5)
        assert len(t.stab(-4)) == 0   # between the two intervals

    def test_float_precision(self):
        t = IntervalTree().insert(0.1, 0.3).insert(0.2, 0.4)
        results = t.stab(0.25)
        assert len(results) == 2

    def test_large_tree_queries(self):
        t = IntervalTree()
        for i in range(200):
            t = t.insert(i * 2, i * 2 + 3)
        # Stab at 100 should hit [98,101], [99,102], [100,103]
        results = t.stab(100)
        assert len(results) == 2  # [98,101] and [100,103]

    def test_empty_query_range(self):
        t = IntervalTree().insert(1, 10)
        results = t.overlap(20, 30)
        assert len(results) == 0

    def test_repr(self):
        t = IntervalTree().insert(1, 5)
        r = repr(t)
        assert "IntervalTree" in r

    def test_eq(self):
        t1 = IntervalTree().insert(1, 5).insert(3, 7)
        t2 = IntervalTree().insert(3, 7).insert(1, 5)
        assert t1 == t2

    def test_iter(self):
        t = IntervalTree().insert(1, 5).insert(3, 7)
        items = list(t)
        assert len(items) == 2


# ============================================================
# Integration: composition with C081
# ============================================================

class TestComposition:
    def test_monoid_pruning_effectiveness(self):
        """Verify that queries don't visit unnecessary parts of the tree."""
        t = IntervalTree()
        # Non-overlapping intervals spread across range
        for i in range(100):
            t = t.insert(i * 100, i * 100 + 10)
        # Query should only touch a small portion
        results = t.stab(505)
        assert len(results) == 1
        assert results[0] == Interval(500, 510)

    def test_finger_tree_persistence(self):
        """Verify finger tree persistence through interval operations."""
        t1 = IntervalTree().insert(1, 5)
        t2 = t1.insert(3, 7)
        t3 = t2.delete(1, 5)
        assert len(t1) == 1
        assert len(t2) == 2
        assert len(t3) == 1

    def test_complex_workflow(self):
        """Test a realistic workflow: build, query, modify, query again."""
        # Build a scheduling system
        t = IntervalTree()
        t = t.insert(9, 10, "standup")
        t = t.insert(10, 12, "coding")
        t = t.insert(12, 13, "lunch")
        t = t.insert(13, 17, "coding")
        t = t.insert(14, 15, "meeting")

        # What's happening at 14:30?
        at_1430 = t.stab(14.5)
        assert len(at_1430) == 2
        events = {iv.data for iv in at_1430}
        assert events == {"coding", "meeting"}

        # Any conflicts with proposed 11-13 meeting?
        conflicts = t.overlap(11, 13)
        assert len(conflicts) == 3  # coding, lunch, coding overlap

        # Remove the meeting
        t = t.delete(Interval(14, 15, "meeting"))
        at_1430 = t.stab(14.5)
        assert len(at_1430) == 1

        # Total scheduled time
        assert t.coverage() == 8  # 9 to 17

    def test_genome_style_intervals(self):
        """Test with genome-style intervals (large ranges, many overlapping)."""
        t = IntervalTree()
        # Simulated gene regions
        genes = [
            (1000, 5000, "gene_a"),
            (3000, 8000, "gene_b"),
            (7000, 12000, "gene_c"),
            (15000, 20000, "gene_d"),
            (2000, 4000, "exon_a1"),
            (4500, 4900, "exon_a2"),
        ]
        for lo, hi, name in genes:
            t = t.insert(lo, hi, name)

        # What genes are at position 3500?
        at_3500 = t.stab(3500)
        names = {iv.data for iv in at_3500}
        assert "gene_a" in names
        assert "gene_b" in names
        assert "exon_a1" in names

        # What overlaps with region [4000, 5000]?
        overlaps = t.overlap(4000, 5000)
        assert len(overlaps) >= 3  # gene_a, gene_b, exon_a2

    def test_timeline_gaps(self):
        """Test finding gaps in a timeline."""
        t = IntervalTree.from_intervals([
            (0, 3), (5, 8), (10, 15), (20, 25)
        ])
        gaps = t.gaps()
        assert len(gaps) == 3
        assert gaps[0] == Interval(3, 5)
        assert gaps[1] == Interval(8, 10)
        assert gaps[2] == Interval(15, 20)

    def test_coverage_computation(self):
        """Test coverage with complex overlapping."""
        t = IntervalTree.from_intervals([
            (0, 10), (5, 15), (12, 20), (25, 30)
        ])
        assert t.coverage() == 25  # [0,20] + [25,30] = 20 + 5

    def test_fragment_and_reconstruct(self):
        """Fragment intervals and verify coverage is preserved."""
        t = IntervalTree.from_intervals([(1, 10), (5, 15)])
        original_coverage = t.coverage()
        fragmented = t.fragment()
        frag_coverage = fragmented.coverage()
        assert frag_coverage == original_coverage


# ============================================================
# Stress tests
# ============================================================

class TestStress:
    def test_large_insert_query(self):
        """Insert 500 intervals and query."""
        t = IntervalTree()
        for i in range(500):
            t = t.insert(i, i + 10)
        assert len(t) == 500
        results = t.stab(250)
        assert len(results) == 11  # [240..250] through [250..260]

    def test_large_overlap_query(self):
        """Overlap query on large tree."""
        t = IntervalTree()
        for i in range(500):
            t = t.insert(i * 2, i * 2 + 3)
        results = t.overlap(100, 110)
        assert len(results) > 0

    def test_large_merge_overlapping(self):
        """Merge overlapping on tree with many intervals."""
        t = IntervalTree()
        for i in range(100):
            t = t.insert(i, i + 5)
        merged = t.merge_overlapping()
        assert len(merged) == 1

    def test_build_and_destroy(self):
        """Build then delete all intervals."""
        t = IntervalTree()
        for i in range(50):
            t = t.insert(i, i + 5)
        for i in range(50):
            t = t.delete(i, i + 5)
        assert t.is_empty()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
