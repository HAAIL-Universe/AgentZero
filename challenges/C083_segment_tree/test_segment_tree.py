"""Tests for C083: Segment Tree with Lazy Propagation."""

import pytest
from math import inf, gcd
from segment_tree import (
    SegmentTree, SumMonoid, MinMonoid, MaxMonoid, GCDMonoid, ProductMonoid,
    AddLazy, SetLazy, AddLazyMin, SetLazyMin, AddLazyMax, SetLazyMax,
    PersistentSegmentTree,
    SegmentTreeBeats,
    MergeSortTree,
    SegmentTree2D,
    SparseSegmentTree,
)


# ===================================================================
# Basic Segment Tree (Sum + Add)
# ===================================================================

class TestSegmentTreeBasic:
    def test_build_from_array(self):
        st = SegmentTree([1, 2, 3, 4, 5])
        assert st.query(0, 4) == 15

    def test_point_query(self):
        st = SegmentTree([10, 20, 30])
        assert st.query_point(0) == 10
        assert st.query_point(1) == 20
        assert st.query_point(2) == 30

    def test_range_query(self):
        st = SegmentTree([1, 3, 5, 7, 9, 11])
        assert st.query(0, 5) == 36
        assert st.query(1, 3) == 15
        assert st.query(2, 2) == 5
        assert st.query(0, 0) == 1

    def test_point_update(self):
        st = SegmentTree([1, 2, 3, 4, 5])
        st.update_point(2, 10)
        assert st.query(0, 4) == 22
        assert st.query_point(2) == 10

    def test_range_add(self):
        st = SegmentTree([1, 2, 3, 4, 5])
        st.update_range(1, 3, 10)  # add 10 to indices 1,2,3
        assert st.query(0, 4) == 45
        assert st.query_point(1) == 12
        assert st.query_point(0) == 1

    def test_multiple_range_adds(self):
        st = SegmentTree([0, 0, 0, 0, 0])
        st.update_range(0, 2, 5)
        st.update_range(2, 4, 3)
        assert st.to_list() == [5, 5, 8, 3, 3]

    def test_empty(self):
        st = SegmentTree([])
        assert len(st) == 0
        assert st.query(0, 0) == 0

    def test_single_element(self):
        st = SegmentTree([42])
        assert st.query(0, 0) == 42
        st.update_range(0, 0, 8)
        assert st.query(0, 0) == 50

    def test_to_list(self):
        st = SegmentTree([1, 2, 3, 4])
        assert st.to_list() == [1, 2, 3, 4]

    def test_len(self):
        st = SegmentTree([1, 2, 3])
        assert len(st) == 3

    def test_negative_values(self):
        st = SegmentTree([-5, -3, -1, 2, 4])
        assert st.query(0, 4) == -3
        assert st.query(0, 2) == -9

    def test_large_range_add(self):
        n = 1000
        st = SegmentTree([0] * n)
        st.update_range(0, n - 1, 1)
        assert st.query(0, n - 1) == n

    def test_overlapping_range_adds(self):
        st = SegmentTree([0, 0, 0, 0, 0, 0])
        st.update_range(0, 3, 1)
        st.update_range(2, 5, 2)
        st.update_range(1, 4, 3)
        assert st.to_list() == [1, 4, 6, 6, 5, 2]

    def test_out_of_range_query(self):
        st = SegmentTree([1, 2, 3])
        # l > r should return identity
        assert st.query(2, 1) == 0


# ===================================================================
# Min Monoid Segment Tree
# ===================================================================

class TestMinSegmentTree:
    def test_min_query(self):
        st = SegmentTree([5, 2, 8, 1, 9], monoid=MinMonoid, lazy_op=AddLazyMin)
        assert st.query(0, 4) == 1
        assert st.query(0, 2) == 2
        assert st.query(3, 4) == 1

    def test_min_range_add(self):
        st = SegmentTree([5, 2, 8, 1, 9], monoid=MinMonoid, lazy_op=AddLazyMin)
        st.update_range(2, 4, -10)
        assert st.query(0, 4) == -9  # 1 + (-10) = -9
        assert st.query(0, 1) == 2

    def test_min_point_update(self):
        st = SegmentTree([5, 2, 8, 1, 9], monoid=MinMonoid, lazy_op=AddLazyMin)
        st.update_point(3, 100)
        assert st.query(0, 4) == 2
        assert st.query(3, 4) == 9

    def test_min_set_lazy(self):
        st = SegmentTree([5, 2, 8, 1, 9], monoid=MinMonoid, lazy_op=SetLazyMin)
        st.update_range(0, 4, 3)
        assert st.query(0, 4) == 3
        assert st.query(2, 2) == 3


# ===================================================================
# Max Monoid Segment Tree
# ===================================================================

class TestMaxSegmentTree:
    def test_max_query(self):
        st = SegmentTree([5, 2, 8, 1, 9], monoid=MaxMonoid, lazy_op=AddLazyMax)
        assert st.query(0, 4) == 9
        assert st.query(0, 2) == 8

    def test_max_range_add(self):
        st = SegmentTree([5, 2, 8, 1, 9], monoid=MaxMonoid, lazy_op=AddLazyMax)
        st.update_range(0, 2, 10)
        assert st.query(0, 4) == 18  # 8 + 10

    def test_max_set(self):
        st = SegmentTree([5, 2, 8, 1, 9], monoid=MaxMonoid, lazy_op=SetLazyMax)
        st.update_range(0, 4, 7)
        assert st.query(0, 4) == 7


# ===================================================================
# GCD Monoid
# ===================================================================

class TestGCDSegmentTree:
    def test_gcd_query(self):
        st = SegmentTree([12, 8, 16, 24], monoid=GCDMonoid)
        assert st.query(0, 3) == 4
        assert st.query(0, 1) == 4
        assert st.query(1, 3) == 8

    def test_gcd_point_update(self):
        st = SegmentTree([12, 8, 16, 24], monoid=GCDMonoid)
        st.update_point(0, 6)
        assert st.query(0, 3) == 2


# ===================================================================
# Product Monoid
# ===================================================================

class TestProductSegmentTree:
    def test_product_query(self):
        st = SegmentTree([2, 3, 4, 5], monoid=ProductMonoid)
        assert st.query(0, 3) == 120
        assert st.query(1, 2) == 12

    def test_product_point_update(self):
        st = SegmentTree([2, 3, 4, 5], monoid=ProductMonoid)
        st.update_point(2, 1)
        assert st.query(0, 3) == 30


# ===================================================================
# Set Lazy (range assignment)
# ===================================================================

class TestSetLazy:
    def test_range_set(self):
        st = SegmentTree([1, 2, 3, 4, 5], lazy_op=SetLazy)
        st.update_range(1, 3, 10)
        assert st.to_list() == [1, 10, 10, 10, 5]
        assert st.query(0, 4) == 36

    def test_overlapping_sets(self):
        st = SegmentTree([0, 0, 0, 0, 0], lazy_op=SetLazy)
        st.update_range(0, 4, 5)
        st.update_range(2, 4, 3)
        assert st.to_list() == [5, 5, 3, 3, 3]

    def test_set_then_query(self):
        st = SegmentTree([1, 1, 1, 1], lazy_op=SetLazy)
        st.update_range(0, 3, 7)
        assert st.query(0, 3) == 28


# ===================================================================
# Find First / Find Last
# ===================================================================

class TestFindOperations:
    def test_find_first_prefix_sum(self):
        st = SegmentTree([1, 2, 3, 4, 5])
        # Find first index where prefix sum >= 6
        idx = st.find_first(0, 4, lambda s: s >= 6)
        assert idx == 2  # 1+2+3 = 6

    def test_find_first_no_match(self):
        st = SegmentTree([1, 1, 1])
        idx = st.find_first(0, 2, lambda s: s >= 100)
        assert idx == -1

    def test_find_first_immediate(self):
        st = SegmentTree([10, 1, 1])
        idx = st.find_first(0, 2, lambda s: s >= 5)
        assert idx == 0

    def test_find_last(self):
        st = SegmentTree([1, 2, 3, 4, 5])
        # Find rightmost where suffix sum >= 9
        idx = st.find_last(0, 4, lambda s: s >= 9)
        assert idx == 3  # 5+4 = 9

    def test_find_first_subrange(self):
        st = SegmentTree([1, 2, 3, 4, 5])
        idx = st.find_first(2, 4, lambda s: s >= 7)
        assert idx == 3  # 3+4 = 7

    def test_find_first_empty(self):
        st = SegmentTree([])
        assert st.find_first(0, 0, lambda s: s >= 1) == -1
        assert st.find_last(0, 0, lambda s: s >= 1) == -1


# ===================================================================
# Kth Element (frequency tree)
# ===================================================================

class TestKthElement:
    def test_kth_basic(self):
        # Frequency array: index i has count[i] elements
        # Suppose values 0..4, counts = [2, 0, 3, 1, 4]
        st = SegmentTree([2, 0, 3, 1, 4])
        assert st.kth_element(1) == 0  # 1st element is in bucket 0
        assert st.kth_element(2) == 0  # 2nd is also bucket 0
        assert st.kth_element(3) == 2  # 3rd is in bucket 2
        assert st.kth_element(5) == 2  # 5th is in bucket 2
        assert st.kth_element(6) == 3  # 6th is in bucket 3

    def test_kth_empty(self):
        st = SegmentTree([])
        assert st.kth_element(1) == -1

    def test_kth_invalid(self):
        st = SegmentTree([1, 2, 3])
        assert st.kth_element(0) == -1
        assert st.kth_element(-1) == -1


# ===================================================================
# Persistent Segment Tree
# ===================================================================

class TestPersistentSegmentTree:
    def test_build_and_query(self):
        pst = PersistentSegmentTree([1, 2, 3, 4, 5])
        assert pst.query(0, 0, 4) == 15
        assert pst.query(0, 1, 3) == 9

    def test_point_update_creates_version(self):
        pst = PersistentSegmentTree([1, 2, 3, 4, 5])
        v1 = pst.update_point(0, 2, 10)
        assert pst.query(0, 2, 2) == 3   # old version unchanged
        assert pst.query(v1, 2, 2) == 10  # new version updated
        assert pst.query(v1, 0, 4) == 22

    def test_multiple_versions(self):
        pst = PersistentSegmentTree([0, 0, 0, 0])
        v1 = pst.update_point(0, 0, 5)
        v2 = pst.update_point(v1, 1, 10)
        v3 = pst.update_point(0, 2, 20)  # branch from v0

        assert pst.query(0, 0, 3) == 0
        assert pst.query(v1, 0, 3) == 5
        assert pst.query(v2, 0, 3) == 15
        assert pst.query(v3, 0, 3) == 20

    def test_range_update_persistent(self):
        pst = PersistentSegmentTree([1, 2, 3, 4])
        v1 = pst.update_range(0, 0, 3, 10)
        assert pst.query(0, 0, 3) == 10  # original
        assert pst.query(v1, 0, 3) == 50  # all +10

    def test_version_count(self):
        pst = PersistentSegmentTree([1, 2])
        assert pst.version_count == 1
        pst.update_point(0, 0, 5)
        assert pst.version_count == 2

    def test_empty_persistent(self):
        pst = PersistentSegmentTree([])
        assert pst.query(0, 0, 0) == 0

    def test_persistent_query_point(self):
        pst = PersistentSegmentTree([10, 20, 30])
        v1 = pst.update_point(0, 1, 99)
        assert pst.query_point(0, 1) == 20
        assert pst.query_point(v1, 1) == 99

    def test_persistent_branching(self):
        pst = PersistentSegmentTree([1, 1, 1, 1])
        v1 = pst.update_point(0, 0, 10)   # [10,1,1,1]
        v2 = pst.update_point(v1, 1, 20)  # [10,20,1,1]
        v3 = pst.update_point(v1, 2, 30)  # [10,1,30,1] -- branch from v1
        assert pst.query(v2, 0, 3) == 32
        assert pst.query(v3, 0, 3) == 42


# ===================================================================
# Segment Tree Beats
# ===================================================================

class TestSegmentTreeBeats:
    def test_build_and_query(self):
        stb = SegmentTreeBeats([3, 1, 4, 1, 5, 9])
        assert stb.query_sum(0, 5) == 23
        assert stb.query_min(0, 5) == 1
        assert stb.query_max(0, 5) == 9

    def test_range_chmin(self):
        stb = SegmentTreeBeats([3, 1, 4, 1, 5, 9])
        stb.range_chmin(0, 5, 4)  # clamp all to <= 4
        assert stb.query_max(0, 5) == 4
        assert stb.query_sum(0, 5) == 17  # [3,1,4,1,4,4]
        assert stb.query_min(0, 5) == 1

    def test_range_chmax(self):
        stb = SegmentTreeBeats([3, 1, 4, 1, 5, 9])
        stb.range_chmax(0, 5, 3)  # clamp all to >= 3
        assert stb.query_min(0, 5) == 3
        assert stb.query_sum(0, 5) == 27  # [3,3,4,3,5,9]

    def test_range_add_beats(self):
        stb = SegmentTreeBeats([1, 2, 3, 4, 5])
        stb.range_add(0, 4, 10)
        assert stb.query_sum(0, 4) == 65
        assert stb.query_min(0, 4) == 11
        assert stb.query_max(0, 4) == 15

    def test_chmin_then_query(self):
        stb = SegmentTreeBeats([10, 20, 30, 40, 50])
        stb.range_chmin(0, 4, 25)
        assert stb.to_list() == [10, 20, 25, 25, 25]
        assert stb.query_sum(0, 4) == 105

    def test_chmax_then_query(self):
        stb = SegmentTreeBeats([10, 20, 30, 40, 50])
        stb.range_chmax(0, 4, 25)
        assert stb.to_list() == [25, 25, 30, 40, 50]
        assert stb.query_sum(0, 4) == 170

    def test_combined_chmin_chmax(self):
        stb = SegmentTreeBeats([1, 5, 3, 7, 2])
        stb.range_chmax(0, 4, 3)  # clamp >= 3 -> [3,5,3,7,3]
        stb.range_chmin(0, 4, 5)  # clamp <= 5 -> [3,5,3,5,3]
        assert stb.to_list() == [3, 5, 3, 5, 3]
        assert stb.query_sum(0, 4) == 19

    def test_add_then_chmin(self):
        stb = SegmentTreeBeats([1, 2, 3, 4, 5])
        stb.range_add(0, 4, 5)  # [6,7,8,9,10]
        stb.range_chmin(0, 4, 8)  # [6,7,8,8,8]
        assert stb.to_list() == [6, 7, 8, 8, 8]

    def test_empty_beats(self):
        stb = SegmentTreeBeats([])
        assert stb.query_sum(0, 0) == 0
        assert stb.query_min(0, 0) == inf
        assert stb.query_max(0, 0) == -inf

    def test_single_element_beats(self):
        stb = SegmentTreeBeats([42])
        stb.range_chmin(0, 0, 10)
        assert stb.query_sum(0, 0) == 10
        stb.range_chmax(0, 0, 20)
        assert stb.query_sum(0, 0) == 20

    def test_all_same_values(self):
        stb = SegmentTreeBeats([5, 5, 5, 5])
        stb.range_chmin(0, 3, 5)  # no change
        assert stb.query_sum(0, 3) == 20
        stb.range_chmin(0, 3, 3)
        assert stb.to_list() == [3, 3, 3, 3]

    def test_subrange_operations(self):
        stb = SegmentTreeBeats([1, 2, 3, 4, 5, 6])
        stb.range_chmin(2, 4, 3)  # [1,2,3,3,3,6]
        assert stb.query_sum(0, 5) == 18
        assert stb.query_max(0, 5) == 6


# ===================================================================
# Merge Sort Tree
# ===================================================================

class TestMergeSortTree:
    def test_count_less_than(self):
        mst = MergeSortTree([3, 1, 4, 1, 5, 9, 2, 6])
        assert mst.count_less_than(0, 7, 5) == 5  # 3,1,4,1,2
        assert mst.count_less_than(0, 7, 1) == 0
        assert mst.count_less_than(0, 7, 10) == 8

    def test_count_less_equal(self):
        mst = MergeSortTree([3, 1, 4, 1, 5, 9, 2, 6])
        assert mst.count_less_equal(0, 7, 4) == 5  # 3,1,4,1,2

    def test_count_in_range(self):
        mst = MergeSortTree([3, 1, 4, 1, 5, 9, 2, 6])
        assert mst.count_in_range(0, 7, 2, 5) == 4  # 3,4,5,2

    def test_kth_smallest(self):
        mst = MergeSortTree([3, 1, 4, 1, 5])
        assert mst.kth_smallest(0, 4, 1) == 1
        assert mst.kth_smallest(0, 4, 2) == 1
        assert mst.kth_smallest(0, 4, 3) == 3
        assert mst.kth_smallest(0, 4, 4) == 4
        assert mst.kth_smallest(0, 4, 5) == 5

    def test_kth_subrange(self):
        mst = MergeSortTree([5, 2, 8, 3, 7])
        assert mst.kth_smallest(1, 3, 1) == 2  # [2,8,3] -> sorted [2,3,8]
        assert mst.kth_smallest(1, 3, 2) == 3
        assert mst.kth_smallest(1, 3, 3) == 8

    def test_kth_invalid(self):
        mst = MergeSortTree([1, 2, 3])
        assert mst.kth_smallest(0, 2, 4) is None
        assert mst.kth_smallest(0, 2, 0) is None

    def test_empty_merge_sort_tree(self):
        mst = MergeSortTree([])
        assert mst.count_less_than(0, 0, 5) == 0

    def test_count_subrange(self):
        mst = MergeSortTree([10, 20, 30, 40, 50])
        assert mst.count_less_than(2, 4, 40) == 1  # [30,40,50], 30 < 40

    def test_single_element_mst(self):
        mst = MergeSortTree([42])
        assert mst.count_less_than(0, 0, 42) == 0
        assert mst.count_less_than(0, 0, 43) == 1
        assert mst.kth_smallest(0, 0, 1) == 42


# ===================================================================
# 2D Segment Tree
# ===================================================================

class TestSegmentTree2D:
    def test_build_and_query(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        st2d = SegmentTree2D(matrix)
        assert st2d.query(0, 0, 2, 2) == 45  # sum of all
        assert st2d.query(0, 0, 0, 0) == 1
        assert st2d.query(1, 1, 2, 2) == 28  # 5+6+8+9

    def test_point_update_2d(self):
        matrix = [
            [1, 2],
            [3, 4],
        ]
        st2d = SegmentTree2D(matrix)
        assert st2d.query(0, 0, 1, 1) == 10
        st2d.update(0, 0, 10)
        assert st2d.query(0, 0, 1, 1) == 19
        assert st2d.query(0, 0, 0, 0) == 10

    def test_row_query(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        st2d = SegmentTree2D(matrix)
        assert st2d.query(0, 0, 0, 2) == 6  # row 0
        assert st2d.query(1, 0, 1, 2) == 15  # row 1

    def test_col_query(self):
        matrix = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        st2d = SegmentTree2D(matrix)
        assert st2d.query(0, 0, 2, 0) == 9   # col 0
        assert st2d.query(0, 1, 2, 1) == 12  # col 1

    def test_single_cell(self):
        matrix = [[42]]
        st2d = SegmentTree2D(matrix)
        assert st2d.query(0, 0, 0, 0) == 42
        st2d.update(0, 0, 99)
        assert st2d.query(0, 0, 0, 0) == 99

    def test_empty_2d(self):
        st2d = SegmentTree2D([])
        assert st2d.query(0, 0, 0, 0) == 0

    def test_multiple_updates_2d(self):
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
        ]
        st2d = SegmentTree2D(matrix)
        st2d.update(0, 1, 5)
        st2d.update(1, 2, 3)
        assert st2d.query(0, 0, 1, 2) == 8


# ===================================================================
# Sparse Segment Tree
# ===================================================================

class TestSparseSegmentTree:
    def test_basic(self):
        sst = SparseSegmentTree(0, 99)
        sst.update_point(5, 10)
        sst.update_point(50, 20)
        assert sst.query(0, 99) == 30
        assert sst.query(0, 5) == 10
        assert sst.query(6, 49) == 0
        assert sst.query(50, 50) == 20

    def test_range_add_sparse(self):
        sst = SparseSegmentTree(0, 99)
        sst.update_range(10, 20, 5)
        assert sst.query(10, 20) == 55  # 11 elements * 5
        assert sst.query(0, 9) == 0

    def test_large_range(self):
        sst = SparseSegmentTree(0, 10**9)
        sst.update_point(0, 1)
        sst.update_point(10**9, 2)
        assert sst.query(0, 10**9) == 3

    def test_query_point_sparse(self):
        sst = SparseSegmentTree(0, 999)
        sst.update_point(42, 100)
        assert sst.query_point(42) == 100
        assert sst.query_point(41) == 0

    def test_overlapping_range_adds_sparse(self):
        sst = SparseSegmentTree(0, 9)
        sst.update_range(0, 5, 1)
        sst.update_range(3, 9, 2)
        assert sst.query(0, 2) == 3   # 3 * 1
        assert sst.query(3, 5) == 9   # 3 * (1+2)
        assert sst.query(6, 9) == 8   # 4 * 2

    def test_negative_range(self):
        sst = SparseSegmentTree(-100, 100)
        sst.update_point(-50, 7)
        sst.update_point(50, 3)
        assert sst.query(-100, 100) == 10
        assert sst.query(-50, -50) == 7

    def test_empty_sparse(self):
        sst = SparseSegmentTree(0, 999)
        assert sst.query(0, 999) == 0


# ===================================================================
# Stress Tests / Combined
# ===================================================================

class TestStress:
    def test_range_add_matches_brute_force(self):
        import random
        random.seed(42)
        n = 50
        data = [random.randint(-100, 100) for _ in range(n)]
        brute = list(data)
        st = SegmentTree(data)

        for _ in range(100):
            op = random.randint(0, 1)
            if op == 0:  # range add
                l = random.randint(0, n - 1)
                r = random.randint(l, n - 1)
                v = random.randint(-50, 50)
                st.update_range(l, r, v)
                for i in range(l, r + 1):
                    brute[i] += v
            else:  # query
                l = random.randint(0, n - 1)
                r = random.randint(l, n - 1)
                expected = sum(brute[l:r + 1])
                assert st.query(l, r) == expected

    def test_range_set_matches_brute_force(self):
        import random
        random.seed(99)
        n = 50
        data = [random.randint(0, 100) for _ in range(n)]
        brute = list(data)
        st = SegmentTree(data, lazy_op=SetLazy)

        for _ in range(100):
            op = random.randint(0, 1)
            if op == 0:  # range set
                l = random.randint(0, n - 1)
                r = random.randint(l, n - 1)
                v = random.randint(0, 100)
                st.update_range(l, r, v)
                for i in range(l, r + 1):
                    brute[i] = v
            else:  # query
                l = random.randint(0, n - 1)
                r = random.randint(l, n - 1)
                expected = sum(brute[l:r + 1])
                assert st.query(l, r) == expected

    def test_min_range_add_stress(self):
        import random
        random.seed(77)
        n = 30
        data = [random.randint(0, 100) for _ in range(n)]
        brute = list(data)
        st = SegmentTree(data, monoid=MinMonoid, lazy_op=AddLazyMin)

        for _ in range(80):
            op = random.randint(0, 1)
            if op == 0:
                l = random.randint(0, n - 1)
                r = random.randint(l, n - 1)
                v = random.randint(-10, 10)
                st.update_range(l, r, v)
                for i in range(l, r + 1):
                    brute[i] += v
            else:
                l = random.randint(0, n - 1)
                r = random.randint(l, n - 1)
                expected = min(brute[l:r + 1])
                assert st.query(l, r) == expected

    def test_beats_chmin_stress(self):
        import random
        random.seed(123)
        n = 30
        data = [random.randint(0, 100) for _ in range(n)]
        brute = list(data)
        stb = SegmentTreeBeats(data)

        for _ in range(60):
            op = random.randint(0, 2)
            l = random.randint(0, n - 1)
            r = random.randint(l, n - 1)
            if op == 0:  # chmin
                v = random.randint(0, 100)
                stb.range_chmin(l, r, v)
                for i in range(l, r + 1):
                    brute[i] = min(brute[i], v)
            elif op == 1:  # chmax
                v = random.randint(0, 100)
                stb.range_chmax(l, r, v)
                for i in range(l, r + 1):
                    brute[i] = max(brute[i], v)
            else:  # query sum
                expected = sum(brute[l:r + 1])
                assert stb.query_sum(l, r) == expected

    def test_persistent_stress(self):
        import random
        random.seed(55)
        n = 20
        data = [random.randint(0, 50) for _ in range(n)]
        pst = PersistentSegmentTree(data)
        versions = [list(data)]

        for _ in range(30):
            v_idx = random.randint(0, len(versions) - 1)
            idx = random.randint(0, n - 1)
            val = random.randint(0, 50)
            new_v = pst.update_point(v_idx, idx, val)
            new_arr = list(versions[v_idx])
            new_arr[idx] = val
            versions.append(new_arr)

        # Verify all versions
        for v_idx, arr in enumerate(versions):
            for i in range(n):
                assert pst.query_point(v_idx, i) == arr[i]

    def test_2d_stress(self):
        import random
        random.seed(88)
        rows, cols = 8, 8
        matrix = [[random.randint(0, 50) for _ in range(cols)] for _ in range(rows)]
        st2d = SegmentTree2D(matrix)

        for _ in range(30):
            # Random update
            r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            v = random.randint(0, 50)
            st2d.update(r, c, v)
            matrix[r][c] = v

            # Random query
            r1 = random.randint(0, rows - 1)
            r2 = random.randint(r1, rows - 1)
            c1 = random.randint(0, cols - 1)
            c2 = random.randint(c1, cols - 1)
            expected = sum(matrix[i][j] for i in range(r1, r2 + 1) for j in range(c1, c2 + 1))
            assert st2d.query(r1, c1, r2, c2) == expected

    def test_merge_sort_tree_kth_stress(self):
        import random
        random.seed(44)
        n = 50
        data = [random.randint(1, 100) for _ in range(n)]
        mst = MergeSortTree(data)

        for _ in range(30):
            l = random.randint(0, n - 1)
            r = random.randint(l, n - 1)
            k = random.randint(1, r - l + 1)
            expected = sorted(data[l:r + 1])[k - 1]
            assert mst.kth_smallest(l, r, k) == expected


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_power_of_two_size(self):
        st = SegmentTree([1, 2, 3, 4, 5, 6, 7, 8])
        assert st.query(0, 7) == 36

    def test_non_power_of_two(self):
        st = SegmentTree([1, 2, 3, 4, 5, 6, 7])
        assert st.query(0, 6) == 28

    def test_two_elements(self):
        st = SegmentTree([3, 7])
        assert st.query(0, 1) == 10
        st.update_range(0, 1, 1)
        assert st.query(0, 1) == 12

    def test_all_zeros(self):
        st = SegmentTree([0, 0, 0, 0])
        assert st.query(0, 3) == 0

    def test_large_values(self):
        st = SegmentTree([10**9, 10**9, 10**9])
        assert st.query(0, 2) == 3 * 10**9

    def test_alternating_add_query(self):
        st = SegmentTree([1, 1, 1, 1, 1])
        for i in range(5):
            st.update_range(i, i, i)
        assert st.to_list() == [1, 2, 3, 4, 5]

    def test_full_range_add_then_point_query(self):
        n = 100
        st = SegmentTree([0] * n)
        st.update_range(0, n - 1, 7)
        for i in range(n):
            assert st.query_point(i) == 7

    def test_persistent_no_data(self):
        pst = PersistentSegmentTree()
        assert pst.version_count == 1
        assert pst.n == 0

    def test_sparse_single_point(self):
        sst = SparseSegmentTree(0, 0)
        sst.update_point(0, 42)
        assert sst.query(0, 0) == 42

    def test_beats_identical_values(self):
        stb = SegmentTreeBeats([5, 5, 5, 5, 5])
        stb.range_chmin(0, 4, 3)
        assert stb.to_list() == [3, 3, 3, 3, 3]
        stb.range_chmax(0, 4, 7)
        assert stb.to_list() == [7, 7, 7, 7, 7]

    def test_beats_add_and_chmin(self):
        stb = SegmentTreeBeats([1, 2, 3])
        stb.range_add(0, 2, 10)  # [11, 12, 13]
        stb.range_chmin(0, 2, 12)  # [11, 12, 12]
        assert stb.to_list() == [11, 12, 12]

    def test_sparse_boundary_queries(self):
        sst = SparseSegmentTree(0, 10)
        sst.update_point(0, 1)
        sst.update_point(10, 2)
        assert sst.query(0, 0) == 1
        assert sst.query(10, 10) == 2
        assert sst.query(1, 9) == 0

    def test_merge_sort_all_same(self):
        mst = MergeSortTree([5, 5, 5, 5])
        assert mst.count_less_than(0, 3, 5) == 0
        assert mst.count_less_equal(0, 3, 5) == 4
        assert mst.kth_smallest(0, 3, 1) == 5
        assert mst.kth_smallest(0, 3, 4) == 5

    def test_persistent_range_update_versions(self):
        pst = PersistentSegmentTree([1, 2, 3, 4, 5])
        v1 = pst.update_range(0, 1, 2, 10)  # add 10 to [1,2]
        assert pst.query(v1, 1, 2) == 25  # 12 + 13
        assert pst.query(0, 1, 2) == 5  # original unchanged

    def test_segment_tree_many_point_updates(self):
        st = SegmentTree([0] * 10)
        for i in range(10):
            st.update_point(i, i * i)
        assert st.query(0, 9) == sum(i * i for i in range(10))
