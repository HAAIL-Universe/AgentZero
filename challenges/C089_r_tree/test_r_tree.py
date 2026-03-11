"""Tests for C089: R-Tree spatial index."""

import math
import pytest
from r_tree import (
    BoundingBox, union_all, RTreeEntry, RTreeNode,
    RTree, RStarTree, str_bulk_load, spatial_join, SpatialIndex
)


# ============================================================
# BoundingBox Tests
# ============================================================

class TestBoundingBox:
    def test_create_2d(self):
        bb = BoundingBox([0, 0], [10, 10])
        assert bb.mins == (0, 0)
        assert bb.maxs == (10, 10)
        assert bb.dims == 2

    def test_create_3d(self):
        bb = BoundingBox([1, 2, 3], [4, 5, 6])
        assert bb.dims == 3

    def test_invalid_min_gt_max(self):
        with pytest.raises(ValueError):
            BoundingBox([5, 0], [3, 10])

    def test_invalid_length_mismatch(self):
        with pytest.raises(ValueError):
            BoundingBox([0, 0], [1, 1, 1])

    def test_from_point(self):
        bb = BoundingBox.from_point((3, 4))
        assert bb.mins == (3, 4)
        assert bb.maxs == (3, 4)
        assert bb.area() == 0

    def test_area_2d(self):
        bb = BoundingBox([0, 0], [3, 4])
        assert bb.area() == 12.0

    def test_area_3d(self):
        bb = BoundingBox([0, 0, 0], [2, 3, 4])
        assert bb.area() == 24.0

    def test_area_zero_volume(self):
        bb = BoundingBox([0, 0], [5, 0])
        assert bb.area() == 0.0

    def test_margin(self):
        bb = BoundingBox([0, 0], [3, 4])
        assert bb.margin() == 7.0

    def test_center(self):
        bb = BoundingBox([0, 0], [10, 20])
        assert bb.center() == (5.0, 10.0)

    def test_intersects_overlapping(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([3, 3], [8, 8])
        assert a.intersects(b)
        assert b.intersects(a)

    def test_intersects_touching(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([5, 0], [10, 5])
        assert a.intersects(b)

    def test_intersects_disjoint(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([6, 6], [10, 10])
        assert not a.intersects(b)

    def test_contains_yes(self):
        a = BoundingBox([0, 0], [10, 10])
        b = BoundingBox([2, 2], [8, 8])
        assert a.contains(b)
        assert not b.contains(a)

    def test_contains_self(self):
        a = BoundingBox([0, 0], [5, 5])
        assert a.contains(a)

    def test_contains_point_inside(self):
        bb = BoundingBox([0, 0], [10, 10])
        assert bb.contains_point((5, 5))

    def test_contains_point_boundary(self):
        bb = BoundingBox([0, 0], [10, 10])
        assert bb.contains_point((0, 0))
        assert bb.contains_point((10, 10))

    def test_contains_point_outside(self):
        bb = BoundingBox([0, 0], [10, 10])
        assert not bb.contains_point((11, 5))

    def test_union(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([3, 3], [8, 8])
        u = a.union(b)
        assert u.mins == (0, 0)
        assert u.maxs == (8, 8)

    def test_intersection_overlap(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([3, 3], [8, 8])
        i = a.intersection(b)
        assert i is not None
        assert i.mins == (3, 3)
        assert i.maxs == (5, 5)

    def test_intersection_disjoint(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([6, 6], [10, 10])
        assert a.intersection(b) is None

    def test_enlargement(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([3, 3], [8, 8])
        # union is [0,0]-[8,8] = 64, a is 25, so enlargement = 39
        assert a.enlargement(b) == 39.0

    def test_overlap_area(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([3, 3], [8, 8])
        assert a.overlap_area(b) == 4.0  # [3,3]-[5,5] = 2*2

    def test_overlap_area_disjoint(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([6, 6], [10, 10])
        assert a.overlap_area(b) == 0.0

    def test_min_distance_to_point_inside(self):
        bb = BoundingBox([0, 0], [10, 10])
        assert bb.min_distance_to_point((5, 5)) == 0.0

    def test_min_distance_to_point_outside(self):
        bb = BoundingBox([0, 0], [10, 10])
        d = bb.min_distance_to_point((13, 14))
        assert abs(d - 5.0) < 1e-9  # sqrt(9+16)=5

    def test_min_distance_to_bbox_overlap(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([3, 3], [8, 8])
        assert a.min_distance_to_bbox(b) == 0.0

    def test_min_distance_to_bbox_disjoint(self):
        a = BoundingBox([0, 0], [3, 3])
        b = BoundingBox([6, 7], [10, 10])
        d = a.min_distance_to_bbox(b)
        assert abs(d - 5.0) < 1e-9  # sqrt(9+16)=5

    def test_equality(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([0, 0], [5, 5])
        assert a == b

    def test_hash(self):
        a = BoundingBox([0, 0], [5, 5])
        b = BoundingBox([0, 0], [5, 5])
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_repr(self):
        bb = BoundingBox([1, 2], [3, 4])
        assert "BBox" in repr(bb)


class TestUnionAll:
    def test_basic(self):
        bbs = [
            BoundingBox([0, 0], [1, 1]),
            BoundingBox([5, 5], [6, 6]),
            BoundingBox([2, 2], [3, 3]),
        ]
        u = union_all(bbs)
        assert u.mins == (0, 0)
        assert u.maxs == (6, 6)


# ============================================================
# RTreeEntry and RTreeNode Tests
# ============================================================

class TestRTreeEntry:
    def test_leaf_entry(self):
        bb = BoundingBox([0, 0], [5, 5])
        e = RTreeEntry(bb, data_id="a")
        assert e.is_leaf_entry()
        assert e.data_id == "a"

    def test_internal_entry(self):
        bb = BoundingBox([0, 0], [5, 5])
        child = RTreeNode(is_leaf=True)
        e = RTreeEntry(bb, child=child)
        assert not e.is_leaf_entry()


class TestRTreeNode:
    def test_empty_bbox(self):
        n = RTreeNode()
        assert n.bbox() is None

    def test_bbox(self):
        n = RTreeNode()
        n.entries = [
            RTreeEntry(BoundingBox([0, 0], [5, 5])),
            RTreeEntry(BoundingBox([3, 3], [8, 8])),
        ]
        bb = n.bbox()
        assert bb.mins == (0, 0)
        assert bb.maxs == (8, 8)


# ============================================================
# RTree Tests (Classic)
# ============================================================

class TestRTree:
    def test_create_empty(self):
        t = RTree()
        assert len(t) == 0
        assert t.height == 1

    def test_insert_one(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        assert len(t) == 1

    def test_insert_multiple(self):
        t = RTree(max_entries=4)
        for i in range(10):
            t.insert(BoundingBox([i, i], [i + 1, i + 1]), data_id=i)
        assert len(t) == 10
        assert t.height >= 2

    def test_insert_triggers_split(self):
        t = RTree(max_entries=4)
        for i in range(5):
            t.insert(BoundingBox([i * 10, 0], [i * 10 + 5, 5]), data_id=i)
        assert len(t) == 5
        assert t.height == 2

    def test_insert_point(self):
        t = RTree()
        t.insert((3, 4), data_id="pt")
        assert len(t) == 1
        results = t.point_query((3, 4))
        assert len(results) == 1

    def test_insert_list_bbox(self):
        t = RTree()
        t.insert(([0, 0], [5, 5]), data_id="rect")
        assert len(t) == 1

    def test_search_basic(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        t.insert(BoundingBox([10, 10], [15, 15]), data_id="b")
        t.insert(BoundingBox([3, 3], [8, 8]), data_id="c")

        results = t.search(BoundingBox([4, 4], [6, 6]))
        ids = {r[1] for r in results}
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    def test_search_no_results(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        results = t.search(BoundingBox([20, 20], [30, 30]))
        assert len(results) == 0

    def test_search_all(self):
        t = RTree()
        for i in range(10):
            t.insert(BoundingBox([0, 0], [100, 100]), data_id=i)
        results = t.search(BoundingBox([0, 0], [100, 100]))
        assert len(results) == 10

    def test_search_with_list_query(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        results = t.search(([3, 3], [8, 8]))
        assert len(results) == 1

    def test_contains_query(self):
        t = RTree()
        t.insert(BoundingBox([2, 2], [4, 4]), data_id="inside")
        t.insert(BoundingBox([0, 0], [10, 10]), data_id="large")
        results = t.contains(BoundingBox([1, 1], [5, 5]))
        ids = {r[1] for r in results}
        assert "inside" in ids
        assert "large" not in ids

    def test_point_query(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        t.insert(BoundingBox([3, 3], [8, 8]), data_id="b")
        t.insert(BoundingBox([10, 10], [15, 15]), data_id="c")

        results = t.point_query((4, 4))
        ids = {r[1] for r in results}
        assert ids == {"a", "b"}

    def test_nearest_basic(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [1, 1]), data_id="a")
        t.insert(BoundingBox([10, 10], [11, 11]), data_id="b")
        t.insert(BoundingBox([5, 5], [6, 6]), data_id="c")

        results = t.nearest((0, 0), k=1)
        assert len(results) == 1
        assert results[0][1] == "a"

    def test_nearest_k(self):
        t = RTree()
        for i in range(10):
            t.insert(BoundingBox([i * 5, 0], [i * 5 + 1, 1]), data_id=i)

        results = t.nearest((0, 0), k=3)
        assert len(results) == 3
        # Closest should be id=0
        assert results[0][1] == 0

    def test_nearest_inside(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [10, 10]), data_id="big")
        results = t.nearest((5, 5), k=1)
        assert results[0][2] == 0.0  # distance is 0

    def test_delete_basic(self):
        t = RTree(max_entries=4)
        bb = BoundingBox([0, 0], [5, 5])
        t.insert(bb, data_id="a")
        assert len(t) == 1
        assert t.delete(bb, data_id="a")
        assert len(t) == 0

    def test_delete_nonexistent(self):
        t = RTree()
        assert not t.delete(BoundingBox([0, 0], [5, 5]))

    def test_delete_from_multi(self):
        t = RTree(max_entries=4)
        bbs = []
        for i in range(8):
            bb = BoundingBox([i * 10, 0], [i * 10 + 5, 5])
            bbs.append(bb)
            t.insert(bb, data_id=i)
        assert len(t) == 8

        assert t.delete(bbs[3], data_id=3)
        assert len(t) == 7
        # Should not find deleted entry
        results = t.search(bbs[3])
        ids = {r[1] for r in results}
        assert 3 not in ids

    def test_delete_causes_condense(self):
        t = RTree(max_entries=4, min_entries=2)
        for i in range(8):
            bb = BoundingBox([i * 10, 0], [i * 10 + 5, 5])
            t.insert(bb, data_id=i)

        # Delete enough to trigger condense
        for i in range(6):
            bb = BoundingBox([i * 10, 0], [i * 10 + 5, 5])
            t.delete(bb, data_id=i)

        assert len(t) == 2
        all_e = t.all_entries()
        ids = {e[1] for e in all_e}
        assert ids == {6, 7}

    def test_delete_shrinks_root(self):
        t = RTree(max_entries=4)
        for i in range(5):
            t.insert(BoundingBox([i * 10, 0], [i * 10 + 5, 5]), data_id=i)
        assert t.height == 2

        for i in range(4):
            t.delete(BoundingBox([i * 10, 0], [i * 10 + 5, 5]), data_id=i)
        assert len(t) == 1

    def test_all_entries(self):
        t = RTree()
        for i in range(5):
            t.insert(BoundingBox([i, i], [i + 1, i + 1]), data_id=i)
        entries = t.all_entries()
        assert len(entries) == 5
        ids = {e[1] for e in entries}
        assert ids == {0, 1, 2, 3, 4}

    def test_clear(self):
        t = RTree()
        for i in range(5):
            t.insert(BoundingBox([i, i], [i + 1, i + 1]), data_id=i)
        t.clear()
        assert len(t) == 0
        assert t.height == 1

    def test_bbox_root(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]))
        t.insert(BoundingBox([10, 10], [15, 15]))
        bb = t.bbox()
        assert bb.mins == (0, 0)
        assert bb.maxs == (15, 15)

    def test_depth_stats(self):
        t = RTree(max_entries=4)
        for i in range(20):
            t.insert(BoundingBox([i, i], [i + 1, i + 1]), data_id=i)
        stats = t.depth_stats()
        assert stats['entries'] == 20
        assert stats['height'] >= 2
        assert stats['nodes'] > 0
        assert 0 < stats['fill_factor'] <= 1.0

    def test_3d_rtree(self):
        t = RTree(dims=3)
        t.insert(BoundingBox([0, 0, 0], [5, 5, 5]), data_id="a")
        t.insert(BoundingBox([3, 3, 3], [8, 8, 8]), data_id="b")
        results = t.search(BoundingBox([4, 4, 4], [6, 6, 6]))
        ids = {r[1] for r in results}
        assert ids == {"a", "b"}

    def test_many_inserts_and_queries(self):
        t = RTree(max_entries=8)
        n = 100
        for i in range(n):
            x, y = (i * 7) % 100, (i * 13) % 100
            t.insert(BoundingBox([x, y], [x + 2, y + 2]), data_id=i)
        assert len(t) == n

        # Search a region
        results = t.search(BoundingBox([0, 0], [50, 50]))
        assert len(results) > 0

        # All entries recoverable
        all_e = t.all_entries()
        assert len(all_e) == n

    def test_overlapping_entries(self):
        t = RTree(max_entries=4)
        for i in range(10):
            t.insert(BoundingBox([0, 0], [10, 10]), data_id=i)
        results = t.search(BoundingBox([5, 5], [6, 6]))
        assert len(results) == 10


# ============================================================
# R*-Tree Tests
# ============================================================

class TestRStarTree:
    def test_create(self):
        t = RStarTree()
        assert len(t) == 0

    def test_insert_and_search(self):
        t = RStarTree(max_entries=4)
        for i in range(20):
            t.insert(BoundingBox([i * 5, 0], [i * 5 + 3, 3]), data_id=i)
        assert len(t) == 20

        results = t.search(BoundingBox([0, 0], [10, 5]))
        assert len(results) > 0

    def test_rstar_handles_splits(self):
        t = RStarTree(max_entries=4)
        for i in range(50):
            x = (i * 7) % 100
            y = (i * 13) % 100
            t.insert(BoundingBox([x, y], [x + 3, y + 3]), data_id=i)
        assert len(t) == 50

        all_e = t.all_entries()
        assert len(all_e) == 50

    def test_rstar_nearest(self):
        t = RStarTree(max_entries=4)
        for i in range(20):
            t.insert(BoundingBox([i * 10, 0], [i * 10 + 2, 2]), data_id=i)

        results = t.nearest((0, 0), k=3)
        assert len(results) == 3
        assert results[0][1] == 0

    def test_rstar_delete(self):
        t = RStarTree(max_entries=4)
        bbs = []
        for i in range(10):
            bb = BoundingBox([i * 10, 0], [i * 10 + 5, 5])
            bbs.append(bb)
            t.insert(bb, data_id=i)

        assert t.delete(bbs[5], data_id=5)
        assert len(t) == 9

    def test_rstar_forced_reinsert(self):
        """With small max_entries, forced reinsert should trigger."""
        t = RStarTree(max_entries=4)
        for i in range(30):
            t.insert(BoundingBox([i * 3, i * 3], [i * 3 + 2, i * 3 + 2]), data_id=i)
        assert len(t) == 30
        all_e = t.all_entries()
        assert len(all_e) == 30

    def test_rstar_3d(self):
        t = RStarTree(max_entries=4, dims=3)
        for i in range(15):
            t.insert(BoundingBox([i, i, i], [i + 2, i + 2, i + 2]), data_id=i)
        assert len(t) == 15
        results = t.search(BoundingBox([0, 0, 0], [5, 5, 5]))
        assert len(results) > 0

    def test_rstar_point_query(self):
        t = RStarTree(max_entries=4)
        t.insert(BoundingBox([0, 0], [10, 10]), data_id="big")
        t.insert(BoundingBox([5, 5], [15, 15]), data_id="shifted")
        results = t.point_query((7, 7))
        ids = {r[1] for r in results}
        assert ids == {"big", "shifted"}

    def test_rstar_contains(self):
        t = RStarTree()
        t.insert(BoundingBox([2, 2], [4, 4]), data_id="small")
        t.insert(BoundingBox([0, 0], [10, 10]), data_id="big")
        results = t.contains(BoundingBox([1, 1], [5, 5]))
        ids = {r[1] for r in results}
        assert "small" in ids

    def test_rstar_many_entries(self):
        t = RStarTree(max_entries=8)
        n = 200
        for i in range(n):
            x = (i * 17) % 500
            y = (i * 31) % 500
            t.insert(BoundingBox([x, y], [x + 5, y + 5]), data_id=i)
        assert len(t) == n

        all_e = t.all_entries()
        assert len(all_e) == n
        ids = {e[1] for e in all_e}
        assert ids == set(range(n))

    def test_rstar_stats(self):
        t = RStarTree(max_entries=4)
        for i in range(20):
            t.insert(BoundingBox([i, i], [i + 1, i + 1]), data_id=i)
        stats = t.depth_stats()
        assert stats['entries'] == 20
        assert stats['fill_factor'] > 0


# ============================================================
# STR Bulk Loading Tests
# ============================================================

class TestSTRBulkLoad:
    def test_empty(self):
        t = str_bulk_load([])
        assert len(t) == 0

    def test_single(self):
        items = [(BoundingBox([0, 0], [5, 5]), "a")]
        t = str_bulk_load(items)
        assert len(t) == 1

    def test_small_batch(self):
        items = [(BoundingBox([i, i], [i + 1, i + 1]), i) for i in range(5)]
        t = str_bulk_load(items, max_entries=8)
        assert len(t) == 5
        all_e = t.all_entries()
        assert len(all_e) == 5

    def test_large_batch(self):
        items = [(BoundingBox([i * 5, 0], [i * 5 + 3, 3]), i) for i in range(100)]
        t = str_bulk_load(items, max_entries=8)
        assert len(t) == 100
        all_e = t.all_entries()
        assert len(all_e) == 100

    def test_search_after_bulk(self):
        items = [(BoundingBox([i * 5, 0], [i * 5 + 3, 3]), i) for i in range(50)]
        t = str_bulk_load(items, max_entries=8)

        results = t.search(BoundingBox([0, 0], [20, 5]))
        assert len(results) > 0

    def test_nearest_after_bulk(self):
        items = [(BoundingBox([i * 10, 0], [i * 10 + 2, 2]), i) for i in range(30)]
        t = str_bulk_load(items, max_entries=8)

        results = t.nearest((0, 0), k=1)
        assert results[0][1] == 0

    def test_bulk_with_tuple_format(self):
        items = [([i, i], [i + 1, i + 1], i) for i in range(10)]
        t = str_bulk_load(items, max_entries=4)
        assert len(t) == 10

    def test_bulk_3d(self):
        items = [(BoundingBox([i, i, i], [i + 1, i + 1, i + 1]), i) for i in range(20)]
        t = str_bulk_load(items, max_entries=4, dims=3)
        assert len(t) == 20

    def test_bulk_quality(self):
        """Bulk loaded tree should have reasonable fill factor."""
        items = [(BoundingBox([i * 3, 0], [i * 3 + 2, 2]), i) for i in range(64)]
        t = str_bulk_load(items, max_entries=8)
        stats = t.depth_stats()
        assert stats['fill_factor'] >= 0.5

    def test_bulk_vs_sequential_completeness(self):
        """Both methods should find all entries."""
        n = 50
        items = [(BoundingBox([i * 5, i * 3], [i * 5 + 4, i * 3 + 4]), i) for i in range(n)]

        t_bulk = str_bulk_load(items, max_entries=8)
        t_seq = RTree(max_entries=8)
        for bbox, data_id in items:
            t_seq.insert(bbox, data_id=data_id)

        bulk_entries = t_bulk.all_entries()
        seq_entries = t_seq.all_entries()
        assert len(bulk_entries) == n
        assert len(seq_entries) == n


# ============================================================
# Spatial Join Tests
# ============================================================

class TestSpatialJoin:
    def test_basic_join(self):
        t1 = RTree()
        t2 = RTree()
        t1.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        t1.insert(BoundingBox([10, 10], [15, 15]), data_id="b")
        t2.insert(BoundingBox([3, 3], [8, 8]), data_id="x")
        t2.insert(BoundingBox([12, 12], [20, 20]), data_id="y")

        pairs = spatial_join(t1, t2)
        pair_ids = {(p[0][1], p[1][1]) for p in pairs}
        assert ("a", "x") in pair_ids
        assert ("b", "y") in pair_ids

    def test_join_no_overlap(self):
        t1 = RTree()
        t2 = RTree()
        t1.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        t2.insert(BoundingBox([10, 10], [15, 15]), data_id="x")
        pairs = spatial_join(t1, t2)
        assert len(pairs) == 0

    def test_join_empty(self):
        t1 = RTree()
        t2 = RTree()
        pairs = spatial_join(t1, t2)
        assert len(pairs) == 0

    def test_join_multi_level(self):
        t1 = RTree(max_entries=4)
        t2 = RTree(max_entries=4)
        for i in range(10):
            t1.insert(BoundingBox([i * 5, 0], [i * 5 + 3, 3]), data_id=f"a{i}")
            t2.insert(BoundingBox([i * 5 + 1, 0], [i * 5 + 4, 3]), data_id=f"b{i}")
        pairs = spatial_join(t1, t2)
        assert len(pairs) > 0

    def test_join_self(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        t.insert(BoundingBox([3, 3], [8, 8]), data_id="b")
        pairs = spatial_join(t, t)
        # Should include all intersecting pairs including self-self
        assert len(pairs) >= 2  # a-a, b-b, a-b, b-a possible


# ============================================================
# SpatialIndex Tests
# ============================================================

class TestSpatialIndex:
    def test_create_rstar(self):
        idx = SpatialIndex(variant='rstar')
        assert len(idx) == 0

    def test_create_classic(self):
        idx = SpatialIndex(variant='classic')
        assert len(idx) == 0

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            SpatialIndex(variant='invalid')

    def test_insert_and_query(self):
        idx = SpatialIndex()
        idx.insert(BoundingBox([0, 0], [5, 5]), "a", data={"name": "rect1"})
        idx.insert(BoundingBox([10, 10], [15, 15]), "b", data={"name": "rect2"})
        assert len(idx) == 2

        results = idx.query(BoundingBox([3, 3], [8, 8]))
        ids = {r[1] for r in results}
        assert "a" in ids

    def test_query_point(self):
        idx = SpatialIndex()
        idx.insert(BoundingBox([0, 0], [10, 10]), "a")
        idx.insert(BoundingBox([20, 20], [30, 30]), "b")
        results = idx.query_point((5, 5))
        assert len(results) == 1
        assert results[0][1] == "a"

    def test_nearest(self):
        idx = SpatialIndex()
        idx.insert(BoundingBox([0, 0], [1, 1]), "close")
        idx.insert(BoundingBox([50, 50], [51, 51]), "far")
        results = idx.nearest((0, 0), k=1)
        assert results[0][1] == "close"

    def test_delete(self):
        idx = SpatialIndex()
        bb = BoundingBox([0, 0], [5, 5])
        idx.insert(bb, "a", data="hello")
        assert idx.delete(bb, "a")
        assert len(idx) == 0

    def test_get_data(self):
        idx = SpatialIndex()
        idx.insert(BoundingBox([0, 0], [5, 5]), "a", data={"key": "value"})
        assert idx.get_data("a") == {"key": "value"}
        assert idx.get_data("nonexistent") is None

    def test_stats(self):
        idx = SpatialIndex()
        for i in range(10):
            idx.insert(BoundingBox([i, i], [i + 1, i + 1]), i)
        stats = idx.stats()
        assert stats['entries'] == 10

    def test_all_entries(self):
        idx = SpatialIndex()
        for i in range(5):
            idx.insert(BoundingBox([i, i], [i + 1, i + 1]), i)
        entries = idx.all_entries()
        assert len(entries) == 5

    def test_insert_with_lists(self):
        idx = SpatialIndex()
        idx.insert(([0, 0], [5, 5]), "a")
        idx.insert((3, 4), "b")  # point
        assert len(idx) == 2


# ============================================================
# Edge Cases and Stress Tests
# ============================================================

class TestEdgeCases:
    def test_degenerate_line_bbox(self):
        bb = BoundingBox([0, 0], [10, 0])
        assert bb.area() == 0

    def test_single_point_bbox(self):
        bb = BoundingBox([5, 5], [5, 5])
        assert bb.area() == 0
        assert bb.contains_point((5, 5))

    def test_identical_entries(self):
        t = RTree(max_entries=4)
        for i in range(10):
            t.insert(BoundingBox([0, 0], [1, 1]), data_id=i)
        assert len(t) == 10
        results = t.search(BoundingBox([0, 0], [1, 1]))
        assert len(results) == 10

    def test_very_large_bbox(self):
        t = RTree()
        t.insert(BoundingBox([-1e9, -1e9], [1e9, 1e9]), data_id="huge")
        results = t.point_query((0, 0))
        assert len(results) == 1

    def test_negative_coordinates(self):
        t = RTree()
        t.insert(BoundingBox([-10, -10], [-5, -5]), data_id="neg")
        results = t.search(BoundingBox([-8, -8], [-3, -3]))
        assert len(results) == 1

    def test_insert_delete_insert(self):
        t = RTree(max_entries=4)
        bb = BoundingBox([0, 0], [5, 5])
        t.insert(bb, data_id="a")
        t.delete(bb, data_id="a")
        t.insert(bb, data_id="b")
        assert len(t) == 1
        results = t.search(bb)
        assert results[0][1] == "b"

    def test_delete_all(self):
        t = RTree(max_entries=4)
        bbs = []
        for i in range(10):
            bb = BoundingBox([i * 10, 0], [i * 10 + 5, 5])
            bbs.append(bb)
            t.insert(bb, data_id=i)

        for i, bb in enumerate(bbs):
            t.delete(bb, data_id=i)
        assert len(t) == 0

    def test_min_entries_constraint(self):
        t = RTree(max_entries=8, min_entries=3)
        assert t.min_entries == 3

    def test_1d_rtree(self):
        t = RTree(dims=1, max_entries=4)
        for i in range(10):
            t.insert(BoundingBox([i * 10], [i * 10 + 5]), data_id=i)
        assert len(t) == 10
        results = t.search(BoundingBox([0], [15]))
        assert len(results) > 0

    def test_high_dim(self):
        t = RTree(dims=5, max_entries=4)
        for i in range(10):
            mins = [i] * 5
            maxs = [i + 1] * 5
            t.insert(BoundingBox(mins, maxs), data_id=i)
        assert len(t) == 10

    def test_sequential_insert_search_delete(self):
        """Full lifecycle: insert, verify, delete, verify."""
        t = RStarTree(max_entries=4)
        bbs = []
        for i in range(20):
            bb = BoundingBox([i * 5, i * 3], [i * 5 + 4, i * 3 + 4])
            bbs.append(bb)
            t.insert(bb, data_id=i)

        assert len(t) == 20

        # Search should find entries
        for i, bb in enumerate(bbs):
            c = bb.center()
            results = t.point_query(c)
            found = any(r[1] == i for r in results)
            assert found, f"Entry {i} not found at center {c}"

        # Delete half
        for i in range(0, 20, 2):
            assert t.delete(bbs[i], data_id=i)

        assert len(t) == 10

        # Remaining should still be findable
        for i in range(1, 20, 2):
            c = bbs[i].center()
            results = t.point_query(c)
            found = any(r[1] == i for r in results)
            assert found

    def test_stress_100_items(self):
        t = RStarTree(max_entries=8)
        bbs = []
        for i in range(100):
            x = (i * 17 + 3) % 200
            y = (i * 31 + 7) % 200
            bb = BoundingBox([x, y], [x + 5, y + 5])
            bbs.append(bb)
            t.insert(bb, data_id=i)

        assert len(t) == 100
        all_e = t.all_entries()
        assert len(all_e) == 100

        # Nearest should work
        results = t.nearest((100, 100), k=5)
        assert len(results) == 5

    def test_touching_bboxes(self):
        t = RTree()
        t.insert(BoundingBox([0, 0], [5, 5]), data_id="a")
        t.insert(BoundingBox([5, 5], [10, 10]), data_id="b")
        # Touching at corner (5,5)
        results = t.search(BoundingBox([5, 5], [5, 5]))
        ids = {r[1] for r in results}
        assert "a" in ids
        assert "b" in ids

    def test_nested_bboxes(self):
        t = RTree(max_entries=4)
        # Concentric squares
        for i in range(1, 6):
            t.insert(BoundingBox([-i, -i], [i, i]), data_id=i)

        results = t.point_query((0, 0))
        assert len(results) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
