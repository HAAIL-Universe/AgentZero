"""Tests for C088: KD-Tree"""

import math
import pytest
import random
from kd_tree import (
    KDTree, KDNode, BallTree, SpatialIndex,
    euclidean_distance, euclidean_distance_sq,
    manhattan_distance, chebyshev_distance,
)


# ===================================================================
# Distance functions
# ===================================================================

class TestDistanceFunctions:
    def test_euclidean_distance_sq_2d(self):
        assert euclidean_distance_sq((0, 0), (3, 4)) == 25

    def test_euclidean_distance_2d(self):
        assert euclidean_distance((0, 0), (3, 4)) == 5.0

    def test_euclidean_distance_3d(self):
        d = euclidean_distance((1, 2, 3), (4, 6, 3))
        assert abs(d - 5.0) < 1e-10

    def test_euclidean_same_point(self):
        assert euclidean_distance((5, 5), (5, 5)) == 0.0

    def test_manhattan_distance(self):
        assert manhattan_distance((0, 0), (3, 4)) == 7

    def test_manhattan_3d(self):
        assert manhattan_distance((1, 2, 3), (4, 6, 3)) == 7

    def test_chebyshev_distance(self):
        assert chebyshev_distance((0, 0), (3, 4)) == 4

    def test_chebyshev_3d(self):
        assert chebyshev_distance((1, 2, 3), (4, 6, 3)) == 4


# ===================================================================
# KDTree -- construction
# ===================================================================

class TestKDTreeConstruction:
    def test_empty_tree(self):
        tree = KDTree()
        assert tree.size == 0
        assert tree.root is None

    def test_single_point(self):
        tree = KDTree([(3, 4)])
        assert tree.size == 1
        assert tree.root.point == (3, 4)

    def test_build_from_points(self):
        points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
        tree = KDTree(points)
        assert tree.size == 6
        assert tree.dimensions == 2

    def test_build_from_points_with_data(self):
        points = [((2, 3), "a"), ((5, 4), "b"), ((9, 6), "c")]
        tree = KDTree(points)
        assert tree.size == 3

    def test_3d_construction(self):
        points = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (2, 1, 5)]
        tree = KDTree(points)
        assert tree.dimensions == 3
        assert tree.size == 4

    def test_explicit_dimensions(self):
        tree = KDTree(dimensions=5)
        assert tree.dimensions == 5

    def test_balanced_depth(self):
        points = [(i, i) for i in range(15)]
        tree = KDTree(points)
        # Balanced tree of 15 nodes should have depth ~4
        assert tree.depth() <= 5

    def test_duplicate_points(self):
        points = [(1, 1), (1, 1), (2, 2)]
        tree = KDTree(points)
        assert tree.size == 3


# ===================================================================
# KDTree -- insert
# ===================================================================

class TestKDTreeInsert:
    def test_insert_into_empty(self):
        tree = KDTree()
        tree.insert((3, 4))
        assert tree.size == 1
        assert tree.contains((3, 4))

    def test_insert_multiple(self):
        tree = KDTree()
        for i in range(10):
            tree.insert((i, i * 2))
        assert tree.size == 10

    def test_insert_with_data(self):
        tree = KDTree()
        tree.insert((3, 4), data="hello")
        result = tree.nearest((3, 4))
        assert result == ((3, 4), "hello")

    def test_insert_preserves_search(self):
        tree = KDTree([(1, 1), (5, 5), (9, 9)])
        tree.insert((3, 3))
        result = tree.nearest((3, 2))
        assert result[0] == (3, 3)


# ===================================================================
# KDTree -- delete
# ===================================================================

class TestKDTreeDelete:
    def test_delete_existing(self):
        tree = KDTree([(1, 1), (2, 2), (3, 3)])
        assert tree.delete((2, 2))
        assert tree.size == 2
        assert not tree.contains((2, 2))

    def test_delete_nonexistent(self):
        tree = KDTree([(1, 1)])
        assert not tree.delete((9, 9))
        assert tree.size == 1

    def test_delete_then_nearest(self):
        tree = KDTree([(1, 1), (2, 2), (3, 3)])
        tree.delete((2, 2))
        result = tree.nearest((2, 2))
        # Should find (1,1) or (3,3), not (2,2)
        assert result[0] != (2, 2)

    def test_delete_all(self):
        pts = [(1, 1), (2, 2), (3, 3)]
        tree = KDTree(pts)
        for p in pts:
            tree.delete(p)
        assert tree.size == 0

    def test_delete_double(self):
        tree = KDTree([(1, 1), (2, 2)])
        assert tree.delete((1, 1))
        assert not tree.delete((1, 1))


# ===================================================================
# KDTree -- contains
# ===================================================================

class TestKDTreeContains:
    def test_contains_existing(self):
        tree = KDTree([(1, 2), (3, 4), (5, 6)])
        assert tree.contains((3, 4))

    def test_not_contains(self):
        tree = KDTree([(1, 2), (3, 4)])
        assert not tree.contains((9, 9))

    def test_contains_after_delete(self):
        tree = KDTree([(1, 1)])
        tree.delete((1, 1))
        assert not tree.contains((1, 1))

    def test_contains_empty(self):
        tree = KDTree()
        assert not tree.contains((1, 1))


# ===================================================================
# KDTree -- nearest neighbor
# ===================================================================

class TestKDTreeNearest:
    def test_nearest_exact(self):
        tree = KDTree([(1, 1), (5, 5), (9, 9)])
        result = tree.nearest((5, 5))
        assert result[0] == (5, 5)

    def test_nearest_close(self):
        tree = KDTree([(0, 0), (10, 10), (20, 20)])
        result = tree.nearest((1, 1))
        assert result[0] == (0, 0)

    def test_nearest_with_distance(self):
        tree = KDTree([(0, 0), (3, 4)])
        result = tree.nearest((0, 0), return_distance=True)
        assert result[0] == (0, 0)
        assert result[2] == 0.0

    def test_nearest_returns_data(self):
        tree = KDTree([((1, 1), "first"), ((5, 5), "second")])
        result = tree.nearest((4, 4))
        assert result == ((5, 5), "second")

    def test_nearest_empty(self):
        tree = KDTree()
        assert tree.nearest((1, 1)) is None

    def test_nearest_single_point(self):
        tree = KDTree([(7, 7)])
        result = tree.nearest((0, 0))
        assert result[0] == (7, 7)

    def test_nearest_3d(self):
        tree = KDTree([(0, 0, 0), (10, 10, 10), (5, 5, 5)])
        result = tree.nearest((4, 4, 4))
        assert result[0] == (5, 5, 5)

    def test_nearest_negative_coords(self):
        tree = KDTree([(-5, -5), (5, 5), (0, 0)])
        result = tree.nearest((-4, -4))
        assert result[0] == (-5, -5)

    def test_nearest_brute_force_validation(self):
        """Validate against brute force for random points."""
        random.seed(42)
        points = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(50)]
        tree = KDTree(points)

        for _ in range(20):
            query = (random.uniform(-100, 100), random.uniform(-100, 100))
            kd_result = tree.nearest(query, return_distance=True)

            # Brute force
            bf_dist = min(euclidean_distance(query, p) for p in points)
            assert abs(kd_result[2] - bf_dist) < 1e-10


# ===================================================================
# KDTree -- k-nearest neighbors
# ===================================================================

class TestKDTreeKNN:
    def test_knn_basic(self):
        points = [(i, 0) for i in range(10)]
        tree = KDTree(points)
        results = tree.k_nearest((0, 0), 3)
        result_pts = [r[0] for r in results]
        assert (0, 0) in result_pts
        assert (1, 0) in result_pts
        assert (2, 0) in result_pts

    def test_knn_with_distances(self):
        tree = KDTree([(0, 0), (3, 4), (6, 8)])
        results = tree.k_nearest((0, 0), 2, return_distances=True)
        assert results[0][0] == (0, 0)
        assert results[0][2] == 0.0
        assert results[1][0] == (3, 4)
        assert abs(results[1][2] - 5.0) < 1e-10

    def test_knn_sorted_by_distance(self):
        tree = KDTree([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        results = tree.k_nearest((0, 0), 5, return_distances=True)
        distances = [r[2] for r in results]
        assert distances == sorted(distances)

    def test_knn_k_larger_than_size(self):
        tree = KDTree([(0, 0), (1, 1)])
        results = tree.k_nearest((0, 0), 5)
        assert len(results) == 2

    def test_knn_empty(self):
        tree = KDTree()
        assert tree.k_nearest((0, 0), 3) == []

    def test_knn_brute_force_validation(self):
        random.seed(123)
        points = [(random.uniform(-50, 50), random.uniform(-50, 50)) for _ in range(40)]
        tree = KDTree(points)

        query = (0, 0)
        k = 5
        kd_results = tree.k_nearest(query, k, return_distances=True)

        # Brute force k-nearest
        all_dists = sorted((euclidean_distance(query, p), p) for p in points)
        bf_dists = [d for d, _ in all_dists[:k]]
        kd_dists = [r[2] for r in kd_results]

        for a, b in zip(kd_dists, bf_dists):
            assert abs(a - b) < 1e-10

    def test_knn_k1_matches_nearest(self):
        random.seed(77)
        points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(30)]
        tree = KDTree(points)
        query = (5, 5)

        nn = tree.nearest(query, return_distance=True)
        knn = tree.k_nearest(query, 1, return_distances=True)

        assert nn[0] == knn[0][0]
        assert abs(nn[2] - knn[0][2]) < 1e-10


# ===================================================================
# KDTree -- range search
# ===================================================================

class TestKDTreeRangeSearch:
    def test_range_basic(self):
        points = [(i, j) for i in range(5) for j in range(5)]
        tree = KDTree(points)
        results = tree.range_search((1, 1), (3, 3))
        result_pts = {r[0] for r in results}
        expected = {(i, j) for i in range(1, 4) for j in range(1, 4)}
        assert result_pts == expected

    def test_range_empty_result(self):
        tree = KDTree([(0, 0), (1, 1)])
        results = tree.range_search((5, 5), (10, 10))
        assert results == []

    def test_range_single_point(self):
        tree = KDTree([(3, 3)])
        results = tree.range_search((2, 2), (4, 4))
        assert len(results) == 1
        assert results[0][0] == (3, 3)

    def test_range_boundary_inclusive(self):
        tree = KDTree([(1, 1), (2, 2), (3, 3)])
        results = tree.range_search((1, 1), (3, 3))
        assert len(results) == 3

    def test_range_3d(self):
        points = [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]
        tree = KDTree(points)
        results = tree.range_search((1, 1, 1), (2, 2, 2))
        assert len(results) == 8  # 2x2x2 cube

    def test_range_with_data(self):
        tree = KDTree([((1, 1), "a"), ((2, 2), "b"), ((3, 3), "c")])
        results = tree.range_search((0, 0), (2, 2))
        data_set = {r[1] for r in results}
        assert data_set == {"a", "b"}


# ===================================================================
# KDTree -- radius search
# ===================================================================

class TestKDTreeRadiusSearch:
    def test_radius_basic(self):
        tree = KDTree([(0, 0), (1, 0), (0, 1), (5, 5)])
        results = tree.radius_search((0, 0), 1.5)
        result_pts = {r[0] for r in results}
        assert (0, 0) in result_pts
        assert (1, 0) in result_pts
        assert (0, 1) in result_pts
        assert (5, 5) not in result_pts

    def test_radius_with_distance(self):
        tree = KDTree([(3, 4)])
        results = tree.radius_search((0, 0), 6.0, return_distances=True)
        assert len(results) == 1
        assert abs(results[0][2] - 5.0) < 1e-10

    def test_radius_empty(self):
        tree = KDTree([(0, 0)])
        results = tree.radius_search((100, 100), 1.0)
        assert results == []

    def test_radius_zero(self):
        tree = KDTree([(1, 1), (2, 2)])
        results = tree.radius_search((1, 1), 0.0)
        assert len(results) == 1
        assert results[0][0] == (1, 1)

    def test_radius_brute_force(self):
        random.seed(99)
        points = [(random.uniform(-20, 20), random.uniform(-20, 20)) for _ in range(50)]
        tree = KDTree(points)
        center = (0, 0)
        radius = 10.0

        kd_results = tree.radius_search(center, radius)
        kd_pts = {r[0] for r in kd_results}

        bf_pts = {p for p in [tuple(pt) for pt in points]
                  if euclidean_distance(center, p) <= radius}

        assert kd_pts == bf_pts


# ===================================================================
# KDTree -- rebalance / points
# ===================================================================

class TestKDTreeMisc:
    def test_points_returns_all(self):
        pts = [(1, 2), (3, 4), (5, 6)]
        tree = KDTree(pts)
        result = tree.points()
        result_pts = {r[0] for r in result}
        assert result_pts == {(1, 2), (3, 4), (5, 6)}

    def test_points_excludes_deleted(self):
        tree = KDTree([(1, 1), (2, 2), (3, 3)])
        tree.delete((2, 2))
        result_pts = {r[0] for r in tree.points()}
        assert result_pts == {(1, 1), (3, 3)}

    def test_rebalance(self):
        tree = KDTree()
        # Inserting in order creates an unbalanced tree
        for i in range(15):
            tree.insert((i, i))
        depth_before = tree.depth()
        tree.rebalance()
        depth_after = tree.depth()
        assert depth_after <= depth_before
        assert tree.size == 15

    def test_rebalance_after_delete(self):
        tree = KDTree([(i, i) for i in range(10)])
        for i in range(5):
            tree.delete((i, i))
        tree.rebalance()
        assert tree.size == 5
        # All remaining points should still be findable
        for i in range(5, 10):
            assert tree.contains((i, i))

    def test_depth_empty(self):
        tree = KDTree()
        assert tree.depth() == 0

    def test_depth_single(self):
        tree = KDTree([(1, 1)])
        assert tree.depth() == 1


# ===================================================================
# KDTree -- custom distance
# ===================================================================

class TestKDTreeCustomDistance:
    def test_manhattan_nearest(self):
        tree = KDTree([(0, 0), (3, 0), (0, 3)], distance_fn=manhattan_distance)
        result = tree.nearest((2, 0), return_distance=True)
        # Manhattan: (2,0)->(0,0)=2, (2,0)->(3,0)=1, (2,0)->(0,3)=5
        assert result[0] == (3, 0)
        assert result[2] == 1

    def test_chebyshev_nearest(self):
        tree = KDTree([(0, 0), (2, 1), (1, 2)], distance_fn=chebyshev_distance)
        result = tree.nearest((3, 3), return_distance=True)
        # Chebyshev: (3,3)->(0,0)=3, (3,3)->(2,1)=2, (3,3)->(1,2)=2
        assert result[2] == 2


# ===================================================================
# BallTree -- construction
# ===================================================================

class TestBallTreeConstruction:
    def test_empty(self):
        tree = BallTree()
        assert tree.size == 0

    def test_single_point(self):
        tree = BallTree([(1, 2)])
        assert tree.size == 1

    def test_build(self):
        points = [(i, j) for i in range(5) for j in range(5)]
        tree = BallTree(points)
        assert tree.size == 25

    def test_with_data(self):
        tree = BallTree([((1, 1), "a"), ((2, 2), "b")])
        assert tree.size == 2

    def test_3d(self):
        points = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        tree = BallTree(points)
        assert tree.dimensions == 3


# ===================================================================
# BallTree -- nearest
# ===================================================================

class TestBallTreeNearest:
    def test_nearest_basic(self):
        tree = BallTree([(0, 0), (5, 5), (10, 10)])
        result = tree.nearest((4, 4))
        assert result[0] == (5, 5)

    def test_nearest_exact(self):
        tree = BallTree([(1, 1), (2, 2)])
        result = tree.nearest((1, 1))
        assert result[0] == (1, 1)

    def test_nearest_with_distance(self):
        tree = BallTree([(0, 0), (3, 4)])
        result = tree.nearest((0, 0), return_distance=True)
        assert result[0] == (0, 0)
        assert result[2] == 0.0

    def test_nearest_empty(self):
        tree = BallTree()
        assert tree.nearest((0, 0)) is None

    def test_nearest_brute_force(self):
        random.seed(42)
        points = [(random.uniform(-50, 50), random.uniform(-50, 50)) for _ in range(60)]
        tree = BallTree(points)

        for _ in range(15):
            query = (random.uniform(-50, 50), random.uniform(-50, 50))
            bt_result = tree.nearest(query, return_distance=True)
            bf_dist = min(euclidean_distance(query, p) for p in points)
            assert abs(bt_result[2] - bf_dist) < 1e-10


# ===================================================================
# BallTree -- k-nearest
# ===================================================================

class TestBallTreeKNN:
    def test_knn_basic(self):
        points = [(i, 0) for i in range(10)]
        tree = BallTree(points)
        results = tree.k_nearest((0, 0), 3)
        result_pts = [r[0] for r in results]
        assert (0, 0) in result_pts

    def test_knn_sorted(self):
        tree = BallTree([(0, 0), (1, 1), (2, 2), (3, 3)])
        results = tree.k_nearest((0, 0), 4, return_distances=True)
        dists = [r[2] for r in results]
        assert dists == sorted(dists)

    def test_knn_brute_force(self):
        random.seed(55)
        points = [(random.uniform(-30, 30), random.uniform(-30, 30)) for _ in range(40)]
        tree = BallTree(points)
        query = (0, 0)
        k = 5

        bt_results = tree.k_nearest(query, k, return_distances=True)
        all_dists = sorted(euclidean_distance(query, p) for p in points)
        bt_dists = [r[2] for r in bt_results]

        for a, b in zip(bt_dists, all_dists[:k]):
            assert abs(a - b) < 1e-10


# ===================================================================
# BallTree -- radius search
# ===================================================================

class TestBallTreeRadius:
    def test_radius_basic(self):
        tree = BallTree([(0, 0), (1, 0), (5, 5)])
        results = tree.radius_search((0, 0), 1.5)
        pts = {r[0] for r in results}
        assert (0, 0) in pts
        assert (1, 0) in pts
        assert (5, 5) not in pts

    def test_radius_with_distance(self):
        tree = BallTree([(3, 4)])
        results = tree.radius_search((0, 0), 6.0, return_distances=True)
        assert len(results) == 1
        assert abs(results[0][2] - 5.0) < 1e-10

    def test_radius_brute_force(self):
        random.seed(88)
        points = [(random.uniform(-20, 20), random.uniform(-20, 20)) for _ in range(50)]
        tree = BallTree(points)
        center = (0, 0)
        radius = 10.0

        bt_results = tree.radius_search(center, radius)
        bt_pts = {r[0] for r in bt_results}
        bf_pts = {tuple(p) for p in points if euclidean_distance(center, p) <= radius}

        assert bt_pts == bf_pts

    def test_points_collection(self):
        pts = [(1, 1), (2, 2), (3, 3)]
        tree = BallTree(pts)
        result = tree.points()
        result_pts = {r[0] for r in result}
        assert result_pts == {(1, 1), (2, 2), (3, 3)}


# ===================================================================
# SpatialIndex -- kdtree backend
# ===================================================================

class TestSpatialIndexKDTree:
    def test_create_empty(self):
        idx = SpatialIndex(dimensions=2)
        assert idx.size == 0

    def test_create_with_points(self):
        idx = SpatialIndex([(1, 2), (3, 4), (5, 6)])
        assert idx.size == 3

    def test_insert(self):
        idx = SpatialIndex(dimensions=2)
        idx.insert((1, 2))
        idx.insert((3, 4))
        assert idx.size == 2

    def test_delete(self):
        idx = SpatialIndex([(1, 1), (2, 2)])
        idx.delete((1, 1))
        assert idx.size == 1

    def test_nearest(self):
        idx = SpatialIndex([(0, 0), (5, 5), (10, 10)])
        result = idx.nearest((4, 4))
        assert result[0] == (5, 5)

    def test_knn(self):
        idx = SpatialIndex([(i, 0) for i in range(10)])
        results = idx.k_nearest((0, 0), 3)
        assert len(results) == 3

    def test_range_search(self):
        idx = SpatialIndex([(i, i) for i in range(10)])
        results = idx.range_search((2, 2), (5, 5))
        pts = {r[0] for r in results}
        assert pts == {(2, 2), (3, 3), (4, 4), (5, 5)}

    def test_radius_search(self):
        idx = SpatialIndex([(0, 0), (1, 0), (10, 10)])
        results = idx.radius_search((0, 0), 1.5)
        assert len(results) == 2

    def test_contains(self):
        idx = SpatialIndex([(1, 1), (2, 2)])
        assert idx.contains((1, 1))
        assert not idx.contains((9, 9))

    def test_rebalance(self):
        idx = SpatialIndex(dimensions=2)
        for i in range(20):
            idx.insert((i, i))
        idx.rebalance()
        assert idx.size == 20

    def test_bounding_box(self):
        idx = SpatialIndex([(1, 2), (5, 8), (3, 4)])
        bb = idx.bounding_box()
        assert bb == ((1, 2), (5, 8))

    def test_bounding_box_empty(self):
        idx = SpatialIndex(dimensions=2)
        assert idx.bounding_box() is None

    def test_bounding_box_3d(self):
        idx = SpatialIndex([(1, 2, 3), (4, 5, 6), (0, 0, 0)])
        bb = idx.bounding_box()
        assert bb == ((0, 0, 0), (4, 5, 6))

    def test_points(self):
        idx = SpatialIndex([(1, 1), (2, 2)])
        pts = {r[0] for r in idx.points()}
        assert pts == {(1, 1), (2, 2)}


# ===================================================================
# SpatialIndex -- balltree backend
# ===================================================================

class TestSpatialIndexBallTree:
    def test_create(self):
        idx = SpatialIndex([(1, 2), (3, 4)], backend='balltree')
        assert idx.size == 2

    def test_nearest(self):
        idx = SpatialIndex([(0, 0), (5, 5), (10, 10)], backend='balltree')
        result = idx.nearest((4, 4))
        assert result[0] == (5, 5)

    def test_knn(self):
        idx = SpatialIndex([(i, 0) for i in range(10)], backend='balltree')
        results = idx.k_nearest((0, 0), 3, return_distances=True)
        assert len(results) == 3

    def test_radius(self):
        idx = SpatialIndex([(0, 0), (1, 0), (10, 10)], backend='balltree')
        results = idx.radius_search((0, 0), 1.5)
        assert len(results) == 2

    def test_insert_raises(self):
        idx = SpatialIndex([(1, 1)], backend='balltree')
        with pytest.raises(NotImplementedError):
            idx.insert((2, 2))

    def test_delete_raises(self):
        idx = SpatialIndex([(1, 1)], backend='balltree')
        with pytest.raises(NotImplementedError):
            idx.delete((1, 1))

    def test_range_raises(self):
        idx = SpatialIndex([(1, 1)], backend='balltree')
        with pytest.raises(NotImplementedError):
            idx.range_search((0, 0), (2, 2))

    def test_points(self):
        idx = SpatialIndex([(1, 1), (2, 2)], backend='balltree')
        pts = {r[0] for r in idx.points()}
        assert pts == {(1, 1), (2, 2)}

    def test_bounding_box(self):
        idx = SpatialIndex([(0, 0), (5, 5)], backend='balltree')
        bb = idx.bounding_box()
        assert bb == ((0, 0), (5, 5))


# ===================================================================
# SpatialIndex -- all-pairs nearest
# ===================================================================

class TestAllPairsNearest:
    def test_basic(self):
        idx = SpatialIndex([(0, 0), (1, 0), (10, 0)])
        results = idx.all_pairs_nearest()
        assert len(results) == 3
        # (0,0) nearest is (1,0), (1,0) nearest is (0,0), (10,0) nearest is (1,0)
        result_map = {r[0]: (r[1], r[2]) for r in results}
        assert result_map[(0, 0)][0] == (1, 0)
        assert result_map[(1, 0)][0] == (0, 0)
        assert result_map[(10, 0)][0] == (1, 0)

    def test_all_pairs_distances(self):
        idx = SpatialIndex([(0, 0), (3, 4)])
        results = idx.all_pairs_nearest()
        for _, _, dist in results:
            assert abs(dist - 5.0) < 1e-10


# ===================================================================
# SpatialIndex -- convex hull 2D
# ===================================================================

class TestConvexHull2D:
    def test_triangle(self):
        idx = SpatialIndex([(0, 0), (1, 0), (0, 1)])
        hull = idx.convex_hull_2d()
        assert len(hull) == 3
        assert set(hull) == {(0, 0), (1, 0), (0, 1)}

    def test_square(self):
        idx = SpatialIndex([(0, 0), (1, 0), (1, 1), (0, 1)])
        hull = idx.convex_hull_2d()
        assert len(hull) == 4

    def test_collinear(self):
        idx = SpatialIndex([(0, 0), (1, 0), (2, 0)])
        hull = idx.convex_hull_2d()
        # Collinear points: hull is just endpoints
        assert len(hull) == 2
        assert set(hull) == {(0, 0), (2, 0)}

    def test_with_interior_points(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5), (3, 3), (7, 7)]
        idx = SpatialIndex(pts)
        hull = idx.convex_hull_2d()
        assert len(hull) == 4
        assert set(hull) == {(0, 0), (10, 0), (10, 10), (0, 10)}

    def test_single_point(self):
        idx = SpatialIndex([(5, 5)])
        hull = idx.convex_hull_2d()
        assert hull == [(5, 5)]

    def test_empty(self):
        idx = SpatialIndex(dimensions=2)
        hull = idx.convex_hull_2d()
        assert hull == []

    def test_duplicate_points(self):
        idx = SpatialIndex([(0, 0), (0, 0), (1, 0), (0, 1)])
        hull = idx.convex_hull_2d()
        assert len(hull) == 3

    def test_many_points(self):
        """Hull of circle should have many vertices."""
        random.seed(42)
        pts = []
        for i in range(100):
            angle = 2 * math.pi * i / 100
            pts.append((math.cos(angle) * 10, math.sin(angle) * 10))
        # Add some interior points
        for _ in range(50):
            r = random.uniform(0, 5)
            a = random.uniform(0, 2 * math.pi)
            pts.append((r * math.cos(a), r * math.sin(a)))

        idx = SpatialIndex(pts)
        hull = idx.convex_hull_2d()
        # Circle has 100 boundary points, all should be on hull
        assert len(hull) >= 90  # Some might be collinear


# ===================================================================
# High-dimensional tests
# ===================================================================

class TestHighDimensional:
    def test_5d_construction(self):
        random.seed(42)
        points = [tuple(random.uniform(0, 10) for _ in range(5)) for _ in range(20)]
        tree = KDTree(points)
        assert tree.size == 20
        assert tree.dimensions == 5

    def test_5d_nearest(self):
        random.seed(42)
        points = [tuple(random.uniform(0, 10) for _ in range(5)) for _ in range(30)]
        tree = KDTree(points)
        query = (5, 5, 5, 5, 5)
        result = tree.nearest(query, return_distance=True)

        bf_dist = min(euclidean_distance(query, p) for p in points)
        assert abs(result[2] - bf_dist) < 1e-10

    def test_10d_knn(self):
        random.seed(42)
        dims = 10
        points = [tuple(random.uniform(0, 1) for _ in range(dims)) for _ in range(50)]
        tree = KDTree(points)
        query = tuple(0.5 for _ in range(dims))

        results = tree.k_nearest(query, 5, return_distances=True)
        assert len(results) == 5

        # Verify sorted
        dists = [r[2] for r in results]
        assert dists == sorted(dists)

    def test_5d_balltree(self):
        random.seed(42)
        points = [tuple(random.uniform(0, 10) for _ in range(5)) for _ in range(30)]
        tree = BallTree(points)
        query = (5, 5, 5, 5, 5)
        result = tree.nearest(query, return_distance=True)

        bf_dist = min(euclidean_distance(query, p) for p in points)
        assert abs(result[2] - bf_dist) < 1e-10

    def test_high_dim_range_search(self):
        random.seed(42)
        dims = 4
        points = [tuple(random.uniform(0, 10) for _ in range(dims)) for _ in range(30)]
        tree = KDTree(points)

        min_pt = (3, 3, 3, 3)
        max_pt = (7, 7, 7, 7)
        results = tree.range_search(min_pt, max_pt)

        # Brute force verify
        bf = [p for p in points if all(3 <= p[i] <= 7 for i in range(dims))]
        assert len(results) == len(bf)


# ===================================================================
# Edge cases and stress tests
# ===================================================================

class TestEdgeCases:
    def test_all_same_point(self):
        pts = [(5, 5)] * 10
        tree = KDTree(pts)
        result = tree.nearest((5, 5))
        assert result[0] == (5, 5)

    def test_all_same_x(self):
        pts = [(0, i) for i in range(10)]
        tree = KDTree(pts)
        result = tree.nearest((0, 4.5))
        assert result[0] in [(0, 4), (0, 5)]

    def test_all_same_y(self):
        pts = [(i, 0) for i in range(10)]
        tree = KDTree(pts)
        result = tree.nearest((4.5, 0))
        assert result[0] in [(4, 0), (5, 0)]

    def test_large_coordinates(self):
        pts = [(1e10, 1e10), (-1e10, -1e10), (0, 0)]
        tree = KDTree(pts)
        result = tree.nearest((1, 1))
        assert result[0] == (0, 0)

    def test_float_precision(self):
        tree = KDTree([(0.1 + 0.2, 0.3)])
        # Should still be findable despite float precision
        result = tree.nearest((0.3, 0.3))
        assert result is not None

    def test_1d_tree(self):
        pts = [(i,) for i in range(10)]
        tree = KDTree(pts)
        assert tree.dimensions == 1
        result = tree.nearest((4.5,))
        assert result[0] in [(4,), (5,)]

    def test_insert_then_search_many(self):
        tree = KDTree(dimensions=2)
        random.seed(42)
        points = []
        for _ in range(100):
            p = (random.uniform(-100, 100), random.uniform(-100, 100))
            points.append(p)
            tree.insert(p)

        # Verify all points findable
        for p in points:
            assert tree.contains(p)

    def test_stress_knn(self):
        """Stress test k-NN against brute force."""
        random.seed(42)
        points = [(random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(200)]
        tree = KDTree(points)

        for _ in range(10):
            query = (random.uniform(-100, 100), random.uniform(-100, 100))
            k = 10
            kd = tree.k_nearest(query, k, return_distances=True)

            bf_dists = sorted(euclidean_distance(query, p) for p in points)[:k]
            kd_dists = [r[2] for r in kd]

            for a, b in zip(kd_dists, bf_dists):
                assert abs(a - b) < 1e-9


# ===================================================================
# KDTree with lazy delete interaction tests
# ===================================================================

class TestDeleteInteractions:
    def test_delete_then_knn(self):
        tree = KDTree([(0, 0), (1, 0), (2, 0), (3, 0)])
        tree.delete((1, 0))
        results = tree.k_nearest((0, 0), 2)
        pts = {r[0] for r in results}
        assert (1, 0) not in pts
        assert (0, 0) in pts

    def test_delete_then_range(self):
        tree = KDTree([(1, 1), (2, 2), (3, 3)])
        tree.delete((2, 2))
        results = tree.range_search((0, 0), (4, 4))
        pts = {r[0] for r in results}
        assert (2, 2) not in pts
        assert len(pts) == 2

    def test_delete_then_radius(self):
        tree = KDTree([(0, 0), (1, 0)])
        tree.delete((0, 0))
        results = tree.radius_search((0, 0), 0.5)
        assert len(results) == 0

    def test_rebalance_removes_deleted(self):
        tree = KDTree([(i, i) for i in range(10)])
        tree.delete((3, 3))
        tree.delete((7, 7))
        tree.rebalance()
        assert tree.size == 8
        assert not tree.contains((3, 3))
        assert not tree.contains((7, 7))
        assert tree.contains((5, 5))
