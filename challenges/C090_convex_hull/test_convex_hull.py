"""Tests for C090: Convex Hull."""

import math
import pytest
from convex_hull import (
    cross, dist, dist_sq, point_in_convex_polygon, polygon_area, perimeter,
    centroid, graham_scan, monotone_chain, gift_wrapping, quickhull,
    rotating_calipers, diameter, width, min_bounding_rectangle,
    minkowski_sum, ConvexHullTrick, LiChaoTree, DynamicConvexHull,
    half_plane_intersection, convex_hull_of_points, convex_hull_area,
    convex_hull_union, farthest_pair, closest_pair,
    upper_tangent, lower_tangent, EPS
)


# ── Helpers ────────────────────────────────────────────────

def hull_equiv(h1, h2):
    """Check if two hulls represent the same polygon (may start at different vertex)."""
    if len(h1) != len(h2):
        return False
    if not h1:
        return True
    n = len(h1)
    # Find starting point of h1 in h2
    for offset in range(n):
        if all(abs(h1[i][0] - h2[(i + offset) % n][0]) < EPS and
               abs(h1[i][1] - h2[(i + offset) % n][1]) < EPS for i in range(n)):
            return True
    return False


def is_ccw(hull):
    """Check hull is in counter-clockwise order."""
    area = 0
    n = len(hull)
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return area > 0


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ── Primitives ─────────────────────────────────────────────

class TestPrimitives:
    def test_cross_positive(self):
        assert cross((0, 0), (1, 0), (0, 1)) == 1  # CCW

    def test_cross_negative(self):
        assert cross((0, 0), (0, 1), (1, 0)) == -1  # CW

    def test_cross_zero(self):
        assert cross((0, 0), (1, 1), (2, 2)) == 0  # Collinear

    def test_dist_sq(self):
        assert dist_sq((0, 0), (3, 4)) == 25

    def test_dist(self):
        assert approx(dist((0, 0), (3, 4)), 5.0)


class TestPolygonOps:
    def test_area_triangle(self):
        hull = [(0, 0), (4, 0), (0, 3)]
        assert approx(polygon_area(hull), 6.0)

    def test_area_square(self):
        hull = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert approx(polygon_area(hull), 1.0)

    def test_area_empty(self):
        assert polygon_area([]) == 0.0
        assert polygon_area([(1, 2)]) == 0.0
        assert polygon_area([(0, 0), (1, 1)]) == 0.0

    def test_perimeter_triangle(self):
        hull = [(0, 0), (3, 0), (0, 4)]
        assert approx(perimeter(hull), 12.0)

    def test_perimeter_square(self):
        hull = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert approx(perimeter(hull), 4.0)

    def test_centroid_triangle(self):
        hull = [(0, 0), (6, 0), (0, 6)]
        cx, cy = centroid(hull)
        assert approx(cx, 2.0) and approx(cy, 2.0)

    def test_centroid_square(self):
        hull = [(0, 0), (2, 0), (2, 2), (0, 2)]
        cx, cy = centroid(hull)
        assert approx(cx, 1.0) and approx(cy, 1.0)

    def test_centroid_single(self):
        assert centroid([(3, 4)]) == (3, 4)

    def test_point_in_polygon_inside(self):
        hull = [(0, 0), (4, 0), (4, 4), (0, 4)]
        assert point_in_convex_polygon((2, 2), hull) == 1

    def test_point_in_polygon_outside(self):
        hull = [(0, 0), (4, 0), (4, 4), (0, 4)]
        assert point_in_convex_polygon((5, 5), hull) == -1

    def test_point_in_polygon_on_edge(self):
        hull = [(0, 0), (4, 0), (4, 4), (0, 4)]
        assert point_in_convex_polygon((2, 0), hull) == 0

    def test_point_in_polygon_vertex(self):
        hull = [(0, 0), (4, 0), (4, 4), (0, 4)]
        assert point_in_convex_polygon((0, 0), hull) == 0


# ── Graham Scan ────────────────────────────────────────────

class TestGrahamScan:
    def test_triangle(self):
        pts = [(0, 0), (4, 0), (2, 3)]
        hull = graham_scan(pts)
        assert len(hull) == 3
        assert approx(polygon_area(hull), polygon_area(pts))

    def test_square(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        hull = graham_scan(pts)
        assert len(hull) == 4

    def test_with_interior_points(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 1), (3, 3)]
        hull = graham_scan(pts)
        assert len(hull) == 4
        assert approx(polygon_area(hull), 16.0)

    def test_collinear(self):
        pts = [(0, 0), (1, 1), (2, 2), (3, 3)]
        hull = graham_scan(pts)
        assert len(hull) == 2

    def test_single_point(self):
        hull = graham_scan([(5, 5)])
        assert hull == [(5, 5)]

    def test_two_points(self):
        hull = graham_scan([(0, 0), (3, 4)])
        assert len(hull) == 2

    def test_duplicate_points(self):
        pts = [(0, 0), (0, 0), (1, 0), (1, 0), (0, 1), (0, 1)]
        hull = graham_scan(pts)
        assert len(hull) == 3

    def test_large_circle(self):
        n = 100
        pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
        hull = graham_scan(pts)
        assert len(hull) == n
        assert approx(polygon_area(hull), math.pi, tol=0.01)

    def test_ccw_order(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        hull = graham_scan(pts)
        assert is_ccw(hull)


# ── Monotone Chain ─────────────────────────────────────────

class TestMonotoneChain:
    def test_triangle(self):
        pts = [(0, 0), (4, 0), (2, 3)]
        hull = monotone_chain(pts)
        assert len(hull) == 3

    def test_square_with_interior(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 3)]
        hull = monotone_chain(pts)
        assert len(hull) == 4
        assert approx(polygon_area(hull), 16.0)

    def test_ccw(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        hull = monotone_chain(pts)
        assert is_ccw(hull)

    def test_collinear(self):
        pts = [(0, 0), (1, 0), (2, 0), (3, 0)]
        hull = monotone_chain(pts)
        assert len(hull) == 2

    def test_five_points(self):
        pts = [(0, 0), (2, 0), (3, 1), (1, 3), (-1, 1)]
        hull = monotone_chain(pts)
        assert len(hull) == 5

    def test_matches_graham(self):
        pts = [(1, 2), (3, 1), (5, 5), (2, 4), (0, 3), (4, 0)]
        h1 = graham_scan(pts)
        h2 = monotone_chain(pts)
        assert approx(polygon_area(h1), polygon_area(h2))


# ── Gift Wrapping ──────────────────────────────────────────

class TestGiftWrapping:
    def test_triangle(self):
        pts = [(0, 0), (4, 0), (2, 3)]
        hull = gift_wrapping(pts)
        assert len(hull) == 3

    def test_square(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        hull = gift_wrapping(pts)
        assert len(hull) == 4

    def test_with_interior(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        hull = gift_wrapping(pts)
        assert len(hull) == 4
        assert approx(polygon_area(hull), 16.0)

    def test_matches_monotone(self):
        pts = [(1, 0), (2, 1), (1, 3), (-1, 2), (0, -1)]
        h1 = gift_wrapping(pts)
        h2 = monotone_chain(pts)
        assert approx(polygon_area(h1), polygon_area(h2))

    def test_ccw(self):
        pts = [(0, 0), (5, 0), (5, 5), (0, 5), (2, 3)]
        hull = gift_wrapping(pts)
        assert is_ccw(hull)


# ── QuickHull ──────────────────────────────────────────────

class TestQuickHull:
    def test_triangle(self):
        pts = [(0, 0), (4, 0), (2, 3)]
        hull = quickhull(pts)
        assert len(hull) == 3

    def test_square_with_interior(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        hull = quickhull(pts)
        assert len(hull) == 4
        assert approx(polygon_area(hull), 16.0)

    def test_matches_monotone_area(self):
        pts = [(1, 2), (3, 0), (5, 4), (2, 5), (0, 1), (4, 2)]
        h1 = quickhull(pts)
        h2 = monotone_chain(pts)
        assert approx(polygon_area(h1), polygon_area(h2))

    def test_collinear(self):
        pts = [(0, 0), (1, 1), (2, 2)]
        hull = quickhull(pts)
        assert len(hull) == 2

    def test_two_points(self):
        hull = quickhull([(0, 0), (1, 1)])
        assert len(hull) == 2


# ── Rotating Calipers ─────────────────────────────────────

class TestRotatingCalipers:
    def test_square_diameter(self):
        hull = [(0, 0), (1, 0), (1, 1), (0, 1)]
        d = diameter(hull)
        assert approx(d, math.sqrt(2))

    def test_triangle_diameter(self):
        hull = [(0, 0), (4, 0), (2, 3)]
        d = diameter(hull)
        expected = max(dist(hull[i], hull[j]) for i in range(3) for j in range(i + 1, 3))
        assert approx(d, expected)

    def test_width_square(self):
        hull = [(0, 0), (1, 0), (1, 1), (0, 1)]
        w = width(hull)
        assert approx(w, 1.0)

    def test_width_triangle(self):
        hull = [(0, 0), (4, 0), (2, 3)]
        w = width(hull)
        assert w > 0

    def test_antipodal_pairs(self):
        hull = [(0, 0), (2, 0), (2, 2), (0, 2)]
        pairs = rotating_calipers(hull)
        assert len(pairs) >= 2  # At least one pair per edge

    def test_diameter_line(self):
        hull = [(0, 0), (10, 0)]
        assert approx(diameter(hull), 10.0)


# ── Min Bounding Rectangle ────────────────────────────────

class TestMinBoundingRect:
    def test_square(self):
        hull = [(0, 0), (1, 0), (1, 1), (0, 1)]
        area, corners = min_bounding_rectangle(hull)
        assert approx(area, 1.0)

    def test_triangle(self):
        hull = [(0, 0), (4, 0), (2, 3)]
        area, corners = min_bounding_rectangle(hull)
        # MBR of this triangle should be <= bounding box 4*3=12
        assert area <= 12.0 + EPS
        assert len(corners) == 4

    def test_rotated_rect(self):
        # A diamond should have MBR = area of bounding square
        hull = [(1, 0), (2, 1), (1, 2), (0, 1)]
        area, corners = min_bounding_rectangle(hull)
        # The diamond has area 2, its MBR aligned to edges has area 2
        assert approx(area, 2.0)


# ── Minkowski Sum ──────────────────────────────────────────

class TestMinkowskiSum:
    def test_two_triangles(self):
        P = [(0, 0), (1, 0), (0, 1)]
        Q = [(0, 0), (1, 0), (0, 1)]
        result = minkowski_sum(P, Q)
        assert len(result) >= 3
        # Minkowski sum of two unit right triangles: area = 3 * (0.5) + 2 * 0.5 = ?
        # Actually: hexagon-ish shape with area = 2.0
        # Triangle area = 0.5 each, Minkowski sum area = area(P) + area(Q) + perimeter_cross
        # For identical right triangles: area = 0.5 + 0.5 + 1.0 = 2.0
        assert approx(polygon_area(result), 2.0)

    def test_point_plus_polygon(self):
        P = [(3, 4)]
        Q = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = minkowski_sum(P, Q)
        assert len(result) == 4
        # Should just translate Q by (3,4)
        assert approx(polygon_area(result), 1.0)

    def test_two_squares(self):
        P = [(0, 0), (1, 0), (1, 1), (0, 1)]
        Q = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = minkowski_sum(P, Q)
        # Sum of two unit squares = 2x2 square, area = 4
        assert approx(polygon_area(result), 4.0)

    def test_empty(self):
        assert minkowski_sum([], [(1, 2)]) == []


# ── Convex Hull Trick ──────────────────────────────────────

class TestConvexHullTrick:
    def test_basic(self):
        cht = ConvexHullTrick()
        cht.add_line(1, 0)   # y = x
        cht.add_line(0, 5)   # y = 5
        cht.add_line(-1, 10) # y = -x + 10
        assert approx(cht.query(0), 0)    # y=x gives 0
        assert approx(cht.query(3), 3)    # y=x gives 3
        assert approx(cht.query(5), 5)    # y=x and y=5 both give 5

    def test_decreasing_slopes(self):
        cht = ConvexHullTrick()
        # Slopes decreasing: 3, 1, -1
        cht.add_line(3, 0)
        cht.add_line(1, 2)
        cht.add_line(-1, 10)
        # At x=0: min(0, 2, 10) = 0
        assert approx(cht.query(0), 0)
        # At x=5: min(15, 7, 5) = 5
        assert approx(cht.query(5), 5)

    def test_single_line(self):
        cht = ConvexHullTrick()
        cht.add_line(2, 3)
        assert approx(cht.query(4), 11)

    def test_empty(self):
        cht = ConvexHullTrick()
        assert cht.query(0) == float('inf')

    def test_len(self):
        cht = ConvexHullTrick()
        cht.add_line(3, 0)
        cht.add_line(1, 0)
        assert len(cht) >= 1


# ── Li Chao Tree ──────────────────────────────────────────

class TestLiChaoTree:
    def test_basic(self):
        tree = LiChaoTree(-100, 100)
        tree.add_line(1, 0)   # y = x
        tree.add_line(-1, 10) # y = -x + 10
        assert approx(tree.query(0), 0)
        assert approx(tree.query(5), 5)
        assert approx(tree.query(10), 0)  # min(10, 0) = 0

    def test_many_lines(self):
        tree = LiChaoTree(-1000, 1000)
        lines = [(3, 0), (2, 1), (1, 3), (0, 6), (-1, 10)]
        for m, b in lines:
            tree.add_line(m, b)
        # At each x, verify minimum
        for x in [-10, -5, 0, 5, 10]:
            expected = min(m * x + b for m, b in lines)
            assert approx(tree.query(x), expected)

    def test_parallel_lines(self):
        tree = LiChaoTree(-100, 100)
        tree.add_line(1, 0)
        tree.add_line(1, 5)
        assert approx(tree.query(3), 3)  # y=x+0 is always lower

    def test_negative_query(self):
        tree = LiChaoTree(-100, 100)
        tree.add_line(1, 0)
        tree.add_line(-1, 0)
        assert approx(tree.query(-5), -5)
        assert approx(tree.query(5), -5)

    def test_len(self):
        tree = LiChaoTree()
        tree.add_line(1, 0)
        tree.add_line(-1, 0)
        assert len(tree) >= 1


# ── Dynamic Convex Hull ───────────────────────────────────

class TestDynamicConvexHull:
    def test_insert(self):
        dch = DynamicConvexHull()
        dch.insert((0, 0))
        dch.insert((4, 0))
        dch.insert((2, 3))
        hull = dch.get_hull()
        assert len(hull) == 3
        assert approx(polygon_area(hull), 6.0)

    def test_insert_interior(self):
        dch = DynamicConvexHull()
        for p in [(0, 0), (4, 0), (4, 4), (0, 4)]:
            dch.insert(p)
        dch.insert((2, 2))  # Interior point
        hull = dch.get_hull()
        assert len(hull) == 4

    def test_contains(self):
        dch = DynamicConvexHull()
        for p in [(0, 0), (4, 0), (4, 4), (0, 4)]:
            dch.insert(p)
        assert dch.contains((2, 2))
        assert dch.contains((0, 0))  # vertex
        assert dch.contains((2, 0))  # edge
        assert not dch.contains((5, 5))

    def test_remove(self):
        dch = DynamicConvexHull()
        for p in [(0, 0), (4, 0), (4, 4), (0, 4)]:
            dch.insert(p)
        assert approx(dch.area(), 16.0)
        dch.remove((4, 4))
        hull = dch.get_hull()
        assert len(hull) == 3
        assert approx(dch.area(), 8.0)

    def test_empty(self):
        dch = DynamicConvexHull()
        assert dch.get_hull() == []
        assert not dch.contains((0, 0))
        assert len(dch) == 0

    def test_single_point(self):
        dch = DynamicConvexHull()
        dch.insert((3, 4))
        assert dch.contains((3, 4))
        assert not dch.contains((3, 5))

    def test_two_points(self):
        dch = DynamicConvexHull()
        dch.insert((0, 0))
        dch.insert((4, 0))
        assert dch.contains((2, 0))
        assert not dch.contains((2, 1))


# ── Half-Plane Intersection ───────────────────────────────

class TestHalfPlaneIntersection:
    def test_single_plane(self):
        # x <= 5
        result = half_plane_intersection([(1, 0, 5)], bounds=(-10, -10, 10, 10))
        assert len(result) >= 3
        for p in result:
            assert p[0] <= 5 + EPS

    def test_box(self):
        # x >= 0, x <= 4, y >= 0, y <= 4
        planes = [(-1, 0, 0), (1, 0, 4), (0, -1, 0), (0, 1, 4)]
        result = half_plane_intersection(planes, bounds=(-10, -10, 10, 10))
        assert len(result) == 4
        assert approx(polygon_area(result), 16.0)

    def test_triangle(self):
        # y >= 0, x >= 0, x + y <= 4
        planes = [(0, -1, 0), (-1, 0, 0), (1, 1, 4)]
        result = half_plane_intersection(planes, bounds=(-10, -10, 10, 10))
        assert len(result) == 3
        assert approx(polygon_area(result), 8.0)

    def test_empty_intersection(self):
        # x <= -1 and x >= 1 -> empty
        planes = [(1, 0, -1), (-1, 0, -1)]
        result = half_plane_intersection(planes, bounds=(-10, -10, 10, 10))
        assert len(result) == 0


# ── Convex Hull Area / Union ──────────────────────────────

class TestHullArea:
    def test_area(self):
        pts = [(0, 0), (4, 0), (4, 3), (0, 3)]
        assert approx(convex_hull_area(pts), 12.0)

    def test_of_points(self):
        pts = [(0, 0), (1, 0), (0.5, 0.5), (1, 1), (0, 1)]
        hull = convex_hull_of_points(pts)
        assert len(hull) == 4

    def test_union(self):
        h1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
        h2 = [(1, 1), (3, 1), (3, 3), (1, 3)]
        hull = convex_hull_union(h1, h2)
        assert polygon_area(hull) > max(polygon_area(h1), polygon_area(h2))


# ── Farthest / Closest Pair ──────────────────────────────

class TestFarthestPair:
    def test_basic(self):
        pts = [(0, 0), (3, 4), (1, 1)]
        d, p1, p2 = farthest_pair(pts)
        assert approx(d, 5.0)

    def test_square(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        d, p1, p2 = farthest_pair(pts)
        assert approx(d, math.sqrt(2))


class TestClosestPair:
    def test_basic(self):
        pts = [(0, 0), (3, 4), (1, 0)]
        d, p1, p2 = closest_pair(pts)
        assert approx(d, 1.0)

    def test_many_points(self):
        pts = [(i, 0) for i in range(10)]
        d, p1, p2 = closest_pair(pts)
        assert approx(d, 1.0)

    def test_close_points(self):
        pts = [(0, 0), (0.1, 0), (1, 1), (5, 5)]
        d, p1, p2 = closest_pair(pts)
        assert approx(d, 0.1)

    def test_two_points(self):
        d, p1, p2 = closest_pair([(0, 0), (3, 4)])
        assert approx(d, 5.0)


# ── Tangent Lines ─────────────────────────────────────────

class TestTangentLines:
    def test_upper_tangent(self):
        h1 = [(0, 0), (1, 0), (1, 2), (0, 2)]
        h2 = [(3, 0), (4, 0), (4, 2), (3, 2)]
        i, j = upper_tangent(h1, h2)
        # Upper tangent should connect top of h1 to top of h2
        assert hull_point_approx(h1[i], (1, 2)) or hull_point_approx(h1[i], (0, 2))
        assert hull_point_approx(h2[j], (3, 2)) or hull_point_approx(h2[j], (4, 2))

    def test_lower_tangent(self):
        h1 = [(0, 0), (1, 0), (1, 2), (0, 2)]
        h2 = [(3, 0), (4, 0), (4, 2), (3, 2)]
        i, j = lower_tangent(h1, h2)
        assert hull_point_approx(h1[i], (1, 0)) or hull_point_approx(h1[i], (0, 0))
        assert hull_point_approx(h2[j], (3, 0)) or hull_point_approx(h2[j], (4, 0))


def hull_point_approx(a, b, tol=1e-6):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


# ── Algorithm Agreement ───────────────────────────────────

class TestAgreement:
    """All four algorithms should produce the same hull (same area, same points)."""

    def _check_all(self, pts):
        h1 = graham_scan(pts)
        h2 = monotone_chain(pts)
        h3 = gift_wrapping(pts)
        h4 = quickhull(pts)
        a1 = polygon_area(h1)
        a2 = polygon_area(h2)
        a3 = polygon_area(h3)
        a4 = polygon_area(h4)
        assert approx(a1, a2), f"graham {a1} != monotone {a2}"
        assert approx(a1, a3), f"graham {a1} != gift {a3}"
        assert approx(a1, a4), f"graham {a1} != quickhull {a4}"
        return a1

    def test_triangle(self):
        self._check_all([(0, 0), (4, 0), (2, 3)])

    def test_square(self):
        self._check_all([(0, 0), (1, 0), (1, 1), (0, 1)])

    def test_pentagon(self):
        pts = [(math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5)) for i in range(5)]
        self._check_all(pts)

    def test_with_interior(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5), (3, 7), (7, 3), (2, 8)]
        a = self._check_all(pts)
        assert approx(a, 100.0)

    def test_random_like(self):
        # Deterministic "random-like" points
        pts = [((i * 7 + 3) % 20, (i * 13 + 5) % 20) for i in range(30)]
        self._check_all(pts)

    def test_grid(self):
        pts = [(i, j) for i in range(5) for j in range(5)]
        a = self._check_all(pts)
        assert approx(a, 16.0)

    def test_circle(self):
        n = 20
        pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
        self._check_all(pts)

    def test_star_shape(self):
        pts = []
        for i in range(5):
            a = 2 * math.pi * i / 5
            pts.append((5 * math.cos(a), 5 * math.sin(a)))
            pts.append((2 * math.cos(a + math.pi / 5), 2 * math.sin(a + math.pi / 5)))
        self._check_all(pts)


# ── Edge Cases ─────────────────────────────────────────────

class TestEdgeCases:
    def test_all_same_point(self):
        pts = [(3, 3)] * 10
        for fn in [graham_scan, monotone_chain, gift_wrapping, quickhull]:
            hull = fn(pts)
            assert len(hull) == 1

    def test_two_distinct_points(self):
        pts = [(0, 0), (5, 0)]
        for fn in [graham_scan, monotone_chain, gift_wrapping, quickhull]:
            hull = fn(pts)
            assert len(hull) == 2

    def test_collinear_many(self):
        pts = [(i, i) for i in range(10)]
        for fn in [graham_scan, monotone_chain]:
            hull = fn(pts)
            assert len(hull) == 2

    def test_negative_coords(self):
        pts = [(-5, -3), (2, -1), (4, 5), (-3, 4)]
        for fn in [graham_scan, monotone_chain, gift_wrapping, quickhull]:
            hull = fn(pts)
            assert len(hull) == 4

    def test_very_close_points(self):
        pts = [(0, 0), (1e-10, 0), (0, 1e-10), (1, 1)]
        hull = monotone_chain(pts)
        assert len(hull) >= 2


# ── Performance Smoke Test ─────────────────────────────────

class TestPerformance:
    def test_monotone_1000_points(self):
        pts = [((i * 997 + 7) % 10000, (i * 991 + 13) % 10000) for i in range(1000)]
        hull = monotone_chain(pts)
        assert len(hull) >= 3
        assert polygon_area(hull) > 0

    def test_graham_1000_points(self):
        pts = [((i * 997 + 7) % 10000, (i * 991 + 13) % 10000) for i in range(1000)]
        hull = graham_scan(pts)
        assert len(hull) >= 3

    def test_dynamic_hull_50_inserts(self):
        dch = DynamicConvexHull()
        for i in range(50):
            dch.insert(((i * 7) % 100, (i * 13) % 100))
        assert len(dch.get_hull()) >= 3

    def test_lichao_100_lines(self):
        tree = LiChaoTree(-10000, 10000)
        for i in range(100):
            tree.add_line(i - 50, (i * 7) % 100)
        val = tree.query(0)
        assert val < float('inf')


# ── Integration: Hull + Operations ─────────────────────────

class TestIntegration:
    def test_hull_then_diameter(self):
        pts = [(0, 0), (3, 0), (3, 4), (0, 4), (1, 2)]
        hull = monotone_chain(pts)
        d = diameter(hull)
        assert approx(d, 5.0)

    def test_hull_then_centroid(self):
        pts = [(0, 0), (6, 0), (6, 6), (0, 6)]
        hull = monotone_chain(pts)
        cx, cy = centroid(hull)
        assert approx(cx, 3.0) and approx(cy, 3.0)

    def test_hull_then_point_test(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]
        hull = monotone_chain(pts)
        assert point_in_convex_polygon((5, 5), hull) == 1
        assert point_in_convex_polygon((11, 5), hull) == -1
        assert point_in_convex_polygon((5, 0), hull) == 0

    def test_dynamic_hull_area_grows(self):
        dch = DynamicConvexHull()
        dch.insert((0, 0))
        dch.insert((1, 0))
        dch.insert((0, 1))
        a1 = dch.area()
        dch.insert((1, 1))
        a2 = dch.area()
        assert a2 >= a1

    def test_minkowski_then_point_test(self):
        P = [(0, 0), (1, 0), (1, 1), (0, 1)]
        Q = [(0, 0), (1, 0), (0, 1)]
        msum = minkowski_sum(P, Q)
        # Point (0.5, 0.5) should be inside Minkowski sum
        assert point_in_convex_polygon((0.5, 0.5), msum) >= 0

    def test_cht_dp_pattern(self):
        """Convex hull trick for DP: dp[i] = min(dp[j] + cost(j, i))."""
        # Simulate: dp[i] = min over j of (a[j]*i + dp[j])
        n = 10
        a = [n - i for i in range(n)]  # decreasing slopes
        dp = [0] * n
        cht = ConvexHullTrick()
        cht.add_line(a[0], dp[0])
        for i in range(1, n):
            dp[i] = cht.query(i) + i * i  # some cost
            cht.add_line(a[i], dp[i])
        # Just verify it ran without error and values are finite
        assert all(math.isfinite(d) for d in dp)

    def test_farthest_closest_inverse(self):
        pts = [(0, 0), (1, 0), (2, 0), (3, 0), (1.5, 2)]
        fd, _, _ = farthest_pair(pts)
        cd, _, _ = closest_pair(pts)
        assert fd > cd

    def test_hull_perimeter(self):
        pts = [(0, 0), (3, 0), (3, 4), (0, 4)]
        hull = monotone_chain(pts)
        p = perimeter(hull)
        assert approx(p, 14.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
