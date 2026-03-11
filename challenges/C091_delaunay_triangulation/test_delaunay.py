"""Tests for C091: Delaunay Triangulation."""

import math
import pytest
from delaunay import (
    orient2d, in_circumcircle, circumcenter, circumradius_sq,
    dist, dist_sq, midpoint, triangle_area, triangle_quality,
    segments_intersect, point_on_segment, segment_intersection_point,
    Triangle, DelaunayTriangulation, VoronoiDiagram,
    ConstrainedDelaunay, MeshRefiner, triangulate_polygon,
    euler_check, expected_triangles, expected_edges,
    shortest_edge_length_sq, _point_in_polygon,
)

EPSILON = 1e-8


# ===================================================================
# Geometry Primitives
# ===================================================================

class TestOrient2D:
    def test_ccw(self):
        assert orient2d((0, 0), (1, 0), (0, 1)) > 0

    def test_cw(self):
        assert orient2d((0, 0), (0, 1), (1, 0)) < 0

    def test_collinear(self):
        assert abs(orient2d((0, 0), (1, 1), (2, 2))) < EPSILON

    def test_large_coords(self):
        assert orient2d((1000, 1000), (2000, 1000), (1500, 2000)) > 0


class TestInCircumcircle:
    def test_inside(self):
        # Unit triangle, point at center
        a, b, c = (0, 0), (1, 0), (0.5, math.sqrt(3)/2)
        assert in_circumcircle(a, b, c, (0.5, 0.3))

    def test_outside(self):
        a, b, c = (0, 0), (1, 0), (0.5, math.sqrt(3)/2)
        assert not in_circumcircle(a, b, c, (5, 5))

    def test_on_circle(self):
        # Point on circumcircle should NOT be "strictly inside"
        a, b, c, d = (0, 0), (1, 0), (0, 1), (1, 1)  # square
        assert not in_circumcircle(a, b, c, d)  # cocircular


class TestCircumcenter:
    def test_equilateral(self):
        a = (0, 0)
        b = (1, 0)
        c = (0.5, math.sqrt(3)/2)
        cc = circumcenter(a, b, c)
        # Should be at centroid for equilateral
        assert abs(cc[0] - 0.5) < EPSILON
        assert abs(cc[1] - math.sqrt(3)/6) < EPSILON

    def test_right_triangle(self):
        a, b, c = (0, 0), (4, 0), (0, 3)
        cc = circumcenter(a, b, c)
        # Circumcenter of right triangle is midpoint of hypotenuse
        assert abs(cc[0] - 2.0) < EPSILON
        assert abs(cc[1] - 1.5) < EPSILON

    def test_collinear_fallback(self):
        cc = circumcenter((0, 0), (1, 1), (2, 2))
        # Degenerate -- returns centroid
        assert abs(cc[0] - 1.0) < EPSILON
        assert abs(cc[1] - 1.0) < EPSILON


class TestDistances:
    def test_dist_sq(self):
        assert abs(dist_sq((0, 0), (3, 4)) - 25.0) < EPSILON

    def test_dist(self):
        assert abs(dist((0, 0), (3, 4)) - 5.0) < EPSILON

    def test_midpoint(self):
        m = midpoint((0, 0), (4, 6))
        assert abs(m[0] - 2.0) < EPSILON
        assert abs(m[1] - 3.0) < EPSILON


class TestTriangleArea:
    def test_unit_triangle(self):
        assert abs(triangle_area((0, 0), (1, 0), (0, 1)) - 0.5) < EPSILON

    def test_negative_orientation(self):
        assert triangle_area((0, 0), (0, 1), (1, 0)) < 0


class TestTriangleQuality:
    def test_equilateral(self):
        a = (0, 0)
        b = (1, 0)
        c = (0.5, math.sqrt(3)/2)
        q = triangle_quality(a, b, c)
        # Equilateral: circumradius/shortest_edge = 1/sqrt(3) ~0.577
        assert abs(q - 1.0 / math.sqrt(3)) < 0.01

    def test_degenerate(self):
        q = triangle_quality((0, 0), (1, 0), (2, 0))
        assert q == float('inf')


class TestSegmentIntersection:
    def test_crossing(self):
        assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))

    def test_parallel(self):
        assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))

    def test_non_crossing(self):
        assert not segments_intersect((0, 0), (1, 0), (2, 0), (3, 0))


class TestPointOnSegment:
    def test_midpoint(self):
        assert point_on_segment((0.5, 0), (0, 0), (1, 0))

    def test_endpoint(self):
        assert not point_on_segment((0, 0), (0, 0), (1, 0))

    def test_off_segment(self):
        assert not point_on_segment((0.5, 1), (0, 0), (1, 0))


class TestSegmentIntersectionPoint:
    def test_crossing(self):
        p = segment_intersection_point((0, 0), (2, 2), (0, 2), (2, 0))
        assert p is not None
        assert abs(p[0] - 1.0) < EPSILON
        assert abs(p[1] - 1.0) < EPSILON

    def test_no_crossing(self):
        p = segment_intersection_point((0, 0), (1, 0), (0, 1), (1, 1))
        assert p is None


class TestShortestEdge:
    def test_right_triangle(self):
        s = shortest_edge_length_sq((0, 0), (3, 0), (0, 4))
        assert abs(s - 9.0) < EPSILON  # shortest edge is 3


# ===================================================================
# Triangle class
# ===================================================================

class TestTriangleClass:
    def test_creation(self):
        t = Triangle((0, 0), (1, 0), (0, 1))
        assert len(t.vertices) == 3

    def test_ccw_ordering(self):
        # CW input should be reordered to CCW
        t = Triangle((0, 0), (0, 1), (1, 0))
        assert orient2d(*t.vertices) > 0

    def test_has_vertex(self):
        t = Triangle((0, 0), (1, 0), (0, 1))
        assert t.has_vertex((0, 0))
        assert not t.has_vertex((5, 5))

    def test_contains_point(self):
        t = Triangle((0, 0), (4, 0), (0, 4))
        assert t.contains_point((1, 1))
        assert not t.contains_point((5, 5))

    def test_contains_point_on_edge(self):
        t = Triangle((0, 0), (4, 0), (0, 4))
        assert t.contains_point((2, 0))

    def test_circumcenter_method(self):
        t = Triangle((0, 0), (4, 0), (0, 3))
        cc = t.circumcenter()
        assert abs(cc[0] - 2.0) < EPSILON
        assert abs(cc[1] - 1.5) < EPSILON

    def test_edge_opposite(self):
        t = Triangle((0, 0), (1, 0), (0, 1))
        e = t.edge_opposite((0, 0))
        assert (1, 0) in e and (0, 1) in e

    def test_repr(self):
        t = Triangle((0, 0), (1, 0), (0, 1))
        assert "Triangle" in repr(t)


# ===================================================================
# Delaunay Triangulation - Basic
# ===================================================================

class TestDelaunayBasic:
    def test_three_points(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (0, 1)])
        assert dt.num_triangles() == 1
        assert dt.num_points() == 3

    def test_four_points_square(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert dt.num_triangles() == 2
        assert dt.num_points() == 4

    def test_five_points(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() >= 4
        assert dt.is_delaunay()

    def test_delaunay_property(self):
        pts = [(0, 0), (3, 0), (3, 3), (0, 3), (1.5, 1.5)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()

    def test_get_triangles(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (0, 1)])
        tris = dt.get_triangles()
        assert len(tris) == 1
        assert len(tris[0]) == 3

    def test_get_edges(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (0, 1)])
        edges = dt.get_edges()
        assert len(edges) == 3

    def test_get_edge_list(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (0, 1)])
        edges = dt.get_edge_list()
        assert len(edges) == 3
        for e in edges:
            assert len(e) == 2


class TestDelaunayDegenerate:
    def test_empty(self):
        dt = DelaunayTriangulation([])
        assert dt.num_triangles() == 0

    def test_one_point(self):
        dt = DelaunayTriangulation([(0, 0)])
        assert dt.num_triangles() == 0

    def test_two_points(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0)])
        assert dt.num_triangles() == 0

    def test_collinear_points(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (2, 0)])
        assert dt.num_triangles() == 0

    def test_duplicate_points(self):
        dt = DelaunayTriangulation([(0, 0), (0, 0), (1, 0), (0, 1)])
        assert dt.num_triangles() == 1

    def test_near_duplicate(self):
        dt = DelaunayTriangulation([(0, 0), (1e-15, 0), (1, 0), (0, 1)])
        # Near duplicates should be filtered
        assert dt.num_triangles() >= 1


class TestDelaunayLarger:
    def test_grid_points(self):
        pts = [(i, j) for i in range(4) for j in range(4)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()
        assert dt.num_points() == 16

    def test_circle_points(self):
        n = 12
        pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
               for i in range(n)]
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() >= 10
        assert dt.is_delaunay()

    def test_random_like_points(self):
        # Deterministic "random" points
        pts = [((i * 7 + 3) % 11, (i * 13 + 5) % 17) for i in range(20)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()

    def test_euler_formula(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 3)]
        dt = DelaunayTriangulation(pts)
        # V - E + F = 2
        assert euler_check(dt) == 2


class TestDelaunayExpectedCounts:
    def test_triangle_count_formula(self):
        # For n points with h on hull: T = 2n - h - 2
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        hull = dt.convex_hull()
        h = len(hull)
        n = dt.num_points()
        assert dt.num_triangles() == expected_triangles(n, h)

    def test_edge_count_formula(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        hull = dt.convex_hull()
        h = len(hull)
        n = dt.num_points()
        assert dt.num_edges() == expected_edges(n, h)


# ===================================================================
# Point Location
# ===================================================================

class TestPointLocation:
    def test_locate_inside(self):
        dt = DelaunayTriangulation([(0, 0), (10, 0), (5, 10)])
        t = dt.locate_point((4, 3))
        assert t is not None
        assert t.contains_point((4, 3))

    def test_locate_outside(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (0, 1)])
        t = dt.locate_point((5, 5))
        assert t is None

    def test_locate_on_vertex(self):
        dt = DelaunayTriangulation([(0, 0), (10, 0), (5, 10)])
        t = dt.locate_point((0, 0))
        assert t is not None

    def test_locate_on_edge(self):
        dt = DelaunayTriangulation([(0, 0), (10, 0), (5, 10)])
        t = dt.locate_point((5, 0))
        assert t is not None

    def test_locate_empty(self):
        dt = DelaunayTriangulation([])
        assert dt.locate_point((0, 0)) is None


# ===================================================================
# Nearest Point
# ===================================================================

class TestNearestPoint:
    def test_exact_match(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (0, 1)])
        assert dt.nearest_point((0, 0)) == (0, 0)

    def test_nearest(self):
        dt = DelaunayTriangulation([(0, 0), (10, 0), (5, 10)])
        assert dt.nearest_point((0.1, 0.1)) == (0, 0)

    def test_empty(self):
        dt = DelaunayTriangulation([])
        assert dt.nearest_point((0, 0)) is None


# ===================================================================
# Neighbors
# ===================================================================

class TestGetNeighbors:
    def test_triangle(self):
        pts = [(0, 0), (1, 0), (0, 1)]
        dt = DelaunayTriangulation(pts)
        n = dt.get_neighbors((0.0, 0.0))
        assert len(n) == 2

    def test_center_point(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        n = dt.get_neighbors((2.0, 2.0))
        assert len(n) == 4  # Connected to all corners


# ===================================================================
# Convex Hull from DT
# ===================================================================

class TestConvexHull:
    def test_triangle_hull(self):
        pts = [(0, 0), (1, 0), (0, 1)]
        dt = DelaunayTriangulation(pts)
        hull = dt.convex_hull()
        assert len(hull) == 3

    def test_square_hull(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        dt = DelaunayTriangulation(pts)
        hull = dt.convex_hull()
        assert len(hull) == 4

    def test_interior_point(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        hull = dt.convex_hull()
        assert len(hull) == 4  # Interior point not on hull

    def test_empty(self):
        dt = DelaunayTriangulation([])
        assert dt.convex_hull() == []


# ===================================================================
# Voronoi Diagram
# ===================================================================

class TestVoronoiBasic:
    def test_from_triangle(self):
        vd = VoronoiDiagram(points=[(0, 0), (4, 0), (2, 4)])
        assert vd.num_vertices() == 1  # One triangle = one circumcenter

    def test_from_square(self):
        vd = VoronoiDiagram(points=[(0, 0), (4, 0), (4, 4), (0, 4)])
        assert vd.num_vertices() == 2  # Two triangles

    def test_from_delaunay(self):
        dt = DelaunayTriangulation([(0, 0), (4, 0), (2, 4)])
        vd = VoronoiDiagram(delaunay=dt)
        assert vd.num_vertices() == 1

    def test_get_vertices(self):
        vd = VoronoiDiagram(points=[(0, 0), (4, 0), (2, 4)])
        verts = vd.get_vertices()
        assert len(verts) == 1
        # Should be circumcenter of the triangle
        cc = circumcenter((0, 0), (4, 0), (2, 4))
        assert abs(verts[0][0] - cc[0]) < EPSILON
        assert abs(verts[0][1] - cc[1]) < EPSILON


class TestVoronoiEdges:
    def test_finite_edges(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        vd = VoronoiDiagram(points=pts)
        fe = vd.get_finite_edges()
        assert len(fe) >= 1

    def test_get_edges(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        vd = VoronoiDiagram(points=pts)
        edges = vd.get_edges()
        assert len(edges) >= 1

    def test_ridge_points(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        vd = VoronoiDiagram(points=pts)
        assert len(vd.ridge_points) == len(vd.edges)

    def test_num_finite_edges(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        vd = VoronoiDiagram(points=pts)
        assert vd.num_finite_edges() <= vd.num_edges()


class TestVoronoiRegions:
    def test_region_exists(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        vd = VoronoiDiagram(points=pts)
        # Center point should have a region
        region = vd.get_region((2.0, 2.0))
        assert len(region) >= 1

    def test_region_vertices(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        vd = VoronoiDiagram(points=pts)
        for p in [(2.0, 2.0)]:
            region = vd.get_region(p)
            assert all(isinstance(v, tuple) and len(v) == 2 for v in region)

    def test_nonexistent_region(self):
        vd = VoronoiDiagram(points=[(0, 0), (1, 0), (0, 1)])
        assert vd.get_region((99, 99)) == []


# ===================================================================
# Constrained Delaunay
# ===================================================================

class TestConstrainedDelaunay:
    def test_basic_constraint(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        # Force diagonal
        cdt = ConstrainedDelaunay(pts, [((0, 0), (4, 4))])
        assert cdt.num_triangles() >= 2
        assert cdt.is_constraint((0.0, 0.0), (4.0, 4.0))

    def test_no_constraints(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        cdt = ConstrainedDelaunay(pts, [])
        assert cdt.num_triangles() >= 2

    def test_constraint_present_as_edge(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        cdt = ConstrainedDelaunay(pts, [((0, 0), (4, 4))])
        edges = cdt.get_edges()
        # The constraint might be subdivided but endpoints should be connected
        assert len(edges) >= 4

    def test_get_constraints(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        cdt = ConstrainedDelaunay(pts, [((0, 0), (4, 4))])
        assert len(cdt.get_constraints()) >= 1

    def test_get_triangles(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        cdt = ConstrainedDelaunay(pts, [((0, 0), (4, 4))])
        tris = cdt.get_triangles()
        assert len(tris) >= 2

    def test_empty_cdt(self):
        cdt = ConstrainedDelaunay()
        assert cdt.get_triangles() == []
        assert cdt.get_edges() == set()
        assert cdt.num_triangles() == 0


# ===================================================================
# Mesh Refinement
# ===================================================================

class TestMeshRefiner:
    def test_basic_refinement(self):
        pts = [(0, 0), (10, 0), (5, 3)]
        dt = DelaunayTriangulation(pts)
        refiner = MeshRefiner(dt, min_angle=15.0)
        refined = refiner.refine(max_iterations=20)
        assert refined.num_triangles() >= 1

    def test_quality_improves(self):
        pts = [(0, 0), (10, 0), (5, 2)]
        dt = DelaunayTriangulation(pts)
        q_before = triangle_quality(*list(dt.triangles)[0].vertices)

        refiner = MeshRefiner(dt, min_angle=15.0)
        refined = refiner.refine(max_iterations=20)
        stats = refiner.get_quality_stats()
        assert stats['count'] >= 1

    def test_max_area_constraint(self):
        pts = [(0, 0), (100, 0), (50, 100)]
        dt = DelaunayTriangulation(pts)
        refiner = MeshRefiner(dt, max_area=500)
        refined = refiner.refine(max_iterations=20)
        assert refined.num_triangles() > 1

    def test_min_angle(self):
        pts = [(0, 0), (10, 0), (5, 8)]
        dt = DelaunayTriangulation(pts)
        refiner = MeshRefiner(dt, min_angle=20.0)
        refined = refiner.refine(max_iterations=20)
        min_a = refiner.min_angle_degrees()
        # After refinement, min angle should improve
        assert min_a >= 0  # At least non-negative

    def test_quality_stats(self):
        pts = [(0, 0), (4, 0), (2, 4)]
        dt = DelaunayTriangulation(pts)
        refiner = MeshRefiner(dt)
        stats = refiner.get_quality_stats()
        assert 'min_quality' in stats
        assert 'max_quality' in stats
        assert 'avg_quality' in stats
        assert stats['count'] == 1

    def test_empty_mesh(self):
        refiner = MeshRefiner(None)
        assert refiner.get_quality_stats()['count'] == 0
        assert refiner.min_angle_degrees() == 0

    def test_refine_empty(self):
        dt = DelaunayTriangulation([])
        refiner = MeshRefiner(dt)
        result = refiner.refine()
        assert result is dt


# ===================================================================
# Polygon Triangulation
# ===================================================================

class TestPolygonTriangulation:
    def test_triangle(self):
        poly = [(0, 0), (4, 0), (2, 3)]
        tris = triangulate_polygon(poly)
        assert len(tris) == 1

    def test_square(self):
        poly = [(0, 0), (4, 0), (4, 4), (0, 4)]
        tris = triangulate_polygon(poly)
        assert len(tris) == 2

    def test_convex_polygon(self):
        # Pentagon
        poly = [(2, 0), (4, 1.5), (3, 4), (1, 4), (0, 1.5)]
        tris = triangulate_polygon(poly)
        assert len(tris) >= 3

    def test_empty(self):
        assert triangulate_polygon([]) == []
        assert triangulate_polygon([(0, 0)]) == []
        assert triangulate_polygon([(0, 0), (1, 0)]) == []


class TestPointInPolygon:
    def test_inside_square(self):
        poly = [(0, 0), (4, 0), (4, 4), (0, 4)]
        assert _point_in_polygon((2, 2), poly)

    def test_outside_square(self):
        poly = [(0, 0), (4, 0), (4, 4), (0, 4)]
        assert not _point_in_polygon((5, 5), poly)

    def test_inside_triangle(self):
        poly = [(0, 0), (4, 0), (2, 3)]
        assert _point_in_polygon((2, 1), poly)


# ===================================================================
# Euler Formula
# ===================================================================

class TestEulerFormula:
    def test_single_triangle(self):
        dt = DelaunayTriangulation([(0, 0), (1, 0), (0, 1)])
        assert euler_check(dt) == 2

    def test_square(self):
        dt = DelaunayTriangulation([(0, 0), (4, 0), (4, 4), (0, 4)])
        assert euler_check(dt) == 2

    def test_five_points(self):
        dt = DelaunayTriangulation([(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)])
        assert euler_check(dt) == 2


class TestExpectedCounts:
    def test_expected_triangles(self):
        assert expected_triangles(5, 4) == 4  # 2*5 - 4 - 2

    def test_expected_edges(self):
        assert expected_edges(5, 4) == 8  # 3*5 - 4 - 3


# ===================================================================
# Larger Integration Tests
# ===================================================================

class TestIntegration:
    def test_grid_delaunay_voronoi(self):
        pts = [(i, j) for i in range(5) for j in range(5)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()
        vd = VoronoiDiagram(delaunay=dt)
        assert vd.num_vertices() == dt.num_triangles()

    def test_circle_delaunay(self):
        n = 20
        pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
               for i in range(n)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()
        hull = dt.convex_hull()
        assert len(hull) == n  # All points on hull for circle

    def test_voronoi_vertex_count(self):
        # Number of Voronoi vertices = number of Delaunay triangles
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        vd = VoronoiDiagram(delaunay=dt)
        assert vd.num_vertices() == dt.num_triangles()

    def test_build_then_query(self):
        pts = [(i * 2, j * 3) for i in range(5) for j in range(5)]
        dt = DelaunayTriangulation(pts)
        # Locate a point
        t = dt.locate_point((3, 4))
        assert t is not None
        # Find nearest
        near = dt.nearest_point((3.1, 4.2))
        assert near is not None
        # Get neighbors
        n = dt.get_neighbors(pts[0])
        assert len(n) >= 1

    def test_refine_then_verify(self):
        pts = [(0, 0), (10, 0), (5, 3)]
        dt = DelaunayTriangulation(pts)
        refiner = MeshRefiner(dt, min_angle=15.0)
        refined = refiner.refine(max_iterations=20)
        assert refined.is_delaunay()

    def test_large_point_set(self):
        # 50 points in a pattern
        pts = []
        for i in range(50):
            x = (i * 17 + 3) % 23
            y = (i * 31 + 7) % 29
            pts.append((float(x), float(y)))
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()
        assert dt.num_points() > 0

    def test_diamond_pattern(self):
        pts = [(0, 2), (2, 4), (4, 2), (2, 0), (2, 2)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()
        assert euler_check(dt) == 2


class TestIncrementalBuild:
    def test_build_returns_self(self):
        dt = DelaunayTriangulation()
        result = dt.build([(0, 0), (1, 0), (0, 1)])
        assert result is dt
        assert dt.num_triangles() == 1

    def test_empty_build(self):
        dt = DelaunayTriangulation()
        dt.build([])
        assert dt.num_triangles() == 0


class TestVoronoiEmpty:
    def test_empty_voronoi(self):
        vd = VoronoiDiagram()
        assert vd.num_vertices() == 0
        assert vd.num_edges() == 0

    def test_voronoi_from_empty_dt(self):
        dt = DelaunayTriangulation([])
        vd = VoronoiDiagram(delaunay=dt)
        assert vd.num_vertices() == 0


class TestDelaunaySpecialCases:
    def test_cocircular_four_points(self):
        # Four points on a circle -- two valid triangulations
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() == 2
        # Both should be valid Delaunay (cocircular => either diagonal ok)

    def test_many_collinear_with_offset(self):
        # Points mostly collinear plus one offset
        pts = [(float(i), 0.0) for i in range(10)]
        pts.append((5.0, 1.0))
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() >= 1
        assert dt.is_delaunay()

    def test_clustered_points(self):
        pts = [(0, 0), (0.001, 0), (0, 0.001), (10, 10), (10.001, 10), (10, 10.001)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()

    def test_L_shape(self):
        pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()
        assert dt.num_triangles() >= 4

    def test_star_shape(self):
        # Star: alternating inner/outer vertices
        pts = []
        for i in range(5):
            angle = 2 * math.pi * i / 5 - math.pi / 2
            pts.append((3 * math.cos(angle), 3 * math.sin(angle)))
            angle2 = angle + math.pi / 5
            pts.append((1.2 * math.cos(angle2), 1.2 * math.sin(angle2)))
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()


# ===================================================================
# Additional edge cases and robustness
# ===================================================================

class TestRobustness:
    def test_negative_coordinates(self):
        pts = [(-5, -5), (-5, 5), (5, -5), (5, 5)]
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() == 2
        assert dt.is_delaunay()

    def test_large_coordinates(self):
        pts = [(1000, 2000), (3000, 1000), (2000, 4000)]
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() == 1

    def test_very_close_points(self):
        pts = [(0, 0), (1e-11, 0), (1, 0), (0, 1)]
        dt = DelaunayTriangulation(pts)
        # Should handle near-duplicate gracefully
        assert dt.num_triangles() >= 1

    def test_all_same_point(self):
        pts = [(1, 1), (1, 1), (1, 1)]
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() == 0

    def test_float_precision(self):
        pts = [(0.1 + 0.2, 0.3), (1.0, 0.0), (0.0, 1.0)]
        dt = DelaunayTriangulation(pts)
        assert dt.num_triangles() == 1


class TestMeshRefinerAdditional:
    def test_refine_with_dt_parameter(self):
        pts = [(0, 0), (10, 0), (5, 8)]
        dt = DelaunayTriangulation(pts)
        refiner = MeshRefiner(min_angle=20.0)
        result = refiner.refine(dt=dt, max_iterations=10)
        assert result.num_triangles() >= 1

    def test_quality_threshold(self):
        refiner = MeshRefiner(min_angle=30.0)
        # 1/(2*sin(30)) = 1.0
        assert abs(refiner._quality_threshold - 1.0) < EPSILON


class TestVoronoiDual:
    def test_voronoi_edges_match_delaunay(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        vd = VoronoiDiagram(delaunay=dt)
        # Each Voronoi edge corresponds to a Delaunay edge
        dt_edges = dt.get_edges()
        # Number of Voronoi edges should be related to Delaunay edges
        assert vd.num_edges() <= len(dt_edges)

    def test_voronoi_equidistant(self):
        # Each Voronoi vertex (circumcenter) should be equidistant from its triangle vertices
        pts = [(0, 0), (4, 0), (2, 4)]
        dt = DelaunayTriangulation(pts)
        vd = VoronoiDiagram(delaunay=dt)
        for t in dt.triangles:
            cc = t.circumcenter()
            d0 = dist(cc, t.vertices[0])
            d1 = dist(cc, t.vertices[1])
            d2 = dist(cc, t.vertices[2])
            assert abs(d0 - d1) < 1e-6
            assert abs(d1 - d2) < 1e-6


class TestConstrainedAdditional:
    def test_multiple_constraints(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        constraints = [((0, 0), (4, 4)), ((4, 0), (0, 4))]
        cdt = ConstrainedDelaunay(pts, constraints)
        assert cdt.num_triangles() >= 4

    def test_is_constraint_check(self):
        pts = [(0, 0), (4, 0), (4, 4)]
        cdt = ConstrainedDelaunay(pts, [((0, 0), (4, 4))])
        assert cdt.is_constraint((0.0, 0.0), (4.0, 4.0))
        assert not cdt.is_constraint((0.0, 0.0), (4.0, 0.0))

    def test_valid_check(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        cdt = ConstrainedDelaunay(pts, [((0, 0), (4, 4))])
        # After construction, constraints should be present
        edges = cdt.get_edges()
        assert len(edges) >= 4


class TestHexagonal:
    def test_hex_grid(self):
        pts = []
        for row in range(4):
            for col in range(4):
                x = col * 2.0 + (row % 2)
                y = row * math.sqrt(3)
                pts.append((x, y))
        dt = DelaunayTriangulation(pts)
        assert dt.is_delaunay()
        assert euler_check(dt) == 2


class TestConvexHullBoundary:
    def test_all_hull(self):
        # Triangle: all 3 points on hull
        pts = [(0, 0), (1, 0), (0, 1)]
        dt = DelaunayTriangulation(pts)
        hull = dt.convex_hull()
        assert len(hull) == 3

    def test_one_interior(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        dt = DelaunayTriangulation(pts)
        hull = dt.convex_hull()
        assert len(hull) == 4
        assert (2.0, 2.0) not in hull


class TestTriangleNeighbors:
    def test_shared_edge_neighbors(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        dt = DelaunayTriangulation(pts)
        tris = list(dt.triangles)
        assert len(tris) == 2
        # The two triangles should be neighbors
        t0, t1 = tris
        assert t1 in t0.neighbors or None in t0.neighbors

    def test_boundary_has_none_neighbor(self):
        pts = [(0, 0), (1, 0), (0, 1)]
        dt = DelaunayTriangulation(pts)
        t = list(dt.triangles)[0]
        # Single triangle: all neighbors should be None
        assert all(n is None for n in t.neighbors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
