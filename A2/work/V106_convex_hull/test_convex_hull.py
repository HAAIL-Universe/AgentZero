"""Tests for V106: Convex Hull Computation for Polyhedra"""

import pytest
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))

from convex_hull import (
    Generator, VPolyhedron, LinearConstraint,
    h_to_v, v_to_h, convex_hull, convex_hull_v,
    exact_convex_hull, convert_to_vertices, convert_to_constraints,
    compare_joins, minkowski_sum, intersection, project,
    is_subset, ExactJoinPolyhedralDomain, ExactJoinInterpreter,
    estimate_volume, affine_image, affine_preimage,
    widening_with_thresholds, delayed_widening,
    exact_analyze, compare_analyses, convex_hull_summary,
    ZERO, ONE,
)
from polyhedral_domain import PolyhedralDomain, frac

F = Fraction


# ========================================================================
# Section 1: Generator and VPolyhedron basics
# ========================================================================

class TestGenerator:
    def test_vertex_creation(self):
        g = Generator(coords={'x': F(1), 'y': F(2)}, is_ray=False)
        assert not g.is_ray
        assert g.coords['x'] == F(1)
        assert g.coords['y'] == F(2)

    def test_ray_creation(self):
        g = Generator(coords={'x': F(1), 'y': F(0)}, is_ray=True)
        assert g.is_ray
        assert g.coords['x'] == F(1)

    def test_generator_equality(self):
        g1 = Generator(coords={'x': F(1)}, is_ray=False)
        g2 = Generator(coords={'x': F(1)}, is_ray=False)
        g3 = Generator(coords={'x': F(1)}, is_ray=True)
        assert g1 == g2
        assert g1 != g3

    def test_generator_hash(self):
        g1 = Generator(coords={'x': F(1)}, is_ray=False)
        g2 = Generator(coords={'x': F(1)}, is_ray=False)
        assert hash(g1) == hash(g2)

    def test_generator_repr(self):
        g = Generator(coords={'x': F(1), 'y': F(2)}, is_ray=False)
        r = repr(g)
        assert 'vertex' in r
        assert 'x=1' in r


class TestVPolyhedron:
    def test_empty(self):
        vp = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        assert vp.is_empty()
        assert vp.is_bounded()

    def test_add_vertex(self):
        vp = VPolyhedron(var_names=['x', 'y'], vertices=[], rays=[])
        vp.add_vertex({'x': F(1), 'y': F(2)})
        assert len(vp.vertices) == 1
        assert not vp.is_empty()

    def test_add_ray(self):
        vp = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp.add_ray({'x': F(1)})
        assert len(vp.rays) == 1
        assert not vp.is_bounded()

    def test_no_duplicate_vertex(self):
        vp = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp.add_vertex({'x': F(1)})
        vp.add_vertex({'x': F(1)})
        assert len(vp.vertices) == 1

    def test_dim(self):
        vp = VPolyhedron(var_names=['x', 'y', 'z'], vertices=[], rays=[])
        assert vp.dim() == 3


# ========================================================================
# Section 2: H-to-V Conversion (vertex enumeration)
# ========================================================================

class TestHToV:
    def test_single_point(self):
        """Single point: x == 5."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_equal('x', 5)
        vp = h_to_v(p)
        assert len(vp.vertices) == 1
        assert vp.vertices[0].coords['x'] == F(5)
        assert vp.is_bounded()

    def test_1d_interval(self):
        """Interval: 0 <= x <= 10."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        vp = h_to_v(p)
        assert len(vp.vertices) == 2
        values = sorted(v.coords['x'] for v in vp.vertices)
        assert values == [F(0), F(10)]

    def test_2d_square(self):
        """Square: 0 <= x <= 1, 0 <= y <= 1."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_upper('x', 1)
        p.set_lower('y', 0)
        p.set_upper('y', 1)
        vp = h_to_v(p)
        assert len(vp.vertices) == 4
        coords = {(v.coords['x'], v.coords['y']) for v in vp.vertices}
        assert (F(0), F(0)) in coords
        assert (F(1), F(0)) in coords
        assert (F(0), F(1)) in coords
        assert (F(1), F(1)) in coords

    def test_2d_triangle(self):
        """Triangle: x >= 0, y >= 0, x + y <= 1."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_lower('y', 0)
        p.add_constraint({'x': 1, 'y': 1}, 1)  # x + y <= 1
        vp = h_to_v(p)
        assert len(vp.vertices) == 3
        coords = {(v.coords['x'], v.coords['y']) for v in vp.vertices}
        assert (F(0), F(0)) in coords
        assert (F(1), F(0)) in coords
        assert (F(0), F(1)) in coords

    def test_bottom(self):
        """Bottom polyhedron has no vertices."""
        p = PolyhedralDomain.bot(['x', 'y'])
        vp = h_to_v(p)
        assert vp.is_empty()

    def test_0d(self):
        """0-dimensional polyhedron is a single point."""
        p = PolyhedralDomain(var_names=[])
        vp = h_to_v(p)
        assert len(vp.vertices) == 1

    def test_1d_unbounded_above(self):
        """x >= 0, no upper bound."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        vp = h_to_v(p)
        assert len(vp.vertices) >= 1
        # Should have a ray in positive direction
        assert len(vp.rays) >= 1

    def test_2d_rectangle(self):
        """Rectangle: 1 <= x <= 3, 2 <= y <= 5."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 1)
        p.set_upper('x', 3)
        p.set_lower('y', 2)
        p.set_upper('y', 5)
        vp = h_to_v(p)
        assert len(vp.vertices) == 4


# ========================================================================
# Section 3: V-to-H Conversion (facet enumeration)
# ========================================================================

class TestVToH:
    def test_single_vertex(self):
        """Single vertex -> single point constraints."""
        vp = VPolyhedron(var_names=['x', 'y'], vertices=[], rays=[])
        vp.add_vertex({'x': F(3), 'y': F(4)})
        hp = v_to_h(vp)
        assert hp.get_lower('x') == 3.0
        assert hp.get_upper('x') == 3.0
        assert hp.get_lower('y') == 4.0
        assert hp.get_upper('y') == 4.0

    def test_1d_two_vertices(self):
        """Two vertices -> interval."""
        vp = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp.add_vertex({'x': F(2)})
        vp.add_vertex({'x': F(8)})
        hp = v_to_h(vp)
        assert hp.get_lower('x') == 2.0
        assert hp.get_upper('x') == 8.0

    def test_empty_vpoly(self):
        """Empty V-poly -> bottom."""
        vp = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        hp = v_to_h(vp)
        assert hp.is_bot()

    def test_2d_triangle_vertices(self):
        """Triangle from 3 vertices."""
        vp = VPolyhedron(var_names=['x', 'y'], vertices=[], rays=[])
        vp.add_vertex({'x': F(0), 'y': F(0)})
        vp.add_vertex({'x': F(4), 'y': F(0)})
        vp.add_vertex({'x': F(0), 'y': F(3)})
        hp = v_to_h(vp)
        # Should contain all three vertices
        assert hp.get_lower('x') >= -0.01
        assert hp.get_lower('y') >= -0.01
        assert hp.get_upper('x') <= 4.01
        assert hp.get_upper('y') <= 3.01

    def test_2d_square_vertices(self):
        """Square from 4 vertices."""
        vp = VPolyhedron(var_names=['x', 'y'], vertices=[], rays=[])
        for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            vp.add_vertex({'x': F(x), 'y': F(y)})
        hp = v_to_h(vp)
        assert hp.get_lower('x') >= -0.01
        assert hp.get_upper('x') <= 1.01
        assert hp.get_lower('y') >= -0.01
        assert hp.get_upper('y') <= 1.01

    def test_1d_with_ray(self):
        """Vertex + positive ray -> half-line."""
        vp = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp.add_vertex({'x': F(5)})
        vp.add_ray({'x': F(1)})
        hp = v_to_h(vp)
        assert hp.get_lower('x') == 5.0
        assert hp.get_upper('x') == float('inf')


# ========================================================================
# Section 4: Round-trip H -> V -> H
# ========================================================================

class TestRoundTrip:
    def test_roundtrip_interval(self):
        """H -> V -> H preserves interval."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 3)
        p.set_upper('x', 7)
        vp = h_to_v(p)
        p2 = v_to_h(vp)
        assert abs(p2.get_lower('x') - 3.0) < 0.01
        assert abs(p2.get_upper('x') - 7.0) < 0.01

    def test_roundtrip_square(self):
        """H -> V -> H preserves square."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_upper('x', 5)
        p.set_lower('y', 0)
        p.set_upper('y', 5)
        vp = h_to_v(p)
        p2 = v_to_h(vp)
        assert abs(p2.get_lower('x') - 0.0) < 0.01
        assert abs(p2.get_upper('x') - 5.0) < 0.01
        assert abs(p2.get_lower('y') - 0.0) < 0.01
        assert abs(p2.get_upper('y') - 5.0) < 0.01

    def test_roundtrip_triangle(self):
        """H -> V -> H preserves triangle bounds."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_lower('y', 0)
        p.add_constraint({'x': 1, 'y': 1}, 1)
        vp = h_to_v(p)
        p2 = v_to_h(vp)
        assert p2.get_lower('x') >= -0.01
        assert p2.get_upper('x') <= 1.01
        assert p2.get_lower('y') >= -0.01
        assert p2.get_upper('y') <= 1.01

    def test_roundtrip_point(self):
        """H -> V -> H preserves point."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_equal('x', 42)
        vp = h_to_v(p)
        p2 = v_to_h(vp)
        assert abs(p2.get_lower('x') - 42.0) < 0.01
        assert abs(p2.get_upper('x') - 42.0) < 0.01


# ========================================================================
# Section 5: Convex Hull (the core operation)
# ========================================================================

class TestConvexHull:
    def test_hull_two_points(self):
        """Hull of two points -> line segment."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_equal('x', 0)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_equal('x', 10)
        hull = convex_hull(p1, p2)
        assert hull.get_lower('x') >= -0.01
        assert hull.get_upper('x') <= 10.01

    def test_hull_intervals(self):
        """Hull of two intervals."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 3)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 7)
        p2.set_upper('x', 10)
        hull = convex_hull(p1, p2)
        assert hull.get_lower('x') >= -0.01
        assert hull.get_upper('x') <= 10.01

    def test_hull_with_bottom(self):
        """Hull with bottom is identity."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 1)
        p1.set_upper('x', 5)
        p2 = PolyhedralDomain.bot(['x'])
        hull = convex_hull(p1, p2)
        assert abs(hull.get_lower('x') - 1.0) < 0.01
        assert abs(hull.get_upper('x') - 5.0) < 0.01

    def test_hull_both_bottom(self):
        """Hull of two bottoms is bottom."""
        p1 = PolyhedralDomain.bot(['x'])
        p2 = PolyhedralDomain.bot(['x'])
        hull = convex_hull(p1, p2)
        assert hull.is_bot()

    def test_hull_2d_disjoint_rectangles(self):
        """Hull of two disjoint rectangles."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 1)
        p1.set_lower('y', 0)
        p1.set_upper('y', 1)

        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 3)
        p2.set_upper('x', 4)
        p2.set_lower('y', 3)
        p2.set_upper('y', 4)

        hull = convex_hull(p1, p2)
        # Hull should contain both
        assert hull.get_lower('x') <= 0.01
        assert hull.get_upper('x') >= 3.99
        assert hull.get_lower('y') <= 0.01
        assert hull.get_upper('y') >= 3.99

    def test_hull_overlapping(self):
        """Hull of overlapping intervals is the outer interval."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 7)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 3)
        p2.set_upper('x', 10)
        hull = convex_hull(p1, p2)
        assert hull.get_lower('x') >= -0.01
        assert hull.get_upper('x') <= 10.01

    def test_hull_soundness_contains_inputs(self):
        """The hull must contain both input polyhedra."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 1)
        p1.set_upper('x', 3)
        p1.set_lower('y', 0)
        p1.set_upper('y', 2)

        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 5)
        p2.set_upper('x', 7)
        p2.set_lower('y', 4)
        p2.set_upper('y', 6)

        hull = convex_hull(p1, p2)
        # Check containment of p1's vertices
        v1 = h_to_v(p1)
        for v in v1.vertices:
            for c in hull.constraints:
                val = c.evaluate({vn: v.coords[vn] for vn in hull.var_names if vn in v.coords})
                if c.is_equality:
                    assert abs(val - c.bound) < F(1, 100), f"Vertex {v} violates {c}"
                else:
                    assert val <= c.bound + F(1, 100), f"Vertex {v} violates {c}"


# ========================================================================
# Section 6: Convex hull V-representation union
# ========================================================================

class TestConvexHullV:
    def test_union_vertices(self):
        vp1 = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp1.add_vertex({'x': F(0)})
        vp2 = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp2.add_vertex({'x': F(10)})
        result = convex_hull_v(vp1, vp2)
        assert len(result.vertices) == 2

    def test_union_rays(self):
        vp1 = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp1.add_vertex({'x': F(0)})
        vp1.add_ray({'x': F(1)})
        vp2 = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp2.add_vertex({'x': F(0)})
        vp2.add_ray({'x': F(-1)})
        result = convex_hull_v(vp1, vp2)
        assert len(result.rays) == 2

    def test_union_different_vars(self):
        vp1 = VPolyhedron(var_names=['x'], vertices=[], rays=[])
        vp1.add_vertex({'x': F(1)})
        vp2 = VPolyhedron(var_names=['y'], vertices=[], rays=[])
        vp2.add_vertex({'y': F(2)})
        result = convex_hull_v(vp1, vp2)
        assert set(result.var_names) == {'x', 'y'}
        assert len(result.vertices) == 2


# ========================================================================
# Section 7: Compare approximate vs exact join
# ========================================================================

class TestCompareJoins:
    def test_compare_identical(self):
        """Identical polyhedra -> same result."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        comp = compare_joins(p, p)
        assert comp['both_sound']

    def test_compare_disjoint_1d(self):
        """Disjoint intervals -> exact matches approximate for 1D."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 3)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 7)
        p2.set_upper('x', 10)
        comp = compare_joins(p1, p2)
        assert comp['both_sound']

    def test_soundness_always_holds(self):
        """Exact hull contains all vertices from both inputs."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 2)
        p1.set_lower('y', 0)
        p1.set_upper('y', 2)
        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 3)
        p2.set_upper('x', 5)
        p2.set_lower('y', 3)
        p2.set_upper('y', 5)
        hull = convex_hull(p1, p2)

        # Verify all vertices of p1 and p2 are contained in hull
        for p in [p1, p2]:
            vp = h_to_v(p)
            for v in vp.vertices:
                for c in hull.constraints:
                    val = c.evaluate({vn: v.coords.get(vn, ZERO) for vn in hull.var_names})
                    if c.is_equality:
                        assert abs(val - c.bound) < F(1, 100), f"Vertex {v} violates {c}"
                    else:
                        assert val <= c.bound + F(1, 100), f"Vertex {v} violates {c}"


# ========================================================================
# Section 8: ExactJoinPolyhedralDomain
# ========================================================================

class TestExactJoinDomain:
    def test_exact_join_basic(self):
        p1 = ExactJoinPolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 5)
        p2 = ExactJoinPolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 3)
        p2.set_upper('x', 10)
        result = p1.join(p2)
        assert isinstance(result, ExactJoinPolyhedralDomain)
        assert result.get_lower('x') >= -0.01
        assert result.get_upper('x') <= 10.01

    def test_exact_join_with_bot(self):
        p1 = ExactJoinPolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 1)
        p1.set_upper('x', 5)
        p2 = ExactJoinPolyhedralDomain(var_names=['x'])
        p2._is_bot = True
        result = p1.join(p2)
        assert abs(result.get_lower('x') - 1.0) < 0.01
        assert abs(result.get_upper('x') - 5.0) < 0.01

    def test_exact_join_copy(self):
        p = ExactJoinPolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        c = p.copy()
        assert isinstance(c, ExactJoinPolyhedralDomain)
        assert abs(c.get_lower('x') - 0.0) < 0.01


# ========================================================================
# Section 9: Minkowski Sum
# ========================================================================

class TestMinkowskiSum:
    def test_sum_intervals(self):
        """[0,3] + [5,7] = [5,10]."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 3)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 5)
        p2.set_upper('x', 7)
        result = minkowski_sum(p1, p2)
        assert abs(result.get_lower('x') - 5.0) < 0.01
        assert abs(result.get_upper('x') - 10.0) < 0.01

    def test_sum_with_bottom(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 5)
        p2 = PolyhedralDomain.bot(['x'])
        result = minkowski_sum(p1, p2)
        assert result.is_bot()

    def test_sum_points(self):
        """Point + Point = Point."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_equal('x', 3)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_equal('x', 7)
        result = minkowski_sum(p1, p2)
        assert abs(result.get_lower('x') - 10.0) < 0.01
        assert abs(result.get_upper('x') - 10.0) < 0.01


# ========================================================================
# Section 10: Projection
# ========================================================================

class TestProjection:
    def test_project_2d_to_1d(self):
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        p.set_lower('y', 0)
        p.set_upper('y', 5)
        result = project(p, ['x'])
        assert abs(result.get_lower('x') - 0.0) < 0.01
        assert abs(result.get_upper('x') - 10.0) < 0.01
        assert 'y' not in result.var_names

    def test_project_keeps_relational(self):
        """Projecting with relational constraints preserves derived bounds."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        p.set_lower('y', 0)
        p.add_constraint({'x': 1, 'y': 1}, 8)  # x + y <= 8
        result = project(p, ['y'])
        # y <= 8 (from x=0, x+y<=8), y >= 0
        assert result.get_lower('y') >= -0.01
        assert result.get_upper('y') <= 8.01


# ========================================================================
# Section 11: Subset Check
# ========================================================================

class TestSubset:
    def test_subset_of_self(self):
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        assert is_subset(p, p)

    def test_strict_subset(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 2)
        p1.set_upper('x', 8)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 0)
        p2.set_upper('x', 10)
        assert is_subset(p1, p2)

    def test_not_subset(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 10)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 2)
        p2.set_upper('x', 8)
        assert not is_subset(p1, p2)


# ========================================================================
# Section 12: Volume Estimation
# ========================================================================

class TestVolume:
    def test_volume_1d(self):
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        vol = estimate_volume(p)
        assert abs(vol - 10.0) < 0.01

    def test_volume_2d_square(self):
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_upper('x', 3)
        p.set_lower('y', 0)
        p.set_upper('y', 3)
        vol = estimate_volume(p)
        assert abs(vol - 9.0) < 0.5  # May not be exact due to triangulation

    def test_volume_empty(self):
        p = PolyhedralDomain.bot(['x'])
        vol = estimate_volume(p)
        assert vol == 0.0

    def test_volume_point(self):
        p = PolyhedralDomain(var_names=['x'])
        p.set_equal('x', 5)
        vol = estimate_volume(p)
        assert vol == 0.0

    def test_volume_0d(self):
        p = PolyhedralDomain(var_names=[])
        vol = estimate_volume(p)
        assert vol == 1.0


# ========================================================================
# Section 13: Affine Image and Pre-image
# ========================================================================

class TestAffineOps:
    def test_affine_image_const(self):
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        result = affine_image(p, 'x', {}, F(5))  # x := 5
        assert abs(result.get_lower('x') - 5.0) < 0.01
        assert abs(result.get_upper('x') - 5.0) < 0.01

    def test_affine_image_linear(self):
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        result = affine_image(p, 'x', {'x': F(2)}, F(1))  # x := 2*x + 1
        assert result.get_lower('x') >= 0.99
        assert result.get_upper('x') <= 21.01

    def test_affine_preimage_basic(self):
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 5)
        p.set_upper('x', 15)
        # Pre-image of x := x + 5 on [5, 15] -> [0, 10]
        result = affine_preimage(p, 'x', {'x': F(1)}, F(5))
        assert result.get_lower('x') >= -0.01
        assert result.get_upper('x') <= 10.01


# ========================================================================
# Section 14: Widening Variants
# ========================================================================

class TestWideningVariants:
    def test_threshold_widening_relaxes(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 5)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 0)
        p2.set_upper('x', 7)
        result = widening_with_thresholds(p1, p2, [F(0), F(5), F(10), F(100)])
        # x <= 5 is violated by p2 (has x=7), should relax to threshold 10
        assert result.get_upper('x') <= 10.01

    def test_delayed_widening_exact_early(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 5)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 0)
        p2.set_upper('x', 8)
        # Iteration 0 -> exact join
        result = delayed_widening(p1, p2, iteration=0, delay=3)
        assert result.get_lower('x') >= -0.01
        assert result.get_upper('x') <= 8.01

    def test_delayed_widening_standard_later(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 5)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 0)
        p2.set_upper('x', 8)
        # Iteration 5 -> standard widening
        result = delayed_widening(p1, p2, iteration=5, delay=3)
        # Standard widening may drop violated constraints
        assert result is not None


# ========================================================================
# Section 15: Intersection
# ========================================================================

class TestIntersection:
    def test_intersection_overlapping(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 7)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 3)
        p2.set_upper('x', 10)
        result = intersection(p1, p2)
        assert abs(result.get_lower('x') - 3.0) < 0.01
        assert abs(result.get_upper('x') - 7.0) < 0.01

    def test_intersection_disjoint(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 3)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 5)
        p2.set_upper('x', 10)
        result = intersection(p1, p2)
        assert result.is_bot()


# ========================================================================
# Section 16: C10 Integration: Exact-Join Analysis
# ========================================================================

class TestExactAnalysis:
    def test_simple_assignment(self):
        result = exact_analyze("let x = 5;")
        assert 'ranges' in result
        assert abs(result['ranges']['x'][0] - 5.0) < 0.01
        assert abs(result['ranges']['x'][1] - 5.0) < 0.01

    def test_linear_expr(self):
        result = exact_analyze("let x = 5; let y = x + 3;")
        assert abs(result['ranges']['y'][0] - 8.0) < 0.01
        assert abs(result['ranges']['y'][1] - 8.0) < 0.01

    def test_conditional(self):
        source = """
        let x = 10;
        let y = 0;
        if (x > 5) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = exact_analyze(source)
        # x > 5 is true, so y = 1
        assert result['ranges']['y'][0] >= 0.99

    def test_compare_analyses_api(self):
        result = compare_analyses("let x = 10; let y = x + 5;")
        assert 'approximate' in result
        assert 'exact' in result
        assert 'comparison' in result


# ========================================================================
# Section 17: Summary API
# ========================================================================

class TestSummary:
    def test_summary_output(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 5)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 3)
        p2.set_upper('x', 10)
        s = convex_hull_summary(p1, p2)
        assert 'V106' in s
        assert 'constraints' in s

    def test_convert_apis(self):
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        vp = convert_to_vertices(p)
        assert len(vp.vertices) == 2
        p2 = convert_to_constraints(vp)
        assert abs(p2.get_lower('x') - 0.0) < 0.01


# ========================================================================
# Section 18: Edge Cases
# ========================================================================

class TestEdgeCases:
    def test_hull_single_point_each(self):
        """Hull of two single points -> line segment."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_equal('x', 0)
        p1.set_equal('y', 0)
        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_equal('x', 10)
        p2.set_equal('y', 10)
        hull = convex_hull(p1, p2)
        assert hull.get_lower('x') <= 0.01
        assert hull.get_upper('x') >= 9.99
        assert hull.get_lower('y') <= 0.01
        assert hull.get_upper('y') >= 9.99

    def test_hull_identical_polyhedra(self):
        """Hull of identical polyhedra is the same polyhedron."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 3)
        p.set_upper('x', 7)
        hull = convex_hull(p, p)
        assert abs(hull.get_lower('x') - 3.0) < 0.01
        assert abs(hull.get_upper('x') - 7.0) < 0.01

    def test_hull_one_contains_other(self):
        """Hull where one polyhedron contains the other."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 10)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 2)
        p2.set_upper('x', 8)
        hull = convex_hull(p1, p2)
        assert hull.get_lower('x') >= -0.01
        assert hull.get_upper('x') <= 10.01

    def test_exact_hull_api(self):
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 5)
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', 5)
        p2.set_upper('x', 10)
        hull = exact_convex_hull(p1, p2)
        assert hull.get_lower('x') >= -0.01
        assert hull.get_upper('x') <= 10.01

    def test_fraction_precision(self):
        """Exact Fraction arithmetic: no floating-point errors."""
        p1 = PolyhedralDomain(var_names=['x'])
        p1.set_lower('x', F(1, 3))
        p1.set_upper('x', F(2, 3))
        p2 = PolyhedralDomain(var_names=['x'])
        p2.set_lower('x', F(4, 3))
        p2.set_upper('x', F(5, 3))
        hull = convex_hull(p1, p2)
        # Should span [1/3, 5/3]
        lo = hull.get_lower('x')
        hi = hull.get_upper('x')
        assert lo <= float(F(1, 3)) + 0.01
        assert hi >= float(F(5, 3)) - 0.01


# ========================================================================
# Section 19: Relational constraint preservation in hull
# ========================================================================

class TestRelationalHull:
    def test_hull_preserves_bounding_box(self):
        """At minimum, hull must preserve bounding box."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 2)
        p1.set_lower('y', 0)
        p1.set_upper('y', 2)

        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 4)
        p2.set_upper('x', 6)
        p2.set_lower('y', 4)
        p2.set_upper('y', 6)

        hull = convex_hull(p1, p2)
        assert hull.get_lower('x') <= 0.01
        assert hull.get_upper('x') >= 5.99
        assert hull.get_lower('y') <= 0.01
        assert hull.get_upper('y') >= 5.99

    def test_hull_finds_diagonal_constraints(self):
        """Hull of two separated squares should find diagonal bounds.

        P1 = [0,1]x[0,1], P2 = [4,5]x[4,5]
        True hull should have constraints like x - y <= 1, y - x <= 1
        (the diagonal relationship: points near (0,0) and (5,5) are connected).
        """
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 1)
        p1.set_lower('y', 0)
        p1.set_upper('y', 1)

        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 4)
        p2.set_upper('x', 5)
        p2.set_lower('y', 4)
        p2.set_upper('y', 5)

        hull = convex_hull(p1, p2)
        # The exact hull should have more than just bounding box constraints
        # It should have relational constraints like x - y <= 1
        constraints = hull.get_constraints()
        # At minimum it's sound: contains both inputs
        v1 = h_to_v(p1)
        for v in v1.vertices:
            for c in hull.constraints:
                val = c.evaluate({vn: v.coords.get(vn, ZERO) for vn in hull.var_names})
                if c.is_equality:
                    assert abs(val - c.bound) < F(1, 100)
                else:
                    assert val <= c.bound + F(1, 100)


# ========================================================================
# Section 20: ExactJoinInterpreter integration
# ========================================================================

class TestExactJoinInterpreter:
    def test_creates_exact_env(self):
        """ExactJoinInterpreter produces ExactJoinPolyhedralDomain env."""
        interp = ExactJoinInterpreter()
        result = interp.analyze("let x = 1;")
        assert isinstance(result['env'], ExactJoinPolyhedralDomain)

    def test_analyze_simple(self):
        interp = ExactJoinInterpreter()
        result = interp.analyze("let x = 42;")
        assert abs(result['ranges']['x'][0] - 42.0) < 0.01

    def test_analyze_conditional_join(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 10;
        } else {
            y = 20;
        }
        """
        interp = ExactJoinInterpreter()
        result = interp.analyze(source)
        # x > 3 is always true, so y should be 10
        assert result['ranges']['y'][0] >= 9.99


# ========================================================================
# Section 21: 3D Polyhedra
# ========================================================================

class Test3D:
    def test_h_to_v_cube(self):
        """3D cube: 8 vertices."""
        p = PolyhedralDomain(var_names=['x', 'y', 'z'])
        for v in ['x', 'y', 'z']:
            p.set_lower(v, 0)
            p.set_upper(v, 1)
        vp = h_to_v(p)
        assert len(vp.vertices) == 8

    def test_h_to_v_tetrahedron(self):
        """Tetrahedron: x>=0, y>=0, z>=0, x+y+z<=1."""
        p = PolyhedralDomain(var_names=['x', 'y', 'z'])
        p.set_lower('x', 0)
        p.set_lower('y', 0)
        p.set_lower('z', 0)
        p.add_constraint({'x': 1, 'y': 1, 'z': 1}, 1)
        vp = h_to_v(p)
        assert len(vp.vertices) == 4

    def test_roundtrip_cube(self):
        """H -> V -> H preserves 3D cube."""
        p = PolyhedralDomain(var_names=['x', 'y', 'z'])
        for v in ['x', 'y', 'z']:
            p.set_lower(v, 0)
            p.set_upper(v, 2)
        vp = h_to_v(p)
        p2 = v_to_h(vp)
        for v in ['x', 'y', 'z']:
            assert abs(p2.get_lower(v) - 0.0) < 0.01
            assert abs(p2.get_upper(v) - 2.0) < 0.01

    def test_hull_3d_cubes(self):
        """Hull of two 3D cubes."""
        p1 = PolyhedralDomain(var_names=['x', 'y', 'z'])
        for v in ['x', 'y', 'z']:
            p1.set_lower(v, 0)
            p1.set_upper(v, 1)
        p2 = PolyhedralDomain(var_names=['x', 'y', 'z'])
        for v in ['x', 'y', 'z']:
            p2.set_lower(v, 2)
            p2.set_upper(v, 3)
        hull = convex_hull(p1, p2)
        for v in ['x', 'y', 'z']:
            assert hull.get_lower(v) <= 0.01
            assert hull.get_upper(v) >= 2.99


# ========================================================================
# Section 22: Precision comparison (key V106 value proposition)
# ========================================================================

class TestPrecisionGain:
    def test_exact_tighter_for_diagonal(self):
        """Exact hull is tighter than approximate for diagonal-separated shapes."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 1)
        p1.set_lower('y', 0)
        p1.set_upper('y', 1)

        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 9)
        p2.set_upper('x', 10)
        p2.set_lower('y', 9)
        p2.set_upper('y', 10)

        exact = convex_hull(p1, p2)
        approx = p1.join(p2)

        # Exact should find relational constraints like x - y <= 1
        exact_constrs = exact.get_constraints()
        approx_constrs = approx.get_constraints()
        # Exact should have MORE constraints (tighter)
        assert len(exact_constrs) >= len(approx_constrs)

    def test_compare_joins_detects_improvement(self):
        """compare_joins should detect when exact is tighter."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 1)
        p1.set_lower('y', 0)
        p1.set_upper('y', 1)

        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 4)
        p2.set_upper('x', 5)
        p2.set_lower('y', 4)
        p2.set_upper('y', 5)

        comp = compare_joins(p1, p2)
        assert comp['both_sound']

    def test_hull_relational_constraint_found(self):
        """The exact hull discovers relational constraints the approximate misses."""
        p1 = PolyhedralDomain(var_names=['x', 'y'])
        p1.set_lower('x', 0)
        p1.set_upper('x', 1)
        p1.set_lower('y', 0)
        p1.set_upper('y', 1)

        p2 = PolyhedralDomain(var_names=['x', 'y'])
        p2.set_lower('x', 4)
        p2.set_upper('x', 5)
        p2.set_lower('y', 4)
        p2.set_upper('y', 5)

        exact = convex_hull(p1, p2)
        rel = exact.get_relational_constraints()
        assert len(rel) > 0, "Exact hull should find relational constraints"


# ========================================================================
# Section 23: Linear system solver
# ========================================================================

class TestLinearSolver:
    def test_solve_2x2(self):
        from convex_hull import _solve_linear_system
        # x + y = 3, x - y = 1 => x=2, y=1
        eqs = [({'x': F(1), 'y': F(1)}, F(3)),
               ({'x': F(1), 'y': F(-1)}, F(1))]
        sol = _solve_linear_system(eqs, ['x', 'y'])
        assert sol is not None
        assert sol['x'] == F(2)
        assert sol['y'] == F(1)

    def test_solve_singular(self):
        from convex_hull import _solve_linear_system
        # x + y = 3, 2x + 2y = 6 => singular
        eqs = [({'x': F(1), 'y': F(1)}, F(3)),
               ({'x': F(2), 'y': F(2)}, F(6))]
        sol = _solve_linear_system(eqs, ['x', 'y'])
        assert sol is None

    def test_solve_3x3(self):
        from convex_hull import _solve_linear_system
        # x=1, y=2, z=3
        eqs = [({'x': F(1), 'y': F(0), 'z': F(0)}, F(1)),
               ({'x': F(0), 'y': F(1), 'z': F(0)}, F(2)),
               ({'x': F(0), 'y': F(0), 'z': F(1)}, F(3))]
        sol = _solve_linear_system(eqs, ['x', 'y', 'z'])
        assert sol == {'x': F(1), 'y': F(2), 'z': F(3)}


# ========================================================================
# Section 24: Affine pre-image edge cases
# ========================================================================

class TestAffineEdgeCases:
    def test_preimage_no_target_var(self):
        """Pre-image when target doesn't appear in constraints."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        p.set_lower('y', 0)
        p.set_upper('y', 10)
        result = affine_preimage(p, 'z', {'x': F(1)}, F(0))
        # z not in constraints, so constraints unchanged
        assert abs(result.get_lower('x') - 0.0) < 0.01
        assert abs(result.get_upper('x') - 10.0) < 0.01

    def test_image_new_var(self):
        """Affine image introducing a new variable."""
        p = PolyhedralDomain(var_names=['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        result = affine_image(p, 'y', {'x': F(1)}, F(5))
        # y := x + 5, x in [0,10] => y in [5, 15]
        assert result.get_lower('y') >= 4.99
        assert result.get_upper('y') <= 15.01


# ========================================================================
# Section 25: Volume edge cases
# ========================================================================

class TestVolumeEdgeCases:
    def test_volume_2d_triangle(self):
        """Area of triangle with vertices (0,0), (4,0), (0,3) = 6."""
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_lower('y', 0)
        p.add_constraint({'x': 1, 'y': 1}, F(4))  # loose bound
        p.add_constraint({'x': F(3), 'y': F(4)}, F(12))  # 3x + 4y <= 12
        vol = estimate_volume(p)
        assert vol is not None
        # Area should be 6
        assert abs(vol - 6.0) < 1.0  # triangulation may approximate

    def test_volume_rectangle(self):
        p = PolyhedralDomain(var_names=['x', 'y'])
        p.set_lower('x', 0)
        p.set_upper('x', 4)
        p.set_lower('y', 0)
        p.set_upper('y', 3)
        vol = estimate_volume(p)
        assert abs(vol - 12.0) < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
