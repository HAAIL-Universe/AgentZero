"""Tests for C136: Interpolation and Approximation."""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from interpolation import (
    lagrange_interpolate, lagrange_weights, barycentric_interpolate,
    divided_differences, newton_interpolate, newton_add_point,
    chebyshev_nodes, chebyshev_coefficients, chebyshev_evaluate, ChebyshevApprox,
    LinearSpline, CubicSpline, PchipInterpolator, AkimaInterpolator,
    rational_interpolate, RBFInterpolator, BSpline, bspline_interpolate,
    pade_approximant, pade_evaluate,
    polyfit, polyval, exponential_fit, power_fit,
    BilinearInterpolator, BicubicInterpolator,
    trig_interpolate, MonotoneInterpolator, interpolate,
    _solve_tridiagonal,
)


# ============================================================
# Lagrange Interpolation
# ============================================================

class TestLagrange:
    def test_single_point(self):
        assert lagrange_interpolate([1.0], [3.0], 1.0) == pytest.approx(3.0)

    def test_linear(self):
        xs = [0.0, 1.0]
        ys = [0.0, 1.0]
        assert lagrange_interpolate(xs, ys, 0.5) == pytest.approx(0.5)

    def test_quadratic(self):
        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 1.0, 4.0]
        assert lagrange_interpolate(xs, ys, 1.5) == pytest.approx(2.25)

    def test_exact_at_nodes(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 8.0, 27.0, 64.0]
        for x, y in zip(xs, ys):
            assert lagrange_interpolate(xs, ys, x) == pytest.approx(y)

    def test_cubic_interpolation(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [x**3 for x in xs]
        assert lagrange_interpolate(xs, ys, 1.5) == pytest.approx(1.5**3)

    def test_barycentric_matches(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [math.sin(x) for x in xs]
        w = lagrange_weights(xs)
        for x in [0.5, 1.5, 2.5]:
            expected = lagrange_interpolate(xs, ys, x)
            actual = barycentric_interpolate(xs, ys, w, x)
            assert actual == pytest.approx(expected, abs=1e-12)

    def test_barycentric_at_node(self):
        xs = [0.0, 1.0, 2.0]
        ys = [1.0, 2.0, 5.0]
        w = lagrange_weights(xs)
        assert barycentric_interpolate(xs, ys, w, 1.0) == pytest.approx(2.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            lagrange_interpolate([], [], 0.0)


# ============================================================
# Newton Interpolation
# ============================================================

class TestNewton:
    def test_divided_diff_linear(self):
        coefs = divided_differences([0.0, 1.0], [0.0, 1.0])
        assert coefs[0] == pytest.approx(0.0)
        assert coefs[1] == pytest.approx(1.0)

    def test_newton_quadratic(self):
        xs = [0.0, 1.0, 2.0]
        ys = [1.0, 3.0, 9.0]
        val = newton_interpolate(xs, ys, 1.5)
        expected = lagrange_interpolate(xs, ys, 1.5)
        assert val == pytest.approx(expected)

    def test_newton_matches_lagrange(self):
        xs = [0.0, 1.0, 3.0, 4.0]
        ys = [math.sin(x) for x in xs]
        for x in [0.5, 2.0, 3.5]:
            n = newton_interpolate(xs, ys, x)
            l = lagrange_interpolate(xs, ys, x)
            assert n == pytest.approx(l, abs=1e-10)

    def test_add_point(self):
        xs = [0.0, 1.0]
        ys = [0.0, 1.0]
        coefs = divided_differences(xs, ys)
        coefs = list(coefs)
        newton_add_point(xs, ys, coefs, 2.0, 4.0)
        val = newton_interpolate(xs, ys, 1.5)
        assert val == pytest.approx(2.25)

    def test_exact_at_nodes(self):
        xs = [1.0, 2.0, 4.0]
        ys = [3.0, 7.0, 15.0]
        for x, y in zip(xs, ys):
            assert newton_interpolate(xs, ys, x) == pytest.approx(y)


# ============================================================
# Chebyshev
# ============================================================

class TestChebyshev:
    def test_nodes_count(self):
        nodes = chebyshev_nodes(5)
        assert len(nodes) == 5

    def test_nodes_in_range(self):
        nodes = chebyshev_nodes(10, 0, 1)
        for n in nodes:
            assert 0.0 <= n <= 1.0

    def test_approx_sin(self):
        approx = ChebyshevApprox(math.sin, 15, 0, math.pi)
        for x in [0.1, 0.5, 1.0, 2.0, 3.0]:
            assert approx(x) == pytest.approx(math.sin(x), abs=1e-10)

    def test_approx_polynomial(self):
        f = lambda x: x**3 - 2*x + 1
        approx = ChebyshevApprox(f, 4, -1, 1)
        for x in [-0.5, 0.0, 0.5]:
            assert approx(x) == pytest.approx(f(x), abs=1e-10)

    def test_truncate(self):
        approx = ChebyshevApprox(math.sin, 20, -1, 1)
        trunc = approx.truncate(5)
        assert trunc.n == 5
        # Less accurate but still reasonable
        assert trunc(0.5) == pytest.approx(math.sin(0.5), abs=1e-3)

    def test_derivative(self):
        approx = ChebyshevApprox(math.sin, 20, 0, math.pi)
        d = approx.derivative()
        assert d(0.5) == pytest.approx(math.cos(0.5), abs=1e-6)
        assert d(1.0) == pytest.approx(math.cos(1.0), abs=1e-6)

    def test_clenshaw_constant(self):
        coefs = [5.0]
        assert chebyshev_evaluate(coefs, 0.3) == pytest.approx(5.0)

    def test_clenshaw_empty(self):
        assert chebyshev_evaluate([], 0.5) == 0.0


# ============================================================
# Linear Spline
# ============================================================

class TestLinearSpline:
    def test_basic(self):
        s = LinearSpline([0, 1, 2], [0, 1, 0])
        assert s(0.5) == pytest.approx(0.5)
        assert s(1.5) == pytest.approx(0.5)

    def test_exact_at_nodes(self):
        s = LinearSpline([0, 1, 2, 3], [0, 2, 1, 3])
        for x, y in zip([0, 1, 2, 3], [0, 2, 1, 3]):
            assert s(x) == pytest.approx(y)

    def test_extrapolation(self):
        s = LinearSpline([0, 1], [0, 2])
        assert s(-1) == pytest.approx(-2.0)
        assert s(2) == pytest.approx(4.0)

    def test_unsorted_input(self):
        s = LinearSpline([2, 0, 1], [4, 0, 2])
        assert s(0.5) == pytest.approx(1.0)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            LinearSpline([1], [1])


# ============================================================
# Cubic Spline
# ============================================================

class TestCubicSpline:
    def test_natural_exact_at_nodes(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 0]
        s = CubicSpline(xs, ys)
        for x, y in zip(xs, ys):
            assert s(x) == pytest.approx(y, abs=1e-10)

    def test_natural_smooth(self):
        xs = [0, 1, 2, 3]
        ys = [0, 1, 0, -1]
        s = CubicSpline(xs, ys)
        # Second derivative at endpoints should be 0
        assert s.derivative(0.0, order=2) == pytest.approx(0.0, abs=1e-10)
        assert s.derivative(3.0, order=2) == pytest.approx(0.0, abs=1e-10)

    def test_clamped(self):
        xs = [0, 1, 2]
        ys = [0, 1, 0]
        s = CubicSpline(xs, ys, bc_type='clamped', clamped_slopes=(1.0, -1.0))
        assert s.derivative(0.0) == pytest.approx(1.0, abs=1e-10)
        assert s.derivative(2.0) == pytest.approx(-1.0, abs=1e-10)

    def test_not_a_knot(self):
        xs = [0, 1, 2, 3, 4]
        ys = [x**3 for x in xs]
        s = CubicSpline(xs, ys, bc_type='not-a-knot')
        # Cubic should be exact
        assert s(2.5) == pytest.approx(2.5**3, abs=1e-6)

    def test_periodic(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, -1, 0]  # y[0] == y[-1]
        s = CubicSpline(xs, ys, bc_type='periodic')
        for x, y in zip(xs, ys):
            assert s(x) == pytest.approx(y, abs=1e-10)

    def test_two_points(self):
        s = CubicSpline([0, 1], [0, 1])
        assert s(0.5) == pytest.approx(0.5)

    def test_derivative_order_1(self):
        xs = [0, 1, 2, 3]
        ys = [x**2 for x in xs]
        s = CubicSpline(xs, ys, bc_type='not-a-knot')
        # Derivative of x^2 is 2x
        assert s.derivative(1.5) == pytest.approx(3.0, abs=0.1)

    def test_integrate(self):
        xs = [0, 1, 2, 3]
        ys = [0, 1, 0, -1]
        s = CubicSpline(xs, ys)
        # Integration over full range
        val = s.integrate(0, 3)
        assert isinstance(val, float)

    def test_integrate_partial(self):
        xs = [0, 1, 2]
        ys = [0, 1, 0]
        s = CubicSpline(xs, ys)
        full = s.integrate(0, 2)
        half1 = s.integrate(0, 1)
        half2 = s.integrate(1, 2)
        assert full == pytest.approx(half1 + half2, abs=1e-10)

    def test_integrate_reversed(self):
        xs = [0, 1, 2]
        ys = [0, 1, 0]
        s = CubicSpline(xs, ys)
        assert s.integrate(0, 2) == pytest.approx(-s.integrate(2, 0))


# ============================================================
# PCHIP
# ============================================================

class TestPchip:
    def test_exact_at_nodes(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 0]
        p = PchipInterpolator(xs, ys)
        for x, y in zip(xs, ys):
            assert p(x) == pytest.approx(y, abs=1e-10)

    def test_monotone_data(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 2, 4, 8]
        p = PchipInterpolator(xs, ys)
        # Should be monotone: p(0.5) < p(1.5) < p(2.5) < p(3.5)
        vals = [p(x) for x in [0.5, 1.5, 2.5, 3.5]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_two_points(self):
        p = PchipInterpolator([0, 1], [0, 1])
        assert p(0.5) == pytest.approx(0.5)

    def test_shape_preserving(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 0, 1, 1, 1]
        p = PchipInterpolator(xs, ys)
        # No undershoot below 0 or overshoot above 1
        for x_test in [x * 0.1 for x in range(41)]:
            val = p(x_test)
            assert val >= -0.1  # Allow tiny numerical error
            assert val <= 1.1


# ============================================================
# Akima
# ============================================================

class TestAkima:
    def test_exact_at_nodes(self):
        xs = [0, 1, 2, 3, 4, 5]
        ys = [0, 1, 0, 1, 0, 1]
        a = AkimaInterpolator(xs, ys)
        for x, y in zip(xs, ys):
            assert a(x) == pytest.approx(y, abs=1e-10)

    def test_outlier_resistance(self):
        xs = [0, 1, 2, 3, 4, 5]
        ys = [0, 0, 0, 10, 0, 0]  # Outlier at x=3
        a = AkimaInterpolator(xs, ys)
        # At x=1.5, should not be significantly affected by outlier
        assert abs(a(0.5)) < 2.0

    def test_smooth(self):
        xs = [0, 1, 2, 3, 4]
        ys = [x**2 for x in xs]
        a = AkimaInterpolator(xs, ys)
        assert a(1.5) == pytest.approx(2.25, abs=0.5)

    def test_three_points(self):
        xs = [0, 1, 2]
        ys = [0, 1, 0]
        a = AkimaInterpolator(xs, ys)
        assert a(0.5) == pytest.approx(0.5, abs=0.5)


# ============================================================
# Rational Interpolation
# ============================================================

class TestRational:
    def test_polynomial_data(self):
        xs = [0, 1, 2, 3]
        ys = [1, 2, 5, 10]
        val = rational_interpolate(xs, ys, 1.5)
        expected = lagrange_interpolate(xs, ys, 1.5)
        assert val == pytest.approx(expected, abs=0.5)

    def test_exact_at_nodes(self):
        xs = [0, 1, 2]
        ys = [1, 0.5, 0.333]
        for x, y in zip(xs, ys):
            assert rational_interpolate(xs, ys, x) == pytest.approx(y, abs=1e-6)


# ============================================================
# RBF
# ============================================================

class TestRBF:
    def test_gaussian_exact_at_nodes(self):
        pts = [0, 1, 2, 3]
        vals = [0, 1, 0, -1]
        rbf = RBFInterpolator(pts, vals, kernel='gaussian', epsilon=1.0)
        for p, v in zip(pts, vals):
            assert rbf(p) == pytest.approx(v, abs=1e-6)

    def test_multiquadric(self):
        pts = [0, 1, 2]
        vals = [0, 1, 4]
        rbf = RBFInterpolator(pts, vals, kernel='multiquadric', epsilon=1.0)
        for p, v in zip(pts, vals):
            assert rbf(p) == pytest.approx(v, abs=1e-6)

    def test_2d_interpolation(self):
        pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        vals = [0, 1, 1, 2]
        rbf = RBFInterpolator(pts, vals, kernel='gaussian', epsilon=1.0)
        for p, v in zip(pts, vals):
            assert rbf(p) == pytest.approx(v, abs=1e-6)

    def test_inverse_multiquadric(self):
        pts = [0, 1, 2, 3, 4]
        vals = [math.sin(x) for x in pts]
        rbf = RBFInterpolator(pts, vals, kernel='inverse_multiquadric', epsilon=0.5)
        for p, v in zip(pts, vals):
            assert rbf(p) == pytest.approx(v, abs=1e-6)


# ============================================================
# B-Spline
# ============================================================

class TestBSpline:
    def test_linear_bspline(self):
        knots = [0, 0, 1, 1]
        cp = [0, 1]
        bs = BSpline(knots, cp, 1)
        assert bs(0.5) == pytest.approx(0.5)

    def test_quadratic_bspline(self):
        knots = [0, 0, 0, 1, 1, 1]
        cp = [0, 1, 0]
        bs = BSpline(knots, cp, 2)
        assert bs(0.0) == pytest.approx(0.0)
        assert bs(1.0) == pytest.approx(0.0)
        assert bs(0.5) == pytest.approx(0.5)

    def test_interpolating_bspline(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 0]
        bs = bspline_interpolate(xs, ys, degree=3)
        for x, y in zip(xs, ys):
            assert bs(x) == pytest.approx(y, abs=1e-6)

    def test_derivative(self):
        knots = [0, 0, 0, 1, 1, 1]
        cp = [0, 1, 0]
        bs = BSpline(knots, cp, 2)
        # Derivative at midpoint should be ~0
        assert abs(bs.derivative(0.5)) < 0.1

    def test_low_degree_interpolation(self):
        xs = [0, 1, 2]
        ys = [0, 1, 0]
        bs = bspline_interpolate(xs, ys, degree=2)
        for x, y in zip(xs, ys):
            assert bs(x) == pytest.approx(y, abs=1e-6)


# ============================================================
# Pade Approximant
# ============================================================

class TestPade:
    def test_exp_pade(self):
        # Taylor of exp(x): 1, 1, 1/2, 1/6, 1/24, ...
        coefs = [1.0, 1.0, 0.5, 1.0/6, 1.0/24, 1.0/120]
        p, q = pade_approximant(coefs, 2, 2)
        # Should approximate exp(x) well near 0
        for x in [-0.5, 0.0, 0.5]:
            val = pade_evaluate(p, q, x)
            assert val == pytest.approx(math.exp(x), abs=0.01)

    def test_trivial_pade(self):
        coefs = [1.0, 2.0, 3.0]
        p, q = pade_approximant(coefs, 2, 0)
        assert q == [1.0]
        assert p == pytest.approx([1.0, 2.0, 3.0])

    def test_pade_11(self):
        # [1/1] Pade of 1/(1-x) = 1 + x + x^2 + ...
        coefs = [1.0, 1.0, 1.0]
        p, q = pade_approximant(coefs, 1, 1)
        assert pade_evaluate(p, q, 0.5) == pytest.approx(2.0, abs=0.01)


# ============================================================
# Polynomial Fitting
# ============================================================

class TestPolyfit:
    def test_exact_fit(self):
        xs = [0, 1, 2]
        ys = [1, 3, 7]
        coefs = polyfit(xs, ys, 2)
        for x, y in zip(xs, ys):
            assert polyval(coefs, x) == pytest.approx(y, abs=1e-6)

    def test_linear_fit(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0.1, 1.9, 4.1, 5.9, 8.1]
        coefs = polyfit(xs, ys, 1)
        # Should be close to y = 2x
        assert coefs[1] == pytest.approx(2.0, abs=0.2)

    def test_polyval(self):
        coefs = [1, 2, 3]  # 1 + 2x + 3x^2
        assert polyval(coefs, 0) == pytest.approx(1.0)
        assert polyval(coefs, 1) == pytest.approx(6.0)
        assert polyval(coefs, 2) == pytest.approx(17.0)

    def test_exponential_fit(self):
        xs = [0, 1, 2, 3]
        a, b = 2.0, 0.5
        ys = [a * math.exp(b * x) for x in xs]
        a_fit, b_fit = exponential_fit(xs, ys)
        assert a_fit == pytest.approx(a, abs=0.1)
        assert b_fit == pytest.approx(b, abs=0.1)

    def test_power_fit(self):
        xs = [1, 2, 3, 4, 5]
        a, b = 3.0, 2.0
        ys = [a * x**b for x in xs]
        a_fit, b_fit = power_fit(xs, ys)
        assert a_fit == pytest.approx(a, abs=0.2)
        assert b_fit == pytest.approx(b, abs=0.1)


# ============================================================
# Bilinear / Bicubic
# ============================================================

class TestBilinear:
    def test_corners(self):
        xs = [0, 1]
        ys = [0, 1]
        zs = [[0, 1], [2, 3]]
        bl = BilinearInterpolator(xs, ys, zs)
        assert bl(0, 0) == pytest.approx(0.0)
        assert bl(1, 0) == pytest.approx(2.0)
        assert bl(0, 1) == pytest.approx(1.0)
        assert bl(1, 1) == pytest.approx(3.0)

    def test_center(self):
        xs = [0, 1]
        ys = [0, 1]
        zs = [[0, 0], [0, 4]]
        bl = BilinearInterpolator(xs, ys, zs)
        assert bl(0.5, 0.5) == pytest.approx(1.0)

    def test_larger_grid(self):
        xs = [0, 1, 2]
        ys = [0, 1, 2]
        zs = [[x + y for y in ys] for x in xs]
        bl = BilinearInterpolator(xs, ys, zs)
        assert bl(0.5, 0.5) == pytest.approx(1.0)
        assert bl(1.5, 1.5) == pytest.approx(3.0)


class TestBicubic:
    def test_exact_at_nodes(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 2, 3, 4]
        zs = [[float(x + y) for y in ys] for x in xs]
        bc = BicubicInterpolator(xs, ys, zs)
        assert bc(2, 2) == pytest.approx(4.0, abs=0.1)

    def test_smooth_surface(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 2, 3, 4]
        zs = [[math.sin(x) * math.cos(y) for y in ys] for x in xs]
        bc = BicubicInterpolator(xs, ys, zs)
        val = bc(2.0, 2.0)
        expected = math.sin(2.0) * math.cos(2.0)
        assert val == pytest.approx(expected, abs=0.2)


# ============================================================
# Trigonometric Interpolation
# ============================================================

class TestTrig:
    def test_exact_at_nodes(self):
        n = 5
        xs = [2 * math.pi * k / n for k in range(n)]
        ys = [math.sin(x) for x in xs]
        for x, y in zip(xs, ys):
            assert trig_interpolate(xs, ys, x) == pytest.approx(y, abs=1e-6)

    def test_sin_interpolation(self):
        n = 8
        xs = [2 * math.pi * k / n for k in range(n)]
        ys = [math.sin(x) for x in xs]
        val = trig_interpolate(xs, ys, math.pi / 4)
        assert val == pytest.approx(math.sin(math.pi / 4), abs=0.1)


# ============================================================
# Monotone Interpolation
# ============================================================

class TestMonotone:
    def test_monotone_increasing(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 3, 6, 10]
        m = MonotoneInterpolator(xs, ys)
        vals = [m(x * 0.5) for x in range(9)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-10

    def test_monotone_decreasing(self):
        xs = [0, 1, 2, 3, 4]
        ys = [10, 6, 3, 1, 0]
        m = MonotoneInterpolator(xs, ys)
        vals = [m(x * 0.5) for x in range(9)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-10

    def test_exact_at_nodes(self):
        xs = [0, 1, 2, 3]
        ys = [0, 2, 5, 9]
        m = MonotoneInterpolator(xs, ys)
        for x, y in zip(xs, ys):
            assert m(x) == pytest.approx(y, abs=1e-10)

    def test_two_points(self):
        m = MonotoneInterpolator([0, 1], [0, 1])
        assert m(0.5) == pytest.approx(0.5)


# ============================================================
# Tridiagonal Solver
# ============================================================

class TestTridiagonal:
    def test_simple(self):
        lower = [1.0, 1.0]
        diag = [2.0, 3.0, 2.0]
        upper = [1.0, 1.0]
        rhs = [1.0, 2.0, 3.0]
        x = _solve_tridiagonal(lower, diag, upper, rhs)
        # Verify Ax = rhs
        assert diag[0] * x[0] + upper[0] * x[1] == pytest.approx(rhs[0], abs=1e-6)

    def test_identity(self):
        n = 4
        x = _solve_tridiagonal([0]*3, [1]*4, [0]*3, [1, 2, 3, 4])
        assert x == pytest.approx([1, 2, 3, 4])


# ============================================================
# High-level API
# ============================================================

class TestInterpolateAPI:
    def test_linear(self):
        xs = [0, 1, 2]
        ys = [0, 1, 0]
        f = interpolate(xs, ys, method='linear')
        assert f(0.5) == pytest.approx(0.5)

    def test_cubic(self):
        xs = [0, 1, 2, 3]
        ys = [0, 1, 0, 1]
        f = interpolate(xs, ys, method='cubic')
        assert f(0) == pytest.approx(0.0)

    def test_pchip(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 0]
        f = interpolate(xs, ys, method='pchip')
        for x, y in zip(xs, ys):
            assert f(x) == pytest.approx(y, abs=1e-10)

    def test_akima(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 0]
        f = interpolate(xs, ys, method='akima')
        assert f(0) == pytest.approx(0.0)

    def test_monotone(self):
        xs = [0, 1, 2, 3]
        ys = [0, 1, 3, 6]
        f = interpolate(xs, ys, method='monotone')
        assert f(1.5) > 1.0

    def test_lagrange_api(self):
        xs = [0, 1, 2]
        ys = [0, 1, 4]
        f = interpolate(xs, ys, method='lagrange')
        assert f(1.5) == pytest.approx(2.25)

    def test_newton_api(self):
        xs = [0, 1, 2]
        ys = [0, 1, 4]
        f = interpolate(xs, ys, method='newton')
        assert f(1.5) == pytest.approx(2.25)

    def test_bspline_api(self):
        xs = [0, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 0]
        f = interpolate(xs, ys, method='bspline')
        for x, y in zip(xs, ys):
            assert f(x) == pytest.approx(y, abs=1e-5)

    def test_cubic_clamped_api(self):
        xs = [0, 1, 2]
        ys = [0, 1, 0]
        f = interpolate(xs, ys, method='cubic_clamped', clamped_slopes=(1.0, -1.0))
        assert f(0) == pytest.approx(0.0)

    def test_cubic_not_a_knot_api(self):
        xs = [0, 1, 2, 3, 4]
        ys = [x**3 for x in xs]
        f = interpolate(xs, ys, method='cubic_not_a_knot')
        assert f(2.5) == pytest.approx(2.5**3, abs=1e-5)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            interpolate([0, 1], [0, 1], method='unknown')


# ============================================================
# Edge cases and integration tests
# ============================================================

class TestEdgeCases:
    def test_lagrange_high_degree(self):
        n = 10
        xs = [i / (n - 1) for i in range(n)]
        ys = [math.sin(math.pi * x) for x in xs]
        val = lagrange_interpolate(xs, ys, 0.5)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_spline_sin(self):
        xs = [i * 0.5 for i in range(13)]  # 0 to 6
        ys = [math.sin(x) for x in xs]
        s = CubicSpline(xs, ys, bc_type='not-a-knot')
        for x_test in [0.25, 1.3, 2.7, 4.1, 5.5]:
            assert s(x_test) == pytest.approx(math.sin(x_test), abs=0.01)

    def test_chebyshev_runge(self):
        # Runge function: 1/(1+25x^2), Chebyshev handles it without Runge phenomenon
        f = lambda x: 1.0 / (1.0 + 25 * x**2)
        approx = ChebyshevApprox(f, 80, -1, 1)
        for x in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            assert approx(x) == pytest.approx(f(x), abs=1e-5)

    def test_rbf_cubic_kernel(self):
        pts = [0, 1, 2, 3]
        vals = [0, 1, 0, 1]
        rbf = RBFInterpolator(pts, vals, kernel='cubic', epsilon=0.5)
        for p, v in zip(pts, vals):
            assert rbf(p) == pytest.approx(v, abs=1e-6)

    def test_newton_high_order(self):
        xs = list(range(6))
        ys = [x**4 for x in xs]
        val = newton_interpolate(xs, ys, 2.5)
        assert val == pytest.approx(2.5**4, abs=1e-6)

    def test_pade_sin(self):
        # Taylor of sin(x): 0, 1, 0, -1/6, 0, 1/120
        coefs = [0, 1, 0, -1.0/6, 0, 1.0/120, 0]
        p, q = pade_approximant(coefs, 3, 2)
        val = pade_evaluate(p, q, 0.5)
        assert val == pytest.approx(math.sin(0.5), abs=0.01)

    def test_all_methods_quadratic(self):
        """All methods should reproduce a quadratic exactly (or very close)."""
        xs = [0, 1, 2, 3, 4]
        ys = [x**2 for x in xs]
        methods = ['linear', 'cubic', 'pchip', 'akima', 'monotone', 'lagrange', 'newton']
        for method in methods:
            f = interpolate(xs, ys, method=method)
            for x_test in [0.5, 1.5, 2.5, 3.5]:
                expected = x_test**2
                actual = f(x_test)
                # Linear won't be exact, others should be close
                if method == 'linear':
                    assert abs(actual - expected) < 1.0
                else:
                    assert actual == pytest.approx(expected, abs=0.5), f"{method} failed at {x_test}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
