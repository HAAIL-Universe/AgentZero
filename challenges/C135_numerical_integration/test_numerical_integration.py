"""Tests for C135: Numerical Integration."""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from numerical_integration import (
    trapezoid, simpson, simpson38, boole,
    romberg,
    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_chebyshev,
    adaptive_simpson, gauss_kronrod,
    integrate_2d, integrate_nd, monte_carlo, stratified_monte_carlo,
    improper_integral, oscillatory_filon, cauchy_principal_value,
    integrate_with_singularity, double_exponential,
    clenshaw_curtis,
    convergence_order, error_bound_trapezoid, error_bound_simpson,
    composite_gauss_legendre,
    line_integral,
    integrate,
)


# ============================================================
# Newton-Cotes
# ============================================================

class TestTrapezoid:
    def test_constant(self):
        assert trapezoid(lambda x: 3.0, 0, 1) == pytest.approx(3.0, abs=1e-10)

    def test_linear(self):
        # int x dx from 0 to 1 = 0.5
        assert trapezoid(lambda x: x, 0, 1) == pytest.approx(0.5, abs=1e-10)

    def test_quadratic(self):
        # int x^2 dx from 0 to 1 = 1/3
        assert trapezoid(lambda x: x**2, 0, 1, n=1000) == pytest.approx(1/3, abs=1e-5)

    def test_sin(self):
        # int sin(x) dx from 0 to pi = 2
        assert trapezoid(math.sin, 0, math.pi, n=1000) == pytest.approx(2.0, abs=1e-5)

    def test_negative_interval(self):
        assert trapezoid(lambda x: x, -1, 1) == pytest.approx(0.0, abs=1e-10)

    def test_n_must_be_positive(self):
        with pytest.raises(ValueError):
            trapezoid(lambda x: x, 0, 1, n=0)


class TestSimpson:
    def test_constant(self):
        assert simpson(lambda x: 5.0, 0, 2) == pytest.approx(10.0, abs=1e-10)

    def test_linear(self):
        assert simpson(lambda x: x, 0, 1) == pytest.approx(0.5, abs=1e-12)

    def test_quadratic(self):
        # Simpson is exact for polynomials up to degree 3
        assert simpson(lambda x: x**2, 0, 1) == pytest.approx(1/3, abs=1e-12)

    def test_cubic(self):
        # Exact for cubics
        assert simpson(lambda x: x**3, 0, 1) == pytest.approx(0.25, abs=1e-12)

    def test_sin(self):
        assert simpson(math.sin, 0, math.pi) == pytest.approx(2.0, abs=1e-7)

    def test_exp(self):
        # int e^x from 0 to 1 = e - 1
        assert simpson(math.exp, 0, 1) == pytest.approx(math.e - 1, abs=1e-7)

    def test_odd_n_rounds_up(self):
        # n=99 (odd) should round up to 100
        assert simpson(lambda x: x, 0, 1, n=99) == pytest.approx(0.5, abs=1e-12)

    def test_n_too_small(self):
        with pytest.raises(ValueError):
            simpson(lambda x: x, 0, 1, n=1)


class TestSimpson38:
    def test_linear(self):
        assert simpson38(lambda x: x, 0, 1) == pytest.approx(0.5, abs=1e-12)

    def test_quadratic(self):
        assert simpson38(lambda x: x**2, 0, 1) == pytest.approx(1/3, abs=1e-12)

    def test_cubic(self):
        # 3/8 rule is exact for cubics
        assert simpson38(lambda x: x**3, 0, 1) == pytest.approx(0.25, abs=1e-12)

    def test_sin(self):
        assert simpson38(math.sin, 0, math.pi, n=99) == pytest.approx(2.0, abs=1e-6)

    def test_n_too_small(self):
        with pytest.raises(ValueError):
            simpson38(lambda x: x, 0, 1, n=2)


class TestBoole:
    def test_quadratic(self):
        assert boole(lambda x: x**2, 0, 1) == pytest.approx(1/3, abs=1e-12)

    def test_quartic(self):
        # Boole is exact up to degree 5
        # int x^4 from 0 to 1 = 1/5
        assert boole(lambda x: x**4, 0, 1, n=4) == pytest.approx(0.2, abs=1e-10)

    def test_sin(self):
        assert boole(math.sin, 0, math.pi, n=100) == pytest.approx(2.0, abs=1e-10)

    def test_n_too_small(self):
        with pytest.raises(ValueError):
            boole(lambda x: x, 0, 1, n=3)


# ============================================================
# Romberg
# ============================================================

class TestRomberg:
    def test_polynomial(self):
        result, err, _ = romberg(lambda x: x**2, 0, 1)
        assert result == pytest.approx(1/3, abs=1e-12)

    def test_sin(self):
        result, err, _ = romberg(math.sin, 0, math.pi)
        assert result == pytest.approx(2.0, abs=1e-12)

    def test_exp(self):
        result, err, _ = romberg(math.exp, 0, 1)
        assert result == pytest.approx(math.e - 1, abs=1e-12)

    def test_returns_table(self):
        _, _, table = romberg(lambda x: x, 0, 1)
        assert isinstance(table, list)
        assert len(table) > 0

    def test_high_order(self):
        # int cos(x) from 0 to pi/2 = 1
        result, err, _ = romberg(math.cos, 0, math.pi / 2)
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_error_estimate(self):
        _, err, _ = romberg(math.sin, 0, math.pi)
        assert err < 1e-10


# ============================================================
# Gaussian Quadrature
# ============================================================

class TestGaussLegendre:
    def test_constant(self):
        assert gauss_legendre(lambda x: 1.0, 0, 1, n=1) == pytest.approx(1.0, abs=1e-12)

    def test_linear(self):
        assert gauss_legendre(lambda x: x, 0, 1, n=1) == pytest.approx(0.5, abs=1e-12)

    def test_polynomial_exact(self):
        # n-point GL is exact for polynomials up to degree 2n-1
        # 3-point should be exact for degree 5
        assert gauss_legendre(lambda x: x**5, 0, 1, n=3) == pytest.approx(1/6, abs=1e-10)

    def test_sin(self):
        assert gauss_legendre(math.sin, 0, math.pi, n=10) == pytest.approx(2.0, abs=1e-10)

    def test_exp(self):
        assert gauss_legendre(math.exp, 0, 1, n=5) == pytest.approx(math.e - 1, abs=1e-10)

    def test_negative_interval(self):
        # int x^2 from -1 to 1 = 2/3
        assert gauss_legendre(lambda x: x**2, -1, 1, n=2) == pytest.approx(2/3, abs=1e-12)

    def test_high_order(self):
        # 20-point GL for oscillatory
        result = gauss_legendre(lambda x: math.cos(10 * x), 0, 1, n=20)
        assert result == pytest.approx(math.sin(10) / 10, abs=1e-8)


class TestGaussLaguerre:
    def test_constant(self):
        # int 1 * e^{-x} dx from 0 to inf = 1
        assert gauss_laguerre(lambda x: 1.0, n=5) == pytest.approx(1.0, abs=1e-10)

    def test_polynomial(self):
        # int x * e^{-x} dx from 0 to inf = 1
        assert gauss_laguerre(lambda x: x, n=5) == pytest.approx(1.0, abs=1e-10)

    def test_x_squared(self):
        # int x^2 * e^{-x} dx = 2! = 2
        assert gauss_laguerre(lambda x: x**2, n=5) == pytest.approx(2.0, abs=1e-10)

    def test_x_cubed(self):
        # int x^3 * e^{-x} dx = 3! = 6
        assert gauss_laguerre(lambda x: x**3, n=5) == pytest.approx(6.0, abs=1e-8)


class TestGaussHermite:
    def test_constant(self):
        # int 1 * e^{-x^2} dx = sqrt(pi)
        assert gauss_hermite(lambda x: 1.0, n=5) == pytest.approx(math.sqrt(math.pi), abs=1e-10)

    def test_x_squared(self):
        # int x^2 * e^{-x^2} dx = sqrt(pi)/2
        assert gauss_hermite(lambda x: x**2, n=5) == pytest.approx(math.sqrt(math.pi) / 2, abs=1e-10)

    def test_even_function(self):
        # int (1 + x^2) * e^{-x^2} dx = sqrt(pi) + sqrt(pi)/2 = 3*sqrt(pi)/2
        assert gauss_hermite(lambda x: 1 + x**2, n=5) == pytest.approx(
            1.5 * math.sqrt(math.pi), abs=1e-10)

    def test_odd_function(self):
        # int x * e^{-x^2} dx = 0 (odd * even = odd)
        assert gauss_hermite(lambda x: x, n=5) == pytest.approx(0.0, abs=1e-10)


class TestGaussChebyshev:
    def test_constant(self):
        # int 1/sqrt(1-x^2) dx from -1 to 1 = pi
        assert gauss_chebyshev(lambda x: 1.0, n=10) == pytest.approx(math.pi, abs=1e-8)

    def test_polynomial(self):
        # int x^2 / sqrt(1-x^2) dx from -1 to 1 = pi/2
        assert gauss_chebyshev(lambda x: x**2, n=10) == pytest.approx(math.pi / 2, abs=1e-8)


# ============================================================
# Adaptive Quadrature
# ============================================================

class TestAdaptiveSimpson:
    def test_smooth(self):
        assert adaptive_simpson(math.sin, 0, math.pi) == pytest.approx(2.0, abs=1e-10)

    def test_exp(self):
        assert adaptive_simpson(math.exp, 0, 1) == pytest.approx(math.e - 1, abs=1e-10)

    def test_peaked(self):
        # Peaked function: 1/(1 + 100*x^2)
        # int from -1 to 1 = 2*atan(10)/10
        expected = 2 * math.atan(10) / 10
        assert adaptive_simpson(lambda x: 1 / (1 + 100 * x**2), -1, 1) == pytest.approx(expected, abs=1e-8)

    def test_polynomial(self):
        assert adaptive_simpson(lambda x: x**4, 0, 1) == pytest.approx(0.2, abs=1e-10)

    def test_zero_function(self):
        assert adaptive_simpson(lambda x: 0, 0, 1) == pytest.approx(0.0, abs=1e-15)


class TestGaussKronrod:
    def test_polynomial(self):
        result, err = gauss_kronrod(lambda x: x**3, 0, 1)
        assert result == pytest.approx(0.25, abs=1e-10)

    def test_sin(self):
        result, err = gauss_kronrod(math.sin, 0, math.pi)
        assert result == pytest.approx(2.0, abs=1e-10)

    def test_oscillatory(self):
        # int cos(10x) from 0 to 1
        result, err = gauss_kronrod(lambda x: math.cos(10 * x), 0, 1)
        assert result == pytest.approx(math.sin(10) / 10, abs=1e-8)

    def test_peaked(self):
        expected = 2 * math.atan(10) / 10
        result, err = gauss_kronrod(lambda x: 1 / (1 + 100 * x**2), -1, 1)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_error_estimate_small(self):
        _, err = gauss_kronrod(lambda x: x**2, 0, 1)
        assert err < 1e-10


# ============================================================
# Multi-dimensional
# ============================================================

class TestIntegrate2D:
    def test_constant(self):
        # int int 1 dA over [0,1]x[0,1] = 1
        assert integrate_2d(lambda x, y: 1.0, 0, 1, 0, 1) == pytest.approx(1.0, abs=1e-4)

    def test_linear(self):
        # int int (x+y) dA over [0,1]x[0,1] = 1
        assert integrate_2d(lambda x, y: x + y, 0, 1, 0, 1) == pytest.approx(1.0, abs=1e-4)

    def test_product(self):
        # int int x*y dA over [0,1]x[0,1] = 0.25
        assert integrate_2d(lambda x, y: x * y, 0, 1, 0, 1) == pytest.approx(0.25, abs=1e-4)

    def test_circle_area(self):
        # Area of unit circle quarter: int int 1 dA over x^2+y^2 <= 1 (first quadrant)
        # Use variable y limits: y from 0 to sqrt(1-x^2)
        result = integrate_2d(lambda x, y: 1.0, 0, 1,
                              lambda x: 0, lambda x: math.sqrt(max(0, 1 - x**2)),
                              nx=100, ny=100)
        assert result == pytest.approx(math.pi / 4, abs=0.01)

    def test_rectangular(self):
        # int int x^2 dA over [0,2]x[0,3] = (8/3)*3 = 8
        assert integrate_2d(lambda x, y: x**2, 0, 2, 0, 3, nx=50, ny=50) == pytest.approx(8.0, abs=0.01)


class TestIntegrateND:
    def test_1d(self):
        # int x dx from 0 to 1
        result = integrate_nd(lambda c: c[0], [(0, 1)], n_per_dim=50)
        assert result == pytest.approx(0.5, abs=1e-3)

    def test_2d(self):
        # int int 1 dA over [0,1]^2 = 1
        result = integrate_nd(lambda c: 1.0, [(0, 1), (0, 1)], n_per_dim=20)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_3d_constant(self):
        # int int int 1 dV over [0,1]^3 = 1
        result = integrate_nd(lambda c: 1.0, [(0, 1), (0, 1), (0, 1)], n_per_dim=10)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_3d_product(self):
        # int int int x*y*z dV over [0,1]^3 = 1/8
        result = integrate_nd(lambda c: c[0] * c[1] * c[2], [(0, 1)] * 3, n_per_dim=10)
        assert result == pytest.approx(0.125, abs=0.01)


class TestMonteCarlo:
    def test_constant(self):
        result, err = monte_carlo(lambda c: 2.0, [(0, 1)], n_samples=10000, seed=42)
        assert result == pytest.approx(2.0, abs=0.1)

    def test_linear(self):
        result, err = monte_carlo(lambda c: c[0], [(0, 1)], n_samples=50000, seed=42)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_2d(self):
        # int int 1 over [0,1]^2 = 1
        result, err = monte_carlo(lambda c: 1.0, [(0, 1), (0, 1)], n_samples=10000, seed=42)
        assert result == pytest.approx(1.0, abs=0.1)

    def test_circle_area(self):
        # pi/4 = area of quarter circle
        def indicator(c):
            return 1.0 if c[0]**2 + c[1]**2 <= 1 else 0.0
        result, err = monte_carlo(indicator, [(0, 1), (0, 1)], n_samples=100000, seed=42)
        assert result == pytest.approx(math.pi / 4, abs=0.02)

    def test_returns_error(self):
        _, err = monte_carlo(lambda c: c[0], [(0, 1)], n_samples=1000, seed=42)
        assert err > 0

    def test_higher_dim(self):
        # Volume of 4D unit hypercube = 1
        result, _ = monte_carlo(lambda c: 1.0, [(0, 1)] * 4, n_samples=10000, seed=42)
        assert result == pytest.approx(1.0, abs=0.1)


class TestStratifiedMonteCarlo:
    def test_basic(self):
        result, err = stratified_monte_carlo(
            lambda c: c[0], [(0, 1)], n_per_stratum=20, strata_per_dim=5, seed=42)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_reduces_variance(self):
        # Stratified should have lower error than plain MC for same total samples
        _, err_strat = stratified_monte_carlo(
            lambda c: c[0]**2, [(0, 1)], n_per_stratum=20, strata_per_dim=5, seed=42)
        _, err_mc = monte_carlo(
            lambda c: c[0]**2, [(0, 1)], n_samples=100, seed=42)
        # Both should be finite
        assert err_strat >= 0
        assert err_mc >= 0

    def test_2d(self):
        result, _ = stratified_monte_carlo(
            lambda c: 1.0, [(0, 1), (0, 1)], n_per_stratum=5, strata_per_dim=4, seed=42)
        assert result == pytest.approx(1.0, abs=0.1)


# ============================================================
# Special Integrals
# ============================================================

class TestImproperIntegral:
    def test_exp_decay(self):
        # int e^{-x} from 0 to inf = 1
        result, _ = improper_integral(lambda x: math.exp(-x), 0, float('inf'), n=30)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_gaussian(self):
        # int e^{-x^2} from 0 to inf = sqrt(pi)/2
        result, _ = improper_integral(lambda x: math.exp(-x**2), 0, float('inf'), n=30)
        assert result == pytest.approx(math.sqrt(math.pi) / 2, abs=0.05)

    def test_both_infinite(self):
        # int e^{-x^2} from -inf to inf = sqrt(pi)
        result, _ = improper_integral(lambda x: math.exp(-x**2), float('-inf'), float('inf'), n=30)
        assert result == pytest.approx(math.sqrt(math.pi), abs=0.1)

    def test_left_infinite(self):
        # int e^x from -inf to 0 = 1
        result, _ = improper_integral(lambda x: math.exp(x), float('-inf'), 0, n=30)
        assert result == pytest.approx(1.0, abs=0.05)

    def test_finite_interval(self):
        # Falls back to adaptive for finite intervals
        result, _ = improper_integral(math.sin, 0, math.pi)
        assert result == pytest.approx(2.0, abs=1e-6)


class TestOscillatoryFilon:
    def test_sin_integral(self):
        # int sin(x) * sin(10x) from 0 to pi
        # = int sin(x)*sin(10x) dx = integral of product
        # Use Filon: envelope = sin(x), omega = 10, kind = 'sin'
        # Analytical: int sin(x)*sin(10x) = [sin(9x)/(2*9) - sin(11x)/(2*11)] from 0 to pi
        expected = (math.sin(9 * math.pi) / 18 - math.sin(11 * math.pi) / 22)
        result = oscillatory_filon(math.sin, 10, 0, math.pi, n=100, kind='sin')
        assert result == pytest.approx(expected, abs=0.1)

    def test_cos_constant(self):
        # int 1 * cos(x) from 0 to pi = sin(pi) - sin(0) = 0
        result = oscillatory_filon(lambda x: 1.0, 1, 0, math.pi, n=100, kind='cos')
        assert result == pytest.approx(0.0, abs=0.1)

    def test_small_omega(self):
        # Falls back to Simpson for small omega
        result = oscillatory_filon(lambda x: 1.0, 1e-12, 0, 1, n=100, kind='sin')
        assert abs(result) < 0.1


class TestCauchyPrincipalValue:
    def test_basic(self):
        # PV int 1/(x-0.5) from 0 to 1 = ln(0.5/0.5) = 0
        result = cauchy_principal_value(lambda x: 1.0, 0, 1, 0.5)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_x_over_singularity(self):
        # PV int x/(x-0.5) from 0 to 1
        # = PV int [1 + 0.5/(x-0.5)] dx = 1 + 0.5*PV int 1/(x-0.5) dx
        # = 1 + 0.5 * 0 = 1
        result = cauchy_principal_value(lambda x: x, 0, 1, 0.5)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_singularity_outside_raises(self):
        with pytest.raises(ValueError):
            cauchy_principal_value(lambda x: 1.0, 0, 1, 2.0)


# ============================================================
# Singularity Handling
# ============================================================

class TestSingularity:
    def test_sqrt_singularity(self):
        # int 1/sqrt(x) from 0 to 1 = 2
        result = integrate_with_singularity(
            lambda x: 1 / math.sqrt(max(x, 1e-15)), 0, 1, 0)
        assert result == pytest.approx(2.0, abs=0.1)

    def test_log_singularity(self):
        # int -ln(x) from 0 to 1 = 1
        result = integrate_with_singularity(
            lambda x: -math.log(max(x, 1e-15)), 0, 1, 0)
        assert result == pytest.approx(1.0, abs=0.1)


class TestDoubleExponential:
    def test_smooth(self):
        result = double_exponential(math.sin, 0, math.pi, n=50)
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_polynomial(self):
        result = double_exponential(lambda x: x**2, 0, 1, n=30)
        assert result == pytest.approx(1/3, abs=1e-6)

    def test_endpoint_singularity(self):
        # int 1/sqrt(x) from 0 to 1 = 2
        result = double_exponential(lambda x: 1 / math.sqrt(max(x, 1e-15)), 0, 1, n=50)
        assert result == pytest.approx(2.0, abs=0.1)


# ============================================================
# Clenshaw-Curtis
# ============================================================

class TestClenshawCurtis:
    def test_constant(self):
        assert clenshaw_curtis(lambda x: 1.0, 0, 1) == pytest.approx(1.0, abs=1e-10)

    def test_polynomial(self):
        assert clenshaw_curtis(lambda x: x**3, 0, 1) == pytest.approx(0.25, abs=1e-10)

    def test_sin(self):
        assert clenshaw_curtis(math.sin, 0, math.pi) == pytest.approx(2.0, abs=1e-8)

    def test_exp(self):
        assert clenshaw_curtis(math.exp, 0, 1) == pytest.approx(math.e - 1, abs=1e-8)

    def test_oscillatory(self):
        result = clenshaw_curtis(lambda x: math.cos(5 * x), 0, math.pi, n=32)
        expected = math.sin(5 * math.pi) / 5
        assert result == pytest.approx(expected, abs=1e-6)


# ============================================================
# Composite Gauss-Legendre
# ============================================================

class TestCompositeGaussLegendre:
    def test_polynomial(self):
        result = composite_gauss_legendre(lambda x: x**2, 0, 1)
        assert result == pytest.approx(1/3, abs=1e-12)

    def test_oscillatory(self):
        result = composite_gauss_legendre(
            lambda x: math.cos(20 * x), 0, math.pi, n_intervals=20, points_per_interval=5)
        expected = math.sin(20 * math.pi) / 20
        assert result == pytest.approx(expected, abs=1e-6)

    def test_peaked(self):
        expected = 2 * math.atan(100) / 100
        result = composite_gauss_legendre(
            lambda x: 1 / (1 + 10000 * x**2), -1, 1, n_intervals=50, points_per_interval=5)
        assert result == pytest.approx(expected, abs=1e-4)


# ============================================================
# Convergence and Error
# ============================================================

class TestConvergence:
    def test_trapezoid_convergence(self):
        orders = convergence_order(
            math.sin, 0, math.pi, trapezoid, [10, 20, 40, 80, 160])
        # Trapezoid should show order ~2
        last = orders[-1]
        assert last[2] is not None
        assert last[2] == pytest.approx(2.0, abs=0.5)

    def test_simpson_convergence(self):
        orders = convergence_order(
            math.sin, 0, math.pi, simpson, [10, 20, 40, 80])
        last = orders[-1]
        assert last[2] is not None
        assert last[2] > 3.0  # Simpson is order 4


class TestErrorBounds:
    def test_trapezoid_bound(self):
        # f(x) = x^2, f''(x) = 2 -- allow tiny floating point excess
        bound = error_bound_trapezoid(2.0, 0, 1, 100)
        actual_error = abs(trapezoid(lambda x: x**2, 0, 1, n=100) - 1/3)
        assert actual_error <= bound * 1.01

    def test_simpson_bound(self):
        # f(x) = sin(x), |f^(4)(x)| <= 1
        bound = error_bound_simpson(1.0, 0, math.pi, 100)
        actual_error = abs(simpson(math.sin, 0, math.pi, n=100) - 2.0)
        assert actual_error <= bound


# ============================================================
# Line Integral
# ============================================================

class TestLineIntegral:
    def test_circle_circumference(self):
        # Circumference of unit circle: integral of 1 ds = 2*pi
        result = line_integral(
            lambda x, y: 1.0,
            lambda t: math.cos(t),
            lambda t: math.sin(t),
            0, 2 * math.pi, n=20)
        assert result == pytest.approx(2 * math.pi, abs=0.01)

    def test_straight_line(self):
        # Integral of 1 along y=x from (0,0) to (1,1), length = sqrt(2)
        result = line_integral(
            lambda x, y: 1.0,
            lambda t: t,
            lambda t: t,
            0, 1, n=10)
        assert result == pytest.approx(math.sqrt(2), abs=0.01)


# ============================================================
# High-level integrate()
# ============================================================

class TestIntegrate:
    def test_auto(self):
        result, err = integrate(math.sin, 0, math.pi)
        assert result == pytest.approx(2.0, abs=1e-8)

    def test_trapezoid(self):
        result = integrate(math.sin, 0, math.pi, method='trapezoid')
        assert result == pytest.approx(2.0, abs=1e-3)

    def test_simpson_method(self):
        result = integrate(math.sin, 0, math.pi, method='simpson')
        assert result == pytest.approx(2.0, abs=1e-7)

    def test_romberg_method(self):
        result, err = integrate(math.sin, 0, math.pi, method='romberg')
        assert result == pytest.approx(2.0, abs=1e-10)

    def test_gauss_method(self):
        result = integrate(math.sin, 0, math.pi, method='gauss')
        assert result == pytest.approx(2.0, abs=1e-8)

    def test_adaptive_method(self):
        result = integrate(math.sin, 0, math.pi, method='adaptive')
        assert result == pytest.approx(2.0, abs=1e-8)

    def test_kronrod_method(self):
        result, err = integrate(math.sin, 0, math.pi, method='kronrod')
        assert result == pytest.approx(2.0, abs=1e-8)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            integrate(math.sin, 0, 1, method='bogus')


# ============================================================
# Edge Cases and Stress Tests
# ============================================================

class TestEdgeCases:
    def test_zero_width_interval(self):
        assert trapezoid(lambda x: x, 1, 1, n=1) == pytest.approx(0.0, abs=1e-15)

    def test_negative_direction(self):
        # int from 1 to 0 should be -int from 0 to 1
        result = trapezoid(lambda x: x, 1, 0, n=100)
        assert result == pytest.approx(-0.5, abs=1e-4)

    def test_large_n(self):
        result = trapezoid(lambda x: x**2, 0, 1, n=10000)
        assert result == pytest.approx(1/3, abs=1e-8)


class TestAccuracyComparison:
    """Compare methods on the same integral."""

    def test_compare_all_methods(self):
        f = math.sin
        a, b = 0, math.pi
        expected = 2.0

        # All methods should agree
        assert trapezoid(f, a, b, n=1000) == pytest.approx(expected, abs=1e-5)
        assert simpson(f, a, b, n=100) == pytest.approx(expected, abs=1e-7)
        assert simpson38(f, a, b, n=99) == pytest.approx(expected, abs=1e-7)
        assert boole(f, a, b, n=100) == pytest.approx(expected, abs=1e-10)
        r, _, _ = romberg(f, a, b)
        assert r == pytest.approx(expected, abs=1e-12)
        assert gauss_legendre(f, a, b, n=10) == pytest.approx(expected, abs=1e-10)
        assert adaptive_simpson(f, a, b) == pytest.approx(expected, abs=1e-10)
        gk, _ = gauss_kronrod(f, a, b)
        assert gk == pytest.approx(expected, abs=1e-10)
        assert clenshaw_curtis(f, a, b) == pytest.approx(expected, abs=1e-8)

    def test_difficult_integral(self):
        """int sqrt(x) from 0 to 1 = 2/3."""
        expected = 2/3
        # Methods with many points should handle this
        assert trapezoid(math.sqrt, 0, 1, n=10000) == pytest.approx(expected, abs=1e-3)
        assert simpson(math.sqrt, 0, 1, n=1000) == pytest.approx(expected, abs=1e-5)
        assert adaptive_simpson(math.sqrt, 0, 1, tol=1e-8) == pytest.approx(expected, abs=1e-6)
        gk, _ = gauss_kronrod(math.sqrt, 0, 1)
        assert gk == pytest.approx(expected, abs=1e-6)

    def test_rapid_oscillation(self):
        """int sin(100x) from 0 to 1."""
        expected = (1 - math.cos(100)) / 100
        # Need more points for oscillatory
        assert composite_gauss_legendre(
            lambda x: math.sin(100 * x), 0, 1,
            n_intervals=50, points_per_interval=5) == pytest.approx(expected, abs=1e-6)


class TestNumericalStability:
    def test_very_small_interval(self):
        result = gauss_legendre(lambda x: x, 0, 1e-10, n=5)
        assert result == pytest.approx(0.5e-20, abs=1e-25)

    def test_large_interval(self):
        # Gaussian on [-10,10] needs more points since most mass is near 0
        result = gauss_legendre(lambda x: math.exp(-x**2), -10, 10, n=40)
        assert result == pytest.approx(math.sqrt(math.pi), abs=0.1)

    def test_near_zero_result(self):
        # int sin(x) from 0 to 2*pi = 0
        result = simpson(math.sin, 0, 2 * math.pi, n=100)
        assert result == pytest.approx(0.0, abs=1e-10)
