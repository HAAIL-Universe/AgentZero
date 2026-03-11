"""Tests for C138: Numerical Optimization Library."""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from optimization import (
    # Unconstrained
    steepest_descent, newton, bfgs, lbfgs, conjugate_gradient,
    nelder_mead, trust_region, minimize,
    # Constrained
    projected_gradient, augmented_lagrangian, penalty_method, sqp,
    # Least squares
    gauss_newton, levenberg_marquardt, curve_fit,
    # Root finding
    newton_raphson, bisection, brent, newton_system,
    # Line search
    backtracking_line_search, wolfe_line_search, golden_section_search,
    # Utilities
    numerical_gradient, numerical_hessian, numerical_jacobian,
    OptimizeResult,
)


# ===========================================================================
# Test functions
# ===========================================================================

def rosenbrock(x):
    """Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2. Min at (1,1)."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def quadratic(x):
    """Simple quadratic: f(x) = x0^2 + x1^2. Min at (0,0)."""
    return x[0]**2 + x[1]**2

def beale(x):
    """Beale's function. Min at (3, 0.5)."""
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def booth(x):
    """Booth: (x+2y-7)^2 + (2x+y-5)^2. Min at (1,3)."""
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def sphere(x):
    """Sphere: sum(xi^2). Min at origin."""
    return sum(xi**2 for xi in x)

def matyas(x):
    """Matyas: 0.26(x^2+y^2) - 0.48xy. Min at (0,0)."""
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


# ===========================================================================
# Numerical utilities tests
# ===========================================================================

class TestNumericalGradient:
    def test_quadratic_gradient(self):
        g = numerical_gradient(quadratic, [3.0, 4.0])
        assert abs(g[0] - 6.0) < 1e-5
        assert abs(g[1] - 8.0) < 1e-5

    def test_rosenbrock_gradient_at_minimum(self):
        g = numerical_gradient(rosenbrock, [1.0, 1.0])
        assert abs(g[0]) < 1e-5
        assert abs(g[1]) < 1e-5

    def test_high_dimensional(self):
        g = numerical_gradient(sphere, [1.0, 2.0, 3.0, 4.0])
        for i, gi in enumerate(g):
            assert abs(gi - 2.0 * (i + 1)) < 1e-5


class TestNumericalHessian:
    def test_quadratic_hessian(self):
        H = numerical_hessian(quadratic, [1.0, 1.0])
        assert abs(H[0][0] - 2.0) < 1e-3
        assert abs(H[1][1] - 2.0) < 1e-3
        assert abs(H[0][1]) < 1e-3
        assert abs(H[1][0]) < 1e-3

    def test_symmetric(self):
        H = numerical_hessian(rosenbrock, [0.5, 0.5])
        assert abs(H[0][1] - H[1][0]) < 1e-3


class TestNumericalJacobian:
    def test_scalar_function(self):
        J = numerical_jacobian(lambda x: [x[0]**2 + x[1]], [2.0, 3.0])
        assert abs(J[0][0] - 4.0) < 1e-5
        assert abs(J[0][1] - 1.0) < 1e-5

    def test_vector_function(self):
        def F(x):
            return [x[0] + x[1], x[0] * x[1]]
        J = numerical_jacobian(F, [2.0, 3.0])
        assert abs(J[0][0] - 1.0) < 1e-5  # df1/dx1
        assert abs(J[0][1] - 1.0) < 1e-5  # df1/dx2
        assert abs(J[1][0] - 3.0) < 1e-5  # df2/dx1
        assert abs(J[1][1] - 2.0) < 1e-5  # df2/dx2


# ===========================================================================
# Line search tests
# ===========================================================================

class TestBacktracking:
    def test_basic(self):
        alpha, fnew, nf = backtracking_line_search(
            quadratic, [1.0, 1.0], [-1.0, -1.0], [2.0, 2.0])
        assert alpha > 0
        assert fnew < quadratic([1.0, 1.0])

    def test_not_descent(self):
        # Direction is not descent
        alpha, fnew, nf = backtracking_line_search(
            quadratic, [1.0, 1.0], [1.0, 1.0], [2.0, 2.0])
        assert alpha <= 1e-9


class TestWolfeLineSearch:
    def test_basic(self):
        def gf(x):
            return [2*x[0], 2*x[1]]
        alpha, fnew, gnew, nf = wolfe_line_search(
            quadratic, gf, [1.0, 1.0], [-1.0, -1.0])
        assert alpha > 0
        assert fnew < quadratic([1.0, 1.0])


class TestGoldenSection:
    def test_parabola(self):
        xmin, fmin, nf = golden_section_search(lambda x: (x - 3)**2, 0, 10)
        assert abs(xmin - 3.0) < 1e-6
        assert abs(fmin) < 1e-10

    def test_shifted(self):
        xmin, fmin, nf = golden_section_search(lambda x: (x + 2)**2 + 1, -5, 5)
        assert abs(xmin + 2.0) < 1e-6
        assert abs(fmin - 1.0) < 1e-10


# ===========================================================================
# Steepest Descent tests
# ===========================================================================

class TestSteepestDescent:
    def test_quadratic(self):
        result = steepest_descent(quadratic, [5.0, 5.0])
        assert result.success
        assert abs(result.x[0]) < 1e-5
        assert abs(result.x[1]) < 1e-5

    def test_booth(self):
        result = steepest_descent(booth, [0.0, 0.0], max_iter=5000)
        assert result.success
        assert abs(result.x[0] - 1.0) < 1e-3
        assert abs(result.x[1] - 3.0) < 1e-3

    def test_with_history(self):
        result = steepest_descent(quadratic, [5.0, 5.0], history=True)
        assert result.success
        assert len(result.history) > 0
        # History should be monotonically decreasing
        for i in range(1, len(result.history)):
            assert result.history[i] <= result.history[i-1] + 1e-10

    def test_fixed_lr(self):
        result = steepest_descent(quadratic, [5.0, 5.0], lr=0.1,
                                   line_search=None, max_iter=200)
        assert abs(result.x[0]) < 1e-4
        assert abs(result.x[1]) < 1e-4

    def test_wolfe_line_search(self):
        result = steepest_descent(quadratic, [5.0, 5.0], line_search='wolfe')
        assert result.success

    def test_numerical_grad(self):
        result = steepest_descent(quadratic, [5.0, 5.0], use_ad=False)
        assert result.success
        assert abs(result.x[0]) < 1e-4

    def test_custom_gradient(self):
        def grad_q(x):
            return [2*x[0], 2*x[1]]
        result = steepest_descent(quadratic, [5.0, 5.0], grad_f=grad_q)
        assert result.success


# ===========================================================================
# Newton's Method tests
# ===========================================================================

class TestNewton:
    def test_quadratic(self):
        result = newton(quadratic, [5.0, 5.0])
        assert result.success
        assert abs(result.x[0]) < 1e-6
        assert abs(result.x[1]) < 1e-6
        assert result.nit <= 5  # Quadratic should converge in 1 step

    def test_rosenbrock(self):
        result = newton(rosenbrock, [0.0, 0.0], max_iter=200)
        assert abs(result.x[0] - 1.0) < 1e-3
        assert abs(result.x[1] - 1.0) < 1e-3

    def test_with_custom_hessian(self):
        def hess_q(x):
            return [[2.0, 0.0], [0.0, 2.0]]
        result = newton(quadratic, [5.0, 5.0], hess_f=hess_q)
        assert result.success
        assert result.nit <= 2

    def test_no_line_search(self):
        result = newton(quadratic, [5.0, 5.0], line_search=None)
        assert result.success

    def test_numerical(self):
        result = newton(quadratic, [5.0, 5.0], use_ad=False)
        assert result.success


# ===========================================================================
# BFGS tests
# ===========================================================================

class TestBFGS:
    def test_quadratic(self):
        result = bfgs(quadratic, [5.0, 5.0])
        assert result.success
        assert abs(result.x[0]) < 1e-6
        assert abs(result.x[1]) < 1e-6

    def test_rosenbrock(self):
        result = bfgs(rosenbrock, [-1.0, 1.0], max_iter=500)
        assert abs(result.x[0] - 1.0) < 1e-3
        assert abs(result.x[1] - 1.0) < 1e-3

    def test_booth(self):
        result = bfgs(booth, [0.0, 0.0])
        assert result.success
        assert abs(result.x[0] - 1.0) < 1e-4
        assert abs(result.x[1] - 3.0) < 1e-4

    def test_matyas(self):
        result = bfgs(matyas, [5.0, 5.0])
        assert result.success
        assert abs(result.x[0]) < 1e-4
        assert abs(result.x[1]) < 1e-4

    def test_history(self):
        result = bfgs(quadratic, [5.0, 5.0], history=True)
        assert len(result.history) > 0

    def test_high_dim_sphere(self):
        x0 = [float(i) for i in range(5)]
        result = bfgs(sphere, x0)
        assert result.success
        for xi in result.x:
            assert abs(xi) < 1e-4


# ===========================================================================
# L-BFGS tests
# ===========================================================================

class TestLBFGS:
    def test_quadratic(self):
        result = lbfgs(quadratic, [5.0, 5.0])
        assert result.success
        assert abs(result.x[0]) < 1e-6

    def test_rosenbrock(self):
        result = lbfgs(rosenbrock, [-1.0, 1.0], max_iter=500)
        assert abs(result.x[0] - 1.0) < 1e-3
        assert abs(result.x[1] - 1.0) < 1e-3

    def test_high_dim(self):
        x0 = [float(i) for i in range(10)]
        result = lbfgs(sphere, x0, m=5)
        assert result.success
        for xi in result.x:
            assert abs(xi) < 1e-4

    def test_memory_limit(self):
        # m=2 should still work, just slower
        result = lbfgs(quadratic, [5.0, 5.0], m=2)
        assert result.success


# ===========================================================================
# Conjugate Gradient tests
# ===========================================================================

class TestConjugateGradient:
    def test_quadratic_FR(self):
        result = conjugate_gradient(quadratic, [5.0, 5.0], method='FR')
        assert result.success

    def test_quadratic_PR(self):
        result = conjugate_gradient(quadratic, [5.0, 5.0], method='PR')
        assert result.success

    def test_quadratic_PR_plus(self):
        result = conjugate_gradient(quadratic, [5.0, 5.0], method='PR+')
        assert result.success

    def test_quadratic_HS(self):
        result = conjugate_gradient(quadratic, [5.0, 5.0], method='HS')
        assert result.success

    def test_rosenbrock(self):
        result = conjugate_gradient(rosenbrock, [0.0, 0.0], max_iter=5000)
        assert abs(result.x[0] - 1.0) < 0.1
        assert abs(result.x[1] - 1.0) < 0.1

    def test_booth(self):
        result = conjugate_gradient(booth, [0.0, 0.0])
        assert result.success
        assert abs(result.x[0] - 1.0) < 1e-3


# ===========================================================================
# Nelder-Mead tests
# ===========================================================================

class TestNelderMead:
    def test_quadratic(self):
        result = nelder_mead(quadratic, [5.0, 5.0])
        assert result.success
        assert abs(result.x[0]) < 1e-4
        assert abs(result.x[1]) < 1e-4

    def test_rosenbrock(self):
        result = nelder_mead(rosenbrock, [-1.0, 1.0], max_iter=5000)
        assert abs(result.x[0] - 1.0) < 0.05
        assert abs(result.x[1] - 1.0) < 0.05

    def test_booth(self):
        result = nelder_mead(booth, [0.0, 0.0], max_iter=2000)
        assert abs(result.x[0] - 1.0) < 0.01
        assert abs(result.x[1] - 3.0) < 0.01

    def test_no_gradient_needed(self):
        # Nelder-Mead works on non-smooth functions
        result = nelder_mead(lambda x: abs(x[0] - 3) + abs(x[1] - 2),
                              [0.0, 0.0], max_iter=3000)
        assert abs(result.x[0] - 3.0) < 0.1
        assert abs(result.x[1] - 2.0) < 0.1

    def test_non_adaptive(self):
        result = nelder_mead(quadratic, [5.0, 5.0], adaptive=False)
        assert result.success

    def test_initial_simplex(self):
        simplex = [[5.0, 5.0], [6.0, 5.0], [5.0, 6.0]]
        result = nelder_mead(quadratic, [5.0, 5.0], initial_simplex=simplex)
        assert result.success

    def test_history(self):
        result = nelder_mead(quadratic, [5.0, 5.0], history=True)
        assert len(result.history) > 0

    def test_1d(self):
        result = nelder_mead(lambda x: (x[0] - 3)**2, [0.0])
        assert abs(result.x[0] - 3.0) < 1e-4

    def test_3d(self):
        result = nelder_mead(sphere, [1.0, 2.0, 3.0], max_iter=3000)
        for xi in result.x:
            assert abs(xi) < 0.01


# ===========================================================================
# Trust Region tests
# ===========================================================================

class TestTrustRegion:
    def test_quadratic(self):
        result = trust_region(quadratic, [5.0, 5.0])
        assert result.success
        assert abs(result.x[0]) < 1e-5
        assert abs(result.x[1]) < 1e-5

    def test_rosenbrock(self):
        result = trust_region(rosenbrock, [0.0, 0.0], max_iter=500)
        assert abs(result.x[0] - 1.0) < 0.05
        assert abs(result.x[1] - 1.0) < 0.05

    def test_with_history(self):
        result = trust_region(quadratic, [5.0, 5.0], history=True)
        assert len(result.history) > 0


# ===========================================================================
# Projected Gradient tests
# ===========================================================================

class TestProjectedGradient:
    def test_unconstrained_equivalent(self):
        bounds = [(None, None), (None, None)]
        result = projected_gradient(quadratic, [5.0, 5.0], bounds)
        assert result.success
        assert abs(result.x[0]) < 1e-4

    def test_box_constrained(self):
        # Min of x^2 + y^2 with x >= 1, y >= 2
        bounds = [(1.0, None), (2.0, None)]
        result = projected_gradient(quadratic, [5.0, 5.0], bounds)
        assert result.success
        assert abs(result.x[0] - 1.0) < 1e-4
        assert abs(result.x[1] - 2.0) < 1e-4

    def test_upper_bounds(self):
        # Min of (x-5)^2 + (y-5)^2 with x <= 2, y <= 3
        def f(x):
            return (x[0] - 5)**2 + (x[1] - 5)**2
        bounds = [(None, 2.0), (None, 3.0)]
        result = projected_gradient(f, [0.0, 0.0], bounds)
        assert result.success
        assert abs(result.x[0] - 2.0) < 1e-3
        assert abs(result.x[1] - 3.0) < 1e-3

    def test_both_bounds(self):
        bounds = [(0.5, 2.0), (0.5, 2.0)]
        result = projected_gradient(quadratic, [5.0, 5.0], bounds)
        assert 0.5 - 1e-6 <= result.x[0] <= 2.0 + 1e-6
        assert 0.5 - 1e-6 <= result.x[1] <= 2.0 + 1e-6


# ===========================================================================
# Augmented Lagrangian tests
# ===========================================================================

class TestAugmentedLagrangian:
    def test_equality_constraint(self):
        # min x^2 + y^2 s.t. x + y = 1
        # Solution: x = y = 0.5
        result = augmented_lagrangian(
            quadratic, [2.0, 2.0],
            eq_constraints=[lambda x: x[0] + x[1] - 1])
        assert result.success
        assert abs(result.x[0] - 0.5) < 0.01
        assert abs(result.x[1] - 0.5) < 0.01

    def test_inequality_constraint(self):
        # min x^2 + y^2 s.t. x >= 1 (i.e., -x + 1 <= 0)
        result = augmented_lagrangian(
            quadratic, [5.0, 5.0],
            ineq_constraints=[lambda x: -x[0] + 1])
        assert result.success
        assert abs(result.x[0] - 1.0) < 0.05
        assert abs(result.x[1]) < 0.05

    def test_both_constraints(self):
        # min x^2 + y^2 s.t. x + y = 2, x >= 0.5
        result = augmented_lagrangian(
            quadratic, [3.0, 3.0],
            eq_constraints=[lambda x: x[0] + x[1] - 2],
            ineq_constraints=[lambda x: -x[0] + 0.5])
        assert result.success
        assert abs(result.x[0] + result.x[1] - 2.0) < 0.05


# ===========================================================================
# Penalty Method tests
# ===========================================================================

class TestPenaltyMethod:
    def test_equality(self):
        # min x^2 + y^2 s.t. x + y = 1
        result = penalty_method(
            quadratic, [2.0, 2.0],
            eq_constraints=[lambda x: x[0] + x[1] - 1])
        assert result.success
        assert abs(result.x[0] - 0.5) < 0.05
        assert abs(result.x[1] - 0.5) < 0.05

    def test_inequality(self):
        # min x^2 + y^2 s.t. x >= 2
        result = penalty_method(
            quadratic, [5.0, 5.0],
            ineq_constraints=[lambda x: -x[0] + 2])
        assert result.success
        assert abs(result.x[0] - 2.0) < 0.1


# ===========================================================================
# SQP tests
# ===========================================================================

class TestSQP:
    def test_equality(self):
        # min x^2 + y^2 s.t. x + y = 1
        result = sqp(quadratic, [2.0, 2.0],
                      eq_constraints=[lambda x: x[0] + x[1] - 1])
        assert abs(result.x[0] - 0.5) < 0.1
        assert abs(result.x[1] - 0.5) < 0.1

    def test_inequality(self):
        # min (x-2)^2 + (y-1)^2 s.t. x + y <= 2, x >= 0, y >= 0
        def f(x):
            return (x[0] - 2)**2 + (x[1] - 1)**2
        result = sqp(f, [0.5, 0.5],
                      ineq_constraints=[
                          lambda x: x[0] + x[1] - 2,
                          lambda x: -x[0],
                          lambda x: -x[1],
                      ])
        # Constraint should be approximately satisfied
        assert result.x[0] + result.x[1] <= 2.1


# ===========================================================================
# Gauss-Newton tests
# ===========================================================================

class TestGaussNewton:
    def test_linear_fit(self):
        # Fit y = a*x + b to data
        xdata = [1.0, 2.0, 3.0, 4.0, 5.0]
        ydata = [2.1, 3.9, 6.1, 8.0, 9.9]  # approx y = 2x + 0

        def residuals(p):
            return [p[0] * xdata[i] + p[1] - ydata[i] for i in range(len(xdata))]

        result = gauss_newton(residuals, [0.0, 0.0])
        assert result.success
        assert abs(result.x[0] - 2.0) < 0.1  # slope ~ 2
        assert abs(result.x[1]) < 0.5  # intercept ~ 0

    def test_exponential_fit(self):
        # Fit y = a * exp(b * x)
        xdata = [0.0, 0.5, 1.0, 1.5, 2.0]
        ydata = [1.0, 1.65, 2.72, 4.48, 7.39]  # exp(x)

        def residuals(p):
            return [p[0] * math.exp(p[1] * xdata[i]) - ydata[i]
                    for i in range(len(xdata))]

        result = gauss_newton(residuals, [1.0, 0.5])
        assert abs(result.x[0] - 1.0) < 0.1
        assert abs(result.x[1] - 1.0) < 0.1


# ===========================================================================
# Levenberg-Marquardt tests
# ===========================================================================

class TestLevenbergMarquardt:
    def test_linear_fit(self):
        xdata = [1.0, 2.0, 3.0, 4.0, 5.0]
        ydata = [2.1, 3.9, 6.1, 8.0, 9.9]

        def residuals(p):
            return [p[0] * xdata[i] + p[1] - ydata[i] for i in range(len(xdata))]

        result = levenberg_marquardt(residuals, [0.0, 0.0])
        assert result.success
        assert abs(result.x[0] - 2.0) < 0.1

    def test_circle_fit(self):
        # Fit circle: (x-cx)^2 + (y-cy)^2 = r^2
        # Points on circle centered at (1, 1) radius 2
        angles = [i * math.pi / 4 for i in range(8)]
        px = [1 + 2 * math.cos(a) for a in angles]
        py = [1 + 2 * math.sin(a) for a in angles]

        def residuals(p):
            cx, cy, r = p
            return [math.sqrt((px[i]-cx)**2 + (py[i]-cy)**2) - r
                    for i in range(len(px))]

        result = levenberg_marquardt(residuals, [0.0, 0.0, 1.0])
        assert abs(result.x[0] - 1.0) < 0.01
        assert abs(result.x[1] - 1.0) < 0.01
        assert abs(result.x[2] - 2.0) < 0.01

    def test_convergence_from_bad_start(self):
        def residuals(p):
            return [p[0]**2 - 4, p[0] + p[1] - 3]
        result = levenberg_marquardt(residuals, [10.0, 10.0])
        # Should find (2, 1) or (-2, 5)
        assert result.fun < 1e-6


# ===========================================================================
# Curve fitting tests
# ===========================================================================

class TestCurveFit:
    def test_linear(self):
        xdata = [1.0, 2.0, 3.0, 4.0, 5.0]
        ydata = [2.5, 4.5, 6.5, 8.5, 10.5]

        def model(x, a, b):
            return a * x + b

        result = curve_fit(model, xdata, ydata, [0.0, 0.0])
        assert abs(result.x[0] - 2.0) < 0.01
        assert abs(result.x[1] - 0.5) < 0.05

    def test_exponential(self):
        xdata = list(range(5))
        ydata = [2 * math.exp(0.5 * x) for x in xdata]

        def model(x, a, b):
            return a * math.exp(b * x)

        result = curve_fit(model, xdata, ydata, [1.0, 0.3])
        assert abs(result.x[0] - 2.0) < 0.1
        assert abs(result.x[1] - 0.5) < 0.1

    def test_polynomial(self):
        xdata = [-2, -1, 0, 1, 2]
        # y = 3x^2 + 2x + 1
        ydata = [3*x**2 + 2*x + 1 for x in xdata]

        def model(x, a, b, c):
            return a * x**2 + b * x + c

        result = curve_fit(model, xdata, ydata, [0.0, 0.0, 0.0])
        assert abs(result.x[0] - 3.0) < 0.01
        assert abs(result.x[1] - 2.0) < 0.01
        assert abs(result.x[2] - 1.0) < 0.01

    def test_gauss_newton_method(self):
        xdata = [1.0, 2.0, 3.0]
        ydata = [2.0, 4.0, 6.0]
        def model(x, a, b):
            return a * x + b
        result = curve_fit(model, xdata, ydata, [0.0, 0.0], method='gn')
        assert abs(result.x[0] - 2.0) < 0.01

    def test_covariance_returned(self):
        xdata = list(range(10))
        ydata = [2.0 * x + 1.0 + 0.01 * (x % 3) for x in xdata]
        def model(x, a, b):
            return a * x + b
        result = curve_fit(model, xdata, ydata, [0.0, 0.0])
        assert result.pcov is not None
        assert len(result.pcov) == 2


# ===========================================================================
# Root finding tests
# ===========================================================================

class TestNewtonRaphson:
    def test_sqrt2(self):
        result = newton_raphson(lambda x: x**2 - 2, 1.5)
        assert result.success
        assert abs(result.x[0] - math.sqrt(2)) < 1e-8

    def test_with_derivative(self):
        result = newton_raphson(lambda x: x**2 - 2, 1.5, df=lambda x: 2*x)
        assert result.success
        assert abs(result.x[0] - math.sqrt(2)) < 1e-10

    def test_cubic(self):
        # x^3 - x - 1 = 0, real root ~ 1.3247
        result = newton_raphson(lambda x: x**3 - x - 1, 1.5)
        assert result.success
        assert abs(result.x[0]**3 - result.x[0] - 1) < 1e-8

    def test_trig(self):
        # cos(x) = x, root ~ 0.7391
        result = newton_raphson(lambda x: math.cos(x) - x, 0.5)
        assert result.success
        assert abs(math.cos(result.x[0]) - result.x[0]) < 1e-8


class TestBisection:
    def test_sqrt2(self):
        result = bisection(lambda x: x**2 - 2, 1, 2)
        assert result.success
        assert abs(result.x[0] - math.sqrt(2)) < 1e-8

    def test_sign_check(self):
        with pytest.raises(ValueError):
            bisection(lambda x: x**2, 1, 2)

    def test_negative_root(self):
        result = bisection(lambda x: x + 1, -3, 0)
        assert result.success
        assert abs(result.x[0] + 1.0) < 1e-8


class TestBrent:
    def test_sqrt2(self):
        result = brent(lambda x: x**2 - 2, 1, 2)
        assert result.success
        assert abs(result.x[0] - math.sqrt(2)) < 1e-8

    def test_cubic(self):
        result = brent(lambda x: x**3 - x - 1, 1, 2)
        assert result.success
        assert abs(result.x[0]**3 - result.x[0] - 1) < 1e-8

    def test_converges_faster_than_bisection(self):
        f = lambda x: x**3 - 2*x - 5
        r1 = bisection(f, 2, 3)
        r2 = brent(f, 2, 3)
        # Brent should use fewer evaluations (or same)
        assert r2.nfev <= r1.nfev + 5  # Allow some margin


class TestNewtonSystem:
    def test_linear(self):
        # x + y = 3, x - y = 1 => x=2, y=1
        def F(x):
            return [x[0] + x[1] - 3, x[0] - x[1] - 1]
        result = newton_system(F, [0.0, 0.0])
        assert result.success
        assert abs(result.x[0] - 2.0) < 1e-8
        assert abs(result.x[1] - 1.0) < 1e-8

    def test_nonlinear(self):
        # x^2 + y^2 = 4, x*y = 1
        def F(x):
            return [x[0]**2 + x[1]**2 - 4, x[0]*x[1] - 1]
        result = newton_system(F, [1.5, 0.5])
        assert result.success
        assert abs(result.x[0]**2 + result.x[1]**2 - 4) < 1e-6
        assert abs(result.x[0] * result.x[1] - 1) < 1e-6

    def test_3d(self):
        # 3 linear equations, 3 unknowns: x+y+z=6, 2x-y+z=3, x+y-z=2
        # Solution: x=1.5, y=2.5, z=2
        def F(x):
            return [x[0] + x[1] + x[2] - 6,
                    2*x[0] - x[1] + x[2] - 3,
                    x[0] + x[1] - x[2] - 2]
        result = newton_system(F, [0.0, 0.0, 0.0])
        assert result.success
        assert abs(result.fun) < 1e-6

    def test_with_jacobian(self):
        def F(x):
            return [x[0] + x[1] - 3, x[0] - x[1] - 1]
        def J(x):
            return [[1, 1], [1, -1]]
        result = newton_system(F, [0.0, 0.0], J=J)
        assert result.success


# ===========================================================================
# minimize() unified interface tests
# ===========================================================================

class TestMinimize:
    def test_all_methods_quadratic(self):
        methods = ['steepest_descent', 'newton', 'bfgs', 'lbfgs', 'cg',
                    'nelder_mead', 'trust_region']
        for method in methods:
            result = minimize(quadratic, [5.0, 5.0], method=method)
            assert abs(result.x[0]) < 0.1, f"{method} failed"
            assert abs(result.x[1]) < 0.1, f"{method} failed"

    def test_projected_gradient_via_minimize(self):
        bounds = [(1.0, None), (2.0, None)]
        result = minimize(quadratic, [5.0, 5.0], method='projected_gradient',
                          bounds=bounds)
        assert abs(result.x[0] - 1.0) < 0.01

    def test_augmented_lagrangian_via_minimize(self):
        result = minimize(quadratic, [2.0, 2.0], method='augmented_lagrangian',
                          eq_constraints=[lambda x: x[0] + x[1] - 1])
        assert abs(result.x[0] - 0.5) < 0.05

    def test_penalty_via_minimize(self):
        result = minimize(quadratic, [2.0, 2.0], method='penalty',
                          eq_constraints=[lambda x: x[0] + x[1] - 1])
        assert abs(result.x[0] - 0.5) < 0.1

    def test_sqp_via_minimize(self):
        result = minimize(quadratic, [2.0, 2.0], method='sqp',
                          eq_constraints=[lambda x: x[0] + x[1] - 1])
        assert abs(result.x[0] - 0.5) < 0.2

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            minimize(quadratic, [1.0], method='nonexistent')

    def test_hyphenated_name(self):
        result = minimize(quadratic, [5.0, 5.0], method='steepest-descent')
        assert abs(result.x[0]) < 0.1

    def test_projected_gradient_requires_bounds(self):
        with pytest.raises(ValueError):
            minimize(quadratic, [1.0, 1.0], method='projected_gradient')


# ===========================================================================
# OptimizeResult tests
# ===========================================================================

class TestOptimizeResult:
    def test_repr(self):
        r = OptimizeResult([1.0, 2.0], 0.5, nit=10, success=True, message='ok')
        s = repr(r)
        assert 'fun=0.5' in s
        assert 'nit=10' in s
        assert 'success=True' in s

    def test_fields(self):
        r = OptimizeResult([1, 2], 3.0, [0.1, 0.2], nit=5, nfev=20, ngev=10,
                            success=True, message='done', history=[3, 2, 1])
        assert r.x == [1, 2]
        assert r.fun == 3.0
        assert r.grad == [0.1, 0.2]
        assert r.nit == 5
        assert r.nfev == 20
        assert r.ngev == 10
        assert r.history == [3, 2, 1]


# ===========================================================================
# Cross-method comparison tests
# ===========================================================================

class TestCrossMethodComparison:
    """Ensure multiple methods agree on known solutions."""

    def test_all_find_rosenbrock_min(self):
        methods_with_kw = [
            ('bfgs', {}),
            ('lbfgs', {}),
            ('newton', {'max_iter': 200}),
            ('trust_region', {'max_iter': 500}),
        ]
        for method, kw in methods_with_kw:
            result = minimize(rosenbrock, [-1.0, 1.0], method=method, **kw)
            assert abs(result.x[0] - 1.0) < 0.1, f"{method} failed"
            assert abs(result.x[1] - 1.0) < 0.1, f"{method} failed"

    def test_constrained_methods_agree(self):
        # min x^2 + y^2 s.t. x + y = 1
        eq = [lambda x: x[0] + x[1] - 1]
        methods = ['augmented_lagrangian', 'penalty']
        for method in methods:
            result = minimize(quadratic, [2.0, 2.0], method=method,
                              eq_constraints=eq)
            assert abs(result.x[0] - 0.5) < 0.1, f"{method} failed"

    def test_least_squares_methods_agree(self):
        xdata = [1, 2, 3, 4, 5]
        ydata = [2.1, 3.9, 6.1, 8.0, 9.9]
        def residuals(p):
            return [p[0]*xdata[i] + p[1] - ydata[i] for i in range(len(xdata))]
        r1 = gauss_newton(residuals, [0.0, 0.0])
        r2 = levenberg_marquardt(residuals, [0.0, 0.0])
        assert abs(r1.x[0] - r2.x[0]) < 0.1
        assert abs(r1.x[1] - r2.x[1]) < 0.5


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    def test_already_at_minimum(self):
        result = bfgs(quadratic, [0.0, 0.0])
        assert result.success
        assert result.nit == 0

    def test_1d_optimization(self):
        result = bfgs(lambda x: (x[0] - 3)**2, [0.0])
        assert result.success
        assert abs(result.x[0] - 3.0) < 1e-5

    def test_high_dim_5d(self):
        x0 = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = lbfgs(sphere, x0)
        assert result.success
        for xi in result.x:
            assert abs(xi) < 1e-4

    def test_ill_conditioned(self):
        # f = x^2 + 1000*y^2 (very different curvatures)
        def f(x):
            return x[0]**2 + 1000 * x[1]**2
        result = bfgs(f, [1.0, 1.0])
        assert result.success
        assert abs(result.x[0]) < 1e-4
        assert abs(result.x[1]) < 1e-5

    def test_max_iter_reached(self):
        result = steepest_descent(rosenbrock, [0.0, 0.0], max_iter=5)
        assert not result.success
        assert 'Maximum iterations' in result.message

    def test_flat_region(self):
        # Very flat near minimum
        result = bfgs(lambda x: x[0]**4 + x[1]**4, [1.0, 1.0], max_iter=500)
        assert abs(result.x[0]) < 0.1
        assert abs(result.x[1]) < 0.1


# ===========================================================================
# Composition tests (using AD and LinAlg)
# ===========================================================================

class TestComposition:
    def test_ad_gradients_vs_numerical(self):
        """Verify AD and numerical gradients agree."""
        x = [1.5, 2.5]
        vg_ad = _make_grad_fn_test(rosenbrock, 2, True)
        vg_num = _make_grad_fn_test(rosenbrock, 2, False)
        _, g_ad = vg_ad(x)
        _, g_num = vg_num(x)
        for ga, gn in zip(g_ad, g_num):
            assert abs(ga - gn) < 1e-3

    def test_newton_uses_hessian(self):
        """Newton converges in 1 step for quadratic."""
        result = newton(quadratic, [5.0, 5.0])
        assert result.success
        assert result.nit <= 2  # Should be ~1 for pure quadratic


def _make_grad_fn_test(f, n, use_ad):
    """Helper to test gradient computation."""
    from optimization import _make_grad_fn
    return _make_grad_fn(f, n, use_ad)


# ===========================================================================
# Stress tests
# ===========================================================================

class TestStress:
    def test_10d_sphere(self):
        x0 = list(range(1, 11))
        result = lbfgs(lambda x: sum(xi**2 for xi in x), x0)
        assert result.success
        assert all(abs(xi) < 1e-3 for xi in result.x)

    def test_beale_bfgs(self):
        result = bfgs(beale, [0.0, 0.0], max_iter=500)
        assert abs(result.x[0] - 3.0) < 0.5
        assert abs(result.x[1] - 0.5) < 0.5

    def test_many_residuals(self):
        # 50 data points
        import random
        random.seed(42)
        xdata = [i * 0.1 for i in range(50)]
        ydata = [2.0 * x + 1.0 + random.gauss(0, 0.01) for x in xdata]

        def residuals(p):
            return [p[0] * xdata[i] + p[1] - ydata[i] for i in range(len(xdata))]

        result = levenberg_marquardt(residuals, [0.0, 0.0])
        assert result.success
        assert abs(result.x[0] - 2.0) < 0.05
        assert abs(result.x[1] - 1.0) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
