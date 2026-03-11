"""Tests for C127: Convex Optimization"""

import pytest
import math
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from convex_optimization import (
    VectorOps, V, OptStatus, OptResult,
    numerical_gradient, numerical_hessian, backtracking_line_search,
    GradientDescent, NewtonMethod, BFGS, ConjugateGradient,
    BarrierMethod, AugmentedLagrangian, ProximalGradient, ADMM,
    QuadraticProgram, Lasso,
    rosenbrock, rosenbrock_grad, rosenbrock_hess, quadratic,
)


# ===== VectorOps =====

class TestVectorOps:
    def test_dot(self):
        assert V.dot([1, 2, 3], [4, 5, 6]) == 32

    def test_add(self):
        assert V.add([1, 2], [3, 4]) == [4, 6]

    def test_sub(self):
        assert V.sub([5, 3], [1, 2]) == [4, 1]

    def test_scale(self):
        assert V.scale(2, [1, 2, 3]) == [2, 4, 6]

    def test_norm(self):
        assert abs(V.norm([3, 4]) - 5.0) < 1e-10

    def test_zeros(self):
        assert V.zeros(3) == [0.0, 0.0, 0.0]

    def test_ones(self):
        assert V.ones(3) == [1.0, 1.0, 1.0]

    def test_mat_vec(self):
        A = [[1, 2], [3, 4]]
        x = [1, 1]
        assert V.mat_vec(A, x) == [3, 7]

    def test_mat_mat(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = V.mat_mat(A, B)
        assert C == [[19, 22], [43, 50]]

    def test_transpose(self):
        A = [[1, 2, 3], [4, 5, 6]]
        At = V.transpose(A)
        assert At == [[1, 4], [2, 5], [3, 6]]

    def test_transpose_empty(self):
        assert V.transpose([]) == []

    def test_identity(self):
        I = V.identity(3)
        assert I[0][0] == 1 and I[0][1] == 0 and I[1][1] == 1

    def test_outer(self):
        o = V.outer([1, 2], [3, 4])
        assert o == [[3, 4], [6, 8]]

    def test_mat_add(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        assert V.mat_add(A, B) == [[6, 8], [10, 12]]

    def test_mat_scale(self):
        A = [[1, 2], [3, 4]]
        assert V.mat_scale(2, A) == [[2, 4], [6, 8]]

    def test_solve_2x2(self):
        A = [[2, 1], [1, 3]]
        b = [5, 10]
        x = V.solve_2x2(A, b)
        assert abs(x[0] - 1.0) < 1e-10
        assert abs(x[1] - 3.0) < 1e-10

    def test_solve_2x2_singular(self):
        A = [[1, 2], [2, 4]]
        assert V.solve_2x2(A, [1, 2]) is None

    def test_solve_linear(self):
        A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        b = [8, -11, -3]
        x = V.solve_linear(A, b)
        assert x is not None
        assert abs(x[0] - 2.0) < 1e-8
        assert abs(x[1] - 3.0) < 1e-8
        assert abs(x[2] - (-1.0)) < 1e-8

    def test_solve_linear_singular(self):
        A = [[1, 2], [2, 4]]
        assert V.solve_linear(A, [1, 2]) is None

    def test_cholesky(self):
        A = [[4, 2], [2, 3]]
        L = V.cholesky(A)
        assert L is not None
        # Verify L @ L^T = A
        Lt = V.transpose(L)
        result = V.mat_mat(L, Lt)
        for i in range(2):
            for j in range(2):
                assert abs(result[i][j] - A[i][j]) < 1e-10

    def test_cholesky_not_pd(self):
        A = [[1, 2], [2, 1]]  # not positive definite
        assert V.cholesky(A) is None

    def test_solve_cholesky(self):
        A = [[4, 2], [2, 3]]
        b = [6, 5]
        L = V.cholesky(A)
        x = V.solve_cholesky(L, b)
        # Check A @ x = b
        Ax = V.mat_vec(A, x)
        assert abs(Ax[0] - b[0]) < 1e-10
        assert abs(Ax[1] - b[1]) < 1e-10


# ===== Numerical differentiation =====

class TestNumericalDiff:
    def test_gradient_quadratic(self):
        f = lambda x: x[0]**2 + x[1]**2
        g = numerical_gradient(f, [1.0, 2.0])
        assert abs(g[0] - 2.0) < 1e-5
        assert abs(g[1] - 4.0) < 1e-5

    def test_gradient_rosenbrock(self):
        g = numerical_gradient(rosenbrock, [1.0, 1.0])
        # At minimum, gradient should be ~0
        assert abs(g[0]) < 1e-4
        assert abs(g[1]) < 1e-4

    def test_hessian_quadratic(self):
        f = lambda x: x[0]**2 + 3*x[1]**2
        H = numerical_hessian(f, [0.0, 0.0])
        assert abs(H[0][0] - 2.0) < 1e-3
        assert abs(H[1][1] - 6.0) < 1e-3
        assert abs(H[0][1]) < 1e-3

    def test_hessian_cross_term(self):
        f = lambda x: x[0]*x[1]
        H = numerical_hessian(f, [1.0, 1.0])
        assert abs(H[0][1] - 1.0) < 1e-3
        assert abs(H[1][0] - 1.0) < 1e-3


# ===== Line Search =====

class TestLineSearch:
    def test_backtracking_quadratic(self):
        f = lambda x: x[0]**2 + x[1]**2
        x = [2.0, 2.0]
        g = [4.0, 4.0]
        d = [-4.0, -4.0]  # steepest descent
        t = backtracking_line_search(f, x, d, g)
        assert t > 0

    def test_backtracking_not_descent(self):
        f = lambda x: x[0]**2
        x = [1.0]
        g = [2.0]
        d = [1.0]  # ascent direction
        t = backtracking_line_search(f, x, d, g)
        assert t == 0.0


# ===== Gradient Descent =====

class TestGradientDescent:
    def test_simple_quadratic(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        gd = GradientDescent(f, grad, tol=1e-8)
        r = gd.solve([5.0, 5.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.objective) < 1e-6
        assert abs(r.x[0]) < 1e-3
        assert abs(r.x[1]) < 1e-3

    def test_numerical_gradient(self):
        f = lambda x: (x[0] - 3)**2 + (x[1] + 1)**2
        gd = GradientDescent(f, tol=1e-6)
        r = gd.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 3.0) < 0.01
        assert abs(r.x[1] + 1.0) < 0.01

    def test_ill_conditioned(self):
        # Highly elongated quadratic
        f = lambda x: x[0]**2 + 100*x[1]**2
        grad = lambda x: [2*x[0], 200*x[1]]
        gd = GradientDescent(f, grad, tol=1e-6, max_iter=5000)
        r = gd.solve([10.0, 10.0])
        assert abs(r.x[0]) < 0.1
        assert abs(r.x[1]) < 0.01

    def test_history_decreasing(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        gd = GradientDescent(f, grad, tol=1e-8)
        r = gd.solve([5.0, 5.0])
        # Objective should be (mostly) decreasing
        for i in range(1, min(10, len(r.history))):
            assert r.history[i] <= r.history[i-1] + 1e-10

    def test_already_optimal(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        gd = GradientDescent(f, grad, tol=1e-6)
        r = gd.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        assert r.iterations == 0


# ===== Newton's Method =====

class TestNewtonMethod:
    def test_quadratic_exact(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        hess = lambda x: [[2, 0], [0, 2]]
        nm = NewtonMethod(f, grad, hess, tol=1e-10)
        r = nm.solve([5.0, 3.0])
        assert r.status == OptStatus.OPTIMAL
        assert r.iterations <= 2  # Newton on quadratic converges in 1 step
        assert abs(r.objective) < 1e-15

    def test_rosenbrock(self):
        nm = NewtonMethod(rosenbrock, rosenbrock_grad, rosenbrock_hess, tol=1e-8, max_iter=200)
        r = nm.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 1e-4
        assert abs(r.x[1] - 1.0) < 1e-4

    def test_numerical_hessian(self):
        f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
        grad = lambda x: [2*(x[0] - 2), 2*(x[1] - 3)]
        nm = NewtonMethod(f, grad, tol=1e-8)
        r = nm.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 2.0) < 1e-4
        assert abs(r.x[1] - 3.0) < 1e-4

    def test_higher_dim(self):
        # 4D quadratic
        f = lambda x: sum(xi**2 for xi in x)
        grad = lambda x: [2*xi for xi in x]
        hess = lambda x: [[2 if i == j else 0 for j in range(4)] for i in range(4)]
        nm = NewtonMethod(f, grad, hess, tol=1e-10)
        r = nm.solve([1.0, 2.0, 3.0, 4.0])
        assert r.status == OptStatus.OPTIMAL
        assert all(abs(xi) < 1e-8 for xi in r.x)

    def test_regularization_fallback(self):
        # Indefinite Hessian at start
        f = lambda x: x[0]**4 + x[1]**4  # Hessian=0 at origin
        nm = NewtonMethod(f, tol=1e-6, max_iter=200)
        r = nm.solve([2.0, 2.0])
        assert abs(r.x[0]) < 0.1
        assert abs(r.x[1]) < 0.1


# ===== BFGS =====

class TestBFGS:
    def test_quadratic(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        bfgs = BFGS(f, grad, tol=1e-8)
        r = bfgs.solve([5.0, 3.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.objective) < 1e-6

    def test_rosenbrock(self):
        bfgs = BFGS(rosenbrock, rosenbrock_grad, tol=1e-6, max_iter=1000)
        r = bfgs.solve([-1.0, 1.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 0.01
        assert abs(r.x[1] - 1.0) < 0.01

    def test_numerical_gradient(self):
        f = lambda x: (x[0] - 1)**2 + 2*(x[1] - 2)**2
        bfgs = BFGS(f, tol=1e-6)
        r = bfgs.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 0.01
        assert abs(r.x[1] - 2.0) < 0.01

    def test_higher_dim(self):
        # 5D quadratic
        n = 5
        Q = [[2.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        c = [float(-2*i) for i in range(n)]
        f, grad = quadratic(Q, c)
        bfgs = BFGS(f, grad, tol=1e-8)
        r = bfgs.solve([0.0] * n)
        assert r.status == OptStatus.OPTIMAL
        # x* = -Q^{-1} c = [0, 1, 2, 3, 4]
        for i in range(n):
            assert abs(r.x[i] - i) < 0.01

    def test_convergence_faster_than_gd(self):
        f = lambda x: x[0]**2 + 100*x[1]**2
        grad = lambda x: [2*x[0], 200*x[1]]
        bfgs = BFGS(f, grad, tol=1e-8)
        r_bfgs = bfgs.solve([10.0, 10.0])
        gd = GradientDescent(f, grad, tol=1e-8, max_iter=5000)
        r_gd = gd.solve([10.0, 10.0])
        # BFGS should converge in fewer iterations
        assert r_bfgs.iterations < r_gd.iterations


# ===== Conjugate Gradient =====

class TestConjugateGradient:
    def test_quadratic(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        cg = ConjugateGradient(f, grad, tol=1e-8)
        r = cg.solve([5.0, 3.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.objective) < 1e-6

    def test_rosenbrock(self):
        cg = ConjugateGradient(rosenbrock, rosenbrock_grad, tol=1e-5, max_iter=2000)
        r = cg.solve([0.0, 0.0])
        assert abs(r.x[0] - 1.0) < 0.05
        assert abs(r.x[1] - 1.0) < 0.05

    def test_faster_than_gd(self):
        # CG should outperform steepest descent on ill-conditioned problems
        f = lambda x: x[0]**2 + 50*x[1]**2
        grad = lambda x: [2*x[0], 100*x[1]]
        cg = ConjugateGradient(f, grad, tol=1e-6)
        r = cg.solve([10.0, 10.0])
        assert r.status == OptStatus.OPTIMAL
        assert r.iterations < 200


# ===== Barrier Method =====

class TestBarrierMethod:
    def test_simple_constrained(self):
        # minimize x^2 subject to x >= 1 (i.e. -x + 1 <= 0)
        f = lambda x: x[0]**2
        constraints = [lambda x: -x[0] + 1]  # -x + 1 <= 0 means x >= 1
        bm = BarrierMethod(f, constraints=constraints, tol=1e-4)
        r = bm.solve([2.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 0.05

    def test_2d_constrained(self):
        # minimize x^2 + y^2 subject to x + y >= 2
        f = lambda x: x[0]**2 + x[1]**2
        constraints = [lambda x: -x[0] - x[1] + 2]  # -(x+y) + 2 <= 0
        bm = BarrierMethod(f, constraints=constraints, tol=1e-4)
        r = bm.solve([2.0, 2.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 0.1
        assert abs(r.x[1] - 1.0) < 0.1

    def test_box_constraints(self):
        # minimize (x-3)^2 + (y-3)^2 subject to 0 <= x <= 2, 0 <= y <= 2
        f = lambda x: (x[0] - 3)**2 + (x[1] - 3)**2
        constraints = [
            lambda x: -x[0],       # x >= 0
            lambda x: x[0] - 2,    # x <= 2
            lambda x: -x[1],       # y >= 0
            lambda x: x[1] - 2,    # y <= 2
        ]
        bm = BarrierMethod(f, constraints=constraints, tol=1e-4)
        r = bm.solve([1.0, 1.0])
        assert r.status == OptStatus.OPTIMAL
        # Optimum is at (2, 2)
        assert abs(r.x[0] - 2.0) < 0.1
        assert abs(r.x[1] - 2.0) < 0.1

    def test_infeasible_start(self):
        f = lambda x: x[0]**2
        constraints = [lambda x: x[0] - 1]  # x <= 1
        bm = BarrierMethod(f, constraints=constraints, tol=1e-4)
        r = bm.solve([2.0])  # starts infeasible
        assert r.status == OptStatus.INFEASIBLE

    def test_multiple_constraints(self):
        # minimize -x - y subject to x + y <= 4, x <= 3, y <= 3, x >= 0, y >= 0
        f = lambda x: -x[0] - x[1]
        constraints = [
            lambda x: x[0] + x[1] - 4,
            lambda x: x[0] - 3,
            lambda x: x[1] - 3,
            lambda x: -x[0],
            lambda x: -x[1],
        ]
        bm = BarrierMethod(f, constraints=constraints, tol=1e-4, max_iter=100)
        r = bm.solve([1.0, 1.0])
        assert r.status == OptStatus.OPTIMAL
        # Optimum near (3, 1) or (1, 3) -- obj = -4
        assert abs(r.objective + 4.0) < 0.2


# ===== Augmented Lagrangian =====

class TestAugmentedLagrangian:
    def test_equality_constraint(self):
        # minimize x^2 + y^2 subject to x + y = 2
        f = lambda x: x[0]**2 + x[1]**2
        eq = [lambda x: x[0] + x[1] - 2]
        al = AugmentedLagrangian(f, eq_constraints=eq, tol=1e-5)
        r = al.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 0.05
        assert abs(r.x[1] - 1.0) < 0.05

    def test_two_equalities(self):
        # minimize x^2 + y^2 + z^2 subject to x+y+z=3, x-y=0
        f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
        eq = [
            lambda x: x[0] + x[1] + x[2] - 3,
            lambda x: x[0] - x[1],
        ]
        al = AugmentedLagrangian(f, eq_constraints=eq, tol=1e-4)
        r = al.solve([0.0, 0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        # x=y, x+y+z=3, minimize -> x=y=z=1
        assert abs(r.x[0] - 1.0) < 0.1
        assert abs(r.x[1] - 1.0) < 0.1
        assert abs(r.x[2] - 1.0) < 0.1

    def test_nonlinear_equality(self):
        # minimize x^2 + y^2 subject to x^2 + y^2 = 1 (on unit circle)
        # Trivially, any point on the circle has obj = 1
        f = lambda x: x[0]**2 + x[1]**2
        eq = [lambda x: x[0]**2 + x[1]**2 - 1]
        al = AugmentedLagrangian(f, eq_constraints=eq, tol=1e-4, max_iter=100)
        r = al.solve([0.5, 0.5])
        assert abs(r.objective - 1.0) < 0.1


# ===== Proximal Gradient =====

class TestProximalGradient:
    def test_l1_regularized(self):
        # minimize (1/2)(x-3)^2 + |x|
        # Solution: soft_threshold(3, 1) = 2
        f = lambda x: 0.5 * (x[0] - 3)**2
        grad_f = lambda x: [x[0] - 3]
        g = lambda x: abs(x[0])
        prox_g = lambda x, t: [max(0, x[0] - t) - max(0, -x[0] - t)]

        pg = ProximalGradient(f, grad_f, prox_g, g, tol=1e-8, lr=1.0)
        r = pg.solve([0.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 2.0) < 0.01

    def test_accelerated(self):
        # Same problem, accelerated should converge
        f = lambda x: 0.5 * (x[0] - 3)**2
        grad_f = lambda x: [x[0] - 3]
        g = lambda x: abs(x[0])
        prox_g = lambda x, t: [max(0, x[0] - t) - max(0, -x[0] - t)]

        pg = ProximalGradient(f, grad_f, prox_g, g, tol=1e-8, lr=1.0, accelerated=True)
        r = pg.solve([0.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 2.0) < 0.01

    def test_indicator_constraint(self):
        # minimize x^2 subject to x >= 1 via indicator
        # prox of indicator_{x>=1} is projection: max(x, 1)
        f = lambda x: x[0]**2
        grad_f = lambda x: [2*x[0]]
        prox_g = lambda x, t: [max(x[0], 1.0)]

        pg = ProximalGradient(f, grad_f, prox_g, tol=1e-8, lr=0.4)
        r = pg.solve([5.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 0.01

    def test_2d_l1(self):
        # minimize (1/2)||x - [3,4]||^2 + 2*||x||_1
        f = lambda x: 0.5 * ((x[0]-3)**2 + (x[1]-4)**2)
        grad_f = lambda x: [x[0]-3, x[1]-4]
        lam = 2.0
        g = lambda x: lam * (abs(x[0]) + abs(x[1]))
        prox_g = lambda x, t: [
            max(0, x[i] - lam*t) - max(0, -x[i] - lam*t)
            for i in range(2)
        ]
        pg = ProximalGradient(f, grad_f, prox_g, g, tol=1e-8, lr=1.0)
        r = pg.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        # soft_threshold(3, 2) = 1, soft_threshold(4, 2) = 2
        assert abs(r.x[0] - 1.0) < 0.01
        assert abs(r.x[1] - 2.0) < 0.01


# ===== ADMM =====

class TestADMM:
    def test_consensus(self):
        # minimize (1/2)||x - b||^2 + lam*||z||_1, subject to x = z
        # (Lasso in ADMM form with A=I, b=target)
        b = [3.0, 4.0]
        lam = 1.0

        def x_update(z, u, rho):
            # argmin (1/2)||x-b||^2 + (rho/2)||x - z + u||^2
            # x = (b + rho*(z - u)) / (1 + rho)
            return [(bi + rho*(zi - ui)) / (1 + rho) for bi, zi, ui in zip(b, z, u)]

        def z_update(x, u, rho):
            # argmin lam*||z||_1 + (rho/2)||x - z + u||^2
            # soft threshold
            v = V.add(x, u)
            threshold = lam / rho
            return [max(0, vi - threshold) - max(0, -vi - threshold) for vi in v]

        admm = ADMM(x_update, z_update, rho=1.0, tol=1e-6, max_iter=500)
        r = admm.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        # soft_threshold(3, 1) = 2, soft_threshold(4, 1) = 3
        assert abs(r.x[0] - 2.0) < 0.1
        assert abs(r.x[1] - 3.0) < 0.1

    def test_simple_equality(self):
        # minimize x^2, z^2 subject to x = z
        # x* = z* = 0
        def x_update(z, u, rho):
            return [rho*(z[0] - u[0]) / (2 + rho)]

        def z_update(x, u, rho):
            return [rho*(x[0] + u[0]) / (2 + rho)]

        admm = ADMM(x_update, z_update, rho=1.0, tol=1e-6)
        r = admm.solve([5.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0]) < 0.01


# ===== Quadratic Program =====

class TestQuadraticProgram:
    def test_unconstrained(self):
        # minimize x^2 + y^2 - 2x - 4y
        Q = [[2, 0], [0, 2]]
        c = [-2, -4]
        qp = QuadraticProgram(Q, c)
        r = qp.solve()
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 1.0) < 1e-6
        assert abs(r.x[1] - 2.0) < 1e-6

    def test_constrained(self):
        # minimize x^2 + y^2 subject to x + y <= 1, x >= 0, y >= 0
        Q = [[2, 0], [0, 2]]
        c = [0, 0]
        A = [[1, 1], [-1, 0], [0, -1]]
        b = [1, 0, 0]
        qp = QuadraticProgram(Q, c, A, b)
        r = qp.solve([0.3, 0.3])
        assert r.status == OptStatus.OPTIMAL
        # Unconstrained min is (0,0), feasible, so x*=(0,0)
        assert abs(r.x[0]) < 0.1
        assert abs(r.x[1]) < 0.1

    def test_constrained_binding(self):
        # minimize (x-2)^2 + (y-2)^2 subject to x+y <= 1, x >= 0, y >= 0
        # Q = 2I, c = [-4, -4] + constant
        Q = [[2, 0], [0, 2]]
        c = [-4, -4]
        A = [[1, 1], [-1, 0], [0, -1]]
        b = [1, 0, 0]
        qp = QuadraticProgram(Q, c, A, b)
        r = qp.solve([0.3, 0.3])
        assert r.status == OptStatus.OPTIMAL
        # Constrained min: x + y = 1, x = y = 0.5
        assert abs(r.x[0] - 0.5) < 0.15
        assert abs(r.x[1] - 0.5) < 0.15

    def test_1d_unconstrained(self):
        Q = [[4]]
        c = [-8]
        qp = QuadraticProgram(Q, c)
        r = qp.solve()
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0] - 2.0) < 1e-6


# ===== Lasso =====

class TestLasso:
    def test_sparse_recovery(self):
        # A = identity, b = [3, 0.1, 4], lambda = 0.5
        # Solution should zero out small entries
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [3, 0.1, 4]
        lasso = Lasso(A, b, lam=0.5, tol=1e-6, max_iter=2000)
        r = lasso.solve()
        assert r.status == OptStatus.OPTIMAL
        # x[0] = soft(3, 0.5) = 2.5
        assert abs(r.x[0] - 2.5) < 0.1
        # x[1] = soft(0.1, 0.5) = 0 (shrunk to zero)
        assert abs(r.x[1]) < 0.1
        # x[2] = soft(4, 0.5) = 3.5
        assert abs(r.x[2] - 3.5) < 0.1

    def test_overdetermined(self):
        # 3 equations, 2 unknowns with L1
        A = [[1, 0], [0, 1], [1, 1]]
        b = [1, 1, 2]
        lasso = Lasso(A, b, lam=0.01, tol=1e-6, max_iter=2000)
        r = lasso.solve()
        assert r.status == OptStatus.OPTIMAL
        # Should be close to [1, 1]
        assert abs(r.x[0] - 1.0) < 0.2
        assert abs(r.x[1] - 1.0) < 0.2

    def test_high_regularization(self):
        # High lambda should push everything to zero
        A = [[1, 0], [0, 1]]
        b = [1, 1]
        lasso = Lasso(A, b, lam=100.0, tol=1e-6, max_iter=1000)
        r = lasso.solve()
        assert all(abs(xi) < 0.1 for xi in r.x)


# ===== Rosenbrock utility =====

class TestRosenbrock:
    def test_value_at_min(self):
        assert rosenbrock([1, 1]) == 0

    def test_grad_at_min(self):
        g = rosenbrock_grad([1, 1])
        assert abs(g[0]) < 1e-10
        assert abs(g[1]) < 1e-10

    def test_hess_at_min(self):
        H = rosenbrock_hess([1, 1])
        assert abs(H[0][0] - 802) < 1e-6
        assert abs(H[0][1] - (-400)) < 1e-6
        assert abs(H[1][1] - 200) < 1e-6

    def test_value_nonmin(self):
        assert rosenbrock([0, 0]) == 1.0


# ===== Quadratic utility =====

class TestQuadraticUtil:
    def test_quadratic_function(self):
        Q = [[2, 0], [0, 4]]
        c = [-2, -4]
        f, grad = quadratic(Q, c)
        assert f([0, 0]) == 0
        assert f([1, 1]) == 0.5 * (2 + 4) + (-2 - 4)  # 3 - 6 = -3
        g = grad([1, 1])
        assert g[0] == 0.0  # 2*1 + (-2) = 0
        assert g[1] == 0.0  # 4*1 + (-4) = 0


# ===== OptResult =====

class TestOptResult:
    def test_optimal_repr(self):
        r = OptResult(OptStatus.OPTIMAL, [1, 2], 3.0, 10, 0.001)
        s = repr(r)
        assert "OPTIMAL" in s
        assert "3.000000" in s

    def test_non_optimal_repr(self):
        r = OptResult(OptStatus.MAX_ITER, iterations=100)
        assert "max_iterations" in repr(r)


# ===== Integration tests =====

class TestIntegration:
    def test_all_solvers_on_quadratic(self):
        """All solvers should find the minimum of a simple quadratic."""
        f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
        grad = lambda x: [2*(x[0]-1), 2*(x[1]-2)]
        x0 = [10.0, 10.0]

        for SolverClass in [GradientDescent, NewtonMethod, BFGS, ConjugateGradient]:
            solver = SolverClass(f, grad, tol=1e-6)
            r = solver.solve(x0)
            assert abs(r.x[0] - 1.0) < 0.01, f"{SolverClass.__name__} failed on x[0]"
            assert abs(r.x[1] - 2.0) < 0.01, f"{SolverClass.__name__} failed on x[1]"

    def test_newton_vs_bfgs_rosenbrock(self):
        """Both Newton and BFGS should solve Rosenbrock."""
        for Solver, kwargs in [
            (NewtonMethod, dict(f=rosenbrock, grad_f=rosenbrock_grad, hess_f=rosenbrock_hess, tol=1e-6, max_iter=200)),
            (BFGS, dict(f=rosenbrock, grad_f=rosenbrock_grad, tol=1e-6, max_iter=1000)),
        ]:
            solver = Solver(**kwargs)
            r = solver.solve([0.0, 0.0])
            assert abs(r.x[0] - 1.0) < 0.05, f"{Solver.__name__} failed"
            assert abs(r.x[1] - 1.0) < 0.05, f"{Solver.__name__} failed"

    def test_barrier_then_check_kkt(self):
        """Barrier solution should approximately satisfy KKT conditions."""
        # minimize x^2 + y^2 subject to x + y >= 2
        f = lambda x: x[0]**2 + x[1]**2
        constraints = [lambda x: -x[0] - x[1] + 2]
        bm = BarrierMethod(f, constraints=constraints, tol=1e-5)
        r = bm.solve([2.0, 2.0])
        # At optimum: grad f = lambda * grad g
        # grad f = [2x, 2y], grad g = [-1, -1]
        # KKT: [2x, 2y] + lambda*[-1, -1] = 0 => x = y = lambda/2
        # constraint: x+y = 2 => lambda = 2, x = y = 1
        assert abs(r.x[0] - 1.0) < 0.1
        assert abs(r.x[1] - 1.0) < 0.1

    def test_qp_matches_direct(self):
        """QP solver should match analytical solution."""
        # minimize (1/2)x^T [[4,1],[1,2]] x + [-1,-1]^T x
        Q = [[4, 1], [1, 2]]
        c = [-1, -1]
        qp = QuadraticProgram(Q, c)
        r = qp.solve()
        # Q x = -c => [[4,1],[1,2]] x = [1,1]
        # x = Q^{-1} [1,1] = (1/7) [[2,-1],[-1,4]] [1,1] = (1/7) [1,3]
        assert abs(r.x[0] - 1/7) < 1e-5
        assert abs(r.x[1] - 3/7) < 1e-5

    def test_solver_comparison_iterations(self):
        """Newton should use fewer iterations than GD on well-conditioned problems."""
        f = lambda x: (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2
        grad = lambda x: [2*(x[0]-1), 2*(x[1]-2), 2*(x[2]-3)]
        hess = lambda x: [[2,0,0],[0,2,0],[0,0,2]]
        x0 = [10.0, 10.0, 10.0]

        nm = NewtonMethod(f, grad, hess, tol=1e-10)
        r_nm = nm.solve(x0)

        gd = GradientDescent(f, grad, tol=1e-10, max_iter=5000)
        r_gd = gd.solve(x0)

        assert r_nm.iterations <= 2
        # GD with backtracking on a simple quadratic also converges in 1 step
        # (exact line search on quadratic = 1 step), so just check Newton is fast
        assert r_nm.iterations <= r_gd.iterations

    def test_augmented_lagrangian_circle(self):
        """Find point on unit circle closest to (2, 0)."""
        f = lambda x: (x[0] - 2)**2 + x[1]**2
        eq = [lambda x: x[0]**2 + x[1]**2 - 1]
        al = AugmentedLagrangian(f, eq_constraints=eq, tol=1e-4, max_iter=100)
        r = al.solve([0.5, 0.5])
        assert r.status == OptStatus.OPTIMAL
        # Closest point on unit circle to (2,0) is (1,0)
        assert abs(r.x[0] - 1.0) < 0.1
        assert abs(r.x[1]) < 0.1


# ===== Edge cases =====

class TestEdgeCases:
    def test_1d_optimization(self):
        f = lambda x: (x[0] - 5)**2
        grad = lambda x: [2*(x[0] - 5)]
        gd = GradientDescent(f, grad, tol=1e-8)
        r = gd.solve([0.0])
        assert abs(r.x[0] - 5.0) < 1e-3

    def test_zero_start(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        gd = GradientDescent(f, grad, tol=1e-6)
        r = gd.solve([0.0, 0.0])
        assert r.status == OptStatus.OPTIMAL
        assert r.iterations == 0

    def test_large_starting_point(self):
        f = lambda x: x[0]**2 + x[1]**2
        grad = lambda x: [2*x[0], 2*x[1]]
        bfgs = BFGS(f, grad, tol=1e-6)
        r = bfgs.solve([1000.0, 1000.0])
        assert r.status == OptStatus.OPTIMAL
        assert abs(r.x[0]) < 0.01
        assert abs(r.x[1]) < 0.01

    def test_max_iter_reached(self):
        # Use a function where gradient never reaches 1e-20 in 3 iters
        # Beale's function is hard for GD
        def beale(x):
            return ((1.5 - x[0] + x[0]*x[1])**2 +
                    (2.25 - x[0] + x[0]*x[1]**2)**2 +
                    (2.625 - x[0] + x[0]*x[1]**3)**2)
        gd = GradientDescent(beale, tol=1e-20, max_iter=3)
        r = gd.solve([0.5, 0.5])
        assert r.status == OptStatus.MAX_ITER
        assert r.iterations == 3
