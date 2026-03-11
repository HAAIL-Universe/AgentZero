"""Tests for C137: ODE Solvers."""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ode_solvers import (
    ODEResult, euler_step, midpoint_step, rk4_step, rk45_step,
    solve_fixed, solve_rk45, solve_implicit, solve_bdf,
    solve_adams_bashforth, solve_adams_moulton,
    backward_euler_step, trapezoidal_step, bdf_step,
    detect_stiffness, DenseOutput, to_first_order, make_system,
    solve_ivp, solve_scalar,
    _norm, _vec_add, _vec_sub, _vec_scale, _vec_axpy,
    _weighted_norm, _finite_diff_jacobian,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C132_linear_algebra'))
from linear_algebra import Matrix


# ============================================================
# Helper: standard test ODEs
# ============================================================

def exp_decay(t, y):
    """y' = -y, solution: y(t) = exp(-t)."""
    return [-y[0]]

def exp_growth(t, y):
    """y' = y, solution: y(t) = exp(t)."""
    return [y[0]]

def harmonic_osc(t, y):
    """y'' = -y -> [y, v], v' = -y, y' = v. Solution: y=cos(t), v=-sin(t)."""
    return [y[1], -y[0]]

def lotka_volterra(t, y, a=1.0, b=0.1, c=0.1, d=1.0):
    """Predator-prey: x' = ax - bxy, y' = cxy - dy."""
    return [a * y[0] - b * y[0] * y[1],
            c * y[0] * y[1] - d * y[1]]

def vanderpol(t, y, mu=1.0):
    """Van der Pol oscillator: y'' - mu(1-y^2)y' + y = 0."""
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

def stiff_system(t, y):
    """Stiff system: y' = -1000*y + 1000*sin(t)."""
    return [-1000 * y[0] + 1000 * math.sin(t)]

def linear_2d(t, y):
    """y1' = -y1 + y2, y2' = -y2. Solution: y2 = e^{-t}, y1 = te^{-t}."""
    return [-y[0] + y[1], -y[1]]


# ============================================================
# 1. Helpers
# ============================================================

class TestHelpers:
    def test_norm(self):
        assert _norm([3, 4]) == 5.0
        assert _norm([0, 0, 0]) == 0.0

    def test_vec_add(self):
        assert _vec_add([1, 2], [3, 4]) == [4, 6]

    def test_vec_sub(self):
        assert _vec_sub([5, 3], [2, 1]) == [3, 2]

    def test_vec_scale(self):
        assert _vec_scale(2, [3, 4]) == [6, 8]

    def test_vec_axpy(self):
        assert _vec_axpy(2, [1, 2], [3, 4]) == [5, 8]

    def test_weighted_norm(self):
        v = [1.0, 0.0]
        n = _weighted_norm(v, 1.0, 0.0, [0.0, 0.0])
        assert abs(n - math.sqrt(0.5)) < 1e-10

    def test_finite_diff_jacobian(self):
        # f(t, y) = [-y[0], 2*y[1]], J = [[-1, 0], [0, 2]]
        def f(t, y):
            return [-y[0], 2 * y[1]]
        J = _finite_diff_jacobian(f, 0, [1.0, 1.0], [-1.0, 2.0])
        assert abs(J[0, 0] - (-1)) < 1e-5
        assert abs(J[0, 1]) < 1e-5
        assert abs(J[1, 0]) < 1e-5
        assert abs(J[1, 1] - 2) < 1e-5


# ============================================================
# 2. ODEResult
# ============================================================

class TestODEResult:
    def test_creation(self):
        r = ODEResult([0, 1], [[1.0], [0.5]], success=True, message="ok", nfev=10)
        assert r.success
        assert r.nfev == 10
        assert len(r.t) == 2

    def test_repr(self):
        r = ODEResult([0], [[1]], nfev=5)
        assert "points=1" in repr(r)
        assert "nfev=5" in repr(r)

    def test_defaults(self):
        r = ODEResult([], [])
        assert r.t_events == []
        assert r.y_events == []
        assert r.njev == 0
        assert r.nlu == 0


# ============================================================
# 3. Euler step
# ============================================================

class TestEulerStep:
    def test_single_step(self):
        y, nf = euler_step(exp_decay, 0, [1.0], 0.1)
        # y = 1 + 0.1*(-1) = 0.9
        assert abs(y[0] - 0.9) < 1e-12
        assert nf == 1

    def test_system_step(self):
        y, nf = euler_step(harmonic_osc, 0, [1.0, 0.0], 0.01)
        # y = [1 + 0.01*0, 0 + 0.01*(-1)] = [1.0, -0.01]
        assert abs(y[0] - 1.0) < 1e-12
        assert abs(y[1] - (-0.01)) < 1e-12


# ============================================================
# 4. Midpoint step
# ============================================================

class TestMidpointStep:
    def test_single_step(self):
        y, nf = midpoint_step(exp_decay, 0, [1.0], 0.1)
        assert nf == 2
        # More accurate than Euler
        exact = math.exp(-0.1)
        assert abs(y[0] - exact) < 0.001

    def test_second_order(self):
        # Midpoint should be exact for linear ODEs
        def linear(t, y):
            return [2.0]  # y' = 2, y = 2t + C
        y, _ = midpoint_step(linear, 0, [0.0], 0.5)
        assert abs(y[0] - 1.0) < 1e-12


# ============================================================
# 5. RK4 step
# ============================================================

class TestRK4Step:
    def test_single_step(self):
        y, nf = rk4_step(exp_decay, 0, [1.0], 0.1)
        assert nf == 4
        exact = math.exp(-0.1)
        assert abs(y[0] - exact) < 1e-7

    def test_system_step(self):
        y, nf = rk4_step(harmonic_osc, 0, [1.0, 0.0], 0.01)
        # y ~ [cos(0.01), -sin(0.01)]
        assert abs(y[0] - math.cos(0.01)) < 1e-10
        assert abs(y[1] - (-math.sin(0.01))) < 1e-10


# ============================================================
# 6. RK45 step
# ============================================================

class TestRK45Step:
    def test_step_and_error(self):
        y5, err, k, nf = rk45_step(exp_decay, 0, [1.0], 0.1)
        assert nf == 6
        exact = math.exp(-0.1)
        assert abs(y5[0] - exact) < 1e-9
        # Error should be very small
        assert abs(err[0]) < 1e-8


# ============================================================
# 7. Solve fixed (Euler)
# ============================================================

class TestSolveFixedEuler:
    def test_exp_decay(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=1000, method='euler')
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01

    def test_n_points(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=10, method='euler')
        assert len(r.t) == 11  # initial + 10 steps


# ============================================================
# 8. Solve fixed (RK4)
# ============================================================

class TestSolveFixedRK4:
    def test_exp_decay(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=100, method='rk4')
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 1e-8

    def test_harmonic(self):
        r = solve_fixed(harmonic_osc, (0, 2 * math.pi), [1.0, 0.0],
                        n_steps=200, method='rk4')
        # After full period: y ~ [1, 0]
        assert abs(r.y[-1][0] - 1.0) < 1e-4
        assert abs(r.y[-1][1]) < 1e-4


# ============================================================
# 9. Solve fixed (Midpoint)
# ============================================================

class TestSolveFixedMidpoint:
    def test_exp_decay(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=200, method='midpoint')
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.001


# ============================================================
# 10. RK45 adaptive
# ============================================================

class TestSolveRK45:
    def test_exp_decay(self):
        r = solve_rk45(exp_decay, (0, 5), [1.0])
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-5)) < 1e-5

    def test_harmonic(self):
        r = solve_rk45(harmonic_osc, (0, 10), [1.0, 0.0])
        assert r.success
        assert abs(r.y[-1][0] - math.cos(10)) < 1e-3
        assert abs(r.y[-1][1] - (-math.sin(10))) < 1e-3

    def test_adaptive_steps(self):
        # Adaptive should take fewer steps than fixed for smooth problems
        r = solve_rk45(exp_decay, (0, 10), [1.0])
        assert r.success
        assert len(r.t) < 500

    def test_tight_tolerance(self):
        r = solve_rk45(exp_decay, (0, 1), [1.0], rtol=1e-10, atol=1e-12)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 1e-9

    def test_max_nfev(self):
        r = solve_rk45(exp_decay, (0, 100), [1.0], max_nfev=10)
        assert not r.success
        assert "Max function evaluations" in r.message


# ============================================================
# 11. Backward Euler step
# ============================================================

class TestBackwardEulerStep:
    def test_single_step(self):
        y, nfev, njev, nlu = backward_euler_step(exp_decay, 0, [1.0], 0.1)
        # Implicit Euler: y = y0 / (1 + h) = 1/1.1
        exact = 1.0 / 1.1
        assert abs(y[0] - exact) < 1e-8

    def test_with_jacobian(self):
        def jac(t, y):
            J = Matrix(1)
            J[0, 0] = -1.0
            return J
        y, nfev, njev, nlu = backward_euler_step(exp_decay, 0, [1.0], 0.1, jac=jac)
        assert abs(y[0] - 1.0 / 1.1) < 1e-8
        assert njev >= 1


# ============================================================
# 12. Trapezoidal step
# ============================================================

class TestTrapezoidalStep:
    def test_single_step(self):
        y, nfev, njev, nlu = trapezoidal_step(exp_decay, 0, [1.0], 0.1)
        # Trapz: y = y0 * (1 - h/2) / (1 + h/2) = 0.95/1.05
        exact = 0.95 / 1.05
        assert abs(y[0] - exact) < 1e-8

    def test_higher_order(self):
        # Trapezoidal should be more accurate than backward Euler
        y_be, _, _, _ = backward_euler_step(exp_decay, 0, [1.0], 0.1)
        y_tr, _, _, _ = trapezoidal_step(exp_decay, 0, [1.0], 0.1)
        exact = math.exp(-0.1)
        assert abs(y_tr[0] - exact) < abs(y_be[0] - exact)


# ============================================================
# 13. Implicit solver
# ============================================================

class TestSolveImplicit:
    def test_backward_euler(self):
        r = solve_implicit(exp_decay, (0, 1), [1.0], n_steps=100,
                           method='backward_euler')
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01

    def test_trapezoidal(self):
        r = solve_implicit(exp_decay, (0, 1), [1.0], n_steps=100,
                           method='trapz')
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 1e-4

    def test_stiff(self):
        # Implicit methods handle stiff systems
        r = solve_implicit(stiff_system, (0, 0.1), [0.0], n_steps=100,
                           method='backward_euler')
        assert r.success
        # Should not blow up


# ============================================================
# 14. BDF step
# ============================================================

class TestBDFStep:
    def test_order1(self):
        # BDF1 = backward Euler
        y, nfev, _, _ = bdf_step(exp_decay, 0.1, [[1.0]], 0.1, 1)
        assert abs(y[0] - 1.0 / 1.1) < 1e-8

    def test_order2(self):
        y, _, _, _ = bdf_step(exp_decay, 0.2,
                               [[math.exp(-0.1)], [1.0]], 0.1, 2)
        assert abs(y[0] - math.exp(-0.2)) < 0.01


# ============================================================
# 15. BDF solver
# ============================================================

class TestSolveBDF:
    def test_exp_decay(self):
        r = solve_bdf(exp_decay, (0, 1), [1.0], n_steps=200, order=2)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01

    def test_order3(self):
        r = solve_bdf(exp_decay, (0, 1), [1.0], n_steps=200, order=3)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01

    def test_stiff(self):
        r = solve_bdf(stiff_system, (0, 0.1), [0.0], n_steps=200, order=2)
        assert r.success

    def test_system(self):
        r = solve_bdf(linear_2d, (0, 1), [0.0, 1.0], n_steps=200, order=2)
        assert r.success
        # y2(1) = e^{-1}
        assert abs(r.y[-1][1] - math.exp(-1)) < 0.01


# ============================================================
# 16. Adams-Bashforth
# ============================================================

class TestAdamsBashforth:
    def test_exp_decay_order1(self):
        r = solve_adams_bashforth(exp_decay, (0, 1), [1.0], n_steps=200, order=1)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.02

    def test_exp_decay_order4(self):
        r = solve_adams_bashforth(exp_decay, (0, 1), [1.0], n_steps=200, order=4)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.001

    def test_harmonic(self):
        r = solve_adams_bashforth(harmonic_osc, (0, 2 * math.pi),
                                  [1.0, 0.0], n_steps=500, order=4)
        assert r.success
        assert abs(r.y[-1][0] - 1.0) < 0.01

    def test_order2(self):
        r = solve_adams_bashforth(exp_decay, (0, 1), [1.0], n_steps=200, order=2)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01


# ============================================================
# 17. Adams-Moulton
# ============================================================

class TestAdamsMoulton:
    def test_exp_decay(self):
        r = solve_adams_moulton(exp_decay, (0, 1), [1.0], n_steps=200, order=3)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.001

    def test_harmonic(self):
        r = solve_adams_moulton(harmonic_osc, (0, 2 * math.pi),
                                [1.0, 0.0], n_steps=500, order=3)
        assert r.success
        assert abs(r.y[-1][0] - 1.0) < 0.01

    def test_order1(self):
        # AM1 = trapezoidal
        r = solve_adams_moulton(exp_decay, (0, 1), [1.0], n_steps=200, order=1)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.001


# ============================================================
# 18. Event detection
# ============================================================

class TestEventDetection:
    def test_zero_crossing(self):
        # y' = -y, y(0)=1. Find when y = 0.5 (t = ln(2))
        def event_half(t, y):
            return y[0] - 0.5

        r = solve_rk45(exp_decay, (0, 5), [1.0], events=[event_half])
        assert r.success
        assert len(r.t_events[0]) >= 1
        assert abs(r.t_events[0][0] - math.log(2)) < 0.1

    def test_harmonic_zero(self):
        # Find when cos(t) = 0, i.e. t = pi/2
        def event_zero(t, y):
            return y[0]

        r = solve_rk45(harmonic_osc, (0, 4), [1.0, 0.0], events=[event_zero])
        assert r.success
        assert len(r.t_events[0]) >= 1
        assert abs(r.t_events[0][0] - math.pi / 2) < 0.05

    def test_multiple_events(self):
        def ev1(t, y):
            return y[0] - 0.5
        def ev2(t, y):
            return y[0] - 0.1

        r = solve_rk45(exp_decay, (0, 10), [1.0], events=[ev1, ev2])
        assert r.success
        assert len(r.t_events) == 2

    def test_callable_event(self):
        def event(t, y):
            return y[0] - 0.5
        r = solve_rk45(exp_decay, (0, 3), [1.0], events=event)
        assert len(r.t_events[0]) >= 1


# ============================================================
# 19. Dense Output
# ============================================================

class TestDenseOutput:
    def test_interpolation(self):
        r = solve_rk45(exp_decay, (0, 2), [1.0])
        dense = DenseOutput(r.t, r.y, exp_decay)

        # Check at several points
        for t_test in [0.1, 0.5, 1.0, 1.5]:
            y_interp = dense(t_test)
            exact = math.exp(-t_test)
            assert abs(y_interp[0] - exact) < 0.01

    def test_boundary(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=10, method='rk4')
        dense = DenseOutput(r.t, r.y, exp_decay)
        # At boundaries
        assert abs(dense(0)[0] - 1.0) < 1e-10
        assert abs(dense(1)[0] - r.y[-1][0]) < 1e-10

    def test_outside_range(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=10, method='rk4')
        dense = DenseOutput(r.t, r.y, exp_decay)
        # Before start
        assert abs(dense(-1)[0] - 1.0) < 1e-10
        # After end
        assert abs(dense(2)[0] - r.y[-1][0]) < 1e-10


# ============================================================
# 20. Stiffness detection
# ============================================================

class TestStiffnessDetection:
    def test_nonstiff(self):
        is_stiff, ratio = detect_stiffness(exp_decay, 0, [1.0], 0.01)
        # Simple decay: lambda = -1, h*lambda = 0.01, not stiff
        assert not is_stiff

    def test_stiff(self):
        is_stiff, ratio = detect_stiffness(stiff_system, 0, [0.0], 0.01)
        # lambda ~ -1000, h*lambda = 10, should be stiff
        assert is_stiff
        assert ratio > 3.0


# ============================================================
# 21. to_first_order
# ============================================================

class TestToFirstOrder:
    def test_harmonic(self):
        # y'' = -y -> [y, y'], y' = v, v' = -y
        def f_second(t, y, v):
            return -y  # y'' = -y
        f = to_first_order(f_second, 2)
        dydt = f(0, [1.0, 0.0])
        assert dydt == [0.0, -1.0]

    def test_damped_oscillator(self):
        # y'' + 0.1*y' + y = 0
        def f_second(t, y, v):
            return -0.1 * v - y
        f = to_first_order(f_second, 2)
        dydt = f(0, [1.0, 0.0])
        assert abs(dydt[0]) < 1e-12
        assert abs(dydt[1] - (-1.0)) < 1e-12


# ============================================================
# 22. make_system
# ============================================================

class TestMakeSystem:
    def test_lotka_volterra(self):
        eqs = [
            lambda t, x, y: x - 0.1 * x * y,
            lambda t, x, y: 0.1 * x * y - y,
        ]
        f = make_system(eqs)
        dydt = f(0, [10, 5])
        assert abs(dydt[0] - (10 - 0.1 * 10 * 5)) < 1e-12
        assert abs(dydt[1] - (0.1 * 10 * 5 - 5)) < 1e-12


# ============================================================
# 23. solve_ivp API
# ============================================================

class TestSolveIVP:
    def test_rk45(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='RK45')
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 1e-5

    def test_rk4(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='RK4', n_steps=100)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 1e-6

    def test_euler(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='Euler', n_steps=1000)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01

    def test_midpoint(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='Midpoint', n_steps=200)
        assert r.success

    def test_backward_euler(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='BackwardEuler', n_steps=100)
        assert r.success

    def test_trapz(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='Trapz', n_steps=100)
        assert r.success

    def test_bdf(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='BDF', n_steps=200)
        assert r.success

    def test_ab(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='AB', n_steps=200)
        assert r.success

    def test_am(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='AM', n_steps=200)
        assert r.success

    def test_dense_output(self):
        r = solve_ivp(exp_decay, (0, 2), [1.0], method='RK45', dense_output=True)
        assert r.success
        assert hasattr(r, 'sol')
        y_mid = r.sol(1.0)
        assert abs(y_mid[0] - math.exp(-1)) < 0.01

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            solve_ivp(exp_decay, (0, 1), [1.0], method='UNKNOWN')

    def test_scalar_input(self):
        r = solve_ivp(lambda t, y: [-y[0]], (0, 1), 1.0, method='RK45')
        assert r.success

    def test_crank_nicolson_alias(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='CN', n_steps=100)
        assert r.success

    def test_backward_euler_alias(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='Backward_Euler', n_steps=100)
        assert r.success


# ============================================================
# 24. solve_scalar
# ============================================================

class TestSolveScalar:
    def test_exp_decay(self):
        r = solve_scalar(lambda t, y: -y, (0, 1), 1.0)
        assert r.success
        # y values are scalars
        assert isinstance(r.y[-1], float)
        assert abs(r.y[-1] - math.exp(-1)) < 1e-5

    def test_logistic(self):
        # y' = y(1-y), y(0) = 0.1
        def logistic(t, y):
            return y * (1 - y)
        r = solve_scalar(logistic, (0, 10), 0.1)
        assert r.success
        # Should approach 1.0
        assert abs(r.y[-1] - 1.0) < 0.01


# ============================================================
# 25. Convergence order tests
# ============================================================

class TestConvergenceOrder:
    def _error_at_h(self, method, n_steps):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=n_steps, method=method)
        return abs(r.y[-1][0] - math.exp(-1))

    def test_euler_first_order(self):
        e1 = self._error_at_h('euler', 100)
        e2 = self._error_at_h('euler', 200)
        ratio = e1 / e2
        # First order: ratio ~ 2
        assert 1.5 < ratio < 2.5

    def test_rk4_fourth_order(self):
        e1 = self._error_at_h('rk4', 50)
        e2 = self._error_at_h('rk4', 100)
        ratio = e1 / e2
        # Fourth order: ratio ~ 16
        assert ratio > 10


# ============================================================
# 26. Systems
# ============================================================

class TestSystems:
    def test_lotka_volterra(self):
        r = solve_rk45(lotka_volterra, (0, 10), [10.0, 5.0])
        assert r.success
        # Populations should stay positive
        for yi in r.y:
            assert yi[0] > 0
            assert yi[1] > 0

    def test_vanderpol(self):
        r = solve_rk45(vanderpol, (0, 20), [2.0, 0.0])
        assert r.success
        # Should be oscillatory, not blow up
        for yi in r.y:
            assert abs(yi[0]) < 10

    def test_3d_lorenz(self):
        def lorenz(t, y, sigma=10, rho=28, beta=8/3):
            return [
                sigma * (y[1] - y[0]),
                y[0] * (rho - y[2]) - y[1],
                y[0] * y[1] - beta * y[2],
            ]
        r = solve_rk45(lorenz, (0, 1), [1.0, 1.0, 1.0])
        assert r.success
        assert len(r.y[-1]) == 3


# ============================================================
# 27. Stiff systems with implicit
# ============================================================

class TestStiffSystems:
    def test_stiff_backward_euler(self):
        r = solve_implicit(stiff_system, (0, 0.1), [0.0], n_steps=100,
                           method='backward_euler')
        assert r.success
        # Should track sin(t) approximately
        # Exact solution approaches sin(t) as transient decays

    def test_stiff_bdf(self):
        r = solve_bdf(stiff_system, (0, 0.1), [0.0], n_steps=200, order=2)
        assert r.success

    def test_stiff_trapz(self):
        r = solve_implicit(stiff_system, (0, 0.1), [0.0], n_steps=100,
                           method='trapz')
        assert r.success


# ============================================================
# 28. Energy conservation (symplectic-ish)
# ============================================================

class TestEnergyConservation:
    def test_harmonic_energy_rk4(self):
        r = solve_fixed(harmonic_osc, (0, 10), [1.0, 0.0], n_steps=1000,
                        method='rk4')
        # Energy = y^2 + v^2 should be ~ 1
        for yi in r.y:
            energy = yi[0] ** 2 + yi[1] ** 2
            assert abs(energy - 1.0) < 0.01

    def test_harmonic_energy_rk45(self):
        r = solve_rk45(harmonic_osc, (0, 10), [1.0, 0.0], rtol=1e-8)
        for yi in r.y:
            energy = yi[0] ** 2 + yi[1] ** 2
            assert abs(energy - 1.0) < 0.001


# ============================================================
# 29. Backward integration
# ============================================================

class TestBackwardIntegration:
    def test_backward_euler_fixed(self):
        # Integrate from t=1 back to t=0
        # y(1) = e^{-1}, y(0) should be 1
        r = solve_fixed(exp_decay, (1, 0), [math.exp(-1)], n_steps=100,
                        method='rk4')
        assert r.success
        assert abs(r.y[-1][0] - 1.0) < 0.01

    def test_backward_rk45(self):
        r = solve_rk45(exp_decay, (1, 0), [math.exp(-1)])
        assert r.success
        assert abs(r.y[-1][0] - 1.0) < 1e-4


# ============================================================
# 30. Edge cases
# ============================================================

class TestEdgeCases:
    def test_zero_interval(self):
        r = solve_fixed(exp_decay, (0, 0), [1.0], h=0.1, method='rk4')
        assert r.success
        assert len(r.t) == 1
        assert r.y[0] == [1.0]

    def test_single_component(self):
        r = solve_rk45(lambda t, y: [0.0], (0, 1), [42.0])
        assert r.success
        assert abs(r.y[-1][0] - 42.0) < 1e-10

    def test_large_system(self):
        n = 20
        def f(t, y):
            return [-y[i] * (i + 1) for i in range(n)]
        y0 = [1.0] * n
        r = solve_rk45(f, (0, 1), y0)
        assert r.success
        for i in range(n):
            assert abs(r.y[-1][i] - math.exp(-(i + 1))) < 0.01


# ============================================================
# 31. Comparison of methods on same problem
# ============================================================

class TestMethodComparison:
    def test_all_methods_agree(self):
        exact = math.exp(-1)
        methods = {
            'Euler': {'n_steps': 10000},
            'Midpoint': {'n_steps': 1000},
            'RK4': {'n_steps': 100},
            'RK45': {},
            'BackwardEuler': {'n_steps': 1000},
            'Trapz': {'n_steps': 100},
            'BDF': {'n_steps': 200},
            'AB': {'n_steps': 200},
            'AM': {'n_steps': 200},
        }
        for method, kwargs in methods.items():
            r = solve_ivp(exp_decay, (0, 1), [1.0], method=method, **kwargs)
            assert r.success, f"{method} failed"
            assert abs(r.y[-1][0] - exact) < 0.05, f"{method} error too large"


# ============================================================
# 32. BDF bootstrap behavior
# ============================================================

class TestBDFBootstrap:
    def test_order_rampup(self):
        # BDF5 should start with BDF1, then ramp up
        r = solve_bdf(exp_decay, (0, 1), [1.0], n_steps=100, order=5)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01


# ============================================================
# 33. Adams-Bashforth bootstrap
# ============================================================

class TestABBootstrap:
    def test_rk4_bootstrap(self):
        # AB5 uses RK4 for first 4 steps
        r = solve_adams_bashforth(exp_decay, (0, 1), [1.0], n_steps=100, order=5)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01


# ============================================================
# 34. Jacobian usage
# ============================================================

class TestJacobian:
    def test_backward_euler_with_jac(self):
        def jac(t, y):
            J = Matrix(1)
            J[0, 0] = -1.0
            return J
        r = solve_implicit(exp_decay, (0, 1), [1.0], n_steps=100,
                           method='backward_euler', jac=jac)
        assert r.success
        assert r.njev > 0

    def test_bdf_with_jac(self):
        def jac(t, y):
            J = Matrix(1)
            J[0, 0] = -1.0
            return J
        r = solve_bdf(exp_decay, (0, 1), [1.0], n_steps=100, order=2, jac=jac)
        assert r.success
        assert r.njev > 0


# ============================================================
# 35. nfev tracking
# ============================================================

class TestNfevTracking:
    def test_euler_nfev(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=10, method='euler')
        assert r.nfev == 10

    def test_rk4_nfev(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], n_steps=10, method='rk4')
        assert r.nfev == 40

    def test_rk45_nfev(self):
        r = solve_rk45(exp_decay, (0, 1), [1.0])
        assert r.nfev > 0


# ============================================================
# 36. Exact solutions
# ============================================================

class TestExactSolutions:
    def test_constant(self):
        # y' = 0, y(0) = 5
        r = solve_rk45(lambda t, y: [0.0], (0, 10), [5.0])
        assert all(abs(yi[0] - 5.0) < 1e-10 for yi in r.y)

    def test_linear_growth(self):
        # y' = 1, y(0) = 0 -> y = t
        r = solve_rk45(lambda t, y: [1.0], (0, 5), [0.0])
        assert abs(r.y[-1][0] - 5.0) < 1e-4

    def test_quadratic(self):
        # y' = 2t, y(0) = 0 -> y = t^2
        r = solve_rk45(lambda t, y: [2 * t], (0, 3), [0.0])
        assert abs(r.y[-1][0] - 9.0) < 1e-3


# ============================================================
# 37. Long-time integration
# ============================================================

class TestLongTimeIntegration:
    def test_decay_to_zero(self):
        r = solve_rk45(exp_decay, (0, 50), [1.0])
        assert r.success
        assert abs(r.y[-1][0]) < 1e-8

    def test_exp_growth_bounded(self):
        r = solve_rk45(exp_growth, (0, 5), [1.0])
        assert r.success
        assert abs(r.y[-1][0] - math.exp(5)) < 1.0


# ============================================================
# 38. Dense output integration test
# ============================================================

class TestDenseOutputIntegration:
    def test_solve_ivp_dense(self):
        r = solve_ivp(exp_decay, (0, 3), [1.0], method='RK45', dense_output=True)
        assert r.success
        # Check at evenly spaced points
        for i in range(31):
            t = i * 0.1
            y = r.sol(t)
            exact = math.exp(-t)
            assert abs(y[0] - exact) < 0.02


# ============================================================
# 39. System energy test
# ============================================================

class TestSystemEnergy:
    def test_pendulum(self):
        g, L = 9.81, 1.0
        def pendulum(t, y):
            return [y[1], -g / L * math.sin(y[0])]

        r = solve_rk45(pendulum, (0, 5), [0.5, 0.0], rtol=1e-8)
        assert r.success
        # Energy = 0.5*v^2 - g/L*cos(theta)
        E0 = 0.5 * 0.0 ** 2 - g / L * math.cos(0.5)
        for yi in r.y:
            E = 0.5 * yi[1] ** 2 - g / L * math.cos(yi[0])
            assert abs(E - E0) < 0.01


# ============================================================
# 40. RK45 max_step
# ============================================================

class TestRK45MaxStep:
    def test_max_step(self):
        r = solve_rk45(exp_decay, (0, 10), [1.0], max_step=0.1)
        assert r.success
        # With max_step=0.1 over [0,10], should have at least 100 steps
        assert len(r.t) >= 100

    def test_h0(self):
        r = solve_rk45(exp_decay, (0, 1), [1.0], h0=0.001)
        assert r.success


# ============================================================
# 41. Implicit with system
# ============================================================

class TestImplicitSystem:
    def test_backward_euler_system(self):
        r = solve_implicit(harmonic_osc, (0, 1), [1.0, 0.0], n_steps=200,
                           method='backward_euler')
        assert r.success
        assert abs(r.y[-1][0] - math.cos(1)) < 0.1

    def test_trapz_system(self):
        r = solve_implicit(harmonic_osc, (0, 1), [1.0, 0.0], n_steps=200,
                           method='trapz')
        assert r.success
        assert abs(r.y[-1][0] - math.cos(1)) < 0.01


# ============================================================
# 42. AM orders
# ============================================================

class TestAMOrders:
    def test_order2(self):
        r = solve_adams_moulton(exp_decay, (0, 1), [1.0], n_steps=200, order=2)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.001

    def test_order4(self):
        r = solve_adams_moulton(exp_decay, (0, 1), [1.0], n_steps=200, order=4)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.001


# ============================================================
# 43. AB orders
# ============================================================

class TestABOrders:
    def test_order3(self):
        r = solve_adams_bashforth(exp_decay, (0, 1), [1.0], n_steps=200, order=3)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01

    def test_order5(self):
        r = solve_adams_bashforth(exp_decay, (0, 1), [1.0], n_steps=200, order=5)
        assert r.success
        assert abs(r.y[-1][0] - math.exp(-1)) < 0.01


# ============================================================
# 44. BDF orders
# ============================================================

class TestBDFOrders:
    def test_order1(self):
        r = solve_bdf(exp_decay, (0, 1), [1.0], n_steps=200, order=1)
        assert r.success

    def test_order4(self):
        r = solve_bdf(exp_decay, (0, 1), [1.0], n_steps=200, order=4)
        assert r.success

    def test_order5(self):
        r = solve_bdf(exp_decay, (0, 1), [1.0], n_steps=200, order=5)
        assert r.success


# ============================================================
# 45. Solve IVP with events
# ============================================================

class TestSolveIVPEvents:
    def test_event_via_api(self):
        def event(t, y):
            return y[0] - 0.5

        r = solve_ivp(exp_decay, (0, 5), [1.0], method='RK45', events=[event])
        assert r.success
        assert len(r.t_events[0]) >= 1


# ============================================================
# 46. Nonlinear ODE
# ============================================================

class TestNonlinearODE:
    def test_riccati(self):
        # y' = y^2 + 1, y(0) = 0 -> y = tan(t), solution exists for t < pi/2
        def riccati(t, y):
            return [y[0] ** 2 + 1]
        r = solve_rk45(riccati, (0, 1.0), [0.0])
        assert r.success
        assert abs(r.y[-1][0] - math.tan(1.0)) < 0.01

    def test_bernoulli(self):
        # y' = y - y^2, y(0) = 0.5 -> y = 1/(1 + e^{-t})
        def bernoulli(t, y):
            return [y[0] - y[0] ** 2]
        r = solve_rk45(bernoulli, (0, 5), [0.5])
        exact = 1.0 / (1.0 + math.exp(-5))
        assert abs(r.y[-1][0] - exact) < 0.01


# ============================================================
# 47. Higher-order ODE via to_first_order
# ============================================================

class TestHigherOrderODE:
    def test_spring(self):
        # y'' + y = 0, y(0)=1, y'(0)=0 -> y=cos(t)
        f = to_first_order(lambda t, y, v: -y, 2)
        r = solve_rk45(f, (0, 2 * math.pi), [1.0, 0.0])
        assert r.success
        assert abs(r.y[-1][0] - 1.0) < 0.01

    def test_third_order(self):
        # y''' = -y'', y(0)=0, y'(0)=1, y''(0)=1 -> can compute numerically
        f = to_first_order(lambda t, y, v, a: -a, 3)
        r = solve_rk45(f, (0, 1), [0.0, 1.0, 1.0])
        assert r.success
        assert len(r.y[-1]) == 3


# ============================================================
# 48. make_system integration
# ============================================================

class TestMakeSystemIntegration:
    def test_coupled_decay(self):
        # x' = -x, y' = x - y
        eqs = [
            lambda t, x, y: -x,
            lambda t, x, y: x - y,
        ]
        f = make_system(eqs)
        r = solve_rk45(f, (0, 5), [1.0, 0.0])
        assert r.success
        # x(t) = e^{-t}
        assert abs(r.y[-1][0] - math.exp(-5)) < 1e-4


# ============================================================
# 49. Implicit convergence
# ============================================================

class TestImplicitConvergence:
    def test_trapz_higher_order_than_be(self):
        # Trapezoidal is O(h^2), backward Euler is O(h)
        e_be = abs(solve_implicit(exp_decay, (0, 1), [1.0], n_steps=100,
                                   method='backward_euler').y[-1][0] - math.exp(-1))
        e_tr = abs(solve_implicit(exp_decay, (0, 1), [1.0], n_steps=100,
                                   method='trapz').y[-1][0] - math.exp(-1))
        assert e_tr < e_be


# ============================================================
# 50. Comprehensive system test: SIR model
# ============================================================

class TestSIRModel:
    def test_sir(self):
        beta, gamma = 0.3, 0.1

        def sir(t, y):
            S, I, R = y
            return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

        r = solve_rk45(sir, (0, 100), [0.99, 0.01, 0.0])
        assert r.success
        # Population conservation: S + I + R = 1
        for yi in r.y:
            assert abs(sum(yi) - 1.0) < 1e-4
        # Eventually I -> 0
        assert r.y[-1][1] < 0.01


# ============================================================
# Additional tests for coverage
# ============================================================

class TestAdditional:
    def test_rk45_very_small_initial(self):
        r = solve_rk45(exp_decay, (0, 1), [1e-10])
        assert r.success

    def test_solve_ivp_adams_bashforth_alias(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='Adams_Bashforth', n_steps=200)
        assert r.success

    def test_solve_ivp_adams_moulton_alias(self):
        r = solve_ivp(exp_decay, (0, 1), [1.0], method='Adams_Moulton', n_steps=200)
        assert r.success

    def test_multiple_event_crossings(self):
        # Harmonic oscillator crosses y=0 twice per period
        def event_zero(t, y):
            return y[0]
        r = solve_rk45(harmonic_osc, (0, 4 * math.pi), [1.0, 0.0],
                        events=[event_zero])
        assert r.success
        # Should find ~4 zero crossings in 2 periods
        assert len(r.t_events[0]) >= 3

    def test_stiffness_with_jac(self):
        def jac(t, y):
            J = Matrix(1)
            J[0, 0] = -1000.0
            return J
        is_stiff, ratio = detect_stiffness(stiff_system, 0, [0.0], 0.01, jac=jac)
        assert is_stiff

    def test_bdf_jac_solve(self):
        def jac(t, y):
            J = Matrix(2)
            J[0, 0] = -1.0
            J[0, 1] = 1.0
            J[1, 0] = 0.0
            J[1, 1] = -1.0
            return J
        r = solve_bdf(linear_2d, (0, 1), [0.0, 1.0], n_steps=100,
                      order=2, jac=jac)
        assert r.success

    def test_solve_fixed_default_h(self):
        r = solve_fixed(exp_decay, (0, 1), [1.0], method='rk4')
        assert r.success
        assert len(r.t) == 101  # default 100 steps

    def test_solve_bdf_default_h(self):
        r = solve_bdf(exp_decay, (0, 1), [1.0], order=2)
        assert r.success

    def test_solve_am_default_h(self):
        r = solve_adams_moulton(exp_decay, (0, 1), [1.0], order=3)
        assert r.success

    def test_solve_ab_default_h(self):
        r = solve_adams_bashforth(exp_decay, (0, 1), [1.0], order=4)
        assert r.success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
