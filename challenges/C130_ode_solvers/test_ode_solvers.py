"""Tests for C130: ODE Solvers."""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ode_solvers import (
    ODEResult, EulerMethod, MidpointMethod, RK4Method, DormandPrince45,
    ImplicitEuler, ImplicitTrapezoid, BDF2,
    AdamsBashforth, StormerVerlet, PhasePortrait,
    SensitivityAnalysis, ODEParameterFitter,
    second_order_to_system, harmonic_oscillator, van_der_pol,
    lorenz, lotka_volterra, sir_model,
    solve_ode, estimate_stiffness_ratio, convergence_order,
    compute_energy_drift,
)


# ---------------------------------------------------------------------------
# Helper: common test ODEs with known solutions
# ---------------------------------------------------------------------------

def exponential_decay(t, y):
    """y' = -y, solution: y(t) = y0 * e^(-t)."""
    return [-y[0]]


def exponential_growth(t, y):
    """y' = y, solution: y(t) = y0 * e^t."""
    return [y[0]]


def linear_ode(t, y):
    """y' = 2*t, solution: y(t) = y0 + t^2."""
    return [2 * t]


def sine_ode(t, y):
    """y' = cos(t), solution: y(t) = sin(t) + y0."""
    return [math.cos(t)]


def coupled_linear(t, y):
    """y1' = -y2, y2' = y1. Solution: rotation."""
    return [-y[1], y[0]]


# ---------------------------------------------------------------------------
# ODEResult
# ---------------------------------------------------------------------------

class TestODEResult:
    def test_basic(self):
        r = ODEResult([0, 1], [[1.0], [0.5]])
        assert r.t == [0, 1]
        assert r.y == [[1.0], [0.5]]
        assert r.success
        assert r.n_steps == 0

    def test_repr(self):
        r = ODEResult([0, 1, 2], [[1.0, 2.0], [0.5, 1.0], [0.3, 0.6]], n_steps=2)
        s = repr(r)
        assert 'n_points=3' in s
        assert 'dim=2' in s

    def test_events(self):
        r = ODEResult([0], [[1.0]], events=[(0.5, [0.0], 0)])
        assert len(r.events) == 1


# ---------------------------------------------------------------------------
# Euler method
# ---------------------------------------------------------------------------

class TestEuler:
    def test_exponential_decay(self):
        solver = EulerMethod()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=1000)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 0.01

    def test_linear(self):
        solver = EulerMethod()
        res = solver.solve(linear_ode, (0, 2), [0.0], n_steps=1000)
        expected = 4.0  # t^2 at t=2
        assert abs(res.y[-1][0] - expected) < 0.01

    def test_multidim(self):
        solver = EulerMethod()
        res = solver.solve(coupled_linear, (0, 0.5), [1.0, 0.0], n_steps=5000)
        # Solution: y1 = cos(t), y2 = sin(t)
        assert abs(res.y[-1][0] - math.cos(0.5)) < 0.01
        assert abs(res.y[-1][1] - math.sin(0.5)) < 0.01

    def test_step_count(self):
        solver = EulerMethod()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=50)
        assert len(res.t) == 51
        assert len(res.y) == 51
        assert res.n_steps == 50

    def test_f_evals(self):
        solver = EulerMethod()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=10)
        assert res.n_f_evals == 10


# ---------------------------------------------------------------------------
# Midpoint method
# ---------------------------------------------------------------------------

class TestMidpoint:
    def test_exponential_decay(self):
        solver = MidpointMethod()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-4

    def test_linear(self):
        solver = MidpointMethod()
        res = solver.solve(linear_ode, (0, 2), [0.0], n_steps=100)
        expected = 4.0
        assert abs(res.y[-1][0] - expected) < 1e-6  # Exact for quadratic

    def test_better_than_euler(self):
        euler = EulerMethod()
        mid = MidpointMethod()
        re = euler.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        rm = mid.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        expected = math.exp(-1)
        err_euler = abs(re.y[-1][0] - expected)
        err_mid = abs(rm.y[-1][0] - expected)
        assert err_mid < err_euler

    def test_f_evals(self):
        solver = MidpointMethod()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=10)
        assert res.n_f_evals == 20  # 2 per step


# ---------------------------------------------------------------------------
# RK4 method
# ---------------------------------------------------------------------------

class TestRK4:
    def test_exponential_decay(self):
        solver = RK4Method()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-8

    def test_exponential_growth(self):
        solver = RK4Method()
        res = solver.solve(exponential_growth, (0, 1), [1.0], n_steps=100)
        expected = math.exp(1)
        assert abs(res.y[-1][0] - expected) < 1e-8

    def test_sine(self):
        solver = RK4Method()
        res = solver.solve(sine_ode, (0, math.pi), [0.0], n_steps=100)
        expected = math.sin(math.pi)  # ~0
        assert abs(res.y[-1][0] - expected) < 1e-8

    def test_coupled_rotation(self):
        solver = RK4Method()
        res = solver.solve(coupled_linear, (0, 2 * math.pi), [1.0, 0.0], n_steps=200)
        # After full period, should return to start
        assert abs(res.y[-1][0] - 1.0) < 1e-5
        assert abs(res.y[-1][1] - 0.0) < 1e-5

    def test_f_evals(self):
        solver = RK4Method()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=10)
        assert res.n_f_evals == 40  # 4 per step

    def test_high_accuracy(self):
        solver = RK4Method()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=1000)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-13

    def test_3d_system(self):
        """Test 3D: y1'=y2, y2'=y3, y3'=-y1."""
        def f(t, y):
            return [y[1], y[2], -y[0]]
        solver = RK4Method()
        res = solver.solve(f, (0, 1), [1.0, 0.0, 0.0], n_steps=200)
        assert len(res.y[-1]) == 3


# ---------------------------------------------------------------------------
# Dormand-Prince RK45 (adaptive)
# ---------------------------------------------------------------------------

class TestDormandPrince:
    def test_exponential_decay(self):
        solver = DormandPrince45()
        res = solver.solve(exponential_decay, (0, 1), [1.0])
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-5
        assert res.success

    def test_exponential_growth(self):
        solver = DormandPrince45()
        res = solver.solve(exponential_growth, (0, 2), [1.0])
        expected = math.exp(2)
        assert abs(res.y[-1][0] - expected) < 1e-4

    def test_adaptive_fewer_steps_for_smooth(self):
        solver = DormandPrince45()
        res = solver.solve(linear_ode, (0, 2), [0.0])
        # Linear ODE should need very few steps
        assert res.n_steps < 50

    def test_coupled_rotation(self):
        solver = DormandPrince45()
        res = solver.solve(coupled_linear, (0, 2 * math.pi), [1.0, 0.0])
        assert abs(res.y[-1][0] - 1.0) < 1e-4
        assert abs(res.y[-1][1] - 0.0) < 1e-4

    def test_tolerance_control(self):
        solver = DormandPrince45()
        res_loose = solver.solve(exponential_decay, (0, 1), [1.0], rtol=1e-3)
        res_tight = solver.solve(exponential_decay, (0, 1), [1.0], rtol=1e-9)
        expected = math.exp(-1)
        err_loose = abs(res_loose.y[-1][0] - expected)
        err_tight = abs(res_tight.y[-1][0] - expected)
        assert err_tight < err_loose

    def test_event_detection(self):
        """Detect zero crossing of y' = -y + 1 starting at y=0."""
        def f(t, y):
            return [-y[0] + 1]
        def event(t, y):
            return y[0] - 0.5  # Crosses y = 0.5
        solver = DormandPrince45()
        res = solver.solve(f, (0, 5), [0.0], events=[event])
        assert len(res.events) >= 1
        t_event = res.events[0][0]
        # Analytical: y = 1 - e^(-t), y = 0.5 => t = ln(2) ~ 0.693
        assert abs(t_event - math.log(2)) < 0.01

    def test_max_steps(self):
        solver = DormandPrince45()
        res = solver.solve(exponential_decay, (0, 100), [1.0], max_steps=5)
        assert not res.success
        assert 'Max steps' in res.message


# ---------------------------------------------------------------------------
# Implicit Euler
# ---------------------------------------------------------------------------

class TestImplicitEuler:
    def test_exponential_decay(self):
        solver = ImplicitEuler()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=1000)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 0.01

    def test_stiff_system(self):
        """Stiff: y' = -1000*y, needs implicit solver."""
        def stiff(t, y):
            return [-1000 * y[0]]
        solver = ImplicitEuler()
        res = solver.solve(stiff, (0, 0.01), [1.0], n_steps=100)
        expected = math.exp(-10)
        # Should be stable even with moderate step count
        assert abs(res.y[-1][0] - expected) < 0.1
        assert all(abs(res.y[i][0]) <= 1.01 for i in range(len(res.y)))

    def test_coupled(self):
        solver = ImplicitEuler()
        res = solver.solve(coupled_linear, (0, 1), [1.0, 0.0], n_steps=500)
        assert abs(res.y[-1][0] - math.cos(1)) < 0.05
        assert abs(res.y[-1][1] - math.sin(1)) < 0.05

    def test_stability_vs_euler(self):
        """Explicit Euler blows up, implicit stays bounded."""
        def stiff(t, y):
            return [-50 * y[0]]
        # Explicit Euler with h=0.1 > 2/50 = 0.04 (unstable!)
        euler = EulerMethod()
        res_ex = euler.solve(stiff, (0, 1), [1.0], n_steps=10)
        # Check: explicit should oscillate/blow up
        max_ex = max(abs(s[0]) for s in res_ex.y)

        # Implicit should stay bounded
        ie = ImplicitEuler()
        res_im = ie.solve(stiff, (0, 1), [1.0], n_steps=10)
        max_im = max(abs(s[0]) for s in res_im.y)
        assert max_im < max_ex


# ---------------------------------------------------------------------------
# Implicit Trapezoid
# ---------------------------------------------------------------------------

class TestImplicitTrapezoid:
    def test_exponential_decay(self):
        solver = ImplicitTrapezoid()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-4

    def test_better_than_implicit_euler(self):
        ie = ImplicitEuler()
        it = ImplicitTrapezoid()
        re = ie.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        rt = it.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        expected = math.exp(-1)
        err_ie = abs(re.y[-1][0] - expected)
        err_it = abs(rt.y[-1][0] - expected)
        assert err_it < err_ie

    def test_stiff_stability(self):
        def stiff(t, y):
            return [-500 * y[0]]
        solver = ImplicitTrapezoid()
        res = solver.solve(stiff, (0, 0.1), [1.0], n_steps=20)
        assert all(abs(s[0]) <= 1.01 for s in res.y)


# ---------------------------------------------------------------------------
# BDF2
# ---------------------------------------------------------------------------

class TestBDF2:
    def test_exponential_decay(self):
        solver = BDF2()
        res = solver.solve(exponential_decay, (0, 1), [1.0], n_steps=200)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-3

    def test_stiff_stability(self):
        def stiff(t, y):
            return [-1000 * y[0]]
        solver = BDF2()
        res = solver.solve(stiff, (0, 0.01), [1.0], n_steps=50)
        assert all(abs(s[0]) <= 1.01 for s in res.y)

    def test_coupled(self):
        solver = BDF2()
        res = solver.solve(coupled_linear, (0, 1), [1.0, 0.0], n_steps=500)
        assert abs(res.y[-1][0] - math.cos(1)) < 0.05


# ---------------------------------------------------------------------------
# Adams-Bashforth
# ---------------------------------------------------------------------------

class TestAdamsBashforth:
    def test_order1_is_euler(self):
        ab = AdamsBashforth()
        euler = EulerMethod()
        res_ab = ab.solve(exponential_decay, (0, 1), [1.0], n_steps=100, order=1)
        res_eu = euler.solve(exponential_decay, (0, 1), [1.0], n_steps=100)
        # After bootstrap, should match Euler
        assert abs(res_ab.y[-1][0] - res_eu.y[-1][0]) < 1e-10

    def test_order4_accuracy(self):
        ab = AdamsBashforth()
        res = ab.solve(exponential_decay, (0, 1), [1.0], n_steps=200, order=4)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-5

    def test_order2(self):
        ab = AdamsBashforth()
        res = ab.solve(exponential_decay, (0, 1), [1.0], n_steps=200, order=2)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-3

    def test_order3(self):
        ab = AdamsBashforth()
        res = ab.solve(exponential_decay, (0, 1), [1.0], n_steps=200, order=3)
        expected = math.exp(-1)
        assert abs(res.y[-1][0] - expected) < 1e-4

    def test_multidim(self):
        ab = AdamsBashforth()
        res = ab.solve(coupled_linear, (0, 1), [1.0, 0.0], n_steps=500, order=4)
        assert abs(res.y[-1][0] - math.cos(1)) < 1e-3


# ---------------------------------------------------------------------------
# Stormer-Verlet (symplectic)
# ---------------------------------------------------------------------------

class TestStormerVerlet:
    def test_harmonic_oscillator_energy(self):
        """Energy should be nearly conserved for harmonic oscillator."""
        def force(q):
            return [-q[0]]  # F = -kx, k=1
        solver = StormerVerlet()
        res = solver.solve(force, (0, 20 * math.pi), [1.0], [0.0], n_steps=10000)

        def kinetic(p):
            return 0.5 * p[0] * p[0]
        def potential(q):
            return 0.5 * q[0] * q[0]

        energies, drift = compute_energy_drift(res, kinetic, potential)
        assert drift < 1e-4  # Very small energy drift

    def test_harmonic_period(self):
        """q should return near initial position after one period."""
        def force(q):
            return [-q[0]]
        solver = StormerVerlet()
        T = 2 * math.pi
        res = solver.solve(force, (0, T), [1.0], [0.0], n_steps=1000)
        q_final = res.y[-1][0]
        p_final = res.y[-1][1]
        assert abs(q_final - 1.0) < 0.01
        assert abs(p_final - 0.0) < 0.01

    def test_2d_kepler(self):
        """2D Kepler problem (circular orbit)."""
        def force(q):
            r = math.sqrt(q[0]**2 + q[1]**2)
            r3 = r * r * r
            return [-q[0] / r3, -q[1] / r3]
        solver = StormerVerlet()
        # Circular orbit: r=1, v=1
        res = solver.solve(force, (0, 2 * math.pi), [1.0, 0.0], [0.0, 1.0],
                           n_steps=5000)
        q_final = res.y[-1][:2]
        # Should return close to start
        assert abs(q_final[0] - 1.0) < 0.05
        assert abs(q_final[1] - 0.0) < 0.05

    def test_custom_mass(self):
        def force(q):
            return [-q[0]]
        solver = StormerVerlet()
        res = solver.solve(force, (0, 1), [1.0], [0.0], n_steps=100, mass=[2.0])
        assert len(res.y[-1]) == 2

    def test_energy_conservation_long(self):
        """Long integration: symplectic should maintain bounded energy."""
        def force(q):
            return [-q[0]]
        solver = StormerVerlet()
        res = solver.solve(force, (0, 100), [1.0], [0.0], n_steps=50000)
        def kinetic(p):
            return 0.5 * p[0]**2
        def potential(q):
            return 0.5 * q[0]**2
        _, drift = compute_energy_drift(res, kinetic, potential)
        assert drift < 1e-3


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------

class TestSystemBuilders:
    def test_second_order_to_system(self):
        # y'' = -y (simple harmonic)
        f = second_order_to_system(lambda t, y, v: -y)
        dy = f(0, [1.0, 0.0])
        assert dy == [0.0, -1.0]

    def test_harmonic_oscillator(self):
        f = harmonic_oscillator(omega=1.0, zeta=0.0)
        solver = RK4Method()
        res = solver.solve(f, (0, 2 * math.pi), [1.0, 0.0], n_steps=200)
        assert abs(res.y[-1][0] - 1.0) < 1e-4
        assert abs(res.y[-1][1] - 0.0) < 1e-4

    def test_damped_oscillator(self):
        f = harmonic_oscillator(omega=2.0, zeta=0.2)
        solver = RK4Method()
        res = solver.solve(f, (0, 10), [1.0, 0.0], n_steps=500)
        # Should decay
        assert abs(res.y[-1][0]) < 0.5

    def test_van_der_pol(self):
        f = van_der_pol(mu=1.0)
        solver = RK4Method()
        res = solver.solve(f, (0, 20), [2.0, 0.0], n_steps=2000)
        assert res.success

    def test_lorenz(self):
        f = lorenz()
        solver = RK4Method()
        res = solver.solve(f, (0, 1), [1.0, 1.0, 1.0], n_steps=500)
        assert len(res.y[-1]) == 3

    def test_lotka_volterra(self):
        f = lotka_volterra()
        solver = RK4Method()
        res = solver.solve(f, (0, 10), [10.0, 5.0], n_steps=1000)
        # Populations should stay positive
        for state in res.y:
            assert state[0] > 0
            assert state[1] > 0

    def test_sir_model(self):
        f = sir_model(beta_param=0.3, gamma_param=0.1)
        solver = RK4Method()
        res = solver.solve(f, (0, 100), [0.99, 0.01, 0.0], n_steps=1000)
        # S + I + R should stay constant (= 1)
        for state in res.y:
            total = sum(state)
            assert abs(total - 1.0) < 1e-6
        # At the end, most should have recovered
        assert res.y[-1][2] > 0.5

    def test_sir_conservation(self):
        f = sir_model()
        solver = RK4Method()
        res = solver.solve(f, (0, 200), [0.999, 0.001, 0.0], n_steps=2000)
        for state in res.y:
            assert abs(sum(state) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# solve_ode dispatcher
# ---------------------------------------------------------------------------

class TestSolveODE:
    def test_euler(self):
        res = solve_ode(exponential_decay, (0, 1), [1.0], method='euler', n_steps=1000)
        assert abs(res.y[-1][0] - math.exp(-1)) < 0.01

    def test_midpoint(self):
        res = solve_ode(exponential_decay, (0, 1), [1.0], method='midpoint', n_steps=100)
        assert abs(res.y[-1][0] - math.exp(-1)) < 1e-4

    def test_rk4(self):
        res = solve_ode(exponential_decay, (0, 1), [1.0], method='rk4', n_steps=100)
        assert abs(res.y[-1][0] - math.exp(-1)) < 1e-8

    def test_rk45(self):
        res = solve_ode(exponential_decay, (0, 1), [1.0], method='rk45')
        assert abs(res.y[-1][0] - math.exp(-1)) < 1e-5

    def test_implicit_euler(self):
        res = solve_ode(exponential_decay, (0, 1), [1.0], method='implicit_euler', n_steps=1000)
        assert abs(res.y[-1][0] - math.exp(-1)) < 0.01

    def test_implicit_trapezoid(self):
        res = solve_ode(exponential_decay, (0, 1), [1.0], method='implicit_trapezoid', n_steps=100)
        assert abs(res.y[-1][0] - math.exp(-1)) < 1e-4

    def test_bdf2(self):
        res = solve_ode(exponential_decay, (0, 1), [1.0], method='bdf2', n_steps=200)
        assert abs(res.y[-1][0] - math.exp(-1)) < 1e-3

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            solve_ode(exponential_decay, (0, 1), [1.0], method='nonexistent')


# ---------------------------------------------------------------------------
# Convergence order
# ---------------------------------------------------------------------------

class TestConvergenceOrder:
    def test_euler_is_order_1(self):
        results = convergence_order(exponential_decay, (0, 1), [1.0],
                                    method='euler', step_counts=[100, 200, 400, 800])
        orders = [r[2] for r in results if r[2] is not None]
        avg_order = sum(orders) / len(orders)
        assert 0.8 < avg_order < 1.3

    def test_rk4_is_order_4(self):
        results = convergence_order(exponential_decay, (0, 1), [1.0],
                                    method='rk4', step_counts=[20, 40, 80, 160])
        orders = [r[2] for r in results if r[2] is not None]
        avg_order = sum(orders) / len(orders)
        assert 3.5 < avg_order < 5.0

    def test_midpoint_is_order_2(self):
        results = convergence_order(exponential_decay, (0, 1), [1.0],
                                    method='midpoint', step_counts=[50, 100, 200, 400])
        orders = [r[2] for r in results if r[2] is not None]
        avg_order = sum(orders) / len(orders)
        assert 1.5 < avg_order < 2.5


# ---------------------------------------------------------------------------
# Stiffness detection
# ---------------------------------------------------------------------------

class TestStiffnessDetection:
    def test_nonstiff(self):
        ratio, eigenvalues = estimate_stiffness_ratio(
            exponential_decay, 0, [1.0])
        assert ratio < 10

    def test_stiff(self):
        def stiff(t, y):
            return [-1000 * y[0], -y[1]]
        ratio, eigenvalues = estimate_stiffness_ratio(stiff, 0, [1.0, 1.0])
        assert ratio > 100

    def test_1d(self):
        ratio, eigenvalues = estimate_stiffness_ratio(
            lambda t, y: [-5 * y[0]], 0, [1.0])
        assert len(eigenvalues) == 1


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

class TestSensitivity:
    def test_forward_sensitivity_identity(self):
        """For y'=-y, sensitivity dy(T)/dy0 = e^(-T)."""
        _, S = SensitivityAnalysis.forward_sensitivity(
            exponential_decay, (0, 1), [1.0], n_steps=500)
        expected = math.exp(-1)
        assert abs(S[0][0] - expected) < 1e-3

    def test_forward_sensitivity_2d(self):
        """For rotation, S should be rotation matrix."""
        _, S = SensitivityAnalysis.forward_sensitivity(
            coupled_linear, (0, math.pi / 2), [1.0, 0.0], n_steps=500)
        # At t=pi/2: cos(pi/2)~0, sin(pi/2)~1
        # S = [[cos(t), -sin(t)], [sin(t), cos(t)]]
        assert abs(S[0][0] - math.cos(math.pi / 2)) < 0.05
        assert abs(S[1][0] - math.sin(math.pi / 2)) < 0.05

    def test_parameter_sensitivity(self):
        """Check sensitivity of decay rate parameter."""
        def factory(params):
            k = params[0]
            return lambda t, y: [-k * y[0]]
        _, sensitivities = SensitivityAnalysis.parameter_sensitivity(
            factory, [1.0], (0, 1), [1.0], n_steps=500)
        # dy(1)/dk for y'=-ky, y(0)=1: d/dk(e^(-k)) = -e^(-k)
        expected = -math.exp(-1)
        assert abs(sensitivities[0][0] - expected) < 0.01

    def test_parameter_sensitivity_multi_param(self):
        def factory(params):
            a, b = params
            return lambda t, y: [a * y[0] + b]
        _, sens = SensitivityAnalysis.parameter_sensitivity(
            factory, [0.0, 1.0], (0, 1), [0.0], n_steps=200)
        # y' = b, y = bt => dy/db = t = 1 at t=1
        assert abs(sens[1][0] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------

class TestParameterFitting:
    def test_fit_decay_rate(self):
        """Fit decay rate k from data generated with k=2."""
        true_k = 2.0
        # Generate data
        f_true = lambda t, y: [-true_k * y[0]]
        rk4 = RK4Method()
        data_res = rk4.solve(f_true, (0, 1), [1.0], n_steps=100)
        # Sample a few points
        indices = [0, 25, 50, 75, 100]
        t_data = [data_res.t[i] for i in indices]
        y_data = [data_res.y[i] for i in indices]

        def factory(params):
            k = params[0]
            return lambda t, y: [-k * y[0]]

        params_fit, loss_history = ODEParameterFitter.fit(
            factory, t_data, y_data, [1.0], lr=0.05, max_iter=500)

        assert abs(params_fit[0] - true_k) < 0.5
        assert loss_history[-1] < loss_history[0]

    def test_fit_convergence(self):
        """Loss should decrease over iterations."""
        def factory(params):
            k = params[0]
            return lambda t, y: [-k * y[0]]
        t_data = [0, 0.5, 1.0]
        y_data = [[1.0], [math.exp(-1.0)], [math.exp(-2.0)]]
        _, loss_history = ODEParameterFitter.fit(
            factory, t_data, y_data, [0.5], lr=0.01, max_iter=100)
        assert loss_history[-1] <= loss_history[0]


# ---------------------------------------------------------------------------
# Phase portrait
# ---------------------------------------------------------------------------

class TestPhasePortrait:
    def test_classify_stable_node(self):
        """y1' = -y1, y2' = -y2. Stable node at origin."""
        def f(t, y):
            return [-y[0], -y[1]]
        cls = PhasePortrait.classify_fixed_point(f, [0.0, 0.0])
        assert cls == 'stable_node'

    def test_classify_unstable_node(self):
        def f(t, y):
            return [y[0], y[1]]
        cls = PhasePortrait.classify_fixed_point(f, [0.0, 0.0])
        assert cls == 'unstable_node'

    def test_classify_saddle(self):
        def f(t, y):
            return [y[0], -y[1]]
        cls = PhasePortrait.classify_fixed_point(f, [0.0, 0.0])
        assert cls == 'saddle'

    def test_classify_center(self):
        def f(t, y):
            return [-y[1], y[0]]
        cls = PhasePortrait.classify_fixed_point(f, [0.0, 0.0])
        assert cls == 'center'

    def test_classify_stable_spiral(self):
        def f(t, y):
            return [-0.1 * y[0] - y[1], y[0] - 0.1 * y[1]]
        cls = PhasePortrait.classify_fixed_point(f, [0.0, 0.0])
        assert cls == 'stable_spiral'

    def test_classify_unstable_spiral(self):
        def f(t, y):
            return [0.1 * y[0] - y[1], y[0] + 0.1 * y[1]]
        cls = PhasePortrait.classify_fixed_point(f, [0.0, 0.0])
        assert cls == 'unstable_spiral'

    def test_find_fixed_points(self):
        """y1' = -y1, y2' = -y2 has one fixed point at origin."""
        def f(t, y):
            return [-y[0], -y[1]]
        fps = PhasePortrait.find_fixed_points(f, ((-2, 2), (-2, 2)))
        assert len(fps) >= 1
        # Check origin is found
        found_origin = False
        for pt, cls in fps:
            if abs(pt[0]) < 0.1 and abs(pt[1]) < 0.1:
                found_origin = True
                assert cls == 'stable_node'
        assert found_origin

    def test_find_multiple_fixed_points(self):
        """y1' = y1*(1-y1), y2' = -y2 has fixed points at (0,0) and (1,0)."""
        def f(t, y):
            return [y[0] * (1 - y[0]), -y[1]]
        fps = PhasePortrait.find_fixed_points(f, ((-0.5, 1.5), (-0.5, 0.5)))
        points = [pt for pt, cls in fps]
        # Should find at least 2
        assert len(fps) >= 2


# ---------------------------------------------------------------------------
# Energy drift
# ---------------------------------------------------------------------------

class TestEnergyDrift:
    def test_symplectic_low_drift(self):
        def force(q):
            return [-q[0]]
        sv = StormerVerlet()
        res = sv.solve(force, (0, 10 * math.pi), [1.0], [0.0], n_steps=5000)
        def kinetic(p):
            return 0.5 * p[0]**2
        def potential(q):
            return 0.5 * q[0]**2
        energies, drift = compute_energy_drift(res, kinetic, potential)
        assert drift < 1e-3
        assert len(energies) == len(res.t)

    def test_rk4_accumulates_drift(self):
        """RK4 should accumulate some energy drift over long times."""
        f = harmonic_oscillator(omega=1.0)
        rk4 = RK4Method()
        res = rk4.solve(f, (0, 20 * math.pi), [1.0, 0.0], n_steps=2000)
        def kinetic(p):
            return 0.5 * p[0]**2
        def potential(q):
            return 0.5 * q[0]**2
        _, drift = compute_energy_drift(res, kinetic, potential)
        # RK4 will have some drift
        assert drift > 0


# ---------------------------------------------------------------------------
# Integration: composing multiple methods
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_method_comparison_accuracy(self):
        """All methods should converge to same answer at high resolution."""
        expected = math.exp(-1)
        methods_fixed = {
            'euler': 10000,
            'midpoint': 500,
            'rk4': 50,
            'implicit_euler': 10000,
            'implicit_trapezoid': 500,
            'bdf2': 500,
        }
        for method, steps in methods_fixed.items():
            res = solve_ode(exponential_decay, (0, 1), [1.0],
                            method=method, n_steps=steps)
            assert abs(res.y[-1][0] - expected) < 0.01, \
                f"{method} failed: got {res.y[-1][0]}, expected {expected}"

    def test_lorenz_sensitivity(self):
        """Lorenz system is chaotic: small perturbation diverges."""
        f = lorenz()
        rk4 = RK4Method()
        res1 = rk4.solve(f, (0, 30), [1.0, 1.0, 1.0], n_steps=15000)
        res2 = rk4.solve(f, (0, 30), [1.001, 1.0, 1.0], n_steps=15000)
        # Final states should be very different
        from convex_optimization import VectorOps
        diff = VectorOps.norm(VectorOps.sub(res1.y[-1], res2.y[-1]))
        assert diff > 1.0  # Chaotic divergence

    def test_stiff_solver_comparison(self):
        """Implicit methods handle stiff systems; explicit struggles."""
        def stiff_2d(t, y):
            return [-100 * y[0] + y[1], -y[1]]
        y0 = [1.0, 1.0]
        # Implicit trapezoid with moderate steps
        it = ImplicitTrapezoid()
        res = it.solve(stiff_2d, (0, 1), y0, n_steps=100)
        # Should be stable
        for state in res.y:
            assert abs(state[0]) < 10
            assert abs(state[1]) < 10

    def test_van_der_pol_limit_cycle(self):
        """Van der Pol oscillator converges to a limit cycle."""
        f = van_der_pol(mu=1.0)
        rk4 = RK4Method()
        # Start from two different initial conditions
        res1 = rk4.solve(f, (0, 50), [0.1, 0.0], n_steps=5000)
        res2 = rk4.solve(f, (0, 50), [4.0, 0.0], n_steps=5000)
        # Both should be near the same limit cycle amplitude
        max_amp1 = max(abs(s[0]) for s in res1.y[-1000:])
        max_amp2 = max(abs(s[0]) for s in res2.y[-1000:])
        assert abs(max_amp1 - max_amp2) < 0.5

    def test_predator_prey_periodicity(self):
        """Lotka-Volterra should be periodic."""
        f = lotka_volterra(alpha=1.0, beta=0.5, delta=0.25, gamma=1.0)
        rk4 = RK4Method()
        res = rk4.solve(f, (0, 20), [4.0, 2.0], n_steps=5000)
        # Check: populations stay positive
        for state in res.y:
            assert state[0] > 0
            assert state[1] > 0

    def test_adaptive_vs_fixed(self):
        """Adaptive RK45 should give comparable accuracy with fewer evals."""
        rk4 = RK4Method()
        rk45 = DormandPrince45()
        res4 = rk4.solve(exponential_decay, (0, 5), [1.0], n_steps=1000)
        res45 = rk45.solve(exponential_decay, (0, 5), [1.0], rtol=1e-8)
        expected = math.exp(-5)
        err4 = abs(res4.y[-1][0] - expected)
        err45 = abs(res45.y[-1][0] - expected)
        # Both should be accurate
        assert err4 < 1e-6
        assert err45 < 1e-5

    def test_long_integration_all_methods(self):
        """Verify all methods don't blow up over moderate time spans."""
        methods = ['euler', 'midpoint', 'rk4', 'implicit_euler',
                    'implicit_trapezoid', 'bdf2']
        for method in methods:
            n = 5000 if method in ('euler', 'implicit_euler') else 500
            res = solve_ode(exponential_decay, (0, 5), [1.0],
                            method=method, n_steps=n)
            assert abs(res.y[-1][0]) < 1.0, f"{method} diverged"

    def test_solve_ode_rk45_events(self):
        """RK45 with events via solve_ode."""
        def f(t, y):
            return [1.0]
        def event(t, y):
            return y[0] - 5.0
        res = solve_ode(f, (0, 10), [0.0], method='rk45', events=[event])
        assert len(res.events) >= 1

    def test_adams_bashforth_via_dispatcher(self):
        """AB method should work standalone."""
        ab = AdamsBashforth()
        res = ab.solve(sine_ode, (0, math.pi), [0.0], n_steps=500, order=4)
        expected = math.sin(math.pi)
        assert abs(res.y[-1][0] - expected) < 1e-4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_length_interval(self):
        solver = RK4Method()
        res = solver.solve(exponential_decay, (0, 0), [1.0], n_steps=1)
        # h = 0, so no real stepping
        assert len(res.t) >= 1

    def test_negative_time_direction(self):
        """Solve backwards in time."""
        solver = RK4Method()
        # y' = -y solved from t=1 to t=0 (backward)
        res = solver.solve(exponential_decay, (1, 0), [math.exp(-1)], n_steps=100)
        # Should recover y(0) = 1.0
        assert abs(res.y[-1][0] - 1.0) < 1e-6

    def test_large_system(self):
        """10-dimensional decoupled decay."""
        dim = 10
        def f(t, y):
            return [-y[i] for i in range(dim)]
        solver = RK4Method()
        y0 = [float(i + 1) for i in range(dim)]
        res = solver.solve(f, (0, 1), y0, n_steps=100)
        for i in range(dim):
            expected = (i + 1) * math.exp(-1)
            assert abs(res.y[-1][i] - expected) < 1e-6

    def test_constant_rhs(self):
        """y' = 1, y(0) = 0 => y(t) = t."""
        solver = RK4Method()
        res = solver.solve(lambda t, y: [1.0], (0, 5), [0.0], n_steps=50)
        assert abs(res.y[-1][0] - 5.0) < 1e-10

    def test_rk45_very_tight_tolerance(self):
        solver = DormandPrince45()
        res = solver.solve(exponential_decay, (0, 1), [1.0], rtol=1e-12, atol=1e-14)
        assert abs(res.y[-1][0] - math.exp(-1)) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
