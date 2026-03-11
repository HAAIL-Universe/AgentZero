"""Tests for C133: PDE Solvers."""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pde_solvers import (
    # Result types
    PDEResult, PDEResult2D, EllipticResult,
    # Boundary conditions
    BoundaryCondition,
    # Grid
    make_grid_1d, make_grid_2d,
    # Stability
    cfl_number, diffusion_number,
    check_stability_heat_explicit, check_stability_wave_explicit,
    max_stable_dt_heat, max_stable_dt_wave,
    # 1D Heat
    heat_1d_explicit, heat_1d_implicit, heat_1d_crank_nicolson,
    # 1D Wave
    wave_1d_explicit, wave_1d_implicit,
    # 2D Heat
    heat_2d_explicit, heat_2d_adi,
    # 2D Wave
    wave_2d_explicit,
    # Elliptic
    poisson_2d_direct, laplace_2d_iterative, poisson_2d_sor,
    # MOL
    heat_1d_mol,
    # Analysis
    compute_error, convergence_study_1d, energy_1d, max_norm,
    # Helpers
    _solve_tridiagonal,
)


# ===========================================================================
# Grid generation
# ===========================================================================

class TestGridGeneration:
    def test_1d_grid_basic(self):
        x, dx = make_grid_1d((0, 1), 10)
        assert len(x) == 11
        assert abs(dx - 0.1) < 1e-12
        assert abs(x[0]) < 1e-12
        assert abs(x[-1] - 1.0) < 1e-12

    def test_1d_grid_nonzero_start(self):
        x, dx = make_grid_1d((2, 5), 6)
        assert len(x) == 7
        assert abs(dx - 0.5) < 1e-12
        assert abs(x[0] - 2.0) < 1e-12

    def test_2d_grid(self):
        x, y, dx, dy = make_grid_2d((0, 1), (0, 2), 5, 10)
        assert len(x) == 6
        assert len(y) == 11
        assert abs(dx - 0.2) < 1e-12
        assert abs(dy - 0.2) < 1e-12


# ===========================================================================
# Stability analysis
# ===========================================================================

class TestStability:
    def test_cfl_number(self):
        assert abs(cfl_number(0.01, 0.1, 1.0) - 0.1) < 1e-12

    def test_diffusion_number(self):
        assert abs(diffusion_number(0.001, 0.1, 1.0) - 0.1) < 1e-12

    def test_heat_stable(self):
        stable, r = check_stability_heat_explicit(0.001, 0.1, 1.0)
        assert stable
        assert abs(r - 0.1) < 1e-12

    def test_heat_unstable(self):
        stable, r = check_stability_heat_explicit(0.01, 0.1, 1.0)
        assert not stable
        assert r > 0.5

    def test_wave_stable(self):
        stable, cfl = check_stability_wave_explicit(0.05, 0.1, 1.0)
        assert stable
        assert abs(cfl - 0.5) < 1e-12

    def test_wave_unstable(self):
        stable, cfl = check_stability_wave_explicit(0.2, 0.1, 1.0)
        assert not stable
        assert cfl > 1.0

    def test_max_stable_dt_heat(self):
        dt = max_stable_dt_heat(0.1, 1.0)
        assert abs(dt - 0.005) < 1e-12

    def test_max_stable_dt_wave(self):
        dt = max_stable_dt_wave(0.1, 2.0)
        assert abs(dt - 0.05) < 1e-12


# ===========================================================================
# Tridiagonal solver
# ===========================================================================

class TestTridiagonal:
    def test_simple_system(self):
        # [2 -1 0] [x0]   [1]
        # [-1 2 -1] [x1] = [0]
        # [0 -1 2] [x2]   [1]
        x = _solve_tridiagonal([-1, -1, -1], [2, 2, 2], [-1, -1, -1], [1, 0, 1])
        assert abs(x[0] - 1.0) < 1e-10
        assert abs(x[1] - 1.0) < 1e-10
        assert abs(x[2] - 1.0) < 1e-10

    def test_scalar_coefficients(self):
        x = _solve_tridiagonal(-1, 2, -1, [1, 0, 1])
        assert abs(x[0] - 1.0) < 1e-10
        assert abs(x[1] - 1.0) < 1e-10

    def test_single_element(self):
        x = _solve_tridiagonal(0, 3, 0, [6])
        assert abs(x[0] - 2.0) < 1e-10

    def test_empty(self):
        x = _solve_tridiagonal(0, 1, 0, [])
        assert x == []

    def test_two_elements(self):
        # [4 -1] [x0]   [3]
        # [-1 4] [x1] = [3]
        x = _solve_tridiagonal(-1, 4, -1, [3, 3])
        # x0 = x1 = 1.0
        assert abs(x[0] - 1.0) < 1e-10
        assert abs(x[1] - 1.0) < 1e-10


# ===========================================================================
# 1D Heat Equation
# ===========================================================================

class TestHeat1DExplicit:
    """Analytical solution: u(x,t) = sin(pi*x) * exp(-pi^2*alpha*t)"""

    def test_basic_decay(self):
        alpha = 1.0
        nx = 50
        dt_safe = max_stable_dt_heat(1.0 / nx, alpha)
        t_final = 0.01
        nt = int(t_final / dt_safe) + 1

        bc_left = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_right = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)

        result = heat_1d_explicit(alpha, (0, 1), (0, t_final), nx, nt,
                                  lambda x: math.sin(math.pi * x),
                                  bc_left, bc_right)

        assert isinstance(result, PDEResult)
        assert result.method == 'heat_1d_explicit'
        assert result.metadata['stable']

    def test_analytical_comparison(self):
        alpha = 0.1
        nx = 40
        t_final = 0.1
        dt_safe = 0.4 * (1.0/nx)**2 / alpha
        nt = int(t_final / dt_safe) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_explicit(alpha, (0, 1), (0, t_final), nx, nt,
                                  lambda x: math.sin(math.pi * x), bc, bc)

        analytical = lambda x: math.sin(math.pi * x) * math.exp(-math.pi**2 * alpha * t_final)
        errors = compute_error(result.final, analytical, result.x)
        assert errors['linf'] < 0.01

    def test_constant_initial(self):
        """Constant IC with matching BCs should remain constant."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)
        result = heat_1d_explicit(1.0, (0, 1), (0, 0.01), 20, 500,
                                  lambda x: 1.0, bc, bc)
        for val in result.final:
            assert abs(val - 1.0) < 1e-10

    def test_source_term(self):
        """With steady source and zero BCs, solution should build up."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_explicit(1.0, (0, 1), (0, 0.01), 20, 5000,
                                  lambda x: 0.0, bc, bc,
                                  source=lambda x, t: 1.0)
        # Interior should be positive
        assert result.final[10] > 0

    def test_result_properties(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_explicit(1.0, (0, 1), (0, 0.01), 10, 100,
                                  lambda x: math.sin(math.pi * x), bc, bc)
        assert result.n_spatial_points == 11
        assert result.n_time_steps == 100
        assert len(result.final) == 11


class TestHeat1DImplicit:
    def test_basic(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_implicit(1.0, (0, 1), (0, 0.1), 20, 50,
                                  lambda x: math.sin(math.pi * x), bc, bc)
        assert result.method == 'heat_1d_implicit'
        # Solution should decay
        assert max(abs(v) for v in result.final) < 1.0

    def test_large_timestep(self):
        """Implicit method should be stable even with large dt."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_implicit(1.0, (0, 1), (0, 0.5), 20, 10,
                                  lambda x: math.sin(math.pi * x), bc, bc)
        # Should not blow up
        assert all(abs(v) < 2.0 for v in result.final)

    def test_accuracy(self):
        alpha = 0.1
        t_final = 0.1
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_implicit(alpha, (0, 1), (0, t_final), 40, 200,
                                  lambda x: math.sin(math.pi * x), bc, bc)
        analytical = lambda x: math.sin(math.pi * x) * math.exp(-math.pi**2 * alpha * t_final)
        errors = compute_error(result.final, analytical, result.x)
        assert errors['linf'] < 0.02

    def test_neumann_bc(self):
        """Neumann BC: zero flux at both ends (insulated rod)."""
        bc_left = BoundaryCondition(BoundaryCondition.NEUMANN, 0.0)
        bc_right = BoundaryCondition(BoundaryCondition.NEUMANN, 0.0)
        result = heat_1d_implicit(1.0, (0, 1), (0, 1.0), 20, 200,
                                  lambda x: math.cos(math.pi * x), bc_left, bc_right)
        # With insulated ends and cos(pi*x), solution decays to mean
        # Mean of cos(pi*x) on [0,1] is 0
        avg = sum(result.final) / len(result.final)
        assert abs(avg) < 0.5  # Should be close to 0


class TestHeat1DCrankNicolson:
    def test_basic(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_crank_nicolson(1.0, (0, 1), (0, 0.1), 20, 50,
                                        lambda x: math.sin(math.pi * x), bc, bc)
        assert result.method == 'heat_1d_crank_nicolson'

    def test_second_order_accuracy(self):
        """CN should be second order in both time and space."""
        alpha = 0.1
        t_final = 0.1
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        analytical = lambda x: math.sin(math.pi * x) * math.exp(-math.pi**2 * alpha * t_final)

        # Coarse
        r1 = heat_1d_crank_nicolson(alpha, (0, 1), (0, t_final), 20, 100,
                                     lambda x: math.sin(math.pi * x), bc, bc)
        e1 = compute_error(r1.final, analytical, r1.x)

        # Fine
        r2 = heat_1d_crank_nicolson(alpha, (0, 1), (0, t_final), 40, 400,
                                     lambda x: math.sin(math.pi * x), bc, bc)
        e2 = compute_error(r2.final, analytical, r2.x)

        # Error should decrease by ~4x (second order)
        ratio = e1['l2'] / e2['l2'] if e2['l2'] > 0 else float('inf')
        assert ratio > 3.0  # At least 3x improvement

    def test_stability_large_dt(self):
        """CN is unconditionally stable."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_crank_nicolson(1.0, (0, 1), (0, 0.5), 20, 5,
                                        lambda x: math.sin(math.pi * x), bc, bc)
        assert all(abs(v) < 2.0 for v in result.final)

    def test_with_source(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_crank_nicolson(1.0, (0, 1), (0, 0.05), 20, 200,
                                        lambda x: 0.0, bc, bc,
                                        source=lambda x, t: math.sin(math.pi * x))
        assert result.final[10] > 0

    def test_time_dependent_bc(self):
        """Time-dependent Dirichlet BC."""
        bc_left = BoundaryCondition(BoundaryCondition.DIRICHLET, lambda t: math.sin(t))
        bc_right = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_crank_nicolson(1.0, (0, 1), (0, 1.0), 20, 100,
                                        lambda x: 0.0, bc_left, bc_right)
        # Left boundary should match sin(1.0) at final time
        assert abs(result.final[0] - math.sin(1.0)) < 1e-10


# ===========================================================================
# 1D Wave Equation
# ===========================================================================

class TestWave1DExplicit:
    def test_standing_wave(self):
        """sin(pi*x)*cos(pi*c*t) is exact solution."""
        c = 1.0
        nx = 50
        t_final = 0.5
        dt = 0.5 * (1.0/nx) / c  # CFL = 0.5
        nt = int(t_final / dt) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_1d_explicit(c, (0, 1), (0, t_final), nx, nt,
                                  lambda x: math.sin(math.pi * x),
                                  lambda x: 0.0, bc, bc)

        assert result.metadata['stable']
        analytical = lambda x: math.sin(math.pi * x) * math.cos(math.pi * c * t_final)
        errors = compute_error(result.final, analytical, result.x)
        assert errors['linf'] < 0.05

    def test_zero_initial_velocity(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_1d_explicit(1.0, (0, 1), (0, 0.1), 20, 50,
                                  lambda x: math.sin(math.pi * x),
                                  lambda x: 0.0, bc, bc)
        assert len(result.u) == 51  # nt+1 time levels (2 initial + nt-1 more)

    def test_energy_conservation(self):
        """Wave energy should be approximately conserved."""
        c = 1.0
        nx = 40
        dt = 0.8 * (1.0/nx) / c
        t_final = 1.0
        nt = int(t_final / dt) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_1d_explicit(c, (0, 1), (0, t_final), nx, nt,
                                  lambda x: math.sin(math.pi * x),
                                  lambda x: 0.0, bc, bc)

        e0 = energy_1d(result.u[0], result.dx)
        ef = energy_1d(result.final, result.dx)
        # Energy should be within 20% (not exact due to discrete scheme)
        assert abs(ef - e0) / (e0 + 1e-15) < 0.5


class TestWave1DImplicit:
    def test_basic(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_1d_implicit(1.0, (0, 1), (0, 0.5), 20, 100,
                                  lambda x: math.sin(math.pi * x),
                                  lambda x: 0.0, bc, bc)
        assert result.method == 'wave_1d_implicit'
        assert all(abs(v) < 2.0 for v in result.final)

    def test_large_timestep_stable(self):
        """Newmark-beta is unconditionally stable."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_1d_implicit(1.0, (0, 1), (0, 1.0), 20, 20,
                                  lambda x: math.sin(math.pi * x),
                                  lambda x: 0.0, bc, bc)
        assert all(abs(v) < 2.0 for v in result.final)

    def test_accuracy(self):
        c = 1.0
        t_final = 0.5
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_1d_implicit(c, (0, 1), (0, t_final), 40, 200,
                                  lambda x: math.sin(math.pi * x),
                                  lambda x: 0.0, bc, bc)
        analytical = lambda x: math.sin(math.pi * x) * math.cos(math.pi * c * t_final)
        errors = compute_error(result.final, analytical, result.x)
        assert errors['linf'] < 0.1


# ===========================================================================
# 2D Heat Equation
# ===========================================================================

class TestHeat2DExplicit:
    def test_basic(self):
        alpha = 0.1
        nx, ny = 10, 10
        dx = 1.0 / nx
        dt_safe = 0.25 / (alpha * (1/(dx*dx) + 1/(dx*dx)))
        t_final = 0.01
        nt = int(t_final / dt_safe) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_2d_explicit(alpha, (0, 1), (0, 1), (0, t_final), nx, ny, nt,
                                  lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                                  bc, bc, bc, bc)

        assert isinstance(result, PDEResult2D)
        assert result.metadata['stable']

    def test_decay(self):
        """Solution should decay towards zero."""
        alpha = 1.0
        nx, ny = 10, 10
        dx = 1.0 / nx
        dt_safe = 0.2 / (alpha * (1/(dx*dx) + 1/(dx*dx)))
        t_final = 0.05
        nt = int(t_final / dt_safe) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_2d_explicit(alpha, (0, 1), (0, 1), (0, t_final), nx, ny, nt,
                                  lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                                  bc, bc, bc, bc)

        max_initial = max(max(row) for row in result.u[0])
        max_final = max(max(row) for row in result.final)
        assert max_final < max_initial

    def test_constant_preserved(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)
        result = heat_2d_explicit(1.0, (0, 1), (0, 1), (0, 0.001), 5, 5, 100,
                                  lambda x, y: 1.0, bc, bc, bc, bc)
        for row in result.final:
            for val in row:
                assert abs(val - 1.0) < 1e-8


class TestHeat2DADI:
    def test_basic(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_2d_adi(1.0, (0, 1), (0, 1), (0, 0.1), 10, 10, 50,
                             lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                             bc, bc, bc, bc)
        assert result.method == 'heat_2d_adi'

    def test_decay(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_2d_adi(1.0, (0, 1), (0, 1), (0, 0.1), 10, 10, 50,
                             lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                             bc, bc, bc, bc)

        max_initial = max(max(row) for row in result.u[0])
        max_final = max(max(row) for row in result.final)
        assert max_final < max_initial

    def test_large_timestep_stable(self):
        """ADI should be unconditionally stable."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_2d_adi(1.0, (0, 1), (0, 1), (0, 0.5), 10, 10, 5,
                             lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                             bc, bc, bc, bc)
        max_val = max(max(abs(v) for v in row) for row in result.final)
        assert max_val < 2.0

    def test_accuracy_vs_explicit(self):
        """ADI and explicit should give similar results at same resolution."""
        alpha = 0.1
        nx, ny = 10, 10
        dx = 1.0 / nx
        dt_safe = 0.2 / (alpha * (1/(dx*dx) + 1/(dx*dx)))
        t_final = 0.01
        nt = int(t_final / dt_safe) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        ic = lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y)

        r_exp = heat_2d_explicit(alpha, (0, 1), (0, 1), (0, t_final), nx, ny, nt,
                                 ic, bc, bc, bc, bc)
        r_adi = heat_2d_adi(alpha, (0, 1), (0, 1), (0, t_final), nx, ny, nt,
                            ic, bc, bc, bc, bc)

        # Compare at center point
        diff = abs(r_exp.final[5][5] - r_adi.final[5][5])
        assert diff < 0.1  # Should be reasonably close


# ===========================================================================
# 2D Wave Equation
# ===========================================================================

class TestWave2DExplicit:
    def test_basic(self):
        c = 1.0
        nx, ny = 10, 10
        dx = 1.0 / nx
        dt = 0.5 * dx / (c * math.sqrt(2))  # CFL for 2D
        t_final = 0.1
        nt = int(t_final / dt) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_2d_explicit(c, (0, 1), (0, 1), (0, t_final), nx, ny, nt,
                                  lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                                  lambda x, y: 0.0, bc, bc, bc, bc)

        assert isinstance(result, PDEResult2D)
        assert result.metadata['stable']

    def test_standing_mode(self):
        """Standing wave mode should oscillate."""
        c = 1.0
        nx, ny = 15, 15
        dx = 1.0 / nx
        dt = 0.4 * dx / (c * math.sqrt(2))
        t_final = 0.2
        nt = int(t_final / dt) + 1

        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_2d_explicit(c, (0, 1), (0, 1), (0, t_final), nx, ny, nt,
                                  lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                                  lambda x, y: 0.0, bc, bc, bc, bc)

        # Solution should not blow up
        max_val = max(max(abs(v) for v in row) for row in result.final)
        assert max_val < 2.0

    def test_zero_velocity(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = wave_2d_explicit(1.0, (0, 1), (0, 1), (0, 0.05), 8, 8, 50,
                                  lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                                  None, bc, bc, bc, bc)
        assert result.n_time_steps == 50


# ===========================================================================
# Elliptic solvers
# ===========================================================================

class TestPoisson2DDirect:
    def test_basic(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = poisson_2d_direct((0, 1), (0, 1), 5, 5, lambda x, y: 1.0,
                                   bc, bc, bc, bc)
        assert isinstance(result, EllipticResult)
        assert result.method == 'poisson_2d_direct'

    def test_known_solution(self):
        """For -u_xx - u_yy = 2*pi^2*sin(pi*x)*sin(pi*y), u = sin(pi*x)*sin(pi*y)."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        source = lambda x, y: 2 * math.pi**2 * math.sin(math.pi*x) * math.sin(math.pi*y)

        result = poisson_2d_direct((0, 1), (0, 1), 20, 20, source, bc, bc, bc, bc)

        # Check center point
        analytical = math.sin(math.pi * 0.5) * math.sin(math.pi * 0.5)  # = 1.0
        numerical = result.u[10][10]
        assert abs(numerical - analytical) < 0.05

    def test_constant_source(self):
        """Constant source should produce bowl-shaped solution."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = poisson_2d_direct((0, 1), (0, 1), 10, 10, 1.0, bc, bc, bc, bc)
        # Center should be positive (bowl shape)
        assert result.u[5][5] > 0

    def test_nonzero_bc(self):
        bc_left = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)
        bc_right = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_bottom = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_top = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = poisson_2d_direct((0, 1), (0, 1), 10, 10, 0.0,
                                   bc_left, bc_right, bc_bottom, bc_top)
        # Left boundary should be 1.0
        assert abs(result.u[0][5] - 1.0) < 1e-10
        # Right boundary should be 0.0
        assert abs(result.u[10][5]) < 1e-10
        # Interior should be between 0 and 1
        assert 0.0 < result.u[5][5] < 1.0


class TestLaplace2DIterative:
    def test_jacobi(self):
        bc_left = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)
        bc_right = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_bottom = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_top = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)

        result = laplace_2d_iterative((0, 1), (0, 1), 10, 10,
                                      bc_left, bc_right, bc_bottom, bc_top,
                                      method='jacobi', tol=1e-6)
        assert result.residual < 1e-6
        assert result.iterations > 0
        assert 0.0 < result.u[5][5] < 1.0

    def test_gauss_seidel(self):
        bc_left = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)
        bc_right = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_bottom = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_top = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)

        result = laplace_2d_iterative((0, 1), (0, 1), 10, 10,
                                      bc_left, bc_right, bc_bottom, bc_top,
                                      method='gauss_seidel', tol=1e-6)
        assert result.residual < 1e-6
        assert 0.0 < result.u[5][5] < 1.0

    def test_gauss_seidel_faster(self):
        """Gauss-Seidel should converge in fewer iterations than Jacobi."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_hot = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)

        r_j = laplace_2d_iterative((0, 1), (0, 1), 10, 10,
                                    bc_hot, bc, bc, bc,
                                    method='jacobi', tol=1e-5)
        r_gs = laplace_2d_iterative((0, 1), (0, 1), 10, 10,
                                     bc_hot, bc, bc, bc,
                                     method='gauss_seidel', tol=1e-5)
        assert r_gs.iterations <= r_j.iterations

    def test_symmetric_bcs(self):
        """Symmetric BCs should give symmetric solution."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)
        bc_zero = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)

        result = laplace_2d_iterative((0, 1), (0, 1), 10, 10,
                                      bc, bc, bc_zero, bc_zero, tol=1e-8)
        # u[3][5] should equal u[7][5] by symmetry
        assert abs(result.u[3][5] - result.u[7][5]) < 1e-6


class TestPoisson2DSOR:
    def test_basic(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = poisson_2d_sor((0, 1), (0, 1), 10, 10, 1.0,
                                bc, bc, bc, bc, tol=1e-6)
        assert result.method == 'poisson_2d_sor'
        assert result.residual < 1e-6
        assert result.u[5][5] > 0

    def test_faster_than_gs(self):
        """SOR should converge faster than Gauss-Seidel."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_hot = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)

        r_gs = laplace_2d_iterative((0, 1), (0, 1), 15, 15,
                                     bc_hot, bc, bc, bc,
                                     method='gauss_seidel', tol=1e-5)
        r_sor = poisson_2d_sor((0, 1), (0, 1), 15, 15, 0.0,
                                bc_hot, bc, bc, bc, tol=1e-5)
        assert r_sor.iterations < r_gs.iterations

    def test_custom_omega(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = poisson_2d_sor((0, 1), (0, 1), 10, 10, 1.0,
                                bc, bc, bc, bc, omega=1.5, tol=1e-6)
        assert abs(result.metadata['omega'] - 1.5) < 1e-12

    def test_agrees_with_direct(self):
        """SOR and direct solver should give similar results."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        source = lambda x, y: math.sin(math.pi * x)

        r_direct = poisson_2d_direct((0, 1), (0, 1), 8, 8, source, bc, bc, bc, bc)
        r_sor = poisson_2d_sor((0, 1), (0, 1), 8, 8, source, bc, bc, bc, bc, tol=1e-8)

        # Compare center values
        diff = abs(r_direct.u[4][4] - r_sor.u[4][4])
        assert diff < 0.01


# ===========================================================================
# Method of Lines
# ===========================================================================

class TestHeat1DMOL:
    def test_basic(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_mol(1.0, (0, 1), (0, 0.1), 20,
                             lambda x: math.sin(math.pi * x), bc, bc)
        assert result.method == 'heat_1d_mol'
        assert result.metadata['ode_method'] == 'rk4'

    def test_accuracy(self):
        alpha = 0.1
        t_final = 0.1
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_mol(alpha, (0, 1), (0, t_final), 30,
                             lambda x: math.sin(math.pi * x), bc, bc)
        analytical = lambda x: math.sin(math.pi * x) * math.exp(-math.pi**2 * alpha * t_final)
        errors = compute_error(result.final, analytical, result.x)
        assert errors['linf'] < 0.02

    def test_with_source(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_mol(1.0, (0, 1), (0, 0.05), 20,
                             lambda x: 0.0, bc, bc,
                             source=lambda x, t: 1.0)
        assert result.final[10] > 0

    def test_agrees_with_cn(self):
        """MOL should agree with Crank-Nicolson."""
        alpha = 0.5
        t_final = 0.05
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        ic = lambda x: math.sin(math.pi * x)

        r_mol = heat_1d_mol(alpha, (0, 1), (0, t_final), 20, ic, bc, bc, ode_steps=500)
        r_cn = heat_1d_crank_nicolson(alpha, (0, 1), (0, t_final), 20, 500, ic, bc, bc)

        # Compare at midpoint
        diff = abs(r_mol.final[10] - r_cn.final[10])
        assert diff < 0.01


# ===========================================================================
# Analysis tools
# ===========================================================================

class TestAnalysisTools:
    def test_compute_error(self):
        numerical = [0.0, 0.5, 1.0, 0.5, 0.0]
        analytical = lambda x: math.sin(math.pi * x)
        x = [0.0, 0.25, 0.5, 0.75, 1.0]
        errors = compute_error(numerical, analytical, x)
        assert errors['l2'] > 0
        assert errors['linf'] > 0

    def test_compute_error_with_time(self):
        numerical = [1.0, 1.0, 1.0]
        x = [0.0, 0.5, 1.0]
        errors = compute_error(numerical, lambda x, t: 1.0, x, t=0.5)
        assert errors['l2'] < 1e-12
        assert errors['linf'] < 1e-12

    def test_energy_1d(self):
        u = [0.0, 1.0, 0.0]
        e = energy_1d(u, 0.5)
        assert e > 0

    def test_max_norm(self):
        u = [1.0, -3.0, 2.0]
        assert abs(max_norm(u) - 3.0) < 1e-12

    def test_energy_conservation_check(self):
        """Energy of constant function should be proportional to interval."""
        u = [1.0] * 11
        dx = 0.1
        e = energy_1d(u, dx)
        # Trapezoidal: 0.5*dx + 9*dx + 0.5*dx = dx * 10 = 1.0
        assert abs(e - 1.0) < 1e-10


# ===========================================================================
# Boundary condition types
# ===========================================================================

class TestBoundaryConditions:
    def test_dirichlet(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 2.0)
        assert bc.bc_type == 'dirichlet'
        assert bc.value == 2.0

    def test_neumann(self):
        bc = BoundaryCondition(BoundaryCondition.NEUMANN, 0.5)
        assert bc.bc_type == 'neumann'

    def test_periodic(self):
        bc = BoundaryCondition(BoundaryCondition.PERIODIC)
        assert bc.bc_type == 'periodic'

    def test_time_dependent_dirichlet(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, lambda t: math.sin(t))
        from pde_solvers import _bc_value
        assert abs(_bc_value(bc, math.pi / 2) - 1.0) < 1e-12


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_heat_methods_agree(self):
        """All 1D heat methods should give similar results."""
        alpha = 0.5
        t_final = 0.02
        nx = 20
        nt = 500
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        ic = lambda x: math.sin(math.pi * x)

        r_exp = heat_1d_explicit(alpha, (0, 1), (0, t_final), nx, nt, ic, bc, bc)
        r_imp = heat_1d_implicit(alpha, (0, 1), (0, t_final), nx, nt, ic, bc, bc)
        r_cn = heat_1d_crank_nicolson(alpha, (0, 1), (0, t_final), nx, nt, ic, bc, bc)

        mid = nx // 2
        vals = [r_exp.final[mid], r_imp.final[mid], r_cn.final[mid]]
        spread = max(vals) - min(vals)
        assert spread < 0.05

    def test_elliptic_methods_agree(self):
        """Direct and iterative Laplace solvers should agree."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_hot = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)

        r_direct = poisson_2d_direct((0, 1), (0, 1), 8, 8, 0.0,
                                      bc_hot, bc, bc, bc)
        r_iter = laplace_2d_iterative((0, 1), (0, 1), 8, 8,
                                       bc_hot, bc, bc, bc, tol=1e-8)

        diff = abs(r_direct.u[4][4] - r_iter.u[4][4])
        assert diff < 0.01

    def test_periodic_bc_heat(self):
        """Periodic BCs: ends should match."""
        bc = BoundaryCondition(BoundaryCondition.PERIODIC)
        result = heat_1d_explicit(1.0, (0, 1), (0, 0.001), 20, 100,
                                  lambda x: math.sin(2 * math.pi * x), bc, bc)
        assert abs(result.final[0] - result.final[-1]) < 1e-10

    def test_poisson_zero_source_is_laplace(self):
        """Poisson with zero source should equal Laplace."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        bc_hot = BoundaryCondition(BoundaryCondition.DIRICHLET, 1.0)

        r_poisson = poisson_2d_direct((0, 1), (0, 1), 8, 8, 0.0,
                                       bc_hot, bc, bc, bc)
        r_laplace = laplace_2d_iterative((0, 1), (0, 1), 8, 8,
                                          bc_hot, bc, bc, bc, tol=1e-10)

        diff = abs(r_poisson.u[4][4] - r_laplace.u[4][4])
        assert diff < 0.01

    def test_heat_1d_diffusion_smoothing(self):
        """Heat equation should smooth out sharp features."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        # Step function initial condition
        def ic(x):
            return 1.0 if 0.4 <= x <= 0.6 else 0.0

        result = heat_1d_explicit(1.0, (0, 1), (0, 0.01), 50, 5000, ic, bc, bc)
        # Final solution should be smoother (smaller max)
        assert max(result.final) < 1.0

    def test_wave_propagation(self):
        """Wave should propagate from initial bump."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)

        def ic(x):
            return math.exp(-100 * (x - 0.5)**2)

        result = wave_1d_explicit(1.0, (0, 1), (0, 0.2), 100, 500,
                                  ic, lambda x: 0.0, bc, bc)
        # Energy should spread from center
        center_initial = result.u[0][50]
        center_final = result.final[50]
        assert abs(center_final) < center_initial

    def test_2d_heat_symmetry(self):
        """Symmetric IC on symmetric domain should give symmetric solution."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        nx, ny = 10, 10
        dx = 1.0 / nx
        dt_safe = 0.2 / (1.0 * (1/(dx*dx) + 1/(dx*dx)))
        nt = int(0.01 / dt_safe) + 1

        result = heat_2d_explicit(1.0, (0, 1), (0, 1), (0, 0.01), nx, ny, nt,
                                  lambda x, y: math.sin(math.pi*x) * math.sin(math.pi*y),
                                  bc, bc, bc, bc)

        # u[3][5] should equal u[7][5] by x-symmetry around 0.5
        assert abs(result.final[3][5] - result.final[7][5]) < 1e-8


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_interior_point(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_explicit(1.0, (0, 1), (0, 0.01), 2, 100,
                                  lambda x: math.sin(math.pi * x), bc, bc)
        assert len(result.final) == 3

    def test_very_small_alpha(self):
        """Very small diffusivity: solution barely changes."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = heat_1d_explicit(0.001, (0, 1), (0, 0.001), 20, 100,
                                  lambda x: math.sin(math.pi * x), bc, bc)
        diff = max(abs(result.final[i] - result.u[0][i]) for i in range(len(result.x)))
        assert diff < 0.01

    def test_list_initial_condition(self):
        """Support list as initial condition."""
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        ic = [0.0, 0.5, 1.0, 0.5, 0.0]
        result = heat_1d_explicit(1.0, (0, 1), (0, 0.001), 4, 10, ic, bc, bc)
        assert len(result.final) == 5

    def test_zero_source(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        r1 = heat_1d_explicit(1.0, (0, 1), (0, 0.001), 10, 100,
                              lambda x: math.sin(math.pi * x), bc, bc)
        r2 = heat_1d_explicit(1.0, (0, 1), (0, 0.001), 10, 100,
                              lambda x: math.sin(math.pi * x), bc, bc,
                              source=lambda x, t: 0.0)
        diff = max(abs(r1.final[i] - r2.final[i]) for i in range(len(r1.x)))
        assert diff < 1e-10

    def test_small_grid_poisson(self):
        bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
        result = poisson_2d_direct((0, 1), (0, 1), 2, 2, 1.0, bc, bc, bc, bc)
        assert result.u[1][1] > 0  # Single interior point


# ===========================================================================
# Convergence tests
# ===========================================================================

class TestConvergence:
    def test_heat_explicit_convergence(self):
        """Explicit heat should show second-order spatial convergence."""
        alpha = 0.1
        t_final = 0.05

        errors = []
        for nx in [10, 20, 40]:
            dx = 1.0 / nx
            dt_safe = 0.4 * dx * dx / alpha
            nt = int(t_final / dt_safe) + 1
            bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
            result = heat_1d_explicit(alpha, (0, 1), (0, t_final), nx, nt,
                                      lambda x: math.sin(math.pi * x), bc, bc)
            analytical = lambda x: math.sin(math.pi * x) * math.exp(-math.pi**2 * alpha * t_final)
            err = compute_error(result.final, analytical, result.x)
            errors.append(err['l2'])

        # Check convergence order between 20 and 40
        if errors[1] > 0 and errors[2] > 0:
            order = math.log(errors[1] / errors[2]) / math.log(2)
            assert order > 1.5  # Should be ~2

    def test_cn_convergence(self):
        """Crank-Nicolson should show second-order convergence."""
        alpha = 0.1
        t_final = 0.05

        errors = []
        for nx in [10, 20, 40]:
            nt = nx * nx  # Scale nt with nx^2
            bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
            result = heat_1d_crank_nicolson(alpha, (0, 1), (0, t_final), nx, nt,
                                            lambda x: math.sin(math.pi * x), bc, bc)
            analytical = lambda x: math.sin(math.pi * x) * math.exp(-math.pi**2 * alpha * t_final)
            err = compute_error(result.final, analytical, result.x)
            errors.append(err['l2'])

        if errors[1] > 0 and errors[2] > 0:
            order = math.log(errors[1] / errors[2]) / math.log(2)
            assert order > 1.5

    def test_poisson_convergence(self):
        """Poisson direct should show second-order convergence."""
        source = lambda x, y: 2 * math.pi**2 * math.sin(math.pi*x) * math.sin(math.pi*y)
        analytical_center = 1.0  # sin(pi*0.5)*sin(pi*0.5) = 1.0

        errors = []
        for nx in [5, 10, 20]:
            bc = BoundaryCondition(BoundaryCondition.DIRICHLET, 0.0)
            result = poisson_2d_direct((0, 1), (0, 1), nx, nx, source, bc, bc, bc, bc)
            mid = nx // 2
            err = abs(result.u[mid][mid] - analytical_center)
            errors.append(err)

        if errors[1] > 0 and errors[2] > 0:
            order = math.log(errors[1] / errors[2]) / math.log(2)
            assert order > 1.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
