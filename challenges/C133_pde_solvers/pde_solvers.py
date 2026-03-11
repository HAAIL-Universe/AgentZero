"""C133: PDE Solvers -- Finite Difference Methods for Partial Differential Equations.

Composes C130 ODE Solvers + C132 Linear Algebra.

Supports:
- 1D/2D heat equation (parabolic)
- 1D/2D wave equation (hyperbolic)
- 2D Laplace/Poisson equation (elliptic)
- Method of Lines (MOL) -- reduce PDE to ODE system, solve with C130
- Crank-Nicolson implicit time stepping
- ADI (Alternating Direction Implicit) for 2D problems
- Boundary conditions: Dirichlet, Neumann, periodic
- Stability analysis and CFL condition checking
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C132_linear_algebra'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C130_ode_solvers'))

from linear_algebra import Matrix, lu_solve, lu_decompose
from ode_solvers import solve_ode, RK4Method


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class PDEResult:
    """Container for PDE solution results."""

    def __init__(self, x, t, u, method, dx, dt, metadata=None):
        self.x = x          # spatial grid (list of floats)
        self.t = t          # time points (list of floats)
        self.u = u          # solution: u[time_idx][space_idx] for 1D
        self.method = method
        self.dx = dx
        self.dt = dt
        self.metadata = metadata or {}

    @property
    def final(self):
        """Final time solution."""
        return self.u[-1]

    @property
    def n_time_steps(self):
        return len(self.t) - 1

    @property
    def n_spatial_points(self):
        return len(self.x)


class PDEResult2D:
    """Container for 2D PDE solution results."""

    def __init__(self, x, y, t, u, method, dx, dy, dt, metadata=None):
        self.x = x          # x grid
        self.y = y          # y grid
        self.t = t          # time points
        self.u = u          # solution: u[time_idx][i][j]
        self.method = method
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.metadata = metadata or {}

    @property
    def final(self):
        return self.u[-1]

    @property
    def n_time_steps(self):
        return len(self.t) - 1


class EllipticResult:
    """Container for elliptic (steady-state) PDE results."""

    def __init__(self, x, y, u, method, dx, dy, iterations=0, residual=0.0, metadata=None):
        self.x = x
        self.y = y
        self.u = u          # u[i][j]
        self.method = method
        self.dx = dx
        self.dy = dy
        self.iterations = iterations
        self.residual = residual
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Boundary condition types
# ---------------------------------------------------------------------------

class BoundaryCondition:
    DIRICHLET = 'dirichlet'
    NEUMANN = 'neumann'
    PERIODIC = 'periodic'

    def __init__(self, bc_type, value=0.0):
        self.bc_type = bc_type
        self.value = value  # for Dirichlet: fixed value or callable(t)
                            # for Neumann: flux value or callable(t)


def _bc_value(bc, t=0.0):
    """Get boundary condition value, supporting callable or constant."""
    if callable(bc.value):
        return bc.value(t)
    return bc.value


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def make_grid_1d(x_span, nx):
    """Create uniform 1D grid. Returns (x, dx)."""
    x0, x1 = x_span
    dx = (x1 - x0) / nx
    x = [x0 + i * dx for i in range(nx + 1)]
    return x, dx


def make_grid_2d(x_span, y_span, nx, ny):
    """Create uniform 2D grid. Returns (x, y, dx, dy)."""
    x, dx = make_grid_1d(x_span, nx)
    y, dy = make_grid_1d(y_span, ny)
    return x, y, dx, dy


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def cfl_number(dt, dx, c):
    """Compute CFL number: c * dt / dx."""
    return abs(c) * dt / dx


def diffusion_number(dt, dx, alpha):
    """Compute diffusion number: alpha * dt / dx^2."""
    return alpha * dt / (dx * dx)


def check_stability_heat_explicit(dt, dx, alpha):
    """Check stability for explicit heat equation. Requires r <= 0.5."""
    r = diffusion_number(dt, dx, alpha)
    return r <= 0.5, r


def check_stability_wave_explicit(dt, dx, c):
    """Check CFL condition for explicit wave equation. Requires CFL <= 1."""
    cfl = cfl_number(dt, dx, c)
    return cfl <= 1.0, cfl


def max_stable_dt_heat(dx, alpha):
    """Maximum stable dt for explicit heat equation."""
    return 0.5 * dx * dx / alpha


def max_stable_dt_wave(dx, c):
    """Maximum stable dt for wave equation (CFL=1)."""
    return dx / abs(c)


# ---------------------------------------------------------------------------
# 1D Heat Equation: u_t = alpha * u_xx + source(x, t)
# ---------------------------------------------------------------------------

def heat_1d_explicit(alpha, x_span, t_span, nx, nt, initial, bc_left, bc_right,
                     source=None):
    """Solve 1D heat equation using explicit (FTCS) finite differences.

    Args:
        alpha: thermal diffusivity
        x_span: (x0, x1)
        t_span: (t0, t_final)
        nx: number of spatial intervals
        nt: number of time steps
        initial: callable(x) -> u0 or list of values
        bc_left, bc_right: BoundaryCondition objects
        source: optional callable(x, t) -> source term
    """
    x, dx = make_grid_1d(x_span, nx)
    t0, tf = t_span
    dt = (tf - t0) / nt
    r = alpha * dt / (dx * dx)

    # Initialize
    if callable(initial):
        u = [initial(xi) for xi in x]
    else:
        u = list(initial)

    t_points = [t0]
    u_history = [list(u)]

    for n in range(nt):
        t_curr = t0 + n * dt
        t_next = t_curr + dt
        u_new = list(u)

        # Interior points
        for i in range(1, nx):
            s = source(x[i], t_curr) if source else 0.0
            u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1]) + dt * s

        # Boundary conditions
        _apply_bc_1d(u_new, bc_left, bc_right, dx, t_next, nx)

        u = u_new
        t_points.append(t_next)
        u_history.append(list(u))

    stable, r_val = check_stability_heat_explicit(dt, dx, alpha)
    return PDEResult(x, t_points, u_history, 'heat_1d_explicit', dx, dt,
                     metadata={'r': r_val, 'stable': stable, 'alpha': alpha})


def heat_1d_implicit(alpha, x_span, t_span, nx, nt, initial, bc_left, bc_right,
                     source=None):
    """Solve 1D heat equation using fully implicit (backward Euler) method.

    Unconditionally stable. Solves tridiagonal system each step.
    """
    x, dx = make_grid_1d(x_span, nx)
    t0, tf = t_span
    dt = (tf - t0) / nt
    r = alpha * dt / (dx * dx)

    if callable(initial):
        u = [initial(xi) for xi in x]
    else:
        u = list(initial)

    t_points = [t0]
    u_history = [list(u)]

    for n in range(nt):
        t_curr = t0 + n * dt
        t_next = t_curr + dt

        # Build tridiagonal system for interior points
        # (1 + 2r) u_i^{n+1} - r u_{i-1}^{n+1} - r u_{i+1}^{n+1} = u_i^n + dt*source
        size = nx - 1  # interior points
        if size <= 0:
            t_points.append(t_next)
            u_history.append(list(u))
            continue

        rhs = [0.0] * size
        for i in range(size):
            idx = i + 1
            s = source(x[idx], t_next) if source else 0.0
            rhs[i] = u[idx] + dt * s

        # Apply boundary conditions to RHS
        left_val = _get_bc_dirichlet_value(bc_left, u, dx, t_next, 'left')
        right_val = _get_bc_dirichlet_value(bc_right, u, dx, t_next, 'right')
        rhs[0] += r * left_val
        rhs[-1] += r * right_val

        # Solve tridiagonal
        u_interior = _solve_tridiagonal(-r, 1 + 2*r, -r, rhs)

        u_new = [left_val] + u_interior + [right_val]

        # Handle Neumann BCs
        if bc_left.bc_type == BoundaryCondition.NEUMANN:
            flux = _bc_value(bc_left, t_next)
            u_new[0] = u_new[1] - dx * flux
        if bc_right.bc_type == BoundaryCondition.NEUMANN:
            flux = _bc_value(bc_right, t_next)
            u_new[nx] = u_new[nx-1] + dx * flux

        u = u_new
        t_points.append(t_next)
        u_history.append(list(u))

    return PDEResult(x, t_points, u_history, 'heat_1d_implicit', dx, dt,
                     metadata={'r': r, 'alpha': alpha})


def heat_1d_crank_nicolson(alpha, x_span, t_span, nx, nt, initial, bc_left, bc_right,
                           source=None):
    """Solve 1D heat equation using Crank-Nicolson (2nd order in time and space).

    Unconditionally stable, O(dt^2 + dx^2).
    """
    x, dx = make_grid_1d(x_span, nx)
    t0, tf = t_span
    dt = (tf - t0) / nt
    r = alpha * dt / (dx * dx)

    if callable(initial):
        u = [initial(xi) for xi in x]
    else:
        u = list(initial)

    t_points = [t0]
    u_history = [list(u)]

    for n in range(nt):
        t_curr = t0 + n * dt
        t_next = t_curr + dt

        size = nx - 1
        if size <= 0:
            t_points.append(t_next)
            u_history.append(list(u))
            continue

        # RHS: explicit part
        rhs = [0.0] * size
        for i in range(size):
            idx = i + 1
            explicit_part = (r/2) * (u[idx+1] - 2*u[idx] + u[idx-1])
            s_curr = source(x[idx], t_curr) if source else 0.0
            s_next = source(x[idx], t_next) if source else 0.0
            rhs[i] = u[idx] + explicit_part + dt/2 * (s_curr + s_next)

        left_val = _get_bc_dirichlet_value(bc_left, u, dx, t_next, 'left')
        right_val = _get_bc_dirichlet_value(bc_right, u, dx, t_next, 'right')
        rhs[0] += (r/2) * left_val
        rhs[-1] += (r/2) * right_val

        # LHS tridiagonal: (1 + r) on diagonal, -r/2 off-diagonal
        u_interior = _solve_tridiagonal(-r/2, 1 + r, -r/2, rhs)

        u_new = [left_val] + u_interior + [right_val]

        if bc_left.bc_type == BoundaryCondition.NEUMANN:
            flux = _bc_value(bc_left, t_next)
            u_new[0] = u_new[1] - dx * flux
        if bc_right.bc_type == BoundaryCondition.NEUMANN:
            flux = _bc_value(bc_right, t_next)
            u_new[nx] = u_new[nx-1] + dx * flux

        u = u_new
        t_points.append(t_next)
        u_history.append(list(u))

    return PDEResult(x, t_points, u_history, 'heat_1d_crank_nicolson', dx, dt,
                     metadata={'r': r, 'alpha': alpha})


# ---------------------------------------------------------------------------
# 1D Wave Equation: u_tt = c^2 * u_xx
# ---------------------------------------------------------------------------

def wave_1d_explicit(c, x_span, t_span, nx, nt, initial_displacement,
                     initial_velocity, bc_left, bc_right):
    """Solve 1D wave equation using explicit central differences.

    Uses three-level scheme: u^{n+1} = 2u^n - u^{n-1} + cfl^2 * (u_{i+1} - 2u_i + u_{i-1})
    Stable when CFL = c*dt/dx <= 1.
    """
    x, dx = make_grid_1d(x_span, nx)
    t0, tf = t_span
    dt = (tf - t0) / nt
    cfl = c * dt / dx
    cfl2 = cfl * cfl

    # Level 0: initial displacement
    if callable(initial_displacement):
        u_prev = [initial_displacement(xi) for xi in x]
    else:
        u_prev = list(initial_displacement)

    # Level 1: use initial velocity for first step
    if callable(initial_velocity):
        v0 = [initial_velocity(xi) for xi in x]
    else:
        v0 = list(initial_velocity)

    u_curr = [0.0] * (nx + 1)
    for i in range(1, nx):
        u_curr[i] = (u_prev[i] + dt * v0[i] +
                      0.5 * cfl2 * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]))

    _apply_bc_1d(u_curr, bc_left, bc_right, dx, t0 + dt, nx)

    t_points = [t0, t0 + dt]
    u_history = [list(u_prev), list(u_curr)]

    for n in range(1, nt):
        t_next = t0 + (n + 1) * dt
        u_new = [0.0] * (nx + 1)

        for i in range(1, nx):
            u_new[i] = (2*u_curr[i] - u_prev[i] +
                        cfl2 * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1]))

        _apply_bc_1d(u_new, bc_left, bc_right, dx, t_next, nx)

        u_prev = u_curr
        u_curr = u_new
        t_points.append(t_next)
        u_history.append(list(u_new))

    stable, cfl_val = check_stability_wave_explicit(dt, dx, c)
    return PDEResult(x, t_points, u_history, 'wave_1d_explicit', dx, dt,
                     metadata={'cfl': cfl_val, 'stable': stable, 'c': c})


def wave_1d_implicit(c, x_span, t_span, nx, nt, initial_displacement,
                     initial_velocity, bc_left, bc_right):
    """Solve 1D wave equation using implicit Newmark-beta method (beta=0.25, gamma=0.5).

    Unconditionally stable, second-order accurate.
    """
    x, dx = make_grid_1d(x_span, nx)
    t0, tf = t_span
    dt = (tf - t0) / nt
    beta = 0.25
    gamma = 0.5

    # Initialize displacement and velocity
    if callable(initial_displacement):
        u = [initial_displacement(xi) for xi in x]
    else:
        u = list(initial_displacement)

    if callable(initial_velocity):
        v = [initial_velocity(xi) for xi in x]
    else:
        v = list(initial_velocity)

    # Compute initial acceleration: a = c^2 * u_xx
    a = [0.0] * (nx + 1)
    for i in range(1, nx):
        a[i] = c*c * (u[i+1] - 2*u[i] + u[i-1]) / (dx*dx)

    t_points = [t0]
    u_history = [list(u)]

    coeff = c * c / (dx * dx)

    for n in range(nt):
        t_next = t0 + (n + 1) * dt

        # Predictor
        u_pred = [0.0] * (nx + 1)
        v_pred = [0.0] * (nx + 1)
        for i in range(nx + 1):
            u_pred[i] = u[i] + dt * v[i] + dt*dt * (0.5 - beta) * a[i]
            v_pred[i] = v[i] + dt * (1 - gamma) * a[i]

        # Build system for interior: (1 + beta*dt^2*c^2*2/dx^2) a_new[i] - beta*dt^2*c^2/dx^2 * (a_new[i-1] + a_new[i+1]) = c^2 * u_xx_pred
        size = nx - 1
        if size <= 0:
            t_points.append(t_next)
            u_history.append(list(u))
            continue

        bdt2 = beta * dt * dt * coeff

        rhs = [0.0] * size
        for i in range(size):
            idx = i + 1
            rhs[i] = coeff * (u_pred[idx+1] - 2*u_pred[idx] + u_pred[idx-1])

        a_interior = _solve_tridiagonal(-bdt2, 1 + 2*bdt2, -bdt2, rhs)

        a_new = [0.0] * (nx + 1)
        for i in range(size):
            a_new[i+1] = a_interior[i]

        # Corrector
        u_new = [0.0] * (nx + 1)
        v_new = [0.0] * (nx + 1)
        for i in range(nx + 1):
            u_new[i] = u_pred[i] + beta * dt * dt * a_new[i]
            v_new[i] = v_pred[i] + gamma * dt * a_new[i]

        _apply_bc_1d(u_new, bc_left, bc_right, dx, t_next, nx)

        u = u_new
        v = v_new
        a = a_new
        t_points.append(t_next)
        u_history.append(list(u_new))

    return PDEResult(x, t_points, u_history, 'wave_1d_implicit', dx, dt,
                     metadata={'c': c, 'beta': beta, 'gamma': gamma})


# ---------------------------------------------------------------------------
# 2D Heat Equation: u_t = alpha * (u_xx + u_yy)
# ---------------------------------------------------------------------------

def heat_2d_explicit(alpha, x_span, y_span, t_span, nx, ny, nt, initial,
                     bc_x_left, bc_x_right, bc_y_bottom, bc_y_top, source=None):
    """Solve 2D heat equation using explicit FTCS.

    Stability requires: alpha * dt * (1/dx^2 + 1/dy^2) <= 0.5
    """
    x, y, dx, dy = make_grid_2d(x_span, y_span, nx, ny)
    t0, tf = t_span
    dt = (tf - t0) / nt
    rx = alpha * dt / (dx * dx)
    ry = alpha * dt / (dy * dy)

    # Initialize
    u = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(nx + 1):
        for j in range(ny + 1):
            if callable(initial):
                u[i][j] = initial(x[i], y[j])
            else:
                u[i][j] = initial[i][j]

    t_points = [t0]
    u_history = [_copy_2d(u)]

    for n in range(nt):
        t_curr = t0 + n * dt
        t_next = t_curr + dt
        u_new = [[0.0] * (ny + 1) for _ in range(nx + 1)]

        # Interior
        for i in range(1, nx):
            for j in range(1, ny):
                s = source(x[i], y[j], t_curr) if source else 0.0
                u_new[i][j] = (u[i][j] +
                               rx * (u[i+1][j] - 2*u[i][j] + u[i-1][j]) +
                               ry * (u[i][j+1] - 2*u[i][j] + u[i][j-1]) +
                               dt * s)

        _apply_bc_2d(u_new, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                     x, y, dx, dy, nx, ny, t_next)

        u = u_new
        t_points.append(t_next)
        u_history.append(_copy_2d(u))

    r_total = rx + ry
    return PDEResult2D(x, y, t_points, u_history, 'heat_2d_explicit', dx, dy, dt,
                       metadata={'rx': rx, 'ry': ry, 'stable': r_total <= 0.5})


def heat_2d_adi(alpha, x_span, y_span, t_span, nx, ny, nt, initial,
                bc_x_left, bc_x_right, bc_y_bottom, bc_y_top, source=None):
    """Solve 2D heat equation using ADI (Alternating Direction Implicit) method.

    Peaceman-Rachford ADI: unconditionally stable, O(dt^2 + dx^2 + dy^2).
    Each half-step solves a tridiagonal system.
    """
    x, y, dx, dy = make_grid_2d(x_span, y_span, nx, ny)
    t0, tf = t_span
    dt = (tf - t0) / nt
    rx = alpha * dt / (2 * dx * dx)
    ry = alpha * dt / (2 * dy * dy)

    # Initialize
    u = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(nx + 1):
        for j in range(ny + 1):
            if callable(initial):
                u[i][j] = initial(x[i], y[j])
            else:
                u[i][j] = initial[i][j]

    t_points = [t0]
    u_history = [_copy_2d(u)]

    for n in range(nt):
        t_curr = t0 + n * dt
        t_half = t_curr + dt / 2
        t_next = t_curr + dt

        # Half-step 1: implicit in x, explicit in y
        u_half = [[0.0] * (ny + 1) for _ in range(nx + 1)]

        for j in range(1, ny):
            # Build tridiagonal for x-direction at fixed j
            size = nx - 1
            rhs = [0.0] * size
            for i in range(size):
                idx = i + 1
                s = source(x[idx], y[j], t_curr) if source else 0.0
                rhs[i] = (u[idx][j] +
                          ry * (u[idx][j+1] - 2*u[idx][j] + u[idx][j-1]) +
                          dt/2 * s)

            # BCs in x
            left_val = _bc_value(bc_x_left, t_half)
            right_val = _bc_value(bc_x_right, t_half)
            rhs[0] += rx * left_val
            rhs[-1] += rx * right_val

            interior = _solve_tridiagonal(-rx, 1 + 2*rx, -rx, rhs)
            u_half[0][j] = left_val
            u_half[nx][j] = right_val
            for i in range(size):
                u_half[i+1][j] = interior[i]

        # Half-step 2: implicit in y, explicit in x
        u_new = [[0.0] * (ny + 1) for _ in range(nx + 1)]

        for i in range(1, nx):
            size = ny - 1
            rhs = [0.0] * size
            for j in range(size):
                jdx = j + 1
                s = source(x[i], y[jdx], t_half) if source else 0.0
                rhs[j] = (u_half[i][jdx] +
                          rx * (u_half[i+1][jdx] - 2*u_half[i][jdx] + u_half[i-1][jdx]) +
                          dt/2 * s)

            bottom_val = _bc_value(bc_y_bottom, t_next)
            top_val = _bc_value(bc_y_top, t_next)
            rhs[0] += ry * bottom_val
            rhs[-1] += ry * top_val

            interior = _solve_tridiagonal(-ry, 1 + 2*ry, -ry, rhs)
            u_new[i][0] = bottom_val
            u_new[i][ny] = top_val
            for j in range(size):
                u_new[i][j+1] = interior[j]

        _apply_bc_2d(u_new, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                     x, y, dx, dy, nx, ny, t_next)

        u = u_new
        t_points.append(t_next)
        u_history.append(_copy_2d(u))

    return PDEResult2D(x, y, t_points, u_history, 'heat_2d_adi', dx, dy, dt,
                       metadata={'rx': rx, 'ry': ry})


# ---------------------------------------------------------------------------
# 2D Wave Equation: u_tt = c^2 * (u_xx + u_yy)
# ---------------------------------------------------------------------------

def wave_2d_explicit(c, x_span, y_span, t_span, nx, ny, nt,
                     initial_displacement, initial_velocity,
                     bc_x_left, bc_x_right, bc_y_bottom, bc_y_top):
    """Solve 2D wave equation using explicit central differences.

    CFL condition: c * dt * sqrt(1/dx^2 + 1/dy^2) <= 1
    """
    x, y, dx, dy = make_grid_2d(x_span, y_span, nx, ny)
    t0, tf = t_span
    dt = (tf - t0) / nt
    rx2 = (c * dt / dx) ** 2
    ry2 = (c * dt / dy) ** 2

    # Level 0
    u_prev = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(nx + 1):
        for j in range(ny + 1):
            if callable(initial_displacement):
                u_prev[i][j] = initial_displacement(x[i], y[j])
            else:
                u_prev[i][j] = initial_displacement[i][j]

    # Level 1 from velocity
    v0 = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    if callable(initial_velocity):
        for i in range(nx + 1):
            for j in range(ny + 1):
                v0[i][j] = initial_velocity(x[i], y[j])
    elif initial_velocity is not None:
        for i in range(nx + 1):
            for j in range(ny + 1):
                v0[i][j] = initial_velocity[i][j]

    u_curr = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(1, nx):
        for j in range(1, ny):
            u_curr[i][j] = (u_prev[i][j] + dt * v0[i][j] +
                            0.5 * rx2 * (u_prev[i+1][j] - 2*u_prev[i][j] + u_prev[i-1][j]) +
                            0.5 * ry2 * (u_prev[i][j+1] - 2*u_prev[i][j] + u_prev[i][j-1]))

    _apply_bc_2d(u_curr, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                 x, y, dx, dy, nx, ny, t0 + dt)

    t_points = [t0, t0 + dt]
    u_history = [_copy_2d(u_prev), _copy_2d(u_curr)]

    for n in range(1, nt):
        t_next = t0 + (n + 1) * dt
        u_new = [[0.0] * (ny + 1) for _ in range(nx + 1)]

        for i in range(1, nx):
            for j in range(1, ny):
                u_new[i][j] = (2*u_curr[i][j] - u_prev[i][j] +
                               rx2 * (u_curr[i+1][j] - 2*u_curr[i][j] + u_curr[i-1][j]) +
                               ry2 * (u_curr[i][j+1] - 2*u_curr[i][j] + u_curr[i][j-1]))

        _apply_bc_2d(u_new, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                     x, y, dx, dy, nx, ny, t_next)

        u_prev = u_curr
        u_curr = u_new
        t_points.append(t_next)
        u_history.append(_copy_2d(u_new))

    cfl_2d = c * dt * math.sqrt(1/(dx*dx) + 1/(dy*dy))
    return PDEResult2D(x, y, t_points, u_history, 'wave_2d_explicit', dx, dy, dt,
                       metadata={'cfl_2d': cfl_2d, 'stable': cfl_2d <= 1.0, 'c': c})


# ---------------------------------------------------------------------------
# Elliptic: Laplace/Poisson equation -nabla^2 u = f(x,y)
# ---------------------------------------------------------------------------

def poisson_2d_direct(x_span, y_span, nx, ny, source, bc_x_left, bc_x_right,
                      bc_y_bottom, bc_y_top):
    """Solve 2D Poisson equation using direct matrix solve (LU decomposition).

    -u_xx - u_yy = f(x,y) with Dirichlet BCs.
    Uses C132 lu_solve for the linear system.
    """
    x, y, dx, dy = make_grid_2d(x_span, y_span, nx, ny)

    # Interior grid: (nx-1) * (ny-1) unknowns
    m = nx - 1
    n = ny - 1
    N = m * n

    if N == 0:
        u = [[0.0] * (ny + 1) for _ in range(nx + 1)]
        _apply_bc_2d_static(u, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                            x, y, dx, dy, nx, ny)
        return EllipticResult(x, y, u, 'poisson_2d_direct', dx, dy)

    # Map (i,j) interior -> linear index k
    def idx(i, j):
        return (i - 1) * n + (j - 1)

    # Build system Au = b
    A_data = [[0.0] * N for _ in range(N)]
    b = [0.0] * N

    cx = 1.0 / (dx * dx)
    cy = 1.0 / (dy * dy)
    diag = 2 * cx + 2 * cy

    for i in range(1, nx):
        for j in range(1, ny):
            k = idx(i, j)
            A_data[k][k] = diag

            # x neighbors
            if i > 1:
                A_data[k][idx(i-1, j)] = -cx
            else:
                b[k] += cx * _bc_value(bc_x_left)

            if i < nx - 1:
                A_data[k][idx(i+1, j)] = -cx
            else:
                b[k] += cx * _bc_value(bc_x_right)

            # y neighbors
            if j > 1:
                A_data[k][idx(i, j-1)] = -cy
            else:
                b[k] += cy * _bc_value(bc_y_bottom)

            if j < ny - 1:
                A_data[k][idx(i, j+1)] = -cy
            else:
                b[k] += cy * _bc_value(bc_y_top)

            # Source term
            if callable(source):
                b[k] += source(x[i], y[j])
            else:
                b[k] += source

    A_mat = Matrix(A_data)
    u_vec = lu_solve(A_mat, b)

    # Unpack into 2D grid
    u = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(1, nx):
        for j in range(1, ny):
            u[i][j] = u_vec[idx(i, j)]

    _apply_bc_2d_static(u, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                        x, y, dx, dy, nx, ny)

    return EllipticResult(x, y, u, 'poisson_2d_direct', dx, dy,
                          metadata={'system_size': N})


def laplace_2d_iterative(x_span, y_span, nx, ny, bc_x_left, bc_x_right,
                         bc_y_bottom, bc_y_top, tol=1e-6, max_iter=10000,
                         method='gauss_seidel'):
    """Solve 2D Laplace equation iteratively (Jacobi or Gauss-Seidel).

    -u_xx - u_yy = 0 with Dirichlet BCs.
    """
    x, y, dx, dy = make_grid_2d(x_span, y_span, nx, ny)

    u = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    _apply_bc_2d_static(u, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                        x, y, dx, dy, nx, ny)

    cx = 1.0 / (dx * dx)
    cy = 1.0 / (dy * dy)
    denom = 2 * (cx + cy)

    iterations = 0
    residual = float('inf')

    for it in range(max_iter):
        max_diff = 0.0

        if method == 'jacobi':
            u_new = _copy_2d(u)
            for i in range(1, nx):
                for j in range(1, ny):
                    val = (cx * (u[i+1][j] + u[i-1][j]) +
                           cy * (u[i][j+1] + u[i][j-1])) / denom
                    max_diff = max(max_diff, abs(val - u[i][j]))
                    u_new[i][j] = val
            u = u_new
        else:  # gauss_seidel
            for i in range(1, nx):
                for j in range(1, ny):
                    val = (cx * (u[i+1][j] + u[i-1][j]) +
                           cy * (u[i][j+1] + u[i][j-1])) / denom
                    max_diff = max(max_diff, abs(val - u[i][j]))
                    u[i][j] = val

        iterations = it + 1
        residual = max_diff

        if max_diff < tol:
            break

    return EllipticResult(x, y, u, f'laplace_2d_{method}', dx, dy,
                          iterations=iterations, residual=residual)


def poisson_2d_sor(x_span, y_span, nx, ny, source, bc_x_left, bc_x_right,
                   bc_y_bottom, bc_y_top, omega=None, tol=1e-6, max_iter=10000):
    """Solve 2D Poisson equation using SOR (Successive Over-Relaxation).

    If omega not specified, uses optimal omega for Laplace on rectangle.
    """
    x, y, dx, dy = make_grid_2d(x_span, y_span, nx, ny)

    if omega is None:
        # Optimal omega for Laplace on rectangle
        rho = math.cos(math.pi / nx)
        omega = 2.0 / (1.0 + math.sqrt(1.0 - rho * rho))

    u = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    _apply_bc_2d_static(u, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                        x, y, dx, dy, nx, ny)

    cx = 1.0 / (dx * dx)
    cy = 1.0 / (dy * dy)
    denom = 2 * (cx + cy)

    iterations = 0
    residual = float('inf')

    for it in range(max_iter):
        max_diff = 0.0

        for i in range(1, nx):
            for j in range(1, ny):
                s = source(x[i], y[j]) if callable(source) else source
                gs_val = (cx * (u[i+1][j] + u[i-1][j]) +
                          cy * (u[i][j+1] + u[i][j-1]) + s) / denom
                new_val = (1 - omega) * u[i][j] + omega * gs_val
                max_diff = max(max_diff, abs(new_val - u[i][j]))
                u[i][j] = new_val

        iterations = it + 1
        residual = max_diff

        if max_diff < tol:
            break

    return EllipticResult(x, y, u, 'poisson_2d_sor', dx, dy,
                          iterations=iterations, residual=residual,
                          metadata={'omega': omega})


# ---------------------------------------------------------------------------
# Method of Lines (MOL): reduce PDE to ODE system, solve with C130
# ---------------------------------------------------------------------------

def heat_1d_mol(alpha, x_span, t_span, nx, initial, bc_left, bc_right,
                source=None, ode_method='rk4', ode_steps=None):
    """Solve 1D heat equation using Method of Lines + C130 ODE solver.

    Spatial discretization -> ODE system, then use RK4 or other ODE solver.
    """
    x, dx = make_grid_1d(x_span, nx)
    t0, tf = t_span

    if ode_steps is None:
        # Choose steps based on stability (use a safe CFL-like criterion)
        dt_safe = 0.4 * dx * dx / alpha
        ode_steps = max(100, int((tf - t0) / dt_safe) + 1)

    # Initial condition for interior points
    if callable(initial):
        y0 = [initial(x[i]) for i in range(1, nx)]
    else:
        y0 = list(initial[1:nx])

    # RHS: du/dt = alpha * (u_{i-1} - 2*u_i + u_{i+1}) / dx^2
    def rhs(t, y):
        dydt = [0.0] * len(y)
        left_val = _bc_value(bc_left, t)
        right_val = _bc_value(bc_right, t)

        for i in range(len(y)):
            u_left = y[i-1] if i > 0 else left_val
            u_right = y[i+1] if i < len(y) - 1 else right_val
            s = source(x[i+1], t) if source else 0.0
            dydt[i] = alpha * (u_left - 2*y[i] + u_right) / (dx*dx) + s

        return dydt

    result = solve_ode(rhs, (t0, tf), y0, method=ode_method, n_steps=ode_steps)

    # Reconstruct full solution with boundaries
    t_points = result.t
    u_history = []
    for k, t_val in enumerate(t_points):
        left_val = _bc_value(bc_left, t_val)
        right_val = _bc_value(bc_right, t_val)
        u_full = [left_val] + list(result.y[k]) + [right_val]
        u_history.append(u_full)

    dt = (tf - t0) / ode_steps
    return PDEResult(x, t_points, u_history, 'heat_1d_mol', dx, dt,
                     metadata={'ode_method': ode_method, 'ode_steps': ode_steps,
                               'alpha': alpha})


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------

def compute_error(numerical, analytical_fn, x, t=None):
    """Compute L2 and Linf error between numerical solution and analytical.

    Args:
        numerical: list of values at grid points
        analytical_fn: callable(x) or callable(x, t)
        x: grid points
        t: time (if analytical depends on t)
    """
    n = len(numerical)
    l2_sum = 0.0
    linf = 0.0
    for i in range(n):
        if t is not None:
            exact = analytical_fn(x[i], t)
        else:
            exact = analytical_fn(x[i])
        err = abs(numerical[i] - exact)
        l2_sum += err * err
        linf = max(linf, err)
    l2 = math.sqrt(l2_sum / n)
    return {'l2': l2, 'linf': linf}


def convergence_study_1d(solver_fn, analytical_fn, nx_values, t_final,
                         **solver_kwargs):
    """Run convergence study: solve at multiple resolutions, compute errors and orders.

    Returns list of (nx, l2_error, linf_error, l2_order, linf_order).
    """
    results = []
    prev_l2 = None
    prev_linf = None
    prev_dx = None

    for nx in nx_values:
        # Scale nt with nx^2 for parabolic stability
        nt = solver_kwargs.get('nt', nx * nx)
        kw = dict(solver_kwargs)
        kw['nx'] = nx
        kw['nt'] = nt

        result = solver_fn(**kw)

        errors = compute_error(result.final, analytical_fn, result.x, t=t_final)
        l2_err = errors['l2']
        linf_err = errors['linf']

        l2_order = 0.0
        linf_order = 0.0
        if prev_l2 is not None and l2_err > 0 and prev_l2 > 0:
            l2_order = math.log(prev_l2 / l2_err) / math.log(prev_dx / result.dx)
            linf_order = math.log(prev_linf / linf_err) / math.log(prev_dx / result.dx)

        results.append({
            'nx': nx,
            'dx': result.dx,
            'l2': l2_err,
            'linf': linf_err,
            'l2_order': l2_order,
            'linf_order': linf_order,
        })

        prev_l2 = l2_err
        prev_linf = linf_err
        prev_dx = result.dx

    return results


def energy_1d(u, dx):
    """Compute L2 energy integral: integral(u^2 dx)."""
    e = 0.0
    n = len(u)
    for i in range(n):
        w = dx if 0 < i < n - 1 else dx / 2
        e += u[i] * u[i] * w
    return e


def max_norm(u):
    """Compute max norm of a 1D solution."""
    return max(abs(v) for v in u)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _solve_tridiagonal(a, b, c, d):
    """Solve tridiagonal system using Thomas algorithm.

    a: sub-diagonal (scalar or list)
    b: diagonal (scalar or list)
    c: super-diagonal (scalar or list)
    d: right-hand side (list)

    Returns solution x.
    """
    n = len(d)
    if n == 0:
        return []
    if n == 1:
        bval = b if isinstance(b, (int, float)) else b[0]
        return [d[0] / bval]

    # Convert scalars to lists
    if isinstance(a, (int, float)):
        a = [a] * n
    if isinstance(b, (int, float)):
        b = [b] * n
    if isinstance(c, (int, float)):
        c = [c] * n

    # Forward sweep
    c_prime = [0.0] * n
    d_prime = [0.0] * n

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        m = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / m if i < n - 1 else 0.0
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m

    # Back substitution
    x = [0.0] * n
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


def _apply_bc_1d(u, bc_left, bc_right, dx, t, nx):
    """Apply boundary conditions to 1D solution."""
    if bc_left.bc_type == BoundaryCondition.DIRICHLET:
        u[0] = _bc_value(bc_left, t)
    elif bc_left.bc_type == BoundaryCondition.NEUMANN:
        flux = _bc_value(bc_left, t)
        u[0] = u[1] - dx * flux
    elif bc_left.bc_type == BoundaryCondition.PERIODIC:
        u[0] = u[nx]

    if bc_right.bc_type == BoundaryCondition.DIRICHLET:
        u[nx] = _bc_value(bc_right, t)
    elif bc_right.bc_type == BoundaryCondition.NEUMANN:
        flux = _bc_value(bc_right, t)
        u[nx] = u[nx-1] + dx * flux
    elif bc_right.bc_type == BoundaryCondition.PERIODIC:
        u[nx] = u[0]


def _get_bc_dirichlet_value(bc, u, dx, t, side):
    """Get effective Dirichlet value for implicit solvers."""
    if bc.bc_type == BoundaryCondition.DIRICHLET:
        return _bc_value(bc, t)
    elif bc.bc_type == BoundaryCondition.NEUMANN:
        # For Neumann, return neighbor estimate (will be corrected after)
        flux = _bc_value(bc, t)
        if side == 'left':
            return u[1] - dx * flux
        else:
            return u[-2] + dx * flux
    return 0.0


def _apply_bc_2d(u, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                 x, y, dx, dy, nx, ny, t):
    """Apply boundary conditions to 2D solution."""
    left_val = _bc_value(bc_x_left, t)
    right_val = _bc_value(bc_x_right, t)
    bottom_val = _bc_value(bc_y_bottom, t)
    top_val = _bc_value(bc_y_top, t)

    for j in range(ny + 1):
        u[0][j] = left_val
        u[nx][j] = right_val
    for i in range(nx + 1):
        u[i][0] = bottom_val
        u[i][ny] = top_val


def _apply_bc_2d_static(u, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                        x, y, dx, dy, nx, ny):
    """Apply boundary conditions (no time dependence)."""
    _apply_bc_2d(u, bc_x_left, bc_x_right, bc_y_bottom, bc_y_top,
                 x, y, dx, dy, nx, ny, 0.0)


def _copy_2d(u):
    """Deep copy 2D array."""
    return [list(row) for row in u]
