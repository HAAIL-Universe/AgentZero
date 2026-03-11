"""
C137: ODE Solvers
Ordinary Differential Equation solvers composing C132 Linear Algebra.

Explicit: Euler, Midpoint, RK4, RK45 (Dormand-Prince) with adaptive stepping.
Implicit: Backward Euler, Trapezoidal (Crank-Nicolson), BDF (orders 1-5).
Multi-step: Adams-Bashforth (orders 1-5), Adams-Moulton (orders 1-4).
Features: Event detection, dense output, stiffness detection, solve_ivp API.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C132_linear_algebra'))
from linear_algebra import Matrix, lu_solve


# ============================================================
# ODE Result
# ============================================================

class ODEResult:
    """Result of an ODE integration."""

    def __init__(self, t, y, success=True, message="", nfev=0, njev=0,
                 nlu=0, t_events=None, y_events=None):
        self.t = t          # list of time points
        self.y = y          # list of state vectors (each is a list)
        self.success = success
        self.message = message
        self.nfev = nfev    # number of function evaluations
        self.njev = njev    # number of Jacobian evaluations
        self.nlu = nlu      # number of LU decompositions
        self.t_events = t_events or []
        self.y_events = y_events or []

    def __repr__(self):
        return (f"ODEResult(success={self.success}, points={len(self.t)}, "
                f"nfev={self.nfev}, message='{self.message}')")


# ============================================================
# Helpers
# ============================================================

def _norm(v):
    """Euclidean norm of a vector (list)."""
    return math.sqrt(sum(x * x for x in v))


def _vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_scale(s, v):
    return [s * vi for vi in v]


def _vec_axpy(a, x, y):
    """y + a*x"""
    return [yi + a * xi for xi, yi in zip(x, y)]


def _weighted_norm(v, atol, rtol, y_ref):
    """Weighted RMS norm for error control."""
    n = len(v)
    s = 0.0
    for i in range(n):
        sc = atol + rtol * abs(y_ref[i])
        s += (v[i] / sc) ** 2
    return math.sqrt(s / n)


def _finite_diff_jacobian(f, t, y, fty, eps=1e-8):
    """Compute Jacobian df/dy by finite differences."""
    n = len(y)
    J = Matrix.zeros(n, n)
    for j in range(n):
        y_pert = y[:]
        h = eps * max(abs(y[j]), 1.0)
        y_pert[j] += h
        f_pert = f(t, y_pert)
        for i in range(n):
            J[i, j] = (f_pert[i] - fty[i]) / h
    return J


# ============================================================
# Explicit Methods
# ============================================================

def euler_step(f, t, y, h):
    """Forward Euler: y_{n+1} = y_n + h * f(t_n, y_n)."""
    k = f(t, y)
    return _vec_axpy(h, k, y), 1


def midpoint_step(f, t, y, h):
    """Explicit midpoint (RK2): y_{n+1} = y_n + h * f(t+h/2, y+h/2*k1)."""
    k1 = f(t, y)
    y_mid = _vec_axpy(h / 2, k1, y)
    k2 = f(t + h / 2, y_mid)
    return _vec_axpy(h, k2, y), 2


def rk4_step(f, t, y, h):
    """Classic RK4."""
    k1 = f(t, y)
    k2 = f(t + h / 2, _vec_axpy(h / 2, k1, y))
    k3 = f(t + h / 2, _vec_axpy(h / 2, k2, y))
    k4 = f(t + h, _vec_axpy(h, k3, y))
    n = len(y)
    y_new = [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
    return y_new, 4


# ============================================================
# RK45 Dormand-Prince (adaptive)
# ============================================================

# Dormand-Prince coefficients
_DP_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
]

# 5th order weights (for solution)
_DP_B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# 4th order weights (for error estimate)
_DP_E = [71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]


def rk45_step(f, t, y, h):
    """Dormand-Prince RK45 step. Returns (y5, y4, k, nfev)."""
    n = len(y)
    k = [None] * 7
    k[0] = f(t, y)

    for s in range(1, 7):
        yy = y[:]
        for i in range(n):
            acc = 0.0
            for j in range(s):
                acc += _DP_A[s][j] * k[j][i]
            yy[i] = y[i] + h * acc
        ts = t + h * sum(_DP_A[s])
        k[s] = f(ts, yy)

    # 5th order solution
    y5 = y[:]
    for i in range(n):
        for s in range(7):
            y5[i] += h * _DP_B[s] * k[s][i]

    # Error estimate (difference between 5th and 4th order)
    err = [0.0] * n
    for i in range(n):
        for s in range(7):
            err[i] += h * _DP_E[s] * k[s][i]

    return y5, err, k, 6


def solve_rk45(f, t_span, y0, rtol=1e-6, atol=1e-9, max_step=None,
               h0=None, max_nfev=100000, events=None, dense=False):
    """Adaptive RK45 (Dormand-Prince) solver."""
    t0, tf = t_span
    n = len(y0)
    direction = 1.0 if tf > t0 else -1.0

    if max_step is None:
        max_step = abs(tf - t0) / 10

    # Initial step size estimate
    if h0 is None:
        f0 = f(t0, y0)
        d0 = _norm(y0) / max(n, 1) ** 0.5
        d1 = _norm(f0) / max(n, 1) ** 0.5
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1
        h0 = min(h0, max_step)
    else:
        h0 = abs(h0)

    h = h0 * direction
    t = t0
    y = y0[:]
    ts = [t0]
    ys = [y0[:]]
    nfev = 1  # from h0 estimate
    t_events_list = []
    y_events_list = []

    # For event detection
    if events:
        if callable(events):
            events = [events]
        prev_event_vals = [ev(t, y) for ev in events]
        t_events_list = [[] for _ in events]
        y_events_list = [[] for _ in events]

    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0

    while direction * (t - tf) < 0:
        # Don't overshoot
        if direction * (t + h - tf) > 0:
            h = tf - t

        y5, err, k, nf = rk45_step(f, t, y, h)
        nfev += nf

        # Error norm
        err_norm = _weighted_norm(err, atol, rtol, y)

        if err_norm <= 1.0:
            # Accept step
            t_old, y_old = t, y[:]
            t = t + h
            y = y5

            ts.append(t)
            ys.append(y[:])

            # Event detection
            if events:
                for ei, ev in enumerate(events):
                    val = ev(t, y)
                    if prev_event_vals[ei] * val < 0:
                        # Sign change -- bisect to find event time
                        te = _bisect_event(ev, f, t_old, y_old, t, y, rk4_step)
                        ye = _interp_linear(t_old, y_old, t, y, te)
                        t_events_list[ei].append(te)
                        y_events_list[ei].append(ye)
                    prev_event_vals[ei] = val

            # Step size adjustment
            if err_norm == 0:
                factor = max_factor
            else:
                factor = safety * err_norm ** (-0.2)
                factor = max(min_factor, min(factor, max_factor))
            h = h * factor
            if abs(h) > max_step:
                h = max_step * direction

        else:
            # Reject step
            factor = safety * err_norm ** (-0.2)
            factor = max(min_factor, factor)
            h = h * factor

        if nfev > max_nfev:
            return ODEResult(ts, ys, success=False,
                             message="Max function evaluations exceeded",
                             nfev=nfev, t_events=t_events_list,
                             y_events=y_events_list)

    return ODEResult(ts, ys, success=True, message="Success",
                     nfev=nfev, t_events=t_events_list,
                     y_events=y_events_list)


def _bisect_event(ev, f, t0, y0, t1, y1, step_fn, tol=1e-10, max_iter=50):
    """Find event time by bisection."""
    for _ in range(max_iter):
        tm = (t0 + t1) / 2
        if abs(t1 - t0) < tol:
            return tm
        ym = _interp_linear(t0, y0, t1, y1, tm)
        vm = ev(tm, ym)
        v0 = ev(t0, y0)
        if v0 * vm <= 0:
            t1 = tm
        else:
            t0 = tm
    return (t0 + t1) / 2


def _interp_linear(t0, y0, t1, y1, t):
    """Linear interpolation between two states."""
    if abs(t1 - t0) < 1e-15:
        return y0[:]
    s = (t - t0) / (t1 - t0)
    return [y0[i] + s * (y1[i] - y0[i]) for i in range(len(y0))]


# ============================================================
# Fixed-step solver
# ============================================================

def solve_fixed(f, t_span, y0, h=None, n_steps=None, method='rk4'):
    """Fixed-step solver using Euler, midpoint, or RK4."""
    t0, tf = t_span
    if n_steps is not None:
        h = (tf - t0) / n_steps
    elif h is None:
        h = (tf - t0) / 100

    step_fns = {
        'euler': euler_step,
        'midpoint': midpoint_step,
        'rk4': rk4_step,
    }
    step_fn = step_fns[method]

    t = t0
    y = y0[:]
    ts = [t]
    ys = [y[:]]
    nfev = 0
    direction = 1.0 if tf > t0 else -1.0
    h = abs(h) * direction

    while direction * (t - tf) < -1e-12 * abs(tf):
        if direction * (t + h - tf) > 1e-12 * abs(tf):
            h = tf - t
        y, nf = step_fn(f, t, y, h)
        nfev += nf
        t = t + h
        ts.append(t)
        ys.append(y[:])

    return ODEResult(ts, ys, success=True, message="Success", nfev=nfev)


# ============================================================
# Implicit Methods (composing C132 for linear solves)
# ============================================================

def backward_euler_step(f, t, y, h, jac=None, tol=1e-10, max_iter=20):
    """Backward Euler: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1}).
    Solved by Newton iteration."""
    n = len(y)
    nfev = 0
    njev = 0
    nlu = 0

    # Initial guess: forward Euler
    f0 = f(t, y)
    nfev += 1
    y_new = _vec_axpy(h, f0, y)

    t_new = t + h

    for _ in range(max_iter):
        fval = f(t_new, y_new)
        nfev += 1
        # Residual: G(y_new) = y_new - y - h*f(t_new, y_new) = 0
        residual = [y_new[i] - y[i] - h * fval[i] for i in range(n)]

        if _norm(residual) < tol:
            return y_new, nfev, njev, nlu

        # Jacobian: dG/dy = I - h * J
        if jac is not None:
            J = jac(t_new, y_new)
            njev += 1
        else:
            J = _finite_diff_jacobian(f, t_new, y_new, fval)
            njev += 1
            nfev += n

        # dG/dy = I - h*J
        A = Matrix.identity(n)
        for i in range(n):
            for j in range(n):
                A[i, j] -= h * J[i, j]

        b = [-r for r in residual]
        nlu += 1
        delta = lu_solve(A, b)
        y_new = _vec_add(y_new, delta)

    return y_new, nfev, njev, nlu


def trapezoidal_step(f, t, y, h, jac=None, tol=1e-10, max_iter=20):
    """Trapezoidal (Crank-Nicolson): y_{n+1} = y_n + h/2*(f_n + f_{n+1}).
    Solved by Newton iteration."""
    n = len(y)
    nfev = 0
    njev = 0
    nlu = 0

    f0 = f(t, y)
    nfev += 1
    y_new = _vec_axpy(h, f0, y)  # initial guess
    t_new = t + h

    for _ in range(max_iter):
        fval = f(t_new, y_new)
        nfev += 1
        residual = [y_new[i] - y[i] - h / 2 * (f0[i] + fval[i]) for i in range(n)]

        if _norm(residual) < tol:
            return y_new, nfev, njev, nlu

        if jac is not None:
            J = jac(t_new, y_new)
            njev += 1
        else:
            J = _finite_diff_jacobian(f, t_new, y_new, fval)
            njev += 1
            nfev += n

        A = Matrix.identity(n)
        for i in range(n):
            for j in range(n):
                A[i, j] -= h / 2 * J[i, j]

        b = [-r for r in residual]
        nlu += 1
        delta = lu_solve(A, b)
        y_new = _vec_add(y_new, delta)

    return y_new, nfev, njev, nlu


def solve_implicit(f, t_span, y0, h=None, n_steps=None, method='trapz',
                   jac=None, tol=1e-10):
    """Fixed-step implicit solver (backward_euler or trapz)."""
    t0, tf = t_span
    if n_steps is not None:
        h = (tf - t0) / n_steps
    elif h is None:
        h = (tf - t0) / 100

    step_fns = {
        'backward_euler': backward_euler_step,
        'trapz': trapezoidal_step,
    }
    step_fn = step_fns[method]

    t = t0
    y = y0[:]
    ts = [t]
    ys = [y[:]]
    nfev = 0
    njev = 0
    nlu = 0

    direction = 1.0 if tf > t0 else -1.0
    h = abs(h) * direction

    while direction * (t - tf) < -1e-12 * abs(tf):
        if direction * (t + h - tf) > 1e-12 * abs(tf):
            h = tf - t
        y, nf, nj, nl = step_fn(f, t, y, h, jac=jac, tol=tol)
        nfev += nf
        njev += nj
        nlu += nl
        t = t + h
        ts.append(t)
        ys.append(y[:])

    return ODEResult(ts, ys, success=True, message="Success",
                     nfev=nfev, njev=njev, nlu=nlu)


# ============================================================
# BDF Methods (Backward Differentiation Formulas)
# ============================================================

# BDF coefficients: sum(alpha_i * y_{n-i}) = h * beta * f_{n+1}
# alpha_0 * y_{n+1} + alpha_1 * y_n + ... = h * beta * f_{n+1}
_BDF_COEFFS = {
    1: {'alpha': [1, -1], 'beta': 1},
    2: {'alpha': [3/2, -2, 1/2], 'beta': 1},
    3: {'alpha': [11/6, -3, 3/2, -1/3], 'beta': 1},
    4: {'alpha': [25/12, -4, 3, -4/3, 1/4], 'beta': 1},
    5: {'alpha': [137/60, -5, 5, -10/3, 5/4, -1/5], 'beta': 1},
}


def bdf_step(f, t_new, y_history, h, order, jac=None, tol=1e-10, max_iter=20):
    """Single BDF step of given order.
    y_history = [y_n, y_{n-1}, ...] (most recent first)."""
    n = len(y_history[0])
    coeffs = _BDF_COEFFS[order]
    alpha = coeffs['alpha']
    beta = coeffs['beta']

    nfev = 0
    njev = 0
    nlu = 0

    # RHS: -sum(alpha[i+1] * y_{n-i}) for i=0..order-1
    rhs = [0.0] * n
    for k in range(1, order + 1):
        for i in range(n):
            rhs[i] -= alpha[k] * y_history[k - 1][i]

    # Initial guess: extrapolation from y_n
    y_new = y_history[0][:]

    for _ in range(max_iter):
        fval = f(t_new, y_new)
        nfev += 1

        # Residual: alpha[0]*y_new - rhs - h*beta*f(t_new, y_new)
        residual = [alpha[0] * y_new[i] - rhs[i] - h * beta * fval[i]
                     for i in range(n)]

        if _norm(residual) < tol:
            return y_new, nfev, njev, nlu

        if jac is not None:
            J = jac(t_new, y_new)
            njev += 1
        else:
            J = _finite_diff_jacobian(f, t_new, y_new, fval)
            njev += 1
            nfev += n

        # dG/dy = alpha[0]*I - h*beta*J
        A = Matrix.zeros(n, n)
        for i in range(n):
            A[i, i] = alpha[0]
            for j in range(n):
                A[i, j] -= h * beta * J[i, j]

        b_vec = [-r for r in residual]
        nlu += 1
        delta = lu_solve(A, b_vec)
        y_new = _vec_add(y_new, delta)

    return y_new, nfev, njev, nlu


def solve_bdf(f, t_span, y0, h=None, n_steps=None, order=2, jac=None, tol=1e-10):
    """BDF solver of given order (1-5). Uses lower-order BDF to bootstrap."""
    t0, tf = t_span
    if n_steps is not None:
        h = (tf - t0) / n_steps
    elif h is None:
        h = (tf - t0) / 100

    n = len(y0)
    t = t0
    y_history = [y0[:]]  # most recent first
    ts = [t0]
    ys = [y0[:]]
    nfev = 0
    njev = 0
    nlu = 0

    direction = 1.0 if tf > t0 else -1.0
    h = abs(h) * direction
    step_count = 0

    while direction * (t - tf) < -1e-12 * abs(tf):
        if direction * (t + h - tf) > 1e-12 * abs(tf):
            h = tf - t

        # Use min(current order, available history)
        cur_order = min(order, len(y_history))
        t_new = t + h

        y_new, nf, nj, nl = bdf_step(f, t_new, y_history, h, cur_order,
                                       jac=jac, tol=tol)
        nfev += nf
        njev += nj
        nlu += nl

        # Update history
        y_history.insert(0, y_new[:])
        if len(y_history) > order + 1:
            y_history.pop()

        t = t_new
        ts.append(t)
        ys.append(y_new[:])
        step_count += 1

    return ODEResult(ts, ys, success=True, message="Success",
                     nfev=nfev, njev=njev, nlu=nlu)


# ============================================================
# Adams-Bashforth (Explicit Multi-step)
# ============================================================

_AB_COEFFS = {
    1: [1],
    2: [3/2, -1/2],
    3: [23/12, -16/12, 5/12],
    4: [55/24, -59/24, 37/24, -9/24],
    5: [1901/720, -2774/720, 2616/720, -1274/720, 251/720],
}


def solve_adams_bashforth(f, t_span, y0, h=None, n_steps=None, order=4):
    """Adams-Bashforth explicit multi-step method.
    Uses RK4 to bootstrap the first `order` steps."""
    t0, tf = t_span
    if n_steps is not None:
        h = (tf - t0) / n_steps
    elif h is None:
        h = (tf - t0) / 100

    n = len(y0)
    t = t0
    y = y0[:]
    ts = [t]
    ys = [y[:]]
    nfev = 0

    direction = 1.0 if tf > t0 else -1.0
    h = abs(h) * direction

    # f_history: [f_n, f_{n-1}, ...] (most recent first)
    f0 = f(t, y)
    nfev += 1
    f_history = [f0]

    # Bootstrap with RK4
    for _ in range(order - 1):
        if direction * (t - tf) >= -1e-12 * abs(tf):
            break
        step_h = h
        if direction * (t + step_h - tf) > 1e-12 * abs(tf):
            step_h = tf - t
        y, nf = rk4_step(f, t, y, step_h)
        nfev += nf
        t = t + step_h
        ts.append(t)
        ys.append(y[:])
        fval = f(t, y)
        nfev += 1
        f_history.insert(0, fval)

    coeffs = _AB_COEFFS[order]

    while direction * (t - tf) < -1e-12 * abs(tf):
        step_h = h
        if direction * (t + step_h - tf) > 1e-12 * abs(tf):
            step_h = tf - t

        cur_order = min(order, len(f_history))
        c = _AB_COEFFS[cur_order]

        y_new = y[:]
        for i in range(n):
            acc = 0.0
            for k in range(cur_order):
                acc += c[k] * f_history[k][i]
            y_new[i] += step_h * acc

        t = t + step_h
        y = y_new
        ts.append(t)
        ys.append(y[:])

        fval = f(t, y)
        nfev += 1
        f_history.insert(0, fval)
        if len(f_history) > order:
            f_history.pop()

    return ODEResult(ts, ys, success=True, message="Success", nfev=nfev)


# ============================================================
# Adams-Moulton (Implicit Multi-step)
# ============================================================

_AM_COEFFS = {
    1: [1/2, 1/2],             # Trapezoidal
    2: [5/12, 8/12, -1/12],
    3: [9/24, 19/24, -5/24, 1/24],
    4: [251/720, 646/720, -264/720, 106/720, -19/720],
}


def solve_adams_moulton(f, t_span, y0, h=None, n_steps=None, order=3,
                         tol=1e-10, max_corrector=5):
    """Adams-Moulton PECE (predict-evaluate-correct-evaluate).
    Uses Adams-Bashforth as predictor, Adams-Moulton as corrector."""
    t0, tf = t_span
    if n_steps is not None:
        h = (tf - t0) / n_steps
    elif h is None:
        h = (tf - t0) / 100

    n = len(y0)
    t = t0
    y = y0[:]
    ts = [t]
    ys = [y[:]]
    nfev = 0

    direction = 1.0 if tf > t0 else -1.0
    h = abs(h) * direction

    f0 = f(t, y)
    nfev += 1
    f_history = [f0]

    # Bootstrap with RK4
    for _ in range(order - 1):
        if direction * (t - tf) >= -1e-12 * abs(tf):
            break
        step_h = h
        if direction * (t + step_h - tf) > 1e-12 * abs(tf):
            step_h = tf - t
        y, nf = rk4_step(f, t, y, step_h)
        nfev += nf
        t = t + step_h
        ts.append(t)
        ys.append(y[:])
        fval = f(t, y)
        nfev += 1
        f_history.insert(0, fval)

    ab_coeffs = _AB_COEFFS[min(order, len(_AB_COEFFS))]
    am_coeffs = _AM_COEFFS[order]

    while direction * (t - tf) < -1e-12 * abs(tf):
        step_h = h
        if direction * (t + step_h - tf) > 1e-12 * abs(tf):
            step_h = tf - t

        # Predict (Adams-Bashforth)
        cur_ab_order = min(order, len(f_history), len(_AB_COEFFS))
        ab_c = _AB_COEFFS[cur_ab_order]
        y_pred = y[:]
        for i in range(n):
            acc = 0.0
            for k in range(min(cur_ab_order, len(f_history))):
                acc += ab_c[k] * f_history[k][i]
            y_pred[i] += step_h * acc

        t_new = t + step_h

        # Evaluate
        f_pred = f(t_new, y_pred)
        nfev += 1

        # Correct (Adams-Moulton)
        cur_am_order = min(order, len(f_history))
        am_c = _AM_COEFFS[cur_am_order]

        for _ in range(max_corrector):
            y_corr = y[:]
            f_list = [f_pred] + f_history[:cur_am_order]
            for i in range(n):
                acc = 0.0
                for k in range(min(len(am_c), len(f_list))):
                    acc += am_c[k] * f_list[k][i]
                y_corr[i] += step_h * acc

            f_corr = f(t_new, y_corr)
            nfev += 1

            if _norm(_vec_sub(y_corr, y_pred)) < tol:
                y_pred = y_corr
                f_pred = f_corr
                break
            y_pred = y_corr
            f_pred = f_corr

        t = t_new
        y = y_pred
        ts.append(t)
        ys.append(y[:])

        f_history.insert(0, f_pred)
        if len(f_history) > max(order + 1, 5):
            f_history.pop()

    return ODEResult(ts, ys, success=True, message="Success", nfev=nfev)


# ============================================================
# Stiffness Detection
# ============================================================

def detect_stiffness(f, t, y, h, jac=None):
    """Estimate stiffness ratio. Returns (is_stiff, ratio).
    Uses eigenvalue estimation of the Jacobian."""
    fty = f(t, y)
    if jac is not None:
        J = jac(t, y)
    else:
        J = _finite_diff_jacobian(f, t, y, fty)

    n = len(y)
    # Power iteration for largest eigenvalue magnitude
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(20):
        # w = J * v
        w = [0.0] * n
        for i in range(n):
            for j in range(n):
                w[i] += J[i, j] * v[j]
        nw = _norm(w)
        if nw < 1e-15:
            return False, 0.0
        v = [wi / nw for wi in w]

    # Rayleigh quotient: v^T J v
    Jv = [0.0] * n
    for i in range(n):
        for j in range(n):
            Jv[i] += J[i, j] * v[j]
    lam = sum(vi * jvi for vi, jvi in zip(v, Jv))

    stiffness_ratio = abs(lam) * abs(h)
    return stiffness_ratio > 3.0, stiffness_ratio


# ============================================================
# Dense Output (Hermite interpolation)
# ============================================================

class DenseOutput:
    """Cubic Hermite interpolation between solution points."""

    def __init__(self, t_points, y_points, f_func):
        self.t = t_points
        self.y = y_points
        self.f = f_func
        self._f_cache = {}

    def _get_f(self, idx):
        if idx not in self._f_cache:
            self._f_cache[idx] = self.f(self.t[idx], self.y[idx])
        return self._f_cache[idx]

    def __call__(self, t):
        """Evaluate solution at arbitrary time t."""
        # Find bracketing interval
        if t <= self.t[0]:
            return self.y[0][:]
        if t >= self.t[-1]:
            return self.y[-1][:]

        # Binary search
        lo, hi = 0, len(self.t) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.t[mid] <= t:
                lo = mid
            else:
                hi = mid

        t0, t1 = self.t[lo], self.t[hi]
        y0, y1 = self.y[lo], self.y[hi]
        f0 = self._get_f(lo)
        f1 = self._get_f(hi)

        h = t1 - t0
        s = (t - t0) / h
        n = len(y0)

        # Cubic Hermite
        result = [0.0] * n
        for i in range(n):
            h00 = 2 * s ** 3 - 3 * s ** 2 + 1
            h10 = s ** 3 - 2 * s ** 2 + s
            h01 = -2 * s ** 3 + 3 * s ** 2
            h11 = s ** 3 - s ** 2
            result[i] = h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i]

        return result


# ============================================================
# System Constructors
# ============================================================

def to_first_order(f_higher, order):
    """Convert higher-order ODE to first-order system.
    f_higher(t, y, y', y'', ...) returns the highest derivative.
    Returns f(t, Y) where Y = [y, y', y'', ...]."""
    def f(t, Y):
        dY = [0.0] * (order)
        for i in range(order - 1):
            dY[i] = Y[i + 1]
        dY[order - 1] = f_higher(t, *Y)
        return dY
    return f


def make_system(equations):
    """Create a system from a list of equation functions.
    equations = [f1(t, y1, y2, ...), f2(t, y1, y2, ...), ...]
    Returns f(t, Y) for the ODE system."""
    def f(t, Y):
        return [eq(t, *Y) for eq in equations]
    return f


# ============================================================
# High-level API: solve_ivp
# ============================================================

def solve_ivp(f, t_span, y0, method='RK45', rtol=1e-6, atol=1e-9,
              h=None, n_steps=None, events=None, dense_output=False,
              jac=None, max_step=None, h0=None, max_nfev=100000,
              bdf_order=2, ab_order=4, am_order=3):
    """Solve an initial value problem for a system of ODEs.

    Parameters:
        f: callable f(t, y) -> dy/dt
        t_span: (t0, tf) integration interval
        y0: initial state (list)
        method: 'RK45', 'RK4', 'Euler', 'Midpoint',
                'BackwardEuler', 'Trapz', 'BDF', 'AB', 'AM'
        rtol, atol: tolerances (for adaptive methods)
        h: step size (for fixed-step methods)
        n_steps: number of steps (alternative to h)
        events: event function(s) for event detection
        dense_output: if True, result includes sol attribute
        jac: Jacobian function jac(t, y) -> Matrix (for implicit methods)
        max_step: maximum step size (for adaptive methods)

    Returns:
        ODEResult with t, y, and optional event/dense data
    """
    if isinstance(y0, (int, float)):
        y0 = [float(y0)]
        scalar = True
    else:
        y0 = [float(v) for v in y0]
        scalar = False

    method_upper = method.upper()

    if method_upper == 'RK45':
        result = solve_rk45(f, t_span, y0, rtol=rtol, atol=atol,
                            max_step=max_step, h0=h0, max_nfev=max_nfev,
                            events=events)
    elif method_upper in ('RK4', 'EULER', 'MIDPOINT'):
        name_map = {'RK4': 'rk4', 'EULER': 'euler', 'MIDPOINT': 'midpoint'}
        result = solve_fixed(f, t_span, y0, h=h, n_steps=n_steps,
                             method=name_map[method_upper])
    elif method_upper in ('BACKWARDEULER', 'BACKWARD_EULER'):
        result = solve_implicit(f, t_span, y0, h=h, n_steps=n_steps,
                                method='backward_euler', jac=jac)
    elif method_upper in ('TRAPZ', 'TRAPEZOIDAL', 'CN', 'CRANK_NICOLSON'):
        result = solve_implicit(f, t_span, y0, h=h, n_steps=n_steps,
                                method='trapz', jac=jac)
    elif method_upper == 'BDF':
        result = solve_bdf(f, t_span, y0, h=h, n_steps=n_steps,
                           order=bdf_order, jac=jac)
    elif method_upper in ('AB', 'ADAMS_BASHFORTH'):
        result = solve_adams_bashforth(f, t_span, y0, h=h, n_steps=n_steps,
                                       order=ab_order)
    elif method_upper in ('AM', 'ADAMS_MOULTON'):
        result = solve_adams_moulton(f, t_span, y0, h=h, n_steps=n_steps,
                                     order=am_order)
    else:
        raise ValueError(f"Unknown method: {method}")

    if dense_output and result.success:
        result.sol = DenseOutput(result.t, result.y, f)

    return result


# ============================================================
# Convenience: solve scalar ODE
# ============================================================

def solve_scalar(f, t_span, y0, **kwargs):
    """Solve a scalar ODE y' = f(t, y).
    f can be f(t, y) -> float. Returns result with scalar y values."""
    def f_vec(t, Y):
        return [f(t, Y[0])]

    result = solve_ivp(f_vec, t_span, [float(y0)], **kwargs)
    # Flatten y
    result.y = [yi[0] for yi in result.y]
    return result
