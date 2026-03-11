"""
C138: Numerical Optimization Library
Composing C128 (Automatic Differentiation) + C132 (Linear Algebra)

Unconstrained optimization:
  - Steepest descent (with line search)
  - Newton's method (exact Hessian)
  - BFGS (quasi-Newton, rank-2 Hessian updates)
  - L-BFGS (limited-memory BFGS)
  - Conjugate gradient (Fletcher-Reeves, Polak-Ribiere)
  - Nelder-Mead (derivative-free simplex)
  - Trust region (Cauchy point + dogleg)

Constrained optimization:
  - Projected gradient descent (box constraints)
  - Augmented Lagrangian (equality + inequality)
  - Sequential Quadratic Programming (SQP)
  - Penalty method

Line search:
  - Backtracking (Armijo)
  - Wolfe conditions (strong Wolfe)
  - Golden section (exact 1D)

Utilities:
  - Numerical gradient / Hessian
  - Finite difference Jacobian
  - Convergence diagnostics
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C128_automatic_differentiation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C132_linear_algebra'))

from autodiff import ReverseAD, ForwardAD, hessian as ad_hessian, value_and_grad as ad_value_and_grad
from linear_algebra import (
    Matrix, lu_solve, cholesky, cholesky_solve, conjugate_gradient as cg_solve,
    dot, vec_norm, vec_add, vec_sub, vec_scale, eigenvalues, is_positive_definite
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class OptimizeResult:
    """Result of an optimization run."""
    __slots__ = ('x', 'fun', 'grad', 'nit', 'nfev', 'ngev', 'success',
                 'message', 'history', 'x_history', 'pcov')

    def __init__(self, x, fun, grad=None, nit=0, nfev=0, ngev=0,
                 success=True, message='', history=None, x_history=None,
                 pcov=None):
        self.x = list(x)
        self.fun = fun
        self.grad = list(grad) if grad is not None else None
        self.nit = nit
        self.nfev = nfev
        self.ngev = ngev
        self.success = success
        self.message = message
        self.history = history or []
        self.x_history = x_history or []
        self.pcov = pcov

    def __repr__(self):
        return (f"OptimizeResult(fun={self.fun:.6g}, nit={self.nit}, "
                f"success={self.success}, message='{self.message}')")


# ---------------------------------------------------------------------------
# Numerical derivatives (fallback when AD not used)
# ---------------------------------------------------------------------------

def numerical_gradient(f, x, eps=1e-7):
    """Central difference gradient."""
    n = len(x)
    g = [0.0] * n
    for i in range(n):
        xp = list(x)
        xm = list(x)
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


def numerical_hessian(f, x, eps=1e-5):
    """Central difference Hessian."""
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp = list(x); xpm = list(x); xmp = list(x); xmm = list(x)
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            H[i][j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * eps * eps)
            H[j][i] = H[i][j]
    return H


def numerical_jacobian(f, x, eps=1e-7):
    """Central difference Jacobian for f: R^n -> R^m."""
    n = len(x)
    f0 = f(x)
    if isinstance(f0, (int, float)):
        f0 = [f0]
    m = len(f0)
    J = [[0.0] * n for _ in range(m)]
    for j in range(n):
        xp = list(x); xm = list(x)
        xp[j] += eps; xm[j] -= eps
        fp = f(xp); fm = f(xm)
        if isinstance(fp, (int, float)):
            fp = [fp]
        if isinstance(fm, (int, float)):
            fm = [fm]
        for i in range(m):
            J[i][j] = (fp[i] - fm[i]) / (2.0 * eps)
    return J


# ---------------------------------------------------------------------------
# Line search methods
# ---------------------------------------------------------------------------

def backtracking_line_search(f, x, d, grad, alpha=1.0, c1=1e-4, rho=0.5,
                              max_iter=50):
    """Armijo backtracking line search.
    Find alpha such that f(x + alpha*d) <= f(x) + c1*alpha*grad.d
    """
    f0 = f(x)
    slope = sum(g * di for g, di in zip(grad, d))
    if slope >= 0:
        # Not a descent direction -- return tiny step
        return 1e-10, f0, 1
    nfev = 0
    for _ in range(max_iter):
        xnew = [xi + alpha * di for xi, di in zip(x, d)]
        fnew = f(xnew)
        nfev += 1
        if fnew <= f0 + c1 * alpha * slope:
            return alpha, fnew, nfev
        alpha *= rho
    return alpha, f(vec_add(x, vec_scale(d, alpha))), nfev + 1


def wolfe_line_search(f, grad_f, x, d, g0=None, f0=None,
                       c1=1e-4, c2=0.9, max_iter=20, alpha_max=50.0):
    """Strong Wolfe conditions line search.
    Finds alpha satisfying:
      f(x + alpha*d) <= f(x) + c1*alpha*g.d   (sufficient decrease)
      |g(x+alpha*d).d| <= c2*|g(x).d|         (curvature)
    """
    if f0 is None:
        f0 = f(x)
    if g0 is None:
        g0 = grad_f(x)
    slope0 = sum(gi * di for gi, di in zip(g0, d))
    if slope0 >= 0:
        return 1e-10, f0, g0, 2

    nfev = 0

    def phi(a):
        nonlocal nfev
        nfev += 1
        return f(vec_add(x, vec_scale(d, a)))

    def dphi(a):
        g = grad_f(vec_add(x, vec_scale(d, a)))
        return sum(gi * di for gi, di in zip(g, d)), g

    alpha_prev = 0.0
    alpha = 1.0
    phi_prev = f0
    dphi0 = slope0

    for i in range(max_iter):
        phi_curr = phi(alpha)

        if phi_curr > f0 + c1 * alpha * dphi0 or (i > 0 and phi_curr >= phi_prev):
            return _zoom(phi, dphi, alpha_prev, alpha, f0, dphi0, c1, c2, phi_prev, phi_curr, nfev)

        dphi_curr, g_curr = dphi(alpha)
        nfev += 1

        if abs(dphi_curr) <= -c2 * dphi0:
            return alpha, phi_curr, g_curr, nfev

        if dphi_curr >= 0:
            return _zoom(phi, dphi, alpha, alpha_prev, f0, dphi0, c1, c2, phi_curr, phi_prev, nfev)

        phi_prev = phi_curr
        alpha_prev = alpha
        alpha = min(2.0 * alpha, alpha_max)

    # Fallback
    fnew = phi(alpha)
    gnew = grad_f(vec_add(x, vec_scale(d, alpha)))
    return alpha, fnew, gnew, nfev + 1


def _zoom(phi, dphi, alpha_lo, alpha_hi, f0, dphi0, c1, c2, phi_lo, phi_hi, nfev):
    """Zoom phase for Wolfe line search."""
    for _ in range(20):
        # Bisection
        alpha = 0.5 * (alpha_lo + alpha_hi)
        phi_j = phi(alpha)
        nfev += 1

        if phi_j > f0 + c1 * alpha * dphi0 or phi_j >= phi_lo:
            alpha_hi = alpha
            phi_hi = phi_j
        else:
            dphi_j, g_j = dphi(alpha)
            nfev += 1
            if abs(dphi_j) <= -c2 * dphi0:
                return alpha, phi_j, g_j, nfev
            if dphi_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
                phi_hi = phi_lo
            alpha_lo = alpha
            phi_lo = phi_j

    # Return best so far
    return alpha, phi(alpha), None, nfev + 1


def golden_section_search(f, a, b, tol=1e-8, max_iter=200):
    """Golden section search for 1D minimization on [a, b]."""
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    nfev = 0
    for _ in range(max_iter):
        fc = f(c); fd = f(d)
        nfev += 2
        if fc < fd:
            b = d
        else:
            a = c
        if abs(b - a) < tol:
            break
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    xmin = (a + b) / 2
    return xmin, f(xmin), nfev + 1


# ---------------------------------------------------------------------------
# Gradient wrapper -- uses AD or numerical fallback
# ---------------------------------------------------------------------------

def _make_grad_fn(f, n, use_ad=True):
    """Create value-and-gradient function."""
    if use_ad:
        vg = ad_value_and_grad(f, mode='reverse')
        def val_grad(x):
            v, g = vg(x)
            return float(v), [float(gi) for gi in g]
        return val_grad
    else:
        def val_grad(x):
            v = f(x)
            g = numerical_gradient(f, x)
            return v, g
        return val_grad


def _make_hessian_fn(f, n, use_ad=True):
    """Create Hessian function."""
    if use_ad:
        def hess(x):
            _, H = ad_hessian(f, x)
            return H
        return hess
    else:
        def hess(x):
            return numerical_hessian(f, x)
        return hess


# ---------------------------------------------------------------------------
# Unconstrained optimization: Steepest Descent
# ---------------------------------------------------------------------------

def steepest_descent(f, x0, grad_f=None, lr=None, max_iter=1000, tol=1e-8,
                      use_ad=True, line_search='backtracking', history=False):
    """Steepest descent with optional line search.

    Args:
        f: Objective function f(x) -> float
        x0: Initial point
        grad_f: Gradient function (optional, uses AD if None)
        lr: Fixed learning rate (if None, uses line search)
        max_iter: Maximum iterations
        tol: Gradient norm tolerance
        use_ad: Use automatic differentiation for gradients
        line_search: 'backtracking', 'wolfe', or None (fixed lr)
        history: Record function value history
    """
    n = len(x0)
    x = list(x0)

    if grad_f is not None:
        def val_grad(x):
            return f(x), grad_f(x)
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    hist = []
    x_hist = []
    nfev = 0
    ngev = 0

    for it in range(max_iter):
        fval, g = val_grad(x)
        nfev += 1; ngev += 1
        if history:
            hist.append(fval)
            x_hist.append(list(x))

        gnorm = vec_norm(g)
        if gnorm < tol:
            return OptimizeResult(x, fval, g, it, nfev, ngev, True,
                                   'Converged: gradient norm below tolerance',
                                   hist, x_hist)

        d = vec_scale(g, -1.0)

        if lr is not None and line_search is None:
            alpha = lr
            x = vec_add(x, vec_scale(d, alpha))
            nfev += 1
        elif line_search == 'wolfe':
            def gf(xp):
                _, gp = val_grad(xp)
                return gp
            alpha, fnew, gnew, nf = wolfe_line_search(f, gf, x, d, g, fval)
            x = vec_add(x, vec_scale(d, alpha))
            nfev += nf
        else:
            alpha, fnew, nf = backtracking_line_search(f, x, d, g)
            x = vec_add(x, vec_scale(d, alpha))
            nfev += nf

    fval, g = val_grad(x)
    nfev += 1; ngev += 1
    return OptimizeResult(x, fval, g, max_iter, nfev, ngev, False,
                           'Maximum iterations reached', hist, x_hist)


# ---------------------------------------------------------------------------
# Newton's Method
# ---------------------------------------------------------------------------

def newton(f, x0, grad_f=None, hess_f=None, max_iter=100, tol=1e-8,
           use_ad=True, history=False, line_search='backtracking'):
    """Newton's method with exact Hessian.

    Uses H^{-1} g as the search direction, with optional line search.
    Falls back to steepest descent if Hessian is not positive definite.
    """
    n = len(x0)
    x = list(x0)

    if grad_f is not None:
        def val_grad(x):
            return f(x), grad_f(x)
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    if hess_f is None:
        hess_f = _make_hessian_fn(f, n, use_ad)

    hist = []
    x_hist = []
    nfev = 0; ngev = 0

    for it in range(max_iter):
        fval, g = val_grad(x)
        nfev += 1; ngev += 1
        if history:
            hist.append(fval)
            x_hist.append(list(x))

        gnorm = vec_norm(g)
        if gnorm < tol:
            return OptimizeResult(x, fval, g, it, nfev, ngev, True,
                                   'Converged: gradient norm below tolerance',
                                   hist, x_hist)

        H = hess_f(x)
        nfev += 1  # Hessian evaluation counts

        # Try to solve H @ d = -g
        Hmat = Matrix(H)
        b = [-gi for gi in g]
        try:
            # Try Cholesky first (faster, requires SPD)
            d = cholesky_solve(Hmat, b)
        except (ValueError, ZeroDivisionError):
            try:
                d = lu_solve(Hmat, b)
            except (ValueError, ZeroDivisionError):
                # Hessian is singular -- fall back to steepest descent
                d = b

        # Check descent direction
        slope = sum(di * gi for di, gi in zip(d, g))
        if slope > 0:
            # Not descent -- use negative gradient instead
            d = [-gi for gi in g]

        if line_search == 'wolfe':
            def gf(xp):
                _, gp = val_grad(xp)
                return gp
            alpha, fnew, gnew, nf = wolfe_line_search(f, gf, x, d, g, fval)
            nfev += nf
        elif line_search == 'backtracking':
            alpha, fnew, nf = backtracking_line_search(f, x, d, g)
            nfev += nf
        else:
            alpha = 1.0

        x = vec_add(x, vec_scale(d, alpha))

    fval, g = val_grad(x)
    nfev += 1; ngev += 1
    return OptimizeResult(x, fval, g, max_iter, nfev, ngev, False,
                           'Maximum iterations reached', hist, x_hist)


# ---------------------------------------------------------------------------
# BFGS (Broyden-Fletcher-Goldfarb-Shanno)
# ---------------------------------------------------------------------------

def bfgs(f, x0, grad_f=None, max_iter=200, tol=1e-8, use_ad=True,
          history=False):
    """BFGS quasi-Newton method.

    Maintains an approximate inverse Hessian using rank-2 updates.
    Uses strong Wolfe line search.
    """
    n = len(x0)
    x = list(x0)

    if grad_f is not None:
        def val_grad(x):
            return f(x), grad_f(x)
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    fval, g = val_grad(x)
    nfev = 1; ngev = 1
    hist = []
    x_hist = []

    # Initialize inverse Hessian approximation as identity
    Hinv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for it in range(max_iter):
        if history:
            hist.append(fval)
            x_hist.append(list(x))

        gnorm = vec_norm(g)
        if gnorm < tol:
            return OptimizeResult(x, fval, g, it, nfev, ngev, True,
                                   'Converged: gradient norm below tolerance',
                                   hist, x_hist)

        # Search direction: d = -Hinv @ g
        d = [0.0] * n
        for i in range(n):
            for j in range(n):
                d[i] -= Hinv[i][j] * g[j]

        # Check descent
        slope = sum(di * gi for di, gi in zip(d, g))
        if slope > 0:
            d = [-gi for gi in g]
            # Reset Hinv to identity
            Hinv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        # Wolfe line search
        def gf(xp):
            _, gp = val_grad(xp)
            return gp
        alpha, fnew, gnew, nf = wolfe_line_search(f, gf, x, d, g, fval)
        nfev += nf

        x_new = vec_add(x, vec_scale(d, alpha))

        if gnew is None:
            fnew_val, gnew = val_grad(x_new)
            nfev += 1; ngev += 1
            fnew = fnew_val
        else:
            ngev += 1

        # BFGS update
        s = vec_sub(x_new, x)  # step
        y = vec_sub(gnew, g)   # gradient change
        sy = dot(s, y)

        if sy > 1e-12:
            rho = 1.0 / sy
            # Hinv_new = (I - rho*s*y^T) @ Hinv @ (I - rho*y*s^T) + rho*s*s^T
            # Compute V = I - rho*y*s^T
            # Hinv_new = V^T @ Hinv @ V + rho*s*s^T

            # More efficient: direct formula
            Hs = [0.0] * n
            for i in range(n):
                for j in range(n):
                    Hs[i] += Hinv[i][j] * y[j]

            yHy = dot(y, Hs)

            for i in range(n):
                for j in range(n):
                    Hinv[i][j] += rho * ((1.0 + rho * yHy) * s[i] * s[j]
                                          - Hs[i] * s[j] - s[i] * Hs[j])

        x = x_new
        fval = fnew
        g = gnew

    return OptimizeResult(x, fval, g, max_iter, nfev, ngev, False,
                           'Maximum iterations reached', hist, x_hist)


# ---------------------------------------------------------------------------
# L-BFGS (Limited-memory BFGS)
# ---------------------------------------------------------------------------

def lbfgs(f, x0, grad_f=None, m=10, max_iter=200, tol=1e-8, use_ad=True,
           history=False):
    """L-BFGS: limited-memory BFGS.

    Stores only the last m (s, y) pairs instead of the full inverse Hessian.
    Two-loop recursion for efficient Hinv @ g computation.
    """
    n = len(x0)
    x = list(x0)

    if grad_f is not None:
        def val_grad(x):
            return f(x), grad_f(x)
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    fval, g = val_grad(x)
    nfev = 1; ngev = 1
    hist = []
    x_hist = []

    ss = []  # s vectors
    ys = []  # y vectors
    rhos = []  # 1 / (y^T s)

    for it in range(max_iter):
        if history:
            hist.append(fval)
            x_hist.append(list(x))

        gnorm = vec_norm(g)
        if gnorm < tol:
            return OptimizeResult(x, fval, g, it, nfev, ngev, True,
                                   'Converged: gradient norm below tolerance',
                                   hist, x_hist)

        # Two-loop recursion to compute d = -H @ g
        q = list(g)
        k = len(ss)
        alphas = [0.0] * k

        for i in range(k - 1, -1, -1):
            alphas[i] = rhos[i] * dot(ss[i], q)
            q = vec_sub(q, vec_scale(ys[i], alphas[i]))

        # Initial Hessian scaling
        if k > 0:
            gamma = dot(ss[-1], ys[-1]) / dot(ys[-1], ys[-1])
        else:
            gamma = 1.0
        r = vec_scale(q, gamma)

        for i in range(k):
            beta = rhos[i] * dot(ys[i], r)
            r = vec_add(r, vec_scale(ss[i], alphas[i] - beta))

        d = vec_scale(r, -1.0)

        # Check descent
        slope = sum(di * gi for di, gi in zip(d, g))
        if slope > 0:
            d = [-gi for gi in g]
            ss.clear(); ys.clear(); rhos.clear()

        # Wolfe line search
        def gf(xp):
            _, gp = val_grad(xp)
            return gp
        alpha, fnew, gnew, nf = wolfe_line_search(f, gf, x, d, g, fval)
        nfev += nf

        x_new = vec_add(x, vec_scale(d, alpha))

        if gnew is None:
            fnew_val, gnew = val_grad(x_new)
            nfev += 1; ngev += 1
            fnew = fnew_val
        else:
            ngev += 1

        s = vec_sub(x_new, x)
        y = vec_sub(gnew, g)
        sy = dot(s, y)

        if sy > 1e-12:
            if len(ss) >= m:
                ss.pop(0); ys.pop(0); rhos.pop(0)
            ss.append(s)
            ys.append(y)
            rhos.append(1.0 / sy)

        x = x_new
        fval = fnew
        g = gnew

    return OptimizeResult(x, fval, g, max_iter, nfev, ngev, False,
                           'Maximum iterations reached', hist, x_hist)


# ---------------------------------------------------------------------------
# Conjugate Gradient (nonlinear)
# ---------------------------------------------------------------------------

def conjugate_gradient(f, x0, grad_f=None, method='PR+', max_iter=1000,
                        tol=1e-8, use_ad=True, history=False):
    """Nonlinear conjugate gradient method.

    Args:
        method: 'FR' (Fletcher-Reeves), 'PR' (Polak-Ribiere),
                'PR+' (PR with restart, default), 'HS' (Hestenes-Stiefel)
    """
    n = len(x0)
    x = list(x0)

    if grad_f is not None:
        def val_grad(x):
            return f(x), grad_f(x)
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    fval, g = val_grad(x)
    nfev = 1; ngev = 1
    hist = []
    x_hist = []

    d = [-gi for gi in g]
    g_prev = list(g)

    for it in range(max_iter):
        if history:
            hist.append(fval)
            x_hist.append(list(x))

        gnorm = vec_norm(g)
        if gnorm < tol:
            return OptimizeResult(x, fval, g, it, nfev, ngev, True,
                                   'Converged: gradient norm below tolerance',
                                   hist, x_hist)

        # Line search
        def gf(xp):
            _, gp = val_grad(xp)
            return gp
        alpha, fnew, gnew, nf = wolfe_line_search(f, gf, x, d, g, fval, c2=0.1)
        nfev += nf

        x_new = vec_add(x, vec_scale(d, alpha))

        if gnew is None:
            fnew_val, gnew = val_grad(x_new)
            nfev += 1; ngev += 1
            fnew = fnew_val
        else:
            ngev += 1

        g_prev = g
        g = gnew
        x = x_new
        fval = fnew

        # Compute beta
        gg_new = dot(g, g)
        gg_old = dot(g_prev, g_prev)

        if gg_old < 1e-30:
            beta = 0.0
        elif method == 'FR':
            beta = gg_new / gg_old
        elif method == 'PR':
            beta = dot(g, vec_sub(g, g_prev)) / gg_old
        elif method == 'PR+':
            beta = max(0.0, dot(g, vec_sub(g, g_prev)) / gg_old)
        elif method == 'HS':
            dy = vec_sub(g, g_prev)
            ddy = dot(d, dy)
            beta = dot(g, dy) / ddy if abs(ddy) > 1e-30 else 0.0
        else:
            beta = 0.0

        # Restart every n iterations
        if (it + 1) % n == 0:
            beta = 0.0

        d = vec_add(vec_scale(g, -1.0), vec_scale(d, beta))

    return OptimizeResult(x, fval, g, max_iter, nfev, ngev, False,
                           'Maximum iterations reached', hist, x_hist)


# ---------------------------------------------------------------------------
# Nelder-Mead (derivative-free)
# ---------------------------------------------------------------------------

def nelder_mead(f, x0, max_iter=1000, tol=1e-8, adaptive=True, history=False,
                initial_simplex=None):
    """Nelder-Mead simplex method (derivative-free).

    Args:
        adaptive: Use adaptive parameters based on dimension (Gao & Han 2012)
        initial_simplex: List of n+1 points (optional)
    """
    n = len(x0)
    nfev = 0

    # Adaptive parameters
    if adaptive:
        alpha = 1.0
        gamma = 1.0 + 2.0 / n
        rho_nm = 0.75 - 1.0 / (2.0 * n)
        sigma = 1.0 - 1.0 / n
    else:
        alpha = 1.0
        gamma = 2.0
        rho_nm = 0.5
        sigma = 0.5

    # Initialize simplex
    if initial_simplex is not None:
        simplex = [list(v) for v in initial_simplex]
    else:
        simplex = [list(x0)]
        for i in range(n):
            v = list(x0)
            v[i] += 0.05 if abs(v[i]) < 1e-8 else 0.05 * abs(v[i])
            simplex.append(v)

    # Evaluate
    fvals = [f(v) for v in simplex]
    nfev += len(simplex)

    hist = []
    x_hist = []

    for it in range(max_iter):
        # Sort
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if history:
            hist.append(fvals[0])
            x_hist.append(list(simplex[0]))

        # Convergence check: function value spread
        frange = abs(fvals[-1] - fvals[0])
        # Also check simplex size
        size = 0.0
        for i in range(1, n + 1):
            size = max(size, vec_norm(vec_sub(simplex[i], simplex[0])))

        if frange < tol and size < tol:
            return OptimizeResult(simplex[0], fvals[0], None, it, nfev, 0, True,
                                   'Converged: simplex size below tolerance',
                                   hist, x_hist)

        # Centroid of all but worst
        centroid = [0.0] * n
        for i in range(n):
            for j in range(n):
                centroid[j] += simplex[i][j]
        centroid = [c / n for c in centroid]

        # Reflection
        worst = simplex[-1]
        xr = vec_add(centroid, vec_scale(vec_sub(centroid, worst), alpha))
        fr = f(xr); nfev += 1

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
            continue

        if fr < fvals[0]:
            # Expansion
            xe = vec_add(centroid, vec_scale(vec_sub(xr, centroid), gamma))
            fe = f(xe); nfev += 1
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
            continue

        # Contraction
        if fr < fvals[-1]:
            # Outside contraction
            xc = vec_add(centroid, vec_scale(vec_sub(xr, centroid), rho_nm))
            fc = f(xc); nfev += 1
            if fc <= fr:
                simplex[-1] = xc; fvals[-1] = fc
                continue
        else:
            # Inside contraction
            xc = vec_add(centroid, vec_scale(vec_sub(worst, centroid), rho_nm))
            fc = f(xc); nfev += 1
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
                continue

        # Shrink
        for i in range(1, n + 1):
            simplex[i] = vec_add(simplex[0], vec_scale(vec_sub(simplex[i], simplex[0]), sigma))
            fvals[i] = f(simplex[i]); nfev += 1

    order = sorted(range(n + 1), key=lambda i: fvals[i])
    return OptimizeResult(simplex[order[0]], fvals[order[0]], None, max_iter, nfev, 0,
                           False, 'Maximum iterations reached', hist, x_hist)


# ---------------------------------------------------------------------------
# Trust Region (Cauchy point + Dogleg)
# ---------------------------------------------------------------------------

def trust_region(f, x0, grad_f=None, hess_f=None, max_iter=200, tol=1e-8,
                  use_ad=True, delta0=1.0, delta_max=10.0, eta=0.15,
                  history=False):
    """Trust region method with dogleg step.

    Solves a quadratic model within a trust region of radius delta.
    """
    n = len(x0)
    x = list(x0)

    if grad_f is not None:
        def val_grad(x):
            return f(x), grad_f(x)
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    if hess_f is None:
        hess_f = _make_hessian_fn(f, n, use_ad)

    fval, g = val_grad(x)
    nfev = 1; ngev = 1
    delta = delta0
    hist = []
    x_hist = []

    for it in range(max_iter):
        if history:
            hist.append(fval)
            x_hist.append(list(x))

        gnorm = vec_norm(g)
        if gnorm < tol:
            return OptimizeResult(x, fval, g, it, nfev, ngev, True,
                                   'Converged: gradient norm below tolerance',
                                   hist, x_hist)

        H = hess_f(x)
        nfev += 1

        # Compute dogleg step
        step = _dogleg(g, H, delta, n)

        # Actual vs predicted reduction
        x_new = vec_add(x, step)
        f_new = f(x_new)
        nfev += 1

        # Predicted reduction: -g^T s - 0.5 s^T H s
        Hs = _matvec(H, step, n)
        pred = -(dot(g, step) + 0.5 * dot(step, Hs))

        actual = fval - f_new

        if pred > 0:
            rho_k = actual / pred
        else:
            rho_k = 0.0

        # Update trust region
        step_norm = vec_norm(step)
        if rho_k < 0.25:
            delta = 0.25 * step_norm
        elif rho_k > 0.75 and abs(step_norm - delta) < 1e-10 * delta:
            delta = min(2.0 * delta, delta_max)

        # Accept or reject step
        if rho_k > eta:
            x = x_new
            fval = f_new
            fval2, g = val_grad(x)
            nfev += 1; ngev += 1
        # else: keep x, g unchanged

    return OptimizeResult(x, fval, g, max_iter, nfev, ngev, False,
                           'Maximum iterations reached', hist, x_hist)


def _dogleg(g, H, delta, n):
    """Compute dogleg step within trust region."""
    # Cauchy point: steepest descent step that minimizes quadratic in gradient direction
    Hg = _matvec(H, g, n)
    gHg = dot(g, Hg)
    gnorm = vec_norm(g)

    if gHg <= 0:
        # Negative curvature -- go to boundary in gradient direction
        tau = delta / gnorm
        return vec_scale(g, -tau)

    # Cauchy point
    tau_c = gnorm * gnorm / gHg
    pc = vec_scale(g, -tau_c)
    pc_norm = vec_norm(pc)

    if pc_norm >= delta:
        # Cauchy point outside trust region -- scale to boundary
        return vec_scale(g, -delta / gnorm)

    # Newton point
    Hmat = Matrix(H)
    try:
        pn = lu_solve(Hmat, [-gi for gi in g])
    except (ValueError, ZeroDivisionError):
        return vec_scale(g, -delta / gnorm)

    pn_norm = vec_norm(pn)
    if pn_norm <= delta:
        return pn

    # Dogleg: interpolate between Cauchy and Newton on boundary
    diff = vec_sub(pn, pc)
    # Solve ||pc + tau * diff||^2 = delta^2
    a = dot(diff, diff)
    b = 2.0 * dot(pc, diff)
    c = dot(pc, pc) - delta * delta
    disc = b * b - 4.0 * a * c
    if disc < 0 or a < 1e-30:
        return vec_scale(g, -delta / gnorm)
    tau = (-b + math.sqrt(disc)) / (2.0 * a)
    tau = max(0.0, min(1.0, tau))
    return vec_add(pc, vec_scale(diff, tau))


def _matvec(H, v, n):
    """Matrix-vector product H @ v where H is list-of-lists."""
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += H[i][j] * v[j]
    return result


# ---------------------------------------------------------------------------
# Constrained: Projected Gradient Descent (box constraints)
# ---------------------------------------------------------------------------

def projected_gradient(f, x0, bounds, grad_f=None, max_iter=1000, tol=1e-8,
                        use_ad=True, history=False):
    """Projected gradient descent with box constraints.

    Args:
        bounds: List of (lower, upper) tuples. Use None for unbounded.
    """
    n = len(x0)
    x = list(x0)

    if grad_f is not None:
        def val_grad(x):
            return f(x), grad_f(x)
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    # Normalize bounds
    lb = [b[0] if b[0] is not None else -1e30 for b in bounds]
    ub = [b[1] if b[1] is not None else 1e30 for b in bounds]

    def project(v):
        return [max(lb[i], min(ub[i], v[i])) for i in range(n)]

    x = project(x)
    hist = []
    x_hist = []
    nfev = 0; ngev = 0

    for it in range(max_iter):
        fval, g = val_grad(x)
        nfev += 1; ngev += 1
        if history:
            hist.append(fval)
            x_hist.append(list(x))

        # Projected gradient norm
        xg = project(vec_sub(x, g))
        pg = vec_sub(xg, x)
        pgnorm = vec_norm(pg)

        if pgnorm < tol:
            return OptimizeResult(x, fval, g, it, nfev, ngev, True,
                                   'Converged: projected gradient below tolerance',
                                   hist, x_hist)

        # Backtracking on projected step
        alpha = 1.0
        slope = dot(g, pg)
        for _ in range(50):
            x_new = project(vec_add(x, vec_scale(pg, alpha)))
            f_new = f(x_new)
            nfev += 1
            if f_new <= fval + 1e-4 * alpha * slope:
                break
            alpha *= 0.5

        x = x_new

    fval, g = val_grad(x)
    nfev += 1; ngev += 1
    return OptimizeResult(x, fval, g, max_iter, nfev, ngev, False,
                           'Maximum iterations reached', hist, x_hist)


# ---------------------------------------------------------------------------
# Augmented Lagrangian
# ---------------------------------------------------------------------------

def augmented_lagrangian(f, x0, eq_constraints=None, ineq_constraints=None,
                          max_iter=50, max_inner=200, tol=1e-6,
                          mu0=1.0, mu_factor=10.0, use_ad=True, history=False):
    """Augmented Lagrangian method for constrained optimization.

    min f(x)  subject to  h_i(x) = 0,  g_j(x) <= 0

    Args:
        eq_constraints: List of functions h_i(x) -> float (should equal 0)
        ineq_constraints: List of functions g_j(x) -> float (should be <= 0)
        mu0: Initial penalty parameter
        mu_factor: Penalty increase factor
    """
    if eq_constraints is None:
        eq_constraints = []
    if ineq_constraints is None:
        ineq_constraints = []

    n = len(x0)
    x = list(x0)
    ne = len(eq_constraints)
    ni = len(ineq_constraints)

    # Lagrange multipliers
    lam_eq = [0.0] * ne
    lam_ineq = [0.0] * ni
    mu = mu0

    hist = []
    nfev = 0

    for outer in range(max_iter):
        # Define augmented Lagrangian
        def aug_lag(xp):
            val = f(xp)
            for i, h in enumerate(eq_constraints):
                hi = h(xp)
                val += lam_eq[i] * hi + 0.5 * mu * hi * hi
            for j, g in enumerate(ineq_constraints):
                gj = g(xp)
                # max(0, lam/mu + g)
                term = lam_ineq[j] / mu + gj
                if term > 0:
                    val += 0.5 * mu * term * term
                # else: inactive constraint contributes -lam^2/(2*mu) (constant)
            return val

        # Solve unconstrained subproblem
        result = lbfgs(aug_lag, x, max_iter=max_inner, tol=tol * 0.1,
                        use_ad=use_ad)
        x = result.x
        nfev += result.nfev

        if history:
            hist.append(f(x))
            nfev += 1

        # Evaluate constraint violations
        max_viol = 0.0

        # Update multipliers
        for i, h in enumerate(eq_constraints):
            hi = h(x)
            lam_eq[i] += mu * hi
            max_viol = max(max_viol, abs(hi))

        for j, g in enumerate(ineq_constraints):
            gj = g(x)
            lam_ineq[j] = max(0.0, lam_ineq[j] + mu * gj)
            max_viol = max(max_viol, max(0.0, gj))

        if max_viol < tol:
            return OptimizeResult(x, f(x), None, outer + 1, nfev + 1, 0, True,
                                   'Converged: constraint violation below tolerance',
                                   hist, [])

        mu *= mu_factor

    fval = f(x)
    return OptimizeResult(x, fval, None, max_iter, nfev + 1, 0, False,
                           'Maximum outer iterations reached', hist, [])


# ---------------------------------------------------------------------------
# Penalty Method
# ---------------------------------------------------------------------------

def penalty_method(f, x0, eq_constraints=None, ineq_constraints=None,
                    max_iter=30, max_inner=200, tol=1e-6,
                    mu0=1.0, mu_factor=10.0, use_ad=True, history=False):
    """Quadratic penalty method.

    Simpler than augmented Lagrangian but converges slower.
    """
    if eq_constraints is None:
        eq_constraints = []
    if ineq_constraints is None:
        ineq_constraints = []

    n = len(x0)
    x = list(x0)
    mu = mu0
    hist = []
    nfev = 0

    for outer in range(max_iter):
        def penalized(xp):
            val = f(xp)
            for h in eq_constraints:
                hi = h(xp)
                val += 0.5 * mu * hi * hi
            for g in ineq_constraints:
                gj = g(xp)
                if gj > 0:
                    val += 0.5 * mu * gj * gj
            return val

        result = lbfgs(penalized, x, max_iter=max_inner, tol=tol * 0.01,
                        use_ad=use_ad)
        x = result.x
        nfev += result.nfev

        if history:
            hist.append(f(x))
            nfev += 1

        # Check violations
        max_viol = 0.0
        for h in eq_constraints:
            max_viol = max(max_viol, abs(h(x)))
        for g in ineq_constraints:
            max_viol = max(max_viol, max(0.0, g(x)))

        if max_viol < tol:
            return OptimizeResult(x, f(x), None, outer + 1, nfev + 1, 0, True,
                                   'Converged: constraint violation below tolerance',
                                   hist, [])

        mu *= mu_factor

    return OptimizeResult(x, f(x), None, max_iter, nfev + 1, 0, False,
                           'Maximum outer iterations reached', hist, [])


# ---------------------------------------------------------------------------
# SQP (Sequential Quadratic Programming)
# ---------------------------------------------------------------------------

def sqp(f, x0, eq_constraints=None, ineq_constraints=None,
         grad_f=None, max_iter=100, tol=1e-6, use_ad=True, history=False):
    """Sequential Quadratic Programming.

    At each step, solves a QP subproblem to get the search direction.
    Uses BFGS approximation for the Lagrangian Hessian.
    """
    if eq_constraints is None:
        eq_constraints = []
    if ineq_constraints is None:
        ineq_constraints = []

    n = len(x0)
    x = list(x0)
    ne = len(eq_constraints)
    ni = len(ineq_constraints)

    if grad_f is not None:
        val_grad = lambda xp: (f(xp), grad_f(xp))
    else:
        val_grad = _make_grad_fn(f, n, use_ad)

    # Lagrange multipliers
    lam_eq = [0.0] * ne
    lam_ineq = [0.0] * ni

    # BFGS Hessian approx of Lagrangian
    B = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    hist = []
    nfev = 0

    fval, g = val_grad(x)
    nfev += 1

    for it in range(max_iter):
        if history:
            hist.append(fval)

        # Evaluate constraints
        h_vals = [h(x) for h in eq_constraints]
        g_vals = [gf(x) for gf in ineq_constraints]

        # Check KKT
        max_viol = 0.0
        for hi in h_vals:
            max_viol = max(max_viol, abs(hi))
        for gi in g_vals:
            max_viol = max(max_viol, max(0.0, gi))

        gnorm = vec_norm(g)
        if gnorm < tol and max_viol < tol:
            return OptimizeResult(x, fval, g, it, nfev, 0, True,
                                   'Converged: KKT conditions satisfied', hist, [])

        # Constraint Jacobians (numerical)
        Ae = []
        for h in eq_constraints:
            Ae.append(numerical_gradient(h, x))
        Ai = []
        for gf in ineq_constraints:
            Ai.append(numerical_gradient(gf, x))

        # Solve QP subproblem via KKT system
        # min 0.5 d^T B d + g^T d
        # s.t. Ae d + h = 0, Ai d + g <= 0
        # For simplicity, convert to equality-only with active set
        d = _solve_sqp_qp(B, g, Ae, h_vals, Ai, g_vals, lam_ineq, n, ne, ni)

        if d is None or vec_norm(d) < 1e-15:
            return OptimizeResult(x, fval, g, it, nfev, 0, False,
                                   'QP subproblem failed', hist, [])

        # Merit function line search
        merit_mu = max(10.0, *(abs(l) for l in lam_eq), *(abs(l) for l in lam_ineq)) if (lam_eq or lam_ineq) else 10.0

        def merit(xp):
            val = f(xp)
            for h in eq_constraints:
                val += merit_mu * abs(h(xp))
            for gf in ineq_constraints:
                val += merit_mu * max(0.0, gf(xp))
            return val

        alpha = 1.0
        m0 = merit(x)
        for _ in range(30):
            x_new = vec_add(x, vec_scale(d, alpha))
            if merit(x_new) < m0 - 1e-4 * alpha * vec_norm(d):
                break
            alpha *= 0.5
        nfev += 30

        x_new = vec_add(x, vec_scale(d, alpha))
        fval_new, g_new = val_grad(x_new)
        nfev += 1

        # BFGS update on Lagrangian Hessian
        s = vec_scale(d, alpha)
        # Gradient of Lagrangian
        gl_old = list(g)
        gl_new = list(g_new)
        for i, h in enumerate(eq_constraints):
            jac = Ae[i]
            for j in range(n):
                gl_old[j] += lam_eq[i] * jac[j]

        # Recompute constraint Jacobians at new point
        for i, h in enumerate(eq_constraints):
            jac_new = numerical_gradient(h, x_new)
            for j in range(n):
                gl_new[j] += lam_eq[i] * jac_new[j]

        y = vec_sub(gl_new, gl_old)
        sy = dot(s, y)

        if sy > 1e-12:
            Bs = _matvec(B, s, n)
            sBs = dot(s, Bs)
            for i in range(n):
                for j in range(n):
                    B[i][j] += y[i] * y[j] / sy - Bs[i] * Bs[j] / sBs

        # Update multipliers from QP solution
        # Simple update: project
        for i in range(ne):
            hi = eq_constraints[i](x_new)
            lam_eq[i] += merit_mu * hi
        for i in range(ni):
            gi = ineq_constraints[i](x_new)
            lam_ineq[i] = max(0.0, lam_ineq[i] + merit_mu * gi)

        x = x_new
        fval = fval_new
        g = g_new

    return OptimizeResult(x, fval, g, max_iter, nfev, 0, False,
                           'Maximum iterations reached', hist, [])


def _solve_sqp_qp(B, g, Ae, h_vals, Ai, g_vals, lam_ineq, n, ne, ni):
    """Solve QP subproblem for SQP.
    min 0.5 d^T B d + g^T d  s.t. Ae d + h = 0

    For inequality constraints, use active set (constraints near boundary).
    """
    # Determine active inequalities
    active_i = []
    for j in range(ni):
        if g_vals[j] > -1e-4 or lam_ineq[j] > 1e-8:
            active_i.append(j)

    m = ne + len(active_i)  # total active constraints
    if m == 0:
        # Unconstrained QP: solve B d = -g
        try:
            d = lu_solve(Matrix(B), [-gi for gi in g])
            return d
        except (ValueError, ZeroDivisionError):
            return [-gi for gi in g]

    # Build KKT system:
    # [B  A^T] [d]   = [-g]
    # [A   0 ] [lam]   [-c]
    sz = n + m
    K = [[0.0] * sz for _ in range(sz)]

    # Fill B
    for i in range(n):
        for j in range(n):
            K[i][j] = B[i][j]

    # Fill A and A^T (equality)
    for i in range(ne):
        for j in range(n):
            K[n + i][j] = Ae[i][j]
            K[j][n + i] = Ae[i][j]

    # Fill A and A^T (active inequality)
    for idx, ai in enumerate(active_i):
        row = ne + idx
        for j in range(n):
            K[n + row][j] = Ai[ai][j]
            K[j][n + row] = Ai[ai][j]

    # RHS
    rhs = [0.0] * sz
    for i in range(n):
        rhs[i] = -g[i]
    for i in range(ne):
        rhs[n + i] = -h_vals[i]
    for idx, ai in enumerate(active_i):
        rhs[n + ne + idx] = -g_vals[ai]

    try:
        sol = lu_solve(Matrix(K), rhs)
        return sol[:n]
    except (ValueError, ZeroDivisionError):
        # Fallback
        try:
            return lu_solve(Matrix(B), [-gi for gi in g])
        except (ValueError, ZeroDivisionError):
            return [-gi for gi in g]


# ---------------------------------------------------------------------------
# Convenience: minimize()
# ---------------------------------------------------------------------------

def minimize(f, x0, method='bfgs', bounds=None, eq_constraints=None,
              ineq_constraints=None, grad_f=None, hess_f=None,
              max_iter=None, tol=1e-8, use_ad=True, history=False, **kwargs):
    """Unified interface for optimization.

    Args:
        method: 'steepest_descent', 'newton', 'bfgs', 'lbfgs',
                'cg', 'nelder_mead', 'trust_region',
                'projected_gradient', 'augmented_lagrangian',
                'penalty', 'sqp'
    """
    method = method.lower().replace('-', '_').replace(' ', '_')

    if method == 'steepest_descent':
        kw = dict(max_iter=max_iter or 1000, tol=tol, use_ad=use_ad,
                   history=history, grad_f=grad_f)
        kw.update(kwargs)
        return steepest_descent(f, x0, **kw)

    elif method == 'newton':
        kw = dict(max_iter=max_iter or 100, tol=tol, use_ad=use_ad,
                   history=history, grad_f=grad_f, hess_f=hess_f)
        kw.update(kwargs)
        return newton(f, x0, **kw)

    elif method == 'bfgs':
        kw = dict(max_iter=max_iter or 200, tol=tol, use_ad=use_ad,
                   history=history, grad_f=grad_f)
        kw.update(kwargs)
        return bfgs(f, x0, **kw)

    elif method == 'lbfgs' or method == 'l_bfgs':
        kw = dict(max_iter=max_iter or 200, tol=tol, use_ad=use_ad,
                   history=history, grad_f=grad_f)
        kw.update(kwargs)
        return lbfgs(f, x0, **kw)

    elif method in ('cg', 'conjugate_gradient'):
        kw = dict(max_iter=max_iter or 1000, tol=tol, use_ad=use_ad,
                   history=history, grad_f=grad_f)
        kw.update(kwargs)
        return conjugate_gradient(f, x0, **kw)

    elif method == 'nelder_mead':
        kw = dict(max_iter=max_iter or 1000, tol=tol, history=history)
        kw.update(kwargs)
        return nelder_mead(f, x0, **kw)

    elif method == 'trust_region':
        kw = dict(max_iter=max_iter or 200, tol=tol, use_ad=use_ad,
                   history=history, grad_f=grad_f, hess_f=hess_f)
        kw.update(kwargs)
        return trust_region(f, x0, **kw)

    elif method == 'projected_gradient':
        if bounds is None:
            raise ValueError("projected_gradient requires bounds")
        kw = dict(max_iter=max_iter or 1000, tol=tol, use_ad=use_ad,
                   history=history, grad_f=grad_f)
        kw.update(kwargs)
        return projected_gradient(f, x0, bounds, **kw)

    elif method == 'augmented_lagrangian':
        kw = dict(max_iter=max_iter or 50, tol=tol or 1e-6, use_ad=use_ad,
                   history=history)
        kw.update(kwargs)
        return augmented_lagrangian(f, x0, eq_constraints=eq_constraints,
                                      ineq_constraints=ineq_constraints, **kw)

    elif method == 'penalty':
        kw = dict(max_iter=max_iter or 30, tol=tol or 1e-6, use_ad=use_ad,
                   history=history)
        kw.update(kwargs)
        return penalty_method(f, x0, eq_constraints=eq_constraints,
                                ineq_constraints=ineq_constraints, **kw)

    elif method == 'sqp':
        kw = dict(max_iter=max_iter or 100, tol=tol or 1e-6, use_ad=use_ad,
                   history=history, grad_f=grad_f)
        kw.update(kwargs)
        return sqp(f, x0, eq_constraints=eq_constraints,
                    ineq_constraints=ineq_constraints, **kw)

    else:
        raise ValueError(f"Unknown method: {method}. Options: steepest_descent, "
                          "newton, bfgs, lbfgs, cg, nelder_mead, trust_region, "
                          "projected_gradient, augmented_lagrangian, penalty, sqp")


# ---------------------------------------------------------------------------
# Specialized: Least squares (Gauss-Newton + Levenberg-Marquardt)
# ---------------------------------------------------------------------------

def gauss_newton(residuals, x0, jacobian_f=None, max_iter=100, tol=1e-8,
                  use_ad=True, history=False):
    """Gauss-Newton method for nonlinear least squares.

    Minimizes sum(r_i(x)^2) where residuals(x) returns [r_1, ..., r_m].
    """
    n = len(x0)
    x = list(x0)
    hist = []
    nfev = 0

    for it in range(max_iter):
        r = residuals(x)
        nfev += 1
        m = len(r)
        fval = 0.5 * sum(ri * ri for ri in r)

        if history:
            hist.append(fval)

        if jacobian_f is not None:
            J = jacobian_f(x)
        else:
            J = numerical_jacobian(residuals, x)
        nfev += 1

        # Normal equations: J^T J d = -J^T r
        JtJ = [[0.0] * n for _ in range(n)]
        Jtr = [0.0] * n
        for i in range(m):
            for j in range(n):
                Jtr[j] += J[i][j] * r[i]
                for k in range(n):
                    JtJ[j][k] += J[i][j] * J[i][k]

        try:
            d = lu_solve(Matrix(JtJ), [-v for v in Jtr])
        except (ValueError, ZeroDivisionError):
            return OptimizeResult(x, fval, Jtr, it, nfev, 0, False,
                                   'Singular Jacobian', hist, [])

        dnorm = vec_norm(d)
        if dnorm < tol:
            return OptimizeResult(x, fval, Jtr, it, nfev, 0, True,
                                   'Converged: step size below tolerance', hist, [])

        # Simple backtracking
        alpha = 1.0
        for _ in range(20):
            x_new = vec_add(x, vec_scale(d, alpha))
            r_new = residuals(x_new)
            nfev += 1
            f_new = 0.5 * sum(ri * ri for ri in r_new)
            if f_new < fval:
                break
            alpha *= 0.5

        x = vec_add(x, vec_scale(d, alpha))

    r = residuals(x)
    fval = 0.5 * sum(ri * ri for ri in r)
    return OptimizeResult(x, fval, None, max_iter, nfev, 0, False,
                           'Maximum iterations reached', hist, [])


def levenberg_marquardt(residuals, x0, jacobian_f=None, max_iter=200,
                         tol=1e-8, lambda0=1e-3, use_ad=True, history=False):
    """Levenberg-Marquardt method for nonlinear least squares.

    Interpolates between Gauss-Newton and gradient descent via damping.
    """
    n = len(x0)
    x = list(x0)
    lam = lambda0
    hist = []
    nfev = 0
    nu = 2.0

    r = residuals(x)
    nfev += 1
    fval = 0.5 * sum(ri * ri for ri in r)

    for it in range(max_iter):
        if history:
            hist.append(fval)

        if jacobian_f is not None:
            J = jacobian_f(x)
        else:
            J = numerical_jacobian(residuals, x)
        nfev += 1

        m = len(r)

        # J^T J + lambda * I
        JtJ = [[0.0] * n for _ in range(n)]
        Jtr = [0.0] * n
        for i in range(m):
            for j in range(n):
                Jtr[j] += J[i][j] * r[i]
                for k in range(n):
                    JtJ[j][k] += J[i][j] * J[i][k]

        grad_norm = vec_norm(Jtr)
        if grad_norm < tol:
            return OptimizeResult(x, fval, Jtr, it, nfev, 0, True,
                                   'Converged: gradient norm below tolerance',
                                   hist, [])

        # Damped normal equations
        for i in range(n):
            JtJ[i][i] += lam

        try:
            d = lu_solve(Matrix(JtJ), [-v for v in Jtr])
        except (ValueError, ZeroDivisionError):
            lam *= nu
            nu *= 2.0
            continue

        dnorm = vec_norm(d)
        if dnorm < tol * (vec_norm(x) + tol):
            return OptimizeResult(x, fval, Jtr, it, nfev, 0, True,
                                   'Converged: step size below tolerance',
                                   hist, [])

        x_new = vec_add(x, d)
        r_new = residuals(x_new)
        nfev += 1
        f_new = 0.5 * sum(ri * ri for ri in r_new)

        # Gain ratio
        # Predicted reduction: -g^T d - 0.5 d^T (J^T J) d
        # But we already solved (JtJ + lam*I) d = -Jtr, so
        # pred = 0.5 d^T (lam * d - Jtr)  -- from the LM formula
        pred = 0.5 * (lam * dot(d, d) + dot(d, Jtr))

        if pred > 0:
            rho_k = (fval - f_new) / pred
        else:
            rho_k = 0.0 if f_new >= fval else 1.0

        if rho_k > 0.25:
            x = x_new
            r = r_new
            fval = f_new
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho_k - 1.0) ** 3)
            nu = 2.0
        else:
            lam *= nu
            nu *= 2.0

    return OptimizeResult(x, fval, None, max_iter, nfev, 0, False,
                           'Maximum iterations reached', hist, [])


# ---------------------------------------------------------------------------
# Specialized: Root finding
# ---------------------------------------------------------------------------

def newton_raphson(f, x0, df=None, tol=1e-10, max_iter=100):
    """Newton-Raphson root finding for scalar f(x) = 0."""
    x = float(x0)
    nfev = 0
    for it in range(max_iter):
        fx = f(x); nfev += 1
        if abs(fx) < tol:
            return OptimizeResult([x], abs(fx), None, it, nfev, 0, True,
                                   'Converged')
        if df is not None:
            dfx = df(x)
        else:
            eps = 1e-7
            dfx = (f(x + eps) - f(x - eps)) / (2 * eps)
            nfev += 2
        if abs(dfx) < 1e-30:
            return OptimizeResult([x], abs(fx), None, it, nfev, 0, False,
                                   'Zero derivative')
        x -= fx / dfx

    fx = f(x); nfev += 1
    return OptimizeResult([x], abs(fx), None, max_iter, nfev, 0, False,
                           'Maximum iterations reached')


def bisection(f, a, b, tol=1e-10, max_iter=100):
    """Bisection method for f(x) = 0 on [a, b]. Requires f(a)*f(b) < 0."""
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    nfev = 2
    for it in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c); nfev += 1
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return OptimizeResult([c], abs(fc), None, it, nfev, 0, True,
                                   'Converged')
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    c = (a + b) / 2.0
    return OptimizeResult([c], abs(f(c)), None, max_iter, nfev + 1, 0, False,
                           'Maximum iterations reached')


def brent(f, a, b, tol=1e-10, max_iter=100):
    """Brent's method for root finding (combines bisection + secant + IQI)."""
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    nfev = 2

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a; fc = fa
    mflag = True
    d = 0.0

    for it in range(max_iter):
        if abs(fb) < tol:
            return OptimizeResult([b], abs(fb), None, it, nfev, 0, True,
                                   'Converged')
        if abs(b - a) < tol:
            return OptimizeResult([b], abs(fb), None, it, nfev, 0, True,
                                   'Converged')

        if abs(fa - fc) > 1e-30 and abs(fb - fc) > 1e-30:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant
            if abs(fb - fa) < 1e-30:
                s = b
            else:
                s = b - fb * (b - a) / (fb - fa)

        # Conditions for bisection
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s); nfev += 1
        d = c
        c = b; fc = fb

        if fa * fs < 0:
            b = s; fb = fs
        else:
            a = s; fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return OptimizeResult([b], abs(fb), None, max_iter, nfev, 0, False,
                           'Maximum iterations reached')


# ---------------------------------------------------------------------------
# Multi-dimensional root finding
# ---------------------------------------------------------------------------

def newton_system(F, x0, J=None, tol=1e-10, max_iter=100):
    """Newton's method for systems of equations F(x) = 0.

    Args:
        F: Function returning list of residuals [f1(x), f2(x), ...]
        x0: Initial guess
        J: Jacobian function (optional, uses numerical if None)
    """
    n = len(x0)
    x = list(x0)
    nfev = 0

    for it in range(max_iter):
        Fx = F(x); nfev += 1
        fnorm = math.sqrt(sum(fi * fi for fi in Fx))

        if fnorm < tol:
            return OptimizeResult(x, fnorm, None, it, nfev, 0, True,
                                   'Converged')

        if J is not None:
            Jx = J(x)
        else:
            Jx = numerical_jacobian(F, x)
            nfev += 2 * n

        # Solve J @ d = -F
        try:
            d = lu_solve(Matrix(Jx), [-fi for fi in Fx])
        except (ValueError, ZeroDivisionError):
            return OptimizeResult(x, fnorm, None, it, nfev, 0, False,
                                   'Singular Jacobian')

        # Line search on ||F||^2
        alpha = 1.0
        for _ in range(20):
            x_new = vec_add(x, vec_scale(d, alpha))
            Fx_new = F(x_new); nfev += 1
            fnorm_new = math.sqrt(sum(fi * fi for fi in Fx_new))
            if fnorm_new < fnorm:
                break
            alpha *= 0.5

        x = vec_add(x, vec_scale(d, alpha))

    Fx = F(x); nfev += 1
    fnorm = math.sqrt(sum(fi * fi for fi in Fx))
    return OptimizeResult(x, fnorm, None, max_iter, nfev, 0, False,
                           'Maximum iterations reached')


# ---------------------------------------------------------------------------
# Specialized: Curve fitting
# ---------------------------------------------------------------------------

def curve_fit(model, xdata, ydata, p0, method='lm', max_iter=200, tol=1e-8):
    """Nonlinear curve fitting.

    Fits model(x, *params) to (xdata, ydata) by minimizing sum of squared residuals.

    Args:
        model: Function model(x, *params) -> y
        xdata: List of x values
        ydata: List of y values
        p0: Initial parameter guess
        method: 'lm' (Levenberg-Marquardt) or 'gn' (Gauss-Newton)
    """
    def residuals(params):
        return [model(xdata[i], *params) - ydata[i] for i in range(len(xdata))]

    if method == 'lm':
        result = levenberg_marquardt(residuals, p0, max_iter=max_iter, tol=tol)
    elif method == 'gn':
        result = gauss_newton(residuals, p0, max_iter=max_iter, tol=tol)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute residuals and covariance estimate
    r = residuals(result.x)
    m = len(r)
    n = len(p0)
    J = numerical_jacobian(residuals, result.x)

    # Covariance: (J^T J)^-1 * s^2
    JtJ = [[0.0] * n for _ in range(n)]
    for i in range(m):
        for j in range(n):
            for k in range(n):
                JtJ[j][k] += J[i][j] * J[i][k]

    s2 = sum(ri * ri for ri in r) / max(1, m - n)

    try:
        from linear_algebra import matrix_inverse
        cov = matrix_inverse(Matrix(JtJ))
        pcov = [[cov[i, j] * s2 for j in range(n)] for i in range(n)]
    except (ValueError, ZeroDivisionError):
        pcov = None

    result.pcov = pcov
    return result
