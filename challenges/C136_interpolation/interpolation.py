"""
C136: Interpolation and Approximation
Composing C132 (Linear Algebra).

Polynomial interpolation (Lagrange, Newton, Chebyshev), splines (linear, cubic,
natural, clamped, not-a-knot, periodic), rational interpolation, RBF interpolation,
piecewise Hermite, Akima, B-splines, least-squares fitting, and Pade approximants.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C132_linear_algebra'))
from linear_algebra import Matrix, lu_solve, least_squares, qr_solve


# ============================================================
# Lagrange Interpolation
# ============================================================

def lagrange_interpolate(xs, ys, x):
    """Evaluate Lagrange interpolant at x given data points (xs, ys)."""
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must have same length")
    if n == 0:
        raise ValueError("Need at least one data point")
    result = 0.0
    for i in range(n):
        basis = 1.0
        for j in range(n):
            if i != j:
                basis *= (x - xs[j]) / (xs[i] - xs[j])
        result += ys[i] * basis
    return result


def lagrange_weights(xs):
    """Compute barycentric weights for Lagrange interpolation."""
    n = len(xs)
    w = [1.0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (xs[i] - xs[j])
    return w


def barycentric_interpolate(xs, ys, weights, x):
    """Barycentric Lagrange interpolation -- O(n) per evaluation."""
    n = len(xs)
    numer = 0.0
    denom = 0.0
    for i in range(n):
        if abs(x - xs[i]) < 1e-15:
            return ys[i]
        t = weights[i] / (x - xs[i])
        numer += t * ys[i]
        denom += t
    return numer / denom


# ============================================================
# Newton Interpolation (Divided Differences)
# ============================================================

def divided_differences(xs, ys):
    """Compute Newton divided difference coefficients."""
    n = len(xs)
    coefs = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coefs[i] = (coefs[i] - coefs[i - 1]) / (xs[i] - xs[i - j])
    return coefs


def newton_interpolate(xs, ys, x):
    """Evaluate Newton interpolant at x."""
    coefs = divided_differences(xs, ys)
    n = len(coefs)
    result = coefs[n - 1]
    for i in range(n - 2, -1, -1):
        result = result * (x - xs[i]) + coefs[i]
    return result


def newton_add_point(xs, ys, coefs, x_new, y_new):
    """Add a point to Newton interpolation incrementally."""
    xs.append(x_new)
    ys.append(y_new)
    n = len(xs)
    # Compute new divided difference
    new_coef = y_new
    for j in range(n - 1):
        new_coef = (new_coef - _newton_eval_partial(xs, coefs, x_new, j)) / (x_new - xs[j])
    coefs.append(new_coef)
    return coefs


def _newton_eval_partial(xs, coefs, x, k):
    """Evaluate first k terms of Newton polynomial."""
    if k == 0:
        return coefs[0]
    result = coefs[k]
    for i in range(k - 1, -1, -1):
        result = result * (x - xs[i]) + coefs[i]
    return result


# ============================================================
# Chebyshev Interpolation
# ============================================================

def chebyshev_nodes(n, a=-1.0, b=1.0):
    """Generate n Chebyshev nodes on [a, b]."""
    nodes = []
    for k in range(n):
        t = math.cos(math.pi * (2 * k + 1) / (2 * n))
        nodes.append(0.5 * (a + b) + 0.5 * (b - a) * t)
    return nodes


def chebyshev_coefficients(f, n, a=-1.0, b=1.0):
    """Compute Chebyshev coefficients of f on [a, b] using DCT-like formula."""
    nodes = chebyshev_nodes(n, a, b)
    fvals = [f(x) for x in nodes]
    coefs = []
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += fvals[k] * math.cos(math.pi * j * (2 * k + 1) / (2 * n))
        coefs.append(2.0 * s / n)
    coefs[0] /= 2.0
    return coefs


def chebyshev_evaluate(coefs, x, a=-1.0, b=1.0):
    """Evaluate Chebyshev expansion using Clenshaw recurrence."""
    t = (2.0 * x - (a + b)) / (b - a)
    n = len(coefs)
    if n == 0:
        return 0.0
    if n == 1:
        return coefs[0]
    b_k1 = 0.0
    b_k2 = 0.0
    for j in range(n - 1, 0, -1):
        b_k1, b_k2 = 2.0 * t * b_k1 - b_k2 + coefs[j], b_k1
    return t * b_k1 - b_k2 + coefs[0]


class ChebyshevApprox:
    """Chebyshev polynomial approximation of a function on [a, b]."""

    def __init__(self, f, n, a=-1.0, b=1.0):
        self.a = a
        self.b = b
        self.n = n
        self.coefs = chebyshev_coefficients(f, n, a, b)

    def __call__(self, x):
        return chebyshev_evaluate(self.coefs, x, self.a, self.b)

    def truncate(self, m):
        """Return approximation with only first m coefficients."""
        approx = ChebyshevApprox.__new__(ChebyshevApprox)
        approx.a = self.a
        approx.b = self.b
        approx.n = m
        approx.coefs = self.coefs[:m]
        return approx

    def derivative(self):
        """Return Chebyshev approximation of the derivative."""
        n = len(self.coefs)
        if n <= 1:
            d_coefs = [0.0]
        else:
            # Chebyshev derivative recurrence: d'_j = d'_{j+2} + 2(j+1)c_{j+1}
            # d_coefs has n elements (same size), d_coefs[n-1] = 0 sentinel
            d_coefs = [0.0] * n
            d_coefs[n - 1] = 0.0
            if n >= 2:
                d_coefs[n - 2] = 2.0 * (n - 1) * self.coefs[n - 1]
            for j in range(n - 3, -1, -1):
                d_coefs[j] = d_coefs[j + 2] + 2.0 * (j + 1) * self.coefs[j + 1]
            d_coefs[0] /= 2.0
            d_coefs = d_coefs[:-1]  # Remove sentinel
            # Scale for [a, b]
            scale = 2.0 / (self.b - self.a)
            d_coefs = [c * scale for c in d_coefs]
        approx = ChebyshevApprox.__new__(ChebyshevApprox)
        approx.a = self.a
        approx.b = self.b
        approx.n = len(d_coefs)
        approx.coefs = d_coefs
        return approx

    def integral(self):
        """Return Chebyshev approximation of the integral (constant = 0)."""
        n = len(self.coefs)
        scale = (self.b - self.a) / 2.0
        i_coefs = [0.0] * (n + 1)
        # First coef handled via integration formula
        for j in range(1, n):
            i_coefs[j] = scale * (self.coefs[j - 1] / (2.0 * j) - (self.coefs[j + 1] / (2.0 * j) if j + 1 < n else 0.0))
        i_coefs[0] = 0.0  # Integration constant
        # Fix: recompute c_0 so integral at a = 0
        # Actually use proper Chebyshev integration
        if n > 0:
            i_coefs[1] = scale * self.coefs[0]
            for j in range(2, n + 1):
                c_prev = self.coefs[j - 1] if j - 1 < n else 0.0
                c_next = self.coefs[j + 1] if j + 1 < n else 0.0
                i_coefs[j] = scale * (c_prev - c_next) / (2.0 * j) if j > 0 else 0.0
        approx = ChebyshevApprox.__new__(ChebyshevApprox)
        approx.a = self.a
        approx.b = self.b
        approx.n = len(i_coefs)
        approx.coefs = i_coefs
        return approx


# ============================================================
# Linear Spline
# ============================================================

class LinearSpline:
    """Piecewise linear interpolation."""

    def __init__(self, xs, ys):
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("Need at least 2 data points with matching xs, ys")
        # Sort by x
        pairs = sorted(zip(xs, ys))
        self.xs = [p[0] for p in pairs]
        self.ys = [p[1] for p in pairs]
        self.n = len(self.xs)

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        xs, ys = self.xs, self.ys
        # Clamp to endpoints
        if x <= xs[0]:
            return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
        if x >= xs[-1]:
            return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1])
        # Binary search
        lo, hi = 0, self.n - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if xs[mid + 1] < x:
                lo = mid + 1
            else:
                hi = mid
        t = (x - xs[lo]) / (xs[lo + 1] - xs[lo])
        return ys[lo] + t * (ys[lo + 1] - ys[lo])


# ============================================================
# Cubic Spline
# ============================================================

class CubicSpline:
    """Cubic spline interpolation with multiple boundary conditions."""

    def __init__(self, xs, ys, bc_type='natural', clamped_slopes=None):
        """
        bc_type: 'natural' (S''=0), 'clamped' (specified slopes), 'not-a-knot', 'periodic'
        clamped_slopes: (slope_left, slope_right) for 'clamped' type
        """
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("Need at least 2 data points")
        pairs = sorted(zip(xs, ys))
        self.xs = [p[0] for p in pairs]
        self.ys = [p[1] for p in pairs]
        self.n = len(self.xs)
        self.bc_type = bc_type

        if self.n == 2:
            # Linear case
            self.a = [self.ys[0]]
            self.b = [(self.ys[1] - self.ys[0]) / (self.xs[1] - self.xs[0])]
            self.c = [0.0]
            self.d = [0.0]
            return

        n = self.n
        h = [self.xs[i + 1] - self.xs[i] for i in range(n - 1)]

        # Set up tridiagonal system for second derivatives
        if bc_type == 'natural':
            self._solve_natural(h)
        elif bc_type == 'clamped':
            if clamped_slopes is None:
                raise ValueError("clamped_slopes required for clamped BC")
            self._solve_clamped(h, clamped_slopes)
        elif bc_type == 'not-a-knot':
            self._solve_not_a_knot(h)
        elif bc_type == 'periodic':
            self._solve_periodic(h)
        else:
            raise ValueError(f"Unknown bc_type: {bc_type}")

    def _solve_natural(self, h):
        n = self.n
        # c[0] = c[n-1] = 0, solve for c[1..n-2]
        c = [0.0] * n
        if n <= 2:
            self._compute_abd(h, c)
            return
        # Tridiagonal system
        size = n - 2
        diag = [2.0 * (h[i] + h[i + 1]) for i in range(size)]
        upper = [h[i + 1] for i in range(size - 1)]
        lower = [h[i + 1] for i in range(size - 1)]
        rhs = [3.0 * ((self.ys[i + 2] - self.ys[i + 1]) / h[i + 1] - (self.ys[i + 1] - self.ys[i]) / h[i]) for i in range(size)]
        sol = _solve_tridiagonal(lower, diag, upper, rhs)
        for i in range(size):
            c[i + 1] = sol[i]
        self._compute_abd(h, c)

    def _solve_clamped(self, h, slopes):
        n = self.n
        s0, sn = slopes
        c = [0.0] * n
        size = n
        diag = [0.0] * size
        upper = [0.0] * (size - 1)
        lower = [0.0] * (size - 1)
        rhs = [0.0] * size
        # First equation
        diag[0] = 2.0 * h[0]
        upper[0] = h[0]
        rhs[0] = 3.0 * ((self.ys[1] - self.ys[0]) / h[0] - s0)
        # Interior
        for i in range(1, n - 1):
            lower[i - 1] = h[i - 1]
            diag[i] = 2.0 * (h[i - 1] + h[i])
            if i < n - 1:
                if i < size - 1:
                    upper[i] = h[i]
            rhs[i] = 3.0 * ((self.ys[i + 1] - self.ys[i]) / h[i] - (self.ys[i] - self.ys[i - 1]) / h[i - 1])
        # Last equation
        lower[n - 2] = h[n - 2]
        diag[n - 1] = 2.0 * h[n - 2]
        rhs[n - 1] = 3.0 * (sn - (self.ys[n - 1] - self.ys[n - 2]) / h[n - 2])
        sol = _solve_tridiagonal(lower, diag, upper, rhs)
        for i in range(n):
            c[i] = sol[i]
        self._compute_abd(h, c)

    def _solve_not_a_knot(self, h):
        n = self.n
        if n == 3:
            # Single cubic through all 3 points
            c = [0.0] * n
            # Not-a-knot: c[0] = c[1] and c[1] = c[2] => uniform c
            d0 = (self.ys[1] - self.ys[0]) / h[0]
            d1 = (self.ys[2] - self.ys[1]) / h[1]
            c_val = (d1 - d0) / (h[0] + h[1])
            c[0] = c[1] = c[2] = c_val
            self._compute_abd(h, c)
            return
        c = [0.0] * n
        size = n - 2
        diag = [0.0] * size
        upper = [0.0] * (size - 1)
        lower = [0.0] * (size - 1)
        rhs = [0.0] * size

        for i in range(size):
            diag[i] = 2.0 * (h[i] + h[i + 1])
            if i < size - 1:
                upper[i] = h[i + 1]
            if i > 0:
                lower[i - 1] = h[i]
            rhs[i] = 3.0 * ((self.ys[i + 2] - self.ys[i + 1]) / h[i + 1] - (self.ys[i + 1] - self.ys[i]) / h[i])

        # Not-a-knot: d[0] = d[1] => c[0] derived from c[1], c[2]
        # Modify first equation
        diag[0] = h[1] * (h[1] + h[0])
        upper[0] = (h[0] + h[1]) * (h[0] + h[1])
        rhs[0] = rhs[0] * h[1] / (h[0] + h[1])
        # Wait, this is getting complicated. Use a different approach.
        # Build full system with not-a-knot conditions
        # c[0] = c[1] * (1 + h[0]/h[1]) - c[2] * h[0]/h[1]
        # c[n-1] = c[n-2] * (1 + h[n-2]/h[n-3]) - c[n-3] * h[n-2]/h[n-3]

        # Rebuild: solve full n x n system
        A = [[0.0] * n for _ in range(n)]
        b = [0.0] * n

        # Not-a-knot at left: d[0] = d[1]
        # d[i] = (c[i+1] - c[i]) / (3 h[i])
        # d[0] = d[1] => (c[1] - c[0])/(3h[0]) = (c[2] - c[1])/(3h[1])
        # h[1](c[1] - c[0]) = h[0](c[2] - c[1])
        A[0][0] = -h[1]
        A[0][1] = h[1] + h[0]
        A[0][2] = -h[0]
        b[0] = 0.0

        # Interior equations
        for i in range(1, n - 1):
            A[i][i - 1] = h[i - 1]
            A[i][i] = 2.0 * (h[i - 1] + h[i])
            A[i][i + 1] = h[i]
            b[i] = 3.0 * ((self.ys[i + 1] - self.ys[i]) / h[i] - (self.ys[i] - self.ys[i - 1]) / h[i - 1])

        # Not-a-knot at right: d[n-3] = d[n-2]
        A[n - 1][n - 3] = -h[n - 2]
        A[n - 1][n - 2] = h[n - 2] + h[n - 3]
        A[n - 1][n - 1] = -h[n - 3]
        b[n - 1] = 0.0

        sol = lu_solve(Matrix(A), b)
        for i in range(n):
            c[i] = sol[i]
        self._compute_abd(h, c)

    def _solve_periodic(self, h):
        n = self.n
        if abs(self.ys[0] - self.ys[-1]) > 1e-10:
            raise ValueError("Periodic spline requires y[0] == y[-1]")
        # Solve (n-1) x (n-1) system with wrap-around
        m = n - 1
        A = [[0.0] * m for _ in range(m)]
        b = [0.0] * m
        for i in range(m):
            ip = (i + 1) % m
            im = (i - 1) % m
            A[i][im] = h[im]
            A[i][i] = 2.0 * (h[im] + h[i % (n - 1)])
            A[i][ip] = h[i % (n - 1)]
            b[i] = 3.0 * ((self.ys[ip + (1 if ip > i else 0)] - self.ys[i + (0 if ip > i else 0)]) / h[i % (n - 1)]
                          - (self.ys[i] - self.ys[im]) / h[im])

        # This is a cyclic tridiagonal; solve with dense LU
        sol = lu_solve(Matrix(A), b)
        c = [sol[i] for i in range(m)] + [sol[0]]
        self._compute_abd(h, c)

    def _compute_abd(self, h, c):
        n = self.n
        self.a = list(self.ys[:-1])
        self.b = [0.0] * (n - 1)
        self.c = c
        self.d = [0.0] * (n - 1)
        for i in range(n - 1):
            self.b[i] = (self.ys[i + 1] - self.ys[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0
            self.d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        xs = self.xs
        n = self.n
        if x <= xs[0]:
            i = 0
        elif x >= xs[-1]:
            i = n - 2
        else:
            lo, hi = 0, n - 2
            while lo < hi:
                mid = (lo + hi) // 2
                if xs[mid + 1] < x:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        dx = x - xs[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3

    def derivative(self, x, order=1):
        """Evaluate derivative at x."""
        xs = self.xs
        n = self.n
        if x <= xs[0]:
            i = 0
        elif x >= xs[-1]:
            i = n - 2
        else:
            lo, hi = 0, n - 2
            while lo < hi:
                mid = (lo + hi) // 2
                if xs[mid + 1] < x:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        dx = x - xs[i]
        if order == 1:
            return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2
        elif order == 2:
            return 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        else:
            raise ValueError("Only order 1 and 2 supported")

    def integrate(self, a=None, b=None):
        """Integrate spline over [a, b]."""
        if a is None:
            a = self.xs[0]
        if b is None:
            b = self.xs[-1]
        if a > b:
            return -self.integrate(b, a)
        total = 0.0
        for i in range(self.n - 1):
            x0 = max(a, self.xs[i])
            x1 = min(b, self.xs[i + 1])
            if x0 >= x1:
                continue
            dx0 = x0 - self.xs[i]
            dx1 = x1 - self.xs[i]
            ai, bi, ci, di = self.a[i], self.b[i], self.c[i], self.d[i]
            def antideriv(dx):
                return ai * dx + bi * dx**2 / 2.0 + ci * dx**3 / 3.0 + di * dx**4 / 4.0
            total += antideriv(dx1) - antideriv(dx0)
        return total


def _solve_tridiagonal(lower, diag, upper, rhs):
    """Solve tridiagonal system using Thomas algorithm."""
    n = len(diag)
    if n == 0:
        return []
    c = list(upper)
    d = list(rhs)
    diag = list(diag)
    # Forward sweep
    for i in range(1, n):
        if abs(diag[i - 1]) < 1e-30:
            raise ValueError("Zero pivot in tridiagonal solve")
        m = lower[i - 1] / diag[i - 1]
        diag[i] -= m * c[i - 1] if i - 1 < len(c) else 0
        d[i] -= m * d[i - 1]
    # Back substitution
    x = [0.0] * n
    x[n - 1] = d[n - 1] / diag[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - (c[i] * x[i + 1] if i < len(c) else 0)) / diag[i]
    return x


# ============================================================
# Hermite Interpolation (Piecewise Cubic Hermite -- PCHIP)
# ============================================================

class PchipInterpolator:
    """Piecewise Cubic Hermite Interpolating Polynomial (shape-preserving)."""

    def __init__(self, xs, ys):
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("Need at least 2 points")
        pairs = sorted(zip(xs, ys))
        self.xs = [p[0] for p in pairs]
        self.ys = [p[1] for p in pairs]
        self.n = len(self.xs)
        self._compute_slopes()

    def _compute_slopes(self):
        xs, ys, n = self.xs, self.ys, self.n
        h = [xs[i + 1] - xs[i] for i in range(n - 1)]
        delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
        self.slopes = [0.0] * n

        if n == 2:
            self.slopes[0] = self.slopes[1] = delta[0]
            return

        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] <= 0:
                self.slopes[i] = 0.0
            else:
                # Fritsch-Carlson
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                self.slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

        # Endpoints: one-sided
        self.slopes[0] = _pchip_end_slope(h[0], h[1], delta[0], delta[1])
        self.slopes[-1] = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2])

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        xs, ys = self.xs, self.ys
        n = self.n
        if x <= xs[0]:
            i = 0
        elif x >= xs[-1]:
            i = n - 2
        else:
            lo, hi = 0, n - 2
            while lo < hi:
                mid = (lo + hi) // 2
                if xs[mid + 1] < x:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h
        # Hermite basis
        h00 = (1 + 2 * t) * (1 - t)**2
        h10 = t * (1 - t)**2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)
        return h00 * ys[i] + h10 * h * self.slopes[i] + h01 * ys[i + 1] + h11 * h * self.slopes[i + 1]


def _pchip_end_slope(h1, h2, d1, d2):
    """One-sided slope for PCHIP endpoints."""
    s = ((2 * h1 + h2) * d1 - h1 * d2) / (h1 + h2)
    if s * d1 <= 0:
        return 0.0
    if d1 * d2 <= 0 and abs(s) > 3 * abs(d1):
        return 3.0 * d1
    return s


# ============================================================
# Akima Interpolation
# ============================================================

class AkimaInterpolator:
    """Akima spline -- reduces oscillation near outliers."""

    def __init__(self, xs, ys):
        if len(xs) != len(ys) or len(xs) < 3:
            raise ValueError("Need at least 3 points")
        pairs = sorted(zip(xs, ys))
        self.xs = [p[0] for p in pairs]
        self.ys = [p[1] for p in pairs]
        self.n = len(self.xs)
        self._compute()

    def _compute(self):
        xs, ys, n = self.xs, self.ys, self.n
        m = [0.0] * (n + 3)
        # Interior slopes
        for i in range(n - 1):
            m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
        # Extrapolated slopes
        m[1] = 2 * m[2] - m[3]
        m[0] = 2 * m[1] - m[2]
        m[n + 1] = 2 * m[n] - m[n - 1]
        m[n + 2] = 2 * m[n + 1] - m[n]

        self.slopes = [0.0] * n
        for i in range(n):
            w1 = abs(m[i + 3] - m[i + 2])
            w2 = abs(m[i + 1] - m[i])
            if w1 + w2 < 1e-30:
                self.slopes[i] = 0.5 * (m[i + 1] + m[i + 2])
            else:
                self.slopes[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2)

        # Build Hermite coefficients
        self.a = list(ys[:-1])
        self.b = list(self.slopes[:-1])
        self.c = [0.0] * (n - 1)
        self.d = [0.0] * (n - 1)
        for i in range(n - 1):
            h = xs[i + 1] - xs[i]
            delta = (ys[i + 1] - ys[i]) / h
            self.c[i] = (3 * delta - 2 * self.slopes[i] - self.slopes[i + 1]) / h
            self.d[i] = (self.slopes[i] + self.slopes[i + 1] - 2 * delta) / (h * h)

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        xs = self.xs
        n = self.n
        if x <= xs[0]:
            i = 0
        elif x >= xs[-1]:
            i = n - 2
        else:
            lo, hi = 0, n - 2
            while lo < hi:
                mid = (lo + hi) // 2
                if xs[mid + 1] < x:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        dx = x - xs[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3


# ============================================================
# Rational Interpolation (Bulirsch-Stoer)
# ============================================================

def rational_interpolate(xs, ys, x):
    """Rational (diagonal Pade) interpolation via Bulirsch-Stoer algorithm."""
    n = len(xs)
    c = list(ys)
    d = list(ys)
    ns = 0
    dif = abs(x - xs[0])
    for i in range(1, n):
        dift = abs(x - xs[i])
        if dift < dif:
            ns = i
            dif = dift
    y = ys[ns]
    ns -= 1
    for m in range(1, n):
        for i in range(n - m):
            w = c[i + 1] - d[i]
            h = xs[i + m] - x
            if abs(h) < 1e-30:
                # x coincides with a node -- return that value
                return ys[i + m]
            t = (xs[i] - x) * d[i] / h
            dd = t - c[i + 1]
            if abs(dd) < 1e-30:
                # Pole detected -- fallback to polynomial
                return lagrange_interpolate(xs, ys, x)
            dd = w / dd
            d[i] = c[i + 1] * dd
            c[i] = t * dd
        if 2 * (ns + 1) < n - m:
            y += c[ns + 1]
        else:
            y += d[ns]
            ns -= 1
    return y


# ============================================================
# RBF Interpolation
# ============================================================

def _rbf_gaussian(r, epsilon):
    return math.exp(-(epsilon * r)**2)

def _rbf_multiquadric(r, epsilon):
    return math.sqrt(1.0 + (epsilon * r)**2)

def _rbf_inverse_multiquadric(r, epsilon):
    return 1.0 / math.sqrt(1.0 + (epsilon * r)**2)

def _rbf_thin_plate(r, epsilon):
    if r < 1e-30:
        return 0.0
    return (r * epsilon)**2 * math.log(r * epsilon)

def _rbf_cubic(r, epsilon):
    return (epsilon * r)**3

_RBF_FUNCTIONS = {
    'gaussian': _rbf_gaussian,
    'multiquadric': _rbf_multiquadric,
    'inverse_multiquadric': _rbf_inverse_multiquadric,
    'thin_plate': _rbf_thin_plate,
    'cubic': _rbf_cubic,
}


class RBFInterpolator:
    """Radial Basis Function interpolation (1D or multi-dim)."""

    def __init__(self, points, values, kernel='gaussian', epsilon=1.0):
        """
        points: list of points (scalars for 1D, tuples for multi-D)
        values: list of values
        kernel: 'gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate', 'cubic'
        """
        self.n = len(points)
        if self.n != len(values):
            raise ValueError("points and values must have same length")
        # Normalize to tuples
        if isinstance(points[0], (int, float)):
            self.points = [(p,) for p in points]
        else:
            self.points = [tuple(p) for p in points]
        self.values = list(values)
        self.epsilon = epsilon
        self.kernel = _RBF_FUNCTIONS[kernel]

        # Build interpolation matrix and solve
        A = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                r = self._dist(self.points[i], self.points[j])
                A[i][j] = self.kernel(r, self.epsilon)
        self.weights = lu_solve(Matrix(A), self.values)

    def _dist(self, p1, p2):
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    def __call__(self, point):
        return self.evaluate(point)

    def evaluate(self, point):
        if isinstance(point, (int, float)):
            point = (point,)
        else:
            point = tuple(point)
        result = 0.0
        for i in range(self.n):
            r = self._dist(point, self.points[i])
            result += self.weights[i] * self.kernel(r, self.epsilon)
        return result


# ============================================================
# B-Spline
# ============================================================

class BSpline:
    """B-spline curve evaluation."""

    def __init__(self, knots, control_points, degree):
        """
        knots: knot vector (non-decreasing)
        control_points: list of control point values
        degree: polynomial degree
        """
        self.knots = list(knots)
        self.control_points = list(control_points)
        self.degree = degree
        n = len(control_points)
        k = degree + 1
        if len(knots) != n + k:
            raise ValueError(f"Need {n + k} knots for {n} control points of degree {degree}")

    def _basis(self, i, k, t):
        """Cox-de Boor recursion for B-spline basis function N_{i,k}."""
        knots = self.knots
        if k == 1:
            if knots[i] <= t < knots[i + 1]:
                return 1.0
            # Handle last knot
            if t == knots[-1] and knots[i] <= t <= knots[i + 1]:
                return 1.0
            return 0.0
        d1 = knots[i + k - 1] - knots[i]
        d2 = knots[i + k] - knots[i + 1]
        c1 = ((t - knots[i]) / d1 * self._basis(i, k - 1, t)) if d1 > 1e-30 else 0.0
        c2 = ((knots[i + k] - t) / d2 * self._basis(i + 1, k - 1, t)) if d2 > 1e-30 else 0.0
        return c1 + c2

    def __call__(self, t):
        return self.evaluate(t)

    def evaluate(self, t):
        result = 0.0
        k = self.degree + 1
        for i in range(len(self.control_points)):
            result += self.control_points[i] * self._basis(i, k, t)
        return result

    def derivative(self, t):
        """Evaluate first derivative at t."""
        n = len(self.control_points)
        p = self.degree
        if p == 0:
            return 0.0
        # Derivative control points
        d_cp = []
        for i in range(n - 1):
            denom = self.knots[i + p + 1] - self.knots[i + 1]
            if abs(denom) < 1e-30:
                d_cp.append(0.0)
            else:
                d_cp.append(p * (self.control_points[i + 1] - self.control_points[i]) / denom)
        d_knots = self.knots[1:-1]
        d_spline = BSpline(d_knots, d_cp, p - 1)
        return d_spline.evaluate(t)


def bspline_interpolate(xs, ys, degree=3):
    """Create interpolating B-spline through data points."""
    n = len(xs)
    if n < degree + 1:
        degree = n - 1
    k = degree + 1
    # Clamped knot vector
    knots = [xs[0]] * k
    if n > k:
        # Interior knots: average of degree consecutive xs
        for j in range(1, n - degree):
            avg = sum(xs[j:j + degree]) / degree
            knots.append(avg)
    knots += [xs[-1]] * k

    # Build collocation matrix
    m = len(knots) - k  # Should equal n
    A = [[0.0] * m for _ in range(n)]
    temp_spline = BSpline(knots, [0.0] * m, degree)
    for i in range(n):
        t = xs[i]
        if i == n - 1:
            t = xs[-1]  # Ensure we hit the last knot
        for j in range(m):
            A[i][j] = temp_spline._basis(j, k, t)

    control_points = lu_solve(Matrix(A), ys)
    return BSpline(knots, control_points, degree)


# ============================================================
# Pade Approximant
# ============================================================

def pade_approximant(coefs, m, n):
    """
    Compute Pade approximant [m/n] from Taylor coefficients.
    Returns (numerator_coefs, denominator_coefs).
    c[0] + c[1]x + ... ~ p(x)/q(x) where q(0) = 1.
    """
    if len(coefs) < m + n + 1:
        raise ValueError(f"Need at least {m + n + 1} Taylor coefficients")

    if n == 0:
        return coefs[:m + 1], [1.0]

    # Standard Pade: solve for q[1..n] from the equations
    # c[m+1] + c[m]*q[1] + c[m-1]*q[2] + ... = 0
    # c[m+2] + c[m+1]*q[1] + c[m]*q[2] + ... = 0
    # etc.
    def c(k):
        if 0 <= k < len(coefs):
            return coefs[k]
        return 0.0

    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(n):
        for j in range(n):
            A[i][j] = c(m + i + 1 - (j + 1))
        b[i] = -c(m + i + 1)

    q_tail = lu_solve(Matrix(A), b)
    q = [1.0] + list(q_tail)

    # Compute numerator: p[k] = sum_{j=0}^{min(k,n)} q[j] * c[k-j]
    p = [0.0] * (m + 1)
    for k in range(m + 1):
        for j in range(min(k, n) + 1):
            p[k] += q[j] * c(k - j)

    return p, q


def pade_evaluate(p, q, x):
    """Evaluate Pade approximant p(x)/q(x)."""
    numer = sum(c * x**i for i, c in enumerate(p))
    denom = sum(c * x**i for i, c in enumerate(q))
    if abs(denom) < 1e-30:
        return float('inf') if numer >= 0 else float('-inf')
    return numer / denom


# ============================================================
# Least Squares Fitting
# ============================================================

def polyfit(xs, ys, degree):
    """Fit polynomial of given degree to data points (least squares)."""
    n = len(xs)
    m = degree + 1
    A = [[xs[i]**j for j in range(m)] for i in range(n)]
    coefs = lu_solve(Matrix(A), ys) if n == m else _least_squares_solve(A, ys)
    return coefs  # coefs[0] + coefs[1]*x + coefs[2]*x^2 + ...


def _least_squares_solve(A_data, b):
    """Solve least squares via normal equations."""
    A = Matrix(A_data)
    AtA = A.T.matmul(A)
    Atb_vec = [sum(A.data[j][i] * b[j] for j in range(A.rows)) for i in range(A.cols)]
    return lu_solve(AtA, Atb_vec)


def polyval(coefs, x):
    """Evaluate polynomial with given coefficients at x."""
    result = 0.0
    for i, c in enumerate(coefs):
        result += c * x**i
    return result


def exponential_fit(xs, ys):
    """Fit y = a * exp(b * x) to data. Returns (a, b)."""
    log_ys = [math.log(y) for y in ys if y > 0]
    if len(log_ys) != len(ys):
        raise ValueError("All y values must be positive for exponential fit")
    coefs = polyfit(xs, log_ys, 1)
    return math.exp(coefs[0]), coefs[1]


def power_fit(xs, ys):
    """Fit y = a * x^b to data. Returns (a, b)."""
    log_xs = [math.log(x) for x in xs if x > 0]
    log_ys = [math.log(y) for y in ys if y > 0]
    if len(log_xs) != len(xs) or len(log_ys) != len(ys):
        raise ValueError("All x and y values must be positive for power fit")
    coefs = polyfit(log_xs, log_ys, 1)
    return math.exp(coefs[0]), coefs[1]


# ============================================================
# Multi-dimensional Interpolation
# ============================================================

class BilinearInterpolator:
    """Bilinear interpolation on a regular grid."""

    def __init__(self, xs, ys_grid, zs):
        """
        xs: x coordinates (sorted)
        ys_grid: y coordinates (sorted)
        zs: 2D array zs[i][j] = f(xs[i], ys_grid[j])
        """
        self.xs = list(xs)
        self.ys = list(ys_grid)
        self.zs = [[float(v) for v in row] for row in zs]

    def __call__(self, x, y):
        return self.evaluate(x, y)

    def evaluate(self, x, y):
        xs, ys, zs = self.xs, self.ys, self.zs
        # Find indices
        ix = max(0, min(len(xs) - 2, _bisect(xs, x)))
        iy = max(0, min(len(ys) - 2, _bisect(ys, y)))
        x1, x2 = xs[ix], xs[ix + 1]
        y1, y2 = ys[iy], ys[iy + 1]
        tx = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
        ty = (y - y1) / (y2 - y1) if y2 != y1 else 0.0
        tx = max(0.0, min(1.0, tx))
        ty = max(0.0, min(1.0, ty))
        z00 = zs[ix][iy]
        z10 = zs[ix + 1][iy]
        z01 = zs[ix][iy + 1]
        z11 = zs[ix + 1][iy + 1]
        return z00 * (1 - tx) * (1 - ty) + z10 * tx * (1 - ty) + z01 * (1 - tx) * ty + z11 * tx * ty


class BicubicInterpolator:
    """Bicubic interpolation on a regular grid."""

    def __init__(self, xs, ys_grid, zs):
        self.xs = list(xs)
        self.ys = list(ys_grid)
        self.zs = [[float(v) for v in row] for row in zs]
        self.nx = len(xs)
        self.ny = len(ys_grid)

    def __call__(self, x, y):
        return self.evaluate(x, y)

    def evaluate(self, x, y):
        xs, ys, zs = self.xs, self.ys, self.zs
        ix = max(1, min(self.nx - 3, _bisect(xs, x)))
        iy = max(1, min(self.ny - 3, _bisect(ys, y)))

        # Interpolate along x for 4 y-rows
        tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix]) if xs[ix + 1] != xs[ix] else 0.0
        cols = []
        for jj in range(iy - 1, iy + 3):
            jj = max(0, min(self.ny - 1, jj))
            vals = [zs[max(0, min(self.nx - 1, ii))][jj] for ii in range(ix - 1, ix + 3)]
            cols.append(_cubic_interp(tx, vals))

        ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy]) if ys[iy + 1] != ys[iy] else 0.0
        return _cubic_interp(ty, cols)


def _cubic_interp(t, v):
    """Catmull-Rom cubic interpolation."""
    return (
        v[1] + 0.5 * t * (
            v[2] - v[0] + t * (
                2.0 * v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3] + t * (
                    -v[0] + 3.0 * v[1] - 3.0 * v[2] + v[3]
                )
            )
        )
    )


def _bisect(arr, x):
    """Binary search returning insertion index - 1."""
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid + 1] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ============================================================
# Trigonometric Interpolation
# ============================================================

def trig_interpolate(xs, ys, x):
    """Trigonometric interpolation for periodic data on [0, 2*pi)."""
    n = len(xs)
    result = 0.0
    for k in range(n):
        basis = 1.0
        for j in range(n):
            if j != k:
                numer = math.sin(0.5 * (x - xs[j]))
                denom = math.sin(0.5 * (xs[k] - xs[j]))
                if abs(denom) < 1e-30:
                    continue
                basis *= numer / denom
        result += ys[k] * basis
    return result


# ============================================================
# Monotone interpolation
# ============================================================

class MonotoneInterpolator:
    """Monotonicity-preserving piecewise cubic interpolation (Fritsch-Carlson)."""

    def __init__(self, xs, ys):
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("Need at least 2 points")
        pairs = sorted(zip(xs, ys))
        self.xs = [p[0] for p in pairs]
        self.ys = [p[1] for p in pairs]
        self.n = len(self.xs)
        self._compute_slopes()

    def _compute_slopes(self):
        xs, ys, n = self.xs, self.ys, self.n
        h = [xs[i + 1] - xs[i] for i in range(n - 1)]
        delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
        self.slopes = [0.0] * n

        if n == 2:
            self.slopes[0] = self.slopes[1] = delta[0]
            return

        # Initial slopes: average of adjacent secants
        self.slopes[0] = delta[0]
        self.slopes[-1] = delta[-1]
        for i in range(1, n - 1):
            self.slopes[i] = (delta[i - 1] + delta[i]) / 2.0

        # Fritsch-Carlson modification
        for i in range(n - 1):
            if abs(delta[i]) < 1e-30:
                self.slopes[i] = 0.0
                self.slopes[i + 1] = 0.0
            else:
                alpha = self.slopes[i] / delta[i]
                beta = self.slopes[i + 1] / delta[i]
                r2 = alpha**2 + beta**2
                if r2 > 9.0:
                    tau = 3.0 / math.sqrt(r2)
                    self.slopes[i] = tau * alpha * delta[i]
                    self.slopes[i + 1] = tau * beta * delta[i]

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        xs, ys = self.xs, self.ys
        n = self.n
        if x <= xs[0]:
            i = 0
        elif x >= xs[-1]:
            i = n - 2
        else:
            lo, hi = 0, n - 2
            while lo < hi:
                mid = (lo + hi) // 2
                if xs[mid + 1] < x:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h
        h00 = (1 + 2 * t) * (1 - t)**2
        h10 = t * (1 - t)**2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)
        return h00 * ys[i] + h10 * h * self.slopes[i] + h01 * ys[i + 1] + h11 * h * self.slopes[i + 1]


# ============================================================
# High-level interpolation API
# ============================================================

def interpolate(xs, ys, method='cubic', **kwargs):
    """
    Create an interpolator using the specified method.

    Methods: 'linear', 'cubic' (natural), 'cubic_clamped', 'cubic_not_a_knot',
             'pchip', 'akima', 'monotone', 'lagrange', 'newton', 'bspline'
    Returns a callable interpolator.
    """
    if method == 'linear':
        return LinearSpline(xs, ys)
    elif method == 'cubic':
        return CubicSpline(xs, ys, bc_type='natural')
    elif method == 'cubic_clamped':
        slopes = kwargs.get('clamped_slopes', (0.0, 0.0))
        return CubicSpline(xs, ys, bc_type='clamped', clamped_slopes=slopes)
    elif method == 'cubic_not_a_knot':
        return CubicSpline(xs, ys, bc_type='not-a-knot')
    elif method == 'pchip':
        return PchipInterpolator(xs, ys)
    elif method == 'akima':
        return AkimaInterpolator(xs, ys)
    elif method == 'monotone':
        return MonotoneInterpolator(xs, ys)
    elif method == 'lagrange':
        w = lagrange_weights(xs)
        return lambda x: barycentric_interpolate(xs, ys, w, x)
    elif method == 'newton':
        coefs = divided_differences(xs, ys)
        return lambda x: _newton_eval_partial(xs, coefs, x, len(coefs) - 1)
    elif method == 'bspline':
        degree = kwargs.get('degree', 3)
        return bspline_interpolate(xs, ys, degree)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
