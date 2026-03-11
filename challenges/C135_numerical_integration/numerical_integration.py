"""
C135: Numerical Integration
Composing C132 Linear Algebra for matrix operations in Gaussian quadrature.

Covers:
- Newton-Cotes: trapezoid, Simpson, Simpson 3/8, Boole
- Romberg integration (Richardson extrapolation)
- Gaussian quadrature: Gauss-Legendre, Gauss-Laguerre, Gauss-Hermite, Gauss-Chebyshev
- Adaptive methods: adaptive Simpson, Gauss-Kronrod (G7K15)
- Multi-dimensional: product rules, Monte Carlo, stratified sampling
- Special: improper integrals, oscillatory (Filon), Cauchy principal value
- Utilities: convergence estimation, error bounds
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C132_linear_algebra'))
from linear_algebra import Matrix


# ============================================================
# Newton-Cotes Quadrature
# ============================================================

def trapezoid(f, a, b, n=100):
    """Composite trapezoidal rule with n subintervals."""
    if n < 1:
        raise ValueError("n must be >= 1")
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h


def simpson(f, a, b, n=100):
    """Composite Simpson's 1/3 rule. n must be even."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            s += 2 * f(x)
        else:
            s += 4 * f(x)
    return s * h / 3


def simpson38(f, a, b, n=99):
    """Composite Simpson's 3/8 rule. n must be divisible by 3."""
    if n < 3:
        raise ValueError("n must be >= 3")
    while n % 3 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 3 == 0:
            s += 2 * f(x)
        else:
            s += 3 * f(x)
    return s * 3 * h / 8


def boole(f, a, b, n=100):
    """Composite Boole's rule. n must be divisible by 4."""
    if n < 4:
        raise ValueError("n must be >= 4")
    while n % 4 != 0:
        n += 1
    h = (b - a) / n
    s = 7 * (f(a) + f(b))
    for i in range(1, n):
        x = a + i * h
        r = i % 4
        if r == 0:
            s += 14 * f(x)
        elif r == 1 or r == 3:
            s += 32 * f(x)
        else:  # r == 2
            s += 12 * f(x)
    return s * 2 * h / 45


# ============================================================
# Romberg Integration
# ============================================================

def romberg(f, a, b, max_order=10, tol=1e-12):
    """Romberg integration using Richardson extrapolation.
    Returns (result, error_estimate, table).
    """
    R = [[0.0] * (max_order + 1) for _ in range(max_order + 1)]
    R[0][0] = 0.5 * (b - a) * (f(a) + f(b))

    for i in range(1, max_order + 1):
        n = 2 ** i
        h = (b - a) / n
        # Trapezoidal with 2^i points
        s = 0.0
        for k in range(1, n, 2):
            s += f(a + k * h)
        R[i][0] = 0.5 * R[i - 1][0] + s * h

        # Richardson extrapolation
        for j in range(1, i + 1):
            factor = 4 ** j
            R[i][j] = (factor * R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1)

        if i >= 2 and abs(R[i][i] - R[i - 1][i - 1]) < tol:
            return R[i][i], abs(R[i][i] - R[i - 1][i - 1]), R

    return R[max_order][max_order], abs(R[max_order][max_order] - R[max_order - 1][max_order - 1]), R


# ============================================================
# Gaussian Quadrature
# ============================================================

def _legendre_nodes_weights(n):
    """Compute Gauss-Legendre nodes and weights on [-1, 1] using eigenvalue method."""
    if n == 1:
        return [0.0], [2.0]

    # Golub-Welsch: eigenvalues of tridiagonal Jacobi matrix
    # For Legendre: alpha_i = 0, beta_i = i / sqrt(4*i^2 - 1)
    diag = [0.0] * n
    off_diag = [0.0] * (n - 1)
    for i in range(1, n):
        off_diag[i - 1] = i / math.sqrt(4.0 * i * i - 1.0)

    # Use QR algorithm for symmetric tridiagonal
    nodes, vecs = _symmetric_tridiag_eig(diag, off_diag, n)

    weights = [2.0 * vecs[i][0] ** 2 for i in range(n)]
    # Sort by node
    pairs = sorted(zip(nodes, weights))
    nodes = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    return nodes, weights


def _symmetric_tridiag_eig(diag, off_diag, n):
    """QR algorithm for symmetric tridiagonal matrix eigenvalues + first components of eigenvectors."""
    d = diag[:]
    e = off_diag[:] + [0.0]
    # Store first component of each eigenvector
    z = [0.0] * n
    z[0] = 1.0

    # Use implicit QL algorithm with shifts
    vecs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        vecs[i][i] = 1.0

    for l in range(n):
        max_iter = 100
        for _ in range(max_iter):
            # Find small off-diagonal element
            m = l
            while m < n - 1:
                if abs(e[m]) <= 1e-15 * (abs(d[m]) + abs(d[m + 1])):
                    break
                m += 1
            if m == l:
                break

            # QL iteration with implicit shift
            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = math.sqrt(g * g + 1.0)
            sign_g = 1.0 if g >= 0 else -1.0
            g = d[m] - d[l] + e[l] / (g + sign_g * r)

            s = 1.0
            c = 1.0
            p = 0.0

            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) >= abs(g):
                    c = g / f
                    r = math.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c = c * s
                else:
                    s = f / g
                    r = math.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s = s * c

                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                # Track eigenvectors
                for k in range(n):
                    t = vecs[i + 1][k]
                    vecs[i + 1][k] = s * vecs[i][k] + c * t
                    vecs[i][k] = c * vecs[i][k] - s * t

            d[l] -= p
            e[l] = g
            e[m] = 0.0

    # Return eigenvalues and first components
    result_vecs = []
    for i in range(n):
        result_vecs.append([vecs[i][k] for k in range(n)])

    # Return eigenvalues and the eigenvector rows
    return d, result_vecs


def gauss_legendre(f, a, b, n=5):
    """Gauss-Legendre quadrature with n points on [a, b]."""
    nodes, weights = _legendre_nodes_weights(n)
    # Map from [-1, 1] to [a, b]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    result = 0.0
    for i in range(n):
        x = mid + half * nodes[i]
        result += weights[i] * f(x)
    return result * half


def _laguerre_nodes_weights(n):
    """Gauss-Laguerre nodes and weights for integral of f(x)*exp(-x) on [0, inf)."""
    # Jacobi matrix for Laguerre: alpha_i = 2*i + 1, beta_i = i + 1
    diag = [2.0 * i + 1.0 for i in range(n)]
    off_diag = [float(i + 1) for i in range(n - 1)]

    nodes, vecs = _symmetric_tridiag_eig(diag, off_diag, n)
    weights = [vecs[i][0] ** 2 for i in range(n)]
    pairs = sorted(zip(nodes, weights))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def gauss_laguerre(f, n=5):
    """Gauss-Laguerre quadrature for integral of f(x)*exp(-x) on [0, inf).
    Integrates f(x)*exp(-x)dx. To integrate g(x)dx where g decays exponentially,
    pass f(x) = g(x)*exp(x).
    """
    nodes, weights = _laguerre_nodes_weights(n)
    result = 0.0
    for i in range(n):
        result += weights[i] * f(nodes[i])
    return result


def _hermite_nodes_weights(n):
    """Gauss-Hermite nodes and weights for integral of f(x)*exp(-x^2) on (-inf, inf)."""
    # Jacobi matrix for Hermite: alpha_i = 0, beta_i = sqrt(i/2)
    # Actually: beta_i = sqrt((i+1)/2) for the probabilist convention
    # For physicist: beta_i = sqrt(i/2) where i starts from 1
    diag = [0.0] * n
    off_diag = [math.sqrt(0.5 * (i + 1)) for i in range(n - 1)]

    nodes, vecs = _symmetric_tridiag_eig(diag, off_diag, n)
    weights = [math.sqrt(math.pi) * vecs[i][0] ** 2 for i in range(n)]
    pairs = sorted(zip(nodes, weights))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def gauss_hermite(f, n=5):
    """Gauss-Hermite quadrature for integral of f(x)*exp(-x^2) on (-inf, inf)."""
    nodes, weights = _hermite_nodes_weights(n)
    result = 0.0
    for i in range(n):
        result += weights[i] * f(nodes[i])
    return result


def gauss_chebyshev(f, n=5):
    """Gauss-Chebyshev quadrature for integral of f(x)/sqrt(1-x^2) on [-1, 1].
    Uses Chebyshev nodes: x_k = cos((2k-1)*pi/(2n)), weights = pi/n.
    """
    result = 0.0
    w = math.pi / n
    for k in range(1, n + 1):
        x = math.cos((2 * k - 1) * math.pi / (2 * n))
        result += f(x)
    return result * w


# ============================================================
# Adaptive Quadrature
# ============================================================

def adaptive_simpson(f, a, b, tol=1e-10, max_depth=50):
    """Adaptive Simpson's rule with error estimation."""
    whole = _simpson_basic(f, a, b)
    return _adaptive_simpson_recursive(f, a, b, tol, whole, max_depth)


def _simpson_basic(f, a, b):
    """Basic Simpson's rule on [a,b]."""
    m = 0.5 * (a + b)
    h = (b - a) / 6.0
    return h * (f(a) + 4 * f(m) + f(b))


def _adaptive_simpson_recursive(f, a, b, tol, whole, depth):
    """Recursive adaptive Simpson with Richardson error estimate."""
    m = 0.5 * (a + b)
    left = _simpson_basic(f, a, m)
    right = _simpson_basic(f, m, b)
    combined = left + right
    error = (combined - whole) / 15.0  # Richardson error estimate

    if depth <= 0 or abs(error) < tol:
        return combined + error  # Apply correction

    return (_adaptive_simpson_recursive(f, a, m, tol / 2, left, depth - 1) +
            _adaptive_simpson_recursive(f, m, b, tol / 2, right, depth - 1))


def gauss_kronrod(f, a, b, tol=1e-10, max_depth=30):
    """Adaptive Gauss-Kronrod G7K15 quadrature.
    Uses 7-point Gauss and 15-point Kronrod rules for error estimation.
    Returns (result, error_estimate).
    """
    result, error = _gauss_kronrod_recursive(f, a, b, tol, max_depth)
    return result, error


# G7K15 nodes and weights (on [-1, 1])
_K15_NODES = [
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
    -0.7415311855993945, -0.5860872354676911, -0.4058451513773972,
    -0.2077849550078985, 0.0,
    0.2077849550078985, 0.4058451513773972, 0.5860872354676911,
    0.7415311855993945, 0.8648644233597691, 0.9491079123427585,
    0.9914553711208126
]

_K15_WEIGHTS = [
    0.0229353220105292, 0.0630920926299786, 0.1047900103222502,
    0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
    0.2044329400752989, 0.2094821410847278,
    0.2044329400752989, 0.1903505780647854, 0.1690047266392679,
    0.1406532597155259, 0.1047900103222502, 0.0630920926299786,
    0.0229353220105292
]

_G7_WEIGHTS = [
    0.1294849661688697, 0.2797053914892767, 0.3818300505051189,
    0.4179591836734694,
    0.3818300505051189, 0.2797053914892767, 0.1294849661688697
]

# G7 uses nodes at indices 1, 3, 5, 7, 9, 11, 13 of K15
_G7_INDICES = [1, 3, 5, 7, 9, 11, 13]


def _gauss_kronrod_recursive(f, a, b, tol, depth):
    """Single interval G7K15 with adaptive subdivision."""
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    # Evaluate K15
    fvals = [f(mid + half * _K15_NODES[i]) for i in range(15)]

    # K15 result
    k15 = sum(fvals[i] * _K15_WEIGHTS[i] for i in range(15)) * half

    # G7 result (subset of K15 points)
    g7 = sum(fvals[_G7_INDICES[i]] * _G7_WEIGHTS[i] for i in range(7)) * half

    error = abs(k15 - g7)

    if depth <= 0 or error < tol:
        return k15, error

    # Subdivide
    left_val, left_err = _gauss_kronrod_recursive(f, a, mid, tol / 2, depth - 1)
    right_val, right_err = _gauss_kronrod_recursive(f, mid, b, tol / 2, depth - 1)
    return left_val + right_val, left_err + right_err


# ============================================================
# Multi-dimensional Integration
# ============================================================

def integrate_2d(f, ax, bx, ay, by, nx=50, ny=50):
    """2D integration using product trapezoidal rule.
    f(x, y) integrated over [ax,bx] x [ay,by].
    ay, by can be functions of x for non-rectangular regions.
    """
    hx = (bx - ax) / nx
    total = 0.0

    for i in range(nx + 1):
        x = ax + i * hx
        wx = 0.5 if (i == 0 or i == nx) else 1.0

        # Allow variable y limits
        y_lo = ay(x) if callable(ay) else ay
        y_hi = by(x) if callable(by) else by
        hy = (y_hi - y_lo) / ny

        for j in range(ny + 1):
            y = y_lo + j * hy
            wy = 0.5 if (j == 0 or j == ny) else 1.0
            total += wx * wy * f(x, y) * hy

    return total * hx


def integrate_nd(f, bounds, n_per_dim=10):
    """N-dimensional integration using product trapezoidal rule.
    f takes a list of coordinates. bounds = [(a1,b1), (a2,b2), ...].
    """
    dim = len(bounds)
    # Generate grid points
    grids = []
    steps = []
    for a_i, b_i in bounds:
        h = (b_i - a_i) / n_per_dim
        grids.append([a_i + k * h for k in range(n_per_dim + 1)])
        steps.append(h)

    # Iterate over all grid points
    total = 0.0
    indices = [0] * dim

    while True:
        # Compute weight and coordinate
        w = 1.0
        coords = []
        for d in range(dim):
            coords.append(grids[d][indices[d]])
            if indices[d] == 0 or indices[d] == n_per_dim:
                w *= 0.5
        total += w * f(coords)

        # Increment indices
        carry = True
        for d in range(dim - 1, -1, -1):
            if carry:
                indices[d] += 1
                if indices[d] > n_per_dim:
                    indices[d] = 0
                    carry = True
                else:
                    carry = False
        if carry:
            break

    for h in steps:
        total *= h
    return total


def monte_carlo(f, bounds, n_samples=10000, seed=None):
    """Monte Carlo integration.
    f takes a list of coordinates. bounds = [(a1,b1), ...].
    Returns (estimate, standard_error).
    """
    rng = random.Random(seed)
    dim = len(bounds)
    volume = 1.0
    for a_i, b_i in bounds:
        volume *= (b_i - a_i)

    values = []
    for _ in range(n_samples):
        point = [rng.uniform(a_i, b_i) for a_i, b_i in bounds]
        values.append(f(point))

    mean = sum(values) / n_samples
    variance = sum((v - mean) ** 2 for v in values) / (n_samples - 1) if n_samples > 1 else 0.0
    std_error = math.sqrt(variance / n_samples) * volume

    return mean * volume, std_error


def stratified_monte_carlo(f, bounds, n_per_stratum=10, strata_per_dim=5, seed=None):
    """Stratified Monte Carlo integration for variance reduction.
    Divides domain into strata and samples within each.
    Returns (estimate, standard_error).
    """
    rng = random.Random(seed)
    dim = len(bounds)

    # Create strata boundaries
    strata_bounds = []
    for a_i, b_i in bounds:
        h = (b_i - a_i) / strata_per_dim
        strata_bounds.append([(a_i + k * h, a_i + (k + 1) * h) for k in range(strata_per_dim)])

    volume = 1.0
    for a_i, b_i in bounds:
        volume *= (b_i - a_i)

    total_strata = strata_per_dim ** dim
    stratum_volume = volume / total_strata

    total_sum = 0.0
    total_var = 0.0

    # Iterate over all strata
    indices = [0] * dim
    while True:
        values = []
        for _ in range(n_per_stratum):
            point = [rng.uniform(strata_bounds[d][indices[d]][0],
                                 strata_bounds[d][indices[d]][1])
                     for d in range(dim)]
            values.append(f(point))

        stratum_mean = sum(values) / n_per_stratum
        total_sum += stratum_mean * stratum_volume

        if n_per_stratum > 1:
            var = sum((v - stratum_mean) ** 2 for v in values) / (n_per_stratum - 1)
            total_var += var / n_per_stratum * stratum_volume ** 2

        # Increment
        carry = True
        for d in range(dim - 1, -1, -1):
            if carry:
                indices[d] += 1
                if indices[d] >= strata_per_dim:
                    indices[d] = 0
                else:
                    carry = False
                    break
        if carry:
            break

    return total_sum, math.sqrt(total_var)


# ============================================================
# Special Integrals
# ============================================================

def improper_integral(f, a, b, tol=1e-8, n=20):
    """Handle improper integrals where a or b can be float('inf') or float('-inf').
    Uses substitution to map infinite limits to finite domain.
    """
    a_inf = (a == float('-inf'))
    b_inf = (b == float('inf'))

    if a_inf and b_inf:
        # Split at 0
        left, _ = improper_integral(f, float('-inf'), 0, tol, n)
        right, _ = improper_integral(f, 0, float('inf'), tol, n)
        return left + right, tol

    if b_inf:
        # Substitution: t = 1/(x-a+1), x = a + 1/t - 1, dx = -1/t^2
        def g(t):
            if t <= 1e-15:
                return 0.0
            x = a + 1.0 / t - 1.0
            return f(x) / (t * t)
        result = gauss_legendre(g, 0, 1, n)
        return result, tol

    if a_inf:
        # Substitution: t = 1/(b-x+1), x = b - 1/t + 1, dx = 1/t^2
        def g(t):
            if t <= 1e-15:
                return 0.0
            x = b - 1.0 / t + 1.0
            return f(x) / (t * t)
        result = gauss_legendre(g, 0, 1, n)
        return result, tol

    # Finite limits -- just use adaptive
    result = adaptive_simpson(f, a, b, tol)
    return result, tol


def oscillatory_filon(f_envelope, omega, a, b, n=100, kind='sin'):
    """Filon-type quadrature for integrals of f(x)*sin(omega*x) or f(x)*cos(omega*x).
    f_envelope is the non-oscillatory part.
    n must be even.
    """
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    theta = omega * h

    if abs(theta) < 1e-10:
        # Small theta: fall back to Simpson
        if kind == 'sin':
            return simpson(lambda x: f_envelope(x) * math.sin(omega * x), a, b, n)
        else:
            return simpson(lambda x: f_envelope(x) * math.cos(omega * x), a, b, n)

    # Filon coefficients
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    t2 = theta * theta
    t3 = t2 * theta

    alpha = (t2 + theta * sin_t * cos_t - 2 * sin_t * sin_t) / t3
    beta = 2 * (theta * (1 + cos_t * cos_t) - 2 * sin_t * cos_t) / t3
    gamma = 4 * (sin_t - theta * cos_t) / t3

    # Evaluate f at nodes
    x = [a + i * h for i in range(n + 1)]
    fvals = [f_envelope(xi) for xi in x]

    if kind == 'sin':
        # Even sum (C_2k)
        even_sum = sum(fvals[2 * i] * math.sin(omega * x[2 * i]) for i in range(n // 2 + 1))
        # Odd sum (C_2k+1)
        odd_sum = sum(fvals[2 * i + 1] * math.sin(omega * x[2 * i + 1]) for i in range(n // 2))

        result = h * (alpha * (fvals[0] * math.cos(omega * a) - fvals[n] * math.cos(omega * b))
                      + beta * even_sum + gamma * odd_sum)
    else:  # cos
        even_sum = sum(fvals[2 * i] * math.cos(omega * x[2 * i]) for i in range(n // 2 + 1))
        odd_sum = sum(fvals[2 * i + 1] * math.cos(omega * x[2 * i + 1]) for i in range(n // 2))

        result = h * (-alpha * (fvals[0] * math.sin(omega * a) - fvals[n] * math.sin(omega * b))
                      + beta * even_sum + gamma * odd_sum)

    return result


def cauchy_principal_value(f, a, b, c, n=20):
    """Cauchy principal value of integral of f(x)/(x-c) on [a,b] where a < c < b.
    Uses subtraction of singularity method:
    PV int f(x)/(x-c) dx = int [f(x)-f(c)]/(x-c) dx + f(c) * ln(|(b-c)/(a-c)|)
    """
    if c <= a or c >= b:
        raise ValueError("Singularity c must be strictly inside [a, b]")

    fc = f(c)

    def g(x):
        if abs(x - c) < 1e-15:
            # Use numerical derivative approximation
            h = 1e-8
            return (f(c + h) - f(c - h)) / (2 * h)
        return (f(x) - fc) / (x - c)

    regular_part = gauss_legendre(g, a, b, n)
    log_part = fc * math.log(abs((b - c) / (a - c)))

    return regular_part + log_part


# ============================================================
# Singularity handling
# ============================================================

def integrate_with_singularity(f, a, b, singularity, n=20, tol=1e-8):
    """Integrate f on [a,b] where f has an integrable singularity at the given point.
    Uses IMT (tanh-sinh) transformation for endpoint singularities.
    """
    if abs(singularity - a) < 1e-15:
        # Left endpoint singularity: use tanh-sinh
        return _tanh_sinh(f, a, b, n)
    elif abs(singularity - b) < 1e-15:
        # Right endpoint singularity: reverse
        return _tanh_sinh(lambda x: f(a + b - x), a, b, n)
    else:
        # Interior singularity: split
        left = _tanh_sinh(lambda x: f(a + (singularity - a) * x / (b - a) * (b - a) + a),
                          a, singularity, n)
        # Actually simpler: just split and handle each endpoint
        left = integrate_with_singularity(f, a, singularity, singularity, n, tol)
        right = integrate_with_singularity(f, singularity, b, singularity, n, tol)
        return left + right


def _tanh_sinh(f, a, b, n=20):
    """Tanh-sinh (double exponential) quadrature for endpoint singularities."""
    h = 4.0 / n  # Step size in t-space
    total = 0.0
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    for i in range(-n, n + 1):
        t = i * h
        # x = mid + half * tanh(pi/2 * sinh(t))
        sinh_t = math.sinh(t)
        u = 0.5 * math.pi * sinh_t
        if abs(u) > 20:
            continue
        tanh_u = math.tanh(u)
        x = mid + half * tanh_u

        if a <= x <= b:
            cosh_t = math.cosh(t)
            cosh_u = math.cosh(u)
            # Weight: h * pi/2 * cosh(t) / cosh^2(pi/2 * sinh(t))
            w = h * 0.5 * math.pi * cosh_t / (cosh_u * cosh_u)
            try:
                fval = f(x)
                if math.isfinite(fval):
                    total += w * fval * half
            except (ValueError, ZeroDivisionError, OverflowError):
                pass

    return total


# ============================================================
# Clenshaw-Curtis Quadrature
# ============================================================

def clenshaw_curtis(f, a, b, n=32):
    """Clenshaw-Curtis quadrature using Chebyshev points.
    Highly accurate for smooth functions.
    """
    if n < 2:
        n = 2

    # Chebyshev points (including endpoints): x_k = cos(k*pi/n), k=0..n
    theta = [math.pi * k / n for k in range(n + 1)]
    x_cheb = [math.cos(t) for t in theta]

    # Map to [a, b]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    # Evaluate function at Chebyshev points
    fvals = [f(mid + half * x_cheb[k]) for k in range(n + 1)]

    # Compute Chebyshev coefficients via DCT-I
    # a_j = (2/n) * sum_{k=0}^{n} '' f_k * cos(j*k*pi/n)
    # where '' means first and last terms halved
    coeffs = [0.0] * (n + 1)
    for j in range(n + 1):
        s = 0.0
        for k in range(n + 1):
            val = fvals[k] * math.cos(j * k * math.pi / n)
            if k == 0 or k == n:
                val *= 0.5
            s += val
        coeffs[j] = 2.0 * s / n
    # a_0 and a_n need halving (standard DCT-I normalization)
    coeffs[0] *= 0.5
    coeffs[n] *= 0.5

    # Integrate: integral of T_j(x) from -1 to 1 = 2/(1-j^2) for even j, 0 for odd j
    result = 0.0
    for j in range(0, n + 1, 2):
        if j == 0:
            integral_tj = 2.0
        else:
            integral_tj = 2.0 / (1.0 - j * j)
        result += coeffs[j] * integral_tj

    return result * half


# ============================================================
# Convergence and Error Estimation
# ============================================================

def convergence_order(f, a, b, method, n_values):
    """Estimate convergence order of a quadrature method.
    Returns list of (n, result, estimated_order) tuples.
    """
    results = []
    for n in n_values:
        results.append((n, method(f, a, b, n)))

    orders = []
    for i in range(len(results)):
        if i < 2:
            orders.append((results[i][0], results[i][1], None))
        else:
            # Richardson estimate: order = log(|I_{n-2} - I_{n-1}| / |I_{n-1} - I_n|) / log(r)
            e1 = abs(results[i - 2][1] - results[i - 1][1])
            e2 = abs(results[i - 1][1] - results[i][1])
            if e2 > 1e-16 and e1 > 1e-16:
                r = results[i - 1][0] / results[i][0]
                if r != 1 and r > 0:
                    order = math.log(e1 / e2) / math.log(results[i][0] / results[i - 1][0])
                    orders.append((results[i][0], results[i][1], order))
                else:
                    orders.append((results[i][0], results[i][1], None))
            else:
                orders.append((results[i][0], results[i][1], None))

    return orders


def error_bound_trapezoid(f_second_deriv_bound, a, b, n):
    """Error bound for composite trapezoidal rule.
    |E| <= (b-a)^3 / (12*n^2) * max|f''(x)|
    """
    return abs(b - a) ** 3 * f_second_deriv_bound / (12.0 * n * n)


def error_bound_simpson(f_fourth_deriv_bound, a, b, n):
    """Error bound for composite Simpson's rule.
    |E| <= (b-a)^5 / (180*n^4) * max|f^(4)(x)|
    """
    return abs(b - a) ** 5 * f_fourth_deriv_bound / (180.0 * n ** 4)


# ============================================================
# Composite Gauss-Legendre
# ============================================================

def composite_gauss_legendre(f, a, b, n_intervals=10, points_per_interval=5):
    """Composite Gauss-Legendre: divide [a,b] into n_intervals, apply GL on each."""
    h = (b - a) / n_intervals
    total = 0.0
    nodes, weights = _legendre_nodes_weights(points_per_interval)

    for i in range(n_intervals):
        ai = a + i * h
        bi = ai + h
        mid = 0.5 * (ai + bi)
        half = 0.5 * h
        for j in range(points_per_interval):
            x = mid + half * nodes[j]
            total += weights[j] * f(x) * half

    return total


# ============================================================
# Double Exponential (DE) Quadrature
# ============================================================

def double_exponential(f, a, b, n=50):
    """Double exponential (tanh-sinh) quadrature for general integrals.
    Extremely effective for endpoint singularities and smooth functions.
    """
    return _tanh_sinh(f, a, b, n)


# ============================================================
# Line Integral
# ============================================================

def line_integral(f, curve_x, curve_y, t_start, t_end, n=100):
    """Line integral of f along a parameterized curve (curve_x(t), curve_y(t)).
    Computes integral of f(x(t), y(t)) * |r'(t)| dt.
    """
    h = 1e-6

    def integrand(t):
        x = curve_x(t)
        y = curve_y(t)
        # Approximate derivatives
        dx = (curve_x(t + h) - curve_x(t - h)) / (2 * h)
        dy = (curve_y(t + h) - curve_y(t - h)) / (2 * h)
        speed = math.sqrt(dx * dx + dy * dy)
        return f(x, y) * speed

    return gauss_legendre(integrand, t_start, t_end, n)


# ============================================================
# Convenience: auto-integrate
# ============================================================

def integrate(f, a, b, tol=1e-10, method='auto'):
    """High-level integration interface.
    method: 'auto', 'trapezoid', 'simpson', 'romberg', 'gauss', 'adaptive', 'kronrod'
    For 'auto': uses Gauss-Kronrod adaptive.
    Returns result (or (result, error) for methods that provide error estimates).
    """
    if method == 'trapezoid':
        return trapezoid(f, a, b)
    elif method == 'simpson':
        return simpson(f, a, b)
    elif method == 'romberg':
        result, err, _ = romberg(f, a, b, tol=tol)
        return result, err
    elif method == 'gauss':
        return gauss_legendre(f, a, b, n=20)
    elif method == 'adaptive':
        return adaptive_simpson(f, a, b, tol=tol)
    elif method == 'kronrod':
        return gauss_kronrod(f, a, b, tol=tol)
    elif method == 'auto':
        return gauss_kronrod(f, a, b, tol=tol)
    else:
        raise ValueError(f"Unknown method: {method}")
