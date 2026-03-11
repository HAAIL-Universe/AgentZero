"""
C127: Convex Optimization

A complete convex optimization library featuring:
1. VectorOps -- basic vector/matrix operations (no numpy dependency)
2. GradientDescent -- steepest descent with backtracking line search
3. NewtonMethod -- Newton's method for unconstrained optimization
4. BFGS -- quasi-Newton method with BFGS Hessian approximation
5. ConjugateGradient -- nonlinear conjugate gradient (Fletcher-Reeves)
6. BarrierMethod -- interior point method for constrained optimization
7. AugmentedLagrangian -- equality-constrained optimization
8. ProximalGradient -- for composite problems (smooth + non-smooth)
9. ADMM -- alternating direction method of multipliers
10. QuadraticProgram -- QP solver via barrier method

Composes: C124 (LinearProgramming) concepts but standalone implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from enum import Enum
import math


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class OptStatus(Enum):
    OPTIMAL = "optimal"
    MAX_ITER = "max_iterations"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NOT_SOLVED = "not_solved"


@dataclass
class OptResult:
    status: OptStatus
    x: Optional[List[float]] = None
    objective: Optional[float] = None
    iterations: int = 0
    gradient_norm: float = 0.0
    history: List[float] = field(default_factory=list)

    def __repr__(self):
        if self.status == OptStatus.OPTIMAL:
            return f"OptResult(OPTIMAL, obj={self.objective:.6f}, iter={self.iterations})"
        return f"OptResult({self.status.value}, iter={self.iterations})"


# ---------------------------------------------------------------------------
# Vector/Matrix operations (no numpy)
# ---------------------------------------------------------------------------

class VectorOps:
    """Basic linear algebra without numpy."""

    @staticmethod
    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    @staticmethod
    def add(a, b):
        return [ai + bi for ai, bi in zip(a, b)]

    @staticmethod
    def sub(a, b):
        return [ai - bi for ai, bi in zip(a, b)]

    @staticmethod
    def scale(s, a):
        return [s * ai for ai in a]

    @staticmethod
    def norm(a):
        return math.sqrt(sum(ai * ai for ai in a))

    @staticmethod
    def zeros(n):
        return [0.0] * n

    @staticmethod
    def ones(n):
        return [1.0] * n

    @staticmethod
    def mat_vec(A, x):
        """Matrix-vector multiply: A @ x"""
        return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

    @staticmethod
    def mat_mat(A, B):
        """Matrix multiply: A @ B"""
        m = len(A)
        n = len(B[0])
        k = len(B)
        return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)] for i in range(m)]

    @staticmethod
    def transpose(A):
        if not A:
            return []
        m, n = len(A), len(A[0])
        return [[A[i][j] for i in range(m)] for j in range(n)]

    @staticmethod
    def identity(n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    @staticmethod
    def outer(a, b):
        """Outer product: a @ b^T"""
        return [[ai * bj for bj in b] for ai in a]

    @staticmethod
    def mat_add(A, B):
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def mat_scale(s, A):
        return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def solve_2x2(A, b):
        """Solve 2x2 linear system Ax = b."""
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if abs(det) < 1e-15:
            return None
        return [
            (A[1][1] * b[0] - A[0][1] * b[1]) / det,
            (A[0][0] * b[1] - A[1][0] * b[0]) / det,
        ]

    @staticmethod
    def solve_linear(A, b):
        """Solve Ax = b via Gaussian elimination with partial pivoting."""
        n = len(b)
        # Augmented matrix
        M = [row[:] + [b[i]] for i, row in enumerate(A)]

        for col in range(n):
            # Partial pivoting
            max_row = col
            max_val = abs(M[col][col])
            for row in range(col + 1, n):
                if abs(M[row][col]) > max_val:
                    max_val = abs(M[row][col])
                    max_row = row
            if max_val < 1e-15:
                return None  # singular
            M[col], M[max_row] = M[max_row], M[col]

            # Eliminate below
            pivot = M[col][col]
            for row in range(col + 1, n):
                factor = M[row][col] / pivot
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = M[i][n]
            for j in range(i + 1, n):
                s -= M[i][j] * x[j]
            if abs(M[i][i]) < 1e-15:
                return None
            x[i] = s / M[i][i]

        return x

    @staticmethod
    def cholesky(A):
        """Cholesky decomposition: A = L @ L^T. Returns L."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    if val <= 0:
                        return None  # not positive definite
                    L[i][j] = math.sqrt(val)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
        return L

    @staticmethod
    def solve_cholesky(L, b):
        """Solve L @ L^T @ x = b given Cholesky factor L."""
        n = len(b)
        # Forward: L @ y = b
        y = [0.0] * n
        for i in range(n):
            s = b[i] - sum(L[i][j] * y[j] for j in range(i))
            y[i] = s / L[i][i]
        # Backward: L^T @ x = y
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))
            x[i] = s / L[i][i]
        return x


V = VectorOps


# ---------------------------------------------------------------------------
# Numerical differentiation
# ---------------------------------------------------------------------------

def numerical_gradient(f, x, h=1e-7):
    """Central difference gradient."""
    n = len(x)
    grad = [0.0] * n
    for i in range(n):
        xp = x[:]
        xm = x[:]
        xp[i] += h
        xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2 * h)
    return grad


def numerical_hessian(f, x, h=1e-5):
    """Central difference Hessian."""
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp = x[:]
            xpm = x[:]
            xmp = x[:]
            xmm = x[:]
            xpp[i] += h; xpp[j] += h
            xpm[i] += h; xpm[j] -= h
            xmp[i] -= h; xmp[j] += h
            xmm[i] -= h; xmm[j] -= h
            H[i][j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
            H[j][i] = H[i][j]
    return H


# ---------------------------------------------------------------------------
# Backtracking line search (Armijo condition)
# ---------------------------------------------------------------------------

def backtracking_line_search(f, x, d, grad, alpha=1.0, beta=0.5, c1=1e-4, max_iter=50):
    """
    Armijo backtracking: find t such that f(x + t*d) <= f(x) + c1*t*grad^T*d
    """
    t = alpha
    fx = f(x)
    gd = V.dot(grad, d)
    if gd >= 0:
        return 0.0  # not a descent direction
    for _ in range(max_iter):
        x_new = V.add(x, V.scale(t, d))
        if f(x_new) <= fx + c1 * t * gd:
            return t
        t *= beta
    return t


# ---------------------------------------------------------------------------
# Gradient Descent
# ---------------------------------------------------------------------------

class GradientDescent:
    """
    Steepest descent with backtracking line search.

    Args:
        f: objective function
        grad_f: gradient function (or None for numerical)
        tol: convergence tolerance on gradient norm
        max_iter: maximum iterations
        lr: initial step size for line search
    """

    def __init__(self, f, grad_f=None, tol=1e-6, max_iter=1000, lr=1.0):
        self.f = f
        self.grad_f = grad_f
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr

    def solve(self, x0):
        x = x0[:]
        history = []

        for i in range(self.max_iter):
            fx = self.f(x)
            history.append(fx)

            g = self.grad_f(x) if self.grad_f else numerical_gradient(self.f, x)
            gn = V.norm(g)

            if gn < self.tol:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=i, gradient_norm=gn, history=history
                )

            d = V.scale(-1, g)
            t = backtracking_line_search(self.f, x, d, g, alpha=self.lr)
            if t == 0:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=i, gradient_norm=gn, history=history
                )
            x = V.add(x, V.scale(t, d))

        fx = self.f(x)
        g = self.grad_f(x) if self.grad_f else numerical_gradient(self.f, x)
        history.append(fx)
        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=fx,
            iterations=self.max_iter, gradient_norm=V.norm(g), history=history
        )


# ---------------------------------------------------------------------------
# Newton's Method
# ---------------------------------------------------------------------------

class NewtonMethod:
    """
    Newton's method for unconstrained optimization.

    Uses exact or numerical Hessian. Falls back to gradient descent
    step if Hessian is not positive definite.
    """

    def __init__(self, f, grad_f=None, hess_f=None, tol=1e-8, max_iter=100):
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, x0):
        x = x0[:]
        n = len(x)
        history = []

        for i in range(self.max_iter):
            fx = self.f(x)
            history.append(fx)

            g = self.grad_f(x) if self.grad_f else numerical_gradient(self.f, x)
            gn = V.norm(g)

            if gn < self.tol:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=i, gradient_norm=gn, history=history
                )

            H = self.hess_f(x) if self.hess_f else numerical_hessian(self.f, x)

            # Try Cholesky solve (works if H is PD)
            neg_g = V.scale(-1, g)
            L = V.cholesky(H)
            if L is not None:
                d = V.solve_cholesky(L, neg_g)
            else:
                # Regularize: H + lambda*I
                lam = 1e-3
                for _ in range(10):
                    H_reg = V.mat_add(H, V.mat_scale(lam, V.identity(n)))
                    L2 = V.cholesky(H_reg)
                    if L2 is not None:
                        d = V.solve_cholesky(L2, neg_g)
                        break
                    lam *= 10
                else:
                    d = neg_g  # fallback to gradient

            t = backtracking_line_search(self.f, x, d, g)
            if t == 0:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=i, gradient_norm=gn, history=history
                )
            x = V.add(x, V.scale(t, d))

        fx = self.f(x)
        g = self.grad_f(x) if self.grad_f else numerical_gradient(self.f, x)
        history.append(fx)
        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=fx,
            iterations=self.max_iter, gradient_norm=V.norm(g), history=history
        )


# ---------------------------------------------------------------------------
# BFGS (quasi-Newton)
# ---------------------------------------------------------------------------

class BFGS:
    """
    BFGS quasi-Newton method.

    Maintains an approximation to the inverse Hessian.
    """

    def __init__(self, f, grad_f=None, tol=1e-6, max_iter=500):
        self.f = f
        self.grad_f = grad_f
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, x0):
        x = x0[:]
        n = len(x)
        H_inv = V.identity(n)  # inverse Hessian approximation
        history = []

        g = self.grad_f(x) if self.grad_f else numerical_gradient(self.f, x)

        for i in range(self.max_iter):
            fx = self.f(x)
            history.append(fx)
            gn = V.norm(g)

            if gn < self.tol:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=i, gradient_norm=gn, history=history
                )

            # Search direction: d = -H_inv @ g
            d = V.scale(-1, V.mat_vec(H_inv, g))

            # Line search
            t = backtracking_line_search(self.f, x, d, g)
            if t == 0:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=i, gradient_norm=gn, history=history
                )

            x_new = V.add(x, V.scale(t, d))
            g_new = self.grad_f(x_new) if self.grad_f else numerical_gradient(self.f, x_new)

            # BFGS update
            s = V.sub(x_new, x)
            y = V.sub(g_new, g)
            sy = V.dot(s, y)

            if sy > 1e-10:
                rho = 1.0 / sy
                # H_inv = (I - rho*s*y^T) @ H_inv @ (I - rho*y*s^T) + rho*s*s^T
                Hy = V.mat_vec(H_inv, y)
                yHy = V.dot(y, Hy)

                # Sherman-Morrison-Woodbury form
                term1 = V.outer(s, s)
                term1_scaled = V.mat_scale((sy + yHy) * rho * rho, term1)

                sHy = V.outer(s, Hy)
                Hys = V.outer(Hy, s)
                term2 = V.mat_add(sHy, Hys)
                term2_scaled = V.mat_scale(-rho, term2)

                H_inv = V.mat_add(V.mat_add(H_inv, term1_scaled), term2_scaled)

            x = x_new
            g = g_new

        fx = self.f(x)
        history.append(fx)
        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=fx,
            iterations=self.max_iter, gradient_norm=V.norm(g), history=history
        )


# ---------------------------------------------------------------------------
# Conjugate Gradient (nonlinear, Fletcher-Reeves)
# ---------------------------------------------------------------------------

class ConjugateGradient:
    """Nonlinear conjugate gradient with Fletcher-Reeves formula."""

    def __init__(self, f, grad_f=None, tol=1e-6, max_iter=500):
        self.f = f
        self.grad_f = grad_f
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, x0):
        x = x0[:]
        history = []
        g = self.grad_f(x) if self.grad_f else numerical_gradient(self.f, x)
        d = V.scale(-1, g)
        g_norm_sq = V.dot(g, g)

        for i in range(self.max_iter):
            fx = self.f(x)
            history.append(fx)
            gn = math.sqrt(g_norm_sq)

            if gn < self.tol:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=i, gradient_norm=gn, history=history
                )

            t = backtracking_line_search(self.f, x, d, g)
            if t == 0:
                # restart with steepest descent
                d = V.scale(-1, g)
                t = backtracking_line_search(self.f, x, d, g)
                if t == 0:
                    return OptResult(
                        status=OptStatus.OPTIMAL, x=x, objective=fx,
                        iterations=i, gradient_norm=gn, history=history
                    )

            x = V.add(x, V.scale(t, d))
            g_new = self.grad_f(x) if self.grad_f else numerical_gradient(self.f, x)
            g_new_norm_sq = V.dot(g_new, g_new)

            # Fletcher-Reeves
            if g_norm_sq < 1e-30:
                beta = 0.0
            else:
                beta = g_new_norm_sq / g_norm_sq

            # Restart every n iterations
            n = len(x)
            if (i + 1) % n == 0:
                beta = 0.0

            d = V.add(V.scale(-1, g_new), V.scale(beta, d))
            g = g_new
            g_norm_sq = g_new_norm_sq

        fx = self.f(x)
        history.append(fx)
        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=fx,
            iterations=self.max_iter, gradient_norm=math.sqrt(g_norm_sq), history=history
        )


# ---------------------------------------------------------------------------
# Barrier Method (Interior Point) for constrained optimization
# ---------------------------------------------------------------------------

class BarrierMethod:
    """
    Log-barrier interior point method for:
      minimize   f(x)
      subject to g_i(x) <= 0   for i = 1..m

    Uses Newton's method as the inner solver.
    """

    def __init__(self, f, grad_f=None, hess_f=None,
                 constraints=None, constraint_grads=None,
                 tol=1e-6, max_iter=50, mu=10.0, t0=1.0):
        """
        Args:
            f: objective function
            constraints: list of g_i(x) functions, each <= 0 at feasible points
            constraint_grads: list of gradient functions for g_i
            mu: barrier parameter growth factor
            t0: initial barrier weight
        """
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.constraints = constraints or []
        self.constraint_grads = constraint_grads or []
        self.tol = tol
        self.max_iter = max_iter
        self.mu = mu
        self.t0 = t0

    def _barrier(self, x, t):
        """t*f(x) - sum(log(-g_i(x)))"""
        val = t * self.f(x)
        for gi in self.constraints:
            gv = gi(x)
            if gv >= 0:
                return float('inf')
            val -= math.log(-gv)
        return val

    def _is_feasible(self, x):
        return all(gi(x) < 0 for gi in self.constraints)

    def solve(self, x0):
        if not self._is_feasible(x0):
            return OptResult(status=OptStatus.INFEASIBLE, iterations=0)

        x = x0[:]
        m = len(self.constraints)
        t = self.t0
        history = []
        total_iter = 0

        for outer in range(self.max_iter):
            # Inner: minimize barrier function using Newton
            barrier_f = lambda xx, _t=t: self._barrier(xx, _t)
            newton = NewtonMethod(barrier_f, tol=self.tol / 10, max_iter=50)
            result = newton.solve(x)

            if result.x is not None and self._is_feasible(result.x):
                x = result.x
            total_iter += result.iterations

            fx = self.f(x)
            history.append(fx)

            # Duality gap
            if m > 0:
                gap = m / t
                if gap < self.tol:
                    return OptResult(
                        status=OptStatus.OPTIMAL, x=x, objective=fx,
                        iterations=total_iter, gradient_norm=gap, history=history
                    )

            t *= self.mu

        fx = self.f(x)
        history.append(fx)
        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=fx,
            iterations=total_iter, history=history
        )


# ---------------------------------------------------------------------------
# Augmented Lagrangian for equality constraints
# ---------------------------------------------------------------------------

class AugmentedLagrangian:
    """
    Augmented Lagrangian method for:
      minimize   f(x)
      subject to h_i(x) = 0   for i = 1..p

    Uses penalty term: (rho/2) * sum(h_i(x)^2) + sum(lambda_i * h_i(x))
    """

    def __init__(self, f, grad_f=None, eq_constraints=None, eq_grads=None,
                 tol=1e-6, max_iter=50, rho0=1.0, rho_max=1e6, rho_factor=10.0):
        self.f = f
        self.grad_f = grad_f
        self.eq_constraints = eq_constraints or []
        self.eq_grads = eq_grads or []
        self.tol = tol
        self.max_iter = max_iter
        self.rho0 = rho0
        self.rho_max = rho_max
        self.rho_factor = rho_factor

    def solve(self, x0, lam0=None):
        x = x0[:]
        p = len(self.eq_constraints)
        lam = lam0[:] if lam0 else [0.0] * p
        rho = self.rho0
        history = []
        total_iter = 0

        for outer in range(self.max_iter):
            # Augmented Lagrangian: f(x) + sum(lam_i * h_i(x)) + (rho/2)*sum(h_i(x)^2)
            def aug_lag(xx, _lam=lam[:], _rho=rho):
                val = self.f(xx)
                for i, hi in enumerate(self.eq_constraints):
                    hv = hi(xx)
                    val += _lam[i] * hv + (_rho / 2) * hv * hv
                return val

            # Inner solve with BFGS
            solver = BFGS(aug_lag, tol=self.tol / 10, max_iter=200)
            result = solver.solve(x)
            if result.x is not None:
                x = result.x
            total_iter += result.iterations

            fx = self.f(x)
            history.append(fx)

            # Check constraint violation
            h_vals = [hi(x) for hi in self.eq_constraints]
            violation = math.sqrt(sum(hv * hv for hv in h_vals)) if h_vals else 0.0

            if violation < self.tol:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=fx,
                    iterations=total_iter, gradient_norm=violation, history=history
                )

            # Update multipliers and penalty
            for i in range(p):
                lam[i] += rho * h_vals[i]
            rho = min(rho * self.rho_factor, self.rho_max)

        fx = self.f(x)
        history.append(fx)
        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=fx,
            iterations=total_iter, history=history
        )


# ---------------------------------------------------------------------------
# Proximal Gradient (for composite optimization: f(x) + g(x))
# ---------------------------------------------------------------------------

class ProximalGradient:
    """
    Proximal gradient method for:
      minimize   f(x) + g(x)
    where f is smooth and g has a known proximal operator.

    prox_g(x, t) = argmin_u { g(u) + (1/2t)||u - x||^2 }
    """

    def __init__(self, f, grad_f, prox_g, g=None, tol=1e-6, max_iter=1000, lr=1.0, accelerated=False):
        """
        Args:
            f: smooth part
            grad_f: gradient of f
            prox_g: proximal operator of g: prox_g(x, t) -> x
            g: non-smooth part (for objective evaluation, optional)
            accelerated: use FISTA (Nesterov acceleration)
        """
        self.f = f
        self.grad_f = grad_f
        self.prox_g = prox_g
        self.g = g if g else lambda x: 0.0
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.accelerated = accelerated

    def solve(self, x0):
        x = x0[:]
        y = x[:]  # for acceleration
        t_fista = 1.0
        history = []

        for i in range(self.max_iter):
            obj = self.f(x) + self.g(x)
            history.append(obj)

            point = y if self.accelerated else x
            g = self.grad_f(point)

            # Gradient step then proximal step
            z = V.sub(point, V.scale(self.lr, g))
            x_new = self.prox_g(z, self.lr)

            # Check convergence: ||x_new - x|| / max(1, ||x||) < tol
            diff = V.norm(V.sub(x_new, x))
            rel_diff = diff / max(1.0, V.norm(x))

            if rel_diff < self.tol:
                obj = self.f(x_new) + self.g(x_new)
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x_new, objective=obj,
                    iterations=i + 1, gradient_norm=diff, history=history
                )

            if self.accelerated:
                t_new = (1 + math.sqrt(1 + 4 * t_fista * t_fista)) / 2
                momentum = (t_fista - 1) / t_new
                y = V.add(x_new, V.scale(momentum, V.sub(x_new, x)))
                t_fista = t_new
            else:
                y = x_new

            x = x_new

        obj = self.f(x) + self.g(x)
        history.append(obj)
        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=obj,
            iterations=self.max_iter, history=history
        )


# ---------------------------------------------------------------------------
# ADMM (Alternating Direction Method of Multipliers)
# ---------------------------------------------------------------------------

class ADMM:
    """
    ADMM for:
      minimize  f(x) + g(z)
      subject to  Ax + Bz = c

    Simplest case: A=I, B=-I, c=0  (consensus)
      minimize  f(x) + g(z)
      subject to  x = z

    x-update: argmin_x { f(x) + (rho/2)||x - z + u||^2 }
    z-update: argmin_z { g(z) + (rho/2)||x - z + u||^2 }
    u-update: u = u + x - z
    """

    def __init__(self, x_update, z_update, rho=1.0, tol=1e-6, max_iter=500):
        """
        Args:
            x_update(z, u, rho) -> x: solve x subproblem
            z_update(x, u, rho) -> z: solve z subproblem
            rho: penalty parameter
        """
        self.x_update = x_update
        self.z_update = z_update
        self.rho = rho
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, x0, z0=None, u0=None):
        n = len(x0)
        x = x0[:]
        z = z0[:] if z0 else x0[:]
        u = u0[:] if u0 else V.zeros(n)
        history = []

        for i in range(self.max_iter):
            x = self.x_update(z, u, self.rho)
            z_new = self.z_update(x, u, self.rho)

            # Dual update
            u = V.add(u, V.sub(x, z_new))

            # Convergence: primal and dual residuals
            primal_res = V.norm(V.sub(x, z_new))
            dual_res = self.rho * V.norm(V.sub(z_new, z))
            z = z_new

            history.append(primal_res)

            if primal_res < self.tol and dual_res < self.tol:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x, objective=None,
                    iterations=i + 1, gradient_norm=primal_res, history=history
                )

        return OptResult(
            status=OptStatus.MAX_ITER, x=x, objective=None,
            iterations=self.max_iter, gradient_norm=primal_res, history=history
        )


# ---------------------------------------------------------------------------
# Quadratic Program solver
# ---------------------------------------------------------------------------

class QuadraticProgram:
    """
    Solve:
      minimize   (1/2) x^T Q x + c^T x
      subject to Ax <= b

    Uses barrier method internally.
    """

    def __init__(self, Q, c, A=None, b=None, tol=1e-6):
        """
        Args:
            Q: n x n positive semidefinite matrix
            c: n-vector
            A: m x n constraint matrix (optional)
            b: m-vector (optional)
        """
        self.Q = Q
        self.c = c
        self.n = len(c)
        self.A = A
        self.b = b
        self.tol = tol

    def _objective(self, x):
        Qx = V.mat_vec(self.Q, x)
        return 0.5 * V.dot(x, Qx) + V.dot(self.c, x)

    def _gradient(self, x):
        Qx = V.mat_vec(self.Q, x)
        return V.add(Qx, self.c)

    def _hessian(self, x):
        return [row[:] for row in self.Q]

    def solve(self, x0=None):
        if x0 is None:
            x0 = V.zeros(self.n)

        if self.A is None or not self.A:
            # Unconstrained QP: solve Qx = -c
            neg_c = V.scale(-1, self.c)
            x = V.solve_linear(self.Q, neg_c)
            if x is not None:
                return OptResult(
                    status=OptStatus.OPTIMAL, x=x,
                    objective=self._objective(x), iterations=1
                )
            # Q is singular, use Newton
            newton = NewtonMethod(self._objective, self._gradient, self._hessian, tol=self.tol)
            return newton.solve(x0)

        # Constrained: use barrier method
        constraints = []
        for i in range(len(self.b)):
            row = self.A[i]
            bi = self.b[i]
            constraints.append(lambda x, _r=row, _b=bi: V.dot(_r, x) - _b)

        # Find feasible starting point
        if not all(gi(x0) < 0 for gi in constraints):
            # Try to find feasible point by solving phase-1
            x0 = self._find_feasible(x0)
            if x0 is None:
                return OptResult(status=OptStatus.INFEASIBLE, iterations=0)

        barrier = BarrierMethod(
            self._objective, self._gradient, self._hessian,
            constraints=constraints, tol=self.tol, max_iter=100
        )
        return barrier.solve(x0)

    def _find_feasible(self, x0):
        """Find a strictly feasible point for Ax < b."""
        # Simple: start from x0, move toward interior
        x = x0[:]
        for _ in range(100):
            violations = []
            for i in range(len(self.b)):
                val = V.dot(self.A[i], x) - self.b[i]
                if val >= 0:
                    violations.append((i, val))
            if not violations:
                # Check strict feasibility
                if all(V.dot(self.A[i], x) - self.b[i] < -1e-10 for i in range(len(self.b))):
                    return x
            # Move away from most violated constraint
            for idx, val in violations:
                # Push x in direction -A[idx] (away from constraint boundary)
                step = V.scale(-(val + 0.1), self.A[idx])
                norm_sq = V.dot(self.A[idx], self.A[idx])
                if norm_sq > 1e-15:
                    step = V.scale(1.0 / norm_sq, step)
                    x = V.add(x, step)
        return None


# ---------------------------------------------------------------------------
# Lasso (L1-regularized least squares) via Proximal Gradient
# ---------------------------------------------------------------------------

class Lasso:
    """
    Solve:  minimize (1/2)||Ax - b||^2 + lambda * ||x||_1

    Uses proximal gradient (FISTA).
    """

    def __init__(self, A, b, lam=1.0, tol=1e-6, max_iter=1000):
        self.A = A
        self.b = b
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.m = len(A)
        self.n = len(A[0])

    def _f(self, x):
        """Smooth part: (1/2)||Ax - b||^2"""
        Ax = V.mat_vec(self.A, x)
        r = V.sub(Ax, self.b)
        return 0.5 * V.dot(r, r)

    def _grad_f(self, x):
        """Gradient: A^T(Ax - b)"""
        Ax = V.mat_vec(self.A, x)
        r = V.sub(Ax, self.b)
        At = V.transpose(self.A)
        return V.mat_vec(At, r)

    def _g(self, x):
        """Non-smooth part: lambda * ||x||_1"""
        return self.lam * sum(abs(xi) for xi in x)

    def _prox_g(self, x, t):
        """Proximal of lambda * ||.||_1: soft thresholding."""
        threshold = self.lam * t
        return [max(0, xi - threshold) - max(0, -xi - threshold) for xi in x]

    def solve(self, x0=None):
        if x0 is None:
            x0 = V.zeros(self.n)

        # Estimate step size: 1 / L where L = ||A^T A|| (spectral norm)
        # Simple estimate: use 1 / (max column norm squared)
        At = V.transpose(self.A)
        AtA = V.mat_mat(At, self.A)
        # Rough L estimate: trace(A^T A) as upper bound on spectral norm
        L = sum(AtA[i][i] for i in range(self.n))
        if L < 1e-10:
            L = 1.0
        lr = 1.0 / L

        pg = ProximalGradient(
            self._f, self._grad_f, self._prox_g, self._g,
            tol=self.tol, max_iter=self.max_iter, lr=lr, accelerated=True
        )
        return pg.solve(x0)


# ---------------------------------------------------------------------------
# Utility: standard test functions
# ---------------------------------------------------------------------------

def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx = -2*(1 - x[0]) + 100 * 2 * (x[1] - x[0]**2) * (-2*x[0])
    dy = 100 * 2 * (x[1] - x[0]**2)
    return [dx, dy]

def rosenbrock_hess(x):
    dxx = 2 + 100 * (8*x[0]*x[0] - 4*(x[1] - x[0]**2))
    dxy = -400 * x[0]
    dyy = 200
    return [[dxx, dxy], [dxy, dyy]]


def quadratic(Q, c):
    """Create quadratic function (1/2)x^T Q x + c^T x and its gradient."""
    def f(x):
        Qx = V.mat_vec(Q, x)
        return 0.5 * V.dot(x, Qx) + V.dot(c, x)
    def grad(x):
        Qx = V.mat_vec(Q, x)
        return V.add(Qx, c)
    return f, grad
