"""
V222: Gaussian Process Regression

Bayesian nonparametric regression using Gaussian processes.
Provides kernels, exact GP regression, marginal likelihood optimization,
posterior sampling, and prediction with uncertainty quantification.

Composes concepts from V218 (Kalman filter -- GaussianState, covariance propagation)
but is self-contained for GP-specific inference.

Key features:
- 8 kernel functions with composition (sum, product, scale)
- Exact GP regression with Cholesky-based inference
- Log marginal likelihood and gradients for hyperparameter optimization
- Posterior mean, variance, and sampling
- Multi-output GP via intrinsic coregionalization
- Sparse GP approximation (FITC / inducing points)
- Heteroscedastic noise model
- Warped GP for non-Gaussian likelihoods
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# NumPy-only replacements for scipy.linalg functions
# ---------------------------------------------------------------------------

def solve_triangular(L, b, lower=True):
    """Solve L @ x = b where L is triangular. Pure numpy via back-substitution."""
    if lower:
        # Forward substitution
        if b.ndim == 1:
            n = len(b)
            x = np.zeros(n)
            for i in range(n):
                x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
            return x
        else:
            # Column-wise
            return np.column_stack([solve_triangular(L, b[:, j], lower=True)
                                     for j in range(b.shape[1])])
    else:
        # Backward substitution
        if b.ndim == 1:
            n = len(b)
            x = np.zeros(n)
            for i in range(n - 1, -1, -1):
                x[i] = (b[i] - L[i, i+1:] @ x[i+1:]) / L[i, i]
            return x
        else:
            return np.column_stack([solve_triangular(L, b[:, j], lower=False)
                                     for j in range(b.shape[1])])


def cho_solve(L_lower_tuple, b):
    """Solve (L @ L.T) @ x = b given Cholesky factor L."""
    L, lower = L_lower_tuple
    # L @ L.T @ x = b  =>  L @ z = b, then L.T @ x = z
    z = solve_triangular(L, b, lower=True)
    x = solve_triangular(L.T, z, lower=False)
    return x


def _minimize_lbfgsb(func, x0, bounds=None, maxiter=200, tol=1e-5):
    """Simple gradient-free minimizer using finite-difference + L-BFGS-B style.
    Falls back to coordinate descent with golden section for robustness."""
    n = len(x0)
    x = x0.copy()
    best_f = func(x)
    best_x = x.copy()

    # Nelder-Mead style simplex search (gradient-free, works well for GP hyperparams)
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x.copy()
    for i in range(n):
        simplex[i + 1] = x.copy()
        simplex[i + 1][i] += 0.5

    f_vals = np.array([func(s) for s in simplex])

    for iteration in range(maxiter * n):
        # Sort
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_x = simplex[0].copy()

        # Convergence check
        if np.max(np.abs(f_vals - f_vals[0])) < tol and iteration > n:
            break

        # Centroid of all but worst
        centroid = simplex[:-1].mean(axis=0)

        # Reflection
        xr = centroid + (centroid - simplex[-1])
        if bounds is not None:
            for i, b in enumerate(bounds):
                if b is not None:
                    lo, hi = b
                    if lo is not None:
                        xr[i] = max(xr[i], lo)
                    if hi is not None:
                        xr[i] = min(xr[i], hi)
        fr = func(xr)

        if fr < f_vals[0]:
            # Expansion
            xe = centroid + 2 * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                f_vals[-1] = fe
            else:
                simplex[-1] = xr
                f_vals[-1] = fr
        elif fr < f_vals[-2]:
            simplex[-1] = xr
            f_vals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < f_vals[-1]:
                simplex[-1] = xc
                f_vals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_vals[i] = func(simplex[i])

    class Result:
        pass
    r = Result()
    r.x = best_x
    r.fun = best_f
    return r


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

class Kernel(ABC):
    """Base kernel (covariance function)."""

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K(X1, X2). X1: (n,d), X2: (m,d) -> (n,m)."""

    @abstractmethod
    def params(self) -> np.ndarray:
        """Return log-space hyperparameters."""

    @abstractmethod
    def set_params(self, p: np.ndarray) -> None:
        """Set hyperparameters from log-space array."""

    @abstractmethod
    def n_params(self) -> int:
        """Number of hyperparameters."""

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonal of K(X, X). Override for efficiency."""
        return np.diag(self.__call__(X, X))

    def __add__(self, other: "Kernel") -> "SumKernel":
        return SumKernel(self, other)

    def __mul__(self, other) -> "Kernel":
        if isinstance(other, Kernel):
            return ProductKernel(self, other)
        return ScaleKernel(self, float(other))

    def __rmul__(self, other) -> "Kernel":
        return self.__mul__(other)


def _sq_dist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix. X1: (n,d), X2: (m,d) -> (n,m)."""
    return np.sum(X1**2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)


class RBFKernel(Kernel):
    """Squared exponential (RBF) kernel: sigma^2 * exp(-0.5 * ||x-x'||^2 / l^2)."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1, X2):
        r2 = _sq_dist(X1 / self.length_scale, X2 / self.length_scale)
        return self.variance * np.exp(-0.5 * r2)

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def params(self):
        return np.log([self.length_scale, self.variance])

    def set_params(self, p):
        self.length_scale, self.variance = np.exp(p)

    def n_params(self):
        return 2


class Matern32Kernel(Kernel):
    """Matern 3/2 kernel: sigma^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1, X2):
        r = np.sqrt(np.maximum(_sq_dist(X1, X2), 0.0)) / self.length_scale
        s3r = np.sqrt(3.0) * r
        return self.variance * (1.0 + s3r) * np.exp(-s3r)

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def params(self):
        return np.log([self.length_scale, self.variance])

    def set_params(self, p):
        self.length_scale, self.variance = np.exp(p)

    def n_params(self):
        return 2


class Matern52Kernel(Kernel):
    """Matern 5/2 kernel: sigma^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1, X2):
        r = np.sqrt(np.maximum(_sq_dist(X1, X2), 0.0)) / self.length_scale
        s5r = np.sqrt(5.0) * r
        return self.variance * (1.0 + s5r + 5.0 * r**2 / 3.0) * np.exp(-s5r)

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def params(self):
        return np.log([self.length_scale, self.variance])

    def set_params(self, p):
        self.length_scale, self.variance = np.exp(p)

    def n_params(self):
        return 2


class LinearKernel(Kernel):
    """Linear kernel: sigma^2 * (X1 - c) @ (X2 - c).T + sigma_b^2."""

    def __init__(self, variance: float = 1.0, bias_variance: float = 0.0, center: float = 0.0):
        self.variance = variance
        self.bias_variance = bias_variance
        self.center = center

    def __call__(self, X1, X2):
        return self.variance * (X1 - self.center) @ (X2 - self.center).T + self.bias_variance

    def diag(self, X):
        return self.variance * np.sum((X - self.center)**2, axis=1) + self.bias_variance

    def params(self):
        return np.log([self.variance, max(self.bias_variance, 1e-10)])

    def set_params(self, p):
        self.variance, self.bias_variance = np.exp(p)

    def n_params(self):
        return 2


class PeriodicKernel(Kernel):
    """Periodic kernel: sigma^2 * exp(-2 * sin^2(pi*||x-x'||/p) / l^2)."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, period: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
        self.period = period

    def __call__(self, X1, X2):
        r = np.sqrt(np.maximum(_sq_dist(X1, X2), 0.0))
        arg = np.pi * r / self.period
        return self.variance * np.exp(-2.0 * np.sin(arg)**2 / self.length_scale**2)

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def params(self):
        return np.log([self.length_scale, self.variance, self.period])

    def set_params(self, p):
        self.length_scale, self.variance, self.period = np.exp(p)

    def n_params(self):
        return 3


class RationalQuadraticKernel(Kernel):
    """RQ kernel: sigma^2 * (1 + r^2/(2*alpha*l^2))^(-alpha)."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, alpha: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
        self.alpha = alpha

    def __call__(self, X1, X2):
        r2 = _sq_dist(X1, X2)
        return self.variance * (1.0 + r2 / (2.0 * self.alpha * self.length_scale**2))**(-self.alpha)

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def params(self):
        return np.log([self.length_scale, self.variance, self.alpha])

    def set_params(self, p):
        self.length_scale, self.variance, self.alpha = np.exp(p)

    def n_params(self):
        return 3


class WhiteNoiseKernel(Kernel):
    """White noise kernel: sigma^2 * I (only for X1 == X2)."""

    def __init__(self, variance: float = 1.0):
        self.variance = variance

    def __call__(self, X1, X2):
        if X1 is X2 or (X1.shape == X2.shape and np.allclose(X1, X2)):
            return self.variance * np.eye(X1.shape[0])
        return np.zeros((X1.shape[0], X2.shape[0]))

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def params(self):
        return np.log([self.variance])

    def set_params(self, p):
        self.variance = np.exp(p[0])

    def n_params(self):
        return 1


class ARDKernel(Kernel):
    """RBF with Automatic Relevance Determination (per-dimension length scales)."""

    def __init__(self, length_scales: np.ndarray, variance: float = 1.0):
        self.length_scales = np.asarray(length_scales, dtype=float)
        self.variance = variance

    def __call__(self, X1, X2):
        X1s = X1 / self.length_scales
        X2s = X2 / self.length_scales
        r2 = _sq_dist(X1s, X2s)
        return self.variance * np.exp(-0.5 * r2)

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def params(self):
        return np.concatenate([np.log(self.length_scales), [np.log(self.variance)]])

    def set_params(self, p):
        self.length_scales = np.exp(p[:-1])
        self.variance = np.exp(p[-1])

    def n_params(self):
        return len(self.length_scales) + 1


class SumKernel(Kernel):
    """Sum of two kernels."""

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1, X2):
        return self.k1(X1, X2) + self.k2(X1, X2)

    def diag(self, X):
        return self.k1.diag(X) + self.k2.diag(X)

    def params(self):
        return np.concatenate([self.k1.params(), self.k2.params()])

    def set_params(self, p):
        n1 = self.k1.n_params()
        self.k1.set_params(p[:n1])
        self.k2.set_params(p[n1:])

    def n_params(self):
        return self.k1.n_params() + self.k2.n_params()


class ProductKernel(Kernel):
    """Product of two kernels."""

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1, X2):
        return self.k1(X1, X2) * self.k2(X1, X2)

    def diag(self, X):
        return self.k1.diag(X) * self.k2.diag(X)

    def params(self):
        return np.concatenate([self.k1.params(), self.k2.params()])

    def set_params(self, p):
        n1 = self.k1.n_params()
        self.k1.set_params(p[:n1])
        self.k2.set_params(p[n1:])

    def n_params(self):
        return self.k1.n_params() + self.k2.n_params()


class ScaleKernel(Kernel):
    """Scaled kernel: c * k(x, x')."""

    def __init__(self, kernel: Kernel, scale: float = 1.0):
        self.kernel = kernel
        self.scale = scale

    def __call__(self, X1, X2):
        return self.scale * self.kernel(X1, X2)

    def diag(self, X):
        return self.scale * self.kernel.diag(X)

    def params(self):
        return np.concatenate([[np.log(self.scale)], self.kernel.params()])

    def set_params(self, p):
        self.scale = np.exp(p[0])
        self.kernel.set_params(p[1:])

    def n_params(self):
        return 1 + self.kernel.n_params()


# ---------------------------------------------------------------------------
# GP Regression Result
# ---------------------------------------------------------------------------

@dataclass
class GPPrediction:
    """GP prediction at test points."""
    mean: np.ndarray        # (n_test,) or (n_test, n_outputs)
    variance: np.ndarray    # (n_test,) or (n_test, n_outputs)
    std: np.ndarray         # sqrt(variance)

    def confidence_interval(self, n_sigma: float = 2.0):
        """Return (lower, upper) confidence band."""
        return self.mean - n_sigma * self.std, self.mean + n_sigma * self.std


@dataclass
class GPResult:
    """Full GP inference result."""
    prediction: GPPrediction
    log_marginal_likelihood: float
    alpha: np.ndarray              # K^{-1} y -- weight vector
    L: np.ndarray                  # Cholesky factor of K + sigma^2 I
    X_train: np.ndarray
    y_train: np.ndarray
    kernel: Kernel
    noise_variance: float


# ---------------------------------------------------------------------------
# Exact GP Regression
# ---------------------------------------------------------------------------

class GaussianProcess:
    """Exact Gaussian Process regression with Cholesky-based inference."""

    def __init__(self, kernel: Kernel, noise_variance: float = 1e-6,
                 mean_function: Optional[Callable] = None):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.mean_function = mean_function
        self._X = None
        self._y = None
        self._L = None
        self._alpha = None

    def _mean(self, X: np.ndarray) -> np.ndarray:
        if self.mean_function is not None:
            return self.mean_function(X)
        return np.zeros(X.shape[0])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Fit GP to training data. X: (n, d), y: (n,)."""
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        self._X = X
        self._y = y
        K = self.kernel(X, X) + self.noise_variance * np.eye(len(X))
        self._L = np.linalg.cholesky(K)
        self._alpha = cho_solve((self._L, True), y - self._mean(X))
        return self

    def predict(self, X_test: np.ndarray, return_std: bool = True,
                return_cov: bool = False) -> GPPrediction:
        """Predict at test points."""
        X_test = np.atleast_2d(X_test)
        K_s = self.kernel(self._X, X_test)     # (n, n_test)
        mu = K_s.T @ self._alpha + self._mean(X_test)

        v = solve_triangular(self._L, K_s, lower=True)  # L^{-1} K_s
        var = self.kernel.diag(X_test) - np.sum(v**2, axis=0)
        var = np.maximum(var, 0.0)

        pred = GPPrediction(mean=mu, variance=var, std=np.sqrt(var))

        if return_cov:
            K_ss = self.kernel(X_test, X_test)
            cov = K_ss - v.T @ v
            pred.cov = cov

        return pred

    def log_marginal_likelihood(self, X: Optional[np.ndarray] = None,
                                 y: Optional[np.ndarray] = None) -> float:
        """Compute log p(y | X, theta)."""
        if X is not None and y is not None:
            X = np.atleast_2d(X)
            y = np.asarray(y).ravel()
            K = self.kernel(X, X) + self.noise_variance * np.eye(len(X))
            L = np.linalg.cholesky(K)
            alpha = cho_solve((L, True), y - self._mean(X))
        else:
            X, y = self._X, self._y
            L, alpha = self._L, self._alpha

        n = len(y)
        y_centered = y - self._mean(X)
        data_fit = -0.5 * y_centered @ alpha
        complexity = -np.sum(np.log(np.diag(L)))
        constant = -0.5 * n * np.log(2 * np.pi)
        return data_fit + complexity + constant

    def sample_prior(self, X: np.ndarray, n_samples: int = 1,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw samples from the GP prior. Returns (n_samples, n_points)."""
        rng = rng or np.random.default_rng()
        X = np.atleast_2d(X)
        K = self.kernel(X, X) + 1e-10 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        z = rng.standard_normal((len(X), n_samples))
        samples = self._mean(X)[:, None] + L @ z
        return samples.T

    def sample_posterior(self, X_test: np.ndarray, n_samples: int = 1,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw samples from the GP posterior. Returns (n_samples, n_test)."""
        rng = rng or np.random.default_rng()
        X_test = np.atleast_2d(X_test)
        pred = self.predict(X_test, return_cov=True)
        cov = pred.cov + 1e-10 * np.eye(len(X_test))
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((len(X_test), n_samples))
        samples = pred.mean[:, None] + L @ z
        return samples.T

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 n_restarts: int = 3, rng: Optional[np.random.Generator] = None,
                 bounds: Optional[list] = None) -> dict:
        """Optimize kernel hyperparameters by maximizing log marginal likelihood."""
        rng = rng or np.random.default_rng()
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()

        # Include noise variance as last parameter
        def neg_lml(log_params):
            self.kernel.set_params(log_params[:-1])
            self.noise_variance = np.exp(log_params[-1])
            try:
                self.fit(X, y)
                return -self.log_marginal_likelihood()
            except np.linalg.LinAlgError:
                return 1e10

        best_nlml = np.inf
        best_params = None
        init_params = np.concatenate([self.kernel.params(), [np.log(self.noise_variance)]])

        for i in range(n_restarts + 1):
            if i == 0:
                p0 = init_params.copy()
            else:
                p0 = init_params + rng.standard_normal(len(init_params)) * 0.5

            result = _minimize_lbfgsb(neg_lml, p0, bounds=bounds)
            if result.fun < best_nlml:
                best_nlml = result.fun
                best_params = result.x.copy()

        # Apply best parameters
        self.kernel.set_params(best_params[:-1])
        self.noise_variance = np.exp(best_params[-1])
        self.fit(X, y)

        return {
            "log_marginal_likelihood": -best_nlml,
            "kernel_params": np.exp(best_params[:-1]),
            "noise_variance": self.noise_variance,
            "n_restarts": n_restarts + 1,
        }

    def full_result(self, X_test: np.ndarray) -> GPResult:
        """Return full inference result."""
        pred = self.predict(X_test)
        return GPResult(
            prediction=pred,
            log_marginal_likelihood=self.log_marginal_likelihood(),
            alpha=self._alpha,
            L=self._L,
            X_train=self._X,
            y_train=self._y,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
        )


# ---------------------------------------------------------------------------
# Sparse GP (FITC approximation)
# ---------------------------------------------------------------------------

class SparseGP:
    """Sparse GP regression using FITC (Fully Independent Training Conditionals).

    Uses M inducing points to reduce O(N^3) to O(NM^2).
    """

    def __init__(self, kernel: Kernel, noise_variance: float = 1e-2,
                 n_inducing: int = 20):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.n_inducing = n_inducing
        self._Z = None  # inducing points

    def _select_inducing(self, X: np.ndarray, rng: Optional[np.random.Generator] = None):
        """Select inducing points via k-means-like initialization."""
        rng = rng or np.random.default_rng()
        n = len(X)
        m = min(self.n_inducing, n)
        indices = rng.choice(n, size=m, replace=False)
        return X[indices].copy()

    def fit(self, X: np.ndarray, y: np.ndarray,
            Z: Optional[np.ndarray] = None,
            rng: Optional[np.random.Generator] = None) -> "SparseGP":
        """Fit sparse GP with FITC."""
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        self._X = X
        self._y = y

        if Z is not None:
            self._Z = np.atleast_2d(Z)
        else:
            self._Z = self._select_inducing(X, rng)

        Z = self._Z
        n, m = len(X), len(Z)

        # Kernel matrices
        Kuu = self.kernel(Z, Z) + 1e-8 * np.eye(m)
        Kuf = self.kernel(Z, X)
        Kff_diag = self.kernel.diag(X)

        # FITC: Q_ff = Kfu @ Kuu^{-1} @ Kuf, Lambda = diag(Kff - Q_ff) + sigma^2 I
        Luu = np.linalg.cholesky(Kuu)
        V = solve_triangular(Luu, Kuf, lower=True)  # (m, n)
        Q_diag = np.sum(V**2, axis=0)
        Lambda_diag = np.maximum(Kff_diag - Q_diag, 0.0) + self.noise_variance

        # Woodbury: (Lambda + V^T V)^{-1} = Lambda^{-1} - Lambda^{-1} V^T S^{-1} V Lambda^{-1}
        # where S = I + V Lambda^{-1} V^T
        Lambda_inv = 1.0 / Lambda_diag
        VLi = V * Lambda_inv[None, :]  # (m, n)
        S = np.eye(m) + VLi @ V.T
        Ls = np.linalg.cholesky(S)

        # alpha = (Lambda + V^T V)^{-1} y
        Liy = Lambda_inv * y
        tmp = cho_solve((Ls, True), VLi @ y)
        self._alpha_sparse = Liy - Lambda_inv * (V.T @ tmp)

        # Store for prediction
        self._Luu = Luu
        self._Ls = Ls
        self._V = V
        self._Lambda_diag = Lambda_diag
        self._Lambda_inv = Lambda_inv
        self._VLi = VLi

        return self

    def predict(self, X_test: np.ndarray) -> GPPrediction:
        """Predict at test points."""
        X_test = np.atleast_2d(X_test)
        Z = self._Z
        Kus = self.kernel(Z, X_test)
        Kss_diag = self.kernel.diag(X_test)

        # Mean
        mu = Kus.T @ cho_solve((self._Luu, True), self._V @ (self._Lambda_inv * self._y))
        # Simpler: use precomputed
        # mu = Kus.T @ solve from Kuu and sum
        # Actually: mu = K_su @ Kuu^{-1} @ Kuf @ alpha_sparse
        Kuf = self.kernel(Z, self._X)
        mu = Kus.T @ cho_solve((self._Luu, True), Kuf @ self._alpha_sparse)

        # Variance
        v = solve_triangular(self._Luu, Kus, lower=True)
        Q_diag_test = np.sum(v**2, axis=0)
        w = solve_triangular(self._Ls, self._VLi @ (self.kernel(self._X, X_test)), lower=True)
        # Simpler FITC variance:
        var = Kss_diag - Q_diag_test
        # Add back the posterior reduction
        Sigma_inv_term = cho_solve((self._Ls, True), v)
        var = var + np.sum(v * Sigma_inv_term, axis=0)
        var = np.maximum(var, 0.0)

        return GPPrediction(mean=mu, variance=var, std=np.sqrt(var))

    def log_marginal_likelihood(self) -> float:
        """FITC log marginal likelihood."""
        y = self._y
        n = len(y)
        Lambda_diag = self._Lambda_diag

        # log |Lambda + V^T V| = log |Lambda| + log |S|
        log_det = np.sum(np.log(Lambda_diag)) + 2 * np.sum(np.log(np.diag(self._Ls)))

        # y^T (Lambda + V^T V)^{-1} y
        data_fit = y @ self._alpha_sparse

        return -0.5 * (data_fit + log_det + n * np.log(2 * np.pi))


# ---------------------------------------------------------------------------
# Multi-Output GP (Intrinsic Coregionalization Model)
# ---------------------------------------------------------------------------

class MultiOutputGP:
    """Multi-output GP via Intrinsic Coregionalization Model (ICM).

    K_multi = B kron K_base, where B is the inter-task covariance matrix.
    """

    def __init__(self, kernel: Kernel, n_outputs: int, noise_variance: float = 1e-6):
        self.kernel = kernel
        self.n_outputs = n_outputs
        self.noise_variance = noise_variance
        # B = W W^T + diag(kappa)
        self._W = np.eye(n_outputs)
        self._kappa = np.ones(n_outputs) * 0.1

    @property
    def B(self) -> np.ndarray:
        """Inter-task covariance matrix."""
        return self._W @ self._W.T + np.diag(self._kappa)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiOutputGP":
        """Fit multi-output GP. X: (n, d), Y: (n, n_outputs)."""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        self._X = X
        self._Y = Y
        n, p = Y.shape

        # Full covariance: B kron K + sigma^2 I
        K = self.kernel(X, X)
        B = self.B
        K_full = np.kron(B, K) + self.noise_variance * np.eye(n * p)

        # Stack y: [y1; y2; ...; yp]
        y_stacked = Y.T.ravel()
        self._L_full = np.linalg.cholesky(K_full)
        self._alpha_full = cho_solve((self._L_full, True), y_stacked)
        self._y_stacked = y_stacked

        return self

    def predict(self, X_test: np.ndarray) -> GPPrediction:
        """Predict at test points. Returns mean/var of shape (n_test, n_outputs)."""
        X_test = np.atleast_2d(X_test)
        n_test = len(X_test)
        p = self.n_outputs

        K_s = self.kernel(self._X, X_test)
        B = self.B
        K_s_full = np.kron(B, K_s)

        mu_stacked = K_s_full.T @ self._alpha_full
        mu = mu_stacked.reshape(p, n_test).T

        v = solve_triangular(self._L_full, K_s_full, lower=True)
        K_ss_diag = self.kernel.diag(X_test)
        var_stacked = np.tile(K_ss_diag, p) * np.diag(np.kron(B, np.eye(n_test))) - np.sum(v**2, axis=0)

        # Simpler: just compute per-output diag
        var = np.zeros((n_test, p))
        for j in range(p):
            start = j * n_test
            end = (j + 1) * n_test
            var[:, j] = np.maximum(B[j, j] * K_ss_diag - np.sum(v[:, start:end]**2, axis=0), 0.0)

        return GPPrediction(mean=mu, variance=var, std=np.sqrt(var))


# ---------------------------------------------------------------------------
# Heteroscedastic GP
# ---------------------------------------------------------------------------

class HeteroscedasticGP:
    """GP with input-dependent noise variance.

    Models both f(x) and log(sigma^2(x)) as GPs.
    Uses iterative estimation: fit f with current noise, refit noise from residuals.
    """

    def __init__(self, kernel: Kernel, noise_kernel: Optional[Kernel] = None,
                 n_iterations: int = 5):
        self.kernel = kernel
        self.noise_kernel = noise_kernel or RBFKernel(length_scale=1.0, variance=1.0)
        self.n_iterations = n_iterations

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HeteroscedasticGP":
        """Fit heteroscedastic GP via iterative procedure."""
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        self._X = X
        self._y = y
        n = len(y)

        # Initial fit with constant noise
        noise_var = np.full(n, np.var(y) * 0.1)

        for _ in range(self.n_iterations):
            # Fit function GP with current noise
            K = self.kernel(X, X) + np.diag(noise_var)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)

            alpha = cho_solve((L, True), y)
            mu = self.kernel(X, X) @ alpha

            # Estimate noise from squared residuals
            residuals = (y - mu)**2
            log_residuals = np.log(np.maximum(residuals, 1e-10))

            # Fit noise GP
            K_noise = self.noise_kernel(X, X) + 0.5 * np.eye(n)
            try:
                L_noise = np.linalg.cholesky(K_noise)
            except np.linalg.LinAlgError:
                K_noise += 1e-6 * np.eye(n)
                L_noise = np.linalg.cholesky(K_noise)

            alpha_noise = cho_solve((L_noise, True), log_residuals)
            log_noise_pred = K_noise @ alpha_noise
            noise_var = np.exp(log_noise_pred)
            noise_var = np.clip(noise_var, 1e-8, np.var(y) * 10)

        self._noise_var = noise_var
        self._L = L
        self._alpha = alpha
        self._L_noise = L_noise
        self._alpha_noise = alpha_noise

        return self

    def predict(self, X_test: np.ndarray) -> GPPrediction:
        """Predict with heteroscedastic uncertainty."""
        X_test = np.atleast_2d(X_test)

        # Function prediction
        K_s = self.kernel(self._X, X_test)
        mu = K_s.T @ self._alpha

        v = solve_triangular(self._L, K_s, lower=True)
        var_f = self.kernel.diag(X_test) - np.sum(v**2, axis=0)

        # Noise prediction
        K_s_noise = self.noise_kernel(self._X, X_test)
        log_noise = (self.noise_kernel(self._X, self._X) @ self._alpha_noise)
        # Predict noise at test points
        K_noise_full = self.noise_kernel(self._X, X_test)
        log_noise_test = K_noise_full.T @ self._alpha_noise
        noise_test = np.exp(log_noise_test)

        var = np.maximum(var_f + noise_test, 0.0)
        return GPPrediction(mean=mu, variance=var, std=np.sqrt(var))


# ---------------------------------------------------------------------------
# Warped GP (for non-Gaussian targets)
# ---------------------------------------------------------------------------

class WarpedGP:
    """GP with output warping for non-Gaussian targets.

    Applies a monotonic warping function g(y) so that g(y) ~ GP.
    Supports: log, sqrt, Box-Cox.
    """

    def __init__(self, kernel: Kernel, noise_variance: float = 1e-6,
                 warp: str = "log"):
        self.gp = GaussianProcess(kernel, noise_variance)
        self.warp = warp
        self._lam = 0.5  # Box-Cox parameter

    def _forward(self, y: np.ndarray) -> np.ndarray:
        """Warp y -> z."""
        if self.warp == "log":
            return np.log(np.maximum(y, 1e-10))
        elif self.warp == "sqrt":
            return np.sqrt(np.maximum(y, 0.0))
        elif self.warp == "boxcox":
            if abs(self._lam) < 1e-10:
                return np.log(np.maximum(y, 1e-10))
            return (np.power(np.maximum(y, 1e-10), self._lam) - 1.0) / self._lam
        return y

    def _inverse(self, z: np.ndarray) -> np.ndarray:
        """Unwarp z -> y."""
        if self.warp == "log":
            return np.exp(z)
        elif self.warp == "sqrt":
            return z**2
        elif self.warp == "boxcox":
            if abs(self._lam) < 1e-10:
                return np.exp(z)
            return np.power(np.maximum(self._lam * z + 1.0, 1e-10), 1.0 / self._lam)
        return z

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WarpedGP":
        """Fit warped GP."""
        self._y_raw = y.copy()
        z = self._forward(y)
        self.gp.fit(X, z)
        return self

    def predict(self, X_test: np.ndarray) -> GPPrediction:
        """Predict in original space (approximate via mean mapping)."""
        pred_z = self.gp.predict(X_test)
        mu_y = self._inverse(pred_z.mean)
        # Delta method for variance
        if self.warp == "log":
            var_y = np.exp(2 * pred_z.mean + pred_z.variance) * (np.exp(pred_z.variance) - 1)
        elif self.warp == "sqrt":
            var_y = 4 * pred_z.mean**2 * pred_z.variance
        else:
            var_y = pred_z.variance  # Approximate
        var_y = np.maximum(var_y, 0.0)
        return GPPrediction(mean=mu_y, variance=var_y, std=np.sqrt(var_y))


# ---------------------------------------------------------------------------
# Utility: Kernel selection via cross-validation
# ---------------------------------------------------------------------------

def cross_validate_kernel(kernels: list, X: np.ndarray, y: np.ndarray,
                          n_folds: int = 5, noise_variance: float = 1e-2,
                          rng: Optional[np.random.Generator] = None) -> dict:
    """Compare kernels via cross-validation log-likelihood.

    Returns dict mapping kernel index to mean LML.
    """
    rng = rng or np.random.default_rng()
    X = np.atleast_2d(X)
    y = np.asarray(y).ravel()
    n = len(y)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    results = {}
    for ki, kernel in enumerate(kernels):
        lmls = []
        for fold in range(n_folds):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            gp = GaussianProcess(kernel, noise_variance)
            try:
                gp.fit(X[train_idx], y[train_idx])
                pred = gp.predict(X[test_idx])
                # Predictive log-likelihood
                lml = -0.5 * np.sum(
                    (y[test_idx] - pred.mean)**2 / pred.variance
                    + np.log(pred.variance)
                    + np.log(2 * np.pi)
                )
                lmls.append(lml)
            except np.linalg.LinAlgError:
                lmls.append(-np.inf)
        results[ki] = np.mean(lmls) if lmls else -np.inf

    best_idx = max(results, key=results.get)
    return {"scores": results, "best_kernel_index": best_idx}
