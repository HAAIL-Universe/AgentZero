"""
C155: Gaussian Processes
Composing C154 (Variational Inference) + C140 (Neural Networks)

Gaussian processes for regression, classification, and beyond.
Kernels, exact/sparse/variational inference, hyperparameter optimization.
"""

import math
import random
import sys
import os

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C154_variational_inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from variational_inference import (
    Normal, MultivariateNormal, KLDivergence, ELBO,
    MeanFieldVI, ReparameterizedVI, VIDiagnostics
)

# ============================================================
# Kernels
# ============================================================

class Kernel:
    """Base kernel class."""

    def __call__(self, X1, X2=None):
        return self.compute(X1, X2)

    def compute(self, X1, X2=None):
        raise NotImplementedError

    def diag(self, X):
        """Diagonal of K(X, X). Override for efficiency."""
        K = self.compute(X)
        return np.diag(K)

    def get_params(self):
        """Return dict of hyperparameters."""
        raise NotImplementedError

    def set_params(self, **params):
        """Set hyperparameters."""
        for k, v in params.items():
            setattr(self, k, v)

    def __add__(self, other):
        return SumKernel(self, other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ScaleKernel(self, other)
        return ProductKernel(self, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return ScaleKernel(self, other)
        return NotImplemented


def _ensure_2d(X):
    """Convert input to 2D numpy array."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _sq_dist(X1, X2):
    """Squared Euclidean distance matrix."""
    X1 = _ensure_2d(X1)
    X2 = _ensure_2d(X2)
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    return sq1 + sq2.T - 2.0 * X1 @ X2.T


class RBFKernel(Kernel):
    """Radial Basis Function (squared exponential) kernel.
    k(x, x') = variance * exp(-0.5 * ||x - x'||^2 / lengthscale^2)
    """

    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def compute(self, X1, X2=None):
        X1 = _ensure_2d(X1)
        X2 = X1 if X2 is None else _ensure_2d(X2)
        D = _sq_dist(X1, X2)
        return self.variance * np.exp(-0.5 * D / self.lengthscale ** 2)

    def diag(self, X):
        return np.full(len(_ensure_2d(X)), self.variance)

    def get_params(self):
        return {'lengthscale': self.lengthscale, 'variance': self.variance}


class MaternKernel(Kernel):
    """Matern kernel with nu in {0.5, 1.5, 2.5}.
    nu=0.5: exponential, nu=1.5: once-differentiable, nu=2.5: twice-differentiable.
    """

    def __init__(self, lengthscale=1.0, variance=1.0, nu=1.5):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be 0.5, 1.5, or 2.5")
        self.lengthscale = lengthscale
        self.variance = variance
        self.nu = nu

    def compute(self, X1, X2=None):
        X1 = _ensure_2d(X1)
        X2 = X1 if X2 is None else _ensure_2d(X2)
        D = np.sqrt(np.maximum(_sq_dist(X1, X2), 1e-36))
        r = D / self.lengthscale

        if self.nu == 0.5:
            K = np.exp(-r)
        elif self.nu == 1.5:
            s3 = math.sqrt(3.0) * r
            K = (1.0 + s3) * np.exp(-s3)
        else:  # 2.5
            s5 = math.sqrt(5.0) * r
            K = (1.0 + s5 + 5.0 / 3.0 * r ** 2) * np.exp(-s5)

        return self.variance * K

    def diag(self, X):
        return np.full(len(_ensure_2d(X)), self.variance)

    def get_params(self):
        return {'lengthscale': self.lengthscale, 'variance': self.variance, 'nu': self.nu}


class LinearKernel(Kernel):
    """Linear kernel: k(x, x') = variance * (x - center)^T (x' - center) + bias.
    """

    def __init__(self, variance=1.0, bias=0.0, center=0.0):
        self.variance = variance
        self.bias = bias
        self.center = center

    def compute(self, X1, X2=None):
        X1 = _ensure_2d(X1)
        X2 = X1 if X2 is None else _ensure_2d(X2)
        X1c = X1 - self.center
        X2c = X2 - self.center
        return self.variance * (X1c @ X2c.T) + self.bias

    def diag(self, X):
        X = _ensure_2d(X)
        Xc = X - self.center
        return self.variance * np.sum(Xc ** 2, axis=1) + self.bias

    def get_params(self):
        return {'variance': self.variance, 'bias': self.bias, 'center': self.center}


class PeriodicKernel(Kernel):
    """Periodic kernel: k(x, x') = variance * exp(-2 sin^2(pi |x-x'| / period) / lengthscale^2).
    """

    def __init__(self, lengthscale=1.0, variance=1.0, period=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period

    def compute(self, X1, X2=None):
        X1 = _ensure_2d(X1)
        X2 = X1 if X2 is None else _ensure_2d(X2)
        D = np.sqrt(np.maximum(_sq_dist(X1, X2), 1e-36))
        sin_term = np.sin(math.pi * D / self.period)
        return self.variance * np.exp(-2.0 * sin_term ** 2 / self.lengthscale ** 2)

    def diag(self, X):
        return np.full(len(_ensure_2d(X)), self.variance)

    def get_params(self):
        return {'lengthscale': self.lengthscale, 'variance': self.variance, 'period': self.period}


class PolynomialKernel(Kernel):
    """Polynomial kernel: k(x, x') = (variance * x^T x' + bias)^degree.
    """

    def __init__(self, degree=2, variance=1.0, bias=1.0):
        self.degree = degree
        self.variance = variance
        self.bias = bias

    def compute(self, X1, X2=None):
        X1 = _ensure_2d(X1)
        X2 = X1 if X2 is None else _ensure_2d(X2)
        return (self.variance * (X1 @ X2.T) + self.bias) ** self.degree

    def diag(self, X):
        X = _ensure_2d(X)
        return (self.variance * np.sum(X ** 2, axis=1) + self.bias) ** self.degree

    def get_params(self):
        return {'degree': self.degree, 'variance': self.variance, 'bias': self.bias}


class SumKernel(Kernel):
    """Sum of two kernels: k(x, x') = k1(x, x') + k2(x, x')."""

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def compute(self, X1, X2=None):
        return self.k1.compute(X1, X2) + self.k2.compute(X1, X2)

    def diag(self, X):
        return self.k1.diag(X) + self.k2.diag(X)

    def get_params(self):
        return {'k1': self.k1.get_params(), 'k2': self.k2.get_params()}


class ProductKernel(Kernel):
    """Product of two kernels: k(x, x') = k1(x, x') * k2(x, x')."""

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def compute(self, X1, X2=None):
        return self.k1.compute(X1, X2) * self.k2.compute(X1, X2)

    def diag(self, X):
        return self.k1.diag(X) * self.k2.diag(X)

    def get_params(self):
        return {'k1': self.k1.get_params(), 'k2': self.k2.get_params()}


class ScaleKernel(Kernel):
    """Scaled kernel: k(x, x') = scale * k_base(x, x')."""

    def __init__(self, base_kernel, scale=1.0):
        self.base_kernel = base_kernel
        self.scale = scale

    def compute(self, X1, X2=None):
        return self.scale * self.base_kernel.compute(X1, X2)

    def diag(self, X):
        return self.scale * self.base_kernel.diag(X)

    def get_params(self):
        return {'scale': self.scale, 'base': self.base_kernel.get_params()}


class ARDKernel(Kernel):
    """Automatic Relevance Determination kernel (RBF with per-dimension lengthscales).
    k(x, x') = variance * exp(-0.5 * sum_d (x_d - x'_d)^2 / l_d^2)
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=None):
        self.input_dim = input_dim
        self.variance = variance
        self.lengthscales = np.ones(input_dim) if lengthscales is None else np.asarray(lengthscales, dtype=np.float64)

    def compute(self, X1, X2=None):
        X1 = _ensure_2d(X1)
        X2 = X1 if X2 is None else _ensure_2d(X2)
        X1s = X1 / self.lengthscales
        X2s = X2 / self.lengthscales
        sq1 = np.sum(X1s ** 2, axis=1, keepdims=True)
        sq2 = np.sum(X2s ** 2, axis=1, keepdims=True)
        D = sq1 + sq2.T - 2.0 * X1s @ X2s.T
        return self.variance * np.exp(-0.5 * D)

    def diag(self, X):
        return np.full(len(_ensure_2d(X)), self.variance)

    def get_params(self):
        return {'variance': self.variance, 'lengthscales': self.lengthscales.tolist()}


# ============================================================
# Utility functions
# ============================================================

def _jitter_cholesky(K, jitter=1e-6, max_tries=6):
    """Cholesky factorization with increasing jitter for numerical stability."""
    n = K.shape[0]
    for i in range(max_tries):
        try:
            Kj = K + (jitter * (10 ** i)) * np.eye(n)
            L = np.linalg.cholesky(Kj)
            return L
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky failed even with large jitter")


# ============================================================
# Exact GP Regression
# ============================================================

class GPRegression:
    """Exact Gaussian Process regression.

    p(f* | X*, X, y) = N(mu*, Sigma*)
    mu* = K(X*, X) [K(X, X) + sigma_n^2 I]^{-1} y
    Sigma* = K(X*, X*) - K(X*, X) [K(X, X) + sigma_n^2 I]^{-1} K(X, X*)
    """

    def __init__(self, kernel, noise_variance=1e-2):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self._L = None
        self._alpha = None

    def fit(self, X, y):
        """Fit the GP to training data."""
        self.X_train = _ensure_2d(X)
        self.y_train = np.asarray(y, dtype=np.float64).ravel()
        n = len(self.y_train)

        K = self.kernel.compute(self.X_train)
        K += self.noise_variance * np.eye(n)
        self._L = _jitter_cholesky(K)
        self._alpha = cho_solve((self._L, True), self.y_train)
        return self

    def predict(self, X_test, return_std=False, return_cov=False):
        """Predict at test points.

        Returns:
            mu: mean predictions
            std: standard deviations (if return_std=True)
            cov: full covariance (if return_cov=True)
        """
        X_test = _ensure_2d(X_test)
        K_star = self.kernel.compute(X_test, self.X_train)
        mu = K_star @ self._alpha

        if not return_std and not return_cov:
            return mu

        v = solve_triangular(self._L, K_star.T, lower=True)

        if return_cov:
            K_ss = self.kernel.compute(X_test)
            cov = K_ss - v.T @ v
            return mu, cov

        K_ss_diag = self.kernel.diag(X_test)
        var = K_ss_diag - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)

    def log_marginal_likelihood(self):
        """Compute log marginal likelihood: log p(y | X, theta).
        = -0.5 y^T alpha - sum(log(diag(L))) - n/2 log(2pi)
        """
        n = len(self.y_train)
        log_det = np.sum(np.log(np.diag(self._L)))
        data_fit = -0.5 * self.y_train @ self._alpha
        complexity = -log_det
        constant = -0.5 * n * np.log(2 * np.pi)
        return data_fit + complexity + constant

    def sample_prior(self, X, n_samples=5, rng=None):
        """Draw samples from the GP prior."""
        if rng is None:
            rng = np.random.RandomState()
        X = _ensure_2d(X)
        K = self.kernel.compute(X) + 1e-6 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        samples = []
        for _ in range(n_samples):
            z = rng.randn(len(X))
            samples.append(L @ z)
        return samples

    def sample_posterior(self, X_test, n_samples=5, rng=None):
        """Draw samples from the GP posterior."""
        if rng is None:
            rng = np.random.RandomState()
        mu, cov = self.predict(X_test, return_cov=True)
        cov += 1e-6 * np.eye(len(mu))
        L = np.linalg.cholesky(cov)
        samples = []
        for _ in range(n_samples):
            z = rng.randn(len(mu))
            samples.append(mu + L @ z)
        return samples


# ============================================================
# Sparse GP (FITC / VFE)
# ============================================================

class SparseGP:
    """Sparse GP using inducing points.

    Supports FITC (Fully Independent Training Conditional) and
    VFE (Variational Free Energy) approximations.
    """

    def __init__(self, kernel, noise_variance=1e-2, method='vfe'):
        if method not in ('fitc', 'vfe'):
            raise ValueError("method must be 'fitc' or 'vfe'")
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.method = method
        self.X_train = None
        self.y_train = None
        self.Z = None  # inducing points
        self._cache = {}

    def fit(self, X, y, Z=None, n_inducing=None, rng=None):
        """Fit sparse GP.

        Args:
            X: training inputs
            y: training targets
            Z: inducing points (if None, selected from X)
            n_inducing: number of inducing points (default: min(20, n))
        """
        self.X_train = _ensure_2d(X)
        self.y_train = np.asarray(y, dtype=np.float64).ravel()
        n = len(self.y_train)

        if Z is not None:
            self.Z = _ensure_2d(Z)
        else:
            if rng is None:
                rng = np.random.RandomState(42)
            m = n_inducing or min(20, n)
            idx = rng.choice(n, size=min(m, n), replace=False)
            self.Z = self.X_train[idx].copy()

        self._compute_cache()
        return self

    def _compute_cache(self):
        """Precompute quantities for predictions."""
        n = len(self.y_train)
        m = len(self.Z)
        sigma2 = self.noise_variance

        Kuu = self.kernel.compute(self.Z) + 1e-6 * np.eye(m)
        Kuf = self.kernel.compute(self.Z, self.X_train)
        Kff_diag = self.kernel.diag(self.X_train)

        Luu = np.linalg.cholesky(Kuu)
        # V = Luu^{-1} Kuf
        V = solve_triangular(Luu, Kuf, lower=True)
        Qff_diag = np.sum(V ** 2, axis=0)

        if self.method == 'fitc':
            # FITC: Lambda = diag(Kff - Qff) + sigma^2 I
            Lambda_diag = np.maximum(Kff_diag - Qff_diag, 1e-10) + sigma2
        else:
            # VFE: Lambda = sigma^2 I (trace term handled in bound)
            Lambda_diag = np.full(n, sigma2)

        Lambda_inv = 1.0 / Lambda_diag

        # Sigma = Kuu + Kuf Lambda^{-1} Kfu
        S = Kuu + (Kuf * Lambda_inv[None, :]) @ Kuf.T
        Ls = _jitter_cholesky(S)

        # alpha = Sigma^{-1} Kuf Lambda^{-1} y
        b = (Kuf * Lambda_inv[None, :]) @ self.y_train
        alpha = cho_solve((Ls, True), b)

        self._cache = {
            'Kuu': Kuu, 'Kuf': Kuf, 'Luu': Luu, 'V': V, 'Ls': Ls,
            'alpha': alpha, 'Lambda_diag': Lambda_diag,
            'Qff_diag': Qff_diag, 'Kff_diag': Kff_diag,
        }

    def predict(self, X_test, return_std=False, return_cov=False):
        """Predict at test points."""
        X_test = _ensure_2d(X_test)
        Kus = self.kernel.compute(self.Z, X_test)
        mu = Kus.T @ self._cache['alpha']

        if not return_std and not return_cov:
            return mu

        Luu = self._cache['Luu']
        Ls = self._cache['Ls']
        v = solve_triangular(Luu, Kus, lower=True)
        w = solve_triangular(Ls, Kus, lower=True)

        if return_cov:
            Kss = self.kernel.compute(X_test)
            cov = Kss - v.T @ v + w.T @ w
            return mu, cov

        Kss_diag = self.kernel.diag(X_test)
        var = Kss_diag - np.sum(v ** 2, axis=0) + np.sum(w ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)

    def log_marginal_likelihood(self):
        """Compute (approximate) log marginal likelihood."""
        n = len(self.y_train)
        Lambda_diag = self._cache['Lambda_diag']
        Ls = self._cache['Ls']
        alpha = self._cache['alpha']

        # Data fit via Woodbury
        Lambda_inv = 1.0 / Lambda_diag
        Kuf = self._cache['Kuf']
        b = (Kuf * Lambda_inv[None, :]) @ self.y_train
        data_fit = -0.5 * np.sum(Lambda_inv * self.y_train ** 2) + 0.5 * b @ cho_solve((Ls, True), b)

        # Complexity
        log_det = np.sum(np.log(Lambda_diag)) + 2 * np.sum(np.log(np.diag(Ls))) - 2 * np.sum(np.log(np.diag(self._cache['Luu'])))
        complexity = -0.5 * log_det

        constant = -0.5 * n * np.log(2 * np.pi)

        lml = data_fit + complexity + constant

        if self.method == 'vfe':
            # VFE adds trace penalty: -0.5/sigma^2 * trace(Kff - Qff)
            trace_term = np.sum(self._cache['Kff_diag'] - self._cache['Qff_diag'])
            lml -= 0.5 * trace_term / self.noise_variance

        return lml


# ============================================================
# GP Classification (Laplace Approximation)
# ============================================================

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class GPClassification:
    """Binary GP classification with Laplace approximation.

    Uses logistic likelihood p(y=1|f) = sigma(f).
    Laplace approximation finds mode of p(f|X,y) then approximates
    with a Gaussian.
    """

    def __init__(self, kernel, max_iter=50, tol=1e-6):
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.X_train = None
        self.y_train = None
        self._f_hat = None
        self._W = None
        self._L = None
        self._K = None

    def fit(self, X, y):
        """Fit GP classifier. y should be 0/1 labels."""
        self.X_train = _ensure_2d(X)
        self.y_train = np.asarray(y, dtype=np.float64).ravel()
        n = len(self.y_train)

        K = self.kernel.compute(self.X_train)
        self._K = K

        # Newton's method to find posterior mode
        f = np.zeros(n)
        for _ in range(self.max_iter):
            pi = _sigmoid(f)
            W = pi * (1 - pi)
            W = np.maximum(W, 1e-10)  # numerical stability
            W_sqrt = np.sqrt(W)

            B = np.eye(n) + np.outer(W_sqrt, W_sqrt) * K
            L = _jitter_cholesky(B)

            grad = self.y_train - pi
            b = W * f + grad
            a = b - W_sqrt * cho_solve((L, True), (W_sqrt * (K @ b)))
            f_new = K @ a

            if np.max(np.abs(f_new - f)) < self.tol:
                f = f_new
                break
            f = f_new

        self._f_hat = f
        pi = _sigmoid(f)
        self._W = pi * (1 - pi)
        self._W = np.maximum(self._W, 1e-10)
        W_sqrt = np.sqrt(self._W)
        B = np.eye(n) + np.outer(W_sqrt, W_sqrt) * K
        self._L = _jitter_cholesky(B)
        return self

    def predict(self, X_test, return_prob=True):
        """Predict at test points.

        Args:
            return_prob: if True, return class probabilities; else return latent mean/var.
        """
        X_test = _ensure_2d(X_test)
        n = len(self.y_train)
        K_star = self.kernel.compute(X_test, self.X_train)
        K_ss_diag = self.kernel.diag(X_test)

        # Latent mean
        grad = self.y_train - _sigmoid(self._f_hat)
        f_mean = K_star @ grad

        # Latent variance
        W_sqrt = np.sqrt(self._W)
        v = solve_triangular(self._L, (W_sqrt[:, None] * K_star.T), lower=True)
        f_var = K_ss_diag - np.sum(v ** 2, axis=0)
        f_var = np.maximum(f_var, 1e-10)

        if not return_prob:
            return f_mean, f_var

        # Approximate predictive probability using probit approximation
        kappa = 1.0 / np.sqrt(1.0 + np.pi * f_var / 8.0)
        prob = _sigmoid(kappa * f_mean)
        return prob

    def log_marginal_likelihood(self):
        """Laplace approximation to log marginal likelihood."""
        pi = _sigmoid(self._f_hat)
        n = len(self.y_train)
        log_lik = np.sum(self.y_train * np.log(pi + 1e-10) +
                         (1 - self.y_train) * np.log(1 - pi + 1e-10))
        log_det = np.sum(np.log(np.diag(self._L)))
        log_prior = -0.5 * self._f_hat @ cho_solve(
            (np.linalg.cholesky(self._K + 1e-6 * np.eye(n)), True), self._f_hat)
        return log_lik + log_prior - log_det


# ============================================================
# Variational GP (using C154)
# ============================================================

class VariationalGP:
    """Variational GP using stochastic variational inference.

    Uses inducing points with variational distribution q(u) = N(m, S)
    and optimizes the ELBO.
    """

    def __init__(self, kernel, noise_variance=1e-2, n_inducing=20):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.n_inducing = n_inducing
        self.Z = None
        self.m = None  # variational mean
        self.S = None  # variational covariance
        self._Kuu = None
        self._Luu = None
        self.elbo_history = []

    def fit(self, X, y, Z=None, n_iter=200, lr=0.01, batch_size=None, rng=None, verbose=False):
        """Fit variational GP via ELBO optimization."""
        if rng is None:
            rng = np.random.RandomState(42)

        X = _ensure_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        # Initialize inducing points
        if Z is not None:
            self.Z = _ensure_2d(Z)
        else:
            m = min(self.n_inducing, n)
            idx = rng.choice(n, size=m, replace=False)
            self.Z = X[idx].copy()

        m = len(self.Z)
        self._Kuu = self.kernel.compute(self.Z) + 1e-6 * np.eye(m)
        self._Luu = np.linalg.cholesky(self._Kuu)

        # Initialize variational params
        self.m = np.zeros(m)
        self.S_chol = np.eye(m) * 0.1  # Cholesky of S

        if batch_size is None:
            batch_size = min(n, 100)

        self.elbo_history = []

        for iteration in range(n_iter):
            # Mini-batch
            idx = rng.choice(n, size=min(batch_size, n), replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            scale = n / len(idx)

            elbo, grad_m, grad_S_chol = self._compute_elbo_and_grads(
                X_batch, y_batch, scale)

            # SGD updates with clipping
            grad_m = np.clip(grad_m, -10, 10)
            grad_S_chol = np.clip(grad_S_chol, -1, 1)
            self.m += lr * grad_m
            self.S_chol += lr * grad_S_chol
            # Ensure S_chol diagonal stays positive and bounded
            np.fill_diagonal(self.S_chol, np.clip(np.diag(self.S_chol), 1e-4, 10.0))

            self.elbo_history.append(elbo)

            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iter {iteration + 1}: ELBO = {elbo:.4f}")

        self.S = self.S_chol @ self.S_chol.T
        return self

    def _compute_elbo_and_grads(self, X_batch, y_batch, scale):
        """Compute ELBO and gradients."""
        m_pts = len(self.Z)
        S = self.S_chol @ self.S_chol.T

        Kuf = self.kernel.compute(self.Z, X_batch)
        A = solve_triangular(self._Luu, Kuf, lower=True)

        # q(f) = N(mu_f, Sigma_f)
        Linv_m = solve_triangular(self._Luu, self.m, lower=True)
        mu_f = A.T @ Linv_m
        Linv_S = solve_triangular(self._Luu, S, lower=True)
        Linv_S_LinvT = solve_triangular(self._Luu, Linv_S.T, lower=True).T

        sigma_f = self.kernel.diag(X_batch) - np.sum(A ** 2, axis=0) + np.sum(A * (Linv_S_LinvT @ A), axis=0)
        sigma_f = np.maximum(sigma_f, 1e-10)

        # Expected log likelihood (Gaussian)
        diff = y_batch - mu_f
        ell = -0.5 * np.sum(diff ** 2 / self.noise_variance + sigma_f / self.noise_variance +
                            np.log(2 * np.pi * self.noise_variance))
        ell *= scale

        # KL divergence KL(q(u) || p(u))
        # KL = 0.5 [tr(Kuu^{-1} S) + m^T Kuu^{-1} m - k + log|Kuu|/|S|]
        Kuu_inv_S = cho_solve((self._Luu, True), S)
        Kuu_inv_m = cho_solve((self._Luu, True), self.m)
        kl = 0.5 * (np.trace(Kuu_inv_S) + self.m @ Kuu_inv_m - m_pts +
                     2 * np.sum(np.log(np.diag(self._Luu))) - 2 * np.sum(np.log(np.abs(np.diag(self.S_chol)) + 1e-10)))

        elbo = ell - kl

        # Gradients for ascent (maximize ELBO)
        # dELBO/dm = scale * Kuu^{-1} Kuf (y - mu_f) / sigma^2 - Kuu^{-1} m
        grad_m = scale * cho_solve((self._Luu, True), Kuf) @ (diff / self.noise_variance) - Kuu_inv_m
        # Gradient for S_chol diagonal (simplified)
        grad_S_chol = np.zeros_like(self.S_chol)
        for i in range(m_pts):
            # dELBO/dS_ii = -0.5 scale * Kuu^{-1} Kuf Kfu Kuu^{-1}[i,i] / sigma^2 - 0.5 Kuu^{-1}[i,i] + 0.5 S^{-1}[i,i]
            grad_S_chol[i, i] = (-0.5 * scale * np.sum(A[i] ** 2) / self.noise_variance
                                  - 0.5 * Kuu_inv_S[i, i]
                                  + 0.5 / (self.S_chol[i, i] ** 2 + 1e-10)) * 2 * self.S_chol[i, i]

        return elbo, grad_m, 0.01 * grad_S_chol

    def predict(self, X_test, return_std=False):
        """Predict at test points."""
        X_test = _ensure_2d(X_test)
        Kuf = self.kernel.compute(self.Z, X_test)
        A = solve_triangular(self._Luu, Kuf, lower=True)
        Linv_m = solve_triangular(self._Luu, self.m, lower=True)
        mu = A.T @ Linv_m

        if not return_std:
            return mu

        S = self.S_chol @ self.S_chol.T
        Linv_S = solve_triangular(self._Luu, S, lower=True)
        Linv_S_LinvT = solve_triangular(self._Luu, Linv_S.T, lower=True).T

        var = self.kernel.diag(X_test) - np.sum(A ** 2, axis=0) + np.sum(A * (Linv_S_LinvT @ A), axis=0)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)


# ============================================================
# Hyperparameter Optimization
# ============================================================

class GPOptimizer:
    """Optimize GP hyperparameters by maximizing log marginal likelihood.

    Uses finite-difference gradients with L-BFGS-like updates.
    """

    def __init__(self, gp, param_names=None):
        """
        Args:
            gp: a GPRegression instance
            param_names: list of parameter names to optimize.
                         Default: all kernel params + 'noise_variance'
        """
        self.gp = gp
        if param_names is None:
            self.param_names = list(gp.kernel.get_params().keys()) + ['noise_variance']
        else:
            self.param_names = param_names

    def _get_params(self):
        """Get current parameters as log-transformed vector."""
        params = []
        kp = self.gp.kernel.get_params()
        for name in self.param_names:
            if name == 'noise_variance':
                params.append(np.log(self.gp.noise_variance))
            else:
                val = kp[name]
                if isinstance(val, (list, np.ndarray)):
                    params.extend(np.log(np.asarray(val)))
                else:
                    params.append(np.log(val))
        return np.array(params)

    def _set_params(self, log_params):
        """Set parameters from log-transformed vector."""
        idx = 0
        kp = self.gp.kernel.get_params()
        for name in self.param_names:
            if name == 'noise_variance':
                self.gp.noise_variance = np.exp(log_params[idx])
                idx += 1
            else:
                val = kp[name]
                if isinstance(val, (list, np.ndarray)):
                    length = len(val)
                    new_val = np.exp(log_params[idx:idx + length])
                    self.gp.kernel.set_params(**{name: new_val})
                    idx += length
                else:
                    self.gp.kernel.set_params(**{name: np.exp(log_params[idx])})
                    idx += 1

    def _objective(self, log_params):
        """Negative log marginal likelihood."""
        self._set_params(log_params)
        self.gp.fit(self.gp.X_train, self.gp.y_train)
        return -self.gp.log_marginal_likelihood()

    def _gradient(self, log_params, eps=1e-5):
        """Finite-difference gradient."""
        grad = np.zeros_like(log_params)
        f0 = self._objective(log_params)
        for i in range(len(log_params)):
            log_params_plus = log_params.copy()
            log_params_plus[i] += eps
            grad[i] = (self._objective(log_params_plus) - f0) / eps
        self._set_params(log_params)  # restore
        self.gp.fit(self.gp.X_train, self.gp.y_train)
        return grad

    def optimize(self, n_iter=100, lr=0.01, verbose=False):
        """Optimize hyperparameters using gradient descent with momentum.

        Returns:
            dict: optimized parameters and LML history
        """
        log_params = self._get_params()
        velocity = np.zeros_like(log_params)
        momentum = 0.9
        history = []

        for i in range(n_iter):
            lml = -self._objective(log_params)
            history.append(lml)

            grad = self._gradient(log_params)
            velocity = momentum * velocity - lr * grad
            log_params = log_params + velocity

            self._set_params(log_params)
            self.gp.fit(self.gp.X_train, self.gp.y_train)

            if verbose and (i + 1) % 10 == 0:
                print(f"Iter {i + 1}: LML = {lml:.4f}")

        final_params = {}
        kp = self.gp.kernel.get_params()
        for name in self.param_names:
            if name == 'noise_variance':
                final_params[name] = self.gp.noise_variance
            else:
                final_params[name] = kp[name]

        return {
            'params': final_params,
            'lml_history': history,
            'final_lml': history[-1] if history else None
        }


# ============================================================
# Multi-output GP (Intrinsic Coregionalization Model)
# ============================================================

class MultiOutputGP:
    """Multi-output GP using the Intrinsic Coregionalization Model (ICM).

    k((x,d), (x',d')) = B[d,d'] * k_base(x, x')

    where B is a positive semi-definite coregionalization matrix.
    """

    def __init__(self, base_kernel, n_outputs, noise_variance=1e-2):
        self.base_kernel = base_kernel
        self.n_outputs = n_outputs
        self.noise_variance = noise_variance
        self.B = np.eye(n_outputs)  # coregionalization matrix
        self.X_train = None
        self.y_train = None
        self.d_train = None
        self._L = None
        self._alpha = None

    def set_coregionalization(self, B):
        """Set the coregionalization matrix B (must be PSD)."""
        self.B = np.asarray(B, dtype=np.float64)

    def fit(self, X, y, d):
        """Fit multi-output GP.

        Args:
            X: inputs (n,) or (n, D)
            y: outputs (n,) -- all outputs stacked
            d: output indices (n,) -- which output each y belongs to
        """
        self.X_train = _ensure_2d(X)
        self.y_train = np.asarray(y, dtype=np.float64).ravel()
        self.d_train = np.asarray(d, dtype=int).ravel()
        n = len(self.y_train)

        # Build full kernel matrix
        K_base = self.base_kernel.compute(self.X_train)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.B[self.d_train[i], self.d_train[j]] * K_base[i, j]

        K += self.noise_variance * np.eye(n)
        self._L = _jitter_cholesky(K)
        self._alpha = cho_solve((self._L, True), self.y_train)
        return self

    def predict(self, X_test, d_test, return_std=False):
        """Predict for specific output dimensions.

        Args:
            X_test: test inputs
            d_test: output indices for test points
        """
        X_test = _ensure_2d(X_test)
        d_test = np.asarray(d_test, dtype=int).ravel()
        n_train = len(self.y_train)
        n_test = len(d_test)

        K_base_star = self.base_kernel.compute(X_test, self.X_train)
        K_star = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_star[i, j] = self.B[d_test[i], self.d_train[j]] * K_base_star[i, j]

        mu = K_star @ self._alpha

        if not return_std:
            return mu

        v = solve_triangular(self._L, K_star.T, lower=True)
        K_ss_diag = np.array([self.B[d_test[i], d_test[i]] * self.base_kernel.diag(X_test[i:i+1])[0]
                              for i in range(n_test)])
        var = K_ss_diag - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)


# ============================================================
# GP Utilities
# ============================================================

class GPUtils:
    """Utility functions for Gaussian Processes."""

    @staticmethod
    def kernel_matrix_analysis(kernel, X):
        """Analyze properties of a kernel matrix."""
        X = _ensure_2d(X)
        K = kernel.compute(X)
        eigenvalues = np.linalg.eigvalsh(K)
        return {
            'shape': K.shape,
            'min_eigenvalue': float(np.min(eigenvalues)),
            'max_eigenvalue': float(np.max(eigenvalues)),
            'condition_number': float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)),
            'trace': float(np.trace(K)),
            'is_positive_definite': bool(np.all(eigenvalues > -1e-10)),
            'rank': int(np.sum(eigenvalues > 1e-10)),
            'eigenvalues': eigenvalues.tolist(),
        }

    @staticmethod
    def compare_kernels(kernels, X, y, noise_variance=1e-2):
        """Compare multiple kernels by their log marginal likelihood.

        Args:
            kernels: dict of {name: kernel}
            X, y: training data
            noise_variance: observation noise

        Returns:
            dict of {name: {'lml': float, 'bic': float}}
        """
        X = _ensure_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        results = {}

        for name, kernel in kernels.items():
            gp = GPRegression(kernel, noise_variance=noise_variance)
            gp.fit(X, y)
            lml = gp.log_marginal_likelihood()
            n_params = len([v for v in kernel.get_params().values()
                           if not isinstance(v, dict)])
            bic = -2 * lml + n_params * np.log(n)
            results[name] = {'lml': float(lml), 'bic': float(bic), 'n_params': n_params}

        return results

    @staticmethod
    def cross_validate(gp, X, y, n_folds=5, rng=None):
        """K-fold cross-validation for GP regression.

        Returns:
            dict with 'mse', 'nlpd' (negative log predictive density), 'fold_scores'
        """
        if rng is None:
            rng = np.random.RandomState(42)
        X = _ensure_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        indices = rng.permutation(n)
        fold_size = n // n_folds

        mses = []
        nlpds = []

        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else n
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            gp.fit(X[train_idx], y[train_idx])
            mu, std = gp.predict(X[test_idx], return_std=True)

            mse = np.mean((y[test_idx] - mu) ** 2)
            mses.append(float(mse))

            nlpd = 0.5 * np.mean(np.log(2 * np.pi * std ** 2) + (y[test_idx] - mu) ** 2 / std ** 2)
            nlpds.append(float(nlpd))

        return {
            'mse': float(np.mean(mses)),
            'mse_std': float(np.std(mses)),
            'nlpd': float(np.mean(nlpds)),
            'nlpd_std': float(np.std(nlpds)),
            'fold_mses': mses,
            'fold_nlpds': nlpds,
        }

    @staticmethod
    def lengthscale_sensitivity(gp, X, y, lengthscales, param_name='lengthscale'):
        """Analyze sensitivity to lengthscale parameter.

        Returns:
            list of {'lengthscale': float, 'lml': float}
        """
        X = _ensure_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        results = []
        for ls in lengthscales:
            gp.kernel.set_params(**{param_name: ls})
            gp.fit(X, y)
            lml = gp.log_marginal_likelihood()
            results.append({'lengthscale': float(ls), 'lml': float(lml)})
        return results


# ============================================================
# Heteroscedastic GP
# ============================================================

class HeteroscedasticGP:
    """GP with input-dependent noise.

    Models log noise as a separate GP: log(sigma^2(x)) ~ GP.
    Uses iterative approach:
    1. Fit main GP
    2. Estimate noise from residuals
    3. Fit noise GP
    4. Refit main GP with estimated noise
    """

    def __init__(self, kernel, noise_kernel=None, n_iterations=5):
        self.kernel = kernel
        self.noise_kernel = noise_kernel or RBFKernel(lengthscale=1.0, variance=1.0)
        self.n_iterations = n_iterations
        self.main_gp = None
        self.noise_gp = None

    def fit(self, X, y):
        """Fit heteroscedastic GP."""
        X = _ensure_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        # Initial fit with constant noise
        self.main_gp = GPRegression(self.kernel, noise_variance=np.var(y) * 0.1)
        self.main_gp.fit(X, y)

        for _ in range(self.n_iterations):
            # Estimate residuals
            mu = self.main_gp.predict(X)
            residuals = (y - mu) ** 2
            log_residuals = np.log(residuals + 1e-10)

            # Fit noise GP
            self.noise_gp = GPRegression(self.noise_kernel, noise_variance=0.5)
            self.noise_gp.fit(X, log_residuals)

            # Predict noise at training points
            log_noise = self.noise_gp.predict(X)
            noise_var = np.exp(log_noise)
            noise_var = np.maximum(noise_var, 1e-6)

            # Refit main GP with heteroscedastic noise
            K = self.kernel.compute(X)
            K += np.diag(noise_var)
            self.main_gp._L = _jitter_cholesky(K)
            self.main_gp._alpha = cho_solve((self.main_gp._L, True), y)

        return self

    def predict(self, X_test, return_std=False, return_noise=False):
        """Predict with estimated noise."""
        X_test = _ensure_2d(X_test)
        mu = self.main_gp.predict(X_test)

        if not return_std and not return_noise:
            return mu

        # Get predictive variance from main GP
        K_star = self.kernel.compute(X_test, self.main_gp.X_train)
        v = solve_triangular(self.main_gp._L, K_star.T, lower=True)
        K_ss_diag = self.kernel.diag(X_test)
        model_var = K_ss_diag - np.sum(v ** 2, axis=0)
        model_var = np.maximum(model_var, 1e-10)

        # Get predicted noise
        log_noise = self.noise_gp.predict(X_test)
        noise_var = np.exp(log_noise)

        total_var = model_var + noise_var
        total_std = np.sqrt(np.maximum(total_var, 1e-10))

        result = [mu, total_std]
        if return_noise:
            result.append(np.sqrt(np.maximum(noise_var, 1e-10)))
        return tuple(result)


# ============================================================
# Warped GP
# ============================================================

class WarpedGP:
    """GP with output warping for non-Gaussian targets.

    Applies a monotonic transformation to outputs before GP modeling.
    Supports: 'log', 'sqrt', 'box-cox'.
    """

    def __init__(self, kernel, noise_variance=1e-2, warp='log', lam=0.5):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.warp = warp
        self.lam = lam  # Box-Cox parameter
        self.gp = GPRegression(kernel, noise_variance)

    def _transform(self, y):
        """Forward warp."""
        if self.warp == 'log':
            return np.log(y + 1e-10)
        elif self.warp == 'sqrt':
            return np.sqrt(np.maximum(y, 0))
        elif self.warp == 'box-cox':
            if abs(self.lam) < 1e-10:
                return np.log(y + 1e-10)
            return (np.power(np.maximum(y, 1e-10), self.lam) - 1.0) / self.lam
        return y

    def _inverse_transform(self, z):
        """Inverse warp."""
        if self.warp == 'log':
            return np.exp(z)
        elif self.warp == 'sqrt':
            return z ** 2
        elif self.warp == 'box-cox':
            if abs(self.lam) < 1e-10:
                return np.exp(z)
            return np.power(np.maximum(self.lam * z + 1, 1e-10), 1.0 / self.lam)
        return z

    def fit(self, X, y):
        """Fit warped GP."""
        self._y_original = np.asarray(y, dtype=np.float64).ravel()
        y_warped = self._transform(self._y_original)
        self.gp.fit(X, y_warped)
        return self

    def predict(self, X_test, return_std=False):
        """Predict in original (unwarped) space."""
        if return_std:
            mu_w, std_w = self.gp.predict(X_test, return_std=True)
            mu = self._inverse_transform(mu_w)
            # Approximate std in original space using delta method
            # Jacobian of inverse transform at mu_w
            if self.warp == 'log':
                jac = np.exp(mu_w)
            elif self.warp == 'sqrt':
                jac = 2 * np.maximum(mu_w, 1e-10)
            elif self.warp == 'box-cox':
                if abs(self.lam) < 1e-10:
                    jac = np.exp(mu_w)
                else:
                    jac = np.power(np.maximum(self.lam * mu_w + 1, 1e-10), (1.0 / self.lam) - 1)
            else:
                jac = np.ones_like(mu_w)
            std = np.abs(jac) * std_w
            return mu, std
        else:
            mu_w = self.gp.predict(X_test)
            return self._inverse_transform(mu_w)


# ============================================================
# Student-t GP
# ============================================================

class StudentTGP:
    """GP with Student-t likelihood for robust regression.

    Uses Laplace approximation for inference with heavy-tailed noise.
    """

    def __init__(self, kernel, df=4.0, scale=1.0, max_iter=100, tol=1e-6):
        self.kernel = kernel
        self.df = df  # degrees of freedom
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.X_train = None
        self.y_train = None
        self._f_hat = None
        self._W = None
        self._K = None

    def fit(self, X, y):
        """Fit Student-t GP using Laplace approximation with damped Newton."""
        self.X_train = _ensure_2d(X)
        self.y_train = np.asarray(y, dtype=np.float64).ravel()
        n = len(self.y_train)
        K = self.kernel.compute(self.X_train)
        self._K = K

        nu = self.df
        s2 = self.scale ** 2

        # Damped Newton's method
        f = np.zeros(n)
        for it in range(self.max_iter):
            r = self.y_train - f
            denom = nu * s2 + r ** 2
            grad_loglik = (nu + 1) * r / denom

            # Negative Hessian of log-likelihood
            W = (nu + 1) * (nu * s2 - r ** 2) / denom ** 2
            # Where W is negative (near outliers), use small positive value
            W_safe = np.maximum(W, 1e-6)

            W_sqrt = np.sqrt(W_safe)
            B = np.eye(n) + np.outer(W_sqrt, W_sqrt) * K
            L = _jitter_cholesky(B)

            # Newton direction
            b = W_safe * f + grad_loglik
            a = b - W_sqrt * cho_solve((L, True), (W_sqrt * (K @ b)))
            f_new = K @ a

            # Damping for stability
            step = f_new - f
            step_size = min(1.0, 3.0 / (np.max(np.abs(step)) + 1e-10))
            f_new = f + step_size * step

            if np.max(np.abs(f_new - f)) < self.tol:
                f = f_new
                break
            f = f_new

        self._f_hat = f
        r = self.y_train - f
        denom = nu * s2 + r ** 2
        self._W = np.maximum((nu + 1) * (nu * s2 - r ** 2) / denom ** 2, 1e-6)
        W_sqrt = np.sqrt(self._W)
        B = np.eye(n) + np.outer(W_sqrt, W_sqrt) * K
        self._L = _jitter_cholesky(B)
        return self

    def predict(self, X_test, return_std=False):
        """Predict at test points."""
        X_test = _ensure_2d(X_test)
        K_star = self.kernel.compute(X_test, self.X_train)
        K_ss_diag = self.kernel.diag(X_test)

        nu = self.df
        s2 = self.scale ** 2
        r = self.y_train - self._f_hat
        denom = nu * s2 + r ** 2
        grad = (nu + 1) * r / denom

        mu = K_star @ grad

        if not return_std:
            return mu

        W_sqrt = np.sqrt(self._W)
        v = solve_triangular(self._L, (W_sqrt[:, None] * K_star.T), lower=True)
        var = K_ss_diag - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)
