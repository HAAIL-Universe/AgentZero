"""
C152: Bayesian Optimization

Global optimization of expensive black-box functions using probabilistic
surrogate models and acquisition functions.

7 components:
  1. Kernel          -- Covariance functions (RBF, Matern, Rational Quadratic, composite)
  2. GaussianProcess -- GP regression with Cholesky factorization
  3. AcquisitionFunction -- EI, PI, UCB, Thompson Sampling
  4. BayesianOptimizer -- Main optimization loop
  5. MultiObjectiveBayesianOptimizer -- Pareto-aware multi-objective BO
  6. BatchBayesianOptimizer -- Parallel batch suggestions via kriging believer
  7. ConstrainedBayesianOptimizer -- BO with black-box constraints
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict, Any
from enum import Enum, auto
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. Kernel -- Covariance functions
# ---------------------------------------------------------------------------

class Kernel:
    """Base kernel class."""

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        """Efficient diagonal of K(X, X)."""
        return np.array([self(X[i:i+1], X[i:i+1])[0, 0] for i in range(len(X))])

    def __add__(self, other: 'Kernel') -> 'SumKernel':
        return SumKernel(self, other)

    def __mul__(self, other) -> 'ProductKernel':
        if isinstance(other, (int, float)):
            return ScaleKernel(self, other)
        return ProductKernel(self, other)

    def __rmul__(self, other) -> 'ProductKernel':
        if isinstance(other, (int, float)):
            return ScaleKernel(self, other)
        return ProductKernel(other, self)

    def get_params(self) -> Dict[str, float]:
        return {}

    def set_params(self, **params):
        pass


class RBFKernel(Kernel):
    """Radial Basis Function (squared exponential) kernel.
    k(x, x') = variance * exp(-||x - x'||^2 / (2 * lengthscale^2))
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sqdist = np.sum(X1**2, axis=1, keepdims=True) + \
                 np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        sqdist = np.maximum(sqdist, 0.0)
        return self.variance * np.exp(-0.5 * sqdist / self.lengthscale**2)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.variance)

    def get_params(self) -> Dict[str, float]:
        return {'lengthscale': self.lengthscale, 'variance': self.variance}

    def set_params(self, **params):
        if 'lengthscale' in params:
            self.lengthscale = params['lengthscale']
        if 'variance' in params:
            self.variance = params['variance']


class MaternKernel(Kernel):
    """Matern kernel with nu=1.5 or nu=2.5.
    More flexible than RBF -- controls smoothness.
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0, nu: float = 2.5):
        self.lengthscale = lengthscale
        self.variance = variance
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be 0.5, 1.5, or 2.5")
        self.nu = nu

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sqdist = np.sum(X1**2, axis=1, keepdims=True) + \
                 np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        sqdist = np.maximum(sqdist, 0.0)
        r = np.sqrt(sqdist) / self.lengthscale

        if self.nu == 0.5:
            K = np.exp(-r)
        elif self.nu == 1.5:
            sqrt3_r = math.sqrt(3) * r
            K = (1 + sqrt3_r) * np.exp(-sqrt3_r)
        else:  # 2.5
            sqrt5_r = math.sqrt(5) * r
            K = (1 + sqrt5_r + 5.0/3.0 * r**2) * np.exp(-sqrt5_r)

        return self.variance * K

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.variance)

    def get_params(self) -> Dict[str, float]:
        return {'lengthscale': self.lengthscale, 'variance': self.variance}

    def set_params(self, **params):
        if 'lengthscale' in params:
            self.lengthscale = params['lengthscale']
        if 'variance' in params:
            self.variance = params['variance']


class RationalQuadraticKernel(Kernel):
    """Rational Quadratic kernel -- infinite mixture of RBFs.
    k(x,x') = variance * (1 + ||x-x'||^2 / (2*alpha*l^2))^(-alpha)
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0, alpha: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.alpha = alpha

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sqdist = np.sum(X1**2, axis=1, keepdims=True) + \
                 np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        sqdist = np.maximum(sqdist, 0.0)
        return self.variance * (1 + sqdist / (2 * self.alpha * self.lengthscale**2)) ** (-self.alpha)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.variance)


class LinearKernel(Kernel):
    """Linear kernel: k(x,x') = variance * x . x' + bias."""

    def __init__(self, variance: float = 1.0, bias: float = 0.0):
        self.variance = variance
        self.bias = bias

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return self.variance * (X1 @ X2.T) + self.bias


class SumKernel(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) + self.k2(X1, X2)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        return self.k1.diagonal(X) + self.k2.diagonal(X)


class ProductKernel(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) * self.k2(X1, X2)


class ScaleKernel(Kernel):
    def __init__(self, kernel: Kernel, scale: float):
        self.kernel = kernel
        self.scale = scale

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.scale * self.kernel(X1, X2)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        return self.scale * self.kernel.diagonal(X)


# ---------------------------------------------------------------------------
# 2. GaussianProcess -- GP regression
# ---------------------------------------------------------------------------

class GaussianProcess:
    """Gaussian Process regression with Cholesky decomposition.

    Supports:
    - Posterior mean and variance prediction
    - Log marginal likelihood for hyperparameter optimization
    - Noise estimation
    """

    def __init__(self, kernel: Kernel, noise: float = 1e-6, normalize_y: bool = True):
        self.kernel = kernel
        self.noise = noise
        self.normalize_y = normalize_y
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcess':
        """Fit GP to training data."""
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).ravel()

        self.X_train = X.copy()

        if self.normalize_y:
            self.y_mean_ = np.mean(y)
            self.y_std_ = max(np.std(y), 1e-10)
            self.y_train = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_ = 0.0
            self.y_std_ = 1.0
            self.y_train = y.copy()

        K = self.kernel(X, X)
        K += self.noise * np.eye(len(X))

        # Add jitter for numerical stability
        jitter = 1e-8
        for _ in range(5):
            try:
                self.L_ = np.linalg.cholesky(K)
                break
            except np.linalg.LinAlgError:
                K += jitter * np.eye(len(X))
                jitter *= 10

        self.alpha_ = cho_solve(
            (self.L_, True), self.y_train
        )
        return self

    def predict(self, X: np.ndarray, return_std: bool = False,
                return_cov: bool = False) -> Any:
        """Predict mean and optionally variance at new points."""
        X = np.atleast_2d(X)

        K_star = self.kernel(X, self.X_train)
        mu = K_star @ self.alpha_

        # Denormalize
        mu = mu * self.y_std_ + self.y_mean_

        if not return_std and not return_cov:
            return mu

        v = cho_solve((self.L_, True), K_star.T)
        K_ss = self.kernel(X, X)
        cov = K_ss - K_star @ v

        if return_std:
            var = np.maximum(np.diag(cov) * self.y_std_**2, 0.0)
            return mu, np.sqrt(var)

        return mu, cov * self.y_std_**2

    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood of the training data."""
        if self.L_ is None:
            return -np.inf
        n = len(self.y_train)
        log_det = 2.0 * np.sum(np.log(np.diag(self.L_)))
        data_fit = self.y_train @ self.alpha_
        return -0.5 * (data_fit + log_det + n * np.log(2 * np.pi))

    def sample_posterior(self, X: np.ndarray, n_samples: int = 1,
                         rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Draw samples from the posterior distribution."""
        X = np.atleast_2d(X)
        if rng is None:
            rng = np.random.RandomState()

        mu, cov = self.predict(X, return_cov=True)
        cov += 1e-8 * np.eye(len(X))

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        samples = mu[:, None] + L @ rng.randn(len(X), n_samples)
        return samples.T  # (n_samples, n_points)

    def optimize_hyperparameters(self, n_restarts: int = 3):
        """Optimize kernel hyperparameters by maximizing log marginal likelihood."""
        if self.X_train is None:
            return

        params = self.kernel.get_params()
        if not params:
            return

        param_names = list(params.keys())
        best_lml = -np.inf
        best_params = params.copy()

        for _ in range(n_restarts):
            x0 = np.log([max(params[k] * np.exp(np.random.randn()), 1e-6)
                         for k in param_names])

            def neg_lml(log_params):
                p = {k: np.exp(v) for k, v in zip(param_names, log_params)}
                self.kernel.set_params(**p)
                self.fit(self.X_train, self.y_train * self.y_std_ + self.y_mean_)
                return -self.log_marginal_likelihood()

            try:
                result = scipy_minimize(neg_lml, x0, method='L-BFGS-B',
                                        options={'maxiter': 50})
                if -result.fun > best_lml:
                    best_lml = -result.fun
                    best_params = {k: np.exp(v) for k, v in zip(param_names, result.x)}
            except Exception:
                pass

        self.kernel.set_params(**best_params)
        self.fit(self.X_train, self.y_train * self.y_std_ + self.y_mean_)


# ---------------------------------------------------------------------------
# 3. AcquisitionFunction -- Guide the search
# ---------------------------------------------------------------------------

class AcquisitionType(Enum):
    EI = auto()    # Expected Improvement
    PI = auto()    # Probability of Improvement
    UCB = auto()   # Upper Confidence Bound
    TS = auto()    # Thompson Sampling


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function."""
    zero_mask = sigma < 1e-10
    sigma_safe = np.maximum(sigma, 1e-10)
    imp = mu - y_best - xi
    Z = imp / sigma_safe
    ei = imp * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    ei[zero_mask] = 0.0
    return ei


def probability_of_improvement(mu: np.ndarray, sigma: np.ndarray,
                                y_best: float, xi: float = 0.01) -> np.ndarray:
    """Probability of Improvement acquisition function."""
    sigma = np.maximum(sigma, 1e-10)
    Z = (mu - y_best - xi) / sigma
    return norm.cdf(Z)


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray,
                           beta: float = 2.0) -> np.ndarray:
    """Upper Confidence Bound acquisition function."""
    return mu + beta * sigma


class AcquisitionFunction:
    """Wrapper for acquisition functions with optimization."""

    def __init__(self, acq_type: AcquisitionType = AcquisitionType.EI,
                 xi: float = 0.01, beta: float = 2.0):
        self.acq_type = acq_type
        self.xi = xi
        self.beta = beta

    def evaluate(self, X: np.ndarray, gp: GaussianProcess,
                 y_best: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Evaluate acquisition function at points X."""
        X = np.atleast_2d(X)

        if self.acq_type == AcquisitionType.TS:
            if rng is None:
                rng = np.random.RandomState()
            return gp.sample_posterior(X, n_samples=1, rng=rng).ravel()

        mu, sigma = gp.predict(X, return_std=True)

        if self.acq_type == AcquisitionType.EI:
            return expected_improvement(mu, sigma, y_best, self.xi)
        elif self.acq_type == AcquisitionType.PI:
            return probability_of_improvement(mu, sigma, y_best, self.xi)
        elif self.acq_type == AcquisitionType.UCB:
            return upper_confidence_bound(mu, sigma, self.beta)
        else:
            raise ValueError(f"Unknown acquisition type: {self.acq_type}")

    def optimize(self, gp: GaussianProcess, bounds: np.ndarray,
                 y_best: float, n_candidates: int = 1000,
                 n_restarts: int = 5,
                 rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Find the point that maximizes the acquisition function."""
        if rng is None:
            rng = np.random.RandomState()

        dim = bounds.shape[0]

        # Random candidates
        candidates = np.column_stack([
            rng.uniform(bounds[d, 0], bounds[d, 1], n_candidates)
            for d in range(dim)
        ])

        acq_values = self.evaluate(candidates, gp, y_best, rng=rng)
        best_idx = np.argmax(acq_values)
        best_x = candidates[best_idx]
        best_acq = acq_values[best_idx]

        # Local optimization from best candidates
        top_indices = np.argsort(acq_values)[-n_restarts:]

        for idx in top_indices:
            x0 = candidates[idx]

            def neg_acq(x):
                x_2d = np.atleast_2d(x)
                return -self.evaluate(x_2d, gp, y_best, rng=rng)[0]

            try:
                result = scipy_minimize(
                    neg_acq, x0,
                    bounds=[(bounds[d, 0], bounds[d, 1]) for d in range(dim)],
                    method='L-BFGS-B',
                    options={'maxiter': 20}
                )
                if -result.fun > best_acq:
                    best_acq = -result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x


# ---------------------------------------------------------------------------
# 4. BayesianOptimizer -- Main optimization loop
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result of Bayesian Optimization."""
    best_x: np.ndarray
    best_y: float
    X_observed: np.ndarray
    y_observed: np.ndarray
    n_iterations: int
    convergence_history: List[float] = field(default_factory=list)


class BayesianOptimizer:
    """Bayesian Optimization for black-box function maximization.

    Uses GP surrogate + acquisition function to efficiently find
    the global optimum of expensive-to-evaluate functions.
    """

    def __init__(self, bounds: np.ndarray,
                 kernel: Optional[Kernel] = None,
                 acquisition: Optional[AcquisitionFunction] = None,
                 noise: float = 1e-6,
                 n_initial: int = 5,
                 seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.kernel = kernel or RBFKernel(lengthscale=1.0, variance=1.0)
        self.acquisition = acquisition or AcquisitionFunction(AcquisitionType.EI)
        self.noise = noise
        self.n_initial = n_initial
        self.rng = np.random.RandomState(seed)
        self.gp: Optional[GaussianProcess] = None
        self.X_observed: Optional[np.ndarray] = None
        self.y_observed: Optional[np.ndarray] = None

    def _initial_samples(self, n: int) -> np.ndarray:
        """Latin Hypercube Sampling for initial points."""
        samples = np.zeros((n, self.dim))
        for d in range(self.dim):
            perms = self.rng.permutation(n)
            for i in range(n):
                samples[i, d] = (perms[i] + self.rng.uniform()) / n
            samples[:, d] = self.bounds[d, 0] + samples[:, d] * (self.bounds[d, 1] - self.bounds[d, 0])
        return samples

    def suggest(self) -> np.ndarray:
        """Suggest the next point to evaluate."""
        if self.X_observed is None or len(self.X_observed) < self.n_initial:
            # Still in initial phase
            return self._initial_samples(1)[0]

        self.gp = GaussianProcess(self.kernel, noise=self.noise)
        self.gp.fit(self.X_observed, self.y_observed)

        y_best = np.max(self.y_observed)
        return self.acquisition.optimize(
            self.gp, self.bounds, y_best,
            rng=self.rng
        )

    def tell(self, x: np.ndarray, y: float):
        """Record an observation."""
        x = np.atleast_1d(x)
        if self.X_observed is None:
            self.X_observed = x.reshape(1, -1)
            self.y_observed = np.array([y])
        else:
            self.X_observed = np.vstack([self.X_observed, x])
            self.y_observed = np.append(self.y_observed, y)

    def maximize(self, objective: Callable, n_iter: int = 50,
                 callback: Optional[Callable] = None) -> OptimizationResult:
        """Run full Bayesian Optimization loop.

        Args:
            objective: Function to maximize. Takes 1D array, returns scalar.
            n_iter: Number of iterations (including initial samples).
            callback: Optional callback(iteration, x, y, best_y).
        """
        convergence = []

        for i in range(n_iter):
            x = self.suggest()
            y = float(objective(x))
            self.tell(x, y)

            best_y = np.max(self.y_observed)
            convergence.append(best_y)

            if callback:
                callback(i, x, y, best_y)

        best_idx = np.argmax(self.y_observed)
        return OptimizationResult(
            best_x=self.X_observed[best_idx].copy(),
            best_y=float(self.y_observed[best_idx]),
            X_observed=self.X_observed.copy(),
            y_observed=self.y_observed.copy(),
            n_iterations=n_iter,
            convergence_history=convergence
        )

    def minimize(self, objective: Callable, n_iter: int = 50,
                 callback: Optional[Callable] = None) -> OptimizationResult:
        """Minimize objective by maximizing its negation."""
        def neg_obj(x):
            return -objective(x)
        result = self.maximize(neg_obj, n_iter, callback)
        result.best_y = -result.best_y
        result.y_observed = -result.y_observed
        result.convergence_history = [-v for v in result.convergence_history]
        return result


# ---------------------------------------------------------------------------
# 5. MultiObjectiveBayesianOptimizer -- Pareto-aware BO
# ---------------------------------------------------------------------------

def _dominates(y1: np.ndarray, y2: np.ndarray) -> bool:
    """Does y1 dominate y2? (all >= and at least one >), for maximization."""
    return np.all(y1 >= y2) and np.any(y1 > y2)


def compute_pareto_front(Y: np.ndarray) -> np.ndarray:
    """Compute Pareto front indices (for maximization)."""
    n = len(Y)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if _dominates(Y[j], Y[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def hypervolume_contribution(Y: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """Approximate hypervolume contribution of each point (2D exact, nD approximate)."""
    n = len(Y)
    if n == 0:
        return np.array([])

    contributions = np.zeros(n)

    if Y.shape[1] == 2:
        # Exact 2D hypervolume contribution
        # Sort by first objective descending
        sorted_idx = np.argsort(-Y[:, 0])
        sorted_Y = Y[sorted_idx]

        for rank, i in enumerate(range(len(sorted_Y))):
            y = sorted_Y[i]
            if np.any(y < ref_point):
                contributions[sorted_idx[i]] = 0.0
                continue

            # Width: bounded by neighbors in sorted order
            left_y1 = float('inf') if rank == 0 else sorted_Y[rank - 1][1]
            right_x0 = ref_point[0] if rank == len(sorted_Y) - 1 else sorted_Y[rank + 1][0]

            width = y[0] - right_x0
            height = y[1] - ref_point[1]
            contributions[sorted_idx[i]] = max(width * height, 0.0)
    else:
        # nD: approximate via Monte Carlo
        for i in range(n):
            subset = np.delete(Y, i, axis=0)
            # Simple approximation: volume between point and ref
            vol = np.prod(np.maximum(Y[i] - ref_point, 0.0))
            contributions[i] = max(vol, 0.0)

    return contributions


class MultiObjectiveBayesianOptimizer:
    """Multi-objective Bayesian Optimization using EHVI-like approach.

    Maintains separate GP surrogates for each objective and uses
    Expected Hypervolume Improvement for acquisition.
    """

    def __init__(self, bounds: np.ndarray, n_objectives: int = 2,
                 kernel: Optional[Kernel] = None,
                 noise: float = 1e-6, n_initial: int = 5,
                 ref_point: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.n_objectives = n_objectives
        self.kernel = kernel or RBFKernel()
        self.noise = noise
        self.n_initial = n_initial
        self.ref_point = ref_point if ref_point is not None else np.full(n_objectives, -1e6)
        self.rng = np.random.RandomState(seed)
        self.X_observed: Optional[np.ndarray] = None
        self.Y_observed: Optional[np.ndarray] = None
        self.gps: List[GaussianProcess] = []

    def _initial_samples(self, n: int) -> np.ndarray:
        samples = np.zeros((n, self.dim))
        for d in range(self.dim):
            perms = self.rng.permutation(n)
            for i in range(n):
                samples[i, d] = (perms[i] + self.rng.uniform()) / n
            samples[:, d] = self.bounds[d, 0] + samples[:, d] * (self.bounds[d, 1] - self.bounds[d, 0])
        return samples

    def tell(self, x: np.ndarray, y: np.ndarray):
        """Record a multi-objective observation."""
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if self.X_observed is None:
            self.X_observed = x.reshape(1, -1)
            self.Y_observed = y.reshape(1, -1)
        else:
            self.X_observed = np.vstack([self.X_observed, x])
            self.Y_observed = np.vstack([self.Y_observed, y])

    def suggest(self) -> np.ndarray:
        """Suggest next evaluation point."""
        if self.X_observed is None or len(self.X_observed) < self.n_initial:
            return self._initial_samples(1)[0]

        # Fit GP for each objective
        self.gps = []
        for obj_idx in range(self.n_objectives):
            gp = GaussianProcess(RBFKernel(
                lengthscale=self.kernel.lengthscale if hasattr(self.kernel, 'lengthscale') else 1.0,
                variance=self.kernel.variance if hasattr(self.kernel, 'variance') else 1.0
            ), noise=self.noise)
            gp.fit(self.X_observed, self.Y_observed[:, obj_idx])
            self.gps.append(gp)

        # Use Thompson sampling for multi-objective
        n_candidates = 500
        candidates = np.column_stack([
            self.rng.uniform(self.bounds[d, 0], self.bounds[d, 1], n_candidates)
            for d in range(self.dim)
        ])

        # Sample from each GP posterior
        sampled_objectives = np.zeros((n_candidates, self.n_objectives))
        for obj_idx, gp in enumerate(self.gps):
            samples = gp.sample_posterior(candidates, n_samples=1, rng=self.rng)
            sampled_objectives[:, obj_idx] = samples.ravel()

        # Score by hypervolume contribution to current Pareto front
        pareto_idx = compute_pareto_front(self.Y_observed)
        pareto_Y = self.Y_observed[pareto_idx]

        best_score = -np.inf
        best_idx = 0

        for i in range(n_candidates):
            y_new = sampled_objectives[i]
            if np.any(y_new < self.ref_point):
                continue
            # Check if it would be non-dominated
            dominated = False
            for py in pareto_Y:
                if _dominates(py, y_new):
                    dominated = True
                    break
            if not dominated:
                # Approximate improvement
                score = np.sum(np.maximum(y_new - self.ref_point, 0.0))
                if score > best_score:
                    best_score = score
                    best_idx = i

        return candidates[best_idx]

    def maximize(self, objectives: Callable, n_iter: int = 50) -> Dict[str, Any]:
        """Run multi-objective optimization.

        Args:
            objectives: Function taking 1D array, returning array of objective values.
            n_iter: Number of iterations.
        """
        for i in range(n_iter):
            x = self.suggest()
            y = np.asarray(objectives(x))
            self.tell(x, y)

        pareto_idx = compute_pareto_front(self.Y_observed)
        return {
            'pareto_X': self.X_observed[pareto_idx].copy(),
            'pareto_Y': self.Y_observed[pareto_idx].copy(),
            'X_observed': self.X_observed.copy(),
            'Y_observed': self.Y_observed.copy(),
            'n_pareto': len(pareto_idx),
            'n_iterations': n_iter,
        }


# ---------------------------------------------------------------------------
# 6. BatchBayesianOptimizer -- Parallel batch suggestions
# ---------------------------------------------------------------------------

class BatchBayesianOptimizer:
    """Batch Bayesian Optimization using kriging believer strategy.

    Suggests multiple points per iteration for parallel evaluation.
    Uses the GP mean as a "hallucinated" observation for each successive
    point in the batch.
    """

    def __init__(self, bounds: np.ndarray, batch_size: int = 4,
                 kernel: Optional[Kernel] = None,
                 acquisition: Optional[AcquisitionFunction] = None,
                 noise: float = 1e-6, n_initial: int = 5,
                 seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.batch_size = batch_size
        self.kernel = kernel or RBFKernel()
        self.acquisition = acquisition or AcquisitionFunction(AcquisitionType.EI)
        self.noise = noise
        self.n_initial = n_initial
        self.rng = np.random.RandomState(seed)
        self.X_observed: Optional[np.ndarray] = None
        self.y_observed: Optional[np.ndarray] = None

    def _initial_samples(self, n: int) -> np.ndarray:
        samples = np.zeros((n, self.dim))
        for d in range(self.dim):
            perms = self.rng.permutation(n)
            for i in range(n):
                samples[i, d] = (perms[i] + self.rng.uniform()) / n
            samples[:, d] = self.bounds[d, 0] + samples[:, d] * (self.bounds[d, 1] - self.bounds[d, 0])
        return samples

    def tell(self, x: np.ndarray, y: float):
        x = np.atleast_1d(x)
        if self.X_observed is None:
            self.X_observed = x.reshape(1, -1)
            self.y_observed = np.array([y])
        else:
            self.X_observed = np.vstack([self.X_observed, x])
            self.y_observed = np.append(self.y_observed, y)

    def suggest_batch(self) -> np.ndarray:
        """Suggest a batch of points using kriging believer."""
        if self.X_observed is None or len(self.X_observed) < self.n_initial:
            return self._initial_samples(self.batch_size)

        batch = []
        # Work with hallucinated data
        X_hall = self.X_observed.copy()
        y_hall = self.y_observed.copy()

        for _ in range(self.batch_size):
            gp = GaussianProcess(
                RBFKernel(
                    lengthscale=self.kernel.lengthscale if hasattr(self.kernel, 'lengthscale') else 1.0,
                    variance=self.kernel.variance if hasattr(self.kernel, 'variance') else 1.0
                ),
                noise=self.noise
            )
            gp.fit(X_hall, y_hall)

            y_best = np.max(y_hall)
            x_new = self.acquisition.optimize(gp, self.bounds, y_best, rng=self.rng)
            batch.append(x_new)

            # Hallucinate: use GP mean as observation
            y_hall_new = float(gp.predict(x_new.reshape(1, -1))[0])
            X_hall = np.vstack([X_hall, x_new])
            y_hall = np.append(y_hall, y_hall_new)

        return np.array(batch)

    def maximize(self, objective: Callable, n_batches: int = 10) -> OptimizationResult:
        """Run batch optimization."""
        convergence = []
        total_evals = 0

        for b in range(n_batches):
            batch = self.suggest_batch()
            for x in batch:
                y = float(objective(x))
                self.tell(x, y)
                total_evals += 1
                convergence.append(float(np.max(self.y_observed)))

        best_idx = np.argmax(self.y_observed)
        return OptimizationResult(
            best_x=self.X_observed[best_idx].copy(),
            best_y=float(self.y_observed[best_idx]),
            X_observed=self.X_observed.copy(),
            y_observed=self.y_observed.copy(),
            n_iterations=total_evals,
            convergence_history=convergence
        )


# ---------------------------------------------------------------------------
# 7. ConstrainedBayesianOptimizer -- BO with black-box constraints
# ---------------------------------------------------------------------------

class ConstrainedBayesianOptimizer:
    """Bayesian Optimization with black-box constraints.

    Models constraints with separate GPs and uses probability of
    feasibility to weight the acquisition function.
    """

    def __init__(self, bounds: np.ndarray,
                 n_constraints: int = 1,
                 kernel: Optional[Kernel] = None,
                 acquisition: Optional[AcquisitionFunction] = None,
                 noise: float = 1e-6, n_initial: int = 5,
                 seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.n_constraints = n_constraints
        self.kernel = kernel or RBFKernel()
        self.acquisition = acquisition or AcquisitionFunction(AcquisitionType.EI)
        self.noise = noise
        self.n_initial = n_initial
        self.rng = np.random.RandomState(seed)
        self.X_observed: Optional[np.ndarray] = None
        self.y_observed: Optional[np.ndarray] = None
        self.c_observed: Optional[np.ndarray] = None  # constraint values (>= 0 means feasible)

    def _initial_samples(self, n: int) -> np.ndarray:
        samples = np.zeros((n, self.dim))
        for d in range(self.dim):
            perms = self.rng.permutation(n)
            for i in range(n):
                samples[i, d] = (perms[i] + self.rng.uniform()) / n
            samples[:, d] = self.bounds[d, 0] + samples[:, d] * (self.bounds[d, 1] - self.bounds[d, 0])
        return samples

    def tell(self, x: np.ndarray, y: float, constraints: np.ndarray):
        """Record observation with constraint values.
        constraints: array where >= 0 means feasible for that constraint.
        """
        x = np.atleast_1d(x)
        constraints = np.atleast_1d(constraints)
        if self.X_observed is None:
            self.X_observed = x.reshape(1, -1)
            self.y_observed = np.array([y])
            self.c_observed = constraints.reshape(1, -1)
        else:
            self.X_observed = np.vstack([self.X_observed, x])
            self.y_observed = np.append(self.y_observed, y)
            self.c_observed = np.vstack([self.c_observed, constraints])

    def _probability_of_feasibility(self, X: np.ndarray) -> np.ndarray:
        """Compute P(all constraints satisfied) at each point."""
        X = np.atleast_2d(X)
        pof = np.ones(len(X))

        for c_idx in range(self.n_constraints):
            gp_c = GaussianProcess(
                RBFKernel(
                    lengthscale=self.kernel.lengthscale if hasattr(self.kernel, 'lengthscale') else 1.0,
                    variance=self.kernel.variance if hasattr(self.kernel, 'variance') else 1.0
                ),
                noise=self.noise
            )
            gp_c.fit(self.X_observed, self.c_observed[:, c_idx])
            mu_c, sigma_c = gp_c.predict(X, return_std=True)
            sigma_c = np.maximum(sigma_c, 1e-10)
            # P(c >= 0)
            pof *= norm.cdf(mu_c / sigma_c)

        return pof

    def suggest(self) -> np.ndarray:
        """Suggest next point considering constraints."""
        if self.X_observed is None or len(self.X_observed) < self.n_initial:
            return self._initial_samples(1)[0]

        # Fit objective GP
        gp_obj = GaussianProcess(
            RBFKernel(
                lengthscale=self.kernel.lengthscale if hasattr(self.kernel, 'lengthscale') else 1.0,
                variance=self.kernel.variance if hasattr(self.kernel, 'variance') else 1.0
            ),
            noise=self.noise
        )
        gp_obj.fit(self.X_observed, self.y_observed)

        # Find best feasible
        feasible_mask = np.all(self.c_observed >= 0, axis=1)
        if np.any(feasible_mask):
            y_best = np.max(self.y_observed[feasible_mask])
        else:
            y_best = np.min(self.y_observed)  # No feasible point yet

        # Generate candidates
        n_candidates = 1000
        candidates = np.column_stack([
            self.rng.uniform(self.bounds[d, 0], self.bounds[d, 1], n_candidates)
            for d in range(self.dim)
        ])

        # Compute acquisition * P(feasible)
        mu, sigma = gp_obj.predict(candidates, return_std=True)
        acq_values = expected_improvement(mu, sigma, y_best, self.acquisition.xi)
        pof = self._probability_of_feasibility(candidates)
        weighted = acq_values * pof

        best_idx = np.argmax(weighted)
        return candidates[best_idx]

    def maximize(self, objective: Callable, constraints: Callable,
                 n_iter: int = 50) -> Dict[str, Any]:
        """Run constrained optimization.

        Args:
            objective: f(x) -> scalar (to maximize)
            constraints: g(x) -> array (each >= 0 means feasible)
        """
        for i in range(n_iter):
            x = self.suggest()
            y = float(objective(x))
            c = np.asarray(constraints(x))
            self.tell(x, y, c)

        feasible_mask = np.all(self.c_observed >= 0, axis=1)
        if np.any(feasible_mask):
            feasible_y = self.y_observed[feasible_mask]
            feasible_X = self.X_observed[feasible_mask]
            best_idx = np.argmax(feasible_y)
            best_x = feasible_X[best_idx]
            best_y = float(feasible_y[best_idx])
        else:
            best_x = self.X_observed[0]
            best_y = float(self.y_observed[0])

        return {
            'best_x': best_x,
            'best_y': best_y,
            'feasible': bool(np.any(feasible_mask)),
            'n_feasible': int(np.sum(feasible_mask)),
            'X_observed': self.X_observed.copy(),
            'y_observed': self.y_observed.copy(),
            'c_observed': self.c_observed.copy(),
            'n_iterations': n_iter,
        }
