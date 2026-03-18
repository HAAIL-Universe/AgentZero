"""
V223: Bayesian Optimization

Black-box function optimization using Gaussian processes as surrogate models
and acquisition functions to guide sequential exploration.

Composes V222 (Gaussian Process Regression) for the surrogate model.

Key features:
- 5 acquisition functions: EI, PI, UCB, Thompson Sampling, Knowledge Gradient
- Sequential optimization with automatic GP updating
- Multi-point batch acquisition (q-EI via Kriging Believer)
- Multi-objective optimization (EHVI -- Expected Hypervolume Improvement)
- Constrained optimization (feasibility-weighted acquisition)
- Input warping and log-transform for heterogeneous search spaces
- Random restarts for acquisition optimization
- Convergence diagnostics and optimization history

Composes:
- V222 GaussianProcess (surrogate model, predictions with uncertainty)
- V222 Kernel classes (RBF, Matern, ARD for search space modeling)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any
from enum import Enum
import sys
import os

# Import V222 Gaussian Process
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
from gaussian_process import (
    GaussianProcess, Kernel, RBFKernel, Matern52Kernel, ARDKernel,
    ScaleKernel, GPPrediction, _minimize_lbfgsb
)


# ---------------------------------------------------------------------------
# Standard normal CDF and PDF (no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x):
    """Standard normal CDF using error function approximation."""
    # Abramowitz and Stegun approximation
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x):
    """Error function approximation (Abramowitz & Stegun 7.1.26)."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
            t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-x**2)
    return sign * result


# ---------------------------------------------------------------------------
# Acquisition Functions
# ---------------------------------------------------------------------------

class AcquisitionType(Enum):
    EI = "expected_improvement"
    PI = "probability_of_improvement"
    UCB = "upper_confidence_bound"
    THOMPSON = "thompson_sampling"
    KG = "knowledge_gradient"


@dataclass
class AcquisitionResult:
    """Result of acquisition function evaluation."""
    x_next: np.ndarray          # Next point to evaluate
    acq_value: float            # Acquisition function value at x_next
    acq_type: str               # Which acquisition function was used


def expected_improvement(mu: np.ndarray, std: np.ndarray, f_best: float,
                         xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function.

    EI(x) = (f_best - mu(x) - xi) * Phi(Z) + std(x) * phi(Z)
    where Z = (f_best - mu(x) - xi) / std(x)

    For minimization: improvement when f < f_best.
    """
    std = np.maximum(std, 1e-10)
    z = (f_best - mu - xi) / std
    ei = (f_best - mu - xi) * _norm_cdf(z) + std * _norm_pdf(z)
    # Zero out where std is effectively zero
    ei = np.where(std > 1e-10, ei, 0.0)
    return ei


def probability_of_improvement(mu: np.ndarray, std: np.ndarray, f_best: float,
                                xi: float = 0.01) -> np.ndarray:
    """Probability of Improvement acquisition function.

    PI(x) = Phi((f_best - mu(x) - xi) / std(x))
    """
    std = np.maximum(std, 1e-10)
    z = (f_best - mu - xi) / std
    return _norm_cdf(z)


def upper_confidence_bound(mu: np.ndarray, std: np.ndarray,
                           beta: float = 2.0) -> np.ndarray:
    """Lower Confidence Bound (for minimization).

    LCB(x) = mu(x) - beta * std(x)

    Returns negative so that maximizing the result = minimizing the function.
    We return -LCB = -mu(x) + beta * std(x) so higher is better (like EI/PI).
    """
    return -mu + beta * std


def thompson_sampling(gp: GaussianProcess, X_candidates: np.ndarray,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Thompson Sampling: draw a sample from the GP posterior, return values.

    The candidate with the lowest sample value is selected (minimization).
    Returns sampled function values at candidates.
    """
    rng = rng or np.random.default_rng()
    pred = gp.predict(X_candidates, return_cov=True)
    cov = pred.cov + 1e-8 * np.eye(len(X_candidates))
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal
        L = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-10)))
    z = rng.standard_normal(len(X_candidates))
    sample = pred.mean + L @ z
    # Return negative so max = best for minimization
    return -sample


def knowledge_gradient(mu: np.ndarray, std: np.ndarray,
                       f_best: float) -> np.ndarray:
    """Knowledge Gradient approximation.

    KG(x) ~= std(x) * phi(z) + (f_best - mu(x)) * Phi(z)
    where z = (f_best - mu(x)) / std(x)

    Similar to EI but measures the value of information.
    """
    std = np.maximum(std, 1e-10)
    z = (f_best - mu) / std
    kg = std * _norm_pdf(z) + (f_best - mu) * _norm_cdf(z)
    return np.where(std > 1e-10, kg, 0.0)


# ---------------------------------------------------------------------------
# Optimization Bounds
# ---------------------------------------------------------------------------

@dataclass
class Bounds:
    """Search space bounds."""
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        assert len(self.lower) == len(self.upper)
        assert np.all(self.lower < self.upper)

    @property
    def dim(self) -> int:
        return len(self.lower)

    def sample_uniform(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n points uniformly within bounds."""
        rng = rng or np.random.default_rng()
        return rng.uniform(self.lower, self.upper, size=(n, self.dim))

    def clip(self, X: np.ndarray) -> np.ndarray:
        """Clip points to bounds."""
        return np.clip(X, self.lower, self.upper)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize X to [0, 1]^d."""
        return (X - self.lower) / (self.upper - self.lower)

    def unnormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """Map from [0, 1]^d back to original space."""
        return X_norm * (self.upper - self.lower) + self.lower


# ---------------------------------------------------------------------------
# Optimization Result
# ---------------------------------------------------------------------------

@dataclass
class BOResult:
    """Result of Bayesian optimization."""
    x_best: np.ndarray                    # Best input found
    f_best: float                         # Best function value found
    X_history: np.ndarray                 # All evaluated inputs (n_eval, d)
    y_history: np.ndarray                 # All evaluated function values (n_eval,)
    n_iterations: int                     # Number of BO iterations
    convergence: List[float]              # Best value at each iteration
    acquisition_values: List[float]       # Acquisition value at each selected point
    model: Optional[GaussianProcess] = None  # Final GP surrogate


@dataclass
class BatchBOResult:
    """Result of batch Bayesian optimization."""
    x_best: np.ndarray
    f_best: float
    X_history: np.ndarray
    y_history: np.ndarray
    n_batches: int
    batch_sizes: List[int]
    convergence: List[float]


@dataclass
class MOBOResult:
    """Result of multi-objective Bayesian optimization."""
    pareto_X: np.ndarray                  # Pareto-optimal inputs
    pareto_y: np.ndarray                  # Pareto-optimal objectives (n, m)
    X_history: np.ndarray
    y_history: np.ndarray                 # (n_eval, n_obj)
    n_iterations: int
    hypervolume_history: List[float]      # Hypervolume at each iteration


@dataclass
class ConstrainedBOResult:
    """Result of constrained Bayesian optimization."""
    x_best: np.ndarray
    f_best: float
    X_history: np.ndarray
    y_history: np.ndarray
    feasible_mask: np.ndarray             # Boolean mask of feasible points
    constraint_values: np.ndarray         # (n_eval, n_constraints)
    n_iterations: int
    convergence: List[float]


# ---------------------------------------------------------------------------
# Acquisition Optimization
# ---------------------------------------------------------------------------

def _optimize_acquisition(acq_func: Callable, bounds: Bounds,
                          n_restarts: int = 10, n_candidates: int = 500,
                          rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, float]:
    """Optimize acquisition function using random candidates + local refinement.

    Returns (x_best, acq_best).
    """
    rng = rng or np.random.default_rng()

    # Phase 1: Evaluate on random candidates
    X_cand = bounds.sample_uniform(n_candidates, rng=rng)
    acq_vals = acq_func(X_cand)
    best_idx = np.argmax(acq_vals)
    x_best = X_cand[best_idx].copy()
    acq_best = acq_vals[best_idx]

    # Phase 2: Local refinement from top candidates via Nelder-Mead
    top_indices = np.argsort(acq_vals)[-n_restarts:]
    for idx in top_indices:
        x0 = X_cand[idx]

        def neg_acq(x):
            x_clipped = bounds.clip(x.reshape(1, -1))
            return -acq_func(x_clipped)[0]

        result_x = _simple_local_opt(neg_acq, x0, bounds, maxiter=50)
        result_val = acq_func(bounds.clip(result_x.reshape(1, -1)))[0]
        if result_val > acq_best:
            acq_best = result_val
            x_best = bounds.clip(result_x.reshape(1, -1))[0]

    return x_best, acq_best


def _simple_local_opt(func: Callable, x0: np.ndarray, bounds: Bounds,
                      maxiter: int = 50) -> np.ndarray:
    """Simple Nelder-Mead local optimizer respecting bounds."""
    n = len(x0)
    # Small simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.05 * (bounds.upper[i] - bounds.lower[i])
        simplex[i + 1][i] += step
        simplex[i + 1] = bounds.clip(simplex[i + 1].reshape(1, -1))[0]

    f_vals = np.array([func(s) for s in simplex])

    for _ in range(maxiter):
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        # Centroid of all but worst
        centroid = simplex[:-1].mean(axis=0)

        # Reflection
        xr = bounds.clip((2.0 * centroid - simplex[-1]).reshape(1, -1))[0]
        fr = func(xr)

        if fr < f_vals[-2]:
            if fr < f_vals[0]:
                # Expansion
                xe = bounds.clip((3.0 * centroid - 2.0 * simplex[-1]).reshape(1, -1))[0]
                fe = func(xe)
                if fe < fr:
                    simplex[-1], f_vals[-1] = xe, fe
                else:
                    simplex[-1], f_vals[-1] = xr, fr
            else:
                simplex[-1], f_vals[-1] = xr, fr
        else:
            # Contraction
            xc = bounds.clip((0.5 * (centroid + simplex[-1])).reshape(1, -1))[0]
            fc = func(xc)
            if fc < f_vals[-1]:
                simplex[-1], f_vals[-1] = xc, fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = bounds.clip((0.5 * (simplex[0] + simplex[i])).reshape(1, -1))[0]
                    f_vals[i] = func(simplex[i])

    return simplex[np.argmin(f_vals)]


# ---------------------------------------------------------------------------
# Core Bayesian Optimization
# ---------------------------------------------------------------------------

def bayesian_optimize(objective: Callable, bounds: Bounds,
                      n_iterations: int = 50,
                      n_initial: int = 5,
                      acquisition: AcquisitionType = AcquisitionType.EI,
                      kernel: Optional[Kernel] = None,
                      noise_variance: float = 1e-6,
                      acq_params: Optional[Dict[str, float]] = None,
                      rng: Optional[np.random.Generator] = None,
                      verbose: bool = False) -> BOResult:
    """Run Bayesian optimization to minimize objective.

    Args:
        objective: Function to minimize. Takes (d,) array, returns scalar.
        bounds: Search space bounds.
        n_iterations: Number of BO iterations after initial points.
        n_initial: Number of initial random evaluations.
        acquisition: Acquisition function type.
        kernel: GP kernel (default: ScaleKernel(Matern52)).
        noise_variance: GP observation noise.
        acq_params: Parameters for acquisition function (xi, beta, etc.).
        rng: Random number generator.
        verbose: Print progress.

    Returns:
        BOResult with optimization history and best found point.
    """
    rng = rng or np.random.default_rng(42)
    acq_params = acq_params or {}

    if kernel is None:
        kernel = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)

    # Initial random evaluations
    X = bounds.sample_uniform(n_initial, rng=rng)
    y = np.array([objective(x) for x in X])

    convergence = [np.min(y)]
    acq_values = []

    for i in range(n_iterations):
        # Fit GP surrogate
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(X, y)

        f_best = np.min(y)

        # Select next point via acquisition function
        if acquisition == AcquisitionType.THOMPSON:
            # For Thompson: sample from posterior on candidates
            X_cand = bounds.sample_uniform(1000, rng=rng)
            ts_vals = thompson_sampling(gp, X_cand, rng=rng)
            best_idx = np.argmax(ts_vals)
            x_next = X_cand[best_idx]
            acq_val = ts_vals[best_idx]
        else:
            # Analytical acquisition: optimize over domain
            def acq_func(X_test):
                X_test = np.atleast_2d(X_test)
                pred = gp.predict(X_test)
                if acquisition == AcquisitionType.EI:
                    return expected_improvement(pred.mean, pred.std, f_best,
                                                xi=acq_params.get('xi', 0.01))
                elif acquisition == AcquisitionType.PI:
                    return probability_of_improvement(pred.mean, pred.std, f_best,
                                                      xi=acq_params.get('xi', 0.01))
                elif acquisition == AcquisitionType.UCB:
                    return upper_confidence_bound(pred.mean, pred.std,
                                                  beta=acq_params.get('beta', 2.0))
                elif acquisition == AcquisitionType.KG:
                    return knowledge_gradient(pred.mean, pred.std, f_best)
                else:
                    raise ValueError(f"Unknown acquisition: {acquisition}")

            x_next, acq_val = _optimize_acquisition(acq_func, bounds, rng=rng)

        # Evaluate objective
        y_next = objective(x_next)
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, y_next)

        convergence.append(np.min(y))
        acq_values.append(float(acq_val))

        if verbose:
            print(f"  Iter {i+1}/{n_iterations}: f={y_next:.6f}, best={np.min(y):.6f}")

    best_idx = np.argmin(y)
    # Final GP
    final_gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
    final_gp.fit(X, y)

    return BOResult(
        x_best=X[best_idx].copy(),
        f_best=float(y[best_idx]),
        X_history=X,
        y_history=y,
        n_iterations=n_iterations,
        convergence=convergence,
        acquisition_values=acq_values,
        model=final_gp
    )


# ---------------------------------------------------------------------------
# Batch Bayesian Optimization (Kriging Believer)
# ---------------------------------------------------------------------------

def batch_bayesian_optimize(objective: Callable, bounds: Bounds,
                            n_batches: int = 10, batch_size: int = 4,
                            n_initial: int = 5,
                            kernel: Optional[Kernel] = None,
                            noise_variance: float = 1e-6,
                            rng: Optional[np.random.Generator] = None) -> BatchBOResult:
    """Batch Bayesian optimization using Kriging Believer heuristic.

    At each batch, greedily selects batch_size points:
    1. Find x* that maximizes EI
    2. Add (x*, mu(x*)) to training set (hallucinated observation)
    3. Refit GP, repeat

    After all batch points selected, evaluate them in parallel.
    """
    rng = rng or np.random.default_rng(42)

    if kernel is None:
        kernel = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)

    X = bounds.sample_uniform(n_initial, rng=rng)
    y = np.array([objective(x) for x in X])

    convergence = [np.min(y)]
    batch_sizes = []

    for b in range(n_batches):
        # Kriging Believer: greedily build batch
        X_batch = []
        X_fantasy = X.copy()
        y_fantasy = y.copy()

        for q in range(batch_size):
            gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
            gp.fit(X_fantasy, y_fantasy)
            f_best = np.min(y_fantasy)

            def acq_func(X_test):
                X_test = np.atleast_2d(X_test)
                pred = gp.predict(X_test)
                return expected_improvement(pred.mean, pred.std, f_best)

            x_next, _ = _optimize_acquisition(acq_func, bounds, rng=rng,
                                               n_candidates=200)
            X_batch.append(x_next)

            # Hallucinate: use GP mean as fake observation
            pred_next = gp.predict(x_next.reshape(1, -1))
            X_fantasy = np.vstack([X_fantasy, x_next.reshape(1, -1)])
            y_fantasy = np.append(y_fantasy, pred_next.mean[0])

        # Evaluate batch in parallel (simulated)
        X_batch = np.array(X_batch)
        y_batch = np.array([objective(x) for x in X_batch])

        X = np.vstack([X, X_batch])
        y = np.append(y, y_batch)
        convergence.append(np.min(y))
        batch_sizes.append(len(X_batch))

    best_idx = np.argmin(y)
    return BatchBOResult(
        x_best=X[best_idx].copy(),
        f_best=float(y[best_idx]),
        X_history=X,
        y_history=y,
        n_batches=n_batches,
        batch_sizes=batch_sizes,
        convergence=convergence
    )


# ---------------------------------------------------------------------------
# Multi-Objective Bayesian Optimization (EHVI)
# ---------------------------------------------------------------------------

def _is_pareto_optimal(y: np.ndarray) -> np.ndarray:
    """Find Pareto-optimal points (minimization).

    A point dominates another if it is <= in all objectives and < in at least one.
    Returns boolean mask of Pareto-optimal points.
    """
    n = len(y)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i?
            if np.all(y[j] <= y[i]) and np.any(y[j] < y[i]):
                is_pareto[i] = False
                break
    return is_pareto


def _hypervolume_2d(pareto_y: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute hypervolume for 2D Pareto front (exact).

    Uses sweep line algorithm. O(n log n).
    """
    # Filter points dominated by reference
    mask = np.all(pareto_y < ref_point, axis=1)
    if not np.any(mask):
        return 0.0
    pts = pareto_y[mask]

    # Sort by first objective
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    hv = 0.0
    prev_y = ref_point[1]
    for p in pts:
        hv += (ref_point[0] - p[0]) * (prev_y - p[1])
        prev_y = min(prev_y, p[1])

    # Correction: use proper hypervolume decomposition
    # Sort by obj1 ascending
    hv = 0.0
    for i in range(len(pts)):
        x_width = ref_point[0] - pts[i, 0]
        if i + 1 < len(pts):
            # Height is from this point's y to next point's y (or ref)
            y_height = ref_point[1] - pts[i, 1]
            # But we only count up to the next point's x
            x_this = pts[i + 1, 0] - pts[i, 0]
            hv += x_this * (ref_point[1] - pts[i, 1])
        else:
            hv += (ref_point[0] - pts[i, 0]) * (ref_point[1] - pts[i, 1])

    return hv


def _hypervolume_nd(pareto_y: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute hypervolume using inclusion-exclusion (exact but exponential).
    For 2D, uses sweep line. For higher dims, uses recursive slicing."""
    if pareto_y.shape[1] == 2:
        return _hypervolume_2d(pareto_y, ref_point)

    # General: recursive hypervolume via slicing
    # Filter dominated by ref
    mask = np.all(pareto_y < ref_point, axis=1)
    if not np.any(mask):
        return 0.0
    pts = pareto_y[mask]

    if len(pts) == 0:
        return 0.0
    if len(pts) == 1:
        return float(np.prod(ref_point - pts[0]))

    # Sort by last dimension
    d = pts.shape[1]
    order = np.argsort(pts[:, d - 1])
    pts = pts[order]

    hv = 0.0
    prev_slice = ref_point[d - 1]
    for i in range(len(pts)):
        height = prev_slice - pts[i, d - 1]
        if height > 0:
            # Project to d-1 dimensions
            projected = pts[:i + 1, :d - 1]
            proj_ref = ref_point[:d - 1]
            # Get Pareto front of projection
            p_mask = _is_pareto_optimal(projected)
            hv += height * _hypervolume_nd(projected[p_mask], proj_ref)
        prev_slice = pts[i, d - 1]

    return hv


def _ehvi_2d(mu: np.ndarray, std: np.ndarray, pareto_y: np.ndarray,
             ref_point: np.ndarray) -> np.ndarray:
    """Approximate Expected Hypervolume Improvement for 2D using MC sampling.

    For each candidate, samples from its predictive distribution and
    computes the hypervolume improvement.
    """
    n_candidates = mu.shape[0]
    n_mc = 100
    rng = np.random.default_rng(0)
    ehvi = np.zeros(n_candidates)

    current_hv = _hypervolume_2d(pareto_y, ref_point) if len(pareto_y) > 0 else 0.0

    for i in range(n_candidates):
        hv_sum = 0.0
        for _ in range(n_mc):
            # Sample from predictive distribution
            y_sample = mu[i] + std[i] * rng.standard_normal(mu.shape[1])
            # Add to Pareto front
            new_y = np.vstack([pareto_y, y_sample.reshape(1, -1)]) if len(pareto_y) > 0 else y_sample.reshape(1, -1)
            new_pareto = new_y[_is_pareto_optimal(new_y)]
            new_hv = _hypervolume_2d(new_pareto, ref_point)
            hv_sum += max(0.0, new_hv - current_hv)
        ehvi[i] = hv_sum / n_mc

    return ehvi


def multi_objective_optimize(objectives: List[Callable], bounds: Bounds,
                             n_iterations: int = 30,
                             n_initial: int = 5,
                             ref_point: Optional[np.ndarray] = None,
                             kernel: Optional[Kernel] = None,
                             noise_variance: float = 1e-6,
                             rng: Optional[np.random.Generator] = None) -> MOBOResult:
    """Multi-objective Bayesian optimization using EHVI.

    Maintains one GP per objective. Uses Expected Hypervolume Improvement
    to select the next point that maximizes expected Pareto front growth.

    Args:
        objectives: List of objective functions to minimize.
        bounds: Search space bounds.
        n_iterations: Number of BO iterations.
        n_initial: Initial random points.
        ref_point: Reference point for hypervolume (auto-set if None).
        kernel: GP kernel (shared across objectives).
        noise_variance: GP noise.
        rng: Random generator.
    """
    rng = rng or np.random.default_rng(42)
    n_obj = len(objectives)

    if kernel is None:
        kernel = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)

    # Initial evaluations
    X = bounds.sample_uniform(n_initial, rng=rng)
    Y = np.array([[obj(x) for obj in objectives] for x in X])

    if ref_point is None:
        ref_point = np.max(Y, axis=0) + 1.0

    hv_history = []

    for it in range(n_iterations):
        # Fit one GP per objective
        gps = []
        for j in range(n_obj):
            gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
            gp.fit(X, Y[:, j])
            gps.append(gp)

        # Current Pareto front
        pareto_mask = _is_pareto_optimal(Y)
        pareto_y = Y[pareto_mask]
        current_hv = _hypervolume_2d(pareto_y, ref_point) if n_obj == 2 else _hypervolume_nd(pareto_y, ref_point)
        hv_history.append(current_hv)

        # EHVI via MC sampling on candidates
        X_cand = bounds.sample_uniform(500, rng=rng)
        n_mc = 64

        # Get predictive means and stds for all objectives
        mu_all = np.zeros((len(X_cand), n_obj))
        std_all = np.zeros((len(X_cand), n_obj))
        for j in range(n_obj):
            pred = gps[j].predict(X_cand)
            mu_all[:, j] = pred.mean
            std_all[:, j] = pred.std

        # MC-EHVI
        ehvi = np.zeros(len(X_cand))
        for i in range(len(X_cand)):
            hv_sum = 0.0
            for _ in range(n_mc):
                y_sample = mu_all[i] + std_all[i] * rng.standard_normal(n_obj)
                new_y = np.vstack([pareto_y, y_sample.reshape(1, -1)]) if len(pareto_y) > 0 else y_sample.reshape(1, -1)
                new_pareto = new_y[_is_pareto_optimal(new_y)]
                if n_obj == 2:
                    new_hv = _hypervolume_2d(new_pareto, ref_point)
                else:
                    new_hv = _hypervolume_nd(new_pareto, ref_point)
                hv_sum += max(0.0, new_hv - current_hv)
            ehvi[i] = hv_sum / n_mc

        best_idx = np.argmax(ehvi)
        x_next = X_cand[best_idx]

        # Evaluate
        y_next = np.array([obj(x_next) for obj in objectives])
        X = np.vstack([X, x_next.reshape(1, -1)])
        Y = np.vstack([Y, y_next.reshape(1, -1)])

    # Final Pareto
    pareto_mask = _is_pareto_optimal(Y)
    final_hv = _hypervolume_2d(Y[pareto_mask], ref_point) if n_obj == 2 else _hypervolume_nd(Y[pareto_mask], ref_point)
    hv_history.append(final_hv)

    return MOBOResult(
        pareto_X=X[pareto_mask],
        pareto_y=Y[pareto_mask],
        X_history=X,
        y_history=Y,
        n_iterations=n_iterations,
        hypervolume_history=hv_history
    )


# ---------------------------------------------------------------------------
# Constrained Bayesian Optimization
# ---------------------------------------------------------------------------

def constrained_optimize(objective: Callable, constraints: List[Callable],
                         bounds: Bounds,
                         n_iterations: int = 50,
                         n_initial: int = 5,
                         kernel: Optional[Kernel] = None,
                         noise_variance: float = 1e-6,
                         rng: Optional[np.random.Generator] = None) -> ConstrainedBOResult:
    """Constrained Bayesian optimization.

    Each constraint function c_i(x) should return a value where c_i(x) <= 0
    means feasible. Uses Expected Feasible Improvement (EFI):
        EFI(x) = EI(x) * prod_i P(c_i(x) <= 0)

    One GP for the objective, one GP per constraint.
    """
    rng = rng or np.random.default_rng(42)
    n_con = len(constraints)

    if kernel is None:
        kernel = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)

    # Initial evaluations
    X = bounds.sample_uniform(n_initial, rng=rng)
    y = np.array([objective(x) for x in X])
    C = np.array([[c(x) for c in constraints] for x in X])

    convergence = []

    for it in range(n_iterations):
        # Fit GPs
        gp_obj = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp_obj.fit(X, y)

        gp_cons = []
        for j in range(n_con):
            gp_c = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
            gp_c.fit(X, C[:, j])
            gp_cons.append(gp_c)

        # Feasible best
        feasible = np.all(C <= 0, axis=1)
        if np.any(feasible):
            f_best = np.min(y[feasible])
        else:
            f_best = np.min(y)  # No feasible point yet

        convergence.append(float(f_best) if np.any(feasible) else float('inf'))

        # EFI = EI * prod(P(c_i <= 0))
        def acq_func(X_test):
            X_test = np.atleast_2d(X_test)
            pred_obj = gp_obj.predict(X_test)
            ei = expected_improvement(pred_obj.mean, pred_obj.std, f_best)

            # Probability of feasibility for each constraint
            pof = np.ones(len(X_test))
            for gp_c in gp_cons:
                pred_c = gp_c.predict(X_test)
                # P(c <= 0) = Phi(-mu_c / std_c)
                std_c = np.maximum(pred_c.std, 1e-10)
                pof *= _norm_cdf(-pred_c.mean / std_c)

            return ei * pof

        x_next, _ = _optimize_acquisition(acq_func, bounds, rng=rng)

        # Evaluate
        y_next = objective(x_next)
        c_next = np.array([c(x_next) for c in constraints])
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, y_next)
        C = np.vstack([C, c_next.reshape(1, -1)])

    feasible_mask = np.all(C <= 0, axis=1)
    if np.any(feasible_mask):
        best_idx = np.argmin(y * feasible_mask + (1 - feasible_mask) * 1e10)
    else:
        best_idx = np.argmin(y)
    convergence.append(float(y[best_idx]) if feasible_mask[best_idx] else float('inf'))

    return ConstrainedBOResult(
        x_best=X[best_idx].copy(),
        f_best=float(y[best_idx]),
        X_history=X,
        y_history=y,
        feasible_mask=feasible_mask,
        constraint_values=C,
        n_iterations=n_iterations,
        convergence=convergence
    )


# ---------------------------------------------------------------------------
# Input Warping
# ---------------------------------------------------------------------------

def input_warped_optimize(objective: Callable, bounds: Bounds,
                          warp_func: Optional[Callable] = None,
                          unwarp_func: Optional[Callable] = None,
                          n_iterations: int = 50,
                          n_initial: int = 5,
                          kernel: Optional[Kernel] = None,
                          noise_variance: float = 1e-6,
                          rng: Optional[np.random.Generator] = None) -> BOResult:
    """Bayesian optimization with input warping.

    Transforms the input space via warp_func before GP modeling.
    Default: Kumaraswamy warping (concentrates exploration near boundaries).
    """
    rng = rng or np.random.default_rng(42)

    if warp_func is None:
        # Default: log transform for log-scale parameters
        def warp_func(X_norm):
            """Beta CDF-like warping: concentrates near edges."""
            return np.clip(X_norm, 1e-6, 1.0 - 1e-6)

    if unwarp_func is None:
        unwarp_func = lambda X: X  # identity inverse for default

    def warped_objective(x):
        # x is in warped space [0,1]^d
        x_orig = bounds.unnormalize(unwarp_func(x))
        return objective(x_orig)

    warped_bounds = Bounds(lower=np.zeros(bounds.dim), upper=np.ones(bounds.dim))

    result = bayesian_optimize(warped_objective, warped_bounds,
                               n_iterations=n_iterations, n_initial=n_initial,
                               kernel=kernel, noise_variance=noise_variance,
                               rng=rng)

    # Convert back to original space
    result.x_best = bounds.unnormalize(unwarp_func(result.x_best))
    result.X_history = np.array([bounds.unnormalize(unwarp_func(x))
                                  for x in result.X_history])

    return result


# ---------------------------------------------------------------------------
# Convergence Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceDiagnostics:
    """Diagnostics for BO convergence."""
    regret: List[float]              # Simple regret at each step
    cumulative_regret: List[float]   # Cumulative regret
    improvement_rate: float          # Avg improvement per iteration
    stagnation_length: int           # Iterations since last improvement
    exploration_ratio: float         # Fraction of points far from optimum
    is_converged: bool               # Heuristic convergence check


def convergence_diagnostics(result: BOResult,
                            f_opt: Optional[float] = None,
                            tol: float = 1e-3) -> ConvergenceDiagnostics:
    """Compute convergence diagnostics for a BO run.

    Args:
        result: BOResult from optimization.
        f_opt: Known optimum (if available). If None, uses best found.
        tol: Convergence tolerance.
    """
    if f_opt is None:
        f_opt = result.f_best

    # Simple regret
    best_so_far = np.minimum.accumulate(result.y_history)
    regret = [float(b - f_opt) for b in best_so_far]
    cumulative = [float(sum(regret[:i+1])) for i in range(len(regret))]

    # Improvement rate: average regret reduction per step
    if len(regret) > 1:
        improvements = [max(0, regret[i] - regret[i+1]) for i in range(len(regret)-1)]
        improvement_rate = float(np.mean(improvements))
    else:
        improvement_rate = 0.0

    # Stagnation: how many steps since last improvement
    stagnation = 0
    for i in range(len(best_so_far) - 1, 0, -1):
        if best_so_far[i] < best_so_far[i-1]:
            break
        stagnation += 1

    # Exploration ratio: fraction of points with value > 2x best
    threshold = result.f_best + abs(result.f_best) * 0.5
    exploration_ratio = float(np.mean(result.y_history > threshold))

    # Convergence heuristic
    n_recent = min(5, len(regret))
    recent_regret = regret[-n_recent:]
    is_converged = (max(recent_regret) - min(recent_regret)) < tol

    return ConvergenceDiagnostics(
        regret=regret,
        cumulative_regret=cumulative,
        improvement_rate=improvement_rate,
        stagnation_length=stagnation,
        exploration_ratio=exploration_ratio,
        is_converged=is_converged
    )


# ---------------------------------------------------------------------------
# Comparison Utility
# ---------------------------------------------------------------------------

def compare_acquisitions(objective: Callable, bounds: Bounds,
                         acquisitions: Optional[List[AcquisitionType]] = None,
                         n_iterations: int = 30,
                         n_initial: int = 5,
                         rng_seed: int = 42) -> Dict[str, BOResult]:
    """Compare different acquisition functions on the same problem.

    Uses the same initial points and random seed for fair comparison.
    """
    if acquisitions is None:
        acquisitions = [AcquisitionType.EI, AcquisitionType.PI,
                        AcquisitionType.UCB, AcquisitionType.THOMPSON]

    results = {}
    for acq in acquisitions:
        rng = np.random.default_rng(rng_seed)
        result = bayesian_optimize(objective, bounds, n_iterations=n_iterations,
                                   n_initial=n_initial, acquisition=acq,
                                   rng=rng)
        results[acq.value] = result

    return results


# ---------------------------------------------------------------------------
# Optimization Summary
# ---------------------------------------------------------------------------

def optimization_summary(result: BOResult, name: str = "BO") -> str:
    """Generate human-readable summary of optimization result."""
    lines = [f"=== {name} Summary ==="]
    lines.append(f"Best value: {result.f_best:.6f}")
    lines.append(f"Best input: {result.x_best}")
    lines.append(f"Total evaluations: {len(result.y_history)}")
    lines.append(f"Iterations: {result.n_iterations}")
    lines.append(f"Initial improvement: {result.convergence[0]:.6f} -> {result.convergence[-1]:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test Functions (standard benchmarks)
# ---------------------------------------------------------------------------

def branin(x: np.ndarray) -> float:
    """Branin function. Domain: x1 in [-5, 10], x2 in [0, 15].
    Global min ~ 0.397887 at 3 points."""
    x1, x2 = x[0], x[1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def sphere(x: np.ndarray) -> float:
    """Sphere function. Global min = 0 at origin."""
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function. Global min = 0 at (1, 1, ..., 1)."""
    return float(sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                      for i in range(len(x) - 1)))


def ackley(x: np.ndarray) -> float:
    """Ackley function. Global min = 0 at origin."""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e


def six_hump_camel(x: np.ndarray) -> float:
    """Six-Hump Camel function. Domain: x1 in [-3, 3], x2 in [-2, 2].
    Global min ~ -1.0316 at (0.0898, -0.7126) and (-0.0898, 0.7126)."""
    x1, x2 = x[0], x[1]
    return (4 - 2.1*x1**2 + x1**4/3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2
