"""
V233: Robust Bayesian Optimization
Composes V230 (Transfer BO) + V222 (Gaussian Process) with adversarial/noise robustness.

Handles optimization under uncertainty:
- Input noise: optimize when inputs can't be set precisely
- Distributional robustness: worst-case over ambiguity sets
- Min-max: robust to adversarial environmental variables
- Robust transfer: share robustness profiles across related tasks

Built from scratch using NumPy. No ML libraries.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple
from enum import Enum
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V230_transfer_bo'))

from gaussian_process import (
    GaussianProcess, RBFKernel, Matern52Kernel, Kernel
)
from transfer_bo import (
    BOTask, TransferBOResult, TaskDatabase,
    transfer_bo, cold_start_bo, compute_transfer_weight,
    auto_select_sources, find_source_tasks
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RobustStrategy(Enum):
    """Robustness strategy."""
    INPUT_NOISE = "input_noise"           # Noisy input BO
    DISTRIBUTIONALLY_ROBUST = "dro"       # Worst-case over distribution set
    MINIMAX = "minimax"                   # Min-max over adversarial vars
    WORST_CASE_SENSITIVITY = "wcs"        # Worst-case sensitivity analysis


@dataclass
class RobustBOResult:
    """Result of robust optimization."""
    x_best: np.ndarray
    f_best: float
    X_history: np.ndarray
    y_history: np.ndarray
    total_evaluations: int
    convergence: List[float]
    robustness_scores: List[float]  # Per-iteration robustness measure
    strategy: str
    input_noise_std: Optional[float] = None
    worst_case_gap: Optional[float] = None  # gap between nominal and worst-case
    sensitivity_profile: Optional[Dict] = None


@dataclass
class RobustnessProfile:
    """Robustness profile for a solution."""
    x: np.ndarray
    nominal_value: float
    worst_case_value: float
    robustness_radius: float      # Max perturbation before degradation
    sensitivity: np.ndarray       # Per-dimension sensitivity
    robustness_score: float       # 0-1 score (1 = perfectly robust)


@dataclass
class AdversarialResult:
    """Result of min-max optimization."""
    x_best: np.ndarray              # Design variables (our choice)
    z_worst: np.ndarray             # Adversarial variables (worst-case)
    f_robust: float                 # Worst-case objective value
    f_nominal: float                # Nominal (no adversary) value
    X_design_history: np.ndarray
    Z_adversary_history: np.ndarray
    y_history: np.ndarray
    robustness_gap: float           # f_nominal - f_robust


# ---------------------------------------------------------------------------
# Acquisition functions for robust BO
# ---------------------------------------------------------------------------

def expected_improvement(mu, sigma, f_best):
    """Standard EI."""
    with np.errstate(divide='ignore', invalid='ignore'):
        improvement = mu - f_best
        z = np.where(sigma > 1e-10, improvement / sigma, 0.0)
        ei = np.where(
            sigma > 1e-10,
            improvement * _norm_cdf(z) + sigma * _norm_pdf(z),
            np.maximum(improvement, 0.0)
        )
    return ei


def robust_expected_improvement(gp, X_candidates, f_best, noise_std, n_samples=50, rng=None):
    """
    Expected improvement averaged over input perturbations.
    EI_robust(x) = E_{eps ~ N(0, noise_std^2)} [EI(x + eps)]
    Uses unscented transform for efficiency.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = X_candidates.shape[1]
    rei = np.zeros(X_candidates.shape[0])

    # Sigma points (unscented transform)
    sigma_points = _unscented_sigma_points(d, noise_std)
    n_sigma = len(sigma_points)
    weight = 1.0 / n_sigma

    for sp in sigma_points:
        perturbed = X_candidates + sp.reshape(1, -1)
        pred = gp.predict(perturbed)
        mu = pred.mean.ravel()
        std = pred.std.ravel()
        rei += weight * expected_improvement(mu, std, f_best)

    return rei


def worst_case_expected_improvement(gp, X_candidates, f_best, noise_std, n_adversarial=20, rng=None):
    """
    Worst-case EI over input perturbations.
    EI_wc(x) = min_{||eps|| <= noise_std} EI(x + eps)
    Approximated by sampling perturbations and taking minimum.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = X_candidates.shape[1]
    n = X_candidates.shape[0]
    wc_ei = np.full(n, np.inf)

    for _ in range(n_adversarial):
        # Random direction, scaled to noise_std ball
        direction = rng.randn(n, d)
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        direction = direction / norms
        radius = rng.uniform(0, noise_std, size=(n, 1))
        perturbation = direction * radius

        perturbed = X_candidates + perturbation
        pred = gp.predict(perturbed)
        mu = pred.mean.ravel()
        std = pred.std.ravel()
        ei = expected_improvement(mu, std, f_best)
        wc_ei = np.minimum(wc_ei, ei)

    # Replace inf with 0 (no valid perturbation found)
    wc_ei = np.where(np.isinf(wc_ei), 0.0, wc_ei)
    return wc_ei


def distributionally_robust_ei(gp, X_candidates, f_best, ambiguity_radius=0.1, n_samples=30, rng=None):
    """
    Distributionally robust EI using CVaR (Conditional Value-at-Risk).
    Optimizes worst-case EI over distributions within Wasserstein ball.
    Approximated via CVaR_alpha where alpha controls conservatism.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = X_candidates.shape[0]
    d = X_candidates.shape[1]

    # Sample perturbations from reference distribution
    all_ei = []
    for _ in range(n_samples):
        perturbation = rng.randn(n, d) * ambiguity_radius
        perturbed = X_candidates + perturbation
        pred = gp.predict(perturbed)
        mu = pred.mean.ravel()
        std = pred.std.ravel()
        ei = expected_improvement(mu, std, f_best)
        all_ei.append(ei)

    all_ei = np.array(all_ei)  # (n_samples, n_candidates)

    # CVaR at alpha=0.2 (worst 20% of samples)
    alpha = 0.2
    k = max(1, int(np.ceil(alpha * n_samples)))
    sorted_ei = np.sort(all_ei, axis=0)  # ascending
    cvar = np.mean(sorted_ei[:k], axis=0)  # mean of worst k samples

    return cvar


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    """Normal CDF via Abramowitz-Stegun."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _erf(x):
    """Error function approximation (Abramowitz-Stegun)."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
           t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def _unscented_sigma_points(d, noise_std):
    """Generate 2d+1 sigma points for unscented transform."""
    points = [np.zeros(d)]
    scale = noise_std * np.sqrt(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = scale
        points.append(e)
        points.append(-e)
    return points


def _latin_hypercube_sample(bounds, n_samples, rng):
    """Latin hypercube sampling within bounds."""
    d = bounds.shape[0]
    result = np.zeros((n_samples, d))
    for i in range(d):
        perm = rng.permutation(n_samples)
        intervals = np.linspace(0, 1, n_samples + 1)
        points = intervals[:-1] + rng.uniform(0, 1.0 / n_samples, size=n_samples)
        result[:, i] = bounds[i, 0] + points[perm] * (bounds[i, 1] - bounds[i, 0])
    return result


def _clip_to_bounds(X, bounds):
    """Clip points to bounds."""
    clipped = X.copy()
    for i in range(bounds.shape[0]):
        clipped[:, i] = np.clip(clipped[:, i], bounds[i, 0], bounds[i, 1])
    return clipped


# ---------------------------------------------------------------------------
# Core: Robust BO with input noise
# ---------------------------------------------------------------------------

def robust_bo(
    objective: Callable,
    bounds: np.ndarray,
    input_noise_std: float = 0.1,
    n_iterations: int = 30,
    n_initial: int = 5,
    acquisition: str = "robust_ei",  # "robust_ei", "worst_case_ei", "dro_ei"
    kernel: Optional[Kernel] = None,
    noise_variance: float = 0.01,
    n_candidates: int = 200,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> RobustBOResult:
    """
    Bayesian optimization robust to input noise.

    The optimizer accounts for the fact that the chosen input x will be
    perturbed by noise ~ N(0, input_noise_std^2) at execution time.

    Args:
        objective: f(x) -> float (maximized)
        bounds: (d, 2) array of [lo, hi] per dimension
        input_noise_std: Standard deviation of input perturbation
        n_iterations: Number of BO iterations
        n_initial: Number of initial random evaluations
        acquisition: Which robust acquisition function to use
        kernel: GP kernel (default: RBF)
        noise_variance: GP observation noise
        n_candidates: Number of candidate points per iteration
        rng: Random state
        verbose: Print progress

    Returns:
        RobustBOResult with optimization history and robustness scores
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = bounds.shape[0]
    if kernel is None:
        length_scale = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
        kernel = RBFKernel(length_scale=length_scale, variance=1.0)

    # Initial samples
    X = _latin_hypercube_sample(bounds, n_initial, rng)
    y = np.array([objective(x) for x in X])

    convergence = [float(np.max(y))]
    robustness_scores = [0.0]  # Unknown initially

    for i in range(n_iterations):
        # Fit GP
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(X, y)

        f_best = float(np.max(y))

        # Generate candidates
        candidates = _latin_hypercube_sample(bounds, n_candidates, rng)

        # Compute robust acquisition
        if acquisition == "robust_ei":
            acq_values = robust_expected_improvement(
                gp, candidates, f_best, input_noise_std, rng=rng
            )
        elif acquisition == "worst_case_ei":
            acq_values = worst_case_expected_improvement(
                gp, candidates, f_best, input_noise_std, rng=rng
            )
        elif acquisition == "dro_ei":
            acq_values = distributionally_robust_ei(
                gp, candidates, f_best, ambiguity_radius=input_noise_std, rng=rng
            )
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

        # Select best candidate
        best_idx = np.argmax(acq_values)
        x_new = candidates[best_idx]

        # Evaluate (with simulated input noise for robustness score)
        y_new = objective(x_new)
        X = np.vstack([X, x_new.reshape(1, -1)])
        y = np.append(y, y_new)

        # Compute robustness score for this point
        r_score = _compute_robustness_score(objective, x_new, bounds, input_noise_std, rng)
        robustness_scores.append(r_score)
        convergence.append(float(np.max(y)))

        if verbose:
            print(f"  Iter {i+1}: f={y_new:.4f}, best={np.max(y):.4f}, robust={r_score:.3f}")

    best_idx = np.argmax(y)
    # Compute worst-case gap
    nominal = float(y[best_idx])
    wc = _worst_case_value(objective, X[best_idx], bounds, input_noise_std, rng)

    return RobustBOResult(
        x_best=X[best_idx].copy(),
        f_best=nominal,
        X_history=X.copy(),
        y_history=y.copy(),
        total_evaluations=len(y),
        convergence=convergence,
        robustness_scores=robustness_scores,
        strategy=acquisition,
        input_noise_std=input_noise_std,
        worst_case_gap=nominal - wc
    )


def _compute_robustness_score(objective, x, bounds, noise_std, rng, n_samples=30):
    """Robustness score: fraction of perturbed evaluations within 80% of nominal."""
    nominal = objective(x)
    d = len(x)
    count_good = 0
    for _ in range(n_samples):
        perturbation = rng.randn(d) * noise_std
        x_pert = np.clip(x + perturbation, bounds[:, 0], bounds[:, 1])
        val = objective(x_pert)
        # "Good" if within 80% of nominal (or better)
        if nominal >= 0:
            if val >= 0.8 * nominal:
                count_good += 1
        else:
            if val >= nominal - 0.2 * abs(nominal):
                count_good += 1
    return count_good / n_samples


def _worst_case_value(objective, x, bounds, noise_std, rng, n_samples=50):
    """Estimate worst-case objective value under perturbation."""
    d = len(x)
    worst = objective(x)
    for _ in range(n_samples):
        direction = rng.randn(d)
        direction = direction / max(np.linalg.norm(direction), 1e-10)
        x_pert = np.clip(x + direction * noise_std, bounds[:, 0], bounds[:, 1])
        val = objective(x_pert)
        worst = min(worst, val)
    return worst


# ---------------------------------------------------------------------------
# Min-max (adversarial) BO
# ---------------------------------------------------------------------------

def minimax_bo(
    objective: Callable,
    design_bounds: np.ndarray,
    adversary_bounds: np.ndarray,
    n_iterations: int = 30,
    n_initial: int = 5,
    n_inner: int = 10,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 0.01,
    n_candidates: int = 200,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> AdversarialResult:
    """
    Min-max Bayesian optimization: max_x min_z f(x, z).

    The design player (us) chooses x to maximize; the adversary chooses z
    to minimize. We alternate: for each design candidate, find worst-case z,
    then pick x with best worst-case value.

    Args:
        objective: f(x, z) -> float where x is design, z is adversarial
        design_bounds: (d_x, 2) bounds for design variables
        adversary_bounds: (d_z, 2) bounds for adversarial variables
        n_iterations: Number of outer iterations
        n_initial: Initial random evaluations
        n_inner: Inner adversary optimization samples
        kernel: GP kernel
        noise_variance: GP noise
        n_candidates: Candidates per iteration
        rng: Random state
        verbose: Print progress

    Returns:
        AdversarialResult with design and adversary histories
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d_x = design_bounds.shape[0]
    d_z = adversary_bounds.shape[0]
    d_total = d_x + d_z

    if kernel is None:
        total_bounds = np.vstack([design_bounds, adversary_bounds])
        length_scale = 0.3 * np.mean(total_bounds[:, 1] - total_bounds[:, 0])
        kernel = RBFKernel(length_scale=length_scale, variance=1.0)

    total_bounds = np.vstack([design_bounds, adversary_bounds])

    # Initial samples in joint (x, z) space
    XZ = _latin_hypercube_sample(total_bounds, n_initial, rng)
    y = np.array([objective(xz[:d_x], xz[d_x:]) for xz in XZ])

    X_design_hist = []
    Z_adversary_hist = []

    for i in range(n_iterations):
        # Fit joint GP over (x, z)
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(XZ, y)

        # Generate design candidates
        x_candidates = _latin_hypercube_sample(design_bounds, n_candidates, rng)

        # For each design candidate, find worst-case z
        worst_case_values = np.full(n_candidates, np.inf)
        worst_case_z = np.zeros((n_candidates, d_z))

        z_samples = _latin_hypercube_sample(adversary_bounds, n_inner, rng)

        for j, x_cand in enumerate(x_candidates):
            # Evaluate GP at (x_cand, z) for each z sample
            xz_pairs = np.hstack([
                np.tile(x_cand, (n_inner, 1)),
                z_samples
            ])
            pred = gp.predict(xz_pairs, return_std=False)
            mu = pred.mean.ravel()

            # Adversary picks z that minimizes
            worst_idx = np.argmin(mu)
            worst_case_values[j] = mu[worst_idx]
            worst_case_z[j] = z_samples[worst_idx]

        # Design player picks x with best worst-case
        best_design_idx = np.argmax(worst_case_values)
        x_new = x_candidates[best_design_idx]
        z_new = worst_case_z[best_design_idx]

        # Evaluate true objective at (x_new, z_new)
        y_new = objective(x_new, z_new)

        xz_new = np.concatenate([x_new, z_new])
        XZ = np.vstack([XZ, xz_new.reshape(1, -1)])
        y = np.append(y, y_new)

        X_design_hist.append(x_new.copy())
        Z_adversary_hist.append(z_new.copy())

        if verbose:
            print(f"  Iter {i+1}: f(x,z)={y_new:.4f}, wc={worst_case_values[best_design_idx]:.4f}")

    # Find best robust design: the one with highest worst-case value
    # Re-evaluate worst-case for all observed design points
    best_robust_val = -np.inf
    best_x = None
    best_z_worst = None

    unique_designs = XZ[:, :d_x]
    for x_d in unique_designs:
        z_test = _latin_hypercube_sample(adversary_bounds, n_inner * 2, rng)
        xz_test = np.hstack([np.tile(x_d, (len(z_test), 1)), z_test])
        pred = gp.predict(xz_test, return_std=False)
        mu = pred.mean.ravel()
        wc_idx = np.argmin(mu)
        wc_val = mu[wc_idx]
        if wc_val > best_robust_val:
            best_robust_val = wc_val
            best_x = x_d.copy()
            best_z_worst = z_test[wc_idx].copy()

    # Nominal value (adversary at midpoint)
    z_mid = 0.5 * (adversary_bounds[:, 0] + adversary_bounds[:, 1])
    f_nominal = objective(best_x, z_mid)
    f_robust = objective(best_x, best_z_worst)

    return AdversarialResult(
        x_best=best_x,
        z_worst=best_z_worst,
        f_robust=f_robust,
        f_nominal=f_nominal,
        X_design_history=np.array(X_design_hist) if X_design_hist else np.empty((0, d_x)),
        Z_adversary_history=np.array(Z_adversary_hist) if Z_adversary_hist else np.empty((0, d_z)),
        y_history=y.copy(),
        robustness_gap=f_nominal - f_robust
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    objective: Callable,
    x: np.ndarray,
    bounds: np.ndarray,
    perturbation_scales: Optional[List[float]] = None,
    n_samples: int = 50,
    rng: Optional[np.random.RandomState] = None
) -> RobustnessProfile:
    """
    Analyze robustness of a solution point.

    Computes per-dimension sensitivity, worst-case value, and robustness
    radius (max perturbation before significant degradation).

    Args:
        objective: f(x) -> float
        x: Solution point to analyze
        bounds: (d, 2) bounds
        perturbation_scales: List of perturbation radii to test
        n_samples: Samples per perturbation level
        rng: Random state

    Returns:
        RobustnessProfile with full analysis
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = len(x)
    nominal = objective(x)

    if perturbation_scales is None:
        extent = np.mean(bounds[:, 1] - bounds[:, 0])
        perturbation_scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
        perturbation_scales = [s * extent for s in perturbation_scales]

    # Per-dimension sensitivity (finite differences)
    sensitivity = np.zeros(d)
    for dim in range(d):
        delta = 0.01 * (bounds[dim, 1] - bounds[dim, 0])
        x_plus = x.copy()
        x_plus[dim] = min(x[dim] + delta, bounds[dim, 1])
        x_minus = x.copy()
        x_minus[dim] = max(x[dim] - delta, bounds[dim, 0])
        f_plus = objective(x_plus)
        f_minus = objective(x_minus)
        sensitivity[dim] = abs(f_plus - f_minus) / (2 * delta) if delta > 0 else 0

    # Worst-case value and robustness radius
    worst_case = nominal
    robustness_radius = perturbation_scales[-1]  # Default: robust at all scales
    degradation_threshold = 0.2 * abs(nominal) if abs(nominal) > 1e-10 else 0.1

    for scale in perturbation_scales:
        scale_worst = nominal
        for _ in range(n_samples):
            direction = rng.randn(d)
            direction = direction / max(np.linalg.norm(direction), 1e-10)
            x_pert = np.clip(x + direction * scale, bounds[:, 0], bounds[:, 1])
            val = objective(x_pert)
            scale_worst = min(scale_worst, val)

        worst_case = min(worst_case, scale_worst)

        if nominal - scale_worst > degradation_threshold and scale <= robustness_radius:
            robustness_radius = scale
            break

    # Robustness score: 1 - (nominal - worst_case) / |nominal|
    if abs(nominal) > 1e-10:
        gap_ratio = (nominal - worst_case) / abs(nominal)
        robustness_score = max(0.0, 1.0 - gap_ratio)
    else:
        robustness_score = 1.0 if abs(nominal - worst_case) < 0.01 else 0.5

    return RobustnessProfile(
        x=x.copy(),
        nominal_value=nominal,
        worst_case_value=worst_case,
        robustness_radius=robustness_radius,
        sensitivity=sensitivity,
        robustness_score=robustness_score
    )


# ---------------------------------------------------------------------------
# Robust transfer BO (composes V230)
# ---------------------------------------------------------------------------

def robust_transfer_bo(
    objective: Callable,
    bounds: np.ndarray,
    source_tasks: List[BOTask],
    input_noise_std: float = 0.1,
    n_iterations: int = 30,
    n_initial: int = 5,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 0.01,
    n_candidates: int = 200,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> RobustBOResult:
    """
    Robust BO with transfer learning from source tasks.

    Uses source task data to warm-start the GP, then applies robust
    acquisition functions. Source tasks provide prior knowledge about
    the function landscape, reducing the iterations needed.

    Also transfers robustness information: solutions from source tasks
    that were robust under perturbation receive higher transfer weight.

    Args:
        objective: f(x) -> float
        bounds: (d, 2) bounds
        source_tasks: List of BOTask from related problems
        input_noise_std: Input perturbation std
        n_iterations: BO iterations
        n_initial: Initial evaluations
        kernel: GP kernel
        noise_variance: GP noise
        n_candidates: Candidates per iteration
        rng: Random state
        verbose: Print progress

    Returns:
        RobustBOResult with transfer information
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = bounds.shape[0]
    if kernel is None:
        length_scale = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
        kernel = RBFKernel(length_scale=length_scale, variance=1.0)

    # Compute transfer weights with robustness bonus
    transfer_weights = []
    for task in source_tasks:
        base_weight = compute_transfer_weight(task, BOTask(
            X=np.empty((0, d)), y=np.empty(0), bounds=bounds
        ), kernel=kernel, noise_variance=noise_variance)

        # Robustness bonus: evaluate how robust the source's best point is
        if task.x_best is not None and task.f_best is not None:
            rob_score = _source_robustness_score(task, bounds, input_noise_std, rng)
            adjusted_weight = base_weight * (0.5 + 0.5 * rob_score)
        else:
            adjusted_weight = base_weight

        transfer_weights.append(adjusted_weight)

    # Normalize weights
    total = sum(transfer_weights)
    if total > 0:
        transfer_weights = [w / total for w in transfer_weights]
    else:
        transfer_weights = [1.0 / len(source_tasks)] * len(source_tasks) if source_tasks else []

    # Collect weighted source data
    source_X_list = []
    source_y_list = []
    for task, weight in zip(source_tasks, transfer_weights):
        if weight > 0.05 and len(task.X) > 0:
            # Subsample proportional to weight
            n_use = max(1, int(weight * 20))
            n_use = min(n_use, len(task.X))
            indices = rng.choice(len(task.X), n_use, replace=False)
            source_X_list.append(task.X[indices])
            source_y_list.append(task.y[indices])

    # Initial samples
    X = _latin_hypercube_sample(bounds, n_initial, rng)
    y = np.array([objective(x) for x in X])

    # Combine with source data
    if source_X_list:
        all_source_X = np.vstack(source_X_list)
        all_source_y = np.concatenate(source_y_list)
        # Clip source data to target bounds
        all_source_X = _clip_to_bounds(all_source_X, bounds)
        X_combined = np.vstack([all_source_X, X])
        y_combined = np.concatenate([all_source_y, y])
    else:
        X_combined = X.copy()
        y_combined = y.copy()

    convergence = [float(np.max(y))]
    robustness_scores = [0.0]

    for i in range(n_iterations):
        # Fit GP on combined data
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(X_combined, y_combined)

        f_best = float(np.max(y))  # Best from target evaluations only

        # Robust acquisition on candidates
        candidates = _latin_hypercube_sample(bounds, n_candidates, rng)
        acq_values = robust_expected_improvement(
            gp, candidates, f_best, input_noise_std, rng=rng
        )

        best_idx = np.argmax(acq_values)
        x_new = candidates[best_idx]
        y_new = objective(x_new)

        # Update target data
        X = np.vstack([X, x_new.reshape(1, -1)])
        y = np.append(y, y_new)

        # Update combined data (re-weight source data with decay)
        decay = max(0.1, 1.0 - i / n_iterations)  # Reduce source influence over time
        if source_X_list:
            decayed_source_y = all_source_y * decay
            X_combined = np.vstack([all_source_X, X])
            y_combined = np.concatenate([decayed_source_y, y])
        else:
            X_combined = X.copy()
            y_combined = y.copy()

        r_score = _compute_robustness_score(objective, x_new, bounds, input_noise_std, rng)
        robustness_scores.append(r_score)
        convergence.append(float(np.max(y)))

        if verbose:
            print(f"  Iter {i+1}: f={y_new:.4f}, best={np.max(y):.4f}, robust={r_score:.3f}")

    best_idx = np.argmax(y)
    nominal = float(y[best_idx])
    wc = _worst_case_value(objective, X[best_idx], bounds, input_noise_std, rng)

    return RobustBOResult(
        x_best=X[best_idx].copy(),
        f_best=nominal,
        X_history=X.copy(),
        y_history=y.copy(),
        total_evaluations=len(y),
        convergence=convergence,
        robustness_scores=robustness_scores,
        strategy="robust_transfer",
        input_noise_std=input_noise_std,
        worst_case_gap=nominal - wc
    )


def _source_robustness_score(task, target_bounds, noise_std, rng, n_samples=20):
    """Estimate how robust a source task's best point is (proxy via variance)."""
    if task.x_best is None or len(task.X) < 2:
        return 0.5

    # Use the variance of y values near x_best as a proxy
    dists = np.linalg.norm(task.X - task.x_best.reshape(1, -1), axis=1)
    radius = noise_std * 2
    nearby = dists < radius
    if np.sum(nearby) < 2:
        return 0.5

    y_nearby = task.y[nearby]
    if abs(task.f_best) > 1e-10:
        cv = np.std(y_nearby) / abs(task.f_best)
        return max(0.0, 1.0 - cv)
    return 0.5


# ---------------------------------------------------------------------------
# Compare strategies
# ---------------------------------------------------------------------------

def compare_robust_strategies(
    objective: Callable,
    bounds: np.ndarray,
    input_noise_std: float = 0.1,
    n_iterations: int = 20,
    n_initial: int = 5,
    noise_variance: float = 0.01,
    rng_seed: int = 42
) -> Dict[str, RobustBOResult]:
    """
    Compare different robust acquisition strategies on the same problem.

    Returns dict mapping strategy name to RobustBOResult.
    """
    strategies = ["robust_ei", "worst_case_ei", "dro_ei"]
    results = {}

    for strategy in strategies:
        rng = np.random.RandomState(rng_seed)
        result = robust_bo(
            objective=objective,
            bounds=bounds,
            input_noise_std=input_noise_std,
            n_iterations=n_iterations,
            n_initial=n_initial,
            acquisition=strategy,
            noise_variance=noise_variance,
            rng=rng,
            verbose=False
        )
        results[strategy] = result

    # Also run non-robust (standard BO) for comparison
    rng = np.random.RandomState(rng_seed)
    cold_result = cold_start_bo(
        objective=objective,
        bounds=bounds,
        n_iterations=n_iterations,
        n_initial=n_initial,
        noise_variance=noise_variance,
        rng=rng
    )
    # Wrap cold_result into RobustBOResult
    results["standard_bo"] = RobustBOResult(
        x_best=cold_result.x_best,
        f_best=cold_result.f_best,
        X_history=cold_result.X_history,
        y_history=cold_result.y_history,
        total_evaluations=cold_result.total_evaluations,
        convergence=cold_result.convergence,
        robustness_scores=[],
        strategy="standard_bo",
        input_noise_std=input_noise_std
    )

    return results


# ---------------------------------------------------------------------------
# Robustness certification
# ---------------------------------------------------------------------------

def certify_robustness(
    objective: Callable,
    x: np.ndarray,
    bounds: np.ndarray,
    epsilon: float,
    n_samples: int = 100,
    confidence: float = 0.95,
    rng: Optional[np.random.RandomState] = None
) -> Dict:
    """
    Statistical certification of robustness.

    Estimates probability that f(x + eps) >= threshold for ||eps|| <= epsilon.
    Uses Hoeffding-style bound for finite-sample guarantee.

    Args:
        objective: f(x) -> float
        x: Point to certify
        bounds: Input bounds
        epsilon: Perturbation radius
        n_samples: Number of Monte Carlo samples
        confidence: Confidence level for the bound
        rng: Random state

    Returns:
        Dict with certification results
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = len(x)
    nominal = objective(x)

    # Threshold: tolerate up to 20% degradation from nominal
    if abs(nominal) > 1e-10:
        threshold = nominal - 0.2 * abs(nominal)
    else:
        threshold = -0.1

    # Sample perturbations uniformly from epsilon-ball
    successes = 0
    values = []

    for _ in range(n_samples):
        direction = rng.randn(d)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.zeros(d)
            direction[0] = 1.0
            norm = 1.0
        direction = direction / norm
        radius = rng.uniform(0, epsilon)
        x_pert = np.clip(x + direction * radius, bounds[:, 0], bounds[:, 1])
        val = objective(x_pert)
        values.append(val)
        if val >= threshold:
            successes += 1

    # Empirical probability
    p_hat = successes / n_samples

    # Hoeffding bound for confidence interval
    hoeffding_term = np.sqrt(np.log(2 / (1 - confidence)) / (2 * n_samples))
    p_lower = max(0.0, p_hat - hoeffding_term)
    p_upper = min(1.0, p_hat + hoeffding_term)

    values = np.array(values)
    return {
        "certified": p_lower >= 0.8,  # Certify if lower bound >= 80%
        "probability_robust": p_hat,
        "confidence_lower": p_lower,
        "confidence_upper": p_upper,
        "confidence_level": confidence,
        "nominal_value": nominal,
        "threshold": threshold,
        "epsilon": epsilon,
        "n_samples": n_samples,
        "mean_perturbed": float(np.mean(values)),
        "std_perturbed": float(np.std(values)),
        "min_perturbed": float(np.min(values)),
        "max_perturbed": float(np.max(values))
    }


# ---------------------------------------------------------------------------
# Pareto-robust multi-objective
# ---------------------------------------------------------------------------

def pareto_robust_bo(
    objectives: List[Callable],
    bounds: np.ndarray,
    input_noise_std: float = 0.1,
    n_iterations: int = 30,
    n_initial: int = 5,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 0.01,
    n_candidates: int = 200,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Dict:
    """
    Multi-objective robust BO.

    Finds Pareto-optimal solutions that are also robust to input noise.
    Uses robust expected hypervolume improvement.

    Args:
        objectives: List of f_i(x) -> float (all maximized)
        bounds: (d, 2) bounds
        input_noise_std: Perturbation std
        n_iterations: Number of iterations
        n_initial: Initial evaluations
        kernel: GP kernel
        noise_variance: GP noise
        n_candidates: Candidates per iteration
        rng: Random state
        verbose: Print progress

    Returns:
        Dict with pareto_front, X_history, Y_history, robustness_scores
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = bounds.shape[0]
    n_obj = len(objectives)

    if kernel is None:
        length_scale = 0.3 * np.mean(bounds[:, 1] - bounds[:, 0])
        kernel = RBFKernel(length_scale=length_scale, variance=1.0)

    # Initial samples
    X = _latin_hypercube_sample(bounds, n_initial, rng)
    Y = np.array([[obj(x) for obj in objectives] for x in X])

    for i in range(n_iterations):
        # Fit one GP per objective
        gps = []
        for j in range(n_obj):
            gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
            gp.fit(X, Y[:, j])
            gps.append(gp)

        # Generate candidates
        candidates = _latin_hypercube_sample(bounds, n_candidates, rng)

        # Compute robust scalarized acquisition (Chebyshev scalarization with random weights)
        weights = rng.dirichlet(np.ones(n_obj))
        ref_point = np.min(Y, axis=0) - 0.1 * np.abs(np.min(Y, axis=0) + 1e-10)

        # Robust prediction: average over sigma points
        sigma_points = _unscented_sigma_points(d, input_noise_std)
        n_sp = len(sigma_points)

        acq_values = np.zeros(n_candidates)
        for sp in sigma_points:
            perturbed = candidates + sp.reshape(1, -1)
            perturbed = _clip_to_bounds(perturbed, bounds)
            pred_means = np.zeros((n_candidates, n_obj))
            for j, gp in enumerate(gps):
                pred = gp.predict(perturbed, return_std=False)
                pred_means[:, j] = pred.mean.ravel()

            # Chebyshev scalarization: min_j weights[j] * (pred_j - ref_j)
            weighted = weights * (pred_means - ref_point)
            scalarized = np.min(weighted, axis=1)
            acq_values += scalarized / n_sp

        # Select best
        best_idx = np.argmax(acq_values)
        x_new = candidates[best_idx]
        y_new = np.array([obj(x_new) for obj in objectives])

        X = np.vstack([X, x_new.reshape(1, -1)])
        Y = np.vstack([Y, y_new.reshape(1, -1)])

        if verbose:
            print(f"  Iter {i+1}: y={y_new}")

    # Extract Pareto front
    pareto_mask = _pareto_efficient(Y)
    pareto_X = X[pareto_mask]
    pareto_Y = Y[pareto_mask]

    # Compute robustness for each Pareto point
    pareto_robustness = []
    for x_p, y_p in zip(pareto_X, pareto_Y):
        scores = []
        for j, obj in enumerate(objectives):
            r = _compute_robustness_score(obj, x_p, bounds, input_noise_std, rng, n_samples=20)
            scores.append(r)
        pareto_robustness.append(float(np.mean(scores)))

    return {
        "pareto_X": pareto_X,
        "pareto_Y": pareto_Y,
        "pareto_robustness": pareto_robustness,
        "X_history": X,
        "Y_history": Y,
        "n_pareto": len(pareto_X),
        "total_evaluations": len(X)
    }


def _pareto_efficient(Y):
    """Find Pareto-efficient points (all objectives maximized)."""
    n = len(Y)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        for j in range(n):
            if i == j or not is_efficient[j]:
                continue
            # j dominates i if j >= i in all objectives and j > i in at least one
            if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                is_efficient[i] = False
                break
    return is_efficient


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def robust_bo_summary(result: RobustBOResult, name: str = "Robust BO") -> str:
    """Human-readable summary of robust BO result."""
    lines = [
        f"=== {name} ===",
        f"Strategy: {result.strategy}",
        f"Best value: {result.f_best:.6f}",
        f"Best point: {result.x_best}",
        f"Total evaluations: {result.total_evaluations}",
    ]
    if result.input_noise_std is not None:
        lines.append(f"Input noise std: {result.input_noise_std:.4f}")
    if result.worst_case_gap is not None:
        lines.append(f"Worst-case gap: {result.worst_case_gap:.6f}")
    if result.robustness_scores:
        avg_rob = np.mean(result.robustness_scores[-5:]) if len(result.robustness_scores) >= 5 else np.mean(result.robustness_scores)
        lines.append(f"Avg robustness (last 5): {avg_rob:.3f}")
    return "\n".join(lines)
