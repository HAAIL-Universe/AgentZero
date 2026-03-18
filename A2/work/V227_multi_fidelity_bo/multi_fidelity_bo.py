"""V227: Multi-Fidelity Bayesian Optimization

Composes V223 (Bayesian Optimization) + V222 (Gaussian Process) to enable
cost-efficient optimization using cheap low-fidelity evaluations to guide
expensive high-fidelity evaluations.

Key concepts:
- Multiple fidelity levels with different costs and accuracies
- Multi-fidelity GP: models correlation across fidelities
- Cost-aware acquisition: information per unit cost
- Continuous fidelity: fidelity as a continuous parameter [0, 1]

Methods implemented:
1. Multi-Fidelity GP (fidelity as input dimension with correlation kernel)
2. Cost-Aware Expected Improvement (EI / cost)
3. Multi-Fidelity Knowledge Gradient (value of information at each fidelity)
4. Continuous-Fidelity BO (fidelity as continuous optimization variable)
5. Multi-Task BO (independent GP per fidelity with shared structure)
6. Entropy Search for Multi-Fidelity (information-theoretic)

Composition: V222 GaussianProcess for surrogate modeling,
             V223 acquisition functions and optimization infrastructure.
"""

import numpy as np
from numpy.random import Generator, default_rng
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Optional, Tuple, Any
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V223_bayesian_optimization'))

from gaussian_process import (
    GaussianProcess, Kernel, RBFKernel, Matern52Kernel, ScaleKernel,
    ARDKernel, GPPrediction
)
from bayesian_optimization import (
    expected_improvement, Bounds, BOResult, _norm_pdf, _norm_cdf,
    _optimize_acquisition, _simple_local_opt
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MFAcquisitionType(Enum):
    """Multi-fidelity acquisition function types."""
    COST_AWARE_EI = "cost_aware_ei"
    MF_KNOWLEDGE_GRADIENT = "mf_knowledge_gradient"
    COST_AWARE_UCB = "cost_aware_ucb"
    ENTROPY_SEARCH = "entropy_search"
    MAX_VALUE_ENTROPY = "max_value_entropy"


@dataclass
class FidelityLevel:
    """Description of a fidelity level."""
    level: int          # 0 = lowest fidelity, M-1 = highest
    cost: float         # Evaluation cost (relative)
    name: str = ""      # Optional descriptive name

    def __post_init__(self):
        if not self.name:
            self.name = f"fidelity_{self.level}"


@dataclass
class MFObservation:
    """A single multi-fidelity observation."""
    x: np.ndarray       # Input point (d,)
    fidelity: int       # Fidelity level
    y: float            # Observed value
    cost: float         # Cost of this evaluation


@dataclass
class MFBOResult:
    """Result of multi-fidelity Bayesian optimization."""
    x_best: np.ndarray              # Best input (at highest fidelity)
    f_best: float                   # Best function value at highest fidelity
    X_history: np.ndarray           # All evaluated inputs (n, d)
    y_history: np.ndarray           # All observed values (n,)
    fidelity_history: np.ndarray    # Fidelity level for each observation (n,)
    cost_history: np.ndarray        # Cost of each evaluation (n,)
    total_cost: float               # Total cost spent
    n_evaluations: Dict[int, int]   # Number of evaluations per fidelity
    convergence: List[float]        # Best HF value at each iteration
    cost_convergence: List[float]   # Total cost at each iteration


@dataclass
class ContinuousFidelityResult:
    """Result of continuous-fidelity BO."""
    x_best: np.ndarray
    f_best: float
    X_history: np.ndarray           # (n, d) input points
    s_history: np.ndarray           # (n,) fidelity values in [0, 1]
    y_history: np.ndarray           # (n,) observations
    cost_history: np.ndarray        # (n,) costs
    total_cost: float
    convergence: List[float]


@dataclass
class MFComparison:
    """Comparison of MF-BO vs standard BO."""
    mf_result: MFBOResult
    sf_result: BOResult
    cost_ratio: float               # total_cost_SF / total_cost_MF
    speedup: float                  # iterations_to_match / mf_iterations
    mf_evaluations: Dict[int, int]  # Fidelity breakdown


# ---------------------------------------------------------------------------
# Multi-Fidelity Gaussian Process
# ---------------------------------------------------------------------------

class MultiFidelityKernel(Kernel):
    """Kernel that models correlation across fidelity levels.

    Uses an augmented input space: [x, fidelity_encoded].
    The fidelity dimension uses a separate RBF kernel with its own
    length scale, capturing inter-fidelity correlation.

    K([x,s], [x',s']) = k_x(x, x') * k_s(s, s')

    where s is the normalized fidelity (0 to 1).
    """

    def __init__(self, base_kernel: Kernel, fidelity_length_scale: float = 0.5,
                 fidelity_variance: float = 1.0):
        self.base_kernel = base_kernel
        self._fidelity_ls = fidelity_length_scale
        self._fidelity_var = fidelity_variance
        # Log-space params: [base_params..., log(fid_ls), log(fid_var)]

    def __call__(self, X1, X2):
        # Last column is fidelity
        x1, s1 = X1[:, :-1], X1[:, -1:]
        x2, s2 = X2[:, :-1], X2[:, -1:]

        K_x = self.base_kernel(x1, x2)

        # RBF over fidelity dimension
        dist_sq = (s1 - s2.T) ** 2
        K_s = self._fidelity_var * np.exp(-0.5 * dist_sq / (self._fidelity_ls ** 2))

        return K_x * K_s

    def diag(self, X):
        x, s = X[:, :-1], X[:, -1:]
        d_x = self.base_kernel.diag(x)
        d_s = np.full(len(X), self._fidelity_var)
        return d_x * d_s

    def params(self):
        base_p = self.base_kernel.params()
        return np.concatenate([base_p,
                               [np.log(self._fidelity_ls), np.log(self._fidelity_var)]])

    def set_params(self, p):
        n_base = self.base_kernel.n_params()
        self.base_kernel.set_params(p[:n_base])
        self._fidelity_ls = np.exp(p[n_base])
        self._fidelity_var = np.exp(p[n_base + 1])

    def n_params(self):
        return self.base_kernel.n_params() + 2


class MultiFidelityGP:
    """Gaussian Process that models multiple fidelity levels.

    Augments the input space with a fidelity dimension, allowing the GP
    to learn correlations between fidelity levels. Low-fidelity data
    informs the high-fidelity model.
    """

    def __init__(self, n_fidelities: int, base_kernel: Kernel = None,
                 noise_variance: float = 1e-4):
        self.n_fidelities = n_fidelities
        base = base_kernel or ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
        self.kernel = MultiFidelityKernel(base)
        self.noise_variance = noise_variance
        self.gp = GaussianProcess(self.kernel, noise_variance=noise_variance)
        self._fitted = False

    def _encode_fidelity(self, X: np.ndarray, fidelity: np.ndarray) -> np.ndarray:
        """Augment X with normalized fidelity column."""
        s = fidelity.astype(float) / max(self.n_fidelities - 1, 1)
        return np.column_stack([X, s])

    def fit(self, X: np.ndarray, fidelities: np.ndarray, y: np.ndarray):
        """Fit the multi-fidelity GP.

        Args:
            X: Input points (n, d)
            fidelities: Fidelity level per observation (n,), integers 0..M-1
            y: Observations (n,)
        """
        X_aug = self._encode_fidelity(X, fidelities)
        self.gp.fit(X_aug, y)
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, fidelity: int) -> GPPrediction:
        """Predict at a specific fidelity level.

        Args:
            X_test: Test points (n, d)
            fidelity: Which fidelity level to predict at

        Returns:
            GPPrediction with mean, variance, std
        """
        fids = np.full(len(X_test), fidelity)
        X_aug = self._encode_fidelity(X_test, fids)
        return self.gp.predict(X_aug)

    def predict_highest(self, X_test: np.ndarray) -> GPPrediction:
        """Predict at the highest fidelity level."""
        return self.predict(X_test, self.n_fidelities - 1)


# ---------------------------------------------------------------------------
# Linear Multi-Fidelity Model (AR1 / Kennedy-O'Hagan)
# ---------------------------------------------------------------------------

class LinearMultiFidelityGP:
    """Auto-regressive multi-fidelity GP (AR1 model).

    f_t(x) = rho_t * f_{t-1}(x) + delta_t(x)

    Each fidelity is a scaled version of the previous plus a discrepancy GP.
    This is the Kennedy & O'Hagan (2000) model.
    """

    def __init__(self, n_fidelities: int, kernel: Kernel = None,
                 noise_variance: float = 1e-4):
        self.n_fidelities = n_fidelities
        self.noise_variance = noise_variance
        self._kernel_template = kernel or ScaleKernel(Matern52Kernel(), scale=1.0)

        # One GP per fidelity level for the discrepancy
        self.delta_gps: List[GaussianProcess] = []
        self.rho: List[float] = []  # Scaling factors
        self._X_per_fid: List[np.ndarray] = []
        self._y_per_fid: List[np.ndarray] = []
        self._fitted = False

    def fit(self, X: np.ndarray, fidelities: np.ndarray, y: np.ndarray):
        """Fit the AR1 multi-fidelity model.

        Fits level 0 directly, then fits rho and delta for each subsequent level.
        """
        self._X_per_fid = []
        self._y_per_fid = []
        self.delta_gps = []
        self.rho = []

        for t in range(self.n_fidelities):
            mask = fidelities == t
            self._X_per_fid.append(X[mask])
            self._y_per_fid.append(y[mask])

        # Level 0: fit directly
        k0 = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
        gp0 = GaussianProcess(k0, noise_variance=self.noise_variance)
        if len(self._X_per_fid[0]) > 0:
            gp0.fit(self._X_per_fid[0], self._y_per_fid[0])
        self.delta_gps.append(gp0)
        self.rho.append(1.0)  # rho_0 is unused (identity)

        # Levels 1..M-1: fit rho and delta
        for t in range(1, self.n_fidelities):
            X_t = self._X_per_fid[t]
            y_t = self._y_per_fid[t]

            if len(X_t) == 0:
                kt = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
                gp_t = GaussianProcess(kt, noise_variance=self.noise_variance)
                self.delta_gps.append(gp_t)
                self.rho.append(1.0)
                continue

            # Get predictions from level t-1 at X_t
            pred_prev = self._predict_level(X_t, t - 1)
            mu_prev = pred_prev.mean

            # Estimate rho via least squares: y_t = rho * mu_prev + delta
            if np.std(mu_prev) > 1e-10:
                rho_t = np.dot(y_t - np.mean(y_t), mu_prev - np.mean(mu_prev)) / (
                    np.dot(mu_prev - np.mean(mu_prev), mu_prev - np.mean(mu_prev)) + 1e-10)
            else:
                rho_t = 1.0

            # Fit discrepancy GP: delta_t = y_t - rho_t * mu_prev
            residuals = y_t - rho_t * mu_prev
            kt = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
            gp_t = GaussianProcess(kt, noise_variance=self.noise_variance)
            gp_t.fit(X_t, residuals)

            self.delta_gps.append(gp_t)
            self.rho.append(rho_t)

        self._fitted = True
        return self

    def _predict_level(self, X_test: np.ndarray, level: int) -> GPPrediction:
        """Recursive prediction at a given level."""
        if level == 0:
            if self.delta_gps[0]._X is not None:
                return self.delta_gps[0].predict(X_test)
            else:
                return GPPrediction(
                    mean=np.zeros(len(X_test)),
                    variance=np.ones(len(X_test)),
                    std=np.ones(len(X_test))
                )

        # f_t(x) = rho_t * f_{t-1}(x) + delta_t(x)
        pred_prev = self._predict_level(X_test, level - 1)
        rho_t = self.rho[level]

        if self.delta_gps[level]._X is not None:
            pred_delta = self.delta_gps[level].predict(X_test)
        else:
            pred_delta = GPPrediction(
                mean=np.zeros(len(X_test)),
                variance=np.ones(len(X_test)) * 0.1,
                std=np.ones(len(X_test)) * np.sqrt(0.1)
            )

        mean = rho_t * pred_prev.mean + pred_delta.mean
        variance = rho_t**2 * pred_prev.variance + pred_delta.variance
        std = np.sqrt(np.maximum(variance, 1e-10))

        return GPPrediction(mean=mean, variance=variance, std=std)

    def predict(self, X_test: np.ndarray, fidelity: int) -> GPPrediction:
        """Predict at a specific fidelity level."""
        return self._predict_level(X_test, fidelity)

    def predict_highest(self, X_test: np.ndarray) -> GPPrediction:
        """Predict at the highest fidelity level."""
        return self._predict_level(X_test, self.n_fidelities - 1)


# ---------------------------------------------------------------------------
# Acquisition Functions for Multi-Fidelity
# ---------------------------------------------------------------------------

def cost_aware_ei(mu: np.ndarray, std: np.ndarray, f_best: float,
                  cost: float, xi: float = 0.01) -> np.ndarray:
    """Cost-Aware Expected Improvement: EI(x) / cost(fidelity).

    Balances information gain against evaluation cost.
    """
    ei = expected_improvement(mu, std, f_best, xi=xi)
    return ei / max(cost, 1e-10)


def cost_aware_ucb(mu: np.ndarray, std: np.ndarray, beta: float = 2.0,
                   cost: float = 1.0) -> np.ndarray:
    """Cost-Aware Upper Confidence Bound (for minimization).

    UCB(x) = -(mu - beta * std) / cost
    """
    ucb = -(mu - beta * std)
    return ucb / max(cost, 1e-10)


def multi_fidelity_knowledge_gradient(
    mf_gp, X_candidates: np.ndarray, fidelity: int,
    f_best_hf: float, cost: float, n_fantasies: int = 20,
    rng: Generator = None
) -> np.ndarray:
    """Multi-Fidelity Knowledge Gradient.

    Estimates the value of evaluating each candidate at a given fidelity,
    normalized by cost. The value is the expected improvement in the
    high-fidelity optimum after observing the new point.

    KG(x, s) = E[min f_HF(X*) after observing (x,s)] - f_best_hf
    """
    rng = rng or default_rng(42)
    n_fidelities = mf_gp.n_fidelities
    hf = n_fidelities - 1

    # Get current HF predictions at candidates
    pred_hf = mf_gp.predict_highest(X_candidates)

    # Get predictions at the proposed fidelity
    pred_s = mf_gp.predict(X_candidates, fidelity)

    kg_values = np.zeros(len(X_candidates))

    for i in range(len(X_candidates)):
        # Fantasy: sample possible observations at (x_i, fidelity)
        fantasies = rng.normal(pred_s.mean[i], max(pred_s.std[i], 1e-6), n_fantasies)

        # For each fantasy, estimate the change in HF best
        improvements = []
        for y_fantasy in fantasies:
            # How much does observing y at this fidelity help the HF prediction?
            # Approximate: correlation between fidelity s and HF gives partial info
            corr = 1.0 - abs(fidelity - hf) / max(n_fidelities - 1, 1)
            # Expected HF value given fantasy
            mu_hf_updated = pred_hf.mean[i] + corr * (y_fantasy - pred_s.mean[i]) * (
                pred_hf.std[i] / max(pred_s.std[i], 1e-6))
            improvement = max(f_best_hf - mu_hf_updated, 0.0)
            improvements.append(improvement)

        kg_values[i] = np.mean(improvements)

    return kg_values / max(cost, 1e-10)


def max_value_entropy_search(
    mf_gp, X_candidates: np.ndarray, fidelity: int,
    f_best_hf: float, cost: float, n_samples: int = 50,
    rng: Generator = None
) -> np.ndarray:
    """Multi-Fidelity Max-Value Entropy Search (MF-MES).

    Information-theoretic acquisition: how much does observing (x, s)
    reduce entropy of f* (the minimum value at highest fidelity)?

    Approximated via Gumbel sampling of f*.
    """
    rng = rng or default_rng(42)
    hf = mf_gp.n_fidelities - 1

    pred_hf = mf_gp.predict_highest(X_candidates)
    pred_s = mf_gp.predict(X_candidates, fidelity)

    # Sample f* values using Gumbel distribution centered on f_best
    # f* ~ Gumbel(f_best - std_mean, scale)
    scale = max(np.mean(pred_hf.std), 1e-6)
    f_star_samples = f_best_hf - scale * np.log(-np.log(
        rng.uniform(0.01, 0.99, n_samples)))

    mes_values = np.zeros(len(X_candidates))

    for i in range(len(X_candidates)):
        mu_s = pred_s.mean[i]
        std_s = max(pred_s.std[i], 1e-8)

        # Correlation factor between fidelity s and HF
        corr = 1.0 - abs(fidelity - hf) / max(mf_gp.n_fidelities - 1, 1)

        # For each f* sample, compute the info gain
        info_gains = []
        for f_s in f_star_samples:
            # Truncated Gaussian: P(y | f* = f_s)
            gamma = (f_s - mu_s * corr) / (std_s * corr + 1e-10)
            # Info gain approximation via gamma
            pdf_val = _norm_pdf(gamma)
            cdf_val = _norm_cdf(gamma)
            if cdf_val > 1e-10:
                ig = gamma * pdf_val / (2 * cdf_val) - np.log(cdf_val)
            else:
                ig = 0.0
            info_gains.append(max(ig, 0.0))

        mes_values[i] = np.mean(info_gains)

    return mes_values / max(cost, 1e-10)


# ---------------------------------------------------------------------------
# Multi-Fidelity Bayesian Optimization
# ---------------------------------------------------------------------------

def multi_fidelity_bo(
    objectives: Dict[int, Callable],
    bounds: Bounds,
    fidelity_costs: Dict[int, float],
    budget: float = 100.0,
    n_initial_per_fidelity: int = 3,
    acquisition: MFAcquisitionType = MFAcquisitionType.COST_AWARE_EI,
    model: str = "augmented",
    kernel: Kernel = None,
    noise_variance: float = 1e-4,
    rng: Generator = None,
    verbose: bool = False
) -> MFBOResult:
    """Multi-Fidelity Bayesian Optimization.

    Args:
        objectives: Dict mapping fidelity level -> callable.
                    Each callable: (x,) -> float. Level 0 = cheapest.
        bounds: Search space bounds for x.
        fidelity_costs: Dict mapping fidelity level -> cost.
        budget: Total evaluation budget (in cost units).
        n_initial_per_fidelity: Initial random samples per fidelity.
        acquisition: Which acquisition function to use.
        model: "augmented" (single GP with fidelity dim) or "ar1" (linear MF).
        kernel: Base kernel for GP (default: ScaleKernel(Matern52)).
        noise_variance: GP observation noise.
        rng: Random number generator.
        verbose: Print progress.

    Returns:
        MFBOResult with optimization history and best point.
    """
    rng = rng or default_rng(42)
    n_fidelities = len(objectives)
    fidelity_levels = sorted(objectives.keys())
    highest_fid = fidelity_levels[-1]

    # Collect initial data
    X_all = []
    y_all = []
    fid_all = []
    cost_all = []
    total_cost = 0.0

    for fid in fidelity_levels:
        cost_fid = fidelity_costs[fid]
        n_init = n_initial_per_fidelity
        if total_cost + n_init * cost_fid > budget:
            n_init = max(1, int((budget - total_cost) / cost_fid))

        X_init = bounds.sample_uniform(n_init, rng=rng)
        for x in X_init:
            y_val = objectives[fid](x)
            X_all.append(x)
            y_all.append(y_val)
            fid_all.append(fid)
            cost_all.append(cost_fid)
            total_cost += cost_fid

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    fid_all = np.array(fid_all, dtype=int)
    cost_all = np.array(cost_all)

    # Track best HF value
    hf_mask = fid_all == highest_fid
    if np.any(hf_mask):
        f_best_hf = np.min(y_all[hf_mask])
        x_best = X_all[hf_mask][np.argmin(y_all[hf_mask])]
    else:
        f_best_hf = np.inf
        x_best = X_all[0]

    convergence = [f_best_hf]
    cost_convergence = [total_cost]

    # Build multi-fidelity GP
    def build_model():
        if model == "ar1":
            mf = LinearMultiFidelityGP(n_fidelities, kernel, noise_variance)
        else:
            mf = MultiFidelityGP(n_fidelities, kernel, noise_variance)
        mf.fit(X_all, fid_all, y_all)
        return mf

    # Optimization loop
    iteration = 0
    while total_cost < budget:
        iteration += 1
        mf_gp = build_model()

        # Generate candidates
        n_candidates = min(500, max(100, 50 * bounds.dim))
        X_cand = bounds.sample_uniform(n_candidates, rng=rng)

        # Evaluate acquisition for each (candidate, fidelity) pair
        best_acq = -np.inf
        best_x = None
        best_fid = highest_fid

        # Periodically force HF evaluation (every ~5 iterations or when stuck)
        force_hf = (iteration % 5 == 0) and (total_cost + fidelity_costs[highest_fid] <= budget)

        for fid in fidelity_levels:
            cost_fid = fidelity_costs[fid]
            if total_cost + cost_fid > budget:
                continue
            if force_hf and fid != highest_fid:
                continue

            pred_hf = mf_gp.predict_highest(X_cand)

            if acquisition == MFAcquisitionType.COST_AWARE_EI:
                # Use HF prediction for EI, scaled by cost at this fidelity
                # For HF evaluations, add bonus for direct observation
                acq_vals = cost_aware_ei(
                    pred_hf.mean, pred_hf.std, f_best_hf, cost_fid)
                if fid == highest_fid:
                    # Boost HF: direct observation reduces uncertainty more
                    acq_vals *= 1.5
            elif acquisition == MFAcquisitionType.COST_AWARE_UCB:
                acq_vals = cost_aware_ucb(
                    pred_hf.mean, pred_hf.std, beta=2.0, cost=cost_fid)
                if fid == highest_fid:
                    acq_vals *= 1.5
            elif acquisition == MFAcquisitionType.MF_KNOWLEDGE_GRADIENT:
                acq_vals = multi_fidelity_knowledge_gradient(
                    mf_gp, X_cand, fid, f_best_hf, cost_fid, rng=rng)
            elif acquisition == MFAcquisitionType.ENTROPY_SEARCH:
                acq_vals = max_value_entropy_search(
                    mf_gp, X_cand, fid, f_best_hf, cost_fid, rng=rng)
            elif acquisition == MFAcquisitionType.MAX_VALUE_ENTROPY:
                acq_vals = max_value_entropy_search(
                    mf_gp, X_cand, fid, f_best_hf, cost_fid, rng=rng)
            else:
                acq_vals = cost_aware_ei(
                    pred_hf.mean, pred_hf.std, f_best_hf, cost_fid)

            idx_best = np.argmax(acq_vals)
            if acq_vals[idx_best] > best_acq:
                best_acq = acq_vals[idx_best]
                best_x = X_cand[idx_best]
                best_fid = fid

        if best_x is None:
            break

        # Evaluate
        y_new = objectives[best_fid](best_x)
        cost_new = fidelity_costs[best_fid]

        X_all = np.vstack([X_all, best_x.reshape(1, -1)])
        y_all = np.append(y_all, y_new)
        fid_all = np.append(fid_all, best_fid)
        cost_all = np.append(cost_all, cost_new)
        total_cost += cost_new

        # Update best HF
        if best_fid == highest_fid and y_new < f_best_hf:
            f_best_hf = y_new
            x_best = best_x.copy()

        convergence.append(f_best_hf)
        cost_convergence.append(total_cost)

        if verbose:
            print(f"  Iter {iteration}: fid={best_fid}, cost={cost_new:.1f}, "
                  f"y={y_new:.4f}, best_hf={f_best_hf:.4f}, total_cost={total_cost:.1f}")

    # Count evaluations per fidelity
    n_evals = {}
    for fid in fidelity_levels:
        n_evals[fid] = int(np.sum(fid_all == fid))

    return MFBOResult(
        x_best=x_best,
        f_best=f_best_hf,
        X_history=X_all,
        y_history=y_all,
        fidelity_history=fid_all,
        cost_history=cost_all,
        total_cost=total_cost,
        n_evaluations=n_evals,
        convergence=convergence,
        cost_convergence=cost_convergence
    )


# ---------------------------------------------------------------------------
# Continuous-Fidelity BO
# ---------------------------------------------------------------------------

def continuous_fidelity_bo(
    objective: Callable,
    bounds: Bounds,
    cost_function: Callable = None,
    budget: float = 100.0,
    n_initial: int = 10,
    kernel: Kernel = None,
    noise_variance: float = 1e-4,
    rng: Generator = None,
    verbose: bool = False
) -> ContinuousFidelityResult:
    """Continuous-Fidelity Bayesian Optimization.

    Fidelity is a continuous parameter s in [0, 1] where s=1 is highest fidelity.
    The optimizer jointly selects both x and s to maximize information per cost.

    Args:
        objective: f(x, s) -> float. x is the input, s in [0,1] is fidelity.
        bounds: Search space bounds for x (not including fidelity).
        cost_function: cost(s) -> float. Default: 0.1 + 0.9 * s^2.
        budget: Total cost budget.
        n_initial: Number of initial samples.
        kernel: Base kernel for GP.
        noise_variance: GP noise.
        rng: Random generator.
        verbose: Print progress.

    Returns:
        ContinuousFidelityResult.
    """
    rng = rng or default_rng(42)

    if cost_function is None:
        cost_function = lambda s: 0.1 + 0.9 * s ** 2

    # Build augmented GP over [x, s]
    d = bounds.dim
    base_k = kernel or ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
    mf_kernel = MultiFidelityKernel(base_k, fidelity_length_scale=0.5)
    gp = GaussianProcess(mf_kernel, noise_variance=noise_variance)

    # Initial samples across x and s
    X_all = []
    s_all = []
    y_all = []
    cost_all = []
    total_cost = 0.0

    for _ in range(n_initial):
        x = bounds.sample_uniform(1, rng=rng)[0]
        s = rng.uniform(0.0, 1.0)
        cost_s = cost_function(s)
        if total_cost + cost_s > budget:
            continue
        y = objective(x, s)
        X_all.append(x)
        s_all.append(s)
        y_all.append(y)
        cost_all.append(cost_s)
        total_cost += cost_s

    X_all = np.array(X_all)
    s_all = np.array(s_all)
    y_all = np.array(y_all)
    cost_all = np.array(cost_all)

    # Best at s=1
    s1_mask = s_all > 0.8
    if np.any(s1_mask):
        f_best = np.min(y_all[s1_mask])
        x_best = X_all[s1_mask][np.argmin(y_all[s1_mask])]
    else:
        f_best = np.min(y_all) if len(y_all) > 0 else np.inf
        x_best = X_all[np.argmin(y_all)] if len(y_all) > 0 else bounds.sample_uniform(1, rng=rng)[0]

    convergence = [f_best]

    # Optimization loop
    iteration = 0
    while total_cost < budget:
        iteration += 1

        # Fit GP on augmented data
        X_aug = np.column_stack([X_all, s_all])
        gp.fit(X_aug, y_all)

        # Generate candidates over [x, s]
        n_cand = min(500, max(100, 50 * d))
        X_cand = bounds.sample_uniform(n_cand, rng=rng)
        s_cand = rng.uniform(0.0, 1.0, n_cand)

        # For each candidate, compute cost-aware EI w.r.t. HF prediction
        # Predict at s=1 for all candidates
        X_hf = np.column_stack([X_cand, np.ones(n_cand)])
        pred_hf = gp.predict(X_hf)

        # Compute cost-weighted acquisition
        best_acq = -np.inf
        best_x = None
        best_s = 1.0

        for i in range(n_cand):
            cost_s = cost_function(s_cand[i])
            if total_cost + cost_s > budget:
                continue

            ei_val = expected_improvement(
                pred_hf.mean[i:i+1], pred_hf.std[i:i+1], f_best)
            acq = ei_val[0] / max(cost_s, 1e-10)

            # Bonus for higher fidelity when uncertainty is low
            acq *= (0.5 + 0.5 * s_cand[i])

            if acq > best_acq:
                best_acq = acq
                best_x = X_cand[i]
                best_s = s_cand[i]

        if best_x is None:
            break

        # Evaluate
        y_new = objective(best_x, best_s)
        cost_new = cost_function(best_s)

        X_all = np.vstack([X_all, best_x.reshape(1, -1)])
        s_all = np.append(s_all, best_s)
        y_all = np.append(y_all, y_new)
        cost_all = np.append(cost_all, cost_new)
        total_cost += cost_new

        # Update best at high fidelity
        if best_s > 0.8 and y_new < f_best:
            f_best = y_new
            x_best = best_x.copy()

        convergence.append(f_best)

        if verbose:
            print(f"  Iter {iteration}: s={best_s:.2f}, cost={cost_new:.2f}, "
                  f"y={y_new:.4f}, best={f_best:.4f}, total_cost={total_cost:.1f}")

    return ContinuousFidelityResult(
        x_best=x_best,
        f_best=f_best,
        X_history=X_all,
        s_history=s_all,
        y_history=y_all,
        cost_history=cost_all,
        total_cost=total_cost,
        convergence=convergence
    )


# ---------------------------------------------------------------------------
# Multi-Task BO (independent GPs with shared exploration)
# ---------------------------------------------------------------------------

def multi_task_bo(
    objectives: Dict[int, Callable],
    bounds: Bounds,
    fidelity_costs: Dict[int, float],
    budget: float = 100.0,
    n_initial_per_fidelity: int = 3,
    noise_variance: float = 1e-4,
    rng: Generator = None,
    verbose: bool = False
) -> MFBOResult:
    """Multi-Task Bayesian Optimization.

    Maintains independent GPs per fidelity level. Low-fidelity GPs
    suggest promising regions, and high-fidelity GP refines.

    Selection strategy: alternate between exploring with cheap LF
    evaluations and exploiting with expensive HF evaluations based
    on cost-weighted expected improvement.
    """
    rng = rng or default_rng(42)
    fidelity_levels = sorted(objectives.keys())
    highest_fid = fidelity_levels[-1]
    n_fidelities = len(fidelity_levels)

    # Per-fidelity data
    data = {fid: {"X": [], "y": []} for fid in fidelity_levels}

    X_all = []
    y_all = []
    fid_all = []
    cost_all = []
    total_cost = 0.0

    # Initial sampling
    for fid in fidelity_levels:
        cost_fid = fidelity_costs[fid]
        for _ in range(n_initial_per_fidelity):
            if total_cost + cost_fid > budget:
                break
            x = bounds.sample_uniform(1, rng=rng)[0]
            y = objectives[fid](x)
            data[fid]["X"].append(x)
            data[fid]["y"].append(y)
            X_all.append(x)
            y_all.append(y)
            fid_all.append(fid)
            cost_all.append(cost_fid)
            total_cost += cost_fid

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    fid_all = np.array(fid_all, dtype=int)
    cost_all = np.array(cost_all)

    # Best HF value
    hf_mask = fid_all == highest_fid
    if np.any(hf_mask):
        f_best_hf = np.min(y_all[hf_mask])
        x_best = X_all[hf_mask][np.argmin(y_all[hf_mask])]
    else:
        f_best_hf = np.inf
        x_best = X_all[0]

    convergence = [f_best_hf]
    cost_convergence = [total_cost]

    # Optimization loop
    iteration = 0
    while total_cost < budget:
        iteration += 1

        # Fit independent GPs
        gps = {}
        for fid in fidelity_levels:
            Xf = np.array(data[fid]["X"]) if data[fid]["X"] else np.zeros((0, bounds.dim))
            yf = np.array(data[fid]["y"]) if data[fid]["y"] else np.array([])
            if len(Xf) >= 2:
                k = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
                gp = GaussianProcess(k, noise_variance=noise_variance)
                gp.fit(Xf, yf)
                gps[fid] = gp

        # Generate candidates
        n_cand = min(500, max(100, 50 * bounds.dim))
        X_cand = bounds.sample_uniform(n_cand, rng=rng)

        # Evaluate acquisition for each fidelity
        best_acq = -np.inf
        best_x = None
        best_fid = highest_fid

        for fid in fidelity_levels:
            cost_fid = fidelity_costs[fid]
            if total_cost + cost_fid > budget:
                continue
            if fid not in gps:
                continue

            pred = gps[fid].predict(X_cand)

            # Use this fidelity's prediction as proxy for HF
            # Scale by empirical correlation if HF GP exists
            f_ref = f_best_hf if f_best_hf < np.inf else np.min(pred.mean)
            acq_vals = cost_aware_ei(pred.mean, pred.std, f_ref, cost_fid)

            idx = np.argmax(acq_vals)
            if acq_vals[idx] > best_acq:
                best_acq = acq_vals[idx]
                best_x = X_cand[idx]
                best_fid = fid

        if best_x is None:
            break

        # Evaluate
        y_new = objectives[best_fid](best_x)
        cost_new = fidelity_costs[best_fid]

        data[best_fid]["X"].append(best_x)
        data[best_fid]["y"].append(y_new)
        X_all = np.vstack([X_all, best_x.reshape(1, -1)])
        y_all = np.append(y_all, y_new)
        fid_all = np.append(fid_all, best_fid)
        cost_all = np.append(cost_all, cost_new)
        total_cost += cost_new

        if best_fid == highest_fid and y_new < f_best_hf:
            f_best_hf = y_new
            x_best = best_x.copy()

        convergence.append(f_best_hf)
        cost_convergence.append(total_cost)

        if verbose:
            print(f"  Iter {iteration}: fid={best_fid}, y={y_new:.4f}, "
                  f"best_hf={f_best_hf:.4f}, total_cost={total_cost:.1f}")

    n_evals = {}
    for fid in fidelity_levels:
        n_evals[fid] = int(np.sum(fid_all == fid))

    return MFBOResult(
        x_best=x_best,
        f_best=f_best_hf,
        X_history=X_all,
        y_history=y_all,
        fidelity_history=fid_all,
        cost_history=cost_all,
        total_cost=total_cost,
        n_evaluations=n_evals,
        convergence=convergence,
        cost_convergence=cost_convergence
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_mf_vs_single(
    hf_objective: Callable,
    lf_objectives: Dict[int, Callable],
    bounds: Bounds,
    fidelity_costs: Dict[int, float],
    budget: float = 100.0,
    n_initial: int = 3,
    rng_seed: int = 42
) -> MFComparison:
    """Compare multi-fidelity BO against standard single-fidelity BO.

    Runs both approaches with the same budget and compares results.

    Args:
        hf_objective: High-fidelity objective function.
        lf_objectives: Dict of lower fidelity objectives (level -> callable).
        bounds: Search space.
        fidelity_costs: Cost per fidelity level.
        budget: Total budget for both approaches.
        n_initial: Initial samples per fidelity (MF) or total (SF).
        rng_seed: Random seed for reproducibility.
    """
    # Multi-fidelity
    highest_fid = max(fidelity_costs.keys())
    all_objectives = dict(lf_objectives)
    all_objectives[highest_fid] = hf_objective

    rng_mf = default_rng(rng_seed)
    mf_result = multi_fidelity_bo(
        objectives=all_objectives,
        bounds=bounds,
        fidelity_costs=fidelity_costs,
        budget=budget,
        n_initial_per_fidelity=n_initial,
        rng=rng_mf
    )

    # Single-fidelity (only HF evaluations)
    hf_cost = fidelity_costs[highest_fid]
    n_sf_iters = int(budget / hf_cost) - n_initial

    rng_sf = default_rng(rng_seed)

    # Manual single-fidelity BO
    X_sf = bounds.sample_uniform(n_initial, rng=rng_sf)
    y_sf = np.array([hf_objective(x) for x in X_sf])
    f_best_sf = np.min(y_sf)
    x_best_sf = X_sf[np.argmin(y_sf)]
    convergence_sf = [f_best_sf]

    k = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)

    for _ in range(max(n_sf_iters, 0)):
        gp = GaussianProcess(k, noise_variance=1e-4)
        gp.fit(X_sf, y_sf)

        X_cand = bounds.sample_uniform(200, rng=rng_sf)
        pred = gp.predict(X_cand)
        ei = expected_improvement(pred.mean, pred.std, f_best_sf)
        idx = np.argmax(ei)
        x_new = X_cand[idx]
        y_new = hf_objective(x_new)

        X_sf = np.vstack([X_sf, x_new.reshape(1, -1)])
        y_sf = np.append(y_sf, y_new)
        if y_new < f_best_sf:
            f_best_sf = y_new
            x_best_sf = x_new.copy()
        convergence_sf.append(f_best_sf)

    sf_result = BOResult(
        x_best=x_best_sf,
        f_best=f_best_sf,
        X_history=X_sf,
        y_history=y_sf,
        n_iterations=len(y_sf) - n_initial,
        convergence=convergence_sf,
        acquisition_values=[]
    )

    sf_total_cost = len(y_sf) * hf_cost
    cost_ratio = sf_total_cost / max(mf_result.total_cost, 1e-10)

    # Speedup: how many SF iterations to match MF result
    sf_match = len(convergence_sf)
    for i, c in enumerate(convergence_sf):
        if c <= mf_result.f_best * 1.05:
            sf_match = i
            break

    mf_iters = len(mf_result.convergence)
    speedup = sf_match / max(mf_iters, 1)

    return MFComparison(
        mf_result=mf_result,
        sf_result=sf_result,
        cost_ratio=cost_ratio,
        speedup=speedup,
        mf_evaluations=mf_result.n_evaluations
    )


# ---------------------------------------------------------------------------
# Benchmark functions for multi-fidelity testing
# ---------------------------------------------------------------------------

def branin_hf(x: np.ndarray) -> float:
    """High-fidelity Branin function."""
    x1, x2 = x[0], x[1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def branin_lf(x: np.ndarray) -> float:
    """Low-fidelity Branin: biased + noisy approximation."""
    return branin_hf(x) * 0.8 + 2.0 * (x[0] + 5) / 15.0 + 1.0


def branin_mf(x: np.ndarray) -> float:
    """Medium-fidelity Branin: slight bias."""
    return branin_hf(x) * 0.95 + 0.5 * np.sin(x[0])


def sphere_hf(x: np.ndarray) -> float:
    """High-fidelity Sphere function."""
    return float(np.sum(x**2))


def sphere_lf(x: np.ndarray) -> float:
    """Low-fidelity Sphere: offset + scaled."""
    return float(np.sum(x**2)) * 0.7 + 0.5


def hartmann3_hf(x: np.ndarray) -> float:
    """High-fidelity 3D Hartmann function. Min ~ -3.86 at ~(0.114, 0.556, 0.853)."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])
    P = np.array([
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0382, 0.5743, 0.8828]
    ])
    result = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x[:3] - P[i])**2)
        result -= alpha[i] * np.exp(-inner)
    return result


def hartmann3_lf(x: np.ndarray) -> float:
    """Low-fidelity Hartmann: perturbed coefficients."""
    return hartmann3_hf(x) * 0.75 + 0.3 * np.sin(2 * np.pi * x[0])


def continuous_fidelity_branin(x: np.ndarray, s: float) -> float:
    """Branin with continuous fidelity s in [0, 1].

    s=1 gives true Branin. Lower s adds bias proportional to (1-s).
    """
    true_val = branin_hf(x)
    bias = (1 - s) * (2.0 * (x[0] + 5) / 15.0 + 1.0 + 0.5 * np.sin(x[1]))
    noise_std = 0.1 * (1 - s)  # Lower fidelity = more noise
    return true_val + bias


# ---------------------------------------------------------------------------
# Summary utility
# ---------------------------------------------------------------------------

def mf_optimization_summary(result: MFBOResult, name: str = "MF-BO") -> str:
    """Generate a human-readable summary of MF-BO results."""
    lines = [
        f"=== {name} Summary ===",
        f"Best value: {result.f_best:.6f}",
        f"Best input: {result.x_best}",
        f"Total cost: {result.total_cost:.1f}",
        f"Total evaluations: {len(result.y_history)}",
        "Evaluations per fidelity:"
    ]
    for fid in sorted(result.n_evaluations.keys()):
        lines.append(f"  Fidelity {fid}: {result.n_evaluations[fid]}")

    if result.convergence:
        lines.append(f"Final convergence: {result.convergence[-1]:.6f}")

    return "\n".join(lines)
