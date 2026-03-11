"""
C167: Bayesian Optimization
Composing C155 (Gaussian Processes) + C166 (Bayesian Neural Network)

Black-box optimization using surrogate models and acquisition functions.

Components:
1. AcquisitionFunction -- base class for acquisition functions
2. ExpectedImprovement -- EI acquisition
3. ProbabilityOfImprovement -- PI acquisition
4. UpperConfidenceBound -- UCB acquisition
5. ThompsonSampling -- posterior sampling
6. KnowledgeGradient -- one-step lookahead value of information
7. BayesianOptimizer -- main optimizer with GP surrogate
8. BNNBayesianOptimizer -- optimizer using BNN surrogate
9. MultiObjectiveBO -- multi-objective optimization (Pareto front)
10. BatchBayesianOptimizer -- batch acquisition (q-EI via kriging believer)
11. ConstrainedBO -- optimization with black-box constraints
12. BayesOptHistory -- tracking and analysis of optimization runs

Author: AgentZero (Session 169)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C155_gaussian_processes'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C166_bayesian_neural_network'))

from gaussian_processes import GPRegression, RBFKernel, MaternKernel, ARDKernel
from bayesian_neural_network import (
    build_bnn, BayesByBackprop, BNNPredictive, UncertaintyMetrics
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _standard_normal_cdf(x):
    """CDF of standard normal using error function approximation."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _standard_normal_pdf(x):
    """PDF of standard normal."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


def _erf(x):
    """Error function approximation (Abramowitz & Stegun)."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
            + t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-x * x)
    return sign * result


# ---------------------------------------------------------------------------
# Acquisition Functions
# ---------------------------------------------------------------------------

class AcquisitionFunction:
    """Base class for acquisition functions."""

    def __init__(self, name='base'):
        self.name = name

    def evaluate(self, mean, std, best_y, **kwargs):
        """Evaluate acquisition at points given predicted mean and std."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement (EI).

    EI(x) = (mu - best_y - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (mu - best_y - xi) / sigma
    """

    def __init__(self, xi=0.01):
        super().__init__('expected_improvement')
        self.xi = xi

    def evaluate(self, mean, std, best_y, **kwargs):
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        improvement = mean - best_y - self.xi
        # Avoid division by zero
        mask = std > 1e-12
        ei = np.zeros_like(mean)
        z = np.zeros_like(mean)
        z[mask] = improvement[mask] / std[mask]
        ei[mask] = improvement[mask] * _standard_normal_cdf(z[mask]) + std[mask] * _standard_normal_pdf(z[mask])
        # EI is always non-negative
        ei = np.maximum(ei, 0.0)
        return ei


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement (PI).

    PI(x) = Phi((mu - best_y - xi) / sigma)
    """

    def __init__(self, xi=0.01):
        super().__init__('probability_of_improvement')
        self.xi = xi

    def evaluate(self, mean, std, best_y, **kwargs):
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        mask = std > 1e-12
        pi = np.zeros_like(mean)
        z = np.zeros_like(mean)
        z[mask] = (mean[mask] - best_y - self.xi) / std[mask]
        pi[mask] = _standard_normal_cdf(z[mask])
        return pi


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound (UCB).

    UCB(x) = mu + kappa * sigma
    """

    def __init__(self, kappa=2.0):
        super().__init__('upper_confidence_bound')
        self.kappa = kappa

    def evaluate(self, mean, std, best_y, **kwargs):
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        return mean + self.kappa * std


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling -- draw from posterior and maximize."""

    def __init__(self, seed=42):
        super().__init__('thompson_sampling')
        self.rng = np.random.RandomState(seed)

    def evaluate(self, mean, std, best_y, **kwargs):
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        return mean + std * self.rng.randn(len(mean))


class KnowledgeGradient(AcquisitionFunction):
    """Knowledge Gradient -- one-step lookahead value of information.

    Approximated via the EI of the posterior mean improvement.
    KG(x) ~ E[max(mu_n+1) - max(mu_n) | x_n+1 = x]
    """

    def __init__(self, n_fantasies=10, seed=42):
        super().__init__('knowledge_gradient')
        self.n_fantasies = n_fantasies
        self.rng = np.random.RandomState(seed)

    def evaluate(self, mean, std, best_y, **kwargs):
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        # Monte Carlo estimate: sample fantasy observations, compute mean improvement
        kg = np.zeros_like(mean)
        for _ in range(self.n_fantasies):
            fantasy = mean + std * self.rng.randn(len(mean))
            # Value = max posterior mean after observing fantasy minus current best
            improvement = np.maximum(fantasy - best_y, 0.0)
            kg += improvement
        kg /= self.n_fantasies
        return kg


# ---------------------------------------------------------------------------
# Optimization History
# ---------------------------------------------------------------------------

class BayesOptHistory:
    """Tracks and analyzes Bayesian optimization runs."""

    def __init__(self):
        self.X = []
        self.y = []
        self.acquisition_values = []
        self.best_so_far = []
        self.iterations = 0

    def record(self, x, y_val, acq_val=None):
        """Record an observation."""
        self.X.append(np.asarray(x).copy())
        self.y.append(float(y_val))
        if acq_val is not None:
            self.acquisition_values.append(float(acq_val))
        current_best = max(self.y)
        self.best_so_far.append(current_best)
        self.iterations += 1

    def get_best(self):
        """Return (best_x, best_y)."""
        if not self.y:
            return None, None
        idx = int(np.argmax(self.y))
        return self.X[idx].copy(), self.y[idx]

    def get_regret(self, optimal_value):
        """Compute simple regret over iterations."""
        return [optimal_value - b for b in self.best_so_far]

    def get_cumulative_regret(self, optimal_value):
        """Compute cumulative regret."""
        regrets = [optimal_value - yi for yi in self.y]
        cum = []
        total = 0.0
        for r in regrets:
            total += r
            cum.append(total)
        return cum

    def convergence_rate(self, window=5):
        """Compute improvement rate over last `window` iterations."""
        if len(self.best_so_far) < window + 1:
            return None
        recent = self.best_so_far[-window:]
        return recent[-1] - recent[0]

    def summary(self):
        """Return summary dict."""
        best_x, best_y = self.get_best()
        return {
            'iterations': self.iterations,
            'best_x': best_x,
            'best_y': best_y,
            'best_history': list(self.best_so_far),
        }


# ---------------------------------------------------------------------------
# Bayesian Optimizer (GP Surrogate)
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """Bayesian Optimization with GP surrogate model.

    Args:
        bounds: list of (low, high) tuples for each dimension
        kernel: GP kernel (default: Matern 5/2)
        acquisition: AcquisitionFunction instance (default: EI)
        noise_variance: GP observation noise
        n_initial: number of initial random points
        seed: random seed
    """

    def __init__(self, bounds, kernel=None, acquisition=None,
                 noise_variance=1e-4, n_initial=5, seed=42):
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.kernel = kernel or MaternKernel(nu=2.5)
        self.acquisition = acquisition or ExpectedImprovement()
        self.noise_variance = noise_variance
        self.n_initial = n_initial
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.gp = GPRegression(self.kernel, noise_variance=noise_variance)
        self.history = BayesOptHistory()
        self._fitted = False

    def _random_points(self, n):
        """Generate n random points within bounds."""
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            points[:, d] = self.rng.uniform(self.bounds[d, 0], self.bounds[d, 1], n)
        return points

    def _maximize_acquisition(self, n_candidates=1000):
        """Find point that maximizes the acquisition function."""
        candidates = self._random_points(n_candidates)
        X_data = np.array(self.history.X)
        y_data = np.array(self.history.y)

        self.gp.fit(X_data, y_data)
        self._fitted = True

        mean, std = self.gp.predict(candidates, return_std=True)
        best_y = max(self.history.y)
        acq_values = self.acquisition.evaluate(mean, std, best_y)
        best_idx = np.argmax(acq_values)
        return candidates[best_idx], acq_values[best_idx]

    def suggest(self, n_candidates=1000):
        """Suggest the next point to evaluate.

        Returns: (x_next, acquisition_value)
        """
        if len(self.history.y) < self.n_initial:
            x = self._random_points(1)[0]
            return x, 0.0
        return self._maximize_acquisition(n_candidates)

    def observe(self, x, y_val):
        """Record an observation."""
        self.history.record(x, y_val)

    def optimize(self, objective, n_iter=20, n_candidates=1000, verbose=False):
        """Run full optimization loop.

        Args:
            objective: callable f(x) -> scalar
            n_iter: total iterations (including initial)
            n_candidates: candidates for acquisition maximization
            verbose: print progress

        Returns: (best_x, best_y, history)
        """
        for i in range(n_iter):
            x_next, acq_val = self.suggest(n_candidates)
            y_val = objective(x_next)
            self.observe(x_next, y_val)
            self.history.acquisition_values.append(acq_val) if len(self.history.acquisition_values) < len(self.history.y) else None
            if verbose:
                best_x, best_y = self.history.get_best()
                print(f"Iter {i+1}/{n_iter}: y={y_val:.4f}, best={best_y:.4f}")

        best_x, best_y = self.history.get_best()
        return best_x, best_y, self.history

    def predict(self, X):
        """Predict using the GP surrogate (must have enough data)."""
        if not self._fitted:
            X_data = np.array(self.history.X)
            y_data = np.array(self.history.y)
            self.gp.fit(X_data, y_data)
            self._fitted = True
        return self.gp.predict(np.asarray(X), return_std=True)

    def get_best(self):
        """Return (best_x, best_y)."""
        return self.history.get_best()


# ---------------------------------------------------------------------------
# BNN Bayesian Optimizer
# ---------------------------------------------------------------------------

class BNNBayesianOptimizer:
    """Bayesian Optimization using BNN surrogate.

    Uses a Bayesian Neural Network instead of GP for scalability
    to higher dimensions and larger datasets.

    Args:
        bounds: list of (low, high) tuples
        hidden_sizes: BNN hidden layer sizes
        acquisition: AcquisitionFunction instance
        n_initial: initial random samples
        n_epochs: BNN training epochs per refit
        n_samples: MC samples for uncertainty
        lr: learning rate
        seed: random seed
    """

    def __init__(self, bounds, hidden_sizes=None, acquisition=None,
                 n_initial=5, n_epochs=100, n_samples=30, lr=0.01, seed=42):
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.hidden_sizes = hidden_sizes or [32, 32]
        self.acquisition = acquisition or ExpectedImprovement()
        self.n_initial = n_initial
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.lr = lr
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.network = None
        self.trainer = None
        self.predictive = None
        self.history = BayesOptHistory()
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

    def _build_network(self):
        """Build BNN with appropriate architecture."""
        layer_sizes = [self.dim] + self.hidden_sizes + [1]
        self.network = build_bnn(layer_sizes, activation='relu', seed=self.seed)
        self.trainer = BayesByBackprop(self.network, lr=self.lr, seed=self.seed)
        self.predictive = BNNPredictive(self.network, method='bbb')

    def _normalize_X(self, X):
        """Normalize inputs to [0, 1]."""
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        ranges = np.where(ranges < 1e-12, 1.0, ranges)
        return (X - self.bounds[:, 0]) / ranges

    def _normalize_y(self, y):
        """Standardize outputs."""
        y = np.asarray(y)
        self._y_mean = np.mean(y)
        self._y_std = max(np.std(y), 1e-6)
        return (y - self._y_mean) / self._y_std

    def _denormalize_y_stats(self, mean, std):
        """Convert predictions back to original scale."""
        return mean * self._y_std + self._y_mean, std * self._y_std

    def _random_points(self, n):
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            points[:, d] = self.rng.uniform(self.bounds[d, 0], self.bounds[d, 1], n)
        return points

    def _refit(self):
        """Retrain BNN on all data."""
        X = np.array(self.history.X)
        y = np.array(self.history.y).reshape(-1, 1)

        X_norm = self._normalize_X(X)
        y_norm = self._normalize_y(y)

        self._build_network()
        self.trainer.fit(X_norm, y_norm, epochs=self.n_epochs, verbose=False)

    def suggest(self, n_candidates=500):
        """Suggest next point to evaluate."""
        if len(self.history.y) < self.n_initial:
            return self._random_points(1)[0], 0.0

        self._refit()

        candidates = self._random_points(n_candidates)
        cand_norm = self._normalize_X(candidates)

        pred = self.predictive.predict(cand_norm, n_samples=self.n_samples, seed=self.seed)
        mean_norm = pred['mean'].ravel()
        std_norm = pred['std'].ravel()

        mean, std = self._denormalize_y_stats(mean_norm, std_norm)
        best_y = max(self.history.y)
        acq_values = self.acquisition.evaluate(mean, std, best_y)
        best_idx = np.argmax(acq_values)
        return candidates[best_idx], acq_values[best_idx]

    def observe(self, x, y_val):
        self.history.record(x, y_val)

    def optimize(self, objective, n_iter=20, n_candidates=500, verbose=False):
        for i in range(n_iter):
            x_next, acq_val = self.suggest(n_candidates)
            y_val = objective(x_next)
            self.observe(x_next, y_val)
            if verbose:
                _, best_y = self.history.get_best()
                print(f"Iter {i+1}/{n_iter}: y={y_val:.4f}, best={best_y:.4f}")

        best_x, best_y = self.history.get_best()
        return best_x, best_y, self.history

    def get_best(self):
        return self.history.get_best()


# ---------------------------------------------------------------------------
# Batch Bayesian Optimization
# ---------------------------------------------------------------------------

class BatchBayesianOptimizer:
    """Batch (parallel) Bayesian Optimization using kriging believer.

    Selects a batch of q points to evaluate in parallel by iteratively
    picking the best acquisition point, then "hallucinating" the observation
    at the GP mean (kriging believer strategy).

    Args:
        bounds: list of (low, high) tuples
        batch_size: number of points per batch
        kernel: GP kernel
        acquisition: AcquisitionFunction instance
        noise_variance: GP noise
        n_initial: initial random points
        seed: random seed
    """

    def __init__(self, bounds, batch_size=4, kernel=None, acquisition=None,
                 noise_variance=1e-4, n_initial=5, seed=42):
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.batch_size = batch_size
        self.kernel = kernel or MaternKernel(nu=2.5)
        self.acquisition = acquisition or ExpectedImprovement()
        self.noise_variance = noise_variance
        self.n_initial = n_initial
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.history = BayesOptHistory()

    def _random_points(self, n):
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            points[:, d] = self.rng.uniform(self.bounds[d, 0], self.bounds[d, 1], n)
        return points

    def suggest_batch(self, n_candidates=1000):
        """Suggest a batch of points using kriging believer."""
        if len(self.history.y) < self.n_initial:
            n_need = min(self.batch_size, self.n_initial - len(self.history.y))
            return self._random_points(n_need)

        X_data = np.array(self.history.X)
        y_data = np.array(self.history.y)

        # Kriging believer: iteratively select points
        batch = []
        X_aug = X_data.copy()
        y_aug = y_data.copy()

        for _ in range(self.batch_size):
            gp = GPRegression(self.kernel, noise_variance=self.noise_variance)
            gp.fit(X_aug, y_aug)

            candidates = self._random_points(n_candidates)
            mean, std = gp.predict(candidates, return_std=True)
            best_y = max(y_aug)
            acq_values = self.acquisition.evaluate(mean, std, best_y)
            best_idx = np.argmax(acq_values)
            x_new = candidates[best_idx]
            batch.append(x_new)

            # Hallucinate: add with GP mean as "observation"
            X_aug = np.vstack([X_aug, x_new.reshape(1, -1)])
            y_aug = np.append(y_aug, float(mean[best_idx]))

        return np.array(batch)

    def observe_batch(self, X_batch, y_batch):
        """Record a batch of observations."""
        for x, y_val in zip(X_batch, y_batch):
            self.history.record(x, y_val)

    def optimize(self, objective, n_batches=10, n_candidates=1000, verbose=False):
        """Run batch optimization.

        Args:
            objective: callable f(x) -> scalar (or vectorized f(X) -> array)
            n_batches: number of batch iterations
            n_candidates: candidates for acq maximization

        Returns: (best_x, best_y, history)
        """
        for b in range(n_batches):
            batch = self.suggest_batch(n_candidates)
            y_vals = np.array([objective(x) for x in batch])
            self.observe_batch(batch, y_vals)
            if verbose:
                _, best_y = self.history.get_best()
                print(f"Batch {b+1}/{n_batches}: batch_best={max(y_vals):.4f}, overall_best={best_y:.4f}")

        return self.history.get_best()[0], self.history.get_best()[1], self.history

    def get_best(self):
        return self.history.get_best()


# ---------------------------------------------------------------------------
# Multi-Objective Bayesian Optimization
# ---------------------------------------------------------------------------

def _dominates(a, b):
    """Check if solution a dominates b (all objectives >= and at least one >)."""
    a, b = np.asarray(a), np.asarray(b)
    return np.all(a >= b) and np.any(a > b)


def _compute_pareto_front(Y):
    """Compute indices of Pareto-optimal points (maximization)."""
    Y = np.asarray(Y)
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


def _hypervolume_2d(pareto_Y, ref_point):
    """Compute 2D hypervolume indicator."""
    pareto_Y = np.asarray(pareto_Y)
    ref = np.asarray(ref_point)
    # Filter points that dominate the reference point
    valid = np.all(pareto_Y > ref, axis=1)
    if not np.any(valid):
        return 0.0
    pts = pareto_Y[valid]
    # Sort by first objective descending
    order = np.argsort(-pts[:, 0])
    pts = pts[order]

    hv = 0.0
    prev_y = ref[1]
    for p in pts:
        if p[1] > prev_y:
            hv += (p[0] - ref[0]) * (p[1] - prev_y)
            prev_y = p[1]
    return hv


class MultiObjectiveBO:
    """Multi-Objective Bayesian Optimization.

    Uses Expected Hypervolume Improvement (EHVI) approximation
    with independent GP models per objective.

    Args:
        bounds: list of (low, high) tuples
        n_objectives: number of objectives
        ref_point: reference point for hypervolume (list of floats)
        kernel: GP kernel (shared across objectives)
        noise_variance: GP noise
        n_initial: initial random points
        seed: random seed
    """

    def __init__(self, bounds, n_objectives=2, ref_point=None, kernel=None,
                 noise_variance=1e-4, n_initial=5, seed=42):
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.n_objectives = n_objectives
        self.ref_point = np.asarray(ref_point) if ref_point is not None else np.full(n_objectives, -10.0)
        self.kernel = kernel or MaternKernel(nu=2.5)
        self.noise_variance = noise_variance
        self.n_initial = n_initial
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.X = []
        self.Y = []  # list of arrays, each of shape (n_objectives,)
        self._gps = None

    def _random_points(self, n):
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            points[:, d] = self.rng.uniform(self.bounds[d, 0], self.bounds[d, 1], n)
        return points

    def _fit_gps(self):
        """Fit one GP per objective."""
        X = np.array(self.X)
        Y = np.array(self.Y)
        self._gps = []
        for obj_idx in range(self.n_objectives):
            gp = GPRegression(self.kernel, noise_variance=self.noise_variance)
            gp.fit(X, Y[:, obj_idx])
            self._gps.append(gp)

    def _ehvi_acquisition(self, candidates, n_mc=100):
        """Approximate EHVI via Monte Carlo sampling."""
        n_cand = len(candidates)
        # Get GP predictions for each objective
        means = []
        stds = []
        for gp in self._gps:
            m, s = gp.predict(candidates, return_std=True)
            means.append(m.ravel())
            stds.append(s.ravel())

        # Current Pareto front
        Y_arr = np.array(self.Y)
        pareto_idx = _compute_pareto_front(Y_arr)
        pareto_Y = Y_arr[pareto_idx]

        if len(pareto_Y) > 0 and self.n_objectives == 2:
            current_hv = _hypervolume_2d(pareto_Y, self.ref_point)
        else:
            current_hv = 0.0

        # MC estimate of EHVI
        ehvi = np.zeros(n_cand)
        for _ in range(n_mc):
            # Sample from posterior for each objective
            samples = np.zeros((n_cand, self.n_objectives))
            for obj_idx in range(self.n_objectives):
                samples[:, obj_idx] = means[obj_idx] + stds[obj_idx] * self.rng.randn(n_cand)

            for i in range(n_cand):
                # Add fantasy point to Pareto front
                aug_Y = np.vstack([pareto_Y, samples[i:i+1]]) if len(pareto_Y) > 0 else samples[i:i+1]
                new_pareto_idx = _compute_pareto_front(aug_Y)
                new_pareto = aug_Y[new_pareto_idx]
                if self.n_objectives == 2:
                    new_hv = _hypervolume_2d(new_pareto, self.ref_point)
                else:
                    # Fallback: sum of improvements
                    new_hv = current_hv + max(0, np.min(samples[i] - self.ref_point))
                ehvi[i] += max(0.0, new_hv - current_hv)

        ehvi /= n_mc
        return ehvi

    def suggest(self, n_candidates=500):
        """Suggest next point to evaluate."""
        if len(self.Y) < self.n_initial:
            return self._random_points(1)[0]

        self._fit_gps()
        candidates = self._random_points(n_candidates)
        ehvi = self._ehvi_acquisition(candidates, n_mc=50)
        return candidates[np.argmax(ehvi)]

    def observe(self, x, y_vec):
        """Record observation. y_vec is array of shape (n_objectives,)."""
        self.X.append(np.asarray(x).copy())
        self.Y.append(np.asarray(y_vec).copy())

    def optimize(self, objectives, n_iter=20, n_candidates=500, verbose=False):
        """Run optimization.

        Args:
            objectives: callable f(x) -> array of shape (n_objectives,)
            n_iter: number of iterations
            n_candidates: acquisition candidates

        Returns: (pareto_X, pareto_Y, all_X, all_Y)
        """
        for i in range(n_iter):
            x_next = self.suggest(n_candidates)
            y_val = np.asarray(objectives(x_next))
            self.observe(x_next, y_val)
            if verbose:
                print(f"Iter {i+1}/{n_iter}: y={y_val}")

        return self.get_pareto_front()

    def get_pareto_front(self):
        """Return (pareto_X, pareto_Y, all_X, all_Y)."""
        if len(self.Y) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        Y_arr = np.array(self.Y)
        X_arr = np.array(self.X)
        pareto_idx = _compute_pareto_front(Y_arr)
        return X_arr[pareto_idx], Y_arr[pareto_idx], X_arr, Y_arr

    def hypervolume(self):
        """Compute current hypervolume of Pareto front."""
        if len(self.Y) < 2:
            return 0.0
        Y_arr = np.array(self.Y)
        pareto_idx = _compute_pareto_front(Y_arr)
        pareto_Y = Y_arr[pareto_idx]
        if self.n_objectives == 2:
            return _hypervolume_2d(pareto_Y, self.ref_point)
        return 0.0


# ---------------------------------------------------------------------------
# Constrained Bayesian Optimization
# ---------------------------------------------------------------------------

class ConstrainedBO:
    """Bayesian Optimization with black-box constraints.

    Uses Expected Feasible Improvement: EFI(x) = EI(x) * prod(P(c_j(x) >= 0))
    Each constraint has its own GP model.

    Args:
        bounds: list of (low, high) tuples
        n_constraints: number of constraint functions
        kernel: GP kernel
        acquisition: base acquisition function
        noise_variance: GP noise
        n_initial: initial random points
        seed: random seed
    """

    def __init__(self, bounds, n_constraints=1, kernel=None, acquisition=None,
                 noise_variance=1e-4, n_initial=5, seed=42):
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.n_constraints = n_constraints
        self.kernel = kernel or MaternKernel(nu=2.5)
        self.acquisition = acquisition or ExpectedImprovement()
        self.noise_variance = noise_variance
        self.n_initial = n_initial
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.history = BayesOptHistory()
        self.constraint_values = []  # list of arrays, each (n_constraints,)

    def _random_points(self, n):
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            points[:, d] = self.rng.uniform(self.bounds[d, 0], self.bounds[d, 1], n)
        return points

    def suggest(self, n_candidates=1000):
        """Suggest next point using Expected Feasible Improvement."""
        if len(self.history.y) < self.n_initial:
            return self._random_points(1)[0]

        X_data = np.array(self.history.X)
        y_data = np.array(self.history.y)
        c_data = np.array(self.constraint_values)  # (n_obs, n_constraints)

        # Fit objective GP
        obj_gp = GPRegression(self.kernel, noise_variance=self.noise_variance)
        obj_gp.fit(X_data, y_data)

        # Fit constraint GPs
        constraint_gps = []
        for j in range(self.n_constraints):
            cgp = GPRegression(self.kernel, noise_variance=self.noise_variance)
            cgp.fit(X_data, c_data[:, j])
            constraint_gps.append(cgp)

        candidates = self._random_points(n_candidates)

        # Objective acquisition
        mean_obj, std_obj = obj_gp.predict(candidates, return_std=True)

        # Best feasible value
        feasible_mask = np.all(c_data >= 0, axis=1) if len(c_data) > 0 else np.array([])
        if np.any(feasible_mask):
            best_y = max(y_data[feasible_mask])
        else:
            best_y = min(y_data)  # no feasible yet, use worst as reference

        acq = self.acquisition.evaluate(mean_obj, std_obj, best_y)

        # Probability of feasibility for each constraint
        pof = np.ones(len(candidates))
        for j in range(self.n_constraints):
            mean_c, std_c = constraint_gps[j].predict(candidates, return_std=True)
            mean_c = mean_c.ravel()
            std_c = std_c.ravel()
            # P(c_j(x) >= 0) = Phi(mean / std)
            mask = std_c > 1e-12
            z = np.zeros_like(mean_c)
            z[mask] = mean_c[mask] / std_c[mask]
            pof *= _standard_normal_cdf(z)

        # EFI = acquisition * product of feasibility probabilities
        efi = acq * pof
        best_idx = np.argmax(efi)
        return candidates[best_idx]

    def observe(self, x, y_val, constraint_vals):
        """Record observation with constraint values.

        constraint_vals: array of shape (n_constraints,), >= 0 means feasible.
        """
        self.history.record(x, y_val)
        self.constraint_values.append(np.asarray(constraint_vals).copy())

    def optimize(self, objective, constraints, n_iter=20, n_candidates=1000, verbose=False):
        """Run constrained optimization.

        Args:
            objective: callable f(x) -> scalar
            constraints: callable g(x) -> array of shape (n_constraints,), >= 0 is feasible
            n_iter: number of iterations
            n_candidates: acquisition candidates

        Returns: (best_feasible_x, best_feasible_y, history)
        """
        for i in range(n_iter):
            x_next = self.suggest(n_candidates)
            y_val = objective(x_next)
            c_vals = np.asarray(constraints(x_next))
            self.observe(x_next, y_val, c_vals)
            if verbose:
                feasible = np.all(c_vals >= 0)
                _, best_y = self.get_best_feasible()
                print(f"Iter {i+1}/{n_iter}: y={y_val:.4f}, feasible={feasible}, best_feasible={best_y}")

        return self.get_best_feasible()[0], self.get_best_feasible()[1], self.history

    def get_best_feasible(self):
        """Return best feasible (x, y) or (None, None) if none feasible."""
        if len(self.constraint_values) == 0:
            return None, None
        c_arr = np.array(self.constraint_values)
        feasible_mask = np.all(c_arr >= 0, axis=1)
        if not np.any(feasible_mask):
            return None, None
        feasible_idx = np.where(feasible_mask)[0]
        y_arr = np.array(self.history.y)
        best_idx = feasible_idx[np.argmax(y_arr[feasible_idx])]
        return self.history.X[best_idx].copy(), self.history.y[best_idx]


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def create_optimizer(bounds, method='gp', acquisition='ei', **kwargs):
    """Factory function to create optimizers.

    Args:
        bounds: list of (low, high) tuples
        method: 'gp', 'bnn', 'batch', 'multi_objective', 'constrained'
        acquisition: 'ei', 'pi', 'ucb', 'thompson', 'kg'
        **kwargs: additional arguments passed to optimizer

    Returns: optimizer instance
    """
    acq_map = {
        'ei': ExpectedImprovement,
        'pi': ProbabilityOfImprovement,
        'ucb': UpperConfidenceBound,
        'thompson': ThompsonSampling,
        'kg': KnowledgeGradient,
    }

    acq_cls = acq_map.get(acquisition, ExpectedImprovement)
    acq = acq_cls(**{k: v for k, v in kwargs.items()
                     if k in ('xi', 'kappa', 'n_fantasies', 'seed') and k in acq_cls.__init__.__code__.co_varnames})

    if method == 'gp':
        return BayesianOptimizer(bounds, acquisition=acq, **{k: v for k, v in kwargs.items()
                                 if k in ('kernel', 'noise_variance', 'n_initial', 'seed')})
    elif method == 'bnn':
        return BNNBayesianOptimizer(bounds, acquisition=acq, **{k: v for k, v in kwargs.items()
                                    if k in ('hidden_sizes', 'n_initial', 'n_epochs', 'n_samples', 'lr', 'seed')})
    elif method == 'batch':
        return BatchBayesianOptimizer(bounds, acquisition=acq, **{k: v for k, v in kwargs.items()
                                     if k in ('batch_size', 'kernel', 'noise_variance', 'n_initial', 'seed')})
    elif method == 'multi_objective':
        return MultiObjectiveBO(bounds, **{k: v for k, v in kwargs.items()
                                if k in ('n_objectives', 'ref_point', 'kernel', 'noise_variance', 'n_initial', 'seed')})
    elif method == 'constrained':
        return ConstrainedBO(bounds, acquisition=acq, **{k: v for k, v in kwargs.items()
                             if k in ('n_constraints', 'kernel', 'noise_variance', 'n_initial', 'seed')})
    else:
        raise ValueError(f"Unknown method: {method}")
