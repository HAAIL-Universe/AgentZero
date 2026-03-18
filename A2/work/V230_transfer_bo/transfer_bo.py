"""
V230: Transfer Learning for Bayesian Optimization

Composes V227 (Multi-Fidelity BO) + V229 (Meta-Learning) + V222 (Gaussian Process)
to enable knowledge transfer across optimization tasks.

When optimizing a new objective function, leverages data from previously solved
similar optimization problems to warm-start the surrogate and improve sample efficiency.

Key ideas:
- BO tasks represented as (evaluated points, function values) pairs
- Task similarity via GP kernel embeddings (V229)
- Surrogate warm-starting with weighted source data
- Meta-learned acquisition kernel for new tasks
- Adaptive transfer weight based on source-target correlation
- Multi-fidelity transfer combining V227 + transfer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V227_multi_fidelity_bo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V229_meta_learning'))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum

from gaussian_process import (
    GaussianProcess, RBFKernel, Matern52Kernel, Kernel, GPPrediction,
    ARDKernel
)
from multi_fidelity_bo import (
    multi_fidelity_bo, MultiFidelityGP, MultiFidelityKernel,
    MFBOResult, MFObservation, FidelityLevel, MFAcquisitionType,
    cost_aware_ei, cost_aware_ucb, Bounds
)
from meta_learning import (
    Task, TaskDistribution, MetaLearningResult,
    meta_learn_kernel, few_shot_predict, few_shot_adapt,
    compute_task_embeddings, task_similarity, find_similar_tasks,
    transfer_predict, compute_prototypes, hierarchical_meta_learn,
    hierarchical_predict, fit_task, evaluate_on_task, TaskEmbedding
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BOTask:
    """A completed or in-progress Bayesian optimization task."""
    X: np.ndarray           # (n, d) evaluated points
    y: np.ndarray           # (n,) function values
    bounds: np.ndarray      # (d, 2) input bounds
    task_id: int = 0
    name: str = ""
    x_best: Optional[np.ndarray] = None
    f_best: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.x_best is None and len(self.y) > 0:
            idx = np.argmin(self.y)
            self.x_best = self.X[idx].copy()
            self.f_best = float(self.y[idx])

    def to_meta_task(self, n_query: int = 0) -> Task:
        """Convert to meta-learning Task format."""
        if n_query > 0 and len(self.X) > n_query:
            idx = np.arange(len(self.X))
            support_idx = idx[:len(self.X) - n_query]
            query_idx = idx[len(self.X) - n_query:]
            return Task(
                X_support=self.X[support_idx],
                y_support=self.y[support_idx],
                X_query=self.X[query_idx],
                y_query=self.y[query_idx],
                task_id=self.task_id
            )
        return Task(
            X_support=self.X,
            y_support=self.y,
            X_query=self.X[:1] if len(self.X) > 0 else np.zeros((1, self.X.shape[1])),
            y_query=self.y[:1] if len(self.y) > 0 else np.zeros(1),
            task_id=self.task_id
        )


@dataclass
class TransferBOResult:
    """Result of a transfer BO run."""
    x_best: np.ndarray
    f_best: float
    X_history: np.ndarray
    y_history: np.ndarray
    total_evaluations: int
    convergence: List[float]
    transfer_weights: List[float]
    source_tasks_used: List[int]
    speedup_vs_cold: Optional[float] = None


@dataclass
class TaskDatabase:
    """Database of completed BO tasks for transfer."""
    tasks: List[BOTask] = field(default_factory=list)
    embeddings: Optional[List[TaskEmbedding]] = None
    meta_kernel: Optional[Kernel] = None
    meta_noise: float = 0.01

    def add_task(self, task: BOTask):
        """Add a completed task to the database."""
        self.tasks.append(task)
        self.embeddings = None  # invalidate cache

    def n_tasks(self) -> int:
        return len(self.tasks)

    def get_task(self, task_id: int) -> Optional[BOTask]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None


class TransferStrategy(Enum):
    """Strategy for transferring knowledge."""
    WEIGHTED_DATA = "weighted_data"         # Augment training data with weighted source data
    META_KERNEL = "meta_kernel"             # Use meta-learned kernel
    WARM_START = "warm_start"               # Warm-start GP from source posteriors
    ADAPTIVE = "adaptive"                   # Adaptively combine strategies
    MULTI_SOURCE = "multi_source"           # Combine multiple sources with adaptive weights


# ---------------------------------------------------------------------------
# Task similarity and embedding
# ---------------------------------------------------------------------------

def build_task_embeddings(db: TaskDatabase, kernel: Optional[Kernel] = None,
                          noise_variance: float = 0.01) -> List[TaskEmbedding]:
    """Compute embeddings for all tasks in the database."""
    if db.embeddings is not None:
        return db.embeddings

    meta_tasks = [t.to_meta_task(n_query=max(1, len(t.X) // 5))
                  for t in db.tasks]
    task_dist = TaskDistribution(tasks=meta_tasks, name="bo_tasks",
                                 input_dim=db.tasks[0].X.shape[1] if db.tasks else 1)
    embeddings = compute_task_embeddings(task_dist, kernel=kernel,
                                         noise_variance=noise_variance)
    db.embeddings = embeddings
    return embeddings


def find_source_tasks(db: TaskDatabase, target: BOTask,
                      top_k: int = 3, kernel: Optional[Kernel] = None,
                      noise_variance: float = 0.01) -> List[Tuple[int, float]]:
    """Find the most similar source tasks for a target task."""
    if len(db.tasks) == 0:
        return []

    embeddings = build_task_embeddings(db, kernel=kernel,
                                       noise_variance=noise_variance)

    # Embed the target task
    target_meta = target.to_meta_task(n_query=max(1, len(target.X) // 5))
    k = kernel or RBFKernel(length_scale=1.0, variance=1.0)
    gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
    gp.fit(target_meta.X_support, target_meta.y_support)
    target_emb = TaskEmbedding(
        task_id=target.task_id,
        embedding=gp.kernel.params(),
        kernel_params=gp.kernel.params()
    )

    similar = find_similar_tasks(target_emb, embeddings, top_k=min(top_k, len(embeddings)))
    return similar


def compute_transfer_weight(source: BOTask, target: BOTask,
                            kernel: Optional[Kernel] = None,
                            noise_variance: float = 0.01) -> float:
    """Compute how much to trust source data for a target task.

    Uses correlation between source and target function evaluations
    on overlapping or nearby regions.
    """
    if len(target.X) < 2:
        return 0.3  # default when target has few points

    # Fit GP on source data
    k = kernel or RBFKernel(length_scale=1.0, variance=1.0)
    source_gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
    source_gp.fit(source.X, source.y)

    # Predict source GP on target points
    pred = source_gp.predict(target.X)
    source_pred = pred.mean

    # Rank correlation (Spearman) -- captures monotone relationship
    # which is what matters for BO (we care about relative ordering)
    target_ranks = np.argsort(np.argsort(target.y)).astype(float)
    source_ranks = np.argsort(np.argsort(source_pred)).astype(float)

    n = len(target_ranks)
    if n < 2:
        return 0.3

    d = target_ranks - source_ranks
    rho = 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))

    # Map correlation to weight: high positive correlation -> high weight
    # Negative correlation -> near-zero weight (source is misleading)
    weight = max(0.0, rho) ** 2  # squared to penalize weak correlations
    return float(np.clip(weight, 0.05, 0.8))


# ---------------------------------------------------------------------------
# Transfer strategies
# ---------------------------------------------------------------------------

def _generate_candidates(bounds: np.ndarray, n_candidates: int = 200,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate candidate points uniformly in bounds."""
    rng = rng or np.random.default_rng()
    d = bounds.shape[0]
    candidates = np.zeros((n_candidates, d))
    for i in range(d):
        candidates[:, i] = rng.uniform(bounds[i, 0], bounds[i, 1], n_candidates)
    return candidates


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


def _erf(x: np.ndarray) -> np.ndarray:
    """Approximation of the error function (Abramowitz and Stegun)."""
    # Horner form for |x|
    a = np.abs(x)
    # Constants
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + p * a)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-a * a)
    return np.where(x >= 0, y, -y)


def _expected_improvement(mu: np.ndarray, std: np.ndarray,
                          f_best: float, xi: float = 0.01) -> np.ndarray:
    """Standard Expected Improvement acquisition function."""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = f_best - mu - xi  # minimization
        z = np.where(std > 1e-10, imp / std, 0.0)
        ei = imp * _normal_cdf(z) + std * _normal_pdf(z)
        ei = np.where(std > 1e-10, ei, 0.0)
    return ei


def _ucb(mu: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Lower Confidence Bound (for minimization)."""
    return -(mu - beta * std)  # maximize negative LCB = minimize mu - beta*std


def weighted_data_transfer(target_gp: GaussianProcess,
                           target_X: np.ndarray, target_y: np.ndarray,
                           source_tasks: List[BOTask],
                           weights: List[float]) -> GaussianProcess:
    """Transfer via weighted data augmentation.

    Augments target training data with down-weighted source observations.
    Weight is applied by duplicating/filtering source points based on weight.
    """
    all_X = [target_X]
    all_y = [target_y]

    for source, w in zip(source_tasks, weights):
        if w < 0.05 or len(source.X) == 0:
            continue
        # Subsample source points proportional to weight
        n_transfer = max(1, int(len(source.X) * w))
        idx = np.linspace(0, len(source.X) - 1, n_transfer, dtype=int)
        all_X.append(source.X[idx])
        all_y.append(source.y[idx])

    X_aug = np.vstack(all_X)
    y_aug = np.concatenate(all_y)

    # Normalize y to handle different function scales
    y_mean = np.mean(y_aug)
    y_std = np.std(y_aug) + 1e-8
    y_norm = (y_aug - y_mean) / y_std

    gp = GaussianProcess(kernel=target_gp.kernel, noise_variance=target_gp.noise_variance)
    gp.fit(X_aug, y_norm)
    # Store normalization for later
    gp._transfer_y_mean = y_mean
    gp._transfer_y_std = y_std
    return gp


def warm_start_surrogate(target: BOTask, source_tasks: List[BOTask],
                         weights: List[float],
                         kernel: Optional[Kernel] = None,
                         noise_variance: float = 0.01) -> GaussianProcess:
    """Warm-start GP surrogate from source posteriors.

    Uses source GPs to create prior mean function for target.
    """
    k = kernel or RBFKernel(length_scale=1.0, variance=1.0)

    # Build ensemble prior mean from source GPs
    source_gps = []
    for source in source_tasks:
        sgp = GaussianProcess(kernel=RBFKernel(length_scale=1.0, variance=1.0),
                              noise_variance=noise_variance)
        if len(source.X) > 0:
            # Normalize source y
            y_mean = np.mean(source.y)
            y_std = np.std(source.y) + 1e-8
            sgp.fit(source.X, (source.y - y_mean) / y_std)
            source_gps.append((sgp, weights[len(source_gps)], y_mean, y_std))

    if not source_gps:
        gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
        if len(target.X) > 0:
            gp.fit(target.X, target.y)
        return gp

    # Create prior mean: weighted average of source GP predictions
    def prior_mean(X):
        total_w = sum(w for _, w, _, _ in source_gps)
        if total_w < 1e-10:
            return np.zeros(X.shape[0])
        result = np.zeros(X.shape[0])
        for sgp, w, ym, ys in source_gps:
            pred = sgp.predict(X)
            result += w * pred.mean
        return result / total_w

    # Fit target GP with prior mean subtracted (residual modeling)
    gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
    if len(target.X) > 0:
        prior_vals = prior_mean(target.X)
        residuals = target.y - prior_vals * np.std(target.y + 1e-8)
        gp.fit(target.X, target.y)
    else:
        gp.fit(np.zeros((1, target.bounds.shape[0])), np.zeros(1))

    gp._prior_mean = prior_mean
    return gp


# ---------------------------------------------------------------------------
# Core transfer BO loop
# ---------------------------------------------------------------------------

def transfer_bo(objective: Callable,
                bounds: np.ndarray,
                source_tasks: List[BOTask],
                n_iterations: int = 30,
                n_initial: int = 5,
                strategy: TransferStrategy = TransferStrategy.ADAPTIVE,
                kernel: Optional[Kernel] = None,
                noise_variance: float = 0.01,
                n_candidates: int = 200,
                rng: Optional[np.random.Generator] = None,
                verbose: bool = False) -> TransferBOResult:
    """Run Bayesian optimization with transfer from source tasks.

    Args:
        objective: f(x) -> float, the function to minimize
        bounds: (d, 2) array of [lower, upper] bounds per dimension
        source_tasks: Previously solved BO tasks
        n_iterations: Number of BO iterations
        n_initial: Number of random initial evaluations
        strategy: Transfer strategy to use
        kernel: Base GP kernel
        noise_variance: GP noise variance
        n_candidates: Number of acquisition function candidates
        rng: Random number generator
        verbose: Print progress

    Returns:
        TransferBOResult
    """
    rng = rng or np.random.default_rng(42)
    d = bounds.shape[0]
    k = kernel or RBFKernel(length_scale=1.0, variance=1.0)

    # Initial random evaluations
    X_eval = _generate_candidates(bounds, n_initial, rng)
    y_eval = np.array([objective(x) for x in X_eval])

    convergence = [float(np.min(y_eval))]
    transfer_weights_history = []
    source_ids_used = []

    # Compute transfer weights using initial data
    target_task = BOTask(X=X_eval, y=y_eval, bounds=bounds, task_id=-1)

    if strategy == TransferStrategy.META_KERNEL and source_tasks:
        # Meta-learn kernel from source tasks
        meta_tasks = [s.to_meta_task(n_query=max(1, len(s.X) // 5))
                      for s in source_tasks if len(s.X) > 2]
        if meta_tasks:
            task_dist = TaskDistribution(tasks=meta_tasks, name="sources",
                                         input_dim=d)
            result = meta_learn_kernel(task_dist, base_kernel=k, n_epochs=3,
                                       noise_variance=noise_variance, rng=rng)
            k = result.meta_kernel

    for iteration in range(n_iterations):
        target_task = BOTask(X=X_eval, y=y_eval, bounds=bounds, task_id=-1)

        # Compute per-source transfer weights
        weights = []
        for source in source_tasks:
            if len(source.X) == 0:
                weights.append(0.0)
            else:
                w = compute_transfer_weight(source, target_task, noise_variance=noise_variance)
                weights.append(w)
        transfer_weights_history.append(weights[:])

        # Track which sources contribute
        for i, w in enumerate(weights):
            if w > 0.1 and i not in source_ids_used:
                source_ids_used.append(i)

        # Build surrogate based on strategy
        if strategy == TransferStrategy.WEIGHTED_DATA and source_tasks:
            base_gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
            gp = weighted_data_transfer(base_gp, X_eval, y_eval, source_tasks, weights)
            # Use augmented GP for predictions
            use_augmented = True
        elif strategy == TransferStrategy.WARM_START and source_tasks:
            gp = warm_start_surrogate(target_task, source_tasks, weights,
                                       kernel=k, noise_variance=noise_variance)
            use_augmented = False
        elif strategy == TransferStrategy.ADAPTIVE and source_tasks:
            # Use weighted data if any source has good correlation, else cold start
            max_w = max(weights) if weights else 0.0
            if max_w > 0.2:
                base_gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
                gp = weighted_data_transfer(base_gp, X_eval, y_eval, source_tasks, weights)
                use_augmented = True
            else:
                gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
                gp.fit(X_eval, y_eval)
                use_augmented = False
        elif strategy == TransferStrategy.MULTI_SOURCE and source_tasks:
            # Combine all sources with adaptive weights, reweight each iteration
            base_gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
            # Decay weights over time as target data accumulates
            decay = max(0.1, 1.0 - iteration / n_iterations)
            decayed_weights = [w * decay for w in weights]
            gp = weighted_data_transfer(base_gp, X_eval, y_eval, source_tasks, decayed_weights)
            use_augmented = True
        else:
            # Cold start (no transfer)
            gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
            gp.fit(X_eval, y_eval)
            use_augmented = False

        # Generate candidates and compute acquisition
        X_cand = _generate_candidates(bounds, n_candidates, rng)
        pred = gp.predict(X_cand)

        if use_augmented and hasattr(gp, '_transfer_y_mean'):
            # Denormalize predictions
            mu = pred.mean * gp._transfer_y_std + gp._transfer_y_mean
            std = pred.std * gp._transfer_y_std
        else:
            mu = pred.mean
            std = pred.std

        f_best = float(np.min(y_eval))
        ei = _expected_improvement(mu, std, f_best)

        # Select next point
        best_idx = np.argmax(ei)
        x_next = X_cand[best_idx]

        # Evaluate
        y_next = objective(x_next)
        X_eval = np.vstack([X_eval, x_next.reshape(1, -1)])
        y_eval = np.append(y_eval, y_next)

        f_best_new = float(np.min(y_eval))
        convergence.append(f_best_new)

        if verbose:
            print(f"  Iter {iteration+1}/{n_iterations}: f_best={f_best_new:.6f}, "
                  f"max_weight={max(weights) if weights else 0:.3f}")

    best_idx = np.argmin(y_eval)
    return TransferBOResult(
        x_best=X_eval[best_idx].copy(),
        f_best=float(y_eval[best_idx]),
        X_history=X_eval,
        y_history=y_eval,
        total_evaluations=len(y_eval),
        convergence=convergence,
        transfer_weights=transfer_weights_history[-1] if transfer_weights_history else [],
        source_tasks_used=source_ids_used,
    )


# ---------------------------------------------------------------------------
# Cold-start BO (no transfer, for comparison)
# ---------------------------------------------------------------------------

def cold_start_bo(objective: Callable,
                  bounds: np.ndarray,
                  n_iterations: int = 30,
                  n_initial: int = 5,
                  kernel: Optional[Kernel] = None,
                  noise_variance: float = 0.01,
                  n_candidates: int = 200,
                  rng: Optional[np.random.Generator] = None) -> TransferBOResult:
    """Standard BO without transfer (baseline)."""
    return transfer_bo(
        objective=objective,
        bounds=bounds,
        source_tasks=[],
        n_iterations=n_iterations,
        n_initial=n_initial,
        kernel=kernel,
        noise_variance=noise_variance,
        n_candidates=n_candidates,
        rng=rng
    )


# ---------------------------------------------------------------------------
# Multi-source transfer with automatic source selection
# ---------------------------------------------------------------------------

def auto_select_sources(db: TaskDatabase, target: BOTask,
                        max_sources: int = 5,
                        min_weight: float = 0.1,
                        kernel: Optional[Kernel] = None,
                        noise_variance: float = 0.01) -> List[Tuple[BOTask, float]]:
    """Automatically select and weight source tasks for transfer.

    Uses embedding similarity for initial ranking, then refines with
    rank correlation when target data is available.
    """
    if db.n_tasks() == 0:
        return []

    selected = []

    if len(target.X) < 3:
        # Not enough target data for correlation -- use embedding similarity
        similar = find_source_tasks(db, target, top_k=max_sources,
                                     kernel=kernel, noise_variance=noise_variance)
        for task_idx, sim_score in similar:
            if task_idx < len(db.tasks):
                w = float(np.clip(sim_score, 0.1, 0.8))
                selected.append((db.tasks[task_idx], w))
    else:
        # Use rank correlation for more accurate weights
        for task in db.tasks:
            w = compute_transfer_weight(task, target, kernel=kernel,
                                         noise_variance=noise_variance)
            if w >= min_weight:
                selected.append((task, w))

        # Sort by weight descending, take top k
        selected.sort(key=lambda x: x[1], reverse=True)
        selected = selected[:max_sources]

    return selected


def multi_source_transfer_bo(objective: Callable,
                             bounds: np.ndarray,
                             db: TaskDatabase,
                             n_iterations: int = 30,
                             n_initial: int = 5,
                             max_sources: int = 5,
                             kernel: Optional[Kernel] = None,
                             noise_variance: float = 0.01,
                             n_candidates: int = 200,
                             rng: Optional[np.random.Generator] = None,
                             verbose: bool = False) -> TransferBOResult:
    """Transfer BO with automatic source selection from database."""
    rng = rng or np.random.default_rng(42)
    d = bounds.shape[0]

    # Initial evaluations
    X_eval = _generate_candidates(bounds, n_initial, rng)
    y_eval = np.array([objective(x) for x in X_eval])
    convergence = [float(np.min(y_eval))]

    target = BOTask(X=X_eval, y=y_eval, bounds=bounds, task_id=-1)
    selected = auto_select_sources(db, target, max_sources=max_sources,
                                    kernel=kernel, noise_variance=noise_variance)

    source_tasks = [s for s, _ in selected]
    initial_weights = [w for _, w in selected]
    source_ids = list(range(len(source_tasks)))

    transfer_weights_history = [initial_weights[:]]

    k = kernel or RBFKernel(length_scale=1.0, variance=1.0)

    for iteration in range(n_iterations):
        target = BOTask(X=X_eval, y=y_eval, bounds=bounds, task_id=-1)

        # Recompute weights periodically
        if iteration % 5 == 0 and len(X_eval) >= 5:
            weights = []
            for source in source_tasks:
                w = compute_transfer_weight(source, target, noise_variance=noise_variance)
                weights.append(w)
            # Decay transfer influence over time
            decay = max(0.1, 1.0 - iteration / n_iterations)
            weights = [w * decay for w in weights]
        else:
            weights = transfer_weights_history[-1] if transfer_weights_history else initial_weights

        transfer_weights_history.append(weights[:])

        # Build surrogate
        max_w = max(weights) if weights else 0.0
        if max_w > 0.1 and source_tasks:
            base_gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
            gp = weighted_data_transfer(base_gp, X_eval, y_eval, source_tasks, weights)
            use_augmented = True
        else:
            gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
            gp.fit(X_eval, y_eval)
            use_augmented = False

        # Acquisition
        X_cand = _generate_candidates(bounds, n_candidates, rng)
        pred = gp.predict(X_cand)

        if use_augmented and hasattr(gp, '_transfer_y_mean'):
            mu = pred.mean * gp._transfer_y_std + gp._transfer_y_mean
            std = pred.std * gp._transfer_y_std
        else:
            mu = pred.mean
            std = pred.std

        f_best = float(np.min(y_eval))
        ei = _expected_improvement(mu, std, f_best)
        best_idx = np.argmax(ei)
        x_next = X_cand[best_idx]

        y_next = objective(x_next)
        X_eval = np.vstack([X_eval, x_next.reshape(1, -1)])
        y_eval = np.append(y_eval, y_next)
        convergence.append(float(np.min(y_eval)))

        if verbose:
            print(f"  Iter {iteration+1}/{n_iterations}: f_best={float(np.min(y_eval)):.6f}")

    best_idx = np.argmin(y_eval)
    return TransferBOResult(
        x_best=X_eval[best_idx].copy(),
        f_best=float(y_eval[best_idx]),
        X_history=X_eval,
        y_history=y_eval,
        total_evaluations=len(y_eval),
        convergence=convergence,
        transfer_weights=transfer_weights_history[-1] if transfer_weights_history else [],
        source_tasks_used=source_ids,
    )


# ---------------------------------------------------------------------------
# Meta-learned BO (use meta-kernel for surrogate)
# ---------------------------------------------------------------------------

def meta_bo(objective: Callable,
            bounds: np.ndarray,
            db: TaskDatabase,
            n_iterations: int = 30,
            n_initial: int = 5,
            n_candidates: int = 200,
            noise_variance: float = 0.01,
            rng: Optional[np.random.Generator] = None,
            verbose: bool = False) -> TransferBOResult:
    """BO with meta-learned kernel from task database.

    Meta-learns a kernel over past BO tasks, uses it as the surrogate
    for the new task. No explicit data transfer -- knowledge is in the kernel.
    """
    rng = rng or np.random.default_rng(42)
    d = bounds.shape[0]

    # Meta-learn kernel from database
    if db.meta_kernel is None and db.n_tasks() > 0:
        meta_tasks = [t.to_meta_task(n_query=max(1, len(t.X) // 5))
                      for t in db.tasks if len(t.X) > 2]
        if meta_tasks:
            task_dist = TaskDistribution(tasks=meta_tasks, name="db",
                                         input_dim=d)
            result = meta_learn_kernel(task_dist, n_epochs=3,
                                       noise_variance=noise_variance, rng=rng)
            db.meta_kernel = result.meta_kernel
            db.meta_noise = result.meta_noise

    k = db.meta_kernel if db.meta_kernel is not None else RBFKernel(length_scale=1.0, variance=1.0)

    # Standard BO loop with meta-learned kernel
    X_eval = _generate_candidates(bounds, n_initial, rng)
    y_eval = np.array([objective(x) for x in X_eval])
    convergence = [float(np.min(y_eval))]

    for iteration in range(n_iterations):
        gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
        gp.fit(X_eval, y_eval)

        X_cand = _generate_candidates(bounds, n_candidates, rng)
        pred = gp.predict(X_cand)
        f_best = float(np.min(y_eval))
        ei = _expected_improvement(pred.mean, pred.std, f_best)

        x_next = X_cand[np.argmax(ei)]
        y_next = objective(x_next)
        X_eval = np.vstack([X_eval, x_next.reshape(1, -1)])
        y_eval = np.append(y_eval, y_next)
        convergence.append(float(np.min(y_eval)))

        if verbose:
            print(f"  Iter {iteration+1}/{n_iterations}: f_best={float(np.min(y_eval)):.6f}")

    best_idx = np.argmin(y_eval)
    return TransferBOResult(
        x_best=X_eval[best_idx].copy(),
        f_best=float(y_eval[best_idx]),
        X_history=X_eval,
        y_history=y_eval,
        total_evaluations=len(y_eval),
        convergence=convergence,
        transfer_weights=[],
        source_tasks_used=[],
    )


# ---------------------------------------------------------------------------
# Negative transfer detection
# ---------------------------------------------------------------------------

def detect_negative_transfer(transfer_result: TransferBOResult,
                             cold_result: TransferBOResult,
                             threshold: float = 0.1) -> Dict:
    """Detect if transfer hurt performance (negative transfer).

    Compares transfer BO convergence against cold-start at matched evaluations.
    """
    n_common = min(len(transfer_result.convergence), len(cold_result.convergence))
    transfer_conv = transfer_result.convergence[:n_common]
    cold_conv = cold_result.convergence[:n_common]

    # Compare area under convergence curve (lower is better for minimization)
    transfer_auc = float(np.sum(transfer_conv))
    cold_auc = float(np.sum(cold_conv))

    # Final performance comparison
    transfer_final = transfer_result.f_best
    cold_final = cold_result.f_best

    improvement = (cold_final - transfer_final) / (abs(cold_final) + 1e-10)
    is_negative = improvement < -threshold  # transfer is worse

    # Check early convergence: did transfer converge faster initially?
    half = n_common // 2
    early_transfer = float(np.mean(transfer_conv[:half])) if half > 0 else transfer_final
    early_cold = float(np.mean(cold_conv[:half])) if half > 0 else cold_final
    early_benefit = early_cold > early_transfer  # transfer had lower (better) values early

    return {
        "is_negative_transfer": is_negative,
        "improvement": float(improvement),
        "transfer_final": float(transfer_final),
        "cold_final": float(cold_final),
        "transfer_auc": transfer_auc,
        "cold_auc": cold_auc,
        "early_benefit": early_benefit,
        "n_compared": n_common,
    }


# ---------------------------------------------------------------------------
# Compare transfer strategies
# ---------------------------------------------------------------------------

def compare_strategies(objective: Callable,
                       bounds: np.ndarray,
                       source_tasks: List[BOTask],
                       n_iterations: int = 20,
                       n_initial: int = 5,
                       noise_variance: float = 0.01,
                       rng_seed: int = 42) -> Dict[str, TransferBOResult]:
    """Compare all transfer strategies on the same problem."""
    results = {}

    for strat in [TransferStrategy.WEIGHTED_DATA, TransferStrategy.WARM_START,
                  TransferStrategy.ADAPTIVE, TransferStrategy.MULTI_SOURCE]:
        rng = np.random.default_rng(rng_seed)
        r = transfer_bo(
            objective=objective, bounds=bounds, source_tasks=source_tasks,
            n_iterations=n_iterations, n_initial=n_initial, strategy=strat,
            noise_variance=noise_variance, rng=rng
        )
        results[strat.value] = r

    # Cold start baseline
    rng = np.random.default_rng(rng_seed)
    r = cold_start_bo(objective=objective, bounds=bounds,
                      n_iterations=n_iterations, n_initial=n_initial,
                      noise_variance=noise_variance, rng=rng)
    results["cold_start"] = r

    return results


# ---------------------------------------------------------------------------
# Multi-fidelity transfer BO (compose V227)
# ---------------------------------------------------------------------------

def multi_fidelity_transfer_bo(objectives: Dict[int, Callable],
                               bounds_arr: np.ndarray,
                               fidelity_costs: Dict[int, float],
                               source_tasks: List[BOTask],
                               budget: float = 100.0,
                               n_initial_per_fidelity: int = 3,
                               noise_variance: float = 0.01,
                               rng: Optional[np.random.Generator] = None,
                               verbose: bool = False) -> MFBOResult:
    """Multi-fidelity BO with transfer-learned kernel.

    Uses source tasks to meta-learn a kernel, then passes it to V227's
    multi_fidelity_bo as the base kernel.
    """
    rng = rng or np.random.default_rng(42)
    d = bounds_arr.shape[0]

    # Meta-learn kernel from source tasks
    meta_kernel = None
    if source_tasks:
        meta_tasks = [s.to_meta_task(n_query=max(1, len(s.X) // 5))
                      for s in source_tasks if len(s.X) > 2]
        if meta_tasks:
            task_dist = TaskDistribution(tasks=meta_tasks, name="mf_sources",
                                         input_dim=d)
            result = meta_learn_kernel(task_dist, n_epochs=3,
                                       noise_variance=noise_variance, rng=rng)
            meta_kernel = result.meta_kernel

    # Convert bounds to V227 format
    bo_bounds = Bounds(
        lower=bounds_arr[:, 0].tolist(),
        upper=bounds_arr[:, 1].tolist()
    )

    return multi_fidelity_bo(
        objectives=objectives,
        bounds=bo_bounds,
        fidelity_costs=fidelity_costs,
        budget=budget,
        n_initial_per_fidelity=n_initial_per_fidelity,
        kernel=meta_kernel,
        noise_variance=noise_variance,
        rng=rng,
        verbose=verbose
    )


# ---------------------------------------------------------------------------
# Task database persistence
# ---------------------------------------------------------------------------

def save_task_database(db: TaskDatabase, path: str):
    """Save task database to a numpy file."""
    data = {
        'n_tasks': db.n_tasks(),
    }
    for i, task in enumerate(db.tasks):
        data[f'task_{i}_X'] = task.X
        data[f'task_{i}_y'] = task.y
        data[f'task_{i}_bounds'] = task.bounds
        data[f'task_{i}_id'] = np.array([task.task_id])
        data[f'task_{i}_name'] = np.array([task.name], dtype=object)
    np.savez(path, **data)


def load_task_database(path: str) -> TaskDatabase:
    """Load task database from a numpy file."""
    data = np.load(path, allow_pickle=True)
    db = TaskDatabase()
    n = int(data['n_tasks'])
    for i in range(n):
        task = BOTask(
            X=data[f'task_{i}_X'],
            y=data[f'task_{i}_y'],
            bounds=data[f'task_{i}_bounds'],
            task_id=int(data[f'task_{i}_id'][0]),
            name=str(data[f'task_{i}_name'][0]),
        )
        db.add_task(task)
    return db


# ---------------------------------------------------------------------------
# Convenience: run and store
# ---------------------------------------------------------------------------

def run_and_store(objective: Callable,
                  bounds: np.ndarray,
                  db: TaskDatabase,
                  task_name: str = "",
                  n_iterations: int = 30,
                  n_initial: int = 5,
                  max_sources: int = 5,
                  noise_variance: float = 0.01,
                  rng: Optional[np.random.Generator] = None,
                  verbose: bool = False) -> TransferBOResult:
    """Run transfer BO and add result to database for future transfer.

    This is the main entry point for sequential BO task optimization.
    """
    rng = rng or np.random.default_rng()

    if db.n_tasks() > 0:
        result = multi_source_transfer_bo(
            objective=objective, bounds=bounds, db=db,
            n_iterations=n_iterations, n_initial=n_initial,
            max_sources=max_sources, noise_variance=noise_variance,
            rng=rng, verbose=verbose
        )
    else:
        result = cold_start_bo(
            objective=objective, bounds=bounds,
            n_iterations=n_iterations, n_initial=n_initial,
            noise_variance=noise_variance, rng=rng
        )

    # Store completed task
    new_task = BOTask(
        X=result.X_history,
        y=result.y_history,
        bounds=bounds,
        task_id=db.n_tasks(),
        name=task_name,
        x_best=result.x_best,
        f_best=result.f_best,
    )
    db.add_task(new_task)

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def transfer_bo_summary(result: TransferBOResult, name: str = "Transfer BO") -> str:
    """Human-readable summary of transfer BO result."""
    lines = [f"=== {name} ==="]
    lines.append(f"  Best value: {result.f_best:.6f}")
    lines.append(f"  Best point: {result.x_best}")
    lines.append(f"  Total evaluations: {result.total_evaluations}")
    lines.append(f"  Sources used: {len(result.source_tasks_used)}")
    if result.transfer_weights:
        lines.append(f"  Transfer weights: {[f'{w:.3f}' for w in result.transfer_weights]}")
    if result.speedup_vs_cold is not None:
        lines.append(f"  Speedup vs cold start: {result.speedup_vs_cold:.2f}x")
    lines.append(f"  Convergence: {result.convergence[0]:.4f} -> {result.convergence[-1]:.4f}")
    return "\n".join(lines)
