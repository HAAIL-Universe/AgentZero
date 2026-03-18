"""V226: Active Learning -- Data-efficient ML via intelligent query selection.

Composes V222 (Gaussian Process) for surrogate modeling and uncertainty
quantification. Implements pool-based, stream-based, and query synthesis
active learning with multiple acquisition strategies.

Core idea: instead of labeling all data, select the most informative
samples to label, achieving better models with fewer labels.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, List, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
from gaussian_process import (
    GaussianProcess, RBFKernel, Matern52Kernel, Kernel,
    GPPrediction,
)

# ---------------------------------------------------------------------------
# Query Strategy Enum
# ---------------------------------------------------------------------------

class QueryStrategy(Enum):
    """Active learning acquisition strategies."""
    UNCERTAINTY = "uncertainty"          # Max predictive variance
    ENTROPY = "entropy"                  # Max predictive entropy (classification)
    MARGIN = "margin"                    # Min margin between top two classes
    QBC = "qbc"                         # Query-by-committee disagreement
    EXPECTED_IMPROVEMENT = "ei"         # Max expected model improvement
    BALD = "bald"                       # Bayesian active learning by disagreement
    BATCH_GREEDY = "batch_greedy"       # Greedy batch selection (diversity + uncertainty)
    RANDOM = "random"                   # Random baseline


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ALResult:
    """Result of an active learning loop."""
    X_labeled: np.ndarray          # All labeled points (n_labeled, d)
    y_labeled: np.ndarray          # All labels (n_labeled,)
    query_indices: list            # Indices queried at each step
    model: GaussianProcess         # Final trained model
    scores_history: list           # Acquisition scores at each step
    model_performance: list        # Performance metric at each step
    n_queries: int                 # Total queries made


@dataclass
class BatchALResult:
    """Result of batch active learning."""
    X_labeled: np.ndarray
    y_labeled: np.ndarray
    batch_indices: list            # List of lists: indices per batch
    model: GaussianProcess
    model_performance: list
    n_batches: int
    total_queries: int


@dataclass
class StreamALResult:
    """Result of stream-based active learning."""
    X_labeled: np.ndarray
    y_labeled: np.ndarray
    n_queried: int                 # How many were queried
    n_seen: int                    # How many were seen
    query_rate: float              # Fraction queried
    model: GaussianProcess
    model_performance: list


@dataclass
class SynthesisResult:
    """Result of query synthesis active learning."""
    X_synthesized: np.ndarray      # Synthesized query points
    y_synthesized: np.ndarray      # Labels for synthesized points
    model: GaussianProcess
    model_performance: list
    n_queries: int


# ---------------------------------------------------------------------------
# Acquisition Functions (operate on GP predictions)
# ---------------------------------------------------------------------------

def uncertainty_sampling(gp: GaussianProcess, X_candidates: np.ndarray,
                         **kwargs) -> np.ndarray:
    """Score candidates by predictive variance (higher = more uncertain)."""
    pred = gp.predict(X_candidates, return_std=True)
    return pred.variance


def entropy_sampling(gp: GaussianProcess, X_candidates: np.ndarray,
                     **kwargs) -> np.ndarray:
    """Score by predictive entropy.

    For regression GP: entropy = 0.5 * log(2 * pi * e * variance).
    Higher entropy = more informative.
    """
    pred = gp.predict(X_candidates, return_std=True)
    # Differential entropy of Gaussian
    return 0.5 * np.log(2 * np.pi * np.e * np.maximum(pred.variance, 1e-12))


def margin_sampling(gp: GaussianProcess, X_candidates: np.ndarray,
                    threshold: float = 0.0, **kwargs) -> np.ndarray:
    """Score by proximity to decision boundary (for binary classification proxy).

    Uses |mean - threshold| as margin. Lower margin = more uncertain about class.
    Returns negative margin so argmax selects smallest margin.
    """
    pred = gp.predict(X_candidates, return_std=True)
    margin = np.abs(pred.mean - threshold)
    return -margin  # Negate so argmax picks smallest margin


def qbc_sampling(gp: GaussianProcess, X_candidates: np.ndarray,
                 n_committee: int = 5, rng: np.random.Generator = None,
                 **kwargs) -> np.ndarray:
    """Query-by-committee: score by disagreement among posterior samples.

    Draw posterior samples as committee members, score by variance of predictions.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw committee predictions from posterior
    samples = gp.sample_posterior(X_candidates, n_samples=n_committee, rng=rng)
    # Disagreement = variance across committee members
    return np.var(samples, axis=0)


def expected_model_change(gp: GaussianProcess, X_candidates: np.ndarray,
                          **kwargs) -> np.ndarray:
    """Score by expected model change -- how much the model would change
    if we observed the label at this point.

    Approximation: variance * gradient magnitude. Points where the model
    is uncertain AND in regions of high gradient are most informative.
    """
    pred = gp.predict(X_candidates, return_std=True)

    n = len(X_candidates)
    if n < 2:
        return pred.variance

    # Approximate gradient magnitude via finite differences on mean
    grad_mag = np.zeros(n)
    for i in range(n):
        diffs = X_candidates - X_candidates[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask = (dists > 0) & (dists < np.median(dists[dists > 0]) if np.any(dists > 0) else True)
        if np.any(mask):
            mean_diffs = np.abs(pred.mean[mask] - pred.mean[i])
            grad_mag[i] = np.mean(mean_diffs / np.maximum(dists[mask], 1e-12))

    return pred.variance * (1.0 + grad_mag)


def bald_sampling(gp: GaussianProcess, X_candidates: np.ndarray,
                  n_samples: int = 10, rng: np.random.Generator = None,
                  **kwargs) -> np.ndarray:
    """Bayesian Active Learning by Disagreement (BALD).

    Mutual information I(y; w | x, D) = H(y|x,D) - E_w[H(y|x,w)]
    For GPs: total entropy - noise entropy.
    Points with high BALD score reduce model uncertainty the most.
    """
    if rng is None:
        rng = np.random.default_rng()

    pred = gp.predict(X_candidates, return_std=True)

    # Total entropy H(y|x,D) -- uses full predictive variance
    total_entropy = 0.5 * np.log(2 * np.pi * np.e * np.maximum(pred.variance, 1e-12))

    # Expected conditional entropy E_w[H(y|x,w)] -- just noise
    noise_var = gp.noise_variance
    noise_entropy = 0.5 * np.log(2 * np.pi * np.e * max(noise_var, 1e-12))

    # BALD = mutual information
    return total_entropy - noise_entropy


def random_sampling(gp: GaussianProcess, X_candidates: np.ndarray,
                    rng: np.random.Generator = None, **kwargs) -> np.ndarray:
    """Random baseline -- uniform random scores."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.random(len(X_candidates))


# Strategy dispatch
_STRATEGY_MAP = {
    QueryStrategy.UNCERTAINTY: uncertainty_sampling,
    QueryStrategy.ENTROPY: entropy_sampling,
    QueryStrategy.MARGIN: margin_sampling,
    QueryStrategy.QBC: qbc_sampling,
    QueryStrategy.EXPECTED_IMPROVEMENT: expected_model_change,
    QueryStrategy.BALD: bald_sampling,
    QueryStrategy.RANDOM: random_sampling,
}


# ---------------------------------------------------------------------------
# Pool-Based Active Learning
# ---------------------------------------------------------------------------

def pool_based_active_learning(
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    n_queries: int = 20,
    strategy: QueryStrategy = QueryStrategy.UNCERTAINTY,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 1e-4,
    eval_func: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
    strategy_params: Optional[dict] = None,
) -> ALResult:
    """Pool-based active learning loop.

    Args:
        oracle: Function to query labels. oracle(X) -> y for (n, d) -> (n,)
        X_pool: Unlabeled pool (n_pool, d)
        X_initial: Initial labeled points (n_init, d)
        y_initial: Initial labels (n_init,)
        n_queries: Number of queries to make
        strategy: Query strategy to use
        kernel: GP kernel (default: Matern52)
        noise_variance: GP noise variance
        eval_func: Optional evaluation function(gp, X_pool) -> float
        rng: Random number generator
        strategy_params: Extra params for strategy function

    Returns:
        ALResult with labeled data, model, and history
    """
    if rng is None:
        rng = np.random.default_rng()
    if kernel is None:
        kernel = Matern52Kernel(length_scale=1.0, variance=1.0)
    if strategy_params is None:
        strategy_params = {}

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)

    # Pool as mutable list of indices
    pool_mask = np.ones(len(X_pool), dtype=bool)

    query_indices = []
    scores_history = []
    model_performance = []

    acq_fn = _STRATEGY_MAP.get(strategy, uncertainty_sampling)

    for step in range(n_queries):
        # Fit GP to current labeled set
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(X_labeled, y_labeled)

        # Evaluate model if eval_func provided
        if eval_func is not None:
            perf = eval_func(gp, X_pool)
            model_performance.append(perf)

        # Score remaining pool points
        available = np.where(pool_mask)[0]
        if len(available) == 0:
            break

        X_avail = X_pool[available]
        scores = acq_fn(gp, X_avail, rng=rng, **strategy_params)
        scores_history.append(scores.copy())

        # Select best point
        best_local = np.argmax(scores)
        best_pool_idx = available[best_local]

        # Query oracle
        x_query = X_pool[best_pool_idx:best_pool_idx+1]
        y_query = oracle(x_query)
        if np.ndim(y_query) > 0:
            y_query = y_query[0]

        # Add to labeled set
        X_labeled = np.vstack([X_labeled, x_query])
        y_labeled = np.append(y_labeled, y_query)

        # Remove from pool
        pool_mask[best_pool_idx] = False
        query_indices.append(int(best_pool_idx))

    # Final model
    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
    gp.fit(X_labeled, y_labeled)
    if eval_func is not None:
        model_performance.append(eval_func(gp, X_pool))

    return ALResult(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        query_indices=query_indices,
        model=gp,
        scores_history=scores_history,
        model_performance=model_performance,
        n_queries=len(query_indices),
    )


# ---------------------------------------------------------------------------
# Batch Active Learning
# ---------------------------------------------------------------------------

def batch_active_learning(
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    n_batches: int = 5,
    batch_size: int = 4,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 1e-4,
    eval_func: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
    diversity_weight: float = 0.5,
) -> BatchALResult:
    """Batch active learning with greedy diversity-aware selection.

    For each batch, greedily selects points that are both uncertain
    AND diverse (far from already selected points in the batch).

    Args:
        oracle: Label function oracle(X) -> y
        X_pool: Unlabeled pool
        X_initial: Initial labeled points
        y_initial: Initial labels
        n_batches: Number of batches
        batch_size: Points per batch
        kernel: GP kernel
        noise_variance: GP noise
        eval_func: Optional evaluation function
        rng: Random generator
        diversity_weight: Weight for diversity vs uncertainty (0=pure uncertainty, 1=pure diversity)
    """
    if rng is None:
        rng = np.random.default_rng()
    if kernel is None:
        kernel = Matern52Kernel(length_scale=1.0, variance=1.0)

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)
    pool_mask = np.ones(len(X_pool), dtype=bool)

    batch_indices = []
    model_performance = []

    for batch_idx in range(n_batches):
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(X_labeled, y_labeled)

        if eval_func is not None:
            model_performance.append(eval_func(gp, X_pool))

        available = np.where(pool_mask)[0]
        if len(available) == 0:
            break

        actual_batch_size = min(batch_size, len(available))
        X_avail = X_pool[available]

        # Get uncertainty scores
        pred = gp.predict(X_avail, return_std=True)
        unc_scores = pred.variance.copy()

        # Normalize uncertainty to [0, 1]
        unc_range = unc_scores.max() - unc_scores.min()
        if unc_range > 0:
            unc_norm = (unc_scores - unc_scores.min()) / unc_range
        else:
            unc_norm = np.ones_like(unc_scores)

        # Greedy batch selection
        batch_pool_indices = []
        selected_local = []

        for _ in range(actual_batch_size):
            if len(selected_local) == 0:
                # First point: pure uncertainty
                combined = unc_norm
            else:
                # Diversity: min distance to already selected batch points
                X_selected = X_avail[selected_local]
                dists = np.min(
                    np.sqrt(np.sum(
                        (X_avail[:, None, :] - X_selected[None, :, :]) ** 2,
                        axis=2
                    )),
                    axis=1
                )
                # Normalize diversity
                d_range = dists.max() - dists.min()
                if d_range > 0:
                    div_norm = (dists - dists.min()) / d_range
                else:
                    div_norm = np.ones_like(dists)

                combined = (1 - diversity_weight) * unc_norm + diversity_weight * div_norm

            # Mask already selected
            for s in selected_local:
                combined[s] = -np.inf

            best_local = np.argmax(combined)
            selected_local.append(best_local)
            batch_pool_indices.append(int(available[best_local]))

        # Query oracle for batch
        X_batch = X_pool[batch_pool_indices]
        y_batch = oracle(X_batch)

        # Update labeled set
        X_labeled = np.vstack([X_labeled, X_batch])
        y_labeled = np.concatenate([y_labeled, y_batch])

        # Update pool mask
        for idx in batch_pool_indices:
            pool_mask[idx] = False

        batch_indices.append(batch_pool_indices)

    # Final model
    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
    gp.fit(X_labeled, y_labeled)
    if eval_func is not None:
        model_performance.append(eval_func(gp, X_pool))

    return BatchALResult(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        batch_indices=batch_indices,
        model=gp,
        model_performance=model_performance,
        n_batches=len(batch_indices),
        total_queries=sum(len(b) for b in batch_indices),
    )


# ---------------------------------------------------------------------------
# Stream-Based Active Learning
# ---------------------------------------------------------------------------

def stream_active_learning(
    oracle: Callable,
    X_stream: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    threshold: float = 0.5,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 1e-4,
    eval_func: Optional[Callable] = None,
    budget: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> StreamALResult:
    """Stream-based active learning -- decide whether to query each point as it arrives.

    For each stream point, compute uncertainty. If above threshold, query oracle.
    Optionally enforce a budget constraint.

    Args:
        oracle: Label function
        X_stream: Stream of unlabeled points (n, d)
        X_initial: Initial labeled data
        y_initial: Initial labels
        threshold: Uncertainty threshold for querying (quantile of variance)
        kernel: GP kernel
        noise_variance: GP noise
        eval_func: Optional evaluation function
        budget: Max number of queries (None = unlimited)
        rng: Random generator
    """
    if rng is None:
        rng = np.random.default_rng()
    if kernel is None:
        kernel = Matern52Kernel(length_scale=1.0, variance=1.0)

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)

    n_queried = 0
    model_performance = []

    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
    gp.fit(X_labeled, y_labeled)

    # Compute initial variance scale for thresholding
    pred_init = gp.predict(X_stream[:min(50, len(X_stream))], return_std=True)
    var_threshold = np.quantile(pred_init.variance, threshold)

    for i in range(len(X_stream)):
        if budget is not None and n_queried >= budget:
            break

        x_i = X_stream[i:i+1]
        pred = gp.predict(x_i, return_std=True)

        if pred.variance[0] >= var_threshold:
            # Query this point
            y_i = oracle(x_i)
            if np.ndim(y_i) > 0:
                y_i = y_i[0]

            X_labeled = np.vstack([X_labeled, x_i])
            y_labeled = np.append(y_labeled, y_i)
            n_queried += 1

            # Refit periodically (every query to keep model current)
            gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
            gp.fit(X_labeled, y_labeled)

            # Update threshold based on current model's variance distribution
            if n_queried % 5 == 0 and i + 1 < len(X_stream):
                remaining = X_stream[i+1:min(i+51, len(X_stream))]
                if len(remaining) > 0:
                    pred_rem = gp.predict(remaining, return_std=True)
                    var_threshold = np.quantile(pred_rem.variance, threshold)

        if eval_func is not None and (i % max(1, len(X_stream) // 20) == 0):
            model_performance.append(eval_func(gp, X_stream))

    if eval_func is not None:
        model_performance.append(eval_func(gp, X_stream))

    return StreamALResult(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        n_queried=n_queried,
        n_seen=min(len(X_stream), i + 1) if budget is not None else len(X_stream),
        query_rate=n_queried / max(1, len(X_stream)),
        model=gp,
        model_performance=model_performance,
    )


# ---------------------------------------------------------------------------
# Query Synthesis Active Learning
# ---------------------------------------------------------------------------

def query_synthesis(
    oracle: Callable,
    bounds: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    n_queries: int = 20,
    n_candidates: int = 1000,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 1e-4,
    eval_func: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> SynthesisResult:
    """Query synthesis -- generate optimal query points from scratch.

    Instead of selecting from a pool, synthesizes new points in the input
    space that maximize information gain. Uses GP variance maximization
    over random candidate sets.

    Args:
        oracle: Label function
        bounds: Input space bounds (d, 2) where bounds[i] = [low, high]
        X_initial: Initial labeled data
        y_initial: Initial labels
        n_queries: Number of synthesized queries
        n_candidates: Number of random candidates to evaluate per step
        kernel: GP kernel
        noise_variance: GP noise
        eval_func: Optional evaluation function
        rng: Random generator
    """
    if rng is None:
        rng = np.random.default_rng()
    if kernel is None:
        kernel = Matern52Kernel(length_scale=1.0, variance=1.0)

    bounds = np.array(bounds)
    d = bounds.shape[0]

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)

    X_synthesized = []
    y_synthesized = []
    model_performance = []

    for step in range(n_queries):
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(X_labeled, y_labeled)

        if eval_func is not None:
            # Evaluate on a grid or random test set
            X_test = rng.uniform(bounds[:, 0], bounds[:, 1], size=(200, d))
            model_performance.append(eval_func(gp, X_test))

        # Generate random candidates
        X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, d))

        # Score by uncertainty
        pred = gp.predict(X_cand, return_std=True)
        best_idx = np.argmax(pred.variance)
        x_new = X_cand[best_idx:best_idx+1]

        # Query oracle
        y_new = oracle(x_new)
        if np.ndim(y_new) > 0:
            y_new = y_new[0]

        X_synthesized.append(x_new[0])
        y_synthesized.append(y_new)

        X_labeled = np.vstack([X_labeled, x_new])
        y_labeled = np.append(y_labeled, y_new)

    # Final model
    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
    gp.fit(X_labeled, y_labeled)
    if eval_func is not None:
        X_test = rng.uniform(bounds[:, 0], bounds[:, 1], size=(200, d))
        model_performance.append(eval_func(gp, X_test))

    return SynthesisResult(
        X_synthesized=np.array(X_synthesized),
        y_synthesized=np.array(y_synthesized),
        model=gp,
        model_performance=model_performance,
        n_queries=len(X_synthesized),
    )


# ---------------------------------------------------------------------------
# Strategy Comparison
# ---------------------------------------------------------------------------

def compare_strategies(
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    strategies: Optional[List[QueryStrategy]] = None,
    n_queries: int = 20,
    eval_func: Optional[Callable] = None,
    kernel: Optional[Kernel] = None,
    noise_variance: float = 1e-4,
    rng_seed: int = 42,
) -> Dict[str, ALResult]:
    """Compare multiple active learning strategies on the same problem.

    All strategies start from the same initial labeled set and use
    the same random seed for reproducibility.
    """
    if strategies is None:
        strategies = [QueryStrategy.UNCERTAINTY, QueryStrategy.ENTROPY,
                      QueryStrategy.BALD, QueryStrategy.RANDOM]

    results = {}
    for strategy in strategies:
        rng = np.random.default_rng(rng_seed)
        result = pool_based_active_learning(
            oracle=oracle,
            X_pool=X_pool,
            X_initial=X_initial.copy(),
            y_initial=y_initial.copy(),
            n_queries=n_queries,
            strategy=strategy,
            kernel=kernel,
            noise_variance=noise_variance,
            eval_func=eval_func,
            rng=rng,
        )
        results[strategy.value] = result

    return results


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def rmse_evaluation(gp: GaussianProcess, X_test: np.ndarray,
                    y_true: np.ndarray = None, oracle: Callable = None) -> float:
    """Evaluate GP model by RMSE on test set.

    Either y_true or oracle must be provided.
    """
    if y_true is None:
        if oracle is None:
            raise ValueError("Either y_true or oracle must be provided")
        y_true = oracle(X_test)

    pred = gp.predict(X_test, return_std=False)
    return float(np.sqrt(np.mean((pred.mean - y_true) ** 2)))


def nlpd_evaluation(gp: GaussianProcess, X_test: np.ndarray,
                    y_true: np.ndarray) -> float:
    """Negative log predictive density -- measures calibration.

    Lower is better. Penalizes both wrong predictions and overconfidence.
    """
    pred = gp.predict(X_test, return_std=True)
    var = np.maximum(pred.variance, 1e-12)
    nlpd = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y_true - pred.mean) ** 2 / var
    return float(np.mean(nlpd))


def coverage_evaluation(gp: GaussianProcess, X_test: np.ndarray,
                        y_true: np.ndarray, n_sigma: float = 2.0) -> float:
    """Coverage: fraction of test points within n_sigma confidence interval."""
    pred = gp.predict(X_test, return_std=True)
    lower = pred.mean - n_sigma * pred.std
    upper = pred.mean + n_sigma * pred.std
    covered = np.sum((y_true >= lower) & (y_true <= upper))
    return float(covered / len(y_true))


def make_rmse_evaluator(oracle: Callable, X_eval: np.ndarray) -> Callable:
    """Create an evaluation function that computes RMSE using a fixed test set."""
    y_eval = oracle(X_eval)

    def evaluator(gp: GaussianProcess, X_pool: np.ndarray) -> float:
        return rmse_evaluation(gp, X_eval, y_true=y_eval)

    return evaluator


# ---------------------------------------------------------------------------
# Active Learning Summary
# ---------------------------------------------------------------------------

def active_learning_summary(result, name: str = "AL") -> str:
    """Generate a human-readable summary of an active learning result."""
    lines = [f"=== {name} Summary ==="]

    if isinstance(result, ALResult):
        lines.append(f"Queries: {result.n_queries}")
        lines.append(f"Total labeled: {len(result.y_labeled)}")
        if result.model_performance:
            lines.append(f"Initial RMSE: {result.model_performance[0]:.4f}")
            lines.append(f"Final RMSE: {result.model_performance[-1]:.4f}")
            improvement = (result.model_performance[0] - result.model_performance[-1]) / max(result.model_performance[0], 1e-12) * 100
            lines.append(f"Improvement: {improvement:.1f}%")
    elif isinstance(result, BatchALResult):
        lines.append(f"Batches: {result.n_batches}")
        lines.append(f"Total queries: {result.total_queries}")
        lines.append(f"Total labeled: {len(result.y_labeled)}")
    elif isinstance(result, StreamALResult):
        lines.append(f"Queried: {result.n_queried} / {result.n_seen}")
        lines.append(f"Query rate: {result.query_rate:.2%}")
        lines.append(f"Total labeled: {len(result.y_labeled)}")
    elif isinstance(result, SynthesisResult):
        lines.append(f"Synthesized queries: {result.n_queries}")
        lines.append(f"Total labeled: {len(result.model.X_train)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark Functions
# ---------------------------------------------------------------------------

def sinusoidal_1d(X: np.ndarray) -> np.ndarray:
    """1D sinusoidal test function: sin(2*pi*x) + noise."""
    X = np.atleast_2d(X)
    return np.sin(2 * np.pi * X[:, 0])


def bumps_1d(X: np.ndarray) -> np.ndarray:
    """1D function with multiple bumps of varying width."""
    X = np.atleast_2d(X)
    x = X[:, 0]
    return np.sin(3 * x) + 0.5 * np.sin(9 * x) + 0.3 * np.cos(15 * x)


def step_function_1d(X: np.ndarray) -> np.ndarray:
    """1D step function -- hard for GPs, active learning should focus on boundaries."""
    X = np.atleast_2d(X)
    x = X[:, 0]
    return np.where(x < 0.3, -1.0, np.where(x < 0.7, 0.5, 1.0))


def friedman_2d(X: np.ndarray) -> np.ndarray:
    """2D Friedman-like function with interaction."""
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(np.pi * x1 * x2) + 2 * (x2 - 0.5) ** 2


def heteroscedastic_1d(X: np.ndarray) -> np.ndarray:
    """1D function with varying complexity -- flat region + wiggly region."""
    X = np.atleast_2d(X)
    x = X[:, 0]
    return np.where(x < 0.5, 0.2 * x, np.sin(10 * x) * x)
