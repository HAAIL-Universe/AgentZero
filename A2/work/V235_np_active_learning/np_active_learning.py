"""V235: Neural Process Active Learning

Composes V232 (Neural Processes) + V226 (Active Learning) for amortized,
meta-learned active learning. Instead of fitting a GP from scratch at each
query step, we use a Neural Process trained on a task distribution to provide
fast amortized predictions that guide query selection.

Key capabilities:
- NP-guided acquisition functions (uncertainty, entropy, BALD, EI)
- Pool-based AL with NP surrogate (no GP refitting per step)
- Meta-learned AL: train NP on task distribution, deploy on new tasks
- Batch AL with NP + determinantal diversity
- Adaptive model selection: switch NP variant based on context size
- Few-shot warm-start: leverage meta-learned priors with minimal labels
- Comparison: NP-AL vs GP-AL baselines
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from enum import Enum

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V232_neural_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V226_active_learning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))

from neural_process import (
    ConditionalNeuralProcess, NeuralProcess, AttentiveNeuralProcess,
    ConvCNP, GPNeuralProcess, NPPrediction, NPTrainResult,
    BasisEncoder, mean_aggregator
)
from active_learning import (
    ALResult, QueryStrategy, pool_based_active_learning,
    rmse_evaluation, make_rmse_evaluator,
    uncertainty_sampling, bald_sampling
)
from gaussian_process import GaussianProcess, RBFKernel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NPALResult:
    """Result of NP-guided active learning."""
    X_labeled: np.ndarray
    y_labeled: np.ndarray
    query_indices: list
    model_performance: list
    scores_history: list
    n_queries: int
    np_model_name: str


@dataclass
class MetaALResult:
    """Result of meta-learned active learning."""
    train_result: NPTrainResult
    al_results: List[NPALResult]
    mean_final_rmse: float
    mean_queries_to_threshold: Optional[float]


@dataclass
class AdaptiveALResult:
    """Result of adaptive model selection AL."""
    X_labeled: np.ndarray
    y_labeled: np.ndarray
    query_indices: list
    model_performance: list
    models_used: List[str]
    n_queries: int


@dataclass
class NPALComparison:
    """Comparison of NP-AL vs GP-AL."""
    model_names: List[str]
    final_rmses: Dict[str, List[float]]
    learning_curves: Dict[str, List[List[float]]]
    mean_final_rmse: Dict[str, float]
    queries_to_threshold: Dict[str, List[Optional[int]]]


# ---------------------------------------------------------------------------
# NP Acquisition Functions
# ---------------------------------------------------------------------------

class NPAcquisition(Enum):
    """Acquisition functions for NP-guided AL."""
    UNCERTAINTY = "uncertainty"
    ENTROPY = "entropy"
    BALD = "bald"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    RANDOM = "random"


def np_uncertainty(np_model, X_context, y_context, X_candidates):
    """Predictive variance from NP."""
    pred = np_model.predict(X_context, y_context, X_candidates)
    return pred.std ** 2


def np_entropy(np_model, X_context, y_context, X_candidates):
    """Differential entropy of NP predictive distribution."""
    pred = np_model.predict(X_context, y_context, X_candidates)
    variance = pred.std ** 2
    variance = np.maximum(variance, 1e-10)
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def np_bald(np_model, X_context, y_context, X_candidates, n_samples=10, rng=None):
    """BALD: mutual information between predictions and model parameters.

    For NeuralProcess with latent z, we sample multiple z values and
    measure disagreement. For CNP/ANP, we use variance as proxy.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if isinstance(np_model, NeuralProcess):
        # Sample from latent distribution
        pred = np_model.predict(X_context, y_context, X_candidates,
                                n_samples=n_samples, rng=rng)
        if pred.samples is not None and len(pred.samples) > 1:
            # Total entropy from mean prediction
            total_var = pred.std ** 2
            total_entropy = 0.5 * np.log(2 * np.pi * np.e * np.maximum(total_var, 1e-10))

            # Expected entropy per sample (noise entropy)
            # Estimate per-sample variance as small fraction of total
            noise_var = np.mean(np.var(pred.samples, axis=0)) * 0.1
            noise_entropy = 0.5 * np.log(2 * np.pi * np.e * max(noise_var, 1e-10))

            return total_entropy - noise_entropy
        else:
            return np_entropy(np_model, X_context, y_context, X_candidates)
    else:
        # For deterministic NPs, BALD degenerates to entropy
        return np_entropy(np_model, X_context, y_context, X_candidates)


def np_expected_improvement(np_model, X_context, y_context, X_candidates,
                            best_y=None, **kwargs):
    """Expected improvement acquisition for NP."""
    pred = np_model.predict(X_context, y_context, X_candidates)
    if best_y is None:
        best_y = np.min(y_context)

    mu = pred.mean
    sigma = np.maximum(pred.std, 1e-10)
    z = (best_y - mu) / sigma

    # EI = sigma * (z * Phi(z) + phi(z))
    # Using simple approximations
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + _erf_approx(z / np.sqrt(2)))
    ei = sigma * (z * Phi + phi)
    return ei


def np_random(np_model, X_context, y_context, X_candidates, rng=None):
    """Random acquisition (baseline)."""
    if rng is None:
        rng = np.random.RandomState()
    return rng.random(len(X_candidates))


def _erf_approx(x):
    """Approximate error function."""
    # Abramowitz and Stegun approximation
    a = 0.278393
    b = 0.230389
    c = 0.000972
    d = 0.078108
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (0.254829592 * t - 0.284496736 * t**2 +
                1.421413741 * t**3 - 1.453152027 * t**4 +
                1.061405429 * t**5) * np.exp(-x * x)
    return sign * y


_ACQUISITION_MAP = {
    NPAcquisition.UNCERTAINTY: lambda m, xc, yc, xp, **kw: np_uncertainty(m, xc, yc, xp),
    NPAcquisition.ENTROPY: lambda m, xc, yc, xp, **kw: np_entropy(m, xc, yc, xp),
    NPAcquisition.BALD: lambda m, xc, yc, xp, **kw: np_bald(m, xc, yc, xp, **kw),
    NPAcquisition.EXPECTED_IMPROVEMENT: lambda m, xc, yc, xp, **kw: np_expected_improvement(m, xc, yc, xp, **kw),
    NPAcquisition.RANDOM: lambda m, xc, yc, xp, **kw: np_random(m, xc, yc, xp, **kw),
}


# ---------------------------------------------------------------------------
# Pool-based NP Active Learning
# ---------------------------------------------------------------------------

def np_active_learning(
    np_model,
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    n_queries: int = 20,
    acquisition: NPAcquisition = NPAcquisition.UNCERTAINTY,
    eval_func: Optional[Callable] = None,
    rng=None,
    **acq_kwargs
) -> NPALResult:
    """Pool-based active learning using a Neural Process as surrogate.

    Unlike GP-based AL (V226), no model refitting per query -- the NP uses
    the growing context set directly for amortized prediction.

    Args:
        np_model: Trained Neural Process (CNP, NP, ANP, ConvCNP, or GPNP)
        oracle: Function f(X) -> y providing true labels
        X_pool: (n_pool, d) candidate points
        X_initial: (n_init, d) initial labeled points
        y_initial: (n_init,) initial labels
        n_queries: Number of queries to make
        acquisition: Acquisition function to use
        eval_func: Optional evaluation function(model_pred, X_test) -> score
        rng: Random state
        **acq_kwargs: Extra kwargs for acquisition function

    Returns:
        NPALResult with labeled data, query history, and performance
    """
    if rng is None:
        rng = np.random.RandomState(42)

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)
    available = list(range(len(X_pool)))
    query_indices = []
    scores_history = []
    model_performance = []

    acq_fn = _ACQUISITION_MAP[acquisition]

    # Evaluate initial performance
    if eval_func is not None:
        perf = eval_func(np_model, X_labeled, y_labeled)
        model_performance.append(perf)

    for step in range(n_queries):
        if not available:
            break

        X_cand = X_pool[available]

        # Compute acquisition scores using NP predictions
        scores = acq_fn(np_model, X_labeled, y_labeled, X_cand,
                        rng=rng, **acq_kwargs)
        scores_history.append(scores.copy())

        # Select best candidate
        best_local = np.argmax(scores)
        best_pool = available[best_local]
        query_indices.append(best_pool)
        available.remove(best_pool)

        # Query oracle and add to labeled set
        x_new = X_pool[best_pool:best_pool+1]
        y_new = oracle(x_new)
        if np.ndim(y_new) > 0:
            y_new = y_new.ravel()

        X_labeled = np.vstack([X_labeled, x_new])
        y_labeled = np.concatenate([y_labeled, y_new])

        # Evaluate performance (NP predicts using growing context)
        if eval_func is not None:
            perf = eval_func(np_model, X_labeled, y_labeled)
            model_performance.append(perf)

    model_name = type(np_model).__name__
    return NPALResult(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        query_indices=query_indices,
        model_performance=model_performance,
        scores_history=scores_history,
        n_queries=len(query_indices),
        np_model_name=model_name
    )


# ---------------------------------------------------------------------------
# Batch NP Active Learning
# ---------------------------------------------------------------------------

def np_batch_active_learning(
    np_model,
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    n_batches: int = 5,
    batch_size: int = 4,
    acquisition: NPAcquisition = NPAcquisition.UNCERTAINTY,
    diversity_weight: float = 0.5,
    eval_func: Optional[Callable] = None,
    rng=None,
    **acq_kwargs
) -> NPALResult:
    """Batch active learning with NP surrogate + diversity.

    Selects batches by greedily picking points that are both informative
    (high acquisition score) and diverse (far from already-selected batch points).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)
    available = list(range(len(X_pool)))
    all_query_indices = []
    scores_history = []
    model_performance = []

    acq_fn = _ACQUISITION_MAP[acquisition]

    if eval_func is not None:
        perf = eval_func(np_model, X_labeled, y_labeled)
        model_performance.append(perf)

    for batch_i in range(n_batches):
        if not available:
            break

        X_cand = X_pool[available]
        base_scores = acq_fn(np_model, X_labeled, y_labeled, X_cand,
                             rng=rng, **acq_kwargs)

        # Normalize scores to [0, 1]
        s_min, s_max = base_scores.min(), base_scores.max()
        if s_max > s_min:
            norm_scores = (base_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(base_scores)

        batch_indices = []
        batch_selected_local = []

        for k in range(min(batch_size, len(available))):
            if k == 0:
                combined = norm_scores
            else:
                # Compute diversity: min distance to already-selected batch points
                selected_X = X_cand[batch_selected_local]
                dists = np.array([
                    np.min(np.sum((X_cand - sx) ** 2, axis=-1) + 1e-10)
                    for sx in selected_X
                ])
                # Actually compute min dist from each candidate to batch
                dist_matrix = np.zeros(len(X_cand))
                for sx in selected_X:
                    d = np.sum((X_cand - sx) ** 2, axis=-1)
                    dist_matrix = np.maximum(dist_matrix, 0)
                    # We want min distance to any selected point
                    if len(batch_selected_local) == 1:
                        dist_matrix = d
                    else:
                        dist_matrix = np.minimum(dist_matrix, d)

                d_min, d_max = dist_matrix.min(), dist_matrix.max()
                if d_max > d_min:
                    div_scores = (dist_matrix - d_min) / (d_max - d_min)
                else:
                    div_scores = np.ones_like(dist_matrix)

                combined = (1 - diversity_weight) * norm_scores + diversity_weight * div_scores

            # Mask already selected
            for idx in batch_selected_local:
                combined[idx] = -np.inf

            best_local = np.argmax(combined)
            batch_selected_local.append(best_local)
            batch_indices.append(available[best_local])

        scores_history.append(base_scores.copy())

        # Query oracle for entire batch
        for pool_idx in batch_indices:
            x_new = X_pool[pool_idx:pool_idx+1]
            y_new = oracle(x_new)
            if np.ndim(y_new) > 0:
                y_new = y_new.ravel()
            X_labeled = np.vstack([X_labeled, x_new])
            y_labeled = np.concatenate([y_labeled, y_new])
            available.remove(pool_idx)

        all_query_indices.extend(batch_indices)

        if eval_func is not None:
            perf = eval_func(np_model, X_labeled, y_labeled)
            model_performance.append(perf)

    return NPALResult(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        query_indices=all_query_indices,
        model_performance=model_performance,
        scores_history=scores_history,
        n_queries=len(all_query_indices),
        np_model_name=type(np_model).__name__
    )


# ---------------------------------------------------------------------------
# Meta-Learned Active Learning
# ---------------------------------------------------------------------------

def meta_active_learning(
    task_distribution,
    n_train_epochs: int = 50,
    np_model_class=None,
    np_kwargs: Optional[dict] = None,
    n_test_tasks: int = 5,
    n_queries: int = 15,
    n_initial: int = 3,
    acquisition: NPAcquisition = NPAcquisition.UNCERTAINTY,
    eval_X: Optional[np.ndarray] = None,
    rng=None,
    **acq_kwargs
) -> MetaALResult:
    """Meta-learned active learning: train NP on task distribution, then deploy.

    Phase 1: Meta-train NP on tasks from the distribution
    Phase 2: For each test task, run NP-guided AL with the trained model

    Args:
        task_distribution: Object with .sample_task(rng) method returning
                           Task(X_context, y_context, X_target, y_target)
        n_train_epochs: NP training epochs
        np_model_class: NP class to use (default: ConditionalNeuralProcess)
        np_kwargs: Constructor kwargs for NP
        n_test_tasks: Number of test tasks for evaluation
        n_queries: Queries per test task
        n_initial: Initial labeled points per test task
        acquisition: Acquisition function
        eval_X: Optional fixed evaluation grid
        rng: Random state

    Returns:
        MetaALResult with training result and per-task AL results
    """
    if rng is None:
        rng = np.random.RandomState(42)
    if np_model_class is None:
        np_model_class = ConditionalNeuralProcess
    if np_kwargs is None:
        np_kwargs = {}

    # Phase 1: Meta-train
    model = np_model_class(**np_kwargs)
    train_result = model.train(task_distribution, n_epochs=n_train_epochs,
                               lr=0.01, rng=rng)

    # Phase 2: Deploy on test tasks
    # Use last n_test_tasks from the distribution
    all_tasks = task_distribution.tasks
    test_tasks = all_tasks[-n_test_tasks:]

    al_results = []
    for t, task in enumerate(test_tasks):
        # Build pool and initial set from task data
        all_X = np.concatenate([task.X_support, task.X_query])
        all_y = np.concatenate([task.y_support, task.y_query])

        # Shuffle
        perm = rng.permutation(len(all_X))
        all_X = all_X[perm]
        all_y = all_y[perm]

        X_init = all_X[:n_initial]
        y_init = all_y[:n_initial]
        X_pool = all_X[n_initial:]

        # Evaluation: RMSE on eval_X or on target points
        if eval_X is not None:
            eval_points = eval_X
        else:
            eval_points = task.X_query

        def oracle(X):
            # Interpolate from all data
            # For simplicity, use nearest neighbor from task data
            dists = np.sum((all_X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
            nearest = np.argmin(dists, axis=0)
            return all_y[nearest]

        def eval_fn(np_m, X_ctx, y_ctx):
            pred = np_m.predict(X_ctx, y_ctx, eval_points)
            # Need true values at eval points
            dists = np.sum((all_X[:, None, :] - eval_points[None, :, :]) ** 2, axis=-1)
            nearest = np.argmin(dists, axis=0)
            y_true = all_y[nearest]
            return np.sqrt(np.mean((pred.mean - y_true) ** 2))

        actual_queries = min(n_queries, len(X_pool))
        result = np_active_learning(
            model, oracle, X_pool, X_init, y_init,
            n_queries=actual_queries, acquisition=acquisition,
            eval_func=eval_fn, rng=rng, **acq_kwargs
        )
        al_results.append(result)

    # Aggregate results
    final_rmses = [r.model_performance[-1] for r in al_results if r.model_performance]
    mean_rmse = np.mean(final_rmses) if final_rmses else float('inf')

    return MetaALResult(
        train_result=train_result,
        al_results=al_results,
        mean_final_rmse=mean_rmse,
        mean_queries_to_threshold=None
    )


# ---------------------------------------------------------------------------
# Adaptive Model Selection AL
# ---------------------------------------------------------------------------

def adaptive_np_active_learning(
    np_models: Dict[str, Any],
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    n_queries: int = 20,
    switch_points: Optional[List[int]] = None,
    acquisition: NPAcquisition = NPAcquisition.UNCERTAINTY,
    eval_func: Optional[Callable] = None,
    rng=None,
    **acq_kwargs
) -> AdaptiveALResult:
    """Active learning with adaptive NP model selection.

    Different NP variants work better at different context sizes:
    - CNP: fast, good with small context
    - ANP: better with medium context (attention helps)
    - NP: best with larger context (latent variable captures global structure)

    Switches between models based on current context size.

    Args:
        np_models: Dict mapping name -> trained NP model
        oracle: Label function
        X_pool: Candidate pool
        X_initial, y_initial: Initial labeled set
        n_queries: Total queries
        switch_points: Context sizes at which to switch models.
                       Default: switch at n=5 and n=15
        acquisition: Acquisition function
        eval_func: Evaluation function(np_model, X_ctx, y_ctx) -> score
        rng: Random state

    Returns:
        AdaptiveALResult with model usage history
    """
    if rng is None:
        rng = np.random.RandomState(42)

    model_names = list(np_models.keys())
    if switch_points is None:
        # Default: first model for n<5, second for 5<=n<15, third for n>=15
        n_models = len(model_names)
        if n_models == 1:
            switch_points = []
        elif n_models == 2:
            switch_points = [8]
        else:
            switch_points = [5, 15]

    def _select_model(n_context):
        """Select model based on context size."""
        idx = 0
        for sp in switch_points:
            if n_context >= sp and idx + 1 < len(model_names):
                idx += 1
        return model_names[idx], np_models[model_names[idx]]

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)
    available = list(range(len(X_pool)))
    query_indices = []
    model_performance = []
    models_used = []

    acq_fn = _ACQUISITION_MAP[acquisition]

    if eval_func is not None:
        name, model = _select_model(len(X_labeled))
        perf = eval_func(model, X_labeled, y_labeled)
        model_performance.append(perf)

    for step in range(n_queries):
        if not available:
            break

        name, model = _select_model(len(X_labeled))
        models_used.append(name)

        X_cand = X_pool[available]
        scores = acq_fn(model, X_labeled, y_labeled, X_cand, rng=rng, **acq_kwargs)

        best_local = np.argmax(scores)
        best_pool = available[best_local]
        query_indices.append(best_pool)
        available.remove(best_pool)

        x_new = X_pool[best_pool:best_pool+1]
        y_new = oracle(x_new)
        if np.ndim(y_new) > 0:
            y_new = y_new.ravel()

        X_labeled = np.vstack([X_labeled, x_new])
        y_labeled = np.concatenate([y_labeled, y_new])

        if eval_func is not None:
            _, cur_model = _select_model(len(X_labeled))
            perf = eval_func(cur_model, X_labeled, y_labeled)
            model_performance.append(perf)

    return AdaptiveALResult(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        query_indices=query_indices,
        model_performance=model_performance,
        models_used=models_used,
        n_queries=len(query_indices)
    )


# ---------------------------------------------------------------------------
# NP-AL vs GP-AL Comparison
# ---------------------------------------------------------------------------

def compare_np_vs_gp_al(
    np_models: Dict[str, Any],
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_queries: int = 20,
    n_repeats: int = 3,
    rng_seed: int = 42,
) -> NPALComparison:
    """Compare NP-guided AL against GP-based AL (V226 baseline).

    Runs each model multiple times with different seeds and reports
    learning curves and final RMSE.
    """
    model_names = list(np_models.keys()) + ["GP-AL"]
    final_rmses = {name: [] for name in model_names}
    learning_curves = {name: [] for name in model_names}
    queries_to_threshold = {name: [] for name in model_names}

    for rep in range(n_repeats):
        seed = rng_seed + rep
        rng = np.random.RandomState(seed)

        # --- NP models ---
        for name, np_model in np_models.items():
            def np_eval(m, X_ctx, y_ctx, _np=np_model, _Xe=X_eval, _ye=y_eval):
                pred = _np.predict(X_ctx, y_ctx, _Xe)
                return np.sqrt(np.mean((pred.mean - _ye) ** 2))

            result = np_active_learning(
                np_model, oracle, X_pool, X_initial, y_initial,
                n_queries=n_queries,
                acquisition=NPAcquisition.UNCERTAINTY,
                eval_func=np_eval,
                rng=np.random.RandomState(seed)
            )
            final_rmses[name].append(result.model_performance[-1] if result.model_performance else float('inf'))
            learning_curves[name].append(result.model_performance)

        # --- GP baseline ---
        def gp_eval_fn(gp, X_ignored, X_test=X_eval, y_true=y_eval):
            pred = gp.predict(X_test)
            return np.sqrt(np.mean((pred.mean - y_true) ** 2))

        gp_result = pool_based_active_learning(
            oracle, X_pool, X_initial, y_initial,
            n_queries=n_queries,
            strategy=QueryStrategy.UNCERTAINTY,
            eval_func=gp_eval_fn,
            rng=np.random.RandomState(seed)
        )
        gp_name = "GP-AL"
        final_rmses[gp_name].append(
            gp_result.model_performance[-1] if gp_result.model_performance else float('inf')
        )
        learning_curves[gp_name].append(gp_result.model_performance)

    mean_final = {name: float(np.mean(vals)) for name, vals in final_rmses.items()}

    return NPALComparison(
        model_names=model_names,
        final_rmses=final_rmses,
        learning_curves=learning_curves,
        mean_final_rmse=mean_final,
        queries_to_threshold=queries_to_threshold
    )


# ---------------------------------------------------------------------------
# Few-Shot Warm-Start AL
# ---------------------------------------------------------------------------

def few_shot_warm_start(
    np_model,
    oracle: Callable,
    X_pool: np.ndarray,
    n_initial: int = 2,
    n_queries: int = 15,
    acquisition: NPAcquisition = NPAcquisition.UNCERTAINTY,
    eval_func: Optional[Callable] = None,
    rng=None,
    **acq_kwargs
) -> NPALResult:
    """Few-shot warm-start: begin AL with very few labels using NP priors.

    The meta-trained NP can make reasonable predictions even with 1-2 context
    points, enabling effective query selection from the start. A GP would
    have nearly uniform uncertainty with so few points.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Select initial points via max-spread
    indices = list(range(len(X_pool)))
    init_indices = _maxmin_select(X_pool, n_initial, rng)

    X_init = X_pool[init_indices]
    y_init = oracle(X_init)
    if np.ndim(y_init) > 0:
        y_init = y_init.ravel()

    # Remove initial points from pool
    remaining_indices = [i for i in indices if i not in init_indices]
    X_remaining = X_pool[remaining_indices]

    # Map from local index back to pool index
    result = np_active_learning(
        np_model, oracle, X_remaining, X_init, y_init,
        n_queries=n_queries, acquisition=acquisition,
        eval_func=eval_func, rng=rng, **acq_kwargs
    )

    # Remap query indices to original pool indices
    remapped = [remaining_indices[qi] for qi in result.query_indices]
    result.query_indices = list(init_indices) + remapped

    return result


def _maxmin_select(X, n, rng):
    """Select n points from X with maximum minimum distance (greedy)."""
    n_pool = len(X)
    if n >= n_pool:
        return list(range(n_pool))

    selected = [rng.randint(n_pool)]
    for _ in range(n - 1):
        dists = np.min([
            np.sum((X - X[s]) ** 2, axis=-1) for s in selected
        ], axis=0)
        # Don't re-select
        for s in selected:
            dists[s] = -1
        selected.append(int(np.argmax(dists)))
    return selected


# ---------------------------------------------------------------------------
# Query Budget Allocation
# ---------------------------------------------------------------------------

def budget_allocated_al(
    np_models: Dict[str, Any],
    oracle: Callable,
    X_pool: np.ndarray,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    total_budget: int = 20,
    eval_func: Optional[Callable] = None,
    rng=None,
) -> NPALResult:
    """Budget-allocated AL: distribute query budget across NP models.

    Each model proposes queries; we pick the one with highest acquisition
    score across all models at each step. This lets the best model for the
    current context automatically dominate.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    X_labeled = np.array(X_initial, dtype=float)
    y_labeled = np.array(y_initial, dtype=float)
    available = list(range(len(X_pool)))
    query_indices = []
    scores_history = []
    model_performance = []

    if eval_func is not None:
        # Use first model for initial eval
        first_model = list(np_models.values())[0]
        perf = eval_func(first_model, X_labeled, y_labeled)
        model_performance.append(perf)

    for step in range(total_budget):
        if not available:
            break

        X_cand = X_pool[available]

        # Each model votes
        best_score = -np.inf
        best_pool_idx = None
        best_model_name = None

        for name, model in np_models.items():
            scores = np_uncertainty(model, X_labeled, y_labeled, X_cand)
            local_best = np.argmax(scores)
            if scores[local_best] > best_score:
                best_score = scores[local_best]
                best_pool_idx = available[local_best]
                best_model_name = name

        query_indices.append(best_pool_idx)
        available.remove(best_pool_idx)
        scores_history.append(best_score)

        x_new = X_pool[best_pool_idx:best_pool_idx+1]
        y_new = oracle(x_new)
        if np.ndim(y_new) > 0:
            y_new = y_new.ravel()
        X_labeled = np.vstack([X_labeled, x_new])
        y_labeled = np.concatenate([y_labeled, y_new])

        if eval_func is not None:
            perf = eval_func(np_models[best_model_name], X_labeled, y_labeled)
            model_performance.append(perf)

    return NPALResult(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        query_indices=query_indices,
        model_performance=model_performance,
        scores_history=scores_history,
        n_queries=len(query_indices),
        np_model_name="budget_ensemble"
    )


# ---------------------------------------------------------------------------
# Information-Theoretic Query Selection
# ---------------------------------------------------------------------------

def np_information_gain(np_model, X_context, y_context, X_candidates,
                        n_fantasy=5, rng=None):
    """Information gain: expected reduction in posterior entropy after query.

    For each candidate x*, fantasize y* values and measure how much the
    NP's predictive uncertainty decreases on average.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    pred_before = np_model.predict(X_context, y_context, X_candidates)
    var_before = pred_before.std ** 2

    info_gains = np.zeros(len(X_candidates))

    for i in range(len(X_candidates)):
        x_star = X_candidates[i:i+1]

        # Fantasy labels from current predictive distribution
        mu_i = pred_before.mean[i]
        std_i = pred_before.std[i]
        fantasies = rng.normal(mu_i, max(std_i, 1e-6), size=n_fantasy)

        avg_var_after = 0.0
        for y_f in fantasies:
            X_aug = np.vstack([X_context, x_star])
            y_aug = np.concatenate([y_context, [y_f]])
            pred_after = np_model.predict(X_aug, y_aug, X_candidates)
            avg_var_after += np.mean(pred_after.std ** 2)
        avg_var_after /= n_fantasy

        # Information gain = variance reduction
        info_gains[i] = np.mean(var_before) - avg_var_after

    return info_gains


# ---------------------------------------------------------------------------
# Learning Curve Analysis
# ---------------------------------------------------------------------------

def np_al_learning_curve(
    np_model,
    oracle: Callable,
    X_pool: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_queries: int = 20,
    n_initial: int = 3,
    acquisition: NPAcquisition = NPAcquisition.UNCERTAINTY,
    rng=None,
) -> Dict[str, Any]:
    """Run NP-AL and return detailed learning curve.

    Returns dict with:
    - n_labeled: list of context sizes
    - rmse: RMSE at each step
    - mean_std: average predictive std at each step
    - query_efficiency: RMSE improvement per query
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Initial selection
    init_idx = _maxmin_select(X_pool, n_initial, rng)
    X_init = X_pool[init_idx]
    y_init = oracle(X_init)
    if np.ndim(y_init) > 0:
        y_init = y_init.ravel()

    remaining = [i for i in range(len(X_pool)) if i not in init_idx]
    X_remaining = X_pool[remaining]

    n_labeled_history = []
    rmse_history = []
    mean_std_history = []

    def eval_fn(m, X_ctx, y_ctx):
        pred = m.predict(X_ctx, y_ctx, X_eval)
        rmse = np.sqrt(np.mean((pred.mean - y_eval) ** 2))
        mean_std = np.mean(pred.std)
        n_labeled_history.append(len(X_ctx))
        rmse_history.append(rmse)
        mean_std_history.append(mean_std)
        return rmse

    np_active_learning(
        np_model, oracle, X_remaining, X_init, y_init,
        n_queries=min(n_queries, len(X_remaining)),
        acquisition=acquisition,
        eval_func=eval_fn,
        rng=rng
    )

    # Query efficiency: RMSE improvement per query
    efficiency = []
    for i in range(1, len(rmse_history)):
        efficiency.append(rmse_history[i-1] - rmse_history[i])

    return {
        'n_labeled': n_labeled_history,
        'rmse': rmse_history,
        'mean_std': mean_std_history,
        'query_efficiency': efficiency,
    }


# ---------------------------------------------------------------------------
# Summary / Formatting
# ---------------------------------------------------------------------------

def npal_summary(result: NPALResult, name: str = "NP-AL") -> str:
    """Format NP-AL result as human-readable text."""
    lines = [f"=== {name} ({result.np_model_name}) ==="]
    lines.append(f"Queries: {result.n_queries}")
    lines.append(f"Final labeled set: {len(result.X_labeled)} points")
    if result.model_performance:
        lines.append(f"Initial RMSE: {result.model_performance[0]:.4f}")
        lines.append(f"Final RMSE:   {result.model_performance[-1]:.4f}")
        improvement = result.model_performance[0] - result.model_performance[-1]
        lines.append(f"Improvement:  {improvement:.4f}")
    return "\n".join(lines)


def comparison_summary(result: NPALComparison) -> str:
    """Format comparison result as table."""
    lines = ["=== NP-AL vs GP-AL Comparison ==="]
    lines.append(f"{'Model':<20} {'Mean RMSE':>12}")
    lines.append("-" * 34)
    # Sort by mean RMSE
    sorted_models = sorted(result.mean_final_rmse.items(), key=lambda x: x[1])
    for name, rmse in sorted_models:
        lines.append(f"{name:<20} {rmse:>12.4f}")
    return "\n".join(lines)
