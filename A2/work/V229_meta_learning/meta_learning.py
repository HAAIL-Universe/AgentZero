"""V229: Meta-Learning -- Learning to learn across task distributions.

Composes V226 (Active Learning) + V222 (Gaussian Process) to build systems
that transfer knowledge across related tasks for few-shot learning.

Core ideas:
  - Meta-learn GP hyperparameters across a distribution of tasks
  - Task-adaptive kernels that share structure across tasks
  - Few-shot regression/classification via task embeddings
  - Meta-active-learning: learn which queries generalize across tasks
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V226_active_learning'))

from gaussian_process import (
    GaussianProcess, RBFKernel, Matern52Kernel, Kernel, GPPrediction,
    SumKernel, ProductKernel, ScaleKernel, WhiteNoiseKernel, ARDKernel,
)
from active_learning import (
    pool_based_active_learning, QueryStrategy, ALResult,
    uncertainty_sampling, make_rmse_evaluator,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single learning task with support and query sets."""
    X_support: np.ndarray    # (n_support, d) -- training points
    y_support: np.ndarray    # (n_support,)
    X_query: np.ndarray      # (n_query, d) -- test points
    y_query: np.ndarray      # (n_query,)
    task_id: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskDistribution:
    """A distribution of related tasks for meta-learning."""
    tasks: List[Task]
    name: str = "unnamed"
    input_dim: int = 1

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def train_test_split(self, n_test: int = 5,
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple['TaskDistribution', 'TaskDistribution']:
        """Split into meta-train and meta-test distributions."""
        if rng is None:
            rng = np.random.default_rng(42)
        indices = rng.permutation(self.n_tasks)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        train_tasks = [self.tasks[i] for i in train_idx]
        test_tasks = [self.tasks[i] for i in test_idx]
        return (
            TaskDistribution(train_tasks, f"{self.name}_train", self.input_dim),
            TaskDistribution(test_tasks, f"{self.name}_test", self.input_dim),
        )


@dataclass
class MetaLearningResult:
    """Result of meta-learning across tasks."""
    meta_kernel: Kernel              # Learned meta-kernel
    meta_noise: float                # Learned noise variance
    train_losses: List[float]        # Per-epoch meta-train loss
    val_losses: List[float]          # Per-epoch meta-val loss (if available)
    task_performances: List[float]   # Per-task RMSE on query sets
    n_tasks_seen: int
    n_epochs: int


@dataclass
class FewShotResult:
    """Result of few-shot prediction on a new task."""
    predictions: np.ndarray     # (n_query,) predicted values
    uncertainty: np.ndarray     # (n_query,) predictive std
    rmse: float                 # RMSE on query set
    nlpd: float                 # Negative log predictive density
    model: GaussianProcess      # Fitted model
    n_support: int              # Number of support examples used


@dataclass
class MetaALResult:
    """Result of meta-active-learning."""
    meta_kernel: Kernel
    meta_noise: float
    strategy_scores: Dict[str, float]   # Strategy -> avg performance
    best_strategy: str
    per_task_results: List[ALResult]
    n_tasks: int


@dataclass
class TaskEmbedding:
    """Embedding of a task in a metric space."""
    task_id: int
    embedding: np.ndarray       # (embed_dim,)
    kernel_params: np.ndarray   # Fitted kernel params for this task


# ---------------------------------------------------------------------------
# Task Generators (Benchmark Distributions)
# ---------------------------------------------------------------------------

def sinusoidal_task_distribution(n_tasks: int = 20, n_support: int = 5,
                                  n_query: int = 50,
                                  rng: Optional[np.random.Generator] = None
                                  ) -> TaskDistribution:
    """Generate sinusoidal regression tasks with varying amplitude/phase/freq.

    Each task: y = A * sin(omega * x + phi) + noise
    A ~ U[0.5, 5], omega ~ U[0.5, 2], phi ~ U[0, 2*pi]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    tasks = []
    for i in range(n_tasks):
        A = rng.uniform(0.5, 5.0)
        omega = rng.uniform(0.5, 2.0)
        phi = rng.uniform(0, 2 * np.pi)
        noise_std = 0.1

        X_all = rng.uniform(-5, 5, size=(n_support + n_query, 1))
        y_all = A * np.sin(omega * X_all[:, 0] + phi) + rng.normal(0, noise_std, n_support + n_query)

        tasks.append(Task(
            X_support=X_all[:n_support],
            y_support=y_all[:n_support],
            X_query=X_all[n_support:],
            y_query=y_all[n_support:],
            task_id=i,
            metadata={'A': A, 'omega': omega, 'phi': phi},
        ))
    return TaskDistribution(tasks, "sinusoidal", input_dim=1)


def polynomial_task_distribution(n_tasks: int = 20, n_support: int = 5,
                                  n_query: int = 50, max_degree: int = 3,
                                  rng: Optional[np.random.Generator] = None
                                  ) -> TaskDistribution:
    """Generate polynomial regression tasks with varying coefficients.

    Each task: y = sum(c_k * x^k for k in 0..degree) + noise
    Coefficients ~ N(0, 1/(k+1)), degree ~ U{1, max_degree}
    """
    if rng is None:
        rng = np.random.default_rng(42)
    tasks = []
    for i in range(n_tasks):
        degree = rng.integers(1, max_degree + 1)
        coeffs = rng.normal(0, 1.0 / np.arange(1, degree + 2), size=degree + 1)
        noise_std = 0.1

        X_all = rng.uniform(-2, 2, size=(n_support + n_query, 1))
        x = X_all[:, 0]
        y_all = sum(coeffs[k] * x**k for k in range(degree + 1))
        y_all = y_all + rng.normal(0, noise_std, len(y_all))

        tasks.append(Task(
            X_support=X_all[:n_support],
            y_support=y_all[:n_support],
            X_query=X_all[n_support:],
            y_query=y_all[n_support:],
            task_id=i,
            metadata={'degree': degree, 'coeffs': coeffs.tolist()},
        ))
    return TaskDistribution(tasks, "polynomial", input_dim=1)


def step_task_distribution(n_tasks: int = 20, n_support: int = 5,
                           n_query: int = 50,
                           rng: Optional[np.random.Generator] = None
                           ) -> TaskDistribution:
    """Generate step function tasks with varying thresholds/heights.

    Each task: y = h1 if x < t else h2 (+ noise)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    tasks = []
    for i in range(n_tasks):
        threshold = rng.uniform(-2, 2)
        h1 = rng.uniform(-3, 3)
        h2 = rng.uniform(-3, 3)
        noise_std = 0.15

        X_all = rng.uniform(-5, 5, size=(n_support + n_query, 1))
        y_all = np.where(X_all[:, 0] < threshold, h1, h2)
        y_all = y_all + rng.normal(0, noise_std, len(y_all))

        tasks.append(Task(
            X_support=X_all[:n_support],
            y_support=y_all[:n_support],
            X_query=X_all[n_support:],
            y_query=y_all[n_support:],
            task_id=i,
            metadata={'threshold': threshold, 'h1': h1, 'h2': h2},
        ))
    return TaskDistribution(tasks, "step", input_dim=1)


def multidim_task_distribution(n_tasks: int = 20, n_support: int = 10,
                                n_query: int = 50, input_dim: int = 2,
                                rng: Optional[np.random.Generator] = None
                                ) -> TaskDistribution:
    """Generate multi-dimensional regression tasks.

    Each task: y = w^T x + b + noise, w ~ N(0, I), b ~ N(0, 1)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    tasks = []
    for i in range(n_tasks):
        w = rng.normal(0, 1, size=input_dim)
        b = rng.normal(0, 1)
        noise_std = 0.1

        X_all = rng.uniform(-2, 2, size=(n_support + n_query, input_dim))
        y_all = X_all @ w + b + rng.normal(0, noise_std, n_support + n_query)

        tasks.append(Task(
            X_support=X_all[:n_support],
            y_support=y_all[:n_support],
            X_query=X_all[n_support:],
            y_query=y_all[n_support:],
            task_id=i,
            metadata={'w': w.tolist(), 'b': b},
        ))
    return TaskDistribution(tasks, "multidim_linear", input_dim=input_dim)


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _nlpd(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Negative log predictive density (Gaussian)."""
    var = std ** 2 + 1e-10
    return float(0.5 * np.mean(np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var))


def evaluate_on_task(gp: GaussianProcess, task: Task) -> Tuple[float, float]:
    """Evaluate a GP on a task's query set. Returns (rmse, nlpd)."""
    pred = gp.predict(task.X_query, return_std=True)
    rmse = _rmse(task.y_query, pred.mean)
    nlpd = _nlpd(task.y_query, pred.mean, pred.std)
    return rmse, nlpd


def fit_task(kernel: Kernel, noise_variance: float,
             task: Task) -> GaussianProcess:
    """Fit a GP to a task's support set."""
    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
    gp.fit(task.X_support, task.y_support)
    return gp


# ---------------------------------------------------------------------------
# Meta-Learning: Kernel Transfer
# ---------------------------------------------------------------------------

def meta_learn_kernel(task_dist: TaskDistribution,
                      base_kernel: Optional[Kernel] = None,
                      n_epochs: int = 5,
                      noise_variance: float = 0.01,
                      learning_rate: float = 0.1,
                      val_fraction: float = 0.2,
                      rng: Optional[np.random.Generator] = None,
                      ) -> MetaLearningResult:
    """Meta-learn GP kernel hyperparameters across a task distribution.

    Optimizes kernel parameters to maximize average log marginal likelihood
    across all tasks in the distribution (empirical Bayes / Type-II ML).

    This is the GP analog of MAML: find kernel params that, when used as
    initialization, lead to good per-task fits with minimal data.

    Args:
        task_dist: Distribution of tasks to meta-learn from.
        base_kernel: Starting kernel (default: RBFKernel).
        n_epochs: Number of meta-optimization epochs.
        noise_variance: GP noise variance.
        learning_rate: Step size for parameter updates.
        val_fraction: Fraction of tasks for validation.
        rng: Random generator.

    Returns:
        MetaLearningResult with learned kernel and training history.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if base_kernel is None:
        base_kernel = RBFKernel(length_scale=1.0, variance=1.0)

    # Split tasks into train/val
    n_val = max(1, int(task_dist.n_tasks * val_fraction))
    n_train = task_dist.n_tasks - n_val
    perm = rng.permutation(task_dist.n_tasks)
    train_tasks = [task_dist.tasks[i] for i in perm[:n_train]]
    val_tasks = [task_dist.tasks[i] for i in perm[n_train:]]

    # Current kernel params (log-space)
    current_params = base_kernel.params()
    best_params = current_params.copy()
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Compute average negative LML on train tasks
        base_kernel.set_params(current_params)
        train_lml = 0.0
        n_valid_tasks = 0
        for task in train_tasks:
            if len(task.X_support) < 2:
                continue
            gp = GaussianProcess(kernel=base_kernel, noise_variance=noise_variance)
            gp.fit(task.X_support, task.y_support)
            lml = gp.log_marginal_likelihood()
            if np.isfinite(lml):
                train_lml += lml
                n_valid_tasks += 1

        if n_valid_tasks > 0:
            avg_train_lml = train_lml / n_valid_tasks
        else:
            avg_train_lml = -1e10
        train_losses.append(-avg_train_lml)

        # Compute val loss
        val_lml = 0.0
        n_val_valid = 0
        for task in val_tasks:
            if len(task.X_support) < 2:
                continue
            gp = GaussianProcess(kernel=base_kernel, noise_variance=noise_variance)
            gp.fit(task.X_support, task.y_support)
            lml = gp.log_marginal_likelihood()
            if np.isfinite(lml):
                val_lml += lml
                n_val_valid += 1

        if n_val_valid > 0:
            avg_val_lml = val_lml / n_val_valid
        else:
            avg_val_lml = -1e10
        val_losses.append(-avg_val_lml)

        if -avg_val_lml < best_val_loss:
            best_val_loss = -avg_val_lml
            best_params = current_params.copy()

        # Finite-difference gradient on kernel params (log-space)
        grad = np.zeros_like(current_params)
        eps = 1e-4
        for j in range(len(current_params)):
            params_plus = current_params.copy()
            params_plus[j] += eps
            base_kernel.set_params(params_plus)

            lml_plus = 0.0
            n_plus = 0
            for task in train_tasks:
                if len(task.X_support) < 2:
                    continue
                gp = GaussianProcess(kernel=base_kernel, noise_variance=noise_variance)
                gp.fit(task.X_support, task.y_support)
                l = gp.log_marginal_likelihood()
                if np.isfinite(l):
                    lml_plus += l
                    n_plus += 1

            if n_plus > 0:
                grad[j] = (lml_plus / n_plus - avg_train_lml) / eps

        # Gradient ascent on LML (maximize)
        current_params = current_params + learning_rate * grad

    # Use best params
    base_kernel.set_params(best_params)

    # Evaluate per-task performance with meta-learned kernel
    task_perfs = []
    for task in task_dist.tasks:
        if len(task.X_support) < 2:
            task_perfs.append(float('inf'))
            continue
        gp = GaussianProcess(kernel=base_kernel, noise_variance=noise_variance)
        gp.fit(task.X_support, task.y_support)
        pred = gp.predict(task.X_query, return_std=True)
        task_perfs.append(_rmse(task.y_query, pred.mean))

    return MetaLearningResult(
        meta_kernel=base_kernel,
        meta_noise=noise_variance,
        train_losses=train_losses,
        val_losses=val_losses,
        task_performances=task_perfs,
        n_tasks_seen=task_dist.n_tasks,
        n_epochs=n_epochs,
    )


# ---------------------------------------------------------------------------
# Few-Shot Learning with Meta-Learned Prior
# ---------------------------------------------------------------------------

def few_shot_predict(meta_kernel: Kernel, meta_noise: float,
                     task: Task) -> FewShotResult:
    """Make few-shot predictions on a new task using a meta-learned kernel.

    The meta-learned kernel encodes prior knowledge from the task distribution,
    enabling good predictions from very few support examples.
    """
    gp = GaussianProcess(kernel=meta_kernel, noise_variance=meta_noise)
    gp.fit(task.X_support, task.y_support)
    pred = gp.predict(task.X_query, return_std=True)

    rmse = _rmse(task.y_query, pred.mean)
    nlpd = _nlpd(task.y_query, pred.mean, pred.std)

    return FewShotResult(
        predictions=pred.mean,
        uncertainty=pred.std,
        rmse=rmse,
        nlpd=nlpd,
        model=gp,
        n_support=len(task.X_support),
    )


def few_shot_adapt(meta_kernel: Kernel, meta_noise: float,
                   task: Task, n_adapt_steps: int = 3,
                   adapt_lr: float = 0.05) -> FewShotResult:
    """Few-shot prediction with per-task adaptation (fine-tuning).

    Starts from meta-learned kernel, then takes a few gradient steps
    on the task's support set to adapt the kernel parameters.
    """
    # Save meta params
    meta_params = meta_kernel.params().copy()

    # Fine-tune on support set
    current_params = meta_params.copy()
    for step in range(n_adapt_steps):
        meta_kernel.set_params(current_params)
        gp = GaussianProcess(kernel=meta_kernel, noise_variance=meta_noise)
        gp.fit(task.X_support, task.y_support)
        base_lml = gp.log_marginal_likelihood()

        # Finite-difference gradient
        grad = np.zeros_like(current_params)
        eps = 1e-4
        for j in range(len(current_params)):
            p_plus = current_params.copy()
            p_plus[j] += eps
            meta_kernel.set_params(p_plus)
            gp_plus = GaussianProcess(kernel=meta_kernel, noise_variance=meta_noise)
            gp_plus.fit(task.X_support, task.y_support)
            lml_plus = gp_plus.log_marginal_likelihood()
            if np.isfinite(lml_plus) and np.isfinite(base_lml):
                grad[j] = (lml_plus - base_lml) / eps

        current_params = current_params + adapt_lr * grad

    # Final prediction with adapted kernel
    meta_kernel.set_params(current_params)
    gp = GaussianProcess(kernel=meta_kernel, noise_variance=meta_noise)
    gp.fit(task.X_support, task.y_support)
    pred = gp.predict(task.X_query, return_std=True)

    rmse = _rmse(task.y_query, pred.mean)
    nlpd = _nlpd(task.y_query, pred.mean, pred.std)

    # Restore meta params (don't permanently modify)
    meta_kernel.set_params(meta_params)

    return FewShotResult(
        predictions=pred.mean,
        uncertainty=pred.std,
        rmse=rmse,
        nlpd=nlpd,
        model=gp,
        n_support=len(task.X_support),
    )


# ---------------------------------------------------------------------------
# Task Embedding via Kernel Parameters
# ---------------------------------------------------------------------------

def compute_task_embeddings(task_dist: TaskDistribution,
                            kernel: Optional[Kernel] = None,
                            noise_variance: float = 0.01,
                            ) -> List[TaskEmbedding]:
    """Embed tasks by fitting a GP to each and using kernel params as embedding.

    Tasks with similar structure will have similar kernel parameters,
    creating a natural metric space over tasks.
    """
    if kernel is None:
        kernel = RBFKernel(length_scale=1.0, variance=1.0)

    embeddings = []
    base_params = kernel.params().copy()

    for task in task_dist.tasks:
        # Reset to base params for each task
        kernel.set_params(base_params.copy())

        if len(task.X_support) < 2:
            # Not enough data -- use base params
            embeddings.append(TaskEmbedding(
                task_id=task.task_id,
                embedding=base_params.copy(),
                kernel_params=base_params.copy(),
            ))
            continue

        # Optimize kernel for this task
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        try:
            opt_result = gp.optimize(task.X_support, task.y_support, n_restarts=1)
            fitted_params = kernel.params().copy()
        except Exception:
            fitted_params = base_params.copy()

        embeddings.append(TaskEmbedding(
            task_id=task.task_id,
            embedding=fitted_params,
            kernel_params=fitted_params,
        ))

    # Reset kernel
    kernel.set_params(base_params)
    return embeddings


def task_similarity(emb1: TaskEmbedding, emb2: TaskEmbedding) -> float:
    """Compute similarity between two tasks via their embeddings.

    Uses RBF kernel in embedding space.
    """
    diff = emb1.embedding - emb2.embedding
    return float(np.exp(-0.5 * np.dot(diff, diff)))


def find_similar_tasks(target_embedding: TaskEmbedding,
                       all_embeddings: List[TaskEmbedding],
                       top_k: int = 5) -> List[Tuple[int, float]]:
    """Find the k most similar tasks to a target task.

    Returns list of (task_id, similarity_score) sorted by similarity.
    """
    similarities = []
    for emb in all_embeddings:
        if emb.task_id == target_embedding.task_id:
            continue
        sim = task_similarity(target_embedding, emb)
        similarities.append((emb.task_id, sim))
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


# ---------------------------------------------------------------------------
# Transfer Learning: Warm-Start from Similar Tasks
# ---------------------------------------------------------------------------

def transfer_predict(meta_kernel: Kernel, meta_noise: float,
                     target_task: Task,
                     source_tasks: List[Task],
                     transfer_weight: float = 0.3,
                     ) -> FewShotResult:
    """Few-shot prediction with transfer from source tasks.

    Augments the target task's support set with weighted data from
    the most relevant source tasks. More similar sources contribute more.
    """
    # Collect augmented training data
    X_aug = [target_task.X_support]
    y_aug = [target_task.y_support]

    for src in source_tasks:
        # Subsample source data proportional to transfer_weight
        n_transfer = max(1, int(len(src.X_support) * transfer_weight))
        if n_transfer > len(src.X_support):
            n_transfer = len(src.X_support)
        X_aug.append(src.X_support[:n_transfer])
        y_aug.append(src.y_support[:n_transfer])

    X_train = np.vstack(X_aug)
    y_train = np.concatenate(y_aug)

    gp = GaussianProcess(kernel=meta_kernel, noise_variance=meta_noise)
    gp.fit(X_train, y_train)
    pred = gp.predict(target_task.X_query, return_std=True)

    rmse = _rmse(target_task.y_query, pred.mean)
    nlpd = _nlpd(target_task.y_query, pred.mean, pred.std)

    return FewShotResult(
        predictions=pred.mean,
        uncertainty=pred.std,
        rmse=rmse,
        nlpd=nlpd,
        model=gp,
        n_support=len(X_train),
    )


# ---------------------------------------------------------------------------
# Meta-Active Learning
# ---------------------------------------------------------------------------

def meta_active_learning(task_dist: TaskDistribution,
                         n_queries_per_task: int = 10,
                         strategies: Optional[List[QueryStrategy]] = None,
                         kernel: Optional[Kernel] = None,
                         noise_variance: float = 0.01,
                         rng: Optional[np.random.Generator] = None,
                         ) -> MetaALResult:
    """Meta-learn the best active learning strategy across tasks.

    Runs multiple AL strategies on the task distribution and identifies
    which strategy generalizes best across tasks.

    Args:
        task_dist: Task distribution to meta-learn AL strategy from.
        n_queries_per_task: Budget per task.
        strategies: Strategies to compare (default: uncertainty, bald, random).
        kernel: GP kernel for AL.
        noise_variance: GP noise.
        rng: Random generator.

    Returns:
        MetaALResult with best strategy and per-strategy scores.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if strategies is None:
        strategies = [QueryStrategy.UNCERTAINTY, QueryStrategy.BALD, QueryStrategy.RANDOM]
    if kernel is None:
        kernel = RBFKernel(length_scale=1.0, variance=1.0)

    strategy_scores: Dict[str, List[float]] = {s.value: [] for s in strategies}
    all_results = []

    for task in task_dist.tasks:
        if len(task.X_support) < 2:
            continue

        # Build pool from query points (simulate having unlabeled data)
        X_pool = task.X_query.copy()
        # Use 2 initial labeled points from support
        n_init = min(2, len(task.X_support))
        X_init = task.X_support[:n_init]
        y_init = task.y_support[:n_init]

        # Oracle from task
        # Build a simple oracle: nearest-neighbor lookup in full query data
        def make_oracle(t):
            def oracle(X):
                # Find closest point in query set
                results = np.zeros(len(X))
                all_X = np.vstack([t.X_support, t.X_query])
                all_y = np.concatenate([t.y_support, t.y_query])
                for i, x in enumerate(X):
                    dists = np.sum((all_X - x.reshape(1, -1)) ** 2, axis=1)
                    results[i] = all_y[np.argmin(dists)]
                return results
            return oracle

        oracle = make_oracle(task)
        n_q = min(n_queries_per_task, len(X_pool) - 1)
        if n_q < 1:
            continue

        # Evaluation: RMSE on remaining query points
        eval_X = task.X_query
        eval_y = task.y_query

        def make_eval(ex, ey):
            def ev(gp):
                pred = gp.predict(ex, return_std=True)
                return _rmse(ey, pred.mean)
            return ev

        eval_func = make_eval(eval_X, eval_y)

        for strat in strategies:
            try:
                result = pool_based_active_learning(
                    oracle=oracle,
                    X_pool=X_pool,
                    X_initial=X_init.copy(),
                    y_initial=y_init.copy(),
                    n_queries=n_q,
                    strategy=strat,
                    kernel=kernel,
                    noise_variance=noise_variance,
                    eval_func=eval_func,
                    rng=rng,
                )
                # Final RMSE
                final_rmse = eval_func(result.model)
                strategy_scores[strat.value].append(final_rmse)
                if strat == strategies[0]:
                    all_results.append(result)
            except Exception:
                strategy_scores[strat.value].append(float('inf'))

    # Average scores per strategy
    avg_scores = {}
    for s, scores in strategy_scores.items():
        valid = [x for x in scores if np.isfinite(x)]
        avg_scores[s] = np.mean(valid) if valid else float('inf')

    best_strategy = min(avg_scores, key=avg_scores.get)

    return MetaALResult(
        meta_kernel=kernel,
        meta_noise=noise_variance,
        strategy_scores=avg_scores,
        best_strategy=best_strategy,
        per_task_results=all_results,
        n_tasks=task_dist.n_tasks,
    )


# ---------------------------------------------------------------------------
# Prototypical Few-Shot Learning
# ---------------------------------------------------------------------------

def compute_prototypes(task_dist: TaskDistribution,
                       kernel: Optional[Kernel] = None,
                       noise_variance: float = 0.01,
                       ) -> Dict[int, np.ndarray]:
    """Compute prototype representations for each task.

    A task's prototype is the mean of its GP posterior at a fixed grid,
    creating a functional fingerprint of the task.
    """
    if kernel is None:
        kernel = RBFKernel(length_scale=1.0, variance=1.0)

    # Determine grid range from all tasks
    all_X = np.vstack([t.X_support for t in task_dist.tasks])
    x_min = all_X.min(axis=0)
    x_max = all_X.max(axis=0)
    d = task_dist.input_dim

    # Grid for prototype computation
    n_grid = 20
    if d == 1:
        grid = np.linspace(x_min[0], x_max[0], n_grid).reshape(-1, 1)
    else:
        # For multi-dim, use a smaller grid along each dimension
        n_per_dim = max(3, int(n_grid ** (1.0 / d)))
        grids = [np.linspace(x_min[j], x_max[j], n_per_dim) for j in range(d)]
        mesh = np.meshgrid(*grids, indexing='ij')
        grid = np.column_stack([m.ravel() for m in mesh])

    prototypes = {}
    for task in task_dist.tasks:
        if len(task.X_support) < 2:
            prototypes[task.task_id] = np.zeros(len(grid))
            continue
        gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
        gp.fit(task.X_support, task.y_support)
        pred = gp.predict(grid, return_std=False)
        prototypes[task.task_id] = pred.mean

    return prototypes


def prototype_nearest_predict(prototypes: Dict[int, np.ndarray],
                               task_dist: TaskDistribution,
                               target_task: Task,
                               kernel: Optional[Kernel] = None,
                               noise_variance: float = 0.01,
                               top_k: int = 3,
                               ) -> FewShotResult:
    """Few-shot prediction by finding tasks with nearest prototypes.

    Identifies the k nearest tasks in prototype space and transfers
    their data to improve predictions on the target task.
    """
    if kernel is None:
        kernel = RBFKernel(length_scale=1.0, variance=1.0)

    # Compute target prototype
    gp_target = GaussianProcess(kernel=kernel, noise_variance=noise_variance)
    if len(target_task.X_support) >= 2:
        gp_target.fit(target_task.X_support, target_task.y_support)

        # Same grid as compute_prototypes
        all_X = np.vstack([t.X_support for t in task_dist.tasks])
        x_min = all_X.min(axis=0)
        x_max = all_X.max(axis=0)
        d = task_dist.input_dim
        n_grid = 20
        if d == 1:
            grid = np.linspace(x_min[0], x_max[0], n_grid).reshape(-1, 1)
        else:
            n_per_dim = max(3, int(n_grid ** (1.0 / d)))
            grids = [np.linspace(x_min[j], x_max[j], n_per_dim) for j in range(d)]
            mesh = np.meshgrid(*grids, indexing='ij')
            grid = np.column_stack([m.ravel() for m in mesh])

        pred = gp_target.predict(grid, return_std=False)
        target_proto = pred.mean
    else:
        # Too few points -- use zero prototype
        first_proto = next(iter(prototypes.values()))
        target_proto = np.zeros_like(first_proto)

    # Find nearest prototypes
    distances = []
    for tid, proto in prototypes.items():
        dist = np.sqrt(np.sum((target_proto - proto) ** 2))
        distances.append((tid, dist))
    distances.sort(key=lambda x: x[1])
    nearest = distances[:top_k]

    # Transfer from nearest tasks
    source_tasks = []
    for tid, _ in nearest:
        for t in task_dist.tasks:
            if t.task_id == tid:
                source_tasks.append(t)
                break

    return transfer_predict(kernel, noise_variance, target_task, source_tasks)


# ---------------------------------------------------------------------------
# N-Shot Learning Curve
# ---------------------------------------------------------------------------

def n_shot_learning_curve(meta_kernel: Kernel, meta_noise: float,
                          task_dist: TaskDistribution,
                          n_shots: List[int] = None,
                          rng: Optional[np.random.Generator] = None,
                          ) -> Dict[int, List[float]]:
    """Compute learning curves: performance vs number of support examples.

    For each n in n_shots, subsample the support set to n examples
    and measure query-set RMSE. Averages across all tasks.

    Returns dict mapping n_shot -> list of per-task RMSEs.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if n_shots is None:
        n_shots = [1, 2, 3, 5, 10, 20]

    results = {n: [] for n in n_shots}

    for task in task_dist.tasks:
        max_support = len(task.X_support)
        for n in n_shots:
            if n > max_support:
                continue
            # Subsample support set
            idx = rng.choice(max_support, size=n, replace=False)
            sub_task = Task(
                X_support=task.X_support[idx],
                y_support=task.y_support[idx],
                X_query=task.X_query,
                y_query=task.y_query,
                task_id=task.task_id,
            )
            if n < 2:
                # GP needs at least 2 points for non-degenerate fit
                results[n].append(float('inf'))
                continue
            fs = few_shot_predict(meta_kernel, meta_noise, sub_task)
            results[n].append(fs.rmse)

    return results


# ---------------------------------------------------------------------------
# Comparison: Meta-Learned vs Baseline
# ---------------------------------------------------------------------------

def compare_meta_vs_baseline(task_dist: TaskDistribution,
                             meta_kernel: Optional[Kernel] = None,
                             baseline_kernel: Optional[Kernel] = None,
                             meta_noise: float = 0.01,
                             baseline_noise: float = 0.01,
                             n_meta_epochs: int = 5,
                             rng: Optional[np.random.Generator] = None,
                             ) -> Dict[str, any]:
    """Compare meta-learned GP vs default GP on the task distribution.

    Returns performance comparison showing benefit of meta-learning.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if baseline_kernel is None:
        baseline_kernel = RBFKernel(length_scale=1.0, variance=1.0)
    if meta_kernel is None:
        meta_kernel = RBFKernel(length_scale=1.0, variance=1.0)

    # Split into train/test
    n_test = max(1, task_dist.n_tasks // 5)
    train_dist, test_dist = task_dist.train_test_split(n_test=n_test, rng=rng)

    # Meta-learn on train tasks
    meta_result = meta_learn_kernel(
        train_dist,
        base_kernel=meta_kernel,
        n_epochs=n_meta_epochs,
        noise_variance=meta_noise,
        rng=rng,
    )

    # Evaluate on test tasks
    meta_rmses = []
    baseline_rmses = []

    for task in test_dist.tasks:
        if len(task.X_support) < 2:
            continue
        # Meta-learned
        fs_meta = few_shot_predict(meta_result.meta_kernel, meta_result.meta_noise, task)
        meta_rmses.append(fs_meta.rmse)

        # Baseline
        fs_base = few_shot_predict(baseline_kernel, baseline_noise, task)
        baseline_rmses.append(fs_base.rmse)

    return {
        'meta_mean_rmse': np.mean(meta_rmses) if meta_rmses else float('inf'),
        'baseline_mean_rmse': np.mean(baseline_rmses) if baseline_rmses else float('inf'),
        'meta_rmses': meta_rmses,
        'baseline_rmses': baseline_rmses,
        'improvement': (
            (np.mean(baseline_rmses) - np.mean(meta_rmses)) / np.mean(baseline_rmses) * 100
            if baseline_rmses and np.mean(baseline_rmses) > 0 else 0.0
        ),
        'n_test_tasks': len(test_dist.tasks),
        'meta_result': meta_result,
    }


# ---------------------------------------------------------------------------
# Adaptive Kernel Selection
# ---------------------------------------------------------------------------

def adaptive_kernel_selection(task: Task,
                              candidate_kernels: Optional[List[Kernel]] = None,
                              noise_variance: float = 0.01,
                              ) -> Tuple[Kernel, float]:
    """Select the best kernel for a specific task via cross-validation.

    Uses leave-one-out cross-validation on the support set to pick
    the kernel with highest log marginal likelihood.

    Returns (best_kernel, best_lml).
    """
    if candidate_kernels is None:
        candidate_kernels = [
            RBFKernel(length_scale=1.0, variance=1.0),
            Matern52Kernel(length_scale=1.0, variance=1.0),
            RBFKernel(length_scale=0.5, variance=1.0),
            RBFKernel(length_scale=2.0, variance=1.0),
        ]

    best_kernel = candidate_kernels[0]
    best_lml = -float('inf')

    for k in candidate_kernels:
        if len(task.X_support) < 2:
            continue
        gp = GaussianProcess(kernel=k, noise_variance=noise_variance)
        gp.fit(task.X_support, task.y_support)
        lml = gp.log_marginal_likelihood()
        if np.isfinite(lml) and lml > best_lml:
            best_lml = lml
            best_kernel = k

    return best_kernel, best_lml


# ---------------------------------------------------------------------------
# Hierarchical Meta-Learning
# ---------------------------------------------------------------------------

def hierarchical_meta_learn(task_dist: TaskDistribution,
                            n_clusters: int = 3,
                            noise_variance: float = 0.01,
                            n_epochs: int = 5,
                            rng: Optional[np.random.Generator] = None,
                            ) -> Dict[str, any]:
    """Two-level meta-learning: cluster tasks, then meta-learn per cluster.

    1. Embed tasks via kernel parameters
    2. Cluster embeddings (k-means)
    3. Meta-learn a separate kernel per cluster
    4. For new tasks: assign to cluster, use cluster-specific kernel

    This handles heterogeneous task distributions where a single
    meta-kernel is insufficient.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Step 1: Embed tasks
    base_kernel = RBFKernel(length_scale=1.0, variance=1.0)
    embeddings = compute_task_embeddings(task_dist, kernel=base_kernel,
                                         noise_variance=noise_variance)

    emb_matrix = np.array([e.embedding for e in embeddings])

    # Step 2: K-means clustering
    n_clusters = min(n_clusters, len(embeddings))
    cluster_assignments = _kmeans(emb_matrix, n_clusters, rng=rng)

    # Step 3: Meta-learn per cluster
    cluster_kernels = {}
    cluster_results = {}
    for c in range(n_clusters):
        cluster_tasks = [task_dist.tasks[i] for i in range(len(task_dist.tasks))
                        if cluster_assignments[i] == c]
        if len(cluster_tasks) < 2:
            cluster_kernels[c] = RBFKernel(length_scale=1.0, variance=1.0)
            cluster_results[c] = None
            continue

        cluster_dist = TaskDistribution(cluster_tasks, f"cluster_{c}", task_dist.input_dim)
        result = meta_learn_kernel(
            cluster_dist,
            base_kernel=RBFKernel(length_scale=1.0, variance=1.0),
            n_epochs=n_epochs,
            noise_variance=noise_variance,
            rng=rng,
        )
        cluster_kernels[c] = result.meta_kernel
        cluster_results[c] = result

    return {
        'cluster_assignments': cluster_assignments,
        'cluster_kernels': cluster_kernels,
        'cluster_results': cluster_results,
        'embeddings': embeddings,
        'n_clusters': n_clusters,
    }


def hierarchical_predict(hier_result: Dict, task: Task,
                         noise_variance: float = 0.01,
                         ) -> FewShotResult:
    """Predict using hierarchical meta-learning: assign to cluster, use its kernel."""
    embeddings = hier_result['embeddings']
    cluster_assignments = hier_result['cluster_assignments']
    cluster_kernels = hier_result['cluster_kernels']

    # Compute task embedding
    base_kernel = RBFKernel(length_scale=1.0, variance=1.0)
    if len(task.X_support) >= 2:
        gp = GaussianProcess(kernel=base_kernel, noise_variance=noise_variance)
        try:
            gp.optimize(task.X_support, task.y_support, n_restarts=1)
            task_emb = base_kernel.params()
        except Exception:
            task_emb = base_kernel.params()
    else:
        task_emb = base_kernel.params()

    # Find nearest cluster
    emb_matrix = np.array([e.embedding for e in embeddings])
    centroids = {}
    for i, c in enumerate(cluster_assignments):
        if c not in centroids:
            centroids[c] = []
        centroids[c].append(emb_matrix[i])

    best_cluster = 0
    best_dist = float('inf')
    for c, members in centroids.items():
        centroid = np.mean(members, axis=0)
        dist = np.sqrt(np.sum((task_emb - centroid) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_cluster = c

    # Predict with cluster kernel
    kernel = cluster_kernels[best_cluster]
    return few_shot_predict(kernel, noise_variance, task)


def _kmeans(X: np.ndarray, k: int, max_iter: int = 50,
            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simple k-means clustering. Returns cluster assignments."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(X)
    if k >= n:
        return np.arange(n) % k

    # Random init
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()
    assignments = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign
        new_assignments = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.sum((centroids - X[i]) ** 2, axis=1)
            new_assignments[i] = np.argmin(dists)

        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        # Update centroids
        for c in range(k):
            members = X[assignments == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    return assignments


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def meta_learning_summary(result: MetaLearningResult,
                          name: str = "Meta-Learning") -> str:
    """Generate human-readable summary of meta-learning results."""
    lines = [
        f"=== {name} Summary ===",
        f"Tasks seen: {result.n_tasks_seen}",
        f"Epochs: {result.n_epochs}",
        f"Final train loss: {result.train_losses[-1]:.4f}" if result.train_losses else "",
        f"Final val loss: {result.val_losses[-1]:.4f}" if result.val_losses else "",
        f"Mean task RMSE: {np.mean([x for x in result.task_performances if np.isfinite(x)]):.4f}",
        f"Kernel: {type(result.meta_kernel).__name__}",
        f"Kernel params: {result.meta_kernel.params()}",
        f"Noise variance: {result.meta_noise:.6f}",
    ]
    return "\n".join(l for l in lines if l)


def few_shot_summary(result: FewShotResult, name: str = "Few-Shot") -> str:
    """Generate human-readable summary of few-shot prediction."""
    return (
        f"=== {name} Summary ===\n"
        f"Support examples: {result.n_support}\n"
        f"RMSE: {result.rmse:.4f}\n"
        f"NLPD: {result.nlpd:.4f}\n"
        f"Mean uncertainty: {np.mean(result.uncertainty):.4f}"
    )


def meta_al_summary(result: MetaALResult) -> str:
    """Generate summary of meta-active-learning comparison."""
    lines = [
        "=== Meta-Active-Learning Summary ===",
        f"Tasks evaluated: {result.n_tasks}",
        f"Best strategy: {result.best_strategy}",
        "Strategy scores (avg RMSE, lower is better):",
    ]
    for strat, score in sorted(result.strategy_scores.items(), key=lambda x: x[1]):
        lines.append(f"  {strat}: {score:.4f}")
    return "\n".join(lines)
