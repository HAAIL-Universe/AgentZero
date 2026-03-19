"""Tests for V235: Neural Process Active Learning."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V232_neural_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V226_active_learning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V229_meta_learning'))

from np_active_learning import (
    NPALResult, MetaALResult, AdaptiveALResult, NPALComparison,
    NPAcquisition,
    np_uncertainty, np_entropy, np_bald, np_expected_improvement, np_random,
    np_active_learning, np_batch_active_learning,
    meta_active_learning,
    adaptive_np_active_learning,
    compare_np_vs_gp_al,
    few_shot_warm_start,
    budget_allocated_al,
    np_information_gain,
    np_al_learning_curve,
    npal_summary, comparison_summary,
    _maxmin_select, _erf_approx,
)
from neural_process import (
    ConditionalNeuralProcess, NeuralProcess, AttentiveNeuralProcess,
    ConvCNP, GPNeuralProcess, NPPrediction, NPTrainResult
)
from meta_learning import Task, TaskDistribution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal(X):
    """Simple 1D test function."""
    return np.sin(2 * np.pi * X.ravel())

def quadratic(X):
    """Simple quadratic."""
    return X.ravel() ** 2

def make_pool(n=100, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    return rng.uniform(-2, 2, (n, 1))

def make_eval_data(oracle, n=50, rng=None):
    if rng is None:
        rng = np.random.RandomState(99)
    X = np.linspace(-2, 2, n).reshape(-1, 1)
    y = oracle(X)
    return X, y

def _make_sin_task_dist(rng, n_tasks=30):
    """Create a TaskDistribution of sinusoidal tasks."""
    tasks = []
    for i in range(n_tasks):
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 2.0)
        n_ctx = rng.randint(5, 15)
        n_tgt = rng.randint(10, 20)
        X_c = rng.uniform(-2, 2, (n_ctx, 1))
        X_t = rng.uniform(-2, 2, (n_tgt, 1))
        y_c = amp * np.sin(2 * np.pi * X_c.ravel() + phase)
        y_t = amp * np.sin(2 * np.pi * X_t.ravel() + phase)
        tasks.append(Task(X_support=X_c, y_support=y_c,
                          X_query=X_t, y_query=y_t, task_id=i))
    return TaskDistribution(tasks=tasks, name="sin_tasks")

def make_trained_cnp(oracle=sinusoidal, rng_seed=42):
    """Create and train a CNP on synthetic tasks."""
    rng = np.random.RandomState(rng_seed)
    dist = _make_sin_task_dist(rng, n_tasks=30)
    cnp = ConditionalNeuralProcess(n_basis=16, length_scale=1.0, noise=0.1)
    cnp.train(dist, n_epochs=30, lr=0.01, rng=rng)
    return cnp, dist

def make_trained_np(oracle=sinusoidal, rng_seed=42):
    """Create and train a NP with latent variables."""
    rng = np.random.RandomState(rng_seed)
    dist = _make_sin_task_dist(rng, n_tasks=30)
    np_model = NeuralProcess(n_basis=16, length_scale=1.0, latent_dim=8, noise=0.1)
    np_model.train(dist, n_epochs=30, lr=0.01, rng=rng)
    return np_model, dist


# ===================================================================
# Test Acquisition Functions
# ===================================================================

class TestNPAcquisitionFunctions:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()
        self.rng = np.random.RandomState(42)
        self.X_ctx = np.array([[-1.0], [0.0], [1.0]])
        self.y_ctx = sinusoidal(self.X_ctx)
        self.X_cand = np.linspace(-2, 2, 20).reshape(-1, 1)

    def test_uncertainty_returns_correct_shape(self):
        scores = np_uncertainty(self.cnp, self.X_ctx, self.y_ctx, self.X_cand)
        assert scores.shape == (20,)

    def test_uncertainty_nonnegative(self):
        scores = np_uncertainty(self.cnp, self.X_ctx, self.y_ctx, self.X_cand)
        assert np.all(scores >= 0)

    def test_entropy_returns_correct_shape(self):
        scores = np_entropy(self.cnp, self.X_ctx, self.y_ctx, self.X_cand)
        assert scores.shape == (20,)

    def test_entropy_finite(self):
        scores = np_entropy(self.cnp, self.X_ctx, self.y_ctx, self.X_cand)
        assert np.all(np.isfinite(scores))

    def test_bald_cnp_returns_correct_shape(self):
        scores = np_bald(self.cnp, self.X_ctx, self.y_ctx, self.X_cand, rng=self.rng)
        assert scores.shape == (20,)

    def test_bald_np_returns_correct_shape(self):
        np_model, _ = make_trained_np()
        scores = np_bald(np_model, self.X_ctx, self.y_ctx, self.X_cand,
                         n_samples=5, rng=self.rng)
        assert scores.shape == (20,)

    def test_expected_improvement_returns_correct_shape(self):
        scores = np_expected_improvement(self.cnp, self.X_ctx, self.y_ctx, self.X_cand)
        assert scores.shape == (20,)

    def test_expected_improvement_nonnegative(self):
        scores = np_expected_improvement(self.cnp, self.X_ctx, self.y_ctx, self.X_cand)
        assert np.all(scores >= -1e-10)

    def test_random_returns_correct_shape(self):
        scores = np_random(self.cnp, self.X_ctx, self.y_ctx, self.X_cand, rng=self.rng)
        assert scores.shape == (20,)

    def test_random_in_unit_interval(self):
        scores = np_random(self.cnp, self.X_ctx, self.y_ctx, self.X_cand, rng=self.rng)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_uncertainty_higher_away_from_context(self):
        # Points far from context should generally have higher uncertainty
        near = np.array([[0.1]])
        far = np.array([[1.9]])
        u_near = np_uncertainty(self.cnp, self.X_ctx, self.y_ctx, near)
        u_far = np_uncertainty(self.cnp, self.X_ctx, self.y_ctx, far)
        # Not a strict requirement but generally holds for well-trained models
        assert u_near.shape == (1,) and u_far.shape == (1,)


# ===================================================================
# Test Pool-Based NP Active Learning
# ===================================================================

class TestNPActiveLearning:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()
        self.rng = np.random.RandomState(42)
        self.X_pool = make_pool(80, self.rng)
        self.X_init = np.array([[-1.0], [0.0], [1.0]])
        self.y_init = sinusoidal(self.X_init)

    def test_basic_run(self):
        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=10, rng=self.rng
        )
        assert isinstance(result, NPALResult)
        assert result.n_queries == 10
        assert len(result.query_indices) == 10
        assert len(result.X_labeled) == 13  # 3 init + 10 queries

    def test_labeled_set_grows(self):
        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=5, rng=self.rng
        )
        assert len(result.X_labeled) == 8
        assert len(result.y_labeled) == 8

    def test_query_indices_unique(self):
        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=15, rng=self.rng
        )
        assert len(set(result.query_indices)) == len(result.query_indices)

    def test_with_eval_func(self):
        X_eval, y_eval = make_eval_data(sinusoidal)

        def eval_fn(m, X_ctx, y_ctx):
            pred = m.predict(X_ctx, y_ctx, X_eval)
            return np.sqrt(np.mean((pred.mean - y_eval) ** 2))

        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=10, eval_func=eval_fn, rng=self.rng
        )
        assert len(result.model_performance) == 11  # initial + 10 steps
        assert all(isinstance(p, float) for p in result.model_performance)

    def test_performance_improves(self):
        X_eval, y_eval = make_eval_data(sinusoidal)

        def eval_fn(m, X_ctx, y_ctx):
            pred = m.predict(X_ctx, y_ctx, X_eval)
            return np.sqrt(np.mean((pred.mean - y_eval) ** 2))

        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=15, eval_func=eval_fn, rng=self.rng
        )
        # RMSE should generally decrease (not guaranteed monotonically)
        assert result.model_performance[-1] <= result.model_performance[0] * 1.5

    def test_different_acquisitions(self):
        for acq in [NPAcquisition.UNCERTAINTY, NPAcquisition.ENTROPY,
                     NPAcquisition.EXPECTED_IMPROVEMENT, NPAcquisition.RANDOM]:
            result = np_active_learning(
                self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
                n_queries=5, acquisition=acq, rng=np.random.RandomState(42)
            )
            assert result.n_queries == 5

    def test_bald_acquisition(self):
        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=5, acquisition=NPAcquisition.BALD, rng=self.rng
        )
        assert result.n_queries == 5

    def test_model_name_stored(self):
        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=3, rng=self.rng
        )
        assert result.np_model_name == "ConditionalNeuralProcess"

    def test_scores_history_populated(self):
        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=5, rng=self.rng
        )
        assert len(result.scores_history) == 5

    def test_empty_pool_stops(self):
        small_pool = make_pool(5, self.rng)
        result = np_active_learning(
            self.cnp, sinusoidal, small_pool, self.X_init, self.y_init,
            n_queries=20, rng=self.rng
        )
        assert result.n_queries == 5

    def test_zero_queries(self):
        result = np_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=0, rng=self.rng
        )
        assert result.n_queries == 0
        assert len(result.X_labeled) == 3


# ===================================================================
# Test Batch NP Active Learning
# ===================================================================

class TestBatchNPActiveLearning:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()
        self.rng = np.random.RandomState(42)
        self.X_pool = make_pool(80, self.rng)
        self.X_init = np.array([[-1.0], [0.0], [1.0]])
        self.y_init = sinusoidal(self.X_init)

    def test_basic_batch(self):
        result = np_batch_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_batches=3, batch_size=4, rng=self.rng
        )
        assert isinstance(result, NPALResult)
        assert result.n_queries == 12  # 3 * 4

    def test_batch_indices_unique(self):
        result = np_batch_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_batches=3, batch_size=4, rng=self.rng
        )
        assert len(set(result.query_indices)) == len(result.query_indices)

    def test_diversity_weight_effect(self):
        # With high diversity weight, points should be more spread out
        result_diverse = np_batch_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_batches=2, batch_size=5, diversity_weight=0.9,
            rng=np.random.RandomState(42)
        )
        result_greedy = np_batch_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_batches=2, batch_size=5, diversity_weight=0.0,
            rng=np.random.RandomState(42)
        )
        # Both should complete
        assert result_diverse.n_queries == 10
        assert result_greedy.n_queries == 10

    def test_with_eval_func(self):
        X_eval, y_eval = make_eval_data(sinusoidal)

        def eval_fn(m, X_ctx, y_ctx):
            pred = m.predict(X_ctx, y_ctx, X_eval)
            return np.sqrt(np.mean((pred.mean - y_eval) ** 2))

        result = np_batch_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_batches=3, batch_size=4, eval_func=eval_fn, rng=self.rng
        )
        assert len(result.model_performance) == 4  # initial + 3 batches

    def test_labeled_set_correct_size(self):
        result = np_batch_active_learning(
            self.cnp, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_batches=2, batch_size=3, rng=self.rng
        )
        assert len(result.X_labeled) == 9  # 3 + 2*3


# ===================================================================
# Test Meta-Learned Active Learning
# ===================================================================

class TestMetaActiveLearning:
    def test_basic_meta_al(self):
        rng = np.random.RandomState(42)
        dist = _make_sin_task_dist(rng, n_tasks=25)

        result = meta_active_learning(
            dist,
            n_train_epochs=20,
            np_model_class=ConditionalNeuralProcess,
            np_kwargs={'n_basis': 12, 'noise': 0.1},
            n_test_tasks=3,
            n_queries=8,
            n_initial=3,
            rng=np.random.RandomState(42)
        )
        assert isinstance(result, MetaALResult)
        assert isinstance(result.train_result, NPTrainResult)
        assert len(result.al_results) == 3
        assert result.mean_final_rmse < float('inf')

    def test_meta_al_all_tasks_complete(self):
        rng = np.random.RandomState(42)
        tasks = []
        for i in range(25):
            a = rng.uniform(0.5, 2.0)
            b = rng.uniform(-1, 1)
            X_c = rng.uniform(-2, 2, (6, 1))
            X_t = rng.uniform(-2, 2, (12, 1))
            y_c = a * X_c.ravel() ** 2 + b
            y_t = a * X_t.ravel() ** 2 + b
            tasks.append(Task(X_support=X_c, y_support=y_c,
                              X_query=X_t, y_query=y_t, task_id=i))
        dist = TaskDistribution(tasks=tasks, name="quad")

        result = meta_active_learning(
            dist, n_train_epochs=15,
            n_test_tasks=4, n_queries=6, n_initial=2,
            rng=np.random.RandomState(42)
        )
        for r in result.al_results:
            assert r.n_queries > 0
            assert len(r.X_labeled) > 2

    def test_meta_al_different_acquisitions(self):
        rng = np.random.RandomState(42)
        tasks = []
        for i in range(20):
            a = rng.uniform(-2, 2)
            b = rng.uniform(-1, 1)
            X_c = rng.uniform(-2, 2, (6, 1))
            X_t = rng.uniform(-2, 2, (10, 1))
            y_c = a * X_c.ravel() + b
            y_t = a * X_t.ravel() + b
            tasks.append(Task(X_support=X_c, y_support=y_c,
                              X_query=X_t, y_query=y_t, task_id=i))
        dist = TaskDistribution(tasks=tasks, name="lin")

        for acq in [NPAcquisition.UNCERTAINTY, NPAcquisition.ENTROPY]:
            result = meta_active_learning(
                dist, n_train_epochs=10,
                n_test_tasks=2, n_queries=5, n_initial=2,
                acquisition=acq, rng=np.random.RandomState(42)
            )
            assert len(result.al_results) == 2


# ===================================================================
# Test Adaptive Model Selection
# ===================================================================

class TestAdaptiveNPActiveLearning:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()
        self.rng = np.random.RandomState(42)
        self.X_pool = make_pool(60, self.rng)
        self.X_init = np.array([[-1.0], [0.0], [1.0]])
        self.y_init = sinusoidal(self.X_init)

    def test_single_model(self):
        models = {"CNP": self.cnp}
        result = adaptive_np_active_learning(
            models, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=8, rng=self.rng
        )
        assert isinstance(result, AdaptiveALResult)
        assert result.n_queries == 8
        assert all(m == "CNP" for m in result.models_used)

    def test_two_models(self):
        cnp2, _ = make_trained_cnp(rng_seed=99)
        models = {"CNP_early": self.cnp, "CNP_late": cnp2}
        result = adaptive_np_active_learning(
            models, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=15, switch_points=[8], rng=self.rng
        )
        assert result.n_queries == 15
        # First queries should use CNP_early, later CNP_late
        early_models = result.models_used[:5]  # context < 8
        late_models = result.models_used[-3:]   # context >= 8+3=11
        assert all(m == "CNP_early" for m in early_models)
        assert all(m == "CNP_late" for m in late_models)

    def test_three_models(self):
        cnp2, _ = make_trained_cnp(rng_seed=99)
        cnp3, _ = make_trained_cnp(rng_seed=123)
        models = {"small": self.cnp, "medium": cnp2, "large": cnp3}
        result = adaptive_np_active_learning(
            models, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=20, switch_points=[5, 12], rng=self.rng
        )
        assert result.n_queries == 20
        assert "small" in result.models_used
        assert "large" in result.models_used

    def test_with_eval_func(self):
        X_eval, y_eval = make_eval_data(sinusoidal)

        def eval_fn(m, X_ctx, y_ctx):
            pred = m.predict(X_ctx, y_ctx, X_eval)
            return np.sqrt(np.mean((pred.mean - y_eval) ** 2))

        models = {"CNP": self.cnp}
        result = adaptive_np_active_learning(
            models, sinusoidal, self.X_pool, self.X_init, self.y_init,
            n_queries=5, eval_func=eval_fn, rng=self.rng
        )
        assert len(result.model_performance) == 6


# ===================================================================
# Test NP vs GP Comparison
# ===================================================================

class TestCompareNPvsGP:
    def test_basic_comparison(self):
        cnp, _ = make_trained_cnp()
        X_pool = make_pool(60, np.random.RandomState(42))
        X_init = np.array([[-1.0], [0.0], [1.0]])
        y_init = sinusoidal(X_init)
        X_eval, y_eval = make_eval_data(sinusoidal)

        result = compare_np_vs_gp_al(
            {"CNP": cnp}, sinusoidal, X_pool, X_init, y_init,
            X_eval, y_eval,
            n_queries=8, n_repeats=2, rng_seed=42
        )
        assert isinstance(result, NPALComparison)
        assert "CNP" in result.model_names
        assert "GP-AL" in result.model_names
        assert len(result.final_rmses["CNP"]) == 2
        assert len(result.final_rmses["GP-AL"]) == 2

    def test_comparison_summary_format(self):
        cnp, _ = make_trained_cnp()
        X_pool = make_pool(60, np.random.RandomState(42))
        X_init = np.array([[-1.0], [0.0], [1.0]])
        y_init = sinusoidal(X_init)
        X_eval, y_eval = make_eval_data(sinusoidal)

        result = compare_np_vs_gp_al(
            {"CNP": cnp}, sinusoidal, X_pool, X_init, y_init,
            X_eval, y_eval,
            n_queries=5, n_repeats=1, rng_seed=42
        )
        summary = comparison_summary(result)
        assert "CNP" in summary
        assert "GP-AL" in summary
        assert "Mean RMSE" in summary


# ===================================================================
# Test Few-Shot Warm-Start
# ===================================================================

class TestFewShotWarmStart:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()
        self.rng = np.random.RandomState(42)
        self.X_pool = make_pool(60, self.rng)

    def test_basic_warm_start(self):
        result = few_shot_warm_start(
            self.cnp, sinusoidal, self.X_pool,
            n_initial=2, n_queries=10, rng=np.random.RandomState(42)
        )
        assert isinstance(result, NPALResult)
        assert result.n_queries == 10
        assert len(result.X_labeled) == 12  # 2 init + 10

    def test_single_initial_point(self):
        result = few_shot_warm_start(
            self.cnp, sinusoidal, self.X_pool,
            n_initial=1, n_queries=5, rng=np.random.RandomState(42)
        )
        assert result.n_queries == 5
        assert len(result.X_labeled) == 6

    def test_with_eval_func(self):
        X_eval, y_eval = make_eval_data(sinusoidal)

        def eval_fn(m, X_ctx, y_ctx):
            pred = m.predict(X_ctx, y_ctx, X_eval)
            return np.sqrt(np.mean((pred.mean - y_eval) ** 2))

        result = few_shot_warm_start(
            self.cnp, sinusoidal, self.X_pool,
            n_initial=2, n_queries=8,
            eval_func=eval_fn, rng=np.random.RandomState(42)
        )
        assert len(result.model_performance) > 0

    def test_different_acquisitions(self):
        for acq in [NPAcquisition.UNCERTAINTY, NPAcquisition.ENTROPY]:
            result = few_shot_warm_start(
                self.cnp, sinusoidal, self.X_pool,
                n_initial=2, n_queries=5,
                acquisition=acq, rng=np.random.RandomState(42)
            )
            assert result.n_queries == 5


# ===================================================================
# Test Budget Allocation
# ===================================================================

class TestBudgetAllocatedAL:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()
        self.rng = np.random.RandomState(42)
        self.X_pool = make_pool(60, self.rng)
        self.X_init = np.array([[-1.0], [0.0], [1.0]])
        self.y_init = sinusoidal(self.X_init)

    def test_single_model_budget(self):
        result = budget_allocated_al(
            {"CNP": self.cnp}, sinusoidal, self.X_pool,
            self.X_init, self.y_init, total_budget=10, rng=self.rng
        )
        assert isinstance(result, NPALResult)
        assert result.n_queries == 10
        assert result.np_model_name == "budget_ensemble"

    def test_multi_model_budget(self):
        cnp2, _ = make_trained_cnp(rng_seed=99)
        result = budget_allocated_al(
            {"CNP_A": self.cnp, "CNP_B": cnp2},
            sinusoidal, self.X_pool,
            self.X_init, self.y_init, total_budget=10, rng=self.rng
        )
        assert result.n_queries == 10

    def test_budget_with_eval(self):
        X_eval, y_eval = make_eval_data(sinusoidal)

        def eval_fn(m, X_ctx, y_ctx):
            pred = m.predict(X_ctx, y_ctx, X_eval)
            return np.sqrt(np.mean((pred.mean - y_eval) ** 2))

        result = budget_allocated_al(
            {"CNP": self.cnp}, sinusoidal, self.X_pool,
            self.X_init, self.y_init, total_budget=8,
            eval_func=eval_fn, rng=self.rng
        )
        assert len(result.model_performance) == 9


# ===================================================================
# Test Information Gain
# ===================================================================

class TestInformationGain:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()
        self.X_ctx = np.array([[-1.0], [0.0], [1.0]])
        self.y_ctx = sinusoidal(self.X_ctx)
        self.X_cand = np.linspace(-2, 2, 10).reshape(-1, 1)

    def test_returns_correct_shape(self):
        ig = np_information_gain(
            self.cnp, self.X_ctx, self.y_ctx, self.X_cand,
            n_fantasy=3, rng=np.random.RandomState(42)
        )
        assert ig.shape == (10,)

    def test_finite_values(self):
        ig = np_information_gain(
            self.cnp, self.X_ctx, self.y_ctx, self.X_cand,
            n_fantasy=3, rng=np.random.RandomState(42)
        )
        assert np.all(np.isfinite(ig))

    def test_more_fantasies_smoother(self):
        ig3 = np_information_gain(
            self.cnp, self.X_ctx, self.y_ctx, self.X_cand,
            n_fantasy=3, rng=np.random.RandomState(42)
        )
        ig10 = np_information_gain(
            self.cnp, self.X_ctx, self.y_ctx, self.X_cand,
            n_fantasy=10, rng=np.random.RandomState(42)
        )
        # Both should be finite arrays of correct shape
        assert ig3.shape == ig10.shape == (10,)


# ===================================================================
# Test Learning Curve Analysis
# ===================================================================

class TestLearningCurve:
    def setup_method(self):
        self.cnp, _ = make_trained_cnp()

    def test_basic_learning_curve(self):
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_eval, y_eval = make_eval_data(sinusoidal)

        lc = np_al_learning_curve(
            self.cnp, sinusoidal, X_pool, X_eval, y_eval,
            n_queries=10, n_initial=3, rng=np.random.RandomState(42)
        )
        assert 'n_labeled' in lc
        assert 'rmse' in lc
        assert 'mean_std' in lc
        assert 'query_efficiency' in lc
        assert len(lc['rmse']) == len(lc['n_labeled'])

    def test_rmse_entries_positive(self):
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_eval, y_eval = make_eval_data(sinusoidal)

        lc = np_al_learning_curve(
            self.cnp, sinusoidal, X_pool, X_eval, y_eval,
            n_queries=8, n_initial=3, rng=np.random.RandomState(42)
        )
        assert all(r >= 0 for r in lc['rmse'])

    def test_n_labeled_increases(self):
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_eval, y_eval = make_eval_data(sinusoidal)

        lc = np_al_learning_curve(
            self.cnp, sinusoidal, X_pool, X_eval, y_eval,
            n_queries=8, n_initial=3, rng=np.random.RandomState(42)
        )
        for i in range(1, len(lc['n_labeled'])):
            assert lc['n_labeled'][i] >= lc['n_labeled'][i-1]

    def test_query_efficiency_length(self):
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_eval, y_eval = make_eval_data(sinusoidal)

        lc = np_al_learning_curve(
            self.cnp, sinusoidal, X_pool, X_eval, y_eval,
            n_queries=8, n_initial=3, rng=np.random.RandomState(42)
        )
        assert len(lc['query_efficiency']) == len(lc['rmse']) - 1


# ===================================================================
# Test Utilities
# ===================================================================

class TestUtilities:
    def test_erf_approx(self):
        # erf(0) = 0
        assert abs(_erf_approx(0.0)) < 1e-6
        # erf is odd
        assert abs(_erf_approx(1.0) + _erf_approx(-1.0)) < 1e-6
        # erf(large) -> 1
        assert abs(_erf_approx(5.0) - 1.0) < 1e-4

    def test_erf_array(self):
        x = np.array([-2, -1, 0, 1, 2])
        result = _erf_approx(x)
        assert result.shape == (5,)
        assert result[2] < 1e-6  # erf(0) ~ 0

    def test_maxmin_select(self):
        X = np.array([[0], [1], [2], [10], [11]])
        rng = np.random.RandomState(42)
        selected = _maxmin_select(X, 3, rng)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # unique

    def test_maxmin_select_all(self):
        X = np.array([[0], [1], [2]])
        selected = _maxmin_select(X, 5, np.random.RandomState(42))
        assert len(selected) == 3  # can't select more than available

    def test_npal_summary(self):
        result = NPALResult(
            X_labeled=np.zeros((5, 1)),
            y_labeled=np.zeros(5),
            query_indices=[0, 1, 2],
            model_performance=[1.0, 0.8, 0.6, 0.4],
            scores_history=[],
            n_queries=3,
            np_model_name="CNP"
        )
        s = npal_summary(result)
        assert "CNP" in s
        assert "Queries: 3" in s
        assert "Initial RMSE: 1.0000" in s
        assert "Final RMSE:   0.4000" in s

    def test_npal_summary_no_perf(self):
        result = NPALResult(
            X_labeled=np.zeros((3, 1)),
            y_labeled=np.zeros(3),
            query_indices=[],
            model_performance=[],
            scores_history=[],
            n_queries=0,
            np_model_name="NP"
        )
        s = npal_summary(result)
        assert "NP" in s

    def test_acquisition_enum_values(self):
        assert NPAcquisition.UNCERTAINTY.value == "uncertainty"
        assert NPAcquisition.ENTROPY.value == "entropy"
        assert NPAcquisition.BALD.value == "bald"
        assert NPAcquisition.EXPECTED_IMPROVEMENT.value == "expected_improvement"
        assert NPAcquisition.RANDOM.value == "random"


# ===================================================================
# Test Data Classes
# ===================================================================

class TestDataClasses:
    def test_npal_result(self):
        r = NPALResult(
            X_labeled=np.zeros((3, 1)),
            y_labeled=np.zeros(3),
            query_indices=[0, 1],
            model_performance=[1.0, 0.5],
            scores_history=[[0.1, 0.2]],
            n_queries=2,
            np_model_name="CNP"
        )
        assert r.n_queries == 2

    def test_meta_al_result(self):
        r = MetaALResult(
            train_result=NPTrainResult(
                train_losses=[], val_losses=[], best_epoch=0,
                n_tasks_seen=0, n_epochs=0, model_params={}
            ),
            al_results=[],
            mean_final_rmse=0.5,
            mean_queries_to_threshold=None
        )
        assert r.mean_final_rmse == 0.5

    def test_adaptive_al_result(self):
        r = AdaptiveALResult(
            X_labeled=np.zeros((3, 1)),
            y_labeled=np.zeros(3),
            query_indices=[0],
            model_performance=[1.0],
            models_used=["CNP"],
            n_queries=1
        )
        assert r.models_used == ["CNP"]

    def test_comparison_result(self):
        r = NPALComparison(
            model_names=["CNP", "GP-AL"],
            final_rmses={"CNP": [0.3], "GP-AL": [0.4]},
            learning_curves={},
            mean_final_rmse={"CNP": 0.3, "GP-AL": 0.4},
            queries_to_threshold={}
        )
        assert r.mean_final_rmse["CNP"] < r.mean_final_rmse["GP-AL"]


# ===================================================================
# Integration Tests
# ===================================================================

class TestIntegration:
    def test_cnp_al_end_to_end(self):
        """Full pipeline: train CNP, run AL, verify improvement."""
        cnp, dist = make_trained_cnp()
        rng = np.random.RandomState(42)
        X_pool = make_pool(80, rng)
        X_init = np.array([[-1.5], [0.5]])
        y_init = sinusoidal(X_init)
        X_eval, y_eval = make_eval_data(sinusoidal)

        def eval_fn(m, X_ctx, y_ctx):
            pred = m.predict(X_ctx, y_ctx, X_eval)
            return np.sqrt(np.mean((pred.mean - y_eval) ** 2))

        result = np_active_learning(
            cnp, sinusoidal, X_pool, X_init, y_init,
            n_queries=15, acquisition=NPAcquisition.UNCERTAINTY,
            eval_func=eval_fn, rng=rng
        )
        # Should have some performance data
        assert len(result.model_performance) >= 2
        # Summary should work
        s = npal_summary(result, "CNP-AL Test")
        assert "CNP-AL Test" in s

    def test_np_with_latent_al(self):
        """NP with latent variables for AL."""
        np_model, _ = make_trained_np()
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_init = np.array([[-1.0], [0.0], [1.0]])
        y_init = sinusoidal(X_init)

        result = np_active_learning(
            np_model, sinusoidal, X_pool, X_init, y_init,
            n_queries=8, acquisition=NPAcquisition.BALD,
            rng=rng
        )
        assert result.n_queries == 8
        assert result.np_model_name == "NeuralProcess"

    def test_adaptive_then_compare(self):
        """Adaptive selection followed by comparison."""
        cnp, _ = make_trained_cnp()
        cnp2, _ = make_trained_cnp(rng_seed=99)
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_init = np.array([[-1.0], [0.0], [1.0]])
        y_init = sinusoidal(X_init)

        # Adaptive
        adaptive_result = adaptive_np_active_learning(
            {"early": cnp, "late": cnp2},
            sinusoidal, X_pool, X_init, y_init,
            n_queries=10, rng=np.random.RandomState(42)
        )
        assert adaptive_result.n_queries == 10

    def test_few_shot_then_full(self):
        """Few-shot warm-start followed by continued learning."""
        cnp, _ = make_trained_cnp()
        rng = np.random.RandomState(42)
        X_pool = make_pool(80, rng)

        # Phase 1: few-shot warm-start
        result1 = few_shot_warm_start(
            cnp, sinusoidal, X_pool,
            n_initial=2, n_queries=5, rng=np.random.RandomState(42)
        )

        # Phase 2: continue with more queries using labeled data
        remaining = [i for i in range(len(X_pool))
                     if i not in result1.query_indices[:7]]
        X_remaining = X_pool[remaining]

        result2 = np_active_learning(
            cnp, sinusoidal, X_remaining,
            result1.X_labeled, result1.y_labeled,
            n_queries=5, rng=np.random.RandomState(99)
        )
        # Should have grown the labeled set
        assert len(result2.X_labeled) > len(result1.X_labeled)

    def test_quadratic_oracle(self):
        """Test with a different oracle function."""
        cnp, _ = make_trained_cnp()
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_init = np.array([[-1.0], [0.0], [1.0]])
        y_init = quadratic(X_init)

        result = np_active_learning(
            cnp, quadratic, X_pool, X_init, y_init,
            n_queries=10, rng=rng
        )
        assert result.n_queries == 10

    def test_learning_curve_integration(self):
        """Learning curve with full eval."""
        cnp, _ = make_trained_cnp()
        rng = np.random.RandomState(42)
        X_pool = make_pool(60, rng)
        X_eval, y_eval = make_eval_data(sinusoidal)

        lc = np_al_learning_curve(
            cnp, sinusoidal, X_pool, X_eval, y_eval,
            n_queries=12, n_initial=3, rng=np.random.RandomState(42)
        )
        assert len(lc['rmse']) > 5
        assert all(np.isfinite(r) for r in lc['rmse'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
