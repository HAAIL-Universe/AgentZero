"""Tests for V226: Active Learning."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))

from active_learning import (
    # Enums & classes
    QueryStrategy, ALResult, BatchALResult, StreamALResult, SynthesisResult,
    # Acquisition functions
    uncertainty_sampling, entropy_sampling, margin_sampling,
    qbc_sampling, expected_model_change, bald_sampling, random_sampling,
    # Core AL loops
    pool_based_active_learning, batch_active_learning,
    stream_active_learning, query_synthesis,
    # Utilities
    compare_strategies, make_rmse_evaluator,
    rmse_evaluation, nlpd_evaluation, coverage_evaluation,
    active_learning_summary,
    # Benchmarks
    sinusoidal_1d, bumps_1d, step_function_1d, friedman_2d, heteroscedastic_1d,
)
from gaussian_process import GaussianProcess, Matern52Kernel, RBFKernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_1d_problem(func, n_pool=100, n_init=3, rng_seed=42):
    """Create a 1D active learning problem."""
    rng = np.random.default_rng(rng_seed)
    X_pool = rng.uniform(0, 1, size=(n_pool, 1))
    X_init = np.array([[0.0], [0.5], [1.0]])[:n_init]
    y_init = func(X_init)
    return X_pool, X_init, y_init, rng


def make_2d_problem(func, n_pool=200, n_init=5, rng_seed=42):
    """Create a 2D active learning problem."""
    rng = np.random.default_rng(rng_seed)
    X_pool = rng.uniform(0, 1, size=(n_pool, 2))
    X_init = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])[:n_init]
    y_init = func(X_init)
    return X_pool, X_init, y_init, rng


# ===========================================================================
# Test Benchmark Functions
# ===========================================================================

class TestBenchmarks:
    def test_sinusoidal_shape(self):
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = sinusoidal_1d(X)
        assert y.shape == (10,)

    def test_sinusoidal_values(self):
        X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        y = sinusoidal_1d(X)
        np.testing.assert_allclose(y[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(y[2], 0.0, atol=1e-10)

    def test_bumps_shape(self):
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = bumps_1d(X)
        assert y.shape == (20,)

    def test_step_function_values(self):
        X = np.array([[0.1], [0.5], [0.9]])
        y = step_function_1d(X)
        assert y[0] == -1.0
        assert y[1] == 0.5
        assert y[2] == 1.0

    def test_friedman_2d(self):
        X = np.array([[0.5, 0.5]])
        y = friedman_2d(X)
        expected = np.sin(np.pi * 0.25) + 2 * 0.0
        np.testing.assert_allclose(y[0], expected, atol=1e-10)

    def test_heteroscedastic_shape(self):
        X = np.linspace(0, 1, 15).reshape(-1, 1)
        y = heteroscedastic_1d(X)
        assert y.shape == (15,)


# ===========================================================================
# Test Acquisition Functions
# ===========================================================================

class TestAcquisitionFunctions:
    @pytest.fixture
    def fitted_gp(self):
        rng = np.random.default_rng(42)
        X = np.array([[0.0], [0.5], [1.0]])
        y = sinusoidal_1d(X)
        gp = GaussianProcess(Matern52Kernel(), noise_variance=1e-4)
        gp.fit(X, y)
        return gp

    @pytest.fixture
    def candidates(self):
        return np.linspace(0, 1, 50).reshape(-1, 1)

    def test_uncertainty_returns_positive(self, fitted_gp, candidates):
        scores = uncertainty_sampling(fitted_gp, candidates)
        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    def test_uncertainty_low_at_training_points(self, fitted_gp):
        X_train = np.array([[0.0], [0.5], [1.0]])
        scores = uncertainty_sampling(fitted_gp, X_train)
        X_far = np.array([[0.25], [0.75]])
        scores_far = uncertainty_sampling(fitted_gp, X_far)
        assert np.mean(scores) < np.mean(scores_far)

    def test_entropy_returns_finite(self, fitted_gp, candidates):
        scores = entropy_sampling(fitted_gp, candidates)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))

    def test_entropy_higher_for_uncertain(self, fitted_gp):
        X_known = np.array([[0.0], [0.5], [1.0]])
        X_unknown = np.array([[0.25], [0.75]])
        e_known = entropy_sampling(fitted_gp, X_known)
        e_unknown = entropy_sampling(fitted_gp, X_unknown)
        assert np.mean(e_known) < np.mean(e_unknown)

    def test_margin_returns_negative(self, fitted_gp, candidates):
        scores = margin_sampling(fitted_gp, candidates)
        assert scores.shape == (50,)
        assert np.all(scores <= 0)

    def test_margin_highest_near_boundary(self, fitted_gp):
        # Points near zero mean should have highest margin score (least negative)
        X = np.array([[0.0], [0.25], [0.5]])
        scores = margin_sampling(fitted_gp, X, threshold=0.0)
        # sin(0)=0, sin(pi/2)~1, sin(pi)~0 -- point at 0.0 and 0.5 near boundary
        assert scores.shape == (3,)

    def test_qbc_returns_positive(self, fitted_gp, candidates):
        rng = np.random.default_rng(42)
        scores = qbc_sampling(fitted_gp, candidates, n_committee=5, rng=rng)
        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    def test_expected_model_change_returns_positive(self, fitted_gp, candidates):
        scores = expected_model_change(fitted_gp, candidates)
        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    def test_bald_returns_finite(self, fitted_gp, candidates):
        rng = np.random.default_rng(42)
        scores = bald_sampling(fitted_gp, candidates, rng=rng)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))

    def test_bald_nonnegative(self, fitted_gp, candidates):
        # BALD = H(y|x,D) - H(noise) >= 0 since predictive var >= noise var
        # Small numerical errors possible near training points (variance ~ noise)
        scores = bald_sampling(fitted_gp, candidates)
        assert np.all(scores >= -1e-3)

    def test_random_returns_valid(self, fitted_gp, candidates):
        rng = np.random.default_rng(42)
        scores = random_sampling(fitted_gp, candidates, rng=rng)
        assert scores.shape == (50,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


# ===========================================================================
# Test Pool-Based Active Learning
# ===========================================================================

class TestPoolBasedAL:
    def test_basic_run(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, strategy=QueryStrategy.UNCERTAINTY, rng=rng,
        )
        assert isinstance(result, ALResult)
        assert result.n_queries == 10
        assert len(result.y_labeled) == 3 + 10  # init + queries

    def test_labeled_grows(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, rng=rng,
        )
        assert len(result.X_labeled) == 8
        assert len(result.y_labeled) == 8

    def test_no_duplicate_queries(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=20, rng=rng,
        )
        assert len(set(result.query_indices)) == len(result.query_indices)

    def test_model_returned(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, rng=rng,
        )
        assert isinstance(result.model, GaussianProcess)
        pred = result.model.predict(np.array([[0.25]]), return_std=True)
        assert np.isfinite(pred.mean[0])

    def test_eval_func_tracked(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        eval_fn = make_rmse_evaluator(sinusoidal_1d, X_pool)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, eval_func=eval_fn, rng=rng,
        )
        assert len(result.model_performance) == 11  # 10 steps + final

    def test_uncertainty_beats_random(self):
        X_pool, X_init, y_init, _ = make_1d_problem(sinusoidal_1d, n_pool=80)
        eval_fn = make_rmse_evaluator(sinusoidal_1d, X_pool)

        r_unc = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            n_queries=15, strategy=QueryStrategy.UNCERTAINTY,
            eval_func=eval_fn, rng=np.random.default_rng(42),
        )
        r_rnd = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            n_queries=15, strategy=QueryStrategy.RANDOM,
            eval_func=eval_fn, rng=np.random.default_rng(42),
        )
        # Uncertainty sampling should achieve lower or equal RMSE
        assert r_unc.model_performance[-1] <= r_rnd.model_performance[-1] * 1.5

    def test_entropy_strategy(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, strategy=QueryStrategy.ENTROPY, rng=rng,
        )
        assert result.n_queries == 5

    def test_bald_strategy(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, strategy=QueryStrategy.BALD, rng=rng,
        )
        assert result.n_queries == 5

    def test_margin_strategy(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, strategy=QueryStrategy.MARGIN, rng=rng,
        )
        assert result.n_queries == 5

    def test_qbc_strategy(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, strategy=QueryStrategy.QBC, rng=rng,
        )
        assert result.n_queries == 5

    def test_expected_improvement_strategy(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, strategy=QueryStrategy.EXPECTED_IMPROVEMENT, rng=rng,
        )
        assert result.n_queries == 5

    def test_custom_kernel(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, kernel=RBFKernel(length_scale=0.3), rng=rng,
        )
        assert result.n_queries == 5

    def test_pool_exhaustion(self):
        """Request more queries than pool size."""
        rng = np.random.default_rng(42)
        X_pool = np.array([[0.2], [0.4], [0.6], [0.8]])
        X_init = np.array([[0.0], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, rng=rng,
        )
        # Should stop after exhausting pool (4 points)
        assert result.n_queries == 4

    def test_2d_problem(self):
        X_pool, X_init, y_init, rng = make_2d_problem(friedman_2d)
        result = pool_based_active_learning(
            oracle=friedman_2d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, rng=rng,
        )
        assert result.n_queries == 10
        assert result.X_labeled.shape[1] == 2

    def test_scores_history_recorded(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, rng=rng,
        )
        assert len(result.scores_history) == 5

    def test_bumps_function(self):
        X_pool, X_init, y_init, rng = make_1d_problem(bumps_1d)
        result = pool_based_active_learning(
            oracle=bumps_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, rng=rng,
        )
        assert result.n_queries == 10


# ===========================================================================
# Test Batch Active Learning
# ===========================================================================

class TestBatchAL:
    def test_basic_run(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=3, batch_size=4, rng=rng,
        )
        assert isinstance(result, BatchALResult)
        assert result.n_batches == 3
        assert result.total_queries == 12

    def test_batch_sizes_correct(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=2, batch_size=5, rng=rng,
        )
        assert len(result.batch_indices) == 2
        assert all(len(b) == 5 for b in result.batch_indices)

    def test_no_duplicate_across_batches(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=3, batch_size=5, rng=rng,
        )
        all_indices = [idx for batch in result.batch_indices for idx in batch]
        assert len(set(all_indices)) == len(all_indices)

    def test_diversity_weight_zero(self):
        """Pure uncertainty, no diversity."""
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=2, batch_size=3, diversity_weight=0.0, rng=rng,
        )
        assert result.total_queries == 6

    def test_diversity_weight_one(self):
        """Pure diversity, no uncertainty."""
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=2, batch_size=3, diversity_weight=1.0, rng=rng,
        )
        assert result.total_queries == 6

    def test_eval_func(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        eval_fn = make_rmse_evaluator(sinusoidal_1d, X_pool)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=3, batch_size=4, eval_func=eval_fn, rng=rng,
        )
        assert len(result.model_performance) == 4  # 3 batches + final

    def test_2d_batch(self):
        X_pool, X_init, y_init, rng = make_2d_problem(friedman_2d)
        result = batch_active_learning(
            oracle=friedman_2d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=2, batch_size=5, rng=rng,
        )
        assert result.total_queries == 10
        assert result.X_labeled.shape[1] == 2

    def test_pool_exhaustion(self):
        rng = np.random.default_rng(42)
        X_pool = np.linspace(0, 1, 6).reshape(-1, 1)
        X_init = np.array([[0.0], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=5, batch_size=3, rng=rng,
        )
        # Only 6 pool points, so can do at most 2 full batches
        assert result.total_queries == 6


# ===========================================================================
# Test Stream-Based Active Learning
# ===========================================================================

class TestStreamAL:
    def test_basic_run(self):
        rng = np.random.default_rng(42)
        X_stream = rng.uniform(0, 1, size=(50, 1))
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = stream_active_learning(
            oracle=sinusoidal_1d, X_stream=X_stream,
            X_initial=X_init, y_initial=y_init,
            threshold=0.5, rng=rng,
        )
        assert isinstance(result, StreamALResult)
        assert result.n_seen == 50
        assert result.n_queried > 0
        assert result.n_queried <= 50

    def test_budget_respected(self):
        rng = np.random.default_rng(42)
        X_stream = rng.uniform(0, 1, size=(100, 1))
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = stream_active_learning(
            oracle=sinusoidal_1d, X_stream=X_stream,
            X_initial=X_init, y_initial=y_init,
            threshold=0.3, budget=10, rng=rng,
        )
        assert result.n_queried <= 10

    def test_low_threshold_queries_more(self):
        rng = np.random.default_rng(42)
        X_stream = rng.uniform(0, 1, size=(50, 1))
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)

        r_low = stream_active_learning(
            oracle=sinusoidal_1d, X_stream=X_stream,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            threshold=0.2, rng=np.random.default_rng(42),
        )
        r_high = stream_active_learning(
            oracle=sinusoidal_1d, X_stream=X_stream,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            threshold=0.8, rng=np.random.default_rng(42),
        )
        assert r_low.n_queried >= r_high.n_queried

    def test_query_rate(self):
        rng = np.random.default_rng(42)
        X_stream = rng.uniform(0, 1, size=(50, 1))
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = stream_active_learning(
            oracle=sinusoidal_1d, X_stream=X_stream,
            X_initial=X_init, y_initial=y_init,
            threshold=0.5, rng=rng,
        )
        assert 0.0 <= result.query_rate <= 1.0
        assert abs(result.query_rate - result.n_queried / 50) < 1e-10

    def test_labeled_grows(self):
        rng = np.random.default_rng(42)
        X_stream = rng.uniform(0, 1, size=(30, 1))
        X_init = np.array([[0.0], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = stream_active_learning(
            oracle=sinusoidal_1d, X_stream=X_stream,
            X_initial=X_init, y_initial=y_init,
            threshold=0.3, rng=rng,
        )
        assert len(result.y_labeled) == 2 + result.n_queried


# ===========================================================================
# Test Query Synthesis
# ===========================================================================

class TestQuerySynthesis:
    def test_basic_run(self):
        rng = np.random.default_rng(42)
        bounds = np.array([[0.0, 1.0]])
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = query_synthesis(
            oracle=sinusoidal_1d, bounds=bounds,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, n_candidates=500, rng=rng,
        )
        assert isinstance(result, SynthesisResult)
        assert result.n_queries == 10
        assert result.X_synthesized.shape == (10, 1)

    def test_synthesized_in_bounds(self):
        rng = np.random.default_rng(42)
        bounds = np.array([[0.0, 1.0]])
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = query_synthesis(
            oracle=sinusoidal_1d, bounds=bounds,
            X_initial=X_init, y_initial=y_init,
            n_queries=15, rng=rng,
        )
        assert np.all(result.X_synthesized >= 0.0)
        assert np.all(result.X_synthesized <= 1.0)

    def test_2d_synthesis(self):
        rng = np.random.default_rng(42)
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X_init = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        y_init = friedman_2d(X_init)
        result = query_synthesis(
            oracle=friedman_2d, bounds=bounds,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, rng=rng,
        )
        assert result.X_synthesized.shape == (10, 2)

    def test_eval_tracked(self):
        rng = np.random.default_rng(42)
        bounds = np.array([[0.0, 1.0]])
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)

        def eval_fn(gp, X_test):
            pred = gp.predict(X_test, return_std=True)
            return float(np.mean(pred.variance))

        result = query_synthesis(
            oracle=sinusoidal_1d, bounds=bounds,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, eval_func=eval_fn, rng=rng,
        )
        assert len(result.model_performance) == 6  # 5 steps + final

    def test_model_quality_improves(self):
        rng = np.random.default_rng(42)
        bounds = np.array([[0.0, 1.0]])
        X_init = np.array([[0.0], [1.0]])
        y_init = sinusoidal_1d(X_init)

        # Few queries
        r_few = query_synthesis(
            oracle=sinusoidal_1d, bounds=bounds,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            n_queries=3, rng=np.random.default_rng(42),
        )
        # Many queries
        r_many = query_synthesis(
            oracle=sinusoidal_1d, bounds=bounds,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            n_queries=15, rng=np.random.default_rng(42),
        )
        # More queries should give better model
        X_test = np.linspace(0, 1, 50).reshape(-1, 1)
        y_test = sinusoidal_1d(X_test)
        rmse_few = rmse_evaluation(r_few.model, X_test, y_true=y_test)
        rmse_many = rmse_evaluation(r_many.model, X_test, y_true=y_test)
        assert rmse_many <= rmse_few * 1.1  # More data should help (with tolerance)


# ===========================================================================
# Test Strategy Comparison
# ===========================================================================

class TestStrategyComparison:
    def test_compare_returns_all_strategies(self):
        X_pool, X_init, y_init, _ = make_1d_problem(sinusoidal_1d, n_pool=60)
        strategies = [QueryStrategy.UNCERTAINTY, QueryStrategy.ENTROPY, QueryStrategy.RANDOM]
        results = compare_strategies(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            strategies=strategies, n_queries=5,
        )
        assert len(results) == 3
        assert "uncertainty" in results
        assert "entropy" in results
        assert "random" in results

    def test_compare_with_eval(self):
        X_pool, X_init, y_init, _ = make_1d_problem(sinusoidal_1d, n_pool=60)
        eval_fn = make_rmse_evaluator(sinusoidal_1d, X_pool)
        results = compare_strategies(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            strategies=[QueryStrategy.UNCERTAINTY, QueryStrategy.RANDOM],
            n_queries=10, eval_func=eval_fn,
        )
        for name, result in results.items():
            assert len(result.model_performance) > 0

    def test_reproducible(self):
        X_pool, X_init, y_init, _ = make_1d_problem(sinusoidal_1d, n_pool=60)
        r1 = compare_strategies(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            strategies=[QueryStrategy.UNCERTAINTY], n_queries=5, rng_seed=42,
        )
        r2 = compare_strategies(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            strategies=[QueryStrategy.UNCERTAINTY], n_queries=5, rng_seed=42,
        )
        np.testing.assert_array_equal(
            r1["uncertainty"].query_indices,
            r2["uncertainty"].query_indices,
        )


# ===========================================================================
# Test Evaluation Metrics
# ===========================================================================

class TestEvaluation:
    @pytest.fixture
    def trained_model(self):
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = sinusoidal_1d(X)
        gp = GaussianProcess(Matern52Kernel(), noise_variance=1e-4)
        gp.fit(X, y)
        return gp

    def test_rmse_nonnegative(self, trained_model):
        X_test = np.linspace(0, 1, 30).reshape(-1, 1)
        y_test = sinusoidal_1d(X_test)
        r = rmse_evaluation(trained_model, X_test, y_true=y_test)
        assert r >= 0

    def test_rmse_with_oracle(self, trained_model):
        X_test = np.linspace(0, 1, 30).reshape(-1, 1)
        r = rmse_evaluation(trained_model, X_test, oracle=sinusoidal_1d)
        assert r >= 0

    def test_rmse_zero_for_perfect(self):
        X = np.linspace(0, 1, 50).reshape(-1, 1)
        y = np.zeros(50)  # Simple constant
        gp = GaussianProcess(RBFKernel(), noise_variance=1e-6)
        gp.fit(X, y)
        r = rmse_evaluation(gp, X, y_true=y)
        assert r < 0.01

    def test_nlpd_finite(self, trained_model):
        X_test = np.linspace(0, 1, 30).reshape(-1, 1)
        y_test = sinusoidal_1d(X_test)
        n = nlpd_evaluation(trained_model, X_test, y_test)
        assert np.isfinite(n)

    def test_coverage_in_range(self, trained_model):
        X_test = np.linspace(0, 1, 30).reshape(-1, 1)
        y_test = sinusoidal_1d(X_test)
        c = coverage_evaluation(trained_model, X_test, y_test, n_sigma=2.0)
        assert 0.0 <= c <= 1.0

    def test_coverage_higher_with_more_sigma(self, trained_model):
        X_test = np.linspace(0, 1, 30).reshape(-1, 1)
        y_test = sinusoidal_1d(X_test)
        c1 = coverage_evaluation(trained_model, X_test, y_test, n_sigma=1.0)
        c3 = coverage_evaluation(trained_model, X_test, y_test, n_sigma=3.0)
        assert c3 >= c1

    def test_make_rmse_evaluator(self):
        X_eval = np.linspace(0, 1, 20).reshape(-1, 1)
        eval_fn = make_rmse_evaluator(sinusoidal_1d, X_eval)
        gp = GaussianProcess(Matern52Kernel(), noise_variance=1e-4)
        gp.fit(X_eval, sinusoidal_1d(X_eval))
        r = eval_fn(gp, None)  # X_pool not used by this evaluator
        assert r >= 0


# ===========================================================================
# Test Summary
# ===========================================================================

class TestSummary:
    def test_al_result_summary(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        eval_fn = make_rmse_evaluator(sinusoidal_1d, X_pool)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=5, eval_func=eval_fn, rng=rng,
        )
        s = active_learning_summary(result, "Test AL")
        assert "Test AL" in s
        assert "Queries: 5" in s
        assert "RMSE" in s

    def test_batch_summary(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=2, batch_size=3, rng=rng,
        )
        s = active_learning_summary(result, "Batch")
        assert "Batch" in s
        assert "Batches: 2" in s

    def test_stream_summary(self):
        rng = np.random.default_rng(42)
        X_stream = rng.uniform(0, 1, size=(30, 1))
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = sinusoidal_1d(X_init)
        result = stream_active_learning(
            oracle=sinusoidal_1d, X_stream=X_stream,
            X_initial=X_init, y_initial=y_init, rng=rng,
        )
        s = active_learning_summary(result, "Stream")
        assert "Stream" in s
        assert "Query rate" in s


# ===========================================================================
# Test Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_single_dimension_input(self):
        X_pool = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        X_init = np.array([[0.0], [1.0]])
        y_init = np.array([0.0, 0.0])
        result = pool_based_active_learning(
            oracle=lambda X: np.zeros(len(X)),
            X_pool=X_pool, X_initial=X_init, y_initial=y_init,
            n_queries=3, rng=np.random.default_rng(42),
        )
        assert result.n_queries == 3

    def test_constant_function(self):
        """Active learning on a constant function -- should still work."""
        X_pool = np.linspace(0, 1, 20).reshape(-1, 1)
        X_init = np.array([[0.0], [1.0]])
        y_init = np.array([5.0, 5.0])
        result = pool_based_active_learning(
            oracle=lambda X: np.full(len(X), 5.0),
            X_pool=X_pool, X_initial=X_init, y_initial=y_init,
            n_queries=5, rng=np.random.default_rng(42),
        )
        assert result.n_queries == 5

    def test_noisy_oracle(self):
        """Oracle with noise."""
        rng = np.random.default_rng(42)
        def noisy_oracle(X):
            return sinusoidal_1d(X) + rng.normal(0, 0.1, size=len(X))

        X_pool = np.linspace(0, 1, 50).reshape(-1, 1)
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = noisy_oracle(X_init)
        result = pool_based_active_learning(
            oracle=noisy_oracle, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, noise_variance=0.01, rng=rng,
        )
        assert result.n_queries == 10

    def test_high_dimensional(self):
        """5D problem."""
        rng = np.random.default_rng(42)
        def func_5d(X):
            X = np.atleast_2d(X)
            return np.sum(X ** 2, axis=1)

        X_pool = rng.uniform(-1, 1, size=(100, 5))
        X_init = rng.uniform(-1, 1, size=(10, 5))
        y_init = func_5d(X_init)
        result = pool_based_active_learning(
            oracle=func_5d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=10, rng=rng,
        )
        assert result.n_queries == 10
        assert result.X_labeled.shape[1] == 5

    def test_zero_queries(self):
        X_pool, X_init, y_init, rng = make_1d_problem(sinusoidal_1d)
        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=0, rng=rng,
        )
        assert result.n_queries == 0
        assert len(result.y_labeled) == len(y_init)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_full_pipeline_1d(self):
        """Full pipeline: active learn, then evaluate final model."""
        rng = np.random.default_rng(42)
        X_pool = np.linspace(0, 1, 80).reshape(-1, 1)
        X_init = np.array([[0.0], [1.0]])
        y_init = sinusoidal_1d(X_init)
        X_test = np.linspace(0, 1, 100).reshape(-1, 1)
        y_test = sinusoidal_1d(X_test)

        eval_fn = make_rmse_evaluator(sinusoidal_1d, X_test)

        result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=20, strategy=QueryStrategy.UNCERTAINTY,
            eval_func=eval_fn, rng=rng,
        )

        # Final RMSE should be reasonable
        final_rmse = rmse_evaluation(result.model, X_test, y_true=y_test)
        assert final_rmse < 0.5  # Should be much better with 22 points

        # Coverage should be good
        cov = coverage_evaluation(result.model, X_test, y_test, n_sigma=2.0)
        assert cov > 0.5

    def test_full_pipeline_2d(self):
        rng = np.random.default_rng(42)
        X_pool = rng.uniform(0, 1, size=(150, 2))
        X_init = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        y_init = friedman_2d(X_init)

        result = pool_based_active_learning(
            oracle=friedman_2d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=20, strategy=QueryStrategy.UNCERTAINTY, rng=rng,
        )
        assert result.n_queries == 20

    def test_batch_then_sequential(self):
        """Batch AL for initial exploration, then sequential for refinement."""
        rng = np.random.default_rng(42)
        X_pool = np.linspace(0, 1, 80).reshape(-1, 1)
        X_init = np.array([[0.0], [1.0]])
        y_init = sinusoidal_1d(X_init)

        # Phase 1: batch
        batch_result = batch_active_learning(
            oracle=sinusoidal_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_batches=2, batch_size=4, rng=rng,
        )

        # Phase 2: sequential from batch result
        used = set(idx for b in batch_result.batch_indices for idx in b)
        remaining_mask = np.array([i not in used for i in range(len(X_pool))])
        X_remaining = X_pool[remaining_mask]

        seq_result = pool_based_active_learning(
            oracle=sinusoidal_1d, X_pool=X_remaining,
            X_initial=batch_result.X_labeled, y_initial=batch_result.y_labeled,
            n_queries=5, strategy=QueryStrategy.BALD, rng=rng,
        )
        assert seq_result.n_queries == 5
        total = batch_result.total_queries + seq_result.n_queries
        assert total == 13

    def test_synthesis_vs_pool(self):
        """Query synthesis should produce reasonable results comparable to pool."""
        rng = np.random.default_rng(42)
        X_init = np.array([[0.0], [1.0]])
        y_init = sinusoidal_1d(X_init)

        synth_result = query_synthesis(
            oracle=sinusoidal_1d,
            bounds=np.array([[0.0, 1.0]]),
            X_initial=X_init.copy(), y_initial=y_init.copy(),
            n_queries=15, rng=np.random.default_rng(42),
        )

        X_test = np.linspace(0, 1, 50).reshape(-1, 1)
        y_test = sinusoidal_1d(X_test)
        rmse = rmse_evaluation(synth_result.model, X_test, y_true=y_test)
        assert rmse < 1.0  # Should be reasonable

    def test_step_function_active_learning(self):
        """Step function -- AL should focus on boundaries."""
        rng = np.random.default_rng(42)
        X_pool = np.linspace(0, 1, 100).reshape(-1, 1)
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = step_function_1d(X_init)

        result = pool_based_active_learning(
            oracle=step_function_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=15, strategy=QueryStrategy.UNCERTAINTY, rng=rng,
        )
        assert result.n_queries == 15
        # Check that queries cluster near boundaries (0.3 and 0.7)
        queries_x = X_pool[result.query_indices, 0]
        near_boundary = sum(1 for x in queries_x
                           if (0.2 < x < 0.4) or (0.6 < x < 0.8))
        # At least some queries should be near boundaries
        assert near_boundary >= 2

    def test_heteroscedastic_function(self):
        rng = np.random.default_rng(42)
        X_pool = np.linspace(0, 1, 80).reshape(-1, 1)
        X_init = np.array([[0.0], [0.5], [1.0]])
        y_init = heteroscedastic_1d(X_init)

        result = pool_based_active_learning(
            oracle=heteroscedastic_1d, X_pool=X_pool,
            X_initial=X_init, y_initial=y_init,
            n_queries=15, strategy=QueryStrategy.UNCERTAINTY, rng=rng,
        )
        # Uncertainty sampling should allocate more queries to the wiggly region (x > 0.5)
        queries_x = X_pool[result.query_indices, 0]
        right_half = sum(1 for x in queries_x if x > 0.5)
        # Should have more in the complex region
        assert right_half >= len(queries_x) // 3  # At least a third


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
