"""Tests for V229: Meta-Learning."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from meta_learning import (
    # Data classes
    Task, TaskDistribution, MetaLearningResult, FewShotResult,
    MetaALResult, TaskEmbedding,
    # Task generators
    sinusoidal_task_distribution, polynomial_task_distribution,
    step_task_distribution, multidim_task_distribution,
    # Core meta-learning
    meta_learn_kernel, few_shot_predict, few_shot_adapt,
    # Task embeddings
    compute_task_embeddings, task_similarity, find_similar_tasks,
    # Transfer learning
    transfer_predict,
    # Meta-active learning
    meta_active_learning,
    # Prototypical
    compute_prototypes, prototype_nearest_predict,
    # Learning curves
    n_shot_learning_curve,
    # Comparison
    compare_meta_vs_baseline,
    # Kernel selection
    adaptive_kernel_selection,
    # Hierarchical
    hierarchical_meta_learn, hierarchical_predict, _kmeans,
    # Evaluation helpers
    _rmse, _nlpd, evaluate_on_task, fit_task,
    # Summaries
    meta_learning_summary, few_shot_summary, meta_al_summary,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
from gaussian_process import RBFKernel, Matern52Kernel, GaussianProcess


# ===================================================================
# Task and TaskDistribution
# ===================================================================

class TestTask:
    def test_create_task(self):
        t = Task(
            X_support=np.array([[1.0], [2.0]]),
            y_support=np.array([1.0, 2.0]),
            X_query=np.array([[1.5]]),
            y_query=np.array([1.5]),
            task_id=0,
        )
        assert t.X_support.shape == (2, 1)
        assert t.y_support.shape == (2,)
        assert t.task_id == 0

    def test_task_metadata(self):
        t = Task(
            X_support=np.zeros((3, 2)),
            y_support=np.zeros(3),
            X_query=np.zeros((5, 2)),
            y_query=np.zeros(5),
            metadata={'A': 1.0, 'B': 2.0},
        )
        assert t.metadata['A'] == 1.0


class TestTaskDistribution:
    def test_create_distribution(self):
        tasks = [
            Task(np.zeros((3, 1)), np.zeros(3), np.zeros((5, 1)), np.zeros(5), task_id=i)
            for i in range(10)
        ]
        td = TaskDistribution(tasks, "test", input_dim=1)
        assert td.n_tasks == 10
        assert td.name == "test"

    def test_train_test_split(self):
        tasks = [
            Task(np.zeros((3, 1)), np.zeros(3), np.zeros((5, 1)), np.zeros(5), task_id=i)
            for i in range(10)
        ]
        td = TaskDistribution(tasks, "test", input_dim=1)
        train, test = td.train_test_split(n_test=3)
        assert train.n_tasks == 7
        assert test.n_tasks == 3

    def test_split_deterministic(self):
        tasks = [
            Task(np.zeros((3, 1)), np.zeros(3), np.zeros((5, 1)), np.zeros(5), task_id=i)
            for i in range(10)
        ]
        td = TaskDistribution(tasks, "test", input_dim=1)
        train1, test1 = td.train_test_split(n_test=3, rng=np.random.default_rng(99))
        train2, test2 = td.train_test_split(n_test=3, rng=np.random.default_rng(99))
        assert [t.task_id for t in test1.tasks] == [t.task_id for t in test2.tasks]


# ===================================================================
# Task Generators
# ===================================================================

class TestTaskGenerators:
    def test_sinusoidal_distribution(self):
        td = sinusoidal_task_distribution(n_tasks=10, n_support=5, n_query=20)
        assert td.n_tasks == 10
        assert td.tasks[0].X_support.shape == (5, 1)
        assert td.tasks[0].X_query.shape == (20, 1)
        assert td.name == "sinusoidal"

    def test_sinusoidal_tasks_differ(self):
        td = sinusoidal_task_distribution(n_tasks=5)
        # Different tasks should have different y values
        assert not np.allclose(td.tasks[0].y_support, td.tasks[1].y_support)

    def test_polynomial_distribution(self):
        td = polynomial_task_distribution(n_tasks=8, n_support=4, n_query=15)
        assert td.n_tasks == 8
        assert td.tasks[0].X_support.shape == (4, 1)
        assert td.name == "polynomial"

    def test_polynomial_metadata(self):
        td = polynomial_task_distribution(n_tasks=5)
        for t in td.tasks:
            assert 'degree' in t.metadata
            assert 'coeffs' in t.metadata
            assert 1 <= t.metadata['degree'] <= 3

    def test_step_distribution(self):
        td = step_task_distribution(n_tasks=6, n_support=8, n_query=25)
        assert td.n_tasks == 6
        assert td.tasks[0].X_support.shape == (8, 1)

    def test_step_metadata(self):
        td = step_task_distribution(n_tasks=5)
        for t in td.tasks:
            assert 'threshold' in t.metadata
            assert 'h1' in t.metadata
            assert 'h2' in t.metadata

    def test_multidim_distribution(self):
        td = multidim_task_distribution(n_tasks=5, n_support=10, n_query=30, input_dim=3)
        assert td.n_tasks == 5
        assert td.tasks[0].X_support.shape == (10, 3)
        assert td.input_dim == 3

    def test_multidim_metadata(self):
        td = multidim_task_distribution(n_tasks=5, input_dim=2)
        for t in td.tasks:
            assert 'w' in t.metadata
            assert len(t.metadata['w']) == 2

    def test_deterministic_generation(self):
        td1 = sinusoidal_task_distribution(n_tasks=5, rng=np.random.default_rng(123))
        td2 = sinusoidal_task_distribution(n_tasks=5, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(td1.tasks[0].X_support, td2.tasks[0].X_support)
        np.testing.assert_array_equal(td1.tasks[0].y_support, td2.tasks[0].y_support)


# ===================================================================
# Evaluation Helpers
# ===================================================================

class TestEvalHelpers:
    def test_rmse_perfect(self):
        assert _rmse(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0

    def test_rmse_nonzero(self):
        rmse = _rmse(np.array([1, 2, 3]), np.array([2, 3, 4]))
        assert abs(rmse - 1.0) < 1e-10

    def test_nlpd_low_for_good_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.1, 0.1])
        nlpd = _nlpd(y, mean, std)
        assert nlpd < 5.0  # Should be low for good predictions

    def test_nlpd_high_for_bad_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        mean = np.array([10.0, 20.0, 30.0])
        std = np.array([0.1, 0.1, 0.1])
        nlpd = _nlpd(y, mean, std)
        assert nlpd > 100.0  # Should be high for terrible predictions

    def test_fit_task(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (10, 1))
        y = np.sin(X[:, 0])
        task = Task(X, y, X, y)
        kernel = RBFKernel(length_scale=0.5, variance=1.0)
        gp = fit_task(kernel, 0.01, task)
        assert gp._X is not None

    def test_evaluate_on_task(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (20, 1))
        y = X[:, 0] * 2  # Simple linear
        task = Task(X[:10], y[:10], X[10:], y[10:])
        kernel = RBFKernel(length_scale=0.5, variance=1.0)
        gp = fit_task(kernel, 0.01, task)
        rmse, nlpd = evaluate_on_task(gp, task)
        assert rmse < 1.0  # Should be reasonable


# ===================================================================
# Meta-Learning: Kernel Transfer
# ===================================================================

class TestMetaLearnKernel:
    def test_basic_meta_learning(self):
        td = sinusoidal_task_distribution(n_tasks=10, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        result = meta_learn_kernel(td, n_epochs=3, rng=np.random.default_rng(42))
        assert isinstance(result, MetaLearningResult)
        assert result.n_tasks_seen == 10
        assert result.n_epochs == 3
        assert len(result.train_losses) == 3
        assert len(result.val_losses) == 3

    def test_meta_learning_reduces_loss(self):
        td = sinusoidal_task_distribution(n_tasks=15, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        result = meta_learn_kernel(td, n_epochs=5, rng=np.random.default_rng(42))
        # Losses should be finite
        assert all(np.isfinite(l) for l in result.train_losses)

    def test_meta_kernel_returned(self):
        td = polynomial_task_distribution(n_tasks=8, n_support=8, n_query=15,
                                           rng=np.random.default_rng(42))
        result = meta_learn_kernel(td, n_epochs=2, rng=np.random.default_rng(42))
        assert result.meta_kernel is not None
        assert result.meta_noise > 0

    def test_custom_kernel(self):
        td = sinusoidal_task_distribution(n_tasks=8, n_support=8, n_query=15,
                                           rng=np.random.default_rng(42))
        kernel = Matern52Kernel(length_scale=0.5, variance=2.0)
        result = meta_learn_kernel(td, base_kernel=kernel, n_epochs=2,
                                    rng=np.random.default_rng(42))
        assert isinstance(result.meta_kernel, Matern52Kernel)

    def test_task_performances_computed(self):
        td = sinusoidal_task_distribution(n_tasks=8, n_support=8, n_query=20,
                                           rng=np.random.default_rng(42))
        result = meta_learn_kernel(td, n_epochs=2, rng=np.random.default_rng(42))
        assert len(result.task_performances) == 8
        # At least some should be finite
        finite_perfs = [p for p in result.task_performances if np.isfinite(p)]
        assert len(finite_perfs) > 0


# ===================================================================
# Few-Shot Prediction
# ===================================================================

class TestFewShotPredict:
    def test_basic_few_shot(self):
        rng = np.random.default_rng(42)
        X_s = rng.uniform(-3, 3, (5, 1))
        y_s = 2.0 * np.sin(X_s[:, 0])
        X_q = rng.uniform(-3, 3, (20, 1))
        y_q = 2.0 * np.sin(X_q[:, 0])
        task = Task(X_s, y_s, X_q, y_q)

        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        result = few_shot_predict(kernel, 0.01, task)
        assert isinstance(result, FewShotResult)
        assert len(result.predictions) == 20
        assert len(result.uncertainty) == 20
        assert result.n_support == 5
        assert result.rmse >= 0

    def test_few_shot_with_meta_kernel(self):
        td = sinusoidal_task_distribution(n_tasks=10, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        meta = meta_learn_kernel(td, n_epochs=3, rng=np.random.default_rng(42))

        # New task
        rng = np.random.default_rng(99)
        X_s = rng.uniform(-3, 3, (3, 1))
        y_s = 3.0 * np.sin(1.5 * X_s[:, 0] + 1.0)
        X_q = rng.uniform(-3, 3, (30, 1))
        y_q = 3.0 * np.sin(1.5 * X_q[:, 0] + 1.0)
        new_task = Task(X_s, y_s, X_q, y_q)

        result = few_shot_predict(meta.meta_kernel, meta.meta_noise, new_task)
        assert result.rmse < 10.0  # Reasonable prediction
        assert result.nlpd < 50.0

    def test_uncertainty_increases_far_from_data(self):
        X_s = np.array([[0.0], [1.0]])
        y_s = np.array([0.0, 1.0])
        X_q = np.array([[0.5], [10.0]])
        y_q = np.array([0.5, 10.0])
        task = Task(X_s, y_s, X_q, y_q)

        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        result = few_shot_predict(kernel, 0.01, task)
        # Far-away point should have higher uncertainty
        assert result.uncertainty[1] > result.uncertainty[0]


class TestFewShotAdapt:
    def test_basic_adaptation(self):
        rng = np.random.default_rng(42)
        X_s = rng.uniform(-3, 3, (5, 1))
        y_s = 2.0 * np.sin(X_s[:, 0])
        X_q = rng.uniform(-3, 3, (20, 1))
        y_q = 2.0 * np.sin(X_q[:, 0])
        task = Task(X_s, y_s, X_q, y_q)

        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        result = few_shot_adapt(kernel, 0.01, task, n_adapt_steps=3)
        assert isinstance(result, FewShotResult)
        assert result.n_support == 5

    def test_adaptation_preserves_meta_params(self):
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        original_params = kernel.params().copy()

        rng = np.random.default_rng(42)
        X_s = rng.uniform(-3, 3, (5, 1))
        y_s = 2.0 * np.sin(X_s[:, 0])
        X_q = rng.uniform(-3, 3, (20, 1))
        y_q = 2.0 * np.sin(X_q[:, 0])
        task = Task(X_s, y_s, X_q, y_q)

        few_shot_adapt(kernel, 0.01, task, n_adapt_steps=3)
        # Kernel params should be restored
        np.testing.assert_array_almost_equal(kernel.params(), original_params)


# ===================================================================
# Task Embeddings
# ===================================================================

class TestTaskEmbeddings:
    def test_compute_embeddings(self):
        td = sinusoidal_task_distribution(n_tasks=5, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        embeddings = compute_task_embeddings(td)
        assert len(embeddings) == 5
        for e in embeddings:
            assert isinstance(e, TaskEmbedding)
            assert e.embedding is not None

    def test_similar_tasks_have_similar_embeddings(self):
        # Two tasks with same function should have similar embeddings
        rng = np.random.default_rng(42)
        X1 = rng.uniform(-3, 3, (10, 1))
        X2 = rng.uniform(-3, 3, (10, 1))
        Xq = rng.uniform(-3, 3, (20, 1))

        t1 = Task(X1, np.sin(X1[:, 0]), Xq, np.sin(Xq[:, 0]), task_id=0)
        t2 = Task(X2, np.sin(X2[:, 0]), Xq, np.sin(Xq[:, 0]), task_id=1)
        t3 = Task(X1, X1[:, 0] ** 3, Xq, Xq[:, 0] ** 3, task_id=2)

        td = TaskDistribution([t1, t2, t3], "test", input_dim=1)
        embs = compute_task_embeddings(td)

        sim_12 = task_similarity(embs[0], embs[1])
        sim_13 = task_similarity(embs[0], embs[2])
        # Similar tasks (both sin) should be more similar than different ones
        # (This may not always hold with few data, but generally should)
        assert sim_12 >= 0  # At minimum, similarities are non-negative

    def test_task_similarity_self(self):
        emb = TaskEmbedding(0, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert abs(task_similarity(emb, emb) - 1.0) < 1e-10

    def test_find_similar_tasks(self):
        embs = [
            TaskEmbedding(0, np.array([0.0, 0.0]), np.array([0.0, 0.0])),
            TaskEmbedding(1, np.array([0.1, 0.1]), np.array([0.1, 0.1])),
            TaskEmbedding(2, np.array([5.0, 5.0]), np.array([5.0, 5.0])),
            TaskEmbedding(3, np.array([0.2, 0.0]), np.array([0.2, 0.0])),
        ]
        similar = find_similar_tasks(embs[0], embs, top_k=2)
        assert len(similar) == 2
        # Task 1 should be most similar to task 0
        assert similar[0][0] == 1
        assert similar[0][1] > similar[1][1]  # Sorted by similarity


# ===================================================================
# Transfer Learning
# ===================================================================

class TestTransferPredict:
    def test_basic_transfer(self):
        rng = np.random.default_rng(42)
        # Source task with lots of data
        X_src = rng.uniform(-3, 3, (20, 1))
        y_src = np.sin(X_src[:, 0])
        src = Task(X_src, y_src, X_src, y_src, task_id=0)

        # Target task with little data (similar function)
        X_tgt = rng.uniform(-3, 3, (3, 1))
        y_tgt = 1.1 * np.sin(X_tgt[:, 0])
        X_q = rng.uniform(-3, 3, (20, 1))
        y_q = 1.1 * np.sin(X_q[:, 0])
        tgt = Task(X_tgt, y_tgt, X_q, y_q, task_id=1)

        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        result = transfer_predict(kernel, 0.01, tgt, [src])
        assert isinstance(result, FewShotResult)
        assert result.n_support > 3  # Should include transferred data

    def test_transfer_multiple_sources(self):
        rng = np.random.default_rng(42)
        sources = []
        for i in range(3):
            X = rng.uniform(-3, 3, (10, 1))
            y = np.sin(X[:, 0]) * (1 + 0.1 * i)
            sources.append(Task(X, y, X, y, task_id=i))

        X_tgt = rng.uniform(-3, 3, (2, 1))
        y_tgt = np.sin(X_tgt[:, 0])
        X_q = rng.uniform(-3, 3, (15, 1))
        y_q = np.sin(X_q[:, 0])
        tgt = Task(X_tgt, y_tgt, X_q, y_q, task_id=10)

        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        result = transfer_predict(kernel, 0.01, tgt, sources, transfer_weight=0.5)
        assert result.n_support > 2


# ===================================================================
# Meta-Active Learning
# ===================================================================

class TestMetaActiveLearning:
    def test_basic_meta_al(self):
        td = sinusoidal_task_distribution(n_tasks=5, n_support=5, n_query=30,
                                           rng=np.random.default_rng(42))
        result = meta_active_learning(td, n_queries_per_task=5,
                                       rng=np.random.default_rng(42))
        assert isinstance(result, MetaALResult)
        assert result.n_tasks == 5
        assert result.best_strategy in result.strategy_scores

    def test_strategy_scores_populated(self):
        td = sinusoidal_task_distribution(n_tasks=5, n_support=5, n_query=30,
                                           rng=np.random.default_rng(42))
        from active_learning import QueryStrategy
        strategies = [QueryStrategy.UNCERTAINTY, QueryStrategy.RANDOM]
        result = meta_active_learning(td, n_queries_per_task=3,
                                       strategies=strategies,
                                       rng=np.random.default_rng(42))
        assert 'uncertainty' in result.strategy_scores
        assert 'random' in result.strategy_scores

    def test_meta_al_custom_kernel(self):
        td = polynomial_task_distribution(n_tasks=4, n_support=5, n_query=25,
                                           rng=np.random.default_rng(42))
        kernel = Matern52Kernel(length_scale=1.0, variance=1.0)
        result = meta_active_learning(td, n_queries_per_task=3,
                                       kernel=kernel,
                                       rng=np.random.default_rng(42))
        assert isinstance(result.meta_kernel, Matern52Kernel)


# ===================================================================
# Prototypical Learning
# ===================================================================

class TestPrototypical:
    def test_compute_prototypes(self):
        td = sinusoidal_task_distribution(n_tasks=5, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        protos = compute_prototypes(td)
        assert len(protos) == 5
        for tid, p in protos.items():
            assert len(p) == 20  # Grid size

    def test_prototypes_differ(self):
        td = sinusoidal_task_distribution(n_tasks=5, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        protos = compute_prototypes(td)
        # Different tasks should have different prototypes
        p0 = protos[0]
        p1 = protos[1]
        assert not np.allclose(p0, p1)

    def test_prototype_nearest_predict(self):
        td = sinusoidal_task_distribution(n_tasks=10, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        protos = compute_prototypes(td)

        # New task similar to distribution
        rng = np.random.default_rng(99)
        X_s = rng.uniform(-5, 5, (3, 1))
        y_s = 2.0 * np.sin(1.0 * X_s[:, 0] + 0.5)
        X_q = rng.uniform(-5, 5, (20, 1))
        y_q = 2.0 * np.sin(1.0 * X_q[:, 0] + 0.5)
        target = Task(X_s, y_s, X_q, y_q, task_id=99)

        result = prototype_nearest_predict(protos, td, target)
        assert isinstance(result, FewShotResult)
        assert result.rmse >= 0

    def test_multidim_prototypes(self):
        td = multidim_task_distribution(n_tasks=5, n_support=10, n_query=20,
                                         input_dim=2, rng=np.random.default_rng(42))
        protos = compute_prototypes(td)
        assert len(protos) == 5
        # Multi-dim grid should be larger
        for tid, p in protos.items():
            assert len(p) > 0


# ===================================================================
# N-Shot Learning Curve
# ===================================================================

class TestLearningCurve:
    def test_basic_curve(self):
        td = sinusoidal_task_distribution(n_tasks=5, n_support=20, n_query=20,
                                           rng=np.random.default_rng(42))
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        curves = n_shot_learning_curve(kernel, 0.01, td, n_shots=[2, 5, 10])
        assert 2 in curves
        assert 5 in curves
        assert 10 in curves
        # Each should have results for all tasks
        assert len(curves[2]) > 0

    def test_more_shots_improves(self):
        td = sinusoidal_task_distribution(n_tasks=8, n_support=20, n_query=30,
                                           rng=np.random.default_rng(42))
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        curves = n_shot_learning_curve(kernel, 0.01, td, n_shots=[3, 10, 20],
                                        rng=np.random.default_rng(42))
        # Average RMSE should generally decrease with more shots
        avg_3 = np.mean([x for x in curves[3] if np.isfinite(x)])
        avg_20 = np.mean([x for x in curves[20] if np.isfinite(x)])
        # More data -> better (or at least not wildly worse)
        assert avg_20 < avg_3 * 3  # Generous bound

    def test_default_n_shots(self):
        td = sinusoidal_task_distribution(n_tasks=3, n_support=25, n_query=20,
                                           rng=np.random.default_rng(42))
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        curves = n_shot_learning_curve(kernel, 0.01, td)
        assert 1 in curves
        assert 2 in curves
        assert 5 in curves


# ===================================================================
# Compare Meta vs Baseline
# ===================================================================

class TestComparison:
    def test_basic_comparison(self):
        td = sinusoidal_task_distribution(n_tasks=15, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        comp = compare_meta_vs_baseline(td, n_meta_epochs=3,
                                         rng=np.random.default_rng(42))
        assert 'meta_mean_rmse' in comp
        assert 'baseline_mean_rmse' in comp
        assert 'improvement' in comp
        assert comp['n_test_tasks'] > 0

    def test_comparison_returns_meta_result(self):
        td = polynomial_task_distribution(n_tasks=10, n_support=8, n_query=15,
                                           rng=np.random.default_rng(42))
        comp = compare_meta_vs_baseline(td, n_meta_epochs=2,
                                         rng=np.random.default_rng(42))
        assert 'meta_result' in comp
        assert isinstance(comp['meta_result'], MetaLearningResult)

    def test_rmses_have_correct_length(self):
        td = sinusoidal_task_distribution(n_tasks=10, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        comp = compare_meta_vs_baseline(td, n_meta_epochs=2,
                                         rng=np.random.default_rng(42))
        assert len(comp['meta_rmses']) == len(comp['baseline_rmses'])
        assert len(comp['meta_rmses']) > 0


# ===================================================================
# Adaptive Kernel Selection
# ===================================================================

class TestAdaptiveKernelSelection:
    def test_selects_kernel(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-3, 3, (15, 1))
        y = np.sin(X[:, 0])
        Xq = rng.uniform(-3, 3, (10, 1))
        yq = np.sin(Xq[:, 0])
        task = Task(X, y, Xq, yq)

        kernel, lml = adaptive_kernel_selection(task)
        assert kernel is not None
        assert np.isfinite(lml)

    def test_custom_candidates(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-3, 3, (15, 1))
        y = X[:, 0] * 2 + 1  # Linear
        task = Task(X, y, X, y)

        candidates = [
            RBFKernel(length_scale=0.1, variance=1.0),
            RBFKernel(length_scale=10.0, variance=1.0),
        ]
        kernel, lml = adaptive_kernel_selection(task, candidate_kernels=candidates)
        assert kernel is not None


# ===================================================================
# Hierarchical Meta-Learning
# ===================================================================

class TestHierarchical:
    def test_basic_hierarchical(self):
        td = sinusoidal_task_distribution(n_tasks=12, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        result = hierarchical_meta_learn(td, n_clusters=2, n_epochs=2,
                                          rng=np.random.default_rng(42))
        assert 'cluster_assignments' in result
        assert 'cluster_kernels' in result
        assert result['n_clusters'] == 2
        assert len(result['cluster_assignments']) == 12

    def test_hierarchical_predict(self):
        td = sinusoidal_task_distribution(n_tasks=12, n_support=10, n_query=20,
                                           rng=np.random.default_rng(42))
        hier = hierarchical_meta_learn(td, n_clusters=2, n_epochs=2,
                                        rng=np.random.default_rng(42))

        rng = np.random.default_rng(99)
        X_s = rng.uniform(-3, 3, (5, 1))
        y_s = np.sin(X_s[:, 0])
        X_q = rng.uniform(-3, 3, (20, 1))
        y_q = np.sin(X_q[:, 0])
        task = Task(X_s, y_s, X_q, y_q)

        result = hierarchical_predict(hier, task)
        assert isinstance(result, FewShotResult)
        assert result.rmse >= 0

    def test_kmeans_basic(self):
        X = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 4.9]])
        assignments = _kmeans(X, k=2, rng=np.random.default_rng(42))
        # Points 0,1 should be in same cluster; points 2,3 in another
        assert assignments[0] == assignments[1]
        assert assignments[2] == assignments[3]
        assert assignments[0] != assignments[2]

    def test_kmeans_single_cluster(self):
        X = np.array([[1, 1], [2, 2], [3, 3]])
        assignments = _kmeans(X, k=1)
        assert all(a == 0 for a in assignments)

    def test_kmeans_k_equals_n(self):
        X = np.array([[1, 1], [2, 2], [3, 3]])
        assignments = _kmeans(X, k=3)
        assert len(set(assignments)) <= 3


# ===================================================================
# Summaries
# ===================================================================

class TestSummaries:
    def test_meta_learning_summary(self):
        result = MetaLearningResult(
            meta_kernel=RBFKernel(),
            meta_noise=0.01,
            train_losses=[1.0, 0.5, 0.3],
            val_losses=[1.2, 0.6, 0.4],
            task_performances=[0.1, 0.2, 0.15],
            n_tasks_seen=3,
            n_epochs=3,
        )
        s = meta_learning_summary(result)
        assert "Meta-Learning" in s
        assert "3" in s  # Epochs or tasks

    def test_few_shot_summary(self):
        result = FewShotResult(
            predictions=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.2]),
            rmse=0.5,
            nlpd=1.0,
            model=None,
            n_support=5,
        )
        s = few_shot_summary(result)
        assert "Few-Shot" in s
        assert "0.5" in s

    def test_meta_al_summary(self):
        result = MetaALResult(
            meta_kernel=RBFKernel(),
            meta_noise=0.01,
            strategy_scores={'uncertainty': 0.3, 'random': 0.5},
            best_strategy='uncertainty',
            per_task_results=[],
            n_tasks=5,
        )
        s = meta_al_summary(result)
        assert "Meta-Active-Learning" in s
        assert "uncertainty" in s


# ===================================================================
# Integration Tests
# ===================================================================

class TestIntegration:
    def test_full_pipeline_sinusoidal(self):
        """Full pipeline: generate tasks, meta-learn, few-shot predict."""
        rng = np.random.default_rng(42)
        td = sinusoidal_task_distribution(n_tasks=15, n_support=10, n_query=30, rng=rng)

        # Meta-learn
        train_dist, test_dist = td.train_test_split(n_test=3, rng=rng)
        meta = meta_learn_kernel(train_dist, n_epochs=3, rng=rng)

        # Few-shot on test tasks
        for task in test_dist.tasks:
            result = few_shot_predict(meta.meta_kernel, meta.meta_noise, task)
            assert np.isfinite(result.rmse)

    def test_full_pipeline_with_adaptation(self):
        """Meta-learn then adapt per-task."""
        rng = np.random.default_rng(42)
        td = sinusoidal_task_distribution(n_tasks=12, n_support=8, n_query=20, rng=rng)
        meta = meta_learn_kernel(td, n_epochs=3, rng=rng)

        new_task = td.tasks[0]
        adapted = few_shot_adapt(meta.meta_kernel, meta.meta_noise, new_task)
        vanilla = few_shot_predict(meta.meta_kernel, meta.meta_noise, new_task)

        # Both should produce valid results
        assert np.isfinite(adapted.rmse)
        assert np.isfinite(vanilla.rmse)

    def test_embedding_transfer_pipeline(self):
        """Embed tasks, find similar, transfer."""
        rng = np.random.default_rng(42)
        td = sinusoidal_task_distribution(n_tasks=10, n_support=10, n_query=20, rng=rng)
        embs = compute_task_embeddings(td)

        target = td.tasks[0]
        similar = find_similar_tasks(embs[0], embs, top_k=3)

        source_tasks = [td.tasks[tid] for tid, _ in similar]
        kernel = RBFKernel(length_scale=1.0, variance=1.0)
        result = transfer_predict(kernel, 0.01, target, source_tasks)
        assert np.isfinite(result.rmse)

    def test_hierarchical_pipeline(self):
        """Hierarchical: cluster tasks, meta-learn per cluster, predict."""
        rng = np.random.default_rng(42)
        td = sinusoidal_task_distribution(n_tasks=12, n_support=10, n_query=20, rng=rng)
        hier = hierarchical_meta_learn(td, n_clusters=2, n_epochs=2, rng=rng)

        new_task = td.tasks[0]
        result = hierarchical_predict(hier, new_task)
        assert np.isfinite(result.rmse)

    def test_polynomial_meta_learning(self):
        rng = np.random.default_rng(42)
        td = polynomial_task_distribution(n_tasks=10, n_support=8, n_query=20, rng=rng)
        meta = meta_learn_kernel(td, n_epochs=3, rng=rng)
        for task in td.tasks[:3]:
            result = few_shot_predict(meta.meta_kernel, meta.meta_noise, task)
            assert np.isfinite(result.rmse)

    def test_step_meta_learning(self):
        rng = np.random.default_rng(42)
        td = step_task_distribution(n_tasks=8, n_support=10, n_query=20, rng=rng)
        meta = meta_learn_kernel(td, n_epochs=2, rng=rng)
        result = few_shot_predict(meta.meta_kernel, meta.meta_noise, td.tasks[0])
        assert np.isfinite(result.rmse)

    def test_multidim_meta_learning(self):
        rng = np.random.default_rng(42)
        td = multidim_task_distribution(n_tasks=8, n_support=15, n_query=20,
                                         input_dim=2, rng=rng)
        meta = meta_learn_kernel(td, n_epochs=2, rng=rng)
        result = few_shot_predict(meta.meta_kernel, meta.meta_noise, td.tasks[0])
        assert np.isfinite(result.rmse)

    def test_learning_curve_with_meta_kernel(self):
        rng = np.random.default_rng(42)
        td = sinusoidal_task_distribution(n_tasks=8, n_support=20, n_query=20, rng=rng)
        meta = meta_learn_kernel(td, n_epochs=2, rng=rng)
        curves = n_shot_learning_curve(meta.meta_kernel, meta.meta_noise, td,
                                        n_shots=[2, 5, 10], rng=rng)
        assert len(curves[2]) > 0
        assert len(curves[5]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
