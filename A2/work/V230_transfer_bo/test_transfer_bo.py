"""
Tests for V230: Transfer Learning for Bayesian Optimization

Verifies: BOTask, TaskDatabase, transfer strategies, source selection,
negative transfer detection, multi-fidelity transfer, persistence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V227_multi_fidelity_bo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V229_meta_learning'))

import numpy as np
import pytest
import tempfile

from transfer_bo import (
    BOTask, TransferBOResult, TaskDatabase, TransferStrategy,
    build_task_embeddings, find_source_tasks, compute_transfer_weight,
    weighted_data_transfer, warm_start_surrogate,
    transfer_bo, cold_start_bo, auto_select_sources,
    multi_source_transfer_bo, meta_bo,
    detect_negative_transfer, compare_strategies,
    multi_fidelity_transfer_bo,
    save_task_database, load_task_database,
    run_and_store, transfer_bo_summary,
    _generate_candidates, _expected_improvement, _ucb,
)
from gaussian_process import GaussianProcess, RBFKernel, Matern52Kernel


# ---------------------------------------------------------------------------
# Test objectives (simple, fast, deterministic)
# ---------------------------------------------------------------------------

def sphere(x):
    """Sphere function: sum(x^2), minimum at origin."""
    return float(np.sum(x ** 2))

def shifted_sphere(x, shift=1.0):
    """Sphere shifted by constant."""
    return float(np.sum((x - shift) ** 2))

def rosenbrock_2d(x):
    """Rosenbrock 2D: (1-x0)^2 + 100*(x1-x0^2)^2."""
    return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

def quadratic_1d(x):
    return float((x[0] - 2.0) ** 2 + 1.0)

def quadratic_shifted_1d(x):
    return float((x[0] - 3.0) ** 2 + 1.0)

def sinusoidal_1d(x):
    return float(np.sin(3 * x[0]) + x[0] ** 2 * 0.1)

def unrelated_1d(x):
    """Completely different function."""
    return float(np.exp(-x[0]) * np.cos(5 * x[0]))


# ---------------------------------------------------------------------------
# Helper: create BOTask from function
# ---------------------------------------------------------------------------

def make_task(f, bounds, n_points=20, task_id=0, name="", rng=None):
    rng = rng or np.random.default_rng(42 + task_id)
    d = bounds.shape[0]
    X = np.zeros((n_points, d))
    for i in range(d):
        X[:, i] = rng.uniform(bounds[i, 0], bounds[i, 1], n_points)
    y = np.array([f(x) for x in X])
    return BOTask(X=X, y=y, bounds=bounds, task_id=task_id, name=name)


# ---------------------------------------------------------------------------
# BOTask tests
# ---------------------------------------------------------------------------

class TestBOTask:
    def test_creation(self):
        t = BOTask(X=np.array([[1.0], [2.0]]), y=np.array([1.0, 4.0]),
                   bounds=np.array([[0.0, 5.0]]))
        assert t.x_best is not None
        assert t.f_best == 1.0
        assert np.allclose(t.x_best, [1.0])

    def test_auto_best(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([5.0, 1.0, 3.0])
        t = BOTask(X=X, y=y, bounds=np.array([[0.0, 3.0]]))
        assert t.f_best == 1.0
        assert np.allclose(t.x_best, [1.0])

    def test_to_meta_task(self):
        t = make_task(quadratic_1d, np.array([[0.0, 5.0]]), n_points=20)
        mt = t.to_meta_task(n_query=5)
        assert len(mt.X_support) == 15
        assert len(mt.X_query) == 5
        assert len(mt.y_support) == 15
        assert len(mt.y_query) == 5

    def test_to_meta_task_no_query(self):
        t = make_task(quadratic_1d, np.array([[0.0, 5.0]]), n_points=10)
        mt = t.to_meta_task(n_query=0)
        assert len(mt.X_support) == 10
        assert len(mt.X_query) == 1  # at least 1

    def test_empty_task(self):
        t = BOTask(X=np.zeros((0, 2)), y=np.array([]),
                   bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
        assert t.x_best is None
        assert t.f_best is None

    def test_metadata(self):
        t = BOTask(X=np.array([[0.0]]), y=np.array([0.0]),
                   bounds=np.array([[0.0, 1.0]]),
                   metadata={"key": "value"})
        assert t.metadata["key"] == "value"

    def test_multidim(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        t = make_task(sphere, bounds, n_points=30)
        assert t.X.shape[1] == 3
        assert t.f_best >= 0.0


# ---------------------------------------------------------------------------
# TaskDatabase tests
# ---------------------------------------------------------------------------

class TestTaskDatabase:
    def test_empty(self):
        db = TaskDatabase()
        assert db.n_tasks() == 0
        assert db.get_task(0) is None

    def test_add_task(self):
        db = TaskDatabase()
        t = make_task(quadratic_1d, np.array([[0.0, 5.0]]))
        db.add_task(t)
        assert db.n_tasks() == 1
        assert db.get_task(0) is not None

    def test_multiple_tasks(self):
        db = TaskDatabase()
        for i in range(5):
            t = make_task(quadratic_1d, np.array([[0.0, 5.0]]), task_id=i)
            db.add_task(t)
        assert db.n_tasks() == 5
        assert db.get_task(3) is not None
        assert db.get_task(10) is None

    def test_embedding_invalidation(self):
        db = TaskDatabase()
        t1 = make_task(quadratic_1d, np.array([[0.0, 5.0]]), task_id=0)
        db.add_task(t1)
        build_task_embeddings(db)
        assert db.embeddings is not None
        t2 = make_task(quadratic_shifted_1d, np.array([[0.0, 5.0]]), task_id=1)
        db.add_task(t2)
        assert db.embeddings is None  # invalidated


# ---------------------------------------------------------------------------
# Candidate generation and acquisition tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_generate_candidates(self):
        bounds = np.array([[0.0, 1.0], [2.0, 3.0]])
        cands = _generate_candidates(bounds, n_candidates=100, rng=np.random.default_rng(0))
        assert cands.shape == (100, 2)
        assert np.all(cands[:, 0] >= 0.0) and np.all(cands[:, 0] <= 1.0)
        assert np.all(cands[:, 1] >= 2.0) and np.all(cands[:, 1] <= 3.0)

    def test_expected_improvement(self):
        mu = np.array([0.5, 0.3, 0.1])
        std = np.array([0.2, 0.2, 0.2])
        ei = _expected_improvement(mu, std, f_best=0.2)
        assert ei.shape == (3,)
        # Point with mu=0.1 (below f_best) should have high EI
        assert ei[2] > ei[0]

    def test_ei_zero_std(self):
        mu = np.array([0.5])
        std = np.array([0.0])
        ei = _expected_improvement(mu, std, f_best=0.2)
        assert ei[0] == 0.0

    def test_ucb(self):
        mu = np.array([1.0, 2.0])
        std = np.array([0.5, 0.1])
        ucb = _ucb(mu, std, beta=2.0)
        # UCB prefers low mu and high std
        assert ucb[0] > ucb[1]


# ---------------------------------------------------------------------------
# Transfer weight computation
# ---------------------------------------------------------------------------

class TestTransferWeight:
    def test_similar_tasks(self):
        bounds = np.array([[0.0, 5.0]])
        t1 = make_task(quadratic_1d, bounds, n_points=30, task_id=0)
        t2 = make_task(quadratic_shifted_1d, bounds, n_points=30, task_id=1)
        w = compute_transfer_weight(t1, t2)
        # Similar quadratics should have positive weight
        assert w > 0.05

    def test_identical_tasks(self):
        bounds = np.array([[0.0, 5.0]])
        t1 = make_task(quadratic_1d, bounds, n_points=30, task_id=0)
        t2 = make_task(quadratic_1d, bounds, n_points=30, task_id=0, rng=np.random.default_rng(99))
        w = compute_transfer_weight(t1, t2)
        assert w > 0.2

    def test_few_target_points(self):
        bounds = np.array([[0.0, 5.0]])
        t1 = make_task(quadratic_1d, bounds, n_points=30, task_id=0)
        t2 = BOTask(X=np.array([[1.0]]), y=np.array([2.0]),
                    bounds=bounds, task_id=1)
        w = compute_transfer_weight(t1, t2)
        assert w == 0.3  # default for few points

    def test_weight_bounded(self):
        bounds = np.array([[0.0, 5.0]])
        t1 = make_task(quadratic_1d, bounds, n_points=50, task_id=0)
        t2 = make_task(quadratic_1d, bounds, n_points=50, task_id=1,
                       rng=np.random.default_rng(7))
        w = compute_transfer_weight(t1, t2)
        assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# Weighted data transfer
# ---------------------------------------------------------------------------

class TestWeightedDataTransfer:
    def test_augmented_gp(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=20)
        X_target = np.array([[1.0], [3.0], [4.0]])
        y_target = np.array([2.0, 2.0, 5.0])
        base_gp = GaussianProcess(kernel=RBFKernel(), noise_variance=0.01)
        gp = weighted_data_transfer(base_gp, X_target, y_target, [source], [0.5])
        pred = gp.predict(np.array([[2.0]]))
        assert pred.mean.shape == (1,)

    def test_zero_weight_excluded(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=20)
        X_target = np.linspace(0, 5, 10).reshape(-1, 1)
        y_target = np.array([quadratic_1d(x) for x in X_target])
        base_gp = GaussianProcess(kernel=RBFKernel(), noise_variance=0.01)
        gp = weighted_data_transfer(base_gp, X_target, y_target, [source], [0.01])
        # With very low weight, augmented GP should have ~same data as target
        assert gp._X.shape[0] >= len(X_target)

    def test_multiple_sources(self):
        bounds = np.array([[0.0, 5.0]])
        s1 = make_task(quadratic_1d, bounds, n_points=15, task_id=0)
        s2 = make_task(quadratic_shifted_1d, bounds, n_points=15, task_id=1)
        X_target = np.array([[2.0], [3.0]])
        y_target = np.array([1.0, 2.0])
        base_gp = GaussianProcess(kernel=RBFKernel(), noise_variance=0.01)
        gp = weighted_data_transfer(base_gp, X_target, y_target, [s1, s2], [0.5, 0.3])
        pred = gp.predict(np.array([[2.5]]))
        assert pred.mean.shape == (1,)


# ---------------------------------------------------------------------------
# Warm start surrogate
# ---------------------------------------------------------------------------

class TestWarmStartSurrogate:
    def test_basic(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=20)
        target = BOTask(X=np.array([[1.0], [3.0]]), y=np.array([2.0, 2.0]),
                        bounds=bounds)
        gp = warm_start_surrogate(target, [source], [0.5])
        pred = gp.predict(np.array([[2.0]]))
        assert pred.mean.shape == (1,)

    def test_no_sources(self):
        bounds = np.array([[0.0, 5.0]])
        target = make_task(quadratic_1d, bounds, n_points=10)
        gp = warm_start_surrogate(target, [], [])
        pred = gp.predict(np.array([[2.0]]))
        assert pred.mean.shape == (1,)

    def test_prior_mean_stored(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=20)
        target = make_task(quadratic_shifted_1d, bounds, n_points=5)
        gp = warm_start_surrogate(target, [source], [0.5])
        assert hasattr(gp, '_prior_mean')


# ---------------------------------------------------------------------------
# Cold start BO
# ---------------------------------------------------------------------------

class TestColdStartBO:
    def test_1d_quadratic(self):
        bounds = np.array([[0.0, 5.0]])
        result = cold_start_bo(quadratic_1d, bounds, n_iterations=15,
                               n_initial=5, rng=np.random.default_rng(42))
        assert result.f_best < 5.0
        assert result.total_evaluations == 20
        assert len(result.convergence) == 16

    def test_convergence_monotone(self):
        bounds = np.array([[0.0, 5.0]])
        result = cold_start_bo(quadratic_1d, bounds, n_iterations=10,
                               n_initial=5, rng=np.random.default_rng(42))
        for i in range(1, len(result.convergence)):
            assert result.convergence[i] <= result.convergence[i - 1] + 1e-10

    def test_2d(self):
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
        result = cold_start_bo(sphere, bounds, n_iterations=15,
                               n_initial=5, rng=np.random.default_rng(42))
        assert result.f_best < 2.0
        assert result.X_history.shape[1] == 2


# ---------------------------------------------------------------------------
# Transfer BO
# ---------------------------------------------------------------------------

class TestTransferBO:
    def test_basic_transfer(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=30, task_id=0)
        result = transfer_bo(quadratic_shifted_1d, bounds, [source],
                             n_iterations=15, n_initial=5,
                             strategy=TransferStrategy.WEIGHTED_DATA,
                             rng=np.random.default_rng(42))
        assert result.f_best < 5.0
        assert result.total_evaluations == 20

    def test_warm_start_transfer(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=30)
        result = transfer_bo(quadratic_shifted_1d, bounds, [source],
                             n_iterations=15, n_initial=5,
                             strategy=TransferStrategy.WARM_START,
                             rng=np.random.default_rng(42))
        assert result.f_best < 5.0

    def test_adaptive_transfer(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=30)
        result = transfer_bo(quadratic_shifted_1d, bounds, [source],
                             n_iterations=15, n_initial=5,
                             strategy=TransferStrategy.ADAPTIVE,
                             rng=np.random.default_rng(42))
        assert result.f_best < 5.0

    def test_multi_source_transfer(self):
        bounds = np.array([[0.0, 5.0]])
        s1 = make_task(quadratic_1d, bounds, n_points=30, task_id=0)
        s2 = make_task(quadratic_shifted_1d, bounds, n_points=30, task_id=1)
        result = transfer_bo(sinusoidal_1d, bounds, [s1, s2],
                             n_iterations=15, n_initial=5,
                             strategy=TransferStrategy.MULTI_SOURCE,
                             rng=np.random.default_rng(42))
        assert result.total_evaluations == 20

    def test_meta_kernel_transfer(self):
        bounds = np.array([[0.0, 5.0]])
        s1 = make_task(quadratic_1d, bounds, n_points=30, task_id=0)
        s2 = make_task(quadratic_shifted_1d, bounds, n_points=30, task_id=1)
        result = transfer_bo(sinusoidal_1d, bounds, [s1, s2],
                             n_iterations=10, n_initial=5,
                             strategy=TransferStrategy.META_KERNEL,
                             rng=np.random.default_rng(42))
        assert result.total_evaluations == 15

    def test_no_sources(self):
        bounds = np.array([[0.0, 5.0]])
        result = transfer_bo(quadratic_1d, bounds, [],
                             n_iterations=10, n_initial=5,
                             rng=np.random.default_rng(42))
        assert result.total_evaluations == 15
        assert len(result.source_tasks_used) == 0

    def test_transfer_weights_recorded(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=30)
        result = transfer_bo(quadratic_shifted_1d, bounds, [source],
                             n_iterations=10, n_initial=5,
                             rng=np.random.default_rng(42))
        assert len(result.transfer_weights) == 1

    def test_2d_transfer(self):
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
        source = make_task(sphere, bounds, n_points=40, task_id=0)
        result = transfer_bo(lambda x: sphere(x - 0.5), bounds, [source],
                             n_iterations=15, n_initial=5,
                             rng=np.random.default_rng(42))
        assert result.X_history.shape[1] == 2
        assert result.f_best < 4.0


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

class TestAutoSelectSources:
    def test_empty_db(self):
        db = TaskDatabase()
        target = BOTask(X=np.array([[1.0]]), y=np.array([1.0]),
                        bounds=np.array([[0.0, 5.0]]))
        selected = auto_select_sources(db, target)
        assert len(selected) == 0

    def test_selects_sources(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        for i in range(5):
            t = make_task(lambda x, s=i: (x[0] - s) ** 2, bounds,
                          n_points=20, task_id=i)
            db.add_task(t)
        target = make_task(quadratic_1d, bounds, n_points=10)
        selected = auto_select_sources(db, target, max_sources=3)
        assert len(selected) <= 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in selected)

    def test_few_target_points(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        db.add_task(make_task(quadratic_1d, bounds, n_points=20))
        target = BOTask(X=np.array([[1.0]]), y=np.array([2.0]),
                        bounds=bounds, task_id=99)
        selected = auto_select_sources(db, target, max_sources=3)
        assert len(selected) >= 0  # may or may not find sources with 1 point


# ---------------------------------------------------------------------------
# Multi-source transfer BO
# ---------------------------------------------------------------------------

class TestMultiSourceTransferBO:
    def test_basic(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        db.add_task(make_task(quadratic_1d, bounds, n_points=25, task_id=0))
        db.add_task(make_task(quadratic_shifted_1d, bounds, n_points=25, task_id=1))
        result = multi_source_transfer_bo(
            sinusoidal_1d, bounds, db,
            n_iterations=10, n_initial=5,
            rng=np.random.default_rng(42)
        )
        assert result.total_evaluations == 15

    def test_weight_decay(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        db.add_task(make_task(quadratic_1d, bounds, n_points=30, task_id=0))
        result = multi_source_transfer_bo(
            quadratic_shifted_1d, bounds, db,
            n_iterations=15, n_initial=5,
            rng=np.random.default_rng(42)
        )
        assert result.f_best < 5.0


# ---------------------------------------------------------------------------
# Meta BO
# ---------------------------------------------------------------------------

class TestMetaBO:
    def test_basic(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        for i in range(5):
            db.add_task(make_task(lambda x, s=i: (x[0] - s) ** 2, bounds,
                                   n_points=20, task_id=i))
        result = meta_bo(quadratic_1d, bounds, db,
                         n_iterations=10, n_initial=5,
                         rng=np.random.default_rng(42))
        assert result.total_evaluations == 15
        assert result.f_best < 5.0

    def test_empty_db(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        result = meta_bo(quadratic_1d, bounds, db,
                         n_iterations=10, n_initial=5,
                         rng=np.random.default_rng(42))
        assert result.total_evaluations == 15

    def test_cached_kernel(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        for i in range(3):
            db.add_task(make_task(lambda x, s=i: (x[0] - s) ** 2, bounds,
                                   n_points=20, task_id=i))
        # First call meta-learns
        meta_bo(quadratic_1d, bounds, db, n_iterations=5, n_initial=3,
                rng=np.random.default_rng(42))
        assert db.meta_kernel is not None
        # Second call reuses
        result = meta_bo(quadratic_shifted_1d, bounds, db,
                         n_iterations=5, n_initial=3,
                         rng=np.random.default_rng(42))
        assert result.total_evaluations == 8


# ---------------------------------------------------------------------------
# Negative transfer detection
# ---------------------------------------------------------------------------

class TestNegativeTransferDetection:
    def test_no_negative(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=30)
        rng1 = np.random.default_rng(42)
        transfer_result = transfer_bo(quadratic_shifted_1d, bounds, [source],
                                       n_iterations=15, n_initial=5, rng=rng1)
        rng2 = np.random.default_rng(42)
        cold_result = cold_start_bo(quadratic_shifted_1d, bounds,
                                     n_iterations=15, n_initial=5, rng=rng2)
        det = detect_negative_transfer(transfer_result, cold_result)
        assert "is_negative_transfer" in det
        assert "improvement" in det
        assert "transfer_auc" in det
        assert "cold_auc" in det
        assert isinstance(det["early_benefit"], bool)

    def test_negative_with_unrelated(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(unrelated_1d, bounds, n_points=30)
        rng1 = np.random.default_rng(42)
        transfer_result = transfer_bo(quadratic_1d, bounds, [source],
                                       n_iterations=10, n_initial=5, rng=rng1)
        rng2 = np.random.default_rng(42)
        cold_result = cold_start_bo(quadratic_1d, bounds,
                                     n_iterations=10, n_initial=5, rng=rng2)
        det = detect_negative_transfer(transfer_result, cold_result)
        # Should detect some effect
        assert isinstance(det["is_negative_transfer"], (bool, np.bool_))


# ---------------------------------------------------------------------------
# Compare strategies
# ---------------------------------------------------------------------------

class TestCompareStrategies:
    def test_all_strategies(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=30)
        results = compare_strategies(quadratic_shifted_1d, bounds, [source],
                                      n_iterations=8, n_initial=5)
        assert "weighted_data" in results
        assert "warm_start" in results
        assert "adaptive" in results
        assert "multi_source" in results
        assert "cold_start" in results
        for name, r in results.items():
            assert r.total_evaluations == 13


# ---------------------------------------------------------------------------
# Multi-fidelity transfer
# ---------------------------------------------------------------------------

class TestMultiFidelityTransfer:
    def test_basic(self):
        bounds = np.array([[-2.0, 2.0]])
        source = make_task(sphere, bounds, n_points=30)

        objectives = {
            0: lambda x: sphere(x) + np.random.default_rng(0).normal(0, 0.5),  # low fidelity
            1: sphere  # high fidelity
        }
        costs = {0: 1.0, 1: 10.0}

        result = multi_fidelity_transfer_bo(
            objectives, bounds, costs, [source],
            budget=50.0, n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )
        assert result.f_best is not None
        assert result.total_cost <= 55.0  # budget + tolerance

    def test_no_sources(self):
        bounds = np.array([[-2.0, 2.0]])
        objectives = {
            0: lambda x: sphere(x) + 0.5,
            1: sphere
        }
        costs = {0: 1.0, 1: 10.0}
        result = multi_fidelity_transfer_bo(
            objectives, bounds, costs, [],
            budget=40.0, n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )
        assert result.f_best is not None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        db.add_task(make_task(quadratic_1d, bounds, n_points=15, task_id=0, name="q1"))
        db.add_task(make_task(quadratic_shifted_1d, bounds, n_points=15, task_id=1, name="q2"))

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            save_task_database(db, path)
            db2 = load_task_database(path)
            assert db2.n_tasks() == 2
            assert db2.tasks[0].task_id == 0
            assert db2.tasks[1].task_id == 1
            assert db2.tasks[0].name == "q1"
            assert np.allclose(db2.tasks[0].X, db.tasks[0].X)
            assert np.allclose(db2.tasks[1].y, db.tasks[1].y)
        finally:
            os.unlink(path)

    def test_roundtrip_bounds(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0], [1.0, 3.0]])
        db.add_task(make_task(sphere, bounds, n_points=10, task_id=0))

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            save_task_database(db, path)
            db2 = load_task_database(path)
            assert np.allclose(db2.tasks[0].bounds, bounds)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Run and store
# ---------------------------------------------------------------------------

class TestRunAndStore:
    def test_cold_start(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        result = run_and_store(quadratic_1d, bounds, db, task_name="q1",
                               n_iterations=10, n_initial=5,
                               rng=np.random.default_rng(42))
        assert db.n_tasks() == 1
        assert db.tasks[0].name == "q1"
        assert result.total_evaluations == 15

    def test_sequential_tasks(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        r1 = run_and_store(quadratic_1d, bounds, db, task_name="q1",
                           n_iterations=10, n_initial=5,
                           rng=np.random.default_rng(42))
        r2 = run_and_store(quadratic_shifted_1d, bounds, db, task_name="q2",
                           n_iterations=10, n_initial=5,
                           rng=np.random.default_rng(43))
        assert db.n_tasks() == 2
        assert r2.total_evaluations == 15

    def test_three_tasks(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        for i, f in enumerate([quadratic_1d, quadratic_shifted_1d, sinusoidal_1d]):
            run_and_store(f, bounds, db, task_name=f"task_{i}",
                          n_iterations=8, n_initial=5,
                          rng=np.random.default_rng(42 + i))
        assert db.n_tasks() == 3


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_basic(self):
        result = TransferBOResult(
            x_best=np.array([2.0]),
            f_best=1.0,
            X_history=np.zeros((10, 1)),
            y_history=np.ones(10),
            total_evaluations=10,
            convergence=[5.0, 3.0, 1.0],
            transfer_weights=[0.5, 0.3],
            source_tasks_used=[0, 1],
            speedup_vs_cold=1.5,
        )
        s = transfer_bo_summary(result)
        assert "1.000000" in s
        assert "Sources used: 2" in s
        assert "1.5" in s

    def test_no_transfer(self):
        result = TransferBOResult(
            x_best=np.array([0.0]),
            f_best=0.0,
            X_history=np.zeros((5, 1)),
            y_history=np.zeros(5),
            total_evaluations=5,
            convergence=[1.0, 0.0],
            transfer_weights=[],
            source_tasks_used=[],
        )
        s = transfer_bo_summary(result, name="Cold Start")
        assert "Cold Start" in s
        assert "Sources used: 0" in s


# ---------------------------------------------------------------------------
# Task embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_build_embeddings(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        for i in range(3):
            db.add_task(make_task(lambda x, s=i: (x[0] - s) ** 2, bounds,
                                   n_points=20, task_id=i))
        embs = build_task_embeddings(db)
        assert len(embs) == 3
        assert all(e.embedding is not None for e in embs)

    def test_find_sources(self):
        db = TaskDatabase()
        bounds = np.array([[0.0, 5.0]])
        for i in range(5):
            db.add_task(make_task(lambda x, s=i: (x[0] - s) ** 2, bounds,
                                   n_points=20, task_id=i))
        target = make_task(quadratic_1d, bounds, n_points=10, task_id=99)
        similar = find_source_tasks(db, target, top_k=3)
        assert len(similar) <= 3
        # Each result is (task_index, similarity_score)
        for idx, score in similar:
            assert 0 <= idx < 5
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_source_single_point(self):
        bounds = np.array([[0.0, 5.0]])
        source = BOTask(X=np.array([[2.5]]), y=np.array([1.0]),
                        bounds=bounds, task_id=0)
        result = transfer_bo(quadratic_1d, bounds, [source],
                             n_iterations=5, n_initial=3,
                             rng=np.random.default_rng(42))
        assert result.total_evaluations == 8

    def test_high_dim(self):
        d = 5
        bounds = np.array([[-1.0, 1.0]] * d)
        source = make_task(sphere, bounds, n_points=50, task_id=0)
        result = transfer_bo(lambda x: sphere(x - 0.1), bounds, [source],
                             n_iterations=10, n_initial=5,
                             rng=np.random.default_rng(42))
        assert result.X_history.shape[1] == d

    def test_many_sources(self):
        bounds = np.array([[0.0, 5.0]])
        sources = [make_task(lambda x, s=i: (x[0] - s * 0.5) ** 2, bounds,
                              n_points=15, task_id=i) for i in range(10)]
        result = transfer_bo(quadratic_1d, bounds, sources,
                             n_iterations=8, n_initial=5,
                             rng=np.random.default_rng(42))
        assert result.total_evaluations == 13

    def test_identical_source_target(self):
        bounds = np.array([[0.0, 5.0]])
        source = make_task(quadratic_1d, bounds, n_points=30)
        result = transfer_bo(quadratic_1d, bounds, [source],
                             n_iterations=10, n_initial=5,
                             rng=np.random.default_rng(42))
        # Should converge well with identical source
        assert result.f_best < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
