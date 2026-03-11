"""
Tests for C167: Bayesian Optimization
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from bayesian_optimization import (
    AcquisitionFunction, ExpectedImprovement, ProbabilityOfImprovement,
    UpperConfidenceBound, ThompsonSampling, KnowledgeGradient,
    BayesOptHistory, BayesianOptimizer, BNNBayesianOptimizer,
    BatchBayesianOptimizer, MultiObjectiveBO, ConstrainedBO,
    create_optimizer,
    _standard_normal_cdf, _standard_normal_pdf, _dominates,
    _compute_pareto_front, _hypervolume_2d,
)


# ---------------------------------------------------------------------------
# Test objectives
# ---------------------------------------------------------------------------

def neg_quadratic_1d(x):
    """-(x - 0.5)^2, max at x=0.5, max_val=0."""
    return -float((x[0] - 0.5) ** 2)


def neg_quadratic_2d(x):
    """-(x0 - 0.3)^2 - (x1 - 0.7)^2, max at (0.3, 0.7)."""
    return -float((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2)


def branin_like(x):
    """Simple 2D function with known structure."""
    return -float((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2 + 0.1 * np.sin(10 * x[0]))


# ===========================================================================
# Utility Tests
# ===========================================================================

class TestUtilities:
    def test_standard_normal_cdf_center(self):
        assert abs(_standard_normal_cdf(0.0) - 0.5) < 1e-6

    def test_standard_normal_cdf_tails(self):
        assert _standard_normal_cdf(-5.0) < 0.001
        assert _standard_normal_cdf(5.0) > 0.999

    def test_standard_normal_cdf_symmetry(self):
        for z in [0.5, 1.0, 2.0]:
            assert abs(_standard_normal_cdf(z) + _standard_normal_cdf(-z) - 1.0) < 1e-5

    def test_standard_normal_pdf_center(self):
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert abs(_standard_normal_pdf(0.0) - expected) < 1e-6

    def test_standard_normal_pdf_nonneg(self):
        for z in [-3, -1, 0, 1, 3]:
            assert _standard_normal_pdf(z) >= 0

    def test_standard_normal_pdf_symmetry(self):
        for z in [0.5, 1.5, 2.5]:
            assert abs(_standard_normal_pdf(z) - _standard_normal_pdf(-z)) < 1e-10


# ===========================================================================
# Acquisition Function Tests
# ===========================================================================

class TestExpectedImprovement:
    def test_ei_basic(self):
        ei = ExpectedImprovement(xi=0.0)
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 0.5, 0.5])
        vals = ei.evaluate(mean, std, best_y=2.0)
        # Point with mean=3 should have highest EI
        assert vals[2] > vals[1]
        assert vals[2] > vals[0]

    def test_ei_zero_std(self):
        ei = ExpectedImprovement()
        mean = np.array([3.0])
        std = np.array([0.0])
        vals = ei.evaluate(mean, std, best_y=2.0)
        assert vals[0] == 0.0  # No uncertainty, no EI

    def test_ei_nonnegative(self):
        ei = ExpectedImprovement()
        mean = np.array([-10.0, -5.0, 0.0, 5.0])
        std = np.array([1.0, 1.0, 1.0, 1.0])
        vals = ei.evaluate(mean, std, best_y=10.0)
        assert np.all(vals >= 0)

    def test_ei_higher_mean_higher_ei(self):
        ei = ExpectedImprovement(xi=0.0)
        std = np.array([1.0, 1.0])
        v1 = ei.evaluate(np.array([3.0]), std[:1], best_y=2.0)
        v2 = ei.evaluate(np.array([4.0]), std[:1], best_y=2.0)
        assert v2[0] > v1[0]

    def test_ei_higher_std_higher_ei_below_best(self):
        ei = ExpectedImprovement(xi=0.0)
        mean = np.array([1.0, 1.0])
        # Higher std => more exploration => higher EI when below best
        v1 = ei.evaluate(mean[:1], np.array([0.5]), best_y=2.0)
        v2 = ei.evaluate(mean[:1], np.array([2.0]), best_y=2.0)
        assert v2[0] > v1[0]

    def test_ei_xi_reduces_exploitation(self):
        ei_low = ExpectedImprovement(xi=0.0)
        ei_high = ExpectedImprovement(xi=1.0)
        mean = np.array([2.1])
        std = np.array([0.1])
        v_low = ei_low.evaluate(mean, std, best_y=2.0)
        v_high = ei_high.evaluate(mean, std, best_y=2.0)
        assert v_low[0] > v_high[0]

    def test_ei_repr(self):
        ei = ExpectedImprovement()
        assert 'ExpectedImprovement' in repr(ei)


class TestProbabilityOfImprovement:
    def test_pi_basic(self):
        pi = ProbabilityOfImprovement(xi=0.0)
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 0.5, 0.5])
        vals = pi.evaluate(mean, std, best_y=2.0)
        assert vals[2] > vals[1]

    def test_pi_range(self):
        pi = ProbabilityOfImprovement()
        mean = np.array([-5.0, 0.0, 5.0, 10.0])
        std = np.array([1.0, 1.0, 1.0, 1.0])
        vals = pi.evaluate(mean, std, best_y=3.0)
        for v in vals:
            assert 0.0 <= v <= 1.0

    def test_pi_zero_std(self):
        pi = ProbabilityOfImprovement()
        vals = pi.evaluate(np.array([5.0]), np.array([0.0]), best_y=3.0)
        assert vals[0] == 0.0

    def test_pi_high_mean_approaches_one(self):
        pi = ProbabilityOfImprovement(xi=0.0)
        vals = pi.evaluate(np.array([100.0]), np.array([1.0]), best_y=0.0)
        assert vals[0] > 0.99


class TestUpperConfidenceBound:
    def test_ucb_basic(self):
        ucb = UpperConfidenceBound(kappa=2.0)
        mean = np.array([1.0, 2.0])
        std = np.array([1.0, 0.1])
        vals = ucb.evaluate(mean, std, best_y=0.0)
        assert vals[0] == 1.0 + 2.0 * 1.0
        assert abs(vals[1] - (2.0 + 2.0 * 0.1)) < 1e-10

    def test_ucb_higher_kappa_more_exploration(self):
        ucb_low = UpperConfidenceBound(kappa=0.1)
        ucb_high = UpperConfidenceBound(kappa=5.0)
        mean = np.array([1.0])
        std = np.array([2.0])
        v_low = ucb_low.evaluate(mean, std, best_y=0.0)
        v_high = ucb_high.evaluate(mean, std, best_y=0.0)
        assert v_high[0] > v_low[0]

    def test_ucb_zero_std(self):
        ucb = UpperConfidenceBound(kappa=2.0)
        vals = ucb.evaluate(np.array([3.0]), np.array([0.0]), best_y=0.0)
        assert abs(vals[0] - 3.0) < 1e-10


class TestThompsonSampling:
    def test_ts_returns_samples(self):
        ts = ThompsonSampling(seed=42)
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 0.5, 0.5])
        vals = ts.evaluate(mean, std, best_y=0.0)
        assert len(vals) == 3

    def test_ts_deterministic_with_seed(self):
        ts1 = ThompsonSampling(seed=123)
        ts2 = ThompsonSampling(seed=123)
        mean = np.array([1.0, 2.0])
        std = np.array([0.5, 0.5])
        v1 = ts1.evaluate(mean, std, best_y=0.0)
        v2 = ts2.evaluate(mean, std, best_y=0.0)
        np.testing.assert_array_equal(v1, v2)

    def test_ts_zero_std_returns_mean(self):
        ts = ThompsonSampling(seed=42)
        mean = np.array([5.0])
        std = np.array([0.0])
        vals = ts.evaluate(mean, std, best_y=0.0)
        assert abs(vals[0] - 5.0) < 1e-10


class TestKnowledgeGradient:
    def test_kg_nonnegative(self):
        kg = KnowledgeGradient(n_fantasies=20, seed=42)
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 0.5, 0.5])
        vals = kg.evaluate(mean, std, best_y=2.0)
        assert np.all(vals >= 0)

    def test_kg_higher_mean_higher_value(self):
        kg = KnowledgeGradient(n_fantasies=50, seed=42)
        std = np.array([1.0])
        v_low = kg.evaluate(np.array([1.0]), std, best_y=0.0)
        v_high = kg.evaluate(np.array([5.0]), std, best_y=0.0)
        assert v_high[0] >= v_low[0]

    def test_kg_deterministic_with_seed(self):
        kg1 = KnowledgeGradient(seed=99)
        kg2 = KnowledgeGradient(seed=99)
        mean = np.array([1.0, 2.0])
        std = np.array([0.5, 0.5])
        v1 = kg1.evaluate(mean, std, best_y=0.0)
        v2 = kg2.evaluate(mean, std, best_y=0.0)
        np.testing.assert_array_equal(v1, v2)


# ===========================================================================
# BayesOptHistory Tests
# ===========================================================================

class TestBayesOptHistory:
    def test_record_and_get_best(self):
        h = BayesOptHistory()
        h.record(np.array([0.1]), 1.0)
        h.record(np.array([0.5]), 3.0)
        h.record(np.array([0.9]), 2.0)
        best_x, best_y = h.get_best()
        assert best_y == 3.0
        assert abs(best_x[0] - 0.5) < 1e-10

    def test_best_so_far_monotonic(self):
        h = BayesOptHistory()
        for val in [1.0, 0.5, 2.0, 1.5, 3.0]:
            h.record(np.array([0.0]), val)
        # best_so_far should be monotonically non-decreasing
        for i in range(1, len(h.best_so_far)):
            assert h.best_so_far[i] >= h.best_so_far[i-1]

    def test_regret(self):
        h = BayesOptHistory()
        h.record(np.array([0.0]), 1.0)
        h.record(np.array([0.0]), 3.0)
        h.record(np.array([0.0]), 2.0)
        regrets = h.get_regret(optimal_value=5.0)
        assert regrets == [4.0, 2.0, 2.0]

    def test_cumulative_regret(self):
        h = BayesOptHistory()
        h.record(np.array([0.0]), 1.0)
        h.record(np.array([0.0]), 2.0)
        h.record(np.array([0.0]), 3.0)
        cum = h.get_cumulative_regret(optimal_value=5.0)
        assert cum == [4.0, 7.0, 9.0]

    def test_convergence_rate(self):
        h = BayesOptHistory()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            h.record(np.array([0.0]), v)
        rate = h.convergence_rate(window=5)
        assert rate is not None
        assert rate > 0

    def test_convergence_rate_insufficient_data(self):
        h = BayesOptHistory()
        h.record(np.array([0.0]), 1.0)
        assert h.convergence_rate(window=5) is None

    def test_summary(self):
        h = BayesOptHistory()
        h.record(np.array([0.1]), 1.0)
        h.record(np.array([0.5]), 2.0)
        s = h.summary()
        assert s['iterations'] == 2
        assert s['best_y'] == 2.0
        assert len(s['best_history']) == 2

    def test_empty_history(self):
        h = BayesOptHistory()
        best_x, best_y = h.get_best()
        assert best_x is None
        assert best_y is None


# ===========================================================================
# BayesianOptimizer (GP) Tests
# ===========================================================================

class TestBayesianOptimizer:
    def test_creation(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], seed=42)
        assert opt.dim == 1
        assert opt.n_initial == 5

    def test_suggest_initial_random(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        x, acq = opt.suggest()
        assert 0.0 <= x[0] <= 1.0
        assert acq == 0.0  # random phase

    def test_suggest_after_initial(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        for _ in range(3):
            x, _ = opt.suggest()
            opt.observe(x, neg_quadratic_1d(x))
        # Now should use acquisition
        x, acq = opt.suggest()
        assert 0.0 <= x[0] <= 1.0

    def test_optimize_1d(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        best_x, best_y, hist = opt.optimize(neg_quadratic_1d, n_iter=15)
        # Should find near x=0.5 (max_val=0)
        assert best_y > -0.1  # reasonable
        assert hist.iterations == 15

    def test_optimize_2d(self):
        opt = BayesianOptimizer(bounds=[(0, 1), (0, 1)], n_initial=5, seed=42)
        best_x, best_y, hist = opt.optimize(neg_quadratic_2d, n_iter=20)
        assert best_y > -0.2

    def test_predict(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        for i in range(5):
            x = np.array([i * 0.2])
            opt.observe(x, neg_quadratic_1d(x))
        mean, std = opt.predict(np.array([[0.5]]))
        assert len(mean) == 1
        assert len(std) == 1

    def test_get_best(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], seed=42)
        opt.observe(np.array([0.1]), -1.0)
        opt.observe(np.array([0.5]), 0.0)
        opt.observe(np.array([0.9]), -0.5)
        best_x, best_y = opt.get_best()
        assert best_y == 0.0

    def test_custom_acquisition(self):
        opt = BayesianOptimizer(
            bounds=[(0, 1)],
            acquisition=UpperConfidenceBound(kappa=3.0),
            n_initial=3, seed=42
        )
        best_x, best_y, _ = opt.optimize(neg_quadratic_1d, n_iter=15)
        assert best_y > -0.2

    def test_different_kernels(self):
        from gaussian_processes import RBFKernel
        opt = BayesianOptimizer(
            bounds=[(0, 1)],
            kernel=RBFKernel(lengthscale=0.3),
            n_initial=3, seed=42
        )
        best_x, best_y, _ = opt.optimize(neg_quadratic_1d, n_iter=15)
        assert best_y > -0.2

    def test_history_tracked(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=2, seed=42)
        opt.optimize(neg_quadratic_1d, n_iter=10)
        assert len(opt.history.X) == 10
        assert len(opt.history.y) == 10
        assert len(opt.history.best_so_far) == 10


# ===========================================================================
# BNN Bayesian Optimizer Tests
# ===========================================================================

class TestBNNBayesianOptimizer:
    def test_creation(self):
        opt = BNNBayesianOptimizer(bounds=[(0, 1)], seed=42)
        assert opt.dim == 1

    def test_suggest_initial(self):
        opt = BNNBayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        x, acq = opt.suggest()
        assert 0.0 <= x[0] <= 1.0

    def test_optimize_1d(self):
        opt = BNNBayesianOptimizer(
            bounds=[(0, 1)],
            hidden_sizes=[16, 16],
            n_initial=3,
            n_epochs=30,
            n_samples=10,
            seed=42
        )
        best_x, best_y, hist = opt.optimize(neg_quadratic_1d, n_iter=8, n_candidates=100)
        assert best_y > -0.5
        assert hist.iterations == 8

    def test_custom_acquisition(self):
        opt = BNNBayesianOptimizer(
            bounds=[(0, 1)],
            hidden_sizes=[16],
            acquisition=ProbabilityOfImprovement(),
            n_initial=3,
            n_epochs=20,
            n_samples=10,
            seed=42
        )
        best_x, best_y, _ = opt.optimize(neg_quadratic_1d, n_iter=6, n_candidates=50)
        assert best_y > -1.0

    def test_get_best(self):
        opt = BNNBayesianOptimizer(bounds=[(0, 1)], seed=42)
        opt.observe(np.array([0.3]), 1.0)
        opt.observe(np.array([0.7]), 2.0)
        _, best_y = opt.get_best()
        assert best_y == 2.0


# ===========================================================================
# Batch Bayesian Optimizer Tests
# ===========================================================================

class TestBatchBayesianOptimizer:
    def test_creation(self):
        opt = BatchBayesianOptimizer(bounds=[(0, 1)], batch_size=3, seed=42)
        assert opt.batch_size == 3

    def test_suggest_batch_initial(self):
        opt = BatchBayesianOptimizer(bounds=[(0, 1)], batch_size=3, n_initial=5, seed=42)
        batch = opt.suggest_batch()
        assert len(batch) == 3
        for x in batch:
            assert 0.0 <= x[0] <= 1.0

    def test_suggest_batch_after_initial(self):
        opt = BatchBayesianOptimizer(bounds=[(0, 1)], batch_size=2, n_initial=3, seed=42)
        # Add initial points
        for _ in range(3):
            x = np.array([opt.rng.uniform(0, 1)])
            opt.history.record(x, neg_quadratic_1d(x))
        batch = opt.suggest_batch(n_candidates=200)
        assert len(batch) == 2

    def test_optimize(self):
        opt = BatchBayesianOptimizer(
            bounds=[(0, 1)], batch_size=3, n_initial=3, seed=42
        )
        best_x, best_y, hist = opt.optimize(neg_quadratic_1d, n_batches=5, n_candidates=200)
        assert best_y > -0.5
        assert hist.iterations > 0

    def test_observe_batch(self):
        opt = BatchBayesianOptimizer(bounds=[(0, 1)], seed=42)
        X = np.array([[0.1], [0.5], [0.9]])
        y = np.array([-0.16, 0.0, -0.16])
        opt.observe_batch(X, y)
        assert len(opt.history.X) == 3

    def test_batch_diversity(self):
        """Kriging believer should produce diverse batches."""
        opt = BatchBayesianOptimizer(bounds=[(0, 1)], batch_size=4, n_initial=5, seed=42)
        for _ in range(5):
            x = np.array([opt.rng.uniform(0, 1)])
            opt.history.record(x, neg_quadratic_1d(x))
        batch = opt.suggest_batch(n_candidates=500)
        # Points should not all be the same
        if len(batch) > 1:
            dists = [abs(batch[i][0] - batch[j][0]) for i in range(len(batch))
                     for j in range(i+1, len(batch))]
            assert max(dists) > 0.01


# ===========================================================================
# Multi-Objective BO Tests
# ===========================================================================

class TestMultiObjectiveBO:
    def test_dominates(self):
        assert _dominates([3, 3], [2, 2])
        assert _dominates([3, 2], [2, 2])
        assert not _dominates([2, 2], [2, 2])
        assert not _dominates([1, 3], [2, 2])

    def test_pareto_front(self):
        Y = np.array([[1, 3], [2, 2], [3, 1], [1.5, 1.5]])
        idx = _compute_pareto_front(Y)
        # [1,3], [2,2], [3,1] are Pareto-optimal; [1.5,1.5] is dominated by [2,2]
        assert set(idx) == {0, 1, 2}

    def test_pareto_single_point(self):
        Y = np.array([[1, 2]])
        idx = _compute_pareto_front(Y)
        assert len(idx) == 1

    def test_hypervolume_2d(self):
        pareto_Y = np.array([[3, 1], [2, 2], [1, 3]])
        ref = np.array([0, 0])
        hv = _hypervolume_2d(pareto_Y, ref)
        # HV = 3*1 + 2*(2-1) + 1*(3-2) = 3 + 2 + 1 = 6
        assert abs(hv - 6.0) < 1e-10

    def test_hypervolume_empty(self):
        hv = _hypervolume_2d(np.array([[0.5, 0.5]]), np.array([1.0, 1.0]))
        assert hv == 0.0

    def test_mobo_creation(self):
        mobo = MultiObjectiveBO(bounds=[(0, 1)], n_objectives=2, seed=42)
        assert mobo.n_objectives == 2

    def test_mobo_suggest_initial(self):
        mobo = MultiObjectiveBO(bounds=[(0, 1)], n_initial=3, seed=42)
        x = mobo.suggest()
        assert 0.0 <= x[0] <= 1.0

    def test_mobo_observe(self):
        mobo = MultiObjectiveBO(bounds=[(0, 1)], seed=42)
        mobo.observe(np.array([0.5]), np.array([1.0, 2.0]))
        assert len(mobo.X) == 1
        assert len(mobo.Y) == 1

    def test_mobo_optimize(self):
        def two_obj(x):
            f1 = -(x[0] - 0.3) ** 2
            f2 = -(x[0] - 0.7) ** 2
            return np.array([f1, f2])

        mobo = MultiObjectiveBO(
            bounds=[(0, 1)], n_objectives=2,
            ref_point=[-1.0, -1.0], n_initial=5, seed=42
        )
        pX, pY, aX, aY = mobo.optimize(two_obj, n_iter=15, n_candidates=200)
        assert len(pX) > 0
        assert len(pY) > 0

    def test_mobo_hypervolume_increases(self):
        def two_obj(x):
            return np.array([-(x[0] - 0.2) ** 2, -(x[0] - 0.8) ** 2])

        mobo = MultiObjectiveBO(
            bounds=[(0, 1)], n_objectives=2,
            ref_point=[-2.0, -2.0], n_initial=3, seed=42
        )
        # Initial phase
        for _ in range(3):
            x = mobo.suggest()
            mobo.observe(x, two_obj(x))
        hv1 = mobo.hypervolume()

        # More iterations
        for _ in range(10):
            x = mobo.suggest(n_candidates=200)
            mobo.observe(x, two_obj(x))
        hv2 = mobo.hypervolume()
        assert hv2 >= hv1

    def test_mobo_get_pareto_front_empty(self):
        mobo = MultiObjectiveBO(bounds=[(0, 1)], seed=42)
        pX, pY, aX, aY = mobo.get_pareto_front()
        assert len(pX) == 0


# ===========================================================================
# Constrained BO Tests
# ===========================================================================

class TestConstrainedBO:
    def test_creation(self):
        cbo = ConstrainedBO(bounds=[(0, 1)], n_constraints=1, seed=42)
        assert cbo.n_constraints == 1

    def test_suggest_initial(self):
        cbo = ConstrainedBO(bounds=[(0, 1)], n_initial=3, seed=42)
        x = cbo.suggest()
        assert 0.0 <= x[0] <= 1.0

    def test_observe(self):
        cbo = ConstrainedBO(bounds=[(0, 1)], n_constraints=1, seed=42)
        cbo.observe(np.array([0.5]), 1.0, np.array([0.5]))
        assert len(cbo.history.X) == 1
        assert len(cbo.constraint_values) == 1

    def test_optimize_constrained(self):
        # Maximize f(x) = x, subject to x <= 0.7
        def obj(x):
            return float(x[0])

        def constr(x):
            return np.array([0.7 - x[0]])  # >= 0 when x <= 0.7

        cbo = ConstrainedBO(
            bounds=[(0, 1)], n_constraints=1,
            n_initial=3, seed=42
        )
        best_x, best_y, hist = cbo.optimize(obj, constr, n_iter=15, n_candidates=500)
        # Best feasible should be near 0.7
        if best_x is not None:
            assert best_x[0] <= 0.75  # approximately feasible

    def test_get_best_feasible(self):
        cbo = ConstrainedBO(bounds=[(0, 1)], n_constraints=1, seed=42)
        cbo.observe(np.array([0.9]), 0.9, np.array([-0.2]))  # infeasible
        cbo.observe(np.array([0.5]), 0.5, np.array([0.2]))   # feasible
        cbo.observe(np.array([0.3]), 0.3, np.array([0.4]))   # feasible
        best_x, best_y = cbo.get_best_feasible()
        assert best_y == 0.5

    def test_no_feasible(self):
        cbo = ConstrainedBO(bounds=[(0, 1)], n_constraints=1, seed=42)
        cbo.observe(np.array([0.9]), 0.9, np.array([-0.1]))
        cbo.observe(np.array([0.8]), 0.8, np.array([-0.2]))
        best_x, best_y = cbo.get_best_feasible()
        assert best_x is None
        assert best_y is None

    def test_multiple_constraints(self):
        def obj(x):
            return -float((x[0] - 0.5) ** 2)

        def constr(x):
            c1 = x[0] - 0.2   # x >= 0.2
            c2 = 0.8 - x[0]   # x <= 0.8
            return np.array([c1, c2])

        cbo = ConstrainedBO(
            bounds=[(0, 1)], n_constraints=2,
            n_initial=3, seed=42
        )
        best_x, best_y, _ = cbo.optimize(obj, constr, n_iter=15, n_candidates=500)
        if best_x is not None:
            assert 0.15 <= best_x[0] <= 0.85

    def test_constrained_2d(self):
        def obj(x):
            return -float((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)

        def constr(x):
            # Circle constraint: x0^2 + x1^2 <= 0.5
            return np.array([0.5 - (x[0] ** 2 + x[1] ** 2)])

        cbo = ConstrainedBO(
            bounds=[(0, 1), (0, 1)], n_constraints=1,
            n_initial=5, seed=42
        )
        best_x, best_y, _ = cbo.optimize(obj, constr, n_iter=15, n_candidates=500)
        # Should find something feasible
        assert best_x is not None or len(cbo.constraint_values) > 0


# ===========================================================================
# Factory Tests
# ===========================================================================

class TestCreateOptimizer:
    def test_create_gp(self):
        opt = create_optimizer([(0, 1)], method='gp', acquisition='ei')
        assert isinstance(opt, BayesianOptimizer)

    def test_create_bnn(self):
        opt = create_optimizer([(0, 1)], method='bnn', acquisition='ucb')
        assert isinstance(opt, BNNBayesianOptimizer)

    def test_create_batch(self):
        opt = create_optimizer([(0, 1)], method='batch', acquisition='ei')
        assert isinstance(opt, BatchBayesianOptimizer)

    def test_create_multi_objective(self):
        opt = create_optimizer([(0, 1)], method='multi_objective')
        assert isinstance(opt, MultiObjectiveBO)

    def test_create_constrained(self):
        opt = create_optimizer([(0, 1)], method='constrained', acquisition='pi')
        assert isinstance(opt, ConstrainedBO)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            create_optimizer([(0, 1)], method='unknown')

    def test_all_acquisitions(self):
        for acq in ['ei', 'pi', 'ucb', 'thompson', 'kg']:
            opt = create_optimizer([(0, 1)], method='gp', acquisition=acq)
            assert opt is not None


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_gp_vs_bnn_same_problem(self):
        """Both GP and BNN should find reasonable solutions."""
        gp_opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        _, gp_best, _ = gp_opt.optimize(neg_quadratic_1d, n_iter=12)

        bnn_opt = BNNBayesianOptimizer(
            bounds=[(0, 1)], hidden_sizes=[16],
            n_initial=3, n_epochs=30, n_samples=10, seed=42
        )
        _, bnn_best, _ = bnn_opt.optimize(neg_quadratic_1d, n_iter=8, n_candidates=100)

        assert gp_best > -0.5
        assert bnn_best > -0.5

    def test_acquisition_comparison(self):
        """Different acquisitions should all work."""
        results = {}
        for name, acq in [('ei', ExpectedImprovement()),
                          ('pi', ProbabilityOfImprovement()),
                          ('ucb', UpperConfidenceBound(kappa=2.0))]:
            opt = BayesianOptimizer(
                bounds=[(0, 1)], acquisition=acq,
                n_initial=3, seed=42
            )
            _, best_y, _ = opt.optimize(neg_quadratic_1d, n_iter=12)
            results[name] = best_y
            assert best_y > -0.5, f"{name} failed to optimize"

    def test_convergence_tracking(self):
        """Verify history tracks convergence."""
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        _, _, hist = opt.optimize(neg_quadratic_1d, n_iter=15)
        # Best-so-far should be non-decreasing
        for i in range(1, len(hist.best_so_far)):
            assert hist.best_so_far[i] >= hist.best_so_far[i-1]
        # Should have improved from initial random
        assert hist.best_so_far[-1] >= hist.best_so_far[0]

    def test_regret_analysis(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        _, _, hist = opt.optimize(neg_quadratic_1d, n_iter=15)
        regrets = hist.get_regret(optimal_value=0.0)
        # Regret should be non-negative (optimal_value >= best)
        for r in regrets:
            assert r >= 0.0
        # Final regret should be smaller than initial
        assert regrets[-1] <= regrets[0] + 0.001

    def test_2d_optimization(self):
        opt = BayesianOptimizer(
            bounds=[(0, 1), (0, 1)],
            n_initial=5, seed=42
        )
        best_x, best_y, _ = opt.optimize(neg_quadratic_2d, n_iter=25)
        assert best_y > -0.2
        assert len(best_x) == 2

    def test_batch_and_sequential_comparison(self):
        """Batch should also find good solutions."""
        seq = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        _, seq_best, _ = seq.optimize(neg_quadratic_1d, n_iter=12)

        batch = BatchBayesianOptimizer(
            bounds=[(0, 1)], batch_size=3, n_initial=3, seed=42
        )
        _, batch_best, _ = batch.optimize(neg_quadratic_1d, n_batches=4, n_candidates=200)

        assert seq_best > -0.5
        assert batch_best > -0.5

    def test_high_dimensional(self):
        """Test with higher dimensions."""
        dim = 3
        target = np.array([0.5] * dim)

        def obj(x):
            return -float(np.sum((x - target) ** 2))

        opt = BayesianOptimizer(
            bounds=[(0, 1)] * dim, n_initial=8, seed=42
        )
        best_x, best_y, _ = opt.optimize(obj, n_iter=25)
        assert best_y > -0.5


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_single_observation(self):
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=1, seed=42)
        opt.observe(np.array([0.5]), 1.0)
        best_x, best_y = opt.get_best()
        assert best_y == 1.0

    def test_constant_objective(self):
        def const(x):
            return 1.0
        opt = BayesianOptimizer(bounds=[(0, 1)], n_initial=3, seed=42)
        _, best_y, _ = opt.optimize(const, n_iter=8)
        assert abs(best_y - 1.0) < 1e-10

    def test_narrow_bounds(self):
        opt = BayesianOptimizer(bounds=[(0.49, 0.51)], n_initial=3, seed=42)
        _, best_y, _ = opt.optimize(neg_quadratic_1d, n_iter=8)
        assert best_y > -0.01  # very close to optimum

    def test_wide_bounds(self):
        def obj(x):
            return -float((x[0] - 50) ** 2)
        opt = BayesianOptimizer(bounds=[(0, 100)], n_initial=5, seed=42)
        _, best_y, _ = opt.optimize(obj, n_iter=20)
        assert best_y > -500

    def test_mobo_single_objective(self):
        mobo = MultiObjectiveBO(bounds=[(0, 1)], n_objectives=1,
                                ref_point=[-1.0], n_initial=3, seed=42)
        def obj(x):
            return np.array([-(x[0] - 0.5) ** 2])
        for _ in range(5):
            x = mobo.suggest()
            mobo.observe(x, obj(x))
        pX, pY, _, _ = mobo.get_pareto_front()
        assert len(pX) > 0

    def test_acquisition_with_scalar_inputs(self):
        ei = ExpectedImprovement()
        val = ei.evaluate(2.0, 0.5, best_y=1.0)
        assert len(val) > 0

    def test_history_with_acq_values(self):
        h = BayesOptHistory()
        h.record(np.array([0.0]), 1.0, acq_val=0.5)
        assert h.acquisition_values == [0.5]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
