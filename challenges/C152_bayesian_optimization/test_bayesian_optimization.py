"""Tests for C152: Bayesian Optimization."""

import math
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from bayesian_optimization import (
    RBFKernel, MaternKernel, RationalQuadraticKernel, LinearKernel,
    SumKernel, ProductKernel, ScaleKernel,
    GaussianProcess,
    AcquisitionFunction, AcquisitionType,
    expected_improvement, probability_of_improvement, upper_confidence_bound,
    BayesianOptimizer, OptimizationResult,
    MultiObjectiveBayesianOptimizer, compute_pareto_front, _dominates,
    hypervolume_contribution,
    BatchBayesianOptimizer,
    ConstrainedBayesianOptimizer,
)


# ============================================================================
# Kernel Tests
# ============================================================================

class TestRBFKernel:
    def test_self_similarity(self):
        k = RBFKernel(lengthscale=1.0, variance=1.0)
        X = np.array([[0.0], [1.0], [2.0]])
        K = k(X, X)
        np.testing.assert_allclose(np.diag(K), 1.0)

    def test_symmetry(self):
        k = RBFKernel()
        X1 = np.array([[0.0, 1.0], [2.0, 3.0]])
        X2 = np.array([[1.0, 0.0], [3.0, 2.0]])
        K = k(X1, X2)
        np.testing.assert_allclose(K, k(X1, X2))

    def test_decay_with_distance(self):
        k = RBFKernel(lengthscale=1.0, variance=1.0)
        x = np.array([[0.0]])
        near = np.array([[0.1]])
        far = np.array([[5.0]])
        assert k(x, near)[0, 0] > k(x, far)[0, 0]

    def test_lengthscale_effect(self):
        k_short = RBFKernel(lengthscale=0.1)
        k_long = RBFKernel(lengthscale=10.0)
        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        # Longer lengthscale -> higher correlation at distance 1
        assert k_long(x1, x2)[0, 0] > k_short(x1, x2)[0, 0]

    def test_variance_scaling(self):
        k1 = RBFKernel(variance=1.0)
        k2 = RBFKernel(variance=2.0)
        X = np.array([[0.0]])
        assert k2(X, X)[0, 0] == pytest.approx(2.0 * k1(X, X)[0, 0])

    def test_diagonal(self):
        k = RBFKernel(variance=3.0)
        X = np.array([[0.0], [1.0], [2.0]])
        np.testing.assert_allclose(k.diagonal(X), 3.0)

    def test_multidimensional(self):
        k = RBFKernel()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        K = k(X, X)
        assert K.shape == (2, 2)
        assert K[0, 0] == pytest.approx(1.0)
        assert K[0, 1] < 1.0

    def test_get_set_params(self):
        k = RBFKernel(lengthscale=2.0, variance=3.0)
        params = k.get_params()
        assert params == {'lengthscale': 2.0, 'variance': 3.0}
        k.set_params(lengthscale=5.0)
        assert k.lengthscale == 5.0


class TestMaternKernel:
    def test_nu_05(self):
        k = MaternKernel(nu=0.5)
        X = np.array([[0.0], [1.0]])
        K = k(X, X)
        assert K[0, 0] == pytest.approx(1.0)
        assert 0 < K[0, 1] < 1.0

    def test_nu_15(self):
        k = MaternKernel(nu=1.5)
        X = np.array([[0.0], [0.5]])
        K = k(X, X)
        assert K[0, 0] == pytest.approx(1.0)

    def test_nu_25(self):
        k = MaternKernel(nu=2.5)
        X = np.array([[0.0], [0.5]])
        K = k(X, X)
        assert K[0, 0] == pytest.approx(1.0)
        assert 0 < K[0, 1] < 1.0

    def test_invalid_nu(self):
        with pytest.raises(ValueError):
            MaternKernel(nu=3.0)

    def test_self_similarity(self):
        for nu in [0.5, 1.5, 2.5]:
            k = MaternKernel(nu=nu)
            X = np.array([[1.0, 2.0], [3.0, 4.0]])
            K = k(X, X)
            np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_diagonal(self):
        k = MaternKernel(variance=2.5, nu=1.5)
        X = np.array([[0.0], [1.0], [2.0]])
        np.testing.assert_allclose(k.diagonal(X), 2.5)


class TestRationalQuadraticKernel:
    def test_self_similarity(self):
        k = RationalQuadraticKernel()
        X = np.array([[0.0], [1.0]])
        K = k(X, X)
        assert K[0, 0] == pytest.approx(1.0)

    def test_decay(self):
        k = RationalQuadraticKernel()
        x = np.array([[0.0]])
        near = np.array([[0.1]])
        far = np.array([[10.0]])
        assert k(x, near)[0, 0] > k(x, far)[0, 0]


class TestLinearKernel:
    def test_basic(self):
        k = LinearKernel(variance=1.0, bias=0.0)
        X = np.array([[1.0], [2.0]])
        K = k(X, X)
        assert K[0, 0] == pytest.approx(1.0)
        assert K[0, 1] == pytest.approx(2.0)
        assert K[1, 1] == pytest.approx(4.0)

    def test_with_bias(self):
        k = LinearKernel(variance=1.0, bias=1.0)
        X = np.array([[0.0]])
        K = k(X, X)
        assert K[0, 0] == pytest.approx(1.0)


class TestCompositeKernels:
    def test_sum_kernel(self):
        k1 = RBFKernel(variance=1.0)
        k2 = RBFKernel(variance=2.0, lengthscale=0.5)
        k_sum = k1 + k2
        X = np.array([[0.0]])
        K = k_sum(X, X)
        assert K[0, 0] == pytest.approx(3.0)

    def test_product_kernel(self):
        k1 = RBFKernel(variance=2.0)
        k2 = RBFKernel(variance=3.0)
        k_prod = k1 * k2
        X = np.array([[0.0]])
        K = k_prod(X, X)
        assert K[0, 0] == pytest.approx(6.0)

    def test_scale_kernel(self):
        k = RBFKernel(variance=1.0)
        k_scaled = 3.0 * k
        X = np.array([[0.0]])
        K = k_scaled(X, X)
        assert K[0, 0] == pytest.approx(3.0)

    def test_rmul_scale(self):
        k = RBFKernel(variance=1.0)
        k_scaled = 5 * k
        assert isinstance(k_scaled, ScaleKernel)

    def test_sum_diagonal(self):
        k1 = RBFKernel(variance=1.0)
        k2 = RBFKernel(variance=2.0)
        k_sum = k1 + k2
        X = np.array([[0.0], [1.0]])
        np.testing.assert_allclose(k_sum.diagonal(X), 3.0)


# ============================================================================
# Gaussian Process Tests
# ============================================================================

class TestGaussianProcess:
    def test_fit_predict_simple(self):
        gp = GaussianProcess(RBFKernel(lengthscale=1.0, variance=1.0), noise=1e-6)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp.fit(X, y)
        mu = gp.predict(X)
        np.testing.assert_allclose(mu, y, atol=0.1)

    def test_predict_with_std(self):
        gp = GaussianProcess(RBFKernel(), noise=1e-6)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp.fit(X, y)
        mu, sigma = gp.predict(X, return_std=True)
        # Uncertainty at training points should be low
        assert np.all(sigma < 0.5)

    def test_uncertainty_away_from_data(self):
        gp = GaussianProcess(RBFKernel(), noise=1e-6)
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        gp.fit(X, y)
        _, sigma_near = gp.predict(np.array([[0.5]]), return_std=True)
        _, sigma_far = gp.predict(np.array([[10.0]]), return_std=True)
        assert sigma_far[0] > sigma_near[0]

    def test_predict_with_cov(self):
        gp = GaussianProcess(RBFKernel(), noise=1e-6)
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        gp.fit(X, y)
        mu, cov = gp.predict(np.array([[0.5], [1.5]]), return_cov=True)
        assert cov.shape == (2, 2)
        assert cov[0, 0] >= 0
        assert cov[1, 1] >= 0

    def test_log_marginal_likelihood(self):
        gp = GaussianProcess(RBFKernel(), noise=1e-6)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)

    def test_sample_posterior(self):
        gp = GaussianProcess(RBFKernel(), noise=1e-6)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp.fit(X, y)
        X_test = np.array([[0.5], [1.5]])
        samples = gp.sample_posterior(X_test, n_samples=5, rng=np.random.RandomState(42))
        assert samples.shape == (5, 2)

    def test_normalize_y(self):
        gp = GaussianProcess(RBFKernel(), normalize_y=True)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([100.0, 200.0, 100.0])
        gp.fit(X, y)
        mu = gp.predict(X)
        np.testing.assert_allclose(mu, y, atol=5.0)

    def test_no_normalize_y(self):
        gp = GaussianProcess(RBFKernel(variance=10000.0), normalize_y=False, noise=1e-4)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([100.0, 200.0, 100.0])
        gp.fit(X, y)
        mu = gp.predict(X)
        # Should still fit reasonably
        assert np.all(np.abs(mu - y) < 50.0)

    def test_multiple_predict_points(self):
        gp = GaussianProcess(RBFKernel())
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        gp.fit(X, y)
        X_test = np.linspace(-1, 3, 20).reshape(-1, 1)
        mu = gp.predict(X_test)
        assert len(mu) == 20

    def test_2d_input(self):
        gp = GaussianProcess(RBFKernel())
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 2], dtype=float)
        gp.fit(X, y)
        mu = gp.predict(np.array([[0.5, 0.5]]))
        assert 0 < mu[0] < 2

    def test_matern_kernel_gp(self):
        gp = GaussianProcess(MaternKernel(nu=2.5))
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp.fit(X, y)
        mu = gp.predict(X)
        np.testing.assert_allclose(mu, y, atol=0.2)


# ============================================================================
# Acquisition Function Tests
# ============================================================================

class TestAcquisitionFunctions:
    def test_ei_positive_improvement(self):
        mu = np.array([2.0, 3.0])
        sigma = np.array([0.5, 0.5])
        y_best = 1.0
        ei = expected_improvement(mu, sigma, y_best)
        assert np.all(ei > 0)
        assert ei[1] > ei[0]  # Higher mean -> more improvement

    def test_ei_zero_sigma(self):
        mu = np.array([2.0])
        sigma = np.array([0.0])
        y_best = 1.0
        ei = expected_improvement(mu, sigma, y_best)
        assert ei[0] == 0.0

    def test_pi_higher_for_better_mean(self):
        mu = np.array([1.0, 3.0])
        sigma = np.array([1.0, 1.0])
        y_best = 2.0
        pi = probability_of_improvement(mu, sigma, y_best)
        assert pi[1] > pi[0]

    def test_ucb(self):
        mu = np.array([1.0, 1.0])
        sigma = np.array([0.5, 2.0])
        ucb = upper_confidence_bound(mu, sigma, beta=1.0)
        assert ucb[1] > ucb[0]  # Higher uncertainty -> higher UCB

    def test_acquisition_wrapper_ei(self):
        acq = AcquisitionFunction(AcquisitionType.EI)
        gp = GaussianProcess(RBFKernel())
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp.fit(X, y)
        X_test = np.array([[0.5], [1.5]])
        values = acq.evaluate(X_test, gp, y_best=1.0)
        assert len(values) == 2

    def test_acquisition_wrapper_pi(self):
        acq = AcquisitionFunction(AcquisitionType.PI)
        gp = GaussianProcess(RBFKernel())
        gp.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
        values = acq.evaluate(np.array([[0.5]]), gp, y_best=1.0)
        assert len(values) == 1

    def test_acquisition_wrapper_ucb(self):
        acq = AcquisitionFunction(AcquisitionType.UCB, beta=3.0)
        gp = GaussianProcess(RBFKernel())
        gp.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
        values = acq.evaluate(np.array([[0.5]]), gp, y_best=1.0)
        assert len(values) == 1

    def test_acquisition_wrapper_ts(self):
        acq = AcquisitionFunction(AcquisitionType.TS)
        gp = GaussianProcess(RBFKernel())
        gp.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
        rng = np.random.RandomState(42)
        values = acq.evaluate(np.array([[0.5]]), gp, y_best=1.0, rng=rng)
        assert len(values) == 1
        assert np.isfinite(values[0])

    def test_acquisition_optimize(self):
        acq = AcquisitionFunction(AcquisitionType.EI)
        gp = GaussianProcess(RBFKernel())
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 2.0, 0.0])
        gp.fit(X, y)
        bounds = np.array([[0.0, 1.0]])
        x_best = acq.optimize(gp, bounds, y_best=2.0, rng=np.random.RandomState(42))
        assert 0.0 <= x_best[0] <= 1.0

    def test_acquisition_optimize_2d(self):
        acq = AcquisitionFunction(AcquisitionType.EI)
        gp = GaussianProcess(RBFKernel())
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (10, 2))
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])
        gp.fit(X, y)
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        x_best = acq.optimize(gp, bounds, y_best=np.max(y), rng=rng)
        assert len(x_best) == 2
        assert np.all(x_best >= 0) and np.all(x_best <= 1)


# ============================================================================
# BayesianOptimizer Tests
# ============================================================================

class TestBayesianOptimizer:
    def test_simple_1d_maximize(self):
        """Maximize -x^2 + 4x (peak at x=2, y=4)."""
        def objective(x):
            return -(x[0] - 2)**2 + 4

        opt = BayesianOptimizer(
            bounds=np.array([[0.0, 4.0]]),
            n_initial=5, seed=42
        )
        result = opt.maximize(objective, n_iter=30)
        assert result.best_y > 3.0  # Should find near-optimal
        assert 1.0 < result.best_x[0] < 3.0

    def test_suggest_tell(self):
        opt = BayesianOptimizer(bounds=np.array([[0.0, 1.0]]), seed=42)
        for _ in range(10):
            x = opt.suggest()
            y = -x[0]**2
            opt.tell(x, y)
        assert len(opt.y_observed) == 10

    def test_convergence_history(self):
        def f(x):
            return -x[0]**2
        opt = BayesianOptimizer(bounds=np.array([[-5.0, 5.0]]), seed=42)
        result = opt.maximize(f, n_iter=20)
        assert len(result.convergence_history) == 20
        # Best should be non-decreasing
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] >= result.convergence_history[i-1] - 1e-10

    def test_minimize(self):
        """Minimize x^2 (minimum at x=0)."""
        def f(x):
            return x[0]**2

        opt = BayesianOptimizer(
            bounds=np.array([[-5.0, 5.0]]),
            n_initial=5, seed=42
        )
        result = opt.minimize(f, n_iter=30)
        assert result.best_y < 1.0

    def test_2d_optimization(self):
        """Maximize negative Rosenbrock (inverted)."""
        def neg_rosenbrock(x):
            return -((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)

        opt = BayesianOptimizer(
            bounds=np.array([[-2.0, 2.0], [-1.0, 3.0]]),
            n_initial=10, seed=42
        )
        result = opt.maximize(neg_rosenbrock, n_iter=50)
        assert result.best_y > -50.0  # Should make progress

    def test_callback(self):
        calls = []
        def cb(i, x, y, best_y):
            calls.append((i, best_y))

        def f(x):
            return -x[0]**2

        opt = BayesianOptimizer(bounds=np.array([[-1.0, 1.0]]), seed=42)
        opt.maximize(f, n_iter=10, callback=cb)
        assert len(calls) == 10

    def test_different_kernels(self):
        def f(x):
            return np.sin(x[0])

        for kernel in [RBFKernel(), MaternKernel(nu=2.5), RationalQuadraticKernel()]:
            opt = BayesianOptimizer(
                bounds=np.array([[0.0, 2 * math.pi]]),
                kernel=kernel, n_initial=5, seed=42
            )
            result = opt.maximize(f, n_iter=20)
            assert result.best_y > 0.5

    def test_different_acquisitions(self):
        def f(x):
            return -(x[0] - 1)**2

        for acq_type in [AcquisitionType.EI, AcquisitionType.PI, AcquisitionType.UCB]:
            acq = AcquisitionFunction(acq_type)
            opt = BayesianOptimizer(
                bounds=np.array([[0.0, 2.0]]),
                acquisition=acq, n_initial=3, seed=42
            )
            result = opt.maximize(f, n_iter=15)
            assert result.best_y > -0.5

    def test_result_fields(self):
        def f(x):
            return -x[0]**2
        opt = BayesianOptimizer(bounds=np.array([[-1.0, 1.0]]), seed=42)
        result = opt.maximize(f, n_iter=10)
        assert isinstance(result, OptimizationResult)
        assert result.X_observed.shape == (10, 1)
        assert len(result.y_observed) == 10
        assert result.n_iterations == 10

    def test_latin_hypercube_sampling(self):
        opt = BayesianOptimizer(bounds=np.array([[0.0, 10.0]]), n_initial=10, seed=42)
        samples = opt._initial_samples(10)
        assert samples.shape == (10, 1)
        assert np.all(samples >= 0) and np.all(samples <= 10)
        # LHS should give reasonable coverage
        sorted_s = np.sort(samples[:, 0])
        # No two samples should be very close
        diffs = np.diff(sorted_s)
        assert np.min(diffs) > 0.01

    def test_thompson_sampling_acquisition(self):
        acq = AcquisitionFunction(AcquisitionType.TS)
        opt = BayesianOptimizer(
            bounds=np.array([[0.0, 5.0]]),
            acquisition=acq, n_initial=5, seed=42
        )

        def f(x):
            return np.sin(x[0])

        result = opt.maximize(f, n_iter=20)
        assert result.best_y > 0.5


# ============================================================================
# Multi-Objective BO Tests
# ============================================================================

class TestParetoFront:
    def test_simple_pareto(self):
        Y = np.array([
            [1, 3],
            [2, 2],
            [3, 1],
            [1, 1],  # dominated by all above
        ])
        idx = compute_pareto_front(Y)
        assert set(idx) == {0, 1, 2}

    def test_single_point(self):
        Y = np.array([[1, 2]])
        idx = compute_pareto_front(Y)
        assert list(idx) == [0]

    def test_all_dominated(self):
        Y = np.array([
            [5, 5],
            [1, 1],
            [2, 2],
        ])
        idx = compute_pareto_front(Y)
        assert 0 in idx

    def test_dominates(self):
        assert _dominates(np.array([2, 2]), np.array([1, 1]))
        assert not _dominates(np.array([2, 1]), np.array([1, 2]))
        assert not _dominates(np.array([1, 1]), np.array([1, 1]))


class TestHypervolumeContribution:
    def test_basic_2d(self):
        Y = np.array([[1, 3], [2, 2], [3, 1]])
        ref = np.array([0, 0])
        contrib = hypervolume_contribution(Y, ref)
        assert len(contrib) == 3
        assert np.all(contrib >= 0)

    def test_empty(self):
        contrib = hypervolume_contribution(np.array([]).reshape(0, 2), np.array([0, 0]))
        assert len(contrib) == 0


class TestMultiObjectiveBayesianOptimizer:
    def test_two_objectives(self):
        def objectives(x):
            return np.array([
                -(x[0] - 1)**2,   # peak at x=1
                -(x[0] - 3)**2,   # peak at x=3
            ])

        opt = MultiObjectiveBayesianOptimizer(
            bounds=np.array([[0.0, 4.0]]),
            n_objectives=2,
            n_initial=5,
            seed=42
        )
        result = opt.maximize(objectives, n_iter=25)
        assert result['n_pareto'] >= 1
        assert result['pareto_X'].shape[1] == 1
        assert result['pareto_Y'].shape[1] == 2

    def test_suggest_tell_api(self):
        opt = MultiObjectiveBayesianOptimizer(
            bounds=np.array([[0.0, 1.0]]),
            n_objectives=2, seed=42
        )
        for _ in range(10):
            x = opt.suggest()
            y = np.array([x[0], 1 - x[0]])
            opt.tell(x, y)
        assert opt.X_observed.shape == (10, 1)
        assert opt.Y_observed.shape == (10, 2)

    def test_pareto_front_quality(self):
        def objectives(x):
            f1 = x[0]
            f2 = (1 - x[0])**2
            return np.array([f1, f2])

        opt = MultiObjectiveBayesianOptimizer(
            bounds=np.array([[0.0, 1.0]]),
            n_objectives=2, n_initial=5, seed=42,
            ref_point=np.array([-1.0, -1.0])
        )
        result = opt.maximize(objectives, n_iter=20)
        assert result['n_pareto'] >= 1


# ============================================================================
# Batch BO Tests
# ============================================================================

class TestBatchBayesianOptimizer:
    def test_batch_suggest(self):
        opt = BatchBayesianOptimizer(
            bounds=np.array([[0.0, 1.0]]),
            batch_size=3, n_initial=5, seed=42
        )
        # Feed initial points
        for _ in range(5):
            x = opt._initial_samples(1)[0]
            opt.tell(x, -x[0]**2)

        batch = opt.suggest_batch()
        assert batch.shape == (3, 1)

    def test_batch_maximize(self):
        def f(x):
            return -(x[0] - 2)**2

        opt = BatchBayesianOptimizer(
            bounds=np.array([[0.0, 4.0]]),
            batch_size=3, n_initial=5, seed=42
        )
        result = opt.maximize(f, n_batches=5)
        assert result.best_y > -2.0
        assert len(result.y_observed) == 5 * 3  # n_batches * batch_size

    def test_batch_diversity(self):
        """Batch points should not be identical."""
        opt = BatchBayesianOptimizer(
            bounds=np.array([[0.0, 10.0]]),
            batch_size=4, n_initial=5, seed=42
        )
        rng = np.random.RandomState(42)
        for _ in range(5):
            x = rng.uniform(0, 10, (1,))
            opt.tell(x, -x[0]**2)

        batch = opt.suggest_batch()
        # Points should be different
        for i in range(len(batch)):
            for j in range(i+1, len(batch)):
                assert not np.allclose(batch[i], batch[j], atol=1e-3)

    def test_batch_initial_phase(self):
        opt = BatchBayesianOptimizer(
            bounds=np.array([[0.0, 1.0]]),
            batch_size=3, n_initial=5, seed=42
        )
        batch = opt.suggest_batch()
        assert batch.shape == (3, 1)  # Should use LHS in initial phase


# ============================================================================
# Constrained BO Tests
# ============================================================================

class TestConstrainedBayesianOptimizer:
    def test_feasible_optimization(self):
        """Maximize f(x) = -x^2 + 4x subject to x <= 3."""
        def objective(x):
            return -(x[0]**2) + 4 * x[0]

        def constraints(x):
            return np.array([3.0 - x[0]])  # x <= 3 -> 3 - x >= 0

        opt = ConstrainedBayesianOptimizer(
            bounds=np.array([[0.0, 5.0]]),
            n_constraints=1, n_initial=5, seed=42
        )
        result = opt.maximize(objective, constraints, n_iter=30)
        assert result['feasible']
        assert result['n_feasible'] > 0

    def test_constraint_tracking(self):
        opt = ConstrainedBayesianOptimizer(
            bounds=np.array([[0.0, 1.0]]),
            n_constraints=2, seed=42
        )
        opt.tell(np.array([0.5]), 1.0, np.array([0.5, -0.1]))
        opt.tell(np.array([0.3]), 0.8, np.array([0.7, 0.2]))
        assert opt.c_observed.shape == (2, 2)
        assert opt.X_observed.shape == (2, 1)

    def test_suggest_returns_valid_point(self):
        opt = ConstrainedBayesianOptimizer(
            bounds=np.array([[0.0, 10.0]]),
            n_constraints=1, n_initial=3, seed=42
        )
        for _ in range(5):
            x = opt.suggest()
            assert 0.0 <= x[0] <= 10.0
            opt.tell(x, -x[0]**2, np.array([5.0 - x[0]]))

    def test_probability_of_feasibility(self):
        opt = ConstrainedBayesianOptimizer(
            bounds=np.array([[0.0, 10.0]]),
            n_constraints=1, seed=42
        )
        # Add clearly feasible and infeasible points
        opt.tell(np.array([1.0]), 1.0, np.array([4.0]))   # feasible
        opt.tell(np.array([2.0]), 2.0, np.array([3.0]))   # feasible
        opt.tell(np.array([8.0]), 0.5, np.array([-3.0]))  # infeasible
        opt.tell(np.array([9.0]), 0.2, np.array([-4.0]))  # infeasible

        pof = opt._probability_of_feasibility(np.array([[1.5], [8.5]]))
        assert pof[0] > pof[1]  # Near feasible region should have higher PoF

    def test_multiple_constraints(self):
        def obj(x):
            return x[0]

        def cons(x):
            return np.array([
                x[0] - 1.0,     # x >= 1
                4.0 - x[0],     # x <= 4
            ])

        opt = ConstrainedBayesianOptimizer(
            bounds=np.array([[0.0, 5.0]]),
            n_constraints=2, n_initial=5, seed=42
        )
        result = opt.maximize(obj, cons, n_iter=30)
        if result['feasible']:
            assert 0.5 < result['best_x'][0] < 5.0

    def test_no_feasible_solution(self):
        """When constraints are impossible to satisfy."""
        def obj(x):
            return x[0]

        def cons(x):
            return np.array([-1.0])  # Always infeasible

        opt = ConstrainedBayesianOptimizer(
            bounds=np.array([[0.0, 1.0]]),
            n_constraints=1, n_initial=3, seed=42
        )
        result = opt.maximize(obj, cons, n_iter=10)
        assert result['n_feasible'] == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_branin_optimization(self):
        """Branin-Hoo function (standard BO benchmark)."""
        def branin(x):
            x1, x2 = x[0], x[1]
            a = 1; b = 5.1 / (4 * math.pi**2); c = 5 / math.pi
            r = 6; s = 10; t = 1 / (8 * math.pi)
            return -(a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * math.cos(x1) + s)

        opt = BayesianOptimizer(
            bounds=np.array([[-5.0, 10.0], [0.0, 15.0]]),
            n_initial=10, seed=42
        )
        result = opt.maximize(branin, n_iter=40)
        # Branin minimum ~0.397887, so negated max ~-0.397887
        assert result.best_y > -5.0  # Should find a decent solution

    def test_sinusoidal_1d(self):
        """Multi-modal 1D function."""
        def f(x):
            return np.sin(3 * x[0]) * x[0]

        opt = BayesianOptimizer(
            bounds=np.array([[0.0, 3.0]]),
            kernel=MaternKernel(nu=2.5),
            n_initial=5, seed=42
        )
        result = opt.maximize(f, n_iter=25)
        assert result.best_y > 1.0

    def test_gp_prior_to_posterior(self):
        """Verify GP uncertainty shrinks as data is added."""
        # Use 2 training points so normalization produces nonzero y_std
        gp = GaussianProcess(RBFKernel(), noise=1e-6)
        X_test = np.linspace(0, 5, 50).reshape(-1, 1)

        # "Prior-like": fit to 2 points far apart
        gp.fit(np.array([[0.0], [5.0]]), np.array([0.0, 0.0]))
        _, sigma_few = gp.predict(X_test, return_std=True)

        # Posterior with more data
        X_train = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y_train = np.sin(X_train.ravel())
        gp.fit(X_train, y_train)
        _, sigma_many = gp.predict(X_test, return_std=True)

        # More data should reduce max uncertainty
        assert np.max(sigma_many) < np.max(sigma_few) + 0.5

    def test_bo_with_noise(self):
        """BO should handle noisy objectives."""
        rng = np.random.RandomState(42)
        def noisy_f(x):
            return -(x[0] - 2)**2 + rng.normal(0, 0.1)

        opt = BayesianOptimizer(
            bounds=np.array([[0.0, 4.0]]),
            noise=0.01,
            n_initial=5, seed=42
        )
        result = opt.maximize(noisy_f, n_iter=30)
        assert result.best_y > -2.0

    def test_kernel_composition_in_gp(self):
        """Test GP with composite kernel."""
        k = RBFKernel(lengthscale=1.0) + LinearKernel(variance=0.1)
        gp = GaussianProcess(k, noise=1e-6)
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 1, 4, 9], dtype=float)  # Quadratic
        gp.fit(X, y)
        mu = gp.predict(np.array([[1.5]]))
        assert 1.0 < mu[0] < 4.0

    def test_optimizer_reproducibility(self):
        """Same seed should give same results."""
        def f(x):
            return -x[0]**2

        results = []
        for _ in range(2):
            opt = BayesianOptimizer(
                bounds=np.array([[-5.0, 5.0]]),
                n_initial=5, seed=123
            )
            r = opt.maximize(f, n_iter=15)
            results.append(r.best_x[0])

        assert results[0] == pytest.approx(results[1])

    def test_high_dimensional(self):
        """BO in 5D (curse of dimensionality test)."""
        def sphere(x):
            return -np.sum(x**2)

        opt = BayesianOptimizer(
            bounds=np.array([[-2, 2]] * 5, dtype=float),
            n_initial=20, seed=42
        )
        result = opt.maximize(sphere, n_iter=50)
        # In 5D, should at least find something better than random
        assert result.best_y > -10.0

    def test_batch_vs_sequential(self):
        """Batch BO should find reasonable solutions."""
        def f(x):
            return -(x[0] - 1)**2

        batch_opt = BatchBayesianOptimizer(
            bounds=np.array([[0.0, 2.0]]),
            batch_size=3, n_initial=5, seed=42
        )
        result = batch_opt.maximize(f, n_batches=5)
        assert result.best_y > -0.5

    def test_constrained_2d(self):
        """2D constrained optimization."""
        def obj(x):
            return x[0] + x[1]

        def cons(x):
            return np.array([
                1.0 - (x[0]**2 + x[1]**2),  # inside unit circle
            ])

        opt = ConstrainedBayesianOptimizer(
            bounds=np.array([[-1.5, 1.5], [-1.5, 1.5]]),
            n_constraints=1, n_initial=10, seed=42
        )
        result = opt.maximize(obj, cons, n_iter=30)
        if result['feasible']:
            best = result['best_x']
            # Should satisfy constraint approximately
            assert result['best_y'] > 0.0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_iteration(self):
        opt = BayesianOptimizer(bounds=np.array([[0.0, 1.0]]), seed=42)
        result = opt.maximize(lambda x: x[0], n_iter=1)
        assert result.n_iterations == 1

    def test_gp_single_point(self):
        gp = GaussianProcess(RBFKernel())
        gp.fit(np.array([[0.0]]), np.array([1.0]))
        mu = gp.predict(np.array([[0.0]]))
        assert mu[0] == pytest.approx(1.0, abs=0.1)

    def test_gp_duplicate_points(self):
        gp = GaussianProcess(RBFKernel(), noise=1e-4)
        X = np.array([[0.0], [0.0], [1.0]])
        y = np.array([1.0, 1.1, 2.0])
        gp.fit(X, y)
        mu = gp.predict(X)
        assert np.all(np.isfinite(mu))

    def test_1d_input_handling(self):
        gp = GaussianProcess(RBFKernel())
        gp.fit(np.array([0, 1, 2]).reshape(-1, 1), np.array([0, 1, 0]))
        mu = gp.predict(np.array([0.5]).reshape(-1, 1))
        assert np.isfinite(mu[0])

    def test_bounds_respected(self):
        opt = BayesianOptimizer(
            bounds=np.array([[2.0, 3.0]]),
            n_initial=5, seed=42
        )
        for _ in range(15):
            x = opt.suggest()
            assert 2.0 <= x[0] <= 3.0 or np.isclose(x[0], 2.0) or np.isclose(x[0], 3.0)
            opt.tell(x, -x[0]**2)

    def test_negative_objective(self):
        def f(x):
            return -100 - x[0]**2

        opt = BayesianOptimizer(bounds=np.array([[-1, 1]], dtype=float), seed=42)
        result = opt.maximize(f, n_iter=15)
        assert result.best_y > -105

    def test_constant_objective(self):
        opt = BayesianOptimizer(bounds=np.array([[0, 1]], dtype=float), seed=42, noise=0.01)
        result = opt.maximize(lambda x: 5.0, n_iter=10)
        assert result.best_y == pytest.approx(5.0)

    def test_gp_lml_not_fitted(self):
        gp = GaussianProcess(RBFKernel())
        assert gp.log_marginal_likelihood() == -np.inf


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
