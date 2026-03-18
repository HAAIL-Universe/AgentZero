"""
Tests for V223: Bayesian Optimization

Composes V222 (Gaussian Process) with acquisition functions for
black-box optimization.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from bayesian_optimization import (
    # Core
    bayesian_optimize, batch_bayesian_optimize,
    multi_objective_optimize, constrained_optimize,
    input_warped_optimize,
    # Acquisition functions
    expected_improvement, probability_of_improvement,
    upper_confidence_bound, thompson_sampling, knowledge_gradient,
    AcquisitionType,
    # Bounds
    Bounds,
    # Results
    BOResult, BatchBOResult, MOBOResult, ConstrainedBOResult,
    # Diagnostics
    convergence_diagnostics, ConvergenceDiagnostics,
    # Comparison
    compare_acquisitions,
    # Summary
    optimization_summary,
    # Test functions
    branin, sphere, rosenbrock, ackley, six_hump_camel,
    # Pareto
    _is_pareto_optimal, _hypervolume_2d,
    # Normal CDF/PDF
    _norm_cdf, _norm_pdf, _erf,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
from gaussian_process import (
    GaussianProcess, RBFKernel, Matern52Kernel, ScaleKernel, ARDKernel
)


# ===========================================================================
# Normal distribution utilities
# ===========================================================================

class TestNormalDistribution:
    def test_norm_pdf_peak(self):
        """PDF peak at 0."""
        assert abs(_norm_pdf(0.0) - 1.0 / np.sqrt(2 * np.pi)) < 1e-10

    def test_norm_pdf_symmetry(self):
        """PDF is symmetric."""
        assert abs(_norm_pdf(1.5) - _norm_pdf(-1.5)) < 1e-10

    def test_norm_pdf_tails(self):
        """PDF decays at tails."""
        assert _norm_pdf(5.0) < 1e-5

    def test_norm_cdf_center(self):
        """CDF(0) = 0.5."""
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_norm_cdf_limits(self):
        """CDF approaches 0 and 1."""
        assert _norm_cdf(-10.0) < 1e-6
        assert _norm_cdf(10.0) > 1 - 1e-6

    def test_norm_cdf_known_values(self):
        """Check known CDF values."""
        assert abs(_norm_cdf(1.0) - 0.8413) < 0.001
        assert abs(_norm_cdf(-1.0) - 0.1587) < 0.001
        assert abs(_norm_cdf(2.0) - 0.9772) < 0.001

    def test_erf_zero(self):
        assert abs(_erf(0.0)) < 1e-10

    def test_erf_limits(self):
        assert abs(_erf(10.0) - 1.0) < 1e-6
        assert abs(_erf(-10.0) + 1.0) < 1e-6

    def test_norm_cdf_array(self):
        """CDF works on arrays."""
        x = np.array([-2, -1, 0, 1, 2])
        result = _norm_cdf(x)
        assert result.shape == (5,)
        assert np.all(np.diff(result) > 0)  # monotonically increasing


# ===========================================================================
# Bounds
# ===========================================================================

class TestBounds:
    def test_creation(self):
        b = Bounds(lower=[-5, 0], upper=[10, 15])
        assert b.dim == 2
        np.testing.assert_array_equal(b.lower, [-5, 0])
        np.testing.assert_array_equal(b.upper, [10, 15])

    def test_sample_uniform(self):
        b = Bounds(lower=[0, 0], upper=[1, 1])
        rng = np.random.default_rng(42)
        X = b.sample_uniform(100, rng=rng)
        assert X.shape == (100, 2)
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_clip(self):
        b = Bounds(lower=[0, 0], upper=[1, 1])
        X = np.array([[-1, 2], [0.5, 0.5]])
        clipped = b.clip(X)
        np.testing.assert_array_equal(clipped[0], [0, 1])
        np.testing.assert_array_equal(clipped[1], [0.5, 0.5])

    def test_normalize_unnormalize(self):
        b = Bounds(lower=[-5, 0], upper=[10, 15])
        X = np.array([[2.5, 7.5]])
        X_norm = b.normalize(X)
        np.testing.assert_allclose(X_norm, [[0.5, 0.5]])
        X_back = b.unnormalize(X_norm)
        np.testing.assert_allclose(X_back, X)

    def test_invalid_bounds(self):
        with pytest.raises(AssertionError):
            Bounds(lower=[5], upper=[3])  # lower > upper


# ===========================================================================
# Acquisition Functions
# ===========================================================================

class TestExpectedImprovement:
    def test_positive_improvement(self):
        """EI > 0 when there's potential improvement."""
        mu = np.array([0.5])
        std = np.array([1.0])
        f_best = 0.0
        ei = expected_improvement(mu, std, f_best)
        assert ei[0] > 0

    def test_zero_std(self):
        """EI = 0 when std = 0."""
        mu = np.array([0.5])
        std = np.array([0.0])
        ei = expected_improvement(mu, std, f_best=0.0)
        assert ei[0] == 0.0

    def test_monotonic_in_std(self):
        """More uncertainty -> more EI (all else equal)."""
        mu = np.array([0.5, 0.5, 0.5])
        std = np.array([0.1, 0.5, 1.0])
        ei = expected_improvement(mu, std, f_best=0.0)
        assert np.all(np.diff(ei) > 0)

    def test_better_mean_higher_ei(self):
        """Lower mean -> higher EI (minimization)."""
        mu = np.array([2.0, 1.0, 0.1])
        std = np.array([0.5, 0.5, 0.5])
        ei = expected_improvement(mu, std, f_best=0.0)
        assert np.all(np.diff(ei) > 0)

    def test_xi_exploration(self):
        """Higher xi encourages more exploration."""
        mu = np.array([0.0])
        std = np.array([0.5])
        ei_low = expected_improvement(mu, std, f_best=0.0, xi=0.0)
        ei_high = expected_improvement(mu, std, f_best=0.0, xi=0.5)
        # Higher xi penalizes more, so EI should be lower
        assert ei_low[0] >= ei_high[0]

    def test_array_input(self):
        """Works with array inputs."""
        mu = np.array([0.5, 1.0, 1.5, 2.0])
        std = np.array([0.2, 0.3, 0.4, 0.5])
        ei = expected_improvement(mu, std, f_best=0.0)
        assert ei.shape == (4,)
        assert np.all(ei >= 0)


class TestProbabilityOfImprovement:
    def test_high_pi_below_best(self):
        """PI ~ 1 when mean << f_best."""
        mu = np.array([-5.0])
        std = np.array([0.1])
        pi = probability_of_improvement(mu, std, f_best=0.0)
        assert pi[0] > 0.99

    def test_low_pi_above_best(self):
        """PI ~ 0 when mean >> f_best."""
        mu = np.array([10.0])
        std = np.array([0.1])
        pi = probability_of_improvement(mu, std, f_best=0.0)
        assert pi[0] < 0.01

    def test_pi_range(self):
        """PI always in [0, 1]."""
        mu = np.array([-2, -1, 0, 1, 2])
        std = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        pi = probability_of_improvement(mu, std, f_best=0.0)
        assert np.all(pi >= 0) and np.all(pi <= 1)


class TestUpperConfidenceBound:
    def test_ucb_prefers_low_mean(self):
        """UCB (for min) prefers lower mean."""
        mu = np.array([0.0, 1.0])
        std = np.array([0.5, 0.5])
        ucb = upper_confidence_bound(mu, std, beta=2.0)
        assert ucb[0] > ucb[1]

    def test_ucb_prefers_high_std(self):
        """UCB (for min) prefers higher uncertainty."""
        mu = np.array([0.5, 0.5])
        std = np.array([0.1, 1.0])
        ucb = upper_confidence_bound(mu, std, beta=2.0)
        assert ucb[1] > ucb[0]

    def test_beta_controls_exploration(self):
        """Higher beta = more exploration (more weight on std)."""
        mu = np.array([0.0])
        std = np.array([1.0])
        ucb_low = upper_confidence_bound(mu, std, beta=0.1)
        ucb_high = upper_confidence_bound(mu, std, beta=5.0)
        assert ucb_high[0] > ucb_low[0]


class TestThompsonSampling:
    def test_returns_values(self):
        """Thompson sampling returns values for all candidates."""
        gp = GaussianProcess(kernel=RBFKernel(length_scale=1.0), noise_variance=1e-4)
        X_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([1.0, 0.0, 1.0])
        gp.fit(X_train, y_train)

        X_cand = np.linspace(-1, 3, 20).reshape(-1, 1)
        rng = np.random.default_rng(42)
        vals = thompson_sampling(gp, X_cand, rng=rng)
        assert vals.shape == (20,)

    def test_different_seeds_different_results(self):
        """Different seeds give different samples."""
        gp = GaussianProcess(kernel=RBFKernel(length_scale=1.0), noise_variance=1e-4)
        gp.fit(np.array([[0.0], [1.0]]), np.array([1.0, 0.0]))

        X_cand = np.linspace(0, 1, 10).reshape(-1, 1)
        vals1 = thompson_sampling(gp, X_cand, rng=np.random.default_rng(1))
        vals2 = thompson_sampling(gp, X_cand, rng=np.random.default_rng(2))
        assert not np.allclose(vals1, vals2)


class TestKnowledgeGradient:
    def test_kg_positive(self):
        """KG is non-negative."""
        mu = np.array([0.5, 1.0, 1.5])
        std = np.array([0.3, 0.5, 0.7])
        kg = knowledge_gradient(mu, std, f_best=0.0)
        assert np.all(kg >= 0)

    def test_kg_zero_std(self):
        """KG handles zero std."""
        mu = np.array([0.5])
        std = np.array([0.0])
        kg = knowledge_gradient(mu, std, f_best=0.0)
        assert kg[0] == 0.0


# ===========================================================================
# Test Functions (Benchmark)
# ===========================================================================

class TestBenchmarkFunctions:
    def test_sphere_minimum(self):
        assert sphere(np.zeros(3)) == 0.0
        assert sphere(np.ones(3)) == 3.0

    def test_branin_shape(self):
        """Branin returns scalar."""
        val = branin(np.array([0.0, 0.0]))
        assert isinstance(val, float)

    def test_branin_known_minimum(self):
        """Branin global min ~ 0.397887."""
        # Known minima: (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
        val = branin(np.array([np.pi, 2.275]))
        assert val < 0.5  # close to global min

    def test_rosenbrock_minimum(self):
        assert rosenbrock(np.array([1.0, 1.0])) == 0.0

    def test_ackley_minimum(self):
        assert abs(ackley(np.zeros(2))) < 1e-10

    def test_six_hump_camel_shape(self):
        val = six_hump_camel(np.array([0.0, 0.0]))
        assert isinstance(val, float)

    def test_six_hump_camel_near_minimum(self):
        """Known global min ~ -1.0316."""
        val = six_hump_camel(np.array([0.0898, -0.7126]))
        assert val < -1.0


# ===========================================================================
# Pareto / Hypervolume
# ===========================================================================

class TestPareto:
    def test_pareto_simple(self):
        """Single dominated point."""
        Y = np.array([[1, 3], [2, 1], [3, 2]])
        mask = _is_pareto_optimal(Y)
        # (1,3) and (2,1) are Pareto optimal; (3,2) is dominated by (2,1)
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False

    def test_pareto_all_optimal(self):
        """All points on front."""
        Y = np.array([[1, 3], [2, 2], [3, 1]])
        mask = _is_pareto_optimal(Y)
        assert np.all(mask)

    def test_pareto_one_dominates_all(self):
        """One point dominates everything."""
        Y = np.array([[0, 0], [1, 1], [2, 2]])
        mask = _is_pareto_optimal(Y)
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == False

    def test_hypervolume_2d_simple(self):
        """Simple 2D hypervolume."""
        pareto_y = np.array([[1, 2]])
        ref = np.array([3, 4])
        hv = _hypervolume_2d(pareto_y, ref)
        # Single point: (3-1) * (4-2) = 4
        assert abs(hv - 4.0) < 1e-10

    def test_hypervolume_2d_two_points(self):
        """Two points on Pareto front."""
        pareto_y = np.array([[1, 3], [2, 1]])
        ref = np.array([4, 4])
        hv = _hypervolume_2d(pareto_y, ref)
        # (2-1)*(4-3) + (4-2)*(4-1) = 1 + 6 = 7
        assert abs(hv - 7.0) < 1e-10

    def test_hypervolume_empty(self):
        """No points dominated by ref."""
        pareto_y = np.array([[5, 5]])
        ref = np.array([3, 3])
        hv = _hypervolume_2d(pareto_y, ref)
        assert hv == 0.0


# ===========================================================================
# Core Bayesian Optimization
# ===========================================================================

class TestBayesianOptimize:
    def test_sphere_1d(self):
        """Optimize 1D sphere (quadratic)."""
        bounds = Bounds(lower=[-5], upper=[5])
        result = bayesian_optimize(sphere, bounds, n_iterations=20,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert isinstance(result, BOResult)
        assert result.f_best < 0.5  # Should get close to 0
        assert abs(result.x_best[0]) < 1.5

    def test_sphere_2d(self):
        """Optimize 2D sphere."""
        bounds = Bounds(lower=[-3, -3], upper=[3, 3])
        result = bayesian_optimize(sphere, bounds, n_iterations=25,
                                    n_initial=5, rng=np.random.default_rng(42))
        assert result.f_best < 2.0

    def test_convergence_tracked(self):
        """Convergence list is monotonically non-increasing."""
        bounds = Bounds(lower=[-5], upper=[5])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=3, rng=np.random.default_rng(42))
        for i in range(1, len(result.convergence)):
            assert result.convergence[i] <= result.convergence[i-1] + 1e-10

    def test_history_length(self):
        """History has correct length."""
        bounds = Bounds(lower=[-5], upper=[5])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=5, rng=np.random.default_rng(42))
        assert len(result.y_history) == 15  # 5 initial + 10 iterations
        assert result.X_history.shape == (15, 1)

    def test_model_returned(self):
        """Final GP model is returned."""
        bounds = Bounds(lower=[-5], upper=[5])
        result = bayesian_optimize(sphere, bounds, n_iterations=5,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert result.model is not None
        pred = result.model.predict(np.array([[0.0]]))
        assert pred.mean is not None

    def test_ei_acquisition(self):
        """EI acquisition works."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    acquisition=AcquisitionType.EI,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert result.f_best < 1.0

    def test_pi_acquisition(self):
        """PI acquisition works."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    acquisition=AcquisitionType.PI,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert result.f_best < 2.0

    def test_ucb_acquisition(self):
        """UCB acquisition works."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    acquisition=AcquisitionType.UCB,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert result.f_best < 2.0

    def test_thompson_acquisition(self):
        """Thompson sampling works."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    acquisition=AcquisitionType.THOMPSON,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert result.f_best < 3.0

    def test_kg_acquisition(self):
        """Knowledge gradient works."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    acquisition=AcquisitionType.KG,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert result.f_best < 2.0

    def test_custom_kernel(self):
        """Custom kernel works."""
        kernel = ScaleKernel(RBFKernel(length_scale=1.0), scale=2.0)
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=3, kernel=kernel,
                                    rng=np.random.default_rng(42))
        assert isinstance(result, BOResult)

    def test_acq_params(self):
        """Custom acquisition parameters work."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=5,
                                    acquisition=AcquisitionType.UCB,
                                    acq_params={'beta': 5.0},
                                    n_initial=3, rng=np.random.default_rng(42))
        assert isinstance(result, BOResult)


# ===========================================================================
# Batch Bayesian Optimization
# ===========================================================================

class TestBatchBO:
    def test_batch_sphere(self):
        """Batch BO finds good solution for sphere."""
        bounds = Bounds(lower=[-3, -3], upper=[3, 3])
        result = batch_bayesian_optimize(sphere, bounds, n_batches=5,
                                          batch_size=3, n_initial=5,
                                          rng=np.random.default_rng(42))
        assert isinstance(result, BatchBOResult)
        assert result.f_best < 3.0

    def test_batch_sizes(self):
        """Batch sizes are correct."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = batch_bayesian_optimize(sphere, bounds, n_batches=3,
                                          batch_size=4, n_initial=5,
                                          rng=np.random.default_rng(42))
        assert len(result.batch_sizes) == 3
        assert all(s == 4 for s in result.batch_sizes)
        assert len(result.y_history) == 5 + 3 * 4  # initial + batches

    def test_batch_convergence(self):
        """Batch convergence is tracked."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = batch_bayesian_optimize(sphere, bounds, n_batches=3,
                                          batch_size=2, n_initial=3,
                                          rng=np.random.default_rng(42))
        assert len(result.convergence) == 4  # 1 initial + 3 batches


# ===========================================================================
# Multi-Objective Optimization
# ===========================================================================

class TestMultiObjective:
    def test_mobo_two_objectives(self):
        """Multi-objective optimization on conflicting objectives."""
        def obj1(x):
            return sphere(x)

        def obj2(x):
            return sphere(x - 2)

        bounds = Bounds(lower=[-3, -3], upper=[5, 5])
        result = multi_objective_optimize(
            [obj1, obj2], bounds, n_iterations=10, n_initial=5,
            ref_point=np.array([30.0, 30.0]),
            rng=np.random.default_rng(42)
        )
        assert isinstance(result, MOBOResult)
        assert len(result.pareto_X) > 0
        assert len(result.pareto_y) > 0
        assert result.pareto_y.shape[1] == 2

    def test_mobo_hypervolume_increases(self):
        """Hypervolume should generally increase."""
        def obj1(x):
            return x[0]**2

        def obj2(x):
            return (x[0] - 2)**2

        bounds = Bounds(lower=[-3], upper=[5])
        result = multi_objective_optimize(
            [obj1, obj2], bounds, n_iterations=10, n_initial=5,
            ref_point=np.array([30.0, 30.0]),
            rng=np.random.default_rng(42)
        )
        # Hypervolume should be non-negative and generally growing
        assert all(h >= 0 for h in result.hypervolume_history)

    def test_mobo_pareto_front_valid(self):
        """All returned Pareto points are actually Pareto optimal."""
        def obj1(x):
            return sphere(x)

        def obj2(x):
            return sphere(x - 1)

        bounds = Bounds(lower=[-2, -2], upper=[3, 3])
        result = multi_objective_optimize(
            [obj1, obj2], bounds, n_iterations=8, n_initial=5,
            ref_point=np.array([20.0, 20.0]),
            rng=np.random.default_rng(42)
        )
        # Verify Pareto optimality
        mask = _is_pareto_optimal(result.pareto_y)
        assert np.all(mask)


# ===========================================================================
# Constrained Optimization
# ===========================================================================

class TestConstrainedBO:
    def test_constrained_sphere(self):
        """Constrained sphere: min x^2 s.t. x >= 1."""
        def constraint(x):
            return 1.0 - x[0]  # x[0] >= 1 means 1 - x[0] <= 0

        bounds = Bounds(lower=[-5], upper=[5])
        result = constrained_optimize(
            sphere, [constraint], bounds,
            n_iterations=15, n_initial=5,
            rng=np.random.default_rng(42)
        )
        assert isinstance(result, ConstrainedBOResult)
        # Best feasible should be near x=1
        if np.any(result.feasible_mask):
            assert result.x_best[0] >= 0.5  # near constraint boundary

    def test_constrained_feasibility(self):
        """Feasibility mask is correct."""
        def constraint(x):
            return x[0] - 2.0  # x[0] <= 2

        bounds = Bounds(lower=[0], upper=[5])
        result = constrained_optimize(
            sphere, [constraint], bounds,
            n_iterations=10, n_initial=5,
            rng=np.random.default_rng(42)
        )
        # Check feasibility
        for i in range(len(result.y_history)):
            expected = result.constraint_values[i, 0] <= 0
            assert result.feasible_mask[i] == expected

    def test_multiple_constraints(self):
        """Multiple constraints."""
        def c1(x):
            return x[0] - 3.0  # x <= 3

        def c2(x):
            return -x[0] - 1.0  # x >= -1

        bounds = Bounds(lower=[-5], upper=[5])
        result = constrained_optimize(
            sphere, [c1, c2], bounds,
            n_iterations=10, n_initial=5,
            rng=np.random.default_rng(42)
        )
        assert result.constraint_values.shape[1] == 2


# ===========================================================================
# Input Warping
# ===========================================================================

class TestInputWarping:
    def test_warped_sphere(self):
        """Input warping doesn't break optimization."""
        bounds = Bounds(lower=[-3, -3], upper=[3, 3])
        result = input_warped_optimize(sphere, bounds, n_iterations=15,
                                        n_initial=5,
                                        rng=np.random.default_rng(42))
        assert isinstance(result, BOResult)
        assert result.f_best < 5.0

    def test_warped_preserves_bounds(self):
        """Warped results stay within original bounds."""
        bounds = Bounds(lower=[-3, -3], upper=[3, 3])
        result = input_warped_optimize(sphere, bounds, n_iterations=5,
                                        n_initial=3,
                                        rng=np.random.default_rng(42))
        for x in result.X_history:
            assert np.all(x >= bounds.lower - 1e-6)
            assert np.all(x <= bounds.upper + 1e-6)


# ===========================================================================
# Convergence Diagnostics
# ===========================================================================

class TestConvergenceDiagnostics:
    def test_diagnostics_basic(self):
        """Basic diagnostics computation."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=3, rng=np.random.default_rng(42))
        diag = convergence_diagnostics(result, f_opt=0.0)
        assert isinstance(diag, ConvergenceDiagnostics)
        assert len(diag.regret) == len(result.y_history)
        assert all(r >= -1e-6 for r in diag.regret)

    def test_regret_non_increasing(self):
        """Simple regret should be non-increasing."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=3, rng=np.random.default_rng(42))
        diag = convergence_diagnostics(result, f_opt=0.0)
        for i in range(1, len(diag.regret)):
            assert diag.regret[i] <= diag.regret[i-1] + 1e-10

    def test_cumulative_regret(self):
        """Cumulative regret is monotonically increasing."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=3, rng=np.random.default_rng(42))
        diag = convergence_diagnostics(result, f_opt=0.0)
        for i in range(1, len(diag.cumulative_regret)):
            assert diag.cumulative_regret[i] >= diag.cumulative_regret[i-1] - 1e-10

    def test_stagnation_length(self):
        """Stagnation length is non-negative."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=3, rng=np.random.default_rng(42))
        diag = convergence_diagnostics(result)
        assert diag.stagnation_length >= 0

    def test_exploration_ratio(self):
        """Exploration ratio in [0, 1]."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=10,
                                    n_initial=3, rng=np.random.default_rng(42))
        diag = convergence_diagnostics(result)
        assert 0 <= diag.exploration_ratio <= 1


# ===========================================================================
# Comparison Utility
# ===========================================================================

class TestComparison:
    def test_compare_acquisitions(self):
        """Compare multiple acquisition functions."""
        bounds = Bounds(lower=[-3], upper=[3])
        results = compare_acquisitions(
            sphere, bounds,
            acquisitions=[AcquisitionType.EI, AcquisitionType.UCB],
            n_iterations=5, n_initial=3
        )
        assert 'expected_improvement' in results
        assert 'upper_confidence_bound' in results
        assert isinstance(results['expected_improvement'], BOResult)

    def test_compare_same_initial_points(self):
        """All strategies start from same initial points."""
        bounds = Bounds(lower=[-3], upper=[3])
        results = compare_acquisitions(
            sphere, bounds,
            acquisitions=[AcquisitionType.EI, AcquisitionType.PI],
            n_iterations=5, n_initial=3
        )
        # Same seed -> same initial points
        np.testing.assert_array_equal(
            results['expected_improvement'].X_history[:3],
            results['probability_of_improvement'].X_history[:3]
        )


# ===========================================================================
# Summary
# ===========================================================================

class TestSummary:
    def test_summary_output(self):
        """Summary returns readable string."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=5,
                                    n_initial=3, rng=np.random.default_rng(42))
        summary = optimization_summary(result, name="Sphere Test")
        assert "Sphere Test" in summary
        assert "Best value" in summary
        assert "Total evaluations" in summary


# ===========================================================================
# Integration: Branin optimization
# ===========================================================================

class TestBraninIntegration:
    def test_branin_bo(self):
        """BO on Branin should find near-global minimum."""
        bounds = Bounds(lower=[-5, 0], upper=[10, 15])
        result = bayesian_optimize(branin, bounds, n_iterations=30,
                                    n_initial=5, rng=np.random.default_rng(42))
        # Global min ~ 0.397887
        assert result.f_best < 2.0  # Should get reasonably close

    def test_branin_convergence(self):
        """Branin convergence is monotonically decreasing."""
        bounds = Bounds(lower=[-5, 0], upper=[10, 15])
        result = bayesian_optimize(branin, bounds, n_iterations=15,
                                    n_initial=5, rng=np.random.default_rng(42))
        for i in range(1, len(result.convergence)):
            assert result.convergence[i] <= result.convergence[i-1] + 1e-10


# ===========================================================================
# Integration: Six-Hump Camel
# ===========================================================================

class TestSixHumpCamelIntegration:
    def test_six_hump_camel_bo(self):
        """BO on Six-Hump Camel."""
        bounds = Bounds(lower=[-3, -2], upper=[3, 2])
        result = bayesian_optimize(six_hump_camel, bounds, n_iterations=25,
                                    n_initial=5, rng=np.random.default_rng(42))
        # Global min ~ -1.0316
        assert result.f_best < 0.0


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_single_iteration(self):
        """Single BO iteration works."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(sphere, bounds, n_iterations=1,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert len(result.y_history) == 4

    def test_high_dimensional(self):
        """BO works in higher dimensions (slower but functional)."""
        bounds = Bounds(lower=-3 * np.ones(5), upper=3 * np.ones(5))
        result = bayesian_optimize(sphere, bounds, n_iterations=5,
                                    n_initial=10, rng=np.random.default_rng(42))
        assert isinstance(result, BOResult)

    def test_noisy_objective(self):
        """BO handles noisy objectives."""
        rng = np.random.default_rng(42)
        def noisy_sphere(x):
            return sphere(x) + rng.normal() * 0.1

        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(noisy_sphere, bounds, n_iterations=10,
                                    n_initial=3, noise_variance=0.01,
                                    rng=np.random.default_rng(42))
        assert isinstance(result, BOResult)

    def test_flat_objective(self):
        """BO handles flat objective (all values equal)."""
        bounds = Bounds(lower=[-3], upper=[3])
        result = bayesian_optimize(lambda x: 1.0, bounds, n_iterations=5,
                                    n_initial=3, rng=np.random.default_rng(42))
        assert result.f_best == 1.0

    def test_1d_bounds(self):
        """1D optimization works."""
        bounds = Bounds(lower=[0], upper=[10])
        result = bayesian_optimize(lambda x: (x[0] - 5)**2, bounds,
                                    n_iterations=15, n_initial=3,
                                    rng=np.random.default_rng(42))
        assert result.f_best < 2.0


# ===========================================================================
# Rosenbrock (harder test)
# ===========================================================================

class TestRosenbrock:
    def test_rosenbrock_bo(self):
        """BO on 2D Rosenbrock -- harder landscape."""
        bounds = Bounds(lower=[-2, -2], upper=[2, 2])
        result = bayesian_optimize(rosenbrock, bounds, n_iterations=30,
                                    n_initial=5, rng=np.random.default_rng(42))
        # Global min = 0 at (1,1), BO should get reasonably close
        assert result.f_best < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
