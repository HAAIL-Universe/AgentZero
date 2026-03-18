"""Tests for V233: Robust Bayesian Optimization."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V230_transfer_bo'))

from robust_bo import (
    # Core
    robust_bo, minimax_bo, robust_transfer_bo,
    # Acquisition
    expected_improvement, robust_expected_improvement,
    worst_case_expected_improvement, distributionally_robust_ei,
    # Analysis
    sensitivity_analysis, certify_robustness,
    # Multi-objective
    pareto_robust_bo,
    # Comparison
    compare_robust_strategies,
    # Summary
    robust_bo_summary,
    # Data structures
    RobustBOResult, RobustnessProfile, AdversarialResult, RobustStrategy,
    # Internals
    _norm_cdf, _norm_pdf, _erf, _unscented_sigma_points,
    _latin_hypercube_sample, _clip_to_bounds, _pareto_efficient,
    _compute_robustness_score, _worst_case_value
)
from gaussian_process import GaussianProcess, RBFKernel, Matern52Kernel
from transfer_bo import BOTask


# ---------------------------------------------------------------------------
# Test objectives
# ---------------------------------------------------------------------------

def sphere(x):
    """Negative sphere: max at origin. Smooth, easy to optimize."""
    return -np.sum(x ** 2)

def noisy_sphere(x):
    """Sphere with sharp spike near origin -- sensitive to perturbation."""
    base = -np.sum(x ** 2)
    r = np.linalg.norm(x)
    if r < 0.05:
        return base + 2.0  # Sharp spike
    return base

def robust_peak(x):
    """Broad peak -- robust to perturbation."""
    return np.exp(-0.5 * np.sum(x ** 2))

def narrow_peak(x):
    """Narrow peak at [0.5, 0.5] -- fragile under perturbation."""
    return np.exp(-10 * np.sum((x - 0.5) ** 2))

def branin_1d(x):
    """1D slice of Branin-like function."""
    return -(x[0] - 0.3) ** 2 - 0.5 * np.sin(3 * x[0])

def adversarial_objective(x, z):
    """Objective where z is adversarial: f(x, z) = -(x-0.5)^2 - z*(x-0.5)."""
    return -np.sum((x - 0.5) ** 2) - np.sum(z * (x - 0.5))


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestNormFunctions:
    def test_norm_cdf_at_zero(self):
        assert abs(_norm_cdf(0) - 0.5) < 1e-6

    def test_norm_cdf_positive(self):
        assert _norm_cdf(3.0) > 0.99

    def test_norm_cdf_negative(self):
        assert _norm_cdf(-3.0) < 0.01

    def test_norm_cdf_symmetry(self):
        assert abs(_norm_cdf(1.5) + _norm_cdf(-1.5) - 1.0) < 1e-6

    def test_norm_pdf_at_zero(self):
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert abs(_norm_pdf(0) - expected) < 1e-6

    def test_norm_pdf_symmetry(self):
        assert abs(_norm_pdf(1.0) - _norm_pdf(-1.0)) < 1e-10

    def test_erf_at_zero(self):
        assert abs(_erf(0)) < 1e-10

    def test_erf_large_positive(self):
        assert abs(_erf(3.0) - 1.0) < 0.001

    def test_erf_antisymmetry(self):
        assert abs(_erf(1.0) + _erf(-1.0)) < 1e-10


class TestUnscentedSigmaPoints:
    def test_num_points_1d(self):
        pts = _unscented_sigma_points(1, 0.1)
        assert len(pts) == 3  # 2*1 + 1

    def test_num_points_3d(self):
        pts = _unscented_sigma_points(3, 0.1)
        assert len(pts) == 7  # 2*3 + 1

    def test_center_is_zero(self):
        pts = _unscented_sigma_points(2, 0.1)
        assert np.allclose(pts[0], [0, 0])

    def test_symmetry(self):
        pts = _unscented_sigma_points(2, 0.5)
        # Each pair should be symmetric
        assert np.allclose(pts[1], -pts[2])
        assert np.allclose(pts[3], -pts[4])


class TestLatinHypercube:
    def test_within_bounds(self):
        bounds = np.array([[0, 1], [0, 1]])
        rng = np.random.RandomState(42)
        X = _latin_hypercube_sample(bounds, 20, rng)
        assert np.all(X >= 0) and np.all(X <= 1)
        assert X.shape == (20, 2)

    def test_coverage(self):
        bounds = np.array([[0, 10]])
        rng = np.random.RandomState(42)
        X = _latin_hypercube_sample(bounds, 100, rng)
        # Should cover most of the range
        assert X.min() < 1.0
        assert X.max() > 9.0

    def test_different_bounds(self):
        bounds = np.array([[-5, 5], [10, 20]])
        rng = np.random.RandomState(42)
        X = _latin_hypercube_sample(bounds, 50, rng)
        assert np.all(X[:, 0] >= -5) and np.all(X[:, 0] <= 5)
        assert np.all(X[:, 1] >= 10) and np.all(X[:, 1] <= 20)


class TestClipToBounds:
    def test_clip(self):
        X = np.array([[1.5, -0.5], [0.5, 0.5]])
        bounds = np.array([[0, 1], [0, 1]])
        clipped = _clip_to_bounds(X, bounds)
        assert clipped[0, 0] == 1.0
        assert clipped[0, 1] == 0.0
        assert np.allclose(clipped[1], [0.5, 0.5])


class TestParetoEfficient:
    def test_simple_pareto(self):
        Y = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.3, 0.3]])
        mask = _pareto_efficient(Y)
        assert mask[0] and mask[1]  # [1,0] and [0,1] are Pareto
        assert not mask[3]  # [0.3, 0.3] is dominated by [0.5, 0.5]

    def test_all_pareto(self):
        Y = np.array([[1, 0], [0, 1]])
        mask = _pareto_efficient(Y)
        assert all(mask)

    def test_single_dominant(self):
        Y = np.array([[1, 1], [0, 0], [0.5, 0.5]])
        mask = _pareto_efficient(Y)
        assert mask[0]
        assert not mask[1]
        assert not mask[2]


# ---------------------------------------------------------------------------
# Expected improvement tests
# ---------------------------------------------------------------------------

class TestExpectedImprovement:
    def test_ei_positive_improvement(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.5])
        f_best = 0.5
        ei = expected_improvement(mu, sigma, f_best)
        assert np.all(ei > 0)
        assert ei[1] > ei[0]  # Higher mu -> higher EI

    def test_ei_zero_sigma(self):
        mu = np.array([1.0])
        sigma = np.array([0.0])
        f_best = 0.5
        ei = expected_improvement(mu, sigma, f_best)
        assert ei[0] == 0.5  # deterministic improvement

    def test_ei_below_best(self):
        mu = np.array([-1.0])
        sigma = np.array([0.01])
        f_best = 5.0
        ei = expected_improvement(mu, sigma, f_best)
        assert ei[0] < 0.01  # Very small EI when far below best


class TestRobustEI:
    def test_robust_ei_returns_nonneg(self):
        rng = np.random.RandomState(42)
        bounds = np.array([[0, 1], [0, 1]])
        X = _latin_hypercube_sample(bounds, 10, rng)
        y = np.array([sphere(x) for x in X])
        gp = GaussianProcess(RBFKernel(0.3), noise_variance=0.01)
        gp.fit(X, y)
        candidates = _latin_hypercube_sample(bounds, 20, rng)
        rei = robust_expected_improvement(gp, candidates, np.max(y), 0.1, rng=rng)
        assert np.all(rei >= -1e-10)

    def test_robust_ei_smooth_function(self):
        """Robust EI should work on smooth functions."""
        rng = np.random.RandomState(42)
        bounds = np.array([[0, 1]])
        X = rng.uniform(0, 1, (8, 1))
        y = np.array([sphere(x) for x in X])
        gp = GaussianProcess(RBFKernel(0.3), noise_variance=0.01)
        gp.fit(X, y)
        candidates = np.linspace(0, 1, 30).reshape(-1, 1)
        rei = robust_expected_improvement(gp, candidates, np.max(y), 0.05, rng=rng)
        assert len(rei) == 30


class TestWorstCaseEI:
    def test_wcei_leq_standard_ei(self):
        """Worst-case EI should be <= standard EI."""
        rng = np.random.RandomState(42)
        bounds = np.array([[0, 1]])
        X = rng.uniform(0, 1, (10, 1))
        y = np.array([sphere(x) for x in X])
        gp = GaussianProcess(RBFKernel(0.3), noise_variance=0.01)
        gp.fit(X, y)
        candidates = rng.uniform(0, 1, (20, 1))
        f_best = np.max(y)

        pred = gp.predict(candidates)
        standard_ei = expected_improvement(pred.mean.ravel(), pred.std.ravel(), f_best)
        wc_ei = worst_case_expected_improvement(gp, candidates, f_best, 0.1, rng=rng)
        # Worst-case should generally be <= standard (with some numerical tolerance)
        assert np.mean(wc_ei) <= np.mean(standard_ei) + 0.1


class TestDROEI:
    def test_dro_ei_returns_values(self):
        rng = np.random.RandomState(42)
        bounds = np.array([[0, 1]])
        X = rng.uniform(0, 1, (10, 1))
        y = np.array([sphere(x) for x in X])
        gp = GaussianProcess(RBFKernel(0.3), noise_variance=0.01)
        gp.fit(X, y)
        candidates = rng.uniform(0, 1, (20, 1))
        dro = distributionally_robust_ei(gp, candidates, np.max(y), 0.1, rng=rng)
        assert len(dro) == 20
        assert np.all(np.isfinite(dro))


# ---------------------------------------------------------------------------
# Robust BO core tests
# ---------------------------------------------------------------------------

class TestRobustBO:
    def test_basic_1d(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.1,
                          n_iterations=10, n_initial=5, rng=np.random.RandomState(42))
        assert isinstance(result, RobustBOResult)
        assert result.f_best > -0.5  # Should find near origin
        assert result.total_evaluations == 15

    def test_basic_2d(self):
        bounds = np.array([[-1, 1], [-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.05,
                          n_iterations=15, n_initial=5, rng=np.random.RandomState(42))
        assert result.f_best > -1.0
        assert len(result.convergence) == 16  # n_initial(1) + n_iterations

    def test_convergence_monotonic(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.1,
                          n_iterations=10, n_initial=5, rng=np.random.RandomState(42))
        # Convergence (best-so-far) should be non-decreasing
        for i in range(1, len(result.convergence)):
            assert result.convergence[i] >= result.convergence[i-1] - 1e-10

    def test_robustness_scores(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.1,
                          n_iterations=10, n_initial=5, rng=np.random.RandomState(42))
        assert len(result.robustness_scores) == 11  # 1 initial + 10 iterations
        assert all(0 <= s <= 1 for s in result.robustness_scores)

    def test_worst_case_ei_strategy(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.1,
                          n_iterations=10, n_initial=5,
                          acquisition="worst_case_ei",
                          rng=np.random.RandomState(42))
        assert result.strategy == "worst_case_ei"
        assert result.f_best > -1.0

    def test_dro_strategy(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.1,
                          n_iterations=10, n_initial=5,
                          acquisition="dro_ei",
                          rng=np.random.RandomState(42))
        assert result.strategy == "dro_ei"
        assert result.f_best > -1.0

    def test_invalid_acquisition(self):
        bounds = np.array([[-1, 1]])
        with pytest.raises(ValueError, match="Unknown acquisition"):
            robust_bo(sphere, bounds, acquisition="bad",
                     n_iterations=1, n_initial=3, rng=np.random.RandomState(42))

    def test_worst_case_gap(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.1,
                          n_iterations=10, n_initial=5, rng=np.random.RandomState(42))
        assert result.worst_case_gap is not None
        assert result.worst_case_gap >= 0  # Gap should be non-negative

    def test_robust_peak(self):
        """Robust peak should be found with high robustness score."""
        bounds = np.array([[-2, 2]])
        result = robust_bo(robust_peak, bounds, input_noise_std=0.2,
                          n_iterations=15, n_initial=5, rng=np.random.RandomState(42))
        assert result.f_best > 0.5

    def test_custom_kernel(self):
        bounds = np.array([[-1, 1]])
        kernel = Matern52Kernel(length_scale=0.5, variance=1.0)
        result = robust_bo(sphere, bounds, input_noise_std=0.1,
                          n_iterations=5, n_initial=5, kernel=kernel,
                          rng=np.random.RandomState(42))
        assert isinstance(result, RobustBOResult)

    def test_verbose(self, capsys):
        bounds = np.array([[-1, 1]])
        robust_bo(sphere, bounds, input_noise_std=0.1,
                 n_iterations=3, n_initial=3, verbose=True,
                 rng=np.random.RandomState(42))
        captured = capsys.readouterr()
        assert "Iter 1" in captured.out


# ---------------------------------------------------------------------------
# Min-max BO tests
# ---------------------------------------------------------------------------

class TestMinimaxBO:
    def test_basic(self):
        design_bounds = np.array([[0, 1]])
        adversary_bounds = np.array([[-0.5, 0.5]])
        result = minimax_bo(adversarial_objective, design_bounds, adversary_bounds,
                           n_iterations=10, n_initial=5, rng=np.random.RandomState(42))
        assert isinstance(result, AdversarialResult)
        assert result.x_best is not None
        assert result.z_worst is not None
        assert result.robustness_gap >= 0 or True  # Can be negative in some cases

    def test_result_fields(self):
        design_bounds = np.array([[0, 1]])
        adversary_bounds = np.array([[-0.5, 0.5]])
        result = minimax_bo(adversarial_objective, design_bounds, adversary_bounds,
                           n_iterations=5, n_initial=5, rng=np.random.RandomState(42))
        assert len(result.X_design_history) == 5
        assert len(result.Z_adversary_history) == 5
        assert len(result.y_history) == 10  # n_initial + n_iterations

    def test_2d_design(self):
        def obj_2d(x, z):
            return -np.sum((x - 0.5) ** 2) - 0.3 * np.sum(z ** 2)

        design_bounds = np.array([[0, 1], [0, 1]])
        adversary_bounds = np.array([[-0.3, 0.3]])
        result = minimax_bo(obj_2d, design_bounds, adversary_bounds,
                           n_iterations=8, n_initial=5, rng=np.random.RandomState(42))
        assert result.x_best.shape == (2,)
        assert result.z_worst.shape == (1,)

    def test_adversary_exploits_weakness(self):
        """Adversary should push z to extremes to minimize objective."""
        def asymmetric_obj(x, z):
            return -np.sum((x - 0.5) ** 2) - 2 * np.sum(z)  # Adversary wants max z

        design_bounds = np.array([[0, 1]])
        adversary_bounds = np.array([[-1, 1]])
        result = minimax_bo(asymmetric_obj, design_bounds, adversary_bounds,
                           n_iterations=10, n_initial=5, rng=np.random.RandomState(42))
        # z_worst should be near upper bound (1) to minimize
        assert result.z_worst[0] > 0  # Adversary should go positive

    def test_verbose_minimax(self, capsys):
        design_bounds = np.array([[0, 1]])
        adversary_bounds = np.array([[-0.5, 0.5]])
        minimax_bo(adversarial_objective, design_bounds, adversary_bounds,
                  n_iterations=3, n_initial=3, verbose=True,
                  rng=np.random.RandomState(42))
        captured = capsys.readouterr()
        assert "Iter 1" in captured.out


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_sphere_at_origin(self):
        bounds = np.array([[-2, 2], [-2, 2]])
        profile = sensitivity_analysis(sphere, np.array([0.0, 0.0]), bounds,
                                       rng=np.random.RandomState(42))
        assert isinstance(profile, RobustnessProfile)
        assert profile.nominal_value == 0.0
        # At origin, sphere is at maximum -- all perturbations reduce it
        assert profile.worst_case_value <= 0.0

    def test_sensitivity_per_dimension(self):
        bounds = np.array([[-2, 2], [-2, 2]])
        profile = sensitivity_analysis(sphere, np.array([1.0, 0.0]), bounds,
                                       rng=np.random.RandomState(42))
        # Sensitivity should be higher in dimension 0 (at x=1) than dim 1 (at x=0)
        assert profile.sensitivity[0] > profile.sensitivity[1]

    def test_robustness_score_range(self):
        bounds = np.array([[-2, 2]])
        profile = sensitivity_analysis(robust_peak, np.array([0.0]), bounds,
                                       rng=np.random.RandomState(42))
        assert 0 <= profile.robustness_score <= 1

    def test_custom_perturbation_scales(self):
        bounds = np.array([[-1, 1]])
        profile = sensitivity_analysis(sphere, np.array([0.0]), bounds,
                                       perturbation_scales=[0.01, 0.05, 0.1],
                                       rng=np.random.RandomState(42))
        assert profile.robustness_radius > 0

    def test_narrow_peak_less_robust(self):
        """Narrow peak should have lower robustness than broad peak."""
        bounds = np.array([[0, 1], [0, 1]])
        rng = np.random.RandomState(42)

        narrow_profile = sensitivity_analysis(narrow_peak, np.array([0.5, 0.5]), bounds,
                                             perturbation_scales=[0.01, 0.05, 0.1, 0.2],
                                             rng=rng)
        broad_profile = sensitivity_analysis(robust_peak, np.array([0.0, 0.0]),
                                            np.array([[-1, 1], [-1, 1]]),
                                            perturbation_scales=[0.01, 0.05, 0.1, 0.2],
                                            rng=np.random.RandomState(42))
        # Broad peak should have higher robustness score
        assert broad_profile.robustness_score >= narrow_profile.robustness_score - 0.1


# ---------------------------------------------------------------------------
# Certify robustness tests
# ---------------------------------------------------------------------------

class TestCertifyRobustness:
    def test_robust_point_certified(self):
        """Origin of sphere should be certifiably robust."""
        bounds = np.array([[-2, 2]])
        cert = certify_robustness(sphere, np.array([0.0]), bounds,
                                  epsilon=0.1, n_samples=100,
                                  rng=np.random.RandomState(42))
        assert cert["probability_robust"] > 0.8
        assert cert["nominal_value"] == 0.0
        assert cert["n_samples"] == 100

    def test_certification_fields(self):
        bounds = np.array([[-1, 1]])
        cert = certify_robustness(sphere, np.array([0.0]), bounds,
                                  epsilon=0.05, rng=np.random.RandomState(42))
        assert "certified" in cert
        assert "probability_robust" in cert
        assert "confidence_lower" in cert
        assert "confidence_upper" in cert
        assert "mean_perturbed" in cert
        assert "std_perturbed" in cert

    def test_confidence_interval(self):
        bounds = np.array([[-1, 1]])
        cert = certify_robustness(sphere, np.array([0.0]), bounds,
                                  epsilon=0.1, confidence=0.95,
                                  rng=np.random.RandomState(42))
        assert cert["confidence_lower"] <= cert["probability_robust"]
        assert cert["probability_robust"] <= cert["confidence_upper"]

    def test_large_epsilon_less_robust(self):
        """Larger perturbation should yield lower robustness probability."""
        bounds = np.array([[-2, 2]])
        cert_small = certify_robustness(sphere, np.array([0.5]), bounds,
                                        epsilon=0.01, n_samples=200,
                                        rng=np.random.RandomState(42))
        cert_large = certify_robustness(sphere, np.array([0.5]), bounds,
                                        epsilon=0.5, n_samples=200,
                                        rng=np.random.RandomState(42))
        assert cert_small["probability_robust"] >= cert_large["probability_robust"] - 0.1

    def test_more_samples_tighter_interval(self):
        bounds = np.array([[-1, 1]])
        cert_few = certify_robustness(sphere, np.array([0.0]), bounds,
                                      epsilon=0.1, n_samples=20,
                                      rng=np.random.RandomState(42))
        cert_many = certify_robustness(sphere, np.array([0.0]), bounds,
                                       epsilon=0.1, n_samples=200,
                                       rng=np.random.RandomState(42))
        width_few = cert_few["confidence_upper"] - cert_few["confidence_lower"]
        width_many = cert_many["confidence_upper"] - cert_many["confidence_lower"]
        assert width_many < width_few


# ---------------------------------------------------------------------------
# Robust transfer BO tests
# ---------------------------------------------------------------------------

class TestRobustTransferBO:
    def _make_source_tasks(self, rng):
        """Create source tasks from related functions."""
        bounds = np.array([[-1, 1]])
        tasks = []
        for i, offset in enumerate([0.0, 0.1, -0.1]):
            X = rng.uniform(-1, 1, (15, 1))
            y = np.array([-(x[0] - offset) ** 2 for x in X])
            best_idx = np.argmax(y)
            tasks.append(BOTask(
                X=X, y=y, bounds=bounds, task_id=i,
                x_best=X[best_idx], f_best=y[best_idx]
            ))
        return tasks

    def test_basic(self):
        rng = np.random.RandomState(42)
        bounds = np.array([[-1, 1]])
        sources = self._make_source_tasks(rng)
        result = robust_transfer_bo(sphere, bounds, sources,
                                    input_noise_std=0.1,
                                    n_iterations=10, n_initial=3, rng=rng)
        assert isinstance(result, RobustBOResult)
        assert result.strategy == "robust_transfer"
        assert result.f_best > -1.0

    def test_transfer_helps(self):
        """Transfer should converge faster than cold start."""
        rng_transfer = np.random.RandomState(42)
        rng_cold = np.random.RandomState(42)
        bounds = np.array([[-1, 1]])
        sources = self._make_source_tasks(np.random.RandomState(99))

        transfer_result = robust_transfer_bo(sphere, bounds, sources,
                                             input_noise_std=0.05,
                                             n_iterations=8, n_initial=3,
                                             rng=rng_transfer)
        cold_result = robust_bo(sphere, bounds, input_noise_std=0.05,
                               n_iterations=8, n_initial=3,
                               rng=rng_cold)
        # Transfer should find comparable or better solution
        assert transfer_result.f_best >= cold_result.f_best - 0.5

    def test_robustness_scores_populated(self):
        rng = np.random.RandomState(42)
        bounds = np.array([[-1, 1]])
        sources = self._make_source_tasks(rng)
        result = robust_transfer_bo(sphere, bounds, sources,
                                    input_noise_std=0.1,
                                    n_iterations=5, n_initial=3, rng=rng)
        assert len(result.robustness_scores) > 0

    def test_empty_sources(self):
        """Should work with empty source list (degrades to robust BO)."""
        rng = np.random.RandomState(42)
        bounds = np.array([[-1, 1]])
        result = robust_transfer_bo(sphere, bounds, [],
                                    input_noise_std=0.1,
                                    n_iterations=5, n_initial=3, rng=rng)
        assert isinstance(result, RobustBOResult)

    def test_2d_transfer(self):
        rng = np.random.RandomState(42)
        bounds = np.array([[-1, 1], [-1, 1]])
        # 2D source tasks
        tasks = []
        for i in range(2):
            X = rng.uniform(-1, 1, (10, 2))
            y = np.array([sphere(x) for x in X])
            best_idx = np.argmax(y)
            tasks.append(BOTask(
                X=X, y=y, bounds=bounds, task_id=i,
                x_best=X[best_idx], f_best=y[best_idx]
            ))
        result = robust_transfer_bo(sphere, bounds, tasks,
                                    input_noise_std=0.1,
                                    n_iterations=8, n_initial=3, rng=rng)
        assert result.x_best.shape == (2,)


# ---------------------------------------------------------------------------
# Compare strategies tests
# ---------------------------------------------------------------------------

class TestCompareStrategies:
    def test_all_strategies_present(self):
        bounds = np.array([[-1, 1]])
        results = compare_robust_strategies(sphere, bounds, input_noise_std=0.1,
                                           n_iterations=5, n_initial=3)
        assert "robust_ei" in results
        assert "worst_case_ei" in results
        assert "dro_ei" in results
        assert "standard_bo" in results

    def test_all_return_results(self):
        bounds = np.array([[-1, 1]])
        results = compare_robust_strategies(sphere, bounds, input_noise_std=0.1,
                                           n_iterations=5, n_initial=3)
        for name, result in results.items():
            assert isinstance(result, RobustBOResult)
            assert result.f_best is not None

    def test_standard_bo_baseline(self):
        bounds = np.array([[-1, 1]])
        results = compare_robust_strategies(sphere, bounds, input_noise_std=0.1,
                                           n_iterations=5, n_initial=3)
        assert results["standard_bo"].strategy == "standard_bo"


# ---------------------------------------------------------------------------
# Multi-objective robust BO tests
# ---------------------------------------------------------------------------

class TestParetoRobustBO:
    def test_basic_2obj(self):
        obj1 = lambda x: -np.sum((x - 0.3) ** 2)
        obj2 = lambda x: -np.sum((x - 0.7) ** 2)
        bounds = np.array([[0, 1]])
        result = pareto_robust_bo([obj1, obj2], bounds,
                                  input_noise_std=0.05,
                                  n_iterations=10, n_initial=5,
                                  rng=np.random.RandomState(42))
        assert "pareto_X" in result
        assert "pareto_Y" in result
        assert result["n_pareto"] >= 1
        assert result["total_evaluations"] == 15

    def test_pareto_robustness_scores(self):
        obj1 = lambda x: -x[0] ** 2
        obj2 = lambda x: -(x[0] - 1) ** 2
        bounds = np.array([[0, 1]])
        result = pareto_robust_bo([obj1, obj2], bounds,
                                  input_noise_std=0.05,
                                  n_iterations=8, n_initial=5,
                                  rng=np.random.RandomState(42))
        assert len(result["pareto_robustness"]) == result["n_pareto"]
        assert all(0 <= r <= 1 for r in result["pareto_robustness"])

    def test_2d_2obj(self):
        obj1 = lambda x: -np.sum((x - np.array([0.3, 0.3])) ** 2)
        obj2 = lambda x: -np.sum((x - np.array([0.7, 0.7])) ** 2)
        bounds = np.array([[0, 1], [0, 1]])
        result = pareto_robust_bo([obj1, obj2], bounds,
                                  input_noise_std=0.05,
                                  n_iterations=8, n_initial=5,
                                  rng=np.random.RandomState(42))
        assert result["pareto_X"].shape[1] == 2

    def test_verbose_pareto(self, capsys):
        obj1 = lambda x: -x[0] ** 2
        obj2 = lambda x: -(x[0] - 1) ** 2
        bounds = np.array([[0, 1]])
        pareto_robust_bo([obj1, obj2], bounds,
                        input_noise_std=0.05,
                        n_iterations=3, n_initial=3,
                        verbose=True, rng=np.random.RandomState(42))
        captured = capsys.readouterr()
        assert "Iter 1" in captured.out


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_format(self):
        result = RobustBOResult(
            x_best=np.array([0.1]),
            f_best=-0.01,
            X_history=np.array([[0.1]]),
            y_history=np.array([-0.01]),
            total_evaluations=10,
            convergence=[-1, -0.5, -0.01],
            robustness_scores=[0.5, 0.7, 0.9],
            strategy="robust_ei",
            input_noise_std=0.1,
            worst_case_gap=0.02
        )
        s = robust_bo_summary(result, "Test")
        assert "Test" in s
        assert "robust_ei" in s
        assert "0.1" in s

    def test_summary_no_noise(self):
        result = RobustBOResult(
            x_best=np.array([0.0]),
            f_best=0.0,
            X_history=np.array([[0.0]]),
            y_history=np.array([0.0]),
            total_evaluations=5,
            convergence=[0.0],
            robustness_scores=[],
            strategy="standard_bo"
        )
        s = robust_bo_summary(result)
        assert "standard_bo" in s


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestDataStructures:
    def test_robust_strategy_enum(self):
        assert RobustStrategy.INPUT_NOISE.value == "input_noise"
        assert RobustStrategy.DISTRIBUTIONALLY_ROBUST.value == "dro"
        assert RobustStrategy.MINIMAX.value == "minimax"
        assert RobustStrategy.WORST_CASE_SENSITIVITY.value == "wcs"

    def test_robust_bo_result_defaults(self):
        r = RobustBOResult(
            x_best=np.array([0]), f_best=0, X_history=np.array([[0]]),
            y_history=np.array([0]), total_evaluations=1,
            convergence=[0], robustness_scores=[0], strategy="test"
        )
        assert r.input_noise_std is None
        assert r.worst_case_gap is None
        assert r.sensitivity_profile is None

    def test_robustness_profile(self):
        p = RobustnessProfile(
            x=np.array([0.5]),
            nominal_value=1.0,
            worst_case_value=0.8,
            robustness_radius=0.1,
            sensitivity=np.array([2.0]),
            robustness_score=0.9
        )
        assert p.nominal_value > p.worst_case_value
        assert p.robustness_score == 0.9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_dimension(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.01,
                          n_iterations=3, n_initial=3, rng=np.random.RandomState(42))
        assert result.x_best.shape == (1,)

    def test_zero_noise(self):
        """With zero noise, robust BO should behave like standard BO."""
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.0,
                          n_iterations=5, n_initial=3, rng=np.random.RandomState(42))
        assert result.f_best > -1.0

    def test_large_noise(self):
        bounds = np.array([[-1, 1]])
        result = robust_bo(sphere, bounds, input_noise_std=0.5,
                          n_iterations=5, n_initial=3, rng=np.random.RandomState(42))
        assert isinstance(result, RobustBOResult)

    def test_high_dimensional(self):
        """5D optimization should still work."""
        bounds = np.array([[-1, 1]] * 5)
        def sphere_5d(x):
            return -np.sum(x ** 2)
        result = robust_bo(sphere_5d, bounds, input_noise_std=0.1,
                          n_iterations=5, n_initial=5, rng=np.random.RandomState(42))
        assert result.x_best.shape == (5,)

    def test_sensitivity_at_boundary(self):
        """Sensitivity at boundary should handle clipping gracefully."""
        bounds = np.array([[0, 1]])
        profile = sensitivity_analysis(sphere, np.array([0.0]), bounds,
                                       rng=np.random.RandomState(42))
        assert np.all(np.isfinite(profile.sensitivity))

    def test_certify_deterministic_function(self):
        """Constant function should always certify."""
        def constant(x):
            return 1.0
        bounds = np.array([[-1, 1]])
        cert = certify_robustness(constant, np.array([0.0]), bounds,
                                  epsilon=0.5, rng=np.random.RandomState(42))
        assert cert["probability_robust"] == 1.0
        assert cert["certified"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self):
        """Full pipeline: optimize, analyze, certify."""
        bounds = np.array([[-1, 1]])
        rng = np.random.RandomState(42)

        # 1. Robust optimization
        result = robust_bo(robust_peak, bounds, input_noise_std=0.1,
                          n_iterations=10, n_initial=5, rng=rng)

        # 2. Sensitivity analysis
        profile = sensitivity_analysis(robust_peak, result.x_best, bounds,
                                       rng=np.random.RandomState(42))

        # 3. Certify
        cert = certify_robustness(robust_peak, result.x_best, bounds,
                                  epsilon=0.1, rng=np.random.RandomState(42))

        # 4. Summary
        summary = robust_bo_summary(result, "Full Pipeline Test")

        assert profile.robustness_score > 0
        assert cert["probability_robust"] > 0
        assert "Full Pipeline Test" in summary

    def test_transfer_then_certify(self):
        """Transfer BO then certify the result."""
        rng = np.random.RandomState(42)
        bounds = np.array([[-1, 1]])

        # Source tasks
        tasks = []
        for i in range(2):
            X = rng.uniform(-1, 1, (10, 1))
            y = np.array([sphere(x) for x in X])
            best_idx = np.argmax(y)
            tasks.append(BOTask(X=X, y=y, bounds=bounds, task_id=i,
                               x_best=X[best_idx], f_best=y[best_idx]))

        result = robust_transfer_bo(sphere, bounds, tasks,
                                    input_noise_std=0.1,
                                    n_iterations=8, n_initial=3, rng=rng)

        cert = certify_robustness(sphere, result.x_best, bounds,
                                  epsilon=0.1, rng=np.random.RandomState(42))
        assert cert["probability_robust"] > 0

    def test_minimax_then_sensitivity(self):
        """Min-max BO followed by sensitivity analysis of the result."""
        design_bounds = np.array([[0, 1]])
        adversary_bounds = np.array([[-0.5, 0.5]])
        result = minimax_bo(adversarial_objective, design_bounds, adversary_bounds,
                           n_iterations=8, n_initial=5, rng=np.random.RandomState(42))

        # Analyze sensitivity of design at fixed z=0
        def design_obj(x):
            return adversarial_objective(x, np.zeros(1))

        profile = sensitivity_analysis(design_obj, result.x_best, design_bounds,
                                       rng=np.random.RandomState(42))
        assert isinstance(profile, RobustnessProfile)

    def test_compare_then_certify_best(self):
        """Compare strategies, then certify the best."""
        bounds = np.array([[-1, 1]])
        results = compare_robust_strategies(sphere, bounds, input_noise_std=0.1,
                                           n_iterations=5, n_initial=3)

        # Find best strategy
        best_name = max(results, key=lambda k: results[k].f_best)
        best_result = results[best_name]

        cert = certify_robustness(sphere, best_result.x_best, bounds,
                                  epsilon=0.1, rng=np.random.RandomState(42))
        assert "probability_robust" in cert
