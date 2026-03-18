"""Tests for V222: Gaussian Process Regression."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from gaussian_process import (
    RBFKernel, Matern32Kernel, Matern52Kernel, LinearKernel,
    PeriodicKernel, RationalQuadraticKernel, WhiteNoiseKernel, ARDKernel,
    SumKernel, ProductKernel, ScaleKernel,
    GaussianProcess, GPPrediction, GPResult,
    SparseGP, MultiOutputGP, HeteroscedasticGP, WarpedGP,
    cross_validate_kernel, _sq_dist,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def sin_data(rng):
    X = rng.uniform(-3, 3, (30, 1))
    y = np.sin(X.ravel()) + rng.normal(0, 0.1, 30)
    return X, y

@pytest.fixture
def linear_data(rng):
    X = rng.uniform(0, 5, (20, 1))
    y = 2.0 * X.ravel() + 1.0 + rng.normal(0, 0.1, 20)
    return X, y

@pytest.fixture
def multi_d_data(rng):
    X = rng.uniform(-2, 2, (40, 3))
    y = X[:, 0] + 0.5 * X[:, 1]**2 - X[:, 2] + rng.normal(0, 0.1, 40)
    return X, y


# ===========================================================================
# KERNEL TESTS
# ===========================================================================

class TestRBFKernel:
    def test_shape(self):
        k = RBFKernel()
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.5], [1.5]])
        K = k(X1, X2)
        assert K.shape == (3, 2)

    def test_positive_definite(self, rng):
        k = RBFKernel(length_scale=1.0, variance=2.0)
        X = rng.uniform(-5, 5, (20, 2))
        K = k(X, X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_symmetry(self, rng):
        k = RBFKernel()
        X = rng.uniform(-3, 3, (10, 2))
        K = k(X, X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_diagonal_is_variance(self, rng):
        k = RBFKernel(variance=3.0)
        X = rng.uniform(-2, 2, (15, 1))
        K = k(X, X)
        np.testing.assert_allclose(np.diag(K), 3.0, atol=1e-12)

    def test_diag_method(self, rng):
        k = RBFKernel(variance=2.5)
        X = rng.uniform(-2, 2, (10, 1))
        np.testing.assert_allclose(k.diag(X), np.diag(k(X, X)), atol=1e-12)

    def test_params_roundtrip(self):
        k = RBFKernel(length_scale=2.0, variance=3.0)
        p = k.params()
        k2 = RBFKernel()
        k2.set_params(p)
        assert abs(k2.length_scale - 2.0) < 1e-10
        assert abs(k2.variance - 3.0) < 1e-10

    def test_length_scale_effect(self):
        k_short = RBFKernel(length_scale=0.1)
        k_long = RBFKernel(length_scale=10.0)
        X = np.array([[0.0], [1.0]])
        # Short length scale -> lower correlation at distance 1
        assert k_short(X, X)[0, 1] < k_long(X, X)[0, 1]


class TestMatern32Kernel:
    def test_positive_definite(self, rng):
        k = Matern32Kernel()
        X = rng.uniform(-3, 3, (15, 2))
        K = k(X, X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_diagonal(self):
        k = Matern32Kernel(variance=2.0)
        X = np.array([[0.0], [1.0]])
        np.testing.assert_allclose(k.diag(X), 2.0)

    def test_at_zero_distance(self):
        k = Matern32Kernel(variance=4.0)
        X = np.array([[1.0]])
        np.testing.assert_allclose(k(X, X), [[4.0]], atol=1e-10)


class TestMatern52Kernel:
    def test_positive_definite(self, rng):
        k = Matern52Kernel()
        X = rng.uniform(-3, 3, (15, 2))
        K = k(X, X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_params(self):
        k = Matern52Kernel(length_scale=0.5, variance=2.0)
        assert k.n_params() == 2
        p = k.params()
        np.testing.assert_allclose(np.exp(p), [0.5, 2.0])


class TestLinearKernel:
    def test_linear_covariance(self, rng):
        k = LinearKernel(variance=1.0, bias_variance=0.0, center=0.0)
        X = rng.uniform(-2, 2, (10, 2))
        K = k(X, X)
        np.testing.assert_allclose(K, X @ X.T, atol=1e-10)

    def test_with_bias(self):
        k = LinearKernel(variance=1.0, bias_variance=1.0, center=0.0)
        X = np.array([[1.0], [2.0]])
        K = k(X, X)
        expected = X @ X.T + 1.0
        np.testing.assert_allclose(K, expected, atol=1e-10)


class TestPeriodicKernel:
    def test_periodicity(self):
        k = PeriodicKernel(period=2.0, variance=1.0, length_scale=1.0)
        X1 = np.array([[0.0]])
        X2 = np.array([[2.0]])
        X3 = np.array([[4.0]])
        # k(0, 2) should equal k(0, 4) due to period=2
        np.testing.assert_allclose(k(X1, X2), k(X1, X3), atol=1e-10)

    def test_n_params(self):
        k = PeriodicKernel()
        assert k.n_params() == 3


class TestRationalQuadraticKernel:
    def test_positive_definite(self, rng):
        k = RationalQuadraticKernel()
        X = rng.uniform(-3, 3, (15, 2))
        K = k(X, X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_params(self):
        k = RationalQuadraticKernel(length_scale=2.0, variance=3.0, alpha=0.5)
        assert k.n_params() == 3


class TestWhiteNoiseKernel:
    def test_identity_for_same_points(self):
        k = WhiteNoiseKernel(variance=2.0)
        X = np.array([[0.0], [1.0], [2.0]])
        K = k(X, X)
        np.testing.assert_allclose(K, 2.0 * np.eye(3), atol=1e-10)

    def test_zero_for_different_points(self):
        k = WhiteNoiseKernel(variance=2.0)
        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[2.0], [3.0]])
        K = k(X1, X2)
        np.testing.assert_allclose(K, np.zeros((2, 2)), atol=1e-10)


class TestARDKernel:
    def test_per_dim_length_scales(self, rng):
        ls = np.array([0.5, 2.0, 1.0])
        k = ARDKernel(ls, variance=1.0)
        X = rng.uniform(-2, 2, (10, 3))
        K = k(X, X)
        assert K.shape == (10, 10)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_n_params(self):
        k = ARDKernel(np.ones(5))
        assert k.n_params() == 6  # 5 length scales + 1 variance


class TestKernelComposition:
    def test_sum(self, rng):
        k1 = RBFKernel(variance=1.0)
        k2 = LinearKernel(variance=1.0)
        k_sum = k1 + k2
        X = rng.uniform(-2, 2, (10, 1))
        K = k_sum(X, X)
        np.testing.assert_allclose(K, k1(X, X) + k2(X, X), atol=1e-10)

    def test_product(self, rng):
        k1 = RBFKernel(variance=1.0)
        k2 = PeriodicKernel(variance=1.0)
        k_prod = k1 * k2
        X = rng.uniform(-2, 2, (10, 1))
        K = k_prod(X, X)
        np.testing.assert_allclose(K, k1(X, X) * k2(X, X), atol=1e-10)

    def test_scale(self, rng):
        k = RBFKernel(variance=1.0)
        k_scaled = 3.0 * k
        X = rng.uniform(-2, 2, (10, 1))
        np.testing.assert_allclose(k_scaled(X, X), 3.0 * k(X, X), atol=1e-10)

    def test_composition_params(self):
        k1 = RBFKernel()
        k2 = Matern32Kernel()
        k_sum = k1 + k2
        assert k_sum.n_params() == 4
        p = k_sum.params()
        assert len(p) == 4

    def test_nested_composition(self, rng):
        k = (RBFKernel() + LinearKernel()) * PeriodicKernel()
        X = rng.uniform(-2, 2, (8, 1))
        K = k(X, X)
        assert K.shape == (8, 8)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)


# ===========================================================================
# EXACT GP REGRESSION TESTS
# ===========================================================================

class TestGPFit:
    def test_fit_predict_shape(self, sin_data):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
        pred = gp.predict(X_test)
        assert pred.mean.shape == (50,)
        assert pred.variance.shape == (50,)
        assert pred.std.shape == (50,)

    def test_interpolation_at_training_points(self, sin_data):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(length_scale=0.5, variance=1.0), noise_variance=1e-6)
        gp.fit(X, y)
        pred = gp.predict(X)
        # With tiny noise, predictions at training points should match y closely
        np.testing.assert_allclose(pred.mean, y, atol=0.25)

    def test_uncertainty_increases_away_from_data(self, rng):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp = GaussianProcess(RBFKernel(length_scale=1.0), noise_variance=1e-6)
        gp.fit(X, y)
        pred_near = gp.predict(np.array([[0.5]]))
        pred_far = gp.predict(np.array([[10.0]]))
        assert pred_far.variance[0] > pred_near.variance[0]

    def test_variance_nonnegative(self, sin_data, rng):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = rng.uniform(-5, 5, (100, 1))
        pred = gp.predict(X_test)
        assert np.all(pred.variance >= 0)

    def test_confidence_interval(self, sin_data):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
        pred = gp.predict(X_test)
        lo, hi = pred.confidence_interval(2.0)
        assert np.all(lo < hi)
        assert np.all(lo <= pred.mean)
        assert np.all(hi >= pred.mean)


class TestGPLogMarginalLikelihood:
    def test_lml_finite(self, sin_data):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)

    def test_lml_prefers_good_fit(self, linear_data):
        X, y = linear_data
        gp_good = GaussianProcess(LinearKernel(), noise_variance=0.01)
        gp_good.fit(X, y)
        gp_bad = GaussianProcess(PeriodicKernel(period=0.1), noise_variance=0.01)
        gp_bad.fit(X, y)
        # Linear kernel should fit linear data better
        assert gp_good.log_marginal_likelihood() > gp_bad.log_marginal_likelihood()

    def test_lml_with_explicit_data(self, sin_data):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        lml1 = gp.log_marginal_likelihood()
        lml2 = gp.log_marginal_likelihood(X, y)
        np.testing.assert_allclose(lml1, lml2, atol=1e-8)


class TestGPSampling:
    def test_prior_samples_shape(self, rng):
        gp = GaussianProcess(RBFKernel())
        X = np.linspace(-3, 3, 20).reshape(-1, 1)
        samples = gp.sample_prior(X, n_samples=5, rng=rng)
        assert samples.shape == (5, 20)

    def test_posterior_samples_shape(self, sin_data, rng):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.linspace(-3, 3, 15).reshape(-1, 1)
        samples = gp.sample_posterior(X_test, n_samples=3, rng=rng)
        assert samples.shape == (3, 15)

    def test_posterior_samples_near_mean(self, sin_data, rng):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(length_scale=0.5), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.linspace(-2, 2, 10).reshape(-1, 1)
        pred = gp.predict(X_test)
        samples = gp.sample_posterior(X_test, n_samples=100, rng=rng)
        sample_mean = samples.mean(axis=0)
        # Sample mean should be close to posterior mean
        np.testing.assert_allclose(sample_mean, pred.mean, atol=0.5)


class TestGPOptimize:
    def test_optimize_improves_lml(self, sin_data, rng):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(length_scale=0.01, variance=0.01), noise_variance=1.0)
        gp.fit(X, y)
        lml_before = gp.log_marginal_likelihood()
        result = gp.optimize(X, y, n_restarts=2, rng=rng)
        assert result["log_marginal_likelihood"] >= lml_before - 1.0  # Should generally improve

    def test_optimize_returns_dict(self, sin_data, rng):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.1)
        result = gp.optimize(X, y, n_restarts=1, rng=rng)
        assert "log_marginal_likelihood" in result
        assert "kernel_params" in result
        assert "noise_variance" in result

    def test_optimize_finds_reasonable_params(self, linear_data, rng):
        X, y = linear_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.1)
        result = gp.optimize(X, y, n_restarts=2, rng=rng)
        # After optimization, noise should be small
        assert result["noise_variance"] < 1.0


class TestGPFullResult:
    def test_full_result_fields(self, sin_data):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.linspace(-3, 3, 10).reshape(-1, 1)
        result = gp.full_result(X_test)
        assert isinstance(result, GPResult)
        assert result.prediction.mean.shape == (10,)
        assert result.alpha.shape == (30,)
        assert result.L.shape == (30, 30)
        assert np.isfinite(result.log_marginal_likelihood)


class TestGPCov:
    def test_return_cov(self, sin_data):
        X, y = sin_data
        gp = GaussianProcess(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.linspace(-2, 2, 8).reshape(-1, 1)
        pred = gp.predict(X_test, return_cov=True)
        assert hasattr(pred, "cov")
        assert pred.cov.shape == (8, 8)
        # Cov diagonal should match variance
        np.testing.assert_allclose(np.diag(pred.cov), pred.variance, atol=1e-6)


class TestGPMeanFunction:
    def test_linear_mean(self, linear_data, rng):
        X, y = linear_data
        def mean_fn(X):
            return 2.0 * X.ravel()
        gp = GaussianProcess(RBFKernel(), noise_variance=0.1, mean_function=mean_fn)
        gp.fit(X, y)
        pred = gp.predict(X)
        # Predictions should be reasonable
        assert np.mean(np.abs(pred.mean - y)) < 1.0


class TestGPMultiDimensional:
    def test_multi_d_input(self, multi_d_data):
        X, y = multi_d_data
        gp = GaussianProcess(RBFKernel(length_scale=1.0), noise_variance=0.1)
        gp.fit(X, y)
        pred = gp.predict(X)
        # Should explain some variance
        residuals = np.var(pred.mean - y)
        assert residuals < np.var(y)

    def test_ard_kernel_multi_d(self, multi_d_data, rng):
        X, y = multi_d_data
        k = ARDKernel(np.ones(3), variance=1.0)
        gp = GaussianProcess(k, noise_variance=0.1)
        gp.fit(X, y)
        pred = gp.predict(X)
        assert pred.mean.shape == (40,)


# ===========================================================================
# SPARSE GP TESTS
# ===========================================================================

class TestSparseGP:
    def test_fit_predict(self, sin_data, rng):
        X, y = sin_data
        sgp = SparseGP(RBFKernel(), noise_variance=0.1, n_inducing=10)
        sgp.fit(X, y, rng=rng)
        X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
        pred = sgp.predict(X_test)
        assert pred.mean.shape == (20,)
        assert pred.variance.shape == (20,)
        assert np.all(pred.variance >= 0)

    def test_sparse_approximates_exact(self, sin_data, rng):
        X, y = sin_data
        # Exact GP
        gp = GaussianProcess(RBFKernel(length_scale=1.0), noise_variance=0.1)
        gp.fit(X, y)
        # Sparse GP with many inducing points (should be close to exact)
        sgp = SparseGP(RBFKernel(length_scale=1.0), noise_variance=0.1, n_inducing=25)
        sgp.fit(X, y, rng=rng)
        X_test = np.linspace(-2, 2, 10).reshape(-1, 1)
        pred_exact = gp.predict(X_test)
        pred_sparse = sgp.predict(X_test)
        # Means should be similar
        np.testing.assert_allclose(pred_sparse.mean, pred_exact.mean, atol=0.5)

    def test_custom_inducing_points(self, sin_data):
        X, y = sin_data
        Z = np.linspace(-3, 3, 8).reshape(-1, 1)
        sgp = SparseGP(RBFKernel(), noise_variance=0.1)
        sgp.fit(X, y, Z=Z)
        assert sgp._Z.shape == (8, 1)

    def test_lml_finite(self, sin_data, rng):
        X, y = sin_data
        sgp = SparseGP(RBFKernel(), noise_variance=0.1, n_inducing=10)
        sgp.fit(X, y, rng=rng)
        lml = sgp.log_marginal_likelihood()
        assert np.isfinite(lml)


# ===========================================================================
# MULTI-OUTPUT GP TESTS
# ===========================================================================

class TestMultiOutputGP:
    def test_fit_predict(self, rng):
        X = rng.uniform(-2, 2, (20, 1))
        Y = np.column_stack([np.sin(X.ravel()), np.cos(X.ravel())])
        Y += rng.normal(0, 0.1, Y.shape)
        mgp = MultiOutputGP(RBFKernel(), n_outputs=2, noise_variance=0.1)
        mgp.fit(X, Y)
        X_test = np.linspace(-2, 2, 10).reshape(-1, 1)
        pred = mgp.predict(X_test)
        assert pred.mean.shape == (10, 2)
        assert pred.variance.shape == (10, 2)

    def test_B_matrix(self):
        mgp = MultiOutputGP(RBFKernel(), n_outputs=3)
        B = mgp.B
        assert B.shape == (3, 3)
        # B should be positive semi-definite
        eigvals = np.linalg.eigvalsh(B)
        assert np.all(eigvals >= -1e-10)

    def test_variance_nonneg(self, rng):
        X = rng.uniform(-2, 2, (15, 1))
        Y = np.column_stack([X.ravel()**2, -X.ravel()])
        mgp = MultiOutputGP(RBFKernel(), n_outputs=2, noise_variance=0.1)
        mgp.fit(X, Y)
        pred = mgp.predict(rng.uniform(-3, 3, (8, 1)))
        assert np.all(pred.variance >= 0)


# ===========================================================================
# HETEROSCEDASTIC GP TESTS
# ===========================================================================

class TestHeteroscedasticGP:
    def test_fit_predict(self, rng):
        X = np.linspace(-3, 3, 40).reshape(-1, 1)
        noise = 0.1 + 0.5 * np.abs(X.ravel())  # Noise increases with |x|
        y = np.sin(X.ravel()) + rng.normal(0, noise)
        hgp = HeteroscedasticGP(RBFKernel(), n_iterations=3)
        hgp.fit(X, y)
        X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
        pred = hgp.predict(X_test)
        assert pred.mean.shape == (20,)
        assert np.all(pred.variance >= 0)

    def test_heteroscedastic_captures_varying_noise(self, rng):
        X = np.linspace(0, 5, 50).reshape(-1, 1)
        noise_var = 0.01 + 0.5 * X.ravel()**2  # Quadratically increasing noise
        y = X.ravel() + rng.normal(0, np.sqrt(noise_var))
        hgp = HeteroscedasticGP(RBFKernel(length_scale=2.0), n_iterations=5)
        hgp.fit(X, y)
        pred_low = hgp.predict(np.array([[0.5]]))
        pred_high = hgp.predict(np.array([[4.5]]))
        # Higher uncertainty at x=4.5 than x=0.5
        assert pred_high.variance[0] > pred_low.variance[0]


# ===========================================================================
# WARPED GP TESTS
# ===========================================================================

class TestWarpedGP:
    def test_log_warp(self, rng):
        X = rng.uniform(0.1, 3, (30, 1))
        y = np.exp(np.sin(X.ravel())) + rng.normal(0, 0.1, 30)
        y = np.abs(y)  # Ensure positive
        wgp = WarpedGP(RBFKernel(), noise_variance=0.1, warp="log")
        wgp.fit(X, y)
        pred = wgp.predict(X)
        assert pred.mean.shape == (30,)
        assert np.all(pred.variance >= 0)

    def test_sqrt_warp(self, rng):
        X = rng.uniform(0, 5, (25, 1))
        y = X.ravel()**2 + rng.normal(0, 0.5, 25)
        y = np.abs(y)
        wgp = WarpedGP(RBFKernel(), noise_variance=0.1, warp="sqrt")
        wgp.fit(X, y)
        pred = wgp.predict(X)
        assert pred.mean.shape == (25,)

    def test_boxcox_warp(self, rng):
        X = rng.uniform(0.5, 3, (20, 1))
        y = np.exp(X.ravel()) + rng.normal(0, 0.1, 20)
        y = np.abs(y)
        wgp = WarpedGP(RBFKernel(), warp="boxcox")
        wgp.fit(X, y)
        pred = wgp.predict(X)
        assert np.all(pred.mean > 0)  # Inverse warp should produce positive values


# ===========================================================================
# CROSS VALIDATION TESTS
# ===========================================================================

class TestCrossValidation:
    def test_selects_best_kernel(self, rng):
        X = rng.uniform(0, 5, (40, 1))
        y = 3.0 * X.ravel() + 1.0 + rng.normal(0, 0.1, 40)
        kernels = [RBFKernel(), LinearKernel(variance=1.0, bias_variance=10.0), PeriodicKernel(period=0.5)]
        result = cross_validate_kernel(kernels, X, y, n_folds=4, noise_variance=0.1, rng=rng)
        assert "best_kernel_index" in result
        assert "scores" in result
        # RBF and Linear (with intercept) should both score well; periodic should be worst
        assert result["scores"][2] < result["scores"][0]  # Periodic < RBF

    def test_returns_all_scores(self, sin_data, rng):
        X, y = sin_data
        kernels = [RBFKernel(), Matern52Kernel()]
        result = cross_validate_kernel(kernels, X, y, n_folds=3, rng=rng)
        assert len(result["scores"]) == 2
        assert all(np.isfinite(v) for v in result["scores"].values())


# ===========================================================================
# UTILITY TESTS
# ===========================================================================

class TestSqDist:
    def test_self_distance(self):
        X = np.array([[0.0], [1.0], [2.0]])
        D = _sq_dist(X, X)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_known_distances(self):
        X1 = np.array([[0.0, 0.0]])
        X2 = np.array([[3.0, 4.0]])
        D = _sq_dist(X1, X2)
        np.testing.assert_allclose(D, [[25.0]], atol=1e-10)

    def test_shape(self):
        X1 = np.array([[0.0], [1.0]])
        X2 = np.array([[2.0], [3.0], [4.0]])
        D = _sq_dist(X1, X2)
        assert D.shape == (2, 3)


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

class TestGPIntegration:
    def test_sin_regression_quality(self, rng):
        """GP should fit sin(x) well with reasonable kernel."""
        X = rng.uniform(-np.pi, np.pi, (40, 1))
        y = np.sin(X.ravel())
        gp = GaussianProcess(RBFKernel(length_scale=1.0, variance=1.0), noise_variance=1e-4)
        gp.fit(X, y)
        X_test = np.linspace(-np.pi, np.pi, 50).reshape(-1, 1)
        pred = gp.predict(X_test)
        rmse = np.sqrt(np.mean((pred.mean - np.sin(X_test.ravel()))**2))
        assert rmse < 0.15

    def test_linear_regression_quality(self, rng):
        """Linear kernel on linear data."""
        X = rng.uniform(0, 10, (30, 1))
        y = 2.5 * X.ravel() - 1.0 + rng.normal(0, 0.1, 30)
        gp = GaussianProcess(LinearKernel(variance=10.0, bias_variance=10.0), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.array([[5.0]])
        pred = gp.predict(X_test)
        assert abs(pred.mean[0] - 11.5) < 2.0  # 2.5*5 - 1 = 11.5

    def test_periodic_data(self, rng):
        """Periodic kernel on periodic data."""
        X = rng.uniform(0, 4 * np.pi, (50, 1))
        y = np.sin(X.ravel()) + rng.normal(0, 0.05, 50)
        k = PeriodicKernel(period=2 * np.pi, length_scale=1.0, variance=1.0)
        gp = GaussianProcess(k, noise_variance=0.01)
        gp.fit(X, y)
        # Test extrapolation: should predict well at 5*pi
        X_test = np.array([[5 * np.pi]])
        pred = gp.predict(X_test)
        expected = np.sin(5 * np.pi)
        assert abs(pred.mean[0] - expected) < 0.5

    def test_composite_kernel(self, rng):
        """Trending + periodic data with composite kernel."""
        X = rng.uniform(0, 10, (60, 1))
        y = 0.5 * X.ravel() + np.sin(2 * X.ravel()) + rng.normal(0, 0.1, 60)
        k = LinearKernel(variance=1.0) + RBFKernel(length_scale=0.5, variance=1.0)
        gp = GaussianProcess(k, noise_variance=0.1)
        gp.fit(X, y)
        pred = gp.predict(X)
        rmse = np.sqrt(np.mean((pred.mean - y)**2))
        assert rmse < 1.5  # Should capture both trend and oscillation

    def test_gp_with_many_points(self, rng):
        """Ensure GP handles moderate-sized datasets."""
        n = 200
        X = rng.uniform(-5, 5, (n, 1))
        y = np.sin(X.ravel()) + rng.normal(0, 0.1, n)
        gp = GaussianProcess(RBFKernel(), noise_variance=0.1)
        gp.fit(X, y)
        pred = gp.predict(np.array([[0.0]]))
        assert abs(pred.mean[0]) < 0.5  # sin(0) = 0

    def test_sparse_on_large_data(self, rng):
        """Sparse GP handles larger datasets efficiently."""
        n = 200
        X = rng.uniform(-5, 5, (n, 1))
        y = np.sin(X.ravel()) + rng.normal(0, 0.1, n)
        sgp = SparseGP(RBFKernel(), noise_variance=0.1, n_inducing=30)
        sgp.fit(X, y, rng=rng)
        pred = sgp.predict(np.array([[0.0]]))
        assert abs(pred.mean[0]) < 1.0

    def test_multi_output_correlated(self, rng):
        """Multi-output GP captures correlations."""
        X = rng.uniform(-2, 2, (30, 1))
        f1 = np.sin(X.ravel())
        f2 = np.cos(X.ravel())  # Correlated with f1
        Y = np.column_stack([f1, f2]) + rng.normal(0, 0.05, (30, 2))
        mgp = MultiOutputGP(RBFKernel(length_scale=1.0), n_outputs=2, noise_variance=0.1)
        mgp.fit(X, Y)
        X_test = np.array([[0.0]])
        pred = mgp.predict(X_test)
        # sin(0) ~ 0, cos(0) ~ 1
        assert abs(pred.mean[0, 0]) < 0.5
        assert abs(pred.mean[0, 1] - 1.0) < 0.5
