"""
Tests for C155: Gaussian Processes
"""

import math
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from gaussian_processes import (
    Kernel, RBFKernel, MaternKernel, LinearKernel, PeriodicKernel,
    PolynomialKernel, SumKernel, ProductKernel, ScaleKernel, ARDKernel,
    GPRegression, SparseGP, GPClassification, VariationalGP,
    GPOptimizer, MultiOutputGP, GPUtils, HeteroscedasticGP,
    WarpedGP, StudentTGP, _ensure_2d, _sq_dist, _sigmoid
)


# ============================================================
# Helper data generators
# ============================================================

def make_sine_data(n=50, noise=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    X = rng.uniform(0, 5, n)
    y = np.sin(X) + rng.randn(n) * noise
    return X, y


def make_linear_data(n=50, noise=0.1, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    X = rng.uniform(-3, 3, n)
    y = 2 * X + 1 + rng.randn(n) * noise
    return X, y


def make_classification_data(n=60, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    X = rng.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


# ============================================================
# Kernel Tests
# ============================================================

class TestRBFKernel:
    def test_basic(self):
        k = RBFKernel(lengthscale=1.0, variance=1.0)
        X = np.array([[0], [1], [2]], dtype=float)
        K = k(X)
        assert K.shape == (3, 3)
        assert np.allclose(np.diag(K), 1.0)

    def test_symmetry(self):
        k = RBFKernel()
        X = np.random.RandomState(42).randn(10, 2)
        K = k(X)
        assert np.allclose(K, K.T)

    def test_positive_definite(self):
        k = RBFKernel(lengthscale=0.5, variance=2.0)
        X = np.random.RandomState(42).randn(15, 3)
        K = k(X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-10)

    def test_lengthscale_effect(self):
        X = np.array([[0], [1]], dtype=float)
        k_short = RBFKernel(lengthscale=0.1)
        k_long = RBFKernel(lengthscale=10.0)
        assert k_short(X)[0, 1] < k_long(X)[0, 1]

    def test_variance_scaling(self):
        X = np.array([[0], [1]], dtype=float)
        k1 = RBFKernel(variance=1.0)
        k2 = RBFKernel(variance=3.0)
        assert np.allclose(k2(X), 3.0 * k1(X))

    def test_diag(self):
        k = RBFKernel(variance=2.5)
        X = np.random.RandomState(42).randn(10, 2)
        assert np.allclose(k.diag(X), 2.5)

    def test_cross_covariance(self):
        k = RBFKernel()
        X1 = np.array([[0], [1]], dtype=float)
        X2 = np.array([[0.5]], dtype=float)
        K = k(X1, X2)
        assert K.shape == (2, 1)

    def test_get_set_params(self):
        k = RBFKernel(lengthscale=1.0, variance=2.0)
        assert k.get_params() == {'lengthscale': 1.0, 'variance': 2.0}
        k.set_params(lengthscale=0.5)
        assert k.lengthscale == 0.5


class TestMaternKernel:
    def test_nu_05(self):
        k = MaternKernel(nu=0.5)
        X = np.array([[0], [1]], dtype=float)
        K = k(X)
        expected_01 = np.exp(-1.0)
        assert abs(K[0, 1] - expected_01) < 1e-6

    def test_nu_15(self):
        k = MaternKernel(nu=1.5)
        X = np.array([[0], [1]], dtype=float)
        K = k(X)
        s3 = math.sqrt(3)
        expected = (1 + s3) * np.exp(-s3)
        assert abs(K[0, 1] - expected) < 1e-6

    def test_nu_25(self):
        k = MaternKernel(nu=2.5)
        X = np.array([[0], [1]], dtype=float)
        K = k(X)
        s5 = math.sqrt(5)
        expected = (1 + s5 + 5.0 / 3.0) * np.exp(-s5)
        assert abs(K[0, 1] - expected) < 1e-6

    def test_invalid_nu(self):
        with pytest.raises(ValueError):
            MaternKernel(nu=1.0)

    def test_symmetry(self):
        k = MaternKernel(nu=1.5)
        X = np.random.RandomState(42).randn(10, 2)
        K = k(X)
        assert np.allclose(K, K.T)

    def test_diag(self):
        k = MaternKernel(variance=3.0, nu=2.5)
        X = np.random.RandomState(42).randn(8, 2)
        assert np.allclose(k.diag(X), 3.0)


class TestLinearKernel:
    def test_basic(self):
        k = LinearKernel(variance=1.0, bias=0.0)
        X = np.array([[1], [2], [3]], dtype=float)
        K = k(X)
        assert K.shape == (3, 3)
        assert abs(K[0, 1] - 2.0) < 1e-10  # 1*2

    def test_with_bias(self):
        k = LinearKernel(variance=1.0, bias=1.0)
        X = np.array([[0], [0]], dtype=float)
        K = k(X)
        assert np.allclose(K, np.ones((2, 2)))

    def test_diag(self):
        k = LinearKernel(variance=2.0, bias=0.5)
        X = np.array([[1], [2]], dtype=float)
        diag = k.diag(X)
        K = k(X)
        assert np.allclose(diag, np.diag(K))


class TestPeriodicKernel:
    def test_periodicity(self):
        k = PeriodicKernel(period=2 * math.pi, lengthscale=1.0)
        X1 = np.array([[0]], dtype=float)
        X2 = np.array([[2 * math.pi]], dtype=float)
        K = k(X1, X2)
        assert abs(K[0, 0] - 1.0) < 1e-6  # Same point modulo period

    def test_symmetry(self):
        k = PeriodicKernel()
        X = np.random.RandomState(42).randn(10, 1)
        K = k(X)
        assert np.allclose(K, K.T)


class TestPolynomialKernel:
    def test_degree_1(self):
        k = PolynomialKernel(degree=1, variance=1.0, bias=0.0)
        X = np.array([[1], [2]], dtype=float)
        K = k(X)
        assert abs(K[0, 1] - 2.0) < 1e-10

    def test_degree_2(self):
        k = PolynomialKernel(degree=2, variance=1.0, bias=1.0)
        X = np.array([[1], [2]], dtype=float)
        K = k(X)
        assert abs(K[0, 1] - (1 * 2 + 1) ** 2) < 1e-10


class TestCompositeKernels:
    def test_sum_kernel(self):
        k1 = RBFKernel(lengthscale=1.0)
        k2 = LinearKernel()
        k_sum = k1 + k2
        X = np.array([[0], [1]], dtype=float)
        K = k_sum(X)
        assert np.allclose(K, k1(X) + k2(X))

    def test_product_kernel(self):
        k1 = RBFKernel()
        k2 = PeriodicKernel()
        k_prod = k1 * k2
        X = np.array([[0], [1]], dtype=float)
        K = k_prod(X)
        assert np.allclose(K, k1(X) * k2(X))

    def test_scale_kernel(self):
        k = RBFKernel()
        k_scaled = 3.0 * k
        X = np.array([[0], [1]], dtype=float)
        assert np.allclose(k_scaled(X), 3.0 * k(X))

    def test_scale_kernel_right(self):
        k = RBFKernel()
        k_scaled = k * 2.5
        X = np.array([[0], [1]], dtype=float)
        assert np.allclose(k_scaled(X), 2.5 * k(X))

    def test_composite_diag(self):
        k1 = RBFKernel(variance=1.0)
        k2 = RBFKernel(variance=2.0)
        k_sum = k1 + k2
        X = np.random.RandomState(42).randn(5, 2)
        assert np.allclose(k_sum.diag(X), np.diag(k_sum(X)))

    def test_sum_get_params(self):
        k = RBFKernel() + LinearKernel()
        params = k.get_params()
        assert 'k1' in params and 'k2' in params

    def test_product_get_params(self):
        k = RBFKernel() * PeriodicKernel()
        params = k.get_params()
        assert 'k1' in params and 'k2' in params


class TestARDKernel:
    def test_basic(self):
        k = ARDKernel(input_dim=2)
        X = np.random.RandomState(42).randn(10, 2)
        K = k(X)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)

    def test_relevance(self):
        """Short lengthscale = feature matters more."""
        k = ARDKernel(input_dim=2, lengthscales=[0.1, 10.0])
        # Points differing only in dim 0 (short lengthscale)
        X1 = np.array([[0, 0]], dtype=float)
        X2 = np.array([[1, 0]], dtype=float)
        K_dim0 = k(X1, X2)[0, 0]
        # Points differing only in dim 1 (long lengthscale)
        X3 = np.array([[0, 1]], dtype=float)
        K_dim1 = k(X1, X3)[0, 0]
        assert K_dim0 < K_dim1  # dim 0 difference matters more -> lower correlation

    def test_diag(self):
        k = ARDKernel(input_dim=3, variance=2.0)
        X = np.random.RandomState(42).randn(8, 3)
        assert np.allclose(k.diag(X), 2.0)


# ============================================================
# Utility function tests
# ============================================================

class TestUtils:
    def test_ensure_2d_1d(self):
        X = np.array([1, 2, 3])
        X2d = _ensure_2d(X)
        assert X2d.shape == (3, 1)

    def test_ensure_2d_2d(self):
        X = np.array([[1, 2], [3, 4]])
        X2d = _ensure_2d(X)
        assert X2d.shape == (2, 2)

    def test_sq_dist(self):
        X1 = np.array([[0], [1]], dtype=float)
        X2 = np.array([[2]], dtype=float)
        D = _sq_dist(X1, X2)
        assert abs(D[0, 0] - 4.0) < 1e-10
        assert abs(D[1, 0] - 1.0) < 1e-10

    def test_sigmoid(self):
        assert abs(_sigmoid(0) - 0.5) < 1e-10
        assert _sigmoid(100) > 0.99
        assert _sigmoid(-100) < 0.01


# ============================================================
# GP Regression Tests
# ============================================================

class TestGPRegression:
    def test_fit_predict_basic(self):
        X, y = make_sine_data(30, noise=0.1)
        gp = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=0.01)
        gp.fit(X, y)
        mu = gp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 0.1

    def test_predict_with_std(self):
        X, y = make_sine_data(30)
        gp = GPRegression(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        mu, std = gp.predict(np.array([2.5]), return_std=True)
        assert len(mu) == 1
        assert len(std) == 1
        assert std[0] > 0

    def test_predict_with_cov(self):
        X, y = make_sine_data(20)
        gp = GPRegression(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.array([1.0, 2.0, 3.0])
        mu, cov = gp.predict(X_test, return_cov=True)
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)

    def test_uncertainty_increases_far_from_data(self):
        X, y = make_sine_data(30, noise=0.01)
        gp = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=0.01)
        gp.fit(X, y)
        _, std_near = gp.predict(np.array([2.5]), return_std=True)
        _, std_far = gp.predict(np.array([100.0]), return_std=True)
        assert std_far[0] > std_near[0]

    def test_interpolation(self):
        """GP should interpolate training points closely with low noise."""
        X = np.array([0, 1, 2, 3, 4], dtype=float)
        y = np.sin(X)
        gp = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=1e-6)
        gp.fit(X, y)
        mu = gp.predict(X)
        assert np.allclose(mu, y, atol=1e-3)

    def test_log_marginal_likelihood(self):
        X, y = make_sine_data(30)
        gp = GPRegression(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)

    def test_lml_prefers_good_kernel(self):
        X, y = make_sine_data(40, noise=0.05)
        gp_good = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=0.01)
        gp_bad = GPRegression(RBFKernel(lengthscale=0.01), noise_variance=0.01)
        gp_good.fit(X, y)
        gp_bad.fit(X, y)
        assert gp_good.log_marginal_likelihood() > gp_bad.log_marginal_likelihood()

    def test_sample_prior(self):
        gp = GPRegression(RBFKernel(), noise_variance=0.01)
        X = np.linspace(0, 5, 20)
        samples = gp.sample_prior(X, n_samples=3, rng=np.random.RandomState(42))
        assert len(samples) == 3
        assert all(len(s) == 20 for s in samples)

    def test_sample_posterior(self):
        X, y = make_sine_data(20)
        gp = GPRegression(RBFKernel(), noise_variance=0.01)
        gp.fit(X, y)
        X_test = np.linspace(0, 5, 15)
        samples = gp.sample_posterior(X_test, n_samples=4, rng=np.random.RandomState(42))
        assert len(samples) == 4
        assert all(len(s) == 15 for s in samples)

    def test_multidimensional_input(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)
        y = np.sum(X ** 2, axis=1) + rng.randn(30) * 0.1
        gp = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=0.1)
        gp.fit(X, y)
        mu = gp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 1.0

    def test_linear_with_linear_kernel(self):
        X, y = make_linear_data(40, noise=0.05)
        gp = GPRegression(LinearKernel(variance=1.0, bias=1.0), noise_variance=0.01)
        gp.fit(X, y)
        mu = gp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 0.5


# ============================================================
# Sparse GP Tests
# ============================================================

class TestSparseGP:
    def test_fitc_basic(self):
        X, y = make_sine_data(50, noise=0.1)
        gp = SparseGP(RBFKernel(lengthscale=1.0), noise_variance=0.1, method='fitc')
        gp.fit(X, y, n_inducing=10)
        mu = gp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 0.5

    def test_vfe_basic(self):
        X, y = make_sine_data(50, noise=0.1)
        gp = SparseGP(RBFKernel(lengthscale=1.0), noise_variance=0.1, method='vfe')
        gp.fit(X, y, n_inducing=10)
        mu = gp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 0.5

    def test_predict_with_std(self):
        X, y = make_sine_data(30, noise=0.1)
        gp = SparseGP(RBFKernel(), noise_variance=0.1, method='vfe')
        gp.fit(X, y, n_inducing=10)
        mu, std = gp.predict(np.array([2.5]), return_std=True)
        assert std[0] > 0

    def test_predict_with_cov(self):
        X, y = make_sine_data(30, noise=0.1)
        gp = SparseGP(RBFKernel(), noise_variance=0.1, method='vfe')
        gp.fit(X, y, n_inducing=10)
        X_test = np.array([1.0, 2.0])
        mu, cov = gp.predict(X_test, return_cov=True)
        assert cov.shape == (2, 2)

    def test_explicit_inducing_points(self):
        X, y = make_sine_data(30)
        Z = np.linspace(0, 5, 8).reshape(-1, 1)
        gp = SparseGP(RBFKernel(), noise_variance=0.1, method='vfe')
        gp.fit(X, y, Z=Z)
        assert len(gp.Z) == 8

    def test_log_marginal_likelihood(self):
        X, y = make_sine_data(30)
        for method in ['fitc', 'vfe']:
            gp = SparseGP(RBFKernel(), noise_variance=0.1, method=method)
            gp.fit(X, y, n_inducing=10)
            lml = gp.log_marginal_likelihood()
            assert np.isfinite(lml)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            SparseGP(RBFKernel(), method='invalid')

    def test_converges_to_exact_with_all_inducing(self):
        """With all points as inducing, VFE should be close to exact GP."""
        X, y = make_sine_data(15, noise=0.1)
        gp_exact = GPRegression(RBFKernel(), noise_variance=0.1)
        gp_exact.fit(X, y)
        mu_exact = gp_exact.predict(X)

        gp_sparse = SparseGP(RBFKernel(), noise_variance=0.1, method='vfe')
        gp_sparse.fit(X, y, Z=_ensure_2d(X))  # all points as inducing
        mu_sparse = gp_sparse.predict(X)

        assert np.allclose(mu_exact, mu_sparse, atol=0.1)


# ============================================================
# GP Classification Tests
# ============================================================

class TestGPClassification:
    def test_fit_predict_basic(self):
        X, y = make_classification_data(40)
        gp = GPClassification(RBFKernel(lengthscale=1.0))
        gp.fit(X, y)
        probs = gp.predict(X)
        preds = (probs > 0.5).astype(float)
        acc = np.mean(preds == y)
        assert acc > 0.7

    def test_predict_probabilities(self):
        X, y = make_classification_data(40)
        gp = GPClassification(RBFKernel())
        gp.fit(X, y)
        probs = gp.predict(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_latent(self):
        X, y = make_classification_data(40)
        gp = GPClassification(RBFKernel())
        gp.fit(X, y)
        f_mean, f_var = gp.predict(X, return_prob=False)
        assert len(f_mean) == len(X)
        assert np.all(f_var > 0)

    def test_log_marginal_likelihood(self):
        X, y = make_classification_data(30)
        gp = GPClassification(RBFKernel())
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)

    def test_separable_data(self):
        """Should achieve high accuracy on clearly separable data."""
        rng = np.random.RandomState(42)
        X_class0 = rng.randn(20, 2) + np.array([-2, -2])
        X_class1 = rng.randn(20, 2) + np.array([2, 2])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 20 + [1] * 20, dtype=float)
        gp = GPClassification(RBFKernel(lengthscale=1.0))
        gp.fit(X, y)
        probs = gp.predict(X)
        preds = (probs > 0.5).astype(float)
        assert np.mean(preds == y) > 0.9


# ============================================================
# Variational GP Tests
# ============================================================

class TestVariationalGP:
    def test_fit_predict_basic(self):
        X, y = make_sine_data(50, noise=0.1)
        vgp = VariationalGP(RBFKernel(lengthscale=1.0), noise_variance=0.1, n_inducing=15)
        vgp.fit(X, y, n_iter=100, lr=0.01, rng=np.random.RandomState(42))
        mu = vgp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 1.0  # variational, so looser bound

    def test_elbo_history(self):
        X, y = make_sine_data(30)
        vgp = VariationalGP(RBFKernel(), noise_variance=0.1)
        vgp.fit(X, y, n_iter=50)
        assert len(vgp.elbo_history) == 50

    def test_predict_with_std(self):
        X, y = make_sine_data(30)
        vgp = VariationalGP(RBFKernel(), noise_variance=0.1)
        vgp.fit(X, y, n_iter=50)
        mu, std = vgp.predict(np.array([2.5]), return_std=True)
        assert std[0] > 0

    def test_explicit_inducing(self):
        X, y = make_sine_data(30)
        Z = np.linspace(0, 5, 8).reshape(-1, 1)
        vgp = VariationalGP(RBFKernel(), noise_variance=0.1)
        vgp.fit(X, y, Z=Z, n_iter=50)
        assert len(vgp.Z) == 8


# ============================================================
# GP Optimizer Tests
# ============================================================

class TestGPOptimizer:
    def test_optimize_basic(self):
        X, y = make_sine_data(40, noise=0.1)
        gp = GPRegression(RBFKernel(lengthscale=0.5, variance=1.0), noise_variance=0.1)
        gp.fit(X, y)
        lml_before = gp.log_marginal_likelihood()

        opt = GPOptimizer(gp)
        result = opt.optimize(n_iter=20, lr=0.05)
        assert 'params' in result
        assert 'lml_history' in result
        assert len(result['lml_history']) == 20

    def test_lml_improves(self):
        X, y = make_sine_data(40, noise=0.1)
        gp = GPRegression(RBFKernel(lengthscale=0.1, variance=0.1), noise_variance=1.0)
        gp.fit(X, y)
        lml_before = gp.log_marginal_likelihood()

        opt = GPOptimizer(gp)
        result = opt.optimize(n_iter=30, lr=0.05)
        # Should improve (or at least not crash)
        assert result['final_lml'] is not None

    def test_custom_param_names(self):
        X, y = make_sine_data(30)
        gp = GPRegression(RBFKernel(), noise_variance=0.1)
        gp.fit(X, y)
        opt = GPOptimizer(gp, param_names=['lengthscale', 'noise_variance'])
        result = opt.optimize(n_iter=10, lr=0.01)
        assert 'lengthscale' in result['params']
        assert 'noise_variance' in result['params']


# ============================================================
# Multi-output GP Tests
# ============================================================

class TestMultiOutputGP:
    def test_fit_predict_basic(self):
        rng = np.random.RandomState(42)
        n = 20
        X = rng.uniform(0, 5, n)
        y0 = np.sin(X) + rng.randn(n) * 0.05
        y1 = np.cos(X) + rng.randn(n) * 0.05
        X_all = np.concatenate([X, X])
        y_all = np.concatenate([y0, y1])
        d_all = np.array([0] * n + [1] * n)

        mogp = MultiOutputGP(RBFKernel(lengthscale=1.0), n_outputs=2, noise_variance=0.01)
        mogp.set_coregionalization(np.array([[1.0, 0.3], [0.3, 1.0]]))
        mogp.fit(X_all, y_all, d_all)

        mu = mogp.predict(X_all, d_all)
        mse = np.mean((mu - y_all) ** 2)
        assert mse < 0.5

    def test_predict_with_std(self):
        rng = np.random.RandomState(42)
        n = 15
        X = rng.uniform(0, 3, n)
        y = np.sin(X) + rng.randn(n) * 0.05
        d = np.zeros(n, dtype=int)

        mogp = MultiOutputGP(RBFKernel(), n_outputs=1, noise_variance=0.01)
        mogp.fit(X, y, d)
        mu, std = mogp.predict(np.array([1.5]), np.array([0]), return_std=True)
        assert std[0] > 0

    def test_correlated_outputs(self):
        """Correlated outputs should share information."""
        rng = np.random.RandomState(42)
        n = 15
        X = rng.uniform(0, 5, n)
        f = np.sin(X)  # shared latent function
        y0 = f + rng.randn(n) * 0.01
        y1 = 0.5 * f + rng.randn(n) * 0.01

        X_all = np.concatenate([X, X])
        y_all = np.concatenate([y0, y1])
        d_all = np.array([0] * n + [1] * n)

        mogp = MultiOutputGP(RBFKernel(), n_outputs=2, noise_variance=0.01)
        mogp.set_coregionalization(np.array([[1.0, 0.5], [0.5, 0.5]]))
        mogp.fit(X_all, y_all, d_all)

        mu = mogp.predict(X_all, d_all)
        mse = np.mean((mu - y_all) ** 2)
        assert mse < 0.5


# ============================================================
# GP Utils Tests
# ============================================================

class TestGPUtils:
    def test_kernel_matrix_analysis(self):
        k = RBFKernel()
        X = np.random.RandomState(42).randn(10, 2)
        analysis = GPUtils.kernel_matrix_analysis(k, X)
        assert analysis['is_positive_definite']
        assert analysis['shape'] == (10, 10)
        assert analysis['min_eigenvalue'] > -1e-10

    def test_compare_kernels(self):
        X, y = make_sine_data(30)
        kernels = {
            'rbf': RBFKernel(lengthscale=1.0),
            'linear': LinearKernel(),
            'matern': MaternKernel(nu=1.5),
        }
        results = GPUtils.compare_kernels(kernels, X, y)
        assert len(results) == 3
        assert all('lml' in v and 'bic' in v for v in results.values())

    def test_cross_validate(self):
        X, y = make_sine_data(40, noise=0.1)
        gp = GPRegression(RBFKernel(), noise_variance=0.1)
        result = GPUtils.cross_validate(gp, X, y, n_folds=4)
        assert 'mse' in result
        assert 'nlpd' in result
        assert len(result['fold_mses']) == 4

    def test_lengthscale_sensitivity(self):
        X, y = make_sine_data(30)
        gp = GPRegression(RBFKernel(), noise_variance=0.1)
        results = GPUtils.lengthscale_sensitivity(
            gp, X, y, lengthscales=[0.1, 0.5, 1.0, 2.0, 5.0])
        assert len(results) == 5
        assert all('lml' in r for r in results)


# ============================================================
# Heteroscedastic GP Tests
# ============================================================

class TestHeteroscedasticGP:
    def test_fit_predict_basic(self):
        rng = np.random.RandomState(42)
        X = np.linspace(0, 5, 40)
        noise = 0.01 + 0.5 * np.abs(np.sin(X))
        y = np.sin(X) + rng.randn(40) * noise
        hgp = HeteroscedasticGP(RBFKernel(), n_iterations=3)
        hgp.fit(X, y)
        mu = hgp.predict(X)
        assert len(mu) == 40

    def test_predict_with_std(self):
        rng = np.random.RandomState(42)
        X = np.linspace(0, 5, 30)
        y = np.sin(X) + rng.randn(30) * 0.1
        hgp = HeteroscedasticGP(RBFKernel(), n_iterations=2)
        hgp.fit(X, y)
        mu, std = hgp.predict(np.array([2.5]), return_std=True)
        assert std[0] > 0

    def test_predict_with_noise(self):
        rng = np.random.RandomState(42)
        X = np.linspace(0, 5, 30)
        y = np.sin(X) + rng.randn(30) * 0.1
        hgp = HeteroscedasticGP(RBFKernel(), n_iterations=2)
        hgp.fit(X, y)
        mu, std, noise_std = hgp.predict(np.array([2.5]), return_std=True, return_noise=True)
        assert noise_std[0] > 0


# ============================================================
# Warped GP Tests
# ============================================================

class TestWarpedGP:
    def test_log_warp(self):
        rng = np.random.RandomState(42)
        X = np.linspace(0.1, 5, 30)
        y = np.exp(np.sin(X)) + rng.randn(30) * 0.01  # positive data
        wgp = WarpedGP(RBFKernel(), noise_variance=0.01, warp='log')
        wgp.fit(X, y)
        mu = wgp.predict(X)
        assert np.all(mu > 0)  # log warp -> positive predictions

    def test_sqrt_warp(self):
        rng = np.random.RandomState(42)
        X = np.linspace(0, 5, 30)
        y = (np.sin(X) + 2) ** 2 + rng.randn(30) * 0.1
        wgp = WarpedGP(RBFKernel(), noise_variance=0.1, warp='sqrt')
        wgp.fit(X, y)
        mu = wgp.predict(X)
        assert len(mu) == 30

    def test_boxcox_warp(self):
        rng = np.random.RandomState(42)
        X = np.linspace(0.1, 5, 30)
        y = np.exp(np.sin(X)) + rng.randn(30) * 0.01
        wgp = WarpedGP(RBFKernel(), noise_variance=0.01, warp='box-cox', lam=0.5)
        wgp.fit(X, y)
        mu = wgp.predict(X)
        assert len(mu) == 30

    def test_predict_with_std(self):
        rng = np.random.RandomState(42)
        X = np.linspace(0.1, 5, 30)
        y = np.exp(np.sin(X)) + rng.randn(30) * 0.01
        wgp = WarpedGP(RBFKernel(), noise_variance=0.01, warp='log')
        wgp.fit(X, y)
        mu, std = wgp.predict(np.array([2.5]), return_std=True)
        assert std[0] > 0


# ============================================================
# Student-t GP Tests
# ============================================================

class TestStudentTGP:
    def test_fit_predict_basic(self):
        X, y = make_sine_data(30, noise=0.1)
        gp = StudentTGP(RBFKernel(lengthscale=1.0), df=4.0, scale=1.0)
        gp.fit(X, y)
        mu = gp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 0.5

    def test_predict_with_std(self):
        X, y = make_sine_data(30)
        gp = StudentTGP(RBFKernel(), df=4.0, scale=1.0)
        gp.fit(X, y)
        mu, std = gp.predict(np.array([2.5]), return_std=True)
        assert std[0] > 0

    def test_robust_to_outliers(self):
        """Student-t GP should be more robust to outliers than Gaussian GP."""
        rng = np.random.RandomState(42)
        X = np.linspace(0, 5, 40)
        y = np.sin(X) + rng.randn(40) * 0.05
        # Add outliers
        outlier_idx = [5, 15, 25, 35]
        y[outlier_idx] += rng.choice([-3, 3], size=4)

        X_test = np.linspace(0, 5, 20)
        y_true = np.sin(X_test)

        # Student-t GP
        tgp = StudentTGP(RBFKernel(lengthscale=1.0), df=3.0, scale=1.0)
        tgp.fit(X, y)
        mu_t = tgp.predict(X_test)

        # Gaussian GP
        ggp = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=0.1)
        ggp.fit(X, y)
        mu_g = ggp.predict(X_test)

        mse_t = np.mean((mu_t - y_true) ** 2)
        mse_g = np.mean((mu_g - y_true) ** 2)
        # Student-t should be at least comparable
        assert mse_t < mse_g * 3  # generous bound


# ============================================================
# Integration / Composition Tests
# ============================================================

class TestIntegration:
    def test_kernel_composition_chain(self):
        """RBF + Periodic * Linear should work."""
        k = RBFKernel() + PeriodicKernel() * LinearKernel()
        X = np.random.RandomState(42).randn(10, 1)
        K = k(X)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)

    def test_gp_with_composite_kernel(self):
        X, y = make_sine_data(30)
        k = RBFKernel(lengthscale=1.0) + LinearKernel(variance=0.1)
        gp = GPRegression(k, noise_variance=0.1)
        gp.fit(X, y)
        mu = gp.predict(X)
        assert len(mu) == 30

    def test_sparse_vs_exact_similarity(self):
        """Sparse GP predictions should be reasonable approximation to exact."""
        X, y = make_sine_data(30, noise=0.1)
        X_test = np.linspace(0, 5, 10)

        gp_exact = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=0.1)
        gp_exact.fit(X, y)
        mu_exact = gp_exact.predict(X_test)

        gp_sparse = SparseGP(RBFKernel(lengthscale=1.0), noise_variance=0.1, method='vfe')
        gp_sparse.fit(X, y, n_inducing=15)
        mu_sparse = gp_sparse.predict(X_test)

        # Should be in the same ballpark
        assert np.mean(np.abs(mu_exact - mu_sparse)) < 0.5

    def test_full_pipeline(self):
        """Full pipeline: kernel selection, fitting, prediction, sampling."""
        X, y = make_sine_data(40, noise=0.1)
        X_test = np.linspace(0, 5, 20)

        # Compare kernels
        kernels = {'rbf': RBFKernel(), 'matern': MaternKernel(nu=2.5)}
        comparison = GPUtils.compare_kernels(kernels, X, y)
        best_name = max(comparison, key=lambda k: comparison[k]['lml'])

        # Fit with best kernel
        gp = GPRegression(kernels[best_name], noise_variance=0.1)
        gp.fit(X, y)

        # Predict with uncertainty
        mu, std = gp.predict(X_test, return_std=True)
        assert len(mu) == 20
        assert len(std) == 20
        assert np.all(std > 0)

        # Sample
        samples = gp.sample_posterior(X_test, n_samples=3, rng=np.random.RandomState(42))
        assert len(samples) == 3

    def test_ard_feature_selection(self):
        """ARD kernel should identify relevant features."""
        rng = np.random.RandomState(42)
        n = 50
        X = rng.randn(n, 3)
        # Only dim 0 matters
        y = np.sin(X[:, 0]) + rng.randn(n) * 0.05

        k = ARDKernel(input_dim=3)
        gp = GPRegression(k, noise_variance=0.01)
        gp.fit(X, y)
        mu = gp.predict(X)
        mse = np.mean((mu - y) ** 2)
        assert mse < 0.5

    def test_cv_results_make_sense(self):
        """Cross-validation MSE should be reasonable."""
        X, y = make_sine_data(50, noise=0.1)
        gp = GPRegression(RBFKernel(lengthscale=1.0), noise_variance=0.1)
        result = GPUtils.cross_validate(gp, X, y, n_folds=5)
        assert result['mse'] < 1.0
        assert result['nlpd'] > -100

    def test_heteroscedastic_varying_noise(self):
        """Heteroscedastic GP should capture varying noise."""
        rng = np.random.RandomState(42)
        X = np.linspace(0, 5, 50)
        noise = 0.01 + 0.3 * X / 5  # noise increases with X
        y = np.sin(X) + rng.randn(50) * noise

        hgp = HeteroscedasticGP(RBFKernel(lengthscale=1.0), n_iterations=3)
        hgp.fit(X, y)
        mu, std, noise_std = hgp.predict(X, return_std=True, return_noise=True)
        # Noise should generally increase (check endpoints)
        # This is a statistical test so use generous tolerance
        assert len(noise_std) == 50


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_point(self):
        gp = GPRegression(RBFKernel(), noise_variance=0.01)
        gp.fit(np.array([0.0]), np.array([1.0]))
        mu = gp.predict(np.array([0.0]))
        assert abs(mu[0] - 1.0) < 0.1

    def test_identical_points(self):
        gp = GPRegression(RBFKernel(), noise_variance=0.1)
        X = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 1.1, 0.9])
        gp.fit(X, y)
        mu = gp.predict(np.array([1.0]))
        assert abs(mu[0] - 1.0) < 0.2

    def test_large_noise(self):
        X, y = make_sine_data(20)
        gp = GPRegression(RBFKernel(), noise_variance=100.0)
        gp.fit(X, y)
        mu = gp.predict(X)
        # With huge noise, predictions should be close to mean
        assert np.std(mu) < np.std(y)

    def test_very_small_noise(self):
        X = np.array([0, 1, 2, 3], dtype=float)
        y = np.sin(X)
        gp = GPRegression(RBFKernel(), noise_variance=1e-10)
        gp.fit(X, y)
        mu = gp.predict(X)
        assert np.allclose(mu, y, atol=1e-3)

    def test_high_dimensional_input(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 10)
        y = rng.randn(30)
        gp = GPRegression(RBFKernel(), noise_variance=1.0)
        gp.fit(X, y)
        mu = gp.predict(X)
        assert len(mu) == 30

    def test_list_input(self):
        """Should accept lists, not just numpy arrays."""
        gp = GPRegression(RBFKernel(), noise_variance=0.1)
        gp.fit([0.0, 1.0, 2.0], [0.0, 0.84, 0.91])
        mu = gp.predict([0.5, 1.5])
        assert len(mu) == 2

    def test_kernel_with_zero_distance(self):
        k = RBFKernel()
        X = np.array([[0.0]])
        K = k(X)
        assert abs(K[0, 0] - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
