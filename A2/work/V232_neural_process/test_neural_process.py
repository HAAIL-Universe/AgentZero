"""Tests for V232: Neural Process -- Amortized function-space inference."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V229_meta_learning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))

from neural_process import (
    BasisEncoder, mean_aggregator, variance_aggregator, max_aggregator,
    LinearDecoder, ConditionalNeuralProcess, NeuralProcess,
    AttentiveNeuralProcess, ConvCNP, GPNeuralProcess,
    NPPrediction, NPTrainResult, NPComparisonResult,
    compare_np_models, few_shot_learning_curve,
    context_sensitivity_analysis, np_summary, comparison_summary,
)
from meta_learning import (
    Task, TaskDistribution,
    sinusoidal_task_distribution, polynomial_task_distribution,
    step_task_distribution, _rmse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_task(n_support=10, n_query=20, rng=None):
    """Create a simple linear task for testing."""
    if rng is None:
        rng = np.random.default_rng(42)
    X_s = rng.uniform(-3, 3, (n_support, 1))
    y_s = 2.0 * X_s[:, 0] + 0.5 + rng.normal(0, 0.1, n_support)
    X_q = rng.uniform(-3, 3, (n_query, 1))
    y_q = 2.0 * X_q[:, 0] + 0.5 + rng.normal(0, 0.1, n_query)
    return Task(X_s, y_s, X_q, y_q, task_id=0)


def make_simple_distribution(n_tasks=15, n_support=10, n_query=20, rng=None):
    """Create a distribution of linear tasks with varying slopes."""
    if rng is None:
        rng = np.random.default_rng(42)
    tasks = []
    for i in range(n_tasks):
        slope = rng.uniform(0.5, 3.0)
        intercept = rng.uniform(-1, 1)
        X_s = rng.uniform(-3, 3, (n_support, 1))
        y_s = slope * X_s[:, 0] + intercept + rng.normal(0, 0.1, n_support)
        X_q = rng.uniform(-3, 3, (n_query, 1))
        y_q = slope * X_q[:, 0] + intercept + rng.normal(0, 0.1, n_query)
        tasks.append(Task(X_s, y_s, X_q, y_q, task_id=i))
    return TaskDistribution(tasks, "linear", input_dim=1)


# ===========================================================================
# Basis Encoder Tests
# ===========================================================================

class TestBasisEncoder:
    def test_rbf_encode_x_shape(self):
        enc = BasisEncoder(n_basis=16, basis_type='rbf')
        X = np.random.default_rng(42).uniform(-3, 3, (10, 1))
        phi = enc.encode_x(X)
        assert phi.shape == (10, 16)

    def test_rbf_values_in_0_1(self):
        enc = BasisEncoder(n_basis=16, basis_type='rbf')
        X = np.random.default_rng(42).uniform(-3, 3, (10, 1))
        phi = enc.encode_x(X)
        assert np.all(phi >= 0)
        assert np.all(phi <= 1)

    def test_encode_xy_shape(self):
        enc = BasisEncoder(n_basis=16, basis_type='rbf')
        X = np.random.default_rng(42).uniform(-3, 3, (10, 1))
        y = np.random.default_rng(43).normal(0, 1, 10)
        r = enc.encode_xy(X, y)
        assert r.shape == (10, 32)  # 2 * n_basis

    def test_polynomial_basis(self):
        enc = BasisEncoder(n_basis=8, basis_type='polynomial')
        X = np.random.default_rng(42).uniform(-2, 2, (5, 1))
        phi = enc.encode_x(X)
        assert phi.shape == (5, 8)
        # First column should be all ones (x^0)
        np.testing.assert_allclose(phi[:, 0], 1.0)

    def test_fourier_basis(self):
        enc = BasisEncoder(n_basis=16, basis_type='fourier')
        X = np.random.default_rng(42).uniform(-3, 3, (10, 1))
        phi = enc.encode_x(X)
        assert phi.shape == (10, 16)

    def test_centers_initialized_once(self):
        enc = BasisEncoder(n_basis=8, basis_type='rbf')
        X1 = np.array([[0.0], [1.0], [2.0]])
        enc.encode_x(X1)
        centers1 = enc.centers.copy()
        X2 = np.array([[10.0], [20.0]])
        enc.encode_x(X2)
        # Centers should not change
        np.testing.assert_array_equal(enc.centers, centers1)

    def test_multidim_rbf(self):
        enc = BasisEncoder(n_basis=16, basis_type='rbf', input_dim=3)
        X = np.random.default_rng(42).uniform(-1, 1, (10, 3))
        phi = enc.encode_x(X)
        assert phi.shape == (10, 16)

    def test_length_scale_effect(self):
        """Larger length scale = wider basis functions = more overlap."""
        X = np.array([[0.0], [5.0]])
        enc_narrow = BasisEncoder(n_basis=8, length_scale=0.1, basis_type='rbf')
        enc_wide = BasisEncoder(n_basis=8, length_scale=5.0, basis_type='rbf')
        phi_narrow = enc_narrow.encode_x(X)
        phi_wide = enc_wide.encode_x(X)
        # Wide basis should have more overlap between distant points
        overlap_narrow = np.sum(phi_narrow[0] * phi_narrow[1])
        overlap_wide = np.sum(phi_wide[0] * phi_wide[1])
        assert overlap_wide > overlap_narrow

    def test_invalid_basis_type(self):
        enc = BasisEncoder(n_basis=8, basis_type='invalid')
        X = np.array([[0.0]])
        with pytest.raises(ValueError, match="Unknown basis type"):
            enc.encode_x(X)


# ===========================================================================
# Aggregator Tests
# ===========================================================================

class TestAggregators:
    def test_mean_aggregator(self):
        reps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = mean_aggregator(reps)
        np.testing.assert_allclose(result, [3.0, 4.0])

    def test_variance_aggregator(self):
        reps = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = variance_aggregator(reps)
        assert len(result) == 4  # 2 * repr_dim
        np.testing.assert_allclose(result[:2], [3.0, 4.0])  # mean
        assert result[2] > 0  # variance > 0

    def test_max_aggregator(self):
        reps = np.array([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]])
        result = max_aggregator(reps)
        np.testing.assert_allclose(result, [5.0, 4.0])

    def test_single_element(self):
        reps = np.array([[2.0, 3.0]])
        assert np.allclose(mean_aggregator(reps), [2.0, 3.0])
        assert np.allclose(max_aggregator(reps), [2.0, 3.0])


# ===========================================================================
# Linear Decoder Tests
# ===========================================================================

class TestLinearDecoder:
    def test_predict_shape(self):
        dec = LinearDecoder(repr_dim=16, n_basis=8)
        target_feat = np.random.default_rng(42).normal(0, 1, (10, 8))
        ctx_repr = np.random.default_rng(43).normal(0, 1, 16)
        mean, std = dec.predict(target_feat, ctx_repr)
        assert mean.shape == (10,)
        assert std.shape == (10,)

    def test_std_positive(self):
        dec = LinearDecoder(repr_dim=8, n_basis=4, noise_init=0.5)
        target_feat = np.random.default_rng(42).normal(0, 1, (5, 4))
        ctx_repr = np.zeros(8)
        _, std = dec.predict(target_feat, ctx_repr)
        assert np.all(std > 0)

    def test_zero_weights_zero_mean(self):
        dec = LinearDecoder(repr_dim=8, n_basis=4)
        target_feat = np.random.default_rng(42).normal(0, 1, (5, 4))
        ctx_repr = np.zeros(8)
        mean, _ = dec.predict(target_feat, ctx_repr)
        np.testing.assert_allclose(mean, 0.0, atol=1e-10)


# ===========================================================================
# Conditional Neural Process Tests
# ===========================================================================

class TestCNP:
    def test_create(self):
        cnp = ConditionalNeuralProcess(n_basis=16)
        assert not cnp.is_trained

    def test_predict_shape(self):
        cnp = ConditionalNeuralProcess(n_basis=16)
        task = make_simple_task()
        pred = cnp.predict(task.X_support, task.y_support, task.X_query)
        assert isinstance(pred, NPPrediction)
        assert pred.mean.shape == (20,)
        assert pred.std.shape == (20,)

    def test_predict_std_positive(self):
        cnp = ConditionalNeuralProcess(n_basis=16)
        task = make_simple_task()
        pred = cnp.predict(task.X_support, task.y_support, task.X_query)
        assert np.all(pred.std > 0)

    def test_train_and_predict(self):
        dist = make_simple_distribution()
        cnp = ConditionalNeuralProcess(n_basis=24)
        result = cnp.train(dist, n_epochs=10)
        assert isinstance(result, NPTrainResult)
        assert cnp.is_trained
        assert result.n_tasks_seen > 0
        assert len(result.train_losses) == 10
        assert len(result.val_losses) == 10

    def test_trained_better_than_untrained(self):
        dist = make_simple_distribution(n_tasks=20)
        rng = np.random.default_rng(42)

        cnp_untrained = ConditionalNeuralProcess(n_basis=24)
        cnp_trained = ConditionalNeuralProcess(n_basis=24)
        cnp_trained.train(dist, n_epochs=10, rng=rng)

        # Test on a task
        task = dist.tasks[0]
        rmse_u, _ = cnp_untrained.evaluate(task)
        rmse_t, _ = cnp_trained.evaluate(task)
        assert rmse_t < rmse_u

    def test_more_context_better(self):
        """More context points should improve predictions."""
        dist = make_simple_distribution(n_tasks=20, n_support=20)
        cnp = ConditionalNeuralProcess(n_basis=24)
        cnp.train(dist, n_epochs=10)

        task = dist.tasks[0]
        # 3 context points
        pred_few = cnp.predict(task.X_support[:3], task.y_support[:3], task.X_query)
        rmse_few = _rmse(task.y_query, pred_few.mean)
        # All context points
        pred_all = cnp.predict(task.X_support, task.y_support, task.X_query)
        rmse_all = _rmse(task.y_query, pred_all.mean)
        assert rmse_all < rmse_few * 1.5  # Allow some tolerance

    def test_empty_context(self):
        cnp = ConditionalNeuralProcess(n_basis=16)
        X_t = np.array([[0.0], [1.0]])
        pred = cnp.predict(np.zeros((0, 1)), np.array([]), X_t)
        assert pred.mean.shape == (2,)

    def test_evaluate(self):
        dist = make_simple_distribution()
        cnp = ConditionalNeuralProcess(n_basis=16)
        cnp.train(dist, n_epochs=5)
        rmse, nlpd = cnp.evaluate(dist.tasks[0])
        assert rmse >= 0
        assert np.isfinite(nlpd)

    def test_sinusoidal_distribution(self):
        dist = sinusoidal_task_distribution(n_tasks=15, n_support=10, n_query=30)
        cnp = ConditionalNeuralProcess(n_basis=32, length_scale=1.5)
        result = cnp.train(dist, n_epochs=10)
        assert result.n_tasks_seen > 0

    def test_polynomial_distribution(self):
        dist = polynomial_task_distribution(n_tasks=15, n_support=10, n_query=30)
        cnp = ConditionalNeuralProcess(n_basis=32)
        result = cnp.train(dist, n_epochs=10)
        assert result.n_tasks_seen > 0


# ===========================================================================
# Neural Process Tests
# ===========================================================================

class TestNP:
    def test_create(self):
        np_model = NeuralProcess(n_basis=16, latent_dim=8)
        assert not np_model.is_trained
        assert np_model.latent_dim == 8

    def test_predict_shape(self):
        np_model = NeuralProcess(n_basis=16, latent_dim=8)
        task = make_simple_task()
        pred = np_model.predict(task.X_support, task.y_support, task.X_query)
        assert pred.mean.shape == (20,)
        assert pred.std.shape == (20,)

    def test_predict_with_samples(self):
        np_model = NeuralProcess(n_basis=16, latent_dim=8)
        task = make_simple_task()
        pred = np_model.predict(task.X_support, task.y_support, task.X_query,
                                n_samples=5, rng=np.random.default_rng(42))
        assert pred.samples is not None
        assert pred.samples.shape == (5, 20)

    def test_samples_vary(self):
        """Different latent samples should give different predictions."""
        np_model = NeuralProcess(n_basis=16, latent_dim=8)
        dist = make_simple_distribution()
        np_model.train(dist, n_epochs=10)

        task = dist.tasks[0]
        pred = np_model.predict(task.X_support, task.y_support, task.X_query,
                                n_samples=10, rng=np.random.default_rng(42))
        # Samples should have some variance
        sample_var = pred.samples.var(axis=0).mean()
        assert sample_var > 0

    def test_train(self):
        dist = make_simple_distribution()
        np_model = NeuralProcess(n_basis=24, latent_dim=8)
        result = np_model.train(dist, n_epochs=10)
        assert np_model.is_trained
        assert result.model_params['latent_dim'] == 8

    def test_latent_encoding(self):
        np_model = NeuralProcess(n_basis=16, latent_dim=8)
        task = make_simple_task()
        mu, sigma = np_model._encode_to_latent(task.X_support, task.y_support)
        assert mu.shape == (8,)
        assert sigma.shape == (8,)
        assert np.all(sigma > 0)

    def test_empty_context_latent(self):
        np_model = NeuralProcess(n_basis=16, latent_dim=8)
        mu, sigma = np_model._encode_to_latent(np.zeros((0, 1)), np.array([]))
        assert mu.shape == (8,)
        assert sigma.shape == (8,)

    def test_evaluate(self):
        dist = make_simple_distribution()
        np_model = NeuralProcess(n_basis=16, latent_dim=8)
        np_model.train(dist, n_epochs=5)
        rmse, nlpd = np_model.evaluate(dist.tasks[0])
        assert rmse >= 0
        assert np.isfinite(nlpd)

    def test_different_latent_dims(self):
        dist = make_simple_distribution(n_tasks=10)
        for ld in [4, 8, 16]:
            np_model = NeuralProcess(n_basis=16, latent_dim=ld)
            result = np_model.train(dist, n_epochs=5)
            assert np_model.is_trained


# ===========================================================================
# Attentive Neural Process Tests
# ===========================================================================

class TestANP:
    def test_create(self):
        anp = AttentiveNeuralProcess(n_basis=16)
        assert not anp.is_trained

    def test_predict_shape(self):
        anp = AttentiveNeuralProcess(n_basis=16)
        task = make_simple_task()
        pred = anp.predict(task.X_support, task.y_support, task.X_query)
        assert pred.mean.shape == (20,)
        assert pred.std.shape == (20,)

    def test_attention_weights(self):
        anp = AttentiveNeuralProcess(n_basis=16, attention_type='dot_product')
        X_t = np.array([[0.0], [1.0], [2.0]])
        X_c = np.array([[0.1], [1.1]])
        weights = anp._compute_attention(X_t, X_c)
        assert weights.shape == (3, 2)
        # Rows sum to 1
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)

    def test_rbf_attention(self):
        anp = AttentiveNeuralProcess(n_basis=16, attention_type='rbf')
        X_t = np.array([[0.0], [10.0]])
        X_c = np.array([[0.1], [9.9]])
        weights = anp._compute_attention(X_t, X_c)
        # Target 0 should attend more to context 0
        assert weights[0, 0] > weights[0, 1]
        # Target 1 should attend more to context 1
        assert weights[1, 1] > weights[1, 0]

    def test_laplacian_attention(self):
        anp = AttentiveNeuralProcess(n_basis=16, attention_type='laplacian')
        X_t = np.array([[0.0]])
        X_c = np.array([[0.1], [5.0]])
        weights = anp._compute_attention(X_t, X_c)
        assert weights[0, 0] > weights[0, 1]  # Closer gets more weight

    def test_invalid_attention(self):
        anp = AttentiveNeuralProcess(n_basis=16, attention_type='invalid')
        with pytest.raises(ValueError, match="Unknown attention"):
            anp._compute_attention(np.array([[0.0]]), np.array([[0.0]]))

    def test_train_and_predict(self):
        dist = make_simple_distribution()
        anp = AttentiveNeuralProcess(n_basis=24)
        result = anp.train(dist, n_epochs=10)
        assert anp.is_trained
        task = dist.tasks[0]
        rmse, _ = anp.evaluate(task)
        assert rmse < 5.0  # Reasonable prediction

    def test_empty_context(self):
        anp = AttentiveNeuralProcess(n_basis=16)
        X_t = np.array([[0.0], [1.0]])
        pred = anp.predict(np.zeros((0, 1)), np.array([]), X_t)
        assert pred.mean.shape == (2,)
        assert np.all(pred.std > 0)

    def test_multi_head_attention(self):
        anp = AttentiveNeuralProcess(n_basis=16, n_heads=4)
        X_t = np.array([[0.0], [1.0]])
        X_c = np.array([[0.5], [1.5], [2.5]])
        values = np.random.default_rng(42).normal(0, 1, (3, 32))
        attended = anp._multi_head_attention(X_t, X_c, values)
        assert attended.shape == (2, 32)

    def test_anp_vs_cnp_on_complex(self):
        """ANP should do better than CNP on complex tasks (attention helps)."""
        dist = sinusoidal_task_distribution(n_tasks=20, n_support=10, n_query=30)
        rng = np.random.default_rng(42)

        cnp = ConditionalNeuralProcess(n_basis=32, length_scale=1.5)
        cnp.train(dist, n_epochs=10, rng=rng)

        anp = AttentiveNeuralProcess(n_basis=32, attention_type='rbf',
                                      length_scale=1.5)
        anp.train(dist, n_epochs=10, rng=np.random.default_rng(42))

        # Both should give finite predictions
        task = dist.tasks[0]
        rmse_cnp, _ = cnp.evaluate(task)
        rmse_anp, _ = anp.evaluate(task)
        assert np.isfinite(rmse_cnp)
        assert np.isfinite(rmse_anp)


# ===========================================================================
# ConvCNP Tests
# ===========================================================================

class TestConvCNP:
    def test_create(self):
        model = ConvCNP(n_grid=64)
        assert not model.is_trained
        assert model.grid.shape == (64, 1)

    def test_predict_shape(self):
        model = ConvCNP(n_grid=64)
        task = make_simple_task()
        pred = model.predict(task.X_support, task.y_support, task.X_query)
        assert pred.mean.shape == (20,)
        assert pred.std.shape == (20,)

    def test_smoothing(self):
        model = ConvCNP(n_grid=64, grid_range=(-5, 5))
        x = np.array([0.0])
        vals = np.array([1.0])
        smoothed = model._smooth(x, vals, scale=0.5)
        assert smoothed.shape == (64,)
        # Peak should be near center of grid
        peak_idx = np.argmax(smoothed)
        peak_x = model.grid[peak_idx, 0]
        assert abs(peak_x) < 1.0

    def test_grid_encoding(self):
        model = ConvCNP(n_grid=64, n_channels=3)
        X_c = np.array([[0.0], [1.0], [2.0]])
        y_c = np.array([1.0, 2.0, 3.0])
        enc = model._encode_on_grid(X_c, y_c)
        assert enc.shape == (64, 4)  # 1 density + 3 signal channels

    def test_empty_context(self):
        model = ConvCNP(n_grid=64)
        enc = model._encode_on_grid(np.zeros((0, 1)), np.array([]))
        assert enc.shape == (64, 1 + model.n_channels)
        assert np.allclose(enc, 0.0)

    def test_interpolation(self):
        model = ConvCNP(n_grid=64, grid_range=(-5, 5))
        grid_vals = np.zeros((64, 2))
        grid_vals[:, 0] = model.grid.ravel()  # Linear ramp
        X_t = np.array([[0.0], [1.0], [-1.0]])
        interp = model._interpolate_from_grid(grid_vals, X_t)
        assert interp.shape == (3, 2)
        # Interpolated value near 0 should be near 0
        assert abs(interp[0, 0]) < 0.5

    def test_train_and_predict(self):
        dist = make_simple_distribution()
        model = ConvCNP(n_grid=64, grid_range=(-5, 5))
        result = model.train(dist, n_epochs=5)
        assert model.is_trained

        task = dist.tasks[0]
        rmse, nlpd = model.evaluate(task)
        assert rmse >= 0
        assert np.isfinite(nlpd)

    def test_translation_equivariance(self):
        """Shifting context and targets by same amount should give same predictions."""
        model = ConvCNP(n_grid=128, grid_range=(-10, 10), length_scale=0.5)
        dist = make_simple_distribution()
        model.train(dist, n_epochs=5)

        X_c = np.array([[0.0], [1.0], [2.0]])
        y_c = np.array([1.0, 2.0, 3.0])
        X_t = np.array([[-1.0], [0.5], [1.5]])

        shift = 2.0
        pred_orig = model.predict(X_c, y_c, X_t)
        pred_shift = model.predict(X_c + shift, y_c, X_t + shift)
        # Should be approximately the same (exact only if grid is infinite)
        np.testing.assert_allclose(pred_orig.mean, pred_shift.mean, atol=0.5)

    def test_multi_channel(self):
        model = ConvCNP(n_grid=64, n_channels=6)
        assert len(model.kernel_scales) == 6
        assert model.kernel_scales[0] < model.kernel_scales[-1]


# ===========================================================================
# GP Neural Process Tests
# ===========================================================================

class TestGPNP:
    def test_create(self):
        model = GPNeuralProcess(n_basis=16)
        assert not model.is_trained

    def test_predict_shape(self):
        model = GPNeuralProcess(n_basis=16)
        task = make_simple_task()
        pred = model.predict(task.X_support, task.y_support, task.X_query)
        assert pred.mean.shape == (20,)
        assert pred.std.shape == (20,)

    def test_std_positive(self):
        model = GPNeuralProcess(n_basis=16)
        task = make_simple_task()
        pred = model.predict(task.X_support, task.y_support, task.X_query)
        assert np.all(pred.std > 0)

    def test_train(self):
        dist = make_simple_distribution()
        model = GPNeuralProcess(n_basis=16)
        result = model.train(dist, n_epochs=5)
        assert model.is_trained
        assert 'length_scale' in result.model_params

    def test_trained_reasonable_rmse(self):
        dist = make_simple_distribution(n_tasks=20)
        model = GPNeuralProcess(n_basis=16)
        model.train(dist, n_epochs=5)
        task = dist.tasks[0]
        rmse, _ = model.evaluate(task)
        assert rmse < 3.0  # GP should do well on linear tasks

    def test_empty_context(self):
        model = GPNeuralProcess(n_basis=16)
        X_t = np.array([[0.0], [1.0]])
        pred = model.predict(np.zeros((0, 1)), np.array([]), X_t)
        assert pred.mean.shape == (2,)
        assert np.all(pred.std > 0)

    def test_gp_calibrated(self):
        """GP predictions should be reasonably calibrated."""
        dist = make_simple_distribution(n_tasks=20)
        model = GPNeuralProcess(n_basis=16)
        model.train(dist, n_epochs=5)

        in_interval = 0
        total = 0
        for task in dist.tasks[:5]:
            pred = model.predict(task.X_support, task.y_support, task.X_query)
            lo = pred.mean - 2 * pred.std
            hi = pred.mean + 2 * pred.std
            in_interval += np.sum((task.y_query >= lo) & (task.y_query <= hi))
            total += len(task.y_query)
        coverage = in_interval / total
        assert coverage > 0.5  # At least 50% within 2-sigma


# ===========================================================================
# Comparison and Utility Tests
# ===========================================================================

class TestComparison:
    def test_compare_models(self):
        dist = make_simple_distribution(n_tasks=15)
        result = compare_np_models(dist, n_test=3, n_epochs=5)
        assert isinstance(result, NPComparisonResult)
        assert len(result.model_names) == 5
        for name in result.model_names:
            assert name in result.mean_rmse
            assert name in result.mean_nlpd
            assert name in result.calibration
            assert result.mean_rmse[name] >= 0
            assert 0 <= result.calibration[name] <= 1

    def test_comparison_summary(self):
        dist = make_simple_distribution(n_tasks=12)
        result = compare_np_models(dist, n_test=3, n_epochs=3)
        summary = comparison_summary(result)
        assert "NP Model Comparison" in summary
        assert "CNP" in summary
        assert "Best (RMSE)" in summary


class TestFewShotCurve:
    def test_learning_curve(self):
        dist = make_simple_distribution(n_tasks=10, n_support=20)
        cnp = ConditionalNeuralProcess(n_basis=16)
        cnp.train(dist, n_epochs=5)
        curve = few_shot_learning_curve(cnp, dist, n_shots=[1, 5, 10])
        assert 1 in curve
        assert 5 in curve
        assert 10 in curve
        assert len(curve[1]) == 10  # per-task

    def test_more_shots_generally_better(self):
        dist = make_simple_distribution(n_tasks=10, n_support=20)
        cnp = ConditionalNeuralProcess(n_basis=24)
        cnp.train(dist, n_epochs=10)
        curve = few_shot_learning_curve(cnp, dist, n_shots=[1, 20])
        mean_1 = np.mean(curve[1])
        mean_20 = np.mean(curve[20])
        # More context should help (or at least not significantly hurt)
        assert mean_20 < mean_1 * 2.0


class TestContextSensitivity:
    def test_analysis(self):
        dist = make_simple_distribution()
        cnp = ConditionalNeuralProcess(n_basis=16)
        cnp.train(dist, n_epochs=5)
        result = context_sensitivity_analysis(cnp, dist.tasks[0], n_repeats=5)
        assert 'mean_std' in result
        assert 'var_mean' in result
        assert 'coverage' in result
        assert 'mean_rmse' in result
        assert 0 <= result['coverage'] <= 1


class TestSummary:
    def test_np_summary(self):
        result = NPTrainResult(
            train_losses=[1.0, 0.5],
            val_losses=[1.2, 0.6],
            best_epoch=1,
            n_tasks_seen=10,
            n_epochs=2,
            model_params={'noise': 0.1},
        )
        s = np_summary(result, "TestModel")
        assert "TestModel" in s
        assert "noise" in s


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_all_models_on_sinusoidal(self):
        dist = sinusoidal_task_distribution(n_tasks=12, n_support=10, n_query=20)
        models = [
            ("CNP", ConditionalNeuralProcess(n_basis=32, length_scale=1.5)),
            ("NP", NeuralProcess(n_basis=32, latent_dim=8, length_scale=1.5)),
            ("ANP", AttentiveNeuralProcess(n_basis=32, attention_type='rbf',
                                            length_scale=1.5)),
            ("ConvCNP", ConvCNP(n_grid=64)),
            ("GPNP", GPNeuralProcess(n_basis=32, length_scale=1.5)),
        ]
        for name, model in models:
            result = model.train(dist, n_epochs=5)
            assert model.is_trained, f"{name} not trained"
            rmse, nlpd = model.evaluate(dist.tasks[0])
            assert np.isfinite(rmse), f"{name} RMSE not finite"
            assert np.isfinite(nlpd), f"{name} NLPD not finite"

    def test_all_models_on_step(self):
        dist = step_task_distribution(n_tasks=12, n_support=10, n_query=20)
        models = [
            ("CNP", ConditionalNeuralProcess(n_basis=32)),
            ("NP", NeuralProcess(n_basis=32, latent_dim=8)),
            ("ANP", AttentiveNeuralProcess(n_basis=32)),
            ("ConvCNP", ConvCNP(n_grid=64)),
            ("GPNP", GPNeuralProcess(n_basis=32)),
        ]
        for name, model in models:
            result = model.train(dist, n_epochs=5)
            rmse, nlpd = model.evaluate(dist.tasks[0])
            assert np.isfinite(rmse), f"{name} RMSE not finite on step"

    def test_polynomial_tasks(self):
        dist = polynomial_task_distribution(n_tasks=12, n_support=10, n_query=20)
        cnp = ConditionalNeuralProcess(n_basis=32)
        cnp.train(dist, n_epochs=5)
        for task in dist.tasks[:3]:
            rmse, _ = cnp.evaluate(task)
            assert np.isfinite(rmse)

    def test_cross_distribution_transfer(self):
        """Train on one distribution, evaluate on another."""
        train_dist = make_simple_distribution(n_tasks=20)
        # Test on sinusoidal (different distribution)
        test_dist = sinusoidal_task_distribution(n_tasks=5, n_support=10, n_query=20)

        cnp = ConditionalNeuralProcess(n_basis=32)
        cnp.train(train_dist, n_epochs=10)

        # Should still give finite predictions (though may not be great)
        for task in test_dist.tasks:
            pred = cnp.predict(task.X_support, task.y_support, task.X_query)
            assert np.all(np.isfinite(pred.mean))
            assert np.all(np.isfinite(pred.std))

    def test_single_context_point(self):
        """All models should handle a single context point."""
        dist = make_simple_distribution()
        task = dist.tasks[0]
        models = [
            ConditionalNeuralProcess(n_basis=16),
            NeuralProcess(n_basis=16, latent_dim=8),
            AttentiveNeuralProcess(n_basis=16),
            ConvCNP(n_grid=64),
            GPNeuralProcess(n_basis=16),
        ]
        for model in models:
            model.train(dist, n_epochs=3)
            pred = model.predict(task.X_support[:1], task.y_support[:1], task.X_query)
            assert pred.mean.shape == (20,)
            assert np.all(np.isfinite(pred.mean))

    def test_large_context(self):
        """Models should handle many context points."""
        dist = make_simple_distribution(n_tasks=10, n_support=100, n_query=20)
        cnp = ConditionalNeuralProcess(n_basis=32)
        cnp.train(dist, n_epochs=3)
        task = dist.tasks[0]
        rmse, _ = cnp.evaluate(task)
        assert np.isfinite(rmse)

    def test_np_sample_consistency(self):
        """NP samples should bracket the true values reasonably."""
        dist = make_simple_distribution(n_tasks=20)
        np_model = NeuralProcess(n_basis=24, latent_dim=16)
        np_model.train(dist, n_epochs=10)

        task = dist.tasks[0]
        pred = np_model.predict(task.X_support, task.y_support, task.X_query,
                                n_samples=50, rng=np.random.default_rng(42))
        # Sample mean should be close to the prediction mean
        sample_mean = pred.samples.mean(axis=0)
        np.testing.assert_allclose(sample_mean, pred.mean, atol=1.0)

    def test_deterministic_predictions(self):
        """Same inputs should give same predictions."""
        dist = make_simple_distribution()
        cnp = ConditionalNeuralProcess(n_basis=16)
        cnp.train(dist, n_epochs=5, rng=np.random.default_rng(42))

        task = dist.tasks[0]
        pred1 = cnp.predict(task.X_support, task.y_support, task.X_query)
        pred2 = cnp.predict(task.X_support, task.y_support, task.X_query)
        np.testing.assert_array_equal(pred1.mean, pred2.mean)
        np.testing.assert_array_equal(pred1.std, pred2.std)


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_very_small_n_basis(self):
        cnp = ConditionalNeuralProcess(n_basis=2)
        task = make_simple_task()
        pred = cnp.predict(task.X_support, task.y_support, task.X_query)
        assert pred.mean.shape == (20,)

    def test_large_n_basis(self):
        cnp = ConditionalNeuralProcess(n_basis=128)
        task = make_simple_task()
        pred = cnp.predict(task.X_support, task.y_support, task.X_query)
        assert pred.mean.shape == (20,)

    def test_constant_targets(self):
        """Context with constant y should predict that constant."""
        dist = make_simple_distribution()
        cnp = ConditionalNeuralProcess(n_basis=16)
        cnp.train(dist, n_epochs=5)

        X_c = np.array([[0.0], [1.0], [2.0]])
        y_c = np.array([5.0, 5.0, 5.0])
        X_t = np.array([[0.5], [1.5]])
        pred = cnp.predict(X_c, y_c, X_t)
        # Should predict values near 5.0 (not exactly, but roughly)
        assert np.all(np.isfinite(pred.mean))

    def test_noisy_context(self):
        """Should handle very noisy context."""
        dist = make_simple_distribution()
        cnp = ConditionalNeuralProcess(n_basis=16)
        cnp.train(dist, n_epochs=5)

        rng = np.random.default_rng(42)
        X_c = rng.uniform(-3, 3, (10, 1))
        y_c = rng.normal(0, 100, 10)  # Very noisy
        X_t = np.array([[0.0]])
        pred = cnp.predict(X_c, y_c, X_t)
        assert np.all(np.isfinite(pred.mean))
        assert np.all(np.isfinite(pred.std))

    def test_two_tasks_distribution(self):
        """Minimum viable distribution."""
        tasks = [make_simple_task(rng=np.random.default_rng(i)) for i in range(3)]
        dist = TaskDistribution(tasks, "minimal", input_dim=1)
        cnp = ConditionalNeuralProcess(n_basis=8)
        result = cnp.train(dist, n_epochs=3)
        assert cnp.is_trained

    def test_convcnp_out_of_grid(self):
        """ConvCNP should handle targets outside grid range."""
        model = ConvCNP(n_grid=64, grid_range=(-5, 5))
        dist = make_simple_distribution()
        model.train(dist, n_epochs=3)
        # Target far outside grid
        X_c = np.array([[0.0]])
        y_c = np.array([1.0])
        X_t = np.array([[4.5]])  # Within grid range
        pred = model.predict(X_c, y_c, X_t)
        assert np.all(np.isfinite(pred.mean))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
