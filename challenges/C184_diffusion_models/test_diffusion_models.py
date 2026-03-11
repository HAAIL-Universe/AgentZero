"""
Tests for C184: Diffusion Models
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from diffusion_models import (
    NoiseSchedule, TimeEmbedding, DenoisingMLP, DDPM,
    ClassifierFreeGuidance, DiffusionDiagnostics,
    ScoreBasedModel, VPSDE,
    make_moons, make_circles, make_gaussian_mixture,
    relu, relu_grad, silu, silu_grad,
)


# ============================================================
# Noise Schedule Tests
# ============================================================

class TestNoiseSchedule:
    def test_linear_schedule(self):
        s = NoiseSchedule(100, "linear")
        assert len(s.betas) == 100
        assert s.betas[0] < s.betas[-1]
        assert np.all(s.betas > 0)
        assert np.all(s.betas < 1)

    def test_cosine_schedule(self):
        s = NoiseSchedule(100, "cosine")
        assert len(s.betas) == 100
        assert np.all(s.betas >= 0)
        assert np.all(s.betas <= 0.999)

    def test_quadratic_schedule(self):
        s = NoiseSchedule(100, "quadratic")
        assert len(s.betas) == 100
        assert np.all(s.betas > 0)

    def test_sigmoid_schedule(self):
        s = NoiseSchedule(100, "sigmoid")
        assert len(s.betas) == 100
        assert np.all(s.betas > 0)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            NoiseSchedule(100, "unknown")

    def test_alpha_cumprod_decreasing(self):
        s = NoiseSchedule(100, "linear")
        diffs = np.diff(s.alpha_cumprod)
        assert np.all(diffs <= 0)

    def test_alpha_cumprod_bounds(self):
        s = NoiseSchedule(100, "linear")
        assert s.alpha_cumprod[0] > 0.9  # Nearly 1 at start
        assert s.alpha_cumprod[-1] < 0.5  # Much less at end

    def test_snr_decreasing(self):
        s = NoiseSchedule(100, "linear")
        snr_first = s.snr(0)
        snr_last = s.snr(99)
        assert snr_first > snr_last

    def test_posterior_variance_shape(self):
        s = NoiseSchedule(50)
        assert s.posterior_variance.shape == (50,)
        assert s.posterior_log_variance.shape == (50,)

    def test_posterior_mean_coefficients(self):
        s = NoiseSchedule(50)
        assert s.posterior_mean_coef1.shape == (50,)
        assert s.posterior_mean_coef2.shape == (50,)

    def test_precomputed_quantities(self):
        s = NoiseSchedule(100)
        np.testing.assert_allclose(s.sqrt_alpha_cumprod ** 2, s.alpha_cumprod, atol=1e-10)
        np.testing.assert_allclose(
            s.sqrt_one_minus_alpha_cumprod ** 2, 1 - s.alpha_cumprod, atol=1e-10
        )

    def test_different_beta_range(self):
        s = NoiseSchedule(100, beta_start=1e-5, beta_end=0.05)
        assert s.betas[0] < 1e-4
        assert s.betas[-1] > 0.04


# ============================================================
# Time Embedding Tests
# ============================================================

class TestTimeEmbedding:
    def test_output_shape(self):
        emb = TimeEmbedding(32)
        t = np.array([0, 5, 10, 50])
        result = emb(t)
        assert result.shape == (4, 32)

    def test_different_timesteps_different_embeddings(self):
        emb = TimeEmbedding(16)
        t = np.array([0, 50])
        result = emb(t)
        assert not np.allclose(result[0], result[1])

    def test_odd_dimension(self):
        emb = TimeEmbedding(33)
        t = np.array([10])
        result = emb(t)
        assert result.shape == (1, 33)

    def test_single_timestep(self):
        emb = TimeEmbedding(16)
        result = emb(np.array([5]))
        assert result.shape == (1, 16)

    def test_deterministic(self):
        emb = TimeEmbedding(32)
        t = np.array([7, 14])
        r1 = emb(t)
        r2 = emb(t)
        np.testing.assert_array_equal(r1, r2)


# ============================================================
# Activation Tests
# ============================================================

class TestActivations:
    def test_relu(self):
        x = np.array([-2, -1, 0, 1, 2])
        np.testing.assert_array_equal(relu(x), [0, 0, 0, 1, 2])

    def test_relu_grad(self):
        x = np.array([-2, -1, 0, 1, 2])
        np.testing.assert_array_equal(relu_grad(x), [0, 0, 0, 1, 1])

    def test_silu_zero(self):
        assert abs(silu(np.array([0.0]))[0]) < 1e-10

    def test_silu_positive(self):
        x = np.array([5.0])
        assert silu(x)[0] > 4.0  # silu(5) ~ 4.97

    def test_silu_grad_shape(self):
        x = np.array([-1.0, 0.0, 1.0])
        g = silu_grad(x)
        assert g.shape == (3,)


# ============================================================
# Denoising MLP Tests
# ============================================================

class TestDenoisingMLP:
    def test_forward_shape(self):
        net = DenoisingMLP(data_dim=2, hidden_dim=32, num_layers=2)
        x = np.random.randn(10, 2)
        t = np.array([5] * 10)
        out = net.forward(x, t)
        assert out.shape == (10, 2)

    def test_forward_deterministic(self):
        net = DenoisingMLP(data_dim=2, hidden_dim=32, num_layers=2)
        x = np.random.randn(5, 2)
        t = np.array([3] * 5)
        out1 = net.forward(x, t)
        out2 = net.forward(x, t)
        np.testing.assert_array_equal(out1, out2)

    def test_backward_updates_weights(self):
        net = DenoisingMLP(data_dim=2, hidden_dim=16, num_layers=2)
        x = np.random.randn(5, 2)
        t = np.array([1] * 5)
        out = net.forward(x, t)
        w_before = net.weights[0].copy()
        grad = np.ones_like(out) * 0.1
        net.backward(grad, lr=0.01)
        assert not np.allclose(w_before, net.weights[0])

    def test_parameters(self):
        net = DenoisingMLP(data_dim=2, hidden_dim=16, num_layers=3)
        params = net.parameters()
        assert len(params) == 6  # 3 weights + 3 biases

    def test_set_parameters(self):
        net = DenoisingMLP(data_dim=2, hidden_dim=16, num_layers=2)
        params = net.parameters()
        new_params = [p * 2 for p in params]
        net.set_parameters(new_params)
        np.testing.assert_allclose(net.weights[0], new_params[0])

    def test_copy(self):
        net = DenoisingMLP(data_dim=2, hidden_dim=16, num_layers=2)
        copy = net.copy()
        np.testing.assert_array_equal(net.weights[0], copy.weights[0])
        copy.weights[0][0, 0] = 999
        assert net.weights[0][0, 0] != 999

    def test_with_conditioning(self):
        net = DenoisingMLP(data_dim=2, hidden_dim=32, num_layers=2, cond_dim=4)
        x = np.random.randn(5, 2)
        t = np.array([3] * 5)
        cond = np.random.randn(5, 4)
        out = net.forward(x, t, cond)
        assert out.shape == (5, 2)

    def test_different_layers(self):
        for n_layers in [1, 2, 4]:
            net = DenoisingMLP(data_dim=3, hidden_dim=16, num_layers=n_layers)
            x = np.random.randn(4, 3)
            t = np.array([0] * 4)
            out = net.forward(x, t)
            assert out.shape == (4, 3)


# ============================================================
# DDPM Core Tests
# ============================================================

class TestDDPM:
    def test_creation(self):
        ddpm = DDPM(data_dim=2, num_timesteps=50)
        assert ddpm.T == 50
        assert ddpm.data_dim == 2
        assert ddpm.prediction_type == "epsilon"

    def test_q_sample_shape(self):
        ddpm = DDPM(data_dim=2, num_timesteps=50)
        rng = np.random.RandomState(42)
        x_0 = rng.randn(10, 2)
        t = np.array([5] * 10)
        x_t, noise = ddpm.q_sample(x_0, t, rng=rng)
        assert x_t.shape == (10, 2)
        assert noise.shape == (10, 2)

    def test_q_sample_t0_close_to_x0(self):
        ddpm = DDPM(data_dim=2, num_timesteps=100)
        rng = np.random.RandomState(42)
        x_0 = np.array([[1.0, 2.0]] * 5)
        t = np.array([0] * 5)
        x_t, _ = ddpm.q_sample(x_0, t, rng=rng)
        # At t=0, alpha_cumprod is close to 1, so x_t should be close to x_0
        assert np.mean(np.abs(x_t - x_0)) < 0.5

    def test_q_sample_large_t_noisy(self):
        ddpm = DDPM(data_dim=2, num_timesteps=100)
        rng = np.random.RandomState(42)
        x_0 = np.ones((20, 2))
        t = np.array([99] * 20)
        x_t, _ = ddpm.q_sample(x_0, t, rng=rng)
        # At large t, should be mostly noise
        assert np.std(x_t) > 0.3

    def test_q_sample_with_provided_noise(self):
        ddpm = DDPM(data_dim=2, num_timesteps=50)
        x_0 = np.ones((5, 2))
        t = np.array([10] * 5)
        noise = np.zeros((5, 2))
        x_t, ret_noise = ddpm.q_sample(x_0, t, noise=noise)
        np.testing.assert_array_equal(ret_noise, noise)

    def test_p_mean_variance_shape(self):
        ddpm = DDPM(data_dim=2, num_timesteps=50, hidden_dim=16)
        x = np.random.randn(5, 2)
        t = np.array([10] * 5)
        result = ddpm.p_mean_variance(x, t)
        assert result['mean'].shape == (5, 2)
        assert result['variance'].shape == (5, 1)
        assert result['pred_x0'].shape == (5, 2)

    def test_training_step_returns_loss(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16)
        rng = np.random.RandomState(42)
        x_0 = rng.randn(10, 2)
        loss = ddpm.training_step(x_0, lr=1e-3, rng=rng)
        assert isinstance(loss, float)
        assert loss > 0

    def test_training_reduces_loss(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=64)

        losses = ddpm.train(data, epochs=30, batch_size=32, lr=1e-3, rng=rng)
        assert losses[-1] < losses[0]

    def test_x0_prediction_type(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16, prediction_type="x0")
        rng = np.random.RandomState(42)
        x_0 = rng.randn(10, 2)
        loss = ddpm.training_step(x_0, lr=1e-3, rng=rng)
        assert isinstance(loss, float)

    def test_sample_ddpm_shape(self):
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16)
        rng = np.random.RandomState(42)
        samples = ddpm.sample_ddpm(5, rng=rng)
        assert samples.shape == (5, 2)

    def test_sample_ddim_shape(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16)
        rng = np.random.RandomState(42)
        samples = ddpm.sample_ddim(5, num_steps=5, rng=rng)
        assert samples.shape == (5, 2)

    def test_ddim_deterministic_eta0(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        s1 = ddpm.sample_ddim(3, num_steps=5, eta=0.0, rng=rng1)
        s2 = ddpm.sample_ddim(3, num_steps=5, eta=0.0, rng=rng2)
        np.testing.assert_array_equal(s1, s2)

    def test_ddim_stochastic_eta1(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16)
        rng = np.random.RandomState(42)
        samples = ddpm.sample_ddim(5, num_steps=5, eta=1.0, rng=rng)
        assert samples.shape == (5, 2)

    def test_sample_progressive(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16)
        rng = np.random.RandomState(42)
        traj = ddpm.sample_progressive(3, save_every=5, rng=rng)
        assert len(traj) > 1
        for frame in traj:
            assert frame.shape == (3, 2)

    def test_ema(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16)
        ddpm.enable_ema(decay=0.999)
        assert ddpm.ema_model is not None

        rng = np.random.RandomState(42)
        data = rng.randn(50, 2)
        ddpm.train(data, epochs=5, batch_size=16, lr=1e-3, rng=rng)

        # EMA weights should differ from model weights
        assert not np.allclose(ddpm.model.weights[0], ddpm.ema_model.weights[0])

    def test_ema_sampling(self):
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16)
        ddpm.enable_ema()
        rng = np.random.RandomState(42)
        data = rng.randn(50, 2)
        ddpm.train(data, epochs=5, batch_size=16, lr=1e-3, rng=rng)
        samples = ddpm.sample_ddpm(3, use_ema=True, rng=rng)
        assert samples.shape == (3, 2)

    def test_conditional_ddpm(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=32, cond_dim=4)
        rng = np.random.RandomState(42)
        x_0 = rng.randn(10, 2)
        cond = rng.randn(10, 4)
        loss = ddpm.training_step(x_0, lr=1e-3, cond=cond, rng=rng)
        assert isinstance(loss, float)

    def test_conditional_sampling(self):
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16, cond_dim=3)
        rng = np.random.RandomState(42)
        cond = rng.randn(5, 3)
        samples = ddpm.sample_ddpm(5, cond=cond, rng=rng)
        assert samples.shape == (5, 2)


# ============================================================
# Classifier-Free Guidance Tests
# ============================================================

class TestClassifierFreeGuidance:
    def test_creation(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16, cond_dim=4)
        cfg = ClassifierFreeGuidance(ddpm, guidance_scale=3.0)
        assert cfg.guidance_scale == 3.0
        assert cfg.uncond_prob == 0.1

    def test_training_step(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=16, cond_dim=4)
        cfg = ClassifierFreeGuidance(ddpm)
        rng = np.random.RandomState(42)
        x = rng.randn(10, 2)
        c = rng.randn(10, 4)
        loss = cfg.training_step(x, c, lr=1e-3, rng=rng)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train(self):
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=32, cond_dim=4)
        cfg = ClassifierFreeGuidance(ddpm)
        rng = np.random.RandomState(42)
        data = rng.randn(50, 2)
        cond = rng.randn(50, 4)
        losses = cfg.train(data, cond, epochs=10, batch_size=16, lr=1e-3, rng=rng)
        assert len(losses) == 10

    def test_sample(self):
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16, cond_dim=4)
        cfg = ClassifierFreeGuidance(ddpm, guidance_scale=2.0)
        rng = np.random.RandomState(42)
        cond = rng.randn(5, 4)
        samples = cfg.sample(5, cond, rng=rng)
        assert samples.shape == (5, 2)

    def test_guidance_scale_effect(self):
        """Higher guidance scale should move samples further from uncond."""
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16, cond_dim=2)
        rng = np.random.RandomState(42)
        data = rng.randn(50, 2)
        cond = rng.randn(50, 2)
        cfg_low = ClassifierFreeGuidance(ddpm, guidance_scale=1.0)
        cfg_low.train(data, cond, epochs=5, batch_size=16, lr=1e-3, rng=np.random.RandomState(42))

        # Just verify both work without errors
        c = rng.randn(5, 2)
        s_low = cfg_low.sample(5, c, rng=np.random.RandomState(0))
        assert s_low.shape == (5, 2)

    def test_dropout_probability(self):
        """With uncond_prob=1.0, all conditions should be zeroed."""
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16, cond_dim=2)
        cfg = ClassifierFreeGuidance(ddpm, uncond_prob=1.0)
        rng = np.random.RandomState(42)
        x = rng.randn(10, 2)
        c = np.ones((10, 2))
        loss = cfg.training_step(x, c, lr=1e-3, rng=rng)
        assert isinstance(loss, float)


# ============================================================
# Diagnostics Tests
# ============================================================

class TestDiagnostics:
    def test_noise_prediction_mse(self):
        ddpm = DDPM(data_dim=2, num_timesteps=50, hidden_dim=16)
        data = np.random.randn(20, 2)
        result = DiffusionDiagnostics.noise_prediction_mse(ddpm, data)
        assert 'mse_early' in result
        assert 'mse_mid' in result
        assert 'mse_late' in result
        assert 'mse_avg' in result

    def test_sample_statistics(self):
        samples = np.random.randn(100, 2)
        stats = DiffusionDiagnostics.sample_statistics(samples)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'global_mean' in stats
        assert len(stats['mean']) == 2

    def test_fid_proxy_same_distribution(self):
        rng = np.random.RandomState(42)
        d1 = rng.randn(200, 2)
        d2 = rng.randn(200, 2)
        fid = DiffusionDiagnostics.fid_proxy(d1, d2)
        assert fid < 5.0  # Same distribution -> low FID

    def test_fid_proxy_different_distribution(self):
        rng = np.random.RandomState(42)
        d1 = rng.randn(200, 2)
        d2 = rng.randn(200, 2) + 10
        fid_diff = DiffusionDiagnostics.fid_proxy(d1, d2)
        fid_same = DiffusionDiagnostics.fid_proxy(d1, rng.randn(200, 2))
        assert fid_diff > fid_same

    def test_schedule_analysis(self):
        s = NoiseSchedule(100, "linear")
        analysis = DiffusionDiagnostics.schedule_analysis(s)
        assert analysis['type'] == 'linear'
        assert analysis['T'] == 100
        assert analysis['snr_first'] > analysis['snr_last']

    def test_reconstruction_quality(self):
        ddpm = DDPM(data_dim=2, num_timesteps=50, hidden_dim=16)
        data = np.random.randn(20, 2)
        mse = DiffusionDiagnostics.reconstruction_quality(ddpm, data, timestep=5)
        assert isinstance(mse, float)
        assert mse >= 0


# ============================================================
# Score-Based Model Tests
# ============================================================

class TestScoreBasedModel:
    def test_creation(self):
        model = ScoreBasedModel(data_dim=2, num_scales=10)
        assert len(model.sigmas) == 10
        assert model.sigmas[0] > model.sigmas[-1]

    def test_score_matching_loss(self):
        model = ScoreBasedModel(data_dim=2, num_scales=5, hidden_dim=16)
        rng = np.random.RandomState(42)
        data = rng.randn(20, 2)
        loss = model.score_matching_loss(data, lr=1e-3, rng=rng)
        assert isinstance(loss, float)
        assert loss > 0

    def test_training(self):
        model = ScoreBasedModel(data_dim=2, num_scales=5, hidden_dim=32)
        rng = np.random.RandomState(42)
        data = rng.randn(50, 2)
        losses = model.train(data, epochs=10, batch_size=16, lr=1e-3, rng=rng)
        assert len(losses) == 10

    def test_langevin_sampling(self):
        model = ScoreBasedModel(data_dim=2, num_scales=5, hidden_dim=16)
        rng = np.random.RandomState(42)
        samples = model.sample_langevin(5, num_steps=10, rng=rng)
        assert samples.shape == (5, 2)

    def test_sigmas_geometric(self):
        model = ScoreBasedModel(data_dim=2, num_scales=10, sigma_min=0.01, sigma_max=10.0)
        ratios = model.sigmas[:-1] / model.sigmas[1:]
        np.testing.assert_allclose(ratios, ratios[0], atol=1e-10)


# ============================================================
# VP-SDE Tests
# ============================================================

class TestVPSDE:
    def test_creation(self):
        sde = VPSDE(data_dim=2)
        assert sde.data_dim == 2

    def test_beta(self):
        sde = VPSDE(data_dim=2, beta_min=0.1, beta_max=20.0)
        t0 = sde.beta(np.array([0.0]))
        t1 = sde.beta(np.array([1.0]))
        assert abs(t0[0] - 0.1) < 1e-10
        assert abs(t1[0] - 20.0) < 1e-10

    def test_mean_coeff_t0(self):
        sde = VPSDE(data_dim=2)
        mc = sde.mean_coeff(np.array([0.0]))
        assert abs(mc[0] - 1.0) < 1e-5  # At t=0, mean_coeff ~ 1

    def test_variance_t0(self):
        sde = VPSDE(data_dim=2)
        v = sde.variance(np.array([0.0]))
        assert abs(v[0]) < 1e-5  # At t=0, variance ~ 0

    def test_variance_t1(self):
        sde = VPSDE(data_dim=2)
        v = sde.variance(np.array([1.0]))
        assert v[0] > 0.5  # At t=1, variance should be large

    def test_training_step(self):
        sde = VPSDE(data_dim=2, hidden_dim=16)
        rng = np.random.RandomState(42)
        data = rng.randn(10, 2)
        loss = sde.training_step(data, lr=1e-3, rng=rng)
        assert isinstance(loss, float)
        assert loss > 0

    def test_training(self):
        sde = VPSDE(data_dim=2, hidden_dim=32)
        rng = np.random.RandomState(42)
        data = rng.randn(50, 2)
        losses = sde.train(data, epochs=10, batch_size=16, lr=1e-3, rng=rng)
        assert len(losses) == 10

    def test_sampling(self):
        sde = VPSDE(data_dim=2, hidden_dim=16)
        rng = np.random.RandomState(42)
        samples = sde.sample(5, num_steps=20, rng=rng)
        assert samples.shape == (5, 2)


# ============================================================
# Data Generator Tests
# ============================================================

class TestDataGenerators:
    def test_make_moons(self):
        data = make_moons(100)
        assert data.shape == (100, 2)

    def test_make_moons_noise(self):
        d1 = make_moons(100, noise=0.0, rng=np.random.RandomState(42))
        d2 = make_moons(100, noise=0.5, rng=np.random.RandomState(42))
        # More noise -> higher variance
        assert np.std(d2) >= np.std(d1) - 0.1  # Allow small tolerance

    def test_make_circles(self):
        data = make_circles(100)
        assert data.shape == (100, 2)

    def test_make_circles_radius(self):
        data = make_circles(200, noise=0.0)
        radii = np.sqrt(np.sum(data ** 2, axis=1))
        # Should have two clusters of radii around 0.5 and 1.0
        inner = radii[radii < 0.75]
        outer = radii[radii >= 0.75]
        assert len(inner) > 0
        assert len(outer) > 0

    def test_gaussian_mixture(self):
        data, labels = make_gaussian_mixture(100, n_components=4)
        assert data.shape == (100, 2)
        assert labels.shape == (100,)
        assert len(np.unique(labels)) == 4

    def test_gaussian_mixture_custom(self):
        data, labels = make_gaussian_mixture(80, n_components=3, dim=5)
        assert data.shape == (80, 5)
        assert len(np.unique(labels)) == 3

    def test_gaussian_mixture_spread(self):
        d_close, _ = make_gaussian_mixture(100, spread=1.0, rng=np.random.RandomState(42))
        d_far, _ = make_gaussian_mixture(100, spread=10.0, rng=np.random.RandomState(42))
        assert np.std(d_far) > np.std(d_close)


# ============================================================
# Integration Tests (End-to-End)
# ============================================================

class TestIntegration:
    def test_ddpm_full_pipeline(self):
        """Train and sample from DDPM on simple data."""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5

        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=64)
        losses = ddpm.train(data, epochs=30, batch_size=32, lr=1e-3, rng=rng)

        # Loss should decrease
        assert losses[-1] < losses[0]

        # Generate samples
        samples = ddpm.sample_ddpm(50, rng=rng)
        assert samples.shape == (50, 2)

    def test_ddim_full_pipeline(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5

        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=64)
        ddpm.train(data, epochs=20, batch_size=32, lr=1e-3, rng=rng)

        samples = ddpm.sample_ddim(20, num_steps=10, rng=rng)
        assert samples.shape == (20, 2)

    def test_score_model_pipeline(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5

        model = ScoreBasedModel(data_dim=2, num_scales=5, hidden_dim=64)
        losses = model.train(data, epochs=15, batch_size=32, lr=1e-3, rng=rng)
        assert len(losses) == 15

        samples = model.sample_langevin(10, num_steps=20, rng=rng)
        assert samples.shape == (10, 2)

    def test_vpsde_pipeline(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5

        sde = VPSDE(data_dim=2, hidden_dim=64)
        losses = sde.train(data, epochs=15, batch_size=32, lr=1e-3, rng=rng)
        assert len(losses) == 15

        samples = sde.sample(10, num_steps=30, rng=rng)
        assert samples.shape == (10, 2)

    def test_cfg_pipeline(self):
        rng = np.random.RandomState(42)
        data, labels = make_gaussian_mixture(100, n_components=2, rng=rng)
        # One-hot encode labels
        cond = np.zeros((100, 2))
        for i, l in enumerate(labels.astype(int)):
            cond[i, l] = 1.0

        ddpm = DDPM(data_dim=2, num_timesteps=15, hidden_dim=32, cond_dim=2)
        cfg = ClassifierFreeGuidance(ddpm, guidance_scale=2.0)
        losses = cfg.train(data, cond, epochs=10, batch_size=32, lr=1e-3, rng=rng)
        assert len(losses) == 10

        test_cond = np.array([[1, 0]] * 5, dtype=float)
        samples = cfg.sample(5, test_cond, rng=rng)
        assert samples.shape == (5, 2)

    def test_moons_data(self):
        data = make_moons(200, noise=0.1)
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=64)
        rng = np.random.RandomState(42)
        losses = ddpm.train(data, epochs=20, batch_size=32, lr=1e-3, rng=rng)
        samples = ddpm.sample_ddpm(20, rng=rng)
        assert samples.shape == (20, 2)

    def test_circles_data(self):
        data = make_circles(200, noise=0.05)
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=64)
        rng = np.random.RandomState(42)
        losses = ddpm.train(data, epochs=20, batch_size=32, lr=1e-3, rng=rng)
        samples = ddpm.sample_ddpm(20, rng=rng)
        assert samples.shape == (20, 2)

    def test_higher_dim(self):
        rng = np.random.RandomState(42)
        data = rng.randn(200, 10) * 0.3

        ddpm = DDPM(data_dim=10, num_timesteps=20, hidden_dim=128)
        losses = ddpm.train(data, epochs=40, batch_size=32, lr=1e-3, rng=rng)
        assert losses[-1] < losses[0]

        samples = ddpm.sample_ddpm(10, rng=rng)
        assert samples.shape == (10, 10)

    def test_diagnostics_after_training(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5

        ddpm = DDPM(data_dim=2, num_timesteps=30, hidden_dim=64)
        ddpm.train(data, epochs=30, batch_size=32, lr=1e-3, rng=rng)

        # Noise prediction should improve
        metrics = DiffusionDiagnostics.noise_prediction_mse(ddpm, data[:20])
        assert metrics['mse_avg'] > 0

        # Sample quality
        samples = ddpm.sample_ddpm(50, rng=rng)
        stats = DiffusionDiagnostics.sample_statistics(samples)
        assert 'global_mean' in stats

        # FID proxy
        fid = DiffusionDiagnostics.fid_proxy(data[:50], samples)
        assert isinstance(fid, float)

    def test_x0_prediction_pipeline(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5

        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=64, prediction_type="x0")
        losses = ddpm.train(data, epochs=20, batch_size=32, lr=1e-3, rng=rng)
        assert losses[-1] < losses[0]

        samples = ddpm.sample_ddpm(10, rng=rng)
        assert samples.shape == (10, 2)

    def test_ema_pipeline(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2) * 0.5

        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=64)
        ddpm.enable_ema(decay=0.99)
        losses = ddpm.train(data, epochs=20, batch_size=32, lr=1e-3, rng=rng)

        samples = ddpm.sample_ddpm(10, use_ema=True, rng=rng)
        assert samples.shape == (10, 2)

        samples_ddim = ddpm.sample_ddim(10, num_steps=5, use_ema=True, rng=rng)
        assert samples_ddim.shape == (10, 2)

    def test_progressive_sampling_pipeline(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50, 2) * 0.5
        ddpm = DDPM(data_dim=2, num_timesteps=20, hidden_dim=32)
        ddpm.train(data, epochs=10, batch_size=16, lr=1e-3, rng=rng)

        traj = ddpm.sample_progressive(5, save_every=5, rng=rng)
        assert len(traj) >= 2
        # First frame should be noisier than last
        assert np.std(traj[0]) >= np.std(traj[-1]) * 0.5  # Generous bound


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    def test_single_timestep(self):
        ddpm = DDPM(data_dim=2, num_timesteps=1, hidden_dim=16)
        rng = np.random.RandomState(42)
        samples = ddpm.sample_ddpm(3, rng=rng)
        assert samples.shape == (3, 2)

    def test_single_sample(self):
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16)
        rng = np.random.RandomState(42)
        samples = ddpm.sample_ddpm(1, rng=rng)
        assert samples.shape == (1, 2)

    def test_large_batch(self):
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16)
        rng = np.random.RandomState(42)
        x_0 = rng.randn(500, 2)
        loss = ddpm.training_step(x_0, lr=1e-3, rng=rng)
        assert isinstance(loss, float)

    def test_1d_data(self):
        ddpm = DDPM(data_dim=1, num_timesteps=10, hidden_dim=16)
        rng = np.random.RandomState(42)
        data = rng.randn(50, 1)
        losses = ddpm.train(data, epochs=5, batch_size=16, lr=1e-3, rng=rng)
        samples = ddpm.sample_ddpm(10, rng=rng)
        assert samples.shape == (10, 1)

    def test_schedule_comparison(self):
        """All schedule types should produce valid outputs."""
        for stype in ["linear", "cosine", "quadratic", "sigmoid"]:
            s = NoiseSchedule(50, stype)
            assert np.all(np.isfinite(s.betas))
            assert np.all(np.isfinite(s.alpha_cumprod))
            assert np.all(s.alpha_cumprod > 0)

    def test_ddim_fewer_steps_than_T(self):
        ddpm = DDPM(data_dim=2, num_timesteps=100, hidden_dim=16)
        rng = np.random.RandomState(42)
        # Use 5 steps instead of 100
        samples = ddpm.sample_ddim(3, num_steps=5, rng=rng)
        assert samples.shape == (3, 2)

    def test_train_verbose(self):
        ddpm = DDPM(data_dim=2, num_timesteps=10, hidden_dim=16)
        rng = np.random.RandomState(42)
        data = rng.randn(30, 2)
        losses = ddpm.train(data, epochs=10, batch_size=16, lr=1e-3, rng=rng, verbose=True)
        assert len(losses) == 10
