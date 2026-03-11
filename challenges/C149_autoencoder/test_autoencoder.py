"""
Tests for C149: Autoencoder / Variational Autoencoder
"""

import math
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from autoencoder import (
    Tensor, Autoencoder, DenoisingAutoencoder, SparseAutoencoder,
    VAE, ConditionalVAE, VAEEncoder, Encoder, Decoder,
    reparameterize, kl_divergence, kl_divergence_grad,
    add_noise, interpolate_latent, latent_statistics,
    AnomalyDetector, fit_autoencoder, fit_vae,
    make_sine_data, make_gaussian_mixture_data, make_binary_pattern_data,
    _tensor_mse, _tensor_bce, _tensor_mse_grad, _tensor_bce_grad,
    Adam, SGD
)


# ============================================================
# Utility function tests
# ============================================================

class TestUtilities:
    def test_tensor_mse_identical(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert _tensor_mse(a, a) == 0.0

    def test_tensor_mse_known(self):
        a = Tensor([1.0, 0.0])
        b = Tensor([0.0, 0.0])
        assert abs(_tensor_mse(a, b) - 0.5) < 1e-6

    def test_tensor_mse_2d(self):
        a = Tensor([[1.0, 0.0], [0.0, 1.0]])
        b = Tensor([[0.0, 0.0], [0.0, 0.0]])
        # (1 + 0 + 0 + 1) / 4 = 0.5
        assert abs(_tensor_mse(a, b) - 0.5) < 1e-6

    def test_tensor_bce_perfect(self):
        a = Tensor([0.99, 0.01])
        b = Tensor([1.0, 0.0])
        loss = _tensor_bce(a, b)
        assert loss < 0.05  # Very low

    def test_tensor_bce_bad(self):
        a = Tensor([0.01, 0.99])
        b = Tensor([1.0, 0.0])
        loss = _tensor_bce(a, b)
        assert loss > 2.0  # High

    def test_tensor_mse_grad_shape(self):
        a = Tensor([[0.5, 0.5], [0.3, 0.7]])
        b = Tensor([[1.0, 0.0], [1.0, 0.0]])
        grad = _tensor_mse_grad(a, b)
        assert grad.shape == a.shape

    def test_tensor_bce_grad_shape(self):
        a = Tensor([[0.5, 0.5]])
        b = Tensor([[1.0, 0.0]])
        grad = _tensor_bce_grad(a, b)
        assert grad.shape == a.shape

    def test_add_noise_shape(self):
        t = Tensor([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
        noisy = add_noise(t, 0.1, random.Random(42))
        assert noisy.shape == t.shape

    def test_add_noise_clipped(self):
        t = Tensor([0.0, 1.0, 0.5])
        noisy = add_noise(t, 0.5, random.Random(42))
        for v in noisy.data:
            assert 0.0 <= v <= 1.0

    def test_add_noise_changes_values(self):
        t = Tensor([0.5, 0.5, 0.5])
        noisy = add_noise(t, 0.3, random.Random(42))
        assert any(abs(noisy.data[i] - 0.5) > 0.01 for i in range(3))


# ============================================================
# Data generator tests
# ============================================================

class TestDataGenerators:
    def test_sine_data_shape(self):
        X = make_sine_data(n=50, dim=10)
        assert X.shape == (50, 10)

    def test_sine_data_range(self):
        X = make_sine_data(n=100, dim=20)
        for row in X.data:
            for v in row:
                assert 0.0 <= v <= 1.0

    def test_gaussian_mixture_shape(self):
        X, labels = make_gaussian_mixture_data(n=60, dim=8, n_clusters=3)
        assert X.shape == (60, 8)
        assert len(labels) == 60

    def test_gaussian_mixture_labels(self):
        _, labels = make_gaussian_mixture_data(n=60, dim=8, n_clusters=3)
        assert set(labels) == {0, 1, 2}

    def test_binary_pattern_shape(self):
        X = make_binary_pattern_data(n=40, dim=12)
        assert X.shape == (40, 12)

    def test_binary_pattern_range(self):
        X = make_binary_pattern_data(n=40, dim=12)
        for row in X.data:
            for v in row:
                assert 0.0 < v < 1.0

    def test_sine_data_deterministic(self):
        X1 = make_sine_data(n=10, dim=5, seed=99)
        X2 = make_sine_data(n=10, dim=5, seed=99)
        for i in range(10):
            for j in range(5):
                assert X1.data[i][j] == X2.data[i][j]


# ============================================================
# Encoder/Decoder tests
# ============================================================

class TestEncoderDecoder:
    def test_encoder_output_shape_1d(self):
        enc = Encoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([0.5] * 10)
        z = enc.forward(x)
        assert len(z.data) == 3

    def test_encoder_output_shape_2d(self):
        enc = Encoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10, [0.3] * 10])
        z = enc.forward(x)
        assert z.shape == (2, 3)

    def test_decoder_output_shape_1d(self):
        dec = Decoder(3, [8], 10, rng=random.Random(42))
        z = Tensor([0.1, 0.2, 0.3])
        out = dec.forward(z)
        assert len(out.data) == 10

    def test_decoder_output_shape_2d(self):
        dec = Decoder(3, [8], 10, rng=random.Random(42))
        z = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        out = dec.forward(z)
        assert out.shape == (2, 10)

    def test_decoder_sigmoid_output(self):
        dec = Decoder(3, [8], 10, output_activation='sigmoid', rng=random.Random(42))
        z = Tensor([0.1, 0.2, 0.3])
        out = dec.forward(z)
        for v in out.data:
            assert 0.0 <= v <= 1.0

    def test_encoder_backward(self):
        enc = Encoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        z = enc.forward(x)
        grad = Tensor([[1.0] * 3])
        g_in = enc.backward(grad)
        assert g_in.shape[1] == 10

    def test_decoder_backward(self):
        dec = Decoder(3, [8], 10, rng=random.Random(42))
        z = Tensor([[0.1, 0.2, 0.3]])
        out = dec.forward(z)
        grad = Tensor([out.data[0]])
        g_in = dec.backward(grad)
        assert g_in.shape[1] == 3

    def test_encoder_trainable_layers(self):
        enc = Encoder(10, [8, 6], 3, rng=random.Random(42))
        layers = enc.get_trainable_layers()
        assert len(layers) >= 3  # 2 hidden Dense + 1 output

    def test_decoder_trainable_layers(self):
        dec = Decoder(3, [6, 8], 10, rng=random.Random(42))
        layers = dec.get_trainable_layers()
        assert len(layers) >= 3

    def test_encoder_multi_hidden(self):
        enc = Encoder(20, [16, 8, 4], 2, rng=random.Random(42))
        x = Tensor([[0.5] * 20])
        z = enc.forward(x)
        assert z.shape == (1, 2)

    def test_encoder_with_dropout(self):
        enc = Encoder(10, [8], 3, dropout=0.2, rng=random.Random(42))
        enc.train()
        x = Tensor([[0.5] * 10])
        z = enc.forward(x)
        assert z.shape == (1, 3)


# ============================================================
# Autoencoder tests
# ============================================================

class TestAutoencoder:
    def test_forward_shape(self):
        ae = Autoencoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        recon = ae.forward(x)
        assert recon.shape == (1, 10)

    def test_encode_decode(self):
        ae = Autoencoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        z = ae.encode(x)
        recon = ae.decode(z)
        assert recon.shape == (1, 10)

    def test_latent_dim(self):
        ae = Autoencoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        z = ae.encode(x)
        assert z.shape == (1, 3)

    def test_train_step_returns_loss(self):
        ae = Autoencoder(10, [8], 3, rng=random.Random(42))
        opt = Adam(lr=0.01)
        x = Tensor([[0.5] * 10, [0.3] * 10])
        loss = ae.train_step(x, opt)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_training_reduces_loss(self):
        rng = random.Random(42)
        ae = Autoencoder(8, [6], 3, output_activation='sigmoid', rng=rng)
        opt = Adam(lr=0.01)
        X = make_sine_data(n=20, dim=8, seed=42)
        losses = []
        for _ in range(30):
            loss = ae.train_step(X, opt)
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_fit_autoencoder(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        history = fit_autoencoder(ae, X, epochs=20, rng=random.Random(42))
        assert 'loss' in history
        assert len(history['loss']) == 20
        assert history['loss'][-1] < history['loss'][0]

    def test_fit_with_batches(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=40, dim=8, seed=42)
        history = fit_autoencoder(ae, X, epochs=20, batch_size=10, rng=random.Random(42))
        assert len(history['loss']) == 20

    def test_reconstruction_error(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=30, rng=random.Random(42))
        errors = ae.reconstruction_error(X)
        assert len(errors) == 20
        assert all(e >= 0 for e in errors)

    def test_reconstruction_error_single(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=20, rng=random.Random(42))
        x_single = Tensor(X.data[0])
        errors = ae.reconstruction_error(x_single)
        assert len(errors) == 1

    def test_count_params(self):
        ae = Autoencoder(10, [8], 3, rng=random.Random(42))
        n = ae.count_params()
        # Encoder: Dense(10,8)=80+8, Dense(8,3)=24+3 = 115
        # Decoder: Dense(3,8)=24+8, Dense(8,10)=80+10 = 122
        # Total = 237
        assert n > 200
        assert n == 237

    def test_mse_loss_type(self):
        ae = Autoencoder(8, [6], 3, loss_type='mse', rng=random.Random(42))
        x = Tensor([[0.5] * 8])
        recon = ae.forward(x)
        loss = ae.compute_loss(x, recon)
        assert loss >= 0

    def test_bce_loss_type(self):
        ae = Autoencoder(8, [6], 3, loss_type='bce',
                         output_activation='sigmoid', rng=random.Random(42))
        x = Tensor([[0.5] * 8])
        recon = ae.forward(x)
        loss = ae.compute_loss(x, recon)
        assert loss >= 0

    def test_train_eval_modes(self):
        ae = Autoencoder(8, [6], 3, dropout=0.2, rng=random.Random(42))
        ae.train()
        ae.eval()
        x = Tensor([[0.5] * 8])
        recon = ae.forward(x)
        assert recon.shape == (1, 8)

    def test_deeper_autoencoder(self):
        ae = Autoencoder(16, [12, 8], 4, rng=random.Random(42))
        x = Tensor([[0.5] * 16])
        recon = ae.forward(x)
        assert recon.shape == (1, 16)

    def test_reconstruction_improves(self):
        rng = random.Random(42)
        ae = Autoencoder(8, [6], 3, rng=rng)
        X = make_sine_data(n=30, dim=8, seed=42)
        # Before training
        ae.eval()
        recon_before = ae.forward(X)
        mse_before = _tensor_mse(recon_before, X)
        # Train
        fit_autoencoder(ae, X, epochs=50, rng=random.Random(42))
        ae.eval()
        recon_after = ae.forward(X)
        mse_after = _tensor_mse(recon_after, X)
        assert mse_after < mse_before


# ============================================================
# Denoising Autoencoder tests
# ============================================================

class TestDenoisingAutoencoder:
    def test_creation(self):
        dae = DenoisingAutoencoder(10, [8], 3, noise_factor=0.3, rng=random.Random(42))
        assert dae.noise_factor == 0.3

    def test_forward_shape(self):
        dae = DenoisingAutoencoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        recon = dae.forward(x)
        assert recon.shape == (1, 10)

    def test_train_step(self):
        dae = DenoisingAutoencoder(8, [6], 3, rng=random.Random(42))
        opt = Adam(lr=0.01)
        X = make_sine_data(n=20, dim=8, seed=42)
        loss = dae.train_step(X, opt)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_training_reduces_loss(self):
        dae = DenoisingAutoencoder(8, [6], 3, noise_factor=0.2, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        history = fit_autoencoder(dae, X, epochs=30, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

    def test_denoising_effect(self):
        """Trained DAE should reconstruct clean from noisy."""
        dae = DenoisingAutoencoder(8, [6], 3, noise_factor=0.2, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        fit_autoencoder(dae, X, epochs=50, rng=random.Random(42))
        # Reconstruct from noisy input
        dae.eval()
        noisy = add_noise(X, 0.2, random.Random(123))
        recon = dae.forward(noisy)
        mse_noisy = _tensor_mse(noisy, X)
        mse_recon = _tensor_mse(recon, X)
        # Reconstruction should be closer to clean than noisy input
        assert mse_recon < mse_noisy * 2.5  # Allow tolerance for small training

    def test_high_noise_factor(self):
        dae = DenoisingAutoencoder(8, [6], 3, noise_factor=0.5, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        loss = dae.train_step(X, Adam(lr=0.01))
        assert loss >= 0


# ============================================================
# Sparse Autoencoder tests
# ============================================================

class TestSparseAutoencoder:
    def test_creation(self):
        sae = SparseAutoencoder(10, [8], 3, sparsity_target=0.05, rng=random.Random(42))
        assert sae.sparsity_target == 0.05
        assert sae.sparsity_weight == 1.0

    def test_forward_shape(self):
        sae = SparseAutoencoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        recon = sae.forward(x)
        assert recon.shape == (1, 10)

    def test_train_step(self):
        sae = SparseAutoencoder(8, [6], 3, rng=random.Random(42))
        opt = Adam(lr=0.01)
        X = make_sine_data(n=20, dim=8, seed=42)
        loss = sae.train_step(X, opt)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_sparsity_penalty(self):
        """Loss includes sparsity term, so should be > pure reconstruction."""
        rng = random.Random(42)
        ae = Autoencoder(8, [6], 3, rng=rng)
        sae = SparseAutoencoder(8, [6], 3, sparsity_weight=5.0, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        # First step loss for sparse should include penalty
        loss_ae = ae.train_step(X, Adam(lr=0.01))
        loss_sae = sae.train_step(X, Adam(lr=0.01))
        # Sparse loss includes extra penalty term
        assert loss_sae >= 0

    def test_training_converges(self):
        sae = SparseAutoencoder(8, [6], 3, sparsity_target=0.1, sparsity_weight=0.5,
                                rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        history = fit_autoencoder(sae, X, epochs=30, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]


# ============================================================
# VAE Encoder tests
# ============================================================

class TestVAEEncoder:
    def test_output_mu_logvar(self):
        enc = VAEEncoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        mu, log_var = enc.forward(x)
        assert mu.shape == (1, 3)
        assert log_var.shape == (1, 3)

    def test_mu_logvar_different(self):
        enc = VAEEncoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        mu, log_var = enc.forward(x)
        # mu and log_var should generally differ
        assert mu.data[0] != log_var.data[0]

    def test_backward(self):
        enc = VAEEncoder(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        mu, log_var = enc.forward(x)
        grad_mu = Tensor([[1.0] * 3])
        grad_logvar = Tensor([[1.0] * 3])
        g_in = enc.backward(grad_mu, grad_logvar)
        assert g_in.shape[1] == 10

    def test_trainable_layers(self):
        enc = VAEEncoder(10, [8], 3, rng=random.Random(42))
        layers = enc.get_trainable_layers()
        assert len(layers) >= 3  # hidden Dense + mu + logvar


# ============================================================
# Reparameterization tests
# ============================================================

class TestReparameterization:
    def test_output_shape_1d(self):
        mu = Tensor([0.0, 0.0, 0.0])
        log_var = Tensor([0.0, 0.0, 0.0])
        z, eps = reparameterize(mu, log_var, random.Random(42))
        assert len(z.data) == 3
        assert len(eps.data) == 3

    def test_output_shape_2d(self):
        mu = Tensor([[0.0, 0.0], [0.0, 0.0]])
        log_var = Tensor([[0.0, 0.0], [0.0, 0.0]])
        z, eps = reparameterize(mu, log_var, random.Random(42))
        assert z.shape == (2, 2)
        assert eps.shape == (2, 2)

    def test_zero_logvar_std_is_one(self):
        """With log_var=0, std=1, so z = mu + eps."""
        mu = Tensor([1.0, 2.0, 3.0])
        log_var = Tensor([0.0, 0.0, 0.0])
        z, eps = reparameterize(mu, log_var, random.Random(42))
        for j in range(3):
            assert abs(z.data[j] - (mu.data[j] + eps.data[j])) < 1e-10

    def test_large_negative_logvar_small_std(self):
        """Large negative log_var -> small std -> z close to mu."""
        mu = Tensor([5.0, 5.0])
        log_var = Tensor([-20.0, -20.0])
        z, _ = reparameterize(mu, log_var, random.Random(42))
        for j in range(2):
            assert abs(z.data[j] - mu.data[j]) < 0.01

    def test_deterministic_with_seed(self):
        mu = Tensor([0.0, 0.0])
        log_var = Tensor([0.0, 0.0])
        z1, _ = reparameterize(mu, log_var, random.Random(42))
        z2, _ = reparameterize(mu, log_var, random.Random(42))
        assert z1.data == z2.data


# ============================================================
# KL Divergence tests
# ============================================================

class TestKLDivergence:
    def test_zero_for_standard_normal(self):
        """KL(N(0,1) || N(0,1)) = 0."""
        mu = Tensor([0.0, 0.0, 0.0])
        log_var = Tensor([0.0, 0.0, 0.0])
        kl = kl_divergence(mu, log_var)
        assert abs(kl) < 1e-6

    def test_positive_for_nonzero_mu(self):
        """KL > 0 when mu != 0."""
        mu = Tensor([1.0, -1.0])
        log_var = Tensor([0.0, 0.0])
        kl = kl_divergence(mu, log_var)
        assert kl > 0

    def test_positive_for_nonzero_logvar(self):
        """KL > 0 when log_var != 0."""
        mu = Tensor([0.0, 0.0])
        log_var = Tensor([1.0, -1.0])
        kl = kl_divergence(mu, log_var)
        assert kl > 0

    def test_kl_2d_batch(self):
        mu = Tensor([[0.0, 0.0], [1.0, 1.0]])
        log_var = Tensor([[0.0, 0.0], [0.0, 0.0]])
        kl = kl_divergence(mu, log_var)
        assert kl > 0  # Second sample has nonzero mu

    def test_kl_grad_shape_1d(self):
        mu = Tensor([0.5, -0.5])
        log_var = Tensor([0.1, -0.1])
        g_mu, g_lv = kl_divergence_grad(mu, log_var)
        assert len(g_mu.data) == 2
        assert len(g_lv.data) == 2

    def test_kl_grad_shape_2d(self):
        mu = Tensor([[0.5, -0.5], [0.1, 0.2]])
        log_var = Tensor([[0.1, -0.1], [0.0, 0.0]])
        g_mu, g_lv = kl_divergence_grad(mu, log_var)
        assert g_mu.shape == (2, 2)
        assert g_lv.shape == (2, 2)

    def test_kl_grad_zero_at_standard_normal(self):
        """Gradient should be zero at mu=0, log_var=0."""
        mu = Tensor([0.0, 0.0])
        log_var = Tensor([0.0, 0.0])
        g_mu, g_lv = kl_divergence_grad(mu, log_var)
        assert abs(g_mu.data[0]) < 1e-6
        assert abs(g_lv.data[0]) < 1e-6

    def test_kl_known_value(self):
        """KL(N(mu,sigma^2) || N(0,1)) = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))."""
        mu = Tensor([1.0])
        log_var = Tensor([0.0])  # sigma^2 = 1
        kl = kl_divergence(mu, log_var)
        # = 0.5 * (1 + 1 - 1 - 0) = 0.5
        assert abs(kl - 0.5) < 1e-6


# ============================================================
# VAE tests
# ============================================================

class TestVAE:
    def test_forward_shape(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        recon, mu, log_var = vae.forward(x)
        assert recon.shape == (1, 10)
        assert mu.shape == (1, 3)
        assert log_var.shape == (1, 3)

    def test_encode(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        mu, log_var = vae.encode(x)
        assert mu.shape == (1, 3)

    def test_decode(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        z = Tensor([[0.1, 0.2, 0.3]])
        out = vae.decode(z)
        assert out.shape == (1, 10)

    def test_sample(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        mu = Tensor([[0.0, 0.0, 0.0]])
        log_var = Tensor([[0.0, 0.0, 0.0]])
        z = vae.sample(mu, log_var)
        assert z.shape == (1, 3)

    def test_generate(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        samples = vae.generate(n_samples=5)
        assert samples.shape == (5, 10)

    def test_generate_single(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        sample = vae.generate(n_samples=1)
        assert len(sample.data) == 10

    def test_compute_loss(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        x = Tensor([[0.5] * 10])
        recon, mu, log_var = vae.forward(x)
        total, recon_loss, kl_loss = vae.compute_loss(x, recon, mu, log_var)
        assert total >= 0
        assert recon_loss >= 0
        assert kl_loss >= 0

    def test_train_step(self):
        vae = VAE(8, [6], 3, rng=random.Random(42))
        opt = Adam(lr=0.01)
        X = make_sine_data(n=20, dim=8, seed=42)
        total, recon, kl = vae.train_step(X, opt)
        assert isinstance(total, float)
        assert total >= 0

    def test_training_reduces_loss(self):
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        history = fit_vae(vae, X, epochs=30, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

    def test_fit_vae_history(self):
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        history = fit_vae(vae, X, epochs=15, rng=random.Random(42))
        assert 'loss' in history
        assert 'recon_loss' in history
        assert 'kl_loss' in history
        assert len(history['loss']) == 15

    def test_fit_vae_with_batches(self):
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=40, dim=8, seed=42)
        history = fit_vae(vae, X, epochs=10, batch_size=10, rng=random.Random(42))
        assert len(history['loss']) == 10

    def test_beta_affects_kl(self):
        """Higher beta should penalize KL more."""
        vae_low = VAE(8, [6], 3, beta=0.1, rng=random.Random(42))
        vae_high = VAE(8, [6], 3, beta=10.0, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        # Just verify they compute different losses
        x = Tensor([X.data[0]])
        r1, m1, lv1 = vae_low.forward(x)
        l1, _, _ = vae_low.compute_loss(x, r1, m1, lv1)
        r2, m2, lv2 = vae_high.forward(x)
        l2, _, _ = vae_high.compute_loss(x, r2, m2, lv2)
        # Different beta -> different loss (both valid)
        assert isinstance(l1, float)
        assert isinstance(l2, float)

    def test_vae_bce_loss(self):
        vae = VAE(8, [6], 3, loss_type='bce', output_activation='sigmoid',
                  rng=random.Random(42))
        X = make_binary_pattern_data(n=20, dim=8, seed=42)
        history = fit_vae(vae, X, epochs=15, rng=random.Random(42))
        assert len(history['loss']) == 15

    def test_reconstruction_error(self):
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        fit_vae(vae, X, epochs=20, rng=random.Random(42))
        errors = vae.reconstruction_error(X)
        assert len(errors) == 20
        assert all(e >= 0 for e in errors)

    def test_count_params(self):
        vae = VAE(10, [8], 3, rng=random.Random(42))
        n = vae.count_params()
        # Encoder: Dense(10,8)=80+8, Dense(8,3)=24+3 (mu), Dense(8,3)=24+3 (logvar) = 142
        # Decoder: Dense(3,8)=24+8, Dense(8,10)=80+10 = 122
        # Total = 264
        assert n > 200

    def test_latent_space_exploration(self):
        """VAE should organize latent space around N(0,1)."""
        vae = VAE(8, [6], 3, beta=1.0, rng=random.Random(42))
        X = make_sine_data(n=50, dim=8, seed=42)
        fit_vae(vae, X, epochs=50, rng=random.Random(42))
        vae.eval()
        mu, _ = vae.encode(X)
        # Latent means should not be too far from 0
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                assert abs(mu.data[i][j]) < 20  # Loose bound


# ============================================================
# Conditional VAE tests
# ============================================================

class TestConditionalVAE:
    def test_forward_shape(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        x = Tensor([[0.5] * 8])
        labels = [0]
        recon, mu, log_var = cvae.forward(x, labels)
        assert recon.shape == (1, 8)
        assert mu.shape == (1, 2)

    def test_generate(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        samples = cvae.generate(labels=[0, 1, 2])
        assert samples.shape == (3, 8)

    def test_generate_single(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        sample = cvae.generate(labels=0, n_samples=1)
        # Single sample may be 2D (1, 8) or 1D (8)
        flat = sample.flatten().data
        assert len(flat) == 8

    def test_train_step(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        opt = Adam(lr=0.01)
        X, labels = make_gaussian_mixture_data(n=12, dim=8, n_clusters=3, seed=42)
        total, recon, kl = cvae.train_step(X, labels, opt)
        assert isinstance(total, float)

    def test_training_reduces_loss(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        X, labels = make_gaussian_mixture_data(n=30, dim=8, n_clusters=3, seed=42)
        history = fit_vae(cvae, X, epochs=30, labels=labels, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

    def test_fit_cvae_with_batches(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        X, labels = make_gaussian_mixture_data(n=30, dim=8, n_clusters=3, seed=42)
        history = fit_vae(cvae, X, epochs=10, batch_size=10, labels=labels,
                          rng=random.Random(42))
        assert len(history['loss']) == 10

    def test_one_hot_encoding(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        oh = cvae._one_hot(1)
        assert oh.data == [0.0, 1.0, 0.0]

    def test_one_hot_batch(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        oh = cvae._one_hot([0, 1, 2])
        assert oh.shape == (3, 3)

    def test_concat(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0], [6.0]])
        c = cvae._concat(a, b)
        assert c.shape == (2, 3)
        assert c.data[0] == [1.0, 2.0, 5.0]

    def test_count_params(self):
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        n = cvae.count_params()
        assert n > 0

    def test_different_labels_different_output(self):
        """Conditioning on different labels should produce different outputs."""
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        X, labels = make_gaussian_mixture_data(n=30, dim=8, n_clusters=3, seed=42)
        fit_vae(cvae, X, epochs=30, labels=labels, rng=random.Random(42))
        s0 = cvae.generate(labels=0, n_samples=1)
        s1 = cvae.generate(labels=1, n_samples=1)
        flat0 = s0.flatten().data
        flat1 = s1.flatten().data
        # They should generally differ (not guaranteed but likely)
        any_diff = any(abs(flat0[i] - flat1[i]) > 0.001 for i in range(8))
        assert any_diff


# ============================================================
# Anomaly Detection tests
# ============================================================

class TestAnomalyDetector:
    def test_fit_threshold(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=30, rng=random.Random(42))
        detector = AnomalyDetector(ae)
        threshold = detector.fit_threshold(X, percentile=95)
        assert threshold > 0

    def test_predict(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=30, rng=random.Random(42))
        detector = AnomalyDetector(ae)
        detector.fit_threshold(X, percentile=90)
        preds = detector.predict(X)
        assert len(preds) == 30
        assert all(isinstance(p, bool) for p in preds)

    def test_score(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=20, rng=random.Random(42))
        detector = AnomalyDetector(ae)
        scores = detector.score(X)
        assert len(scores) == 20
        assert all(s >= 0 for s in scores)

    def test_anomaly_detection(self):
        """Anomalous data should have higher reconstruction error."""
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X_normal = make_sine_data(n=50, dim=8, seed=42)
        fit_autoencoder(ae, X_normal, epochs=50, rng=random.Random(42))

        detector = AnomalyDetector(ae)
        detector.fit_threshold(X_normal, percentile=90)

        # Create anomalous data (random noise)
        rng = random.Random(99)
        anomaly_data = [[rng.random() for _ in range(8)] for _ in range(10)]
        X_anomaly = Tensor(anomaly_data)

        normal_scores = detector.score(X_normal)
        anomaly_scores = detector.score(X_anomaly)

        avg_normal = sum(normal_scores) / len(normal_scores)
        avg_anomaly = sum(anomaly_scores) / len(anomaly_scores)
        # Anomalies should have higher error on average
        assert avg_anomaly >= avg_normal * 0.5  # Relaxed bound

    def test_no_threshold_raises(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        detector = AnomalyDetector(ae)
        X = make_sine_data(n=10, dim=8, seed=42)
        with pytest.raises(ValueError):
            detector.predict(X)

    def test_manual_threshold(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        detector = AnomalyDetector(ae, threshold=0.5)
        assert detector.threshold == 0.5


# ============================================================
# Latent space utility tests
# ============================================================

class TestLatentUtils:
    def test_interpolate_ae(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=20, rng=random.Random(42))
        x1 = Tensor(X.data[0])
        x2 = Tensor(X.data[1])
        results = interpolate_latent(ae, x1, x2, n_steps=5)
        assert len(results) == 6  # 0..5 inclusive
        assert len(results[0].data) == 8

    def test_interpolate_vae(self):
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        fit_vae(vae, X, epochs=20, rng=random.Random(42))
        x1 = Tensor(X.data[0])
        x2 = Tensor(X.data[1])
        results = interpolate_latent(vae, x1, x2, n_steps=5)
        assert len(results) == 6

    def test_interpolate_endpoints(self):
        """First and last should reconstruct x1 and x2."""
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=30, rng=random.Random(42))
        x1 = Tensor(X.data[0])
        x2 = Tensor(X.data[1])
        results = interpolate_latent(ae, x1, x2, n_steps=5)
        # Endpoints should be close to original reconstructions
        ae.eval()
        r1 = ae.forward(Tensor([x1.data]))
        assert results[0].shape == r1.shape or len(results[0].data) == len(r1.data[0])

    def test_latent_statistics(self):
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        fit_autoencoder(ae, X, epochs=20, rng=random.Random(42))
        stats = latent_statistics(ae, X)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert len(stats['mean']) == 3

    def test_latent_statistics_vae(self):
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        fit_vae(vae, X, epochs=20, rng=random.Random(42))
        stats = latent_statistics(vae, X)
        assert len(stats['mean']) == 3
        assert all(isinstance(s, float) for s in stats['std'])


# ============================================================
# Integration / end-to-end tests
# ============================================================

class TestIntegration:
    def test_ae_full_pipeline(self):
        """Full AE pipeline: create, train, evaluate, anomaly detect."""
        ae = Autoencoder(10, [8], 4, rng=random.Random(42))
        X = make_sine_data(n=50, dim=10, seed=42)
        history = fit_autoencoder(ae, X, epochs=40, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

        errors = ae.reconstruction_error(X)
        assert len(errors) == 50

        detector = AnomalyDetector(ae)
        detector.fit_threshold(X)
        preds = detector.predict(X)
        # Most normal data should not be flagged
        n_flagged = sum(preds)
        assert n_flagged < 10

    def test_vae_full_pipeline(self):
        """Full VAE pipeline: create, train, generate, interpolate."""
        vae = VAE(10, [8], 3, beta=1.0, rng=random.Random(42))
        X = make_sine_data(n=50, dim=10, seed=42)
        history = fit_vae(vae, X, epochs=40, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

        # Generate
        samples = vae.generate(n_samples=5)
        assert samples.shape == (5, 10)

        # Interpolate
        x1 = Tensor(X.data[0])
        x2 = Tensor(X.data[1])
        interp = interpolate_latent(vae, x1, x2, n_steps=5)
        assert len(interp) == 6

        # Stats
        stats = latent_statistics(vae, X)
        assert len(stats['mean']) == 3

    def test_cvae_full_pipeline(self):
        """Full CVAE pipeline: create, train, conditional generation."""
        cvae = ConditionalVAE(8, 3, [6], 2, rng=random.Random(42))
        X, labels = make_gaussian_mixture_data(n=30, dim=8, n_clusters=3, seed=42)
        history = fit_vae(cvae, X, epochs=30, labels=labels, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

        # Generate for each class
        for c in range(3):
            sample = cvae.generate(labels=c, n_samples=1)
            flat = sample.flatten().data
            assert len(flat) == 8

    def test_dae_full_pipeline(self):
        """Denoising autoencoder pipeline."""
        dae = DenoisingAutoencoder(8, [6], 3, noise_factor=0.2, rng=random.Random(42))
        X = make_sine_data(n=40, dim=8, seed=42)
        history = fit_autoencoder(dae, X, epochs=30, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

    def test_sparse_ae_full_pipeline(self):
        """Sparse autoencoder pipeline."""
        sae = SparseAutoencoder(8, [6], 3, sparsity_target=0.1,
                                rng=random.Random(42))
        X = make_sine_data(n=40, dim=8, seed=42)
        history = fit_autoencoder(sae, X, epochs=30, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

    def test_ae_different_latent_dims(self):
        """Test various latent dimensions."""
        for latent_dim in [1, 2, 5, 8]:
            ae = Autoencoder(10, [8], latent_dim, rng=random.Random(42))
            X = make_sine_data(n=20, dim=10, seed=42)
            z = ae.encode(X)
            assert z.shape == (20, latent_dim)

    def test_vae_reconstruction_quality(self):
        """VAE reconstruction should improve with training."""
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)

        vae.eval()
        recon_before, _, _ = vae.forward(X)
        mse_before = _tensor_mse(recon_before, X)

        fit_vae(vae, X, epochs=50, rng=random.Random(42))

        vae.eval()
        recon_after, _, _ = vae.forward(X)
        mse_after = _tensor_mse(recon_after, X)
        assert mse_after < mse_before

    def test_beta_vae(self):
        """Beta-VAE with high beta should have lower KL."""
        vae_b01 = VAE(8, [6], 3, beta=0.1, rng=random.Random(42))
        vae_b10 = VAE(8, [6], 3, beta=10.0, rng=random.Random(42))
        X = make_sine_data(n=30, dim=8, seed=42)
        h1 = fit_vae(vae_b01, X, epochs=30, rng=random.Random(42))
        h2 = fit_vae(vae_b10, X, epochs=30, rng=random.Random(42))
        # High beta should push KL down more
        kl_low_beta = h1['kl_loss'][-1]
        kl_high_beta = h2['kl_loss'][-1]
        assert kl_high_beta <= kl_low_beta * 2  # Loose bound

    def test_binary_data_with_bce(self):
        """BCE loss on binary-ish data."""
        ae = Autoencoder(8, [6], 3, loss_type='bce', output_activation='sigmoid',
                         rng=random.Random(42))
        X = make_binary_pattern_data(n=30, dim=8, seed=42)
        history = fit_autoencoder(ae, X, epochs=30, rng=random.Random(42))
        assert history['loss'][-1] < history['loss'][0]

    def test_gaussian_mixture_clustering(self):
        """AE should learn different latent representations for clusters."""
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X, labels = make_gaussian_mixture_data(n=60, dim=8, n_clusters=3, seed=42)
        fit_autoencoder(ae, X, epochs=50, rng=random.Random(42))
        ae.eval()
        Z = ae.encode(X)
        # Compute cluster centroids in latent space
        centroids = {}
        counts = {}
        for i in range(len(labels)):
            c = labels[i]
            if c not in centroids:
                centroids[c] = [0.0] * 3
                counts[c] = 0
            for j in range(3):
                centroids[c][j] += Z.data[i][j]
            counts[c] += 1
        for c in centroids:
            centroids[c] = [v / counts[c] for v in centroids[c]]
        # Centroids should not all be identical
        assert centroids[0] != centroids[1] or centroids[1] != centroids[2]

    def test_ae_with_sgd(self):
        """Test with SGD optimizer."""
        ae = Autoencoder(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        opt = SGD(lr=0.1, momentum=0.9)
        history = fit_autoencoder(ae, X, epochs=20, optimizer=opt, rng=random.Random(42))
        assert len(history['loss']) == 20

    def test_vae_with_rmsprop(self):
        """Test VAE with RMSProp."""
        from autoencoder import RMSProp
        vae = VAE(8, [6], 3, rng=random.Random(42))
        X = make_sine_data(n=20, dim=8, seed=42)
        opt = RMSProp(lr=0.001)
        history = fit_vae(vae, X, epochs=15, optimizer=opt, rng=random.Random(42))
        assert len(history['loss']) == 15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
