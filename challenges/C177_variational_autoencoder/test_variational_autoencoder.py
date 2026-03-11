"""Tests for C177: Variational Autoencoder."""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from variational_autoencoder import (
    Tensor, _RNG, _add_tensors, _sub_tensors, _mul_tensors, _scale_tensor,
    _apply_tensor, _tensor_sum, _tensor_mean, _ones_like, _zeros_like,
    _get_row, _set_row, _stack_rows, _concat_tensors, _batch_size,
    _tensor_to_flat, _flat_to_tensor,
    Encoder, Decoder, Reparameterize, VAELoss,
    VAE, ConditionalVAE, BetaVAE,
    VectorQuantizer, VQVAE,
    LatentSpace, VAETrainer, VQVAETrainer,
    make_vae_data, make_labeled_data,
)


class TestRNG(unittest.TestCase):
    def test_reproducibility(self):
        r1, r2 = _RNG(42), _RNG(42)
        self.assertEqual(r1.uniform(), r2.uniform())
        self.assertEqual(r1.normal(), r2.normal())

    def test_uniform_range(self):
        rng = _RNG(1)
        for _ in range(50):
            v = rng.uniform(0.0, 1.0)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_normal_tensor(self):
        rng = _RNG(7)
        t = rng.normal_tensor((3, 4))
        self.assertEqual(t.shape, (3, 4))

    def test_normal_tensor_1d(self):
        rng = _RNG(7)
        t = rng.normal_tensor((5,))
        self.assertEqual(t.shape, (5,))


class TestTensorHelpers(unittest.TestCase):
    def test_add_1d(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = _add_tensors(a, b)
        self.assertEqual(c.data, [5.0, 7.0, 9.0])

    def test_add_2d(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = _add_tensors(a, b)
        self.assertEqual(c.data, [[6.0, 8.0], [10.0, 12.0]])

    def test_sub(self):
        a = Tensor([5.0, 3.0])
        b = Tensor([1.0, 1.0])
        c = _sub_tensors(a, b)
        self.assertEqual(c.data, [4.0, 2.0])

    def test_mul(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = _mul_tensors(a, b)
        self.assertEqual(c.data, [8.0, 15.0])

    def test_scale(self):
        t = Tensor([2.0, 4.0])
        c = _scale_tensor(t, 0.5)
        self.assertEqual(c.data, [1.0, 2.0])

    def test_apply(self):
        t = Tensor([1.0, 4.0, 9.0])
        c = _apply_tensor(t, math.sqrt)
        self.assertAlmostEqual(c.data[0], 1.0)
        self.assertAlmostEqual(c.data[1], 2.0)
        self.assertAlmostEqual(c.data[2], 3.0)

    def test_sum_mean(self):
        t = Tensor([2.0, 4.0, 6.0])
        self.assertEqual(_tensor_sum(t), 12.0)
        self.assertEqual(_tensor_mean(t), 4.0)

    def test_ones_zeros_like(self):
        t = Tensor([1.0, 2.0])
        self.assertEqual(_ones_like(t).data, [1.0, 1.0])
        self.assertEqual(_zeros_like(t).data, [0.0, 0.0])

    def test_get_set_row(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        row = _get_row(t, 0)
        self.assertEqual(row.data, [1.0, 2.0])
        _set_row(t, 1, Tensor([5.0, 6.0]))
        self.assertEqual(t.data[1], [5.0, 6.0])

    def test_stack_rows(self):
        rows = [Tensor([1.0, 2.0]), Tensor([3.0, 4.0])]
        t = _stack_rows(rows)
        self.assertEqual(t.shape, (2, 2))

    def test_concat(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = _concat_tensors(a, b)
        self.assertEqual(c.data, [1.0, 2.0, 3.0, 4.0])

    def test_batch_size(self):
        t = Tensor([[1.0], [2.0], [3.0]])
        self.assertEqual(_batch_size(t), 3)

    def test_flat_conversions(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        flat = _tensor_to_flat(t)
        self.assertEqual(flat, [1.0, 2.0, 3.0, 4.0])
        t2 = _flat_to_tensor(flat, (2, 2))
        self.assertEqual(t2.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_flat_1d(self):
        t = Tensor([1.0, 2.0, 3.0])
        flat = _tensor_to_flat(t)
        self.assertEqual(flat, [1.0, 2.0, 3.0])
        t2 = _flat_to_tensor(flat, (3,))
        self.assertEqual(t2.data, [1.0, 2.0, 3.0])


class TestEncoder(unittest.TestCase):
    def test_creation(self):
        enc = Encoder(8, [16, 8], 4)
        self.assertEqual(enc.input_dim, 8)
        self.assertEqual(enc.latent_dim, 4)

    def test_forward_1d(self):
        enc = Encoder(4, [8], 2, rng=_RNG(42))
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        mu, logvar = enc.forward(x)
        self.assertEqual(mu.shape, (2,))
        self.assertEqual(logvar.shape, (2,))

    def test_forward_batch(self):
        enc = Encoder(4, [8], 2, rng=_RNG(42))
        x = Tensor([[0.5, 0.3, 0.7, 0.2], [0.1, 0.9, 0.4, 0.6]])
        mu, logvar = enc.forward(x)
        self.assertEqual(mu.shape, (2, 2))
        self.assertEqual(logvar.shape, (2, 2))

    def test_backward(self):
        enc = Encoder(4, [8], 2, rng=_RNG(42))
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        mu, logvar = enc.forward(x)
        grad_mu = Tensor([0.1, -0.1])
        grad_logvar = Tensor([0.05, -0.05])
        grad_input = enc.backward(grad_mu, grad_logvar)
        self.assertEqual(grad_input.shape, (4,))

    def test_trainable_layers(self):
        enc = Encoder(4, [8], 2, rng=_RNG(42))
        layers = enc.get_trainable_layers()
        self.assertGreater(len(layers), 0)

    def test_train_eval(self):
        enc = Encoder(4, [8], 2, rng=_RNG(42))
        enc.train()
        enc.eval()  # No error

    def test_no_hidden(self):
        enc = Encoder(4, [], 2, rng=_RNG(42))
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        mu, logvar = enc.forward(x)
        self.assertEqual(mu.shape, (2,))

    def test_with_batchnorm(self):
        enc = Encoder(4, [8], 2, use_batchnorm=True, rng=_RNG(42))
        x = Tensor([[0.5, 0.3, 0.7, 0.2], [0.1, 0.9, 0.4, 0.6]])
        mu, logvar = enc.forward(x)
        self.assertEqual(mu.shape, (2, 2))


class TestDecoder(unittest.TestCase):
    def test_creation(self):
        dec = Decoder(2, [8], 4)
        self.assertEqual(dec.latent_dim, 2)
        self.assertEqual(dec.output_dim, 4)

    def test_forward_1d(self):
        dec = Decoder(2, [8], 4, rng=_RNG(42))
        z = Tensor([0.5, -0.3])
        out = dec.forward(z)
        self.assertEqual(out.shape, (4,))

    def test_forward_batch(self):
        dec = Decoder(2, [8], 4, rng=_RNG(42))
        z = Tensor([[0.5, -0.3], [0.1, 0.2]])
        out = dec.forward(z)
        self.assertEqual(out.shape, (2, 4))

    def test_backward(self):
        dec = Decoder(2, [8], 4, rng=_RNG(42))
        z = Tensor([0.5, -0.3])
        out = dec.forward(z)
        grad = Tensor([0.1, -0.1, 0.05, -0.05])
        grad_z = dec.backward(grad)
        self.assertEqual(grad_z.shape, (2,))

    def test_no_output_activation(self):
        dec = Decoder(2, [8], 4, output_activation='none', rng=_RNG(42))
        z = Tensor([0.5, -0.3])
        out = dec.forward(z)
        self.assertEqual(out.shape, (4,))

    def test_sigmoid_output(self):
        dec = Decoder(2, [8], 4, output_activation='sigmoid', rng=_RNG(42))
        z = Tensor([0.5, -0.3])
        out = dec.forward(z)
        for v in out.data:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_trainable_layers(self):
        dec = Decoder(2, [8], 4, rng=_RNG(42))
        layers = dec.get_trainable_layers()
        self.assertGreater(len(layers), 0)


class TestReparameterize(unittest.TestCase):
    def test_forward_1d(self):
        reparam = Reparameterize(rng=_RNG(42))
        mu = Tensor([1.0, 2.0])
        logvar = Tensor([0.0, 0.0])  # std = 1
        z = reparam.forward(mu, logvar)
        self.assertEqual(z.shape, (2,))

    def test_forward_2d(self):
        reparam = Reparameterize(rng=_RNG(42))
        mu = Tensor([[1.0, 2.0], [3.0, 4.0]])
        logvar = Tensor([[0.0, 0.0], [0.0, 0.0]])
        z = reparam.forward(mu, logvar)
        self.assertEqual(z.shape, (2, 2))

    def test_mean_close_to_mu(self):
        # With logvar=very negative (tiny std), z should be close to mu
        reparam = Reparameterize(rng=_RNG(42))
        mu = Tensor([5.0, -3.0])
        logvar = Tensor([-20.0, -20.0])
        z = reparam.forward(mu, logvar)
        self.assertAlmostEqual(z.data[0], 5.0, places=3)
        self.assertAlmostEqual(z.data[1], -3.0, places=3)

    def test_backward(self):
        reparam = Reparameterize(rng=_RNG(42))
        mu = Tensor([1.0, 2.0])
        logvar = Tensor([0.0, 0.0])
        z = reparam.forward(mu, logvar)
        grad_z = Tensor([1.0, 1.0])
        grad_mu, grad_logvar = reparam.backward(grad_z)
        self.assertEqual(grad_mu.shape, (2,))
        self.assertEqual(grad_logvar.shape, (2,))
        # grad_mu = grad_z (identity)
        self.assertAlmostEqual(grad_mu.data[0], 1.0)

    def test_backward_2d(self):
        reparam = Reparameterize(rng=_RNG(42))
        mu = Tensor([[1.0, 2.0]])
        logvar = Tensor([[0.0, 0.0]])
        z = reparam.forward(mu, logvar)
        grad_z = Tensor([[1.0, 1.0]])
        grad_mu, grad_logvar = reparam.backward(grad_z)
        self.assertEqual(grad_mu.shape, (1, 2))


class TestVAELoss(unittest.TestCase):
    def test_creation_mse(self):
        loss = VAELoss(recon_type='mse')
        self.assertEqual(loss.recon_type, 'mse')

    def test_creation_bce(self):
        loss = VAELoss(recon_type='bce')
        self.assertEqual(loss.recon_type, 'bce')

    def test_forward(self):
        loss = VAELoss(recon_type='mse', beta=1.0)
        x_recon = Tensor([0.5, 0.6])
        x_orig = Tensor([0.5, 0.5])
        mu = Tensor([0.0, 0.0])
        logvar = Tensor([0.0, 0.0])
        total, recon, kl = loss.forward(x_recon, x_orig, mu, logvar)
        self.assertGreaterEqual(recon, 0.0)
        # KL with N(0,1) should be 0
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_kl_nonzero(self):
        loss = VAELoss(recon_type='mse', beta=1.0)
        x_recon = Tensor([0.5])
        x_orig = Tensor([0.5])
        mu = Tensor([2.0])
        logvar = Tensor([0.0])
        total, recon, kl = loss.forward(x_recon, x_orig, mu, logvar)
        # KL should be > 0 when mu != 0
        self.assertGreater(kl, 0.0)

    def test_beta_scaling(self):
        loss1 = VAELoss(recon_type='mse', beta=1.0)
        loss2 = VAELoss(recon_type='mse', beta=5.0)
        x_recon = Tensor([0.5])
        x_orig = Tensor([0.5])
        mu = Tensor([2.0])
        logvar = Tensor([0.0])
        t1, _, kl1 = loss1.forward(x_recon, x_orig, mu, logvar)
        t2, _, kl2 = loss2.forward(x_recon, x_orig, mu, logvar)
        # Same KL, but beta=5 means higher total
        self.assertAlmostEqual(kl1, kl2, places=5)
        self.assertGreater(t2, t1)

    def test_backward_recon(self):
        loss = VAELoss(recon_type='mse')
        x_recon = Tensor([0.5, 0.6])
        x_orig = Tensor([0.5, 0.5])
        grad = loss.backward_recon(x_recon, x_orig)
        self.assertEqual(grad.shape, (2,))

    def test_backward_kl(self):
        loss = VAELoss(recon_type='mse', beta=1.0)
        mu = Tensor([2.0, -1.0])
        logvar = Tensor([0.5, -0.5])
        grad_mu, grad_logvar = loss.backward_kl(mu, logvar)
        self.assertEqual(grad_mu.shape, (2,))
        self.assertEqual(grad_logvar.shape, (2,))

    def test_kl_batch(self):
        loss = VAELoss(recon_type='mse')
        mu = Tensor([[0.0, 0.0], [0.0, 0.0]])
        logvar = Tensor([[0.0, 0.0], [0.0, 0.0]])
        x_recon = Tensor([[0.5, 0.5], [0.5, 0.5]])
        x_orig = Tensor([[0.5, 0.5], [0.5, 0.5]])
        total, recon, kl = loss.forward(x_recon, x_orig, mu, logvar)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_invalid_recon_type(self):
        with self.assertRaises(ValueError):
            VAELoss(recon_type='invalid')


class TestVAE(unittest.TestCase):
    def test_creation(self):
        vae = VAE(8, [16], 4, seed=42)
        self.assertEqual(vae.input_dim, 8)
        self.assertEqual(vae.latent_dim, 4)

    def test_forward_1d(self):
        vae = VAE(4, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = vae.forward(x)
        self.assertEqual(x_recon.shape, (4,))
        self.assertEqual(mu.shape, (2,))
        self.assertEqual(logvar.shape, (2,))
        self.assertEqual(z.shape, (2,))

    def test_forward_batch(self):
        vae = VAE(4, [8], 2, seed=42)
        x = Tensor([[0.5, 0.3, 0.7, 0.2], [0.1, 0.9, 0.4, 0.6]])
        x_recon, mu, logvar, z = vae.forward(x)
        self.assertEqual(x_recon.shape, (2, 4))
        self.assertEqual(mu.shape, (2, 2))

    def test_compute_loss(self):
        vae = VAE(4, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = vae.forward(x)
        total, recon, kl = vae.compute_loss(x_recon, x, mu, logvar)
        self.assertIsInstance(total, float)
        self.assertGreaterEqual(recon, 0.0)

    def test_backward(self):
        vae = VAE(4, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = vae.forward(x)
        vae.backward(x_recon, x, mu, logvar)  # No error

    def test_train_step(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        vae.train()
        x_recon, mu, logvar, z = vae.forward(x)
        total1, _, _ = vae.compute_loss(x_recon, x, mu, logvar)
        vae.backward(x_recon, x, mu, logvar)
        opt.step(vae.get_trainable_layers())
        # After step, loss should change
        x_recon2, mu2, logvar2, z2 = vae.forward(x)
        total2, _, _ = vae.compute_loss(x_recon2, x, mu2, logvar2)
        self.assertNotAlmostEqual(total1, total2, places=5)

    def test_encode(self):
        vae = VAE(4, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        mu, logvar = vae.encode(x)
        self.assertEqual(mu.shape, (2,))

    def test_decode(self):
        vae = VAE(4, [8], 2, seed=42)
        z = Tensor([0.5, -0.3])
        out = vae.decode(z)
        self.assertEqual(out.shape, (4,))

    def test_sample(self):
        vae = VAE(4, [8], 2, seed=42)
        samples = vae.sample(n_samples=5)
        self.assertEqual(samples.shape, (5, 4))

    def test_reconstruct(self):
        vae = VAE(4, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        recon = vae.reconstruct(x)
        self.assertEqual(recon.shape, (4,))

    def test_get_trainable_layers(self):
        vae = VAE(4, [8], 2, seed=42)
        layers = vae.get_trainable_layers()
        self.assertGreater(len(layers), 0)

    def test_train_eval_modes(self):
        vae = VAE(4, [8], 2, seed=42)
        vae.train()
        vae.eval()

    def test_sigmoid_output_range(self):
        vae = VAE(4, [8], 2, output_activation='sigmoid', seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        recon = vae.reconstruct(x)
        for v in recon.data:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_multiple_hidden(self):
        vae = VAE(8, [16, 8], 4, seed=42)
        x = Tensor([0.5] * 8)
        x_recon, mu, logvar, z = vae.forward(x)
        self.assertEqual(x_recon.shape, (8,))
        self.assertEqual(mu.shape, (4,))


class TestConditionalVAE(unittest.TestCase):
    def test_creation(self):
        cvae = ConditionalVAE(8, 3, [16], 4, seed=42)
        self.assertEqual(cvae.input_dim, 8)
        self.assertEqual(cvae.num_classes, 3)
        self.assertEqual(cvae.latent_dim, 4)

    def test_forward_1d(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = cvae.forward(x, 1)
        self.assertEqual(x_recon.shape, (4,))
        self.assertEqual(mu.shape, (2,))

    def test_forward_batch(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        x = Tensor([[0.5, 0.3, 0.7, 0.2], [0.1, 0.9, 0.4, 0.6]])
        x_recon, mu, logvar, z = cvae.forward(x, [0, 1])
        self.assertEqual(x_recon.shape, (2, 4))

    def test_compute_loss(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = cvae.forward(x, 0)
        total, recon, kl = cvae.compute_loss(x_recon, x, mu, logvar)
        self.assertIsInstance(total, float)

    def test_backward(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = cvae.forward(x, 0)
        cvae.backward(x_recon, x, mu, logvar)

    def test_generate(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        samples = cvae.generate(1, n_samples=5)
        self.assertEqual(samples.shape, (5, 4))

    def test_different_classes(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        s0 = cvae.generate(0, n_samples=1)
        s1 = cvae.generate(1, n_samples=1)
        # Different classes should produce different outputs
        diff = sum(abs(s0.data[0][i] - s1.data[0][i]) for i in range(4))
        # They won't be identical (different conditioning)
        self.assertGreater(diff, 0.0)

    def test_one_hot(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        oh = cvae._one_hot(2)
        self.assertEqual(oh.data, [0.0, 0.0, 1.0])

    def test_one_hot_batch(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        oh = cvae._one_hot([0, 1, 2])
        self.assertEqual(len(oh.data), 3)
        self.assertEqual(oh.data[0], [1.0, 0.0, 0.0])

    def test_trainable_layers(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        layers = cvae.get_trainable_layers()
        self.assertGreater(len(layers), 0)

    def test_train_eval(self):
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        cvae.train()
        cvae.eval()


class TestBetaVAE(unittest.TestCase):
    def test_creation(self):
        bvae = BetaVAE(8, [16], 4, beta=4.0, seed=42)
        self.assertEqual(bvae.base_beta, 4.0)

    def test_higher_beta_more_kl_weight(self):
        bvae = BetaVAE(4, [8], 2, beta=10.0, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = bvae.forward(x)
        total, recon, kl = bvae.compute_loss(x_recon, x, mu, logvar)
        self.assertGreaterEqual(total, recon)

    def test_warmup_schedule(self):
        bvae = BetaVAE(4, [8], 2, beta=4.0, beta_schedule='warmup', seed=42)
        bvae.update_beta(0, 100)
        self.assertAlmostEqual(bvae.get_beta(), 0.0)
        bvae.update_beta(25, 100)
        self.assertAlmostEqual(bvae.get_beta(), 2.0)
        bvae.update_beta(50, 100)
        self.assertAlmostEqual(bvae.get_beta(), 4.0)

    def test_cyclical_schedule(self):
        bvae = BetaVAE(4, [8], 2, beta=4.0, beta_schedule='cyclical', seed=42)
        bvae.update_beta(0, 100)
        b0 = bvae.get_beta()
        bvae.update_beta(12, 100)
        b12 = bvae.get_beta()
        # Should increase over cycle
        self.assertGreater(b12, b0)

    def test_no_schedule(self):
        bvae = BetaVAE(4, [8], 2, beta=4.0, beta_schedule=None, seed=42)
        bvae.update_beta(50, 100)
        self.assertAlmostEqual(bvae.get_beta(), 4.0)

    def test_inherits_vae(self):
        bvae = BetaVAE(4, [8], 2, beta=4.0, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = bvae.forward(x)
        self.assertEqual(x_recon.shape, (4,))


class TestVectorQuantizer(unittest.TestCase):
    def test_creation(self):
        vq = VectorQuantizer(16, 4)
        self.assertEqual(vq.num_embeddings, 16)
        self.assertEqual(vq.embedding_dim, 4)

    def test_forward_1d(self):
        vq = VectorQuantizer(8, 3, rng=_RNG(42))
        z_e = Tensor([0.5, -0.3, 0.1])
        z_q, indices, vq_loss = vq.forward(z_e)
        self.assertEqual(z_q.shape, (3,))
        self.assertEqual(len(indices), 1)
        self.assertIsInstance(vq_loss, float)

    def test_forward_batch(self):
        vq = VectorQuantizer(8, 3, rng=_RNG(42))
        z_e = Tensor([[0.5, -0.3, 0.1], [0.2, 0.4, -0.1]])
        z_q, indices, vq_loss = vq.forward(z_e)
        self.assertEqual(z_q.shape, (2, 3))
        self.assertEqual(len(indices), 2)

    def test_quantized_is_codebook_entry(self):
        vq = VectorQuantizer(8, 3, rng=_RNG(42))
        z_e = Tensor([0.5, -0.3, 0.1])
        z_q, indices, _ = vq.forward(z_e)
        # z_q should be the codebook entry
        idx = indices[0]
        for i in range(3):
            self.assertAlmostEqual(z_q.data[i], vq.embeddings.data[idx][i])

    def test_backward_straight_through(self):
        vq = VectorQuantizer(8, 3, rng=_RNG(42))
        z_e = Tensor([0.5, -0.3, 0.1])
        vq.forward(z_e)
        grad = Tensor([1.0, 1.0, 1.0])
        out = vq.backward(grad)
        # Straight-through: gradient passes through unchanged
        self.assertEqual(out.data, [1.0, 1.0, 1.0])

    def test_codebook_usage(self):
        vq = VectorQuantizer(8, 3, rng=_RNG(42))
        vq.reset_usage()
        z_e = Tensor([0.5, -0.3, 0.1])
        vq.forward(z_e)
        usage = vq.get_codebook_usage()
        self.assertGreater(usage, 0.0)
        self.assertLessEqual(usage, 1.0)

    def test_reset_usage(self):
        vq = VectorQuantizer(8, 3, rng=_RNG(42))
        z_e = Tensor([0.5, -0.3, 0.1])
        vq.forward(z_e)
        vq.reset_usage()
        self.assertEqual(sum(vq._usage_count), 0)

    def test_commitment_cost(self):
        vq1 = VectorQuantizer(8, 3, commitment_cost=0.1, rng=_RNG(42))
        vq2 = VectorQuantizer(8, 3, commitment_cost=1.0, rng=_RNG(42))
        z_e = Tensor([0.5, -0.3, 0.1])
        _, _, loss1 = vq1.forward(z_e)
        _, _, loss2 = vq2.forward(z_e)
        # Higher commitment cost -> higher VQ loss
        self.assertGreater(loss2, loss1)


class TestVQVAE(unittest.TestCase):
    def test_creation(self):
        vqvae = VQVAE(8, [16], 4, num_embeddings=32, seed=42)
        self.assertEqual(vqvae.input_dim, 8)
        self.assertEqual(vqvae.latent_dim, 4)

    def test_forward(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=16, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, z_e, z_q, indices, vq_loss = vqvae.forward(x)
        self.assertEqual(x_recon.shape, (4,))
        self.assertEqual(z_e.shape, (2,))
        self.assertEqual(z_q.shape, (2,))
        self.assertIsInstance(vq_loss, float)

    def test_forward_batch(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=16, seed=42)
        x = Tensor([[0.5, 0.3, 0.7, 0.2], [0.1, 0.9, 0.4, 0.6]])
        x_recon, z_e, z_q, indices, vq_loss = vqvae.forward(x)
        self.assertEqual(x_recon.shape, (2, 4))

    def test_compute_loss(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=16, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, z_e, z_q, indices, vq_loss = vqvae.forward(x)
        total, recon, vq = vqvae.compute_loss(x_recon, x, vq_loss)
        self.assertIsInstance(total, float)
        self.assertGreaterEqual(recon, 0.0)

    def test_backward(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=16, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, z_e, z_q, indices, vq_loss = vqvae.forward(x)
        vqvae.backward(x_recon, x)

    def test_train_step(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=16, seed=42)
        opt = Adam(lr=0.01)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        vqvae.train()
        x_recon, z_e, z_q, indices, vq_loss = vqvae.forward(x)
        vqvae.backward(x_recon, x)
        opt.step(vqvae.get_trainable_layers())

    def test_codebook_usage(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=16, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        vqvae.forward(x)
        usage = vqvae.get_codebook_usage()
        self.assertGreater(usage, 0.0)

    def test_train_eval(self):
        vqvae = VQVAE(4, [8], 2, seed=42)
        vqvae.train()
        vqvae.eval()


class TestLatentSpace(unittest.TestCase):
    def setUp(self):
        self.vae = VAE(4, [8], 2, seed=42)
        self.ls = LatentSpace(self.vae)

    def test_encode_to_latent(self):
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        mu = self.ls.encode_to_latent(x)
        self.assertEqual(mu.shape, (2,))

    def test_interpolate(self):
        x1 = Tensor([0.5, 0.3, 0.7, 0.2])
        x2 = Tensor([0.2, 0.8, 0.1, 0.9])
        results = self.ls.interpolate(x1, x2, n_steps=5)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(r.shape, (4,))

    def test_interpolate_endpoints(self):
        x1 = Tensor([0.5, 0.3, 0.7, 0.2])
        x2 = Tensor([0.2, 0.8, 0.1, 0.9])
        results = self.ls.interpolate(x1, x2, n_steps=3)
        # First and last should decode from mu1 and mu2 respectively
        self.assertEqual(len(results), 3)

    def test_spherical_interpolate(self):
        x1 = Tensor([0.5, 0.3, 0.7, 0.2])
        x2 = Tensor([0.2, 0.8, 0.1, 0.9])
        results = self.ls.spherical_interpolate(x1, x2, n_steps=5)
        self.assertEqual(len(results), 5)

    def test_traverse_dimension(self):
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        results = self.ls.traverse_dimension(x, dim=0)
        self.assertEqual(len(results), 7)  # Default 7 values

    def test_traverse_custom_values(self):
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        results = self.ls.traverse_dimension(x, dim=0, values=[-1.0, 0.0, 1.0])
        self.assertEqual(len(results), 3)

    def test_compute_latent_stats(self):
        data = [Tensor([0.5, 0.3, 0.7, 0.2]), Tensor([0.2, 0.8, 0.1, 0.9])]
        stats = self.ls.compute_latent_stats(data)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('active_dims', stats)
        self.assertIn('mean_kl_per_dim', stats)
        self.assertIn('total_mean_kl', stats)
        self.assertEqual(len(stats['mean']), 2)

    def test_compute_latent_stats_empty(self):
        stats = self.ls.compute_latent_stats([])
        self.assertEqual(stats['active_dims'], 0)

    def test_nearest_neighbors(self):
        data = [Tensor([0.5, 0.3, 0.7, 0.2]),
                Tensor([0.4, 0.4, 0.6, 0.3]),
                Tensor([0.1, 0.9, 0.1, 0.9])]
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        nn = self.ls.nearest_neighbors(x, data, k=2)
        self.assertEqual(len(nn), 2)
        # First neighbor should be self (index 0)
        self.assertEqual(nn[0][0], 0)


class TestVAETrainer(unittest.TestCase):
    def test_creation(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        self.assertIsNotNone(trainer)

    def test_train_epoch(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        data = make_vae_data(n=10, dim=4, seed=42)
        total, recon, kl = trainer.train_epoch(data, epoch=0)
        self.assertIsInstance(total, float)

    def test_train_multiple_epochs(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        data = make_vae_data(n=10, dim=4, seed=42)
        history = trainer.train(data, epochs=5)
        self.assertEqual(len(history['total_loss']), 5)
        self.assertEqual(len(history['recon_loss']), 5)
        self.assertEqual(len(history['kl_loss']), 5)

    def test_loss_decreases(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        data = make_vae_data(n=20, dim=4, seed=42)
        history = trainer.train(data, epochs=20)
        # Loss should generally decrease
        self.assertLess(history['total_loss'][-1], history['total_loss'][0])

    def test_kl_warmup(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt, kl_warmup_epochs=5)
        data = make_vae_data(n=10, dim=4, seed=42)
        trainer.train(data, epochs=10)
        # Beta should start at 0 and increase
        self.assertAlmostEqual(trainer.history['beta'][0], 0.0)
        self.assertAlmostEqual(trainer.history['beta'][4], 0.8)

    def test_evaluate(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        data = make_vae_data(n=10, dim=4, seed=42)
        total, recon, kl = trainer.evaluate(data)
        self.assertIsInstance(total, float)

    def test_batch_training(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        data = make_vae_data(n=20, dim=4, seed=42)
        total, recon, kl = trainer.train_epoch(data, batch_size=5, epoch=0)
        self.assertIsInstance(total, float)

    def test_tensor_batch_input(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        x = Tensor([[0.5, 0.3, 0.7, 0.2], [0.1, 0.9, 0.4, 0.6]])
        total, recon, kl = trainer.train_epoch(x, epoch=0)
        self.assertIsInstance(total, float)


class TestVQVAETrainer(unittest.TestCase):
    def test_creation(self):
        vqvae = VQVAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VQVAETrainer(vqvae, opt)
        self.assertIsNotNone(trainer)

    def test_train_epoch(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=8, seed=42)
        opt = Adam(lr=0.01)
        trainer = VQVAETrainer(vqvae, opt)
        data = make_vae_data(n=10, dim=4, seed=42)
        total, recon, vq = trainer.train_epoch(data)
        self.assertIsInstance(total, float)

    def test_train_multiple_epochs(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=8, seed=42)
        opt = Adam(lr=0.01)
        trainer = VQVAETrainer(vqvae, opt)
        data = make_vae_data(n=10, dim=4, seed=42)
        history = trainer.train(data, epochs=5)
        self.assertEqual(len(history['total_loss']), 5)
        self.assertEqual(len(history['codebook_usage']), 5)

    def test_codebook_usage_tracked(self):
        vqvae = VQVAE(4, [8], 2, num_embeddings=8, seed=42)
        opt = Adam(lr=0.01)
        trainer = VQVAETrainer(vqvae, opt)
        data = make_vae_data(n=10, dim=4, seed=42)
        trainer.train(data, epochs=3)
        for usage in trainer.history['codebook_usage']:
            self.assertGreater(usage, 0.0)
            self.assertLessEqual(usage, 1.0)


class TestDataGeneration(unittest.TestCase):
    def test_make_vae_data(self):
        data = make_vae_data(n=50, dim=8, n_modes=3, seed=42)
        self.assertEqual(len(data), 50)
        self.assertEqual(data[0].shape, (8,))

    def test_data_in_range(self):
        data = make_vae_data(n=50, dim=4, seed=42)
        for sample in data:
            for v in sample.data:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_make_labeled_data(self):
        samples, labels = make_labeled_data(n=30, dim=4, n_classes=3, seed=42)
        self.assertEqual(len(samples), 30)
        self.assertEqual(len(labels), 30)
        self.assertTrue(all(0 <= l < 3 for l in labels))

    def test_reproducibility(self):
        d1 = make_vae_data(n=10, dim=4, seed=99)
        d2 = make_vae_data(n=10, dim=4, seed=99)
        for i in range(10):
            self.assertEqual(d1[i].data, d2[i].data)


class TestIntegration(unittest.TestCase):
    """Integration tests for full training pipelines."""

    def test_vae_full_pipeline(self):
        # Create data, train, evaluate, sample, reconstruct
        data = make_vae_data(n=30, dim=4, n_modes=2, seed=42)
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        history = trainer.train(data, epochs=10)

        # Evaluate
        total, recon, kl = trainer.evaluate(data)
        self.assertIsInstance(total, float)

        # Sample
        samples = vae.sample(n_samples=3)
        self.assertEqual(samples.shape, (3, 4))

        # Reconstruct
        recon = vae.reconstruct(data[0])
        self.assertEqual(recon.shape, (4,))

    def test_cvae_full_pipeline(self):
        samples, labels = make_labeled_data(n=30, dim=4, n_classes=3, seed=42)
        cvae = ConditionalVAE(4, 3, [8], 2, seed=42)
        opt = Adam(lr=0.01)

        # Manual training
        cvae.train()
        batch = _stack_rows(samples[:10])
        batch_labels = labels[:10]
        x_recon, mu, logvar, z = cvae.forward(batch, batch_labels)
        total, recon, kl = cvae.compute_loss(x_recon, batch, mu, logvar)
        cvae.backward(x_recon, batch, mu, logvar)
        opt.step(cvae.get_trainable_layers())

        # Generate per class
        for cls in range(3):
            gen = cvae.generate(cls, n_samples=2)
            self.assertEqual(gen.shape, (2, 4))

    def test_beta_vae_disentanglement(self):
        data = make_vae_data(n=30, dim=4, seed=42)
        bvae = BetaVAE(4, [8], 2, beta=4.0, beta_schedule='warmup', seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(bvae, opt, kl_warmup_epochs=5)
        history = trainer.train(data, epochs=10)
        self.assertEqual(len(history['total_loss']), 10)

    def test_vqvae_full_pipeline(self):
        data = make_vae_data(n=30, dim=4, seed=42)
        vqvae = VQVAE(4, [8], 2, num_embeddings=8, seed=42)
        opt = Adam(lr=0.01)
        trainer = VQVAETrainer(vqvae, opt)
        history = trainer.train(data, epochs=10)
        self.assertEqual(len(history['total_loss']), 10)

    def test_latent_space_analysis(self):
        data = make_vae_data(n=20, dim=4, seed=42)
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        trainer.train(data, epochs=5)

        ls = LatentSpace(vae)
        stats = ls.compute_latent_stats(data)
        self.assertEqual(len(stats['mean']), 2)
        self.assertIn('active_dims', stats)

    def test_latent_interpolation(self):
        data = make_vae_data(n=20, dim=4, seed=42)
        vae = VAE(4, [8], 2, seed=42)
        ls = LatentSpace(vae)
        interp = ls.interpolate(data[0], data[1], n_steps=5)
        self.assertEqual(len(interp), 5)

    def test_vae_mse_recon(self):
        vae = VAE(4, [8], 2, recon_type='mse', output_activation='none', seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = vae.forward(x)
        total, recon, kl = vae.compute_loss(x_recon, x, mu, logvar)
        self.assertIsInstance(total, float)

    def test_vae_bce_recon(self):
        vae = VAE(4, [8], 2, recon_type='bce', output_activation='sigmoid', seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = vae.forward(x)
        total, recon, kl = vae.compute_loss(x_recon, x, mu, logvar)
        self.assertIsInstance(total, float)

    def test_training_reduces_reconstruction_error(self):
        data = make_vae_data(n=30, dim=4, n_modes=2, seed=42)
        vae = VAE(4, [16, 8], 2, seed=42)
        opt = Adam(lr=0.005)
        trainer = VAETrainer(vae, opt)

        # Evaluate before training
        pre_total, pre_recon, _ = trainer.evaluate(data)

        # Train
        trainer.train(data, epochs=30)

        # Evaluate after
        post_total, post_recon, _ = trainer.evaluate(data)
        self.assertLess(post_recon, pre_recon)

    def test_kl_warmup_schedule(self):
        data = make_vae_data(n=20, dim=4, seed=42)
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt, kl_warmup_epochs=5)
        trainer.train(data, epochs=10)
        # Beta should be 0 at start, 1 at epoch 5+
        self.assertAlmostEqual(trainer.history['beta'][0], 0.0)
        self.assertAlmostEqual(trainer.history['beta'][5], 1.0)

    def test_different_latent_dims(self):
        for latent_dim in [1, 2, 4, 8]:
            vae = VAE(4, [8], latent_dim, seed=42)
            x = Tensor([0.5, 0.3, 0.7, 0.2])
            x_recon, mu, logvar, z = vae.forward(x)
            self.assertEqual(mu.shape, (latent_dim,))
            self.assertEqual(z.shape, (latent_dim,))

    def test_different_architectures(self):
        configs = [
            ([8], 2),
            ([16, 8], 4),
            ([32, 16, 8], 4),
        ]
        for hidden, latent in configs:
            vae = VAE(4, hidden, latent, seed=42)
            x = Tensor([0.5, 0.3, 0.7, 0.2])
            x_recon, mu, logvar, z = vae.forward(x)
            self.assertEqual(x_recon.shape, (4,))


class TestEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        vae = VAE(4, [8], 2, seed=42)
        opt = Adam(lr=0.01)
        trainer = VAETrainer(vae, opt)
        data = [Tensor([0.5, 0.3, 0.7, 0.2])]
        total, recon, kl = trainer.train_epoch(data, epoch=0)
        self.assertIsInstance(total, float)

    def test_single_latent_dim(self):
        vae = VAE(4, [8], 1, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = vae.forward(x)
        self.assertEqual(mu.shape, (1,))

    def test_large_latent_dim(self):
        vae = VAE(4, [8], 16, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = vae.forward(x)
        self.assertEqual(mu.shape, (16,))

    def test_vq_single_embedding(self):
        vq = VectorQuantizer(1, 3, rng=_RNG(42))
        z_e = Tensor([0.5, -0.3, 0.1])
        z_q, indices, _ = vq.forward(z_e)
        self.assertEqual(indices[0], 0)

    def test_beta_vae_zero_beta(self):
        bvae = BetaVAE(4, [8], 2, beta=0.0, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = bvae.forward(x)
        total, recon, kl = bvae.compute_loss(x_recon, x, mu, logvar)
        # With beta=0, total should equal recon
        self.assertAlmostEqual(total, recon, places=5)

    def test_cvae_single_class(self):
        cvae = ConditionalVAE(4, 1, [8], 2, seed=42)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        x_recon, mu, logvar, z = cvae.forward(x, 0)
        self.assertEqual(x_recon.shape, (4,))

    def test_nearest_neighbors_k_larger_than_dataset(self):
        vae = VAE(4, [8], 2, seed=42)
        ls = LatentSpace(vae)
        data = [Tensor([0.5, 0.3, 0.7, 0.2])]
        nn = ls.nearest_neighbors(data[0], data, k=5)
        self.assertEqual(len(nn), 1)  # Only 1 sample exists

    def test_interpolate_same_point(self):
        vae = VAE(4, [8], 2, seed=42)
        ls = LatentSpace(vae)
        x = Tensor([0.5, 0.3, 0.7, 0.2])
        results = ls.interpolate(x, x, n_steps=3)
        self.assertEqual(len(results), 3)

    def test_vq_many_embeddings(self):
        vq = VectorQuantizer(128, 4, rng=_RNG(42))
        z_e = Tensor([0.0, 0.0, 0.0, 0.0])
        z_q, indices, _ = vq.forward(z_e)
        self.assertEqual(z_q.shape, (4,))


# Import Adam for test use
from neural_network import Adam


if __name__ == '__main__':
    unittest.main()
