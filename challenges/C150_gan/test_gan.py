"""
Tests for C150: Generative Adversarial Network (GAN)
"""

import math
import random
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from gan import (
    Tensor, Generator, Discriminator, GAN, WGAN, LSGAN,
    ConditionalGAN, InfoGAN, BiGAN, GANMetrics,
    make_gaussian_mixture, make_line_data, make_grid_data, make_ring_data,
    _random_noise, _concat_tensors, _one_hot, _ones_tensor, _zeros_tensor,
    _apply_sigmoid, _slice_tensor, _sigmoid_scalar,
)


# ============================================================
# Utility function tests
# ============================================================

class TestUtilities:
    def test_random_noise_shape(self):
        z = _random_noise(5, 10, random.Random(42))
        assert z.shape == (5, 10)

    def test_random_noise_different_seeds(self):
        z1 = _random_noise(3, 4, random.Random(1))
        z2 = _random_noise(3, 4, random.Random(2))
        assert z1.data != z2.data

    def test_concat_tensors(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6, 7], [8, 9, 10]])
        c = _concat_tensors(a, b)
        assert c.shape == (2, 5)
        assert c.data[0] == [1, 2, 5, 6, 7]
        assert c.data[1] == [3, 4, 8, 9, 10]

    def test_concat_tensors_mismatch(self):
        a = Tensor([[1, 2]])
        b = Tensor([[3, 4], [5, 6]])
        with pytest.raises(ValueError):
            _concat_tensors(a, b)

    def test_one_hot(self):
        oh = _one_hot([0, 1, 2], 3)
        assert oh.shape == (3, 3)
        assert oh.data[0] == [1.0, 0.0, 0.0]
        assert oh.data[1] == [0.0, 1.0, 0.0]
        assert oh.data[2] == [0.0, 0.0, 1.0]

    def test_ones_tensor(self):
        t = _ones_tensor((2, 3))
        assert t.shape == (2, 3)
        assert all(v == 1.0 for row in t.data for v in row)

    def test_zeros_tensor(self):
        t = _zeros_tensor((2, 3))
        assert t.shape == (2, 3)
        assert all(v == 0.0 for row in t.data for v in row)

    def test_slice_tensor(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        s = _slice_tensor(t, 1, 3)
        assert s.shape == (2, 2)
        assert s.data[0] == [3, 4]

    def test_sigmoid_scalar(self):
        assert abs(_sigmoid_scalar(0) - 0.5) < 1e-6
        assert _sigmoid_scalar(100) > 0.99
        assert _sigmoid_scalar(-100) < 0.01

    def test_apply_sigmoid_2d(self):
        t = Tensor([[0.0, 100.0], [-100.0, 0.0]])
        s = _apply_sigmoid(t)
        assert abs(s.data[0][0] - 0.5) < 1e-6
        assert s.data[0][1] > 0.99
        assert s.data[1][0] < 0.01

    def test_apply_sigmoid_1d(self):
        t = Tensor([0.0, 100.0])
        s = _apply_sigmoid(t)
        assert abs(s.data[0] - 0.5) < 1e-6


# ============================================================
# Discriminator tests
# ============================================================

class TestDiscriminator:
    def test_construction(self):
        d = Discriminator(2, [8, 4], rng=random.Random(42))
        assert d.input_dim == 2

    def test_forward_shape(self):
        d = Discriminator(4, [8], rng=random.Random(42))
        x = Tensor([[0.5, 0.3, 0.7, 0.2], [0.1, 0.9, 0.4, 0.6]])
        out = d.forward(x)
        assert out.shape == (2, 1)

    def test_output_range_sigmoid(self):
        d = Discriminator(2, [4], use_sigmoid=True, rng=random.Random(42))
        x = Tensor([[0.5, 0.5]] * 5)
        out = d.forward(x)
        for row in out.data:
            assert 0.0 <= row[0] <= 1.0

    def test_output_no_sigmoid(self):
        d = Discriminator(2, [4], use_sigmoid=False, rng=random.Random(42))
        x = Tensor([[0.5, 0.5]])
        out = d.forward(x)
        # No sigmoid means output can be anything
        assert out.shape == (1, 1)

    def test_backward(self):
        d = Discriminator(2, [4], rng=random.Random(42))
        x = Tensor([[0.5, 0.3]])
        d.forward(x)
        grad = Tensor([[1.0]])
        out_grad = d.backward(grad)
        assert out_grad.shape == (1, 2)

    def test_clip_weights(self):
        d = Discriminator(2, [4], rng=random.Random(42))
        d.clip_weights(0.01)
        for layer in d.get_trainable_layers():
            for param_tuple in layer.get_params():
                t = param_tuple[0]
                flat = t.flatten().data
                for v in flat:
                    assert -0.01 <= v <= 0.01

    def test_count_params(self):
        d = Discriminator(4, [8, 4], rng=random.Random(42))
        assert d.count_params() > 0

    def test_with_dropout(self):
        d = Discriminator(2, [8], dropout=0.5, rng=random.Random(42))
        x = Tensor([[0.5, 0.3]])
        out = d.forward(x)
        assert out.shape == (1, 1)

    def test_train_eval_modes(self):
        d = Discriminator(2, [4], rng=random.Random(42))
        d.train()
        d.eval()
        x = Tensor([[0.5, 0.3]])
        out = d.forward(x)
        assert out.shape == (1, 1)


# ============================================================
# Generator tests
# ============================================================

class TestGenerator:
    def test_construction(self):
        g = Generator(10, [8, 4], 2, rng=random.Random(42))
        assert g.noise_dim == 10
        assert g.output_dim == 2

    def test_forward_shape(self):
        g = Generator(5, [8], 3, rng=random.Random(42))
        z = _random_noise(4, 5, random.Random(42))
        out = g.forward(z)
        assert out.shape == (4, 3)

    def test_output_range_sigmoid(self):
        g = Generator(5, [8], 2, output_activation='sigmoid', rng=random.Random(42))
        z = _random_noise(3, 5, random.Random(42))
        out = g.forward(z)
        for row in out.data:
            for v in row:
                assert 0.0 <= v <= 1.0

    def test_backward(self):
        g = Generator(5, [8], 2, rng=random.Random(42))
        z = _random_noise(2, 5, random.Random(42))
        g.forward(z)
        grad = Tensor([[1.0, 0.0], [0.0, 1.0]])
        out_grad = g.backward(grad)
        assert out_grad.shape == (2, 5)

    def test_count_params(self):
        g = Generator(10, [16, 8], 4, rng=random.Random(42))
        assert g.count_params() > 0

    def test_tanh_output(self):
        g = Generator(5, [8], 2, output_activation='tanh', rng=random.Random(42))
        z = _random_noise(3, 5, random.Random(42))
        out = g.forward(z)
        for row in out.data:
            for v in row:
                assert -1.0 <= v <= 1.0


# ============================================================
# Vanilla GAN tests
# ============================================================

class TestGAN:
    def test_construction(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        assert gan.data_dim == 2
        assert gan.noise_dim == 5

    def test_generate_shape(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        samples = gan.generate(10)
        assert samples.shape == (10, 2)

    def test_discriminate(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = Tensor([[0.5, 0.5], [0.3, 0.7]])
        scores = gan.discriminate(data)
        assert scores.shape == (2, 1)
        for row in scores.data:
            assert 0.0 <= row[0] <= 1.0

    def test_train_step(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = gan.train_step(batch)
        assert 'd_loss' in metrics
        assert 'g_loss' in metrics
        assert 'd_real' in metrics
        assert 'd_fake' in metrics

    def test_train_step_losses_are_finite(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = gan.train_step(batch)
        assert math.isfinite(metrics['d_loss'])
        assert math.isfinite(metrics['g_loss'])

    def test_fit_returns_history(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        history = gan.fit(data, epochs=3, batch_size=16)
        assert len(history['d_loss']) > 0
        assert len(history['g_loss']) > 0

    def test_training_changes_params(self):
        rng = random.Random(42)
        gan = GAN(2, 5, [8], [8], rng=rng)

        # Get initial params
        g_layers = gan.generator.get_trainable_layers()
        initial_w = g_layers[0].get_params()[0][0].data[0][0]

        data = make_line_data(50, seed=42)
        gan.fit(data, epochs=5, batch_size=16)

        # Params should change
        final_w = g_layers[0].get_params()[0][0].data[0][0]
        assert initial_w != final_w

    def test_label_smoothing(self):
        gan = GAN(2, 5, [8], [8], label_smoothing=0.1, rng=random.Random(42))
        assert gan.label_smoothing == 0.1
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = gan.train_step(batch)
        assert math.isfinite(metrics['d_loss'])

    def test_multiple_disc_updates(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = gan.train_step(batch, n_disc=3)
        assert math.isfinite(metrics['d_loss'])

    def test_sgd_optimizer(self):
        gan = GAN(2, 5, [8], [8], optimizer='sgd', rng=random.Random(42))
        data = make_line_data(20, seed=42)
        gan.fit(data, epochs=2, batch_size=10)

    def test_rmsprop_optimizer(self):
        gan = GAN(2, 5, [8], [8], optimizer='rmsprop', rng=random.Random(42))
        data = make_line_data(20, seed=42)
        gan.fit(data, epochs=2, batch_size=10)

    def test_verbose_training(self, capsys):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        gan.fit(data, epochs=5, batch_size=16, verbose=True)
        captured = capsys.readouterr()
        assert 'Epoch' in captured.out

    def test_history_tracking(self):
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        gan.fit(data, epochs=5, batch_size=25)
        # Multiple batches per epoch = multiple entries
        assert len(gan.history['d_loss']) > 0
        assert len(gan.history['d_real']) > 0
        assert len(gan.history['d_fake']) > 0

    def test_generated_data_in_range(self):
        gan = GAN(2, 5, [8], [8], gen_output='sigmoid', rng=random.Random(42))
        data = make_line_data(50, seed=42)
        gan.fit(data, epochs=5, batch_size=25)
        samples = gan.generate(10)
        for row in samples.data:
            for v in row:
                assert 0.0 <= v <= 1.0


# ============================================================
# WGAN tests
# ============================================================

class TestWGAN:
    def test_construction(self):
        wgan = WGAN(2, 5, [8], [8], rng=random.Random(42))
        assert wgan.data_dim == 2
        assert wgan.clip_value == 0.01
        assert wgan.n_critic == 5

    def test_generate_shape(self):
        wgan = WGAN(2, 5, [8], [8], rng=random.Random(42))
        samples = wgan.generate(5)
        assert samples.shape == (5, 2)

    def test_train_step(self):
        wgan = WGAN(2, 5, [8], [8], n_critic=2, rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = wgan.train_step(batch)
        assert 'w_dist' in metrics
        assert 'g_loss' in metrics
        assert 'c_real' in metrics
        assert 'c_fake' in metrics

    def test_weight_clipping_enforced(self):
        wgan = WGAN(2, 5, [8], [8], clip_value=0.01, n_critic=1, rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        wgan.train_step(batch)

        # Verify clipping
        for layer in wgan.critic.get_trainable_layers():
            for param_tuple in layer.get_params():
                t = param_tuple[0]
                for v in t.flatten().data:
                    assert -0.01 - 1e-10 <= v <= 0.01 + 1e-10

    def test_fit(self):
        wgan = WGAN(2, 5, [8], [8], n_critic=2, rng=random.Random(42))
        data = make_line_data(50, seed=42)
        history = wgan.fit(data, epochs=3, batch_size=16)
        assert len(history['w_dist']) > 0

    def test_wasserstein_distance_finite(self):
        wgan = WGAN(2, 5, [8], [8], n_critic=2, rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = wgan.train_step(batch)
        assert math.isfinite(metrics['w_dist'])

    def test_no_sigmoid_in_critic(self):
        wgan = WGAN(2, 5, [8], [8], rng=random.Random(42))
        assert wgan.critic.use_sigmoid is False

    def test_tanh_generator(self):
        wgan = WGAN(2, 5, [8], [8], gen_output='tanh', rng=random.Random(42))
        samples = wgan.generate(5)
        for row in samples.data:
            for v in row:
                assert -1.0 <= v <= 1.0

    def test_verbose_training(self, capsys):
        wgan = WGAN(2, 5, [8], [8], n_critic=1, rng=random.Random(42))
        data = make_line_data(50, seed=42)
        wgan.fit(data, epochs=3, batch_size=16, verbose=True)
        captured = capsys.readouterr()
        assert 'Epoch' in captured.out


# ============================================================
# LSGAN tests
# ============================================================

class TestLSGAN:
    def test_construction(self):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        assert lsgan.data_dim == 2

    def test_generate_shape(self):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        samples = lsgan.generate(5)
        assert samples.shape == (5, 2)

    def test_train_step(self):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = lsgan.train_step(batch)
        assert 'd_loss' in metrics
        assert 'g_loss' in metrics

    def test_losses_finite(self):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = lsgan.train_step(batch)
        assert math.isfinite(metrics['d_loss'])
        assert math.isfinite(metrics['g_loss'])

    def test_no_sigmoid_in_disc(self):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        assert lsgan.discriminator.use_sigmoid is False

    def test_fit(self):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        history = lsgan.fit(data, epochs=3, batch_size=16)
        assert len(history['d_loss']) > 0
        assert len(history['g_loss']) > 0

    def test_multiple_disc_steps(self):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = lsgan.train_step(batch, n_disc=3)
        assert math.isfinite(metrics['d_loss'])

    def test_verbose(self, capsys):
        lsgan = LSGAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        lsgan.fit(data, epochs=3, batch_size=16, verbose=True)
        captured = capsys.readouterr()
        assert 'Epoch' in captured.out


# ============================================================
# Conditional GAN tests
# ============================================================

class TestConditionalGAN:
    def test_construction(self):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        assert cgan.num_classes == 3

    def test_train_step(self):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        data, labels = make_gaussian_mixture(20, n_modes=3, seed=42)
        labels = [l % 3 for l in labels]
        batch_data = Tensor(data.data[:10])
        batch_labels = labels[:10]
        metrics = cgan.train_step(batch_data, batch_labels)
        assert 'd_loss' in metrics
        assert 'g_loss' in metrics

    def test_losses_finite(self):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        data, labels = make_gaussian_mixture(20, n_modes=3, seed=42)
        labels = [l % 3 for l in labels]
        batch_data = Tensor(data.data[:10])
        batch_labels = labels[:10]
        metrics = cgan.train_step(batch_data, batch_labels)
        assert math.isfinite(metrics['d_loss'])
        assert math.isfinite(metrics['g_loss'])

    def test_conditional_generation(self):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        samples, gen_labels = cgan.generate([0, 1, 2], n_per_class=2)
        assert samples.shape == (6, 2)
        assert gen_labels == [0, 0, 1, 1, 2, 2]

    def test_fit(self):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        data, labels = make_gaussian_mixture(50, n_modes=3, seed=42)
        labels = [l % 3 for l in labels]
        history = cgan.fit(data, labels, epochs=3, batch_size=16)
        assert len(history['d_loss']) > 0

    def test_label_smoothing(self):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], label_smoothing=0.1, rng=random.Random(42))
        data, labels = make_gaussian_mixture(20, n_modes=3, seed=42)
        labels = [l % 3 for l in labels]
        batch_data = Tensor(data.data[:10])
        batch_labels = labels[:10]
        metrics = cgan.train_step(batch_data, batch_labels)
        assert math.isfinite(metrics['d_loss'])

    def test_verbose(self, capsys):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        data, labels = make_gaussian_mixture(50, n_modes=3, seed=42)
        labels = [l % 3 for l in labels]
        cgan.fit(data, labels, epochs=3, batch_size=16, verbose=True)
        captured = capsys.readouterr()
        assert 'Epoch' in captured.out

    def test_generate_single_class(self):
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        samples, gen_labels = cgan.generate([1], n_per_class=5)
        assert samples.shape == (5, 2)
        assert all(l == 1 for l in gen_labels)


# ============================================================
# InfoGAN tests
# ============================================================

class TestInfoGAN:
    def test_construction(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=2,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        assert infogan.n_cat_codes == 1
        assert infogan.cat_dim == 3
        assert infogan.n_cont_codes == 2

    def test_train_step(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=1,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = infogan.train_step(batch)
        assert 'd_loss' in metrics
        assert 'g_loss' in metrics
        assert 'info_loss' in metrics

    def test_losses_finite(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=1,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = infogan.train_step(batch)
        assert math.isfinite(metrics['d_loss'])
        assert math.isfinite(metrics['g_loss'])
        assert math.isfinite(metrics['info_loss'])

    def test_fit(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=1,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        history = infogan.fit(data, epochs=3, batch_size=16)
        assert len(history['d_loss']) > 0
        assert len(history['info_loss']) > 0

    def test_generate_random(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=1,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        samples = infogan.generate(5)
        assert samples.shape == (5, 2)

    def test_generate_with_specific_cat_codes(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=1,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        samples = infogan.generate(3, cat_codes=[[0], [1], [2]])
        assert samples.shape == (3, 2)

    def test_generate_with_specific_cont_codes(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=2,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        samples = infogan.generate(2, cont_codes=[[1.0, -1.0], [0.0, 0.0]])
        assert samples.shape == (2, 2)

    def test_info_weight(self):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=1,
                          gen_hidden=[8], disc_hidden=[8], info_weight=0.5, rng=random.Random(42))
        assert infogan.info_weight == 0.5

    def test_multiple_cat_codes(self):
        infogan = InfoGAN(2, 5, n_cat_codes=2, cat_dim=4, n_cont_codes=0,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = infogan.train_step(batch)
        assert math.isfinite(metrics['info_loss'])

    def test_verbose(self, capsys):
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=3, n_cont_codes=1,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        infogan.fit(data, epochs=3, batch_size=16, verbose=True)
        captured = capsys.readouterr()
        assert 'Epoch' in captured.out


# ============================================================
# BiGAN tests
# ============================================================

class TestBiGAN:
    def test_construction(self):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        assert bigan.data_dim == 2
        assert bigan.noise_dim == 5

    def test_generate_shape(self):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        samples = bigan.generate(5)
        assert samples.shape == (5, 2)

    def test_encode_shape(self):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        data = Tensor([[0.5, 0.3], [0.7, 0.2]])
        encoded = bigan.encode(data)
        assert encoded.shape == (2, 5)

    def test_train_step(self):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = bigan.train_step(batch)
        assert 'd_loss' in metrics
        assert 'ge_loss' in metrics

    def test_losses_finite(self):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = bigan.train_step(batch)
        assert math.isfinite(metrics['d_loss'])
        assert math.isfinite(metrics['ge_loss'])

    def test_fit(self):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        history = bigan.fit(data, epochs=3, batch_size=16)
        assert len(history['d_loss']) > 0
        assert len(history['ge_loss']) > 0

    def test_encode_decode_shapes(self):
        bigan = BiGAN(4, 3, [8], [8], [8], rng=random.Random(42))
        x = Tensor([[0.5, 0.3, 0.7, 0.2]])
        z = bigan.encode(x)
        assert z.shape == (1, 3)
        samples = bigan.generate(1)
        assert samples.shape == (1, 4)

    def test_verbose(self, capsys):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        data = make_line_data(50, seed=42)
        bigan.fit(data, epochs=3, batch_size=16, verbose=True)
        captured = capsys.readouterr()
        assert 'Epoch' in captured.out

    def test_multiple_disc_steps(self):
        bigan = BiGAN(2, 5, [8], [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = bigan.train_step(batch, n_disc=3)
        assert math.isfinite(metrics['d_loss'])


# ============================================================
# GANMetrics tests
# ============================================================

class TestGANMetrics:
    def test_mean_std_2d(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        means, stds = GANMetrics.mean_std(t)
        assert len(means) == 2
        assert abs(means[0] - 3.0) < 1e-6
        assert abs(means[1] - 4.0) < 1e-6

    def test_mean_std_1d(self):
        t = Tensor([1, 2, 3, 4, 5])
        means, stds = GANMetrics.mean_std(t)
        assert abs(means[0] - 3.0) < 1e-6

    def test_distribution_distance_same(self):
        t = Tensor([[1, 2], [3, 4]])
        dist = GANMetrics.distribution_distance(t, t)
        assert dist < 1e-6

    def test_distribution_distance_different(self):
        a = Tensor([[0, 0], [0, 0]])
        b = Tensor([[10, 10], [10, 10]])
        dist = GANMetrics.distribution_distance(a, b)
        assert dist > 0

    def test_mode_coverage_same(self):
        data = make_gaussian_mixture(100, seed=42)[0]
        coverage = GANMetrics.mode_coverage(data, data)
        assert coverage == 1.0

    def test_mode_coverage_partial(self):
        real = Tensor([[0.1, 0.1], [0.9, 0.9]] * 50)
        fake = Tensor([[0.1, 0.1]] * 100)  # Only covers one mode
        coverage = GANMetrics.mode_coverage(real, fake)
        assert 0.0 < coverage < 1.0

    def test_discriminator_scores(self):
        d = Discriminator(2, [4], rng=random.Random(42))
        real = Tensor([[0.5, 0.5]] * 5)
        fake = Tensor([[0.1, 0.9]] * 5)
        stats = GANMetrics.discriminator_score_stats(d, real, fake)
        assert 'real_mean' in stats
        assert 'fake_mean' in stats
        assert 'real_min' in stats
        assert 'fake_max' in stats


# ============================================================
# Data generator tests
# ============================================================

class TestDataGenerators:
    def test_gaussian_mixture_shape(self):
        data, labels = make_gaussian_mixture(100, n_modes=4)
        assert data.shape == (100, 2)
        assert len(labels) == 100

    def test_gaussian_mixture_modes(self):
        data, labels = make_gaussian_mixture(100, n_modes=4)
        assert set(labels) == {0, 1, 2, 3}

    def test_line_data_shape(self):
        data = make_line_data(50)
        assert data.shape == (50, 2)

    def test_line_data_correlation(self):
        data = make_line_data(100, noise=0.01)
        # x and y should be highly correlated
        xs = [data.data[i][0] for i in range(100)]
        ys = [data.data[i][1] for i in range(100)]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / len(xs)
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        corr = cov / math.sqrt(var_x * var_y + 1e-10)
        assert corr > 0.9

    def test_grid_data_shape(self):
        data = make_grid_data(100, grid_size=3)
        assert data.shape == (100, 2)

    def test_ring_data_shape(self):
        data = make_ring_data(100)
        assert data.shape == (100, 2)

    def test_ring_data_radius(self):
        data = make_ring_data(200, noise=0.01)
        radii = [math.sqrt((data.data[i][0] - 0.5) ** 2 + (data.data[i][1] - 0.5) ** 2)
                 for i in range(200)]
        mean_r = sum(radii) / len(radii)
        assert abs(mean_r - 0.5) < 0.05

    def test_gaussian_mixture_deterministic(self):
        d1, l1 = make_gaussian_mixture(50, seed=42)
        d2, l2 = make_gaussian_mixture(50, seed=42)
        assert d1.data == d2.data
        assert l1 == l2

    def test_line_data_deterministic(self):
        d1 = make_line_data(50, seed=42)
        d2 = make_line_data(50, seed=42)
        assert d1.data == d2.data


# ============================================================
# Integration tests
# ============================================================

class TestIntegration:
    def test_gan_on_gaussian_mixture(self):
        """GAN learns Gaussian mixture -- at least runs without crashing."""
        data, _ = make_gaussian_mixture(100, n_modes=4, seed=42)
        gan = GAN(2, 10, [16], [16], rng=random.Random(42))
        history = gan.fit(data, epochs=10, batch_size=32)
        assert len(history['d_loss']) > 0

        samples = gan.generate(20)
        assert samples.shape == (20, 2)

    def test_wgan_on_line_data(self):
        """WGAN trains on line data."""
        data = make_line_data(100, seed=42)
        wgan = WGAN(2, 10, [16], [16], n_critic=2, rng=random.Random(42))
        history = wgan.fit(data, epochs=5, batch_size=32)
        assert len(history['w_dist']) > 0

    def test_lsgan_on_ring_data(self):
        """LSGAN trains on ring data."""
        data = make_ring_data(100, seed=42)
        lsgan = LSGAN(2, 10, [16], [16], rng=random.Random(42))
        history = lsgan.fit(data, epochs=5, batch_size=32)
        assert len(history['d_loss']) > 0

    def test_conditional_gan_on_mixture(self):
        """cGAN trains on labeled Gaussian mixture."""
        data, labels = make_gaussian_mixture(100, n_modes=3, seed=42)
        labels = [l % 3 for l in labels]
        cgan = ConditionalGAN(2, 10, 3, [16], [16], rng=random.Random(42))
        history = cgan.fit(data, labels, epochs=5, batch_size=32)
        assert len(history['d_loss']) > 0

        # Generate per class
        samples, gen_labels = cgan.generate([0, 1, 2], n_per_class=5)
        assert samples.shape == (15, 2)

    def test_infogan_on_mixture(self):
        """InfoGAN trains on mixture data."""
        data, _ = make_gaussian_mixture(100, n_modes=4, seed=42)
        infogan = InfoGAN(2, 5, n_cat_codes=1, cat_dim=4, n_cont_codes=1,
                          gen_hidden=[16], disc_hidden=[16], rng=random.Random(42))
        history = infogan.fit(data, epochs=5, batch_size=32)
        assert len(history['info_loss']) > 0

    def test_bigan_on_line_data(self):
        """BiGAN trains and provides encoder."""
        data = make_line_data(100, seed=42)
        bigan = BiGAN(2, 5, [16], [16], [16], rng=random.Random(42))
        history = bigan.fit(data, epochs=5, batch_size=32)
        assert len(history['d_loss']) > 0

        # Encoder works
        encoded = bigan.encode(Tensor(data.data[:5]))
        assert encoded.shape == (5, 5)

    def test_metrics_on_trained_gan(self):
        """GANMetrics work on trained GAN output."""
        data = make_line_data(100, seed=42)
        gan = GAN(2, 10, [16], [16], rng=random.Random(42))
        gan.fit(data, epochs=10, batch_size=32)

        fake = gan.generate(100)
        dist = GANMetrics.distribution_distance(data, fake)
        assert math.isfinite(dist)

        coverage = GANMetrics.mode_coverage(data, fake)
        assert 0.0 <= coverage <= 1.0

    def test_all_gans_produce_output(self):
        """All 5 GAN variants produce valid output shapes."""
        rng = random.Random(42)
        data = make_line_data(50, seed=42)

        # GAN
        gan = GAN(2, 5, [8], [8], rng=rng)
        s = gan.generate(3)
        assert s.shape == (3, 2)

        # WGAN
        wgan = WGAN(2, 5, [8], [8], rng=rng)
        s = wgan.generate(3)
        assert s.shape == (3, 2)

        # LSGAN
        lsgan = LSGAN(2, 5, [8], [8], rng=rng)
        s = lsgan.generate(3)
        assert s.shape == (3, 2)

        # InfoGAN
        infogan = InfoGAN(2, 5, 1, 3, 1, [8], [8], rng=rng)
        s = infogan.generate(3)
        assert s.shape == (3, 2)

        # BiGAN
        bigan = BiGAN(2, 5, [8], [8], [8], rng=rng)
        s = bigan.generate(3)
        assert s.shape == (3, 2)

    def test_gan_with_larger_dims(self):
        """GAN works with higher-dimensional data."""
        rng = random.Random(42)
        data = Tensor([[rng.gauss(0, 1) for _ in range(10)] for _ in range(50)])
        gan = GAN(10, 8, [16], [16], gen_output='tanh', rng=rng)
        history = gan.fit(data, epochs=3, batch_size=16)
        assert len(history['d_loss']) > 0
        samples = gan.generate(5)
        assert samples.shape == (5, 10)

    def test_conditional_different_classes_different_output(self):
        """cGAN generates different samples for different classes (at least input differs)."""
        cgan = ConditionalGAN(2, 5, 3, [8], [8], rng=random.Random(42))
        data, labels = make_gaussian_mixture(100, n_modes=3, seed=42)
        labels = [l % 3 for l in labels]
        cgan.fit(data, labels, epochs=10, batch_size=32)

        # Generate from each class
        s0, _ = cgan.generate([0], n_per_class=10)
        s1, _ = cgan.generate([1], n_per_class=10)
        # Means should differ (even slightly, since conditioning changes input)
        m0 = [sum(s0.data[i][j] for i in range(10)) / 10 for j in range(2)]
        m1 = [sum(s1.data[i][j] for i in range(10)) / 10 for j in range(2)]
        # They should be different (not identical)
        diff = sum((m0[j] - m1[j]) ** 2 for j in range(2))
        assert diff > 0  # different conditioning produces different means

    def test_grid_data_modes(self):
        """Grid data has the right number of modes."""
        data = make_grid_data(900, grid_size=3, std=0.01)
        # Check that we have data near each of the 9 grid points
        centers = set()
        for row in data.data:
            cx = round(row[0] * 3 - 0.5)
            cy = round(row[1] * 3 - 0.5)
            centers.add((cx, cy))
        assert len(centers) >= 9


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:
    def test_single_sample_batch(self):
        """Training with very small batch still works (batch_size >= 2 enforced)."""
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(5, seed=42)
        history = gan.fit(data, epochs=2, batch_size=2)
        assert len(history['d_loss']) > 0

    def test_zero_noise_dim(self):
        """Edge case: very small noise dim."""
        gan = GAN(2, 1, [4], [4], rng=random.Random(42))
        samples = gan.generate(3)
        assert samples.shape == (3, 2)

    def test_deep_networks(self):
        """GAN with deeper networks."""
        gan = GAN(2, 5, [8, 8, 4], [8, 8, 4], rng=random.Random(42))
        data = make_line_data(30, seed=42)
        gan.fit(data, epochs=2, batch_size=10)
        samples = gan.generate(5)
        assert samples.shape == (5, 2)

    def test_large_batch(self):
        """Batch larger than dataset."""
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(10, seed=42)
        history = gan.fit(data, epochs=3, batch_size=100)
        assert len(history['d_loss']) > 0

    def test_one_epoch(self):
        """Single epoch training."""
        gan = GAN(2, 5, [8], [8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        history = gan.fit(data, epochs=1, batch_size=10)
        assert len(history['d_loss']) > 0

    def test_infogan_no_cont_codes(self):
        """InfoGAN with only categorical codes."""
        infogan = InfoGAN(2, 5, n_cat_codes=2, cat_dim=3, n_cont_codes=0,
                          gen_hidden=[8], disc_hidden=[8], rng=random.Random(42))
        data = make_line_data(20, seed=42)
        batch = Tensor(data.data[:10])
        metrics = infogan.train_step(batch)
        assert math.isfinite(metrics['info_loss'])

    def test_dropout_in_gan(self):
        """GAN with dropout in both G and D."""
        gan = GAN(2, 5, [8], [8], dropout=0.3, rng=random.Random(42))
        data = make_line_data(30, seed=42)
        gan.fit(data, epochs=3, batch_size=10)
        samples = gan.generate(5)
        assert samples.shape == (5, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
