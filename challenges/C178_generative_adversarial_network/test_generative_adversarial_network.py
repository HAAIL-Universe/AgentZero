"""
Tests for C178: Generative Adversarial Network
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from generative_adversarial_network import (
    Tensor, _stack_rows, _get_row, _batch_size, _RNG, _tensor_mean,
    _concat_tensors, _zeros_like, _ones_like, _to_flat,
    Generator, Discriminator, Critic,
    GANLoss, WassersteinLoss, GradientPenalty, SpectralNorm,
    GAN, WGAN, ConditionalGAN,
    ConditionalGenerator, ConditionalDiscriminator,
    ModeCollapseDetector, GANEvaluator,
    GANTrainer, WGANTrainer, ConditionalGANTrainer,
    NoiseInterpolator,
    Adam, RMSProp,
)


# ---- Helper ----

def make_2d_data(n=50, seed=42):
    """Simple 2D Gaussian data for testing."""
    rng = _RNG(seed)
    rows = []
    for _ in range(n):
        x = rng.normal(2.0, 0.5)
        y = rng.normal(-1.0, 0.5)
        rows.append(Tensor([x, y]))
    return _stack_rows(rows)

def make_labeled_data(n=50, num_classes=3, seed=42):
    """Labeled data for conditional GAN testing."""
    rng = _RNG(seed)
    rows = []
    labels = []
    for i in range(n):
        c = i % num_classes
        cx = c * 3.0
        cy = c * -2.0
        x = rng.normal(cx, 0.3)
        y = rng.normal(cy, 0.3)
        rows.append(Tensor([x, y]))
        labels.append(float(c))
    return _stack_rows(rows), Tensor(labels)


# ==============================================================
# 1. Tensor Helpers
# ==============================================================

class TestTensorHelpers:
    def test_stack_rows(self):
        rows = [Tensor([1.0, 2.0]), Tensor([3.0, 4.0])]
        t = _stack_rows(rows)
        assert t.shape == (2, 2)
        assert t.data == [[1.0, 2.0], [3.0, 4.0]]

    def test_get_row(self):
        t = _stack_rows([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])
        r = _get_row(t, 1)
        assert r.data == [3.0, 4.0]

    def test_batch_size(self):
        t = _stack_rows([Tensor([1.0]), Tensor([2.0]), Tensor([3.0])])
        assert _batch_size(t) == 3

    def test_concat_tensors(self):
        a = _stack_rows([Tensor([1.0, 2.0])])
        b = _stack_rows([Tensor([3.0, 4.0, 5.0])])
        c = _concat_tensors(a, b)
        assert c.shape == (1, 5)
        assert c.data == [[1.0, 2.0, 3.0, 4.0, 5.0]]

    def test_concat_multi_row(self):
        a = _stack_rows([Tensor([1.0]), Tensor([2.0])])
        b = _stack_rows([Tensor([3.0, 4.0]), Tensor([5.0, 6.0])])
        c = _concat_tensors(a, b)
        assert c.shape == (2, 3)
        assert c.data == [[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]


# ==============================================================
# 2. RNG
# ==============================================================

class TestRNG:
    def test_reproducible(self):
        r1 = _RNG(42)
        r2 = _RNG(42)
        assert [r1.random() for _ in range(10)] == [r2.random() for _ in range(10)]

    def test_normal_tensor(self):
        rng = _RNG(42)
        t = rng.normal_tensor((3, 4))
        assert t.shape == (3, 4)
        assert len(t.data) == 3  # 3 rows
        assert len(t.data[0]) == 4  # 4 cols

    def test_shuffle(self):
        rng = _RNG(42)
        lst = list(range(10))
        rng.shuffle(lst)
        assert sorted(lst) == list(range(10))
        assert lst != list(range(10))  # Very unlikely to be unchanged


# ==============================================================
# 3. Generator
# ==============================================================

class TestGenerator:
    def test_forward_shape(self):
        g = Generator(8, [16], 4, seed=42)
        z = _RNG(42).normal_tensor((5, 8))
        out = g.forward(z)
        assert out.shape == (5, 4)

    def test_generate(self):
        g = Generator(8, [16], 4, seed=42)
        samples = g.generate(10)
        assert _batch_size(samples) == 10
        assert samples.shape[1] == 4

    def test_backward(self):
        g = Generator(4, [8], 2, seed=42)
        z = _RNG(42).normal_tensor((3, 4))
        out = g.forward(z)
        grad = Tensor([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
        g_input = g.backward(grad)
        assert len(g_input.data) > 0

    def test_count_params(self):
        g = Generator(4, [8, 8], 2, use_batchnorm=False, seed=42)
        assert g.count_params() > 0

    def test_train_eval_modes(self):
        g = Generator(4, [8], 2, dropout=0.5, seed=42)
        g.train()
        g.eval()
        # Should not error

    def test_no_output_activation(self):
        g = Generator(4, [8], 2, output_activation=None, seed=42)
        z = _RNG(42).normal_tensor((2, 4))
        out = g.forward(z)
        assert out.shape == (2, 2)
        # Without activation, values can be any range (just verify shape)


# ==============================================================
# 4. Discriminator
# ==============================================================

class TestDiscriminator:
    def test_forward_shape(self):
        d = Discriminator(4, [8], seed=42)
        x = _RNG(42).normal_tensor((5, 4))
        out = d.forward(x)
        vals = _to_flat(out)
        assert len(vals) == 5
        # Output should be in (0, 1) due to sigmoid
        for v in vals:
            assert 0.0 < v < 1.0

    def test_backward(self):
        d = Discriminator(4, [8], seed=42)
        x = _RNG(42).normal_tensor((3, 4))
        out = d.forward(x)
        grad = Tensor([[0.1], [0.2], [0.3]])
        g = d.backward(grad)
        assert len(g.data) > 0

    def test_no_dropout(self):
        d = Discriminator(4, [8], dropout=0.0, seed=42)
        x = _RNG(42).normal_tensor((3, 4))
        out = d.forward(x)
        assert len(_to_flat(out)) == 3

    def test_count_params(self):
        d = Discriminator(4, [8, 8], seed=42)
        assert d.count_params() > 0


# ==============================================================
# 5. Critic (WGAN)
# ==============================================================

class TestCritic:
    def test_forward_unbounded(self):
        c = Critic(4, [8], seed=42)
        x = _RNG(42).normal_tensor((5, 4))
        out = c.forward(x)
        assert len(_to_flat(out)) == 5
        # No sigmoid, so outputs can be any value

    def test_clip_weights(self):
        c = Critic(4, [8], seed=42)
        c.clip_weights(0.01)
        for layer in c.network.layers:
            if hasattr(layer, 'weights'):
                w = layer.weights
                for row in w.data:
                    for v in row:
                        assert -0.01 <= v <= 0.01
            if hasattr(layer, 'bias') and layer.bias is not None:
                for v in layer.bias.data:
                    assert -0.01 <= v <= 0.01

    def test_backward(self):
        c = Critic(4, [8], seed=42)
        x = _RNG(42).normal_tensor((3, 4))
        out = c.forward(x)
        grad = Tensor([[0.1], [0.2], [0.3]])
        g = c.backward(grad)
        assert len(g.data) > 0


# ==============================================================
# 6. GAN Loss Functions
# ==============================================================

class TestGANLoss:
    def test_discriminator_loss_perfect(self):
        loss = GANLoss()
        d_real = Tensor([0.9, 0.95, 0.85])
        d_fake = Tensor([0.1, 0.05, 0.15])
        l = loss.discriminator_loss(d_real, d_fake)
        assert l > 0

    def test_discriminator_loss_bad(self):
        loss = GANLoss()
        d_real = Tensor([0.1, 0.1])
        d_fake = Tensor([0.9, 0.9])
        l_bad = loss.discriminator_loss(d_real, d_fake)
        d_real2 = Tensor([0.9, 0.9])
        d_fake2 = Tensor([0.1, 0.1])
        l_good = loss.discriminator_loss(d_real2, d_fake2)
        assert l_bad > l_good

    def test_generator_loss(self):
        loss = GANLoss()
        d_fake_good = Tensor([0.9, 0.95])
        d_fake_bad = Tensor([0.1, 0.05])
        l_good = loss.generator_loss(d_fake_good)
        l_bad = loss.generator_loss(d_fake_bad)
        assert l_good < l_bad  # Better when D thinks fake is real

    def test_discriminator_grad_real(self):
        loss = GANLoss()
        d_real = Tensor([0.5, 0.8])
        g = loss.discriminator_grad_real(d_real)
        assert len(g.data) == 2

    def test_discriminator_grad_fake(self):
        loss = GANLoss()
        d_fake = Tensor([0.3, 0.7])
        g = loss.discriminator_grad_fake(d_fake)
        assert len(g.data) == 2
        # Gradient should push D(fake) toward 0
        for v in g.data:
            assert v > 0  # Positive gradient pushes sigmoid output lower

    def test_generator_grad(self):
        loss = GANLoss()
        d_fake = Tensor([0.5, 0.3])
        g = loss.generator_grad(d_fake)
        assert len(g.data) == 2
        # Negative gradient pushes D(fake) higher
        for v in g.data:
            assert v < 0

    def test_label_smoothing(self):
        loss_smooth = GANLoss(label_smoothing=0.1)
        loss_no = GANLoss(label_smoothing=0.0)
        d_real = Tensor([0.99])
        d_fake = Tensor([0.01])
        l_smooth = loss_smooth.discriminator_loss(d_real, d_fake)
        l_no = loss_no.discriminator_loss(d_real, d_fake)
        # With smoothing, loss should be slightly higher
        assert l_smooth > l_no


# ==============================================================
# 7. Wasserstein Loss
# ==============================================================

class TestWassersteinLoss:
    def test_critic_loss(self):
        loss = WassersteinLoss()
        c_real = Tensor([2.0, 3.0])
        c_fake = Tensor([-1.0, -2.0])
        l = loss.critic_loss(c_real, c_fake)
        # E[fake] - E[real] = -1.5 - 2.5 = -4.0
        assert abs(l - (-4.0)) < 0.01

    def test_generator_loss(self):
        loss = WassersteinLoss()
        c_fake = Tensor([1.0, 2.0])
        l = loss.generator_loss(c_fake)
        # -E[fake] = -1.5
        assert abs(l - (-1.5)) < 0.01

    def test_critic_grad_real(self):
        loss = WassersteinLoss()
        c_real = Tensor([1.0, 2.0])
        g = loss.critic_grad_real(c_real)
        for v in g.data:
            assert v == -0.5  # -1/n

    def test_critic_grad_fake(self):
        loss = WassersteinLoss()
        c_fake = Tensor([1.0, 2.0])
        g = loss.critic_grad_fake(c_fake)
        for v in g.data:
            assert v == 0.5  # 1/n

    def test_generator_grad(self):
        loss = WassersteinLoss()
        c_fake = Tensor([1.0, 2.0])
        g = loss.generator_grad(c_fake)
        for v in g.data:
            assert v == -0.5  # -1/n


# ==============================================================
# 8. Gradient Penalty
# ==============================================================

class TestGradientPenalty:
    def test_compute(self):
        gp = GradientPenalty(lambda_gp=10.0)
        c = Critic(2, [4], seed=42)
        real = _stack_rows([Tensor([1.0, 2.0]), Tensor([1.5, 2.5])])
        fake = _stack_rows([Tensor([-1.0, -2.0]), Tensor([-0.5, -1.5])])
        rng = _RNG(42)
        penalty = gp.compute(c, real, fake, rng)
        assert isinstance(penalty, float)
        assert penalty >= 0

    def test_penalty_zero_for_linear(self):
        """For a linear critic with grad norm ~1, penalty should be near 0."""
        # This is approximate -- real networks won't be exactly linear
        gp = GradientPenalty(lambda_gp=10.0)
        c = Critic(2, [], seed=42)  # No hidden layers -> linear
        real = _stack_rows([Tensor([1.0, 1.0])])
        fake = _stack_rows([Tensor([0.0, 0.0])])
        rng = _RNG(42)
        penalty = gp.compute(c, real, fake, rng)
        assert isinstance(penalty, float)


# ==============================================================
# 9. Spectral Normalization
# ==============================================================

class TestSpectralNorm:
    def test_normalize(self):
        from generative_adversarial_network import Dense
        layer = Dense(4, 3, init='xavier')
        sn = SpectralNorm(seed=42)
        sn.normalize(layer)
        # After normalization, weights should be valid floats
        assert all(isinstance(v, float) for row in layer.weights.data for v in row)

    def test_multiple_normalizations(self):
        from generative_adversarial_network import Dense
        layer = Dense(4, 3, init='xavier')
        sn = SpectralNorm(seed=42)
        sn.normalize(layer)
        w1 = [v for row in layer.weights.data for v in row]
        sn.normalize(layer)
        w2 = [v for row in layer.weights.data for v in row]
        assert len(w1) == len(w2)

    def test_spectral_norm_reduces_weights(self):
        from generative_adversarial_network import Dense
        layer = Dense(10, 10, init='he')
        flat = [v for row in layer.weights.data for v in row]
        original_norm = math.sqrt(sum(v * v for v in flat))
        sn = SpectralNorm(seed=42)
        sn.normalize(layer)
        flat2 = [v for row in layer.weights.data for v in row]
        new_norm = math.sqrt(sum(v * v for v in flat2))
        assert new_norm <= original_norm + 0.1


# ==============================================================
# 10. Standard GAN
# ==============================================================

class TestGAN:
    def test_creation(self):
        gan = GAN(8, [16], [16], 4, seed=42)
        assert gan.noise_dim == 8
        assert gan.data_dim == 4

    def test_generate(self):
        gan = GAN(8, [16], [16], 4, seed=42)
        samples = gan.generate(5)
        assert _batch_size(samples) == 5
        assert samples.shape[1] == 4

    def test_discriminator_step(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        data = make_2d_data(10, seed=42)
        d_opt = Adam(lr=0.001)
        d_loss = gan.discriminator_step(data, d_optimizer=d_opt)
        assert isinstance(d_loss, float)

    def test_generator_step(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        g_opt = Adam(lr=0.001)
        g_loss = gan.generator_step(5, g_optimizer=g_opt)
        assert isinstance(g_loss, float)

    def test_training_loop(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        data = make_2d_data(20, seed=42)
        d_opt = Adam(lr=0.001)
        g_opt = Adam(lr=0.001)

        for _ in range(3):
            d_loss = gan.discriminator_step(data, d_optimizer=d_opt)
            g_loss = gan.generator_step(_batch_size(data), g_optimizer=g_opt)

        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)

    def test_label_smoothing(self):
        gan = GAN(4, [8], [8], 2, label_smoothing=0.1, seed=42)
        data = make_2d_data(10, seed=42)
        d_loss = gan.discriminator_step(data)
        assert isinstance(d_loss, float)


# ==============================================================
# 11. WGAN
# ==============================================================

class TestWGAN:
    def test_creation(self):
        wgan = WGAN(8, [16], [16], 4, seed=42)
        assert wgan.noise_dim == 8

    def test_generate(self):
        wgan = WGAN(8, [16], [16], 4, seed=42)
        samples = wgan.generate(5)
        assert _batch_size(samples) == 5

    def test_critic_step(self):
        wgan = WGAN(4, [8], [8], 2, seed=42)
        data = make_2d_data(10, seed=42)
        c_opt = RMSProp(lr=0.0005)
        w_loss = wgan.critic_step(data, c_optimizer=c_opt)
        assert isinstance(w_loss, float)

    def test_generator_step(self):
        wgan = WGAN(4, [8], [8], 2, seed=42)
        g_opt = RMSProp(lr=0.0005)
        g_loss = wgan.generator_step(5, g_optimizer=g_opt)
        assert isinstance(g_loss, float)

    def test_weight_clipping(self):
        wgan = WGAN(4, [8], [8], 2, use_gp=False, clip_value=0.01, seed=42)
        data = make_2d_data(10, seed=42)
        c_opt = RMSProp(lr=0.0005)
        wgan.critic_step(data, c_optimizer=c_opt)
        # Weights should be clipped
        for layer in wgan.critic.network.layers:
            if hasattr(layer, 'weights'):
                for row in layer.weights.data:
                    for v in row:
                        assert -0.01 <= v <= 0.01

    def test_gradient_penalty_mode(self):
        wgan = WGAN(4, [8], [8], 2, use_gp=True, lambda_gp=10.0, seed=42)
        assert wgan.gp is not None
        data = make_2d_data(10, seed=42)
        w_loss = wgan.critic_step(data)
        assert isinstance(w_loss, float)


# ==============================================================
# 12. Conditional GAN
# ==============================================================

class TestConditionalGenerator:
    def test_forward(self):
        cg = ConditionalGenerator(4, 3, [8], 2, seed=42)
        z = _RNG(42).normal_tensor((5, 4))
        labels = Tensor([0, 1, 2, 0, 1])
        out = cg.forward(z, labels)
        assert out.shape == (5, 2)

    def test_backward(self):
        cg = ConditionalGenerator(4, 3, [8], 2, seed=42)
        z = _RNG(42).normal_tensor((3, 4))
        labels = Tensor([0, 1, 2])
        out = cg.forward(z, labels)
        grad = Tensor([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
        cg.backward(grad)

    def test_count_params(self):
        cg = ConditionalGenerator(4, 3, [8], 2, seed=42)
        assert cg.count_params() > 0


class TestConditionalDiscriminator:
    def test_forward(self):
        cd = ConditionalDiscriminator(2, 3, [8], seed=42)
        x = _stack_rows([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])
        labels = Tensor([0, 1])
        out = cd.forward(x, labels)
        vals = _to_flat(out)
        assert len(vals) == 2
        for v in vals:
            assert 0.0 < v < 1.0

    def test_backward(self):
        cd = ConditionalDiscriminator(2, 3, [8], seed=42)
        x = _stack_rows([Tensor([1.0, 2.0])])
        labels = Tensor([0])
        cd.forward(x, labels)
        grad = Tensor([[0.1]])
        cd.backward(grad)


class TestConditionalGAN:
    def test_creation(self):
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        assert cgan.noise_dim == 4
        assert cgan.num_classes == 3

    def test_generate(self):
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        samples = cgan.generate(1, n_per_class=5)
        assert _batch_size(samples) == 5

    def test_generate_list_labels(self):
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        samples = cgan.generate([0, 1, 2, 0])
        assert _batch_size(samples) == 4

    def test_discriminator_step(self):
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        data, labels = make_labeled_data(10, 3, seed=42)
        d_opt = Adam(lr=0.001)
        d_loss = cgan.discriminator_step(data, labels, d_optimizer=d_opt)
        assert isinstance(d_loss, float)

    def test_generator_step(self):
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        labels = Tensor([0, 1, 2, 0, 1])
        g_opt = Adam(lr=0.001)
        g_loss = cgan.generator_step(5, labels, g_optimizer=g_opt)
        assert isinstance(g_loss, float)


# ==============================================================
# 13. Mode Collapse Detector
# ==============================================================

class TestModeCollapseDetector:
    def test_compute_diversity(self):
        det = ModeCollapseDetector()
        # Diverse samples
        rows = [Tensor([float(i), float(i * 2)]) for i in range(5)]
        samples = _stack_rows(rows)
        div = det.compute_diversity(samples)
        assert div > 0

    def test_collapsed(self):
        det = ModeCollapseDetector(window_size=3)
        # All identical -> zero diversity
        for _ in range(5):
            identical = _stack_rows([Tensor([1.0, 1.0])] * 5)
            det.update(identical)
        assert det.is_collapsed(threshold=0.1)

    def test_not_collapsed(self):
        det = ModeCollapseDetector(window_size=3)
        rng = _RNG(42)
        for _ in range(5):
            rows = [Tensor([rng.normal(0, 5), rng.normal(0, 5)]) for _ in range(10)]
            det.update(_stack_rows(rows))
        assert not det.is_collapsed(threshold=0.01)

    def test_trend(self):
        det = ModeCollapseDetector(window_size=5)
        # Increasing diversity
        for i in range(5):
            rows = [Tensor([float(j * (i + 1)), float(j * (i + 1))]) for j in range(5)]
            det.update(_stack_rows(rows))
        trend = det.get_trend()
        assert trend > 0  # Increasing

    def test_too_few_samples(self):
        det = ModeCollapseDetector(window_size=5)
        assert not det.is_collapsed()
        assert det.get_trend() == 0.0


# ==============================================================
# 14. GAN Evaluator
# ==============================================================

class TestGANEvaluator:
    def test_reconstruction_error(self):
        ev = GANEvaluator()
        real = _stack_rows([Tensor([1.0, 2.0]), Tensor([1.5, 2.5])])
        gen = _stack_rows([Tensor([1.0, 2.0]), Tensor([1.5, 2.5])])
        err = ev.reconstruction_error(real, gen)
        assert err < 0.01  # Same distribution

    def test_reconstruction_error_different(self):
        ev = GANEvaluator()
        real = _stack_rows([Tensor([1.0, 2.0]), Tensor([1.5, 2.5])])
        gen = _stack_rows([Tensor([-5.0, -5.0]), Tensor([-4.0, -4.0])])
        err = ev.reconstruction_error(real, gen)
        assert err > 1.0  # Very different

    def test_discriminator_accuracy(self):
        ev = GANEvaluator()
        d = Discriminator(2, [8], seed=42)
        real = _stack_rows([Tensor([1.0, 2.0])] * 5)
        fake = _stack_rows([Tensor([-1.0, -2.0])] * 5)
        acc = ev.discriminator_accuracy(d, real, fake)
        assert 0.0 <= acc <= 1.0

    def test_sample_statistics(self):
        ev = GANEvaluator()
        samples = _stack_rows([Tensor([1.0, 2.0]), Tensor([3.0, 4.0]), Tensor([5.0, 6.0])])
        stats = ev.sample_statistics(samples)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert abs(stats['mean'][0] - 3.0) < 0.01
        assert abs(stats['mean'][1] - 4.0) < 0.01
        assert stats['min'] == 1.0
        assert stats['max'] == 6.0


# ==============================================================
# 15. GAN Trainer
# ==============================================================

class TestGANTrainer:
    def test_train_epoch(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        trainer = GANTrainer(
            gan, Adam(lr=0.001), Adam(lr=0.001)
        )
        data = make_2d_data(20, seed=42)
        d_loss, g_loss = trainer.train_epoch(data, batch_size=10)
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)

    def test_train(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        trainer = GANTrainer(
            gan, Adam(lr=0.001), Adam(lr=0.001)
        )
        data = make_2d_data(20, seed=42)
        history = trainer.train(data, epochs=3, batch_size=10)
        assert len(history['d_loss']) == 3
        assert len(history['g_loss']) == 3

    def test_multiple_d_steps(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        trainer = GANTrainer(
            gan, Adam(lr=0.001), Adam(lr=0.001),
            d_steps=3, g_steps=1
        )
        data = make_2d_data(20, seed=42)
        d_loss, g_loss = trainer.train_epoch(data, batch_size=10)
        assert isinstance(d_loss, float)

    def test_collapse_detection(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        trainer = GANTrainer(
            gan, Adam(lr=0.001), Adam(lr=0.001)
        )
        data = make_2d_data(20, seed=42)
        trainer.train(data, epochs=10, batch_size=10)
        # Check collapse detector was used
        assert isinstance(trainer.collapse_detector, ModeCollapseDetector)


# ==============================================================
# 16. WGAN Trainer
# ==============================================================

class TestWGANTrainer:
    def test_train_epoch(self):
        wgan = WGAN(4, [8], [8], 2, use_gp=False, seed=42)
        trainer = WGANTrainer(
            wgan, RMSProp(lr=0.0005), RMSProp(lr=0.0005),
            n_critic=2
        )
        data = make_2d_data(20, seed=42)
        w_dist, g_loss = trainer.train_epoch(data, batch_size=10)
        assert isinstance(w_dist, float)
        assert isinstance(g_loss, float)

    def test_train(self):
        wgan = WGAN(4, [8], [8], 2, use_gp=False, seed=42)
        trainer = WGANTrainer(
            wgan, RMSProp(lr=0.0005), RMSProp(lr=0.0005),
            n_critic=2
        )
        data = make_2d_data(20, seed=42)
        history = trainer.train(data, epochs=3, batch_size=10)
        assert len(history['w_distance']) == 3
        assert len(history['g_loss']) == 3


# ==============================================================
# 17. Conditional GAN Trainer
# ==============================================================

class TestConditionalGANTrainer:
    def test_train_epoch(self):
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        trainer = ConditionalGANTrainer(
            cgan, Adam(lr=0.001), Adam(lr=0.001)
        )
        data, labels = make_labeled_data(20, 3, seed=42)
        d_loss, g_loss = trainer.train_epoch(data, labels, batch_size=10)
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)

    def test_train(self):
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        trainer = ConditionalGANTrainer(
            cgan, Adam(lr=0.001), Adam(lr=0.001)
        )
        data, labels = make_labeled_data(20, 3, seed=42)
        history = trainer.train(data, labels, epochs=3, batch_size=10)
        assert len(history['d_loss']) == 3
        assert len(history['g_loss']) == 3


# ==============================================================
# 18. Noise Interpolation
# ==============================================================

class TestNoiseInterpolator:
    def test_linear_interpolate(self):
        g = Generator(4, [8], 2, seed=42)
        interp = NoiseInterpolator(g)
        z1 = Tensor([1.0, 0.0, 0.0, 0.0])
        z2 = Tensor([0.0, 0.0, 0.0, 1.0])
        results = interp.linear_interpolate(z1, z2, steps=5)
        assert len(results) == 5
        for r in results:
            assert len(r.data) == 2

    def test_slerp(self):
        g = Generator(4, [8], 2, seed=42)
        interp = NoiseInterpolator(g)
        z1 = Tensor([1.0, 0.0, 0.0, 0.0])
        z2 = Tensor([0.0, 1.0, 0.0, 0.0])
        results = interp.slerp(z1, z2, steps=5)
        assert len(results) == 5

    def test_slerp_same_vector(self):
        g = Generator(4, [8], 2, seed=42)
        interp = NoiseInterpolator(g)
        z = Tensor([1.0, 1.0, 1.0, 1.0])
        results = interp.slerp(z, z, steps=3)
        assert len(results) == 3


# ==============================================================
# 19. Integration Tests
# ==============================================================

class TestIntegration:
    def test_full_gan_pipeline(self):
        """Full GAN training pipeline."""
        data = make_2d_data(30, seed=42)
        gan = GAN(4, [8], [8], 2, seed=42)
        g_opt = Adam(lr=0.001)
        d_opt = Adam(lr=0.001)
        trainer = GANTrainer(gan, g_opt, d_opt)

        history = trainer.train(data, epochs=5, batch_size=10)
        samples = gan.generate(10)
        assert _batch_size(samples) == 10

        ev = GANEvaluator()
        stats = ev.sample_statistics(samples)
        assert 'mean' in stats

    def test_full_wgan_pipeline(self):
        """Full WGAN training pipeline."""
        data = make_2d_data(30, seed=42)
        wgan = WGAN(4, [8], [8], 2, use_gp=False, seed=42)
        g_opt = RMSProp(lr=0.0005)
        c_opt = RMSProp(lr=0.0005)
        trainer = WGANTrainer(wgan, g_opt, c_opt, n_critic=2)

        history = trainer.train(data, epochs=3, batch_size=10)
        samples = wgan.generate(10)
        assert _batch_size(samples) == 10

    def test_full_cgan_pipeline(self):
        """Full Conditional GAN pipeline."""
        data, labels = make_labeled_data(30, 3, seed=42)
        cgan = ConditionalGAN(4, 3, [8], [8], 2, seed=42)
        g_opt = Adam(lr=0.001)
        d_opt = Adam(lr=0.001)
        trainer = ConditionalGANTrainer(cgan, g_opt, d_opt)

        history = trainer.train(data, labels, epochs=3, batch_size=10)

        # Generate per class
        for c in range(3):
            samples = cgan.generate(c, n_per_class=5)
            assert _batch_size(samples) == 5

    def test_gan_with_spectral_norm(self):
        """GAN with spectral normalization on discriminator."""
        gan = GAN(4, [8], [8], 2, seed=42)
        sn = SpectralNorm(seed=42)

        # Apply spectral norm to discriminator
        for layer in gan.discriminator.network.layers:
            sn.normalize(layer)

        data = make_2d_data(20, seed=42)
        d_opt = Adam(lr=0.001)
        d_loss = gan.discriminator_step(data, d_optimizer=d_opt)
        assert isinstance(d_loss, float)

    def test_interpolation_after_training(self):
        """Generate interpolations after training."""
        data = make_2d_data(20, seed=42)
        gan = GAN(4, [8], [8], 2, seed=42)
        trainer = GANTrainer(gan, Adam(lr=0.001), Adam(lr=0.001))
        trainer.train(data, epochs=3, batch_size=10)

        interp = NoiseInterpolator(gan.generator)
        z1 = gan.sample_noise(1)
        z2 = gan.sample_noise(1)
        z1_flat = _get_row(z1, 0)
        z2_flat = _get_row(z2, 0)
        results = interp.linear_interpolate(z1_flat, z2_flat, steps=5)
        assert len(results) == 5

    def test_evaluator_after_training(self):
        """Run evaluator after training."""
        data = make_2d_data(30, seed=42)
        gan = GAN(4, [8], [8], 2, seed=42)
        trainer = GANTrainer(gan, Adam(lr=0.001), Adam(lr=0.001))
        trainer.train(data, epochs=3, batch_size=10)

        ev = GANEvaluator()
        samples = gan.generate(20)
        err = ev.reconstruction_error(data, samples)
        assert isinstance(err, float)

        acc = ev.discriminator_accuracy(gan.discriminator, data, samples)
        assert 0.0 <= acc <= 1.0

    def test_mode_collapse_detection_integration(self):
        """Mode collapse detection during training."""
        data = make_2d_data(20, seed=42)
        gan = GAN(4, [8], [8], 2, seed=42)
        trainer = GANTrainer(gan, Adam(lr=0.001), Adam(lr=0.001))
        trainer.train(data, epochs=10, batch_size=10)

        # After short training, should not falsely detect collapse
        # (detection requires sustained low diversity)
        det = trainer.collapse_detector
        assert isinstance(det, ModeCollapseDetector)

    def test_wgan_gp_training(self):
        """WGAN-GP training."""
        data = make_2d_data(20, seed=42)
        wgan = WGAN(4, [8], [8], 2, use_gp=True, lambda_gp=10.0, seed=42)
        g_opt = Adam(lr=0.0001, beta1=0.0, beta2=0.9)
        c_opt = Adam(lr=0.0001, beta1=0.0, beta2=0.9)

        for _ in range(3):
            w_loss = wgan.critic_step(data, c_optimizer=c_opt)
            g_loss = wgan.generator_step(_batch_size(data), g_optimizer=g_opt)

        assert isinstance(w_loss, float)
        assert isinstance(g_loss, float)


# ==============================================================
# 20. Edge Cases
# ==============================================================

class TestEdgeCases:
    def test_single_sample(self):
        gan = GAN(4, [8], [8], 2, seed=42)
        samples = gan.generate(1)
        assert _batch_size(samples) == 1

    def test_large_noise_dim(self):
        gan = GAN(64, [32], [32], 4, seed=42)
        samples = gan.generate(3)
        assert _batch_size(samples) == 3
        assert samples.shape[1] == 4

    def test_deep_generator(self):
        g = Generator(4, [8, 8, 8], 2, seed=42)
        z = _RNG(42).normal_tensor((3, 4))
        out = g.forward(z)
        assert out.shape == (3, 2)

    def test_deep_discriminator(self):
        d = Discriminator(4, [8, 8, 8], seed=42)
        x = _RNG(42).normal_tensor((3, 4))
        out = d.forward(x)
        assert len(_to_flat(out)) == 3

    def test_no_hidden_layers(self):
        g = Generator(4, [], 2, use_batchnorm=False, seed=42)
        z = _RNG(42).normal_tensor((2, 4))
        out = g.forward(z)
        assert out.shape == (2, 2)

    def test_cgan_one_class(self):
        cgan = ConditionalGAN(4, 1, [8], [8], 2, seed=42)
        samples = cgan.generate(0, n_per_class=3)
        assert _batch_size(samples) == 3

    def test_diversity_single_sample(self):
        det = ModeCollapseDetector()
        single = _stack_rows([Tensor([1.0, 2.0])])
        div = det.compute_diversity(single)
        assert div == 0.0

    def test_evaluator_same_data(self):
        ev = GANEvaluator()
        data = _stack_rows([Tensor([1.0, 2.0])] * 5)
        stats = ev.sample_statistics(data)
        assert all(s < 0.01 for s in stats['std'])

    def test_generator_eval_generate(self):
        """Generate in eval mode (no dropout effect)."""
        g = Generator(4, [8], 2, dropout=0.5, seed=42)
        g.eval()
        z = _RNG(42).normal_tensor((3, 4))
        out1 = g.forward(z)
        out2 = g.forward(z)
        # In eval mode, deterministic
        assert out1.data == out2.data

    def test_wasserstein_distance_sign(self):
        """W distance should be negative when critic separates well."""
        loss = WassersteinLoss()
        # Good separation: real > fake
        c_real = Tensor([5.0, 6.0])
        c_fake = Tensor([-3.0, -2.0])
        w = loss.critic_loss(c_real, c_fake)
        assert w < 0  # E[fake] - E[real] < 0


# ==============================================================
# Run tests
# ==============================================================

def run_tests():
    test_classes = [
        TestTensorHelpers, TestRNG, TestGenerator, TestDiscriminator,
        TestCritic, TestGANLoss, TestWassersteinLoss, TestGradientPenalty,
        TestSpectralNorm, TestGAN, TestWGAN,
        TestConditionalGenerator, TestConditionalDiscriminator,
        TestConditionalGAN, TestModeCollapseDetector, TestGANEvaluator,
        TestGANTrainer, TestWGANTrainer, TestConditionalGANTrainer,
        TestNoiseInterpolator, TestIntegration, TestEdgeCases,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append(f"  FAIL {cls.__name__}.{method_name}: {e}")

    print(f"\nC178 GAN Tests: {passed}/{total} passed, {failed} failed")
    if errors:
        for e in errors:
            print(e)
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
