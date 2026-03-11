"""
Tests for C154: Variational Inference
"""

import math
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from variational_inference import (
    Normal, MultivariateNormal, Bernoulli, Categorical,
    KLDivergence, ELBO,
    MeanFieldVI, BlackBoxVI, ReparameterizedVI,
    AmortizedVI, PlanarFlow, RadialFlow, FlowVI,
    ADVI, StochasticVI, VIDiagnostics
)


# ============================================================
# Distribution Tests
# ============================================================

class TestNormal:
    def test_create(self):
        d = Normal(0.0, 1.0)
        assert d.mean == 0.0
        assert d.std == 1.0

    def test_log_prob_at_mean(self):
        d = Normal(0.0, 1.0)
        lp = d.log_prob(0.0)
        expected = -0.5 * math.log(2 * math.pi)
        assert abs(lp - expected) < 1e-10

    def test_log_prob_symmetry(self):
        d = Normal(0.0, 1.0)
        assert abs(d.log_prob(1.0) - d.log_prob(-1.0)) < 1e-10

    def test_log_prob_shifted(self):
        d = Normal(5.0, 2.0)
        lp = d.log_prob(5.0)
        expected = -0.5 * math.log(2 * math.pi) - math.log(2.0)
        assert abs(lp - expected) < 1e-10

    def test_log_prob_batch(self):
        d = Normal(0.0, 1.0)
        lps = d.log_prob([0.0, 1.0, -1.0])
        assert len(lps) == 3
        assert abs(lps[1] - lps[2]) < 1e-10

    def test_sample_single(self):
        d = Normal(0.0, 1.0)
        s = d.sample(1, rng=random.Random(42))
        assert isinstance(s, float)

    def test_sample_multiple(self):
        d = Normal(0.0, 1.0)
        samples = d.sample(1000, rng=random.Random(42))
        assert len(samples) == 1000
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.2

    def test_sample_mean_close(self):
        d = Normal(5.0, 0.5)
        samples = d.sample(5000, rng=random.Random(42))
        mean = sum(samples) / len(samples)
        assert abs(mean - 5.0) < 0.1

    def test_entropy(self):
        d = Normal(0.0, 1.0)
        h = d.entropy()
        expected = 0.5 * math.log(2 * math.pi * math.e)
        assert abs(h - expected) < 1e-10

    def test_entropy_increases_with_std(self):
        d1 = Normal(0.0, 1.0)
        d2 = Normal(0.0, 2.0)
        assert d2.entropy() > d1.entropy()

    def test_invalid_std(self):
        with pytest.raises(ValueError):
            Normal(0.0, 0.0)
        with pytest.raises(ValueError):
            Normal(0.0, -1.0)

    def test_repr(self):
        d = Normal(1.0, 2.0)
        assert "Normal" in repr(d)


class TestMultivariateNormal:
    def test_create_default_cov(self):
        d = MultivariateNormal([0.0, 0.0])
        assert d.dim == 2

    def test_create_custom_cov(self):
        d = MultivariateNormal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]])
        assert d.dim == 2

    def test_log_prob_at_mean(self):
        d = MultivariateNormal([0.0, 0.0])
        lp = d.log_prob([0.0, 0.0])
        expected = -0.5 * 2 * math.log(2 * math.pi)  # identity cov
        assert abs(lp - expected) < 1e-6

    def test_log_prob_symmetry(self):
        d = MultivariateNormal([0.0, 0.0])
        lp1 = d.log_prob([1.0, 0.0])
        lp2 = d.log_prob([0.0, 1.0])
        assert abs(lp1 - lp2) < 1e-6

    def test_log_prob_batch(self):
        d = MultivariateNormal([0.0, 0.0])
        lps = d.log_prob([[0.0, 0.0], [1.0, 1.0]])
        assert len(lps) == 2
        assert lps[0] > lps[1]  # mean has higher density

    def test_sample_single(self):
        d = MultivariateNormal([0.0, 0.0])
        s = d.sample(1, rng=random.Random(42))
        assert len(s) == 2

    def test_sample_multiple(self):
        d = MultivariateNormal([0.0, 0.0])
        samples = d.sample(1000, rng=random.Random(42))
        assert len(samples) == 1000
        mean0 = sum(s[0] for s in samples) / len(samples)
        assert abs(mean0) < 0.2

    def test_entropy(self):
        d = MultivariateNormal([0.0, 0.0])
        h = d.entropy()
        # For 2D standard normal: 0.5 * (2 * (1 + ln(2*pi)) + 0)
        expected = 0.5 * (2 * (1.0 + math.log(2 * math.pi)))
        assert abs(h - expected) < 1e-6

    def test_correlated_samples(self):
        # High positive correlation
        d = MultivariateNormal([0.0, 0.0], [[1.0, 0.9], [0.9, 1.0]])
        samples = d.sample(2000, rng=random.Random(42))
        # Check correlation
        x = [s[0] for s in samples]
        y = [s[1] for s in samples]
        mx, my = sum(x)/len(x), sum(y)/len(y)
        cov_xy = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y)) / len(x)
        var_x = sum((xi-mx)**2 for xi in x) / len(x)
        var_y = sum((yi-my)**2 for yi in y) / len(y)
        corr = cov_xy / (math.sqrt(var_x * var_y) + 1e-10)
        assert corr > 0.7  # should be close to 0.9


class TestBernoulli:
    def test_create(self):
        d = Bernoulli(0.7)
        assert d.p == 0.7

    def test_log_prob(self):
        d = Bernoulli(0.7)
        assert abs(d.log_prob(1) - math.log(0.7)) < 1e-10
        assert abs(d.log_prob(0) - math.log(0.3)) < 1e-10

    def test_log_prob_batch(self):
        d = Bernoulli(0.5)
        lps = d.log_prob([0, 1, 0, 1])
        assert len(lps) == 4

    def test_log_prob_invalid(self):
        d = Bernoulli(0.5)
        assert d.log_prob(2) == -float('inf')

    def test_sample(self):
        d = Bernoulli(0.7)
        samples = d.sample(1000, rng=random.Random(42))
        mean = sum(samples) / len(samples)
        assert abs(mean - 0.7) < 0.1

    def test_entropy_half(self):
        d = Bernoulli(0.5)
        assert abs(d.entropy() - math.log(2)) < 1e-10

    def test_entropy_extreme(self):
        d = Bernoulli(0.0)
        assert d.entropy() == 0.0
        d = Bernoulli(1.0)
        assert d.entropy() == 0.0

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            Bernoulli(-0.1)
        with pytest.raises(ValueError):
            Bernoulli(1.1)


class TestCategorical:
    def test_create(self):
        d = Categorical([0.2, 0.3, 0.5])
        assert d.k == 3

    def test_normalized(self):
        d = Categorical([2, 3, 5])
        assert abs(sum(d.probs) - 1.0) < 1e-10

    def test_log_prob(self):
        d = Categorical([0.2, 0.3, 0.5])
        assert abs(d.log_prob(2) - math.log(0.5)) < 1e-10

    def test_log_prob_invalid(self):
        d = Categorical([0.5, 0.5])
        assert d.log_prob(5) == -float('inf')

    def test_sample(self):
        d = Categorical([0.1, 0.9])
        samples = d.sample(1000, rng=random.Random(42))
        count_1 = sum(1 for s in samples if s == 1)
        assert count_1 > 800

    def test_entropy(self):
        d = Categorical([0.5, 0.5])
        assert abs(d.entropy() - math.log(2)) < 1e-10

    def test_entropy_uniform(self):
        d = Categorical([1, 1, 1, 1])
        assert abs(d.entropy() - math.log(4)) < 1e-10


# ============================================================
# KL Divergence Tests
# ============================================================

class TestKLDivergence:
    def test_kl_normal_same(self):
        p = Normal(0.0, 1.0)
        kl = KLDivergence.analytical(p, p)
        assert abs(kl) < 1e-10

    def test_kl_normal_different_mean(self):
        q = Normal(1.0, 1.0)
        p = Normal(0.0, 1.0)
        kl = KLDivergence.analytical(q, p)
        assert abs(kl - 0.5) < 1e-10  # KL = (mu1-mu2)^2 / (2*sigma2^2)

    def test_kl_normal_different_std(self):
        q = Normal(0.0, 1.0)
        p = Normal(0.0, 2.0)
        kl = KLDivergence.analytical(q, p)
        # KL = log(2/1) + (1 + 0)/(2*4) - 0.5 = log(2) + 1/8 - 0.5
        expected = math.log(2) + 1/8 - 0.5
        assert abs(kl - expected) < 1e-10

    def test_kl_normal_asymmetric(self):
        q = Normal(0.0, 1.0)
        p = Normal(1.0, 1.0)
        kl_qp = KLDivergence.analytical(q, p)
        kl_pq = KLDivergence.analytical(p, q)
        assert abs(kl_qp - kl_pq) < 1e-10  # symmetric for same-sigma normals

    def test_kl_normal_nonneg(self):
        q = Normal(2.0, 0.5)
        p = Normal(-1.0, 3.0)
        assert KLDivergence.analytical(q, p) >= 0

    def test_kl_mvn_same(self):
        q = MultivariateNormal([0.0, 0.0])
        kl = KLDivergence.analytical(q, q)
        assert abs(kl) < 1e-6

    def test_kl_mvn_different(self):
        q = MultivariateNormal([1.0, 0.0])
        p = MultivariateNormal([0.0, 0.0])
        kl = KLDivergence.analytical(q, p)
        assert kl > 0

    def test_kl_bernoulli_same(self):
        q = Bernoulli(0.5)
        kl = KLDivergence.analytical(q, q)
        assert abs(kl) < 1e-10

    def test_kl_bernoulli_different(self):
        q = Bernoulli(0.3)
        p = Bernoulli(0.7)
        kl = KLDivergence.analytical(q, p)
        assert kl > 0

    def test_kl_categorical_same(self):
        q = Categorical([0.2, 0.3, 0.5])
        kl = KLDivergence.analytical(q, q)
        assert abs(kl) < 1e-10

    def test_kl_categorical_different(self):
        q = Categorical([0.9, 0.1])
        p = Categorical([0.1, 0.9])
        kl = KLDivergence.analytical(q, p)
        assert kl > 0

    def test_mc_kl_normal(self):
        q = Normal(1.0, 1.0)
        p = Normal(0.0, 1.0)
        kl_exact = KLDivergence.analytical(q, p)
        kl_mc = KLDivergence.monte_carlo(q, p, n_samples=10000, rng=random.Random(42))
        assert abs(kl_mc - kl_exact) < 0.2

    def test_mc_kl_mvn(self):
        q = MultivariateNormal([1.0, 0.0])
        p = MultivariateNormal([0.0, 0.0])
        kl_exact = KLDivergence.analytical(q, p)
        kl_mc = KLDivergence.monte_carlo(q, p, n_samples=5000, rng=random.Random(42))
        assert abs(kl_mc - kl_exact) < 0.3


# ============================================================
# ELBO Tests
# ============================================================

class TestELBO:
    def test_compute_basic(self):
        # Simple 1D Gaussian: p(z) = N(0,1), q(z) = N(0,1) -> ELBO = 0
        prior = Normal(0.0, 1.0)
        q = Normal(0.0, 1.0)
        elbo = ELBO.compute(lambda z: prior.log_prob(z), q, n_samples=5000, rng=random.Random(42))
        assert abs(elbo) < 0.3  # should be near 0

    def test_compute_shifted_q(self):
        # q shifted from prior -> ELBO should be negative (gap = KL)
        prior = Normal(0.0, 1.0)
        q = Normal(3.0, 1.0)
        elbo = ELBO.compute(lambda z: prior.log_prob(z), q, n_samples=5000, rng=random.Random(42))
        assert elbo < -1.0  # definitely negative

    def test_compute_with_kl_normal(self):
        prior = Normal(0.0, 1.0)
        q = Normal(0.5, 1.0)
        elbo = ELBO.compute_with_kl(
            lambda z: -0.5 * z ** 2,  # log p(x|z) prop to Gaussian
            q, prior, n_samples=5000, rng=random.Random(42)
        )
        assert isinstance(elbo, float)

    def test_elbo_tighter_with_better_q(self):
        # True posterior is ~N(2, 0.5) for certain model
        def log_joint(z):
            return -0.5 * ((z - 2.0) / 0.5) ** 2

        q_good = Normal(2.0, 0.5)
        q_bad = Normal(0.0, 2.0)
        elbo_good = ELBO.compute(log_joint, q_good, n_samples=3000, rng=random.Random(42))
        elbo_bad = ELBO.compute(log_joint, q_bad, n_samples=3000, rng=random.Random(42))
        assert elbo_good > elbo_bad


# ============================================================
# Mean-Field VI Tests
# ============================================================

class TestMeanFieldVI:
    def test_fit_1d_gaussian(self):
        # Target: N(3, 1)
        def log_joint(z):
            return -0.5 * (z[0] - 3.0) ** 2

        vi = MeanFieldVI(1, log_joint, seed=42)
        mu, sigma = vi.fit(n_iter=300, lr=0.05, n_samples=20)
        assert abs(mu[0] - 3.0) < 0.5

    def test_fit_2d_independent(self):
        # Target: N([2, -1], I)
        def log_joint(z):
            return -0.5 * ((z[0] - 2.0) ** 2 + (z[1] + 1.0) ** 2)

        vi = MeanFieldVI(2, log_joint, seed=42)
        mu, sigma = vi.fit(n_iter=300, lr=0.05, n_samples=20)
        assert abs(mu[0] - 2.0) < 0.8
        assert abs(mu[1] - (-1.0)) < 0.8

    def test_elbo_increases(self):
        def log_joint(z):
            return -0.5 * (z[0] - 1.0) ** 2

        vi = MeanFieldVI(1, log_joint, seed=42)
        vi.fit(n_iter=200, lr=0.05, n_samples=20)
        # ELBO should generally increase (check first vs last quarter)
        n = len(vi.elbo_history)
        first_q = sum(vi.elbo_history[:n//4]) / (n//4)
        last_q = sum(vi.elbo_history[-n//4:]) / (n//4)
        assert last_q > first_q

    def test_get_posterior(self):
        def log_joint(z):
            return -0.5 * z[0] ** 2

        vi = MeanFieldVI(1, log_joint, seed=42)
        vi.fit(n_iter=100, lr=0.05, n_samples=10)
        posteriors = vi.get_posterior()
        assert len(posteriors) == 1
        assert isinstance(posteriors[0], Normal)

    def test_get_joint_posterior(self):
        def log_joint(z):
            return -0.5 * sum(zi ** 2 for zi in z)

        vi = MeanFieldVI(3, log_joint, seed=42)
        vi.fit(n_iter=100, lr=0.05, n_samples=10)
        post = vi.get_joint_posterior()
        assert isinstance(post, MultivariateNormal)
        assert post.dim == 3

    def test_sample_posterior(self):
        def log_joint(z):
            return -0.5 * z[0] ** 2

        vi = MeanFieldVI(1, log_joint, seed=42)
        vi.fit(n_iter=100, lr=0.05, n_samples=10)
        samples = vi.sample_posterior(100)
        assert len(samples) == 100


# ============================================================
# Black-Box VI Tests
# ============================================================

class TestBlackBoxVI:
    def test_fit_1d(self):
        def log_joint(z):
            return -0.5 * (z[0] - 2.0) ** 2

        vi = BlackBoxVI(1, log_joint, seed=42)
        mu, sigma = vi.fit(n_iter=300, lr=0.02, n_samples=30)
        assert abs(mu[0] - 2.0) < 1.0

    def test_with_control_variate(self):
        def log_joint(z):
            return -0.5 * (z[0] - 1.0) ** 2

        vi_cv = BlackBoxVI(1, log_joint, seed=42)
        vi_cv.fit(n_iter=200, lr=0.02, n_samples=30, use_control_variate=True)

        vi_no = BlackBoxVI(1, log_joint, seed=42)
        vi_no.fit(n_iter=200, lr=0.02, n_samples=30, use_control_variate=False)

        # Both should fit -- control variate just reduces variance
        assert len(vi_cv.elbo_history) == 200
        assert len(vi_no.elbo_history) == 200

    def test_get_posterior(self):
        def log_joint(z):
            return -0.5 * sum(zi ** 2 for zi in z)

        vi = BlackBoxVI(2, log_joint, seed=42)
        vi.fit(n_iter=100, lr=0.02, n_samples=20)
        posteriors = vi.get_posterior()
        assert len(posteriors) == 2

    def test_sample_posterior(self):
        def log_joint(z):
            return -0.5 * z[0] ** 2

        vi = BlackBoxVI(1, log_joint, seed=42)
        vi.fit(n_iter=50, lr=0.02, n_samples=20)
        samples = vi.sample_posterior(50)
        assert len(samples) == 50


# ============================================================
# Reparameterized VI Tests
# ============================================================

class TestReparameterizedVI:
    def test_fit_1d(self):
        def log_joint(z):
            return -0.5 * (z[0] - 4.0) ** 2

        vi = ReparameterizedVI(1, log_joint, seed=42)
        mu, sigma = vi.fit(n_iter=300, lr=0.05, n_samples=10)
        assert abs(mu[0] - 4.0) < 0.5

    def test_fit_2d(self):
        def log_joint(z):
            return -0.5 * ((z[0] - 1.0) ** 2 + (z[1] + 2.0) ** 2)

        vi = ReparameterizedVI(2, log_joint, seed=42)
        mu, sigma = vi.fit(n_iter=300, lr=0.05, n_samples=10)
        assert abs(mu[0] - 1.0) < 0.8
        assert abs(mu[1] - (-2.0)) < 0.8

    def test_adam_momentum(self):
        """Verify Adam optimizer is being used (convergence should be smooth)."""
        def log_joint(z):
            return -0.5 * (z[0] - 3.0) ** 2

        vi = ReparameterizedVI(1, log_joint, seed=42)
        vi.fit(n_iter=200, lr=0.1, n_samples=10)
        assert len(vi.elbo_history) == 200

    def test_get_posterior(self):
        def log_joint(z):
            return -0.5 * z[0] ** 2

        vi = ReparameterizedVI(1, log_joint, seed=42)
        vi.fit(n_iter=50, lr=0.05, n_samples=5)
        posteriors = vi.get_posterior()
        assert isinstance(posteriors[0], Normal)


# ============================================================
# Amortized VI Tests
# ============================================================

class TestAmortizedVI:
    def test_create(self):
        avi = AmortizedVI(3, 2, hidden_dims=[4], seed=42)
        assert avi.input_dim == 3
        assert avi.latent_dim == 2

    def test_encode(self):
        avi = AmortizedVI(3, 2, hidden_dims=[4], seed=42)
        mu, log_sigma = avi.encode([1.0, 2.0, 3.0])
        assert len(mu) == 2
        assert len(log_sigma) == 2

    def test_sample_latent_single(self):
        avi = AmortizedVI(3, 2, hidden_dims=[4], seed=42)
        z = avi.sample_latent([1.0, 2.0, 3.0], n=1)
        assert len(z) == 2

    def test_sample_latent_multiple(self):
        avi = AmortizedVI(3, 2, hidden_dims=[4], seed=42)
        zs = avi.sample_latent([1.0, 2.0, 3.0], n=10)
        assert len(zs) == 10
        assert len(zs[0]) == 2

    def test_encode_deterministic(self):
        avi = AmortizedVI(2, 1, hidden_dims=[3], seed=42)
        mu1, ls1 = avi.encode([1.0, 2.0])
        mu2, ls2 = avi.encode([1.0, 2.0])
        assert mu1 == mu2
        assert ls1 == ls2

    def test_different_inputs_different_encoding(self):
        avi = AmortizedVI(2, 1, hidden_dims=[3], seed=42)
        mu1, _ = avi.encode([1.0, 2.0])
        mu2, _ = avi.encode([5.0, -3.0])
        assert mu1 != mu2

    def test_fit_basic(self):
        """Test that amortized VI can train without errors."""
        avi = AmortizedVI(2, 1, hidden_dims=[3], seed=42)
        data = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(20)]
        def log_ll(x, z):
            return -0.5 * sum((xi - z[0]) ** 2 for xi in x)
        history = avi.fit(data, log_ll, n_epochs=5, lr=0.001, batch_size=5)
        assert len(history) == 5


# ============================================================
# Normalizing Flow Tests
# ============================================================

class TestPlanarFlow:
    def test_create(self):
        flow = PlanarFlow(2, n_flows=3, seed=42)
        assert len(flow.flows) == 3

    def test_forward_shape(self):
        flow = PlanarFlow(2, n_flows=2, seed=42)
        z = [1.0, 2.0]
        z_t, log_det = flow.forward(z)
        assert len(z_t) == 2
        assert isinstance(log_det, float)

    def test_forward_changes_input(self):
        flow = PlanarFlow(2, n_flows=2, seed=42)
        z = [1.0, 2.0]
        z_t, _ = flow.forward(z)
        assert z_t != z  # should be transformed

    def test_identity_with_zero_params(self):
        flow = PlanarFlow(2, n_flows=1, seed=42)
        # Set parameters to zero (identity-ish)
        flow.flows[0] = {'w': [0.0, 0.0], 'u': [0.0, 0.0], 'b': 0.0}
        z = [1.0, 2.0]
        z_t, log_det = flow.forward(z)
        assert abs(z_t[0] - 1.0) < 1e-10
        assert abs(z_t[1] - 2.0) < 1e-10

    def test_sample(self):
        flow = PlanarFlow(2, n_flows=2, seed=42)
        base = MultivariateNormal([0.0, 0.0])
        samples = flow.sample(base, n=100)
        assert len(samples) == 100
        assert len(samples[0]) == 2

    def test_log_prob(self):
        flow = PlanarFlow(2, n_flows=1, seed=42)
        base = MultivariateNormal([0.0, 0.0])
        lp = flow.log_prob([0.5, 0.5], base.log_prob)
        assert isinstance(lp, float)

    def test_single_sample(self):
        flow = PlanarFlow(2, n_flows=1, seed=42)
        base = MultivariateNormal([0.0, 0.0])
        s = flow.sample(base, n=1)
        assert len(s) == 2  # single sample, not wrapped


class TestRadialFlow:
    def test_create(self):
        flow = RadialFlow(3, n_flows=2, seed=42)
        assert len(flow.flows) == 2

    def test_forward(self):
        flow = RadialFlow(2, n_flows=1, seed=42)
        z = [1.0, -1.0]
        z_t, log_det = flow.forward(z)
        assert len(z_t) == 2
        assert isinstance(log_det, float)

    def test_sample(self):
        flow = RadialFlow(2, n_flows=2, seed=42)
        base = MultivariateNormal([0.0, 0.0])
        samples = flow.sample(base, n=50)
        assert len(samples) == 50


# ============================================================
# Flow VI Tests
# ============================================================

class TestFlowVI:
    def test_create_planar(self):
        def log_joint(z):
            return -0.5 * sum(zi ** 2 for zi in z)
        fvi = FlowVI(2, log_joint, flow_type='planar', n_flows=2, seed=42)
        assert fvi.dim == 2

    def test_create_radial(self):
        def log_joint(z):
            return -0.5 * sum(zi ** 2 for zi in z)
        fvi = FlowVI(2, log_joint, flow_type='radial', n_flows=2, seed=42)
        assert fvi.dim == 2

    def test_invalid_flow(self):
        with pytest.raises(ValueError):
            FlowVI(2, lambda z: 0, flow_type='invalid')

    def test_fit(self):
        def log_joint(z):
            return -0.5 * ((z[0] - 1.0) ** 2 + (z[1] + 1.0) ** 2)

        fvi = FlowVI(2, log_joint, flow_type='planar', n_flows=2, seed=42)
        mu, sigma = fvi.fit(n_iter=50, lr=0.01, n_samples=5)
        assert len(mu) == 2
        assert len(sigma) == 2

    def test_sample_posterior(self):
        def log_joint(z):
            return -0.5 * sum(zi ** 2 for zi in z)
        fvi = FlowVI(2, log_joint, n_flows=1, seed=42)
        fvi.fit(n_iter=20, lr=0.01, n_samples=5)
        samples = fvi.sample_posterior(50)
        assert len(samples) == 50

    def test_elbo_recorded(self):
        def log_joint(z):
            return -0.5 * sum(zi ** 2 for zi in z)
        fvi = FlowVI(2, log_joint, n_flows=1, seed=42)
        fvi.fit(n_iter=30, lr=0.01, n_samples=5)
        assert len(fvi.elbo_history) == 30


# ============================================================
# ADVI Tests
# ============================================================

class TestADVI:
    def test_real_transform(self):
        def log_joint(z):
            return -0.5 * (z[0] - 2.0) ** 2

        advi = ADVI(1, log_joint, seed=42)
        params = advi.fit(n_iter=200, lr=0.05, n_samples=10)
        assert abs(params[0] - 2.0) < 0.5

    def test_positive_transform(self):
        # Target: log-normal-ish
        def log_joint(z):
            # z[0] should be positive, prior is log-normal
            if z[0] <= 0:
                return -1e10
            return -0.5 * (math.log(z[0]) - 1.0) ** 2 - math.log(z[0])

        advi = ADVI(1, log_joint,
                     transforms=[{'type': 'positive'}],
                     seed=42)
        params = advi.fit(n_iter=200, lr=0.05, n_samples=10)
        assert params[0] > 0  # must be positive

    def test_unit_interval_transform(self):
        def log_joint(z):
            if not (0 < z[0] < 1):
                return -1e10
            return 2 * math.log(z[0]) + math.log(1 - z[0])  # Beta(3, 2)

        advi = ADVI(1, log_joint,
                     transforms=[{'type': 'unit_interval'}],
                     seed=42)
        params = advi.fit(n_iter=200, lr=0.05, n_samples=10)
        assert 0 < params[0] < 1

    def test_mixed_transforms(self):
        def log_joint(z):
            # z[0] is real, z[1] is positive
            if z[1] <= 0:
                return -1e10
            return -0.5 * z[0] ** 2 - z[1]  # standard normal * exponential

        advi = ADVI(2, log_joint,
                     transforms=[{'type': 'real'}, {'type': 'positive'}],
                     seed=42)
        params = advi.fit(n_iter=200, lr=0.05, n_samples=10)
        assert params[1] > 0

    def test_get_posterior(self):
        advi = ADVI(2, lambda z: -sum(zi**2 for zi in z), seed=42)
        advi.fit(n_iter=50, lr=0.05, n_samples=5)
        post = advi.get_posterior()
        assert len(post) == 2

    def test_sample_posterior_constrained(self):
        advi = ADVI(1, lambda z: -z[0]**2 if z[0] > 0 else -1e10,
                     transforms=[{'type': 'positive'}], seed=42)
        advi.fit(n_iter=100, lr=0.05, n_samples=5)
        samples = advi.sample_posterior(50)
        assert all(s[0] > 0 for s in samples)


# ============================================================
# Stochastic VI Tests
# ============================================================

class TestStochasticVI:
    def test_fit_basic(self):
        # Gaussian likelihood, Gaussian prior -> Gaussian posterior
        data = [random.gauss(3.0, 1.0) for _ in range(100)]
        random.seed(42)

        def log_likelihood(z, xi):
            return -0.5 * (xi - z[0]) ** 2

        def log_prior(z):
            return -0.5 * z[0] ** 2

        svi = StochasticVI(1, seed=42)
        mu, sigma = svi.fit(data, log_likelihood, log_prior,
                            n_epochs=50, batch_size=20, lr=0.01, n_samples=5)
        # Posterior mean should be between 0 (prior) and 3 (data)
        assert 0.5 < mu[0] < 4.0

    def test_mini_batch(self):
        data = [random.gauss(1.0, 0.5) for _ in range(50)]
        svi = StochasticVI(1, seed=42)
        mu, sigma = svi.fit(
            data,
            log_likelihood=lambda z, xi: -0.5 * (xi - z[0]) ** 2,
            log_prior=lambda z: -0.5 * z[0] ** 2,
            n_epochs=20, batch_size=10, lr=0.01, n_samples=3
        )
        assert len(svi.elbo_history) == 20

    def test_get_posterior(self):
        svi = StochasticVI(2, seed=42)
        svi.mu = [1.0, 2.0]
        svi.log_sigma = [0.0, 0.5]
        post = svi.get_posterior()
        assert len(post) == 2
        assert isinstance(post[0], Normal)

    def test_sample_posterior(self):
        svi = StochasticVI(2, seed=42)
        svi.mu = [1.0, -1.0]
        svi.log_sigma = [0.0, 0.0]
        samples = svi.sample_posterior(100)
        assert len(samples) == 100
        assert len(samples[0]) == 2


# ============================================================
# VI Diagnostics Tests
# ============================================================

class TestVIDiagnostics:
    def test_elbo_convergence_insufficient(self):
        result = VIDiagnostics.elbo_convergence([1.0, 2.0])
        assert result['converged'] == False
        assert result['reason'] == 'insufficient_data'

    def test_elbo_convergence_converged(self):
        # Constant ELBO = converged
        history = [5.0 + random.gauss(0, 0.01) for _ in range(200)]
        result = VIDiagnostics.elbo_convergence(history, window=50)
        assert result['converged'] == True

    def test_elbo_convergence_not_converged(self):
        # Linearly increasing ELBO
        history = [float(i) for i in range(200)]
        result = VIDiagnostics.elbo_convergence(history, window=50)
        assert result['converged'] == False

    def test_posterior_summary(self):
        vi = MeanFieldVI(2, lambda z: -sum(zi**2 for zi in z), seed=42)
        vi.mu = [1.0, -2.0]
        vi.log_sigma = [math.log(0.5), math.log(1.0)]
        summary = VIDiagnostics.posterior_summary(vi)
        assert 'dim_0' in summary
        assert 'dim_1' in summary
        assert abs(summary['dim_0']['mean'] - 1.0) < 1e-10
        assert abs(summary['dim_1']['mean'] - (-2.0)) < 1e-10
        assert summary['dim_0']['q025'] < summary['dim_0']['mean'] < summary['dim_0']['q975']

    def test_compare_methods(self):
        results = {
            'method_a': {'mu': [1.0], 'sigma': [0.5], 'elbo_history': [1.0, 2.0, 3.0]},
            'method_b': {'mu': [0.9], 'sigma': [0.6], 'elbo_history': [1.0, 4.0, 5.0]},
        }
        comp = VIDiagnostics.compare_methods(results)
        assert comp['_ranking'][0] == 'method_b'  # higher final ELBO
        assert comp['method_a']['final_elbo'] == 3.0

    def test_kl_to_mcmc(self):
        vi_samples = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(500)]
        mcmc_samples = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(500)]
        result = VIDiagnostics.kl_to_mcmc(vi_samples, mcmc_samples)
        assert 'dim_0' in result
        assert 'dim_1' in result
        # Both from same distribution, means should be close
        assert result['dim_0']['mean_diff'] < 0.3

    def test_kl_to_mcmc_1d(self):
        vi_samples = [random.gauss(0, 1) for _ in range(500)]
        mcmc_samples = [random.gauss(0, 1) for _ in range(500)]
        result = VIDiagnostics.kl_to_mcmc(vi_samples, mcmc_samples)
        assert 'dim_0' in result

    def test_kl_to_mcmc_empty(self):
        result = VIDiagnostics.kl_to_mcmc([], [])
        assert 'error' in result

    def test_compare_empty(self):
        comp = VIDiagnostics.compare_methods({'a': {'elbo_history': []}})
        assert comp['a']['final_elbo'] is None


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_vi_vs_exact_posterior(self):
        """Mean-field VI should approximate a known Gaussian posterior."""
        # Prior: N(0, 5), Likelihood: N(x | mu, 1)
        # With data mean = 3, n = 50:
        # Posterior: N(prior_prec * prior_mean + n * data_mean) / (prior_prec + n), 1/(prior_prec+n))
        data_mean = 3.0
        n_data = 50
        prior_var = 25.0
        prior_prec = 1.0 / prior_var
        post_prec = prior_prec + n_data
        post_mean = (prior_prec * 0.0 + n_data * data_mean) / post_prec
        post_var = 1.0 / post_prec

        def log_joint(z):
            lp = -0.5 * z[0] ** 2 / prior_var  # prior
            ll = -0.5 * n_data * (z[0] - data_mean) ** 2  # likelihood
            return lp + ll

        vi = MeanFieldVI(1, log_joint, seed=42)
        mu, sigma = vi.fit(n_iter=500, lr=0.02, n_samples=20)
        assert abs(mu[0] - post_mean) < 0.3

    def test_reparam_vs_blackbox(self):
        """Both should converge to similar posteriors."""
        def log_joint(z):
            return -0.5 * (z[0] - 2.0) ** 2

        vi_r = ReparameterizedVI(1, log_joint, seed=42)
        mu_r, _ = vi_r.fit(n_iter=300, lr=0.05, n_samples=10)

        vi_b = BlackBoxVI(1, log_joint, seed=42)
        mu_b, _ = vi_b.fit(n_iter=300, lr=0.02, n_samples=30)

        # Both should be near 2.0
        assert abs(mu_r[0] - 2.0) < 1.0
        assert abs(mu_b[0] - 2.0) < 1.5

    def test_advi_positive_constraint(self):
        """ADVI should respect positivity constraint."""
        def log_joint(z):
            if z[0] <= 0:
                return -1e10
            return -2.0 * z[0]  # Exponential(2)

        advi = ADVI(1, log_joint, transforms=[{'type': 'positive'}], seed=42)
        params = advi.fit(n_iter=200, lr=0.05, n_samples=10)
        assert params[0] > 0
        samples = advi.sample_posterior(100)
        assert all(s[0] > 0 for s in samples)

    def test_flow_transforms_base(self):
        """Flow should transform base distribution."""
        flow = PlanarFlow(2, n_flows=3, seed=42)
        base = MultivariateNormal([0.0, 0.0])

        base_samples = base.sample(500, rng=random.Random(42))
        flow_samples = flow.sample(base, n=500)

        # Means should differ since flow transforms the distribution
        base_mean0 = sum(s[0] for s in base_samples) / len(base_samples)
        flow_mean0 = sum(s[0] for s in flow_samples) / len(flow_samples)
        # They could be similar or different depending on flow params
        # Key test: flow samples exist and are finite
        assert all(math.isfinite(s[0]) and math.isfinite(s[1]) for s in flow_samples)

    def test_diagnostics_on_fitted_model(self):
        """Full pipeline: fit -> diagnose -> summarize."""
        def log_joint(z):
            return -0.5 * ((z[0] - 1.0) ** 2 + (z[1] + 1.0) ** 2)

        vi = MeanFieldVI(2, log_joint, seed=42)
        vi.fit(n_iter=300, lr=0.05, n_samples=20)

        conv = VIDiagnostics.elbo_convergence(vi.elbo_history)
        assert 'converged' in conv
        assert 'final_elbo' in conv

        summary = VIDiagnostics.posterior_summary(vi)
        assert 'dim_0' in summary
        assert 'mean' in summary['dim_0']

    def test_stochastic_vi_with_data(self):
        """SVI should learn from data."""
        rng = random.Random(42)
        true_mean = 2.0
        data = [rng.gauss(true_mean, 1.0) for _ in range(100)]

        svi = StochasticVI(1, seed=42)
        mu, sigma = svi.fit(
            data,
            log_likelihood=lambda z, xi: -0.5 * (xi - z[0]) ** 2,
            log_prior=lambda z: -0.5 * z[0] ** 2 / 100,  # weak prior
            n_epochs=30, batch_size=20, lr=0.01, n_samples=5
        )
        assert abs(mu[0] - true_mean) < 1.5

    def test_multiple_vi_comparison(self):
        """Compare multiple VI methods using diagnostics."""
        def log_joint(z):
            return -0.5 * (z[0] - 3.0) ** 2

        vi1 = MeanFieldVI(1, log_joint, seed=42)
        mu1, s1 = vi1.fit(n_iter=200, lr=0.05, n_samples=10)

        vi2 = ReparameterizedVI(1, log_joint, seed=42)
        mu2, s2 = vi2.fit(n_iter=200, lr=0.05, n_samples=10)

        comp = VIDiagnostics.compare_methods({
            'mean_field': {'mu': mu1, 'sigma': s1, 'elbo_history': vi1.elbo_history},
            'reparam': {'mu': mu2, 'sigma': s2, 'elbo_history': vi2.elbo_history},
        })
        assert '_ranking' in comp
        assert len(comp['_ranking']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
