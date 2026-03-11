"""Tests for C156: Probabilistic Programming Language."""

import math
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from probabilistic_programming import (
    # Distributions
    Normal, HalfNormal, Uniform, Beta, Gamma, Exponential,
    Bernoulli, Poisson, StudentT, LogNormal, HalfCauchy,
    Categorical, Dirichlet, MultivariateNormal, MixtureDistribution,
    # Model
    ProbModel, Plate, Guide,
    # Inference
    MCMCInference, VIInference,
    # Predictive
    PriorPredictive, PosteriorPredictive,
    # Diagnostics & Comparison
    Trace, ConvergenceDiagnostics, ModelComparison,
    # Helpers
    linear_regression, logistic_regression, hierarchical_model, mixture_model,
)


# ============================================================
# Distribution Tests
# ============================================================

class TestNormal:
    def test_log_prob(self):
        d = Normal(0, 1)
        lp = d.log_prob(0)
        assert abs(lp - (-0.5 * math.log(2 * math.pi))) < 1e-10

    def test_log_prob_nonzero_mean(self):
        d = Normal(3, 2)
        lp = d.log_prob(3)
        expected = -0.5 * math.log(2 * math.pi) - math.log(2)
        assert abs(lp - expected) < 1e-10

    def test_sample_shape(self):
        d = Normal(0, 1)
        s = d.sample(size=100)
        assert s.shape == (100,)

    def test_sample_mean(self):
        rng = np.random.RandomState(42)
        d = Normal(5, 0.1)
        s = d.sample(rng=rng, size=10000)
        assert abs(np.mean(s) - 5) < 0.05


class TestHalfNormal:
    def test_log_prob_positive(self):
        d = HalfNormal(1)
        lp = d.log_prob(1.0)
        assert lp > -np.inf

    def test_log_prob_negative(self):
        d = HalfNormal(1)
        assert d.log_prob(-1.0) == -np.inf

    def test_sample_positive(self):
        d = HalfNormal(1)
        s = d.sample(size=100)
        assert np.all(s >= 0)

    def test_support(self):
        d = HalfNormal(1)
        assert d.support == (0, np.inf)

    def test_transform(self):
        assert HalfNormal(1).transform_type == 'positive'


class TestUniform:
    def test_log_prob_in_range(self):
        d = Uniform(0, 1)
        assert abs(d.log_prob(0.5) - 0.0) < 1e-10

    def test_log_prob_out_of_range(self):
        d = Uniform(0, 1)
        assert d.log_prob(1.5) == -np.inf

    def test_sample_range(self):
        d = Uniform(2, 5)
        s = d.sample(size=1000)
        assert np.all(s >= 2) and np.all(s <= 5)


class TestBeta:
    def test_log_prob(self):
        d = Beta(2, 3)
        lp = d.log_prob(0.5)
        assert lp > -np.inf

    def test_log_prob_boundary(self):
        d = Beta(1, 1)
        assert d.log_prob(0.0) == -np.inf
        assert d.log_prob(1.0) == -np.inf

    def test_sample_range(self):
        d = Beta(2, 3)
        s = d.sample(size=1000)
        assert np.all(s > 0) and np.all(s < 1)

    def test_transform(self):
        assert Beta(1, 1).transform_type == 'unit'


class TestGamma:
    def test_log_prob_positive(self):
        d = Gamma(2, 1)
        lp = d.log_prob(1.0)
        assert lp > -np.inf

    def test_log_prob_negative(self):
        d = Gamma(2, 1)
        assert d.log_prob(-1.0) == -np.inf

    def test_sample_positive(self):
        d = Gamma(2, 1)
        s = d.sample(size=1000)
        assert np.all(s > 0)


class TestExponential:
    def test_log_prob(self):
        d = Exponential(2)
        lp = d.log_prob(1.0)
        expected = math.log(2) - 2 * 1.0
        assert abs(lp - expected) < 1e-10

    def test_log_prob_negative(self):
        d = Exponential(1)
        assert d.log_prob(-0.1) == -np.inf

    def test_sample_positive(self):
        d = Exponential(1)
        s = d.sample(size=1000)
        assert np.all(s >= 0)


class TestBernoulli:
    def test_log_prob_1(self):
        d = Bernoulli(0.7)
        assert abs(d.log_prob(1) - math.log(0.7)) < 1e-10

    def test_log_prob_0(self):
        d = Bernoulli(0.7)
        assert abs(d.log_prob(0) - math.log(0.3)) < 1e-10

    def test_log_prob_invalid(self):
        d = Bernoulli(0.5)
        assert d.log_prob(0.5) == -np.inf

    def test_sample_values(self):
        d = Bernoulli(0.5)
        s = d.sample(size=1000)
        assert set(np.unique(s)).issubset({0.0, 1.0})


class TestPoisson:
    def test_log_prob(self):
        d = Poisson(3)
        lp = d.log_prob(2)
        expected = 2 * math.log(3) - 3 - math.lgamma(3)
        assert abs(lp - expected) < 1e-10

    def test_log_prob_negative(self):
        d = Poisson(1)
        assert d.log_prob(-1) == -np.inf

    def test_sample_nonneg(self):
        d = Poisson(5)
        s = d.sample(size=100)
        assert np.all(s >= 0)


class TestStudentT:
    def test_log_prob(self):
        d = StudentT(nu=1, mu=0, sigma=1)  # Cauchy
        lp = d.log_prob(0)
        assert lp > -np.inf

    def test_sample_shape(self):
        d = StudentT(nu=5)
        s = d.sample(size=100)
        assert s.shape == (100,)


class TestLogNormal:
    def test_log_prob_positive(self):
        d = LogNormal(0, 1)
        lp = d.log_prob(1.0)
        assert lp > -np.inf

    def test_log_prob_nonpositive(self):
        d = LogNormal(0, 1)
        assert d.log_prob(0) == -np.inf
        assert d.log_prob(-1) == -np.inf

    def test_sample_positive(self):
        d = LogNormal(0, 1)
        s = d.sample(size=100)
        assert np.all(s > 0)


class TestHalfCauchy:
    def test_log_prob_positive(self):
        d = HalfCauchy(1)
        lp = d.log_prob(1.0)
        assert lp > -np.inf

    def test_log_prob_negative(self):
        d = HalfCauchy(1)
        assert d.log_prob(-1) == -np.inf

    def test_sample_positive(self):
        d = HalfCauchy(1)
        s = d.sample(size=100)
        assert np.all(s >= 0)


class TestCategorical:
    def test_log_prob(self):
        d = Categorical([0.3, 0.7])
        assert abs(d.log_prob(0) - math.log(0.3)) < 1e-10
        assert abs(d.log_prob(1) - math.log(0.7)) < 1e-10

    def test_log_prob_invalid(self):
        d = Categorical([0.5, 0.5])
        assert d.log_prob(-1) == -np.inf
        assert d.log_prob(2) == -np.inf

    def test_sample_values(self):
        d = Categorical([0.2, 0.8])
        s = d.sample(size=1000)
        assert set(np.unique(s)).issubset({0.0, 1.0})

    def test_normalization(self):
        d = Categorical([1, 2, 3])
        assert abs(d.probs.sum() - 1.0) < 1e-10


class TestDirichlet:
    def test_log_prob(self):
        d = Dirichlet([1, 1, 1])
        lp = d.log_prob([1/3, 1/3, 1/3])
        assert lp > -np.inf

    def test_log_prob_invalid(self):
        d = Dirichlet([1, 1])
        assert d.log_prob([0.3, 0.3]) == -np.inf  # doesn't sum to 1

    def test_sample_simplex(self):
        d = Dirichlet([2, 2, 2])
        s = d.sample(size=10)
        assert s.shape == (10, 3)
        assert np.allclose(s.sum(axis=1), 1.0)


class TestMultivariateNormal:
    def test_log_prob(self):
        d = MultivariateNormal([0, 0], [[1, 0], [0, 1]])
        lp = d.log_prob([0, 0])
        expected = -math.log(2 * math.pi)
        assert abs(lp - expected) < 1e-6

    def test_sample_shape(self):
        d = MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]])
        s = d.sample(size=100)
        assert s.shape == (100, 2)


class TestMixtureDistribution:
    def test_log_prob(self):
        m = MixtureDistribution([0.5, 0.5], [Normal(0, 1), Normal(5, 1)])
        lp = m.log_prob(2.5)
        assert lp > -np.inf

    def test_sample(self):
        m = MixtureDistribution([0.5, 0.5], [Normal(0, 0.1), Normal(10, 0.1)])
        s = m.sample(size=1000)
        # Should be bimodal
        assert np.any(s < 2) and np.any(s > 8)


# ============================================================
# Model Tests
# ============================================================

class TestProbModel:
    def test_create_model(self):
        m = ProbModel("test")
        assert m.name == "test"

    def test_sample_site(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 1))
        assert "mu" in m._latent_sites
        assert m.num_latent == 1

    def test_observe_site(self):
        m = ProbModel()
        m.observe("y", Normal(0, 1), [1.0, 2.0])
        assert "y" in m._observed_sites

    def test_duplicate_raises(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        with pytest.raises(ValueError, match="already registered"):
            m.sample("x", Normal(0, 1))

    def test_num_latent_multi(self):
        m = ProbModel()
        m.sample("a", Normal(0, 1))
        m.sample("b", Normal(0, 1))
        m.sample("c", Normal(0, 1))
        assert m.num_latent == 3

    def test_num_latent_shaped(self):
        m = ProbModel()
        m.sample("w", Normal(0, 1), shape=(3,))
        assert m.num_latent == 3

    def test_pack_unpack(self):
        m = ProbModel()
        m.sample("a", Normal(0, 1))
        m.sample("b", Normal(0, 1))
        vals = {"a": 1.0, "b": 2.0}
        theta = m._pack_params(vals)
        assert np.allclose(theta, [1.0, 2.0])
        recovered = m._unpack_params(theta)
        assert abs(recovered["a"] - 1.0) < 1e-10
        assert abs(recovered["b"] - 2.0) < 1e-10

    def test_log_prior(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        lp = m.log_prior(np.array([0.0]))
        assert abs(lp - Normal(0, 1).log_prob(0)) < 1e-10

    def test_log_likelihood(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), [0.0])
        ll = m.log_likelihood(np.array([0.0]))
        assert abs(ll - Normal(0, 1).log_prob(0)) < 1e-10

    def test_log_joint(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), [0.0])
        lj = m.log_joint(np.array([0.0]))
        expected = Normal(0, 10).log_prob(0) + Normal(0, 1).log_prob(0)
        assert abs(lj - expected) < 1e-10

    def test_prior_sample(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        rng = np.random.RandomState(42)
        s = m.prior_sample(rng=rng)
        assert "x" in s
        assert isinstance(s["x"], float)

    def test_deterministic_site(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        m.deterministic("x2", lambda v: v["x"] * 2)
        m.observe("y", lambda v: Normal(v["x2"], 1), [2.0])
        # x=1 => x2=2 => y=2 has likelihood Normal(2,1).log_prob(2) = Normal(0,1).log_prob(0)
        ll = m.log_likelihood(np.array([1.0]))
        assert abs(ll - Normal(0, 1).log_prob(0)) < 1e-10

    def test_static_dist_observe(self):
        """Observe with a fixed distribution (not callable)."""
        m = ProbModel()
        m.observe("y", Normal(0, 1), [0.5])
        ll = m.log_likelihood(np.array([]))
        assert abs(ll - Normal(0, 1).log_prob(0.5)) < 1e-10


# ============================================================
# Trace Tests
# ============================================================

class TestTrace:
    def setup_method(self):
        self.trace = Trace({
            "mu": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "sigma": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        })

    def test_len(self):
        assert len(self.trace) == 5

    def test_getitem(self):
        assert np.allclose(self.trace["mu"], [1, 2, 3, 4, 5])

    def test_contains(self):
        assert "mu" in self.trace
        assert "x" not in self.trace

    def test_param_names(self):
        assert set(self.trace.param_names) == {"mu", "sigma"}

    def test_mean(self):
        assert abs(self.trace.mean("mu") - 3.0) < 1e-10

    def test_mean_all(self):
        m = self.trace.mean()
        assert abs(m["mu"] - 3.0) < 1e-10

    def test_std(self):
        assert self.trace.std("mu") > 0

    def test_median(self):
        assert abs(self.trace.median("mu") - 3.0) < 1e-10

    def test_quantile(self):
        q = self.trace.quantile(0.5, "mu")
        assert abs(q - 3.0) < 1e-10

    def test_hdi(self):
        lo, hi = self.trace.hdi(0.94, "mu")
        assert lo <= hi
        assert lo >= 1.0 and hi <= 5.0

    def test_summary(self):
        s = self.trace.summary()
        assert "mu" in s
        assert "mean" in s["mu"]
        assert "std" in s["mu"]
        assert "q2.5" in s["mu"]

    def test_ess(self):
        # With 5 samples, ESS should be small but positive
        ess = self.trace.effective_sample_size("mu")
        assert ess > 0

    def test_r_hat_few_samples(self):
        # With only 5 samples, can't split into 4 chains
        rhat = self.trace.r_hat("mu", n_chains=4)
        assert np.isnan(rhat)


# ============================================================
# MCMC Inference Tests
# ============================================================

class TestMCMCInference:
    def test_metropolis_normal(self):
        """Infer mean of Normal with known variance."""
        rng = np.random.RandomState(42)
        true_mu = 3.0
        data = rng.normal(true_mu, 1.0, size=50)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=2000, n_warmup=500, proposal_scale=0.5)

        assert abs(trace.mean("mu") - true_mu) < 0.5

    def test_slice_normal(self):
        """Infer mean with slice sampler."""
        rng = np.random.RandomState(42)
        data = rng.normal(2.0, 1.0, size=30)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        mcmc = MCMCInference(m, sampler='slice', seed=42)
        trace = mcmc.run(n_samples=1000, n_warmup=200)

        assert abs(trace.mean("mu") - 2.0) < 0.8

    def test_hmc_normal(self):
        """Infer mean with HMC."""
        rng = np.random.RandomState(42)
        data = rng.normal(1.0, 0.5, size=40)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 0.5), data)

        mcmc = MCMCInference(m, sampler='hmc', seed=42)
        trace = mcmc.run(n_samples=500, n_warmup=200, step_size=0.05)

        assert abs(trace.mean("mu") - 1.0) < 0.5

    def test_nuts_normal(self):
        """Infer mean with NUTS."""
        rng = np.random.RandomState(42)
        data = rng.normal(4.0, 1.0, size=30)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        mcmc = MCMCInference(m, sampler='nuts', seed=42)
        trace = mcmc.run(n_samples=500, n_warmup=200, step_size=0.05)

        assert abs(trace.mean("mu") - 4.0) < 1.0

    def test_multi_chain(self):
        """Multiple chains."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, size=20)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=500, n_warmup=200, n_chains=2, proposal_scale=0.5)

        assert len(trace) == 1000  # 500 * 2 chains

    def test_two_param(self):
        """Infer both mean and (log) sigma."""
        rng = np.random.RandomState(42)
        data = rng.normal(2.0, 1.0, size=50)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.sample("log_sigma", Normal(0, 2))
        m.observe("y", lambda v: Normal(v["mu"], max(math.exp(v["log_sigma"]), 1e-6)), data)

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=3000, n_warmup=1000, proposal_scale=0.3)

        assert abs(trace.mean("mu") - 2.0) < 0.8

    def test_unknown_sampler_raises(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        m.observe("y", Normal(0, 1), [0.0])
        mcmc = MCMCInference(m, sampler='gibbs')
        with pytest.raises(ValueError, match="Unknown sampler"):
            mcmc.run()

    def test_trace_param_names(self):
        m = ProbModel()
        m.sample("a", Normal(0, 1))
        m.sample("b", Normal(0, 1))
        m.observe("y", Normal(0, 1), [0.0])
        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=100, n_warmup=50)
        assert set(trace.param_names) == {"a", "b"}


# ============================================================
# VI Inference Tests
# ============================================================

class TestVIInference:
    def test_meanfield(self):
        """Mean-field VI on simple normal."""
        rng = np.random.RandomState(42)
        data = rng.normal(3.0, 1.0, size=50)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        vi = VIInference(m, method='meanfield', seed=42)
        trace = vi.run(n_iter=1000, lr=0.01)

        assert abs(trace.mean("mu") - 3.0) < 1.5

    def test_reparameterized(self):
        """Reparameterized VI."""
        rng = np.random.RandomState(42)
        data = rng.normal(2.0, 1.0, size=30)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        vi = VIInference(m, method='reparameterized', seed=42)
        trace = vi.run(n_iter=1000, lr=0.01)

        assert abs(trace.mean("mu") - 2.0) < 2.0

    def test_advi(self):
        """ADVI with transforms."""
        rng = np.random.RandomState(42)
        data = rng.normal(1.0, 2.0, size=30)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.sample("sigma", HalfNormal(5))
        m.observe("y", lambda v: Normal(v["mu"], max(v["sigma"], 1e-6)), data)

        vi = VIInference(m, method='advi', seed=42)
        trace = vi.run(n_iter=1000, lr=0.01)

        assert "mu" in trace and "sigma" in trace

    def test_elbo_history(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        m.observe("y", lambda v: Normal(v["x"], 1), [0.5])

        vi = VIInference(m, method='meanfield', seed=42)
        trace = vi.run(n_iter=100)
        assert hasattr(trace, 'elbo_history')
        assert len(trace.elbo_history) > 0

    def test_unknown_method_raises(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        m.observe("y", Normal(0, 1), [0.0])
        vi = VIInference(m, method='nonexistent')
        with pytest.raises(ValueError, match="Unknown method"):
            vi.run()

    def test_blackbox(self):
        """Black box VI."""
        rng = np.random.RandomState(42)
        data = rng.normal(1.0, 1.0, size=20)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        vi = VIInference(m, method='blackbox', seed=42)
        trace = vi.run(n_iter=500, lr=0.01)
        assert "mu" in trace


# ============================================================
# Prior Predictive Tests
# ============================================================

class TestPriorPredictive:
    def test_basic(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 1))
        m.observe("y", lambda v: Normal(v["mu"], 1), [0.0])

        pp = PriorPredictive(m, seed=42)
        preds = pp.sample(n=50)

        assert "y" in preds
        assert preds["y"].shape == (50, 1)

    def test_multi_obs(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 1))
        m.observe("y", lambda v: Normal(v["mu"], 1), [0.0, 1.0, 2.0])

        pp = PriorPredictive(m, seed=42)
        preds = pp.sample(n=20)

        assert preds["y"].shape == (20, 3)

    def test_deterministic_in_predictive(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        m.deterministic("x2", lambda v: v["x"] * 2)
        m.observe("y", lambda v: Normal(v["x2"], 0.1), [0.0])

        pp = PriorPredictive(m, seed=42)
        preds = pp.sample(n=100)
        assert preds["y"].shape == (100, 1)


# ============================================================
# Posterior Predictive Tests
# ============================================================

class TestPosteriorPredictive:
    def test_basic(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), [0.0])

        trace = Trace({"mu": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}, m)
        pp = PosteriorPredictive(m, trace, seed=42)
        preds = pp.sample(n=5)

        assert "y" in preds
        assert preds["y"].shape[0] == 5

    def test_mean_prediction(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 0.01), [3.0])

        # Trace concentrated at mu=3
        trace = Trace({"mu": np.full(100, 3.0)}, m)
        pp = PosteriorPredictive(m, trace, seed=42)
        means = pp.mean_prediction()

        assert abs(means["y"][0] - 3.0) < 0.1

    def test_subset(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), [0.0])

        trace = Trace({"mu": np.arange(100, dtype=float)}, m)
        pp = PosteriorPredictive(m, trace, seed=42)
        preds = pp.sample(n=10)
        assert preds["y"].shape[0] == 10


# ============================================================
# Model Comparison Tests
# ============================================================

class TestModelComparison:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.data = rng.normal(2, 1, size=30)

    def test_waic(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), self.data)

        trace = Trace({"mu": np.full(200, 2.0)}, m)
        w = ModelComparison.waic(m, trace)

        assert "waic" in w
        assert "p_waic" in w
        assert "lppd" in w

    def test_loo(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), self.data)

        trace = Trace({"mu": np.full(200, 2.0)}, m)
        l = ModelComparison.loo(m, trace)

        assert "loo" in l
        assert "p_loo" in l

    def test_bic(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), self.data)

        trace = Trace({"mu": np.full(200, 2.0)}, m)
        b = ModelComparison.bic(m, trace)

        assert "bic" in b
        assert b["k"] == 1
        assert b["n"] == 30

    def test_compare(self):
        # Model 1: correct
        m1 = ProbModel("correct")
        m1.sample("mu", Normal(0, 10))
        m1.observe("y", lambda v: Normal(v["mu"], 1), self.data)
        t1 = Trace({"mu": np.full(200, np.mean(self.data))}, m1)

        # Model 2: wrong mean
        m2 = ProbModel("wrong")
        m2.sample("mu", Normal(0, 10))
        m2.observe("y", lambda v: Normal(v["mu"], 1), self.data)
        t2 = Trace({"mu": np.full(200, 10.0)}, m2)

        results = ModelComparison.compare([
            ("correct", m1, t1),
            ("wrong", m2, t2),
        ])

        assert len(results) == 2
        assert results[0]["name"] == "correct"  # should rank first
        assert results[0]["rank"] == 1

    def test_empty_obs(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        trace = Trace({"x": np.array([0.0, 1.0])}, m)
        w = ModelComparison.waic(m, trace)
        assert w["waic"] == 0.0


# ============================================================
# Convergence Diagnostics Tests
# ============================================================

class TestConvergenceDiagnostics:
    def test_summary(self):
        trace = Trace({"x": np.random.randn(100)})
        s = ConvergenceDiagnostics.summary(trace)
        assert "x" in s

    def test_check_convergence(self):
        trace = Trace({"x": np.random.randn(100)})
        c = ConvergenceDiagnostics.check_convergence(trace, n_chains=4)
        assert "x" in c
        assert "ess" in c["x"]
        assert "converged" in c["x"]

    def test_autocorrelation(self):
        trace = Trace({"x": np.random.randn(100)})
        acf = ConvergenceDiagnostics.autocorrelation(trace, "x", max_lag=20)
        assert len(acf) == 20
        assert abs(acf[0] - 1.0) < 1e-10  # lag-0 is always 1


# ============================================================
# Plate Tests
# ============================================================

class TestPlate:
    def test_basic_plate(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        plate = Plate(m, "y", [1, 2, 3, 4, 5])
        plate.observe(lambda v: Normal(v["mu"], 1))

        assert "y" in m._observed_sites

    def test_plate_log_lik(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        Plate(m, "y", [0.0]).observe(lambda v: Normal(v["mu"], 1))

        ll = m.log_likelihood(np.array([0.0]))
        assert abs(ll - Normal(0, 1).log_prob(0)) < 1e-10


# ============================================================
# Guide Tests
# ============================================================

class TestGuide:
    def test_register(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        g = Guide(m)
        g.register("mu", Normal(3, 0.5))
        assert "mu" in g._guide_dists

    def test_register_invalid(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        g = Guide(m)
        with pytest.raises(ValueError, match="not a latent site"):
            g.register("x", Normal(0, 1))

    def test_log_prob(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        g = Guide(m)
        g.register("mu", Normal(3, 1))
        lp = g.log_prob(np.array([3.0]))
        assert abs(lp - Normal(3, 1).log_prob(3.0)) < 1e-10

    def test_sample(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        g = Guide(m)
        g.register("mu", Normal(3, 0.1))
        rng = np.random.RandomState(42)
        s = g.sample(rng=rng)
        assert "mu" in s
        assert abs(s["mu"] - 3.0) < 1.0


# ============================================================
# Helper Model Tests
# ============================================================

class TestLinearRegression:
    def test_build(self):
        m = linear_regression([1, 2, 3], [2, 4, 6])
        assert "alpha" in m._latent_sites
        assert "beta" in m._latent_sites
        assert "sigma" in m._latent_sites
        assert "y" in m._observed_sites
        assert m.num_latent == 3

    def test_inference(self):
        rng = np.random.RandomState(42)
        x = np.linspace(0, 5, 20)
        y = 1.0 + 2.0 * x + rng.normal(0, 0.5, size=20)

        m = ProbModel()
        m.sample("alpha", Normal(0, 10))
        m.sample("beta", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["alpha"] + v["beta"] * np.mean(x), 0.5), y)

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=2000, n_warmup=500, proposal_scale=0.3)
        # Just check it ran
        assert len(trace) == 2000


class TestLogisticRegression:
    def test_build(self):
        m = logistic_regression([1, 2, 3, 4], [0, 0, 1, 1])
        assert "alpha" in m._latent_sites
        assert "beta" in m._latent_sites
        assert m.num_latent == 2


class TestHierarchicalModel:
    def test_build(self):
        groups = {"A": [1, 2, 3], "B": [4, 5, 6]}
        m = hierarchical_model(groups)
        assert "mu" in m._latent_sites
        assert "tau" in m._latent_sites
        assert "sigma" in m._latent_sites
        assert "group_A" in m._latent_sites
        assert "group_B" in m._latent_sites

    def test_num_latent(self):
        groups = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
        m = hierarchical_model(groups)
        # mu + tau + sigma + 3 groups = 6
        assert m.num_latent == 6


class TestMixtureModel:
    def test_build(self):
        m = mixture_model([1, 2, 3, 10, 11, 12], k=2)
        assert "mu_0" in m._latent_sites
        assert "mu_1" in m._latent_sites
        assert "log_sigma_0" in m._latent_sites
        assert "log_sigma_1" in m._latent_sites
        assert "logit_w" in m._latent_sites
        assert m.num_latent == 5


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_full_workflow(self):
        """Full workflow: model -> inference -> predictive -> comparison."""
        rng = np.random.RandomState(42)
        true_mu = 5.0
        data = rng.normal(true_mu, 1.0, size=30)

        # Build model
        m = ProbModel("test")
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        # Prior predictive
        pp = PriorPredictive(m, seed=42)
        prior_preds = pp.sample(n=20)
        assert prior_preds["y"].shape == (20, 30)

        # MCMC inference
        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=1000, n_warmup=500, proposal_scale=0.5)

        assert abs(trace.mean("mu") - true_mu) < 1.0

        # Posterior predictive
        post_pp = PosteriorPredictive(m, trace, seed=42)
        post_preds = post_pp.sample(n=50)
        assert post_preds["y"].shape[0] == 50

        # Diagnostics
        summary = trace.summary()
        assert "mu" in summary

    def test_vi_vs_mcmc(self):
        """Compare VI and MCMC on same model."""
        rng = np.random.RandomState(42)
        data = rng.normal(3.0, 1.0, size=30)

        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1.0), data)

        # MCMC
        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace_mcmc = mcmc.run(n_samples=2000, n_warmup=500, proposal_scale=0.5)

        # VI
        vi = VIInference(m, method='meanfield', seed=42)
        trace_vi = vi.run(n_iter=1000, lr=0.01)

        # Both should be close to true value
        assert abs(trace_mcmc.mean("mu") - 3.0) < 1.0
        assert abs(trace_vi.mean("mu") - 3.0) < 2.0

    def test_model_comparison_workflow(self):
        """Compare two models using WAIC."""
        rng = np.random.RandomState(42)
        data = rng.normal(2, 1, size=20)

        # Good model
        m1 = ProbModel("good")
        m1.sample("mu", Normal(0, 10))
        m1.observe("y", lambda v: Normal(v["mu"], 1), data)
        t1 = Trace({"mu": np.full(100, np.mean(data))}, m1)

        # Bad model (wrong sigma)
        m2 = ProbModel("bad")
        m2.sample("mu", Normal(0, 10))
        m2.observe("y", lambda v: Normal(v["mu"], 10), data)
        t2 = Trace({"mu": np.full(100, np.mean(data))}, m2)

        results = ModelComparison.compare([
            ("good", m1, t1),
            ("bad", m2, t2),
        ])

        assert results[0]["name"] == "good"

    def test_shaped_params(self):
        """Test model with shaped (vector) parameters."""
        m = ProbModel()
        m.sample("weights", Normal(0, 1), shape=(3,))
        assert m.num_latent == 3

        theta = np.array([1.0, 2.0, 3.0])
        vals = m._unpack_params(theta)
        assert vals["weights"].shape == (3,)
        assert np.allclose(vals["weights"], [1, 2, 3])

    def test_convergence_diagnostics(self):
        """Test convergence checking on synthetic trace."""
        # Well-mixed trace
        rng = np.random.RandomState(42)
        trace = Trace({"x": rng.normal(0, 1, size=500)})
        c = ConvergenceDiagnostics.check_convergence(trace, n_chains=4)
        assert c["x"]["ess"] > 0

    def test_guide_workflow(self):
        """Test custom guide."""
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.sample("sigma", HalfNormal(5))
        m.observe("y", lambda v: Normal(v["mu"], max(v["sigma"], 0.01)), [1, 2, 3])

        g = Guide(m)
        g.register("mu", Normal(2, 0.5))
        g.register("sigma", HalfNormal(1))

        rng = np.random.RandomState(42)
        for _ in range(10):
            s = g.sample(rng=rng)
            assert "mu" in s
            assert "sigma" in s

    def test_bernoulli_model(self):
        """Bernoulli model inference."""
        rng = np.random.RandomState(42)
        true_p = 0.7
        data = (rng.uniform(size=50) < true_p).astype(float)

        m = ProbModel()
        m.sample("logit_p", Normal(0, 2))

        def sigmoid(z):
            return 1.0 / (1.0 + math.exp(-max(min(z, 500), -500)))

        for i in range(len(data)):
            m.observe(f"y_{i}",
                      lambda v: Bernoulli(sigmoid(v["logit_p"])),
                      data[i])

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=2000, n_warmup=500, proposal_scale=0.3)

        inferred_p = 1.0 / (1.0 + math.exp(-trace.mean("logit_p")))
        assert abs(inferred_p - true_p) < 0.2

    def test_poisson_model(self):
        """Poisson model inference."""
        rng = np.random.RandomState(42)
        true_rate = 5.0
        data = rng.poisson(true_rate, size=30).astype(float)

        m = ProbModel()
        m.sample("log_rate", Normal(1, 2))

        m.observe("y",
                  lambda v: Poisson(math.exp(v["log_rate"])),
                  data)

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=3000, n_warmup=1000, proposal_scale=0.2)

        inferred_rate = math.exp(trace.mean("log_rate"))
        assert abs(inferred_rate - true_rate) < 2.0


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    def test_single_observation(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y", lambda v: Normal(v["mu"], 1), [5.0])

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=500, n_warmup=200)
        assert len(trace) == 500

    def test_wide_prior(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1000))
        m.observe("y", lambda v: Normal(v["x"], 1), [0.0])

        mcmc = MCMCInference(m, sampler='metropolis', seed=42)
        trace = mcmc.run(n_samples=500, n_warmup=200)
        assert "x" in trace

    def test_multiple_observed_sites(self):
        m = ProbModel()
        m.sample("mu", Normal(0, 10))
        m.observe("y1", lambda v: Normal(v["mu"], 1), [1.0])
        m.observe("y2", lambda v: Normal(v["mu"], 1), [1.5])

        lj = m.log_joint(np.array([1.25]))
        assert lj > -np.inf

    def test_empty_model_prior(self):
        m = ProbModel()
        m.sample("x", Normal(0, 1))
        # No observations
        lj = m.log_joint(np.array([0.0]))
        assert abs(lj - Normal(0, 1).log_prob(0)) < 1e-10

    def test_chain_api(self):
        """Test chaining API."""
        m = ProbModel()
        m.sample("a", Normal(0, 1)).sample("b", Normal(0, 1)).observe("y", Normal(0, 1), [0.0])
        assert m.num_latent == 2

    def test_trace_hdi_full_range(self):
        """HDI with prob=1.0 should cover all data."""
        trace = Trace({"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
        lo, hi = trace.hdi(1.0, "x")
        assert lo == 1.0 and hi == 5.0

    def test_log_joint_boundary(self):
        """Log joint returns -inf when prior is violated."""
        m = ProbModel()
        m.sample("x", Uniform(0, 1))
        m.observe("y", lambda v: Normal(v["x"], 1), [0.5])
        assert m.log_joint(np.array([-1.0])) == -np.inf
        assert m.log_joint(np.array([2.0])) == -np.inf
        assert m.log_joint(np.array([0.5])) > -np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
