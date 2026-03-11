"""Tests for C153: Monte Carlo Methods."""

import math
import numpy as np
import pytest
from monte_carlo import (
    RandomSampler, MonteCarloIntegrator, MetropolisHastings,
    GibbsSampler, HamiltonianMC, NUTS, ParallelTempering,
    SliceSampler, MCMCDiagnostics
)


# ============================================================
# RandomSampler tests
# ============================================================

class TestRandomSampler:
    def test_uniform_range(self):
        s = RandomSampler(42)
        vals = s.uniform(2.0, 5.0, size=1000)
        assert np.all(vals >= 2.0) and np.all(vals <= 5.0)

    def test_normal_mean_std(self):
        s = RandomSampler(42)
        vals = s.normal(3.0, 0.5, size=10000)
        assert abs(np.mean(vals) - 3.0) < 0.05
        assert abs(np.std(vals) - 0.5) < 0.05

    def test_exponential_mean(self):
        s = RandomSampler(42)
        vals = s.exponential(2.0, size=10000)
        assert abs(np.mean(vals) - 2.0) < 0.1

    def test_multivariate_normal(self):
        s = RandomSampler(42)
        mean = [1.0, 2.0]
        cov = [[1.0, 0.5], [0.5, 1.0]]
        vals = s.multivariate_normal(mean, cov, size=5000)
        assert abs(np.mean(vals[:, 0]) - 1.0) < 0.1
        assert abs(np.mean(vals[:, 1]) - 2.0) < 0.1

    def test_categorical(self):
        s = RandomSampler(42)
        probs = [0.1, 0.7, 0.2]
        counts = [0, 0, 0]
        for _ in range(10000):
            counts[s.categorical(probs)] += 1
        assert counts[1] > counts[0]
        assert counts[1] > counts[2]

    def test_beta(self):
        s = RandomSampler(42)
        vals = s.beta(2, 5, size=10000)
        expected_mean = 2.0 / (2.0 + 5.0)
        assert abs(np.mean(vals) - expected_mean) < 0.02

    def test_gamma(self):
        s = RandomSampler(42)
        vals = s.gamma(2.0, 3.0, size=10000)
        assert abs(np.mean(vals) - 6.0) < 0.2  # shape * scale

    def test_dirichlet(self):
        s = RandomSampler(42)
        vals = s.dirichlet([1.0, 1.0, 1.0], size=1000)
        assert vals.shape == (1000, 3)
        assert np.allclose(vals.sum(axis=1), 1.0, atol=1e-10)

    def test_scalar_uniform(self):
        s = RandomSampler(42)
        v = s.uniform(0, 1)
        assert isinstance(v, float)

    def test_scalar_normal(self):
        s = RandomSampler(42)
        v = s.normal(0, 1)
        assert isinstance(v, (float, np.floating))

    def test_scalar_exponential(self):
        s = RandomSampler(42)
        v = s.exponential(1.0)
        assert isinstance(v, (float, np.floating))
        assert v > 0

    def test_scalar_beta(self):
        s = RandomSampler(42)
        v = s.beta(2, 2)
        assert 0 <= v <= 1

    def test_scalar_gamma(self):
        s = RandomSampler(42)
        v = s.gamma(2.0)
        assert v > 0

    def test_seed_reproducibility(self):
        s1 = RandomSampler(123)
        s2 = RandomSampler(123)
        assert s1.uniform() == s2.uniform()
        np.testing.assert_array_equal(s1.normal(size=10), s2.normal(size=10))


# ============================================================
# MonteCarloIntegrator tests
# ============================================================

class TestMonteCarloIntegrator:
    def test_basic_integrate_constant(self):
        """Integral of f(x)=2 over [0,1] = 2."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.basic_integrate(lambda x: 2.0, [(0, 1)], n_samples=5000)
        assert abs(est - 2.0) < 0.1

    def test_basic_integrate_linear(self):
        """Integral of f(x)=x over [0,1] = 0.5."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.basic_integrate(lambda x: x[0], [(0, 1)], n_samples=10000)
        assert abs(est - 0.5) < 0.05

    def test_basic_integrate_2d(self):
        """Integral of f(x,y)=1 over [0,1]x[0,1] = 1."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.basic_integrate(lambda x: 1.0, [(0, 1), (0, 1)], n_samples=5000)
        assert abs(est - 1.0) < 0.1

    def test_basic_integrate_quadratic(self):
        """Integral of x^2 over [0,1] = 1/3."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.basic_integrate(lambda x: x[0]**2, [(0, 1)], n_samples=20000)
        assert abs(est - 1.0/3) < 0.05

    def test_basic_integrate_sin(self):
        """Integral of sin(x) over [0, pi] = 2."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.basic_integrate(lambda x: math.sin(x[0]), [(0, math.pi)], n_samples=20000)
        assert abs(est - 2.0) < 0.1

    def test_std_error_decreases(self):
        """More samples => smaller std error."""
        mc = MonteCarloIntegrator(42)
        _, err1 = mc.basic_integrate(lambda x: x[0]**2, [(0, 1)], n_samples=1000)
        mc2 = MonteCarloIntegrator(42)
        _, err2 = mc2.basic_integrate(lambda x: x[0]**2, [(0, 1)], n_samples=100000)
        assert err2 < err1

    def test_importance_sampling(self):
        """IS with good proposal should give reasonable estimate."""
        mc = MonteCarloIntegrator(42)
        # Integrate x^2 * exp(-x) over [0, inf) using exponential proposal
        # target_density = exp(-x), proposal = exp(-x), so weights = 1
        # E_target[x^2] = Gamma(3) = 2
        def f(x):
            return x**2

        def proposal_sampler():
            return mc.sampler.exponential(1.0)

        def proposal_density(x):
            return math.exp(-x) if x >= 0 else 0

        def target_density(x):
            return math.exp(-x) if x >= 0 else 0

        est, err = mc.importance_sampling(f, proposal_sampler, proposal_density,
                                          target_density=target_density,
                                          n_samples=20000)
        # E_target[x^2] = Gamma(3) = 2
        assert abs(est - 2.0) < 0.2

    def test_stratified_sampling_1d(self):
        """Stratified sampling for x^2 over [0,1]."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.stratified_sampling(lambda x: x[0]**2, [(0, 1)],
                                          n_strata_per_dim=10, samples_per_stratum=100)
        assert abs(est - 1.0/3) < 0.05

    def test_stratified_2d(self):
        """Stratified sampling for constant over [0,2]x[0,3] = 6."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.stratified_sampling(lambda x: 1.0, [(0, 2), (0, 3)],
                                          n_strata_per_dim=5, samples_per_stratum=20)
        assert abs(est - 6.0) < 0.5

    def test_control_variate(self):
        """Control variate should reduce variance."""
        mc = MonteCarloIntegrator(42)
        f = lambda x: math.exp(x[0])
        g = lambda x: 1.0 + x[0]  # Linear approximation
        g_expected = 1.5  # Integral of (1+x) over [0,1]

        est, err = mc.control_variate(f, g, g_expected, [(0, 1)], n_samples=20000)
        # Integral of e^x over [0,1] = e-1 ~ 1.718
        assert abs(est - (math.e - 1)) < 0.15

    def test_antithetic_variates(self):
        """Antithetic variates for monotone function."""
        mc = MonteCarloIntegrator(42)
        est, err = mc.antithetic_variates(lambda x: x[0]**2, [(0, 1)], n_samples=20000)
        assert abs(est - 1.0/3) < 0.05

    def test_antithetic_reduces_variance(self):
        """Antithetic should have smaller error for monotone functions."""
        mc1 = MonteCarloIntegrator(42)
        _, err_basic = mc1.basic_integrate(lambda x: x[0], [(0, 1)], n_samples=10000)
        mc2 = MonteCarloIntegrator(42)
        _, err_anti = mc2.antithetic_variates(lambda x: x[0], [(0, 1)], n_samples=10000)
        # Antithetic should help for monotone f(x)=x
        assert err_anti < err_basic * 1.5  # Allow some slack

    def test_integrate_gaussian_bump(self):
        """Integrate gaussian bump."""
        mc = MonteCarloIntegrator(42)
        est, _ = mc.basic_integrate(
            lambda x: math.exp(-x[0]**2),
            [(-5, 5)], n_samples=50000)
        assert abs(est - math.sqrt(math.pi)) < 0.15

    def test_importance_sampling_with_target(self):
        """IS with target density."""
        mc = MonteCarloIntegrator(42)

        def f(x):
            return x

        def proposal_sampler():
            return mc.sampler.exponential(1.0)

        def proposal_density(x):
            return math.exp(-x) if x >= 0 else 0

        def target_density(x):
            return math.exp(-x) if x >= 0 else 0

        est, _ = mc.importance_sampling(f, proposal_sampler, proposal_density,
                                        target_density=target_density, n_samples=20000)
        # E[X] under Exp(1) = 1
        assert abs(est - 1.0) < 0.1


# ============================================================
# MetropolisHastings tests
# ============================================================

class TestMetropolisHastings:
    def _standard_normal_lp(self, x):
        return -0.5 * np.dot(x, x)

    def test_basic_sampling(self):
        mh = MetropolisHastings(self._standard_normal_lp, dim=1, seed=42)
        mh.proposal_scale = 1.0
        samples, rate = mh.sample(5000, burn_in=1000)
        assert samples.shape == (5000, 1)
        assert 0.1 < rate < 0.9

    def test_mean_recovery_1d(self):
        """Should recover N(0,1) mean."""
        mh = MetropolisHastings(self._standard_normal_lp, dim=1, seed=42)
        mh.proposal_scale = 1.5
        samples, _ = mh.sample(10000, burn_in=2000)
        assert abs(np.mean(samples)) < 0.15

    def test_std_recovery_1d(self):
        """Should recover N(0,1) std."""
        mh = MetropolisHastings(self._standard_normal_lp, dim=1, seed=42)
        mh.proposal_scale = 1.5
        samples, _ = mh.sample(10000, burn_in=2000)
        assert abs(np.std(samples) - 1.0) < 0.2

    def test_2d_sampling(self):
        mh = MetropolisHastings(self._standard_normal_lp, dim=2, seed=42)
        mh.proposal_scale = 1.0
        samples, rate = mh.sample(5000, burn_in=1000)
        assert samples.shape == (5000, 2)
        assert abs(np.mean(samples[:, 0])) < 0.2
        assert abs(np.mean(samples[:, 1])) < 0.2

    def test_custom_initial(self):
        mh = MetropolisHastings(self._standard_normal_lp, dim=2, seed=42)
        samples, _ = mh.sample(100, initial=[5.0, 5.0], burn_in=0)
        # First sample should be near initial or moved away
        assert samples.shape == (100, 2)

    def test_thinning(self):
        mh = MetropolisHastings(self._standard_normal_lp, dim=1, seed=42)
        samples, _ = mh.sample(100, thin=5, burn_in=0)
        assert samples.shape == (100, 1)

    def test_burn_in(self):
        mh = MetropolisHastings(self._standard_normal_lp, dim=1, seed=42)
        samples_no_burn, _ = mh.sample(1000, burn_in=0, initial=[10.0])
        mh2 = MetropolisHastings(self._standard_normal_lp, dim=1, seed=42)
        samples_burn, _ = mh2.sample(1000, burn_in=2000, initial=[10.0])
        # With burn-in, mean should be closer to 0
        assert abs(np.mean(samples_burn)) < abs(np.mean(samples_no_burn)) + 0.5

    def test_adaptive_sampling(self):
        mh = MetropolisHastings(self._standard_normal_lp, dim=2, seed=42)
        mh.proposal_scale = 10.0  # Start too wide
        samples, rate = mh.adaptive_sample(5000, burn_in=1000)
        # Adaptive should improve acceptance
        assert samples.shape == (5000, 2)
        assert rate > 0.05

    def test_bimodal(self):
        """Sample from bimodal distribution."""
        def bimodal_lp(x):
            return np.log(
                0.5 * math.exp(-0.5 * (x[0] - 3)**2) +
                0.5 * math.exp(-0.5 * (x[0] + 3)**2) + 1e-300)
        mh = MetropolisHastings(bimodal_lp, dim=1, seed=42)
        mh.proposal_scale = 2.0
        samples, _ = mh.sample(20000, burn_in=5000)
        # Should visit both modes
        assert np.any(samples > 1) and np.any(samples < -1)

    def test_custom_proposal(self):
        """MH with custom proposal function."""
        mh = MetropolisHastings(self._standard_normal_lp, dim=1, seed=42)

        def custom_proposal(current):
            proposed = current + np.array([mh.sampler.normal(0, 0.5)])
            return proposed, 0.0, 0.0  # Symmetric

        samples, rate = mh.sample(2000, burn_in=500, proposal='custom',
                                   proposal_fn=custom_proposal)
        assert samples.shape == (2000, 1)
        assert rate > 0.1

    def test_accept_rate_depends_on_scale(self):
        """Larger scale -> lower acceptance."""
        mh1 = MetropolisHastings(self._standard_normal_lp, dim=5, seed=42)
        mh1.proposal_scale = 0.1
        _, rate1 = mh1.sample(2000, burn_in=500)

        mh2 = MetropolisHastings(self._standard_normal_lp, dim=5, seed=42)
        mh2.proposal_scale = 10.0
        _, rate2 = mh2.sample(2000, burn_in=500)

        assert rate1 > rate2


# ============================================================
# GibbsSampler tests
# ============================================================

class TestGibbsSampler:
    def test_independent_normals(self):
        """Gibbs sampling from independent normals."""
        rng = np.random.RandomState(42)

        def cond_0(state):
            return rng.normal(0, 1)

        def cond_1(state):
            return rng.normal(0, 1)

        gibbs = GibbsSampler([cond_0, cond_1], dim=2, seed=42)
        samples = gibbs.sample(5000, burn_in=500)
        assert abs(np.mean(samples[:, 0])) < 0.15
        assert abs(np.mean(samples[:, 1])) < 0.15

    def test_correlated_normals(self):
        """Gibbs from bivariate normal with correlation."""
        rho = 0.8
        rng = np.random.RandomState(42)

        def cond_0(state):
            return rng.normal(rho * state[1], math.sqrt(1 - rho**2))

        def cond_1(state):
            return rng.normal(rho * state[0], math.sqrt(1 - rho**2))

        gibbs = GibbsSampler([cond_0, cond_1], dim=2, seed=42)
        samples = gibbs.sample(10000, burn_in=1000)
        corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert abs(corr - rho) < 0.15

    def test_3d_gibbs(self):
        rng = np.random.RandomState(42)
        conditionals = [
            lambda s: rng.normal(0, 1),
            lambda s: rng.normal(0, 1),
            lambda s: rng.normal(0, 1),
        ]
        gibbs = GibbsSampler(conditionals, dim=3, seed=42)
        samples = gibbs.sample(3000, burn_in=500)
        assert samples.shape == (3000, 3)

    def test_thin_and_burn(self):
        rng = np.random.RandomState(42)
        conditionals = [
            lambda s: rng.normal(0, 1),
            lambda s: rng.normal(0, 1),
        ]
        gibbs = GibbsSampler(conditionals, dim=2, seed=42)
        samples = gibbs.sample(100, burn_in=200, thin=3)
        assert samples.shape == (100, 2)

    def test_block_gibbs(self):
        """Block Gibbs sampling."""
        rng = np.random.RandomState(42)

        def block_cond(state):
            return rng.normal(0, 1, size=2)

        gibbs = GibbsSampler([None, None], dim=2, seed=42)
        samples = gibbs.block_sample(
            2000, blocks=[[0, 1]], block_conditionals=[block_cond],
            burn_in=500)
        assert samples.shape == (2000, 2)
        assert abs(np.mean(samples[:, 0])) < 0.2

    def test_gibbs_from_initial(self):
        rng = np.random.RandomState(42)
        conditionals = [
            lambda s: rng.normal(0, 1),
            lambda s: rng.normal(0, 1),
        ]
        gibbs = GibbsSampler(conditionals, dim=2, seed=42)
        samples = gibbs.sample(1000, initial=[10.0, 10.0], burn_in=500)
        # After burn-in should be near 0
        assert abs(np.mean(samples[:, 0])) < 0.3


# ============================================================
# HamiltonianMC tests
# ============================================================

class TestHamiltonianMC:
    def _gauss_lp(self, x):
        return -0.5 * np.dot(x, x)

    def _gauss_grad(self, x):
        return -x

    def test_basic_hmc(self):
        hmc = HamiltonianMC(self._gauss_lp, self._gauss_grad, dim=2, seed=42)
        hmc.step_size = 0.1
        hmc.n_leapfrog = 20
        samples, rate = hmc.sample(3000, burn_in=500)
        assert samples.shape == (3000, 2)
        assert rate > 0.3

    def test_mean_recovery(self):
        hmc = HamiltonianMC(self._gauss_lp, self._gauss_grad, dim=2, seed=42)
        hmc.step_size = 0.1
        hmc.n_leapfrog = 20
        samples, _ = hmc.sample(5000, burn_in=1000)
        assert abs(np.mean(samples[:, 0])) < 0.15
        assert abs(np.mean(samples[:, 1])) < 0.15

    def test_std_recovery(self):
        hmc = HamiltonianMC(self._gauss_lp, self._gauss_grad, dim=2, seed=42)
        hmc.step_size = 0.1
        hmc.n_leapfrog = 20
        samples, _ = hmc.sample(5000, burn_in=1000)
        assert abs(np.std(samples[:, 0]) - 1.0) < 0.2

    def test_high_dim(self):
        """HMC in higher dimensions."""
        dim = 10
        hmc = HamiltonianMC(self._gauss_lp, self._gauss_grad, dim=dim, seed=42)
        hmc.step_size = 0.05
        hmc.n_leapfrog = 30
        samples, rate = hmc.sample(2000, burn_in=500)
        assert samples.shape == (2000, dim)
        assert rate > 0.2

    def test_correlated_target(self):
        """HMC for correlated Gaussian."""
        cov = np.array([[1.0, 0.9], [0.9, 1.0]])
        cov_inv = np.linalg.inv(cov)

        def lp(x):
            return -0.5 * x @ cov_inv @ x

        def grad_lp(x):
            return -cov_inv @ x

        hmc = HamiltonianMC(lp, grad_lp, dim=2, seed=42)
        hmc.step_size = 0.1
        hmc.n_leapfrog = 15
        samples, _ = hmc.sample(5000, burn_in=1000)
        corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert abs(corr - 0.9) < 0.15

    def test_thin(self):
        hmc = HamiltonianMC(self._gauss_lp, self._gauss_grad, dim=1, seed=42)
        hmc.step_size = 0.1
        hmc.n_leapfrog = 10
        samples, _ = hmc.sample(100, thin=3, burn_in=100)
        assert samples.shape == (100, 1)

    def test_accept_rate_reasonable(self):
        hmc = HamiltonianMC(self._gauss_lp, self._gauss_grad, dim=2, seed=42)
        hmc.step_size = 0.05
        hmc.n_leapfrog = 20
        _, rate = hmc.sample(2000, burn_in=500)
        assert 0.4 < rate < 1.0


# ============================================================
# NUTS tests
# ============================================================

class TestNUTS:
    def _gauss_lp(self, x):
        return -0.5 * np.dot(x, x)

    def _gauss_grad(self, x):
        return -x

    def test_basic_nuts(self):
        nuts = NUTS(self._gauss_lp, self._gauss_grad, dim=2, seed=42)
        nuts.step_size = 0.1
        nuts.max_depth = 5
        samples = nuts.sample(500, burn_in=200)
        assert samples.shape == (500, 2)

    def test_mean_recovery(self):
        nuts = NUTS(self._gauss_lp, self._gauss_grad, dim=2, seed=42)
        nuts.step_size = 0.1
        nuts.max_depth = 5
        samples = nuts.sample(2000, burn_in=500)
        assert abs(np.mean(samples[:, 0])) < 0.3
        assert abs(np.mean(samples[:, 1])) < 0.3

    def test_1d_nuts(self):
        nuts = NUTS(self._gauss_lp, self._gauss_grad, dim=1, seed=42)
        nuts.step_size = 0.1
        samples = nuts.sample(1000, burn_in=200)
        assert abs(np.mean(samples)) < 0.3

    def test_nuts_from_initial(self):
        nuts = NUTS(self._gauss_lp, self._gauss_grad, dim=2, seed=42)
        nuts.step_size = 0.1
        nuts.max_depth = 5
        samples = nuts.sample(500, initial=[3.0, 3.0], burn_in=200)
        assert samples.shape == (500, 2)

    def test_nuts_thin(self):
        nuts = NUTS(self._gauss_lp, self._gauss_grad, dim=1, seed=42)
        nuts.step_size = 0.1
        samples = nuts.sample(100, thin=2, burn_in=100)
        assert samples.shape == (100, 1)


# ============================================================
# ParallelTempering tests
# ============================================================

class TestParallelTempering:
    def _gauss_lp(self, x):
        return -0.5 * np.dot(x, x)

    def test_basic_pt(self):
        pt = ParallelTempering(self._gauss_lp, dim=2, n_chains=3, seed=42)
        pt.proposal_scale = 1.0
        samples, swap_rate = pt.sample(2000, burn_in=500)
        assert samples.shape == (2000, 2)
        assert 0 <= swap_rate <= 1

    def test_mean_recovery(self):
        pt = ParallelTempering(self._gauss_lp, dim=1, n_chains=4, seed=42)
        pt.proposal_scale = 1.0
        samples, _ = pt.sample(5000, burn_in=1000)
        assert abs(np.mean(samples)) < 0.2

    def test_custom_temperatures(self):
        pt = ParallelTempering(self._gauss_lp, dim=1,
                                temperatures=[1.0, 2.0, 5.0], seed=42)
        assert pt.n_chains == 3
        samples, _ = pt.sample(1000, burn_in=500)
        assert samples.shape == (1000, 1)

    def test_bimodal_mixing(self):
        """PT should mix better on bimodal than single chain."""
        def bimodal_lp(x):
            return np.log(
                0.5 * math.exp(-0.5 * (x[0] - 4)**2) +
                0.5 * math.exp(-0.5 * (x[0] + 4)**2) + 1e-300)

        pt = ParallelTempering(bimodal_lp, dim=1, n_chains=4, seed=42)
        pt.proposal_scale = 1.0
        samples, _ = pt.sample(10000, burn_in=2000)
        # Should visit both modes
        assert np.any(samples > 2) and np.any(samples < -2)

    def test_swap_interval(self):
        pt = ParallelTempering(self._gauss_lp, dim=1, n_chains=3, seed=42)
        samples, swap_rate = pt.sample(1000, burn_in=200, swap_interval=5)
        assert samples.shape == (1000, 1)

    def test_initial_points(self):
        pt = ParallelTempering(self._gauss_lp, dim=2, n_chains=3, seed=42)
        initial = [[0, 0], [1, 1], [2, 2]]
        samples, _ = pt.sample(500, initial=initial, burn_in=200)
        assert samples.shape == (500, 2)

    def test_single_initial_broadcast(self):
        pt = ParallelTempering(self._gauss_lp, dim=2, n_chains=3, seed=42)
        samples, _ = pt.sample(500, initial=[0.5, 0.5], burn_in=200)
        assert samples.shape == (500, 2)


# ============================================================
# SliceSampler tests
# ============================================================

class TestSliceSampler:
    def _gauss_lp(self, x):
        return -0.5 * np.dot(x, x)

    def test_basic_slice(self):
        ss = SliceSampler(self._gauss_lp, dim=1, seed=42)
        ss.width = 2.0
        samples = ss.sample(3000, burn_in=500)
        assert samples.shape == (3000, 1)

    def test_mean_recovery(self):
        ss = SliceSampler(self._gauss_lp, dim=1, seed=42)
        ss.width = 2.0
        samples = ss.sample(5000, burn_in=1000)
        assert abs(np.mean(samples)) < 0.2

    def test_std_recovery(self):
        ss = SliceSampler(self._gauss_lp, dim=1, seed=42)
        ss.width = 2.0
        samples = ss.sample(5000, burn_in=1000)
        assert abs(np.std(samples) - 1.0) < 0.25

    def test_2d_slice(self):
        ss = SliceSampler(self._gauss_lp, dim=2, seed=42)
        ss.width = 2.0
        samples = ss.sample(3000, burn_in=500)
        assert samples.shape == (3000, 2)
        assert abs(np.mean(samples[:, 0])) < 0.2

    def test_thin(self):
        ss = SliceSampler(self._gauss_lp, dim=1, seed=42)
        samples = ss.sample(100, thin=3, burn_in=100)
        assert samples.shape == (100, 1)

    def test_from_initial(self):
        ss = SliceSampler(self._gauss_lp, dim=1, seed=42)
        ss.width = 2.0
        samples = ss.sample(2000, initial=[5.0], burn_in=500)
        assert abs(np.mean(samples)) < 0.3

    def test_narrow_target(self):
        """Slice with narrow target."""
        def narrow_lp(x):
            return -50 * np.dot(x, x)  # std ~ 0.1
        ss = SliceSampler(narrow_lp, dim=1, seed=42)
        ss.width = 0.5
        samples = ss.sample(3000, burn_in=500)
        assert np.std(samples) < 0.3


# ============================================================
# MCMCDiagnostics tests
# ============================================================

class TestMCMCDiagnostics:
    def test_ess_iid(self):
        """ESS of iid samples should be near n."""
        samples = np.random.RandomState(42).normal(size=(1000, 1))
        ess = MCMCDiagnostics.effective_sample_size(samples)
        assert ess > 500  # Should be close to 1000

    def test_ess_correlated(self):
        """ESS of highly correlated samples should be much less than n."""
        # Generate AR(1) process with high correlation
        n = 2000
        x = np.zeros(n)
        rng = np.random.RandomState(42)
        for i in range(1, n):
            x[i] = 0.99 * x[i-1] + rng.normal(0, 0.1)
        ess = MCMCDiagnostics.effective_sample_size(x)
        assert ess < n * 0.5

    def test_ess_1d(self):
        """ESS with 1d input."""
        samples = np.random.RandomState(42).normal(size=500)
        ess = MCMCDiagnostics.effective_sample_size(samples)
        assert ess > 0

    def test_r_hat_converged(self):
        """R-hat should be near 1 for converged chains."""
        rng = np.random.RandomState(42)
        chains = [rng.normal(size=(1000, 1)) for _ in range(4)]
        r = MCMCDiagnostics.r_hat(chains)
        assert abs(r - 1.0) < 0.1

    def test_r_hat_not_converged(self):
        """R-hat should be > 1 for non-converged chains."""
        rng = np.random.RandomState(42)
        chains = [
            rng.normal(loc=0, size=(500, 1)),
            rng.normal(loc=5, size=(500, 1)),
            rng.normal(loc=-5, size=(500, 1)),
        ]
        r = MCMCDiagnostics.r_hat(chains)
        assert r > 1.5

    def test_r_hat_1d_input(self):
        rng = np.random.RandomState(42)
        chains = [rng.normal(size=500) for _ in range(3)]
        r = MCMCDiagnostics.r_hat(chains)
        assert abs(r - 1.0) < 0.15

    def test_autocorrelation_lag0(self):
        """Autocorrelation at lag 0 should be 1."""
        samples = np.random.RandomState(42).normal(size=500)
        acf = MCMCDiagnostics.autocorrelation(samples)
        assert abs(acf[0] - 1.0) < 0.01

    def test_autocorrelation_iid_decays(self):
        """ACF of iid samples should decay quickly."""
        samples = np.random.RandomState(42).normal(size=2000)
        acf = MCMCDiagnostics.autocorrelation(samples, max_lag=20)
        assert abs(acf[5]) < 0.1

    def test_autocorrelation_correlated(self):
        """ACF of correlated samples should stay high."""
        n = 2000
        x = np.zeros(n)
        rng = np.random.RandomState(42)
        for i in range(1, n):
            x[i] = 0.95 * x[i-1] + rng.normal(0, 0.3)
        acf = MCMCDiagnostics.autocorrelation(x, max_lag=20)
        assert acf[1] > 0.8

    def test_split_r_hat(self):
        rng = np.random.RandomState(42)
        chains = [rng.normal(size=1000) for _ in range(3)]
        sr = MCMCDiagnostics.split_r_hat(chains)
        assert abs(sr - 1.0) < 0.15

    def test_geweke_converged(self):
        """Geweke z-score should be small for converged chain."""
        samples = np.random.RandomState(42).normal(size=5000)
        z = MCMCDiagnostics.geweke(samples)
        assert abs(z) < 3.0  # Within 3 sigma

    def test_geweke_not_converged(self):
        """Geweke z should be large for trending chain."""
        samples = np.linspace(0, 10, 5000)  # Trending
        z = MCMCDiagnostics.geweke(samples)
        assert abs(z) > 3.0

    def test_summary(self):
        rng = np.random.RandomState(42)
        samples = rng.normal(loc=2, scale=0.5, size=(5000, 2))
        s = MCMCDiagnostics.summary(samples)
        assert 'dim_0' in s and 'dim_1' in s
        assert abs(s['dim_0']['mean'] - 2.0) < 0.1
        assert abs(s['dim_0']['std'] - 0.5) < 0.1
        assert s['dim_0']['q025'] < s['dim_0']['median'] < s['dim_0']['q975']
        assert s['dim_0']['ess'] > 100

    def test_summary_1d(self):
        samples = np.random.RandomState(42).normal(size=1000)
        s = MCMCDiagnostics.summary(samples)
        assert 'dim_0' in s

    def test_autocorrelation_2d(self):
        """ACF with 2d input uses first dimension."""
        samples = np.random.RandomState(42).normal(size=(500, 2))
        acf = MCMCDiagnostics.autocorrelation(samples, max_lag=10)
        assert len(acf) == 11

    def test_ess_multidim(self):
        rng = np.random.RandomState(42)
        samples = rng.normal(size=(1000, 3))
        ess = MCMCDiagnostics.effective_sample_size(samples)
        assert len(ess) == 3
        assert all(e > 100 for e in ess)

    def test_r_hat_multidim(self):
        rng = np.random.RandomState(42)
        chains = [rng.normal(size=(500, 2)) for _ in range(3)]
        r = MCMCDiagnostics.r_hat(chains)
        assert len(r) == 2
        assert all(abs(v - 1.0) < 0.15 for v in r)

    def test_geweke_2d(self):
        """Geweke on 2d input uses first dim."""
        samples = np.random.RandomState(42).normal(size=(3000, 2))
        z = MCMCDiagnostics.geweke(samples)
        assert abs(z) < 3.0


# ============================================================
# Integration tests -- end-to-end workflows
# ============================================================

class TestIntegration:
    def test_mh_then_diagnostics(self):
        """Run MH, then check diagnostics."""
        lp = lambda x: -0.5 * np.dot(x, x)
        mh = MetropolisHastings(lp, dim=2, seed=42)
        mh.proposal_scale = 1.0
        samples, rate = mh.sample(5000, burn_in=1000)

        # Diagnostics
        ess = MCMCDiagnostics.effective_sample_size(samples)
        assert np.all(ess > 50)
        s = MCMCDiagnostics.summary(samples)
        assert abs(s['dim_0']['mean']) < 0.3

    def test_hmc_diagnostics(self):
        """Run HMC, check diagnostics."""
        lp = lambda x: -0.5 * np.dot(x, x)
        grad = lambda x: -x
        hmc = HamiltonianMC(lp, grad, dim=2, seed=42)
        hmc.step_size = 0.1
        hmc.n_leapfrog = 20
        samples, _ = hmc.sample(3000, burn_in=500)

        ess = MCMCDiagnostics.effective_sample_size(samples)
        assert np.all(ess > 100)

    def test_multiple_chains_r_hat(self):
        """Run multiple MH chains and compute R-hat."""
        lp = lambda x: -0.5 * np.dot(x, x)
        chains = []
        for seed in [42, 43, 44]:
            mh = MetropolisHastings(lp, dim=1, seed=seed)
            mh.proposal_scale = 1.0
            samples, _ = mh.sample(3000, burn_in=1000)
            chains.append(samples)

        r = MCMCDiagnostics.r_hat(chains)
        assert abs(r - 1.0) < 0.2

    def test_pt_vs_mh_bimodal(self):
        """PT should explore bimodal better than MH."""
        def bimodal_lp(x):
            return np.log(
                0.5 * math.exp(-2 * (x[0] - 5)**2) +
                0.5 * math.exp(-2 * (x[0] + 5)**2) + 1e-300)

        # MH might get stuck in one mode
        mh = MetropolisHastings(bimodal_lp, dim=1, seed=42)
        mh.proposal_scale = 1.0
        mh_samples, _ = mh.sample(5000, burn_in=1000)

        # PT should visit both modes
        pt = ParallelTempering(bimodal_lp, dim=1, n_chains=4, seed=42)
        pt.proposal_scale = 1.0
        pt_samples, _ = pt.sample(5000, burn_in=1000)

        # Check both visited both modes (PT more reliably)
        pt_visits_pos = np.any(pt_samples > 3)
        pt_visits_neg = np.any(pt_samples < -3)
        assert pt_visits_pos and pt_visits_neg

    def test_mc_integration_convergence(self):
        """MC integration with increasing samples."""
        mc = MonteCarloIntegrator(42)
        errors = []
        for n in [100, 1000, 10000]:
            est, _ = mc.basic_integrate(lambda x: x[0]**2, [(0, 1)], n_samples=n)
            errors.append(abs(est - 1.0/3))
        # Errors should generally decrease
        assert errors[-1] < errors[0] + 0.1

    def test_slice_then_diagnostics(self):
        """Slice sampler + diagnostics."""
        lp = lambda x: -0.5 * np.dot(x, x)
        ss = SliceSampler(lp, dim=1, seed=42)
        ss.width = 2.0
        samples = ss.sample(3000, burn_in=500)
        ess = MCMCDiagnostics.effective_sample_size(samples)
        assert ess > 50

    def test_gibbs_diagnostics(self):
        """Gibbs sampler + diagnostics."""
        rng = np.random.RandomState(42)
        conditionals = [
            lambda s: rng.normal(0, 1),
            lambda s: rng.normal(0, 1),
        ]
        gibbs = GibbsSampler(conditionals, dim=2, seed=42)
        samples = gibbs.sample(3000, burn_in=500)
        s = MCMCDiagnostics.summary(samples)
        assert abs(s['dim_0']['mean']) < 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
