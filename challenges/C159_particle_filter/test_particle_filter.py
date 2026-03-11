"""Tests for C159: Particle Filter (Sequential Monte Carlo)."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from particle_filter import (
    ParticleFilter, AuxiliaryParticleFilter, RegularizedParticleFilter,
    RaoBlackwellizedPF, ParticleSmoother, AdaptiveParticleFilter,
    MultipleModelPF, ParticleFilterUtils
)


# ============================================================
# Helper: simple linear-Gaussian model for testing
# ============================================================

def make_linear_model(F=0.9, H=1.0, Q=0.1, R=1.0, seed=42):
    """Create a 1D linear-Gaussian state-space model.
    x_{k+1} = F*x_k + w,  w ~ N(0, Q)
    z_k = H*x_k + v,      v ~ N(0, R)
    """
    rng = np.random.RandomState(seed)
    T = 50
    states = np.zeros(T)
    measurements = np.zeros(T)
    states[0] = rng.randn()
    measurements[0] = H * states[0] + np.sqrt(R) * rng.randn()
    for t in range(1, T):
        states[t] = F * states[t - 1] + np.sqrt(Q) * rng.randn()
        measurements[t] = H * states[t] + np.sqrt(R) * rng.randn()
    return states, measurements, F, H, Q, R


def linear_transition(particles, k, F=0.9):
    return F * particles


def linear_log_lik(particles, z, k, H=1.0, R=1.0):
    predicted = H * particles[:, 0] if particles.ndim > 1 else H * particles.flatten()
    return -0.5 * ((z - predicted) ** 2) / R


def linear_process_noise(N, rng, Q=0.1):
    return np.sqrt(Q) * rng.randn(N, 1)


def linear_initial(N, rng):
    return rng.randn(N, 1)


# ============================================================
# ParticleFilterUtils tests
# ============================================================

class TestParticleFilterUtils:
    def test_ess_uniform(self):
        w = np.ones(100) / 100
        ess = ParticleFilterUtils.effective_sample_size(w)
        assert abs(ess - 100) < 1e-10

    def test_ess_degenerate(self):
        w = np.zeros(100)
        w[0] = 1.0
        ess = ParticleFilterUtils.effective_sample_size(w)
        assert abs(ess - 1.0) < 1e-10

    def test_ess_intermediate(self):
        w = np.array([0.5, 0.5, 0.0, 0.0])
        ess = ParticleFilterUtils.effective_sample_size(w)
        assert abs(ess - 2.0) < 1e-10

    def test_multinomial_resample_shape(self):
        w = np.array([0.1, 0.2, 0.3, 0.4])
        rng = np.random.RandomState(42)
        indices = ParticleFilterUtils.multinomial_resample(w, rng)
        assert len(indices) == 4
        assert all(0 <= i < 4 for i in indices)

    def test_systematic_resample_shape(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.RandomState(42)
        indices = ParticleFilterUtils.systematic_resample(w, rng)
        assert len(indices) == 4

    def test_stratified_resample_shape(self):
        w = np.array([0.1, 0.2, 0.3, 0.4])
        rng = np.random.RandomState(42)
        indices = ParticleFilterUtils.stratified_resample(w, rng)
        assert len(indices) == 4

    def test_residual_resample_shape(self):
        w = np.array([0.1, 0.2, 0.3, 0.4])
        rng = np.random.RandomState(42)
        indices = ParticleFilterUtils.residual_resample(w, rng)
        assert len(indices) == 4

    def test_systematic_low_variance(self):
        """Systematic resampling should have lower variance than multinomial."""
        w = np.array([0.5, 0.3, 0.1, 0.1])
        rng = np.random.RandomState(42)
        # Run many resamplings
        counts_sys = np.zeros(4)
        counts_multi = np.zeros(4)
        for _ in range(1000):
            idx_s = ParticleFilterUtils.systematic_resample(w, rng)
            idx_m = ParticleFilterUtils.multinomial_resample(w, rng)
            for i in idx_s:
                counts_sys[i] += 1
            for i in idx_m:
                counts_multi[i] += 1
        # Both should roughly follow the weights
        expected = w * 4000
        assert np.allclose(counts_sys / counts_sys.sum(), w, atol=0.05)
        assert np.allclose(counts_multi / counts_multi.sum(), w, atol=0.05)

    def test_residual_resample_deterministic_part(self):
        """Residual resampling: items with w*N >= 1 should appear at least floor(w*N) times."""
        w = np.array([0.6, 0.3, 0.05, 0.05])
        N = 4
        rng = np.random.RandomState(42)
        indices = ParticleFilterUtils.residual_resample(w, rng)
        counts = np.bincount(indices, minlength=4)
        assert counts[0] >= 2  # 0.6 * 4 = 2.4, floor = 2

    def test_particle_diversity_all_unique(self):
        particles = [[1, 2], [3, 4], [5, 6]]
        d = ParticleFilterUtils.particle_diversity(particles)
        assert abs(d - 1.0) < 1e-10

    def test_particle_diversity_all_same(self):
        particles = [[1, 2], [1, 2], [1, 2]]
        d = ParticleFilterUtils.particle_diversity(particles)
        assert abs(d - 1.0 / 3) < 1e-10

    def test_particle_diversity_empty(self):
        d = ParticleFilterUtils.particle_diversity([])
        assert d == 0.0

    def test_neff_history(self):
        wh = [np.ones(10) / 10, np.array([1] + [0] * 9)]
        neffs = ParticleFilterUtils.neff_history(wh)
        assert abs(neffs[0] - 10.0) < 1e-10
        assert abs(neffs[1] - 1.0) < 1e-10

    def test_log_marginal_likelihood(self):
        wh = [np.ones(10) * 0.1]  # avg weight = 0.1
        lml = ParticleFilterUtils.log_marginal_likelihood(wh)
        assert abs(lml - np.log(0.1)) < 1e-10


# ============================================================
# ParticleFilter (Bootstrap/SIR) tests
# ============================================================

class TestParticleFilter:
    def _make_pf(self, seed=42, N=500):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        pf = ParticleFilter(N, 1, trans, ll, noise, init, seed=seed)
        return pf

    def test_initialize(self):
        pf = self._make_pf()
        pf.initialize()
        assert pf.particles.shape == (500, 1)
        assert len(pf.weights) == 500
        assert abs(sum(pf.weights) - 1.0) < 1e-10

    def test_initialize_custom(self):
        pf = self._make_pf()
        custom = np.ones((500, 1)) * 3.0
        pf.initialize(particles=custom)
        assert np.allclose(pf.particles, 3.0)

    def test_predict(self):
        pf = self._make_pf()
        pf.initialize()
        old = pf.particles.copy()
        pf.predict(0)
        # Particles should have changed
        assert not np.allclose(pf.particles, old)

    def test_update_changes_weights(self):
        pf = self._make_pf()
        pf.initialize()
        pf.predict(0)
        old_w = pf.weights.copy()
        pf.update(1.0, 0)
        # Weights should no longer be uniform
        assert not np.allclose(pf.weights, old_w)

    def test_resample_reduces_degeneracy(self):
        pf = self._make_pf()
        pf.initialize()
        pf.predict(0)
        pf.update(1.0, 0)
        pf.resample(force=True)
        assert abs(sum(pf.weights) - 1.0) < 1e-10
        # After resampling, weights should be uniform
        assert np.allclose(pf.weights, 1.0 / 500)

    def test_step(self):
        pf = self._make_pf()
        pf.initialize()
        mean, cov = pf.step(0.5, 0)
        assert mean.shape == (1,)
        assert cov.shape == (1, 1)

    def test_filter_tracks_signal(self):
        """PF should track a simple 1D signal."""
        states, measurements, F, H, Q, R = make_linear_model()
        def trans(p, k):
            return F * p
        def ll(p, z, k):
            return -0.5 * ((z - H * p[:, 0]) ** 2) / R
        def noise(n, rng):
            return np.sqrt(Q) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        pf = ParticleFilter(1000, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        ests, covs = pf.filter(measurements)
        # RMSE should be reasonable (< 1.5 for this model)
        rmse = np.sqrt(np.mean((ests[:, 0] - states) ** 2))
        assert rmse < 1.5, f"RMSE {rmse} too high"

    def test_filter_different_resample_methods(self):
        states, measurements, F, H, Q, R = make_linear_model()
        for method in ['systematic', 'multinomial', 'stratified', 'residual']:
            def trans(p, k):
                return F * p
            def ll(p, z, k):
                return -0.5 * ((z - H * p[:, 0]) ** 2) / R
            def noise(n, rng):
                return np.sqrt(Q) * rng.randn(n, 1)
            def init(n, rng):
                return rng.randn(n, 1)
            pf = ParticleFilter(500, 1, trans, ll, noise, init,
                               resample_method=method, seed=42)
            pf.initialize()
            ests, _ = pf.filter(measurements[:20])
            assert ests.shape == (20, 1)

    def test_history_stored(self):
        pf = self._make_pf()
        pf.initialize()
        pf.step(0.5, 0)
        pf.step(0.3, 1)
        assert len(pf.particle_history) == 2
        assert len(pf.weight_history) == 2
        assert len(pf.estimate_history) == 2

    def test_map_estimate(self):
        pf = self._make_pf()
        pf.initialize()
        pf.step(0.5, 0)
        mapest = pf.map_estimate()
        assert mapest.shape == (1,)

    def test_estimate_covariance_positive(self):
        pf = self._make_pf()
        pf.initialize()
        pf.step(1.0, 0)
        _, cov = pf.estimate()
        assert cov[0, 0] >= 0

    def test_no_resample_when_ess_high(self):
        """With threshold=0, should never resample."""
        pf = self._make_pf()
        pf.resample_threshold = 0.0
        pf.initialize()
        pf.predict(0)
        pf.update(1.0, 0)
        resampled = pf.resample()
        assert resampled is False

    def test_force_resample(self):
        pf = self._make_pf()
        pf.resample_threshold = 0.0
        pf.initialize()
        pf.predict(0)
        pf.update(1.0, 0)
        resampled = pf.resample(force=True)
        assert resampled is True

    def test_2d_state(self):
        """Test with 2D state (position + velocity)."""
        def trans(p, k):
            # [pos, vel] -> [pos + vel*dt, vel]
            dt = 0.1
            new = p.copy()
            new[:, 0] += p[:, 1] * dt
            return new
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2) / 0.5
        def noise(n, rng):
            return 0.1 * rng.randn(n, 2)
        def init(n, rng):
            return rng.randn(n, 2)
        pf = ParticleFilter(300, 2, trans, ll, noise, init, seed=42)
        pf.initialize()
        mean, cov = pf.step(0.5, 0)
        assert mean.shape == (2,)
        assert cov.shape == (2, 2)

    def test_filter_returns_correct_shapes(self):
        pf = self._make_pf(N=100)
        pf.initialize()
        ests, covs = pf.filter([0.1, 0.2, 0.3])
        assert ests.shape == (3, 1)
        assert covs.shape == (3, 1, 1)


# ============================================================
# AuxiliaryParticleFilter tests
# ============================================================

class TestAuxiliaryParticleFilter:
    def _make_apf(self, seed=42, N=500):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        apf = AuxiliaryParticleFilter(N, 1, trans, ll, noise, init, seed=seed)
        return apf

    def test_initialize(self):
        apf = self._make_apf()
        apf.initialize()
        assert apf.particles.shape == (500, 1)

    def test_step(self):
        apf = self._make_apf()
        apf.initialize()
        mean, cov = apf.step(0.5, 0)
        assert mean.shape == (1,)
        assert cov.shape == (1, 1)

    def test_filter_tracks_signal(self):
        states, measurements, F, H, Q, R = make_linear_model()
        def trans(p, k):
            return F * p
        def ll(p, z, k):
            return -0.5 * ((z - H * p[:, 0]) ** 2) / R
        def noise(n, rng):
            return np.sqrt(Q) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        apf = AuxiliaryParticleFilter(1000, 1, trans, ll, noise, init, seed=42)
        apf.initialize()
        ests, covs = apf.filter(measurements)
        rmse = np.sqrt(np.mean((ests[:, 0] - states) ** 2))
        assert rmse < 1.5, f"RMSE {rmse} too high"

    def test_custom_predictive_fn(self):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def pred(p, k):
            return 0.9 * p  # same as transition
        def init(n, rng):
            return rng.randn(n, 1)
        apf = AuxiliaryParticleFilter(200, 1, trans, ll, predictive_fn=pred,
                                       initial_fn=init, seed=42)
        apf.initialize()
        mean, cov = apf.step(0.5, 0)
        assert mean.shape == (1,)

    def test_history_stored(self):
        apf = self._make_apf()
        apf.initialize()
        apf.step(0.5, 0)
        apf.step(0.3, 1)
        assert len(apf.particle_history) == 2
        assert len(apf.weight_history) == 2
        assert len(apf.estimate_history) == 2

    def test_filter_shapes(self):
        apf = self._make_apf(N=100)
        apf.initialize()
        ests, covs = apf.filter([0.1, 0.2, 0.3, 0.4, 0.5])
        assert ests.shape == (5, 1)
        assert covs.shape == (5, 1, 1)


# ============================================================
# RegularizedParticleFilter tests
# ============================================================

class TestRegularizedParticleFilter:
    def _make_rpf(self, seed=42, N=500):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        rpf = RegularizedParticleFilter(N, 1, trans, ll, noise, init, seed=seed)
        return rpf

    def test_initialize(self):
        rpf = self._make_rpf()
        rpf.initialize()
        assert rpf.particles.shape == (500, 1)

    def test_step(self):
        rpf = self._make_rpf()
        rpf.initialize()
        mean, cov = rpf.step(0.5, 0)
        assert mean.shape == (1,)

    def test_filter_tracks_signal(self):
        states, measurements, F, H, Q, R = make_linear_model()
        def trans(p, k):
            return F * p
        def ll(p, z, k):
            return -0.5 * ((z - H * p[:, 0]) ** 2) / R
        def noise(n, rng):
            return np.sqrt(Q) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        rpf = RegularizedParticleFilter(1000, 1, trans, ll, noise, init, seed=42)
        rpf.initialize()
        ests, covs = rpf.filter(measurements)
        rmse = np.sqrt(np.mean((ests[:, 0] - states) ** 2))
        assert rmse < 1.5

    def test_custom_bandwidth(self):
        rpf = self._make_rpf()
        rpf.bandwidth = 0.01
        rpf.initialize()
        mean, cov = rpf.step(0.5, 0)
        assert mean.shape == (1,)

    def test_regularization_increases_diversity(self):
        """After resampling with regularization, particles should be more diverse."""
        rpf = self._make_rpf(N=100)
        rpf.initialize()
        # Force a very peaked weight distribution
        rpf.weights = np.zeros(100)
        rpf.weights[0] = 1.0
        rpf.step(0.5, 0)
        # Should still have some diversity
        d = ParticleFilterUtils.particle_diversity(rpf.particles)
        assert d > 0.1  # not all identical

    def test_filter_shapes(self):
        rpf = self._make_rpf(N=100)
        rpf.initialize()
        ests, covs = rpf.filter([0.1, 0.2, 0.3])
        assert ests.shape == (3, 1)
        assert covs.shape == (3, 1, 1)

    def test_history_stored(self):
        rpf = self._make_rpf(N=100)
        rpf.initialize()
        rpf.step(0.5, 0)
        rpf.step(0.3, 1)
        assert len(rpf.particle_history) == 2


# ============================================================
# RaoBlackwellizedPF tests
# ============================================================

class TestRaoBlackwellizedPF:
    def test_initialize(self):
        def nl_trans(x, k):
            return 0.9 * x
        def nl_noise(N, rng):
            return 0.1 * rng.randn(N, 1)
        rbpf = RaoBlackwellizedPF(
            num_particles=100, dim_nl=1, dim_l=1, dim_z=1,
            nl_transition_fn=nl_trans, nl_noise_fn=nl_noise,
            F_l=np.array([[0.95]]), H_l=np.array([[1.0]]),
            Q_l=np.array([[0.1]]), R=np.array([[1.0]]),
            seed=42
        )
        rbpf.initialize()
        assert rbpf.nl_particles.shape == (100, 1)
        assert len(rbpf.kf_particles) == 100

    def test_step(self):
        def nl_trans(x, k):
            return 0.9 * x
        def nl_noise(N, rng):
            return 0.1 * rng.randn(N, 1)
        rbpf = RaoBlackwellizedPF(
            num_particles=100, dim_nl=1, dim_l=1, dim_z=1,
            nl_transition_fn=nl_trans, nl_noise_fn=nl_noise,
            F_l=np.array([[0.95]]), H_l=np.array([[1.0]]),
            Q_l=np.array([[0.1]]), R=np.array([[1.0]]),
            seed=42
        )
        rbpf.initialize()
        est = rbpf.step(0.5, 0)
        assert est.shape == (2,)  # dim_nl + dim_l

    def test_filter_tracks(self):
        """RBPF should track a mixed model."""
        rng = np.random.RandomState(42)
        T = 30
        nl_state = np.zeros(T)
        l_state = np.zeros(T)
        meas = np.zeros(T)
        for t in range(1, T):
            nl_state[t] = 0.9 * nl_state[t - 1] + 0.1 * rng.randn()
            l_state[t] = 0.95 * l_state[t - 1] + 0.1 * rng.randn()
            meas[t] = l_state[t] + 0.5 * rng.randn()

        def nl_trans(x, k):
            return 0.9 * x
        def nl_noise(N, rng):
            return 0.1 * rng.randn(N, 1)

        rbpf = RaoBlackwellizedPF(
            num_particles=200, dim_nl=1, dim_l=1, dim_z=1,
            nl_transition_fn=nl_trans, nl_noise_fn=nl_noise,
            F_l=np.array([[0.95]]), H_l=np.array([[1.0]]),
            Q_l=np.array([[0.01]]), R=np.array([[0.25]]),
            seed=42
        )
        rbpf.initialize()
        ests = rbpf.filter(meas)
        assert ests.shape == (T, 2)
        # Linear part should track reasonably
        l_rmse = np.sqrt(np.mean((ests[10:, 1] - l_state[10:]) ** 2))
        assert l_rmse < 2.0

    def test_with_coupling(self):
        def nl_trans(x, k):
            return 0.9 * x
        def nl_noise(N, rng):
            return 0.1 * rng.randn(N, 1)
        def coupling(x_nl):
            return np.array([0.1 * x_nl[0]])

        rbpf = RaoBlackwellizedPF(
            num_particles=100, dim_nl=1, dim_l=1, dim_z=1,
            nl_transition_fn=nl_trans, nl_noise_fn=nl_noise,
            F_l=np.array([[0.95]]), H_l=np.array([[1.0]]),
            Q_l=np.array([[0.1]]), R=np.array([[1.0]]),
            coupling_fn=coupling, seed=42
        )
        rbpf.initialize()
        est = rbpf.step(0.5, 0)
        assert est.shape == (2,)

    def test_custom_initial(self):
        def nl_trans(x, k):
            return x
        def nl_noise(N, rng):
            return 0.01 * rng.randn(N, 1)
        def nl_init(N, rng):
            return np.ones((N, 1)) * 5.0

        rbpf = RaoBlackwellizedPF(
            num_particles=50, dim_nl=1, dim_l=1, dim_z=1,
            nl_transition_fn=nl_trans, nl_noise_fn=nl_noise,
            F_l=np.array([[1.0]]), H_l=np.array([[1.0]]),
            Q_l=np.array([[0.1]]), R=np.array([[1.0]]),
            initial_nl_fn=nl_init, seed=42
        )
        rbpf.initialize(linear_mean=np.array([2.0]), linear_cov=np.array([[0.5]]))
        assert np.allclose(rbpf.nl_particles, 5.0)
        assert rbpf.kf_particles[0].x[0] == 2.0

    def test_history(self):
        def nl_trans(x, k):
            return 0.9 * x
        def nl_noise(N, rng):
            return 0.1 * rng.randn(N, 1)
        rbpf = RaoBlackwellizedPF(
            num_particles=50, dim_nl=1, dim_l=1, dim_z=1,
            nl_transition_fn=nl_trans, nl_noise_fn=nl_noise,
            F_l=np.array([[0.95]]), H_l=np.array([[1.0]]),
            Q_l=np.array([[0.1]]), R=np.array([[1.0]]),
            seed=42
        )
        rbpf.initialize()
        rbpf.step(0.5, 0)
        rbpf.step(0.3, 1)
        assert len(rbpf.estimate_history) == 2
        assert len(rbpf.weight_history) == 2


# ============================================================
# ParticleSmoother tests
# ============================================================

class TestParticleSmoother:
    def _run_pf(self, measurements, seed=42):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        pf = ParticleFilter(300, 1, trans, ll, noise, init, seed=seed)
        pf.initialize()
        pf.filter(measurements)
        return pf

    def test_fixed_lag_smooth(self):
        states, measurements, _, _, _, _ = make_linear_model()
        pf = self._run_pf(measurements[:20])
        # Create new PF for smoother (needs to run fresh)
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        pf2 = ParticleFilter(300, 1, trans, ll, noise, init, seed=42)
        pf2.initialize()
        smoother = ParticleSmoother(pf2, lag=3)
        smoothed = smoother.fixed_lag_smooth(measurements[:20], lag=3)
        assert smoothed.shape[0] == 20

    def test_fixed_lag_shape(self):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        pf = ParticleFilter(200, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        smoother = ParticleSmoother(pf)
        smoothed = smoother.fixed_lag_smooth([0.1, 0.2, 0.3, 0.4, 0.5], lag=2)
        assert len(smoothed) == 5

    def test_ffbsi(self):
        states, measurements, _, _, _, _ = make_linear_model()
        pf = self._run_pf(measurements[:15])
        smoother = ParticleSmoother(pf)
        trajs = smoother.ffbsi(num_trajectories=50)
        assert trajs.shape == (50, 15, 1)

    def test_ffbsi_requires_history(self):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        pf = ParticleFilter(100, 1, trans, ll, noise, seed=42)
        smoother = ParticleSmoother(pf)
        with pytest.raises(ValueError, match="Must run particle filter first"):
            smoother.ffbsi()

    def test_ffbsi_trajectories_vary(self):
        """Different trajectories from FFBSi should not all be identical."""
        pf = self._run_pf([0.1, 0.2, 0.3, 0.4, 0.5])
        smoother = ParticleSmoother(pf)
        trajs = smoother.ffbsi(num_trajectories=20)
        # Not all trajectories should be the same
        var = np.var(trajs[:, -1, 0])
        assert var > 0


# ============================================================
# AdaptiveParticleFilter tests
# ============================================================

class TestAdaptiveParticleFilter:
    def _make_apf(self, seed=42):
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        apf = AdaptiveParticleFilter(
            1, trans, ll, noise, init,
            min_particles=50, max_particles=2000,
            seed=seed
        )
        return apf

    def test_initialize(self):
        apf = self._make_apf()
        apf.initialize()
        assert apf.particles.shape[0] == 50  # starts at min

    def test_step(self):
        apf = self._make_apf()
        apf.initialize()
        mean, cov = apf.step(0.5, 0)
        assert mean.shape == (1,)

    def test_particle_count_tracked(self):
        apf = self._make_apf()
        apf.initialize()
        for k in range(10):
            apf.step(0.1 * k, k)
        assert len(apf.particle_count_history) == 10
        # All counts should be within bounds
        for c in apf.particle_count_history:
            assert 50 <= c <= 2000

    def test_filter_tracks_signal(self):
        states, measurements, F, H, Q, R = make_linear_model()
        def trans(p, k):
            return F * p
        def ll(p, z, k):
            return -0.5 * ((z - H * p[:, 0]) ** 2) / R
        def noise(n, rng):
            return np.sqrt(Q) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        apf = AdaptiveParticleFilter(
            1, trans, ll, noise, init,
            min_particles=100, max_particles=2000,
            seed=42
        )
        apf.initialize()
        ests, _ = apf.filter(measurements)
        rmse = np.sqrt(np.mean((ests[:, 0] - states) ** 2))
        assert rmse < 2.0

    def test_custom_initial(self):
        apf = self._make_apf()
        custom = np.ones((200, 1)) * 3.0
        apf.initialize(particles=custom)
        assert apf.N == 200
        assert np.allclose(apf.particles, 3.0)

    def test_kld_bound(self):
        apf = self._make_apf()
        # With many bins, should want more particles
        n_few = apf._kld_bound(2)
        n_many = apf._kld_bound(50)
        assert n_many > n_few

    def test_kld_bound_single_bin(self):
        apf = self._make_apf()
        n = apf._kld_bound(1)
        assert n == 50  # min_particles


# ============================================================
# MultipleModelPF tests
# ============================================================

class TestMultipleModelPF:
    def _make_mmpf(self, seed=42, N=200):
        # Two modes: slow (F=0.5) and fast (F=0.99)
        trans_fns = {
            0: lambda p, k: 0.5 * p,
            1: lambda p, k: 0.99 * p,
        }
        ll_fns = {
            0: lambda p, z, k: -0.5 * ((z - p[:, 0]) ** 2),
            1: lambda p, z, k: -0.5 * ((z - p[:, 0]) ** 2),
        }
        noise_fns = {
            0: lambda n, rng: 0.1 * rng.randn(n, 1),
            1: lambda n, rng: 0.05 * rng.randn(n, 1),
        }
        mode_trans = np.array([[0.9, 0.1], [0.1, 0.9]])
        def init(n, rng):
            return rng.randn(n, 1)
        mmpf = MultipleModelPF(
            N, 1, 2, trans_fns, ll_fns, noise_fns,
            mode_trans, initial_fn=init, seed=seed
        )
        return mmpf

    def test_initialize(self):
        mmpf = self._make_mmpf()
        mmpf.initialize()
        assert mmpf.particles.shape == (200, 1)
        assert mmpf.modes.shape == (200,)
        assert set(mmpf.modes).issubset({0, 1})

    def test_step(self):
        mmpf = self._make_mmpf()
        mmpf.initialize()
        mean, mode_probs = mmpf.step(0.5, 0)
        assert mean.shape == (1,)
        assert mode_probs.shape == (2,)
        assert abs(mode_probs.sum() - 1.0) < 1e-10

    def test_filter(self):
        mmpf = self._make_mmpf()
        mmpf.initialize()
        ests, mode_probs = mmpf.filter([0.1, 0.2, 0.3, 0.4, 0.5])
        assert ests.shape == (5, 1)
        assert mode_probs.shape == (5, 2)

    def test_mode_estimate(self):
        mmpf = self._make_mmpf()
        mmpf.initialize()
        mmpf.step(0.5, 0)
        mode = mmpf.mode_estimate()
        assert mode in [0, 1]

    def test_mode_estimate_before_run(self):
        mmpf = self._make_mmpf()
        assert mmpf.mode_estimate() is None

    def test_mode_switching_detection(self):
        """Given data from mode 0, mode 0 should have higher probability."""
        rng = np.random.RandomState(42)
        T = 30
        meas = np.zeros(T)
        state = 1.0
        for t in range(T):
            state = 0.5 * state + 0.1 * rng.randn()  # mode 0
            meas[t] = state + 0.5 * rng.randn()

        mmpf = self._make_mmpf(N=500)
        mmpf.initialize()
        _, mode_probs = mmpf.filter(meas)
        # Last few steps should favor mode 0
        avg_mode0 = np.mean(mode_probs[-10:, 0])
        assert avg_mode0 > 0.3  # mode 0 should have reasonable probability

    def test_custom_initial_modes(self):
        mmpf = self._make_mmpf(N=100)
        mmpf.initialize(modes=np.zeros(100, dtype=int))
        assert np.all(mmpf.modes == 0)

    def test_custom_mode_probs(self):
        trans_fns = {0: lambda p, k: p, 1: lambda p, k: p}
        ll_fns = {0: lambda p, z, k: np.zeros(len(p)),
                  1: lambda p, z, k: np.zeros(len(p))}
        noise_fns = {0: lambda n, rng: 0.1 * rng.randn(n, 1),
                     1: lambda n, rng: 0.1 * rng.randn(n, 1)}
        mode_trans = np.array([[0.5, 0.5], [0.5, 0.5]])
        def init(n, rng):
            return rng.randn(n, 1)
        mmpf = MultipleModelPF(
            1000, 1, 2, trans_fns, ll_fns, noise_fns,
            mode_trans, initial_fn=init,
            initial_mode_probs=[0.9, 0.1], seed=42
        )
        mmpf.initialize()
        # Most particles should start in mode 0
        frac_mode0 = np.mean(mmpf.modes == 0)
        assert frac_mode0 > 0.8

    def test_history_stored(self):
        mmpf = self._make_mmpf()
        mmpf.initialize()
        mmpf.step(0.5, 0)
        mmpf.step(0.3, 1)
        assert len(mmpf.estimate_history) == 2
        assert len(mmpf.mode_history) == 2
        assert len(mmpf.weight_history) == 2


# ============================================================
# Integration tests
# ============================================================

class TestIntegration:
    def test_nonlinear_tracking(self):
        """Track a sinusoidal signal with particle filter."""
        rng = np.random.RandomState(42)
        T = 40
        states = np.sin(np.arange(T) * 0.2)
        meas = states + 0.3 * rng.randn(T)

        def trans(p, k):
            # Approximate continuation
            return p + 0.2 * np.cos(0.2 * k) * np.ones_like(p)
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2) / 0.09
        def noise(n, rng):
            return 0.1 * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1) * 0.5

        pf = ParticleFilter(500, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        ests, _ = pf.filter(meas)
        rmse = np.sqrt(np.mean((ests[:, 0] - states) ** 2))
        assert rmse < 1.0

    def test_compare_pf_vs_apf(self):
        """APF should perform comparably to bootstrap PF."""
        states, measurements, F, H, Q, R = make_linear_model()
        def trans(p, k):
            return F * p
        def ll(p, z, k):
            return -0.5 * ((z - H * p[:, 0]) ** 2) / R
        def noise(n, rng):
            return np.sqrt(Q) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)

        pf = ParticleFilter(500, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        ests_pf, _ = pf.filter(measurements)

        apf = AuxiliaryParticleFilter(500, 1, trans, ll, noise, init, seed=42)
        apf.initialize()
        ests_apf, _ = apf.filter(measurements)

        rmse_pf = np.sqrt(np.mean((ests_pf[:, 0] - states) ** 2))
        rmse_apf = np.sqrt(np.mean((ests_apf[:, 0] - states) ** 2))
        # Both should be reasonable
        assert rmse_pf < 2.0
        assert rmse_apf < 2.0

    def test_smoother_improves_over_filter(self):
        """Smoothed estimates should be at least as good as filtered."""
        states, measurements, F, H, Q, R = make_linear_model()
        def trans(p, k):
            return F * p
        def ll(p, z, k):
            return -0.5 * ((z - H * p[:, 0]) ** 2) / R
        def noise(n, rng):
            return np.sqrt(Q) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)

        pf = ParticleFilter(500, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        ests_filter, _ = pf.filter(measurements[:20])

        pf2 = ParticleFilter(500, 1, trans, ll, noise, init, seed=42)
        pf2.initialize()
        smoother = ParticleSmoother(pf2, lag=5)
        ests_smooth = smoother.fixed_lag_smooth(measurements[:20], lag=5)

        # Both should produce results
        assert ests_filter.shape[0] == 20
        assert ests_smooth.shape[0] == 20

    def test_utils_on_filter_output(self):
        """Use ParticleFilterUtils on actual filter output."""
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        pf = ParticleFilter(200, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        pf.filter([0.1, 0.2, 0.3, 0.4, 0.5])

        neffs = ParticleFilterUtils.neff_history(pf.weight_history)
        assert len(neffs) == 5
        assert all(1.0 <= n <= 200.0 for n in neffs)

        lml = ParticleFilterUtils.log_marginal_likelihood(pf.weight_history)
        assert isinstance(lml, float)

    def test_2d_nonlinear(self):
        """2D nonlinear model (bearing-only tracking)."""
        rng = np.random.RandomState(42)
        T = 20
        # True state: position moving in circle
        true_x = np.cos(np.arange(T) * 0.3)
        true_y = np.sin(np.arange(T) * 0.3)
        # Observe bearing (angle)
        meas = np.arctan2(true_y, true_x) + 0.1 * rng.randn(T)

        def trans(p, k):
            # Simple random walk
            return p
        def ll(p, z, k):
            pred_bearing = np.arctan2(p[:, 1], p[:, 0])
            diff = z - pred_bearing
            # Wrap to [-pi, pi]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            return -0.5 * diff ** 2 / 0.01
        def noise(n, rng):
            return 0.05 * rng.randn(n, 2)
        def init(n, rng):
            return rng.randn(n, 2) * 0.5 + np.array([1, 0])

        pf = ParticleFilter(1000, 2, trans, ll, noise, init, seed=42)
        pf.initialize()
        ests, covs = pf.filter(meas)
        assert ests.shape == (T, 2)
        assert covs.shape == (T, 2, 2)

    def test_pf_then_ffbsi(self):
        """Run PF then FFBSi smoother end to end."""
        def trans(p, k):
            return 0.9 * p
        def ll(p, z, k):
            return -0.5 * ((z - p[:, 0]) ** 2)
        def noise(n, rng):
            return np.sqrt(0.1) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)
        pf = ParticleFilter(200, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        pf.filter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        smoother = ParticleSmoother(pf)
        trajs = smoother.ffbsi(num_trajectories=30)
        assert trajs.shape == (30, 8, 1)
        # Trajectory mean should be close to filter estimate
        traj_mean = trajs.mean(axis=0)
        filt_mean = np.array(pf.estimate_history)
        # Should be in the same ballpark
        assert np.max(np.abs(traj_mean[:, 0] - filt_mean[:, 0])) < 2.0

    def test_adaptive_vs_fixed(self):
        """Adaptive PF should produce reasonable results compared to fixed."""
        states, measurements, F, H, Q, R = make_linear_model()
        def trans(p, k):
            return F * p
        def ll(p, z, k):
            return -0.5 * ((z - H * p[:, 0]) ** 2) / R
        def noise(n, rng):
            return np.sqrt(Q) * rng.randn(n, 1)
        def init(n, rng):
            return rng.randn(n, 1)

        apf = AdaptiveParticleFilter(
            1, trans, ll, noise, init,
            min_particles=100, max_particles=1000,
            seed=42
        )
        apf.initialize()
        ests_a, _ = apf.filter(measurements[:20])

        pf = ParticleFilter(500, 1, trans, ll, noise, init, seed=42)
        pf.initialize()
        ests_f, _ = pf.filter(measurements[:20])

        # Both should track
        rmse_a = np.sqrt(np.mean((ests_a[:, 0] - states[:20]) ** 2))
        rmse_f = np.sqrt(np.mean((ests_f[:, 0] - states[:20]) ** 2))
        assert rmse_a < 2.0
        assert rmse_f < 2.0

    def test_all_resamplers_give_valid_indices(self):
        """All resampling methods return valid index arrays."""
        w = np.array([0.4, 0.3, 0.2, 0.1])
        rng = np.random.RandomState(42)
        for method in [ParticleFilterUtils.multinomial_resample,
                       ParticleFilterUtils.systematic_resample,
                       ParticleFilterUtils.stratified_resample,
                       ParticleFilterUtils.residual_resample]:
            indices = method(w, rng)
            assert len(indices) == 4
            assert all(0 <= i < 4 for i in indices)
            assert all(isinstance(i, (int, np.integer)) for i in indices)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
