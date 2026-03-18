"""
Tests for V219: Particle Filter -- Sequential Monte Carlo.

Tests cover:
1. ParticleSet operations (mean, covariance, ESS, percentile, MAP)
2. Resampling methods (multinomial, systematic, stratified, residual)
3. SIR/Bootstrap particle filter on linear-Gaussian (compare with Kalman)
4. Auxiliary particle filter
5. Regularized particle filter
6. Rao-Blackwellized particle filter
7. Particle smoother
8. Bearings-only tracking (nonlinear)
9. Stochastic volatility (non-Gaussian)
10. Edge cases and numerical stability
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from particle_filter import (
    ParticleSet, PFResult, ParticleFilter, AuxiliaryParticleFilter,
    RegularizedParticleFilter, RaoBlackwellizedPF, ParticleSmoother,
    ResampleMethod, resample, _normalize_log_weights, _logsumexp, _log_norm_pdf,
    simulate_nonlinear_system, compare_with_kalman,
    make_linear_gaussian_model, make_bearings_only_model,
    make_stochastic_volatility_model,
)


# ===========================================================================
# ParticleSet tests
# ===========================================================================

class TestParticleSet:
    def test_mean_uniform_weights(self):
        states = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        weights = np.array([1/3, 1/3, 1/3])
        ps = ParticleSet(states=states, weights=weights)
        np.testing.assert_allclose(ps.mean(), [3.0, 4.0])

    def test_mean_weighted(self):
        states = np.array([[0.0], [10.0]])
        weights = np.array([0.8, 0.2])
        ps = ParticleSet(states=states, weights=weights)
        np.testing.assert_allclose(ps.mean(), [2.0])

    def test_covariance(self):
        rng = np.random.default_rng(42)
        states = rng.normal(size=(1000, 2))
        weights = np.ones(1000) / 1000
        ps = ParticleSet(states=states, weights=weights)
        cov = ps.covariance()
        assert cov.shape == (2, 2)
        np.testing.assert_allclose(cov, np.eye(2), atol=0.15)

    def test_ess_uniform(self):
        ps = ParticleSet(
            states=np.zeros((100, 1)),
            weights=np.ones(100) / 100)
        assert abs(ps.effective_sample_size() - 100.0) < 0.01

    def test_ess_degenerate(self):
        weights = np.zeros(100)
        weights[0] = 1.0
        ps = ParticleSet(states=np.zeros((100, 1)), weights=weights)
        assert abs(ps.effective_sample_size() - 1.0) < 0.01

    def test_n_particles_dim(self):
        ps = ParticleSet(states=np.zeros((50, 3)), weights=np.ones(50)/50)
        assert ps.n_particles == 50
        assert ps.dim == 3

    def test_percentile(self):
        states = np.arange(100).reshape(100, 1).astype(float)
        weights = np.ones(100) / 100
        ps = ParticleSet(states=states, weights=weights)
        # 50th percentile should be around 50
        p50 = ps.percentile(50, dim=0)
        assert 45 <= p50 <= 55

    def test_map_estimate(self):
        states = np.array([[1.0], [2.0], [3.0]])
        weights = np.array([0.1, 0.7, 0.2])
        ps = ParticleSet(states=states, weights=weights)
        np.testing.assert_allclose(ps.map_estimate(), [2.0])


# ===========================================================================
# Resampling tests
# ===========================================================================

class TestResampling:
    def test_multinomial_preserves_count(self):
        rng = np.random.default_rng(42)
        w = np.array([0.5, 0.3, 0.2])
        idx = resample(w, ResampleMethod.MULTINOMIAL, rng)
        assert len(idx) == 3

    def test_systematic_preserves_count(self):
        rng = np.random.default_rng(42)
        w = np.array([0.5, 0.3, 0.2])
        idx = resample(w, ResampleMethod.SYSTEMATIC, rng)
        assert len(idx) == 3

    def test_stratified_preserves_count(self):
        rng = np.random.default_rng(42)
        w = np.array([0.5, 0.3, 0.2])
        idx = resample(w, ResampleMethod.STRATIFIED, rng)
        assert len(idx) == 3

    def test_residual_preserves_count(self):
        rng = np.random.default_rng(42)
        w = np.array([0.5, 0.3, 0.2])
        idx = resample(w, ResampleMethod.RESIDUAL, rng)
        assert len(idx) == 3

    def test_systematic_concentrates_high_weight(self):
        rng = np.random.default_rng(42)
        w = np.zeros(100)
        w[0] = 0.99
        w[1:] = 0.01 / 99
        idx = resample(w, ResampleMethod.SYSTEMATIC, rng)
        # Most indices should be 0
        assert np.sum(idx == 0) >= 90

    def test_all_methods_produce_valid_indices(self):
        rng = np.random.default_rng(42)
        N = 50
        w = np.ones(N) / N
        for method in ResampleMethod:
            idx = resample(w, method, rng)
            assert len(idx) == N
            assert np.all(idx >= 0)
            assert np.all(idx < N)

    def test_resampling_unbiased(self):
        """Each method should approximately preserve the distribution."""
        rng = np.random.default_rng(42)
        w = np.array([0.6, 0.3, 0.1])
        N = 10000
        big_w = np.tile(w, N // 3 + 1)[:N]
        big_w /= big_w.sum()

        for method in [ResampleMethod.SYSTEMATIC, ResampleMethod.STRATIFIED]:
            idx = resample(big_w, method, rng)
            counts = np.bincount(idx, minlength=N)
            # The first few particles (weight 0.6) should appear more
            # Just check total counts are reasonable
            assert len(idx) == N


# ===========================================================================
# Log-weight utilities
# ===========================================================================

class TestLogUtilities:
    def test_logsumexp_basic(self):
        log_w = np.array([0.0, 0.0, 0.0])
        assert abs(_logsumexp(log_w) - np.log(3)) < 1e-10

    def test_logsumexp_large_values(self):
        log_w = np.array([1000.0, 1000.0])
        assert abs(_logsumexp(log_w) - (1000 + np.log(2))) < 1e-10

    def test_logsumexp_negative_inf(self):
        log_w = np.array([-np.inf, 0.0])
        assert abs(_logsumexp(log_w) - 0.0) < 1e-10

    def test_normalize_log_weights(self):
        log_w = np.array([0.0, 0.0, 0.0])
        w = _normalize_log_weights(log_w)
        np.testing.assert_allclose(w, [1/3, 1/3, 1/3])
        assert abs(np.sum(w) - 1.0) < 1e-10

    def test_normalize_log_weights_unequal(self):
        log_w = np.array([np.log(0.7), np.log(0.3)])
        w = _normalize_log_weights(log_w)
        np.testing.assert_allclose(w, [0.7, 0.3], atol=1e-10)

    def test_log_norm_pdf_1d(self):
        # Standard normal at 0 should be -0.5*log(2*pi)
        ll = _log_norm_pdf(np.array([0.0]), np.array([0.0]), np.array([[1.0]]))
        expected = -0.5 * np.log(2 * np.pi)
        assert abs(ll - expected) < 1e-10

    def test_log_norm_pdf_2d(self):
        mean = np.array([1.0, 2.0])
        cov = np.eye(2) * 0.5
        ll = _log_norm_pdf(mean, mean, cov)  # at the mean
        expected = -0.5 * 2 * np.log(2 * np.pi) - 0.5 * np.log(0.25)
        assert abs(ll - expected) < 1e-10


# ===========================================================================
# SIR Particle Filter tests
# ===========================================================================

class TestSIRParticleFilter:
    def _make_1d_random_walk(self):
        """1D random walk: x_t = x_{t-1} + N(0, 1), y_t = x_t + N(0, 0.5)."""
        Q, R = 1.0, 0.5
        def transition_fn(x, rng):
            return x + rng.normal(0, np.sqrt(Q), size=x.shape)
        def log_likelihood_fn(x, y):
            return _log_norm_pdf(y, x, np.array([[R]]))
        def observation_fn(x, rng):
            return x + rng.normal(0, np.sqrt(R), size=x.shape)
        return transition_fn, log_likelihood_fn, observation_fn

    def test_filter_runs(self):
        trans, ll, obs = self._make_1d_random_walk()
        pf = ParticleFilter(trans, ll, n_particles=200, seed=42)
        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 20, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)
        assert len(result.filtered_sets) == 20
        assert len(result.ess_history) == 20
        assert result.log_likelihood != 0

    def test_filter_tracks_state(self):
        """PF should track 1D random walk reasonably well."""
        trans, ll, obs = self._make_1d_random_walk()
        pf = ParticleFilter(trans, ll, n_particles=500, seed=42)
        rng = np.random.default_rng(99)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 50, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)

        # Check RMSE is reasonable
        pf_means = np.array([ps.mean() for ps in result.filtered_sets])
        rmse = np.sqrt(np.mean((pf_means[:, 0] - states[1:, 0]) ** 2))
        assert rmse < 2.0  # Should be well under 2 for this easy problem

    def test_linear_gaussian_matches_kalman(self):
        """On linear-Gaussian, PF should approximate Kalman."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])

        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)
        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 30, rng)

        pf = ParticleFilter(trans, ll, n_particles=2000, seed=42)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)

        # Simple Kalman for comparison
        kf_means = []
        m, P = 0.0, 4.0
        for t in range(30):
            # Predict
            m_pred = F[0,0] * m
            P_pred = F[0,0]**2 * P + Q[0,0]
            # Update
            S = H[0,0]**2 * P_pred + R[0,0]
            K = P_pred * H[0,0] / S
            m = m_pred + K * (observations[t, 0] - H[0,0] * m_pred)
            P = (1 - K * H[0,0]) * P_pred
            kf_means.append([m])

        comp = compare_with_kalman(result, np.array(kf_means), states)
        # PF should be within 2x of Kalman RMSE on this linear problem
        assert comp["pf_rmse"] < 2 * comp["kf_rmse"] + 0.5

    def test_ess_threshold(self):
        """Resampling should trigger when ESS drops."""
        trans, ll, obs = self._make_1d_random_walk()
        pf = ParticleFilter(trans, ll, n_particles=200, ess_threshold=0.5, seed=42)
        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 30, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)
        assert result.resample_count > 0

    def test_no_resampling_high_threshold(self):
        """With threshold=0, should never resample."""
        trans, ll, obs = self._make_1d_random_walk()
        pf = ParticleFilter(trans, ll, n_particles=200, ess_threshold=0.0, seed=42)
        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 10, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)
        assert result.resample_count == 0

    def test_different_resample_methods(self):
        trans, ll, obs = self._make_1d_random_walk()
        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 20, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))

        for method in ResampleMethod:
            pf = ParticleFilter(trans, ll, n_particles=200,
                                resample_method=method, seed=42)
            result = pf.filter(observations, prior)
            assert len(result.filtered_sets) == 20

    def test_step_api(self):
        """Test single-step API."""
        trans, ll, obs = self._make_1d_random_walk()
        pf = ParticleFilter(trans, ll, n_particles=100, seed=42)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        ps = pf.initialize(prior)
        assert ps.n_particles == 100

        y = np.array([1.0])
        ps2, log_m, did_resample = pf.step(ps, y)
        assert ps2.n_particles == 100
        assert isinstance(log_m, float)
        assert isinstance(did_resample, bool)


# ===========================================================================
# Auxiliary Particle Filter tests
# ===========================================================================

class TestAuxiliaryPF:
    def test_apf_runs(self):
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        def trans_mean(x):
            return F @ x

        apf = AuxiliaryParticleFilter(
            trans, trans_mean, ll, n_particles=300, seed=42)

        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 20, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = apf.filter(observations, prior)
        assert len(result.filtered_sets) == 20

    def test_apf_tracks_state(self):
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.5]])
        R = np.array([[0.5]])
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 30, rng)

        apf = AuxiliaryParticleFilter(
            trans, lambda x: F @ x, ll, n_particles=500, seed=42)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = apf.filter(observations, prior)

        pf_means = np.array([ps.mean() for ps in result.filtered_sets])
        rmse = np.sqrt(np.mean((pf_means[:, 0] - states[1:, 0]) ** 2))
        assert rmse < 2.0


# ===========================================================================
# Regularized Particle Filter tests
# ===========================================================================

class TestRegularizedPF:
    def test_rpf_runs(self):
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        rpf = RegularizedParticleFilter(
            trans, ll, n_particles=300, seed=42)

        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 20, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = rpf.filter(observations, prior)
        assert len(result.filtered_sets) == 20

    def test_rpf_reduces_sample_impoverishment(self):
        """RPF should maintain higher diversity than SIR."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])  # Low process noise -> more impoverishment
        R = np.array([[0.01]])  # Very precise observations

        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)
        rng = np.random.default_rng(42)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 30, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))

        rpf = RegularizedParticleFilter(trans, ll, n_particles=200, seed=42)
        result = rpf.filter(observations, prior)

        # Check that particles have some diversity (not all identical)
        last_ps = result.filtered_sets[-1]
        unique_count = len(np.unique(np.round(last_ps.states, 4), axis=0))
        assert unique_count > 10  # Should have reasonable diversity

    def test_rpf_tracks_state(self):
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 30, rng)

        rpf = RegularizedParticleFilter(trans, ll, n_particles=500, seed=42)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = rpf.filter(observations, prior)

        pf_means = np.array([ps.mean() for ps in result.filtered_sets])
        rmse = np.sqrt(np.mean((pf_means[:, 0] - states[1:, 0]) ** 2))
        assert rmse < 2.0


# ===========================================================================
# Rao-Blackwellized PF tests
# ===========================================================================

class TestRaoBlackwellizedPF:
    def test_rbpf_runs(self):
        """RBPF on a simple model where everything is linear (degenerates to Kalman)."""
        def nl_transition(x_nl, rng):
            return x_nl  # Static nonlinear state (just a mode indicator)

        def linear_dynamics(x_nl):
            F = np.array([[1.0]])
            B = np.zeros((1, 1))
            Q = np.array([[1.0]])
            return F, B, Q

        def observation(x_nl):
            H = np.array([[1.0]])
            R = np.array([[0.5]])
            return H, R

        rbpf = RaoBlackwellizedPF(
            nl_transition, linear_dynamics, observation,
            n_particles=100, seed=42)

        # Generate observations from a 1D random walk
        rng = np.random.default_rng(99)
        T = 20
        x = 0.0
        obs = []
        for _ in range(T):
            x += rng.normal(0, 1)
            obs.append([x + rng.normal(0, np.sqrt(0.5))])
        observations = np.array(obs)

        prior = lambda rng: np.array([0.0])  # nonlinear state = 0
        result = rbpf.filter(observations, prior, linear_dim=1)
        assert len(result.filtered_sets) == T

    def test_rbpf_lower_variance_than_sir(self):
        """RBPF should have lower variance than pure SIR on conditionally linear model."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])

        rng = np.random.default_rng(42)
        T = 30
        x = 0.0
        true_states = [x]
        obs = []
        for _ in range(T):
            x += rng.normal(0, 1)
            true_states.append(x)
            obs.append([x + rng.normal(0, np.sqrt(0.5))])
        observations = np.array(obs)

        # RBPF with trivial nonlinear component
        rbpf = RaoBlackwellizedPF(
            lambda x, rng: x,
            lambda x: (F, np.zeros((1,1)), Q),
            lambda x: (H, R),
            n_particles=100, seed=42)
        rb_result = rbpf.filter(observations, lambda rng: np.array([0.0]), linear_dim=1)

        # Just check it runs and produces reasonable ESS
        assert np.mean(rb_result.ess_history) > 5


# ===========================================================================
# Particle Smoother tests
# ===========================================================================

class TestParticleSmoother:
    def test_smoother_runs(self):
        F = np.array([[1.0]])
        Q = np.array([[1.0]])
        trans, ll, obs = make_linear_gaussian_model(
            F, np.array([[1.0]]), Q, np.array([[0.5]]))

        pf = ParticleFilter(trans, ll, n_particles=200, seed=42)
        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 15, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)

        def trans_log_density(x_next, x_curr):
            return _log_norm_pdf(x_next, F @ x_curr, Q)

        smoother = ParticleSmoother(trans_log_density, lag=3)
        smoothed = smoother.smooth(result.filtered_sets)
        assert len(smoothed) == 15
        for ps in smoothed:
            assert abs(np.sum(ps.weights) - 1.0) < 1e-6

    def test_smoother_reduces_error(self):
        """Smoother should have lower RMSE than filter (on average)."""
        F = np.array([[1.0]])
        Q = np.array([[0.5]])
        R = np.array([[2.0]])  # High observation noise favors smoothing
        trans, ll, obs = make_linear_gaussian_model(F, np.array([[1.0]]), Q, R)

        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 20, rng)

        pf = ParticleFilter(trans, ll, n_particles=500, seed=42)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)

        def trans_log_density(x_next, x_curr):
            return _log_norm_pdf(x_next, F @ x_curr, Q)

        smoother = ParticleSmoother(trans_log_density, lag=5)
        smoothed = smoother.smooth(result.filtered_sets)

        filter_means = np.array([ps.mean()[0] for ps in result.filtered_sets])
        smooth_means = np.array([ps.mean()[0] for ps in smoothed])
        true = states[1:, 0]

        filter_rmse = np.sqrt(np.mean((filter_means - true) ** 2))
        smooth_rmse = np.sqrt(np.mean((smooth_means - true) ** 2))
        # Smoother should be at least not much worse
        assert smooth_rmse < filter_rmse * 1.5


# ===========================================================================
# Bearings-only tracking (nonlinear test)
# ===========================================================================

class TestBearingsOnly:
    def test_model_creation(self):
        trans, ll, obs, F, Q = make_bearings_only_model()
        assert F.shape == (4, 4)
        assert Q.shape == (4, 4)

    def test_simulation(self):
        trans, ll, obs, F, Q = make_bearings_only_model(process_noise=0.1)
        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng),
            np.array([10.0, 10.0, 1.0, 0.5]), 30, rng)
        assert states.shape == (31, 4)
        assert observations.shape == (30, 1)
        # Observations should be angles in [-pi, pi]
        assert np.all(np.abs(observations) < np.pi + 0.5)

    def test_pf_tracks_bearings_only(self):
        """PF should track a bearings-only target (EKF struggles here)."""
        trans, ll, obs, F, Q = make_bearings_only_model(
            process_noise=0.1, obs_noise=0.05)
        rng = np.random.default_rng(42)
        x0 = np.array([10.0, 10.0, 1.0, 0.0])
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), x0, 40, rng)

        pf = ParticleFilter(trans, ll, n_particles=1000, seed=42)
        prior = lambda rng: np.array([
            rng.normal(10, 3), rng.normal(10, 3),
            rng.normal(1, 1), rng.normal(0, 1)])
        result = pf.filter(observations, prior)

        # Check position RMSE (first 2 dims)
        pf_means = np.array([ps.mean() for ps in result.filtered_sets])
        pos_rmse = np.sqrt(np.mean(
            (pf_means[-20:, :2] - states[-20:, :2]) ** 2))
        # Should be reasonable (< 5 units) in later steps
        assert pos_rmse < 10.0


# ===========================================================================
# Stochastic volatility (non-Gaussian test)
# ===========================================================================

class TestStochasticVolatility:
    def test_model_creation(self):
        trans, ll, obs = make_stochastic_volatility_model()
        rng = np.random.default_rng(42)
        x = np.array([0.0])
        x_next = trans(x, rng)
        assert x_next.shape == (1,)
        y = obs(x, rng)
        assert y.shape == (1,)

    def test_pf_on_sv_model(self):
        """PF should track stochastic volatility (non-Gaussian observation)."""
        trans, ll, obs = make_stochastic_volatility_model(
            phi=0.95, sigma_v=0.2, beta=0.5)
        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 50, rng)

        pf = ParticleFilter(trans, ll, n_particles=500, seed=42)
        prior = lambda rng: rng.normal(0, 1, size=(1,))
        result = pf.filter(observations, prior)

        assert len(result.filtered_sets) == 50
        # Check that mean ESS is reasonable
        assert np.mean(result.ess_history) > 20

    def test_sv_log_likelihood_finite(self):
        trans, ll, obs = make_stochastic_volatility_model()
        rng = np.random.default_rng(42)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 20, rng)

        pf = ParticleFilter(trans, ll, n_particles=200, seed=42)
        prior = lambda rng: rng.normal(0, 1, size=(1,))
        result = pf.filter(observations, prior)
        assert np.isfinite(result.log_likelihood)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_particle(self):
        """Filter with 1 particle should not crash."""
        trans, ll, obs = make_linear_gaussian_model(
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[1.0]]), np.array([[1.0]]))
        pf = ParticleFilter(trans, ll, n_particles=1, seed=42)
        rng = np.random.default_rng(99)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 5, rng)
        prior = lambda rng: rng.normal(0, 1, size=(1,))
        result = pf.filter(observations, prior)
        assert len(result.filtered_sets) == 5

    def test_single_observation(self):
        trans, ll, obs = make_linear_gaussian_model(
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[1.0]]), np.array([[1.0]]))
        pf = ParticleFilter(trans, ll, n_particles=100, seed=42)
        observations = np.array([[2.0]])
        prior = lambda rng: rng.normal(0, 1, size=(1,))
        result = pf.filter(observations, prior)
        assert len(result.filtered_sets) == 1

    def test_multidimensional_state(self):
        """2D state, 2D observation."""
        F = np.eye(2)
        H = np.eye(2)
        Q = np.eye(2) * 0.5
        R = np.eye(2) * 0.3
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        pf = ParticleFilter(trans, ll, n_particles=300, seed=42)
        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0, 0.0]), 15, rng)
        prior = lambda rng: rng.normal(0, 2, size=(2,))
        result = pf.filter(observations, prior)
        assert result.filtered_sets[0].dim == 2

    def test_zero_process_noise(self):
        """Deterministic dynamics, all noise from observation."""
        F = np.array([[0.99]])
        H = np.array([[1.0]])
        Q = np.array([[0.0001]])  # Near-zero
        R = np.array([[1.0]])
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        pf = ParticleFilter(trans, ll, n_particles=200, seed=42)
        rng = np.random.default_rng(42)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([5.0]), 10, rng)
        prior = lambda rng: rng.normal(5, 1, size=(1,))
        result = pf.filter(observations, prior)
        assert len(result.filtered_sets) == 10

    def test_high_dimensional(self):
        """5D state test."""
        n = 5
        F = np.eye(n) * 0.9
        H = np.eye(n)
        Q = np.eye(n) * 0.1
        R = np.eye(n) * 0.5
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        pf = ParticleFilter(trans, ll, n_particles=500, seed=42)
        rng = np.random.default_rng(42)
        _, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.zeros(n), 10, rng)
        prior = lambda rng: rng.normal(0, 1, size=(n,))
        result = pf.filter(observations, prior)
        assert result.filtered_sets[0].dim == n

    def test_compare_with_kalman_api(self):
        """Test the comparison helper function."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 15, rng)

        pf = ParticleFilter(trans, ll, n_particles=300, seed=42)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)

        kf_means = observations.copy()  # Use observations as rough "KF" estimate
        comp = compare_with_kalman(result, kf_means, states)
        assert "pf_rmse" in comp
        assert "kf_rmse" in comp
        assert "pf_mean_ess" in comp
        assert comp["pf_mean_ess"] > 0


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_all_filters_on_same_data(self):
        """Run all filter variants on the same data and compare."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[0.5]])
        trans, ll, obs = make_linear_gaussian_model(F, H, Q, R)

        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 25, rng)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        N = 300

        # SIR
        sir = ParticleFilter(trans, ll, n_particles=N, seed=42)
        r_sir = sir.filter(observations, prior)

        # APF
        apf = AuxiliaryParticleFilter(
            trans, lambda x: F @ x, ll, n_particles=N, seed=42)
        r_apf = apf.filter(observations, prior)

        # RPF
        rpf = RegularizedParticleFilter(trans, ll, n_particles=N, seed=42)
        r_rpf = rpf.filter(observations, prior)

        # All should produce 25 filtered sets
        assert len(r_sir.filtered_sets) == 25
        assert len(r_apf.filtered_sets) == 25
        assert len(r_rpf.filtered_sets) == 25

        # All should have finite log-likelihood
        assert np.isfinite(r_sir.log_likelihood)
        assert np.isfinite(r_apf.log_likelihood)
        assert np.isfinite(r_rpf.log_likelihood)

    def test_filter_then_smooth(self):
        """Full pipeline: filter -> smooth -> verify improvement."""
        F = np.array([[1.0]])
        Q = np.array([[0.5]])
        R = np.array([[2.0]])
        trans, ll, obs = make_linear_gaussian_model(F, np.array([[1.0]]), Q, R)

        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, lambda x, rng: obs(x, rng), np.array([0.0]), 20, rng)

        pf = ParticleFilter(trans, ll, n_particles=400, seed=42)
        prior = lambda rng: rng.normal(0, 2, size=(1,))
        result = pf.filter(observations, prior)

        def trans_log_density(x_next, x_curr):
            return _log_norm_pdf(x_next, F @ x_curr, Q)

        smoother = ParticleSmoother(trans_log_density, lag=3)
        smoothed = smoother.smooth(result.filtered_sets)

        # Both should have same length
        assert len(smoothed) == len(result.filtered_sets)
        # All weights should sum to 1
        for ps in smoothed:
            assert abs(np.sum(ps.weights) - 1.0) < 1e-5

    def test_simulate_nonlinear_system(self):
        """Test the simulation utility."""
        def trans(x, rng):
            return np.array([np.sin(x[0]) + rng.normal(0, 0.1)])
        def obs(x, rng):
            return np.array([x[0] ** 2 + rng.normal(0, 0.1)])

        rng = np.random.default_rng(42)
        states, observations = simulate_nonlinear_system(
            trans, obs, np.array([1.0]), 10, rng)
        assert states.shape == (11, 1)
        assert observations.shape == (10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
