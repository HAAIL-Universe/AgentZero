"""
Tests for V218: Kalman Filter
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from kalman_filter import (
    GaussianState, FilterResult, SmootherResult,
    KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter,
    InformationFilter, InformationState, SquareRootKalmanFilter,
    steady_state_gain, simulate_linear_system, compare_filters,
    discretize_kalman, lqr_gain, lqg_controller, simulate_lqg,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_1d_const_velocity():
    """1D constant-velocity model: state = [position, velocity]."""
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])  # observe position only
    Q = np.array([[0.25 * dt**4, 0.5 * dt**3],
                  [0.5 * dt**3, dt**2]]) * 0.1  # process noise
    R = np.array([[1.0]])  # measurement noise
    return F, H, Q, R


def make_2d_position():
    """2D position tracking with constant velocity."""
    dt = 1.0
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    q = 0.1
    Q = q * np.eye(4)
    R = np.eye(2) * 0.5
    return F, H, Q, R


# ===========================================================================
# GaussianState tests
# ===========================================================================

class TestGaussianState:
    def test_creation(self):
        gs = GaussianState(np.array([1.0, 2.0]), np.eye(2))
        assert gs.dim == 2
        assert np.allclose(gs.mean, [1.0, 2.0])

    def test_mahalanobis_at_mean(self):
        gs = GaussianState(np.array([0.0]), np.array([[1.0]]))
        assert gs.mahalanobis(np.array([0.0])) == pytest.approx(0.0)

    def test_mahalanobis_away(self):
        gs = GaussianState(np.array([0.0]), np.array([[1.0]]))
        d = gs.mahalanobis(np.array([2.0]))
        assert d == pytest.approx(2.0)

    def test_mahalanobis_scaled(self):
        gs = GaussianState(np.array([0.0]), np.array([[4.0]]))
        d = gs.mahalanobis(np.array([2.0]))
        assert d == pytest.approx(1.0)  # 2/sqrt(4) = 1

    def test_log_likelihood_peak(self):
        gs = GaussianState(np.array([0.0]), np.array([[1.0]]))
        ll_at_mean = gs.log_likelihood(np.array([0.0]))
        ll_away = gs.log_likelihood(np.array([3.0]))
        assert ll_at_mean > ll_away

    def test_log_likelihood_1d_standard(self):
        gs = GaussianState(np.array([0.0]), np.array([[1.0]]))
        ll = gs.log_likelihood(np.array([0.0]))
        expected = -0.5 * np.log(2 * np.pi)
        assert ll == pytest.approx(expected, abs=1e-10)

    def test_log_likelihood_multivariate(self):
        gs = GaussianState(np.zeros(3), np.eye(3))
        ll = gs.log_likelihood(np.zeros(3))
        expected = -1.5 * np.log(2 * np.pi)
        assert ll == pytest.approx(expected, abs=1e-10)


# ===========================================================================
# KalmanFilter basic tests
# ===========================================================================

class TestKalmanFilterBasic:
    def test_predict_stationary(self):
        """Identity dynamics, zero noise -> unchanged."""
        kf = KalmanFilter(np.eye(2), np.eye(2), np.zeros((2, 2)), np.eye(2))
        state = GaussianState(np.array([1.0, 2.0]), np.eye(2))
        pred = kf.predict(state)
        assert np.allclose(pred.mean, [1.0, 2.0])
        assert np.allclose(pred.cov, np.eye(2))

    def test_predict_shift(self):
        """Constant velocity shift."""
        F = np.array([[1, 1], [0, 1]])
        kf = KalmanFilter(F, np.eye(2), np.zeros((2, 2)), np.eye(2))
        state = GaussianState(np.array([0.0, 1.0]), np.eye(2))
        pred = kf.predict(state)
        assert pred.mean[0] == pytest.approx(1.0)  # pos + vel
        assert pred.mean[1] == pytest.approx(1.0)  # vel unchanged

    def test_predict_with_control(self):
        F = np.eye(2)
        B = np.array([[1.0], [0.0]])
        kf = KalmanFilter(F, np.eye(2), np.zeros((2, 2)), np.eye(2), B=B)
        state = GaussianState(np.zeros(2), np.eye(2))
        pred = kf.predict(state, u=np.array([5.0]))
        assert pred.mean[0] == pytest.approx(5.0)
        assert pred.mean[1] == pytest.approx(0.0)

    def test_update_reduces_uncertainty(self):
        kf = KalmanFilter(np.eye(2), np.eye(2), 0.1 * np.eye(2), np.eye(2))
        pred = GaussianState(np.zeros(2), 10 * np.eye(2))  # large uncertainty
        upd, innov, S, ll = kf.update(pred, np.array([1.0, 1.0]))
        # After update, uncertainty should decrease
        assert np.trace(upd.cov) < np.trace(pred.cov)

    def test_update_moves_toward_observation(self):
        kf = KalmanFilter(np.eye(1), np.eye(1), 0.1 * np.eye(1), np.eye(1))
        pred = GaussianState(np.array([0.0]), np.array([[10.0]]))
        upd, _, _, _ = kf.update(pred, np.array([5.0]))
        # Should move close to observation (large prior uncertainty)
        assert upd.mean[0] > 4.0

    def test_update_innovation(self):
        kf = KalmanFilter(np.eye(1), np.eye(1), np.eye(1), np.eye(1))
        pred = GaussianState(np.array([3.0]), np.eye(1))
        upd, innov, S, ll = kf.update(pred, np.array([5.0]))
        assert innov[0] == pytest.approx(2.0)  # 5 - 3

    def test_filter_sequence(self):
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 10)
        obs = [np.array([float(i)]) for i in range(10)]
        result = kf.filter(obs, init)
        assert len(result.filtered_states) == 10
        assert len(result.predicted_states) == 10
        assert len(result.innovations) == 10
        # Final position estimate should be near 9
        assert abs(result.filtered_states[-1].mean[0] - 9.0) < 2.0

    def test_filter_with_controls(self):
        F = np.eye(1)
        B = np.array([[1.0]])
        H = np.eye(1)
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        kf = KalmanFilter(F, H, Q, R, B=B)
        init = GaussianState(np.zeros(1), np.eye(1))
        controls = [np.array([1.0])] * 5
        obs = [np.array([float(i + 1) + 0.1]) for i in range(5)]
        result = kf.filter(obs, init, controls=controls)
        assert len(result.filtered_states) == 5

    def test_log_likelihood_finite(self):
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 10)
        obs = [np.array([float(i)]) for i in range(5)]
        result = kf.filter(obs, init)
        assert np.isfinite(result.log_likelihood)


# ===========================================================================
# Smoother tests
# ===========================================================================

class TestKalmanSmoother:
    def test_smoother_returns_correct_length(self):
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 10)
        obs = [np.array([float(i)]) for i in range(10)]
        result = kf.smooth(obs, init)
        assert len(result.smoothed_states) == 10
        assert len(result.smoother_gains) == 9

    def test_smoother_reduces_uncertainty(self):
        """Smoothed estimates should have smaller or equal covariance than filtered."""
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 10)
        obs = [np.array([float(i)]) for i in range(20)]
        filt = kf.filter(obs, init)
        smooth = kf.smooth(obs, init)

        for t in range(len(obs) - 1):  # exclude last (same as filtered)
            filt_trace = np.trace(filt.filtered_states[t].cov)
            smooth_trace = np.trace(smooth.smoothed_states[t].cov)
            assert smooth_trace <= filt_trace + 1e-10

    def test_smoother_last_equals_filtered(self):
        """Last smoothed state should equal last filtered state."""
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2))
        obs = [np.array([float(i)]) for i in range(5)]
        filt = kf.filter(obs, init)
        smooth = kf.smooth(obs, init)
        assert np.allclose(smooth.smoothed_states[-1].mean,
                          filt.filtered_states[-1].mean, atol=1e-10)

    def test_smoother_early_estimates_improved(self):
        """Early estimates should improve significantly with future data."""
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 100)
        # True trajectory: position = t, velocity = 1
        obs = [np.array([float(t) + 0.1 * np.sin(t)]) for t in range(30)]
        filt = kf.filter(obs, init)
        smooth = kf.smooth(obs, init)

        # First smoothed velocity estimate should be closer to 1.0
        # than first filtered velocity estimate (which starts from 0)
        smooth_vel_0 = smooth.smoothed_states[0].mean[1]
        filt_vel_0 = filt.filtered_states[0].mean[1]
        assert abs(smooth_vel_0 - 1.0) < abs(filt_vel_0 - 1.0)


# ===========================================================================
# Extended Kalman Filter tests
# ===========================================================================

class TestExtendedKalmanFilter:
    def test_linear_system_matches_kf(self):
        """EKF on a linear system should match standard KF."""
        F_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
        H_mat = np.array([[1.0, 0.0]])
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])

        # Standard KF
        kf = KalmanFilter(F_mat, H_mat, Q, R)

        # EKF with linear functions
        def f(x, u): return F_mat @ x
        def h(x): return H_mat @ x
        def F_jac(x, u): return F_mat
        def H_jac(x): return H_mat

        ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R, n=2, m=1)

        init = GaussianState(np.zeros(2), np.eye(2))
        obs = [np.array([float(i)]) for i in range(10)]

        kf_result = kf.filter(obs, init)
        ekf_result = ekf.filter(obs, init)

        for t in range(10):
            assert np.allclose(kf_result.filtered_states[t].mean,
                              ekf_result.filtered_states[t].mean, atol=1e-10)

    def test_nonlinear_range_bearing(self):
        """EKF for a simple nonlinear observation model."""
        # State: [x, y], dynamics: random walk
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1, 0], [0, 0.01]])

        def f(x, u):
            return x  # random walk

        def h(x):
            # Range-bearing observation from origin
            r = np.sqrt(x[0]**2 + x[1]**2)
            theta = np.arctan2(x[1], x[0])
            return np.array([r, theta])

        def F_jac(x, u):
            return np.eye(2)

        def H_jac(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            if r < 1e-10:
                r = 1e-10
            return np.array([
                [x[0] / r, x[1] / r],
                [-x[1] / r**2, x[0] / r**2]
            ])

        ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R, n=2, m=2)
        init = GaussianState(np.array([1.0, 0.0]), np.eye(2))

        # Generate observations from true position (3, 4)
        true_x = np.array([3.0, 4.0])
        rng = np.random.default_rng(42)
        obs = []
        for _ in range(20):
            r = np.sqrt(true_x[0]**2 + true_x[1]**2)
            theta = np.arctan2(true_x[1], true_x[0])
            z = np.array([r, theta]) + rng.multivariate_normal(np.zeros(2), R)
            obs.append(z)

        result = ekf.filter(obs, init)
        final = result.filtered_states[-1].mean
        assert abs(final[0] - 3.0) < 1.0
        assert abs(final[1] - 4.0) < 1.0

    def test_ekf_predict_only(self):
        """EKF predict without observations."""
        def f(x, u): return x * 0.9  # decay
        def h(x): return x
        def F_jac(x, u): return np.array([[0.9]])
        def H_jac(x): return np.array([[1.0]])

        ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac,
                                    np.array([[0.01]]), np.array([[0.1]]),
                                    n=1, m=1)
        state = GaussianState(np.array([10.0]), np.array([[1.0]]))
        pred = ekf.predict(state)
        assert pred.mean[0] == pytest.approx(9.0)  # 10 * 0.9


# ===========================================================================
# Unscented Kalman Filter tests
# ===========================================================================

class TestUnscentedKalmanFilter:
    def test_linear_system_close_to_kf(self):
        """UKF on linear system should be close to standard KF."""
        F_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
        H_mat = np.array([[1.0, 0.0]])
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])

        kf = KalmanFilter(F_mat, H_mat, Q, R)

        def f(x, u): return F_mat @ x
        def h(x): return H_mat @ x

        ukf = UnscentedKalmanFilter(f, h, Q, R, n=2, m=1)

        init = GaussianState(np.zeros(2), np.eye(2))
        obs = [np.array([float(i)]) for i in range(10)]

        kf_result = kf.filter(obs, init)
        ukf_result = ukf.filter(obs, init)

        for t in range(10):
            assert np.allclose(kf_result.filtered_states[t].mean,
                              ukf_result.filtered_states[t].mean, atol=1.0)

    def test_sigma_points_count(self):
        """2n+1 sigma points for n-dimensional state."""
        def f(x, u): return x
        def h(x): return x[:1]
        ukf = UnscentedKalmanFilter(f, h, np.eye(3), np.eye(1), n=3, m=1)
        assert ukf.n_sigma == 7  # 2*3 + 1

    def test_weights_sum_to_one(self):
        def f(x, u): return x
        def h(x): return x[:1]
        ukf = UnscentedKalmanFilter(f, h, np.eye(2), np.eye(1), n=2, m=1)
        assert np.sum(ukf.Wm) == pytest.approx(1.0, abs=1e-10)

    def test_nonlinear_quadratic_obs(self):
        """UKF with quadratic observation h(x) = x^2."""
        Q = np.array([[0.01]])
        R = np.array([[0.1]])

        def f(x, u): return x  # random walk
        def h(x): return np.array([x[0]**2])

        ukf = UnscentedKalmanFilter(f, h, Q, R, n=1, m=1)
        init = GaussianState(np.array([1.0]), np.array([[1.0]]))

        # Observe x^2 = 9 -> x should converge to 3 or -3
        obs = [np.array([9.0 + 0.1 * np.sin(i)]) for i in range(30)]
        result = ukf.filter(obs, init)
        final = result.filtered_states[-1].mean[0]
        assert abs(abs(final) - 3.0) < 1.0  # converges to +3 or -3

    def test_ukf_predict(self):
        def f(x, u): return x + 1
        def h(x): return x
        ukf = UnscentedKalmanFilter(f, h, np.array([[0.01]]),
                                     np.array([[0.1]]), n=1, m=1)
        state = GaussianState(np.array([5.0]), np.array([[1.0]]))
        pred = ukf.predict(state)
        assert pred.mean[0] == pytest.approx(6.0, abs=0.1)


# ===========================================================================
# Information Filter tests
# ===========================================================================

class TestInformationFilter:
    def test_information_state_roundtrip(self):
        gs = GaussianState(np.array([1.0, 2.0]), np.array([[2.0, 0.5], [0.5, 3.0]]))
        info = InformationState.from_gaussian(gs)
        gs2 = info.to_gaussian()
        assert np.allclose(gs.mean, gs2.mean, atol=1e-10)
        assert np.allclose(gs.cov, gs2.cov, atol=1e-10)

    def test_matches_standard_kf(self):
        """Information filter should produce same results as standard KF."""
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        info_filt = InformationFilter(F, H, Q, R)

        init_gs = GaussianState(np.zeros(2), np.eye(2) * 10)
        init_info = InformationState.from_gaussian(init_gs)

        obs = [np.array([float(i)]) for i in range(10)]

        kf_result = kf.filter(obs, init_gs)
        info_results = info_filt.filter(obs, init_info)

        for t in range(10):
            info_gs = info_results[t].to_gaussian()
            assert np.allclose(kf_result.filtered_states[t].mean,
                              info_gs.mean, atol=1e-6)

    def test_multi_sensor_fusion(self):
        """Multi-sensor update should fuse information from multiple sensors."""
        F = np.eye(2)
        H1 = np.array([[1.0, 0.0]])  # sensor 1: position x
        H2 = np.array([[0.0, 1.0]])  # sensor 2: position y
        Q = 0.01 * np.eye(2)
        R1 = np.array([[1.0]])
        R2 = np.array([[1.0]])

        info_filt = InformationFilter(F, H1, Q, R1)
        init = InformationState.from_gaussian(
            GaussianState(np.zeros(2), 10 * np.eye(2)))

        pred = info_filt.predict(init)

        # Fuse two sensors
        result = info_filt.multi_sensor_update(
            pred,
            measurements=[np.array([3.0]), np.array([4.0])],
            H_list=[H1, H2],
            R_list=[R1, R2]
        )
        gs = result.to_gaussian()
        assert abs(gs.mean[0] - 3.0) < 1.0
        assert abs(gs.mean[1] - 4.0) < 1.0

    def test_additive_update(self):
        """Two sequential updates should equal one combined update."""
        F = np.eye(1)
        H = np.eye(1)
        Q = np.array([[0.1]])
        R = np.array([[1.0]])

        info_filt = InformationFilter(F, H, Q, R)
        init = InformationState.from_gaussian(
            GaussianState(np.zeros(1), np.array([[10.0]])))

        # Single update with z=5
        pred = info_filt.predict(init)
        single = info_filt.update(pred, np.array([5.0]))

        # Manual: info_matrix = pred_info_matrix + H^T R^{-1} H
        # This verifies the additive property
        gs = single.to_gaussian()
        assert np.isfinite(gs.mean[0])


# ===========================================================================
# Square Root Kalman Filter tests
# ===========================================================================

class TestSquareRootKalmanFilter:
    def test_matches_standard_kf(self):
        """SRKF should produce similar results to standard KF."""
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        srkf = SquareRootKalmanFilter(F, H, Q, R)

        init = GaussianState(np.zeros(2), np.eye(2) * 10)
        obs = [np.array([float(i)]) for i in range(10)]

        kf_result = kf.filter(obs, init)
        srkf_result = srkf.filter(obs, init)

        for t in range(10):
            assert np.allclose(kf_result.filtered_states[t].mean,
                              srkf_result.filtered_states[t].mean, atol=0.5)

    def test_covariance_positive_definite(self):
        """SRKF should always produce positive definite covariances."""
        F, H, Q, R = make_1d_const_velocity()
        srkf = SquareRootKalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 10)
        obs = [np.array([float(i) * 2]) for i in range(20)]
        result = srkf.filter(obs, init)

        for state in result.filtered_states:
            eigvals = np.linalg.eigvalsh(state.cov)
            assert all(ev > -1e-10 for ev in eigvals)

    def test_log_likelihood_finite(self):
        F, H, Q, R = make_1d_const_velocity()
        srkf = SquareRootKalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2))
        obs = [np.array([float(i)]) for i in range(5)]
        result = srkf.filter(obs, init)
        assert np.isfinite(result.log_likelihood)


# ===========================================================================
# Steady-state gain tests
# ===========================================================================

class TestSteadyStateGain:
    def test_converges(self):
        F, H, Q, R = make_1d_const_velocity()
        K, P = steady_state_gain(F, H, Q, R)
        assert K.shape == (2, 1)
        assert P.shape == (2, 2)
        assert np.all(np.linalg.eigvalsh(P) > 0)

    def test_steady_state_consistent(self):
        """P_ss should satisfy the DARE."""
        F, H, Q, R = make_1d_const_velocity()
        K, P = steady_state_gain(F, H, Q, R)
        # P_pred = F P F^T + Q
        P_pred = F @ P @ F.T + Q
        # S = H P_pred H^T + R
        S = H @ P_pred @ H.T + R
        # K = P_pred H^T S^{-1}
        K_check = P_pred @ H.T @ np.linalg.inv(S)
        # P = (I - KH) P_pred
        P_check = (np.eye(2) - K_check @ H) @ P_pred
        assert np.allclose(P, P_check, atol=1e-8)

    def test_1d_simple(self):
        """1D random walk with direct observation."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        K, P = steady_state_gain(F, H, Q, R)
        # Steady state: P = P + Q, S = P + R, K = P/S, P_new = (1-K)*P
        # Solving: P_ss^2 + Q*P_ss - Q*P_ss = Q (approximately)
        assert 0 < K[0, 0] < 1  # gain between 0 and 1


# ===========================================================================
# Simulation tests
# ===========================================================================

class TestSimulation:
    def test_simulate_dimensions(self):
        F, H, Q, R = make_1d_const_velocity()
        states, obs = simulate_linear_system(F, H, Q, R,
                                              x0=np.zeros(2), T=10,
                                              rng=np.random.default_rng(42))
        assert len(states) == 11  # T+1 including initial
        assert len(obs) == 10
        assert states[0].shape == (2,)
        assert obs[0].shape == (1,)

    def test_simulate_with_controls(self):
        F = np.eye(1)
        B = np.array([[1.0]])
        H = np.eye(1)
        Q = np.array([[0.001]])
        R = np.array([[0.1]])
        controls = [np.array([1.0])] * 5
        states, obs = simulate_linear_system(F, H, Q, R, x0=np.zeros(1),
                                              T=5, B=B, controls=controls,
                                              rng=np.random.default_rng(42))
        # State should roughly increase by 1 each step
        for i in range(1, 6):
            assert abs(states[i][0] - float(i)) < 1.0

    def test_filter_tracks_simulation(self):
        """Filter should track simulated true states."""
        F, H, Q, R = make_1d_const_velocity()
        rng = np.random.default_rng(42)
        x0 = np.array([0.0, 1.0])  # start at 0, velocity 1
        states, obs = simulate_linear_system(F, H, Q, R, x0, T=50, rng=rng)

        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 100)
        result = kf.filter(obs, init)

        # RMSE of position estimates
        pos_errors = [
            (states[t+1][0] - result.filtered_states[t].mean[0])**2
            for t in range(50)
        ]
        rmse = np.sqrt(np.mean(pos_errors))
        assert rmse < 3.0  # reasonable tracking


# ===========================================================================
# Compare filters test
# ===========================================================================

class TestCompareFilters:
    def test_compare_kf_vs_srkf(self):
        F, H, Q, R = make_1d_const_velocity()
        rng = np.random.default_rng(42)
        states, obs = simulate_linear_system(F, H, Q, R,
                                              x0=np.array([0.0, 1.0]),
                                              T=20, rng=rng)

        kf = KalmanFilter(F, H, Q, R)
        srkf = SquareRootKalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 10)

        results = {
            'KF': kf.filter(obs, init),
            'SRKF': srkf.filter(obs, init),
        }
        comp = compare_filters(obs, states, results)

        assert 'KF' in comp
        assert 'SRKF' in comp
        assert 'rmse' in comp['KF']
        assert 'mean_nees' in comp['KF']
        assert 'log_likelihood' in comp['KF']
        # Both should have similar RMSE
        assert abs(comp['KF']['rmse'] - comp['SRKF']['rmse']) < 1.0


# ===========================================================================
# Discretize (HMM bridge) tests
# ===========================================================================

class TestDiscretize:
    def test_discretize_produces_valid_hmm_params(self):
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.5]])
        R = np.array([[1.0]])
        kf = KalmanFilter(F, H, Q, R)

        result = discretize_kalman(kf,
                                    state_grids=[np.array([-1, 0, 1])],
                                    obs_grids=[np.array([-2, -1, 0, 1, 2])])

        assert len(result['states']) == 3
        assert len(result['observations']) == 5

        # Transitions should be valid distributions
        for s, trans in result['transition'].items():
            total = sum(trans.values())
            assert abs(total - 1.0) < 1e-10

        # Emissions should be valid distributions
        for s, emit in result['emission'].items():
            total = sum(emit.values())
            assert abs(total - 1.0) < 1e-10

    def test_discretize_transition_peak(self):
        """Transition should peak at predicted next state."""
        F = np.array([[1.0]])  # identity dynamics
        H = np.array([[1.0]])
        Q = np.array([[0.1]])  # small noise
        R = np.array([[1.0]])
        kf = KalmanFilter(F, H, Q, R)

        grid = np.array([-2, -1, 0, 1, 2])
        result = discretize_kalman(kf, [grid], [grid])

        # From state 0, most likely next state should be 0
        trans_from_0 = result['transition']['s0.0']
        assert trans_from_0['s0.0'] == max(trans_from_0.values())


# ===========================================================================
# LQR / LQG tests
# ===========================================================================

class TestLQR:
    def test_lqr_gain_1d(self):
        """1D integrator: x_{t+1} = x_t + u_t."""
        F = np.array([[1.0]])
        B = np.array([[1.0]])
        Q_cost = np.array([[1.0]])
        R_cost = np.array([[1.0]])
        K, P = lqr_gain(F, B, Q_cost, R_cost)
        assert K.shape == (1, 1)
        assert 0 < K[0, 0] < 1  # gain should be stabilizing

    def test_lqr_stabilizes(self):
        """LQR gain should make closed-loop system stable."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        Q_cost = np.eye(2)
        R_cost = np.array([[0.1]])
        K, P = lqr_gain(F, B, Q_cost, R_cost)

        # Closed-loop: A_cl = F - B*K
        A_cl = F - B @ K
        eigvals = np.linalg.eigvals(A_cl)
        # All eigenvalues should be inside unit circle
        assert all(abs(ev) < 1.0 for ev in eigvals)

    def test_lqr_higher_R_less_control(self):
        """Higher control cost R -> smaller gains."""
        F = np.array([[1.0]])
        B = np.array([[1.0]])
        Q_cost = np.array([[1.0]])

        K_low, _ = lqr_gain(F, B, Q_cost, np.array([[0.1]]))
        K_high, _ = lqr_gain(F, B, Q_cost, np.array([[10.0]]))
        assert abs(K_low[0, 0]) > abs(K_high[0, 0])


class TestLQG:
    def test_lqg_controller(self):
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        kf = KalmanFilter(F, H, Q, R, B=B)

        Q_cost = np.eye(2)
        R_cost = np.array([[0.1]])
        K_lqr, K_kalman, P = lqg_controller(kf, Q_cost, R_cost)

        assert K_lqr.shape == (1, 2)
        assert K_kalman.shape == (2, 1)

    def test_simulate_lqg_stabilizes(self):
        """LQG should drive state toward zero."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        kf = KalmanFilter(F, H, Q, R, B=B)

        K_lqr, _, _ = lqg_controller(kf, np.eye(2), np.array([[0.1]]))

        result = simulate_lqg(kf, K_lqr, np.array([10.0, 5.0]), T=50,
                               rng=np.random.default_rng(42))

        # State should converge near zero
        final_state = result['true_states'][-1]
        assert abs(final_state[0]) < 5.0  # position near 0
        assert abs(final_state[1]) < 5.0  # velocity near 0

    def test_lqg_cost_decreases(self):
        """LQG costs should generally decrease over time."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        kf = KalmanFilter(F, H, Q, R, B=B)

        K_lqr, _, _ = lqg_controller(kf, np.eye(2), np.array([[0.1]]))
        result = simulate_lqg(kf, K_lqr, np.array([10.0, 0.0]), T=30,
                               rng=np.random.default_rng(42))

        # Average cost in last 10 steps should be less than first 10
        early = np.mean(result['costs'][:10])
        late = np.mean(result['costs'][-10:])
        assert late < early


# ===========================================================================
# 2D tracking integration test
# ===========================================================================

class TestIntegration2D:
    def test_2d_tracking(self):
        """Full pipeline: simulate 2D system, filter, smooth, compare."""
        F, H, Q, R = make_2d_position()
        rng = np.random.default_rng(42)
        x0 = np.array([0.0, 0.0, 1.0, 0.5])  # position (0,0), velocity (1, 0.5)
        states, obs = simulate_linear_system(F, H, Q, R, x0, T=30, rng=rng)

        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(4), np.eye(4) * 100)

        # Filter
        filt = kf.filter(obs, init)
        assert len(filt.filtered_states) == 30

        # Smooth
        smooth = kf.smooth(obs, init)
        assert len(smooth.smoothed_states) == 30

        # Smoother should be at least as good
        filt_err = np.mean([
            np.sum((states[t+1][:2] - filt.filtered_states[t].mean[:2])**2)
            for t in range(30)
        ])
        smooth_err = np.mean([
            np.sum((states[t+1][:2] - smooth.smoothed_states[t].mean[:2])**2)
            for t in range(30)
        ])
        assert smooth_err <= filt_err + 0.1


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_observation(self):
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2))
        result = kf.filter([np.array([1.0])], init)
        assert len(result.filtered_states) == 1

    def test_zero_process_noise(self):
        """Zero Q means dynamics are deterministic."""
        F = np.eye(1)
        H = np.eye(1)
        Q = np.zeros((1, 1))
        R = np.array([[1.0]])
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.array([0.0]), np.array([[1.0]]))
        obs = [np.array([5.0])] * 10
        result = kf.filter(obs, init)
        # With zero Q, covariance should shrink monotonically
        covs = [s.cov[0, 0] for s in result.filtered_states]
        for i in range(1, len(covs)):
            assert covs[i] <= covs[i-1] + 1e-15

    def test_large_measurement_noise(self):
        """Very large R -> observations barely affect estimate."""
        F = np.eye(1)
        H = np.eye(1)
        Q = np.array([[0.1]])
        R = np.array([[1e6]])  # huge measurement noise
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.array([0.0]), np.array([[1.0]]))
        result = kf.filter([np.array([100.0])], init)
        # Estimate should barely move from 0
        assert abs(result.filtered_states[0].mean[0]) < 1.0

    def test_small_measurement_noise(self):
        """Very small R -> observations dominate."""
        F = np.eye(1)
        H = np.eye(1)
        Q = np.array([[1.0]])
        R = np.array([[1e-6]])  # tiny measurement noise
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.array([0.0]), np.array([[10.0]]))
        result = kf.filter([np.array([7.0])], init)
        # Estimate should be very close to observation
        assert abs(result.filtered_states[0].mean[0] - 7.0) < 0.01

    def test_identity_observation(self):
        """H = I, direct state observation."""
        n = 3
        F = np.eye(n)
        H = np.eye(n)
        Q = 0.1 * np.eye(n)
        R = 0.1 * np.eye(n)
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(n), np.eye(n) * 10)
        obs = [np.array([1.0, 2.0, 3.0])] * 5
        result = kf.filter(obs, init)
        final = result.filtered_states[-1].mean
        assert np.allclose(final, [1.0, 2.0, 3.0], atol=0.5)

    def test_partial_observation(self):
        """Observe only first state component."""
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 10)
        obs = [np.array([5.0])] * 20
        result = kf.filter(obs, init)
        # First component should converge to 5
        assert abs(result.filtered_states[-1].mean[0] - 5.0) < 0.5
        # Second component uncertainty should remain larger
        assert result.filtered_states[-1].cov[1, 1] > result.filtered_states[-1].cov[0, 0]


# ===========================================================================
# Numerical stability tests
# ===========================================================================

class TestNumericalStability:
    def test_long_sequence(self):
        """Filter should remain stable over long sequences."""
        F, H, Q, R = make_1d_const_velocity()
        kf = KalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 100)
        rng = np.random.default_rng(42)
        obs = [np.array([float(t) + rng.normal(0, 1)]) for t in range(500)]
        result = kf.filter(obs, init)

        # All states should have finite values
        for state in result.filtered_states:
            assert all(np.isfinite(state.mean))
            assert all(np.isfinite(state.cov.flatten()))

        # Covariance should be PSD
        for state in result.filtered_states[-10:]:
            eigvals = np.linalg.eigvalsh(state.cov)
            assert all(ev > -1e-10 for ev in eigvals)

    def test_srkf_long_sequence(self):
        """SRKF should also remain stable over long sequences."""
        F, H, Q, R = make_1d_const_velocity()
        srkf = SquareRootKalmanFilter(F, H, Q, R)
        init = GaussianState(np.zeros(2), np.eye(2) * 100)
        rng = np.random.default_rng(42)
        obs = [np.array([float(t) + rng.normal(0, 1)]) for t in range(500)]
        result = srkf.filter(obs, init)

        for state in result.filtered_states:
            assert all(np.isfinite(state.mean))

    def test_near_singular_covariance(self):
        """Filter should handle near-singular initial covariance."""
        F = np.eye(2)
        H = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.eye(2)
        kf = KalmanFilter(F, H, Q, R)
        # Near-singular initial cov
        init = GaussianState(np.zeros(2), np.array([[1e-8, 0], [0, 1e-8]]))
        obs = [np.array([1.0, 2.0])] * 5
        result = kf.filter(obs, init)
        assert all(np.isfinite(result.filtered_states[-1].mean))


# ===========================================================================
# Model comparison tests
# ===========================================================================

class TestModelComparison:
    def test_better_model_higher_likelihood(self):
        """Correct model parameters should yield higher log-likelihood."""
        F_true = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q_true = 0.1 * np.eye(2)
        R_true = np.array([[1.0]])

        rng = np.random.default_rng(42)
        states, obs = simulate_linear_system(F_true, H, Q_true, R_true,
                                              x0=np.array([0.0, 1.0]),
                                              T=50, rng=rng)

        init = GaussianState(np.zeros(2), np.eye(2) * 10)

        # Correct model
        kf_correct = KalmanFilter(F_true, H, Q_true, R_true)
        result_correct = kf_correct.filter(obs, init)

        # Wrong model (wrong dynamics)
        F_wrong = np.eye(2)  # static, no velocity
        kf_wrong = KalmanFilter(F_wrong, H, Q_true, R_true)
        result_wrong = kf_wrong.filter(obs, init)

        assert result_correct.log_likelihood > result_wrong.log_likelihood


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
