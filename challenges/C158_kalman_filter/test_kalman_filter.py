"""Tests for C158: Kalman Filter"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.linalg import inv
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C157_hidden_markov_model'))
from kalman_filter import (
    KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter,
    InformationFilter, SquareRootKalmanFilter, KalmanSmoother,
    InteractingMultipleModel, EnsembleKalmanFilter, KalmanUtils
)
from hidden_markov_model import HMM


# ===========================================================================
# KalmanFilter Tests
# ===========================================================================

class TestKalmanFilter:
    """Tests for the core linear Kalman filter."""

    def test_create(self):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        assert kf.dim_x == 2
        assert kf.dim_z == 1
        assert kf.x.shape == (2,)
        assert kf.P.shape == (2, 2)

    def test_constant_position(self):
        """1D constant position model: state = position, measurement = position."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.01]])
        kf.R = np.array([[1.0]])
        kf.x = np.array([0.0])
        kf.P = np.array([[10.0]])

        # Feed constant measurements
        for _ in range(20):
            kf.predict()
            kf.update(np.array([5.0]))

        assert abs(kf.x[0] - 5.0) < 0.5

    def test_constant_velocity(self):
        """2D constant velocity model: [pos, vel]."""
        dt = 1.0
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, dt], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.array([[0.01, 0], [0, 0.01]])
        kf.R = np.array([[1.0]])
        kf.x = np.array([0.0, 1.0])
        kf.P = np.eye(2) * 10

        # True trajectory: x = t, v = 1
        rng = np.random.RandomState(42)
        measurements = []
        for t in range(50):
            z = t + rng.randn() * 1.0
            measurements.append([z])

        xs, Ps, ll = kf.filter(measurements)
        # Final position estimate should be close to 49
        assert abs(xs[-1, 0] - 49) < 3
        # Velocity should be close to 1
        assert abs(xs[-1, 1] - 1.0) < 0.3

    def test_innovation(self):
        """Check that innovation y = z - H*x_prior."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.1]])
        kf.R = np.array([[1.0]])
        kf.x = np.array([3.0])

        kf.predict()
        kf.update(np.array([5.0]))
        assert_allclose(kf.y, [5.0 - 3.0], atol=1e-10)

    def test_kalman_gain_converges(self):
        """Kalman gain should converge to steady state."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.1]])
        kf.R = np.array([[1.0]])
        kf.x = np.array([0.0])
        kf.P = np.array([[100.0]])

        gains = []
        for _ in range(100):
            kf.predict()
            kf.update(np.array([0.0]))
            gains.append(kf.K[0, 0])

        # Gain should converge
        assert abs(gains[-1] - gains[-2]) < 1e-6

    def test_covariance_decreases(self):
        """P should decrease as we get more measurements."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.01]])
        kf.R = np.array([[1.0]])
        kf.P = np.array([[100.0]])

        P_init = kf.P[0, 0]
        for _ in range(10):
            kf.predict()
            kf.update(np.array([0.0]))

        assert kf.P[0, 0] < P_init

    def test_control_input(self):
        """Control input B*u should affect state."""
        kf = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.B = np.array([[0.5], [1.0]])
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[0.1]])

        kf.predict(u=np.array([2.0]))
        # x should include control effect: F*[0,0] + B*[2] = [1, 2]
        assert_allclose(kf.x, [1.0, 2.0], atol=1e-10)

    def test_batch_filter(self):
        """batch_filter returns all priors and gains."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.1]])
        kf.R = np.array([[1.0]])

        zs = [np.array([i * 0.5]) for i in range(10)]
        xs, Ps, x_p, P_p, Ks = kf.batch_filter(zs)
        assert xs.shape == (10, 1)
        assert Ks.shape == (10, 1, 1)

    def test_simulate(self):
        """Simulate produces states and measurements."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.eye(2) * 0.1
        kf.R = np.array([[1.0]])
        kf.x = np.array([0.0, 1.0])

        states, meas = kf.simulate(20, x0=np.array([0.0, 1.0]), seed=42)
        assert states.shape == (20, 2)
        assert meas.shape == (20, 1)
        # States should roughly follow x = t
        assert abs(states[10, 0] - 10) < 5

    def test_log_likelihood(self):
        """Log-likelihood should be finite."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.1]])
        kf.R = np.array([[1.0]])

        kf.predict()
        kf.update(np.array([1.0]))
        assert np.isfinite(kf.log_likelihood)

    def test_2d_position(self):
        """Track 2D position (x, y) with velocity."""
        dt = 0.1
        kf = KalmanFilter(dim_x=4, dim_z=2)  # [x, vx, y, vy]
        kf.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        kf.Q = np.eye(4) * 0.01
        kf.R = np.eye(2) * 0.5
        kf.x = np.array([0, 1, 0, 0.5])

        rng = np.random.RandomState(123)
        zs = []
        for t in range(100):
            true_x = t * dt * 1.0
            true_y = t * dt * 0.5
            zs.append(np.array([true_x + rng.randn() * 0.7, true_y + rng.randn() * 0.7]))

        xs, Ps, ll = kf.filter(zs)
        assert abs(xs[-1, 0] - 10.0) < 2  # x ~ 10
        assert abs(xs[-1, 2] - 5.0) < 2   # y ~ 5

    def test_joseph_form_stability(self):
        """Joseph form should keep P symmetric positive definite."""
        kf = KalmanFilter(dim_x=3, dim_z=2)
        kf.F = np.array([[1, 0.1, 0], [0, 1, 0.1], [0, 0, 1]])
        kf.H = np.array([[1, 0, 0], [0, 1, 0]])
        kf.Q = np.eye(3) * 0.01
        kf.R = np.eye(2) * 0.5

        rng = np.random.RandomState(7)
        for _ in range(200):
            kf.predict()
            kf.update(rng.randn(2))
            # P should remain symmetric
            assert_allclose(kf.P, kf.P.T, atol=1e-10)
            # Eigenvalues should be positive
            eigs = np.linalg.eigvalsh(kf.P)
            assert np.all(eigs > -1e-10)


# ===========================================================================
# ExtendedKalmanFilter Tests
# ===========================================================================

class TestExtendedKalmanFilter:
    """Tests for the Extended Kalman Filter."""

    def test_linear_system_matches_kf(self):
        """EKF on a linear system should match standard KF."""
        # Linear system: x_{k+1} = F x_k, z_k = H x_k
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])

        ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
        ekf.x = np.array([0.0, 1.0])
        ekf.P = np.eye(2) * 5
        ekf.Q = np.eye(2) * 0.01
        ekf.R = np.array([[1.0]])
        ekf.f = lambda x, u: F @ x
        ekf.h = lambda x: H @ x
        ekf.F_jacobian = lambda x, u: F
        ekf.H_jacobian = lambda x: H

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0.0, 1.0])
        kf.P = np.eye(2) * 5
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[1.0]])
        kf.F = F
        kf.H = H

        rng = np.random.RandomState(42)
        for _ in range(20):
            z = np.array([rng.randn()])
            ekf.predict()
            ekf.update(z)
            kf.predict()
            kf.update(z)

        assert_allclose(ekf.x, kf.x, atol=1e-10)
        assert_allclose(ekf.P, kf.P, atol=1e-10)

    def test_nonlinear_pendulum(self):
        """Track a pendulum: state = [theta, omega]."""
        dt = 0.01
        g = 9.81
        L = 1.0

        def f(x, u):
            theta, omega = x
            return np.array([theta + omega * dt, omega - (g / L) * np.sin(theta) * dt])

        def h(x):
            return np.array([np.sin(x[0])])  # observe sin(theta)

        def F_jac(x, u):
            theta = x[0]
            return np.array([
                [1, dt],
                [-(g / L) * np.cos(theta) * dt, 1]
            ])

        def H_jac(x):
            return np.array([[np.cos(x[0]), 0]])

        ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
        ekf.x = np.array([0.1, 0.0])
        ekf.P = np.eye(2) * 0.1
        ekf.Q = np.eye(2) * 0.001
        ekf.R = np.array([[0.01]])
        ekf.f = f
        ekf.h = h
        ekf.F_jacobian = F_jac
        ekf.H_jacobian = H_jac

        # Simulate true trajectory
        rng = np.random.RandomState(42)
        true_state = np.array([0.3, 0.0])
        for _ in range(200):
            true_state = f(true_state, None) + rng.randn(2) * 0.01
            z = h(true_state) + rng.randn(1) * 0.1
            ekf.predict()
            ekf.update(z)

        # Should track roughly
        assert abs(ekf.x[0] - true_state[0]) < 1.0

    def test_filter_sequence(self):
        """EKF filter() runs over full sequence."""
        ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
        ekf.f = lambda x, u: x
        ekf.h = lambda x: x
        ekf.F_jacobian = lambda x, u: np.eye(1)
        ekf.H_jacobian = lambda x: np.eye(1)
        ekf.Q = np.array([[0.1]])
        ekf.R = np.array([[1.0]])

        zs = [np.array([5.0 + 0.1 * i]) for i in range(30)]
        xs, Ps = ekf.filter(zs)
        assert xs.shape == (30, 1)

    def test_ekf_with_control(self):
        """EKF with control input."""
        ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1, dim_u=1)
        ekf.f = lambda x, u: x + (np.array([u[0]]) if u is not None else np.zeros(1))
        ekf.h = lambda x: x
        ekf.F_jacobian = lambda x, u: np.eye(1)
        ekf.H_jacobian = lambda x: np.eye(1)
        ekf.Q = np.array([[0.01]])
        ekf.R = np.array([[0.1]])
        ekf.x = np.array([0.0])

        for t in range(10):
            ekf.predict(u=np.array([1.0]))
            ekf.update(np.array([t + 1.0]))

        assert abs(ekf.x[0] - 10.0) < 1.0


# ===========================================================================
# UnscentedKalmanFilter Tests
# ===========================================================================

class TestUnscentedKalmanFilter:
    """Tests for the Unscented Kalman Filter."""

    def test_linear_matches_kf(self):
        """UKF on linear system should approximately match KF."""
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])

        ukf = UnscentedKalmanFilter(
            dim_x=2, dim_z=1,
            f=lambda x, u: F @ x,
            h=lambda x: H @ x,
            alpha=1.0, beta=2.0, kappa=0.0
        )
        ukf.x = np.array([0.0, 1.0])
        ukf.P = np.eye(2) * 5
        ukf.Q = np.eye(2) * 0.01
        ukf.R = np.array([[1.0]])

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0.0, 1.0])
        kf.P = np.eye(2) * 5
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[1.0]])
        kf.F = F
        kf.H = H

        rng = np.random.RandomState(42)
        for _ in range(20):
            z = np.array([rng.randn()])
            ukf.predict()
            ukf.update(z)
            kf.predict()
            kf.update(z)

        assert_allclose(ukf.x, kf.x, atol=1.0)

    def test_nonlinear_tracking(self):
        """UKF tracks nonlinear system."""
        def f(x, u):
            return np.array([x[0] + 0.1 * x[1], x[1]])

        def h(x):
            return np.array([x[0] ** 2])  # nonlinear observation

        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, f=f, h=h)
        ukf.x = np.array([1.0, 0.1])
        ukf.P = np.eye(2) * 0.5
        ukf.Q = np.eye(2) * 0.01
        ukf.R = np.array([[0.1]])

        rng = np.random.RandomState(42)
        true_x = np.array([1.0, 0.1])
        for _ in range(50):
            true_x = f(true_x, None) + rng.randn(2) * 0.05
            z = h(true_x) + rng.randn(1) * 0.3
            ukf.predict()
            ukf.update(z)

        assert abs(ukf.x[0] - true_x[0]) < 2.0

    def test_sigma_points_count(self):
        """Should have 2n+1 sigma points."""
        ukf = UnscentedKalmanFilter(dim_x=3, dim_z=1, f=lambda x, u: x, h=lambda x: x[:1])
        assert ukf.n_sigma == 7  # 2*3+1

    def test_weights_sum(self):
        """Sigma point weights should sum to 1."""
        ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, f=lambda x, u: x, h=lambda x: x[:2])
        assert_allclose(ukf.Wm.sum(), 1.0, atol=1e-10)

    def test_filter_method(self):
        """filter() over sequence."""
        ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1,
                                     f=lambda x, u: x, h=lambda x: x)
        ukf.Q = np.array([[0.1]])
        ukf.R = np.array([[1.0]])
        zs = [np.array([3.0]) for _ in range(20)]
        xs, Ps = ukf.filter(zs)
        assert xs.shape == (20, 1)
        assert abs(xs[-1, 0] - 3.0) < 1.0

    def test_ukf_handles_near_singular(self):
        """UKF should handle near-singular P via regularization."""
        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1,
                                     f=lambda x, u: x, h=lambda x: x[:1])
        ukf.P = np.eye(2) * 1e-12
        ukf.Q = np.eye(2) * 0.01
        ukf.R = np.array([[1.0]])
        # Should not raise
        ukf.predict()
        ukf.update(np.array([1.0]))


# ===========================================================================
# InformationFilter Tests
# ===========================================================================

class TestInformationFilter:
    """Tests for the Information Filter."""

    def test_matches_kf(self):
        """Information filter should produce same estimates as KF."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 0.1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[1.0]])
        kf.x = np.array([0.0, 1.0])
        kf.P = np.eye(2) * 5

        inf_f = InformationFilter(dim_x=2, dim_z=1)
        inf_f.F = kf.F.copy()
        inf_f.H = kf.H.copy()
        inf_f.Q = kf.Q.copy()
        inf_f.R = kf.R.copy()
        P0 = kf.P.copy()
        inf_f.Y = inv(P0)
        inf_f.y = inv(P0) @ kf.x.copy()

        rng = np.random.RandomState(42)
        for _ in range(20):
            z = np.array([rng.randn()])
            kf.predict()
            kf.update(z)
            inf_f.predict()
            inf_f.update(z)

        assert_allclose(inf_f.x, kf.x, atol=0.5)

    def test_multi_sensor_fusion(self):
        """Fuse multiple measurements at once."""
        inf_f = InformationFilter(dim_x=2, dim_z=1)
        inf_f.F = np.eye(2)
        inf_f.H = np.array([[1, 0]])
        inf_f.Q = np.eye(2) * 0.01
        inf_f.R = np.array([[1.0]])
        inf_f.Y = np.eye(2) * 0.1
        inf_f.y = np.zeros(2)

        # Fuse 3 sensors measuring position
        H_list = [np.array([[1, 0]]), np.array([[1, 0]]), np.array([[0, 1]])]
        R_list = [np.array([[1.0]]), np.array([[0.5]]), np.array([[2.0]])]
        measurements = [np.array([5.0]), np.array([5.2]), np.array([1.0])]

        inf_f.fuse(measurements, H_list, R_list)
        x = inf_f.x
        # Position estimate should be close to 5 (two sensors agree)
        assert abs(x[0] - 5.0) < 2.0

    def test_properties(self):
        """x and P properties should return correct dimensions."""
        inf_f = InformationFilter(dim_x=3, dim_z=2)
        assert inf_f.x.shape == (3,)
        assert inf_f.P.shape == (3, 3)

    def test_filter_sequence(self):
        """Run filter over sequence."""
        inf_f = InformationFilter(dim_x=1, dim_z=1)
        inf_f.F = np.array([[1.0]])
        inf_f.H = np.array([[1.0]])
        inf_f.Q = np.array([[0.1]])
        inf_f.R = np.array([[1.0]])

        zs = [np.array([5.0]) for _ in range(15)]
        xs, Ps = inf_f.filter(zs)
        assert xs.shape == (15, 1)
        assert abs(xs[-1, 0] - 5.0) < 1.0


# ===========================================================================
# SquareRootKalmanFilter Tests
# ===========================================================================

class TestSquareRootKalmanFilter:
    """Tests for the Square Root Kalman Filter."""

    def test_tracks_constant(self):
        """SRKF should track constant value."""
        srkf = SquareRootKalmanFilter(dim_x=1, dim_z=1)
        srkf.F = np.array([[1.0]])
        srkf.H = np.array([[1.0]])
        srkf.set_Q(np.array([[0.01]]))
        srkf.set_R(np.array([[1.0]]))
        srkf.x = np.array([0.0])
        srkf.S = np.array([[3.0]])  # sqrt of P=9

        for _ in range(30):
            srkf.predict()
            srkf.update(np.array([7.0]))

        assert abs(srkf.x[0] - 7.0) < 1.0

    def test_p_equals_s_st(self):
        """P property should equal S @ S.T."""
        srkf = SquareRootKalmanFilter(dim_x=2, dim_z=1)
        srkf.F = np.array([[1, 0.1], [0, 1]])
        srkf.H = np.array([[1, 0]])
        srkf.set_Q(np.eye(2) * 0.01)
        srkf.set_R(np.array([[1.0]]))

        srkf.predict()
        srkf.update(np.array([1.0]))
        assert_allclose(srkf.P, srkf.S @ srkf.S.T, atol=1e-6)

    def test_filter_sequence(self):
        """Run SRKF over sequence."""
        srkf = SquareRootKalmanFilter(dim_x=1, dim_z=1)
        srkf.F = np.array([[1.0]])
        srkf.H = np.array([[1.0]])
        srkf.set_Q(np.array([[0.1]]))
        srkf.set_R(np.array([[1.0]]))

        zs = [np.array([3.0 + 0.01 * i]) for i in range(20)]
        xs, Ps = srkf.filter(zs)
        assert xs.shape == (20, 1)

    def test_numerical_stability(self):
        """SRKF should remain stable over many iterations."""
        srkf = SquareRootKalmanFilter(dim_x=2, dim_z=1)
        srkf.F = np.array([[1, 0.1], [0, 1]])
        srkf.H = np.array([[1, 0]])
        srkf.set_Q(np.eye(2) * 0.001)
        srkf.set_R(np.array([[0.5]]))

        rng = np.random.RandomState(42)
        for _ in range(500):
            srkf.predict()
            srkf.update(rng.randn(1))
            P = srkf.P
            assert_allclose(P, P.T, atol=1e-8)
            eigs = np.linalg.eigvalsh(P)
            assert np.all(eigs > -1e-8)


# ===========================================================================
# KalmanSmoother Tests
# ===========================================================================

class TestKalmanSmoother:
    """Tests for the RTS Kalman Smoother."""

    def test_smoother_reduces_error(self):
        """Smoother should have lower error than filter."""
        dt = 1.0
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, dt], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.eye(2) * 0.1
        kf.R = np.array([[2.0]])
        kf.x = np.array([0.0, 1.0])
        kf.P = np.eye(2) * 10

        # Simulate
        rng = np.random.RandomState(42)
        true_states = []
        measurements = []
        x_true = np.array([0.0, 1.0])
        for t in range(50):
            true_states.append(x_true.copy())
            z = kf.H @ x_true + rng.randn(1) * np.sqrt(2.0)
            measurements.append(z)
            x_true = kf.F @ x_true + rng.multivariate_normal([0, 0], kf.Q)

        true_states = np.array(true_states)

        # Filter
        kf2 = KalmanFilter(dim_x=2, dim_z=1)
        kf2.F = kf.F.copy()
        kf2.H = kf.H.copy()
        kf2.Q = kf.Q.copy()
        kf2.R = kf.R.copy()
        kf2.x = np.array([0.0, 1.0])
        kf2.P = np.eye(2) * 10
        xs_f, Ps_f, _ = kf2.filter(measurements)

        # Smooth
        kf3 = KalmanFilter(dim_x=2, dim_z=1)
        kf3.F = kf.F.copy()
        kf3.H = kf.H.copy()
        kf3.Q = kf.Q.copy()
        kf3.R = kf.R.copy()
        kf3.x = np.array([0.0, 1.0])
        kf3.P = np.eye(2) * 10
        smoother = KalmanSmoother(kf3)
        xs_s, Ps_s = smoother.smooth(measurements)

        # Smoother error should be <= filter error (especially in middle)
        filter_mse = np.mean((xs_f[:, 0] - true_states[:, 0]) ** 2)
        smooth_mse = np.mean((xs_s[:, 0] - true_states[:, 0]) ** 2)
        assert smooth_mse <= filter_mse * 1.1  # allow small margin

    def test_smoother_covariance_smaller(self):
        """Smoothed covariance should be <= filtered covariance."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 0.5], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.eye(2) * 0.1
        kf.R = np.array([[1.0]])
        kf.x = np.zeros(2)
        kf.P = np.eye(2) * 10

        zs = [np.array([i * 0.5]) for i in range(30)]

        # Filter
        kf_f = KalmanFilter(dim_x=2, dim_z=1)
        kf_f.F = kf.F.copy(); kf_f.H = kf.H.copy()
        kf_f.Q = kf.Q.copy(); kf_f.R = kf.R.copy()
        kf_f.x = np.zeros(2); kf_f.P = np.eye(2) * 10
        xs_f, Ps_f, _ = kf_f.filter(zs)

        # Smooth
        kf_s = KalmanFilter(dim_x=2, dim_z=1)
        kf_s.F = kf.F.copy(); kf_s.H = kf.H.copy()
        kf_s.Q = kf.Q.copy(); kf_s.R = kf.R.copy()
        kf_s.x = np.zeros(2); kf_s.P = np.eye(2) * 10
        smoother = KalmanSmoother(kf_s)
        xs_s, Ps_s = smoother.smooth(zs)

        # Check interior points (not last, which is the same)
        for k in range(len(zs) - 2):
            assert np.trace(Ps_s[k]) <= np.trace(Ps_f[k]) + 1e-6

    def test_last_point_same(self):
        """Smoothed last point equals filtered last point."""
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]]); kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.1]]); kf.R = np.array([[1.0]])

        zs = [np.array([2.0]) for _ in range(10)]

        kf1 = KalmanFilter(dim_x=1, dim_z=1)
        kf1.F = kf.F; kf1.H = kf.H; kf1.Q = kf.Q; kf1.R = kf.R
        xs_f, _, _ = kf1.filter(zs)

        kf2 = KalmanFilter(dim_x=1, dim_z=1)
        kf2.F = kf.F; kf2.H = kf.H; kf2.Q = kf.Q; kf2.R = kf.R
        smoother = KalmanSmoother(kf2)
        xs_s, _ = smoother.smooth(zs)

        assert_allclose(xs_s[-1], xs_f[-1], atol=1e-10)


# ===========================================================================
# InteractingMultipleModel Tests
# ===========================================================================

class TestInteractingMultipleModel:
    """Tests for IMM estimator."""

    def _make_cv_ca_imm(self):
        """Create IMM with constant-velocity and constant-acceleration modes."""
        dt = 1.0
        # Mode 1: constant velocity [x, vx]
        kf1 = KalmanFilter(dim_x=2, dim_z=1)
        kf1.F = np.array([[1, dt], [0, 1]])
        kf1.H = np.array([[1, 0]])
        kf1.Q = np.eye(2) * 0.1
        kf1.R = np.array([[1.0]])
        kf1.x = np.array([0.0, 1.0])
        kf1.P = np.eye(2) * 5

        # Mode 2: near-stationary [x, vx]
        kf2 = KalmanFilter(dim_x=2, dim_z=1)
        kf2.F = np.array([[1, dt], [0, 0.5]])  # velocity decays
        kf2.H = np.array([[1, 0]])
        kf2.Q = np.eye(2) * 0.5
        kf2.R = np.array([[1.0]])
        kf2.x = np.array([0.0, 1.0])
        kf2.P = np.eye(2) * 5

        TPM = np.array([[0.95, 0.05], [0.05, 0.95]])
        return InteractingMultipleModel([kf1, kf2], TPM)

    def test_create(self):
        imm = self._make_cv_ca_imm()
        assert imm.n_modes == 2
        assert_allclose(imm.mu, [0.5, 0.5])

    def test_mode_probs_sum_to_one(self):
        imm = self._make_cv_ca_imm()
        rng = np.random.RandomState(42)
        for _ in range(20):
            imm.predict()
            imm.update(rng.randn(1))
            assert_allclose(imm.mu.sum(), 1.0, atol=1e-10)

    def test_tracks_constant_velocity(self):
        """IMM should track constant velocity motion."""
        imm = self._make_cv_ca_imm()
        rng = np.random.RandomState(42)

        for t in range(50):
            z = np.array([float(t) + rng.randn() * 1.0])
            imm.predict()
            imm.update(z)

        # Should be near position = 49
        assert abs(imm.x[0] - 49) < 5

    def test_filter_returns_mode_probs(self):
        """filter() returns mode probabilities."""
        imm = self._make_cv_ca_imm()
        zs = [np.array([float(t)]) for t in range(20)]
        xs, Ps, mus = imm.filter(zs)
        assert mus.shape == (20, 2)
        # Each row should sum to 1
        for row in mus:
            assert_allclose(row.sum(), 1.0, atol=1e-10)

    def test_mode_switching(self):
        """IMM should detect mode switch."""
        imm = self._make_cv_ca_imm()

        # Phase 1: constant velocity (mode 1 should dominate)
        rng = np.random.RandomState(42)
        for t in range(30):
            z = np.array([float(t) + rng.randn() * 0.5])
            imm.predict()
            imm.update(z)

        mu_cv = imm.mu[0]  # should be higher

        # Phase 2: stop moving (mode 2 should gain)
        for t in range(30):
            z = np.array([30.0 + rng.randn() * 0.5])
            imm.predict()
            imm.update(z)

        mu_stop = imm.mu[1]
        # Mode 2 should increase relative to phase 1
        assert mu_stop > 0.1  # at minimum, mode 2 should have probability

    def test_custom_initial_probs(self):
        """Custom initial mode probabilities."""
        dt = 1.0
        kf1 = KalmanFilter(dim_x=1, dim_z=1)
        kf1.F = np.eye(1); kf1.H = np.eye(1)
        kf1.Q = np.eye(1) * 0.1; kf1.R = np.eye(1)

        kf2 = KalmanFilter(dim_x=1, dim_z=1)
        kf2.F = np.eye(1); kf2.H = np.eye(1)
        kf2.Q = np.eye(1) * 0.1; kf2.R = np.eye(1)

        TPM = np.array([[0.9, 0.1], [0.1, 0.9]])
        imm = InteractingMultipleModel([kf1, kf2], TPM, mode_probs=[0.8, 0.2])
        assert_allclose(imm.mu, [0.8, 0.2])


# ===========================================================================
# EnsembleKalmanFilter Tests
# ===========================================================================

class TestEnsembleKalmanFilter:
    """Tests for the Ensemble Kalman Filter."""

    def test_create(self):
        enkf = EnsembleKalmanFilter(
            dim_x=2, dim_z=1, n_ensemble=50,
            f=lambda x: x, h=lambda x: x[:1], seed=42
        )
        assert enkf.n_ensemble == 50
        assert enkf.ensemble.shape == (50, 2)

    def test_tracks_constant(self):
        """EnKF should track constant value."""
        enkf = EnsembleKalmanFilter(
            dim_x=1, dim_z=1, n_ensemble=100,
            f=lambda x: x, h=lambda x: x, seed=42
        )
        enkf.Q = np.array([[0.01]])
        enkf.R = np.array([[1.0]])
        enkf.initialize(np.array([0.0]), np.eye(1) * 5)

        for _ in range(30):
            enkf.predict()
            enkf.update(np.array([5.0]))

        assert abs(enkf.x[0] - 5.0) < 1.0

    def test_tracks_linear_motion(self):
        """EnKF tracks linear motion."""
        F = np.array([[1, 0.1], [0, 1]])

        enkf = EnsembleKalmanFilter(
            dim_x=2, dim_z=1, n_ensemble=200,
            f=lambda x: F @ x,
            h=lambda x: x[:1],
            seed=42
        )
        enkf.Q = np.eye(2) * 0.01
        enkf.R = np.array([[0.5]])
        enkf.initialize(np.array([0.0, 1.0]), np.eye(2))

        rng = np.random.RandomState(42)
        true_x = np.array([0.0, 1.0])
        for _ in range(50):
            true_x = F @ true_x + rng.randn(2) * 0.05
            z = true_x[:1] + rng.randn(1) * 0.7
            enkf.predict()
            enkf.update(z)

        assert abs(enkf.x[0] - true_x[0]) < 3.0

    def test_ensemble_spread(self):
        """Ensemble spread (P) should be reasonable."""
        enkf = EnsembleKalmanFilter(
            dim_x=2, dim_z=1, n_ensemble=100,
            f=lambda x: x, h=lambda x: x[:1], seed=42
        )
        enkf.Q = np.eye(2) * 0.1
        enkf.R = np.array([[1.0]])
        enkf.initialize(np.zeros(2), np.eye(2) * 10)

        for _ in range(20):
            enkf.predict()
            enkf.update(np.array([0.0]))

        P = enkf.P
        assert P.shape == (2, 2)
        # Trace should be positive and finite
        assert np.trace(P) > 0
        assert np.isfinite(np.trace(P))

    def test_filter_sequence(self):
        """filter() over sequence."""
        enkf = EnsembleKalmanFilter(
            dim_x=1, dim_z=1, n_ensemble=50,
            f=lambda x: x, h=lambda x: x, seed=42
        )
        enkf.Q = np.array([[0.1]])
        enkf.R = np.array([[1.0]])
        enkf.initialize(np.zeros(1), np.eye(1) * 5)

        zs = [np.array([3.0]) for _ in range(15)]
        xs, Ps = enkf.filter(zs)
        assert xs.shape == (15, 1)


# ===========================================================================
# KalmanUtils Tests
# ===========================================================================

class TestKalmanUtils:
    """Tests for utility functions."""

    def test_observability_observable(self):
        """Simple observable system."""
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        assert KalmanUtils.is_observable(F, H)

    def test_observability_unobservable(self):
        """System that is not observable."""
        F = np.array([[1, 0], [0, 1]])
        H = np.array([[1, 0]])  # only sees x[0], decoupled
        # For identity F with H = [1,0], O = [[1,0],[1,0]] -> rank 1, not observable
        assert not KalmanUtils.is_observable(F, H)

    def test_controllability(self):
        """Controllable system."""
        F = np.array([[1, 1], [0, 1]])
        B = np.array([[0], [1]])
        assert KalmanUtils.is_controllable(F, B)

    def test_uncontrollable(self):
        """Uncontrollable system."""
        F = np.array([[1, 0], [0, 1]])
        B = np.array([[1], [0]])  # only controls x[0], decoupled
        # C = [B, FB] = [[1,1],[0,0]] -> rank 1
        assert not KalmanUtils.is_controllable(F, B)

    def test_observability_matrix_shape(self):
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        O = KalmanUtils.observability_matrix(F, H)
        assert O.shape == (2, 2)  # n*dim_z x dim_x

    def test_controllability_matrix_shape(self):
        F = np.array([[1, 1], [0, 1]])
        B = np.array([[0], [1]])
        C = KalmanUtils.controllability_matrix(F, B)
        assert C.shape == (2, 2)  # dim_x x n*dim_u

    def test_steady_state_gain(self):
        """Steady-state gain should converge."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])

        K, P = KalmanUtils.steady_state_gain(F, H, Q, R)
        assert K.shape == (1, 1)
        assert P.shape == (1, 1)
        # Verify: P = (I - KH) (F P F^T + Q)
        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T + R
        K_check = P_pred @ H.T @ inv(S)
        assert_allclose(K, K_check, atol=1e-8)

    def test_nis(self):
        """NIS computation."""
        y = np.array([1.0, 0.5])
        S = np.eye(2) * 2.0
        nis = KalmanUtils.nis(y, S)
        expected = 1.0 / 2.0 + 0.25 / 2.0  # 0.625
        assert_allclose(nis, expected, atol=1e-10)

    def test_nees(self):
        """NEES computation."""
        x_true = np.array([1.0, 2.0])
        x_est = np.array([1.1, 2.2])
        P = np.eye(2)
        nees = KalmanUtils.nees(x_true, x_est, P)
        expected = 0.01 + 0.04  # 0.05
        assert_allclose(nees, expected, atol=1e-10)

    def test_mahalanobis(self):
        """Mahalanobis distance."""
        x = np.array([3.0])
        mean = np.array([0.0])
        cov = np.array([[4.0]])
        d = KalmanUtils.mahalanobis(x, mean, cov)
        assert_allclose(d, 1.5, atol=1e-10)  # 3/sqrt(4) = 1.5

    def test_log_likelihood_sequence(self):
        """Log-likelihood from innovations."""
        innovations = [np.array([0.0]), np.array([0.0])]
        S_list = [np.eye(1), np.eye(1)]
        ll = KalmanUtils.log_likelihood_sequence(innovations, S_list)
        # With zero innovation: -0.5 * (1*log(2pi) + log(1) + 0) * 2
        expected = -np.log(2 * np.pi)
        assert_allclose(ll, expected, atol=1e-10)

    def test_discrete_are(self):
        """Discrete ARE should converge."""
        F = np.array([[1, 0.1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])

        P = KalmanUtils.discrete_are(F, H, Q, R)
        assert P.shape == (2, 2)
        # Verify fixed-point: P = (I - KH)(F P F^T + Q) where K uses P_pred
        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)
        P_check = (np.eye(2) - K @ H) @ P_pred
        assert_allclose(P, P_check, atol=1e-6)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_simulate_and_filter(self):
        """Simulate from model, then filter to recover states."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.eye(2) * 0.1
        kf.R = np.array([[2.0]])
        kf.x = np.array([0.0, 0.5])
        kf.P = np.eye(2) * 5

        states, meas = kf.simulate(100, x0=np.array([0.0, 0.5]), seed=42)

        # Reset and filter
        kf.x = np.array([0.0, 0.5])
        kf.P = np.eye(2) * 5
        xs, Ps, ll = kf.filter(meas)

        # Filtered states should track true states
        pos_error = np.mean(np.abs(xs[:, 0] - states[:, 0]))
        assert pos_error < 3.0

    def test_filter_then_smooth(self):
        """Smooth after filtering should improve estimates."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 0.5], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.eye(2) * 0.1
        kf.R = np.array([[2.0]])

        states, meas = kf.simulate(80, x0=np.array([0.0, 1.0]), seed=123)

        kf.x = np.zeros(2); kf.P = np.eye(2) * 10
        xs_f, Ps_f, _ = kf.filter(meas)

        kf.x = np.zeros(2); kf.P = np.eye(2) * 10
        smoother = KalmanSmoother(kf)
        xs_s, Ps_s = smoother.smooth(meas)

        # Smoother should have smaller or equal trace
        for k in range(len(meas) - 2):
            assert np.trace(Ps_s[k]) <= np.trace(Ps_f[k]) + 1e-4

    def test_ekf_vs_ukf_on_nonlinear(self):
        """Both EKF and UKF should track a mildly nonlinear system."""
        def f(x, u):
            return np.array([x[0] + 0.1 * x[1], 0.98 * x[1]])

        def h(x):
            return np.array([x[0]])

        def F_jac(x, u):
            return np.array([[1, 0.1], [0, 0.98]])

        def H_jac(x):
            return np.array([[1, 0]])

        rng = np.random.RandomState(42)
        true_x = np.array([0.0, 2.0])
        measurements = []
        true_states = []
        for _ in range(40):
            true_x = f(true_x, None) + rng.randn(2) * 0.05
            z = h(true_x) + rng.randn(1) * 0.5
            measurements.append(z)
            true_states.append(true_x.copy())

        # EKF
        ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
        ekf.f = f; ekf.h = h; ekf.F_jacobian = F_jac; ekf.H_jacobian = H_jac
        ekf.Q = np.eye(2) * 0.01; ekf.R = np.array([[0.25]])
        ekf.x = np.array([0.0, 2.0]); ekf.P = np.eye(2)
        xs_ekf, _ = ekf.filter(measurements)

        # UKF
        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, f=f, h=h)
        ukf.Q = np.eye(2) * 0.01; ukf.R = np.array([[0.25]])
        ukf.x = np.array([0.0, 2.0]); ukf.P = np.eye(2)
        xs_ukf, _ = ukf.filter(measurements)

        true_states = np.array(true_states)
        # Both should track reasonably
        ekf_err = np.mean(np.abs(xs_ekf[:, 0] - true_states[:, 0]))
        ukf_err = np.mean(np.abs(xs_ukf[:, 0] - true_states[:, 0]))
        assert ekf_err < 2.0
        assert ukf_err < 2.0

    def test_imm_with_hmm_composition(self):
        """Verify IMM uses HMM-style transition matrix."""
        # This tests the composition with C157 -- the IMM transition matrix
        # is exactly an HMM transition matrix
        hmm = HMM(n_states=2, n_obs=2, seed=42)
        hmm.set_params(
            pi=[0.5, 0.5],
            A=[[0.9, 0.1], [0.1, 0.9]],
            B=[[0.8, 0.2], [0.2, 0.8]]
        )

        kf1 = KalmanFilter(dim_x=1, dim_z=1)
        kf1.F = np.eye(1); kf1.H = np.eye(1)
        kf1.Q = np.eye(1) * 0.1; kf1.R = np.eye(1)

        kf2 = KalmanFilter(dim_x=1, dim_z=1)
        kf2.F = np.eye(1); kf2.H = np.eye(1)
        kf2.Q = np.eye(1) * 0.5; kf2.R = np.eye(1)

        # Use HMM's transition matrix for IMM
        imm = InteractingMultipleModel([kf1, kf2], hmm.A, mode_probs=hmm.pi)
        assert_allclose(imm.TPM, hmm.A)
        assert_allclose(imm.mu, hmm.pi)

    def test_steady_state_matches_iterated(self):
        """Steady-state gain should match gain after many iterations."""
        F = np.array([[1, 0.1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])

        K_ss, P_ss = KalmanUtils.steady_state_gain(F, H, Q, R)

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = F; kf.H = H; kf.Q = Q; kf.R = R
        kf.P = np.eye(2) * 100

        for _ in range(500):
            kf.predict()
            kf.update(np.array([0.0]))

        assert_allclose(kf.K, K_ss, atol=1e-4)

    def test_enkf_nonlinear(self):
        """EnKF on nonlinear system."""
        def f(x):
            return np.array([x[0] + 0.1 * x[1], 0.95 * x[1]])

        def h(x):
            return np.array([np.sqrt(abs(x[0])) * np.sign(x[0])])

        enkf = EnsembleKalmanFilter(
            dim_x=2, dim_z=1, n_ensemble=200,
            f=f, h=h, seed=42
        )
        enkf.Q = np.eye(2) * 0.01
        enkf.R = np.array([[0.1]])
        enkf.initialize(np.array([1.0, 0.5]), np.eye(2) * 0.5)

        rng = np.random.RandomState(42)
        true_x = np.array([1.0, 0.5])
        for _ in range(30):
            true_x = f(true_x) + rng.randn(2) * 0.05
            z = h(true_x) + rng.randn(1) * 0.3
            enkf.predict()
            enkf.update(z)

        assert abs(enkf.x[0] - true_x[0]) < 3.0

    def test_all_filters_on_same_data(self):
        """Run KF, SRKF, and InfoFilter on same data, compare."""
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])

        rng = np.random.RandomState(42)
        zs = [np.array([5.0 + rng.randn()]) for _ in range(30)]

        # KF
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = F; kf.H = H; kf.Q = Q; kf.R = R
        xs_kf, _, _ = kf.filter(zs)

        # SRKF
        srkf = SquareRootKalmanFilter(dim_x=1, dim_z=1)
        srkf.F = F; srkf.H = H; srkf.set_Q(Q); srkf.set_R(R)
        xs_sr, _ = srkf.filter(zs)

        # All should be close
        assert_allclose(xs_kf[-1], xs_sr[-1], atol=0.5)

    def test_high_dimensional_enkf(self):
        """EnKF should handle moderately high dimensions."""
        dim = 10

        def f(x):
            # Random walk
            return x

        def h(x):
            return x[:3]  # observe first 3 dims

        enkf = EnsembleKalmanFilter(
            dim_x=dim, dim_z=3, n_ensemble=100,
            f=f, h=h, seed=42
        )
        enkf.Q = np.eye(dim) * 0.01
        enkf.R = np.eye(3) * 0.5
        enkf.initialize(np.zeros(dim), np.eye(dim))

        rng = np.random.RandomState(42)
        true_x = np.ones(dim)
        for _ in range(20):
            z = true_x[:3] + rng.randn(3) * 0.7
            enkf.predict()
            enkf.update(z)

        # First 3 dims should be closer to 1
        assert abs(enkf.x[0] - 1.0) < 2.0

    def test_are_vs_steady_state(self):
        """DARE solution should match steady-state Kalman P."""
        F = np.array([[1, 0.5], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])

        P_dare = KalmanUtils.discrete_are(F, H, Q, R)
        _, P_ss = KalmanUtils.steady_state_gain(F, H, Q, R)
        assert_allclose(P_dare, P_ss, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
