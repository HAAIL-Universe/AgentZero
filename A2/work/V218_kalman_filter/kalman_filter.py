"""
V218: Kalman Filter -- Continuous-state estimation for linear-Gaussian systems.

The Kalman filter is the continuous-state analog of the HMM (V215).
Where HMMs use discrete hidden states with forward/backward on probability tables,
the Kalman filter tracks continuous state vectors with Gaussian distributions
(mean vectors + covariance matrices).

Components:
1. KalmanFilter -- Standard linear Kalman filter (predict/update/smooth)
2. ExtendedKalmanFilter -- Nonlinear systems via local linearization (Jacobians)
3. UnscentedKalmanFilter -- Nonlinear systems via sigma points (no Jacobians needed)
4. InformationFilter -- Dual (inverse covariance) representation
5. SquareRootKalmanFilter -- Numerically stable via Cholesky factors
6. KalmanSmoother -- RTS (Rauch-Tung-Striebel) fixed-interval smoother

Uses NumPy for matrix operations (permitted by CLAUDE.md).

Composes with:
- V215 (HMM): discrete-to-continuous bridge, hybrid systems
- V213 (MDP): LQR/LQG control (Kalman + optimal control)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GaussianState:
    """Multivariate Gaussian: N(mean, covariance)."""
    mean: np.ndarray  # (n,)
    cov: np.ndarray   # (n, n)

    @property
    def dim(self) -> int:
        return len(self.mean)

    def mahalanobis(self, x: np.ndarray) -> float:
        """Mahalanobis distance of x from this Gaussian."""
        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        return float(np.sqrt(diff @ inv_cov @ diff))

    def log_likelihood(self, x: np.ndarray) -> float:
        """Log probability density at x."""
        k = self.dim
        diff = x - self.mean
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            return -np.inf
        inv_cov = np.linalg.inv(self.cov)
        return -0.5 * (k * np.log(2 * np.pi) + logdet + diff @ inv_cov @ diff)


@dataclass
class FilterResult:
    """Result of running a Kalman filter over a sequence of observations."""
    filtered_states: list  # list of GaussianState (after update at each step)
    predicted_states: list  # list of GaussianState (after predict at each step)
    log_likelihood: float   # total log-likelihood of observations
    innovations: list       # list of innovation vectors (y - H*x_pred)
    innovation_covs: list   # list of innovation covariance matrices


@dataclass
class SmootherResult:
    """Result of Kalman smoothing."""
    smoothed_states: list   # list of GaussianState
    smoother_gains: list    # list of smoother gain matrices


# ---------------------------------------------------------------------------
# Standard Kalman Filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Linear Kalman filter for the system:
        x_{t+1} = F * x_t + B * u_t + w_t,  w_t ~ N(0, Q)
        z_t     = H * x_t + v_t,             v_t ~ N(0, R)

    Parameters:
        F: State transition matrix (n x n)
        H: Observation matrix (m x n)
        Q: Process noise covariance (n x n)
        R: Measurement noise covariance (m x m)
        B: Control input matrix (n x k), optional
    """

    def __init__(self, F: np.ndarray, H: np.ndarray,
                 Q: np.ndarray, R: np.ndarray,
                 B: Optional[np.ndarray] = None):
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.B = np.asarray(B, dtype=float) if B is not None else None
        self.n = self.F.shape[0]  # state dimension
        self.m = self.H.shape[0]  # observation dimension

    def predict(self, state: GaussianState,
                u: Optional[np.ndarray] = None) -> GaussianState:
        """Predict step: propagate state through dynamics."""
        mean_pred = self.F @ state.mean
        if u is not None and self.B is not None:
            mean_pred = mean_pred + self.B @ np.asarray(u, dtype=float)
        cov_pred = self.F @ state.cov @ self.F.T + self.Q
        return GaussianState(mean_pred, cov_pred)

    def update(self, predicted: GaussianState,
               z: np.ndarray) -> tuple:
        """
        Update step: incorporate observation.
        Returns (updated_state, innovation, innovation_cov, log_lik).
        """
        z = np.asarray(z, dtype=float)
        # Innovation
        innovation = z - self.H @ predicted.mean
        S = self.H @ predicted.cov @ self.H.T + self.R  # innovation covariance

        # Kalman gain
        K = predicted.cov @ self.H.T @ np.linalg.inv(S)

        # Updated state
        mean_upd = predicted.mean + K @ innovation
        # Joseph form for numerical stability
        IKH = np.eye(self.n) - K @ self.H
        cov_upd = IKH @ predicted.cov @ IKH.T + K @ self.R @ K.T

        # Log-likelihood of this observation
        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (self.m * np.log(2 * np.pi) + logdet +
                      innovation @ np.linalg.inv(S) @ innovation)

        return GaussianState(mean_upd, cov_upd), innovation, S, float(ll)

    def filter(self, observations: list,
               initial: GaussianState,
               controls: Optional[list] = None) -> FilterResult:
        """Run filter over a sequence of observations."""
        state = initial
        filtered = []
        predicted = []
        innovations = []
        innovation_covs = []
        total_ll = 0.0

        for t, z in enumerate(observations):
            u = controls[t] if controls is not None else None
            pred = self.predict(state, u)
            predicted.append(pred)

            upd, innov, S, ll = self.update(pred, z)
            filtered.append(upd)
            innovations.append(innov)
            innovation_covs.append(S)
            total_ll += ll
            state = upd

        return FilterResult(filtered, predicted, total_ll,
                            innovations, innovation_covs)

    def smooth(self, observations: list,
               initial: GaussianState,
               controls: Optional[list] = None) -> SmootherResult:
        """RTS (Rauch-Tung-Striebel) fixed-interval smoother."""
        # Forward pass
        filt_result = self.filter(observations, initial, controls)
        T = len(observations)

        smoothed = [None] * T
        gains = [None] * (T - 1)
        smoothed[T - 1] = filt_result.filtered_states[T - 1]

        # Backward pass
        for t in range(T - 2, -1, -1):
            filt_t = filt_result.filtered_states[t]
            pred_tp1 = filt_result.predicted_states[t + 1]

            # Smoother gain
            G = filt_t.cov @ self.F.T @ np.linalg.inv(pred_tp1.cov)
            gains[t] = G

            # Smoothed estimate
            mean_s = filt_t.mean + G @ (smoothed[t + 1].mean - pred_tp1.mean)
            cov_s = filt_t.cov + G @ (smoothed[t + 1].cov - pred_tp1.cov) @ G.T

            smoothed[t] = GaussianState(mean_s, cov_s)

        return SmootherResult(smoothed, gains)


# ---------------------------------------------------------------------------
# Extended Kalman Filter (nonlinear via Jacobians)
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """
    Extended Kalman filter for nonlinear systems:
        x_{t+1} = f(x_t, u_t) + w_t,  w_t ~ N(0, Q)
        z_t     = h(x_t) + v_t,        v_t ~ N(0, R)

    Requires Jacobian functions F_jacobian(x, u) and H_jacobian(x).
    """

    def __init__(self, f: Callable, h: Callable,
                 F_jacobian: Callable, H_jacobian: Callable,
                 Q: np.ndarray, R: np.ndarray,
                 n: int, m: int):
        self.f = f              # state transition function
        self.h = h              # observation function
        self.F_jac = F_jacobian # df/dx
        self.H_jac = H_jacobian # dh/dx
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.n = n
        self.m = m

    def predict(self, state: GaussianState,
                u: Optional[np.ndarray] = None) -> GaussianState:
        mean_pred = self.f(state.mean, u)
        F = self.F_jac(state.mean, u)
        cov_pred = F @ state.cov @ F.T + self.Q
        return GaussianState(np.asarray(mean_pred, dtype=float), cov_pred)

    def update(self, predicted: GaussianState,
               z: np.ndarray) -> tuple:
        z = np.asarray(z, dtype=float)
        H = self.H_jac(predicted.mean)
        z_pred = self.h(predicted.mean)
        innovation = z - z_pred
        S = H @ predicted.cov @ H.T + self.R
        K = predicted.cov @ H.T @ np.linalg.inv(S)

        mean_upd = predicted.mean + K @ innovation
        IKH = np.eye(self.n) - K @ H
        cov_upd = IKH @ predicted.cov @ IKH.T + K @ self.R @ K.T

        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (self.m * np.log(2 * np.pi) + logdet +
                      innovation @ np.linalg.inv(S) @ innovation)

        return GaussianState(mean_upd, cov_upd), innovation, S, float(ll)

    def filter(self, observations: list,
               initial: GaussianState,
               controls: Optional[list] = None) -> FilterResult:
        state = initial
        filtered = []
        predicted = []
        innovations = []
        innovation_covs = []
        total_ll = 0.0

        for t, z in enumerate(observations):
            u = controls[t] if controls is not None else None
            pred = self.predict(state, u)
            predicted.append(pred)
            upd, innov, S, ll = self.update(pred, z)
            filtered.append(upd)
            innovations.append(innov)
            innovation_covs.append(S)
            total_ll += ll
            state = upd

        return FilterResult(filtered, predicted, total_ll,
                            innovations, innovation_covs)


# ---------------------------------------------------------------------------
# Unscented Kalman Filter (sigma points, no Jacobians)
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    """
    Unscented Kalman filter for nonlinear systems.
    Uses sigma points to capture mean and covariance through nonlinear transforms.

    Parameters:
        f: State transition function f(x, u) -> x'
        h: Observation function h(x) -> z
        Q: Process noise covariance
        R: Measurement noise covariance
        n: State dimension
        m: Observation dimension
        alpha: Spread of sigma points (default 1e-3)
        beta: Prior knowledge parameter (2 for Gaussian)
        kappa: Secondary scaling (default 0)
    """

    def __init__(self, f: Callable, h: Callable,
                 Q: np.ndarray, R: np.ndarray,
                 n: int, m: int,
                 alpha: float = 0.5, beta: float = 2.0, kappa: float = None):
        self.f = f
        self.h = h
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        if kappa is None:
            kappa = max(0.0, 3.0 - n)  # standard choice for Gaussians

        # Compute weights
        lam = alpha ** 2 * (n + kappa) - n
        self.lam = lam
        self.n_sigma = 2 * n + 1

        # Mean weights
        self.Wm = np.zeros(self.n_sigma)
        self.Wm[0] = lam / (n + lam)
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1.0 / (2 * (n + lam))

        # Covariance weights
        self.Wc = np.zeros(self.n_sigma)
        self.Wc[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)
        for i in range(1, self.n_sigma):
            self.Wc[i] = 1.0 / (2 * (n + lam))

    def _sigma_points(self, state: GaussianState) -> np.ndarray:
        """Generate 2n+1 sigma points."""
        n = self.n
        sigma = np.zeros((self.n_sigma, n))
        sigma[0] = state.mean

        scaled_cov = (n + self.lam) * state.cov
        # Ensure positive definite
        scaled_cov = 0.5 * (scaled_cov + scaled_cov.T)
        eigvals = np.linalg.eigvalsh(scaled_cov)
        if eigvals[0] < 1e-10:
            scaled_cov = scaled_cov + (1e-8 - min(eigvals[0], 0)) * np.eye(n)
        try:
            L = np.linalg.cholesky(scaled_cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(scaled_cov + 1e-6 * np.eye(n))

        for i in range(n):
            sigma[i + 1] = state.mean + L[i]
            sigma[n + i + 1] = state.mean - L[i]

        return sigma

    def predict(self, state: GaussianState,
                u: Optional[np.ndarray] = None) -> GaussianState:
        sigma = self._sigma_points(state)

        # Propagate sigma points through f
        sigma_pred = np.array([self.f(s, u) for s in sigma])

        # Recover mean and covariance
        mean_pred = np.zeros(self.n)
        for i in range(self.n_sigma):
            mean_pred += self.Wm[i] * sigma_pred[i]

        cov_pred = np.zeros((self.n, self.n))
        for i in range(self.n_sigma):
            d = sigma_pred[i] - mean_pred
            cov_pred += self.Wc[i] * np.outer(d, d)
        cov_pred += self.Q

        return GaussianState(mean_pred, cov_pred)

    def update(self, predicted: GaussianState,
               z: np.ndarray) -> tuple:
        z = np.asarray(z, dtype=float)
        sigma = self._sigma_points(predicted)

        # Propagate sigma points through h
        z_sigma = np.array([self.h(s) for s in sigma])

        # Predicted measurement mean
        z_mean = np.zeros(self.m)
        for i in range(self.n_sigma):
            z_mean += self.Wm[i] * z_sigma[i]

        # Innovation covariance S and cross-covariance Pxz
        S = np.zeros((self.m, self.m))
        Pxz = np.zeros((self.n, self.m))
        for i in range(self.n_sigma):
            dz = z_sigma[i] - z_mean
            dx = sigma[i] - predicted.mean
            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        S += self.R

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        innovation = z - z_mean
        mean_upd = predicted.mean + K @ innovation
        cov_upd = predicted.cov - K @ S @ K.T
        # Force symmetry and PSD
        cov_upd = 0.5 * (cov_upd + cov_upd.T)
        eigvals = np.linalg.eigvalsh(cov_upd)
        if eigvals[0] < 0:
            cov_upd = cov_upd + (1e-8 - eigvals[0]) * np.eye(self.n)

        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (self.m * np.log(2 * np.pi) + logdet +
                      innovation @ np.linalg.inv(S) @ innovation)

        return GaussianState(mean_upd, cov_upd), innovation, S, float(ll)

    def filter(self, observations: list,
               initial: GaussianState,
               controls: Optional[list] = None) -> FilterResult:
        state = initial
        filtered = []
        predicted = []
        innovations = []
        innovation_covs = []
        total_ll = 0.0

        for t, z in enumerate(observations):
            u = controls[t] if controls is not None else None
            pred = self.predict(state, u)
            predicted.append(pred)
            upd, innov, S, ll = self.update(pred, z)
            filtered.append(upd)
            innovations.append(innov)
            innovation_covs.append(S)
            total_ll += ll
            state = upd

        return FilterResult(filtered, predicted, total_ll,
                            innovations, innovation_covs)


# ---------------------------------------------------------------------------
# Information Filter (inverse covariance / canonical form)
# ---------------------------------------------------------------------------

@dataclass
class InformationState:
    """Information form: stores precision matrix and info vector."""
    info_vector: np.ndarray   # Lambda * mean  (n,)
    info_matrix: np.ndarray   # Lambda = cov^{-1} (n, n)

    def to_gaussian(self) -> GaussianState:
        cov = np.linalg.inv(self.info_matrix)
        mean = cov @ self.info_vector
        return GaussianState(mean, cov)

    @staticmethod
    def from_gaussian(gs: GaussianState) -> 'InformationState':
        info_matrix = np.linalg.inv(gs.cov)
        info_vector = info_matrix @ gs.mean
        return InformationState(info_vector, info_matrix)


class InformationFilter:
    """
    Kalman filter in information (canonical) form.
    Efficient for multi-sensor fusion (updates are additive).

    System model same as KalmanFilter.
    """

    def __init__(self, F: np.ndarray, H: np.ndarray,
                 Q: np.ndarray, R: np.ndarray):
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.n = self.F.shape[0]
        self.m = self.H.shape[0]
        # Precompute R^{-1} and information contribution
        self.R_inv = np.linalg.inv(self.R)
        self.info_contribution_matrix = self.H.T @ self.R_inv @ self.H
        self.info_contribution_vector_factor = self.H.T @ self.R_inv

    def predict(self, state: InformationState) -> InformationState:
        """Predict in information form."""
        # Convert to moment form for prediction (info predict is expensive)
        gs = state.to_gaussian()
        mean_pred = self.F @ gs.mean
        cov_pred = self.F @ gs.cov @ self.F.T + self.Q
        pred_gs = GaussianState(mean_pred, cov_pred)
        return InformationState.from_gaussian(pred_gs)

    def update(self, predicted: InformationState,
               z: np.ndarray) -> InformationState:
        """Update: simply add information contributions (the key advantage)."""
        z = np.asarray(z, dtype=float)
        info_matrix = predicted.info_matrix + self.info_contribution_matrix
        info_vector = predicted.info_vector + self.info_contribution_vector_factor @ z
        return InformationState(info_vector, info_matrix)

    def multi_sensor_update(self, predicted: InformationState,
                            measurements: list,
                            H_list: list,
                            R_list: list) -> InformationState:
        """Fuse multiple sensor measurements (additive in information form)."""
        info_matrix = predicted.info_matrix.copy()
        info_vector = predicted.info_vector.copy()

        for z, H, R in zip(measurements, H_list, R_list):
            z = np.asarray(z, dtype=float)
            H = np.asarray(H, dtype=float)
            R_inv = np.linalg.inv(np.asarray(R, dtype=float))
            info_matrix = info_matrix + H.T @ R_inv @ H
            info_vector = info_vector + H.T @ R_inv @ z

        return InformationState(info_vector, info_matrix)

    def filter(self, observations: list,
               initial: InformationState) -> list:
        """Run filter, return list of InformationState."""
        state = initial
        results = []
        for z in observations:
            pred = self.predict(state)
            state = self.update(pred, z)
            results.append(state)
        return results


# ---------------------------------------------------------------------------
# Square Root Kalman Filter (Cholesky factor propagation)
# ---------------------------------------------------------------------------

class SquareRootKalmanFilter:
    """
    Square-root Kalman filter: propagates Cholesky factor of covariance.
    Guarantees positive semi-definiteness and has better numerical conditioning.

    Uses QR decomposition for factor updates.
    """

    def __init__(self, F: np.ndarray, H: np.ndarray,
                 Q: np.ndarray, R: np.ndarray):
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.n = self.F.shape[0]
        self.m = self.H.shape[0]
        # Regularize Q if not positive definite (e.g., rank-deficient noise)
        try:
            self.S_Q = np.linalg.cholesky(self.Q)
        except np.linalg.LinAlgError:
            self.S_Q = np.linalg.cholesky(self.Q + 1e-10 * np.eye(self.n))
        self.S_R = np.linalg.cholesky(self.R)

    def predict(self, mean: np.ndarray,
                S_P: np.ndarray) -> tuple:
        """
        Predict step with square-root factor.
        S_P: lower Cholesky factor of prior covariance (n x n).
        Returns (mean_pred, S_pred).
        """
        mean_pred = self.F @ mean

        # P_pred = F @ P @ F^T + Q -> Cholesky
        P_pred = self.F @ (S_P @ S_P.T) @ self.F.T + self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        try:
            S_pred = np.linalg.cholesky(P_pred)
        except np.linalg.LinAlgError:
            S_pred = np.linalg.cholesky(P_pred + 1e-10 * np.eye(self.n))

        return mean_pred, S_pred

    def update(self, mean_pred: np.ndarray, S_pred: np.ndarray,
               z: np.ndarray) -> tuple:
        """
        Update step with square-root factor.
        Returns (mean_upd, S_upd, innovation, S_innovation).
        """
        z = np.asarray(z, dtype=float)
        P_pred = S_pred @ S_pred.T
        innovation = z - self.H @ mean_pred

        # Innovation covariance
        S_mat = self.H @ P_pred @ self.H.T + self.R
        S_mat = 0.5 * (S_mat + S_mat.T)
        try:
            S_innov = np.linalg.cholesky(S_mat)
        except np.linalg.LinAlgError:
            S_innov = np.linalg.cholesky(S_mat + 1e-10 * np.eye(self.m))

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S_mat)

        mean_upd = mean_pred + K @ innovation

        # Joseph form for updated covariance, then re-Cholesky
        IKH = np.eye(self.n) - K @ self.H
        P_upd = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)
        try:
            S_upd = np.linalg.cholesky(P_upd)
        except np.linalg.LinAlgError:
            S_upd = np.linalg.cholesky(P_upd + 1e-10 * np.eye(self.n))

        return mean_upd, S_upd, innovation, S_innov

    def filter(self, observations: list,
               initial: GaussianState) -> FilterResult:
        """Run square-root filter over observations."""
        mean = initial.mean.copy()
        try:
            S = np.linalg.cholesky(initial.cov)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(initial.cov + 1e-10 * np.eye(self.n))

        filtered = []
        predicted = []
        innovations = []
        innovation_covs = []
        total_ll = 0.0

        for z in observations:
            mean_pred, S_pred = self.predict(mean, S)
            predicted.append(GaussianState(mean_pred, S_pred @ S_pred.T))

            mean, S, innov, S_innov = self.update(mean_pred, S_pred, z)
            filtered.append(GaussianState(mean.copy(), S @ S.T))
            innovations.append(innov)

            S_full = S_innov @ S_innov.T
            innovation_covs.append(S_full)

            # Log-likelihood
            sign, logdet = np.linalg.slogdet(S_full)
            ll = -0.5 * (self.m * np.log(2 * np.pi) + logdet +
                          innov @ np.linalg.inv(S_full) @ innov)
            total_ll += ll

        return FilterResult(filtered, predicted, total_ll,
                            innovations, innovation_covs)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def steady_state_gain(F: np.ndarray, H: np.ndarray,
                      Q: np.ndarray, R: np.ndarray,
                      max_iter: int = 1000,
                      tol: float = 1e-10) -> tuple:
    """
    Compute steady-state Kalman gain by iterating the DARE
    (Discrete Algebraic Riccati Equation).

    Returns (K_ss, P_ss) -- steady-state gain and covariance.
    """
    F = np.asarray(F, dtype=float)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    n = F.shape[0]

    P = np.eye(n)  # initial covariance guess
    for _ in range(max_iter):
        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        P_new = (np.eye(n) - K @ H) @ P_pred

        if np.max(np.abs(P_new - P)) < tol:
            return K, P_new
        P = P_new

    return K, P


def simulate_linear_system(F: np.ndarray, H: np.ndarray,
                           Q: np.ndarray, R: np.ndarray,
                           x0: np.ndarray, T: int,
                           B: Optional[np.ndarray] = None,
                           controls: Optional[list] = None,
                           rng: Optional[np.random.Generator] = None
                           ) -> tuple:
    """
    Simulate a linear-Gaussian system for T steps.

    Returns (states, observations) where:
        states: list of true state vectors (T+1 including x0)
        observations: list of observation vectors (T)
    """
    if rng is None:
        rng = np.random.default_rng()

    F = np.asarray(F, dtype=float)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    n = F.shape[0]
    m = H.shape[0]

    states = [np.asarray(x0, dtype=float)]
    observations = []

    for t in range(T):
        x = states[-1]
        # Process noise
        w = rng.multivariate_normal(np.zeros(n), Q)
        x_next = F @ x + w
        if B is not None and controls is not None:
            x_next = x_next + B @ np.asarray(controls[t], dtype=float)

        # Observation noise
        v = rng.multivariate_normal(np.zeros(m), R)
        z = H @ x_next + v

        states.append(x_next)
        observations.append(z)

    return states, observations


def compare_filters(observations: list, true_states: list,
                    filter_results: dict) -> dict:
    """
    Compare multiple filter results against true states.

    Args:
        observations: list of observation vectors
        true_states: list of true state vectors (len = len(obs) + 1)
        filter_results: dict of {name: FilterResult}

    Returns dict of {name: {rmse, mean_nees, log_likelihood}}.
    """
    T = len(observations)
    comparison = {}

    for name, result in filter_results.items():
        # RMSE
        errors = []
        nees_values = []
        for t in range(T):
            true_x = true_states[t + 1]  # states[0] is initial
            est = result.filtered_states[t]
            err = true_x - est.mean
            errors.append(np.sum(err ** 2))

            # NEES (Normalized Estimation Error Squared)
            try:
                inv_cov = np.linalg.inv(est.cov)
                nees = float(err @ inv_cov @ err)
                nees_values.append(nees)
            except np.linalg.LinAlgError:
                pass

        rmse = float(np.sqrt(np.mean(errors)))
        mean_nees = float(np.mean(nees_values)) if nees_values else float('nan')

        comparison[name] = {
            'rmse': rmse,
            'mean_nees': mean_nees,
            'log_likelihood': result.log_likelihood
        }

    return comparison


# ---------------------------------------------------------------------------
# Composition with V215 (HMM bridge)
# ---------------------------------------------------------------------------

def discretize_kalman(kf: KalmanFilter, state_grids: list,
                      obs_grids: list) -> dict:
    """
    Approximate a linear Kalman system as a discrete HMM.

    Discretizes the continuous state and observation spaces onto grids,
    computing approximate transition and emission probabilities.

    Args:
        kf: KalmanFilter instance
        state_grids: list of 1D arrays, one per state dimension
        obs_grids: list of 1D arrays, one per observation dimension

    Returns dict with keys: states, observations, initial, transition, emission
    suitable for constructing V215 HiddenMarkovModel.
    """
    from itertools import product as cartesian

    # Create state grid points
    state_points = list(cartesian(*state_grids))
    state_names = [f"s{'_'.join(f'{v:.1f}' for v in s)}" for s in state_points]

    obs_points = list(cartesian(*obs_grids))
    obs_names = [f"o{'_'.join(f'{v:.1f}' for v in o)}" for o in obs_points]

    n_states = len(state_points)
    n_obs = len(obs_points)

    # Compute transition probabilities
    transition = {}
    for i, (sp, sn) in enumerate(zip(state_points, state_names)):
        sp_arr = np.array(sp)
        # Mean of next state
        mean_next = kf.F @ sp_arr
        trans_probs = {}
        raw = []
        for j, (sp2, sn2) in enumerate(zip(state_points, state_names)):
            diff = np.array(sp2) - mean_next
            # Unnormalized Gaussian density
            try:
                inv_Q = np.linalg.inv(kf.Q)
                log_p = -0.5 * diff @ inv_Q @ diff
                raw.append((sn2, np.exp(log_p)))
            except np.linalg.LinAlgError:
                raw.append((sn2, 0.0))

        total = sum(v for _, v in raw)
        if total > 0:
            for sn2, v in raw:
                trans_probs[sn2] = v / total
        else:
            # Uniform fallback
            for sn2, _ in raw:
                trans_probs[sn2] = 1.0 / n_states
        transition[sn] = trans_probs

    # Compute emission probabilities
    emission = {}
    for i, (sp, sn) in enumerate(zip(state_points, state_names)):
        sp_arr = np.array(sp)
        mean_obs = kf.H @ sp_arr
        emit_probs = {}
        raw = []
        for j, (op, on) in enumerate(zip(obs_points, obs_names)):
            diff = np.array(op) - mean_obs
            try:
                inv_R = np.linalg.inv(kf.R)
                log_p = -0.5 * diff @ inv_R @ diff
                raw.append((on, np.exp(log_p)))
            except np.linalg.LinAlgError:
                raw.append((on, 0.0))

        total = sum(v for _, v in raw)
        if total > 0:
            for on, v in raw:
                emit_probs[on] = v / total
        else:
            for on, _ in raw:
                emit_probs[on] = 1.0 / n_obs
        emission[sn] = emit_probs

    # Uniform initial distribution
    initial = {sn: 1.0 / n_states for sn in state_names}

    return {
        'states': state_names,
        'observations': obs_names,
        'initial': initial,
        'transition': transition,
        'emission': emission,
    }


# ---------------------------------------------------------------------------
# LQR / LQG Control (compose with V213 MDP concepts)
# ---------------------------------------------------------------------------

def lqr_gain(F: np.ndarray, B: np.ndarray,
             Q_cost: np.ndarray, R_cost: np.ndarray,
             max_iter: int = 1000, tol: float = 1e-10) -> tuple:
    """
    Compute the steady-state LQR gain for the system:
        x_{t+1} = F*x_t + B*u_t
        cost = sum(x^T Q x + u^T R u)

    Solves the DARE (Discrete Algebraic Riccati Equation):
        P = F^T P F - F^T P B (R + B^T P B)^{-1} B^T P F + Q

    Returns (K, P) where u* = -K*x is the optimal control.
    """
    F = np.asarray(F, dtype=float)
    B = np.asarray(B, dtype=float)
    Q_cost = np.asarray(Q_cost, dtype=float)
    R_cost = np.asarray(R_cost, dtype=float)

    P = Q_cost.copy()
    for _ in range(max_iter):
        K = np.linalg.inv(R_cost + B.T @ P @ B) @ B.T @ P @ F
        P_new = F.T @ P @ F - F.T @ P @ B @ K + Q_cost

        if np.max(np.abs(P_new - P)) < tol:
            K = np.linalg.inv(R_cost + B.T @ P_new @ B) @ B.T @ P_new @ F
            return K, P_new
        P = P_new

    K = np.linalg.inv(R_cost + B.T @ P @ B) @ B.T @ P @ F
    return K, P


def lqg_controller(kf: KalmanFilter, Q_cost: np.ndarray,
                    R_cost: np.ndarray) -> tuple:
    """
    LQG controller: separation principle -- Kalman filter + LQR.

    Returns (K_lqr, K_kalman, P_lqr) where:
        - K_lqr: LQR gain (u = -K_lqr @ x_hat)
        - K_kalman: steady-state Kalman gain
        - P_lqr: LQR cost matrix
    """
    assert kf.B is not None, "LQG requires control input matrix B"

    K_lqr, P_lqr = lqr_gain(kf.F, kf.B, Q_cost, R_cost)
    K_kalman, _ = steady_state_gain(kf.F, kf.H, kf.Q, kf.R)

    return K_lqr, K_kalman, P_lqr


def simulate_lqg(kf: KalmanFilter, K_lqr: np.ndarray,
                  x0: np.ndarray, T: int,
                  initial_cov: Optional[np.ndarray] = None,
                  rng: Optional[np.random.Generator] = None) -> dict:
    """
    Simulate LQG control: Kalman filter estimates state, LQR controls.

    Returns dict with keys:
        true_states, estimated_states, controls, observations, costs
    """
    if rng is None:
        rng = np.random.default_rng()

    n = kf.n
    if initial_cov is None:
        initial_cov = np.eye(n)

    x_true = np.asarray(x0, dtype=float)
    state_est = GaussianState(np.zeros(n), initial_cov)

    true_states = [x_true.copy()]
    estimated_states = [state_est.mean.copy()]
    controls = []
    observations = []
    costs = []

    for t in range(T):
        # Control based on estimate
        u = -K_lqr @ state_est.mean
        controls.append(u)

        # True dynamics
        w = rng.multivariate_normal(np.zeros(n), kf.Q)
        x_true = kf.F @ x_true + kf.B @ u + w
        true_states.append(x_true.copy())

        # Observation
        v = rng.multivariate_normal(np.zeros(kf.m), kf.R)
        z = kf.H @ x_true + v
        observations.append(z)

        # Kalman filter update
        pred = kf.predict(state_est, u)
        state_est, _, _, _ = kf.update(pred, z)
        estimated_states.append(state_est.mean.copy())

        # Cost
        cost = float(x_true @ x_true + u @ u)
        costs.append(cost)

    return {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'controls': controls,
        'observations': observations,
        'costs': costs,
    }
