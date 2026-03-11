"""
C158: Kalman Filter
Composing C157 (Hidden Markov Model)

State-space models with continuous state estimation.

Components:
- KalmanFilter: Classic linear Kalman filter (predict/update/smooth)
- ExtendedKalmanFilter: EKF for nonlinear systems (Jacobian linearization)
- UnscentedKalmanFilter: UKF with sigma points
- InformationFilter: Information (inverse covariance) form
- SquareRootKalmanFilter: Cholesky-based numerically stable form
- KalmanSmoother: RTS fixed-interval smoother
- InteractingMultipleModel: IMM for switching dynamics (composes C157 HMM)
- EnsembleKalmanFilter: EnKF for high-dimensional systems
- KalmanUtils: Observability, controllability, steady-state, NIS/NEES
"""

import numpy as np
from numpy.linalg import inv, det, cholesky, solve
from scipy.linalg import block_diag, sqrtm, cho_factor, cho_solve
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C157_hidden_markov_model'))
from hidden_markov_model import HMM


# ---------------------------------------------------------------------------
# Classic Linear Kalman Filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Linear Kalman filter for state-space model:
        x_{k+1} = F x_k + B u_k + w_k,  w_k ~ N(0, Q)
        z_k = H x_k + v_k,              v_k ~ N(0, R)
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros(dim_x)          # state estimate
        self.P = np.eye(dim_x)            # state covariance
        self.F = np.eye(dim_x)            # state transition
        self.H = np.zeros((dim_z, dim_x)) # observation model
        self.R = np.eye(dim_z)            # measurement noise
        self.Q = np.eye(dim_x)            # process noise
        self.B = np.zeros((dim_x, max(dim_u, 1)))  # control input

        # Post-update storage
        self.x_prior = None
        self.P_prior = None
        self.y = None  # innovation
        self.S = None  # innovation covariance
        self.K = None  # Kalman gain
        self.log_likelihood = 0.0

    def predict(self, u=None):
        """Predict step."""
        if u is not None:
            self.x = self.F @ self.x + self.B @ np.asarray(u)
        else:
            self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z):
        """Update step with measurement z."""
        z = np.asarray(z, dtype=float)
        self.y = z - self.H @ self.x  # innovation
        self.S = self.H @ self.P @ self.H.T + self.R  # innovation cov
        self.K = self.P @ self.H.T @ inv(self.S)  # Kalman gain
        self.x = self.x + self.K @ self.y
        I = np.eye(self.dim_x)
        # Joseph form for numerical stability
        IKH = I - self.K @ self.H
        self.P = IKH @ self.P @ IKH.T + self.K @ self.R @ self.K.T

        # Log-likelihood of this measurement
        n = len(self.y)
        self.log_likelihood = -0.5 * (
            n * np.log(2 * np.pi) + np.log(max(det(self.S), 1e-300))
            + self.y @ inv(self.S) @ self.y
        )

    def filter(self, measurements, controls=None):
        """Run filter over sequence. Returns (xs, Ps, log_lik)."""
        xs, Ps = [], []
        total_ll = 0.0
        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            self.predict(u)
            self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
            total_ll += self.log_likelihood
        return np.array(xs), np.array(Ps), total_ll

    def batch_filter(self, measurements, controls=None):
        """Run filter, return (xs, Ps, x_priors, P_priors, Ks)."""
        n = len(measurements)
        xs = np.zeros((n, self.dim_x))
        Ps = np.zeros((n, self.dim_x, self.dim_x))
        x_priors = np.zeros((n, self.dim_x))
        P_priors = np.zeros((n, self.dim_x, self.dim_x))
        Ks = np.zeros((n, self.dim_x, self.dim_z))

        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            self.predict(u)
            x_priors[k] = self.x_prior
            P_priors[k] = self.P_prior
            self.update(z)
            xs[k] = self.x
            Ps[k] = self.P
            Ks[k] = self.K

        return xs, Ps, x_priors, P_priors, Ks

    def simulate(self, n_steps, x0=None, controls=None, seed=None):
        """Simulate state-space model. Returns (states, measurements)."""
        rng = np.random.RandomState(seed)
        states = np.zeros((n_steps, self.dim_x))
        measurements = np.zeros((n_steps, self.dim_z))

        x = x0 if x0 is not None else rng.multivariate_normal(self.x, self.P)
        for k in range(n_steps):
            if k > 0:
                u = controls[k] if controls is not None else np.zeros(max(self.dim_u, 1))
                x = self.F @ x + self.B @ u + rng.multivariate_normal(
                    np.zeros(self.dim_x), self.Q
                )
            states[k] = x
            measurements[k] = self.H @ x + rng.multivariate_normal(
                np.zeros(self.dim_z), self.R
            )
        return states, measurements


# ---------------------------------------------------------------------------
# Extended Kalman Filter
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear systems:
        x_{k+1} = f(x_k, u_k) + w_k
        z_k = h(x_k) + v_k

    User provides f, h and their Jacobians F_jacobian, H_jacobian.
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

        # Nonlinear functions (to be set by user)
        self.f = None  # f(x, u) -> x_next
        self.h = None  # h(x) -> z
        self.F_jacobian = None  # df/dx at (x, u) -> matrix
        self.H_jacobian = None  # dh/dx at (x) -> matrix

        self.x_prior = None
        self.P_prior = None
        self.y = None
        self.S = None
        self.K = None

    def predict(self, u=None):
        """Predict using nonlinear f and Jacobian F."""
        F = self.F_jacobian(self.x, u)
        self.x = self.f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z):
        """Update using nonlinear h and Jacobian H."""
        z = np.asarray(z, dtype=float)
        H = self.H_jacobian(self.x)
        self.y = z - self.h(self.x)
        self.S = H @ self.P @ H.T + self.R
        self.K = self.P @ H.T @ inv(self.S)
        self.x = self.x + self.K @ self.y
        I = np.eye(self.dim_x)
        IKH = I - self.K @ H
        self.P = IKH @ self.P @ IKH.T + self.K @ self.R @ self.K.T

    def filter(self, measurements, controls=None):
        """Run EKF over sequence."""
        xs, Ps = [], []
        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            self.predict(u)
            self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
        return np.array(xs), np.array(Ps)


# ---------------------------------------------------------------------------
# Unscented Kalman Filter
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    """Unscented Kalman Filter using sigma points.
    No Jacobians needed -- handles nonlinearities via deterministic sampling.
    """

    def __init__(self, dim_x, dim_z, f, h, alpha=1e-3, beta=2.0, kappa=0.0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.f = f  # f(x, u) -> x_next
        self.h = h  # h(x) -> z

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

        # Sigma point parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

        self.x_prior = None
        self.P_prior = None

    def _compute_weights(self):
        n = self.dim_x
        lam = self.alpha ** 2 * (n + self.kappa) - n
        self.lam = lam

        self.n_sigma = 2 * n + 1
        self.Wm = np.full(self.n_sigma, 1.0 / (2 * (n + lam)))
        self.Wc = np.full(self.n_sigma, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)

    def _sigma_points(self, x, P):
        n = self.dim_x
        # Ensure P is symmetric positive definite
        P_sym = (P + P.T) / 2
        P_sym += np.eye(n) * 1e-10
        try:
            L = cholesky(P_sym * (n + self.lam))
        except np.linalg.LinAlgError:
            # Fallback: use sqrtm
            L = np.real(sqrtm(P_sym * (n + self.lam)))

        sigmas = np.zeros((self.n_sigma, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + L[:, i]
            sigmas[n + i + 1] = x - L[:, i]
        return sigmas

    def _unscented_transform(self, sigmas, Wm, Wc, noise_cov):
        """Compute mean and covariance from sigma points."""
        mean = Wm @ sigmas
        n = sigmas.shape[1]
        cov = np.zeros((n, n))
        for i, s in enumerate(sigmas):
            d = s - mean
            cov += Wc[i] * np.outer(d, d)
        cov += noise_cov
        return mean, cov

    def predict(self, u=None):
        sigmas = self._sigma_points(self.x, self.P)
        # Transform through f
        sigmas_f = np.array([self.f(s, u) for s in sigmas])
        self.x, self.P = self._unscented_transform(
            sigmas_f, self.Wm, self.Wc, self.Q
        )
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self._sigmas_f = sigmas_f

    def update(self, z):
        z = np.asarray(z, dtype=float)
        sigmas_f = self._sigmas_f
        # Transform through h
        sigmas_h = np.array([self.h(s) for s in sigmas_f])
        z_mean, S = self._unscented_transform(
            sigmas_h, self.Wm, self.Wc, self.R
        )

        # Cross-covariance
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.n_sigma):
            dx = sigmas_f[i] - self.x
            dz = sigmas_h[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ inv(S)
        self.x = self.x + K @ (z - z_mean)
        self.P = self.P - K @ S @ K.T

    def filter(self, measurements, controls=None):
        xs, Ps = [], []
        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            self.predict(u)
            self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
        return np.array(xs), np.array(Ps)


# ---------------------------------------------------------------------------
# Information Filter
# ---------------------------------------------------------------------------

class InformationFilter:
    """Information filter -- dual of Kalman filter using information (inverse cov) form.
    Useful for multi-sensor fusion and decentralized estimation.

    State: y = P^{-1} x  (information state)
           Y = P^{-1}    (information matrix)
    """

    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.Y = np.eye(dim_x)  # information matrix (P^{-1})
        self.y = np.zeros(dim_x)  # information state (P^{-1} x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    @property
    def x(self):
        """State estimate."""
        return solve(self.Y, self.y)

    @property
    def P(self):
        """State covariance."""
        return inv(self.Y)

    def predict(self):
        """Predict step in information form."""
        # Save current state estimate before updating Y
        x_curr = solve(self.Y, self.y)
        F_inv = inv(self.F)
        M = F_inv.T @ self.Y @ F_inv
        Q_inv = inv(self.Q)
        # Woodbury/matrix inversion lemma for (F P F^T + Q)^{-1}
        self.Y = M - M @ inv(M + Q_inv) @ M
        # Predicted state: x_pred = F x_curr
        x_pred = self.F @ x_curr
        self.y = self.Y @ x_pred

    def update(self, z):
        """Update step -- simple additive in information form."""
        z = np.asarray(z, dtype=float)
        R_inv = inv(self.R)
        I_z = self.H.T @ R_inv @ self.H  # information contribution
        i_z = self.H.T @ R_inv @ z        # information contribution
        self.Y = self.Y + I_z
        self.y = self.y + i_z

    def fuse(self, measurements, H_list=None, R_list=None):
        """Fuse multiple measurements at once (multi-sensor fusion)."""
        for k, z in enumerate(measurements):
            H = H_list[k] if H_list is not None else self.H
            R = R_list[k] if R_list is not None else self.R
            z = np.asarray(z, dtype=float)
            R_inv = inv(R)
            self.Y += H.T @ R_inv @ H
            self.y += H.T @ R_inv @ z

    def filter(self, measurements):
        """Run information filter over sequence."""
        xs, Ps = [], []
        for z in measurements:
            self.predict()
            self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
        return np.array(xs), np.array(Ps)


# ---------------------------------------------------------------------------
# Square Root Kalman Filter
# ---------------------------------------------------------------------------

class SquareRootKalmanFilter:
    """Cholesky-based square root Kalman filter.
    Propagates S where P = S S^T for numerical stability.
    Uses Potter's method -- recomputes Cholesky from Joseph-form P each step.
    """

    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros(dim_x)
        self.S = np.eye(dim_x)  # S such that P = S @ S.T
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def set_Q(self, Q):
        self.Q = np.array(Q, dtype=float)

    def set_R(self, R):
        self.R = np.array(R, dtype=float)

    @property
    def P(self):
        return self.S @ self.S.T

    def _recholesky(self, P):
        """Recompute Cholesky, with regularization fallback."""
        P = (P + P.T) / 2
        try:
            return cholesky(P)
        except np.linalg.LinAlgError:
            P += np.eye(self.dim_x) * 1e-10
            return cholesky(P)

    def predict(self):
        """Predict: propagate P = F S S^T F^T + Q, then re-Cholesky."""
        P_pred = self.F @ self.P @ self.F.T + self.Q
        self.S = self._recholesky(P_pred)
        self.x = self.F @ self.x

    def update(self, z):
        """Update using Joseph form, then re-Cholesky."""
        z = np.asarray(z, dtype=float)
        P = self.P
        y = z - self.H @ self.x
        S_innov = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ inv(S_innov)
        self.x = self.x + K @ y
        # Joseph form
        I = np.eye(self.dim_x)
        IKH = I - K @ self.H
        P_new = IKH @ P @ IKH.T + K @ self.R @ K.T
        self.S = self._recholesky(P_new)

    def filter(self, measurements):
        xs, Ps = [], []
        for z in measurements:
            self.predict()
            self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
        return np.array(xs), np.array(Ps)


# ---------------------------------------------------------------------------
# Kalman Smoother (Rauch-Tung-Striebel)
# ---------------------------------------------------------------------------

class KalmanSmoother:
    """RTS fixed-interval smoother. Runs forward KF then backward smoothing pass."""

    def __init__(self, kf):
        """Takes a KalmanFilter instance."""
        self.kf = kf

    def smooth(self, measurements, controls=None):
        """Run forward filter + backward smoother.
        Returns (xs_smooth, Ps_smooth).
        """
        kf = self.kf
        # Reset filter state
        x0 = kf.x.copy()
        P0 = kf.P.copy()

        # Forward pass
        xs, Ps, x_priors, P_priors, Ks = kf.batch_filter(measurements, controls)
        n = len(measurements)

        # Backward pass
        xs_s = np.zeros_like(xs)
        Ps_s = np.zeros_like(Ps)
        xs_s[-1] = xs[-1]
        Ps_s[-1] = Ps[-1]

        for k in range(n - 2, -1, -1):
            # Smoother gain
            C = Ps[k] @ kf.F.T @ inv(P_priors[k + 1])
            xs_s[k] = xs[k] + C @ (xs_s[k + 1] - x_priors[k + 1])
            Ps_s[k] = Ps[k] + C @ (Ps_s[k + 1] - P_priors[k + 1]) @ C.T

        return xs_s, Ps_s


# ---------------------------------------------------------------------------
# Interacting Multiple Model (composes C157 HMM)
# ---------------------------------------------------------------------------

class InteractingMultipleModel:
    """IMM estimator for switching/hybrid systems.
    Uses C157 HMM for mode transition probabilities.

    Each mode has its own KalmanFilter with different dynamics.
    Mode transitions follow an HMM with discrete states.
    """

    def __init__(self, filters, transition_matrix, mode_probs=None):
        """
        filters: list of KalmanFilter instances (one per mode)
        transition_matrix: NxN mode transition probability matrix
        mode_probs: initial mode probabilities (uniform if None)
        """
        self.filters = filters
        self.n_modes = len(filters)
        self.TPM = np.array(transition_matrix, dtype=float)

        if mode_probs is not None:
            self.mu = np.array(mode_probs, dtype=float)
        else:
            self.mu = np.ones(self.n_modes) / self.n_modes

        self.x = None  # combined state estimate
        self.P = None  # combined covariance

    def _mixing(self):
        """Compute mixing probabilities and mixed initial conditions."""
        n = self.n_modes
        dim_x = self.filters[0].dim_x

        # c_bar[j] = sum_i mu[i] * TPM[i,j]
        c_bar = self.mu @ self.TPM

        # Mixing probabilities: mu_{i|j} = mu[i] * TPM[i,j] / c_bar[j]
        mixing = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mixing[i, j] = self.mu[i] * self.TPM[i, j] / max(c_bar[j], 1e-300)

        # Mixed state estimates and covariances
        x_mixed = []
        P_mixed = []
        for j in range(n):
            x_j = np.zeros(dim_x)
            for i in range(n):
                x_j += mixing[i, j] * self.filters[i].x
            P_j = np.zeros((dim_x, dim_x))
            for i in range(n):
                dx = self.filters[i].x - x_j
                P_j += mixing[i, j] * (self.filters[i].P + np.outer(dx, dx))
            x_mixed.append(x_j)
            P_mixed.append(P_j)

        return c_bar, x_mixed, P_mixed

    def predict(self, u=None):
        """IMM predict: mix, then predict each filter."""
        c_bar, x_mixed, P_mixed = self._mixing()

        for j in range(self.n_modes):
            self.filters[j].x = x_mixed[j]
            self.filters[j].P = P_mixed[j]
            self.filters[j].predict(u)

        self._c_bar = c_bar

    def update(self, z):
        """IMM update: update each filter, then combine."""
        z = np.asarray(z, dtype=float)
        dim_x = self.filters[0].dim_x
        likelihoods = np.zeros(self.n_modes)

        for j in range(self.n_modes):
            self.filters[j].update(z)
            # Mode likelihood from innovation
            y = self.filters[j].y
            S = self.filters[j].S
            n_z = len(y)
            log_lik = -0.5 * (
                n_z * np.log(2 * np.pi) + np.log(max(det(S), 1e-300))
                + y @ inv(S) @ y
            )
            likelihoods[j] = np.exp(log_lik)

        # Update mode probabilities
        c = self._c_bar * likelihoods
        total = c.sum()
        if total > 0:
            self.mu = c / total
        else:
            self.mu = np.ones(self.n_modes) / self.n_modes

        # Combined estimate
        self.x = np.zeros(dim_x)
        for j in range(self.n_modes):
            self.x += self.mu[j] * self.filters[j].x

        self.P = np.zeros((dim_x, dim_x))
        for j in range(self.n_modes):
            dx = self.filters[j].x - self.x
            self.P += self.mu[j] * (self.filters[j].P + np.outer(dx, dx))

    def filter(self, measurements, controls=None):
        """Run IMM over sequence."""
        xs, Ps, mus = [], [], []
        for k, z in enumerate(measurements):
            u = controls[k] if controls is not None else None
            self.predict(u)
            self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
            mus.append(self.mu.copy())
        return np.array(xs), np.array(Ps), np.array(mus)


# ---------------------------------------------------------------------------
# Ensemble Kalman Filter
# ---------------------------------------------------------------------------

class EnsembleKalmanFilter:
    """Ensemble Kalman Filter for high-dimensional systems.
    Approximates covariance via ensemble of state samples.
    """

    def __init__(self, dim_x, dim_z, n_ensemble, f, h, seed=None):
        """
        f: state transition f(x) -> x_next  (no control for simplicity)
        h: observation h(x) -> z
        n_ensemble: ensemble size
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.n_ensemble = n_ensemble
        self.f = f
        self.h = h

        self.rng = np.random.RandomState(seed)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

        # Initialize ensemble
        self.ensemble = self.rng.randn(n_ensemble, dim_x)

    @property
    def x(self):
        return self.ensemble.mean(axis=0)

    @property
    def P(self):
        A = self.ensemble - self.x
        return (A.T @ A) / (self.n_ensemble - 1)

    def initialize(self, x0, P0):
        """Initialize ensemble around x0 with covariance P0."""
        L = cholesky(P0)
        self.ensemble = x0 + (self.rng.randn(self.n_ensemble, self.dim_x) @ L.T)

    def predict(self):
        """Propagate each ensemble member through f + noise."""
        noise = self.rng.multivariate_normal(
            np.zeros(self.dim_x), self.Q, self.n_ensemble
        )
        for i in range(self.n_ensemble):
            self.ensemble[i] = self.f(self.ensemble[i]) + noise[i]

    def update(self, z):
        """Perturbed observation EnKF update."""
        z = np.asarray(z, dtype=float)

        # Predicted observations
        H_ens = np.array([self.h(e) for e in self.ensemble])
        H_mean = H_ens.mean(axis=0)

        # Anomalies
        X_a = self.ensemble - self.x  # state anomalies
        H_a = H_ens - H_mean           # obs anomalies

        # Covariances
        P_HH = (H_a.T @ H_a) / (self.n_ensemble - 1)
        P_XH = (X_a.T @ H_a) / (self.n_ensemble - 1)

        # Kalman gain
        K = P_XH @ inv(P_HH + self.R)

        # Perturbed observations
        perturbed_z = z + self.rng.multivariate_normal(
            np.zeros(self.dim_z), self.R, self.n_ensemble
        )

        # Update each member
        for i in range(self.n_ensemble):
            self.ensemble[i] += K @ (perturbed_z[i] - H_ens[i])

    def filter(self, measurements):
        xs, Ps = [], []
        for z in measurements:
            self.predict()
            self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
        return np.array(xs), np.array(Ps)


# ---------------------------------------------------------------------------
# Kalman Utilities
# ---------------------------------------------------------------------------

class KalmanUtils:
    """Utility functions for analysis and diagnostics."""

    @staticmethod
    def observability_matrix(F, H, n=None):
        """Compute observability matrix O = [H; HF; HF^2; ... HF^{n-1}]."""
        if n is None:
            n = F.shape[0]
        rows = []
        HFk = H.copy()
        for k in range(n):
            rows.append(HFk)
            HFk = HFk @ F
        return np.vstack(rows)

    @staticmethod
    def is_observable(F, H, tol=1e-10):
        """Check if system is observable (rank of O == dim_x)."""
        O = KalmanUtils.observability_matrix(F, H)
        return np.linalg.matrix_rank(O, tol=tol) == F.shape[0]

    @staticmethod
    def controllability_matrix(F, B, n=None):
        """Compute controllability matrix C = [B, FB, F^2B, ..., F^{n-1}B]."""
        if n is None:
            n = F.shape[0]
        cols = []
        FkB = B.copy()
        for k in range(n):
            cols.append(FkB)
            FkB = F @ FkB
        return np.hstack(cols)

    @staticmethod
    def is_controllable(F, B, tol=1e-10):
        """Check if system is controllable."""
        C = KalmanUtils.controllability_matrix(F, B)
        return np.linalg.matrix_rank(C, tol=tol) == F.shape[0]

    @staticmethod
    def steady_state_gain(F, H, Q, R, max_iter=1000, tol=1e-12):
        """Compute steady-state Kalman gain by iterating Riccati equation."""
        dim_x = F.shape[0]
        P = np.eye(dim_x)

        for _ in range(max_iter):
            P_pred = F @ P @ F.T + Q
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ inv(S)
            P_new = (np.eye(dim_x) - K @ H) @ P_pred
            if np.max(np.abs(P_new - P)) < tol:
                return K, P_new
            P = P_new

        return K, P

    @staticmethod
    def nis(y, S):
        """Normalized Innovation Squared: y^T S^{-1} y.
        Should be chi-squared distributed with dim_z degrees of freedom.
        """
        return y @ inv(S) @ y

    @staticmethod
    def nees(x_true, x_est, P):
        """Normalized Estimation Error Squared: e^T P^{-1} e.
        Should be chi-squared with dim_x degrees of freedom.
        """
        e = x_true - x_est
        return e @ inv(P) @ e

    @staticmethod
    def mahalanobis(x, mean, cov):
        """Mahalanobis distance."""
        d = x - mean
        return np.sqrt(d @ inv(cov) @ d)

    @staticmethod
    def log_likelihood_sequence(innovations, S_list):
        """Total log-likelihood from innovation sequence."""
        total = 0.0
        for y, S in zip(innovations, S_list):
            n = len(y)
            total += -0.5 * (
                n * np.log(2 * np.pi) + np.log(max(det(S), 1e-300))
                + y @ inv(S) @ y
            )
        return total

    @staticmethod
    def discrete_are(F, H, Q, R, max_iter=1000, tol=1e-12):
        """Solve discrete algebraic Riccati equation by iteration.
        Returns steady-state posterior covariance P.
        Same fixed point as steady_state_gain.
        """
        dim_x = F.shape[0]
        P = np.eye(dim_x)
        for _ in range(max_iter):
            P_pred = F @ P @ F.T + Q
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ inv(S)
            P_new = (np.eye(dim_x) - K @ H) @ P_pred
            if np.max(np.abs(P_new - P)) < tol:
                return P_new
            P = P_new
        return P
