"""
C198: Advanced Time Series Forecasting
Components: GARCH, VAR, Kalman Filter/State Space, Prophet-style decomposition,
            Theta method, Croston's method, ensemble forecasting, backtesting,
            forecast combination, prediction intervals.

Beyond C188's basic ARIMA/ExpSmoothing -- this is the full forecasting toolkit.
"""

import math
import numpy as np
from scipy import linalg as scipy_linalg
from scipy.optimize import minimize
from scipy.stats import norm


# ============================================================
# Utility functions
# ============================================================

def _validate_series(y):
    """Convert to numpy array and validate."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("Series must be 1-dimensional")
    if len(y) < 2:
        raise ValueError("Series must have at least 2 observations")
    return y


def _validate_matrix(Y):
    """Convert to 2D numpy array."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    return Y


def difference(y, d=1):
    """Difference a series d times."""
    y = np.asarray(y, dtype=float)
    for _ in range(d):
        y = np.diff(y)
    return y


def undifference(diffed, original, d=1):
    """Reverse differencing using original series anchors."""
    result = diffed.copy()
    for i in range(d - 1, -1, -1):
        anchors = np.asarray(original[:i + 1], dtype=float)
        new = np.empty(len(result) + 1)
        new[0] = anchors[i]
        for j in range(len(result)):
            new[j + 1] = new[j] + result[j]
        result = new[1:]  # drop the anchor
    return result


# ============================================================
# GARCH(p,q) - Volatility Modeling
# ============================================================

class GARCH:
    """
    Generalized Autoregressive Conditional Heteroskedasticity.
    Models time-varying volatility: sigma_t^2 = omega + sum(alpha_i * e_{t-i}^2) + sum(beta_j * sigma_{t-j}^2)
    """

    def __init__(self, p=1, q=1):
        self.p = p  # GARCH terms (lagged variances)
        self.q = q  # ARCH terms (lagged squared residuals)
        self.omega = None
        self.alpha = None  # ARCH coefficients
        self.beta = None   # GARCH coefficients
        self.mu = None
        self._residuals = None
        self._sigma2 = None
        self._fitted = False

    def fit(self, y, max_iter=200):
        """Fit GARCH(p,q) via maximum likelihood."""
        y = _validate_series(y)
        n = len(y)
        self.mu = np.mean(y)
        eps = y - self.mu  # residuals from mean

        # Initial params: omega, alpha_1..q, beta_1..p
        n_params = 1 + self.q + self.p
        x0 = np.zeros(n_params)
        var_eps = np.var(eps)
        x0[0] = var_eps * 0.1  # omega
        for i in range(self.q):
            x0[1 + i] = 0.05
        for i in range(self.p):
            x0[1 + self.q + i] = 0.85 / self.p

        def neg_log_likelihood(params):
            omega = params[0]
            alpha = params[1:1 + self.q]
            beta = params[1 + self.q:]

            if omega <= 1e-10 or np.any(alpha < 0) or np.any(beta < 0):
                return 1e10
            if np.sum(alpha) + np.sum(beta) >= 1.0:
                return 1e10

            sigma2 = np.full(n, var_eps)
            for t in range(max(self.p, self.q), n):
                s = omega
                for i in range(self.q):
                    s += alpha[i] * eps[t - 1 - i] ** 2
                for j in range(self.p):
                    s += beta[j] * sigma2[t - 1 - j]
                sigma2[t] = max(s, 1e-10)

            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps ** 2 / sigma2)
            return -ll

        bounds = [(1e-10, None)] + [(0, 0.999)] * (self.q + self.p)
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter})

        params = result.x
        self.omega = params[0]
        self.alpha = params[1:1 + self.q]
        self.beta = params[1 + self.q:]

        # Compute fitted sigma2
        sigma2 = np.full(n, var_eps)
        for t in range(max(self.p, self.q), n):
            s = self.omega
            for i in range(self.q):
                s += self.alpha[i] * eps[t - 1 - i] ** 2
            for j in range(self.p):
                s += self.beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(s, 1e-10)

        self._residuals = eps
        self._sigma2 = sigma2
        self._fitted = True
        return self

    def forecast_volatility(self, steps=1):
        """Forecast conditional variance steps ahead."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        forecasts = np.zeros(steps)
        eps = self._residuals
        sigma2 = self._sigma2

        # Build initial state
        last_eps2 = [eps[-(i + 1)] ** 2 for i in range(self.q)]
        last_sigma2 = [sigma2[-(i + 1)] for i in range(self.p)]

        for h in range(steps):
            s = self.omega
            for i in range(self.q):
                if h <= i:
                    s += self.alpha[i] * last_eps2[i - h] if (i - h) < len(last_eps2) else 0
                else:
                    # E[eps^2] = sigma^2
                    s += self.alpha[i] * forecasts[h - 1 - i] if (h - 1 - i) >= 0 else 0
            for j in range(self.p):
                if h <= j:
                    s += self.beta[j] * last_sigma2[j - h] if (j - h) < len(last_sigma2) else 0
                else:
                    s += self.beta[j] * forecasts[h - 1 - j] if (h - 1 - j) >= 0 else 0
            forecasts[h] = max(s, 1e-10)

        return forecasts

    @property
    def persistence(self):
        """Sum of alpha + beta. Close to 1 means high persistence."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return float(np.sum(self.alpha) + np.sum(self.beta))

    @property
    def unconditional_variance(self):
        """Long-run variance: omega / (1 - alpha - beta)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        denom = 1.0 - self.persistence
        if denom <= 0:
            return float('inf')
        return self.omega / denom

    @property
    def conditional_volatility(self):
        """Fitted conditional standard deviations."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return np.sqrt(self._sigma2)


# ============================================================
# VAR(p) - Vector Autoregression
# ============================================================

class VAR:
    """
    Vector Autoregression of order p.
    Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + e_t
    """

    def __init__(self, p=1):
        self.p = p
        self.intercept = None  # (k,) vector
        self.coefs = None      # list of p (k,k) matrices
        self.sigma_u = None    # (k,k) residual covariance
        self._Y = None
        self._k = None
        self._fitted = False

    def fit(self, Y):
        """Fit VAR(p) via OLS."""
        Y = _validate_matrix(Y)
        n, k = Y.shape
        self._k = k

        if n <= self.p:
            raise ValueError(f"Need more than {self.p} observations")

        # Build design matrix
        # For each t = p..n-1: y_t = [Y[t]], X_t = [1, Y[t-1], ..., Y[t-p]]
        T = n - self.p
        Ymat = Y[self.p:]  # (T, k)
        Xmat = np.ones((T, 1 + self.p * k))
        for t in range(T):
            for lag in range(self.p):
                start = 1 + lag * k
                Xmat[t, start:start + k] = Y[self.p + t - 1 - lag]

        # OLS: B = (X'X)^{-1} X'Y
        XtX = Xmat.T @ Xmat
        XtY = Xmat.T @ Ymat
        B = np.linalg.solve(XtX, XtY)  # (1+p*k, k)

        self.intercept = B[0]
        self.coefs = []
        for lag in range(self.p):
            start = 1 + lag * k
            self.coefs.append(B[start:start + k].T)  # (k, k)

        # Residual covariance
        residuals = Ymat - Xmat @ B
        self.sigma_u = (residuals.T @ residuals) / T
        self._Y = Y
        self._fitted = True
        return self

    def forecast(self, steps=1, Y_history=None):
        """Forecast steps ahead."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        Y = Y_history if Y_history is not None else self._Y
        Y = _validate_matrix(Y)
        k = self._k
        forecasts = np.zeros((steps, k))

        # Extend with forecasts
        extended = list(Y[-self.p:])

        for h in range(steps):
            pred = self.intercept.copy()
            for lag in range(self.p):
                idx = len(extended) - 1 - lag
                pred = pred + self.coefs[lag] @ extended[idx]
            forecasts[h] = pred
            extended.append(pred)

        return forecasts

    def impulse_response(self, steps=10, shock_var=0):
        """Orthogonalized impulse response function."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        k = self._k
        # Cholesky decomposition for orthogonalization
        P = np.linalg.cholesky(self.sigma_u)

        # MA representation
        Phi = [np.eye(k)]  # Phi_0 = I
        for s in range(1, steps + 1):
            phi_s = np.zeros((k, k))
            for j in range(min(s, self.p)):
                phi_s += self.coefs[j] @ Phi[s - 1 - j]
            Phi.append(phi_s)

        # Orthogonalized IRF: Theta_s = Phi_s @ P
        irf = np.zeros((steps + 1, k))
        for s in range(steps + 1):
            irf[s] = (Phi[s] @ P)[:, shock_var]

        return irf

    def granger_causality(self, cause_var, effect_var):
        """
        Test if cause_var Granger-causes effect_var.
        Returns F-statistic and approximate p-value.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        Y = self._Y
        n, k = Y.shape
        T = n - self.p

        # Unrestricted model residuals (already fitted)
        Ymat = Y[self.p:]
        Xmat = np.ones((T, 1 + self.p * k))
        for t in range(T):
            for lag in range(self.p):
                start = 1 + lag * k
                Xmat[t, start:start + k] = Y[self.p + t - 1 - lag]

        B_full = np.linalg.solve(Xmat.T @ Xmat, Xmat.T @ Ymat)
        resid_full = Ymat[:, effect_var] - Xmat @ B_full[:, effect_var]
        rss_full = np.sum(resid_full ** 2)

        # Restricted model: remove cause_var lags
        cols_to_keep = [0]  # intercept
        for lag in range(self.p):
            for v in range(k):
                col = 1 + lag * k + v
                if v != cause_var:
                    cols_to_keep.append(col)

        Xr = Xmat[:, cols_to_keep]
        B_r = np.linalg.solve(Xr.T @ Xr, Xr.T @ Ymat[:, effect_var])
        resid_r = Ymat[:, effect_var] - Xr @ B_r
        rss_restricted = np.sum(resid_r ** 2)

        # F-test
        q = self.p  # number of restrictions
        df2 = T - 1 - self.p * k
        if df2 <= 0:
            return 0.0, 1.0
        F = ((rss_restricted - rss_full) / q) / (rss_full / df2)

        # Approximate p-value using F-distribution
        # Using beta regularized incomplete function approximation
        p_value = self._f_pvalue(F, q, df2)
        return float(F), float(p_value)

    @staticmethod
    def _f_pvalue(F, d1, d2):
        """Approximate p-value for F distribution."""
        if F <= 0:
            return 1.0
        x = d2 / (d2 + d1 * F)
        # Use scipy for accuracy if available
        from scipy.stats import f as f_dist
        return float(1.0 - f_dist.cdf(F, d1, d2))

    def forecast_error_variance_decomposition(self, steps=10):
        """FEVD: proportion of forecast error variance attributed to each shock."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        k = self._k
        P = np.linalg.cholesky(self.sigma_u)

        # MA coefficients
        Phi = [np.eye(k)]
        for s in range(1, steps + 1):
            phi_s = np.zeros((k, k))
            for j in range(min(s, self.p)):
                phi_s += self.coefs[j] @ Phi[s - 1 - j]
            Phi.append(phi_s)

        # Orthogonalized MA coefficients
        Theta = [phi @ P for phi in Phi]

        # FEVD
        fevd = np.zeros((steps + 1, k, k))
        cumulative = np.zeros((k, k))
        for s in range(steps + 1):
            cumulative += Theta[s] ** 2
            row_sums = cumulative.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            fevd[s] = cumulative / row_sums

        return fevd


# ============================================================
# Kalman Filter / State Space Model
# ============================================================

class KalmanFilter:
    """
    Linear Gaussian State Space Model:
      x_t = F x_{t-1} + B u_t + w_t,  w_t ~ N(0, Q)
      z_t = H x_t + v_t,              v_t ~ N(0, R)
    """

    def __init__(self, state_dim, obs_dim, control_dim=0):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.control_dim = control_dim

        # System matrices
        self.F = np.eye(state_dim)        # State transition
        self.H = np.eye(obs_dim, state_dim)  # Observation
        self.Q = np.eye(state_dim) * 0.1  # Process noise
        self.R = np.eye(obs_dim) * 1.0    # Measurement noise
        self.B = np.zeros((state_dim, max(control_dim, 1)))  # Control

        # State
        self.x = np.zeros(state_dim)      # State estimate
        self.P = np.eye(state_dim)        # Estimate covariance

        self._log_likelihood = 0.0
        self._filtered_states = []
        self._filtered_covs = []
        self._predictions = []

    def predict(self, u=None):
        """Prediction step."""
        x_pred = self.F @ self.x
        if u is not None:
            x_pred += self.B @ np.asarray(u)
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, z, x_pred=None, P_pred=None):
        """Update step with observation z."""
        if x_pred is None:
            x_pred, P_pred = self.predict()

        z = np.asarray(z, dtype=float).ravel()

        # Innovation
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        # Log-likelihood contribution
        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (self.obs_dim * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(S, y))
        self._log_likelihood += ll

        return self.x.copy(), self.P.copy()

    def filter(self, observations, controls=None):
        """Run filter over sequence of observations."""
        observations = np.asarray(observations, dtype=float)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        n = len(observations)
        self._log_likelihood = 0.0
        self._filtered_states = []
        self._filtered_covs = []
        self._predictions = []

        for t in range(n):
            u = controls[t] if controls is not None else None
            x_pred, P_pred = self.predict(u)
            self._predictions.append(self.H @ x_pred)
            x_filt, P_filt = self.update(observations[t], x_pred, P_pred)
            self._filtered_states.append(x_filt.copy())
            self._filtered_covs.append(P_filt.copy())

        return np.array(self._filtered_states)

    def smooth(self, observations, controls=None):
        """Rauch-Tung-Striebel smoother."""
        filtered = self.filter(observations, controls)
        n = len(filtered)

        smoothed_states = [None] * n
        smoothed_covs = [None] * n
        smoothed_states[-1] = self._filtered_states[-1].copy()
        smoothed_covs[-1] = self._filtered_covs[-1].copy()

        for t in range(n - 2, -1, -1):
            x_filt = self._filtered_states[t]
            P_filt = self._filtered_covs[t]
            P_pred = self.F @ P_filt @ self.F.T + self.Q

            # Smoother gain
            L = P_filt @ self.F.T @ np.linalg.inv(P_pred)

            smoothed_states[t] = x_filt + L @ (smoothed_states[t + 1] - self.F @ x_filt)
            smoothed_covs[t] = P_filt + L @ (smoothed_covs[t + 1] - P_pred) @ L.T

        return np.array(smoothed_states)

    def forecast(self, steps=1):
        """Forecast steps ahead from current state."""
        forecasts = np.zeros((steps, self.obs_dim))
        x = self.x.copy()
        P = self.P.copy()
        intervals = []

        for h in range(steps):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            z_pred = self.H @ x
            z_var = self.H @ P @ self.H.T + self.R
            forecasts[h] = z_pred
            std = np.sqrt(np.diag(z_var))
            intervals.append((z_pred - 1.96 * std, z_pred + 1.96 * std))

        return forecasts, intervals

    @property
    def log_likelihood(self):
        return self._log_likelihood


# ============================================================
# Local Level Model (structural time series via Kalman)
# ============================================================

class LocalLevelModel:
    """
    Local level (random walk + noise) model:
      level_t = level_{t-1} + eta_t,  eta ~ N(0, sigma_level^2)
      y_t = level_t + eps_t,          eps ~ N(0, sigma_obs^2)
    """

    def __init__(self):
        self.sigma_level = 1.0
        self.sigma_obs = 1.0
        self._kf = None
        self._fitted = False

    def fit(self, y):
        """Fit via MLE using Kalman filter."""
        y = _validate_series(y)

        def neg_ll(params):
            sl, so = params
            if sl <= 0 or so <= 0:
                return 1e10
            kf = KalmanFilter(state_dim=1, obs_dim=1)
            kf.F = np.array([[1.0]])
            kf.H = np.array([[1.0]])
            kf.Q = np.array([[sl ** 2]])
            kf.R = np.array([[so ** 2]])
            kf.x = np.array([y[0]])
            kf.P = np.array([[so ** 2 + sl ** 2]])
            kf.filter(y.reshape(-1, 1))
            return -kf.log_likelihood

        result = minimize(neg_ll, [np.std(y) * 0.1, np.std(y)],
                          method='Nelder-Mead', options={'maxiter': 500})
        self.sigma_level, self.sigma_obs = abs(result.x[0]), abs(result.x[1])

        self._kf = KalmanFilter(state_dim=1, obs_dim=1)
        self._kf.F = np.array([[1.0]])
        self._kf.H = np.array([[1.0]])
        self._kf.Q = np.array([[self.sigma_level ** 2]])
        self._kf.R = np.array([[self.sigma_obs ** 2]])
        self._kf.x = np.array([y[0]])
        self._kf.P = np.array([[self.sigma_obs ** 2 + self.sigma_level ** 2]])
        self._kf.filter(y.reshape(-1, 1))
        self._fitted = True
        return self

    def forecast(self, steps=1):
        """Forecast with prediction intervals."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        fc, intervals = self._kf.forecast(steps)
        return fc.ravel(), [(lo[0], hi[0]) for lo, hi in intervals]

    def smoothed_level(self, y):
        """Return smoothed level estimates."""
        y = _validate_series(y)
        self.fit(y)
        smoothed = self._kf.smooth(y.reshape(-1, 1))
        return smoothed.ravel()


# ============================================================
# Local Linear Trend Model
# ============================================================

class LocalLinearTrend:
    """
    Local linear trend model:
      level_t = level_{t-1} + trend_{t-1} + eta_level
      trend_t = trend_{t-1} + eta_trend
      y_t = level_t + eps_t
    """

    def __init__(self):
        self.sigma_level = 1.0
        self.sigma_trend = 0.1
        self.sigma_obs = 1.0
        self._kf = None
        self._fitted = False

    def _build_kf(self, y, sl, st, so):
        kf = KalmanFilter(state_dim=2, obs_dim=1)
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.Q = np.diag([sl ** 2, st ** 2])
        kf.R = np.array([[so ** 2]])
        kf.x = np.array([y[0], 0.0])
        kf.P = np.eye(2) * 10.0
        return kf

    def fit(self, y):
        y = _validate_series(y)

        def neg_ll(params):
            sl, st, so = params
            if sl <= 0 or st <= 0 or so <= 0:
                return 1e10
            kf = self._build_kf(y, sl, st, so)
            kf.filter(y.reshape(-1, 1))
            return -kf.log_likelihood

        std_y = max(np.std(y), 1e-6)
        result = minimize(neg_ll, [std_y * 0.1, std_y * 0.01, std_y],
                          method='Nelder-Mead', options={'maxiter': 500})
        self.sigma_level = abs(result.x[0])
        self.sigma_trend = abs(result.x[1])
        self.sigma_obs = abs(result.x[2])

        self._kf = self._build_kf(y, self.sigma_level, self.sigma_trend, self.sigma_obs)
        self._kf.filter(y.reshape(-1, 1))
        self._fitted = True
        return self

    def forecast(self, steps=1):
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        fc, intervals = self._kf.forecast(steps)
        return fc.ravel(), [(lo[0], hi[0]) for lo, hi in intervals]


# ============================================================
# Prophet-Style Decomposition
# ============================================================

class ProphetDecomposition:
    """
    Additive decomposition with piecewise linear trend + Fourier seasonality.
    y = trend + seasonality + residual

    Simplified Prophet: fits via linear regression with changepoints.
    """

    def __init__(self, n_changepoints=10, seasonality_period=None,
                 fourier_order=3):
        self.n_changepoints = n_changepoints
        self.seasonality_period = seasonality_period
        self.fourier_order = fourier_order
        self._trend_coefs = None
        self._season_coefs = None
        self._changepoints = None
        self._y = None
        self._fitted = False

    def fit(self, y):
        y = _validate_series(y)
        self._y = y
        n = len(y)
        t = np.arange(n, dtype=float) / max(n - 1, 1)  # normalized [0, 1]

        # Changepoint indicators
        if self.n_changepoints > 0 and n > self.n_changepoints + 2:
            cp_indices = np.linspace(0, int(0.8 * n), self.n_changepoints + 2, dtype=int)[1:-1]
            self._changepoints = cp_indices
            s_t = t[cp_indices]
        else:
            cp_indices = np.array([], dtype=int)
            self._changepoints = cp_indices
            s_t = np.array([])

        # Build design matrix
        # Trend: intercept + slope + changepoint adjustments
        n_cp = len(cp_indices)
        A = np.zeros((n, n_cp))
        for j, cp in enumerate(cp_indices):
            A[cp:, j] = t[cp:] - t[cp]

        # Seasonality: Fourier terms
        season_cols = []
        if self.seasonality_period is not None and self.seasonality_period > 0:
            for k in range(1, self.fourier_order + 1):
                season_cols.append(np.sin(2 * np.pi * k * np.arange(n) / self.seasonality_period))
                season_cols.append(np.cos(2 * np.pi * k * np.arange(n) / self.seasonality_period))

        n_season = len(season_cols)
        X = np.column_stack([np.ones(n), t] +
                            [A[:, j] for j in range(n_cp)] +
                            season_cols) if (n_cp > 0 or n_season > 0) else np.column_stack([np.ones(n), t])

        # Ridge regression for stability
        lam = 0.1
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        XtY = X.T @ y
        coefs = np.linalg.solve(XtX, XtY)

        self._trend_coefs = coefs[:2 + n_cp]
        self._season_coefs = coefs[2 + n_cp:]
        self._X = X
        self._coefs = coefs
        self._t = t
        self._fitted = True
        return self

    @property
    def trend(self):
        if not self._fitted:
            raise RuntimeError("Not fitted")
        n = len(self._y)
        n_cp = len(self._changepoints)
        X_trend = self._X[:, :2 + n_cp]
        return X_trend @ self._trend_coefs

    @property
    def seasonality(self):
        if not self._fitted:
            raise RuntimeError("Not fitted")
        if len(self._season_coefs) == 0:
            return np.zeros(len(self._y))
        n_cp = len(self._changepoints)
        X_season = self._X[:, 2 + n_cp:]
        return X_season @ self._season_coefs

    @property
    def residual(self):
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return self._y - self.trend - self.seasonality

    def forecast(self, steps=1):
        """Forecast by extrapolating trend + seasonality."""
        if not self._fitted:
            raise RuntimeError("Not fitted")

        n = len(self._y)
        n_cp = len(self._changepoints)
        t_orig = self._t

        # Future time points
        dt = 1.0 / max(n - 1, 1)
        t_future = np.array([t_orig[-1] + (i + 1) * dt for i in range(steps)])

        # Trend extrapolation
        trend_pred = self._trend_coefs[0] + self._trend_coefs[1] * t_future
        for j in range(n_cp):
            cp_t = t_orig[self._changepoints[j]]
            delta = self._trend_coefs[2 + j]
            mask = t_future > cp_t
            trend_pred[mask] += delta * (t_future[mask] - cp_t)

        # Seasonality extrapolation
        season_pred = np.zeros(steps)
        if self.seasonality_period is not None and len(self._season_coefs) > 0:
            idx = 0
            for k in range(1, self.fourier_order + 1):
                future_idx = np.arange(n, n + steps)
                sin_val = np.sin(2 * np.pi * k * future_idx / self.seasonality_period)
                cos_val = np.cos(2 * np.pi * k * future_idx / self.seasonality_period)
                season_pred += self._season_coefs[idx] * sin_val
                season_pred += self._season_coefs[idx + 1] * cos_val
                idx += 2

        return trend_pred + season_pred


# ============================================================
# Theta Method
# ============================================================

class ThetaMethod:
    """
    Theta method: decomposes into theta-lines and recombines.
    Excellent for M3 competition data. Simple yet effective.
    """

    def __init__(self, theta=2.0):
        self.theta = theta
        self._y = None
        self._drift = None
        self._ses_level = None
        self._fitted = False

    def fit(self, y):
        y = _validate_series(y)
        self._y = y
        n = len(y)

        # Drift from linear regression
        t = np.arange(n, dtype=float)
        t_mean = t.mean()
        y_mean = y.mean()
        self._drift = np.sum((t - t_mean) * (y - y_mean)) / max(np.sum((t - t_mean) ** 2), 1e-10)

        # SES on theta-line 2 (amplified second differences)
        # theta_line_2 = theta * y - (theta - 1) * linear_trend
        linear = y_mean + self._drift * (t - t_mean)
        theta_2 = self.theta * y - (self.theta - 1) * linear

        # Simple exponential smoothing
        alpha = 0.5  # could optimize but 0.5 works well for theta
        level = theta_2[0]
        for i in range(1, n):
            level = alpha * theta_2[i] + (1 - alpha) * level
        self._ses_level = level
        self._alpha = alpha
        self._fitted = True
        return self

    def forecast(self, steps=1):
        if not self._fitted:
            raise RuntimeError("Not fitted")

        n = len(self._y)
        forecasts = np.zeros(steps)
        for h in range(steps):
            # Theta combination: average of theta-line 1 (linear) and SES forecast
            ses_fc = self._ses_level  # SES is flat
            linear_fc = self._y.mean() + self._drift * (n + h - (n - 1) / 2)
            # Standard theta: (theta_line_0 + theta_line_2) / 2
            forecasts[h] = (linear_fc + ses_fc) / 2 + self._drift * (h + 1) / 2

        return forecasts


# ============================================================
# Croston's Method (Intermittent Demand)
# ============================================================

class CrostonMethod:
    """
    Croston's method for intermittent demand forecasting.
    Separately smooths demand sizes and inter-arrival intervals.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self._z = None  # smoothed demand size
        self._p = None  # smoothed inter-demand interval
        self._fitted = False

    def fit(self, y):
        y = _validate_series(y)

        # Initialize
        non_zero = y[y > 0]
        if len(non_zero) == 0:
            self._z = 0.0
            self._p = float('inf')
            self._fitted = True
            return self

        z = non_zero[0]  # demand size
        # Find first inter-demand interval
        first_nz = np.where(y > 0)[0]
        if len(first_nz) > 1:
            p = float(first_nz[1] - first_nz[0])
        else:
            p = float(len(y))

        q = 0  # periods since last demand
        for t in range(len(y)):
            q += 1
            if y[t] > 0:
                z = self.alpha * y[t] + (1 - self.alpha) * z
                p = self.alpha * q + (1 - self.alpha) * p
                q = 0

        self._z = z
        self._p = max(p, 1e-10)
        self._fitted = True
        return self

    def forecast(self, steps=1):
        """Forecast demand rate per period."""
        if not self._fitted:
            raise RuntimeError("Not fitted")
        rate = self._z / self._p
        return np.full(steps, rate)

    @property
    def demand_size(self):
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return self._z

    @property
    def demand_interval(self):
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return self._p


# ============================================================
# Exponential Smoothing (Holt-Winters) - Enhanced
# ============================================================

class HoltWinters:
    """
    Triple exponential smoothing (Holt-Winters).
    Supports additive and multiplicative seasonality.
    """

    def __init__(self, seasonal_period=None, seasonal='additive'):
        self.seasonal_period = seasonal_period
        self.seasonal = seasonal  # 'additive' or 'multiplicative'
        self.alpha = 0.3  # level
        self.beta = 0.1   # trend
        self.gamma = 0.1  # seasonal
        self._level = None
        self._trend = None
        self._seasons = None
        self._y = None
        self._fitted = False

    def fit(self, y, optimize=True):
        y = _validate_series(y)
        self._y = y
        n = len(y)
        m = self.seasonal_period

        if optimize:
            best_params = self._optimize(y)
            self.alpha, self.beta, self.gamma = best_params

        # Initialize
        if m is not None and m > 0 and n >= 2 * m:
            # Level: average of first season
            self._level = np.mean(y[:m])
            # Trend: average slope over first two seasons
            self._trend = np.mean(y[m:2 * m] - y[:m]) / m
            # Seasonal
            if self.seasonal == 'additive':
                self._seasons = list(y[:m] - self._level)
            else:
                self._seasons = list(y[:m] / max(self._level, 1e-10))
        else:
            self._level = y[0]
            self._trend = y[1] - y[0] if n > 1 else 0.0
            self._seasons = [0.0] if self.seasonal == 'additive' else [1.0]
            m = 1

        # Fit
        levels = [self._level]
        trends = [self._trend]
        fitted = np.zeros(n)

        for t in range(n):
            s_idx = t % m
            if self.seasonal == 'additive':
                fitted[t] = self._level + self._trend + self._seasons[s_idx]
                new_level = self.alpha * (y[t] - self._seasons[s_idx]) + (1 - self.alpha) * (self._level + self._trend)
                new_trend = self.beta * (new_level - self._level) + (1 - self.beta) * self._trend
                self._seasons[s_idx] = self.gamma * (y[t] - new_level) + (1 - self.gamma) * self._seasons[s_idx]
            else:
                s = max(self._seasons[s_idx], 1e-10)
                fitted[t] = (self._level + self._trend) * s
                new_level = self.alpha * (y[t] / s) + (1 - self.alpha) * (self._level + self._trend)
                new_trend = self.beta * (new_level - self._level) + (1 - self.beta) * self._trend
                self._seasons[s_idx] = self.gamma * (y[t] / max(new_level, 1e-10)) + (1 - self.gamma) * s

            self._level = new_level
            self._trend = new_trend
            levels.append(new_level)
            trends.append(new_trend)

        self._fitted_values = fitted
        self._fitted = True
        return self

    def _optimize(self, y):
        """Optimize smoothing parameters via grid search."""
        n = len(y)
        m = self.seasonal_period or 1
        best_sse = float('inf')
        best_params = (0.3, 0.1, 0.1)

        for a in np.arange(0.1, 0.9, 0.2):
            for b in np.arange(0.01, 0.5, 0.15):
                for g in np.arange(0.01, 0.5, 0.15):
                    hw = HoltWinters(self.seasonal_period, self.seasonal)
                    hw.alpha, hw.beta, hw.gamma = a, b, g
                    hw.fit(y, optimize=False)
                    sse = np.sum((y - hw._fitted_values) ** 2)
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (a, b, g)

        return best_params

    def forecast(self, steps=1):
        if not self._fitted:
            raise RuntimeError("Not fitted")

        m = self.seasonal_period or 1
        forecasts = np.zeros(steps)
        for h in range(steps):
            s_idx = (len(self._y) + h) % m
            if self.seasonal == 'additive':
                forecasts[h] = self._level + (h + 1) * self._trend + self._seasons[s_idx]
            else:
                forecasts[h] = (self._level + (h + 1) * self._trend) * self._seasons[s_idx]
        return forecasts


# ============================================================
# Ensemble Forecaster
# ============================================================

class EnsembleForecaster:
    """
    Combines multiple forecasting models.
    Supports: equal weights, inverse-error weights, and stacking.
    """

    def __init__(self, models=None, method='inverse_error'):
        """
        models: list of (name, model_instance) tuples
        method: 'equal', 'inverse_error', or 'stacking'
        """
        self.models = models or []
        self.method = method
        self.weights = None
        self._fitted = False

    def fit(self, y, validation_size=None):
        """
        Fit all models and compute weights.
        validation_size: number of observations for validation (for weight estimation).
        """
        y = _validate_series(y)

        if validation_size is None:
            validation_size = max(int(len(y) * 0.2), 1)

        train = y[:-validation_size]
        val = y[-validation_size:]

        # Fit and evaluate each model
        errors = []
        fitted_models = []
        for name, model in self.models:
            try:
                model.fit(train)
                fc = model.forecast(validation_size)
                if isinstance(fc, tuple):
                    fc = fc[0]  # some models return (forecast, intervals)
                fc = np.asarray(fc).ravel()[:len(val)]
                mse = np.mean((val[:len(fc)] - fc) ** 2)
                errors.append(max(mse, 1e-10))
                fitted_models.append((name, model))
            except Exception:
                errors.append(float('inf'))
                fitted_models.append((name, model))

        # Compute weights
        if self.method == 'equal':
            n_valid = sum(1 for e in errors if e < float('inf'))
            self.weights = [1.0 / max(n_valid, 1) if e < float('inf') else 0.0 for e in errors]
        elif self.method == 'inverse_error':
            inv_errors = [1.0 / e if e < float('inf') else 0.0 for e in errors]
            total = sum(inv_errors)
            self.weights = [w / max(total, 1e-10) for w in inv_errors]
        else:  # stacking
            self.weights = self._stack_weights(train, val, fitted_models)

        # Refit on full data
        for name, model in self.models:
            try:
                model.fit(y)
            except Exception:
                pass

        self._fitted = True
        return self

    def _stack_weights(self, train, val, models):
        """Learn stacking weights via constrained regression."""
        n_models = len(models)
        n_val = len(val)

        # Get validation predictions
        preds = np.zeros((n_val, n_models))
        for i, (name, model) in enumerate(models):
            try:
                model.fit(train)
                fc = model.forecast(n_val)
                if isinstance(fc, tuple):
                    fc = fc[0]
                preds[:, i] = np.asarray(fc).ravel()[:n_val]
            except Exception:
                preds[:, i] = np.mean(train)

        # Non-negative least squares (simplified)
        XtX = preds.T @ preds + 0.01 * np.eye(n_models)
        XtY = preds.T @ val
        w = np.linalg.solve(XtX, XtY)
        w = np.maximum(w, 0)
        total = np.sum(w)
        if total > 0:
            w = w / total
        else:
            w = np.ones(n_models) / n_models
        return list(w)

    def forecast(self, steps=1):
        if not self._fitted:
            raise RuntimeError("Not fitted")

        combined = np.zeros(steps)
        for i, (name, model) in enumerate(self.models):
            if self.weights[i] > 0:
                try:
                    fc = model.forecast(steps)
                    if isinstance(fc, tuple):
                        fc = fc[0]
                    fc = np.asarray(fc).ravel()
                    combined += self.weights[i] * fc[:steps]
                except Exception:
                    pass

        return combined


# ============================================================
# Backtesting / Time Series Cross-Validation
# ============================================================

class Backtester:
    """
    Walk-forward backtesting for time series models.
    Expands training window, forecasts, measures error.
    """

    def __init__(self, model_factory, metric='mse'):
        """
        model_factory: callable that returns a fresh model instance
        metric: 'mse', 'mae', 'rmse', 'mape', 'smape'
        """
        self.model_factory = model_factory
        self.metric = metric
        self.results = []

    def run(self, y, initial_window, horizon=1, step=1):
        """
        Run walk-forward backtest.
        initial_window: minimum training size
        horizon: forecast horizon
        step: how many observations to add each iteration
        """
        y = _validate_series(y)
        n = len(y)
        self.results = []

        t = initial_window
        while t + horizon <= n:
            train = y[:t]
            actual = y[t:t + horizon]

            model = self.model_factory()
            try:
                model.fit(train)
                fc = model.forecast(horizon)
                if isinstance(fc, tuple):
                    fc = fc[0]
                fc = np.asarray(fc).ravel()[:len(actual)]

                error = self._compute_error(actual, fc)
                self.results.append({
                    'train_end': t,
                    'forecast': fc,
                    'actual': actual,
                    'error': error
                })
            except Exception as e:
                self.results.append({
                    'train_end': t,
                    'forecast': None,
                    'actual': actual,
                    'error': float('inf'),
                    'exception': str(e)
                })

            t += step

        return self

    def _compute_error(self, actual, forecast):
        """Compute error metric."""
        actual = np.asarray(actual)
        forecast = np.asarray(forecast)

        if self.metric == 'mse':
            return float(np.mean((actual - forecast) ** 2))
        elif self.metric == 'rmse':
            return float(np.sqrt(np.mean((actual - forecast) ** 2)))
        elif self.metric == 'mae':
            return float(np.mean(np.abs(actual - forecast)))
        elif self.metric == 'mape':
            mask = actual != 0
            if not np.any(mask):
                return 0.0
            return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)
        elif self.metric == 'smape':
            denom = np.abs(actual) + np.abs(forecast)
            mask = denom > 0
            if not np.any(mask):
                return 0.0
            return float(np.mean(2 * np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100)
        else:
            return float(np.mean((actual - forecast) ** 2))

    @property
    def mean_error(self):
        valid = [r['error'] for r in self.results if r['error'] < float('inf')]
        return float(np.mean(valid)) if valid else float('inf')

    @property
    def std_error(self):
        valid = [r['error'] for r in self.results if r['error'] < float('inf')]
        return float(np.std(valid)) if valid else float('inf')

    @property
    def error_series(self):
        return [r['error'] for r in self.results]


# ============================================================
# Prediction Intervals
# ============================================================

class PredictionInterval:
    """
    Compute prediction intervals via:
    - Parametric (normal assumption)
    - Bootstrap (residual resampling)
    - Conformal prediction
    """

    @staticmethod
    def parametric(forecasts, residuals, confidence=0.95):
        """Normal-based prediction intervals."""
        z = norm.ppf((1 + confidence) / 2)
        sigma = np.std(residuals)
        forecasts = np.asarray(forecasts)
        lower = forecasts - z * sigma
        upper = forecasts + z * sigma
        return lower, upper

    @staticmethod
    def bootstrap(model_factory, y, steps=1, n_bootstrap=100, confidence=0.95):
        """Bootstrap prediction intervals via residual resampling."""
        y = _validate_series(y)

        # Fit original model
        model = model_factory()
        model.fit(y)
        fc_orig = model.forecast(steps)
        if isinstance(fc_orig, tuple):
            fc_orig = fc_orig[0]
        fc_orig = np.asarray(fc_orig).ravel()

        # Get residuals (simple: one-step in-sample)
        residuals = []
        for t in range(max(10, len(y) // 5), len(y)):
            m = model_factory()
            m.fit(y[:t])
            fc1 = m.forecast(1)
            if isinstance(fc1, tuple):
                fc1 = fc1[0]
            residuals.append(y[t] - np.asarray(fc1).ravel()[0])
        residuals = np.array(residuals)

        # Bootstrap
        rng = np.random.RandomState(42)
        bootstrap_forecasts = np.zeros((n_bootstrap, steps))
        for b in range(n_bootstrap):
            # Resample residuals and add to series
            boot_resid = rng.choice(residuals, size=len(y), replace=True)
            y_boot = y + boot_resid * 0.5  # damped bootstrap
            m = model_factory()
            try:
                m.fit(y_boot)
                fc = m.forecast(steps)
                if isinstance(fc, tuple):
                    fc = fc[0]
                bootstrap_forecasts[b] = np.asarray(fc).ravel()[:steps]
            except Exception:
                bootstrap_forecasts[b] = fc_orig

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_forecasts, alpha / 2 * 100, axis=0)
        upper = np.percentile(bootstrap_forecasts, (1 - alpha / 2) * 100, axis=0)
        return lower, upper

    @staticmethod
    def conformal(model_factory, y, steps=1, confidence=0.95):
        """
        Split conformal prediction intervals.
        """
        y = _validate_series(y)
        n = len(y)
        cal_size = max(int(n * 0.3), steps + 1)
        train = y[:n - cal_size]
        cal = y[n - cal_size:]

        # Compute nonconformity scores on calibration set
        scores = []
        for t in range(len(cal) - steps):
            m = model_factory()
            m.fit(np.concatenate([train, cal[:t]]))
            fc = m.forecast(steps)
            if isinstance(fc, tuple):
                fc = fc[0]
            fc = np.asarray(fc).ravel()
            actual = cal[t:t + steps]
            score = np.max(np.abs(actual[:len(fc)] - fc[:len(actual)]))
            scores.append(score)

        if len(scores) == 0:
            scores = [np.std(y)]

        # Quantile
        q = np.percentile(scores, confidence * 100)

        # Forecast on full data
        model = model_factory()
        model.fit(y)
        fc = model.forecast(steps)
        if isinstance(fc, tuple):
            fc = fc[0]
        fc = np.asarray(fc).ravel()

        return fc - q, fc + q


# ============================================================
# Forecast Combination Methods
# ============================================================

class ForecastCombiner:
    """
    Static forecast combination methods.
    """

    @staticmethod
    def simple_average(forecasts_list):
        """Equal-weight average."""
        arr = np.array(forecasts_list)
        return np.mean(arr, axis=0)

    @staticmethod
    def median(forecasts_list):
        """Median combination (robust to outliers)."""
        arr = np.array(forecasts_list)
        return np.median(arr, axis=0)

    @staticmethod
    def trimmed_mean(forecasts_list, trim_fraction=0.1):
        """Trimmed mean: drop extreme forecasts."""
        arr = np.array(forecasts_list)
        n = arr.shape[0]
        k = max(int(n * trim_fraction), 0)
        if k == 0 or n <= 2:
            return np.mean(arr, axis=0)
        sorted_arr = np.sort(arr, axis=0)
        return np.mean(sorted_arr[k:n - k], axis=0)

    @staticmethod
    def variance_weighted(forecasts_list, errors_list):
        """Weight by inverse variance of errors."""
        forecasts = np.array(forecasts_list)
        variances = np.array([np.var(e) if len(e) > 0 else float('inf') for e in errors_list])
        inv_var = np.where(variances > 0, 1.0 / variances, 0)
        total = np.sum(inv_var)
        if total == 0:
            return np.mean(forecasts, axis=0)
        weights = inv_var / total
        return np.average(forecasts, axis=0, weights=weights)


# ============================================================
# Diebold-Mariano Test
# ============================================================

def diebold_mariano_test(actual, forecast1, forecast2, loss='squared'):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    H0: both forecasts are equally accurate.
    Returns test statistic and p-value.
    """
    actual = np.asarray(actual)
    forecast1 = np.asarray(forecast1)
    forecast2 = np.asarray(forecast2)

    if loss == 'squared':
        e1 = (actual - forecast1) ** 2
        e2 = (actual - forecast2) ** 2
    elif loss == 'absolute':
        e1 = np.abs(actual - forecast1)
        e2 = np.abs(actual - forecast2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = e1 - e2
    n = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    if d_var == 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)


# ============================================================
# Data generation utilities
# ============================================================

def generate_seasonal_series(n=200, trend=0.01, seasonal_period=12,
                              seasonal_amp=5.0, noise_std=1.0, seed=42):
    """Generate a series with trend, seasonality, and noise."""
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    trend_comp = trend * t
    season_comp = seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)
    noise = rng.normal(0, noise_std, n)
    return trend_comp + season_comp + noise + 100  # offset for positivity


def generate_intermittent_series(n=200, demand_prob=0.3, mean_size=10.0, seed=42):
    """Generate intermittent demand series."""
    rng = np.random.RandomState(seed)
    series = np.zeros(n)
    for t in range(n):
        if rng.random() < demand_prob:
            series[t] = rng.exponential(mean_size)
    return series


def generate_garch_series(n=500, omega=0.1, alpha=0.1, beta=0.8, mu=0.0, seed=42):
    """Generate a series with GARCH(1,1) volatility."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta) if (alpha + beta) < 1 else 1.0
    y[0] = mu + rng.normal(0, np.sqrt(sigma2[0]))

    for t in range(1, n):
        sigma2[t] = omega + alpha * (y[t - 1] - mu) ** 2 + beta * sigma2[t - 1]
        y[t] = mu + rng.normal(0, np.sqrt(sigma2[t]))

    return y, sigma2


def generate_var_series(n=200, k=2, p=1, seed=42):
    """Generate multivariate VAR(p) series."""
    rng = np.random.RandomState(seed)
    Y = np.zeros((n, k))
    # Generate a stable coefficient matrix for any k
    A = rng.randn(k, k) * 0.3
    # Ensure stability: spectral radius < 1
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.95:
        A = A * 0.9 / max_eig

    for t in range(p, n):
        Y[t] = A @ Y[t - 1] + rng.multivariate_normal(np.zeros(k), np.eye(k) * 0.5)

    return Y
