"""
C188: Time Series Analysis
Built from scratch using only numpy.

Components:
- TimeSeries: core container with indexing, slicing, resampling
- Autocorrelation: ACF, PACF, Ljung-Box test
- SeasonalDecompose: additive/multiplicative decomposition (STL-like)
- ExponentialSmoothing: SES, Holt, Holt-Winters (additive/multiplicative)
- ARIMA: AR, MA, ARMA, ARIMA with differencing
- FourierAnalysis: FFT, spectral density, dominant frequencies, filtering
- ChangePointDetection: CUSUM, binary segmentation, PELT-like
- Forecaster: unified forecasting interface
"""

import numpy as np
from collections import namedtuple
import math


# ---------------------------------------------------------------------------
# TimeSeries container
# ---------------------------------------------------------------------------

class TimeSeries:
    """Core time series container."""

    def __init__(self, values, times=None, name="series"):
        self.values = np.array(values, dtype=float)
        if times is not None:
            self.times = np.array(times, dtype=float)
        else:
            self.times = np.arange(len(self.values), dtype=float)
        self.name = name

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TimeSeries(self.values[idx], self.times[idx], self.name)
        return self.values[idx]

    @property
    def mean(self):
        return float(np.mean(self.values))

    @property
    def std(self):
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 0.0

    @property
    def var(self):
        return float(np.var(self.values, ddof=1)) if len(self.values) > 1 else 0.0

    def diff(self, periods=1):
        """First difference."""
        d = self.values[periods:] - self.values[:-periods]
        return TimeSeries(d, self.times[periods:], f"{self.name}_diff{periods}")

    def rolling_mean(self, window):
        """Simple moving average."""
        kernel = np.ones(window) / window
        # Valid convolution
        smoothed = np.convolve(self.values, kernel, mode='valid')
        offset = window - 1
        return TimeSeries(smoothed, self.times[offset:], f"{self.name}_sma{window}")

    def rolling_std(self, window):
        """Rolling standard deviation."""
        n = len(self.values)
        result = np.zeros(n - window + 1)
        for i in range(len(result)):
            result[i] = np.std(self.values[i:i+window], ddof=1)
        offset = window - 1
        return TimeSeries(result, self.times[offset:], f"{self.name}_rstd{window}")

    def ewma(self, alpha):
        """Exponential weighted moving average."""
        result = np.zeros(len(self.values))
        result[0] = self.values[0]
        for i in range(1, len(self.values)):
            result[i] = alpha * self.values[i] + (1 - alpha) * result[i-1]
        return TimeSeries(result, self.times.copy(), f"{self.name}_ewma")

    def lag(self, k=1):
        """Return series lagged by k periods."""
        if k >= len(self.values):
            return TimeSeries([], [], f"{self.name}_lag{k}")
        return TimeSeries(self.values[:-k], self.times[:-k], f"{self.name}_lag{k}")

    def resample(self, factor):
        """Downsample by integer factor (average within each block)."""
        n = len(self.values)
        trimmed = n - (n % factor) if n % factor != 0 else n
        reshaped = self.values[:trimmed].reshape(-1, factor)
        new_vals = reshaped.mean(axis=1)
        new_times = self.times[:trimmed].reshape(-1, factor).mean(axis=1)
        return TimeSeries(new_vals, new_times, f"{self.name}_resample{factor}")

    def normalize(self):
        """Z-score normalization."""
        m, s = self.mean, self.std
        if s == 0:
            return TimeSeries(np.zeros_like(self.values), self.times.copy(), f"{self.name}_norm")
        return TimeSeries((self.values - m) / s, self.times.copy(), f"{self.name}_norm")

    def to_dict(self):
        return {"name": self.name, "values": self.values.tolist(), "times": self.times.tolist()}

    @staticmethod
    def from_dict(d):
        return TimeSeries(d["values"], d["times"], d.get("name", "series"))


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

class Autocorrelation:
    """ACF, PACF, and statistical tests."""

    @staticmethod
    def acf(series, max_lag=None):
        """Autocorrelation function."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        if max_lag is None:
            max_lag = min(n // 2, 40)
        max_lag = min(max_lag, n - 1)
        mean = np.mean(values)
        centered = values - mean
        c0 = np.sum(centered ** 2)
        if c0 == 0:
            return np.zeros(max_lag + 1)
        result = np.zeros(max_lag + 1)
        result[0] = 1.0
        for k in range(1, max_lag + 1):
            result[k] = np.sum(centered[:n-k] * centered[k:]) / c0
        return result

    @staticmethod
    def pacf(series, max_lag=None):
        """Partial autocorrelation via Durbin-Levinson recursion."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        if max_lag is None:
            max_lag = min(n // 2, 40)
        max_lag = min(max_lag, n - 1)

        acf_vals = Autocorrelation.acf(series, max_lag)
        result = np.zeros(max_lag + 1)
        result[0] = 1.0
        if max_lag == 0:
            return result

        # Durbin-Levinson
        phi = np.zeros((max_lag + 1, max_lag + 1))
        phi[1, 1] = acf_vals[1]
        result[1] = acf_vals[1]

        for k in range(2, max_lag + 1):
            num = acf_vals[k] - sum(phi[k-1, j] * acf_vals[k-j] for j in range(1, k))
            den = 1.0 - sum(phi[k-1, j] * acf_vals[j] for j in range(1, k))
            if abs(den) < 1e-15:
                break
            phi[k, k] = num / den
            for j in range(1, k):
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
            result[k] = phi[k, k]

        return result

    @staticmethod
    def ljung_box(series, max_lag=None):
        """Ljung-Box Q statistic for white noise test."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        if max_lag is None:
            max_lag = min(n // 4, 20)
        max_lag = min(max_lag, n - 1)
        acf_vals = Autocorrelation.acf(series, max_lag)
        q = 0.0
        for k in range(1, max_lag + 1):
            q += (acf_vals[k] ** 2) / (n - k)
        q *= n * (n + 2)
        # p-value approximation using chi-squared CDF (Wilson-Hilferty)
        p_value = 1.0 - _chi2_cdf(q, max_lag)
        return {"Q": float(q), "p_value": float(p_value), "lags": max_lag}


def _chi2_cdf(x, df):
    """Approximate chi-squared CDF using regularized incomplete gamma."""
    if x <= 0 or df <= 0:
        return 0.0
    return _regularized_gamma_p(df / 2.0, x / 2.0)


def _regularized_gamma_p(a, x):
    """Lower regularized incomplete gamma function P(a, x) via series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    if x < a + 1:
        # Series expansion
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * 1e-12:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        # Continued fraction (Lentz's method)
        f = 1e-30
        c = 1e-30
        d = 1.0 / (x - a + 1.0)
        f = d
        for i in range(1, 200):
            an = -i * (i - a)
            bn = x - a + 1.0 + 2.0 * i
            d = bn + an * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = bn + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            f *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return 1.0 - f * math.exp(-x + a * math.log(x) - math.lgamma(a))


# ---------------------------------------------------------------------------
# Seasonal Decomposition
# ---------------------------------------------------------------------------

DecompResult = namedtuple("DecompResult", ["observed", "trend", "seasonal", "residual"])


class SeasonalDecompose:
    """Classical seasonal decomposition (additive or multiplicative)."""

    @staticmethod
    def decompose(series, period, model="additive"):
        """
        Decompose series into trend, seasonal, and residual.
        model: 'additive' or 'multiplicative'
        """
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)

        # Trend via centered moving average
        if period % 2 == 0:
            # 2x moving average for even periods
            k1 = np.convolve(values, np.ones(period) / period, mode='valid')
            k2 = (k1[:-1] + k1[1:]) / 2.0
            offset = period // 2
            trend = np.full(n, np.nan)
            trend[offset:offset + len(k2)] = k2
        else:
            k = np.convolve(values, np.ones(period) / period, mode='valid')
            offset = period // 2
            trend = np.full(n, np.nan)
            trend[offset:offset + len(k)] = k

        # Detrend
        if model == "multiplicative":
            detrended = np.where(np.isnan(trend) | (trend == 0), np.nan, values / trend)
        else:
            detrended = np.where(np.isnan(trend), np.nan, values - trend)

        # Seasonal component: average detrended values for each position in the cycle
        seasonal = np.zeros(n)
        for i in range(period):
            indices = range(i, n, period)
            vals = [detrended[j] for j in indices if not np.isnan(detrended[j])]
            if vals:
                seasonal_val = np.mean(vals)
            else:
                seasonal_val = 0.0 if model == "additive" else 1.0
            for j in indices:
                seasonal[j] = seasonal_val

        # Normalize seasonal: additive sums to 0, multiplicative averages to 1
        if model == "additive":
            seasonal -= np.mean(seasonal[:period])
        else:
            seasonal_mean = np.mean(seasonal[:period])
            if seasonal_mean != 0:
                seasonal /= seasonal_mean

        # Residual
        if model == "multiplicative":
            residual = np.where(
                np.isnan(trend) | (seasonal == 0),
                np.nan,
                values / (trend * seasonal)
            )
        else:
            residual = np.where(np.isnan(trend), np.nan, values - trend - seasonal)

        return DecompResult(
            observed=values,
            trend=trend,
            seasonal=seasonal,
            residual=residual
        )


# ---------------------------------------------------------------------------
# Exponential Smoothing
# ---------------------------------------------------------------------------

class ExponentialSmoothing:
    """
    Exponential smoothing methods:
    - SES (Simple): level only
    - Holt: level + trend
    - Holt-Winters: level + trend + seasonal (additive or multiplicative)
    """

    @staticmethod
    def ses(series, alpha=None):
        """Simple Exponential Smoothing. Auto-fits alpha if None."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        if alpha is None:
            alpha = ExponentialSmoothing._optimize_ses(values)
        n = len(values)
        level = np.zeros(n + 1)
        level[0] = values[0]
        fitted = np.zeros(n)
        for t in range(n):
            fitted[t] = level[t]
            level[t+1] = alpha * values[t] + (1 - alpha) * level[t]
        return {
            "fitted": fitted,
            "level": level,
            "alpha": alpha,
            "forecast_next": float(level[n])
        }

    @staticmethod
    def _optimize_ses(values):
        """Grid search for optimal alpha."""
        best_alpha, best_sse = 0.1, float('inf')
        for a in np.arange(0.05, 1.0, 0.05):
            level = values[0]
            sse = 0.0
            for t in range(1, len(values)):
                sse += (values[t] - level) ** 2
                level = a * values[t] + (1 - a) * level
            if sse < best_sse:
                best_sse = sse
                best_alpha = a
        return float(round(best_alpha, 2))

    @staticmethod
    def holt(series, alpha=None, beta=None):
        """Holt's linear trend method."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        if alpha is None:
            alpha = 0.3
        if beta is None:
            beta = 0.1
        n = len(values)
        level = np.zeros(n + 1)
        trend = np.zeros(n + 1)
        level[0] = values[0]
        trend[0] = values[1] - values[0] if n > 1 else 0.0
        fitted = np.zeros(n)
        for t in range(n):
            fitted[t] = level[t] + trend[t]
            level[t+1] = alpha * values[t] + (1 - alpha) * (level[t] + trend[t])
            trend[t+1] = beta * (level[t+1] - level[t]) + (1 - beta) * trend[t]
        return {
            "fitted": fitted,
            "level": level,
            "trend": trend,
            "alpha": alpha,
            "beta": beta,
            "forecast_next": float(level[n] + trend[n])
        }

    @staticmethod
    def holt_winters(series, period, alpha=None, beta=None, gamma=None, model="additive"):
        """Holt-Winters seasonal method."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        if alpha is None:
            alpha = 0.3
        if beta is None:
            beta = 0.1
        if gamma is None:
            gamma = 0.3

        # Initialize
        level = np.zeros(n + 1)
        trend = np.zeros(n + 1)
        seasonal = np.zeros(n + period)

        # Initial level: mean of first period
        level[0] = np.mean(values[:period])
        # Initial trend: average change over first two periods
        if n >= 2 * period:
            trend[0] = np.mean(values[period:2*period] - values[:period]) / period
        else:
            trend[0] = (values[min(period, n-1)] - values[0]) / max(period, 1)

        # Initial seasonal indices
        for i in range(period):
            if model == "multiplicative":
                seasonal[i] = values[i] / level[0] if level[0] != 0 else 1.0
            else:
                seasonal[i] = values[i] - level[0]

        fitted = np.zeros(n)
        for t in range(n):
            if model == "multiplicative":
                fitted[t] = (level[t] + trend[t]) * seasonal[t]
                if seasonal[t] != 0:
                    level[t+1] = alpha * (values[t] / seasonal[t]) + (1 - alpha) * (level[t] + trend[t])
                else:
                    level[t+1] = alpha * values[t] + (1 - alpha) * (level[t] + trend[t])
                trend[t+1] = beta * (level[t+1] - level[t]) + (1 - beta) * trend[t]
                if (level[t+1] + trend[t+1]) != 0:
                    seasonal[t + period] = gamma * (values[t] / (level[t+1])) + (1 - gamma) * seasonal[t]
                else:
                    seasonal[t + period] = seasonal[t]
            else:
                fitted[t] = level[t] + trend[t] + seasonal[t]
                level[t+1] = alpha * (values[t] - seasonal[t]) + (1 - alpha) * (level[t] + trend[t])
                trend[t+1] = beta * (level[t+1] - level[t]) + (1 - beta) * trend[t]
                seasonal[t + period] = gamma * (values[t] - level[t+1]) + (1 - gamma) * seasonal[t]

        # Forecast
        forecasts = []
        for h in range(1, period + 1):
            if model == "multiplicative":
                forecasts.append(float((level[n] + h * trend[n]) * seasonal[n + h - period]))
            else:
                forecasts.append(float(level[n] + h * trend[n] + seasonal[n + h - period]))

        return {
            "fitted": fitted,
            "level": level,
            "trend": trend,
            "seasonal": seasonal,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "period": period,
            "model": model,
            "forecast": forecasts
        }


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

class ARIMA:
    """
    ARIMA(p, d, q) model.
    - AR(p): autoregressive
    - I(d): differencing
    - MA(q): moving average
    Estimation via conditional least squares (Yule-Walker for AR, then MA residuals).
    """

    def __init__(self, p=1, d=0, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.ar_params = None
        self.ma_params = None
        self.intercept = 0.0
        self._original = None
        self._differenced = None
        self._residuals = None

    def fit(self, series):
        """Fit ARIMA model to data."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        self._original = values.copy()

        # Differencing
        diff_values = values.copy()
        for _ in range(self.d):
            diff_values = np.diff(diff_values)
        self._differenced = diff_values

        n = len(diff_values)

        # Fit AR(p) via Yule-Walker
        if self.p > 0 and n > self.p:
            self.ar_params = self._fit_ar(diff_values, self.p)
        else:
            self.ar_params = np.array([])

        # Compute AR residuals
        residuals = self._compute_ar_residuals(diff_values)

        # Fit MA(q) from residuals
        if self.q > 0 and len(residuals) > self.q:
            self.ma_params = self._fit_ma(diff_values, residuals, self.q)
        else:
            self.ma_params = np.array([])

        # Recompute residuals with full model
        self._residuals = self._compute_residuals(diff_values)
        self.intercept = float(np.mean(diff_values) * (1 - np.sum(self.ar_params))) if self.p > 0 else float(np.mean(diff_values))

        return self

    def _fit_ar(self, values, p):
        """Yule-Walker estimation for AR coefficients."""
        n = len(values)
        mean = np.mean(values)
        centered = values - mean

        # Autocorrelation
        r = np.zeros(p + 1)
        for k in range(p + 1):
            r[k] = np.sum(centered[:n-k] * centered[k:]) / n

        if r[0] == 0:
            return np.zeros(p)

        # Toeplitz system
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = r[abs(i - j)]
        rhs = r[1:p+1]

        try:
            ar_coeffs = np.linalg.solve(R, rhs)
        except np.linalg.LinAlgError:
            ar_coeffs = np.zeros(p)

        return ar_coeffs

    def _fit_ma(self, values, residuals, q):
        """Estimate MA coefficients from innovations."""
        n = len(residuals)
        mean_r = np.mean(residuals)
        centered = residuals - mean_r

        # Cross-correlation of residuals
        r = np.zeros(q + 1)
        for k in range(q + 1):
            if n - k > 0:
                r[k] = np.sum(centered[:n-k] * centered[k:]) / n

        if r[0] == 0:
            return np.zeros(q)

        # Simple estimation: use autocorrelations of residuals
        ma_coeffs = np.zeros(q)
        for i in range(q):
            ma_coeffs[i] = r[i+1] / r[0] if r[0] != 0 else 0.0

        # Clamp for stability
        for i in range(q):
            ma_coeffs[i] = max(-0.99, min(0.99, ma_coeffs[i]))

        return ma_coeffs

    def _compute_ar_residuals(self, values):
        """Compute residuals from AR part only."""
        n = len(values)
        mean = np.mean(values)
        residuals = np.zeros(n)
        for t in range(n):
            pred = mean
            for i in range(min(self.p, t)):
                if i < len(self.ar_params):
                    pred += self.ar_params[i] * (values[t - i - 1] - mean)
            residuals[t] = values[t] - pred
        return residuals

    def _compute_residuals(self, values):
        """Compute residuals from full ARIMA model."""
        n = len(values)
        mean = np.mean(values)
        residuals = np.zeros(n)
        for t in range(n):
            pred = self.intercept
            # AR part
            for i in range(min(self.p, t)):
                if i < len(self.ar_params):
                    pred += self.ar_params[i] * values[t - i - 1]
            # MA part
            for j in range(min(self.q, t)):
                if j < len(self.ma_params):
                    pred += self.ma_params[j] * residuals[t - j - 1]
            residuals[t] = values[t] - pred
        return residuals

    def forecast(self, steps=1):
        """Forecast future values."""
        if self._differenced is None:
            raise ValueError("Model not fitted")

        diff_values = self._differenced
        residuals = self._residuals if self._residuals is not None else np.zeros(len(diff_values))
        n = len(diff_values)

        # Forecast on differenced scale
        forecasts_diff = []
        extended_vals = list(diff_values)
        extended_res = list(residuals)

        for h in range(steps):
            pred = self.intercept
            for i in range(self.p):
                idx = n + h - i - 1
                if idx >= 0 and idx < len(extended_vals):
                    pred += self.ar_params[i] * extended_vals[idx]
            for j in range(self.q):
                idx = n + h - j - 1
                if idx >= 0 and idx < len(extended_res):
                    pred += self.ma_params[j] * extended_res[idx]
            forecasts_diff.append(pred)
            extended_vals.append(pred)
            extended_res.append(0.0)  # Future residuals are 0

        # Undo differencing
        forecasts = np.array(forecasts_diff)
        original = self._original
        for _ in range(self.d):
            last_val = original[-1]
            undiffed = np.zeros(len(forecasts))
            undiffed[0] = last_val + forecasts[0]
            for i in range(1, len(forecasts)):
                undiffed[i] = undiffed[i-1] + forecasts[i]
            forecasts = undiffed
            original = np.append(original, forecasts)

        return forecasts

    def fitted_values(self):
        """Return fitted values on original scale."""
        if self._differenced is None:
            raise ValueError("Model not fitted")
        diff_vals = self._differenced
        residuals = self._residuals
        fitted_diff = diff_vals - residuals

        # Undo differencing for fitted values
        if self.d == 0:
            return fitted_diff

        # For d>0, we need to reconstruct from first d values of original
        result = self._original.copy()
        # The fitted differenced values start after d values
        fitted_orig = np.zeros(len(self._original))
        fitted_orig[:self.d] = self._original[:self.d]
        for i in range(len(fitted_diff)):
            if self.d == 1:
                fitted_orig[i + self.d] = fitted_orig[i + self.d - 1] + fitted_diff[i]
            elif self.d == 2:
                fitted_orig[i + self.d] = 2 * fitted_orig[i + self.d - 1] - fitted_orig[i + self.d - 2] + fitted_diff[i]
        return fitted_orig

    def aic(self):
        """Akaike Information Criterion."""
        if self._residuals is None:
            raise ValueError("Model not fitted")
        n = len(self._residuals)
        sigma2 = np.sum(self._residuals ** 2) / n
        if sigma2 <= 0:
            return float('inf')
        k = self.p + self.q + 1  # +1 for intercept
        return float(n * np.log(sigma2) + 2 * k)

    def bic(self):
        """Bayesian Information Criterion."""
        if self._residuals is None:
            raise ValueError("Model not fitted")
        n = len(self._residuals)
        sigma2 = np.sum(self._residuals ** 2) / n
        if sigma2 <= 0:
            return float('inf')
        k = self.p + self.q + 1
        return float(n * np.log(sigma2) + k * np.log(n))


# ---------------------------------------------------------------------------
# Fourier Analysis
# ---------------------------------------------------------------------------

class FourierAnalysis:
    """FFT-based spectral analysis."""

    @staticmethod
    def fft(series):
        """Compute FFT and return frequencies and magnitudes."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        # Remove mean
        centered = values - np.mean(values)
        fft_vals = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(n)
        magnitudes = np.abs(fft_vals) * 2.0 / n
        return {"frequencies": freqs, "magnitudes": magnitudes, "complex": fft_vals}

    @staticmethod
    def power_spectral_density(series):
        """Power spectral density (periodogram)."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        centered = values - np.mean(values)
        fft_vals = np.fft.rfft(centered)
        psd = (np.abs(fft_vals) ** 2) / n
        freqs = np.fft.rfftfreq(n)
        return {"frequencies": freqs, "psd": psd}

    @staticmethod
    def dominant_frequencies(series, top_k=5):
        """Find the top-k dominant frequencies."""
        result = FourierAnalysis.fft(series)
        freqs = result["frequencies"]
        mags = result["magnitudes"]
        # Skip DC component (index 0)
        if len(mags) <= 1:
            return []
        indices = np.argsort(mags[1:])[::-1][:top_k] + 1
        return [
            {"frequency": float(freqs[i]), "period": float(1.0 / freqs[i]) if freqs[i] > 0 else float('inf'), "magnitude": float(mags[i])}
            for i in indices if mags[i] > 0
        ]

    @staticmethod
    def low_pass_filter(series, cutoff_freq):
        """Low-pass filter: remove frequencies above cutoff."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        mean = np.mean(values)
        fft_vals = np.fft.rfft(values - mean)
        freqs = np.fft.rfftfreq(n)
        fft_vals[freqs > cutoff_freq] = 0
        filtered = np.fft.irfft(fft_vals, n=n) + mean
        return TimeSeries(filtered, name="filtered") if isinstance(series, TimeSeries) else filtered

    @staticmethod
    def band_pass_filter(series, low_freq, high_freq):
        """Band-pass filter: keep frequencies between low and high."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        mean = np.mean(values)
        fft_vals = np.fft.rfft(values - mean)
        freqs = np.fft.rfftfreq(n)
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        fft_vals[~mask] = 0
        filtered = np.fft.irfft(fft_vals, n=n) + mean
        return TimeSeries(filtered, name="band_filtered") if isinstance(series, TimeSeries) else filtered

    @staticmethod
    def reconstruct(series, n_components):
        """Reconstruct series using top N frequency components."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        mean = np.mean(values)
        centered = values - mean
        fft_vals = np.fft.rfft(centered)
        mags = np.abs(fft_vals)
        # Keep top n_components (by magnitude, excluding DC)
        if len(mags) <= 1:
            return TimeSeries(np.full(n, mean), name="reconstructed")
        threshold_idx = np.argsort(mags[1:])[::-1][:n_components] + 1
        filtered = np.zeros_like(fft_vals)
        for idx in threshold_idx:
            filtered[idx] = fft_vals[idx]
        result = np.fft.irfft(filtered, n=n) + mean
        return TimeSeries(result, name="reconstructed")


# ---------------------------------------------------------------------------
# Change Point Detection
# ---------------------------------------------------------------------------

class ChangePointDetection:
    """Detect structural breaks in time series."""

    @staticmethod
    def cusum(series, threshold=None):
        """CUSUM (cumulative sum) change point detection."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        mean = np.mean(values)
        cusum = np.cumsum(values - mean)

        if threshold is None:
            threshold = 2.0 * np.std(values) * np.sqrt(n)

        # Find point of maximum absolute CUSUM
        abs_cusum = np.abs(cusum)
        max_idx = int(np.argmax(abs_cusum))
        max_val = abs_cusum[max_idx]

        change_points = []
        if max_val > threshold:
            change_points.append({"index": max_idx, "cusum": float(cusum[max_idx]), "significance": float(max_val / threshold)})

        return {"cusum": cusum, "threshold": threshold, "change_points": change_points}

    @staticmethod
    def binary_segmentation(series, min_segment=5, max_change_points=10, penalty=None):
        """Binary segmentation for multiple change points."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)

        if penalty is None:
            penalty = np.log(n) * np.var(values)

        change_points = []
        segments = [(0, n)]

        for _ in range(max_change_points):
            best_gain = -float('inf')
            best_point = None
            best_seg_idx = None

            for seg_idx, (start, end) in enumerate(segments):
                if end - start < 2 * min_segment:
                    continue

                seg = values[start:end]
                total_cost = _segment_cost(seg)

                for cp in range(min_segment, len(seg) - min_segment):
                    left_cost = _segment_cost(seg[:cp])
                    right_cost = _segment_cost(seg[cp:])
                    gain = total_cost - left_cost - right_cost - penalty

                    if gain > best_gain:
                        best_gain = gain
                        best_point = start + cp
                        best_seg_idx = seg_idx

            if best_gain <= 0 or best_point is None:
                break

            change_points.append(best_point)
            # Split segment
            old_start, old_end = segments[best_seg_idx]
            segments[best_seg_idx] = (old_start, best_point)
            segments.insert(best_seg_idx + 1, (best_point, old_end))

        change_points.sort()
        return {"change_points": change_points, "n_segments": len(segments), "segments": segments}

    @staticmethod
    def pelt(series, min_segment=2, penalty=None):
        """
        PELT (Pruned Exact Linear Time) change point detection.
        Uses mean-shift cost function.
        """
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)

        if penalty is None:
            penalty = np.log(n) * np.var(values) if np.var(values) > 0 else 1.0

        # Cumulative sums for O(1) cost computation
        cum_sum = np.zeros(n + 1)
        cum_sq_sum = np.zeros(n + 1)
        for i in range(n):
            cum_sum[i+1] = cum_sum[i] + values[i]
            cum_sq_sum[i+1] = cum_sq_sum[i] + values[i] ** 2

        def cost(start, end):
            """Cost of segment [start, end)."""
            length = end - start
            if length <= 0:
                return 0.0
            s = cum_sum[end] - cum_sum[start]
            sq = cum_sq_sum[end] - cum_sq_sum[start]
            return sq - (s ** 2) / length

        # Dynamic programming
        F = np.full(n + 1, float('inf'))
        F[0] = -penalty
        cp_list = [[] for _ in range(n + 1)]
        candidates = [0]

        for t in range(min_segment, n + 1):
            best_cost = float('inf')
            best_tau = 0
            new_candidates = []

            for tau in candidates:
                if t - tau < min_segment:
                    continue
                c = F[tau] + cost(tau, t) + penalty
                if c < best_cost:
                    best_cost = c
                    best_tau = tau

            F[t] = best_cost
            cp_list[t] = cp_list[best_tau] + ([best_tau] if best_tau > 0 else [])

            # Pruning
            for tau in candidates:
                if F[tau] + cost(tau, t) <= F[t]:
                    new_candidates.append(tau)
            new_candidates.append(t)
            candidates = new_candidates

        change_points = sorted(cp_list[n])
        return {"change_points": change_points, "cost": float(F[n])}

    @staticmethod
    def mean_shift_test(series, index):
        """Test if there's a significant mean shift at given index."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        left = values[:index]
        right = values[index:]
        if len(left) < 2 or len(right) < 2:
            return {"significant": False, "t_stat": 0.0, "mean_diff": 0.0}

        mean_diff = np.mean(right) - np.mean(left)
        pooled_var = (np.var(left, ddof=1) / len(left) + np.var(right, ddof=1) / len(right))
        if pooled_var <= 0:
            t_stat = float('inf') if mean_diff != 0 else 0.0
        else:
            t_stat = abs(mean_diff) / np.sqrt(pooled_var)

        # Approximate p-value using t-distribution -> normal for large n
        p_value = 2 * (1 - _normal_cdf(t_stat))
        return {
            "significant": p_value < 0.05,
            "t_stat": float(t_stat),
            "mean_diff": float(mean_diff),
            "p_value": float(p_value)
        }


def _segment_cost(segment):
    """Cost function: sum of squared deviations from mean."""
    if len(segment) == 0:
        return 0.0
    return float(np.sum((segment - np.mean(segment)) ** 2))


def _normal_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Stationarity Tests
# ---------------------------------------------------------------------------

class StationarityTests:
    """Tests for stationarity."""

    @staticmethod
    def adf_test(series, max_lag=None):
        """
        Augmented Dickey-Fuller test (simplified).
        Tests H0: unit root (non-stationary) vs H1: stationary.
        Returns test statistic and critical values.
        """
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)
        if max_lag is None:
            max_lag = int(np.floor((n - 1) ** (1/3)))
        max_lag = min(max_lag, n // 3)

        # Difference
        dy = np.diff(values)
        y_lag = values[:-1]

        # Build regression matrix: dy_t = alpha + beta * y_{t-1} + sum(gamma_i * dy_{t-i}) + e_t
        T = len(dy) - max_lag
        if T < 5:
            return {"statistic": 0.0, "critical_values": {}, "stationary": False}

        X = np.ones((T, 2 + max_lag))
        Y = dy[max_lag:]
        X[:, 1] = y_lag[max_lag:max_lag + T]
        for i in range(max_lag):
            X[:, 2 + i] = dy[max_lag - i - 1:max_lag - i - 1 + T]

        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"statistic": 0.0, "critical_values": {}, "stationary": False}

        residuals = Y - X @ beta
        sigma2 = np.sum(residuals ** 2) / (T - X.shape[1])
        if sigma2 <= 0:
            return {"statistic": 0.0, "critical_values": {}, "stationary": False}

        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return {"statistic": 0.0, "critical_values": {}, "stationary": False}

        se_beta = np.sqrt(abs(cov[1, 1]))
        if se_beta == 0:
            return {"statistic": 0.0, "critical_values": {}, "stationary": False}

        t_stat = beta[1] / se_beta

        # Critical values (approximate for n=100)
        critical = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
        stationary = t_stat < critical["5%"]

        return {
            "statistic": float(t_stat),
            "critical_values": critical,
            "stationary": stationary,
            "lag_order": max_lag
        }

    @staticmethod
    def kpss_test(series, regression="c"):
        """
        KPSS test (simplified).
        H0: stationary vs H1: unit root.
        regression: 'c' (constant) or 'ct' (constant + trend)
        """
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        n = len(values)

        if regression == "ct":
            t = np.arange(n)
            X = np.column_stack([np.ones(n), t])
        else:
            X = np.ones((n, 1))

        try:
            beta = np.linalg.lstsq(X, values, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"statistic": 0.0, "critical_values": {}, "stationary": True}

        residuals = values - X @ beta
        cum_resid = np.cumsum(residuals)
        s2 = np.sum(residuals ** 2) / n

        # Long-run variance (Bartlett kernel)
        lag_trunc = int(np.sqrt(n))
        for l in range(1, lag_trunc + 1):
            weight = 1 - l / (lag_trunc + 1)
            autocovar = np.sum(residuals[:n-l] * residuals[l:]) / n
            s2 += 2 * weight * autocovar

        if s2 <= 0:
            return {"statistic": 0.0, "critical_values": {}, "stationary": True}

        stat = np.sum(cum_resid ** 2) / (n ** 2 * s2)

        if regression == "ct":
            critical = {"1%": 0.216, "5%": 0.146, "10%": 0.119}
        else:
            critical = {"1%": 0.739, "5%": 0.463, "10%": 0.347}

        stationary = stat < critical["5%"]

        return {
            "statistic": float(stat),
            "critical_values": critical,
            "stationary": stationary
        }


# ---------------------------------------------------------------------------
# Forecaster (unified interface)
# ---------------------------------------------------------------------------

class Forecaster:
    """Unified forecasting interface."""

    def __init__(self, method="auto"):
        """
        method: 'auto', 'ses', 'holt', 'holt_winters', 'arima'
        """
        self.method = method
        self._model = None
        self._result = None
        self._series = None

    def fit(self, series, period=None, **kwargs):
        """Fit the forecaster to data."""
        values = np.array(series.values if isinstance(series, TimeSeries) else series)
        self._series = values

        if self.method == "auto":
            self.method = self._auto_select(values, period)

        if self.method == "ses":
            self._result = ExponentialSmoothing.ses(values, **kwargs)
        elif self.method == "holt":
            self._result = ExponentialSmoothing.holt(values, **kwargs)
        elif self.method == "holt_winters":
            if period is None:
                period = self._detect_period(values)
            self._result = ExponentialSmoothing.holt_winters(values, period, **kwargs)
        elif self.method == "arima":
            p = kwargs.get("p", 1)
            d = kwargs.get("d", 0)
            q = kwargs.get("q", 0)
            self._model = ARIMA(p, d, q)
            self._model.fit(values)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def forecast(self, steps=1):
        """Generate forecasts."""
        if self.method == "arima" and self._model is not None:
            return self._model.forecast(steps)
        elif self._result is not None:
            if "forecast" in self._result:
                return np.array(self._result["forecast"][:steps])
            else:
                # SES/Holt: repeat last forecast
                next_val = self._result["forecast_next"]
                return np.full(steps, next_val)
        raise ValueError("Model not fitted")

    def fitted_values(self):
        """Return in-sample fitted values."""
        if self.method == "arima" and self._model is not None:
            return self._model.fitted_values()
        elif self._result is not None:
            return self._result["fitted"]
        raise ValueError("Model not fitted")

    def _auto_select(self, values, period):
        """Auto-select best method."""
        n = len(values)
        if period is not None and period > 1 and n >= 2 * period:
            return "holt_winters"
        # Check for trend
        slope = np.polyfit(np.arange(n), values, 1)[0]
        if abs(slope) > 0.01 * np.std(values):
            return "holt"
        return "ses"

    def _detect_period(self, values):
        """Auto-detect seasonality period via ACF."""
        acf_vals = Autocorrelation.acf(values, max_lag=min(len(values) // 2, 100))
        # Find first significant peak after lag 1
        for i in range(2, len(acf_vals) - 1):
            if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1]:
                if acf_vals[i] > 0.1:
                    return i
        return 12  # Default


# ---------------------------------------------------------------------------
# Utility: generate synthetic time series
# ---------------------------------------------------------------------------

def generate_series(n=200, trend=0.0, seasonal_period=0, seasonal_amp=0.0,
                    noise_std=1.0, ar_coefs=None, change_points=None, seed=None):
    """Generate synthetic time series for testing."""
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n, dtype=float)
    values = np.zeros(n)

    # Trend
    values += trend * t

    # Seasonality
    if seasonal_period > 0 and seasonal_amp > 0:
        values += seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)

    # AR process
    if ar_coefs is not None:
        noise = np.random.randn(n) * noise_std
        for i in range(len(ar_coefs), n):
            for j, c in enumerate(ar_coefs):
                values[i] += c * values[i - j - 1]
            values[i] += noise[i]
    else:
        values += np.random.randn(n) * noise_std

    # Change points (mean shift)
    if change_points is not None:
        for cp_idx, shift in change_points:
            values[cp_idx:] += shift

    return TimeSeries(values, t)
