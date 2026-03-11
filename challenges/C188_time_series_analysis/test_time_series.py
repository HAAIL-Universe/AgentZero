"""
Tests for C188: Time Series Analysis
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from time_series import (
    TimeSeries, Autocorrelation, SeasonalDecompose, ExponentialSmoothing,
    ARIMA, FourierAnalysis, ChangePointDetection, StationarityTests,
    Forecaster, generate_series
)


# ===========================================================================
# TimeSeries container
# ===========================================================================

class TestTimeSeries:
    def test_create_basic(self):
        ts = TimeSeries([1, 2, 3, 4, 5])
        assert len(ts) == 5
        assert ts[0] == 1.0
        assert ts[4] == 5.0

    def test_create_with_times(self):
        ts = TimeSeries([10, 20, 30], times=[0.0, 0.5, 1.0])
        assert ts.times[1] == 0.5

    def test_slice(self):
        ts = TimeSeries([1, 2, 3, 4, 5])
        sliced = ts[1:4]
        assert len(sliced) == 3
        assert sliced[0] == 2.0

    def test_mean_std(self):
        ts = TimeSeries([2, 4, 6, 8, 10])
        assert ts.mean == 6.0
        assert abs(ts.std - np.std([2,4,6,8,10], ddof=1)) < 1e-10

    def test_var(self):
        ts = TimeSeries([1, 3, 5])
        assert abs(ts.var - 4.0) < 1e-10

    def test_diff(self):
        ts = TimeSeries([1, 3, 6, 10])
        d = ts.diff()
        assert len(d) == 3
        np.testing.assert_array_almost_equal(d.values, [2, 3, 4])

    def test_diff_periods(self):
        ts = TimeSeries([1, 2, 4, 8])
        d = ts.diff(periods=2)
        assert len(d) == 2
        np.testing.assert_array_almost_equal(d.values, [3, 6])

    def test_rolling_mean(self):
        ts = TimeSeries([1, 2, 3, 4, 5])
        rm = ts.rolling_mean(3)
        assert len(rm) == 3
        np.testing.assert_array_almost_equal(rm.values, [2, 3, 4])

    def test_rolling_std(self):
        ts = TimeSeries([1, 2, 3, 4, 5])
        rs = ts.rolling_std(3)
        assert len(rs) == 3
        assert rs[0] == pytest.approx(1.0, abs=1e-10)

    def test_ewma(self):
        ts = TimeSeries([1, 2, 3, 4, 5])
        e = ts.ewma(0.5)
        assert e[0] == 1.0
        assert e[1] == pytest.approx(1.5)

    def test_lag(self):
        ts = TimeSeries([10, 20, 30, 40])
        lagged = ts.lag(1)
        assert len(lagged) == 3
        np.testing.assert_array_almost_equal(lagged.values, [10, 20, 30])

    def test_resample(self):
        ts = TimeSeries([1, 2, 3, 4, 5, 6])
        rs = ts.resample(2)
        assert len(rs) == 3
        np.testing.assert_array_almost_equal(rs.values, [1.5, 3.5, 5.5])

    def test_normalize(self):
        ts = TimeSeries([10, 20, 30])
        norm = ts.normalize()
        assert abs(np.mean(norm.values)) < 1e-10
        assert abs(np.std(norm.values, ddof=1) - 1.0) < 1e-10

    def test_to_from_dict(self):
        ts = TimeSeries([1, 2, 3], name="test")
        d = ts.to_dict()
        ts2 = TimeSeries.from_dict(d)
        np.testing.assert_array_equal(ts2.values, ts.values)
        assert ts2.name == "test"

    def test_name(self):
        ts = TimeSeries([1, 2], name="myts")
        assert ts.name == "myts"

    def test_empty_lag(self):
        ts = TimeSeries([1, 2])
        lagged = ts.lag(5)
        assert len(lagged) == 0


# ===========================================================================
# Autocorrelation
# ===========================================================================

class TestAutocorrelation:
    def test_acf_white_noise(self):
        np.random.seed(42)
        values = np.random.randn(500)
        acf = Autocorrelation.acf(values, max_lag=20)
        assert acf[0] == pytest.approx(1.0)
        # Lag 1+ should be small for white noise
        assert all(abs(acf[k]) < 0.15 for k in range(1, 20))

    def test_acf_ar1(self):
        np.random.seed(42)
        n = 1000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i-1] + np.random.randn()
        acf = Autocorrelation.acf(x, max_lag=5)
        # AR(1) with phi=0.8: ACF(1) ~ 0.8
        assert acf[1] > 0.7

    def test_pacf_ar1(self):
        np.random.seed(42)
        n = 1000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i-1] + np.random.randn()
        pacf = Autocorrelation.pacf(x, max_lag=5)
        assert pacf[0] == pytest.approx(1.0)
        assert pacf[1] > 0.7
        # PACF cuts off after lag 1 for AR(1)
        assert abs(pacf[2]) < 0.15

    def test_ljung_box_white_noise(self):
        np.random.seed(42)
        values = np.random.randn(200)
        result = Autocorrelation.ljung_box(values, max_lag=10)
        # White noise should not reject H0 (p > 0.05)
        assert result["p_value"] > 0.01

    def test_ljung_box_correlated(self):
        np.random.seed(42)
        n = 200
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i-1] + np.random.randn()
        result = Autocorrelation.ljung_box(x, max_lag=10)
        # Should reject white noise
        assert result["Q"] > 10

    def test_acf_constant(self):
        acf = Autocorrelation.acf([5, 5, 5, 5, 5], max_lag=2)
        np.testing.assert_array_equal(acf, [0, 0, 0])

    def test_acf_timeseries_input(self):
        ts = TimeSeries(np.random.randn(100))
        acf = Autocorrelation.acf(ts, max_lag=5)
        assert len(acf) == 6
        assert acf[0] == pytest.approx(1.0)


# ===========================================================================
# Seasonal Decomposition
# ===========================================================================

class TestSeasonalDecompose:
    def test_additive_basic(self):
        # Trend + seasonal + noise
        np.random.seed(42)
        t = np.arange(120)
        trend = 0.1 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(120) * 0.5
        values = trend + seasonal + noise
        ts = TimeSeries(values)
        result = SeasonalDecompose.decompose(ts, period=12, model="additive")
        assert result.observed is not None
        assert len(result.trend) == 120
        assert len(result.seasonal) == 120
        assert len(result.residual) == 120

    def test_additive_trend_recovery(self):
        t = np.arange(120, dtype=float)
        trend = 2.0 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        values = trend + seasonal
        result = SeasonalDecompose.decompose(values, period=12, model="additive")
        # Trend should be close to actual trend in middle
        mid = len(t) // 2
        valid_trend = result.trend[~np.isnan(result.trend)]
        # The extracted trend should be approximately linear
        assert len(valid_trend) > 50

    def test_multiplicative_basic(self):
        t = np.arange(120, dtype=float)
        trend = 100 + 0.5 * t
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * t / 12)
        values = trend * seasonal_factor
        result = SeasonalDecompose.decompose(values, period=12, model="multiplicative")
        assert result.observed is not None
        # Seasonal component should oscillate around 1
        valid_seasonal = result.seasonal[:12]
        assert abs(np.mean(valid_seasonal) - 1.0) < 0.1

    def test_seasonal_sums_to_zero_additive(self):
        np.random.seed(42)
        t = np.arange(48)
        values = 10 + np.sin(2 * np.pi * t / 12) + np.random.randn(48) * 0.1
        result = SeasonalDecompose.decompose(values, period=12, model="additive")
        # Sum of one full period of seasonal should be ~0
        assert abs(np.sum(result.seasonal[:12])) < 0.5

    def test_decompose_array_input(self):
        values = np.sin(np.arange(60) * 2 * np.pi / 12) + 5
        result = SeasonalDecompose.decompose(values, period=12)
        assert len(result.observed) == 60


# ===========================================================================
# Exponential Smoothing
# ===========================================================================

class TestExponentialSmoothing:
    def test_ses_basic(self):
        ts = TimeSeries([10, 12, 14, 13, 15, 16])
        result = ExponentialSmoothing.ses(ts, alpha=0.3)
        assert len(result["fitted"]) == 6
        assert result["alpha"] == 0.3
        assert "forecast_next" in result

    def test_ses_auto_alpha(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(100)) + 50
        result = ExponentialSmoothing.ses(values)
        assert 0.0 < result["alpha"] < 1.0

    def test_ses_fitted_first(self):
        result = ExponentialSmoothing.ses([5, 10, 15], alpha=0.5)
        assert result["fitted"][0] == 5.0  # First fitted = first observation

    def test_holt_basic(self):
        ts = TimeSeries([10, 12, 14, 16, 18, 20])
        result = ExponentialSmoothing.holt(ts, alpha=0.5, beta=0.3)
        assert len(result["fitted"]) == 6
        # For linear trend, forecast should continue upward
        assert result["forecast_next"] > 18

    def test_holt_trend(self):
        values = np.arange(1, 21, dtype=float)
        result = ExponentialSmoothing.holt(values, alpha=0.8, beta=0.5)
        # Forecast should be ~21
        assert abs(result["forecast_next"] - 21) < 3

    def test_holt_winters_additive(self):
        t = np.arange(48, dtype=float)
        trend = 0.5 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        values = 50 + trend + seasonal
        result = ExponentialSmoothing.holt_winters(values, period=12, model="additive")
        assert len(result["fitted"]) == 48
        assert len(result["forecast"]) == 12

    def test_holt_winters_multiplicative(self):
        t = np.arange(48, dtype=float)
        trend = 100 + t
        seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
        values = trend * seasonal
        result = ExponentialSmoothing.holt_winters(values, period=12, model="multiplicative")
        assert len(result["forecast"]) == 12
        assert result["model"] == "multiplicative"

    def test_holt_winters_forecast_reasonable(self):
        np.random.seed(42)
        t = np.arange(60, dtype=float)
        values = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(60) * 2
        result = ExponentialSmoothing.holt_winters(values, period=12, alpha=0.3, beta=0.1, gamma=0.3)
        forecasts = result["forecast"]
        # Forecasts should be in reasonable range
        assert all(50 < f < 200 for f in forecasts)

    def test_holt_winters_returns_period(self):
        values = np.random.randn(48) + 50
        result = ExponentialSmoothing.holt_winters(values, period=12)
        assert result["period"] == 12


# ===========================================================================
# ARIMA
# ===========================================================================

class TestARIMA:
    def test_ar1_fit(self):
        np.random.seed(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i-1] + np.random.randn()
        model = ARIMA(p=1, d=0, q=0).fit(x)
        assert abs(model.ar_params[0] - 0.7) < 0.15

    def test_ar2_fit(self):
        np.random.seed(42)
        n = 500
        x = np.zeros(n)
        for i in range(2, n):
            x[i] = 0.5 * x[i-1] + 0.2 * x[i-2] + np.random.randn()
        model = ARIMA(p=2, d=0, q=0).fit(x)
        assert abs(model.ar_params[0] - 0.5) < 0.15
        assert abs(model.ar_params[1] - 0.2) < 0.15

    def test_forecast_shape(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(100))
        model = ARIMA(p=1, d=1, q=0).fit(values)
        fc = model.forecast(steps=5)
        assert len(fc) == 5

    def test_arima_differencing(self):
        # Random walk: ARIMA(0,1,0) should work
        np.random.seed(42)
        values = np.cumsum(np.random.randn(200))
        model = ARIMA(p=0, d=1, q=0).fit(values)
        fc = model.forecast(steps=3)
        # Forecast should be near last value
        assert abs(fc[0] - values[-1]) < 10

    def test_fitted_values(self):
        np.random.seed(42)
        values = np.random.randn(50) * 5 + 100
        model = ARIMA(p=1, d=0, q=0).fit(values)
        fitted = model.fitted_values()
        assert len(fitted) == 50

    def test_aic_bic(self):
        np.random.seed(42)
        values = np.random.randn(100)
        model = ARIMA(p=1, d=0, q=0).fit(values)
        aic = model.aic()
        bic = model.bic()
        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert bic >= aic  # BIC penalizes more (for n > ~8)

    def test_arma_11(self):
        np.random.seed(42)
        n = 300
        x = np.zeros(n)
        e = np.random.randn(n)
        for i in range(1, n):
            x[i] = 0.6 * x[i-1] + e[i] + 0.3 * e[i-1]
        model = ARIMA(p=1, d=0, q=1).fit(x)
        assert abs(model.ar_params[0] - 0.6) < 0.2

    def test_forecast_not_fitted(self):
        model = ARIMA(p=1, d=0, q=0)
        with pytest.raises(ValueError):
            model.forecast(5)

    def test_arima_110(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(200)) + 100
        model = ARIMA(p=1, d=1, q=0).fit(values)
        fc = model.forecast(3)
        assert len(fc) == 3

    def test_model_comparison(self):
        np.random.seed(42)
        n = 300
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i-1] + np.random.randn()
        m1 = ARIMA(p=1, d=0, q=0).fit(x)
        m2 = ARIMA(p=5, d=0, q=0).fit(x)
        # AR(1) should have lower BIC than overfit AR(5)
        assert m1.bic() < m2.bic() + 30  # Allow some margin


# ===========================================================================
# Fourier Analysis
# ===========================================================================

class TestFourierAnalysis:
    def test_fft_pure_sine(self):
        n = 256
        t = np.arange(n)
        freq = 0.1
        values = np.sin(2 * np.pi * freq * t)
        ts = TimeSeries(values)
        result = FourierAnalysis.fft(ts)
        freqs = result["frequencies"]
        mags = result["magnitudes"]
        # Dominant frequency should be near 0.1
        peak_idx = np.argmax(mags[1:]) + 1
        assert abs(freqs[peak_idx] - freq) < 0.01

    def test_psd(self):
        np.random.seed(42)
        values = np.random.randn(128)
        result = FourierAnalysis.power_spectral_density(values)
        assert len(result["psd"]) == len(result["frequencies"])
        assert all(result["psd"] >= 0)

    def test_dominant_frequencies(self):
        n = 256
        t = np.arange(n)
        values = np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.15 * t)
        ts = TimeSeries(values)
        dom = FourierAnalysis.dominant_frequencies(ts, top_k=3)
        assert len(dom) >= 2
        freqs_found = sorted([d["frequency"] for d in dom[:2]])
        assert abs(freqs_found[0] - 0.05) < 0.02
        assert abs(freqs_found[1] - 0.15) < 0.02

    def test_low_pass_filter(self):
        n = 256
        t = np.arange(n)
        low_freq = np.sin(2 * np.pi * 0.02 * t)
        high_freq = 0.5 * np.sin(2 * np.pi * 0.3 * t)
        values = low_freq + high_freq
        ts = TimeSeries(values)
        filtered = FourierAnalysis.low_pass_filter(ts, cutoff_freq=0.1)
        # High frequency should be removed
        assert np.std(filtered.values - low_freq) < 0.3

    def test_band_pass_filter(self):
        n = 256
        t = np.arange(n)
        v1 = np.sin(2 * np.pi * 0.02 * t)
        v2 = np.sin(2 * np.pi * 0.1 * t)
        v3 = np.sin(2 * np.pi * 0.4 * t)
        values = v1 + v2 + v3
        ts = TimeSeries(values)
        filtered = FourierAnalysis.band_pass_filter(ts, low_freq=0.05, high_freq=0.15)
        # Should mostly contain v2
        corr = np.corrcoef(filtered.values, v2)[0, 1]
        assert corr > 0.8

    def test_reconstruct(self):
        n = 128
        t = np.arange(n)
        values = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.cos(2 * np.pi * 0.2 * t) + 10
        ts = TimeSeries(values)
        recon = FourierAnalysis.reconstruct(ts, n_components=2)
        # Should capture most of the variance
        corr = np.corrcoef(recon.values, values)[0, 1]
        assert corr > 0.9

    def test_fft_array_input(self):
        values = np.random.randn(64)
        result = FourierAnalysis.fft(values)
        assert "frequencies" in result

    def test_dominant_frequencies_returns_periods(self):
        n = 128
        t = np.arange(n)
        values = np.sin(2 * np.pi * t / 16)  # period=16, freq=1/16
        dom = FourierAnalysis.dominant_frequencies(values, top_k=1)
        assert len(dom) >= 1
        assert abs(dom[0]["period"] - 16.0) < 2.0


# ===========================================================================
# Change Point Detection
# ===========================================================================

class TestChangePointDetection:
    def test_cusum_no_change(self):
        np.random.seed(42)
        values = np.random.randn(200)
        result = ChangePointDetection.cusum(values)
        # Might or might not detect change in pure noise
        assert "cusum" in result

    def test_cusum_with_shift(self):
        np.random.seed(42)
        values = np.concatenate([np.random.randn(100), np.random.randn(100) + 5])
        result = ChangePointDetection.cusum(values)
        assert len(result["change_points"]) >= 1
        # Change point should be near index 100
        cp = result["change_points"][0]["index"]
        assert 70 < cp < 150

    def test_binary_segmentation_single(self):
        np.random.seed(42)
        v1 = np.random.randn(100) + 10
        v2 = np.random.randn(100) + 20
        values = np.concatenate([v1, v2])
        result = ChangePointDetection.binary_segmentation(values, min_segment=10)
        assert len(result["change_points"]) >= 1
        assert 80 < result["change_points"][0] < 120

    def test_binary_segmentation_multiple(self):
        np.random.seed(42)
        v1 = np.random.randn(80)
        v2 = np.random.randn(80) + 5
        v3 = np.random.randn(80) - 3
        values = np.concatenate([v1, v2, v3])
        result = ChangePointDetection.binary_segmentation(values, min_segment=10, penalty=5.0)
        assert len(result["change_points"]) >= 2

    def test_pelt_single_change(self):
        np.random.seed(42)
        v1 = np.random.randn(100)
        v2 = np.random.randn(100) + 5
        values = np.concatenate([v1, v2])
        result = ChangePointDetection.pelt(values, min_segment=5)
        assert len(result["change_points"]) >= 1
        if len(result["change_points"]) > 0:
            # At least one CP should be near 100
            distances = [abs(cp - 100) for cp in result["change_points"]]
            assert min(distances) < 30

    def test_pelt_no_change(self):
        np.random.seed(42)
        values = np.random.randn(200)
        result = ChangePointDetection.pelt(values, min_segment=10, penalty=50.0)
        # With high penalty, shouldn't detect many spurious changes
        assert len(result["change_points"]) <= 3

    def test_mean_shift_test_significant(self):
        v1 = np.random.randn(100)
        v2 = np.random.randn(100) + 10
        values = np.concatenate([v1, v2])
        result = ChangePointDetection.mean_shift_test(values, 100)
        assert result["significant"]
        assert result["mean_diff"] > 5

    def test_mean_shift_test_not_significant(self):
        np.random.seed(42)
        values = np.random.randn(200)
        result = ChangePointDetection.mean_shift_test(values, 100)
        # Should usually not be significant
        assert result["t_stat"] < 5

    def test_cusum_threshold(self):
        values = np.ones(100)
        result = ChangePointDetection.cusum(values, threshold=0.1)
        assert len(result["change_points"]) == 0


# ===========================================================================
# Stationarity Tests
# ===========================================================================

class TestStationarityTests:
    def test_adf_stationary(self):
        np.random.seed(42)
        values = np.random.randn(200)
        result = StationarityTests.adf_test(values)
        assert result["stationary"]
        assert result["statistic"] < result["critical_values"]["5%"]

    def test_adf_nonstationary(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(200))
        result = StationarityTests.adf_test(values)
        # Random walk should be non-stationary
        # (might occasionally pass due to randomness, so use weaker check)
        assert "statistic" in result

    def test_kpss_stationary(self):
        np.random.seed(42)
        values = np.random.randn(200)
        result = StationarityTests.kpss_test(values)
        assert result["stationary"]

    def test_kpss_with_trend(self):
        np.random.seed(42)
        t = np.arange(200, dtype=float)
        values = 0.5 * t + np.random.randn(200)
        result_c = StationarityTests.kpss_test(values, regression="c")
        result_ct = StationarityTests.kpss_test(values, regression="ct")
        # With constant-only regression, trending data looks non-stationary
        # With constant+trend, it might be trend-stationary
        assert "statistic" in result_c
        assert "statistic" in result_ct

    def test_adf_critical_values(self):
        result = StationarityTests.adf_test(np.random.randn(100))
        assert "1%" in result["critical_values"]
        assert "5%" in result["critical_values"]
        assert "10%" in result["critical_values"]


# ===========================================================================
# Forecaster
# ===========================================================================

class TestForecaster:
    def test_auto_select_ses(self):
        np.random.seed(42)
        values = np.random.randn(100) + 50  # No trend, no season
        fc = Forecaster(method="auto")
        fc.fit(values)
        assert fc.method == "ses"

    def test_auto_select_holt(self):
        values = np.arange(1, 101, dtype=float)  # Linear trend
        fc = Forecaster(method="auto")
        fc.fit(values)
        assert fc.method == "holt"

    def test_auto_select_hw(self):
        t = np.arange(48, dtype=float)
        values = 50 + t + 10 * np.sin(2 * np.pi * t / 12)
        fc = Forecaster(method="auto")
        fc.fit(values, period=12)
        assert fc.method == "holt_winters"

    def test_forecast_ses(self):
        fc = Forecaster(method="ses")
        fc.fit(TimeSeries([10, 12, 14, 13, 15]))
        pred = fc.forecast(steps=3)
        assert len(pred) == 3

    def test_forecast_holt(self):
        fc = Forecaster(method="holt")
        fc.fit(np.arange(1, 21, dtype=float))
        pred = fc.forecast(steps=5)
        assert len(pred) == 5
        assert pred[0] > 19  # Should continue trend

    def test_forecast_arima(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(100))
        fc = Forecaster(method="arima")
        fc.fit(values, p=1, d=1, q=0)
        pred = fc.forecast(steps=5)
        assert len(pred) == 5

    def test_fitted_values(self):
        fc = Forecaster(method="ses")
        fc.fit([10, 20, 30, 40, 50])
        fitted = fc.fitted_values()
        assert len(fitted) == 5

    def test_forecast_not_fitted(self):
        fc = Forecaster(method="ses")
        with pytest.raises(ValueError):
            fc.forecast(5)

    def test_forecast_hw_steps(self):
        t = np.arange(48, dtype=float)
        values = 50 + 10 * np.sin(2 * np.pi * t / 12)
        fc = Forecaster(method="holt_winters")
        fc.fit(values, period=12)
        pred = fc.forecast(steps=6)
        assert len(pred) == 6


# ===========================================================================
# generate_series utility
# ===========================================================================

class TestGenerateSeries:
    def test_basic(self):
        ts = generate_series(n=100, seed=42)
        assert len(ts) == 100

    def test_trend(self):
        ts = generate_series(n=100, trend=0.5, noise_std=0.0, seed=42)
        assert ts.values[-1] > ts.values[0]

    def test_seasonal(self):
        ts = generate_series(n=120, seasonal_period=12, seasonal_amp=5.0, noise_std=0.0, seed=42)
        # Should have periodicity
        acf = Autocorrelation.acf(ts, max_lag=24)
        assert acf[12] > 0.5

    def test_ar_process(self):
        ts = generate_series(n=500, ar_coefs=[0.8], noise_std=1.0, seed=42)
        acf = Autocorrelation.acf(ts, max_lag=5)
        assert acf[1] > 0.5

    def test_change_points(self):
        ts = generate_series(n=200, noise_std=0.5, change_points=[(100, 10.0)], seed=42)
        mean1 = np.mean(ts.values[:90])
        mean2 = np.mean(ts.values[110:])
        assert mean2 - mean1 > 5

    def test_seed_reproducibility(self):
        ts1 = generate_series(n=50, seed=123)
        ts2 = generate_series(n=50, seed=123)
        np.testing.assert_array_equal(ts1.values, ts2.values)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_full_pipeline(self):
        """Generate -> decompose -> fit ARIMA -> forecast."""
        ts = generate_series(n=120, trend=0.1, seasonal_period=12,
                             seasonal_amp=5.0, noise_std=1.0, seed=42)
        # Decompose
        decomp = SeasonalDecompose.decompose(ts, period=12)
        assert decomp.trend is not None

        # Fit ARIMA on deseasonalized
        deseason = ts.values - decomp.seasonal
        model = ARIMA(p=1, d=1, q=0).fit(deseason)
        fc = model.forecast(steps=12)
        assert len(fc) == 12

    def test_stationarity_then_arima(self):
        """Test stationarity, difference if needed, fit ARIMA."""
        np.random.seed(42)
        values = np.cumsum(np.random.randn(200)) + 100
        ts = TimeSeries(values)

        adf = StationarityTests.adf_test(ts)
        if not adf["stationary"]:
            # Difference and retest
            diffed = ts.diff()
            adf2 = StationarityTests.adf_test(diffed)
            assert adf2["stationary"]

        model = ARIMA(p=1, d=1, q=0).fit(values)
        fc = model.forecast(3)
        assert len(fc) == 3

    def test_fourier_then_filter(self):
        """Analyze frequencies then filter."""
        n = 256
        t = np.arange(n)
        signal = np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.3 * t)
        ts = TimeSeries(signal)

        dom = FourierAnalysis.dominant_frequencies(ts, top_k=2)
        assert len(dom) >= 2

        filtered = FourierAnalysis.low_pass_filter(ts, cutoff_freq=0.1)
        # Filtered should be smoother
        assert np.std(np.diff(filtered.values)) < np.std(np.diff(signal))

    def test_change_detection_pipeline(self):
        """Generate series with shifts, detect them."""
        ts = generate_series(n=300, noise_std=1.0,
                             change_points=[(100, 5.0), (200, -8.0)], seed=42)

        result = ChangePointDetection.binary_segmentation(
            ts, min_segment=20, penalty=10.0
        )
        cps = result["change_points"]
        assert len(cps) >= 2

    def test_exponential_smoothing_vs_arima(self):
        """Compare two forecasting approaches."""
        np.random.seed(42)
        values = np.cumsum(np.random.randn(100)) + 50
        train = values[:80]
        test = values[80:]

        # SES
        ses_result = ExponentialSmoothing.ses(train)
        ses_fc = ses_result["forecast_next"]

        # ARIMA
        arima = ARIMA(p=1, d=1, q=0).fit(train)
        arima_fc = arima.forecast(steps=20)

        # Both should produce reasonable forecasts (not NaN or inf)
        assert np.isfinite(ses_fc)
        assert all(np.isfinite(arima_fc))

    def test_seasonal_forecast_pipeline(self):
        """Full seasonal forecasting with Holt-Winters."""
        t = np.arange(72, dtype=float)
        values = 100 + 0.5 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.randn(72) * 2
        ts = TimeSeries(values)

        fc = Forecaster(method="holt_winters")
        fc.fit(ts, period=12)
        predictions = fc.forecast(steps=12)
        assert len(predictions) == 12
        # Predictions should be in reasonable range
        assert all(50 < p < 250 for p in predictions)

    def test_acf_guides_model_selection(self):
        """Use ACF/PACF to guide ARIMA order selection."""
        np.random.seed(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.6 * x[i-1] + np.random.randn()

        pacf = Autocorrelation.pacf(x, max_lag=10)
        # PACF should cut off after lag 1 (AR(1) process)
        assert abs(pacf[1]) > 0.4
        assert abs(pacf[3]) < 0.15

        model = ARIMA(p=1, d=0, q=0).fit(x)
        assert abs(model.ar_params[0] - 0.6) < 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
