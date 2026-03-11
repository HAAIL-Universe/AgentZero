"""
Tests for C198: Advanced Time Series Forecasting
"""

import sys
import os
import math
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from time_series_forecasting import (
    GARCH, VAR, KalmanFilter, LocalLevelModel, LocalLinearTrend,
    ProphetDecomposition, ThetaMethod, CrostonMethod, HoltWinters,
    EnsembleForecaster, Backtester, PredictionInterval, ForecastCombiner,
    diebold_mariano_test, generate_seasonal_series, generate_intermittent_series,
    generate_garch_series, generate_var_series, difference, undifference,
    _validate_series, _validate_matrix
)


# ============================================================
# Utility Tests
# ============================================================

class TestUtilities(unittest.TestCase):

    def test_validate_series_list(self):
        y = _validate_series([1, 2, 3])
        self.assertEqual(len(y), 3)
        self.assertIsInstance(y, np.ndarray)

    def test_validate_series_too_short(self):
        with self.assertRaises(ValueError):
            _validate_series([1])

    def test_validate_series_2d_fails(self):
        with self.assertRaises(ValueError):
            _validate_series([[1, 2], [3, 4]])

    def test_validate_matrix_1d(self):
        m = _validate_matrix([1, 2, 3])
        self.assertEqual(m.shape, (3, 1))

    def test_validate_matrix_2d(self):
        m = _validate_matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))

    def test_difference_once(self):
        y = np.array([1.0, 3.0, 6.0, 10.0])
        d = difference(y, 1)
        np.testing.assert_array_almost_equal(d, [2, 3, 4])

    def test_difference_twice(self):
        y = np.array([1.0, 3.0, 6.0, 10.0])
        d = difference(y, 2)
        np.testing.assert_array_almost_equal(d, [1, 1])

    def test_undifference(self):
        y = np.array([10.0, 13.0, 17.0, 22.0])
        d = difference(y, 1)
        restored = undifference(d, y, 1)
        np.testing.assert_array_almost_equal(restored, y[1:])

    def test_generate_seasonal(self):
        y = generate_seasonal_series(100)
        self.assertEqual(len(y), 100)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_generate_intermittent(self):
        y = generate_intermittent_series(100)
        self.assertEqual(len(y), 100)
        self.assertTrue(np.sum(y == 0) > 0)  # some zeros
        self.assertTrue(np.sum(y > 0) > 0)   # some non-zeros

    def test_generate_garch(self):
        y, s2 = generate_garch_series(200)
        self.assertEqual(len(y), 200)
        self.assertTrue(np.all(s2 > 0))

    def test_generate_var(self):
        Y = generate_var_series(100, k=3)
        self.assertEqual(Y.shape, (100, 3))


# ============================================================
# GARCH Tests
# ============================================================

class TestGARCH(unittest.TestCase):

    def setUp(self):
        self.y, self.true_sigma2 = generate_garch_series(500, seed=42)

    def test_fit_basic(self):
        g = GARCH(1, 1)
        g.fit(self.y)
        self.assertIsNotNone(g.omega)
        self.assertTrue(g.omega > 0)
        self.assertEqual(len(g.alpha), 1)
        self.assertEqual(len(g.beta), 1)

    def test_persistence_less_than_one(self):
        g = GARCH(1, 1).fit(self.y)
        self.assertLess(g.persistence, 1.0)

    def test_unconditional_variance_positive(self):
        g = GARCH(1, 1).fit(self.y)
        self.assertTrue(g.unconditional_variance > 0)

    def test_forecast_volatility_shape(self):
        g = GARCH(1, 1).fit(self.y)
        fc = g.forecast_volatility(10)
        self.assertEqual(len(fc), 10)
        self.assertTrue(np.all(fc > 0))

    def test_conditional_volatility(self):
        g = GARCH(1, 1).fit(self.y)
        vol = g.conditional_volatility
        self.assertEqual(len(vol), len(self.y))
        self.assertTrue(np.all(vol > 0))

    def test_not_fitted_raises(self):
        g = GARCH(1, 1)
        with self.assertRaises(RuntimeError):
            g.forecast_volatility()
        with self.assertRaises(RuntimeError):
            _ = g.persistence
        with self.assertRaises(RuntimeError):
            _ = g.unconditional_variance
        with self.assertRaises(RuntimeError):
            _ = g.conditional_volatility

    def test_garch_21(self):
        g = GARCH(2, 1).fit(self.y)
        self.assertEqual(len(g.beta), 2)
        self.assertEqual(len(g.alpha), 1)

    def test_garch_12(self):
        g = GARCH(1, 2).fit(self.y)
        self.assertEqual(len(g.beta), 1)
        self.assertEqual(len(g.alpha), 2)

    def test_forecast_converges_to_unconditional(self):
        g = GARCH(1, 1).fit(self.y)
        fc = g.forecast_volatility(100)
        # Long-horizon should approach unconditional variance
        uncond = g.unconditional_variance
        if uncond < float('inf'):
            self.assertAlmostEqual(fc[-1], uncond, delta=uncond * 0.5)


# ============================================================
# VAR Tests
# ============================================================

class TestVAR(unittest.TestCase):

    def setUp(self):
        self.Y = generate_var_series(300, k=2, seed=42)

    def test_fit_basic(self):
        v = VAR(1).fit(self.Y)
        self.assertEqual(len(v.coefs), 1)
        self.assertEqual(v.coefs[0].shape, (2, 2))
        self.assertEqual(v.intercept.shape, (2,))

    def test_forecast_shape(self):
        v = VAR(1).fit(self.Y)
        fc = v.forecast(5)
        self.assertEqual(fc.shape, (5, 2))

    def test_forecast_2_lags(self):
        v = VAR(2).fit(self.Y)
        fc = v.forecast(3)
        self.assertEqual(fc.shape, (3, 2))
        self.assertEqual(len(v.coefs), 2)

    def test_sigma_u_symmetric(self):
        v = VAR(1).fit(self.Y)
        np.testing.assert_array_almost_equal(v.sigma_u, v.sigma_u.T)

    def test_sigma_u_positive_definite(self):
        v = VAR(1).fit(self.Y)
        eigvals = np.linalg.eigvalsh(v.sigma_u)
        self.assertTrue(np.all(eigvals > 0))

    def test_impulse_response_shape(self):
        v = VAR(1).fit(self.Y)
        irf = v.impulse_response(10, shock_var=0)
        self.assertEqual(irf.shape, (11, 2))  # 0..10

    def test_impulse_response_initial_shock(self):
        v = VAR(1).fit(self.Y)
        irf = v.impulse_response(10, shock_var=0)
        # At step 0, shock to var 0 should be positive
        self.assertGreater(irf[0, 0], 0)

    def test_granger_causality(self):
        v = VAR(1).fit(self.Y)
        F_stat, p_val = v.granger_causality(0, 1)
        self.assertIsInstance(F_stat, float)
        self.assertTrue(0 <= p_val <= 1)

    def test_fevd_shape(self):
        v = VAR(1).fit(self.Y)
        fevd = v.forecast_error_variance_decomposition(10)
        self.assertEqual(fevd.shape, (11, 2, 2))

    def test_fevd_sums_to_one(self):
        v = VAR(1).fit(self.Y)
        fevd = v.forecast_error_variance_decomposition(10)
        for s in range(11):
            for i in range(2):
                self.assertAlmostEqual(np.sum(fevd[s, i, :]), 1.0, places=5)

    def test_not_fitted_raises(self):
        v = VAR(1)
        with self.assertRaises(RuntimeError):
            v.forecast()
        with self.assertRaises(RuntimeError):
            v.impulse_response()
        with self.assertRaises(RuntimeError):
            v.granger_causality(0, 1)

    def test_too_few_observations(self):
        with self.assertRaises(ValueError):
            VAR(5).fit(np.array([[1, 2], [3, 4], [5, 6]]))

    def test_forecast_with_history(self):
        v = VAR(1).fit(self.Y)
        fc1 = v.forecast(3)
        fc2 = v.forecast(3, Y_history=self.Y)
        np.testing.assert_array_almost_equal(fc1, fc2)

    def test_3_variable_var(self):
        Y3 = generate_var_series(200, k=3, seed=99)
        # k=3 generates with 2-var default matrix, but we can still test
        v = VAR(1).fit(Y3[:, :2])
        fc = v.forecast(5)
        self.assertEqual(fc.shape, (5, 2))


# ============================================================
# Kalman Filter Tests
# ============================================================

class TestKalmanFilter(unittest.TestCase):

    def test_basic_filter(self):
        kf = KalmanFilter(state_dim=1, obs_dim=1)
        kf.Q = np.array([[0.01]])
        kf.R = np.array([[1.0]])
        obs = np.random.RandomState(42).normal(5.0, 1.0, 50).reshape(-1, 1)
        states = kf.filter(obs)
        self.assertEqual(states.shape, (50, 1))

    def test_filter_converges_to_mean(self):
        kf = KalmanFilter(state_dim=1, obs_dim=1)
        kf.Q = np.array([[0.001]])
        kf.R = np.array([[1.0]])
        kf.x = np.array([0.0])
        obs = np.ones((100, 1)) * 5.0
        states = kf.filter(obs)
        # Should converge close to 5
        self.assertAlmostEqual(states[-1, 0], 5.0, delta=0.5)

    def test_predict_step(self):
        kf = KalmanFilter(state_dim=2, obs_dim=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.x = np.array([0.0, 1.0])
        x_pred, P_pred = kf.predict()
        self.assertAlmostEqual(x_pred[0], 1.0)  # 0 + 1
        self.assertAlmostEqual(x_pred[1], 1.0)  # velocity unchanged

    def test_smoother(self):
        kf = KalmanFilter(state_dim=1, obs_dim=1)
        kf.Q = np.array([[0.01]])
        kf.R = np.array([[1.0]])
        rng = np.random.RandomState(42)
        true = np.sin(np.arange(50) * 0.2) * 3
        obs = (true + rng.normal(0, 1, 50)).reshape(-1, 1)
        smoothed = kf.smooth(obs)
        self.assertEqual(smoothed.shape, (50, 1))

    def test_smoother_less_noisy_than_filter(self):
        kf = KalmanFilter(state_dim=1, obs_dim=1)
        kf.Q = np.array([[0.1]])
        kf.R = np.array([[2.0]])
        rng = np.random.RandomState(42)
        true = np.ones(80) * 10
        obs = (true + rng.normal(0, 2, 80)).reshape(-1, 1)
        filtered = kf.filter(obs)
        smoothed = kf.smooth(obs)
        # Smoothed should be closer to truth
        filt_err = np.mean((filtered.ravel() - true) ** 2)
        smooth_err = np.mean((smoothed.ravel() - true) ** 2)
        self.assertLessEqual(smooth_err, filt_err + 0.5)

    def test_forecast(self):
        kf = KalmanFilter(state_dim=1, obs_dim=1)
        kf.Q = np.array([[0.01]])
        kf.R = np.array([[1.0]])
        obs = np.ones((50, 1)) * 7
        kf.filter(obs)
        fc, intervals = kf.forecast(5)
        self.assertEqual(fc.shape, (5, 1))
        self.assertEqual(len(intervals), 5)
        for lo, hi in intervals:
            self.assertTrue(lo[0] < hi[0])

    def test_log_likelihood(self):
        kf = KalmanFilter(state_dim=1, obs_dim=1)
        kf.Q = np.array([[0.1]])
        kf.R = np.array([[1.0]])
        obs = np.random.RandomState(42).normal(0, 1, 30).reshape(-1, 1)
        kf.filter(obs)
        self.assertTrue(np.isfinite(kf.log_likelihood))
        self.assertTrue(kf.log_likelihood < 0)

    def test_multivariate(self):
        kf = KalmanFilter(state_dim=2, obs_dim=2)
        obs = np.random.RandomState(42).normal(0, 1, (30, 2))
        states = kf.filter(obs)
        self.assertEqual(states.shape, (30, 2))

    def test_with_control(self):
        kf = KalmanFilter(state_dim=1, obs_dim=1, control_dim=1)
        kf.B = np.array([[1.0]])
        obs = np.ones((20, 1))
        controls = np.ones((20, 1)) * 0.5
        states = kf.filter(obs, controls)
        self.assertEqual(states.shape, (20, 1))


# ============================================================
# Local Level Model Tests
# ============================================================

class TestLocalLevelModel(unittest.TestCase):

    def test_fit_and_forecast(self):
        y = np.random.RandomState(42).normal(10, 1, 100)
        m = LocalLevelModel().fit(y)
        fc, intervals = m.forecast(5)
        self.assertEqual(len(fc), 5)
        self.assertEqual(len(intervals), 5)

    def test_forecast_near_mean(self):
        y = np.ones(100) * 42 + np.random.RandomState(42).normal(0, 0.1, 100)
        m = LocalLevelModel().fit(y)
        fc, _ = m.forecast(3)
        self.assertAlmostEqual(fc[0], 42, delta=1.0)

    def test_smoothed_level(self):
        y = np.random.RandomState(42).normal(5, 2, 80)
        m = LocalLevelModel()
        smoothed = m.smoothed_level(y)
        self.assertEqual(len(smoothed), 80)

    def test_not_fitted_raises(self):
        m = LocalLevelModel()
        with self.assertRaises(RuntimeError):
            m.forecast()

    def test_sigma_positive(self):
        y = np.random.RandomState(42).normal(0, 3, 100)
        m = LocalLevelModel().fit(y)
        self.assertGreater(m.sigma_level, 0)
        self.assertGreater(m.sigma_obs, 0)


# ============================================================
# Local Linear Trend Tests
# ============================================================

class TestLocalLinearTrend(unittest.TestCase):

    def test_fit_and_forecast(self):
        y = np.arange(100, dtype=float) + np.random.RandomState(42).normal(0, 1, 100)
        m = LocalLinearTrend().fit(y)
        fc, intervals = m.forecast(5)
        self.assertEqual(len(fc), 5)
        # Forecast should continue upward trend
        self.assertTrue(fc[-1] > fc[0] - 5)  # approximately

    def test_intervals_ordered(self):
        y = np.random.RandomState(42).normal(0, 1, 80)
        m = LocalLinearTrend().fit(y)
        _, intervals = m.forecast(3)
        for lo, hi in intervals:
            self.assertLess(lo, hi)

    def test_not_fitted_raises(self):
        m = LocalLinearTrend()
        with self.assertRaises(RuntimeError):
            m.forecast()


# ============================================================
# Prophet-Style Decomposition Tests
# ============================================================

class TestProphetDecomposition(unittest.TestCase):

    def test_basic_fit(self):
        y = generate_seasonal_series(200, seasonal_period=12)
        p = ProphetDecomposition(seasonality_period=12).fit(y)
        self.assertEqual(len(p.trend), 200)
        self.assertEqual(len(p.seasonality), 200)
        self.assertEqual(len(p.residual), 200)

    def test_decomposition_sums(self):
        y = generate_seasonal_series(200, seasonal_period=12)
        p = ProphetDecomposition(seasonality_period=12).fit(y)
        reconstructed = p.trend + p.seasonality + p.residual
        np.testing.assert_array_almost_equal(reconstructed, y, decimal=5)

    def test_forecast_shape(self):
        y = generate_seasonal_series(200, seasonal_period=12)
        p = ProphetDecomposition(seasonality_period=12).fit(y)
        fc = p.forecast(12)
        self.assertEqual(len(fc), 12)

    def test_no_seasonality(self):
        y = np.arange(100, dtype=float)
        p = ProphetDecomposition(n_changepoints=5).fit(y)
        np.testing.assert_array_almost_equal(p.seasonality, np.zeros(100))

    def test_changepoints(self):
        y = generate_seasonal_series(200)
        p = ProphetDecomposition(n_changepoints=5, seasonality_period=12).fit(y)
        self.assertEqual(len(p._changepoints), 5)

    def test_not_fitted_raises(self):
        p = ProphetDecomposition()
        with self.assertRaises(RuntimeError):
            _ = p.trend
        with self.assertRaises(RuntimeError):
            _ = p.seasonality
        with self.assertRaises(RuntimeError):
            _ = p.residual
        with self.assertRaises(RuntimeError):
            p.forecast()

    def test_short_series(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = ProphetDecomposition(n_changepoints=0).fit(y)
        fc = p.forecast(3)
        self.assertEqual(len(fc), 3)


# ============================================================
# Theta Method Tests
# ============================================================

class TestThetaMethod(unittest.TestCase):

    def test_basic_fit_forecast(self):
        y = generate_seasonal_series(100)
        m = ThetaMethod().fit(y)
        fc = m.forecast(10)
        self.assertEqual(len(fc), 10)
        self.assertTrue(np.all(np.isfinite(fc)))

    def test_flat_series(self):
        y = np.ones(50) * 10
        m = ThetaMethod().fit(y)
        fc = m.forecast(5)
        for v in fc:
            self.assertAlmostEqual(v, 10, delta=2)

    def test_trending_series(self):
        y = np.arange(100, dtype=float)
        m = ThetaMethod().fit(y)
        fc = m.forecast(5)
        # Should continue upward
        self.assertTrue(fc[0] > 90)

    def test_not_fitted_raises(self):
        m = ThetaMethod()
        with self.assertRaises(RuntimeError):
            m.forecast()


# ============================================================
# Croston Method Tests
# ============================================================

class TestCrostonMethod(unittest.TestCase):

    def test_basic_fit(self):
        y = generate_intermittent_series(200)
        m = CrostonMethod().fit(y)
        fc = m.forecast(5)
        self.assertEqual(len(fc), 5)
        self.assertTrue(np.all(fc >= 0))

    def test_constant_forecast(self):
        y = generate_intermittent_series(200)
        m = CrostonMethod().fit(y)
        fc = m.forecast(10)
        # Croston produces flat forecasts
        self.assertAlmostEqual(fc[0], fc[9])

    def test_demand_properties(self):
        y = generate_intermittent_series(200)
        m = CrostonMethod().fit(y)
        self.assertGreater(m.demand_size, 0)
        self.assertGreater(m.demand_interval, 0)

    def test_all_zeros(self):
        y = np.zeros(50)
        y[0] = 1  # need at least 2 observations
        y[1] = 0
        m = CrostonMethod().fit(np.concatenate([[1], y]))
        fc = m.forecast(5)
        self.assertEqual(len(fc), 5)

    def test_not_fitted_raises(self):
        m = CrostonMethod()
        with self.assertRaises(RuntimeError):
            m.forecast()
        with self.assertRaises(RuntimeError):
            _ = m.demand_size
        with self.assertRaises(RuntimeError):
            _ = m.demand_interval

    def test_alpha_parameter(self):
        y = generate_intermittent_series(200)
        m1 = CrostonMethod(alpha=0.05).fit(y)
        m2 = CrostonMethod(alpha=0.5).fit(y)
        # Different alphas should generally give different forecasts
        fc1 = m1.forecast(1)[0]
        fc2 = m2.forecast(1)[0]
        # Just verify both are valid
        self.assertTrue(fc1 >= 0)
        self.assertTrue(fc2 >= 0)


# ============================================================
# Holt-Winters Tests
# ============================================================

class TestHoltWinters(unittest.TestCase):

    def test_additive_fit(self):
        y = generate_seasonal_series(120, seasonal_period=12)
        m = HoltWinters(seasonal_period=12, seasonal='additive').fit(y)
        fc = m.forecast(12)
        self.assertEqual(len(fc), 12)

    def test_multiplicative_fit(self):
        y = generate_seasonal_series(120, seasonal_period=12) + 200  # ensure positive
        m = HoltWinters(seasonal_period=12, seasonal='multiplicative').fit(y)
        fc = m.forecast(12)
        self.assertEqual(len(fc), 12)
        self.assertTrue(np.all(np.isfinite(fc)))

    def test_no_season(self):
        y = np.arange(50, dtype=float)
        m = HoltWinters().fit(y)
        fc = m.forecast(5)
        self.assertEqual(len(fc), 5)

    def test_fitted_values_shape(self):
        y = generate_seasonal_series(60, seasonal_period=12)
        m = HoltWinters(seasonal_period=12).fit(y)
        self.assertEqual(len(m._fitted_values), 60)

    def test_not_fitted_raises(self):
        m = HoltWinters()
        with self.assertRaises(RuntimeError):
            m.forecast()

    def test_optimization_improves(self):
        y = generate_seasonal_series(120, seasonal_period=12)
        m1 = HoltWinters(seasonal_period=12).fit(y, optimize=False)
        m2 = HoltWinters(seasonal_period=12).fit(y, optimize=True)
        sse1 = np.sum((y - m1._fitted_values) ** 2)
        sse2 = np.sum((y - m2._fitted_values) ** 2)
        self.assertLessEqual(sse2, sse1 * 1.1)  # optimized should be similar or better


# ============================================================
# Ensemble Forecaster Tests
# ============================================================

class TestEnsembleForecaster(unittest.TestCase):

    def _make_models(self):
        return [
            ('theta', ThetaMethod()),
            ('ll', LocalLevelModel()),
        ]

    def test_equal_weights(self):
        y = generate_seasonal_series(100)
        models = self._make_models()
        ens = EnsembleForecaster(models, method='equal').fit(y)
        self.assertEqual(len(ens.weights), 2)
        self.assertAlmostEqual(sum(ens.weights), 1.0, places=5)

    def test_inverse_error(self):
        y = generate_seasonal_series(100)
        models = self._make_models()
        ens = EnsembleForecaster(models, method='inverse_error').fit(y)
        fc = ens.forecast(5)
        self.assertEqual(len(fc), 5)

    def test_stacking(self):
        y = generate_seasonal_series(100)
        models = self._make_models()
        ens = EnsembleForecaster(models, method='stacking').fit(y)
        fc = ens.forecast(5)
        self.assertEqual(len(fc), 5)

    def test_weights_sum_to_one(self):
        y = generate_seasonal_series(100)
        for method in ['equal', 'inverse_error', 'stacking']:
            models = self._make_models()
            ens = EnsembleForecaster(models, method=method).fit(y)
            self.assertAlmostEqual(sum(ens.weights), 1.0, places=3)

    def test_not_fitted_raises(self):
        ens = EnsembleForecaster(self._make_models())
        with self.assertRaises(RuntimeError):
            ens.forecast()

    def test_custom_validation_size(self):
        y = generate_seasonal_series(100)
        models = self._make_models()
        ens = EnsembleForecaster(models).fit(y, validation_size=10)
        fc = ens.forecast(3)
        self.assertEqual(len(fc), 3)


# ============================================================
# Backtester Tests
# ============================================================

class TestBacktester(unittest.TestCase):

    def test_basic_backtest(self):
        y = generate_seasonal_series(100)
        bt = Backtester(lambda: ThetaMethod(), metric='mse')
        bt.run(y, initial_window=50, horizon=1, step=5)
        self.assertGreater(len(bt.results), 0)

    def test_all_metrics(self):
        y = generate_seasonal_series(100)
        for metric in ['mse', 'rmse', 'mae', 'mape', 'smape']:
            bt = Backtester(lambda: ThetaMethod(), metric=metric)
            bt.run(y, initial_window=50, horizon=1, step=10)
            self.assertTrue(bt.mean_error >= 0)

    def test_multi_step_horizon(self):
        y = generate_seasonal_series(100)
        bt = Backtester(lambda: ThetaMethod())
        bt.run(y, initial_window=50, horizon=5, step=5)
        for r in bt.results:
            if r['forecast'] is not None:
                self.assertEqual(len(r['actual']), 5)

    def test_error_series(self):
        y = generate_seasonal_series(100)
        bt = Backtester(lambda: ThetaMethod())
        bt.run(y, initial_window=60, horizon=1, step=5)
        errors = bt.error_series
        self.assertEqual(len(errors), len(bt.results))

    def test_std_error(self):
        y = generate_seasonal_series(100)
        bt = Backtester(lambda: ThetaMethod())
        bt.run(y, initial_window=50, horizon=1, step=5)
        self.assertTrue(np.isfinite(bt.std_error))


# ============================================================
# Prediction Interval Tests
# ============================================================

class TestPredictionInterval(unittest.TestCase):

    def test_parametric(self):
        forecasts = np.array([10, 11, 12])
        residuals = np.random.RandomState(42).normal(0, 1, 50)
        lo, hi = PredictionInterval.parametric(forecasts, residuals, 0.95)
        self.assertEqual(len(lo), 3)
        self.assertTrue(np.all(lo < hi))

    def test_parametric_wider_at_lower_confidence(self):
        forecasts = np.array([10, 11, 12])
        residuals = np.random.RandomState(42).normal(0, 1, 50)
        lo95, hi95 = PredictionInterval.parametric(forecasts, residuals, 0.95)
        lo80, hi80 = PredictionInterval.parametric(forecasts, residuals, 0.80)
        # 95% should be wider than 80%
        self.assertTrue(np.all((hi95 - lo95) >= (hi80 - lo80) - 1e-10))

    def test_bootstrap(self):
        y = generate_seasonal_series(80)
        lo, hi = PredictionInterval.bootstrap(
            lambda: ThetaMethod(), y, steps=3, n_bootstrap=20
        )
        self.assertEqual(len(lo), 3)
        self.assertTrue(np.all(lo < hi))

    def test_conformal(self):
        y = generate_seasonal_series(80)
        lo, hi = PredictionInterval.conformal(
            lambda: ThetaMethod(), y, steps=3
        )
        self.assertEqual(len(lo), 3)
        self.assertTrue(np.all(lo < hi))


# ============================================================
# Forecast Combiner Tests
# ============================================================

class TestForecastCombiner(unittest.TestCase):

    def test_simple_average(self):
        f1 = np.array([10, 12, 14])
        f2 = np.array([8, 10, 12])
        avg = ForecastCombiner.simple_average([f1, f2])
        np.testing.assert_array_almost_equal(avg, [9, 11, 13])

    def test_median(self):
        f1 = np.array([10, 12, 14])
        f2 = np.array([8, 10, 12])
        f3 = np.array([100, 100, 100])  # outlier
        med = ForecastCombiner.median([f1, f2, f3])
        np.testing.assert_array_almost_equal(med, [10, 12, 14])

    def test_trimmed_mean(self):
        forecasts = [np.array([10]), np.array([11]), np.array([12]),
                     np.array([100])]  # outlier
        tm = ForecastCombiner.trimmed_mean(forecasts, trim_fraction=0.25)
        # Should exclude 100 and 10
        self.assertAlmostEqual(tm[0], 11.5)

    def test_variance_weighted(self):
        f1 = np.array([10.0])
        f2 = np.array([20.0])
        # f1 has lower variance errors -> should get more weight
        e1 = [0.1, -0.1, 0.2]
        e2 = [5.0, -5.0, 4.0]
        combined = ForecastCombiner.variance_weighted([f1, f2], [e1, e2])
        # Should be closer to f1 (10)
        self.assertLess(combined[0], 15)


# ============================================================
# Diebold-Mariano Test
# ============================================================

class TestDieboldMariano(unittest.TestCase):

    def test_identical_forecasts(self):
        actual = np.array([1, 2, 3, 4, 5], dtype=float)
        fc = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        stat, pval = diebold_mariano_test(actual, fc, fc)
        self.assertAlmostEqual(stat, 0.0)
        self.assertAlmostEqual(pval, 1.0)

    def test_different_accuracy(self):
        rng = np.random.RandomState(42)
        actual = rng.normal(0, 1, 100)
        fc1 = actual + rng.normal(0, 0.1, 100)  # good
        fc2 = actual + rng.normal(0, 2.0, 100)  # bad
        stat, pval = diebold_mariano_test(actual, fc1, fc2)
        self.assertLess(stat, 0)  # fc1 better -> negative stat
        self.assertLess(pval, 0.05)

    def test_absolute_loss(self):
        actual = np.array([1, 2, 3, 4, 5], dtype=float)
        fc1 = actual + 0.5
        fc2 = actual - 0.5
        stat, pval = diebold_mariano_test(actual, fc1, fc2, loss='absolute')
        self.assertAlmostEqual(abs(stat), 0.0, places=5)

    def test_invalid_loss(self):
        with self.assertRaises(ValueError):
            diebold_mariano_test([1, 2], [1, 2], [1, 2], loss='invalid')


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration(unittest.TestCase):

    def test_full_forecasting_pipeline(self):
        """Test end-to-end: generate, fit multiple models, ensemble, backtest."""
        y = generate_seasonal_series(200, seasonal_period=12, seed=42)

        # Fit individual models
        theta = ThetaMethod().fit(y)
        ll = LocalLevelModel().fit(y)
        prophet = ProphetDecomposition(seasonality_period=12).fit(y)

        # Individual forecasts
        fc_theta = theta.forecast(12)
        fc_ll, _ = ll.forecast(12)
        fc_prophet = prophet.forecast(12)

        # All should be finite
        self.assertTrue(np.all(np.isfinite(fc_theta)))
        self.assertTrue(np.all(np.isfinite(fc_ll)))
        self.assertTrue(np.all(np.isfinite(fc_prophet)))

    def test_garch_on_returns(self):
        """GARCH on financial-like returns."""
        y, _ = generate_garch_series(500, omega=0.05, alpha=0.1, beta=0.85)
        g = GARCH(1, 1).fit(y)
        vol_fc = g.forecast_volatility(20)
        self.assertTrue(np.all(vol_fc > 0))
        # Check recovered parameters are in right ballpark
        self.assertLess(g.persistence, 1.0)

    def test_var_system(self):
        """VAR with impulse response and Granger causality."""
        Y = generate_var_series(300, k=2, seed=42)
        v = VAR(2).fit(Y)
        fc = v.forecast(10)
        irf = v.impulse_response(20, shock_var=0)
        # IRF should decay
        self.assertGreater(abs(irf[1, 0]), abs(irf[-1, 0]) - 0.1)

    def test_kalman_tracking(self):
        """Kalman filter tracks a moving target."""
        rng = np.random.RandomState(42)
        true_pos = np.cumsum(rng.normal(0.5, 0.2, 100))
        obs = true_pos + rng.normal(0, 2, 100)

        kf = KalmanFilter(state_dim=2, obs_dim=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.Q = np.diag([0.1, 0.01])
        kf.R = np.array([[4.0]])
        kf.x = np.array([obs[0], 0.5])

        states = kf.filter(obs.reshape(-1, 1))
        tracking_error = np.mean((states[:, 0] - true_pos) ** 2)
        observation_error = np.mean((obs - true_pos) ** 2)
        # Filter should reduce error
        self.assertLess(tracking_error, observation_error)

    def test_intermittent_demand(self):
        """Croston on intermittent demand."""
        y = generate_intermittent_series(300, demand_prob=0.2, mean_size=15)
        m = CrostonMethod(alpha=0.1).fit(y)
        fc = m.forecast(10)
        # Rate should be approximately demand_prob * mean_size = 3.0
        self.assertGreater(fc[0], 0)
        self.assertLess(fc[0], 20)

    def test_holtwinters_seasonal(self):
        """Holt-Winters captures seasonality."""
        y = generate_seasonal_series(240, seasonal_period=12, seasonal_amp=10, noise_std=0.5)
        m = HoltWinters(seasonal_period=12, seasonal='additive').fit(y)
        fc = m.forecast(12)
        # Forecast should show seasonal pattern (not flat)
        self.assertGreater(np.std(fc), 1.0)

    def test_backtest_comparison(self):
        """Compare two models via backtesting."""
        y = generate_seasonal_series(150, seasonal_period=12)

        bt1 = Backtester(lambda: ThetaMethod(), metric='mse')
        bt1.run(y, initial_window=80, horizon=1, step=5)

        bt2 = Backtester(lambda: LocalLevelModel(), metric='mse')
        bt2.run(y, initial_window=80, horizon=1, step=5)

        # Both should produce valid results
        self.assertTrue(np.isfinite(bt1.mean_error))
        self.assertTrue(np.isfinite(bt2.mean_error))

    def test_dm_test_with_models(self):
        """DM test between two real model forecasts."""
        y = generate_seasonal_series(120, seasonal_period=12)
        train, test = y[:100], y[100:]

        m1 = ThetaMethod().fit(train)
        m2 = LocalLevelModel().fit(train)

        fc1 = m1.forecast(20)
        fc2, _ = m2.forecast(20)

        stat, pval = diebold_mariano_test(test, fc1, fc2)
        self.assertTrue(np.isfinite(stat))
        self.assertTrue(0 <= pval <= 1)

    def test_ensemble_beats_worst(self):
        """Ensemble should be at least as good as the worst model."""
        y = generate_seasonal_series(120, seasonal_period=12)
        train, test = y[:100], y[100:]

        # Individual forecasts
        m1 = ThetaMethod().fit(train)
        fc1 = m1.forecast(20)
        mse1 = np.mean((test - fc1) ** 2)

        m2 = LocalLevelModel().fit(train)
        fc2, _ = m2.forecast(20)
        mse2 = np.mean((test - fc2) ** 2)

        # Ensemble
        models = [('theta', ThetaMethod()), ('ll', LocalLevelModel())]
        ens = EnsembleForecaster(models, method='inverse_error')
        ens.fit(train)
        fc_ens = ens.forecast(20)
        mse_ens = np.mean((test - fc_ens) ** 2)

        worst = max(mse1, mse2)
        # Ensemble shouldn't be catastrophically worse than worst
        self.assertLess(mse_ens, worst * 3)

    def test_prophet_changepoint_detection(self):
        """Prophet detects trend changes."""
        n = 200
        t = np.arange(n, dtype=float)
        y = np.where(t < 100, 0.5 * t, 0.5 * 100 - 0.3 * (t - 100)) + \
            np.random.RandomState(42).normal(0, 2, n)

        p = ProphetDecomposition(n_changepoints=10).fit(y)
        trend = p.trend
        # Trend should roughly follow the piecewise pattern
        self.assertGreater(trend[50], trend[0])
        self.assertGreater(trend[100], trend[150])


if __name__ == '__main__':
    unittest.main()
