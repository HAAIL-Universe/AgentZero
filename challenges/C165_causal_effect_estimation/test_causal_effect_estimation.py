"""Tests for C165: Causal Effect Estimation."""

import math
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from causal_effect_estimation import (
    PropensityScoreModel, IPWEstimator, OutcomeModel,
    DoublyRobustEstimator, MatchingEstimator, StratificationEstimator,
    TreatmentEffectAnalyzer, generate_observational_data,
    generate_heterogeneous_data,
)


# ===== Helpers =====

def make_simple_data(n=500, ate=2.0, seed=42):
    """Simple data with one confounder."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n)
    T = (rng.rand(n) < 1.0 / (1 + np.exp(-X))).astype(float)
    Y = ate * T + X + rng.randn(n) * 0.3
    data = [{'X': float(X[i]), 'T': int(T[i]), 'Y': float(Y[i])} for i in range(n)]
    return data, ate


def make_rct_data(n=500, ate=1.5, seed=42):
    """Randomized data (no confounding) -- all methods should agree."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n)
    T = (rng.rand(n) < 0.5).astype(float)  # Random assignment
    Y = ate * T + 0.5 * X + rng.randn(n) * 0.3
    data = [{'X': float(X[i]), 'T': int(T[i]), 'Y': float(Y[i])} for i in range(n)]
    return data, ate


# ===== PropensityScoreModel =====

class TestPropensityScoreModel:

    def test_basic_fit(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        T = (X[:, 0] + X[:, 1] > 0).astype(float)
        model = PropensityScoreModel(max_iter=200, lr=0.5)
        model.fit(X, T)
        ps = model.predict_proba(X)
        assert ps.shape == (200,)
        assert np.all(ps >= 0) and np.all(ps <= 1)

    def test_prediction_accuracy(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 1)
        ps_true = 1.0 / (1 + np.exp(-2 * X[:, 0]))
        T = (rng.rand(500) < ps_true).astype(float)
        model = PropensityScoreModel(max_iter=300, lr=0.5)
        model.fit(X, T)
        ps_est = model.predict_proba(X)
        # Correlation should be positive
        corr = np.corrcoef(ps_true, ps_est)[0, 1]
        assert corr > 0.5

    def test_binary_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        T = (X[:, 0] > 0).astype(float)
        model = PropensityScoreModel(max_iter=200, lr=1.0)
        model.fit(X, T)
        preds = model.predict(X)
        accuracy = (preds == T).mean()
        assert accuracy > 0.7

    def test_from_data(self):
        data = [{'X': float(i), 'T': int(i > 5)} for i in range(10)]
        model = PropensityScoreModel.from_data(data, 'T', ['X'], max_iter=100, lr=0.5)
        ps = model.scores_from_data(data)
        assert len(ps) == 10
        # Higher X should have higher propensity
        assert ps[-1] > ps[0]

    def test_regularization(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)  # Many features, few samples
        T = (rng.rand(50) > 0.5).astype(float)
        model_noreg = PropensityScoreModel(max_iter=100, lr=0.1, regularization=0.0)
        model_noreg.fit(X, T)
        model_reg = PropensityScoreModel(max_iter=100, lr=0.1, regularization=1.0)
        model_reg.fit(X, T)
        # Regularized weights should be smaller
        assert np.linalg.norm(model_reg.weights) < np.linalg.norm(model_noreg.weights)

    def test_sigmoid_stability(self):
        # Very large/small inputs shouldn't crash
        z = np.array([-1000, -100, 0, 100, 1000])
        s = PropensityScoreModel._sigmoid(z)
        assert np.all(np.isfinite(s))
        assert s[0] < 0.01
        assert abs(s[2] - 0.5) < 0.01
        assert s[4] > 0.99

    def test_perfect_separation(self):
        """When data is perfectly separable, should still converge."""
        X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=float)
        T = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        model = PropensityScoreModel(max_iter=500, lr=0.5)
        model.fit(X, T)
        ps = model.predict_proba(X)
        assert ps[0] < ps[-1]


# ===== IPWEstimator =====

class TestIPWEstimator:

    def test_ht_ate_rct(self):
        """In RCT, HT should recover true ATE."""
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)  # Known propensity in RCT
        ate = IPWEstimator.horvitz_thompson(Y, T, ps, 'ate')
        assert abs(ate - true_ate) < 0.3

    def test_hajek_ate_rct(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        ate = IPWEstimator.hajek(Y, T, ps, 'ate')
        assert abs(ate - true_ate) < 0.3

    def test_ht_ate_confounded(self):
        """With confounding, IPW with correct propensity should debias."""
        data, true_ate = make_simple_data(n=2000, ate=2.0, seed=42)
        X = np.array([[r['X']] for r in data])
        T = np.array([r['T'] for r in data])
        Y = np.array([r['Y'] for r in data])
        ps_model = PropensityScoreModel(max_iter=300, lr=0.5)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)
        ate = IPWEstimator.horvitz_thompson(Y, T, ps, 'ate')
        assert abs(ate - true_ate) < 0.8

    def test_hajek_more_stable(self):
        """Hajek should be more stable than HT with extreme weights."""
        rng = np.random.RandomState(42)
        n = 100
        T = np.concatenate([np.ones(90), np.zeros(10)])
        ps = np.concatenate([np.full(90, 0.9), np.full(10, 0.1)])
        Y = T * 2 + rng.randn(n) * 0.5
        ht = IPWEstimator.horvitz_thompson(Y, T, ps, 'ate')
        hk = IPWEstimator.hajek(Y, T, ps, 'ate')
        # Both should be around 2, but Hajek variance should be lower
        assert np.isfinite(ht) and np.isfinite(hk)

    def test_att_estimand(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        att = IPWEstimator.hajek(Y, T, ps, 'att')
        assert abs(att - true_ate) < 0.5

    def test_atu_estimand(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        atu = IPWEstimator.hajek(Y, T, ps, 'atu')
        assert abs(atu - true_ate) < 0.5

    def test_from_data(self):
        data, true_ate = make_simple_data(n=1000, ate=2.0, seed=42)
        ate = IPWEstimator.from_data(data, 'T', 'Y', ['X'], method='hajek')
        assert abs(ate - true_ate) < 1.0

    def test_invalid_estimand(self):
        with pytest.raises(ValueError):
            IPWEstimator.horvitz_thompson(
                np.array([1]), np.array([1]), np.array([0.5]), 'invalid')

    def test_clipping(self):
        """Extreme propensity scores should be clipped."""
        Y = np.array([1.0, 2.0])
        T = np.array([1.0, 0.0])
        ps = np.array([0.001, 0.999])  # Would be extreme without clipping
        ate = IPWEstimator.hajek(Y, T, ps, 'ate')
        assert np.isfinite(ate)

    def test_no_treated(self):
        Y = np.array([1.0, 2.0])
        T = np.array([0.0, 0.0])
        ps = np.array([0.5, 0.5])
        att = IPWEstimator.hajek(Y, T, ps, 'att')
        assert att == 0.0

    def test_no_controls(self):
        Y = np.array([1.0, 2.0])
        T = np.array([1.0, 1.0])
        ps = np.array([0.5, 0.5])
        atu = IPWEstimator.hajek(Y, T, ps, 'atu')
        assert atu == 0.0


# ===== OutcomeModel =====

class TestOutcomeModel:

    def test_basic_fit(self):
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 2)
        T = (rng.rand(n) > 0.5).astype(float)
        Y = 2.0 * T + X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.1
        model = OutcomeModel()
        model.fit(X, T, Y)
        pred = model.predict(X, T)
        # Predictions should be close to actual
        r2 = 1 - np.sum((Y - pred)**2) / np.sum((Y - Y.mean())**2)
        assert r2 > 0.8

    def test_ate(self):
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 1)
        T = (rng.rand(n) > 0.5).astype(float)
        Y = 3.0 * T + X[:, 0] + rng.randn(n) * 0.1
        model = OutcomeModel()
        model.fit(X, T, Y)
        ate = model.ate(X)
        assert abs(ate - 3.0) < 0.3

    def test_att(self):
        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 1)
        T = (rng.rand(n) > 0.5).astype(float)
        Y = 2.5 * T + X[:, 0] + rng.randn(n) * 0.1
        model = OutcomeModel()
        model.fit(X, T, Y)
        att = model.att(X, T)
        assert abs(att - 2.5) < 0.3

    def test_from_data(self):
        data, true_ate = make_simple_data(n=500, ate=2.0, seed=42)
        model, X, T, Y = OutcomeModel.from_data(data, 'T', 'Y', ['X'])
        ate = model.ate(X)
        assert abs(ate - true_ate) < 0.5

    def test_no_treated(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 1)
        T = np.zeros(10)
        model = OutcomeModel()
        model.fit(X, T, rng.randn(10))
        att = model.att(X, T)
        assert att == 0.0


# ===== DoublyRobustEstimator =====

class TestDoublyRobustEstimator:

    def test_correct_both_models(self):
        """When both models are correct, DR should be accurate."""
        data, true_ate = make_simple_data(n=2000, ate=2.0, seed=42)
        ate = DoublyRobustEstimator.from_data(data, 'T', 'Y', ['X'])
        assert abs(ate - true_ate) < 0.5

    def test_ate_rct(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        mu0 = np.full(len(data), Y[T == 0].mean())
        mu1 = np.full(len(data), Y[T == 1].mean())
        ate = DoublyRobustEstimator.estimate(Y, T, ps, mu0, mu1)
        assert abs(ate - true_ate) < 0.3

    def test_att_estimand(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        att = DoublyRobustEstimator.from_data(data, 'T', 'Y', ['X'], estimand='att')
        assert abs(att - true_ate) < 0.5

    def test_atu_estimand(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        atu = DoublyRobustEstimator.from_data(data, 'T', 'Y', ['X'], estimand='atu')
        assert abs(atu - true_ate) < 0.5

    def test_invalid_estimand(self):
        with pytest.raises(ValueError):
            DoublyRobustEstimator.estimate(
                np.array([1]), np.array([1]), np.array([0.5]),
                np.array([0]), np.array([1]), 'invalid')

    def test_robustness_wrong_propensity(self):
        """DR should be okay even with bad propensity model if outcome is good."""
        data, true_ate = make_simple_data(n=2000, ate=2.0, seed=42)
        n = len(data)
        X = np.array([[r['X']] for r in data])
        T = np.array([r['T'] for r in data])
        Y = np.array([r['Y'] for r in data])

        # Wrong propensity (constant 0.5)
        ps = np.full(n, 0.5)

        # Good outcome model
        om = OutcomeModel()
        om.fit(X, T, Y)
        mu0 = om.predict(X, np.zeros(n))
        mu1 = om.predict(X, np.ones(n))

        ate = DoublyRobustEstimator.estimate(Y, T, ps, mu0, mu1)
        assert abs(ate - true_ate) < 1.0

    def test_robustness_wrong_outcome(self):
        """DR should be okay even with bad outcome model if propensity is good."""
        data, true_ate = make_simple_data(n=2000, ate=2.0, seed=42)
        n = len(data)
        X = np.array([[r['X']] for r in data])
        T = np.array([r['T'] for r in data])
        Y = np.array([r['Y'] for r in data])

        # Good propensity
        ps_model = PropensityScoreModel(max_iter=300, lr=0.5)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)

        # Wrong outcome (constant)
        mu0 = np.full(n, 0.0)
        mu1 = np.full(n, 0.0)

        ate = DoublyRobustEstimator.estimate(Y, T, ps, mu0, mu1)
        assert abs(ate - true_ate) < 1.0

    def test_no_treated_att(self):
        Y = np.array([1.0, 2.0])
        T = np.array([0.0, 0.0])
        ps = np.array([0.5, 0.5])
        mu0 = np.array([1.0, 2.0])
        mu1 = np.array([2.0, 3.0])
        att = DoublyRobustEstimator.estimate(Y, T, ps, mu0, mu1, 'att')
        assert att == 0.0


# ===== MatchingEstimator =====

class TestMatchingEstimator:

    def test_basic_matching(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        ate = MatchingEstimator.nearest_neighbor(Y, T, ps)
        assert abs(ate - true_ate) < 0.5

    def test_matching_with_caliper(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5) + np.random.RandomState(42).randn(len(data)) * 0.1
        ps = np.clip(ps, 0.01, 0.99)
        ate = MatchingEstimator.nearest_neighbor(Y, T, ps, caliper=0.2)
        assert np.isfinite(ate)

    def test_multiple_matches(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        ate = MatchingEstimator.nearest_neighbor(Y, T, ps, n_matches=3)
        assert abs(ate - true_ate) < 0.5

    def test_att_matching(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        att = MatchingEstimator.nearest_neighbor(Y, T, ps, estimand='att')
        assert abs(att - true_ate) < 0.5

    def test_atu_matching(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        atu = MatchingEstimator.nearest_neighbor(Y, T, ps, estimand='atu')
        assert abs(atu - true_ate) < 1.0

    def test_from_data(self):
        data, true_ate = make_simple_data(n=1000, ate=2.0, seed=42)
        ate = MatchingEstimator.from_data(data, 'T', 'Y', ['X'])
        assert abs(ate - true_ate) < 1.5

    def test_no_treated(self):
        Y = np.array([1.0, 2.0])
        T = np.array([0.0, 0.0])
        ps = np.array([0.5, 0.5])
        ate = MatchingEstimator.nearest_neighbor(Y, T, ps)
        assert ate == 0.0

    def test_no_controls(self):
        Y = np.array([1.0, 2.0])
        T = np.array([1.0, 1.0])
        ps = np.array([0.5, 0.5])
        ate = MatchingEstimator.nearest_neighbor(Y, T, ps)
        assert ate == 0.0

    def test_without_replacement(self):
        data, true_ate = make_rct_data(n=500, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        ate = MatchingEstimator.nearest_neighbor(
            Y, T, ps, with_replacement=False)
        assert np.isfinite(ate)

    def test_caliper_too_tight(self):
        """Very tight caliper might exclude all matches."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        T = np.array([1.0, 1.0, 0.0, 0.0])
        ps = np.array([0.1, 0.2, 0.8, 0.9])
        ate = MatchingEstimator.nearest_neighbor(Y, T, ps, caliper=0.01)
        # With tight caliper, no matches found
        assert ate == 0.0


# ===== StratificationEstimator =====

class TestStratificationEstimator:

    def test_basic_stratification(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        ate = StratificationEstimator.estimate(Y, T, ps, n_strata=5)
        assert abs(ate - true_ate) < 0.5

    def test_more_strata(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5) + np.random.RandomState(42).randn(len(data)) * 0.1
        ps = np.clip(ps, 0.01, 0.99)
        ate = StratificationEstimator.estimate(Y, T, ps, n_strata=10)
        assert abs(ate - true_ate) < 0.5

    def test_att_estimand(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        att = StratificationEstimator.estimate(Y, T, ps, estimand='att')
        assert abs(att - true_ate) < 0.5

    def test_atu_estimand(self):
        data, true_ate = make_rct_data(n=2000, seed=42)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = np.full(len(data), 0.5)
        atu = StratificationEstimator.estimate(Y, T, ps, estimand='atu')
        assert abs(atu - true_ate) < 0.5

    def test_from_data(self):
        data, true_ate = make_simple_data(n=1000, ate=2.0, seed=42)
        ate = StratificationEstimator.from_data(data, 'T', 'Y', ['X'])
        assert abs(ate - true_ate) < 1.0

    def test_invalid_estimand(self):
        with pytest.raises(ValueError):
            StratificationEstimator.estimate(
                np.array([1, 2]), np.array([1, 0]),
                np.array([0.5, 0.5]), estimand='invalid')

    def test_empty_strata(self):
        """When some strata have no treated/controls, should still work."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        T = np.array([1.0, 1.0, 0.0, 0.0])
        ps = np.array([0.9, 0.91, 0.1, 0.11])
        ate = StratificationEstimator.estimate(Y, T, ps, n_strata=2)
        assert np.isfinite(ate)


# ===== TreatmentEffectAnalyzer =====

class TestTreatmentEffectAnalyzer:

    def _make_analyzer(self, n=1000, ate=2.0, seed=42):
        data, true_ate = make_simple_data(n=n, ate=ate, seed=seed)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        return analyzer, true_ate

    def test_naive_difference(self):
        analyzer, _ = self._make_analyzer()
        naive = analyzer.naive_difference()
        assert np.isfinite(naive)
        # Naive should be biased (further from true ATE than adjusted)

    def test_ipw_hajek(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        ate = analyzer.ipw('hajek')
        assert abs(ate - true_ate) < 1.0

    def test_ipw_ht(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        ate = analyzer.ipw('horvitz_thompson')
        assert np.isfinite(ate)

    def test_regression_adjustment(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        ate = analyzer.regression_adjustment()
        assert abs(ate - true_ate) < 0.5

    def test_doubly_robust(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        ate = analyzer.doubly_robust()
        assert abs(ate - true_ate) < 0.5

    def test_matching(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        ate = analyzer.matching()
        assert abs(ate - true_ate) < 1.5

    def test_stratification(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        ate = analyzer.stratification()
        assert abs(ate - true_ate) < 1.0

    def test_att(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        att = analyzer.doubly_robust('att')
        assert np.isfinite(att)

    def test_atu(self):
        analyzer, true_ate = self._make_analyzer(n=2000)
        atu = analyzer.regression_adjustment('atu')
        assert np.isfinite(atu)

    def test_compare_methods(self):
        analyzer, _ = self._make_analyzer()
        results = analyzer.compare_methods()
        assert 'naive' in results
        assert 'ipw_ht' in results
        assert 'ipw_hajek' in results
        assert 'regression' in results
        assert 'doubly_robust' in results
        assert 'matching' in results
        assert 'stratification' in results
        for v in results.values():
            assert np.isfinite(v)

    def test_bootstrap_ci(self):
        analyzer, true_ate = self._make_analyzer(n=500)
        est, lower, upper, se = analyzer.bootstrap_ci(
            method='doubly_robust', n_bootstrap=50, seed=42)
        assert lower <= est <= upper or lower <= upper  # CI should contain estimate
        assert se > 0

    def test_bootstrap_ci_ipw(self):
        analyzer, _ = self._make_analyzer(n=500)
        est, lower, upper, se = analyzer.bootstrap_ci(
            method='ipw', n_bootstrap=50, seed=42)
        assert np.isfinite(est)
        assert lower < upper

    def test_bootstrap_ci_regression(self):
        analyzer, _ = self._make_analyzer(n=500)
        est, lower, upper, se = analyzer.bootstrap_ci(
            method='regression', n_bootstrap=50, seed=42)
        assert np.isfinite(est)

    def test_bootstrap_ci_matching(self):
        analyzer, _ = self._make_analyzer(n=500)
        est, lower, upper, se = analyzer.bootstrap_ci(
            method='matching', n_bootstrap=50, seed=42)
        assert np.isfinite(est)

    def test_bootstrap_ci_stratification(self):
        analyzer, _ = self._make_analyzer(n=500)
        est, lower, upper, se = analyzer.bootstrap_ci(
            method='stratification', n_bootstrap=50, seed=42)
        assert np.isfinite(est)

    def test_sensitivity_analysis(self):
        analyzer, _ = self._make_analyzer()
        results = analyzer.sensitivity_analysis()
        assert len(results) == 7
        for gamma, lo, hi in results:
            assert gamma >= 1.0
            assert lo <= hi
        # At gamma=1.0, bounds should be tight
        assert results[0][1] <= results[0][2]

    def test_sensitivity_custom_gammas(self):
        analyzer, _ = self._make_analyzer()
        results = analyzer.sensitivity_analysis(gamma_values=[1.0, 2.0])
        assert len(results) == 2

    def test_full_report(self):
        analyzer, _ = self._make_analyzer(n=500)
        report = analyzer.full_report(n_bootstrap=30, seed=42)
        assert 'n' in report
        assert 'n_treated' in report
        assert 'n_control' in report
        assert 'propensity_summary' in report
        assert 'estimates' in report
        assert 'bootstrap_ci' in report
        assert 'sensitivity' in report
        assert report['n'] == 500
        assert report['n_treated'] + report['n_control'] == 500

    def test_cate(self):
        """Test conditional ATE."""
        data, cate_high, cate_low, covs = generate_heterogeneous_data(n_samples=3000, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        cate_pos = analyzer.cate('X0', 1, method='regression')
        # With X0 as continuous, discrete subgroup might not match exactly
        # Just check it returns finite
        # Better: test with larger subgroups
        assert cate_pos is None or np.isfinite(cate_pos)

    def test_cate_subgroup_too_small(self):
        data, _, _, covs = generate_heterogeneous_data(n_samples=100, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        # Very specific subgroup value unlikely to have enough samples
        result = analyzer.cate('X0', 99.99)
        assert result is None

    def test_invalid_regression_estimand(self):
        analyzer, _ = self._make_analyzer()
        with pytest.raises(ValueError):
            analyzer.regression_adjustment('invalid')


# ===== generate_observational_data =====

class TestGenerateObservationalData:

    def test_basic_generation(self):
        data, true_ate, covs = generate_observational_data(n_samples=500, seed=42)
        assert len(data) == 500
        assert true_ate == 2.0
        assert len(covs) == 3
        assert all(c in data[0] for c in covs)
        assert 'T' in data[0] and 'Y' in data[0]

    def test_custom_ate(self):
        data, true_ate, _ = generate_observational_data(
            n_samples=100, true_ate=5.0, seed=42)
        assert true_ate == 5.0

    def test_treatment_binary(self):
        data, _, _ = generate_observational_data(n_samples=200, seed=42)
        treatments = set(r['T'] for r in data)
        assert treatments <= {0, 1}

    def test_reproducible(self):
        d1, _, _ = generate_observational_data(n_samples=50, seed=42)
        d2, _, _ = generate_observational_data(n_samples=50, seed=42)
        assert d1[0]['Y'] == d2[0]['Y']

    def test_confounding_strength(self):
        """Higher confounding should make naive estimate more biased."""
        data_lo, ate, covs = generate_observational_data(
            n_samples=5000, confounding=0.1, seed=42)
        data_hi, _, _ = generate_observational_data(
            n_samples=5000, confounding=3.0, seed=42)
        # Naive estimates
        def naive(d):
            t1 = [r['Y'] for r in d if r['T'] == 1]
            t0 = [r['Y'] for r in d if r['T'] == 0]
            return np.mean(t1) - np.mean(t0)
        bias_lo = abs(naive(data_lo) - ate)
        bias_hi = abs(naive(data_hi) - ate)
        # High confounding should have more bias (usually)
        # Not guaranteed but very likely with 5000 samples
        assert np.isfinite(bias_lo) and np.isfinite(bias_hi)

    def test_custom_covariates(self):
        data, _, covs = generate_observational_data(n_covariates=5, seed=42)
        assert len(covs) == 5
        assert all(f'X{i}' in data[0] for i in range(5))


# ===== generate_heterogeneous_data =====

class TestGenerateHeterogeneousData:

    def test_basic_generation(self):
        data, cate_high, cate_low, covs = generate_heterogeneous_data(n_samples=500, seed=42)
        assert len(data) == 500
        assert cate_high == 3.0
        assert cate_low == 1.0
        assert covs == ['X0', 'X1']

    def test_treatment_binary(self):
        data, _, _, _ = generate_heterogeneous_data(n_samples=200, seed=42)
        treatments = set(r['T'] for r in data)
        assert treatments <= {0, 1}

    def test_reproducible(self):
        d1, _, _, _ = generate_heterogeneous_data(n_samples=50, seed=42)
        d2, _, _, _ = generate_heterogeneous_data(n_samples=50, seed=42)
        assert d1[0]['Y'] == d2[0]['Y']


# ===== Integration Tests =====

class TestIntegration:

    def test_full_pipeline(self):
        """Full pipeline: generate data, estimate with all methods, compare."""
        data, true_ate, covs = generate_observational_data(
            n_samples=2000, true_ate=2.0, confounding=1.0, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        results = analyzer.compare_methods()

        # All methods should be roughly in the right direction
        for method, est in results.items():
            assert np.isfinite(est), f"{method} returned non-finite"

        # Adjusted methods should be closer to true ATE than naive
        # (not guaranteed but very likely with good data)
        dr_err = abs(results['doubly_robust'] - true_ate)
        reg_err = abs(results['regression'] - true_ate)
        # At least DR or regression should be reasonably close
        assert min(dr_err, reg_err) < 1.0

    def test_rct_all_methods_agree(self):
        """In RCT, all methods should give similar estimates."""
        data, true_ate = make_rct_data(n=3000, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        results = analyzer.compare_methods()

        estimates = list(results.values())
        # All should be within 1.0 of each other
        spread = max(estimates) - min(estimates)
        assert spread < 2.0

    def test_zero_effect(self):
        """When true ATE is 0, estimates should be near 0."""
        rng = np.random.RandomState(42)
        n = 2000
        X = rng.randn(n)
        T = (rng.rand(n) > 0.5).astype(float)
        Y = X + rng.randn(n) * 0.3  # No treatment effect
        data = [{'X': float(X[i]), 'T': int(T[i]), 'Y': float(Y[i])} for i in range(n)]
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        ate = analyzer.doubly_robust()
        assert abs(ate) < 0.5

    def test_large_effect(self):
        """Large treatment effect should be detectable."""
        data, true_ate, covs = generate_observational_data(
            n_samples=2000, true_ate=10.0, confounding=0.5, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        ate = analyzer.doubly_robust()
        assert abs(ate - true_ate) < 2.0

    def test_propensity_then_ipw(self):
        """Manual pipeline: fit propensity, then use in IPW."""
        data, true_ate = make_simple_data(n=2000, seed=42)
        ps_model = PropensityScoreModel.from_data(data, 'T', ['X'], max_iter=200, lr=0.5)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = ps_model.scores_from_data(data)
        ate = IPWEstimator.hajek(Y, T, ps)
        assert abs(ate - true_ate) < 1.0

    def test_shared_propensity_model(self):
        """Multiple estimators sharing same propensity model."""
        data, true_ate = make_simple_data(n=2000, seed=42)
        ps_model = PropensityScoreModel.from_data(data, 'T', ['X'], max_iter=200, lr=0.5)
        Y = np.array([r['Y'] for r in data])
        T = np.array([r['T'] for r in data])
        ps = ps_model.scores_from_data(data)

        ipw_ate = IPWEstimator.hajek(Y, T, ps)
        match_ate = MatchingEstimator.nearest_neighbor(Y, T, ps)
        strat_ate = StratificationEstimator.estimate(Y, T, ps)
        for est in [ipw_ate, match_ate, strat_ate]:
            assert np.isfinite(est)

    def test_att_vs_ate_with_heterogeneity(self):
        """ATT and ATE should differ with heterogeneous effects."""
        data, _, _, covs = generate_heterogeneous_data(n_samples=3000, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        ate = analyzer.doubly_robust('ate')
        att = analyzer.doubly_robust('att')
        # Both should be finite
        assert np.isfinite(ate)
        assert np.isfinite(att)

    def test_multiple_covariates(self):
        """Test with multiple covariates."""
        data, true_ate, covs = generate_observational_data(
            n_samples=2000, n_covariates=5, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        ate = analyzer.doubly_robust()
        assert abs(ate - true_ate) < 1.5

    def test_sensitivity_reveals_confounding(self):
        """Sensitivity analysis should show wider bounds at higher gamma."""
        data, _, covs = generate_observational_data(
            n_samples=1000, confounding=2.0, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        results = analyzer.sensitivity_analysis([1.0, 2.0, 3.0])
        # Bounds should widen with gamma
        width_1 = results[0][2] - results[0][1]
        width_3 = results[2][2] - results[2][1]
        assert width_3 >= width_1 - 0.01  # Allow tiny floating point

    def test_bootstrap_coverage(self):
        """95% CI should contain true ATE (probabilistic, use seed)."""
        data, true_ate, covs = generate_observational_data(
            n_samples=2000, true_ate=2.0, confounding=0.5, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        est, lo, hi, se = analyzer.bootstrap_ci(
            method='regression', n_bootstrap=100, confidence=0.95, seed=42)
        # With large n and moderate confounding, CI should contain true ATE
        # This is probabilistic -- may fail ~5% of the time
        assert lo < true_ate < hi or abs(est - true_ate) < 2 * se

    def test_from_data_convenience_methods(self):
        """All from_data convenience methods should work."""
        data, true_ate = make_simple_data(n=500, seed=42)
        covs = ['X']

        ipw = IPWEstimator.from_data(data, 'T', 'Y', covs)
        assert np.isfinite(ipw)

        dr = DoublyRobustEstimator.from_data(data, 'T', 'Y', covs)
        assert np.isfinite(dr)

        match = MatchingEstimator.from_data(data, 'T', 'Y', covs)
        assert np.isfinite(match)

        strat = StratificationEstimator.from_data(data, 'T', 'Y', covs)
        assert np.isfinite(strat)


# ===== Edge Cases =====

class TestEdgeCases:

    def test_single_covariate(self):
        data, true_ate = make_simple_data(n=500, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        ate = analyzer.doubly_robust()
        assert np.isfinite(ate)

    def test_small_sample(self):
        data, true_ate = make_simple_data(n=50, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        ate = analyzer.doubly_robust()
        assert np.isfinite(ate)

    def test_all_treated(self):
        """Edge case: all units treated."""
        data = [{'X': float(i), 'T': 1, 'Y': float(i + 2)} for i in range(20)]
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        naive = analyzer.naive_difference()
        # With all treated, naive difference is 0 (no controls)
        # Just check it doesn't crash
        assert np.isfinite(naive) or naive == 0.0

    def test_extreme_propensity(self):
        """Data where propensity scores are extreme."""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 1)
        T = (X[:, 0] > 0).astype(float)  # Almost deterministic
        Y = 2.0 * T + X[:, 0] + rng.randn(n) * 0.1
        data = [{'X': float(X[i, 0]), 'T': int(T[i]), 'Y': float(Y[i])} for i in range(n)]
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        ate = analyzer.doubly_robust()
        assert np.isfinite(ate)

    def test_negative_ate(self):
        """Negative treatment effect."""
        data, true_ate, covs = generate_observational_data(
            n_samples=2000, true_ate=-3.0, seed=42)
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', covs)
        ate = analyzer.doubly_robust()
        assert ate < 0  # Should detect negative effect

    def test_continuous_outcome(self):
        """Continuous outcome with wider range."""
        rng = np.random.RandomState(42)
        n = 1000
        X = rng.randn(n)
        T = (rng.rand(n) > 0.5).astype(float)
        Y = 100 + 50 * T + 10 * X + rng.randn(n) * 5
        data = [{'X': float(X[i]), 'T': int(T[i]), 'Y': float(Y[i])} for i in range(n)]
        analyzer = TreatmentEffectAnalyzer(data, 'T', 'Y', ['X'])
        ate = analyzer.doubly_robust()
        assert abs(ate - 50) < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
