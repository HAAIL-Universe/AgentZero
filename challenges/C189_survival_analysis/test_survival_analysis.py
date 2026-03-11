"""Tests for C189: Survival Analysis."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from survival_analysis import (
    SurvivalData, KaplanMeier, NelsonAalen, LogRankTest,
    CoxPH, ExponentialModel, WeibullModel, LogNormalModel,
    LogLogisticModel, LifeTable, SchoenfeldTest, CumulativeIncidence
)


# =========================================================================
# SurvivalData
# =========================================================================

class TestSurvivalData:
    def test_basic_creation(self):
        sd = SurvivalData([1, 2, 3], [1, 0, 1])
        assert sd.n_samples == 3
        assert sd.n_events == 2
        assert sd.n_censored == 1

    def test_float_conversion(self):
        sd = SurvivalData([1, 2, 3], [1, 0, 1])
        assert sd.times.dtype == float
        assert sd.events.dtype == float

    def test_negative_time_raises(self):
        try:
            SurvivalData([-1, 2], [1, 1])
            assert False, "Should raise"
        except ValueError:
            pass

    def test_mismatched_lengths(self):
        try:
            SurvivalData([1, 2, 3], [1, 0])
            assert False, "Should raise"
        except ValueError:
            pass

    def test_covariates(self):
        sd = SurvivalData([1, 2], [1, 1], covariates=[[0.5], [1.0]])
        assert sd.covariates.shape == (2, 1)

    def test_covariates_1d_reshape(self):
        sd = SurvivalData([1, 2], [1, 1], covariates=[0.5, 1.0])
        assert sd.covariates.shape == (2, 1)

    def test_groups(self):
        sd = SurvivalData([1, 2, 3], [1, 0, 1], groups=['A', 'B', 'A'])
        assert len(sd.groups) == 3

    def test_subset(self):
        sd = SurvivalData([1, 2, 3, 4], [1, 0, 1, 1], groups=[0, 1, 0, 1])
        sub = sd.subset(sd.groups == 0)
        assert sub.n_samples == 2
        assert sub.groups is not None

    def test_subset_no_covariates(self):
        sd = SurvivalData([1, 2, 3], [1, 0, 1])
        sub = sd.subset(np.array([True, False, True]))
        assert sub.n_samples == 2
        assert sub.covariates is None

    def test_all_censored(self):
        sd = SurvivalData([1, 2, 3], [0, 0, 0])
        assert sd.n_events == 0
        assert sd.n_censored == 3


# =========================================================================
# Kaplan-Meier
# =========================================================================

class TestKaplanMeier:
    def _make_data(self):
        # Classic example: 10 patients
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        events = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        return SurvivalData(times, events)

    def test_fit_basic(self):
        km = KaplanMeier().fit(self._make_data())
        assert km._fitted
        assert len(km.event_times) > 0

    def test_survival_starts_at_one(self):
        km = KaplanMeier().fit(self._make_data())
        assert km.survival_function(0) == 1.0

    def test_survival_decreases(self):
        km = KaplanMeier().fit(self._make_data())
        s = km.survival_function(km.event_times)
        for i in range(1, len(s)):
            assert s[i] <= s[i - 1]

    def test_survival_function_step(self):
        km = KaplanMeier().fit(self._make_data())
        # Between event times, S should be constant
        s1 = km.survival_function(1.5)
        s2 = km.survival_function(1.9)
        assert s1 == s2

    def test_all_events_no_censoring(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        km = KaplanMeier().fit(sd)
        # S(1) = 4/5, S(2) = 3/5, S(3) = 2/5, S(4) = 1/5, S(5) = 0
        assert abs(km.survival_function(1) - 0.8) < 0.01
        assert abs(km.survival_function(5) - 0.0) < 0.01

    def test_no_events(self):
        sd = SurvivalData([1, 2, 3], [0, 0, 0])
        km = KaplanMeier().fit(sd)
        assert len(km.event_times) == 0
        assert km.survival_function(5) == 1.0

    def test_median_survival(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        km = KaplanMeier().fit(sd)
        med = km.median_survival()
        assert 2.5 <= med <= 3.5

    def test_median_undefined(self):
        sd = SurvivalData([1, 2, 3], [0, 0, 0])
        km = KaplanMeier().fit(sd)
        assert np.isnan(km.median_survival())

    def test_confidence_interval(self):
        sd = SurvivalData(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
        )
        km = KaplanMeier().fit(sd)
        lower, upper = km.confidence_interval()
        assert len(lower) == len(km.event_times)
        assert np.all(lower <= km.survival_prob)
        assert np.all(upper >= km.survival_prob)
        assert np.all(lower >= 0)
        assert np.all(upper <= 1)

    def test_restricted_mean(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        km = KaplanMeier().fit(sd)
        rmst = km.restricted_mean()
        assert rmst > 0
        assert rmst < 5

    def test_restricted_mean_with_tau(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        km = KaplanMeier().fit(sd)
        rmst1 = km.restricted_mean(tau=3)
        rmst2 = km.restricted_mean(tau=5)
        assert rmst1 < rmst2

    def test_large_dataset(self):
        np.random.seed(42)
        n = 200
        times = np.random.exponential(5, n)
        events = np.random.binomial(1, 0.7, n).astype(float)
        sd = SurvivalData(times, events)
        km = KaplanMeier().fit(sd)
        assert km._fitted
        assert km.survival_function(0) == 1.0
        # Survival should be < 1 after some time
        assert km.survival_function(10) < 1.0

    def test_tied_event_times(self):
        sd = SurvivalData([1, 1, 2, 2, 3], [1, 1, 1, 0, 1])
        km = KaplanMeier().fit(sd)
        assert km._fitted
        s = km.survival_function(km.event_times)
        assert s[0] < 1.0

    def test_survival_array_input(self):
        km = KaplanMeier().fit(self._make_data())
        s = km.survival_function([0, 1, 5, 100])
        assert len(s) == 4
        assert s[0] == 1.0


# =========================================================================
# Nelson-Aalen
# =========================================================================

class TestNelsonAalen:
    def test_fit_basic(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 0, 1, 1])
        na = NelsonAalen().fit(sd)
        assert na._fitted
        assert len(na.cumulative_hazard) > 0

    def test_hazard_increases(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        na = NelsonAalen().fit(sd)
        h = na.cumulative_hazard
        for i in range(1, len(h)):
            assert h[i] >= h[i - 1]

    def test_hazard_at_zero(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        na = NelsonAalen().fit(sd)
        assert na.hazard_at(0) == 0.0

    def test_survival_from_hazard(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        na = NelsonAalen().fit(sd)
        s = na.survival_function(np.array([1, 3, 5]))
        assert len(s) == 3
        assert s[0] > s[1] > s[2]

    def test_consistency_with_km(self):
        np.random.seed(123)
        times = np.random.exponential(3, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)

        km = KaplanMeier().fit(sd)
        na = NelsonAalen().fit(sd)

        # For large samples with all events, NA and KM should be close
        for t in [1, 2, 3]:
            s_km = km.survival_function(t)
            s_na = na.survival_function(t)
            assert abs(s_km - s_na) < 0.15  # approximate agreement

    def test_no_events(self):
        sd = SurvivalData([1, 2, 3], [0, 0, 0])
        na = NelsonAalen().fit(sd)
        assert len(na.cumulative_hazard) == 0
        assert na.hazard_at(5) == 0.0


# =========================================================================
# Log-Rank Test
# =========================================================================

class TestLogRankTest:
    def _make_two_groups(self, separation=0):
        np.random.seed(42)
        # Group 1: baseline
        t1 = np.random.exponential(5, 30)
        e1 = np.random.binomial(1, 0.8, 30)
        # Group 2: shifted
        t2 = np.random.exponential(5 + separation, 30)
        e2 = np.random.binomial(1, 0.8, 30)

        times = np.concatenate([t1, t2])
        events = np.concatenate([e1, e2]).astype(float)
        groups = np.array([0] * 30 + [1] * 30)
        return SurvivalData(times, events, groups=groups)

    def test_no_difference(self):
        sd = self._make_two_groups(separation=0)
        result = LogRankTest.test(sd)
        assert result.n_groups == 2
        assert result.p_value > 0.01  # No significant difference expected

    def test_significant_difference(self):
        sd = self._make_two_groups(separation=20)
        result = LogRankTest.test(sd)
        assert result.p_value < 0.1  # Large separation

    def test_observed_expected(self):
        sd = self._make_two_groups(separation=0)
        result = LogRankTest.test(sd)
        assert len(result.observed) == 2
        assert len(result.expected) == 2

    def test_three_groups(self):
        np.random.seed(42)
        times = np.concatenate([
            np.random.exponential(3, 20),
            np.random.exponential(5, 20),
            np.random.exponential(10, 20)
        ])
        events = np.ones(60)
        groups = np.array([0] * 20 + [1] * 20 + [2] * 20)
        sd = SurvivalData(times, events, groups=groups)
        result = LogRankTest.test(sd)
        assert result.n_groups == 3
        assert result.test_statistic >= 0

    def test_wilcoxon_weight(self):
        sd = self._make_two_groups(separation=5)
        r1 = LogRankTest.test(sd, weights=None)
        r2 = LogRankTest.test(sd, weights='wilcoxon')
        # Both should give valid results
        assert r1.test_statistic >= 0
        assert r2.test_statistic >= 0

    def test_tarone_ware_weight(self):
        sd = self._make_two_groups(separation=5)
        r = LogRankTest.test(sd, weights='tarone-ware')
        assert r.test_statistic >= 0

    def test_no_groups_raises(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        try:
            LogRankTest.test(sd)
            assert False, "Should raise"
        except ValueError:
            pass

    def test_single_group_raises(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1], groups=[0, 0, 0])
        try:
            LogRankTest.test(sd)
            assert False, "Should raise"
        except ValueError:
            pass

    def test_significant_property(self):
        sd = self._make_two_groups(separation=0)
        result = LogRankTest.test(sd)
        # significant is a derived property
        assert isinstance(result.significant, (bool, np.bool_))


# =========================================================================
# Cox PH
# =========================================================================

class TestCoxPH:
    def _make_data(self, n=100, seed=42):
        np.random.seed(seed)
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # True coefficients: [0.5, -0.3]
        hazard = np.exp(0.5 * x1 - 0.3 * x2)
        times = np.random.exponential(1.0 / hazard)
        events = np.random.binomial(1, 0.8, n).astype(float)
        covariates = np.column_stack([x1, x2])
        return SurvivalData(times, events, covariates=covariates)

    def test_fit_basic(self):
        sd = self._make_data()
        cox = CoxPH()
        result = cox.fit(sd)
        assert len(result.coefficients) == 2
        assert len(result.standard_errors) == 2

    def test_coefficient_direction(self):
        sd = self._make_data(n=200)
        cox = CoxPH()
        result = cox.fit(sd)
        # x1 has positive true coef, x2 negative
        assert result.coefficients[0] > 0
        assert result.coefficients[1] < 0

    def test_hazard_ratios(self):
        sd = self._make_data()
        cox = CoxPH()
        result = cox.fit(sd)
        assert np.allclose(result.hazard_ratios, np.exp(result.coefficients))

    def test_concordance(self):
        sd = self._make_data()
        cox = CoxPH()
        result = cox.fit(sd)
        assert 0.5 < result.concordance < 1.0

    def test_predict_survival(self):
        sd = self._make_data()
        cox = CoxPH()
        cox.fit(sd)
        surv = cox.predict_survival(np.array([[0, 0]]))
        assert surv.shape[1] > 0
        # Survival starts at 1 (approximately) and decreases
        assert surv[0, 0] > surv[0, -1]

    def test_predict_survival_at_times(self):
        sd = self._make_data()
        cox = CoxPH()
        cox.fit(sd)
        surv = cox.predict_survival(np.array([[0, 0]]), t=[0.5, 1.0, 2.0])
        assert surv.shape == (1, 3)
        assert surv[0, 0] >= surv[0, 1] >= surv[0, 2]

    def test_predict_hazard_ratio(self):
        sd = self._make_data()
        cox = CoxPH()
        cox.fit(sd)
        hr = cox.predict_hazard_ratio(np.array([[0, 0]]))
        assert abs(hr[0] - 1.0) < 0.01  # Baseline = HR 1.0

    def test_l2_penalty(self):
        sd = self._make_data()
        cox1 = CoxPH(l2_penalty=0)
        cox2 = CoxPH(l2_penalty=10)
        r1 = cox1.fit(sd)
        r2 = cox2.fit(sd)
        # Penalized coefficients should be shrunk toward zero
        assert np.sum(r2.coefficients ** 2) <= np.sum(r1.coefficients ** 2) + 0.1

    def test_summary(self):
        sd = self._make_data()
        cox = CoxPH()
        result = cox.fit(sd)
        summary = result.summary()
        assert len(summary) == 2
        assert 'coef' in summary[0]
        assert 'hr' in summary[0]
        assert 'p' in summary[0]

    def test_no_covariates_raises(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        try:
            CoxPH().fit(sd)
            assert False
        except ValueError:
            pass

    def test_single_covariate(self):
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        times = np.random.exponential(np.exp(-0.5 * x))
        events = np.ones(n)
        sd = SurvivalData(times, events, covariates=x)
        result = CoxPH().fit(sd)
        assert len(result.coefficients) == 1


# =========================================================================
# Exponential Model
# =========================================================================

class TestExponentialModel:
    def test_fit(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        model = ExponentialModel().fit(sd)
        assert model.rate > 0

    def test_survival(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        model = ExponentialModel().fit(sd)
        s = model.survival([0, 1, 5, 10])
        assert s[0] == 1.0
        assert s[1] > s[2] > s[3]

    def test_constant_hazard(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        model = ExponentialModel().fit(sd)
        h = model.hazard([1, 5, 10])
        assert np.allclose(h, h[0])  # Constant

    def test_cumulative_hazard_linear(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        model = ExponentialModel().fit(sd)
        ch = model.cumulative_hazard([1, 2, 3])
        # Should be linear: H(t) = lambda * t
        assert abs(ch[1] - 2 * ch[0]) < 1e-10
        assert abs(ch[2] - 3 * ch[0]) < 1e-10

    def test_mean_survival(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        model = ExponentialModel().fit(sd)
        assert model.mean_survival() == 1.0 / model.rate

    def test_median_survival(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        model = ExponentialModel().fit(sd)
        assert abs(model.median_survival() - np.log(2) / model.rate) < 1e-10

    def test_log_likelihood(self):
        sd = SurvivalData([1, 2, 3], [1, 1, 1])
        model = ExponentialModel().fit(sd)
        ll = model.log_likelihood(sd)
        assert np.isfinite(ll)

    def test_aic(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        model = ExponentialModel().fit(sd)
        aic = model.aic(sd)
        assert np.isfinite(aic)

    def test_with_censoring(self):
        sd = SurvivalData([1, 2, 3, 4, 5], [1, 0, 1, 0, 1])
        model = ExponentialModel().fit(sd)
        assert model.rate > 0
        # Rate should be lower with censoring (fewer events, same total time)
        sd2 = SurvivalData([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
        model2 = ExponentialModel().fit(sd2)
        assert model.rate < model2.rate


# =========================================================================
# Weibull Model
# =========================================================================

class TestWeibullModel:
    def test_fit_exponential_like(self):
        # With shape ~1, should behave like exponential
        np.random.seed(42)
        times = np.random.exponential(3, 100)
        events = np.ones(100)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        assert abs(model.shape - 1.0) < 0.5

    def test_increasing_hazard(self):
        # Weibull with k > 1 has increasing hazard
        np.random.seed(42)
        # Generate Weibull data with k=2
        u = np.random.uniform(0, 1, 100)
        lam = 5
        k = 2.0
        times = lam * (-np.log(u)) ** (1.0 / k)
        events = np.ones(100)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        assert model.shape > 1.0

    def test_survival(self):
        np.random.seed(42)
        times = np.random.exponential(3, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        s = model.survival([0, 1, 5, 10])
        assert abs(s[0] - 1.0) < 0.01
        assert s[1] > s[2] > s[3]

    def test_hazard_positive(self):
        np.random.seed(42)
        times = np.random.exponential(3, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        h = model.hazard([1, 2, 3])
        assert np.all(h > 0)

    def test_cumulative_hazard(self):
        np.random.seed(42)
        times = np.random.exponential(3, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        ch = model.cumulative_hazard([1, 2, 3])
        assert ch[0] < ch[1] < ch[2]

    def test_mean_survival(self):
        np.random.seed(42)
        times = np.random.exponential(3, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        m = model.mean_survival()
        assert m > 0

    def test_median_survival(self):
        np.random.seed(42)
        times = np.random.exponential(3, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        med = model.median_survival()
        assert med > 0

    def test_aic(self):
        np.random.seed(42)
        times = np.random.exponential(3, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = WeibullModel().fit(sd)
        aic = model.aic(sd)
        assert np.isfinite(aic)


# =========================================================================
# Log-Normal Model
# =========================================================================

class TestLogNormalModel:
    def test_fit_uncensored(self):
        np.random.seed(42)
        times = np.random.lognormal(1, 0.5, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = LogNormalModel().fit(sd)
        assert abs(model.mu - 1.0) < 0.3
        assert abs(model.sigma - 0.5) < 0.3

    def test_fit_with_censoring(self):
        np.random.seed(42)
        times = np.random.lognormal(1, 0.5, 50)
        events = np.random.binomial(1, 0.7, 50).astype(float)
        sd = SurvivalData(times, events)
        model = LogNormalModel().fit(sd)
        assert model._fitted

    def test_survival(self):
        np.random.seed(42)
        times = np.random.lognormal(1, 0.5, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = LogNormalModel().fit(sd)
        s = model.survival([0.1, 1, 5, 20])
        assert s[0] > s[1] > s[2] > s[3]

    def test_hazard(self):
        np.random.seed(42)
        times = np.random.lognormal(1, 0.5, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = LogNormalModel().fit(sd)
        h = model.hazard([1, 2, 3])
        assert np.all(h > 0)

    def test_median(self):
        np.random.seed(42)
        times = np.random.lognormal(2, 0.3, 100)
        events = np.ones(100)
        sd = SurvivalData(times, events)
        model = LogNormalModel().fit(sd)
        # Median of lognormal = exp(mu)
        expected = np.exp(2)
        assert abs(model.median_survival() - expected) / expected < 0.3


# =========================================================================
# Log-Logistic Model
# =========================================================================

class TestLogLogisticModel:
    def test_fit(self):
        np.random.seed(42)
        # Generate log-logistic data
        u = np.random.uniform(0, 1, 100)
        alpha, beta = 5.0, 2.0
        times = alpha * (u / (1 - u)) ** (1.0 / beta)
        events = np.ones(100)
        sd = SurvivalData(times, events)
        model = LogLogisticModel().fit(sd)
        assert model._fitted
        assert model.alpha > 0
        assert model.beta > 0

    def test_survival(self):
        np.random.seed(42)
        u = np.random.uniform(0.01, 0.99, 50)
        times = 5.0 * (u / (1 - u)) ** 0.5
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = LogLogisticModel().fit(sd)
        s = model.survival([0.1, 1, 5, 20])
        assert s[0] > s[-1]

    def test_hazard(self):
        np.random.seed(42)
        u = np.random.uniform(0.01, 0.99, 50)
        times = 5.0 * (u / (1 - u)) ** 0.5
        events = np.ones(50)
        sd = SurvivalData(times, events)
        model = LogLogisticModel().fit(sd)
        h = model.hazard([1, 2, 3])
        assert np.all(h > 0)

    def test_median(self):
        np.random.seed(42)
        u = np.random.uniform(0.01, 0.99, 100)
        alpha = 5.0
        times = alpha * (u / (1 - u)) ** 0.5
        events = np.ones(100)
        sd = SurvivalData(times, events)
        model = LogLogisticModel().fit(sd)
        # Median of log-logistic = alpha
        assert abs(model.median_survival() - alpha) / alpha < 0.5


# =========================================================================
# Life Table
# =========================================================================

class TestLifeTable:
    def test_basic_fit(self):
        np.random.seed(42)
        times = np.random.exponential(5, 100)
        events = np.random.binomial(1, 0.7, 100).astype(float)
        sd = SurvivalData(times, events)
        lt = LifeTable(n_intervals=5).fit(sd)
        assert lt._fitted
        assert len(lt.table) == 5

    def test_custom_intervals(self):
        np.random.seed(42)
        times = np.random.exponential(5, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        lt = LifeTable(intervals=[0, 2, 4, 6, 10]).fit(sd)
        assert len(lt.table) == 4

    def test_get_table(self):
        np.random.seed(42)
        times = np.random.exponential(5, 50)
        events = np.ones(50)
        sd = SurvivalData(times, events)
        lt = LifeTable(n_intervals=5).fit(sd)
        table = lt.get_table()
        assert len(table) == 5
        assert 'q_x' in table[0]
        assert 'S_x' in table[0]
        assert 'h_x' in table[0]

    def test_survival_decreases(self):
        np.random.seed(42)
        times = np.random.exponential(3, 100)
        events = np.ones(100)
        sd = SurvivalData(times, events)
        lt = LifeTable(n_intervals=5).fit(sd)
        table = lt.get_table()
        for i in range(1, len(table)):
            assert table[i]['S_x'] <= table[i - 1]['S_x']

    def test_mortality_rate_bounds(self):
        np.random.seed(42)
        times = np.random.exponential(3, 100)
        events = np.ones(100)
        sd = SurvivalData(times, events)
        lt = LifeTable(n_intervals=5).fit(sd)
        table = lt.get_table()
        for row in table:
            assert 0 <= row['q_x'] <= 1


# =========================================================================
# Schoenfeld Test
# =========================================================================

class TestSchoenfeldTest:
    def test_ph_holds(self):
        # Generate data satisfying PH assumption
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        times = np.random.exponential(np.exp(-0.5 * x))
        events = np.ones(n)
        sd = SurvivalData(times, events, covariates=x)
        cox = CoxPH()
        result = cox.fit(sd)
        sch = SchoenfeldTest.test(sd, result)
        assert 'covariate_0' in sch
        # PH should hold for properly generated data
        assert isinstance(sch['covariate_0']['ph_holds'], (bool, np.bool_))

    def test_multiple_covariates(self):
        np.random.seed(42)
        n = 80
        X = np.random.normal(0, 1, (n, 3))
        times = np.random.exponential(1, n)
        events = np.ones(n)
        sd = SurvivalData(times, events, covariates=X)
        cox = CoxPH()
        result = cox.fit(sd)
        sch = SchoenfeldTest.test(sd, result)
        assert len(sch) == 3

    def test_residuals_exist(self):
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        times = np.random.exponential(1, n)
        events = np.ones(n)
        sd = SurvivalData(times, events, covariates=x)
        cox = CoxPH()
        result = cox.fit(sd)
        sch = SchoenfeldTest.test(sd, result)
        assert 'residuals' in sch['covariate_0']
        assert len(sch['covariate_0']['residuals']) > 0


# =========================================================================
# Cumulative Incidence (Competing Risks)
# =========================================================================

class TestCumulativeIncidence:
    def test_basic_fit(self):
        np.random.seed(42)
        n = 100
        times = np.random.exponential(5, n)
        events = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
        ci = CumulativeIncidence().fit(times, events)
        assert ci._fitted
        assert 1 in ci.cif
        assert 2 in ci.cif

    def test_cif_increases(self):
        np.random.seed(42)
        n = 100
        times = np.random.exponential(5, n)
        events = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])
        ci = CumulativeIncidence().fit(times, events)
        cif1 = ci.cif[1]
        for i in range(1, len(cif1)):
            assert cif1[i] >= cif1[i - 1] - 1e-10

    def test_sum_less_than_one(self):
        np.random.seed(42)
        n = 200
        times = np.random.exponential(5, n)
        events = np.random.choice([0, 1, 2], n, p=[0.1, 0.5, 0.4])
        ci = CumulativeIncidence().fit(times, events)
        # Sum of CIFs at any time should be <= 1
        for i in range(len(ci.event_times)):
            total = sum(ci.cif[et][i] for et in ci.cif if i < len(ci.cif[et]))
            assert total <= 1.0 + 1e-6

    def test_incidence_at(self):
        np.random.seed(42)
        n = 100
        times = np.random.exponential(5, n)
        events = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])
        ci = CumulativeIncidence().fit(times, events)
        inc = ci.incidence_at(3.0, 1)
        assert 0 <= inc <= 1

    def test_incidence_at_early(self):
        np.random.seed(42)
        times = np.random.exponential(5, 50) + 1  # minimum time = 1
        events = np.random.choice([0, 1], 50, p=[0.3, 0.7])
        ci = CumulativeIncidence().fit(times, events)
        # Before any events, CIF should be 0
        assert ci.incidence_at(0.01, 1) == 0.0

    def test_custom_event_types(self):
        np.random.seed(42)
        n = 60
        times = np.random.exponential(5, n)
        events = np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.3, 0.3, 0.2])
        ci = CumulativeIncidence().fit(times, events, event_types=[1, 3])
        assert 1 in ci.cif
        assert 3 in ci.cif
        assert 2 not in ci.cif


# =========================================================================
# Integration tests
# =========================================================================

class TestIntegration:
    def test_km_vs_exponential(self):
        """KM and exponential model should agree on survival for exp data."""
        np.random.seed(42)
        times = np.random.exponential(5, 200)
        events = np.ones(200)
        sd = SurvivalData(times, events)

        km = KaplanMeier().fit(sd)
        exp = ExponentialModel().fit(sd)

        # At mean survival time, both should give roughly S = exp(-1) ~ 0.37
        t_check = exp.mean_survival()
        s_km = km.survival_function(t_check)
        s_exp = float(exp.survival([t_check])[0])
        assert abs(s_km - s_exp) < 0.15

    def test_weibull_vs_exponential(self):
        """Weibull with shape=1 should match exponential."""
        np.random.seed(42)
        times = np.random.exponential(3, 100)
        events = np.ones(100)
        sd = SurvivalData(times, events)

        exp = ExponentialModel().fit(sd)
        weib = WeibullModel().fit(sd)

        # Shape should be near 1
        assert abs(weib.shape - 1.0) < 0.3
        # AIC comparison
        aic_exp = exp.aic(sd)
        aic_weib = weib.aic(sd)
        assert np.isfinite(aic_exp)
        assert np.isfinite(aic_weib)

    def test_cox_baseline_survival(self):
        """Cox PH baseline (all covariates = 0) should resemble KM."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 0.1, n)  # Weak effect
        times = np.random.exponential(5, n)
        events = np.ones(n)
        sd = SurvivalData(times, events, covariates=x)

        cox = CoxPH()
        cox.fit(sd)
        km = KaplanMeier().fit(sd)

        # At some mid-point
        t_mid = np.median(times)
        s_cox = cox.predict_survival(np.array([[0]]), t=[t_mid])[0, 0]
        s_km = km.survival_function(t_mid)
        assert abs(s_cox - s_km) < 0.3

    def test_model_comparison(self):
        """Compare multiple parametric models using AIC."""
        np.random.seed(42)
        times = np.random.exponential(5, 100)
        events = np.random.binomial(1, 0.8, 100).astype(float)
        sd = SurvivalData(times, events)

        exp = ExponentialModel().fit(sd)
        weib = WeibullModel().fit(sd)

        aic_exp = exp.aic(sd)
        aic_weib = weib.aic(sd)

        # Both should be finite
        assert np.isfinite(aic_exp)
        assert np.isfinite(aic_weib)

    def test_full_pipeline(self):
        """Full analysis pipeline: KM -> log-rank -> Cox -> predict."""
        np.random.seed(42)
        n = 60

        # Two groups with different survival
        x = np.concatenate([np.zeros(30), np.ones(30)])
        times = np.concatenate([
            np.random.exponential(5, 30),
            np.random.exponential(10, 30)
        ])
        events = np.random.binomial(1, 0.8, n).astype(float)

        sd = SurvivalData(times, events, covariates=x, groups=x)

        # Step 1: KM per group
        km0 = KaplanMeier().fit(sd.subset(x == 0))
        km1 = KaplanMeier().fit(sd.subset(x == 1))
        assert km0._fitted and km1._fitted

        # Step 2: Log-rank test
        lr = LogRankTest.test(sd)
        assert lr.n_groups == 2

        # Step 3: Cox PH
        cox = CoxPH()
        result = cox.fit(sd)
        assert len(result.coefficients) == 1

        # Step 4: Predict
        hr = cox.predict_hazard_ratio(np.array([[0]]))
        assert hr[0] > 0


# =========================================================================
# Run all tests
# =========================================================================

def run_tests():
    test_classes = [
        TestSurvivalData, TestKaplanMeier, TestNelsonAalen,
        TestLogRankTest, TestCoxPH, TestExponentialModel,
        TestWeibullModel, TestLogNormalModel, TestLogLogisticModel,
        TestLifeTable, TestSchoenfeldTest, TestCumulativeIncidence,
        TestIntegration
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS: {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  FAIL: {cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailures:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
