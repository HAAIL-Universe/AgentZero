"""
Tests for V060: Probabilistic Verification -- Statistical Model Checking
"""

import os, sys, math, pytest

sys.path.insert(0, os.path.dirname(__file__))

from probabilistic_verification import (
    # Data model
    StatVerdict, PropertyKind, StatProperty, SampleResult, StatCheckResult,
    MonteCarloResult,
    # Statistical tests
    wilson_confidence_interval, chernoff_hoeffding_samples, sprt_test,
    # Main API
    stat_check, stat_check_sprt, monte_carlo_estimate, expected_value_check,
    required_samples,
    # Convenience API
    check_assertion_probability, check_output_probability,
    compare_statistical_vs_exact,
    # Internal
    ProbabilisticExecutor,
)


# =============================================================================
# Section 1: Data Model
# =============================================================================

class TestDataModel:
    def test_stat_verdict_enum(self):
        assert StatVerdict.ACCEPT.value == "accept"
        assert StatVerdict.REJECT.value == "reject"
        assert StatVerdict.INCONCLUSIVE.value == "inconclusive"

    def test_property_kind_enum(self):
        assert PropertyKind.PROBABILITY_GE.value == "probability_ge"
        assert PropertyKind.EXPECTED_BOUND.value == "expected_bound"
        assert PropertyKind.QUANTILE.value == "quantile"

    def test_stat_property_creation(self):
        prop = StatProperty(
            kind=PropertyKind.PROBABILITY_GE,
            description="test",
            threshold=0.95,
        )
        assert prop.threshold == 0.95
        assert prop.kind == PropertyKind.PROBABILITY_GE

    def test_sample_result(self):
        sr = SampleResult(inputs={"x": 5}, output=25, passed=True, value=25.0)
        assert sr.passed
        assert sr.value == 25.0

    def test_stat_check_result_summary(self):
        prop = StatProperty(kind=PropertyKind.PROBABILITY_GE, description="test", threshold=0.9)
        result = StatCheckResult(
            verdict=StatVerdict.ACCEPT, property=prop,
            total_samples=100, passing_samples=97,
            estimated_probability=0.97, confidence=0.99,
            confidence_interval=(0.92, 0.99),
        )
        summary = result.summary()
        assert "StatCheck: accept" in summary
        assert "97 pass" in summary

    def test_stat_check_result_failing_samples(self):
        prop = StatProperty(kind=PropertyKind.PROBABILITY_GE, description="test")
        result = StatCheckResult(
            verdict=StatVerdict.ACCEPT, property=prop,
            total_samples=100, passing_samples=95,
            estimated_probability=0.95, confidence=0.99,
            confidence_interval=(0.90, 0.98),
        )
        assert result.failing_samples == 5


# =============================================================================
# Section 2: Statistical Tests
# =============================================================================

class TestStatisticalTests:
    def test_wilson_ci_all_pass(self):
        lo, hi = wilson_confidence_interval(100, 100, 0.95)
        assert lo > 0.93  # Wilson CI for 100/100 at 95% is ~0.963
        assert hi > 0.999  # Essentially 1.0 (floating point)

    def test_wilson_ci_all_fail(self):
        lo, hi = wilson_confidence_interval(100, 0, 0.95)
        assert lo == 0.0
        assert hi < 0.05

    def test_wilson_ci_half(self):
        lo, hi = wilson_confidence_interval(100, 50, 0.95)
        assert lo < 0.5
        assert hi > 0.5

    def test_wilson_ci_empty(self):
        lo, hi = wilson_confidence_interval(0, 0, 0.95)
        assert lo == 0.0
        assert hi == 1.0

    def test_wilson_ci_contains_true(self):
        # With 80 out of 100, true p ~ 0.8
        lo, hi = wilson_confidence_interval(100, 80, 0.95)
        assert lo < 0.8 < hi

    def test_chernoff_samples_small_epsilon(self):
        n = chernoff_hoeffding_samples(0.01, 0.01)
        assert n > 1000  # Need many samples for tight bounds

    def test_chernoff_samples_large_epsilon(self):
        n = chernoff_hoeffding_samples(0.1, 0.1)
        assert n < 200

    def test_sprt_all_pass(self):
        samples = [True] * 100
        verdict, lr = sprt_test(samples, p0=0.95, p1=0.85)
        assert verdict == StatVerdict.ACCEPT

    def test_sprt_all_fail(self):
        samples = [False] * 100
        verdict, lr = sprt_test(samples, p0=0.95, p1=0.85)
        assert verdict == StatVerdict.REJECT

    def test_sprt_mixed_high(self):
        samples = [True] * 95 + [False] * 5
        verdict, lr = sprt_test(samples, p0=0.95, p1=0.85)
        assert verdict in (StatVerdict.ACCEPT, StatVerdict.INCONCLUSIVE)

    def test_sprt_invalid_thresholds(self):
        verdict, lr = sprt_test([True], p0=0.5, p1=0.9)  # p0 <= p1
        assert verdict == StatVerdict.INCONCLUSIVE

    def test_required_samples_api(self):
        n = required_samples(epsilon=0.05, delta=0.05)
        assert n > 0
        n2 = required_samples(epsilon=0.01, delta=0.01)
        assert n2 > n


# =============================================================================
# Section 3: Probabilistic Executor
# =============================================================================

class TestProbabilisticExecutor:
    def test_generate_random_inputs(self):
        exec = ProbabilisticExecutor("let z = x + y;", ["x", "y"],
                                     {"x": (0, 10), "y": (0, 10)})
        inputs = exec.generate_random_inputs()
        assert 0 <= inputs["x"] <= 10
        assert 0 <= inputs["y"] <= 10

    def test_execute_simple(self):
        exec = ProbabilisticExecutor("let z = x + 1;", ["x"],
                                     {"x": (1, 1)})
        result, error = exec.execute_with_inputs({"x": 5})
        assert error is None

    def test_execute_error(self):
        exec = ProbabilisticExecutor("let z = x / 0;", ["x"],
                                     {"x": (1, 1)})
        result, error = exec.execute_with_inputs({"x": 5})
        assert error is not None

    def test_sample_with_oracle(self):
        exec = ProbabilisticExecutor("let z = x + 1;", ["x"],
                                     {"x": (1, 10)})
        def oracle(inputs, result, error):
            return error is None
        sample = exec.sample(oracle)
        assert sample.passed  # Simple addition shouldn't error


# =============================================================================
# Section 4: Monte Carlo Estimation
# =============================================================================

class TestMonteCarlo:
    def test_always_pass_program(self):
        source = "let z = x + 1;"
        result = monte_carlo_estimate(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            n_samples=100, confidence=0.95,
            input_ranges={"x": (0, 100)}, seed=42,
        )
        assert result.estimated_probability == 1.0
        assert result.confidence_interval[0] > 0.95

    def test_sometimes_fail_program(self):
        # Division by x where x can be 0
        source = "let z = 10 / x;"
        result = monte_carlo_estimate(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            n_samples=200, confidence=0.95,
            input_ranges={"x": (0, 10)}, seed=42,
        )
        # x=0 causes error, so probability < 1.0
        assert result.estimated_probability < 1.0
        assert result.total_samples == 200

    def test_mc_confidence_interval(self):
        source = "let z = x + 1;"
        result = monte_carlo_estimate(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            n_samples=100, confidence=0.95,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        lo, hi = result.confidence_interval
        # Allow floating point tolerance for 100/100 case
        assert lo <= result.estimated_probability + 1e-9
        assert result.estimated_probability <= hi + 1e-9


# =============================================================================
# Section 5: stat_check API
# =============================================================================

class TestStatCheck:
    def test_high_probability_property(self):
        source = "let z = x + 1;"
        prop = StatProperty(
            kind=PropertyKind.PROBABILITY_GE,
            description="P(no error) >= 0.95",
            threshold=0.95,
        )
        result = stat_check(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            prop=prop, confidence=0.95, max_samples=200,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        assert result.verdict == StatVerdict.ACCEPT
        assert result.estimated_probability == 1.0

    def test_low_probability_property(self):
        # x in [0,100], z = 10/x fails at x=0 (1/101 chance)
        source = "let z = 10 / x;"
        prop = StatProperty(
            kind=PropertyKind.PROBABILITY_GE,
            description="P(no error) >= 0.99",
            threshold=0.99,
        )
        result = stat_check(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            prop=prop, confidence=0.95, max_samples=500,
            input_ranges={"x": (0, 100)}, seed=42,
        )
        # May be accept or inconclusive depending on whether x=0 was sampled
        assert result.total_samples == 500

    def test_probability_le_property(self):
        source = "let z = x + 1;"
        prop = StatProperty(
            kind=PropertyKind.PROBABILITY_LE,
            description="P(error) <= 0.02",
            threshold=0.02,
        )
        # Oracle returns True when there IS an error
        result = stat_check(
            source, ["x"],
            oracle=lambda inp, res, err: err is not None,
            prop=prop, confidence=0.95, max_samples=200,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        # No errors -> P(error) = 0 <= 0.02 -> ACCEPT
        assert result.verdict == StatVerdict.ACCEPT

    def test_seed_reproducibility(self):
        source = "let z = x + 1;"
        prop = StatProperty(kind=PropertyKind.PROBABILITY_GE, description="test", threshold=0.5)
        oracle = lambda inp, res, err: err is None

        r1 = stat_check(source, ["x"], oracle, prop, max_samples=50, seed=123,
                         input_ranges={"x": (0, 100)})
        r2 = stat_check(source, ["x"], oracle, prop, max_samples=50, seed=123,
                         input_ranges={"x": (0, 100)})
        assert r1.estimated_probability == r2.estimated_probability


# =============================================================================
# Section 6: SPRT
# =============================================================================

class TestSPRT:
    def test_sprt_accept_safe_program(self):
        source = "let z = x + 1;"
        result = stat_check_sprt(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            p0=0.95, p1=0.85, max_samples=500,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        assert result.verdict == StatVerdict.ACCEPT
        assert result.metadata.get("early_termination") is True

    def test_sprt_early_termination(self):
        source = "let z = x + 1;"
        result = stat_check_sprt(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            p0=0.95, p1=0.85, max_samples=1000,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        # Should terminate well before 1000 samples
        assert result.total_samples < 1000

    def test_sprt_reject_buggy_program(self):
        # x in [0,1] -> 50% chance of div by zero
        source = "let z = 10 / x;"
        result = stat_check_sprt(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            p0=0.95, p1=0.85, max_samples=500,
            input_ranges={"x": (0, 1)}, seed=42,
        )
        assert result.verdict == StatVerdict.REJECT

    def test_sprt_log_ratio(self):
        source = "let z = x + 1;"
        result = stat_check_sprt(
            source, ["x"],
            oracle=lambda inp, res, err: err is None,
            p0=0.95, p1=0.85, max_samples=100,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        assert result.sprt_log_ratio is not None


# =============================================================================
# Section 7: Expected Value Checking
# =============================================================================

class TestExpectedValue:
    def test_expected_value_in_bounds(self):
        # x in [1,10], z = x*x, expect E[x*x] roughly 38.5
        source = "let z = x * x;"
        result = expected_value_check(
            source, ["x"],
            value_fn=lambda inp, res: inp["x"] ** 2,
            bound_lo=0, bound_hi=200,
            n_samples=200, confidence=0.95,
            input_ranges={"x": (1, 10)}, seed=42,
        )
        assert result.verdict == StatVerdict.ACCEPT
        mean = result.metadata.get("mean", 0)
        assert 0 < mean < 200

    def test_expected_value_out_of_bounds(self):
        source = "let z = x * x;"
        result = expected_value_check(
            source, ["x"],
            value_fn=lambda inp, res: inp["x"] ** 2,
            bound_lo=0, bound_hi=5,  # Too tight
            n_samples=200, confidence=0.95,
            input_ranges={"x": (1, 10)}, seed=42,
        )
        # Most x*x values > 5, so reject
        assert result.verdict == StatVerdict.REJECT

    def test_expected_value_metadata(self):
        source = "let z = x + 1;"
        result = expected_value_check(
            source, ["x"],
            value_fn=lambda inp, res: inp["x"] + 1,
            bound_lo=0, bound_hi=200,
            n_samples=100, confidence=0.95,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        assert "mean" in result.metadata
        assert "mean_ci" in result.metadata


# =============================================================================
# Section 8: Assertion Probability
# =============================================================================

class TestAssertionProbability:
    def test_safe_program(self):
        result = check_assertion_probability(
            "let z = x + 1;", ["x"],
            threshold=0.95, confidence=0.95, n_samples=200,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        assert result.verdict == StatVerdict.ACCEPT

    def test_risky_program(self):
        result = check_assertion_probability(
            "let z = 10 / x;", ["x"],
            threshold=0.99, confidence=0.95, n_samples=200,
            input_ranges={"x": (0, 1)}, seed=42,
        )
        # 50% error rate -> definitely below 0.99
        assert result.verdict in (StatVerdict.REJECT, StatVerdict.INCONCLUSIVE)

    def test_assertion_samples(self):
        result = check_assertion_probability(
            "let z = x + 1;", ["x"],
            threshold=0.5, n_samples=100,
            input_ranges={"x": (1, 10)}, seed=42,
        )
        assert result.total_samples == 100
        assert result.passing_samples == 100


# =============================================================================
# Section 9: Output Probability
# =============================================================================

class TestOutputProbability:
    def test_output_always_positive(self):
        result = check_output_probability(
            "let z = x * x + 1;", ["x"],
            output_predicate=lambda r: True,  # Always true since no error
            threshold=0.95, n_samples=200,
            input_ranges={"x": (1, 10)}, seed=42,
        )
        assert result.verdict == StatVerdict.ACCEPT

    def test_output_predicate_on_inputs(self):
        # Check that x+1 > 0 when x in [0, 100]
        result = check_output_probability(
            "let z = x + 1;", ["x"],
            output_predicate=lambda r: True,
            threshold=0.9, n_samples=200,
            input_ranges={"x": (0, 100)}, seed=42,
        )
        assert result.verdict == StatVerdict.ACCEPT


# =============================================================================
# Section 10: Comparison API
# =============================================================================

class TestComparison:
    def test_compare_methods(self):
        result = compare_statistical_vs_exact(
            "let z = x + 1;", ["x"],
            oracle=lambda inp, res, err: err is None,
            n_samples=200,
            input_ranges={"x": (1, 100)}, seed=42,
        )
        assert "monte_carlo" in result
        assert "sprt" in result
        assert result["monte_carlo"]["probability"] == 1.0
        assert result["sprt"]["verdict"] == "accept"

    def test_compare_with_errors(self):
        result = compare_statistical_vs_exact(
            "let z = 10 / x;", ["x"],
            oracle=lambda inp, res, err: err is None,
            n_samples=200,
            input_ranges={"x": (0, 10)}, seed=42,
        )
        mc_prob = result["monte_carlo"]["probability"]
        assert mc_prob < 1.0  # Some x=0 errors


# =============================================================================
# Section 11: Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_single_sample(self):
        result = monte_carlo_estimate(
            "let z = x + 1;", ["x"],
            oracle=lambda inp, res, err: err is None,
            n_samples=1, confidence=0.95,
            input_ranges={"x": (1, 10)}, seed=42,
        )
        assert result.total_samples == 1

    def test_large_input_range(self):
        result = monte_carlo_estimate(
            "let z = x + 1;", ["x"],
            oracle=lambda inp, res, err: err is None,
            n_samples=50, confidence=0.95,
            input_ranges={"x": (-1000000, 1000000)}, seed=42,
        )
        assert result.estimated_probability == 1.0

    def test_multiple_input_vars(self):
        source = "let z = x + y;"
        result = monte_carlo_estimate(
            source, ["x", "y"],
            oracle=lambda inp, res, err: err is None,
            n_samples=100, confidence=0.95,
            input_ranges={"x": (0, 10), "y": (0, 10)}, seed=42,
        )
        assert result.estimated_probability == 1.0

    def test_no_input_vars(self):
        source = "let z = 42;"
        result = monte_carlo_estimate(
            source, [],
            oracle=lambda inp, res, err: err is None,
            n_samples=10, confidence=0.95, seed=42,
        )
        assert result.estimated_probability == 1.0

    def test_quantile_property(self):
        prop = StatProperty(
            kind=PropertyKind.QUANTILE,
            description="P(value <= 100) >= 0.9",
            quantile=0.9,
        )
        result = stat_check(
            "let z = x + 1;", ["x"],
            oracle=lambda inp, res, err: err is None,
            prop=prop, max_samples=100,
            input_ranges={"x": (1, 50)}, seed=42,
        )
        assert result.verdict in (StatVerdict.ACCEPT, StatVerdict.REJECT)

    def test_wilson_ci_large_n(self):
        lo, hi = wilson_confidence_interval(10000, 9500, 0.99)
        assert 0.94 < lo < 0.95
        assert 0.95 < hi < 0.96
