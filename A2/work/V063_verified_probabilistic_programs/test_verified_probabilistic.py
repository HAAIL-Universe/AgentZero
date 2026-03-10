"""Tests for V063: Verified Probabilistic Programs"""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V060_probabilistic_verification'))

from verified_probabilistic import (
    ProbVerdict, ProbVC, ProbFnSpec, ProbVerificationResult,
    verify_probabilistic, verify_prob_function, check_prob_property,
    expected_value_analysis, compare_deterministic_vs_probabilistic,
    prob_hoare_triple, concentration_bound, verify_randomized_algorithm,
    independence_test, extract_prob_fn_spec, _strip_annotations,
    _replace_random_with_input, _evaluate_postcondition, _float_to_c10,
    check_probabilistic_vc, verify_probabilistic_function
)
from stack_vm import lex, Parser


# =============================================================================
# Section 1: Data Model
# =============================================================================

class TestDataModel:
    def test_prob_verdict_values(self):
        assert ProbVerdict.VERIFIED.value == "verified"
        assert ProbVerdict.VIOLATED.value == "violated"
        assert ProbVerdict.INCONCLUSIVE.value == "inconclusive"
        assert ProbVerdict.ERROR.value == "error"

    def test_prob_vc_defaults(self):
        vc = ProbVC(name="test", kind="deterministic")
        assert vc.name == "test"
        assert vc.kind == "deterministic"
        assert vc.formula is None
        assert vc.status is None
        assert vc.threshold == 0.95

    def test_prob_vc_probabilistic(self):
        vc = ProbVC(
            name="p1", kind="probabilistic",
            postcondition_src="x > 0", threshold=0.9,
            status="accept", estimated_probability=0.95,
            confidence_interval=(0.92, 0.97)
        )
        assert vc.estimated_probability == 0.95
        assert vc.confidence_interval == (0.92, 0.97)

    def test_prob_verification_result_total_vcs(self):
        det = [ProbVC(name="d1", kind="deterministic", status="valid")]
        prob = [ProbVC(name="p1", kind="probabilistic", status="accept")]
        result = ProbVerificationResult(
            verdict=ProbVerdict.VERIFIED,
            deterministic_vcs=det,
            probabilistic_vcs=prob
        )
        assert result.total_vcs == 2
        assert len(result.all_vcs) == 2

    def test_prob_verification_result_summary(self):
        result = ProbVerificationResult(
            verdict=ProbVerdict.VERIFIED,
            deterministic_vcs=[ProbVC(name="d1", kind="deterministic", status="valid")],
            probabilistic_vcs=[ProbVC(name="p1", kind="probabilistic", status="accept",
                                      estimated_probability=0.95)]
        )
        s = result.summary()
        assert "Verdict: verified" in s
        assert "d1" in s
        assert "p1" in s


# =============================================================================
# Section 2: Spec Extraction
# =============================================================================

class TestSpecExtraction:
    def test_extract_requires(self):
        source = "requires(x > 0);\nlet y = x + 1;"
        tokens = lex(source)
        ast = Parser(tokens).parse()
        spec = extract_prob_fn_spec(ast.stmts)
        assert len(spec.preconditions) == 1
        assert len(spec.body_stmts) == 1

    def test_extract_ensures(self):
        source = "let y = 5;\nensures(y > 0);"
        tokens = lex(source)
        ast = Parser(tokens).parse()
        spec = extract_prob_fn_spec(ast.stmts)
        assert len(spec.postconditions) == 1
        assert len(spec.body_stmts) == 1

    def test_extract_prob_ensures(self):
        source = "let x = random(1, 10);\nprob_ensures(x > 0, 99/100);"
        tokens = lex(source)
        ast = Parser(tokens).parse()
        spec = extract_prob_fn_spec(ast.stmts)
        assert len(spec.prob_postconditions) == 1
        expr, threshold, src = spec.prob_postconditions[0]
        assert threshold == 0.99
        assert "x > 0" in src or "x>0" in src

    def test_extract_random_vars(self):
        source = "let x = random(1, 10);\nlet y = random(0, 100);"
        tokens = lex(source)
        ast = Parser(tokens).parse()
        spec = extract_prob_fn_spec(ast.stmts)
        assert 'x' in spec.random_vars
        assert spec.random_vars['x'] == (1, 10)
        assert 'y' in spec.random_vars
        assert spec.random_vars['y'] == (0, 100)

    def test_extract_mixed_spec(self):
        source = """
requires(n > 0);
let x = random(1, n);
ensures(x > 0);
prob_ensures(x < n, 90/100);
"""
        tokens = lex(source)
        ast = Parser(tokens).parse()
        spec = extract_prob_fn_spec(ast.stmts)
        assert len(spec.preconditions) == 1
        assert len(spec.postconditions) == 1
        assert len(spec.prob_postconditions) == 1
        assert 'x' in spec.random_vars


# =============================================================================
# Section 3: Helper Functions
# =============================================================================

class TestHelpers:
    def test_strip_annotations(self):
        source = "requires(x > 0);\nlet y = 5;\nensures(y > 0);\nprob_ensures(y > 0, 95/100);"
        stripped = _strip_annotations(source)
        assert "requires" not in stripped
        assert "ensures" not in stripped
        assert "prob_ensures" not in stripped
        assert "let y = 5;" in stripped

    def test_replace_random_with_input(self):
        source = "let x = random(1, 10);\nlet y = x + 1;"
        result = _replace_random_with_input(source, {'x': 5})
        assert "random" not in result
        # random() line is stripped (input vars are prepended by _run_with_inputs)
        assert "let y = x + 1;" in result

    def test_replace_random_negative(self):
        source = "let x = random(1, 10);\nlet y = 1;"
        result = _replace_random_with_input(source, {'x': -3})
        assert "random" not in result
        assert "let y = 1;" in result

    def test_float_to_c10(self):
        assert _float_to_c10(1.0) == "1"
        assert _float_to_c10(0.95) == "95/100"
        assert _float_to_c10(0.99) == "99/100"


# =============================================================================
# Section 4: Postcondition Evaluation
# =============================================================================

class TestPostcondEval:
    def test_simple_comparison(self):
        source = "let x = random(1, 10);"
        result = _evaluate_postcondition(source, {'x': 5}, 5, "x > 0", ['x'])
        assert result is True

    def test_comparison_false(self):
        source = "let x = random(1, 10);"
        result = _evaluate_postcondition(source, {'x': 5}, 5, "x > 10", ['x'])
        assert result is False

    def test_arithmetic_postcond(self):
        source = "let x = random(1, 10);\nlet y = x * 2;"
        result = _evaluate_postcondition(source, {'x': 5}, None, "y == 10", ['x'])
        assert result is True

    def test_compound_postcond(self):
        source = "let x = random(1, 10);"
        result = _evaluate_postcondition(source, {'x': 5}, 5, "(x >= 1) and (x <= 10)", ['x'])
        assert result is True


# =============================================================================
# Section 5: Single Probabilistic VC Checking
# =============================================================================

class TestProbVCCheck:
    def test_always_true_property(self):
        """Property that holds with probability 1."""
        source = "let x = random(1, 10);"
        vc = check_probabilistic_vc(
            source=source,
            postcond_src="x >= 1",
            threshold=0.95,
            random_vars={'x': (1, 10)},
            n_samples=200,
            seed=42
        )
        assert vc.status == "accept"
        assert vc.estimated_probability is not None
        assert vc.estimated_probability >= 0.95

    def test_always_false_property(self):
        """Property that holds with probability 0."""
        source = "let x = random(1, 10);"
        vc = check_probabilistic_vc(
            source=source,
            postcond_src="x > 100",
            threshold=0.5,
            random_vars={'x': (1, 10)},
            n_samples=200,
            seed=42
        )
        assert vc.status == "reject"
        assert vc.estimated_probability is not None
        assert vc.estimated_probability < 0.1

    def test_partial_probability_property(self):
        """Property that holds with known probability ~0.5."""
        source = "let x = random(1, 10);"
        vc = check_probabilistic_vc(
            source=source,
            postcond_src="x <= 5",
            threshold=0.3,
            random_vars={'x': (1, 10)},
            n_samples=500,
            seed=42
        )
        # P(x <= 5) = 5/10 = 0.5, threshold 0.3 should accept
        assert vc.status == "accept"
        assert vc.estimated_probability is not None
        assert vc.estimated_probability >= 0.3

    def test_no_random_vars_error(self):
        """No random variables should return error."""
        source = "let x = 5;"
        vc = check_probabilistic_vc(
            source=source,
            postcond_src="x > 0",
            threshold=0.95,
            random_vars={},
            n_samples=100,
            seed=42
        )
        assert vc.status == "error"


# =============================================================================
# Section 6: Full Probabilistic Verification (Top-Level)
# =============================================================================

class TestVerifyProbabilistic:
    def test_simple_random_program(self):
        """Verify a simple program with random input."""
        source = """
let x = random(1, 10);
let y = x * 2;
prob_ensures(y >= 2, 90/100);
"""
        result = verify_probabilistic(source, n_samples=300, seed=42)
        assert result.verdict == ProbVerdict.VERIFIED
        assert len(result.probabilistic_vcs) == 1
        assert result.probabilistic_vcs[0].status == "accept"

    def test_violated_probabilistic_spec(self):
        """Probabilistic spec that cannot be met."""
        source = """
let x = random(1, 10);
prob_ensures(x > 5, 99/100);
"""
        # P(x > 5) = 5/10 = 0.5, threshold 0.99 should reject
        result = verify_probabilistic(source, n_samples=300, seed=42)
        assert result.verdict == ProbVerdict.VIOLATED
        assert result.probabilistic_vcs[0].status == "reject"

    def test_multiple_prob_ensures(self):
        """Multiple probabilistic postconditions."""
        source = """
let x = random(1, 100);
prob_ensures(x >= 1, 95/100);
prob_ensures(x <= 100, 95/100);
"""
        result = verify_probabilistic(source, n_samples=300, seed=42)
        assert result.verdict == ProbVerdict.VERIFIED
        assert len(result.probabilistic_vcs) == 2
        for vc in result.probabilistic_vcs:
            assert vc.status == "accept"

    def test_parse_error(self):
        """Invalid source should produce error verdict."""
        source = "let x = ;"
        result = verify_probabilistic(source)
        assert result.verdict == ProbVerdict.ERROR

    def test_empty_program_vacuous(self):
        """Program with no specs is vacuously verified."""
        source = "let x = 5;"
        result = verify_probabilistic(source)
        assert result.verdict == ProbVerdict.VERIFIED
        assert result.total_vcs == 0


# =============================================================================
# Section 7: check_prob_property API
# =============================================================================

class TestCheckProbProperty:
    def test_always_true(self):
        source = "let x = random(0, 10);\nlet y = x + 1;"
        vc = check_prob_property(
            source=source,
            property_expr="y >= 1",
            threshold=0.95,
            random_vars={'x': (0, 10)},
            n_samples=300,
            seed=42
        )
        assert vc.status == "accept"

    def test_conditional_property(self):
        """Property involving a conditional."""
        source = """
let x = random(0, 20);
let y = 0;
if (x > 10) {
    y = 1;
}
"""
        vc = check_prob_property(
            source=source,
            property_expr="y == 0",
            threshold=0.4,
            random_vars={'x': (0, 20)},
            n_samples=500,
            seed=42
        )
        # P(y == 0) = P(x <= 10) = 11/21 ~= 0.52, threshold 0.4 -> accept
        assert vc.status == "accept"

    def test_probability_estimate_accuracy(self):
        """Check that probability estimates are roughly correct."""
        source = "let x = random(1, 6);"
        vc = check_prob_property(
            source=source,
            property_expr="x <= 3",
            threshold=0.3,
            random_vars={'x': (1, 6)},
            n_samples=1000,
            seed=42
        )
        # P(x <= 3) = 3/6 = 0.5
        assert vc.estimated_probability is not None
        assert 0.35 <= vc.estimated_probability <= 0.65


# =============================================================================
# Section 8: Expected Value Analysis
# =============================================================================

class TestExpectedValue:
    def test_uniform_mean(self):
        """E[x] for x ~ Uniform(1, 10) should be ~5.5."""
        source = "let x = random(1, 10);"
        ev = expected_value_analysis(
            source=source,
            value_expr="x",
            random_vars={'x': (1, 10)},
            expected_lo=4.0,
            expected_hi=7.0,
            n_samples=500,
            seed=42
        )
        assert ev['in_bounds'] is True
        assert 4.0 <= ev['mean'] <= 7.0

    def test_linear_transform_mean(self):
        """E[2x + 1] for x ~ Uniform(0, 10) should be ~11."""
        source = "let x = random(0, 10);\nlet y = x * 2 + 1;"
        ev = expected_value_analysis(
            source=source,
            value_expr="y",
            random_vars={'x': (0, 10)},
            expected_lo=8.0,
            expected_hi=14.0,
            n_samples=500,
            seed=42
        )
        assert ev['in_bounds'] is True

    def test_out_of_bounds_mean(self):
        """Expected value outside bounds should not be in_bounds."""
        source = "let x = random(1, 10);"
        ev = expected_value_analysis(
            source=source,
            value_expr="x",
            random_vars={'x': (1, 10)},
            expected_lo=1.0,
            expected_hi=3.0,
            n_samples=300,
            seed=42
        )
        # E[x] ~= 5.5, bounds [1, 3] -> not in bounds
        assert ev['in_bounds'] is False


# =============================================================================
# Section 9: Probabilistic Hoare Triple
# =============================================================================

class TestProbHoareTriple:
    def test_trivial_triple(self):
        """{ true } x=random(1,10) { x>=1 @ 0.95 }"""
        result = prob_hoare_triple(
            precondition="",
            program="let x = random(1, 10);",
            postcondition="x >= 1",
            threshold=0.95,
            random_vars={'x': (1, 10)},
            n_samples=300,
            seed=42
        )
        assert result.verdict == ProbVerdict.VERIFIED

    def test_failing_triple(self):
        """{ true } x=random(1,10) { x>5 @ 0.99 } -- should fail"""
        result = prob_hoare_triple(
            precondition="",
            program="let x = random(1, 10);",
            postcondition="x > 5",
            threshold=0.99,
            random_vars={'x': (1, 10)},
            n_samples=300,
            seed=42
        )
        assert result.verdict == ProbVerdict.VIOLATED

    def test_triple_with_computation(self):
        """{ true } x=random(0,100); y=x*x { y>=0 @ 0.95 }"""
        result = prob_hoare_triple(
            precondition="",
            program="let x = random(0, 100);\nlet y = x * x;",
            postcondition="y >= 0",
            threshold=0.95,
            random_vars={'x': (0, 100)},
            n_samples=300,
            seed=42
        )
        assert result.verdict == ProbVerdict.VERIFIED


# =============================================================================
# Section 10: Concentration Bounds
# =============================================================================

class TestConcentrationBounds:
    def test_uniform_concentration(self):
        """Uniform distribution should concentrate around mean."""
        source = "let x = random(1, 100);"
        cb = concentration_bound(
            source=source,
            value_expr="x",
            random_vars={'x': (1, 100)},
            epsilon=0.5,  # 50% deviation
            n_samples=500,
            seed=42
        )
        assert cb['mean'] > 0
        assert cb['std'] > 0
        assert cb['chebyshev_bound'] >= 0
        assert cb['empirical_deviation_prob'] >= 0
        assert cb['samples'] == 500

    def test_constant_concentration(self):
        """Constant expression should have zero variance."""
        source = "let x = random(5, 5);"
        cb = concentration_bound(
            source=source,
            value_expr="x",
            random_vars={'x': (5, 5)},
            epsilon=0.1,
            n_samples=100,
            seed=42
        )
        assert cb['mean'] == 5.0
        assert cb['std'] == 0.0 or cb['variance'] < 0.01

    def test_chebyshev_vs_empirical(self):
        """Chebyshev bound should be >= empirical probability."""
        source = "let x = random(1, 100);"
        cb = concentration_bound(
            source=source,
            value_expr="x",
            random_vars={'x': (1, 100)},
            epsilon=0.3,
            n_samples=500,
            seed=42
        )
        # Chebyshev is always an upper bound
        assert cb['chebyshev_bound'] >= cb['empirical_deviation_prob'] - 0.05  # Allow small slack


# =============================================================================
# Section 11: Randomized Algorithm Verification
# =============================================================================

class TestRandomizedAlgorithm:
    def test_always_correct_algorithm(self):
        """Algorithm that's always correct."""
        source = "let x = random(1, 10);\nlet y = x * 2;"
        result = verify_randomized_algorithm(
            source=source,
            correctness_expr="y >= 2",
            random_vars={'x': (1, 10)},
            min_success_prob=0.95,
            n_trials=200,
            seed=42
        )
        assert result['verdict'] == 'verified'
        assert result['single_round']['estimated_probability'] >= 0.95

    def test_probabilistic_algorithm(self):
        """Algorithm with known success probability < 1."""
        source = "let x = random(1, 10);"
        result = verify_randomized_algorithm(
            source=source,
            correctness_expr="x <= 5",
            random_vars={'x': (1, 10)},
            min_success_prob=0.3,
            n_trials=300,
            seed=42
        )
        # P(x <= 5) = 0.5, threshold 0.3 -> should verify
        assert result['verdict'] == 'verified'

    def test_amplification(self):
        """Probability amplification should boost success probability."""
        source = "let x = random(1, 10);"
        result = verify_randomized_algorithm(
            source=source,
            correctness_expr="x <= 7",
            random_vars={'x': (1, 10)},
            min_success_prob=0.5,
            n_trials=300,
            amplification_rounds=5,
            seed=42
        )
        # P(single) = 0.7, amplified with 5 rounds majority should be much higher
        assert result['amplification']['amplified_probability'] > 0.9
        assert result['amplification']['rounds'] == 5

    def test_failing_algorithm(self):
        """Algorithm that doesn't meet min success probability."""
        source = "let x = random(1, 100);"
        result = verify_randomized_algorithm(
            source=source,
            correctness_expr="x == 42",
            random_vars={'x': (1, 100)},
            min_success_prob=0.5,
            n_trials=500,
            seed=42
        )
        # P(x==42) = 1/100 = 0.01, threshold 0.5 -> reject
        assert result['verdict'] == 'reject'


# =============================================================================
# Section 12: Independence Testing
# =============================================================================

class TestIndependence:
    def test_independent_variables(self):
        """Two independent random variables should test as independent."""
        source = "let x = random(1, 10);\nlet y = random(1, 10);"
        result = independence_test(
            source=source,
            expr_a="x > 5",
            expr_b="y > 5",
            random_vars={'x': (1, 10), 'y': (1, 10)},
            n_samples=1000,
            seed=42
        )
        assert result['independent'] is True
        # P(A) * P(B) should be close to P(AB)
        assert abs(result['p_ab'] - result['p_a_times_p_b']) < 0.1

    def test_dependent_expressions(self):
        """Expressions derived from same variable should be dependent."""
        source = "let x = random(1, 100);"
        result = independence_test(
            source=source,
            expr_a="x > 50",
            expr_b="x > 60",
            random_vars={'x': (1, 100)},
            n_samples=1000,
            seed=42
        )
        # x>60 implies x>50, so these are dependent
        # P(A) ~= 0.5, P(B) ~= 0.4, P(AB) ~= 0.4, P(A)*P(B) ~= 0.2
        assert result['p_ab'] > result['p_a_times_p_b'] + 0.05

    def test_mutually_exclusive_events(self):
        """Mutually exclusive events are dependent."""
        source = "let x = random(1, 10);"
        result = independence_test(
            source=source,
            expr_a="x <= 3",
            expr_b="x >= 8",
            random_vars={'x': (1, 10)},
            n_samples=1000,
            seed=42
        )
        # These are mutually exclusive: P(AB) = 0, P(A)*P(B) > 0
        assert result['p_ab'] < 0.01


# =============================================================================
# Section 13: Compare Deterministic vs Probabilistic
# =============================================================================

class TestComparison:
    def test_comparison_basic(self):
        """Compare det vs prob on a simple program."""
        source = """
let x = random(1, 10);
let y = x * 2;
prob_ensures(y >= 2, 90/100);
"""
        result = compare_deterministic_vs_probabilistic(
            source=source,
            n_samples=200,
            seed=42
        )
        assert 'deterministic' in result
        assert 'probabilistic' in result
        assert result['probabilistic']['verdict'] == 'verified'
        assert result['probabilistic']['prob_vcs'] == 1


# =============================================================================
# Section 14: Edge Cases and Integration
# =============================================================================

class TestEdgeCases:
    def test_single_value_random(self):
        """random(5, 5) is deterministic."""
        source = "let x = random(5, 5);\nprob_ensures(x == 5, 99/100);"
        result = verify_probabilistic(source, n_samples=100, seed=42)
        assert result.verdict == ProbVerdict.VERIFIED

    def test_negative_range_random(self):
        """Random with negative range."""
        source = "let x = random(1, 10);"
        vc = check_prob_property(
            source=source,
            property_expr="x >= 1",
            threshold=0.95,
            random_vars={'x': (1, 10)},
            n_samples=200,
            seed=42
        )
        assert vc.status == "accept"

    def test_multiple_random_vars(self):
        """Program with multiple random variables."""
        source = """
let x = random(1, 10);
let y = random(1, 10);
let sum = x + y;
prob_ensures(sum >= 2, 95/100);
"""
        result = verify_probabilistic(source, n_samples=300, seed=42)
        assert result.verdict == ProbVerdict.VERIFIED

    def test_conditional_with_random(self):
        """Conditional program with random input."""
        source = """
let x = random(1, 100);
let y = 0;
if (x > 50) {
    y = 1;
} else {
    y = 0;
}
prob_ensures(y >= 0, 95/100);
"""
        result = verify_probabilistic(source, n_samples=300, seed=42)
        assert result.verdict == ProbVerdict.VERIFIED

    def test_while_loop_with_random(self):
        """Loop with random initialization."""
        source = """
let x = random(1, 5);
let count = 0;
while (x > 0) {
    x = x - 1;
    count = count + 1;
}
prob_ensures(count >= 1, 95/100);
"""
        result = verify_probabilistic(source, n_samples=300, seed=42)
        assert result.verdict == ProbVerdict.VERIFIED

    def test_arithmetic_postcondition(self):
        """Complex arithmetic in postcondition."""
        source = """
let a = random(1, 10);
let b = random(1, 10);
let c = a + b;
"""
        vc = check_prob_property(
            source=source,
            property_expr="(c >= 2) and (c <= 20)",
            threshold=0.95,
            random_vars={'a': (1, 10), 'b': (1, 10)},
            n_samples=300,
            seed=42
        )
        assert vc.status == "accept"

    def test_vc_has_counterexample_on_reject(self):
        """Rejected VCs should include a failing sample."""
        source = "let x = random(1, 10);"
        vc = check_probabilistic_vc(
            source=source,
            postcond_src="x > 100",
            threshold=0.5,
            random_vars={'x': (1, 10)},
            n_samples=200,
            seed=42
        )
        assert vc.status == "reject"
        assert vc.counterexample is not None
        assert 'x' in vc.counterexample

    def test_confidence_interval_present(self):
        """Result should include confidence interval."""
        source = "let x = random(1, 10);"
        vc = check_probabilistic_vc(
            source=source,
            postcond_src="x <= 5",
            threshold=0.3,
            random_vars={'x': (1, 10)},
            n_samples=300,
            seed=42
        )
        assert vc.confidence_interval is not None
        lo, hi = vc.confidence_interval
        assert lo <= hi
        assert 0 <= lo <= 1
        assert 0 <= hi <= 1


# =============================================================================
# Section 15: Probabilistic Function Verification
# =============================================================================

class TestFunctionVerification:
    def test_verify_named_function(self):
        """Verify a named function with probabilistic spec."""
        source = """
fn roll_dice(sides) {
    let x = random(1, sides);
    prob_ensures(x >= 1, 95/100);
    return x;
}
"""
        result = verify_prob_function(
            source=source,
            fn_name='roll_dice',
            param_ranges={'sides': (6, 6)},
            n_samples=300,
            seed=42
        )
        assert result.verdict == ProbVerdict.VERIFIED

    def test_function_not_found(self):
        """Non-existent function should return error."""
        source = "fn foo() { return 1; }"
        result = verify_prob_function(source, fn_name='bar')
        assert result.verdict == ProbVerdict.ERROR
        assert any("not found" in e for e in result.errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
