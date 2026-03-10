"""
Tests for V051: Counterexample-Guided Optimization Verification (CEGOV)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from cegov import (
    analyze_cegov, validate_and_diagnose, check_pass_with_cex,
    compare_optimization_behavior,
    CEGOVResult, CounterexampleDiag, GuidedTestCase, FailureKind,
    _extract_counterexample_inputs, _classify_counterexample,
    _execute_program, _execute_optimized,
    _make_test_program, _check_boundary_coverage,
    _generate_guided_tests, _cross_validate_counterexample,
)


# --- Test Programs ---

SIMPLE_ARITHMETIC = """
fn calc(x) {
    let a = x + 1;
    let b = a + 2;
    return b;
}
"""

CONSTANT_FOLDABLE = """
fn const_calc() {
    let a = 2 + 3;
    let b = a * 2;
    return b;
}
"""

WITH_LOOP = """
fn sum_to(n) {
    let s = 0;
    let i = 0;
    while (i < n) {
        s = s + i;
        i = i + 1;
    }
    return s;
}
"""

SIMPLE_CONDITIONAL = """
fn max(a, b) {
    let r = a;
    if (b > a) {
        r = b;
    }
    return r;
}
"""

IDENTITY = """
fn id(x) {
    return x;
}
"""

STRENGTH_REDUCIBLE = """
fn double(x) {
    return x * 2;
}
"""

DEAD_CODE = """
fn f(x) {
    return x;
    let y = 42;
}
"""

MULTI_OP = """
fn compute(x) {
    let a = x + 0;
    let b = a * 1;
    let c = b + b;
    return c;
}
"""

TRIVIAL = """
let x = 42;
"""


# === Section 1: Data Types ===

class TestFailureKind(unittest.TestCase):
    def test_all_kinds_exist(self):
        assert FailureKind.CONFIRMED_BUG.value == "confirmed_bug"
        assert FailureKind.SPURIOUS.value == "spurious"
        assert FailureKind.BOUNDARY_CASE.value == "boundary_case"
        assert FailureKind.UNREACHABLE.value == "unreachable"
        assert FailureKind.UNKNOWN.value == "unknown"


class TestCEGOVResult(unittest.TestCase):
    def test_empty_result(self):
        r = CEGOVResult(source="", validation_passed=True)
        assert r.has_real_bugs is False
        assert r.all_spurious is False

    def test_with_confirmed_bug(self):
        r = CEGOVResult(source="", validation_passed=False,
                        confirmed_bugs=1, failed_obligations=1)
        assert r.has_real_bugs is True
        assert r.all_spurious is False

    def test_all_spurious(self):
        r = CEGOVResult(source="", validation_passed=False,
                        confirmed_bugs=0, spurious_failures=2, failed_obligations=2)
        assert r.has_real_bugs is False
        assert r.all_spurious is True

    def test_summary_string(self):
        r = CEGOVResult(source="test", validation_passed=True,
                        total_obligations=5, failed_obligations=0)
        s = r.summary()
        assert "CEGOV Report" in s
        assert "PASSED" in s


class TestCounterexampleDiag(unittest.TestCase):
    def test_creation(self):
        d = CounterexampleDiag(
            pass_name="constant_fold",
            obligation_name="test_ob",
            raw_counterexample={"x": 5},
            concrete_inputs={"x": 5},
        )
        assert d.pass_name == "constant_fold"
        assert d.kind == FailureKind.UNKNOWN

    def test_with_results(self):
        d = CounterexampleDiag(
            pass_name="test", obligation_name="ob",
            raw_counterexample={}, concrete_inputs={},
            results_match=True, kind=FailureKind.SPURIOUS,
        )
        assert d.results_match is True


class TestGuidedTestCase(unittest.TestCase):
    def test_creation(self):
        t = GuidedTestCase(inputs={"x": 5})
        assert t.covers_boundary is False

    def test_boundary_flag(self):
        t = GuidedTestCase(inputs={"x": 5}, covers_boundary=True)
        assert t.covers_boundary is True


# === Section 2: Helper Functions ===

class TestExtractInputs(unittest.TestCase):
    def test_empty(self):
        assert _extract_counterexample_inputs({}) == {}

    def test_none(self):
        assert _extract_counterexample_inputs(None) == {}

    def test_filters_internal(self):
        cex = {"x": 1, "__a": 5, "y": 2, "__b": 10}
        inputs = _extract_counterexample_inputs(cex)
        assert inputs == {"x": 1, "y": 2}

    def test_keeps_all_user_vars(self):
        cex = {"a": 10, "b": 20}
        inputs = _extract_counterexample_inputs(cex)
        assert inputs == {"a": 10, "b": 20}


class TestClassifyCounterexample(unittest.TestCase):
    def test_matching_is_spurious(self):
        d = CounterexampleDiag("p", "o", {}, {}, results_match=True)
        assert _classify_counterexample(d) == FailureKind.SPURIOUS

    def test_mismatch_is_bug(self):
        d = CounterexampleDiag("p", "o", {}, {}, results_match=False)
        assert _classify_counterexample(d) == FailureKind.CONFIRMED_BUG

    def test_none_is_unknown(self):
        d = CounterexampleDiag("p", "o", {}, {}, results_match=None)
        assert _classify_counterexample(d) == FailureKind.UNKNOWN


class TestMakeTestProgram(unittest.TestCase):
    def test_empty_inputs(self):
        result = _make_test_program("return 1;", {})
        assert result == "return 1;"

    def test_int_input(self):
        result = _make_test_program("return x;", {"x": 5})
        assert "let x = 5;" in result

    def test_negative_input(self):
        result = _make_test_program("return x;", {"x": -3})
        assert "let x = 0 - 3;" in result

    def test_bool_input(self):
        result = _make_test_program("return x;", {"x": True})
        assert "let x = true;" in result


class TestCheckBoundaryCoverage(unittest.TestCase):
    def test_exact_match(self):
        tests = [GuidedTestCase(inputs={"x": 5})]
        _check_boundary_coverage(tests, [{"x": 5}])
        assert tests[0].covers_boundary is True

    def test_near_match(self):
        tests = [GuidedTestCase(inputs={"x": 4})]
        _check_boundary_coverage(tests, [{"x": 5}])
        assert tests[0].covers_boundary is True  # |4-5| = 1 <= 1

    def test_far_no_match(self):
        tests = [GuidedTestCase(inputs={"x": 100})]
        _check_boundary_coverage(tests, [{"x": 5}])
        assert tests[0].covers_boundary is False

    def test_empty_cex(self):
        tests = [GuidedTestCase(inputs={"x": 5})]
        _check_boundary_coverage(tests, [{}])
        assert tests[0].covers_boundary is False


# === Section 3: Execution ===

class TestExecuteProgram(unittest.TestCase):
    def test_simple(self):
        r = _execute_program(TRIVIAL)
        assert r["error"] is None

    def test_function(self):
        r = _execute_program(CONSTANT_FOLDABLE)
        assert r["error"] is None

    def test_invalid_syntax(self):
        r = _execute_program("this is not valid!!!")
        assert r["error"] is not None


class TestExecuteOptimized(unittest.TestCase):
    def test_simple(self):
        r = _execute_optimized(TRIVIAL)
        assert isinstance(r, dict)

    def test_function(self):
        r = _execute_optimized(CONSTANT_FOLDABLE)
        assert isinstance(r, dict)


class TestCompareOptimization(unittest.TestCase):
    def test_simple_match(self):
        r = compare_optimization_behavior(TRIVIAL)
        assert isinstance(r, dict)
        assert "match" in r

    def test_constant_foldable(self):
        r = compare_optimization_behavior(CONSTANT_FOLDABLE)
        assert isinstance(r, dict)

    def test_identity(self):
        r = compare_optimization_behavior(IDENTITY)
        assert isinstance(r, dict)


# === Section 4: Guided Test Generation ===

class TestGuidedTestGeneration(unittest.TestCase):
    def test_simple_function(self):
        tests = _generate_guided_tests(SIMPLE_ARITHMETIC)
        assert isinstance(tests, list)

    def test_conditional(self):
        tests = _generate_guided_tests(SIMPLE_CONDITIONAL)
        assert isinstance(tests, list)

    def test_invalid_source(self):
        tests = _generate_guided_tests("not valid!")
        assert tests == []


# === Section 5: Cross Validation ===

class TestCrossValidate(unittest.TestCase):
    def test_with_empty_cex(self):
        diag = _cross_validate_counterexample(
            TRIVIAL, {}, "test_pass", "test_ob"
        )
        assert isinstance(diag, CounterexampleDiag)
        assert diag.pass_name == "test_pass"
        assert diag.results_match is not None

    def test_with_inputs(self):
        diag = _cross_validate_counterexample(
            CONSTANT_FOLDABLE, {"x": 5}, "constant_fold", "fold_1"
        )
        assert isinstance(diag, CounterexampleDiag)
        assert diag.kind in list(FailureKind)


# === Section 6: Full Pipeline ===

class TestAnalyzeCEGOV(unittest.TestCase):
    def test_simple_program(self):
        r = analyze_cegov(SIMPLE_ARITHMETIC)
        assert isinstance(r, CEGOVResult)
        assert r.source == SIMPLE_ARITHMETIC
        assert r.total_obligations >= 0

    def test_constant_foldable(self):
        r = analyze_cegov(CONSTANT_FOLDABLE)
        assert isinstance(r, CEGOVResult)

    def test_with_loop(self):
        r = analyze_cegov(WITH_LOOP)
        assert isinstance(r, CEGOVResult)

    def test_strength_reducible(self):
        r = analyze_cegov(STRENGTH_REDUCIBLE)
        assert isinstance(r, CEGOVResult)

    def test_dead_code(self):
        r = analyze_cegov(DEAD_CODE)
        assert isinstance(r, CEGOVResult)

    def test_multi_op(self):
        r = analyze_cegov(MULTI_OP)
        assert isinstance(r, CEGOVResult)

    def test_trivial(self):
        r = analyze_cegov(TRIVIAL)
        assert isinstance(r, CEGOVResult)

    def test_identity(self):
        r = analyze_cegov(IDENTITY)
        assert isinstance(r, CEGOVResult)


class TestValidateAndDiagnose(unittest.TestCase):
    def test_returns_string(self):
        s = validate_and_diagnose(CONSTANT_FOLDABLE)
        assert isinstance(s, str)
        assert "CEGOV Report" in s

    def test_simple(self):
        s = validate_and_diagnose(SIMPLE_ARITHMETIC)
        assert "Validation:" in s


class TestCheckPassWithCex(unittest.TestCase):
    def test_constant_fold(self):
        r = check_pass_with_cex(CONSTANT_FOLDABLE, "constant_fold")
        assert isinstance(r, CEGOVResult)

    def test_strength_reduction(self):
        r = check_pass_with_cex(STRENGTH_REDUCIBLE, "strength_reduction")
        assert isinstance(r, CEGOVResult)

    def test_dead_code_elimination(self):
        r = check_pass_with_cex(DEAD_CODE, "dead_code_elimination")
        assert isinstance(r, CEGOVResult)

    def test_unknown_pass(self):
        r = check_pass_with_cex(TRIVIAL, "nonexistent_pass")
        assert isinstance(r, CEGOVResult)
        assert r.validation_passed is False

    def test_peephole(self):
        r = check_pass_with_cex(MULTI_OP, "peephole")
        assert isinstance(r, CEGOVResult)


# === Section 7: Result Properties ===

class TestResultProperties(unittest.TestCase):
    def test_passed_program_properties(self):
        """A validated program should have no confirmed bugs."""
        r = analyze_cegov(CONSTANT_FOLDABLE)
        if r.validation_passed:
            assert r.confirmed_bugs == 0
            assert r.has_real_bugs is False

    def test_summary_contains_counts(self):
        r = analyze_cegov(SIMPLE_ARITHMETIC)
        s = r.summary()
        assert "Obligations:" in s

    def test_guided_tests_list(self):
        r = analyze_cegov(SIMPLE_CONDITIONAL)
        assert isinstance(r.guided_tests, list)


# === Section 8: Robustness ===

class TestRobustness(unittest.TestCase):
    def test_empty_source(self):
        r = analyze_cegov("")
        assert isinstance(r, CEGOVResult)

    def test_invalid_source(self):
        r = analyze_cegov("not valid C10!!!")
        assert isinstance(r, CEGOVResult)

    def test_compare_invalid(self):
        r = compare_optimization_behavior("not valid!!!")
        assert isinstance(r, dict)
        assert r.get("match") is not None or r.get("unoptimized", {}).get("error") is not None


if __name__ == '__main__':
    unittest.main()
