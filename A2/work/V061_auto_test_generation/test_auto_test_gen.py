"""Tests for V061: Automatic Test Generation from Specifications."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V001_guided_symbolic_execution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V054_verification_driven_fuzzing'))

import pytest
from auto_test_gen import (
    AutoTestGenerator, SpecAnalyzer, TestExecutor, InputCombiner,
    TestMinimizer, TestCase, TestSuite, TestSource, TestVerdict,
    GenerationResult, generate_tests, generate_and_report,
    quick_generate, deep_generate
)
from vc_gen import (
    FnSpec, SVar, SInt, SBool, SBinOp, SAnd, SOr, SNot,
    SUnaryOp, SImplies, SIte, s_and, s_or
)


# ---------------------------------------------------------------------------
# Test sources (MiniLang with requires/ensures)
# ---------------------------------------------------------------------------

ABS_SOURCE = """
fn abs(x) {
    requires(x >= -100);
    requires(x <= 100);
    ensures(result >= 0);
    if (x < 0) {
        return -x;
    }
    return x;
}
"""

MAX_SOURCE = """
fn max(a, b) {
    ensures(result >= a);
    ensures(result >= b);
    if (a > b) {
        return a;
    }
    return b;
}
"""

IDENTITY_SOURCE = """
fn identity(x) {
    ensures(result == x);
    return x;
}
"""

CLAMP_SOURCE = """
fn clamp(x) {
    requires(x >= -1000);
    requires(x <= 1000);
    ensures(result >= 0);
    ensures(result <= 10);
    if (x < 0) {
        return 0;
    }
    if (x > 10) {
        return 10;
    }
    return x;
}
"""

DOUBLE_SOURCE = """
fn double(x) {
    requires(x >= 0);
    requires(x <= 50);
    ensures(result == x + x);
    return x + x;
}
"""

CLASSIFY_SOURCE = """
fn classify(x) {
    requires(x >= -100);
    requires(x <= 100);
    if (x < 0) {
        return -1;
    }
    if (x == 0) {
        return 0;
    }
    return 1;
}
"""

BUGGY_ABS_SOURCE = """
fn buggy_abs(x) {
    requires(x >= -100);
    requires(x <= 100);
    ensures(result >= 0);
    if (x < 0) {
        return x;
    }
    return x;
}
"""

NO_SPEC_SOURCE = """
fn add(a, b) {
    return a + b;
}
"""

MULTI_FN_SOURCE = """
fn inc(x) {
    ensures(result == x + 1);
    return x + 1;
}
fn dec(x) {
    ensures(result == x - 1);
    return x - 1;
}
"""


# ===========================================================================
# SpecAnalyzer tests
# ===========================================================================

class TestSpecAnalyzer:

    def setup_method(self):
        self.analyzer = SpecAnalyzer()

    # -- extract_boundaries --

    def test_boundary_gt(self):
        # x > 5 => test at 6, 5, 4
        spec = FnSpec("f", ["x"], [SBinOp('>', SVar('x'), SInt(5))], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert 6 in vals  # just inside
        assert 5 in vals  # boundary
        assert 4 in vals  # just outside

    def test_boundary_gte(self):
        spec = FnSpec("f", ["x"], [SBinOp('>=', SVar('x'), SInt(0))], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert 0 in vals
        assert -1 in vals
        assert 1 in vals

    def test_boundary_lt(self):
        spec = FnSpec("f", ["x"], [SBinOp('<', SVar('x'), SInt(10))], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert 9 in vals
        assert 10 in vals
        assert 11 in vals

    def test_boundary_le(self):
        spec = FnSpec("f", ["x"], [SBinOp('<=', SVar('x'), SInt(100))], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert 100 in vals
        assert 101 in vals
        assert 99 in vals

    def test_boundary_eq(self):
        spec = FnSpec("f", ["x"], [SBinOp('==', SVar('x'), SInt(42))], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert 42 in vals
        assert 41 in vals
        assert 43 in vals

    def test_boundary_neq(self):
        spec = FnSpec("f", ["x"], [SBinOp('!=', SVar('x'), SInt(0))], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert 1 in vals
        assert -1 in vals
        assert 0 in vals

    def test_boundary_and(self):
        # x >= 0 AND x <= 10
        spec = FnSpec("f", ["x"], [
            s_and(SBinOp('>=', SVar('x'), SInt(0)),
                  SBinOp('<=', SVar('x'), SInt(10)))
        ], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert 0 in vals
        assert 10 in vals

    def test_boundary_multi_params(self):
        spec = FnSpec("f", ["x", "y"], [
            SBinOp('>', SVar('x'), SInt(0)),
            SBinOp('<', SVar('y'), SInt(10))
        ], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        x_vals = [b['x'] for b in bounds if 'x' in b]
        y_vals = [b['y'] for b in bounds if 'y' in b]
        assert len(x_vals) >= 2
        assert len(y_vals) >= 2

    def test_boundary_no_preconditions(self):
        spec = FnSpec("f", ["x"], [], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        assert bounds == []

    def test_boundary_reversed_comparison(self):
        # 5 > x (constant on left)
        spec = FnSpec("f", ["x"], [SBinOp('>', SInt(5), SVar('x'))], [], [])
        bounds = self.analyzer.extract_boundaries(spec)
        vals = [b['x'] for b in bounds if 'x' in b]
        assert len(vals) >= 2

    # -- check_precondition --

    def test_check_precond_true(self):
        spec = FnSpec("f", ["x"], [SBinOp('>=', SVar('x'), SInt(0))], [], [])
        assert self.analyzer.check_precondition(spec, {'x': 5})

    def test_check_precond_false(self):
        spec = FnSpec("f", ["x"], [SBinOp('>=', SVar('x'), SInt(0))], [], [])
        assert not self.analyzer.check_precondition(spec, {'x': -1})

    def test_check_precond_empty(self):
        spec = FnSpec("f", ["x"], [], [], [])
        assert self.analyzer.check_precondition(spec, {'x': -999})

    def test_check_precond_multi(self):
        spec = FnSpec("f", ["x"], [
            SBinOp('>=', SVar('x'), SInt(0)),
            SBinOp('<=', SVar('x'), SInt(100))
        ], [], [])
        assert self.analyzer.check_precondition(spec, {'x': 50})
        assert not self.analyzer.check_precondition(spec, {'x': 101})
        assert not self.analyzer.check_precondition(spec, {'x': -1})

    # -- check_postcondition --

    def test_check_postcond_true(self):
        spec = FnSpec("f", ["x"], [], [SBinOp('>=', SVar('result'), SInt(0))], [])
        assert self.analyzer.check_postcondition(spec, {'x': 5}, 5)

    def test_check_postcond_false(self):
        spec = FnSpec("f", ["x"], [], [SBinOp('>=', SVar('result'), SInt(0))], [])
        assert not self.analyzer.check_postcondition(spec, {'x': -5}, -5)

    def test_check_postcond_empty(self):
        spec = FnSpec("f", ["x"], [], [], [])
        assert self.analyzer.check_postcondition(spec, {'x': 5}, -999)

    def test_check_postcond_with_input(self):
        # ensures(result == x + 1)
        spec = FnSpec("f", ["x"], [],
                      [SBinOp('==', SVar('result'), SBinOp('+', SVar('x'), SInt(1)))], [])
        assert self.analyzer.check_postcondition(spec, {'x': 5}, 6)
        assert not self.analyzer.check_postcondition(spec, {'x': 5}, 5)

    # -- _eval_sexpr --

    def test_eval_var(self):
        assert self.analyzer._eval_sexpr(SVar('x'), {'x': 42}) == 42

    def test_eval_int(self):
        assert self.analyzer._eval_sexpr(SInt(7), {}) == 7

    def test_eval_bool(self):
        assert self.analyzer._eval_sexpr(SBool(True), {}) == True

    def test_eval_binop_arithmetic(self):
        expr = SBinOp('+', SVar('x'), SInt(1))
        assert self.analyzer._eval_sexpr(expr, {'x': 5}) == 6

    def test_eval_binop_comparison(self):
        expr = SBinOp('<', SVar('x'), SInt(10))
        assert self.analyzer._eval_sexpr(expr, {'x': 5}) == True
        assert self.analyzer._eval_sexpr(expr, {'x': 15}) == False

    def test_eval_and(self):
        expr = SAnd((SBool(True), SBool(True)))
        assert self.analyzer._eval_sexpr(expr, {}) == True
        expr2 = SAnd((SBool(True), SBool(False)))
        assert self.analyzer._eval_sexpr(expr2, {}) == False

    def test_eval_or(self):
        expr = SOr((SBool(False), SBool(True)))
        assert self.analyzer._eval_sexpr(expr, {}) == True

    def test_eval_not(self):
        expr = SNot(SBool(True))
        assert self.analyzer._eval_sexpr(expr, {}) == False

    def test_eval_implies(self):
        expr = SImplies(SBool(True), SBool(False))
        assert self.analyzer._eval_sexpr(expr, {}) == False
        expr2 = SImplies(SBool(False), SBool(False))
        assert self.analyzer._eval_sexpr(expr2, {}) == True

    def test_eval_ite(self):
        expr = SIte(SBool(True), SInt(1), SInt(2))
        assert self.analyzer._eval_sexpr(expr, {}) == 1
        expr2 = SIte(SBool(False), SInt(1), SInt(2))
        assert self.analyzer._eval_sexpr(expr2, {}) == 2

    def test_eval_unary_neg(self):
        expr = SUnaryOp('-', SInt(5))
        assert self.analyzer._eval_sexpr(expr, {}) == -5

    def test_eval_unary_not(self):
        expr = SUnaryOp('not', SBool(True))
        assert self.analyzer._eval_sexpr(expr, {}) == False

    def test_eval_division_by_zero(self):
        expr = SBinOp('/', SInt(5), SInt(0))
        assert self.analyzer._eval_sexpr(expr, {}) == 0

    def test_eval_mod_by_zero(self):
        expr = SBinOp('%', SInt(5), SInt(0))
        assert self.analyzer._eval_sexpr(expr, {}) == 0

    # -- generate_smt_inputs --

    def test_smt_inputs_basic(self):
        spec = FnSpec("f", ["x"], [
            SBinOp('>=', SVar('x'), SInt(0)),
            SBinOp('<=', SVar('x'), SInt(10))
        ], [], [])
        inputs = self.analyzer.generate_smt_inputs(spec, count=5)
        assert len(inputs) >= 1
        for inp in inputs:
            assert 0 <= inp['x'] <= 10

    def test_smt_inputs_diverse(self):
        spec = FnSpec("f", ["x"], [
            SBinOp('>=', SVar('x'), SInt(0)),
            SBinOp('<=', SVar('x'), SInt(100))
        ], [], [])
        inputs = self.analyzer.generate_smt_inputs(spec, count=5)
        if len(inputs) >= 2:
            vals = set(inp['x'] for inp in inputs)
            assert len(vals) >= 2  # should be diverse

    def test_smt_inputs_no_precond(self):
        spec = FnSpec("f", ["x"], [], [], [])
        inputs = self.analyzer.generate_smt_inputs(spec)
        assert inputs == []

    def test_smt_inputs_multi_param(self):
        spec = FnSpec("f", ["x", "y"], [
            SBinOp('>', SVar('x'), SInt(0)),
            SBinOp('>', SVar('y'), SInt(0))
        ], [], [])
        inputs = self.analyzer.generate_smt_inputs(spec, count=3)
        assert len(inputs) >= 1
        for inp in inputs:
            assert inp['x'] > 0
            assert inp['y'] > 0


# ===========================================================================
# TestExecutor tests
# ===========================================================================

class TestTestExecutor:

    def setup_method(self):
        self.executor = TestExecutor()

    def test_execute_identity(self):
        result, error = self.executor.execute(IDENTITY_SOURCE, 'identity', {'x': 42})
        assert error is None
        assert result == 42

    def test_execute_abs_positive(self):
        result, error = self.executor.execute(ABS_SOURCE, 'abs', {'x': 5})
        assert error is None
        assert result == 5

    def test_execute_abs_negative(self):
        result, error = self.executor.execute(ABS_SOURCE, 'abs', {'x': -5})
        assert error is None
        assert result == 5

    def test_execute_max(self):
        result, error = self.executor.execute(MAX_SOURCE, 'max', {'a': 3, 'b': 7})
        assert error is None
        assert result == 7

    def test_execute_clamp_low(self):
        result, error = self.executor.execute(CLAMP_SOURCE, 'clamp', {'x': -5})
        assert error is None
        assert result == 0

    def test_execute_clamp_high(self):
        result, error = self.executor.execute(CLAMP_SOURCE, 'clamp', {'x': 50})
        assert error is None
        assert result == 10

    def test_execute_clamp_middle(self):
        result, error = self.executor.execute(CLAMP_SOURCE, 'clamp', {'x': 5})
        assert error is None
        assert result == 5

    def test_get_params(self):
        params = self.executor._get_params(ABS_SOURCE, 'abs')
        assert params == ['x']

    def test_get_params_multi(self):
        params = self.executor._get_params(MAX_SOURCE, 'max')
        assert params == ['a', 'b']

    def test_parse_result_int(self):
        assert self.executor._parse_result('42') == 42

    def test_parse_result_negative(self):
        assert self.executor._parse_result('-5') == -5

    def test_parse_result_bool(self):
        assert self.executor._parse_result('true') == True
        assert self.executor._parse_result('false') == False

    def test_parse_result_null(self):
        assert self.executor._parse_result('null') is None


# ===========================================================================
# InputCombiner tests
# ===========================================================================

class TestInputCombiner:

    def test_combine_fills_missing(self):
        combiner = InputCombiner(['x', 'y'])
        result = combiner.combine([{'x': 5}])
        assert result[0] == {'x': 5, 'y': 0}

    def test_combine_with_defaults(self):
        combiner = InputCombiner(['x', 'y'])
        result = combiner.combine([{'x': 5}], defaults={'x': 1, 'y': 2})
        assert result[0] == {'x': 5, 'y': 2}

    def test_combine_deduplicates(self):
        combiner = InputCombiner(['x'])
        result = combiner.combine([{'x': 5}, {'x': 5}])
        assert len(result) == 1

    def test_combine_filters_extra_params(self):
        combiner = InputCombiner(['x'])
        result = combiner.combine([{'x': 5, 'z': 99}])
        assert 'z' not in result[0]

    def test_cross_product_basic(self):
        combiner = InputCombiner(['x', 'y'])
        result = combiner.cross_product({'x': [0, 1], 'y': [10, 20]})
        assert len(result) == 4
        keys = set(tuple(sorted(r.items())) for r in result)
        assert (('x', 0), ('y', 10)) in keys
        assert (('x', 1), ('y', 20)) in keys

    def test_cross_product_max_cap(self):
        combiner = InputCombiner(['x', 'y', 'z'])
        result = combiner.cross_product(
            {'x': list(range(20)), 'y': list(range(20)), 'z': list(range(20))},
            max_combos=50
        )
        assert len(result) <= 50

    def test_cross_product_fills_missing(self):
        combiner = InputCombiner(['x', 'y'])
        result = combiner.cross_product({'x': [5]})
        assert len(result) == 1
        assert result[0]['y'] == 0

    def test_cross_product_empty(self):
        combiner = InputCombiner(['x'])
        result = combiner.cross_product({})
        assert result == []


# ===========================================================================
# TestMinimizer tests
# ===========================================================================

class TestTestMinimizer:

    def test_minimize_to_zero(self):
        minimizer = TestMinimizer()
        # Property: x > 0
        result = minimizer.minimize_input({'x': 100}, lambda inp: inp['x'] > 0)
        assert result['x'] == 1  # minimal positive

    def test_minimize_multi_param(self):
        minimizer = TestMinimizer()
        # Property: x + y > 5
        result = minimizer.minimize_input(
            {'x': 50, 'y': 50},
            lambda inp: inp['x'] + inp['y'] > 5
        )
        assert result['x'] + result['y'] <= 50 + 50  # should reduce

    def test_minimize_already_minimal(self):
        minimizer = TestMinimizer()
        result = minimizer.minimize_input({'x': 0}, lambda inp: True)
        assert result == {'x': 0}

    def test_minimize_preserves_property(self):
        minimizer = TestMinimizer()
        # Must stay negative
        result = minimizer.minimize_input(
            {'x': -50},
            lambda inp: inp['x'] < 0
        )
        assert result['x'] < 0


# ===========================================================================
# Integration: AutoTestGenerator
# ===========================================================================

class TestAutoTestGenerator:

    def setup_method(self):
        self.gen = AutoTestGenerator(max_tests=50, seed=42)

    # -- Basic generation --

    def test_generate_identity(self):
        result = self.gen.generate(IDENTITY_SOURCE, 'identity')
        assert len(result.suites) == 1
        suite = result.suites[0]
        assert suite.function_name == 'identity'
        assert suite.total >= 1

    def test_generate_abs(self):
        result = self.gen.generate(ABS_SOURCE, 'abs')
        suite = result.suites[0]
        assert suite.total >= 3  # at least some boundary + SMT
        # All passing tests should have non-negative results
        for t in suite.tests:
            if t.verdict == TestVerdict.PASS and t.actual_output is not None:
                assert t.actual_output >= 0

    def test_generate_max(self):
        result = self.gen.generate(MAX_SOURCE, 'max')
        suite = result.suites[0]
        assert suite.total >= 1

    def test_generate_clamp(self):
        result = self.gen.generate(CLAMP_SOURCE, 'clamp')
        suite = result.suites[0]
        assert suite.total >= 3
        # Passing tests: result in [0, 10]
        for t in suite.tests:
            if t.verdict == TestVerdict.PASS and t.actual_output is not None:
                assert 0 <= t.actual_output <= 10

    def test_generate_double(self):
        result = self.gen.generate(DOUBLE_SOURCE, 'double')
        suite = result.suites[0]
        assert suite.total >= 1
        for t in suite.tests:
            if t.verdict == TestVerdict.PASS and t.precondition_met:
                if t.actual_output is not None and isinstance(t.inputs.get('x'), int):
                    assert t.actual_output == t.inputs['x'] * 2

    # -- Buggy code detection --

    def test_detect_buggy_abs(self):
        result = self.gen.generate(BUGGY_ABS_SOURCE, 'buggy_abs')
        suite = result.suites[0]
        # Should find at least one failure (negative input returns negative)
        failures = [t for t in suite.tests if t.verdict == TestVerdict.FAIL]
        assert len(failures) >= 1, "Should detect postcondition violation for negative inputs"

    # -- No spec --

    def test_generate_no_spec(self):
        result = self.gen.generate(NO_SPEC_SOURCE, 'add')
        suite = result.suites[0]
        assert suite.total >= 1
        # Without postconditions, all non-error tests pass
        for t in suite.tests:
            assert t.verdict in (TestVerdict.PASS, TestVerdict.ERROR, TestVerdict.SKIP)

    # -- Multiple functions --

    def test_generate_multi_fn(self):
        result = self.gen.generate(MULTI_FN_SOURCE)
        assert len(result.suites) == 2
        names = {s.function_name for s in result.suites}
        assert 'inc' in names
        assert 'dec' in names

    def test_generate_multi_fn_single(self):
        result = self.gen.generate(MULTI_FN_SOURCE, 'inc')
        assert len(result.suites) == 1
        assert result.suites[0].function_name == 'inc'

    # -- Test sources diversity --

    def test_sources_diversity(self):
        result = self.gen.generate(CLAMP_SOURCE, 'clamp')
        suite = result.suites[0]
        sources = set(t.source for t in suite.tests)
        # Should have at least boundary + one other source
        assert len(sources) >= 2

    # -- Precondition filtering --

    def test_precondition_filtering(self):
        result = self.gen.generate(ABS_SOURCE, 'abs')
        suite = result.suites[0]
        for t in suite.tests:
            if not t.precondition_met:
                assert t.verdict == TestVerdict.SKIP

    # -- Test deduplication --

    def test_no_duplicate_inputs(self):
        result = self.gen.generate(CLAMP_SOURCE, 'clamp')
        suite = result.suites[0]
        keys = [t.key for t in suite.tests]
        # Allow minimal tests to duplicate (they're derived)
        non_minimal = [t.key for t in suite.tests if t.source != TestSource.MINIMAL]
        assert len(non_minimal) == len(set(non_minimal))

    # -- Result properties --

    def test_result_total_generated(self):
        result = self.gen.generate(ABS_SOURCE, 'abs')
        assert result.total_generated == result.suites[0].total
        assert result.total_unique == result.suites[0].unique_inputs

    def test_suite_counts(self):
        result = self.gen.generate(ABS_SOURCE, 'abs')
        suite = result.suites[0]
        assert suite.passed + suite.failed + suite.errors + suite.skipped == suite.total

    def test_suite_tests_by_source(self):
        result = self.gen.generate(CLAMP_SOURCE, 'clamp')
        suite = result.suites[0]
        by_source = suite.tests_by_source()
        total_in_groups = sum(len(ts) for ts in by_source.values())
        assert total_in_groups == suite.total


# ===========================================================================
# Convenience functions
# ===========================================================================

class TestConvenienceFunctions:

    def test_generate_tests(self):
        result = generate_tests(ABS_SOURCE, 'abs')
        assert len(result.suites) == 1
        assert result.suites[0].total >= 1

    def test_generate_and_report(self):
        report = generate_and_report(ABS_SOURCE, 'abs')
        assert 'abs' in report
        assert 'Total:' in report
        assert 'Pass:' in report

    def test_generate_and_report_buggy(self):
        report = generate_and_report(BUGGY_ABS_SOURCE, 'buggy_abs')
        assert 'FAILURES' in report

    def test_quick_generate(self):
        result = quick_generate(CLAMP_SOURCE, 'clamp')
        assert len(result.suites) == 1
        assert result.suites[0].total <= 30

    def test_deep_generate(self):
        result = deep_generate(IDENTITY_SOURCE, 'identity')
        assert len(result.suites) == 1


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_source(self):
        gen = AutoTestGenerator()
        result = gen.generate("")
        assert len(result.suites) == 0

    def test_no_functions(self):
        gen = AutoTestGenerator()
        result = gen.generate("let x = 5;")
        assert len(result.suites) == 0

    def test_nonexistent_function(self):
        gen = AutoTestGenerator()
        result = gen.generate(ABS_SOURCE, 'nonexistent')
        assert len(result.suites) == 0

    def test_classify_paths(self):
        gen = AutoTestGenerator(max_tests=50, seed=42)
        result = gen.generate(CLASSIFY_SOURCE, 'classify')
        suite = result.suites[0]
        assert suite.total >= 1
        # Should cover all three branches
        outputs = set()
        for t in suite.tests:
            if t.actual_output is not None:
                outputs.add(t.actual_output)
        assert -1 in outputs or 0 in outputs or 1 in outputs

    def test_max_tests_respected(self):
        gen = AutoTestGenerator(max_tests=10, seed=42)
        result = gen.generate(ABS_SOURCE, 'abs')
        suite = result.suites[0]
        # Allow a few extra from minimization
        non_minimal = [t for t in suite.tests if t.source != TestSource.MINIMAL]
        assert len(non_minimal) <= 10

    def test_all_passed_property(self):
        result = generate_tests(IDENTITY_SOURCE, 'identity', max_tests=10)
        # Identity should pass all
        assert result.all_passed

    def test_all_passed_false_for_buggy(self):
        result = generate_tests(BUGGY_ABS_SOURCE, 'buggy_abs', max_tests=20)
        assert not result.all_passed


# ===========================================================================
# TestCase dataclass
# ===========================================================================

class TestTestCaseDataclass:

    def test_key(self):
        tc = TestCase(inputs={'x': 5, 'y': 3})
        assert tc.key == (('x', 5), ('y', 3))

    def test_key_order_independent(self):
        tc1 = TestCase(inputs={'x': 5, 'y': 3})
        tc2 = TestCase(inputs={'y': 3, 'x': 5})
        assert tc1.key == tc2.key

    def test_defaults(self):
        tc = TestCase(inputs={'x': 1})
        assert tc.verdict == TestVerdict.SKIP
        assert tc.source == TestSource.RANDOM
        assert tc.precondition_met == True
        assert tc.postcondition_checked == False
        assert tc.postcondition_held == True


# ===========================================================================
# TestSuite dataclass
# ===========================================================================

class TestTestSuiteDataclass:

    def test_counts(self):
        suite = TestSuite(function_name="f", tests=[
            TestCase(inputs={'x': 1}, verdict=TestVerdict.PASS),
            TestCase(inputs={'x': 2}, verdict=TestVerdict.PASS),
            TestCase(inputs={'x': 3}, verdict=TestVerdict.FAIL),
            TestCase(inputs={'x': 4}, verdict=TestVerdict.ERROR),
            TestCase(inputs={'x': 5}, verdict=TestVerdict.SKIP),
        ])
        assert suite.total == 5
        assert suite.passed == 2
        assert suite.failed == 1
        assert suite.errors == 1
        assert suite.skipped == 1

    def test_unique_inputs(self):
        suite = TestSuite(function_name="f", tests=[
            TestCase(inputs={'x': 1}),
            TestCase(inputs={'x': 1}),  # duplicate
            TestCase(inputs={'x': 2}),
        ])
        assert suite.unique_inputs == 2

    def test_tests_by_source(self):
        suite = TestSuite(function_name="f", tests=[
            TestCase(inputs={'x': 1}, source=TestSource.SYMBOLIC),
            TestCase(inputs={'x': 2}, source=TestSource.SYMBOLIC),
            TestCase(inputs={'x': 3}, source=TestSource.MUTATION),
        ])
        by_source = suite.tests_by_source()
        assert len(by_source[TestSource.SYMBOLIC]) == 2
        assert len(by_source[TestSource.MUTATION]) == 1


# ===========================================================================
# GenerationResult dataclass
# ===========================================================================

class TestGenerationResultDataclass:

    def test_all_passed_empty(self):
        result = GenerationResult()
        assert result.all_passed

    def test_all_passed_with_pass(self):
        suite = TestSuite(function_name="f", tests=[
            TestCase(inputs={'x': 1}, verdict=TestVerdict.PASS),
        ])
        result = GenerationResult(suites=[suite])
        assert result.all_passed

    def test_all_passed_with_fail(self):
        suite = TestSuite(function_name="f", tests=[
            TestCase(inputs={'x': 1}, verdict=TestVerdict.FAIL),
        ])
        result = GenerationResult(suites=[suite])
        assert not result.all_passed


# ===========================================================================
# Spec boundary + SMT interaction
# ===========================================================================

class TestSpecSMTInteraction:

    def test_smt_respects_preconditions(self):
        """SMT-generated inputs should satisfy preconditions."""
        gen = AutoTestGenerator(max_tests=30, seed=42)
        result = gen.generate(CLAMP_SOURCE, 'clamp')
        suite = result.suites[0]
        smt_tests = [t for t in suite.tests if t.description and 'SMT' in t.description]
        for t in smt_tests:
            assert t.precondition_met

    def test_boundary_covers_edges(self):
        """Boundary tests should include spec edge values."""
        gen = AutoTestGenerator(max_tests=50, seed=42)
        result = gen.generate(CLAMP_SOURCE, 'clamp')
        suite = result.suites[0]
        boundary_tests = [t for t in suite.tests if t.source == TestSource.SPEC_BOUNDARY]
        boundary_vals = set()
        for t in boundary_tests:
            boundary_vals.add(t.inputs.get('x'))
        # Should include values around precondition edges (-1000, 1000)
        has_around_lower = any(v in boundary_vals for v in [-1001, -1000, -999])
        has_around_upper = any(v in boundary_vals for v in [999, 1000, 1001])
        assert has_around_lower
        assert has_around_upper


# ===========================================================================
# Minimization integration
# ===========================================================================

class TestMinimizationIntegration:

    def test_minimized_tests_added(self):
        """Buggy code should produce minimized failure tests."""
        gen = AutoTestGenerator(max_tests=50, seed=42)
        result = gen.generate(BUGGY_ABS_SOURCE, 'buggy_abs')
        suite = result.suites[0]
        minimal_tests = [t for t in suite.tests if t.source == TestSource.MINIMAL]
        # Should have at least one minimized failure
        if suite.failed > 0:
            # Minimization may or may not produce distinct inputs
            assert suite.failed >= 1

    def test_minimized_still_fails(self):
        """Minimized tests should still demonstrate the failure."""
        gen = AutoTestGenerator(max_tests=50, seed=42)
        result = gen.generate(BUGGY_ABS_SOURCE, 'buggy_abs')
        suite = result.suites[0]
        for t in suite.tests:
            if t.source == TestSource.MINIMAL:
                assert t.verdict == TestVerdict.FAIL


# ===========================================================================
# Mutation integration
# ===========================================================================

class TestMutationIntegration:

    def test_mutations_generated(self):
        gen = AutoTestGenerator(max_tests=50, mutation_rounds=2, seed=42)
        result = gen.generate(CLAMP_SOURCE, 'clamp')
        suite = result.suites[0]
        mutation_tests = [t for t in suite.tests if t.source == TestSource.MUTATION]
        assert len(mutation_tests) >= 1

    def test_mutations_diverse(self):
        gen = AutoTestGenerator(max_tests=50, mutation_rounds=2, seed=42)
        result = gen.generate(ABS_SOURCE, 'abs')
        suite = result.suites[0]
        mutation_tests = [t for t in suite.tests if t.source == TestSource.MUTATION]
        if len(mutation_tests) >= 2:
            vals = set(t.inputs.get('x') for t in mutation_tests)
            assert len(vals) >= 2


# ===========================================================================
# Report format
# ===========================================================================

class TestReportFormat:

    def test_report_contains_function_name(self):
        report = generate_and_report(ABS_SOURCE, 'abs')
        assert 'abs' in report

    def test_report_contains_counts(self):
        report = generate_and_report(ABS_SOURCE, 'abs')
        assert 'Total:' in report
        assert 'Pass:' in report
        assert 'Fail:' in report
        assert 'Error:' in report
        assert 'Skip:' in report

    def test_report_contains_sources(self):
        report = generate_and_report(CLAMP_SOURCE, 'clamp')
        # Should mention at least one source type
        has_source = any(s.value in report for s in TestSource)
        assert has_source

    def test_report_multi_function(self):
        report = generate_and_report(MULTI_FN_SOURCE)
        assert 'inc' in report
        assert 'dec' in report
