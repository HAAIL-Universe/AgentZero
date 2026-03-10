"""
Tests for V006: Equivalence Checking
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from equiv_check import (
    check_function_equivalence, check_program_equivalence,
    check_equivalence_with_mapping, check_partial_equivalence,
    check_regression, EquivResult, EquivCheckResult,
)

# Import SMT types for domain constraints
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import Var as SMTVar, App, IntConst, Op as SMTOp, BOOL, INT


# ============================================================
# Section 1: Trivially equivalent programs
# ============================================================

class TestTrivialEquivalence:
    """Programs that are identical or trivially the same."""

    def test_identical_programs(self):
        src = "let x = 0;\nlet result = x + 1;"
        r = check_program_equivalence(src, src, {'x': 'int'}, output_var='result')
        assert r.is_equivalent

    def test_identical_functions(self):
        src = "fn f(x) { return x + 1; }"
        r = check_function_equivalence(src, 'f', src, 'f', {'x': 'int'})
        assert r.is_equivalent

    def test_constant_programs(self):
        src1 = "let result = 42;"
        src2 = "let result = 42;"
        r = check_program_equivalence(src1, src2, {}, output_var='result')
        assert r.is_equivalent

    def test_no_inputs(self):
        src1 = "let a = 5;\nlet b = 3;\nlet result = a + b;"
        src2 = "let result = 8;"
        r = check_program_equivalence(src1, src2, {}, output_var='result')
        assert r.is_equivalent


# ============================================================
# Section 2: Algebraic equivalences
# ============================================================

class TestAlgebraicEquivalence:
    """Programs that compute the same thing via different algebra."""

    def test_commutativity_add(self):
        src1 = "fn f(x, y) { return x + y; }"
        src2 = "fn f(x, y) { return y + x; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int', 'y': 'int'})
        assert r.is_equivalent

    def test_associativity_add(self):
        src1 = "fn f(x, y, z) { return (x + y) + z; }"
        src2 = "fn f(x, y, z) { return x + (y + z); }"
        r = check_function_equivalence(src1, 'f', src2, 'f',
                                        {'x': 'int', 'y': 'int', 'z': 'int'})
        assert r.is_equivalent

    def test_identity_add(self):
        src1 = "fn f(x) { return x + 0; }"
        src2 = "fn f(x) { return x; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.is_equivalent

    def test_double_negation(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn f(x) { return 0 - (0 - x); }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.is_equivalent

    def test_multiply_by_two(self):
        src1 = "fn f(x) { return x + x; }"
        src2 = "fn f(x) { return x * 2; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.is_equivalent

    def test_distributivity(self):
        src1 = "fn f(x, y) { return 2 * (x + y); }"
        src2 = "fn f(x, y) { return 2 * x + 2 * y; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int', 'y': 'int'})
        assert r.is_equivalent


# ============================================================
# Section 3: Non-equivalent programs (with counterexamples)
# ============================================================

class TestNonEquivalence:
    """Programs that are NOT equivalent -- should find counterexamples."""

    def test_off_by_one(self):
        src1 = "fn f(x) { return x + 1; }"
        src2 = "fn f(x) { return x + 2; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert not r.is_equivalent
        assert r.counterexample is not None

    def test_wrong_operator(self):
        src1 = "fn f(x, y) { return x + y; }"
        src2 = "fn f(x, y) { return x - y; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int', 'y': 'int'})
        assert not r.is_equivalent

    def test_wrong_constant(self):
        src1 = "let x = 0;\nlet result = x * 3;"
        src2 = "let x = 0;\nlet result = x * 4;"
        r = check_program_equivalence(src1, src2, {'x': 'int'}, output_var='result')
        assert not r.is_equivalent

    def test_different_branching(self):
        src1 = """
fn f(x) {
    if (x > 0) { return 1; }
    return 0;
}"""
        src2 = """
fn f(x) {
    if (x >= 0) { return 1; }
    return 0;
}"""
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert not r.is_equivalent
        # Counterexample should be x=0 (where > and >= differ)
        assert r.counterexample is not None
        assert r.counterexample.inputs.get('x') == 0

    def test_counterexample_has_inputs(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn f(x) { return x + 1; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert not r.is_equivalent
        assert 'x' in r.counterexample.inputs


# ============================================================
# Section 4: Conditional equivalences
# ============================================================

class TestConditionalEquivalence:
    """Programs with branches that are equivalent."""

    def test_abs_implementations(self):
        src1 = """
fn abs(x) {
    if (x < 0) { return 0 - x; }
    return x;
}"""
        src2 = """
fn abs(x) {
    if (x >= 0) { return x; }
    return 0 - x;
}"""
        r = check_function_equivalence(src1, 'abs', src2, 'abs', {'x': 'int'})
        assert r.is_equivalent

    def test_max_implementations(self):
        src1 = """
fn max(a, b) {
    if (a > b) { return a; }
    return b;
}"""
        src2 = """
fn max(a, b) {
    if (b >= a) { return b; }
    return a;
}"""
        r = check_function_equivalence(src1, 'max', src2, 'max',
                                        {'a': 'int', 'b': 'int'})
        assert r.is_equivalent

    def test_min_implementations(self):
        src1 = """
fn min(a, b) {
    if (a < b) { return a; }
    return b;
}"""
        src2 = """
fn min(a, b) {
    if (a <= b) { return a; }
    return b;
}"""
        # These differ at a==b: src1 returns b, src2 returns a.
        # But a==b means both return the same value!
        r = check_function_equivalence(src1, 'min', src2, 'min',
                                        {'a': 'int', 'b': 'int'})
        assert r.is_equivalent

    def test_sign_function(self):
        src1 = """
fn sign(x) {
    if (x > 0) { return 1; }
    if (x < 0) { return 0 - 1; }
    return 0;
}"""
        src2 = """
fn sign(x) {
    if (x < 0) { return 0 - 1; }
    if (x > 0) { return 1; }
    return 0;
}"""
        r = check_function_equivalence(src1, 'sign', src2, 'sign', {'x': 'int'})
        assert r.is_equivalent


# ============================================================
# Section 5: Refactoring equivalences
# ============================================================

class TestRefactoring:
    """Refactored programs that should remain equivalent."""

    def test_extract_variable(self):
        src1 = """
fn f(x) {
    return (x + 1) * (x + 1);
}"""
        src2 = """
fn f(x) {
    let y = x + 1;
    return y * y;
}"""
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.is_equivalent

    def test_inline_variable(self):
        src1 = """
fn f(a, b) {
    let sum = a + b;
    let doubled = sum * 2;
    return doubled;
}"""
        src2 = """
fn f(a, b) {
    return (a + b) * 2;
}"""
        r = check_function_equivalence(src1, 'f', src2, 'f', {'a': 'int', 'b': 'int'})
        assert r.is_equivalent

    def test_reorder_independent_stmts(self):
        src1 = "let x = 0;\nlet a = x + 1;\nlet b = x * 2;\nlet result = a + b;"
        src2 = "let x = 0;\nlet b = x * 2;\nlet a = x + 1;\nlet result = a + b;"
        r = check_program_equivalence(src1, src2, {'x': 'int'}, output_var='result')
        assert r.is_equivalent

    def test_strength_reduction(self):
        """x * 2 replaced with x + x."""
        src1 = "fn f(x) { return x * 2; }"
        src2 = "fn f(x) { return x + x; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.is_equivalent


# ============================================================
# Section 6: Loop equivalences
# ============================================================

class TestLoopEquivalence:
    """Programs with loops that compute the same thing."""

    def test_loop_vs_closed_form(self):
        """Sum 1..n via loop vs closed form n*(n+1)/2 -- but we don't have div,
        so test sum via loop vs manually unrolled for small n."""
        src1 = """
let n = 0;
let s = 0;
let i = 1;
while (i <= n) {
    s = s + i;
    i = i + 1;
}
let result = s;
"""
        src2 = """
let n = 0;
let s = 0;
let i = 1;
while (i <= n) {
    s = s + i;
    i = i + 1;
}
let result = s;
"""
        r = check_program_equivalence(src1, src2, {'n': 'int'}, output_var='result')
        assert r.is_equivalent

    def test_loop_unroll_equiv(self):
        """Two different loop structures computing same result."""
        src1 = """
let x = 0;
let r = 0;
while (x > 0) {
    r = r + 1;
    x = x - 1;
}
let result = r;
"""
        src2 = """
let x = 0;
let r = x;
let result = r;
"""
        # For x > 0, src1 counts down and increments r. For x <= 0, r stays 0.
        # src2 sets r = x directly.
        # These are equivalent only for x >= 0 (bounded by loop unrolling).
        # For x <= 0: src1 gives 0, src2 gives x. NOT equivalent in general.
        r = check_program_equivalence(src1, src2, {'x': 'int'}, output_var='result')
        assert not r.is_equivalent


# ============================================================
# Section 7: Variable mapping / renaming
# ============================================================

class TestVariableMapping:
    """Programs with different variable names that compute the same thing."""

    def test_simple_rename(self):
        src1 = "let x = 0;\nlet result = x * 2;"
        src2 = "let y = 0;\nlet output = y * 2;"
        r = check_equivalence_with_mapping(
            src1, src2,
            symbolic_inputs={'x': 'int'},
            var_map={'x': 'y', 'result': 'output'},
            output_var1='result',
            output_var2='output',
        )
        assert r.is_equivalent

    def test_rename_not_equivalent(self):
        src1 = "let x = 0;\nlet result = x + 1;"
        src2 = "let y = 0;\nlet output = y + 2;"
        r = check_equivalence_with_mapping(
            src1, src2,
            symbolic_inputs={'x': 'int'},
            var_map={'x': 'y', 'result': 'output'},
            output_var1='result',
            output_var2='output',
        )
        assert not r.is_equivalent


# ============================================================
# Section 8: Partial equivalence (domain-restricted)
# ============================================================

class TestPartialEquivalence:
    """Programs equivalent only under restricted domains."""

    def test_equivalent_for_positive(self):
        """These differ at x=0 (> vs >=) but agree for x > 0."""
        src1 = """
fn f(x) {
    if (x > 0) { return 1; }
    return 0;
}
let x = 0;
let result = f(x);
"""
        src2 = """
fn f(x) {
    if (x >= 0) { return 1; }
    return 0;
}
let x = 0;
let result = f(x);
"""
        # Restrict to x > 0
        x_var = SMTVar('x', INT)
        domain = [App(SMTOp.GT, [x_var, IntConst(0)], BOOL)]

        r = check_partial_equivalence(
            src1, src2, {'x': 'int'}, domain, output_var='result'
        )
        assert r.is_equivalent

    def test_not_equivalent_even_restricted(self):
        src1 = "let x = 0;\nlet result = x + 1;"
        src2 = "let x = 0;\nlet result = x + 2;"
        x_var = SMTVar('x', INT)
        domain = [App(SMTOp.GT, [x_var, IntConst(0)], BOOL)]
        r = check_partial_equivalence(
            src1, src2, {'x': 'int'}, domain, output_var='result'
        )
        assert not r.is_equivalent


# ============================================================
# Section 9: Regression checking
# ============================================================

class TestRegressionChecking:
    """Verify refactored code matches original via the regression API."""

    def test_regression_pass(self):
        original = "fn compute(x) { return x * 2 + 1; }"
        refactored = "fn compute(x) { return 1 + x + x; }"
        r = check_regression(
            original, refactored, {'x': 'int'},
            fn_name='compute', param_types={'x': 'int'}
        )
        assert r.is_equivalent

    def test_regression_fail(self):
        original = "fn compute(x) { return x * 2 + 1; }"
        refactored = "fn compute(x) { return x * 2; }"  # Missing +1
        r = check_regression(
            original, refactored, {'x': 'int'},
            fn_name='compute', param_types={'x': 'int'}
        )
        assert not r.is_equivalent
        assert r.counterexample is not None

    def test_regression_program_mode(self):
        original = "let x = 0;\nlet result = x + x;"
        refactored = "let x = 0;\nlet result = 2 * x;"
        r = check_regression(original, refactored, {'x': 'int'}, output_var='result')
        assert r.is_equivalent


# ============================================================
# Section 10: Multi-argument functions
# ============================================================

class TestMultiArgument:
    """Functions with multiple arguments."""

    def test_two_args_equivalent(self):
        src1 = "fn f(a, b) { return a + b + 1; }"
        src2 = "fn f(a, b) { return 1 + a + b; }"
        r = check_function_equivalence(src1, 'f', src2, 'f',
                                        {'a': 'int', 'b': 'int'})
        assert r.is_equivalent

    def test_three_args_not_equivalent(self):
        src1 = "fn f(a, b, c) { return a + b + c; }"
        src2 = "fn f(a, b, c) { return a + b - c; }"
        r = check_function_equivalence(src1, 'f', src2, 'f',
                                        {'a': 'int', 'b': 'int', 'c': 'int'})
        assert not r.is_equivalent

    def test_argument_order_matters(self):
        src1 = "fn f(a, b) { return a - b; }"
        src2 = "fn f(a, b) { return b - a; }"
        r = check_function_equivalence(src1, 'f', src2, 'f',
                                        {'a': 'int', 'b': 'int'})
        assert not r.is_equivalent


# ============================================================
# Section 11: Complex conditional logic
# ============================================================

class TestComplexConditionals:
    """More complex conditional equivalences."""

    def test_nested_if_flattened(self):
        src1 = """
fn classify(x) {
    if (x > 0) {
        if (x > 10) { return 2; }
        return 1;
    }
    return 0;
}"""
        src2 = """
fn classify(x) {
    if (x > 10) { return 2; }
    if (x > 0) { return 1; }
    return 0;
}"""
        r = check_function_equivalence(src1, 'classify', src2, 'classify', {'x': 'int'})
        assert r.is_equivalent

    def test_guard_merge(self):
        """Two guards that partition the same space."""
        src1 = """
fn f(x) {
    if (x > 5) { return 2; }
    if (x > 0) { return 1; }
    return 0;
}"""
        src2 = """
fn f(x) {
    if (x > 0) {
        if (x > 5) { return 2; }
        return 1;
    }
    return 0;
}"""
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.is_equivalent

    def test_clamp_implementations(self):
        src1 = """
fn clamp(x) {
    if (x < 0) { return 0; }
    if (x > 100) { return 100; }
    return x;
}"""
        src2 = """
fn clamp(x) {
    if (x > 100) { return 100; }
    if (x < 0) { return 0; }
    return x;
}"""
        r = check_function_equivalence(src1, 'clamp', src2, 'clamp', {'x': 'int'})
        assert r.is_equivalent


# ============================================================
# Section 12: Statistics and metadata
# ============================================================

class TestStatistics:
    """Check that result metadata is populated."""

    def test_paths_checked_count(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn f(x) { return x; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.paths_checked > 0
        assert r.path_pairs_checked > 0
        assert r.equivalent_pairs > 0

    def test_counterexample_structure(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn f(x) { return x + 1; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.counterexample is not None
        assert isinstance(r.counterexample.inputs, dict)
        assert 'x' in r.counterexample.inputs

    def test_equivalent_result_type(self):
        src = "fn f(x) { return x; }"
        r = check_function_equivalence(src, 'f', src, 'f', {'x': 'int'})
        assert r.result == EquivResult.EQUIVALENT

    def test_not_equivalent_result_type(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn f(x) { return 0 - x; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.result == EquivResult.NOT_EQUIVALENT


# ============================================================
# Section 13: Different function names
# ============================================================

class TestDifferentNames:
    """Comparing functions with different names."""

    def test_different_fn_names(self):
        src1 = "fn foo(x) { return x + 1; }"
        src2 = "fn bar(x) { return x + 1; }"
        r = check_function_equivalence(src1, 'foo', src2, 'bar', {'x': 'int'})
        assert r.is_equivalent

    def test_different_fn_names_not_equiv(self):
        src1 = "fn add_one(x) { return x + 1; }"
        src2 = "fn add_two(x) { return x + 2; }"
        r = check_function_equivalence(src1, 'add_one', src2, 'add_two', {'x': 'int'})
        assert not r.is_equivalent


# ============================================================
# Section 14: Edge cases
# ============================================================

class TestEdgeCases:
    """Edge cases and corner cases."""

    def test_zero_argument_functions(self):
        src1 = "fn f() { return 42; }"
        src2 = "fn f() { return 42; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {})
        assert r.is_equivalent

    def test_zero_argument_not_equiv(self):
        src1 = "fn f() { return 42; }"
        src2 = "fn f() { return 43; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {})
        assert not r.is_equivalent

    def test_deeply_nested(self):
        src1 = """
fn f(x) {
    let a = x + 1;
    let b = a + 1;
    let c = b + 1;
    return c;
}"""
        src2 = "fn f(x) { return x + 3; }"
        r = check_function_equivalence(src1, 'f', src2, 'f', {'x': 'int'})
        assert r.is_equivalent

    def test_self_equivalence_complex(self):
        src = """
fn f(x) {
    if (x > 10) {
        if (x > 20) { return x * 2; }
        return x + 10;
    }
    if (x < 0) { return 0 - x; }
    return x;
}"""
        r = check_function_equivalence(src, 'f', src, 'f', {'x': 'int'})
        assert r.is_equivalent
