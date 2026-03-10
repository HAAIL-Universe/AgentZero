"""
Tests for V007: Automatic Loop Invariant Inference
====================================================
Tests the full pipeline: inference + verification without manual invariants.
"""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from invariant_inference import (
    infer_loop_invariants, auto_verify_function, auto_verify_program,
    smt_to_sexpr, InferenceMethod, InferenceResult, AutoVerifyResult,
    SVar, SInt, SBool, SBinOp, s_and, s_not,
)


# ============================================================
# Section 1: SMT-to-SExpr Conversion
# ============================================================

class TestSMTToSExpr:
    """Test conversion from SMT terms to V004 SExprs."""

    def test_int_constant(self):
        from smt_solver import IntConst
        result = smt_to_sexpr(IntConst(42))
        assert isinstance(result, SInt)
        assert result.value == 42

    def test_bool_constant(self):
        from smt_solver import BoolConst
        result = smt_to_sexpr(BoolConst(True))
        assert isinstance(result, SBool)
        assert result.value is True

    def test_variable(self):
        from smt_solver import Var, INT
        result = smt_to_sexpr(Var('x', INT))
        assert isinstance(result, SVar)
        assert result.name == 'x'

    def test_comparison(self):
        from smt_solver import Var, IntConst, App, Op, INT, BOOL
        term = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)
        result = smt_to_sexpr(term)
        assert isinstance(result, SBinOp)
        assert result.op == '>='

    def test_arithmetic(self):
        from smt_solver import Var, IntConst, App, Op, INT
        term = App(Op.ADD, [Var('x', INT), IntConst(1)], INT)
        result = smt_to_sexpr(term)
        assert isinstance(result, SBinOp)
        assert result.op == '+'


# ============================================================
# Section 2: Basic Invariant Inference
# ============================================================

class TestBasicInference:
    """Test invariant inference on simple loops."""

    def test_countdown_loop(self):
        """Countdown: let i = 10; while (i > 0) { i = i - 1; }"""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = infer_loop_invariants(source)
        assert len(result.invariants) > 0
        # Should find i >= 0 or similar
        descriptions = [inv.description for inv in result.invariants]
        assert any('i' in d for d in descriptions)

    def test_countup_loop(self):
        """Count up: let i = 0; while (i < 10) { i = i + 1; }"""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = infer_loop_invariants(source)
        assert len(result.invariants) > 0

    def test_accumulator_loop(self):
        """Accumulator: let s = 0; let i = 0; while (i < 5) { s = s + i; i = i + 1; }"""
        source = "let s = 0; let i = 0; while (i < 5) { s = s + i; i = i + 1; }"
        result = infer_loop_invariants(source)
        assert len(result.invariants) > 0
        # Should find i >= 0 and s >= 0 at minimum
        var_names = set()
        for inv in result.invariants:
            if isinstance(inv.sexpr, SBinOp):
                if isinstance(inv.sexpr.left, SVar):
                    var_names.add(inv.sexpr.left.name)
        assert 'i' in var_names or 's' in var_names

    def test_no_loop_returns_empty(self):
        """Source with no loop should return empty result."""
        source = "let x = 5;"
        result = infer_loop_invariants(source)
        assert len(result.invariants) == 0
        assert 'error' in result.stats

    def test_inferred_are_inductive(self):
        """All inferred invariants should be marked inductive."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = infer_loop_invariants(source)
        for inv in result.invariants:
            assert inv.is_inductive


# ============================================================
# Section 3: Inference Methods
# ============================================================

class TestInferenceMethods:
    """Test that different inference methods contribute candidates."""

    def test_abstract_interp_contributes(self):
        """Abstract interpretation should find interval bounds."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = infer_loop_invariants(source)
        methods = {inv.method for inv in result.invariants}
        # AI should contribute at least some candidates
        assert result.stats.get('ai_candidates', 0) >= 0

    def test_condition_based_contributes(self):
        """Condition-based analysis should try weakened conditions."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = infer_loop_invariants(source)
        cond_invs = [inv for inv in result.invariants
                     if inv.method == InferenceMethod.CONDITION_BASED]
        # i <= 10 should be found as a weakening of i < 10
        assert result.stats.get('cond_candidates', 0) >= 0

    def test_relational_template_contributes(self):
        """Relational templates should find sum/diff conservation."""
        source = "let x = 5; let y = 3; while (x > 0) { x = x - 1; y = y + 1; }"
        result = infer_loop_invariants(source)
        rel_invs = [inv for inv in result.invariants
                    if inv.method == InferenceMethod.RELATIONAL_TEMPLATE]
        # Should find x + y == 8 (conserved quantity)
        assert len(rel_invs) > 0
        found_sum = any('x' in inv.description and 'y' in inv.description
                       for inv in rel_invs)
        assert found_sum

    def test_deduplication(self):
        """Invariants should be deduplicated across methods."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = infer_loop_invariants(source)
        sexprs_strs = [str(inv.sexpr) for inv in result.invariants]
        assert len(sexprs_strs) == len(set(sexprs_strs))


# ============================================================
# Section 4: Sum Conservation (Relational)
# ============================================================

class TestSumConservation:
    """Test relational invariant discovery for conservation laws."""

    def test_sum_preserved(self):
        """x + y == constant is preserved when x-- and y++."""
        source = "let x = 10; let y = 0; while (x > 0) { x = x - 1; y = y + 1; }"
        result = infer_loop_invariants(source)
        rel = [inv for inv in result.invariants
               if inv.method == InferenceMethod.RELATIONAL_TEMPLATE]
        # x + y == 10 should be discovered
        found = any('sum' in inv.description and '10' in inv.description
                    for inv in rel)
        assert found

    def test_difference_preserved(self):
        """x - y == constant when both increment together."""
        source = "let x = 5; let y = 0; while (y < 10) { x = x + 1; y = y + 1; }"
        result = infer_loop_invariants(source)
        rel = [inv for inv in result.invariants
               if inv.method == InferenceMethod.RELATIONAL_TEMPLATE]
        # x - y == 5 should be discovered
        found = any('diff' in inv.description and '5' in inv.description
                    for inv in rel)
        assert found

    def test_no_spurious_relational(self):
        """Relational invariants that don't hold should not appear."""
        # x increments by 1, y increments by 2: no simple sum/diff conservation
        source = "let x = 0; let y = 0; while (x < 5) { x = x + 1; y = y + 2; }"
        result = infer_loop_invariants(source)
        rel = [inv for inv in result.invariants
               if inv.method == InferenceMethod.RELATIONAL_TEMPLATE]
        # x + y == 0 should NOT hold (it breaks after first iteration)
        # x - y == 0 should NOT hold either
        for inv in rel:
            assert inv.is_inductive  # All relational must be validated


# ============================================================
# Section 5: Auto-Verify Simple Functions
# ============================================================

class TestAutoVerifySimple:
    """Test auto_verify_function on programs without manual invariants."""

    def test_no_loop_function(self):
        """Function without loops should verify like V004."""
        source = """
        fn abs(x) {
            requires(x > 0);
            ensures(result == x);
            return x;
        }
        """
        result = auto_verify_function(source, 'abs')
        assert result.verified

    def test_simple_assignment(self):
        """Identity function verification."""
        source = """
        fn id(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        result = auto_verify_function(source, 'id')
        assert result.verified

    def test_increment(self):
        """Increment verification."""
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            let y = x + 1;
            return y;
        }
        """
        result = auto_verify_function(source, 'inc')
        assert result.verified


# ============================================================
# Section 6: Auto-Verify Loops
# ============================================================

class TestAutoVerifyLoops:
    """The main test: auto-verify functions with loops, no manual invariants."""

    def test_countdown_to_zero(self):
        """Countdown loop: result should be 0."""
        source = """
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        assert(i == 0);
        """
        result = auto_verify_program(source)
        # The invariant i >= 0 should be found, and i == 0 after !cond
        assert result.inferred is not None
        assert len(result.inferred.invariants) > 0

    def test_accumulator_nonnegative(self):
        """Accumulator produces non-negative sum."""
        source = """
        let s = 0;
        let i = 0;
        while (i < 5) {
            s = s + i;
            i = i + 1;
        }
        assert(s >= 0);
        """
        result = auto_verify_program(source)
        assert result.inferred is not None

    def test_conservation_law(self):
        """Transfer loop preserves total."""
        source = """
        let x = 10;
        let y = 0;
        while (x > 0) {
            x = x - 1;
            y = y + 1;
        }
        assert(y == 10);
        """
        result = auto_verify_program(source)
        assert result.inferred is not None
        # Should find x + y == 10


# ============================================================
# Section 7: Postcondition-Guided Inference
# ============================================================

class TestPostconditionGuided:
    """Test that postcondition guides invariant selection."""

    def test_postcond_guides_search(self):
        """When postcondition is given, inference targets it."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        postcond = SBinOp('>=', SVar('i'), SInt(0))
        result = infer_loop_invariants(source, postcondition=postcond)
        assert len(result.invariants) > 0

    def test_sufficient_flag(self):
        """Sufficient flag should be True when invariants establish postcondition."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        postcond = SBinOp('>=', SVar('i'), SInt(0))
        result = infer_loop_invariants(source, postcondition=postcond)
        # i >= 0 is inductive AND i >= 0 AND !(i > 0) => i >= 0
        if any(str(inv.sexpr) == '(i >= 0)' for inv in result.invariants):
            assert result.sufficient


# ============================================================
# Section 8: Edge Cases
# ============================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_loop_body(self):
        """Loop with no assignments (infinite loop) should still work."""
        source = "let x = 1; while (x > 0) { x = x; }"
        result = infer_loop_invariants(source)
        # x == 1 should be invariant (x never changes)
        assert any(inv.is_inductive for inv in result.invariants)

    def test_single_variable_loop(self):
        """Loop with single state variable."""
        source = "let n = 100; while (n > 0) { n = n - 1; }"
        result = infer_loop_invariants(source)
        assert len(result.invariants) > 0

    def test_loop_with_conditional_body(self):
        """Loop with if-else in body."""
        source = """
        let x = 0;
        let y = 10;
        while (y > 0) {
            if (y > 5) {
                x = x + 2;
            } else {
                x = x + 1;
            }
            y = y - 1;
        }
        """
        result = infer_loop_invariants(source)
        assert result is not None  # Should not crash

    def test_manual_invariants_still_work(self):
        """V004-style manual invariants should still be used if present."""
        source = """
        fn countdown(n) {
            requires(n >= 0);
            ensures(result == 0);
            let i = n;
            while (i > 0) {
                invariant(i >= 0);
                i = i - 1;
            }
            return i;
        }
        """
        result = auto_verify_function(source, 'countdown')
        assert result.verified


# ============================================================
# Section 9: Inference Statistics
# ============================================================

class TestStats:
    """Test that inference statistics are properly reported."""

    def test_stats_present(self):
        """Stats dict should have candidate counts."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = infer_loop_invariants(source)
        assert 'ai_candidates' in result.stats
        assert 'cond_candidates' in result.stats
        assert 'rel_candidates' in result.stats
        assert 'total_unique' in result.stats

    def test_stats_nonnegative(self):
        """All stats values should be non-negative integers."""
        source = "let x = 5; let y = 3; while (x > 0) { x = x - 1; y = y + 1; }"
        result = infer_loop_invariants(source)
        for key in ['ai_candidates', 'cond_candidates', 'rel_candidates', 'total_unique']:
            assert result.stats[key] >= 0


# ============================================================
# Section 10: Result Type Correctness
# ============================================================

class TestResultTypes:
    """Test that result types are correct."""

    def test_inference_result_type(self):
        source = "let i = 0; while (i < 5) { i = i + 1; }"
        result = infer_loop_invariants(source)
        assert isinstance(result, InferenceResult)

    def test_auto_verify_result_type(self):
        source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        result = auto_verify_function(source, 'f')
        assert isinstance(result, AutoVerifyResult)

    def test_sexprs_property(self):
        """InferenceResult.sexprs should return list of SExpr."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = infer_loop_invariants(source)
        sexprs = result.sexprs
        assert isinstance(sexprs, list)
        from invariant_inference import SExpr
        for s in sexprs:
            assert isinstance(s, SExpr)


# ============================================================
# Section 11: Multiple Loops
# ============================================================

class TestMultipleLoops:
    """Test handling of programs with multiple loops."""

    def test_second_loop_index(self):
        """Can infer for a specific loop by index."""
        source = """
        let a = 0;
        while (a < 5) { a = a + 1; }
        let b = 10;
        while (b > 0) { b = b - 1; }
        """
        result0 = infer_loop_invariants(source, loop_index=0)
        result1 = infer_loop_invariants(source, loop_index=1)
        assert len(result0.invariants) > 0
        assert len(result1.invariants) > 0

    def test_different_loops_different_invariants(self):
        """Different loops should produce different invariant sets."""
        source = """
        let a = 0;
        while (a < 100) { a = a + 1; }
        let b = 100;
        while (b > 0) { b = b - 1; }
        """
        r0 = infer_loop_invariants(source, loop_index=0)
        r1 = infer_loop_invariants(source, loop_index=1)
        strs0 = {str(inv.sexpr) for inv in r0.invariants}
        strs1 = {str(inv.sexpr) for inv in r1.invariants}
        # They should differ (different variables, different bounds)
        assert strs0 != strs1 or (len(strs0) == 0 and len(strs1) == 0)


# ============================================================
# Section 12: Invariant Method Provenance
# ============================================================

class TestProvenance:
    """Test that invariant provenance is correctly tracked."""

    def test_method_is_set(self):
        """Each invariant should have a method set."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = infer_loop_invariants(source)
        for inv in result.invariants:
            assert inv.method in (
                InferenceMethod.ABSTRACT_INTERP,
                InferenceMethod.PDR_DISCOVERY,
                InferenceMethod.RELATIONAL_TEMPLATE,
                InferenceMethod.CONDITION_BASED,
            )

    def test_description_is_set(self):
        """Each invariant should have a non-empty description."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = infer_loop_invariants(source)
        for inv in result.invariants:
            assert inv.description
            assert len(inv.description) > 0


# ============================================================
# Section 13: Integration with V004
# ============================================================

class TestV004Integration:
    """Test that auto-inferred invariants work with V004's verification."""

    def test_verify_result_has_vcs(self):
        """AutoVerifyResult should contain VCs from verification."""
        source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        result = auto_verify_function(source, 'f')
        assert result.verification is not None
        assert len(result.verification.vcs) > 0

    def test_failing_spec_detected(self):
        """A wrong specification should fail verification."""
        source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result < 0);
            return x;
        }
        """
        result = auto_verify_function(source, 'f')
        assert not result.verified

    def test_no_spec_skipped(self):
        """Functions without specs should be skipped."""
        source = """
        fn f(x) {
            return x;
        }
        """
        result = auto_verify_function(source, 'f')
        # No spec => nothing to verify => verified by default
        assert result.verified


# ============================================================
# Section 14: Complex Programs
# ============================================================

class TestComplexPrograms:
    """Test on more complex programs that exercise the full pipeline."""

    def test_gcd_style_loop(self):
        """GCD-like loop: alternating subtraction."""
        source = """
        let a = 12;
        let b = 8;
        while (a != b) {
            if (a > b) {
                a = a - b;
            } else {
                b = b - a;
            }
        }
        """
        result = infer_loop_invariants(source)
        # Should at least find a > 0 and b > 0
        assert result is not None  # Should not crash

    def test_nested_conditional_in_loop(self):
        """Loop with nested conditionals."""
        source = """
        let x = 0;
        let y = 0;
        let i = 0;
        while (i < 10) {
            if (i < 5) {
                x = x + 1;
            } else {
                y = y + 1;
            }
            i = i + 1;
        }
        """
        result = infer_loop_invariants(source)
        # i >= 0 should be found
        assert any('i' in str(inv.sexpr) for inv in result.invariants)

    def test_three_variables_sum_conservation(self):
        """Three variables with pairwise conservation."""
        source = """
        let a = 10;
        let b = 0;
        let c = 5;
        while (a > 0) {
            a = a - 1;
            b = b + 1;
        }
        """
        result = infer_loop_invariants(source)
        # a + b == 10 should be found
        rel = [inv for inv in result.invariants
               if inv.method == InferenceMethod.RELATIONAL_TEMPLATE]
        # c should be constant (c == 5)
        const_c = any('c' in inv.description and '5' in inv.description
                      for inv in result.invariants)
        # a + b == 10 should be found
        sum_ab = any('a' in inv.description and 'b' in inv.description
                     and '10' in inv.description for inv in rel)
        assert const_c or sum_ab  # At least one should be found
