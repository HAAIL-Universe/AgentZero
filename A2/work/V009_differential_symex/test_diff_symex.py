"""
Tests for V009: Differential Symbolic Execution

Sections:
  1. AST Diff -- structural comparison
  2. Identical programs -- no changes detected
  3. Value changes -- same structure, different constants
  4. Structural changes -- added/removed statements
  5. Conditional changes -- modified branch behavior
  6. Function changes -- modified function bodies
  7. Focused vs full comparison -- path pair skipping
  8. Semantic diff -- syntactic vs behavioral changes
  9. Regression checking -- detecting regressions
  10. Domain-constrained diff -- restricted input analysis
  11. Multi-change programs -- multiple simultaneous changes
  12. Change impact analysis -- per-change impact
  13. Non-equivalent refactoring -- breaking changes
  14. Edge cases -- empty programs, single statements
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from diff_symex import (
    compute_ast_diff, diff_programs, diff_functions,
    check_regression, semantic_diff, change_impact_analysis,
    diff_with_constraints,
    ASTDiff, ASTChange, ChangeType,
    DiffResult, DiffImpact, BehavioralDiff, ChangeSummary,
    _parse, _ast_equal, _stmt_signature,
)

# SMT imports for domain constraints
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import Var as SMTVar, App, IntConst, Op as SMTOp, INT, BOOL


# ============================================================
# Section 1: AST Diff -- structural comparison
# ============================================================

class TestASTDiff:
    def test_identical_programs(self):
        src = "let x = 5;"
        diff = compute_ast_diff(src, src)
        assert not diff.has_changes
        assert diff.change_count == 0

    def test_value_change(self):
        old = "let x = 5;"
        new = "let x = 10;"
        diff = compute_ast_diff(old, new)
        assert diff.has_changes
        assert diff.change_count == 1
        assert diff.changes[0].change_type == ChangeType.MODIFIED

    def test_added_statement(self):
        old = "let x = 5;"
        new = "let x = 5;\nlet y = 10;"
        diff = compute_ast_diff(old, new)
        assert diff.has_changes
        assert any(c.change_type == ChangeType.ADDED for c in diff.changes)

    def test_removed_statement(self):
        old = "let x = 5;\nlet y = 10;"
        new = "let x = 5;"
        diff = compute_ast_diff(old, new)
        assert diff.has_changes
        assert any(c.change_type == ChangeType.REMOVED for c in diff.changes)

    def test_multiple_changes(self):
        old = "let x = 5;\nlet y = 10;\nlet z = 15;"
        new = "let x = 99;\nlet y = 10;\nlet z = 0;"
        diff = compute_ast_diff(old, new)
        assert diff.has_changes
        assert diff.change_count == 2  # x and z changed, y unchanged

    def test_stmt_count(self):
        old = "let x = 1;\nlet y = 2;"
        new = "let x = 1;\nlet y = 2;\nlet z = 3;"
        diff = compute_ast_diff(old, new)
        assert diff.old_stmt_count == 2
        assert diff.new_stmt_count == 3


# ============================================================
# Section 2: Identical programs -- no changes detected
# ============================================================

class TestIdenticalPrograms:
    def test_identical_simple(self):
        src = "let x = 5;\nlet y = x + 1;"
        result = diff_programs(src, src, {'x': 'int'}, output_var='y')
        assert result.is_safe
        assert result.impact == DiffImpact.NO_BEHAVIORAL_CHANGE

    def test_identical_conditional(self):
        src = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 0 - x;
        }
        """
        result = diff_programs(src, src, {'x': 'int'}, output_var='y')
        assert result.is_safe

    def test_identical_no_ast_changes(self):
        src = "let a = 1;\nlet b = 2;\nlet c = a + b;"
        result = diff_programs(src, src, {'a': 'int'}, output_var='c')
        assert result.impact == DiffImpact.NO_BEHAVIORAL_CHANGE
        assert result.ast_diff.change_count == 0


# ============================================================
# Section 3: Value changes -- same structure, different constants
# ============================================================

class TestValueChanges:
    def test_constant_change_affects_output(self):
        old = "let x = 0;\nlet y = x + 5;"
        new = "let x = 0;\nlet y = x + 10;"
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert result.has_regression
        assert len(result.behavioral_diffs) > 0

    def test_constant_change_in_unused_var(self):
        old = "let x = 0;\nlet unused = 5;\nlet y = x + 1;"
        new = "let x = 0;\nlet unused = 99;\nlet y = x + 1;"
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert result.is_safe

    def test_initial_value_change(self):
        old = "let x = 0;\nlet y = x + x;"
        new = "let x = 0;\nlet y = x * 2;"
        # x + x == x * 2 for all x, so semantically equivalent
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert result.is_safe


# ============================================================
# Section 4: Structural changes -- added/removed statements
# ============================================================

class TestStructuralChanges:
    def test_added_assignment(self):
        old = "let x = 0;\nlet y = x;"
        new = "let x = 0;\nlet y = x;\ny = y + 1;"
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert result.has_regression or result.impact == DiffImpact.PARTIAL_CHANGE
        assert len(result.behavioral_diffs) > 0

    def test_removed_assignment(self):
        old = "let x = 0;\nlet y = x;\ny = y + 1;"
        new = "let x = 0;\nlet y = x;"
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert len(result.behavioral_diffs) > 0

    def test_added_conditional(self):
        old = "let x = 0;\nlet y = x;"
        new = """
        let x = 0;
        let y = x;
        if (x > 0) {
            y = y + 1;
        }
        """
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        # For x > 0, output differs (y = x vs y = x + 1)
        assert len(result.behavioral_diffs) > 0


# ============================================================
# Section 5: Conditional changes -- modified branch behavior
# ============================================================

class TestConditionalChanges:
    def test_condition_change(self):
        old = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 0;
        }
        """
        new = """
        let x = 0;
        let y = 0;
        if (x >= 0) {
            y = 1;
        } else {
            y = 0;
        }
        """
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        # At x=0: old gives y=0, new gives y=1
        assert len(result.behavioral_diffs) > 0

    def test_branch_body_change(self):
        old = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 0 - 1;
        }
        """
        new = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 0;
        }
        """
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        # For x <= 0: old gives y=-1, new gives y=0
        assert len(result.behavioral_diffs) > 0

    def test_semantically_equivalent_conditional_rewrite(self):
        # abs(x) implemented two ways
        old = """
        let x = 0;
        let y = 0;
        if (x >= 0) {
            y = x;
        } else {
            y = 0 - x;
        }
        """
        new = """
        let x = 0;
        let y = 0;
        if (x < 0) {
            y = 0 - x;
        } else {
            y = x;
        }
        """
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert result.is_safe


# ============================================================
# Section 6: Function changes -- modified function bodies
# ============================================================

class TestFunctionChanges:
    def test_function_body_change(self):
        old = "fn f(a) { return a + 1; }"
        new = "fn f(a) { return a + 2; }"
        result = diff_functions(old, 'f', new, 'f', {'a': 'int'})
        assert len(result.behavioral_diffs) > 0

    def test_function_semantically_equivalent(self):
        old = "fn f(a) { return a + a; }"
        new = "fn f(a) { return a * 2; }"
        result = diff_functions(old, 'f', new, 'f', {'a': 'int'})
        assert result.is_safe

    def test_function_with_conditional_change(self):
        old = """
        fn abs_val(x) {
            if (x >= 0) { return x; }
            return 0 - x;
        }
        """
        new = """
        fn abs_val(x) {
            if (x > 0) { return x; }
            return 0 - x;
        }
        """
        # At x=0: old returns 0, new returns 0 (0-0=0). Actually equivalent!
        result = diff_functions(old, 'abs_val', new, 'abs_val', {'x': 'int'})
        assert result.is_safe


# ============================================================
# Section 7: Focused vs full comparison
# ============================================================

class TestFocusedComparison:
    def test_focused_skips_unchanged_pairs(self):
        old = """
        let x = 0;
        let y = x + 1;
        let z = y + 1;
        """
        new = """
        let x = 0;
        let y = x + 1;
        let z = y + 2;
        """
        result_focused = diff_programs(old, new, {'x': 'int'},
                                       output_var='z', focused=True)
        result_full = diff_programs(old, new, {'x': 'int'},
                                    output_var='z', focused=False)
        # Both should detect the behavioral change
        assert len(result_focused.behavioral_diffs) > 0
        assert len(result_full.behavioral_diffs) > 0

    def test_focused_identifies_safe_change(self):
        old = "let x = 0;\nlet unused = 5;\nlet y = x + 1;"
        new = "let x = 0;\nlet unused = 99;\nlet y = x + 1;"
        result = diff_programs(old, new, {'x': 'int'}, output_var='y', focused=True)
        assert result.is_safe


# ============================================================
# Section 8: Semantic diff -- syntactic vs behavioral changes
# ============================================================

class TestSemanticDiff:
    def test_syntactic_only_change(self):
        old = "let x = 0;\nlet y = x + x;"
        new = "let x = 0;\nlet y = x * 2;"
        summary = semantic_diff(old, new, {'x': 'int'}, output_var='y')
        assert summary.total_diffs == 0  # No behavioral diffs

    def test_behavioral_change(self):
        old = "let x = 0;\nlet y = x + 1;"
        new = "let x = 0;\nlet y = x + 2;"
        summary = semantic_diff(old, new, {'x': 'int'}, output_var='y')
        assert summary.total_diffs > 0

    def test_added_behavior(self):
        old = "let x = 0;\nlet y = x;"
        new = "let x = 0;\nlet y = x;\nlet z = x + 1;"
        summary = semantic_diff(old, new, {'x': 'int'}, output_var='y')
        assert len(summary.added_behaviors) > 0

    def test_removed_behavior(self):
        old = "let x = 0;\nlet y = x;\nlet z = x + 1;"
        new = "let x = 0;\nlet y = x;"
        summary = semantic_diff(old, new, {'x': 'int'}, output_var='y')
        assert len(summary.removed_behaviors) > 0


# ============================================================
# Section 9: Regression checking
# ============================================================

class TestRegressionChecking:
    def test_safe_refactoring(self):
        old = "let x = 0;\nlet y = x + x;"
        new = "let x = 0;\nlet y = x * 2;"
        result = check_regression(old, new, {'x': 'int'}, output_var='y')
        assert result.is_safe

    def test_breaking_change(self):
        old = "let x = 0;\nlet y = x + 1;"
        new = "let x = 0;\nlet y = x + 2;"
        result = check_regression(old, new, {'x': 'int'}, output_var='y')
        assert result.has_regression

    def test_function_regression(self):
        old = "fn f(a) { return a + 1; }"
        new = "fn f(a) { return a + 2; }"
        result = check_regression(old, new, {'a': 'int'}, fn_name='f')
        assert result.has_regression

    def test_function_safe_refactor(self):
        old = "fn f(a) { return a + a; }"
        new = "fn f(a) { return a * 2; }"
        result = check_regression(old, new, {'a': 'int'}, fn_name='f')
        assert result.is_safe


# ============================================================
# Section 10: Domain-constrained diff
# ============================================================

class TestDomainConstrainedDiff:
    def test_diff_within_domain(self):
        old = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 0;
        }
        """
        new = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 1;
        }
        """
        # Only differs for x <= 0
        # Constrain to x > 0: should be safe
        x_var = SMTVar('x', INT)
        domain = [App(SMTOp.GT, [x_var, IntConst(0)], BOOL)]
        result = diff_with_constraints(old, new, {'x': 'int'}, domain,
                                        output_var='y')
        assert result.is_safe

    def test_diff_outside_domain(self):
        old = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 0;
        }
        """
        new = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 1;
        }
        """
        # Constrain to x <= 0: should detect diff
        x_var = SMTVar('x', INT)
        domain = [App(SMTOp.LE, [x_var, IntConst(0)], BOOL)]
        result = diff_with_constraints(old, new, {'x': 'int'}, domain,
                                        output_var='y')
        assert len(result.behavioral_diffs) > 0


# ============================================================
# Section 11: Multi-change programs
# ============================================================

class TestMultiChange:
    def test_two_independent_changes(self):
        old = "let x = 0;\nlet y = x + 1;\nlet z = x + 2;"
        new = "let x = 0;\nlet y = x + 10;\nlet z = x + 20;"
        result = diff_programs(old, new, {'x': 'int'}, output_var='z')
        assert len(result.behavioral_diffs) > 0
        assert result.ast_diff.change_count == 2

    def test_cancelling_changes(self):
        # Two changes that cancel each other out
        old = "let x = 0;\nlet y = x + 5;\nlet z = y + 5;"
        new = "let x = 0;\nlet y = x + 10;\nlet z = y + 0;"
        # old: z = x + 10, new: z = x + 10
        result = diff_programs(old, new, {'x': 'int'}, output_var='z')
        assert result.is_safe

    def test_three_changes_one_behavioral(self):
        old = "let x = 0;\nlet a = 1;\nlet b = 2;\nlet y = x + a;"
        new = "let x = 0;\nlet a = 1;\nlet b = 99;\nlet y = x + a;"
        # Only b changed, and y doesn't depend on b
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert result.is_safe


# ============================================================
# Section 12: Change impact analysis
# ============================================================

class TestChangeImpact:
    def test_impact_no_changes(self):
        src = "let x = 5;"
        impact = change_impact_analysis(src, src, {'x': 'int'})
        assert impact.get('no_changes') is True

    def test_impact_with_behavioral_change(self):
        old = "let x = 0;\nlet y = x + 1;"
        new = "let x = 0;\nlet y = x + 2;"
        impact = change_impact_analysis(old, new, {'x': 'int'}, output_var='y')
        assert impact['total_syntactic_changes'] > 0
        assert impact['total_behavioral_diffs'] > 0

    def test_impact_syntactic_only(self):
        old = "let x = 0;\nlet y = x + x;"
        new = "let x = 0;\nlet y = x * 2;"
        impact = change_impact_analysis(old, new, {'x': 'int'}, output_var='y')
        assert impact['total_syntactic_changes'] > 0
        assert impact['total_behavioral_diffs'] == 0


# ============================================================
# Section 13: Non-equivalent refactoring
# ============================================================

class TestNonEquivalentRefactoring:
    def test_off_by_one(self):
        old = """
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 0;
        }
        """
        new = """
        let x = 0;
        let y = 0;
        if (x >= 0) {
            y = x;
        } else {
            y = 0;
        }
        """
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        # At x=0: old gives y=0, new gives y=0 (same!)
        # Actually these are equivalent: when x=0, both give y=0.
        # For x>0: both give y=x. For x<0: both give y=0.
        assert result.is_safe

    def test_sign_flip(self):
        old = "let x = 0;\nlet y = x + 1;"
        new = "let x = 0;\nlet y = x - 1;"
        result = diff_programs(old, new, {'x': 'int'}, output_var='y')
        assert result.has_regression
        assert len(result.behavioral_diffs) > 0
        # Counterexample should show x+1 != x-1
        bd = result.behavioral_diffs[0]
        assert bd.old_output != bd.new_output

    def test_wrong_operator(self):
        old = "fn f(a, b) { return a + b; }"
        new = "fn f(a, b) { return a - b; }"
        result = diff_functions(old, 'f', new, 'f', {'a': 'int', 'b': 'int'})
        assert result.has_regression


# ============================================================
# Section 14: Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_statement(self):
        old = "let x = 5;"
        new = "let x = 10;"
        diff = compute_ast_diff(old, new)
        assert diff.change_count == 1

    def test_ast_equal_literals(self):
        ast1 = _parse("let x = 5;")
        ast2 = _parse("let x = 5;")
        assert _ast_equal(ast1, ast2)

    def test_ast_not_equal(self):
        ast1 = _parse("let x = 5;")
        ast2 = _parse("let x = 10;")
        assert not _ast_equal(ast1, ast2)

    def test_stmt_signature_let(self):
        ast = _parse("let x = 5;")
        sig = _stmt_signature(ast.stmts[0])
        assert "let" in sig and "x" in sig

    def test_diff_result_properties(self):
        result = DiffResult(
            impact=DiffImpact.NO_BEHAVIORAL_CHANGE,
            ast_diff=ASTDiff(),
        )
        assert result.is_safe
        assert not result.has_regression

    def test_diff_result_regression_properties(self):
        result = DiffResult(
            impact=DiffImpact.BEHAVIORAL_CHANGE,
            ast_diff=ASTDiff(),
        )
        assert result.has_regression
        assert not result.is_safe

    def test_change_summary_empty(self):
        summary = ChangeSummary()
        assert summary.total_diffs == 0

    def test_ast_diff_changed_locations(self):
        old = "let x = 1;\nlet y = 2;\nlet z = 3;"
        new = "let x = 1;\nlet y = 99;\nlet z = 3;"
        diff = compute_ast_diff(old, new)
        assert len(diff.changed_locations_new) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
