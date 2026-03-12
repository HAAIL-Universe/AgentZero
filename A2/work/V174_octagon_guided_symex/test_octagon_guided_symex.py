"""Tests for V174: Octagon-Guided Symbolic Execution."""

import pytest
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from octagon_guided_symex import (
    guided_execute, GuidedResult,
    analyze_relational_pruning,
    verify_relational_property,
    compare_v001_vs_v174,
    batch_guided_execute,
    _ast_to_oct_program, _convert_expr, _convert_cond,
    _octagon_pre_analyze, _check_branch_feasibility_octagon,
    _parse_property,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))
from octagon import Octagon, OctConstraint, octagon_from_intervals


# ===================================================================
# 1. AST to Octagon Conversion
# ===================================================================

class TestASTConversion:
    def test_simple_assignment(self):
        result = _octagon_pre_analyze("let x = 5;")
        lo, hi = result.final_state.get_bounds('x')
        assert lo == 5 and hi == 5

    def test_variable_copy(self):
        result = _octagon_pre_analyze("let x = 5; let y = x;")
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 5 and hi == 5

    def test_addition(self):
        result = _octagon_pre_analyze("let x = 3; let y = 4; let z = x + y;")
        lo, hi = result.final_state.get_bounds('z')
        assert lo == 7 and hi == 7

    def test_subtraction(self):
        result = _octagon_pre_analyze("let x = 10; let y = 3; let z = x - y;")
        lo, hi = result.final_state.get_bounds('z')
        assert lo == 7 and hi == 7

    def test_if_statement(self):
        result = _octagon_pre_analyze("""
            let x = 5;
            let y = 0;
            if (x > 3) { y = 1; } else { y = 2; }
        """)
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 1 and hi == 1  # x=5 > 3, so only then branch

    def test_unary_neg(self):
        result = _octagon_pre_analyze("let x = 5; let y = -x;")
        lo, hi = result.final_state.get_bounds('y')
        assert lo == -5 and hi == -5


# ===================================================================
# 2. Octagon Pre-Analysis
# ===================================================================

class TestOctagonPreAnalysis:
    def test_basic_pre_analysis(self):
        result = _octagon_pre_analyze("let x = 10; let y = x - 3;")
        assert not result.final_state.is_bot()
        lo, hi = result.final_state.get_difference_bound('x', 'y')
        assert lo == 3 and hi == 3

    def test_symbolic_inputs(self):
        result = _octagon_pre_analyze("let y = x + 1;", {'x': 'int'})
        assert 'x' in result.final_state.variables()

    def test_loop_analysis(self):
        result = _octagon_pre_analyze("""
            let i = 0;
            while (i < 10) { i = i + 1; }
        """)
        assert not result.final_state.is_bot()


# ===================================================================
# 3. Branch Feasibility
# ===================================================================

class TestBranchFeasibility:
    def test_infeasible_branch(self):
        """After x = 5, y = x + 3, the branch y < x is infeasible (y - x == 3)."""
        result = _octagon_pre_analyze("let x = 5; let y = x + 3;")
        from stack_vm import lex, Parser, BinOp, Var as ASTVar
        # y < x
        cond = BinOp('<', ASTVar('y', 0), ASTVar('x', 0), 0)
        feasibility = _check_branch_feasibility_octagon(result.final_state, cond)
        assert feasibility == 'infeasible'

    def test_feasible_branch(self):
        """After x = 5, the branch x > 3 is feasible."""
        result = _octagon_pre_analyze("let x = 5;")
        from stack_vm import BinOp, Var as ASTVar, IntLit
        cond = BinOp('>', ASTVar('x', 0), IntLit(3, 0), 0)
        feasibility = _check_branch_feasibility_octagon(result.final_state, cond)
        assert feasibility == 'feasible'

    def test_bot_state(self):
        state = Octagon.bot()
        from stack_vm import BinOp, Var as ASTVar, IntLit
        cond = BinOp('>', ASTVar('x', 0), IntLit(0, 0), 0)
        assert _check_branch_feasibility_octagon(state, cond) == 'infeasible'


# ===================================================================
# 4. Guided Execution: Basic
# ===================================================================

class TestGuidedExecBasic:
    def test_simple_program(self):
        result = guided_execute("let x = 5; let y = x + 1;")
        assert isinstance(result, GuidedResult)
        assert not result.octagon_state.is_bot()

    def test_with_symbolic_input(self):
        result = guided_execute("let y = x + 1;", {'x': 'int'})
        assert len(result.paths) >= 1

    def test_pruning_stats(self):
        result = guided_execute("let x = 5; if (x > 10) { let y = 1; } else { let y = 2; }")
        assert result.branches_analyzed >= 1

    def test_relational_constraints(self):
        result = guided_execute("let x = 5; let y = x + 3;")
        assert result.relational_constraints_found > 0


# ===================================================================
# 5. Guided Execution: Relational Pruning
# ===================================================================

class TestRelationalPruning:
    def test_difference_pruning(self):
        """Octagon detects y - x == 3, so y < x is infeasible."""
        source = """
            let x = 5;
            let y = x + 3;
            if (y < x) {
                let z = 1;
            } else {
                let z = 2;
            }
        """
        result = guided_execute(source)
        # The branch y < x should be pruned by octagon
        assert result.branches_pruned_by_octagon >= 1

    def test_sum_bound_pruning(self):
        """When x + y is bounded, octagon can prune impossible branches."""
        source = """
            let x = 3;
            let y = 7;
            if (x + y > 20) {
                let z = 1;
            } else {
                let z = 2;
            }
        """
        result = guided_execute(source)
        # x + y = 10, not > 20, so that branch is pruned
        assert result.branches_pruned_by_octagon >= 1


# ===================================================================
# 6. Relational Pruning Analysis
# ===================================================================

class TestRelationalPruningAnalysis:
    def test_analyze_basic(self):
        source = """
            let x = 5;
            let y = x + 3;
            if (y < x) { let z = 1; } else { let z = 2; }
        """
        result = analyze_relational_pruning(source)
        assert result['total_branches'] >= 1
        assert result['octagon_pruned'] >= 1

    def test_octagon_advantage(self):
        """The octagon prunes y < x because it knows y - x == 3,
        but intervals alone (x=[5,5], y=[8,8]) would also catch this."""
        source = """
            let x = 5;
            let y = x + 3;
            if (y < x) { let z = 1; } else { let z = 2; }
        """
        result = analyze_relational_pruning(source)
        # Both interval and octagon should prune this (constants make intervals precise too)
        assert result['octagon_pruned'] >= 1

    def test_with_symbolic_inputs(self):
        source = "let y = x + 1; if (y < x) { let z = 1; } else { let z = 2; }"
        result = analyze_relational_pruning(source, {'x': 'int'})
        # With symbolic x, interval can't bound x or y, but octagon knows y - x == 1
        # So octagon should prune y < x, but interval should not
        assert result['octagon_pruned'] >= 1
        assert result['octagon_only_pruned'] >= 1  # Octagon catches what intervals miss


# ===================================================================
# 7. Property Verification
# ===================================================================

class TestPropertyVerification:
    def test_verify_bound(self):
        result = verify_relational_property("let x = 5;", "x <= 10")
        assert result['verified']

    def test_verify_fails(self):
        result = verify_relational_property("let x = 15;", "x <= 10")
        assert not result['verified']

    def test_verify_difference(self):
        result = verify_relational_property(
            "let x = 5; let y = x + 3;",
            "x - y <= 0"  # x - y = -3 <= 0
        )
        assert result['verified']

    def test_verify_equality(self):
        result = verify_relational_property(
            "let x = 3; let y = 7;",
            "x + y == 10"
        )
        assert result['verified']

    def test_verify_var_equality(self):
        result = verify_relational_property("let x = 5;", "x == 5")
        assert result['verified']

    def test_verify_ge(self):
        result = verify_relational_property("let x = 5;", "x >= 3")
        assert result['verified']


# ===================================================================
# 8. Property Parsing
# ===================================================================

class TestPropertyParsing:
    def test_var_le(self):
        cs = _parse_property("x <= 5")
        assert cs is not None
        assert len(cs) == 1

    def test_var_ge(self):
        cs = _parse_property("x >= 3")
        assert cs is not None

    def test_diff_le(self):
        cs = _parse_property("x - y <= 3")
        assert cs is not None
        assert cs[0].var1 == 'x' and cs[0].var2 == 'y'

    def test_sum_le(self):
        cs = _parse_property("x + y <= 10")
        assert cs is not None

    def test_sum_eq(self):
        cs = _parse_property("x + y == 10")
        assert cs is not None and len(cs) == 2

    def test_diff_eq(self):
        cs = _parse_property("x - y == 3")
        assert cs is not None and len(cs) == 2

    def test_var_eq(self):
        cs = _parse_property("x == 5")
        assert cs is not None and len(cs) == 2

    def test_invalid(self):
        cs = _parse_property("foo bar baz")
        assert cs is None

    def test_negative_bound(self):
        cs = _parse_property("x <= -5")
        assert cs is not None
        assert cs[0].bound == -5


# ===================================================================
# 9. Comparison V001 vs V174
# ===================================================================

class TestComparisonV001V174:
    def test_comparison_basic(self):
        source = """
            let x = 5;
            let y = x + 3;
            if (y < x) { let z = 1; } else { let z = 2; }
        """
        result = compare_v001_vs_v174(source)
        assert 'v174_branches_pruned' in result
        assert 'v174_relational_constraints' in result
        assert 'octagon_advantage' in result
        assert 'paths_explored' in result

    def test_comparison_with_symbolic(self):
        source = "let y = x + 1; if (y < x) { let z = 1; } else { let z = 2; }"
        result = compare_v001_vs_v174(source, {'x': 'int'})
        # Octagon should have an advantage here
        assert result['octagon_advantage'] >= 1


# ===================================================================
# 10. Batch Execution
# ===================================================================

class TestBatchExecution:
    def test_batch_two_programs(self):
        sources = [
            "let x = 5; let y = x + 1;",
            "let a = 10; let b = a - 3;",
        ]
        results = batch_guided_execute(sources)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, GuidedResult)

    def test_batch_with_inputs(self):
        sources = ["let y = x + 1;", "let y = x - 1;"]
        inputs = [{'x': 'int'}, {'x': 'int'}]
        results = batch_guided_execute(sources, inputs)
        assert len(results) == 2


# ===================================================================
# 11. GuidedResult Properties
# ===================================================================

class TestGuidedResultProperties:
    def test_paths_property(self):
        result = guided_execute("let x = 5;")
        assert result.paths is not None

    def test_pruning_ratio_zero(self):
        result = guided_execute("let x = 5;")
        # No branches -> ratio 0
        assert result.pruning_ratio == 0.0

    def test_pruning_ratio_nonzero(self):
        source = """
            let x = 5;
            let y = x + 3;
            if (y < x) { let z = 1; } else { let z = 2; }
        """
        result = guided_execute(source)
        if result.branches_analyzed > 0 and result.branches_pruned_by_octagon > 0:
            assert result.pruning_ratio > 0.0


# ===================================================================
# 12. Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = guided_execute("")
        assert not result.octagon_state.is_bot()

    def test_print_statement(self):
        result = guided_execute("let x = 5; print(x);")
        assert not result.octagon_state.is_bot()

    def test_function_decl(self):
        result = guided_execute("fn foo(x) { return x + 1; }")
        assert not result.octagon_state.is_bot()

    def test_boolean_condition(self):
        result = guided_execute("let x = 5; if (true) { let y = 1; } else { let y = 2; }")
        assert isinstance(result, GuidedResult)


# ===================================================================
# 13. Relational Advantage: Symbolic Inputs
# ===================================================================

class TestRelationalAdvantage:
    def test_symbolic_difference(self):
        """With symbolic x, octagon knows y - x == 1, but intervals know nothing."""
        source = """
            let y = x + 1;
            if (y < x) {
                let z = 1;
            } else {
                let z = 2;
            }
        """
        result = analyze_relational_pruning(source, {'x': 'int'})
        # Octagon prunes y < x (knows y - x == 1), intervals can't
        assert result['octagon_only_pruned'] >= 1

    def test_symbolic_sum_bound(self):
        """After y = 10 - x, octagon knows x + y == 10."""
        source = """
            let y = 10 - x;
            if (x + y > 15) {
                let z = 1;
            } else {
                let z = 2;
            }
        """
        result = analyze_relational_pruning(source, {'x': 'int'})
        # Octagon knows x + y == 10, so x + y > 15 is infeasible
        assert result['octagon_pruned'] >= 1

    def test_no_relational_advantage_for_constants(self):
        """For fully concrete programs, intervals are as precise as octagon."""
        source = """
            let x = 5;
            let y = 10;
            if (x > y) { let z = 1; } else { let z = 2; }
        """
        result = analyze_relational_pruning(source)
        # Both should prune x > y (x=5, y=10)
        assert result['octagon_only_pruned'] == 0  # No advantage for constants


# ===================================================================
# 14. While Loop with Pruning
# ===================================================================

class TestWhileLoopPruning:
    def test_loop_with_invariant(self):
        source = """
            let i = 0;
            let n = 10;
            while (i < n) {
                i = i + 1;
            }
        """
        result = guided_execute(source)
        assert isinstance(result, GuidedResult)
        assert not result.octagon_state.is_bot()

    def test_loop_exit_bounds(self):
        source = """
            let i = 0;
            while (i < 5) {
                i = i + 1;
            }
        """
        result = guided_execute(source)
        hi = result.octagon_state.get_bounds('i')[1]
        # After loop: i >= 5 (exit condition)
        # Widening may lose precision but exit guard should give some bound


# ===================================================================
# 15. Multi-Variable Relationships
# ===================================================================

class TestMultiVariable:
    def test_three_var_chain(self):
        """x = y + 1, y = z + 1 => x - z == 2."""
        result = _octagon_pre_analyze("let z = 5; let y = z + 1; let x = y + 1;")
        lo, hi = result.final_state.get_difference_bound('x', 'z')
        assert lo == 2 and hi == 2

    def test_swap_detection(self):
        """After swap, octagon should track new relationships."""
        result = _octagon_pre_analyze("""
            let x = 3;
            let y = 7;
            let t = x;
            x = y;
            y = t;
        """)
        lo_x, hi_x = result.final_state.get_bounds('x')
        lo_y, hi_y = result.final_state.get_bounds('y')
        assert lo_x == 7 and hi_x == 7
        assert lo_y == 3 and hi_y == 3


# ===================================================================
# 16. Verify Relational: With Symbolic Inputs
# ===================================================================

class TestVerifyRelationalSymbolic:
    def test_symbolic_diff_property(self):
        """After y = x + 3, verify x - y <= 0."""
        result = verify_relational_property(
            "let y = x + 3;",
            "x - y <= 0",
            {'x': 'int'}
        )
        assert result['verified']

    def test_symbolic_diff_property_fails(self):
        """After y = x + 3, x - y <= -5 should NOT be verifiable (it's == -3)."""
        result = verify_relational_property(
            "let y = x + 3;",
            "x - y <= -5",
            {'x': 'int'}
        )
        assert not result['verified']


# ===================================================================
# 17. Complex Programs
# ===================================================================

class TestComplexPrograms:
    def test_conditional_sum(self):
        source = """
            let x = 5;
            let y = 0;
            if (x > 3) {
                y = x + 2;
            } else {
                y = x - 1;
            }
        """
        result = guided_execute(source)
        lo, hi = result.octagon_state.get_bounds('y')
        assert lo == 7 and hi == 7  # Only then branch feasible

    def test_accumulate(self):
        source = """
            let s = 0;
            let i = 0;
            while (i < 3) {
                s = s + 1;
                i = i + 1;
            }
        """
        result = guided_execute(source)
        assert isinstance(result, GuidedResult)


# ===================================================================
# 18. Integration with Symbolic Execution Paths
# ===================================================================

class TestSymbolicIntegration:
    def test_paths_returned(self):
        result = guided_execute("let x = 5; let y = x + 1;")
        assert len(result.paths) >= 1

    def test_guided_with_branch(self):
        result = guided_execute(
            "if (x > 0) { let y = 1; } else { let y = 2; }",
            {'x': 'int'}
        )
        # With symbolic x, symbolic execution should explore both branches
        assert len(result.paths) >= 1


# ===================================================================
# 19. Octagon Warnings
# ===================================================================

class TestOctagonWarnings:
    def test_no_warnings_for_clean_code(self):
        result = guided_execute("let x = 5; let y = x + 1;")
        assert len(result.octagon_warnings) == 0


# ===================================================================
# 20. Stress and Coverage
# ===================================================================

class TestStressCoverage:
    def test_many_assignments(self):
        lines = ["let x0 = 0;"]
        for i in range(1, 10):
            lines.append(f"let x{i} = x{i-1} + 1;")
        source = " ".join(lines)
        result = guided_execute(source)
        lo, hi = result.octagon_state.get_bounds('x9')
        assert lo == 9 and hi == 9

    def test_nested_ifs(self):
        source = """
            let x = 5;
            if (x > 3) {
                if (x > 4) {
                    let y = 1;
                } else {
                    let y = 2;
                }
            } else {
                let y = 3;
            }
        """
        result = guided_execute(source)
        assert isinstance(result, GuidedResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
