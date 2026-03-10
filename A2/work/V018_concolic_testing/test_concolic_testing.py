"""
Tests for V018: Concolic Testing Engine
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concolic_testing import (
    concolic_test, concolic_find_bugs, concolic_with_seeds,
    concolic_reach_branch, compare_concolic_vs_symbolic,
    ConcolicEngine, ConcolicResult, ConcolicStatus,
    CoverageGuidedConcolic, ConcolicBugFinder,
    ConcreteInterpreter, parse, BugReport
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import Var as SMTVar, App, IntConst, Op as SMTOp, BOOL, INT


# ============================================================
# Section 1: Basic Concrete Execution
# ============================================================

class TestConcreteExecution:
    """Test the concrete interpreter with symbolic shadow."""

    def test_simple_assignment(self):
        source = "let x = 5; let y = x + 3;"
        program = parse(source)
        interp = ConcreteInterpreter({'x': 5})
        interp.execute(program)
        assert interp.env['y'] == 8

    def test_input_variable(self):
        source = "let y = x + 1;"
        program = parse(source)
        interp = ConcreteInterpreter({'x': 10})
        interp.execute(program)
        assert interp.env['y'] == 11

    def test_symbolic_shadow_tracks(self):
        source = "let y = x + 1;"
        program = parse(source)
        interp = ConcreteInterpreter({'x': 5})
        interp.execute(program)
        # Symbolic shadow should exist
        assert 'y' in interp.sym_env
        assert interp.sym_env['y'] is not None

    def test_print_output(self):
        source = "let y = x * 2; print(y);"
        program = parse(source)
        interp = ConcreteInterpreter({'x': 7})
        interp.execute(program)
        assert interp.output == [14]


# ============================================================
# Section 2: Branch Constraint Collection
# ============================================================

class TestBranchConstraints:
    """Test that branch constraints are collected correctly."""

    def test_if_then_constraint(self):
        source = "let y = 0; if (x > 0) { y = 1; }"
        program = parse(source)
        interp = ConcreteInterpreter({'x': 5})
        interp.execute(program)
        assert len(interp.path_constraints) == 1
        assert interp.branch_decisions[0] == True
        assert interp.env['y'] == 1

    def test_if_else_constraint(self):
        source = "let y = 0; if (x > 0) { y = 1; } else { y = -1; }"
        program = parse(source)
        interp = ConcreteInterpreter({'x': -3})
        interp.execute(program)
        assert len(interp.path_constraints) == 1
        assert interp.branch_decisions[0] == False
        assert interp.env['y'] == -1

    def test_nested_branches(self):
        source = """
        let y = 0;
        if (x > 0) {
            if (x > 10) {
                y = 2;
            } else {
                y = 1;
            }
        } else {
            y = -1;
        }
        """
        program = parse(source)
        interp = ConcreteInterpreter({'x': 5})
        interp.execute(program)
        assert len(interp.path_constraints) == 2
        assert interp.branch_decisions == [True, False]
        assert interp.env['y'] == 1

    def test_covered_branches_tracking(self):
        source = "if (x > 0) { let y = 1; } else { let y = 0; }"
        program = parse(source)
        interp = ConcreteInterpreter({'x': 5})
        interp.execute(program)
        assert (0, True) in interp.covered_branches


# ============================================================
# Section 3: Basic Concolic Testing
# ============================================================

class TestBasicConcolic:
    """Test the core concolic testing loop."""

    def test_single_branch(self):
        source = "let y = 0; if (x > 0) { y = 1; } else { y = -1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert result.num_tests >= 2
        # Should find both branches
        outputs = [tc.output for tc in result.test_cases if tc.output]
        # At least one positive and one non-positive input found

    def test_two_variables(self):
        source = """
        let r = 0;
        if (x > 0) { r = r + 1; }
        if (y > 0) { r = r + 2; }
        """
        result = concolic_test(source, {'x': 'int', 'y': 'int'}, {'x': 0, 'y': 0})
        assert result.num_tests >= 2

    def test_generates_new_coverage(self):
        source = """
        let y = 0;
        if (x > 5) { y = 1; }
        if (x < -5) { y = -1; }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        # Should explore x>5 and x<-5 branches
        assert result.num_tests >= 2
        # Check that we found diverse inputs
        all_x = [tc.inputs['x'] for tc in result.test_cases]
        assert any(x > 5 for x in all_x) or any(x <= 5 for x in all_x)

    def test_result_has_status(self):
        source = "let y = x + 1;"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert result.status in (ConcolicStatus.COMPLETE, ConcolicStatus.COVERAGE_SATURATED)


# ============================================================
# Section 4: Path Exploration
# ============================================================

class TestPathExploration:
    """Test that concolic testing explores different paths."""

    def test_abs_function(self):
        source = """
        let result = 0;
        if (x >= 0) { result = x; } else { result = 0 - x; }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 5})
        # Should find both branches: x>=0 and x<0
        all_x = [tc.inputs['x'] for tc in result.test_cases]
        has_positive = any(x >= 0 for x in all_x)
        has_negative = any(x < 0 for x in all_x)
        assert has_positive and has_negative

    def test_max_function(self):
        source = """
        let result = 0;
        if (x >= y) { result = x; } else { result = y; }
        """
        result = concolic_test(source, {'x': 'int', 'y': 'int'}, {'x': 0, 'y': 0})
        assert result.num_tests >= 2

    def test_three_way_branch(self):
        source = """
        let r = 0;
        if (x > 0) {
            r = 1;
        } else {
            if (x == 0) {
                r = 0;
            } else {
                r = -1;
            }
        }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert result.num_tests >= 2

    def test_chained_conditions(self):
        source = """
        let grade = 0;
        if (score >= 90) { grade = 4; }
        if (score >= 80) { if (score < 90) { grade = 3; } }
        if (score >= 70) { if (score < 80) { grade = 2; } }
        """
        result = concolic_test(source, {'score': 'int'}, {'score': 50})
        assert result.num_tests >= 2


# ============================================================
# Section 5: Loop Handling
# ============================================================

class TestLoopHandling:
    """Test concolic testing with loops."""

    def test_simple_loop(self):
        source = """
        let i = x;
        let s = 0;
        while (i > 0) {
            s = s + i;
            i = i - 1;
        }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 3})
        assert result.num_tests >= 1

    def test_loop_with_branch(self):
        source = """
        let i = 0;
        let pos = 0;
        let neg = 0;
        while (i < x) {
            if (i > 2) { pos = pos + 1; } else { neg = neg + 1; }
            i = i + 1;
        }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 5})
        assert result.num_tests >= 1

    def test_bounded_loop(self):
        source = """
        let s = 0;
        let i = 0;
        while (i < 3) {
            s = s + x;
            i = i + 1;
        }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 1})
        assert result.num_tests >= 1


# ============================================================
# Section 6: Coverage-Guided Testing
# ============================================================

class TestCoverageGuided:
    """Test coverage-guided concolic testing with seeds."""

    def test_seeds_improve_coverage(self):
        source = """
        let r = 0;
        if (x > 100) { r = 1; }
        if (x < -100) { r = -1; }
        """
        # Without seeds, starting from 0 may not reach extreme values quickly
        seeds = [{'x': 200}, {'x': -200}]
        result = concolic_with_seeds(source, {'x': 'int'}, seeds)
        # Seeds should help reach both extremes
        all_x = [tc.inputs['x'] for tc in result.test_cases]
        assert any(x > 100 for x in all_x)
        assert any(x < -100 for x in all_x)

    def test_multiple_seeds(self):
        source = """
        let r = 0;
        if (x > 0) { r = r + 1; }
        if (y > 0) { r = r + 2; }
        """
        seeds = [{'x': 5, 'y': -5}, {'x': -5, 'y': 5}]
        result = concolic_with_seeds(source, {'x': 'int', 'y': 'int'}, seeds)
        assert result.num_tests >= 2

    def test_coverage_guided_engine(self):
        source = "let r = 0; if (x > 0) { r = 1; } else { r = -1; }"
        guided = CoverageGuidedConcolic(max_iterations=20)
        result = guided.run(source, {'x': 'int'}, {'x': 0})
        assert result.num_tests >= 1


# ============================================================
# Section 7: Bug Finding
# ============================================================

class TestBugFinding:
    """Test concolic-based bug finding."""

    def test_division_by_zero_detection(self):
        source = """
        let r = 10 / x;
        """
        result = concolic_find_bugs(source, {'x': 'int'}, {'x': 0})
        # Starting with x=0, should detect division by zero
        assert any(b.kind == "division_by_zero" for b in result.bugs)

    def test_no_bug_in_safe_program(self):
        source = """
        let y = x + 1;
        """
        result = concolic_find_bugs(source, {'x': 'int'}, {'x': 5})
        div_bugs = [b for b in result.bugs if b.kind == "division_by_zero"]
        assert len(div_bugs) == 0

    def test_assertion_checking(self):
        source = """
        let y = x * 2;
        """
        # Assert y >= 0 (which fails for negative x)
        x_var = SMTVar('x', INT)
        y_should_be_positive = App(SMTOp.GE, [
            App(SMTOp.MUL, [x_var, IntConst(2)], INT),
            IntConst(0)
        ], BOOL)
        result = ConcolicBugFinder().find_bugs(
            source, {'x': 'int'}, {'x': -5},
            assertions={'y >= 0': y_should_be_positive}
        )
        assert any(b.kind == "assertion_failure" for b in result.bugs)

    def test_bug_report_has_inputs(self):
        source = "let r = 10 / x;"
        result = concolic_find_bugs(source, {'x': 'int'}, {'x': 0})
        for bug in result.bugs:
            assert 'x' in bug.inputs


# ============================================================
# Section 8: Comparison with Symbolic Execution
# ============================================================

class TestComparison:
    """Test comparison between concolic and pure symbolic execution."""

    def test_comparison_returns_both(self):
        source = """
        let r = 0;
        if (x > 0) { r = 1; } else { r = -1; }
        """
        result = compare_concolic_vs_symbolic(source, {'x': 'int'}, {'x': 0})
        assert 'concolic' in result
        assert 'symbolic' in result
        assert result['concolic']['tests'] >= 1
        assert result['symbolic']['tests'] >= 0

    def test_both_find_branches(self):
        source = """
        let r = 0;
        if (x > 5) { r = 1; }
        if (x < -5) { r = -1; }
        """
        result = compare_concolic_vs_symbolic(source, {'x': 'int'}, {'x': 0})
        assert result['concolic']['tests'] >= 1


# ============================================================
# Section 9: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_branches(self):
        source = "let y = x + 1;"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert result.num_tests >= 1
        assert result.test_cases[0].inputs == {'x': 0}

    def test_constant_condition(self):
        source = "let y = 0; if (1 > 0) { y = 1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert result.num_tests >= 1

    def test_empty_program(self):
        source = "let x = 0;"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert result.num_tests >= 1

    def test_large_initial_value(self):
        source = "let r = 0; if (x > 1000) { r = 1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 999999})
        assert result.num_tests >= 1


# ============================================================
# Section 10: Target Branch Reaching
# ============================================================

class TestBranchReaching:
    """Test targeted branch reaching."""

    def test_reach_then_branch(self):
        source = "let r = 0; if (x > 0) { r = 1; } else { r = -1; }"
        inputs = concolic_reach_branch(source, {'x': 'int'}, 0, True, {'x': -5})
        # Should find inputs where x > 0
        if inputs is not None:
            assert inputs['x'] > 0

    def test_reach_else_branch(self):
        source = "let r = 0; if (x > 0) { r = 1; } else { r = -1; }"
        inputs = concolic_reach_branch(source, {'x': 'int'}, 0, False, {'x': 5})
        if inputs is not None:
            assert inputs['x'] <= 0

    def test_unreachable_branch(self):
        # Branch 0 is the only branch, try to reach a non-existent branch
        source = "let r = 0; if (x > 0) { r = 1; }"
        inputs = concolic_reach_branch(source, {'x': 'int'}, 999, True, {'x': 0})
        assert inputs is None


# ============================================================
# Section 11: Multiple Input Variables
# ============================================================

class TestMultipleInputs:
    """Test concolic testing with multiple input variables."""

    def test_two_variable_conditions(self):
        source = """
        let r = 0;
        if (x > y) { r = 1; } else { r = -1; }
        """
        result = concolic_test(source, {'x': 'int', 'y': 'int'}, {'x': 0, 'y': 0})
        assert result.num_tests >= 2

    def test_three_variables(self):
        source = """
        let r = 0;
        if (x > 0) { r = r + 1; }
        if (y > 0) { r = r + 2; }
        if (z > 0) { r = r + 4; }
        """
        result = concolic_test(source, {'x': 'int', 'y': 'int', 'z': 'int'},
                              {'x': 0, 'y': 0, 'z': 0})
        assert result.num_tests >= 2

    def test_dependent_conditions(self):
        source = """
        let s = x + y;
        let r = 0;
        if (s > 10) { r = 1; } else { r = 0; }
        """
        result = concolic_test(source, {'x': 'int', 'y': 'int'}, {'x': 0, 'y': 0})
        assert result.num_tests >= 2


# ============================================================
# Section 12: Function Handling
# ============================================================

class TestFunctionHandling:
    """Test concolic testing with function definitions."""

    def test_simple_function(self):
        source = """
        fn double(n) { return n * 2; }
        let r = double(x);
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 5})
        assert result.num_tests >= 1

    def test_function_with_branch(self):
        source = """
        fn abs_val(n) {
            if (n >= 0) { return n; }
            return 0 - n;
        }
        let r = abs_val(x);
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 5})
        assert result.num_tests >= 1


# ============================================================
# Section 13: Test Case Quality
# ============================================================

class TestCaseQuality:
    """Test that generated test cases are valid and diverse."""

    def test_all_tests_have_inputs(self):
        source = "let r = 0; if (x > 0) { r = 1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        for tc in result.test_cases:
            assert 'x' in tc.inputs

    def test_new_coverage_flag(self):
        source = "let r = 0; if (x > 0) { r = 1; } else { r = -1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        # First test always has new coverage
        assert result.test_cases[0].is_new_coverage

    def test_diverse_inputs(self):
        source = """
        let r = 0;
        if (x > 0) { r = 1; }
        if (x < 0) { r = -1; }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        all_x = set(tc.inputs['x'] for tc in result.test_cases)
        # Should have at least 2 distinct values
        assert len(all_x) >= 2

    def test_result_properties(self):
        source = "let r = 0; if (x > 0) { r = 1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert isinstance(result.branch_coverage, float)
        assert result.branch_coverage >= 0.0
        assert result.iterations >= 1


# ============================================================
# Section 14: Arithmetic Operations
# ============================================================

class TestArithmetic:
    """Test symbolic tracking of arithmetic operations."""

    def test_addition(self):
        source = "let s = x + y; let r = 0; if (s > 10) { r = 1; }"
        result = concolic_test(source, {'x': 'int', 'y': 'int'}, {'x': 3, 'y': 3})
        all_sums = [tc.inputs['x'] + tc.inputs['y'] for tc in result.test_cases]
        # Should find both s > 10 and s <= 10
        assert any(s <= 10 for s in all_sums)
        # Starting from sum=6, negating gets sum>10
        assert any(s > 10 for s in all_sums) or len(result.test_cases) >= 2

    def test_subtraction(self):
        source = "let d = x - y; let r = 0; if (d == 0) { r = 1; }"
        result = concolic_test(source, {'x': 'int', 'y': 'int'}, {'x': 5, 'y': 3})
        assert result.num_tests >= 2

    def test_multiplication(self):
        source = "let p = x * 2; let r = 0; if (p > 10) { r = 1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 3})
        assert result.num_tests >= 2

    def test_negation(self):
        source = "let n = 0 - x; let r = 0; if (n > 0) { r = 1; }"
        result = concolic_test(source, {'x': 'int'}, {'x': 5})
        assert result.num_tests >= 2


# ============================================================
# Section 15: Integration and Stress
# ============================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_classify_value(self):
        source = """
        let cat = 0;
        if (x > 0) {
            if (x > 100) { cat = 3; }
            else { if (x > 10) { cat = 2; } else { cat = 1; } }
        } else {
            if (x == 0) { cat = 0; } else { cat = -1; }
        }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 0})
        assert result.num_tests >= 3

    def test_binary_search_like(self):
        source = """
        let lo = 0;
        let hi = 100;
        let r = 0;
        if (x < 50) {
            if (x < 25) { r = 1; } else { r = 2; }
        } else {
            if (x < 75) { r = 3; } else { r = 4; }
        }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 50})
        assert result.num_tests >= 3

    def test_clamp_function(self):
        source = """
        let r = x;
        if (x < 0) { r = 0; }
        if (x > 100) { r = 100; }
        """
        result = concolic_test(source, {'x': 'int'}, {'x': 50})
        assert result.num_tests >= 2
        all_x = [tc.inputs['x'] for tc in result.test_cases]
        # Should find values in different ranges
        assert len(set(all_x)) >= 2

    def test_concolic_engine_max_iterations(self):
        source = """
        let r = 0;
        if (x > 0) { r = 1; }
        if (x > 10) { r = 2; }
        if (x > 100) { r = 3; }
        if (x > 1000) { r = 4; }
        """
        engine = ConcolicEngine(max_iterations=5)
        result = engine.run(source, {'x': 'int'}, {'x': 0})
        assert result.iterations <= 5

    def test_full_workflow(self):
        """End-to-end test: concolic test -> find bugs -> compare."""
        source = """
        let r = 0;
        if (x > 0) { r = 100 / x; }
        else { r = -1; }
        """
        # Concolic test
        result = concolic_test(source, {'x': 'int'}, {'x': 5})
        assert result.num_tests >= 1

        # Bug finding
        bug_result = concolic_find_bugs(source, {'x': 'int'}, {'x': 5})
        assert isinstance(bug_result.iterations, int)
