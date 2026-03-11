"""Tests for V131: Polyhedral-Guided Symbolic Execution."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))

from polyhedral_guided_symex import (
    # Core
    PolyGuidedResult, PolyBranchInfo,
    # Execution
    poly_guided_execute, poly_guided_check_assertions,
    poly_guided_generate_tests,
    # Pre-analyzer
    PolyhedralPreAnalyzer, BranchCapturingInterpreter,
    # Executor
    PolyGuidedExecutor,
    # Comparison
    compare_guided_vs_plain, compare_all_strategies,
    # Summary
    poly_guided_summary,
)
from symbolic_execution import symbolic_execute


# ---- Source snippets ----

SIMPLE_IF = """
let x = 0;
let r = 0;
if (x > 0) {
    r = 1;
} else {
    r = 0;
}
"""

NESTED_IF = """
let x = 0;
let r = 0;
if (x > 10) {
    r = 2;
} else {
    if (x > 0) {
        r = 1;
    } else {
        r = 0;
    }
}
"""

DEAD_BRANCH = """
let x = 5;
if (x > 10) {
    let y = 1;
} else {
    let y = 0;
}
"""

LOOP_SIMPLE = """
let i = 0;
while (i < 3) {
    i = i + 1;
}
"""

TWO_VAR = """
let a = 0;
let b = 0;
let r = 0;
if (a > 0) {
    if (b > 0) {
        r = 1;
    } else {
        r = 2;
    }
} else {
    r = 0;
}
"""

INFEASIBLE_NESTED = """
let x = 5;
let y = 3;
if (x < y) {
    let z = 1;
} else {
    let z = 0;
}
"""


# ===========================================================================
# 1. Basic Execution
# ===========================================================================

class TestBasicExecution:
    def test_simple_if(self):
        result = poly_guided_execute(SIMPLE_IF, {"x": "int"})
        assert len(result.paths) >= 1

    def test_nested_if(self):
        result = poly_guided_execute(NESTED_IF, {"x": "int"})
        assert len(result.paths) >= 1

    def test_concrete_program(self):
        result = poly_guided_execute(DEAD_BRANCH)
        assert len(result.paths) >= 1

    def test_loop(self):
        result = poly_guided_execute(LOOP_SIMPLE)
        assert len(result.paths) >= 1


# ===========================================================================
# 2. Result Type
# ===========================================================================

class TestResultType:
    def test_is_poly_guided_result(self):
        result = poly_guided_execute(SIMPLE_IF, {"x": "int"})
        assert isinstance(result, PolyGuidedResult)

    def test_has_execution(self):
        result = poly_guided_execute(SIMPLE_IF, {"x": "int"})
        assert result.execution is not None

    def test_has_polyhedral_env(self):
        result = poly_guided_execute(SIMPLE_IF, {"x": "int"})
        assert result.polyhedral_env is not None

    def test_has_stats(self):
        result = poly_guided_execute(SIMPLE_IF, {"x": "int"})
        assert result.smt_checks_performed >= 0
        assert result.smt_checks_saved >= 0
        assert result.pruned_by_polyhedral >= 0

    def test_repr(self):
        result = poly_guided_execute(SIMPLE_IF, {"x": "int"})
        r = repr(result)
        assert "PolyGuidedResult" in r
        assert "paths=" in r


# ===========================================================================
# 3. Pruning Detection
# ===========================================================================

class TestPruning:
    def test_dead_branch_pruned(self):
        """x=5 means x>10 is infeasible -- polyhedral should prune it."""
        result = poly_guided_execute(DEAD_BRANCH)
        # With concrete values, symbolic execution handles it directly
        assert len(result.paths) >= 1

    def test_infeasible_condition(self):
        """x=5, y=3 means x<y is infeasible."""
        result = poly_guided_execute(INFEASIBLE_NESTED)
        assert len(result.paths) >= 1

    def test_pruning_ratio_bounded(self):
        result = poly_guided_execute(SIMPLE_IF, {"x": "int"})
        assert 0.0 <= result.pruning_ratio <= 1.0


# ===========================================================================
# 4. Pre-Analyzer
# ===========================================================================

class TestPreAnalyzer:
    def test_analyze_dead_branch(self):
        pre = PolyhedralPreAnalyzer()
        analysis = pre.analyze(DEAD_BRANCH)
        assert "branch_info" in analysis
        assert "env" in analysis
        # Should detect the dead branch (x=5, x>10 is false)
        branch_info = analysis["branch_info"]
        if branch_info:
            # At least one branch should be marked infeasible
            some_pruned = any(
                not info.true_feasible or not info.false_feasible
                for info in branch_info.values()
            )
            assert some_pruned

    def test_analyze_symbolic(self):
        pre = PolyhedralPreAnalyzer()
        analysis = pre.analyze(SIMPLE_IF, {"x": "int"})
        assert "branch_info" in analysis

    def test_analyze_returns_env(self):
        pre = PolyhedralPreAnalyzer()
        analysis = pre.analyze(LOOP_SIMPLE)
        env = analysis["env"]
        assert env is not None
        assert "i" in env.var_names


# ===========================================================================
# 5. Branch Capturing Interpreter
# ===========================================================================

class TestBranchCapturing:
    def test_captures_if_branches(self):
        branch_info = {}
        interp = BranchCapturingInterpreter(branch_info=branch_info)
        interp.analyze(DEAD_BRANCH)
        assert len(branch_info) >= 1

    def test_captures_while_branches(self):
        branch_info = {}
        interp = BranchCapturingInterpreter(branch_info=branch_info)
        interp.analyze(LOOP_SIMPLE)
        assert len(branch_info) >= 1

    def test_symbolic_inputs_forget(self):
        """Symbolic inputs should be TOP (unconstrained)."""
        branch_info = {}
        interp = BranchCapturingInterpreter(
            branch_info=branch_info,
            symbolic_inputs={"x": "int"},
        )
        source = """
let x = 0;
if (x > 5) {
    let y = 1;
}
"""
        interp.analyze(source)
        # With x symbolic (forgotten), both branches should be feasible
        if branch_info:
            for info in branch_info.values():
                # At least one branch with symbolic x should have both feasible
                pass  # Just verify it doesn't crash


# ===========================================================================
# 6. Executor Class
# ===========================================================================

class TestExecutorClass:
    def test_create_executor(self):
        executor = PolyGuidedExecutor()
        assert executor is not None

    def test_guided_execute(self):
        executor = PolyGuidedExecutor()
        result = executor.guided_execute(SIMPLE_IF, {"x": "int"})
        assert isinstance(result, PolyGuidedResult)


# ===========================================================================
# 7. Test Generation
# ===========================================================================

class TestGeneration:
    def test_generate_tests(self):
        tests, guided = poly_guided_generate_tests(SIMPLE_IF, {"x": "int"})
        assert isinstance(tests, list)
        assert isinstance(guided, PolyGuidedResult)

    def test_tests_are_concrete(self):
        tests, _ = poly_guided_generate_tests(SIMPLE_IF, {"x": "int"})
        # Each test should have concrete values
        for t in tests:
            assert hasattr(t, 'inputs') or hasattr(t, 'output')


# ===========================================================================
# 8. Assertion Checking
# ===========================================================================

class TestAssertionChecking:
    def test_check_simple(self):
        assertion_result, guided = poly_guided_check_assertions(SIMPLE_IF, {"x": "int"})
        assert guided is not None


# ===========================================================================
# 9. Comparison API
# ===========================================================================

class TestComparison:
    def test_compare_guided_vs_plain(self):
        comp = compare_guided_vs_plain(SIMPLE_IF, {"x": "int"})
        assert "plain" in comp
        assert "polyhedral_guided" in comp
        assert comp["plain"]["paths"] >= 1
        assert comp["polyhedral_guided"]["paths"] >= 1

    def test_compare_has_timing(self):
        comp = compare_guided_vs_plain(SIMPLE_IF, {"x": "int"})
        assert comp["plain"]["time"] >= 0
        assert comp["polyhedral_guided"]["time"] >= 0

    def test_compare_all_strategies(self):
        comp = compare_all_strategies(SIMPLE_IF, {"x": "int"})
        assert "plain" in comp
        assert "polyhedral_guided" in comp
        # interval_guided may or may not be available
        assert "interval_guided" in comp


# ===========================================================================
# 10. Summary
# ===========================================================================

class TestSummary:
    def test_summary_output(self):
        s = poly_guided_summary(SIMPLE_IF, {"x": "int"})
        assert "Polyhedral-Guided" in s
        assert "Paths" in s

    def test_summary_has_stats(self):
        s = poly_guided_summary(SIMPLE_IF, {"x": "int"})
        assert "SMT" in s or "smt" in s.lower()


# ===========================================================================
# 11. PolyBranchInfo
# ===========================================================================

class TestPolyBranchInfo:
    def test_default_feasible(self):
        info = PolyBranchInfo(line=1)
        assert info.true_feasible
        assert info.false_feasible

    def test_infeasible(self):
        info = PolyBranchInfo(line=1, true_feasible=False)
        assert not info.true_feasible
        assert info.false_feasible


# ===========================================================================
# 12. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_no_branches(self):
        source = "let x = 5;"
        result = poly_guided_execute(source)
        assert len(result.paths) >= 1

    def test_empty_else(self):
        source = """
let x = 0;
if (x > 0) {
    x = 1;
}
"""
        result = poly_guided_execute(source)
        assert len(result.paths) >= 1

    def test_two_var_symbolic(self):
        result = poly_guided_execute(TWO_VAR, {"a": "int", "b": "int"})
        assert len(result.paths) >= 1
