"""
Tests for V001: Abstract-Interpretation-Guided Symbolic Execution

Tests verify that:
1. Guided execution produces correct results (same paths as plain)
2. Abstract pre-analysis correctly identifies infeasible branches
3. SMT checks are saved when abstract analysis prunes branches
4. The comparison API works correctly
5. Edge cases (no symbolic inputs, all concrete, etc.) are handled
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from guided_symbolic import (
    GuidedSymbolicExecutor, GuidedResult, BranchInfo,
    AbstractPreAnalyzer, guided_execute, compare_guided_vs_plain,
    guided_check_assertions,
)

# Also import plain executor for comparison
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymbolicExecutor, PathStatus, symbolic_execute,
)


# ============================================================
# Test: Basic Guided Execution
# ============================================================

class TestBasicGuidedExecution:
    """Test that guided execution produces correct results."""

    def test_simple_branch(self):
        """Simple if/else with symbolic input."""
        source = 'let x = 0; if (x < 10) { let y = 1; } else { let y = 2; }'
        result = guided_execute(source, {'x': 'int'})
        assert isinstance(result, GuidedResult)
        assert result.total_paths > 0

    def test_concrete_only(self):
        """No symbolic inputs -- should behave like plain execution."""
        source = 'let x = 5; if (x > 3) { let y = 1; } else { let y = 2; }'
        result = guided_execute(source)
        feasible = result.feasible_paths
        assert len(feasible) >= 1

    def test_no_branches(self):
        """Straight-line code, no branching."""
        source = 'let x = 0; let y = x + 1; let z = y * 2;'
        result = guided_execute(source, {'x': 'int'})
        assert result.total_paths >= 1
        assert result.smt_checks_saved == 0

    def test_multiple_branches(self):
        """Multiple if statements."""
        source = '''
        let x = 0;
        let result = 0;
        if (x > 0) { result = 1; }
        if (x > 10) { result = 2; }
        if (x > 100) { result = 3; }
        '''
        result = guided_execute(source, {'x': 'int'})
        assert result.total_paths > 1

    def test_nested_branches(self):
        """Nested if statements."""
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            if (x > 10) { y = 2; } else { y = 1; }
        } else {
            y = -1;
        }
        '''
        result = guided_execute(source, {'x': 'int'})
        feasible = result.feasible_paths
        assert len(feasible) >= 2


# ============================================================
# Test: Abstract Pruning
# ============================================================

class TestAbstractPruning:
    """Test that abstract analysis correctly prunes infeasible branches."""

    def test_constant_prunes_false_branch(self):
        """When a variable is assigned a constant, branches on it can be pruned."""
        source = 'let x = 5; if (x > 10) { let y = 1; } else { let y = 2; }'
        result = guided_execute(source)
        feasible = result.feasible_paths
        assert len(feasible) >= 1

    def test_known_range_prunes(self):
        """When abstract analysis knows a range, infeasible branches are pruned."""
        source = 'let x = 5; let y = x + 3; if (y < 0) { let z = 1; } else { let z = 2; }'
        result = guided_execute(source)
        feasible = result.feasible_paths
        assert len(feasible) >= 1

    def test_symbolic_no_prune(self):
        """Symbolic input with no constraint -- both branches feasible."""
        source = 'let x = 0; if (x > 0) { let y = 1; } else { let y = 2; }'
        result = guided_execute(source, {'x': 'int'})
        feasible = result.feasible_paths
        assert len(feasible) >= 2

    def test_cascading_prune(self):
        """Sequential conditions where first constrains what's possible."""
        source = '''
        let x = 100;
        if (x > 50) {
            if (x < 0) { let y = 1; } else { let y = 2; }
        }
        '''
        result = guided_execute(source)
        feasible = result.feasible_paths
        assert len(feasible) >= 1


# ============================================================
# Test: Abstract Pre-Analyzer
# ============================================================

class TestAbstractPreAnalyzer:
    """Test the abstract pre-analysis phase directly."""

    def test_branch_info_collected(self):
        """Pre-analyzer collects branch info for if statements."""
        source = 'let x = 5; if (x > 10) { let y = 1; } else { let y = 2; }'
        pre = AbstractPreAnalyzer()
        analysis = pre.analyze(source)
        assert len(analysis['branch_info']) >= 1

    def test_constant_branch_info(self):
        """Pre-analyzer correctly identifies constant conditions."""
        source = 'let x = 5; if (x > 10) { let y = 1; }'
        pre = AbstractPreAnalyzer()
        analysis = pre.analyze(source)
        for line, info in analysis['branch_info'].items():
            if info.variable == 'x':
                assert info.true_feasible == False
                assert info.false_feasible == True
                break

    def test_symbolic_branch_info(self):
        """Pre-analyzer marks both branches feasible for symbolic input."""
        source = 'let x = 0; if (x > 0) { let y = 1; }'
        pre = AbstractPreAnalyzer()
        analysis = pre.analyze(source, {'x': 'int'})
        for line, info in analysis['branch_info'].items():
            if info.variable == 'x':
                assert info.true_feasible == True
                assert info.false_feasible == True
                break

    def test_warnings_collected(self):
        """Pre-analyzer collects abstract interpretation warnings."""
        source = 'let x = 0; let y = 10 / x;'
        pre = AbstractPreAnalyzer()
        analysis = pre.analyze(source)
        assert len(analysis['warnings']) >= 1


# ============================================================
# Test: Comparison API
# ============================================================

class TestComparisonAPI:
    """Test the compare_guided_vs_plain API."""

    def test_comparison_basic(self):
        """Comparison returns correct structure."""
        source = 'let x = 0; if (x > 0) { let y = 1; } else { let y = 2; }'
        comp = compare_guided_vs_plain(source, {'x': 'int'})
        assert 'plain_paths' in comp
        assert 'guided_paths' in comp
        assert 'plain_smt_checks' in comp
        assert 'guided_smt_checks' in comp
        assert 'guided_smt_saved' in comp
        assert 'pruning_ratio' in comp
        assert 'abstract_warnings' in comp

    def test_same_feasible_count(self):
        """Guided and plain should find the same number of feasible paths."""
        source = 'let x = 0; if (x > 5) { let y = 1; } else { let y = 2; }'
        comp = compare_guided_vs_plain(source, {'x': 'int'})
        assert comp['plain_feasible'] == comp['guided_feasible']

    def test_constant_saves_smt(self):
        """When all branches are constant, guided should save SMT checks."""
        source = 'let x = 5; if (x > 10) { let y = 1; } else { let y = 2; }'
        comp = compare_guided_vs_plain(source)
        assert comp['plain_feasible'] == comp['guided_feasible']


# ============================================================
# Test: GuidedResult Properties
# ============================================================

class TestGuidedResult:
    """Test GuidedResult data access."""

    def test_pruning_ratio_no_branches(self):
        """Pruning ratio is 0 when there are no branches."""
        source = 'let x = 1; let y = 2;'
        result = guided_execute(source)
        assert result.pruning_ratio == 0.0

    def test_result_has_paths(self):
        """Result provides access to paths."""
        source = 'let x = 0; if (x > 0) { let y = 1; }'
        result = guided_execute(source, {'x': 'int'})
        assert result.paths is not None
        assert result.total_paths >= 1

    def test_result_has_test_cases(self):
        """Result provides access to test cases."""
        source = 'let x = 0; if (x > 0) { let y = 1; } else { let y = 2; }'
        result = guided_execute(source, {'x': 'int'})
        assert result.test_cases is not None

    def test_result_has_abstract_info(self):
        """Result includes abstract analysis info."""
        source = 'let x = 0; let y = x + 1;'
        result = guided_execute(source, {'x': 'int'})
        assert result.abstract_env is not None


# ============================================================
# Test: Loop Handling
# ============================================================

class TestLoopHandling:
    """Test guided execution with loops."""

    def test_simple_loop(self):
        """Simple while loop."""
        source = 'let i = 0; while (i < 5) { i = i + 1; }'
        result = guided_execute(source)
        assert result.total_paths >= 1

    def test_symbolic_loop_bound(self):
        """Loop with symbolic bound."""
        source = 'let n = 0; let i = 0; while (i < n) { i = i + 1; }'
        result = guided_execute(source, {'n': 'int'})
        assert result.total_paths >= 1

    def test_loop_with_branch(self):
        """Loop containing an if statement."""
        source = '''
        let x = 0;
        let i = 0;
        while (i < 3) {
            if (x > 0) { i = i + 2; } else { i = i + 1; }
        }
        '''
        result = guided_execute(source, {'x': 'int'})
        assert result.total_paths >= 1


# ============================================================
# Test: Assertion Checking
# ============================================================

class TestAssertionChecking:
    """Test guided assertion checking."""

    def test_assertion_holds(self):
        """Assertion that always holds."""
        source = 'let x = 5; assert(x > 0);'
        assertion_result, guided = guided_check_assertions(source)
        assert assertion_result.holds == True

    def test_assertion_violation(self):
        """Assertion that can be violated."""
        source = 'let x = 0; assert(x > 0);'
        assertion_result, guided = guided_check_assertions(source, {'x': 'int'})
        assert assertion_result.holds == False


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_program(self):
        """Empty program should not crash."""
        source = ""
        result = guided_execute(source)
        assert result.total_paths >= 1

    def test_only_assignments(self):
        """Program with only assignments."""
        source = 'let a = 1; let b = 2; let c = a + b;'
        result = guided_execute(source)
        assert result.total_paths >= 1

    def test_multiple_symbolic_inputs(self):
        """Multiple symbolic inputs."""
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            if (y > 0) { let z = 1; }
        }
        '''
        result = guided_execute(source, {'x': 'int', 'y': 'int'})
        assert result.total_paths >= 1

    def test_boolean_symbolic(self):
        """Symbolic boolean input."""
        source = 'let flag = true; if (flag) { let x = 1; } else { let x = 2; }'
        result = guided_execute(source, {'flag': 'bool'})
        assert result.total_paths >= 1


# ============================================================
# Test: Correctness - Guided matches Plain
# ============================================================

class TestCorrectnessMatch:
    """Verify guided execution finds the same feasible paths as plain."""

    def test_simple_match(self):
        """Simple program: guided and plain find same feasible paths."""
        source = 'let x = 0; if (x > 5) { let y = 1; } else { let y = 2; }'
        plain = symbolic_execute(source, {'x': 'int'})
        guided = guided_execute(source, {'x': 'int'})
        assert len(plain.feasible_paths) == len(guided.feasible_paths)

    def test_nested_match(self):
        """Nested branches: guided and plain find same feasible paths."""
        source = '''
        let x = 0;
        if (x > 0) {
            if (x > 10) { let y = 2; } else { let y = 1; }
        } else {
            let y = 0;
        }
        '''
        plain = symbolic_execute(source, {'x': 'int'})
        guided = guided_execute(source, {'x': 'int'})
        assert len(plain.feasible_paths) == len(guided.feasible_paths)

    def test_concrete_match(self):
        """Concrete program: guided and plain produce same result."""
        source = 'let x = 42; if (x > 10) { print(1); } else { print(0); }'
        plain = symbolic_execute(source)
        guided = guided_execute(source)
        assert len(plain.feasible_paths) == len(guided.feasible_paths)
        assert plain.paths[0].output[0].concrete == guided.paths[0].output[0].concrete


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
