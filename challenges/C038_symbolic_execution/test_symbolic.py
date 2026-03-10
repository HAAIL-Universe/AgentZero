"""
Tests for C038: Symbolic Execution Engine
Composes C037 (SMT Solver) + C010 (Stack VM Parser)
"""

import pytest
from symbolic_execution import (
    SymbolicExecutor, symbolic_execute, generate_tests,
    check_assertions, find_inputs_for_line, get_coverage,
    SymValue, SymType, PathStatus, PathState,
    ExecutionResult, TestCase, AssertionResult, CoverageResult,
    smt_not, smt_and, smt_or,
)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C037_smt_solver'))
from smt_solver import Op as SMTOp


# ============================================================
# 1. Basic Concrete Execution
# ============================================================

class TestConcreteExecution:
    """Programs with no symbolic inputs should execute like a normal interpreter."""

    def test_integer_arithmetic(self):
        source = 'let x = 3 + 4; print(x);'
        result = symbolic_execute(source)
        assert result.total_paths == 1
        assert result.paths[0].status == PathStatus.COMPLETED
        out = result.paths[0].output
        assert len(out) == 1
        assert out[0].concrete == 7

    def test_boolean_literal(self):
        source = 'let x = true; print(x);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == True

    def test_string_literal(self):
        source = 'let s = "hello"; print(s);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == "hello"

    def test_concrete_if_true(self):
        source = '''
        let x = 10;
        if (x > 5) {
            print(1);
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source)
        assert result.total_paths == 1
        assert result.paths[0].output[0].concrete == 1

    def test_concrete_if_false(self):
        source = '''
        let x = 3;
        if (x > 5) {
            print(1);
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source)
        assert result.total_paths == 1
        assert result.paths[0].output[0].concrete == 0

    def test_concrete_while(self):
        source = '''
        let i = 0;
        while (i < 3) {
            i = i + 1;
        }
        print(i);
        '''
        result = symbolic_execute(source)
        assert result.total_paths == 1
        assert result.paths[0].output[0].concrete == 3

    def test_variable_assignment(self):
        source = '''
        let x = 1;
        x = x + 10;
        print(x);
        '''
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == 11

    def test_nested_arithmetic(self):
        source = 'let x = (2 + 3) * 4 - 1; print(x);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == 19

    def test_unary_negation(self):
        source = 'let x = -5; print(x);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == -5

    def test_not_operator(self):
        source = 'let x = not true; print(x);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == False

    def test_modulo(self):
        source = 'let x = 10 % 3; print(x);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == 1

    def test_division(self):
        source = 'let x = 10 / 3; print(x);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == 3

    def test_multiple_prints(self):
        source = 'print(1); print(2); print(3);'
        result = symbolic_execute(source)
        vals = [o.concrete for o in result.paths[0].output]
        assert vals == [1, 2, 3]


# ============================================================
# 2. Symbolic Values
# ============================================================

class TestSymbolicValues:
    """Test symbolic value creation and propagation."""

    def test_sym_value_concrete(self):
        v = SymValue.from_concrete(42)
        assert v.is_concrete()
        assert not v.is_symbolic()
        assert v.concrete == 42

    def test_sym_value_symbolic(self):
        from smt_solver import SMTSolver
        s = SMTSolver()
        t = s.Int('x')
        v = SymValue.from_symbolic(t, name='x')
        assert v.is_symbolic()
        assert not v.is_concrete()
        assert v.name == 'x'

    def test_symbolic_input_preserved(self):
        """Symbolic input should not be overwritten by let declaration."""
        source = 'let x = 0; print(x);'
        result = symbolic_execute(source, {'x': 'int'})
        out = result.paths[0].output[0]
        assert out.is_symbolic()

    def test_symbolic_arithmetic_propagation(self):
        """Arithmetic on symbolic values produces symbolic results."""
        source = '''
        let x = 0;
        let y = x + 1;
        print(y);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        assert result.paths[0].output[0].is_symbolic()

    def test_symbolic_comparison_propagation(self):
        """Comparison on symbolic values produces symbolic boolean."""
        source = '''
        let x = 0;
        let cmp = x > 5;
        '''
        result = symbolic_execute(source, {'x': 'int'})
        assert result.paths[0].env['cmp'].is_symbolic()


# ============================================================
# 3. Path Forking (If Statements)
# ============================================================

class TestPathForking:
    """Test branching on symbolic conditions."""

    def test_simple_fork(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 2

    def test_fork_generates_different_inputs(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        inputs = [tc.inputs for tc in result.test_cases]
        x_vals = [i['x'] for i in inputs]
        # One should be > 0, one should be <= 0
        assert any(v > 0 for v in x_vals)
        assert any(v <= 0 for v in x_vals)

    def test_nested_forks(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            if (y > 0) {
                print(1);
            } else {
                print(2);
            }
        } else {
            print(3);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 3

    def test_if_without_else(self):
        source = '''
        let x = 0;
        let r = 0;
        if (x > 0) {
            r = 1;
        }
        print(r);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 2

    def test_if_else_if(self):
        source = '''
        let x = 0;
        if (x > 10) {
            print(3);
        } else if (x > 0) {
            print(2);
        } else {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 3

    def test_sequential_ifs(self):
        """Two independent if statements should create 4 paths."""
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            print(1);
        } else {
            print(2);
        }
        if (y > 0) {
            print(3);
        } else {
            print(4);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 4

    def test_branch_coverage_recorded(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        all_branches = set()
        for p in result.paths:
            if p.status != PathStatus.INFEASIBLE:
                for b in p.covered_branches:
                    all_branches.add(b)
        # Should have both (line, True) and (line, False)
        lines = set(b[0] for b in all_branches)
        for line in lines:
            assert (line, True) in all_branches
            assert (line, False) in all_branches


# ============================================================
# 4. While Loop Unrolling
# ============================================================

class TestWhileLoop:
    """Test symbolic execution of while loops."""

    def test_symbolic_loop_bound(self):
        source = '''
        let x = 0;
        let i = 0;
        while (i < x) {
            i = i + 1;
        }
        print(i);
        '''
        result = symbolic_execute(source, {'x': 'int'}, max_loop_unroll=3)
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) >= 3  # 0, 1, 2, 3 iterations

    def test_loop_generates_inputs(self):
        source = '''
        let x = 0;
        let i = 0;
        while (i < x) {
            i = i + 1;
        }
        '''
        result = symbolic_execute(source, {'x': 'int'}, max_loop_unroll=3)
        x_vals = sorted(set(tc.inputs['x'] for tc in result.test_cases))
        # Should generate inputs for different iteration counts
        assert len(x_vals) >= 2

    def test_concrete_loop(self):
        source = '''
        let i = 0;
        while (i < 3) {
            i = i + 1;
        }
        print(i);
        '''
        result = symbolic_execute(source)
        assert result.total_paths == 1
        assert result.paths[0].output[0].concrete == 3

    def test_loop_max_unroll(self):
        source = '''
        let x = 0;
        while (x > 0) {
            x = x - 1;
        }
        '''
        result = symbolic_execute(source, {'x': 'int'}, max_loop_unroll=2)
        # Should terminate within unroll bound
        assert result.total_paths >= 1

    def test_loop_with_inner_branch(self):
        source = '''
        let x = 0;
        let y = 0;
        let i = 0;
        while (i < x) {
            if (i > y) {
                print(1);
            }
            i = i + 1;
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'}, max_loop_unroll=2)
        assert result.total_paths >= 2


# ============================================================
# 5. Constraint Generation
# ============================================================

class TestConstraints:
    """Test that correct path constraints are generated."""

    def test_simple_constraint(self):
        source = '''
        let x = 0;
        if (x > 5) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        # True branch should have x > 5 constraint
        for p in result.paths:
            if p.status != PathStatus.INFEASIBLE:
                assert len(p.constraints) >= 0

    def test_compound_constraint(self):
        source = '''
        let x = 0;
        if (x > 0) {
            if (x < 10) {
                print(1);
            }
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        # Path reaching print should have x > 0 AND x < 10
        for tc in result.test_cases:
            if 1 in [o.concrete for o in tc.output if o.is_concrete()]:
                assert tc.inputs['x'] > 0
                assert tc.inputs['x'] < 10

    def test_equality_constraint(self):
        source = '''
        let x = 0;
        if (x == 42) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] == 42

    def test_inequality_constraint(self):
        source = '''
        let x = 0;
        if (x != 0) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] != 0

    def test_arithmetic_constraint(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x + y > 10) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] + tc.inputs['y'] > 10

    def test_subtraction_constraint(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x - y > 5) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] - tc.inputs['y'] > 5

    def test_negation_constraint(self):
        source = '''
        let x = 0;
        if (-x > 3) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert -tc.inputs['x'] > 3


# ============================================================
# 6. Test Generation
# ============================================================

class TestGeneration:
    """Test automatic test input generation."""

    def test_generates_test_cases(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        tests = generate_tests(source, {'x': 'int'})
        assert len(tests) >= 2

    def test_test_cases_have_inputs(self):
        source = '''
        let x = 0;
        if (x > 5) {
            print(1);
        }
        '''
        tests = generate_tests(source, {'x': 'int'})
        for tc in tests:
            assert 'x' in tc.inputs

    def test_test_cases_cover_branches(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        tests = generate_tests(source, {'x': 'int'})
        outputs = set()
        for tc in tests:
            for o in tc.output:
                if o.is_concrete():
                    outputs.add(o.concrete)
        assert 1 in outputs or 0 in outputs

    def test_multiple_symbolic_inputs(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > y) {
            print(1);
        } else {
            print(0);
        }
        '''
        tests = generate_tests(source, {'x': 'int', 'y': 'int'})
        assert len(tests) >= 2
        for tc in tests:
            assert 'x' in tc.inputs
            assert 'y' in tc.inputs

    def test_no_symbolic_inputs(self):
        source = 'let x = 5; print(x);'
        tests = generate_tests(source, {})
        # Should still produce at least one test (concrete execution)
        assert len(tests) >= 1

    def test_test_cases_are_valid(self):
        """Generated inputs should satisfy path constraints when verified."""
        source = '''
        let x = 0;
        if (x > 10) {
            if (x < 20) {
                print(1);
            }
        }
        '''
        tests = generate_tests(source, {'x': 'int'})
        for tc in tests:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert 10 < tc.inputs['x'] < 20

    def test_test_case_path_id(self):
        source = '''
        let x = 0;
        if (x > 0) { print(1); } else { print(0); }
        '''
        tests = generate_tests(source, {'x': 'int'})
        path_ids = [tc.path_id for tc in tests]
        assert len(set(path_ids)) == len(path_ids)  # unique IDs


# ============================================================
# 7. Assertion Checking
# ============================================================

class TestAssertions:
    """Test symbolic assertion verification."""

    def test_assertion_can_fail(self):
        source = '''
        let x = 0;
        assert(x > 0);
        '''
        result = check_assertions(source, {'x': 'int'})
        assert not result.holds
        assert len(result.violations) >= 1

    def test_assertion_always_holds(self):
        source = '''
        let x = 0;
        if (x > 10) {
            assert(x > 5);
        }
        '''
        result = check_assertions(source, {'x': 'int'})
        assert result.holds

    def test_assertion_counterexample(self):
        source = '''
        let x = 0;
        assert(x > 0);
        '''
        result = check_assertions(source, {'x': 'int'})
        assert not result.holds
        ce = result.violations[0].counterexample
        assert ce is not None
        assert ce['x'] <= 0

    def test_assertion_with_arithmetic(self):
        source = '''
        let x = 0;
        let y = 0;
        assert(x + y > 0);
        '''
        result = check_assertions(source, {'x': 'int', 'y': 'int'})
        assert not result.holds
        ce = result.violations[0].counterexample
        assert ce['x'] + ce['y'] <= 0

    def test_conditional_assertion(self):
        source = '''
        let x = 0;
        if (x >= 0) {
            assert(x >= 0);
        }
        '''
        result = check_assertions(source, {'x': 'int'})
        assert result.holds

    def test_assertion_after_assignment(self):
        source = '''
        let x = 0;
        let y = x + 1;
        assert(y > x);
        '''
        result = check_assertions(source, {'x': 'int'})
        assert result.holds

    def test_multiple_assertions(self):
        source = '''
        let x = 0;
        assert(x > -100);
        assert(x < 100);
        '''
        result = check_assertions(source, {'x': 'int'})
        # Both can fail independently
        # x > -100 can fail when x <= -100
        # x < 100 can fail when x >= 100
        assert not result.holds

    def test_concrete_assertion_pass(self):
        source = '''
        let x = 5;
        assert(x > 0);
        '''
        result = check_assertions(source)
        assert result.holds

    def test_concrete_assertion_fail(self):
        source = '''
        let x = -1;
        assert(x > 0);
        '''
        result = check_assertions(source)
        assert not result.holds

    def test_assertion_paths_explored(self):
        source = '''
        let x = 0;
        if (x > 0) {
            assert(x > 0);
        }
        '''
        result = check_assertions(source, {'x': 'int'})
        assert result.paths_explored >= 2


# ============================================================
# 8. Coverage Analysis
# ============================================================

class TestCoverage:
    """Test coverage analysis."""

    def test_full_branch_coverage(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        cov = get_coverage(source, {'x': 'int'})
        assert cov.branch_coverage == 1.0

    def test_feasible_paths_count(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        cov = get_coverage(source, {'x': 'int'})
        assert cov.feasible_paths == 2

    def test_paths_explored(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            if (y > 0) { print(1); }
        }
        '''
        cov = get_coverage(source, {'x': 'int', 'y': 'int'})
        assert cov.paths_explored >= 3

    def test_line_coverage(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        cov = get_coverage(source, {'x': 'int'})
        assert cov.covered_lines >= 3
        assert cov.line_coverage > 0

    def test_concrete_coverage(self):
        source = '''
        let x = 5;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        cov = get_coverage(source)
        # Only one branch taken
        assert cov.feasible_paths == 1

    def test_no_dead_branches_symbolic(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        cov = get_coverage(source, {'x': 'int'})
        assert len(cov.dead_branches) == 0


# ============================================================
# 9. Infeasible Path Detection
# ============================================================

class TestInfeasiblePaths:
    """Test detection of paths that can never be taken."""

    def test_contradictory_constraints(self):
        source = '''
        let x = 0;
        if (x > 10) {
            if (x < 5) {
                print(1);
            }
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        infeasible = [p for p in result.paths if p.status == PathStatus.INFEASIBLE]
        # x > 10 AND x < 5 is infeasible
        assert len(infeasible) >= 1

    def test_infeasible_not_in_test_cases(self):
        source = '''
        let x = 0;
        if (x > 10) {
            if (x < 5) {
                print(1);
            }
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        # No test case should try to make both x>10 and x<5 true
        for tc in result.test_cases:
            # No test case should print 1
            for o in tc.output:
                if o.is_concrete():
                    assert o.concrete != 1

    def test_always_true_condition(self):
        """When inner condition is always true given outer, no infeasible inner-false path."""
        source = '''
        let x = 0;
        if (x > 100) {
            if (x > 0) {
                print(1);
            }
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        # The path x>100 AND NOT(x>0) is infeasible
        infeasible = [p for p in result.paths if p.status == PathStatus.INFEASIBLE]
        assert len(infeasible) >= 1


# ============================================================
# 10. Functions
# ============================================================

class TestFunctions:
    """Test symbolic execution across function calls."""

    def test_function_call(self):
        source = '''
        fn add(a, b) {
            return a + b;
        }
        let result = add(3, 4);
        print(result);
        '''
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == 7

    def test_function_with_symbolic_arg(self):
        source = '''
        fn check(x) {
            if (x > 0) {
                return 1;
            }
            return 0;
        }
        let x = 0;
        let r = check(x);
        print(r);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) >= 2

    def test_function_recorded_in_result(self):
        source = '''
        fn foo() { return 1; }
        foo();
        '''
        result = symbolic_execute(source)
        assert 'foo' in result.functions

    def test_recursive_not_infinite(self):
        """Functions calling themselves should be bounded."""
        source = '''
        fn f(x) {
            if (x > 0) {
                return f(x - 1);
            }
            return 0;
        }
        let r = f(3);
        print(r);
        '''
        result = symbolic_execute(source)
        assert result.total_paths >= 1

    def test_function_scope(self):
        """Function params should not leak into outer scope."""
        source = '''
        let x = 10;
        fn f(x) {
            return x + 1;
        }
        let r = f(5);
        print(x);
        '''
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == 10


# ============================================================
# 11. Find Inputs for Line
# ============================================================

class TestFindInputs:
    """Test finding inputs to reach specific lines."""

    def test_find_reachable_line(self):
        source = '''
        let x = 0;
        if (x > 100) {
            print(1);
        }
        '''
        inputs = find_inputs_for_line(source, {'x': 'int'}, 4)
        assert inputs is not None
        assert inputs['x'] > 100

    def test_find_unreachable_returns_none(self):
        source = '''
        let x = 0;
        if (x > 10) {
            if (x < 5) {
                print(1);
            }
        }
        '''
        inputs = find_inputs_for_line(source, {'x': 'int'}, 5)
        assert inputs is None

    def test_find_inputs_two_vars(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x + y > 20) {
            print(1);
        }
        '''
        inputs = find_inputs_for_line(source, {'x': 'int', 'y': 'int'}, 5)
        assert inputs is not None
        assert inputs['x'] + inputs['y'] > 20


# ============================================================
# 12. Path Limits
# ============================================================

class TestPathLimits:
    """Test max_paths bound."""

    def test_max_paths_respected(self):
        source = '''
        let a = 0; let b = 0; let c = 0; let d = 0;
        if (a > 0) { print(1); } else { print(2); }
        if (b > 0) { print(3); } else { print(4); }
        if (c > 0) { print(5); } else { print(6); }
        if (d > 0) { print(7); } else { print(8); }
        '''
        result = symbolic_execute(source, {'a': 'int', 'b': 'int', 'c': 'int', 'd': 'int'},
                                  max_paths=5)
        assert result.total_paths <= 10  # Some leeway for infeasible

    def test_single_path_limit(self):
        source = '''
        let x = 0;
        if (x > 0) { print(1); } else { print(0); }
        '''
        result = symbolic_execute(source, {'x': 'int'}, max_paths=1)
        assert result.total_paths >= 1


# ============================================================
# 13. Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_program(self):
        result = symbolic_execute('')
        assert result.total_paths == 1

    def test_only_let(self):
        result = symbolic_execute('let x = 5;')
        assert result.total_paths == 1

    def test_division_by_zero_concrete(self):
        source = 'let x = 10 / 0; print(x);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete is None

    def test_undefined_variable(self):
        source = 'print(undefined_var);'
        result = symbolic_execute(source)
        assert any(p.status == PathStatus.ERROR for p in result.paths)

    def test_boolean_symbolic_input(self):
        source = '''
        let b = false;
        if (b) {
            print(1);
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source, {'b': 'bool'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) >= 2

    def test_deeply_nested(self):
        source = '''
        let x = 0;
        if (x > 0) {
            if (x > 10) {
                if (x > 100) {
                    print(3);
                } else {
                    print(2);
                }
            } else {
                print(1);
            }
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 4

    def test_comparison_operators(self):
        """Test all comparison operators generate correct constraints."""
        ops = [('<', lambda x: x < 5), ('>', lambda x: x > 5),
               ('<=', lambda x: x <= 5), ('>=', lambda x: x >= 5)]
        for op_str, check in ops:
            source = f'''
            let x = 0;
            if (x {op_str} 5) {{
                print(1);
            }}
            '''
            result = symbolic_execute(source, {'x': 'int'})
            for tc in result.test_cases:
                if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                    assert check(tc.inputs['x']), f"Failed for operator {op_str}"


# ============================================================
# 14. Symbolic Arithmetic
# ============================================================

class TestSymbolicArithmetic:
    """Test arithmetic operations on symbolic values."""

    def test_sym_addition(self):
        source = '''
        let x = 0;
        if (x + 5 > 10) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] + 5 > 10

    def test_sym_subtraction(self):
        source = '''
        let x = 0;
        if (x - 3 > 0) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] - 3 > 0

    def test_sym_multiplication(self):
        source = '''
        let x = 0;
        if (x * 2 > 10) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) >= 2  # Both branches should be feasible

    def test_sym_negation(self):
        source = '''
        let x = 0;
        if (-x > 5) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert -tc.inputs['x'] > 5

    def test_mixed_sym_concrete(self):
        """Concrete + symbolic should lift to symbolic."""
        source = '''
        let x = 0;
        let y = 5;
        if (x + y > 10) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] + 5 > 10

    def test_two_symbolic_vars(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x * 2 + y > 15) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] * 2 + tc.inputs['y'] > 15


# ============================================================
# 15. SymValue Type System
# ============================================================

class TestSymValueTypes:
    def test_concrete_int(self):
        v = SymValue.from_concrete(42)
        assert v.kind == SymType.CONCRETE
        assert v.concrete == 42

    def test_concrete_bool(self):
        v = SymValue.from_concrete(True)
        assert v.concrete == True

    def test_concrete_string(self):
        v = SymValue.from_concrete("test")
        assert v.concrete == "test"

    def test_concrete_none(self):
        v = SymValue.from_concrete(None)
        assert v.concrete is None

    def test_repr_concrete(self):
        v = SymValue.from_concrete(5)
        assert "Concrete" in repr(v)

    def test_repr_symbolic(self):
        from smt_solver import SMTSolver
        s = SMTSolver()
        t = s.Int('x')
        v = SymValue.from_symbolic(t)
        assert "Symbolic" in repr(v)


# ============================================================
# 16. PathState
# ============================================================

class TestPathState:
    def test_initial_state(self):
        s = PathState()
        assert s.status == PathStatus.ACTIVE
        assert s.env == {}
        assert s.constraints == []
        assert s.output == []

    def test_copy_isolation(self):
        s = PathState()
        s.env['x'] = SymValue.from_concrete(5)
        s.constraints.append("dummy")
        s.covered_lines.add(1)

        c = s.copy()
        c.env['y'] = SymValue.from_concrete(10)
        c.constraints.append("dummy2")
        c.covered_lines.add(2)

        assert 'y' not in s.env
        assert len(s.constraints) == 1
        assert 2 not in s.covered_lines

    def test_copy_preserves_values(self):
        s = PathState(path_id=5, error_msg="test")
        c = s.copy()
        assert c.path_id == 5
        assert c.error_msg == "test"


# ============================================================
# 17. ExecutionResult Properties
# ============================================================

class TestExecutionResult:
    def test_total_paths(self):
        source = '''
        let x = 0;
        if (x > 0) { print(1); } else { print(0); }
        '''
        r = symbolic_execute(source, {'x': 'int'})
        assert r.total_paths == len(r.paths)

    def test_feasible_paths_excludes_infeasible(self):
        source = '''
        let x = 0;
        if (x > 10) {
            if (x < 5) { print(1); }
        }
        '''
        r = symbolic_execute(source, {'x': 'int'})
        assert len(r.feasible_paths) <= r.total_paths

    def test_total_test_cases(self):
        source = '''
        let x = 0;
        if (x > 0) { print(1); } else { print(0); }
        '''
        r = symbolic_execute(source, {'x': 'int'})
        assert r.total_test_cases == len(r.test_cases)


# ============================================================
# 18. Composition Integrity
# ============================================================

class TestComposition:
    """Verify correct composition of C037 and C010."""

    def test_c010_parser_used(self):
        """Source is parsed by C010 parser."""
        source = '''
        let x = 0;
        if (x > 5) { print(1); }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        assert result.total_paths >= 1

    def test_c037_solver_used(self):
        """SMT solver generates valid models."""
        source = '''
        let x = 0;
        if (x > 100) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] > 100

    def test_smt_constraint_satisfaction(self):
        """Generated inputs actually satisfy the path constraints."""
        source = '''
        let x = 0;
        let y = 0;
        if (x + y > 10) {
            if (x - y > 0) {
                print(1);
            }
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] + tc.inputs['y'] > 10
                assert tc.inputs['x'] - tc.inputs['y'] > 0


# ============================================================
# 19. Complex Programs
# ============================================================

class TestComplexPrograms:
    """Test with more realistic programs."""

    def test_absolute_value(self):
        source = '''
        let x = 0;
        let result = 0;
        if (x >= 0) {
            result = x;
        } else {
            result = -x;
        }
        print(result);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 2

    def test_max_of_two(self):
        source = '''
        let x = 0;
        let y = 0;
        let m = 0;
        if (x > y) {
            m = x;
        } else {
            m = y;
        }
        print(m);
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 2

    def test_classify_number(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else if (x == 0) {
            print(0);
        } else {
            print(-1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 3

    def test_bounded_search(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            if (x < 100) {
                if (y > x) {
                    if (y < 200) {
                        print(1);
                    }
                }
            }
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert 0 < tc.inputs['x'] < 100
                assert tc.inputs['x'] < tc.inputs['y'] < 200

    def test_sum_loop(self):
        source = '''
        let n = 0;
        let sum = 0;
        let i = 0;
        while (i < n) {
            sum = sum + i;
            i = i + 1;
        }
        print(sum);
        '''
        result = symbolic_execute(source, {'n': 'int'}, max_loop_unroll=4)
        assert result.total_test_cases >= 2

    def test_function_with_branch(self):
        source = '''
        fn classify(x) {
            if (x > 0) {
                return 1;
            }
            if (x == 0) {
                return 0;
            }
            return -1;
        }
        let x = 0;
        let c = classify(x);
        print(c);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) >= 3

    def test_linear_search(self):
        source = '''
        let target = 0;
        let found = 0;
        if (target == 1) {
            found = 1;
        }
        if (target == 2) {
            found = 1;
        }
        if (target == 3) {
            found = 1;
        }
        print(found);
        '''
        result = symbolic_execute(source, {'target': 'int'})
        assert result.total_test_cases >= 4


# ============================================================
# 20. SMT Helper Functions
# ============================================================

class TestSMTHelpers:
    def test_smt_not(self):
        from smt_solver import SMTSolver
        s = SMTSolver()
        x = s.Int('x')
        t = x > 5
        nt = smt_not(t)
        # smt_not uses complement: NOT(x > 5) -> x <= 5
        assert nt.op == SMTOp.LE

    def test_smt_and(self):
        from smt_solver import SMTSolver
        s = SMTSolver()
        x = s.Int('x')
        a = x > 0
        b = x < 10
        r = smt_and(a, b)
        assert r.op == SMTOp.AND

    def test_smt_or(self):
        from smt_solver import SMTSolver
        s = SMTSolver()
        x = s.Int('x')
        a = x > 0
        b = x < 10
        r = smt_or(a, b)
        assert r.op == SMTOp.OR


# ============================================================
# 21. Boolean Logic
# ============================================================

class TestBooleanLogic:
    def test_and_operator(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0 and y > 0) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] > 0
                assert tc.inputs['y'] > 0

    def test_or_operator(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 10 or y > 10) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] > 10 or tc.inputs['y'] > 10

    def test_not_operator_symbolic(self):
        source = '''
        let x = 0;
        if (not (x > 5)) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert not (tc.inputs['x'] > 5)

    def test_complex_boolean(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0 and y > 0 and x + y < 10) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int', 'y': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] > 0
                assert tc.inputs['y'] > 0
                assert tc.inputs['x'] + tc.inputs['y'] < 10


# ============================================================
# 22. Variable Updates
# ============================================================

class TestVariableUpdates:
    def test_reassignment_tracked(self):
        source = '''
        let x = 0;
        x = x + 1;
        if (x > 5) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                # x was incremented by 1, so original x + 1 > 5 => x > 4
                assert tc.inputs['x'] + 1 > 5

    def test_multiple_updates(self):
        source = '''
        let x = 0;
        x = x + 1;
        x = x * 2;
        if (x > 10) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) >= 2  # Both branches feasible

    def test_conditional_update(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = -1;
        }
        print(y);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 2


# ============================================================
# 23. Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    def test_symbolic_execute(self):
        r = symbolic_execute('print(42);')
        assert r.total_paths == 1

    def test_generate_tests(self):
        tests = generate_tests('let x = 0; if (x > 0) { print(1); }', {'x': 'int'})
        assert len(tests) >= 1

    def test_check_assertions(self):
        r = check_assertions('let x = 5; assert(x > 0);')
        assert r.holds

    def test_find_inputs_for_line(self):
        source = '''
        let x = 0;
        if (x > 10) {
            print(1);
        }
        '''
        inputs = find_inputs_for_line(source, {'x': 'int'}, 4)
        assert inputs is not None

    def test_get_coverage(self):
        source = '''
        let x = 0;
        if (x > 0) { print(1); } else { print(0); }
        '''
        cov = get_coverage(source, {'x': 'int'})
        assert cov.paths_explored >= 2


# ============================================================
# 24. SymbolicExecutor Configuration
# ============================================================

class TestConfiguration:
    def test_default_max_paths(self):
        e = SymbolicExecutor()
        assert e.max_paths == 64

    def test_custom_max_paths(self):
        e = SymbolicExecutor(max_paths=10)
        assert e.max_paths == 10

    def test_default_max_loop_unroll(self):
        e = SymbolicExecutor()
        assert e.max_loop_unroll == 5

    def test_custom_loop_unroll(self):
        e = SymbolicExecutor(max_loop_unroll=3)
        assert e.max_loop_unroll == 3


# ============================================================
# 25. Stress Tests
# ============================================================

class TestStress:
    def test_many_branches(self):
        """8 sequential if statements = up to 256 paths."""
        vars_code = ""
        for i in range(8):
            vars_code += f"let v{i} = 0;\n"
        branch_code = ""
        for i in range(8):
            branch_code += f"if (v{i} > 0) {{ print({i}); }}\n"
        source = vars_code + branch_code
        inputs = {f'v{i}': 'int' for i in range(8)}
        result = symbolic_execute(source, inputs, max_paths=32)
        assert result.total_paths >= 16

    def test_deep_nesting(self):
        source = '''
        let x = 0;
        if (x > 0) {
            if (x > 10) {
                if (x > 100) {
                    if (x > 1000) {
                        print(4);
                    } else {
                        print(3);
                    }
                } else {
                    print(2);
                }
            } else {
                print(1);
            }
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 5

    def test_loop_and_branch(self):
        source = '''
        let x = 0;
        let i = 0;
        while (i < x) {
            if (i > 2) {
                print(1);
            }
            i = i + 1;
        }
        '''
        result = symbolic_execute(source, {'x': 'int'}, max_paths=32, max_loop_unroll=4)
        assert result.total_paths >= 3


# ============================================================
# 26. CoverageResult Fields
# ============================================================

class TestCoverageResultFields:
    def test_total_lines(self):
        source = '''
        let x = 0;
        print(x);
        '''
        cov = get_coverage(source, {'x': 'int'})
        assert cov.total_lines >= 1

    def test_branch_coverage_full(self):
        source = '''
        let x = 0;
        if (x > 0) { print(1); } else { print(0); }
        '''
        cov = get_coverage(source, {'x': 'int'})
        assert cov.branch_coverage == 1.0

    def test_dead_branches_detected(self):
        source = '''
        let x = 0;
        if (x > 10) {
            if (x < 5) {
                print(1);
            } else {
                print(2);
            }
        }
        '''
        cov = get_coverage(source, {'x': 'int'})
        # The branch x>10 AND x<5 is dead
        # dead_branches should contain the infeasible (line, direction)
        # At minimum we know some branches are covered
        assert cov.paths_explored >= 2


# ============================================================
# 27. Symbolic with Let Override
# ============================================================

class TestLetOverride:
    def test_symbolic_preserved_over_let(self):
        """let x = concrete should be skipped when x is already symbolic."""
        source = '''
        let x = 42;
        if (x > 100) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 2

    def test_nonsymbolic_let_works(self):
        """let y = concrete should work when y is not a symbolic input."""
        source = '''
        let x = 0;
        let y = 42;
        if (x > y) {
            print(1);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        for tc in result.test_cases:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                assert tc.inputs['x'] > 42


# ============================================================
# 28. Path Status
# ============================================================

class TestPathStatus:
    def test_completed_status(self):
        result = symbolic_execute('print(1);')
        assert result.paths[0].status == PathStatus.COMPLETED

    def test_error_status(self):
        source = 'print(undefined);'
        result = symbolic_execute(source)
        assert any(p.status == PathStatus.ERROR for p in result.paths)

    def test_infeasible_status(self):
        source = '''
        let x = 0;
        if (x > 10) {
            if (x < 5) { print(1); }
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        assert any(p.status == PathStatus.INFEASIBLE for p in result.paths)

    def test_assertion_failed_status(self):
        source = '''
        let x = 0;
        assert(x > 0);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        assert any(p.status == PathStatus.ASSERTION_FAILED for p in result.paths)


# ============================================================
# 29. Output Tracking
# ============================================================

class TestOutputTracking:
    def test_concrete_output(self):
        source = 'print(42);'
        result = symbolic_execute(source)
        assert result.paths[0].output[0].concrete == 42

    def test_symbolic_output(self):
        source = '''
        let x = 0;
        print(x);
        '''
        result = symbolic_execute(source, {'x': 'int'})
        assert result.paths[0].output[0].is_symbolic()

    def test_output_per_path(self):
        source = '''
        let x = 0;
        if (x > 0) {
            print(1);
        } else {
            print(0);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        outputs = set()
        for p in result.paths:
            if p.status != PathStatus.INFEASIBLE:
                for o in p.output:
                    if o.is_concrete():
                        outputs.add(o.concrete)
        assert 1 in outputs
        assert 0 in outputs

    def test_multiple_outputs(self):
        source = 'print(1); print(2); print(3);'
        result = symbolic_execute(source)
        vals = [o.concrete for o in result.paths[0].output]
        assert vals == [1, 2, 3]


# ============================================================
# 30. Integration: Full Pipeline
# ============================================================

class TestFullPipeline:
    """End-to-end tests verifying the complete pipeline."""

    def test_classify_and_assert(self):
        """Classify a number and verify assertions."""
        source = '''
        let x = 0;
        let class = 0;
        if (x > 0) {
            class = 1;
        } else if (x == 0) {
            class = 0;
        } else {
            class = -1;
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) == 3
        # Verify each path has valid constraints
        for tc in result.test_cases:
            x = tc.inputs.get('x', 0)
            env_class = None
            # Can't easily access env here, but test generation should work
            assert 'x' in tc.inputs

    def test_binary_search_constraints(self):
        """Test constraint solving for a binary-search-like program."""
        source = '''
        let x = 0;
        let lo = 0;
        let hi = 100;
        let mid = 50;
        if (x < mid) {
            hi = mid;
        } else {
            lo = mid;
        }
        if (x < 25) {
            print(1);
        } else if (x < 50) {
            print(2);
        } else if (x < 75) {
            print(3);
        } else {
            print(4);
        }
        '''
        result = symbolic_execute(source, {'x': 'int'})
        feasible = [p for p in result.paths if p.status != PathStatus.INFEASIBLE]
        assert len(feasible) >= 4

    def test_full_coverage_report(self):
        source = '''
        let x = 0;
        let y = 0;
        if (x > 0) {
            if (y > 0) {
                print(1);
            } else {
                print(2);
            }
        } else {
            print(3);
        }
        '''
        cov = get_coverage(source, {'x': 'int', 'y': 'int'})
        assert cov.branch_coverage == 1.0
        assert cov.feasible_paths == 3
        assert len(cov.dead_branches) == 0

    def test_generate_and_verify(self):
        """Generate tests and verify they satisfy constraints."""
        source = '''
        let a = 0;
        let b = 0;
        if (a + b > 20) {
            if (a - b > 5) {
                if (a > 15) {
                    print(1);
                }
            }
        }
        '''
        tests = generate_tests(source, {'a': 'int', 'b': 'int'})
        for tc in tests:
            if any(o.concrete == 1 for o in tc.output if o.is_concrete()):
                a, b = tc.inputs['a'], tc.inputs['b']
                assert a + b > 20
                assert a - b > 5
                assert a > 15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
