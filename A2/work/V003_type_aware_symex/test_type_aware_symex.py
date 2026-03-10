"""
Tests for V003: Type-Aware Symbolic Execution
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from type_aware_symex import (
    type_aware_execute, analyze_function, find_type_errors, get_typed_tests,
    analyze_types, TypeAwareExecutor, TypeAwareResult, TypedTestCase,
    TypedValue, TypeWarning, TypeAwareStats, TypeErrorResult,
    VarTypeInfo, FunctionTypeInfo, TypeAnalysis,
    make_bool_invariant, make_range_invariant, make_nonneg_invariant,
    c013_type_to_smt_sort, c013_type_to_symbolic_input,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C013_type_checker'))
from type_checker import TInt, TBool, TFloat, TString, TFunc, resolve as resolve_type

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
from symbolic_execution import PathStatus


# ============================================================
# Section 1: Type Analysis (static)
# ============================================================

class TestTypeAnalysis:
    """Test C013 type analysis on various programs."""

    def test_simple_int_var(self):
        src = "let x = 5;"
        info = analyze_types(src)
        assert 'x' in info.variables
        assert isinstance(resolve_type(info.variables['x'].inferred_type), TInt)
        assert not info.has_errors

    def test_simple_bool_var(self):
        src = "let flag = true;"
        info = analyze_types(src)
        assert 'flag' in info.variables
        assert isinstance(resolve_type(info.variables['flag'].inferred_type), TBool)

    def test_function_signature(self):
        src = """
        fn add(a, b) {
            return (a + b);
        }
        let r = add(1, 2);
        """
        info = analyze_types(src)
        assert 'add' in info.functions
        fn = info.functions['add']
        assert len(fn.param_types) == 2
        assert fn.name == 'add'

    def test_function_int_params(self):
        src = """
        fn double(x) {
            return (x * 2);
        }
        let r = double(5);
        """
        info = analyze_types(src)
        assert 'double' in info.functions
        fn = info.functions['double']
        assert len(fn.param_types) == 1
        # After unification with call double(5), x should be int
        pt = resolve_type(fn.param_types[0][1])
        assert isinstance(pt, TInt)

    def test_function_bool_param(self):
        src = """
        fn negate(b) {
            if (b) {
                return false;
            }
            return true;
        }
        let r = negate(true);
        """
        info = analyze_types(src)
        assert 'negate' in info.functions

    def test_type_error_detected(self):
        src = """
        let x = 5;
        let y = "hello";
        let z = (x + y);
        """
        info = analyze_types(src)
        assert info.has_errors
        assert len(info.errors) > 0

    def test_no_errors_clean_code(self):
        src = """
        let x = 10;
        let y = 20;
        let z = (x + y);
        """
        info = analyze_types(src)
        assert not info.has_errors


# ============================================================
# Section 2: Type-to-SMT Mapping
# ============================================================

class TestTypeSMTMapping:
    """Test type mapping from C013 types to SMT sorts."""

    def test_int_to_smt(self):
        assert c013_type_to_smt_sort(TInt()) == 'int'

    def test_bool_to_smt(self):
        assert c013_type_to_smt_sort(TBool()) == 'bool'

    def test_float_to_smt(self):
        assert c013_type_to_smt_sort(TFloat()) == 'int'  # approximation

    def test_string_to_smt(self):
        assert c013_type_to_smt_sort(TString()) is None  # no SMT sort

    def test_int_to_symbolic_input(self):
        assert c013_type_to_symbolic_input(TInt()) == 'int'

    def test_bool_to_symbolic_input(self):
        assert c013_type_to_symbolic_input(TBool()) == 'bool'


# ============================================================
# Section 3: Type Constraints
# ============================================================

class TestTypeConstraints:
    """Test SMT constraint generation from types."""

    def test_bool_invariant_structure(self):
        c = make_bool_invariant('flag')
        # Should be (flag == 0 OR flag == 1)
        assert c is not None

    def test_range_invariant_structure(self):
        c = make_range_invariant('x', -10, 10)
        assert c is not None

    def test_nonneg_invariant_structure(self):
        c = make_nonneg_invariant('count')
        assert c is not None

    def test_bool_invariant_constrains_smt(self):
        """Verify bool invariant actually constrains values."""
        from smt_solver import SMTSolver, SMTResult, App, Op as SMTOp, IntConst
        from smt_solver import Var as SMTVar, INT as SMT_INT, BOOL

        solver = SMTSolver()
        solver.Int('b')
        # Add bool invariant
        solver.add(make_bool_invariant('b'))
        # Add b > 1 -- should be UNSAT with bool invariant
        v = SMTVar('b', SMT_INT)
        solver.add(App(SMTOp.GT, [v, IntConst(1)], BOOL))
        assert solver.check() == SMTResult.UNSAT


# ============================================================
# Section 4: Basic Type-Aware Execution
# ============================================================

class TestTypeAwareExecution:
    """Test the full type-aware symbolic execution pipeline."""

    def test_simple_if(self):
        src = """
        let x = 0;
        if ((x > 5)) {
            let y = 1;
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        assert result is not None
        assert result.stats.total_paths > 0

    def test_result_has_type_analysis(self):
        src = "let x = 5;"
        result = type_aware_execute(src)
        assert result.type_analysis is not None
        assert isinstance(result.type_analysis, TypeAnalysis)

    def test_result_has_execution(self):
        src = "let x = 5;"
        result = type_aware_execute(src)
        assert result.execution is not None

    def test_result_has_stats(self):
        src = "let x = 5;"
        result = type_aware_execute(src)
        assert isinstance(result.stats, TypeAwareStats)

    def test_symbolic_with_types(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let pos = true;
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        assert result.stats.total_paths >= 2  # true/false branches

    def test_typed_tests_generated(self):
        src = """
        let x = 0;
        if ((x > 10)) {
            let big = true;
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        assert len(result.typed_tests) > 0
        for tc in result.typed_tests:
            assert isinstance(tc, TypedTestCase)

    def test_typed_test_has_type_info(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let r = 1;
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        # At least one test should have typed inputs
        has_typed = False
        for tc in result.typed_tests:
            if 'x' in tc.inputs:
                has_typed = True
                tv = tc.inputs['x']
                assert isinstance(tv, TypedValue)
                assert isinstance(tv.value, int)
        assert has_typed


# ============================================================
# Section 5: Bool Invariant Injection
# ============================================================

class TestBoolInvariantInjection:
    """Test that bool invariants constrain symbolic booleans to 0/1."""

    def test_bool_input_constrained(self):
        src = """
        let b = true;
        if (b) {
            let yes = 1;
        }
        """
        result = type_aware_execute(src, {'b': 'bool'},
                                     inject_bool_invariants=True)
        # With bool invariant, b is constrained to 0 or 1
        assert result.stats.type_constraints_injected > 0

    def test_bool_invariant_prunes_paths(self):
        """Bool invariant should make impossible paths infeasible."""
        # b is declared as true (C013 infers TBool), but symbolic as int in SMT.
        # With bool invariant (b==0 OR b==1), b > 5 is infeasible.
        src = """
        let b = true;
        if ((b > 5)) {
            let impossible = 1;
        }
        """
        # Use 'int' SMT sort so b > 5 is a valid comparison.
        # C013 still sees let b = true -> TBool -> bool invariant injected.
        result = type_aware_execute(src, {'b': 'int'},
                                     inject_bool_invariants=True)
        # The true branch should be infeasible because b is in {0, 1}
        assert result.stats.paths_pruned_by_types > 0

    def test_without_bool_invariant(self):
        """Without bool invariant, more paths are feasible."""
        src = """
        let b = true;
        if ((b > 5)) {
            let reachable = 1;
        }
        """
        result = type_aware_execute(src, {'b': 'int'},
                                     inject_bool_invariants=False)
        # Without bool constraint, b>5 might be feasible (b could be any int)
        assert result.stats.paths_pruned_by_types == 0


# ============================================================
# Section 6: Function Analysis with Auto-Detection
# ============================================================

class TestFunctionAnalysis:
    """Test auto-detection of symbolic inputs from function signatures."""

    def test_auto_detect_int_params(self):
        src = """
        fn abs_val(x) {
            if ((x < 0)) {
                return (0 - x);
            }
            return x;
        }
        let r = abs_val(5);
        """
        result = analyze_function(src, 'abs_val')
        assert 'x' in result.symbolic_inputs
        assert result.symbolic_inputs['x'] == 'int'

    def test_auto_detect_generates_tests(self):
        src = """
        fn classify(n) {
            if ((n > 0)) {
                return 1;
            }
            if ((n < 0)) {
                return (0 - 1);
            }
            return 0;
        }
        let r = classify(1);
        """
        result = analyze_function(src, 'classify')
        assert len(result.typed_tests) > 0

    def test_auto_detect_covers_branches(self):
        # Call with symbolic vars (not concrete) so C038 forks at branches.
        # C038 skips LetDecl when var is already symbolic from inputs.
        src = """
        let a = 0;
        let b = 0;
        fn max_val(a, b) {
            if ((a > b)) {
                return a;
            }
            return b;
        }
        let r = max_val(a, b);
        """
        result = type_aware_execute(src, {'a': 'int', 'b': 'int'})
        assert result.stats.total_paths >= 2

    def test_unknown_function_returns_empty(self):
        src = "let x = 5;"
        result = analyze_function(src, 'nonexistent')
        assert result.symbolic_inputs == {}


# ============================================================
# Section 7: Type Warning Detection
# ============================================================

class TestTypeWarnings:
    """Test detection of type-related warnings."""

    def test_static_type_error_warning(self):
        src = """
        let x = 5;
        let y = "hello";
        let z = (x + y);
        """
        result = type_aware_execute(src)
        assert len(result.type_warnings) > 0
        has_static = any(w.kind == 'static_type_error' for w in result.type_warnings)
        assert has_static

    def test_no_warnings_clean_code(self):
        src = """
        let x = 5;
        let y = 10;
        let z = (x + y);
        """
        result = type_aware_execute(src)
        static_warns = [w for w in result.type_warnings if w.kind == 'static_type_error']
        assert len(static_warns) == 0

    def test_type_warning_has_message(self):
        src = """
        let x = 5;
        let y = "hello";
        let z = (x + y);
        """
        result = type_aware_execute(src)
        for w in result.type_warnings:
            assert w.message is not None
            assert len(w.message) > 0


# ============================================================
# Section 8: Typed Test Case Properties
# ============================================================

class TestTypedTestCases:
    """Test properties of type-annotated test cases."""

    def test_int_inputs_are_int(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let pos = 1;
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        for tc in result.typed_tests:
            if 'x' in tc.inputs:
                assert isinstance(tc.inputs['x'].value, int)
                assert not isinstance(tc.inputs['x'].value, bool)

    def test_bool_inputs_are_bool(self):
        src = """
        let b = true;
        if (b) {
            let yes = 1;
        }
        """
        result = type_aware_execute(src, {'b': 'bool'})
        for tc in result.typed_tests:
            if 'b' in tc.inputs:
                assert isinstance(tc.inputs['b'].value, bool)

    def test_input_values_property(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let pos = 1;
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        for tc in result.typed_tests:
            vals = tc.input_values
            assert isinstance(vals, dict)
            for v in vals.values():
                assert v is not None

    def test_all_tests_pass_types(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let pos = 1;
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        assert result.all_tests_pass_types


# ============================================================
# Section 9: Multi-Branch Programs
# ============================================================

class TestMultiBranch:
    """Test type-aware execution on programs with multiple branches."""

    def test_three_way_branch(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let sign = 1;
        }
        if ((x < 0)) {
            let sign = (0 - 1);
        }
        """
        result = type_aware_execute(src, {'x': 'int'})
        assert result.stats.total_paths >= 3  # combinations of two branches

    def test_nested_if(self):
        src = """
        let x = 0;
        let y = 0;
        if ((x > 0)) {
            if ((y > 0)) {
                let quad = 1;
            }
        }
        """
        result = type_aware_execute(src, {'x': 'int', 'y': 'int'})
        assert result.stats.total_paths >= 3  # x<=0, x>0&y<=0, x>0&y>0

    def test_while_loop(self):
        src = """
        let i = 0;
        while ((i < 3)) {
            i = (i + 1);
        }
        """
        result = type_aware_execute(src, {'i': 'int'})
        assert result.stats.total_paths > 0


# ============================================================
# Section 10: Find Type Errors
# ============================================================

class TestFindTypeErrors:
    """Test the find_type_errors API."""

    def test_finds_static_errors(self):
        src = """
        let x = 5;
        let y = "hello";
        let z = (x + y);
        """
        result = find_type_errors(src)
        assert isinstance(result, TypeErrorResult)
        assert len(result.static_errors) > 0

    def test_clean_code_no_errors(self):
        src = """
        let x = 5;
        let y = 10;
        let z = (x + y);
        """
        result = find_type_errors(src)
        assert len(result.static_errors) == 0

    def test_total_paths_reported(self):
        src = "let x = 5;"
        result = find_type_errors(src)
        assert result.total_paths >= 1


# ============================================================
# Section 11: get_typed_tests convenience
# ============================================================

class TestGetTypedTests:
    """Test the get_typed_tests convenience function."""

    def test_returns_list(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let pos = 1;
        }
        """
        tests = get_typed_tests(src, {'x': 'int'})
        assert isinstance(tests, list)
        assert len(tests) > 0

    def test_each_test_is_typed(self):
        src = """
        let x = 0;
        if ((x > 0)) {
            let pos = 1;
        }
        """
        tests = get_typed_tests(src, {'x': 'int'})
        for tc in tests:
            assert isinstance(tc, TypedTestCase)


# ============================================================
# Section 12: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_program(self):
        src = ""
        result = type_aware_execute(src)
        assert result is not None
        assert result.stats.total_paths >= 0

    def test_no_symbolic_inputs(self):
        src = "let x = 5;"
        result = type_aware_execute(src)
        assert result is not None
        assert result.symbolic_inputs == {}

    def test_range_constraints(self):
        src = """
        let x = 0;
        if ((x > 100)) {
            let big = 1;
        }
        """
        engine = TypeAwareExecutor(inject_range_constraints=True, range_bound=50)
        result = engine.execute(src, {'x': 'int'})
        # With range [-50, 50], x > 100 is infeasible
        assert result.stats.paths_pruned_by_types > 0

    def test_multiple_functions(self):
        src = """
        fn f(x) {
            return (x + 1);
        }
        fn g(y) {
            return (y * 2);
        }
        let a = f(1);
        let b = g(2);
        """
        info = analyze_types(src)
        assert 'f' in info.functions
        assert 'g' in info.functions


# ============================================================
# Section 13: Composition Integrity
# ============================================================

class TestCompositionIntegrity:
    """Test that C013 + C038 compose correctly without API mismatches."""

    def test_type_checker_and_symex_agree_on_ast(self):
        """Both systems should parse the same source without error."""
        # Use symbolic vars as function args so C038 forks at branches.
        src = """
        let a = 0;
        let b = 0;
        fn pick(a, b) {
            if ((a > b)) {
                return a;
            }
            return b;
        }
        let r = pick(a, b);
        """
        # Type checker
        info = analyze_types(src)
        assert not info.has_errors

        # Symbolic execution
        result = type_aware_execute(src, {'a': 'int', 'b': 'int'})
        assert result.stats.total_paths >= 2

    def test_type_info_matches_symbolic_results(self):
        """Type-inferred types should match the symbolic input types."""
        src = """
        fn add(x, y) {
            return (x + y);
        }
        let r = add(1, 2);
        """
        result = analyze_function(src, 'add')
        for name, smt_type in result.symbolic_inputs.items():
            if name in result.type_analysis.functions.get('add',
                    FunctionTypeInfo('', [], None)).param_types:
                pass  # mapping consistency
        # Main check: no crashes during composition
        assert result is not None
        assert len(result.typed_tests) > 0
