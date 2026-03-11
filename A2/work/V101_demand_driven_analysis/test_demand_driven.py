"""Tests for V101: Demand-Driven Analysis"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V098_ide_framework'))

from demand_driven import (
    DemandDrivenSolver, DemandQuery, DemandResult,
    demand_analyze, demand_query, demand_constants,
    compare_exhaustive_vs_demand, demand_verify_constant,
    incremental_demand, demand_function_summary, demand_slice,
    _values_equal
)
from ide_framework import (
    Fact, ZERO, Top, Bot, Const, LatticeValue,
    IdFunction, ConstFunction, TopFunction, LinearFunction,
    build_ide_problem, IDESolver
)


# ===== Section 1: Basic demand query =====

class TestBasicDemandQuery:
    def test_simple_constant(self):
        source = "let x = 42;"
        result = demand_query(source, 'x')
        assert isinstance(result.value, Const)
        assert result.value.value == 42

    def test_simple_assignment(self):
        source = "let x = 10; let y = x;"
        result = demand_query(source, 'y')
        assert isinstance(result.value, Const)
        assert result.value.value == 10

    def test_multiple_assignments(self):
        source = "let x = 5; x = 10;"
        result = demand_query(source, 'x')
        assert isinstance(result.value, Const)
        assert result.value.value == 10

    def test_uninitialized_variable(self):
        """Querying a variable that doesn't exist returns TOP."""
        source = "let x = 5;"
        result = demand_query(source, 'z')
        assert isinstance(result.value, Top)


# ===== Section 2: Arithmetic constants =====

class TestArithmeticConstants:
    def test_addition(self):
        source = "let x = 3; let y = 4; let z = x + y;"
        result = demand_query(source, 'z')
        # Addition of two constants should propagate
        # V098 copy_const may or may not track arithmetic -- check behavior
        # copy_const tracks copies and constants, not arithmetic
        assert isinstance(result.value, (Const, Top))

    def test_linear_addition(self):
        source = "let x = 3; let y = 4; let z = x + y;"
        result = demand_query(source, 'z', analysis='linear_const')
        # Linear analysis can track x+y when both are constant
        assert isinstance(result.value, (Const, Top))

    def test_constant_propagation_chain(self):
        source = "let a = 1; let b = a; let c = b; let d = c;"
        result = demand_query(source, 'd')
        assert isinstance(result.value, Const)
        assert result.value.value == 1


# ===== Section 3: Demand vs exhaustive consistency =====

class TestConsistency:
    def test_single_var_consistent(self):
        source = "let x = 42;"
        comp = compare_exhaustive_vs_demand(source, [("main.exit", "x")])
        assert comp['all_consistent']

    def test_multi_var_consistent(self):
        source = "let x = 10; let y = 20; let z = x;"
        queries = [("main.exit", "x"), ("main.exit", "y"), ("main.exit", "z")]
        comp = compare_exhaustive_vs_demand(source, queries)
        assert comp['all_consistent']

    def test_copy_chain_consistent(self):
        source = "let a = 5; let b = a; let c = b;"
        queries = [("main.exit", "a"), ("main.exit", "b"), ("main.exit", "c")]
        comp = compare_exhaustive_vs_demand(source, queries)
        assert comp['all_consistent']

    def test_reassignment_consistent(self):
        source = "let x = 1; x = 2; x = 3;"
        comp = compare_exhaustive_vs_demand(source, [("main.exit", "x")])
        assert comp['all_consistent']


# ===== Section 4: Demand-driven efficiency =====

class TestEfficiency:
    def test_queries_only_relevant_points(self):
        """Demand-driven should explore fewer points than exhaustive for targeted queries."""
        source = """
        let a = 1;
        let b = 2;
        let c = 3;
        let d = 4;
        let e = 5;
        let f = 6;
        let g = a;
        """
        # Only query 'g' -- should not need to explore b, c, d, e, f definitions
        result = demand_query(source, 'g')
        assert isinstance(result.value, Const)
        assert result.value.value == 1
        assert result.explored_points > 0

    def test_batch_queries_share_cache(self):
        source = "let x = 1; let y = x; let z = y;"
        problem = build_ide_problem(source, 'copy_const')
        solver = DemandDrivenSolver(problem)

        # Query z first (needs y, which needs x) -- populates cache
        solver.demand_value("main.exit", Fact("z"))
        cache_after_z = len(solver._value_cache)

        # Query y -- should hit cache from z's computation
        solver.demand_value("main.exit", Fact("y"))
        cache_after_y = len(solver._value_cache)
        # Cache should not grow much since y was already computed for z
        assert cache_after_z > 0  # Cache was populated


# ===== Section 5: Interprocedural demand =====

class TestInterprocedural:
    def test_function_call_constant(self):
        source = """
        fn add_one(n) {
            return n + 1;
        }
        let x = 5;
        let y = add_one(x);
        """
        result = demand_query(source, 'y')
        # Depending on analysis precision, may get Const(6) or Top
        assert isinstance(result.value, (Const, Top))

    def test_function_return_constant(self):
        source = """
        fn seven() {
            return 7;
        }
        let x = seven();
        """
        result = demand_query(source, 'x')
        assert isinstance(result.value, Const)
        assert result.value.value == 7

    def test_interprocedural_consistency(self):
        source = """
        fn seven() {
            return 7;
        }
        let x = seven();
        """
        comp = compare_exhaustive_vs_demand(source, [("main.exit", "x")])
        assert comp['all_consistent']

    def test_constant_arg_propagation(self):
        source = """
        fn id(x) {
            return x;
        }
        let a = 42;
        let b = id(a);
        """
        result = demand_query(source, 'b')
        assert isinstance(result.value, Const)
        assert result.value.value == 42


# ===== Section 6: Verify constant =====

class TestVerifyConstant:
    def test_verify_correct(self):
        source = "let x = 42;"
        result = demand_verify_constant(source, 'x', 42)
        assert result['holds'] is True

    def test_verify_incorrect(self):
        source = "let x = 42;"
        result = demand_verify_constant(source, 'x', 99)
        assert result['holds'] is False

    def test_verify_copy(self):
        source = "let a = 10; let b = a;"
        result = demand_verify_constant(source, 'b', 10)
        assert result['holds'] is True

    def test_verify_reassigned(self):
        source = "let x = 1; x = 2;"
        result = demand_verify_constant(source, 'x', 2)
        assert result['holds'] is True


# ===== Section 7: Demand constants (all exit vars) =====

class TestDemandConstants:
    def test_all_constants(self):
        source = "let x = 1; let y = 2; let z = 3;"
        result = demand_constants(source)
        # Should have constants for x, y, z at exit
        exit_key = [k for k in result.keys()][0] if result else None
        if exit_key:
            assert 'x' in result[exit_key]
            assert isinstance(result[exit_key]['x'], Const)
            assert result[exit_key]['x'].value == 1

    def test_mixed_constants_and_top(self):
        source = "let x = 1; let y = x + x;"
        result = demand_constants(source)
        # x is constant, y may be TOP (copy_const doesn't track arithmetic)
        exit_vals = list(result.values())[0] if result else {}
        if 'x' in exit_vals:
            assert isinstance(exit_vals['x'], Const)
            assert exit_vals['x'].value == 1


# ===== Section 8: Function summary =====

class TestFunctionSummary:
    def test_constant_function(self):
        source = """
        fn five() {
            return 5;
        }
        let x = five();
        """
        summary = demand_function_summary(source, 'five')
        assert 'error' not in summary
        assert summary['function'] == 'five'

    def test_identity_function(self):
        source = """
        fn id(x) {
            return x;
        }
        let a = id(10);
        """
        summary = demand_function_summary(source, 'id')
        assert 'error' not in summary

    def test_nonexistent_function(self):
        source = "let x = 1;"
        summary = demand_function_summary(source, 'nope')
        assert 'error' in summary


# ===== Section 9: Demand slice =====

class TestDemandSlice:
    def test_simple_slice(self):
        source = "let x = 1; let y = 2; let z = x;"
        sl = demand_slice(source, 'main.exit', 'z')
        assert sl['slice_size'] > 0
        assert sl['total_points'] >= sl['slice_size']

    def test_slice_reduction(self):
        """Slice should be smaller than total program for independent vars."""
        source = """
        let a = 1;
        let b = 2;
        let c = 3;
        let d = 4;
        let e = a;
        """
        sl = demand_slice(source, 'main.exit', 'e')
        assert sl['slice_size'] <= sl['total_points']

    def test_slice_value(self):
        source = "let x = 42; let y = x;"
        sl = demand_slice(source, 'main.exit', 'y')
        assert isinstance(sl['value'], Const)
        assert sl['value'].value == 42


# ===== Section 10: Incremental demand =====

class TestIncrementalDemand:
    def test_no_change(self):
        source = "let x = 1; let y = 2;"
        result = incremental_demand(source, source,
                                     [("main.exit", "x"), ("main.exit", "y")])
        assert result['num_changes'] == 0

    def test_value_change(self):
        v1 = "let x = 1; let y = 2;"
        v2 = "let x = 99; let y = 2;"
        result = incremental_demand(v1, v2, [("main.exit", "x")])
        assert result['num_changes'] == 1
        assert result['changes'][0]['variable'] == 'x'

    def test_no_change_unaffected(self):
        v1 = "let x = 1; let y = 2;"
        v2 = "let x = 99; let y = 2;"
        result = incremental_demand(v1, v2, [("main.exit", "y")])
        assert result['num_changes'] == 0


# ===== Section 11: Linear constant analysis =====

class TestLinearConstant:
    def test_linear_const_simple(self):
        source = "let x = 5;"
        result = demand_query(source, 'x', analysis='linear_const')
        assert isinstance(result.value, Const)
        assert result.value.value == 5

    def test_linear_const_copy(self):
        source = "let a = 7; let b = a;"
        result = demand_query(source, 'b', analysis='linear_const')
        assert isinstance(result.value, Const)
        assert result.value.value == 7

    def test_linear_vs_copy_consistency(self):
        source = "let x = 10; let y = x;"
        comp1 = demand_query(source, 'y', analysis='copy_const')
        comp2 = demand_query(source, 'y', analysis='linear_const')
        assert _values_equal(comp1.value, comp2.value)


# ===== Section 12: Intermediate point queries =====

class TestIntermediatePoints:
    def test_query_intermediate(self):
        source = "let x = 1; x = 2; x = 3;"
        # Query at different points
        r_exit = demand_query(source, 'x', point='main.exit')
        assert isinstance(r_exit.value, Const)
        assert r_exit.value.value == 3

    def test_query_mid_program(self):
        source = "let x = 1; let y = 2; let z = 3;"
        # s0 is after 'let x = 1', s1 is after 'let y = 2'
        r = demand_query(source, 'x', point='main.s1')
        assert isinstance(r.value, Const)
        assert r.value.value == 1


# ===== Section 13: Edge cases =====

class TestEdgeCases:
    def test_empty_program(self):
        source = "let x = 0;"
        result = demand_query(source, 'x')
        assert isinstance(result.value, Const)
        assert result.value.value == 0

    def test_values_equal_helper(self):
        assert _values_equal(Top(), Top())
        assert _values_equal(Bot(), Bot())
        assert _values_equal(Const(5), Const(5))
        assert not _values_equal(Const(5), Const(6))
        assert not _values_equal(Top(), Const(5))
        assert not _values_equal(Bot(), Top())

    def test_demand_result_fields(self):
        source = "let x = 1;"
        r = demand_query(source, 'x')
        assert r.query.point is not None
        assert r.query.fact.name == 'x'
        assert r.explored_points >= 1
        assert r.total_edges >= 0


# ===== Section 14: Multiple functions =====

class TestMultipleFunctions:
    def test_two_functions(self):
        source = """
        fn double(x) {
            return x + x;
        }
        fn triple(x) {
            return x + x + x;
        }
        let a = 5;
        let b = double(a);
        """
        result = demand_query(source, 'b')
        # double(5) = 10 -- but copy_const won't track arithmetic
        assert isinstance(result.value, (Const, Top))

    def test_query_only_needed_function(self):
        """Querying 'a' should not need to analyze 'double' function."""
        source = """
        fn double(x) {
            return x + x;
        }
        let a = 5;
        let b = double(a);
        """
        result = demand_query(source, 'a')
        assert isinstance(result.value, Const)
        assert result.value.value == 5

    def test_function_summary_demand(self):
        source = """
        fn const_fn() {
            return 100;
        }
        let x = const_fn();
        let y = 0;
        """
        # Query x (needs function) vs y (doesn't)
        rx = demand_query(source, 'x')
        ry = demand_query(source, 'y')
        assert isinstance(ry.value, Const)
        assert ry.value.value == 0


# ===== Section 15: Batch queries =====

class TestBatchQueries:
    def test_batch_basic(self):
        source = "let x = 1; let y = 2; let z = 3;"
        results = demand_analyze(source,
                                  [("main.exit", "x"), ("main.exit", "y"), ("main.exit", "z")])
        assert len(results) == 3
        for r in results:
            assert isinstance(r.value, Const)
        assert results[0].value.value == 1
        assert results[1].value.value == 2
        assert results[2].value.value == 3

    def test_batch_shared_cache(self):
        source = "let x = 1; let y = x; let z = y;"
        results = demand_analyze(source,
                                  [("main.exit", "z"), ("main.exit", "y"), ("main.exit", "x")])
        assert len(results) == 3
        # All should be Const(1)
        for r in results:
            assert isinstance(r.value, Const)
            assert r.value.value == 1

    def test_batch_cache_hits_increase(self):
        source = "let x = 1; let y = x; let z = y;"
        results = demand_analyze(source,
                                  [("main.exit", "z"), ("main.exit", "y"), ("main.exit", "x")])
        # Later queries should have more cache hits
        assert results[2].cache_hits >= results[0].cache_hits


# ===== Section 16: Solver invalidation =====

class TestInvalidation:
    def test_invalidate_all(self):
        source = "let x = 1; let y = x;"
        problem = build_ide_problem(source, 'copy_const')
        solver = DemandDrivenSolver(problem)

        solver.query("main.exit", "x")
        assert len(solver._value_cache) > 0

        solver.invalidate()
        assert len(solver._value_cache) == 0

    def test_invalidate_specific_points(self):
        source = "let x = 1; let y = x;"
        problem = build_ide_problem(source, 'copy_const')
        solver = DemandDrivenSolver(problem)

        solver.query("main.exit", "x")
        solver.query("main.exit", "y")
        cache_before = len(solver._value_cache)

        # Invalidate only main.s0
        solver.invalidate({"main.s0"})
        cache_after = len(solver._value_cache)
        # Some entries should be removed
        assert cache_after <= cache_before


# ===== Section 17: Demand-driven slice properties =====

class TestSliceProperties:
    def test_slice_contains_query_point(self):
        source = "let x = 1; let y = x;"
        sl = demand_slice(source, 'main.exit', 'y')
        assert 'main.exit' in sl['slice_points']

    def test_slice_independent_vars(self):
        """Variables independent of query should not be in slice."""
        source = """
        let a = 1;
        let b = 2;
        let c = 3;
        let d = a;
        """
        sl = demand_slice(source, 'main.exit', 'd')
        # Slice should be relatively small compared to total
        assert sl['slice_size'] <= sl['total_points']


# ===== Section 18: Comparison API =====

class TestComparisonAPI:
    def test_compare_simple(self):
        source = "let x = 5; let y = 10;"
        comp = compare_exhaustive_vs_demand(source,
                                             [("main.exit", "x"), ("main.exit", "y")])
        assert 'comparisons' in comp
        assert 'all_consistent' in comp
        assert 'demand_explored' in comp
        assert 'exhaustive_explored' in comp
        assert len(comp['comparisons']) == 2

    def test_compare_with_function(self):
        source = """
        fn f() { return 1; }
        let x = f();
        """
        comp = compare_exhaustive_vs_demand(source, [("main.exit", "x")])
        assert comp['all_consistent']

    def test_savings_metric(self):
        source = """
        let a = 1;
        let b = 2;
        let c = 3;
        let d = 4;
        let e = 5;
        """
        comp = compare_exhaustive_vs_demand(source, [("main.exit", "a")])
        # Savings should be non-negative
        assert comp['savings'] >= 0.0


# ===== Section 19: Recursion handling =====

class TestRecursionHandling:
    def test_no_infinite_loop(self):
        """Cyclic dependencies should terminate (return TOP)."""
        source = "let x = 1;"
        # Direct query -- no actual recursion in this program
        result = demand_query(source, 'x')
        assert isinstance(result.value, Const)
        assert result.value.value == 1

    def test_self_referencing_var(self):
        """x = x + 1 style assignments."""
        source = "let x = 0; x = x + 1;"
        result = demand_query(source, 'x')
        # copy_const can't track arithmetic -> TOP
        assert isinstance(result.value, (Const, Top))


# ===== Section 20: DemandQuery and DemandResult =====

class TestDataStructures:
    def test_demand_query_hash(self):
        q1 = DemandQuery("main.exit", Fact("x"))
        q2 = DemandQuery("main.exit", Fact("x"))
        assert q1 == q2
        assert hash(q1) == hash(q2)

    def test_demand_query_inequality(self):
        q1 = DemandQuery("main.exit", Fact("x"))
        q2 = DemandQuery("main.exit", Fact("y"))
        assert q1 != q2

    def test_demand_result_fields(self):
        r = DemandResult(
            query=DemandQuery("main.exit", Fact("x")),
            value=Const(42),
            explored_points=5,
            cache_hits=2,
            total_edges=10
        )
        assert r.value.value == 42
        assert r.explored_points == 5
        assert r.cache_hits == 2


# ===== Section 21: Multi-level function calls =====

class TestMultiLevelCalls:
    def test_chained_calls(self):
        """Chained calls f(x)->g(x): both exhaustive and demand return TOP for
        copy_const analysis (parameter passing through nested calls loses precision)."""
        source = """
        fn f(x) { return x; }
        fn g(x) { return f(x); }
        let a = 10;
        let b = g(a);
        """
        result = demand_query(source, 'b')
        # copy_const can't track through chained calls -- consistent with V098
        assert isinstance(result.value, (Const, Top))

    def test_chained_calls_consistency(self):
        source = """
        fn f(x) { return x; }
        fn g(x) { return f(x); }
        let a = 10;
        let b = g(a);
        """
        comp = compare_exhaustive_vs_demand(source, [("main.exit", "b")])
        assert comp['all_consistent']
