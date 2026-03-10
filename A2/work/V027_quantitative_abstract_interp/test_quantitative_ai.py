"""Tests for V027: Quantitative Abstract Interpretation -- Resource Bound Analysis."""

import sys
import os
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from quantitative_ai import (
    analyze_bounds, loop_bound, complexity_class, resource_count,
    verify_bound, compare_bounds, bound_summary,
    BoundExpr, ComplexityClass, LoopBoundResult, BoundResult,
    parse, find_all_loops, count_operations, find_symbolic_params,
    collect_pre_assignments, compute_total_bound,
)


# ==================== BoundExpr tests ====================

class TestBoundExpr:
    def test_constant(self):
        b = BoundExpr.constant(10)
        assert b.kind == "constant"
        assert b.value == 10
        assert str(b) == "10"
        assert b.evaluate({}) == 10

    def test_linear_single_param(self):
        b = BoundExpr.linear({'n': 1, '_const': 0}, {'n'})
        assert b.kind == "linear"
        assert 'n' in str(b)
        assert b.evaluate({'n': 10}) == 10
        assert b.evaluate({'n': 0}) == 0

    def test_linear_with_constant(self):
        b = BoundExpr.linear({'n': 1, '_const': -1}, {'n'})
        assert b.evaluate({'n': 10}) == 9
        assert b.evaluate({'n': 1}) == 0  # max(0, 0)

    def test_linear_two_params(self):
        b = BoundExpr.linear({'n': 1, 'm': 1, '_const': 0}, {'n', 'm'})
        assert b.evaluate({'n': 5, 'm': 3}) == 8

    def test_product(self):
        a = BoundExpr.constant(10)
        b = BoundExpr.constant(5)
        p = BoundExpr.product(a, b)
        assert p.kind == "product"
        assert p.evaluate({}) == 50

    def test_product_symbolic(self):
        a = BoundExpr.linear({'n': 1, '_const': 0}, {'n'})
        b = BoundExpr.linear({'m': 1, '_const': 0}, {'m'})
        p = BoundExpr.product(a, b)
        assert p.evaluate({'n': 10, 'm': 5}) == 50

    def test_get_params_constant(self):
        assert BoundExpr.constant(5).get_params() == set()

    def test_get_params_linear(self):
        b = BoundExpr.linear({'n': 1, '_const': 0}, {'n'})
        assert b.get_params() == {'n'}

    def test_get_params_product(self):
        a = BoundExpr.linear({'n': 1, '_const': 0}, {'n'})
        b = BoundExpr.linear({'m': 1, '_const': 0}, {'m'})
        p = BoundExpr.product(a, b)
        assert p.get_params() == {'n', 'm'}

    def test_evaluate_missing_param(self):
        b = BoundExpr.linear({'n': 1, '_const': 0}, {'n'})
        assert b.evaluate({}) is None


# ==================== AST utility tests ====================

class TestASTUtils:
    def test_parse_simple(self):
        prog = parse("let x = 5;")
        assert len(prog.stmts) == 1

    def test_find_loops_none(self):
        prog = parse("let x = 5;")
        loops = find_all_loops(prog)
        assert len(loops) == 0

    def test_find_loops_single(self):
        prog = parse("let i = 0; while (i < 10) { i = i + 1; }")
        loops = find_all_loops(prog)
        assert len(loops) == 1
        assert loops[0][1] == 0  # depth 0

    def test_find_loops_nested(self):
        src = """
        let i = 0;
        while (i < 10) {
            let j = 0;
            while (j < 5) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        prog = parse(src)
        loops = find_all_loops(prog)
        assert len(loops) == 2
        assert loops[0][1] == 0  # outer at depth 0
        assert loops[1][1] == 1  # inner at depth 1

    def test_find_loops_sequential(self):
        src = """
        let i = 0;
        while (i < 10) { i = i + 1; }
        let j = 0;
        while (j < 5) { j = j + 1; }
        """
        prog = parse(src)
        loops = find_all_loops(prog)
        assert len(loops) == 2
        assert loops[0][1] == 0
        assert loops[1][1] == 0

    def test_count_operations(self):
        prog = parse("let x = 1 + 2; let y = x * 3;")
        counts = count_operations(prog)
        assert counts.assignments == 2
        assert counts.arithmetic_ops == 2

    def test_count_operations_with_comparison(self):
        prog = parse("let x = 5; if (x > 3) { x = 1; }")
        counts = count_operations(prog)
        assert counts.assignments >= 1
        assert counts.comparisons >= 1

    def test_find_symbolic_params(self):
        params = find_symbolic_params("let y = x + 1;")
        assert 'x' in params
        assert 'y' not in params

    def test_find_symbolic_params_none(self):
        params = find_symbolic_params("let x = 5; let y = x + 1;")
        assert len(params) == 0

    def test_collect_pre_assignments(self):
        prog = parse("let x = 10; let y = 20; while (x > 0) { x = x - 1; }")
        pre = collect_pre_assignments(prog.stmts, 0)
        assert pre['x'] == 10
        assert pre['y'] == 20


# ==================== Loop bound tests ====================

class TestLoopBound:
    def test_simple_countdown(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        assert lb.bound.kind == "constant"
        assert lb.bound.value == 10
        assert lb.is_tight

    def test_countup(self):
        src = "let i = 0; while (i < 10) { i = i + 1; }"
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        assert lb.bound.value == 10 or lb.bound.evaluate({}) == 10

    def test_countdown_from_20(self):
        src = "let i = 20; while (i > 0) { i = i - 1; }"
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        assert lb.bound.value == 20

    def test_step_by_2(self):
        src = "let i = 0; while (i < 10) { i = i + 2; }"
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        # Ranking function is 10 - i, initial value 10
        # Actual iterations: 5, but bound is 10 (ranking decreases by 2 per step)
        assert lb.bound.value <= 10

    def test_parametric_bound(self):
        src = "while (n > 0) { n = n - 1; }"
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        params = lb.bound.get_params()
        assert 'n' in params

    def test_no_loop(self):
        src = "let x = 5;"
        result = analyze_bounds(src)
        assert len(result.loop_bounds) == 0
        assert result.complexity == ComplexityClass.O_1

    def test_bound_with_ranking(self):
        src = "let x = 100; while (x > 0) { x = x - 1; }"
        lb = loop_bound(src, 0)
        assert lb.ranking_function is not None
        assert lb.bound.value == 100


# ==================== Complexity classification tests ====================

class TestComplexity:
    def test_no_loop_o1(self):
        src = "let x = 5; let y = x + 1;"
        cc, text = complexity_class(src)
        assert cc == ComplexityClass.O_1

    def test_single_loop_constant_o1(self):
        src = "let i = 0; while (i < 10) { i = i + 1; }"
        cc, text = complexity_class(src)
        assert cc == ComplexityClass.O_1  # Constant bound = O(1)

    def test_parametric_loop_on(self):
        src = "while (n > 0) { n = n - 1; }"
        cc, text = complexity_class(src)
        assert cc == ComplexityClass.O_N
        assert 'n' in text

    def test_nested_loops_on2(self):
        src = """
        while (n > 0) {
            let j = m;
            while (j > 0) {
                j = j - 1;
            }
            n = n - 1;
        }
        """
        cc, text = complexity_class(src)
        assert cc == ComplexityClass.O_N2

    def test_sequential_loops_on(self):
        src = """
        let i = n;
        while (i > 0) { i = i - 1; }
        let j = n;
        while (j > 0) { j = j - 1; }
        """
        cc, text = complexity_class(src)
        # Two sequential O(n) loops = O(n)
        assert cc == ComplexityClass.O_N


# ==================== Resource counting tests ====================

class TestResourceCount:
    def test_empty_program(self):
        counts = resource_count("let x = 5;")
        assert counts.assignments == 1
        assert counts.comparisons == 0

    def test_arithmetic(self):
        counts = resource_count("let x = 1 + 2 * 3;")
        assert counts.arithmetic_ops == 2

    def test_loop_body(self):
        counts = resource_count("""
            let i = 0;
            while (i < 10) {
                let x = i * 2;
                i = i + 1;
            }
        """)
        assert counts.assignments >= 3  # i=0, x=i*2, i=i+1
        assert counts.arithmetic_ops >= 2

    def test_function_call(self):
        counts = resource_count("""
            fn f(x) { return x + 1; }
            let y = f(5);
        """)
        assert counts.function_calls >= 1


# ==================== Full analysis tests ====================

class TestAnalyzeBounds:
    def test_simple_program(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_bounds(src)
        assert isinstance(result, BoundResult)
        assert len(result.loop_bounds) == 1
        assert result.loop_bounds[0].bound is not None
        assert result.loop_bounds[0].bound.value == 10

    def test_no_loops(self):
        src = "let x = 1; let y = 2; let z = x + y;"
        result = analyze_bounds(src)
        assert len(result.loop_bounds) == 0
        assert result.complexity == ComplexityClass.O_1

    def test_parametric_program(self):
        src = "while (n > 0) { n = n - 1; }"
        result = analyze_bounds(src)
        assert len(result.loop_bounds) == 1
        assert result.params == {'n'} or 'n' in result.params

    def test_nested_loop_analysis(self):
        src = """
        while (n > 0) {
            let j = m;
            while (j > 0) {
                j = j - 1;
            }
            n = n - 1;
        }
        """
        result = analyze_bounds(src)
        assert len(result.loop_bounds) == 2
        # Both loops should have bounds
        assert all(lb.bound is not None for lb in result.loop_bounds)

    def test_multiple_loops(self):
        src = """
        let a = 5;
        while (a > 0) { a = a - 1; }
        let b = 3;
        while (b > 0) { b = b - 1; }
        """
        result = analyze_bounds(src)
        assert len(result.loop_bounds) == 2

    def test_total_bound_single(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_bounds(src)
        assert result.total_bound is not None
        assert result.total_bound.evaluate({}) == 10

    def test_resource_counts_in_result(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_bounds(src)
        assert result.resource_counts.assignments >= 2


# ==================== Bound verification tests ====================

class TestVerifyBound:
    def test_verify_correct_concrete(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        valid, msg = verify_bound(src, 0, 10)
        assert valid

    def test_verify_loose_concrete(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        valid, msg = verify_bound(src, 0, 20)
        assert valid

    def test_verify_tight_concrete_fails(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        valid, msg = verify_bound(src, 0, 5)
        assert not valid

    def test_verify_parametric(self):
        src = "while (n > 0) { n = n - 1; }"
        # Proposed bound: n
        valid, msg = verify_bound(src, 0, {'n': 1, '_const': 0})
        assert valid

    def test_verify_parametric_too_tight(self):
        src = "while (n > 0) { n = n - 1; }"
        # Proposed bound: n - 5 (too tight for large n but wrong for small n)
        valid, msg = verify_bound(src, 0, {'n': 1, '_const': -5})
        # This should fail because n=3 gives bound=-2 but actual iterations=3
        # Actually the ranking function is n, init value is n, so bound=n
        # Proposed n-5 < n for n > 0, so SMT should find counterexample
        assert not valid


# ==================== Compare bounds tests ====================

class TestCompareBounds:
    def test_compare_same_complexity(self):
        src1 = "let i = 10; while (i > 0) { i = i - 1; }"
        src2 = "let i = 20; while (i > 0) { i = i - 1; }"
        result = compare_bounds(src1, src2)
        assert result['same_complexity']

    def test_compare_different_complexity(self):
        src1 = "let i = 10; while (i > 0) { i = i - 1; }"
        src2 = """
        while (n > 0) {
            let j = m;
            while (j > 0) { j = j - 1; }
            n = n - 1;
        }
        """
        result = compare_bounds(src1, src2)
        assert not result['same_complexity']

    def test_compare_with_params(self):
        src1 = "while (n > 0) { n = n - 1; }"
        src2 = "while (n > 0) { n = n - 1; }"
        result = compare_bounds(src1, src2, params={'n': 10})
        assert 'concrete_bound' in result['program1']


# ==================== Summary tests ====================

class TestSummary:
    def test_bound_summary_simple(self):
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        summary = bound_summary(src)
        assert "Loop 0" in summary
        assert "10" in summary

    def test_bound_summary_no_loops(self):
        src = "let x = 5;"
        summary = bound_summary(src)
        assert "O(1)" in summary

    def test_bound_summary_parametric(self):
        src = "while (n > 0) { n = n - 1; }"
        summary = bound_summary(src)
        assert "Loop 0" in summary
        assert "n" in summary


# ==================== Edge cases ====================

class TestEdgeCases:
    def test_zero_iteration_loop(self):
        src = "let i = 0; while (i > 0) { i = i - 1; }"
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        assert lb.bound.value == 0 or lb.bound.evaluate({}) == 0

    def test_single_iteration(self):
        src = "let i = 1; while (i > 0) { i = i - 1; }"
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        assert lb.bound.value == 1

    def test_loop_with_conditional_body(self):
        src = """
        let i = 10;
        while (i > 0) {
            if (i > 5) {
                i = i - 2;
            } else {
                i = i - 1;
            }
        }
        """
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        # Bound is 10 (ranking function decreases by at least 1 per step)
        assert lb.bound.value <= 10

    def test_accumulator_loop(self):
        src = """
        let sum = 0;
        let i = 10;
        while (i > 0) {
            sum = sum + i;
            i = i - 1;
        }
        """
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        assert lb.bound.value == 10

    def test_bound_evaluate_concrete(self):
        b = BoundExpr.constant(42)
        assert b.evaluate({'anything': 99}) == 42

    def test_nested_three_levels(self):
        src = """
        while (a > 0) {
            while (b > 0) {
                while (c > 0) {
                    c = c - 1;
                }
                b = b - 1;
            }
            a = a - 1;
        }
        """
        result = analyze_bounds(src)
        assert len(result.loop_bounds) == 3
        assert result.complexity == ComplexityClass.O_N3

    def test_loop_with_multiple_vars(self):
        src = """
        let x = 10;
        let y = 5;
        while (x > 0) {
            x = x - 1;
            y = y + 1;
        }
        """
        lb = loop_bound(src, 0)
        assert lb.bound is not None
        assert lb.bound.value == 10

    def test_complexity_text_contains_param(self):
        src = "while (n > 0) { n = n - 1; }"
        cc, text = complexity_class(src)
        assert 'n' in text

    def test_resource_counts_complete(self):
        src = """
        let x = 0;
        let i = 5;
        while (i > 0) {
            x = x + i;
            i = i - 1;
        }
        """
        counts = resource_count(src)
        assert counts.assignments >= 4  # x=0, i=5, x=x+i, i=i-1
        assert counts.arithmetic_ops >= 2  # x+i, i-1

    def test_bound_expr_symbolic(self):
        b = BoundExpr.symbolic("2^n")
        assert str(b) == "2^n"
        assert b.evaluate({}) is None

    def test_linear_negative_coeff(self):
        b = BoundExpr.linear({'n': -1, '_const': 10}, {'n'})
        assert b.evaluate({'n': 3}) == 7
        assert b.evaluate({'n': 15}) == 0  # max(0, -5) = 0


# ==================== Integration tests ====================

class TestIntegration:
    def test_full_pipeline_simple(self):
        """End-to-end: simple countdown analyzed correctly."""
        src = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_bounds(src)

        assert result.complexity == ComplexityClass.O_1
        assert len(result.loop_bounds) == 1
        assert result.loop_bounds[0].bound.value == 10
        assert result.total_bound.evaluate({}) == 10

    def test_full_pipeline_parametric(self):
        """End-to-end: parametric loop analyzed correctly."""
        src = "while (n > 0) { n = n - 1; }"
        result = analyze_bounds(src)

        assert result.complexity == ComplexityClass.O_N
        assert 'n' in result.params
        assert len(result.loop_bounds) == 1
        assert result.loop_bounds[0].bound.get_params() == {'n'}

    def test_full_pipeline_nested(self):
        """End-to-end: nested loops give quadratic complexity."""
        src = """
        while (n > 0) {
            let j = m;
            while (j > 0) {
                j = j - 1;
            }
            n = n - 1;
        }
        """
        result = analyze_bounds(src)

        assert result.complexity == ComplexityClass.O_N2
        assert len(result.loop_bounds) == 2

    def test_summary_readable(self):
        """Summary output is human-readable."""
        src = """
        let i = 10;
        while (i > 0) {
            let j = 5;
            while (j > 0) {
                j = j - 1;
            }
            i = i - 1;
        }
        """
        summary = bound_summary(src)
        assert "Loop 0" in summary
        assert "Loop 1" in summary

    def test_verify_then_compare(self):
        """Verify a bound then compare two programs."""
        src1 = "let i = 10; while (i > 0) { i = i - 1; }"
        src2 = "let i = 20; while (i > 0) { i = i - 1; }"

        v1, _ = verify_bound(src1, 0, 10)
        v2, _ = verify_bound(src2, 0, 20)
        assert v1 and v2

        cmp = compare_bounds(src1, src2)
        assert cmp['same_complexity']

    def test_countup_to_n(self):
        """Count up to parameter n."""
        src = "let i = 0; while (i < n) { i = i + 1; }"
        result = analyze_bounds(src)
        assert len(result.loop_bounds) == 1
        lb = result.loop_bounds[0]
        assert lb.bound is not None
        assert 'n' in lb.bound.get_params()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
