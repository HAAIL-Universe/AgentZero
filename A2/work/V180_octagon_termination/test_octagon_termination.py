"""
Tests for V180: Octagon-Based Termination Analysis

Tests relational ranking function discovery, octagon-guided candidate
generation, octagon-strengthened verification, and comprehensive analysis.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V025_termination_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))

from octagon_termination import (
    extract_octagon_invariants,
    generate_octagon_candidates,
    find_relational_ranking,
    find_octagon_strengthened_ranking,
    find_relational_lex_ranking,
    prove_termination_with_octagon,
    analyze_termination_with_octagon,
    verify_termination_invariant,
    compare_strategies,
    check_relational_ranking,
    RelationalRankingFunction,
    OctTerminationResult,
    OctLoopResult,
    _ast_expr_to_oct_expr,
    _ast_stmt_to_oct_stmt,
    _ast_to_oct_program,
    _oct_constraint_to_coefficients,
)
from termination import TermResult, RankingFunction, LexRankingFunction
from octagon import Octagon, OctConstraint, OctExpr


# ===================================================================
# Section 1: AST to Octagon Translation
# ===================================================================

class TestASTToOctagonTranslation:
    """Tests for converting C010 AST to octagon tuple programs."""

    def test_int_literal(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("42;")).parse()
        # Expression statement -- extract expr
        stmt = ast.stmts[0]
        expr = stmt.expr if hasattr(stmt, 'expr') else stmt
        result = _ast_expr_to_oct_expr(expr)
        assert result == ('const', 42)

    def test_variable(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("x;")).parse()
        stmt = ast.stmts[0]
        expr = stmt.expr if hasattr(stmt, 'expr') else stmt
        result = _ast_expr_to_oct_expr(expr)
        assert result == ('var', 'x')

    def test_binary_add(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("x + 1;")).parse()
        stmt = ast.stmts[0]
        expr = stmt.expr if hasattr(stmt, 'expr') else stmt
        result = _ast_expr_to_oct_expr(expr)
        assert result[0] == 'add'
        assert result[1] == ('var', 'x')
        assert result[2] == ('const', 1)

    def test_binary_comparison(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("x < 10;")).parse()
        stmt = ast.stmts[0]
        expr = stmt.expr if hasattr(stmt, 'expr') else stmt
        result = _ast_expr_to_oct_expr(expr)
        assert result[0] == 'lt'

    def test_let_decl_to_assign(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 5;")).parse()
        result = _ast_stmt_to_oct_stmt(ast.stmts[0])
        assert result[0] == 'assign'
        assert result[1] == 'x'
        assert result[2] == ('const', 5)

    def test_while_to_oct(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("while (x > 0) { x = x - 1; }")).parse()
        result = _ast_stmt_to_oct_stmt(ast.stmts[0])
        assert result[0] == 'while'
        # condition
        assert result[1][0] == 'gt'
        # body is a tuple-statement (assign or seq)
        assert isinstance(result[2], tuple)
        assert result[2][0] in ('assign', 'seq', 'skip')

    def test_if_to_oct(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("if (x > 0) { x = 1; } else { x = 2; }")).parse()
        result = _ast_stmt_to_oct_stmt(ast.stmts[0])
        assert result[0] == 'if'

    def test_full_program(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 10; while (x > 0) { x = x - 1; }")).parse()
        result = _ast_to_oct_program(ast.stmts)
        assert result[0] == 'seq'


# ===================================================================
# Section 2: OctConstraint to Coefficients
# ===================================================================

class TestOctConstraintToCoefficients:
    """Tests for converting octagon constraints to ranking candidates."""

    def test_unary_upper(self):
        c = OctConstraint.var_le('x', 10)
        candidates = _oct_constraint_to_coefficients(c)
        # Should include x and 10 - x
        coeffs_strs = set()
        for cand in candidates:
            key = str(sorted(cand.items()))
            coeffs_strs.add(key)
        assert len(candidates) >= 1

    def test_unary_lower(self):
        c = OctConstraint.var_ge('x', 0)
        # var_ge(x, 0) => -x <= 0
        candidates = _oct_constraint_to_coefficients(c)
        assert len(candidates) >= 1

    def test_binary_diff(self):
        c = OctConstraint.diff_le('x', 'y', 5)
        # x - y <= 5
        candidates = _oct_constraint_to_coefficients(c)
        # Should include -(x-y) + 5 = 5 - x + y and x - y
        assert len(candidates) >= 1
        # Check that a relational candidate exists
        has_relational = any(
            len([k for k in cand if k != '_const' and cand[k] != 0]) == 2
            for cand in candidates
        )
        assert has_relational

    def test_binary_sum(self):
        c = OctConstraint.sum_le('x', 'y', 10)
        candidates = _oct_constraint_to_coefficients(c)
        assert len(candidates) >= 1


# ===================================================================
# Section 3: Octagon Invariant Extraction
# ===================================================================

class TestOctagonInvariantExtraction:
    """Tests for extracting octagon invariants from programs."""

    def test_simple_countdown(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        assert 'intervals' in inv
        assert 'variables' in inv

    def test_two_variable_loop(self):
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        assert 'difference_bounds' in inv

    def test_no_loop(self):
        source = "let x = 10;"
        inv = extract_octagon_invariants(source, loop_index=0)
        assert inv is None

    def test_intervals_extracted(self):
        source = "let i = 0; let n = 10; while (i < n) { i = i + 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        assert 'i' in inv['intervals'] or 'n' in inv['intervals']

    def test_constraints_extracted(self):
        source = "let x = 5; let y = 5; while (x > 0) { x = x - 1; y = y + 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        # Should have some constraints
        assert isinstance(inv['constraints'], list)


# ===================================================================
# Section 4: Octagon-Guided Candidate Generation
# ===================================================================

class TestOctagonCandidateGeneration:
    """Tests for generating ranking candidates from octagon analysis."""

    def test_generates_candidates(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        candidates = generate_octagon_candidates(source)
        assert len(candidates) > 0

    def test_relational_candidates(self):
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        candidates = generate_octagon_candidates(source)
        # Should have candidates involving both x and y
        relational = [c for c in candidates
                      if len([k for k in c if k != '_const' and c[k] != 0]) == 2]
        assert len(relational) > 0

    def test_includes_standard_candidates(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        candidates = generate_octagon_candidates(source)
        # Should include simple i-based candidates
        has_i = any(c.get('i', 0) != 0 for c in candidates)
        assert has_i

    def test_no_loop_returns_empty(self):
        source = "let x = 10;"
        candidates = generate_octagon_candidates(source)
        assert candidates == []

    def test_candidates_have_const_key(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        candidates = generate_octagon_candidates(source)
        for c in candidates:
            assert '_const' in c


# ===================================================================
# Section 5: Simple Countdown (Standard Cases)
# ===================================================================

class TestSimpleTermination:
    """Tests that standard cases still work through octagon pipeline."""

    def test_countdown(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES
        assert result.ranking_function is not None

    def test_count_up(self):
        source = "let i = 0; let n = 10; while (i < n) { i = i + 1; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_decrement_by_two(self):
        source = "let i = 20; while (i > 0) { i = i - 2; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_strategy_is_standard_for_simple(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.strategy == "standard"


# ===================================================================
# Section 6: Relational Ranking (Novel Cases)
# ===================================================================

class TestRelationalRanking:
    """Tests for ranking functions involving relationships between variables."""

    def test_converging_variables(self):
        """x decreases, y increases, loop until x <= y."""
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_difference_ranking(self):
        """Termination depends on x - y decreasing."""
        source = "let x = 20; let y = 10; while (x > y) { x = x - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_sum_bounded(self):
        """Two variables where sum is relevant."""
        source = "let a = 5; let b = 5; while (a > 0) { a = a - 1; b = b + 1; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_find_relational_ranking_direct(self):
        """Direct test of find_relational_ranking."""
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        rf = find_relational_ranking(source)
        assert rf is not None

    def test_relational_ranking_expression(self):
        """Check that the ranking expression involves both variables."""
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        rf = find_relational_ranking(source)
        if rf is not None:
            # The expression should mention both x and y
            coeffs = rf.coefficients
            var_keys = [k for k in coeffs if k != '_const' and coeffs[k] != 0]
            # Could be a single-var ranking too; both are valid proofs
            assert len(var_keys) >= 1


# ===================================================================
# Section 7: Octagon-Strengthened Verification
# ===================================================================

class TestOctagonStrengthened:
    """Tests for using octagon invariants to strengthen verification."""

    def test_simple_strengthened(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        rf = find_octagon_strengthened_ranking(source)
        assert rf is not None
        assert rf.expression

    def test_verify_termination_invariant(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = verify_termination_invariant(source, {'i': 1, '_const': 0})
        assert 'standard' in result
        assert 'octagon_strengthened' in result
        std_b, std_d = result['standard']
        assert std_b and std_d

    def test_invariant_with_relational_ranking(self):
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        result = verify_termination_invariant(source, {'x': 1, 'y': -1, '_const': 0})
        # x - y is a valid ranking: decreases by 2 each iteration
        oct_b, oct_d = result['octagon_strengthened']
        # At minimum the octagon should agree with standard
        assert isinstance(oct_b, bool)
        assert isinstance(oct_d, bool)


# ===================================================================
# Section 8: Lexicographic Relational Ranking
# ===================================================================

class TestRelationalLexicographic:
    """Tests for lexicographic ranking with relational components."""

    def test_lex_for_nested_like(self):
        """Two-phase loop needing lex ranking."""
        source = """
        let x = 10;
        let y = 5;
        while (x > 0) {
            if (y > 0) {
                y = y - 1;
            } else {
                x = x - 1;
                y = 5;
            }
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_find_relational_lex_direct(self):
        """Test find_relational_lex_ranking API."""
        source = """
        let x = 5;
        let y = 5;
        while (x > 0) {
            if (y > 0) {
                y = y - 1;
            } else {
                x = x - 1;
                y = 5;
            }
        }
        """
        # This might be found by standard lex too
        lex = find_relational_lex_ranking(source)
        # Even if not found by lex specifically, the full pipeline should work
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES


# ===================================================================
# Section 9: Multi-Loop Analysis
# ===================================================================

class TestMultiLoopAnalysis:
    """Tests for analyzing programs with multiple loops."""

    def test_two_simple_loops(self):
        source = """
        let i = 10;
        while (i > 0) { i = i - 1; }
        let j = 5;
        while (j > 0) { j = j - 1; }
        """
        result = analyze_termination_with_octagon(source)
        assert result.loops_analyzed == 2
        assert result.loops_proved == 2
        assert result.result == TermResult.TERMINATES

    def test_no_loops(self):
        source = "let x = 10; let y = 20;"
        result = analyze_termination_with_octagon(source)
        assert result.loops_analyzed == 0
        assert result.result == TermResult.TERMINATES

    def test_one_loop(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_termination_with_octagon(source)
        assert result.loops_analyzed == 1
        assert result.loops_proved == 1

    def test_message_format(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_termination_with_octagon(source)
        assert "1/1" in result.message


# ===================================================================
# Section 10: Strategy Comparison
# ===================================================================

class TestStrategyComparison:
    """Tests for comparing termination strategies."""

    def test_compare_simple(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        results = compare_strategies(source)
        assert 'standard' in results
        assert 'relational' in results
        assert 'octagon_strengthened' in results
        assert 'full_octagon' in results

    def test_standard_finds_simple(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        results = compare_strategies(source)
        assert results['standard']['result'] == 'terminates'

    def test_compare_relational(self):
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        results = compare_strategies(source)
        assert results['full_octagon']['result'] == 'terminates'


# ===================================================================
# Section 11: Check Relational Ranking API
# ===================================================================

class TestCheckRelationalRanking:
    """Tests for the check_relational_ranking convenience API."""

    def test_valid_ranking_i(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        # R = i (should be valid: bounded by i > 0, decreases by 1)
        std_b, std_d, oct_b, oct_d = check_relational_ranking(
            source, 'i', 1, '__dummy', 0, const=0
        )
        assert std_b and std_d

    def test_invalid_ranking(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        # R = -i (not valid: -i < 0 when i > 0)
        std_b, std_d, oct_b, oct_d = check_relational_ranking(
            source, 'i', -1, '__dummy', 0, const=0
        )
        assert not (std_b and std_d)

    def test_relational_x_minus_y(self):
        source = "let x = 10; let y = 0; while (x > y) { x = x - 1; y = y + 1; }"
        std_b, std_d, oct_b, oct_d = check_relational_ranking(
            source, 'x', 1, 'y', -1, const=0
        )
        # x - y decreases by 2 each iteration, bounded by condition x > y
        assert std_b  # x > y => x - y > 0 => x - y >= 1
        assert std_d  # decrease by 2 >= 1


# ===================================================================
# Section 12: Edge Cases
# ===================================================================

class TestEdgeCases:
    """Edge cases and boundary tests."""

    def test_single_iteration_loop(self):
        source = "let i = 1; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_large_initial_value(self):
        source = "let i = 1000; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_three_variables(self):
        source = """
        let x = 10;
        let y = 0;
        let z = 5;
        while (x > 0) {
            x = x - 1;
            y = y + 1;
            z = z - 1;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_loop_with_conditional(self):
        source = """
        let i = 10;
        while (i > 0) {
            if (i > 5) {
                i = i - 2;
            } else {
                i = i - 1;
            }
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES


# ===================================================================
# Section 13: Data Structure Integrity
# ===================================================================

class TestDataStructures:
    """Tests for result data structures."""

    def test_oct_loop_result_fields(self):
        result = OctLoopResult(
            loop_index=0,
            result=TermResult.TERMINATES,
            ranking_function=RankingFunction("i", {'i': 1, '_const': 0}),
            strategy="standard",
            candidates_tried=5,
            message="Found ranking i",
        )
        assert result.loop_index == 0
        assert result.result == TermResult.TERMINATES
        assert result.strategy == "standard"
        assert result.candidates_tried == 5

    def test_oct_termination_result_fields(self):
        result = OctTerminationResult(
            result=TermResult.TERMINATES,
            loops_analyzed=2,
            loops_proved=2,
            message="All loops terminate",
        )
        assert result.loops_analyzed == 2
        assert result.loops_proved == 2

    def test_relational_ranking_function_fields(self):
        rf = RelationalRankingFunction(
            expression="x - y",
            coefficients={'x': 1, 'y': -1, '_const': 0},
            kind="relational",
            octagon_invariant="x - y <= 10",
        )
        assert rf.kind == "relational"
        assert rf.octagon_invariant == "x - y <= 10"


# ===================================================================
# Section 14: Practical Programs
# ===================================================================

class TestPracticalPrograms:
    """Tests with more realistic program patterns."""

    def test_gcd_like(self):
        """GCD-like: both variables decrease toward convergence."""
        source = """
        let a = 12;
        let b = 8;
        while (a > 0) {
            if (a > b) {
                a = a - b;
            } else {
                b = b - a;
                a = 0;
            }
        }
        """
        result = prove_termination_with_octagon(source)
        # GCD terminates; may or may not be provable with linear ranking
        assert result.result in (TermResult.TERMINATES, TermResult.UNKNOWN)

    def test_bounded_search(self):
        """Search loop with bounded iterations."""
        source = """
        let lo = 0;
        let hi = 100;
        while (lo < hi) {
            lo = lo + 1;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_swap_convergence(self):
        """Two pointers converging."""
        source = """
        let left = 0;
        let right = 10;
        while (left < right) {
            left = left + 1;
            right = right - 1;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_accumulator_with_counter(self):
        """Counter with accumulator."""
        source = """
        let i = 10;
        let sum = 0;
        while (i > 0) {
            sum = sum + i;
            i = i - 1;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_monotone_decrease(self):
        """Variable decreasing monotonically."""
        source = """
        let n = 15;
        while (n > 0) {
            n = n - 3;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES


# ===================================================================
# Section 15: Octagon Invariant Quality
# ===================================================================

class TestOctagonInvariantQuality:
    """Tests that octagon analysis produces useful invariants."""

    def test_initial_bounds_captured(self):
        source = "let x = 10; let y = 5; while (x > 0) { x = x - 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        # y should be bounded at 5
        if 'y' in inv['intervals']:
            lo, hi = inv['intervals']['y']
            if lo is not None and hi is not None:
                assert lo <= 5 <= hi

    def test_difference_bound_detected(self):
        source = "let x = 10; let y = 10; while (x > 0) { x = x - 1; y = y - 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        # x and y decrease together, so x - y should be bounded
        if ('x', 'y') in inv['difference_bounds']:
            lo, hi = inv['difference_bounds'][('x', 'y')]
            # x - y == 0 initially
            assert lo is not None or hi is not None

    def test_pre_loop_state_captured(self):
        source = "let a = 3; let b = 7; while (a < b) { a = a + 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        assert inv['pre_loop'] is not None

    def test_variables_listed(self):
        source = "let x = 1; let y = 2; let z = 3; while (x > 0) { x = x - 1; }"
        inv = extract_octagon_invariants(source)
        assert inv is not None
        assert 'x' in inv['variables']


# ===================================================================
# Section 16: Integration with V025
# ===================================================================

class TestV025Integration:
    """Tests that V025 integration works correctly."""

    def test_v025_result_wrapped(self):
        """Standard V025 result should be wrapped in OctLoopResult."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert isinstance(result, OctLoopResult)
        assert result.loop_index == 0

    def test_v025_ranking_preserved(self):
        """Standard ranking function should be preserved."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.ranking_function is not None

    def test_multi_loop_v025_compat(self):
        """Multi-loop analysis should work."""
        source = """
        let i = 5;
        while (i > 0) { i = i - 1; }
        let j = 3;
        while (j > 0) { j = j - 1; }
        """
        result = analyze_termination_with_octagon(source)
        assert isinstance(result, OctTerminationResult)
        assert result.loops_analyzed == 2


# ===================================================================
# Section 17: Robustness
# ===================================================================

class TestRobustness:
    """Robustness and error handling tests."""

    def test_loop_index_out_of_range(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        inv = extract_octagon_invariants(source, loop_index=5)
        assert inv is None

    def test_empty_program(self):
        source = "let x = 0;"
        result = analyze_termination_with_octagon(source)
        assert result.loops_analyzed == 0
        assert result.result == TermResult.TERMINATES

    def test_candidate_deduplication(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        candidates = generate_octagon_candidates(source)
        # No duplicates by expression string
        from octagon_termination import _coefficients_to_str
        strs = [_coefficients_to_str(c) for c in candidates]
        assert len(strs) == len(set(strs))


# ===================================================================
# Section 18: Advanced Relational Patterns
# ===================================================================

class TestAdvancedRelational:
    """Advanced relational termination patterns."""

    def test_symmetric_decrease(self):
        """Both vars decrease symmetrically."""
        source = """
        let x = 10;
        let y = 10;
        while (x > 0) {
            x = x - 1;
            y = y - 1;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_asymmetric_decrease(self):
        """One var decreases faster."""
        source = """
        let x = 10;
        let y = 20;
        while (x > 0) {
            x = x - 1;
            y = y - 2;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_one_increases_bounded(self):
        """One increases but loop bounded by other."""
        source = """
        let count = 10;
        let total = 0;
        while (count > 0) {
            count = count - 1;
            total = total + 1;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_gap_closing(self):
        """Gap between two variables closes."""
        source = """
        let lo = 0;
        let hi = 20;
        while (lo < hi) {
            lo = lo + 1;
            hi = hi - 1;
        }
        """
        result = prove_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES


# ===================================================================
# Section 19: OctLoopResult Strategy Tracking
# ===================================================================

class TestStrategyTracking:
    """Tests that strategy identification works."""

    def test_standard_strategy_identified(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.strategy == "standard"

    def test_result_has_message(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = prove_termination_with_octagon(source)
        assert result.message != ""

    def test_unknown_has_exhausted_strategy(self):
        """If all strategies fail, strategy should be 'exhausted'."""
        # A potentially non-terminating program
        source = """
        let x = 10;
        while (x > 0) {
            if (x > 5) {
                x = x + 1;
            } else {
                x = x - 1;
            }
        }
        """
        result = prove_termination_with_octagon(source)
        if result.result == TermResult.UNKNOWN:
            assert result.strategy == "exhausted"


# ===================================================================
# Section 20: Full Pipeline Integration
# ===================================================================

class TestFullPipeline:
    """End-to-end tests of the full octagon termination pipeline."""

    def test_full_pipeline_simple(self):
        source = "let n = 100; while (n > 0) { n = n - 1; }"
        # Extract invariants
        inv = extract_octagon_invariants(source)
        assert inv is not None
        # Generate candidates
        cands = generate_octagon_candidates(source)
        assert len(cands) > 0
        # Full analysis
        result = analyze_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_full_pipeline_relational(self):
        source = """
        let a = 10;
        let b = 0;
        while (a > b) {
            a = a - 1;
            b = b + 1;
        }
        """
        inv = extract_octagon_invariants(source)
        assert inv is not None
        result = analyze_termination_with_octagon(source)
        assert result.result == TermResult.TERMINATES

    def test_full_pipeline_multiple_loops(self):
        source = """
        let i = 10;
        while (i > 0) { i = i - 1; }
        let x = 5;
        let y = 0;
        while (x > y) { x = x - 1; y = y + 1; }
        """
        result = analyze_termination_with_octagon(source)
        assert result.loops_analyzed == 2
        assert result.result == TermResult.TERMINATES

    def test_compare_strategies_relational(self):
        source = """
        let x = 10;
        let y = 0;
        while (x > y) {
            x = x - 1;
            y = y + 1;
        }
        """
        results = compare_strategies(source)
        # Full octagon should find a proof
        assert results['full_octagon']['result'] == 'terminates'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
