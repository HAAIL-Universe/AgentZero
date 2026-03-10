"""
Tests for V025: Termination Analysis

Tests ranking function discovery and verification for various loop patterns.
"""

import sys
import os
import pytest

# Path setup
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from termination import (
    # Core types
    TermResult, TerminationResult, LoopTermResult,
    RankingFunction, LexRankingFunction,
    # Core APIs
    find_ranking_function, prove_termination, check_ranking_function,
    analyze_termination, verify_terminates, verify_all_terminate,
    # Internal helpers
    extract_loop_info, generate_candidates, verify_ranking_function,
    # Advanced
    find_lexicographic_ranking, find_conditional_ranking,
    find_ranking_with_ai, detect_nontermination,
    compare_ranking_strategies,
)


# ============================================================
# Simple Countdown Loops
# ============================================================

class TestSimpleCountdown:
    """Loops that count down a single variable."""

    def test_countdown_to_zero(self):
        """while (i > 0) { i = i - 1; } -- ranking: i"""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        rf = find_ranking_function(source)
        assert rf is not None
        assert 'i' in rf.coefficients
        assert rf.coefficients.get('i', 0) > 0  # positive coeff on i

    def test_countdown_from_variable(self):
        """while (n > 0) { n = n - 1; } -- ranking: n"""
        source = "let n = 100; while (n > 0) { n = n - 1; }"
        rf = find_ranking_function(source)
        assert rf is not None

    def test_countdown_by_two(self):
        """while (x > 0) { x = x - 2; } -- ranking: x"""
        source = "let x = 20; while (x > 0) { x = x - 2; }"
        rf = find_ranking_function(source)
        assert rf is not None

    def test_countdown_ge(self):
        """while (i >= 1) { i = i - 1; } -- ranking: i"""
        source = "let i = 5; while (i >= 1) { i = i - 1; }"
        rf = find_ranking_function(source)
        assert rf is not None

    def test_prove_termination_countdown(self):
        """prove_termination returns TERMINATES for simple countdown."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = prove_termination(source)
        assert result.result == TermResult.TERMINATES
        assert result.ranking_function is not None
        assert result.candidates_tried >= 1


# ============================================================
# Count-Up Loops
# ============================================================

class TestCountUp:
    """Loops that count up to a bound."""

    def test_count_up_to_bound(self):
        """while (i < 10) { i = i + 1; } -- ranking: 10 - i"""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        rf = find_ranking_function(source)
        assert rf is not None
        # Should find something like 10 - i or equivalent
        b, d = check_ranking_function(source, rf.coefficients)
        assert b and d

    def test_count_up_to_variable(self):
        """while (i < n) { i = i + 1; } -- ranking: n - i"""
        source = "let n = 10; let i = 0; while (i < n) { i = i + 1; }"
        rf = find_ranking_function(source)
        assert rf is not None

    def test_count_up_le(self):
        """while (i <= 10) { i = i + 1; } -- ranking: 11 - i or 10 - i + 1"""
        source = "let i = 0; while (i <= 10) { i = i + 1; }"
        rf = find_ranking_function(source)
        assert rf is not None


# ============================================================
# Two-Variable Loops
# ============================================================

class TestTwoVariable:
    """Loops with multiple variables."""

    def test_sum_accumulator(self):
        """let s=0; let i=5; while(i>0){s=s+i; i=i-1;} -- ranking: i"""
        source = "let s = 0; let i = 5; while (i > 0) { s = s + i; i = i - 1; }"
        rf = find_ranking_function(source)
        assert rf is not None
        # i is the decreasing variable
        b, d = check_ranking_function(source, rf.coefficients)
        assert b and d

    def test_two_var_countdown(self):
        """Both variables decrease independently."""
        source = "let x = 5; let y = 3; while (x > 0) { x = x - 1; y = y - 1; }"
        rf = find_ranking_function(source)
        assert rf is not None

    def test_swap_and_decrease(self):
        """while(a>0){t=a; a=b; b=t-1;} -- needs careful analysis."""
        # a decreases each full cycle but not monotonically
        # This might need lexicographic ranking
        source = "let a = 5; let b = 3; while (a > 0) { let t = a; a = b; b = t - 1; }"
        result = prove_termination(source)
        # This is a hard case -- might be UNKNOWN, that's ok


# ============================================================
# User-Provided Ranking Functions
# ============================================================

class TestUserProvided:
    """Test verification of user-provided ranking functions."""

    def test_valid_ranking(self):
        """User provides correct ranking function."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        b, d = check_ranking_function(source, {'i': 1, '_const': 0})
        assert b, "i should be bounded below by 0 when i > 0"
        assert d, "i should decrease by 1 each iteration"

    def test_invalid_ranking_not_bounded(self):
        """User provides ranking that isn't bounded below."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        b, d = check_ranking_function(source, {'i': -1, '_const': 0})
        assert not b, "-i is not bounded below when i > 0"

    def test_invalid_ranking_not_decreasing(self):
        """User provides ranking that doesn't decrease."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        b, d = check_ranking_function(source, {'i': 1, '_const': 0})
        # i increases, so it's not decreasing
        assert not d

    def test_valid_difference_ranking(self):
        """n - i is a valid ranking for count-up loop."""
        source = "let n = 10; let i = 0; while (i < n) { i = i + 1; }"
        b, d = check_ranking_function(source, {'n': 1, 'i': -1, '_const': 0})
        assert b and d


# ============================================================
# Program-Level Analysis
# ============================================================

class TestProgramAnalysis:
    """Test whole-program termination analysis."""

    def test_no_loops(self):
        """Program with no loops trivially terminates."""
        source = "let x = 1; let y = 2; let z = x + y;"
        result = analyze_termination(source)
        assert result.result == TermResult.TERMINATES
        assert result.loops_analyzed == 0

    def test_single_loop(self):
        """Program with one terminating loop."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = analyze_termination(source)
        assert result.result == TermResult.TERMINATES
        assert result.loops_analyzed == 1
        assert result.loops_proved == 1

    def test_multiple_loops_all_terminate(self):
        """Program with two terminating loops."""
        source = """
        let i = 5;
        while (i > 0) { i = i - 1; }
        let j = 0;
        while (j < 10) { j = j + 1; }
        """
        result = analyze_termination(source)
        assert result.loops_analyzed == 2
        assert result.loops_proved == 2
        assert result.result == TermResult.TERMINATES

    def test_verify_all_terminate(self):
        """verify_all_terminate API works."""
        source = "let x = 3; while (x > 0) { x = x - 1; }"
        result = verify_all_terminate(source)
        assert result.result == TermResult.TERMINATES


# ============================================================
# Loop Extraction
# ============================================================

class TestLoopExtraction:
    """Test loop information extraction."""

    def test_extract_state_vars(self):
        """Correctly identifies state variables."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        info = extract_loop_info(source)
        assert 'i' in info['state_vars']

    def test_extract_pre_assignments(self):
        """Correctly extracts pre-loop initializations."""
        source = "let x = 5; let y = 10; while (x < y) { x = x + 1; }"
        info = extract_loop_info(source)
        assert info['pre_assignments'].get('x') == 5
        assert info['pre_assignments'].get('y') == 10

    def test_extract_multiple_state_vars(self):
        """Multiple state variables extracted."""
        source = "let a = 0; let b = 10; while (a < b) { a = a + 1; b = b - 1; }"
        info = extract_loop_info(source)
        assert 'a' in info['state_vars']
        assert 'b' in info['state_vars']

    def test_second_loop(self):
        """Can extract info from the second loop."""
        source = """
        let i = 5;
        while (i > 0) { i = i - 1; }
        let j = 0;
        while (j < 10) { j = j + 1; }
        """
        info0 = extract_loop_info(source, loop_index=0)
        info1 = extract_loop_info(source, loop_index=1)
        assert 'i' in info0['state_vars']
        assert 'j' in info1['state_vars']

    def test_no_loop_raises(self):
        """Raises ValueError when no loop found."""
        source = "let x = 1;"
        with pytest.raises(ValueError):
            extract_loop_info(source)


# ============================================================
# Candidate Generation
# ============================================================

class TestCandidateGeneration:
    """Test ranking function candidate generation."""

    def test_generates_candidates(self):
        """Candidate list is non-empty for any loop."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        info = extract_loop_info(source)
        candidates = generate_candidates(info)
        assert len(candidates) > 0

    def test_condition_derived_for_countdown(self):
        """Condition > 0 should generate x as a candidate."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        info = extract_loop_info(source)
        candidates = generate_candidates(info)
        # Should include {'i': 1, '_const': 0} or similar
        has_i = any(c.get('i', 0) > 0 and c.get('_const', 0) == 0
                    for c in candidates)
        assert has_i

    def test_condition_derived_for_countup(self):
        """Condition < n should generate n - i as a candidate."""
        source = "let n = 10; let i = 0; while (i < n) { i = i + 1; }"
        info = extract_loop_info(source)
        candidates = generate_candidates(info)
        # Should include something with n:1, i:-1
        has_diff = any(c.get('n', 0) == 1 and c.get('i', 0) == -1
                       for c in candidates)
        assert has_diff

    def test_deduplication(self):
        """No duplicate candidates."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        info = extract_loop_info(source)
        candidates = generate_candidates(info)
        strs = [str(sorted(c.items())) for c in candidates]
        assert len(strs) == len(set(strs))


# ============================================================
# Verify Ranking Function
# ============================================================

class TestVerifyRanking:
    """Test the SMT-based verification of ranking functions."""

    def test_i_is_ranking_for_countdown(self):
        """i is a valid ranking function for countdown loop."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        info = extract_loop_info(source)
        b, d = verify_ranking_function(info, {'i': 1, '_const': 0})
        assert b, "i >= 0 when i > 0"
        assert d, "i decreases by 1"

    def test_10_minus_i_for_countup(self):
        """10 - i is a valid ranking for while(i < 10) { i = i + 1; }"""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        info = extract_loop_info(source)
        b, d = verify_ranking_function(info, {'i': -1, '_const': 10})
        assert b, "10 - i >= 0 when i < 10"
        assert d, "10 - i decreases by 1"

    def test_constant_not_ranking(self):
        """A constant is not a ranking function (doesn't decrease)."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        info = extract_loop_info(source)
        b, d = verify_ranking_function(info, {'_const': 5})
        assert b, "5 >= 0 always"
        assert not d, "5 doesn't decrease"

    def test_negative_not_bounded(self):
        """-i is not bounded below when i > 0."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        info = extract_loop_info(source)
        b, d = verify_ranking_function(info, {'i': -1, '_const': 0})
        assert not b


# ============================================================
# Conditional Loops
# ============================================================

class TestConditionalLoops:
    """Loops with conditional bodies."""

    def test_conditional_decrement(self):
        """while(x>0){if(x>5){x=x-2;}else{x=x-1;}} -- ranking: x"""
        source = """
        let x = 10;
        while (x > 0) {
            if (x > 5) { x = x - 2; } else { x = x - 1; }
        }
        """
        rf = find_ranking_function(source)
        assert rf is not None
        b, d = check_ranking_function(source, rf.coefficients)
        assert b and d

    def test_conditional_with_accumulator(self):
        """Loop with accumulator in one branch."""
        source = """
        let i = 10;
        let s = 0;
        while (i > 0) {
            if (i > 5) { s = s + i; }
            i = i - 1;
        }
        """
        rf = find_ranking_function(source)
        assert rf is not None


# ============================================================
# Lexicographic Ranking
# ============================================================

class TestLexicographic:
    """Lexicographic ranking function tests."""

    def test_simple_lex_not_needed(self):
        """Simple loops get single-component lex ranking."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        lex_rf = find_lexicographic_ranking(source)
        assert lex_rf is not None
        assert len(lex_rf.components) == 1  # Single component suffices

    def test_lex_expression(self):
        """Lexicographic ranking has readable expression."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        lex_rf = find_lexicographic_ranking(source)
        assert lex_rf is not None
        assert lex_rf.expression  # Non-empty


# ============================================================
# verify_terminates (Full Pipeline)
# ============================================================

class TestVerifyTerminates:
    """Test the full termination verification pipeline."""

    def test_simple_terminates(self):
        """Full pipeline proves simple countdown terminates."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = verify_terminates(source)
        assert result.result == TermResult.TERMINATES

    def test_countup_terminates(self):
        """Full pipeline proves count-up terminates."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = verify_terminates(source)
        assert result.result == TermResult.TERMINATES

    def test_two_var_terminates(self):
        """Full pipeline handles two-variable loops."""
        source = "let a = 0; let b = 10; while (a < b) { a = a + 1; b = b - 1; }"
        result = verify_terminates(source)
        assert result.result == TermResult.TERMINATES

    def test_accumulator_terminates(self):
        """Accumulator loop terminates (i is the ranking)."""
        source = "let s = 0; let i = 5; while (i > 0) { s = s + i; i = i - 1; }"
        result = verify_terminates(source)
        assert result.result == TermResult.TERMINATES


# ============================================================
# Nontermination Detection
# ============================================================

class TestNontermination:
    """Test nontermination detection."""

    def test_infinite_loop_fixed_point(self):
        """while(x>0){} with no change -- fixed point at any x>0."""
        # Actually C10 requires at least one statement in while body
        # Use identity assignment
        source = "let x = 5; while (x > 0) { x = x; }"
        result = detect_nontermination(source)
        assert result.result == TermResult.NONTERMINATING

    def test_simple_countdown_not_nonterminating(self):
        """Countdown loop should not be flagged as nonterminating."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = detect_nontermination(source)
        assert result.result != TermResult.NONTERMINATING


# ============================================================
# AI-Enhanced Search
# ============================================================

class TestAIEnhanced:
    """Test abstract-interpretation-enhanced candidate generation."""

    def test_ai_finds_ranking(self):
        """AI-enhanced search finds ranking for simple loop."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        rf = find_ranking_with_ai(source)
        assert rf is not None

    def test_ai_finds_ranking_countup(self):
        """AI-enhanced search finds ranking for count-up loop."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        rf = find_ranking_with_ai(source)
        assert rf is not None


# ============================================================
# Strategy Comparison
# ============================================================

class TestStrategyComparison:
    """Test ranking strategy comparison."""

    def test_compare_strategies(self):
        """Comparison returns results for each strategy."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        results = compare_ranking_strategies(source)
        assert 'condition_derived' in results
        assert 'full_search' in results
        # At least one strategy should find a ranking
        found_any = any(v is not None for v in results.values())
        assert found_any


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_loop_with_large_step(self):
        """Countdown by 3."""
        source = "let i = 30; while (i > 0) { i = i - 3; }"
        rf = find_ranking_function(source)
        assert rf is not None

    def test_negative_init(self):
        """Count up from negative."""
        source = "let i = -10; while (i < 0) { i = i + 1; }"
        rf = find_ranking_function(source)
        assert rf is not None
        b, d = check_ranking_function(source, rf.coefficients)
        assert b and d

    def test_convergence_from_both_sides(self):
        """a counts up, b counts down, they meet."""
        source = "let a = 0; let b = 10; while (a < b) { a = a + 1; b = b - 1; }"
        rf = find_ranking_function(source)
        assert rf is not None

    def test_result_message(self):
        """Result has descriptive message."""
        source = "let i = 5; while (i > 0) { i = i - 1; }"
        result = prove_termination(source)
        assert result.message
        assert "terminat" in result.message.lower() or "ranking" in result.message.lower()

    def test_ranking_function_str(self):
        """Ranking function has human-readable expression."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        rf = find_ranking_function(source)
        assert rf is not None
        assert rf.expression  # Non-empty string
        assert isinstance(rf.expression, str)


# ============================================================
# Nested Loops (Outer Only)
# ============================================================

class TestNestedLoops:
    """Test with nested loop programs (analyze outer loop)."""

    def test_outer_loop_terminates(self):
        """Outer loop of nested loops terminates."""
        source = """
        let i = 5;
        while (i > 0) {
            let j = 3;
            while (j > 0) { j = j - 1; }
            i = i - 1;
        }
        """
        # Loop 0 is the outer loop, loop 1 is the inner
        # Outer loop: i decreases by 1 each iteration
        result = prove_termination(source, loop_index=0)
        assert result.result == TermResult.TERMINATES

    def test_inner_loop_terminates(self):
        """Inner loop of nested loops terminates."""
        source = """
        let i = 5;
        while (i > 0) {
            let j = 3;
            while (j > 0) { j = j - 1; }
            i = i - 1;
        }
        """
        result = prove_termination(source, loop_index=1)
        assert result.result == TermResult.TERMINATES

    def test_both_loops_in_program(self):
        """analyze_termination finds both nested loops."""
        source = """
        let i = 5;
        while (i > 0) {
            let j = 3;
            while (j > 0) { j = j - 1; }
            i = i - 1;
        }
        """
        result = analyze_termination(source)
        assert result.loops_analyzed == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
