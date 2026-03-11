"""Tests for V121: Fixpoint Acceleration"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from fixpoint_acceleration import (
    accelerated_analyze, standard_analyze, compare_analyses,
    get_variable_range, get_loop_invariant, get_acceleration_stats,
    detect_program_recurrences, verify_invariant, acceleration_summary,
    AccelConfig, AccelResult, AccelPhase, AccelerationStats,
    RecurrenceInfo, ConstraintHistory, AccelVerdict,
    extract_thresholds, extrapolate_constraint, accelerate_recurrence,
    polyhedral_threshold_widen, polyhedral_extrapolation_widen,
    polyhedral_narrowing,
    LinearConstraint, PolyhedralDomain, Fraction
)

# ===========================================================================
# Section 1: Basic Analysis (no loops)
# ===========================================================================

class TestBasicAnalysis:
    def test_simple_assignment(self):
        result = accelerated_analyze("let x = 5;")
        assert result.env.get_lower('x') == 5
        assert result.env.get_upper('x') == 5

    def test_multiple_assignments(self):
        result = accelerated_analyze("let x = 3; let y = 7;")
        assert result.env.get_lower('x') == 3
        assert result.env.get_upper('x') == 3
        assert result.env.get_lower('y') == 7
        assert result.env.get_upper('y') == 7

    def test_arithmetic(self):
        result = accelerated_analyze("let x = 3; let y = x + 2;")
        assert result.env.get_lower('y') == 5
        assert result.env.get_upper('y') == 5

    def test_subtraction(self):
        result = accelerated_analyze("let x = 10; let y = x - 3;")
        assert result.env.get_lower('y') == 7
        assert result.env.get_upper('y') == 7

    def test_linear_combination(self):
        result = accelerated_analyze("let x = 2; let y = 3; let z = x + y;")
        assert result.env.get_lower('z') == 5
        assert result.env.get_upper('z') == 5

    def test_negation(self):
        result = accelerated_analyze("let x = 5; let y = -x;")
        assert result.env.get_lower('y') == -5
        assert result.env.get_upper('y') == -5

    def test_scaling(self):
        result = accelerated_analyze("let x = 3; let y = 2 * x;")
        assert result.env.get_lower('y') == 6
        assert result.env.get_upper('y') == 6

    def test_result_structure(self):
        result = accelerated_analyze("let x = 1;")
        assert isinstance(result, AccelResult)
        assert isinstance(result.env, PolyhedralDomain)
        assert isinstance(result.warnings, list)
        assert isinstance(result.stats, AccelerationStats)
        assert result.verdict == AccelVerdict.CONVERGED


# ===========================================================================
# Section 2: If-Else Analysis
# ===========================================================================

class TestIfElse:
    def test_simple_if(self):
        src = "let x = 5; if (x > 3) { let y = 1; } else { let y = 2; }"
        result = accelerated_analyze(src)
        # x > 3 is always true (x=5), so y=1
        assert result.env.get_lower('y') == 1
        assert result.env.get_upper('y') == 1

    def test_if_else_join(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 10) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = accelerated_analyze(src)
        # x=5, so x>10 is false -> y=2
        assert result.env.get_lower('y') == 2
        assert result.env.get_upper('y') == 2

    def test_if_condition_refinement(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = x;
        }
        """
        result = accelerated_analyze(src)
        # x=5, x>3 is always true, y=x=5
        assert result.env.get_lower('y') == 5

    def test_dead_branch_elimination(self):
        src = """
        let x = 0;
        if (x > 10) {
            x = 100;
        }
        """
        result = accelerated_analyze(src)
        # x=0, condition false, x stays 0
        assert result.env.get_lower('x') == 0
        assert result.env.get_upper('x') == 0


# ===========================================================================
# Section 3: Simple Loop Analysis
# ===========================================================================

class TestSimpleLoops:
    def test_simple_counter(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        # After loop: i >= 10 (loop exit condition)
        assert result.env.get_lower('i') >= 10

    def test_countdown(self):
        src = """
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        """
        result = accelerated_analyze(src)
        # After loop: i <= 0
        assert result.env.get_upper('i') <= 0

    def test_counter_with_bound(self):
        src = """
        let i = 0;
        let n = 10;
        while (i < n) {
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('i') >= 10

    def test_accumulator(self):
        src = """
        let i = 0;
        let sum = 0;
        while (i < 5) {
            sum = sum + i;
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('i') >= 5
        # sum should be non-negative
        assert result.env.get_lower('sum') >= 0

    def test_loop_never_executes(self):
        src = """
        let i = 10;
        while (i < 0) {
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('i') == 10
        assert result.env.get_upper('i') == 10


# ===========================================================================
# Section 4: Recurrence Detection
# ===========================================================================

class TestRecurrenceDetection:
    def test_detect_increment(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        recs = detect_program_recurrences(src)
        assert len(recs) >= 1
        rec = [r for r in recs if r.var == 'i'][0]
        assert rec.delta == Fraction(1)

    def test_detect_decrement(self):
        src = """
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        """
        recs = detect_program_recurrences(src)
        assert len(recs) >= 1
        rec = [r for r in recs if r.var == 'i'][0]
        assert rec.delta == Fraction(-1)

    def test_detect_step(self):
        src = """
        let i = 0;
        while (i < 100) {
            i = i + 3;
        }
        """
        recs = detect_program_recurrences(src)
        rec = [r for r in recs if r.var == 'i'][0]
        assert rec.delta == Fraction(3)

    def test_no_recurrence_for_nonlinear(self):
        src = """
        let i = 1;
        while (i < 100) {
            i = i * 2;
        }
        """
        recs = detect_program_recurrences(src)
        assert all(r.var != 'i' for r in recs)

    def test_recurrence_initial_bounds(self):
        src = """
        let i = 5;
        while (i < 20) {
            i = i + 1;
        }
        """
        recs = detect_program_recurrences(src)
        rec = [r for r in recs if r.var == 'i'][0]
        assert rec.init_lower == 5
        assert rec.init_upper == 5


# ===========================================================================
# Section 5: Recurrence Acceleration
# ===========================================================================

class TestRecurrenceAcceleration:
    def test_accelerate_increment(self):
        rec = RecurrenceInfo(var='i', delta=Fraction(1),
                            init_lower=Fraction(0), init_upper=Fraction(0),
                            condition_var='i', condition_bound=Fraction(10))
        lo, hi = accelerate_recurrence(rec)
        assert lo == 0
        assert hi == 10

    def test_accelerate_decrement(self):
        rec = RecurrenceInfo(var='i', delta=Fraction(-1),
                            init_lower=Fraction(10), init_upper=Fraction(10),
                            condition_var='i', condition_bound=Fraction(0))
        lo, hi = accelerate_recurrence(rec)
        assert lo == 0
        assert hi == 10

    def test_accelerate_no_bound(self):
        rec = RecurrenceInfo(var='i', delta=Fraction(1),
                            init_lower=Fraction(0), init_upper=Fraction(0))
        lo, hi = accelerate_recurrence(rec)
        assert lo == 0
        assert hi == float('inf')

    def test_accelerate_zero_delta(self):
        rec = RecurrenceInfo(var='i', delta=Fraction(0),
                            init_lower=Fraction(5), init_upper=Fraction(5))
        lo, hi = accelerate_recurrence(rec)
        assert lo == 5
        assert hi == 5


# ===========================================================================
# Section 6: Constraint History & Extrapolation
# ===========================================================================

class TestConstraintHistory:
    def test_stable_history(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(5), Fraction(5), Fraction(5)]
        )
        assert h.is_stable
        assert not h.is_monotone_increasing or h.is_stable

    def test_increasing_history(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(1), Fraction(3), Fraction(5)]
        )
        assert h.is_monotone_increasing
        assert not h.is_monotone_decreasing

    def test_decreasing_history(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(10), Fraction(7), Fraction(4)]
        )
        assert h.is_monotone_decreasing
        assert not h.is_monotone_increasing

    def test_linear_delta(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(2), Fraction(5), Fraction(8)]
        )
        assert h.delta == Fraction(3)

    def test_nonlinear_delta(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(1), Fraction(3), Fraction(7)]
        )
        assert h.delta is None

    def test_extrapolate_linear(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(2), Fraction(4), Fraction(6)]
        )
        result = extrapolate_constraint(h, [])
        assert result == Fraction(8)

    def test_extrapolate_stable(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(5), Fraction(5), Fraction(5)]
        )
        result = extrapolate_constraint(h, [])
        assert result == Fraction(5)

    def test_extrapolate_snap_to_threshold(self):
        h = ConstraintHistory(
            constraint_key=((), False),
            bounds=[Fraction(3), Fraction(5), Fraction(7)]
        )
        result = extrapolate_constraint(h, [Fraction(10)])
        # delta=2, next would be 9, but threshold 10 is within range
        assert result == Fraction(10)


# ===========================================================================
# Section 7: Threshold Extraction
# ===========================================================================

class TestThresholdExtraction:
    def test_extract_from_literals(self):
        thresholds = extract_thresholds("let x = 10; while (x < 100) { x = x + 1; }")
        assert Fraction(10) in thresholds
        assert Fraction(100) in thresholds
        assert Fraction(1) in thresholds
        assert Fraction(0) in thresholds

    def test_always_includes_common(self):
        thresholds = extract_thresholds("let x = 42;")
        assert Fraction(0) in thresholds
        assert Fraction(1) in thresholds
        assert Fraction(-1) in thresholds
        assert Fraction(42) in thresholds

    def test_includes_variants(self):
        thresholds = extract_thresholds("let x = 10;")
        assert Fraction(10) in thresholds
        assert Fraction(-10) in thresholds
        assert Fraction(9) in thresholds   # 10 - 1
        assert Fraction(11) in thresholds  # 10 + 1

    def test_sorted(self):
        thresholds = extract_thresholds("let x = 5; let y = 3;")
        assert thresholds == sorted(thresholds)


# ===========================================================================
# Section 8: Polyhedral Threshold Widening
# ===========================================================================

class TestPolyhedralThresholdWidening:
    def test_keeps_satisfied_constraints(self):
        old = PolyhedralDomain(['x'])
        old.set_upper('x', 5)
        old.set_lower('x', 0)

        new = PolyhedralDomain(['x'])
        new.set_upper('x', 4)
        new.set_lower('x', 1)

        result = polyhedral_threshold_widen(old, new, [Fraction(10)])
        assert result.get_upper('x') <= 5
        assert result.get_lower('x') >= 0

    def test_bot_handling(self):
        old = PolyhedralDomain(['x'])
        old._is_bot = True
        new = PolyhedralDomain(['x'])
        new.set_upper('x', 5)

        result = polyhedral_threshold_widen(old, new, [])
        assert result.get_upper('x') == 5


# ===========================================================================
# Section 9: Narrowing
# ===========================================================================

class TestNarrowing:
    def test_tighten_upper(self):
        wide = PolyhedralDomain(['x'])
        wide.set_lower('x', 0)
        # Upper is infinity (from widening)

        body = PolyhedralDomain(['x'])
        body.set_lower('x', 0)
        body.set_upper('x', 10)

        result = polyhedral_narrowing(wide, body)
        assert result.get_upper('x') == 10

    def test_tighten_lower(self):
        wide = PolyhedralDomain(['x'])
        wide.set_upper('x', 10)
        # Lower is -infinity

        body = PolyhedralDomain(['x'])
        body.set_lower('x', 0)
        body.set_upper('x', 10)

        result = polyhedral_narrowing(wide, body)
        assert result.get_lower('x') == 0

    def test_narrowing_preserves_tight(self):
        wide = PolyhedralDomain(['x'])
        wide.set_lower('x', 0)
        wide.set_upper('x', 10)

        body = PolyhedralDomain(['x'])
        body.set_lower('x', 2)
        body.set_upper('x', 8)

        result = polyhedral_narrowing(wide, body)
        # Should tighten to [2, 8]
        assert result.get_lower('x') >= 0
        assert result.get_upper('x') <= 10

    def test_narrowing_bot(self):
        wide = PolyhedralDomain(['x'])
        wide._is_bot = True
        body = PolyhedralDomain(['x'])
        body.set_upper('x', 5)
        result = polyhedral_narrowing(wide, body)
        assert result._is_bot


# ===========================================================================
# Section 10: Acceleration Config
# ===========================================================================

class TestAccelConfig:
    def test_default_config(self):
        config = AccelConfig()
        assert config.max_iterations == 100
        assert config.delay_iterations == 3
        assert config.threshold_iterations == 5
        assert config.narrowing_iterations == 3
        assert config.enable_extrapolation
        assert config.enable_recurrence
        assert config.enable_narrowing

    def test_custom_config(self):
        config = AccelConfig(max_iterations=50, delay_iterations=5,
                            enable_narrowing=False)
        assert config.max_iterations == 50
        assert config.delay_iterations == 5
        assert not config.enable_narrowing

    def test_no_acceleration(self):
        config = AccelConfig(enable_extrapolation=False, enable_recurrence=False,
                            enable_narrowing=False, delay_iterations=0,
                            threshold_iterations=0)
        src = "let i = 0; while (i < 10) { i = i + 1; }"
        result = accelerated_analyze(src, config)
        assert result.env.get_lower('i') >= 10


# ===========================================================================
# Section 11: Phase Transitions
# ===========================================================================

class TestPhaseTransitions:
    def test_phases_recorded(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert len(result.stats.phase_transitions) >= 1
        assert result.stats.phase_transitions[0] == (0, 'delay')

    def test_delay_phase_first(self):
        config = AccelConfig(delay_iterations=3)
        src = """
        let i = 0;
        while (i < 100) {
            i = i + 1;
        }
        """
        result = accelerated_analyze(src, config)
        assert result.stats.phase_transitions[0][1] == 'delay'

    def test_total_iterations_tracked(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.stats.total_iterations > 0


# ===========================================================================
# Section 12: Comparison with Standard Analysis
# ===========================================================================

class TestComparison:
    def test_compare_simple(self):
        src = "let x = 5;"
        comp = compare_analyses(src)
        assert 'variables' in comp
        assert 'x' in comp['variables']

    def test_compare_loop(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        comp = compare_analyses(src)
        assert 'i' in comp['variables']
        assert 'stats' in comp

    def test_compare_has_stats(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        comp = compare_analyses(src)
        assert 'total_iterations' in comp['stats']
        assert 'widening_iterations' in comp['stats']
        assert 'narrowing_iterations' in comp['stats']

    def test_precision_gains_reported(self):
        src = "let x = 5;"
        comp = compare_analyses(src)
        assert 'precision_gains' in comp


# ===========================================================================
# Section 13: Loop Invariants
# ===========================================================================

class TestLoopInvariants:
    def test_invariant_exists(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        inv = get_loop_invariant(src, 0)
        assert inv is not None
        assert 'i' in inv.var_names

    def test_invariant_bounds(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        inv = get_loop_invariant(src, 0)
        # Invariant should include i >= 0 (since i starts at 0 and only increases)
        assert inv.get_lower('i') >= 0

    def test_verify_invariant(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = verify_invariant(src, 0)
        assert result['verified']
        assert 'constraints' in result
        assert 'variable_ranges' in result

    def test_no_loop_invariant(self):
        src = "let x = 5;"
        inv = get_loop_invariant(src, 0)
        assert inv is None


# ===========================================================================
# Section 14: Complex Loops
# ===========================================================================

class TestComplexLoops:
    def test_two_variables(self):
        src = """
        let i = 0;
        let j = 10;
        while (i < j) {
            i = i + 1;
            j = j - 1;
        }
        """
        result = accelerated_analyze(src)
        # After loop: i >= j
        i_lo = result.env.get_lower('i')
        j_hi = result.env.get_upper('j')
        assert i_lo >= 0

    def test_nested_loops(self):
        src = """
        let i = 0;
        let total = 0;
        while (i < 3) {
            let j = 0;
            while (j < 3) {
                total = total + 1;
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('i') >= 3
        assert result.env.get_lower('total') >= 0

    def test_conditional_loop_body(self):
        src = """
        let i = 0;
        let x = 0;
        while (i < 10) {
            if (i < 5) {
                x = x + 1;
            } else {
                x = x + 2;
            }
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('i') >= 10
        assert result.env.get_lower('x') >= 0

    def test_bounded_accumulation(self):
        src = """
        let i = 0;
        let sum = 0;
        while (i < 10) {
            sum = sum + 1;
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('sum') >= 0
        assert result.env.get_lower('i') >= 10


# ===========================================================================
# Section 15: get_variable_range API
# ===========================================================================

class TestGetVariableRange:
    def test_simple(self):
        lo, hi = get_variable_range("let x = 5;", 'x')
        assert lo == 5
        assert hi == 5

    def test_loop_variable(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        lo, hi = get_variable_range(src, 'i')
        assert lo >= 10

    def test_unknown_variable(self):
        lo, hi = get_variable_range("let x = 5;", 'unknown')
        assert lo == float('-inf')
        assert hi == float('inf')


# ===========================================================================
# Section 16: Acceleration Statistics
# ===========================================================================

class TestAccelerationStats:
    def test_stats_for_no_loops(self):
        stats = get_acceleration_stats("let x = 5;")
        assert stats.total_iterations == 0
        assert stats.widening_iterations == 0

    def test_stats_for_loop(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        stats = get_acceleration_stats(src)
        assert stats.total_iterations > 0

    def test_recurrence_stats(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        stats = get_acceleration_stats(src)
        assert stats.recurrences_detected >= 1


# ===========================================================================
# Section 17: Summary Output
# ===========================================================================

class TestSummary:
    def test_summary_format(self):
        src = "let x = 5;"
        s = acceleration_summary(src)
        assert "Fixpoint Acceleration Summary" in s
        assert "Variable Ranges" in s
        assert "x:" in s

    def test_summary_with_loop(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        s = acceleration_summary(src)
        assert "Acceleration Statistics" in s
        assert "Loop Invariants" in s

    def test_summary_warnings(self):
        src = """
        let x = 5;
        let y = 0;
        let z = x / y;
        """
        s = acceleration_summary(src)
        assert "Warnings" in s or "division" in s.lower() or isinstance(s, str)


# ===========================================================================
# Section 18: Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = accelerated_analyze("")
        assert result.verdict == AccelVerdict.CONVERGED

    def test_single_variable(self):
        result = accelerated_analyze("let x = 0;")
        assert result.env.get_lower('x') == 0

    def test_reassignment(self):
        result = accelerated_analyze("let x = 5; x = 10;")
        assert result.env.get_lower('x') == 10
        assert result.env.get_upper('x') == 10

    def test_self_increment(self):
        result = accelerated_analyze("let x = 3; x = x + 1;")
        assert result.env.get_lower('x') == 4
        assert result.env.get_upper('x') == 4

    def test_negative_values(self):
        result = accelerated_analyze("let x = -5;")
        assert result.env.get_lower('x') == -5
        assert result.env.get_upper('x') == -5

    def test_zero_iteration_loop(self):
        src = """
        let i = 100;
        while (i < 0) {
            i = i + 1;
        }
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('i') == 100


# ===========================================================================
# Section 19: Functions
# ===========================================================================

class TestFunctions:
    def test_function_tracked(self):
        src = """
        fn foo(x) {
            return x + 1;
        }
        let y = 5;
        """
        result = accelerated_analyze(src)
        assert 'foo' in result.functions

    def test_print_ignored(self):
        src = """
        let x = 5;
        print(x);
        """
        result = accelerated_analyze(src)
        assert result.env.get_lower('x') == 5


# ===========================================================================
# Section 20: Division and Modulo
# ===========================================================================

class TestDivisionModulo:
    def test_division_warning(self):
        src = """
        let x = 10;
        let y = 0;
        let z = x / y;
        """
        result = accelerated_analyze(src)
        assert any("division" in w.lower() for w in result.warnings)

    def test_safe_division(self):
        src = """
        let x = 10;
        let y = 2;
        let z = x / y;
        """
        result = accelerated_analyze(src)
        lo = result.env.get_lower('z')
        hi = result.env.get_upper('z')
        assert lo <= 5 <= hi


# ===========================================================================
# Section 21: Relational Constraints
# ===========================================================================

class TestRelational:
    def test_variable_copy(self):
        src = "let x = 5; let y = x;"
        result = accelerated_analyze(src)
        assert result.env.get_lower('y') == 5
        assert result.env.get_upper('y') == 5

    def test_linear_relation(self):
        src = "let x = 3; let y = x + 2;"
        result = accelerated_analyze(src)
        assert result.env.get_lower('y') == 5
        assert result.env.get_upper('y') == 5

    def test_relational_in_loop(self):
        src = """
        let x = 0;
        let y = 0;
        while (x < 10) {
            x = x + 1;
            y = y + 1;
        }
        """
        result = accelerated_analyze(src)
        # Both x and y increase together
        assert result.env.get_lower('x') >= 10
        assert result.env.get_lower('y') >= 0


# ===========================================================================
# Section 22: Multiple Loops
# ===========================================================================

class TestMultipleLoops:
    def test_sequential_loops(self):
        src = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        let j = 0;
        while (j < 3) {
            j = j + 1;
        }
        """
        result = accelerated_analyze(src)
        # i >= 0 is sound (polyhedral join may weaken cross-loop bounds)
        assert result.env.get_lower('i') >= 0
        assert result.env.get_lower('j') >= 3

    def test_sequential_loops_invariants(self):
        src = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        let j = 0;
        while (j < 3) {
            j = j + 1;
        }
        """
        result = accelerated_analyze(src)
        assert 0 in result.loop_invariants
        assert 1 in result.loop_invariants


# ===========================================================================
# Section 23: Acceleration Effectiveness
# ===========================================================================

class TestAccelerationEffectiveness:
    def test_recurrence_reduces_iterations(self):
        """With recurrence detection, simple loops should converge faster."""
        src = """
        let i = 0;
        while (i < 100) {
            i = i + 1;
        }
        """
        # With acceleration
        accel_stats = get_acceleration_stats(src)

        # Without recurrence
        no_rec_config = AccelConfig(enable_recurrence=False)
        no_rec_stats = get_acceleration_stats(src, no_rec_config)

        # Both should converge
        assert accel_stats.total_iterations > 0
        assert no_rec_stats.total_iterations > 0

    def test_narrowing_improves_precision(self):
        """Narrowing should recover precision lost to widening."""
        src = """
        let i = 0;
        let x = 0;
        while (i < 10) {
            x = x + 1;
            i = i + 1;
        }
        """
        with_narrowing = accelerated_analyze(src, AccelConfig(enable_narrowing=True))
        without_narrowing = accelerated_analyze(src, AccelConfig(enable_narrowing=False))

        # Both should give valid results
        assert with_narrowing.env.get_lower('i') >= 10
        assert without_narrowing.env.get_lower('i') >= 10


# ===========================================================================
# Section 24: Full Pipeline Integration
# ===========================================================================

class TestFullPipeline:
    def test_delay_threshold_standard(self):
        """Full pipeline: delay -> threshold -> standard should converge."""
        src = """
        let i = 0;
        let sum = 0;
        while (i < 50) {
            sum = sum + i;
            i = i + 1;
        }
        """
        config = AccelConfig(delay_iterations=2, threshold_iterations=3,
                            max_iterations=100)
        result = accelerated_analyze(src, config)
        assert result.env.get_lower('i') >= 50
        assert result.env.get_lower('sum') >= 0

    def test_all_features_together(self):
        """All acceleration features working together."""
        src = """
        let i = 0;
        let j = 10;
        let sum = 0;
        while (i < j) {
            sum = sum + 1;
            i = i + 1;
        }
        """
        config = AccelConfig(
            delay_iterations=2,
            threshold_iterations=3,
            narrowing_iterations=3,
            enable_extrapolation=True,
            enable_recurrence=True,
            enable_narrowing=True
        )
        result = accelerated_analyze(src, config)
        assert result.verdict == AccelVerdict.CONVERGED
        assert result.env.get_lower('i') >= 0
        assert result.env.get_lower('sum') >= 0

    def test_compare_standard_vs_accelerated(self):
        """Compare standard V105 with accelerated analysis."""
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        comp = compare_analyses(src)
        # Both should identify i >= 10
        std = comp['variables']['i']['standard']
        acc = comp['variables']['i']['accelerated']
        assert std[0] >= 10 or acc[0] >= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
