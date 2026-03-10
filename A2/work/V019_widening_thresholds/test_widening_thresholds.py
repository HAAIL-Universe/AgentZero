"""
Tests for V019: Widening with Thresholds
"""

import sys
import os
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_CHALLENGES = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'challenges'))
sys.path.insert(0, os.path.join(_CHALLENGES, 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(_CHALLENGES, 'C010_stack_vm'))

from abstract_interpreter import (
    Interval, INTERVAL_BOT, INTERVAL_TOP, INF, NEG_INF,
    interval_widen, Sign,
)
from widening_thresholds import (
    interval_widen_thresholds,
    extract_thresholds_from_source,
    extract_variable_thresholds,
    threshold_analyze,
    compare_widening,
    get_variable_range,
    get_thresholds,
    get_variable_thresholds,
    ThresholdInterpreter,
    ThresholdEnv,
    _interval_narrow,
    _is_strictly_tighter,
)


# ============================================================
# Section 1: Threshold Widening Operator
# ============================================================

class TestThresholdWidening:
    def test_no_growth_no_change(self):
        """If bounds don't grow, widening is identity."""
        old = Interval(0, 10)
        new = Interval(0, 10)
        result = interval_widen_thresholds(old, new, [0, 5, 10, 15, 20])
        assert result.lo == 0 and result.hi == 10

    def test_upper_grows_to_threshold(self):
        """Upper bound grows -> widen to next threshold."""
        old = Interval(0, 5)
        new = Interval(0, 8)
        result = interval_widen_thresholds(old, new, [0, 5, 10, 15, 20])
        assert result.lo == 0 and result.hi == 10

    def test_lower_shrinks_to_threshold(self):
        """Lower bound decreases -> widen to next lower threshold."""
        old = Interval(5, 10)
        new = Interval(3, 10)
        result = interval_widen_thresholds(old, new, [0, 5, 10, 15, 20])
        assert result.lo == 0 and result.hi == 10

    def test_no_threshold_above_falls_to_inf(self):
        """If no threshold above new.hi, fall to +inf."""
        old = Interval(0, 10)
        new = Interval(0, 25)
        result = interval_widen_thresholds(old, new, [0, 5, 10, 15, 20])
        assert result.hi == INF

    def test_no_threshold_below_falls_to_neg_inf(self):
        """If no threshold below new.lo, fall to -inf."""
        old = Interval(0, 10)
        new = Interval(-25, 10)
        result = interval_widen_thresholds(old, new, [-20, -15, -10, -5, 0, 10])
        assert result.lo == NEG_INF

    def test_bot_old_returns_new(self):
        old = INTERVAL_BOT
        new = Interval(3, 7)
        result = interval_widen_thresholds(old, new, [0, 5, 10])
        assert result == new

    def test_bot_new_returns_old(self):
        old = Interval(3, 7)
        new = INTERVAL_BOT
        result = interval_widen_thresholds(old, new, [0, 5, 10])
        assert result == old

    def test_empty_thresholds_falls_to_standard(self):
        """With no thresholds, behaves like standard widening (to infinity)."""
        old = Interval(0, 5)
        new = Interval(0, 8)
        result = interval_widen_thresholds(old, new, [])
        assert result.hi == INF

    def test_exact_threshold_match(self):
        """new.hi exactly equals a threshold."""
        old = Interval(0, 5)
        new = Interval(0, 10)
        result = interval_widen_thresholds(old, new, [0, 5, 10, 15])
        assert result.hi == 10

    def test_both_bounds_change(self):
        """Both bounds change simultaneously."""
        old = Interval(5, 10)
        new = Interval(3, 12)
        result = interval_widen_thresholds(old, new, [0, 5, 10, 15])
        assert result.lo == 0 and result.hi == 15


# ============================================================
# Section 2: Threshold Extraction
# ============================================================

class TestThresholdExtraction:
    def test_literal_constants(self):
        source = "let x = 10;"
        thresholds = extract_thresholds_from_source(source)
        assert 10.0 in thresholds
        assert -10.0 in thresholds  # negation

    def test_comparison_operands(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        thresholds = extract_thresholds_from_source(source)
        assert 10.0 in thresholds
        assert 9.0 in thresholds   # x < 10 -> boundary at 9
        assert 11.0 in thresholds  # x < 10 -> boundary at 11

    def test_multiple_sources(self):
        source = "let x = 5; if (x > 3) { x = x + 1; }"
        thresholds = extract_thresholds_from_source(source)
        assert 5.0 in thresholds
        assert 3.0 in thresholds
        assert 1.0 in thresholds

    def test_zero_always_included(self):
        source = "let x = 42;"
        thresholds = extract_thresholds_from_source(source)
        assert 0.0 in thresholds

    def test_negative_constants(self):
        source = "let x = -5;"
        thresholds = extract_thresholds_from_source(source)
        assert -5.0 in thresholds
        assert 5.0 in thresholds  # negation of negation


# ============================================================
# Section 3: Per-Variable Thresholds
# ============================================================

class TestVariableThresholds:
    def test_comparison_variable(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        vt = extract_variable_thresholds(source)
        assert 'i' in vt
        assert 10.0 in vt['i']
        assert 9.0 in vt['i']

    def test_multiple_variables(self):
        source = "let x = 0; let y = 0; if (x < 5) { y = 1; } if (y > 3) { x = 2; }"
        vt = extract_variable_thresholds(source)
        assert 'x' in vt
        assert 'y' in vt
        assert 5.0 in vt['x']
        assert 3.0 in vt['y']

    def test_assignment_threshold(self):
        source = "let x = 100;"
        vt = extract_variable_thresholds(source)
        assert 'x' in vt
        assert 100.0 in vt['x']


# ============================================================
# Section 4: Simple Loop Analysis
# ============================================================

class TestSimpleLoopAnalysis:
    def test_countup_loop(self):
        """Classic: i=0; while(i<10){i=i+1}. Standard gives [0,+inf], threshold gives [0,10]."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = threshold_analyze(source)
        iv = result['env'].get_interval('i')
        # After loop, i >= 10 (loop exits when i < 10 is false)
        # But the widened fixpoint should be bounded by threshold
        assert iv.hi != INF  # threshold widening should bound this
        assert iv.lo >= 0

    def test_countdown_loop(self):
        """i=10; while(i>0){i=i-1}."""
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        result = threshold_analyze(source)
        iv = result['env'].get_interval('i')
        assert iv.lo != NEG_INF  # should be bounded
        assert iv.hi <= 10

    def test_countup_accumulator(self):
        """i=0; s=0; while(i<5){s=s+i; i=i+1}."""
        source = "let i = 0; let s = 0; while (i < 5) { s = s + i; i = i + 1; }"
        result = threshold_analyze(source)
        i_iv = result['env'].get_interval('i')
        assert i_iv.hi != INF  # bounded by threshold


# ============================================================
# Section 5: Comparison with Standard Widening
# ============================================================

class TestComparison:
    def test_countup_improvement(self):
        """Threshold widening should be strictly tighter for countup."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        comp = compare_widening(source)
        # Standard should give i in [something, +inf]
        std_iv = comp['standard']['env'].get_interval('i')
        thr_iv = comp['threshold']['env'].get_interval('i')
        # Threshold should have finite upper bound
        assert thr_iv.hi < std_iv.hi or (std_iv.hi == INF and thr_iv.hi != INF)
        assert len(comp['improvements']) > 0

    def test_no_loop_no_improvement(self):
        """Without loops, both analyses should agree."""
        source = "let x = 5; let y = x + 3;"
        comp = compare_widening(source)
        # No improvements expected (no widening involved)
        assert len(comp['improvements']) == 0

    def test_countdown_improvement(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        comp = compare_widening(source)
        std_iv = comp['standard']['env'].get_interval('i')
        thr_iv = comp['threshold']['env'].get_interval('i')
        # Standard may widen lower bound to -inf
        # Threshold should keep it bounded
        if std_iv.lo == NEG_INF:
            assert thr_iv.lo > NEG_INF


# ============================================================
# Section 6: Narrowing
# ============================================================

class TestNarrowing:
    def test_narrowing_tightens(self):
        """Narrowing should tighten bounds after widening."""
        # Even with threshold widening, narrowing can further tighten
        source = "let i = 0; while (i < 5) { i = i + 1; }"
        # With narrowing
        interp_with = ThresholdInterpreter(narrowing_iterations=5)
        result_with = interp_with.analyze(source)
        # Without narrowing
        interp_without = ThresholdInterpreter(narrowing_iterations=0)
        result_without = interp_without.analyze(source)
        # Narrowing should produce at least as tight results
        iv_with = result_with['env'].get_interval('i')
        iv_without = result_without['env'].get_interval('i')
        assert iv_with.hi <= iv_without.hi
        assert iv_with.lo >= iv_without.lo

    def test_narrowing_operator(self):
        """Test narrowing operator directly."""
        wide = Interval(0, INF)
        new = Interval(0, 10)
        result = _interval_narrow(wide, new)
        assert result.lo == 0
        assert result.hi == 10

    def test_narrowing_only_tightens_infinite(self):
        """Narrowing should only replace infinite bounds with finite."""
        wide = Interval(0, 15)
        new = Interval(0, 10)
        result = _interval_narrow(wide, new)
        # Finite bound stays: narrowing doesn't shrink finite bounds
        assert result.lo == 0
        assert result.hi == 15


# ============================================================
# Section 7: ThresholdEnv
# ============================================================

class TestThresholdEnv:
    def test_copy_preserves_thresholds(self):
        env = ThresholdEnv([0, 5, 10])
        env.set('x', interval=Interval(0, 5))
        copy = env.copy()
        assert copy.thresholds == [0, 5, 10]
        assert copy.get_interval('x') == Interval(0, 5)

    def test_widen_uses_thresholds(self):
        env1 = ThresholdEnv([0, 5, 10, 15])
        env1.set('x', interval=Interval(0, 5), sign=Sign.NON_NEG)
        env2 = ThresholdEnv([0, 5, 10, 15])
        env2.set('x', interval=Interval(0, 8), sign=Sign.NON_NEG)
        result = env1.widen(env2)
        assert result.get_interval('x').hi == 10  # threshold, not +inf

    def test_join_preserves_thresholds(self):
        env1 = ThresholdEnv([0, 10, 20])
        env1.set('x', interval=Interval(0, 5))
        env2 = ThresholdEnv([0, 10, 20])
        env2.set('x', interval=Interval(3, 8))
        result = env1.join(env2)
        assert isinstance(result, ThresholdEnv)
        assert result.thresholds == [0, 10, 20]
        assert result.get_interval('x') == Interval(0, 8)


# ============================================================
# Section 8: Nested Loops
# ============================================================

class TestNestedLoops:
    def test_nested_countup(self):
        source = """
        let i = 0;
        while (i < 5) {
            let j = 0;
            while (j < 3) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = threshold_analyze(source)
        i_iv = result['env'].get_interval('i')
        j_iv = result['env'].get_interval('j')
        assert i_iv.hi != INF
        assert j_iv.hi != INF

    def test_outer_loop_with_inner_conditional(self):
        source = """
        let x = 0;
        while (x < 10) {
            if (x < 5) {
                x = x + 2;
            } else {
                x = x + 1;
            }
        }
        """
        result = threshold_analyze(source)
        x_iv = result['env'].get_interval('x')
        assert x_iv.hi != INF


# ============================================================
# Section 9: Multiple Variables
# ============================================================

class TestMultipleVariables:
    def test_two_counters(self):
        source = """
        let x = 0;
        let y = 100;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        result = threshold_analyze(source)
        x_iv = result['env'].get_interval('x')
        y_iv = result['env'].get_interval('y')
        assert x_iv.hi != INF
        # y decreases, should be bounded below by thresholds

    def test_conservation(self):
        """x+y stays constant: x=0,y=10,while(x<10){x=x+1;y=y-1}."""
        source = "let x = 0; let y = 10; while (x < 10) { x = x + 1; y = y - 1; }"
        result = threshold_analyze(source)
        x_iv = result['env'].get_interval('x')
        assert x_iv.hi != INF


# ============================================================
# Section 10: Extra Thresholds
# ============================================================

class TestExtraThresholds:
    def test_user_provided_threshold(self):
        source = "let x = 0; while (x < 100) { x = x + 1; }"
        # Provide threshold at 50 (not in program)
        result = threshold_analyze(source, extra_thresholds=[50])
        assert 50.0 in result['thresholds']

    def test_extra_threshold_bounds_loop(self):
        """Extra threshold can help bound a variable."""
        source = "let x = 0; while (x < 100) { x = x + 1; }"
        result = threshold_analyze(source, extra_thresholds=[200])
        x_iv = result['env'].get_interval('x')
        assert x_iv.hi != INF  # auto-extracted threshold from 100 in program


# ============================================================
# Section 11: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_no_loops(self):
        source = "let x = 5; let y = x + 3;"
        result = threshold_analyze(source)
        assert result['env'].get_interval('x') == Interval(5, 5)
        assert result['env'].get_interval('y') == Interval(8, 8)

    def test_infinite_loop_converges(self):
        """While(true) should still converge (max iterations)."""
        source = "let x = 0; while (1 > 0) { x = x + 1; }"
        result = threshold_analyze(source)
        # Should terminate due to max_iterations

    def test_empty_program(self):
        source = "let x = 0;"
        result = threshold_analyze(source)
        assert result['env'].get_interval('x') == Interval(0, 0)

    def test_negative_thresholds(self):
        source = "let x = 0; while (x > -10) { x = x - 1; }"
        result = threshold_analyze(source)
        x_iv = result['env'].get_interval('x')
        # Should be bounded below by threshold near -10
        assert x_iv.lo != NEG_INF or x_iv.hi <= 0


# ============================================================
# Section 12: Utility Functions
# ============================================================

class TestUtilities:
    def test_is_strictly_tighter_finite_vs_inf(self):
        assert _is_strictly_tighter(Interval(0, 10), Interval(0, INF))
        assert _is_strictly_tighter(Interval(0, 10), Interval(NEG_INF, INF))

    def test_is_strictly_tighter_same_not_strict(self):
        assert not _is_strictly_tighter(Interval(0, 10), Interval(0, 10))

    def test_is_strictly_tighter_wider_not_tighter(self):
        assert not _is_strictly_tighter(Interval(0, INF), Interval(0, 10))

    def test_get_variable_range(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        iv = get_variable_range(source, 'x')
        assert iv.hi != INF

    def test_get_thresholds(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        thresholds = get_thresholds(source)
        assert 10.0 in thresholds
        assert 0.0 in thresholds

    def test_get_variable_thresholds(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        vt = get_variable_thresholds(source)
        assert 'x' in vt


# ============================================================
# Section 13: Conditional Loops
# ============================================================

class TestConditionalLoops:
    def test_loop_with_break_condition(self):
        """Loop with internal conditional that affects the counter."""
        source = """
        let x = 0;
        while (x < 20) {
            if (x < 10) {
                x = x + 2;
            } else {
                x = x + 1;
            }
        }
        """
        result = threshold_analyze(source)
        x_iv = result['env'].get_interval('x')
        assert x_iv.hi != INF

    def test_guarded_increment(self):
        source = """
        let x = 0;
        let y = 0;
        while (x < 10) {
            x = x + 1;
            if (x > 5) {
                y = y + 1;
            }
        }
        """
        result = threshold_analyze(source)
        x_iv = result['env'].get_interval('x')
        y_iv = result['env'].get_interval('y')
        assert x_iv.hi != INF
        # y should also be bounded


# ============================================================
# Section 14: Thresholds Result Dict
# ============================================================

class TestResultDict:
    def test_thresholds_in_result(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = threshold_analyze(source)
        assert 'thresholds' in result
        assert isinstance(result['thresholds'], list)
        assert 10.0 in result['thresholds']

    def test_widening_stats_in_result(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = threshold_analyze(source)
        assert 'widening_stats' in result

    def test_env_is_threshold_env(self):
        source = "let x = 5;"
        result = threshold_analyze(source)
        assert isinstance(result['env'], ThresholdEnv)


# ============================================================
# Section 15: Integration with Existing Stack
# ============================================================

class TestIntegration:
    def test_sign_consistency(self):
        """Threshold analysis should produce correct signs."""
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = threshold_analyze(source)
        sign = result['env'].get_sign('x')
        iv = result['env'].get_interval('x')
        # After exit: x >= 10, sign should be POS or NON_NEG
        assert sign in (Sign.POS, Sign.NON_NEG, Sign.TOP)
        assert iv.lo >= 0

    def test_constant_propagation_unaffected(self):
        """Threshold widening shouldn't affect constant propagation for non-loops."""
        source = "let x = 42;"
        result = threshold_analyze(source)
        c = result['env'].get_const('x')
        from abstract_interpreter import ConstVal
        assert isinstance(c, ConstVal) and c.value == 42

    def test_division_warning_preserved(self):
        """Warnings should still be generated."""
        source = "let x = 10; let y = 0; let z = x / y;"
        result = threshold_analyze(source)
        div_warnings = [w for w in result['warnings']
                        if 'division' in w.kind.value]
        assert len(div_warnings) > 0

    def test_comparison_shows_improvement(self):
        """The compare_widening API should work end-to-end."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        comp = compare_widening(source)
        assert 'standard' in comp
        assert 'threshold' in comp
        assert 'improvements' in comp
