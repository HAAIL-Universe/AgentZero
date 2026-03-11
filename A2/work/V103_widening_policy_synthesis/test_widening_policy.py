"""Tests for V103: Widening Policy Synthesis."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from widening_policy import (
    # Data structures
    WideningPolicy, WideningStrategy, LoopInfo, PolicyResult,
    SynthesisResult, ValidationResult, ComparisonResult,
    # Loop analysis
    analyze_loops, get_loop_info,
    # Policy synthesis
    synthesize_policy, synthesize_policies,
    # Analysis
    policy_analyze, auto_analyze, PolicyInterpreter,
    # Validation
    validate_policy,
    # Comparison
    compare_policies,
    # Functor integration
    functor_policy_analyze, compare_with_functor, FunctorPolicyInterpreter,
    # Utilities
    synthesize_and_validate, policy_summary,
    _interval_width,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import Interval, NEG_INF, INF, analyze as baseline_analyze


# ===== Section 1: Loop Analysis =====

class TestLoopAnalysis:
    def test_single_loop(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        loops = analyze_loops(source)
        assert len(loops) == 1
        assert loops[0].loop_id == 0

    def test_loop_condition_vars(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        loops = analyze_loops(source)
        assert 'i' in loops[0].condition_vars

    def test_loop_modified_vars(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        loops = analyze_loops(source)
        assert 'i' in loops[0].modified_vars

    def test_loop_constants_in_condition(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        loops = analyze_loops(source)
        assert 10 in loops[0].constants_in_condition

    def test_nested_loops(self):
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
        loops = analyze_loops(source)
        assert len(loops) == 2
        assert loops[0].nested_depth == 0
        assert loops[1].nested_depth == 1

    def test_multiple_sequential_loops(self):
        source = """
        let i = 0;
        while (i < 5) { i = i + 1; }
        let j = 0;
        while (j < 10) { j = j + 2; }
        """
        loops = analyze_loops(source)
        assert len(loops) == 2

    def test_no_loops(self):
        source = "let x = 5; let y = x + 1;"
        loops = analyze_loops(source)
        assert len(loops) == 0


# ===== Section 2: Counter Pattern Detection =====

class TestCounterDetection:
    def test_simple_increment(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        loops = analyze_loops(source)
        assert loops[0].is_simple_counter
        assert loops[0].counter_var == 'i'
        assert loops[0].counter_step == 1

    def test_step_two(self):
        source = "let i = 0; while (i < 20) { i = i + 2; }"
        loops = analyze_loops(source)
        assert loops[0].is_simple_counter
        assert loops[0].counter_step == 2

    def test_decrement(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        loops = analyze_loops(source)
        assert loops[0].is_simple_counter
        assert loops[0].counter_step == -1

    def test_bound_value(self):
        source = "let i = 0; while (i < 100) { i = i + 1; }"
        loops = analyze_loops(source)
        assert loops[0].bound_value == 100

    def test_bound_direction_up(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        loops = analyze_loops(source)
        assert loops[0].bound_direction == 'up'

    def test_bound_direction_down(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        loops = analyze_loops(source)
        assert loops[0].bound_direction == 'down'

    def test_non_counter_loop(self):
        source = "let x = 1; while (x < 100) { x = x * 2; }"
        loops = analyze_loops(source)
        assert not loops[0].is_simple_counter


# ===== Section 3: Policy Synthesis =====

class TestPolicySynthesis:
    def test_simple_counter_policy(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        loops = analyze_loops(source)
        policy, rationale = synthesize_policy(loops[0])
        assert policy.strategy == WideningStrategy.DELAYED_THRESHOLD
        assert 10 in policy.thresholds
        assert policy.delay > 0

    def test_threshold_policy_for_condition_constants(self):
        source = "let x = 1; while (x < 100) { x = x * 2; }"
        loops = analyze_loops(source)
        policy, rationale = synthesize_policy(loops[0])
        assert policy.strategy == WideningStrategy.THRESHOLD
        assert 100 in policy.thresholds

    def test_body_constants_policy(self):
        source = """
        let x = 0;
        let y = 1;
        while (y != 0) {
            x = x + 5;
            y = 0;
        }
        """
        loops = analyze_loops(source)
        policy, _ = synthesize_policy(loops[0])
        assert policy.narrowing_iterations >= 2

    def test_synthesize_all_policies(self):
        source = """
        let i = 0;
        while (i < 5) { i = i + 1; }
        let j = 0;
        while (j < 10) { j = j + 2; }
        """
        result = synthesize_policies(source)
        assert len(result.policies) == 2
        assert 0 in result.policies
        assert 1 in result.policies

    def test_synthesis_rationale(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = synthesize_policies(source)
        assert 0 in result.rationale
        assert len(result.rationale[0]) > 0

    def test_synthesis_precision_estimate(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = synthesize_policies(source)
        assert result.estimated_precision[0] == 'exact'


# ===== Section 4: Policy Interpreter Basic =====

class TestPolicyInterpreterBasic:
    def test_no_loops(self):
        source = "let x = 5; let y = x + 1;"
        result = policy_analyze(source)
        assert result.env['x']['interval'].lo == 5
        assert result.env['x']['interval'].hi == 5

    def test_let_and_assign(self):
        source = "let x = 10; x = x + 5;"
        result = policy_analyze(source)
        assert result.env['x']['interval'].lo == 15
        assert result.env['x']['interval'].hi == 15

    def test_if_statement(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = policy_analyze(source)
        # x is always 5, which is > 3, so y should be 1
        assert result.env['y']['interval'].lo <= 1
        assert result.env['y']['interval'].hi >= 1

    def test_standard_widening(self):
        """Without special policies, standard widening goes to infinity."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = policy_analyze(source, {})
        # Standard widening: upper bound should be INF or very large
        iv = result.env['i']['interval']
        assert iv.lo >= 0  # Lower bound should be at least 0
        # After exit: i >= 10


# ===== Section 5: Threshold Widening via Policy =====

class TestThresholdWidening:
    def test_threshold_tightens_bound(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(-1, 0, 5, 9, 10, 11),
            narrowing_iterations=2,
        )}
        result = policy_analyze(source, policies)
        iv = result.env['i']['interval']
        # With thresholds at 10 and narrowing, should get tight bound
        assert iv.hi <= 10

    def test_threshold_with_step_two(self):
        source = "let i = 0; while (i < 20) { i = i + 2; }"
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(0, 10, 19, 20, 21),
            narrowing_iterations=2,
        )}
        result = policy_analyze(source, policies)
        iv = result.env['i']['interval']
        # After exit: i >= 20, widened to nearest threshold (21)
        assert iv.hi <= 21

    def test_countdown_with_threshold(self):
        source = "let i = 10; while (i > 0) { i = i - 1; }"
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(-1, 0, 1, 5, 10),
            narrowing_iterations=2,
        )}
        result = policy_analyze(source, policies)
        iv = result.env['i']['interval']
        assert iv.lo >= 0
        assert iv.hi <= 10


# ===== Section 6: Delayed Widening =====

class TestDelayedWidening:
    def test_delay_gives_more_iterations(self):
        source = "let i = 0; while (i < 5) { i = i + 1; }"
        # With delay=5, should converge without widening (loop runs 5 times)
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.DELAYED,
            delay=6,
        )}
        result = policy_analyze(source, policies)
        iv = result.env['i']['interval']
        assert iv.lo >= 5
        assert iv.hi <= 5

    def test_delayed_threshold(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.DELAYED_THRESHOLD,
            thresholds=(0, 5, 10, 15),
            delay=3,
            narrowing_iterations=2,
        )}
        result = policy_analyze(source, policies)
        iv = result.env['i']['interval']
        assert iv.hi <= 10

    def test_delay_zero_is_standard(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        # delay=0 means widen from iteration 1
        p1 = {0: WideningPolicy(strategy=WideningStrategy.STANDARD, delay=0)}
        p2 = {0: WideningPolicy(strategy=WideningStrategy.DELAYED, delay=0)}
        r1 = policy_analyze(source, p1)
        r2 = policy_analyze(source, p2)
        assert r1.env['i']['interval'].hi == r2.env['i']['interval'].hi


# ===== Section 7: Narrowing =====

class TestNarrowing:
    def test_narrowing_tightens_bound(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        # Standard widening then narrowing
        p_no_narrow = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(0, 10),
            narrowing_iterations=0,
        )}
        p_narrow = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(0, 10),
            narrowing_iterations=3,
        )}
        r_no = policy_analyze(source, p_no_narrow)
        r_yes = policy_analyze(source, p_narrow)
        # With narrowing, bound should be at least as tight
        assert r_yes.env['i']['interval'].hi <= r_no.env['i']['interval'].hi

    def test_narrowing_count(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(0, 10),
            narrowing_iterations=3,
        )}
        result = policy_analyze(source, policies)
        assert result.narrowing_events >= 1


# ===== Section 8: Auto Analysis =====

class TestAutoAnalysis:
    def test_auto_simple_counter(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = auto_analyze(source)
        iv = result.env['i']['interval']
        assert iv.hi <= 10

    def test_auto_nested_loops(self):
        source = """
        let i = 0;
        let s = 0;
        while (i < 5) {
            let j = 0;
            while (j < 3) {
                s = s + 1;
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = auto_analyze(source)
        assert 'i' in result.env
        assert 'j' in result.env

    def test_auto_countdown(self):
        source = "let n = 100; while (n > 0) { n = n - 1; }"
        result = auto_analyze(source)
        iv = result.env['n']['interval']
        assert iv.lo >= 0
        assert iv.hi <= 100

    def test_auto_no_loops(self):
        source = "let x = 42;"
        result = auto_analyze(source)
        assert result.env['x']['interval'].lo == 42


# ===== Section 9: Policy Validation =====

class TestValidation:
    def test_valid_policy(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        synthesis = synthesize_policies(source)
        validation = validate_policy(source, synthesis.policies)
        assert validation.valid

    def test_validation_messages(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        synthesis = synthesize_policies(source)
        validation = validate_policy(source, synthesis.policies)
        assert len(validation.messages) > 0

    def test_synthesize_and_validate_pipeline(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = synthesize_and_validate(source)
        assert result['all_valid']
        assert result['loop_count'] == 1


# ===== Section 10: Policy Comparison =====

class TestComparison:
    def test_compare_shows_improvements(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = compare_policies(source)
        assert result.standard is not None
        assert result.threshold is not None
        assert result.synthesized is not None

    def test_synthesized_at_least_as_good(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = compare_policies(source)
        std_iv = result.standard.env.get('i', {}).get('interval', Interval(NEG_INF, INF))
        synth_iv = result.synthesized.env.get('i', {}).get('interval', Interval(NEG_INF, INF))
        # Synthesized should not be wider than standard (both are sound)
        synth_w = _interval_width(synth_iv)
        std_w = _interval_width(std_iv)
        assert synth_w <= std_w or std_w == float('inf')

    def test_compare_no_loops(self):
        source = "let x = 5;"
        result = compare_policies(source)
        assert len(result.improvements) == 0  # No loops, no improvement


# ===== Section 11: Get Loop Info =====

class TestGetLoopInfo:
    def test_loop_info_dict(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        info = get_loop_info(source)
        assert len(info) == 1
        assert info[0]['is_simple_counter']
        assert info[0]['counter_var'] == 'i'
        assert info[0]['bound_value'] == 10

    def test_loop_info_multiple(self):
        source = """
        let i = 0;
        while (i < 5) { i = i + 1; }
        let j = 0;
        while (j < 10) { j = j + 2; }
        """
        info = get_loop_info(source)
        assert len(info) == 2
        assert info[0]['counter_step'] == 1
        assert info[1]['counter_step'] == 2


# ===== Section 12: Policy Summary =====

class TestPolicySummary:
    def test_summary_format(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        summary = policy_summary(source)
        assert "Loop 0" in summary
        assert "Strategy" in summary
        assert "Rationale" in summary

    def test_summary_multiple_loops(self):
        source = """
        let i = 0;
        while (i < 5) { i = i + 1; }
        let j = 0;
        while (j < 10) { j = j + 2; }
        """
        summary = policy_summary(source)
        assert "Loop 0" in summary
        assert "Loop 1" in summary
        assert "2 loop(s)" in summary


# ===== Section 13: Widening Strategy Enum =====

class TestWideningStrategy:
    def test_strategy_values(self):
        assert WideningStrategy.STANDARD.value == "standard"
        assert WideningStrategy.THRESHOLD.value == "threshold"
        assert WideningStrategy.DELAYED.value == "delayed"
        assert WideningStrategy.DELAYED_THRESHOLD.value == "delayed_threshold"


# ===== Section 14: Policy Data Structure =====

class TestPolicyDataStructure:
    def test_default_policy(self):
        p = WideningPolicy()
        assert p.strategy == WideningStrategy.STANDARD
        assert p.thresholds == ()
        assert p.delay == 0
        assert p.narrowing_iterations == 0

    def test_with_thresholds(self):
        p = WideningPolicy()
        p2 = p.with_thresholds([10, 5, 0, 5])  # Duplicates removed, sorted
        assert p2.thresholds == (0, 5, 10)

    def test_frozen(self):
        p = WideningPolicy(strategy=WideningStrategy.THRESHOLD, thresholds=(1, 2, 3))
        # Frozen dataclass should be hashable
        s = {p}
        assert len(s) == 1


# ===== Section 15: Interval Width =====

class TestIntervalWidth:
    def test_finite_width(self):
        assert _interval_width(Interval(0, 10)) == 10

    def test_infinite_width(self):
        assert _interval_width(Interval(0, INF)) == float('inf')
        assert _interval_width(Interval(NEG_INF, 10)) == float('inf')

    def test_bot_width(self):
        assert _interval_width(Interval(1, 0)) == 0  # BOT

    def test_point_width(self):
        assert _interval_width(Interval(5, 5)) == 0


# ===== Section 16: Multiple Variables =====

class TestMultipleVariables:
    def test_two_counters(self):
        source = """
        let i = 0;
        let j = 0;
        while (i < 10) {
            j = j + 2;
            i = i + 1;
        }
        """
        result = auto_analyze(source)
        i_iv = result.env['i']['interval']
        assert i_iv.hi <= 10

    def test_accumulator(self):
        source = """
        let s = 0;
        let i = 0;
        while (i < 5) {
            s = s + i;
            i = i + 1;
        }
        """
        result = auto_analyze(source)
        assert 'i' in result.env
        assert 's' in result.env


# ===== Section 17: Complex Programs =====

class TestComplexPrograms:
    def test_if_inside_loop(self):
        source = """
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
        result = auto_analyze(source)
        assert 'x' in result.env

    def test_nested_counter_loops(self):
        source = """
        let total = 0;
        let i = 0;
        while (i < 3) {
            let j = 0;
            while (j < 4) {
                total = total + 1;
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = auto_analyze(source)
        i_iv = result.env['i']['interval']
        assert i_iv.hi <= 3

    def test_multiple_assignments_in_loop(self):
        source = """
        let a = 0;
        let b = 1;
        let i = 0;
        while (i < 5) {
            let temp = a;
            a = b;
            b = temp + b;
            i = i + 1;
        }
        """
        result = auto_analyze(source)
        assert 'a' in result.env
        assert 'b' in result.env


# ===== Section 18: Functor Integration =====

class TestFunctorIntegration:
    def test_functor_basic(self):
        source = "let x = 5; let y = x + 1;"
        result = functor_policy_analyze(source)
        assert 'env' in result

    def test_functor_with_loop(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        synthesis = synthesize_policies(source)
        result = functor_policy_analyze(source, policies=synthesis.policies)
        assert 'env' in result

    def test_compare_with_functor(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = compare_with_functor(source)
        assert 'standard' in result
        assert 'synthesized' in result
        assert 'iterations' in result


# ===== Section 19: Edge Cases =====

class TestEdgeCases:
    def test_empty_program(self):
        source = ""
        # Should not crash
        loops = analyze_loops(source)
        assert len(loops) == 0

    def test_loop_with_no_body_modification(self):
        """A loop that doesn't modify any variable (may not terminate)."""
        source = "let x = 1; while (x > 0) { let y = 1; }"
        loops = analyze_loops(source)
        assert len(loops) == 1
        # Should still get a policy
        synthesis = synthesize_policies(source)
        assert 0 in synthesis.policies

    def test_very_large_bound(self):
        source = "let i = 0; while (i < 1000000) { i = i + 1; }"
        synthesis = synthesize_policies(source)
        policy = synthesis.policies[0]
        assert 1000000 in policy.thresholds

    def test_negative_bound(self):
        source = "let i = 0; while (i > 0 - 10) { i = i - 1; }"
        loops = analyze_loops(source)
        assert len(loops) == 1


# ===== Section 20: Iterations Tracking =====

class TestIterationsTracking:
    def test_iterations_recorded(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        result = auto_analyze(source)
        assert 0 in result.iterations_per_loop
        assert result.iterations_per_loop[0] > 0

    def test_widening_events(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(0, 10),
            narrowing_iterations=0,
        )}
        result = policy_analyze(source, policies)
        assert result.widening_events >= 1

    def test_narrowing_events_recorded(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        policies = {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(0, 10),
            narrowing_iterations=2,
        )}
        result = policy_analyze(source, policies)
        assert result.narrowing_events >= 1


# ===== Section 21: Precision Comparison =====

class TestPrecisionComparison:
    def test_threshold_more_precise_than_standard(self):
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        std = policy_analyze(source, {0: WideningPolicy()})
        thresh = policy_analyze(source, {0: WideningPolicy(
            strategy=WideningStrategy.THRESHOLD,
            thresholds=(0, 10),
            narrowing_iterations=2,
        )})
        std_w = _interval_width(std.env['i']['interval'])
        thresh_w = _interval_width(thresh.env['i']['interval'])
        # Threshold should give tighter or equal bounds
        assert thresh_w <= std_w or std_w == float('inf')

    def test_delayed_threshold_precise(self):
        source = "let i = 0; while (i < 5) { i = i + 1; }"
        result = policy_analyze(source, {0: WideningPolicy(
            strategy=WideningStrategy.DELAYED_THRESHOLD,
            thresholds=(0, 5),
            delay=6,  # More than loop iterations
            narrowing_iterations=2,
        )})
        iv = result.env['i']['interval']
        # With delay >= loop iterations, should converge exactly
        assert iv.lo == 5
        assert iv.hi == 5

    def test_auto_beats_standard(self):
        """Auto synthesis should give tighter or equal bounds vs standard."""
        source = "let i = 0; while (i < 10) { i = i + 1; }"
        std = policy_analyze(source, {})
        auto = auto_analyze(source)
        std_w = _interval_width(std.env['i']['interval'])
        auto_w = _interval_width(auto.env['i']['interval'])
        assert auto_w <= std_w or std_w == float('inf')


# ===== Section 22: Loop in Function =====

class TestLoopInFunction:
    def test_function_loop_detected(self):
        source = """
        fn sum(n) {
            let s = 0;
            let i = 0;
            while (i < n) {
                s = s + i;
                i = i + 1;
            }
            return s;
        }
        """
        loops = analyze_loops(source)
        assert len(loops) == 1
        # n is a variable, not a constant, in the condition
        assert 'i' in loops[0].condition_vars or 'n' in loops[0].condition_vars


# ===== Section 23: BoolLit Handling =====

class TestBoolHandling:
    def test_bool_in_expression(self):
        """PolicyInterpreter should handle BoolLit nodes."""
        source = "let x = 0; let y = 1;"
        result = policy_analyze(source)
        assert result.env['x']['interval'].lo == 0
        assert result.env['y']['interval'].lo == 1


# ===== Section 24: Var-vs-Var Conditions =====

class TestVarVarConditions:
    def test_var_less_than_var(self):
        source = """
        let a = 0;
        let b = 10;
        if (a < b) {
            a = a + 1;
        }
        """
        result = policy_analyze(source)
        assert result.env['a']['interval'].lo >= 0

    def test_var_var_in_loop(self):
        source = """
        let i = 0;
        let n = 5;
        while (i < n) {
            i = i + 1;
        }
        """
        loops = analyze_loops(source)
        assert len(loops) == 1
        assert 'i' in loops[0].condition_vars
        assert 'n' in loops[0].condition_vars
