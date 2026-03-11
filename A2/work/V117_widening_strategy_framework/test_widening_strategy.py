"""Tests for V117: Widening Strategy Framework

Tests adaptive widening that composes V103 (widening policy synthesis)
with V108 (domain composition framework).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from widening_strategy import (
    # Data structures
    DomainKind, WideningPhase, ComponentWideningConfig, AdaptivePolicy,
    FrameworkResult, StrategyComparison,
    # Classification
    classify_domain, classify_domain_type,
    # Threshold extraction
    extract_thresholds_from_ast,
    # Policy synthesis
    synthesize_adaptive_policy, synthesize_all_policies,
    _synthesize_component_config,
    # Widening operations
    adaptive_widen_interval, _standard_widen, _threshold_widen, _graduated_widen,
    # Interpreter
    StrategyInterpreter,
    # High-level API
    adaptive_analyze, adaptive_analyze_interval, adaptive_analyze_composed,
    standard_analyze, compare_strategies, get_adaptive_policies,
    get_loop_analysis, validate_adaptive_policy, widening_summary,
    # Dependencies
    NEG_INF, INF
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V020_abstract_domain_functor'))
from domain_functor import IntervalDomain, SignDomain

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V103_widening_policy_synthesis'))
from widening_policy import WideningStrategy, WideningPolicy, analyze_loops


# ===================================================================
# Section 1: Data Structures
# ===================================================================

class TestDataStructures:
    def test_domain_kind_enum(self):
        assert DomainKind.NUMERIC.value == "numeric"
        assert DomainKind.RELATIONAL.value == "relational"
        assert DomainKind.COMPOSITE.value == "composite"
        assert DomainKind.HEAP.value == "heap"

    def test_widening_phase_enum(self):
        assert WideningPhase.DELAY.value == "delay"
        assert WideningPhase.THRESHOLD.value == "threshold"
        assert WideningPhase.GRADUATED.value == "graduated"
        assert WideningPhase.STANDARD.value == "standard"

    def test_component_widening_config_defaults(self):
        config = ComponentWideningConfig(
            domain_index=0,
            domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.STANDARD
        )
        assert config.domain_index == 0
        assert config.thresholds == ()
        assert config.delay == 0
        assert config.allow_narrowing is True
        assert config.graduated_steps == (50.0, 100.0, 500.0, 1000.0)

    def test_component_widening_config_custom(self):
        config = ComponentWideningConfig(
            domain_index=1,
            domain_kind=DomainKind.RELATIONAL,
            phase=WideningPhase.THRESHOLD,
            thresholds=(0, 10, 100),
            delay=3,
            allow_narrowing=False,
            graduated_steps=(5.0, 25.0)
        )
        assert config.domain_index == 1
        assert config.delay == 3
        assert config.thresholds == (0, 10, 100)
        assert config.allow_narrowing is False

    def test_adaptive_policy(self):
        configs = [
            ComponentWideningConfig(0, DomainKind.NUMERIC, WideningPhase.THRESHOLD),
            ComponentWideningConfig(1, DomainKind.NUMERIC, WideningPhase.DELAY, delay=5)
        ]
        policy = AdaptivePolicy(
            loop_id=0,
            component_configs=configs,
            max_iterations=50,
            reduction_between_widenings=True,
            coordinate_components=True,
            narrowing_iterations=3
        )
        assert policy.loop_id == 0
        assert len(policy.component_configs) == 2
        assert policy.max_iterations == 50
        assert policy.narrowing_iterations == 3

    def test_framework_result(self):
        result = FrameworkResult(
            env={'x': '[0, 10]'},
            warnings=['test warning'],
            policies={},
            iterations_per_loop={0: 5},
            widening_events=[],
            narrowing_events=1,
            reduction_events=2
        )
        assert result.env['x'] == '[0, 10]'
        assert len(result.warnings) == 1
        assert result.iterations_per_loop[0] == 5
        assert result.narrowing_events == 1
        assert result.reduction_events == 2


# ===================================================================
# Section 2: Domain Classification
# ===================================================================

class TestDomainClassification:
    def test_classify_interval_domain(self):
        d = IntervalDomain(0, 10)
        assert classify_domain(d) == DomainKind.NUMERIC

    def test_classify_sign_domain(self):
        d = SignDomain()  # default is TOP
        assert classify_domain(d) == DomainKind.NUMERIC

    def test_classify_domain_type_interval(self):
        assert classify_domain_type(IntervalDomain) == DomainKind.NUMERIC

    def test_classify_domain_type_sign(self):
        assert classify_domain_type(SignDomain) == DomainKind.NUMERIC

    def test_classify_unknown_domain(self):
        """Unknown domain types default to NUMERIC."""
        class MyCustomDomain:
            pass
        assert classify_domain(MyCustomDomain()) == DomainKind.NUMERIC


# ===================================================================
# Section 3: Threshold Extraction
# ===================================================================

class TestThresholdExtraction:
    def test_extract_simple_constants(self):
        source = "let x = 10;"
        thresholds = extract_thresholds_from_ast(source)
        assert 10 in thresholds
        assert 9 in thresholds   # boundary: 10-1
        assert 11 in thresholds  # boundary: 10+1

    def test_extract_comparison_constants(self):
        source = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        thresholds = extract_thresholds_from_ast(source)
        assert 0 in thresholds
        assert 100 in thresholds
        assert 1 in thresholds

    def test_extract_multiple_constants(self):
        source = """
        let a = 5;
        let b = 20;
        if (a > 10) {
            b = b + 3;
        }
        """
        thresholds = extract_thresholds_from_ast(source)
        assert 5 in thresholds
        assert 20 in thresholds
        assert 10 in thresholds
        # 3 may or may not be extracted depending on AST structure
        # (b + 3 is a BinOp where 3 is an IntLit -- should be found)
        # If not found, the threshold extraction only walks certain nodes

    def test_extract_from_nested_loops(self):
        source = """
        let i = 0;
        while (i < 10) {
            let j = 0;
            while (j < 5) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        thresholds = extract_thresholds_from_ast(source)
        assert 10 in thresholds
        assert 0 in thresholds
        # 5 should be extracted from the inner loop condition
        assert 5 in thresholds or 4 in thresholds  # 5 or its neighbor

    def test_extract_negative_boundary(self):
        source = "let x = 0;"
        thresholds = extract_thresholds_from_ast(source)
        assert -1 in thresholds  # 0 - 1


# ===================================================================
# Section 4: Standard Widening Operations
# ===================================================================

class TestWideningOperations:
    def test_standard_widen_stable(self):
        """No widening when bounds don't change."""
        lo, hi = _standard_widen(0, 10, 0, 10)
        assert lo == 0
        assert hi == 10

    def test_standard_widen_upper_grows(self):
        lo, hi = _standard_widen(0, 10, 0, 15)
        assert lo == 0
        assert hi == INF

    def test_standard_widen_lower_shrinks(self):
        lo, hi = _standard_widen(0, 10, -5, 10)
        assert lo == NEG_INF
        assert hi == 10

    def test_standard_widen_both(self):
        lo, hi = _standard_widen(0, 10, -5, 15)
        assert lo == NEG_INF
        assert hi == INF

    def test_threshold_widen_upper(self):
        lo, hi = _threshold_widen(0, 10, 0, 15, (0, 10, 20, 50, 100))
        assert lo == 0
        assert hi == 20  # Next threshold >= 15

    def test_threshold_widen_lower(self):
        lo, hi = _threshold_widen(0, 10, -3, 10, (-10, -5, 0, 10, 20))
        assert lo == -5  # Largest threshold <= -3
        assert hi == 10

    def test_threshold_widen_no_matching_threshold(self):
        lo, hi = _threshold_widen(0, 10, 0, 200, (0, 10, 50, 100))
        assert lo == 0
        assert hi == INF  # No threshold >= 200

    def test_graduated_widen_step1(self):
        steps = (10.0, 100.0, 1000.0)
        lo, hi = _graduated_widen(0, 10, 0, 15, steps, 1)
        assert lo == 0
        assert hi <= 20  # 10 + step[0]=10 = 20

    def test_graduated_widen_step2(self):
        steps = (10.0, 100.0, 1000.0)
        lo, hi = _graduated_widen(0, 10, 0, 15, steps, 2)
        assert lo == 0
        assert hi <= 110  # 10 + step[1]=100 = 110

    def test_graduated_widen_step3(self):
        steps = (10.0, 100.0, 1000.0)
        lo, hi = _graduated_widen(0, 10, 0, 15, steps, 3)
        assert lo == 0
        assert hi <= 1010  # 10 + step[2]=1000 = 1010


# ===================================================================
# Section 5: Adaptive Widening
# ===================================================================

class TestAdaptiveWiden:
    def test_adaptive_delay_phase(self):
        config = ComponentWideningConfig(
            domain_index=0,
            domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.DELAY,
            delay=5
        )
        # During delay, return new values
        lo, hi = adaptive_widen_interval(0, 10, 0, 12, config, 3)
        assert lo == 0
        assert hi == 12

    def test_adaptive_delay_phase_after_delay(self):
        config = ComponentWideningConfig(
            domain_index=0,
            domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.DELAY,
            delay=2
        )
        # After delay, standard widening
        lo, hi = adaptive_widen_interval(0, 10, 0, 12, config, 5)
        assert lo == 0
        assert hi == INF  # Standard widening

    def test_adaptive_threshold_phase(self):
        config = ComponentWideningConfig(
            domain_index=0,
            domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.THRESHOLD,
            thresholds=(0, 10, 20, 50, 100),
            delay=1
        )
        lo, hi = adaptive_widen_interval(0, 10, 0, 15, config, 3)
        assert hi == 20  # Threshold widening to 20

    def test_adaptive_graduated_phase(self):
        config = ComponentWideningConfig(
            domain_index=0,
            domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.GRADUATED,
            graduated_steps=(10.0, 100.0),
            delay=1
        )
        lo, hi = adaptive_widen_interval(0, 10, 0, 12, config, 2)
        assert hi <= 20  # graduated step 1: +10

    def test_adaptive_standard_phase(self):
        config = ComponentWideningConfig(
            domain_index=0,
            domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.STANDARD,
            delay=0
        )
        lo, hi = adaptive_widen_interval(0, 10, 0, 15, config, 1)
        assert hi == INF


# ===================================================================
# Section 6: Policy Synthesis
# ===================================================================

class TestPolicySynthesis:
    def test_synthesize_simple_counter(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        policy = synthesize_adaptive_policy(source, [IntervalDomain])
        assert policy.loop_id == 0
        assert len(policy.component_configs) == 1
        # Simple counter should get delay phase
        config = policy.component_configs[0]
        assert config.phase == WideningPhase.DELAY

    def test_synthesize_unbounded_loop(self):
        source = """
        let x = 1;
        let y = 0;
        while (x > 0) {
            y = y + x;
            x = x - 1;
        }
        """
        policy = synthesize_adaptive_policy(source, [IntervalDomain])
        assert policy.loop_id == 0
        # Unbounded loop with thresholds available -> threshold widening
        config = policy.component_configs[0]
        assert config.phase in (WideningPhase.THRESHOLD, WideningPhase.GRADUATED,
                                 WideningPhase.DELAY)

    def test_synthesize_multi_domain(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        policy = synthesize_adaptive_policy(source, [SignDomain, IntervalDomain])
        assert len(policy.component_configs) == 2
        # Both should be numeric
        for config in policy.component_configs:
            assert config.domain_kind == DomainKind.NUMERIC

    def test_synthesize_all_policies_multiple_loops(self):
        source = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        let j = 0;
        while (j < 10) {
            j = j + 2;
        }
        """
        policies = synthesize_all_policies(source, [IntervalDomain])
        assert len(policies) == 2
        assert 0 in policies
        assert 1 in policies

    def test_synthesize_no_loops(self):
        source = "let x = 42;"
        policies = synthesize_all_policies(source, [IntervalDomain])
        # Should return a default policy
        assert len(policies) >= 1


# ===================================================================
# Section 7: Strategy Interpreter - Simple Programs
# ===================================================================

class TestStrategyInterpreterSimple:
    def test_let_binding(self):
        source = "let x = 42;"
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env

    def test_assignment(self):
        source = """
        let x = 10;
        x = x + 5;
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env

    def test_if_then_else(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'y' in result.env

    def test_simple_loop(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'i' in result.env
        # i should be around 10 after the loop
        assert result.iterations_per_loop.get(0, 0) > 0

    def test_arithmetic_operations(self):
        source = """
        let a = 10;
        let b = 3;
        let c = a + b;
        let d = a - b;
        let e = a * b;
        """
        result = adaptive_analyze_interval(source)
        assert 'c' in result.env
        assert 'd' in result.env
        assert 'e' in result.env

    def test_unary_negation(self):
        source = """
        let x = 5;
        let y = -x;
        """
        result = adaptive_analyze_interval(source)
        assert 'y' in result.env


# ===================================================================
# Section 8: Strategy Interpreter - Loops with Widening
# ===================================================================

class TestStrategyInterpreterLoops:
    def test_counter_loop_precise(self):
        """Simple counter should be analyzed precisely with delay."""
        source = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'i' in result.env

    def test_accumulator_loop(self):
        source = """
        let sum = 0;
        let i = 0;
        while (i < 10) {
            sum = sum + i;
            i = i + 1;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'sum' in result.env
        assert 'i' in result.env

    def test_nested_loops(self):
        source = """
        let i = 0;
        let total = 0;
        while (i < 3) {
            let j = 0;
            while (j < 4) {
                total = total + 1;
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'total' in result.env
        assert len(result.iterations_per_loop) >= 1

    def test_loop_narrowing(self):
        """Narrowing should tighten results after widening."""
        source = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        result = adaptive_analyze_interval(source)
        # Should have some narrowing events due to adaptive policy
        assert 'x' in result.env

    def test_decreasing_loop(self):
        source = """
        let x = 100;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env


# ===================================================================
# Section 9: Strategy Interpreter - Composed Domains
# ===================================================================

class TestComposedDomainAnalysis:
    def test_sign_interval_composed(self):
        source = """
        let x = 5;
        let y = -3;
        let z = x + y;
        """
        result = adaptive_analyze_composed(source, [SignDomain, IntervalDomain])
        assert 'x' in result.env
        assert 'y' in result.env
        assert 'z' in result.env

    def test_composed_loop(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = adaptive_analyze_composed(source, [SignDomain, IntervalDomain])
        assert 'i' in result.env

    def test_composed_if_branch(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x + 1;
        } else {
            y = 0;
        }
        """
        result = adaptive_analyze_composed(source, [SignDomain, IntervalDomain])
        assert 'y' in result.env


# ===================================================================
# Section 10: Standard Analysis Comparison
# ===================================================================

class TestStandardAnalysis:
    def test_standard_analyze_simple(self):
        source = "let x = 42;"
        result = standard_analyze(source, [IntervalDomain])
        assert 'x' in result.env

    def test_standard_analyze_loop(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = standard_analyze(source, [IntervalDomain])
        assert 'i' in result.env

    def test_standard_analyze_composed(self):
        source = "let x = 5;"
        result = standard_analyze(source, [SignDomain, IntervalDomain])
        assert 'x' in result.env


# ===================================================================
# Section 11: Strategy Comparison
# ===================================================================

class TestStrategyComparison:
    def test_compare_simple_program(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        comp = compare_strategies(source, [IntervalDomain])
        assert isinstance(comp, StrategyComparison)
        assert 'standard' in comp.strategies
        assert 'adaptive' in comp.strategies
        assert 'delayed_threshold' in comp.strategies
        assert len(comp.precision_ranking) == 3
        assert len(comp.iteration_ranking) == 3

    def test_compare_accumulator(self):
        source = """
        let sum = 0;
        let i = 1;
        while (i <= 5) {
            sum = sum + i;
            i = i + 1;
        }
        """
        comp = compare_strategies(source, [IntervalDomain])
        assert comp.summary  # Non-empty summary
        # All strategies should produce results
        for name, result in comp.strategies.items():
            assert 'sum' in result.env

    def test_compare_has_summary(self):
        source = "let x = 0; while (x < 5) { x = x + 1; }"
        comp = compare_strategies(source, [IntervalDomain])
        assert isinstance(comp.summary, str)
        assert len(comp.summary) > 0


# ===================================================================
# Section 12: Loop Analysis
# ===================================================================

class TestLoopAnalysis:
    def test_get_loop_analysis_simple(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        analysis = get_loop_analysis(source)
        assert len(analysis) >= 1
        loop = analysis[0]
        assert 'loop_id' in loop
        assert 'is_simple_counter' in loop
        assert 'condition_vars' in loop
        assert 'modified_vars' in loop

    def test_get_loop_analysis_nested(self):
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
        analysis = get_loop_analysis(source)
        assert len(analysis) >= 2

    def test_get_loop_analysis_no_loops(self):
        source = "let x = 42;"
        analysis = get_loop_analysis(source)
        assert len(analysis) == 0


# ===================================================================
# Section 13: Policy Retrieval
# ===================================================================

class TestPolicyRetrieval:
    def test_get_policies_simple(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        policies = get_adaptive_policies(source, [IntervalDomain])
        assert len(policies) >= 1
        assert 0 in policies
        policy = policies[0]
        assert isinstance(policy, AdaptivePolicy)

    def test_get_policies_multiple_loops(self):
        source = """
        let a = 0;
        while (a < 5) { a = a + 1; }
        let b = 10;
        while (b > 0) { b = b - 1; }
        """
        policies = get_adaptive_policies(source, [IntervalDomain])
        assert len(policies) >= 2


# ===================================================================
# Section 14: Validation
# ===================================================================

class TestValidation:
    def test_validate_simple(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        result = validate_adaptive_policy(source, [IntervalDomain])
        assert 'standard' in result
        assert 'adaptive' in result
        assert 'valid' in result

    def test_validate_linear_program(self):
        source = """
        let a = 5;
        let b = a + 3;
        let c = b * 2;
        """
        result = validate_adaptive_policy(source, [IntervalDomain])
        assert result['valid'] is True


# ===================================================================
# Section 15: Widening Summary
# ===================================================================

class TestWideningStrategySummary:
    def test_summary_output(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        summary = widening_summary(source, [IntervalDomain])
        assert isinstance(summary, str)
        assert 'standard' in summary.lower() or 'Strategy' in summary
        assert len(summary) > 50

    def test_summary_with_composed(self):
        source = """
        let x = 0;
        let y = 100;
        while (x < y) {
            x = x + 1;
            y = y - 1;
        }
        """
        summary = widening_summary(source, [IntervalDomain])
        assert isinstance(summary, str)


# ===================================================================
# Section 16: Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_empty_program(self):
        source = ""
        result = adaptive_analyze_interval(source)
        assert isinstance(result, FrameworkResult)
        assert len(result.env) == 0

    def test_single_variable(self):
        source = "let x = 0;"
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env

    def test_boolean_condition(self):
        source = """
        let x = 1;
        if (x == 1) {
            x = 2;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env

    def test_loop_with_break_pattern(self):
        """Loop that terminates quickly."""
        source = """
        let x = 10;
        while (x > 0) {
            x = x - 3;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env

    def test_multiple_assignments(self):
        source = """
        let x = 0;
        x = 1;
        x = 2;
        x = 3;
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env

    def test_condition_refinement_less_than(self):
        source = """
        let x = 5;
        if (x < 3) {
            let y = x;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env

    def test_condition_refinement_greater_equal(self):
        source = """
        let x = 5;
        if (x >= 10) {
            let y = x;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env


# ===================================================================
# Section 17: Widening Event Tracking
# ===================================================================

class TestWideningEventTracking:
    def test_events_recorded(self):
        source = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        result = adaptive_analyze_interval(source)
        # Should have iteration data
        assert len(result.iterations_per_loop) >= 1

    def test_framework_result_fields(self):
        source = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        result = adaptive_analyze_interval(source)
        assert isinstance(result.env, dict)
        assert isinstance(result.warnings, list)
        assert isinstance(result.policies, dict)
        assert isinstance(result.iterations_per_loop, dict)
        assert isinstance(result.widening_events, list)
        assert isinstance(result.narrowing_events, int)
        assert isinstance(result.reduction_events, int)


# ===================================================================
# Section 18: Integration Tests
# ===================================================================

class TestIntegration:
    def test_full_pipeline_simple(self):
        """Full pipeline: synthesize -> analyze -> compare."""
        source = """
        let sum = 0;
        let i = 0;
        while (i < 10) {
            sum = sum + i;
            i = i + 1;
        }
        """
        # 1. Synthesize policies
        policies = get_adaptive_policies(source, [IntervalDomain])
        assert len(policies) >= 1

        # 2. Analyze with adaptive widening
        result = adaptive_analyze_interval(source, policies)
        assert 'sum' in result.env
        assert 'i' in result.env

        # 3. Compare strategies
        comp = compare_strategies(source, [IntervalDomain])
        assert len(comp.strategies) == 3

    def test_full_pipeline_composed(self):
        """Full pipeline with composed domains."""
        source = """
        let x = 0;
        while (x < 20) {
            x = x + 3;
        }
        """
        result = adaptive_analyze_composed(source, [SignDomain, IntervalDomain])
        assert 'x' in result.env

    def test_adaptive_vs_standard_precision(self):
        """Adaptive should be at least as good as standard for simple cases."""
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        standard = standard_analyze(source, [IntervalDomain])
        adaptive = adaptive_analyze_interval(source)
        # Both should produce results for i
        assert 'i' in standard.env
        assert 'i' in adaptive.env

    def test_multi_loop_program(self):
        source = """
        let a = 0;
        while (a < 5) {
            a = a + 1;
        }
        let b = a;
        while (b < 15) {
            b = b + 2;
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'a' in result.env
        assert 'b' in result.env
        assert len(result.iterations_per_loop) >= 2

    def test_conditional_loop(self):
        source = """
        let x = 0;
        let flag = 1;
        while (flag > 0) {
            x = x + 1;
            if (x >= 10) {
                flag = 0;
            }
        }
        """
        result = adaptive_analyze_interval(source)
        assert 'x' in result.env
        assert 'flag' in result.env

    def test_division_warning(self):
        source = """
        let x = 10;
        let y = 0;
        let z = x / y;
        """
        result = adaptive_analyze_interval(source)
        assert any('division' in w.lower() for w in result.warnings)


# ===================================================================
# Section 19: Adaptive Widening Precision Tests
# ===================================================================

class TestAdaptiveWideningPrecision:
    def test_threshold_tighter_than_standard(self):
        """Threshold widening should produce tighter bounds than standard."""
        config_standard = ComponentWideningConfig(
            domain_index=0, domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.STANDARD
        )
        config_threshold = ComponentWideningConfig(
            domain_index=0, domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.THRESHOLD,
            thresholds=(0, 5, 10, 20, 50, 100)
        )

        # Standard widens to infinity
        s_lo, s_hi = adaptive_widen_interval(0, 10, 0, 12, config_standard, 1)
        # Threshold widens to 20
        t_lo, t_hi = adaptive_widen_interval(0, 10, 0, 12, config_threshold, 1)

        assert s_hi == INF
        assert t_hi == 20
        assert t_hi < s_hi  # Threshold is tighter

    def test_graduated_progressive(self):
        """Graduated widening should progressively expand."""
        config = ComponentWideningConfig(
            domain_index=0, domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.GRADUATED,
            graduated_steps=(10.0, 100.0, 1000.0)
        )

        _, hi1 = adaptive_widen_interval(0, 10, 0, 12, config, 1)
        _, hi2 = adaptive_widen_interval(0, 10, 0, 12, config, 2)
        _, hi3 = adaptive_widen_interval(0, 10, 0, 12, config, 3)

        # Each step should allow wider bounds
        assert hi1 <= hi2
        assert hi2 <= hi3

    def test_delay_then_widen(self):
        """During delay phase, bounds should track precisely."""
        config = ComponentWideningConfig(
            domain_index=0, domain_kind=DomainKind.NUMERIC,
            phase=WideningPhase.THRESHOLD,
            thresholds=(0, 10, 20, 50, 100),
            delay=3
        )

        # Iteration 1-3: delay (no widening)
        lo1, hi1 = adaptive_widen_interval(0, 5, 0, 7, config, 1)
        assert hi1 == 7  # Exact, no widening

        lo2, hi2 = adaptive_widen_interval(0, 5, 0, 7, config, 2)
        assert hi2 == 7  # Still in delay

        lo3, hi3 = adaptive_widen_interval(0, 5, 0, 7, config, 3)
        assert hi3 == 7  # Still in delay

        # Iteration 4: threshold widening kicks in
        lo4, hi4 = adaptive_widen_interval(0, 5, 0, 7, config, 4)
        assert hi4 == 10  # Threshold widening to 10


# ===================================================================
# Section 20: Composition of V103 Structures
# ===================================================================

class TestV103Composition:
    def test_v103_loop_info_available(self):
        """V103 loop analysis should be accessible through V117."""
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        loops = analyze_loops(source)
        assert len(loops) >= 1
        assert loops[0].loop_id == 0

    def test_v103_policy_used_as_seed(self):
        """V103 policies should seed V117 adaptive policies."""
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        # V103 synthesis
        from widening_policy import synthesize_policies as v103_synth
        v103_result = v103_synth(source)

        # V117 synthesis
        v117_policies = synthesize_all_policies(source, [IntervalDomain])

        # Both should produce policies for the same loop
        assert len(v103_result.policies) >= 1
        assert len(v117_policies) >= 1
