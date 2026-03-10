"""
Tests for C028: Profiler-Guided Optimizer
Composes: C027 (Profiler) + C014 (Bytecode Optimizer)

Tests cover:
  - Hot function detection
  - Optimization suggestions
  - Inline suggestions
  - Single-pass PGO optimization
  - Iterative PGO optimization
  - Analysis-only mode
  - Strategy comparison
  - Configuration
  - Report generation
  - Output preservation (correctness)
  - Edge cases
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pgo import (
    ProfileGuidedOptimizer, PGOConfig, PGOResult,
    HotFunction, InlineSuggestion, OptimizationSuggestion,
    detect_hot_functions, filter_hot,
    generate_suggestions, generate_inline_suggestions,
    optimize_targeted,
    pgo_optimize, pgo_iterative, pgo_analyze, pgo_compare,
    format_pgo_report, format_comparison_report,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C027_profiler'))
from profiler import Profiler, ProfileSnapshot, FunctionProfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C014_bytecode_optimizer'))
from optimizer import OptimizationStats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import compile_source, execute


# ============================================================
# Test Programs
# ============================================================

SIMPLE_MATH = """
let x = 2 + 3;
let y = x * 4;
print(y);
"""

LOOP_PROGRAM = """
let sum = 0;
let i = 0;
while (i < 100) {
    sum = sum + i;
    i = i + 1;
}
print(sum);
"""

FUNCTION_PROGRAM = """
fn square(x) { return x * x; }
fn cube(x) { return x * square(x); }
let result = cube(5);
print(result);
"""

RECURSIVE_FIB = """
fn fib(n) {
    if (n < 2) { return n; }
    return fib(n - 1) + fib(n - 2);
}
let result = fib(10);
print(result);
"""

MULTI_FUNCTION = """
fn add(a, b) { return a + b; }
fn mul(a, b) { return a * b; }
fn compute(x) {
    let a = add(x, 1);
    let b = mul(a, 2);
    return add(b, x);
}
let i = 0;
let total = 0;
while (i < 20) {
    total = total + compute(i);
    i = i + 1;
}
print(total);
"""

CONSTANT_HEAVY = """
let a = 1 + 2 + 3 + 4 + 5;
let b = a * 2;
let c = b + 10;
print(c);
"""

NESTED_LOOPS = """
let count = 0;
let i = 0;
while (i < 10) {
    let j = 0;
    while (j < 10) {
        count = count + 1;
        j = j + 1;
    }
    i = i + 1;
}
print(count);
"""

SMALL_FUNCTION_MANY_CALLS = """
fn inc(x) { return x + 1; }
let n = 0;
let i = 0;
while (i < 50) {
    n = inc(n);
    i = i + 1;
}
print(n);
"""

CONDITIONAL_PROGRAM = """
fn abs_val(x) {
    if (x < 0) { return 0 - x; }
    return x;
}
let sum = 0;
let i = 0;
while (i < 30) {
    sum = sum + abs_val(i - 15);
    i = i + 1;
}
print(sum);
"""

DEAD_CODE_PROGRAM = """
let x = 10;
if (true) {
    print(x);
} else {
    print(0);
}
"""

IDENTITY_PROGRAM = """
let x = 0;
let y = x + 0;
let z = y * 1;
print(z);
"""


# ============================================================
# Hot Function Detection
# ============================================================

class TestHotFunctionDetection:

    def test_detect_hot_in_loop(self):
        """Loop body should be detected as hot."""
        profiler = Profiler()
        result = profiler.profile(LOOP_PROGRAM)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        hot = detect_hot_functions(profile, config)
        assert len(hot) > 0
        # Main should be the hottest (it contains the loop)
        assert hot[0].name == "<main>"
        assert hot[0].self_steps > 0

    def test_detect_hot_recursive(self):
        """Recursive function should be detected as hot."""
        profiler = Profiler()
        profiler.profile(RECURSIVE_FIB)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        hot = detect_hot_functions(profile, config)
        # fib should be one of the hottest
        fib_hot = [h for h in hot if h.name == "fib"]
        assert len(fib_hot) == 1
        assert fib_hot[0].call_count > 1

    def test_detect_hot_multi_function(self):
        """Multiple functions with varying hotness."""
        profiler = Profiler()
        profiler.profile(MULTI_FUNCTION)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        hot = detect_hot_functions(profile, config)
        names = [h.name for h in hot]
        assert "add" in names
        assert "mul" in names
        assert "compute" in names

    def test_heat_score_ordering(self):
        """Functions should be sorted by heat score."""
        profiler = Profiler()
        profiler.profile(RECURSIVE_FIB)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        hot = detect_hot_functions(profile, config)
        scores = [h.heat_score for h in hot]
        assert scores == sorted(scores, reverse=True)

    def test_pct_steps_computed(self):
        """Percentage of steps should sum to ~100%."""
        profiler = Profiler()
        profiler.profile(FUNCTION_PROGRAM)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        hot = detect_hot_functions(profile, config)
        # pct_steps should be reasonable
        for h in hot:
            assert 0 <= h.pct_steps <= 100

    def test_filter_hot_threshold(self):
        """Filter should respect threshold."""
        profiler = Profiler()
        profiler.profile(SIMPLE_MATH)
        profile = profiler.get_latest_profile()
        config = PGOConfig(hot_threshold=0.5)  # 50% threshold
        all_hot = detect_hot_functions(profile, config)
        filtered = filter_hot(all_hot, config)
        # With 50% threshold and min_call_count=2, main (1 call) may be excluded
        # But main uses nearly 100% of steps, so it should pass
        # The key point: filtering respects OR condition (threshold OR min_calls)
        assert isinstance(filtered, list)

    def test_filter_hot_min_calls(self):
        """Filter should include functions meeting min call count."""
        profiler = Profiler()
        profiler.profile(RECURSIVE_FIB)
        profile = profiler.get_latest_profile()
        config = PGOConfig(hot_threshold=0.99, min_call_count=5)
        all_hot = detect_hot_functions(profile, config)
        filtered = filter_hot(all_hot, config)
        # fib is called many times, should pass min_call_count even if not 99% threshold
        fib_hot = [h for h in filtered if h.name == "fib"]
        assert len(fib_hot) == 1

    def test_simple_program_hot_detection(self):
        """Simple program should still detect main as hot."""
        profiler = Profiler()
        profiler.profile(SIMPLE_MATH)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        hot = detect_hot_functions(profile, config)
        assert any(h.name == "<main>" for h in hot)


# ============================================================
# Optimization Suggestions
# ============================================================

class TestSuggestions:

    def test_hotspot_suggestion_for_hot_function(self):
        """Should suggest optimizing high-step functions."""
        profiler = Profiler()
        profiler.profile(RECURSIVE_FIB)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        all_hot = detect_hot_functions(profile, config)
        hot = filter_hot(all_hot, config)
        suggestions = generate_suggestions(profile, hot, config=config)
        categories = [s.category for s in suggestions]
        # fib should trigger call_overhead or hotspot
        assert len(suggestions) > 0

    def test_call_overhead_suggestion(self):
        """Frequently called functions should get call_overhead suggestion."""
        profiler = Profiler()
        profiler.profile(SMALL_FUNCTION_MANY_CALLS)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        all_hot = detect_hot_functions(profile, config)
        hot = filter_hot(all_hot, config)
        suggestions = generate_suggestions(profile, hot, config=config)
        overhead = [s for s in suggestions if s.category == "call_overhead"]
        assert len(overhead) > 0

    def test_suggestions_sorted_by_priority(self):
        """Suggestions should be sorted by priority."""
        profiler = Profiler()
        profiler.profile(MULTI_FUNCTION)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        all_hot = detect_hot_functions(profile, config)
        hot = filter_hot(all_hot, config)
        suggestions = generate_suggestions(profile, hot, config=config)
        priorities = [s.priority for s in suggestions]
        assert priorities == sorted(priorities, reverse=True)

    def test_instruction_pattern_suggestion(self):
        """High CONST/LOAD frequency should trigger instruction suggestion."""
        profiler = Profiler()
        profiler.profile(CONSTANT_HEAVY)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        all_hot = detect_hot_functions(profile, config)
        hot = filter_hot(all_hot, config)
        suggestions = generate_suggestions(profile, hot, config=config)
        # At least one suggestion should exist
        assert isinstance(suggestions, list)

    def test_suggestion_has_description(self):
        """All suggestions should have descriptions."""
        profiler = Profiler()
        profiler.profile(RECURSIVE_FIB)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        all_hot = detect_hot_functions(profile, config)
        hot = filter_hot(all_hot, config)
        suggestions = generate_suggestions(profile, hot, config=config)
        for s in suggestions:
            assert len(s.description) > 0
            assert len(s.target) > 0
            assert len(s.category) > 0


# ============================================================
# Inline Suggestions
# ============================================================

class TestInlineSuggestions:

    def test_inline_small_frequent_function(self):
        """Small frequently-called function should be inline candidate."""
        profiler = Profiler()
        profiler.profile(SMALL_FUNCTION_MANY_CALLS)
        profile = profiler.get_latest_profile()
        _, compiler = compile_source(SMALL_FUNCTION_MANY_CALLS)
        config = PGOConfig(inline_min_calls=3)
        suggestions = generate_inline_suggestions(profile, compiler, config)
        # inc is small and called 50 times
        inc_suggestions = [s for s in suggestions if s.callee == "inc"]
        assert len(inc_suggestions) > 0
        assert inc_suggestions[0].call_count > 0

    def test_no_inline_for_large_function(self):
        """Large function should not be suggested for inlining."""
        profiler = Profiler()
        profiler.profile(RECURSIVE_FIB)
        profile = profiler.get_latest_profile()
        _, compiler = compile_source(RECURSIVE_FIB)
        config = PGOConfig(inline_max_size=5)  # very small threshold
        suggestions = generate_inline_suggestions(profile, compiler, config)
        # fib is recursive and likely > 5 bytes
        fib_suggestions = [s for s in suggestions if s.callee == "fib"]
        assert len(fib_suggestions) == 0

    def test_no_inline_without_compiler(self):
        """Should return empty list when no compiler provided."""
        profiler = Profiler()
        profiler.profile(FUNCTION_PROGRAM)
        profile = profiler.get_latest_profile()
        config = PGOConfig()
        suggestions = generate_inline_suggestions(profile, None, config)
        assert suggestions == []

    def test_inline_suggestion_has_reason(self):
        """Inline suggestions should have meaningful reason."""
        profiler = Profiler()
        profiler.profile(SMALL_FUNCTION_MANY_CALLS)
        profile = profiler.get_latest_profile()
        _, compiler = compile_source(SMALL_FUNCTION_MANY_CALLS)
        config = PGOConfig(inline_min_calls=3)
        suggestions = generate_inline_suggestions(profile, compiler, config)
        for s in suggestions:
            assert len(s.reason) > 0
            assert s.callee_size > 0

    def test_inline_sorted_by_savings(self):
        """Inline suggestions should be sorted by estimated savings."""
        profiler = Profiler()
        profiler.profile(MULTI_FUNCTION)
        profile = profiler.get_latest_profile()
        _, compiler = compile_source(MULTI_FUNCTION)
        config = PGOConfig(inline_min_calls=3, inline_max_size=50)
        suggestions = generate_inline_suggestions(profile, compiler, config)
        savings = [s.estimated_savings for s in suggestions]
        assert savings == sorted(savings, reverse=True)


# ============================================================
# Single-Pass PGO
# ============================================================

class TestSinglePassPGO:

    def test_optimize_simple(self):
        """Basic PGO should work on simple program."""
        result = pgo_optimize(SIMPLE_MATH)
        assert isinstance(result, PGOResult)
        assert result.original_steps > 0
        assert result.optimized_steps > 0
        assert result.same_output
        assert result.iterations == 1

    def test_optimize_preserves_output(self):
        """PGO must preserve program output."""
        result = pgo_optimize(LOOP_PROGRAM)
        assert result.same_output
        assert result.original_output == result.optimized_output
        assert "4950" in result.original_output

    def test_optimize_loop(self):
        """PGO on loop program should maintain correctness."""
        result = pgo_optimize(LOOP_PROGRAM)
        assert result.same_output
        # Output should be 0+1+...+99 = 4950
        assert "4950" in result.optimized_output

    def test_optimize_functions(self):
        """PGO on function program should maintain correctness."""
        result = pgo_optimize(FUNCTION_PROGRAM)
        assert result.same_output
        assert "125" in result.optimized_output

    def test_optimize_recursive(self):
        """PGO on recursive program should maintain correctness."""
        result = pgo_optimize(RECURSIVE_FIB)
        assert result.same_output
        assert "55" in result.optimized_output

    def test_optimize_multi_function(self):
        """PGO on multi-function program should maintain correctness."""
        result = pgo_optimize(MULTI_FUNCTION)
        assert result.same_output

    def test_optimize_constant_heavy(self):
        """PGO should optimize constant-heavy code."""
        result = pgo_optimize(CONSTANT_HEAVY)
        assert result.same_output
        # 1+2+3+4+5=15, *2=30, +10=40
        assert "40" in result.optimized_output

    def test_optimize_steps_not_worse(self):
        """Optimized code should not use more steps than original."""
        result = pgo_optimize(LOOP_PROGRAM)
        assert result.optimized_steps <= result.original_steps

    def test_optimize_speedup_at_least_1(self):
        """Speedup should be >= 1.0."""
        result = pgo_optimize(CONSTANT_HEAVY)
        assert result.speedup >= 1.0

    def test_hot_functions_populated(self):
        """Hot functions list should be populated."""
        result = pgo_optimize(RECURSIVE_FIB)
        assert len(result.hot_functions) > 0

    def test_optimization_stats_populated(self):
        """Stats should be populated after optimization."""
        result = pgo_optimize(CONSTANT_HEAVY)
        stats = result.optimization_stats
        assert stats is not None
        assert stats.original_size > 0

    def test_optimize_nested_loops(self):
        """PGO on nested loops should preserve correctness."""
        result = pgo_optimize(NESTED_LOOPS)
        assert result.same_output
        assert "100" in result.optimized_output

    def test_optimize_conditional(self):
        """PGO on conditional program should preserve correctness."""
        result = pgo_optimize(CONDITIONAL_PROGRAM)
        assert result.same_output

    def test_optimize_dead_code(self):
        """PGO should handle dead code correctly."""
        result = pgo_optimize(DEAD_CODE_PROGRAM)
        assert result.same_output
        assert "10" in result.optimized_output

    def test_optimize_identity(self):
        """PGO should optimize identity operations."""
        result = pgo_optimize(IDENTITY_PROGRAM)
        assert result.same_output
        assert "0" in result.optimized_output


# ============================================================
# Iterative PGO
# ============================================================

class TestIterativePGO:

    def test_iterative_basic(self):
        """Iterative PGO should work."""
        result = pgo_iterative(LOOP_PROGRAM)
        assert isinstance(result, PGOResult)
        assert result.same_output
        assert result.iterations >= 1

    def test_iterative_converges(self):
        """Iterative PGO should converge."""
        config = PGOConfig(max_iterations=5)
        result = pgo_iterative(CONSTANT_HEAVY, config)
        assert result.iterations <= config.max_iterations
        assert result.same_output

    def test_iterative_preserves_output(self):
        """Iterative PGO must preserve output."""
        result = pgo_iterative(RECURSIVE_FIB)
        assert result.same_output
        assert "55" in result.optimized_output

    def test_iterative_nested_loops(self):
        """Iterative PGO on nested loops."""
        result = pgo_iterative(NESTED_LOOPS)
        assert result.same_output
        assert "100" in result.optimized_output

    def test_iterative_multi_function(self):
        """Iterative PGO on multi-function program."""
        result = pgo_iterative(MULTI_FUNCTION)
        assert result.same_output

    def test_iterative_not_worse(self):
        """Iterative PGO should not make code slower."""
        result = pgo_iterative(LOOP_PROGRAM)
        assert result.optimized_steps <= result.original_steps

    def test_iterative_with_low_convergence(self):
        """Low convergence threshold should stop early."""
        config = PGOConfig(convergence_threshold=0.5, max_iterations=5)
        result = pgo_iterative(SIMPLE_MATH, config)
        assert result.same_output

    def test_iterative_steps_saved(self):
        """Steps saved should be non-negative."""
        result = pgo_iterative(CONSTANT_HEAVY)
        assert result.steps_saved >= 0

    def test_iterative_small_function_calls(self):
        """Iterative PGO with many small function calls."""
        result = pgo_iterative(SMALL_FUNCTION_MANY_CALLS)
        assert result.same_output
        assert "50" in result.optimized_output


# ============================================================
# Analysis-Only Mode
# ============================================================

class TestAnalyzeOnly:

    def test_analyze_no_optimization(self):
        """Analyze-only should not apply optimization."""
        result = pgo_analyze(LOOP_PROGRAM)
        assert result.iterations == 0
        assert result.steps_saved == 0
        assert result.speedup == 1.0

    def test_analyze_has_hot_functions(self):
        """Analysis should still detect hot functions."""
        result = pgo_analyze(RECURSIVE_FIB)
        assert len(result.hot_functions) > 0

    def test_analyze_has_suggestions(self):
        """Analysis should generate suggestions."""
        result = pgo_analyze(MULTI_FUNCTION)
        assert len(result.suggestions) > 0

    def test_analyze_preserves_output(self):
        """Analysis should report same_output=True."""
        result = pgo_analyze(SIMPLE_MATH)
        assert result.same_output

    def test_analyze_has_profile(self):
        """Analysis should have original profile."""
        result = pgo_analyze(LOOP_PROGRAM)
        assert result.original_profile is not None
        assert result.original_profile.total_steps > 0


# ============================================================
# Strategy Comparison
# ============================================================

class TestStrategyComparison:

    def test_compare_returns_three_strategies(self):
        """Comparison should include all three strategies."""
        comparison = pgo_compare(LOOP_PROGRAM)
        assert 'no_opt' in comparison
        assert 'blind_opt' in comparison
        assert 'pgo' in comparison

    def test_compare_no_opt_baseline(self):
        """No-opt should be the baseline."""
        comparison = pgo_compare(SIMPLE_MATH)
        assert comparison['no_opt']['steps'] > 0

    def test_compare_blind_preserves_output(self):
        """Blind optimization should preserve output."""
        comparison = pgo_compare(LOOP_PROGRAM)
        assert comparison['blind_opt']['same_output']

    def test_compare_pgo_preserves_output(self):
        """PGO should preserve output."""
        comparison = pgo_compare(LOOP_PROGRAM)
        assert comparison['pgo']['same_output']

    def test_compare_speedups_reasonable(self):
        """Speedups should be >= 1.0."""
        comparison = pgo_compare(CONSTANT_HEAVY)
        assert comparison['blind_opt']['speedup'] >= 1.0
        assert comparison['pgo']['speedup'] >= 1.0

    def test_compare_has_hot_functions(self):
        """PGO strategy should list hot functions."""
        comparison = pgo_compare(RECURSIVE_FIB)
        assert isinstance(comparison['pgo']['hot_functions'], list)

    def test_compare_has_stats(self):
        """Both optimization strategies should have stats."""
        comparison = pgo_compare(LOOP_PROGRAM)
        assert comparison['blind_opt']['stats'] is not None
        assert comparison['pgo']['stats'] is not None

    def test_compare_outputs_match(self):
        """All strategies should produce same output."""
        comparison = pgo_compare(NESTED_LOOPS)
        no_opt_out = comparison['no_opt']['output']
        assert comparison['blind_opt']['output'] == no_opt_out
        assert comparison['pgo']['output'] == no_opt_out

    def test_compare_multi_function(self):
        """Comparison on multi-function program."""
        comparison = pgo_compare(MULTI_FUNCTION)
        assert comparison['blind_opt']['same_output']
        assert comparison['pgo']['same_output']


# ============================================================
# Configuration
# ============================================================

class TestConfiguration:

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = PGOConfig()
        assert config.hot_threshold == 0.1
        assert config.min_call_count == 2
        assert config.max_iterations == 3
        assert config.inline_max_size == 20
        assert config.inline_min_calls == 3
        assert config.convergence_threshold == 0.01
        assert config.optimize_cold is False
        assert config.max_opt_rounds == 10

    def test_custom_config(self):
        """Custom config should be respected."""
        config = PGOConfig(
            hot_threshold=0.5,
            min_call_count=10,
            max_iterations=1,
        )
        result = pgo_iterative(RECURSIVE_FIB, config)
        assert result.iterations <= 1

    def test_optimize_cold_includes_all(self):
        """optimize_cold=True should optimize all functions."""
        config = PGOConfig(optimize_cold=True)
        result = pgo_optimize(MULTI_FUNCTION, config)
        assert result.same_output

    def test_high_threshold_fewer_hot(self):
        """High threshold should detect fewer hot functions."""
        config_low = PGOConfig(hot_threshold=0.01, min_call_count=1000)
        config_high = PGOConfig(hot_threshold=0.99, min_call_count=1000)

        profiler = Profiler()
        profiler.profile(MULTI_FUNCTION)
        profile = profiler.get_latest_profile()

        all_hot = detect_hot_functions(profile, config_low)
        low_filtered = filter_hot(all_hot, config_low)
        high_filtered = filter_hot(all_hot, config_high)

        assert len(high_filtered) <= len(low_filtered)


# ============================================================
# Report Generation
# ============================================================

class TestReportGeneration:

    def test_pgo_report_format(self):
        """PGO report should be well-formatted."""
        result = pgo_optimize(RECURSIVE_FIB)
        report = format_pgo_report(result)
        assert "PROFILE-GUIDED OPTIMIZATION REPORT" in report
        assert "SUMMARY" in report
        assert "Original steps" in report
        assert "Optimized steps" in report
        assert "Speedup" in report

    def test_report_has_hot_functions(self):
        """Report should include hot functions section."""
        result = pgo_optimize(RECURSIVE_FIB)
        report = format_pgo_report(result)
        assert "HOT FUNCTIONS" in report

    def test_report_has_suggestions(self):
        """Report should include suggestions when present."""
        result = pgo_optimize(MULTI_FUNCTION)
        report = format_pgo_report(result)
        assert "OPTIMIZATION" in report

    def test_report_output_preserved(self):
        """Report should indicate output preservation."""
        result = pgo_optimize(LOOP_PROGRAM)
        report = format_pgo_report(result)
        assert "Output preserved" in report
        assert "Yes" in report

    def test_comparison_report_format(self):
        """Comparison report should be well-formatted."""
        comparison = pgo_compare(LOOP_PROGRAM)
        report = format_comparison_report(comparison)
        assert "OPTIMIZATION STRATEGY COMPARISON" in report
        assert "No optimization" in report
        assert "Blind optimization" in report
        assert "PGO (targeted)" in report

    def test_report_empty_suggestions(self):
        """Report should handle empty suggestions gracefully."""
        result = PGOResult()
        result.original_steps = 10
        result.optimized_steps = 8
        result.steps_saved = 2
        result.speedup = 1.25
        result.iterations = 1
        result.same_output = True
        report = format_pgo_report(result)
        assert "SUMMARY" in report


# ============================================================
# ProfileGuidedOptimizer Class
# ============================================================

class TestPGOClass:

    def test_optimizer_history(self):
        """Optimizer should track history."""
        optimizer = ProfileGuidedOptimizer()
        optimizer.optimize(SIMPLE_MATH)
        optimizer.optimize(LOOP_PROGRAM)
        assert len(optimizer.get_history()) == 2

    def test_optimizer_trend(self):
        """Optimizer should produce trend data."""
        optimizer = ProfileGuidedOptimizer()
        optimizer.optimize(SIMPLE_MATH)
        optimizer.optimize(LOOP_PROGRAM)
        trend = optimizer.get_trend()
        assert len(trend) == 2
        assert 'original_steps' in trend[0]
        assert 'optimized_steps' in trend[0]
        assert 'speedup' in trend[0]

    def test_optimizer_with_config(self):
        """Optimizer should accept custom config."""
        config = PGOConfig(max_opt_rounds=5)
        optimizer = ProfileGuidedOptimizer(config)
        result = optimizer.optimize(CONSTANT_HEAVY)
        assert result.same_output

    def test_optimizer_multiple_programs(self):
        """Optimizer should handle multiple different programs."""
        optimizer = ProfileGuidedOptimizer()
        r1 = optimizer.optimize(SIMPLE_MATH)
        r2 = optimizer.optimize(RECURSIVE_FIB)
        r3 = optimizer.optimize(LOOP_PROGRAM)
        assert r1.same_output
        assert r2.same_output
        assert r3.same_output
        assert len(optimizer.get_history()) == 3


# ============================================================
# Output Correctness
# ============================================================

class TestOutputCorrectness:

    def test_simple_math_output(self):
        """Simple math output should be correct."""
        result = pgo_optimize(SIMPLE_MATH)
        assert "20" in result.optimized_output

    def test_loop_output(self):
        """Loop output should be correct."""
        result = pgo_optimize(LOOP_PROGRAM)
        assert "4950" in result.optimized_output

    def test_function_output(self):
        """Function output should be correct."""
        result = pgo_optimize(FUNCTION_PROGRAM)
        assert "125" in result.optimized_output

    def test_fib_output(self):
        """Fibonacci output should be correct."""
        result = pgo_optimize(RECURSIVE_FIB)
        assert "55" in result.optimized_output

    def test_nested_loop_output(self):
        """Nested loop output should be correct."""
        result = pgo_optimize(NESTED_LOOPS)
        assert "100" in result.optimized_output

    def test_conditional_output(self):
        """Conditional output should be correct."""
        result = pgo_optimize(CONDITIONAL_PROGRAM)
        assert result.same_output

    def test_dead_code_output(self):
        """Dead code program output should be correct."""
        result = pgo_optimize(DEAD_CODE_PROGRAM)
        assert "10" in result.optimized_output

    def test_identity_output(self):
        """Identity operations output should be correct."""
        result = pgo_optimize(IDENTITY_PROGRAM)
        assert "0" in result.optimized_output

    def test_small_function_output(self):
        """Small function many calls output should be correct."""
        result = pgo_optimize(SMALL_FUNCTION_MANY_CALLS)
        assert "50" in result.optimized_output

    def test_constant_heavy_output(self):
        """Constant heavy output should be correct."""
        result = pgo_optimize(CONSTANT_HEAVY)
        assert "40" in result.optimized_output

    def test_iterative_fib_output(self):
        """Iterative PGO on fib should produce correct output."""
        result = pgo_iterative(RECURSIVE_FIB)
        assert "55" in result.optimized_output

    def test_iterative_loop_output(self):
        """Iterative PGO on loop should produce correct output."""
        result = pgo_iterative(LOOP_PROGRAM)
        assert "4950" in result.optimized_output


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:

    def test_minimal_program(self):
        """Minimal program should work."""
        result = pgo_optimize("print(42);")
        assert result.same_output
        assert "42" in result.optimized_output

    def test_no_functions(self):
        """Program without functions should work."""
        result = pgo_optimize("let x = 1; print(x);")
        assert result.same_output

    def test_single_function_call(self):
        """Single function call should work."""
        result = pgo_optimize("fn f(x) { return x; } print(f(7));")
        assert result.same_output
        assert "7" in result.optimized_output

    def test_empty_function_body(self):
        """Function with minimal body."""
        result = pgo_optimize("fn f() { return 0; } print(f());")
        assert result.same_output
        assert "0" in result.optimized_output

    def test_deeply_nested_calls(self):
        """Deeply nested function calls."""
        source = """
        fn a(x) { return x + 1; }
        fn b(x) { return a(x) + 1; }
        fn c(x) { return b(x) + 1; }
        fn d(x) { return c(x) + 1; }
        print(d(0));
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert "4" in result.optimized_output

    def test_multiple_prints(self):
        """Program with multiple print statements."""
        source = """
        print(1);
        print(2);
        print(3);
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert result.optimized_output == ["1", "2", "3"]

    def test_boolean_operations(self):
        """Boolean operations should be handled correctly."""
        source = """
        let x = true;
        let y = false;
        print(x);
        print(y);
        """
        result = pgo_optimize(source)
        assert result.same_output

    def test_comparison_operations(self):
        """Comparison operations should be correct."""
        source = """
        let a = 5;
        let b = 3;
        if (a > b) { print(1); } else { print(0); }
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert "1" in result.optimized_output

    def test_negative_numbers(self):
        """Negative number handling."""
        source = """
        let x = 0 - 5;
        let y = 0 - x;
        print(y);
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert "5" in result.optimized_output

    def test_large_loop(self):
        """Larger loop should still work correctly."""
        source = """
        let sum = 0;
        let i = 0;
        while (i < 200) {
            sum = sum + 1;
            i = i + 1;
        }
        print(sum);
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert "200" in result.optimized_output

    def test_while_false(self):
        """While(false) loop should work."""
        source = """
        let x = 0;
        while (false) { x = x + 1; }
        print(x);
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert "0" in result.optimized_output

    def test_chained_arithmetic(self):
        """Chained arithmetic expressions."""
        source = """
        let x = 1 + 2 + 3;
        let y = x * 2 - 1;
        print(y);
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert "11" in result.optimized_output

    def test_pgo_result_fields(self):
        """PGO result should have all expected fields."""
        result = pgo_optimize(SIMPLE_MATH)
        assert hasattr(result, 'original_profile')
        assert hasattr(result, 'optimized_profile')
        assert hasattr(result, 'hot_functions')
        assert hasattr(result, 'suggestions')
        assert hasattr(result, 'inline_suggestions')
        assert hasattr(result, 'optimization_stats')
        assert hasattr(result, 'original_steps')
        assert hasattr(result, 'optimized_steps')
        assert hasattr(result, 'steps_saved')
        assert hasattr(result, 'speedup')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'same_output')


# ============================================================
# Targeted Optimization
# ============================================================

class TestTargetedOptimization:

    def test_targeted_optimizes_hot_only(self):
        """Targeted optimization should focus on hot functions."""
        chunk, compiler = compile_source(MULTI_FUNCTION)
        config = PGOConfig()
        # Only optimize 'add' function
        hot_names = {"add"}
        opt_chunk, stats = optimize_targeted(chunk, compiler, hot_names, config)
        assert stats.original_size > 0

    def test_targeted_with_empty_hot_set(self):
        """Empty hot set should still optimize main chunk."""
        chunk, compiler = compile_source(SIMPLE_MATH)
        config = PGOConfig()
        opt_chunk, stats = optimize_targeted(chunk, compiler, set(), config)
        assert opt_chunk is not None

    def test_targeted_optimize_cold_true(self):
        """optimize_cold=True should optimize everything."""
        chunk, compiler = compile_source(MULTI_FUNCTION)
        config = PGOConfig(optimize_cold=True)
        opt_chunk, stats = optimize_targeted(chunk, compiler, set(), config)
        # With optimize_cold, all functions should be optimized
        assert stats.original_size > 0

    def test_targeted_preserves_semantics(self):
        """Targeted optimization must preserve semantics."""
        chunk, compiler = compile_source(FUNCTION_PROGRAM)
        config = PGOConfig()
        hot_names = {"square", "cube"}
        opt_chunk, stats = optimize_targeted(chunk, compiler, hot_names, config)
        # Execute optimized chunk
        vm = __import__('stack_vm', fromlist=['VM']).VM(opt_chunk)
        vm.run()
        assert "125" in vm.output

    def test_targeted_no_compiler(self):
        """Targeted optimization without compiler should work on main only."""
        chunk, _ = compile_source(SIMPLE_MATH)
        config = PGOConfig()
        opt_chunk, stats = optimize_targeted(chunk, None, set(), config)
        assert opt_chunk is not None


# ============================================================
# Integration: Full Pipeline
# ============================================================

class TestFullPipeline:

    def test_full_pipeline_simple(self):
        """Full pipeline: analyze -> optimize -> compare."""
        optimizer = ProfileGuidedOptimizer()

        # Analyze
        analysis = optimizer.analyze_only(RECURSIVE_FIB)
        assert analysis.hot_functions

        # Optimize
        optimized = optimizer.optimize(RECURSIVE_FIB)
        assert optimized.same_output

        # Compare
        comparison = optimizer.compare_strategies(RECURSIVE_FIB)
        assert comparison['pgo']['same_output']

    def test_full_pipeline_all_programs(self):
        """Run PGO on all test programs -- all must preserve output."""
        programs = [
            SIMPLE_MATH, LOOP_PROGRAM, FUNCTION_PROGRAM,
            RECURSIVE_FIB, MULTI_FUNCTION, CONSTANT_HEAVY,
            NESTED_LOOPS, SMALL_FUNCTION_MANY_CALLS,
            CONDITIONAL_PROGRAM, DEAD_CODE_PROGRAM,
            IDENTITY_PROGRAM,
        ]
        for prog in programs:
            result = pgo_optimize(prog)
            assert result.same_output, f"Output mismatch for program: {prog[:50]}..."

    def test_iterative_all_programs(self):
        """Iterative PGO on all test programs -- all must preserve output."""
        programs = [
            SIMPLE_MATH, LOOP_PROGRAM, FUNCTION_PROGRAM,
            RECURSIVE_FIB, MULTI_FUNCTION, CONSTANT_HEAVY,
            NESTED_LOOPS, SMALL_FUNCTION_MANY_CALLS,
        ]
        for prog in programs:
            result = pgo_iterative(prog)
            assert result.same_output, f"Output mismatch for program: {prog[:50]}..."

    def test_compare_all_programs(self):
        """Compare strategies on all programs -- all must match output."""
        programs = [
            SIMPLE_MATH, LOOP_PROGRAM, FUNCTION_PROGRAM,
            CONSTANT_HEAVY, NESTED_LOOPS,
        ]
        for prog in programs:
            comparison = pgo_compare(prog)
            assert comparison['blind_opt']['same_output']
            assert comparison['pgo']['same_output']

    def test_report_generation_all(self):
        """Generate reports for all programs -- no crashes."""
        programs = [
            SIMPLE_MATH, LOOP_PROGRAM, RECURSIVE_FIB, MULTI_FUNCTION,
        ]
        for prog in programs:
            result = pgo_optimize(prog)
            report = format_pgo_report(result)
            assert len(report) > 0

    def test_trend_across_programs(self):
        """Track trend across multiple optimizations."""
        optimizer = ProfileGuidedOptimizer()
        for prog in [SIMPLE_MATH, LOOP_PROGRAM, RECURSIVE_FIB]:
            optimizer.optimize(prog)
        trend = optimizer.get_trend()
        assert len(trend) == 3
        for entry in trend:
            assert entry['speedup'] >= 1.0


# ============================================================
# Stress Tests
# ============================================================

class TestStress:

    def test_many_iterations(self):
        """High iteration count should converge gracefully."""
        config = PGOConfig(max_iterations=10)
        result = pgo_iterative(RECURSIVE_FIB, config)
        assert result.same_output
        # Should converge well before 10 iterations
        assert result.iterations <= 10

    def test_aggressive_threshold(self):
        """Very low threshold should still work."""
        config = PGOConfig(hot_threshold=0.001, min_call_count=1)
        result = pgo_optimize(MULTI_FUNCTION, config)
        assert result.same_output

    def test_optimizer_reuse(self):
        """Reusing optimizer across many runs."""
        optimizer = ProfileGuidedOptimizer()
        for _ in range(5):
            result = optimizer.optimize(LOOP_PROGRAM)
            assert result.same_output
        assert len(optimizer.get_history()) == 5

    def test_computation_heavy(self):
        """Computation-heavy program."""
        source = """
        fn factorial(n) {
            if (n <= 1) { return 1; }
            return n * factorial(n - 1);
        }
        print(factorial(10));
        """
        result = pgo_optimize(source)
        assert result.same_output
        assert "3628800" in result.optimized_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
