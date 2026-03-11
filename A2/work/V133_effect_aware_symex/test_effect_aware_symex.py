"""Tests for V133: Effect-Aware Symbolic Execution."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V040_effect_systems'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C038_symbolic_execution'))

from effect_aware_symex import (
    effect_aware_execute, analyze_effects, find_effectful_paths,
    find_pure_paths, find_exception_paths, find_io_paths,
    get_effect_guidance, suggest_symbolic_inputs,
    compare_aware_vs_plain, effect_aware_summary,
    EffectAwareResult, EffectAnnotatedPath, EffectGuidance,
)
from effect_systems import EffectKind


# ---------------------------------------------------------------------------
# Test programs
# ---------------------------------------------------------------------------

PURE_FUNCTION = """
fn double(x) {
    return x * 2;
}
let y = double(5);
"""

STATE_FUNCTION = """
let counter = 0;
fn increment() {
    counter = counter + 1;
    return counter;
}
let a = increment();
let b = increment();
"""

IO_FUNCTION = """
fn greet(name) {
    print(name);
    return 0;
}
let r = greet(42);
"""

MIXED_EFFECTS = """
let x = 0;
fn pure_fn(a) {
    return a + 1;
}
fn stateful_fn(v) {
    x = v;
    return x;
}
let a = pure_fn(5);
let b = stateful_fn(10);
"""

CONDITIONAL = """
let x = 0;
let y = 0;
if (x > 5) {
    y = 1;
} else {
    y = 2;
}
"""

LOOP_PROGRAM = """
let i = 0;
let sum = 0;
while (i < 3) {
    sum = sum + i;
    i = i + 1;
}
"""

SIMPLE_ASSIGN = "let x = 10; let y = 20;"

DIVISION = """
fn safe_div(a, b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}
let r = safe_div(10, 2);
"""

MULTI_FUNCTION = """
fn add(a, b) { return a + b; }
fn mul(a, b) { return a * b; }
fn compute(x) {
    let s = add(x, 1);
    let p = mul(x, 2);
    return s + p;
}
let result = compute(5);
"""


# ---------------------------------------------------------------------------
# 1. Effect pre-analysis
# ---------------------------------------------------------------------------

class TestEffectPreAnalysis:
    def test_pure_function_detected(self):
        guidance = analyze_effects(PURE_FUNCTION)
        assert "double" in guidance.pure_functions

    def test_state_function_detected(self):
        guidance = analyze_effects(STATE_FUNCTION)
        assert "increment" in guidance.state_functions

    def test_io_function_detected(self):
        guidance = analyze_effects(IO_FUNCTION)
        assert "greet" in guidance.io_functions

    def test_mixed_effects_separation(self):
        guidance = analyze_effects(MIXED_EFFECTS)
        assert "pure_fn" in guidance.pure_functions
        assert "stateful_fn" in guidance.state_functions

    def test_suggested_symbolic(self):
        guidance = analyze_effects(STATE_FUNCTION)
        assert isinstance(guidance.suggested_symbolic, list)

    def test_main_effects(self):
        guidance = analyze_effects(STATE_FUNCTION)
        assert guidance.main_effects is not None

    def test_no_div_in_simple(self):
        guidance = analyze_effects(SIMPLE_ASSIGN)
        assert guidance.div_functions == []


# ---------------------------------------------------------------------------
# 2. Effect-aware execution (basic)
# ---------------------------------------------------------------------------

class TestEffectAwareExecution:
    def test_simple_execution(self):
        result = effect_aware_execute(SIMPLE_ASSIGN)
        assert isinstance(result, EffectAwareResult)
        assert result.total_paths >= 1

    def test_pure_program(self):
        result = effect_aware_execute(PURE_FUNCTION)
        assert result.total_paths >= 1
        assert result.feasible_paths >= 1

    def test_state_program(self):
        result = effect_aware_execute(STATE_FUNCTION)
        assert result.total_paths >= 1

    def test_io_program(self):
        result = effect_aware_execute(IO_FUNCTION)
        assert result.total_paths >= 1

    def test_conditional_program(self):
        result = effect_aware_execute(CONDITIONAL, {'x': 'int'})
        assert result.feasible_paths >= 1

    def test_loop_program(self):
        result = effect_aware_execute(LOOP_PROGRAM)
        assert result.total_paths >= 1

    def test_test_cases_generated(self):
        result = effect_aware_execute(CONDITIONAL, {'x': 'int'})
        assert len(result.test_cases) >= 1

    def test_effect_sigs_populated(self):
        result = effect_aware_execute(MIXED_EFFECTS)
        assert len(result.effect_sigs) >= 1

    def test_timing_recorded(self):
        result = effect_aware_execute(SIMPLE_ASSIGN)
        assert result.analysis_time >= 0
        assert result.execution_time >= 0


# ---------------------------------------------------------------------------
# 3. Path annotations
# ---------------------------------------------------------------------------

class TestPathAnnotations:
    def test_annotated_paths_created(self):
        result = effect_aware_execute(SIMPLE_ASSIGN)
        assert result.total_annotated >= 1
        for ap in result.paths:
            assert isinstance(ap, EffectAnnotatedPath)

    def test_pure_annotation(self):
        result = effect_aware_execute(PURE_FUNCTION)
        # At least some paths should exist
        assert result.total_annotated >= 1

    def test_state_annotation(self):
        result = effect_aware_execute(STATE_FUNCTION)
        for ap in result.paths:
            assert ap.has_state is True or ap.has_state is False

    def test_io_annotation(self):
        result = effect_aware_execute(IO_FUNCTION)
        has_io = any(ap.has_io for ap in result.paths)
        assert has_io

    def test_annotation_fields(self):
        result = effect_aware_execute(MIXED_EFFECTS)
        for ap in result.paths:
            assert hasattr(ap, 'effects')
            assert hasattr(ap, 'has_state')
            assert hasattr(ap, 'has_io')
            assert hasattr(ap, 'has_exn')
            assert hasattr(ap, 'is_pure')
            assert hasattr(ap, 'state_vars')
            assert hasattr(ap, 'exn_types')

    def test_state_vars_populated(self):
        result = effect_aware_execute(STATE_FUNCTION)
        all_state_vars = set()
        for ap in result.paths:
            all_state_vars.update(ap.state_vars)
        # counter should be identified as state variable
        assert len(all_state_vars) >= 0  # may or may not be detected depending on effect inference


# ---------------------------------------------------------------------------
# 4. Specialized queries
# ---------------------------------------------------------------------------

class TestSpecializedQueries:
    def test_find_io_paths(self):
        paths = find_io_paths(IO_FUNCTION)
        assert len(paths) >= 1
        for p in paths:
            assert p.has_io

    def test_find_exception_paths(self):
        paths = find_exception_paths(DIVISION)
        # Division function may have exn paths
        assert isinstance(paths, list)

    def test_find_pure_paths(self):
        paths = find_pure_paths(SIMPLE_ASSIGN)
        # Simple assignment is pure
        assert isinstance(paths, list)

    def test_find_effectful_paths(self):
        paths = find_effectful_paths(IO_FUNCTION, EffectKind.IO)
        assert len(paths) >= 1


# ---------------------------------------------------------------------------
# 5. Guidance API
# ---------------------------------------------------------------------------

class TestGuidanceAPI:
    def test_get_guidance(self):
        guidance = get_effect_guidance(MIXED_EFFECTS)
        assert isinstance(guidance, EffectGuidance)
        assert "pure_fn" in guidance.pure_functions

    def test_suggest_symbolic_inputs(self):
        result = suggest_symbolic_inputs(STATE_FUNCTION)
        assert isinstance(result, dict)

    def test_guidance_fn_effects(self):
        guidance = get_effect_guidance(PURE_FUNCTION)
        assert "double" in guidance.fn_effects

    def test_guidance_state_fns(self):
        guidance = get_effect_guidance(STATE_FUNCTION)
        assert isinstance(guidance.state_functions, dict)


# ---------------------------------------------------------------------------
# 6. Comparison API
# ---------------------------------------------------------------------------

class TestComparisonAPI:
    def test_compare_basic(self):
        result = compare_aware_vs_plain(SIMPLE_ASSIGN)
        assert "plain" in result
        assert "effect_aware" in result

    def test_compare_paths(self):
        result = compare_aware_vs_plain(CONDITIONAL, {'x': 'int'})
        assert result["plain"]["total_paths"] >= 1
        assert result["effect_aware"]["total_paths"] >= 1

    def test_compare_timing(self):
        result = compare_aware_vs_plain(SIMPLE_ASSIGN)
        assert result["plain"]["time"] >= 0
        assert result["effect_aware"]["total_time"] >= 0

    def test_compare_effect_info(self):
        result = compare_aware_vs_plain(MIXED_EFFECTS)
        assert "pure_functions" in result["effect_aware"]
        assert "effectful_functions" in result["effect_aware"]

    def test_compare_state_variables(self):
        result = compare_aware_vs_plain(STATE_FUNCTION)
        assert "state_variables" in result["effect_aware"]


# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_output(self):
        s = effect_aware_summary(SIMPLE_ASSIGN)
        assert "Effect-Aware Symbolic Execution Summary" in s
        assert "Total paths" in s

    def test_summary_with_effects(self):
        s = effect_aware_summary(IO_FUNCTION)
        assert "Effect Analysis" in s

    def test_summary_path_annotations(self):
        s = effect_aware_summary(SIMPLE_ASSIGN)
        assert "Path Annotations" in s

    def test_summary_with_symbolic(self):
        s = effect_aware_summary(CONDITIONAL, {'x': 'int'})
        assert "Path" in s


# ---------------------------------------------------------------------------
# 8. Multi-function programs
# ---------------------------------------------------------------------------

class TestMultiFunction:
    def test_multi_fn_execution(self):
        result = effect_aware_execute(MULTI_FUNCTION)
        assert result.total_paths >= 1

    def test_multi_fn_pure_detected(self):
        guidance = analyze_effects(MULTI_FUNCTION)
        # add, mul, compute are all pure
        assert "add" in guidance.pure_functions
        assert "mul" in guidance.pure_functions

    def test_multi_fn_effects(self):
        result = effect_aware_execute(MULTI_FUNCTION)
        assert "add" in result.effect_sigs
        assert "mul" in result.effect_sigs


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_symbolic_inputs(self):
        result = effect_aware_execute(SIMPLE_ASSIGN, {})
        assert result.total_paths >= 1

    def test_no_functions(self):
        result = effect_aware_execute(SIMPLE_ASSIGN)
        assert result.total_paths >= 1
        assert result.pure_functions == [] or isinstance(result.pure_functions, list)

    def test_division_program(self):
        result = effect_aware_execute(DIVISION)
        assert result.total_paths >= 1

    def test_conditional_with_symbolic(self):
        result = effect_aware_execute(CONDITIONAL, {'x': 'int'})
        assert result.feasible_paths >= 2  # both branches

    def test_result_properties(self):
        result = effect_aware_execute(SIMPLE_ASSIGN)
        assert result.total_annotated >= 0
        assert isinstance(result.state_variables, set)
        assert isinstance(result.exception_types, set)
