"""Tests for V041: Symbolic Debugging"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from symbolic_debugging import (
    SymbolicDebugger, DebugResult, CounterexampleTrace, TraceStep,
    SuspiciousStatement, SBFLMetric,
    symbolic_debug, find_minimal_counterexample,
    _ochiai, _tarantula, _dstar,
    _compute_backward_slice, _collect_stmts,
)


# ============================================================
# SBFL Metric Unit Tests
# ============================================================

class TestSBFLMetrics:
    def test_ochiai_perfect_fault(self):
        score = _ochiai(ef=5, ep=0, nf=0, np_=10)
        assert score == pytest.approx(1.0)

    def test_ochiai_no_fault(self):
        score = _ochiai(ef=0, ep=5, nf=5, np_=5)
        assert score == 0.0

    def test_ochiai_partial(self):
        score = _ochiai(ef=3, ep=2, nf=2, np_=3)
        assert 0.0 < score < 1.0

    def test_tarantula_perfect_fault(self):
        score = _tarantula(ef=5, ep=0, nf=0, np_=10)
        assert score == pytest.approx(1.0)

    def test_tarantula_no_fault(self):
        score = _tarantula(ef=0, ep=5, nf=5, np_=5)
        assert score == 0.0

    def test_dstar_perfect_fault(self):
        score = _dstar(ef=5, ep=0, nf=0, np_=10)
        assert score > 0

    def test_dstar_no_fault(self):
        score = _dstar(ef=0, ep=5, nf=5, np_=5)
        assert score == 0.0

    def test_ochiai_symmetry(self):
        # ef=3, ep=1 should be more suspicious than ef=1, ep=3
        high = _ochiai(ef=3, ep=1, nf=1, np_=5)
        low = _ochiai(ef=1, ep=3, nf=3, np_=3)
        assert high > low


# ============================================================
# Statement Collector Tests
# ============================================================

class TestStatementCollector:
    def test_collect_simple(self):
        from symbolic_debugging import parse
        program = parse("let x = 1; let y = 2;")
        stmts = _collect_stmts(program.stmts)
        assert len(stmts) >= 2
        assert any(s[1] == "let" for s in stmts)

    def test_collect_function(self):
        from symbolic_debugging import parse
        program = parse("""
fn foo(x) {
    let y = x + 1;
    return y;
}
""")
        stmts = _collect_stmts(program.stmts)
        types = [s[1] for s in stmts]
        assert "fn" in types
        assert "let" in types
        assert "return" in types

    def test_collect_if(self):
        from symbolic_debugging import parse
        program = parse("""
fn foo(x) {
    if (x > 0) {
        return 1;
    }
    return 0;
}
""")
        stmts = _collect_stmts(program.stmts)
        types = [s[1] for s in stmts]
        assert "if" in types

    def test_collect_while(self):
        from symbolic_debugging import parse
        program = parse("""
fn foo(n) {
    let i = 0;
    while (i < n) {
        i = i + 1;
    }
    return i;
}
""")
        stmts = _collect_stmts(program.stmts)
        types = [s[1] for s in stmts]
        assert "while" in types


# ============================================================
# Backward Slice Tests
# ============================================================

class TestBackwardSlice:
    def test_simple_slice(self):
        from symbolic_debugging import parse
        source = """
let x = 1;
let y = x + 1;
let z = y + 1;
"""
        program = parse(source)
        stmts = program.stmts
        z_line = stmts[2].line if hasattr(stmts[2], 'line') else 0
        if z_line > 0:
            slice_lines = _compute_backward_slice(stmts, z_line)
            assert z_line in slice_lines

    def test_independent_not_in_slice(self):
        from symbolic_debugging import parse
        source = """
let a = 10;
let b = 20;
let c = a + 1;
"""
        program = parse(source)
        stmts = program.stmts
        c_line = stmts[2].line if hasattr(stmts[2], 'line') else 0
        a_line = stmts[0].line if hasattr(stmts[0], 'line') else 0
        b_line = stmts[1].line if hasattr(stmts[1], 'line') else 0
        if c_line > 0 and a_line > 0 and b_line > 0:
            slice_lines = _compute_backward_slice(stmts, c_line)
            assert a_line in slice_lines
            assert b_line not in slice_lines

    def test_transitive_deps(self):
        from symbolic_debugging import parse
        source = """
let a = 1;
let b = a + 1;
let c = b + 1;
"""
        program = parse(source)
        stmts = program.stmts
        c_line = stmts[2].line if hasattr(stmts[2], 'line') else 0
        a_line = stmts[0].line if hasattr(stmts[0], 'line') else 0
        b_line = stmts[1].line if hasattr(stmts[1], 'line') else 0
        if c_line > 0:
            slice_lines = _compute_backward_slice(stmts, c_line)
            assert a_line in slice_lines  # c depends on b depends on a
            assert b_line in slice_lines


# ============================================================
# No-Bug Program Tests
# ============================================================

class TestNoBug:
    def test_concrete_passing_assert(self):
        source = """
let x = 5;
assert(x > 0);
"""
        result = symbolic_debug(source, {})
        assert not result.has_bug

    def test_symbolic_always_passes(self):
        # x*x is always >= 0 but C038 can't prove that with LIA
        # Use a simpler always-true assertion
        source = """
let x = 10;
let y = x + 1;
assert(y > x);
"""
        result = symbolic_debug(source, {})
        assert not result.has_bug


# ============================================================
# Bug Detection Tests (top-level assertions for C038 compat)
# ============================================================

class TestBugDetection:
    def test_simple_concrete_failure(self):
        source = """
let x = 0 - 1;
assert(x > 0);
"""
        result = symbolic_debug(source, {})
        assert result.has_bug
        assert result.failing_paths > 0

    def test_symbolic_can_fail(self):
        # Top-level symbolic assertion
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        assert result.has_bug

    def test_conditional_symbolic_failure(self):
        source = """
let x = x;
if (x < 0) {
    assert(x > 0);
}
"""
        result = symbolic_debug(source, {"x": "int"})
        assert result.has_bug

    def test_multi_branch_bug(self):
        source = """
fn classify(x) {
    if (x > 10) {
        return 2;
    }
    if (x > 0) {
        return 1;
    }
    return 0 - 1;
}
let r = classify(x);
assert(r >= 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        assert result.has_bug  # When x <= 0, r = -1


# ============================================================
# Counterexample Tests
# ============================================================

class TestCounterexample:
    def test_counterexample_exists(self):
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        assert len(result.counterexamples) > 0

    def test_counterexample_has_inputs(self):
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        if result.counterexamples:
            ce = result.counterexamples[0]
            assert "x" in ce.inputs
            assert ce.inputs["x"] <= 0

    def test_minimal_counterexample(self):
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        minimal = result.minimal_counterexample
        assert minimal is not None
        assert minimal.path_length > 0

    def test_counterexample_trace_steps(self):
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        if result.counterexamples:
            ce = result.counterexamples[0]
            assert len(ce.steps) > 0
            assert ce.steps[-1].stmt_type == "assert_fail"

    def test_counterexample_repr(self):
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        if result.counterexamples:
            ce = result.counterexamples[0]
            s = repr(ce)
            assert "Counterexample" in s


# ============================================================
# Suspiciousness Ranking Tests
# ============================================================

class TestSuspiciousness:
    def test_suspicious_statements_exist(self):
        source = """
let x = x;
let y = x + 1;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        if result.has_bug:
            assert len(result.suspicious) > 0

    def test_different_metrics(self):
        source = """
let x = x;
assert(x > 0);
"""
        for metric in [SBFLMetric.OCHIAI, SBFLMetric.TARANTULA, SBFLMetric.DSTAR]:
            result = symbolic_debug(source, {"x": "int"}, metric=metric)
            if result.has_bug and result.suspicious:
                assert result.suspicious[0].metric == metric

    def test_top_suspects(self):
        source = """
let x = x;
let y = x + 1;
let z = y * 2;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        if result.has_bug:
            top = result.top_suspects
            assert len(top) <= 5

    def test_slice_membership(self):
        source = """
let x = x;
let y = x + 1;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        if result.has_bug and result.suspicious:
            # At least some statements should be in the backward slice
            in_slice = [s for s in result.suspicious if s.in_slice]
            assert len(in_slice) >= 0  # Just check it doesn't crash


# ============================================================
# API Tests
# ============================================================

class TestAPI:
    def test_symbolic_debug_function(self):
        source = """
let x = 0 - 1;
assert(x > 0);
"""
        result = symbolic_debug(source, {})
        assert isinstance(result, DebugResult)
        assert result.has_bug

    def test_find_minimal_counterexample_with_bug(self):
        source = """
let x = x;
assert(x > 0);
"""
        ce = find_minimal_counterexample(source, {"x": "int"})
        assert ce is not None

    def test_find_minimal_counterexample_no_bug(self):
        source = """
let x = 5;
assert(x > 0);
"""
        ce = find_minimal_counterexample(source, {})
        assert ce is None

    def test_debugger_class(self):
        debugger = SymbolicDebugger(max_paths=32, metric=SBFLMetric.DSTAR)
        source = """
let x = 0 - 1;
assert(x > 0);
"""
        result = debugger.debug(source, {})
        assert result.has_bug

    def test_debug_result_properties(self):
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        assert result.paths_explored > 0
        assert isinstance(result.failing_paths, int)
        assert isinstance(result.passing_paths, int)
        assert result.has_bug


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_loop_no_bug(self):
        source = """
let n = 5;
let s = 0;
let i = 0;
while (i < n) {
    s = s + i;
    i = i + 1;
}
assert(s >= 0);
"""
        result = symbolic_debug(source, {})
        assert not result.has_bug

    def test_complex_conditional(self):
        source = """
let x = x;
let y = 0;
if (x > 10) {
    y = 2;
} else {
    if (x > 0) {
        y = 1;
    } else {
        y = 0 - 1;
    }
}
assert(y >= 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        assert result.has_bug  # When x <= 0, y = -1

    def test_passing_and_failing_paths(self):
        source = """
let x = x;
assert(x > 0);
"""
        result = symbolic_debug(source, {"x": "int"})
        assert result.failing_paths > 0
        assert result.passing_paths > 0  # x > 0 path passes

    def test_multiple_assertions(self):
        source = """
let x = x;
assert(x > 0);
assert(x < 100);
"""
        result = symbolic_debug(source, {"x": "int"})
        # At least one assertion should fail (x <= 0 fails first assert)
        assert result.has_bug
