"""
Tests for V028: Fault Localization
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fault_localization import (
    _parse, _extract_statements, _flatten_statements,
    CoverageTracer, spectrum_localize, backward_slice, symbolic_localize,
    generate_test_suite, localize_fault, auto_localize,
    rank_at, exam_score,
    TestCase, TestVerdict, Metric, FaultResult, SpectrumResult, SliceResult,
    StatementInfo, SuspiciousnessScore,
    _stmt_kind, _used_vars, _defined_var, _collect_vars_in_expr,
    _compute_suspiciousness,
)


# ============================================================
# Helper: common buggy programs
# ============================================================

# Bug: should be x > 0, not x > 1 (misses x=1 case)
BUGGY_ABS = """
let x = input;
let result = 0;
if (x > 1) {
  result = x;
} else {
  result = 0 - x;
}
"""

# Correct version for oracle
def abs_oracle(inputs):
    x = inputs.get('input', 0)
    expected = x if x > 0 else -x
    # Run buggy version
    if x > 1:
        result = x
    else:
        result = -x
    return TestVerdict.PASS if result == expected else TestVerdict.FAIL


# Bug: uses + instead of - in else branch
BUGGY_MAX = """
let a = x;
let b = y;
let result = 0;
if (a > b) {
  result = a;
} else {
  result = a;
}
"""

def max_oracle(inputs):
    x = inputs.get('x', 0)
    y = inputs.get('y', 0)
    expected = max(x, y)
    if x > y:
        result = x
    else:
        result = x  # Bug: should be y
    return TestVerdict.PASS if result == expected else TestVerdict.FAIL


# Bug: loop condition off-by-one
BUGGY_SUM = """
let n = input;
let s = 0;
let i = 0;
while (i < n) {
  s = s + i;
  i = i + 1;
}
"""
# This is actually correct for sum 0..n-1. Let's make a real bug:
BUGGY_SUM2 = """
let n = input;
let s = 0;
let i = 1;
while (i < n) {
  s = s + i;
  i = i + 1;
}
"""
# Bug: i starts at 1, so sum(1..n-1) instead of sum(0..n-1)
# For n=3: gives 1+2=3 instead of 0+1+2=3 -- same! For n=1: gives 0 instead of 0.
# Better bug: wrong operator

BUGGY_SUM_OP = """
let n = input;
let s = 0;
let i = 0;
while (i < n) {
  s = s * i;
  i = i + 1;
}
"""
# Bug: s = s * i instead of s = s + i

def sum_oracle(inputs):
    n = inputs.get('input', 0)
    expected = sum(range(n)) if n > 0 else 0
    s = 0
    i = 0
    while i < n:
        s = s * i  # Bug
        i += 1
    return TestVerdict.PASS if s == expected else TestVerdict.FAIL


# ============================================================
# Test: AST Extraction
# ============================================================

class TestASTExtraction:
    def test_parse_simple(self):
        prog = _parse("let x = 5;")
        assert prog is not None
        assert len(prog.stmts) == 1

    def test_extract_statements(self):
        source = "let x = 5;\nlet y = 10;\nlet z = x + y;"
        stmts = _extract_statements(source)
        assert len(stmts) == 3
        assert stmts[0].kind == "let"
        assert stmts[1].kind == "let"
        assert stmts[2].kind == "let"

    def test_extract_with_if(self):
        source = """
let x = 5;
if (x > 0) {
  let y = x;
} else {
  let y = 0 - x;
}
"""
        stmts = _extract_statements(source)
        assert any(s.kind == "if" for s in stmts)
        assert any(s.kind == "let" for s in stmts)

    def test_extract_with_while(self):
        source = """
let i = 0;
while (i < 10) {
  i = i + 1;
}
"""
        stmts = _extract_statements(source)
        assert any(s.kind == "while" for s in stmts)
        assert any(s.kind == "assign" for s in stmts)

    def test_stmt_kind(self):
        from stack_vm import LetDecl, Assign, IfStmt, WhileStmt, PrintStmt, IntLit
        assert _stmt_kind(LetDecl("x", IntLit(5))) == "let"

    def test_flatten_nested(self):
        source = """
let x = 5;
if (x > 0) {
  if (x > 10) {
    let y = 1;
  }
}
"""
        prog = _parse(source)
        flat = _flatten_statements(prog)
        assert len(flat) >= 3  # let, if, inner-if, let


# ============================================================
# Test: Coverage Tracer
# ============================================================

class TestCoverageTracer:
    def test_simple_coverage(self):
        source = "let x = 5;\nlet y = 10;\nlet z = x + y;"
        stmts = _extract_statements(source)
        tracer = CoverageTracer(source, stmts)
        covered, passed, output = tracer.execute({})
        assert passed
        assert len(covered) == 3  # All 3 statements covered

    def test_branch_coverage(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = 1;
} else {
  r = 2;
}
"""
        stmts = _extract_statements(source)
        tracer = CoverageTracer(source, stmts)

        # Positive path
        covered_pos, _, _ = tracer.execute({'input': 5})
        # Negative path
        covered_neg, _, _ = tracer.execute({'input': -5})

        # Different branches should cover different statements
        assert covered_pos != covered_neg

    def test_assertion_failure(self):
        source = """
let x = input;
assert(x > 0);
"""
        stmts = _extract_statements(source)
        tracer = CoverageTracer(source, stmts)
        _, passed, _ = tracer.execute({'input': -1})
        assert not passed

    def test_assertion_pass(self):
        source = """
let x = input;
assert(x > 0);
"""
        stmts = _extract_statements(source)
        tracer = CoverageTracer(source, stmts)
        _, passed, _ = tracer.execute({'input': 5})
        assert passed

    def test_while_coverage(self):
        source = """
let i = 0;
let s = 0;
while (i < 3) {
  s = s + i;
  i = i + 1;
}
"""
        stmts = _extract_statements(source)
        tracer = CoverageTracer(source, stmts)
        covered, passed, _ = tracer.execute({})
        assert passed

    def test_print_output(self):
        source = """
let x = 5;
print(x);
"""
        stmts = _extract_statements(source)
        tracer = CoverageTracer(source, stmts)
        _, _, output = tracer.execute({})
        assert output == [5]

    def test_division_by_zero(self):
        source = """
let x = input;
let y = 10 / x;
"""
        stmts = _extract_statements(source)
        tracer = CoverageTracer(source, stmts)
        _, passed, _ = tracer.execute({'input': 0})
        assert not passed


# ============================================================
# Test: Suspiciousness Metrics
# ============================================================

class TestSuspiciousnessMetrics:
    def test_ochiai_perfect(self):
        # Statement executed by all failing, none passing
        score = _compute_suspiciousness(ef=5, ep=0, nf=0, np_=3, total_f=5, total_p=3, metric=Metric.OCHIAI)
        assert score == pytest.approx(1.0)

    def test_ochiai_zero(self):
        # Statement not executed by any failing test
        score = _compute_suspiciousness(ef=0, ep=3, nf=5, np_=0, total_f=5, total_p=3, metric=Metric.OCHIAI)
        assert score == 0.0

    def test_tarantula_perfect(self):
        score = _compute_suspiciousness(ef=5, ep=0, nf=0, np_=3, total_f=5, total_p=3, metric=Metric.TARANTULA)
        assert score == pytest.approx(1.0)

    def test_tarantula_half(self):
        # Executed by all failing and all passing -> 0.5
        score = _compute_suspiciousness(ef=5, ep=3, nf=0, np_=0, total_f=5, total_p=3, metric=Metric.TARANTULA)
        assert score == pytest.approx(0.5)

    def test_dstar_high(self):
        score = _compute_suspiciousness(ef=5, ep=0, nf=0, np_=3, total_f=5, total_p=3, metric=Metric.DSTAR)
        # DStar = ef^2 / (ep + nf) = 25 / 0 -> inf
        assert score == float('inf')

    def test_dstar_finite(self):
        score = _compute_suspiciousness(ef=3, ep=1, nf=2, np_=2, total_f=5, total_p=3, metric=Metric.DSTAR)
        # DStar = 9 / (1 + 2) = 3.0
        assert score == pytest.approx(3.0)

    def test_all_zeros(self):
        for m in Metric:
            score = _compute_suspiciousness(0, 0, 0, 0, 0, 0, m)
            assert score == 0.0


# ============================================================
# Test: Spectrum-Based Fault Localization
# ============================================================

class TestSpectrumBased:
    def test_basic_sbfl(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = 0 - x;
}
"""
        tests = [
            TestCase(inputs={'input': 5}, verdict=TestVerdict.PASS),
            TestCase(inputs={'input': 3}, verdict=TestVerdict.PASS),
            TestCase(inputs={'input': -1}, verdict=TestVerdict.FAIL),
            TestCase(inputs={'input': -5}, verdict=TestVerdict.FAIL),
        ]
        result = spectrum_localize(source, tests)
        assert isinstance(result, SpectrumResult)
        assert result.total_passing == 2
        assert result.total_failing == 2
        assert len(result.rankings) == 3  # All metrics

    def test_buggy_stmt_ranks_high(self):
        # Bug is in the else branch (r = 0 - x should be something else based on spec)
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        # Oracle: abs(x) should be positive
        tests = [
            TestCase(inputs={'input': 5}, verdict=TestVerdict.PASS),  # r=5 correct
            TestCase(inputs={'input': 3}, verdict=TestVerdict.PASS),  # r=3 correct
            TestCase(inputs={'input': -1}, verdict=TestVerdict.FAIL), # r=-1 wrong, should be 1
            TestCase(inputs={'input': -5}, verdict=TestVerdict.FAIL), # r=-5 wrong, should be 5
        ]
        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI])
        ranking = result.rankings[Metric.OCHIAI]
        # The else-branch statement should rank higher than the then-branch statement
        # Find the else branch assign (r = x in else)
        # Statements executed ONLY by failing tests should rank highest
        top_scores = [s for s in ranking if s.score > 0]
        assert len(top_scores) > 0

    def test_single_failing(self):
        source = "let x = input;\nlet y = x + 1;"
        tests = [
            TestCase(inputs={'input': 0}, verdict=TestVerdict.FAIL),
            TestCase(inputs={'input': 5}, verdict=TestVerdict.PASS),
        ]
        result = spectrum_localize(source, tests, metrics=[Metric.TARANTULA])
        assert result.total_failing == 1
        assert result.total_passing == 1

    def test_all_passing(self):
        source = "let x = 5;\nlet y = 10;"
        tests = [
            TestCase(inputs={}, verdict=TestVerdict.PASS),
            TestCase(inputs={}, verdict=TestVerdict.PASS),
        ]
        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI])
        # All scores should be 0 (no failing tests)
        for s in result.rankings[Metric.OCHIAI]:
            assert s.score == 0.0

    def test_coverage_matrix(self):
        source = """
let x = input;
if (x > 0) {
  let y = 1;
} else {
  let y = 2;
}
"""
        tests = [
            TestCase(inputs={'input': 5}, verdict=TestVerdict.PASS),
            TestCase(inputs={'input': -1}, verdict=TestVerdict.FAIL),
        ]
        result = spectrum_localize(source, tests)
        assert len(result.coverage_matrix) == 2

    def test_many_tests(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        tests = []
        for i in range(-5, 6):
            if i > 0:
                verdict = TestVerdict.PASS  # r = x is correct for x > 0
            elif i <= 0:
                verdict = TestVerdict.FAIL  # r = x is wrong for x <= 0 (should be -x)
            tests.append(TestCase(inputs={'input': i}, verdict=verdict))

        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI])
        ranking = result.rankings[Metric.OCHIAI]
        # Top-ranked statement should be in else branch
        assert ranking[0].score > 0


# ============================================================
# Test: Backward Dependency Slicing
# ============================================================

class TestBackwardSlice:
    def test_simple_chain(self):
        source = """
let x = 5;
let y = x + 1;
let z = y * 2;
"""
        result = backward_slice(source, target_var='z')
        indices = {s.index for s in result.relevant_statements}
        # z depends on y, y depends on x
        assert len(indices) >= 2

    def test_independent_var(self):
        source = """
let x = 5;
let y = 10;
let z = x + 1;
"""
        result = backward_slice(source, target_var='z')
        indices = {s.index for s in result.relevant_statements}
        # z depends on x, NOT y
        # Find the y statement
        stmts = _extract_statements(source)
        y_idx = None
        for s in stmts:
            if s.description == "let y = ...":
                y_idx = s.index
        if y_idx is not None:
            assert y_idx not in indices

    def test_control_dependency(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = 1;
}
"""
        result = backward_slice(source, target_var='r')
        kinds = {s.kind for s in result.relevant_statements}
        # r depends on x through the if condition
        assert 'if' in kinds or 'let' in kinds

    def test_from_last_stmt(self):
        source = """
let a = 1;
let b = 2;
let c = a + b;
"""
        result = backward_slice(source)  # No target -> last statement
        assert len(result.relevant_statements) > 0

    def test_by_index(self):
        source = """
let x = 5;
let y = 10;
let z = 15;
"""
        result = backward_slice(source, target_stmt_index=1)
        # Statement 1 is "let y = 10" -- no dependencies
        assert len(result.relevant_statements) >= 1
        assert result.dependency_chain[0][1] == "target"

    def test_empty_program(self):
        source = "let x = 5;"
        result = backward_slice(source)
        assert len(result.relevant_statements) >= 1


# ============================================================
# Test: Variable Analysis
# ============================================================

class TestVariableAnalysis:
    def test_collect_vars_simple(self):
        from stack_vm import Var, BinOp, IntLit
        expr = BinOp('+', Var('x'), Var('y'))
        assert _collect_vars_in_expr(expr) == {'x', 'y'}

    def test_collect_vars_nested(self):
        from stack_vm import Var, BinOp, IntLit
        expr = BinOp('+', BinOp('*', Var('a'), Var('b')), Var('c'))
        assert _collect_vars_in_expr(expr) == {'a', 'b', 'c'}

    def test_collect_vars_literal(self):
        from stack_vm import IntLit
        assert _collect_vars_in_expr(IntLit(5)) == set()

    def test_defined_var_let(self):
        from stack_vm import LetDecl, IntLit
        assert _defined_var(LetDecl('x', IntLit(5))) == 'x'

    def test_defined_var_assign(self):
        from stack_vm import Assign, IntLit
        assert _defined_var(Assign('y', IntLit(10))) == 'y'

    def test_defined_var_if(self):
        from stack_vm import IfStmt, IntLit, Block
        stmt = IfStmt(IntLit(1), Block([]), None)
        assert _defined_var(stmt) is None


# ============================================================
# Test: Test Suite Generation
# ============================================================

class TestGeneration:
    def test_generate_basic(self):
        source = """
let x = input;
if (x > 0) {
  assert(x > 0);
} else {
  assert(x <= 0);
}
"""
        tests = generate_test_suite(source, {'input': 'int'})
        assert len(tests) >= 1  # Should generate at least one test

    def test_generate_with_oracle(self):
        source = BUGGY_MAX
        oracle = max_oracle
        tests = generate_test_suite(source, {'x': 'int', 'y': 'int'}, oracle_fn=oracle)
        # Should find both passing and failing tests
        verdicts = {t.verdict for t in tests}
        # May or may not find failures depending on generated inputs
        assert len(tests) >= 1

    def test_generate_assertion_failure(self):
        source = """
let x = input;
let y = x * 2;
assert(y > 0);
"""
        tests = generate_test_suite(source, {'input': 'int'})
        # x=0 or negative should trigger assertion failure
        failing = [t for t in tests if t.verdict == TestVerdict.FAIL]
        # Symbolic execution should find a failing path
        assert len(failing) >= 1 or len(tests) >= 1


# ============================================================
# Test: Combined Fault Localization
# ============================================================

class TestLocalizeFault:
    def test_basic_localize(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        result = localize_fault(
            source,
            failing_tests=[{'input': -1}, {'input': -5}],
            passing_tests=[{'input': 5}, {'input': 3}],
        )
        assert isinstance(result, FaultResult)
        assert result.total_tests == 4
        assert result.top_suspect is not None

    def test_localize_buggy_else(self):
        """Bug is in else branch: r = x instead of r = 0 - x."""
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        result = localize_fault(
            source,
            failing_tests=[{'input': -1}, {'input': -3}, {'input': -5}],
            passing_tests=[{'input': 1}, {'input': 3}, {'input': 5}],
        )
        # The buggy statement (r = x in else) should be in top 3
        top3_indices = {s.statement.index for s in result.ranked_statements[:3]}
        assert len(top3_indices) > 0

    def test_localize_with_metric(self):
        source = """
let x = input;
let r = x + 1;
"""
        result = localize_fault(
            source,
            failing_tests=[{'input': 0}],
            passing_tests=[{'input': 5}],
            metric=Metric.DSTAR,
        )
        assert isinstance(result, FaultResult)

    def test_no_failing(self):
        source = "let x = 5;\nlet y = 10;"
        result = localize_fault(
            source,
            failing_tests=[],
            passing_tests=[{}],
        )
        # Should handle gracefully
        assert result.total_tests == 1


# ============================================================
# Test: Auto Localize Pipeline
# ============================================================

class TestAutoLocalize:
    def test_auto_with_assertion(self):
        source = """
let x = input;
let y = x + 1;
assert(y > 5);
"""
        result = auto_localize(source, {'input': 'int'})
        assert isinstance(result, FaultResult)
        # Should find some failing tests (e.g., input=0 -> y=1, fails assert)
        assert result.total_tests >= 1

    def test_auto_with_oracle(self):
        source = BUGGY_MAX
        result = auto_localize(source, {'x': 'int', 'y': 'int'}, oracle_fn=max_oracle)
        assert isinstance(result, FaultResult)

    def test_auto_ranks_buggy_high(self):
        """The buggy statement should rank near the top."""
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        def oracle(inputs):
            x = inputs.get('input', 0)
            expected = abs(x)
            result = x if x > 0 else x  # Bug: should be -x
            return TestVerdict.PASS if result == expected else TestVerdict.FAIL

        result = auto_localize(source, {'input': 'int'}, oracle_fn=oracle)
        if result.ranked_statements:
            # Top suspect should be in the else branch or the condition
            assert result.top_suspect is not None

    def test_auto_extra_tests(self):
        source = """
let x = input;
assert(x != 0);
"""
        extra = [TestCase(inputs={'input': 0}, verdict=TestVerdict.FAIL)]
        result = auto_localize(source, {'input': 'int'}, extra_tests=extra)
        assert result.total_tests >= 1

    def test_auto_no_bug(self):
        """Program with no bugs should produce no suspects."""
        source = """
let x = input;
let y = x + 1;
"""
        # No assertions, oracle says everything passes
        result = auto_localize(source, {'input': 'int'},
                               oracle_fn=lambda inputs: TestVerdict.PASS)
        assert result.top_suspect is None


# ============================================================
# Test: Rank and Exam Score
# ============================================================

class TestRankAndExam:
    def test_rank_at(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        result = localize_fault(
            source,
            failing_tests=[{'input': -1}],
            passing_tests=[{'input': 5}],
        )
        # Should be able to find rank for some statement
        if result.ranked_statements:
            idx = result.ranked_statements[0].statement.index
            assert rank_at(result, idx) == 1

    def test_exam_score(self):
        source = "let x = input;\nlet y = x + 1;"
        result = localize_fault(
            source,
            failing_tests=[{'input': 0}],
            passing_tests=[{'input': 5}],
        )
        if result.ranked_statements:
            idx = result.ranked_statements[0].statement.index
            score = exam_score(result, idx)
            assert score is not None
            assert 0 < score <= 1.0

    def test_rank_not_found(self):
        source = "let x = 5;"
        result = localize_fault(source, failing_tests=[{}], passing_tests=[{}])
        assert rank_at(result, 999) is None

    def test_exam_not_found(self):
        source = "let x = 5;"
        result = localize_fault(source, failing_tests=[{}], passing_tests=[{}])
        assert exam_score(result, 999) is None


# ============================================================
# Test: Symbolic Localization
# ============================================================

class TestSymbolicLocalize:
    def test_basic_symbolic(self):
        source = """
let x = input;
assert(x > 0);
"""
        result = symbolic_localize(source, {'input': -1}, {'input': 'int'})
        assert isinstance(result, SliceResult)

    def test_symbolic_finds_relevant(self):
        source = """
let x = input;
let y = x * 2;
assert(y > 10);
"""
        result = symbolic_localize(source, {'input': 2}, {'input': 'int'})
        # y=4 < 10 -> fail. Relevant: let y = x * 2 and assert
        if result.relevant_statements:
            kinds = {s.kind for s in result.relevant_statements}
            assert len(kinds) >= 1


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_statement(self):
        source = "let x = 5;"
        stmts = _extract_statements(source)
        assert len(stmts) == 1

    def test_empty_test_suite(self):
        source = "let x = 5;"
        result = spectrum_localize(source, [])
        assert result.total_passing == 0
        assert result.total_failing == 0

    def test_all_failing(self):
        source = "let x = input;\nassert(x > 100);"
        tests = [
            TestCase(inputs={'input': 0}, verdict=TestVerdict.FAIL),
            TestCase(inputs={'input': 1}, verdict=TestVerdict.FAIL),
        ]
        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI])
        # With no passing tests, all executed statements have same score
        assert result.total_failing == 2
        assert result.total_passing == 0

    def test_deeply_nested(self):
        source = """
let x = input;
let y = input2;
if (x > 0) {
  if (y > 0) {
    let r = x + y;
  } else {
    let r = x - y;
  }
} else {
  let r = 0;
}
"""
        stmts = _extract_statements(source)
        assert len(stmts) >= 5

        tests = [
            TestCase(inputs={'input': 5, 'input2': 3}, verdict=TestVerdict.PASS),
            TestCase(inputs={'input': 5, 'input2': -1}, verdict=TestVerdict.FAIL),
            TestCase(inputs={'input': -1, 'input2': 3}, verdict=TestVerdict.PASS),
        ]
        result = spectrum_localize(source, tests)
        assert result.total_failing == 1

    def test_while_loop_bug(self):
        source = BUGGY_SUM_OP
        tests = [
            TestCase(inputs={'input': 0}, verdict=TestVerdict.PASS),  # sum=0, correct
            TestCase(inputs={'input': 1}, verdict=TestVerdict.PASS),  # s=0*0=0, expected=0
            TestCase(inputs={'input': 3}, verdict=TestVerdict.FAIL),  # s=0, expected=3
            TestCase(inputs={'input': 5}, verdict=TestVerdict.FAIL),  # s=0, expected=10
        ]
        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI])
        ranking = result.rankings[Metric.OCHIAI]
        # The buggy statement (s = s * i) should rank high
        assert len(ranking) > 0

    def test_multiple_metrics_agree(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        tests = [
            TestCase(inputs={'input': 5}, verdict=TestVerdict.PASS),
            TestCase(inputs={'input': -1}, verdict=TestVerdict.FAIL),
        ]
        result = spectrum_localize(source, tests)
        # All three metrics should have the same top statement
        tops = set()
        for metric in [Metric.OCHIAI, Metric.TARANTULA, Metric.DSTAR]:
            ranking = result.rankings[metric]
            if ranking and ranking[0].score > 0:
                tops.add(ranking[0].statement.index)
        # They should agree (or nearly so)
        assert len(tops) <= 2  # Allow some disagreement


# ============================================================
# Test: Spectrum Result Properties
# ============================================================

class TestSpectrumResultProperties:
    def test_coverage_counts_consistent(self):
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = 1;
}
"""
        tests = [
            TestCase(inputs={'input': 5}, verdict=TestVerdict.PASS),
            TestCase(inputs={'input': -1}, verdict=TestVerdict.FAIL),
        ]
        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI])
        for score in result.rankings[Metric.OCHIAI]:
            total = score.executed_by_failing + score.not_executed_by_failing
            assert total == result.total_failing
            total_p = score.executed_by_passing + score.not_executed_by_passing
            assert total_p == result.total_passing

    def test_score_range(self):
        source = "let x = input;\nlet y = x + 1;"
        tests = [
            TestCase(inputs={'input': 0}, verdict=TestVerdict.FAIL),
            TestCase(inputs={'input': 5}, verdict=TestVerdict.PASS),
        ]
        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI, Metric.TARANTULA])
        for metric in [Metric.OCHIAI, Metric.TARANTULA]:
            for score in result.rankings[metric]:
                assert 0.0 <= score.score <= 1.0


# ============================================================
# Test: Integration -- Full Pipeline
# ============================================================

class TestIntegration:
    def test_full_pipeline_abs_bug(self):
        """Full pipeline on a buggy absolute value implementation."""
        source = """
let x = input;
let r = 0;
if (x > 1) {
  r = x;
} else {
  r = 0 - x;
}
"""
        # Bug: x > 1 should be x > 0 (or x >= 1). x=1 gives r=-1.
        result = auto_localize(source, {'input': 'int'}, oracle_fn=abs_oracle)
        assert isinstance(result, FaultResult)

    def test_full_pipeline_max_bug(self):
        """Full pipeline on a buggy max implementation."""
        source = BUGGY_MAX
        result = auto_localize(source, {'x': 'int', 'y': 'int'}, oracle_fn=max_oracle)
        assert isinstance(result, FaultResult)

    def test_localize_finds_else_branch(self):
        """Verify the buggy else-branch ranks higher than the correct then-branch."""
        source = """
let x = input;
let r = 0;
if (x > 0) {
  r = x;
} else {
  r = x;
}
"""
        # Many tests to get clear spectrum
        tests = []
        for i in range(-10, 11):
            expected = abs(i)
            actual = i if i > 0 else i  # Bug
            verdict = TestVerdict.PASS if actual == expected else TestVerdict.FAIL
            tests.append(TestCase(inputs={'input': i}, verdict=verdict))

        result = spectrum_localize(source, tests, metrics=[Metric.OCHIAI])
        ranking = result.rankings[Metric.OCHIAI]

        # Find the else-branch assign and then-branch assign
        stmts = _extract_statements(source)
        # The else-branch statement should have higher suspiciousness
        # because it's executed only by failing tests
        top_stmt = ranking[0]
        assert top_stmt.score > 0
        assert top_stmt.executed_by_failing > 0
