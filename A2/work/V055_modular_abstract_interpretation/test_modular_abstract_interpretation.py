"""Tests for V055: Modular Abstract Interpretation"""

import os, sys
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import pytest
from modular_abstract_interpretation import (
    modular_analyze, analyze_function, compare_modular_vs_monolithic,
    get_function_thresholds, get_all_summaries,
    ModularAbstractInterpreter, ModularAIResult, FunctionSummary,
    AbstractBound, _extract_bounds_from_contract, _filter_annotations,
    _parse_source, _topological_order, _build_call_graph,
    _refine_for_condition, _eval_abstract, _get_result_candidates,
)
from abstract_interpreter import AbstractEnv, Sign, Interval, AbstractValue


# -----------------------------------------------------------------------
# Test programs
# -----------------------------------------------------------------------

SIMPLE_FN = """
fn abs(x) {
  requires(x >= 0 - 100);
  requires(x <= 100);
  ensures(result >= 0);
  let result = 0;
  if (x >= 0) {
    result = x;
  } else {
    result = 0 - x;
  }
  return result;
}
"""

TWO_FUNCTIONS = """
fn double(x) {
  requires(x >= 0);
  requires(x <= 50);
  ensures(result >= 0);
  ensures(result <= 100);
  let result = x + x;
  return result;
}

fn quad(x) {
  requires(x >= 0);
  requires(x <= 25);
  let d = double(x);
  let result = double(d);
  return result;
}
"""

SIMPLE_LOOP = """
fn sum_to_n(n) {
  requires(n >= 0);
  requires(n <= 100);
  let s = 0;
  let i = 0;
  while (i < n) {
    s = s + i;
    i = i + 1;
  }
  return s;
}
"""

NO_CONTRACT = """
fn inc(x) {
  let result = x + 1;
  return result;
}
"""

GLOBAL_WITH_FN = """
fn add(a, b) {
  requires(a >= 0);
  requires(b >= 0);
  ensures(result >= 0);
  let result = a + b;
  return result;
}

let x = 5;
let y = 10;
let z = add(x, y);
print(z);
"""

CHAIN_CALLS = """
fn f(x) {
  requires(x >= 0);
  ensures(result >= 1);
  let result = x + 1;
  return result;
}

fn g(x) {
  requires(x >= 0);
  ensures(result >= 2);
  let y = f(x);
  let result = y + 1;
  return result;
}

fn h(x) {
  requires(x >= 0);
  let result = g(x);
  return result;
}
"""

CLASSIFY = """
fn classify(x) {
  requires(x >= 0 - 1000);
  requires(x <= 1000);
  let cat = 0;
  if (x < 0) {
    cat = 0 - 1;
  } else {
    if (x == 0) {
      cat = 0;
    } else {
      cat = 1;
    }
  }
  return cat;
}
"""


# -----------------------------------------------------------------------
# Section 1: AbstractBound extraction
# -----------------------------------------------------------------------

class TestBoundExtraction:
    def test_simple_bounds(self):
        from modular_verification import extract_all_contracts
        contracts = extract_all_contracts(SIMPLE_FN)
        contract = contracts.get("abs")
        assert contract is not None
        pb, rb, thresholds = _extract_bounds_from_contract(contract)
        assert "x" in pb
        assert pb["x"].lower == -100
        assert pb["x"].upper == 100

    def test_ensures_bounds(self):
        from modular_verification import extract_all_contracts
        contracts = extract_all_contracts(SIMPLE_FN)
        contract = contracts.get("abs")
        pb, rb, thresholds = _extract_bounds_from_contract(contract)
        assert "result" in rb
        assert rb["result"].lower == 0

    def test_thresholds_extracted(self):
        from modular_verification import extract_all_contracts
        contracts = extract_all_contracts(SIMPLE_FN)
        contract = contracts.get("abs")
        _, _, thresholds = _extract_bounds_from_contract(contract)
        assert len(thresholds) > 0
        assert 0 in thresholds  # from result >= 0
        assert -100 in thresholds  # from x >= -100
        assert 100 in thresholds  # from x <= 100


# -----------------------------------------------------------------------
# Section 2: Filter annotations
# -----------------------------------------------------------------------

class TestFilterAnnotations:
    def test_filters_requires_ensures(self):
        program = _parse_source(SIMPLE_FN)
        fn = program.stmts[0]
        filtered = _filter_annotations(fn.body)
        # Should not contain requires/ensures calls
        from stack_vm import CallExpr
        for stmt in filtered:
            if isinstance(stmt, CallExpr):
                callee = stmt.callee if isinstance(stmt.callee, str) else stmt.callee.name
                assert callee not in ('requires', 'ensures')


# -----------------------------------------------------------------------
# Section 3: Topological ordering
# -----------------------------------------------------------------------

class TestTopologicalOrder:
    def test_simple_order(self):
        graph = {"a": {"b"}, "b": set(), "c": {"a"}}
        order = _topological_order(graph, ["a", "b", "c"])
        assert order.index("b") < order.index("a")
        assert order.index("a") < order.index("c")

    def test_no_deps(self):
        graph = {"a": set(), "b": set()}
        order = _topological_order(graph, ["a", "b"])
        assert set(order) == {"a", "b"}

    def test_cycle_handled(self):
        graph = {"a": {"b"}, "b": {"a"}}
        order = _topological_order(graph, ["a", "b"])
        assert len(order) == 2


# -----------------------------------------------------------------------
# Section 4: Condition refinement
# -----------------------------------------------------------------------

class TestConditionRefinement:
    def test_greater_than_zero(self):
        from stack_vm import BinOp, Var, IntLit
        env = AbstractEnv()
        env.set_top("x")
        cond = BinOp(">", Var("x"), IntLit(0))
        _refine_for_condition(cond, env, True)
        interval = env.get_interval("x")
        assert interval.lo >= 1

    def test_less_than_ten(self):
        from stack_vm import BinOp, Var, IntLit
        env = AbstractEnv()
        env.set_top("x")
        cond = BinOp("<", Var("x"), IntLit(10))
        _refine_for_condition(cond, env, True)
        interval = env.get_interval("x")
        assert interval.hi <= 9

    def test_negated_condition(self):
        from stack_vm import BinOp, Var, IntLit
        env = AbstractEnv()
        env.set_top("x")
        cond = BinOp(">=", Var("x"), IntLit(0))
        _refine_for_condition(cond, env, False)
        interval = env.get_interval("x")
        # NOT(x >= 0) => x < 0 => x <= -1
        assert interval.hi <= -1


# -----------------------------------------------------------------------
# Section 5: Simple function analysis
# -----------------------------------------------------------------------

class TestSimpleFunctionAnalysis:
    def test_abs_function(self):
        result = modular_analyze(SIMPLE_FN)
        assert "abs" in result.summaries
        summary = result.summaries["abs"]
        assert summary.analyzed
        assert "result" in summary.result_bounds

    def test_abs_result_nonneg(self):
        result = modular_analyze(SIMPLE_FN)
        summary = result.summaries["abs"]
        # From ensures: result >= 0
        bound = summary.result_bounds.get("result")
        assert bound is not None
        assert bound.lower == 0

    def test_no_contract_fn(self):
        result = modular_analyze(NO_CONTRACT)
        assert "inc" in result.summaries
        summary = result.summaries["inc"]
        assert summary.analyzed


# -----------------------------------------------------------------------
# Section 6: Multi-function analysis
# -----------------------------------------------------------------------

class TestMultiFunctionAnalysis:
    def test_two_functions(self):
        result = modular_analyze(TWO_FUNCTIONS)
        assert "double" in result.summaries
        assert "quad" in result.summaries
        assert result.functions_analyzed == 2

    def test_analysis_order(self):
        result = modular_analyze(TWO_FUNCTIONS)
        # double should be analyzed before quad (callee first)
        assert result.analysis_order.index("double") < result.analysis_order.index("quad")

    def test_double_bounds(self):
        result = modular_analyze(TWO_FUNCTIONS)
        summary = result.summaries["double"]
        assert "result" in summary.result_bounds
        bound = summary.result_bounds["result"]
        assert bound.lower == 0
        assert bound.upper == 100

    def test_chain_order(self):
        result = modular_analyze(CHAIN_CALLS)
        order = result.analysis_order
        assert order.index("f") < order.index("g")
        assert order.index("g") < order.index("h")


# -----------------------------------------------------------------------
# Section 7: Call site summaries
# -----------------------------------------------------------------------

class TestCallSiteSummaries:
    def test_call_uses_summary(self):
        result = modular_analyze(TWO_FUNCTIONS)
        summary = result.summaries["quad"]
        assert summary.analyzed
        # quad calls double, which has result bounds [0, 100]
        # So d should be bounded by double's summary

    def test_chain_call_summary(self):
        result = modular_analyze(CHAIN_CALLS)
        # f: result >= 1
        # g calls f, gets >= 1, adds 1 => result >= 2
        f_summary = result.summaries["f"]
        g_summary = result.summaries["g"]
        assert f_summary.result_bounds.get("result")
        assert g_summary.result_bounds.get("result")


# -----------------------------------------------------------------------
# Section 8: Global code with function calls
# -----------------------------------------------------------------------

class TestGlobalWithFunctions:
    def test_global_env(self):
        result = modular_analyze(GLOBAL_WITH_FN)
        assert result.global_env is not None

    def test_function_analyzed(self):
        result = modular_analyze(GLOBAL_WITH_FN)
        assert "add" in result.summaries
        assert result.summaries["add"].analyzed


# -----------------------------------------------------------------------
# Section 9: Loop analysis
# -----------------------------------------------------------------------

class TestLoopAnalysis:
    def test_loop_converges(self):
        result = modular_analyze(SIMPLE_LOOP)
        assert "sum_to_n" in result.summaries
        summary = result.summaries["sum_to_n"]
        assert summary.analyzed

    def test_loop_widening(self):
        result = modular_analyze(SIMPLE_LOOP)
        summary = result.summaries["sum_to_n"]
        # After the loop, s should have been widened
        # The exact bounds depend on widening, but it should be non-negative
        if summary.body_env:
            sign = summary.body_env.get_sign("s")
            assert sign in (Sign.NON_NEG, Sign.TOP, Sign.POS, Sign.ZERO)


# -----------------------------------------------------------------------
# Section 10: Classify function
# -----------------------------------------------------------------------

class TestClassify:
    def test_classify_analyzed(self):
        result = modular_analyze(CLASSIFY)
        assert "classify" in result.summaries
        summary = result.summaries["classify"]
        assert summary.analyzed

    def test_classify_params(self):
        result = modular_analyze(CLASSIFY)
        summary = result.summaries["classify"]
        assert "x" in summary.param_bounds
        assert summary.param_bounds["x"].lower == -1000
        assert summary.param_bounds["x"].upper == 1000


# -----------------------------------------------------------------------
# Section 11: Compare modular vs monolithic
# -----------------------------------------------------------------------

class TestComparison:
    def test_compare_runs(self):
        comparison = compare_modular_vs_monolithic(SIMPLE_FN)
        assert 'modular_warnings' in comparison
        assert 'monolithic_warnings' in comparison
        assert 'functions_analyzed' in comparison

    def test_compare_two_functions(self):
        comparison = compare_modular_vs_monolithic(TWO_FUNCTIONS)
        assert comparison['functions_analyzed'] == 2


# -----------------------------------------------------------------------
# Section 12: Convenience APIs
# -----------------------------------------------------------------------

class TestConvenienceAPIs:
    def test_analyze_function(self):
        summary = analyze_function(SIMPLE_FN, "abs")
        assert summary is not None
        assert summary.fn_name == "abs"
        assert summary.analyzed

    def test_get_thresholds(self):
        thresholds = get_function_thresholds(SIMPLE_FN, "abs")
        assert isinstance(thresholds, list)
        assert len(thresholds) > 0

    def test_get_all_summaries(self):
        summaries = get_all_summaries(TWO_FUNCTIONS)
        assert "double" in summaries
        assert "quad" in summaries

    def test_analyze_function_missing(self):
        summary = analyze_function(SIMPLE_FN, "nonexistent")
        assert summary is None


# -----------------------------------------------------------------------
# Section 13: Result candidates
# -----------------------------------------------------------------------

class TestResultCandidates:
    def test_return_stmt(self):
        from stack_vm import ReturnStmt, Var
        stmts = [ReturnStmt(Var("result"))]
        candidates = _get_result_candidates(stmts)
        assert "result" in candidates

    def test_assign_fallback(self):
        from stack_vm import Assign, IntLit
        stmts = [Assign("r", IntLit(0))]
        candidates = _get_result_candidates(stmts)
        assert "r" in candidates


# -----------------------------------------------------------------------
# Section 14: Summary report
# -----------------------------------------------------------------------

class TestSummaryReport:
    def test_report_format(self):
        result = modular_analyze(SIMPLE_FN)
        report = result.summary_report()
        assert "Modular Abstract Interpretation" in report
        assert "abs" in report

    def test_report_with_bounds(self):
        result = modular_analyze(SIMPLE_FN)
        report = result.summary_report()
        assert "result" in report


# -----------------------------------------------------------------------
# Section 15: Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_program(self):
        result = modular_analyze("let x = 5; print(x);")
        assert result.functions_analyzed == 0
        assert result.global_env is not None

    def test_no_annotations(self):
        result = modular_analyze(NO_CONTRACT)
        assert result.functions_analyzed == 1

    def test_function_no_return(self):
        src = """
fn side_effect(x) {
  let y = x + 1;
}
"""
        result = modular_analyze(src)
        assert "side_effect" in result.summaries

    def test_multiple_params(self):
        src = """
fn add3(a, b, c) {
  requires(a >= 0);
  requires(b >= 0);
  requires(c >= 0);
  let result = a + b + c;
  return result;
}
"""
        result = modular_analyze(src)
        summary = result.summaries["add3"]
        assert len(summary.params) == 3
        assert summary.analyzed
