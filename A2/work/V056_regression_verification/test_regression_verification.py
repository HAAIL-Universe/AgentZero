"""Tests for V056: Regression Verification"""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from regression_verification import (
    verify_regression, check_regression, check_regression_with_fuzz,
    regression_report, RegressionVerifier, compare_verification_strategies,
    RegressionVerdict, RegressionResult, FunctionVerdict, ChangeKind,
    _parse, _extract_functions, _infer_symbolic_inputs, _safe_run,
    _fuzz_for_regression
)


# ---------------------------------------------------------------------------
# Test programs
# ---------------------------------------------------------------------------

PROG_V1 = """\
fn abs(x) {
    if (x < 0) {
        return 0 - x;
    }
    return x;
}
fn double(x) {
    return x + x;
}
"""

PROG_V1_SAME = PROG_V1  # identical

PROG_V2_MODIFIED_ABS = """\
fn abs(x) {
    if (x <= 0) {
        return 0 - x;
    }
    return x;
}
fn double(x) {
    return x + x;
}
"""

PROG_V2_ADDED_FN = """\
fn abs(x) {
    if (x < 0) {
        return 0 - x;
    }
    return x;
}
fn double(x) {
    return x + x;
}
fn triple(x) {
    return x + x + x;
}
"""

PROG_V2_REMOVED_FN = """\
fn abs(x) {
    if (x < 0) {
        return 0 - x;
    }
    return x;
}
"""

PROG_V2_REGRESSION = """\
fn abs(x) {
    return x;
}
fn double(x) {
    return x + x;
}
"""

PROG_V2_REFACTORED = """\
fn abs(x) {
    if (x < 0) {
        return 0 - x;
    }
    return x;
}
fn double(x) {
    return x * 2;
}
"""

PROG_TOPLEVEL_V1 = """\
let x = 5;
let y = x + 1;
"""

PROG_TOPLEVEL_V2 = """\
let x = 5;
let y = x + 2;
"""


# ===========================================================================
# Section 1: Basic parsing helpers
# ===========================================================================

class TestParseHelpers:
    def test_parse_basic(self):
        stmts = _parse("let x = 5;")
        assert len(stmts) == 1

    def test_extract_functions(self):
        stmts = _parse(PROG_V1)
        fns = _extract_functions(stmts)
        assert "abs" in fns
        assert "double" in fns

    def test_infer_inputs_fn(self):
        inputs = _infer_symbolic_inputs(PROG_V1, "abs")
        assert "x" in inputs


# ===========================================================================
# Section 2: Safe run interpreter
# ===========================================================================

class TestSafeRun:
    def test_simple_run(self):
        result = _safe_run("let x = 5; let y = x + 1;", {})
        assert result[0] == "ok"

    def test_run_with_inputs(self):
        result = _safe_run("let x = 0; let y = x + 1;", {"x": 10})
        assert result[0] == "ok"

    def test_div_by_zero(self):
        result = _safe_run("let x = 5; let y = x / 0;", {})
        # Should not crash -- our interpreter protects
        assert result[0] == "ok"


# ===========================================================================
# Section 3: Identical programs (no changes)
# ===========================================================================

class TestIdenticalPrograms:
    def test_no_changes(self):
        result = verify_regression(PROG_V1, PROG_V1_SAME)
        assert result.is_safe
        assert result.functions_regressed == 0

    def test_all_unchanged(self):
        result = verify_regression(PROG_V1, PROG_V1_SAME)
        for fv in result.function_verdicts:
            assert fv.change_kind == ChangeKind.UNCHANGED

    def test_certs_reusable(self):
        # After first verify, second should reuse
        verifier = RegressionVerifier()
        r1 = verifier.verify(PROG_V1)
        r2 = verifier.verify(PROG_V1_SAME)
        assert r2.is_safe


# ===========================================================================
# Section 4: Function added
# ===========================================================================

class TestFunctionAdded:
    def test_added_fn_safe(self):
        result = verify_regression(PROG_V1, PROG_V2_ADDED_FN)
        assert result.is_safe

    def test_added_fn_detected(self):
        result = verify_regression(PROG_V1, PROG_V2_ADDED_FN)
        added = [fv for fv in result.function_verdicts
                 if fv.change_kind == ChangeKind.ADDED]
        assert len(added) == 1
        assert added[0].name == "triple"


# ===========================================================================
# Section 5: Function removed
# ===========================================================================

class TestFunctionRemoved:
    def test_removed_fn_safe(self):
        result = verify_regression(PROG_V1, PROG_V2_REMOVED_FN)
        assert result.is_safe

    def test_removed_fn_detected(self):
        result = verify_regression(PROG_V1, PROG_V2_REMOVED_FN)
        removed = [fv for fv in result.function_verdicts
                   if fv.change_kind == ChangeKind.REMOVED]
        assert len(removed) == 1
        assert removed[0].name == "double"


# ===========================================================================
# Section 6: Modified function (behavioral regression)
# ===========================================================================

class TestBehavioralRegression:
    def test_regression_detected(self):
        result = verify_regression(PROG_V1, PROG_V2_REGRESSION,
                                   symbolic_inputs={"x": "int"})
        assert result.verdict == RegressionVerdict.REGRESSION

    def test_regression_details(self):
        result = verify_regression(PROG_V1, PROG_V2_REGRESSION,
                                   symbolic_inputs={"x": "int"})
        regressed = result.regressions
        assert len(regressed) >= 1
        assert regressed[0].name == "abs"


# ===========================================================================
# Section 7: Modified function (safe refactoring)
# ===========================================================================

class TestSafeRefactoring:
    def test_refactored_safe(self):
        # x + x == x * 2 for all integers
        result = verify_regression(PROG_V1, PROG_V2_REFACTORED,
                                   symbolic_inputs={"x": "int"})
        assert result.is_safe or result.verdict == RegressionVerdict.IMPROVED

    def test_refactored_no_regression(self):
        result = verify_regression(PROG_V1, PROG_V2_REFACTORED,
                                   symbolic_inputs={"x": "int"})
        assert result.functions_regressed == 0


# ===========================================================================
# Section 8: Top-level changes
# ===========================================================================

class TestToplevelChanges:
    def test_toplevel_regression(self):
        result = verify_regression(PROG_TOPLEVEL_V1, PROG_TOPLEVEL_V2,
                                   symbolic_inputs={"x": "int"})
        # y changed from x+1 to x+2 -- behavioral change
        has_toplevel = any(fv.name == "<toplevel>" for fv in result.function_verdicts)
        assert has_toplevel

    def test_toplevel_identical(self):
        result = verify_regression(PROG_TOPLEVEL_V1, PROG_TOPLEVEL_V1)
        assert result.is_safe


# ===========================================================================
# Section 9: Fuzz-based regression detection
# ===========================================================================

class TestFuzzRegression:
    def test_fuzz_detects_regression(self):
        old = "fn inc(x) { return x + 1; }"
        new = "fn inc(x) { return x + 2; }"
        result = check_regression_with_fuzz(old, new,
                                            symbolic_inputs={"x": "int"},
                                            fuzz_budget=50)
        assert result.verdict == RegressionVerdict.REGRESSION

    def test_fuzz_safe_refactoring(self):
        old = "fn dbl(x) { return x + x; }"
        new = "fn dbl(x) { return x * 2; }"
        result = check_regression_with_fuzz(old, new,
                                            symbolic_inputs={"x": "int"},
                                            fuzz_budget=50)
        assert result.is_safe or result.verdict == RegressionVerdict.IMPROVED


# ===========================================================================
# Section 10: RegressionVerifier stateful class
# ===========================================================================

class TestRegressionVerifier:
    def test_first_version(self):
        rv = RegressionVerifier()
        r = rv.verify(PROG_V1)
        assert r.is_safe
        assert rv.version_count == 1

    def test_second_version_same(self):
        rv = RegressionVerifier()
        rv.verify(PROG_V1)
        r = rv.verify(PROG_V1_SAME)
        assert r.is_safe

    def test_version_sequence(self):
        rv = RegressionVerifier()
        results = rv.verify_sequence([PROG_V1, PROG_V2_ADDED_FN, PROG_V2_REFACTORED])
        assert len(results) == 3
        assert rv.version_count == 3


# ===========================================================================
# Section 11: Certificate reuse
# ===========================================================================

class TestCertificateReuse:
    def test_unchanged_reuses_cert(self):
        rv = RegressionVerifier()
        rv.verify(PROG_V1)
        r2 = rv.verify(PROG_V1_SAME)
        reused = [fv for fv in r2.function_verdicts if fv.cert_reused]
        assert len(reused) >= 1

    def test_modified_invalidates_cert(self):
        rv = RegressionVerifier()
        rv.verify(PROG_V1)
        r2 = rv.verify(PROG_V2_REGRESSION, symbolic_inputs={"x": "int"})
        # abs was modified -- cert should NOT be reused
        abs_fv = [fv for fv in r2.function_verdicts if fv.name == "abs"]
        if abs_fv:
            assert not abs_fv[0].cert_reused


# ===========================================================================
# Section 12: Regression report
# ===========================================================================

class TestRegressionReport:
    def test_report_format(self):
        report = regression_report(PROG_V1, PROG_V2_ADDED_FN)
        assert "verdict" in report.lower() or "safe" in report.lower()

    def test_report_regression(self):
        report = regression_report(PROG_V1, PROG_V2_REGRESSION,
                                   symbolic_inputs={"x": "int"})
        assert "regression" in report.lower()


# ===========================================================================
# Section 13: Comparison API
# ===========================================================================

class TestComparison:
    def test_compare_strategies(self):
        comp = compare_verification_strategies(PROG_V1, PROG_V2_ADDED_FN)
        assert "incremental_verdict" in comp
        assert "savings" in comp

    def test_compare_with_regression(self):
        comp = compare_verification_strategies(PROG_V1, PROG_V2_REGRESSION,
                                               symbolic_inputs={"x": "int"})
        assert comp["incremental_verdict"] == "regression"


# ===========================================================================
# Section 14: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_to_program(self):
        result = verify_regression("", PROG_V1)
        # All functions are new
        added = [fv for fv in result.function_verdicts
                 if fv.change_kind == ChangeKind.ADDED]
        assert len(added) >= 1

    def test_program_to_empty(self):
        result = verify_regression(PROG_V1, "")
        removed = [fv for fv in result.function_verdicts
                   if fv.change_kind == ChangeKind.REMOVED]
        assert len(removed) >= 1

    def test_single_fn_unchanged(self):
        prog = "fn f(x) { return x + 1; }"
        result = verify_regression(prog, prog)
        assert result.is_safe


# ===========================================================================
# Section 15: Internal fuzz helper
# ===========================================================================

class TestInternalFuzz:
    def test_fuzz_finds_divergence(self):
        old = "let x = 0; let __result = x + 1;"
        new = "let x = 0; let __result = x + 2;"
        result = _fuzz_for_regression(old, new, ["x"], max_inputs=50)
        divergences = [f for f in result.findings if f.kind == "divergence"]
        assert len(divergences) > 0

    def test_fuzz_no_divergence(self):
        prog = "let x = 0; let __result = x + 1;"
        result = _fuzz_for_regression(prog, prog, ["x"], max_inputs=50)
        divergences = [f for f in result.findings if f.kind == "divergence"]
        assert len(divergences) == 0

    def test_summary_property(self):
        result = verify_regression(PROG_V1, PROG_V2_ADDED_FN)
        s = result.summary
        assert isinstance(s, str)
        assert len(s) > 0
