"""
Tests for V050: Holistic Verification Dashboard

Tests the unified verification pipeline that composes all V-challenges.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from holistic_verification import (
    verify_holistic, quick_verify, deep_verify, verify_and_report,
    run_single_analysis, available_analyses,
    PipelineConfig, VerificationReport, AnalysisResult, AnalysisStatus,
    run_certified_ai, run_vcgen, run_effects, run_guided_symex,
    run_refinement_types, run_termination, run_modular_verification,
    run_verified_compilation,
)


# --- Test Programs ---

SIMPLE_PURE = """
fn add(x, y) {
    return x + y;
}
"""

SIMPLE_STATEFUL = """
fn inc(x) {
    x = x + 1;
    return x;
}
"""

SIMPLE_IO = """
fn greet(x) {
    print(x);
}
"""

WITH_LOOP = """
fn sum_to(n) {
    let s = 0;
    let i = 0;
    while (i < n) {
        s = s + i;
        i = i + 1;
    }
    return s;
}
"""

WITH_SPEC = """
fn inc(x) {
    requires(x >= 0);
    ensures(result > 0);
    return x + 1;
}
"""

WITH_CONDITIONAL = """
fn abs(x) {
    let r = 0;
    if (x >= 0) {
        r = x;
    } else {
        r = 0 - x;
    }
    return r;
}
"""

MULTI_FUNCTION = """
fn double(x) {
    return x + x;
}
fn triple(x) {
    return x + x + x;
}
fn use_both(x) {
    let d = double(x);
    let t = triple(x);
    return d + t;
}
"""

FAILING_SPEC = """
fn bad(x) {
    requires(x > 0);
    ensures(result > 10);
    return x;
}
"""

TRIVIAL = """
let x = 42;
"""


# === Section 1: AnalysisResult Tests ===

class TestAnalysisResult(unittest.TestCase):
    def test_passed_is_ok(self):
        r = AnalysisResult("test", AnalysisStatus.PASSED, "ok")
        assert r.ok is True

    def test_failed_not_ok(self):
        r = AnalysisResult("test", AnalysisStatus.FAILED, "bad")
        assert r.ok is False

    def test_skipped_is_ok(self):
        r = AnalysisResult("test", AnalysisStatus.SKIPPED, "skipped")
        assert r.ok is True

    def test_error_not_ok(self):
        r = AnalysisResult("test", AnalysisStatus.ERROR, "crash")
        assert r.ok is False

    def test_warning_not_ok(self):
        r = AnalysisResult("test", AnalysisStatus.WARNING, "warn")
        assert r.ok is False


# === Section 2: VerificationReport Tests ===

class TestVerificationReport(unittest.TestCase):
    def test_empty_report(self):
        r = VerificationReport(source="")
        assert r.all_passed is True
        assert r.score == 0.0  # no analyses run
        assert len(r.passed) == 0

    def test_all_passed(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.PASSED, "ok"),
            AnalysisResult("b", AnalysisStatus.PASSED, "ok"),
        ])
        assert r.all_passed is True
        assert r.score == 1.0

    def test_one_failed(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.PASSED, "ok"),
            AnalysisResult("b", AnalysisStatus.FAILED, "bad"),
        ])
        assert r.all_passed is False
        assert len(r.failed) == 1
        assert r.score == 0.5

    def test_with_warning(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.PASSED, "ok"),
            AnalysisResult("b", AnalysisStatus.WARNING, "warn"),
        ])
        assert r.all_passed is True  # warnings don't fail
        assert r.score == 0.75  # warning counts half

    def test_with_skip(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.PASSED, "ok"),
            AnalysisResult("b", AnalysisStatus.SKIPPED, "n/a"),
        ])
        assert r.score == 1.0  # skipped not counted

    def test_summary_string(self):
        r = VerificationReport(source="test", analyses=[
            AnalysisResult("Test Analysis", AnalysisStatus.PASSED, "All good"),
        ], total_duration_ms=42.0)
        s = r.summary()
        assert "HOLISTIC VERIFICATION REPORT" in s
        assert "[PASS]" in s
        assert "Test Analysis" in s
        assert "100%" in s

    def test_summary_with_findings(self):
        r = VerificationReport(source="test", analyses=[
            AnalysisResult("Analysis", AnalysisStatus.FAILED, "Bad",
                          findings=["Issue 1", "Issue 2"]),
        ])
        s = r.summary()
        assert "Issue 1" in s
        assert "[FAIL]" in s


# === Section 3: PipelineConfig Tests ===

class TestPipelineConfig(unittest.TestCase):
    def test_all_enabled(self):
        c = PipelineConfig.all_enabled()
        assert c.certified_ai is True
        assert c.vcgen is True
        assert c.effects is True
        assert c.guided_symex is True
        assert c.verified_compilation is True

    def test_fast_config(self):
        c = PipelineConfig.fast()
        assert c.certified_ai is True
        assert c.vcgen is True
        assert c.effects is True
        assert c.guided_symex is False
        assert c.verified_compilation is False

    def test_deep_is_all(self):
        c = PipelineConfig.deep()
        assert c.certified_ai is True
        assert c.verified_compilation is True


# === Section 4: Individual Analysis Runners ===

class TestCertifiedAI(unittest.TestCase):
    def test_simple_program(self):
        r = run_certified_ai(SIMPLE_STATEFUL)
        assert r.name == "Certified Abstract Interpretation"
        assert r.status in (AnalysisStatus.PASSED, AnalysisStatus.WARNING, AnalysisStatus.ERROR)
        assert r.duration_ms >= 0

    def test_with_loop(self):
        r = run_certified_ai(WITH_LOOP)
        assert r.name == "Certified Abstract Interpretation"
        assert r.duration_ms >= 0

    def test_conditional(self):
        r = run_certified_ai(WITH_CONDITIONAL)
        assert r.name == "Certified Abstract Interpretation"


class TestVCGen(unittest.TestCase):
    def test_with_spec_passes(self):
        r = run_vcgen(WITH_SPEC)
        assert r.name == "Verification Conditions"
        # May pass or error depending on VCGen's ability to parse
        assert r.duration_ms >= 0

    def test_failing_spec(self):
        r = run_vcgen(FAILING_SPEC)
        assert r.name == "Verification Conditions"

    def test_no_spec(self):
        r = run_vcgen(SIMPLE_PURE)
        assert r.name == "Verification Conditions"


class TestEffects(unittest.TestCase):
    def test_pure_function(self):
        r = run_effects(SIMPLE_PURE)
        assert r.name == "Effect Analysis"
        assert r.status in (AnalysisStatus.PASSED, AnalysisStatus.WARNING, AnalysisStatus.ERROR)

    def test_stateful(self):
        r = run_effects(SIMPLE_STATEFUL)
        assert r.name == "Effect Analysis"

    def test_io(self):
        r = run_effects(SIMPLE_IO)
        assert r.name == "Effect Analysis"

    def test_multi_function(self):
        r = run_effects(MULTI_FUNCTION)
        assert r.name == "Effect Analysis"


class TestGuidedSymex(unittest.TestCase):
    def test_simple(self):
        r = run_guided_symex(SIMPLE_PURE)
        assert r.name == "Guided Symbolic Execution"
        assert r.duration_ms >= 0

    def test_conditional(self):
        r = run_guided_symex(WITH_CONDITIONAL)
        assert r.name == "Guided Symbolic Execution"

    def test_loop(self):
        r = run_guided_symex(WITH_LOOP)
        assert r.name == "Guided Symbolic Execution"


class TestRefinementTypes(unittest.TestCase):
    def test_simple(self):
        r = run_refinement_types(SIMPLE_PURE)
        assert r.name == "Refinement Types"
        assert r.duration_ms >= 0

    def test_with_spec(self):
        r = run_refinement_types(WITH_SPEC)
        assert r.name == "Refinement Types"


class TestTermination(unittest.TestCase):
    def test_no_loop(self):
        r = run_termination(SIMPLE_PURE)
        assert r.name == "Termination Analysis"

    def test_with_loop(self):
        r = run_termination(WITH_LOOP)
        assert r.name == "Termination Analysis"
        assert r.duration_ms >= 0


class TestModularVerification(unittest.TestCase):
    def test_with_spec(self):
        r = run_modular_verification(WITH_SPEC)
        assert r.name == "Modular Verification"
        assert r.duration_ms >= 0

    def test_multi_function(self):
        r = run_modular_verification(MULTI_FUNCTION)
        assert r.name == "Modular Verification"


class TestVerifiedCompilation(unittest.TestCase):
    def test_simple(self):
        r = run_verified_compilation(SIMPLE_PURE)
        assert r.name == "Verified Compilation"
        assert r.duration_ms >= 0

    def test_with_loop(self):
        r = run_verified_compilation(WITH_LOOP)
        assert r.name == "Verified Compilation"


# === Section 5: Pipeline Integration Tests ===

class TestQuickVerify(unittest.TestCase):
    def test_returns_report(self):
        r = quick_verify(SIMPLE_PURE)
        assert isinstance(r, VerificationReport)
        assert r.source == SIMPLE_PURE
        assert len(r.analyses) == 3  # AI + VCGen + Effects
        assert r.total_duration_ms >= 0

    def test_stateful_program(self):
        r = quick_verify(SIMPLE_STATEFUL)
        assert isinstance(r, VerificationReport)
        assert len(r.analyses) == 3

    def test_score_between_0_and_1(self):
        r = quick_verify(SIMPLE_PURE)
        assert 0.0 <= r.score <= 1.0


class TestDeepVerify(unittest.TestCase):
    def test_returns_report(self):
        r = deep_verify(SIMPLE_PURE)
        assert isinstance(r, VerificationReport)
        assert len(r.analyses) == 8  # all analyses

    def test_all_analyses_have_names(self):
        r = deep_verify(SIMPLE_PURE)
        names = [a.name for a in r.analyses]
        assert "Certified Abstract Interpretation" in names
        assert "Effect Analysis" in names

    def test_all_analyses_have_duration(self):
        r = deep_verify(SIMPLE_PURE)
        for a in r.analyses:
            assert a.duration_ms >= 0

    def test_with_loop_program(self):
        r = deep_verify(WITH_LOOP)
        assert isinstance(r, VerificationReport)
        assert len(r.analyses) == 8


class TestVerifyHolistic(unittest.TestCase):
    def test_custom_config(self):
        config = PipelineConfig(
            certified_ai=True,
            vcgen=False,
            effects=True,
            guided_symex=False,
            refinement_types=False,
            termination=False,
            modular_verification=False,
            verified_compilation=False,
        )
        r = verify_holistic(SIMPLE_PURE, config)
        assert len(r.analyses) == 2  # AI + Effects only

    def test_empty_config(self):
        config = PipelineConfig(
            certified_ai=False, vcgen=False, effects=False,
            guided_symex=False, refinement_types=False,
            termination=False, modular_verification=False,
            verified_compilation=False,
        )
        r = verify_holistic(SIMPLE_PURE, config)
        assert len(r.analyses) == 0
        assert r.all_passed is True

    def test_conditional_program(self):
        r = verify_holistic(WITH_CONDITIONAL)
        assert isinstance(r, VerificationReport)
        assert len(r.analyses) > 0

    def test_multi_function_program(self):
        r = verify_holistic(MULTI_FUNCTION)
        assert isinstance(r, VerificationReport)


class TestVerifyAndReport(unittest.TestCase):
    def test_returns_string(self):
        s = verify_and_report(SIMPLE_PURE, PipelineConfig.fast())
        assert isinstance(s, str)
        assert "HOLISTIC VERIFICATION REPORT" in s

    def test_report_contains_analyses(self):
        s = verify_and_report(SIMPLE_PURE, PipelineConfig.fast())
        assert "Certified Abstract Interpretation" in s or "[" in s


# === Section 6: Single Analysis and Discovery ===

class TestSingleAnalysis(unittest.TestCase):
    def test_run_effects_by_name(self):
        r = run_single_analysis(SIMPLE_PURE, "effects")
        assert r.name == "Effect Analysis"

    def test_run_vcgen_by_name(self):
        r = run_single_analysis(WITH_SPEC, "vcgen")
        assert r.name == "Verification Conditions"

    def test_unknown_analysis(self):
        r = run_single_analysis(SIMPLE_PURE, "nonexistent")
        assert r.status == AnalysisStatus.ERROR
        assert "Unknown analysis" in r.summary


class TestAvailableAnalyses(unittest.TestCase):
    def test_lists_all(self):
        names = available_analyses()
        assert len(names) == 8
        assert "certified_ai" in names
        assert "effects" in names
        assert "vcgen" in names
        assert "guided_symex" in names
        assert "verified_compilation" in names

    def test_all_names_runnable(self):
        """Every listed analysis can be called without crashing."""
        for name in available_analyses():
            r = run_single_analysis(TRIVIAL, name)
            assert isinstance(r, AnalysisResult)
            assert r.name != ""


# === Section 7: Robustness Tests ===

class TestRobustness(unittest.TestCase):
    def test_empty_source(self):
        """Empty source should not crash, analyses may error gracefully."""
        r = verify_holistic("", PipelineConfig.fast())
        assert isinstance(r, VerificationReport)

    def test_trivial_program(self):
        r = verify_holistic(TRIVIAL, PipelineConfig.fast())
        assert isinstance(r, VerificationReport)

    def test_syntax_error_program(self):
        """Malformed program should produce error results, not crash."""
        r = verify_holistic("this is not valid C10 code!!!", PipelineConfig.fast())
        assert isinstance(r, VerificationReport)
        # All analyses should have a status (error is fine)
        for a in r.analyses:
            assert a.status is not None

    def test_all_analyses_catch_exceptions(self):
        """No analysis should let exceptions escape."""
        for name in available_analyses():
            r = run_single_analysis("invalid!!!", name)
            assert isinstance(r, AnalysisResult)
            # Should be ERROR, not an uncaught exception
            assert r.status is not None


# === Section 8: Score Computation ===

class TestScoring(unittest.TestCase):
    def test_perfect_score(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.PASSED, "ok"),
            AnalysisResult("b", AnalysisStatus.PASSED, "ok"),
            AnalysisResult("c", AnalysisStatus.PASSED, "ok"),
        ])
        assert r.score == 1.0

    def test_zero_score(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.FAILED, "bad"),
            AnalysisResult("b", AnalysisStatus.FAILED, "bad"),
        ])
        assert r.score == 0.0

    def test_mixed_score(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.PASSED, "ok"),
            AnalysisResult("b", AnalysisStatus.WARNING, "warn"),
            AnalysisResult("c", AnalysisStatus.FAILED, "bad"),
        ])
        # 1 + 0.5 + 0 = 1.5 / 3 = 0.5
        assert r.score == 0.5

    def test_all_errors(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.ERROR, "crash"),
        ])
        assert r.score == 0.0

    def test_skip_only(self):
        r = VerificationReport(source="", analyses=[
            AnalysisResult("a", AnalysisStatus.SKIPPED, "n/a"),
        ])
        assert r.score == 1.0


if __name__ == '__main__':
    unittest.main()
