"""
Tests for V053: Cross-Analysis Correlation
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from cross_analysis import (
    correlate_analyses, smart_verify, detect_features, recommend_analyses,
    recommendation_to_config,
    AnalysisPair, AnalysisProfile, ProgramFeatures, AnalysisRecommendation,
    CorrelationReport, _is_pass, _is_fail,
)
from holistic_verification import (
    PipelineConfig, VerificationReport, AnalysisStatus,
)


# --- Test Programs ---

PURE_FN = """
fn add(x, y) {
    return x + y;
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

WITH_IO = """
fn greet(x) {
    print(x);
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

MULTI_FN = """
fn double(x) {
    return x + x;
}
fn quadruple(x) {
    let d = double(x);
    return double(d);
}
"""

TRIVIAL = """
let x = 42;
"""


# === Section 1: AnalysisPair ===

class TestAnalysisPair(unittest.TestCase):
    def test_empty(self):
        p = AnalysisPair("a", "b")
        assert p.agreement_rate == 0.0
        assert p.conflict_rate == 0.0
        assert p.is_redundant is False
        assert p.is_complementary is False

    def test_perfect_agreement(self):
        p = AnalysisPair("a", "b", both_pass=5, total=5)
        assert p.agreement_rate == 1.0
        assert p.is_redundant is True

    def test_total_disagreement(self):
        p = AnalysisPair("a", "b", a_pass_b_fail=5, total=5)
        assert p.conflict_rate == 1.0
        assert p.is_redundant is False

    def test_complementary(self):
        p = AnalysisPair("a", "b", a_pass_b_fail=2, a_fail_b_pass=3, total=5)
        assert p.is_complementary is True

    def test_partial_agreement(self):
        p = AnalysisPair("a", "b", both_pass=3, both_fail=1,
                         a_pass_b_fail=1, total=5)
        assert p.agreement_rate == 0.8


# === Section 2: AnalysisProfile ===

class TestAnalysisProfile(unittest.TestCase):
    def test_empty(self):
        p = AnalysisProfile(name="test")
        assert p.pass_rate == 0.0
        assert p.effectiveness == 0.0

    def test_all_pass(self):
        p = AnalysisProfile(name="test", pass_count=10, total=10)
        assert p.pass_rate == 1.0
        assert p.effectiveness == 1.0

    def test_mixed(self):
        p = AnalysisProfile(name="test", pass_count=5, fail_count=3,
                            error_count=2, total=10)
        assert p.pass_rate == 0.5
        assert p.fail_rate == 0.3
        assert p.error_rate == 0.2
        assert p.effectiveness == 0.8


# === Section 3: Feature Detection ===

class TestFeatureDetection(unittest.TestCase):
    def test_pure_function(self):
        f = detect_features(PURE_FN)
        assert f.has_functions is True
        assert f.has_loops is False
        assert f.has_specs is False

    def test_with_loop(self):
        f = detect_features(WITH_LOOP)
        assert f.has_loops is True
        assert f.has_functions is True

    def test_with_conditional(self):
        f = detect_features(WITH_CONDITIONAL)
        assert f.has_conditionals is True

    def test_with_io(self):
        f = detect_features(WITH_IO)
        assert f.has_io is True

    def test_multi_function_recursion(self):
        f = detect_features(MULTI_FN)
        assert f.has_functions is True
        assert f.function_count == 2
        assert f.has_recursion is True  # quadruple calls double

    def test_trivial(self):
        f = detect_features(TRIVIAL)
        assert f.has_functions is False
        assert f.has_loops is False

    def test_invalid_source(self):
        f = detect_features("not valid!!!")
        assert f.has_functions is False

    def test_empty_source(self):
        f = detect_features("")
        assert f.has_functions is False


# === Section 4: Recommendation Engine ===

class TestRecommendations(unittest.TestCase):
    def test_pure_function(self):
        f = detect_features(PURE_FN)
        rec = recommend_analyses(f)
        assert "certified_ai" in rec.recommended
        assert "effects" in rec.recommended
        assert "termination" in rec.skip  # no loops

    def test_with_loop(self):
        f = detect_features(WITH_LOOP)
        rec = recommend_analyses(f)
        assert "termination" in rec.recommended

    def test_with_specs(self):
        f = detect_features(WITH_SPEC)
        rec = recommend_analyses(f)
        assert "vcgen" in rec.recommended
        assert "modular_verification" in rec.recommended

    def test_no_specs(self):
        f = detect_features(PURE_FN)
        rec = recommend_analyses(f)
        assert "vcgen" in rec.skip

    def test_trivial(self):
        f = detect_features(TRIVIAL)
        rec = recommend_analyses(f)
        assert "termination" in rec.skip
        assert "refinement_types" in rec.skip

    def test_reasons_populated(self):
        f = detect_features(WITH_LOOP)
        rec = recommend_analyses(f)
        assert "termination" in rec.reason
        assert len(rec.reason["termination"]) > 0


class TestRecommendationToConfig(unittest.TestCase):
    def test_basic(self):
        rec = AnalysisRecommendation(
            recommended=["certified_ai", "effects"],
            optional=["guided_symex"],
            skip=["vcgen", "termination"],
        )
        config = recommendation_to_config(rec)
        assert config.certified_ai is True
        assert config.effects is True
        assert config.guided_symex is True
        assert config.vcgen is False
        assert config.termination is False

    def test_empty(self):
        rec = AnalysisRecommendation()
        config = recommendation_to_config(rec)
        assert config.certified_ai is False


# === Section 5: Status Helpers ===

class TestStatusHelpers(unittest.TestCase):
    def test_pass(self):
        assert _is_pass(AnalysisStatus.PASSED) is True
        assert _is_pass(AnalysisStatus.SKIPPED) is True
        assert _is_pass(AnalysisStatus.FAILED) is False

    def test_fail(self):
        assert _is_fail(AnalysisStatus.FAILED) is True
        assert _is_fail(AnalysisStatus.PASSED) is False
        assert _is_fail(AnalysisStatus.ERROR) is False


# === Section 6: Correlation Analysis ===

class TestCorrelateAnalyses(unittest.TestCase):
    def test_single_program(self):
        r = correlate_analyses([PURE_FN], PipelineConfig.fast())
        assert r.programs_analyzed == 1
        assert len(r.profiles) > 0

    def test_multiple_programs(self):
        r = correlate_analyses([PURE_FN, WITH_LOOP], PipelineConfig.fast())
        assert r.programs_analyzed == 2

    def test_profiles_populated(self):
        r = correlate_analyses([PURE_FN, WITH_LOOP, WITH_CONDITIONAL],
                               PipelineConfig.fast())
        for name, profile in r.profiles.items():
            assert profile.total > 0

    def test_pairs_exist(self):
        r = correlate_analyses([PURE_FN], PipelineConfig.fast())
        assert len(r.pairs) > 0

    def test_empty_input(self):
        r = correlate_analyses([], PipelineConfig.fast())
        assert r.programs_analyzed == 0
        assert len(r.profiles) == 0

    def test_summary(self):
        r = correlate_analyses([PURE_FN, WITH_LOOP], PipelineConfig.fast())
        s = r.summary()
        assert "Cross-Analysis Correlation Report" in s
        assert "Programs analyzed: 2" in s


# === Section 7: Smart Verify ===

class TestSmartVerify(unittest.TestCase):
    def test_pure_function(self):
        r = smart_verify(PURE_FN)
        assert isinstance(r, VerificationReport)

    def test_with_loop(self):
        r = smart_verify(WITH_LOOP)
        assert isinstance(r, VerificationReport)

    def test_with_specs(self):
        r = smart_verify(WITH_SPEC)
        assert isinstance(r, VerificationReport)
        # Should have VCGen enabled
        names = [a.name for a in r.analyses]
        assert "Verification Conditions" in names

    def test_trivial(self):
        r = smart_verify(TRIVIAL)
        assert isinstance(r, VerificationReport)
        # Should skip unnecessary analyses
        names = [a.name for a in r.analyses]
        assert "Termination Analysis" not in names  # no loops

    def test_conditional(self):
        r = smart_verify(WITH_CONDITIONAL)
        assert isinstance(r, VerificationReport)


# === Section 8: CorrelationReport Properties ===

class TestCorrelationReportProperties(unittest.TestCase):
    def test_redundant_pairs(self):
        r = CorrelationReport()
        r.pairs[("a", "b")] = AnalysisPair("a", "b", both_pass=5, total=5)
        assert len(r.redundant_pairs) == 1

    def test_complementary_pairs(self):
        r = CorrelationReport()
        r.pairs[("a", "b")] = AnalysisPair("a", "b",
                                              a_pass_b_fail=2,
                                              a_fail_b_pass=3, total=5)
        assert len(r.complementary_pairs) == 1

    def test_no_redundant(self):
        r = CorrelationReport()
        r.pairs[("a", "b")] = AnalysisPair("a", "b",
                                              both_pass=3, a_pass_b_fail=2, total=5)
        assert len(r.redundant_pairs) == 0


# === Section 9: Robustness ===

class TestRobustness(unittest.TestCase):
    def test_invalid_programs(self):
        r = correlate_analyses(["not valid!!!"], PipelineConfig.fast())
        assert r.programs_analyzed == 1

    def test_mixed_valid_invalid(self):
        r = correlate_analyses([PURE_FN, "invalid!", WITH_LOOP],
                               PipelineConfig.fast())
        assert r.programs_analyzed == 3

    def test_smart_verify_invalid(self):
        r = smart_verify("not valid!!!")
        assert isinstance(r, VerificationReport)

    def test_empty_smart_verify(self):
        r = smart_verify("")
        assert isinstance(r, VerificationReport)


if __name__ == '__main__':
    unittest.main()
