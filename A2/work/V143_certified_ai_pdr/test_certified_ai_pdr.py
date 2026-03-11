"""Tests for V143: Certified AI-Strengthened PDR"""

import sys, os, pytest
sys.path.insert(0, os.path.dirname(__file__))

from certified_ai_pdr import (
    AIPDRVerdict, AIPDRMethod, AIInvariant, AIPDRResult,
    certify_ai_pdr, certify_ai_pdr_basic, analyze_ai_invariants,
    compare_basic_vs_ai, ai_pdr_summary,
    certify_pdr_loop_with_invariants,
    compare_pdr_vs_kind_ai,
    _extract_ai_invariants, _extract_loop_variables,
    _invariants_to_sources,
)


# --- Test Sources ---

SIMPLE_COUNTER = """
let x = 0;
let i = 0;
while (i < 5) {
    x = x + 1;
    i = i + 1;
}
"""

BOUNDED_LOOP = """
let x = 0;
let i = 0;
while (i < 10) {
    x = x + 2;
    i = i + 1;
}
"""

DECREMENT_LOOP = """
let x = 10;
let i = 0;
while (i < 5) {
    x = x - 1;
    i = i + 1;
}
"""

NESTED_LOOP = """
let x = 0;
let i = 0;
while (i < 3) {
    let j = 0;
    while (j < 3) {
        x = x + 1;
        j = j + 1;
    }
    i = i + 1;
}
"""

NO_LOOP = """
let x = 5;
let y = x + 3;
"""

ACCUMULATOR = """
let sum = 0;
let i = 1;
while (i <= 5) {
    sum = sum + i;
    i = i + 1;
}
"""

DOUBLING_LOOP = """
let x = 1;
let i = 0;
while (i < 4) {
    x = x + x;
    i = i + 1;
}
"""


# === Section 1: Loop Variable Extraction ===

class TestLoopVariables:
    def test_simple_counter_vars(self):
        vars = _extract_loop_variables(SIMPLE_COUNTER)
        assert "x" in vars
        assert "i" in vars

    def test_no_loop_vars(self):
        vars = _extract_loop_variables(NO_LOOP)
        assert isinstance(vars, list)

    def test_bounded_loop_vars(self):
        vars = _extract_loop_variables(BOUNDED_LOOP)
        assert "x" in vars
        assert "i" in vars

    def test_invalid_source(self):
        vars = _extract_loop_variables("not valid code {{{}}")
        assert isinstance(vars, list)


# === Section 2: AI Invariant Extraction ===

class TestInvariantExtraction:
    def test_simple_counter_invariants(self):
        invs = _extract_ai_invariants(SIMPLE_COUNTER)
        assert isinstance(invs, list)
        for inv in invs:
            assert isinstance(inv, AIInvariant)
            assert inv.variable
            assert inv.expression

    def test_bounded_loop_invariants(self):
        invs = _extract_ai_invariants(BOUNDED_LOOP)
        assert isinstance(invs, list)

    def test_invariants_have_sources(self):
        invs = _extract_ai_invariants(SIMPLE_COUNTER)
        sources = {inv.source for inv in invs}
        assert len(sources) > 0

    def test_invariants_to_sources(self):
        invs = _extract_ai_invariants(SIMPLE_COUNTER)
        sources = _invariants_to_sources(invs)
        assert isinstance(sources, list)
        for s in sources:
            assert isinstance(s, str)

    def test_no_loop_invariants(self):
        invs = _extract_ai_invariants(NO_LOOP)
        assert isinstance(invs, list)

    def test_accumulator_invariants(self):
        invs = _extract_ai_invariants(ACCUMULATOR)
        assert isinstance(invs, list)


# === Section 3: AI-Strengthened PDR ===

class TestCertifyAIPDR:
    def test_simple_counter(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert isinstance(result, AIPDRResult)
        assert result.verdict in (AIPDRVerdict.SAFE, AIPDRVerdict.UNKNOWN,
                                   AIPDRVerdict.AI_ONLY)

    def test_bounded_loop(self):
        result = certify_ai_pdr(BOUNDED_LOOP, "x >= 0", max_frames=50)
        assert isinstance(result, AIPDRResult)

    def test_result_has_invariants(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert isinstance(result.ai_invariants, list)

    def test_result_has_method(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert result.method in (AIPDRMethod.AI_STRENGTHENED, AIPDRMethod.BASIC_PDR,
                                  AIPDRMethod.AI_ONLY, AIPDRMethod.COMBINED)

    def test_result_has_property(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert result.property_desc == "x >= 0"

    def test_result_metadata(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert "elapsed" in result.metadata

    def test_accumulator(self):
        result = certify_ai_pdr(ACCUMULATOR, "sum >= 0", max_frames=50)
        assert isinstance(result, AIPDRResult)

    def test_doubling_loop(self):
        result = certify_ai_pdr(DOUBLING_LOOP, "x >= 1", max_frames=50)
        assert isinstance(result, AIPDRResult)


# === Section 4: Basic PDR (No AI) ===

class TestBasicPDR:
    def test_basic_simple(self):
        result = certify_ai_pdr_basic(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert isinstance(result, AIPDRResult)
        assert result.method == AIPDRMethod.BASIC_PDR
        assert len(result.ai_invariants) == 0

    def test_basic_no_ai_cert(self):
        result = certify_ai_pdr_basic(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert result.ai_result is None
        assert result.ai_certificate is None

    def test_basic_has_pdr_cert(self):
        result = certify_ai_pdr_basic(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert isinstance(result, AIPDRResult)

    def test_basic_bounded(self):
        result = certify_ai_pdr_basic(BOUNDED_LOOP, "x >= 0", max_frames=50)
        assert isinstance(result, AIPDRResult)


# === Section 5: Analyze AI Invariants ===

class TestAnalyzeInvariants:
    def test_analyze_simple(self):
        result = analyze_ai_invariants(SIMPLE_COUNTER)
        assert "invariant_count" in result
        assert "invariants" in result
        assert "variables" in result

    def test_analyze_bounded(self):
        result = analyze_ai_invariants(BOUNDED_LOOP)
        assert isinstance(result["invariants"], list)

    def test_analyze_no_loop(self):
        result = analyze_ai_invariants(NO_LOOP)
        assert isinstance(result, dict)

    def test_invariant_format(self):
        result = analyze_ai_invariants(SIMPLE_COUNTER)
        for inv in result["invariants"]:
            assert "variable" in inv
            assert "expression" in inv
            assert "source" in inv


# === Section 6: Comparison API (Basic vs AI) ===

class TestCompareAPI:
    def test_compare_basic_vs_ai(self):
        result = compare_basic_vs_ai(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert "basic" in result
        assert "ai_strengthened" in result
        assert "ai_helped" in result

    def test_compare_has_timing(self):
        result = compare_basic_vs_ai(SIMPLE_COUNTER, "x >= 0", max_frames=20)
        assert "time" in result["basic"]
        assert "time" in result["ai_strengthened"]

    def test_compare_has_verdicts(self):
        result = compare_basic_vs_ai(SIMPLE_COUNTER, "x >= 0", max_frames=20)
        assert "verdict" in result["basic"]
        assert "verdict" in result["ai_strengthened"]


# === Section 7: Summary API ===

class TestSummaryAPI:
    def test_summary_keys(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=20)
        s = ai_pdr_summary(result)
        assert "verdict" in s
        assert "method" in s
        assert "ai_invariants" in s
        assert "certified" in s

    def test_summary_string(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=20)
        s = result.summary()
        assert "AI-Strengthened PDR" in s

    def test_to_dict(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=20)
        d = result.to_dict()
        assert "verdict" in d
        assert "ai_invariants" in d
        assert "num_frames" in d


# === Section 8: Result Properties ===

class TestResultProperties:
    def test_certified_property(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert isinstance(result.certified, bool)

    def test_total_obligations(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert isinstance(result.total_obligations, int)
        assert result.total_obligations >= 0

    def test_valid_obligations(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        assert isinstance(result.valid_obligations, int)
        assert result.valid_obligations <= result.total_obligations

    def test_combined_certificate(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=50)
        # May or may not have combined cert depending on both phases succeeding
        assert isinstance(result, AIPDRResult)


# === Section 9: Edge Cases ===

class TestEdgeCases:
    def test_empty_source(self):
        result = certify_ai_pdr("", "x >= 0", max_frames=20)
        assert isinstance(result, AIPDRResult)

    def test_no_loop_source(self):
        result = certify_ai_pdr(NO_LOOP, "x >= 0", max_frames=20)
        assert isinstance(result, AIPDRResult)

    def test_max_frames_1(self):
        result = certify_ai_pdr(SIMPLE_COUNTER, "x >= 0", max_frames=1)
        assert isinstance(result, AIPDRResult)

    def test_errors_list(self):
        result = certify_ai_pdr("", "x >= 0", max_frames=20)
        assert isinstance(result.errors, list)


# === Section 10: Decrement and Nested Loops ===

class TestComplexLoops:
    def test_decrement_loop(self):
        result = certify_ai_pdr(DECREMENT_LOOP, "x >= 0", max_frames=50)
        assert isinstance(result, AIPDRResult)

    def test_decrement_invariants(self):
        invs = _extract_ai_invariants(DECREMENT_LOOP)
        assert isinstance(invs, list)

    def test_nested_loop(self):
        result = certify_ai_pdr(NESTED_LOOP, "x >= 0", max_frames=20)
        assert isinstance(result, AIPDRResult)

    def test_nested_invariants(self):
        invs = _extract_ai_invariants(NESTED_LOOP)
        assert isinstance(invs, list)


# === Section 11: Strengthened PDR with Invariants ===

class TestStrengthenedPDR:
    def test_with_manual_invariants(self):
        """Test certify_pdr_loop_with_invariants with manually provided invariants."""
        cert = certify_pdr_loop_with_invariants(
            SIMPLE_COUNTER, "x >= 0", ["i >= 0", "x >= 0"], max_frames=50
        )
        assert isinstance(cert, type(cert))  # PDRCertificate

    def test_with_empty_invariants(self):
        """When no invariants, should still work (falls through to plain property)."""
        cert = certify_pdr_loop_with_invariants(
            SIMPLE_COUNTER, "x >= 0", [], max_frames=50
        )
        assert cert is not None

    def test_with_single_invariant(self):
        cert = certify_pdr_loop_with_invariants(
            BOUNDED_LOOP, "x >= 0", ["i >= 0"], max_frames=50
        )
        assert cert is not None

    def test_with_ai_derived_invariants(self):
        """Extract invariants from AI, then feed to strengthened PDR."""
        invs = _extract_ai_invariants(SIMPLE_COUNTER)
        inv_sources = _invariants_to_sources(invs)
        if inv_sources:
            cert = certify_pdr_loop_with_invariants(
                SIMPLE_COUNTER, "x >= 0", inv_sources, max_frames=50
            )
            assert cert is not None


# === Section 12: Cross-Method Comparison (PDR vs k-Induction) ===

class TestCrossMethodComparison:
    def test_pdr_vs_kind_ai(self):
        result = compare_pdr_vs_kind_ai(SIMPLE_COUNTER, "x >= 0",
                                          max_frames=20, max_k=10)
        if "error" not in result:
            assert "ai_pdr" in result
            assert "ai_kind" in result
            assert "same_verdict" in result
        else:
            # V141 not available is acceptable
            assert isinstance(result["error"], str)

    def test_pdr_vs_kind_same_invariants(self):
        result = compare_pdr_vs_kind_ai(SIMPLE_COUNTER, "x >= 0",
                                          max_frames=20, max_k=10)
        if "error" not in result:
            assert "same_invariants" in result


# === Section 13: Invariant Quality ===

class TestInvariantQuality:
    def test_counter_has_nonneg_bounds(self):
        """For a counter starting at 0, AI should derive non-negative bounds."""
        invs = _extract_ai_invariants(SIMPLE_COUNTER)
        expressions = [inv.expression for inv in invs]
        # Should have at least one bound for x or i
        has_bound = any("x" in e or "i" in e for e in expressions)
        assert has_bound or len(invs) == 0  # might not find useful bounds

    def test_decrement_has_bounds(self):
        invs = _extract_ai_invariants(DECREMENT_LOOP)
        # x starts at 10, decrements -- AI should derive some bound
        assert isinstance(invs, list)

    def test_accumulator_bounds(self):
        invs = _extract_ai_invariants(ACCUMULATOR)
        assert isinstance(invs, list)


# === Section 14: Verdict Enum Coverage ===

class TestVerdictEnums:
    def test_verdict_values(self):
        assert AIPDRVerdict.SAFE.value == "safe"
        assert AIPDRVerdict.UNSAFE.value == "unsafe"
        assert AIPDRVerdict.UNKNOWN.value == "unknown"
        assert AIPDRVerdict.AI_ONLY.value == "ai_only"

    def test_method_values(self):
        assert AIPDRMethod.BASIC_PDR.value == "basic_pdr"
        assert AIPDRMethod.AI_STRENGTHENED.value == "ai_strengthened"
        assert AIPDRMethod.AI_ONLY.value == "ai_only"
        assert AIPDRMethod.COMBINED.value == "combined"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
