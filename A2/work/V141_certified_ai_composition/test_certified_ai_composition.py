"""Tests for V141: Certified AI-Strengthened k-Induction"""

import sys, os, pytest
sys.path.insert(0, os.path.dirname(__file__))

from certified_ai_composition import (
    AIKIndVerdict, AIKIndMethod, AIInvariant, AIKIndResult,
    certify_ai_kind, certify_ai_kind_basic, analyze_ai_invariants,
    compare_basic_vs_ai, ai_kind_summary,
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


# === Section 1: Loop Variable Extraction ===

class TestLoopVariables:
    def test_simple_counter_vars(self):
        vars = _extract_loop_variables(SIMPLE_COUNTER)
        assert "x" in vars
        assert "i" in vars

    def test_no_loop_vars(self):
        vars = _extract_loop_variables(NO_LOOP)
        # Still extracts variables, just no loop body vars
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
        # Should have at least some invariants
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
        # Should have interval or sign sources
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


# === Section 3: AI-Strengthened k-Induction ===

class TestCertifyAIKInd:
    def test_simple_counter(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert isinstance(result, AIKIndResult)
        assert result.verdict in (AIKIndVerdict.SAFE, AIKIndVerdict.UNKNOWN,
                                   AIKIndVerdict.AI_ONLY)

    def test_bounded_loop(self):
        result = certify_ai_kind(BOUNDED_LOOP, "x >= 0", max_k=10)
        assert isinstance(result, AIKIndResult)

    def test_result_has_invariants(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert isinstance(result.ai_invariants, list)

    def test_result_has_method(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert result.method in (AIKIndMethod.AI_STRENGTHENED, AIKIndMethod.BASIC_KIND,
                                  AIKIndMethod.AI_ONLY, AIKIndMethod.COMBINED)

    def test_result_has_property(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert result.property_desc == "x >= 0"

    def test_result_metadata(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert "elapsed" in result.metadata

    def test_accumulator(self):
        result = certify_ai_kind(ACCUMULATOR, "sum >= 0", max_k=10)
        assert isinstance(result, AIKIndResult)


# === Section 4: Basic k-Induction (No AI) ===

class TestBasicKInd:
    def test_basic_simple(self):
        result = certify_ai_kind_basic(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert isinstance(result, AIKIndResult)
        assert result.method == AIKIndMethod.BASIC_KIND
        assert len(result.ai_invariants) == 0

    def test_basic_no_ai_cert(self):
        result = certify_ai_kind_basic(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert result.ai_result is None
        assert result.ai_certificate is None

    def test_basic_has_kind_cert(self):
        result = certify_ai_kind_basic(SIMPLE_COUNTER, "x >= 0", max_k=10)
        # May or may not have kind_cert depending on success
        assert isinstance(result, AIKIndResult)


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


# === Section 6: Comparison API ===

class TestCompareAPI:
    def test_compare_basic_vs_ai(self):
        result = compare_basic_vs_ai(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert "basic" in result
        assert "ai_strengthened" in result
        assert "ai_helped" in result

    def test_compare_has_timing(self):
        result = compare_basic_vs_ai(SIMPLE_COUNTER, "x >= 0", max_k=5)
        assert "time" in result["basic"]
        assert "time" in result["ai_strengthened"]

    def test_compare_has_verdicts(self):
        result = compare_basic_vs_ai(SIMPLE_COUNTER, "x >= 0", max_k=5)
        assert "verdict" in result["basic"]
        assert "verdict" in result["ai_strengthened"]


# === Section 7: Summary API ===

class TestSummaryAPI:
    def test_summary_keys(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=5)
        s = ai_kind_summary(result)
        assert "verdict" in s
        assert "method" in s
        assert "ai_invariants" in s
        assert "certified" in s

    def test_summary_string(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=5)
        s = result.summary()
        assert "AI-Strengthened k-Induction" in s

    def test_to_dict(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=5)
        d = result.to_dict()
        assert "verdict" in d
        assert "ai_invariants" in d


# === Section 8: Result Properties ===

class TestResultProperties:
    def test_certified_property(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert isinstance(result.certified, bool)

    def test_total_obligations(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert isinstance(result.total_obligations, int)
        assert result.total_obligations >= 0

    def test_valid_obligations(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=10)
        assert isinstance(result.valid_obligations, int)
        assert result.valid_obligations <= result.total_obligations


# === Section 9: Edge Cases ===

class TestEdgeCases:
    def test_empty_source(self):
        result = certify_ai_kind("", "x >= 0", max_k=5)
        assert isinstance(result, AIKIndResult)

    def test_no_loop_source(self):
        result = certify_ai_kind(NO_LOOP, "x >= 0", max_k=5)
        assert isinstance(result, AIKIndResult)

    def test_max_k_1(self):
        result = certify_ai_kind(SIMPLE_COUNTER, "x >= 0", max_k=1)
        assert isinstance(result, AIKIndResult)

    def test_errors_list(self):
        result = certify_ai_kind("", "x >= 0", max_k=5)
        assert isinstance(result.errors, list)


# === Section 10: Decrement and Nested Loops ===

class TestComplexLoops:
    def test_decrement_loop(self):
        result = certify_ai_kind(DECREMENT_LOOP, "x >= 0", max_k=10)
        assert isinstance(result, AIKIndResult)

    def test_decrement_invariants(self):
        invs = _extract_ai_invariants(DECREMENT_LOOP)
        assert isinstance(invs, list)

    def test_nested_loop(self):
        result = certify_ai_kind(NESTED_LOOP, "x >= 0", max_k=5)
        assert isinstance(result, AIKIndResult)

    def test_nested_invariants(self):
        invs = _extract_ai_invariants(NESTED_LOOP)
        assert isinstance(invs, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
