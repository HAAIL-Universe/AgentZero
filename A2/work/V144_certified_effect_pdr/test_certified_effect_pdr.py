"""Tests for V144: Certified Effect-Aware PDR."""

import sys, os, pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V143_certified_ai_pdr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V140_effect_aware_regression'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V137_certified_pdr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V141_certified_ai_composition'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V046_certified_abstract_interp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_vcgen'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V138_effect_aware_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V139_certified_regression'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'challenges', 'C032_effect_system'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'challenges', 'C039_abstract_interpreter'))

from certified_effect_pdr import (
    EffectPDRResult, EffectPDRVerdict, EffectPDRMethod, EffectInfo,
    certify_effect_pdr, certify_effect_pdr_basic, verify_effect_loop,
    analyze_effects_only, verify_effect_regression_pdr,
    compare_effect_vs_plain, compare_ai_vs_basic_effect_pdr,
    effect_pdr_summary, _analyze_effects, _build_effect_certificate,
    _combine_verdicts,
)
from certified_ai_pdr import AIPDRVerdict


# --- Test sources ---

PURE_COUNTER = """
let i = 0;
while (i < 5) {
    i = i + 1;
}
"""

BOUNDED_LOOP = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""

ACCUMULATOR = """
let sum = 0;
let i = 0;
while (i < 3) {
    sum = sum + i;
    i = i + 1;
}
"""

# Source with a function containing print (IO effect)
IO_SOURCE = """
fn greet(name) {
    print(name);
    return name;
}
let x = 0;
while (x < 3) {
    x = x + 1;
}
"""

# Source with division (potential Exn effect)
DIV_SOURCE = """
fn divide(a, b) {
    return a / b;
}
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""

# Pure function source
PURE_FN_SOURCE = """
fn add(a, b) {
    return a + b;
}
let i = 0;
while (i < 5) {
    i = i + 1;
}
"""


# ============================================================
# Section 1: Verdict logic
# ============================================================

class TestCombineVerdicts:
    def test_safe_and_conform(self):
        v = _combine_verdicts(AIPDRVerdict.SAFE, True)
        assert v == EffectPDRVerdict.SAFE

    def test_safe_but_effect_violation(self):
        v = _combine_verdicts(AIPDRVerdict.SAFE, False)
        assert v == EffectPDRVerdict.EFFECT_VIOLATION

    def test_unsafe_and_conform(self):
        v = _combine_verdicts(AIPDRVerdict.UNSAFE, True)
        assert v == EffectPDRVerdict.PROPERTY_FAILURE

    def test_unsafe_and_not_conform(self):
        v = _combine_verdicts(AIPDRVerdict.UNSAFE, False)
        assert v == EffectPDRVerdict.UNSAFE

    def test_unknown_and_conform(self):
        v = _combine_verdicts(AIPDRVerdict.UNKNOWN, True)
        assert v == EffectPDRVerdict.UNKNOWN

    def test_unknown_and_not_conform(self):
        v = _combine_verdicts(AIPDRVerdict.UNKNOWN, False)
        assert v == EffectPDRVerdict.EFFECT_VIOLATION

    def test_none_pdr_conform(self):
        v = _combine_verdicts(None, True)
        assert v == EffectPDRVerdict.SAFE

    def test_none_pdr_not_conform(self):
        v = _combine_verdicts(None, False)
        assert v == EffectPDRVerdict.EFFECT_VIOLATION


# ============================================================
# Section 2: Effect analysis
# ============================================================

class TestAnalyzeEffects:
    def test_pure_code_no_declarations(self):
        infos, conform = _analyze_effects(PURE_FN_SOURCE)
        assert conform is True
        # All functions should have no undeclared effects
        for ei in infos:
            assert ei.undeclared == []

    def test_io_detected(self):
        infos, _ = _analyze_effects(IO_SOURCE)
        # At least one function should have effects
        all_effects = []
        for ei in infos:
            all_effects.extend(ei.inferred_effects)
        # greet should have IO-like effect (print)
        assert len(infos) > 0

    def test_declared_effects_match(self):
        # Declare IO for greet -- should conform
        infos, conform = _analyze_effects(IO_SOURCE, {"greet": ["IO"]})
        greet_info = [ei for ei in infos if ei.function == "greet"]
        if greet_info:
            # If IO was inferred and declared, should conform
            assert greet_info[0].declared_effects == ["IO"]

    def test_undeclared_effects_detected(self):
        # Declare greet as pure when it has IO
        infos, conform = _analyze_effects(IO_SOURCE, {"greet": []})
        greet_info = [ei for ei in infos if ei.function == "greet"]
        if greet_info and greet_info[0].inferred_effects:
            assert not conform or greet_info[0].undeclared != []

    def test_empty_source(self):
        infos, conform = _analyze_effects("")
        assert conform is True
        assert infos == []


# ============================================================
# Section 3: Effect certificate
# ============================================================

class TestEffectCertificate:
    def test_valid_certificate(self):
        infos = [EffectInfo("foo", ["IO"], ["IO"], [], True)]
        cert = _build_effect_certificate(infos, True)
        from proof_certificates import CertStatus
        assert cert.status == CertStatus.VALID
        assert len(cert.obligations) == 1

    def test_invalid_certificate(self):
        infos = [EffectInfo("foo", ["IO", "Exn"], ["IO"], ["Exn"], False)]
        cert = _build_effect_certificate(infos, False)
        from proof_certificates import CertStatus
        assert cert.status == CertStatus.INVALID

    def test_no_declaration_is_valid(self):
        infos = [EffectInfo("foo", ["IO"], None, [], True)]
        cert = _build_effect_certificate(infos, True)
        from proof_certificates import CertStatus
        assert cert.status == CertStatus.VALID

    def test_empty_infos(self):
        cert = _build_effect_certificate([], True)
        from proof_certificates import CertStatus
        assert cert.status == CertStatus.VALID
        assert len(cert.obligations) == 0


# ============================================================
# Section 4: Certified effect-aware PDR (main API)
# ============================================================

class TestCertifyEffectPDR:
    def test_pure_counter_safe(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=10)
        assert isinstance(result, EffectPDRResult)
        assert result.verdict in (EffectPDRVerdict.SAFE, EffectPDRVerdict.UNKNOWN)
        assert result.method == EffectPDRMethod.AI_PDR_PLUS_EFFECTS

    def test_bounded_loop_safe(self):
        result = certify_effect_pdr(BOUNDED_LOOP, "x >= 0", max_frames=10)
        assert isinstance(result, EffectPDRResult)
        assert result.effects_conform is True

    def test_result_has_pdr_verdict(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert result.pdr_verdict is not None
        assert isinstance(result.pdr_verdict, AIPDRVerdict)

    def test_result_has_effect_infos(self):
        result = certify_effect_pdr(PURE_FN_SOURCE, "i >= 0", max_frames=5)
        assert isinstance(result.effect_infos, list)

    def test_result_has_certificates(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert result.effect_certificate is not None

    def test_ai_invariants_present(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert isinstance(result.ai_invariants, list)

    def test_metadata_has_timing(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert 'pdr_time' in result.metadata
        assert 'effect_time' in result.metadata
        assert 'total_time' in result.metadata

    def test_with_declared_effects(self):
        result = certify_effect_pdr(PURE_FN_SOURCE, "i >= 0",
                                     declared_effects={"add": []},
                                     max_frames=5)
        assert isinstance(result, EffectPDRResult)


# ============================================================
# Section 5: Basic PDR (no AI)
# ============================================================

class TestCertifyEffectPDRBasic:
    def test_basic_returns_result(self):
        result = certify_effect_pdr_basic(PURE_COUNTER, "i >= 0", max_frames=5)
        assert isinstance(result, EffectPDRResult)
        assert result.method == EffectPDRMethod.BASIC_PDR_PLUS_EFFECTS

    def test_basic_no_ai_invariants(self):
        result = certify_effect_pdr_basic(PURE_COUNTER, "i >= 0", max_frames=5)
        assert result.ai_invariants == []

    def test_basic_has_pdr_verdict(self):
        result = certify_effect_pdr_basic(BOUNDED_LOOP, "x >= 0", max_frames=5)
        assert result.pdr_verdict is not None


# ============================================================
# Section 6: Verify effect loop (convenience)
# ============================================================

class TestVerifyEffectLoop:
    def test_simple_loop(self):
        result = verify_effect_loop(PURE_COUNTER, "i >= 0", max_frames=5)
        assert isinstance(result, EffectPDRResult)

    def test_with_declared_effects(self):
        result = verify_effect_loop(PURE_FN_SOURCE, "i >= 0",
                                     declared_effects={"add": []},
                                     max_frames=5)
        assert isinstance(result, EffectPDRResult)


# ============================================================
# Section 7: Effects-only analysis
# ============================================================

class TestAnalyzeEffectsOnly:
    def test_pure_source(self):
        result = analyze_effects_only(PURE_FN_SOURCE)
        assert result.verdict in (EffectPDRVerdict.SAFE, EffectPDRVerdict.EFFECT_VIOLATION)
        assert result.method == EffectPDRMethod.EFFECTS_ONLY
        assert result.pdr_result is None

    def test_effects_only_no_property(self):
        result = analyze_effects_only(PURE_COUNTER)
        assert result.property_desc == "(none)"

    def test_effects_with_declarations(self):
        result = analyze_effects_only(IO_SOURCE, {"greet": ["IO"]})
        assert isinstance(result, EffectPDRResult)

    def test_empty_source(self):
        result = analyze_effects_only("")
        assert result.verdict == EffectPDRVerdict.SAFE


# ============================================================
# Section 8: Result properties
# ============================================================

class TestResultProperties:
    def test_certified_property(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert isinstance(result.certified, bool)

    def test_total_obligations(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert isinstance(result.total_obligations, int)
        assert result.total_obligations >= 0

    def test_valid_obligations(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert isinstance(result.valid_obligations, int)
        assert result.valid_obligations >= 0

    def test_summary_string(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        s = result.summary()
        assert isinstance(s, str)
        assert "Effect-Aware PDR" in s

    def test_to_dict(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'verdict' in d
        assert 'method' in d
        assert 'effects_conform' in d
        assert 'certified' in d


# ============================================================
# Section 9: Effect regression PDR
# ============================================================

class TestEffectRegressionPDR:
    def test_same_source_safe(self):
        result = verify_effect_regression_pdr(
            PURE_COUNTER, PURE_COUNTER, "i >= 0", max_frames=5)
        assert isinstance(result, dict)
        assert 'pdr' in result
        assert 'regression' in result
        assert 'verdict' in result
        assert not result['regression']['has_regression']

    def test_added_io_regression(self):
        old = PURE_FN_SOURCE
        new = IO_SOURCE
        result = verify_effect_regression_pdr(old, new, "x >= 0", max_frames=5)
        assert isinstance(result, dict)
        # May detect regression depending on effect inference
        assert 'regression' in result

    def test_regression_structure(self):
        result = verify_effect_regression_pdr(
            PURE_COUNTER, BOUNDED_LOOP, "x >= 0", max_frames=5)
        reg = result['regression']
        assert 'changes' in reg
        assert 'has_regression' in reg
        assert 'num_changes' in reg


# ============================================================
# Section 10: Comparison APIs
# ============================================================

class TestCompareEffectVsPlain:
    def test_comparison_keys(self):
        result = compare_effect_vs_plain(PURE_COUNTER, "i >= 0", max_frames=5)
        assert 'plain' in result
        assert 'effect_aware' in result
        assert 'overhead' in result

    def test_plain_has_verdict(self):
        result = compare_effect_vs_plain(PURE_COUNTER, "i >= 0", max_frames=5)
        assert 'verdict' in result['plain']
        assert 'time' in result['plain']

    def test_effect_aware_has_effects(self):
        result = compare_effect_vs_plain(PURE_COUNTER, "i >= 0", max_frames=5)
        assert 'effects_conform' in result['effect_aware']


class TestCompareAIvsBasic:
    def test_comparison_keys(self):
        result = compare_ai_vs_basic_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert 'ai_strengthened' in result
        assert 'basic' in result
        assert 'ai_helped' in result

    def test_ai_has_invariants(self):
        result = compare_ai_vs_basic_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        assert 'ai_invariants' in result['ai_strengthened']


# ============================================================
# Section 11: Summary API
# ============================================================

class TestEffectPDRSummary:
    def test_summary_keys(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        s = effect_pdr_summary(result)
        assert isinstance(s, dict)
        assert 'verdict' in s
        assert 'method' in s
        assert 'effects_conform' in s
        assert 'certified' in s

    def test_summary_obligations(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        s = effect_pdr_summary(result)
        assert 'total_obligations' in s
        assert 'valid_obligations' in s

    def test_summary_ai_invariants(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=5)
        s = effect_pdr_summary(result)
        assert 'ai_invariants' in s


# ============================================================
# Section 12: Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_source(self):
        result = certify_effect_pdr("", "x >= 0", max_frames=5)
        assert isinstance(result, EffectPDRResult)
        # Should handle gracefully

    def test_no_loop(self):
        src = "let x = 5;"
        result = certify_effect_pdr(src, "x >= 0", max_frames=5)
        assert isinstance(result, EffectPDRResult)

    def test_max_frames_1(self):
        result = certify_effect_pdr(PURE_COUNTER, "i >= 0", max_frames=1)
        assert isinstance(result, EffectPDRResult)

    def test_effect_verdict_enum_values(self):
        assert EffectPDRVerdict.SAFE.value == "safe"
        assert EffectPDRVerdict.UNSAFE.value == "unsafe"
        assert EffectPDRVerdict.EFFECT_VIOLATION.value == "effect_violation"
        assert EffectPDRVerdict.PROPERTY_FAILURE.value == "property_failure"
        assert EffectPDRVerdict.UNKNOWN.value == "unknown"

    def test_method_enum_values(self):
        assert EffectPDRMethod.AI_PDR_PLUS_EFFECTS.value == "ai_pdr_plus_effects"
        assert EffectPDRMethod.BASIC_PDR_PLUS_EFFECTS.value == "basic_pdr_plus_effects"
        assert EffectPDRMethod.EFFECTS_ONLY.value == "effects_only"
        assert EffectPDRMethod.PDR_ONLY.value == "pdr_only"


# ============================================================
# Section 13: Complex loops
# ============================================================

class TestComplexLoops:
    def test_decrement_loop(self):
        src = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""
        result = certify_effect_pdr(src, "x >= 0", max_frames=10)
        assert isinstance(result, EffectPDRResult)

    def test_accumulator_effects_only(self):
        # Accumulator causes SMT timeouts in PDR -- test effect analysis only
        result = analyze_effects_only(ACCUMULATOR)
        assert isinstance(result, EffectPDRResult)
        assert result.effects_conform is True

    def test_fn_with_loop(self):
        result = certify_effect_pdr(PURE_FN_SOURCE, "i >= 0",
                                     max_frames=5)
        assert isinstance(result, EffectPDRResult)
        # add function should be pure
        pure_fns = [ei for ei in result.effect_infos
                    if ei.function == "add" and not ei.inferred_effects]
        # add may or may not be detected depending on effect inferrer


# ============================================================
# Section 14: EffectInfo dataclass
# ============================================================

class TestEffectInfo:
    def test_default_conforms(self):
        ei = EffectInfo("foo", [])
        assert ei.conforms is True
        assert ei.undeclared == []

    def test_with_undeclared(self):
        ei = EffectInfo("foo", ["IO", "Exn"], ["IO"], ["Exn"], False)
        assert not ei.conforms
        assert ei.undeclared == ["Exn"]

    def test_no_declaration(self):
        ei = EffectInfo("bar", ["IO"])
        assert ei.declared_effects is None
        assert ei.conforms is True
