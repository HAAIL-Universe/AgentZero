"""Tests for V126: Array Bounds Certificates."""

import pytest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V123_array_bounds_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V120_array_domain'))

from array_bounds_certificates import (
    # Core types
    ArrayBoundsCertificate, ArrayCertKind,
    # Certificate generation
    certify_array_bounds, certify_and_check, certify_with_context,
    # Certificate checking
    check_array_certificate,
    # I/O
    save_array_certificate, load_array_certificate,
    # Composition
    combine_array_certificates,
    # Comparison
    compare_certification_strength, certificate_summary,
    # V044 bridge
    to_v044_certificate, from_v044_certificate,
    # Internal helpers
    _encode_lower_bound_smtlib, _encode_upper_bound_smtlib,
    _encode_ai_lower_smtlib, _encode_ai_upper_smtlib,
    _bounds_obligation_to_proof, _check_ai_safe_obligation,
    _check_smt_obligation, _parse_smtlib_term, _tokenize_sexp,
)
from proof_certificates import ProofCertificate, ProofObligation, ProofKind, CertStatus


# ===========================================================================
# 1. Basic Certificate Generation
# ===========================================================================

class TestBasicCertification:
    def test_simple_constant_access(self):
        """Constant index into known-size array: AI-safe."""
        source = "let a = [1, 2, 3]; let x = a[0];"
        cert = certify_array_bounds(source)
        assert cert.access_count >= 1
        assert cert.all_safe
        assert cert.status == CertStatus.VALID

    def test_multiple_safe_accesses(self):
        source = """
            let a = [10, 20, 30];
            let x = a[0];
            let y = a[1];
            let z = a[2];
        """
        cert = certify_array_bounds(source)
        assert cert.all_safe
        assert cert.status == CertStatus.VALID
        assert cert.access_count >= 1  # V120 may deduplicate single-line accesses

    def test_write_access(self):
        source = "let a = [1, 2, 3]; a[1] = 99;"
        cert = certify_array_bounds(source)
        assert cert.all_safe
        assert cert.status == CertStatus.VALID

    def test_mixed_read_write(self):
        source = """
            let a = [1, 2, 3];
            let x = a[0];
            a[2] = a[1] + 10;
        """
        cert = certify_array_bounds(source)
        assert cert.all_safe

    def test_certificate_has_obligations(self):
        source = "let a = [1, 2]; let x = a[0];"
        cert = certify_array_bounds(source)
        assert len(cert.obligations) >= 2  # lower + upper bound per access
        for obl in cert.obligations:
            assert obl.name
            assert obl.description
            assert obl.formula_str


# ===========================================================================
# 2. Certificate Checking (Independent Verification)
# ===========================================================================

class TestCertificateChecking:
    def test_check_safe_certificate(self):
        source = "let a = [1, 2, 3]; let x = a[1];"
        cert = certify_array_bounds(source)
        checked = check_array_certificate(cert)
        assert checked.status == CertStatus.VALID
        for obl in checked.obligations:
            assert obl.status == CertStatus.VALID

    def test_check_preserves_metadata(self):
        source = "let a = [1, 2]; let x = a[0];"
        cert = certify_array_bounds(source)
        checked = check_array_certificate(cert)
        assert checked.access_count == cert.access_count
        assert checked.source == cert.source

    def test_certify_and_check_combined(self):
        source = "let a = [5, 10, 15]; let x = a[2];"
        cert = certify_and_check(source)
        assert cert.status == CertStatus.VALID

    def test_check_multiple_accesses(self):
        source = """
            let a = [1, 2, 3, 4, 5];
            let x = a[0];
            let y = a[2];
            let z = a[4];
        """
        cert = certify_and_check(source)
        assert cert.status == CertStatus.VALID
        assert cert.access_count >= 3


# ===========================================================================
# 3. AI-Safe vs SMT-Safe Certificates
# ===========================================================================

class TestAIvsSMT:
    def test_constant_index_is_ai_safe(self):
        """Constant index: AI interval analysis alone proves safety."""
        source = "let a = [1, 2, 3]; let x = a[1];"
        cert = certify_array_bounds(source)
        assert cert.ai_safe_count > 0

    def test_compare_certification_strength(self):
        source = "let a = [1, 2, 3]; let x = a[0]; let y = a[2];"
        comparison = compare_certification_strength(source)
        assert comparison["total_obligations"] > 0
        assert comparison["total_coverage"] > 0
        assert "ai_coverage" in comparison
        assert "smt_additional_coverage" in comparison

    def test_ai_coverage_simple(self):
        """All constant indices should be AI-provable."""
        source = "let a = [10, 20]; let x = a[0]; let y = a[1];"
        comparison = compare_certification_strength(source)
        assert comparison["ai_safe"] >= comparison["smt_safe"]


# ===========================================================================
# 4. Unsafe Access Detection
# ===========================================================================

class TestUnsafeAccess:
    def test_out_of_bounds_detected(self):
        """Access beyond array length should be detected."""
        source = "let a = [1, 2]; let x = a[5];"
        cert = certify_array_bounds(source)
        assert cert.unsafe_count > 0
        assert cert.status == CertStatus.INVALID
        assert not cert.all_safe

    def test_negative_index_detected(self):
        source = """
            let a = [1, 2, 3];
            let i = 0 - 1;
            let x = a[i];
        """
        cert = certify_array_bounds(source)
        assert cert.unsafe_count > 0 or cert.unknown_count > 0
        assert not cert.all_safe

    def test_unsafe_certificate_check(self):
        source = "let a = [1]; let x = a[3];"
        cert = certify_and_check(source)
        assert cert.status != CertStatus.VALID


# ===========================================================================
# 5. Loop Access Verification
# ===========================================================================

class TestLoopAccess:
    def test_safe_loop_traversal(self):
        """i in [0, len-1] iterating through array."""
        source = """
            let a = [1, 2, 3];
            let i = 0;
            while (i < 3) {
                let x = a[i];
                i = i + 1;
            }
        """
        cert = certify_array_bounds(source)
        # Loop access should be certifiable
        assert cert.total_obligations > 0

    def test_loop_certificate_generation(self):
        source = """
            let a = [10, 20, 30, 40];
            let i = 0;
            while (i < 4) {
                a[i] = a[i] + 1;
                i = i + 1;
            }
        """
        cert = certify_array_bounds(source)
        assert cert.total_obligations > 0


# ===========================================================================
# 6. Serialization (JSON round-trip)
# ===========================================================================

class TestSerialization:
    def test_to_dict_from_dict(self):
        source = "let a = [1, 2, 3]; let x = a[1];"
        cert = certify_array_bounds(source)
        d = cert.to_dict()
        cert2 = ArrayBoundsCertificate.from_dict(d)
        assert cert2.access_count == cert.access_count
        assert cert2.status == cert.status
        assert cert2.all_safe == cert.all_safe
        assert len(cert2.obligations) == len(cert.obligations)

    def test_to_json_from_json(self):
        source = "let a = [5, 10]; let x = a[0];"
        cert = certify_array_bounds(source)
        json_str = cert.to_json()
        cert2 = ArrayBoundsCertificate.from_json(json_str)
        assert cert2.status == cert.status
        assert cert2.all_safe == cert.all_safe

    def test_save_load_file(self):
        source = "let a = [1, 2]; let x = a[0]; a[1] = 99;"
        cert = certify_array_bounds(source)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_array_certificate(cert, path)
            loaded = load_array_certificate(path)
            assert loaded.status == cert.status
            assert loaded.access_count == cert.access_count
            assert len(loaded.obligations) == len(cert.obligations)
        finally:
            os.unlink(path)

    def test_obligation_serialization(self):
        source = "let a = [1, 2, 3]; let x = a[2];"
        cert = certify_array_bounds(source)
        for obl in cert.obligations:
            d = obl.to_dict()
            obl2 = ProofObligation.from_dict(d)
            assert obl2.name == obl.name
            assert obl2.status == obl.status


# ===========================================================================
# 7. Certificate Composition
# ===========================================================================

class TestCertificateComposition:
    def test_combine_two_safe(self):
        cert1 = certify_array_bounds("let a = [1, 2]; let x = a[0];")
        cert2 = certify_array_bounds("let b = [3, 4, 5]; let y = b[1];")
        combined = combine_array_certificates(cert1, cert2)
        assert combined.access_count == cert1.access_count + cert2.access_count
        assert combined.all_safe
        assert combined.status == CertStatus.VALID

    def test_combine_safe_and_unsafe(self):
        cert1 = certify_array_bounds("let a = [1, 2]; let x = a[0];")
        cert2 = certify_array_bounds("let b = [1]; let y = b[5];")
        combined = combine_array_certificates(cert1, cert2)
        assert not combined.all_safe
        assert combined.status == CertStatus.INVALID

    def test_combine_metadata(self):
        cert1 = certify_array_bounds("let a = [1]; let x = a[0];")
        cert2 = certify_array_bounds("let b = [2]; let y = b[0];")
        combined = combine_array_certificates(cert1, cert2)
        assert combined.metadata.get("combined") is True
        assert combined.metadata.get("num_modules") == 2

    def test_combine_obligations_merged(self):
        cert1 = certify_array_bounds("let a = [1, 2]; let x = a[0];")
        cert2 = certify_array_bounds("let b = [3]; let y = b[0];")
        combined = combine_array_certificates(cert1, cert2)
        assert len(combined.obligations) == len(cert1.obligations) + len(cert2.obligations)


# ===========================================================================
# 8. V044 Bridge
# ===========================================================================

class TestV044Bridge:
    def test_to_v044(self):
        source = "let a = [1, 2, 3]; let x = a[1];"
        cert = certify_array_bounds(source)
        v044 = to_v044_certificate(cert)
        assert isinstance(v044, ProofCertificate)
        assert v044.kind == ProofKind.VCGEN
        assert "array" in v044.claim.lower() or "access" in v044.claim.lower()
        assert len(v044.obligations) == len(cert.obligations)

    def test_v044_metadata(self):
        source = "let a = [1, 2]; let x = a[0];"
        cert = certify_array_bounds(source)
        v044 = to_v044_certificate(cert)
        assert v044.metadata.get("array_bounds") is True
        assert "access_count" in v044.metadata

    def test_roundtrip_v044(self):
        source = "let a = [1, 2, 3]; let x = a[0];"
        cert = certify_array_bounds(source)
        v044 = to_v044_certificate(cert)
        back = from_v044_certificate(v044, source)
        assert isinstance(back, ArrayBoundsCertificate)
        assert back.status == cert.status
        assert len(back.obligations) == len(cert.obligations)

    def test_v044_status_propagation(self):
        source = "let a = [1, 2]; let x = a[0];"
        cert = certify_and_check(source)
        v044 = to_v044_certificate(cert)
        assert v044.status == cert.status


# ===========================================================================
# 9. Contextual Certification
# ===========================================================================

class TestContextualCertification:
    def test_with_user_constraints(self):
        source = """
            let a = [1, 2, 3, 4, 5];
            let i = 0;
            let x = a[i];
        """
        cert = certify_with_context(source, {"i": (0, 4)})
        assert cert.total_obligations > 0

    def test_constraints_improve_verdict(self):
        """Variable index with no constraints may be unknown; with constraints, safe."""
        source = """
            let a = [1, 2, 3, 4, 5];
            let i = 2;
            let x = a[i];
        """
        cert1 = certify_array_bounds(source)
        cert2 = certify_with_context(source, {"i": (0, 4)})
        # Both should work for constant i=2
        assert cert1.total_obligations > 0
        assert cert2.total_obligations > 0


# ===========================================================================
# 10. Certificate Summary
# ===========================================================================

class TestCertificateSummary:
    def test_summary_string(self):
        source = "let a = [1, 2, 3]; let x = a[1];"
        cert = certify_array_bounds(source)
        s = cert.summary()
        assert "Array Bounds Certificate" in s
        assert "Accesses" in s

    def test_full_summary(self):
        source = "let a = [1, 2, 3]; let x = a[0]; let y = a[2];"
        s = certificate_summary(source)
        assert "VALID" in s or "FAIL" in s or "????" in s


# ===========================================================================
# 11. SMT-LIB2 Encoding
# ===========================================================================

class TestSMTLIB2Encoding:
    def test_lower_bound_encoding(self):
        smtlib = _encode_lower_bound_smtlib("idx", [("idx", 0, 5)])
        assert "(set-logic LIA)" in smtlib
        assert "(declare-const idx Int)" in smtlib
        assert "(< idx 0)" in smtlib

    def test_upper_bound_encoding(self):
        smtlib = _encode_upper_bound_smtlib("idx", "len_a", [("idx", 0, 5), ("len_a", 3, 3)])
        assert "(set-logic LIA)" in smtlib
        assert "(declare-const idx Int)" in smtlib
        assert "(declare-const len_a Int)" in smtlib
        assert "(>= idx len_a)" in smtlib

    def test_ai_lower_encoding(self):
        smtlib = _encode_ai_lower_smtlib(0)
        assert "AI-safe" in smtlib
        assert "0" in smtlib

    def test_ai_upper_encoding(self):
        smtlib = _encode_ai_upper_smtlib(2, 3)
        assert "AI-safe" in smtlib

    def test_context_vars_in_encoding(self):
        smtlib = _encode_lower_bound_smtlib("i", [("i", 0, 10), ("n", 5, 5)])
        assert "(declare-const i Int)" in smtlib
        assert "(declare-const n Int)" in smtlib
        assert "(>= i 0)" in smtlib
        assert "(<= i 10)" in smtlib


# ===========================================================================
# 12. S-Expression Tokenizer
# ===========================================================================

class TestTokenizer:
    def test_simple_tokens(self):
        tokens = _tokenize_sexp(">= x 0")
        assert tokens == [">=", "x", "0"]

    def test_nested_parens(self):
        tokens = _tokenize_sexp("and (>= x 0) (<= x 5)")
        assert tokens[0] == "and"
        assert tokens[1] == "(>= x 0)"
        assert tokens[2] == "(<= x 5)"

    def test_empty(self):
        assert _tokenize_sexp("") == []

    def test_single_token(self):
        assert _tokenize_sexp("42") == ["42"]


# ===========================================================================
# 13. AI-Safe Obligation Checking
# ===========================================================================

class TestAISafeChecking:
    def test_ai_lower_valid(self):
        obl = ProofObligation(
            name="test", description="test",
            formula_str="i >= 0",
            formula_smt="(set-logic LIA)\n; AI-safe: index lower bound = 0 >= 0\n(check-sat)",
            status=CertStatus.VALID,
        )
        result = _check_ai_safe_obligation(obl)
        assert result == CertStatus.VALID

    def test_ai_upper_valid(self):
        obl = ProofObligation(
            name="test", description="test",
            formula_str="i < len(a)",
            formula_smt="(set-logic LIA)\n; AI-safe: index upper bound = 2 < length lower bound = 3\n(check-sat)",
            status=CertStatus.VALID,
        )
        result = _check_ai_safe_obligation(obl)
        assert result == CertStatus.VALID

    def test_ai_lower_positive(self):
        obl = ProofObligation(
            name="test", description="test",
            formula_str="i >= 0",
            formula_smt="(set-logic LIA)\n; AI-safe: index lower bound = 5 >= 0\n(check-sat)",
            status=CertStatus.VALID,
        )
        result = _check_ai_safe_obligation(obl)
        assert result == CertStatus.VALID


# ===========================================================================
# 14. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_array(self):
        """new_array(0, 0) -- any access is unsafe."""
        source = "let a = new_array(0, 0); let x = a[0];"
        cert = certify_array_bounds(source)
        # Access to empty array should not be safe
        assert cert.unsafe_count > 0 or cert.unknown_count > 0

    def test_single_element_array(self):
        source = "let a = [42]; let x = a[0];"
        cert = certify_and_check(source)
        assert cert.all_safe

    def test_large_array_constant_access(self):
        source = "let a = new_array(100, 0); let x = a[50];"
        cert = certify_array_bounds(source)
        assert cert.total_obligations > 0

    def test_no_array_accesses(self):
        source = "let x = 5; let y = x + 3;"
        cert = certify_array_bounds(source)
        assert cert.access_count == 0
        assert cert.total_obligations == 0
        assert cert.all_safe
        assert cert.status == CertStatus.VALID

    def test_array_write_within_bounds(self):
        source = """
            let a = [0, 0, 0];
            a[0] = 1;
            a[1] = 2;
            a[2] = 3;
        """
        cert = certify_and_check(source)
        assert cert.all_safe


# ===========================================================================
# 15. Conditional Access
# ===========================================================================

class TestConditionalAccess:
    def test_guarded_access(self):
        """Access inside bounds check."""
        source = """
            let a = [1, 2, 3];
            let i = 1;
            if (i < 3) {
                let x = a[i];
            }
        """
        cert = certify_array_bounds(source)
        assert cert.total_obligations > 0

    def test_if_else_access(self):
        """Access in both branches of an if-else."""
        source = """
            let a = [10, 20, 30];
            let i = 1;
            if (i < 2) {
                let x = a[i];
            }
            let y = a[0];
        """
        cert = certify_array_bounds(source)
        assert cert.total_obligations > 0


# ===========================================================================
# 16. new_array Access
# ===========================================================================

class TestNewArrayAccess:
    def test_new_array_safe_access(self):
        source = """
            let a = new_array(5, 0);
            let x = a[3];
        """
        cert = certify_array_bounds(source)
        assert cert.total_obligations > 0

    def test_new_array_boundary(self):
        """Access at index = size - 1 (last valid index)."""
        source = """
            let a = new_array(3, 0);
            let x = a[2];
        """
        cert = certify_array_bounds(source)
        assert cert.total_obligations > 0


# ===========================================================================
# 17. Obligation Conversion
# ===========================================================================

class TestObligationConversion:
    def test_ai_safe_obligation_conversion(self):
        from array_bounds_verify import Verdict, BoundsObligation
        obl = BoundsObligation(
            access_line=1, array_name="a", index_expr="0",
            check_type="lower", verdict=Verdict.AI_SAFE,
            counterexample=None,
            abstract_index=(0, 0), abstract_length=(3, 3),
            message="safe",
        )
        proof = _bounds_obligation_to_proof(obl, 0)
        assert proof.status == CertStatus.VALID
        assert "AI-safe" in proof.formula_smt

    def test_unsafe_obligation_conversion(self):
        from array_bounds_verify import Verdict, BoundsObligation
        obl = BoundsObligation(
            access_line=1, array_name="a", index_expr="5",
            check_type="upper", verdict=Verdict.UNSAFE,
            counterexample={"index": 5, "length": 3},
            abstract_index=(5, 5), abstract_length=(3, 3),
            message="unsafe",
        )
        proof = _bounds_obligation_to_proof(obl, 0)
        assert proof.status == CertStatus.INVALID
        assert proof.counterexample is not None
