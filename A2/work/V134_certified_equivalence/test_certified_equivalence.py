"""Tests for V134: Certified Equivalence Checking"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from certified_equivalence import (
    certify_function_equivalence, certify_program_equivalence,
    certify_regression, certify_partial_equivalence,
    certify_and_check, check_equiv_certificate,
    compare_certified_vs_uncertified, equiv_certificate_summary,
    to_v044_certificate, save_equiv_certificate, load_equiv_certificate,
    EquivCertificate, EquivCertKind, PathPairObligation, CertStatus,
    ProofKind,
)

# C037 for domain constraints
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import Var as SMTVar, App, IntConst, Op as SMTOp, INT, BOOL


# ===== Section 1: Basic function equivalence certification =====

class TestBasicFunctionEquivalence:
    def test_identity_functions(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "equivalent"
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations > 0

    def test_equivalent_with_different_structure(self):
        src1 = "fn f(x) { return x + 1; }"
        src2 = "fn g(x) { return 1 + x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "equivalent"
        assert cert.status == CertStatus.VALID

    def test_not_equivalent_functions(self):
        src1 = "fn f(x) { return x + 1; }"
        src2 = "fn g(x) { return x + 2; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "not_equivalent"
        assert cert.status == CertStatus.INVALID

    def test_not_equivalent_has_counterexample(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x + 1; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.counterexample is not None

    def test_certificate_kind_is_function(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.kind == EquivCertKind.FUNCTION


# ===== Section 2: Program equivalence certification =====

class TestProgramEquivalence:
    def test_simple_program_equivalence(self):
        src1 = "let x = 0;\nlet y = x + 1;"
        src2 = "let x = 0;\nlet y = 1 + x;"
        cert = certify_program_equivalence(src1, src2, {"x": "int"}, output_var="y")
        assert cert.result == "equivalent"
        assert cert.status == CertStatus.VALID

    def test_program_not_equivalent(self):
        src1 = "let x = 0;\nlet y = x * 2;"
        src2 = "let x = 0;\nlet y = x * 3;"
        cert = certify_program_equivalence(src1, src2, {"x": "int"}, output_var="y")
        assert cert.result == "not_equivalent"

    def test_program_with_conditional(self):
        src1 = """
let x = 0;
let y = 0;
if (x > 0) {
    y = x;
} else {
    y = 0 - x;
}
"""
        src2 = """
let x = 0;
let z = 0;
if (x > 0) {
    z = x;
} else {
    z = 0 - x;
}
"""
        cert = certify_program_equivalence(src1, src2, {"x": "int"})
        assert cert.result == "equivalent"

    def test_program_kind(self):
        src1 = "let x = 0;\nlet y = x;"
        src2 = "let x = 0;\nlet y = x;"
        cert = certify_program_equivalence(src1, src2, {"x": "int"})
        assert cert.kind == EquivCertKind.PROGRAM


# ===== Section 3: Regression certification =====

class TestRegressionCertification:
    def test_regression_preserves_behavior(self):
        original = "fn f(x) { return x + x; }"
        refactored = "fn f(x) { return 2 * x; }"
        cert = certify_regression(
            original, refactored, {},
            fn_name="f", param_types={"x": "int"}
        )
        assert cert.result == "equivalent"
        assert cert.kind == EquivCertKind.REGRESSION

    def test_regression_detects_bug(self):
        original = "fn f(x) { return x + 1; }"
        refactored = "fn f(x) { return x; }"
        cert = certify_regression(
            original, refactored, {},
            fn_name="f", param_types={"x": "int"}
        )
        assert cert.result == "not_equivalent"

    def test_regression_program_mode(self):
        original = "let x = 0;\nlet y = x + 1;"
        refactored = "let x = 0;\nlet y = 1 + x;"
        cert = certify_regression(
            original, refactored, {"x": "int"},
            output_var="y"
        )
        assert cert.result == "equivalent"


# ===== Section 4: Path pair obligations =====

class TestPathPairObligations:
    def test_has_path_pairs(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert len(cert.path_pairs) > 0

    def test_path_pairs_match_obligations(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert len(cert.path_pairs) == len(cert.obligations)

    def test_path_pair_has_formula(self):
        src1 = "fn f(x) { return x + 1; }"
        src2 = "fn g(x) { return 1 + x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        for pp in cert.path_pairs:
            assert pp.formula_str != ""

    def test_equivalent_pairs_have_output_equal(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        for pp in cert.path_pairs:
            assert pp.output_equal is True

    def test_conditional_creates_multiple_pairs(self):
        src1 = """
fn f(x) {
    if (x > 0) { return x; }
    return 0 - x;
}
"""
        src2 = """
fn g(x) {
    if (x > 0) { return x; }
    return 0 - x;
}
"""
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert len(cert.path_pairs) >= 2  # At least then x then, else x else


# ===== Section 5: Independent checking =====

class TestIndependentChecking:
    def test_check_valid_certificate(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        checked = check_equiv_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_invalid_certificate(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x + 1; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        checked = check_equiv_certificate(cert)
        assert checked.status == CertStatus.INVALID

    def test_check_sets_metadata_flag(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        checked = check_equiv_certificate(cert)
        assert checked.metadata.get("independently_checked") is True

    def test_check_preserves_claim(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        checked = check_equiv_certificate(cert)
        assert checked.claim == cert.claim


# ===== Section 6: JSON serialization =====

class TestSerialization:
    def test_to_json(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        j = cert.to_json()
        assert isinstance(j, str)
        d = json.loads(j)
        assert d["kind"] == "function"

    def test_from_json_roundtrip(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        j = cert.to_json()
        cert2 = EquivCertificate.from_json(j)
        assert cert2.kind == cert.kind
        assert cert2.result == cert.result
        assert cert2.status == cert.status
        assert len(cert2.obligations) == len(cert.obligations)

    def test_save_load(self, tmp_path):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        path = str(tmp_path / "cert.json")
        save_equiv_certificate(cert, path)
        loaded = load_equiv_certificate(path)
        assert loaded.kind == cert.kind
        assert loaded.result == cert.result
        assert len(loaded.path_pairs) == len(cert.path_pairs)

    def test_path_pairs_serialized(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        d = cert.to_dict()
        assert "path_pairs" in d
        assert len(d["path_pairs"]) > 0


# ===== Section 7: V044 bridge =====

class TestV044Bridge:
    def test_to_v044_certificate(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        v044 = to_v044_certificate(cert)
        assert v044.kind == ProofKind.COMPOSITE
        assert "f(source1)" in v044.claim
        assert v044.status == cert.status

    def test_v044_has_obligations(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        v044 = to_v044_certificate(cert)
        assert len(v044.obligations) == cert.total_obligations

    def test_v044_serializable(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        v044 = to_v044_certificate(cert)
        j = v044.to_json()
        assert isinstance(j, str)
        d = json.loads(j)
        assert d["kind"] == "composite"


# ===== Section 8: certify_and_check =====

class TestCertifyAndCheck:
    def test_function_mode(self):
        src1 = "fn f(x) { return x * 2; }"
        src2 = "fn g(x) { return x + x; }"
        cert = certify_and_check(
            src1, src2,
            fn_name1="f", fn_name2="g", param_types={"x": "int"}
        )
        assert cert.result == "equivalent"
        assert cert.metadata.get("independently_checked") is True

    def test_program_mode(self):
        src1 = "let x = 0;\nlet y = x + 1;"
        src2 = "let x = 0;\nlet y = 1 + x;"
        cert = certify_and_check(src1, src2, symbolic_inputs={"x": "int"}, output_var="y")
        assert cert.result == "equivalent"

    def test_inequivalent_detected(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return 0; }"
        cert = certify_and_check(
            src1, src2,
            fn_name1="f", fn_name2="g", param_types={"x": "int"}
        )
        assert cert.result == "not_equivalent"


# ===== Section 9: Comparison API =====

class TestComparisonAPI:
    def test_compare_returns_dict(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        result = compare_certified_vs_uncertified(
            src1, src2,
            fn_name1="f", fn_name2="g", param_types={"x": "int"}
        )
        assert isinstance(result, dict)
        assert "plain_result" in result
        assert "certified_result" in result
        assert "checked_result" in result

    def test_compare_agreement(self):
        src1 = "fn f(x) { return x + 1; }"
        src2 = "fn g(x) { return 1 + x; }"
        result = compare_certified_vs_uncertified(
            src1, src2,
            fn_name1="f", fn_name2="g", param_types={"x": "int"}
        )
        assert result["agreement"] is True

    def test_compare_has_timing(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        result = compare_certified_vs_uncertified(
            src1, src2,
            fn_name1="f", fn_name2="g", param_types={"x": "int"}
        )
        assert "plain_time_s" in result
        assert "cert_time_s" in result
        assert "overhead_factor" in result


# ===== Section 10: Summary API =====

class TestSummaryAPI:
    def test_summary_returns_dict(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        s = equiv_certificate_summary(cert)
        assert isinstance(s, dict)
        assert s["kind"] == "function"
        assert s["result"] == "equivalent"
        assert s["status"] == "valid"

    def test_summary_counts(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        s = equiv_certificate_summary(cert)
        assert s["total_obligations"] > 0
        assert s["valid"] == s["total_obligations"]
        assert s["invalid"] == 0

    def test_summary_serializable_flag(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        s = equiv_certificate_summary(cert)
        assert s["serializable"] is True


# ===== Section 11: Two-parameter functions =====

class TestTwoParameters:
    def test_two_param_equivalent(self):
        src1 = "fn f(x, y) { return x + y; }"
        src2 = "fn g(x, y) { return y + x; }"
        cert = certify_function_equivalence(
            src1, "f", src2, "g", {"x": "int", "y": "int"}
        )
        assert cert.result == "equivalent"
        assert cert.status == CertStatus.VALID

    def test_two_param_not_equivalent(self):
        src1 = "fn f(x, y) { return x - y; }"
        src2 = "fn g(x, y) { return y - x; }"
        cert = certify_function_equivalence(
            src1, "f", src2, "g", {"x": "int", "y": "int"}
        )
        assert cert.result == "not_equivalent"

    def test_two_param_max(self):
        src1 = """
fn max1(x, y) {
    if (x > y) { return x; }
    return y;
}
"""
        src2 = """
fn max2(x, y) {
    if (y >= x) { return y; }
    return x;
}
"""
        cert = certify_function_equivalence(
            src1, "max1", src2, "max2", {"x": "int", "y": "int"}
        )
        assert cert.result == "equivalent"


# ===== Section 12: Algebraic identities =====

class TestAlgebraicIdentities:
    def test_double_vs_add(self):
        src1 = "fn f(x) { return x * 2; }"
        src2 = "fn g(x) { return x + x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "equivalent"

    def test_subtract_self(self):
        src1 = "fn f(x) { return x - x; }"
        src2 = "fn g(x) { return 0; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "equivalent"

    def test_add_zero(self):
        src1 = "fn f(x) { return x + 0; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "equivalent"


# ===== Section 13: Partial equivalence =====

class TestPartialEquivalence:
    def test_partial_equiv_positive_domain(self):
        src1 = "let x = 0;\nlet y = x;"
        src2 = "let x = 0;\nlet y = x;"
        x_var = SMTVar("x", INT)
        domain = [App(SMTOp.GE, [x_var, IntConst(0)], BOOL)]
        cert = certify_partial_equivalence(
            src1, src2, {"x": "int"}, domain, output_var="y"
        )
        assert cert.result == "equivalent"
        assert cert.kind == EquivCertKind.PARTIAL

    def test_partial_has_domain_metadata(self):
        src1 = "let x = 0;\nlet y = x + 1;"
        src2 = "let x = 0;\nlet y = 1 + x;"
        x_var = SMTVar("x", INT)
        domain = [App(SMTOp.GT, [x_var, IntConst(0)], BOOL)]
        cert = certify_partial_equivalence(
            src1, src2, {"x": "int"}, domain, output_var="y"
        )
        assert "domain_constraints" in cert.metadata


# ===== Section 14: Edge cases =====

class TestEdgeCases:
    def test_empty_function_body(self):
        src1 = "fn f(x) { return 0; }"
        src2 = "fn g(x) { return 0; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "equivalent"

    def test_constant_functions(self):
        src1 = "fn f(x) { return 42; }"
        src2 = "fn g(x) { return 42; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "equivalent"

    def test_different_constants(self):
        src1 = "fn f(x) { return 42; }"
        src2 = "fn g(x) { return 43; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.result == "not_equivalent"

    def test_cert_summary_method(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        s = cert.summary()
        assert "Equivalence Certificate" in s
        assert "valid" in s.lower()


# ===== Section 15: Metadata =====

class TestMetadata:
    def test_function_metadata(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.metadata["fn_name1"] == "f"
        assert cert.metadata["fn_name2"] == "g"
        assert "paths_checked" in cert.metadata

    def test_program_metadata(self):
        src1 = "let x = 0;\nlet y = x;"
        src2 = "let x = 0;\nlet y = x;"
        cert = certify_program_equivalence(src1, src2, {"x": "int"})
        assert "symbolic_inputs" in cert.metadata

    def test_timestamp_set(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.timestamp != ""


# ===== Section 16: Obligation status counts =====

class TestObligationCounts:
    def test_all_valid_for_equivalent(self):
        src1 = "fn f(x) { return x + 1; }"
        src2 = "fn g(x) { return 1 + x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.valid_obligations == cert.total_obligations
        assert cert.invalid_obligations == 0

    def test_some_invalid_for_not_equivalent(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x + 1; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        assert cert.invalid_obligations > 0


# ===== Section 17: Conditional equivalence =====

class TestConditionalEquivalence:
    def test_abs_implementations(self):
        src1 = """
fn abs1(x) {
    if (x >= 0) { return x; }
    return 0 - x;
}
"""
        src2 = """
fn abs2(x) {
    if (x < 0) { return 0 - x; }
    return x;
}
"""
        cert = certify_function_equivalence(
            src1, "abs1", src2, "abs2", {"x": "int"}
        )
        assert cert.result == "equivalent"
        assert cert.status == CertStatus.VALID

    def test_sign_implementations(self):
        src1 = """
fn sign1(x) {
    if (x > 0) { return 1; }
    if (x < 0) { return 0 - 1; }
    return 0;
}
"""
        src2 = """
fn sign2(x) {
    if (x == 0) { return 0; }
    if (x > 0) { return 1; }
    return 0 - 1;
}
"""
        cert = certify_function_equivalence(
            src1, "sign1", src2, "sign2", {"x": "int"}
        )
        assert cert.result == "equivalent"


# ===== Section 18: SMT-LIB2 formulas in certificates =====

class TestSMTLIB:
    def test_obligations_have_smt(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        for pp in cert.path_pairs:
            assert pp.formula_smt != ""
            assert "(set-logic LIA)" in pp.formula_smt

    def test_smt_has_declare_const(self):
        src1 = "fn f(x) { return x + 1; }"
        src2 = "fn g(x) { return 1 + x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        for pp in cert.path_pairs:
            if "declare-const" in pp.formula_smt:
                assert "Int" in pp.formula_smt

    def test_smt_has_check_sat(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        for pp in cert.path_pairs:
            assert "(check-sat)" in pp.formula_smt


# ===== Section 19: Roundtrip check after serialization =====

class TestRoundtripCheck:
    def test_serialize_then_check(self):
        src1 = "fn f(x) { return x * 2; }"
        src2 = "fn g(x) { return x + x; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        j = cert.to_json()
        loaded = EquivCertificate.from_json(j)
        checked = check_equiv_certificate(loaded)
        assert checked.status == CertStatus.VALID

    def test_serialize_invalid_then_check(self):
        src1 = "fn f(x) { return x; }"
        src2 = "fn g(x) { return x + 5; }"
        cert = certify_function_equivalence(src1, "f", src2, "g", {"x": "int"})
        j = cert.to_json()
        loaded = EquivCertificate.from_json(j)
        checked = check_equiv_certificate(loaded)
        assert checked.status == CertStatus.INVALID


# ===== Section 20: Strength reduction equivalence =====

class TestStrengthReduction:
    def test_multiply_by_2_vs_shift(self):
        src1 = "fn f(x) { return x * 2; }"
        src2 = "fn g(x) { return x + x; }"
        cert = certify_and_check(
            src1, src2,
            fn_name1="f", fn_name2="g", param_types={"x": "int"}
        )
        assert cert.result == "equivalent"
        assert cert.status == CertStatus.VALID

    def test_triple_vs_add(self):
        src1 = "fn f(x) { return x * 3; }"
        src2 = "fn g(x) { return x + x + x; }"
        cert = certify_and_check(
            src1, src2,
            fn_name1="f", fn_name2="g", param_types={"x": "int"}
        )
        assert cert.result == "equivalent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
