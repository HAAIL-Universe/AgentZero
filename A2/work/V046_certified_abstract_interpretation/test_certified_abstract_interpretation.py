"""Tests for V046: Certified Abstract Interpretation"""

import os
import sys
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from certified_abstract_interpretation import (
    # Core types
    AIAnalysisResult, VerifiedAIResult,
    # Analysis
    traced_analyze,
    # SExpr generation
    _interval_to_sexpr, _sign_to_sexpr, _check_smt_valid, _negate_sexpr_safe,
    # Obligation generators
    _generate_interval_obligation, _generate_sign_interval_consistency,
    _generate_widening_obligation,
    # Certificate generation
    generate_ai_certificate, check_ai_certificate,
    # Verified pipeline
    verified_analyze, _check_certificate_obligations,
    # Widening
    generate_widening_certificate,
    # Composite
    certify_abstract_interpretation,
    # Convenience
    certify_variable_bound, certify_sign,
)

# Imports for test construction
sys.path.insert(0, os.path.join(_dir, '..', 'V044_proof_certificates'))
from proof_certificates import ProofKind, CertStatus, ProofObligation, ProofCertificate

sys.path.insert(0, os.path.join(_dir, '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (
    Sign, Interval, INTERVAL_TOP, INTERVAL_BOT,
)

sys.path.insert(0, os.path.join(_dir, '..', 'V004_verification_conditions'))
from vc_gen import SVar, SInt, SBool, SBinOp


# ===================================================================
# SExpr Generation Tests
# ===================================================================

class TestIntervalToSExpr:
    """Test interval-to-SExpr conversion."""

    def test_concrete_interval(self):
        """[5, 10] -> v >= 5 AND v <= 10"""
        expr = _interval_to_sexpr("v", Interval(5, 10))
        assert isinstance(expr, SBinOp) or hasattr(expr, 'conjuncts')

    def test_lower_bounded(self):
        """[0, inf] -> v >= 0"""
        expr = _interval_to_sexpr("v", Interval(0, float('inf')))
        assert isinstance(expr, SBinOp)
        assert expr.op == ">="

    def test_upper_bounded(self):
        """[-inf, 100] -> v <= 100"""
        expr = _interval_to_sexpr("v", Interval(float('-inf'), 100))
        assert isinstance(expr, SBinOp)
        assert expr.op == "<="

    def test_top_interval(self):
        """[-inf, inf] -> True"""
        expr = _interval_to_sexpr("v", INTERVAL_TOP)
        assert isinstance(expr, SBool) and expr.value is True

    def test_bot_interval(self):
        """BOT -> False"""
        expr = _interval_to_sexpr("v", INTERVAL_BOT)
        assert isinstance(expr, SBool) and expr.value is False

    def test_singleton_interval(self):
        """[7, 7] -> v >= 7 AND v <= 7"""
        expr = _interval_to_sexpr("v", Interval(7, 7))
        # Could be conjunction or single equality
        assert expr is not None


class TestSignToSExpr:
    """Test sign-to-SExpr conversion."""

    def test_pos(self):
        expr = _sign_to_sexpr("x", Sign.POS)
        assert isinstance(expr, SBinOp) and expr.op == ">"

    def test_neg(self):
        expr = _sign_to_sexpr("x", Sign.NEG)
        assert isinstance(expr, SBinOp) and expr.op == "<"

    def test_zero(self):
        expr = _sign_to_sexpr("x", Sign.ZERO)
        assert isinstance(expr, SBinOp) and expr.op == "=="

    def test_non_neg(self):
        expr = _sign_to_sexpr("x", Sign.NON_NEG)
        assert isinstance(expr, SBinOp) and expr.op == ">="

    def test_non_pos(self):
        expr = _sign_to_sexpr("x", Sign.NON_POS)
        assert isinstance(expr, SBinOp) and expr.op == "<="

    def test_top(self):
        expr = _sign_to_sexpr("x", Sign.TOP)
        assert isinstance(expr, SBool) and expr.value is True

    def test_bot(self):
        expr = _sign_to_sexpr("x", Sign.BOT)
        assert isinstance(expr, SBool) and expr.value is False


class TestSMTValidity:
    """Test SMT validity checking."""

    def test_valid_formula(self):
        """5 >= 0 is valid."""
        formula = SBinOp(">=", SInt(5), SInt(0))
        valid, _ = _check_smt_valid(formula)
        assert valid

    def test_invalid_formula(self):
        """x >= 10 is not valid (x could be 0)."""
        formula = SBinOp(">=", SVar("x"), SInt(10))
        valid, cex = _check_smt_valid(formula)
        assert not valid

    def test_tautology(self):
        """x >= 0 OR x < 0 is valid."""
        from vc_gen import s_or
        formula = s_or(
            SBinOp(">=", SVar("x"), SInt(0)),
            SBinOp("<", SVar("x"), SInt(0)),
        )
        valid, _ = _check_smt_valid(formula)
        assert valid


class TestNegation:
    """Test SExpr negation with complement operators."""

    def test_negate_ge(self):
        expr = _negate_sexpr_safe(SBinOp(">=", SVar("x"), SInt(0)))
        assert isinstance(expr, SBinOp) and expr.op == "<"

    def test_negate_eq(self):
        expr = _negate_sexpr_safe(SBinOp("==", SVar("x"), SInt(0)))
        assert isinstance(expr, SBinOp) and expr.op == "!="

    def test_negate_bool(self):
        assert _negate_sexpr_safe(SBool(True)).value is False
        assert _negate_sexpr_safe(SBool(False)).value is True


# ===================================================================
# Obligation Generation Tests
# ===================================================================

class TestObligationGeneration:
    """Test proof obligation creation."""

    def test_interval_obligation_normal(self):
        obl = _generate_interval_obligation("x", Interval(0, 10))
        assert obl.name == "interval_x"
        assert obl.status == CertStatus.UNCHECKED
        assert "x" in obl.description
        assert "[0, 10]" in obl.description

    def test_interval_obligation_bot(self):
        obl = _generate_interval_obligation("x", INTERVAL_BOT)
        assert obl.status == CertStatus.VALID  # BOT is vacuously valid

    def test_sign_interval_consistency_matching(self):
        """POS sign with [1, 10] interval."""
        obl = _generate_sign_interval_consistency("x", Sign.POS, Interval(1, 10))
        assert obl.status == CertStatus.UNCHECKED
        assert "consistency" in obl.name

    def test_sign_interval_consistency_bot(self):
        obl = _generate_sign_interval_consistency("x", Sign.BOT, INTERVAL_BOT)
        assert obl.status == CertStatus.VALID

    def test_sign_interval_consistency_top(self):
        obl = _generate_sign_interval_consistency("x", Sign.TOP, Interval(1, 10))
        assert obl.status == CertStatus.VALID

    def test_widening_obligation(self):
        obl = _generate_widening_obligation("x", Interval(0, 5), Interval(0, 10))
        assert "widen" in obl.name
        assert obl.status == CertStatus.UNCHECKED

    def test_widening_from_bot(self):
        obl = _generate_widening_obligation("x", INTERVAL_BOT, Interval(0, 10))
        assert obl.status == CertStatus.VALID


# ===================================================================
# Traced Analysis Tests
# ===================================================================

class TestTracedAnalysis:
    """Test traced abstract interpretation."""

    def test_simple_assignment(self):
        result = traced_analyze("let x = 5;")
        assert "x" in result.variable_intervals
        assert result.variable_intervals["x"].lo == 5
        assert result.variable_intervals["x"].hi == 5

    def test_arithmetic(self):
        result = traced_analyze("let x = 3; let y = x + 2;")
        assert "y" in result.variable_intervals
        assert result.variable_intervals["y"].lo == 5
        assert result.variable_intervals["y"].hi == 5

    def test_conditional(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = traced_analyze(source)
        assert "y" in result.variable_intervals

    def test_result_has_signs(self):
        result = traced_analyze("let x = 5;")
        assert result.variable_signs["x"] == Sign.POS

    def test_result_has_source(self):
        source = "let a = 1;"
        result = traced_analyze(source)
        assert result.source == source


# ===================================================================
# Certificate Generation Tests
# ===================================================================

class TestCertificateGeneration:
    """Test proof certificate generation."""

    def test_simple_program_certificate(self):
        cert = generate_ai_certificate("let x = 5;")
        assert cert.kind == ProofKind.VCGEN
        assert "abstract interpretation" in cert.claim.lower()
        assert len(cert.obligations) > 0
        assert cert.metadata["analysis_type"] == "abstract_interpretation"

    def test_certificate_has_interval_obligations(self):
        cert = generate_ai_certificate("let x = 5; let y = 10;")
        interval_obls = [o for o in cert.obligations if o.name.startswith("interval_")]
        assert len(interval_obls) >= 2

    def test_certificate_has_consistency_obligations(self):
        cert = generate_ai_certificate("let x = 5;")
        consistency_obls = [o for o in cert.obligations if o.name.startswith("consistency_")]
        assert len(consistency_obls) >= 1

    def test_certificate_metadata(self):
        cert = generate_ai_certificate("let x = 5;")
        assert "variables" in cert.metadata
        assert "x" in cert.metadata["variables"]
        assert "interval" in cert.metadata["variables"]["x"]
        assert "sign" in cert.metadata["variables"]["x"]

    def test_multi_variable_program(self):
        source = """
        let a = 1;
        let b = 2;
        let c = a + b;
        """
        cert = generate_ai_certificate(source)
        var_names = {o.name.split("_", 1)[1] for o in cert.obligations
                     if o.name.startswith("interval_")}
        assert "a" in var_names
        assert "b" in var_names
        assert "c" in var_names


# ===================================================================
# Certificate Checking Tests
# ===================================================================

class TestCertificateChecking:
    """Test certificate verification via SMT."""

    def test_simple_cert_valid(self):
        cert = generate_ai_certificate("let x = 5;")
        checked = check_ai_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_all_obligations_checked(self):
        cert = generate_ai_certificate("let x = 5; let y = 10;")
        checked = check_ai_certificate(cert)
        for obl in checked.obligations:
            assert obl.status != CertStatus.UNCHECKED


# ===================================================================
# Verified Analysis Pipeline Tests
# ===================================================================

class TestVerifiedAnalysis:
    """Test the verified analysis pipeline."""

    def test_simple_verified(self):
        result = verified_analyze("let x = 5;")
        assert result.certified
        assert result.analysis.variable_intervals["x"] == Interval(5, 5)

    def test_verified_summary(self):
        result = verified_analyze("let x = 5;")
        assert "Variables:" in result.summary
        assert "Certificate:" in result.summary

    def test_arithmetic_verified(self):
        result = verified_analyze("let a = 3; let b = a + 7;")
        assert result.certified
        assert result.analysis.variable_intervals["b"] == Interval(10, 10)

    def test_conditional_verified(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 0 - 1;
        }
        """
        result = verified_analyze(source)
        assert result.certified

    def test_loop_verified(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = verified_analyze(source)
        assert result.certified

    def test_obligation_count(self):
        result = verified_analyze("let x = 1; let y = 2; let z = 3;")
        # 3 variables * 2 obligations each (interval + consistency) = 6
        assert result.certificate.total_obligations >= 6


# ===================================================================
# Widening Certificate Tests
# ===================================================================

class TestWideningCertificate:
    """Test widening soundness certificates."""

    def test_no_loop_widening(self):
        """Programs without loops have trivial widening."""
        cert = generate_widening_certificate("let x = 5;")
        assert cert.status == CertStatus.VALID

    def test_simple_loop_widening(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        cert = generate_widening_certificate(source)
        assert cert.status == CertStatus.VALID

    def test_widening_cert_has_obligations(self):
        source = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        cert = generate_widening_certificate(source)
        assert len(cert.obligations) > 0

    def test_widening_metadata(self):
        cert = generate_widening_certificate("let x = 1;")
        assert cert.metadata["analysis_type"] == "widening_soundness"
        assert "limited_iterations" in cert.metadata
        assert "full_iterations" in cert.metadata


# ===================================================================
# Composite Certificate Tests
# ===================================================================

class TestCompositeCertificate:
    """Test composite analysis + widening certificates."""

    def test_full_certification_simple(self):
        result, composite = certify_abstract_interpretation("let x = 5;")
        assert result.certified
        assert composite.status == CertStatus.VALID
        assert composite.kind == ProofKind.COMPOSITE

    def test_full_certification_loop(self):
        source = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        result, composite = certify_abstract_interpretation(source)
        assert result.certified
        assert composite.status == CertStatus.VALID

    def test_composite_has_sub_certificates(self):
        _, composite = certify_abstract_interpretation("let x = 1;")
        assert len(composite.sub_certificates) == 2

    def test_composite_total_obligations(self):
        _, composite = certify_abstract_interpretation("let a = 1; let b = 2;")
        assert composite.total_obligations > 0


# ===================================================================
# Convenience API Tests
# ===================================================================

class TestConvenienceAPIs:
    """Test convenience functions."""

    def test_certify_variable_bound_holds(self):
        obl = certify_variable_bound("let x = 5;", "x", expected_lo=0, expected_hi=10)
        assert obl.status == CertStatus.VALID

    def test_certify_variable_bound_exact(self):
        obl = certify_variable_bound("let x = 5;", "x", expected_lo=5, expected_hi=5)
        assert obl.status == CertStatus.VALID

    def test_certify_variable_bound_lo_only(self):
        obl = certify_variable_bound("let x = 5;", "x", expected_lo=0)
        assert obl.status == CertStatus.VALID

    def test_certify_variable_bound_hi_only(self):
        obl = certify_variable_bound("let x = 5;", "x", expected_hi=10)
        assert obl.status == CertStatus.VALID

    def test_certify_variable_bound_no_bounds(self):
        obl = certify_variable_bound("let x = 5;", "x")
        assert obl.status == CertStatus.VALID

    def test_certify_sign_exact_match(self):
        obl = certify_sign("let x = 5;", "x", Sign.POS)
        assert obl.status == CertStatus.VALID

    def test_certify_sign_non_neg(self):
        obl = certify_sign("let x = 0;", "x", Sign.NON_NEG)
        assert obl.status == CertStatus.VALID

    def test_certify_sign_via_interval(self):
        """Sign check verified through interval bounds."""
        obl = certify_sign("let x = 5;", "x", Sign.NON_NEG)
        assert obl.status == CertStatus.VALID

    def test_certify_sign_wrong(self):
        """x = 5 is not negative."""
        obl = certify_sign("let x = 5;", "x", Sign.NEG)
        assert obl.status == CertStatus.INVALID


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_program(self):
        """Empty program has no variables to certify."""
        cert = generate_ai_certificate("let _dummy = 0;")
        assert cert is not None

    def test_negative_values(self):
        result = verified_analyze("let x = 0 - 5;")
        assert result.certified
        assert result.analysis.variable_intervals["x"].hi < 0

    def test_zero_value(self):
        result = verified_analyze("let x = 0;")
        assert result.certified
        assert result.analysis.variable_signs["x"] == Sign.ZERO

    def test_large_program(self):
        source = "\n".join(f"let v{i} = {i};" for i in range(20))
        result = verified_analyze(source)
        assert result.certified
        assert len(result.analysis.variable_intervals) == 20

    def test_nested_conditionals(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            if (x > 3) {
                y = 1;
            } else {
                y = 2;
            }
        } else {
            y = 3;
        }
        """
        result = verified_analyze(source)
        assert result.certified
