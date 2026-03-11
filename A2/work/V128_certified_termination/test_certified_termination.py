"""Tests for V128: Certified Termination."""

import pytest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V025_termination_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))

from certified_termination import (
    # Core types
    TerminationCertificate, ProgramTerminationCertificate,
    # Certificate generation
    certify_loop_termination, certify_program_termination, certify_and_check,
    # Certificate checking
    check_termination_certificate,
    # I/O
    save_termination_certificate, load_termination_certificate,
    # V044 bridge
    to_v044_certificate,
    # Comparison
    compare_with_uncertified, termination_certificate_summary,
    # Internal
    _encode_bounded_smtlib, _encode_decreasing_smtlib,
    _coeffs_to_smtlib, _coeffs_to_smtlib_primed,
    _generate_ranking_obligations,
)
from termination import TermResult, extract_loop_info, RankingFunction
from proof_certificates import ProofCertificate, ProofObligation, ProofKind, CertStatus


# ---- C10 source snippets ----

COUNTER = """
let i = 0;
while (i < 10) {
    i = i + 1;
}
"""

COUNTDOWN = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""

TWO_VAR = """
let i = 0;
let n = 5;
while (i < n) {
    i = i + 1;
}
"""

MULTI_LOOP = """
let x = 0;
while (x < 5) {
    x = x + 1;
}
let y = 10;
while (y > 0) {
    y = y - 1;
}
"""


# ===========================================================================
# 1. Basic Loop Certification
# ===========================================================================

class TestBasicCertification:
    def test_counter_terminates(self):
        cert = certify_loop_termination(COUNTER)
        assert cert.result == TermResult.TERMINATES
        assert cert.ranking_expression is not None
        assert cert.status == CertStatus.VALID

    def test_countdown_terminates(self):
        cert = certify_loop_termination(COUNTDOWN)
        assert cert.result == TermResult.TERMINATES
        assert cert.status == CertStatus.VALID

    def test_two_var_terminates(self):
        cert = certify_loop_termination(TWO_VAR)
        assert cert.result == TermResult.TERMINATES

    def test_has_obligations(self):
        cert = certify_loop_termination(COUNTER)
        assert cert.total_obligations >= 2  # bounded + decreasing

    def test_obligations_valid(self):
        cert = certify_loop_termination(COUNTER)
        for obl in cert.obligations:
            assert obl.status == CertStatus.VALID


# ===========================================================================
# 2. Certificate Checking (Independent Verification)
# ===========================================================================

class TestCertificateChecking:
    def test_check_counter(self):
        cert = certify_and_check(COUNTER)
        assert cert.status == CertStatus.VALID

    def test_check_countdown(self):
        cert = certify_and_check(COUNTDOWN)
        assert cert.status == CertStatus.VALID

    def test_check_preserves_result(self):
        cert = certify_loop_termination(COUNTER)
        checked = check_termination_certificate(cert)
        assert checked.result == cert.result
        assert checked.ranking_expression == cert.ranking_expression

    def test_check_preserves_metadata(self):
        cert = certify_loop_termination(COUNTER)
        checked = check_termination_certificate(cert)
        assert checked.loop_index == cert.loop_index
        assert checked.source == cert.source


# ===========================================================================
# 3. Program-Level Certification
# ===========================================================================

class TestProgramCertification:
    def test_single_loop_program(self):
        cert = certify_program_termination(COUNTER)
        assert cert.total_loops >= 1
        assert cert.proved_loops >= 1

    def test_multi_loop_program(self):
        cert = certify_program_termination(MULTI_LOOP)
        assert cert.total_loops >= 2
        assert cert.proved_loops >= 2
        assert cert.all_terminate

    def test_program_status(self):
        cert = certify_program_termination(COUNTER)
        assert cert.status == CertStatus.VALID

    def test_program_summary(self):
        cert = certify_program_termination(MULTI_LOOP)
        s = cert.summary()
        assert "Program Termination Certificate" in s
        assert "Loop" in s


# ===========================================================================
# 4. Serialization
# ===========================================================================

class TestSerialization:
    def test_loop_to_dict_from_dict(self):
        cert = certify_loop_termination(COUNTER)
        d = cert.to_dict()
        cert2 = TerminationCertificate.from_dict(d)
        assert cert2.result == cert.result
        assert cert2.status == cert.status
        assert cert2.ranking_expression == cert.ranking_expression

    def test_loop_json_roundtrip(self):
        cert = certify_loop_termination(COUNTER)
        j = cert.to_json()
        cert2 = TerminationCertificate.from_json(j)
        assert cert2.result == cert.result
        assert len(cert2.obligations) == len(cert.obligations)

    def test_program_to_dict_from_dict(self):
        cert = certify_program_termination(COUNTER)
        d = cert.to_dict()
        cert2 = ProgramTerminationCertificate.from_dict(d)
        assert cert2.all_terminate == cert.all_terminate
        assert cert2.total_loops == cert.total_loops

    def test_save_load_file(self):
        cert = certify_loop_termination(COUNTER)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_termination_certificate(cert, path)
            loaded = load_termination_certificate(path)
            assert loaded.result == cert.result
            assert loaded.status == cert.status
        finally:
            os.unlink(path)


# ===========================================================================
# 5. V044 Bridge
# ===========================================================================

class TestV044Bridge:
    def test_to_v044(self):
        cert = certify_loop_termination(COUNTER)
        v044 = to_v044_certificate(cert)
        assert isinstance(v044, ProofCertificate)
        assert v044.kind == ProofKind.VCGEN
        assert "terminates" in v044.claim.lower()

    def test_v044_has_obligations(self):
        cert = certify_loop_termination(COUNTER)
        v044 = to_v044_certificate(cert)
        assert len(v044.obligations) == len(cert.obligations)

    def test_v044_metadata(self):
        cert = certify_loop_termination(COUNTER)
        v044 = to_v044_certificate(cert)
        assert v044.metadata.get("termination") is True


# ===========================================================================
# 6. SMT-LIB2 Encoding
# ===========================================================================

class TestSMTLIB2Encoding:
    def test_bounded_encoding(self):
        smtlib = _encode_bounded_smtlib({"x": 1, "_const": 0}, ["x"])
        assert "(set-logic LIA)" in smtlib
        assert "(declare-const x Int)" in smtlib
        assert "(< x 0)" in smtlib

    def test_decreasing_encoding(self):
        smtlib = _encode_decreasing_smtlib({"x": 1, "_const": 0}, ["x"])
        assert "(declare-const x Int)" in smtlib
        assert "(declare-const x_next Int)" in smtlib

    def test_coeffs_to_smtlib_single(self):
        expr = _coeffs_to_smtlib({"x": 1, "_const": 0}, ["x"])
        assert "x" in expr

    def test_coeffs_to_smtlib_difference(self):
        expr = _coeffs_to_smtlib({"n": 1, "i": -1, "_const": 0}, ["i", "n"])
        assert "n" in expr

    def test_coeffs_to_smtlib_primed(self):
        expr = _coeffs_to_smtlib_primed({"x": 1, "_const": 0}, ["x"], "_next")
        assert "x_next" in expr

    def test_coeffs_zero(self):
        expr = _coeffs_to_smtlib({"_const": 0}, [])
        assert expr == "0"


# ===========================================================================
# 7. Ranking Function Obligations
# ===========================================================================

class TestRankingObligations:
    def test_generate_for_counter(self):
        loop_info = extract_loop_info(COUNTER)
        rf = RankingFunction(
            expression="10 - i",
            coefficients={"i": -1, "_const": 10},
        )
        obls = _generate_ranking_obligations(rf, loop_info, 0)
        assert len(obls) == 2
        assert any("bounded" in o.name.lower() or "bounded" in o.description.lower() for o in obls)
        assert any("decreasing" in o.name.lower() or "decreasing" in o.description.lower() for o in obls)

    def test_obligations_contain_smtlib(self):
        loop_info = extract_loop_info(COUNTER)
        rf = RankingFunction(
            expression="10 - i",
            coefficients={"i": -1, "_const": 10},
        )
        obls = _generate_ranking_obligations(rf, loop_info, 0)
        for obl in obls:
            assert obl.formula_smt
            assert "(set-logic LIA)" in obl.formula_smt


# ===========================================================================
# 8. Comparison API
# ===========================================================================

class TestComparison:
    def test_compare(self):
        comp = compare_with_uncertified(COUNTER)
        assert "uncertified" in comp
        assert "certified" in comp
        assert comp["uncertified"]["loops_proved"] >= 1
        assert comp["certified"]["loops_proved"] >= 1

    def test_compare_timing(self):
        comp = compare_with_uncertified(COUNTER)
        assert comp["uncertified"]["time"] >= 0
        assert comp["certified"]["time"] >= 0


# ===========================================================================
# 9. Summary
# ===========================================================================

class TestSummary:
    def test_loop_summary(self):
        cert = certify_loop_termination(COUNTER)
        s = cert.summary()
        assert "Termination Certificate" in s
        assert "Ranking function" in s

    def test_full_summary(self):
        s = termination_certificate_summary(COUNTER)
        assert "Termination" in s


# ===========================================================================
# 10. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_trivial_loop(self):
        """Loop that doesn't execute."""
        source = """
            let x = 10;
            while (x < 5) {
                x = x + 1;
            }
        """
        cert = certify_loop_termination(source)
        # Should still find ranking function or report termination
        assert cert.result in (TermResult.TERMINATES, TermResult.UNKNOWN)

    def test_single_step_loop(self):
        source = """
            let x = 0;
            while (x < 1) {
                x = x + 1;
            }
        """
        cert = certify_loop_termination(source)
        assert cert.result == TermResult.TERMINATES

    def test_no_loops(self):
        source = "let x = 5;"
        cert = certify_program_termination(source)
        assert cert.total_loops == 0
        assert cert.all_terminate


# ===========================================================================
# 11. Certificate Fields
# ===========================================================================

class TestCertificateFields:
    def test_metadata(self):
        cert = certify_loop_termination(COUNTER)
        assert "state_vars" in cert.metadata
        assert "ranking_kind" in cert.metadata

    def test_loop_index(self):
        cert = certify_loop_termination(COUNTER, loop_index=0)
        assert cert.loop_index == 0

    def test_valid_obligations_count(self):
        cert = certify_loop_termination(COUNTER)
        assert cert.valid_obligations >= 2
        assert cert.valid_obligations == cert.total_obligations
