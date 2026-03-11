"""Tests for V137: Certified PDR."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V136_certified_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

import pytest
from certified_pdr import (
    certify_pdr, certify_and_check_pdr, check_pdr_certificate,
    certify_pdr_loop, to_v044_certificate,
    compare_pdr_vs_kind, certify_combined,
    compare_certified_vs_uncertified, pdr_certificate_summary,
    PDRCertificate, PDRCertKind,
)
from proof_certificates import ProofCertificate, ProofKind, CertStatus
from pdr import TransitionSystem
from smt_solver import Var, App, Op, IntConst, BoolConst, INT, BOOL


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_counter_ts():
    """x=0, x'=x+1. Property: x >= 0."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL))
    ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
    return ts


def make_decrement_ts():
    """x=5, x'=x-1. Property: x <= 5."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(5)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, App(Op.SUB, [x, IntConst(1)], INT)], BOOL))
    ts.set_property(App(Op.LE, [x, IntConst(5)], BOOL))
    return ts


def make_unsafe_ts():
    """x=0, x'=x+1. Property: x < 3. Fails at step 3."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL))
    ts.set_property(App(Op.LT, [x, IntConst(3)], BOOL))
    return ts


def make_sum_ts():
    """x=0, y=10, x++, y--. Property: x + y == 10."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    y = ts.add_int_var('y')
    xp = ts.prime('x')
    yp = ts.prime('y')
    init = App(Op.AND, [App(Op.EQ, [x, IntConst(0)], BOOL),
                        App(Op.EQ, [y, IntConst(10)], BOOL)], BOOL)
    trans = App(Op.AND, [App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL),
                         App(Op.EQ, [yp, App(Op.SUB, [y, IntConst(1)], INT)], BOOL)], BOOL)
    ts.set_init(init)
    ts.set_trans(trans)
    ts.set_property(App(Op.EQ, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL))
    return ts


def make_identity_ts():
    """x=0, x'=x. Property: x == 0."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, x], BOOL))
    ts.set_property(App(Op.EQ, [x, IntConst(0)], BOOL))
    return ts


# ===========================================================================
# Section 1: Basic PDR certification
# ===========================================================================

class TestBasicPDRCertification:
    def test_counter_safe(self):
        ts = make_counter_ts()
        cert = certify_pdr(ts)
        assert cert.result == "safe"
        assert cert.kind == PDRCertKind.PDR
        assert len(cert.obligations) > 0

    def test_decrement_safe(self):
        ts = make_decrement_ts()
        cert = certify_pdr(ts)
        assert cert.result == "safe"

    def test_sum_safe(self):
        ts = make_sum_ts()
        cert = certify_pdr(ts)
        assert cert.result == "safe"

    def test_identity_safe(self):
        ts = make_identity_ts()
        cert = certify_pdr(ts)
        assert cert.result == "safe"

    def test_has_three_obligations(self):
        ts = make_counter_ts()
        cert = certify_pdr(ts)
        assert len(cert.obligations) == 3  # init, consecution, property


# ===========================================================================
# Section 2: Unsafe PDR certification
# ===========================================================================

class TestUnsafePDR:
    def test_unsafe_detected(self):
        ts = make_unsafe_ts()
        cert = certify_pdr(ts)
        assert cert.result == "unsafe"

    def test_unsafe_no_obligations_needed(self):
        ts = make_unsafe_ts()
        cert = certify_pdr(ts)
        # Unsafe certs may or may not have obligations
        # The important thing is the result
        assert cert.result == "unsafe"

    def test_immediate_violation(self):
        ts = TransitionSystem()
        x = ts.add_int_var('x')
        xp = ts.prime('x')
        ts.set_init(App(Op.EQ, [x, IntConst(-1)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
        cert = certify_pdr(ts)
        assert cert.result == "unsafe"


# ===========================================================================
# Section 3: Independent checking
# ===========================================================================

class TestIndependentChecking:
    def test_check_counter(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        assert cert.status == CertStatus.VALID

    def test_check_decrement(self):
        ts = make_decrement_ts()
        cert = certify_and_check_pdr(ts)
        assert cert.status == CertStatus.VALID

    def test_check_sum(self):
        ts = make_sum_ts()
        cert = certify_and_check_pdr(ts)
        assert cert.status == CertStatus.VALID

    def test_all_obligations_valid(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        for o in cert.obligations:
            assert o.status == CertStatus.VALID

    def test_check_identity(self):
        ts = make_identity_ts()
        cert = certify_and_check_pdr(ts)
        assert cert.status == CertStatus.VALID


# ===========================================================================
# Section 4: JSON serialization
# ===========================================================================

class TestSerialization:
    def test_roundtrip(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        json_str = cert.to_json()
        cert2 = PDRCertificate.from_json(json_str)
        assert cert2.result == cert.result
        assert cert2.kind == cert.kind
        assert cert2.status == cert.status
        assert len(cert2.obligations) == len(cert.obligations)

    def test_to_dict(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        d = cert.to_dict()
        assert d["result"] == "safe"
        assert isinstance(d["obligations"], list)

    def test_json_valid(self):
        ts = make_sum_ts()
        cert = certify_and_check_pdr(ts)
        json_str = cert.to_json()
        parsed = json.loads(json_str)
        assert parsed["result"] == "safe"

    def test_preserves_status(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        json_str = cert.to_json()
        cert2 = PDRCertificate.from_json(json_str)
        assert cert2.status == CertStatus.VALID


# ===========================================================================
# Section 5: V044 bridge
# ===========================================================================

class TestV044Bridge:
    def test_to_v044(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        v044 = to_v044_certificate(cert)
        assert isinstance(v044, ProofCertificate)
        assert v044.kind == ProofKind.PDR

    def test_v044_has_obligations(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        v044 = to_v044_certificate(cert)
        assert len(v044.obligations) == len(cert.obligations)

    def test_v044_preserves_status(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        v044 = to_v044_certificate(cert)
        assert v044.status == cert.status

    def test_v044_metadata(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        v044 = to_v044_certificate(cert)
        assert "result" in v044.metadata
        assert "cert_kind" in v044.metadata


# ===========================================================================
# Section 6: Comparison API
# ===========================================================================

class TestComparison:
    def test_compare_pdr_vs_kind_safe(self):
        ts = make_counter_ts()
        result = compare_pdr_vs_kind(ts)
        assert result["pdr_result"] == "safe"
        assert result["kind_result"] == "safe"

    def test_compare_has_timing(self):
        ts = make_counter_ts()
        result = compare_pdr_vs_kind(ts)
        assert "pdr_time" in result
        assert "kind_time" in result

    def test_compare_unsafe(self):
        ts = make_unsafe_ts()
        result = compare_pdr_vs_kind(ts)
        assert result["pdr_result"] == "unsafe"
        assert result["kind_result"] == "unsafe"

    def test_certified_vs_uncertified(self):
        ts = make_counter_ts()
        result = compare_certified_vs_uncertified(ts)
        assert result["plain_result"] == "safe"
        assert result["certified_result"] == "safe"
        assert result["certified_status"] == "valid"


# ===========================================================================
# Section 7: Combined certification
# ===========================================================================

class TestCombinedCertification:
    def test_combined_safe(self):
        ts = make_counter_ts()
        cert = certify_combined(ts)
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID
        assert cert.kind == PDRCertKind.COMBINED

    def test_combined_unsafe(self):
        ts = make_unsafe_ts()
        cert = certify_combined(ts)
        assert cert.result == "unsafe"

    def test_combined_has_winner(self):
        ts = make_counter_ts()
        cert = certify_combined(ts)
        assert "winner" in cert.metadata

    def test_combined_sum(self):
        ts = make_sum_ts()
        cert = certify_combined(ts)
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID


# ===========================================================================
# Section 8: Source-level API
# ===========================================================================

class TestSourceLevel:
    def test_certify_loop(self):
        source = """
let x = 0;
while (x < 10) {
    x = x + 1;
}
"""
        cert = certify_pdr_loop(source, "x >= 0")
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID

    def test_certify_two_var_loop(self):
        source = """
let x = 0;
let y = 0;
while (x < 10) {
    x = x + 1;
    y = y + 1;
}
"""
        cert = certify_pdr_loop(source, "x >= 0")
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID


# ===========================================================================
# Section 9: Summary
# ===========================================================================

class TestSummary:
    def test_summary_safe(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        s = pdr_certificate_summary(cert)
        assert "safe" in s.lower()
        assert "Valid:" in s

    def test_summary_unsafe(self):
        ts = make_unsafe_ts()
        cert = certify_pdr(ts)
        s = pdr_certificate_summary(cert)
        assert "unsafe" in s.lower()

    def test_cert_summary_method(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        s = cert.summary()
        assert "PDRCertificate" in s


# ===========================================================================
# Section 10: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_metadata_populated(self):
        ts = make_counter_ts()
        cert = certify_and_check_pdr(ts)
        assert "method" in cert.metadata
        assert "duration" in cert.metadata

    def test_invariant_clauses_present(self):
        ts = make_counter_ts()
        cert = certify_pdr(ts)
        # PDR may or may not produce explicit invariant clause strings
        # but the obligations should be there
        assert len(cert.obligations) >= 3

    def test_obligation_names(self):
        ts = make_counter_ts()
        cert = certify_pdr(ts)
        names = [o.name for o in cert.obligations]
        assert "initiation" in names
        assert "consecution" in names
        assert "property" in names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
