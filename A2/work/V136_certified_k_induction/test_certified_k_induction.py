"""Tests for V136: Certified k-Induction."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

import pytest
from certified_k_induction import (
    certify_k_induction, certify_strengthened_k_induction,
    certify_and_check, certify_loop, certify_loop_with_invariants,
    check_kind_certificate, to_v044_certificate,
    compare_certified_vs_uncertified, kind_certificate_summary,
    KIndCertificate, KIndCertKind, _and, _not, _implies,
)
from proof_certificates import ProofObligation, ProofCertificate, ProofKind, CertStatus
from pdr import TransitionSystem
from smt_solver import Var, App, Op, IntConst, BoolConst, INT, BOOL


# ---------------------------------------------------------------------------
# Helper: build formula-based transition systems
# ---------------------------------------------------------------------------

def make_counter_ts():
    """x starts at 0, increments by 1. Property: x >= 0. (1-inductive)"""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL))
    ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
    return ts


def make_decrement_ts():
    """x starts at 5, decrements by 1. Property: x <= 5. (1-inductive)"""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(5)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, App(Op.SUB, [x, IntConst(1)], INT)], BOOL))
    ts.set_property(App(Op.LE, [x, IntConst(5)], BOOL))
    return ts


def make_unsafe_ts():
    """x starts at 0, increments. Property: x < 3. Fails at step 3."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL))
    ts.set_property(App(Op.LT, [x, IntConst(3)], BOOL))
    return ts


def make_sum_ts():
    """x=0, y=10. x++, y--. Property: x + y == 10. (1-inductive)"""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    y = ts.add_int_var('y')
    xp = ts.prime('x')
    yp = ts.prime('y')
    init = _and(App(Op.EQ, [x, IntConst(0)], BOOL),
                App(Op.EQ, [y, IntConst(10)], BOOL))
    trans = _and(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL),
                 App(Op.EQ, [yp, App(Op.SUB, [y, IntConst(1)], INT)], BOOL))
    prop = App(Op.EQ, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL)
    ts.set_init(init)
    ts.set_trans(trans)
    ts.set_property(prop)
    return ts


def make_two_var_ts():
    """x=0, y=0, both increment. Property: x >= 0."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    y = ts.add_int_var('y')
    xp = ts.prime('x')
    yp = ts.prime('y')
    init = _and(App(Op.EQ, [x, IntConst(0)], BOOL),
                App(Op.EQ, [y, IntConst(0)], BOOL))
    trans = _and(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL),
                 App(Op.EQ, [yp, App(Op.ADD, [y, IntConst(1)], INT)], BOOL))
    ts.set_init(init)
    ts.set_trans(trans)
    ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
    return ts


def make_identity_ts():
    """x stays at 0 forever. Property: x == 0."""
    ts = TransitionSystem()
    x = ts.add_int_var('x')
    xp = ts.prime('x')
    ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, x], BOOL))
    ts.set_property(App(Op.EQ, [x, IntConst(0)], BOOL))
    return ts


# ===========================================================================
# Section 1: Basic certificate generation (safe systems)
# ===========================================================================

class TestBasicSafeCertification:
    def test_counter_safe(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        assert cert.result == "safe"
        assert cert.k >= 0
        assert cert.kind == KIndCertKind.BASIC
        assert len(cert.obligations) > 0

    def test_decrement_safe(self):
        ts = make_decrement_ts()
        cert = certify_k_induction(ts)
        assert cert.result == "safe"
        assert len(cert.obligations) >= 2

    def test_sum_safe(self):
        ts = make_sum_ts()
        cert = certify_k_induction(ts)
        assert cert.result == "safe"

    def test_specific_k(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts, k=0)
        assert cert.result == "safe"
        assert cert.k == 0

    def test_obligations_have_smt(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        for o in cert.obligations:
            assert o.formula_smt != ""
            assert "(set-logic LIA)" in o.formula_smt
            assert "(declare-const" in o.formula_smt
            assert "(check-sat)" in o.formula_smt


# ===========================================================================
# Section 2: Unsafe system certificates
# ===========================================================================

class TestUnsafeCertification:
    def test_unsafe_detected(self):
        ts = make_unsafe_ts()
        cert = certify_k_induction(ts)
        assert cert.result == "unsafe"
        assert cert.counterexample is not None

    def test_unsafe_no_obligations(self):
        ts = make_unsafe_ts()
        cert = certify_k_induction(ts)
        assert len(cert.obligations) == 0

    def test_unsafe_has_counterexample(self):
        ts = make_unsafe_ts()
        cert = certify_k_induction(ts)
        assert isinstance(cert.counterexample, list)
        assert len(cert.counterexample) > 0

    def test_immediate_violation(self):
        ts = TransitionSystem()
        x = ts.add_int_var('x')
        xp = ts.prime('x')
        ts.set_init(App(Op.EQ, [x, IntConst(-1)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
        cert = certify_k_induction(ts)
        assert cert.result == "unsafe"


# ===========================================================================
# Section 3: Independent checking
# ===========================================================================

class TestIndependentChecking:
    def test_check_counter(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        checked = check_kind_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_decrement(self):
        ts = make_decrement_ts()
        cert = certify_k_induction(ts)
        checked = check_kind_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_sum(self):
        ts = make_sum_ts()
        cert = certify_k_induction(ts)
        checked = check_kind_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_all_obligations_valid(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        check_kind_certificate(cert)
        for o in cert.obligations:
            assert o.status == CertStatus.VALID

    def test_check_updates_status(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        assert cert.status == CertStatus.UNCHECKED
        check_kind_certificate(cert)
        assert cert.status == CertStatus.VALID


# ===========================================================================
# Section 4: Certify and check (combined API)
# ===========================================================================

class TestCertifyAndCheck:
    def test_combined_safe(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID

    def test_combined_unsafe(self):
        ts = make_unsafe_ts()
        cert = certify_and_check(ts)
        assert cert.result == "unsafe"

    def test_combined_sum(self):
        ts = make_sum_ts()
        cert = certify_and_check(ts)
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID

    def test_combined_identity(self):
        ts = make_identity_ts()
        cert = certify_and_check(ts)
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID


# ===========================================================================
# Section 5: Strengthened k-induction certificates
# ===========================================================================

class TestStrengthenedCertification:
    def test_with_invariant(self):
        ts = make_two_var_ts()
        y = ts.var('y')
        inv = App(Op.GE, [y, IntConst(0)], BOOL)
        cert = certify_strengthened_k_induction(ts, [inv])
        assert cert.result == "safe"
        assert cert.kind == KIndCertKind.STRENGTHENED

    def test_strengthened_has_inv_obligations(self):
        ts = make_two_var_ts()
        y = ts.var('y')
        inv = App(Op.GE, [y, IntConst(0)], BOOL)
        cert = certify_strengthened_k_induction(ts, [inv])
        inv_obs = [o for o in cert.obligations if "inv_" in o.name]
        assert len(inv_obs) >= 2  # init + consecution

    def test_strengthened_check(self):
        ts = make_two_var_ts()
        y = ts.var('y')
        inv = App(Op.GE, [y, IntConst(0)], BOOL)
        cert = certify_and_check(ts, invariants=[inv])
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID

    def test_invariants_recorded(self):
        ts = make_two_var_ts()
        y = ts.var('y')
        inv = App(Op.GE, [y, IntConst(0)], BOOL)
        cert = certify_strengthened_k_induction(ts, [inv])
        assert len(cert.invariants_used) == 1

    def test_multiple_invariants(self):
        ts = make_two_var_ts()
        x = ts.var('x')
        y = ts.var('y')
        inv1 = App(Op.GE, [y, IntConst(0)], BOOL)
        inv2 = App(Op.GE, [x, IntConst(0)], BOOL)
        cert = certify_and_check(ts, invariants=[inv1, inv2])
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID
        assert len(cert.invariants_used) == 2


# ===========================================================================
# Section 6: Obligation structure
# ===========================================================================

class TestObligationStructure:
    def test_base_obligations_named(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        base_obs = [o for o in cert.obligations if o.name.startswith("base_")]
        assert len(base_obs) >= 1

    def test_inductive_obligation_named(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        ind_obs = [o for o in cert.obligations if o.name.startswith("inductive_")]
        assert len(ind_obs) == 1

    def test_obligation_descriptions(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts)
        for o in cert.obligations:
            assert o.description != ""
            assert o.formula_str != ""

    def test_k0_obligations(self):
        ts = make_counter_ts()
        cert = certify_k_induction(ts, k=0)
        if cert.result == "safe":
            base_obs = [o for o in cert.obligations if o.name.startswith("base_")]
            ind_obs = [o for o in cert.obligations if o.name.startswith("inductive_")]
            assert len(base_obs) == 1  # base_0
            assert len(ind_obs) == 1   # inductive_k0


# ===========================================================================
# Section 7: JSON serialization
# ===========================================================================

class TestSerialization:
    def test_to_json_roundtrip(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        json_str = cert.to_json()
        cert2 = KIndCertificate.from_json(json_str)
        assert cert2.result == cert.result
        assert cert2.k == cert.k
        assert cert2.kind == cert.kind
        assert len(cert2.obligations) == len(cert.obligations)

    def test_to_dict(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        d = cert.to_dict()
        assert d["result"] == "safe"
        assert isinstance(d["obligations"], list)

    def test_json_valid(self):
        ts = make_sum_ts()
        cert = certify_and_check(ts)
        json_str = cert.to_json()
        parsed = json.loads(json_str)
        assert parsed["result"] == "safe"

    def test_serialization_preserves_status(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        json_str = cert.to_json()
        cert2 = KIndCertificate.from_json(json_str)
        assert cert2.status == cert.status


# ===========================================================================
# Section 8: V044 bridge
# ===========================================================================

class TestV044Bridge:
    def test_to_v044(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        v044 = to_v044_certificate(cert)
        assert isinstance(v044, ProofCertificate)
        assert v044.kind == ProofKind.COMPOSITE

    def test_v044_has_obligations(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        v044 = to_v044_certificate(cert)
        assert len(v044.obligations) == len(cert.obligations)

    def test_v044_preserves_status(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        v044 = to_v044_certificate(cert)
        assert v044.status == cert.status

    def test_v044_metadata(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        v044 = to_v044_certificate(cert)
        assert "k" in v044.metadata
        assert "cert_kind" in v044.metadata


# ===========================================================================
# Section 9: Comparison API
# ===========================================================================

class TestComparison:
    def test_compare_safe(self):
        ts = make_counter_ts()
        result = compare_certified_vs_uncertified(ts)
        assert result["plain_result"] == "SAFE"
        assert result["certified_result"] == "safe"
        assert result["certified_status"] == "valid"

    def test_compare_has_timing(self):
        ts = make_counter_ts()
        result = compare_certified_vs_uncertified(ts)
        assert "plain_time" in result
        assert "certified_time" in result
        assert "overhead_ratio" in result

    def test_compare_unsafe(self):
        ts = make_unsafe_ts()
        result = compare_certified_vs_uncertified(ts)
        assert result["plain_result"] == "UNSAFE"
        assert result["certified_result"] == "unsafe"


# ===========================================================================
# Section 10: Summary
# ===========================================================================

class TestSummary:
    def test_summary_safe(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        s = kind_certificate_summary(cert)
        assert "safe" in s.lower()
        assert "k=" in s

    def test_summary_unsafe(self):
        ts = make_unsafe_ts()
        cert = certify_k_induction(ts)
        s = kind_certificate_summary(cert)
        assert "unsafe" in s.lower()

    def test_cert_summary_method(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        s = cert.summary()
        assert "Valid:" in s


# ===========================================================================
# Section 11: Source-level API
# ===========================================================================

class TestSourceLevel:
    def test_certify_loop_safe(self):
        source = """
let x = 0;
while (x < 10) {
    x = x + 1;
}
"""
        cert = certify_loop(source, "x >= 0")
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID

    def test_certify_loop_with_invariant(self):
        source = """
let x = 0;
let y = 0;
while (x < 10) {
    x = x + 1;
    y = y + 1;
}
"""
        cert = certify_loop_with_invariants(source, "x >= 0", ["y >= 0"])
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID


# ===========================================================================
# Section 12: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_identity_ts(self):
        ts = make_identity_ts()
        cert = certify_and_check(ts)
        assert cert.result == "safe"
        assert cert.status == CertStatus.VALID

    def test_metadata_populated(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        assert "method" in cert.metadata
        assert "duration" in cert.metadata
        assert cert.metadata["duration"] >= 0

    def test_recheck_after_reset(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        for o in cert.obligations:
            o.status = CertStatus.UNCHECKED
        check_kind_certificate(cert)
        assert cert.status == CertStatus.VALID


# ===========================================================================
# Section 13: Obligation verification details
# ===========================================================================

class TestObligationVerification:
    def test_each_obligation_checked(self):
        ts = make_counter_ts()
        cert = certify_and_check(ts)
        for o in cert.obligations:
            assert o.status != CertStatus.UNCHECKED

    def test_smt_scripts_parseable(self):
        ts = make_sum_ts()
        cert = certify_k_induction(ts)
        for o in cert.obligations:
            assert "(set-logic LIA)" in o.formula_smt
            assert "(check-sat)" in o.formula_smt

    def test_decrement_obligations_valid(self):
        ts = make_decrement_ts()
        cert = certify_and_check(ts)
        for o in cert.obligations:
            assert o.status == CertStatus.VALID


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
