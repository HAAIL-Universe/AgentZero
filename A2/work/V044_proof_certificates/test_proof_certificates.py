"""
Tests for V044: Proof Certificates
"""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from proof_certificates import (
    ProofCertificate, ProofObligation, ProofKind, CertStatus,
    generate_vcgen_certificate, generate_pdr_certificate,
    check_certificate, combine_certificates,
    save_certificate, load_certificate,
    certify_program, certify_transition_system,
    sexpr_to_str, sexpr_to_smtlib, smt_term_to_str, smt_term_to_smtlib,
)


# ============================================================
# Data Structure Tests
# ============================================================

class TestProofObligation:
    def test_create_obligation(self):
        ob = ProofObligation(
            name="test_ob",
            description="A test",
            formula_str="(x > 0)",
            formula_smt="(assert (not (> x 0)))",
        )
        assert ob.name == "test_ob"
        assert ob.status == CertStatus.UNCHECKED

    def test_obligation_roundtrip(self):
        ob = ProofObligation(
            name="test",
            description="desc",
            formula_str="(x > 0)",
            formula_smt="(assert (> x 0))",
            status=CertStatus.VALID,
            counterexample={"x": -1},
        )
        d = ob.to_dict()
        ob2 = ProofObligation.from_dict(d)
        assert ob2.name == ob.name
        assert ob2.status == CertStatus.VALID
        assert ob2.counterexample == {"x": -1}

    def test_obligation_no_counterexample(self):
        ob = ProofObligation(name="t", description="d", formula_str="f", formula_smt="s")
        d = ob.to_dict()
        assert "counterexample" not in d


class TestProofCertificate:
    def test_create_certificate(self):
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test claim")
        assert cert.kind == ProofKind.VCGEN
        assert cert.status == CertStatus.UNCHECKED
        assert cert.total_obligations == 0

    def test_obligation_counting(self):
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test")
        cert.obligations = [
            ProofObligation("a", "d", "f", "s", CertStatus.VALID),
            ProofObligation("b", "d", "f", "s", CertStatus.VALID),
            ProofObligation("c", "d", "f", "s", CertStatus.INVALID),
        ]
        assert cert.total_obligations == 3
        assert cert.valid_obligations == 2
        assert cert.invalid_obligations == 1

    def test_composite_counting(self):
        sub1 = ProofCertificate(kind=ProofKind.VCGEN, claim="sub1")
        sub1.obligations = [
            ProofObligation("a", "d", "f", "s", CertStatus.VALID),
        ]
        sub2 = ProofCertificate(kind=ProofKind.PDR, claim="sub2")
        sub2.obligations = [
            ProofObligation("b", "d", "f", "s", CertStatus.VALID),
            ProofObligation("c", "d", "f", "s", CertStatus.INVALID),
        ]
        composite = ProofCertificate(
            kind=ProofKind.COMPOSITE, claim="all",
            sub_certificates=[sub1, sub2],
        )
        assert composite.total_obligations == 3
        assert composite.valid_obligations == 2
        assert composite.invalid_obligations == 1

    def test_json_roundtrip(self):
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test")
        cert.obligations = [
            ProofObligation("a", "desc", "formula", "smt", CertStatus.VALID),
        ]
        cert.status = CertStatus.VALID
        cert.metadata = {"foo": "bar"}

        json_str = cert.to_json()
        cert2 = ProofCertificate.from_json(json_str)
        assert cert2.kind == ProofKind.VCGEN
        assert cert2.claim == "test"
        assert cert2.status == CertStatus.VALID
        assert len(cert2.obligations) == 1
        assert cert2.obligations[0].name == "a"
        assert cert2.metadata["foo"] == "bar"

    def test_composite_json_roundtrip(self):
        sub = ProofCertificate(kind=ProofKind.PDR, claim="sub")
        sub.obligations = [ProofObligation("x", "d", "f", "s", CertStatus.VALID)]
        comp = ProofCertificate(
            kind=ProofKind.COMPOSITE, claim="comp",
            sub_certificates=[sub],
        )
        json_str = comp.to_json()
        comp2 = ProofCertificate.from_json(json_str)
        assert comp2.kind == ProofKind.COMPOSITE
        assert len(comp2.sub_certificates) == 1
        assert comp2.sub_certificates[0].claim == "sub"

    def test_summary(self):
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="my claim")
        cert.obligations = [
            ProofObligation("a", "d", "f", "s", CertStatus.VALID),
        ]
        cert.status = CertStatus.VALID
        s = cert.summary()
        assert "vcgen" in s
        assert "my claim" in s
        assert "1/1 valid" in s


# ============================================================
# SExpr Serialization Tests
# ============================================================

class TestSExprSerialization:
    def test_sexpr_to_str_var(self):
        from vc_gen import SVar
        assert sexpr_to_str(SVar("x")) == "x"

    def test_sexpr_to_str_int(self):
        from vc_gen import SInt
        assert sexpr_to_str(SInt(42)) == "42"

    def test_sexpr_to_str_bool(self):
        from vc_gen import SBool
        assert sexpr_to_str(SBool(True)) == "true"
        assert sexpr_to_str(SBool(False)) == "false"

    def test_sexpr_to_str_binop(self):
        from vc_gen import SBinOp, SVar, SInt
        expr = SBinOp("+", SVar("x"), SInt(1))
        assert sexpr_to_str(expr) == "(x + 1)"

    def test_sexpr_to_str_implies(self):
        from vc_gen import SImplies, SVar, SBool
        expr = SImplies(SVar("p"), SBool(True))
        assert sexpr_to_str(expr) == "(p => true)"

    def test_sexpr_to_str_and(self):
        from vc_gen import SAnd, SVar
        expr = SAnd((SVar("a"), SVar("b")))
        assert sexpr_to_str(expr) == "(and a b)"

    def test_sexpr_to_str_not(self):
        from vc_gen import SNot, SVar
        expr = SNot(SVar("x"))
        assert sexpr_to_str(expr) == "(not x)"

    def test_sexpr_to_smtlib_complete(self):
        from vc_gen import SBinOp, SVar, SInt
        expr = SBinOp(">=", SVar("x"), SInt(0))
        smtlib = sexpr_to_smtlib(expr)
        assert "(set-logic LIA)" in smtlib
        assert "(declare-const x Int)" in smtlib
        assert "(check-sat)" in smtlib
        assert "(assert (not" in smtlib


# ============================================================
# SMT Term Serialization Tests
# ============================================================

class TestSMTTermSerialization:
    def test_smt_term_to_str_var(self):
        from smt_solver import Var, INT
        assert smt_term_to_str(Var("x", INT)) == "x"

    def test_smt_term_to_str_intconst(self):
        from smt_solver import IntConst
        assert smt_term_to_str(IntConst(5)) == "5"

    def test_smt_term_to_str_app(self):
        from smt_solver import Var, IntConst, App, Op, INT, BOOL
        term = App(Op.GE, [Var("x", INT), IntConst(0)], BOOL)
        s = smt_term_to_str(term)
        assert ">=" in s
        assert "x" in s
        assert "0" in s

    def test_smt_term_to_smtlib(self):
        from smt_solver import Var, IntConst, App, Op, INT, BOOL
        term = App(Op.GE, [Var("x", INT), IntConst(0)], BOOL)
        s = smt_term_to_smtlib(term)
        assert ">=" in s


# ============================================================
# VCGen Certificate Generation Tests
# ============================================================

class TestVCGenCertificate:
    def test_simple_valid_program(self):
        source = """
fn abs(x) {
    requires(x >= 0);
    ensures(result >= 0);
    return x;
}
"""
        cert = generate_vcgen_certificate(source, "abs")
        assert cert.kind == ProofKind.VCGEN
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations > 0
        assert cert.valid_obligations == cert.total_obligations

    def test_invalid_program(self):
        source = """
fn bad(x) {
    requires(x >= 0);
    ensures(result > 0);
    return x;
}
"""
        cert = generate_vcgen_certificate(source, "bad")
        # x=0 satisfies precond but result=0 doesn't satisfy result > 0
        assert cert.status == CertStatus.INVALID
        assert cert.invalid_obligations > 0

    def test_program_with_loop(self):
        source = """
fn sum_to(n) {
    requires(n >= 0);
    ensures(result >= 0);
    let s = 0;
    let i = 0;
    while (i < n) {
        invariant(s >= 0);
        invariant(i >= 0);
        s = s + i;
        i = i + 1;
    }
    return s;
}
"""
        cert = generate_vcgen_certificate(source, "sum_to")
        assert cert.kind == ProofKind.VCGEN
        # Loop generates preservation + postcondition VCs
        assert cert.total_obligations >= 2

    def test_certificate_has_source(self):
        source = "fn f(x) { requires(x > 0); ensures(result > 0); return x; }"
        cert = generate_vcgen_certificate(source, "f")
        assert cert.source == source

    def test_certificate_metadata(self):
        source = "fn f(x) { requires(x > 0); ensures(result > 0); return x; }"
        cert = generate_vcgen_certificate(source, "f")
        assert cert.metadata["fn_name"] == "f"
        assert "total_vcs" in cert.metadata

    def test_multiple_functions(self):
        source = """
fn inc(x) {
    requires(x >= 0);
    ensures(result > 0);
    return x + 1;
}
fn dec(x) {
    requires(x > 0);
    ensures(result >= 0);
    return x - 1;
}
"""
        cert = generate_vcgen_certificate(source)
        assert cert.kind == ProofKind.VCGEN
        assert cert.total_obligations >= 2


# ============================================================
# PDR Certificate Generation Tests
# ============================================================

class TestPDRCertificate:
    def _make_counter_ts(self, bound=5):
        """Counter that increments from 0. Property: x <= bound."""
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        x_prime = ts.prime("x")

        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [x_prime, App(Op.ADD, [x, IntConst(1)], INT)], BOOL))
        ts.set_property(App(Op.LE, [x, IntConst(bound)], BOOL))
        return ts

    def _make_safe_ts(self):
        """x starts at 0, x' = x (stays at 0). Property: x >= 0."""
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        x_prime = ts.prime("x")

        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [x_prime, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
        return ts

    def test_safe_system_certificate(self):
        ts = self._make_safe_ts()
        cert = generate_pdr_certificate(ts)
        assert cert.kind == ProofKind.PDR
        assert cert.status == CertStatus.VALID
        assert cert.metadata["result"] == "safe"
        # Should have 3 obligations: initiation, consecution, property
        assert len(cert.obligations) == 3
        assert cert.obligations[0].name == "initiation"
        assert cert.obligations[1].name == "consecution"
        assert cert.obligations[2].name == "property"

    def test_safe_system_has_invariant(self):
        ts = self._make_safe_ts()
        cert = generate_pdr_certificate(ts)
        assert "invariant" in cert.metadata
        assert len(cert.metadata["invariant"]) > 0

    def test_unsafe_system_certificate(self):
        ts = self._make_counter_ts(bound=3)
        cert = generate_pdr_certificate(ts, max_frames=20)
        assert cert.kind == ProofKind.PDR
        assert cert.status == CertStatus.INVALID
        assert cert.metadata["result"] == "unsafe"

    def test_unsafe_has_counterexample(self):
        ts = self._make_counter_ts(bound=3)
        cert = generate_pdr_certificate(ts, max_frames=20)
        assert "counterexample_trace" in cert.metadata

    def test_obligation_formulas_contain_variables(self):
        ts = self._make_safe_ts()
        cert = generate_pdr_certificate(ts)
        # Obligations should mention x
        for ob in cert.obligations:
            assert "x" in ob.formula_str or "true" in ob.formula_str

    def test_obligation_smtlib_is_complete(self):
        ts = self._make_safe_ts()
        cert = generate_pdr_certificate(ts)
        for ob in cert.obligations:
            assert "(set-logic LIA)" in ob.formula_smt
            assert "(check-sat)" in ob.formula_smt
            assert "(assert (not" in ob.formula_smt

    def test_two_variable_system(self):
        """x starts at 0, y starts at 10. x' = x+1, y' = y-1. Property: x + y == 10."""
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        ts.add_int_var("y")
        x, y = ts.var("x"), ts.var("y")
        xp, yp = ts.prime("x"), ts.prime("y")

        init = App(Op.AND, [
            App(Op.EQ, [x, IntConst(0)], BOOL),
            App(Op.EQ, [y, IntConst(10)], BOOL),
        ], BOOL)
        trans = App(Op.AND, [
            App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL),
            App(Op.EQ, [yp, App(Op.SUB, [y, IntConst(1)], INT)], BOOL),
        ], BOOL)
        prop = App(Op.EQ, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL)

        ts.set_init(init)
        ts.set_trans(trans)
        ts.set_property(prop)

        cert = generate_pdr_certificate(ts)
        assert cert.status == CertStatus.VALID
        assert cert.metadata["variables"]["x"] == "Int"
        assert cert.metadata["variables"]["y"] == "Int"


# ============================================================
# Certificate Checking Tests
# ============================================================

class TestCertificateChecker:
    def test_check_valid_vcgen_cert(self):
        source = "fn f(x) { requires(x >= 0); ensures(result >= 0); return x; }"
        cert = generate_vcgen_certificate(source, "f")
        checked = check_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_invalid_vcgen_cert(self):
        source = "fn f(x) { requires(x >= 0); ensures(result > 0); return x; }"
        cert = generate_vcgen_certificate(source, "f")
        checked = check_certificate(cert)
        assert checked.status == CertStatus.INVALID

    def test_check_pdr_cert(self):
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        cert = generate_pdr_certificate(ts)
        checked = check_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_composite_cert(self):
        source = "fn f(x) { requires(x >= 0); ensures(result >= 0); return x; }"
        cert1 = generate_vcgen_certificate(source, "f")

        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL
        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
        cert2 = generate_pdr_certificate(ts)

        composite = combine_certificates(cert1, cert2, claim="Both hold")
        checked = check_certificate(composite)
        assert checked.status == CertStatus.VALID
        assert checked.total_obligations > 0


# ============================================================
# Certificate I/O Tests
# ============================================================

class TestCertificateIO:
    def test_save_and_load(self, tmp_path):
        source = "fn f(x) { requires(x >= 0); ensures(result >= 0); return x; }"
        cert = generate_vcgen_certificate(source, "f")

        path = str(tmp_path / "cert.json")
        save_certificate(cert, path)

        loaded = load_certificate(path)
        assert loaded.kind == cert.kind
        assert loaded.claim == cert.claim
        assert loaded.status == cert.status
        assert len(loaded.obligations) == len(cert.obligations)

    def test_save_load_composite(self, tmp_path):
        sub1 = ProofCertificate(kind=ProofKind.VCGEN, claim="sub1", status=CertStatus.VALID)
        sub2 = ProofCertificate(kind=ProofKind.PDR, claim="sub2", status=CertStatus.VALID)
        comp = combine_certificates(sub1, sub2)

        path = str(tmp_path / "comp.json")
        save_certificate(comp, path)
        loaded = load_certificate(path)
        assert loaded.kind == ProofKind.COMPOSITE
        assert len(loaded.sub_certificates) == 2

    def test_load_and_recheck(self, tmp_path):
        source = "fn f(x) { requires(x >= 0); ensures(result >= 0); return x; }"
        cert = generate_vcgen_certificate(source, "f")
        path = str(tmp_path / "cert.json")
        save_certificate(cert, path)

        # Load and re-check from file
        loaded = load_certificate(path)
        rechecked = check_certificate(loaded)
        assert rechecked.status == CertStatus.VALID


# ============================================================
# Convenience API Tests
# ============================================================

class TestConvenienceAPI:
    def test_certify_program(self):
        source = "fn f(x) { requires(x > 0); ensures(result > 0); return x; }"
        cert = certify_program(source, "f")
        assert cert.status == CertStatus.VALID

    def test_certify_transition_system(self):
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        cert = certify_transition_system(ts)
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations == 3


# ============================================================
# Edge Cases & Robustness
# ============================================================

class TestEdgeCases:
    def test_empty_obligations_cert(self):
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="empty")
        assert cert.total_obligations == 0
        assert cert.valid_obligations == 0

    def test_cert_timestamp_exists(self):
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test")
        assert cert.timestamp is not None
        assert len(cert.timestamp) > 0

    def test_status_enum_values(self):
        assert CertStatus.VALID.value == "valid"
        assert CertStatus.INVALID.value == "invalid"
        assert CertStatus.UNKNOWN.value == "unknown"
        assert CertStatus.UNCHECKED.value == "unchecked"

    def test_proof_kind_values(self):
        assert ProofKind.VCGEN.value == "vcgen"
        assert ProofKind.PDR.value == "pdr"
        assert ProofKind.COMPOSITE.value == "composite"

    def test_combine_empty_certificates(self):
        comp = combine_certificates()
        assert comp.kind == ProofKind.COMPOSITE
        assert comp.total_obligations == 0

    def test_combine_mixed_status(self):
        valid = ProofCertificate(kind=ProofKind.VCGEN, claim="ok", status=CertStatus.VALID)
        invalid = ProofCertificate(kind=ProofKind.VCGEN, claim="bad", status=CertStatus.INVALID)
        comp = combine_certificates(valid, invalid)
        assert comp.status == CertStatus.INVALID

    def test_nested_composite(self):
        inner = combine_certificates(
            ProofCertificate(kind=ProofKind.VCGEN, claim="a", status=CertStatus.VALID),
            ProofCertificate(kind=ProofKind.PDR, claim="b", status=CertStatus.VALID),
        )
        outer = combine_certificates(
            inner,
            ProofCertificate(kind=ProofKind.VCGEN, claim="c", status=CertStatus.VALID),
        )
        assert outer.kind == ProofKind.COMPOSITE
        assert outer.status == CertStatus.VALID


# ============================================================
# PDR Certificate Checking (Independent Verification)
# ============================================================

class TestPDRCertificateChecking:
    def test_initiation_obligation_checks(self):
        """The initiation obligation (Init => Inv) should be independently verifiable."""
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        cert = generate_pdr_certificate(ts)
        checked = check_certificate(cert)

        init_ob = [o for o in checked.obligations if o.name == "initiation"][0]
        assert init_ob.status == CertStatus.VALID

    def test_consecution_obligation_checks(self):
        """The consecution obligation (Inv AND Trans => Inv') should verify."""
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        cert = generate_pdr_certificate(ts)
        checked = check_certificate(cert)

        consec_ob = [o for o in checked.obligations if o.name == "consecution"][0]
        assert consec_ob.status == CertStatus.VALID

    def test_property_obligation_checks(self):
        """The property obligation (Inv => Property) should verify."""
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        cert = generate_pdr_certificate(ts)
        checked = check_certificate(cert)

        prop_ob = [o for o in checked.obligations if o.name == "property"][0]
        assert prop_ob.status == CertStatus.VALID


# ============================================================
# Full Roundtrip: Generate -> Save -> Load -> Check
# ============================================================

class TestFullRoundtrip:
    def test_vcgen_roundtrip(self, tmp_path):
        """Generate, save, load, and re-check a VCGen certificate."""
        source = """
fn double_pos(x) {
    requires(x > 0);
    ensures(result > 0);
    return x + x;
}
"""
        cert = certify_program(source, "double_pos")
        assert cert.status == CertStatus.VALID

        path = str(tmp_path / "double_pos.json")
        save_certificate(cert, path)

        loaded = load_certificate(path)
        rechecked = check_certificate(loaded)
        assert rechecked.status == CertStatus.VALID

    def test_pdr_roundtrip(self, tmp_path):
        """Generate, save, load PDR certificate (obligations preserved)."""
        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL

        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        cert = certify_transition_system(ts)
        assert cert.status == CertStatus.VALID

        path = str(tmp_path / "pdr.json")
        save_certificate(cert, path)

        loaded = load_certificate(path)
        assert loaded.kind == ProofKind.PDR
        assert len(loaded.obligations) == 3
        assert loaded.status == CertStatus.VALID

    def test_composite_roundtrip(self, tmp_path):
        source = "fn f(x) { requires(x >= 0); ensures(result >= 0); return x; }"
        cert1 = certify_program(source, "f")

        from pdr import TransitionSystem
        from smt_solver import Var, IntConst, App, Op, INT, BOOL
        ts = TransitionSystem()
        ts.add_int_var("x")
        x = ts.var("x")
        xp = ts.prime("x")
        ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
        ts.set_trans(App(Op.EQ, [xp, x], BOOL))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
        cert2 = certify_transition_system(ts)

        composite = combine_certificates(cert1, cert2, claim="Full proof")
        path = str(tmp_path / "full.json")
        save_certificate(composite, path)

        loaded = load_certificate(path)
        assert loaded.kind == ProofKind.COMPOSITE
        assert len(loaded.sub_certificates) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
