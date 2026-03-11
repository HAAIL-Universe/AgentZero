"""Tests for V130: Certified Effect Analysis."""

import pytest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V040_effect_systems'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))

from certified_effect_analysis import (
    # Types
    EffectCertificate, EffectCertKind,
    # Certification
    certify_effect_soundness, certify_effect_purity,
    certify_effect_completeness, certify_full_effects,
    # Checking
    check_effect_certificate, certify_and_check,
    # V044 bridge
    to_v044_certificate, from_v044_certificate,
    # I/O
    save_effect_certificate, load_effect_certificate,
    # Comparison
    compare_with_uncertified, effect_certificate_summary,
    # Internal
    _make_soundness_obligations, _make_purity_obligations,
    _make_completeness_obligations, _make_handler_obligations,
    _effect_smt,
)
from effect_systems import (
    EffectKind, Effect, EffectSet, State, Exn, IO, DIV, PURE, NONDET,
)
from proof_certificates import ProofCertificate, ProofKind, CertStatus


# ---- Source snippets ----

PURE_FN = """
fn add(x, y) {
    return x + y;
}
"""

STATEFUL_FN = """
let counter = 0;
fn increment() {
    counter = counter + 1;
}
"""

IO_FN = """
fn greet(name) {
    print(name);
}
"""

MIXED_FN = """
let total = 0;
fn process(x) {
    total = total + x;
    print(total);
}
"""

LOOP_FN = """
fn count(n) {
    let i = 0;
    while (i < n) {
        i = i + 1;
    }
    return i;
}
"""

MULTI_FN = """
fn pure_add(x, y) {
    return x + y;
}
fn show(x) {
    print(x);
}
fn compute(a, b) {
    let r = pure_add(a, b);
    show(r);
    return r;
}
"""


# ===========================================================================
# 1. Basic Soundness Certification
# ===========================================================================

class TestSoundness:
    def test_pure_fn_sound(self):
        cert = certify_effect_soundness(PURE_FN)
        assert cert.status == CertStatus.VALID
        assert cert.kind == EffectCertKind.SOUNDNESS

    def test_stateful_fn_sound(self):
        cert = certify_effect_soundness(STATEFUL_FN)
        assert cert.status == CertStatus.VALID

    def test_io_fn_sound(self):
        cert = certify_effect_soundness(IO_FN)
        assert cert.status == CertStatus.VALID

    def test_has_obligations(self):
        cert = certify_effect_soundness(STATEFUL_FN)
        assert cert.total_obligations >= 1


# ===========================================================================
# 2. Purity Certification
# ===========================================================================

class TestPurity:
    def test_pure_fn_certified_pure(self):
        cert = certify_effect_purity(PURE_FN, fn_names=["add"])
        assert cert.status == CertStatus.VALID

    def test_stateful_fn_not_pure(self):
        cert = certify_effect_purity(STATEFUL_FN, fn_names=["increment"])
        assert cert.status == CertStatus.INVALID

    def test_io_fn_not_pure(self):
        cert = certify_effect_purity(IO_FN, fn_names=["greet"])
        assert cert.status == CertStatus.INVALID

    def test_loop_fn_purity(self):
        """Loops add DIV effect, so loop functions are not pure."""
        cert = certify_effect_purity(LOOP_FN, fn_names=["count"])
        # Loops have DIV effect
        assert cert.status == CertStatus.INVALID


# ===========================================================================
# 3. Completeness Certification
# ===========================================================================

class TestCompleteness:
    def test_exact_match(self):
        declared = {"add": EffectSet.pure()}
        cert = certify_effect_completeness(PURE_FN, declared)
        assert cert.status == CertStatus.VALID

    def test_over_approximation(self):
        declared = {"add": EffectSet.of(IO)}
        cert = certify_effect_completeness(PURE_FN, declared)
        # IO is declared but not inferred -> backward check fails
        assert cert.status == CertStatus.INVALID

    def test_under_approximation(self):
        declared = {"greet": EffectSet.pure()}
        cert = certify_effect_completeness(IO_FN, declared)
        # IO is inferred but not declared -> forward check fails
        assert cert.status == CertStatus.INVALID


# ===========================================================================
# 4. Full Certification
# ===========================================================================

class TestFullCertification:
    def test_pure_fn_full(self):
        cert = certify_full_effects(PURE_FN)
        assert cert.status == CertStatus.VALID
        assert cert.kind == EffectCertKind.FULL

    def test_mixed_fn_full(self):
        cert = certify_full_effects(MIXED_FN)
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations >= 1

    def test_multi_fn_full(self):
        cert = certify_full_effects(MULTI_FN)
        assert cert.status == CertStatus.VALID


# ===========================================================================
# 5. Independent Checking
# ===========================================================================

class TestChecking:
    def test_check_pure(self):
        cert = certify_and_check(PURE_FN)
        assert cert.status == CertStatus.VALID

    def test_check_preserves_obligations(self):
        cert = certify_full_effects(STATEFUL_FN)
        checked = check_effect_certificate(cert)
        assert len(checked.obligations) == len(cert.obligations)

    def test_check_stateful(self):
        cert = certify_and_check(STATEFUL_FN)
        assert cert.status == CertStatus.VALID


# ===========================================================================
# 6. Serialization
# ===========================================================================

class TestSerialization:
    def test_to_dict_from_dict(self):
        cert = certify_full_effects(PURE_FN)
        d = cert.to_dict()
        cert2 = EffectCertificate.from_dict(d)
        assert cert2.status == cert.status
        assert cert2.kind == cert.kind

    def test_json_roundtrip(self):
        cert = certify_full_effects(PURE_FN)
        j = cert.to_json()
        cert2 = EffectCertificate.from_json(j)
        assert cert2.status == cert.status
        assert len(cert2.obligations) == len(cert.obligations)

    def test_save_load_file(self):
        cert = certify_full_effects(PURE_FN)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_effect_certificate(cert, path)
            loaded = load_effect_certificate(path)
            assert loaded.status == cert.status
            assert loaded.kind == cert.kind
        finally:
            os.unlink(path)


# ===========================================================================
# 7. V044 Bridge
# ===========================================================================

class TestV044Bridge:
    def test_to_v044(self):
        cert = certify_full_effects(PURE_FN)
        v044 = to_v044_certificate(cert)
        assert isinstance(v044, ProofCertificate)
        assert v044.kind == ProofKind.VCGEN
        assert "effect" in v044.claim.lower()

    def test_v044_has_obligations(self):
        cert = certify_full_effects(STATEFUL_FN)
        v044 = to_v044_certificate(cert)
        assert len(v044.obligations) == len(cert.obligations)

    def test_v044_metadata(self):
        cert = certify_full_effects(PURE_FN)
        v044 = to_v044_certificate(cert)
        assert v044.metadata.get("effect_analysis") is True

    def test_from_v044(self):
        cert = certify_full_effects(PURE_FN)
        v044 = to_v044_certificate(cert)
        back = from_v044_certificate(v044, PURE_FN)
        assert isinstance(back, EffectCertificate)
        assert back.status == cert.status


# ===========================================================================
# 8. Comparison
# ===========================================================================

class TestComparison:
    def test_compare(self):
        comp = compare_with_uncertified(PURE_FN)
        assert "uncertified" in comp
        assert "certified" in comp
        assert comp["uncertified"]["time"] >= 0
        assert comp["certified"]["time"] >= 0

    def test_compare_stateful(self):
        comp = compare_with_uncertified(STATEFUL_FN)
        assert comp["certified"]["obligations_total"] >= 1


# ===========================================================================
# 9. Summary
# ===========================================================================

class TestSummary:
    def test_summary_output(self):
        cert = certify_full_effects(PURE_FN)
        s = cert.summary()
        assert "Effect Certificate" in s
        assert cert.status.value.upper() in s.upper()

    def test_full_summary(self):
        s = effect_certificate_summary(MULTI_FN)
        assert "Effect" in s


# ===========================================================================
# 10. Obligation Details
# ===========================================================================

class TestObligationDetails:
    def test_soundness_obligations_have_smt(self):
        obls = _make_soundness_obligations(
            "test_fn", EffectSet.of(IO), EffectSet.of(IO)
        )
        for obl in obls:
            assert obl.formula_smt
            assert "set-logic" in obl.formula_smt

    def test_purity_valid(self):
        obls = _make_purity_obligations("pure_fn", EffectSet.pure())
        assert len(obls) == 1
        assert obls[0].status == CertStatus.VALID

    def test_purity_invalid(self):
        obls = _make_purity_obligations("io_fn", EffectSet.of(IO))
        assert len(obls) == 1
        assert obls[0].status == CertStatus.INVALID

    def test_handler_obligations(self):
        body = EffectSet.of(IO, Effect(EffectKind.EXN, "Error"))
        handled = EffectSet.of(Effect(EffectKind.EXN, "Error"))
        obls = _make_handler_obligations("fn", body, handled)
        assert len(obls) >= 1
        assert all(o.status == CertStatus.VALID for o in obls)


# ===========================================================================
# 11. Effect SMT Encoding
# ===========================================================================

class TestEffectSMT:
    def test_io_encoding(self):
        s = _effect_smt(IO)
        assert "io" in s

    def test_state_encoding(self):
        s = _effect_smt(State("x"))
        assert "state" in s
        assert "x" in s

    def test_exn_encoding(self):
        s = _effect_smt(Exn("TypeError"))
        assert "exn" in s
        assert "TypeError" in s


# ===========================================================================
# 12. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_program(self):
        cert = certify_full_effects("let x = 5;")
        assert cert.status == CertStatus.VALID

    def test_multiple_state_vars(self):
        source = """
let a = 0;
let b = 0;
fn swap() {
    let tmp = a;
    a = b;
    b = tmp;
}
"""
        cert = certify_full_effects(source)
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations >= 1

    def test_function_calls(self):
        cert = certify_full_effects(MULTI_FN)
        # compute calls pure_add (pure) and show (IO)
        assert cert.status == CertStatus.VALID


# ===========================================================================
# 13. Certificate Fields
# ===========================================================================

class TestCertificateFields:
    def test_metadata(self):
        cert = certify_full_effects(PURE_FN)
        assert "fn_count" in cert.metadata

    def test_fn_sigs_populated(self):
        cert = certify_full_effects(MULTI_FN)
        assert len(cert.fn_sigs) >= 3  # pure_add, show, compute

    def test_valid_invalid_counts(self):
        cert = certify_full_effects(PURE_FN)
        assert cert.valid_obligations == cert.total_obligations
        assert cert.invalid_obligations == 0
