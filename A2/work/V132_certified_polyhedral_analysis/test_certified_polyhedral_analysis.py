"""Tests for V132: Certified Polyhedral Analysis."""

import sys
import os
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))

from certified_polyhedral_analysis import (
    PolyhedralCertificate, PolyhedralCertKind,
    certify_polyhedral_bounds, certify_polyhedral_relational,
    certify_polyhedral_feasibility, certify_polyhedral_properties,
    certify_full_polyhedral, check_polyhedral_certificate,
    certify_and_check, to_v044_certificate, from_v044_certificate,
    save_polyhedral_certificate, load_polyhedral_certificate,
    compare_certified_vs_uncertified, polyhedral_certificate_summary,
    get_certified_bounds, get_certified_constraints,
)
from proof_certificates import CertStatus, ProofKind


# ---------------------------------------------------------------------------
# Test programs
# ---------------------------------------------------------------------------

SIMPLE_ASSIGN = "let x = 10; let y = 20;"
LINEAR_COMBO = "let x = 5; let y = x + 3;"
CONDITIONAL = """
let x = 10;
let y = 0;
if (x > 5) {
    y = x + 1;
} else {
    y = x - 1;
}
"""
LOOP_PROGRAM = """
let x = 0;
let y = 100;
while (x < 10) {
    x = x + 1;
    y = y - 1;
}
"""
MULTI_VAR = """
let a = 1;
let b = 2;
let c = a + b;
let d = c * 2;
"""
EMPTY_PROGRAM = "let x = 0;"
RELATIONAL = """
let x = 5;
let y = x + 3;
let z = y - x;
"""

# ---------------------------------------------------------------------------
# 1. Bounds certification
# ---------------------------------------------------------------------------

class TestBoundsCertification:
    def test_simple_bounds(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        assert cert.kind == PolyhedralCertKind.BOUNDS
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations >= 2  # x and y

    def test_bounds_have_valid_status(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        for obl in cert.obligations:
            assert obl.status == CertStatus.VALID

    def test_bounds_obligation_names(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        names = [o.name for o in cert.obligations]
        assert "bounds_x" in names
        assert "bounds_y" in names

    def test_bounds_formula_str(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        x_obl = [o for o in cert.obligations if o.name == "bounds_x"][0]
        assert "10" in x_obl.formula_str

    def test_bounds_smt_generated(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        for obl in cert.obligations:
            assert "set-logic" in obl.formula_smt
            assert "check-sat" in obl.formula_smt

    def test_linear_combo_bounds(self):
        cert = certify_polyhedral_bounds(LINEAR_COMBO)
        assert cert.status == CertStatus.VALID
        y_obl = [o for o in cert.obligations if o.name == "bounds_y"][0]
        assert "8" in y_obl.formula_str

    def test_loop_bounds(self):
        cert = certify_polyhedral_bounds(LOOP_PROGRAM)
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations >= 1

    def test_metadata_variables(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        assert "x" in cert.metadata["variables"]
        assert "y" in cert.metadata["variables"]

    def test_metadata_constraints_count(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        assert cert.metadata["constraints_count"] > 0

    def test_conditional_bounds(self):
        cert = certify_polyhedral_bounds(CONDITIONAL)
        assert cert.status == CertStatus.VALID


# ---------------------------------------------------------------------------
# 2. Relational certification
# ---------------------------------------------------------------------------

class TestRelationalCertification:
    def test_relational_basic(self):
        cert = certify_polyhedral_relational(RELATIONAL)
        assert cert.kind == PolyhedralCertKind.RELATIONAL
        assert cert.status == CertStatus.VALID

    def test_relational_obligation_count(self):
        cert = certify_polyhedral_relational(RELATIONAL)
        assert cert.metadata["relational_count"] >= 0

    def test_relational_smt_generated(self):
        cert = certify_polyhedral_relational(RELATIONAL)
        for obl in cert.obligations:
            assert "set-logic" in obl.formula_smt

    def test_simple_no_relational(self):
        # x = 10 alone has no relational constraints
        cert = certify_polyhedral_relational(EMPTY_PROGRAM)
        assert cert.status == CertStatus.VALID

    def test_linear_combo_relational(self):
        cert = certify_polyhedral_relational(LINEAR_COMBO)
        assert cert.status == CertStatus.VALID


# ---------------------------------------------------------------------------
# 3. Feasibility certification
# ---------------------------------------------------------------------------

class TestFeasibilityCertification:
    def test_feasible_program(self):
        cert = certify_polyhedral_feasibility(SIMPLE_ASSIGN)
        assert cert.kind == PolyhedralCertKind.FEASIBILITY
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations == 1
        assert cert.metadata["feasible"] is True

    def test_feasibility_obligation_name(self):
        cert = certify_polyhedral_feasibility(SIMPLE_ASSIGN)
        assert cert.obligations[0].name == "feasibility"

    def test_empty_program_feasible(self):
        cert = certify_polyhedral_feasibility(EMPTY_PROGRAM)
        assert cert.status == CertStatus.VALID


# ---------------------------------------------------------------------------
# 4. Property certification
# ---------------------------------------------------------------------------

class TestPropertyCertification:
    def test_valid_property(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 10"])
        assert cert.kind == PolyhedralCertKind.PROPERTY
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations == 1

    def test_invalid_property(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 5"])
        assert cert.status == CertStatus.INVALID
        assert cert.obligations[0].status == CertStatus.INVALID

    def test_multiple_properties(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 10", "y <= 20", "x >= 10"])
        assert cert.total_obligations == 3
        assert cert.valid_obligations == 3

    def test_equality_property(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x == 10"])
        assert cert.status == CertStatus.VALID

    def test_ge_property(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x >= 10"])
        assert cert.status == CertStatus.VALID

    def test_invalid_ge_property(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x >= 15"])
        assert cert.status == CertStatus.INVALID

    def test_linear_combo_property(self):
        cert = certify_polyhedral_properties(LINEAR_COMBO, ["y <= 8"])
        assert cert.status == CertStatus.VALID

    def test_property_metadata(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 10", "y >= 0"])
        assert cert.metadata["properties_checked"] == 2
        assert cert.metadata["properties_verified"] >= 1

    def test_property_smt_generated(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 10"])
        assert "set-logic" in cert.obligations[0].formula_smt


# ---------------------------------------------------------------------------
# 5. Full certification
# ---------------------------------------------------------------------------

class TestFullCertification:
    def test_full_basic(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        assert cert.kind == PolyhedralCertKind.FULL
        assert cert.status == CertStatus.VALID
        # At least feasibility + bounds for x, y
        assert cert.total_obligations >= 3

    def test_full_with_properties(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN, properties=["x <= 10"])
        assert cert.status == CertStatus.VALID
        prop_obls = [o for o in cert.obligations if o.name.startswith("property_")]
        assert len(prop_obls) == 1

    def test_full_invalid_property(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN, properties=["x <= 5"])
        assert cert.status == CertStatus.INVALID

    def test_full_metadata(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        assert "variables" in cert.metadata
        assert "constraints_count" in cert.metadata

    def test_full_loop(self):
        cert = certify_full_polyhedral(LOOP_PROGRAM)
        assert cert.status == CertStatus.VALID

    def test_full_conditional(self):
        cert = certify_full_polyhedral(CONDITIONAL)
        assert cert.status == CertStatus.VALID


# ---------------------------------------------------------------------------
# 6. Independent checking
# ---------------------------------------------------------------------------

class TestIndependentChecking:
    def test_check_bounds_cert(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        checked = check_polyhedral_certificate(cert)
        assert checked.status == CertStatus.VALID
        assert checked.valid_obligations == cert.valid_obligations

    def test_check_full_cert(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        checked = check_polyhedral_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_property_cert(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 10"])
        checked = check_polyhedral_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_invalid_property(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 5"])
        checked = check_polyhedral_certificate(cert)
        assert checked.status == CertStatus.INVALID

    def test_certify_and_check(self):
        cert = certify_and_check(SIMPLE_ASSIGN)
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations >= 3

    def test_certify_and_check_with_properties(self):
        cert = certify_and_check(SIMPLE_ASSIGN, properties=["x <= 10", "y <= 20"])
        assert cert.status == CertStatus.VALID

    def test_check_relational_cert(self):
        cert = certify_polyhedral_relational(RELATIONAL)
        checked = check_polyhedral_certificate(cert)
        assert checked.status == CertStatus.VALID

    def test_check_feasibility_cert(self):
        cert = certify_polyhedral_feasibility(SIMPLE_ASSIGN)
        checked = check_polyhedral_certificate(cert)
        assert checked.status == CertStatus.VALID


# ---------------------------------------------------------------------------
# 7. V044 bridge
# ---------------------------------------------------------------------------

class TestV044Bridge:
    def test_to_v044(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        v044 = to_v044_certificate(cert)
        assert v044.kind == ProofKind.VCGEN
        assert v044.status == cert.status
        assert len(v044.obligations) == len(cert.obligations)
        assert v044.metadata.get("polyhedral_analysis") is True

    def test_from_v044(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        v044 = to_v044_certificate(cert)
        roundtrip = from_v044_certificate(v044, source=SIMPLE_ASSIGN)
        assert roundtrip.kind == cert.kind
        assert roundtrip.status == cert.status
        assert len(roundtrip.obligations) == len(cert.obligations)

    def test_v044_roundtrip_status(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 5"])
        v044 = to_v044_certificate(cert)
        roundtrip = from_v044_certificate(v044)
        assert roundtrip.status == CertStatus.INVALID

    def test_v044_metadata_cert_kind(self):
        cert = certify_polyhedral_bounds(SIMPLE_ASSIGN)
        v044 = to_v044_certificate(cert)
        assert v044.metadata["cert_kind"] == "bounds"

    def test_v044_claim(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        v044 = to_v044_certificate(cert)
        assert "polyhedral" in v044.claim.lower()


# ---------------------------------------------------------------------------
# 8. Serialization (I/O)
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        d = cert.to_dict()
        assert d["kind"] == "full"
        assert d["status"] == "valid"
        assert "obligations" in d
        assert "metadata" in d

    def test_from_dict_roundtrip(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        d = cert.to_dict()
        restored = PolyhedralCertificate.from_dict(d)
        assert restored.kind == cert.kind
        assert restored.status == cert.status
        assert len(restored.obligations) == len(cert.obligations)

    def test_to_json(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        j = cert.to_json()
        parsed = json.loads(j)
        assert parsed["kind"] == "full"

    def test_from_json_roundtrip(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        j = cert.to_json()
        restored = PolyhedralCertificate.from_json(j)
        assert restored.kind == cert.kind
        assert restored.status == cert.status
        assert len(restored.obligations) == len(cert.obligations)

    def test_save_and_load(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            save_polyhedral_certificate(cert, path)
            loaded = load_polyhedral_certificate(path)
            assert loaded.kind == cert.kind
            assert loaded.status == cert.status
            assert len(loaded.obligations) == len(cert.obligations)
        finally:
            os.unlink(path)

    def test_save_load_preserves_obligations(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 10"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            save_polyhedral_certificate(cert, path)
            loaded = load_polyhedral_certificate(path)
            assert loaded.obligations[0].name == cert.obligations[0].name
            assert loaded.obligations[0].status == cert.obligations[0].status
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 9. Summary and comparison
# ---------------------------------------------------------------------------

class TestSummaryAndComparison:
    def test_summary_output(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        s = cert.summary()
        assert "Polyhedral Certificate" in s
        assert "valid" in s

    def test_polyhedral_certificate_summary(self):
        s = polyhedral_certificate_summary(SIMPLE_ASSIGN)
        assert "Polyhedral Certificate" in s

    def test_compare_certified_vs_uncertified(self):
        result = compare_certified_vs_uncertified(SIMPLE_ASSIGN)
        assert "uncertified" in result
        assert "certified" in result
        assert result["certified"]["status"] == "valid"
        assert result["certified"]["obligations_total"] >= 1
        assert result["uncertified"]["variables"] >= 2

    def test_compare_timing(self):
        result = compare_certified_vs_uncertified(SIMPLE_ASSIGN)
        assert result["uncertified"]["time"] >= 0
        assert result["certified"]["time"] >= 0


# ---------------------------------------------------------------------------
# 10. Convenience APIs
# ---------------------------------------------------------------------------

class TestConvenienceAPIs:
    def test_get_certified_bounds(self):
        result = get_certified_bounds(SIMPLE_ASSIGN, "x")
        assert result["variable"] == "x"
        assert result["status"] == "valid"
        assert "10" in result["bounds_description"]

    def test_get_certified_bounds_nonexistent(self):
        result = get_certified_bounds(SIMPLE_ASSIGN, "nonexistent")
        assert result["status"] == "unconstrained"

    def test_get_certified_constraints(self):
        constraints = get_certified_constraints(SIMPLE_ASSIGN)
        assert len(constraints) >= 1
        for c in constraints:
            assert "name" in c
            assert "status" in c

    def test_get_certified_constraints_loop(self):
        constraints = get_certified_constraints(LOOP_PROGRAM)
        assert len(constraints) >= 1


# ---------------------------------------------------------------------------
# 11. Certificate properties
# ---------------------------------------------------------------------------

class TestCertificateProperties:
    def test_total_obligations(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        assert cert.total_obligations == len(cert.obligations)

    def test_valid_obligations(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        assert cert.valid_obligations == cert.total_obligations

    def test_invalid_obligations_count(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, ["x <= 5"])
        assert cert.invalid_obligations >= 1

    def test_timestamp_set(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN)
        assert cert.timestamp != ""


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_variable(self):
        cert = certify_full_polyhedral(EMPTY_PROGRAM)
        assert cert.status == CertStatus.VALID

    def test_multi_variable_program(self):
        cert = certify_full_polyhedral(MULTI_VAR)
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations >= 1

    def test_conditional_program(self):
        cert = certify_full_polyhedral(CONDITIONAL)
        assert cert.status == CertStatus.VALID

    def test_relational_program(self):
        cert = certify_full_polyhedral(RELATIONAL)
        assert cert.status == CertStatus.VALID

    def test_empty_properties_list(self):
        cert = certify_polyhedral_properties(SIMPLE_ASSIGN, [])
        assert cert.status == CertStatus.VALID
        assert cert.total_obligations == 0

    def test_full_with_empty_properties(self):
        cert = certify_full_polyhedral(SIMPLE_ASSIGN, properties=[])
        assert cert.status == CertStatus.VALID
