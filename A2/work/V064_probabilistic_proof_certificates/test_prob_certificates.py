"""Tests for V064: Probabilistic Proof Certificates"""

import pytest
import json
import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from prob_certificates import (
    # Data structures
    StatCertKind, StatisticalEvidence, ProbProofCertificate, PROB_CERT_KIND,
    # Certificate operations
    generate_prob_certificate, check_prob_certificate, check_statistical_evidence,
    combine_prob_certificates, certify_probabilistic,
    # I/O
    save_prob_certificate, load_prob_certificate, certify_and_save, load_and_check,
    # Bridge
    to_v044_certificate, from_v044_certificate,
    # Utilities
    chernoff_min_samples, certificate_report,
)
from proof_certificates import CertStatus, ProofObligation, ProofCertificate, ProofKind


# ============================================================
# 1. StatisticalEvidence data structure
# ============================================================

class TestStatisticalEvidence:
    def test_create_basic(self):
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="x >= 0",
            property_source="x >= 0",
            claimed_threshold=0.9,
            n_samples=100,
            n_successes=95,
            observed_probability=0.95,
            confidence_level=0.95,
            ci_lower=0.89,
            ci_upper=0.98,
        )
        assert ev.n_samples == 100
        assert ev.observed_probability == 0.95
        assert ev.verdict == "unchecked"

    def test_serialization_roundtrip(self):
        ev = StatisticalEvidence(
            kind=StatCertKind.SPRT,
            property_desc="y > 0",
            property_source="y > 0",
            claimed_threshold=0.8,
            n_samples=200,
            n_successes=180,
            observed_probability=0.9,
            confidence_level=0.95,
            ci_lower=0.85,
            ci_upper=0.94,
            sprt_log_ratio=5.0,
            sprt_accept_bound=3.0,
            sprt_reject_bound=-3.0,
            random_var_ranges={"x": (1, 10)},
            verdict="accept",
        )
        d = ev.to_dict()
        ev2 = StatisticalEvidence.from_dict(d)
        assert ev2.kind == StatCertKind.SPRT
        assert ev2.n_samples == 200
        assert ev2.sprt_log_ratio == 5.0
        assert ev2.random_var_ranges == {"x": (1, 10)}
        assert ev2.verdict == "accept"

    def test_chernoff_fields(self):
        ev = StatisticalEvidence(
            kind=StatCertKind.CHERNOFF,
            property_desc="z == 1",
            property_source="z == 1",
            claimed_threshold=0.5,
            n_samples=500,
            n_successes=260,
            observed_probability=0.52,
            confidence_level=0.95,
            ci_lower=0.47,
            ci_upper=0.57,
            chernoff_min_samples=100,
            chernoff_epsilon=0.05,
            chernoff_delta=0.05,
        )
        d = ev.to_dict()
        assert d["chernoff_min_samples"] == 100
        ev2 = StatisticalEvidence.from_dict(d)
        assert ev2.chernoff_epsilon == 0.05


# ============================================================
# 2. ProbProofCertificate data structure
# ============================================================

class TestProbProofCertificate:
    def test_create_empty(self):
        cert = ProbProofCertificate(claim="test")
        assert cert.total_checks == 0
        assert cert.status == CertStatus.UNCHECKED

    def test_with_det_obligations(self):
        obl = ProofObligation(
            name="vc1", description="test vc",
            formula_str="x > 0", formula_smt="",
            status=CertStatus.VALID,
        )
        cert = ProbProofCertificate(
            claim="test",
            deterministic_obligations=[obl],
        )
        assert cert.total_checks == 1
        assert cert.valid_checks == 1

    def test_with_stat_evidence(self):
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="p >= 0.9",
            property_source="x >= 0",
            claimed_threshold=0.9,
            n_samples=100,
            n_successes=95,
            observed_probability=0.95,
            confidence_level=0.95,
            ci_lower=0.89,
            ci_upper=0.98,
            verdict="accept",
        )
        cert = ProbProofCertificate(
            claim="test",
            statistical_evidence=[ev],
        )
        assert cert.total_checks == 1
        assert cert.valid_checks == 1

    def test_invalid_counts(self):
        obl = ProofObligation(
            name="vc1", description="bad",
            formula_str="false", formula_smt="",
            status=CertStatus.INVALID,
        )
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="bad prop",
            property_source="x < 0",
            claimed_threshold=0.9,
            n_samples=100,
            n_successes=10,
            observed_probability=0.1,
            confidence_level=0.95,
            ci_lower=0.05,
            ci_upper=0.17,
            verdict="reject",
        )
        cert = ProbProofCertificate(
            claim="test",
            deterministic_obligations=[obl],
            statistical_evidence=[ev],
        )
        assert cert.invalid_checks == 2

    def test_json_roundtrip(self):
        obl = ProofObligation(
            name="vc1", description="desc",
            formula_str="x > 0", formula_smt="",
            status=CertStatus.VALID,
        )
        ev = StatisticalEvidence(
            kind=StatCertKind.SPRT,
            property_desc="prop1",
            property_source="y >= 0",
            claimed_threshold=0.8,
            n_samples=200,
            n_successes=180,
            observed_probability=0.9,
            confidence_level=0.95,
            ci_lower=0.85,
            ci_upper=0.94,
            verdict="accept",
        )
        cert = ProbProofCertificate(
            claim="roundtrip test",
            source="let x = 1;",
            deterministic_obligations=[obl],
            statistical_evidence=[ev],
            metadata={"fn_name": "foo"},
            status=CertStatus.VALID,
        )
        j = cert.to_json()
        cert2 = ProbProofCertificate.from_json(j)
        assert cert2.claim == "roundtrip test"
        assert cert2.status == CertStatus.VALID
        assert len(cert2.deterministic_obligations) == 1
        assert len(cert2.statistical_evidence) == 1
        assert cert2.statistical_evidence[0].verdict == "accept"

    def test_summary(self):
        cert = ProbProofCertificate(claim="summary test", status=CertStatus.VALID)
        s = cert.summary()
        assert "summary test" in s
        assert "valid" in s.lower()


# ============================================================
# 3. Chernoff-Hoeffding bounds
# ============================================================

class TestChernoffBounds:
    def test_basic(self):
        n = chernoff_min_samples(0.05, 0.05)
        assert n > 0
        # For epsilon=0.05, delta=0.05: n >= ln(40)/(2*0.0025) = 3.69/0.005 = 738
        assert n >= 738

    def test_smaller_epsilon(self):
        n1 = chernoff_min_samples(0.05, 0.05)
        n2 = chernoff_min_samples(0.01, 0.05)
        assert n2 > n1  # Tighter epsilon needs more samples

    def test_smaller_delta(self):
        n1 = chernoff_min_samples(0.05, 0.1)
        n2 = chernoff_min_samples(0.05, 0.01)
        assert n2 > n1  # Higher confidence needs more samples

    def test_edge_cases(self):
        assert chernoff_min_samples(0, 0.05) == 1
        assert chernoff_min_samples(0.05, 0) == 1


# ============================================================
# 4. check_statistical_evidence
# ============================================================

class TestCheckStatisticalEvidence:
    def test_clear_accept(self):
        """High observed probability, threshold well below CI lower bound."""
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="always true",
            property_source="1 == 1",
            claimed_threshold=0.5,
            n_samples=200,
            n_successes=200,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.98,
            ci_upper=1.0,
        )
        result = check_statistical_evidence(ev)
        assert result.verdict == "accept"
        assert result.chernoff_min_samples is not None

    def test_clear_reject(self):
        """Low observed probability, threshold above CI upper bound."""
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="rarely true",
            property_source="x > 100",
            claimed_threshold=0.9,
            n_samples=200,
            n_successes=20,
            observed_probability=0.1,
            confidence_level=0.95,
            ci_lower=0.06,
            ci_upper=0.15,
        )
        result = check_statistical_evidence(ev)
        assert result.verdict == "reject"

    def test_inconclusive(self):
        """CI straddles threshold."""
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="borderline",
            property_source="x >= 5",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=50,
            observed_probability=0.5,
            confidence_level=0.95,
            ci_lower=0.40,
            ci_upper=0.60,
        )
        result = check_statistical_evidence(ev)
        assert result.verdict == "inconclusive"

    def test_consistency_check(self):
        """Rejects if observed_probability != n_successes/n_samples."""
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="inconsistent",
            property_source="x > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=90,
            observed_probability=0.5,  # Inconsistent!
            confidence_level=0.95,
            ci_lower=0.4,
            ci_upper=0.6,
        )
        result = check_statistical_evidence(ev)
        assert result.verdict == "reject"

    def test_zero_samples(self):
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="no data",
            property_source="true",
            claimed_threshold=0.5,
            n_samples=0,
            n_successes=0,
            observed_probability=0.0,
            confidence_level=0.95,
            ci_lower=0.0,
            ci_upper=0.0,
        )
        result = check_statistical_evidence(ev)
        assert result.verdict == "reject"

    def test_sprt_cross_check(self):
        """SPRT reject overrides CI-based accept."""
        ev = StatisticalEvidence(
            kind=StatCertKind.SPRT,
            property_desc="sprt test",
            property_source="x > 0",
            claimed_threshold=0.5,
            n_samples=200,
            n_successes=200,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.98,
            ci_upper=1.0,
            sprt_log_ratio=-10.0,  # Strongly rejects
            sprt_accept_bound=3.0,
            sprt_reject_bound=-3.0,
        )
        result = check_statistical_evidence(ev)
        assert result.verdict == "reject"  # SPRT overrides

    def test_ci_recomputation(self):
        """Checker recomputes CI if stored values are off."""
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="recompute ci",
            property_source="x > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=90,
            observed_probability=0.9,
            confidence_level=0.95,
            ci_lower=0.0,  # Obviously wrong
            ci_upper=0.1,  # Obviously wrong
        )
        result = check_statistical_evidence(ev)
        # CI should be recomputed to reasonable values
        assert result.ci_lower > 0.8
        assert result.ci_upper > 0.9
        assert result.verdict == "accept"  # 0.9 is well above 0.5


# ============================================================
# 5. Certificate generation from V063
# ============================================================

class TestGenerateProbCertificate:
    def test_probabilistic_only(self):
        source = """
let x = random(1, 6);
let y = x + 1;
prob_ensures(y >= 2, 99/100);
"""
        cert = generate_prob_certificate(source, seed=42, n_samples=200)
        assert cert.claim
        assert len(cert.statistical_evidence) >= 1
        assert cert.status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_always_true_property(self):
        source = """
let x = random(1, 10);
prob_ensures(x >= 1, 99/100);
"""
        cert = generate_prob_certificate(source, seed=42, n_samples=200)
        assert len(cert.statistical_evidence) >= 1
        # x is always >= 1, so this should pass
        for ev in cert.statistical_evidence:
            if ev.observed_probability > 0:
                assert ev.observed_probability >= 0.99

    def test_with_function(self):
        source = """
fn dice() {
    let x = random(1, 6);
    prob_ensures(x >= 1, 99/100);
    return x;
}
"""
        cert = generate_prob_certificate(source, fn_name="dice", seed=42, n_samples=200)
        assert cert.metadata.get("fn_name") == "dice"

    def test_metadata(self):
        source = """
let x = random(0, 1);
prob_ensures(x >= 0, 99/100);
"""
        cert = generate_prob_certificate(source, seed=123, n_samples=300)
        assert cert.metadata["seed"] == 123
        assert cert.metadata["n_samples"] == 300

    def test_violation(self):
        source = """
let x = random(1, 10);
prob_ensures(x > 100, 99/100);
"""
        cert = generate_prob_certificate(source, seed=42, n_samples=200)
        # x is never > 100, so this should fail
        has_reject = any(
            ev.verdict in ("reject", "inconclusive")
            or ev.observed_probability < 0.5
            for ev in cert.statistical_evidence
        )
        assert has_reject or cert.status == CertStatus.INVALID


# ============================================================
# 6. Certificate checking
# ============================================================

class TestCheckProbCertificate:
    def test_check_valid(self):
        source = """
let x = random(1, 10);
prob_ensures(x >= 1, 99/100);
"""
        cert = generate_prob_certificate(source, seed=42, n_samples=200)
        checked = check_prob_certificate(cert)
        assert checked.status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_check_empty(self):
        cert = ProbProofCertificate(claim="empty")
        checked = check_prob_certificate(cert)
        assert checked.status == CertStatus.VALID  # Vacuous

    def test_check_with_det_obligations(self):
        source = """
fn inc(x) {
    requires(x >= 0);
    ensures(result >= 1);
    return x + 1;
}
"""
        cert = generate_prob_certificate(source, fn_name="inc")
        checked = check_prob_certificate(cert)
        # Should handle deterministic checking gracefully
        assert checked.status in (CertStatus.VALID, CertStatus.UNKNOWN)


# ============================================================
# 7. Certificate I/O
# ============================================================

class TestCertificateIO:
    def test_save_load_roundtrip(self, tmp_path):
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="test prop",
            property_source="x > 0",
            claimed_threshold=0.9,
            n_samples=100,
            n_successes=95,
            observed_probability=0.95,
            confidence_level=0.95,
            ci_lower=0.89,
            ci_upper=0.98,
            verdict="accept",
        )
        cert = ProbProofCertificate(
            claim="io test",
            source="let x = 1;",
            statistical_evidence=[ev],
            status=CertStatus.VALID,
        )
        path = str(tmp_path / "cert.json")
        save_prob_certificate(cert, path)

        loaded = load_prob_certificate(path)
        assert loaded.claim == "io test"
        assert loaded.status == CertStatus.VALID
        assert len(loaded.statistical_evidence) == 1
        assert loaded.statistical_evidence[0].verdict == "accept"

    def test_certify_and_save(self, tmp_path):
        source = """
let x = random(1, 10);
prob_ensures(x >= 1, 99/100);
"""
        path = str(tmp_path / "cert2.json")
        cert = certify_and_save(source, path, seed=42, n_samples=200)
        assert os.path.exists(path)
        assert cert.status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_load_and_check(self, tmp_path):
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="load test",
            property_source="x > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=100,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.96,
            ci_upper=1.0,
            verdict="accept",
        )
        cert = ProbProofCertificate(
            claim="load check test",
            statistical_evidence=[ev],
            status=CertStatus.VALID,
        )
        path = str(tmp_path / "cert3.json")
        save_prob_certificate(cert, path)

        checked = load_and_check(path)
        assert checked.status == CertStatus.VALID


# ============================================================
# 8. Composite certificates
# ============================================================

class TestCompositeCertificates:
    def test_combine_two(self):
        ev1 = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="prop1",
            property_source="x > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=100,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.96,
            ci_upper=1.0,
            verdict="accept",
        )
        ev2 = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="prop2",
            property_source="y > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=100,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.96,
            ci_upper=1.0,
            verdict="accept",
        )
        c1 = ProbProofCertificate(claim="cert1", statistical_evidence=[ev1], status=CertStatus.VALID)
        c2 = ProbProofCertificate(claim="cert2", statistical_evidence=[ev2], status=CertStatus.VALID)

        combined = combine_prob_certificates(c1, c2, claim="both props")
        assert combined.total_checks == 2
        assert combined.valid_checks == 2
        assert combined.status == CertStatus.VALID

    def test_combine_with_invalid(self):
        ev1 = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="good",
            property_source="x > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=100,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.96,
            ci_upper=1.0,
            verdict="accept",
        )
        ev2 = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="bad",
            property_source="x < 0",
            claimed_threshold=0.9,
            n_samples=100,
            n_successes=5,
            observed_probability=0.05,
            confidence_level=0.95,
            ci_lower=0.02,
            ci_upper=0.11,
            verdict="reject",
        )
        c1 = ProbProofCertificate(claim="good", statistical_evidence=[ev1], status=CertStatus.VALID)
        c2 = ProbProofCertificate(claim="bad", statistical_evidence=[ev2], status=CertStatus.INVALID)

        combined = combine_prob_certificates(c1, c2)
        assert combined.status == CertStatus.INVALID

    def test_combine_mixed_det_stat(self):
        obl = ProofObligation(
            name="vc1", description="det",
            formula_str="x > 0", formula_smt="",
            status=CertStatus.VALID,
        )
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="stat",
            property_source="y > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=100,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.96,
            ci_upper=1.0,
            verdict="accept",
        )
        c1 = ProbProofCertificate(claim="det", deterministic_obligations=[obl], status=CertStatus.VALID)
        c2 = ProbProofCertificate(claim="stat", statistical_evidence=[ev], status=CertStatus.VALID)

        combined = combine_prob_certificates(c1, c2)
        assert combined.total_checks == 2
        assert combined.valid_checks == 2
        assert combined.status == CertStatus.VALID


# ============================================================
# 9. Bridge to V044
# ============================================================

class TestV044Bridge:
    def test_to_v044(self):
        obl = ProofObligation(
            name="vc1", description="det vc",
            formula_str="x > 0", formula_smt="",
            status=CertStatus.VALID,
        )
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="stat vc",
            property_source="y > 0",
            claimed_threshold=0.9,
            n_samples=100,
            n_successes=95,
            observed_probability=0.95,
            confidence_level=0.95,
            ci_lower=0.89,
            ci_upper=0.98,
            verdict="accept",
        )
        cert = ProbProofCertificate(
            claim="bridge test",
            source="let x = 1;",
            deterministic_obligations=[obl],
            statistical_evidence=[ev],
            status=CertStatus.VALID,
        )
        v044 = to_v044_certificate(cert)
        assert v044.kind == ProofKind.COMPOSITE
        assert len(v044.obligations) == 2  # 1 det + 1 stat
        assert v044.metadata.get("probabilistic") is True

    def test_roundtrip_bridge(self):
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="rt prop",
            property_source="x > 0",
            claimed_threshold=0.5,
            n_samples=100,
            n_successes=100,
            observed_probability=1.0,
            confidence_level=0.95,
            ci_lower=0.96,
            ci_upper=1.0,
            verdict="accept",
        )
        cert = ProbProofCertificate(
            claim="bridge rt",
            statistical_evidence=[ev],
            metadata={"probabilistic": True},
            status=CertStatus.VALID,
        )
        v044 = to_v044_certificate(cert)
        back = from_v044_certificate(v044)
        assert back is not None
        assert len(back.statistical_evidence) == 1

    def test_non_probabilistic_returns_none(self):
        v044 = ProofCertificate(
            kind=ProofKind.VCGEN,
            claim="not probabilistic",
        )
        result = from_v044_certificate(v044)
        assert result is None


# ============================================================
# 10. Certificate report
# ============================================================

class TestCertificateReport:
    def test_basic_report(self):
        obl = ProofObligation(
            name="vc1", description="det",
            formula_str="x > 0", formula_smt="",
            status=CertStatus.VALID,
        )
        ev = StatisticalEvidence(
            kind=StatCertKind.MONTE_CARLO,
            property_desc="stat prop",
            property_source="y >= 0",
            claimed_threshold=0.9,
            n_samples=100,
            n_successes=95,
            observed_probability=0.95,
            confidence_level=0.95,
            ci_lower=0.89,
            ci_upper=0.98,
            verdict="accept",
        )
        cert = ProbProofCertificate(
            claim="report test",
            deterministic_obligations=[obl],
            statistical_evidence=[ev],
            status=CertStatus.VALID,
        )
        report = certificate_report(cert)
        assert report["claim"] == "report test"
        assert report["status"] == "valid"
        assert report["total_checks"] == 2
        assert len(report["deterministic"]) == 1
        assert len(report["statistical"]) == 1
        assert report["statistical"][0]["threshold"] == 0.9
        assert report["statistical"][0]["verdict"] == "accept"


# ============================================================
# 11. certify_probabilistic (one-shot API)
# ============================================================

class TestCertifyProbabilistic:
    def test_always_true(self):
        source = """
let x = random(1, 10);
prob_ensures(x >= 1, 99/100);
"""
        cert = certify_probabilistic(source, seed=42, n_samples=200)
        assert cert.status in (CertStatus.VALID, CertStatus.UNKNOWN)
        assert cert.total_checks >= 1

    def test_impossible_property(self):
        source = """
let x = random(1, 10);
prob_ensures(x > 100, 99/100);
"""
        cert = certify_probabilistic(source, seed=42, n_samples=200)
        assert cert.status == CertStatus.INVALID or any(
            ev.verdict in ("reject", "inconclusive") for ev in cert.statistical_evidence
        )


# ============================================================
# 12. Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_source(self):
        cert = generate_prob_certificate("let x = 1;", seed=42)
        # No annotations -> no VCs
        assert cert.total_checks == 0

    def test_parse_error(self):
        cert = generate_prob_certificate("this is not valid c10 {{{", seed=42)
        # Should handle gracefully
        assert cert.status in (CertStatus.UNKNOWN, CertStatus.INVALID)

    def test_multiple_prob_ensures(self):
        source = """
let x = random(1, 6);
prob_ensures(x >= 1, 99/100);
prob_ensures(x <= 6, 99/100);
"""
        cert = generate_prob_certificate(source, seed=42, n_samples=200)
        assert len(cert.statistical_evidence) >= 2

    def test_cert_kind_string(self):
        assert PROB_CERT_KIND == "probabilistic"

    def test_stat_cert_kind_values(self):
        assert StatCertKind.MONTE_CARLO.value == "monte_carlo"
        assert StatCertKind.SPRT.value == "sprt"
        assert StatCertKind.CHERNOFF.value == "chernoff"
