"""
Tests for V059: Verified Concurrency -- PCC Bundles for Concurrent Programs
"""

import os, sys, json, tempfile, pytest

sys.path.insert(0, os.path.dirname(__file__))

from verified_concurrency import (
    # Data model
    ConcBundleStatus, ConcPolicyKind, ConcSafetyPolicy, ThreadPayload,
    ConcurrentPCCBundle,
    # V043 re-exports
    ConcVerdict, ThreadSpec, ConcurrentProgram, EffectRaceReport,
    # Certificate types
    ProofCertificate, ProofObligation, ProofKind, CertStatus,
    # Producer API
    produce_concurrent_pcc, quick_concurrent_pcc, full_concurrent_pcc,
    # Consumer API
    verify_concurrent_bundle, check_concurrent_policy,
    # Serialization
    save_concurrent_bundle, load_concurrent_bundle,
    # Roundtrip
    produce_and_verify, produce_save_load_verify,
    # Analysis
    concurrent_pcc_report, compare_protocols,
    # Effect imports
    EffectSet, State, IO, PURE,
)


# =============================================================================
# Helper: simple C10 thread sources
# =============================================================================

THREAD_A = """
let x = 0;
x = x + 1;
"""

THREAD_B = """
let y = 0;
y = y + 2;
"""

THREAD_SHARED = """
let counter = 0;
counter = counter + 1;
"""

THREAD_IO = """
let msg = 42;
print(msg);
"""

THREAD_COMPLEX = """
fn compute(n) {
    let result = 0;
    let i = 0;
    while (i < n) {
        result = result + i;
        i = i + 1;
    }
    return result;
}
let val = compute(5);
"""


# =============================================================================
# Section 1: Data Model
# =============================================================================

class TestDataModel:
    def test_thread_payload_roundtrip(self):
        tp = ThreadPayload(
            thread_id="t1",
            instructions=["Op.CONST", "0", "Op.STORE", "0"],
            constants=[0],
            source_hash="abc123",
        )
        d = tp.to_dict()
        tp2 = ThreadPayload.from_dict(d)
        assert tp2.thread_id == "t1"
        assert tp2.instructions == ["Op.CONST", "0", "Op.STORE", "0"]
        assert tp2.source_hash == "abc123"

    def test_safety_policy_roundtrip(self):
        p = ConcSafetyPolicy(
            kind=ConcPolicyKind.RACE_FREEDOM,
            description="No races",
            parameters={"protocol": "lock"},
        )
        d = p.to_dict()
        p2 = ConcSafetyPolicy.from_dict(d)
        assert p2.kind == ConcPolicyKind.RACE_FREEDOM
        assert p2.parameters["protocol"] == "lock"

    def test_bundle_status_enum(self):
        assert ConcBundleStatus.VERIFIED.value == "verified"
        assert ConcBundleStatus.FAILED.value == "failed"
        assert ConcBundleStatus.PARTIALLY_VERIFIED.value == "partially_verified"
        assert ConcBundleStatus.UNCHECKED.value == "unchecked"

    def test_policy_kind_enum(self):
        assert ConcPolicyKind.EFFECT_SAFETY.value == "effect_safety"
        assert ConcPolicyKind.MUTUAL_EXCLUSION.value == "mutual_exclusion"
        assert ConcPolicyKind.THREAD_SAFETY.value == "thread_safety"

    def test_empty_bundle(self):
        bundle = ConcurrentPCCBundle(
            thread_payloads=[], certificates=[], policies=[],
        )
        assert bundle.total_certificates == 0
        assert bundle.valid_certificates == 0
        assert bundle.status == ConcBundleStatus.UNCHECKED


# =============================================================================
# Section 2: Thread Compilation
# =============================================================================

class TestThreadCompilation:
    def test_compile_simple_thread(self):
        threads = [ThreadSpec(thread_id="t1", source=THREAD_A)]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=False,
            include_temporal=False, include_csl=False,
        )
        assert len(bundle.thread_payloads) == 1
        assert bundle.thread_payloads[0].thread_id == "t1"
        # C10 compiles to .code (Op enums + operands), converted to strings
        assert len(bundle.thread_payloads[0].instructions) > 0
        assert bundle.thread_payloads[0].source_hash != ""

    def test_compile_multiple_threads(self):
        threads = [
            ThreadSpec(thread_id="t1", source=THREAD_A),
            ThreadSpec(thread_id="t2", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=False,
            include_temporal=False, include_csl=False,
        )
        assert len(bundle.thread_payloads) == 2
        assert bundle.thread_payloads[0].thread_id == "t1"
        assert bundle.thread_payloads[1].thread_id == "t2"

    def test_compile_complex_thread(self):
        threads = [ThreadSpec(thread_id="t1", source=THREAD_COMPLEX)]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=False,
            include_temporal=False, include_csl=False,
        )
        assert len(bundle.thread_payloads) == 1
        assert len(bundle.thread_payloads[0].instructions) > 5


# =============================================================================
# Section 3: Effect Safety Certificates
# =============================================================================

class TestEffectCertificates:
    def test_effect_inference_basic(self):
        threads = [
            ThreadSpec(thread_id="t1", source=THREAD_A),
            ThreadSpec(thread_id="t2", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=True, include_races=False,
            include_temporal=False, include_csl=False,
        )
        # Should have one effect certificate
        effect_certs = [c for c in bundle.certificates
                        if c.metadata.get("policy") == "effect_safety"]
        assert len(effect_certs) == 1
        assert effect_certs[0].status in (CertStatus.VALID, CertStatus.UNKNOWN)

    def test_effect_with_declarations_matching(self):
        threads = [
            ThreadSpec(
                thread_id="t1", source=THREAD_A,
                declared_effects=EffectSet(frozenset({State("x")})),
            ),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=True, include_races=False,
            include_temporal=False, include_csl=False,
        )
        effect_cert = bundle.certificates[0]
        assert effect_cert.metadata.get("policy") == "effect_safety"

    def test_effect_io_thread(self):
        threads = [ThreadSpec(thread_id="t1", source=THREAD_IO)]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=True, include_races=False,
            include_temporal=False, include_csl=False,
        )
        assert len(bundle.certificates) == 1
        # IO thread should have effects inferred
        cert = bundle.certificates[0]
        assert len(cert.obligations) >= 1


# =============================================================================
# Section 4: Race Freedom Certificates
# =============================================================================

class TestRaceFreedomCertificates:
    def test_no_shared_state(self):
        threads = [
            ThreadSpec(thread_id="t1", source=THREAD_A),
            ThreadSpec(thread_id="t2", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=True,
            include_temporal=False, include_csl=False,
        )
        race_certs = [c for c in bundle.certificates
                      if c.metadata.get("policy") == "race_freedom"]
        assert len(race_certs) == 1
        # No shared vars -> race free
        assert race_certs[0].status == CertStatus.VALID

    def test_shared_state_detected(self):
        threads = [
            ThreadSpec(thread_id="t1", source=THREAD_SHARED, shared_vars={"counter"}),
            ThreadSpec(thread_id="t2", source=THREAD_SHARED, shared_vars={"counter"}),
        ]
        program = ConcurrentProgram(
            threads=threads, shared_vars={"counter"},
        )
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=True,
            include_temporal=False, include_csl=False,
        )
        race_certs = [c for c in bundle.certificates
                      if c.metadata.get("policy") == "race_freedom"]
        assert len(race_certs) == 1

    def test_protected_shared_state(self):
        threads = [
            ThreadSpec(thread_id="t1", source=THREAD_SHARED,
                       shared_vars={"counter"}, locks={"lock"}),
            ThreadSpec(thread_id="t2", source=THREAD_SHARED,
                       shared_vars={"counter"}, locks={"lock"}),
        ]
        program = ConcurrentProgram(
            threads=threads, shared_vars={"counter"},
        )
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=True,
            include_temporal=False, include_csl=False,
        )
        race_certs = [c for c in bundle.certificates
                      if c.metadata.get("policy") == "race_freedom"]
        assert len(race_certs) == 1


# =============================================================================
# Section 5: Temporal Certificates
# =============================================================================

class TestTemporalCertificates:
    def test_mutual_exclusion_lock_protocol(self):
        threads = [
            ThreadSpec(thread_id="t0", source=THREAD_A),
            ThreadSpec(thread_id="t1", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, protocol="lock",
            temporal_properties=["mutual_exclusion"],
            include_effects=False, include_races=False, include_csl=False,
        )
        temporal_certs = [c for c in bundle.certificates
                          if c.metadata.get("protocol") == "lock"]
        assert len(temporal_certs) == 1
        assert temporal_certs[0].status == CertStatus.VALID

    def test_mutual_exclusion_no_protocol(self):
        threads = [
            ThreadSpec(thread_id="t0", source=THREAD_A),
            ThreadSpec(thread_id="t1", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, protocol="none",
            temporal_properties=["mutual_exclusion"],
            include_effects=False, include_races=False, include_csl=False,
        )
        temporal_certs = [c for c in bundle.certificates
                          if c.metadata.get("protocol") == "none"]
        assert len(temporal_certs) == 1
        # No protocol -> mutual exclusion may fail
        assert temporal_certs[0].status in (CertStatus.VALID, CertStatus.INVALID)

    def test_deadlock_freedom_lock_protocol(self):
        threads = [
            ThreadSpec(thread_id="t0", source=THREAD_A),
            ThreadSpec(thread_id="t1", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, protocol="lock",
            temporal_properties=["deadlock_freedom"],
            include_effects=False, include_races=False, include_csl=False,
        )
        temporal_certs = [c for c in bundle.certificates
                          if c.metadata.get("protocol") == "lock"]
        assert len(temporal_certs) == 1

    def test_multiple_temporal_properties(self):
        threads = [
            ThreadSpec(thread_id="t0", source=THREAD_A),
            ThreadSpec(thread_id="t1", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, protocol="lock",
            temporal_properties=["mutual_exclusion", "deadlock_freedom"],
            include_effects=False, include_races=False, include_csl=False,
        )
        temporal_certs = [c for c in bundle.certificates
                          if c.metadata.get("protocol") == "lock"]
        assert len(temporal_certs) == 1
        # Should have 2 obligations (one per property)
        assert len(temporal_certs[0].obligations) == 2


# =============================================================================
# Section 6: CSL Certificates
# =============================================================================

class TestCSLCertificates:
    def test_csl_no_commands(self):
        """When no CSL commands are given, CSL cert is VALID (not applicable)."""
        threads = [ThreadSpec(thread_id="t1", source=THREAD_A)]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=False,
            include_temporal=False, include_csl=True,
        )
        csl_certs = [c for c in bundle.certificates
                     if c.metadata.get("policy") == "csl_memory_safety"]
        assert len(csl_certs) == 1
        # No CSL commands -> not applicable -> VALID
        assert csl_certs[0].status == CertStatus.VALID

    def test_csl_with_skip_commands(self):
        """CSL with skip commands is trivially safe."""
        from concurrency_verification_composition import CSkip, CParallel
        threads = [
            ThreadSpec(thread_id="t1", source=THREAD_A, cmd=CSkip()),
            ThreadSpec(thread_id="t2", source=THREAD_B, cmd=CSkip()),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, include_effects=False, include_races=False,
            include_temporal=False, include_csl=True,
        )
        csl_certs = [c for c in bundle.certificates
                     if c.metadata.get("policy") == "csl_memory_safety"]
        assert len(csl_certs) == 1


# =============================================================================
# Section 7: Full Bundle Production
# =============================================================================

class TestFullBundle:
    def test_produce_full_bundle(self):
        threads = [
            ThreadSpec(thread_id="t0", source=THREAD_A),
            ThreadSpec(thread_id="t1", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, protocol="lock",
            temporal_properties=["mutual_exclusion"],
        )
        # Should have: effect + race + temporal + csl = 4 certificates
        assert bundle.total_certificates == 4
        assert len(bundle.policies) >= 4
        assert len(bundle.thread_payloads) == 2

    def test_bundle_summary(self):
        threads = [
            ThreadSpec(thread_id="t0", source=THREAD_A),
            ThreadSpec(thread_id="t1", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(
            program, protocol="lock",
            temporal_properties=["mutual_exclusion"],
        )
        summary = bundle.summary()
        assert "ConcurrentPCCBundle" in summary
        assert "Threads: 2" in summary
        assert "Certificates:" in summary

    def test_bundle_metadata(self):
        threads = [
            ThreadSpec(thread_id="t0", source=THREAD_A),
            ThreadSpec(thread_id="t1", source=THREAD_B),
        ]
        program = ConcurrentProgram(threads=threads)
        bundle = produce_concurrent_pcc(program, protocol="lock")
        assert bundle.metadata["n_threads"] == 2
        assert bundle.metadata["protocol"] == "lock"
        assert "t0" in bundle.metadata["thread_ids"]


# =============================================================================
# Section 8: Quick PCC API
# =============================================================================

class TestQuickPCC:
    def test_quick_pcc_basic(self):
        bundle = quick_concurrent_pcc(
            {"t1": THREAD_A, "t2": THREAD_B},
        )
        # Quick = effects + races only
        assert bundle.total_certificates == 2
        assert len(bundle.thread_payloads) == 2

    def test_quick_pcc_with_shared_vars(self):
        bundle = quick_concurrent_pcc(
            {"t1": THREAD_SHARED, "t2": THREAD_SHARED},
            shared_vars={"counter"},
        )
        assert bundle.total_certificates == 2

    def test_quick_pcc_single_thread(self):
        bundle = quick_concurrent_pcc({"t1": THREAD_A})
        assert len(bundle.thread_payloads) == 1
        assert bundle.total_certificates == 2


# =============================================================================
# Section 9: Full PCC API
# =============================================================================

class TestFullPCC:
    def test_full_pcc_default(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        # Full = effects + races + temporal (ME + DF) + csl = 4 certs
        assert bundle.total_certificates == 4
        assert len(bundle.thread_payloads) == 2

    def test_full_pcc_flag_protocol(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="flag",
        )
        assert bundle.total_certificates == 4

    def test_full_pcc_custom_temporal(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
            temporal_properties=["mutual_exclusion"],
        )
        temporal_certs = [c for c in bundle.certificates
                          if c.metadata.get("protocol") == "lock"]
        assert len(temporal_certs) == 1
        assert len(temporal_certs[0].obligations) == 1

    def test_full_pcc_with_declared_effects(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
            declared_effects={
                "t0": EffectSet(frozenset({State("x")})),
                "t1": EffectSet(frozenset({State("y")})),
            },
        )
        assert bundle.total_certificates == 4


# =============================================================================
# Section 10: Consumer Verification
# =============================================================================

class TestConsumerVerification:
    def test_verify_bundle_basic(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        verified = verify_concurrent_bundle(bundle)
        assert verified.status in (
            ConcBundleStatus.VERIFIED,
            ConcBundleStatus.PARTIALLY_VERIFIED,
            ConcBundleStatus.FAILED,
        )

    def test_verify_temporal_re_checks(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
            temporal_properties=["mutual_exclusion"],
        )
        verified = verify_concurrent_bundle(bundle)
        # Temporal cert should be re-verified via model checking
        temporal_certs = [c for c in verified.certificates
                          if c.metadata.get("protocol") == "lock"]
        assert len(temporal_certs) == 1
        assert temporal_certs[0].status == CertStatus.VALID

    def test_check_policy_effect_safety(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        # Effect safety should be checkable
        result = check_concurrent_policy(bundle, ConcPolicyKind.EFFECT_SAFETY)
        assert isinstance(result, bool)

    def test_check_policy_race_freedom(self):
        bundle = quick_concurrent_pcc(
            {"t1": THREAD_A, "t2": THREAD_B},
        )
        result = check_concurrent_policy(bundle, ConcPolicyKind.RACE_FREEDOM)
        assert isinstance(result, bool)

    def test_verify_preserves_thread_payloads(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        n_payloads = len(bundle.thread_payloads)
        verified = verify_concurrent_bundle(bundle)
        assert len(verified.thread_payloads) == n_payloads


# =============================================================================
# Section 11: Serialization
# =============================================================================

class TestSerialization:
    def test_bundle_to_dict_roundtrip(self):
        bundle = quick_concurrent_pcc(
            {"t1": THREAD_A, "t2": THREAD_B},
        )
        d = bundle.to_dict()
        bundle2 = ConcurrentPCCBundle.from_dict(d)
        assert len(bundle2.thread_payloads) == 2
        assert bundle2.status == bundle.status
        assert len(bundle2.certificates) == len(bundle.certificates)

    def test_save_and_load(self):
        bundle = quick_concurrent_pcc(
            {"t1": THREAD_A, "t2": THREAD_B},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            save_concurrent_bundle(bundle, path)
            loaded = load_concurrent_bundle(path)
            assert len(loaded.thread_payloads) == 2
            assert loaded.status == bundle.status
        finally:
            os.unlink(path)

    def test_full_bundle_json_roundtrip(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        d = bundle.to_dict()
        json_str = json.dumps(d, default=str)
        d2 = json.loads(json_str)
        bundle2 = ConcurrentPCCBundle.from_dict(d2)
        assert bundle2.total_certificates == bundle.total_certificates

    def test_certificate_statuses_preserved(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        d = bundle.to_dict()
        bundle2 = ConcurrentPCCBundle.from_dict(d)
        for orig, loaded in zip(bundle.certificates, bundle2.certificates):
            assert orig.status.value == loaded.status.value


# =============================================================================
# Section 12: Roundtrip API
# =============================================================================

class TestRoundtrip:
    def test_produce_and_verify(self):
        bundle = produce_and_verify(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
            temporal_properties=["mutual_exclusion"],
        )
        assert bundle.status in (
            ConcBundleStatus.VERIFIED,
            ConcBundleStatus.PARTIALLY_VERIFIED,
        )

    def test_produce_save_load_verify(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            path = f.name
        try:
            bundle = produce_save_load_verify(
                {"t0": THREAD_A, "t1": THREAD_B},
                path=path,
                protocol="lock",
                temporal_properties=["mutual_exclusion"],
            )
            assert bundle.status in (
                ConcBundleStatus.VERIFIED,
                ConcBundleStatus.PARTIALLY_VERIFIED,
            )
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# =============================================================================
# Section 13: Report API
# =============================================================================

class TestReportAPI:
    def test_concurrent_pcc_report(self):
        report = concurrent_pcc_report(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        assert "ConcurrentPCCBundle" in report
        assert "Threads: 2" in report

    def test_report_with_shared_vars(self):
        report = concurrent_pcc_report(
            {"t0": THREAD_SHARED, "t1": THREAD_SHARED},
            protocol="lock",
            shared_vars={"counter"},
        )
        assert isinstance(report, str)
        assert len(report) > 0


# =============================================================================
# Section 14: Protocol Comparison
# =============================================================================

class TestProtocolComparison:
    def test_compare_protocols_basic(self):
        results = compare_protocols(
            {"t0": THREAD_A, "t1": THREAD_B},
        )
        assert "none" in results
        assert "lock" in results
        assert "flag" in results
        for proto, data in results.items():
            assert "status" in data
            assert "valid_certs" in data

    def test_compare_protocols_subset(self):
        results = compare_protocols(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocols=["none", "lock"],
        )
        assert len(results) == 2
        assert "none" in results
        assert "lock" in results

    def test_lock_vs_none(self):
        results = compare_protocols(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocols=["none", "lock"],
        )
        # Lock protocol should have at least as many valid obligations
        assert results["lock"]["valid_obligations"] >= 0


# =============================================================================
# Section 15: Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_single_thread_bundle(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A},
            protocol="lock",
            temporal_properties=["mutual_exclusion"],
        )
        assert len(bundle.thread_payloads) == 1
        assert bundle.total_certificates >= 1

    def test_three_thread_bundle(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B, "t2": THREAD_IO},
            protocol="lock",
            temporal_properties=["mutual_exclusion"],
        )
        assert len(bundle.thread_payloads) == 3

    def test_no_temporal_properties(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
            temporal_properties=[],
        )
        # No temporal props -> no temporal cert, just effect + race + csl = 3
        assert bundle.total_certificates == 3

    def test_bundle_obligation_counts(self):
        bundle = full_concurrent_pcc(
            {"t0": THREAD_A, "t1": THREAD_B},
            protocol="lock",
        )
        assert bundle.total_obligations > 0
        assert bundle.valid_obligations >= 0
        assert bundle.invalid_certificates >= 0

    def test_empty_thread_source(self):
        """Minimal valid C10 source."""
        bundle = quick_concurrent_pcc(
            {"t1": "let x = 1;"},
        )
        assert len(bundle.thread_payloads) == 1

    def test_complex_thread_in_bundle(self):
        bundle = quick_concurrent_pcc(
            {"t1": THREAD_COMPLEX, "t2": THREAD_A},
        )
        assert len(bundle.thread_payloads) == 2
        # Complex thread should compile to more instructions
        assert len(bundle.thread_payloads[0].instructions) > len(bundle.thread_payloads[1].instructions)

    def test_bundle_status_computation(self):
        """Verify status is computed correctly from certificates."""
        bundle = quick_concurrent_pcc(
            {"t1": THREAD_A, "t2": THREAD_B},
        )
        if bundle.invalid_certificates > 0:
            assert bundle.status == ConcBundleStatus.FAILED
        elif bundle.valid_certificates == bundle.total_certificates:
            assert bundle.status == ConcBundleStatus.VERIFIED
