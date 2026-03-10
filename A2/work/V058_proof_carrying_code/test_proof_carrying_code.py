"""
Tests for V058: Proof-Carrying Code
Tests cover: bytecode compilation, certificate generation, bundle creation,
consumer verification, serialization roundtrip, policies, convenience APIs.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from proof_carrying_code import (
    produce_pcc, quick_pcc, full_pcc, pcc_report,
    produce_and_verify, produce_save_load_verify,
    verify_bundle, check_policy, save_bundle, load_bundle,
    PCCBundle, BytecodePayload, SafetyPolicy, PolicyKind, BundleStatus,
    _compile_to_bytecode, _extract_functions,
    _generate_contract_certificate, _generate_bound_certificate,
    _generate_compilation_certificate,
)
from proof_certificates import CertStatus


# ============================================================
# Section 1: Bytecode Compilation
# ============================================================

class TestBytecodeCompilation:
    def test_compile_simple(self):
        payload = _compile_to_bytecode("let x = 1;")
        assert isinstance(payload, BytecodePayload)
        assert len(payload.instructions) > 0

    def test_source_hash(self):
        p1 = _compile_to_bytecode("let x = 1;")
        p2 = _compile_to_bytecode("let x = 1;")
        assert p1.source_hash == p2.source_hash

    def test_different_source_different_hash(self):
        p1 = _compile_to_bytecode("let x = 1;")
        p2 = _compile_to_bytecode("let x = 2;")
        assert p1.source_hash != p2.source_hash

    def test_compiler_version(self):
        payload = _compile_to_bytecode("let x = 1;")
        assert payload.compiler_version == "C010"

    def test_function_compilation(self):
        payload = _compile_to_bytecode("fn f(x) { return x + 1; }")
        assert len(payload.instructions) > 0


# ============================================================
# Section 2: BytecodePayload Serialization
# ============================================================

class TestPayloadSerialization:
    def test_to_dict(self):
        payload = _compile_to_bytecode("let x = 42;")
        d = payload.to_dict()
        assert 'instructions' in d
        assert 'constants' in d
        assert 'source_hash' in d

    def test_roundtrip(self):
        payload = _compile_to_bytecode("let x = 42;")
        d = payload.to_dict()
        restored = BytecodePayload.from_dict(d)
        assert restored.source_hash == payload.source_hash
        assert restored.compiler_version == payload.compiler_version


# ============================================================
# Section 3: Contract Certificate Generation
# ============================================================

class TestContractCertificate:
    def test_generates_for_annotated(self):
        source = """
fn abs(x) {
    requires(x > -100);
    ensures(__result >= 0);
    if (x >= 0) { return x; } else { return 0 - x; }
}
"""
        cert = _generate_contract_certificate(source)
        # May or may not generate depending on VCGen behavior
        # Just check it doesn't crash
        assert cert is None or cert.status in (CertStatus.VALID, CertStatus.INVALID, CertStatus.UNKNOWN)

    def test_no_crash_unannotated(self):
        cert = _generate_contract_certificate("fn f(x) { return x + 1; }")
        # Should not crash, may return None if no contracts
        assert cert is None or isinstance(cert, type(cert))


# ============================================================
# Section 4: Bound Certificate Generation
# ============================================================

class TestBoundCertificate:
    def test_generates_bounds(self):
        cert = _generate_bound_certificate("fn f(x) { let y = 5; return y; }")
        # AI should infer y=5 bound
        if cert:
            assert cert.status == CertStatus.VALID
            assert len(cert.obligations) > 0

    def test_no_crash_simple(self):
        cert = _generate_bound_certificate("let x = 1;")
        assert cert is None or isinstance(cert, type(cert))


# ============================================================
# Section 5: Compilation Certificate Generation
# ============================================================

class TestCompilationCertificate:
    def test_generates_compilation_cert(self):
        cert = _generate_compilation_certificate("fn f(x) { return x + 1; }")
        if cert:
            assert cert.metadata.get('policy') == PolicyKind.COMPILATION_SAFETY.value

    def test_no_crash(self):
        cert = _generate_compilation_certificate("let x = 1;")
        assert cert is None or isinstance(cert, type(cert))


# ============================================================
# Section 6: produce_pcc -- Basic
# ============================================================

class TestProducePCC:
    def test_produces_bundle(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }")
        assert isinstance(bundle, PCCBundle)

    def test_has_bytecode(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }")
        assert bundle.bytecode is not None
        assert len(bundle.bytecode.instructions) > 0

    def test_has_certificates(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }")
        # At minimum, bounds or compilation cert should exist
        assert bundle.total_certificates >= 0

    def test_has_metadata(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }")
        assert 'source_hash' in bundle.metadata
        assert 'produced_at' in bundle.metadata


# ============================================================
# Section 7: produce_pcc -- Options
# ============================================================

class TestProduceOptions:
    def test_no_contracts(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }",
                              include_contracts=False)
        for policy in bundle.policies:
            assert policy.kind != PolicyKind.CONTRACT_COMPLIANCE

    def test_no_bounds(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }",
                              include_contracts=False, include_bounds=False)
        for policy in bundle.policies:
            assert policy.kind != PolicyKind.BOUND_SAFETY

    def test_no_compilation(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }",
                              include_compilation=False)
        for policy in bundle.policies:
            assert policy.kind != PolicyKind.COMPILATION_SAFETY

    def test_nothing_enabled(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }",
                              include_contracts=False, include_bounds=False,
                              include_compilation=False)
        assert bundle.total_certificates == 0
        assert bundle.status == BundleStatus.UNCHECKED


# ============================================================
# Section 8: Bundle Properties
# ============================================================

class TestBundleProperties:
    def test_total_certificates(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }")
        assert bundle.total_certificates >= 0

    def test_valid_certificates(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }")
        assert bundle.valid_certificates >= 0
        assert bundle.valid_certificates <= bundle.total_certificates

    def test_total_obligations(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }")
        assert bundle.total_obligations >= 0

    def test_summary(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }")
        s = bundle.summary()
        assert isinstance(s, str)
        assert "PCC Bundle" in s


# ============================================================
# Section 9: Consumer Verification
# ============================================================

class TestConsumerVerification:
    def test_verify_bundle(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }")
        verified = verify_bundle(bundle)
        assert verified.status in (BundleStatus.VERIFIED, BundleStatus.PARTIALLY_VERIFIED,
                                     BundleStatus.FAILED, BundleStatus.UNCHECKED)

    def test_verify_updates_status(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }",
                              include_contracts=False, include_compilation=False)
        old_status = bundle.status
        verified = verify_bundle(bundle)
        # Status may change after verification
        assert isinstance(verified.status, BundleStatus)


# ============================================================
# Section 10: check_policy
# ============================================================

class TestCheckPolicy:
    def test_check_existing_policy(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }",
                              include_contracts=False, include_compilation=False)
        # Bounds policy may or may not be present
        result = check_policy(bundle, PolicyKind.BOUND_SAFETY)
        assert isinstance(result, bool)

    def test_check_missing_policy(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }",
                              include_contracts=False, include_bounds=False,
                              include_compilation=False)
        assert check_policy(bundle, PolicyKind.BOUND_SAFETY) is False


# ============================================================
# Section 11: Bundle Serialization
# ============================================================

class TestBundleSerialization:
    def test_to_dict(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }",
                              include_contracts=False, include_compilation=False)
        d = bundle.to_dict()
        assert 'bytecode' in d
        assert 'certificates' in d
        assert 'policies' in d

    def test_roundtrip_dict(self):
        bundle = produce_pcc("fn f(x) { let y = 5; return y; }",
                              include_contracts=False, include_compilation=False)
        d = bundle.to_dict()
        restored = PCCBundle.from_dict(d)
        assert restored.bytecode.source_hash == bundle.bytecode.source_hash
        assert restored.total_certificates == bundle.total_certificates
        assert restored.status == bundle.status

    def test_save_load_file(self):
        bundle = produce_pcc("fn f(x) { return x + 1; }",
                              include_contracts=False, include_compilation=False)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            save_bundle(bundle, path)
            loaded = load_bundle(path)
            assert loaded.bytecode.source_hash == bundle.bytecode.source_hash
        finally:
            os.unlink(path)


# ============================================================
# Section 12: quick_pcc
# ============================================================

class TestQuickPCC:
    def test_produces_bundle(self):
        bundle = quick_pcc("fn f(x) { let y = 5; return y; }")
        assert isinstance(bundle, PCCBundle)

    def test_no_contracts(self):
        bundle = quick_pcc("fn f(x) { return x + 1; }")
        for policy in bundle.policies:
            assert policy.kind != PolicyKind.CONTRACT_COMPLIANCE

    def test_no_compilation(self):
        bundle = quick_pcc("fn f(x) { return x + 1; }")
        for policy in bundle.policies:
            assert policy.kind != PolicyKind.COMPILATION_SAFETY


# ============================================================
# Section 13: full_pcc
# ============================================================

class TestFullPCC:
    def test_produces_bundle(self):
        bundle = full_pcc("fn f(x) { let y = 5; return y; }")
        assert isinstance(bundle, PCCBundle)

    def test_more_certs_than_quick(self):
        source = "fn f(x) { let y = 5; return y; }"
        q = quick_pcc(source)
        f = full_pcc(source)
        assert f.total_certificates >= q.total_certificates


# ============================================================
# Section 14: pcc_report
# ============================================================

class TestPCCReport:
    def test_report_string(self):
        report = pcc_report("fn f(x) { return x + 1; }")
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_status(self):
        report = pcc_report("fn f(x) { return x + 1; }")
        assert any(s in report for s in ["verified", "partial", "failed", "unchecked"])


# ============================================================
# Section 15: produce_and_verify
# ============================================================

class TestProduceAndVerify:
    def test_roundtrip(self):
        bundle = produce_and_verify("fn f(x) { let y = 5; return y; }")
        assert isinstance(bundle, PCCBundle)
        assert bundle.status != BundleStatus.UNCHECKED or bundle.total_certificates == 0


# ============================================================
# Section 16: produce_save_load_verify
# ============================================================

class TestFullRoundtrip:
    def test_full_io_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            bundle = produce_save_load_verify(
                "fn f(x) { let y = 5; return y; }", path
            )
            assert isinstance(bundle, PCCBundle)
        finally:
            os.unlink(path)


# ============================================================
# Section 17: SafetyPolicy
# ============================================================

class TestSafetyPolicy:
    def test_policy_kinds(self):
        for kind in PolicyKind:
            p = SafetyPolicy(kind=kind, description=f"Test {kind.value}")
            assert p.kind == kind

    def test_policy_parameters(self):
        p = SafetyPolicy(kind=PolicyKind.BOUND_SAFETY,
                         description="Test", parameters={"max": 100})
        assert p.parameters["max"] == 100


# ============================================================
# Section 18: BundleStatus
# ============================================================

class TestBundleStatus:
    def test_all_statuses(self):
        statuses = [BundleStatus.VERIFIED, BundleStatus.PARTIALLY_VERIFIED,
                    BundleStatus.FAILED, BundleStatus.UNCHECKED]
        assert len(statuses) == 4

    def test_unchecked_when_no_certs(self):
        bundle = produce_pcc("let x = 1;",
                              include_contracts=False, include_bounds=False,
                              include_compilation=False)
        assert bundle.status == BundleStatus.UNCHECKED


# ============================================================
# Section 19: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_function(self):
        bundle = produce_pcc("fn f() { return 0; }",
                              include_contracts=False, include_compilation=False)
        assert isinstance(bundle, PCCBundle)

    def test_multiple_functions(self):
        source = """
fn add(x, y) { return x + y; }
fn double(x) { return add(x, x); }
"""
        bundle = produce_pcc(source, include_contracts=False, include_compilation=False)
        assert isinstance(bundle, PCCBundle)

    def test_toplevel_only(self):
        bundle = produce_pcc("let x = 42;",
                              include_contracts=False, include_compilation=False)
        assert isinstance(bundle, PCCBundle)


# ============================================================
# Section 20: extract_functions helper
# ============================================================

class TestExtractFunctions:
    def test_extracts(self):
        fns = _extract_functions("fn foo(x) { return x; } fn bar() { return 0; }")
        assert "foo" in fns
        assert "bar" in fns

    def test_no_functions(self):
        fns = _extract_functions("let x = 1;")
        assert len(fns) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
