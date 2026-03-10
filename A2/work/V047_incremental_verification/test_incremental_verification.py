"""Tests for V047: Incremental Verification"""

import os
import sys
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from incremental_verification import (
    # Diff
    ChangeKind, FunctionChange, ProgramDiff,
    diff_programs, _parse_c10, _extract_functions, _fn_signature,
    _stmt_signature, _expr_sig,
    # Cache
    CertificateCache,
    # Incremental engine
    IncrementalResult, IncrementalVerifier,
    # Convenience
    incremental_verify, diff_and_report, verify_with_cache,
)

sys.path.insert(0, os.path.join(_dir, '..', 'V044_proof_certificates'))
from proof_certificates import ProofKind, CertStatus, ProofCertificate


# ===================================================================
# AST Diff Tests
# ===================================================================

class TestParsing:
    """Test C10 parsing and function extraction."""

    def test_parse_simple(self):
        stmts = _parse_c10("let x = 1;")
        assert len(stmts) == 1

    def test_extract_functions(self):
        source = """
        fn foo(x) {
            return x;
        }
        fn bar(y) {
            return y + 1;
        }
        """
        stmts = _parse_c10(source)
        fns = _extract_functions(stmts)
        assert "foo" in fns
        assert "bar" in fns
        assert len(fns) == 2

    def test_fn_signature_deterministic(self):
        source = """
        fn inc(x) {
            return x + 1;
        }
        """
        stmts = _parse_c10(source)
        fns = _extract_functions(stmts)
        sig1 = _fn_signature(fns["inc"])
        sig2 = _fn_signature(fns["inc"])
        assert sig1 == sig2

    def test_different_functions_different_sigs(self):
        source = """
        fn inc(x) {
            return x + 1;
        }
        fn dec(x) {
            return x - 1;
        }
        """
        stmts = _parse_c10(source)
        fns = _extract_functions(stmts)
        assert _fn_signature(fns["inc"]) != _fn_signature(fns["dec"])


class TestDiff:
    """Test program diffing."""

    def test_identical_programs(self):
        source = """
        fn foo(x) {
            return x;
        }
        """
        diff = diff_programs(source, source)
        assert not diff.has_changes
        assert len(diff.unchanged_functions) == 1
        assert "foo" in diff.unchanged_functions

    def test_modified_function(self):
        old = """
        fn foo(x) {
            return x;
        }
        """
        new = """
        fn foo(x) {
            return x + 1;
        }
        """
        diff = diff_programs(old, new)
        assert diff.has_changes
        assert "foo" in diff.changed_functions

    def test_added_function(self):
        old = """
        fn foo(x) {
            return x;
        }
        """
        new = """
        fn foo(x) {
            return x;
        }
        fn bar(y) {
            return y;
        }
        """
        diff = diff_programs(old, new)
        assert "bar" in diff.changed_functions
        changes = {fc.name: fc.kind for fc in diff.function_changes}
        assert changes["bar"] == ChangeKind.ADDED
        assert changes["foo"] == ChangeKind.UNCHANGED

    def test_removed_function(self):
        old = """
        fn foo(x) {
            return x;
        }
        fn bar(y) {
            return y;
        }
        """
        new = """
        fn foo(x) {
            return x;
        }
        """
        diff = diff_programs(old, new)
        changes = {fc.name: fc.kind for fc in diff.function_changes}
        assert changes["bar"] == ChangeKind.REMOVED
        assert changes["foo"] == ChangeKind.UNCHANGED

    def test_toplevel_change(self):
        old = "let x = 1;"
        new = "let x = 2;"
        diff = diff_programs(old, new)
        assert diff.toplevel_changed

    def test_toplevel_unchanged(self):
        old = "let x = 1;"
        new = "let x = 1;"
        diff = diff_programs(old, new)
        assert not diff.toplevel_changed

    def test_mixed_changes(self):
        old = """
        fn a(x) { return x; }
        fn b(x) { return x; }
        fn c(x) { return x; }
        """
        new = """
        fn a(x) { return x; }
        fn b(x) { return x + 1; }
        fn d(x) { return x; }
        """
        diff = diff_programs(old, new)
        changes = {fc.name: fc.kind for fc in diff.function_changes}
        assert changes["a"] == ChangeKind.UNCHANGED
        assert changes["b"] == ChangeKind.MODIFIED
        assert changes["c"] == ChangeKind.REMOVED
        assert changes["d"] == ChangeKind.ADDED

    def test_empty_programs(self):
        diff = diff_programs("let x = 1;", "let x = 1;")
        assert not diff.has_changes


# ===================================================================
# Certificate Cache Tests
# ===================================================================

class TestCertificateCache:
    """Test the certificate cache."""

    def test_empty_cache(self):
        cache = CertificateCache()
        assert cache.size == 0
        assert cache.get("foo") is None

    def test_put_and_get(self):
        cache = CertificateCache()
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test", status=CertStatus.VALID)
        cache.put("foo", "sig1", cert)
        assert cache.get("foo") is cert
        assert cache.size == 1

    def test_has_valid(self):
        cache = CertificateCache()
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test", status=CertStatus.VALID)
        cache.put("foo", "sig1", cert)
        assert cache.has_valid("foo", "sig1")
        assert not cache.has_valid("foo", "sig2")  # different signature
        assert not cache.has_valid("bar", "sig1")  # different function

    def test_has_valid_invalid_cert(self):
        cache = CertificateCache()
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test", status=CertStatus.INVALID)
        cache.put("foo", "sig1", cert)
        assert not cache.has_valid("foo", "sig1")

    def test_invalidate(self):
        cache = CertificateCache()
        cert = ProofCertificate(kind=ProofKind.VCGEN, claim="test", status=CertStatus.VALID)
        cache.put("foo", "sig1", cert)
        cache.invalidate("foo")
        assert cache.get("foo") is None
        assert cache.size == 0

    def test_valid_count(self):
        cache = CertificateCache()
        cert1 = ProofCertificate(kind=ProofKind.VCGEN, claim="t1", status=CertStatus.VALID)
        cert2 = ProofCertificate(kind=ProofKind.VCGEN, claim="t2", status=CertStatus.INVALID)
        cache.put("foo", "s1", cert1)
        cache.put("bar", "s2", cert2)
        assert cache.valid_count == 1


# ===================================================================
# Incremental Verifier Tests
# ===================================================================

class TestIncrementalVerifier:
    """Test the incremental verification engine."""

    def test_first_verification_full(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        """
        verifier = IncrementalVerifier()
        result = verifier.verify(source)
        assert "inc" in result.functions_reverified
        assert result.cache_misses == 1
        assert result.cache_hits == 0

    def test_second_verification_no_change(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        """
        verifier = IncrementalVerifier()
        result1 = verifier.verify(source)
        result2 = verifier.verify(source)
        # Second time should reuse cache
        assert "inc" in result2.functions_reused
        assert result2.cache_hits == 1
        assert result2.cache_misses == 0

    def test_modified_function_reverified(self):
        old = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        new = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        """
        verifier = IncrementalVerifier()
        verifier.verify(old)
        result = verifier.verify(new)
        assert "foo" in result.functions_reverified
        assert result.cache_misses == 1

    def test_unchanged_function_reused(self):
        old = """
        fn stable(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        fn changing(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        new = """
        fn stable(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        fn changing(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        """
        verifier = IncrementalVerifier()
        verifier.verify(old)
        result = verifier.verify(new)
        assert "stable" in result.functions_reused
        assert "changing" in result.functions_reverified

    def test_added_function_verified(self):
        old = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        new = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        fn bar(y) {
            requires(y >= 0);
            ensures(result >= 0);
            return y;
        }
        """
        verifier = IncrementalVerifier()
        verifier.verify(old)
        result = verifier.verify(new)
        assert "foo" in result.functions_reused
        assert "bar" in result.functions_reverified

    def test_removed_function_skipped(self):
        old = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        fn bar(y) {
            requires(y >= 0);
            ensures(result >= 0);
            return y;
        }
        """
        new = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        verifier = IncrementalVerifier()
        verifier.verify(old)
        result = verifier.verify(new)
        assert "bar" in result.functions_skipped

    def test_summary_format(self):
        source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        verifier = IncrementalVerifier()
        result = verifier.verify(source)
        assert "Status:" in result.summary
        assert "Functions:" in result.summary
        assert "Cache:" in result.summary

    def test_no_functions(self):
        source = "let x = 1;"
        verifier = IncrementalVerifier()
        result = verifier.verify(source)
        assert result.total_functions == 0
        assert result.is_valid


# ===================================================================
# Convenience API Tests
# ===================================================================

class TestConvenienceAPIs:
    """Test high-level convenience functions."""

    def test_incremental_verify(self):
        old = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        new = """
        fn foo(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        """
        result = incremental_verify(old, new)
        assert "foo" in result.functions_reverified

    def test_diff_and_report(self):
        old = """
        fn a(x) { return x; }
        fn b(x) { return x; }
        """
        new = """
        fn a(x) { return x; }
        fn b(x) { return x + 1; }
        """
        report = diff_and_report(old, new)
        assert "a" in report
        assert "b" in report
        assert "unchanged" in report.lower() or "modified" in report.lower()

    def test_verify_with_cache_sequence(self):
        v1 = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        v2 = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        fn g(y) {
            requires(y >= 0);
            ensures(result >= 0);
            return y;
        }
        """
        v3 = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        fn g(y) {
            requires(y >= 0);
            ensures(result >= 0);
            return y;
        }
        """
        results = verify_with_cache([v1, v2, v3])
        assert len(results) == 3
        # v1: full verification
        assert results[0].cache_misses == 1
        # v2: f reused, g new
        assert "f" in results[1].functions_reused
        assert "g" in results[1].functions_reverified
        # v3: f modified, g reused
        assert "f" in results[2].functions_reverified
        assert "g" in results[2].functions_reused


# ===================================================================
# Statement Signature Tests
# ===================================================================

class TestSignatures:
    """Test statement/expression signature generation."""

    def test_let_signature(self):
        stmts = _parse_c10("let x = 5;")
        sig = _stmt_signature(stmts[0])
        assert "let" in sig
        assert "x" in sig
        assert "5" in sig

    def test_assign_signature(self):
        stmts = _parse_c10("let x = 1; x = 5;")
        sig = _stmt_signature(stmts[1])
        assert "assign" in sig

    def test_different_values_different_sigs(self):
        stmts1 = _parse_c10("let x = 1;")
        stmts2 = _parse_c10("let x = 2;")
        assert _stmt_signature(stmts1[0]) != _stmt_signature(stmts2[0])

    def test_expr_sig_binop(self):
        from stack_vm import BinOp as BinOpNode, IntLit as IntLitNode
        sig = _expr_sig(BinOpNode("+", IntLitNode(1, 0), IntLitNode(2, 0), 0))
        assert "+" in sig
        assert "1" in sig
        assert "2" in sig


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_function_with_no_annotations(self):
        """Functions without requires/ensures still get certificates."""
        source = """
        fn plain(x) {
            return x;
        }
        """
        verifier = IncrementalVerifier()
        result = verifier.verify(source)
        # Should complete without error
        assert result.total_functions == 1

    def test_multiple_verifications_accumulate_cache(self):
        verifier = IncrementalVerifier()
        v1 = "fn a(x) { return x; }"
        v2 = "fn a(x) { return x; }\nfn b(x) { return x; }"
        v3 = "fn a(x) { return x; }\nfn b(x) { return x; }\nfn c(x) { return x; }"
        verifier.verify(v1)
        verifier.verify(v2)
        verifier.verify(v3)
        assert verifier.cache.size >= 3

    def test_verify_against(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x + 1; }"
        verifier = IncrementalVerifier()
        result = verifier.verify_against(old, new)
        assert "f" in result.functions_reverified

    def test_program_diff_properties(self):
        old = """
        fn a(x) { return x; }
        fn b(x) { return x; }
        """
        new = """
        fn a(x) { return x; }
        fn c(x) { return x; }
        """
        diff = diff_programs(old, new)
        assert "a" in diff.old_functions
        assert "b" in diff.old_functions
        assert "a" in diff.new_functions
        assert "c" in diff.new_functions
        assert "b" not in diff.new_functions
        assert "c" not in diff.old_functions
