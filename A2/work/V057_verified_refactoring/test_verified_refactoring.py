"""
Tests for V057: Verified Refactoring
Tests cover: refactoring detection, equivalence checking, summary comparison,
contract preservation, certificate generation, convenience APIs, edge cases.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from verified_refactoring import (
    verify_refactoring, verify_extract_refactoring, verify_inline_refactoring,
    verify_rename_refactoring, verify_simplification, compare_refactoring_strategies,
    refactoring_report,
    RefactoringKind, RefactoringResult, FunctionVerificationResult,
    _parse, _extract_functions, _collect_variables, _stmt_signature,
    _detect_refactorings, _compare_summaries, _collect_calls_in_fn,
    _generate_refactoring_certificate,
)
from incremental_verification import diff_programs


# ============================================================
# Section 1: AST Helpers
# ============================================================

class TestASTHelpers:
    def test_parse_simple(self):
        stmts = _parse("let x = 1;")
        assert len(stmts) == 1

    def test_extract_functions(self):
        stmts = _parse("fn foo(x) { return x + 1; } fn bar(y) { return y * 2; }")
        fns = _extract_functions(stmts)
        assert "foo" in fns
        assert "bar" in fns
        assert len(fns) == 2

    def test_collect_variables(self):
        stmts = _parse("fn f(x) { let y = x + 1; return y; }")
        fns = _extract_functions(stmts)
        variables = _collect_variables(fns["f"])
        assert "x" in variables
        assert "y" in variables

    def test_stmt_signature_stability(self):
        stmts = _parse("fn f(x) { return x + 1; }")
        fns = _extract_functions(stmts)
        sig1 = _stmt_signature(fns["f"])
        # Same source should give same signature
        stmts2 = _parse("fn f(x) { return x + 1; }")
        fns2 = _extract_functions(stmts2)
        sig2 = _stmt_signature(fns2["f"])
        assert sig1 == sig2

    def test_stmt_signature_different(self):
        fns1 = _extract_functions(_parse("fn f(x) { return x + 1; }"))
        fns2 = _extract_functions(_parse("fn f(x) { return x + 2; }"))
        assert _stmt_signature(fns1["f"]) != _stmt_signature(fns2["f"])


# ============================================================
# Section 2: Refactoring Detection -- Identity (No Change)
# ============================================================

class TestNoChange:
    def test_identical_programs(self):
        src = "fn f(x) { return x + 1; }"
        result = verify_refactoring(src, src)
        assert result.is_verified
        assert result.functions_verified == 0
        assert result.functions_failed == 0

    def test_no_refactorings_detected(self):
        src = "fn f(x) { return x + 1; }"
        result = verify_refactoring(src, src)
        assert len(result.refactorings) == 0


# ============================================================
# Section 3: Refactoring Detection -- Rename Function
# ============================================================

class TestRenameDetection:
    def test_detect_rename(self):
        old = "fn foo(x) { return x + 1; }"
        new = "fn bar(x) { return x + 1; }"
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        kinds = {r.kind for r in refs}
        assert RefactoringKind.RENAME_FUNCTION in kinds

    def test_rename_old_new_names(self):
        old = "fn foo(x) { return x + 1; }"
        new = "fn bar(x) { return x + 1; }"
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        rename = [r for r in refs if r.kind == RefactoringKind.RENAME_FUNCTION][0]
        assert rename.old_name == "foo"
        assert rename.new_name == "bar"


# ============================================================
# Section 4: Refactoring Detection -- Extract Function
# ============================================================

class TestExtractDetection:
    def test_detect_extract(self):
        old = """
fn compute(x) {
    let y = x * 2;
    let z = y + 1;
    return z;
}
"""
        new = """
fn double(x) {
    return x * 2;
}
fn compute(x) {
    let y = double(x);
    let z = y + 1;
    return z;
}
"""
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        kinds = {r.kind for r in refs}
        assert RefactoringKind.EXTRACT_FUNCTION in kinds

    def test_extract_affected_functions(self):
        old = "fn f(x) { let y = x * 2; return y + 1; }"
        new = "fn helper(x) { return x * 2; } fn f(x) { let y = helper(x); return y + 1; }"
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        extract = [r for r in refs if r.kind == RefactoringKind.EXTRACT_FUNCTION]
        if extract:
            assert "helper" in extract[0].affected_functions
            assert "f" in extract[0].affected_functions


# ============================================================
# Section 5: Refactoring Detection -- Inline Function
# ============================================================

class TestInlineDetection:
    def test_detect_inline(self):
        old = "fn helper(x) { return x * 2; } fn f(x) { return helper(x) + 1; }"
        new = "fn f(x) { return x * 2 + 1; }"
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        kinds = {r.kind for r in refs}
        assert RefactoringKind.INLINE_FUNCTION in kinds


# ============================================================
# Section 6: Refactoring Detection -- Add/Remove/Modify
# ============================================================

class TestBasicDetection:
    def test_detect_add(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; } fn g(x) { return x + 1; }"
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        kinds = {r.kind for r in refs}
        assert RefactoringKind.ADD_FUNCTION in kinds

    def test_detect_remove(self):
        old = "fn f(x) { return x; } fn g(x) { return x + 1; }"
        new = "fn f(x) { return x; }"
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        kinds = {r.kind for r in refs}
        assert RefactoringKind.REMOVE_FUNCTION in kinds

    def test_detect_modify(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        diff = diff_programs(old, new)
        refs = _detect_refactorings(diff, old, new)
        kinds = {r.kind for r in refs}
        assert RefactoringKind.MODIFY_FUNCTION in kinds


# ============================================================
# Section 7: Equivalence Verification -- Preserving
# ============================================================

class TestEquivalencePreserving:
    def test_identity_refactoring(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        result = verify_refactoring(old, new, check_summaries=False)
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True

    def test_algebraic_simplification(self):
        old = "fn f(x) { return x + 0; }"
        new = "fn f(x) { return x; }"
        result = verify_refactoring(old, new, check_summaries=False)
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True

    def test_commutative_reorder(self):
        old = "fn f(x, y) { return x + y; }"
        new = "fn f(x, y) { return y + x; }"
        result = verify_refactoring(old, new, check_summaries=False,
                                     param_types={"f": {"x": "int", "y": "int"}})
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True


# ============================================================
# Section 8: Equivalence Verification -- Breaking
# ============================================================

class TestEquivalenceBreaking:
    def test_behavior_change_detected(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new, check_summaries=False)
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is False

    def test_counterexample_provided(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new, check_summaries=False)
        fr = result.function_results.get("f")
        assert fr.counterexample is not None

    def test_overall_not_verified(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x * 2; }"
        result = verify_refactoring(old, new, check_summaries=False)
        assert not result.is_verified


# ============================================================
# Section 9: Summary Comparison
# ============================================================

class TestSummaryComparison:
    def test_summary_compatible_same_behavior(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        result = verify_refactoring(old, new, check_equivalence=False, check_summaries=True)
        fr = result.function_results.get("f")
        if fr and fr.summary_compatible is not None:
            assert fr.summary_compatible is True

    def test_both_summaries_present(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new, check_equivalence=False, check_summaries=True)
        assert result.old_summaries is not None
        assert result.new_summaries is not None

    def test_compare_none_summaries(self):
        compatible, notes = _compare_summaries(None, None)
        assert compatible is True

    def test_compare_one_none(self):
        compatible, notes = _compare_summaries(None, "something")
        assert compatible is True  # Can't compare, not a failure


# ============================================================
# Section 10: Certificate Generation
# ============================================================

class TestCertificates:
    def test_certificate_generated(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        result = verify_refactoring(old, new)
        assert result.certificate is not None

    def test_certificate_valid_for_preserving(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        result = verify_refactoring(old, new, check_summaries=False)
        from proof_certificates import CertStatus
        assert result.certificate.status == CertStatus.VALID

    def test_certificate_invalid_for_breaking(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new, check_summaries=False)
        from proof_certificates import CertStatus
        assert result.certificate.status == CertStatus.INVALID

    def test_certificate_has_obligations(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new, check_summaries=False)
        assert len(result.certificate.obligations) > 0

    def test_certificate_metadata(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new)
        assert 'refactoring_kinds' in result.certificate.metadata


# ============================================================
# Section 11: Convenience API -- verify_rename_refactoring
# ============================================================

class TestRenameAPI:
    def test_rename_verified(self):
        old = "fn foo(x) { return x + 1; }"
        new = "fn bar(x) { return x + 1; }"
        result = verify_rename_refactoring(old, new, "foo", "bar",
                                            param_types={"x": "int"})
        fr = result.function_results.get("bar")
        assert fr is not None
        assert fr.equivalence_verified is True

    def test_rename_with_body_change_fails(self):
        old = "fn foo(x) { return x + 1; }"
        new = "fn bar(x) { return x + 2; }"
        result = verify_rename_refactoring(old, new, "foo", "bar",
                                            param_types={"x": "int"})
        fr = result.function_results.get("bar")
        assert fr is not None
        assert fr.equivalence_verified is False


# ============================================================
# Section 12: Convenience API -- verify_simplification
# ============================================================

class TestSimplificationAPI:
    def test_simplify_preserves(self):
        old = "fn f(x) { let y = x + 0; return y; }"
        new = "fn f(x) { return x; }"
        result = verify_simplification(old, new, "f", param_types={"x": "int"})
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True

    def test_simplify_breaks(self):
        old = "fn f(x) { return x * 2; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_simplification(old, new, "f", param_types={"x": "int"})
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is False


# ============================================================
# Section 13: Convenience API -- verify_extract_refactoring
# ============================================================

class TestExtractAPI:
    def test_extract_preserves_behavior(self):
        old = "fn compute(x) { let y = x * 2; return y + 1; }"
        new = "fn double(x) { return x * 2; } fn compute(x) { let y = double(x); return y + 1; }"
        result = verify_extract_refactoring(old, new, "double", "compute",
                                             param_types={"x": "int"})
        # compute should still produce same results
        fr = result.function_results.get("compute")
        assert fr is not None
        assert fr.equivalence_verified is True


# ============================================================
# Section 14: Convenience API -- verify_inline_refactoring
# ============================================================

class TestInlineAPI:
    def test_inline_preserves_behavior(self):
        old = "fn double(x) { return x * 2; } fn f(x) { return double(x) + 1; }"
        new = "fn f(x) { return x * 2 + 1; }"
        result = verify_inline_refactoring(old, new, "double", "f",
                                            param_types={"x": "int"})
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True


# ============================================================
# Section 15: compare_refactoring_strategies
# ============================================================

class TestCompareStrategies:
    def test_compare_returns_all_strategies(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        comp = compare_refactoring_strategies(old, new)
        assert 'equivalence_only' in comp
        assert 'summary_only' in comp
        assert 'combined' in comp
        assert 'refactorings' in comp

    def test_compare_preserving_all_verified(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        comp = compare_refactoring_strategies(old, new)
        assert comp['equivalence_only']['verified'] is True
        assert comp['combined']['verified'] is True


# ============================================================
# Section 16: refactoring_report
# ============================================================

class TestReport:
    def test_report_contains_status(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        report = refactoring_report(old, new)
        assert "VERIFIED" in report or "FAILED" in report

    def test_report_shows_function_names(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        report = refactoring_report(old, new)
        assert "f" in report


# ============================================================
# Section 17: Multi-function Programs
# ============================================================

class TestMultiFunction:
    def test_only_changed_functions_verified(self):
        old = "fn f(x) { return x + 1; } fn g(x) { return x * 2; }"
        new = "fn f(x) { return x + 1; } fn g(x) { return x * 3; }"
        result = verify_refactoring(old, new, check_summaries=False)
        # f is unchanged, only g should be checked
        assert "g" in result.function_results
        # f should not be in results (unchanged)
        assert "f" not in result.function_results

    def test_multi_function_mixed_results(self):
        old = "fn f(x) { return x + 1; } fn g(x) { return x * 2; }"
        new = "fn f(x) { let y = x + 1; return y; } fn g(x) { return x * 3; }"
        result = verify_refactoring(old, new, check_summaries=False)
        # f should be verified (equivalent), g should fail
        if "f" in result.function_results:
            assert result.function_results["f"].equivalence_verified is True
        assert result.function_results["g"].equivalence_verified is False


# ============================================================
# Section 18: Conditional Refactoring
# ============================================================

class TestConditionalRefactoring:
    def test_if_restructure_preserves(self):
        old = """
fn f(x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}
"""
        new = """
fn f(x) {
    let r = 0;
    if (x > 0) {
        r = 1;
    }
    return r;
}
"""
        result = verify_refactoring(old, new, check_summaries=False)
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True


# ============================================================
# Section 19: RefactoringResult Properties
# ============================================================

class TestResultProperties:
    def test_functions_verified_count(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        result = verify_refactoring(old, new, check_summaries=False)
        assert result.functions_verified >= 0

    def test_refactoring_kinds_set(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new)
        kinds = result.refactoring_kinds
        assert isinstance(kinds, set)

    def test_summary_method(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_is_verified_no_changes(self):
        src = "fn f(x) { return x + 1; }"
        result = verify_refactoring(src, src)
        assert result.is_verified is True


# ============================================================
# Section 20: FunctionVerificationResult
# ============================================================

class TestFunctionVerificationResult:
    def test_is_verified_all_true(self):
        fr = FunctionVerificationResult(fn_name="f",
                                         equivalence_verified=True,
                                         summary_compatible=True)
        assert fr.is_verified is True

    def test_is_verified_one_false(self):
        fr = FunctionVerificationResult(fn_name="f",
                                         equivalence_verified=True,
                                         summary_compatible=False)
        assert fr.is_verified is False

    def test_is_verified_all_none(self):
        fr = FunctionVerificationResult(fn_name="f")
        assert fr.is_verified is False  # No checks performed

    def test_notes_accumulated(self):
        fr = FunctionVerificationResult(fn_name="f")
        fr.notes.append("test note")
        assert len(fr.notes) == 1


# ============================================================
# Section 21: Collect Calls Helper
# ============================================================

class TestCollectCalls:
    def test_simple_call(self):
        stmts = _parse("fn f(x) { return g(x); }")
        fns = _extract_functions(stmts)
        calls = _collect_calls_in_fn(fns["f"])
        assert "g" in calls

    def test_no_calls(self):
        stmts = _parse("fn f(x) { return x + 1; }")
        fns = _extract_functions(stmts)
        calls = _collect_calls_in_fn(fns["f"])
        assert len(calls) == 0

    def test_ignores_builtins(self):
        stmts = _parse("fn f(x) { print(x); return x; }")
        fns = _extract_functions(stmts)
        calls = _collect_calls_in_fn(fns["f"])
        assert "print" not in calls

    def test_nested_call(self):
        stmts = _parse("fn f(x) { let y = g(h(x)); return y; }")
        fns = _extract_functions(stmts)
        calls = _collect_calls_in_fn(fns["f"])
        assert "g" in calls
        assert "h" in calls


# ============================================================
# Section 22: Strength Reduction
# ============================================================

class TestStrengthReduction:
    def test_mul_to_add(self):
        old = "fn f(x) { return x * 2; }"
        new = "fn f(x) { return x + x; }"
        result = verify_refactoring(old, new, check_summaries=False)
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True


# ============================================================
# Section 23: Options Flags
# ============================================================

class TestOptionFlags:
    def test_no_equivalence_check(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new, check_equivalence=False, check_summaries=False)
        for fr in result.function_results.values():
            assert fr.equivalence_verified is None

    def test_no_summary_check(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        result = verify_refactoring(old, new, check_equivalence=False, check_summaries=False)
        assert result.old_summaries is None
        assert result.new_summaries is None

    def test_both_checks(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        result = verify_refactoring(old, new, check_equivalence=True, check_summaries=True)
        fr = result.function_results.get("f")
        if fr:
            assert fr.equivalence_verified is not None


# ============================================================
# Section 24: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_programs(self):
        result = verify_refactoring("let x = 1;", "let x = 1;")
        assert result.is_verified is True

    def test_toplevel_only(self):
        old = "let x = 1;"
        new = "let x = 2;"
        result = verify_refactoring(old, new)
        # No function results, but toplevel changed
        assert isinstance(result, RefactoringResult)

    def test_param_types_passed(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { let y = x + 1; return y; }"
        result = verify_refactoring(old, new, check_summaries=False,
                                     param_types={"f": {"x": "int"}})
        fr = result.function_results.get("f")
        assert fr is not None
        assert fr.equivalence_verified is True


# ============================================================
# Section 25: Certificate for Identity
# ============================================================

class TestCertificateIdentity:
    def test_identity_cert_valid(self):
        src = "fn f(x) { return x + 1; }"
        result = verify_refactoring(src, src)
        from proof_certificates import CertStatus
        assert result.certificate.status == CertStatus.VALID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
