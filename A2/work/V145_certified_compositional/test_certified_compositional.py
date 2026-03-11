"""
Tests for V145: Certified Compositional Verification

Tests modular verification, certificate composition, incremental re-verification,
spec refinement checking, call graph analysis, and comparison with monolithic V004.
"""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from certified_compositional import (
    extract_modules, verify_module, verify_compositional,
    compare_modular_vs_monolithic, verify_incremental,
    check_spec_refinement, analyze_call_graph, analyze_change_impact,
    certify_compositional, compositional_summary,
    CompVerdict, ModuleSpec, ModuleResult, CompositionalResult,
    CallSiteObligation, VCStatus,
)

# Also import V004 for cross-checks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
from vc_gen import verify_function, VCStatus as V4Status

# Also import V044 for certificate checks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
from proof_certificates import ProofCertificate, CertStatus, ProofKind


# ============================================================
# Section 1: Module extraction
# ============================================================

class TestModuleExtraction:
    def test_single_function(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        modules, top = extract_modules(source)
        assert len(modules) == 1
        assert modules[0].name == "inc"
        assert len(modules[0].preconditions) == 1
        assert len(modules[0].postconditions) == 1
        assert len(top) == 0

    def test_multiple_functions(self):
        source = """
        fn add(a, b) {
            requires(a >= 0);
            requires(b >= 0);
            ensures(result >= 0);
            return a + b;
        }
        fn double(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + x;
        }
        """
        modules, top = extract_modules(source)
        assert len(modules) == 2
        names = {m.name for m in modules}
        assert names == {"add", "double"}

    def test_function_with_top_level(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        let y = 10;
        """
        modules, top = extract_modules(source)
        assert len(modules) == 1
        assert len(top) == 1

    def test_no_spec_function(self):
        source = """
        fn foo(x) {
            return x + 1;
        }
        """
        modules, top = extract_modules(source)
        assert len(modules) == 1
        assert modules[0].preconditions == []
        assert modules[0].postconditions == []

    def test_multiple_preconditions(self):
        source = """
        fn clamp(x, lo, hi) {
            requires(lo <= hi);
            requires(x >= 0);
            ensures(result >= lo);
            ensures(result <= hi);
            if (x < lo) { return lo; }
            if (x > hi) { return hi; }
            return x;
        }
        """
        modules, _ = extract_modules(source)
        assert len(modules[0].preconditions) == 2
        assert len(modules[0].postconditions) == 2


# ============================================================
# Section 2: Single module verification
# ============================================================

class TestModuleVerification:
    def test_simple_verified(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        modules, _ = extract_modules(source)
        spec_map = {m.name: m for m in modules}
        result = verify_module(modules[0], spec_map)
        assert result.verified is True
        assert result.name == "inc"

    def test_simple_failing(self):
        source = """
        fn bad(x) {
            requires(x >= 0);
            ensures(result > 100);
            return x + 1;
        }
        """
        modules, _ = extract_modules(source)
        spec_map = {m.name: m for m in modules}
        result = verify_module(modules[0], spec_map)
        assert result.verified is False

    def test_identity_verified(self):
        source = """
        fn identity(x) {
            requires(x > 0);
            ensures(result > 0);
            return x;
        }
        """
        modules, _ = extract_modules(source)
        spec_map = {m.name: m for m in modules}
        result = verify_module(modules[0], spec_map)
        assert result.verified is True

    def test_conditional_verified(self):
        source = """
        fn abs_val(x) {
            ensures(result >= 0);
            let r = x;
            if (x < 0) {
                r = 0 - x;
            }
            return r;
        }
        """
        modules, _ = extract_modules(source)
        spec_map = {m.name: m for m in modules}
        result = verify_module(modules[0], spec_map)
        assert result.verified is True

    def test_module_has_certificate(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        modules, _ = extract_modules(source)
        spec_map = {m.name: m for m in modules}
        result = verify_module(modules[0], spec_map)
        assert result.certificate is not None
        assert result.certificate.kind == ProofKind.VCGEN
        assert result.certificate.status == CertStatus.VALID


# ============================================================
# Section 3: Compositional verification (no inter-function calls)
# ============================================================

class TestCompositionalBasic:
    def test_single_function_sound(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_modules == 1
        assert result.verified_modules == 1

    def test_two_independent_functions(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double(x) {
            requires(x > 0);
            ensures(result > 0);
            return x + x;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_modules == 2
        assert result.verified_modules == 2

    def test_one_failing_module(self):
        source = """
        fn good(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn bad(x) {
            requires(x >= 0);
            ensures(result > 100);
            return x + 1;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.MODULE_FAILURE
        assert result.verified_modules == 1

    def test_empty_program(self):
        source = "let x = 5;"
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_modules == 0

    def test_composed_certificate(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double(x) {
            requires(x > 0);
            ensures(result > 0);
            return x + x;
        }
        """
        result = verify_compositional(source)
        assert result.certificate is not None
        assert result.certificate.kind == ProofKind.COMPOSITE

    def test_metadata(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        result = verify_compositional(source)
        assert "duration" in result.metadata
        assert result.metadata["num_modules"] == 1


# ============================================================
# Section 4: Inter-function call verification
# ============================================================

class TestInterFunctionCalls:
    def test_caller_satisfies_callee_precond(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn use_inc(y) {
            requires(y >= 0);
            ensures(result > 0);
            let z = inc(y);
            return z;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_call_obligations > 0
        assert result.satisfied_call_obligations == result.total_call_obligations

    def test_caller_violates_callee_precond(self):
        source = """
        fn positive_only(x) {
            requires(x > 0);
            ensures(result > 0);
            return x;
        }
        fn bad_caller(y) {
            requires(y >= 0);
            ensures(result > 0);
            let z = positive_only(y);
            return z;
        }
        """
        # y >= 0 does not imply y > 0 (y could be 0)
        result = verify_compositional(source)
        # The call obligation should fail
        failed = [co for co in result.call_obligations if co.status != VCStatus.VALID]
        assert len(failed) > 0

    def test_chain_of_calls(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn inc2(x) {
            requires(x >= 0);
            ensures(result > 0);
            let y = inc(x);
            return y;
        }
        fn inc3(x) {
            requires(x >= 0);
            ensures(result > 0);
            let y = inc2(x);
            return y;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_call_obligations >= 2

    def test_modular_postcondition_use(self):
        """Caller uses callee's postcondition to establish its own."""
        source = """
        fn make_positive(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double_positive(x) {
            requires(x >= 0);
            ensures(result > 0);
            let y = make_positive(x);
            return y;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        # double_positive trusts make_positive's ensures(result > 0)
        # and uses it to satisfy its own ensures(result > 0)


# ============================================================
# Section 5: Comparison with monolithic V004
# ============================================================

class TestComparison:
    def test_agreement_on_valid(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        comp = compare_modular_vs_monolithic(source)
        assert comp["agree"] is True
        assert comp["modular_verdict"] == "sound"
        assert comp["monolithic_verified"] is True

    def test_agreement_on_invalid(self):
        source = """
        fn bad(x) {
            requires(x >= 0);
            ensures(result > 100);
            return x + 1;
        }
        """
        comp = compare_modular_vs_monolithic(source)
        assert comp["agree"] is True
        assert comp["modular_verdict"] == "module_failure"
        assert comp["monolithic_verified"] is False

    def test_timing_data(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        comp = compare_modular_vs_monolithic(source)
        assert "modular_time" in comp
        assert "monolithic_time" in comp
        assert comp["modular_time"] >= 0
        assert comp["monolithic_time"] >= 0


# ============================================================
# Section 6: Incremental verification
# ============================================================

class TestIncremental:
    def test_full_verification_from_scratch(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double(x) {
            requires(x > 0);
            ensures(result > 0);
            return x + x;
        }
        """
        result = verify_incremental(source, ["inc", "double"])
        assert result.verdict == CompVerdict.SOUND
        assert len(result.metadata["reverified"]) == 2
        assert len(result.metadata["reused"]) == 0

    def test_reuse_unchanged_modules(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double(x) {
            requires(x > 0);
            ensures(result > 0);
            return x + x;
        }
        """
        # First: full verification
        full = verify_compositional(source)

        # Second: only inc changed
        incr = verify_incremental(source, ["inc"], cached_results=full.modules)
        assert incr.verdict == CompVerdict.SOUND
        assert "inc" in incr.metadata["reverified"]
        assert "double" in incr.metadata["reused"]

    def test_caller_reverified_when_callee_changes(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn use_inc(y) {
            requires(y >= 0);
            ensures(result > 0);
            let z = inc(y);
            return z;
        }
        """
        full = verify_compositional(source)

        # inc changed -- use_inc calls inc, so should be reverified too
        incr = verify_incremental(source, ["inc"], cached_results=full.modules)
        assert "inc" in incr.metadata["reverified"]
        assert "use_inc" in incr.metadata["reverified"]

    def test_no_changes_all_reused(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        full = verify_compositional(source)
        incr = verify_incremental(source, [], cached_results=full.modules)
        assert incr.verdict == CompVerdict.SOUND
        assert len(incr.metadata["reverified"]) == 0
        assert "inc" in incr.metadata["reused"]


# ============================================================
# Section 7: Spec refinement checking
# ============================================================

class TestSpecRefinement:
    def test_identical_spec_refines(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        result = check_spec_refinement(source, source, "inc")
        assert result["refinement"] is True

    def test_weakened_precondition(self):
        old_source = """
        fn f(x) {
            requires(x > 0);
            ensures(result > 0);
            return x;
        }
        """
        new_source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        result = check_spec_refinement(old_source, new_source, "f")
        # x > 0 => x >= 0 is true (weakened precondition)
        assert result["precond_weakened"] is True

    def test_strengthened_postcondition(self):
        old_source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        """
        new_source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        result = check_spec_refinement(old_source, new_source, "f")
        # result > 0 => result >= 0 is true (stronger postcondition implies old)
        assert result["postcond_strengthened"] is True

    def test_no_refinement_stronger_precond(self):
        old_source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        new_source = """
        fn f(x) {
            requires(x > 0);
            ensures(result >= 0);
            return x;
        }
        """
        result = check_spec_refinement(old_source, new_source, "f")
        # x >= 0 does NOT imply x > 0 (strengthened precondition = not a refinement)
        assert result["precond_weakened"] is False

    def test_function_not_found(self):
        source = """
        fn f(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        result = check_spec_refinement(source, source, "nonexistent")
        assert result["refinement"] is False
        assert "error" in result


# ============================================================
# Section 8: Call graph analysis
# ============================================================

class TestCallGraph:
    def test_simple_call_graph(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn use_inc(y) {
            requires(y >= 0);
            ensures(result > 0);
            let z = inc(y);
            return z;
        }
        """
        graph = analyze_call_graph(source)
        assert "inc" in graph["call_graph"]
        assert "use_inc" in graph["call_graph"]
        assert "inc" in graph["call_graph"]["use_inc"]
        assert graph["call_graph"]["inc"] == []
        assert graph["num_functions"] == 2

    def test_reverse_graph(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn use_inc(y) {
            requires(y >= 0);
            ensures(result > 0);
            let z = inc(y);
            return z;
        }
        """
        graph = analyze_call_graph(source)
        assert "use_inc" in graph["reverse_graph"]["inc"]

    def test_specified_vs_unspecified(self):
        source = """
        fn specified_fn(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn unspecified_fn(x) {
            return x + 2;
        }
        """
        graph = analyze_call_graph(source)
        assert "specified_fn" in graph["specified"]
        assert "unspecified_fn" in graph["unspecified"]

    def test_chain_call_graph(self):
        source = """
        fn a(x) { requires(x >= 0); ensures(result >= 0); return x; }
        fn b(x) { requires(x >= 0); ensures(result >= 0); let y = a(x); return y; }
        fn c(x) { requires(x >= 0); ensures(result >= 0); let y = b(x); return y; }
        """
        graph = analyze_call_graph(source)
        assert "a" in graph["call_graph"]["b"]
        assert "b" in graph["call_graph"]["c"]
        assert "a" not in graph["call_graph"]["c"]  # c doesn't directly call a


# ============================================================
# Section 9: Change impact analysis
# ============================================================

class TestChangeImpact:
    def test_body_change_impact(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn use_inc(y) {
            requires(y >= 0);
            ensures(result > 0);
            let z = inc(y);
            return z;
        }
        """
        impact = analyze_change_impact(source, "inc")
        assert impact["changed"] == "inc"
        assert impact["body_change_impact"] == ["inc"]
        assert "use_inc" in impact["spec_change_impact"]

    def test_leaf_function_impact(self):
        source = """
        fn leaf(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        impact = analyze_change_impact(source, "leaf")
        assert impact["callers"] == []
        assert impact["body_change_impact"] == ["leaf"]
        assert impact["spec_change_impact"] == ["leaf"]


# ============================================================
# Section 10: Certified compositional (V044 integration)
# ============================================================

class TestCertifiedCompositional:
    def test_certify_sound(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        result = certify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.certificate is not None

    def test_certify_failing(self):
        source = """
        fn bad(x) {
            requires(x >= 0);
            ensures(result > 100);
            return x + 1;
        }
        """
        result = certify_compositional(source)
        assert result.verdict == CompVerdict.MODULE_FAILURE

    def test_composite_certificate_structure(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double(x) {
            requires(x > 0);
            ensures(result > 0);
            return x + x;
        }
        """
        result = certify_compositional(source)
        cert = result.certificate
        assert cert is not None
        assert cert.kind == ProofKind.COMPOSITE


# ============================================================
# Section 11: Summary and serialization
# ============================================================

class TestSummary:
    def test_summary_sound(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        result = verify_compositional(source)
        s = compositional_summary(result)
        assert "sound" in s.lower()

    def test_summary_failure(self):
        source = """
        fn bad(x) {
            requires(x >= 0);
            ensures(result > 100);
            return x + 1;
        }
        """
        result = verify_compositional(source)
        s = compositional_summary(result)
        assert "module_failure" in s.lower()

    def test_result_properties(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double(x) {
            requires(x > 0);
            ensures(result > 0);
            return x + x;
        }
        """
        result = verify_compositional(source)
        assert result.total_modules == 2
        assert result.verified_modules == 2


# ============================================================
# Section 12: Edge cases
# ============================================================

class TestEdgeCases:
    def test_no_functions(self):
        source = "let x = 5;"
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_modules == 0

    def test_function_no_spec(self):
        source = """
        fn foo(x) {
            return x + 1;
        }
        """
        result = verify_compositional(source)
        # No spec => trivially verified (no VCs to check, postcond is True)
        assert result.verdict == CompVerdict.SOUND

    def test_multiple_params(self):
        source = """
        fn add(a, b) {
            requires(a >= 0);
            requires(b >= 0);
            ensures(result >= 0);
            return a + b;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND

    def test_abs_function(self):
        source = """
        fn abs_val(x) {
            ensures(result >= 0);
            let r = x;
            if (x < 0) {
                r = 0 - x;
            }
            return r;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND

    def test_max_function(self):
        """Max using single return (V004 WP doesn't handle early returns in branches)."""
        source = """
        fn max_val(a, b) {
            ensures(result >= a);
            ensures(result >= b);
            let r = b;
            if (a >= b) {
                r = a;
            }
            return r;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND


# ============================================================
# Section 13: Complex multi-module programs
# ============================================================

class TestComplexPrograms:
    def test_three_function_chain(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn double_inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            let y = inc(x);
            return y;
        }
        fn triple(x) {
            requires(x >= 0);
            ensures(result > 0);
            let y = double_inc(x);
            return y;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_modules == 3
        assert result.verified_modules == 3
        assert result.total_call_obligations >= 2

    def test_diamond_call_pattern(self):
        """Two callers both use same callee."""
        source = """
        fn base(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn caller_a(x) {
            requires(x >= 0);
            ensures(result > 0);
            let y = base(x);
            return y;
        }
        fn caller_b(x) {
            requires(x > 0);
            ensures(result > 0);
            let y = base(x);
            return y;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.SOUND
        assert result.total_call_obligations >= 2

    def test_mixed_verified_unverified(self):
        """Program with some verified, some failing modules."""
        source = """
        fn good1(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn good2(x) {
            requires(x > 0);
            ensures(result > 0);
            return x;
        }
        fn bad(x) {
            requires(x >= 0);
            ensures(result > 1000);
            return x;
        }
        """
        result = verify_compositional(source)
        assert result.verdict == CompVerdict.MODULE_FAILURE
        assert result.verified_modules == 2  # good1 and good2
        assert result.modules["bad"].verified is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
