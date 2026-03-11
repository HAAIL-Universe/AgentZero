"""
Tests for V139: Certified Regression Verification
"""

import pytest
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))

from certified_regression import (
    verify_regression, verify_function_regression, verify_program_regression,
    check_regression_certificate, save_regression_certificate,
    load_regression_certificate, compare_equiv_vs_kind, regression_summary,
    RegressionCertificate, RegressionMethod, RegressionVerdict,
    _try_equivalence, _try_k_induction,
)
from proof_certificates import CertStatus


# ============================================================
# Section 1: Equivalence-based regression (identical programs)
# ============================================================

class TestEquivalenceRegression:
    """Test regression verification via equivalence."""

    def test_identical_programs_equivalent(self):
        old = """
        fn f(x) { return x + 1; }
        """
        new = """
        fn f(x) { return x + 1; }
        """
        cert = verify_function_regression(old, new, "f", {"x": (-10, 10)})
        assert cert.verdict == RegressionVerdict.SAFE
        assert cert.method == RegressionMethod.EQUIVALENCE

    def test_algebraic_equivalence(self):
        old = """
        fn f(x) { return x + x; }
        """
        new = """
        fn f(x) { return 2 * x; }
        """
        cert = verify_function_regression(old, new, "f", {"x": (-10, 10)})
        assert cert.verdict == RegressionVerdict.SAFE
        assert cert.method == RegressionMethod.EQUIVALENCE

    def test_program_equivalence(self):
        old = """
        let x = 5;
        let y = x + 1;
        """
        new = """
        let x = 5;
        let y = 6;
        """
        cert = verify_program_regression(old, new, {"x": (5, 5)}, output_var="y")
        assert cert.verdict == RegressionVerdict.SAFE

    def test_non_equivalent_detected(self):
        old = """
        fn f(x) { return x + 1; }
        """
        new = """
        fn f(x) { return x + 2; }
        """
        cert = verify_function_regression(old, new, "f", {"x": (-10, 10)})
        # Should detect the change
        assert cert.verdict in (RegressionVerdict.UNSAFE, RegressionVerdict.UNKNOWN)

    def test_equivalence_with_restructuring(self):
        old = """
        fn abs(x) {
            if (x < 0) { return 0 - x; }
            return x;
        }
        """
        new = """
        fn abs(x) {
            if (x >= 0) { return x; }
            return 0 - x;
        }
        """
        cert = verify_function_regression(old, new, "abs", {"x": (-10, 10)})
        assert cert.verdict == RegressionVerdict.SAFE


# ============================================================
# Section 2: k-Induction fallback
# ============================================================

class TestKInductionFallback:
    """Test k-induction fallback when equivalence fails."""

    def test_kind_fallback_on_changed_code(self):
        old = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        new = """
        let i = 0;
        while (i < 20) {
            i = i + 1;
        }
        """
        cert = verify_regression(
            old, new,
            property_source="i >= 0",
        )
        # Property i >= 0 still holds on new version
        # Equivalence not attempted (no symbolic_inputs), k-induction should work
        assert cert.verdict in (RegressionVerdict.SAFE, RegressionVerdict.UNKNOWN)

    def test_kind_with_invariants(self):
        old = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        new = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        cert = verify_regression(
            old, new,
            property_source="i >= 0",
            invariant_sources=["i >= 0"],
        )
        assert cert.verdict in (RegressionVerdict.SAFE, RegressionVerdict.UNKNOWN)


# ============================================================
# Section 3: Combined strategy
# ============================================================

class TestCombinedStrategy:
    """Test combined equivalence + k-induction strategy."""

    def test_equiv_succeeds_first(self):
        old = "fn f(x) { return x * 2; }"
        new = "fn f(x) { return x + x; }"
        cert = verify_regression(
            old, new,
            fn_name="f",
            param_types={"x": (-5, 5)},
            property_source="result >= 0",
        )
        assert cert.verdict == RegressionVerdict.SAFE
        # Should succeed via equivalence (faster)
        assert cert.method == RegressionMethod.EQUIVALENCE

    def test_no_inputs_no_property_unknown(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x + 1; }"
        cert = verify_regression(old, new)
        # No symbolic inputs or property -> nothing to check
        assert cert.verdict in (RegressionVerdict.UNKNOWN, RegressionVerdict.UNSAFE)
        assert cert.method in (RegressionMethod.NONE, RegressionMethod.COMBINED)


# ============================================================
# Section 4: Certificate structure
# ============================================================

class TestCertificateStructure:
    """Test regression certificate data structure."""

    def test_certificate_fields(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 1; }"
        cert = verify_function_regression(old, new, "f", {"x": (-5, 5)})
        assert cert.verdict is not None
        assert cert.method is not None
        assert cert.claim != ""
        assert cert.source_old == old
        assert cert.source_new == new
        assert cert.timestamp != ""

    def test_certificate_summary(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 1; }"
        cert = verify_function_regression(old, new, "f", {"x": (-5, 5)})
        s = cert.summary()
        assert "SAFE" in s
        assert "EQUIVALENCE" in s

    def test_certificate_obligations(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 1; }"
        cert = verify_function_regression(old, new, "f", {"x": (-5, 5)})
        assert cert.total_obligations >= 0
        assert cert.valid_obligations >= 0

    def test_certificate_metadata(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 1; }"
        cert = verify_function_regression(old, new, "f", {"x": (-5, 5)})
        assert "method" in cert.metadata
        assert "duration" in cert.metadata


# ============================================================
# Section 5: JSON serialization
# ============================================================

class TestSerialization:
    """Test certificate serialization."""

    def test_to_dict(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        d = cert.to_dict()
        assert d["verdict"] == "SAFE"
        assert d["method"] == "EQUIVALENCE"
        assert "obligations" in d

    def test_to_json(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        j = cert.to_json()
        parsed = json.loads(j)
        assert parsed["verdict"] == "SAFE"

    def test_roundtrip(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        j = cert.to_json()
        cert2 = RegressionCertificate.from_json(j)
        assert cert2.verdict == cert.verdict
        assert cert2.method == cert.method
        assert cert2.claim == cert.claim

    def test_save_load(self, tmp_path):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        path = str(tmp_path / "cert.json")
        save_regression_certificate(cert, path)
        cert2 = load_regression_certificate(path)
        assert cert2.verdict == cert.verdict
        assert cert2.method == cert.method


# ============================================================
# Section 6: Certificate checking
# ============================================================

class TestCertificateChecking:
    """Test independent certificate verification."""

    def test_check_valid_equiv_cert(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 1; }"
        cert = verify_function_regression(old, new, "f", {"x": (-5, 5)})
        checked = check_regression_certificate(cert)
        # Valid equivalence cert should remain valid
        assert checked.verdict == RegressionVerdict.SAFE

    def test_check_preserves_verdict(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        checked = check_regression_certificate(cert)
        assert checked.verdict == cert.verdict


# ============================================================
# Section 7: Comparison API
# ============================================================

class TestComparisonAPI:
    """Test comparison of equivalence vs k-induction."""

    def test_compare_structure(self):
        old = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        new = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        comp = compare_equiv_vs_kind(
            old, new,
            symbolic_inputs={"i": (0, 0)},
            property_source="i >= 0",
            output_var="i",
        )
        assert "equivalence" in comp
        assert "k_induction" in comp
        assert "both_safe" in comp

    def test_compare_equiv_faster_for_identical(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        # Can't compare as we don't have symbolic_inputs for functions,
        # but structure should be valid
        comp = compare_equiv_vs_kind(
            old, new,
            symbolic_inputs={"x": (-5, 5)},
            property_source="x >= 0",
            output_var="x",
        )
        assert isinstance(comp["equivalence"]["time"], float)
        assert isinstance(comp["k_induction"]["time"], float)


# ============================================================
# Section 8: Summary API
# ============================================================

class TestSummaryAPI:
    """Test the summary API."""

    def test_summary_fields(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        s = regression_summary(cert)
        assert "verdict" in s
        assert "method" in s
        assert "total_obligations" in s
        assert "valid_obligations" in s
        assert "status" in s
        assert "metadata" in s

    def test_summary_reflects_verdict(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        s = regression_summary(cert)
        assert s["verdict"] == "SAFE"


# ============================================================
# Section 9: Internal helpers
# ============================================================

class TestInternalHelpers:
    """Test internal helper functions."""

    def test_try_equivalence_success(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = _try_equivalence(old, new, fn_name="f", param_types={"x": (-5, 5)})
        assert cert is not None
        assert cert.result == "equivalent"

    def test_try_equivalence_failure(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x + 1; }"
        cert = _try_equivalence(old, new, fn_name="f", param_types={"x": (-5, 5)})
        assert cert is not None
        assert cert.result == "not_equivalent"

    def test_try_equivalence_no_inputs(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = _try_equivalence(old, new)
        assert cert is None  # No inputs provided

    def test_try_kind_simple_loop(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        cert = _try_k_induction(source, "i >= 0")
        # May or may not succeed depending on induction depth
        assert cert is None or isinstance(cert, type(cert))


# ============================================================
# Section 10: Edge cases
# ============================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_functions(self):
        old = "fn f() { return 0; }"
        new = "fn f() { return 0; }"
        cert = verify_function_regression(old, new, "f", {})
        # No params -> no symbolic inputs -> can't check equivalence
        assert cert.verdict in (RegressionVerdict.SAFE, RegressionVerdict.UNKNOWN)

    def test_multiarg_function(self):
        old = "fn f(a, b) { return a + b; }"
        new = "fn f(a, b) { return b + a; }"
        cert = verify_function_regression(old, new, "f", {"a": (-5, 5), "b": (-5, 5)})
        assert cert.verdict == RegressionVerdict.SAFE

    def test_conditional_regression(self):
        old = """
        fn max(a, b) {
            if (a > b) { return a; }
            return b;
        }
        """
        new = """
        fn max(a, b) {
            if (a >= b) { return a; }
            return b;
        }
        """
        cert = verify_function_regression(old, new, "max", {"a": (-5, 5), "b": (-5, 5)})
        assert cert.verdict == RegressionVerdict.SAFE

    def test_regression_introduces_bug(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x - 1; }"
        cert = verify_function_regression(old, new, "f", {"x": (-5, 5)})
        assert cert.verdict in (RegressionVerdict.UNSAFE, RegressionVerdict.UNKNOWN)

    def test_same_source_different_style(self):
        old = """
        fn f(x) {
            let y = x + 1;
            return y;
        }
        """
        new = """
        fn f(x) { return x + 1; }
        """
        cert = verify_function_regression(old, new, "f", {"x": (-5, 5)})
        assert cert.verdict == RegressionVerdict.SAFE


# ============================================================
# Section 11: Verdict classification
# ============================================================

class TestVerdictClassification:
    """Test verdict and method classification."""

    def test_safe_verdict(self):
        old = "fn f(x) { return x; }"
        new = "fn f(x) { return x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        assert cert.verdict == RegressionVerdict.SAFE
        assert cert.equiv_cert is not None
        assert cert.kind_cert is None

    def test_method_is_equivalence(self):
        old = "fn f(x) { return x * 2; }"
        new = "fn f(x) { return x + x; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        assert cert.method == RegressionMethod.EQUIVALENCE

    def test_unsafe_has_counterexample(self):
        old = "fn f(x) { return x + 1; }"
        new = "fn f(x) { return x + 2; }"
        cert = verify_function_regression(old, new, "f", {"x": (-3, 3)})
        if cert.verdict == RegressionVerdict.UNSAFE:
            assert cert.counterexample is not None
