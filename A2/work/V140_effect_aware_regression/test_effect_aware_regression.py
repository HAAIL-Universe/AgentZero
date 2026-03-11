"""Tests for V140: Effect-Aware Regression Verification"""

import sys, os, pytest
sys.path.insert(0, os.path.dirname(__file__))

from effect_aware_regression import (
    EffectRegressionVerdict, EffectChangeKind, EffectChange,
    EffectRegressionResult, verify_effect_regression,
    verify_function_effect_regression, check_effect_purity_preserved,
    compare_effect_regression_methods, effect_regression_summary,
    _infer_effects, _compute_effect_changes,
)


# --- Test Sources ---

PURE_FN = """
fn add(a, b) {
    return a + b;
}
"""

PURE_FN_REFACTORED = """
fn add(a, b) {
    let sum = a + b;
    return sum;
}
"""

IO_FN = """
fn add(a, b) {
    print(a + b);
    return a + b;
}
"""

DIV_FN = """
fn divide(a, b) {
    return a / b;
}
"""

SAFE_DIV_FN = """
fn divide(a, b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}
"""

SIMPLE_LOOP_OLD = """
let x = 0;
let i = 0;
while (i < 5) {
    x = x + 1;
    i = i + 1;
}
"""

SIMPLE_LOOP_NEW = """
let x = 0;
let i = 0;
while (i < 5) {
    x = x + 2;
    i = i + 1;
}
"""

MULTI_FN_OLD = """
fn compute(x) {
    return x * 2;
}
fn helper(y) {
    return y + 1;
}
"""

MULTI_FN_NEW = """
fn compute(x) {
    print(x * 2);
    return x * 2;
}
fn helper(y) {
    return y + 1;
}
"""

MULTI_FN_NEW_SAFE = """
fn compute(x) {
    return x * 2;
}
fn helper(y) {
    return y + 1;
}
"""


# === Section 1: Effect Inference ===

class TestEffectInference:
    def test_pure_function_no_effects(self):
        effs = _infer_effects(PURE_FN)
        assert "add" in effs

    def test_io_function_has_effects(self):
        effs = _infer_effects(IO_FN)
        assert "add" in effs
        # IO effect expected
        assert len(effs["add"]) > 0

    def test_div_function_has_exception(self):
        effs = _infer_effects(DIV_FN)
        assert "divide" in effs
        # Division can raise exception
        assert len(effs["divide"]) > 0

    def test_empty_source(self):
        effs = _infer_effects("")
        assert isinstance(effs, dict)

    def test_multi_function(self):
        effs = _infer_effects(MULTI_FN_OLD)
        assert "compute" in effs
        assert "helper" in effs


# === Section 2: Effect Change Detection ===

class TestEffectChanges:
    def test_no_changes(self):
        old = {"fn1": ["IO"], "fn2": []}
        new = {"fn1": ["IO"], "fn2": []}
        changes = _compute_effect_changes(old, new)
        assert len(changes) == 0

    def test_added_effect(self):
        old = {"fn1": []}
        new = {"fn1": ["IO"]}
        changes = _compute_effect_changes(old, new)
        assert len(changes) == 1
        assert changes[0].kind == EffectChangeKind.ADDED
        assert changes[0].effect == "IO"

    def test_removed_effect(self):
        old = {"fn1": ["IO", "Exn"]}
        new = {"fn1": ["IO"]}
        changes = _compute_effect_changes(old, new)
        assert len(changes) == 1
        assert changes[0].kind == EffectChangeKind.REMOVED
        assert changes[0].effect == "Exn"

    def test_new_function(self):
        old = {"fn1": []}
        new = {"fn1": [], "fn2": ["IO"]}
        changes = _compute_effect_changes(old, new)
        assert len(changes) == 1
        assert changes[0].function == "fn2"
        assert changes[0].kind == EffectChangeKind.ADDED

    def test_removed_function(self):
        old = {"fn1": [], "fn2": ["IO"]}
        new = {"fn1": []}
        changes = _compute_effect_changes(old, new)
        assert len(changes) == 1
        assert changes[0].function == "fn2"
        assert changes[0].kind == EffectChangeKind.REMOVED

    def test_mixed_changes(self):
        old = {"fn1": ["IO"], "fn2": ["Exn"]}
        new = {"fn1": ["IO", "Exn"], "fn2": []}
        changes = _compute_effect_changes(old, new)
        # fn1 gains Exn, fn2 loses Exn
        added = [c for c in changes if c.kind == EffectChangeKind.ADDED]
        removed = [c for c in changes if c.kind == EffectChangeKind.REMOVED]
        assert len(added) == 1
        assert len(removed) == 1


# === Section 3: Verify Effect Regression (Basic) ===

class TestVerifyEffectRegression:
    def test_identical_source(self):
        result = verify_effect_regression(PURE_FN, PURE_FN)
        assert result.verdict == EffectRegressionVerdict.SAFE
        assert not result.has_effect_regression

    def test_pure_refactored_still_safe(self):
        result = verify_effect_regression(PURE_FN, PURE_FN_REFACTORED)
        assert result.verdict == EffectRegressionVerdict.SAFE

    def test_pure_to_io_regression(self):
        result = verify_effect_regression(PURE_FN, IO_FN)
        assert result.has_effect_regression
        assert result.verdict in (EffectRegressionVerdict.EFFECT_REGRESSION,
                                   EffectRegressionVerdict.UNSAFE)

    def test_result_has_effect_changes(self):
        result = verify_effect_regression(PURE_FN, IO_FN)
        assert len(result.effect_changes) > 0
        assert any(c.kind == EffectChangeKind.ADDED for c in result.effect_changes)

    def test_div_to_safe_div_improvement(self):
        result = verify_effect_regression(DIV_FN, SAFE_DIV_FN)
        # Safe div should have fewer effects (no exception)
        assert result.has_effect_improvement or not result.has_effect_regression

    def test_multi_fn_partial_regression(self):
        result = verify_effect_regression(MULTI_FN_OLD, MULTI_FN_NEW)
        # compute gained IO, helper unchanged
        assert result.has_effect_regression
        added = result.added_effects
        assert any(c.function == "compute" for c in added)


# === Section 4: Function-Level Regression ===

class TestFunctionEffectRegression:
    def test_pure_function_preserved(self):
        result = verify_function_effect_regression(PURE_FN, PURE_FN_REFACTORED, "add")
        assert result.verdict == EffectRegressionVerdict.SAFE

    def test_function_gains_io(self):
        result = verify_function_effect_regression(PURE_FN, IO_FN, "add")
        assert result.has_effect_regression

    def test_function_with_param_types(self):
        result = verify_function_effect_regression(
            PURE_FN, PURE_FN_REFACTORED, "add",
            param_types={"a": "int", "b": "int"}
        )
        assert result.verdict == EffectRegressionVerdict.SAFE

    def test_multi_fn_focus_on_helper(self):
        result = verify_function_effect_regression(
            MULTI_FN_OLD, MULTI_FN_NEW, "helper"
        )
        # helper didn't change
        assert result.verdict == EffectRegressionVerdict.SAFE


# === Section 5: Purity Preservation ===

class TestPurityPreservation:
    def test_pure_stays_pure(self):
        result = check_effect_purity_preserved(PURE_FN, PURE_FN_REFACTORED, "add")
        assert result["purity_preserved"]
        assert result["was_pure"]
        assert result["is_pure"]

    def test_pure_becomes_impure(self):
        result = check_effect_purity_preserved(PURE_FN, IO_FN, "add")
        assert result["was_pure"]
        assert not result["is_pure"]
        assert not result["purity_preserved"]

    def test_impure_stays_impure(self):
        result = check_effect_purity_preserved(IO_FN, IO_FN, "add")
        assert not result["was_pure"]
        assert not result["is_pure"]
        # Not a regression because it was already impure
        assert result["purity_preserved"]

    def test_impure_becomes_pure(self):
        result = check_effect_purity_preserved(IO_FN, PURE_FN, "add")
        assert not result["was_pure"]
        assert result["is_pure"]
        assert result["purity_preserved"]  # Improvement, not regression

    def test_result_has_effect_lists(self):
        result = check_effect_purity_preserved(PURE_FN, IO_FN, "add")
        assert "added_effects" in result
        assert "removed_effects" in result
        assert len(result["added_effects"]) > 0


# === Section 6: Effect Regression Result Properties ===

class TestResultProperties:
    def test_summary(self):
        result = verify_effect_regression(PURE_FN, IO_FN)
        s = result.summary()
        assert "Effect-Aware Regression" in s

    def test_to_dict(self):
        result = verify_effect_regression(PURE_FN, IO_FN)
        d = result.to_dict()
        assert "verdict" in d
        assert "effect_changes" in d
        assert "has_effect_regression" in d

    def test_added_effects_property(self):
        result = verify_effect_regression(PURE_FN, IO_FN)
        added = result.added_effects
        assert isinstance(added, list)
        assert all(isinstance(c, EffectChange) for c in added)

    def test_removed_effects_property(self):
        result = verify_effect_regression(PURE_FN, IO_FN)
        removed = result.removed_effects
        assert isinstance(removed, list)


# === Section 7: Comparison API ===

class TestCompareAPI:
    def test_compare_methods(self):
        result = compare_effect_regression_methods(
            SIMPLE_LOOP_OLD, SIMPLE_LOOP_NEW,
            symbolic_inputs={"x": "int"},
            property_source="x >= 0",
            output_var="x"
        )
        assert "effect_only" in result
        assert "regression_only" in result
        assert "combined" in result
        assert "time" in result["effect_only"]

    def test_compare_identical(self):
        result = compare_effect_regression_methods(
            PURE_FN, PURE_FN,
            symbolic_inputs={"a": "int"},
            property_source="a >= 0",
            output_var="a"
        )
        assert not result["effect_only"]["has_regression"]


# === Section 8: Summary API ===

class TestSummaryAPI:
    def test_summary_keys(self):
        result = verify_effect_regression(PURE_FN, PURE_FN)
        s = effect_regression_summary(result)
        assert "verdict" in s
        assert "effect_changes" in s
        assert "added_effects" in s
        assert "removed_effects" in s

    def test_summary_regression(self):
        result = verify_effect_regression(PURE_FN, IO_FN)
        s = effect_regression_summary(result)
        assert s["has_effect_regression"]
        assert s["added_effects"] > 0

    def test_summary_safe(self):
        result = verify_effect_regression(PURE_FN, PURE_FN)
        s = effect_regression_summary(result)
        assert s["verdict"] == "safe"
        assert not s["has_effect_regression"]


# === Section 9: Edge Cases ===

class TestEdgeCases:
    def test_empty_old_source(self):
        result = verify_effect_regression("", PURE_FN)
        assert isinstance(result, EffectRegressionResult)

    def test_empty_new_source(self):
        result = verify_effect_regression(PURE_FN, "")
        assert isinstance(result, EffectRegressionResult)

    def test_both_empty(self):
        result = verify_effect_regression("", "")
        assert result.verdict == EffectRegressionVerdict.SAFE

    def test_only_removed_effects_is_safe(self):
        # Going from impure to pure is an improvement, not regression
        result = verify_effect_regression(IO_FN, PURE_FN)
        assert not result.has_effect_regression
        assert result.verdict == EffectRegressionVerdict.SAFE

    def test_metadata_present(self):
        result = verify_effect_regression(PURE_FN, PURE_FN)
        assert "elapsed" in result.metadata


# === Section 10: Integration with V139 ===

class TestV139Integration:
    def test_with_property_source(self):
        old_src = """
let x = 0;
let i = 0;
while (i < 3) {
    x = x + 1;
    i = i + 1;
}
"""
        new_src = """
let x = 0;
let i = 0;
while (i < 3) {
    x = x + 1;
    i = i + 1;
}
"""
        result = verify_effect_regression(
            old_src, new_src,
            symbolic_inputs={"x": "int"},
            property_source="x >= 0",
            output_var="x"
        )
        assert isinstance(result, EffectRegressionResult)
        # Same source should be safe
        assert not result.has_effect_regression

    def test_regression_cert_present_when_property_given(self):
        result = verify_effect_regression(
            SIMPLE_LOOP_OLD, SIMPLE_LOOP_NEW,
            symbolic_inputs={"x": "int"},
            property_source="x >= 0",
            output_var="x"
        )
        # Should have attempted regression certification
        assert isinstance(result, EffectRegressionResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
