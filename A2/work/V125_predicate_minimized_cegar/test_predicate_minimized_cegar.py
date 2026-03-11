"""Tests for V125: Predicate-Minimized CEGAR."""

import pytest
import sys
import os

_here = os.path.dirname(__file__)
sys.path.insert(0, _here)
for p in [
    os.path.join(_here, '..', 'V119_bdd_predicate_abstraction'),
    os.path.join(_here, '..', 'V122_predicate_minimization'),
    os.path.join(_here, '..', 'V110_abstract_reachability_tree'),
    os.path.join(_here, '..', 'V021_bdd_model_checking'),
    os.path.join(_here, '..', '..', '..', 'challenges', 'C037_smt_solver'),
    os.path.join(_here, '..', '..', '..', 'challenges', 'C010_stack_vm'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from predicate_minimized_cegar import (
    MinimizedCEGAR, MinimizationMode, MinCEGARVerdict, MinCEGARResult,
    ComparisonResult, PredicateQuality, IncrementalMinCEGAR,
    minimized_cegar_verify, check_with_minimal_proof,
    compare_minimization_modes, minimized_cegar_summary,
    get_minimal_proof_predicates, verify_with_budget,
    analyze_predicate_quality,
)
from bdd_predicate_abstraction import bdd_verify, BDDVerdict


# --- Test Programs ---

SAFE_SIMPLE = """
let x = 5;
assert(x > 0);
"""

SAFE_CONDITIONAL = """
let x = 5;
if (x > 0) {
    let y = x + 1;
    assert(y > 0);
} else {
    assert(x > 0);
}
"""

UNSAFE_SIMPLE = """
let x = -1;
assert(x > 0);
"""

SAFE_MULTI_PRED = """
let x = 10;
let y = 20;
if (x > 0) {
    if (y > 0) {
        let z = x + y;
        assert(z > 0);
    }
}
"""

SAFE_BRANCH_MERGE = """
let x = 5;
let y = 0;
if (x > 3) {
    y = x;
} else {
    y = 4;
}
assert(y > 0);
"""

SAFE_ARITHMETIC = """
let a = 3;
let b = 7;
let c = a + b;
assert(c > 0);
"""

UNSAFE_BRANCH = """
let x = 5;
if (x > 10) {
    assert(x > 0);
} else {
    assert(x > 100);
}
"""

SAFE_NESTED = """
let x = 10;
if (x > 5) {
    let y = x - 3;
    if (y > 0) {
        assert(x > 0);
    }
}
"""


# ===== Section 1: MinCEGARVerdict Enum =====

class TestMinCEGARVerdict:
    def test_safe_value(self):
        assert MinCEGARVerdict.SAFE.value == "SAFE"

    def test_unsafe_value(self):
        assert MinCEGARVerdict.UNSAFE.value == "UNSAFE"

    def test_unknown_value(self):
        assert MinCEGARVerdict.UNKNOWN.value == "UNKNOWN"


# ===== Section 2: MinimizationMode Enum =====

class TestMinimizationMode:
    def test_post_hoc(self):
        assert MinimizationMode.POST_HOC.value == "post_hoc"

    def test_online(self):
        assert MinimizationMode.ONLINE.value == "online"

    def test_eager(self):
        assert MinimizationMode.EAGER.value == "eager"


# ===== Section 3: Post-Hoc Minimization (Safe Programs) =====

class TestPostHocSafe:
    def test_simple_safe(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "post_hoc")
        assert result.safe
        assert result.verdict == MinCEGARVerdict.SAFE
        assert result.minimal_predicates >= 1
        assert result.minimal_predicates <= result.original_predicates

    def test_conditional_safe(self):
        result = minimized_cegar_verify(SAFE_CONDITIONAL, "post_hoc")
        assert result.safe
        assert result.minimal_predicates <= result.original_predicates

    def test_multi_predicate_safe(self):
        result = minimized_cegar_verify(SAFE_MULTI_PRED, "post_hoc")
        assert result.safe
        assert result.reduction_ratio >= 0.0

    def test_branch_merge_safe(self):
        result = minimized_cegar_verify(SAFE_BRANCH_MERGE, "post_hoc")
        assert result.safe

    def test_arithmetic_safe(self):
        result = minimized_cegar_verify(SAFE_ARITHMETIC, "post_hoc")
        assert result.safe

    def test_nested_safe(self):
        result = minimized_cegar_verify(SAFE_NESTED, "post_hoc")
        assert result.safe


# ===== Section 4: Post-Hoc Minimization (Unsafe Programs) =====

class TestPostHocUnsafe:
    def test_simple_unsafe(self):
        result = minimized_cegar_verify(UNSAFE_SIMPLE, "post_hoc")
        assert not result.safe
        assert result.verdict == MinCEGARVerdict.UNSAFE
        assert result.minimization_iterations == 0  # No minimization for unsafe

    def test_branch_unsafe(self):
        result = minimized_cegar_verify(UNSAFE_BRANCH, "post_hoc")
        assert not result.safe
        assert result.counterexample_inputs is not None

    def test_unsafe_has_counterexample(self):
        result = minimized_cegar_verify(UNSAFE_SIMPLE, "post_hoc")
        assert not result.safe
        assert result.counterexample is not None


# ===== Section 5: Online Minimization =====

class TestOnlineMode:
    def test_simple_safe(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "online")
        assert result.safe
        assert result.minimization_mode == "online"

    def test_conditional_safe(self):
        result = minimized_cegar_verify(SAFE_CONDITIONAL, "online")
        assert result.safe

    def test_multi_pred_safe(self):
        result = minimized_cegar_verify(SAFE_MULTI_PRED, "online")
        assert result.safe
        assert result.minimal_predicates <= result.original_predicates

    def test_unsafe_detected(self):
        result = minimized_cegar_verify(UNSAFE_SIMPLE, "online")
        assert not result.safe

    def test_online_mode_label(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "online")
        assert result.minimization_mode == "online"


# ===== Section 6: Eager Minimization =====

class TestEagerMode:
    def test_simple_safe(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "eager")
        assert result.safe
        assert result.minimization_mode == "eager"

    def test_multi_pred_safe(self):
        result = minimized_cegar_verify(SAFE_MULTI_PRED, "eager")
        assert result.safe

    def test_unsafe_detected(self):
        result = minimized_cegar_verify(UNSAFE_SIMPLE, "eager")
        assert not result.safe

    def test_eager_mode_label(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "eager")
        assert result.minimization_mode == "eager"


# ===== Section 7: MinCEGARResult Properties =====

class TestMinCEGARResult:
    def test_safe_result_fields(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "post_hoc")
        assert result.safe
        assert result.original_predicates >= 1
        assert result.minimal_predicates >= 1
        assert len(result.predicate_names) == result.minimal_predicates
        assert result.cegar_iterations >= 1
        assert result.art_nodes >= 1
        assert result.total_time_ms >= 0

    def test_unsafe_result_fields(self):
        result = minimized_cegar_verify(UNSAFE_SIMPLE, "post_hoc")
        assert not result.safe
        assert result.counterexample is not None
        assert result.reduction_ratio == 0.0

    def test_reduction_ratio_bounded(self):
        result = minimized_cegar_verify(SAFE_MULTI_PRED, "post_hoc")
        assert 0.0 <= result.reduction_ratio <= 1.0

    def test_summary_contains_verdict(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "post_hoc")
        s = result.summary
        assert "SAFE" in s
        assert "Predicate-Minimized CEGAR" in s

    def test_summary_unsafe(self):
        result = minimized_cegar_verify(UNSAFE_SIMPLE, "post_hoc")
        s = result.summary
        assert "UNSAFE" in s


# ===== Section 8: check_with_minimal_proof =====

class TestCheckWithMinimalProof:
    def test_safe(self):
        safe, n_preds, names = check_with_minimal_proof(SAFE_SIMPLE)
        assert safe
        assert n_preds >= 1
        assert len(names) == n_preds

    def test_unsafe(self):
        safe, n_preds, names = check_with_minimal_proof(UNSAFE_SIMPLE)
        assert not safe


# ===== Section 9: compare_minimization_modes =====

class TestCompareMinimizationModes:
    def test_comparison_safe(self):
        comp = compare_minimization_modes(SAFE_SIMPLE)
        assert comp.post_hoc is not None
        assert comp.online is not None
        assert comp.eager is not None
        assert comp.best_mode in ("post_hoc", "online", "eager")
        assert comp.best_predicates >= 1

    def test_comparison_has_standard(self):
        comp = compare_minimization_modes(SAFE_SIMPLE)
        assert comp.standard is not None

    def test_comparison_summary(self):
        comp = compare_minimization_modes(SAFE_SIMPLE)
        assert "Comparison" in comp.summary or "comparison" in comp.summary.lower()

    def test_comparison_unsafe(self):
        comp = compare_minimization_modes(UNSAFE_SIMPLE)
        if comp.post_hoc:
            assert not comp.post_hoc.safe


# ===== Section 10: minimized_cegar_summary =====

class TestMinimizedCEGARSummary:
    def test_safe_summary(self):
        s = minimized_cegar_summary(SAFE_SIMPLE)
        assert "SAFE" in s
        assert "Predicate" in s or "predicate" in s.lower()

    def test_unsafe_summary(self):
        s = minimized_cegar_summary(UNSAFE_SIMPLE)
        assert "UNSAFE" in s


# ===== Section 11: get_minimal_proof_predicates =====

class TestGetMinimalProofPredicates:
    def test_safe_info(self):
        info = get_minimal_proof_predicates(SAFE_SIMPLE)
        assert info['safe']
        assert info['total_predicates'] >= 1
        assert info['minimal_predicates'] >= 1
        assert info['minimal_predicates'] <= info['total_predicates']
        assert len(info['essential']) == info['minimal_predicates']

    def test_quality_field(self):
        info = get_minimal_proof_predicates(SAFE_SIMPLE)
        q = info['quality']
        assert isinstance(q, PredicateQuality)
        assert q.essential >= 1
        assert 0.0 <= q.ratio <= 1.0


# ===== Section 12: verify_with_budget =====

class TestVerifyWithBudget:
    def test_within_budget(self):
        result = verify_with_budget(SAFE_SIMPLE, max_predicates=10)
        assert result.safe
        assert result.minimal_predicates <= 10

    def test_tight_budget(self):
        result = verify_with_budget(SAFE_SIMPLE, max_predicates=1)
        assert result.safe  # Simple program should work with 1 predicate


# ===== Section 13: analyze_predicate_quality =====

class TestAnalyzePredicateQuality:
    def test_quality_safe(self):
        q = analyze_predicate_quality(SAFE_MULTI_PRED)
        assert q.total >= 1
        assert q.essential >= 1
        assert q.essential <= q.total
        assert 0.0 <= q.ratio <= 1.0

    def test_quality_summary(self):
        q = analyze_predicate_quality(SAFE_SIMPLE)
        s = q.summary
        assert "Quality" in s or "essential" in s.lower()


# ===== Section 14: IncrementalMinCEGAR =====

class TestIncrementalMinCEGAR:
    def test_first_run(self):
        inc = IncrementalMinCEGAR()
        result = inc.verify(SAFE_SIMPLE)
        assert result.safe

    def test_cached_second_run(self):
        inc = IncrementalMinCEGAR()
        r1 = inc.verify(SAFE_SIMPLE)
        assert r1.safe
        # Second run should use cache
        r2 = inc.verify(SAFE_SIMPLE)
        assert r2.safe
        assert r2.minimization_mode == "cached"
        assert r2.cegar_iterations == 0  # Skipped CEGAR

    def test_history(self):
        inc = IncrementalMinCEGAR()
        inc.verify(SAFE_SIMPLE)
        inc.verify(SAFE_SIMPLE)
        assert len(inc.history) == 2

    def test_clear_cache(self):
        inc = IncrementalMinCEGAR()
        inc.verify(SAFE_SIMPLE)
        inc.clear_cache()
        r = inc.verify(SAFE_SIMPLE)
        # After clearing, should run full CEGAR again
        assert r.safe
        assert r.minimization_mode != "cached"

    def test_different_programs(self):
        inc = IncrementalMinCEGAR()
        r1 = inc.verify(SAFE_SIMPLE)
        assert r1.safe
        # Different program -- cache may or may not work
        r2 = inc.verify(SAFE_ARITHMETIC)
        assert r2.safe


# ===== Section 15: Mode Consistency =====

class TestModeConsistency:
    """All modes should agree on safe/unsafe verdict."""

    def test_all_modes_agree_safe(self):
        post = minimized_cegar_verify(SAFE_SIMPLE, "post_hoc")
        online = minimized_cegar_verify(SAFE_SIMPLE, "online")
        eager = minimized_cegar_verify(SAFE_SIMPLE, "eager")
        assert post.safe and online.safe and eager.safe

    def test_all_modes_agree_unsafe(self):
        post = minimized_cegar_verify(UNSAFE_SIMPLE, "post_hoc")
        online = minimized_cegar_verify(UNSAFE_SIMPLE, "online")
        eager = minimized_cegar_verify(UNSAFE_SIMPLE, "eager")
        assert not post.safe and not online.safe and not eager.safe

    def test_agree_with_v119(self):
        """V125 should agree with raw V119 on verdict."""
        v119 = bdd_verify(SAFE_CONDITIONAL)
        v125 = minimized_cegar_verify(SAFE_CONDITIONAL, "post_hoc")
        assert v119.safe == v125.safe


# ===== Section 16: Predicate Reduction =====

class TestPredicateReduction:
    """Tests focused on actual predicate reduction."""

    def test_minimal_leq_original(self):
        result = minimized_cegar_verify(SAFE_MULTI_PRED, "post_hoc")
        assert result.safe
        assert result.minimal_predicates <= result.original_predicates

    def test_removed_list_consistency(self):
        result = minimized_cegar_verify(SAFE_MULTI_PRED, "post_hoc")
        if result.safe:
            assert (result.minimal_predicates +
                    len(result.removed_predicates)) == result.original_predicates

    def test_reduction_ratio_consistency(self):
        result = minimized_cegar_verify(SAFE_MULTI_PRED, "post_hoc")
        if result.safe and result.original_predicates > 0:
            expected = ((result.original_predicates - result.minimal_predicates)
                        / result.original_predicates)
            assert abs(result.reduction_ratio - expected) < 0.01


# ===== Section 17: MinimizedCEGAR Class Direct Usage =====

class TestMinimizedCEGARClass:
    def test_post_hoc_direct(self):
        mc = MinimizedCEGAR(SAFE_SIMPLE, MinimizationMode.POST_HOC)
        result = mc.verify()
        assert result.safe

    def test_online_direct(self):
        mc = MinimizedCEGAR(SAFE_CONDITIONAL, MinimizationMode.ONLINE)
        result = mc.verify()
        assert result.safe

    def test_eager_direct(self):
        mc = MinimizedCEGAR(SAFE_SIMPLE, MinimizationMode.EAGER)
        result = mc.verify()
        assert result.safe

    def test_custom_params(self):
        mc = MinimizedCEGAR(SAFE_SIMPLE, MinimizationMode.POST_HOC,
                            max_iterations=10, max_nodes=200)
        result = mc.verify()
        assert result.safe


# ===== Section 18: Edge Cases =====

class TestEdgeCases:
    def test_default_mode(self):
        """Default mode parameter should work."""
        result = minimized_cegar_verify(SAFE_SIMPLE)
        assert result.safe

    def test_nested_conditions(self):
        result = minimized_cegar_verify(SAFE_NESTED, "post_hoc")
        assert result.safe

    def test_timing_positive(self):
        result = minimized_cegar_verify(SAFE_SIMPLE, "post_hoc")
        assert result.total_time_ms >= 0
        assert result.cegar_time_ms >= 0
        assert result.minimization_time_ms >= 0
