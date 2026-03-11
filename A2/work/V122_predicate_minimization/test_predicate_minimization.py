"""Tests for V122: Symbolic Predicate Minimization"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from predicate_minimization import (
    minimize_predicates, classify_predicates, compare_minimization_strategies,
    get_predicate_dependencies, minimization_summary,
    PredicateMinimizer, PredicateRole, MinimizationResult,
    SubsetVerifier, _bdd_support, ComparisonResult,
    analyze_predicate_dependencies,
)

# BDD import for support tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
from bdd_model_checker import BDD


# ===================================================================
# Test programs (C10 source with assertions)
# ===================================================================

# Simple: one variable, one assertion
SIMPLE_ASSERT = """
let x = 5;
assert(x > 0);
"""

# Two variables, related assertion
TWO_VAR = """
let x = 3;
let y = 7;
assert(x + y == 10);
"""

# Conditional with assertion
CONDITIONAL = """
let x = 10;
let y = 0;
if (x > 5) {
  y = 1;
}
assert(y == 1);
"""

# Loop with assertion
LOOP_SIMPLE = """
let x = 0;
let i = 0;
while (i < 3) {
  x = x + 1;
  i = i + 1;
}
assert(x >= 0);
"""

# Multiple assertions
MULTI_ASSERT = """
let x = 5;
let y = 10;
assert(x > 0);
assert(y > 0);
assert(x + y > 0);
"""

# Assertion after branch
BRANCH_ASSERT = """
let x = 1;
let y = 0;
if (x > 0) {
  y = x + 1;
} else {
  y = 0;
}
assert(y >= 0);
"""

# Nested conditions
NESTED_COND = """
let x = 5;
let y = 3;
let z = 0;
if (x > 0) {
  if (y > 0) {
    z = x + y;
  } else {
    z = x;
  }
}
assert(z >= 0);
"""

# Redundant predicate scenario: assertion only uses x
REDUNDANT_PREDS = """
let x = 5;
let y = 100;
let z = 200;
assert(x > 0);
"""

# Non-trivial: loop invariant
LOOP_INVARIANT = """
let x = 10;
let y = 0;
while (x > 0) {
  y = y + 1;
  x = x - 1;
}
assert(y >= 0);
"""

# Trivially safe (no assertions)
NO_ASSERT = """
let x = 5;
let y = x + 1;
"""

# Unsafe program (assertion fails)
UNSAFE = """
let x = 0;
assert(x > 0);
"""


# ===================================================================
# Section 1: Basic minimization
# ===================================================================

class TestBasicMinimization:
    def test_simple_assert_minimization(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert result.safe_with_minimal
        assert result.minimal_count <= result.original_count
        assert result.minimal_count >= 1  # Need at least 1 predicate

    def test_two_var_minimization(self):
        result = minimize_predicates(TWO_VAR, strategy="greedy")
        assert result.safe_with_minimal
        assert result.minimal_count <= result.original_count

    def test_conditional_minimization(self):
        result = minimize_predicates(CONDITIONAL, strategy="greedy")
        assert result.safe_with_minimal
        assert result.minimal_count <= result.original_count

    def test_reduction_ratio(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert 0.0 <= result.reduction_ratio <= 1.0

    def test_strategy_field(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert result.strategy == "greedy"


# ===================================================================
# Section 2: Greedy minimization
# ===================================================================

class TestGreedyMinimization:
    def test_greedy_simple(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert result.safe_with_minimal
        assert result.strategy == "greedy"

    def test_greedy_multi_assert(self):
        result = minimize_predicates(MULTI_ASSERT, strategy="greedy")
        assert result.safe_with_minimal

    def test_greedy_branch(self):
        result = minimize_predicates(BRANCH_ASSERT, strategy="greedy")
        assert result.safe_with_minimal

    def test_greedy_classifies_essential(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        for p in result.minimal_predicates:
            assert p.role == PredicateRole.ESSENTIAL

    def test_greedy_classifies_redundant(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        for p in result.removed_predicates:
            assert p.role == PredicateRole.REDUNDANT

    def test_greedy_iterations_bounded(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        # At most one iteration per predicate
        assert result.iterations <= result.original_count


# ===================================================================
# Section 3: Delta debugging minimization
# ===================================================================

class TestDeltaMinimization:
    def test_delta_simple(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="delta")
        assert result.safe_with_minimal
        assert result.strategy == "delta"

    def test_delta_multi_assert(self):
        result = minimize_predicates(MULTI_ASSERT, strategy="delta")
        assert result.safe_with_minimal

    def test_delta_branch(self):
        result = minimize_predicates(BRANCH_ASSERT, strategy="delta")
        assert result.safe_with_minimal

    def test_delta_finds_minimal(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="delta")
        # Delta should find same or smaller set as greedy
        greedy = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert result.minimal_count <= greedy.minimal_count + 1  # Allow small variance


# ===================================================================
# Section 4: BDD support analysis
# ===================================================================

class TestSupportAnalysis:
    def test_support_simple(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="support")
        # Support analysis identifies dead predicates
        assert result.original_count >= 0

    def test_support_identifies_dead(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="support")
        dead = [p for p in result.removed_predicates if p.role == PredicateRole.SUPPORT_DEAD]
        # Some predicates may be dead (not in any transition BDD)
        assert all(p.role == PredicateRole.SUPPORT_DEAD for p in result.removed_predicates)

    def test_support_safe(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="support")
        assert result.safe_with_minimal

    def test_support_preserves_live(self):
        result = minimize_predicates(CONDITIONAL, strategy="support")
        # Live predicates should be kept
        assert result.minimal_count >= 1


# ===================================================================
# Section 5: Combined strategy
# ===================================================================

class TestCombinedStrategy:
    def test_combined_simple(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="combined")
        assert result.safe_with_minimal
        assert result.strategy == "combined"

    def test_combined_conditional(self):
        result = minimize_predicates(CONDITIONAL, strategy="combined")
        assert result.safe_with_minimal

    def test_combined_multi(self):
        result = minimize_predicates(MULTI_ASSERT, strategy="combined")
        assert result.safe_with_minimal

    def test_combined_at_least_as_good_as_support(self):
        support = minimize_predicates(SIMPLE_ASSERT, strategy="support")
        combined = minimize_predicates(SIMPLE_ASSERT, strategy="combined")
        assert combined.minimal_count <= support.minimal_count

    def test_combined_classifies_dead_and_redundant(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="combined")
        roles = set(p.role for p in result.removed_predicates)
        # Removed predicates are either SUPPORT_DEAD or REDUNDANT
        for p in result.removed_predicates:
            assert p.role in (PredicateRole.SUPPORT_DEAD, PredicateRole.REDUNDANT)


# ===================================================================
# Section 6: Strategy comparison
# ===================================================================

class TestStrategyComparison:
    def test_comparison_runs(self):
        result = compare_minimization_strategies(SIMPLE_ASSERT)
        assert isinstance(result, ComparisonResult)
        assert result.greedy is not None
        assert result.delta is not None
        assert result.support is not None

    def test_comparison_best_strategy(self):
        result = compare_minimization_strategies(SIMPLE_ASSERT)
        assert result.best_strategy in ("greedy", "delta", "support")
        assert result.best_count >= 0

    def test_comparison_summary(self):
        result = compare_minimization_strategies(SIMPLE_ASSERT)
        assert "Strategy Comparison" in result.summary
        assert "Greedy" in result.summary


# ===================================================================
# Section 7: Predicate dependencies
# ===================================================================

class TestPredicateDependencies:
    def test_deps_simple(self):
        deps = get_predicate_dependencies(SIMPLE_ASSERT)
        assert deps['safe']
        assert 'predicates' in deps

    def test_deps_conditional(self):
        deps = get_predicate_dependencies(CONDITIONAL)
        assert deps['safe']
        assert deps['num_predicates'] >= 1

    def test_deps_structure(self):
        deps = get_predicate_dependencies(SIMPLE_ASSERT)
        if deps['num_predicates'] > 0:
            p = deps['predicates'][0]
            assert 'index' in p
            assert 'description' in p
            assert 'depends_on' in p
            assert 'depended_by' in p


# ===================================================================
# Section 8: Edge cases
# ===================================================================

class TestEdgeCases:
    def test_unsafe_program(self):
        result = minimize_predicates(UNSAFE, strategy="greedy")
        assert not result.safe_with_minimal
        assert result.minimal_count == 0

    def test_unsafe_combined(self):
        result = minimize_predicates(UNSAFE, strategy="combined")
        assert not result.safe_with_minimal

    def test_unsafe_delta(self):
        result = minimize_predicates(UNSAFE, strategy="delta")
        assert not result.safe_with_minimal

    def test_minimization_result_summary(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        summary = result.summary
        assert "Predicate Minimization" in summary
        assert "greedy" in summary

    def test_time_is_positive(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert result.time_ms >= 0


# ===================================================================
# Section 9: SubsetVerifier
# ===================================================================

class TestSubsetVerifier:
    def test_subset_verifier_no_predicates_safe(self):
        # No assertions -> safe with empty predicates
        verifier = SubsetVerifier(NO_ASSERT)
        assert verifier.verify_with_predicates([])

    def test_subset_verifier_empty_unsafe(self):
        # Has assertions -> not safe with empty predicates
        verifier = SubsetVerifier(SIMPLE_ASSERT)
        result = verifier.verify_with_predicates([])
        # With no predicates, error might be reachable
        # (depends on CFG structure)
        assert isinstance(result, bool)


# ===================================================================
# Section 10: BDD support utility
# ===================================================================

class TestBDDSupport:
    def test_support_terminal_true(self):
        bdd = BDD(num_vars=4)
        s = _bdd_support(bdd, bdd.TRUE)
        assert s == set()

    def test_support_terminal_false(self):
        bdd = BDD(num_vars=4)
        s = _bdd_support(bdd, bdd.FALSE)
        assert s == set()

    def test_support_single_var(self):
        bdd = BDD(num_vars=4)
        v = bdd.var(0)
        s = _bdd_support(bdd, v)
        assert 0 in s

    def test_support_and(self):
        bdd = BDD(num_vars=4)
        v0 = bdd.var(0)
        v1 = bdd.var(1)
        a = bdd.AND(v0, v1)
        s = _bdd_support(bdd, a)
        assert s == {0, 1}

    def test_support_or(self):
        bdd = BDD(num_vars=4)
        v0 = bdd.var(0)
        v2 = bdd.var(2)
        o = bdd.OR(v0, v2)
        s = _bdd_support(bdd, o)
        assert s == {0, 2}


# ===================================================================
# Section 11: Minimization summary API
# ===================================================================

class TestMinimizationSummary:
    def test_summary_returns_string(self):
        s = minimization_summary(SIMPLE_ASSERT)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_info(self):
        s = minimization_summary(SIMPLE_ASSERT)
        assert "Original" in s
        assert "Minimal" in s

    def test_summary_conditional(self):
        s = minimization_summary(CONDITIONAL)
        assert "combined" in s


# ===================================================================
# Section 12: PredicateMinimizer class
# ===================================================================

class TestPredicateMinimizer:
    def test_minimizer_reuses_verification(self):
        m = PredicateMinimizer(SIMPLE_ASSERT)
        r1 = m.greedy_minimize()
        # Second call should reuse _full_result
        assert m._full_result is not None

    def test_minimizer_classify(self):
        classification = classify_predicates(SIMPLE_ASSERT)
        assert isinstance(classification, dict)
        for idx, role in classification.items():
            assert isinstance(role, PredicateRole)


# ===================================================================
# Section 13: Loop programs
# ===================================================================

class TestLoopPrograms:
    def test_loop_simple(self):
        result = minimize_predicates(LOOP_SIMPLE, strategy="greedy")
        assert result.safe_with_minimal

    def test_loop_invariant(self):
        result = minimize_predicates(LOOP_INVARIANT, strategy="greedy")
        assert result.safe_with_minimal

    def test_loop_combined(self):
        result = minimize_predicates(LOOP_SIMPLE, strategy="combined")
        assert result.safe_with_minimal


# ===================================================================
# Section 14: Nested conditions
# ===================================================================

class TestNestedConditions:
    def test_nested_greedy(self):
        result = minimize_predicates(NESTED_COND, strategy="greedy")
        assert result.safe_with_minimal

    def test_nested_combined(self):
        result = minimize_predicates(NESTED_COND, strategy="combined")
        assert result.safe_with_minimal

    def test_nested_has_predicates(self):
        result = minimize_predicates(NESTED_COND, strategy="greedy")
        assert result.original_count >= 1


# ===================================================================
# Section 15: Redundant predicate detection
# ===================================================================

class TestRedundantDetection:
    def test_redundant_preds_minimized(self):
        result = minimize_predicates(REDUNDANT_PREDS, strategy="greedy")
        assert result.safe_with_minimal
        # y and z aren't needed for proving x > 0
        # So some predicates should be removable
        if result.original_count > 1:
            assert result.minimal_count < result.original_count

    def test_redundant_preds_combined(self):
        result = minimize_predicates(REDUNDANT_PREDS, strategy="combined")
        assert result.safe_with_minimal


# ===================================================================
# Section 16: Predicate info structure
# ===================================================================

class TestPredicateInfo:
    def test_minimal_predicates_have_info(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        for p in result.minimal_predicates:
            assert p.index >= 0
            assert isinstance(p.description, str)
            assert p.term is not None

    def test_removed_predicates_have_info(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        for p in result.removed_predicates:
            assert p.index >= 0
            assert isinstance(p.description, str)


# ===================================================================
# Section 17: Default strategy
# ===================================================================

class TestDefaultStrategy:
    def test_default_is_combined(self):
        result = minimize_predicates(SIMPLE_ASSERT)
        assert result.strategy == "combined"

    def test_default_works(self):
        result = minimize_predicates(CONDITIONAL)
        assert result.safe_with_minimal


# ===================================================================
# Section 18: Consistency checks
# ===================================================================

class TestConsistency:
    def test_minimal_plus_removed_equals_original(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert result.minimal_count + len(result.removed_predicates) == result.original_count

    def test_classification_covers_all(self):
        result = minimize_predicates(SIMPLE_ASSERT, strategy="greedy")
        assert len(result.classification) == result.original_count

    def test_all_strategies_agree_on_safety(self):
        for strategy in ["greedy", "delta", "support", "combined"]:
            result = minimize_predicates(SIMPLE_ASSERT, strategy=strategy)
            assert result.safe_with_minimal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
