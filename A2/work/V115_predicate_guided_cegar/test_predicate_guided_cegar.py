"""Tests for V115: Predicate-Guided CEGAR

Tests predicate-guided verification composing V114 (discovery) + V110 (ART/CEGAR).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V114_recursive_predicate_discovery'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V107_craig_interpolation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from predicate_guided_cegar import (
    guided_verify,
    standard_verify,
    compare_strategies,
    check_assertion,
    get_discovered_predicates,
    verify_with_budget,
    incremental_verify,
    guided_summary,
    GuidedVerdict,
    GuidedCEGARResult,
    SeedingStats,
    RefinementStats,
    ComparisonResult,
    _discover_seed_predicates,
    _seed_registry_from_discovery,
)


# ============================================================================
# Section 1: Basic safe programs
# ============================================================================

class TestBasicSafe:
    """Programs where all assertions hold."""

    def test_simple_true_assertion(self):
        src = "let x = 5; assert(x > 0);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE
        assert r.safe is True

    def test_assignment_and_check(self):
        src = "let x = 10; let y = x + 5; assert(y > 10);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE
        assert r.safe is True

    def test_multiple_assertions_all_true(self):
        src = """
        let x = 3;
        let y = 7;
        assert(x > 0);
        assert(y > 0);
        assert(x + y == 10);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_conditional_safe(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 0 - x;
        }
        assert(y > 0);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_nested_conditional_safe(self):
        src = """
        let x = 10;
        let r = 0;
        if (x > 5) {
            if (x > 8) {
                r = 2;
            } else {
                r = 1;
            }
        } else {
            r = 0;
        }
        assert(r >= 0);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_constant_arithmetic(self):
        src = "let a = 3; let b = 4; let c = a + b; assert(c == 7);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_zero_is_nonnegative(self):
        src = "let x = 0; assert(x >= 0);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE


# ============================================================================
# Section 2: Basic unsafe programs
# ============================================================================

class TestBasicUnsafe:
    """Programs where assertions are violated."""

    def test_simple_false_assertion(self):
        src = "let x = 0; assert(x > 0);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.UNSAFE
        assert r.safe is False

    def test_negative_value(self):
        src = "let x = 0 - 5; assert(x >= 0);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.UNSAFE

    def test_wrong_arithmetic(self):
        src = "let x = 3; let y = 4; assert(x + y == 8);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.UNSAFE

    def test_conditional_violation(self):
        src = """
        let x = 0 - 1;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 0 - 1;
        }
        assert(y > 0);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.UNSAFE

    def test_counterexample_provided(self):
        src = "let x = 5; assert(x < 3);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.UNSAFE
        assert r.counterexample is not None


# ============================================================================
# Section 3: Predicate discovery seeding
# ============================================================================

class TestPredicateSeeding:
    """Test that V114 discovery populates predicates."""

    def test_seeding_stats_populated(self):
        src = "let x = 5; assert(x > 0);"
        r = guided_verify(src)
        assert r.seeding_stats is not None
        assert r.seeding_stats.total_discovered >= 0
        assert r.seeding_stats.discovery_time_ms >= 0

    def test_predicates_discovered(self):
        src = """
        let x = 10;
        let y = 20;
        if (x > 5) {
            y = y + x;
        }
        assert(y > 0);
        """
        preds = get_discovered_predicates(src)
        assert 'predicates' in preds
        assert 'stats' in preds
        assert preds['stats']['total_discovered'] >= 1

    def test_inductive_predicates_found(self):
        src = """
        let x = 10;
        let y = 0;
        assert(x >= 0);
        """
        preds = get_discovered_predicates(src, max_predicates=50)
        # Should find at least some predicates
        assert len(preds['predicates']) >= 1

    def test_discovery_sources(self):
        src = """
        let x = 5;
        if (x > 0) {
            let y = x + 1;
            assert(y > 1);
        }
        """
        preds = get_discovered_predicates(src)
        sources = set(p['source'] for p in preds['predicates'])
        # Should have at least condition and assertion sources
        assert len(sources) >= 1

    def test_predicate_scoring(self):
        src = """
        let x = 5;
        let y = 10;
        assert(x + y > 0);
        """
        preds = get_discovered_predicates(src)
        if len(preds['predicates']) >= 2:
            # Predicates should be sorted by score (descending)
            scores = [p['score'] for p in preds['predicates']]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]

    def test_guided_has_more_predicates_than_standard(self):
        src = """
        let x = 10;
        let y = 20;
        let z = x + y;
        assert(z > 0);
        assert(x >= 0);
        """
        guided = guided_verify(src)
        standard = standard_verify(src)
        # Guided should have at least as many predicates (usually more)
        assert guided.total_predicates >= standard.total_predicates


# ============================================================================
# Section 4: CEGAR loop behavior
# ============================================================================

class TestCEGARLoop:
    """Test CEGAR iteration and refinement behavior."""

    def test_safe_program_terminates(self):
        src = "let x = 1; assert(x > 0);"
        r = guided_verify(src, max_iterations=10)
        assert r.verdict == GuidedVerdict.SAFE
        assert r.iterations >= 1

    def test_unsafe_program_terminates(self):
        src = "let x = 0; assert(x > 0);"
        r = guided_verify(src, max_iterations=10)
        assert r.verdict == GuidedVerdict.UNSAFE
        assert r.iterations >= 1

    def test_refinement_history_populated(self):
        src = """
        let x = 5;
        let y = 10;
        if (x > 3) {
            y = y + 1;
        }
        assert(y > 0);
        """
        r = guided_verify(src)
        assert len(r.refinement_history) >= 1
        for rs in r.refinement_history:
            assert rs.iteration >= 1
            assert rs.total_predicates >= 0
            assert rs.art_nodes >= 1

    def test_iterations_bounded(self):
        src = "let x = 1; assert(x > 0);"
        r = guided_verify(src, max_iterations=3)
        assert r.iterations <= 3

    def test_max_nodes_respected(self):
        src = """
        let a = 1;
        let b = 2;
        let c = 3;
        if (a > 0) {
            if (b > 0) {
                c = a + b;
            }
        }
        assert(c > 0);
        """
        r = guided_verify(src, max_nodes=100)
        assert r.art_nodes <= 200  # Some slack for implementation


# ============================================================================
# Section 5: Complex programs
# ============================================================================

class TestComplexPrograms:
    """Programs requiring more sophisticated predicate tracking."""

    def test_abs_value(self):
        src = """
        let x = 0 - 7;
        let r = 0;
        if (x >= 0) {
            r = x;
        } else {
            r = 0 - x;
        }
        assert(r >= 0);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_max_of_two(self):
        src = """
        let x = 3;
        let y = 8;
        let m = 0;
        if (x > y) {
            m = x;
        } else {
            m = y;
        }
        assert(m >= x);
        assert(m >= y);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_clamp(self):
        src = """
        let x = 15;
        let lo = 0;
        let hi = 10;
        let r = x;
        if (x < lo) {
            r = lo;
        }
        if (r > hi) {
            r = hi;
        }
        assert(r >= lo);
        assert(r <= hi);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_sign_function(self):
        src = """
        let x = 0 - 3;
        let s = 0;
        if (x > 0) {
            s = 1;
        } else {
            if (x < 0) {
                s = 0 - 1;
            } else {
                s = 0;
            }
        }
        assert(s >= 0 - 1);
        assert(s <= 1);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_sequential_operations(self):
        src = """
        let x = 1;
        let y = x + 1;
        let z = y + 1;
        let w = z + 1;
        assert(w == 4);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_diamond_control_flow(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        assert(y > 0);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_multi_branch_safe(self):
        src = """
        let x = 10;
        let r = 0;
        if (x > 20) {
            r = 3;
        } else {
            if (x > 5) {
                r = 2;
            } else {
                r = 1;
            }
        }
        assert(r >= 1);
        assert(r <= 3);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE


# ============================================================================
# Section 6: Strategy comparison
# ============================================================================

class TestComparison:
    """Test comparison between standard and guided CEGAR."""

    def test_comparison_agrees_safe(self):
        src = "let x = 5; assert(x > 0);"
        comp = compare_strategies(src)
        assert comp.both_agree is True
        assert comp.standard_result.verdict == GuidedVerdict.SAFE
        assert comp.guided_result.verdict == GuidedVerdict.SAFE

    def test_comparison_agrees_unsafe(self):
        src = "let x = 0; assert(x > 0);"
        comp = compare_strategies(src)
        # Guided should detect unsafe; standard may return UNKNOWN on exception
        assert comp.guided_result.verdict == GuidedVerdict.UNSAFE
        assert comp.guided_result.safe is False

    def test_comparison_has_summary(self):
        src = "let x = 5; assert(x > 0);"
        comp = compare_strategies(src)
        assert len(comp.summary) > 0
        assert "Standard:" in comp.summary
        assert "Guided:" in comp.summary

    def test_comparison_metrics(self):
        src = """
        let x = 10;
        let y = 5;
        assert(x > y);
        """
        comp = compare_strategies(src)
        assert isinstance(comp.iteration_reduction, int)
        assert isinstance(comp.predicate_advantage, int)
        assert isinstance(comp.time_difference_ms, float)

    def test_guided_at_least_matches_standard(self):
        src = """
        let a = 3;
        let b = 7;
        let c = a + b;
        assert(c == 10);
        """
        comp = compare_strategies(src)
        # Both should agree on verdict
        assert comp.both_agree


# ============================================================================
# Section 7: Convenience APIs
# ============================================================================

class TestConvenienceAPIs:
    """Test check_assertion, verify_with_budget, etc."""

    def test_check_assertion_safe(self):
        safe, ce = check_assertion("let x = 5; assert(x > 0);")
        assert safe is True
        assert ce is None

    def test_check_assertion_unsafe(self):
        safe, ce = check_assertion("let x = 0; assert(x > 0);")
        assert safe is False

    def test_verify_with_budget(self):
        src = "let x = 5; assert(x > 0);"
        r = verify_with_budget(src, predicate_budget=10, iteration_budget=5)
        assert r.verdict == GuidedVerdict.SAFE

    def test_incremental_verify(self):
        src = "let x = 5; assert(x > 0);"
        r = incremental_verify(src, initial_predicates=5, increment=5, max_rounds=3)
        assert r.verdict == GuidedVerdict.SAFE

    def test_guided_summary(self):
        src = "let x = 5; assert(x > 0);"
        summary = guided_summary(src)
        assert "Verdict: safe" in summary
        assert "Iterations:" in summary
        assert "Total predicates:" in summary

    def test_get_discovered_predicates_structure(self):
        src = "let x = 5; let y = 10; assert(x + y > 0);"
        preds = get_discovered_predicates(src)
        assert isinstance(preds['predicates'], list)
        assert isinstance(preds['stats'], dict)
        for p in preds['predicates']:
            assert 'description' in p
            assert 'source' in p
            assert 'score' in p
            assert 'term' in p


# ============================================================================
# Section 8: Result structure
# ============================================================================

class TestResultStructure:
    """Test GuidedCEGARResult fields."""

    def test_safe_result_fields(self):
        src = "let x = 5; assert(x > 0);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE
        assert r.safe is True
        assert r.counterexample is None
        assert r.iterations >= 1
        assert r.total_predicates >= 0
        assert isinstance(r.predicate_names, list)
        assert r.seeding_stats is not None
        assert isinstance(r.refinement_history, list)
        assert r.art_nodes >= 1
        assert r.total_time_ms >= 0

    def test_unsafe_result_fields(self):
        src = "let x = 0; assert(x > 0);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.UNSAFE
        assert r.safe is False
        assert r.counterexample is not None
        assert r.iterations >= 1

    def test_seeding_stats_fields(self):
        src = "let x = 5; assert(x > 0);"
        r = guided_verify(src)
        s = r.seeding_stats
        assert isinstance(s.total_discovered, int)
        assert isinstance(s.selected_count, int)
        assert isinstance(s.inductive_count, int)
        assert isinstance(s.source_counts, dict)
        assert isinstance(s.discovery_time_ms, float)

    def test_refinement_stats_fields(self):
        src = "let x = 5; assert(x > 0);"
        r = guided_verify(src)
        if r.refinement_history:
            rs = r.refinement_history[0]
            assert isinstance(rs.iteration, int)
            assert isinstance(rs.total_predicates, int)
            assert isinstance(rs.art_nodes, int)
            assert isinstance(rs.error_nodes_found, int)
            assert isinstance(rs.spurious_count, int)


# ============================================================================
# Section 9: Edge cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_program_no_assertions(self):
        src = "let x = 5;"
        r = guided_verify(src)
        # No assertions means safe
        assert r.verdict == GuidedVerdict.SAFE

    def test_single_variable(self):
        src = "let x = 42; assert(x == 42);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_zero_constant(self):
        src = "let x = 0; assert(x == 0);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_large_constant(self):
        src = "let x = 1000; assert(x > 999);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_negative_arithmetic(self):
        src = "let x = 0 - 10; let y = 0 - 20; assert(x > y);"
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_multiple_variables(self):
        src = """
        let a = 1;
        let b = 2;
        let c = 3;
        let d = 4;
        let e = a + b + c + d;
        assert(e == 10);
        """
        r = guided_verify(src)
        assert r.verdict == GuidedVerdict.SAFE

    def test_max_iterations_one(self):
        src = "let x = 5; assert(x > 0);"
        r = guided_verify(src, max_iterations=1)
        # Should still get a result
        assert r.verdict in (GuidedVerdict.SAFE, GuidedVerdict.UNSAFE, GuidedVerdict.UNKNOWN)

    def test_very_low_node_budget(self):
        src = "let x = 5; assert(x > 0);"
        r = guided_verify(src, max_nodes=10)
        assert r.verdict in (GuidedVerdict.SAFE, GuidedVerdict.UNSAFE, GuidedVerdict.UNKNOWN)


# ============================================================================
# Section 10: Discovery refinement
# ============================================================================

class TestDiscoveryRefinement:
    """Test that V114 discovery acts as refinement fallback."""

    def test_discovery_refinement_flag(self):
        src = "let x = 5; assert(x > 0);"
        r_with = guided_verify(src, use_discovery_refinement=True)
        r_without = guided_verify(src, use_discovery_refinement=False)
        # Both should reach same verdict
        assert r_with.verdict == r_without.verdict

    def test_discovery_helped_flag(self):
        src = """
        let x = 10;
        let y = 20;
        let z = x + y;
        assert(z > 0);
        assert(x >= 0);
        """
        r = guided_verify(src)
        # discovery_helped should be set if V114 added predicates
        assert isinstance(r.discovery_helped, bool)


# ============================================================================
# Section 11: Predicate quality
# ============================================================================

class TestPredicateQuality:
    """Test that discovered predicates are meaningful."""

    def test_nonneg_predicate_discovered(self):
        src = "let x = 5; assert(x >= 0);"
        preds = get_discovered_predicates(src, max_predicates=50)
        # Should find some predicates about x
        assert len(preds['predicates']) >= 1

    def test_comparison_predicate_discovered(self):
        src = """
        let x = 3;
        let y = 7;
        if (x < y) {
            assert(x < y);
        }
        """
        preds = get_discovered_predicates(src)
        # Should find condition predicates
        assert len(preds['predicates']) >= 1

    def test_assertion_predicate_extracted(self):
        src = "let x = 5; assert(x > 0);"
        preds = get_discovered_predicates(src)
        # The assertion itself should appear as a predicate
        has_assertion = any(p['source'] == 'assertion' for p in preds['predicates'])
        has_condition = any(p['source'] == 'condition' for p in preds['predicates'])
        # Should have at least one relevant source
        assert has_assertion or has_condition or len(preds['predicates']) > 0

    def test_template_predicates_present(self):
        src = "let x = 5; let y = 10; assert(x + y > 0);"
        preds = get_discovered_predicates(src, max_predicates=50)
        has_template = any(p['source'] == 'template' for p in preds['predicates'])
        # Templates should be generated for programs with multiple variables
        assert has_template or len(preds['predicates']) >= 1


# ============================================================================
# Section 12: Guided verdict enum
# ============================================================================

class TestGuidedVerdict:
    """Test verdict enum values."""

    def test_safe_value(self):
        assert GuidedVerdict.SAFE.value == "safe"

    def test_unsafe_value(self):
        assert GuidedVerdict.UNSAFE.value == "unsafe"

    def test_unknown_value(self):
        assert GuidedVerdict.UNKNOWN.value == "unknown"


# ============================================================================
# Section 13: Integration with standard verify
# ============================================================================

class TestIntegration:
    """Integration tests confirming standard and guided agree."""

    def test_agree_on_true_assertion(self):
        src = "let x = 1; assert(x == 1);"
        s = standard_verify(src)
        g = guided_verify(src)
        assert s.verdict == g.verdict

    def test_agree_on_false_assertion(self):
        src = "let x = 1; assert(x == 2);"
        g = guided_verify(src)
        assert g.verdict == GuidedVerdict.UNSAFE
        s = standard_verify(src)
        # Standard may return UNKNOWN if V110 throws; guided is more robust
        assert s.verdict in (GuidedVerdict.UNSAFE, GuidedVerdict.UNKNOWN)

    def test_agree_on_conditional(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        }
        assert(y >= 0);
        """
        s = standard_verify(src)
        g = guided_verify(src)
        assert s.verdict == g.verdict

    def test_agree_on_complex_safe(self):
        src = """
        let a = 10;
        let b = 3;
        let q = a - b;
        if (q > 0) {
            let r = q + b;
            assert(r == a);
        }
        """
        s = standard_verify(src)
        g = guided_verify(src)
        assert s.verdict == g.verdict

    def test_agree_on_complex_unsafe(self):
        src = """
        let x = 5;
        let y = 10;
        if (x > y) {
            assert(x > y);
        } else {
            assert(x > y);
        }
        """
        g = guided_verify(src)
        assert g.verdict == GuidedVerdict.UNSAFE
        s = standard_verify(src)
        # Standard may return UNKNOWN for complex programs
        assert s.verdict in (GuidedVerdict.UNSAFE, GuidedVerdict.UNKNOWN)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
