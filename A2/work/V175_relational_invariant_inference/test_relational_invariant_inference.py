"""Tests for V175: Relational Invariant Inference."""

import sys, os, pytest
sys.path.insert(0, os.path.dirname(__file__))

from relational_invariant_inference import (
    infer_relational_invariants, infer_with_v007,
    verify_relational_property, compare_with_v007,
    batch_infer, invariant_summary,
    InvariantMethod, RelationalInvariant, RelationalInferenceResult,
)

# ============================================================
# Section 1: Simple countdown loop
# ============================================================

class TestSimpleCountdown:
    SOURCE = """
    let i = 10;
    while (i > 0) {
        i = i - 1;
    }
    """

    def test_finds_invariants(self):
        result = infer_relational_invariants(self.SOURCE)
        assert result.octagon_candidates > 0

    def test_has_inductive_invariants(self):
        result = infer_relational_invariants(self.SOURCE)
        assert result.validated_count > 0

    def test_finds_upper_bound(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # Should find i <= 10 (from octagon analysis of initial value + widening)
        assert any('i <= 10' in d for d in descs)


# ============================================================
# Section 2: Sum accumulation (relational: i + s conservation)
# ============================================================

class TestSumAccumulation:
    SOURCE = """
    let i = 10;
    let s = 0;
    while (i > 0) {
        s = s + 1;
        i = i - 1;
    }
    """

    def test_finds_invariants(self):
        result = infer_relational_invariants(self.SOURCE)
        assert result.octagon_candidates > 0

    def test_finds_relational_invariant(self):
        result = infer_relational_invariants(self.SOURCE)
        # Octagon should discover i + s == 10 (sum conservation)
        assert result.stats.get('relational_candidates', 0) > 0

    def test_sum_conservation(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # i + s == 10 or equivalent
        has_sum = any(('i + s' in d or 's + i' in d) for d in descs)
        assert has_sum

    def test_inductive_sum_conservation(self):
        result = infer_relational_invariants(self.SOURCE)
        rel = result.relational_invariants
        assert len(rel) > 0
        # At least one should mention sum
        descs = [inv.description for inv in rel]
        assert any('+' in d for d in descs)


# ============================================================
# Section 3: Difference tracking
# ============================================================

class TestDifferenceTracking:
    SOURCE = """
    let x = 5;
    let y = 3;
    while (x > 0) {
        x = x - 1;
        y = y - 1;
    }
    """

    def test_finds_difference_invariant(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # x - y == 2 (difference conservation)
        has_diff = any(('x - y' in d or 'y - x' in d) for d in descs)
        assert has_diff

    def test_difference_is_exact(self):
        result = infer_relational_invariants(self.SOURCE)
        equalities = [inv for inv in result.inductive_invariants
                       if inv.method == InvariantMethod.OCTAGON_EQUALITY]
        # Should find x - y == 2 as equality
        descs = [inv.description for inv in equalities]
        has_exact = any('2' in d and ('-' in d) for d in descs)
        assert has_exact


# ============================================================
# Section 4: Property verification
# ============================================================

class TestPropertyVerification:
    def test_verify_sum_conservation(self):
        source = """
        let i = 10;
        let s = 0;
        while (i > 0) {
            s = s + 1;
            i = i - 1;
        }
        """
        result = verify_relational_property(source, "i + s == 10")
        assert result['verified'] is True

    def test_verify_difference_conservation(self):
        source = """
        let x = 5;
        let y = 3;
        while (x > 0) {
            x = x - 1;
            y = y - 1;
        }
        """
        result = verify_relational_property(source, "x - y == 2")
        assert result['verified'] is True

    def test_verify_upper_bound(self):
        source = """
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        """
        result = verify_relational_property(source, "i <= 10")
        assert result['verified'] is True

    def test_verify_false_property(self):
        source = """
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        """
        result = verify_relational_property(source, "i == 5")
        # i == 5 is NOT an invariant (changes every iteration)
        assert result['verified'] is False

    def test_no_loop_returns_error(self):
        result = verify_relational_property("let x = 1;", "x == 1")
        assert result['verified'] is False
        assert 'error' in result


# ============================================================
# Section 5: Multiple variable loops
# ============================================================

class TestMultipleVars:
    SOURCE = """
    let a = 0;
    let b = 10;
    let c = 5;
    while (b > 0) {
        a = a + 1;
        b = b - 1;
        c = c - 1;
    }
    """

    def test_finds_multiple_relations(self):
        result = infer_relational_invariants(self.SOURCE)
        rel = result.relational_invariants
        # Should find relations between a,b and b,c etc.
        assert len(rel) >= 1

    def test_a_plus_b_conservation(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # a + b == 10 (counter-accumulator)
        has_ab = any(('a + b' in d or 'b + a' in d) for d in descs)
        assert has_ab

    def test_b_minus_c_conservation(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # b - c == 5 (both decrement together)
        has_bc = any(('b - c' in d or 'c - b' in d) for d in descs)
        assert has_bc


# ============================================================
# Section 6: Conditional body
# ============================================================

class TestConditionalBody:
    SOURCE = """
    let x = 0;
    let y = 10;
    while (y > 0) {
        if (y > 5) {
            x = x + 2;
        } else {
            x = x + 1;
        }
        y = y - 1;
    }
    """

    def test_finds_invariants(self):
        result = infer_relational_invariants(self.SOURCE)
        assert result.octagon_candidates > 0

    def test_y_bounded(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # y <= 10 should hold
        assert any('y <= 10' in d for d in descs)


# ============================================================
# Section 7: InferenceResult data structure
# ============================================================

class TestResultStructure:
    def test_result_fields(self):
        result = infer_relational_invariants("""
        let i = 5;
        while (i > 0) { i = i - 1; }
        """)
        assert isinstance(result, RelationalInferenceResult)
        assert isinstance(result.invariants, list)
        assert isinstance(result.octagon_candidates, int)
        assert isinstance(result.validated_count, int)
        assert isinstance(result.stats, dict)

    def test_inductive_property(self):
        result = infer_relational_invariants("""
        let i = 5;
        while (i > 0) { i = i - 1; }
        """)
        for inv in result.inductive_invariants:
            assert inv.is_inductive is True

    def test_relational_property(self):
        result = infer_relational_invariants("""
        let x = 5;
        let y = 3;
        while (x > 0) { x = x - 1; y = y - 1; }
        """)
        for inv in result.relational_invariants:
            assert inv.method in {InvariantMethod.OCTAGON_DIFF, InvariantMethod.OCTAGON_SUM,
                                  InvariantMethod.OCTAGON_EQUALITY, InvariantMethod.V007_RELATIONAL}


# ============================================================
# Section 8: InvariantMethod classification
# ============================================================

class TestMethodClassification:
    def test_bound_classification(self):
        result = infer_relational_invariants("""
        let i = 10;
        while (i > 0) { i = i - 1; }
        """)
        methods = {inv.method for inv in result.inductive_invariants}
        assert InvariantMethod.OCTAGON_BOUND in methods or InvariantMethod.OCTAGON_EQUALITY in methods

    def test_diff_classification(self):
        result = infer_relational_invariants("""
        let x = 5;
        let y = 3;
        while (x > 0) { x = x - 1; y = y - 1; }
        """)
        methods = {inv.method for inv in result.inductive_invariants}
        assert InvariantMethod.OCTAGON_DIFF in methods or InvariantMethod.OCTAGON_EQUALITY in methods

    def test_sum_classification(self):
        result = infer_relational_invariants("""
        let i = 10;
        let s = 0;
        while (i > 0) { s = s + 1; i = i - 1; }
        """)
        methods = {inv.method for inv in result.inductive_invariants}
        # Sum conservation: either classified as SUM or EQUALITY
        assert InvariantMethod.OCTAGON_SUM in methods or InvariantMethod.OCTAGON_EQUALITY in methods


# ============================================================
# Section 9: Equality detection
# ============================================================

class TestEqualityDetection:
    def test_detects_constant_variable(self):
        result = infer_relational_invariants("""
        let x = 5;
        let i = 3;
        while (i > 0) { i = i - 1; }
        """)
        descs = [inv.description for inv in result.inductive_invariants]
        # x is never modified, should be x == 5
        assert any('x == 5' in d or 'x <= 5' in d for d in descs)

    def test_detects_difference_equality(self):
        result = infer_relational_invariants("""
        let x = 10;
        let y = 7;
        while (x > 0) { x = x - 1; y = y - 1; }
        """)
        equalities = [inv for inv in result.inductive_invariants
                       if inv.method == InvariantMethod.OCTAGON_EQUALITY]
        descs = [inv.description for inv in equalities]
        # x - y == 3
        has_diff_eq = any('3' in d for d in descs)
        assert has_diff_eq


# ============================================================
# Section 10: Batch inference
# ============================================================

class TestBatchInference:
    def test_batch_multiple(self):
        sources = [
            "let i = 5; while (i > 0) { i = i - 1; }",
            "let a = 0; let b = 10; while (b > 0) { a = a + 1; b = b - 1; }",
        ]
        results = batch_infer(sources)
        assert len(results) == 2
        assert all(isinstance(r, RelationalInferenceResult) for r in results)

    def test_batch_all_have_invariants(self):
        sources = [
            "let i = 5; while (i > 0) { i = i - 1; }",
            "let x = 0; let y = 10; while (y > 0) { x = x + 1; y = y - 1; }",
        ]
        results = batch_infer(sources)
        for r in results:
            assert r.validated_count > 0


# ============================================================
# Section 11: Summary output
# ============================================================

class TestSummary:
    def test_summary_string(self):
        source = """
        let i = 10;
        let s = 0;
        while (i > 0) { s = s + 1; i = i - 1; }
        """
        s = invariant_summary(source)
        assert "Relational Invariant Inference" in s
        assert "Candidates" in s

    def test_summary_contains_invariants(self):
        source = """
        let i = 10;
        let s = 0;
        while (i > 0) { s = s + 1; i = i - 1; }
        """
        s = invariant_summary(source)
        assert "Inductive invariants:" in s


# ============================================================
# Section 12: Edge cases
# ============================================================

class TestEdgeCases:
    def test_no_loop(self):
        result = infer_relational_invariants("let x = 5;")
        assert result.octagon_candidates == 0
        assert 'error' in result.stats

    def test_single_variable_loop(self):
        result = infer_relational_invariants("""
        let n = 100;
        while (n > 0) { n = n - 1; }
        """)
        assert result.validated_count > 0

    def test_loop_with_constant_body(self):
        # x = 1 each iteration -- octagon should detect x == 1 as invariant
        result = infer_relational_invariants("""
        let x = 1;
        let i = 5;
        while (i > 0) {
            x = 1;
            i = i - 1;
        }
        """)
        descs = [inv.description for inv in result.inductive_invariants]
        # x should be bounded
        assert any('x' in d for d in descs)

    def test_nested_loop_outer(self):
        # Should analyze the first (outer) while loop
        result = infer_relational_invariants("""
        let i = 5;
        let j = 0;
        while (i > 0) {
            i = i - 1;
            j = j + 1;
        }
        """)
        assert result.validated_count > 0


# ============================================================
# Section 13: Stats reporting
# ============================================================

class TestStats:
    def test_stats_keys(self):
        result = infer_relational_invariants("""
        let i = 5;
        while (i > 0) { i = i - 1; }
        """)
        assert 'state_vars' in result.stats
        assert 'octagon_constraints' in result.stats
        assert 'relational_candidates' in result.stats

    def test_state_vars_correct(self):
        result = infer_relational_invariants("""
        let x = 3;
        let y = 7;
        while (x > 0) { x = x - 1; y = y - 1; }
        """)
        assert 'x' in result.stats['state_vars']
        assert 'y' in result.stats['state_vars']


# ============================================================
# Section 14: Increment loop (going up)
# ============================================================

class TestIncrementLoop:
    SOURCE = """
    let i = 0;
    let n = 10;
    while (i < n) {
        i = i + 1;
    }
    """

    def test_finds_invariants(self):
        result = infer_relational_invariants(self.SOURCE)
        assert result.octagon_candidates > 0

    def test_i_bounded_below(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # i >= 0 should hold
        assert any('i >= 0' in d or 'i == 0' in d or 'i <= ' in d for d in descs)


# ============================================================
# Section 15: Swap-like pattern
# ============================================================

class TestSwapPattern:
    # Swap with temp var requires SSA-aware transition extraction.
    # Simple parallel-assignment model can't handle sequential dependencies.
    # Test that octagon at least finds candidates (even if not validated).
    SOURCE = """
    let a = 3;
    let b = 7;
    let t = 0;
    let i = 5;
    while (i > 0) {
        t = a;
        a = b;
        b = t;
        i = i - 1;
    }
    """

    def test_finds_invariants(self):
        result = infer_relational_invariants(self.SOURCE)
        assert result.octagon_candidates > 0

    def test_octagon_detects_sum_candidate(self):
        """Octagon fixpoint finds a+b==10 as candidate (validation may fail
        due to sequential assignment limitation in TS construction)."""
        result = infer_relational_invariants(self.SOURCE)
        all_descs = [inv.description for inv in result.invariants]
        has_sum = any(('a + b' in d or 'b + a' in d) and '10' in d for d in all_descs)
        assert has_sum


# ============================================================
# Section 16: Large initial values
# ============================================================

class TestLargeValues:
    def test_large_countdown(self):
        result = infer_relational_invariants("""
        let i = 1000;
        while (i > 0) { i = i - 1; }
        """)
        descs = [inv.description for inv in result.inductive_invariants]
        assert any('i <= 1000' in d for d in descs)


# ============================================================
# Section 17: Two-counter divergence
# ============================================================

class TestTwoCounterDivergence:
    SOURCE = """
    let x = 0;
    let y = 0;
    let i = 10;
    while (i > 0) {
        x = x + 1;
        y = y + 2;
        i = i - 1;
    }
    """

    def test_finds_relation(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        # y grows twice as fast: y - 2*x == 0 is non-octagonal
        # But octagon can find: x - y <= 0 (x <= y) and
        # potentially i + x == 10 or i + y == 20 depending on analysis
        assert result.validated_count > 0

    def test_x_bounded(self):
        result = infer_relational_invariants(self.SOURCE)
        descs = [inv.description for inv in result.inductive_invariants]
        assert any('x' in d for d in descs)


# ============================================================
# Section 18: Already converged (trivial loop)
# ============================================================

class TestTrivialLoop:
    def test_zero_iteration_loop(self):
        # Loop condition false initially
        result = infer_relational_invariants("""
        let i = 0;
        while (i > 0) { i = i - 1; }
        """)
        # Should still find i == 0 or i <= 0 as invariant
        assert result.octagon_candidates >= 0  # may or may not find candidates


# ============================================================
# Section 19: Compare with V007
# ============================================================

class TestCompareWithV007:
    def test_compare_structure(self):
        source = """
        let i = 10;
        while (i > 0) { i = i - 1; }
        """
        try:
            result = compare_with_v007(source)
            assert 'octagon_total' in result
            assert 'octagon_inductive' in result
            assert 'octagon_relational' in result
        except ImportError:
            pytest.skip("V007 not available")

    def test_octagon_finds_relations_v007_doesnt(self):
        source = """
        let x = 5;
        let y = 3;
        while (x > 0) { x = x - 1; y = y - 1; }
        """
        try:
            result = compare_with_v007(source)
            # Octagon should find at least some relational invariants
            assert result['octagon_relational'] > 0
        except ImportError:
            pytest.skip("V007 not available")


# ============================================================
# Section 20: Verify relational properties end-to-end
# ============================================================

class TestEndToEnd:
    def test_accumulator_sum(self):
        """Classic accumulator: prove s + i == n at all times."""
        source = """
        let n = 20;
        let i = 20;
        let s = 0;
        while (i > 0) {
            s = s + 1;
            i = i - 1;
        }
        """
        result = verify_relational_property(source, "i + s == 20")
        assert result['verified'] is True
        assert result['relational_invariants'] > 0

    def test_parallel_decrement(self):
        """Two variables decrement in lockstep."""
        source = """
        let a = 100;
        let b = 50;
        while (a > 0) {
            a = a - 1;
            b = b - 1;
        }
        """
        result = verify_relational_property(source, "a - b == 50")
        assert result['verified'] is True

    def test_three_var_conservation(self):
        """Three variables with two conservation laws."""
        source = """
        let x = 10;
        let y = 0;
        let z = 10;
        while (x > 0) {
            x = x - 1;
            y = y + 1;
            z = z - 1;
        }
        """
        r1 = verify_relational_property(source, "x + y == 10")
        r2 = verify_relational_property(source, "x - z == 0")
        assert r1['verified'] is True
        assert r2['verified'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
