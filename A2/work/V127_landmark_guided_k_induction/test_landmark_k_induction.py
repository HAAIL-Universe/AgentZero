"""Tests for V127: Landmark-Guided k-Induction."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V016_auto_strengthened_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))

from landmark_k_induction import (
    # Core
    LandmarkKIndResult, LandmarkCandidate,
    # TS-level
    landmark_k_induction,
    # Source-level
    verify_loop_landmark, verify_loop_landmark_with_config,
    # Inspection
    get_landmark_candidates,
    # Comparison
    compare_strategies,
    # Summary
    landmark_k_summary,
    # Internal
    _extract_candidates_from_profile, _extract_candidates_from_analysis,
    _validate_candidate, _select_best_candidates, _combine_smt,
)
from landmark_widening import LandmarkConfig
from k_induction import _extract_loop_ts
from auto_k_induction import _parse_property
from pdr import TransitionSystem
from smt_solver import Var, IntConst, BoolConst, App, Op, INT, BOOL


# ---- Helper: build TS manually ----

def _simple_counter_ts():
    """x starts at 0, increments by 1, property: x >= 0."""
    ts = TransitionSystem()
    x = ts.add_int_var("x")
    xp = ts.prime("x")
    ts.set_init(App(Op.EQ, [x, IntConst(0)], BOOL))
    ts.set_trans(App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL))
    ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
    return ts


def _countdown_ts():
    """x starts at 10, decrements by 1 while x > 0. Property: x >= 0."""
    ts = TransitionSystem()
    x = ts.add_int_var("x")
    xp = ts.prime("x")
    cond = App(Op.GT, [x, IntConst(0)], BOOL)
    trans = App(Op.AND, [
        cond,
        App(Op.EQ, [xp, App(Op.SUB, [x, IntConst(1)], INT)], BOOL)
    ], BOOL)
    ts.set_init(App(Op.EQ, [x, IntConst(10)], BOOL))
    ts.set_trans(trans)
    ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
    return ts


# C10 source snippets
COUNTER_SOURCE = """
let x = 0;
while (x < 10) {
    x = x + 1;
}
"""

COUNTDOWN_SOURCE = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""

SUM_SOURCE = """
let i = 0;
let s = 0;
while (i < 5) {
    s = s + i;
    i = i + 1;
}
"""

TWO_VAR_SOURCE = """
let x = 0;
let y = 10;
while (x < 10) {
    x = x + 1;
    y = y - 1;
}
"""


# ===========================================================================
# 1. Basic Source-Level Verification
# ===========================================================================

class TestBasicVerification:
    def test_counter_safe(self):
        result = verify_loop_landmark(COUNTER_SOURCE, "x >= 0")
        assert result.result == "SAFE"

    def test_counter_upper_bound(self):
        result = verify_loop_landmark(COUNTER_SOURCE, "x <= 10")
        assert result.result == "SAFE"

    def test_countdown_non_negative(self):
        result = verify_loop_landmark(COUNTDOWN_SOURCE, "x >= 0")
        assert result.result == "SAFE"

    def test_sum_non_negative(self):
        result = verify_loop_landmark(SUM_SOURCE, "s >= 0")
        assert result.result == "SAFE"

    def test_result_has_stats(self):
        result = verify_loop_landmark(COUNTER_SOURCE, "x >= 0")
        assert "time" in result.stats
        assert result.stats.get("phase1_plain") is not None


# ===========================================================================
# 2. Landmark Candidate Extraction
# ===========================================================================

class TestCandidateExtraction:
    def test_counter_candidates(self):
        candidates = get_landmark_candidates(COUNTER_SOURCE, "x >= 0")
        assert len(candidates) >= 1
        descs = [c.description for c in candidates]
        # Should have at least init bound (x >= 0) or condition bound (x <= 10)
        assert any("x" in d for d in descs)

    def test_countdown_candidates(self):
        candidates = get_landmark_candidates(COUNTDOWN_SOURCE, "x >= 0")
        assert len(candidates) >= 1

    def test_two_var_candidates(self):
        candidates = get_landmark_candidates(TWO_VAR_SOURCE, "x >= 0")
        assert len(candidates) >= 1

    def test_candidates_are_validated(self):
        """All returned candidates should be inductive invariants."""
        candidates = get_landmark_candidates(COUNTER_SOURCE, "x >= 0")
        ts, ts_vars = _extract_loop_ts(COUNTER_SOURCE)
        prop_smt = _parse_property("x >= 0", ts_vars)
        ts.set_property(prop_smt)
        for c in candidates:
            assert _validate_candidate(ts, c), f"Candidate not inductive: {c.description}"


# ===========================================================================
# 3. TS-Level Verification
# ===========================================================================

class TestTSLevel:
    def test_simple_counter_ts(self):
        ts = _simple_counter_ts()
        result = landmark_k_induction(ts, COUNTER_SOURCE)
        assert result.result == "SAFE"

    def test_countdown_ts(self):
        ts = _countdown_ts()
        result = landmark_k_induction(ts, COUNTDOWN_SOURCE)
        assert result.result == "SAFE"

    def test_result_type(self):
        ts = _simple_counter_ts()
        result = landmark_k_induction(ts, COUNTER_SOURCE)
        assert isinstance(result, LandmarkKIndResult)


# ===========================================================================
# 4. Violation Detection
# ===========================================================================

class TestViolation:
    def test_unsafe_property(self):
        """x starts at 0 and goes up -- x <= -1 should be UNSAFE."""
        result = verify_loop_landmark(COUNTER_SOURCE, "x <= 0 - 1")
        assert result.result == "UNSAFE"

    def test_immediate_violation(self):
        source = """
            let x = 5;
            while (x < 10) {
                x = x + 1;
            }
        """
        result = verify_loop_landmark(source, "x < 5")
        assert result.result == "UNSAFE"


# ===========================================================================
# 5. Strengthening with Landmarks
# ===========================================================================

class TestStrengthening:
    def test_landmark_provides_strengthening(self):
        """For non-trivially-inductive properties, landmarks should provide candidates."""
        result = verify_loop_landmark(COUNTER_SOURCE, "x >= 0")
        # Either plain k-induction succeeds (phase1), or landmarks help
        assert result.result == "SAFE"

    def test_two_var_strengthening(self):
        result = verify_loop_landmark(TWO_VAR_SOURCE, "x >= 0")
        assert result.result == "SAFE"

    def test_sum_accumulator(self):
        result = verify_loop_landmark(SUM_SOURCE, "i >= 0")
        assert result.result == "SAFE"


# ===========================================================================
# 6. Configuration
# ===========================================================================

class TestConfiguration:
    def test_custom_config(self):
        config = LandmarkConfig(max_iterations=50, delay_iterations=1)
        result = verify_loop_landmark_with_config(COUNTER_SOURCE, "x >= 0", config)
        assert result.result == "SAFE"

    def test_narrowing_config(self):
        config = LandmarkConfig(narrowing_iterations=8)
        result = verify_loop_landmark_with_config(COUNTDOWN_SOURCE, "x >= 0", config)
        assert result.result == "SAFE"


# ===========================================================================
# 7. Comparison API
# ===========================================================================

class TestComparison:
    def test_compare_strategies(self):
        comp = compare_strategies(COUNTER_SOURCE, "x >= 0")
        assert "plain_k_induction" in comp
        assert "auto_k_induction" in comp
        assert "landmark_k_induction" in comp
        assert comp["plain_k_induction"]["result"] in ("SAFE", "UNSAFE", "UNKNOWN")
        assert comp["landmark_k_induction"]["result"] in ("SAFE", "UNSAFE", "UNKNOWN")

    def test_all_agree_safe(self):
        comp = compare_strategies(COUNTER_SOURCE, "x >= 0")
        # All strategies should agree this is SAFE
        for key in ("plain_k_induction", "auto_k_induction", "landmark_k_induction"):
            assert comp[key]["result"] == "SAFE"


# ===========================================================================
# 8. Summary
# ===========================================================================

class TestSummary:
    def test_summary_output(self):
        s = landmark_k_summary(COUNTER_SOURCE, "x >= 0")
        assert "Landmark-Guided k-Induction" in s
        assert "SAFE" in s

    def test_summary_has_candidates(self):
        s = landmark_k_summary(COUNTER_SOURCE, "x >= 0")
        assert "candidates" in s.lower()


# ===========================================================================
# 9. SMT Combining
# ===========================================================================

class TestSMTCombine:
    def test_combine_none(self):
        assert _combine_smt([]) is None

    def test_combine_single(self):
        t = App(Op.GE, [Var("x", INT), IntConst(0)], BOOL)
        assert _combine_smt([t]) is t

    def test_combine_multiple(self):
        t1 = App(Op.GE, [Var("x", INT), IntConst(0)], BOOL)
        t2 = App(Op.LE, [Var("x", INT), IntConst(10)], BOOL)
        combined = _combine_smt([t1, t2])
        assert combined is not None


# ===========================================================================
# 10. LandmarkKIndResult
# ===========================================================================

class TestResultFields:
    def test_repr(self):
        r = LandmarkKIndResult(result="SAFE", k=1)
        assert "SAFE" in repr(r)
        assert "k=1" in repr(r)

    def test_fields_populated(self):
        result = verify_loop_landmark(COUNTER_SOURCE, "x >= 0")
        assert result.result is not None
        assert result.k is not None
        assert isinstance(result.landmark_candidates, list)
        assert isinstance(result.used_candidates, list)
        assert isinstance(result.stats, dict)


# ===========================================================================
# 11. Candidate Validation
# ===========================================================================

class TestCandidateValidation:
    def test_valid_invariant(self):
        ts = _simple_counter_ts()
        c = LandmarkCandidate(
            formula=App(Op.GE, [ts.var("x"), IntConst(0)], BOOL),
            description="x >= 0",
            source_kind="init_bound",
            variable="x",
        )
        assert _validate_candidate(ts, c)

    def test_invalid_invariant(self):
        ts = _simple_counter_ts()
        c = LandmarkCandidate(
            formula=App(Op.LE, [ts.var("x"), IntConst(5)], BOOL),
            description="x <= 5",
            source_kind="threshold",
            variable="x",
        )
        # x goes to infinity, so x <= 5 is not inductive
        assert not _validate_candidate(ts, c)


# ===========================================================================
# 12. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_trivial_loop(self):
        """Loop that doesn't change anything."""
        source = """
            let x = 5;
            while (x < 3) {
                x = x + 1;
            }
        """
        result = verify_loop_landmark(source, "x >= 0")
        assert result.result == "SAFE"

    def test_single_iteration(self):
        source = """
            let x = 0;
            while (x < 1) {
                x = x + 1;
            }
        """
        result = verify_loop_landmark(source, "x >= 0")
        assert result.result == "SAFE"


# ===========================================================================
# 13. Nested Loops
# ===========================================================================

class TestNestedLoops:
    def test_nested_counter(self):
        source = """
            let i = 0;
            while (i < 3) {
                let j = 0;
                while (j < 3) {
                    j = j + 1;
                }
                i = i + 1;
            }
        """
        result = verify_loop_landmark(source, "i >= 0")
        assert result.result == "SAFE"


# ===========================================================================
# 14. Conditional Loop Bodies
# ===========================================================================

class TestConditionalBodies:
    def test_if_in_loop(self):
        source = """
            let x = 0;
            let y = 0;
            while (x < 10) {
                if (x < 5) {
                    y = y + 1;
                }
                x = x + 1;
            }
        """
        result = verify_loop_landmark(source, "x >= 0")
        assert result.result == "SAFE"

    def test_branch_landmark(self):
        source = """
            let x = 0;
            while (x < 20) {
                if (x < 10) {
                    x = x + 2;
                } else {
                    x = x + 1;
                }
            }
        """
        result = verify_loop_landmark(source, "x >= 0")
        assert result.result == "SAFE"


# ===========================================================================
# 15. Multi-Variable Properties
# ===========================================================================

class TestMultiVariable:
    def test_sum_conservation(self):
        """x + y stays constant."""
        result = verify_loop_landmark(TWO_VAR_SOURCE, "x + y >= 10")
        assert result.result == "SAFE"

    def test_both_non_negative(self):
        result = verify_loop_landmark(TWO_VAR_SOURCE, "y >= 0")
        assert result.result == "SAFE"
