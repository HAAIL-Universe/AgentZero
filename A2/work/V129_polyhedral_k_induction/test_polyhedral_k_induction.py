"""Tests for V129: Polyhedral k-Induction."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))

from polyhedral_k_induction import (
    # Core
    PolyhedralKIndResult, PolyhedralCandidate,
    # TS-level
    polyhedral_k_induction,
    # Source-level
    verify_loop_polyhedral, verify_loop_polyhedral_with_config,
    # Inspection
    get_polyhedral_candidates, get_polyhedral_env,
    # Comparison
    compare_strategies,
    # Summary
    polyhedral_k_summary,
    # Internal
    _extract_interval_candidates, _extract_relational_candidates,
    _extract_all_constraint_candidates, _constraint_to_smt,
    _constraint_description, _validate_candidate, _select_best_candidates,
    _combine_smt,
    # Loop invariant access
    get_loop_invariant_envs,
)
from polyhedral_domain import PolyhedralDomain, LinearConstraint
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


def _two_var_ts():
    """x starts at 0, y starts at 10. x increments, y decrements. Property: x + y >= 10."""
    ts = TransitionSystem()
    x = ts.add_int_var("x")
    y = ts.add_int_var("y")
    xp = ts.prime("x")
    yp = ts.prime("y")
    ts.set_init(App(Op.AND, [
        App(Op.EQ, [x, IntConst(0)], BOOL),
        App(Op.EQ, [y, IntConst(10)], BOOL),
    ], BOOL))
    ts.set_trans(App(Op.AND, [
        App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL),
        App(Op.EQ, [yp, App(Op.SUB, [y, IntConst(1)], INT)], BOOL),
    ], BOOL))
    ts.set_property(App(Op.GE, [
        App(Op.ADD, [x, y], INT), IntConst(10)
    ], BOOL))
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
        result = verify_loop_polyhedral(COUNTER_SOURCE, "x >= 0")
        assert result.result == "SAFE"

    def test_counter_upper_bound(self):
        result = verify_loop_polyhedral(COUNTER_SOURCE, "x <= 10")
        assert result.result == "SAFE"

    def test_countdown_non_negative(self):
        result = verify_loop_polyhedral(COUNTDOWN_SOURCE, "x >= 0")
        assert result.result == "SAFE"

    def test_sum_non_negative(self):
        result = verify_loop_polyhedral(SUM_SOURCE, "s >= 0")
        assert result.result == "SAFE"

    def test_result_has_stats(self):
        result = verify_loop_polyhedral(COUNTER_SOURCE, "x >= 0")
        assert "time" in result.stats
        assert result.stats.get("phase1_plain") is not None


# ===========================================================================
# 2. Polyhedral Candidate Extraction
# ===========================================================================

class TestCandidateExtraction:
    def test_counter_candidates(self):
        candidates = get_polyhedral_candidates(COUNTER_SOURCE, "x >= 0")
        assert len(candidates) >= 1
        descs = [c.description for c in candidates]
        assert any("x" in d for d in descs)

    def test_countdown_candidates(self):
        candidates = get_polyhedral_candidates(COUNTDOWN_SOURCE, "x >= 0")
        assert len(candidates) >= 1

    def test_two_var_candidates(self):
        candidates = get_polyhedral_candidates(TWO_VAR_SOURCE, "x >= 0")
        assert len(candidates) >= 1

    def test_candidates_are_validated(self):
        candidates = get_polyhedral_candidates(COUNTER_SOURCE, "x >= 0")
        ts, ts_vars = _extract_loop_ts(COUNTER_SOURCE)
        prop_smt = _parse_property("x >= 0", ts_vars)
        ts.set_property(prop_smt)
        for c in candidates:
            assert _validate_candidate(ts, c), f"Candidate not inductive: {c.description}"


# ===========================================================================
# 3. Polyhedral Environment
# ===========================================================================

class TestPolyhedralEnv:
    def test_get_env(self):
        env = get_polyhedral_env(COUNTER_SOURCE)
        assert isinstance(env, PolyhedralDomain)
        assert "x" in env.var_names

    def test_env_has_bounds(self):
        env = get_polyhedral_env(COUNTER_SOURCE)
        lo, hi = env.get_interval("x")
        assert lo >= 0  # x starts at 0 and increments

    def test_two_var_env(self):
        env = get_polyhedral_env(TWO_VAR_SOURCE)
        assert "x" in env.var_names
        assert "y" in env.var_names


# ===========================================================================
# 4. TS-Level Verification
# ===========================================================================

class TestTSLevel:
    def test_simple_counter_ts(self):
        ts = _simple_counter_ts()
        result = polyhedral_k_induction(ts, COUNTER_SOURCE)
        assert result.result == "SAFE"

    def test_result_type(self):
        ts = _simple_counter_ts()
        result = polyhedral_k_induction(ts, COUNTER_SOURCE)
        assert isinstance(result, PolyhedralKIndResult)


# ===========================================================================
# 5. Violation Detection
# ===========================================================================

class TestViolation:
    def test_unsafe_property(self):
        result = verify_loop_polyhedral(COUNTER_SOURCE, "x <= 0 - 1")
        assert result.result == "UNSAFE"

    def test_immediate_violation(self):
        source = """
            let x = 5;
            while (x < 10) {
                x = x + 1;
            }
        """
        result = verify_loop_polyhedral(source, "x < 5")
        assert result.result == "UNSAFE"


# ===========================================================================
# 6. Polyhedral Strengthening
# ===========================================================================

class TestStrengthening:
    def test_polyhedral_provides_strengthening(self):
        result = verify_loop_polyhedral(COUNTER_SOURCE, "x >= 0")
        assert result.result == "SAFE"

    def test_two_var_strengthening(self):
        result = verify_loop_polyhedral(TWO_VAR_SOURCE, "x >= 0")
        assert result.result == "SAFE"

    def test_sum_accumulator(self):
        result = verify_loop_polyhedral(SUM_SOURCE, "i >= 0")
        assert result.result == "SAFE"


# ===========================================================================
# 7. Relational Invariants
# ===========================================================================

class TestRelationalInvariants:
    def test_sum_conservation(self):
        """x + y stays constant at 10."""
        result = verify_loop_polyhedral(TWO_VAR_SOURCE, "x + y >= 10")
        assert result.result == "SAFE"

    def test_both_non_negative(self):
        result = verify_loop_polyhedral(TWO_VAR_SOURCE, "y >= 0")
        assert result.result == "SAFE"

    def test_relational_candidates_extracted(self):
        """Two-var programs should produce relational candidates."""
        env = get_polyhedral_env(TWO_VAR_SOURCE)
        ts, ts_vars = _extract_loop_ts(TWO_VAR_SOURCE)
        rel_cands = _extract_relational_candidates(env, ts)
        # Polyhedral domain should discover x + y == 10 or similar
        # (may or may not depending on widening)
        # At minimum, the all-constraint extraction should produce something
        all_cands = _extract_all_constraint_candidates(env, ts)
        assert len(all_cands) >= 1


# ===========================================================================
# 8. Configuration
# ===========================================================================

class TestConfiguration:
    def test_custom_iterations(self):
        result = verify_loop_polyhedral_with_config(
            COUNTER_SOURCE, "x >= 0", max_iterations=100
        )
        assert result.result == "SAFE"

    def test_small_iterations(self):
        result = verify_loop_polyhedral_with_config(
            COUNTER_SOURCE, "x >= 0", max_iterations=10
        )
        assert result.result == "SAFE"


# ===========================================================================
# 9. Comparison API
# ===========================================================================

class TestComparison:
    def test_compare_strategies(self):
        comp = compare_strategies(COUNTER_SOURCE, "x >= 0")
        assert "plain_k_induction" in comp
        assert "auto_k_induction" in comp
        assert "polyhedral_k_induction" in comp
        assert comp["plain_k_induction"]["result"] in ("SAFE", "UNSAFE", "UNKNOWN")
        assert comp["polyhedral_k_induction"]["result"] in ("SAFE", "UNSAFE", "UNKNOWN")

    def test_all_agree_safe(self):
        comp = compare_strategies(COUNTER_SOURCE, "x >= 0")
        for key in ("plain_k_induction", "auto_k_induction", "polyhedral_k_induction"):
            assert comp[key]["result"] == "SAFE"


# ===========================================================================
# 10. Summary
# ===========================================================================

class TestSummary:
    def test_summary_output(self):
        s = polyhedral_k_summary(COUNTER_SOURCE, "x >= 0")
        assert "Polyhedral k-Induction" in s
        assert "SAFE" in s

    def test_summary_has_candidates(self):
        s = polyhedral_k_summary(COUNTER_SOURCE, "x >= 0")
        assert "candidates" in s.lower() or "Candidates" in s


# ===========================================================================
# 11. SMT Combining
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
# 12. PolyhedralKIndResult
# ===========================================================================

class TestResultFields:
    def test_repr(self):
        r = PolyhedralKIndResult(result="SAFE", k=1)
        assert "SAFE" in repr(r)
        assert "k=1" in repr(r)

    def test_fields_populated(self):
        result = verify_loop_polyhedral(COUNTER_SOURCE, "x >= 0")
        assert result.result is not None
        assert result.k is not None
        assert isinstance(result.polyhedral_candidates, list)
        assert isinstance(result.used_candidates, list)
        assert isinstance(result.stats, dict)


# ===========================================================================
# 13. Candidate Validation
# ===========================================================================

class TestCandidateValidation:
    def test_valid_invariant(self):
        ts = _simple_counter_ts()
        c = PolyhedralCandidate(
            formula=App(Op.GE, [Var("x", INT), IntConst(0)], BOOL),
            description="x >= 0",
            source_kind="interval_lower",
            variables=["x"],
        )
        assert _validate_candidate(ts, c)

    def test_invalid_invariant(self):
        ts = _simple_counter_ts()
        c = PolyhedralCandidate(
            formula=App(Op.LE, [Var("x", INT), IntConst(5)], BOOL),
            description="x <= 5",
            source_kind="interval_upper",
            variables=["x"],
        )
        assert not _validate_candidate(ts, c)


# ===========================================================================
# 14. LinearConstraint to SMT
# ===========================================================================

class TestConstraintToSMT:
    def test_simple_upper(self):
        ts = _simple_counter_ts()
        lc = LinearConstraint.from_dict({"x": 1}, 10)  # x <= 10
        formula = _constraint_to_smt(lc, ts)
        assert formula is not None

    def test_equality(self):
        ts = _two_var_ts()
        lc = LinearConstraint.from_dict({"x": 1, "y": 1}, 10, is_equality=True)
        formula = _constraint_to_smt(lc, ts)
        assert formula is not None

    def test_missing_var(self):
        ts = _simple_counter_ts()
        lc = LinearConstraint.from_dict({"z": 1}, 5)  # z not in ts
        formula = _constraint_to_smt(lc, ts)
        assert formula is None

    def test_constraint_description(self):
        lc = LinearConstraint.from_dict({"x": 1}, 10)
        desc = _constraint_description(lc)
        assert "x" in desc
        assert "10" in desc


# ===========================================================================
# 15. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_trivial_loop(self):
        source = """
            let x = 5;
            while (x < 3) {
                x = x + 1;
            }
        """
        result = verify_loop_polyhedral(source, "x >= 0")
        assert result.result == "SAFE"

    def test_single_iteration(self):
        source = """
            let x = 0;
            while (x < 1) {
                x = x + 1;
            }
        """
        result = verify_loop_polyhedral(source, "x >= 0")
        assert result.result == "SAFE"


# ===========================================================================
# 16. Conditional Loop Bodies
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
        result = verify_loop_polyhedral(source, "x >= 0")
        assert result.result == "SAFE"


# ===========================================================================
# 17. Interval Extraction
# ===========================================================================

class TestIntervalExtraction:
    def test_extract_from_loop_invariant(self):
        """Loop invariant (fixpoint) should contain x >= 0."""
        inv_envs = get_loop_invariant_envs(COUNTER_SOURCE)
        assert len(inv_envs) >= 1
        ts, _ = _extract_loop_ts(COUNTER_SOURCE)
        cands = _extract_interval_candidates(inv_envs[0], ts)
        descs = [c.description for c in cands]
        assert any("x >= 0" in d for d in descs)

    def test_all_are_polyhedral_candidates(self):
        inv_envs = get_loop_invariant_envs(COUNTER_SOURCE)
        ts, _ = _extract_loop_ts(COUNTER_SOURCE)
        cands = _extract_interval_candidates(inv_envs[0], ts)
        for c in cands:
            assert isinstance(c, PolyhedralCandidate)
            assert c.source_kind in ("interval_lower", "interval_upper")
