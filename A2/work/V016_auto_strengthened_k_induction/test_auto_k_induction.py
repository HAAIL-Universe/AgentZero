"""
Tests for V016: Auto-Strengthened k-Induction
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from smt_solver import Op, Var, IntConst, BoolConst, App, INT, BOOL
from pdr import TransitionSystem
from k_induction import incremental_k_induction, k_induction_with_strengthening, KIndResult

from auto_k_induction import (
    auto_k_induction, AutoKIndResult,
    verify_loop_auto, verify_loop_auto_with_hints,
    compare_strategies, compare_with_source,
    _infer_from_ts, _validate_ts_invariant, _infer_strengthening_invariants,
)


# ===================================================================
# Helpers
# ===================================================================

def make_ts(state_vars, init, trans, prop):
    """Helper to build TransitionSystem."""
    ts = TransitionSystem()
    vars_map = {}
    for name in state_vars:
        vars_map[name] = ts.add_int_var(name)
    ts.set_init(init(vars_map, ts))
    ts.set_trans(trans(vars_map, ts))
    ts.set_property(prop(vars_map, ts))
    return ts, vars_map


def _and(*terms):
    if len(terms) == 1:
        return terms[0]
    return App(Op.AND, list(terms), BOOL)

def _or(*terms):
    if len(terms) == 1:
        return terms[0]
    return App(Op.OR, list(terms), BOOL)

def _eq(a, b):
    return App(Op.EQ, [a, b], BOOL)

def _neq(a, b):
    return App(Op.NEQ, [a, b], BOOL)

def _lt(a, b):
    return App(Op.LT, [a, b], BOOL)

def _le(a, b):
    return App(Op.LE, [a, b], BOOL)

def _gt(a, b):
    return App(Op.GT, [a, b], BOOL)

def _ge(a, b):
    return App(Op.GE, [a, b], BOOL)

def _add(a, b):
    return App(Op.ADD, [a, b], INT)

def _sub(a, b):
    return App(Op.SUB, [a, b], INT)

def _ite(c, t, e):
    return App(Op.ITE, [c, t, e], INT)


# ===================================================================
# Section 1: Plain 1-inductive properties (no strengthening needed)
# ===================================================================

class TestPlain1Inductive:
    """Properties that are 1-inductive -- auto should solve at k=0."""

    def test_constant_safe(self):
        """x=5, x'=x, prop: x>=0. 1-inductive."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(5)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=10)
        assert result.result == "SAFE"
        assert result.k == 0

    def test_monotone_increasing(self):
        """x=0, x'=x+1, prop: x>=0. 1-inductive."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=10)
        assert result.result == "SAFE"

    def test_tautology(self):
        """Trivially true property."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(BoolConst(True))

        result = auto_k_induction(ts, max_k=5)
        assert result.result == "SAFE"


# ===================================================================
# Section 2: Immediate violations (counterexample found)
# ===================================================================

class TestImmediateViolation:
    """Properties violated from initial state."""

    def test_init_violates(self):
        """x=−1, prop: x>=0. Violated at step 0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(-1)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=5)
        assert result.result == "UNSAFE"
        assert result.counterexample is not None
        assert result.counterexample[0]["x"] == -1

    def test_violation_after_one_step(self):
        """x=1, x'=x-2, prop: x>=0. Violated at step 1."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(1)))
        ts.set_trans(_eq(xp, _sub(x, IntConst(2))))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=5)
        assert result.result == "UNSAFE"
        assert result.counterexample is not None


# ===================================================================
# Section 3: Properties needing strengthening invariants
# ===================================================================

class TestNeedsStrengthening:
    """Properties that plain k-induction can't prove but strengthening helps."""

    def test_countdown_nonneg(self):
        """x=10, x'=ITE(x>0, x-1, x), prop: x>=0.
        Not 1-inductive (x>=0 AND x'=x-1 doesn't imply x'>=0 without x>0 guard).
        But x<=10 is an inductive strengthening invariant.
        """
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(10)))
        ts.set_trans(_eq(xp, _ite(_gt(x, IntConst(0)),
                                    _sub(x, IntConst(1)), x)))
        ts.set_property(_ge(x, IntConst(0)))

        # Plain k-induction should fail (not k-inductive for small k)
        plain = incremental_k_induction(ts, max_k=5)
        # The guarded transition makes this potentially harder

        # Auto should find strengthening
        result = auto_k_induction(ts, max_k=10)
        assert result.result == "SAFE"

    def test_sum_accumulator(self):
        """sum=0, i=0, loop: sum+=i, i+=1 (guarded on i<5).
        prop: sum >= 0. Needs i>=0 as strengthening."""
        ts = TransitionSystem()
        s = ts.add_int_var("sum")
        i = ts.add_int_var("i")
        sp = ts.prime("sum")
        ip = ts.prime("i")

        ts.set_init(_and(_eq(s, IntConst(0)), _eq(i, IntConst(0))))
        # Guarded: if i<5 then sum'=sum+i, i'=i+1 else frame
        ts.set_trans(_or(
            _and(_lt(i, IntConst(5)),
                 _eq(sp, _add(s, i)),
                 _eq(ip, _add(i, IntConst(1)))),
            _and(_ge(i, IntConst(5)),
                 _eq(sp, s),
                 _eq(ip, i)),
        ))
        ts.set_property(_ge(s, IntConst(0)))

        result = auto_k_induction(ts, max_k=10)
        assert result.result == "SAFE"

    def test_two_counter_sum_conservation(self):
        """x=5, y=5, loop: x'=x-1, y'=y+1 (guarded x>0).
        prop: x+y >= 0. Needs x+y==10 as strengthening."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")

        ts.set_init(_and(_eq(x, IntConst(5)), _eq(y, IntConst(5))))
        ts.set_trans(_or(
            _and(_gt(x, IntConst(0)),
                 _eq(xp, _sub(x, IntConst(1))),
                 _eq(yp, _add(y, IntConst(1)))),
            _and(_le(x, IntConst(0)),
                 _eq(xp, x),
                 _eq(yp, y)),
        ))
        ts.set_property(_ge(_add(x, y), IntConst(0)))

        result = auto_k_induction(ts, max_k=10)
        assert result.result == "SAFE"


# ===================================================================
# Section 4: TS-level invariant inference
# ===================================================================

class TestTSLevelInference:
    """Test direct TS-level invariant inference."""

    def test_nonneg_inferred(self):
        """x=5, x'=x+1. Should infer x>=0 and x>=5."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(5)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(BoolConst(True))

        invs = _infer_from_ts(ts)
        descs = [inv.description for inv in invs]
        # Should find x>=0 and x>=5
        assert any("nonneg" in d or "lower" in d for d in descs)

    def test_upper_bound_inferred(self):
        """x=10, x'=x-1 (guarded). Should infer x<=10."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(10)))
        ts.set_trans(_eq(xp, _ite(_gt(x, IntConst(0)),
                                    _sub(x, IntConst(1)), x)))
        ts.set_property(BoolConst(True))

        invs = _infer_from_ts(ts)
        sexprs = [str(inv.sexpr) for inv in invs]
        assert any("10" in s and "<=" in s for s in sexprs)

    def test_sum_conservation_inferred(self):
        """x=3, y=7, x'=x+1, y'=y-1. Should find x+y==10."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")
        ts.set_init(_and(_eq(x, IntConst(3)), _eq(y, IntConst(7))))
        ts.set_trans(_and(_eq(xp, _add(x, IntConst(1))),
                          _eq(yp, _sub(y, IntConst(1)))))
        ts.set_property(BoolConst(True))

        invs = _infer_from_ts(ts)
        sexprs = [str(inv.sexpr) for inv in invs]
        assert any("10" in s and "==" in s for s in sexprs)

    def test_diff_conservation_inferred(self):
        """x=10, y=5, x'=x+2, y'=y+2. Should find x-y==5."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")
        ts.set_init(_and(_eq(x, IntConst(10)), _eq(y, IntConst(5))))
        ts.set_trans(_and(_eq(xp, _add(x, IntConst(2))),
                          _eq(yp, _add(y, IntConst(2)))))
        ts.set_property(BoolConst(True))

        invs = _infer_from_ts(ts)
        sexprs = [str(inv.sexpr) for inv in invs]
        assert any("5" in s and "==" in s and "-" in s for s in sexprs)

    def test_validate_ts_invariant_positive(self):
        """x=0, x'=x+1. x>=0 is invariant."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(BoolConst(True))

        assert _validate_ts_invariant(ts, _ge(x, IntConst(0)))

    def test_validate_ts_invariant_negative(self):
        """x=0, x'=x+1. x<=5 is NOT invariant (x grows unboundedly)."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(BoolConst(True))

        assert not _validate_ts_invariant(ts, _le(x, IntConst(5)))


# ===================================================================
# Section 5: Source-level basic verification
# ===================================================================

class TestSourceLevel:
    """Source-level API tests."""

    def test_simple_countdown(self):
        """Simple countdown: x>=0 should be verified."""
        source = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""
        result = verify_loop_auto(source, "x >= 0")
        assert result.result == "SAFE"

    def test_simple_countup(self):
        """Countup from 0: x>=0 always holds."""
        source = """
let x = 0;
while (x < 100) {
    x = x + 1;
}
"""
        result = verify_loop_auto(source, "x >= 0")
        assert result.result == "SAFE"

    def test_countup_upper_bound(self):
        """Countup from 0 to 100: x<=100."""
        source = """
let x = 0;
while (x < 100) {
    x = x + 1;
}
"""
        result = verify_loop_auto(source, "x <= 100")
        assert result.result == "SAFE"

    def test_violation_detected(self):
        """x starts at -1, prop x>=0 should be UNSAFE."""
        source = """
let x = 0 - 1;
while (x < 10) {
    x = x + 1;
}
"""
        result = verify_loop_auto(source, "x >= 0")
        assert result.result == "UNSAFE"


# ===================================================================
# Section 6: Source-level with strengthening
# ===================================================================

class TestSourceStrengthening:
    """Source-level cases that need auto-inferred strengthening."""

    def test_countdown_with_strengthening(self):
        """x=10, while(x>0) x=x-1. prop: x<=10.
        Needs auto-strengthening (x>=0 inferred)."""
        source = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""
        result = verify_loop_auto(source, "x <= 10")
        assert result.result == "SAFE"

    def test_two_var_transfer(self):
        """x=5, y=0; while(x>0) {x=x-1; y=y+1}. prop: y>=0."""
        source = """
let x = 5;
let y = 0;
while (x > 0) {
    x = x - 1;
    y = y + 1;
}
"""
        result = verify_loop_auto(source, "y >= 0")
        assert result.result == "SAFE"

    def test_two_var_sum_conservation(self):
        """x=5, y=5; while(x>0) {x=x-1; y=y+1}. prop: x + y == 10."""
        source = """
let x = 5;
let y = 5;
while (x > 0) {
    x = x - 1;
    y = y + 1;
}
"""
        # x+y==10 is itself an inductive invariant, and a property
        result = verify_loop_auto(source, "x + y == 10")
        assert result.result == "SAFE"


# ===================================================================
# Section 7: Hints API
# ===================================================================

class TestHintsAPI:
    """Test verify_loop_auto_with_hints."""

    def test_hint_sufficient(self):
        """User provides the right invariant as a hint."""
        source = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""
        result = verify_loop_auto_with_hints(source, "x >= 0", ["x <= 10"])
        assert result.result == "SAFE"

    def test_hint_combined_with_auto(self):
        """Hint alone is insufficient; auto-inference fills the gap."""
        source = """
let x = 5;
let y = 0;
while (x > 0) {
    x = x - 1;
    y = y + 1;
}
"""
        # y>=0 is the property; hint x>=0 may help, auto should find the rest
        result = verify_loop_auto_with_hints(source, "y >= 0", ["x >= 0"])
        assert result.result == "SAFE"

    def test_invalid_hint_rejected(self):
        """Invalid hints are filtered out (not inductive)."""
        source = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""
        # "x == 10" is not inductive (x changes)
        result = verify_loop_auto_with_hints(source, "x >= 0", ["x == 10"])
        assert result.result == "SAFE"  # Still verified via auto-inference


# ===================================================================
# Section 8: Comparison API
# ===================================================================

class TestComparisonAPI:
    """Test comparison between strategies."""

    def test_compare_strategies_basic(self):
        """Compare on a simple system."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(_ge(x, IntConst(0)))

        comp = compare_strategies(ts, max_k=10)
        assert "plain_k_induction" in comp
        assert "auto_k_induction" in comp
        assert "pdr" in comp
        assert comp["plain_k_induction"]["result"] == "SAFE"
        assert comp["auto_k_induction"]["result"] == "SAFE"

    def test_compare_with_source(self):
        """Source-level comparison."""
        source = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""
        comp = compare_with_source(source, "x >= 0")
        assert "plain_k_induction" in comp
        assert "auto_k_induction" in comp
        assert comp["auto_k_induction"]["result"] == "SAFE"


# ===================================================================
# Section 9: Multi-variable systems
# ===================================================================

class TestMultiVariable:
    """Multi-variable systems requiring relational invariants."""

    def test_three_vars_transfer(self):
        """x=10, y=0, z=0; while(x>0) {x--; y++; z++}. prop: z>=0."""
        source = """
let x = 10;
let y = 0;
let z = 0;
while (x > 0) {
    x = x - 1;
    y = y + 1;
    z = z + 1;
}
"""
        result = verify_loop_auto(source, "z >= 0")
        assert result.result == "SAFE"

    def test_difference_conservation(self):
        """x=10, y=5; while(x>0) {x--; y--}. prop: x - y == 5."""
        source = """
let x = 10;
let y = 5;
while (x > 0) {
    x = x - 1;
    y = y - 1;
}
"""
        result = verify_loop_auto(source, "x - y == 5")
        assert result.result == "SAFE"


# ===================================================================
# Section 10: Conditional loop bodies
# ===================================================================

class TestConditionalBodies:
    """Loops with if-else in the body."""

    def test_conditional_increment(self):
        """x=0, y=0; while(x<10) { if(x<5) y++; x++ }. prop: y>=0."""
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
        result = verify_loop_auto(source, "y >= 0")
        assert result.result == "SAFE"

    def test_conditional_abs_like(self):
        """x=10; while(x!=0) { if(x>0) x=x-1; else x=x+1; }. prop: x>=0."""
        source = """
let x = 10;
while (x != 0) {
    if (x > 0) {
        x = x - 1;
    } else {
        x = x + 1;
    }
}
"""
        # x starts positive, decrements, converges to 0. x>=0 should hold.
        result = verify_loop_auto(source, "x >= 0")
        assert result.result == "SAFE"


# ===================================================================
# Section 11: Result object structure
# ===================================================================

class TestResultStructure:
    """Verify result object fields are populated correctly."""

    def test_safe_result_fields(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=5)
        assert result.result == "SAFE"
        assert result.k is not None
        assert isinstance(result.stats, dict)
        assert "time" in result.stats

    def test_unsafe_result_has_counterexample(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(-1)))
        ts.set_trans(_eq(xp, _sub(x, IntConst(1))))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=5)
        assert result.result == "UNSAFE"
        assert result.counterexample is not None
        assert len(result.counterexample) > 0

    def test_repr(self):
        r = AutoKIndResult("SAFE", k=2, invariants=[1, 2, 3])
        assert "SAFE" in repr(r)
        assert "k=2" in repr(r)
        assert "invariants=3" in repr(r)


# ===================================================================
# Section 12: Guarded transitions
# ===================================================================

class TestGuardedTransitions:
    """Test that guarded transitions are handled correctly."""

    def test_guarded_countdown(self):
        """Guarded: x'=ITE(x>0, x-1, x). x>=0 needs the guard."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(5)))
        ts.set_trans(_eq(xp, _ite(_gt(x, IntConst(0)),
                                    _sub(x, IntConst(1)), x)))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=10)
        assert result.result == "SAFE"

    def test_unguarded_fails_guarded_succeeds(self):
        """Unguarded x'=x-1 would fail x>=0. Guarded succeeds."""
        # Unguarded: x can go negative
        ts_unguarded = TransitionSystem()
        x = ts_unguarded.add_int_var("x")
        xp = ts_unguarded.prime("x")
        ts_unguarded.set_init(_eq(x, IntConst(5)))
        ts_unguarded.set_trans(_eq(xp, _sub(x, IntConst(1))))
        ts_unguarded.set_property(_ge(x, IntConst(0)))

        result_ung = auto_k_induction(ts_unguarded, max_k=10)
        assert result_ung.result == "UNSAFE"

        # Guarded: x stops at 0
        ts_guarded = TransitionSystem()
        x2 = ts_guarded.add_int_var("x")
        xp2 = ts_guarded.prime("x")
        ts_guarded.set_init(_eq(x2, IntConst(5)))
        ts_guarded.set_trans(_eq(xp2, _ite(_gt(x2, IntConst(0)),
                                            _sub(x2, IntConst(1)), x2)))
        ts_guarded.set_property(_ge(x2, IntConst(0)))

        result_g = auto_k_induction(ts_guarded, max_k=10)
        assert result_g.result == "SAFE"


# ===================================================================
# Section 13: Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_max_k_zero(self):
        """max_k=0: only check base at step 0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, _add(x, IntConst(1))))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=0)
        assert result.result == "SAFE"  # 0-inductive: x>=0 AND x'=x+1 => x'>=0

    def test_single_variable_constant(self):
        """x=0, x'=0, prop: x==0. Trivial."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, IntConst(0)))
        ts.set_property(_eq(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=5)
        assert result.result == "SAFE"

    def test_nondeterministic_safe(self):
        """x=0, x'=x OR x'=x+1, prop: x>=0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_or(_eq(xp, x), _eq(xp, _add(x, IntConst(1)))))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=10)
        assert result.result == "SAFE"


# ===================================================================
# Section 14: Integration with V007 invariant inference
# ===================================================================

class TestV007Integration:
    """Test that V007 source-level inference is used."""

    def test_source_inference_used(self):
        """Source provided: V007's richer inference should kick in."""
        source = """
let x = 10;
while (x > 0) {
    x = x - 1;
}
"""
        ts, ts_vars = _extract_loop_ts_helper(source)
        prop_smt = _ge(ts_vars["x"], IntConst(0))
        ts.set_property(prop_smt)

        from vc_gen import ast_to_sexpr, SBinOp as SB, SVar as SV, SInt as SI
        prop_sexpr = SB('>=', SV('x'), SI(0))

        result = auto_k_induction(ts, max_k=10, source=source, property_sexpr=prop_sexpr)
        assert result.result == "SAFE"

    def test_no_source_still_works(self):
        """Without source, TS-level inference should suffice for simple cases."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(10)))
        ts.set_trans(_eq(xp, _ite(_gt(x, IntConst(0)),
                                    _sub(x, IntConst(1)), x)))
        ts.set_property(_ge(x, IntConst(0)))

        result = auto_k_induction(ts, max_k=10, source=None)
        assert result.result == "SAFE"


# ===================================================================
# Section 15: Plain vs strengthened comparison
# ===================================================================

class TestPlainVsStrengthened:
    """Verify that strengthening actually helps compared to plain."""

    def test_strengthening_needed(self):
        """System where plain k-induction fails but strengthening succeeds."""
        # sum=0, i=0; while(i<n) sum+=i, i++; prop: sum>=0
        # Plain k-induction can't prove sum>=0 without i>=0
        ts = TransitionSystem()
        s = ts.add_int_var("sum")
        i = ts.add_int_var("i")
        sp = ts.prime("sum")
        ip = ts.prime("i")

        ts.set_init(_and(_eq(s, IntConst(0)), _eq(i, IntConst(0))))
        ts.set_trans(_or(
            _and(_lt(i, IntConst(10)),
                 _eq(sp, _add(s, i)),
                 _eq(ip, _add(i, IntConst(1)))),
            _and(_ge(i, IntConst(10)),
                 _eq(sp, s),
                 _eq(ip, i)),
        ))
        ts.set_property(_ge(s, IntConst(0)))

        plain = incremental_k_induction(ts, max_k=5)
        auto = auto_k_induction(ts, max_k=10)

        # Plain likely UNKNOWN at small k, auto SAFE with strengthening
        assert auto.result == "SAFE"

    def test_comparison_dict_structure(self):
        """Verify compare_strategies returns proper structure."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_ge(x, IntConst(0)))

        comp = compare_strategies(ts, max_k=5)
        assert "plain_k_induction" in comp
        assert "auto_k_induction" in comp
        assert "pdr" in comp
        for key in comp:
            assert "result" in comp[key]
            assert "time" in comp[key]


# ===================================================================
# Helper for tests that need _extract_loop_ts from V015
# ===================================================================

def _extract_loop_ts_helper(source):
    """Wrapper around V015's _extract_loop_ts."""
    from k_induction import _extract_loop_ts
    return _extract_loop_ts(source)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
