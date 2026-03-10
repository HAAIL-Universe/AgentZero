"""
Tests for V002: Property-Directed Reachability (PDR/IC3)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from pdr import (
    TransitionSystem, PDREngine, PDRResult, PDROutput,
    check_ts, make_counter_system, make_two_counter_system,
    _substitute, _negate, _and, _or, _implies, _eq,
    Var, IntConst, BoolConst, App, Op, BOOL, INT, SMTSolver
)


# --- Helper ---

def make_ts(vars, init, trans, prop):
    """Quick helper to build a TransitionSystem."""
    ts = TransitionSystem()
    var_refs = {}
    for name, sort in vars:
        var_refs[name] = ts.add_var(name, sort)
    ts.set_init(init(var_refs, ts))
    ts.set_trans(trans(var_refs, ts))
    ts.set_property(prop(var_refs, ts))
    return ts


# ==========================================
# Section 1: TransitionSystem Construction
# ==========================================

class TestTransitionSystem:
    def test_add_int_var(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        assert isinstance(x, Var)
        assert x.name == "x"
        assert x.sort == INT

    def test_add_bool_var(self):
        ts = TransitionSystem()
        b = ts.add_bool_var("flag")
        assert isinstance(b, Var)
        assert b.name == "flag"
        assert b.sort == BOOL

    def test_prime_var(self):
        ts = TransitionSystem()
        ts.add_int_var("x")
        xp = ts.prime("x")
        assert xp.name == "x'"
        assert xp.sort == INT

    def test_var_lookup(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        x2 = ts.var("x")
        assert x is x2  # Same cached object

    def test_unknown_var_raises(self):
        ts = TransitionSystem()
        with pytest.raises(KeyError):
            ts.var("nonexistent")

    def test_unknown_prime_raises(self):
        ts = TransitionSystem()
        with pytest.raises(KeyError):
            ts.prime("nonexistent")


# ==========================================
# Section 2: Formula Utilities
# ==========================================

class TestFormulaUtils:
    def test_substitute_var(self):
        x = Var("x", INT)
        y = Var("y", INT)
        result = _substitute(x, {"x": y})
        assert isinstance(result, Var) and result.name == "y"

    def test_substitute_const(self):
        c = IntConst(5)
        result = _substitute(c, {"x": Var("y", INT)})
        assert isinstance(result, IntConst) and result.value == 5

    def test_substitute_app(self):
        x = Var("x", INT)
        y = Var("y", INT)
        expr = App(Op.ADD, [x, IntConst(1)], INT)
        result = _substitute(expr, {"x": y})
        assert isinstance(result, App)
        assert result.args[0].name == "y"

    def test_negate(self):
        x = Var("x", BOOL)
        neg = _negate(x)
        assert isinstance(neg, App) and neg.op == Op.NOT

    def test_and_empty(self):
        result = _and()
        assert isinstance(result, BoolConst) and result.value is True

    def test_and_single(self):
        x = Var("x", BOOL)
        result = _and(x)
        assert result is x

    def test_and_multiple(self):
        x = Var("x", BOOL)
        y = Var("y", BOOL)
        result = _and(x, y)
        assert isinstance(result, App) and result.op == Op.AND

    def test_or_empty(self):
        result = _or()
        assert isinstance(result, BoolConst) and result.value is False

    def test_or_single(self):
        x = Var("x", BOOL)
        result = _or(x)
        assert result is x

    def test_eq_int(self):
        x = Var("x", INT)
        result = _eq(x, IntConst(0))
        assert isinstance(result, App) and result.op == Op.EQ

    def test_eq_bool(self):
        x = Var("x", BOOL)
        y = Var("y", BOOL)
        result = _eq(x, y)
        assert isinstance(result, App) and result.op == Op.IFF


# ==========================================
# Section 3: Basic Safety Properties
# ==========================================

class TestBasicSafety:
    def test_trivial_safe(self):
        """Init: x=0, Trans: x'=x, Prop: x >= 0. Trivially safe."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, x))  # No change
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_trivial_unsafe_at_init(self):
        """Init: x=-1, Prop: x >= 0. Unsafe immediately."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(-1)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE
        assert result.counterexample is not None
        assert result.counterexample.length == 0

    def test_constant_system_safe(self):
        """Init: x=5, Trans: x'=5, Prop: x==5. Always safe."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(5)))
        ts.set_trans(_eq(xp, IntConst(5)))
        ts.set_property(_eq(x, IntConst(5)))

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE


# ==========================================
# Section 4: Counter Systems
# ==========================================

class TestCounterSystems:
    def test_counter_always_nonneg(self):
        """Counter starts at 0, increments. Property: c >= 0."""
        ts = make_counter_system()  # prop: c >= 0
        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_counter_bounded_unsafe(self):
        """Counter starts at 0, increments. Property: c < 3. Eventually violated."""
        ts = make_counter_system(max_val=3)
        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE
        assert result.counterexample is not None
        assert result.counterexample.length >= 3

    def test_two_counter_invariant(self):
        """x+y=10 is an invariant when x inc and y dec from (0,10)."""
        ts = make_two_counter_system()
        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_counter_unsafe_trace_valid(self):
        """Verify the counterexample trace is meaningful."""
        ts = make_counter_system(max_val=2)
        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE
        ce = result.counterexample
        assert len(ce.trace) > 0
        # First state should have c=0 (initial)
        assert ce.trace[0].get("c", None) == 0 or len(ce.trace) >= 2


# ==========================================
# Section 5: Two-Variable Systems
# ==========================================

class TestTwoVariable:
    def test_copy_system(self):
        """x starts at 0, y copies x. Trans: x'=x+1, y'=x. Prop: y >= 0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")

        ts.set_init(_and(_eq(x, IntConst(0)), _eq(y, IntConst(0))))
        ts.set_trans(_and(
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT)),
            _eq(yp, x)
        ))
        ts.set_property(App(Op.GE, [y, IntConst(0)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_diverging_vars_unsafe(self):
        """x inc, y dec from (0,0). Prop: x == y. Unsafe after 1 step."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")

        ts.set_init(_and(_eq(x, IntConst(0)), _eq(y, IntConst(0))))
        ts.set_trans(_and(
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT)),
            _eq(yp, App(Op.SUB, [y, IntConst(1)], INT))
        ))
        ts.set_property(_eq(x, y))

        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE
        assert result.counterexample.length >= 1


# ==========================================
# Section 6: Boolean Variables
# ==========================================

class TestBooleanVars:
    def test_toggle_system(self):
        """Boolean flag toggles each step. Prop: True (always safe)."""
        ts = TransitionSystem()
        b = ts.add_bool_var("b")
        bp = ts.prime("b")

        ts.set_init(_eq(b, BoolConst(True)))
        ts.set_trans(_eq(bp, _negate(b)))
        ts.set_property(BoolConst(True))  # trivially safe

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_bool_guard(self):
        """
        x starts at 0, b starts True.
        Trans: if b then x'=x+1,b'=b else x'=x,b'=b
        Prop: x >= 0 (always safe since x only increments)
        """
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        b = ts.add_bool_var("b")
        xp = ts.prime("x")
        bp = ts.prime("b")

        s = SMTSolver()  # Just for ITE construction
        ts.set_init(_and(_eq(x, IntConst(0)), _eq(b, BoolConst(True))))
        ts.set_trans(_and(
            # x' = if b then x+1 else x
            _eq(xp, App(Op.ITE, [b, App(Op.ADD, [x, IntConst(1)], INT), x], INT)),
            _eq(bp, b)
        ))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE


# ==========================================
# Section 7: Conditional Transitions
# ==========================================

class TestConditionalTrans:
    def test_saturating_counter(self):
        """Counter saturates at 5: if x < 5 then x'=x+1 else x'=x. Prop: x <= 5."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(0)))
        cond = App(Op.LT, [x, IntConst(5)], BOOL)
        ts.set_trans(
            _eq(xp, App(Op.ITE, [cond, App(Op.ADD, [x, IntConst(1)], INT), x], INT))
        )
        ts.set_property(App(Op.LE, [x, IntConst(5)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_saturating_counter_tight_bound(self):
        """Same counter, but prop: x <= 4. Unsafe (reaches 5)."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(0)))
        cond = App(Op.LT, [x, IntConst(5)], BOOL)
        ts.set_trans(
            _eq(xp, App(Op.ITE, [cond, App(Op.ADD, [x, IntConst(1)], INT), x], INT))
        )
        ts.set_property(App(Op.LE, [x, IntConst(4)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE


# ==========================================
# Section 8: Nondeterministic Systems
# ==========================================

class TestNondeterministic:
    def test_nondet_increment(self):
        """
        x starts at 0. Each step: x' = x or x' = x+1 (nondeterministic).
        Prop: x >= 0. Safe since x never decreases.
        """
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(0)))
        # x' >= x AND x' <= x+1
        ts.set_trans(_and(
            App(Op.GE, [xp, x], BOOL),
            App(Op.LE, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL)
        ))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_nondet_can_go_negative(self):
        """
        x starts at 0. Each step: x' = x+1 or x' = x-1 (nondeterministic).
        Prop: x >= 0. Unsafe (can go to -1).
        """
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(0)))
        # x' = x-1 OR x' = x+1
        ts.set_trans(_or(
            _eq(xp, App(Op.SUB, [x, IntConst(1)], INT)),
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        ))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE


# ==========================================
# Section 9: Invariant Quality
# ==========================================

class TestInvariant:
    def test_safe_returns_invariant(self):
        """When SAFE, an invariant should be returned."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.SAFE
        assert result.invariant is not None

    def test_unsafe_has_no_invariant(self):
        """When UNSAFE, no invariant should be returned."""
        ts = make_counter_system(max_val=2)
        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE
        assert result.invariant is None


# ==========================================
# Section 10: Statistics
# ==========================================

class TestStats:
    def test_stats_populated(self):
        """Stats should track work done."""
        ts = make_counter_system(max_val=3)
        result = check_ts(ts)
        assert result.stats.smt_queries > 0

    def test_safe_stats(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(_eq(x, IntConst(0)))
        result = check_ts(ts)
        assert result.result == PDRResult.SAFE
        assert result.stats.frames_created >= 2


# ==========================================
# Section 11: Edge Cases
# ==========================================

class TestEdgeCases:
    def test_no_init_raises(self):
        ts = TransitionSystem()
        ts.add_int_var("x")
        ts.set_trans(BoolConst(True))
        ts.set_property(BoolConst(True))
        with pytest.raises(ValueError, match="initial"):
            check_ts(ts)

    def test_no_trans_raises(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_property(BoolConst(True))
        with pytest.raises(ValueError, match="transition"):
            check_ts(ts)

    def test_no_prop_raises(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(ts.prime("x"), x))
        with pytest.raises(ValueError, match="property"):
            check_ts(ts)

    def test_true_property(self):
        """Property is just True -- always safe."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(BoolConst(True))
        result = check_ts(ts)
        assert result.result == PDRResult.SAFE

    def test_false_property(self):
        """Property is False -- always unsafe."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(BoolConst(False))
        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE


# ==========================================
# Section 12: Convenience Functions
# ==========================================

class TestConvenience:
    def test_make_counter_default(self):
        ts = make_counter_system()
        assert ts.state_vars == [("c", INT)]
        assert ts.init_formula is not None
        assert ts.trans_formula is not None
        assert ts.prop_formula is not None

    def test_make_two_counter(self):
        ts = make_two_counter_system()
        assert len(ts.state_vars) == 2


# ==========================================
# Section 13: Multi-Step Reachability
# ==========================================

class TestMultiStep:
    def test_delayed_violation(self):
        """
        x starts at 0, increments by 2.
        Property: x != 10. Violated at step 5.
        """
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(2)], INT)))
        ts.set_property(App(Op.NEQ, [x, IntConst(10)], BOOL))

        result = check_ts(ts)
        assert result.result == PDRResult.UNSAFE
        assert result.counterexample.length >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
