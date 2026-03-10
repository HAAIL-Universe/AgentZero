"""
Tests for V021: BDD-based Symbolic Model Checking

Tests cover:
  1. BDD core operations (AND, OR, NOT, XOR, IFF, IMP)
  2. BDD quantification (exists, forall)
  3. BDD restrict and compose
  4. BDD counting and enumeration
  5. Boolean transition system construction
  6. Forward/backward reachability
  7. Safety checking (AG)
  8. CTL: EX, AX
  9. CTL: EF, AG
  10. CTL: EU, AU
  11. CTL: EG, AF
  12. CTL: ER, AR
  13. V002 TransitionSystem conversion
  14. Comparison with PDR
  15. Edge cases and stress tests
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'challenges', 'C037_smt_solver'))

from bdd_model_checker import (
    BDD, BDDNode, BooleanTS, make_boolean_ts, SymbolicModelChecker,
    MCResult, MCOutput,
    check_boolean_system, check_ctl, check_v002_system, compare_with_pdr,
    ts_to_boolean,
)
from pdr import TransitionSystem
from smt_solver import App, Op, Var, IntConst, BoolConst, INT, BOOL


# --- SMT formula helpers ---
def _eq(a, b): return App(Op.EQ, [a, b], BOOL)
def _neq(a, b): return App(Op.NEQ, [a, b], BOOL)
def _lt(a, b): return App(Op.LT, [a, b], BOOL)
def _le(a, b): return App(Op.LE, [a, b], BOOL)
def _ge(a, b): return App(Op.GE, [a, b], BOOL)
def _gt(a, b): return App(Op.GT, [a, b], BOOL)
def _add(a, b): return App(Op.ADD, [a, b], INT)
def _sub(a, b): return App(Op.SUB, [a, b], INT)
def _and(*args): return App(Op.AND, list(args), BOOL) if len(args) > 1 else args[0]
def _or(*args): return App(Op.OR, list(args), BOOL) if len(args) > 1 else args[0]
def _ite(c, t, e): return App(Op.ITE, [c, t, e], INT)
def _const(n): return IntConst(n)


# ============================================================
# Section 1: BDD Core Operations
# ============================================================

class TestBDDCore:
    def test_true_false(self):
        bdd = BDD()
        assert bdd.TRUE.value is True
        assert bdd.FALSE.value is False
        assert bdd.TRUE != bdd.FALSE

    def test_single_variable(self):
        bdd = BDD(2)
        x = bdd.var(0)
        assert not x.is_terminal()
        assert x.lo == bdd.FALSE
        assert x.hi == bdd.TRUE

    def test_and(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        r = bdd.AND(x, y)
        # x AND y: only true when both true
        assert bdd.sat_count(r, 2) == 1

    def test_or(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        r = bdd.OR(x, y)
        # x OR y: true for 3 of 4 assignments
        assert bdd.sat_count(r, 2) == 3

    def test_not(self):
        bdd = BDD(2)
        x = bdd.var(0)
        nx = bdd.NOT(x)
        assert bdd.sat_count(nx, 2) == 2  # !x is true for 2 assignments (x=0,y=0 and x=0,y=1)

    def test_xor(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        r = bdd.XOR(x, y)
        assert bdd.sat_count(r, 2) == 2

    def test_iff(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        r = bdd.IFF(x, y)
        assert bdd.sat_count(r, 2) == 2  # (0,0) and (1,1)

    def test_imp(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        r = bdd.IMP(x, y)
        # x -> y: false only when x=1,y=0
        assert bdd.sat_count(r, 2) == 3

    def test_canonical(self):
        """Same function built differently should give same BDD."""
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        # x AND y built two ways
        r1 = bdd.AND(x, y)
        r2 = bdd.AND(y, x)
        assert r1._id == r2._id  # Canonical!

    def test_demorgan(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        # NOT(x AND y) == NOT(x) OR NOT(y)
        lhs = bdd.NOT(bdd.AND(x, y))
        rhs = bdd.OR(bdd.NOT(x), bdd.NOT(y))
        assert lhs._id == rhs._id

    def test_and_all_or_all(self):
        bdd = BDD(3)
        x, y, z = bdd.var(0), bdd.var(1), bdd.var(2)
        r = bdd.and_all([x, y, z])
        assert bdd.sat_count(r, 3) == 1
        r2 = bdd.or_all([x, y, z])
        assert bdd.sat_count(r2, 3) == 7


# ============================================================
# Section 2: BDD Quantification
# ============================================================

class TestQuantification:
    def test_exists_basic(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        # exists y. (x AND y) = x
        r = bdd.exists(1, bdd.AND(x, y))
        assert r._id == x._id

    def test_forall_basic(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        # forall y. (x AND y) = FALSE (because x AND 0 = 0)
        r = bdd.forall(1, bdd.AND(x, y))
        assert r == bdd.FALSE

    def test_forall_or(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        # forall y. (x OR y): when x=1, (1 OR y)=1 for all y. When x=0, (0 OR 0)=0.
        r = bdd.forall(1, bdd.OR(x, y))
        assert r._id == x._id

    def test_exists_multi(self):
        bdd = BDD(3)
        x, y, z = bdd.var(0), bdd.var(1), bdd.var(2)
        # exists y,z. (x AND y AND z) = x
        r = bdd.exists_multi([1, 2], bdd.and_all([x, y, z]))
        assert r._id == x._id

    def test_forall_multi(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        # forall x,y. TRUE = TRUE
        r = bdd.forall_multi([0, 1], bdd.TRUE)
        assert r == bdd.TRUE
        # forall x,y. x = FALSE
        r2 = bdd.forall_multi([0, 1], x)
        assert r2 == bdd.FALSE


# ============================================================
# Section 3: BDD Restrict and Compose
# ============================================================

class TestRestrict:
    def test_restrict_true(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        f = bdd.AND(x, y)
        # f[x=1] = y
        r = bdd.restrict(f, 0, True)
        assert r._id == y._id

    def test_restrict_false(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        f = bdd.AND(x, y)
        # f[x=0] = FALSE
        r = bdd.restrict(f, 0, False)
        assert r == bdd.FALSE

    def test_compose_basic(self):
        bdd = BDD(3)
        x, y, z = bdd.var(0), bdd.var(1), bdd.var(2)
        f = bdd.AND(x, y)
        # f[x := z] = z AND y
        r = bdd.compose(f, 0, z)
        expected = bdd.AND(z, y)
        assert r._id == expected._id

    def test_compose_constant(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        f = bdd.OR(x, y)
        # f[x := TRUE] = TRUE
        r = bdd.compose(f, 0, bdd.TRUE)
        assert r == bdd.TRUE


# ============================================================
# Section 4: Counting and Enumeration
# ============================================================

class TestCounting:
    def test_sat_count_true(self):
        bdd = BDD(3)
        assert bdd.sat_count(bdd.TRUE, 3) == 8

    def test_sat_count_false(self):
        bdd = BDD(3)
        assert bdd.sat_count(bdd.FALSE, 3) == 0

    def test_sat_count_single_var(self):
        bdd = BDD(3)
        x = bdd.var(0)
        assert bdd.sat_count(x, 3) == 4  # x=1 with 2 don't-care bits

    def test_any_sat(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        f = bdd.AND(x, y)
        result = bdd.any_sat(f)
        assert result is not None
        assert result[0] is True
        assert result[1] is True

    def test_any_sat_false(self):
        bdd = BDD(2)
        assert bdd.any_sat(bdd.FALSE) is None

    def test_all_sat(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        f = bdd.OR(x, y)
        results = bdd.all_sat(f, 2)
        assert len(results) == 3  # (0,1), (1,0), (1,1)

    def test_node_count(self):
        bdd = BDD(2)
        x, y = bdd.var(0), bdd.var(1)
        f = bdd.AND(x, y)
        # Should have: FALSE, TRUE, y (internal), x AND y (root) = 4 nodes
        count = bdd.node_count(f)
        assert count >= 3  # At least root + 2 terminals

    def test_to_expr(self):
        bdd = BDD(2)
        x = bdd.var(0)
        expr = bdd.to_expr(x)
        assert "x0" in expr


# ============================================================
# Section 5: Boolean Transition System Construction
# ============================================================

class TestBooleanTS:
    def test_make_ts(self):
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['a', 'b'])
        assert 'a' in bts.state_vars
        assert 'b' in bts.state_vars
        assert "a'" in bts.next_vars
        assert "b'" in bts.next_vars

    def test_ts_variables(self):
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        assert 'x' in bts.var_indices
        assert 'x' in bts.next_indices
        assert bts.var_indices['x'] != bts.next_indices['x']

    def test_ts_default_init_trans(self):
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        # Default: TRUE (all states are initial, all transitions valid)
        assert bts.init == bdd.TRUE
        assert bts.trans == bdd.TRUE


# ============================================================
# Section 6: Forward/Backward Reachability
# ============================================================

class TestReachability:
    def test_forward_trivial(self):
        """Single-var system: x=0 initially, x'=!x. Reaches {0, 1}."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)  # x=0
        # x' = NOT x (toggle)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)
        reached, iters = mc.forward_reachable()
        assert bdd.sat_count(reached, 1) == 2  # Both states reachable

    def test_forward_fixed_point(self):
        """x=0, x'=0. Only state 0 is reachable."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)  # x=0
        bts.trans = bdd.NOT(xp)  # x'=0 always
        mc = SymbolicModelChecker(bts)
        reached, iters = mc.forward_reachable()
        assert bdd.sat_count(reached, 1) == 1
        assert iters <= 2  # Quick fixpoint

    def test_backward_reach(self):
        """Backward from x=1 in toggle system should reach both states."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)
        target = x  # x=1
        reached, iters = mc.backward_reachable(target)
        assert bdd.sat_count(reached, 1) == 2

    def test_multi_var_reachability(self):
        """2-bit counter: (a,b) starts at (0,0), increments mod 4."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['a', 'b'])
        a = bdd.named_var('a')
        b = bdd.named_var('b')
        ap = bdd.named_var("a'")
        bp = bdd.named_var("b'")

        bts.init = bdd.AND(bdd.NOT(a), bdd.NOT(b))  # a=0, b=0

        # Increment: b' = !b, a' = a XOR b
        bts.trans = bdd.AND(
            bdd.IFF(bp, bdd.NOT(b)),
            bdd.IFF(ap, bdd.XOR(a, b))
        )

        mc = SymbolicModelChecker(bts)
        reached, iters = mc.forward_reachable()
        assert bdd.sat_count(reached, 2) == 4  # All 4 states reachable


# ============================================================
# Section 7: Safety Checking (AG)
# ============================================================

class TestSafety:
    def test_safe_property(self):
        """x=0, x'=0. Property: always x=0. Should be SAFE."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)  # Stay at 0

        mc = SymbolicModelChecker(bts)
        result = mc.check_safety(bdd.NOT(x))
        assert result.result == MCResult.SAFE

    def test_unsafe_immediate(self):
        """x=1 initially, property: x=0. Should be UNSAFE immediately."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = x  # x=1
        bts.trans = bdd.TRUE

        mc = SymbolicModelChecker(bts)
        result = mc.check_safety(bdd.NOT(x))  # prop: x=0
        assert result.result == MCResult.UNSAFE

    def test_unsafe_after_steps(self):
        """x=0, x'=!x. Property: always x=0. Unsafe after 1 step."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))  # Toggle

        mc = SymbolicModelChecker(bts)
        result = mc.check_safety(bdd.NOT(x))  # prop: x=0
        assert result.result == MCResult.UNSAFE
        assert result.witness is not None

    def test_safe_two_var(self):
        """Two vars, x=0,y=0, x'=y, y'=x. Property: x=0 OR y=0. Safe."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x', 'y'])
        x = bdd.named_var('x')
        y = bdd.named_var('y')
        xp = bdd.named_var("x'")
        yp = bdd.named_var("y'")

        bts.init = bdd.AND(bdd.NOT(x), bdd.NOT(y))
        bts.trans = bdd.AND(bdd.IFF(xp, y), bdd.IFF(yp, x))

        mc = SymbolicModelChecker(bts)
        prop = bdd.OR(bdd.NOT(x), bdd.NOT(y))  # x=0 OR y=0
        result = mc.check_safety(prop)
        assert result.result == MCResult.SAFE

    def test_counterexample_trace(self):
        """Verify counterexample is a valid trace."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))

        mc = SymbolicModelChecker(bts)
        result = mc.check_safety(bdd.NOT(x))
        assert result.result == MCResult.UNSAFE
        assert result.witness is not None
        assert len(result.witness) >= 1

    def test_safe_with_invariant(self):
        """Safe system returns invariant (reachable states)."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)

        mc = SymbolicModelChecker(bts)
        result = mc.check_safety(bdd.NOT(x))
        assert result.result == MCResult.SAFE
        assert result.invariant is not None


# ============================================================
# Section 8: CTL - EX, AX
# ============================================================

class TestCTL_EX_AX:
    def _make_toggle_system(self):
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)
        return bdd, mc, x

    def test_ex_basic(self):
        """EX(x=1) from toggle system: states that have successor with x=1."""
        bdd, mc, x = self._make_toggle_system()
        # EX(x=1): predecessor of x=1 is x=0 (via toggle)
        result = mc.EX(x)
        # States where EX(x): should be !x (from x=0, next is x=1)
        assert bdd.sat_count(result, 1) == 1  # Only x=0

    def test_ax_basic(self):
        """AX(x=1) in toggle: states where ALL successors have x=1."""
        bdd, mc, x = self._make_toggle_system()
        result = mc.AX(x)
        # Only from x=0 do we always go to x=1 (deterministic toggle)
        assert bdd.sat_count(result, 1) == 1

    def test_ex_true(self):
        """EX(TRUE): states that have any successor."""
        bdd, mc, x = self._make_toggle_system()
        result = mc.EX(bdd.TRUE)
        # Both states have successors in toggle
        assert bdd.sat_count(result, 1) == 2


# ============================================================
# Section 9: CTL - EF, AG
# ============================================================

class TestCTL_EF_AG:
    def test_ef_reachable(self):
        """EF(x=1) in toggle system starting from x=0."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        ef_x = mc.EF(x)
        # Both states can reach x=1 (x=0 -> x=1, x=1 is already x=1)
        assert bdd.sat_count(ef_x, 1) == 2

    def test_ag_always(self):
        """AG(x=0) in stay-at-0 system."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)  # Always go to 0
        mc = SymbolicModelChecker(bts)

        ag_not_x = mc.AG(bdd.NOT(x))
        # x=0 satisfies AG(x=0) because it stays at 0
        assert bdd.AND(bts.init, ag_not_x) != bdd.FALSE

    def test_ag_fails(self):
        """AG(x=0) in toggle system should fail for x=0."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        ag_not_x = mc.AG(bdd.NOT(x))
        # No state satisfies AG(x=0) in toggle because eventually x=1
        assert ag_not_x == bdd.FALSE

    def test_ef_unreachable(self):
        """EF(x=1) when stuck at x=0."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)  # Stay at 0
        mc = SymbolicModelChecker(bts)

        ef_x = mc.EF(x)
        # Only x=1 satisfies EF(x=1) trivially. x=0 can never reach x=1.
        init_satisfies = bdd.AND(bts.init, ef_x)
        assert init_satisfies == bdd.FALSE


# ============================================================
# Section 10: CTL - EU, AU
# ============================================================

class TestCTL_EU_AU:
    def test_eu_basic(self):
        """E[TRUE U x=1]: can we reach x=1? Same as EF(x=1)."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        eu = mc.EU(bdd.TRUE, x)
        ef = mc.EF(x)
        assert eu._id == ef._id  # E[TRUE U p] = EF p

    def test_au_basic(self):
        """A[TRUE U x=1] in toggle from x=0: eventually x=1 on all paths."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        au = mc.AU(bdd.TRUE, x)
        # From x=0, next step is x=1. So x=0 satisfies A[TRUE U x=1].
        assert bdd.AND(bts.init, au) != bdd.FALSE

    def test_eu_with_condition(self):
        """E[x=0 U x=1]: can reach x=1 while maintaining x=0."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        eu = mc.EU(bdd.NOT(x), x)
        # From x=0, x=0 holds and next is x=1. So x=0 satisfies E[!x U x].
        init_sat = bdd.AND(bts.init, eu)
        assert init_sat != bdd.FALSE


# ============================================================
# Section 11: CTL - EG, AF
# ============================================================

class TestCTL_EG_AF:
    def test_eg_basic(self):
        """EG(x=0) in stay-at-0 system: x=0 holds forever."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)
        mc = SymbolicModelChecker(bts)

        eg = mc.EG(bdd.NOT(x))
        # x=0 satisfies EG(!x) because we stay at 0 forever
        assert bdd.AND(bts.init, eg) != bdd.FALSE

    def test_eg_fails_toggle(self):
        """EG(x=0) in toggle: no state can maintain x=0 forever."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        eg = mc.EG(bdd.NOT(x))
        assert eg == bdd.FALSE

    def test_af_basic(self):
        """AF(x=1) in toggle: on all paths, x=1 eventually."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        af = mc.AF(x)
        # Both states satisfy AF(x=1): x=1 trivially, x=0 in one step
        assert bdd.sat_count(af, 1) == 2

    def test_af_fails(self):
        """AF(x=1) when stuck at x=0: x=1 never reached."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)
        mc = SymbolicModelChecker(bts)

        af = mc.AF(x)
        init_sat = bdd.AND(bts.init, af)
        assert init_sat == bdd.FALSE


# ============================================================
# Section 12: CTL - ER, AR
# ============================================================

class TestCTL_ER_AR:
    def test_er_basic(self):
        """ER is dual of AU: E[p R q] = NOT A[!p U !q]."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        er = mc.ER(x, bdd.NOT(x))
        # E[x R !x]: !x holds until x AND !x (never), OR !x holds forever
        # In toggle, !x can't hold forever. So ER = FALSE? Let's check.
        # Actually ER(p, q) = NOT AU(!p, !q)
        au = mc.AU(bdd.NOT(x), x)
        expected = bdd.NOT(au)
        assert er._id == expected._id

    def test_ar_basic(self):
        """AR is dual of EU: A[p R q] = NOT E[!p U !q]."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = SymbolicModelChecker(bts)

        ar = mc.AR(x, bdd.NOT(x))
        eu = mc.EU(bdd.NOT(x), x)
        expected = bdd.NOT(eu)
        assert ar._id == expected._id


# ============================================================
# Section 13: V002 TransitionSystem Conversion
# ============================================================

class TestV002Conversion:
    def test_simple_counter(self):
        """Convert a simple counter: x=0, x'=x+1, prop: x <= 3, bit_width=3."""
        ts = TransitionSystem()
        x = ts.add_int_var('x')
        xp = ts.prime('x')

        ts.init_formula = _eq(x, _const(0))
        ts.trans_formula = _eq(xp, _add(x, _const(1)))
        ts.prop_formula = _le(x, _const(3))

        result = check_v002_system(ts, bit_width=3)
        # Counter goes 0,1,2,3,4,5,6,7 -- prop x<=3 violated at x=4
        assert result.result == MCResult.UNSAFE

    def test_safe_bounded(self):
        """x=0, x'= if x<2 then x+1 else x. Prop: x<=2. Safe with 2 bits."""
        ts = TransitionSystem()
        x = ts.add_int_var('x')
        xp = ts.prime('x')

        ts.init_formula = _eq(x, _const(0))
        ts.trans_formula = _eq(xp, _ite(_lt(x, _const(2)),
                                         _add(x, _const(1)), x))
        ts.prop_formula = _le(x, _const(2))

        result = check_v002_system(ts, bit_width=2)
        assert result.result == MCResult.SAFE

    def test_conversion_preserves_init(self):
        """Verify converted init states are correct."""
        ts = TransitionSystem()
        x = ts.add_int_var('x')
        xp = ts.prime('x')
        ts.init_formula = _eq(x, _const(0))
        ts.trans_formula = _eq(xp, x)
        ts.prop_formula = _ge(x, _const(0))

        result = check_v002_system(ts, bit_width=3)
        assert result.result == MCResult.SAFE


# ============================================================
# Section 14: Comparison with PDR
# ============================================================

class TestComparison:
    def test_compare_safe(self):
        """Both BDD and PDR should agree on safe system."""
        ts = TransitionSystem()
        x = ts.add_int_var('x')
        xp = ts.prime('x')
        ts.init_formula = _eq(x, _const(0))
        ts.trans_formula = _eq(xp, x)
        ts.prop_formula = _eq(x, _const(0))

        result = compare_with_pdr(ts, bit_width=3)
        assert result['agree'] is True
        assert result['bdd_result'] == 'safe'

    def test_compare_unsafe(self):
        """Both should agree on unsafe system."""
        ts = TransitionSystem()
        x = ts.add_int_var('x')
        xp = ts.prime('x')
        ts.init_formula = _eq(x, _const(0))
        ts.trans_formula = _eq(xp, _add(x, _const(1)))
        ts.prop_formula = _le(x, _const(2))

        result = compare_with_pdr(ts, bit_width=3)
        assert result['agree'] is True
        assert result['bdd_result'] == 'unsafe'


# ============================================================
# Section 15: High-level API and Edge Cases
# ============================================================

class TestHighLevel:
    def test_check_boolean_system(self):
        """Use the high-level check_boolean_system API."""
        result = check_boolean_system(
            state_vars=['x', 'y'],
            init_expr=lambda bdd, v: bdd.AND(bdd.NOT(v['x']), bdd.NOT(v['y'])),
            trans_expr=lambda bdd, c, n: bdd.AND(
                bdd.IFF(n['x'], c['y']),
                bdd.IFF(n['y'], bdd.NOT(c['x']))
            ),
            prop_expr=lambda bdd, v: bdd.OR(bdd.NOT(v['x']), bdd.NOT(v['y'])),
        )
        # System: x'=y, y'=!x. From (0,0)->(0,1)->(1,0)->(1,1)->...
        # Prop: x=0 OR y=0. (1,1) violates it.
        assert result.result == MCResult.UNSAFE

    def test_check_ctl_api(self):
        """Use the high-level check_ctl API."""
        result = check_ctl(
            state_vars=['x'],
            init_expr=lambda bdd, v: bdd.NOT(v['x']),
            trans_expr=lambda bdd, c, n: bdd.IFF(n['x'], bdd.NOT(c['x'])),
            ctl_expr=lambda mc, bdd, v: mc.AF(v['x']),  # AF(x=1)
        )
        assert result['sat_in_init'] is True

    def test_empty_system(self):
        """System with no variables: trivially safe."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, [])
        bts.init = bdd.TRUE
        bts.trans = bdd.TRUE
        mc = SymbolicModelChecker(bts)
        result = mc.check_safety(bdd.TRUE)
        assert result.result == MCResult.SAFE

    def test_unreachable_violation(self):
        """Violation exists but is unreachable from init."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x', 'y'])
        x = bdd.named_var('x')
        y = bdd.named_var('y')
        xp = bdd.named_var("x'")
        yp = bdd.named_var("y'")

        # Init: x=0, y=0. Trans: stay at (0,0) always.
        bts.init = bdd.AND(bdd.NOT(x), bdd.NOT(y))
        bts.trans = bdd.AND(bdd.NOT(xp), bdd.NOT(yp))

        # Prop: NOT (x=1 AND y=1). State (1,1) violates but is unreachable.
        prop = bdd.NOT(bdd.AND(x, y))
        mc = SymbolicModelChecker(bts)
        result = mc.check_safety(prop)
        assert result.result == MCResult.SAFE

    def test_nondeterministic_system(self):
        """System with nondeterministic transitions."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")

        bts.init = bdd.NOT(x)  # x=0
        bts.trans = bdd.TRUE  # Any transition allowed

        mc = SymbolicModelChecker(bts)
        # Property x=0 is unsafe because nondeterminism can go to x=1
        result = mc.check_safety(bdd.NOT(x))
        assert result.result == MCResult.UNSAFE

    def test_three_bit_counter(self):
        """3-bit counter reaches all 8 states."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['a', 'b', 'c'])
        a = bdd.named_var('a')
        b = bdd.named_var('b')
        c = bdd.named_var('c')
        ap = bdd.named_var("a'")
        bp = bdd.named_var("b'")
        cp = bdd.named_var("c'")

        # Init: all 0
        bts.init = bdd.and_all([bdd.NOT(a), bdd.NOT(b), bdd.NOT(c)])

        # 3-bit increment
        bts.trans = bdd.and_all([
            bdd.IFF(cp, bdd.NOT(c)),
            bdd.IFF(bp, bdd.XOR(b, c)),
            bdd.IFF(ap, bdd.XOR(a, bdd.AND(b, c))),
        ])

        mc = SymbolicModelChecker(bts)
        reached, iters = mc.forward_reachable()
        assert bdd.sat_count(reached, 3) == 8

    def test_mutual_exclusion(self):
        """Simple mutual exclusion: two processes can't both be in critical section."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['p1', 'p2'])
        p1 = bdd.named_var('p1')
        p2 = bdd.named_var('p2')
        p1p = bdd.named_var("p1'")
        p2p = bdd.named_var("p2'")

        # Init: both idle (0)
        bts.init = bdd.AND(bdd.NOT(p1), bdd.NOT(p2))
        # Trans: only one can enter at a time (mutex)
        # p1' can become 1 only if p2=0, p2' can become 1 only if p1=0
        enter_p1 = bdd.and_all([bdd.NOT(p2), p1p, bdd.IFF(p2p, p2)])
        enter_p2 = bdd.and_all([bdd.NOT(p1), p2p, bdd.IFF(p1p, p1)])
        stay = bdd.AND(bdd.IFF(p1p, p1), bdd.IFF(p2p, p2))
        exit_p1 = bdd.and_all([p1, bdd.NOT(p1p), bdd.IFF(p2p, p2)])
        exit_p2 = bdd.and_all([p2, bdd.NOT(p2p), bdd.IFF(p1p, p1)])
        trans = bdd.or_all([enter_p1, enter_p2, stay, exit_p1, exit_p2])
        bts.trans = trans

        mc = SymbolicModelChecker(bts)
        # Mutual exclusion: NOT(p1 AND p2)
        prop = bdd.NOT(bdd.AND(p1, p2))
        result = mc.check_safety(prop)
        assert result.result == MCResult.SAFE


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
