"""
Tests for V038: LTL Model Checking

Tests cover:
1. LTL formula construction and NNF conversion
2. Tableau/automaton construction
3. LTL model checking on boolean transition systems
4. Various LTL properties (safety, liveness, fairness)
5. Properties that CTL cannot express
"""

import sys
import os
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_work, "V021_bdd_model_checking"))

from ltl_model_checker import (
    # LTL AST
    LTL, LTLOp, Atom, LTLTrue, LTLFalse, Not, And, Or, Implies, Iff,
    X, F, G, U, R, W,
    # NNF
    to_nnf,
    # Tableau
    ltl_to_gba, subformulas,
    # Model checker
    LTLModelChecker, LTLResult, LTLOutput,
    check_ltl, check_ltl_simple,
)
from bdd_model_checker import BDD, make_boolean_ts, SymbolicModelChecker


# ============================================================
# LTL Formula Tests
# ============================================================

class TestLTLFormulas:
    def test_atom(self):
        p = Atom("p")
        assert p.op == LTLOp.ATOM
        assert p.name == "p"
        assert repr(p) == "p"

    def test_boolean_constants(self):
        assert LTLTrue().op == LTLOp.TRUE
        assert LTLFalse().op == LTLOp.FALSE

    def test_negation_double(self):
        p = Atom("p")
        assert Not(Not(p)) == p

    def test_negation_constants(self):
        assert Not(LTLTrue()).op == LTLOp.FALSE
        assert Not(LTLFalse()).op == LTLOp.TRUE

    def test_and_simplification(self):
        p = Atom("p")
        assert And(p, LTLTrue()) == p
        assert And(LTLTrue(), p) == p
        assert And(p, LTLFalse()).op == LTLOp.FALSE
        assert And(LTLFalse(), p).op == LTLOp.FALSE

    def test_or_simplification(self):
        p = Atom("p")
        assert Or(p, LTLFalse()) == p
        assert Or(LTLFalse(), p) == p
        assert Or(p, LTLTrue()).op == LTLOp.TRUE

    def test_temporal_operators(self):
        p = Atom("p")
        assert X(p).op == LTLOp.NEXT
        assert F(p).op == LTLOp.EVENTUALLY
        assert G(p).op == LTLOp.GLOBALLY
        q = Atom("q")
        assert U(p, q).op == LTLOp.UNTIL
        assert R(p, q).op == LTLOp.RELEASE

    def test_formula_equality(self):
        p = Atom("p")
        q = Atom("q")
        assert G(p) == G(p)
        assert G(p) != F(p)
        assert And(p, q) == And(p, q)

    def test_subformulas(self):
        p = Atom("p")
        q = Atom("q")
        f = G(And(p, q))
        subs = subformulas(f)
        assert p in subs
        assert q in subs
        assert And(p, q) in subs
        assert f in subs


# ============================================================
# NNF Tests
# ============================================================

class TestNNF:
    def test_atom_nnf(self):
        p = Atom("p")
        assert to_nnf(p) == p

    def test_neg_atom_nnf(self):
        p = Atom("p")
        result = to_nnf(Not(p))
        assert result.op == LTLOp.NOT
        assert result.args[0] == p

    def test_double_negation(self):
        p = Atom("p")
        assert to_nnf(Not(Not(p))) == p

    def test_demorgan_and(self):
        p, q = Atom("p"), Atom("q")
        result = to_nnf(Not(And(p, q)))
        assert result.op == LTLOp.OR

    def test_demorgan_or(self):
        p, q = Atom("p"), Atom("q")
        result = to_nnf(Not(Or(p, q)))
        assert result.op == LTLOp.AND

    def test_neg_eventually(self):
        p = Atom("p")
        result = to_nnf(Not(F(p)))
        # !F(p) = G(!p)
        assert result.op == LTLOp.GLOBALLY

    def test_neg_globally(self):
        p = Atom("p")
        result = to_nnf(Not(G(p)))
        # !G(p) = F(!p)
        assert result.op == LTLOp.EVENTUALLY

    def test_neg_until(self):
        p, q = Atom("p"), Atom("q")
        result = to_nnf(Not(U(p, q)))
        # !(p U q) = !p R !q
        assert result.op == LTLOp.RELEASE

    def test_neg_release(self):
        p, q = Atom("p"), Atom("q")
        result = to_nnf(Not(R(p, q)))
        # !(p R q) = !p U !q
        assert result.op == LTLOp.UNTIL

    def test_implies_nnf(self):
        p, q = Atom("p"), Atom("q")
        result = to_nnf(Implies(p, q))
        # p -> q = !p | q
        assert result.op == LTLOp.OR

    def test_eventually_expansion(self):
        p = Atom("p")
        result = to_nnf(F(p))
        # F(p) = true U p
        assert result.op == LTLOp.UNTIL

    def test_globally_expansion(self):
        p = Atom("p")
        result = to_nnf(G(p))
        # G(p) = false R p
        assert result.op == LTLOp.RELEASE


# ============================================================
# Tableau Construction Tests
# ============================================================

class TestTableau:
    def test_atom_tableau(self):
        p = Atom("p")
        gba, states, edges, n = ltl_to_gba(to_nnf(p))
        assert n > 0
        assert len(edges) > 0

    def test_and_tableau(self):
        p, q = Atom("p"), Atom("q")
        gba, states, edges, n = ltl_to_gba(to_nnf(And(p, q)))
        assert n > 0
        # All edges from initial should require both p and q
        for e in edges:
            if e.src in gba.initial:
                assert "p" in e.pos_atoms
                assert "q" in e.pos_atoms

    def test_or_tableau(self):
        p, q = Atom("p"), Atom("q")
        gba, states, edges, n = ltl_to_gba(to_nnf(Or(p, q)))
        assert n > 0

    def test_next_tableau(self):
        p = Atom("p")
        gba, states, edges, n = ltl_to_gba(to_nnf(X(p)))
        assert n >= 2  # At least initial state + next state

    def test_until_tableau(self):
        p, q = Atom("p"), Atom("q")
        gba, states, edges, n = ltl_to_gba(to_nnf(U(p, q)))
        assert n > 0
        # Should have acceptance condition for the Until
        assert len(gba.acceptance) == 1

    def test_globally_tableau(self):
        p = Atom("p")
        gba, states, edges, n = ltl_to_gba(to_nnf(G(p)))
        assert n > 0


# ============================================================
# LTL Model Checking Tests
# ============================================================

class TestLTLModelChecking:
    """Core model checking tests on simple boolean systems."""

    def _make_toggle(self):
        """Toggle system: x starts False, flips each step."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)  # x=0
        bts.trans = bdd.IFF(xp, bdd.NOT(x))  # x' = !x
        return bts, bdd

    def _make_always_true(self):
        """System where x is always true."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = x
        bts.trans = xp  # x' = true always
        return bts, bdd

    def _make_eventually_true(self):
        """System: x starts false, goes to true, stays true."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = xp  # x' = true always (so after 1 step x is true)
        return bts, bdd

    # --- Safety (G) ---

    def test_globally_satisfied(self):
        """G(x) on system where x is always true."""
        bts, bdd = self._make_always_true()
        mc = LTLModelChecker(bts)
        result = mc.check(G(Atom("x")))
        assert result.result == LTLResult.SATISFIED

    def test_globally_violated(self):
        """G(x) on toggle system (x starts false)."""
        bts, bdd = self._make_toggle()
        mc = LTLModelChecker(bts)
        result = mc.check(G(Atom("x")))
        assert result.result == LTLResult.VIOLATED

    # --- Liveness (F) ---

    def test_eventually_satisfied(self):
        """F(x) on system where x eventually becomes true."""
        bts, bdd = self._make_eventually_true()
        mc = LTLModelChecker(bts)
        result = mc.check(F(Atom("x")))
        assert result.result == LTLResult.SATISFIED

    def test_eventually_on_always_false(self):
        """F(x) on system where x is always false -> violated."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)  # x' = false always
        mc = LTLModelChecker(bts)
        result = mc.check(F(Atom("x")))
        assert result.result == LTLResult.VIOLATED

    # --- Next (X) ---

    def test_next_satisfied(self):
        """X(x) on toggle: x starts false, next step x is true."""
        bts, bdd = self._make_toggle()
        mc = LTLModelChecker(bts)
        result = mc.check(X(Atom("x")))
        assert result.result == LTLResult.SATISFIED

    def test_next_violated(self):
        """X(x) on always-false system."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)
        mc = LTLModelChecker(bts)
        result = mc.check(X(Atom("x")))
        assert result.result == LTLResult.VIOLATED

    # --- Until (U) ---

    def test_until_satisfied(self):
        """(!x U x) on eventually-true system."""
        bts, bdd = self._make_eventually_true()
        mc = LTLModelChecker(bts)
        result = mc.check(U(Not(Atom("x")), Atom("x")))
        assert result.result == LTLResult.SATISFIED

    def test_until_violated(self):
        """(!x U x) on always-false system."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)
        mc = LTLModelChecker(bts)
        result = mc.check(U(Not(Atom("x")), Atom("x")))
        assert result.result == LTLResult.VIOLATED

    # --- Infinitely Often: G(F(x)) ---
    # This is a key property that CTL *cannot* express directly

    def test_gf_satisfied_toggle(self):
        """G(F(x)) on toggle system: x is true infinitely often."""
        bts, bdd = self._make_toggle()
        mc = LTLModelChecker(bts)
        result = mc.check(G(F(Atom("x"))))
        assert result.result == LTLResult.SATISFIED

    def test_gf_violated_eventually_stuck(self):
        """G(F(x)) on system that becomes permanently false."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = x  # start true
        bts.trans = bdd.NOT(xp)  # always goes to false
        mc = LTLModelChecker(bts)
        result = mc.check(G(F(Atom("x"))))
        assert result.result == LTLResult.VIOLATED

    # --- Eventually Always: F(G(x)) ---
    # Another property CTL cannot express

    def test_fg_satisfied(self):
        """F(G(x)) on eventually-true system (becomes permanently true)."""
        bts, bdd = self._make_eventually_true()
        mc = LTLModelChecker(bts)
        result = mc.check(F(G(Atom("x"))))
        assert result.result == LTLResult.SATISFIED

    def test_fg_violated_toggle(self):
        """F(G(x)) on toggle: x never stays true permanently."""
        bts, bdd = self._make_toggle()
        mc = LTLModelChecker(bts)
        result = mc.check(F(G(Atom("x"))))
        assert result.result == LTLResult.VIOLATED


class TestLTLMultiVariable:
    """Tests with multiple state variables."""

    def _make_two_var_cycle(self):
        """Two-bit counter: (0,0) -> (1,0) -> (0,1) -> (1,1) -> (0,0)."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x', 'y'])
        x = bdd.named_var('x')
        y = bdd.named_var('y')
        xp = bdd.named_var("x'")
        yp = bdd.named_var("y'")

        bts.init = bdd.AND(bdd.NOT(x), bdd.NOT(y))  # (0,0)
        # x' = !x, y' = y XOR x (binary counter)
        bts.trans = bdd.AND(
            bdd.IFF(xp, bdd.NOT(x)),
            bdd.IFF(yp, bdd.XOR(y, x))
        )
        return bts, bdd

    def test_two_var_globally(self):
        """Can't have both x and y always true in counter."""
        bts, bdd = self._make_two_var_cycle()
        mc = LTLModelChecker(bts)
        # G(x & y) should be violated (only true in state (1,1))
        result = mc.check(G(And(Atom("x"), Atom("y"))))
        assert result.result == LTLResult.VIOLATED

    def test_two_var_gf(self):
        """G(F(x)) on counter: x toggles every step, infinitely often true."""
        bts, bdd = self._make_two_var_cycle()
        mc = LTLModelChecker(bts)
        result = mc.check(G(F(Atom("x"))))
        assert result.result == LTLResult.SATISFIED

    def test_two_var_gf_both(self):
        """G(F(x)) & G(F(y)): both vars are infinitely often true."""
        bts, bdd = self._make_two_var_cycle()
        mc = LTLModelChecker(bts)
        result = mc.check(And(G(F(Atom("x"))), G(F(Atom("y")))))
        assert result.result == LTLResult.SATISFIED

    def test_two_var_eventually_both(self):
        """F(x & y) on counter: eventually both are true (state 1,1)."""
        bts, bdd = self._make_two_var_cycle()
        mc = LTLModelChecker(bts)
        result = mc.check(F(And(Atom("x"), Atom("y"))))
        assert result.result == LTLResult.SATISFIED


class TestLTLImplication:
    """Test response/precedence patterns."""

    def test_response_satisfied(self):
        """G(request -> F(grant)): every request eventually granted."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['req', 'grant'])
        req = bdd.named_var('req')
        grant = bdd.named_var('grant')
        reqp = bdd.named_var("req'")
        grantp = bdd.named_var("grant'")

        # Start: no request, no grant
        bts.init = bdd.AND(bdd.NOT(req), bdd.NOT(grant))
        # Transition: if req, grant next; req toggles
        bts.trans = bdd.AND(
            bdd.IFF(grantp, req),   # grant' = req (grant follows request)
            bdd.IFF(reqp, bdd.NOT(req))  # req toggles
        )

        mc = LTLModelChecker(bts)
        prop = G(Implies(Atom("req"), F(Atom("grant"))))
        result = mc.check(prop)
        assert result.result == LTLResult.SATISFIED

    def test_response_violated(self):
        """G(req -> F(grant)) violated: requests never granted."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['req', 'grant'])
        req = bdd.named_var('req')
        grant = bdd.named_var('grant')
        reqp = bdd.named_var("req'")
        grantp = bdd.named_var("grant'")

        bts.init = bdd.AND(req, bdd.NOT(grant))  # start with request
        bts.trans = bdd.AND(
            bdd.NOT(grantp),  # never grant
            reqp              # always request
        )

        mc = LTLModelChecker(bts)
        prop = G(Implies(Atom("req"), F(Atom("grant"))))
        result = mc.check(prop)
        assert result.result == LTLResult.VIOLATED


class TestHighLevelAPI:
    """Test the check_ltl convenience function."""

    def test_check_ltl_safety(self):
        result = check_ltl(
            state_vars=['x'],
            init_expr=lambda bdd, v: v['x'],  # x=1
            trans_expr=lambda bdd, c, n: n['x'],  # x'=1 always
            prop=G(Atom("x"))
        )
        assert result.result == LTLResult.SATISFIED

    def test_check_ltl_liveness(self):
        result = check_ltl(
            state_vars=['x'],
            init_expr=lambda bdd, v: bdd.NOT(v['x']),
            trans_expr=lambda bdd, c, n: n['x'],  # becomes true
            prop=F(Atom("x"))
        )
        assert result.result == LTLResult.SATISFIED

    def test_check_ltl_gf(self):
        result = check_ltl(
            state_vars=['x'],
            init_expr=lambda bdd, v: bdd.NOT(v['x']),
            trans_expr=lambda bdd, c, n: bdd.IFF(n['x'], bdd.NOT(c['x'])),  # toggle
            prop=G(F(Atom("x")))
        )
        assert result.result == LTLResult.SATISFIED


class TestCheckLTLSimple:
    """Test the simplified API."""

    def test_toggle_gf(self):
        result = check_ltl_simple(
            state_vars=['x'],
            init_map={'x': False},
            transitions=[
                {'x': lambda bdd, c: bdd.NOT(c['x'])}  # toggle
            ],
            prop=G(F(Atom("x")))
        )
        assert result.result == LTLResult.SATISFIED

    def test_stuck_false_f(self):
        result = check_ltl_simple(
            state_vars=['x'],
            init_map={'x': False},
            transitions=[
                {'x': False}  # always false
            ],
            prop=F(Atom("x"))
        )
        assert result.result == LTLResult.VIOLATED

    def test_conditional_transition(self):
        """If x is false, set it true. If true, keep true."""
        result = check_ltl_simple(
            state_vars=['x'],
            init_map={'x': False},
            transitions=[
                (lambda bdd, c: bdd.NOT(c['x']), {'x': True}),   # !x -> x'=true
                (lambda bdd, c: c['x'], {'x': True}),             # x -> x'=true
            ],
            prop=F(G(Atom("x")))  # eventually always true
        )
        assert result.result == LTLResult.SATISFIED


class TestEdgeCases:
    """Edge cases and special formulas."""

    def test_true_property(self):
        """G(true) should always be satisfied."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = x
        bts.trans = bdd.IFF(xp, bdd.NOT(x))
        mc = LTLModelChecker(bts)
        result = mc.check(G(LTLTrue()))
        assert result.result == LTLResult.SATISFIED

    def test_false_property(self):
        """F(false) should always be violated (can never reach false)."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = x
        bts.trans = xp
        mc = LTLModelChecker(bts)
        result = mc.check(F(LTLFalse()))
        # F(false) = true U false, which is unsatisfiable
        # So the property is vacuously violated (every path fails to satisfy it)
        assert result.result == LTLResult.VIOLATED

    def test_next_next(self):
        """X(X(x)): x is true two steps from now."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.IFF(xp, bdd.NOT(x))  # toggle
        mc = LTLModelChecker(bts)
        # From !x, next is x, next-next is !x -> violated
        result = mc.check(X(X(Atom("x"))))
        assert result.result == LTLResult.VIOLATED

    def test_release_basic(self):
        """p R q: q holds until (and including when) p holds."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['p', 'q'])
        p = bdd.named_var('p')
        q = bdd.named_var('q')
        pp = bdd.named_var("p'")
        qp = bdd.named_var("q'")

        # q always true, p eventually true
        bts.init = bdd.AND(bdd.NOT(p), q)
        bts.trans = bdd.AND(pp, qp)  # p'=1, q'=1
        mc = LTLModelChecker(bts)
        result = mc.check(R(Atom("p"), Atom("q")))
        assert result.result == LTLResult.SATISFIED

    def test_stats_present(self):
        """Check that stats are returned."""
        bts, bdd = TestLTLModelChecking()._make_toggle()
        mc = LTLModelChecker(bts)
        result = mc.check(G(Atom("x")))
        assert "auto_states" in result.stats


class TestCounterexample:
    """Test counterexample generation."""

    def test_counterexample_on_violation(self):
        """Violated property should produce a counterexample."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.NOT(xp)  # always false
        mc = LTLModelChecker(bts)
        result = mc.check(F(Atom("x")))
        assert result.result == LTLResult.VIOLATED
        assert result.counterexample is not None

    def test_no_counterexample_on_satisfaction(self):
        """Satisfied property should not have a counterexample."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = x
        bts.trans = xp
        mc = LTLModelChecker(bts)
        result = mc.check(G(Atom("x")))
        assert result.result == LTLResult.SATISFIED
        assert result.counterexample is None


class TestNondeterministic:
    """Test with nondeterministic systems."""

    def test_ndet_eventually(self):
        """Nondeterministic system: x can go true or stay false.
        F(x) should be violated (exists a path staying false)."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        # Nondeterministic: x' can be true or false
        bts.trans = bdd.TRUE  # any transition
        mc = LTLModelChecker(bts)
        # F(x) means on ALL paths eventually x -- violated because
        # there's a path that always stays !x
        result = mc.check(F(Atom("x")))
        assert result.result == LTLResult.VIOLATED

    def test_ndet_globally_or_eventually(self):
        """G(!x) | F(x) is a tautology on all paths."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['x'])
        x = bdd.named_var('x')
        xp = bdd.named_var("x'")
        bts.init = bdd.NOT(x)
        bts.trans = bdd.TRUE
        mc = LTLModelChecker(bts)
        # On any path, either x never becomes true (G(!x)) or it does (F(x))
        prop = Or(G(Not(Atom("x"))), F(Atom("x")))
        result = mc.check(prop)
        assert result.result == LTLResult.SATISFIED


class TestMutualExclusion:
    """Test mutual exclusion property (classic verification example)."""

    def test_mutex_safety(self):
        """Two processes cannot be in critical section simultaneously."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['cs1', 'cs2'])
        cs1 = bdd.named_var('cs1')
        cs2 = bdd.named_var('cs2')
        cs1p = bdd.named_var("cs1'")
        cs2p = bdd.named_var("cs2'")

        # Neither in CS initially
        bts.init = bdd.AND(bdd.NOT(cs1), bdd.NOT(cs2))
        # Mutex: at most one can enter CS
        bts.trans = bdd.AND(
            bdd.NOT(bdd.AND(cs1p, cs2p)),  # mutual exclusion on next
            bdd.TRUE  # otherwise nondeterministic
        )

        mc = LTLModelChecker(bts)
        # Safety: G(!(cs1 & cs2))
        result = mc.check(G(Not(And(Atom("cs1"), Atom("cs2")))))
        assert result.result == LTLResult.SATISFIED

    def test_mutex_starvation(self):
        """Check if process 1 can starve (never enter CS).
        G(F(cs1)) should be violated if process can be permanently blocked."""
        bdd = BDD()
        bts = make_boolean_ts(bdd, ['cs1', 'cs2'])
        cs1 = bdd.named_var('cs1')
        cs2 = bdd.named_var('cs2')
        cs1p = bdd.named_var("cs1'")
        cs2p = bdd.named_var("cs2'")

        bts.init = bdd.AND(bdd.NOT(cs1), bdd.NOT(cs2))
        # cs2 can hog the CS, cs1 never enters
        bts.trans = bdd.AND(
            bdd.NOT(bdd.AND(cs1p, cs2p)),
            bdd.TRUE
        )

        mc = LTLModelChecker(bts)
        # Can cs1 starve?
        result = mc.check(G(F(Atom("cs1"))))
        # Yes, because there's a path where cs1 is never true
        assert result.result == LTLResult.VIOLATED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
