"""
Tests for V023: LTL Model Checking
"""

import sys
import os
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from ltl_model_checker import (
    # LTL AST
    LTL, LTLOp, Atom, LTLTrue, LTLFalse,
    Not, And, Or, Implies, Iff,
    Next, Finally, Globally, Until, Release, WeakUntil,
    # Utilities
    atoms, nnf, subformulas, until_subformulas,
    # Automata
    ltl_to_gba, gba_to_nba, GBA, NBA,
    # Model checking
    LTLModelChecker, LTLResult, FairModelChecker,
    check_ltl, check_ltl_fair, check_ltl_boolean, check_fair_cycle,
    compare_ltl_ctl,
    # Parser
    parse_ltl,
    # BDD imports
    BDD, BooleanTS, make_boolean_ts, SymbolicModelChecker,
)


# ============================================================
# Section 1: LTL Formula Construction
# ============================================================

class TestLTLConstruction:
    def test_atom(self):
        p = Atom("p")
        assert p.op == LTLOp.ATOM
        assert p.name == "p"

    def test_boolean_ops(self):
        p, q = Atom("p"), Atom("q")
        assert And(p, q).op == LTLOp.AND
        assert Or(p, q).op == LTLOp.OR
        assert Not(p).op == LTLOp.NOT
        assert Implies(p, q).op == LTLOp.IMPLIES
        assert Iff(p, q).op == LTLOp.IFF

    def test_temporal_ops(self):
        p = Atom("p")
        assert Next(p).op == LTLOp.X
        assert Finally(p).op == LTLOp.F
        assert Globally(p).op == LTLOp.G
        assert Until(p, Atom("q")).op == LTLOp.U
        assert Release(p, Atom("q")).op == LTLOp.R

    def test_simplifications(self):
        p = Atom("p")
        # NOT(NOT(p)) = p
        assert Not(Not(p)) == p
        # NOT(true) = false
        assert Not(LTLTrue()).op == LTLOp.FALSE
        # AND(true, p) = p
        assert And(LTLTrue(), p) == p
        # OR(false, p) = p
        assert Or(LTLFalse(), p) == p

    def test_repr(self):
        p = Atom("p")
        assert repr(Globally(p)) == "G(p)"
        assert repr(Finally(p)) == "F(p)"
        assert repr(Next(p)) == "X(p)"


# ============================================================
# Section 2: LTL Formula Utilities
# ============================================================

class TestLTLUtilities:
    def test_atoms_extraction(self):
        f = And(Atom("p"), Or(Atom("q"), Atom("r")))
        assert atoms(f) == {"p", "q", "r"}

    def test_atoms_with_temporal(self):
        f = Globally(Implies(Atom("req"), Finally(Atom("ack"))))
        assert atoms(f) == {"req", "ack"}

    def test_nnf_basic(self):
        p = Atom("p")
        # NOT(NOT(p)) -> p
        assert nnf(Not(Not(p))) == p

    def test_nnf_demorgan(self):
        p, q = Atom("p"), Atom("q")
        # NOT(AND(p,q)) -> OR(NOT(p), NOT(q))
        result = nnf(Not(And(p, q)))
        assert result.op == LTLOp.OR

    def test_nnf_temporal_duality(self):
        p = Atom("p")
        # NOT(G(p)) -> F(NOT(p)) -> true U NOT(p)
        result = nnf(Not(Globally(p)))
        assert result.op == LTLOp.U

    def test_nnf_until_release(self):
        p, q = Atom("p"), Atom("q")
        # NOT(p U q) -> NOT(p) R NOT(q)
        result = nnf(Not(Until(p, q)))
        assert result.op == LTLOp.R

    def test_subformulas(self):
        p = Atom("p")
        f = Globally(p)
        sfs = subformulas(f)
        assert f in sfs
        assert p in sfs

    def test_until_subformulas(self):
        p, q = Atom("p"), Atom("q")
        f = And(Until(p, q), Globally(p))
        us = until_subformulas(f)
        assert len(us) >= 1


# ============================================================
# Section 3: LTL Parser
# ============================================================

class TestLTLParser:
    def test_atom(self):
        f = parse_ltl("p")
        assert f.op == LTLOp.ATOM
        assert f.name == "p"

    def test_true_false(self):
        assert parse_ltl("true").op == LTLOp.TRUE
        assert parse_ltl("false").op == LTLOp.FALSE

    def test_negation(self):
        f = parse_ltl("!p")
        assert f.op == LTLOp.NOT

    def test_temporal(self):
        assert parse_ltl("G p").op == LTLOp.G
        assert parse_ltl("F p").op == LTLOp.F
        assert parse_ltl("X p").op == LTLOp.X

    def test_binary(self):
        f = parse_ltl("p & q")
        assert f.op == LTLOp.AND
        f = parse_ltl("p | q")
        assert f.op == LTLOp.OR
        f = parse_ltl("p U q")
        assert f.op == LTLOp.U

    def test_implies(self):
        f = parse_ltl("p -> q")
        assert f.op == LTLOp.IMPLIES

    def test_parenthesized(self):
        f = parse_ltl("G (p -> F q)")
        assert f.op == LTLOp.G

    def test_complex(self):
        f = parse_ltl("G (req -> F ack)")
        assert f.op == LTLOp.G
        inner = f.left
        assert inner.op == LTLOp.IMPLIES
        assert inner.right.op == LTLOp.F


# ============================================================
# Section 4: GBA Construction
# ============================================================

class TestGBAConstruction:
    def test_atom_gba(self):
        gba = ltl_to_gba(Atom("p"))
        assert len(gba.states) > 0
        assert len(gba.initial) > 0
        assert "p" in gba.ap

    def test_globally_gba(self):
        f = nnf(Globally(Atom("p")))
        gba = ltl_to_gba(f)
        assert len(gba.states) > 0

    def test_finally_gba(self):
        f = nnf(Finally(Atom("p")))
        gba = ltl_to_gba(f)
        assert len(gba.states) > 0
        # F(p) = true U p has one Until subformula -> one acceptance set
        assert len(gba.acceptance) >= 1

    def test_until_gba(self):
        f = Until(Atom("p"), Atom("q"))
        gba = ltl_to_gba(f)
        assert len(gba.acceptance) >= 1
        assert "p" in gba.ap
        assert "q" in gba.ap


# ============================================================
# Section 5: NBA Conversion
# ============================================================

class TestNBAConversion:
    def test_single_acceptance(self):
        gba = ltl_to_gba(Atom("p"))
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0
        assert len(nba.initial) > 0

    def test_multiple_acceptance(self):
        # Two Until subformulas -> two acceptance sets in GBA
        p, q, r = Atom("p"), Atom("q"), Atom("r")
        f = And(Until(p, q), Until(q, r))
        gba = ltl_to_gba(nnf(f))
        nba = gba_to_nba(gba)
        # NBA should have more states than GBA (product with counter)
        assert len(nba.states) > 0
        assert len(nba.accepting) > 0

    def test_nba_transitions(self):
        nba = gba_to_nba(ltl_to_gba(Atom("p")))
        # Should have transitions
        total_trans = sum(len(v) for v in nba.transitions.values())
        assert total_trans > 0


# ============================================================
# Section 6: Simple Safety Properties (G)
# ============================================================

class TestSafetyProperties:
    def _make_toggle_system(self):
        """Toggle system: x alternates between 0 and 1."""
        def init_fn(bdd):
            x = bdd.named_var("x")
            return bdd.NOT(x)  # x = 0

        def trans_fn(bdd, cv, nv):
            x = bdd.var(cv["x"])
            x_next = bdd.var(nv["x'"])
            # x' = !x
            return bdd.IFF(x_next, bdd.NOT(x))

        return ["x"], init_fn, trans_fn

    def test_g_true(self):
        """G(true) should always hold."""
        sv, init_fn, trans_fn = self._make_toggle_system()
        result = check_ltl(sv, init_fn, trans_fn, Globally(LTLTrue()))
        assert result.holds

    def test_g_false(self):
        """G(false) should not hold (unless no initial states)."""
        sv, init_fn, trans_fn = self._make_toggle_system()
        result = check_ltl(sv, init_fn, trans_fn, Globally(LTLFalse()))
        assert not result.holds

    def test_g_or(self):
        """G(x | !x) = G(true) should hold."""
        sv, init_fn, trans_fn = self._make_toggle_system()
        result = check_ltl(sv, init_fn, trans_fn,
                          Globally(Or(Atom("x"), Not(Atom("x")))))
        assert result.holds


# ============================================================
# Section 7: Liveness Properties (F)
# ============================================================

class TestLivenessProperties:
    def _make_toggle_system(self):
        def init_fn(bdd):
            return bdd.NOT(bdd.named_var("x"))

        def trans_fn(bdd, cv, nv):
            x = bdd.var(cv["x"])
            x_next = bdd.var(nv["x'"])
            return bdd.IFF(x_next, bdd.NOT(x))

        return ["x"], init_fn, trans_fn

    def test_f_x_holds(self):
        """F(x) should hold -- x eventually becomes true."""
        sv, init_fn, trans_fn = self._make_toggle_system()
        result = check_ltl(sv, init_fn, trans_fn, Finally(Atom("x")))
        assert result.holds

    def test_f_not_x_holds(self):
        """F(!x) should hold -- x eventually becomes false."""
        sv, init_fn, trans_fn = self._make_toggle_system()
        result = check_ltl(sv, init_fn, trans_fn, Finally(Not(Atom("x"))))
        assert result.holds

    def test_gf_x(self):
        """GF(x) -- x is true infinitely often (liveness)."""
        sv, init_fn, trans_fn = self._make_toggle_system()
        result = check_ltl(sv, init_fn, trans_fn,
                          Globally(Finally(Atom("x"))))
        assert result.holds


# ============================================================
# Section 8: Until Properties
# ============================================================

class TestUntilProperties:
    def _make_counter(self):
        """2-bit counter: 00 -> 01 -> 10 -> 11 -> 00 ..."""
        def init_fn(bdd):
            b0 = bdd.named_var("b0")
            b1 = bdd.named_var("b1")
            return bdd.AND(bdd.NOT(b0), bdd.NOT(b1))

        def trans_fn(bdd, cv, nv):
            b0 = bdd.var(cv["b0"])
            b1 = bdd.var(cv["b1"])
            b0n = bdd.var(nv["b0'"])
            b1n = bdd.var(nv["b1'"])
            # Increment: b0' = !b0, b1' = b1 XOR b0
            t1 = bdd.IFF(b0n, bdd.NOT(b0))
            t2 = bdd.IFF(b1n, bdd.XOR(b1, b0))
            return bdd.AND(t1, t2)

        return ["b0", "b1"], init_fn, trans_fn

    def test_until_basic(self):
        """!b1 U b0: b1 stays false until b0 becomes true."""
        sv, init_fn, trans_fn = self._make_counter()
        result = check_ltl(sv, init_fn, trans_fn,
                          Until(Not(Atom("b1")), Atom("b0")))
        # State 00 -> 01: b0 becomes true while b1 is still false
        assert result.holds

    def test_until_eventually(self):
        """true U b1: b1 eventually becomes true."""
        sv, init_fn, trans_fn = self._make_counter()
        result = check_ltl(sv, init_fn, trans_fn,
                          Until(LTLTrue(), Atom("b1")))
        assert result.holds


# ============================================================
# Section 9: Response/Request Patterns
# ============================================================

class TestResponsePatterns:
    def _make_req_ack(self):
        """Simple request-acknowledge system.
        States: idle(00), req(10), ack(11), done(01) -> idle
        req = bit 0, ack = bit 1
        """
        def init_fn(bdd):
            req = bdd.named_var("req")
            ack = bdd.named_var("ack")
            return bdd.AND(bdd.NOT(req), bdd.NOT(ack))  # idle

        def trans_fn(bdd, cv, nv):
            req = bdd.var(cv["req"])
            ack = bdd.var(cv["ack"])
            req_n = bdd.var(nv["req'"])
            ack_n = bdd.var(nv["ack'"])

            # idle -> req: req'=1, ack'=0
            idle = bdd.AND(bdd.NOT(req), bdd.NOT(ack))
            t_idle = bdd.and_all([idle, req_n, bdd.NOT(ack_n)])

            # req -> ack: req'=1, ack'=1
            req_state = bdd.AND(req, bdd.NOT(ack))
            t_req = bdd.and_all([req_state, req_n, ack_n])

            # ack -> done: req'=0, ack'=1
            ack_state = bdd.AND(req, ack)
            t_ack = bdd.and_all([ack_state, bdd.NOT(req_n), ack_n])

            # done -> idle: req'=0, ack'=0
            done = bdd.AND(bdd.NOT(req), ack)
            t_done = bdd.and_all([done, bdd.NOT(req_n), bdd.NOT(ack_n)])

            return bdd.or_all([t_idle, t_req, t_ack, t_done])

        return ["req", "ack"], init_fn, trans_fn

    def test_response(self):
        """G(req -> F ack): every request eventually gets acknowledged."""
        sv, init_fn, trans_fn = self._make_req_ack()
        result = check_ltl(sv, init_fn, trans_fn,
                          Globally(Implies(Atom("req"), Finally(Atom("ack")))))
        assert result.holds

    def test_no_spontaneous_ack(self):
        """G(!req -> !ack) does NOT hold (ack persists in ack->done transition)."""
        sv, init_fn, trans_fn = self._make_req_ack()
        result = check_ltl(sv, init_fn, trans_fn,
                          Globally(Implies(Not(Atom("req")), Not(Atom("ack")))))
        # After ack state (req=1,ack=1) -> done (req=0,ack=1): !req but ack
        assert not result.holds


# ============================================================
# Section 10: Counterexample Generation
# ============================================================

class TestCounterexamples:
    def test_counterexample_exists(self):
        """When property fails, counterexample should be provided."""
        def init_fn(bdd):
            return bdd.NOT(bdd.named_var("x"))

        def trans_fn(bdd, cv, nv):
            x = bdd.var(cv["x"])
            x_next = bdd.var(nv["x'"])
            return bdd.IFF(x_next, bdd.NOT(x))

        # G(x) fails because x starts as false
        result = check_ltl(["x"], init_fn, trans_fn, Globally(Atom("x")))
        assert not result.holds
        assert result.counterexample is not None

    def test_counterexample_structure(self):
        """Counterexample is (prefix, cycle)."""
        def init_fn(bdd):
            return bdd.NOT(bdd.named_var("x"))

        def trans_fn(bdd, cv, nv):
            x_next = bdd.var(nv["x'"])
            return bdd.NOT(x_next)  # x always false

        # F(x) fails because x is always false
        result = check_ltl(["x"], init_fn, trans_fn, Finally(Atom("x")))
        assert not result.holds
        if result.counterexample:
            prefix, cycle = result.counterexample
            assert isinstance(prefix, list)
            assert isinstance(cycle, list)


# ============================================================
# Section 11: Fairness Constraints
# ============================================================

class TestFairnessConstraints:
    def _make_nondeterministic(self):
        """Nondeterministic system: x can be 0 or 1 at each step."""
        def init_fn(bdd):
            return bdd.TRUE  # any initial state

        def trans_fn(bdd, cv, nv):
            return bdd.TRUE  # any transition

        return ["x"], init_fn, trans_fn

    def test_justice_constraint(self):
        """With justice GF(x), unfair paths where x is always false are excluded."""
        sv, init_fn, trans_fn = self._make_nondeterministic()

        # Without fairness, G(!x) would hold on some paths
        # With GF(x) justice, all fair paths visit x infinitely often
        result = check_ltl_fair(
            sv, init_fn, trans_fn,
            Globally(LTLTrue()),  # trivially true under any fairness
            justice=[lambda bdd: bdd.named_var("x")]
        )
        assert result.holds

    def test_no_fairness(self):
        """Without fairness, use standard LTL checking."""
        sv, init_fn, trans_fn = self._make_nondeterministic()
        result = check_ltl_fair(
            sv, init_fn, trans_fn,
            Globally(LTLTrue())
        )
        assert result.holds

    def test_fair_cycle_check(self):
        """Check if a fair cycle exists in a toggle system."""
        bdd = BDD()
        x = bdd.named_var("x")
        x_next = bdd.named_var("x'")

        ts = BooleanTS(
            bdd=bdd,
            state_vars=["x"],
            next_vars=["x'"],
            init=bdd.NOT(x),
            trans=bdd.IFF(x_next, bdd.NOT(x)),
            var_indices={"x": bdd.var_index("x")},
            next_indices={"x": bdd.var_index("x'")}
        )

        # Justice: must visit x=true infinitely often (toggle satisfies this)
        result = check_fair_cycle(ts, justice=[x])
        assert result.holds


# ============================================================
# Section 12: Complex Properties
# ============================================================

class TestComplexProperties:
    def _make_mutex(self):
        """Simple mutual exclusion: two processes, at most one in CS."""
        def init_fn(bdd):
            p1 = bdd.named_var("p1")
            p2 = bdd.named_var("p2")
            return bdd.AND(bdd.NOT(p1), bdd.NOT(p2))  # both idle

        def trans_fn(bdd, cv, nv):
            p1 = bdd.var(cv["p1"])
            p2 = bdd.var(cv["p2"])
            p1n = bdd.var(nv["p1'"])
            p2n = bdd.var(nv["p2'"])

            # Only one can enter CS at a time
            # p1 can enter if p2 is not in CS
            can_enter_1 = bdd.AND(bdd.NOT(p1), bdd.NOT(p2))
            # p1 can exit
            can_exit_1 = p1

            # Transition: either p1 toggles (if allowed) or stays
            # Let's make it simpler: p1' XOR p1 allowed only if NOT(p1 AND p2')
            # and p2' XOR p2 allowed only if NOT(p2 AND p1')

            # Simplification: nondeterministic but mutex-respecting
            # mutex: !(p1' & p2')
            mutex = bdd.NOT(bdd.AND(p1n, p2n))

            return mutex  # any transition respecting mutex

        return ["p1", "p2"], init_fn, trans_fn

    def test_mutual_exclusion(self):
        """G(!(p1 & p2)): mutual exclusion always holds."""
        sv, init_fn, trans_fn = self._make_mutex()
        result = check_ltl(sv, init_fn, trans_fn,
                          Globally(Not(And(Atom("p1"), Atom("p2")))))
        assert result.holds

    def test_no_starvation_may_fail(self):
        """G(F(p1)) may not hold without fairness (p1 might never enter)."""
        sv, init_fn, trans_fn = self._make_mutex()
        # Without fairness, p1 might never get to enter (unfair scheduling)
        result = check_ltl(sv, init_fn, trans_fn,
                          Globally(Finally(Atom("p1"))))
        # This can fail (unfair path where p1 never enters)
        # The nondeterministic system allows paths where p1 stays false
        assert not result.holds


# ============================================================
# Section 13: CTL vs LTL Comparison
# ============================================================

class TestCTLComparison:
    def test_compare_safety(self):
        """Compare LTL G(p|!p) with CTL AG(p|!p) on toggle system."""
        def init_fn(bdd):
            return bdd.NOT(bdd.named_var("x"))

        def trans_fn(bdd, cv, nv):
            x = bdd.var(cv["x"])
            x_next = bdd.var(nv["x'"])
            return bdd.IFF(x_next, bdd.NOT(x))

        ltl_f = Globally(Or(Atom("x"), Not(Atom("x"))))

        def ctl_check(mc):
            bdd = mc.ts.bdd
            x = bdd.var(mc.ts.var_indices["x"])
            prop = bdd.OR(x, bdd.NOT(x))
            output = mc.check_safety(prop)
            from bdd_model_checker import MCResult
            return output.result == MCResult.SAFE

        result = compare_ltl_ctl(["x"], init_fn, trans_fn, ltl_f, ctl_check)
        assert result['ltl_holds'] == True
        assert result['agree'] == True


# ============================================================
# Section 14: Existing BooleanTS Integration
# ============================================================

class TestBooleanTSIntegration:
    def test_check_existing_ts(self):
        """Check LTL on a pre-built BooleanTS."""
        bdd = BDD()
        x = bdd.named_var("x")
        x_next = bdd.named_var("x'")

        ts = BooleanTS(
            bdd=bdd,
            state_vars=["x"],
            next_vars=["x'"],
            init=bdd.NOT(x),
            trans=bdd.IFF(x_next, bdd.NOT(x)),
            var_indices={"x": bdd.var_index("x")},
            next_indices={"x": bdd.var_index("x'")}
        )

        result = check_ltl_boolean(ts, Finally(Atom("x")))
        assert result.holds

    def test_result_metadata(self):
        """LTLResult should include automaton/product info."""
        bdd = BDD()
        x = bdd.named_var("x")
        x_next = bdd.named_var("x'")

        ts = BooleanTS(
            bdd=bdd,
            state_vars=["x"],
            next_vars=["x'"],
            init=bdd.NOT(x),
            trans=bdd.IFF(x_next, bdd.NOT(x)),
            var_indices={"x": bdd.var_index("x")},
            next_indices={"x": bdd.var_index("x'")}
        )

        result = check_ltl_boolean(ts, Globally(LTLTrue()))
        assert result.holds
        assert result.automaton_states >= 1
        assert result.product_vars >= 1


# ============================================================
# Section 15: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_state_system(self):
        """System with one state (self-loop)."""
        def init_fn(bdd):
            return bdd.NOT(bdd.named_var("x"))

        def trans_fn(bdd, cv, nv):
            x = bdd.var(cv["x"])
            x_next = bdd.var(nv["x'"])
            # x stays false
            return bdd.AND(bdd.NOT(x), bdd.NOT(x_next))

        # G(!x) should hold
        result = check_ltl(["x"], init_fn, trans_fn, Globally(Not(Atom("x"))))
        assert result.holds

        # F(x) should not hold
        result = check_ltl(["x"], init_fn, trans_fn, Finally(Atom("x")))
        assert not result.holds

    def test_trivial_true(self):
        """G(true) always holds."""
        def init_fn(bdd):
            return bdd.TRUE

        def trans_fn(bdd, cv, nv):
            return bdd.TRUE

        result = check_ltl(["x"], init_fn, trans_fn, Globally(LTLTrue()))
        assert result.holds

    def test_next_operator(self):
        """X(x) on toggle: x is true at step 1."""
        def init_fn(bdd):
            return bdd.NOT(bdd.named_var("x"))

        def trans_fn(bdd, cv, nv):
            x = bdd.var(cv["x"])
            x_next = bdd.var(nv["x'"])
            return bdd.IFF(x_next, bdd.NOT(x))

        # X(x): at step 1, x is true (starts false, toggles to true)
        result = check_ltl(["x"], init_fn, trans_fn, Next(Atom("x")))
        assert result.holds

    def test_weak_until(self):
        """p W q = (p U q) | G(p): p holds until q, or p holds forever."""
        def init_fn(bdd):
            return bdd.named_var("x")  # x starts true

        def trans_fn(bdd, cv, nv):
            x_next = bdd.var(nv["x'"])
            return x_next  # x always stays true

        # x W false = G(x): x holds forever
        result = check_ltl(["x"], init_fn, trans_fn,
                          WeakUntil(Atom("x"), LTLFalse()))
        assert result.holds

    def test_release(self):
        """p R q = !(!p U !q): q holds until p releases it (or q holds forever)."""
        def init_fn(bdd):
            return bdd.named_var("x")  # x = true

        def trans_fn(bdd, cv, nv):
            x_next = bdd.var(nv["x'"])
            return x_next  # x stays true

        # true R x = G(x): since true is always available to release, x must hold
        # until released, but true releases immediately... actually:
        # p R q = q must hold until and including when p first holds (or forever)
        # true R x: x must hold at least until true holds (which is immediately)
        # So x must hold at step 0, which it does.
        # But for all paths, this means x must hold at step 0 = true
        result = check_ltl(["x"], init_fn, trans_fn,
                          Release(LTLTrue(), Atom("x")))
        assert result.holds

    def test_multi_var_system(self):
        """System with multiple variables."""
        def init_fn(bdd):
            a = bdd.named_var("a")
            b = bdd.named_var("b")
            return bdd.AND(a, bdd.NOT(b))  # a=1, b=0

        def trans_fn(bdd, cv, nv):
            a = bdd.var(cv["a"])
            b = bdd.var(cv["b"])
            an = bdd.var(nv["a'"])
            bn = bdd.var(nv["b'"])
            # Swap: a'=b, b'=a
            return bdd.AND(bdd.IFF(an, b), bdd.IFF(bn, a))

        # G(a | b): at least one is always true (a=1,b=0 -> a=0,b=1 -> ...)
        result = check_ltl(["a", "b"], init_fn, trans_fn,
                          Globally(Or(Atom("a"), Atom("b"))))
        assert result.holds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
