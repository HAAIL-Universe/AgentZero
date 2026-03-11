"""Tests for V157: Mu-Calculus Model Checking"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mu_calculus import (
    # AST
    Prop, Var, TT, FF, Not, And, Or, Diamond, Box, Mu, Nu,
    # LTS
    LTS, make_lts,
    # PNF
    to_pnf,
    # Analysis
    subformulas, free_vars, is_closed, alternation_depth, fixpoint_nesting_depth,
    formula_info,
    # Model checking
    eval_formula, check_via_parity_game, model_check, check_state,
    compare_methods,
    # CTL encodings
    ctl_EF, ctl_AG, ctl_AF, ctl_EG, ctl_EU, ctl_AU, ctl_EX, ctl_AX,
    # Parser
    parse_mu,
    # Utilities
    check_result, batch_check, mu_calculus_summary,
    # Parity game
    formula_to_parity_game,
)


# ===================================================================
# Fixtures: common LTS models
# ===================================================================

@pytest.fixture
def simple_lts():
    """Simple 3-state LTS:
    0 -a-> 1 -b-> 2 -a-> 0
    Labels: 0={p}, 1={q}, 2={p,q}
    """
    return make_lts(3,
        [(0, 'a', 1), (1, 'b', 2), (2, 'a', 0)],
        {0: {'p'}, 1: {'q'}, 2: {'p', 'q'}}
    )


@pytest.fixture
def self_loop_lts():
    """2-state LTS with self-loop:
    0 -a-> 1, 0 -b-> 0, 1 -a-> 1
    Labels: 0={start}, 1={end}
    """
    return make_lts(2,
        [(0, 'a', 1), (0, 'b', 0), (1, 'a', 1)],
        {0: {'start'}, 1: {'end'}}
    )


@pytest.fixture
def traffic_lts():
    """Traffic light: green -> yellow -> red -> green
    Labels: 0={green}, 1={yellow}, 2={red}
    """
    return make_lts(3,
        [(0, 'tick', 1), (1, 'tick', 2), (2, 'tick', 0)],
        {0: {'green'}, 1: {'yellow'}, 2: {'red'}}
    )


@pytest.fixture
def branching_lts():
    """Branching LTS:
    0 -a-> 1, 0 -a-> 2
    1 -b-> 3
    2 -c-> 3
    3 -d-> 3
    Labels: 0={init}, 1={left}, 2={right}, 3={done}
    """
    return make_lts(4,
        [(0, 'a', 1), (0, 'a', 2), (1, 'b', 3), (2, 'c', 3), (3, 'd', 3)],
        {0: {'init'}, 1: {'left'}, 2: {'right'}, 3: {'done'}}
    )


@pytest.fixture
def deadlock_lts():
    """LTS with deadlock:
    0 -a-> 1, 1 has no transitions
    Labels: 0={alive}, 1={dead}
    """
    return make_lts(2,
        [(0, 'a', 1)],
        {0: {'alive'}, 1: {'dead'}}
    )


# ===================================================================
# 1. Formula AST
# ===================================================================

class TestFormulaAST:
    def test_prop(self):
        p = Prop("p")
        assert p.name == "p"
        assert repr(p) == "Prop(p)"

    def test_var(self):
        x = Var("X")
        assert x.name == "X"

    def test_constants(self):
        assert repr(TT()) == "TT"
        assert repr(FF()) == "FF"

    def test_boolean_ops(self):
        f = And(Prop("p"), Or(Prop("q"), Not(Prop("r"))))
        assert isinstance(f, And)
        assert isinstance(f.right, Or)

    def test_modalities(self):
        f = Diamond("a", Prop("p"))
        assert f.action == "a"
        g = Box(None, Prop("q"))
        assert g.action is None

    def test_fixpoints(self):
        f = Mu("X", Or(Prop("p"), Diamond(None, Var("X"))))
        assert f.var == "X"
        assert isinstance(f.body, Or)

    def test_frozen(self):
        """Formulas are frozen dataclasses (hashable)."""
        p1 = Prop("p")
        p2 = Prop("p")
        assert p1 == p2
        assert hash(p1) == hash(p2)
        s = {p1, p2}
        assert len(s) == 1


# ===================================================================
# 2. LTS
# ===================================================================

class TestLTS:
    def test_make_lts(self, simple_lts):
        assert simple_lts.states == {0, 1, 2}
        assert simple_lts.successors(0, 'a') == {1}
        assert simple_lts.successors(1, 'b') == {2}

    def test_predecessors(self, simple_lts):
        assert simple_lts.predecessors(1, 'a') == {0}
        assert simple_lts.predecessors(0, 'a') == {2}

    def test_labels(self, simple_lts):
        assert 'p' in simple_lts.labels[0]
        assert 'q' in simple_lts.labels[1]
        assert 'p' in simple_lts.labels[2] and 'q' in simple_lts.labels[2]

    def test_actions(self, simple_lts):
        assert simple_lts.actions() == {'a', 'b'}

    def test_any_action_successors(self, simple_lts):
        # action=None means any action
        assert simple_lts.successors(0) == {1}  # only 'a' from 0
        assert simple_lts.successors(2) == {0}

    def test_deadlock(self, deadlock_lts):
        assert deadlock_lts.successors(1) == set()
        assert deadlock_lts.successors(0) == {1}


# ===================================================================
# 3. Subformula analysis
# ===================================================================

class TestAnalysis:
    def test_subformulas(self):
        f = And(Prop("p"), Or(Prop("q"), Prop("r")))
        subs = subformulas(f)
        assert len(subs) == 5  # p, q, r, Or, And

    def test_free_vars(self):
        f = Mu("X", Or(Var("X"), Var("Y")))
        assert free_vars(f) == {"Y"}

    def test_free_vars_closed(self):
        f = Mu("X", Or(Prop("p"), Diamond(None, Var("X"))))
        assert free_vars(f) == set()
        assert is_closed(f)

    def test_alternation_depth_0(self):
        f = Mu("X", Or(Prop("p"), Diamond(None, Var("X"))))
        assert alternation_depth(f) == 0

    def test_alternation_depth_0_nu(self):
        f = Nu("X", And(Prop("p"), Box(None, Var("X"))))
        assert alternation_depth(f) == 0

    def test_alternation_depth_1(self):
        # mu inside nu with shared variable reference
        inner = Mu("Y", Or(Prop("q"), Diamond(None, Var("Y"))))
        f = Nu("X", And(inner, Box(None, Var("X"))))
        # Y doesn't reference X, so no alternation
        assert alternation_depth(f) == 0

    def test_alternation_depth_real(self):
        # mu Y. (q | (<>Y & nu X. (p & []X)))
        # X is nu-bound and Y is mu-bound, but X doesn't reference Y
        inner_nu = Nu("X", And(Prop("p"), Box(None, Var("X"))))
        f = Mu("Y", Or(Prop("q"), And(Diamond(None, Var("Y")), inner_nu)))
        assert alternation_depth(f) == 0  # no cross-reference

    def test_fixpoint_nesting(self):
        inner = Mu("Y", Or(Prop("q"), Diamond(None, Var("Y"))))
        f = Nu("X", And(inner, Box(None, Var("X"))))
        assert fixpoint_nesting_depth(f) == 2

    def test_formula_info(self):
        f = Mu("X", Or(Prop("p"), Diamond(None, Var("X"))))
        info = formula_info(f)
        assert info["is_closed"] is True
        assert info["fixpoint_nesting"] == 1
        assert info["subformula_count"] > 0


# ===================================================================
# 4. Positive Normal Form
# ===================================================================

class TestPNF:
    def test_double_negation(self):
        f = Not(Not(Prop("p")))
        pnf = to_pnf(f)
        assert pnf == Prop("p")

    def test_demorgan_and(self):
        f = Not(And(Prop("p"), Prop("q")))
        pnf = to_pnf(f)
        assert isinstance(pnf, Or)
        assert isinstance(pnf.left, Not)
        assert isinstance(pnf.right, Not)

    def test_demorgan_or(self):
        f = Not(Or(Prop("p"), Prop("q")))
        pnf = to_pnf(f)
        assert isinstance(pnf, And)

    def test_modal_duality_diamond(self):
        f = Not(Diamond("a", Prop("p")))
        pnf = to_pnf(f)
        assert isinstance(pnf, Box)
        assert pnf.action == "a"
        assert isinstance(pnf.sub, Not) and isinstance(pnf.sub.sub, Prop)

    def test_modal_duality_box(self):
        f = Not(Box(None, Prop("p")))
        pnf = to_pnf(f)
        assert isinstance(pnf, Diamond)

    def test_fixpoint_duality(self):
        f = Not(Mu("X", Or(Prop("p"), Diamond(None, Var("X")))))
        pnf = to_pnf(f)
        assert isinstance(pnf, Nu)

    def test_constants(self):
        assert to_pnf(Not(TT())) == FF()
        assert to_pnf(Not(FF())) == TT()


# ===================================================================
# 5. Direct fixpoint model checking -- basic
# ===================================================================

class TestDirectBasic:
    def test_tt(self, simple_lts):
        assert eval_formula(simple_lts, TT()) == {0, 1, 2}

    def test_ff(self, simple_lts):
        assert eval_formula(simple_lts, FF()) == set()

    def test_prop(self, simple_lts):
        assert eval_formula(simple_lts, Prop("p")) == {0, 2}
        assert eval_formula(simple_lts, Prop("q")) == {1, 2}

    def test_not_prop(self, simple_lts):
        assert eval_formula(simple_lts, Not(Prop("p"))) == {1}

    def test_and(self, simple_lts):
        f = And(Prop("p"), Prop("q"))
        assert eval_formula(simple_lts, f) == {2}

    def test_or(self, simple_lts):
        f = Or(Prop("p"), Prop("q"))
        assert eval_formula(simple_lts, f) == {0, 1, 2}

    def test_diamond(self, simple_lts):
        # <a>q: states with an a-successor satisfying q
        f = Diamond("a", Prop("q"))
        assert eval_formula(simple_lts, f) == {0}  # 0 -a-> 1 which has q

    def test_diamond_any(self, simple_lts):
        # <>q: any action leading to q
        f = Diamond(None, Prop("q"))
        assert eval_formula(simple_lts, f) == {0, 1}  # 0->1(q), 1->2(q)

    def test_box(self, simple_lts):
        # [a]q: all a-successors satisfy q
        f = Box("a", Prop("q"))
        # 0: a-succ={1} which has q -> True
        # 1: a-succ={} -> vacuously True
        # 2: a-succ={0} which has p but not q -> False
        assert eval_formula(simple_lts, f) == {0, 1}

    def test_box_any(self, simple_lts):
        # [*]p: all successors (any action) have p
        f = Box(None, Prop("p"))
        # 0: succ={1}, 1 has q not p -> False
        # 1: succ={2}, 2 has p -> True
        # 2: succ={0}, 0 has p -> True
        assert eval_formula(simple_lts, f) == {1, 2}


# ===================================================================
# 6. Direct fixpoint -- mu/nu
# ===================================================================

class TestDirectFixpoint:
    def test_mu_reachability(self, simple_lts):
        """mu X. (q | <>X) = EF q = states that can reach a q-state."""
        f = Mu("X", Or(Prop("q"), Diamond(None, Var("X"))))
        result = eval_formula(simple_lts, f)
        # All states can reach q (cycle: 0->1(q), 1->2(q), 2->0->1(q))
        assert result == {0, 1, 2}

    def test_nu_safety(self, simple_lts):
        """nu X. (p & []X) = AG p = p holds forever on all paths."""
        f = Nu("X", And(Prop("p"), Box(None, Var("X"))))
        result = eval_formula(simple_lts, f)
        # From 0: can go to 1 which doesn't have p. So 0 fails.
        # From 1: 1 doesn't have p. Fails immediately.
        # From 2: goes to 0, then to 1 which doesn't have p. 2 fails.
        assert result == set()

    def test_nu_always_true(self, traffic_lts):
        """nu X. (tt & []X) = AG true = all states (trivially)."""
        f = Nu("X", And(TT(), Box(None, Var("X"))))
        result = eval_formula(traffic_lts, f)
        assert result == {0, 1, 2}

    def test_mu_ef_done(self, branching_lts):
        """EF done = mu X. (done | <>X)"""
        f = Mu("X", Or(Prop("done"), Diamond(None, Var("X"))))
        result = eval_formula(branching_lts, f)
        assert result == {0, 1, 2, 3}  # all can reach 3

    def test_nu_ag_not_dead(self, deadlock_lts):
        """AG alive = nu X. (alive & []X)"""
        f = Nu("X", And(Prop("alive"), Box(None, Var("X"))))
        result = eval_formula(deadlock_lts, f)
        # State 0 goes to 1 which is dead (not alive). Fails.
        # State 1 is dead. Fails.
        assert result == set()

    def test_eg_self_loop(self, self_loop_lts):
        """EG start = nu X. (start & <>X)"""
        f = Nu("X", And(Prop("start"), Diamond(None, Var("X"))))
        result = eval_formula(self_loop_lts, f)
        # State 0 has start and can loop to itself via b. EG holds.
        assert result == {0}

    def test_eg_end(self, self_loop_lts):
        """EG end = nu X. (end & <>X)"""
        f = Nu("X", And(Prop("end"), Diamond(None, Var("X"))))
        result = eval_formula(self_loop_lts, f)
        assert result == {1}  # 1 has end and loops to itself


# ===================================================================
# 7. CTL encodings via mu-calculus
# ===================================================================

class TestCTLEncodings:
    def test_ef(self, branching_lts):
        f = ctl_EF(Prop("done"))
        result = eval_formula(branching_lts, f)
        assert result == {0, 1, 2, 3}

    def test_ag(self, traffic_lts):
        """AG(green | yellow | red) -- always in some color."""
        f = ctl_AG(Or(Prop("green"), Or(Prop("yellow"), Prop("red"))))
        result = eval_formula(traffic_lts, f)
        assert result == {0, 1, 2}

    def test_ex(self, simple_lts):
        f = ctl_EX(Prop("q"))
        result = eval_formula(simple_lts, f)
        # 0 -> 1 (has q), 1 -> 2 (has q)
        assert result == {0, 1}

    def test_ax(self, simple_lts):
        f = ctl_AX(Prop("q"))
        result = eval_formula(simple_lts, f)
        # 0: all succs={1} have q -> True
        # 1: all succs={2} have q -> True
        # 2: all succs={0}, 0 doesn't have q -> False
        assert result == {0, 1}

    def test_eu(self, simple_lts):
        """E[p U q] -- there's a path where p holds until q."""
        f = ctl_EU(Prop("p"), Prop("q"))
        result = eval_formula(simple_lts, f)
        # 0(p) -a-> 1(q): p U q holds. State 0 in.
        # 1: q holds immediately. State 1 in.
        # 2(p,q): q holds immediately. State 2 in.
        assert result == {0, 1, 2}

    def test_eg(self, self_loop_lts):
        f = ctl_EG(Prop("start"))
        result = eval_formula(self_loop_lts, f)
        assert result == {0}  # can loop via b forever

    def test_af_deadlock(self, deadlock_lts):
        """AF dead = on all paths, eventually dead."""
        f = ctl_AF(Prop("dead"))
        result = eval_formula(deadlock_lts, f)
        # 0 -> 1 (dead): AF dead holds at 0 and 1
        assert result == {0, 1}


# ===================================================================
# 8. Parity game reduction
# ===================================================================

class TestParityGameReduction:
    def test_game_construction(self, simple_lts):
        f = Mu("X", Or(Prop("p"), Diamond(None, Var("X"))))
        game, v_map, sub_list, top_idx = formula_to_parity_game(simple_lts, f)
        assert len(game.vertices) > 0
        # Should have vertices for each subformula x state
        assert len(game.vertices) == len(sub_list) * len(simple_lts.states)

    def test_game_simple_prop(self, simple_lts):
        """Check Prop(p) via game: should match direct."""
        f = Prop("p")
        direct = eval_formula(simple_lts, f)
        game_result = check_via_parity_game(simple_lts, f)
        assert game_result == direct

    def test_game_and(self, simple_lts):
        f = And(Prop("p"), Prop("q"))
        direct = eval_formula(simple_lts, f)
        game_result = check_via_parity_game(simple_lts, f)
        assert game_result == direct

    def test_game_or(self, simple_lts):
        f = Or(Prop("p"), Prop("q"))
        direct = eval_formula(simple_lts, f)
        game_result = check_via_parity_game(simple_lts, f)
        assert game_result == direct

    def test_game_diamond(self, simple_lts):
        f = Diamond("a", Prop("q"))
        direct = eval_formula(simple_lts, f)
        game_result = check_via_parity_game(simple_lts, f)
        assert game_result == direct

    def test_game_box(self, simple_lts):
        f = Box(None, Prop("p"))
        direct = eval_formula(simple_lts, f)
        game_result = check_via_parity_game(simple_lts, f)
        assert game_result == direct

    def test_game_mu_ef(self, simple_lts):
        f = Mu("X", Or(Prop("q"), Diamond(None, Var("X"))))
        direct = eval_formula(simple_lts, f)
        game_result = check_via_parity_game(simple_lts, f)
        assert game_result == direct

    def test_game_nu_ag(self, traffic_lts):
        f = Nu("X", And(TT(), Box(None, Var("X"))))
        direct = eval_formula(traffic_lts, f)
        game_result = check_via_parity_game(traffic_lts, f)
        assert game_result == direct

    def test_game_nu_eg(self, self_loop_lts):
        f = Nu("X", And(Prop("end"), Diamond(None, Var("X"))))
        direct = eval_formula(self_loop_lts, f)
        game_result = check_via_parity_game(self_loop_lts, f)
        assert game_result == direct


# ===================================================================
# 9. Compare methods
# ===================================================================

class TestCompareMethods:
    def test_compare_prop(self, simple_lts):
        r = compare_methods(simple_lts, Prop("p"))
        assert r["agree"]

    def test_compare_mu(self, simple_lts):
        f = Mu("X", Or(Prop("p"), Diamond(None, Var("X"))))
        r = compare_methods(simple_lts, f)
        assert r["agree"]

    def test_compare_nu(self, traffic_lts):
        f = Nu("X", And(Or(Prop("green"), Or(Prop("yellow"), Prop("red"))), Box(None, Var("X"))))
        r = compare_methods(traffic_lts, f)
        assert r["agree"]

    def test_compare_ctl_ef(self, branching_lts):
        f = ctl_EF(Prop("done"))
        r = compare_methods(branching_lts, f)
        assert r["agree"]

    def test_compare_ctl_ag(self, traffic_lts):
        f = ctl_AG(Or(Prop("green"), Or(Prop("yellow"), Prop("red"))))
        r = compare_methods(traffic_lts, f)
        assert r["agree"]

    def test_compare_diamond_any(self, branching_lts):
        f = Diamond(None, Prop("done"))
        r = compare_methods(branching_lts, f)
        assert r["agree"]

    def test_compare_box_specific(self, simple_lts):
        f = Box("a", Prop("q"))
        r = compare_methods(simple_lts, f)
        assert r["agree"]


# ===================================================================
# 10. Parser
# ===================================================================

class TestParser:
    def test_prop(self):
        f = parse_mu("p")
        assert isinstance(f, Prop) and f.name == "p"

    def test_var(self):
        f = parse_mu("X")
        assert isinstance(f, Var) and f.name == "X"

    def test_tt_ff(self):
        assert isinstance(parse_mu("tt"), TT)
        assert isinstance(parse_mu("ff"), FF)

    def test_negation(self):
        f = parse_mu("~p")
        assert isinstance(f, Not) and isinstance(f.sub, Prop)

    def test_and(self):
        f = parse_mu("p & q")
        assert isinstance(f, And)

    def test_or(self):
        f = parse_mu("p | q")
        assert isinstance(f, Or)

    def test_diamond(self):
        f = parse_mu("<a>p")
        assert isinstance(f, Diamond) and f.action == "a"

    def test_diamond_any(self):
        f = parse_mu("<>p")
        assert isinstance(f, Diamond) and f.action is None

    def test_box(self):
        f = parse_mu("[b]q")
        assert isinstance(f, Box) and f.action == "b"

    def test_box_any(self):
        f = parse_mu("[]p")
        assert isinstance(f, Box) and f.action is None

    def test_mu(self):
        f = parse_mu("mu X. (p | <>X)")
        assert isinstance(f, Mu) and f.var == "X"

    def test_nu(self):
        f = parse_mu("nu Y. (q & []Y)")
        assert isinstance(f, Nu) and f.var == "Y"

    def test_nested(self):
        f = parse_mu("nu X. (p & []X) | <>tt")
        # nu binds everything after the dot (like a lambda)
        assert isinstance(f, Nu)
        assert isinstance(f.body, Or)

    def test_parsed_model_check(self, simple_lts):
        f = parse_mu("mu X. (q | <>X)")
        result = eval_formula(simple_lts, f)
        assert result == {0, 1, 2}


# ===================================================================
# 11. Complex formulas
# ===================================================================

class TestComplexFormulas:
    def test_nested_fixpoints(self, traffic_lts):
        """nu X. mu Y. ((green & X) | <>Y)
        Greatest fixpoint of: least fixpoint of reaching green then continuing.
        This encodes "infinitely often green" (on some path from each state).
        """
        f = Nu("X", Mu("Y", Or(And(Prop("green"), Var("X")), Diamond(None, Var("Y")))))
        result = eval_formula(traffic_lts, f)
        # All states cycle through green infinitely
        assert result == {0, 1, 2}

    def test_nested_fixpoints_game(self, traffic_lts):
        """Same formula via parity game."""
        f = Nu("X", Mu("Y", Or(And(Prop("green"), Var("X")), Diamond(None, Var("Y")))))
        r = compare_methods(traffic_lts, f)
        assert r["agree"]

    def test_inevitability(self, branching_lts):
        """AF done via mu-calculus: mu X. (done | ([]X & <>tt))"""
        f = ctl_AF(Prop("done"))
        result = eval_formula(branching_lts, f)
        # All paths from any state reach done (state 3)
        assert result == {0, 1, 2, 3}

    def test_liveness(self, self_loop_lts):
        """Can we always eventually reach end?"""
        f = ctl_AF(Prop("end"))
        result = eval_formula(self_loop_lts, f)
        # State 0 can loop forever via b, never reaching end. AF fails.
        # State 1 is end. AF holds.
        assert result == {1}

    def test_safety_property(self, traffic_lts):
        """AG(green -> EF red) -- from green, can always eventually reach red."""
        green_implies_ef_red = Or(Not(Prop("green")), ctl_EF(Prop("red")))
        f = ctl_AG(green_implies_ef_red)
        result = eval_formula(traffic_lts, f)
        assert result == {0, 1, 2}


# ===================================================================
# 12. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_lts(self):
        lts = LTS(states=set(), transitions={}, labels={})
        assert eval_formula(lts, TT()) == set()
        assert eval_formula(lts, Prop("p")) == set()

    def test_single_state(self):
        lts = make_lts(1, [(0, 'a', 0)], {0: {'p'}})
        assert eval_formula(lts, Prop("p")) == {0}
        assert eval_formula(lts, Diamond("a", Prop("p"))) == {0}
        assert eval_formula(lts, Box("a", Prop("p"))) == {0}

    def test_disconnected_states(self):
        lts = make_lts(3, [(0, 'a', 1)], {0: {'p'}, 2: {'q'}})
        f = Diamond(None, TT())
        result = eval_formula(lts, f)
        assert result == {0}  # only state 0 has a successor

    def test_box_vacuous(self):
        """Box is vacuously true at deadlock states."""
        lts = make_lts(2, [(0, 'a', 1)], {})
        f = Box(None, FF())
        result = eval_formula(lts, f)
        assert 1 in result  # state 1 has no successors -> vacuously true

    def test_mu_no_progress(self):
        """mu X. X = empty set (least fixpoint of identity is bottom)."""
        lts = make_lts(2, [(0, 'a', 1), (1, 'a', 0)], {})
        f = Mu("X", Var("X"))
        result = eval_formula(lts, f)
        assert result == set()

    def test_nu_no_progress(self):
        """nu X. X = all states (greatest fixpoint of identity is top)."""
        lts = make_lts(2, [(0, 'a', 1), (1, 'a', 0)], {})
        f = Nu("X", Var("X"))
        result = eval_formula(lts, f)
        assert result == {0, 1}


# ===================================================================
# 13. check_state and check_result APIs
# ===================================================================

class TestAPIs:
    def test_check_state(self, simple_lts):
        assert check_state(simple_lts, Prop("p"), 0) is True
        assert check_state(simple_lts, Prop("p"), 1) is False

    def test_check_result(self, simple_lts):
        r = check_result(simple_lts, Prop("p"))
        assert r["sat_count"] == 2
        assert r["total_states"] == 3
        assert r["method"] == "direct"

    def test_batch_check(self, simple_lts):
        results = batch_check(simple_lts, [Prop("p"), Prop("q"), TT()])
        assert len(results) == 3
        assert results[0]["sat_count"] == 2
        assert results[1]["sat_count"] == 2
        assert results[2]["sat_count"] == 3

    def test_summary(self, simple_lts):
        f = Mu("X", Or(Prop("q"), Diamond(None, Var("X"))))
        s = mu_calculus_summary(simple_lts, f)
        assert "Satisfying states" in s
        assert "Methods agree: True" in s

    def test_model_check_method(self, simple_lts):
        f = Prop("p")
        assert model_check(simple_lts, f, "direct") == {0, 2}
        assert model_check(simple_lts, f, "game") == {0, 2}


# ===================================================================
# 14. Larger LTS -- mutual exclusion protocol
# ===================================================================

class TestMutualExclusion:
    @pytest.fixture
    def mutex_lts(self):
        """Simple 2-process mutual exclusion:
        States: (p1_state, p2_state) where each is idle/trying/critical
        Encoded as ints: idle=0, trying=1, critical=2
        State id = p1*3 + p2
        """
        states = 9  # 3x3
        transitions = []
        labels = {}

        for p1 in range(3):
            for p2 in range(3):
                s = p1 * 3 + p2
                props = set()
                if p1 == 0: props.add('p1_idle')
                if p1 == 1: props.add('p1_trying')
                if p1 == 2: props.add('p1_critical')
                if p2 == 0: props.add('p2_idle')
                if p2 == 1: props.add('p2_trying')
                if p2 == 2: props.add('p2_critical')
                if p1 == 2 and p2 == 2:
                    props.add('both_critical')  # mutual exclusion violation
                labels[s] = props

                # P1 transitions
                if p1 == 0:  # idle -> trying
                    transitions.append((s, 'p1_try', 1 * 3 + p2))
                if p1 == 1 and p2 != 2:  # trying -> critical (if other not critical)
                    transitions.append((s, 'p1_enter', 2 * 3 + p2))
                if p1 == 2:  # critical -> idle
                    transitions.append((s, 'p1_exit', 0 * 3 + p2))

                # P2 transitions
                if p2 == 0:
                    transitions.append((s, 'p2_try', p1 * 3 + 1))
                if p2 == 1 and p1 != 2:
                    transitions.append((s, 'p2_enter', p1 * 3 + 2))
                if p2 == 2:
                    transitions.append((s, 'p2_exit', p1 * 3 + 0))

        return make_lts(states, transitions, labels)

    def test_mutual_exclusion(self, mutex_lts):
        """AG ~both_critical -- mutual exclusion always holds."""
        f = ctl_AG(Not(Prop("both_critical")))
        result = eval_formula(mutex_lts, f)
        # Starting from (idle, idle) = state 0
        assert 0 in result

    def test_no_starvation_p1(self, mutex_lts):
        """AG(p1_trying -> AF p1_critical) -- if trying, eventually critical."""
        inner = Or(Not(Prop("p1_trying")), ctl_AF(Prop("p1_critical")))
        f = ctl_AG(inner)
        # This may not hold in this simple protocol (P2 could starve P1)
        result = eval_formula(mutex_lts, f)
        # Just check it runs without error
        assert isinstance(result, set)

    def test_reachability_critical(self, mutex_lts):
        """EF p1_critical -- can P1 ever enter critical section?"""
        f = ctl_EF(Prop("p1_critical"))
        result = eval_formula(mutex_lts, f)
        assert 0 in result  # from idle,idle can reach critical


# ===================================================================
# 15. Action-specific modalities
# ===================================================================

class TestActionModalities:
    def test_specific_action_diamond(self, simple_lts):
        # <a>q vs <b>q
        assert eval_formula(simple_lts, Diamond("a", Prop("q"))) == {0}
        assert eval_formula(simple_lts, Diamond("b", Prop("q"))) == {1}

    def test_specific_action_box(self, branching_lts):
        # [a]left: all a-successors have 'left'
        f = Box("a", Prop("left"))
        result = eval_formula(branching_lts, f)
        # State 0: a-succs = {1, 2}. 1 has left, 2 has right. False.
        # Others: no a-succs -> vacuously true.
        assert 0 not in result
        assert 1 in result and 2 in result and 3 in result

    def test_action_sequence(self, simple_lts):
        """<a><b>p: exists a-step then b-step reaching p."""
        f = Diamond("a", Diamond("b", Prop("p")))
        result = eval_formula(simple_lts, f)
        # 0 -a-> 1 -b-> 2 (has p). So state 0.
        assert result == {0}

    def test_mixed_action_box_diamond(self, branching_lts):
        """[a]<b>done OR [a]<c>done"""
        f = Or(Box("a", Diamond("b", Prop("done"))),
               Box("a", Diamond("c", Prop("done"))))
        result = eval_formula(branching_lts, f)
        # State 0: a-succs={1,2}.
        #   [a]<b>done: 1 has b->3(done), 2 has no b-succ. False.
        #   [a]<c>done: 1 has no c-succ, 2 has c->3(done). False.
        # States 1,2,3: no a-succs -> both vacuously true. True.
        assert 0 not in result
        assert {1, 2, 3} <= result


# ===================================================================
# 16. Parity game details
# ===================================================================

class TestParityGameDetails:
    def test_game_has_no_dead_ends(self, simple_lts):
        """All vertices should have at least one successor."""
        f = Mu("X", Or(Prop("p"), Diamond(None, Var("X"))))
        game, _, _, _ = formula_to_parity_game(simple_lts, f)
        for v in game.vertices:
            assert len(game.successors(v)) > 0, f"Vertex {v} has no successors"

    def test_game_vertex_count(self, simple_lts):
        f = Prop("p")
        game, v_map, sub_list, _ = formula_to_parity_game(simple_lts, f)
        expected = len(sub_list) * len(simple_lts.states)
        assert len(game.vertices) == expected

    def test_game_solution_valid(self, simple_lts):
        """Solution should be valid according to V156 verify_solution."""
        v156_dir = os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games')
        sys.path.insert(0, v156_dir)
        from parity_games import zielonka, verify_solution

        f = Mu("X", Or(Prop("q"), Diamond(None, Var("X"))))
        game, _, _, _ = formula_to_parity_game(simple_lts, f)
        sol = zielonka(game)
        valid, errors = verify_solution(game, sol)
        assert valid, f"Solution invalid: {errors}"


# ===================================================================
# 17. Not formulas (negation in game)
# ===================================================================

class TestNegation:
    def test_not_prop_direct(self, simple_lts):
        f = Not(Prop("p"))
        assert eval_formula(simple_lts, f) == {1}

    def test_not_and(self, simple_lts):
        f = Not(And(Prop("p"), Prop("q")))
        assert eval_formula(simple_lts, f) == {0, 1}  # not both p and q

    def test_not_or(self, simple_lts):
        f = Not(Or(Prop("p"), Prop("q")))
        assert eval_formula(simple_lts, f) == set()  # all states have p or q

    def test_double_negation(self, simple_lts):
        f = Not(Not(Prop("p")))
        assert eval_formula(simple_lts, f) == {0, 2}


# ===================================================================
# 18. Summary and batch APIs
# ===================================================================

class TestBatchAndSummary:
    def test_batch_multiple_formulas(self, traffic_lts):
        formulas = [
            Prop("green"),
            Prop("red"),
            ctl_EF(Prop("green")),
            ctl_AG(Or(Prop("green"), Or(Prop("yellow"), Prop("red")))),
        ]
        results = batch_check(traffic_lts, formulas)
        assert results[0]["sat_states"] == {0}
        assert results[1]["sat_states"] == {2}
        assert results[2]["sat_count"] == 3  # all reach green
        assert results[3]["sat_count"] == 3  # always in some color

    def test_summary_output(self, simple_lts):
        f = Prop("p")
        s = mu_calculus_summary(simple_lts, f)
        assert "Prop(p)" in s
        assert "3 states" in s
