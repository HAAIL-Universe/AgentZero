"""Tests for V208: Strategy Logic (SL)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from strategy_logic import (
    # Formula constructors
    SL, SLOp, sl_atom, sl_true, sl_false, sl_not, sl_and, sl_or, sl_implies,
    exists_strategy, forall_strategy, bind,
    sl_next, sl_globally, sl_finally, sl_until,
    # Analysis
    free_strategy_vars, bound_agents, is_sentence, is_atl_star_fragment,
    sl_subformulas,
    # Strategies
    MemorylessStrategy, BoundedMemoryStrategy, HistoryStrategy,
    StrategyProfile,
    # Core operations
    compute_outcome, check_path, check_sl,
    # Nash equilibrium
    check_nash_equilibrium, find_nash_equilibria,
    # Dominant strategies
    find_dominant_strategy,
    # Strategy sharing
    check_with_shared_strategy,
    # Expressiveness comparison
    compare_sl_vs_atl_fragment,
    # Example games
    make_simple_game, make_coordination_game, make_prisoners_dilemma,
    make_traffic_intersection, make_resource_sharing_game,
)


# ============================================================
# Formula Construction
# ============================================================

class TestFormulaConstruction:
    def test_atom(self):
        p = sl_atom("p")
        assert p.op == SLOp.ATOM
        assert p.name == "p"
        assert str(p) == "p"

    def test_boolean_operators(self):
        p, q = sl_atom("p"), sl_atom("q")
        assert str(sl_not(p)) == "!p"
        assert str(sl_and(p, q)) == "(p & q)"
        assert str(sl_or(p, q)) == "(p | q)"
        assert str(sl_implies(p, q)) == "(p -> q)"

    def test_temporal_operators(self):
        p, q = sl_atom("p"), sl_atom("q")
        assert str(sl_next(p)) == "X p"
        assert str(sl_globally(p)) == "G p"
        assert str(sl_finally(p)) == "F p"
        assert str(sl_until(p, q)) == "(p U q)"

    def test_strategy_quantification(self):
        p = sl_atom("win")
        f = exists_strategy("x", bind("agent1", "x", sl_finally(p)))
        assert f.op == SLOp.EXISTS_STRATEGY
        assert f.strategy_var == "x"
        assert str(f) == "(Ex x. [agent1 bind x] F win)"

    def test_universal_quantification(self):
        p = sl_atom("safe")
        f = forall_strategy("y", bind("agent2", "y", sl_globally(p)))
        assert f.op == SLOp.FORALL_STRATEGY
        assert str(f) == "(Ax y. [agent2 bind y] G safe)"

    def test_nested_quantifiers(self):
        win = sl_atom("win1")
        f = exists_strategy("x",
                forall_strategy("y",
                    bind("agent1", "x",
                        bind("agent2", "y",
                            sl_finally(win)))))
        assert str(f) == "(Ex x. (Ax y. [agent1 bind x] [agent2 bind y] F win1))"

    def test_true_false(self):
        assert str(sl_true()) == "true"
        assert str(sl_false()) == "false"


# ============================================================
# Formula Analysis
# ============================================================

class TestFormulaAnalysis:
    def test_free_vars_atom(self):
        assert free_strategy_vars(sl_atom("p")) == set()

    def test_free_vars_bound(self):
        f = exists_strategy("x", bind("a", "x", sl_atom("p")))
        assert free_strategy_vars(f) == set()

    def test_free_vars_unbound(self):
        f = bind("a", "x", sl_atom("p"))
        assert free_strategy_vars(f) == {"x"}

    def test_free_vars_partial(self):
        f = exists_strategy("x",
                bind("a", "x",
                    bind("b", "y", sl_atom("p"))))
        assert free_strategy_vars(f) == {"y"}

    def test_bound_agents(self):
        f = exists_strategy("x",
                bind("alice", "x",
                    bind("bob", "y", sl_atom("p"))))
        assert bound_agents(f) == {"alice", "bob"}

    def test_is_sentence_true(self):
        f = exists_strategy("x", bind("a", "x", sl_atom("p")))
        assert is_sentence(f) is True

    def test_is_sentence_false(self):
        f = bind("a", "x", sl_atom("p"))
        assert is_sentence(f) is False

    def test_subformulas(self):
        p = sl_atom("p")
        f = sl_not(p)
        subs = sl_subformulas(f)
        assert p in subs
        assert f in subs
        assert len(subs) == 2


class TestATLStarFragment:
    def test_simple_atl_formula(self):
        # exists x. [a bind x] F p -- this is <<{a}>> F p
        f = exists_strategy("x", bind("a", "x", sl_finally(sl_atom("p"))))
        assert is_atl_star_fragment(f) is True

    def test_coalition_atl(self):
        # exists x. exists y. [a bind x] [b bind y] G safe
        f = exists_strategy("x",
                exists_strategy("y",
                    bind("a", "x",
                        bind("b", "y",
                            sl_globally(sl_atom("safe"))))))
        assert is_atl_star_fragment(f) is True

    def test_strategy_sharing_not_atl(self):
        # exists x. [a bind x] [b bind x] F win -- strategy sharing
        f = exists_strategy("x",
                bind("a", "x",
                    bind("b", "x",
                        sl_finally(sl_atom("win")))))
        # Same variable bound to two agents -> not ATL*
        assert is_atl_star_fragment(f) is False

    def test_nested_quantifiers_not_atl(self):
        # exists x. [a bind x] forall y. [b bind y] F win
        f = exists_strategy("x",
                bind("a", "x",
                    forall_strategy("y",
                        bind("b", "y",
                            sl_finally(sl_atom("win"))))))
        assert is_atl_star_fragment(f) is False

    def test_universal_not_atl(self):
        f = forall_strategy("x", bind("a", "x", sl_atom("p")))
        assert is_atl_star_fragment(f) is False


# ============================================================
# Strategy and Profile Tests
# ============================================================

class TestStrategies:
    def test_memoryless_strategy(self):
        s = MemorylessStrategy("a", {"s0": "go", "s1": "stay"})
        assert s.choose("s0") == "go"
        assert s.choose("s1") == "stay"
        assert s.choose("s99") == ""  # unknown state

    def test_bounded_memory_strategy(self):
        s = BoundedMemoryStrategy("a", memory_size=2)
        s.transition = {
            ("s0", 0): ("a", 1),
            ("s0", 1): ("b", 0),
        }
        action, mem = s.choose_with_memory("s0", 0)
        assert action == "a"
        assert mem == 1
        action2, mem2 = s.choose_with_memory("s0", 1)
        assert action2 == "b"
        assert mem2 == 0

    def test_history_strategy(self):
        s = HistoryStrategy("a", default_action="wait")
        s.table[("s0",)] = "go"
        s.table[("s0", "s1")] = "stop"
        assert s.choose("s0") == "go"
        assert s.choose("s1", ["s0"]) == "stop"
        assert s.choose("s2", ["s3"]) == "wait"

    def test_strategy_profile(self):
        s1 = MemorylessStrategy("a", {"s0": "go"})
        s2 = MemorylessStrategy("b", {"s0": "wait"})
        p = StrategyProfile()
        p = p.assign("a", s1)
        p = p.assign("b", s2)
        assert p.get("a") is s1
        assert p.get("b") is s2
        assert p.get("c") is None
        assert p.is_complete({"a", "b"})
        assert not p.is_complete({"a", "b", "c"})


# ============================================================
# Outcome Computation
# ============================================================

class TestOutcome:
    def test_simple_outcome(self):
        cgs = make_simple_game()
        s1 = MemorylessStrategy("agent1", {"s0": "a", "s1": "stay", "s2": "stay", "s3": "stay"})
        s2 = MemorylessStrategy("agent2", {"s0": "c", "s1": "stay", "s2": "stay", "s3": "stay"})
        profile = StrategyProfile({"agent1": s1, "agent2": s2})
        path = compute_outcome(cgs, profile, "s0", max_steps=5)
        assert path[0] == "s0"
        assert path[1] == "s1"  # (a, c) -> s1

    def test_coordination_outcome(self):
        cgs = make_coordination_game()
        sa = MemorylessStrategy("alice", {"s0": "L", "s1": "stay", "s2": "stay"})
        sb = MemorylessStrategy("bob", {"s0": "L", "s1": "stay", "s2": "stay"})
        profile = StrategyProfile({"alice": sa, "bob": sb})
        path = compute_outcome(cgs, profile, "s0", max_steps=3)
        assert "s1" in path  # both L -> coordinated

    def test_mismatched_outcome(self):
        cgs = make_coordination_game()
        sa = MemorylessStrategy("alice", {"s0": "L", "s1": "stay", "s2": "stay"})
        sb = MemorylessStrategy("bob", {"s0": "R", "s1": "stay", "s2": "stay"})
        profile = StrategyProfile({"alice": sa, "bob": sb})
        path = compute_outcome(cgs, profile, "s0", max_steps=3)
        assert "s2" in path  # L vs R -> mismatched


# ============================================================
# Path Checking
# ============================================================

class TestPathChecking:
    def test_atom_true(self):
        cgs = make_simple_game()
        assert check_path(cgs, ["s1"], sl_atom("win1")) is True

    def test_atom_false(self):
        cgs = make_simple_game()
        assert check_path(cgs, ["s1"], sl_atom("win2")) is False

    def test_next(self):
        cgs = make_simple_game()
        assert check_path(cgs, ["s0", "s1"], sl_next(sl_atom("win1"))) is True
        assert check_path(cgs, ["s0", "s2"], sl_next(sl_atom("win1"))) is False

    def test_globally(self):
        cgs = make_simple_game()
        assert check_path(cgs, ["s1", "s1", "s1"],
                          sl_globally(sl_atom("win1"))) is True
        assert check_path(cgs, ["s1", "s2", "s1"],
                          sl_globally(sl_atom("win1"))) is False

    def test_finally(self):
        cgs = make_simple_game()
        assert check_path(cgs, ["s0", "s3", "s1"],
                          sl_finally(sl_atom("win1"))) is True
        assert check_path(cgs, ["s0", "s3", "s2"],
                          sl_finally(sl_atom("win1"))) is False

    def test_until(self):
        cgs = make_simple_game()
        path = ["s0", "s0", "s1"]
        assert check_path(cgs, path,
                          sl_until(sl_atom("start"), sl_atom("win1"))) is True

    def test_boolean_on_path(self):
        cgs = make_simple_game()
        assert check_path(cgs, ["s1"],
                          sl_and(sl_atom("win1"), sl_not(sl_atom("win2")))) is True

    def test_implies_on_path(self):
        cgs = make_simple_game()
        assert check_path(cgs, ["s0"],
                          sl_implies(sl_atom("start"), sl_true())) is True


# ============================================================
# SL Model Checking
# ============================================================

class TestSLModelChecking:
    def test_atom_check(self):
        cgs = make_simple_game()
        assert check_sl(cgs, sl_atom("start"), "s0") is True
        assert check_sl(cgs, sl_atom("win1"), "s0") is False

    def test_existential_simple(self):
        """Agent1 can ensure reaching win1: exists x. [agent1 bind x] F win1"""
        cgs = make_simple_game()
        phi = exists_strategy("x",
                  bind("agent1", "x", sl_finally(sl_atom("win1"))))
        # Agent1 can choose 'a', then if agent2 picks 'c' -> s1 (win1)
        # But agent2 might pick 'd' -> s3 (draw)
        # Since this is existential over agent1 only, agent2 picks arbitrary
        # The default for unbound agents is first action, so agent2 picks 'c'
        result = check_sl(cgs, phi, "s0")
        assert result is True

    def test_forall_opponent(self):
        """Agent1 wins against ALL agent2 strategies?
        exists x. forall y. [agent1 bind x] [agent2 bind y] F win1"""
        cgs = make_simple_game()
        phi = exists_strategy("x",
                  forall_strategy("y",
                      bind("agent1", "x",
                          bind("agent2", "y",
                              sl_finally(sl_atom("win1"))))))
        # No -- agent2 can thwart agent1 (if agent1 picks a, agent2 picks d -> draw)
        result = check_sl(cgs, phi, "s0")
        assert result is False

    def test_existential_both_agents(self):
        """Both agents cooperate for win1:
        exists x. exists y. [agent1 bind x] [agent2 bind y] F win1"""
        cgs = make_simple_game()
        phi = exists_strategy("x",
                  exists_strategy("y",
                      bind("agent1", "x",
                          bind("agent2", "y",
                              sl_finally(sl_atom("win1"))))))
        result = check_sl(cgs, phi, "s0")
        assert result is True  # (a, c) -> s1

    def test_universal_both_agents(self):
        """All strategy combos lead to win1?"""
        cgs = make_simple_game()
        phi = forall_strategy("x",
                  forall_strategy("y",
                      bind("agent1", "x",
                          bind("agent2", "y",
                              sl_finally(sl_atom("win1"))))))
        assert check_sl(cgs, phi, "s0") is False

    def test_boolean_combination(self):
        cgs = make_simple_game()
        phi = sl_and(sl_atom("start"), sl_not(sl_atom("win1")))
        assert check_sl(cgs, phi, "s0") is True

    def test_true_false(self):
        cgs = make_simple_game()
        assert check_sl(cgs, sl_true(), "s0") is True
        assert check_sl(cgs, sl_false(), "s0") is False

    def test_implication(self):
        cgs = make_simple_game()
        phi = sl_implies(sl_atom("start"), sl_not(sl_atom("win1")))
        assert check_sl(cgs, phi, "s0") is True


# ============================================================
# Coordination Game
# ============================================================

class TestCoordinationGame:
    def test_coordination_possible(self):
        """exists x, y. [alice bind x] [bob bind y] F coordinated"""
        cgs = make_coordination_game()
        phi = exists_strategy("x",
                  exists_strategy("y",
                      bind("alice", "x",
                          bind("bob", "y",
                              sl_finally(sl_atom("coordinated"))))))
        assert check_sl(cgs, phi, "s0") is True

    def test_no_dominant_coordination(self):
        """No agent can unilaterally ensure coordination:
        exists x. forall y. [alice bind x] [bob bind y] F coordinated"""
        cgs = make_coordination_game()
        phi = exists_strategy("x",
                  forall_strategy("y",
                      bind("alice", "x",
                          bind("bob", "y",
                              sl_finally(sl_atom("coordinated"))))))
        assert check_sl(cgs, phi, "s0") is False

    def test_shared_strategy_coordination(self):
        """With strategy sharing (SL-only feature), coordination is guaranteed:
        exists x. [alice bind x] [bob bind x] F coordinated"""
        cgs = make_coordination_game()
        result = check_with_shared_strategy(
            cgs, ["alice", "bob"], sl_finally(sl_atom("coordinated")), "s0")
        assert result is not None  # shared strategy exists


# ============================================================
# Nash Equilibrium
# ============================================================

class TestNashEquilibrium:
    def test_simple_game_nash(self):
        cgs = make_simple_game()
        # Profile: agent1->a, agent2->c => s1 (win1)
        s1 = MemorylessStrategy("agent1",
                                {"s0": "a", "s1": "stay", "s2": "stay", "s3": "stay"})
        s2 = MemorylessStrategy("agent2",
                                {"s0": "c", "s1": "stay", "s2": "stay", "s3": "stay"})
        profile = StrategyProfile({"agent1": s1, "agent2": s2})
        objectives = {
            "agent1": sl_finally(sl_atom("win1")),
            "agent2": sl_finally(sl_atom("win2")),
        }
        result = check_nash_equilibrium(cgs, profile, objectives, "s0")
        # agent1 is satisfied (win1), but can agent2 deviate to get win2?
        # If agent2 switches to d: (a, d) -> s3 (draw), not win2
        # So agent2 cannot improve -> this IS a Nash equilibrium
        assert result["is_nash"] is True

    def test_simple_game_not_nash(self):
        cgs = make_simple_game()
        # Profile: agent1->b, agent2->c => s3 (draw)
        s1 = MemorylessStrategy("agent1",
                                {"s0": "b", "s1": "stay", "s2": "stay", "s3": "stay"})
        s2 = MemorylessStrategy("agent2",
                                {"s0": "c", "s1": "stay", "s2": "stay", "s3": "stay"})
        profile = StrategyProfile({"agent1": s1, "agent2": s2})
        objectives = {
            "agent1": sl_finally(sl_atom("win1")),
            "agent2": sl_finally(sl_atom("win2")),
        }
        result = check_nash_equilibrium(cgs, profile, objectives, "s0")
        # agent1 gets draw. If agent1 switches to a: (a, c) -> s1 (win1). Profitable!
        assert result["is_nash"] is False
        assert "agent1" in result["deviations"]

    def test_find_nash_equilibria(self):
        cgs = make_simple_game()
        objectives = {
            "agent1": sl_finally(sl_atom("win1")),
            "agent2": sl_finally(sl_atom("win2")),
        }
        equilibria = find_nash_equilibria(cgs, objectives, "s0")
        assert len(equilibria) >= 1

    def test_coordination_nash(self):
        """In coordination game, both (L,L) and (R,R) are Nash equilibria."""
        cgs = make_coordination_game()
        objectives = {
            "alice": sl_finally(sl_atom("win")),
            "bob": sl_finally(sl_atom("win")),
        }
        equilibria = find_nash_equilibria(cgs, objectives, "s0")
        # Both players want "win" -- (L,L) and (R,R) both satisfy
        assert len(equilibria) >= 2


# ============================================================
# Dominant Strategy
# ============================================================

class TestDominantStrategy:
    def test_no_dominant_in_simple_game(self):
        """In simple game, agent1 has no dominant strategy for win1."""
        cgs = make_simple_game()
        result = find_dominant_strategy(
            cgs, "agent1", sl_finally(sl_atom("win1")),
            ["agent2"], "s0")
        assert result is None

    def test_no_dominant_in_coordination(self):
        """No dominant strategy in coordination game."""
        cgs = make_coordination_game()
        result = find_dominant_strategy(
            cgs, "alice", sl_finally(sl_atom("win")),
            ["bob"], "s0")
        assert result is None


# ============================================================
# Traffic Intersection Game
# ============================================================

class TestTrafficGame:
    def test_game_construction(self):
        cgs = make_traffic_intersection()
        assert len(cgs.states) == 5
        assert len(cgs.agents) == 2

    def test_avoid_crash_exists(self):
        """Each car can individually avoid crash (by waiting)."""
        cgs = make_traffic_intersection()
        phi = exists_strategy("x",
                  bind("car1", "x",
                      sl_globally(sl_not(sl_atom("crash")))))
        # car1 waits -> crash avoided regardless of car2
        assert check_sl(cgs, phi, "approach") is True

    def test_car1_can_go_through(self):
        """car1 can ensure going through if car2 cooperates."""
        cgs = make_traffic_intersection()
        phi = exists_strategy("x",
                  exists_strategy("y",
                      bind("car1", "x",
                          bind("car2", "y",
                              sl_finally(sl_atom("car1_ok"))))))
        assert check_sl(cgs, phi, "approach") is True

    def test_no_dominant_go(self):
        """No car can dominate to go through safely."""
        cgs = make_traffic_intersection()
        result = find_dominant_strategy(
            cgs, "car1",
            sl_and(sl_finally(sl_atom("car1_ok")),
                   sl_globally(sl_not(sl_atom("crash")))),
            ["car2"], "approach")
        assert result is None


# ============================================================
# Strategy Sharing (SL-only)
# ============================================================

class TestStrategySharing:
    def test_shared_strategy_coordination(self):
        """Strategy sharing solves coordination."""
        cgs = make_coordination_game()
        result = check_with_shared_strategy(
            cgs, ["alice", "bob"],
            sl_finally(sl_atom("win")), "s0")
        assert result is not None

    def test_shared_strategy_traffic(self):
        """Strategy sharing in traffic: can both go? No (crash)."""
        cgs = make_traffic_intersection()
        result = check_with_shared_strategy(
            cgs, ["car1", "car2"],
            sl_and(sl_finally(sl_atom("car1_ok")),
                   sl_finally(sl_atom("car2_ok"))),
            "approach")
        # With shared strategy, both cars take same action at same state
        # Both go = crash, both wait = neither goes through
        assert result is None


# ============================================================
# ATL Translation
# ============================================================

class TestATLFragmentDetection:
    def test_atl_fragment_detected(self):
        """ATL* fragment formulas are correctly identified."""
        cgs = make_simple_game()
        phi = exists_strategy("x",
                  exists_strategy("y",
                      bind("agent1", "x",
                          bind("agent2", "y",
                              sl_finally(sl_atom("win1"))))))
        result = compare_sl_vs_atl_fragment(cgs, phi, "s0")
        assert result["is_atl_star_fragment"] is True
        assert result["sl_only"] is False
        assert result["sl_result"] is True

    def test_sl_only_formula(self):
        """SL formulas with strategy sharing are not in ATL*."""
        cgs = make_coordination_game()
        phi = exists_strategy("x",
                  bind("alice", "x",
                      bind("bob", "x",
                          sl_finally(sl_atom("win")))))
        result = compare_sl_vs_atl_fragment(cgs, phi, "s0")
        assert result["sl_only"] is True

    def test_nested_quant_not_atl(self):
        """Nested quantifiers (exists-forall) are not in ATL*."""
        cgs = make_simple_game()
        phi = exists_strategy("x",
                  bind("agent1", "x",
                      forall_strategy("y",
                          bind("agent2", "y",
                              sl_finally(sl_atom("win1"))))))
        result = compare_sl_vs_atl_fragment(cgs, phi, "s0")
        assert result["sl_only"] is True


# ============================================================
# Prisoners Dilemma
# ============================================================

class TestPrisonersDilemma:
    def test_game_construction(self):
        cgs = make_prisoners_dilemma()
        assert len(cgs.agents) == 2
        assert "p1" in cgs.agents
        assert "p2" in cgs.agents

    def test_mutual_cooperate_reachable(self):
        """Both players can cooperate."""
        cgs = make_prisoners_dilemma()
        phi = exists_strategy("x",
                  exists_strategy("y",
                      bind("p1", "x",
                          bind("p2", "y",
                              sl_finally(sl_atom("mutual_cooperate"))))))
        assert check_sl(cgs, phi, "s0") is True

    def test_defect_is_dominant(self):
        """Defect is dominant: no matter what opponent does, defecting avoids sucker."""
        cgs = make_prisoners_dilemma()
        # p1 wants to avoid being the sucker
        result = find_dominant_strategy(
            cgs, "p1",
            sl_globally(sl_not(sl_atom("p1_sucker"))),
            ["p2"], "s0")
        # Defect avoids sucker: if p2 cooperates -> DC (temptation), if p2 defects -> DD
        assert result is not None
        assert result.choices["s0"] == "D"

    def test_mutual_defect_is_nash(self):
        """(D, D) is a Nash equilibrium."""
        cgs = make_prisoners_dilemma()
        s1 = MemorylessStrategy("p1", {s: "D" for s in ["s0", "s_CC", "s_CD", "s_DC", "s_DD"]})
        s2 = MemorylessStrategy("p2", {s: "D" for s in ["s0", "s_CC", "s_CD", "s_DC", "s_DD"]})
        profile = StrategyProfile({"p1": s1, "p2": s2})
        # Each player's objective: avoid being the sucker
        objectives = {
            "p1": sl_globally(sl_not(sl_atom("p1_sucker"))),
            "p2": sl_globally(sl_not(sl_atom("p2_sucker"))),
        }
        result = check_nash_equilibrium(cgs, profile, objectives, "s0")
        assert result["is_nash"] is True


# ============================================================
# Resource Sharing Game
# ============================================================

class TestResourceSharing:
    def test_game_construction(self):
        cgs = make_resource_sharing_game()
        assert len(cgs.agents) == 3
        assert "idle" in cgs.states

    def test_agent_can_get_resource(self):
        """Agent1 can get resource A."""
        cgs = make_resource_sharing_game()
        phi = exists_strategy("x",
                  bind("ag1", "x",
                      sl_finally(sl_atom("ag1_got_A"))))
        assert check_sl(cgs, phi, "idle") is True

    def test_two_agents_can_share(self):
        """Two agents can get different resources."""
        cgs = make_resource_sharing_game()
        phi = exists_strategy("x",
                  exists_strategy("y",
                      bind("ag1", "x",
                          bind("ag2", "y",
                              sl_finally(sl_and(
                                  sl_atom("ag1_got_A"),
                                  sl_atom("ag2_got_B")))))))
        assert check_sl(cgs, phi, "idle") is True

    def test_three_agents_cant_all_get(self):
        """Three agents cannot all get resources (only 2 resources)."""
        cgs = make_resource_sharing_game()
        phi = exists_strategy("x",
                  exists_strategy("y",
                      exists_strategy("z",
                          bind("ag1", "x",
                              bind("ag2", "y",
                                  bind("ag3", "z",
                                      sl_finally(sl_and(
                                          sl_atom("ag1_got_A"),
                                          sl_and(sl_atom("ag2_got_B"),
                                                 sl_atom("ag3_got_A"))))))))))
        # ag1 gets A and ag3 gets A is impossible (only 1 resource A)
        assert check_sl(cgs, phi, "idle") is False


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_vacuous_quantification(self):
        """exists x. p (variable not used)"""
        cgs = make_simple_game()
        phi = exists_strategy("x", sl_atom("start"))
        assert check_sl(cgs, phi, "s0") is True

    def test_nested_boolean(self):
        cgs = make_simple_game()
        phi = sl_or(
            sl_and(sl_atom("start"), sl_true()),
            sl_false()
        )
        assert check_sl(cgs, phi, "s0") is True

    def test_empty_path_globally(self):
        """Globally on empty remainder is vacuously true."""
        cgs = make_simple_game()
        assert check_path(cgs, ["s0"], sl_next(sl_globally(sl_atom("win1")))) is True

    def test_empty_path_finally(self):
        """Finally on empty remainder is false."""
        cgs = make_simple_game()
        assert check_path(cgs, ["s0"], sl_next(sl_finally(sl_atom("win1")))) is False

    def test_until_never_satisfied(self):
        cgs = make_simple_game()
        # s0 has "start", not "win1" or "win2"
        assert check_path(cgs, ["s0", "s0", "s0"],
                          sl_until(sl_atom("win1"), sl_atom("win2"))) is False

    def test_double_negation(self):
        cgs = make_simple_game()
        assert check_sl(cgs, sl_not(sl_not(sl_atom("start"))), "s0") is True

    def test_profile_immutability(self):
        """Assigning to a profile returns a new profile."""
        p1 = StrategyProfile()
        s = MemorylessStrategy("a", {"s0": "go"})
        p2 = p1.assign("a", s)
        assert p1.get("a") is None
        assert p2.get("a") is s
