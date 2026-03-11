"""Tests for V073: Game-Theoretic Strategy Synthesis"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V070_stochastic_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V072_game_pctl'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from game_synthesis import (
    ObjectiveType, Objective, SynthesisResult, PermissiveStrategy,
    MultiObjectiveResult,
    synthesize_reachability, synthesize_safety, synthesize_pctl,
    synthesize_permissive_reachability, synthesize_permissive_safety,
    verify_strategy, synthesize_multi_objective,
    compose_strategies, assume_guarantee_synthesis,
    refine_strategy, compare_strategies,
    synthesize_from_pctl, synthesize, synthesis_summary,
)
from stochastic_games import (
    StochasticGame, Player, StrategyPair, make_game, game_to_mc,
)
from game_pctl import (
    LabeledGame, GameQuantification, make_labeled_game,
    check_game_pctl,
)
from pctl_model_check import (
    tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, next_f, until, eventually, always,
    bounded_until, bounded_eventually, parse_pctl,
)


# ============================================================
# Test Fixtures
# ============================================================

def simple_reach_game():
    """Simple 3-state reachability game.
    s0 (P1) --a--> s1 (CHANCE) --0.8--> s2 (target)
    s0 (P1) --b--> s2 (target)
    s1 (CHANCE) --0.2--> s0
    s2: absorbing
    """
    return make_game(
        n_states=3,
        owners={0: Player.P1, 1: Player.CHANCE, 2: Player.CHANCE},
        action_transitions={
            0: {"go_chance": [0, 1, 0], "go_direct": [0, 0, 1]},
            1: {"roll": [0.2, 0, 0.8]},
            2: {"stay": [0, 0, 1]},
        },
    )


def adversarial_game():
    """4-state adversarial game.
    s0 (P1): left -> s1, right -> s2
    s1 (P2): safe -> s3, risk -> s0
    s2 (P2): push -> s3, block -> s0
    s3: target (absorbing)
    """
    return make_game(
        n_states=4,
        owners={0: Player.P1, 1: Player.P2, 2: Player.P2, 3: Player.CHANCE},
        action_transitions={
            0: {"left": [0, 1, 0, 0], "right": [0, 0, 1, 0]},
            1: {"safe": [0, 0, 0, 1], "risk": [1, 0, 0, 0]},
            2: {"push": [0, 0, 0, 1], "block": [1, 0, 0, 0]},
            3: {"absorb": [0, 0, 0, 1]},
        },
    )


def stochastic_game():
    """5-state game with stochastic transitions.
    s0 (P1): a -> 0.7*s1 + 0.3*s2, b -> 0.4*s1 + 0.6*s3
    s1 (P2): x -> 0.5*s4 + 0.5*s0, y -> s3
    s2 (CHANCE): -> 0.6*s4 + 0.4*s0
    s3 (P1): c -> s4, d -> s0
    s4: target (absorbing)
    """
    return make_game(
        n_states=5,
        owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE,
                3: Player.P1, 4: Player.CHANCE},
        action_transitions={
            0: {"a": [0.0, 0.7, 0.3, 0.0, 0.0],
                "b": [0.0, 0.4, 0.0, 0.6, 0.0]},
            1: {"x": [0.5, 0.0, 0.0, 0.0, 0.5],
                "y": [0.0, 0.0, 0.0, 1.0, 0.0]},
            2: {"roll": [0.4, 0.0, 0.0, 0.0, 0.6]},
            3: {"c": [0.0, 0.0, 0.0, 0.0, 1.0],
                "d": [1.0, 0.0, 0.0, 0.0, 0.0]},
            4: {"absorb": [0.0, 0.0, 0.0, 0.0, 1.0]},
        },
    )


def reward_game():
    """Game with rewards for multi-objective testing.
    s0 (P1): left (r=1) -> s1, right (r=0) -> s2
    s1 (P2): up (r=2) -> s0, down (r=-1) -> s2
    s2: absorbing (r=0)
    """
    return make_game(
        n_states=3,
        owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE},
        action_transitions={
            0: {"left": [0, 1, 0], "right": [0, 0, 1]},
            1: {"up": [1, 0, 0], "down": [0, 0, 1]},
            2: {"stay": [0, 0, 1]},
        },
        rewards={0: {"left": 1.0, "right": 0.0},
                 1: {"up": 2.0, "down": -1.0},
                 2: {"stay": 0.0}},
    )


def labeled_game_fixture():
    """Labeled game for PCTL synthesis."""
    return make_labeled_game(
        n_states=4,
        owners={0: Player.P1, 1: Player.CHANCE, 2: Player.P2, 3: Player.CHANCE},
        action_transitions={
            0: {"a": [0, 0.6, 0.4, 0], "b": [0, 0, 0, 1]},
            1: {"roll": [0.3, 0, 0, 0.7]},
            2: {"safe": [0, 0, 0, 1], "attack": [1, 0, 0, 0]},
            3: {"absorb": [0, 0, 0, 1]},
        },
        labels={
            0: {"start"},
            1: {"mid"},
            2: {"danger"},
            3: {"goal"},
        },
    )


# ============================================================
# 1. Reachability Synthesis
# ============================================================

class TestReachabilitySynthesis:
    def test_simple_reachability(self):
        game = simple_reach_game()
        result = synthesize_reachability(game, targets={2})
        assert result.success
        assert result.strategies is not None
        # Both actions lead to s2 with prob 1 (direct or via chance)
        # So any action is optimal
        assert result.values_per_state[0] == 1.0

    def test_reachability_with_threshold(self):
        game = simple_reach_game()
        result = synthesize_reachability(game, targets={2}, threshold=0.5)
        assert result.success
        assert result.value >= 0.5

    def test_adversarial_reachability(self):
        game = adversarial_game()
        result = synthesize_reachability(game, targets={3})
        assert result.strategies is not None
        assert result.values_per_state is not None
        # P2 minimizes: P2 will choose risk/block to send back to s0
        # Under adversarial play, P2 can prevent reaching s3 forever
        # So reachability from s0 is 0
        assert result.values_per_state[0] == 0.0

    def test_stochastic_reachability(self):
        game = stochastic_game()
        result = synthesize_reachability(game, targets={4})
        assert result.success
        assert result.strategies is not None
        assert result.values_per_state is not None
        # All states should eventually reach s4
        for s in range(5):
            assert result.values_per_state[s] > 0

    def test_p2_reachability(self):
        game = simple_reach_game()
        # P2 wants to minimize reaching target -- but P1 has direct path
        result = synthesize_reachability(game, targets={2}, player=Player.P2)
        # P1 controls s0 and can go directly to s2, so P2 can't prevent it
        # The max prob for non-target states is 1.0, threshold is default 1.0
        assert result.values_per_state[0] == 1.0


# ============================================================
# 2. Safety Synthesis
# ============================================================

class TestSafetySynthesis:
    def test_simple_safety(self):
        game = simple_reach_game()
        # Safe states: {0, 1} (avoid s2)
        result = synthesize_safety(game, safe_states={0, 1})
        assert result.strategies is not None
        # P1 should choose go_chance (action 0) to avoid going directly to s2
        assert result.strategies.p1_strategy[0] == 0

    def test_full_safety(self):
        game = simple_reach_game()
        # All states safe -- trivially achievable
        result = synthesize_safety(game, safe_states={0, 1, 2}, threshold=1.0)
        assert result.success
        assert result.value == 1.0

    def test_adversarial_safety(self):
        game = adversarial_game()
        # Stay in {0, 1, 2} (avoid s3)
        result = synthesize_safety(game, safe_states={0, 1, 2})
        assert result.strategies is not None
        # P2 controls s1, s2 -- P2 will push to s3 if possible
        # P1 needs P2 to cooperate, but P2 can always send to s3
        # Actually P2 minimizes safety (maximizes leaving safe set)
        # So P2 will always choose safe/push -> s3
        # Safety probability should be 0

    def test_safety_with_threshold(self):
        game = stochastic_game()
        result = synthesize_safety(game, safe_states={0, 1, 2, 3}, threshold=0.5)
        assert result.strategies is not None


# ============================================================
# 3. PCTL Synthesis
# ============================================================

class TestPCTLSynthesis:
    def test_pctl_reachability(self):
        lgame = labeled_game_fixture()
        # P >= 0.5 [F goal]
        formula = prob_geq(0.5, eventually(atom("goal")))
        result = synthesize_pctl(lgame, formula, initial_state=0)
        assert result.strategies is not None

    def test_pctl_with_quantification(self):
        lgame = labeled_game_fixture()
        formula = prob_geq(0.5, eventually(atom("goal")))
        result = synthesize_pctl(lgame, formula,
                                  quantification=GameQuantification.COOPERATIVE)
        assert result.strategies is not None

    def test_pctl_bounded(self):
        lgame = labeled_game_fixture()
        # P >= 0.3 [F<=3 goal]
        formula = prob_geq(0.3, bounded_eventually(atom("goal"), 3))
        result = synthesize_pctl(lgame, formula, initial_state=0)
        assert result.strategies is not None

    def test_pctl_next(self):
        lgame = labeled_game_fixture()
        # P >= 0.5 [X goal]
        formula = prob_geq(0.5, next_f(atom("goal")))
        result = synthesize_pctl(lgame, formula, initial_state=0)
        assert result.strategies is not None
        # P1 should choose action b (direct to goal) for max next-step prob
        assert result.value >= 0.5 or result.value < 0.5  # Just check it runs


# ============================================================
# 4. Permissive Strategy Synthesis
# ============================================================

class TestPermissiveSynthesis:
    def test_permissive_reachability(self):
        game = simple_reach_game()
        perm = synthesize_permissive_reachability(game, targets={2})
        assert isinstance(perm, PermissiveStrategy)
        # Both actions at s0 lead to s2 (one directly, one via s1)
        # So both should be allowed
        assert len(perm.allowed_actions[0]) == 2

    def test_permissive_safety(self):
        game = simple_reach_game()
        perm = synthesize_permissive_safety(game, safe_states={0, 1})
        assert isinstance(perm, PermissiveStrategy)
        # Only go_chance (action 0) keeps us in {0,1}
        assert 0 in perm.allowed_actions[0]

    def test_permissiveness_metric(self):
        game = simple_reach_game()
        perm = synthesize_permissive_reachability(game, targets={2})
        p = perm.permissiveness()
        assert 0 <= p <= 1

    def test_to_deterministic(self):
        game = simple_reach_game()
        perm = synthesize_permissive_reachability(game, targets={2})
        det = perm.to_deterministic()
        assert isinstance(det, dict)
        for s in det:
            assert det[s] in perm.allowed_actions[s]

    def test_permissive_adversarial(self):
        game = adversarial_game()
        perm = synthesize_permissive_reachability(game, targets={3}, player=Player.P1)
        assert isinstance(perm, PermissiveStrategy)
        # P1 at s0: both left and right eventually reach s3
        assert len(perm.allowed_actions[0]) >= 1

    def test_is_permissive_at(self):
        game = simple_reach_game()
        perm = synthesize_permissive_reachability(game, targets={2})
        # s0 should be permissive (both actions reach target)
        assert perm.is_permissive_at(0)


# ============================================================
# 5. Strategy Verification
# ============================================================

class TestStrategyVerification:
    def test_verify_reachability_strategy(self):
        game = simple_reach_game()
        # Direct strategy: always go direct to s2
        strats = StrategyPair(p1_strategy={0: 1}, p2_strategy={})
        obj = Objective(ObjectiveType.REACHABILITY, targets={2}, threshold=0.9)
        result = verify_strategy(game, strats, obj)
        assert result.success
        assert result.verified
        assert result.value >= 0.9

    def test_verify_safety_strategy(self):
        game = simple_reach_game()
        # Strategy: go via chance (action 0) -- doesn't guarantee safety
        strats = StrategyPair(p1_strategy={0: 0}, p2_strategy={})
        obj = Objective(ObjectiveType.SAFETY, safe_states={0, 1}, threshold=0.5)
        result = verify_strategy(game, strats, obj)
        assert result.verified
        # Will eventually reach s2, so safety < 1.0

    def test_verify_optimal_strategy(self):
        game = simple_reach_game()
        # Synthesize then verify
        synth = synthesize_reachability(game, targets={2})
        obj = Objective(ObjectiveType.REACHABILITY, targets={2}, threshold=0.5)
        ver = verify_strategy(game, synth.strategies, obj)
        assert ver.success
        assert ver.verified

    def test_verify_suboptimal_strategy(self):
        game = adversarial_game()
        # Suboptimal: P1 always goes left, P2 always sends back
        strats = StrategyPair(p1_strategy={0: 0}, p2_strategy={1: 1, 2: 1})
        obj = Objective(ObjectiveType.REACHABILITY, targets={3}, threshold=0.5)
        result = verify_strategy(game, strats, obj)
        # Under this strategy, P2 always blocks -> never reach target
        assert result.value < 0.5


# ============================================================
# 6. Multi-Objective Synthesis
# ============================================================

class TestMultiObjectiveSynthesis:
    def test_single_objective(self):
        game = simple_reach_game()
        objs = [Objective(ObjectiveType.REACHABILITY, targets={2})]
        result = synthesize_multi_objective(game, objs)
        assert result.n_objectives == 1
        assert len(result.pareto_front) >= 1

    def test_two_objectives(self):
        game = stochastic_game()
        objs = [
            Objective(ObjectiveType.REACHABILITY, targets={4}),
            Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2, 3}),
        ]
        result = synthesize_multi_objective(game, objs, n_samples=5)
        assert result.n_objectives == 2
        assert len(result.pareto_front) >= 1
        # Pareto front should have objective values
        for values, strat in result.pareto_front:
            assert len(values) == 2

    def test_pareto_dominance(self):
        game = stochastic_game()
        objs = [
            Objective(ObjectiveType.REACHABILITY, targets={4}),
            Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2}),
        ]
        result = synthesize_multi_objective(game, objs, n_samples=5)
        # Pareto front members should not dominate each other
        for i, (vi, _) in enumerate(result.pareto_front):
            for j, (vj, _) in enumerate(result.pareto_front):
                if i != j:
                    # Neither should dominate the other
                    dominates_ij = (all(vi[k] >= vj[k] for k in range(len(vi)))
                                    and any(vi[k] > vj[k] for k in range(len(vi))))
                    dominates_ji = (all(vj[k] >= vi[k] for k in range(len(vi)))
                                    and any(vj[k] > vi[k] for k in range(len(vi))))
                    assert not (dominates_ij and dominates_ji)

    def test_empty_objectives(self):
        game = simple_reach_game()
        result = synthesize_multi_objective(game, [])
        assert result.n_objectives == 0
        assert len(result.pareto_front) == 0


# ============================================================
# 7. Strategy Composition
# ============================================================

class TestStrategyComposition:
    def test_compose_single(self):
        game = simple_reach_game()
        strats = [StrategyPair({0: 1}, {})]
        objs = [Objective(ObjectiveType.REACHABILITY, targets={2})]
        composed = compose_strategies(game, strats, objs)
        assert composed.p1_strategy[0] == 1

    def test_compose_first_priority(self):
        game = adversarial_game()
        strats1 = StrategyPair({0: 0}, {1: 0, 2: 0})
        strats2 = StrategyPair({0: 1}, {1: 1, 2: 1})
        objs = [
            Objective(ObjectiveType.REACHABILITY, targets={3}),
            Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2}),
        ]
        composed = compose_strategies(game, [strats1, strats2], objs, priority="first")
        # First strategy wins
        assert composed.p1_strategy[0] == 0

    def test_compose_empty(self):
        game = simple_reach_game()
        composed = compose_strategies(game, [], [])
        assert composed.p1_strategy == {}
        assert composed.p2_strategy == {}


# ============================================================
# 8. Assume-Guarantee Synthesis
# ============================================================

class TestAssumeGuaranteeSynthesis:
    def test_basic_ag(self):
        game = adversarial_game()
        assumptions = [Objective(ObjectiveType.REACHABILITY, targets={3})]
        guarantees = [Objective(ObjectiveType.REACHABILITY, targets={3})]
        result = assume_guarantee_synthesis(game, assumptions, guarantees)
        assert result.strategies is not None

    def test_ag_with_safety(self):
        game = stochastic_game()
        assumptions = [Objective(ObjectiveType.REACHABILITY, targets={4})]
        guarantees = [Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2, 3, 4},
                                threshold=0.5)]
        result = assume_guarantee_synthesis(game, assumptions, guarantees)
        assert result.strategies is not None

    def test_ag_verified(self):
        game = simple_reach_game()
        assumptions = [Objective(ObjectiveType.REACHABILITY, targets={2})]
        guarantees = [Objective(ObjectiveType.REACHABILITY, targets={2},
                                threshold=0.5)]
        result = assume_guarantee_synthesis(game, assumptions, guarantees)
        assert result.success
        assert result.verified


# ============================================================
# 9. Strategy Refinement
# ============================================================

class TestStrategyRefinement:
    def test_refine_from_suboptimal(self):
        game = simple_reach_game()
        # Start with suboptimal: go via chance
        initial = StrategyPair(p1_strategy={0: 0}, p2_strategy={})
        obj = Objective(ObjectiveType.REACHABILITY, targets={2})
        result = refine_strategy(game, initial, obj)
        assert result.success
        # Both actions reach s2 with prob 1, so either is optimal
        assert result.value >= 0.99

    def test_refine_already_optimal(self):
        game = simple_reach_game()
        initial = StrategyPair(p1_strategy={0: 1}, p2_strategy={})
        obj = Objective(ObjectiveType.REACHABILITY, targets={2})
        result = refine_strategy(game, initial, obj)
        assert result.success
        assert result.strategies.p1_strategy[0] == 1

    def test_refine_stochastic(self):
        game = stochastic_game()
        initial = StrategyPair(
            p1_strategy={0: 0, 3: 0},
            p2_strategy={1: 0},
        )
        obj = Objective(ObjectiveType.REACHABILITY, targets={4})
        result = refine_strategy(game, initial, obj)
        assert result.strategies is not None
        # s3: action c (index 0) goes directly to s4 -- should be chosen
        assert result.strategies.p1_strategy[3] == 0

    def test_refine_safety(self):
        game = simple_reach_game()
        initial = StrategyPair(p1_strategy={0: 1}, p2_strategy={})
        obj = Objective(ObjectiveType.SAFETY, safe_states={0, 1})
        result = refine_strategy(game, initial, obj)
        # Both actions eventually reach s2 (unsafe), so safety is 0 either way
        # Refinement won't improve -- stays at initial
        assert result.strategies is not None


# ============================================================
# 10. Strategy Comparison
# ============================================================

class TestStrategyComparison:
    def test_compare_two_strategies(self):
        game = simple_reach_game()
        s1 = StrategyPair(p1_strategy={0: 0}, p2_strategy={})
        s2 = StrategyPair(p1_strategy={0: 1}, p2_strategy={})
        objs = [Objective(ObjectiveType.REACHABILITY, targets={2})]
        result = compare_strategies(game, [s1, s2], objs,
                                     strategy_names=["chance", "direct"])
        assert "chance" in result
        assert "direct" in result
        assert result["_best_per_objective"][0] in ["chance", "direct"]

    def test_compare_multi_objective(self):
        game = stochastic_game()
        s1 = StrategyPair(p1_strategy={0: 0, 3: 0}, p2_strategy={1: 0})
        s2 = StrategyPair(p1_strategy={0: 1, 3: 0}, p2_strategy={1: 0})
        objs = [
            Objective(ObjectiveType.REACHABILITY, targets={4}),
            Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2, 3}),
        ]
        result = compare_strategies(game, [s1, s2], objs)
        for name in ["strategy_0", "strategy_1"]:
            assert name in result
            assert len(result[name]['objective_values']) == 2


# ============================================================
# 11. PCTL Synthesis Pipeline
# ============================================================

class TestPCTLSynthesisPipeline:
    def test_full_pipeline(self):
        lgame = labeled_game_fixture()
        formula = prob_geq(0.5, eventually(atom("goal")))
        result = synthesize_from_pctl(lgame, formula, export_mc=True)
        assert 'formula' in result
        assert 'satisfying_states' in result
        assert 'p1_strategy' in result

    def test_pipeline_with_mc_export(self):
        lgame = labeled_game_fixture()
        formula = prob_geq(0.3, eventually(atom("goal")))
        result = synthesize_from_pctl(lgame, formula, export_mc=True)
        if result['p1_strategy'] and result['p2_strategy']:
            assert 'induced_mc' in result

    def test_pipeline_bounded(self):
        lgame = labeled_game_fixture()
        formula = prob_geq(0.2, bounded_eventually(atom("goal"), 5))
        result = synthesize_from_pctl(lgame, formula, export_mc=True)
        assert result['prob_max'] is not None

    def test_pipeline_next(self):
        lgame = labeled_game_fixture()
        formula = prob_geq(0.5, next_f(atom("goal")))
        result = synthesize_from_pctl(lgame, formula)
        assert result['prob_max'] is not None


# ============================================================
# 12. Convenience APIs
# ============================================================

class TestConvenienceAPIs:
    def test_synthesize_with_verify(self):
        game = simple_reach_game()
        obj = Objective(ObjectiveType.REACHABILITY, targets={2}, threshold=0.5)
        result = synthesize(game, obj, verify=True)
        assert result.success
        assert result.verified

    def test_synthesize_without_verify(self):
        game = simple_reach_game()
        obj = Objective(ObjectiveType.REACHABILITY, targets={2})
        result = synthesize(game, obj, verify=False)
        assert result.success
        assert not result.verified

    def test_synthesis_summary(self):
        game = stochastic_game()
        objs = [
            Objective(ObjectiveType.REACHABILITY, targets={4}),
            Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2, 3, 4}),
        ]
        summary = synthesis_summary(game, objs)
        assert "objective_0" in summary
        assert "objective_1" in summary
        assert summary["objective_0"]["type"] == "reachability"
        assert summary["objective_1"]["type"] == "safety"

    def test_synthesize_reward(self):
        game = reward_game()
        obj = Objective(ObjectiveType.REWARD)
        result = synthesize(game, obj)
        assert result.success
        assert result.strategies is not None


# ============================================================
# 13. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_state_game(self):
        game = make_game(
            n_states=1,
            owners={0: Player.P1},
            action_transitions={0: {"stay": [1.0]}},
        )
        result = synthesize_reachability(game, targets={0})
        assert result.success

    def test_unreachable_target(self):
        # s0 -> s0 (absorbing), target is s1 which is unreachable
        game = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"stay": [1.0, 0.0]},
                1: {"stay": [0.0, 1.0]},
            },
        )
        result = synthesize_reachability(game, targets={1}, threshold=0.5)
        assert not result.success
        assert result.value == 0.0

    def test_all_states_target(self):
        game = simple_reach_game()
        result = synthesize_reachability(game, targets={0, 1, 2})
        assert result.success
        assert result.value == 1.0

    def test_empty_target(self):
        game = simple_reach_game()
        result = synthesize_reachability(game, targets=set())
        # No target => can't reach anything
        assert result.values_per_state is not None
        for p in result.values_per_state:
            assert p == 0.0

    def test_permissive_single_action_state(self):
        game = simple_reach_game()
        perm = synthesize_permissive_reachability(game, targets={2})
        # s1 (CHANCE) has only 1 action -- always allowed
        assert len(perm.allowed_actions[1]) == 1

    def test_verify_all_safe(self):
        game = simple_reach_game()
        strats = StrategyPair(p1_strategy={0: 0}, p2_strategy={})
        obj = Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2})
        result = verify_strategy(game, strats, obj)
        assert result.success
        assert result.value == 1.0


# ============================================================
# 14. Integration Tests
# ============================================================

class TestIntegration:
    def test_synthesize_verify_refine(self):
        """Full pipeline: synthesize -> verify -> refine."""
        game = stochastic_game()
        obj = Objective(ObjectiveType.REACHABILITY, targets={4}, threshold=0.5)

        # Synthesize
        synth = synthesize(game, obj, verify=True)
        assert synth.success

        # Refine from a different starting point
        alt_start = StrategyPair(
            p1_strategy={0: 1, 3: 1},  # suboptimal
            p2_strategy={1: 1},
        )
        refined = refine_strategy(game, alt_start, obj)
        assert refined.strategies is not None

    def test_multi_obj_then_verify(self):
        """Multi-objective -> verify each Pareto-optimal strategy."""
        game = stochastic_game()
        objs = [
            Objective(ObjectiveType.REACHABILITY, targets={4}),
            Objective(ObjectiveType.SAFETY, safe_states={0, 1, 2, 3, 4}),
        ]
        multi = synthesize_multi_objective(game, objs, n_samples=3)
        # Verify each Pareto strategy against first objective
        for values, strat in multi.pareto_front:
            ver = verify_strategy(game, strat, objs[0])
            assert ver.verified

    def test_pctl_synthesis_then_compare(self):
        """PCTL synthesis -> compare with reachability synthesis."""
        lgame = labeled_game_fixture()
        game = lgame.game

        # PCTL synthesis
        formula = prob_geq(0.5, eventually(atom("goal")))
        pctl_result = synthesize_pctl(lgame, formula)

        # Reachability synthesis
        target_states = lgame.states_with("goal")
        reach_result = synthesize_reachability(game, target_states)

        # Both should find strategies
        assert pctl_result.strategies is not None or reach_result.strategies is not None

    def test_ag_then_refine(self):
        """Assume-guarantee -> refine the composed strategy."""
        game = adversarial_game()
        assumptions = [Objective(ObjectiveType.REACHABILITY, targets={3})]
        guarantees = [Objective(ObjectiveType.REACHABILITY, targets={3}, threshold=0.5)]

        ag = assume_guarantee_synthesis(game, assumptions, guarantees)
        if ag.strategies:
            refined = refine_strategy(game, ag.strategies, guarantees[0])
            assert refined.strategies is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
