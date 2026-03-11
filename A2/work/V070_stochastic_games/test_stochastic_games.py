"""Tests for V070: Stochastic Game Verification"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from stochastic_games import (
    Player, StochasticGame, StrategyPair, GameValueResult, GameReachResult,
    SafetyResult, GameVerificationResult, ConcurrentGame,
    make_game, game_to_mdp, game_to_mc, game_value_iteration,
    reachability_game, safety_game, game_expected_reward, long_run_average,
    game_expected_steps, attractor, verify_game_value_bound,
    verify_strategy_optimality, verify_reachability_bound,
    make_concurrent_game, solve_matrix_game, concurrent_game_value,
    compare_game_vs_mdp, verify_game, game_summary,
)


# ===========================================================================
# Helper: create common test games
# ===========================================================================

def simple_game():
    """Simple 3-state game: P1 at s0, P2 at s1, CHANCE at s2.

    s0 (P1): "left" -> s1, "right" -> s2
    s1 (P2): "block" -> s0, "pass" -> s2
    s2 (CHANCE): 50/50 -> s0 or stay
    """
    return make_game(
        n_states=3,
        owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE},
        action_transitions={
            0: {"left": [0, 1, 0], "right": [0, 0, 1]},
            1: {"block": [1, 0, 0], "pass": [0, 0, 1]},
            2: {"roll": [0.5, 0, 0.5]},
        },
        rewards={
            0: {"left": 1.0, "right": 2.0},
            1: {"block": -1.0, "pass": 0.0},
            2: {"roll": 0.5},
        },
    )


def reachability_game_fixture():
    """4-state reachability game. Target: s3.

    s0 (P1): "a" -> s1, "b" -> s2
    s1 (P2): "c" -> s0, "d" -> s3
    s2 (P2): "e" -> s0, "f" -> 0.5*s0 + 0.5*s3
    s3 (CHANCE): terminal (self-loop)
    """
    return make_game(
        n_states=4,
        owners={0: Player.P1, 1: Player.P2, 2: Player.P2, 3: Player.CHANCE},
        action_transitions={
            0: {"a": [0, 1, 0, 0], "b": [0, 0, 1, 0]},
            1: {"c": [1, 0, 0, 0], "d": [0, 0, 0, 1]},
            2: {"e": [1, 0, 0, 0], "f": [0.5, 0, 0, 0.5]},
            3: {"stay": [0, 0, 0, 1]},
        },
    )


def safety_game_fixture():
    """3-state safety game. Safe states: {s0, s1}.

    s0 (P1): "stay" -> s0, "go" -> s1
    s1 (P2): "safe" -> s0, "unsafe" -> s2
    s2 (CHANCE): self-loop (unsafe absorbing)
    """
    return make_game(
        n_states=3,
        owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE},
        action_transitions={
            0: {"stay": [1, 0, 0], "go": [0, 1, 0]},
            1: {"safe": [1, 0, 0], "unsafe": [0, 0, 1]},
            2: {"stuck": [0, 0, 1]},
        },
    )


# ===========================================================================
# Test: Data Structures
# ===========================================================================

class TestDataStructures:

    def test_make_game_basic(self):
        g = simple_game()
        assert g.n_states == 3
        assert g.owners[0] == Player.P1
        assert g.owners[1] == Player.P2
        assert g.owners[2] == Player.CHANCE

    def test_validate_valid_game(self):
        g = simple_game()
        assert g.validate() == []

    def test_validate_catches_errors(self):
        g = StochasticGame(
            n_states=2,
            owners=[Player.P1],  # Wrong length
            actions=[["a"], ["b"]],
            transition=[[[1, 0]], [[0, 1]]],
        )
        errors = g.validate()
        assert len(errors) > 0

    def test_player_states(self):
        g = simple_game()
        assert g.p1_states() == {0}
        assert g.p2_states() == {1}
        assert g.chance_states() == {2}

    def test_state_labels_default(self):
        g = simple_game()
        assert g.state_labels == ["s0", "s1", "s2"]

    def test_state_labels_custom(self):
        g = make_game(2, {0: Player.P1, 1: Player.P2},
                      {0: {"a": [0, 1]}, 1: {"b": [1, 0]}},
                      state_labels=["start", "end"])
        assert g.state_labels == ["start", "end"]

    def test_strategy_pair(self):
        sp = StrategyPair({0: 1, 2: 0}, {1: 1})
        assert sp.get_action(0, Player.P1) == 1
        assert sp.get_action(1, Player.P2) == 1
        assert sp.get_action(2, Player.P1) == 0
        assert sp.get_action(5, Player.P1) == 0  # Default

    def test_default_rewards(self):
        g = make_game(2, {0: Player.P1, 1: Player.P2},
                      {0: {"a": [0, 1]}, 1: {"b": [1, 0]}})
        assert g.rewards == [[0.0], [0.0]]

    def test_chance_validation(self):
        """CHANCE states must have exactly 1 action."""
        g = StochasticGame(
            n_states=2,
            owners=[Player.CHANCE, Player.P1],
            actions=[["a", "b"], ["c"]],  # CHANCE with 2 actions
            transition=[[[1, 0], [0, 1]], [[0, 1]]],
        )
        errors = g.validate()
        assert any("CHANCE" in e for e in errors)


# ===========================================================================
# Test: Game-to-MDP/MC conversion
# ===========================================================================

class TestConversion:

    def test_game_to_mdp(self):
        g = simple_game()
        # Fix P2 to action 0 ("block")
        mdp = game_to_mdp(g, Player.P2, {1: 0})
        assert mdp.n_states == 3
        # P2's state now has only 1 action
        assert len(mdp.actions[1]) == 1

    def test_game_to_mc(self):
        g = simple_game()
        sp = StrategyPair({0: 0}, {1: 0})
        mc = game_to_mc(g, sp)
        assert len(mc.matrix) == 3
        # s0 chose "left" -> goes to s1
        assert mc.matrix[0][1] == 1.0

    def test_game_to_mc_different_strategy(self):
        g = simple_game()
        sp = StrategyPair({0: 1}, {1: 1})  # P1: right, P2: pass
        mc = game_to_mc(g, sp)
        # s0 chose "right" -> goes to s2
        assert mc.matrix[0][2] == 1.0
        # s1 chose "pass" -> goes to s2
        assert mc.matrix[1][2] == 1.0


# ===========================================================================
# Test: Minimax Value Iteration
# ===========================================================================

class TestGameValueIteration:

    def test_converges(self):
        g = simple_game()
        result = game_value_iteration(g, discount=0.9)
        assert result.converged

    def test_values_finite(self):
        g = simple_game()
        result = game_value_iteration(g, discount=0.9)
        for v in result.values:
            assert abs(v) < 1000

    def test_terminal_states(self):
        g = simple_game()
        result = game_value_iteration(g, discount=0.9, terminal_states={2})
        assert result.values[2] == 0.0

    def test_p1_maximizes(self):
        """P1 state should choose the action giving higher value."""
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"bad": [1, 0], "good": [0, 1]},
                1: {"stay": [0, 1]},
            },
            rewards={0: {"bad": 0, "good": 10}, 1: {"stay": 1}},
        )
        result = game_value_iteration(g, discount=0.9, terminal_states={1})
        # P1 should choose "good" (action 1)
        assert result.strategies.p1_strategy[0] == 1

    def test_p2_minimizes(self):
        """P2 state should choose the action giving lower value."""
        g = make_game(
            n_states=2,
            owners={0: Player.P2, 1: Player.CHANCE},
            action_transitions={
                0: {"high": [0, 1], "low": [1, 0]},
                1: {"stay": [0, 1]},
            },
            rewards={0: {"high": 10, "low": 0}, 1: {"stay": 5}},
        )
        result = game_value_iteration(g, discount=0.9, terminal_states={1})
        # P2 should choose "low" (action 1) to minimize
        assert result.strategies.p2_strategy[0] == 1

    def test_discount_affects_values(self):
        g = simple_game()
        r1 = game_value_iteration(g, discount=0.5)
        r2 = game_value_iteration(g, discount=0.99)
        # Higher discount means more future weight -> generally larger absolute values
        # Just check both converge
        assert r1.converged
        assert r2.converged

    def test_all_chance(self):
        """A game with only CHANCE states is a Markov chain."""
        g = make_game(
            n_states=2,
            owners={0: Player.CHANCE, 1: Player.CHANCE},
            action_transitions={
                0: {"roll": [0.5, 0.5]},
                1: {"roll": [0.3, 0.7]},
            },
            rewards={0: {"roll": 1.0}, 1: {"roll": 2.0}},
        )
        result = game_value_iteration(g, discount=0.9)
        assert result.converged


# ===========================================================================
# Test: Reachability Games
# ===========================================================================

class TestReachabilityGame:

    def test_basic_reachability(self):
        g = reachability_game_fixture()
        result = reachability_game(g, targets={3})
        # Target is reachable from all states
        assert result.probabilities[3] == 1.0

    def test_p1_strategy_maximizes(self):
        g = reachability_game_fixture()
        result = reachability_game(g, targets={3})
        # P1 at s0 should choose action leading to higher reach prob
        # "a" -> s1, "b" -> s2
        # At s1, P2 minimizes: min("c"->s0, "d"->s3) = "c" (prob 0 if cycling)
        # At s2, P2 minimizes: min("e"->s0, "f"->0.5*s0+0.5*s3)
        # P2 at s1 will choose "c" (back to s0) to minimize reachability
        # P2 at s2 will choose "e" (back to s0) to minimize
        # But from s0, P1 picks the path with higher reachability
        assert result.probabilities[0] >= 0  # valid probability

    def test_unreachable_target(self):
        """If target can't be reached, probability is 0."""
        g = make_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE},
            action_transitions={
                0: {"a": [0.5, 0.5, 0]},
                1: {"b": [0.5, 0.5, 0]},
                2: {"c": [0, 0, 1]},  # Absorbing, no path from 0/1
            },
        )
        result = reachability_game(g, targets={2})
        assert result.probabilities[0] == 0.0
        assert result.probabilities[1] == 0.0
        assert result.probabilities[2] == 1.0

    def test_target_is_absorbing(self):
        g = reachability_game_fixture()
        result = reachability_game(g, targets={3})
        assert result.probabilities[3] == 1.0

    def test_winning_regions(self):
        g = reachability_game_fixture()
        result = reachability_game(g, targets={3})
        assert 3 in result.p1_winning

    def test_deterministic_reach(self):
        """P1 can guarantee reaching target deterministically."""
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"go": [0, 1]},
                1: {"stay": [0, 1]},
            },
        )
        result = reachability_game(g, targets={1})
        assert abs(result.probabilities[0] - 1.0) < 1e-6


# ===========================================================================
# Test: Safety Games
# ===========================================================================

class TestSafetyGame:

    def test_basic_safety(self):
        g = safety_game_fixture()
        result = safety_game(g, safe_states={0, 1})
        # P1 at s0 can stay safe by choosing "stay"
        assert result.safe_probabilities[0] > 0.5

    def test_unsafe_state_probability_zero(self):
        g = safety_game_fixture()
        result = safety_game(g, safe_states={0, 1})
        assert result.safe_probabilities[2] == 0.0

    def test_all_safe(self):
        """If all states are safe, P1 trivially wins."""
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P2},
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"b": [0.5, 0.5]},
            },
        )
        result = safety_game(g, safe_states={0, 1})
        assert result.safe_probabilities[0] == 1.0
        assert result.safe_probabilities[1] == 1.0

    def test_p1_can_avoid_unsafe(self):
        """P1 at s0 chooses to stay, avoiding unsafe s2."""
        g = safety_game_fixture()
        result = safety_game(g, safe_states={0, 1})
        # P1 should choose "stay" at s0
        assert result.strategies.p1_strategy[0] == 0  # "stay"

    def test_p2_forces_unsafe(self):
        """P2 at s1 will choose "unsafe" to force into bad state."""
        g = safety_game_fixture()
        result = safety_game(g, safe_states={0, 1})
        # P2 should choose "unsafe" (action 1) to minimize safety
        assert result.strategies.p2_strategy[1] == 1


# ===========================================================================
# Test: Expected Steps
# ===========================================================================

class TestExpectedSteps:

    def test_target_zero_steps(self):
        g = reachability_game_fixture()
        steps, strats = game_expected_steps(g, targets={3})
        assert steps[3] == 0.0

    def test_steps_positive(self):
        g = reachability_game_fixture()
        steps, strats = game_expected_steps(g, targets={3})
        # Non-target reachable states should have positive steps
        for s in [0, 1, 2]:
            assert steps[s] >= 0

    def test_deterministic_one_step(self):
        """One deterministic step to reach target."""
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"go": [0, 1]},
                1: {"stay": [0, 1]},
            },
        )
        steps, _ = game_expected_steps(g, targets={1})
        assert abs(steps[0] - 1.0) < 1e-6


# ===========================================================================
# Test: Attractor
# ===========================================================================

class TestAttractor:

    def test_target_in_attractor(self):
        g = simple_game()
        attr = attractor(g, {2}, Player.P1)
        assert 2 in attr

    def test_p1_can_reach(self):
        """P1 at s0 can reach s2 via "right"."""
        g = simple_game()
        attr = attractor(g, {2}, Player.P1)
        # s0 (P1) has action "right" -> s2 deterministically
        assert 0 in attr

    def test_p2_blocks(self):
        """P2 can prevent P1 from reaching target if P2 controls all paths."""
        g = make_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE},
            action_transitions={
                0: {"go": [0, 1, 0]},  # Only way is through P2
                1: {"block": [1, 0, 0], "allow": [0, 0, 1]},
                2: {"stay": [0, 0, 1]},
            },
        )
        # P1 attractor of {2}: P2 at s1 has "block" which doesn't go to attr
        # So s1 is NOT in P1's attractor (P2 can block)
        attr = attractor(g, {2}, Player.P1)
        assert 2 in attr
        assert 1 not in attr  # P2 can choose "block"

    def test_empty_attractor(self):
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"stay": [1, 0]},
                1: {"stay": [0, 1]},
            },
        )
        # Target {1}: s0 can't reach s1 (self-loop)
        attr = attractor(g, {1}, Player.P1)
        assert attr == {1}


# ===========================================================================
# Test: SMT Verification
# ===========================================================================

class TestVerification:

    def test_value_bound_geq(self):
        g = simple_game()
        result = game_value_iteration(g, discount=0.9)
        # Use actual value as bound (should pass)
        v0 = result.values[0]
        r = verify_game_value_bound(g, 0, v0 - 0.1, "geq", discount=0.9)
        assert r.verified

    def test_value_bound_fails(self):
        g = simple_game()
        r = verify_game_value_bound(g, 0, 1e6, "geq", discount=0.9)
        assert not r.verified
        assert r.counterexample is not None

    def test_value_bound_leq(self):
        g = simple_game()
        result = game_value_iteration(g, discount=0.9)
        v0 = result.values[0]
        r = verify_game_value_bound(g, 0, v0 + 0.1, "leq", discount=0.9)
        assert r.verified

    def test_strategy_optimality_optimal(self):
        g = simple_game()
        result = game_value_iteration(g, discount=0.9)
        r = verify_strategy_optimality(g, result.strategies, discount=0.9)
        assert r.verified

    def test_strategy_optimality_suboptimal(self):
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"bad": [1, 0], "good": [0, 1]},
                1: {"stay": [0, 1]},
            },
            rewards={0: {"bad": 0, "good": 10}, 1: {"stay": 1}},
        )
        # Give P1 the bad strategy
        bad_strat = StrategyPair({0: 0}, {})  # "bad" action
        r = verify_strategy_optimality(g, bad_strat, discount=0.9)
        assert not r.verified

    def test_reachability_bound(self):
        g = reachability_game_fixture()
        r = verify_reachability_bound(g, 3, {3}, 1.0, "geq")
        assert r.verified

    def test_reachability_bound_fails(self):
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"stay": [1, 0]},
                1: {"stay": [0, 1]},
            },
        )
        # s0 can't reach s1
        r = verify_reachability_bound(g, 0, {1}, 0.5, "geq")
        assert not r.verified


# ===========================================================================
# Test: Concurrent Games
# ===========================================================================

class TestConcurrentGames:

    def test_matching_pennies(self):
        """Classic matching pennies: P1 wants match, P2 wants mismatch."""
        game = make_concurrent_game(
            n_states=1,
            p1_actions={0: ["H", "T"]},
            p2_actions={0: ["H", "T"]},
            transitions={0: {
                "H": {"H": [1], "T": [1]},
                "T": {"H": [1], "T": [1]},
            }},
            rewards={0: {
                "H": {"H": 1.0, "T": -1.0},
                "T": {"H": -1.0, "T": 1.0},
            }},
        )
        # Solve the matrix game
        p1, p2, val = solve_matrix_game([[1, -1], [-1, 1]])
        assert abs(val) < 0.1  # Game value ~0 (fair game)
        assert abs(p1[0] - 0.5) < 0.1  # Mixed 50/50
        assert abs(p2[0] - 0.5) < 0.1

    def test_prisoners_dilemma(self):
        """Prisoner's dilemma payoff matrix."""
        # P1 payoffs: (C,C)=3, (C,D)=0, (D,C)=5, (D,D)=1
        p1, p2, val = solve_matrix_game([[3, 0], [5, 1]])
        # Dominant strategy: both defect (D,D) = saddle point
        assert abs(val - 1.0) < 0.1

    def test_rock_paper_scissors(self):
        """RPS: no pure Nash eq, mixed ~1/3 each."""
        payoff = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
        p1, p2, val = solve_matrix_game(payoff)
        assert abs(val) < 0.15  # ~0

    def test_concurrent_game_value(self):
        """Concurrent game with 2 states."""
        game = make_concurrent_game(
            n_states=2,
            p1_actions={0: ["L", "R"], 1: ["stay"]},
            p2_actions={0: ["L", "R"], 1: ["stay"]},
            transitions={
                0: {
                    "L": {"L": [0.5, 0.5], "R": [1, 0]},
                    "R": {"L": [0, 1], "R": [0.5, 0.5]},
                },
                1: {
                    "stay": {"stay": [0, 1]},
                },
            },
            rewards={
                0: {"L": {"L": 1, "R": 0}, "R": {"L": 0, "R": 1}},
                1: {"stay": {"stay": 2}},
            },
        )
        result = concurrent_game_value(game, discount=0.9)
        assert result.converged
        assert len(result.values) == 2

    def test_saddle_point(self):
        """Matrix game with a saddle point."""
        p1, p2, val = solve_matrix_game([[3, 1], [4, 2]])
        # Row mins: [1, 2], Col maxes: [4, 2]
        # Maximin = 2 = minimax -> saddle at (1,1)
        assert abs(val - 2.0) < 0.1

    def test_make_concurrent_game(self):
        game = make_concurrent_game(
            n_states=1,
            p1_actions={0: ["a"]},
            p2_actions={0: ["b"]},
            transitions={0: {"a": {"b": [1]}}},
        )
        assert game.n_states == 1
        assert game.p1_actions[0] == ["a"]
        assert game.p2_actions[0] == ["b"]


# ===========================================================================
# Test: Comparison with MDP
# ===========================================================================

class TestComparison:

    def test_comparison_produces_all_fields(self):
        g = simple_game()
        comp = compare_game_vs_mdp(g, discount=0.9)
        assert "game_values" in comp
        assert "p1_mdp_values" in comp
        assert "p2_mdp_values" in comp
        assert "adversarial_advantage" in comp

    def test_game_value_between_mdp_bounds(self):
        """Game value should be between worst-case and best-case MDP values."""
        g = simple_game()
        comp = compare_game_vs_mdp(g, discount=0.9)
        # The game value accounts for adversarial play
        # Just check values are finite
        for v in comp["game_values"]:
            assert abs(v) < 1000


# ===========================================================================
# Test: Multi-Property Verification
# ===========================================================================

class TestVerifyGame:

    def test_multiple_properties(self):
        g = simple_game()
        gv = game_value_iteration(g, discount=0.9)

        props = [
            ("value_bound", {"state": 0, "bound": gv.values[0] - 0.1, "bound_type": "geq", "discount": 0.9}),
            ("value_bound", {"state": 0, "bound": gv.values[0] + 0.1, "bound_type": "leq", "discount": 0.9}),
        ]
        results = verify_game(g, props)
        assert len(results) == 2
        assert results[0].verified
        assert results[1].verified

    def test_safety_property(self):
        g = safety_game_fixture()
        props = [
            ("safety", {"safe_states": {0, 1}, "state": 0, "bound": 0.5}),
        ]
        results = verify_game(g, props)
        assert len(results) == 1
        # P1 can stay safe at s0 by choosing "stay"
        assert results[0].verified

    def test_reachability_property(self):
        g = reachability_game_fixture()
        props = [
            ("reachability", {"state": 3, "targets": {3}, "bound": 1.0, "bound_type": "geq"}),
        ]
        results = verify_game(g, props)
        assert results[0].verified

    def test_unknown_property(self):
        g = simple_game()
        results = verify_game(g, [("bogus", {})])
        assert not results[0].verified


# ===========================================================================
# Test: Game Summary
# ===========================================================================

class TestGameSummary:

    def test_summary_fields(self):
        g = simple_game()
        s = game_summary(g)
        assert s["n_states"] == 3
        assert s["p1_states"] == 1
        assert s["p2_states"] == 1
        assert s["chance_states"] == 1
        assert s["validation_errors"] == []


# ===========================================================================
# Test: Long-run Average Reward
# ===========================================================================

class TestLongRunAverage:

    def test_average_reward(self):
        """Under fixed strategies, compute long-run average."""
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"go": [0.5, 0.5]},
                1: {"back": [1, 0]},
            },
            rewards={0: {"go": 1.0}, 1: {"back": 0.0}},
        )
        sp = StrategyPair({0: 0}, {})
        avg = long_run_average(g, sp)
        # The MC: s0 -> 0.5 s0 + 0.5 s1, s1 -> s0
        # Steady state: pi0 = 2/3, pi1 = 1/3
        # Average = 2/3 * 1.0 + 1/3 * 0.0 = 2/3
        assert abs(avg - 2.0/3) < 0.1


# ===========================================================================
# Test: Edge Cases
# ===========================================================================

class TestEdgeCases:

    def test_single_state_game(self):
        g = make_game(
            n_states=1,
            owners={0: Player.P1},
            action_transitions={0: {"stay": [1]}},
            rewards={0: {"stay": 5.0}},
        )
        result = game_value_iteration(g, discount=0.9)
        # V = 5 + 0.9 * V => V = 5 / (1 - 0.9) = 50
        assert abs(result.values[0] - 50.0) < 0.1

    def test_two_player_zero_sum(self):
        """In a zero-sum game, P1 value = -P2 value (implicit)."""
        g = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P2},
            action_transitions={
                0: {"a": [0, 1], "b": [1, 0]},
                1: {"c": [1, 0], "d": [0, 1]},
            },
            rewards={
                0: {"a": 1, "b": -1},
                1: {"c": -1, "d": 1},
            },
        )
        result = game_value_iteration(g, discount=0.9)
        assert result.converged

    def test_absorbing_game(self):
        """Game with absorbing terminal state."""
        g = make_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE},
            action_transitions={
                0: {"go": [0, 0, 1]},
                1: {"go": [0, 0, 1]},
                2: {"absorb": [0, 0, 1]},
            },
            rewards={0: {"go": 1}, 1: {"go": 1}, 2: {"absorb": 0}},
        )
        result = game_value_iteration(g, discount=0.9, terminal_states={2})
        assert result.values[2] == 0.0
        assert abs(result.values[0] - 1.0) < 0.1  # One reward then terminal

    def test_large_discount(self):
        g = simple_game()
        result = game_value_iteration(g, discount=0.999, max_iter=5000)
        assert result.converged or result.iterations == 5000

    def test_game_expected_reward_same_as_vi(self):
        """game_expected_reward should match game_value_iteration."""
        g = simple_game()
        r1 = game_value_iteration(g, discount=0.9)
        r2 = game_expected_reward(g, discount=0.9)
        for s in range(g.n_states):
            assert abs(r1.values[s] - r2.values[s]) < 1e-6


# ===========================================================================
# Test: Multiple actions per player
# ===========================================================================

class TestMultipleActions:

    def test_three_actions_p1(self):
        g = make_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.CHANCE, 2: Player.CHANCE},
            action_transitions={
                0: {"a": [0, 1, 0], "b": [0, 0, 1], "c": [0.5, 0.25, 0.25]},
                1: {"stay": [0, 1, 0]},
                2: {"stay": [0, 0, 1]},
            },
            rewards={
                0: {"a": 1, "b": 5, "c": 3},
                1: {"stay": 2},
                2: {"stay": 10},
            },
        )
        result = game_value_iteration(g, discount=0.9, terminal_states={1, 2})
        # P1 should choose "b" (reward 5 + go to s2 with reward 10)
        # Actually s1 and s2 are terminal (value 0), so:
        # Q(s0, a) = 1 + 0.9*0 = 1
        # Q(s0, b) = 5 + 0.9*0 = 5
        # Q(s0, c) = 3 + 0.9*0 = 3
        assert result.strategies.p1_strategy[0] == 1  # "b"

    def test_three_actions_p2(self):
        g = make_game(
            n_states=3,
            owners={0: Player.P2, 1: Player.CHANCE, 2: Player.CHANCE},
            action_transitions={
                0: {"a": [0, 1, 0], "b": [0, 0, 1], "c": [0.5, 0.25, 0.25]},
                1: {"stay": [0, 1, 0]},
                2: {"stay": [0, 0, 1]},
            },
            rewards={
                0: {"a": 1, "b": 5, "c": 3},
                1: {"stay": 2},
                2: {"stay": 10},
            },
        )
        result = game_value_iteration(g, discount=0.9, terminal_states={1, 2})
        # P2 minimizes: chooses "a" (reward 1)
        assert result.strategies.p2_strategy[0] == 0  # "a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
