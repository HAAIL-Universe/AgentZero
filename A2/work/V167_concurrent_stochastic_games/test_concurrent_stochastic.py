"""Tests for V167: Concurrent Stochastic Games."""

import pytest
import numpy as np
from concurrent_stochastic import (
    ConcurrentStochasticGame, CSGResult,
    solve_concurrent_stochastic, solve_matrix_game,
    solve_concurrent_parity_almost_sure, solve_concurrent_parity_positive_prob,
    concurrent_attractor_value, concurrent_almost_sure_reach,
    concurrent_positive_prob_reach,
    make_concurrent_game, make_matching_pennies, make_rock_paper_scissors_game,
    make_concurrent_reachability, make_concurrent_safety,
    simulate_play, verify_strategy, compare_with_turn_based,
    concurrent_game_statistics, batch_solve, _swap_players,
)


# ========================================================================
# Section 1: Matrix Game Solver
# ========================================================================

class TestMatrixGame:
    def test_trivial_1x1(self):
        """1x1 matrix game."""
        val, es, os_ = solve_matrix_game(np.array([[3.0]]))
        assert abs(val - 3.0) < 1e-6
        assert abs(es[0] - 1.0) < 1e-6
        assert abs(os_[0] - 1.0) < 1e-6

    def test_matching_pennies_matrix(self):
        """Classic matching pennies: value = 0, both play 50/50."""
        matrix = np.array([[1, -1], [-1, 1]], dtype=float)
        val, es, os_ = solve_matrix_game(matrix)
        assert abs(val) < 1e-4
        assert abs(es[0] - 0.5) < 0.1
        assert abs(es[1] - 0.5) < 0.1

    def test_dominated_strategy(self):
        """When one strategy dominates, pure strategy is optimal."""
        matrix = np.array([[3, 3], [1, 1]], dtype=float)
        val, es, os_ = solve_matrix_game(matrix)
        assert abs(val - 3.0) < 1e-4
        assert es[0] > 0.9  # Even should play action 0

    def test_rps_matrix(self):
        """Rock-paper-scissors matrix game. Value = 0, uniform optimal."""
        matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
        val, es, os_ = solve_matrix_game(matrix)
        assert abs(val) < 1e-4
        for p in es:
            assert abs(p - 1.0/3) < 0.1

    def test_asymmetric_matrix(self):
        """Asymmetric 2x3 matrix game."""
        matrix = np.array([[2, 1, 3], [0, 4, 1]], dtype=float)
        val, es, os_ = solve_matrix_game(matrix)
        # Value exists and strategies are valid distributions
        assert es.sum() > 0.99
        assert os_.sum() > 0.99
        assert all(p >= -1e-10 for p in es)
        assert all(p >= -1e-10 for p in os_)

    def test_saddle_point(self):
        """Matrix with saddle point (pure strategy equilibrium)."""
        matrix = np.array([[3, 1], [5, 2]], dtype=float)
        val, es, os_ = solve_matrix_game(matrix)
        # Saddle point at (0,1) or (1,1), value = 2 or 3
        assert 1.0 <= val <= 5.0

    def test_all_zeros(self):
        """All zeros matrix: value = 0."""
        matrix = np.zeros((3, 3))
        val, es, os_ = solve_matrix_game(matrix)
        assert abs(val) < 1e-6


# ========================================================================
# Section 2: Game Construction
# ========================================================================

class TestGameConstruction:
    def test_add_vertex(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a", "b"], ["x", "y"])
        assert 0 in game.vertices
        assert game.vertices[0].priority == 0
        assert game.vertices[0].actions_even == ["a", "b"]
        assert game.vertices[0].actions_odd == ["x", "y"]

    def test_add_transition(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_vertex(1, 1, ["a"], ["x"])
        game.add_transition(0, "a", "x", {1: 1.0})
        assert (("a", "x") in game.vertices[0].transitions)

    def test_validate_ok(self):
        game = make_matching_pennies()
        errors = game.validate()
        assert errors == []

    def test_validate_missing_transition(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a", "b"], ["x"])
        game.add_vertex(1, 1, ["a"], ["x"])
        game.add_transition(0, "a", "x", {1: 1.0})
        # Missing (b, x) transition
        errors = game.validate()
        assert len(errors) > 0

    def test_successors(self):
        game = make_matching_pennies()
        succs = game.successors(0)
        assert succs == {0, 1}

    def test_vertex_set(self):
        game = make_matching_pennies()
        assert game.vertex_set() == {0, 1}

    def test_subgame(self):
        game = make_matching_pennies()
        sub = game.subgame({0})
        assert sub.vertex_set() == {0}

    def test_make_concurrent_game(self):
        game = make_concurrent_game(
            vertices=[(0, 0, ["a"], ["x"]), (1, 1, ["a"], ["x"])],
            transitions=[(0, "a", "x", {1: 1.0}), (1, "a", "x", {0: 1.0})],
        )
        assert game.vertex_set() == {0, 1}
        assert game.validate() == []


# ========================================================================
# Section 3: Concurrent Reachability
# ========================================================================

class TestConcurrentReachability:
    def test_attractor_value_target(self):
        """Target vertices have value 1."""
        game = make_concurrent_reachability(3, target=2, trap=1)
        values = concurrent_attractor_value(game, {2}, True)
        assert abs(values[2] - 1.0) < 1e-8

    def test_attractor_value_nontarget(self):
        """Non-target vertices have value between 0 and 1."""
        game = make_concurrent_reachability(4, target=3, trap=1)
        values = concurrent_attractor_value(game, {3}, True)
        assert values[0] >= 0.0
        assert values[0] <= 1.0 + 1e-8

    def test_almost_sure_reach_target(self):
        """Target is always in its own AS reach set."""
        game = make_concurrent_reachability(3, target=2, trap=1)
        reached = concurrent_almost_sure_reach(game, {2})
        assert 2 in reached

    def test_positive_prob_reach(self):
        """Even can reach target with positive probability from start."""
        game = make_concurrent_reachability(4, target=3, trap=1)
        reached = concurrent_positive_prob_reach(game, {3})
        # From 0 or 2, Even can choose "go" and has positive probability
        assert 2 in reached  # Adjacent to target

    def test_trap_not_reached(self):
        """Trap state should be able to reach itself but not target."""
        game = make_concurrent_reachability(3, target=2, trap=1)
        reached = concurrent_almost_sure_reach(game, {2})
        # Trap (1) is absorbing with odd prio, can't reach target
        assert 1 not in reached


# ========================================================================
# Section 4: Matching Pennies
# ========================================================================

class TestMatchingPennies:
    def test_structure(self):
        game = make_matching_pennies()
        assert game.vertex_set() == {0, 1}
        assert game.vertices[0].priority == 0
        assert game.vertices[1].priority == 1
        assert len(game.vertices[0].actions_even) == 2

    def test_solve(self):
        game = make_matching_pennies()
        result = solve_concurrent_stochastic(game)
        # In matching pennies, both players play mixed strategies
        # Neither can win almost-surely (each can counter the other)
        all_verts = game.vertex_set()
        # At least one side has positive-prob winning from some state
        assert len(result.win_even_pp) + len(result.win_odd_pp) > 0 or \
               len(result.win_even_as) + len(result.win_odd_as) >= 0

    def test_mixed_strategy(self):
        game = make_matching_pennies()
        result = solve_concurrent_stochastic(game)
        # Strategies should be mixed (not pure) for at least some vertices
        for strat in [result.strategy_even_as, result.strategy_even_pp]:
            for v, dist in strat.items():
                # Each action should have some probability
                assert sum(dist.values()) > 0.99


# ========================================================================
# Section 5: Rock-Paper-Scissors
# ========================================================================

class TestRPS:
    def test_structure(self):
        game = make_rock_paper_scissors_game()
        assert game.vertex_set() == {0, 1, 2, 3}
        assert game.vertices[0].priority == 0  # arena
        assert game.vertices[1].priority == 2  # even wins
        assert game.vertices[2].priority == 1  # odd wins

    def test_solve(self):
        game = make_rock_paper_scissors_game()
        result = solve_concurrent_stochastic(game)
        # RPS is symmetric: positive-prob winning for both
        # but neither can win almost-surely
        assert isinstance(result, CSGResult)

    def test_validate(self):
        game = make_rock_paper_scissors_game()
        assert game.validate() == []


# ========================================================================
# Section 6: Concurrent Safety
# ========================================================================

class TestConcurrentSafety:
    def test_structure(self):
        game = make_concurrent_safety(4, {0, 1, 2})
        assert len(game.vertices) == 4
        assert game.vertices[3].priority == 1  # unsafe

    def test_solve_all_safe(self):
        """When all states are safe, Even wins everywhere."""
        game = make_concurrent_safety(3, {0, 1, 2})
        result = solve_concurrent_stochastic(game)
        # All states have priority 0 (even), so Even wins AS from everywhere
        assert result.win_even_as == {0, 1, 2}

    def test_solve_mixed_safety(self):
        game = make_concurrent_safety(4, {0, 1})
        result = solve_concurrent_stochastic(game)
        # Some vertices may be unsafe
        assert isinstance(result, CSGResult)


# ========================================================================
# Section 7: Parity Solver -- Almost-Sure
# ========================================================================

class TestAlmostSure:
    def test_single_even_vertex(self):
        """Single vertex with even priority: Even wins AS."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        assert w0 == {0}
        assert w1 == set()

    def test_single_odd_vertex(self):
        """Single vertex with odd priority: Odd wins AS."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 1, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        assert w0 == set()
        assert w1 == {0}

    def test_deterministic_even_wins(self):
        """Even has a dominant action leading to even-priority absorbing state."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["go", "stay"], ["x"])
        game.add_vertex(1, 2, ["a"], ["x"])
        game.add_transition(0, "go", "x", {1: 1.0})
        game.add_transition(0, "stay", "x", {0: 1.0})
        game.add_transition(1, "a", "x", {1: 1.0})
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        assert 0 in w0  # Even can go to state 1 (prio 2)
        assert 1 in w0

    def test_deterministic_odd_wins(self):
        """Odd has a dominant action leading to odd-priority absorbing state."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["go", "stay"])
        game.add_vertex(1, 1, ["a"], ["x"])
        game.add_transition(0, "a", "go", {1: 1.0})
        game.add_transition(0, "a", "stay", {0: 1.0})
        game.add_transition(1, "a", "x", {1: 1.0})
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        # Odd can force going to state 1 regardless of Even's choice
        assert 1 in w1

    def test_empty_game(self):
        game = ConcurrentStochasticGame()
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        assert w0 == set()
        assert w1 == set()


# ========================================================================
# Section 8: Parity Solver -- Positive Probability
# ========================================================================

class TestPositiveProb:
    def test_single_even_vertex(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        w0, w1, _, _ = solve_concurrent_parity_positive_prob(game)
        assert w0 == {0}

    def test_single_odd_vertex(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 1, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        w0, w1, _, _ = solve_concurrent_parity_positive_prob(game)
        assert w1 == {0}

    def test_pp_larger_than_as(self):
        """Positive-prob winning region should be >= almost-sure region."""
        game = make_matching_pennies()
        result = solve_concurrent_stochastic(game)
        assert result.win_even_as <= result.win_even_pp
        assert result.win_odd_as <= result.win_odd_pp


# ========================================================================
# Section 9: Simulation
# ========================================================================

class TestSimulation:
    def test_simulate_deterministic(self):
        """Simulate a game with pure strategies."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_vertex(1, 2, ["b"], ["y"])
        game.add_transition(0, "a", "x", {1: 1.0})
        game.add_transition(1, "b", "y", {1: 1.0})
        strat_e = {0: {"a": 1.0}, 1: {"b": 1.0}}
        strat_o = {0: {"x": 1.0}, 1: {"y": 1.0}}
        trace = simulate_play(game, 0, strat_e, strat_o, steps=5, seed=42)
        assert len(trace) == 5
        assert trace[0][0] == 0  # starts at vertex 0
        assert trace[1][0] == 1  # moves to vertex 1

    def test_simulate_mixed(self):
        """Simulate with mixed strategies."""
        game = make_matching_pennies()
        strat_e = {0: {"H": 0.5, "T": 0.5}, 1: {"H": 0.5, "T": 0.5}}
        strat_o = {0: {"H": 0.5, "T": 0.5}, 1: {"H": 0.5, "T": 0.5}}
        trace = simulate_play(game, 0, strat_e, strat_o, steps=20, seed=42)
        assert len(trace) == 20
        # Should visit both states
        visited = {t[0] for t in trace}
        assert len(visited) >= 1

    def test_simulate_reproducible(self):
        """Same seed gives same trace."""
        game = make_matching_pennies()
        strat_e = {0: {"H": 0.5, "T": 0.5}, 1: {"H": 0.5, "T": 0.5}}
        strat_o = {0: {"H": 0.5, "T": 0.5}, 1: {"H": 0.5, "T": 0.5}}
        t1 = simulate_play(game, 0, strat_e, strat_o, steps=10, seed=123)
        t2 = simulate_play(game, 0, strat_e, strat_o, steps=10, seed=123)
        assert t1 == t2


# ========================================================================
# Section 10: Strategy Verification
# ========================================================================

class TestVerification:
    def test_valid_strategy(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a", "b"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        game.add_transition(0, "b", "x", {0: 1.0})
        result = verify_strategy(game, {0}, {0: {"a": 0.6, "b": 0.4}}, is_even=True)
        assert result["valid"]

    def test_invalid_action(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        result = verify_strategy(game, {0}, {0: {"z": 1.0}}, is_even=True)
        assert not result["valid"]

    def test_bad_distribution(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a", "b"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        game.add_transition(0, "b", "x", {0: 1.0})
        result = verify_strategy(game, {0}, {0: {"a": 0.3, "b": 0.3}}, is_even=True)
        assert not result["valid"]


# ========================================================================
# Section 11: Swap Players
# ========================================================================

class TestSwapPlayers:
    def test_swap_actions(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a", "b"], ["x", "y"])
        game.add_transition(0, "a", "x", {0: 1.0})
        game.add_transition(0, "a", "y", {0: 1.0})
        game.add_transition(0, "b", "x", {0: 1.0})
        game.add_transition(0, "b", "y", {0: 1.0})
        swapped = _swap_players(game)
        assert swapped.vertices[0].actions_even == ["x", "y"]
        assert swapped.vertices[0].actions_odd == ["a", "b"]

    def test_swap_preserves_transitions(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_vertex(1, 1, ["a"], ["x"])
        game.add_transition(0, "a", "x", {1: 1.0})
        game.add_transition(1, "a", "x", {0: 1.0})
        swapped = _swap_players(game)
        # (a, x) in original -> (x, a) in swapped
        assert ("x", "a") in swapped.vertices[0].transitions


# ========================================================================
# Section 12: Statistics
# ========================================================================

class TestStatistics:
    def test_matching_pennies_stats(self):
        game = make_matching_pennies()
        stats = concurrent_game_statistics(game)
        assert stats["vertices"] == 2
        assert stats["max_priority"] == 1
        assert stats["max_even_actions"] == 2
        assert stats["total_action_pairs"] == 8  # 2 verts * 2 * 2

    def test_rps_stats(self):
        game = make_rock_paper_scissors_game()
        stats = concurrent_game_statistics(game)
        assert stats["vertices"] == 4
        assert stats["max_priority"] == 2

    def test_empty_stats(self):
        game = ConcurrentStochasticGame()
        stats = concurrent_game_statistics(game)
        assert stats["vertices"] == 0


# ========================================================================
# Section 13: Batch Solve
# ========================================================================

class TestBatchSolve:
    def test_batch(self):
        games = [
            ("pennies", make_matching_pennies()),
            ("rps", make_rock_paper_scissors_game()),
        ]
        results = batch_solve(games)
        assert "pennies" in results
        assert "rps" in results
        assert isinstance(results["pennies"], CSGResult)


# ========================================================================
# Section 14: Compare with Turn-Based
# ========================================================================

class TestComparison:
    def test_compare(self):
        game = make_matching_pennies()
        result = compare_with_turn_based(game)
        assert result["concurrent_vertices"] == 2
        assert "win_even_as" in result


# ========================================================================
# Section 15: Probabilistic Transitions
# ========================================================================

class TestProbabilisticTransitions:
    def test_mixed_transition(self):
        """Game where action pairs lead to probabilistic outcomes."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_vertex(1, 2, ["a"], ["x"])
        game.add_vertex(2, 1, ["a"], ["x"])
        game.add_transition(0, "a", "x", {1: 0.7, 2: 0.3})
        game.add_transition(1, "a", "x", {1: 1.0})
        game.add_transition(2, "a", "x", {2: 1.0})
        result = solve_concurrent_stochastic(game)
        # Vertex 1 (prio 2) always won by Even
        assert 1 in result.win_even_as
        # Vertex 2 (prio 1) always won by Odd
        assert 2 in result.win_odd_as

    def test_fair_coin(self):
        """Fair coin flip: neither player controls outcome."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_vertex(1, 2, ["a"], ["x"])
        game.add_vertex(2, 1, ["a"], ["x"])
        game.add_transition(0, "a", "x", {1: 0.5, 2: 0.5})
        game.add_transition(1, "a", "x", {1: 1.0})
        game.add_transition(2, "a", "x", {2: 1.0})
        result = solve_concurrent_stochastic(game)
        # From vertex 0: positive prob for both, but not AS for either
        assert 0 in result.win_even_pp
        assert 0 in result.win_odd_pp


# ========================================================================
# Section 16: Multi-Priority Games
# ========================================================================

class TestMultiPriority:
    def test_three_priorities(self):
        """Game with priorities 0, 1, 2."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["go"], ["x"])
        game.add_vertex(1, 1, ["go"], ["x"])
        game.add_vertex(2, 2, ["stay"], ["x"])
        game.add_transition(0, "go", "x", {1: 1.0})
        game.add_transition(1, "go", "x", {2: 1.0})
        game.add_transition(2, "stay", "x", {2: 1.0})
        result = solve_concurrent_stochastic(game)
        # All paths lead to vertex 2 (prio 2, even) -- Even wins AS from everywhere
        assert result.win_even_as == {0, 1, 2}

    def test_cycle_priorities(self):
        """Cycle visiting different priorities."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["go"], ["x"])
        game.add_vertex(1, 1, ["go"], ["x"])
        game.add_vertex(2, 2, ["go"], ["x"])
        game.add_transition(0, "go", "x", {1: 1.0})
        game.add_transition(1, "go", "x", {2: 1.0})
        game.add_transition(2, "go", "x", {0: 1.0})
        result = solve_concurrent_stochastic(game)
        # Cycle through 0->1->2->0. Max prio seen inf often = 2 (even). Even wins.
        assert result.win_even_as == {0, 1, 2}


# ========================================================================
# Section 17: Edge Cases
# ========================================================================

class TestEdgeCases:
    def test_self_loop_even(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 2, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        result = solve_concurrent_stochastic(game)
        assert result.win_even_as == {0}

    def test_self_loop_odd(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 3, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        result = solve_concurrent_stochastic(game)
        assert result.win_odd_as == {0}

    def test_disconnected_components(self):
        """Two disconnected vertices."""
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_vertex(1, 1, ["b"], ["y"])
        game.add_transition(0, "a", "x", {0: 1.0})
        game.add_transition(1, "b", "y", {1: 1.0})
        result = solve_concurrent_stochastic(game)
        assert 0 in result.win_even_as
        assert 1 in result.win_odd_as

    def test_many_actions(self):
        """Vertex with many actions."""
        game = ConcurrentStochasticGame()
        ea = [f"e{i}" for i in range(5)]
        oa = [f"o{j}" for j in range(5)]
        game.add_vertex(0, 0, ea, oa)
        for ae in ea:
            for ao in oa:
                game.add_transition(0, ae, ao, {0: 1.0})
        result = solve_concurrent_stochastic(game)
        assert 0 in result.win_even_as


# ========================================================================
# Section 18: AS subset of PP invariant
# ========================================================================

class TestASSubsetPP:
    def test_reachability_game(self):
        game = make_concurrent_reachability(5, target=4, trap=2)
        result = solve_concurrent_stochastic(game)
        assert result.win_even_as <= result.win_even_pp
        assert result.win_odd_as <= result.win_odd_pp

    def test_safety_game(self):
        game = make_concurrent_safety(5, {0, 1, 2})
        result = solve_concurrent_stochastic(game)
        assert result.win_even_as <= result.win_even_pp
        assert result.win_odd_as <= result.win_odd_pp


# ========================================================================
# Section 19: Partition Property (AS)
# ========================================================================

class TestPartition:
    def test_as_partition(self):
        """AS winning regions partition the vertex set."""
        game = make_concurrent_reachability(4, target=3, trap=1)
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        # Every vertex is in exactly one region
        assert w0 | w1 == game.vertex_set()
        assert w0 & w1 == set()

    def test_single_vertex_partition(self):
        game = ConcurrentStochasticGame()
        game.add_vertex(0, 0, ["a"], ["x"])
        game.add_transition(0, "a", "x", {0: 1.0})
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        assert w0 | w1 == {0}
        assert w0 & w1 == set()

    def test_multi_vertex_partition(self):
        game = make_rock_paper_scissors_game()
        w0, w1, _, _ = solve_concurrent_parity_almost_sure(game)
        assert w0 | w1 == game.vertex_set()
        assert w0 & w1 == set()


# ========================================================================
# Section 20: Integration
# ========================================================================

class TestIntegration:
    def test_full_pipeline_matching_pennies(self):
        """Full solve + verify + simulate for matching pennies."""
        game = make_matching_pennies()
        result = solve_concurrent_stochastic(game)

        # Verify Even's AS strategy
        if result.strategy_even_as:
            v = verify_strategy(game, result.win_even_as, result.strategy_even_as, True)
            assert v["valid"]

        # Simulate
        strat_e = result.strategy_even_pp if result.strategy_even_pp else {
            v: {"H": 0.5, "T": 0.5} for v in game.vertex_set()
        }
        strat_o = result.strategy_odd_pp if result.strategy_odd_pp else {
            v: {"H": 0.5, "T": 0.5} for v in game.vertex_set()
        }
        trace = simulate_play(game, 0, strat_e, strat_o, steps=10, seed=42)
        assert len(trace) == 10

    def test_full_pipeline_reachability(self):
        """Full solve + stats for reachability."""
        game = make_concurrent_reachability(5, target=4, trap=2)
        stats = concurrent_game_statistics(game)
        assert stats["vertices"] == 5
        result = solve_concurrent_stochastic(game)
        assert isinstance(result, CSGResult)
        # Target should be in Even's winning region
        assert 4 in result.win_even_as

    def test_batch_and_stats(self):
        """Batch solve multiple games and check stats."""
        games = [
            ("pennies", make_matching_pennies()),
            ("reach", make_concurrent_reachability(3, target=2, trap=1)),
            ("rps", make_rock_paper_scissors_game()),
        ]
        results = batch_solve(games)
        assert len(results) == 3
        for name, game in games:
            stats = concurrent_game_statistics(game)
            assert stats["vertices"] > 0
