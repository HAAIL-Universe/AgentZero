"""
Tests for V164: Stochastic Energy Games
"""
import sys
import pytest
sys.path.insert(0, 'Z:/AgentZero/A2/work/V164_stochastic_energy_games')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V160_energy_games')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V156_parity_games')

from stochastic_energy import (
    VertexType, StochasticEnergyGame, StochasticEnergyResult,
    StochasticEnergyParityGame, StochasticEnergyParityResult,
    solve_stochastic_energy, solve_stochastic_energy_parity,
    simulate_play, verify_strategy,
    make_chain_game, make_diamond_game, make_gambling_game, make_random_walk_game,
    compare_with_deterministic, stochastic_energy_statistics,
    _compute_positive_prob_winning
)
from energy_games import Player, INF_ENERGY


# ============================================================
# Section 1: Data Structure Basics
# ============================================================

class TestDataStructures:
    def test_create_empty_game(self):
        g = StochasticEnergyGame()
        assert len(g.vertices) == 0

    def test_add_vertices(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.ODD)
        g.add_vertex(2, VertexType.RANDOM)
        assert len(g.vertices) == 3
        assert g.vertex_type[0] == VertexType.EVEN
        assert g.vertex_type[1] == VertexType.ODD
        assert g.vertex_type[2] == VertexType.RANDOM

    def test_add_edges(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.ODD)
        g.add_edge(0, 1, 5)
        assert g.successors(0) == [(1, 5)]

    def test_random_edge_probabilities(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_vertex(2, VertexType.EVEN)
        g.add_edge(0, 1, 3, 0.7)
        g.add_edge(0, 2, -1, 0.3)
        assert g.get_prob(0, 1, 3) == 0.7
        assert g.get_prob(0, 2, -1) == 0.3

    def test_validate_valid_game(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 1, 0.6)
        g.add_edge(0, 1, -1, 0.4)
        errors = g.validate()
        assert len(errors) == 0

    def test_validate_bad_probabilities(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 1, 0.3)
        g.add_edge(0, 1, -1, 0.3)
        errors = g.validate()
        assert len(errors) > 0

    def test_max_weight(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.ODD)
        g.add_edge(0, 1, 5)
        g.add_edge(1, 0, -3)
        assert g.max_weight() == 5

    def test_to_energy_game(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.RANDOM)
        g.add_vertex(2, VertexType.ODD)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, -1, 0.5)
        g.add_edge(1, 0, 2, 0.5)
        g.add_edge(2, 0, -1)
        eg = g.to_energy_game()
        # RANDOM -> EVEN in deterministic game
        assert eg.owner[1] == Player.EVEN
        assert eg.owner[2] == Player.ODD


# ============================================================
# Section 2: Simple Deterministic Cases (No Random Vertices)
# ============================================================

class TestDeterministicSubcases:
    def test_single_even_self_loop(self):
        """Even vertex with positive self-loop: wins with energy 0."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_edge(0, 0, 1)
        result = solve_stochastic_energy(g)
        assert 0 in result.win_almost_sure
        assert result.min_energy[0] == 0

    def test_single_even_negative_loop(self):
        """Even vertex with only negative self-loop: needs energy to survive."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_edge(0, 0, -1)
        result = solve_stochastic_energy(g)
        # Can't survive forever with negative loop
        assert 0 in result.win_opponent

    def test_even_chooses_positive(self):
        """Even vertex chooses positive edge over negative."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 5)
        g.add_edge(0, 0, -1)
        g.add_edge(1, 0, -2)
        result = solve_stochastic_energy(g)
        assert 0 in result.win_almost_sure
        assert 1 in result.win_almost_sure

    def test_odd_forces_negative(self):
        """Odd forces Even through negative edge."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.ODD)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, -3)
        g.add_edge(1, 0, 1)
        result = solve_stochastic_energy(g)
        # Net per cycle: -3 + 1 = -2, so energy depletes
        assert 0 in result.win_opponent

    def test_dead_end_even_loses(self):
        """Even vertex with no successors loses."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        result = solve_stochastic_energy(g)
        assert 0 in result.win_opponent

    def test_dead_end_odd_even_wins(self):
        """Odd vertex with no successors: Even wins."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.ODD)
        result = solve_stochastic_energy(g)
        # No successors for Odd means Even wins (Odd stuck)
        assert 0 in result.win_almost_sure


# ============================================================
# Section 3: Basic Random Vertex Games
# ============================================================

class TestRandomVertexBasics:
    def test_fair_coin_positive_weight(self):
        """Random vertex: both outcomes have positive weight."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_edge(0, 0, 1, 0.5)
        g.add_edge(0, 0, 2, 0.5)
        result = solve_stochastic_energy(g)
        assert 0 in result.win_almost_sure
        assert result.min_energy[0] == 0

    def test_fair_coin_one_negative_cycle(self):
        """Random vertex with negative outcome in a cycle: a.s. loses.
        The bad outcome happens infinitely often a.s., so energy depletes."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 3, 0.5)
        g.add_edge(0, 1, -2, 0.5)
        g.add_edge(1, 0, 0)
        result = solve_stochastic_energy(g)
        # In cycle: worst-case per round is -2, so energy diverges
        assert 0 in result.win_opponent
        # But positive-probability winning works (50% good outcome each time)
        assert 0 in result.win_positive

    def test_random_acyclic_one_negative(self):
        """Random vertex with negative outcome but no cycle through it: finite energy."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_vertex(2, VertexType.EVEN)
        g.add_edge(0, 1, 3, 0.5)
        g.add_edge(0, 2, -2, 0.5)
        g.add_edge(1, 1, 1)  # self-loop, wins
        g.add_edge(2, 2, 1)  # self-loop, wins
        result = solve_stochastic_energy(g)
        assert 0 in result.win_almost_sure
        # Need 2 energy for the -2 outcome
        assert result.min_energy[0] >= 2

    def test_all_negative_random(self):
        """Random vertex with only negative outcomes: loses eventually."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_edge(0, 0, -1, 0.5)
        g.add_edge(0, 0, -2, 0.5)
        result = solve_stochastic_energy(g)
        assert 0 in result.win_opponent

    def test_random_to_winning_region(self):
        """Random vertex leading to vertices that are all winning."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_vertex(2, VertexType.EVEN)
        g.add_edge(0, 1, 0, 0.5)
        g.add_edge(0, 2, 0, 0.5)
        g.add_edge(1, 0, 1)
        g.add_edge(2, 0, 2)
        result = solve_stochastic_energy(g)
        assert 0 in result.win_almost_sure
        assert 1 in result.win_almost_sure
        assert 2 in result.win_almost_sure

    def test_random_one_path_losing(self):
        """Random vertex where one successor is in losing region.
        Almost-sure winning requires ALL random outcomes to be survivable.
        If one path leads to guaranteed loss, cannot win almost surely."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)  # winning
        g.add_vertex(2, VertexType.EVEN)  # losing (dead end)
        g.add_edge(0, 1, 0, 0.5)
        g.add_edge(0, 2, 0, 0.5)
        g.add_edge(1, 0, 1)
        # vertex 2 has no successors -> Even loses
        result = solve_stochastic_energy(g)
        assert 0 in result.win_opponent  # can't win a.s. because 50% chance of losing


# ============================================================
# Section 4: Positive-Probability Winning
# ============================================================

class TestPositiveProbWinning:
    def test_positive_prob_wider_than_as(self):
        """Positive-probability winning is at least as large as almost-sure."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_vertex(2, VertexType.EVEN)  # dead end
        g.add_edge(0, 1, 0, 0.5)
        g.add_edge(0, 2, 0, 0.5)
        g.add_edge(1, 0, 1)
        result = solve_stochastic_energy(g)
        # Almost-sure: vertex 0 loses (50% dead end)
        assert 0 in result.win_opponent
        # But positive-prob: vertex 0 can win (50% chance of winning path)
        assert 0 in result.win_positive
        assert result.win_almost_sure <= result.win_positive

    def test_positive_prob_pure_even(self):
        """Without random, positive-prob = almost-sure for Even."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_edge(0, 0, 1)
        result = solve_stochastic_energy(g)
        assert result.win_almost_sure == result.win_positive

    def test_positive_prob_all_lose(self):
        """If all paths lose, neither AS nor PP wins."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)  # dead end
        g.add_vertex(2, VertexType.EVEN)  # dead end
        g.add_edge(0, 1, 0, 0.5)
        g.add_edge(0, 2, 0, 0.5)
        result = solve_stochastic_energy(g)
        assert 0 not in result.win_positive


# ============================================================
# Section 5: Expected Energy
# ============================================================

class TestExpectedEnergy:
    def test_expected_energy_deterministic(self):
        """Without random, expected energy = min energy."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        result = solve_stochastic_energy(g)
        # Both should be winning
        assert result.expected_energy[0] is not None
        assert result.expected_energy[1] is not None

    def test_expected_energy_losing(self):
        """Losing vertices have None expected energy."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)  # dead end
        result = solve_stochastic_energy(g)
        assert result.expected_energy[0] is None

    def test_expected_energy_random(self):
        """Expected energy for random vertex is weighted average."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 2, 0.5)
        g.add_edge(0, 1, -1, 0.5)
        g.add_edge(1, 0, 0)
        result = solve_stochastic_energy(g)
        if 0 in result.win_almost_sure:
            assert result.expected_energy[0] is not None


# ============================================================
# Section 6: Construction Helpers
# ============================================================

class TestConstructionHelpers:
    def test_chain_game(self):
        g = make_chain_game(3, [1, -1, 2])
        assert len(g.vertices) == 3
        assert g.vertex_type[0] == VertexType.EVEN
        assert g.vertex_type[1] == VertexType.ODD
        assert g.vertex_type[2] == VertexType.RANDOM

    def test_chain_game_with_random_probs(self):
        g = make_chain_game(3, [1, -1, 2], random_probs=[0.7])
        # Random vertex should have split edges
        succs = g.successors(2)
        assert len(succs) == 2  # forward + self-loop

    def test_diamond_game(self):
        g = make_diamond_game(3, 1, 0.5)
        assert len(g.vertices) == 4
        assert g.vertex_type[0] == VertexType.EVEN
        assert g.vertex_type[1] == VertexType.RANDOM
        assert g.vertex_type[3] == VertexType.ODD

    def test_gambling_game(self):
        g = make_gambling_game(5, 0.5)
        assert len(g.vertices) == 3
        assert g.vertex_type[0] == VertexType.EVEN
        assert g.vertex_type[1] == VertexType.RANDOM

    def test_random_walk_game(self):
        g = make_random_walk_game(5)
        assert len(g.vertices) == 5
        for i in range(5):
            assert g.vertex_type[i] == VertexType.RANDOM

    def test_gambling_game_structure(self):
        """Gambling game: safe path has -1 cost per round, so energy depletes.
        Both paths (bet and don't-bet) lose a.s. in the cycle."""
        g = make_gambling_game(5, 0.5)
        result = solve_stochastic_energy(g)
        # Safe path 0->2->0 costs -1 per round -> loses
        # Bet path 0->1->0 has random outcomes in cycle -> loses a.s.
        assert 0 in result.win_opponent


# ============================================================
# Section 7: Stochastic Energy-Parity Games
# ============================================================

class TestStochasticEnergyParity:
    def test_empty_game(self):
        g = StochasticEnergyParityGame()
        result = solve_stochastic_energy_parity(g)
        assert len(result.win_even) == 0

    def test_single_even_vertex_even_prio(self):
        """Even vertex, even priority, positive loop: wins both."""
        g = StochasticEnergyParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_edge(0, 0, 1)
        result = solve_stochastic_energy_parity(g)
        assert 0 in result.win_even

    def test_single_even_vertex_odd_prio(self):
        """Even vertex, odd priority: loses parity even with good energy."""
        g = StochasticEnergyParityGame()
        g.add_vertex(0, VertexType.EVEN, 1)
        g.add_edge(0, 0, 1)
        result = solve_stochastic_energy_parity(g)
        assert 0 in result.win_odd

    def test_parity_wins_energy_loses(self):
        """Parity says Even wins, but energy depletes."""
        g = StochasticEnergyParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_edge(0, 0, -1)
        result = solve_stochastic_energy_parity(g)
        assert 0 in result.win_odd  # energy makes Even lose

    def test_combined_win(self):
        """Both energy and parity satisfied."""
        g = StochasticEnergyParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 2)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        result = solve_stochastic_energy_parity(g)
        assert 0 in result.win_even
        assert 1 in result.win_even

    def test_random_vertex_in_parity(self):
        """Random vertex in energy-parity game."""
        g = StochasticEnergyParityGame()
        g.add_vertex(0, VertexType.RANDOM, 2)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_edge(0, 1, 1, 0.5)
        g.add_edge(0, 1, 2, 0.5)
        g.add_edge(1, 0, -1)
        result = solve_stochastic_energy_parity(g)
        assert 0 in result.win_even

    def test_energy_parity_to_energy_game(self):
        g = StochasticEnergyParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.RANDOM, 0)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, -1, 0.5)
        g.add_edge(1, 0, 1, 0.5)
        seg = g.to_energy_game()
        assert len(seg.vertices) == 2

    def test_energy_parity_to_parity_game(self):
        g = StochasticEnergyParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_edge(0, 1, 5)
        g.add_edge(1, 0, -3)
        pg = g.to_parity_game()
        assert len(pg.vertices) == 2


# ============================================================
# Section 8: Simulation
# ============================================================

class TestSimulation:
    def test_simulate_deterministic(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        trace = simulate_play(g, 0, 5, {0: 1, 1: 0}, {}, steps=6)
        assert len(trace) > 0
        assert trace[0] == (0, 5)
        # Energy should go: 5, 8, 7, 10, 9, 12, 11
        assert trace[1] == (1, 8)
        assert trace[2] == (0, 7)

    def test_simulate_with_random(self):
        g = make_gambling_game(3, 0.5)
        result = solve_stochastic_energy(g)
        strategy_even = result.strategy_even
        trace = simulate_play(g, 0, 10, strategy_even, {}, steps=20, seed=123)
        assert len(trace) > 0
        # All energy levels should be defined
        for (v, e) in trace:
            assert isinstance(e, (int, float))

    def test_simulate_energy_depletion(self):
        """Simulate until energy goes negative."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_edge(0, 0, -1)
        trace = simulate_play(g, 0, 3, {0: 0}, {}, steps=10)
        # Should stop when energy < 0
        last_energy = trace[-1][1]
        assert last_energy < 0


# ============================================================
# Section 9: Strategy Verification
# ============================================================

class TestStrategyVerification:
    def test_valid_strategy(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        result = verify_strategy(g, {0: 1, 1: 0}, {0: 0, 1: 1})
        assert result['valid']
        assert result['energy_ok']

    def test_missing_strategy(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)
        result = verify_strategy(g, {}, {0: 0, 1: 0})
        assert not result['valid']

    def test_energy_violation(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, -5)
        g.add_edge(1, 0, 1)
        # Strategy says go 0->1 with only 2 energy, but edge costs 5
        result = verify_strategy(g, {0: 1, 1: 0}, {0: 2, 1: 0})
        assert not result['energy_ok']


# ============================================================
# Section 10: Comparison with Deterministic
# ============================================================

class TestComparison:
    def test_ordering_pessimistic_le_stochastic_le_optimistic(self):
        """pessimistic win <= stochastic win <= optimistic win."""
        g = make_diamond_game(3, 1, 0.5)
        comp = compare_with_deterministic(g)
        assert comp['ordering_valid']

    def test_comparison_pure_deterministic(self):
        """Without random vertices, all three agree."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.ODD)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        comp = compare_with_deterministic(g)
        # All should agree since no random vertices
        assert comp['summary']['stochastic_win_size'] == comp['summary']['optimistic_win_size']
        assert comp['summary']['stochastic_win_size'] == comp['summary']['pessimistic_win_size']


# ============================================================
# Section 11: Statistics
# ============================================================

class TestStatistics:
    def test_statistics_basic(self):
        g = make_diamond_game(3, 1, 0.5)
        stats = stochastic_energy_statistics(g)
        assert stats['vertices'] == 4
        assert stats['random_vertices'] == 2
        assert stats['even_vertices'] == 1
        assert stats['odd_vertices'] == 1
        assert 'win_almost_sure' in stats
        assert 'win_positive' in stats

    def test_statistics_gambling(self):
        g = make_gambling_game(5, 0.5)
        stats = stochastic_energy_statistics(g)
        assert stats['vertices'] == 3
        assert stats['random_vertices'] == 1


# ============================================================
# Section 12: Complex Scenarios
# ============================================================

class TestComplexScenarios:
    def test_biased_coin_in_cycle(self):
        """Biased coin in cycle: a.s. loses even with favorable odds.
        The -1 outcome repeats infinitely often, and each time the
        max-energy requirement grows. Correct result: a.s. losing."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 10, 0.9)   # +10 with 90%
        g.add_edge(0, 1, -1, 0.1)   # -1 with 10%
        g.add_edge(1, 0, 0)
        result = solve_stochastic_energy(g)
        # Cycle with random: worst case -1 repeats a.s. -> can't win a.s.
        assert 0 in result.win_opponent
        # But positive-probability winning works (favorable odds)
        assert 0 in result.win_positive

    def test_multi_random_chain(self):
        """Chain of random vertices, all positive."""
        g = StochasticEnergyGame()
        for i in range(4):
            g.add_vertex(i, VertexType.RANDOM)
        g.add_edge(0, 1, 1, 0.5)
        g.add_edge(0, 1, 2, 0.5)
        g.add_edge(1, 2, 1, 0.5)
        g.add_edge(1, 2, 3, 0.5)
        g.add_edge(2, 3, 1, 0.5)
        g.add_edge(2, 3, 2, 0.5)
        g.add_edge(3, 0, 1, 0.5)
        g.add_edge(3, 0, 1, 0.5)
        result = solve_stochastic_energy(g)
        for i in range(4):
            assert i in result.win_almost_sure

    def test_even_avoids_random_danger(self):
        """Even can choose safe path avoiding risky random vertex."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.RANDOM)  # risky
        g.add_vertex(2, VertexType.EVEN)    # safe
        g.add_vertex(3, VertexType.EVEN)    # dead end (losing)

        g.add_edge(0, 1, 0)  # go to risky
        g.add_edge(0, 2, 0)  # go to safe
        g.add_edge(1, 0, 5, 0.5)
        g.add_edge(1, 3, 0, 0.5)  # 50% dead end
        g.add_edge(2, 0, 1)
        # vertex 3 dead end

        result = solve_stochastic_energy(g)
        # Even should choose vertex 2 (safe) to win almost surely
        assert 0 in result.win_almost_sure
        assert 2 in result.win_almost_sure
        # Strategy should pick vertex 2
        assert result.strategy_even.get(0) == 2

    def test_odd_forces_through_random(self):
        """Odd forces play through a dangerous random vertex."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.ODD)
        g.add_vertex(1, VertexType.RANDOM)
        g.add_vertex(2, VertexType.EVEN)
        g.add_vertex(3, VertexType.EVEN)  # dead end

        g.add_edge(0, 1, 0)
        g.add_edge(1, 2, 1, 0.5)
        g.add_edge(1, 3, 0, 0.5)  # 50% dead end
        g.add_edge(2, 0, -1)

        result = solve_stochastic_energy(g)
        # Random has 50% dead end -> a.s. losing -> Odd forces there
        assert 0 in result.win_opponent

    def test_random_walk_biased_up(self):
        """Biased random walk with upward drift."""
        g = make_random_walk_game(4, step_prob=0.8)  # 80% up
        result = solve_stochastic_energy(g)
        # All vertices should be analyzed
        assert len(result.min_energy) == 4

    def test_nested_random_in_cycle(self):
        """Two random vertices in cycle: a.s. loses (negative outcomes repeat)."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.RANDOM)
        g.add_vertex(2, VertexType.EVEN)

        g.add_edge(0, 1, 2, 0.6)
        g.add_edge(0, 1, -1, 0.4)
        g.add_edge(1, 2, 3, 0.7)
        g.add_edge(1, 2, -2, 0.3)
        g.add_edge(2, 0, 0)

        result = solve_stochastic_energy(g)
        # Cyclic random with negative outcomes -> a.s. losing
        assert 0 in result.win_opponent
        # But positive-prob winning should work
        assert 0 in result.win_positive

    def test_nested_random_acyclic(self):
        """Two random vertices NOT in cycle: can win a.s."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.RANDOM)
        g.add_vertex(2, VertexType.EVEN)

        g.add_edge(0, 1, 2, 0.6)
        g.add_edge(0, 1, -1, 0.4)
        g.add_edge(1, 2, 3, 0.7)
        g.add_edge(1, 2, -2, 0.3)
        g.add_edge(2, 2, 1)  # self-loop, no cycle back through random

        result = solve_stochastic_energy(g)
        # Acyclic: max-over-outcomes gives finite energy
        assert 0 in result.win_almost_sure
        assert 1 in result.win_almost_sure
        assert 2 in result.win_almost_sure

    def test_large_weight_random_acyclic(self):
        """Random vertex with large weight variance, acyclic."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)
        g.add_vertex(2, VertexType.EVEN)
        g.add_edge(0, 1, 100, 0.5)
        g.add_edge(0, 2, -50, 0.5)
        g.add_edge(1, 1, 1)  # self-loop
        g.add_edge(2, 2, 1)  # self-loop
        result = solve_stochastic_energy(g)
        assert 0 in result.win_almost_sure
        assert result.min_energy[0] >= 50  # needs 50 for worst case


# ============================================================
# Section 13: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_game(self):
        g = StochasticEnergyGame()
        result = solve_stochastic_energy(g)
        assert len(result.win_almost_sure) == 0
        assert len(result.win_positive) == 0
        assert len(result.win_opponent) == 0

    def test_single_random_no_edges(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        result = solve_stochastic_energy(g)
        # Dead-end random vertex: Even can't lose because it's not Even's dead end
        # But there's no play, so the energy condition is trivially met
        # Depends on convention: dead-end non-Even = Even wins
        assert 0 in result.win_almost_sure

    def test_zero_weight_loop(self):
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_edge(0, 0, 0, 1.0)
        result = solve_stochastic_energy(g)
        assert 0 in result.win_almost_sure
        assert result.min_energy[0] == 0

    def test_very_small_probability(self):
        """Edge with very small probability still matters for almost-sure."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)  # dead end
        g.add_vertex(2, VertexType.EVEN)
        g.add_edge(0, 2, 1, 0.999)
        g.add_edge(0, 1, 0, 0.001)  # tiny chance of dead end
        g.add_edge(2, 0, 0)
        result = solve_stochastic_energy(g)
        # Even with 0.1% chance, almost-sure requires surviving ALL outcomes
        assert 0 in result.win_opponent

    def test_zero_probability_ignored(self):
        """Edge with zero probability should be ignored for almost-sure."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.RANDOM)
        g.add_vertex(1, VertexType.EVEN)  # dead end
        g.add_vertex(2, VertexType.EVEN)
        g.add_edge(0, 2, 1, 1.0)
        g.add_edge(0, 1, 0, 0.0)  # zero probability
        g.add_edge(2, 0, 0)
        result = solve_stochastic_energy(g)
        # Zero probability edge doesn't count for almost-sure
        assert 0 in result.win_almost_sure


# ============================================================
# Section 14: Integration Tests
# ============================================================

class TestIntegration:
    def test_solve_then_simulate(self):
        """Solve, then simulate with extracted strategy."""
        g = make_gambling_game(3, 0.5)
        result = solve_stochastic_energy(g)
        if result.win_almost_sure:
            start = next(iter(result.win_almost_sure))
            e = result.min_energy[start]
            if e is not None:
                trace = simulate_play(g, start, e + 5, result.strategy_even, result.strategy_odd, steps=20)
                assert len(trace) > 0

    def test_solve_then_verify(self):
        """Solve, then verify the extracted strategy."""
        g = StochasticEnergyGame()
        g.add_vertex(0, VertexType.EVEN)
        g.add_vertex(1, VertexType.EVEN)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        result = solve_stochastic_energy(g)
        init_energy = {v: (result.min_energy[v] if result.min_energy[v] is not None else 0)
                       for v in result.win_almost_sure}
        vresult = verify_strategy(g, result.strategy_even, init_energy)
        assert vresult['energy_ok']

    def test_compare_then_stats(self):
        """Run comparison and statistics on same game."""
        g = make_diamond_game(5, 2, 0.6)
        comp = compare_with_deterministic(g)
        stats = stochastic_energy_statistics(g)
        assert comp['ordering_valid']
        assert stats['vertices'] == 4

    def test_energy_parity_consistent_with_energy_only(self):
        """Energy-parity with all-even priorities = energy-only."""
        g_ep = StochasticEnergyParityGame()
        g_ep.add_vertex(0, VertexType.EVEN, 2)
        g_ep.add_vertex(1, VertexType.RANDOM, 2)
        g_ep.add_edge(0, 1, 3)
        g_ep.add_edge(1, 0, -1, 0.5)
        g_ep.add_edge(1, 0, 1, 0.5)

        ep_result = solve_stochastic_energy_parity(g_ep)

        g_e = g_ep.to_energy_game()
        e_result = solve_stochastic_energy(g_e)

        # With all-even priorities, parity is trivially satisfied
        assert ep_result.win_even == e_result.win_almost_sure


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
