"""Tests for V160: Energy Games."""

import pytest
import sys
sys.path.insert(0, 'Z:/AgentZero/A2/work/V160_energy_games')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V156_parity_games')

from energy_games import (
    EnergyGame, EnergyResult, EnergyParityGame, EnergyParityResult,
    MeanPayoffResult, Player,
    solve_energy, solve_fixed_energy, solve_energy_parity,
    solve_mean_payoff, mean_payoff_threshold,
    simulate_play, verify_energy_strategy,
    make_simple_energy_game, make_energy_parity_game,
    make_chain_energy_game, make_charging_game, make_choice_game,
    compare_energy_vs_parity, energy_game_statistics,
    parity_to_energy, energy_to_mean_payoff,
)
from parity_games import ParityGame, zielonka


# =====================================================================
# Section 1: Basic Energy Game Construction
# =====================================================================

class TestConstruction:
    def test_empty_game(self):
        g = EnergyGame()
        assert len(g.vertices) == 0

    def test_add_vertex(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        assert len(g.vertices) == 2
        assert g.owner[0] == Player.EVEN
        assert g.owner[1] == Player.ODD

    def test_add_edge(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 5)
        g.add_edge(1, 0, -3)
        assert len(g.successors(0)) == 1
        assert g.successors(0)[0] == (1, 5)
        assert g.successors(1)[0] == (0, -3)

    def test_predecessors(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 2)
        preds = g.predecessors(1)
        assert len(preds) == 1
        assert preds[0] == (0, 2)

    def test_max_weight(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 5)
        g.add_edge(1, 0, -7)
        assert g.max_weight() == 7

    def test_total_weight_bound(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 3)
        assert g.total_weight_bound() == 2 * 3  # n * W

    def test_make_simple(self):
        g = make_simple_energy_game(
            edges=[(0, 1, 2), (1, 0, -1)],
            owners={0: Player.EVEN, 1: Player.ODD}
        )
        assert len(g.vertices) == 2
        assert g.owner[0] == Player.EVEN


# =====================================================================
# Section 2: Trivial Energy Games
# =====================================================================

class TestTrivialGames:
    def test_empty_game(self):
        g = EnergyGame()
        r = solve_energy(g)
        assert len(r.win_energy) == 0
        assert len(r.win_opponent) == 0

    def test_single_vertex_self_loop_positive(self):
        """Even vertex with positive self-loop: wins with energy 0."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 1)
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert r.min_energy[0] == 0

    def test_single_vertex_self_loop_negative(self):
        """Even vertex with only negative self-loop: loses (energy always depletes)."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, -1)
        r = solve_energy(g)
        assert 0 in r.win_opponent
        assert r.min_energy[0] is None

    def test_single_vertex_zero_weight(self):
        """Self-loop with weight 0: wins with energy 0."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 0)
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert r.min_energy[0] == 0

    def test_even_dead_end(self):
        """Even vertex with no successors: Even loses."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        r = solve_energy(g)
        assert 0 in r.win_opponent

    def test_odd_dead_end(self):
        """Odd vertex with no successors: Even wins (Odd stuck)."""
        g = EnergyGame()
        g.add_vertex(0, Player.ODD)
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert r.min_energy[0] == 0


# =====================================================================
# Section 3: Two-Vertex Energy Games
# =====================================================================

class TestTwoVertexGames:
    def test_positive_cycle(self):
        """Two vertices in a positive cycle: both win with energy 0."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)
        r = solve_energy(g)
        assert r.win_energy == {0, 1}
        assert r.min_energy[0] == 0
        assert r.min_energy[1] == 0

    def test_negative_cycle(self):
        """Two vertices in a negative cycle: Even always loses."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -1)
        g.add_edge(1, 0, -1)
        r = solve_energy(g)
        assert r.win_opponent == {0, 1}

    def test_balanced_cycle(self):
        """Cycle with total weight 0: wins with sufficient initial energy."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -3)
        g.add_edge(1, 0, 3)
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert 1 in r.win_energy
        assert r.min_energy[0] == 3  # need 3 to survive the -3 edge

    def test_even_choice_good(self):
        """Even can choose positive self-loop over negative edge."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 0, 1)   # positive self-loop
        g.add_edge(0, 1, -5)  # bad edge
        g.add_edge(1, 0, 0)
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert r.min_energy[0] == 0  # just use the self-loop

    def test_odd_forces_negative(self):
        """Odd forces Even through negative edge."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 2)
        g.add_edge(1, 0, -5)  # Odd's only move: costs 5
        r = solve_energy(g)
        # Cycle: +2 -5 = -3 net. Energy always depletes.
        assert r.win_opponent == {0, 1}

    def test_odd_benign(self):
        """Odd has only positive edges: Even wins easily."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -1)
        g.add_edge(1, 0, 3)
        r = solve_energy(g)
        # Cycle: -1 +3 = +2 net. Wins with enough to survive the -1.
        assert r.win_energy == {0, 1}
        assert r.min_energy[0] == 1  # need 1 for the -1 edge


# =====================================================================
# Section 4: Multi-Vertex Games with Strategy
# =====================================================================

class TestStrategyGames:
    def test_even_avoids_trap(self):
        """Even can choose safe path, avoiding opponent's trap."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)  # choice
        g.add_vertex(1, Player.ODD)   # safe
        g.add_vertex(2, Player.ODD)   # trap

        g.add_edge(0, 1, 0)   # go safe
        g.add_edge(0, 2, 0)   # go trap
        g.add_edge(1, 0, 1)   # safe loop: +1
        g.add_edge(2, 2, -1)  # trap: -1 forever

        r = solve_energy(g)
        assert 0 in r.win_energy
        assert 1 in r.win_energy
        assert 2 in r.win_opponent
        # Even's strategy at 0 should go to 1 (safe)
        assert r.strategy_energy.get(0) == 1

    def test_odd_forces_best_path(self):
        """Odd picks the worst path for Even, Even must survive it."""
        g = EnergyGame()
        g.add_vertex(0, Player.ODD)   # Odd chooses
        g.add_vertex(1, Player.EVEN)  # mild cost
        g.add_vertex(2, Player.EVEN)  # heavy cost

        g.add_edge(0, 1, -1)  # mild
        g.add_edge(0, 2, -5)  # heavy
        g.add_edge(1, 0, 3)   # recover
        g.add_edge(2, 0, 7)   # recover more

        r = solve_energy(g)
        # Odd will pick whichever is worse for Even
        # Path 0->1->0: net = -1+3 = +2 (good for Even)
        # Path 0->2->0: net = -5+7 = +2 (also good for Even)
        # But from 0, Odd picks -5 (costs more up front)
        assert 0 in r.win_energy
        assert r.min_energy[0] == 5  # need to survive -5

    def test_diamond_game(self):
        """Diamond: Even chooses top or bottom, Odd at end."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)  # start
        g.add_vertex(1, Player.ODD)   # top
        g.add_vertex(2, Player.ODD)   # bottom
        g.add_vertex(3, Player.EVEN)  # merge

        g.add_edge(0, 1, -2)  # top path costs 2
        g.add_edge(0, 2, -1)  # bottom path costs 1
        g.add_edge(1, 3, 5)   # top gives 5
        g.add_edge(2, 3, 0)   # bottom gives 0
        g.add_edge(3, 0, 0)   # loop back

        r = solve_energy(g)
        assert 0 in r.win_energy
        # Even should go bottom (cheaper): -1 + 0 + 0 = -1 per cycle
        # Or top: -2 + 5 + 0 = +3 per cycle -- actually top is better!
        # But we just need to check that Even wins

    def test_three_cycle_net_positive(self):
        """Three-vertex cycle with net positive weight."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_vertex(2, Player.EVEN)
        g.add_edge(0, 1, -2)
        g.add_edge(1, 2, -1)
        g.add_edge(2, 0, 5)
        r = solve_energy(g)
        # Net: -2-1+5 = +2. Even wins with enough initial energy.
        assert r.win_energy == {0, 1, 2}
        assert r.min_energy[0] == 3  # need to survive -2 then -1


# =====================================================================
# Section 5: Chain and Constructed Games
# =====================================================================

class TestConstructedGames:
    def test_chain_balanced(self):
        """Chain with alternating +1/-1: balanced cycle."""
        g = make_chain_energy_game(4)  # +1, -1, +1, -1
        r = solve_energy(g)
        assert len(r.win_energy) > 0  # at least some vertices win

    def test_chain_positive(self):
        """Chain with all positive weights."""
        g = make_chain_energy_game(3, [1, 1, 1])
        r = solve_energy(g)
        assert r.win_energy == {0, 1, 2}
        for v in g.vertices:
            assert r.min_energy[v] == 0

    def test_chain_negative(self):
        """Chain with all negative weights."""
        g = make_chain_energy_game(3, [-1, -1, -1])
        r = solve_energy(g)
        assert r.win_opponent == {0, 1, 2}

    def test_charging_game_basic(self):
        """Charging game: Even can charge at station."""
        g = make_charging_game(3, charge=5, drain=2)
        r = solve_energy(g)
        # Even at vertex 0 can self-loop to charge
        assert 0 in r.win_energy
        assert r.min_energy[0] == 0

    def test_choice_game(self):
        """Choice game: Even picks safe or risky."""
        g = make_choice_game()
        r = solve_energy(g)
        # Even should be able to win by choosing safe path (vertex 1, +1 loop)
        assert 0 in r.win_energy
        assert 1 in r.win_energy


# =====================================================================
# Section 6: Fixed Initial Energy
# =====================================================================

class TestFixedEnergy:
    def test_sufficient_energy(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -3)
        g.add_edge(1, 0, 5)
        result = solve_fixed_energy(g, initial_energy=3)
        assert result[0] is True
        assert result[1] is True

    def test_insufficient_energy(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -3)
        g.add_edge(1, 0, 5)
        result = solve_fixed_energy(g, initial_energy=2)
        assert result[0] is False  # need 3, have 2

    def test_zero_energy_positive_cycle(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 1)
        result = solve_fixed_energy(g, initial_energy=0)
        assert result[0] is True


# =====================================================================
# Section 7: Mean-Payoff Games
# =====================================================================

class TestMeanPayoff:
    def test_positive_cycle_mean(self):
        """Positive cycle has positive mean payoff."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 3)
        r = solve_mean_payoff(g)
        assert r.values[0] > 0
        assert 0 in r.win_nonneg

    def test_negative_cycle_mean(self):
        """Negative cycle has negative mean payoff."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, -2)
        r = solve_mean_payoff(g)
        assert r.values[0] < 0
        assert 0 in r.win_neg

    def test_zero_mean_payoff(self):
        """Balanced cycle has mean payoff ~0."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -3)
        r = solve_mean_payoff(g)
        # Mean payoff should be approximately 0
        assert abs(r.values[0]) < 1.0
        assert 0 in r.win_nonneg  # energy game with 0-sum cycle is winnable

    def test_even_choice_mean(self):
        """Even chooses between positive and negative cycles."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 2)   # positive self-loop
        r = solve_mean_payoff(g)
        assert r.values[0] > 0

    def test_mean_payoff_threshold(self):
        """Threshold: vertices where Even achieves mean >= threshold."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 5)
        winners = mean_payoff_threshold(g, 3)
        assert 0 in winners  # mean = 5 >= 3

    def test_mean_payoff_threshold_fail(self):
        """Threshold too high: no vertices qualify."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 1)
        winners = mean_payoff_threshold(g, 5)
        # mean = 1 < 5, shifted weight = 1-5 = -4, can't survive
        assert 0 not in winners


# =====================================================================
# Section 8: Simulation and Verification
# =====================================================================

class TestSimulation:
    def test_simulate_positive_loop(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 1)
        trace = simulate_play(g, 0, {0: 0}, {}, initial_energy=0, max_steps=5)
        assert len(trace) == 5
        # Energy should increase each step
        for i, (v, w, e) in enumerate(trace):
            assert e == i + 1

    def test_simulate_energy_depletes(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, -1)
        trace = simulate_play(g, 0, {0: 0}, {}, initial_energy=3, max_steps=10)
        # Energy: 3, 2, 1, 0, -1 (stops at -1)
        last_energy = trace[-1][2]
        assert last_energy < 0

    def test_verify_winning_strategy(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 0, 1)   # safe loop
        g.add_edge(0, 1, -5)  # bad
        g.add_edge(1, 0, 0)
        # Strategy: always self-loop at 0
        ok = verify_energy_strategy(g, 0, {0: 0}, initial_energy=0)
        assert ok is True

    def test_verify_losing_strategy(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, -1)
        # Only move is -1: loses
        ok = verify_energy_strategy(g, 0, {0: 0}, initial_energy=5)
        assert ok is False


# =====================================================================
# Section 9: Energy-Parity Games
# =====================================================================

class TestEnergyParity:
    def test_energy_parity_empty(self):
        g = EnergyParityGame()
        r = solve_energy_parity(g)
        assert len(r.win_energy) == 0

    def test_energy_parity_even_wins(self):
        """Even priority + positive weight: Even wins both conditions."""
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)  # even priority
        g.add_edge(0, 0, 1)  # positive weight
        r = solve_energy_parity(g)
        assert 0 in r.win_energy

    def test_energy_parity_energy_fails(self):
        """Even parity but negative energy: Even loses on energy condition."""
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_edge(0, 0, -1)
        r = solve_energy_parity(g)
        assert 0 in r.win_opponent

    def test_energy_parity_parity_fails(self):
        """Positive energy but odd parity: Even loses on parity condition."""
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 1)  # odd priority
        g.add_edge(0, 0, 1)
        r = solve_energy_parity(g)
        assert 0 in r.win_opponent

    def test_energy_parity_two_vertices(self):
        """Two vertices: even parity vertex with positive energy reachable."""
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)  # even priority
        g.add_vertex(1, Player.ODD, 1)   # odd priority
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)
        # Max priority seen infinitely often: 2 (even) -- Even wins parity
        # Net energy: +2 per cycle -- Even wins energy
        r = solve_energy_parity(g)
        assert 0 in r.win_energy

    def test_energy_parity_construction(self):
        g = make_energy_parity_game(
            edges=[(0, 1, 2), (1, 0, -1)],
            owners={0: Player.EVEN, 1: Player.ODD},
            priorities={0: 2, 1: 1}
        )
        assert len(g.vertices) == 2
        assert g.priority[0] == 2

    def test_to_energy_game(self):
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_edge(0, 0, 1)
        eg = g.to_energy_game()
        assert isinstance(eg, EnergyGame)
        assert 0 in eg.vertices

    def test_to_parity_game(self):
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_edge(0, 0, 1)
        pg = g.to_parity_game()
        assert isinstance(pg, ParityGame)
        assert pg.priority[0] == 2


# =====================================================================
# Section 10: Comparison and Analysis
# =====================================================================

class TestAnalysis:
    def test_compare_energy_vs_parity(self):
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)
        cmp = compare_energy_vs_parity(g)
        assert 'energy_only' in cmp
        assert 'parity_only' in cmp
        assert 'energy_parity' in cmp
        assert 'analysis' in cmp

    def test_statistics(self):
        g = make_choice_game()
        stats = energy_game_statistics(g)
        assert stats['num_vertices'] == 5
        assert stats['num_edges'] > 0
        assert stats['max_weight'] > 0
        assert 'win_energy_count' in stats

    def test_parity_to_energy(self):
        pg = ParityGame()
        pg.add_vertex(0, Player.EVEN, 2)
        pg.add_vertex(1, Player.ODD, 1)
        pg.add_edge(0, 1)
        pg.add_edge(1, 0)
        eg = parity_to_energy(pg)
        assert isinstance(eg, EnergyGame)
        assert eg.owner[0] == Player.EVEN

    def test_energy_to_mean_payoff(self):
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 3)
        r = energy_to_mean_payoff(g)
        assert isinstance(r, MeanPayoffResult)
        assert r.values[0] > 0


# =====================================================================
# Section 11: Complex Scenarios
# =====================================================================

class TestComplexScenarios:
    def test_multi_path_game(self):
        """Game with multiple paths of varying profitability."""
        g = EnergyGame()
        # Hub (Even) -> 3 paths of different quality
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)  # path A: +3 net
        g.add_vertex(2, Player.ODD)  # path B: 0 net
        g.add_vertex(3, Player.ODD)  # path C: -2 net

        g.add_edge(0, 1, -1)
        g.add_edge(0, 2, -2)
        g.add_edge(0, 3, -5)
        g.add_edge(1, 0, 4)   # net: -1+4 = +3
        g.add_edge(2, 0, 2)   # net: -2+2 = 0
        g.add_edge(3, 0, 3)   # net: -5+3 = -2

        r = solve_energy(g)
        assert 0 in r.win_energy  # Even picks path A
        assert r.min_energy[0] == 1  # need 1 for the -1 edge

    def test_adversarial_routing(self):
        """Odd routes Even through expensive path."""
        g = EnergyGame()
        g.add_vertex(0, Player.ODD)   # Odd routes
        g.add_vertex(1, Player.EVEN)  # cheap path
        g.add_vertex(2, Player.EVEN)  # expensive path

        g.add_edge(0, 1, -1)
        g.add_edge(0, 2, -10)
        g.add_edge(1, 0, 2)
        g.add_edge(2, 0, 11)  # expensive but net +1

        r = solve_energy(g)
        # Odd picks worst for Even: -10 edge (higher up-front cost)
        # But net is -10+11 = +1, so Even still wins
        assert 0 in r.win_energy
        assert r.min_energy[0] == 10  # must survive the -10 edge

    def test_cascade_drain(self):
        """Long chain of drains before recovery."""
        g = EnergyGame()
        n = 5
        for i in range(n):
            g.add_vertex(i, Player.ODD)
        g.add_vertex(n, Player.EVEN)  # recovery point

        for i in range(n):
            g.add_edge(i, i + 1, -1)  # drain
        g.add_edge(n, 0, n + 1)  # big recovery: net = -n + (n+1) = +1

        r = solve_energy(g)
        assert 0 in r.win_energy
        assert r.min_energy[0] == n  # need n to survive n drains

    def test_impossible_game(self):
        """All cycles have negative weight: Even cannot win."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.EVEN)
        g.add_edge(0, 1, -1)
        g.add_edge(1, 0, -1)
        r = solve_energy(g)
        assert r.win_opponent == {0, 1}

    def test_large_weight_game(self):
        """Game with large weights."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -100)
        g.add_edge(1, 0, 200)
        r = solve_energy(g)
        assert r.win_energy == {0, 1}
        assert r.min_energy[0] == 100


# =====================================================================
# Section 12: Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_multiple_self_loops(self):
        """Vertex with multiple self-loops: Even picks best."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, -5)
        g.add_edge(0, 0, 3)
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert r.min_energy[0] == 0  # pick the +3 loop

    def test_odd_multiple_self_loops(self):
        """Odd vertex with self-loops: Odd picks worst for Even."""
        g = EnergyGame()
        g.add_vertex(0, Player.ODD)
        g.add_edge(0, 0, 5)
        g.add_edge(0, 0, -2)
        r = solve_energy(g)
        # Odd picks -2: energy always depletes
        assert 0 in r.win_opponent

    def test_disconnected_components(self):
        """Two separate components."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_edge(0, 0, 1)  # positive loop
        g.add_vertex(1, Player.EVEN)
        g.add_edge(1, 1, -1)  # negative loop
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert 1 in r.win_opponent

    def test_all_even_owned(self):
        """All vertices owned by Even."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.EVEN)
        g.add_edge(0, 1, -3)
        g.add_edge(0, 0, 1)   # safe choice
        g.add_edge(1, 0, 4)
        r = solve_energy(g)
        assert 0 in r.win_energy
        assert r.min_energy[0] == 0  # self-loop

    def test_all_odd_owned_positive(self):
        """All vertices owned by Odd, but positive cycle."""
        g = EnergyGame()
        g.add_vertex(0, Player.ODD)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)
        r = solve_energy(g)
        # Odd has no choice (single edge from each), cycle is positive
        assert r.win_energy == {0, 1}

    def test_all_odd_with_choice(self):
        """Odd chooses between positive and negative cycles."""
        g = EnergyGame()
        g.add_vertex(0, Player.ODD)
        g.add_edge(0, 0, 5)   # positive
        g.add_edge(0, 0, -1)  # negative
        r = solve_energy(g)
        # Odd picks -1: drains
        assert 0 in r.win_opponent

    def test_energy_parity_max_priority(self):
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 4)
        g.add_edge(0, 0, 1)
        assert g.max_priority() == 4

    def test_energy_parity_max_weight(self):
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -7)
        assert g.max_weight() == 7


# =====================================================================
# Section 13: Strategy Correctness
# =====================================================================

class TestStrategyCorrectness:
    def test_even_strategy_stays_in_winning(self):
        """Even's strategy only targets winning vertices."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_vertex(2, Player.ODD)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 1)
        g.add_edge(2, 2, -1)  # trap

        r = solve_energy(g)
        # Even's strategy at 0 should go to 1 (winning), not 2 (losing)
        assert r.strategy_energy[0] == 1
        # Verify the strategy actually works
        assert 0 in r.win_energy

    def test_simulation_matches_analysis(self):
        """Simulation with winning strategy maintains energy."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -2)
        g.add_edge(1, 0, 5)

        r = solve_energy(g)
        assert r.min_energy[0] == 2
        trace = simulate_play(g, 0, r.strategy_energy, r.strategy_opponent,
                              initial_energy=r.min_energy[0], max_steps=20)
        # Energy should never go below 0
        for v, w, e in trace:
            assert e >= 0


# =====================================================================
# Section 14: Integration with V156
# =====================================================================

class TestV156Integration:
    def test_parity_game_roundtrip(self):
        """Energy-parity -> parity -> solve -> compare."""
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)

        pg = g.to_parity_game()
        sol = zielonka(pg)
        assert 0 in sol.win_even  # Even wins parity (max prio 2 is even)

    def test_combined_stricter_than_either(self):
        """Energy-parity winning region is subset of intersection."""
        g = EnergyParityGame()
        g.add_vertex(0, Player.EVEN, 2)  # even prio, but negative energy
        g.add_vertex(1, Player.ODD, 3)   # odd prio
        g.add_edge(0, 1, -1)
        g.add_edge(1, 0, -1)

        cmp = compare_energy_vs_parity(g)
        # Combined should be subset of energy-only AND parity-only
        combined = cmp['energy_parity']['win_energy']
        energy_only = cmp['energy_only']['win_energy']
        parity_only = cmp['parity_only']['win_even']
        assert combined <= energy_only
        assert combined <= parity_only


# =====================================================================
# Section 15: Regression and Robustness
# =====================================================================

class TestRobustness:
    def test_repeated_solve(self):
        """Solving same game twice gives same result."""
        g = make_choice_game()
        r1 = solve_energy(g)
        r2 = solve_energy(g)
        assert r1.min_energy == r2.min_energy
        assert r1.win_energy == r2.win_energy

    def test_vertex_ordering_irrelevant(self):
        """Adding vertices in different order gives same result."""
        g1 = EnergyGame()
        g1.add_vertex(0, Player.EVEN)
        g1.add_vertex(1, Player.ODD)
        g1.add_edge(0, 1, -2)
        g1.add_edge(1, 0, 3)

        g2 = EnergyGame()
        g2.add_vertex(1, Player.ODD)
        g2.add_vertex(0, Player.EVEN)
        g2.add_edge(1, 0, 3)
        g2.add_edge(0, 1, -2)

        r1 = solve_energy(g1)
        r2 = solve_energy(g2)
        assert r1.min_energy == r2.min_energy

    def test_all_zeros(self):
        """All-zero weights: everyone wins with energy 0."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, 0)
        g.add_edge(1, 0, 0)
        r = solve_energy(g)
        assert r.win_energy == {0, 1}
        assert r.min_energy[0] == 0
        assert r.min_energy[1] == 0

    def test_single_large_negative(self):
        """One very negative edge in otherwise positive game."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.EVEN)
        g.add_vertex(2, Player.ODD)
        g.add_edge(0, 1, -100)
        g.add_edge(0, 2, 1)   # safe
        g.add_edge(1, 0, 101) # recovery
        g.add_edge(2, 0, 1)

        r = solve_energy(g)
        assert 0 in r.win_energy
        # Even can just go to vertex 2 (safe path)
        assert r.min_energy[0] == 0
