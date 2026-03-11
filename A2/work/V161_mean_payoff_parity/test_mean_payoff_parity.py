"""
Tests for V161: Mean-Payoff Parity Games
"""

import pytest
import sys
sys.path.insert(0, 'Z:/AgentZero/A2/work/V161_mean_payoff_parity')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V156_parity_games')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V160_energy_games')

from mean_payoff_parity import (
    MeanPayoffParityGame, MPPResult, Player,
    solve_mpp, solve_mpp_threshold, compute_mpp_values,
    verify_mpp_strategy, simulate_play, decompose_mpp,
    make_mpp_game, make_chain_mpp, make_choice_mpp,
    make_adversarial_mpp, make_tradeoff_mpp, make_counter_mpp,
    mpp_statistics, mpp_summary, _compute_sccs,
)


# ===========================================================================
# Section 1: Data Structure Basics
# ===========================================================================

class TestDataStructures:
    def test_empty_game(self):
        g = MeanPayoffParityGame()
        assert len(g.vertices) == 0
        assert g.max_priority() == 0

    def test_add_vertex_and_edge(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        assert g.vertices == {0, 1}
        assert g.owner[0] == Player.EVEN
        assert g.priority[1] == 1
        assert g.successors(0) == [(1, 3)]
        assert g.predecessors(0) == [(1, -1)]

    def test_max_weight(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(0, 1, -5)
        g.add_edge(1, 0, 3)
        assert g.max_weight() == 5

    def test_to_parity_game(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 3)
        g.add_edge(0, 1, 5)
        pg = g.to_parity_game()
        assert pg.vertices == {0, 1}
        assert pg.priority[0] == 2

    def test_to_energy_game(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 3)
        g.add_edge(0, 1, 5)
        eg = g.to_energy_game()
        assert eg.vertices == {0, 1}
        assert eg.successors(0) == [(1, 5)]

    def test_to_energy_parity_game(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 3)
        g.add_edge(0, 1, 5)
        g.add_edge(1, 0, -2)
        epg = g.to_energy_parity_game()
        assert epg.priority[0] == 2
        assert epg.successors(0) == [(1, 5)]

    def test_subgame(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 2)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, -1)
        g.add_edge(2, 0, 2)
        sg = g.subgame({0, 1})
        assert sg.vertices == {0, 1}
        assert sg.successors(0) == [(1, 1)]
        assert sg.successors(1) == []  # edge to 2 excluded

    def test_shift_weights(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        shifted = g.shift_weights(-2)
        assert shifted.successors(0) == [(1, 1)]
        assert shifted.successors(1) == [(0, -3)]


# ===========================================================================
# Section 2: Simple Parity-Only Games (mean-payoff trivially >= 0)
# ===========================================================================

class TestParityOnly:
    def test_single_vertex_even_priority(self):
        """Single self-loop vertex with even priority. Even wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_edge(0, 0, 0)
        r = solve_mpp(g)
        assert 0 in r.win_even

    def test_single_vertex_odd_priority(self):
        """Single self-loop vertex with odd priority. Odd wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 1)
        g.add_edge(0, 0, 0)
        r = solve_mpp(g)
        assert 0 in r.win_odd

    def test_two_vertex_even_wins(self):
        """0 (Even, p=2) -> 1 (Odd, p=0) -> 0. All even priorities."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 0)
        g.add_edge(0, 1, 0)
        g.add_edge(1, 0, 0)
        r = solve_mpp(g)
        assert r.win_even == {0, 1}

    def test_two_vertex_odd_wins(self):
        """0 (Odd, p=1) -> 1 (Even, p=1) -> 0. All odd priorities."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.ODD, 1)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 0)
        g.add_edge(1, 0, 0)
        r = solve_mpp(g)
        assert r.win_odd == {0, 1}


# ===========================================================================
# Section 3: Simple Mean-Payoff-Only Games (parity trivially satisfied)
# ===========================================================================

class TestMeanPayoffOnly:
    def test_positive_self_loop(self):
        """Self-loop with positive weight and even priority. Even wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        r = solve_mpp(g)
        assert 0 in r.win_even

    def test_negative_self_loop(self):
        """Self-loop with negative weight and even priority.
        Mean-payoff is negative -> Odd wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -1)
        r = solve_mpp(g)
        assert 0 in r.win_odd

    def test_zero_weight_cycle(self):
        """Cycle with zero total weight. Mean-payoff = 0 >= 0. Even wins."""
        g = make_counter_mpp(3)  # +1, +1, -2 -> mean = 0
        r = solve_mpp(g)
        assert r.win_even == {0, 1, 2}

    def test_positive_mean_cycle(self):
        """Chain with positive mean-payoff."""
        g = make_chain_mpp(3, weights=[2, 1, 0])
        r = solve_mpp(g)
        assert r.win_even == {0, 1, 2}  # mean = 1 > 0

    def test_choice_positive_vs_negative(self):
        """Even chooses between positive and negative cycles."""
        g = make_choice_mpp(good_weight=5, bad_weight=-3,
                           good_prio=0, bad_prio=0)
        r = solve_mpp(g)
        # Even can pick the positive cycle
        assert 0 in r.win_even


# ===========================================================================
# Section 4: Combined Parity + Mean-Payoff Interaction
# ===========================================================================

class TestCombined:
    def test_tradeoff_even_loses(self):
        """Tradeoff game: good parity = bad MP, good MP = bad parity.
        Even can't win both conditions simultaneously."""
        g = make_tradeoff_mpp()
        r = solve_mpp(g)
        # Even can't satisfy both: loses from all vertices
        assert 0 in r.win_odd

    def test_good_parity_and_good_mp(self):
        """Even can satisfy both: even priority + positive weight."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_edge(0, 0, 1)
        r = solve_mpp(g)
        assert 0 in r.win_even

    def test_good_parity_bad_mp(self):
        """Even priority but negative weight -> Odd wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_edge(0, 0, -1)
        r = solve_mpp(g)
        assert 0 in r.win_odd

    def test_bad_parity_good_mp(self):
        """Odd priority but positive weight -> Odd wins (parity fails)."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 1)
        g.add_edge(0, 0, 5)
        r = solve_mpp(g)
        assert 0 in r.win_odd

    def test_choice_between_both_good(self):
        """Even can choose a cycle with both good parity and good MP."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 2)  # good parity
        g.add_vertex(2, Player.EVEN, 1)  # bad parity
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 3)   # good MP
        g.add_edge(2, 0, 10)  # better MP but bad parity
        r = solve_mpp(g)
        assert 0 in r.win_even  # Even picks vertex 1 path


# ===========================================================================
# Section 5: Threshold Tests
# ===========================================================================

class TestThreshold:
    def test_threshold_zero(self):
        """Default threshold = 0. Positive MP + even parity wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        r = solve_mpp_threshold(g, 0.0)
        assert 0 in r.win_even

    def test_threshold_below_mp(self):
        """Mean-payoff is 2, threshold is 1. Even wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 2)
        r = solve_mpp_threshold(g, 1.0)
        assert 0 in r.win_even

    def test_threshold_above_mp(self):
        """Mean-payoff is 1, threshold is 3. Even loses."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        r = solve_mpp_threshold(g, 3.0)
        assert 0 in r.win_odd

    def test_threshold_exactly_at_mp(self):
        """Mean-payoff equals threshold. Even wins (>= threshold)."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        # Mean-payoff = (3 + -1) / 2 = 1.0
        r = solve_mpp_threshold(g, 1.0)
        assert 0 in r.win_even

    def test_negative_threshold(self):
        """Negative threshold: mean-payoff -1 >= -2. Even wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -1)
        r = solve_mpp_threshold(g, -2.0)
        assert 0 in r.win_even

    def test_empty_game_threshold(self):
        g = MeanPayoffParityGame()
        r = solve_mpp_threshold(g, 5.0)
        assert r.win_even == set()
        assert r.win_odd == set()


# ===========================================================================
# Section 6: Optimal Value Computation
# ===========================================================================

class TestOptimalValues:
    def test_self_loop_positive(self):
        """Self-loop with weight 3. Optimal value = 3."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 3)
        r = compute_mpp_values(g)
        assert r.values is not None
        assert abs(r.values[0] - 3.0) < 0.5

    def test_cycle_mean(self):
        """Cycle: weights [2, -1]. Mean = 0.5."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(0, 1, 2)
        g.add_edge(1, 0, -1)
        r = compute_mpp_values(g)
        assert r.values is not None
        # Values should be close to 0.5
        assert abs(r.values[0] - 0.5) < 0.6

    def test_odd_parity_value_negative_inf(self):
        """Odd priority cycle -> Even can't win at any threshold."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 1)
        g.add_edge(0, 0, 5)
        r = compute_mpp_values(g)
        assert 0 in r.win_odd  # Even can't win parity

    def test_choice_optimal(self):
        """Even chooses cycle with best MP that also satisfies parity."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)  # even prio, weight 2
        g.add_vertex(2, Player.EVEN, 0)  # even prio, weight 5
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 2)
        g.add_edge(2, 0, 5)
        r = compute_mpp_values(g)
        assert r.values is not None
        # Even should pick vertex 2 path for value ~2.5
        assert r.values[0] >= 1.0


# ===========================================================================
# Section 7: Adversarial Games
# ===========================================================================

class TestAdversarial:
    def test_adversarial_positive(self):
        """Odd picks between +w and -w cycles. Both even prio.
        Odd picks the negative one -> Even loses if -w < 0."""
        g = make_adversarial_mpp(3, even_prio=0, odd_prio=0)
        r = solve_mpp(g)
        # Odd picks -3 cycle, mean-payoff = -1.5. Even loses.
        assert 0 in r.win_odd

    def test_adversarial_both_positive(self):
        """Odd picks but both cycles are positive. Even wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 3)
        g.add_edge(2, 0, 1)
        r = solve_mpp(g)
        # Odd picks 1 (the worse one) but it's still positive
        # Mean-payoff = 0.5 >= 0
        assert 0 in r.win_even

    def test_adversarial_parity_choice(self):
        """Odd picks cycle: one has bad parity, one has bad MP."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.EVEN, 1)  # odd parity
        g.add_vertex(2, Player.EVEN, 0)  # even parity
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 5)   # good MP but bad parity
        g.add_edge(2, 0, -1)  # bad MP but good parity
        r = solve_mpp(g)
        # Odd picks whichever makes Even lose: either bad parity or bad MP
        assert 0 in r.win_odd


# ===========================================================================
# Section 8: Construction Helpers
# ===========================================================================

class TestConstructors:
    def test_make_mpp_game(self):
        g = make_mpp_game(3,
                          [(0, 1, 2), (1, 2, -1), (2, 0, 1)],
                          {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN},
                          {0: 0, 1: 1, 2: 2})
        assert len(g.vertices) == 3
        assert g.priority[2] == 2
        assert g.successors(0) == [(1, 2)]

    def test_make_chain_mpp(self):
        g = make_chain_mpp(4, weights=[1, 2, 3, -6])
        assert len(g.vertices) == 4
        # Mean-payoff = (1+2+3-6)/4 = 0
        r = solve_mpp(g)
        assert r.win_even == {0, 1, 2, 3}

    def test_make_chain_negative_mean(self):
        g = make_chain_mpp(3, weights=[1, 1, -5])
        # Mean = (1+1-5)/3 = -1 < 0
        r = solve_mpp(g)
        assert r.win_odd == {0, 1, 2}

    def test_make_choice_mpp(self):
        g = make_choice_mpp(5, -3, good_prio=0, bad_prio=0)
        r = solve_mpp(g)
        assert 0 in r.win_even  # Even picks good cycle

    def test_make_counter_mpp(self):
        g = make_counter_mpp(5)
        # Weights: +1, +1, +1, +1, -4. Mean = 0.
        r = solve_mpp(g)
        assert r.win_even == {0, 1, 2, 3, 4}

    def test_make_tradeoff_mpp(self):
        g = make_tradeoff_mpp()
        r = solve_mpp(g)
        assert 0 in r.win_odd  # Even can't satisfy both


# ===========================================================================
# Section 9: Decomposition Analysis
# ===========================================================================

class TestDecomposition:
    def test_decompose_trivial(self):
        """Game where parity and MP are both satisfied."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        d = decompose_mpp(g)
        assert 0 in d["parity_only"]["win_even"]
        assert 0 in d["mean_payoff_only"]["win_even"]
        assert 0 in d["combined"]["win_even"]
        assert len(d["analysis"]["lost_to_interaction"]) == 0

    def test_decompose_tradeoff(self):
        """Tradeoff game: wins individually but not combined."""
        g = make_tradeoff_mpp()
        d = decompose_mpp(g)
        # Parity alone: Even can pick even-priority cycle
        # MP alone: Even can pick positive-weight cycle
        # Combined: can't have both
        assert 0 in d["combined"]["win_odd"]

    def test_decompose_threshold(self):
        """Decomposition with non-zero threshold."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 2)
        d = decompose_mpp(g, threshold=1.0)
        assert 0 in d["combined"]["win_even"]

    def test_decompose_statistics(self):
        g = make_counter_mpp(4)
        d = decompose_mpp(g)
        assert "parity_only" in d
        assert "mean_payoff_only" in d
        assert "combined" in d
        assert "analysis" in d


# ===========================================================================
# Section 10: Strategy Verification
# ===========================================================================

class TestVerification:
    def test_verify_even_wins(self):
        """Verify Even's winning strategy on a simple game."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        strategy = {0: 0}
        result = verify_mpp_strategy(g, strategy, Player.EVEN)
        assert result[0]["valid"]

    def test_verify_odd_wins(self):
        """Verify Odd's winning strategy."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 1)  # odd priority
        g.add_edge(0, 0, 5)
        strategy = {}  # Even has no choice
        result = verify_mpp_strategy(g, strategy, Player.EVEN)
        # Even loses because parity is odd
        assert not result[0]["valid"]

    def test_verify_choice_strategy(self):
        """Even picks the good cycle."""
        g = make_choice_mpp(5, -3, good_prio=0, bad_prio=0)
        strategy = {0: 1, 1: 0}  # pick vertex 1 (good)
        result = verify_mpp_strategy(g, strategy, Player.EVEN)
        assert result[0]["valid"]

    def test_verify_bad_strategy(self):
        """Even picks the bad cycle (negative MP)."""
        g = make_choice_mpp(5, -3, good_prio=0, bad_prio=0)
        strategy = {0: 2, 2: 0}  # pick vertex 2 (bad)
        result = verify_mpp_strategy(g, strategy, Player.EVEN)
        assert result[0]["valid"] == False

    def test_verify_empty_game(self):
        g = MeanPayoffParityGame()
        result = verify_mpp_strategy(g, {}, Player.EVEN)
        assert result == {"valid": True, "reason": "empty game"}


# ===========================================================================
# Section 11: Simulation
# ===========================================================================

class TestSimulation:
    def test_simulate_self_loop(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 3)
        result = simulate_play(g, {0: 0}, {}, 0, steps=10)
        assert result["path"][0] == 0
        assert len(result["weights"]) == 10
        assert all(w == 3 for w in result["weights"])
        assert abs(result["final_mean"] - 3.0) < 0.01

    def test_simulate_cycle(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 2)
        g.add_edge(0, 1, 4)
        g.add_edge(1, 0, -2)
        result = simulate_play(g, {0: 1, 1: 0}, {}, 0, steps=20)
        assert len(result["weights"]) == 20
        # Mean should converge to 1.0
        assert abs(result["final_mean"] - 1.0) < 0.5

    def test_simulate_dead_end(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        result = simulate_play(g, {}, {}, 0, steps=10)
        assert result["steps"] == 0

    def test_simulate_adversarial(self):
        g = make_adversarial_mpp(2, even_prio=0, odd_prio=0)
        result = simulate_play(g, {1: 0, 2: 0}, {0: 2}, 0, steps=20)
        assert len(result["weights"]) == 20


# ===========================================================================
# Section 12: Statistics and Summary
# ===========================================================================

class TestStatistics:
    def test_statistics_basic(self):
        g = make_counter_mpp(4)
        stats = mpp_statistics(g)
        assert stats["vertices"] == 4
        assert stats["edges"] == 4
        assert stats["even_vertices"] == 4
        assert stats["odd_vertices"] == 0
        assert stats["nontrivial_sccs"] == 1

    def test_statistics_mixed_owners(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 2)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, -1)
        g.add_edge(2, 0, 0)
        stats = mpp_statistics(g)
        assert stats["even_vertices"] == 2
        assert stats["odd_vertices"] == 1
        assert stats["weight_range"] == (-1, 1)

    def test_summary_string(self):
        g = make_counter_mpp(3)
        s = mpp_summary(g)
        assert "Mean-Payoff Parity" in s
        assert "Vertices: 3" in s

    def test_scc_computation(self):
        from collections import defaultdict
        edges = defaultdict(list)
        edges[0] = [(1, 0)]
        edges[1] = [(2, 0)]
        edges[2] = [(0, 0)]
        edges[3] = [(3, 0)]
        sccs = _compute_sccs({0, 1, 2, 3}, edges)
        scc_sizes = sorted(len(s) for s in sccs)
        assert scc_sizes == [1, 3]


# ===========================================================================
# Section 13: Multi-Vertex Complex Games
# ===========================================================================

class TestComplex:
    def test_diamond_game(self):
        """Diamond: 0 -> {1, 2} -> 3 -> 0.
        Even picks at 0 to maximize, Odd irrelevant."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_vertex(3, Player.EVEN, 0)
        g.add_edge(0, 1, 1)
        g.add_edge(0, 2, -5)
        g.add_edge(1, 3, 2)
        g.add_edge(2, 3, 3)
        g.add_edge(3, 0, 0)
        r = solve_mpp(g)
        # Even picks 0->1->3->0 with mean = (1+2+0)/3 = 1 >= 0
        assert 0 in r.win_even

    def test_four_vertex_mixed_parity(self):
        """4-vertex game with mixed priorities and weights."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)  # even prio
        g.add_vertex(1, Player.ODD, 1)   # odd prio
        g.add_vertex(2, Player.EVEN, 0)  # even prio
        g.add_vertex(3, Player.ODD, 3)   # odd prio
        g.add_edge(0, 1, 2)
        g.add_edge(0, 2, 1)
        g.add_edge(1, 0, -1)
        g.add_edge(1, 3, 0)
        g.add_edge(2, 0, 3)
        g.add_edge(3, 0, -2)
        r = solve_mpp(g)
        # Even can pick 0->2->0 cycle: priorities {2, 0}, max even priority 2
        # Mean-payoff = (1+3)/2 = 2 >= 0
        assert 0 in r.win_even

    def test_nested_cycles(self):
        """Two nested cycles with different properties."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 0)
        # Inner cycle: 0 <-> 1 (weight +2, -2, mean = 0)
        g.add_edge(0, 1, 2)
        g.add_edge(1, 0, -2)
        # Outer: 0 -> 2 -> 0 (weight +1, +1, mean = 1)
        g.add_edge(0, 2, 1)
        g.add_edge(2, 0, 1)
        r = solve_mpp(g)
        assert 0 in r.win_even  # Even picks outer cycle

    def test_large_chain(self):
        """10-vertex chain, all positive weights."""
        g = make_chain_mpp(10, weights=[1]*10)
        r = solve_mpp(g)
        assert r.win_even == set(range(10))


# ===========================================================================
# Section 14: Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_dead_end_even(self):
        """Even vertex with no outgoing edges -> Even loses."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        r = solve_mpp(g)
        assert 0 in r.win_odd

    def test_dead_end_odd(self):
        """Odd vertex with no outgoing edges -> Odd is stuck, Even wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.ODD, 0)
        r = solve_mpp(g)
        assert 0 in r.win_even

    def test_single_vertex_zero_weight(self):
        """Self-loop with weight 0, even priority. Mean = 0 >= 0. Even wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 0)
        r = solve_mpp(g)
        assert 0 in r.win_even

    def test_disconnected_vertices(self):
        """Two disconnected self-loops."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 0, 1)
        g.add_edge(1, 1, 1)
        r = solve_mpp(g)
        assert 0 in r.win_even   # even prio, positive MP
        assert 1 in r.win_odd    # odd prio

    def test_large_weights(self):
        """Game with large weights."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(0, 1, 1000)
        g.add_edge(1, 0, -999)
        r = solve_mpp(g)
        assert r.win_even == {0, 1}  # mean = 0.5


# ===========================================================================
# Section 15: Solver Consistency
# ===========================================================================

class TestConsistency:
    def test_solve_mpp_equals_threshold_zero(self):
        """solve_mpp and solve_mpp_threshold(0) give same results."""
        g = make_choice_mpp(3, -2, good_prio=0, bad_prio=0)
        r1 = solve_mpp(g)
        r2 = solve_mpp_threshold(g, 0.0)
        assert r1.win_even == r2.win_even
        assert r1.win_odd == r2.win_odd

    def test_monotone_threshold(self):
        """Higher threshold -> fewer (or equal) Even-winning vertices."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -1)
        # Mean-payoff = 1
        r_low = solve_mpp_threshold(g, 0.0)
        r_exact = solve_mpp_threshold(g, 1.0)
        r_high = solve_mpp_threshold(g, 2.0)
        assert r_low.win_even >= r_high.win_even
        assert 0 in r_low.win_even
        assert 0 in r_high.win_odd

    def test_parity_only_agrees(self):
        """With zero weights, solve_mpp should agree with parity-only Zielonka."""
        from parity_games import zielonka
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1, 0)
        g.add_edge(1, 2, 0)
        g.add_edge(2, 0, 0)
        g.add_edge(1, 0, 0)
        mpp_result = solve_mpp(g)
        parity_result = zielonka(g.to_parity_game())
        assert mpp_result.win_even == parity_result.win_even
        assert mpp_result.win_odd == parity_result.win_odd

    def test_decomposition_consistent_with_solver(self):
        """Decomposition combined result matches solver."""
        g = make_tradeoff_mpp()
        d = decompose_mpp(g)
        r = solve_mpp(g)
        assert d["combined"]["win_even"] == r.win_even
        assert d["combined"]["win_odd"] == r.win_odd


# ===========================================================================
# Section 16: Regression Tests
# ===========================================================================

class TestRegression:
    def test_counter_mpp_all_sizes(self):
        """Counter games of various sizes all have mean-payoff = 0."""
        for n in [2, 3, 5, 8]:
            g = make_counter_mpp(n)
            r = solve_mpp(g)
            assert r.win_even == set(range(n)), f"Failed for n={n}"

    def test_choice_always_picks_better(self):
        """Even always picks the better option when one exists."""
        for w in [1, 3, 5, 10]:
            g = make_choice_mpp(w, -w, good_prio=0, bad_prio=0)
            r = solve_mpp(g)
            assert 0 in r.win_even, f"Failed for w={w}"

    def test_adversarial_always_picks_worse(self):
        """Odd always forces Even into the worse cycle."""
        for w in [1, 3, 5]:
            g = make_adversarial_mpp(w, even_prio=0, odd_prio=0)
            r = solve_mpp(g)
            assert 0 in r.win_odd, f"Failed for w={w}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
