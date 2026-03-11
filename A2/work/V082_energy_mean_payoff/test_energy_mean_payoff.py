"""Tests for V082: Energy Games and Mean-Payoff Games."""

import pytest
from math import inf
from energy_mean_payoff import (
    WeightedGame, WeightedParityGame, Player,
    EnergyResult, MeanPayoffResult, EnergyParityResult,
    solve_energy, solve_mean_payoff, solve_energy_parity,
    make_weighted_game, make_weighted_parity_game,
    energy_to_mean_payoff, mean_payoff_to_energy,
    verify_energy_strategy, verify_mean_payoff_strategy,
    compare_energy_mean_payoff, parity_game_to_weighted,
    weighted_game_summary,
)
from parity_games import ParityGame, Player as PPlayer


# ===========================================================================
# Section 1: WeightedGame construction and validation
# ===========================================================================

class TestWeightedGameConstruction:
    def test_empty_game(self):
        g = WeightedGame()
        assert len(g.nodes) == 0
        assert g.validate() == []

    def test_add_nodes_and_edges(self):
        g = WeightedGame()
        g.add_node(0, Player.EVEN)
        g.add_node(1, Player.ODD)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -2)
        assert g.nodes == {0, 1}
        assert g.weight[(0, 1)] == 3
        assert g.weight[(1, 0)] == -2
        assert 1 in g.successors[0]
        assert 0 in g.predecessors[1]

    def test_make_weighted_game(self):
        g = make_weighted_game(
            nodes=[(0, 0), (1, 1), (2, 0)],
            edges=[(0, 1, 5), (1, 2, -3), (2, 0, 1)]
        )
        assert len(g.nodes) == 3
        assert g.owner[0] == Player.EVEN
        assert g.owner[1] == Player.ODD
        assert g.weight[(0, 1)] == 5

    def test_max_min_weight(self):
        g = make_weighted_game(
            nodes=[(0, 0), (1, 1)],
            edges=[(0, 1, 5), (1, 0, -3)]
        )
        assert g.max_weight() == 5
        assert g.min_weight() == -3

    def test_validate_deadlock(self):
        g = WeightedGame()
        g.add_node(0, Player.EVEN)
        issues = g.validate()
        assert any("no successors" in i for i in issues)

    def test_n_edges(self):
        g = make_weighted_game(
            nodes=[(0, 0), (1, 1)],
            edges=[(0, 1, 1), (1, 0, 2)]
        )
        assert g.n_edges() == 2


# ===========================================================================
# Section 2: Energy Games - Basic
# ===========================================================================

class TestEnergyBasic:
    def test_single_node_positive_self_loop(self):
        """Single node with positive self-loop: Even wins with 0 credit."""
        g = make_weighted_game([(0, 0)], [(0, 0, 1)])
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        assert r.min_credit[0] == 0

    def test_single_node_negative_self_loop(self):
        """Single node with negative self-loop: Even loses (energy drops forever)."""
        g = make_weighted_game([(0, 0)], [(0, 0, -1)])
        r = solve_energy(g)
        assert r.winner[0] == Player.ODD

    def test_single_node_zero_self_loop(self):
        """Single node with zero weight: Even wins with 0 credit."""
        g = make_weighted_game([(0, 0)], [(0, 0, 0)])
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        assert r.min_credit[0] == 0

    def test_two_node_cycle_positive(self):
        """Two nodes in a cycle with net positive weight."""
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -1), (1, 0, 3)]
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        assert r.winner[1] == Player.EVEN
        # Node 0 needs credit 1 to survive the -1 edge
        assert r.min_credit[0] == 1
        # Node 1: after going to 0 (+3), then 0->1 (-1), net is +2
        assert r.min_credit[1] == 0

    def test_two_node_cycle_negative(self):
        """Two nodes in cycle with net negative weight: Even loses."""
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -3), (1, 0, 1)]
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.ODD
        assert r.winner[1] == Player.ODD


# ===========================================================================
# Section 3: Energy Games - Player Choice
# ===========================================================================

class TestEnergyPlayerChoice:
    def test_even_chooses_good_edge(self):
        """Even-owned node with choice between good and bad edges."""
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1, -10), (0, 2, 5), (1, 0, 1), (2, 0, 1)]
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        # Even should choose edge 0->2 (weight 5)
        assert r.strategy_even[0] == 2

    def test_odd_chooses_bad_edge(self):
        """Odd-owned node forces worst case for Even."""
        g = make_weighted_game(
            [(0, 1), (1, 0), (2, 0)],
            [(0, 1, -10), (0, 2, 5), (1, 0, 1), (2, 0, 1)]
        )
        r = solve_energy(g)
        # Odd will choose edge 0->1 (weight -10), forcing high credit need
        # From node 1: go to 0 (+1), then 0->1 (-10), net = -9 per cycle
        assert r.winner[0] == Player.ODD

    def test_even_choice_with_different_credits(self):
        """Even chooses between paths with different credit requirements."""
        # Path 1: 0->1->0, weights -2, +3 (net +1, needs credit 2)
        # Path 2: 0->2->0, weights -1, +1 (net 0, needs credit 1)
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1, -2), (0, 2, -1), (1, 0, 3), (2, 0, 1)]
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        # Even prefers the path needing less credit
        assert r.min_credit[0] <= 2


# ===========================================================================
# Section 4: Energy Games - Complex structures
# ===========================================================================

class TestEnergyComplex:
    def test_diamond_game(self):
        """Diamond: 0->(1,2)->3->0, Even at 0,3 Odd at 1,2."""
        g = make_weighted_game(
            [(0, 0), (1, 1), (2, 1), (3, 0)],
            [(0, 1, 1), (0, 2, 1), (1, 3, -5), (2, 3, 2), (3, 0, 0)]
        )
        r = solve_energy(g)
        # Even chooses at 0, Odd chooses at 1,2
        # If Even goes to 1: Odd goes 1->3 (-5), Even goes 3->0 (0). Net: 1-5+0 = -4
        # If Even goes to 2: Odd goes 2->3 (+2), Even goes 3->0 (0). Net: 1+2+0 = +3
        assert r.winner[0] == Player.EVEN

    def test_three_node_chain_with_return(self):
        """Chain: 0->1->2->0 with specific weights."""
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1, -3), (1, 2, -2), (2, 0, 10)]
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        # Need: credit >= 3 after 0->1, then >= 5 after 1->2, then gain 10
        assert r.min_credit[0] == 5  # 0: need 5 to cover -3 then -2
        assert r.min_credit[1] == 2  # 1: need 2 to cover -2

    def test_multiple_sccs(self):
        """Two separate cycles: one winning, one losing."""
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1, 5), (1, 0, -3),  # SCC1: net +2 (winning)
             (2, 3, -5), (3, 2, 1)]  # SCC2: net -4 (losing)
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        assert r.winner[1] == Player.EVEN
        assert r.winner[2] == Player.ODD
        assert r.winner[3] == Player.ODD

    def test_zero_weight_game(self):
        """All-zero weights: Even always wins with 0 credit."""
        g = make_weighted_game(
            [(0, 0), (1, 1)],
            [(0, 1, 0), (1, 0, 0)]
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        assert r.winner[1] == Player.EVEN
        assert r.min_credit[0] == 0
        assert r.min_credit[1] == 0

    def test_large_weight_game(self):
        """Large weights: credit computation is correct."""
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -100), (1, 0, 200)]
        )
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        assert r.min_credit[0] == 100


# ===========================================================================
# Section 5: Energy Game Strategy Verification
# ===========================================================================

class TestEnergyVerification:
    def test_verify_winning_strategy(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -2), (1, 0, 5)]
        )
        r = solve_energy(g)
        report = verify_energy_strategy(g, r)
        assert report['valid']

    def test_verify_positive_cycle(self):
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1, 1), (1, 2, 1), (2, 0, 1)]
        )
        r = solve_energy(g)
        report = verify_energy_strategy(g, r)
        assert report['valid']


# ===========================================================================
# Section 6: Mean-Payoff Games - Basic
# ===========================================================================

class TestMeanPayoffBasic:
    def test_positive_self_loop(self):
        g = make_weighted_game([(0, 0)], [(0, 0, 3)])
        r = solve_mean_payoff(g)
        assert r.value[0] == pytest.approx(3.0, abs=0.1)
        assert r.winner[0] == Player.EVEN

    def test_negative_self_loop(self):
        g = make_weighted_game([(0, 0)], [(0, 0, -2)])
        r = solve_mean_payoff(g)
        assert r.value[0] == pytest.approx(-2.0, abs=0.1)
        assert r.winner[0] == Player.ODD

    def test_zero_mean_cycle(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 3), (1, 0, -3)]
        )
        r = solve_mean_payoff(g)
        assert r.value[0] == pytest.approx(0.0, abs=0.1)

    def test_positive_mean_cycle(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 5), (1, 0, -1)]
        )
        r = solve_mean_payoff(g)
        # Mean = (5 + (-1)) / 2 = 2.0
        assert r.value[0] == pytest.approx(2.0, abs=0.1)
        assert r.winner[0] == Player.EVEN

    def test_negative_mean_cycle(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -5), (1, 0, 1)]
        )
        r = solve_mean_payoff(g)
        # Mean = (-5 + 1) / 2 = -2.0
        assert r.value[0] == pytest.approx(-2.0, abs=0.1)
        assert r.winner[0] == Player.ODD


# ===========================================================================
# Section 7: Mean-Payoff Games - Player Choice
# ===========================================================================

class TestMeanPayoffChoice:
    def test_even_chooses_better_cycle(self):
        """Even picks the cycle with higher mean."""
        # 0 -> 1 -> 0: mean = (2 + (-1))/2 = 0.5
        # 0 -> 2 -> 0: mean = (5 + (-1))/2 = 2.0
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1, 2), (1, 0, -1), (0, 2, 5), (2, 0, -1)]
        )
        r = solve_mean_payoff(g)
        assert r.value[0] == pytest.approx(2.0, abs=0.1)
        assert r.winner[0] == Player.EVEN

    def test_odd_chooses_worse_cycle(self):
        """Odd picks the cycle with lower mean."""
        # 0 -> 1 -> 0: mean = (2 + (-1))/2 = 0.5
        # 0 -> 2 -> 0: mean = (-5 + 1)/2 = -2.0
        g = make_weighted_game(
            [(0, 1), (1, 0), (2, 0)],
            [(0, 1, 2), (1, 0, -1), (0, 2, -5), (2, 0, 1)]
        )
        r = solve_mean_payoff(g)
        assert r.value[0] == pytest.approx(-2.0, abs=0.1)
        assert r.winner[0] == Player.ODD

    def test_threshold_comparison(self):
        """Mean payoff exactly at threshold."""
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 3), (1, 0, -1)]
        )
        # Mean = 1.0
        r1 = solve_mean_payoff(g, threshold=1.0)
        assert r1.winner[0] == Player.EVEN

        r2 = solve_mean_payoff(g, threshold=1.5)
        assert r2.winner[0] == Player.ODD


# ===========================================================================
# Section 8: Mean-Payoff Strategy Verification
# ===========================================================================

class TestMeanPayoffVerification:
    def test_verify_strategy_positive(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 5), (1, 0, -1)]
        )
        r = solve_mean_payoff(g)
        report = verify_mean_payoff_strategy(g, r)
        assert 0 in report['node_reports']

    def test_verify_strategy_negative(self):
        g = make_weighted_game(
            [(0, 0)],
            [(0, 0, -3)]
        )
        r = solve_mean_payoff(g)
        report = verify_mean_payoff_strategy(g, r)
        assert 0 in report['node_reports']


# ===========================================================================
# Section 9: Energy-Mean-Payoff Connection
# ===========================================================================

class TestEnergyMeanPayoffConnection:
    def test_positive_mean_implies_energy_win(self):
        """If mean payoff > 0, Even wins energy game."""
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 5), (1, 0, -1)]
        )
        comparison = compare_energy_mean_payoff(g)
        assert comparison['connection_holds']

    def test_negative_mean_implies_energy_loss(self):
        """If mean payoff < 0, Even loses energy game."""
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -5), (1, 0, 1)]
        )
        comparison = compare_energy_mean_payoff(g)
        assert comparison['connection_holds']

    def test_energy_to_mean_payoff_api(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 3), (1, 0, -1)]
        )
        r = energy_to_mean_payoff(g)
        assert isinstance(r, MeanPayoffResult)
        assert r.value[0] == pytest.approx(1.0, abs=0.1)

    def test_mean_payoff_to_energy_api(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 3), (1, 0, -1)]
        )
        # Mean = 1.0, so shifting by 1.0 gives zero-mean -> Even wins energy
        r = mean_payoff_to_energy(g, shift=0)
        assert r.winner[0] == Player.EVEN


# ===========================================================================
# Section 10: WeightedParityGame construction
# ===========================================================================

class TestWeightedParityGame:
    def test_construction(self):
        g = make_weighted_parity_game(
            [(0, 0, 2), (1, 1, 1)],
            [(0, 1, 3), (1, 0, -1)]
        )
        assert g.priority[0] == 2
        assert g.priority[1] == 1
        assert g.weight[(0, 1)] == 3

    def test_to_parity_game(self):
        g = make_weighted_parity_game(
            [(0, 0, 2), (1, 1, 1)],
            [(0, 1, 3), (1, 0, -1)]
        )
        pg = g.to_parity_game()
        assert isinstance(pg, ParityGame)
        assert pg.priority[0] == 2
        assert pg.priority[1] == 1

    def test_to_weighted_game(self):
        g = make_weighted_parity_game(
            [(0, 0, 2), (1, 1, 1)],
            [(0, 1, 3), (1, 0, -1)]
        )
        wg = g.to_weighted_game()
        assert isinstance(wg, WeightedGame)
        assert wg.weight[(0, 1)] == 3

    def test_max_priority(self):
        g = make_weighted_parity_game(
            [(0, 0, 4), (1, 1, 1), (2, 0, 3)],
            [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
        )
        assert g.max_priority() == 4


# ===========================================================================
# Section 11: Energy-Parity Games
# ===========================================================================

class TestEnergyParity:
    def test_parity_dominates_all_positive(self):
        """All positive weights but odd parity dominates: Odd wins parity."""
        g = make_weighted_parity_game(
            [(0, 0, 1), (1, 0, 1)],  # Both priority 1 (odd) -> Odd wins parity
            [(0, 1, 5), (1, 0, 5)]
        )
        r = solve_energy_parity(g)
        # Parity: max inf priority = 1 (odd) -> Odd wins parity
        assert r.winner[0] == Player.ODD
        assert r.winner[1] == Player.ODD

    def test_even_parity_positive_energy(self):
        """Even parity + positive energy: Even wins both."""
        g = make_weighted_parity_game(
            [(0, 0, 2), (1, 0, 2)],  # Both priority 2 (even) -> Even wins parity
            [(0, 1, 1), (1, 0, 1)]
        )
        r = solve_energy_parity(g)
        assert r.winner[0] == Player.EVEN
        assert r.winner[1] == Player.EVEN

    def test_even_parity_negative_energy(self):
        """Even parity but negative energy: Odd wins (energy fails)."""
        g = make_weighted_parity_game(
            [(0, 0, 2), (1, 0, 2)],  # Even parity
            [(0, 1, -5), (1, 0, -5)]  # Net negative: energy fails
        )
        r = solve_energy_parity(g)
        assert r.winner[0] == Player.ODD
        assert r.winner[1] == Player.ODD

    def test_mixed_priorities_with_choice(self):
        """Even can choose between parity-winning and parity-losing paths."""
        # Node 0 (Even, prio 2): can go to 1 or 2
        # Node 1 (Even, prio 2): loops back to 0 with +1 (good cycle: prio 2, positive energy)
        # Node 2 (Even, prio 1): loops back to 0 with +1 (bad cycle: prio 1, odd wins parity)
        g = make_weighted_parity_game(
            [(0, 0, 2), (1, 0, 2), (2, 0, 1)],
            [(0, 1, 1), (0, 2, 1), (1, 0, 1), (2, 0, 1)]
        )
        r = solve_energy_parity(g)
        assert r.winner[0] == Player.EVEN  # Even can stay in {0,1} cycle

    def test_energy_parity_with_required_credit(self):
        """Even wins parity but needs initial credit for energy."""
        g = make_weighted_parity_game(
            [(0, 0, 2), (1, 0, 2)],
            [(0, 1, -3), (1, 0, 5)]  # Net +2, but need credit 3 at start
        )
        r = solve_energy_parity(g)
        assert r.winner[0] == Player.EVEN
        assert r.min_credit[0] == 3


# ===========================================================================
# Section 12: V076 ParityGame to WeightedGame conversion
# ===========================================================================

class TestParityConversion:
    def test_parity_to_weighted_default(self):
        pg = ParityGame()
        pg.add_node(0, Player.EVEN, 2)
        pg.add_node(1, Player.ODD, 1)
        pg.add_edge(0, 1)
        pg.add_edge(1, 0)

        wg = parity_game_to_weighted(pg)
        assert len(wg.nodes) == 2
        assert wg.weight[(0, 1)] == 0
        assert wg.weight[(1, 0)] == 0

    def test_parity_to_weighted_custom_fn(self):
        pg = ParityGame()
        pg.add_node(0, Player.EVEN, 2)
        pg.add_node(1, Player.ODD, 3)
        pg.add_edge(0, 1)
        pg.add_edge(1, 0)

        wg = parity_game_to_weighted(pg, weight_fn=lambda s, d, g: g.priority[s])
        assert wg.weight[(0, 1)] == 2
        assert wg.weight[(1, 0)] == 3


# ===========================================================================
# Section 13: Utility functions
# ===========================================================================

class TestUtilities:
    def test_weighted_game_summary(self):
        g = make_weighted_game(
            [(0, 0), (1, 1), (2, 0)],
            [(0, 1, 5), (1, 2, -3), (2, 0, 1)]
        )
        s = weighted_game_summary(g)
        assert "3 nodes" in s
        assert "3 edges" in s
        assert "-3" in s
        assert "5" in s

    def test_empty_game_results(self):
        g = WeightedGame()
        er = solve_energy(g)
        assert er.min_credit == {}
        mr = solve_mean_payoff(g)
        assert mr.value == {}


# ===========================================================================
# Section 14: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_even_node_self_loop_zero(self):
        g = make_weighted_game([(0, 0)], [(0, 0, 0)])
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN
        assert r.min_credit[0] == 0

    def test_single_odd_node_positive_self_loop(self):
        """Odd node with positive self-loop: Odd must stay, energy goes up, Even wins."""
        g = make_weighted_game([(0, 1)], [(0, 0, 5)])
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN

    def test_single_odd_node_negative_self_loop(self):
        """Odd node with negative self-loop: Odd stuck, energy drops, Odd wins."""
        g = make_weighted_game([(0, 1)], [(0, 0, -1)])
        r = solve_energy(g)
        assert r.winner[0] == Player.ODD

    def test_disconnected_components(self):
        """Two separate components with different outcomes."""
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1, 3), (1, 0, -1),  # Component 1: net +2
             (2, 3, -3), (3, 2, 1)]  # Component 2: net -2
        )
        er = solve_energy(g)
        assert er.winner[0] == Player.EVEN
        assert er.winner[2] == Player.ODD

        mr = solve_mean_payoff(g)
        assert mr.value[0] == pytest.approx(1.0, abs=0.1)
        assert mr.value[2] == pytest.approx(-1.0, abs=0.1)

    def test_custom_bound(self):
        """Custom energy bound."""
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -2), (1, 0, 5)]
        )
        r = solve_energy(g, bound=100)
        assert r.winner[0] == Player.EVEN


# ===========================================================================
# Section 15: Larger games
# ===========================================================================

class TestLargerGames:
    def test_ring_game_positive(self):
        """Ring of 5 nodes, all positive weights."""
        nodes = [(i, 0 if i % 2 == 0 else 1) for i in range(5)]
        edges = [(i, (i + 1) % 5, 1) for i in range(5)]
        g = make_weighted_game(nodes, edges)
        r = solve_energy(g)
        for i in range(5):
            assert r.winner[i] == Player.EVEN

    def test_ring_game_alternating(self):
        """Ring with alternating +3, -1 weights."""
        nodes = [(i, 0) for i in range(4)]
        edges = [(0, 1, 3), (1, 2, -1), (2, 3, 3), (3, 0, -1)]
        g = make_weighted_game(nodes, edges)
        r = solve_energy(g)
        # Net per cycle: 3-1+3-1 = 4 > 0
        for i in range(4):
            assert r.winner[i] == Player.EVEN

        mr = solve_mean_payoff(g)
        for i in range(4):
            assert mr.value[i] == pytest.approx(1.0, abs=0.1)

    def test_star_game(self):
        """Star: center (Even) connected to 4 leaves, each with self-loop."""
        nodes = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        edges = [
            (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0),
            (1, 0, 5), (2, 0, -3), (3, 0, 10), (4, 0, -1)
        ]
        g = make_weighted_game(nodes, edges)
        r = solve_energy(g)
        assert r.winner[0] == Player.EVEN  # Even picks the positive cycles


# ===========================================================================
# Section 16: Mean-Payoff with multiple SCCs
# ===========================================================================

class TestMeanPayoffMultipleSCCs:
    def test_chain_of_sccs(self):
        """Transient node leading to two possible SCCs."""
        # Node 0 (Even) -> 1 or 2
        # 1 <-> 3: mean = (2+(-1))/2 = 0.5
        # 2 <-> 4: mean = (-2+1)/2 = -0.5
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            [(0, 1, 0), (0, 2, 0), (1, 3, 2), (3, 1, -1), (2, 4, -2), (4, 2, 1)]
        )
        r = solve_mean_payoff(g)
        # Even at node 0 should pick path to SCC {1,3} with value 0.5
        assert r.value[0] == pytest.approx(0.5, abs=0.1)
        assert r.winner[0] == Player.EVEN


# ===========================================================================
# Section 17: Energy-Parity edge cases
# ===========================================================================

class TestEnergyParityEdgeCases:
    def test_empty_game(self):
        g = WeightedParityGame()
        r = solve_energy_parity(g)
        assert r.min_credit == {}
        assert r.winner == {}

    def test_single_node_even_parity_positive(self):
        g = make_weighted_parity_game([(0, 0, 2)], [(0, 0, 1)])
        r = solve_energy_parity(g)
        assert r.winner[0] == Player.EVEN

    def test_single_node_odd_parity(self):
        g = make_weighted_parity_game([(0, 0, 1)], [(0, 0, 5)])
        r = solve_energy_parity(g)
        assert r.winner[0] == Player.ODD  # Parity 1 is odd -> Odd wins


# ===========================================================================
# Section 18: Tarjan SCC correctness
# ===========================================================================

class TestTarjanSCC:
    def test_single_scc(self):
        from energy_mean_payoff import _tarjan_scc
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
        )
        sccs = _tarjan_scc(g)
        assert len(sccs) == 1
        assert sccs[0] == {0, 1, 2}

    def test_two_sccs(self):
        from energy_mean_payoff import _tarjan_scc
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1, 0), (1, 0, 0), (2, 3, 0), (3, 2, 0)]
        )
        sccs = _tarjan_scc(g)
        assert len(sccs) == 2
        scc_sets = [s for s in sccs]
        assert {0, 1} in scc_sets
        assert {2, 3} in scc_sets

    def test_dag(self):
        from energy_mean_payoff import _tarjan_scc
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1, 0), (1, 2, 0), (2, 2, 0)]  # 2 has self-loop
        )
        sccs = _tarjan_scc(g)
        # Node 0 and 1 are singletons, node 2 is SCC (self-loop)
        assert len(sccs) == 3


# ===========================================================================
# Section 19: Combined analysis and comparison
# ===========================================================================

class TestCombinedAnalysis:
    def test_compare_energy_mean_payoff_positive(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, 3), (1, 0, 1)]
        )
        comp = compare_energy_mean_payoff(g)
        assert comp['connection_holds']
        assert comp['agreement'] == 2

    def test_compare_energy_mean_payoff_negative(self):
        g = make_weighted_game(
            [(0, 0), (1, 0)],
            [(0, 1, -5), (1, 0, -1)]
        )
        comp = compare_energy_mean_payoff(g)
        assert comp['connection_holds']

    def test_compare_mixed(self):
        """Disconnected game: some nodes winning, some losing."""
        g = make_weighted_game(
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1, 5), (1, 0, -1),
             (2, 3, -5), (3, 2, 1)]
        )
        comp = compare_energy_mean_payoff(g)
        assert comp['connection_holds']
        assert comp['n_nodes'] == 4
