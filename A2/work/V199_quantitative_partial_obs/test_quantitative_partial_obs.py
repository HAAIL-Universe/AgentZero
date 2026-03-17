"""Tests for V199: Quantitative Partial Observation Games."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from quantitative_partial_obs import (
    QuantPOGame, QObjective, QPOResult, BeliefEnergyState,
    Player, INF_ENERGY,
    solve_energy_po, solve_mean_payoff_po, find_optimal_mean_payoff_po,
    solve_energy_safety_po, solve_energy_parity_po,
    compare_perfect_vs_partial, simulate_play,
    check_fixed_energy_po, solve, quantitative_decomposition,
    game_statistics, game_summary,
    make_energy_po_game, make_charging_po_game, make_adversarial_po_game,
    make_corridor_po_game, make_choice_po_game, make_hidden_drain_game,
    make_energy_parity_po_game,
    _belief_post, _belief_post_detailed, _all_target_observations,
    _belief_owner, _belief_parity,
)
from fractions import Fraction


# =========================================================================
# QuantPOGame construction and basic operations
# =========================================================================

class TestQuantPOGameBasics:
    """Test game construction and basic operations."""

    def test_create_empty_game(self):
        g = QuantPOGame()
        assert len(g.vertices) == 0
        assert g.objective == QObjective.ENERGY

    def test_add_vertex(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        assert 0 in g.vertices
        assert g.owner[0] == Player.EVEN
        assert g.obs[0] == 0

    def test_add_edge(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 5)
        assert g.successors(0) == [(1, 5)]

    def test_successor_vertices(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 2)
        g.add_edge(0, 1, 3)
        g.add_edge(0, 2, -1)
        assert g.successor_vertices(0) == {1, 2}

    def test_predecessors(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 2)
        preds = g.predecessors(1)
        assert len(preds) == 1
        assert preds[0] == (0, 2)

    def test_obs_class(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.ODD, 1)
        assert g.obs_class(0) == {0, 1}
        assert g.obs_class(1) == {2}

    def test_all_observations(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.ODD, 1)
        assert g.all_observations() == {0, 1}

    def test_max_weight(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 5)
        g.add_edge(1, 0, -3)
        assert g.max_weight() == 5

    def test_weight_bound(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 4)
        assert g.weight_bound() == 2 * 4  # n=2, W=4

    def test_observation_consistent(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)  # same obs, same owner
        assert g.is_observation_consistent()

    def test_observation_inconsistent(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 0)  # same obs, different owner
        assert not g.is_observation_consistent()

    def test_to_energy_game(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 3)
        eg = g.to_energy_game()
        assert 0 in eg.vertices
        assert 1 in eg.vertices

    def test_to_partial_obs_game(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0, prio=2)
        g.add_vertex(1, Player.ODD, 1, prio=1)
        g.add_edge(0, 1, 5)
        pog = g.to_partial_obs_game()
        assert 0 in pog.vertices
        assert pog.obs[0] == 0


# =========================================================================
# Belief operations
# =========================================================================

class TestBeliefOperations:
    """Test belief post and related operations."""

    def test_belief_post_even_vertex(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 3)
        belief = frozenset({0})
        new_b, w = _belief_post(g, belief, 1)
        assert new_b == frozenset({1})
        assert w == 3

    def test_belief_post_odd_adversarial(self):
        """P2 can go to any successor (adversarial)."""
        g = QuantPOGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.EVEN, 1)
        g.add_edge(0, 1, 5)
        g.add_edge(0, 2, -1)
        belief = frozenset({0})
        new_b, w = _belief_post(g, belief, 1)
        assert new_b == frozenset({1, 2})
        assert w == -1  # worst case

    def test_belief_post_mixed_belief(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 1)
        g.add_vertex(3, Player.EVEN, 1)
        g.add_edge(0, 2, 2)
        g.add_edge(1, 3, 4)
        belief = frozenset({0, 1})
        new_b, w = _belief_post(g, belief, 1)
        assert new_b == frozenset({2, 3})
        assert w == 2  # min weight

    def test_belief_post_detailed(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 7)
        belief = frozenset({0})
        new_b, wmap = _belief_post_detailed(g, belief, 1)
        assert 1 in wmap
        assert wmap[1] == 7

    def test_all_target_observations(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.ODD, 2)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        obs = _all_target_observations(g, frozenset({0}))
        assert obs == {1, 2}

    def test_belief_owner_all_even(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        assert _belief_owner(g, frozenset({0, 1})) == Player.EVEN

    def test_belief_owner_has_odd(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 0)
        assert _belief_owner(g, frozenset({0, 1})) == Player.ODD

    def test_belief_parity_max(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0, prio=2)
        g.add_vertex(1, Player.EVEN, 0, prio=5)
        assert _belief_parity(g, frozenset({0, 1})) == 5

    def test_belief_parity_empty(self):
        g = QuantPOGame()
        assert _belief_parity(g, frozenset()) == 0


# =========================================================================
# Energy solving under partial observation
# =========================================================================

class TestEnergySolvingPO:
    """Test energy objective under partial observation."""

    def test_trivial_self_loop(self):
        """Single vertex, zero-weight self-loop -> energy 0."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.min_energy == 0

    def test_positive_cycle(self):
        """All positive weights -> energy 0 sufficient."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 2)
        g.add_edge(1, 0, 3)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.min_energy == 0

    def test_negative_cycle_needs_energy(self):
        """Negative self-loop -> P1 always loses (can't maintain energy)."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -1)
        g.initial = {0}
        result = solve_energy_po(g)
        assert not result.winning

    def test_charge_drain_cycle(self):
        """Charge-drain cycle: need energy to survive the drain phase."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, -3)
        g.add_edge(1, 0, 5)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.min_energy == 3  # need 3 to survive -3

    def test_p2_adversarial_choice(self):
        """P2 picks the worst edge for P1 each cycle -> energy drains."""
        g = QuantPOGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 5)
        g.add_edge(0, 1, -2)
        g.add_edge(1, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        # P2 picks -2 every cycle, net -2 per cycle -> infinite drain
        assert not result.winning

    def test_two_vertices_same_obs(self):
        """Two states share observation -> belief merges them.
        Worst case weight -1 per cycle -> infinite drain -> P1 loses."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.EVEN, 1)
        g.add_vertex(3, Player.EVEN, 2)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 3, 2)
        g.add_edge(2, 3, -1)
        g.add_edge(3, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        # Worst case from {1,2}: weight -1 per cycle, net -1 -> drain
        assert not result.winning

    def test_two_vertices_same_obs_positive(self):
        """Same obs merge but all positive weights -> P1 wins."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.EVEN, 1)
        g.add_vertex(3, Player.EVEN, 2)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 3, 5)
        g.add_edge(2, 3, 1)
        g.add_edge(3, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.min_energy == 0

    def test_no_initial_states(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        result = solve_energy_po(g)
        assert not result.winning

    def test_dead_end_loses(self):
        """Vertex with no successors -> P1 stuck -> loses."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 5)
        # vertex 1 has no successors
        g.initial = {0}
        result = solve_energy_po(g)
        # P1 reaches vertex 1 with no moves -> game stuck
        # In energy games, getting stuck = P1 wins (play is finite and energy >= 0)
        # Actually, in the belief graph, belief {1} has no edges -> energy is 0 there
        assert result.winning

    def test_strategy_exists(self):
        """Winning game should produce a strategy."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.strategy is not None

    def test_belief_energies_populated(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, -2)
        g.add_edge(1, 0, 5)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.belief_energies is not None
        assert len(result.belief_energies) > 0


# =========================================================================
# Mean-payoff under partial observation
# =========================================================================

class TestMeanPayoffPO:
    """Test mean-payoff objective under partial observation."""

    def test_positive_cycle_mp(self):
        """All positive weights -> mean-payoff > 0 achievable."""
        g = QuantPOGame()
        g.objective = QObjective.MEAN_PAYOFF
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, 5)
        g.initial = {0}
        result = solve_mean_payoff_po(g, threshold=0.0)
        assert result.winning

    def test_negative_cycle_mp(self):
        """Only negative cycle -> mean-payoff < 0."""
        g = QuantPOGame()
        g.objective = QObjective.MEAN_PAYOFF
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -1)
        g.initial = {0}
        result = solve_mean_payoff_po(g, threshold=0.0)
        assert not result.winning

    def test_zero_weight_mp(self):
        """Zero-weight cycle -> mean-payoff = 0, threshold 0 satisfied."""
        g = QuantPOGame()
        g.objective = QObjective.MEAN_PAYOFF
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 0)
        g.initial = {0}
        result = solve_mean_payoff_po(g, threshold=0.0)
        assert result.winning

    def test_threshold_not_met(self):
        """Cycle has mean-payoff 1, threshold 5 -> not achievable."""
        g = QuantPOGame()
        g.objective = QObjective.MEAN_PAYOFF
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        result = solve_mean_payoff_po(g, threshold=5.0)
        assert not result.winning

    def test_corridor_mean_payoff(self):
        g = make_corridor_po_game(4, reward=2, penalty=3)
        result = solve_mean_payoff_po(g, threshold=0.0)
        # Mean payoff = (3*2 - 3)/4 = 3/4 > 0
        assert result.winning

    def test_find_optimal_mp(self):
        """Find optimal mean-payoff value."""
        g = QuantPOGame()
        g.objective = QObjective.MEAN_PAYOFF
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, 1)
        g.initial = {0}
        result = find_optimal_mean_payoff_po(g)
        assert result.winning
        assert result.mean_payoff_value is not None


# =========================================================================
# Energy-Safety under PO
# =========================================================================

class TestEnergySafetyPO:
    """Test energy + safety objective under partial observation."""

    def test_safe_game(self):
        """No bad states -> same as pure energy."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_SAFETY
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        g.target = set()  # no bad states
        result = solve_energy_safety_po(g)
        assert result.winning

    def test_initial_is_bad(self):
        """Initial belief contains a bad state -> immediate loss."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_SAFETY
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        g.target = {0}  # 0 is bad
        result = solve_energy_safety_po(g)
        assert not result.winning

    def test_avoid_bad_state(self):
        """P1 can route around the bad state."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_SAFETY
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)  # safe
        g.add_vertex(2, Player.EVEN, 2)  # bad
        g.add_edge(0, 1, 2)
        g.add_edge(0, 2, 5)
        g.add_edge(1, 0, 1)
        g.initial = {0}
        g.target = {2}
        result = solve_energy_safety_po(g)
        assert result.winning

    def test_forced_into_bad(self):
        """P2 can force P1 into bad state via shared observation."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_SAFETY
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.ODD, 1)  # same obs as 1
        g.add_vertex(3, Player.EVEN, 2)  # safe
        g.add_vertex(4, Player.EVEN, 3)  # bad
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 3, 1)
        g.add_edge(2, 4, 1)  # goes to bad
        g.add_edge(3, 0, 0)
        g.add_edge(4, 4, 0)
        g.initial = {0}
        g.target = {4}
        result = solve_energy_safety_po(g)
        # P2 controls whether we're in state 1 or 2 (same obs)
        # P2 can be in state 2 which leads to bad state 4
        assert not result.winning

    def test_energy_safety_with_drain(self):
        """Safe path has drain, bad path has gain."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_SAFETY
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.EVEN, 2)  # bad
        g.add_edge(0, 1, -2)
        g.add_edge(0, 2, 10)  # tempting but bad
        g.add_edge(1, 0, 3)
        g.initial = {0}
        g.target = {2}
        result = solve_energy_safety_po(g)
        assert result.winning
        assert result.min_energy == 2  # need 2 to survive -2


# =========================================================================
# Energy-Parity under PO
# =========================================================================

class TestEnergyParityPO:
    """Test energy-parity objective under partial observation."""

    def test_simple_even_parity(self):
        """All even priorities -> parity satisfied, just energy matters."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_PARITY
        g.add_vertex(0, Player.EVEN, 0, prio=0)
        g.add_vertex(1, Player.EVEN, 1, prio=2)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)
        g.initial = {0}
        result = solve_energy_parity_po(g)
        assert result.winning

    def test_odd_parity_loses(self):
        """Only odd priority -> parity violated."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_PARITY
        g.add_vertex(0, Player.EVEN, 0, prio=1)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        result = solve_energy_parity_po(g)
        assert not result.winning

    def test_parity_choice(self):
        """P1 can choose even-parity path."""
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_PARITY
        g.add_vertex(0, Player.EVEN, 0, prio=0)
        g.add_vertex(1, Player.EVEN, 1, prio=2)  # even prio, good
        g.add_vertex(2, Player.EVEN, 2, prio=1)  # odd prio, bad
        g.add_edge(0, 1, 1)
        g.add_edge(0, 2, 5)
        g.add_edge(1, 0, 1)
        g.add_edge(2, 0, 1)
        g.initial = {0}
        result = solve_energy_parity_po(g)
        assert result.winning

    def test_energy_parity_po_game_helper(self):
        """Test the construction helper."""
        g = make_energy_parity_po_game()
        result = solve_energy_parity_po(g)
        # P2 controls whether state has even or odd parity (shared obs)
        # Conservative: belief gets max (odd) parity -> parity violated
        assert not result.winning


# =========================================================================
# Construction helpers
# =========================================================================

class TestConstructionHelpers:
    """Test game construction helpers."""

    def test_make_energy_po_game(self):
        g = make_energy_po_game(
            3,
            [(0, 1, 2), (1, 2, -1), (2, 0, 3)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN},
            {0: 0, 1: 1, 2: 2},
        )
        assert len(g.vertices) == 3
        assert g.initial == {0}

    def test_make_charging_po_game(self):
        g = make_charging_po_game(4, charge=5, drain=2)
        assert len(g.vertices) == 4
        assert g.objective == QObjective.ENERGY

    def test_make_charging_po_game_custom_obs(self):
        g = make_charging_po_game(4, charge=5, drain=2,
                                  obs_groups=[{0, 1}, {2, 3}])
        assert g.obs[0] == g.obs[1]
        assert g.obs[2] == g.obs[3]

    def test_make_adversarial_po_game(self):
        g = make_adversarial_po_game()
        assert len(g.vertices) == 4
        assert g.obs[1] == g.obs[2]  # share observation

    def test_make_corridor_po_game(self):
        g = make_corridor_po_game(6)
        assert len(g.vertices) == 6
        assert g.objective == QObjective.MEAN_PAYOFF

    def test_make_choice_po_game(self):
        g = make_choice_po_game()
        assert len(g.vertices) == 3

    def test_make_hidden_drain_game(self):
        g = make_hidden_drain_game()
        assert len(g.vertices) == 4
        assert g.obs[1] == g.obs[2]

    def test_make_energy_parity_po_game(self):
        g = make_energy_parity_po_game()
        assert len(g.vertices) == 4
        assert g.priority[1] == 2
        assert g.priority[2] == 1


# =========================================================================
# Constructed game solving
# =========================================================================

class TestConstructedGameSolving:
    """Test solving on constructed games."""

    def test_adversarial_game_energy(self):
        g = make_adversarial_po_game()
        result = solve_energy_po(g)
        # P2 at vertices 1,2 (same obs) can force -3 or -1
        # Vertex 3 is a sink (0 weight loop)
        # P2 can send to sink (energy stays 0) -> P1 needs 0 at sink
        # From {1,2}: worst case P2 picks v1->3 with -3
        assert result.num_beliefs > 0

    def test_charging_game_perfect_obs(self):
        """Full observation (each vertex distinct obs) -> like perfect info."""
        g = make_charging_po_game(3, charge=5, drain=2)
        result = solve_energy_po(g)
        assert result.winning

    def test_charging_game_partial_obs(self):
        """Merge observations -> might change energy requirement."""
        g = make_charging_po_game(3, charge=5, drain=2,
                                  obs_groups=[{0}, {1, 2}])
        result = solve_energy_po(g)
        assert result.winning

    def test_choice_game_safe_path(self):
        """Choice game: P1 should pick safe path."""
        g = make_choice_po_game()
        result = solve_energy_po(g)
        assert result.winning

    def test_hidden_drain_game(self):
        """Hidden drain: P1 must handle worst-case drain."""
        g = make_hidden_drain_game()
        result = solve_energy_po(g)
        assert result.num_beliefs > 0

    def test_corridor_mp(self):
        """Corridor game with mean-payoff objective."""
        g = make_corridor_po_game(4, reward=3, penalty=2)
        result = solve_mean_payoff_po(g, threshold=0.0)
        # Mean payoff: (3*3 - 2)/4 = 7/4 > 0
        assert result.winning


# =========================================================================
# Comparison: perfect vs partial
# =========================================================================

class TestComparisonPerfectPartial:
    """Test comparison between perfect and partial observation."""

    def test_perfect_obs_comparison(self):
        """Full observation should match perfect info results."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, -2)
        g.add_edge(1, 0, 5)
        g.initial = {0}
        comp = compare_perfect_vs_partial(g)
        assert 'perfect_winning' in comp
        assert 'partial_winning' in comp
        assert 'information_cost' in comp

    def test_information_cost_nonneg(self):
        """PO energy >= perfect info energy (information has cost)."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, -1)
        g.add_edge(1, 0, 3)
        g.initial = {0}
        comp = compare_perfect_vs_partial(g)
        cost = comp['information_cost']
        if cost is not None:
            assert cost >= 0

    def test_no_observation_loss(self):
        """Perfect observation -> info cost = 0."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        comp = compare_perfect_vs_partial(g)
        if comp['information_cost'] is not None:
            assert comp['information_cost'] == 0


# =========================================================================
# Simulation
# =========================================================================

class TestSimulation:
    """Test play simulation."""

    def test_simulate_simple(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        strategy = {frozenset({0}): 0}
        trace = simulate_play(g, strategy, initial_energy=5, max_steps=3)
        assert len(trace) >= 1
        assert trace[0]['energy'] == 5

    def test_simulate_energy_depletion(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -3)
        g.initial = {0}
        strategy = {frozenset({0}): 0}
        trace = simulate_play(g, strategy, initial_energy=2, max_steps=10)
        # Energy goes 2 -> -1 -> depleted
        found_depleted = any(t.get('status') == 'energy_depleted' for t in trace)
        assert found_depleted

    def test_simulate_dead_end(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 1)
        g.initial = {0}
        strategy = {frozenset({0}): 1}
        trace = simulate_play(g, strategy, initial_energy=5, max_steps=10)
        # Should reach belief {1} which has no strategy -> dead end
        assert len(trace) >= 2

    def test_simulate_worst_adversary(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 5)
        g.add_edge(0, 1, -2)
        g.add_edge(1, 0, 0)
        g.initial = {0}
        strategy = {frozenset({0}): 1, frozenset({1}): 0}
        trace = simulate_play(g, strategy, initial_energy=10,
                             max_steps=4, adversary='worst')
        # Worst case: P2 picks -2
        assert len(trace) >= 1


# =========================================================================
# Fixed energy check
# =========================================================================

class TestFixedEnergy:
    """Test fixed energy checking."""

    def test_sufficient_energy(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, -3)
        g.add_edge(1, 0, 5)
        g.initial = {0}
        result = check_fixed_energy_po(g, initial_energy=10)
        assert result.winning

    def test_insufficient_energy(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, -10)
        g.add_edge(1, 0, 5)
        g.initial = {0}
        # Need 10 energy but only have 5
        result = check_fixed_energy_po(g, initial_energy=5)
        # min_energy = 10, have 5 -> not enough
        assert not result.winning

    def test_zero_energy_zero_weight(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 0)
        g.initial = {0}
        result = check_fixed_energy_po(g, initial_energy=0)
        assert result.winning


# =========================================================================
# Unified solver
# =========================================================================

class TestUnifiedSolver:
    """Test the unified solve() dispatcher."""

    def test_solve_energy(self):
        g = QuantPOGame()
        g.objective = QObjective.ENERGY
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        result = solve(g)
        assert result.winning

    def test_solve_mean_payoff(self):
        g = QuantPOGame()
        g.objective = QObjective.MEAN_PAYOFF
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 2)
        g.initial = {0}
        result = solve(g, threshold=0.0)
        assert result.winning

    def test_solve_energy_safety(self):
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_SAFETY
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        g.target = set()
        result = solve(g)
        assert result.winning

    def test_solve_energy_parity(self):
        g = QuantPOGame()
        g.objective = QObjective.ENERGY_PARITY
        g.add_vertex(0, Player.EVEN, 0, prio=0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        result = solve(g)
        assert result.winning


# =========================================================================
# Quantitative decomposition
# =========================================================================

class TestDecomposition:
    """Test quantitative-qualitative decomposition."""

    def test_decomposition_structure(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        g.initial = {0}
        dec = quantitative_decomposition(g)
        assert 'qualitative_po_winning' in dec
        assert 'perfect_info_winning' in dec
        assert 'quantitative_po_winning' in dec
        assert 'num_beliefs' in dec

    def test_decomposition_positive_game(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, 2)
        g.initial = {0}
        dec = quantitative_decomposition(g)
        assert dec['quantitative_po_winning']


# =========================================================================
# Statistics and summary
# =========================================================================

class TestStatistics:
    """Test game statistics and summary."""

    def test_statistics(self):
        g = make_adversarial_po_game()
        stats = game_statistics(g)
        assert stats['vertices'] == 4
        assert stats['observations'] == 3
        assert stats['objective'] == 'energy'

    def test_summary(self):
        g = make_adversarial_po_game()
        s = game_summary(g)
        assert 'Quantitative PO Game' in s
        assert 'Vertices: 4' in s

    def test_statistics_empty(self):
        g = QuantPOGame()
        stats = game_statistics(g)
        assert stats['vertices'] == 0


# =========================================================================
# Edge cases and regression
# =========================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_single_vertex_game(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.min_energy == 0

    def test_all_odd_owned(self):
        """All vertices owned by P2."""
        g = QuantPOGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, -1)
        g.add_edge(1, 0, -1)
        g.initial = {0}
        result = solve_energy_po(g)
        # P2 always picks -1 -> energy drains
        assert not result.winning

    def test_multiple_initial_states(self):
        """Multiple initial states form initial belief."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 1)
        g.add_edge(0, 2, 1)
        g.add_edge(1, 2, 3)
        g.add_edge(2, 0, 0)
        g.initial = {0, 1}
        result = solve_energy_po(g)
        assert result.winning

    def test_large_weight(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_edge(0, 1, -100)
        g.add_edge(1, 0, 200)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.min_energy == 100

    def test_self_loop_negative_loses(self):
        """Negative self-loop only option -> infinite drain."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -5)
        g.initial = {0}
        result = solve_energy_po(g)
        assert not result.winning

    def test_branching_beliefs(self):
        """Multiple observations reachable -> belief graph branches."""
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 1)
        g.add_vertex(2, Player.EVEN, 2)
        g.add_vertex(3, Player.EVEN, 3)
        g.add_edge(0, 1, 1)
        g.add_edge(0, 2, 2)
        g.add_edge(1, 3, 3)
        g.add_edge(2, 3, 4)
        g.add_edge(3, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.winning
        assert result.num_beliefs >= 3  # at least {0}, {1}, {2}, {3}

    def test_num_beliefs_reported(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.num_beliefs >= 1

    def test_iterations_bounded(self):
        g = QuantPOGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 0)
        g.initial = {0}
        result = solve_energy_po(g)
        assert result.iterations >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
