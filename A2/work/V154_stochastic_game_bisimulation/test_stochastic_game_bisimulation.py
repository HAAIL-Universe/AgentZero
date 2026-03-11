"""Tests for V154: Bisimulation for Stochastic Games."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V070_stochastic_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V148_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from stochastic_game_bisimulation import (
    LabeledGame, GameBisimVerdict, GameBisimResult, GameSimResult,
    GameDistanceResult, make_labeled_game, compute_game_bisimulation,
    check_game_bisimilar, game_bisimulation_quotient,
    compute_game_simulation, check_game_simulates,
    compute_game_bisimulation_distance, check_cross_game_bisimulation,
    check_cross_game_bisimilar_states, strategy_bisimulation,
    compare_strategy_bisimulations, compute_reward_bisimulation,
    verify_game_bisimulation_smt, minimize_game,
    compare_game_vs_mdp_bisimulation, analyze_game_bisimulation,
    game_bisimulation_summary, symmetric_game, asymmetric_game,
    owner_mismatch_game,
)
from stochastic_games import Player, StrategyPair


# ============================================================
# Section 1: Basic Partition Refinement
# ============================================================

class TestBasicPartition:
    def test_single_state(self):
        lgame = make_labeled_game(
            n_states=1, owners=[Player.P1],
            action_transitions={0: [("a", [(0, 1.0)])]},
            labels={0: {"start"}},
        )
        result = compute_game_bisimulation(lgame)
        assert len(result.partition) == 1
        assert 0 in result.partition[0]

    def test_identical_states_bisimilar(self):
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P1, Player.CHANCE],
            action_transitions={
                0: [("a", [(2, 1.0)])],
                1: [("a", [(2, 1.0)])],
                2: [("stay", [(2, 1.0)])],
            },
            labels={0: {"start"}, 1: {"start"}, 2: {"end"}},
        )
        result = compute_game_bisimulation(lgame)
        # States 0 and 1 should be in the same block
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] == mapping[1]
        assert mapping[0] != mapping[2]

    def test_different_labels_not_bisimilar(self):
        lgame = make_labeled_game(
            n_states=2,
            owners=[Player.P1, Player.P1],
            action_transitions={
                0: [("a", [(0, 1.0)])],
                1: [("a", [(1, 1.0)])],
            },
            labels={0: {"red"}, 1: {"blue"}},
        )
        result = compute_game_bisimulation(lgame)
        assert len(result.partition) == 2

    def test_different_owners_not_bisimilar(self):
        lgame = owner_mismatch_game()
        result = compute_game_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] != mapping[1]  # P1 vs P2

    def test_iterations_tracked(self):
        lgame = symmetric_game()
        result = compute_game_bisimulation(lgame)
        assert result.statistics['iterations'] >= 1
        assert result.statistics['n_states'] == 4


# ============================================================
# Section 2: Symmetric and Asymmetric Games
# ============================================================

class TestSymmetricAsymmetric:
    def test_symmetric_game_bisimilar(self):
        lgame = symmetric_game()
        result = compute_game_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] == mapping[1]  # Symmetric P1 states

    def test_asymmetric_game_not_bisimilar(self):
        lgame = asymmetric_game()
        result = compute_game_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] != mapping[1]  # Different action counts

    def test_all_different(self):
        """All states have unique structure."""
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P2, Player.CHANCE],
            action_transitions={
                0: [("a", [(1, 1.0)])],
                1: [("b", [(2, 1.0)])],
                2: [("c", [(0, 1.0)])],
            },
            labels={0: {"a"}, 1: {"b"}, 2: {"c"}},
        )
        result = compute_game_bisimulation(lgame)
        assert len(result.partition) == 3


# ============================================================
# Section 3: Check Bisimilar States
# ============================================================

class TestCheckBisimilar:
    def test_bisimilar_pair(self):
        lgame = symmetric_game()
        result = check_game_bisimilar(lgame, 0, 1)
        assert result.verdict == GameBisimVerdict.BISIMILAR
        assert result.witness is not None

    def test_not_bisimilar_pair(self):
        lgame = asymmetric_game()
        result = check_game_bisimilar(lgame, 0, 1)
        assert result.verdict == GameBisimVerdict.NOT_BISIMILAR
        assert result.witness is not None

    def test_self_bisimilar(self):
        lgame = symmetric_game()
        result = check_game_bisimilar(lgame, 0, 0)
        assert result.verdict == GameBisimVerdict.BISIMILAR

    def test_different_owner_witness(self):
        lgame = owner_mismatch_game()
        result = check_game_bisimilar(lgame, 0, 1)
        assert result.verdict == GameBisimVerdict.NOT_BISIMILAR
        assert "owner" in result.witness.lower()


# ============================================================
# Section 4: Quotient Game Construction
# ============================================================

class TestQuotient:
    def test_quotient_reduces_size(self):
        lgame = symmetric_game()
        quotient, result = game_bisimulation_quotient(lgame)
        assert quotient.game.n_states <= lgame.game.n_states

    def test_quotient_preserves_structure(self):
        lgame = symmetric_game()
        quotient, result = game_bisimulation_quotient(lgame)
        # Quotient should have valid transitions
        for s in range(quotient.game.n_states):
            for ai in range(len(quotient.game.actions[s])):
                trans = quotient.game.transition[s][ai]
                total = sum(trans)
                assert abs(total - 1.0) < 1e-6, f"Transitions from state {s} action {ai} sum to {total}"

    def test_quotient_preserves_owners(self):
        lgame = symmetric_game()
        quotient, result = game_bisimulation_quotient(lgame)
        # Each block's owner should match representative
        for bi, block in enumerate(result.partition):
            rep = min(block)
            for s in block:
                assert lgame.game.owners[s] == lgame.game.owners[rep]

    def test_trivial_quotient(self):
        """Game with no bisimilar states -> quotient same size."""
        lgame = make_labeled_game(
            n_states=2,
            owners=[Player.P1, Player.P2],
            action_transitions={
                0: [("a", [(1, 1.0)])],
                1: [("b", [(0, 1.0)])],
            },
            labels={0: {"x"}, 1: {"y"}},
        )
        quotient, result = game_bisimulation_quotient(lgame)
        assert quotient.game.n_states == 2


# ============================================================
# Section 5: Simulation Preorder
# ============================================================

class TestSimulation:
    def test_reflexive(self):
        lgame = symmetric_game()
        result = compute_game_simulation(lgame)
        for s in range(lgame.game.n_states):
            assert (s, s) in result.relation

    def test_bisimilar_implies_mutual_simulation(self):
        lgame = symmetric_game()
        sim = compute_game_simulation(lgame)
        bisim = compute_game_bisimulation(lgame)
        # Find bisimilar pairs
        for block in bisim.partition:
            states = sorted(block)
            for i, s in enumerate(states):
                for t in states[i + 1:]:
                    assert (s, t) in sim.relation
                    assert (t, s) in sim.relation

    def test_simulation_not_symmetric(self):
        """A state with more distinct actions simulates one with fewer, but not vice versa."""
        lgame = make_labeled_game(
            n_states=4,
            owners=[Player.P1, Player.P1, Player.CHANCE, Player.CHANCE],
            action_transitions={
                0: [("a", [(2, 1.0)]), ("b", [(3, 1.0)])],  # Can go to 2 or 3
                1: [("a", [(2, 1.0)])],  # Can only go to 2
                2: [("stay", [(2, 1.0)])],
                3: [("stay", [(3, 1.0)])],
            },
            labels={0: {"start"}, 1: {"start"}, 2: {"win"}, 3: {"lose"}},
        )
        result = compute_game_simulation(lgame)
        assert (0, 1) in result.relation  # 0 has action matching 1's single action
        # 1 cannot match 0's action "b" (goes to state 3, "lose" label)
        assert (1, 0) not in result.relation

    def test_check_simulates(self):
        lgame = symmetric_game()
        result = check_game_simulates(lgame, 0, 1)
        assert result.verdict == GameBisimVerdict.SIMULATES


# ============================================================
# Section 6: Bisimulation Distance
# ============================================================

class TestDistance:
    def test_identical_states_zero_distance(self):
        lgame = symmetric_game()
        result = compute_game_bisimulation_distance(lgame)
        assert result.distances[0][1] < 1e-6  # Bisimilar -> distance 0

    def test_different_owners_max_distance(self):
        lgame = owner_mismatch_game()
        result = compute_game_bisimulation_distance(lgame)
        assert result.distances[0][1] == 1.0  # Different owners -> distance 1

    def test_self_distance_zero(self):
        lgame = symmetric_game()
        result = compute_game_bisimulation_distance(lgame)
        for s in range(lgame.game.n_states):
            assert result.distances[s][s] == 0.0

    def test_symmetric(self):
        lgame = symmetric_game()
        result = compute_game_bisimulation_distance(lgame)
        n = lgame.game.n_states
        for s in range(n):
            for t in range(n):
                assert abs(result.distances[s][t] - result.distances[t][s]) < 1e-10

    def test_bisimilar_pairs_detected(self):
        lgame = symmetric_game()
        result = compute_game_bisimulation_distance(lgame)
        assert frozenset({0, 1}) in result.bisimilar_pairs


# ============================================================
# Section 7: Cross-System Bisimulation
# ============================================================

class TestCrossSystem:
    def test_identical_games_bisimilar(self):
        lgame1 = make_labeled_game(
            n_states=2,
            owners=[Player.P1, Player.CHANCE],
            action_transitions={
                0: [("a", [(1, 1.0)])],
                1: [("stay", [(1, 1.0)])],
            },
            labels={0: {"start"}, 1: {"end"}},
        )
        lgame2 = make_labeled_game(
            n_states=2,
            owners=[Player.P1, Player.CHANCE],
            action_transitions={
                0: [("a", [(1, 1.0)])],
                1: [("stay", [(1, 1.0)])],
            },
            labels={0: {"start"}, 1: {"end"}},
        )
        result = check_cross_game_bisimilar_states(lgame1, 0, lgame2, 0)
        assert result.verdict == GameBisimVerdict.BISIMILAR

    def test_different_games_not_bisimilar(self):
        lgame1 = make_labeled_game(
            n_states=2,
            owners=[Player.P1, Player.CHANCE],
            action_transitions={
                0: [("a", [(1, 1.0)])],
                1: [("stay", [(1, 1.0)])],
            },
            labels={0: {"start"}, 1: {"end"}},
        )
        lgame2 = make_labeled_game(
            n_states=2,
            owners=[Player.P2, Player.CHANCE],  # Different owner
            action_transitions={
                0: [("a", [(1, 1.0)])],
                1: [("stay", [(1, 1.0)])],
            },
            labels={0: {"start"}, 1: {"end"}},
        )
        result = check_cross_game_bisimilar_states(lgame1, 0, lgame2, 0)
        assert result.verdict == GameBisimVerdict.NOT_BISIMILAR

    def test_cross_system_statistics(self):
        lgame1 = make_labeled_game(
            n_states=2, owners=[Player.P1, Player.CHANCE],
            action_transitions={0: [("a", [(1, 1.0)])], 1: [("stay", [(1, 1.0)])]},
            labels={0: {"s"}, 1: {"e"}},
        )
        lgame2 = make_labeled_game(
            n_states=3, owners=[Player.P1, Player.P1, Player.CHANCE],
            action_transitions={0: [("a", [(2, 1.0)])], 1: [("a", [(2, 1.0)])], 2: [("stay", [(2, 1.0)])]},
            labels={0: {"s"}, 1: {"s"}, 2: {"e"}},
        )
        result = check_cross_game_bisimulation(lgame1, lgame2)
        assert result.statistics['game1_states'] == 2
        assert result.statistics['game2_states'] == 3


# ============================================================
# Section 8: Strategy-Induced Bisimulation
# ============================================================

class TestStrategyBisimulation:
    def test_strategy_induces_mc(self):
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P2, Player.CHANCE],
            action_transitions={
                0: [("a", [(1, 0.5), (2, 0.5)]), ("b", [(1, 1.0)])],
                1: [("c", [(0, 0.5), (2, 0.5)]), ("d", [(2, 1.0)])],
                2: [("stay", [(2, 1.0)])],
            },
            labels={0: {"x"}, 1: {"x"}, 2: {"end"}},
        )
        strat = StrategyPair(p1_strategy={0: 0}, p2_strategy={1: 0})
        result = strategy_bisimulation(lgame, strat)
        assert result.partition is not None
        assert len(result.partition) >= 1

    def test_compare_strategies(self):
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P2, Player.CHANCE],
            action_transitions={
                0: [("a", [(1, 0.5), (2, 0.5)]), ("b", [(2, 1.0)])],
                1: [("c", [(0, 1.0)]), ("d", [(2, 1.0)])],
                2: [("stay", [(2, 1.0)])],
            },
            labels={0: {"x"}, 1: {"y"}, 2: {"end"}},
        )
        strat1 = StrategyPair(p1_strategy={0: 0}, p2_strategy={1: 0})
        strat2 = StrategyPair(p1_strategy={0: 1}, p2_strategy={1: 1})
        comp = compare_strategy_bisimulations(lgame, strat1, strat2)
        assert 'strategy1_blocks' in comp
        assert 'strategy2_blocks' in comp
        assert 'same_partition' in comp


# ============================================================
# Section 9: SMT Verification
# ============================================================

class TestSMTVerification:
    def test_valid_partition_verified(self):
        lgame = symmetric_game()
        result = compute_game_bisimulation(lgame)
        verification = verify_game_bisimulation_smt(lgame, result.partition)
        assert verification['verified'] is True
        assert verification['n_violations'] == 0

    def test_invalid_partition_detected(self):
        lgame = owner_mismatch_game()
        # Force states 0 (P1) and 1 (P2) into the same block
        bad_partition = [{0, 1}, {2}]
        verification = verify_game_bisimulation_smt(lgame, bad_partition)
        assert verification['verified'] is False
        assert verification['n_violations'] > 0


# ============================================================
# Section 10: Minimization
# ============================================================

class TestMinimization:
    def test_minimize_reduces_states(self):
        lgame = symmetric_game()
        minimized, result = minimize_game(lgame)
        assert minimized.game.n_states <= lgame.game.n_states

    def test_minimize_idempotent(self):
        lgame = symmetric_game()
        minimized1, _ = minimize_game(lgame)
        minimized2, _ = minimize_game(minimized1)
        assert minimized1.game.n_states == minimized2.game.n_states


# ============================================================
# Section 11: Reward-Aware Bisimulation
# ============================================================

class TestRewardBisimulation:
    def test_same_rewards_bisimilar(self):
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P1, Player.CHANCE],
            action_transitions={
                0: [("a", [(2, 1.0)])],
                1: [("a", [(2, 1.0)])],
                2: [("stay", [(2, 1.0)])],
            },
            rewards={0: [5.0], 1: [5.0], 2: [0.0]},
            labels={0: {"start"}, 1: {"start"}, 2: {"end"}},
        )
        result = compute_reward_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] == mapping[1]

    def test_different_rewards_not_bisimilar(self):
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P1, Player.CHANCE],
            action_transitions={
                0: [("a", [(2, 1.0)])],
                1: [("a", [(2, 1.0)])],
                2: [("stay", [(2, 1.0)])],
            },
            rewards={0: [5.0], 1: [10.0], 2: [0.0]},
            labels={0: {"start"}, 1: {"start"}, 2: {"end"}},
        )
        result = compute_reward_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] != mapping[1]


# ============================================================
# Section 12: Multi-Player Structure
# ============================================================

class TestMultiPlayer:
    def test_three_player_types(self):
        """Game with all three player types."""
        lgame = make_labeled_game(
            n_states=6,
            owners=[Player.P1, Player.P1, Player.P2, Player.P2, Player.CHANCE, Player.CHANCE],
            action_transitions={
                0: [("a", [(2, 0.5), (4, 0.5)])],
                1: [("a", [(3, 0.5), (5, 0.5)])],
                2: [("b", [(4, 1.0)])],
                3: [("b", [(5, 1.0)])],
                4: [("c", [(0, 0.5), (1, 0.5)])],
                5: [("c", [(0, 0.5), (1, 0.5)])],
            },
            labels={0: {"p1"}, 1: {"p1"}, 2: {"p2"}, 3: {"p2"}, 4: {"ch"}, 5: {"ch"}},
        )
        result = compute_game_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        # Symmetric structure: 0~1, 2~3, 4~5
        assert mapping[0] == mapping[1]
        assert mapping[2] == mapping[3]
        assert mapping[4] == mapping[5]

    def test_chance_nodes_probability_matters(self):
        """CHANCE nodes with different distributions are not bisimilar."""
        lgame = make_labeled_game(
            n_states=4,
            owners=[Player.CHANCE, Player.CHANCE, Player.P1, Player.P1],
            action_transitions={
                0: [("flip", [(2, 0.7), (3, 0.3)])],
                1: [("flip", [(2, 0.5), (3, 0.5)])],
                2: [("stay", [(2, 1.0)])],
                3: [("stay", [(3, 1.0)])],
            },
            labels={0: {"ch"}, 1: {"ch"}, 2: {"win"}, 3: {"lose"}},
        )
        result = compute_game_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] != mapping[1]  # Different probabilities


# ============================================================
# Section 13: Game vs MDP Bisimulation Comparison
# ============================================================

class TestComparison:
    def test_game_vs_mdp(self):
        lgame = make_labeled_game(
            n_states=4,
            owners=[Player.P1, Player.P1, Player.P2, Player.CHANCE],
            action_transitions={
                0: [("a", [(2, 1.0)]), ("b", [(3, 1.0)])],
                1: [("a", [(2, 1.0)]), ("b", [(3, 1.0)])],
                2: [("c", [(3, 1.0)]), ("d", [(0, 1.0)])],
                3: [("stay", [(3, 0.5), (0, 0.5)])],
            },
            labels={0: {"x"}, 1: {"x"}, 2: {"y"}, 3: {"z"}},
        )
        comp = compare_game_vs_mdp_bisimulation(lgame, fix_player=Player.P2)
        assert 'game_blocks' in comp
        assert 'mdp_blocks' in comp
        # Game bisimulation is at least as fine as MDP
        assert comp['game_finer']


# ============================================================
# Section 14: Full Analysis and Summary
# ============================================================

class TestAnalysis:
    def test_full_analysis(self):
        lgame = symmetric_game()
        analysis = analyze_game_bisimulation(lgame)
        assert 'bisimulation' in analysis
        assert 'quotient' in analysis
        assert 'distance' in analysis
        assert 'simulation' in analysis
        assert analysis['n_blocks'] <= analysis['n_states']

    def test_summary_string(self):
        lgame = symmetric_game()
        summary = game_bisimulation_summary(lgame)
        assert "Bisimulation" in summary
        assert "States" in summary
        assert "Blocks" in summary
        assert "Block" in summary


# ============================================================
# Section 15: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_absorbing_states(self):
        """All absorbing states with same owner/labels are bisimilar."""
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.CHANCE, Player.CHANCE, Player.CHANCE],
            action_transitions={
                0: [("stay", [(0, 1.0)])],
                1: [("stay", [(1, 1.0)])],
                2: [("stay", [(2, 1.0)])],
            },
            labels={0: {"end"}, 1: {"end"}, 2: {"end"}},
        )
        result = compute_game_bisimulation(lgame)
        assert len(result.partition) == 1  # All bisimilar

    def test_no_labels(self):
        """Game with no labels -- partition by owner and actions only."""
        lgame = make_labeled_game(
            n_states=2,
            owners=[Player.P1, Player.P1],
            action_transitions={
                0: [("a", [(0, 1.0)])],
                1: [("a", [(1, 1.0)])],
            },
        )
        result = compute_game_bisimulation(lgame)
        # Same owner, no labels, same action structure -> bisimilar
        assert len(result.partition) == 1

    def test_many_actions(self):
        """States with many actions."""
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P1, Player.CHANCE],
            action_transitions={
                0: [("a", [(2, 1.0)]), ("b", [(2, 1.0)]), ("c", [(2, 1.0)])],
                1: [("x", [(2, 1.0)]), ("y", [(2, 1.0)]), ("z", [(2, 1.0)])],
                2: [("stay", [(2, 1.0)])],
            },
            labels={0: {"start"}, 1: {"start"}, 2: {"end"}},
        )
        result = compute_game_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        # Same action signatures (all go to 2 with prob 1.0), names don't matter
        assert mapping[0] == mapping[1]

    def test_action_names_irrelevant(self):
        """Bisimulation ignores action names, only distributions matter."""
        lgame = make_labeled_game(
            n_states=3,
            owners=[Player.P1, Player.P1, Player.P2],
            action_transitions={
                0: [("left", [(2, 1.0)])],
                1: [("right", [(2, 1.0)])],  # Different name, same distribution
                2: [("stay", [(2, 1.0)])],
            },
            labels={0: {"s"}, 1: {"s"}, 2: {"e"}},
        )
        result = compute_game_bisimulation(lgame)
        mapping = {}
        for bi, block in enumerate(result.partition):
            for s in block:
                mapping[s] = bi
        assert mapping[0] == mapping[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
