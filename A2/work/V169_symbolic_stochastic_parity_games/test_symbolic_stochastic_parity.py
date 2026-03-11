"""Tests for V169: Symbolic Stochastic Parity Games."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V165_stochastic_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V159_symbolic_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

from symbolic_stochastic_parity import (
    SymbolicStochasticParityGame, SymbolicStochasticResult,
    stochastic_to_symbolic, symbolic_to_stochastic,
    solve_symbolic_stochastic, solve_symbolic_stochastic_from_sspg,
    solve_almost_sure_symbolic, solve_positive_prob_symbolic,
    verify_symbolic_stochastic, verify_strategy_symbolic,
    make_stochastic_game, make_symbolic_stochastic_chain,
    make_symbolic_reachability_stochastic, make_symbolic_safety_stochastic,
    make_symbolic_buchi_stochastic,
    compare_explicit_vs_symbolic, symbolic_stochastic_statistics,
    batch_solve, _extract_states_sspg,
)
from stochastic_parity import (
    StochasticParityGame, VertexType, StochasticParityResult,
    solve_stochastic_parity, make_game,
)


# ===== Conversion Tests =====

class TestConversion:
    """Test explicit <-> symbolic conversion."""

    def test_empty_game(self):
        game = StochasticParityGame(
            vertices=set(), edges={}, vertex_type={}, priority={}, probabilities={}
        )
        sspg = stochastic_to_symbolic(game)
        assert _extract_states_sspg(sspg, sspg.vertices) == set()

    def test_single_vertex(self):
        game = make_stochastic_game(
            [(0, 'even', 2)],
            [(0, 0)],
        )
        sspg = stochastic_to_symbolic(game)
        assert _extract_states_sspg(sspg, sspg.vertices) == {0}
        assert _extract_states_sspg(sspg, sspg.owner_even) == {0}
        assert _extract_states_sspg(sspg, sspg.owner_random) == set()

    def test_three_types(self):
        game = make_stochastic_game(
            [(0, 'even', 0), (1, 'odd', 1), (2, 'random', 2)],
            [(0, 1), (1, 2), (2, 0), (2, 1)],
            probs={2: {0: 0.5, 1: 0.5}},
        )
        sspg = stochastic_to_symbolic(game)
        assert _extract_states_sspg(sspg, sspg.vertices) == {0, 1, 2}
        assert _extract_states_sspg(sspg, sspg.owner_even) == {0}
        assert _extract_states_sspg(sspg, sspg.owner_random) == {2}
        assert sspg.probabilities == {2: {0: 0.5, 1: 0.5}}

    def test_roundtrip(self):
        game = make_stochastic_game(
            [(0, 'even', 0), (1, 'odd', 1), (2, 'random', 2)],
            [(0, 1), (1, 2), (2, 0), (2, 1)],
            probs={2: {0: 0.7, 1: 0.3}},
        )
        sspg = stochastic_to_symbolic(game)
        game2 = symbolic_to_stochastic(sspg)
        assert game2.vertices == game.vertices
        assert game2.vertex_type == game.vertex_type
        assert game2.priority == game.priority
        for v in game.vertices:
            assert game2.edges.get(v, set()) == game.edges.get(v, set())

    def test_priority_encoding(self):
        game = make_stochastic_game(
            [(0, 'even', 0), (1, 'odd', 3), (2, 'random', 5)],
            [(0, 1), (1, 2), (2, 0)],
        )
        sspg = stochastic_to_symbolic(game)
        assert sspg.max_priority == 5
        assert 0 in sspg.priority_bdds
        assert 3 in sspg.priority_bdds
        assert 5 in sspg.priority_bdds


# ===== Deterministic Game Tests (no RANDOM vertices) =====

class TestDeterministicSubcases:
    """When no RANDOM vertices exist, should match deterministic solving."""

    def test_simple_even_wins(self):
        """Even has a self-loop with even priority -- Even wins everywhere."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'odd', 1)],
            [(0, 0), (0, 1), (1, 0)],
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == {0, 1}
        assert result.win_odd_as == set()

    def test_simple_odd_wins(self):
        """Odd forces play through odd priority forever."""
        game = make_stochastic_game(
            [(0, 'odd', 1), (1, 'odd', 1)],
            [(0, 1), (1, 0)],
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_odd_as == {0, 1}
        assert result.win_even_as == set()

    def test_split_winning(self):
        """Some vertices won by each player."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'odd', 1)],
            [(0, 0), (1, 1)],
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == {0}
        assert result.win_odd_as == {1}

    def test_deterministic_as_equals_pp(self):
        """Without RANDOM, AS and PP should agree."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'odd', 1), (2, 'even', 0)],
            [(0, 1), (1, 2), (2, 0)],
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == result.win_even_pp
        assert result.win_odd_as == result.win_odd_pp


# ===== Almost-Sure Winning Tests =====

class TestAlmostSure:
    """Test almost-sure winning computation."""

    def test_random_all_to_even_wins(self):
        """RANDOM vertex goes only to Even-winning states."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'even', 2)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        # Both successors of RANDOM lead to even-priority self-loops
        assert 0 in result.win_even_as
        assert 2 in result.win_even_as

    def test_random_one_to_odd(self):
        """RANDOM has one successor leading to Odd territory -- AS fails for Even."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        # Vertex 2 is an odd self-loop -- Odd wins there
        # RANDOM vertex 1 has a positive-prob path to Odd territory
        # So AS for Even: vertex 1 is NOT almost-sure Even
        # But PP for Even: vertex 1 IS positive-prob Even (has path to 0)
        assert 2 in result.win_odd_as
        assert 1 not in result.win_even_as

    def test_random_chain_as(self):
        """Chain with RANDOM vertex -- test AS closure."""
        game = make_symbolic_stochastic_chain(4, random_vertex=2, prob_forward=0.8)
        result = solve_symbolic_stochastic(game)
        # Just verify partition
        assert result.win_even_as | result.win_odd_as == game.vertices
        assert result.win_even_as & result.win_odd_as == set()

    def test_multiple_random_vertices(self):
        """Multiple RANDOM vertices in game."""
        game = make_stochastic_game(
            [(0, 'random', 0), (1, 'even', 2), (2, 'random', 0), (3, 'odd', 1)],
            [(0, 1), (0, 3), (1, 2), (2, 1), (2, 3), (3, 3)],
            probs={0: {1: 0.5, 3: 0.5}, 2: {1: 0.5, 3: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        # Vertex 3 is odd self-loop: Odd wins
        # RANDOM vertices reach Odd territory: AS fails for Even at random verts
        assert 3 in result.win_odd_as

    def test_as_high_prob_still_not_sure(self):
        """Even 99% toward Even territory is not almost-sure if 1% leads to Odd."""
        game = make_stochastic_game(
            [(0, 'random', 0), (1, 'even', 2), (2, 'odd', 1)],
            [(0, 1), (0, 2), (1, 1), (2, 2)],
            probs={0: {1: 0.99, 2: 0.01}},
        )
        result = solve_symbolic_stochastic(game)
        # AS: RANDOM 0 has positive-prob path to Odd territory
        assert 0 not in result.win_even_as
        # PP: RANDOM 0 has positive-prob path to Even territory
        assert 0 in result.win_even_pp


# ===== Positive-Probability Winning Tests =====

class TestPositiveProb:
    """Test positive-probability winning computation."""

    def test_pp_any_good_successor(self):
        """PP: even one good successor path suffices."""
        game = make_stochastic_game(
            [(0, 'random', 0), (1, 'even', 2), (2, 'odd', 1)],
            [(0, 1), (0, 2), (1, 1), (2, 2)],
            probs={0: {1: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        assert 0 in result.win_even_pp
        assert 1 in result.win_even_pp

    def test_pp_no_good_successor(self):
        """PP: no successor leads to Even territory."""
        game = make_stochastic_game(
            [(0, 'random', 0), (1, 'odd', 1), (2, 'odd', 3)],
            [(0, 1), (0, 2), (1, 1), (2, 2)],
            probs={0: {1: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_pp == set()

    def test_pp_superset_of_as(self):
        """PP winning region is always a superset of AS winning region."""
        game = make_stochastic_game(
            [(0, 'random', 0), (1, 'even', 2), (2, 'odd', 1), (3, 'even', 2)],
            [(0, 1), (0, 2), (1, 3), (2, 2), (3, 3)],
            probs={0: {1: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as.issubset(result.win_even_pp)

    def test_pp_equals_deterministic(self):
        """PP should match deterministic solve (RANDOM as EVEN)."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        # Check against explicit solver
        explicit = solve_stochastic_parity(game)
        assert result.win_even_pp == explicit.win_even_pp


# ===== Verification Against Explicit Solver =====

class TestVerification:
    """Verify symbolic results match explicit V165 solver."""

    def test_verify_simple(self):
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree'], f"AS mismatch: {check}"
        assert check['pp_agree'], f"PP mismatch: {check}"
        assert check['as_subset_pp']

    def test_verify_chain(self):
        game = make_symbolic_stochastic_chain(5, random_vertex=3, prob_forward=0.6)
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree'], f"AS mismatch: {check}"
        assert check['pp_agree'], f"PP mismatch: {check}"

    def test_verify_all_even(self):
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'even', 0), (2, 'even', 2)],
            [(0, 1), (1, 2), (2, 0)],
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']
        assert check['pp_agree']

    def test_verify_all_odd(self):
        game = make_stochastic_game(
            [(0, 'odd', 1), (1, 'odd', 3)],
            [(0, 1), (1, 0)],
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']
        assert check['pp_agree']

    def test_verify_mixed_priorities(self):
        game = make_stochastic_game(
            [(0, 'even', 0), (1, 'random', 1), (2, 'odd', 2), (3, 'even', 3)],
            [(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)],
            probs={1: {2: 0.4, 3: 0.6}},
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree'], f"AS mismatch: {check}"
        assert check['pp_agree'], f"PP mismatch: {check}"

    def test_verify_larger_game(self):
        """6-vertex game with 2 RANDOM vertices."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1),
             (3, 'even', 2), (4, 'random', 0), (5, 'odd', 1)],
            [(0, 1), (1, 2), (1, 3), (2, 5), (3, 4), (4, 0), (4, 5), (5, 2)],
            probs={1: {2: 0.3, 3: 0.7}, 4: {0: 0.6, 5: 0.4}},
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree'], f"AS mismatch: {check}"
        assert check['pp_agree'], f"PP mismatch: {check}"


# ===== Strategy Verification =====

class TestStrategyVerification:
    """Test strategy extraction and verification."""

    def test_strategy_valid_as(self):
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'odd', 1)],
            [(0, 0), (0, 1), (1, 0)],
        )
        result = solve_symbolic_stochastic(game)
        check = verify_strategy_symbolic(game, result.strategy_even_as,
                                          result.win_even_as, True, 'almost_sure')
        assert check['valid'], check['errors']

    def test_strategy_valid_pp(self):
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        if result.win_even_pp:
            check = verify_strategy_symbolic(game, result.strategy_even_pp,
                                              result.win_even_pp, True, 'positive_prob')
            assert check['valid'], check['errors']

    def test_odd_strategy(self):
        game = make_stochastic_game(
            [(0, 'odd', 1), (1, 'odd', 3)],
            [(0, 1), (1, 0)],
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_odd_as == {0, 1}
        check = verify_strategy_symbolic(game, result.strategy_odd_as,
                                          result.win_odd_as, False, 'almost_sure')
        assert check['valid'], check['errors']


# ===== Game Constructors =====

class TestGameConstructors:
    """Test game construction helpers."""

    def test_chain_game(self):
        game = make_symbolic_stochastic_chain(4, random_vertex=2, prob_forward=0.7)
        assert len(game.vertices) == 4
        assert game.vertex_type[2] == VertexType.RANDOM
        assert game.vertex_type[0] == VertexType.EVEN
        assert game.vertex_type[1] == VertexType.ODD

    def test_reachability_game(self):
        game = make_symbolic_reachability_stochastic(
            4, target={3}, even_states={0, 2}, random_states={1},
            transitions=[(0, 1), (1, 2), (1, 3), (2, 3), (3, 3)],
            probs={1: {2: 0.5, 3: 0.5}},
        )
        assert game.vertex_type[1] == VertexType.RANDOM
        assert game.priority[3] == 2  # target
        assert game.priority[0] == 1  # non-target
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']

    def test_safety_game(self):
        game = make_symbolic_safety_stochastic(
            3, bad={2}, even_states={0}, random_states={1},
            transitions=[(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        assert game.priority[2] == 1  # bad
        assert game.priority[0] == 0  # safe
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']

    def test_buchi_game(self):
        game = make_symbolic_buchi_stochastic(
            3, accepting={0}, even_states={0}, random_states={1},
            transitions=[(0, 1), (1, 0), (1, 2), (2, 0)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        assert game.priority[0] == 2  # accepting
        assert game.priority[1] == 1  # non-accepting
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']

    def test_make_stochastic_game(self):
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'odd', 1), (2, 'random', 0)],
            [(0, 1), (1, 2), (2, 0), (2, 1)],
            probs={2: {0: 0.3, 1: 0.7}},
        )
        assert game.vertices == {0, 1, 2}
        assert game.vertex_type[2] == VertexType.RANDOM


# ===== Comparison & Statistics =====

class TestComparison:
    """Test comparison and statistics APIs."""

    def test_compare_explicit_vs_symbolic(self):
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        cmp = compare_explicit_vs_symbolic(game)
        assert cmp['as_agree']
        assert cmp['pp_agree']
        assert cmp['as_subset_pp']

    def test_statistics(self):
        game = make_stochastic_game(
            [(0, 'even', 0), (1, 'odd', 1), (2, 'random', 2)],
            [(0, 1), (1, 2), (2, 0)],
        )
        stats = symbolic_stochastic_statistics(game)
        assert stats['vertices'] == 3
        assert stats['even_vertices'] == 1
        assert stats['odd_vertices'] == 1
        assert stats['random_vertices'] == 1
        assert stats['max_priority'] == 2
        assert stats['distinct_priorities'] == 3


# ===== Batch Solve =====

class TestBatchSolve:
    """Test batch solving API."""

    def test_batch_solve(self):
        g1 = make_stochastic_game(
            [(0, 'even', 2), (1, 'odd', 1)],
            [(0, 0), (1, 1)],
        )
        g2 = make_stochastic_game(
            [(0, 'odd', 1), (1, 'odd', 3)],
            [(0, 1), (1, 0)],
        )
        results = batch_solve([("g1", g1), ("g2", g2)])
        assert "g1" in results
        assert "g2" in results
        assert results["g1"].win_even_as == {0}
        assert results["g2"].win_odd_as == {0, 1}


# ===== Edge Cases =====

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_even_self_loop(self):
        game = make_stochastic_game([(0, 'even', 2)], [(0, 0)])
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == {0}
        assert result.win_even_pp == {0}

    def test_single_odd_self_loop(self):
        game = make_stochastic_game([(0, 'odd', 1)], [(0, 0)])
        result = solve_symbolic_stochastic(game)
        assert result.win_odd_as == {0}
        assert result.win_odd_pp == {0}

    def test_single_random_self_loop(self):
        """RANDOM self-loop with even priority -- Even wins AS and PP."""
        game = make_stochastic_game(
            [(0, 'random', 2)],
            [(0, 0)],
            probs={0: {0: 1.0}},
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == {0}
        assert result.win_even_pp == {0}

    def test_random_deterministic_edges(self):
        """RANDOM vertex with probability 1.0 to single successor (deterministic)."""
        game = make_stochastic_game(
            [(0, 'random', 0), (1, 'even', 2)],
            [(0, 1), (1, 1)],
            probs={0: {1: 1.0}},
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == {0, 1}

    def test_disconnected_components(self):
        """Two disconnected components."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'odd', 1)],
            [(0, 0), (1, 1)],
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == {0}
        assert result.win_odd_as == {1}

    def test_zero_probability_edge(self):
        """Edge with probability 0 should not affect winning."""
        game = make_stochastic_game(
            [(0, 'random', 0), (1, 'even', 2), (2, 'odd', 1)],
            [(0, 1), (0, 2), (1, 1), (2, 2)],
            probs={0: {1: 1.0, 2: 0.0}},
        )
        result = solve_symbolic_stochastic(game)
        # Zero-prob to Odd territory doesn't affect AS
        assert 0 in result.win_even_as

    def test_partition_property(self):
        """Win_even and win_odd must partition all vertices for AS."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1), (3, 'even', 0)],
            [(0, 1), (1, 2), (1, 3), (2, 0), (3, 3)],
            probs={1: {2: 0.5, 3: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as | result.win_odd_as == game.vertices
        assert result.win_even_as & result.win_odd_as == set()

    def test_as_subset_pp_invariant(self):
        """AS winning region is always a subset of PP winning region."""
        for rv in range(4):
            game = make_symbolic_stochastic_chain(4, random_vertex=rv, prob_forward=0.5)
            result = solve_symbolic_stochastic(game)
            assert result.win_even_as.issubset(result.win_even_pp), \
                f"AS not subset of PP for rv={rv}: AS={result.win_even_as}, PP={result.win_even_pp}"


# ===== SSPG Direct Solving =====

class TestSSPGDirect:
    """Test solving directly from SymbolicStochasticParityGame."""

    def test_solve_from_sspg(self):
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        sspg = stochastic_to_symbolic(game)
        result = solve_symbolic_stochastic_from_sspg(sspg)
        # Should match explicit solver
        explicit = solve_stochastic_parity(game)
        assert result.win_even_as == explicit.win_even_as
        assert result.win_odd_as == explicit.win_odd_as
        assert result.win_even_pp == explicit.win_even_pp
        assert result.win_odd_pp == explicit.win_odd_pp


# ===== Reachability/Safety/Buchi Semantics =====

class TestGameSemantics:
    """Test specific game semantics (reachability, safety, Buchi)."""

    def test_reachability_even_can_reach(self):
        """Even can reach target through RANDOM."""
        game = make_symbolic_reachability_stochastic(
            3, target={2}, even_states={0}, random_states={1},
            transitions=[(0, 1), (1, 2), (2, 2)],
            probs={1: {2: 1.0}},
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_even_as == {0, 1, 2}

    def test_safety_random_leaks(self):
        """RANDOM can reach bad state -- Even may lose AS."""
        game = make_symbolic_safety_stochastic(
            3, bad={2}, even_states={0}, random_states={1},
            transitions=[(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        # RANDOM 1 can reach bad state 2 with positive prob
        # For AS: Even loses (1 can reach bad)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']

    def test_buchi_accepting_reachable(self):
        """Buchi: accepting state reachable and revisitable."""
        game = make_symbolic_buchi_stochastic(
            3, accepting={2}, even_states={0, 2}, random_states={1},
            transitions=[(0, 1), (1, 2), (2, 0)],
            probs={1: {2: 1.0}},
        )
        result = solve_symbolic_stochastic(game)
        # All vertices cycle through accepting state 2
        assert result.win_even_as == {0, 1, 2}


# ===== Complex Topologies =====

class TestComplexTopologies:
    """Test more complex game structures."""

    def test_diamond_random(self):
        """Diamond structure: Even chooses left/right, RANDOM at bottom."""
        game = make_stochastic_game(
            [(0, 'even', 0), (1, 'even', 2), (2, 'odd', 1),
             (3, 'random', 0)],
            [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0), (3, 1)],
            probs={3: {0: 0.5, 1: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']
        assert check['pp_agree']

    def test_two_scc_with_random_bridge(self):
        """Two SCCs connected by a RANDOM bridge."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'even', 0), (2, 'random', 0),
             (3, 'odd', 1), (4, 'odd', 3)],
            [(0, 1), (1, 0), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)],
            probs={2: {0: 0.5, 3: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']
        assert check['pp_agree']

    def test_long_chain_with_random(self):
        """Longer chain to test scalability."""
        game = make_symbolic_stochastic_chain(8, random_vertex=4, prob_forward=0.7)
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']
        assert check['pp_agree']

    def test_multiple_priorities(self):
        """Game with priorities 0-4."""
        game = make_stochastic_game(
            [(0, 'even', 0), (1, 'random', 1), (2, 'odd', 2),
             (3, 'even', 3), (4, 'odd', 4)],
            [(0, 1), (1, 2), (1, 3), (2, 4), (3, 0), (4, 0)],
            probs={1: {2: 0.5, 3: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']
        assert check['pp_agree']

    def test_three_random_vertices(self):
        """Game with 3 RANDOM vertices."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'random', 0),
             (3, 'random', 0), (4, 'odd', 1), (5, 'even', 2)],
            [(0, 1), (1, 2), (1, 4), (2, 3), (2, 5), (3, 0), (3, 4),
             (4, 4), (5, 5)],
            probs={1: {2: 0.6, 4: 0.4}, 2: {3: 0.5, 5: 0.5},
                   3: {0: 0.7, 4: 0.3}},
        )
        result = solve_symbolic_stochastic(game)
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree'], f"AS mismatch: {check}"
        assert check['pp_agree'], f"PP mismatch: {check}"


# ===== Regression =====

class TestRegression:
    """Regression tests for known patterns."""

    def test_random_self_loop_odd_priority(self):
        """RANDOM self-loop with odd priority -- Odd wins."""
        game = make_stochastic_game(
            [(0, 'random', 1)],
            [(0, 0)],
            probs={0: {0: 1.0}},
        )
        result = solve_symbolic_stochastic(game)
        assert result.win_odd_as == {0}
        assert result.win_odd_pp == {0}

    def test_even_can_avoid_random(self):
        """Even has a choice to avoid RANDOM vertex."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'even', 2), (3, 'odd', 1)],
            [(0, 1), (0, 2), (1, 3), (1, 0), (2, 2), (3, 3)],
            probs={1: {3: 0.5, 0: 0.5}},
        )
        result = solve_symbolic_stochastic(game)
        # Even can choose 0 -> 2 (self-loop, even prio) to win AS
        assert 0 in result.win_even_as
        assert 2 in result.win_even_as

    def test_forced_through_random(self):
        """Even is forced through RANDOM vertex."""
        game = make_stochastic_game(
            [(0, 'even', 2), (1, 'random', 0), (2, 'odd', 1)],
            [(0, 1), (1, 0), (1, 2), (2, 2)],
            probs={1: {0: 0.9, 2: 0.1}},
        )
        result = solve_symbolic_stochastic(game)
        # Even must go through RANDOM 1 which has positive-prob to Odd territory
        # AS: Even loses from 0 and 1
        check = verify_symbolic_stochastic(game, result)
        assert check['as_agree']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
