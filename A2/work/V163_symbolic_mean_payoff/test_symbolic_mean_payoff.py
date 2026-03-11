"""
Tests for V163: Symbolic Mean-Payoff Games

Tests BDD-based symbolic solving of mean-payoff parity games,
verifying agreement with explicit V161 solver.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V161_mean_payoff_parity'))

from parity_games import Player, zielonka
from mean_payoff_parity import (
    MeanPayoffParityGame, solve_mpp, solve_mpp_threshold,
    compute_mpp_values, decompose_mpp, make_adversarial_mpp,
)
from symbolic_mean_payoff import (
    encode_mpp_game, solve_symbolic_mpp, compute_symbolic_mpp_values,
    symbolic_attractor, symbolic_reachability, symbolic_safety_check,
    symbolic_decompose_mpp, symbolic_mpp_statistics,
    compare_with_explicit, compare_values, compare_decompositions,
    make_symbolic_chain_mpp, make_symbolic_choice_mpp,
    make_symbolic_diamond_mpp, make_symbolic_grid_mpp,
    _extract_vertices, _make_vertex_set_bdd, _symbolic_zielonka,
    SymbolicMPPEncoding, SymbolicMPPResult,
)


# ============================================================
# Section 1: Encoding
# ============================================================

class TestEncoding:
    """Test BDD encoding of MPP games."""

    def test_encode_simple_chain(self):
        g = make_symbolic_chain_mpp(3, [1, -1, 0], [0, 1, 2])
        enc = encode_mpp_game(g)
        assert enc.num_vertices == 3
        assert enc.num_bits >= 2  # need at least 2 bits for 3 vertices
        assert len(enc.edges) == 3
        assert len(enc.priority_groups) == 3  # priorities 0, 1, 2

    def test_encode_owner_separation(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, -1)
        enc = encode_mpp_game(g)
        even_verts = _extract_vertices(enc, enc.owner_even)
        odd_verts = _extract_vertices(enc, enc.owner_odd)
        assert even_verts == {0}
        assert odd_verts == {1}

    def test_encode_vertex_set_roundtrip(self):
        g = make_symbolic_chain_mpp(5)
        enc = encode_mpp_game(g)
        all_verts = _extract_vertices(enc, enc.vertices_bdd)
        assert all_verts == {0, 1, 2, 3, 4}

    def test_encode_priority_groups(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 2)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, 1)
        g.add_edge(2, 0, 1)
        enc = encode_mpp_game(g)
        assert 0 in enc.priority_groups
        assert 2 in enc.priority_groups
        p0_verts = _extract_vertices(enc, enc.priority_groups[0])
        p2_verts = _extract_vertices(enc, enc.priority_groups[2])
        assert p0_verts == {0, 2}
        assert p2_verts == {1}

    def test_encode_weight_groups(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 0)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, -2)
        enc = encode_mpp_game(g)
        assert 3 in enc.weight_groups
        assert -2 in enc.weight_groups

    def test_encode_empty_game_raises(self):
        g = MeanPayoffParityGame()
        with pytest.raises(ValueError):
            encode_mpp_game(g)


# ============================================================
# Section 2: Symbolic Operations
# ============================================================

class TestSymbolicOps:
    """Test symbolic BDD operations (predecessors, successors, etc.)."""

    def test_vertex_set_bdd(self):
        g = make_symbolic_chain_mpp(4)
        enc = encode_mpp_game(g)
        subset = _make_vertex_set_bdd(enc, {0, 2})
        extracted = _extract_vertices(enc, subset)
        assert extracted == {0, 2}

    def test_reachability_full_chain(self):
        g = make_symbolic_chain_mpp(5)
        reached = symbolic_reachability(g, {0})
        assert reached == {0, 1, 2, 3, 4}

    def test_reachability_partial(self):
        g = MeanPayoffParityGame()
        for v in range(4):
            g.add_vertex(v, Player.EVEN, 0)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1)
        g.add_edge(2, 3, 1)
        g.add_edge(3, 2, 1)
        reached = symbolic_reachability(g, {0})
        assert reached == {0, 1}

    def test_attractor_even(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 2, 0)
        g.add_edge(2, 0, 0)
        enc = encode_mpp_game(g)
        target = _make_vertex_set_bdd(enc, {2})
        attr = symbolic_attractor(enc, target, Player.EVEN)
        # Even at 0 can choose to go to 2 directly -> 0 is in attractor
        # Odd at 1 has only one successor (2) -> 1 is also in attractor
        assert _extract_vertices(enc, attr) == {0, 1, 2}

    def test_attractor_odd(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.ODD, 0)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 0)
        g.add_edge(2, 0, 0)
        enc = encode_mpp_game(g)
        target = _make_vertex_set_bdd(enc, {2})
        attr = symbolic_attractor(enc, target, Player.ODD)
        # Odd at 0 can choose to go to 2 -> 0 in attractor
        attr_verts = _extract_vertices(enc, attr)
        assert 0 in attr_verts
        assert 2 in attr_verts


# ============================================================
# Section 3: Symbolic Parity
# ============================================================

class TestSymbolicParity:
    """Test symbolic Zielonka parity solving."""

    def test_simple_even_wins(self):
        """All even priorities -> Even wins everything."""
        g = make_symbolic_chain_mpp(3, [1, 1, 1], [0, 2, 0])
        enc = encode_mpp_game(g)
        we, wo = _symbolic_zielonka(enc, enc.vertices_bdd)
        assert _extract_vertices(enc, we) == {0, 1, 2}
        assert _extract_vertices(enc, wo) == set()

    def test_simple_odd_wins(self):
        """Odd controls vertex with max odd priority -> Odd wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.ODD, 3)  # Odd, highest prio is 3 (odd)
        g.add_edge(0, 0, 0)
        enc = encode_mpp_game(g)
        we, wo = _symbolic_zielonka(enc, enc.vertices_bdd)
        assert _extract_vertices(enc, we) == set()
        assert _extract_vertices(enc, wo) == {0}

    def test_mixed_priorities(self):
        """Even chooses between even-prio and odd-prio cycles."""
        g = make_symbolic_choice_mpp(1, -1, 2, 3)
        # Even at 0 can go to 1 (prio 2, even) or 2 (prio 3, odd)
        # Even should choose prio 2 path -> Even wins
        enc = encode_mpp_game(g)
        we, wo = _symbolic_zielonka(enc, enc.vertices_bdd)
        assert 0 in _extract_vertices(enc, we)

    def test_parity_agrees_with_explicit(self):
        """Symbolic Zielonka agrees with explicit V156 Zielonka."""
        g = make_symbolic_chain_mpp(4, [1, -1, 1, -1], [0, 1, 2, 3])
        pg = g.to_parity_game()
        explicit = zielonka(pg)
        enc = encode_mpp_game(g)
        we, wo = _symbolic_zielonka(enc, enc.vertices_bdd)
        assert _extract_vertices(enc, we) == set(explicit.win_even)
        assert _extract_vertices(enc, wo) == set(explicit.win_odd)


# ============================================================
# Section 4: Symbolic MPP Solving
# ============================================================

class TestSymbolicMPPSolving:
    """Test symbolic mean-payoff parity solving."""

    def test_empty_game(self):
        g = MeanPayoffParityGame()
        result = solve_symbolic_mpp(g)
        assert result.win_even == set()
        assert result.win_odd == set()

    def test_positive_chain_even_wins(self):
        """Chain with all positive weights -> Even wins (MP >= 0)."""
        g = make_symbolic_chain_mpp(3, [1, 1, 1], [0, 0, 0])
        result = solve_symbolic_mpp(g)
        assert result.win_even == {0, 1, 2}

    def test_negative_chain_even_loses(self):
        """Chain with all negative weights -> Even loses (MP < 0)."""
        g = make_symbolic_chain_mpp(3, [-1, -1, -1], [0, 0, 0])
        result = solve_symbolic_mpp(g)
        assert result.win_even == set()
        assert result.win_odd == {0, 1, 2}

    def test_choice_game_even_picks_good(self):
        """Even chooses between good (parity + MP) and bad cycle."""
        g = make_symbolic_choice_mpp(1, -1, 0, 1)
        result = solve_symbolic_mpp(g)
        # Even can choose the good cycle (weight=1, even prio=0)
        assert 0 in result.win_even

    def test_threshold_positive(self):
        """Test with threshold > 0."""
        g = make_symbolic_chain_mpp(2, [3, -1], [0, 0])
        # Mean payoff = (3 + -1) / 2 = 1.0
        result = solve_symbolic_mpp(g, threshold=0.5)
        assert result.win_even == {0, 1}

    def test_threshold_too_high(self):
        """Threshold higher than achievable mean payoff."""
        g = make_symbolic_chain_mpp(2, [1, 1], [0, 0])
        # Mean payoff = 1.0, threshold = 2.0
        result = solve_symbolic_mpp(g, threshold=2.0)
        assert result.win_even == set()

    def test_agrees_with_explicit(self):
        """Symbolic solver agrees with explicit V161 solver."""
        g = make_symbolic_choice_mpp(2, -1, 0, 1)
        explicit = solve_mpp(g)
        symbolic = solve_symbolic_mpp(g)
        assert explicit.win_even == symbolic.win_even
        assert explicit.win_odd == symbolic.win_odd

    def test_adversarial_game(self):
        """Odd controls the choice -> Odd picks bad for Even."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.ODD, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 5)   # good for Even
        g.add_edge(2, 0, -5)  # bad for Even
        result = solve_symbolic_mpp(g)
        # Odd chooses vertex 2 (bad) -> Even loses
        assert 0 in result.win_odd


# ============================================================
# Section 5: Safety Checking
# ============================================================

class TestSafetyChecking:
    """Test symbolic safety checking."""

    def test_full_safe_set(self):
        g = make_symbolic_chain_mpp(3)
        safety = symbolic_safety_check(g, {0, 1, 2})
        assert safety['safe_count'] == 3

    def test_partial_safe_set(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1, 0)
        g.add_edge(1, 2, 0)
        g.add_edge(2, 0, 0)
        # Safe set = {0, 1} but 1 leads to 2 which is outside
        safety = symbolic_safety_check(g, {0, 1})
        # Odd at 1 can only go to 2 (unsafe) -> 0 and 1 are unsafe
        assert safety['safe_count'] == 0

    def test_even_can_avoid_unsafe(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_vertex(2, Player.EVEN, 0)  # unsafe
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, 0)
        g.add_edge(2, 0, 0)
        safety = symbolic_safety_check(g, {0, 1})
        # Even at 0 can choose to go to 1 (safe) -> {0, 1} are safe
        assert 0 in safety['safe_vertices']
        assert 1 in safety['safe_vertices']


# ============================================================
# Section 6: Comparison API
# ============================================================

class TestComparison:
    """Test comparison of symbolic vs explicit solving."""

    def test_compare_chain(self):
        g = make_symbolic_chain_mpp(4, [1, -1, 2, -1], [0, 1, 0, 1])
        result = compare_with_explicit(g)
        assert result['agree']

    def test_compare_choice(self):
        g = make_symbolic_choice_mpp(3, -2, 0, 1)
        result = compare_with_explicit(g)
        assert result['agree']

    def test_compare_with_threshold(self):
        g = make_symbolic_chain_mpp(3, [2, -1, 0], [0, 0, 0])
        result = compare_with_explicit(g, threshold=0.0)
        assert result['agree']

    def test_compare_adversarial(self):
        g = make_adversarial_mpp(3, 0, 1)
        result = compare_with_explicit(g)
        assert result['agree']

    def test_compare_diamond(self):
        g = make_symbolic_diamond_mpp(0, 2, -1)
        result = compare_with_explicit(g)
        assert result['agree']

    def test_compare_grid(self):
        g = make_symbolic_grid_mpp(2, 2)
        result = compare_with_explicit(g)
        assert result['agree']


# ============================================================
# Section 7: Value Computation
# ============================================================

class TestValues:
    """Test symbolic optimal value computation."""

    def test_constant_chain_value(self):
        """All weight=1 -> value should be 1.0."""
        g = make_symbolic_chain_mpp(3, [1, 1, 1], [0, 0, 0])
        result = compute_symbolic_mpp_values(g)
        for v in g.vertices:
            assert abs(result.values[v] - 1.0) < 0.5

    def test_zero_chain_value(self):
        """Alternating +1/-1 -> value should be 0.0."""
        g = make_symbolic_chain_mpp(2, [1, -1], [0, 0])
        result = compute_symbolic_mpp_values(g)
        for v in g.vertices:
            assert abs(result.values[v]) < 0.5 + 1e-9

    def test_values_agree_with_explicit(self):
        g = make_symbolic_choice_mpp(2, -1, 0, 1)
        result = compare_values(g)
        assert result['value_agree']

    def test_empty_game_values(self):
        g = MeanPayoffParityGame()
        result = compute_symbolic_mpp_values(g)
        assert result.values == {}


# ============================================================
# Section 8: Decomposition
# ============================================================

class TestDecomposition:
    """Test symbolic decomposition analysis."""

    def test_decompose_simple(self):
        g = make_symbolic_chain_mpp(3, [1, 1, 1], [0, 0, 0])
        result = symbolic_decompose_mpp(g)
        # All positive weights + even priorities -> Even wins all
        assert result['parity_only']['win_even'] == {0, 1, 2}
        assert result['mean_payoff_only']['win_even'] == {0, 1, 2}
        assert result['combined']['win_even'] == {0, 1, 2}

    def test_decompose_conflict(self):
        """Game where parity wins but MP doesn't (or vice versa)."""
        g = make_symbolic_choice_mpp(1, -1, 0, 1)
        result = symbolic_decompose_mpp(g)
        # Should have consistent results
        assert result['combined']['win_even'] <= result['parity_only']['win_even']

    def test_decompose_agrees_with_explicit(self):
        g = make_symbolic_chain_mpp(3, [2, -1, 0], [0, 1, 2])
        result = compare_decompositions(g)
        assert result['parity_agree']
        assert result['mp_agree']
        assert result['combined_agree']

    def test_decompose_threshold(self):
        g = make_symbolic_chain_mpp(2, [3, -1], [0, 0])
        result = symbolic_decompose_mpp(g, threshold=0.5)
        # MP = 1.0 >= 0.5, parity all even -> all win
        assert result['combined']['win_even'] == {0, 1}


# ============================================================
# Section 9: Statistics
# ============================================================

class TestStatistics:
    """Test statistics computation."""

    def test_statistics_chain(self):
        g = make_symbolic_chain_mpp(5, [1, -1, 2, -2, 0], [0, 1, 2, 3, 0])
        stats = symbolic_mpp_statistics(g)
        assert stats['num_vertices'] == 5
        assert stats['num_edges'] == 5
        assert stats['num_bits'] >= 3
        assert stats['priority_groups'] == 4  # priorities 0, 1, 2, 3
        assert stats['max_priority'] == 3
        assert stats['max_weight'] == 2

    def test_statistics_diamond(self):
        g = make_symbolic_diamond_mpp()
        stats = symbolic_mpp_statistics(g)
        assert stats['num_vertices'] == 4
        assert stats['num_edges'] == 5

    def test_statistics_grid(self):
        g = make_symbolic_grid_mpp(2, 3)
        stats = symbolic_mpp_statistics(g)
        assert stats['num_vertices'] == 6


# ============================================================
# Section 10: Construction Helpers
# ============================================================

class TestConstruction:
    """Test game construction helpers."""

    def test_chain(self):
        g = make_symbolic_chain_mpp(4)
        assert len(g.vertices) == 4
        for v in range(4):
            succs = g.successors(v)
            assert len(succs) == 1
            assert succs[0][0] == (v + 1) % 4

    def test_choice(self):
        g = make_symbolic_choice_mpp(5, -3, 0, 1)
        assert len(g.vertices) == 3
        assert len(g.successors(0)) == 2

    def test_diamond(self):
        g = make_symbolic_diamond_mpp()
        assert len(g.vertices) == 4
        assert len(g.successors(0)) == 2  # Even top chooses left or right

    def test_grid(self):
        g = make_symbolic_grid_mpp(3, 3)
        assert len(g.vertices) == 9
        # Center vertex (1,1) should have 4 edges (up, down, left, right)
        # Plus reverse edges from neighbors
        center = 1 * 3 + 1
        assert len(g.successors(center)) >= 4


# ============================================================
# Section 11: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_self_loop_even(self):
        """Single vertex with self-loop, even priority."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, 1)
        result = solve_symbolic_mpp(g)
        assert result.win_even == {0}

    def test_single_self_loop_negative(self):
        """Single vertex with self-loop, negative weight."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_edge(0, 0, -1)
        result = solve_symbolic_mpp(g)
        assert result.win_even == set()

    def test_single_self_loop_odd_prio(self):
        """Single vertex with odd priority -> Odd wins."""
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 1)
        g.add_edge(0, 0, 10)  # Even good weight but odd parity
        result = solve_symbolic_mpp(g)
        assert 0 in result.win_odd

    def test_two_vertex_chain(self):
        g = make_symbolic_chain_mpp(2, [1, 1], [0, 0])
        result = solve_symbolic_mpp(g)
        assert result.win_even == {0, 1}

    def test_all_zero_weights(self):
        g = make_symbolic_chain_mpp(3, [0, 0, 0], [0, 0, 0])
        result = solve_symbolic_mpp(g, threshold=0.0)
        assert result.win_even == {0, 1, 2}

    def test_parity_conflict_tradeoff(self):
        """Game where Even must sacrifice MP for parity."""
        g = MeanPayoffParityGame()
        # Vertex 0 (Even): choose between good-MP/bad-parity or bad-MP/good-parity
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 0)   # good parity, bad weight
        g.add_vertex(2, Player.EVEN, 3)   # bad parity (3=odd), good weight
        g.add_edge(0, 1, 0)
        g.add_edge(0, 2, 0)
        g.add_edge(1, 0, -1)  # bad MP
        g.add_edge(2, 0, 10)  # good MP but odd parity
        result = solve_symbolic_mpp(g)
        # Even must choose path through 1 (even parity 0) but MP = -0.5 < 0
        # Or path through 2 with parity 3 (odd) -> Odd wins parity
        # Either way Even loses combined condition
        assert 0 in result.win_odd


# ============================================================
# Section 12: Multi-vertex Agreement
# ============================================================

class TestMultiVertex:
    """Test with larger games to verify symbolic/explicit agreement."""

    def test_five_vertex_mixed(self):
        g = MeanPayoffParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_vertex(2, Player.EVEN, 2)
        g.add_vertex(3, Player.ODD, 1)
        g.add_vertex(4, Player.EVEN, 0)
        g.add_edge(0, 1, 2)
        g.add_edge(0, 2, -1)
        g.add_edge(1, 2, 1)
        g.add_edge(1, 3, -2)
        g.add_edge(2, 4, 1)
        g.add_edge(3, 4, 0)
        g.add_edge(4, 0, 1)
        result = compare_with_explicit(g)
        assert result['agree']

    def test_six_vertex_grid(self):
        g = make_symbolic_grid_mpp(2, 3)
        result = compare_with_explicit(g)
        assert result['agree']

    def test_eight_vertex_complex(self):
        g = MeanPayoffParityGame()
        for i in range(8):
            g.add_vertex(i, Player.EVEN if i % 2 == 0 else Player.ODD, i % 4)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, -1)
        g.add_edge(2, 3, 2)
        g.add_edge(3, 0, -2)
        g.add_edge(4, 5, 1)
        g.add_edge(5, 6, 1)
        g.add_edge(6, 7, -1)
        g.add_edge(7, 4, -1)
        g.add_edge(0, 4, 0)
        g.add_edge(4, 0, 0)
        result = compare_with_explicit(g)
        assert result['agree']
