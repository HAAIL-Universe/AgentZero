"""Tests for V159: Symbolic Parity Games."""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

from symbolic_parity_games import (
    SymbolicParityGame, SymbolicSolution,
    explicit_to_symbolic, symbolic_to_explicit, solve_symbolic,
    extract_winning_sets, extract_strategy, verify_symbolic_solution,
    compare_explicit_vs_symbolic, solve_parity_game, symbolic_game_stats,
    symbolic_attractor, _extract_states, _state_bdd, _preimage, _image,
    make_symbolic_chain_game, make_symbolic_ladder_game,
    make_symbolic_safety_game, make_symbolic_reachability_game,
    make_symbolic_buchi_game, batch_solve,
)
from parity_games import ParityGame, Player, Solution, zielonka, make_game, verify_solution
from bdd_model_checker import BDD


# ===== Helpers =====

def simple_game():
    """Even vertex 0 (prio 0) <-> Odd vertex 1 (prio 1)."""
    return make_game([(0, 0, 0), (1, 1, 1)], [(0, 1), (1, 0)])


def triangle_game():
    """3-vertex triangle: 0->1->2->0. Even owns 0,2. Odd owns 1."""
    return make_game(
        [(0, 0, 2), (1, 1, 1), (2, 0, 0)],
        [(0, 1), (1, 2), (2, 0)]
    )


def diamond_game():
    """Diamond: 0->{1,2}, 1->3, 2->3, 3->0. Even owns 0,3. Odd owns 1,2."""
    return make_game(
        [(0, 0, 0), (1, 1, 1), (2, 1, 1), (3, 0, 2)],
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)]
    )


def self_loop_game():
    """Vertex 0 (Even, prio 0) with self-loop -> Even wins."""
    return make_game([(0, 0, 0)], [(0, 0)])


def forced_loss_game():
    """Vertex 0 (Even, prio 1) self-loop -> Odd wins (highest inf-often = 1)."""
    return make_game([(0, 0, 1)], [(0, 0)])


# ===== Conversion Tests =====

class TestConversion:
    def test_explicit_to_symbolic_simple(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        assert spg.n_bits >= 1
        assert spg.max_priority == 1

    def test_roundtrip_simple(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        g2 = symbolic_to_explicit(spg)
        assert g2.vertices == g.vertices
        for v in g.vertices:
            assert g2.owner[v] == g.owner[v]
            assert g2.priority[v] == g.priority[v]

    def test_roundtrip_triangle(self):
        g = triangle_game()
        spg = explicit_to_symbolic(g)
        g2 = symbolic_to_explicit(spg)
        assert g2.vertices == g.vertices
        for v in g.vertices:
            assert g2.owner[v] == g.owner[v]
            assert g2.priority[v] == g.priority[v]
            assert g2.successors(v) == g.successors(v)

    def test_roundtrip_diamond(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        g2 = symbolic_to_explicit(spg)
        assert g2.vertices == g.vertices
        for v in g.vertices:
            assert g2.successors(v) == g.successors(v)

    def test_extract_states(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        states = _extract_states(spg, spg.vertices)
        assert states == {0, 1}

    def test_extract_owner(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        even_states = _extract_states(spg, spg.owner_even)
        assert even_states == {0}

    def test_extract_priorities(self):
        g = triangle_game()
        spg = explicit_to_symbolic(g)
        p0 = _extract_states(spg, spg.priority_bdds[0])
        p1 = _extract_states(spg, spg.priority_bdds[1])
        p2 = _extract_states(spg, spg.priority_bdds[2])
        assert 2 in p0
        assert 1 in p1
        assert 0 in p2


# ===== BDD Operations Tests =====

class TestBDDOps:
    def test_preimage_simple(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        # Predecessors of {1} should be {0}
        v1 = _state_bdd(spg.bdd, 1, spg.n_bits, spg.var_indices, spg.state_vars)
        pre = _preimage(spg, v1)
        pre_states = _extract_states(spg, pre)
        assert 0 in pre_states

    def test_preimage_diamond(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        # Predecessors of {3} should be {1, 2}
        v3 = _state_bdd(spg.bdd, 3, spg.n_bits, spg.var_indices, spg.state_vars)
        pre = _preimage(spg, v3)
        pre_states = _extract_states(spg, pre)
        assert pre_states == {1, 2}

    def test_image_simple(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        # Successors of {0} should be {1}
        v0 = _state_bdd(spg.bdd, 0, spg.n_bits, spg.var_indices, spg.state_vars)
        img = _image(spg, v0)
        img_states = _extract_states(spg, img)
        assert 1 in img_states

    def test_image_diamond(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        # Successors of {0} should be {1, 2}
        v0 = _state_bdd(spg.bdd, 0, spg.n_bits, spg.var_indices, spg.state_vars)
        img = _image(spg, v0)
        img_states = _extract_states(spg, img)
        assert img_states == {1, 2}


# ===== Attractor Tests =====

class TestAttractor:
    def test_attractor_self(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        # Attractor of {0} for Even: Even owns 0, and 1 (Odd) has only 0 as successor
        v0 = _state_bdd(spg.bdd, 0, spg.n_bits, spg.var_indices, spg.state_vars)
        attr = symbolic_attractor(spg, v0, True)
        attr_states = _extract_states(spg, attr)
        # Both vertices should be in attractor (Odd at 1 is forced to 0)
        assert attr_states == {0, 1}

    def test_attractor_diamond(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        # Attractor of {3} for Even: 3 is Even-owned
        # Pre of {3} = {1,2} (both Odd-owned, both forced into {3})
        # So Attr_Even({3}) = {0,1,2,3}
        v3 = _state_bdd(spg.bdd, 3, spg.n_bits, spg.var_indices, spg.state_vars)
        attr = symbolic_attractor(spg, v3, True)
        attr_states = _extract_states(spg, attr)
        assert attr_states == {0, 1, 2, 3}

    def test_attractor_restricted(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        # Restrict to {1, 2, 3}
        restrict = spg.bdd.FALSE
        for v in [1, 2, 3]:
            restrict = spg.bdd.OR(restrict,
                _state_bdd(spg.bdd, v, spg.n_bits, spg.var_indices, spg.state_vars))
        v3 = _state_bdd(spg.bdd, 3, spg.n_bits, spg.var_indices, spg.state_vars)
        attr = symbolic_attractor(spg, v3, True, restrict)
        attr_states = _extract_states(spg, attr)
        assert 3 in attr_states
        assert 0 not in attr_states


# ===== Solving Tests =====

class TestSolving:
    def test_self_loop_even_wins(self):
        g = self_loop_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert we == {0}
        assert wo == set()

    def test_forced_loss_odd_wins(self):
        g = forced_loss_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert we == set()
        assert wo == {0}

    def test_simple_game(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        # Verify against explicit
        explicit_sol = zielonka(g)
        assert we == explicit_sol.win_even
        assert wo == explicit_sol.win_odd

    def test_triangle_game(self):
        g = triangle_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        explicit_sol = zielonka(g)
        assert we == explicit_sol.win_even
        assert wo == explicit_sol.win_odd

    def test_diamond_game(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        explicit_sol = zielonka(g)
        assert we == explicit_sol.win_even
        assert wo == explicit_sol.win_odd

    def test_all_even_priority(self):
        """All vertices have even priority -> Even wins everything."""
        g = make_game(
            [(0, 0, 0), (1, 1, 2), (2, 0, 4)],
            [(0, 1), (1, 2), (2, 0)]
        )
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert we == {0, 1, 2}
        assert wo == set()

    def test_all_odd_priority(self):
        """All vertices have odd priority -> Odd wins everything."""
        g = make_game(
            [(0, 0, 1), (1, 1, 3), (2, 0, 1)],
            [(0, 1), (1, 2), (2, 0)]
        )
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert we == set()
        assert wo == {0, 1, 2}

    def test_choice_game(self):
        """Even at 0 chooses: go to 1 (prio 0, self-loop) or 2 (prio 1, self-loop).
        Even should choose 1 (even prio) and win."""
        g = make_game(
            [(0, 0, 0), (1, 0, 0), (2, 1, 1)],
            [(0, 1), (0, 2), (1, 1), (2, 2)]
        )
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert 0 in we
        assert 1 in we

    def test_five_vertex_game(self):
        """5-vertex game with mixed priorities."""
        g = make_game(
            [(0, 0, 2), (1, 1, 1), (2, 0, 0), (3, 1, 3), (4, 0, 2)],
            [(0, 1), (0, 2), (1, 3), (2, 4), (3, 0), (4, 0)]
        )
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_larger_game(self):
        """8-vertex game."""
        g = make_game(
            [(0, 0, 0), (1, 1, 1), (2, 0, 2), (3, 1, 1),
             (4, 0, 0), (5, 1, 3), (6, 0, 2), (7, 1, 1)],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
             (0, 4), (4, 0), (2, 6), (6, 2)]
        )
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']
        assert result['explicit']['valid']
        assert result['symbolic']['valid']


# ===== Strategy Tests =====

class TestStrategy:
    def test_strategy_extraction(self):
        g = simple_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        strat_e = extract_strategy(spg, sol.strategy_even)
        strat_o = extract_strategy(spg, sol.strategy_odd)
        # Strategies should map vertices to successors
        we, wo = extract_winning_sets(spg, sol)
        for v in strat_e:
            assert v in we
        for v in strat_o:
            assert v in wo

    def test_strategy_validity(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid, f"Invalid solution: {errors}"

    def test_strategy_choice_game(self):
        """Even should pick successor with even priority."""
        g = make_game(
            [(0, 0, 0), (1, 0, 0), (2, 1, 1)],
            [(0, 1), (0, 2), (1, 1), (2, 2)]
        )
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        strat_e = extract_strategy(spg, sol.strategy_even)
        # Vertex 0's strategy should go to 1 (not 2)
        if 0 in strat_e:
            assert strat_e[0] == 1


# ===== Verification Tests =====

class TestVerification:
    def test_verify_self_loop(self):
        g = self_loop_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_verify_triangle(self):
        g = triangle_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_verify_diamond(self):
        g = diamond_game()
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_verify_larger(self):
        g = make_game(
            [(0, 0, 0), (1, 1, 1), (2, 0, 2), (3, 1, 3), (4, 0, 0)],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (2, 4)]
        )
        spg = explicit_to_symbolic(g)
        sol = solve_symbolic(spg)
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid


# ===== Parametric Game Tests =====

class TestParametricGames:
    def test_chain_game_small(self):
        spg = make_symbolic_chain_game(4)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert len(we) + len(wo) == 4

    def test_chain_game_medium(self):
        spg = make_symbolic_chain_game(8)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert len(we) + len(wo) == 8
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_ladder_game_small(self):
        spg = make_symbolic_ladder_game(3)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert len(we) + len(wo) == 6

    def test_ladder_game_medium(self):
        spg = make_symbolic_ladder_game(4)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert len(we) + len(wo) == 8
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_safety_game(self):
        spg = make_symbolic_safety_game(
            4, bad={2}, even_states={0, 1},
            transitions=[(0, 1), (1, 2), (1, 3), (2, 2), (3, 3)]
        )
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        # Even should avoid vertex 2 by going 0->1->3
        assert 3 in we  # Safe vertex
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_reachability_game(self):
        spg = make_symbolic_reachability_game(
            4, target={3}, even_states={0, 2},
            transitions=[(0, 1), (0, 2), (1, 3), (2, 0)]
        )
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert 3 in we  # Target always reachable
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_buchi_game(self):
        spg = make_symbolic_buchi_game(
            3, accepting={1}, even_states={0, 2},
            transitions=[(0, 1), (1, 2), (2, 0)]
        )
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        # All vertices cycle through accepting vertex 1 -> Even wins all
        assert we == {0, 1, 2}


# ===== Comparison Tests =====

class TestComparison:
    def test_compare_simple(self):
        g = simple_game()
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']
        assert result['explicit']['valid']

    def test_compare_triangle(self):
        g = triangle_game()
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_compare_diamond(self):
        g = diamond_game()
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_compare_forced_loss(self):
        g = forced_loss_game()
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_compare_medium(self):
        """10-vertex game with various priorities."""
        g = make_game(
            [(i, i % 2, i % 4) for i in range(10)],
            [(i, (i + 1) % 10) for i in range(10)] +
            [(0, 5), (5, 0), (2, 7), (7, 2)]
        )
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']


# ===== High-Level API Tests =====

class TestHighLevelAPI:
    def test_solve_parity_game(self):
        g = diamond_game()
        result = solve_parity_game(g)
        assert result['valid']
        assert 'win_even' in result
        assert 'win_odd' in result
        assert 'strategy_even' in result
        assert 'strategy_odd' in result

    def test_symbolic_game_stats(self):
        g = triangle_game()
        spg = explicit_to_symbolic(g)
        stats = symbolic_game_stats(spg)
        assert stats['vertices'] == 3
        assert stats['max_priority'] == 2
        assert stats['even_vertices'] == 2
        assert stats['odd_vertices'] == 1

    def test_batch_solve(self):
        games = [simple_game(), triangle_game(), diamond_game()]
        results = batch_solve(games)
        assert len(results) == 3
        for r in results:
            assert r['valid']


# ===== Edge Cases =====

class TestEdgeCases:
    def test_single_vertex_even_prio(self):
        g = make_game([(0, 0, 0)], [(0, 0)])
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_single_vertex_odd_prio(self):
        g = make_game([(0, 1, 1)], [(0, 0)])
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_two_self_loops(self):
        """Two disconnected self-loops."""
        g = make_game([(0, 0, 0), (1, 1, 1)], [(0, 0), (1, 1)])
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']
        assert result['explicit']['win_even'] == {0}
        assert result['explicit']['win_odd'] == {1}

    def test_high_priority(self):
        g = make_game(
            [(0, 0, 6), (1, 1, 7), (2, 0, 0)],
            [(0, 1), (1, 2), (2, 0)]
        )
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_multiple_even_priorities(self):
        g = make_game(
            [(0, 0, 0), (1, 0, 2), (2, 0, 4), (3, 1, 1)],
            [(0, 1), (1, 2), (2, 3), (3, 0)]
        )
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_dead_end_even(self):
        """Dead-end Even vertex (no successors) -> Odd wins it."""
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(1, 0)
        g.add_edge(1, 1)
        # Vertex 0 has no successors -> Even can't move -> Odd wins 0
        # Vertex 1 (Odd) can go to 0 or self-loop -> Odd should go to 0
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_dead_end_odd(self):
        """Dead-end Odd vertex -> Even wins it."""
        g = ParityGame()
        g.add_vertex(0, Player.ODD, 1)
        g.add_vertex(1, Player.EVEN, 0)
        g.add_edge(1, 0)
        g.add_edge(1, 1)
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_non_contiguous_ids(self):
        """Vertices with non-contiguous IDs."""
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(3, Player.ODD, 1)
        g.add_vertex(7, Player.EVEN, 2)
        g.add_edge(0, 3)
        g.add_edge(3, 7)
        g.add_edge(7, 0)
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']

    def test_complete_bipartite(self):
        """Complete bipartite: all Even -> all Odd, all Odd -> all Even."""
        g = ParityGame()
        for i in range(4):
            g.add_vertex(i, Player.EVEN, 0)
            g.add_vertex(i + 4, Player.ODD, 1)
        for i in range(4):
            for j in range(4, 8):
                g.add_edge(i, j)
                g.add_edge(j, i)
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']


# ===== Stress Tests =====

class TestStress:
    def test_chain_12(self):
        """12-vertex chain."""
        spg = make_symbolic_chain_game(12)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert len(we) + len(wo) == 12

    def test_chain_16(self):
        """16-vertex chain (power of 2)."""
        spg = make_symbolic_chain_game(16)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert len(we) + len(wo) == 16

    def test_ladder_6(self):
        """12-vertex ladder."""
        spg = make_symbolic_ladder_game(6)
        sol = solve_symbolic(spg)
        we, wo = extract_winning_sets(spg, sol)
        assert len(we) + len(wo) == 12
        valid, errors = verify_symbolic_solution(spg, sol)
        assert valid

    def test_random_like_10(self):
        """10-vertex game with various edge patterns."""
        g = make_game(
            [(i, i % 2, (i * 3 + 1) % 5) for i in range(10)],
            [(i, (i * 2 + 1) % 10) for i in range(10)] +
            [(i, (i + 3) % 10) for i in range(10)]
        )
        result = compare_explicit_vs_symbolic(g)
        assert result['agree']
        assert result['symbolic']['valid']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
