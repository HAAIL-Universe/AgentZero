"""Tests for V162: Symbolic Energy Games."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

from energy_games import (
    EnergyGame, EnergyParityGame, Player,
    make_simple_energy_game, make_energy_parity_game,
    make_chain_energy_game, make_charging_game, make_choice_game,
    solve_energy, solve_energy_parity,
)
from symbolic_energy import (
    encode_energy_game, solve_symbolic_energy, solve_symbolic_energy_parity,
    symbolic_attractor, symbolic_reachability, symbolic_safety_check,
    compare_with_explicit, compare_energy_parity,
    make_symbolic_chain, make_symbolic_diamond, make_symbolic_grid,
    symbolic_energy_statistics,
    _symbolic_successors, _symbolic_predecessors, _extract_vertices,
    _int_to_bits, _make_vertex_bdd,
    SymbolicEncoding,
)


# ============================================================
# Section 1: Encoding basics
# ============================================================

class TestEncoding:
    def test_encode_simple_game(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 0, -1)],
            {0: Player.EVEN, 1: Player.ODD}
        )
        enc = encode_energy_game(game)
        assert enc.num_vertices == 2
        assert enc.num_bits >= 1
        assert len(enc.edges) == 2
        assert len(enc.weight_groups) == 2  # weight 1 and -1

    def test_encode_vertices_bdd(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 2, -1), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        enc = encode_energy_game(game)
        verts = _extract_vertices(enc, enc.vertices_bdd)
        assert verts == {0, 1, 2}

    def test_encode_owner_partition(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 2, -1), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        enc = encode_energy_game(game)
        even = _extract_vertices(enc, enc.owner_even)
        odd = _extract_vertices(enc, enc.owner_odd)
        assert even == {0, 2}
        assert odd == {1}

    def test_int_to_bits(self):
        assert _int_to_bits(0, 3) == (False, False, False)
        assert _int_to_bits(5, 3) == (True, False, True)
        assert _int_to_bits(7, 3) == (True, True, True)

    def test_encode_weight_groups(self):
        game = make_simple_energy_game(
            [(0, 1, 3), (1, 2, 3), (2, 0, -2)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        enc = encode_energy_game(game)
        assert 3 in enc.weight_groups
        assert -2 in enc.weight_groups

    def test_encode_single_vertex(self):
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_edge(0, 0, 1)
        enc = encode_energy_game(game)
        assert enc.num_vertices == 1
        assert enc.num_bits >= 1

    def test_encode_empty_game_raises(self):
        game = EnergyGame()
        with pytest.raises(ValueError):
            encode_energy_game(game)


# ============================================================
# Section 2: Symbolic operations
# ============================================================

class TestSymbolicOps:
    def test_successors(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (0, 2, 2), (1, 2, -1), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        enc = encode_energy_game(game)
        # Successors of {0} should be {1, 2}
        v0_bdd = _make_vertex_bdd(enc.bdd, 0, enc.curr_indices, enc.num_bits)
        succs = _extract_vertices(enc, _symbolic_successors(enc, v0_bdd))
        assert succs == {1, 2}

    def test_predecessors(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (0, 2, 2), (1, 2, -1), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        enc = encode_energy_game(game)
        # Predecessors of {2} should be {0, 1}
        verts = sorted(game.vertices)
        vert_to_idx = {v: i for i, v in enumerate(verts)}
        v2_bdd = _make_vertex_bdd(enc.bdd, vert_to_idx[2], enc.curr_indices, enc.num_bits)
        preds = _extract_vertices(enc, _symbolic_predecessors(enc, v2_bdd))
        assert preds == {0, 1}

    def test_successors_of_multiple(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 2, -1), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        enc = encode_energy_game(game)
        # Successors of {0, 1} should be {1, 2}
        v0 = _make_vertex_bdd(enc.bdd, 0, enc.curr_indices, enc.num_bits)
        v1 = _make_vertex_bdd(enc.bdd, 1, enc.curr_indices, enc.num_bits)
        both = enc.bdd.OR(v0, v1)
        succs = _extract_vertices(enc, _symbolic_successors(enc, both))
        assert succs == {1, 2}

    def test_reachability_full_cycle(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 2, -1), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        reached = symbolic_reachability(game, {0})
        assert reached == {0, 1, 2}

    def test_reachability_partial(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 2, -1)],  # no edge back
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        reached = symbolic_reachability(game, {0})
        assert reached == {0, 1, 2}
        reached2 = symbolic_reachability(game, {2})
        assert reached2 == {2}  # vertex 2 has no outgoing edges


# ============================================================
# Section 3: Symbolic attractor
# ============================================================

class TestAttractor:
    def test_attractor_single_vertex(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 0, -1)],
            {0: Player.EVEN, 1: Player.ODD}
        )
        enc = encode_energy_game(game)
        v0 = _make_vertex_bdd(enc.bdd, 0, enc.curr_indices, enc.num_bits)
        # Even's attractor to {0}: Even at 0, Odd at 1. 1->0 forced, so attr = {0, 1}
        attr = symbolic_attractor(enc, v0, Player.EVEN)
        verts = _extract_vertices(enc, attr)
        assert verts == {0, 1}

    def test_attractor_even_choice(self):
        # Even at 0 can choose 1 or 2. Odd at 1, 2.
        game = make_simple_energy_game(
            [(0, 1, 1), (0, 2, -1), (1, 0, 0), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.ODD}
        )
        enc = encode_energy_game(game)
        v1 = _make_vertex_bdd(enc.bdd, 1, enc.curr_indices, enc.num_bits)
        # Even's attractor to {1}: Even at 0 CAN go to 1, so 0 is attracted
        # Odd at 2 has only 2->0, not to 1, but 0 is now in attr, so 2 is attracted too
        attr = symbolic_attractor(enc, v1, Player.EVEN)
        verts = _extract_vertices(enc, attr)
        assert 0 in verts  # Even can choose to go to 1
        assert 1 in verts

    def test_attractor_odd_forces(self):
        # Odd at 0 chooses, Even at 1 forced
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 0, -1)],
            {0: Player.ODD, 1: Player.EVEN}
        )
        enc = encode_energy_game(game)
        v1 = _make_vertex_bdd(enc.bdd, 1, enc.curr_indices, enc.num_bits)
        # Odd's attractor to {1}: Odd at 0 can go to 1 -> attr = {0, 1}
        attr = symbolic_attractor(enc, v1, Player.ODD)
        verts = _extract_vertices(enc, attr)
        assert verts == {0, 1}


# ============================================================
# Section 4: Symbolic energy solving - basic
# ============================================================

class TestSolveBasic:
    def test_positive_cycle(self):
        # Single vertex, self-loop with +1: Even wins with energy 0
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_edge(0, 0, 1)
        result = solve_symbolic_energy(game)
        assert result.win_even == {0}
        assert result.min_energy[0] == 0

    def test_negative_cycle_even(self):
        # Even owns vertex with self-loop -1: energy depletes, Even loses
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_edge(0, 0, -1)
        result = solve_symbolic_energy(game)
        assert result.win_odd == {0}
        assert result.min_energy[0] is None

    def test_two_vertex_simple(self):
        game = make_simple_energy_game(
            [(0, 1, 2), (1, 0, -1)],
            {0: Player.EVEN, 1: Player.ODD}
        )
        result = solve_symbolic_energy(game)
        # Net cycle weight = 2 + (-1) = 1 > 0, so Even wins
        assert 0 in result.win_even
        assert 1 in result.win_even

    def test_dead_end_loses(self):
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_vertex(1, Player.EVEN)
        game.add_edge(0, 1, 1)
        # Vertex 1 has no successors -> Even loses from 1 and therefore from 0
        result = solve_symbolic_energy(game)
        assert result.win_odd == {0, 1}

    def test_empty_game(self):
        game = EnergyGame()
        result = solve_symbolic_energy(game)
        assert result.win_even == set()
        assert result.win_odd == set()


# ============================================================
# Section 5: Comparison with explicit solver
# ============================================================

class TestComparison:
    def test_chain_3(self):
        game = make_chain_energy_game(3)
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_chain_5(self):
        game = make_chain_energy_game(5)
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_charging_game(self):
        game = make_charging_game(4, charge=3, drain=1)
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_choice_game(self):
        game = make_choice_game()
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_diamond(self):
        game = make_symbolic_diamond()
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_all_even_positive(self):
        game = make_simple_energy_game(
            [(0, 1, 2), (1, 2, 3), (2, 0, 1)],
            {0: Player.EVEN, 1: Player.EVEN, 2: Player.EVEN}
        )
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_all_odd_negative(self):
        game = make_simple_energy_game(
            [(0, 1, -1), (1, 2, -2), (2, 0, -1)],
            {0: Player.ODD, 1: Player.ODD, 2: Player.ODD}
        )
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_mixed_weights(self):
        game = make_simple_energy_game(
            [(0, 1, 5), (1, 2, -3), (2, 0, -1), (0, 2, -2)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']


# ============================================================
# Section 6: Safety checking
# ============================================================

class TestSafety:
    def test_all_safe(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 0, -1)],
            {0: Player.EVEN, 1: Player.ODD}
        )
        result = symbolic_safety_check(game, {0, 1})
        assert result['safe_vertices'] == {0, 1}

    def test_partial_safe(self):
        # 0->1->2->0, safe set = {0, 1}
        game = make_simple_energy_game(
            [(0, 1, 1), (1, 2, -1), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.EVEN}
        )
        result = symbolic_safety_check(game, {0, 1})
        # From 1 (Odd), forced to go to 2 (unsafe). So 1 is unsafe.
        # From 0 (Even), forced to go to 1 (unsafe). So 0 is unsafe too.
        assert result['safe_count'] == 0

    def test_even_can_avoid_unsafe(self):
        # 0(Even)->1 or 0->2; 1->0; 2->0. Safe set = {0, 1}
        game = make_simple_energy_game(
            [(0, 1, 1), (0, 2, -1), (1, 0, 0), (2, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.ODD}
        )
        result = symbolic_safety_check(game, {0, 1})
        # Even at 0 can choose to go to 1 (safe), avoiding 2 (unsafe)
        # Odd at 1 must go to 0 (safe). So {0, 1} is safe for Even.
        assert 0 in result['safe_vertices']
        assert 1 in result['safe_vertices']


# ============================================================
# Section 7: Construction helpers
# ============================================================

class TestConstructors:
    def test_chain_basic(self):
        game = make_symbolic_chain(4)
        assert len(game.vertices) == 4
        result = solve_symbolic_energy(game)
        assert len(result.win_even) + len(result.win_odd) == 4

    def test_diamond_basic(self):
        game = make_symbolic_diamond()
        assert len(game.vertices) == 4
        result = solve_symbolic_energy(game)
        # Even chooses positive path, net +2, wins
        assert 0 in result.win_even

    def test_diamond_negative(self):
        game = make_symbolic_diamond(pos_weight=-1, neg_weight=-2)
        result = solve_symbolic_energy(game)
        # Both paths negative, Even loses
        assert 0 in result.win_odd

    def test_grid_2x2(self):
        game = make_symbolic_grid(2, 2)
        assert len(game.vertices) == 4
        result = solve_symbolic_energy(game)
        assert len(result.win_even) + len(result.win_odd) == 4

    def test_grid_3x3(self):
        game = make_symbolic_grid(3, 3)
        assert len(game.vertices) == 9
        result = solve_symbolic_energy(game)
        assert len(result.win_even) + len(result.win_odd) == 9


# ============================================================
# Section 8: Energy-parity solving
# ============================================================

class TestEnergyParity:
    def test_positive_even_priority(self):
        # Single vertex, self-loop +1, priority 0 (even): Even wins both
        game = make_energy_parity_game(
            [(0, 0, 1)],
            {0: Player.EVEN},
            {0: 0}
        )
        result = solve_symbolic_energy_parity(game)
        assert 0 in result.win_even

    def test_negative_cycle(self):
        # Self-loop -1, priority 0: energy fails even though parity OK
        game = make_energy_parity_game(
            [(0, 0, -1)],
            {0: Player.EVEN},
            {0: 0}
        )
        result = solve_symbolic_energy_parity(game)
        assert 0 in result.win_odd

    def test_odd_priority_loses(self):
        # Self-loop +1, priority 1 (odd): parity fails even though energy OK
        game = make_energy_parity_game(
            [(0, 0, 1)],
            {0: Player.EVEN},
            {0: 1}
        )
        result = solve_symbolic_energy_parity(game)
        assert 0 in result.win_odd

    def test_two_vertex_both_even_priority(self):
        # 0(Even,p=0)->1, 1(Odd,p=0)->0. Weight +2,-1.
        # Parity: min priority = 0 (even) -> parity OK
        # Energy: net +1 per cycle -> OK
        game = make_energy_parity_game(
            [(0, 1, 2), (1, 0, -1)],
            {0: Player.EVEN, 1: Player.ODD},
            {0: 0, 1: 0}
        )
        result = solve_symbolic_energy_parity(game)
        assert 0 in result.win_even
        assert 1 in result.win_even

    def test_comparison_api(self):
        game = make_energy_parity_game(
            [(0, 1, 2), (1, 0, -1)],
            {0: Player.EVEN, 1: Player.ODD},
            {0: 0, 1: 0}
        )
        comp = compare_energy_parity(game)
        assert comp['agree']


# ============================================================
# Section 9: Statistics
# ============================================================

class TestStatistics:
    def test_basic_stats(self):
        game = make_symbolic_chain(4)
        stats = symbolic_energy_statistics(game)
        assert stats['num_vertices'] == 4
        assert stats['num_edges'] == 4
        assert stats['num_bits'] >= 2
        assert 'win_even' in stats
        assert 'win_odd' in stats

    def test_diamond_stats(self):
        game = make_symbolic_diamond()
        stats = symbolic_energy_statistics(game)
        assert stats['num_vertices'] == 4
        assert stats['num_edges'] == 5
        assert stats['weight_groups'] >= 2

    def test_encoding_stats_in_result(self):
        game = make_symbolic_chain(6)
        result = solve_symbolic_energy(game)
        assert 'num_vertices' in result.encoding_stats
        assert 'num_bits' in result.encoding_stats
        assert result.encoding_stats['num_vertices'] == 6


# ============================================================
# Section 10: Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_vertex_self_loop(self):
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_edge(0, 0, 0)
        result = solve_symbolic_energy(game)
        assert 0 in result.win_even
        assert result.min_energy[0] == 0

    def test_zero_weight_cycle(self):
        game = make_simple_energy_game(
            [(0, 1, 0), (1, 0, 0)],
            {0: Player.EVEN, 1: Player.ODD}
        )
        result = solve_symbolic_energy(game)
        assert result.win_even == {0, 1}

    def test_large_weights(self):
        game = make_simple_energy_game(
            [(0, 1, 100), (1, 0, -99)],
            {0: Player.EVEN, 1: Player.ODD}
        )
        result = solve_symbolic_energy(game)
        assert 0 in result.win_even

    def test_strategy_exists_for_winners(self):
        game = make_choice_game()
        result = solve_symbolic_energy(game)
        for v in result.win_even:
            if game.owner[v] == Player.EVEN:
                assert v in result.strategy_even

    def test_multiple_edges_same_weight(self):
        game = make_simple_energy_game(
            [(0, 1, 1), (0, 2, 1), (1, 0, -1), (2, 0, -1)],
            {0: Player.EVEN, 1: Player.ODD, 2: Player.ODD}
        )
        enc = encode_energy_game(game)
        assert 1 in enc.weight_groups
        assert -1 in enc.weight_groups
        result = solve_symbolic_energy(game)
        assert result.win_even == {0, 1, 2}

    def test_disconnected_components(self):
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_vertex(1, Player.ODD)
        game.add_vertex(2, Player.EVEN)
        game.add_vertex(3, Player.ODD)
        game.add_edge(0, 1, 1)
        game.add_edge(1, 0, -1)
        game.add_edge(2, 3, 2)
        game.add_edge(3, 2, -1)
        result = solve_symbolic_energy(game)
        # Component {0,1}: net 0, Even wins
        assert 0 in result.win_even
        assert 1 in result.win_even
        # Component {2,3}: net +1, Even wins
        assert 2 in result.win_even
        assert 3 in result.win_even


# ============================================================
# Section 11: Larger games comparison
# ============================================================

class TestLargerGames:
    def test_chain_8(self):
        game = make_chain_energy_game(8)
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_charging_6(self):
        game = make_charging_game(6, charge=5, drain=2)
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_grid_2x3(self):
        game = make_symbolic_grid(2, 3)
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']

    def test_complex_game(self):
        """A game with multiple weight classes and mixed ownership."""
        game = EnergyGame()
        for i in range(6):
            game.add_vertex(i, Player.EVEN if i % 3 == 0 else Player.ODD)
        game.add_edge(0, 1, 3)
        game.add_edge(0, 2, -1)
        game.add_edge(1, 3, -2)
        game.add_edge(2, 3, 1)
        game.add_edge(3, 4, 2)
        game.add_edge(3, 5, -3)
        game.add_edge(4, 0, 1)
        game.add_edge(5, 0, 4)
        comp = compare_with_explicit(game)
        assert comp['agree_winning']
        assert comp['agree_energy']


# ============================================================
# Section 12: Reachability
# ============================================================

class TestReachability:
    def test_full_reachability(self):
        game = make_symbolic_chain(4)
        reached = symbolic_reachability(game, {0})
        assert reached == {0, 1, 2, 3}

    def test_single_vertex_reachability(self):
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_vertex(1, Player.ODD)
        game.add_edge(0, 1, 1)
        reached = symbolic_reachability(game, {1})
        assert reached == {1}

    def test_multiple_start_vertices(self):
        game = EnergyGame()
        game.add_vertex(0, Player.EVEN)
        game.add_vertex(1, Player.ODD)
        game.add_vertex(2, Player.EVEN)
        game.add_edge(0, 1, 1)
        game.add_edge(2, 1, 1)
        reached = symbolic_reachability(game, {0, 2})
        assert reached == {0, 1, 2}

    def test_empty_start(self):
        game = make_symbolic_chain(3)
        reached = symbolic_reachability(game, set())
        assert reached == set()
