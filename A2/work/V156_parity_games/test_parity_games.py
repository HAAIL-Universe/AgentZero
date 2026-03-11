"""Tests for V156: Parity Games."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from parity_games import (
    ParityGame, Player, Solution,
    attractor, zielonka, small_progress_measures, priority_promotion,
    verify_solution, simulate_play, make_game, make_safety_game,
    make_reachability_game, make_buchi_game, make_co_buchi_game,
    solve_all, compare_algorithms, game_statistics, game_summary,
)


# =========================================================================
# Basic Game Construction
# =========================================================================

class TestGameConstruction:
    def test_empty_game(self):
        g = ParityGame()
        assert len(g.vertices) == 0

    def test_add_vertex(self):
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        assert 0 in g.vertices
        assert g.owner[0] == Player.EVEN
        assert g.priority[0] == 2

    def test_add_edge(self):
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1)
        assert 1 in g.successors(0)
        assert 0 in g.predecessors(1)

    def test_subgame(self):
        g = make_game([(0, 0, 0), (1, 1, 1), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        sub = g.subgame({0, 2})
        assert sub.vertices == {0, 2}
        assert 1 not in sub.edges.get(0, set())
        assert 0 in sub.edges.get(2, set())

    def test_dead_ends(self):
        g = make_game([(0, 0, 0), (1, 1, 1)], [(0, 1)])
        dead = g.has_dead_ends()
        assert 1 in dead
        assert 0 not in dead

    def test_max_priority(self):
        g = make_game([(0, 0, 3), (1, 1, 5), (2, 0, 1)], [(0, 1), (1, 2)])
        assert g.max_priority() == 5

    def test_vertices_with_priority(self):
        g = make_game([(0, 0, 2), (1, 1, 2), (2, 0, 3)], [(0, 1)])
        assert g.vertices_with_priority(2) == {0, 1}
        assert g.vertices_with_priority(3) == {2}

    def test_make_game(self):
        g = make_game([(0, 0, 0), (1, 1, 1)], [(0, 1), (1, 0)])
        assert len(g.vertices) == 2
        assert g.owner[0] == Player.EVEN
        assert g.owner[1] == Player.ODD

    def test_player_opponent(self):
        assert Player.EVEN.opponent == Player.ODD
        assert Player.ODD.opponent == Player.EVEN


# =========================================================================
# Attractor Computation
# =========================================================================

class TestAttractor:
    def test_single_vertex(self):
        g = make_game([(0, 0, 0)], [])
        attr = attractor(g, {0}, Player.EVEN)
        assert attr == {0}

    def test_player_can_reach(self):
        # Even owns 0, can choose to go to 1
        g = make_game([(0, 0, 0), (1, 0, 1)], [(0, 1)])
        attr = attractor(g, {1}, Player.EVEN)
        assert attr == {0, 1}

    def test_opponent_forced(self):
        # Odd owns 0, only successor is 1
        g = make_game([(0, 1, 0), (1, 0, 1)], [(0, 1)])
        attr = attractor(g, {1}, Player.EVEN)
        assert attr == {0, 1}  # Odd is forced to go to 1

    def test_opponent_can_escape(self):
        # Odd owns 0, can go to 1 or 2
        g = make_game([(0, 1, 0), (1, 0, 1), (2, 0, 0)], [(0, 1), (0, 2)])
        attr = attractor(g, {1}, Player.EVEN)
        assert attr == {1}  # Odd can escape to 2

    def test_chain_attractor(self):
        g = make_game([(0, 0, 0), (1, 0, 1), (2, 0, 2)],
                      [(0, 1), (1, 2)])
        attr = attractor(g, {2}, Player.EVEN)
        assert attr == {0, 1, 2}

    def test_restricted_attractor(self):
        g = make_game([(0, 0, 0), (1, 0, 1), (2, 0, 2)],
                      [(0, 1), (1, 2)])
        attr = attractor(g, {2}, Player.EVEN, restrict={1, 2})
        assert attr == {1, 2}
        assert 0 not in attr


# =========================================================================
# Zielonka's Algorithm
# =========================================================================

class TestZielonka:
    def test_empty_game(self):
        g = ParityGame()
        sol = zielonka(g)
        assert sol.win_even == set()
        assert sol.win_odd == set()

    def test_single_even_vertex(self):
        # Self-loop with even priority -> Even wins
        g = make_game([(0, 0, 0)], [(0, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0}

    def test_single_odd_vertex(self):
        # Self-loop with odd priority -> Odd wins
        g = make_game([(0, 0, 1)], [(0, 0)])
        sol = zielonka(g)
        assert sol.win_odd == {0}

    def test_two_vertex_cycle_even_wins(self):
        # 0(Even,p=2) -> 1(Odd,p=0) -> 0
        # Max priority 2 (even) seen infinitely -> Even wins
        g = make_game([(0, 0, 2), (1, 1, 0)], [(0, 1), (1, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0, 1}

    def test_two_vertex_cycle_odd_wins(self):
        # 0(Even,p=1) -> 1(Odd,p=0) -> 0
        # Max priority 1 (odd) seen infinitely -> Odd wins
        g = make_game([(0, 0, 1), (1, 1, 0)], [(0, 1), (1, 0)])
        sol = zielonka(g)
        assert sol.win_odd == {0, 1}

    def test_choice_game(self):
        # Even at 0 can go to 1(p=2, self-loop) or 2(p=1, self-loop)
        # Even should choose 1 (even priority)
        g = make_game([(0, 0, 0), (1, 0, 2), (2, 0, 1)],
                      [(0, 1), (0, 2), (1, 1), (2, 2)])
        sol = zielonka(g)
        assert 0 in sol.win_even
        assert 1 in sol.win_even

    def test_three_priority_game(self):
        # 0(E,p=0) -> 1(O,p=1) -> 2(E,p=2) -> 0
        # Cycle priorities: 0,1,2. Max inf-often = 2 (even) -> Even wins
        g = make_game([(0, 0, 0), (1, 1, 1), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0, 1, 2}

    def test_odd_dominates(self):
        # 0(E,p=0) -> 1(O,p=3) -> 0
        # Max priority 3 (odd) -> Odd wins
        g = make_game([(0, 0, 0), (1, 1, 3)], [(0, 1), (1, 0)])
        sol = zielonka(g)
        assert sol.win_odd == {0, 1}

    def test_split_winning_regions(self):
        # Two disconnected components with different winners
        # Component 1: 0(E,p=2) <-> 1(O,p=0) -- Even wins
        # Component 2: 2(E,p=1) <-> 3(O,p=0) -- Odd wins
        g = make_game([(0, 0, 2), (1, 1, 0), (2, 0, 1), (3, 1, 0)],
                      [(0, 1), (1, 0), (2, 3), (3, 2)])
        sol = zielonka(g)
        assert sol.win_even == {0, 1}
        assert sol.win_odd == {2, 3}

    def test_diamond_game(self):
        # Even at 0 chooses left (1, p=2) or right (2, p=3)
        # Both lead to 3 (self-loop, p=0)
        # But 1 has even priority, 2 has odd
        # In the cycle 0->1->3->... max priority is max(0,2,0,...) = 2 (even) -> Even wins via 1
        g = make_game([(0, 0, 0), (1, 0, 2), (2, 0, 3), (3, 1, 0)],
                      [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)])
        sol = zielonka(g)
        # Even can choose to go through 1 (priority 2), avoiding 2 (priority 3)
        assert 0 in sol.win_even

    def test_solution_valid(self):
        g = make_game([(0, 0, 2), (1, 1, 1), (2, 0, 0)],
                      [(0, 1), (1, 2), (2, 0), (1, 0)])
        sol = zielonka(g)
        valid, errors = verify_solution(g, sol)
        assert valid, errors


# =========================================================================
# Small Progress Measures
# =========================================================================

class TestSPM:
    def test_empty_game(self):
        g = ParityGame()
        sol = small_progress_measures(g)
        assert sol.win_even == set()

    def test_single_even(self):
        g = make_game([(0, 0, 0)], [(0, 0)])
        sol = small_progress_measures(g)
        assert sol.win_even == {0}

    def test_single_odd(self):
        g = make_game([(0, 0, 1)], [(0, 0)])
        sol = small_progress_measures(g)
        assert sol.win_odd == {0}

    def test_all_even_priorities(self):
        g = make_game([(0, 0, 0), (1, 1, 2)], [(0, 1), (1, 0)])
        sol = small_progress_measures(g)
        assert sol.win_even == {0, 1}

    def test_agrees_with_zielonka(self):
        g = make_game([(0, 0, 2), (1, 1, 1), (2, 0, 0)],
                      [(0, 1), (1, 2), (2, 0), (1, 0)])
        z = zielonka(g)
        s = small_progress_measures(g)
        assert z.win_even == s.win_even
        assert z.win_odd == s.win_odd

    def test_split_regions(self):
        g = make_game([(0, 0, 2), (1, 1, 0), (2, 0, 1), (3, 1, 0)],
                      [(0, 1), (1, 0), (2, 3), (3, 2)])
        sol = small_progress_measures(g)
        assert sol.win_even == {0, 1}
        assert sol.win_odd == {2, 3}

    def test_choice_game(self):
        g = make_game([(0, 0, 0), (1, 0, 2), (2, 0, 1)],
                      [(0, 1), (0, 2), (1, 1), (2, 2)])
        sol = small_progress_measures(g)
        assert 0 in sol.win_even
        assert 1 in sol.win_even

    def test_solution_valid(self):
        g = make_game([(0, 0, 0), (1, 1, 3), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        sol = small_progress_measures(g)
        valid, errors = verify_solution(g, sol)
        assert valid, errors


# =========================================================================
# Priority Promotion
# =========================================================================

class TestPriorityPromotion:
    def test_empty_game(self):
        g = ParityGame()
        sol = priority_promotion(g)
        assert sol.win_even == set()

    def test_single_even(self):
        g = make_game([(0, 0, 0)], [(0, 0)])
        sol = priority_promotion(g)
        assert sol.win_even == {0}

    def test_single_odd(self):
        g = make_game([(0, 0, 1)], [(0, 0)])
        sol = priority_promotion(g)
        assert sol.win_odd == {0}

    def test_agrees_with_zielonka(self):
        g = make_game([(0, 0, 2), (1, 1, 1), (2, 0, 0)],
                      [(0, 1), (1, 2), (2, 0), (1, 0)])
        z = zielonka(g)
        p = priority_promotion(g)
        assert z.win_even == p.win_even
        assert z.win_odd == p.win_odd

    def test_split_regions(self):
        g = make_game([(0, 0, 2), (1, 1, 0), (2, 0, 1), (3, 1, 0)],
                      [(0, 1), (1, 0), (2, 3), (3, 2)])
        sol = priority_promotion(g)
        assert sol.win_even == {0, 1}
        assert sol.win_odd == {2, 3}

    def test_solution_valid(self):
        g = make_game([(0, 0, 0), (1, 1, 3), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        sol = priority_promotion(g)
        valid, errors = verify_solution(g, sol)
        assert valid, errors


# =========================================================================
# Algorithm Agreement
# =========================================================================

class TestAlgorithmAgreement:
    def _check_agree(self, g):
        results = compare_algorithms(g)
        assert results['all_agree'], f"Disagreement: {results}"
        for name in ['zielonka', 'spm', 'priority_promotion']:
            assert results[name]['valid'], f"{name} invalid: {results[name]['errors']}"

    def test_simple_cycle(self):
        g = make_game([(0, 0, 2), (1, 1, 0)], [(0, 1), (1, 0)])
        self._check_agree(g)

    def test_three_node_cycle(self):
        g = make_game([(0, 0, 0), (1, 1, 1), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        self._check_agree(g)

    def test_diamond(self):
        g = make_game([(0, 0, 0), (1, 0, 2), (2, 0, 3), (3, 1, 0)],
                      [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)])
        self._check_agree(g)

    def test_disconnected(self):
        g = make_game([(0, 0, 2), (1, 1, 0), (2, 0, 1), (3, 1, 0)],
                      [(0, 1), (1, 0), (2, 3), (3, 2)])
        self._check_agree(g)

    def test_larger_game(self):
        # 6 vertices with mixed priorities
        g = make_game([
            (0, 0, 0), (1, 1, 1), (2, 0, 2),
            (3, 1, 3), (4, 0, 0), (5, 1, 2)
        ], [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
            (2, 0), (4, 2)
        ])
        self._check_agree(g)

    def test_many_priorities(self):
        # 5 vertices with priorities 0-4
        g = make_game([
            (0, 0, 0), (1, 1, 1), (2, 0, 2), (3, 1, 3), (4, 0, 4)
        ], [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0)
        ])
        self._check_agree(g)


# =========================================================================
# Verification
# =========================================================================

class TestVerification:
    def test_valid_solution(self):
        g = make_game([(0, 0, 0)], [(0, 0)])
        sol = Solution(win_even={0}, strategy_even={0: 0})
        valid, errors = verify_solution(g, sol)
        assert valid

    def test_incomplete_partition(self):
        g = make_game([(0, 0, 0), (1, 1, 1)], [(0, 1), (1, 0)])
        sol = Solution(win_even={0})  # Missing vertex 1
        valid, errors = verify_solution(g, sol)
        assert not valid

    def test_overlapping_regions(self):
        g = make_game([(0, 0, 0)], [(0, 0)])
        sol = Solution(win_even={0}, win_odd={0})
        valid, errors = verify_solution(g, sol)
        assert not valid

    def test_bad_strategy(self):
        g = make_game([(0, 0, 0), (1, 1, 1)], [(0, 1), (1, 0)])
        sol = Solution(win_even={0, 1}, strategy_even={0: 1})
        # Even's strategy leads 0 to 1, but 1 should be in win_even
        # Odd at 1 can only go to 0, which is in win_even -> valid
        valid, errors = verify_solution(g, sol)
        assert valid


# =========================================================================
# Simulation
# =========================================================================

class TestSimulation:
    def test_simple_play(self):
        g = make_game([(0, 0, 0), (1, 1, 2)], [(0, 1), (1, 0)])
        sol = zielonka(g)
        play = simulate_play(g, sol, 0, max_steps=10)
        assert len(play) > 0
        assert play[0][0] == 0

    def test_self_loop(self):
        g = make_game([(0, 0, 2)], [(0, 0)])
        sol = zielonka(g)
        play = simulate_play(g, sol, 0, max_steps=5)
        assert all(v == 0 for v, _ in play)


# =========================================================================
# Safety Games
# =========================================================================

class TestSafetyGame:
    def test_safe_game(self):
        # No bad states -> Even wins everything
        g = make_safety_game(3, set(), {0, 1, 2}, [(0, 1), (1, 2), (2, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0, 1, 2}

    def test_unavoidable_bad(self):
        # 0 -> 1(bad) -> 0. Even at 0 must go to 1.
        g = make_safety_game(2, {1}, {0}, [(0, 1), (1, 0)])
        sol = zielonka(g)
        assert sol.win_odd == {0, 1}

    def test_avoidable_bad(self):
        # 0(Even) can go to 1(safe, self-loop) or 2(bad)
        g = make_safety_game(3, {2}, {0}, [(0, 1), (0, 2), (1, 1), (2, 2)])
        sol = zielonka(g)
        assert 0 in sol.win_even
        assert 1 in sol.win_even

    def test_opponent_forces_bad(self):
        # 0(Odd) goes to 1(bad)
        g = make_safety_game(2, {1}, set(), [(0, 1), (1, 1)])
        sol = zielonka(g)
        assert sol.win_odd == {0, 1}


# =========================================================================
# Reachability Games
# =========================================================================

class TestReachabilityGame:
    def test_can_reach(self):
        # 0(Even) -> 1 -> 2(target)
        g = make_reachability_game(3, {2}, {0, 1}, [(0, 1), (1, 2)])
        sol = zielonka(g)
        assert 0 in sol.win_even

    def test_cannot_reach(self):
        # 0(Even) -> 1(self-loop), no path to target 2
        g = make_reachability_game(3, {2}, {0}, [(0, 1), (1, 1)])
        sol = zielonka(g)
        assert 0 in sol.win_odd

    def test_opponent_blocks(self):
        # 0(Even)->1(Odd), 1 can go to 2(target) or 3(safe loop)
        g = make_reachability_game(4, {2}, {0}, [(0, 1), (1, 2), (1, 3), (3, 3)])
        sol = zielonka(g)
        # Odd at 1 will go to 3, avoiding the target
        assert 0 in sol.win_odd


# =========================================================================
# Buchi Games
# =========================================================================

class TestBuchiGame:
    def test_accepting_cycle(self):
        # 0(accepting) <-> 1. Both visit accepting state infinitely.
        g = make_buchi_game(2, {0}, {0, 1}, [(0, 1), (1, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0, 1}

    def test_no_accepting_reachable(self):
        # 0 <-> 1, accepting={2} not reachable
        g = make_buchi_game(3, {2}, {0, 1}, [(0, 1), (1, 0), (2, 2)])
        sol = zielonka(g)
        assert 0 in sol.win_odd
        assert 1 in sol.win_odd
        assert 2 in sol.win_even

    def test_opponent_avoids_accepting(self):
        # 0(Even)->1(Odd), Odd can go to 2(accepting, back to 0) or 3(non-accepting loop)
        g = make_buchi_game(4, {2}, {0}, [(0, 1), (1, 2), (1, 3), (2, 0), (3, 3)])
        sol = zielonka(g)
        # Odd will avoid accepting state by going to 3
        assert 0 in sol.win_odd


# =========================================================================
# Co-Buchi Games
# =========================================================================

class TestCoBuchiGame:
    def test_no_rejecting(self):
        g = make_co_buchi_game(2, set(), {0, 1}, [(0, 1), (1, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0, 1}

    def test_rejecting_cycle(self):
        # Forced through rejecting state forever
        g = make_co_buchi_game(2, {1}, {0}, [(0, 1), (1, 0)])
        sol = zielonka(g)
        assert sol.win_odd == {0, 1}

    def test_can_avoid_rejecting(self):
        # 0(Even) can go to 1(safe loop) or 2(rejecting loop)
        g = make_co_buchi_game(3, {2}, {0}, [(0, 1), (0, 2), (1, 1), (2, 2)])
        sol = zielonka(g)
        assert 0 in sol.win_even


# =========================================================================
# Statistics and Summary
# =========================================================================

class TestStatistics:
    def test_game_statistics(self):
        g = make_game([(0, 0, 0), (1, 1, 1), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        stats = game_statistics(g)
        assert stats['vertices'] == 3
        assert stats['edges'] == 3
        assert stats['max_priority'] == 2
        assert stats['num_priorities'] == 3
        assert stats['even_vertices'] == 2
        assert stats['odd_vertices'] == 1
        assert stats['dead_ends'] == 0

    def test_game_summary(self):
        g = make_game([(0, 0, 2), (1, 1, 0)], [(0, 1), (1, 0)])
        s = game_summary(g)
        assert "Parity Game" in s
        assert "W0=" in s
        assert "Valid: True" in s


# =========================================================================
# Edge Cases
# =========================================================================

class TestEdgeCases:
    def test_self_loop_even_priority(self):
        g = make_game([(0, 0, 4)], [(0, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0}
        valid, _ = verify_solution(g, sol)
        assert valid

    def test_self_loop_odd_priority(self):
        g = make_game([(0, 1, 5)], [(0, 0)])
        sol = zielonka(g)
        assert sol.win_odd == {0}

    def test_high_priorities(self):
        g = make_game([(0, 0, 10), (1, 1, 11)], [(0, 1), (1, 0)])
        sol = zielonka(g)
        # Max priority 11 (odd) -> Odd wins
        assert sol.win_odd == {0, 1}

    def test_all_same_priority_even(self):
        g = make_game([(0, 0, 2), (1, 1, 2), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        sol = zielonka(g)
        assert sol.win_even == {0, 1, 2}

    def test_all_same_priority_odd(self):
        g = make_game([(0, 0, 3), (1, 1, 3), (2, 0, 3)],
                      [(0, 1), (1, 2), (2, 0)])
        sol = zielonka(g)
        assert sol.win_odd == {0, 1, 2}

    def test_long_chain_to_cycle(self):
        # 0->1->2->3->4->3 (cycle at 3-4)
        g = make_game([
            (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 2), (4, 1, 0)
        ], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 3)])
        sol = zielonka(g)
        # Cycle 3->4->3, max priority = 2 (even) -> Even wins
        assert sol.win_even == {0, 1, 2, 3, 4}

    def test_two_cycles_different_winners(self):
        # Cycle A: 0<->1, priorities 2,0 -> Even wins
        # Cycle B: 2<->3, priorities 3,0 -> Odd wins
        # 4(Even) can choose: go to cycle A or cycle B
        g = make_game([
            (0, 0, 2), (1, 1, 0), (2, 0, 3), (3, 1, 0), (4, 0, 0)
        ], [(0, 1), (1, 0), (2, 3), (3, 2), (4, 0), (4, 2)])
        sol = zielonka(g)
        # Even at 4 should choose cycle A
        assert 4 in sol.win_even
        assert 0 in sol.win_even
        assert 1 in sol.win_even

    def test_dead_end_even_loses(self):
        # Even at 0, no successors -> Even loses (can't move)
        g = make_game([(0, 0, 0)], [])
        sol = zielonka(g)
        assert 0 in sol.win_odd

    def test_dead_end_odd_loses(self):
        # Odd at 0, no successors -> Odd loses
        g = make_game([(0, 1, 1)], [])
        sol = zielonka(g)
        assert 0 in sol.win_even


# =========================================================================
# Solve All and Compare
# =========================================================================

class TestSolveAll:
    def test_solve_all(self):
        g = make_game([(0, 0, 2), (1, 1, 1)], [(0, 1), (1, 0)])
        results = solve_all(g)
        assert len(results) == 3
        # All should agree
        for sol in results.values():
            assert sol.win_even == results['zielonka'].win_even

    def test_compare_algorithms(self):
        g = make_game([(0, 0, 0), (1, 1, 1), (2, 0, 2)],
                      [(0, 1), (1, 2), (2, 0)])
        result = compare_algorithms(g)
        assert result['all_agree']
        for name in ['zielonka', 'spm', 'priority_promotion']:
            assert result[name]['valid']


# =========================================================================
# Complex Games
# =========================================================================

class TestComplexGames:
    def test_mutual_exclusion_game(self):
        """Model a simple mutual exclusion protocol as a parity game."""
        # States: (idle, idle), (req, idle), (idle, req), (cs, idle), (idle, cs)
        # Encoded as vertices 0-4
        # Process 1 (Even) wants to reach CS (vertex 3)
        # Process 2 (Odd) wants to reach CS (vertex 4)
        # Priority: CS for process 1 = 2 (even), CS for process 2 = 1 (odd)
        g = make_game([
            (0, 0, 0),  # (idle, idle) - Even's turn
            (1, 1, 0),  # (req, idle) - Odd's turn
            (2, 0, 0),  # (idle, req) - Even's turn
            (3, 0, 2),  # (cs, idle) - Even in CS
            (4, 1, 1),  # (idle, cs) - Odd in CS
        ], [
            (0, 1), (0, 2),  # idle -> request
            (1, 3), (1, 2),  # req -> cs or other requests
            (2, 4), (2, 1),  # req -> cs or other requests
            (3, 0),          # cs -> release (back to idle)
            (4, 0),          # cs -> release
        ])
        sol = zielonka(g)
        valid, _ = verify_solution(g, sol)
        assert valid

    def test_streett_game(self):
        """A game encoding a Streett acceptance condition (fairness)."""
        # Streett: for each pair (R_i, G_i), if R_i visited inf often, then G_i visited inf often
        # Encode as parity: R=3, G=2, other=0
        g = make_game([
            (0, 0, 0),  # neutral
            (1, 1, 3),  # request (odd priority -> bad if inf-often without grant)
            (2, 0, 2),  # grant (even priority -> good)
        ], [
            (0, 1), (1, 2), (2, 0), (1, 0)
        ])
        sol = zielonka(g)
        valid, _ = verify_solution(g, sol)
        assert valid

    def test_rabin_game(self):
        """Game with Rabin-like acceptance (dual of Streett)."""
        # Rabin pair (L, U): visit L finitely often AND U infinitely often
        # Encode: L=1 (bad odd), U=2 (good even), other=0
        g = make_game([
            (0, 0, 0), (1, 0, 1), (2, 1, 2), (3, 0, 0)
        ], [
            (0, 1), (0, 2), (1, 3), (2, 3), (3, 0),
            (2, 0)  # Odd can try to keep visiting L
        ])
        sol = zielonka(g)
        valid, _ = verify_solution(g, sol)
        assert valid

    def test_nested_cycles(self):
        """Game with nested cycles of different priorities."""
        # Outer cycle: 0->1->2->0, inner cycle: 1->3->1
        # 0(p=0), 1(p=1), 2(p=4), 3(p=3)
        g = make_game([
            (0, 0, 0), (1, 0, 1), (2, 0, 4), (3, 1, 3)
        ], [
            (0, 1), (1, 2), (2, 0),  # outer cycle
            (1, 3), (3, 1),           # inner cycle
        ])
        sol = zielonka(g)
        # Even at 1 should choose outer cycle (reaches p=4, even)
        # avoiding inner cycle (p=3, odd)
        assert 0 in sol.win_even
        valid, _ = verify_solution(g, sol)
        assert valid


# =========================================================================
# Larger Structured Games
# =========================================================================

class TestLargerGames:
    def test_ladder_game(self):
        """Ladder game: two parallel chains with cross-links."""
        n = 5
        verts = []
        edges = []
        for i in range(n):
            verts.append((2*i, 0, i % 3))      # top row (Even)
            verts.append((2*i+1, 1, (i+1) % 3)) # bottom row (Odd)
            edges.append((2*i, 2*i+1))           # top -> bottom
            if i < n - 1:
                edges.append((2*i+1, 2*(i+1)))   # bottom -> next top
                edges.append((2*i, 2*(i+1)))      # top -> next top
        # Close the ladder
        edges.append((2*(n-1)+1, 0))
        edges.append((2*(n-1), 0))

        g = make_game(verts, edges)
        results = compare_algorithms(g)
        assert results['all_agree']

    def test_10_vertex_game(self):
        """A 10-vertex game with varied structure."""
        verts = [(i, i % 2, (i * 3 + 1) % 5) for i in range(10)]
        edges = []
        for i in range(10):
            edges.append((i, (i + 1) % 10))
            edges.append((i, (i + 3) % 10))
        g = make_game(verts, edges)
        results = compare_algorithms(g)
        assert results['all_agree']
        for name in ['zielonka', 'spm', 'priority_promotion']:
            assert results[name]['valid']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
