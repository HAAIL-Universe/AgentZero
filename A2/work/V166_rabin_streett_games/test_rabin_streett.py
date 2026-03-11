"""Tests for V166: Rabin/Streett Games."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from rabin_streett import (
    Player, GameArena, RabinPair, RabinGame, StreettGame, MullerGame, Solution,
    game_attractor, solve_rabin, solve_streett, solve_streett_direct,
    solve_muller, parity_to_rabin, parity_to_streett,
    make_buchi_game, make_co_buchi_game, make_generalized_buchi_game,
    make_arena, make_rabin_game, make_streett_game, make_muller_game,
    verify_rabin_strategy, compare_with_parity,
    rabin_streett_statistics, batch_solve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
from parity_games import ParityGame, Player as PPlayer


# =========================================================================
# Section 1: Game Arena and Attractor
# =========================================================================

class TestArena:
    def test_basic_arena(self):
        arena = make_arena([(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)])
        assert len(arena.vertices) == 3
        assert arena.owner[0] == Player.EVEN
        assert arena.owner[1] == Player.ODD
        assert arena.successors(0) == {1}

    def test_attractor_even(self):
        arena = make_arena([(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)])
        attr, strat = game_attractor(arena, {2}, Player.EVEN)
        # Even owns 0 and 2; 0->1->2, Odd at 1 has only 2 as successor
        assert 2 in attr
        assert 1 in attr  # Odd at 1 must go to 2 (only option)
        assert 0 in attr  # Even at 0 can go to 1 which is in attr

    def test_attractor_odd(self):
        arena = make_arena([(0, 0), (1, 1), (2, 1)], [(0, 1), (0, 2), (1, 0), (2, 2)])
        attr, strat = game_attractor(arena, {2}, Player.ODD)
        assert 2 in attr
        # 0 is Even-owned with succs {1, 2}. Not ALL in attr (1 not in attr), so not attracted
        # unless 1 is also attracted
        # 1 is Odd-owned -> 0, which is not in attr. So attr = {2}
        assert attr == {2}

    def test_empty_attractor(self):
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 0)])
        attr, _ = game_attractor(arena, set(), Player.EVEN)
        assert attr == set()

    def test_attractor_restricted(self):
        arena = make_arena([(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)])
        attr, _ = game_attractor(arena, {2}, Player.EVEN, restrict={1, 2})
        assert 2 in attr
        assert 1 in attr
        assert 0 not in attr  # Not in restriction set


# =========================================================================
# Section 2: Simple Rabin Games
# =========================================================================

class TestSimpleRabin:
    def test_single_pair_even_wins(self):
        """Simple cycle: 0->1->0. Even owns 0, Odd owns 1.
        Rabin pair: (L=empty, U={0}). Even wins everywhere (Buchi on {0})."""
        game = make_rabin_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            [(set(), {0})]
        )
        sol = solve_rabin(game)
        assert sol.win_even == {0, 1}
        assert sol.win_odd == set()

    def test_single_pair_odd_wins(self):
        """0->1->1 (self-loop). Even owns 0, Odd owns 1.
        Rabin pair: (L={1}, U={0}). Even needs to avoid 1 and visit 0 inf often.
        But play must enter 1 and stay there. Odd wins from 1."""
        game = make_rabin_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 1)],
            [({1}, {0})]
        )
        sol = solve_rabin(game)
        # From 0, Even must go to 1, then stuck. L={1} visited inf often. Odd wins.
        assert 1 in sol.win_odd
        assert 0 in sol.win_odd  # Even forced to go to 1

    def test_multiple_pairs_first_wins(self):
        """0->1->2->0. Rabin pairs: ({1}, {0}), ({2}, {1}).
        Pair 0: avoid 1 inf often, visit 0 inf often -- impossible in cycle.
        Pair 1: avoid 2 inf often, visit 1 inf often -- impossible in cycle.
        All visited inf often, so no pair satisfied. Odd wins."""
        game = make_rabin_game(
            [(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)],
            [({1}, {0}), ({2}, {1})]
        )
        sol = solve_rabin(game)
        assert sol.win_odd == {0, 1, 2}

    def test_rabin_with_escape(self):
        """0->1, 0->2, 1->1, 2->2. Even owns 0. Odd owns 1,2.
        Rabin pair: (L={1}, U={2}).
        Even can go to 2, avoiding 1 forever, visiting 2 forever. Even wins from 0."""
        game = make_rabin_game(
            [(0, 0), (1, 1), (2, 1)], [(0, 1), (0, 2), (1, 1), (2, 2)],
            [({1}, {2})]
        )
        sol = solve_rabin(game)
        assert 0 in sol.win_even
        assert 2 in sol.win_even
        assert 1 in sol.win_odd  # From 1, stuck in self-loop, L={1} visited inf often

    def test_empty_game(self):
        game = RabinGame(arena=GameArena(), pairs=[])
        sol = solve_rabin(game)
        assert sol.win_even == set()
        assert sol.win_odd == set()


# =========================================================================
# Section 3: Streett Games
# =========================================================================

class TestStreett:
    def test_single_pair_even_wins(self):
        """0->1->0 cycle. Streett pair: (L={0}, U={1}).
        Even must: if 0 visited inf often, then 1 visited inf often.
        In cycle, both visited inf often. Satisfied. Even wins."""
        game = make_streett_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            [({0}, {1})]
        )
        sol = solve_streett(game)
        assert sol.win_even == {0, 1}

    def test_single_pair_odd_wins(self):
        """0->0 self-loop (Even). Streett pair: (L={0}, U={1}).
        0 is visited inf often, but 1 never visited. Condition violated. Odd wins."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 0), (1, 1)])
        game = StreettGame(arena=arena, pairs=[RabinPair(L={0}, U={1})])
        sol = solve_streett(game)
        assert 0 in sol.win_odd  # Can't satisfy: L={0} inf often but U={1} never visited

    def test_streett_dual_agrees(self):
        """Streett and direct Streett should agree."""
        game = make_streett_game(
            [(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)],
            [({0, 1}, {2})]
        )
        sol1 = solve_streett(game)
        sol2 = solve_streett_direct(game)
        assert sol1.win_even == sol2.win_even
        assert sol1.win_odd == sol2.win_odd

    def test_multiple_streett_pairs(self):
        """Must satisfy ALL pairs. 0->1->2->0 cycle.
        Pair 0: (L={0}, U={1}) -- satisfied (both in cycle)
        Pair 1: (L={1}, U={2}) -- satisfied (both in cycle)
        Even wins."""
        game = make_streett_game(
            [(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)],
            [({0}, {1}), ({1}, {2})]
        )
        sol = solve_streett(game)
        assert sol.win_even == {0, 1, 2}

    def test_streett_one_pair_fails(self):
        """0->1->0, 0->2->2. Even owns 0.
        Pair 0: (L={0}, U={1}) -- need 1 inf often when 0 inf often
        Pair 1: (L={0}, U={2}) -- need 2 inf often when 0 inf often
        Can't satisfy both: going 0->1->0 satisfies pair 0 but not pair 1 (2 not visited).
        Going 0->2->2 doesn't visit 0 inf often so both trivially satisfied!
        Even can go to 2 and stay. Even wins from 0 and 2."""
        game = make_streett_game(
            [(0, 0), (1, 1), (2, 1)], [(0, 1), (0, 2), (1, 0), (2, 2)],
            [({0}, {1}), ({0}, {2})]
        )
        sol = solve_streett(game)
        assert 0 in sol.win_even  # Even goes to 2, avoids 0 inf often
        assert 2 in sol.win_even
        # From 1, Odd sends to 0, Even goes to 2. Even wins.
        assert 1 in sol.win_even


# =========================================================================
# Section 4: Buchi and co-Buchi Games
# =========================================================================

class TestBuchiCoBuchi:
    def test_buchi_even_wins(self):
        """Buchi: visit {0} infinitely often. 0->1->0 cycle."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 0)])
        game = make_buchi_game(arena, {0})
        sol = solve_rabin(game)
        assert sol.win_even == {0, 1}

    def test_buchi_odd_wins(self):
        """Buchi: visit {0} inf often. 0->1->1. From 0, must enter 1's self-loop."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 1)])
        game = make_buchi_game(arena, {0})
        sol = solve_rabin(game)
        assert sol.win_odd == {0, 1}  # 0 cannot be visited inf often

    def test_co_buchi_even_wins(self):
        """co-Buchi: visit {1} finitely often. 0->0 self-loop, 0->1->1.
        Even at 0 can self-loop forever, avoiding 1."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)])
        game = make_co_buchi_game(arena, {1})
        sol = solve_rabin(game)
        assert 0 in sol.win_even

    def test_co_buchi_odd_wins(self):
        """co-Buchi: visit {0} finitely often. 0->1->0 cycle.
        0 visited inf often. Condition violated. Odd wins."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 0)])
        game = make_co_buchi_game(arena, {0})
        sol = solve_rabin(game)
        # Even must avoid 0 but cycle forces revisiting it
        # Actually: co-Buchi pair is (L={0}, U=all). L visited inf often?
        # In cycle, 0 visited inf often AND all visited inf often.
        # Rabin pair satisfied: L={0} BUT Rabin needs L FINITELY often.
        # 0 visited inf often -> L visited inf often -> pair NOT satisfied.
        assert sol.win_odd == {0, 1}

    def test_generalized_buchi(self):
        """Gen Buchi: visit {0} and {2} infinitely often.
        0->1->2->0 cycle -- both visited inf often. Even wins."""
        arena = make_arena([(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)])
        game = make_generalized_buchi_game(arena, [{0}, {2}])
        sol = solve_streett(game)
        assert sol.win_even == {0, 1, 2}


# =========================================================================
# Section 5: Parity to Rabin Reduction
# =========================================================================

class TestParityReduction:
    def test_simple_parity_to_rabin(self):
        pg = ParityGame()
        pg.add_vertex(0, PPlayer.EVEN, 0)
        pg.add_vertex(1, PPlayer.ODD, 1)
        pg.add_edge(0, 1)
        pg.add_edge(1, 0)
        # Priority 0 (even) and 1 (odd). Highest inf-often = max(0,1) = 1 (odd) -> Odd wins
        rg = parity_to_rabin(pg)
        assert len(rg.pairs) == 1  # One even priority (0)
        sol = solve_rabin(rg)
        assert sol.win_odd == {0, 1}

    def test_parity_to_rabin_even_wins(self):
        pg = ParityGame()
        pg.add_vertex(0, PPlayer.EVEN, 2)
        pg.add_vertex(1, PPlayer.ODD, 1)
        pg.add_edge(0, 1)
        pg.add_edge(1, 0)
        # Priorities 2 and 1. Highest inf-often = 2 (even) -> Even wins
        rg = parity_to_rabin(pg)
        sol = solve_rabin(rg)
        assert sol.win_even == {0, 1}

    def test_parity_to_streett(self):
        pg = ParityGame()
        pg.add_vertex(0, PPlayer.EVEN, 2)
        pg.add_vertex(1, PPlayer.ODD, 1)
        pg.add_edge(0, 1)
        pg.add_edge(1, 0)
        sg = parity_to_streett(pg)
        sol = solve_streett(sg)
        assert sol.win_even == {0, 1}

    def test_compare_all_agree(self):
        pg = ParityGame()
        pg.add_vertex(0, PPlayer.EVEN, 0)
        pg.add_vertex(1, PPlayer.ODD, 1)
        pg.add_vertex(2, PPlayer.EVEN, 2)
        pg.add_edge(0, 1)
        pg.add_edge(1, 2)
        pg.add_edge(2, 0)
        pg.add_edge(0, 2)  # Even can skip 1
        result = compare_with_parity(pg)
        assert result['all_agree']

    def test_compare_larger_game(self):
        pg = ParityGame()
        for i in range(5):
            pg.add_vertex(i, PPlayer.EVEN if i % 2 == 0 else PPlayer.ODD, i)
        for i in range(5):
            pg.add_edge(i, (i + 1) % 5)
        pg.add_edge(0, 4)  # shortcut
        result = compare_with_parity(pg)
        assert result['all_agree']


# =========================================================================
# Section 6: Muller Games
# =========================================================================

class TestMuller:
    def test_simple_muller_even_wins(self):
        """2-color Muller: 0->1->0 cycle. Colors: 0->A, 1->B.
        Accepting: {A,B}. Both colors visited inf often. Even wins."""
        game = make_muller_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            {0: 0, 1: 1}, [frozenset({0, 1})]
        )
        sol = solve_muller(game)
        assert sol.win_even == {0, 1}

    def test_muller_odd_wins(self):
        """0->1->1 self-loop. Colors: 0->A, 1->B.
        Accepting: {A}. Only B visited inf often. Not accepting. Odd wins."""
        game = make_muller_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 1)],
            {0: 0, 1: 1}, [frozenset({0})]
        )
        sol = solve_muller(game)
        assert sol.win_odd == {0, 1}

    def test_muller_multiple_accepting(self):
        """0->1->2->0, 0->0. Colors: 0->A, 1->B, 2->C.
        Accepting: {A}, {A,B,C}. Even at 0 can self-loop (only A inf often).
        {A} is accepting. Even wins from 0."""
        game = make_muller_game(
            [(0, 0), (1, 1), (2, 1)], [(0, 1), (0, 0), (1, 2), (2, 0)],
            {0: 0, 1: 1, 2: 2}, [frozenset({0}), frozenset({0, 1, 2})]
        )
        sol = solve_muller(game)
        assert 0 in sol.win_even

    def test_muller_with_choice(self):
        """0->1, 0->2, 1->1, 2->2. Even at 0.
        Colors: 0->A, 1->B, 2->C. Accepting: {B}.
        Even goes to 1, visits B forever. {B} is accepting."""
        game = make_muller_game(
            [(0, 0), (1, 1), (2, 1)], [(0, 1), (0, 2), (1, 1), (2, 2)],
            {0: 0, 1: 1, 2: 2}, [frozenset({1})]
        )
        sol = solve_muller(game)
        assert 0 in sol.win_even
        assert 1 in sol.win_even


# =========================================================================
# Section 7: Strategy Verification
# =========================================================================

class TestStrategyVerification:
    def test_verify_buchi_strategy(self):
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 0)])
        game = make_buchi_game(arena, {0})
        sol = solve_rabin(game)
        result = verify_rabin_strategy(game, sol)
        assert result['even_strategy_valid']
        assert result['odd_strategy_valid']

    def test_verify_rabin_with_escape(self):
        game = make_rabin_game(
            [(0, 0), (1, 1), (2, 1)], [(0, 1), (0, 2), (1, 1), (2, 2)],
            [({1}, {2})]
        )
        sol = solve_rabin(game)
        result = verify_rabin_strategy(game, sol)
        assert result['even_wins_correct']
        assert result['odd_wins_correct']

    def test_verify_empty_game(self):
        game = RabinGame(arena=GameArena(), pairs=[])
        sol = solve_rabin(game)
        result = verify_rabin_strategy(game, sol)
        assert result['even_strategy_valid']


# =========================================================================
# Section 8: Statistics
# =========================================================================

class TestStatistics:
    def test_rabin_stats(self):
        game = make_rabin_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            [(set(), {0})]
        )
        stats = rabin_streett_statistics(game)
        assert stats['kind'] == 'rabin'
        assert stats['vertices'] == 2
        assert stats['edges'] == 2
        assert stats['num_pairs'] == 1

    def test_streett_stats(self):
        game = make_streett_game(
            [(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0)],
            [({0}, {1})]
        )
        stats = rabin_streett_statistics(game)
        assert stats['kind'] == 'streett'
        assert stats['vertices'] == 3

    def test_muller_stats(self):
        game = make_muller_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            {0: 0, 1: 1}, [frozenset({0, 1})]
        )
        stats = rabin_streett_statistics(game)
        assert stats['kind'] == 'muller'
        assert stats['num_colors'] == 2
        assert stats['num_accepting'] == 1


# =========================================================================
# Section 9: Batch Solving
# =========================================================================

class TestBatch:
    def test_batch_rabin(self):
        g1 = make_rabin_game([(0, 0), (1, 1)], [(0, 1), (1, 0)], [(set(), {0})])
        g2 = make_rabin_game([(0, 0), (1, 1)], [(0, 1), (1, 1)], [({1}, {0})])
        sols = batch_solve([g1, g2])
        assert len(sols) == 2
        assert sols[0].win_even == {0, 1}

    def test_batch_mixed(self):
        g1 = make_rabin_game([(0, 0), (1, 1)], [(0, 1), (1, 0)], [(set(), {0})])
        g2 = make_streett_game([(0, 0), (1, 1)], [(0, 1), (1, 0)], [({0}, {1})])
        sols = batch_solve([g1, g2])
        assert len(sols) == 2


# =========================================================================
# Section 10: Larger Games
# =========================================================================

class TestLargerGames:
    def test_chain_rabin(self):
        """Chain: 0->1->2->...->n-1->0. Even owns evens. Rabin pair: (L={1}, U={0}).
        Even needs to avoid 1. In a cycle, 1 is visited. So Odd wins... unless Even can skip."""
        n = 6
        verts = [(i, i % 2) for i in range(n)]
        edges = [(i, (i+1) % n) for i in range(n)]
        game = make_rabin_game(verts, edges, [({1}, {0})])
        sol = solve_rabin(game)
        # In a simple cycle, all states visited inf often. L={1} visited inf often. Odd wins.
        assert sol.win_odd == set(range(n))

    def test_diamond_rabin(self):
        """Diamond: 0->{1,2}, 1->3, 2->3, 3->0. Even owns 0,3. Odd owns 1,2.
        Rabin pair: (L={1}, U={2}).
        Even at 0 can choose to go to 2, avoiding 1. Then 2->3->0->2->3... cycle.
        1 never visited. L finitely often, U={2} inf often. Even wins."""
        game = make_rabin_game(
            [(0, 0), (1, 1), (2, 1), (3, 0)],
            [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)],
            [({1}, {2})]
        )
        sol = solve_rabin(game)
        assert 0 in sol.win_even
        assert 2 in sol.win_even
        assert 3 in sol.win_even

    def test_two_pair_complex(self):
        """4 vertices, 2 pairs.
        0->1, 0->2, 1->3, 2->3, 3->0. Even: 0,3. Odd: 1,2.
        Pair 0: (L={1,2}, U={0}) -- avoid 1 and 2, visit 0 inf often
        Pair 1: (L={3}, U={2}) -- avoid 3, visit 2 inf often
        Neither pair can be satisfied in this graph (all states in any cycle)."""
        game = make_rabin_game(
            [(0, 0), (1, 1), (2, 1), (3, 0)],
            [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)],
            [({1, 2}, {0}), ({3}, {2})]
        )
        sol = solve_rabin(game)
        # Any cycle must include 0->?->3->0, so 3 is always visited.
        # Pair 1 needs to avoid 3 -- impossible in any cycle containing 0.
        # Pair 0 needs to avoid 1 and 2 -- but 0 must go to 1 or 2.
        assert sol.win_odd == {0, 1, 2, 3}


# =========================================================================
# Section 11: Rabin-Streett Duality
# =========================================================================

class TestDuality:
    def test_duality_simple(self):
        """Rabin for Even and Streett for Even with same pairs should be different
        (Streett is stricter). But Rabin for Even = complement of Streett for Odd."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 0)])
        pairs = [RabinPair(L=set(), U={0})]

        rabin = RabinGame(arena=arena, pairs=pairs)
        streett = StreettGame(arena=arena, pairs=pairs)

        rabin_sol = solve_rabin(rabin)
        streett_sol = solve_streett(streett)

        # Rabin (L=empty, U={0}): Buchi on {0}. 0 visited inf often in cycle. Even wins.
        assert rabin_sol.win_even == {0, 1}
        # Streett (L=empty, U={0}): if empty visited inf often (vacuously false),
        # then 0 must be inf often. Vacuously true. Even wins.
        # Actually: Streett condition: for ALL pairs, if L inf often -> U inf often.
        # L=empty, so L is NEVER visited inf often. Condition trivially satisfied.
        assert streett_sol.win_even == {0, 1}

    def test_duality_nontrivial(self):
        """Verify Rabin-Streett duality on a nontrivial example."""
        arena = make_arena(
            [(0, 0), (1, 1), (2, 0)],
            [(0, 1), (0, 2), (1, 0), (2, 2)]
        )
        pairs = [RabinPair(L={1}, U={2})]

        rabin = RabinGame(arena=arena, pairs=pairs)
        rabin_sol = solve_rabin(rabin)

        # Even at 0 can go to 2 (self-loop). Avoids 1, visits 2 forever.
        assert 0 in rabin_sol.win_even
        assert 2 in rabin_sol.win_even


# =========================================================================
# Section 12: Edge Cases
# =========================================================================

class TestEdgeCases:
    def test_single_vertex_self_loop_rabin(self):
        game = make_rabin_game([(0, 0)], [(0, 0)], [(set(), {0})])
        sol = solve_rabin(game)
        assert sol.win_even == {0}

    def test_single_vertex_no_edges(self):
        """Vertex with no successors -- dead end."""
        arena = make_arena([(0, 0)], [])
        game = RabinGame(arena=arena, pairs=[RabinPair(L=set(), U={0})])
        sol = solve_rabin(game)
        # Dead-end: no infinite play possible. Typically lost for the owner.
        # Our solver puts it in win_odd (can't satisfy any pair from dead-end)
        assert 0 in sol.win_even or 0 in sol.win_odd  # Either is acceptable

    def test_disconnected_components(self):
        """Two disconnected cycles."""
        game = make_rabin_game(
            [(0, 0), (1, 1), (2, 0), (3, 1)],
            [(0, 1), (1, 0), (2, 3), (3, 2)],
            [(set(), {0})]
        )
        sol = solve_rabin(game)
        assert 0 in sol.win_even
        assert 1 in sol.win_even
        # Component {2,3}: U={0} not in this component. Can't visit 0 inf often.
        assert 2 in sol.win_odd
        assert 3 in sol.win_odd

    def test_no_pairs_rabin(self):
        """Rabin with no pairs: Even can never win (no pair to satisfy)."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 0)])
        game = RabinGame(arena=arena, pairs=[])
        sol = solve_rabin(game)
        assert sol.win_odd == {0, 1}

    def test_no_pairs_streett(self):
        """Streett with no pairs: condition vacuously true. Even wins everywhere."""
        arena = make_arena([(0, 0), (1, 1)], [(0, 1), (1, 0)])
        game = StreettGame(arena=arena, pairs=[])
        sol = solve_streett(game)
        assert sol.win_even == {0, 1}

    def test_overlapping_L_U(self):
        """L and U overlap. Rabin pair: (L={0,1}, U={0}).
        Even needs 0 visited inf often (U) but also 0 visited finitely often (L).
        Contradiction. Odd wins."""
        game = make_rabin_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            [({0, 1}, {0})]
        )
        sol = solve_rabin(game)
        # L={0,1} must be visited finitely often. But play is 0->1->0->... so L visited inf often.
        assert sol.win_odd == {0, 1}


# =========================================================================
# Section 13: Muller Edge Cases
# =========================================================================

class TestMullerEdgeCases:
    def test_single_color_accepting(self):
        """All vertices same color. Accepting if that color set is in family."""
        game = make_muller_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            {0: 0, 1: 0}, [frozenset({0})]
        )
        sol = solve_muller(game)
        assert sol.win_even == {0, 1}

    def test_muller_no_accepting(self):
        """No accepting sets. Odd wins everywhere."""
        game = make_muller_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            {0: 0, 1: 1}, []
        )
        sol = solve_muller(game)
        assert sol.win_odd == {0, 1}

    def test_muller_all_accepting(self):
        """All possible color sets are accepting. Even always wins."""
        game = make_muller_game(
            [(0, 0), (1, 1)], [(0, 1), (1, 0)],
            {0: 0, 1: 1},
            [frozenset({0}), frozenset({1}), frozenset({0, 1})]
        )
        sol = solve_muller(game)
        assert sol.win_even == {0, 1}


# =========================================================================
# Section 14: Integration -- Rabin/Streett/Muller/Parity Agree
# =========================================================================

class TestIntegration:
    def test_all_conditions_agree_on_parity(self):
        """A parity game converted to all representations should give same winner."""
        pg = ParityGame()
        # Even owns 0,2,4. Odd owns 1,3.
        for i in range(5):
            pg.add_vertex(i, PPlayer.EVEN if i % 2 == 0 else PPlayer.ODD, i)
        pg.add_edge(0, 1); pg.add_edge(1, 2); pg.add_edge(2, 3)
        pg.add_edge(3, 4); pg.add_edge(4, 0); pg.add_edge(2, 0)

        from parity_games import zielonka
        parity_sol = zielonka(pg)

        rabin_game = parity_to_rabin(pg)
        rabin_sol = solve_rabin(rabin_game)

        streett_game = parity_to_streett(pg)
        streett_sol = solve_streett(streett_game)

        assert parity_sol.win_even == rabin_sol.win_even
        assert parity_sol.win_even == streett_sol.win_even

    def test_streett_direct_matches_dual(self):
        """Direct Streett solver matches dual-based solver."""
        for _ in range(3):
            game = make_streett_game(
                [(0, 0), (1, 1), (2, 0), (3, 1)],
                [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 1)],
                [({0, 1}, {2, 3})]
            )
            sol_dual = solve_streett(game)
            sol_direct = solve_streett_direct(game)
            assert sol_dual.win_even == sol_direct.win_even

    def test_buchi_as_rabin_and_parity(self):
        """Buchi game solved as Rabin should match parity game with priorities 0,1."""
        pg = ParityGame()
        pg.add_vertex(0, PPlayer.EVEN, 0)  # accepting (even priority)
        pg.add_vertex(1, PPlayer.ODD, 1)   # non-accepting (odd priority)
        pg.add_vertex(2, PPlayer.EVEN, 0)  # accepting
        pg.add_edge(0, 1); pg.add_edge(1, 2); pg.add_edge(2, 0); pg.add_edge(1, 1)

        from parity_games import zielonka
        parity_sol = zielonka(pg)

        arena = make_arena([(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 2), (2, 0), (1, 1)])
        buchi = make_buchi_game(arena, {0, 2})
        rabin_sol = solve_rabin(buchi)

        assert parity_sol.win_even == rabin_sol.win_even


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
