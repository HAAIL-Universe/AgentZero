"""Tests for V080: Omega-Regular Game Solving via Parity Reduction."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega_regular_parity import (
    GameArena, AcceptanceCondition, AccType, OmegaParityResult,
    reduce_to_parity, solve_omega_regular, solve_ltl_game,
    conjoin_acceptance, disjoin_acceptance,
    compare_reductions, analyze_reduction,
    make_arena, solve_from_spec, solve_ltl_from_spec,
    muller_to_rabin, solve_muller_via_rabin,
    ltl_to_parity_game,
)

# Also import LTL constructors for formula building
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
sys.path.insert(0, os.path.join(_work, 'V023_ltl_model_checking'))
from ltl_model_checker import (
    Atom, Globally, Finally, Until, And, Or, Not, Next, parse_ltl,
)

sys.path.insert(0, os.path.join(_work, 'V076_parity_games'))
from parity_games import ParityGame, Player as PGPlayer


# ============================================================
# Helper: simple game arenas for testing
# ============================================================

def simple_cycle_arena():
    """A simple cycle: 0 -> 1 -> 2 -> 0, all owned by Even."""
    return make_arena(
        nodes=[(0, 0), (1, 0), (2, 0)],
        edges=[(0, 1), (1, 2), (2, 0)],
    )


def two_player_choice():
    """Even at 0 chooses between 1 (good) and 2 (bad).
    1 -> 1 (self-loop), 2 -> 2 (self-loop).
    Odd at 3 chooses between 1 and 2.
    """
    return make_arena(
        nodes=[(0, 0), (1, 0), (2, 1), (3, 1)],
        edges=[(0, 1), (0, 2), (1, 1), (2, 2), (3, 1), (3, 2)],
    )


def diamond_arena():
    """Diamond: 0 (Even) -> 1, 2. 1 (Odd) -> 3. 2 (Odd) -> 3. 3 -> 0."""
    return make_arena(
        nodes=[(0, 0), (1, 1), (2, 1), (3, 0)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)],
    )


# ============================================================
# Section 1: GameArena basics
# ============================================================

class TestGameArena:
    def test_create_arena(self):
        arena = simple_cycle_arena()
        assert len(arena.nodes) == 3
        assert arena.owner[0] == 0
        assert 1 in arena.successors[0]

    def test_arena_validation(self):
        arena = GameArena()
        arena.add_node(0, 0)
        # Node 0 has no successors
        issues = arena.validate()
        assert any("no successors" in i for i in issues)

    def test_make_arena(self):
        arena = make_arena(
            nodes=[(0, 0), (1, 1)],
            edges=[(0, 1), (1, 0)],
        )
        assert 0 in arena.nodes
        assert 1 in arena.nodes
        assert 1 in arena.successors[0]
        assert 0 in arena.successors[1]

    def test_arena_with_labels(self):
        arena = make_arena(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            labels={0: {"a"}, 1: {"b"}},
        )
        assert arena.labels[0] == {"a"}
        assert arena.labels[1] == {"b"}


# ============================================================
# Section 2: AcceptanceCondition construction
# ============================================================

class TestAcceptanceCondition:
    def test_buchi(self):
        acc = AcceptanceCondition.buchi({0, 2})
        assert acc.acc_type == AccType.BUCHI
        assert acc.accepting == {0, 2}

    def test_cobuchi(self):
        acc = AcceptanceCondition.cobuchi({1})
        assert acc.acc_type == AccType.COBUCHI
        assert acc.rejecting == {1}

    def test_rabin(self):
        acc = AcceptanceCondition.rabin([({1}, {0})])
        assert acc.acc_type == AccType.RABIN
        assert len(acc.rabin_pairs) == 1

    def test_streett(self):
        acc = AcceptanceCondition.streett([({0}, {1})])
        assert acc.acc_type == AccType.STREETT

    def test_muller(self):
        acc = AcceptanceCondition.muller({frozenset({0, 1}), frozenset({2})})
        assert acc.acc_type == AccType.MULLER
        assert len(acc.muller_table) == 2

    def test_parity(self):
        acc = AcceptanceCondition.parity({0: 2, 1: 1, 2: 0})
        assert acc.acc_type == AccType.PARITY
        assert acc.priorities[0] == 2


# ============================================================
# Section 3: Buchi reduction and solving
# ============================================================

class TestBuchi:
    def test_buchi_all_accepting(self):
        """All nodes accepting -> Even always wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({0, 1, 2})
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_buchi_none_accepting(self):
        """No accepting nodes -> Odd always wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi(set())
        result = solve_omega_regular(arena, acc)
        assert result.winner_odd == {0, 1, 2}

    def test_buchi_choice(self):
        """Even can choose to visit accepting node infinitely."""
        arena = two_player_choice()
        acc = AcceptanceCondition.buchi({1})  # Node 1 is accepting
        result = solve_omega_regular(arena, acc)
        assert 0 in result.winner_even  # Even at 0 can go to 1
        assert 1 in result.winner_even  # Node 1 loops on itself

    def test_buchi_odd_controls(self):
        """Odd controls and can avoid accepting."""
        arena = two_player_choice()
        acc = AcceptanceCondition.buchi({1})
        result = solve_omega_regular(arena, acc)
        assert 3 in result.winner_odd  # Odd at 3 goes to 2 (non-accepting)

    def test_buchi_cycle_single_accepting(self):
        """Single accepting node in forced cycle -> Even wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({1})
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_buchi_strategy(self):
        arena = two_player_choice()
        acc = AcceptanceCondition.buchi({1})
        result = solve_omega_regular(arena, acc)
        # Even's strategy at node 0 should go to 1
        if 0 in result.strategy_even:
            assert result.strategy_even[0] == 1


# ============================================================
# Section 4: co-Buchi reduction and solving
# ============================================================

class TestCoBuchi:
    def test_cobuchi_no_rejecting(self):
        """No rejecting -> Even always wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.cobuchi(set())
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_cobuchi_all_rejecting(self):
        """All rejecting in a cycle -> Odd wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.cobuchi({0, 1, 2})
        result = solve_omega_regular(arena, acc)
        assert result.winner_odd == {0, 1, 2}

    def test_cobuchi_choice_avoid_reject(self):
        """Even can choose to avoid rejecting node."""
        arena = two_player_choice()
        acc = AcceptanceCondition.cobuchi({2})  # Node 2 is rejecting
        result = solve_omega_regular(arena, acc)
        assert 0 in result.winner_even  # Even goes to 1, avoids 2
        assert 1 in result.winner_even

    def test_cobuchi_forced_through_reject(self):
        """All nodes rejecting in forced cycle -> Odd wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.cobuchi({0})
        result = solve_omega_regular(arena, acc)
        # Node 0 is rejecting and must be visited infinitely in the only cycle
        assert result.winner_odd == {0, 1, 2}


# ============================================================
# Section 5: Parity (direct) reduction
# ============================================================

class TestParityDirect:
    def test_parity_even_wins(self):
        """Max priority is even -> Even wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.parity({0: 2, 1: 1, 2: 0})
        result = solve_omega_regular(arena, acc)
        # Max inf priority is 2 (even) -> Even wins
        assert result.winner_even == {0, 1, 2}

    def test_parity_odd_wins(self):
        """Max priority is odd -> Odd wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.parity({0: 1, 1: 0, 2: 0})
        result = solve_omega_regular(arena, acc)
        # Max inf priority is 1 (odd) -> Odd wins
        assert result.winner_odd == {0, 1, 2}

    def test_parity_choice(self):
        """Even can choose path with even max priority."""
        arena = two_player_choice()
        acc = AcceptanceCondition.parity({0: 0, 1: 2, 2: 1, 3: 0})
        result = solve_omega_regular(arena, acc)
        assert 0 in result.winner_even  # Even goes to 1 (prio 2)
        assert 1 in result.winner_even

    def test_parity_spm_algorithm(self):
        """Test with SPM algorithm."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.parity({0: 2, 1: 1, 2: 0})
        result = solve_omega_regular(arena, acc, algorithm="spm")
        assert result.winner_even == {0, 1, 2}
        assert result.algorithm == "spm"


# ============================================================
# Section 6: Rabin reduction
# ============================================================

class TestRabin:
    def test_rabin_single_pair(self):
        """Single Rabin pair: visit 0 finitely, visit 1 infinitely."""
        arena = two_player_choice()
        # Pair: fin={2}, inf={1} -- avoid 2 forever, visit 1 forever
        acc = AcceptanceCondition.rabin([({2}, {1})])
        result = solve_omega_regular(arena, acc)
        assert 0 in result.winner_even  # Even goes to 1
        assert 1 in result.winner_even

    def test_rabin_impossible(self):
        """Rabin pair impossible to satisfy in forced cycle."""
        arena = simple_cycle_arena()
        # Must visit 0 finitely AND visit 1 infinitely -- impossible, 0 is in the cycle
        acc = AcceptanceCondition.rabin([({0}, {1})])
        result = solve_omega_regular(arena, acc)
        # All nodes in forced cycle visiting 0 infinitely -> Odd wins
        assert result.winner_odd == {0, 1, 2}

    def test_rabin_satisfiable(self):
        """Rabin pair satisfiable: fin set not on the cycle."""
        arena = simple_cycle_arena()
        # fin={}, inf={1} -- trivially visit nothing finitely, visit 1 infinitely
        acc = AcceptanceCondition.rabin([(set(), {1})])
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_rabin_two_pairs(self):
        """Two Rabin pairs: disjunction -- satisfy either."""
        arena = two_player_choice()
        acc = AcceptanceCondition.rabin([
            ({1}, {2}),  # Pair 0: avoid 1, visit 2
            ({2}, {1}),  # Pair 1: avoid 2, visit 1
        ])
        result = solve_omega_regular(arena, acc)
        # Even at 0 can satisfy pair 1 by going to 1
        assert 0 in result.winner_even


# ============================================================
# Section 7: Streett reduction
# ============================================================

class TestStreett:
    def test_streett_trivial(self):
        """No Streett pairs -> Even wins (vacuously satisfied)."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.streett([])
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_streett_single_pair_satisfied(self):
        """Streett pair: if request visited inf, then response visited inf."""
        arena = simple_cycle_arena()
        # request={0}, response={1} -- both in cycle, both visited inf -> satisfied
        acc = AcceptanceCondition.streett([({0}, {1})])
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_streett_pair_violated(self):
        """Streett pair violated: request visited inf, response not."""
        arena = two_player_choice()
        # If node 1 visited infinitely then node 2 must be visited infinitely
        # But 1 loops on self, never visiting 2 -> violated if stuck at 1
        # However Odd can force going to 2 from node 3
        # Even at 0: if goes to 1, stays at 1 forever, request={1} inf but response={2} not inf
        acc = AcceptanceCondition.streett([({1}, {2})])
        result = solve_omega_regular(arena, acc)
        # Node 1 self-loops: request {1} visited inf, response {2} not visited -> Odd wins at 1
        assert 1 in result.winner_odd


# ============================================================
# Section 8: Muller reduction (LAR construction)
# ============================================================

class TestMuller:
    def test_muller_simple(self):
        """Muller: accepting set = {0,1,2} (the full cycle)."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.muller({frozenset({0, 1, 2})})
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_muller_empty_table(self):
        """Empty Muller table -> Odd always wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.muller(set())
        result = solve_omega_regular(arena, acc)
        assert result.winner_odd == {0, 1, 2}

    def test_muller_singleton_accepting(self):
        """Muller: only {1} is accepting -> Even wins iff can stay at 1."""
        arena = two_player_choice()
        acc = AcceptanceCondition.muller({frozenset({1})})
        result = solve_omega_regular(arena, acc)
        # Node 1 self-loops, inf-visited set = {1} which is accepting
        assert 1 in result.winner_even
        # Even at 0 can go to 1
        assert 0 in result.winner_even

    def test_muller_via_rabin(self):
        """Muller via Rabin conversion gives same result."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.muller({frozenset({0, 1, 2})})
        r1 = solve_omega_regular(arena, acc)
        r2 = solve_muller_via_rabin(arena, acc)
        assert r1.winner_even == r2.winner_even

    def test_muller_to_rabin_conversion(self):
        """Test the Muller-to-Rabin conversion function."""
        table = {frozenset({0, 1}), frozenset({2})}
        pairs = muller_to_rabin(table, {0, 1, 2})
        assert len(pairs) == 2
        for fin, inf in pairs:
            assert isinstance(fin, set)
            assert isinstance(inf, set)

    def test_muller_empty_set_accepting(self):
        """Muller with empty set in table -- but cycle visits all nodes."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.muller({frozenset()})
        result = solve_omega_regular(arena, acc)
        # Empty set is accepting, but the cycle visits {0,1,2} infinitely.
        # {0,1,2} is NOT in the table, so Odd wins.
        assert result.winner_odd == {0, 1, 2}


# ============================================================
# Section 9: reduce_to_parity correctness
# ============================================================

class TestReduceToParity:
    def test_buchi_reduction_structure(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({1})
        pg = reduce_to_parity(arena, acc)
        assert isinstance(pg, ParityGame)
        assert len(pg.nodes) == 3

    def test_cobuchi_reduction_structure(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.cobuchi({1})
        pg = reduce_to_parity(arena, acc)
        assert isinstance(pg, ParityGame)
        # Node 1 should have priority 1 (rejecting)
        assert pg.priority[1] == 1

    def test_parity_reduction_preserves_priorities(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.parity({0: 3, 1: 2, 2: 1})
        pg = reduce_to_parity(arena, acc)
        assert pg.priority[0] == 3
        assert pg.priority[1] == 2
        assert pg.priority[2] == 1

    def test_rabin_reduction_creates_valid_game(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.rabin([(set(), {1})])
        pg = reduce_to_parity(arena, acc)
        issues = pg.validate()
        assert len(issues) == 0


# ============================================================
# Section 10: LTL-to-parity game solving
# ============================================================

class TestLTLGame:
    def test_ltl_globally_true(self):
        """G(a): always a. All nodes labeled 'a' -> Even wins."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            labels={0: {"a"}, 1: {"a"}},
        )
        formula = Globally(Atom("a"))
        result = solve_ltl_game(arena, formula, initial_state=0)
        assert 0 in result.winner_even

    def test_ltl_finally_reachable(self):
        """F(b): eventually b. Node 1 has 'b', reachable from 0."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 1)],
            labels={0: set(), 1: {"b"}},
        )
        formula = Finally(Atom("b"))
        result = solve_ltl_game(arena, formula, initial_state=0)
        assert 0 in result.winner_even

    def test_ltl_globally_false(self):
        """G(a): but node 1 doesn't have 'a' and it's forced."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            labels={0: {"a"}, 1: set()},
        )
        formula = Globally(Atom("a"))
        result = solve_ltl_game(arena, formula, initial_state=0)
        # Must visit node 1 (no 'a') -> G(a) violated -> Odd wins
        assert 0 in result.winner_odd

    def test_ltl_until(self):
        """a U b: a holds until b."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (1, 2), (2, 2)],
            labels={0: {"a"}, 1: {"a"}, 2: {"b"}},
        )
        formula = Until(Atom("a"), Atom("b"))
        result = solve_ltl_game(arena, formula, initial_state=0)
        assert 0 in result.winner_even

    def test_ltl_from_string(self):
        """Test solve_ltl_from_spec with string formula."""
        result = solve_ltl_from_spec(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 1)],
            labels={0: set(), 1: {"b"}},
            formula_str="F(b)",
            initial_state=0,
        )
        assert 0 in result.winner_even

    def test_ltl_parity_game_structure(self):
        """LTL reduction produces a valid parity game."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            labels={0: {"a"}, 1: {"b"}},
        )
        formula = Globally(Atom("a"))
        pg, state_map, init = ltl_to_parity_game(arena, formula, 0)
        assert isinstance(pg, ParityGame)
        assert len(pg.nodes) > 0
        issues = pg.validate()
        assert len(issues) == 0


# ============================================================
# Section 11: Acceptance composition (conjunction/disjunction)
# ============================================================

class TestComposition:
    def test_conjoin_buchi(self):
        """Conjunction of two Buchi conditions."""
        arena = simple_cycle_arena()
        c1 = AcceptanceCondition.buchi({0, 1})
        c2 = AcceptanceCondition.buchi({1, 2})
        result = conjoin_acceptance(arena, [c1, c2])
        # Must visit {0,1} and {1,2} infinitely often
        # In the forced cycle 0->1->2->0, all are visited -> Even wins
        assert result.winner_even == {0, 1, 2}

    def test_conjoin_buchi_impossible(self):
        """Conjunction that can't be satisfied."""
        # Arena: 0 (Even) -> 1 or 2. 1 self-loops. 2 self-loops.
        arena = make_arena(
            nodes=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        c1 = AcceptanceCondition.buchi({1})  # Must visit 1 inf
        c2 = AcceptanceCondition.buchi({2})  # Must visit 2 inf
        result = conjoin_acceptance(arena, [c1, c2])
        # Can't visit both 1 and 2 infinitely -- once you choose, you're stuck
        assert 0 in result.winner_odd

    def test_disjoin_buchi(self):
        """Disjunction: satisfy either condition."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        c1 = AcceptanceCondition.buchi({1})
        c2 = AcceptanceCondition.buchi({2})
        result = disjoin_acceptance(arena, [c1, c2])
        # Even can choose to go to 1 or 2 -- either satisfies one condition
        assert 0 in result.winner_even

    def test_single_condition(self):
        """Single condition returns same as direct solve."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({1})
        r1 = conjoin_acceptance(arena, [acc])
        r2 = solve_omega_regular(arena, acc)
        assert r1.winner_even == r2.winner_even

    def test_conjoin_general(self):
        """Conjunction of mixed types via independent intersection."""
        arena = simple_cycle_arena()
        c1 = AcceptanceCondition.buchi({0, 1, 2})
        c2 = AcceptanceCondition.cobuchi(set())
        result = conjoin_acceptance(arena, [c1, c2])
        # Both trivially satisfied -> Even wins
        assert result.winner_even == {0, 1, 2}


# ============================================================
# Section 12: Comparison and analysis
# ============================================================

class TestAnalysis:
    def test_compare_reductions(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({1})
        comp = compare_reductions(arena, acc)
        assert comp["agree"]
        assert "zielonka" in comp
        assert "spm" in comp

    def test_analyze_reduction(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.rabin([(set(), {1})])
        analysis = analyze_reduction(arena, acc)
        assert analysis["arena_nodes"] == 3
        assert analysis["parity_game_nodes"] >= 3
        assert "priority_distribution" in analysis

    def test_result_summary(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({1})
        result = solve_omega_regular(arena, acc)
        summary = result.summary()
        assert "buchi" in summary
        assert "Even wins" in summary

    def test_result_winner(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({1})
        result = solve_omega_regular(arena, acc)
        assert result.winner(0) == 0  # Even wins
        assert result.winner(1) == 0


# ============================================================
# Section 13: solve_from_spec convenience
# ============================================================

class TestConvenience:
    def test_solve_from_spec_buchi(self):
        result = solve_from_spec(
            nodes=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (1, 2), (2, 0)],
            acc=AcceptanceCondition.buchi({1}),
        )
        assert result.winner_even == {0, 1, 2}

    def test_solve_from_spec_parity(self):
        result = solve_from_spec(
            nodes=[(0, 0), (1, 1)],
            edges=[(0, 1), (1, 0)],
            acc=AcceptanceCondition.parity({0: 2, 1: 1}),
        )
        # Max inf priority is 2 (even) -> Even wins
        assert 0 in result.winner_even

    def test_solve_from_spec_spm(self):
        result = solve_from_spec(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            acc=AcceptanceCondition.buchi({0}),
            algorithm="spm",
        )
        assert result.winner_even == {0, 1}


# ============================================================
# Section 14: Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_node_self_loop_buchi(self):
        arena = make_arena(nodes=[(0, 0)], edges=[(0, 0)])
        acc = AcceptanceCondition.buchi({0})
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0}

    def test_single_node_not_accepting(self):
        arena = make_arena(nodes=[(0, 0)], edges=[(0, 0)])
        acc = AcceptanceCondition.buchi(set())
        result = solve_omega_regular(arena, acc)
        assert result.winner_odd == {0}

    def test_disconnected_components(self):
        """Two disconnected self-loops."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 0), (1, 1)],
        )
        acc = AcceptanceCondition.buchi({0})
        result = solve_omega_regular(arena, acc)
        assert 0 in result.winner_even
        assert 1 in result.winner_odd

    def test_larger_game(self):
        """6-node game with mixed owners."""
        arena = make_arena(
            nodes=[(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)],
            edges=[
                (0, 1), (0, 2),
                (1, 3), (1, 4),
                (2, 4), (2, 5),
                (3, 0), (3, 5),
                (4, 0),
                (5, 5),
            ],
        )
        acc = AcceptanceCondition.buchi({4})
        result = solve_omega_regular(arena, acc)
        # Node 4 -> 0, and from 0 Even can reach 4 again
        assert 0 in result.winner_even
        assert 4 in result.winner_even

    def test_cobuchi_single_reject(self):
        """Single rejecting node in self-loop."""
        arena = make_arena(nodes=[(0, 0)], edges=[(0, 0)])
        acc = AcceptanceCondition.cobuchi({0})
        result = solve_omega_regular(arena, acc)
        assert 0 in result.winner_odd  # Must visit 0 infinitely -> rejected

    def test_muller_two_self_loops(self):
        """Muller with choice between two self-loop nodes."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        # Accepting sets: {1} or {2} -- staying at either is fine
        acc = AcceptanceCondition.muller({frozenset({1}), frozenset({2})})
        result = solve_omega_regular(arena, acc)
        assert 0 in result.winner_even

    def test_streett_empty_pairs(self):
        """No Streett pairs -> Even wins."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.streett([])
        result = solve_omega_regular(arena, acc)
        assert result.winner_even == {0, 1, 2}


# ============================================================
# Section 15: Algorithm agreement
# ============================================================

class TestAlgorithmAgreement:
    def test_zielonka_spm_agree_buchi(self):
        arena = diamond_arena()
        acc = AcceptanceCondition.buchi({3})
        r_z = solve_omega_regular(arena, acc, algorithm="zielonka")
        r_s = solve_omega_regular(arena, acc, algorithm="spm")
        assert r_z.winner_even == r_s.winner_even

    def test_zielonka_spm_agree_rabin(self):
        arena = two_player_choice()
        acc = AcceptanceCondition.rabin([({2}, {1})])
        r_z = solve_omega_regular(arena, acc, algorithm="zielonka")
        r_s = solve_omega_regular(arena, acc, algorithm="spm")
        assert r_z.winner_even == r_s.winner_even

    def test_zielonka_spm_agree_parity(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.parity({0: 4, 1: 3, 2: 2})
        r_z = solve_omega_regular(arena, acc, algorithm="zielonka")
        r_s = solve_omega_regular(arena, acc, algorithm="spm")
        assert r_z.winner_even == r_s.winner_even

    def test_zielonka_spm_agree_cobuchi(self):
        arena = two_player_choice()
        acc = AcceptanceCondition.cobuchi({2})
        r_z = solve_omega_regular(arena, acc, algorithm="zielonka")
        r_s = solve_omega_regular(arena, acc, algorithm="spm")
        assert r_z.winner_even == r_s.winner_even


# ============================================================
# Section 16: Muller via Rabin (alternative path)
# ============================================================

class TestMullerViaRabin:
    def test_muller_via_rabin_simple(self):
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.muller({frozenset({0, 1, 2})})
        result = solve_muller_via_rabin(arena, acc)
        assert result.winner_even == {0, 1, 2}

    def test_muller_via_rabin_singleton(self):
        arena = make_arena(
            nodes=[(0, 0), (1, 0)],
            edges=[(0, 1), (0, 0), (1, 1)],
        )
        acc = AcceptanceCondition.muller({frozenset({0})})
        result = solve_muller_via_rabin(arena, acc)
        # Node 0 can self-loop -> inf set {0} is accepting
        assert 0 in result.winner_even

    def test_muller_rabin_agrees_with_lar(self):
        """Both Muller reductions agree."""
        arena = two_player_choice()
        acc = AcceptanceCondition.muller({frozenset({1})})
        r_lar = solve_omega_regular(arena, acc)
        r_rabin = solve_muller_via_rabin(arena, acc)
        assert r_lar.winner_even == r_rabin.winner_even

    def test_muller_to_rabin_structure(self):
        table = {frozenset({0, 1})}
        pairs = muller_to_rabin(table, {0, 1, 2})
        assert len(pairs) == 1
        fin, inf = pairs[0]
        assert inf == {0, 1}
        assert fin == {2}


# ============================================================
# Section 17: Complex scenarios
# ============================================================

class TestComplexScenarios:
    def test_mutual_exclusion_buchi(self):
        """Model a simple mutual exclusion protocol.
        States: (process_state, turn)
        Even (process 0) wants to visit critical section infinitely.
        """
        # Simplified: 4 states
        # 0: idle (Even), 1: requesting (Even), 2: critical (Even), 3: other (Odd)
        arena = make_arena(
            nodes=[(0, 0), (1, 0), (2, 0), (3, 1)],
            edges=[(0, 1), (1, 2), (1, 3), (2, 0), (3, 0), (3, 1)],
        )
        acc = AcceptanceCondition.buchi({2})  # Visit critical infinitely
        result = solve_omega_regular(arena, acc)
        # Even can always go 0->1->2->0 cycle
        assert 0 in result.winner_even

    def test_repeated_reachability(self):
        """GF(target): reach target infinitely often."""
        arena = make_arena(
            nodes=[(0, 0), (1, 1), (2, 0), (3, 0)],
            edges=[(0, 1), (1, 2), (1, 3), (2, 0), (3, 3)],
        )
        # Even wants to visit 2 infinitely often
        # Odd at 1 can choose 2 or 3; if 3, stuck forever
        acc = AcceptanceCondition.buchi({2})
        result = solve_omega_regular(arena, acc)
        # From 0: goes to 1 (Odd). Odd can go to 3 (trap) -> Odd wins from 0
        assert 0 in result.winner_odd

    def test_safety_via_cobuchi(self):
        """Safety property: never visit bad state."""
        arena = make_arena(
            nodes=[(0, 0), (1, 0), (2, 0)],  # 2 is bad
            edges=[(0, 1), (0, 0), (1, 0), (1, 2), (2, 2)],
        )
        # co-Buchi: rejecting = {2}. Even wins if 2 visited finitely (ideally never).
        acc = AcceptanceCondition.cobuchi({2})
        result = solve_omega_regular(arena, acc)
        # From 0: Even can go to 0 (safe) or 1 -> from 1: Even can go to 0 (safe) or 2 (bad)
        # Even's strategy: always go back to 0 from 1
        assert 0 in result.winner_even

    def test_rabin_with_multiple_cycles(self):
        """Game with two cycles, different Rabin pairs apply."""
        # Cycle A: 0->1->0, Cycle B: 2->3->2
        # Even at 0 chooses cycle A or bridge to cycle B
        arena = make_arena(
            nodes=[(0, 0), (1, 0), (2, 0), (3, 0)],
            edges=[(0, 1), (0, 2), (1, 0), (2, 3), (3, 2)],
        )
        # Rabin: pair 0 requires visiting 3 inf AND NOT visiting 1 inf
        # -> must be in cycle B
        acc = AcceptanceCondition.rabin([({1}, {3})])
        result = solve_omega_regular(arena, acc)
        # Even at 0 can go to 2->3->2 cycle, never visit 1
        assert 0 in result.winner_even


# ============================================================
# Section 18: Integration with V076 types
# ============================================================

class TestV076Integration:
    def test_buchi_game_type_compat(self):
        """Verify our Buchi reduction matches V076's BuchiGame -> parity."""
        from parity_games import BuchiGame as V076BuchiGame, buchi_to_parity as v076_buchi

        arena = simple_cycle_arena()
        # Our reduction
        acc = AcceptanceCondition.buchi({1})
        our_pg = reduce_to_parity(arena, acc)

        # V076's direct reduction
        bg = V076BuchiGame(
            nodes={0, 1, 2},
            owner={0: PGPlayer.EVEN, 1: PGPlayer.EVEN, 2: PGPlayer.EVEN},
            successors={0: {1}, 1: {2}, 2: {0}},
            accepting={1},
        )
        v076_pg = v076_buchi(bg)

        # Same priorities
        for n in arena.nodes:
            assert our_pg.priority[n] == v076_pg.priority[n]

    def test_parity_result_type(self):
        """Result type has expected fields."""
        arena = simple_cycle_arena()
        acc = AcceptanceCondition.buchi({1})
        result = solve_omega_regular(arena, acc)
        assert isinstance(result, OmegaParityResult)
        assert isinstance(result.winner_even, set)
        assert isinstance(result.strategy_even, dict)
        assert isinstance(result.acc_type, AccType)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
