"""Tests for V168: Multi-Objective Parity Games."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))

from multi_objective_parity import (
    MultiParityGame, MultiSolution, Player, Objective,
    Atom, And, Or, Not, BoolExpr,
    solve_conjunctive, solve_disjunctive, solve_boolean,
    solve_conjunctive_streett,
    pareto_analysis,
    make_multi_parity_game, make_safety_liveness_game,
    make_multi_reachability_game,
    verify_multi_solution, compare_methods,
    compare_conjunctive_disjunctive,
    multi_parity_statistics,
    _project_objectives, _collect_atoms, _push_negation,
)


# ===================================================================
# Section 1: Data structures and construction
# ===================================================================

class TestConstruction:
    def test_empty_game(self):
        g = MultiParityGame()
        assert len(g.vertices) == 0
        assert g.k == 0

    def test_add_vertex(self):
        g = MultiParityGame()
        g.add_vertex(0, Player.EVEN, (2, 1))
        assert g.k == 2
        assert 0 in g.vertices
        assert g.owner[0] == Player.EVEN
        assert g.priorities[0] == (2, 1)

    def test_priority_dimension_mismatch(self):
        g = MultiParityGame()
        g.add_vertex(0, Player.EVEN, (1, 2))
        with pytest.raises(AssertionError):
            g.add_vertex(1, Player.ODD, (1, 2, 3))

    def test_add_edge(self):
        g = MultiParityGame()
        g.add_vertex(0, Player.EVEN, (0,))
        g.add_vertex(1, Player.ODD, (1,))
        g.add_edge(0, 1)
        assert 1 in g.successors(0)
        assert 0 in g.predecessors(1)

    def test_max_priority(self):
        g = MultiParityGame()
        g.add_vertex(0, Player.EVEN, (2, 5))
        g.add_vertex(1, Player.ODD, (3, 1))
        assert g.max_priority(0) == 3
        assert g.max_priority(1) == 5

    def test_subgame(self):
        g = MultiParityGame()
        g.add_vertex(0, Player.EVEN, (0, 1))
        g.add_vertex(1, Player.ODD, (1, 2))
        g.add_vertex(2, Player.EVEN, (2, 0))
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        sub = g.subgame({0, 1})
        assert sub.vertices == {0, 1}
        assert 1 in sub.successors(0)
        assert sub.successors(1) == set()  # edge to 2 removed

    def test_projection(self):
        g = MultiParityGame()
        g.add_vertex(0, Player.EVEN, (2, 1))
        g.add_vertex(1, Player.ODD, (3, 0))
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        pg = g.projection(0)
        assert pg.priority[0] == 2
        assert pg.priority[1] == 3

    def test_make_multi_parity_game(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 1)), (1, 1, (1, 2))],
            edges=[(0, 1), (1, 0)],
        )
        assert g.k == 2
        assert len(g.vertices) == 2

    def test_make_safety_liveness(self):
        g = make_safety_liveness_game(
            vertices=[(0, 0), (1, 1), (2, 0)],
            edges=[(0, 1), (1, 2), (2, 0)],
            safe={0, 1, 2},
            live={2},
        )
        assert g.k == 2
        assert g.priorities[0] == (0, 1)  # safe, not live
        assert g.priorities[2] == (0, 2)  # safe, live

    def test_make_multi_reachability(self):
        g = make_multi_reachability_game(
            vertices=[(0, 0), (1, 1), (2, 0)],
            edges=[(0, 1), (1, 2), (2, 0)],
            targets=[{1}, {2}],
        )
        assert g.k == 2
        assert g.priorities[1] == (2, 1)  # target0 yes, target1 no


# ===================================================================
# Section 2: Conjunctive solving -- product construction
# ===================================================================

class TestConjunctive:
    def test_empty_game(self):
        g = MultiParityGame()
        sol = solve_conjunctive(g)
        assert sol.win_even == set()
        assert sol.win_odd == set()

    def test_single_objective(self):
        """Single objective reduces to standard parity game."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2,)), (1, 1, (1,))],
            edges=[(0, 1), (1, 0)],
        )
        sol = solve_conjunctive(g)
        # Even priority 2 dominates odd priority 1 => Even wins from 0
        assert 0 in sol.win_even

    def test_two_objectives_both_even(self):
        """Both objectives have even max priority => Even wins."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 4))],
            edges=[(0, 0)],  # self-loop
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even

    def test_two_objectives_one_odd(self):
        """One objective has odd max priority => Odd wins conjunction."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 1))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive(g)
        # Objective 0: priority 2 (even) -- satisfied
        # Objective 1: priority 1 (odd) -- NOT satisfied
        # Conjunction fails
        assert 0 in sol.win_odd

    def test_two_objectives_both_odd(self):
        """Both objectives have odd max priority => Odd wins."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (1, 3))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_odd

    def test_choice_satisfies_both(self):
        """Even can choose a path satisfying both objectives."""
        # Vertex 0 (Even): choose left (1) or right (2)
        # Vertex 1: self-loop, priorities (2, 2) -- both even
        # Vertex 2: self-loop, priorities (1, 1) -- both odd
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (0, 0)),
                (1, 0, (2, 2)),
                (2, 0, (1, 1)),
            ],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even
        assert 1 in sol.win_even
        assert 2 in sol.win_odd

    def test_tradeoff_objectives(self):
        """Even must choose: satisfy obj 0 OR obj 1, but not both."""
        # Vertex 0 (Even): choose vertex 1 or 2
        # Vertex 1: priorities (2, 1) -- obj 0 good, obj 1 bad
        # Vertex 2: priorities (1, 2) -- obj 0 bad, obj 1 good
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (0, 0)),
                (1, 0, (2, 1)),
                (2, 0, (1, 2)),
            ],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        sol = solve_conjunctive(g)
        # Can't satisfy both simultaneously
        assert 0 in sol.win_odd

    def test_cycle_satisfies_both(self):
        """Cycling through vertices satisfies both objectives."""
        # Cycle: 0 -> 1 -> 0
        # Vertex 0: (2, 1) -- obj 0 even, obj 1 odd
        # Vertex 1: (1, 2) -- obj 0 odd, obj 1 even
        # On the cycle: max even prio dominates for both objectives
        # Obj 0: sees 2 (even) and 1 (odd), 2 dominates => satisfied
        # Obj 1: sees 1 (odd) and 2 (even), 2 dominates => satisfied
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 1)),
                (1, 0, (1, 2)),
            ],
            edges=[(0, 1), (1, 0)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even
        assert 1 in sol.win_even

    def test_opponent_forces_failure(self):
        """Odd can force Even to fail at least one objective."""
        # Vertex 0 (Odd): chooses 1 or 2
        # Vertex 1: (2, 1) -- obj 0 ok, obj 1 fail
        # Vertex 2: (1, 2) -- obj 0 fail, obj 1 ok
        g = make_multi_parity_game(
            vertices=[
                (0, 1, (0, 0)),
                (1, 0, (2, 1)),
                (2, 0, (1, 2)),
            ],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_odd

    def test_three_objectives(self):
        """Three objectives all satisfied on a self-loop."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 4, 6))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even

    def test_three_objectives_one_fails(self):
        """Three objectives, one fails."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 4, 1))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_odd

    def test_partition_property(self):
        """Solution is a valid partition."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 2)), (1, 1, (1, 1)),
                (2, 0, (0, 2)), (3, 1, (2, 0)),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
        )
        sol = solve_conjunctive(g)
        assert sol.win_even | sol.win_odd == g.vertices
        assert not (sol.win_even & sol.win_odd)


# ===================================================================
# Section 3: Disjunctive solving
# ===================================================================

class TestDisjunctive:
    def test_empty_game(self):
        g = MultiParityGame()
        sol = solve_disjunctive(g)
        assert sol.win_even == set()

    def test_single_objective(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2,)), (1, 1, (1,))],
            edges=[(0, 1), (1, 0)],
        )
        sol = solve_disjunctive(g)
        assert 0 in sol.win_even

    def test_disjunction_easier_than_conjunction(self):
        """Even can satisfy at least one objective even if not both."""
        # Vertex 0 (Even): choose 1 or 2
        # Vertex 1: (2, 1) -- obj 0 ok, obj 1 fail
        # Vertex 2: (1, 2) -- obj 0 fail, obj 1 ok
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (0, 0)),
                (1, 0, (2, 1)),
                (2, 0, (1, 2)),
            ],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        conj = solve_conjunctive(g)
        disj = solve_disjunctive(g)
        # Conjunction: Even can't satisfy both => loses from 0
        assert 0 in conj.win_odd
        # Disjunction: Even can satisfy one => wins from 0
        assert 0 in disj.win_even

    def test_both_fail(self):
        """Neither objective can be satisfied."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (1, 1))],
            edges=[(0, 0)],
        )
        sol = solve_disjunctive(g)
        assert 0 in sol.win_odd

    def test_one_satisfiable(self):
        """At least one objective is satisfiable."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 1))],
            edges=[(0, 0)],
        )
        sol = solve_disjunctive(g)
        assert 0 in sol.win_even

    def test_opponent_controls_disjunction(self):
        """Odd chooses, but both paths satisfy at least one objective."""
        g = make_multi_parity_game(
            vertices=[
                (0, 1, (0, 0)),
                (1, 0, (2, 1)),  # obj 0 satisfied
                (2, 0, (1, 2)),  # obj 1 satisfied
            ],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        sol = solve_disjunctive(g)
        # No matter where Odd goes, Even satisfies at least one objective
        assert 0 in sol.win_even

    def test_partition_property(self):
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 1)), (1, 1, (1, 2)),
                (2, 0, (2, 2)), (3, 1, (1, 1)),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        )
        sol = solve_disjunctive(g)
        assert sol.win_even | sol.win_odd == g.vertices
        assert not (sol.win_even & sol.win_odd)

    def test_conj_subset_disj(self):
        """Conjunctive winners are always a subset of disjunctive winners."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 2)), (1, 1, (1, 2)),
                (2, 0, (2, 1)), (3, 1, (1, 1)),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 0)],
        )
        conj = solve_conjunctive(g)
        disj = solve_disjunctive(g)
        assert conj.win_even <= disj.win_even


# ===================================================================
# Section 4: Boolean combinations
# ===================================================================

class TestBoolean:
    def test_single_atom(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 1))],
            edges=[(0, 0)],
        )
        sol = solve_boolean(g, Atom(0))
        assert 0 in sol.win_even

    def test_negated_atom(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 1))],
            edges=[(0, 0)],
        )
        # Not(Atom(0)): complement of obj 0, priority 2 -> 3 (odd) => Odd wins
        sol = solve_boolean(g, Not(Atom(0)))
        assert 0 in sol.win_odd

    def test_and_of_atoms(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 2))],
            edges=[(0, 0)],
        )
        sol = solve_boolean(g, And([Atom(0), Atom(1)]))
        assert 0 in sol.win_even

    def test_or_of_atoms(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 1))],
            edges=[(0, 0)],
        )
        sol = solve_boolean(g, Or([Atom(0), Atom(1)]))
        assert 0 in sol.win_even

    def test_not_and(self):
        """Not(And(A, B)) = Or(Not(A), Not(B))."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 2))],
            edges=[(0, 0)],
        )
        # Not(And(0, 1)): even if both even, negation makes them odd
        sol = solve_boolean(g, Not(And([Atom(0), Atom(1)])))
        # Complement of both: both become odd => Odd wins both => Or(Odd, Odd) = Odd wins
        # Actually: Not(Atom(0)) makes prio 3 (odd), Not(Atom(1)) makes prio 3 (odd)
        # Or(Not(0), Not(1)): either one being odd suffices => Even wins if either complement is satisfied
        # But complement of even-2 is odd-3 => Odd wins complement => Even loses Not(Atom(i))
        # So Or(lose, lose) = lose
        assert 0 in sol.win_odd

    def test_double_negation(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2,))],
            edges=[(0, 0)],
        )
        sol = solve_boolean(g, Not(Not(Atom(0))))
        assert 0 in sol.win_even

    def test_collect_atoms(self):
        expr = And([Atom(0), Or([Atom(1), Not(Atom(2))])])
        assert _collect_atoms(expr) == {0, 1, 2}

    def test_push_negation_demorgan(self):
        # Not(And([A, B])) -> Or([Not(A), Not(B)])
        expr = Not(And([Atom(0), Atom(1)]))
        nnf = _push_negation(expr)
        assert isinstance(nnf, Or)
        assert len(nnf.children) == 2
        assert all(isinstance(c, Not) for c in nnf.children)

    def test_complex_boolean(self):
        """(obj0 AND obj1) OR (NOT obj2)."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 2, 1))],  # obj0=even, obj1=even, obj2=odd
            edges=[(0, 0)],
        )
        expr = Or([And([Atom(0), Atom(1)]), Not(Atom(2))])
        sol = solve_boolean(g, expr)
        # And(0,1) is satisfied (both even), so Or is satisfied
        assert 0 in sol.win_even


# ===================================================================
# Section 5: Streett reduction
# ===================================================================

class TestStreettReduction:
    def test_empty_game(self):
        g = MultiParityGame()
        sol = solve_conjunctive_streett(g)
        assert sol.win_even == set()

    def test_all_even_priorities(self):
        """No odd priorities => Even wins everywhere."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 4)), (1, 1, (0, 2))],
            edges=[(0, 1), (1, 0)],
        )
        sol = solve_conjunctive_streett(g)
        assert sol.win_even == {0, 1}

    def test_matches_product(self):
        """Streett method matches product method."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 1)),
                (1, 1, (1, 2)),
                (2, 0, (2, 2)),
            ],
            edges=[(0, 1), (1, 2), (2, 0), (0, 2)],
        )
        prod = solve_conjunctive(g)
        street = solve_conjunctive_streett(g)
        assert prod.win_even == street.win_even

    def test_simple_cycle(self):
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 1)),
                (1, 0, (1, 2)),
            ],
            edges=[(0, 1), (1, 0)],
        )
        sol = solve_conjunctive_streett(g)
        # Both objectives satisfied on cycle
        assert 0 in sol.win_even

    def test_unsatisfiable(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (1, 1))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive_streett(g)
        assert 0 in sol.win_odd


# ===================================================================
# Section 6: Safety-liveness games
# ===================================================================

class TestSafetyLiveness:
    def test_safe_and_live(self):
        """All vertices safe, one live => Even can cycle through live."""
        g = make_safety_liveness_game(
            vertices=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            safe={0, 1},
            live={1},
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even
        assert 1 in sol.win_even

    def test_unsafe_vertex(self):
        """Forced to visit unsafe vertex => conjunction fails."""
        g = make_safety_liveness_game(
            vertices=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 1)],
            safe={0},  # vertex 1 is unsafe
            live={0, 1},
        )
        sol = solve_conjunctive(g)
        # Vertex 0 must go to 1 (unsafe), then stuck there
        # Safety fails for vertex 1, so even if liveness satisfied, conjunction fails
        assert 1 in sol.win_odd

    def test_no_live_vertex(self):
        """No live vertices => liveness fails."""
        g = make_safety_liveness_game(
            vertices=[(0, 0)],
            edges=[(0, 0)],
            safe={0},
            live=set(),
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_odd


# ===================================================================
# Section 7: Multi-reachability games
# ===================================================================

class TestMultiReachability:
    def test_single_target(self):
        g = make_multi_reachability_game(
            vertices=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            targets=[{1}],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even

    def test_two_targets_reachable(self):
        """Cycle visits both targets."""
        g = make_multi_reachability_game(
            vertices=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (1, 2), (2, 0)],
            targets=[{1}, {2}],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even

    def test_two_targets_choice(self):
        """Even must choose one path, can't visit both targets."""
        g = make_multi_reachability_game(
            vertices=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
            targets=[{1}, {2}],
        )
        sol = solve_conjunctive(g)
        # Can only reach one target (self-loops), not both
        assert 0 in sol.win_odd


# ===================================================================
# Section 8: Pareto analysis
# ===================================================================

class TestParetoAnalysis:
    def test_all_satisfiable(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 2))],
            edges=[(0, 0)],
        )
        result = pareto_analysis(g)
        assert result['per_vertex'][0]['satisfiable_individual'] == {0, 1}
        assert result['per_vertex'][0]['conjunction_size'] == 2

    def test_one_satisfiable(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 1))],
            edges=[(0, 0)],
        )
        result = pareto_analysis(g)
        assert 0 in result['per_vertex'][0]['satisfiable_individual']
        assert 1 not in result['per_vertex'][0]['satisfiable_individual']

    def test_tradeoff(self):
        """Can satisfy each individually but not both."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (0, 0)),
                (1, 0, (2, 1)),
                (2, 0, (1, 2)),
            ],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
        )
        result = pareto_analysis(g)
        # From vertex 0: can satisfy obj 0 (go to 1) or obj 1 (go to 2), not both
        assert result['per_vertex'][0]['satisfiable_individual'] == {0, 1}
        assert result['per_vertex'][0]['conjunction_size'] <= 1


# ===================================================================
# Section 9: Verification
# ===================================================================

class TestVerification:
    def test_valid_solution(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 2)), (1, 1, (1, 1))],
            edges=[(0, 1), (1, 0), (0, 0)],
        )
        sol = solve_conjunctive(g)
        result = verify_multi_solution(g, sol)
        assert result['valid']

    def test_invalid_partition(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2,)), (1, 1, (1,))],
            edges=[(0, 1), (1, 0)],
        )
        bad_sol = MultiSolution(
            win_even={0},
            win_odd=set(),  # missing vertex 1
        )
        result = verify_multi_solution(g, bad_sol)
        assert not result['valid']

    def test_invalid_strategy(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (2,)), (1, 1, (1,))],
            edges=[(0, 1), (1, 0)],
        )
        bad_sol = MultiSolution(
            win_even={0, 1},
            win_odd=set(),
            strategy_even={0: 99},  # non-existent successor
        )
        result = verify_multi_solution(g, bad_sol)
        assert not result['valid']


# ===================================================================
# Section 10: Comparison APIs
# ===================================================================

class TestComparison:
    def test_compare_methods(self):
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 2)),
                (1, 1, (1, 1)),
            ],
            edges=[(0, 1), (1, 0), (0, 0)],
        )
        result = compare_methods(g)
        assert result['agree']

    def test_compare_conj_disj(self):
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 1)),
                (1, 0, (1, 2)),
            ],
            edges=[(0, 1), (1, 0)],
        )
        result = compare_conjunctive_disjunctive(g)
        assert result['conj_subset_disj']

    def test_statistics(self):
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 1)),
                (1, 1, (1, 2)),
            ],
            edges=[(0, 1), (1, 0)],
        )
        stats = multi_parity_statistics(g)
        assert stats['vertices'] == 2
        assert stats['edges'] == 2
        assert stats['objectives'] == 2


# ===================================================================
# Section 11: Edge cases and stress tests
# ===================================================================

class TestEdgeCases:
    def test_single_vertex_even_wins(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (0, 0, 0))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even

    def test_single_vertex_odd_wins(self):
        g = make_multi_parity_game(
            vertices=[(0, 1, (1, 1))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_odd

    def test_linear_chain(self):
        """Chain: 0->1->2->2 with mixed priorities."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (0, 0)),
                (1, 1, (1, 0)),
                (2, 0, (2, 2)),
            ],
            edges=[(0, 1), (1, 2), (2, 2)],
        )
        sol = solve_conjunctive(g)
        # Everyone ends at vertex 2 with (2,2) => Even wins
        assert 2 in sol.win_even

    def test_project_objectives(self):
        g = make_multi_parity_game(
            vertices=[(0, 0, (1, 2, 3)), (1, 1, (4, 5, 6))],
            edges=[(0, 1), (1, 0)],
        )
        sub = _project_objectives(g, [0, 2])
        assert sub.k == 2
        assert sub.priorities[0] == (1, 3)
        assert sub.priorities[1] == (4, 6)

    def test_high_priority_domination(self):
        """Higher even priority dominates lower odd."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (4, 6)),
                (1, 0, (3, 5)),
            ],
            edges=[(0, 1), (1, 0)],
        )
        sol = solve_conjunctive(g)
        # Obj 0: sees 4(even) and 3(odd), 4 dominates => satisfied
        # Obj 1: sees 6(even) and 5(odd), 6 dominates => satisfied
        assert 0 in sol.win_even

    def test_disjunctive_all_fail(self):
        """All objectives fail => disjunction fails."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (1, 3, 5))],
            edges=[(0, 0)],
        )
        sol = solve_disjunctive(g)
        assert 0 in sol.win_odd

    def test_four_objectives(self):
        """Four objectives, all even."""
        g = make_multi_parity_game(
            vertices=[(0, 0, (2, 2, 2, 2))],
            edges=[(0, 0)],
        )
        sol = solve_conjunctive(g)
        assert 0 in sol.win_even

    def test_diamond_game(self):
        """Diamond: 0->{1,2}, 1->3, 2->3, 3->0."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (0, 0)),
                (1, 1, (2, 1)),
                (2, 1, (1, 2)),
                (3, 0, (0, 0)),
            ],
            edges=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)],
        )
        conj = solve_conjunctive(g)
        disj = solve_disjunctive(g)
        # Verify conj <= disj
        assert conj.win_even <= disj.win_even

    def test_large_game(self):
        """10 vertices, 2 objectives."""
        verts = [(i, i % 2, (i % 3, (i + 1) % 3)) for i in range(10)]
        edges = [(i, (i + 1) % 10) for i in range(10)]
        edges += [(i, (i + 3) % 10) for i in range(10)]
        g = make_multi_parity_game(verts, edges)
        sol = solve_conjunctive(g)
        assert sol.win_even | sol.win_odd == g.vertices
        assert not (sol.win_even & sol.win_odd)


# ===================================================================
# Section 12: Integration -- combining features
# ===================================================================

class TestIntegration:
    def test_safety_liveness_with_disjunctive(self):
        """Disjunctive on safety-liveness game: satisfy either."""
        g = make_safety_liveness_game(
            vertices=[(0, 0), (1, 0)],
            edges=[(0, 1), (1, 0)],
            safe={0, 1},
            live=set(),
        )
        sol = solve_disjunctive(g)
        # Safety satisfied (all safe), liveness fails (no live)
        # Disjunction: safety alone suffices
        # Obj 0 (safety): priority 0 everywhere (even) => satisfied
        assert 0 in sol.win_even

    def test_multi_reach_disjunctive(self):
        """Disjunctive multi-reachability: reach any one target."""
        g = make_multi_reachability_game(
            vertices=[(0, 0), (1, 0), (2, 0)],
            edges=[(0, 1), (0, 2), (1, 1), (2, 2)],
            targets=[{1}, {2}],
        )
        sol = solve_disjunctive(g)
        # Can reach target 0 (go to 1) or target 1 (go to 2)
        assert 0 in sol.win_even

    def test_boolean_on_multi_reach(self):
        """Boolean: reach target0 OR (reach target1 AND reach target2)."""
        g = make_multi_reachability_game(
            vertices=[(0, 0), (1, 0), (2, 0), (3, 0)],
            edges=[(0, 1), (0, 2), (1, 1), (2, 3), (3, 2)],
            targets=[{1}, {2}, {3}],
        )
        expr = Or([Atom(0), And([Atom(1), Atom(2)])])
        sol = solve_boolean(g, expr)
        # Can go to 1 (satisfies Atom(0)) => Even wins
        assert 0 in sol.win_even

    def test_full_pipeline(self):
        """Build game, solve conj+disj, verify, compare, pareto."""
        g = make_multi_parity_game(
            vertices=[
                (0, 0, (2, 0)),
                (1, 1, (0, 2)),
                (2, 0, (1, 1)),
            ],
            edges=[(0, 1), (1, 2), (2, 0), (0, 2)],
        )
        conj = solve_conjunctive(g)
        disj = solve_disjunctive(g)
        v_conj = verify_multi_solution(g, conj)
        v_disj = verify_multi_solution(g, disj)
        assert v_conj['valid']
        assert v_disj['valid']
        assert conj.win_even <= disj.win_even

        stats = multi_parity_statistics(g)
        assert stats['objectives'] == 2

        comp = compare_methods(g)
        # Product and Streett should agree
        assert comp['agree']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
