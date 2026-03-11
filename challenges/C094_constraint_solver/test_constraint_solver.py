"""Tests for C094: Constraint Solver."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from constraint_solver import (
    CSPSolver, CSPResult, Variable, SearchStrategy,
    EqualityConstraint, InequalityConstraint, ComparisonConstraint,
    AllDifferentConstraint, TableConstraint, ArithmeticConstraint,
    CallbackConstraint, SumConstraint,
    sudoku, n_queens, graph_coloring, scheduling, magic_square,
    latin_square, knapsack,
)


# ============================================================
# Core Variable Tests
# ============================================================

class TestVariable:
    def test_create_variable(self):
        v = Variable("x", [1, 2, 3])
        assert v.name == "x"
        assert v.domain == {1, 2, 3}

    def test_empty_domain_raises(self):
        with pytest.raises(ValueError):
            Variable("x", [])

    def test_domain_deduplication(self):
        v = Variable("x", [1, 1, 2, 2, 3])
        assert v.domain == {1, 2, 3}

    def test_repr(self):
        v = Variable("x", [3, 1, 2])
        assert "x" in repr(v)


# ============================================================
# Constraint Tests
# ============================================================

class TestEqualityConstraint:
    def test_satisfied(self):
        c = EqualityConstraint("x", "y")
        assert c.satisfied({"x": 5, "y": 5})
        assert not c.satisfied({"x": 5, "y": 6})

    def test_partial_assignment(self):
        c = EqualityConstraint("x", "y")
        assert c.satisfied({"x": 5})

    def test_propagation(self):
        c = EqualityConstraint("x", "y")
        domains = {"x": {3}, "y": {1, 2, 3, 4}}
        result = c.propagate("x", 3, domains)
        assert domains["y"] == {3}
        assert ("y", 1) in result
        assert ("y", 2) in result
        assert ("y", 4) in result


class TestInequalityConstraint:
    def test_satisfied(self):
        c = InequalityConstraint("x", "y")
        assert c.satisfied({"x": 5, "y": 6})
        assert not c.satisfied({"x": 5, "y": 5})

    def test_propagation_removes_value(self):
        c = InequalityConstraint("x", "y")
        domains = {"x": {3}, "y": {1, 2, 3}}
        result = c.propagate("x", 3, domains)
        assert 3 not in domains["y"]
        assert ("y", 3) in result

    def test_propagation_wipeout(self):
        c = InequalityConstraint("x", "y")
        domains = {"x": {3}, "y": {3}}
        result = c.propagate("x", 3, domains)
        assert result is None


class TestComparisonConstraint:
    def test_less_than(self):
        c = ComparisonConstraint("x", "y", '<')
        assert c.satisfied({"x": 3, "y": 5})
        assert not c.satisfied({"x": 5, "y": 3})
        assert not c.satisfied({"x": 3, "y": 3})

    def test_less_equal(self):
        c = ComparisonConstraint("x", "y", '<=')
        assert c.satisfied({"x": 3, "y": 3})
        assert not c.satisfied({"x": 4, "y": 3})

    def test_greater_than(self):
        c = ComparisonConstraint("x", "y", '>')
        assert c.satisfied({"x": 5, "y": 3})
        assert not c.satisfied({"x": 3, "y": 5})

    def test_greater_equal(self):
        c = ComparisonConstraint("x", "y", '>=')
        assert c.satisfied({"x": 3, "y": 3})

    def test_invalid_op(self):
        with pytest.raises(ValueError):
            ComparisonConstraint("x", "y", '==')

    def test_propagation(self):
        c = ComparisonConstraint("x", "y", '<')
        domains = {"x": {1, 2, 3}, "y": {1, 2, 3}}
        result = c.propagate("x", 2, domains)
        # x=2, need y > 2, so y must be 3
        assert 1 not in domains["y"]
        assert 2 not in domains["y"]
        assert 3 in domains["y"]


class TestAllDifferentConstraint:
    def test_satisfied(self):
        c = AllDifferentConstraint(["x", "y", "z"])
        assert c.satisfied({"x": 1, "y": 2, "z": 3})
        assert not c.satisfied({"x": 1, "y": 1, "z": 3})

    def test_partial(self):
        c = AllDifferentConstraint(["x", "y", "z"])
        assert c.satisfied({"x": 1, "y": 2})
        assert not c.satisfied({"x": 1, "y": 1})

    def test_propagation(self):
        c = AllDifferentConstraint(["x", "y", "z"])
        domains = {"x": {1}, "y": {1, 2, 3}, "z": {1, 2, 3}}
        result = c.propagate("x", 1, domains)
        assert 1 not in domains["y"]
        assert 1 not in domains["z"]


class TestTableConstraint:
    def test_satisfied(self):
        c = TableConstraint(["x", "y"], [(1, 2), (2, 3), (3, 1)])
        assert c.satisfied({"x": 1, "y": 2})
        assert c.satisfied({"x": 2, "y": 3})
        assert not c.satisfied({"x": 1, "y": 3})

    def test_partial_check(self):
        c = TableConstraint(["x", "y"], [(1, 2), (2, 3)])
        assert c._check_partial({"x": 1})
        assert not c._check_partial({"x": 3, "y": 3})

    def test_propagation(self):
        c = TableConstraint(["x", "y"], [(1, 2), (1, 3), (2, 3)])
        domains = {"x": {1, 2}, "y": {1, 2, 3}}
        result = c.propagate("x", 1, domains)
        # x=1 -> y in {2, 3}
        assert 1 not in domains["y"]
        assert 2 in domains["y"]
        assert 3 in domains["y"]


class TestArithmeticConstraint:
    def test_equality(self):
        c = ArithmeticConstraint({"x": 1, "y": 1}, '==', 10)
        assert c.satisfied({"x": 3, "y": 7})
        assert not c.satisfied({"x": 3, "y": 6})

    def test_with_coefficients(self):
        c = ArithmeticConstraint({"x": 2, "y": 3}, '<=', 12)
        assert c.satisfied({"x": 3, "y": 2})  # 6+6=12
        assert not c.satisfied({"x": 3, "y": 3})  # 6+9=15

    def test_inequality(self):
        c = ArithmeticConstraint({"x": 1, "y": -1}, '!=', 0)
        assert c.satisfied({"x": 3, "y": 5})
        assert not c.satisfied({"x": 3, "y": 3})


class TestSumConstraint:
    def test_basic(self):
        c = SumConstraint(["x", "y", "z"], 10)
        assert c.satisfied({"x": 2, "y": 3, "z": 5})
        assert not c.satisfied({"x": 2, "y": 3, "z": 4})

    def test_propagation_one_free(self):
        c = SumConstraint(["x", "y", "z"], 10)
        domains = {"x": {3}, "y": {4}, "z": {1, 2, 3, 4, 5}}
        result = c.propagate("x", 3, domains)
        assert domains["z"] == {3}

    def test_propagation_bounds(self):
        c = SumConstraint(["x", "y", "z"], 6)
        domains = {"x": {1}, "y": {1, 2, 3}, "z": {1, 2, 3}}
        result = c.propagate("x", 1, domains)
        # x=1, y+z=5, y in {1,2,3}, z in {1,2,3}
        # y: 5-max(z)=2, 5-min(z)=4 -> y in {2,3}
        # z: 5-max(y)=2, 5-min(y)=4 -> z in {2,3}
        assert 1 not in domains["y"]
        assert 1 not in domains["z"]


class TestCallbackConstraint:
    def test_basic(self):
        c = CallbackConstraint(["x", "y"], lambda a: a["x"] + a["y"] == 10)
        assert c.satisfied({"x": 3, "y": 7})
        assert not c.satisfied({"x": 3, "y": 6})

    def test_partial(self):
        c = CallbackConstraint(["x", "y"], lambda a: a["x"] + a["y"] == 10)
        assert c.satisfied({"x": 3})


# ============================================================
# CSP Solver Core Tests
# ============================================================

class TestCSPSolverBasic:
    def test_trivial_problem(self):
        csp = CSPSolver()
        csp.add_variable("x", [1])
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment == {"x": 1}

    def test_simple_inequality(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2])
        csp.add_variable("y", [1, 2])
        csp.add_inequality("x", "y")
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] != assignment["y"]

    def test_unsatisfiable(self):
        csp = CSPSolver()
        csp.add_variable("x", [1])
        csp.add_variable("y", [1])
        csp.add_inequality("x", "y")
        result, _ = csp.solve()
        assert result == CSPResult.UNSATISFIABLE

    def test_three_vars_alldiff(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_variable("z", [1, 2, 3])
        csp.add_alldiff(["x", "y", "z"])
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert len(set(assignment.values())) == 3

    def test_equality_chain(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_variable("z", [1, 2, 3])
        csp.add_equality("x", "y")
        csp.add_equality("y", "z")
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] == assignment["y"] == assignment["z"]

    def test_comparison_chain(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 5))
        csp.add_variable("y", range(1, 5))
        csp.add_variable("z", range(1, 5))
        csp.add_comparison("x", "y", '<')
        csp.add_comparison("y", "z", '<')
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] < assignment["y"] < assignment["z"]

    def test_mixed_constraints(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 6))
        csp.add_variable("y", range(1, 6))
        csp.add_comparison("x", "y", '<')
        csp.add_arithmetic({"x": 1, "y": 1}, '==', 6)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] < assignment["y"]
        assert assignment["x"] + assignment["y"] == 6

    def test_stats_populated(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_inequality("x", "y")
        csp.solve()
        assert csp.stats['nodes'] >= 0
        assert 'backtracks' in csp.stats


class TestCSPSolverStrategies:
    def _make_problem(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_variable("z", [1, 2, 3])
        csp.add_alldiff(["x", "y", "z"])
        return csp

    def test_backtracking(self):
        csp = self._make_problem()
        result, assignment = csp.solve(strategy=SearchStrategy.BACKTRACKING)
        assert result == CSPResult.SOLVED
        assert len(set(assignment.values())) == 3

    def test_forward_checking(self):
        csp = self._make_problem()
        result, assignment = csp.solve(strategy=SearchStrategy.FORWARD_CHECKING)
        assert result == CSPResult.SOLVED
        assert len(set(assignment.values())) == 3

    def test_mac(self):
        csp = self._make_problem()
        result, assignment = csp.solve(strategy=SearchStrategy.MAC)
        assert result == CSPResult.SOLVED
        assert len(set(assignment.values())) == 3


class TestSolveAll:
    def test_find_all_solutions(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2])
        csp.add_variable("y", [1, 2])
        csp.add_inequality("x", "y")
        solutions = csp.solve_all()
        assert len(solutions) == 2
        assert {"x": 1, "y": 2} in solutions
        assert {"x": 2, "y": 1} in solutions

    def test_max_solutions(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_variable("z", [1, 2, 3])
        csp.add_alldiff(["x", "y", "z"])
        solutions = csp.solve_all(max_solutions=2)
        assert len(solutions) == 2

    def test_no_solutions(self):
        csp = CSPSolver()
        csp.add_variable("x", [1])
        csp.add_variable("y", [1])
        csp.add_inequality("x", "y")
        solutions = csp.solve_all()
        assert solutions == []

    def test_all_permutations(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_variable("z", [1, 2, 3])
        csp.add_alldiff(["x", "y", "z"])
        solutions = csp.solve_all()
        assert len(solutions) == 6  # 3! permutations


# ============================================================
# Arc Consistency Tests
# ============================================================

class TestArcConsistency:
    def test_ac3_reduces_domains(self):
        csp = CSPSolver()
        csp.add_variable("x", [1])
        csp.add_variable("y", [1, 2, 3])
        csp.add_inequality("x", "y")
        domains = {"x": {1}, "y": {1, 2, 3}}
        csp._ac3(domains)
        assert 1 not in domains["y"]

    def test_ac3_detects_wipeout(self):
        csp = CSPSolver()
        csp.add_variable("x", [1])
        csp.add_variable("y", [1])
        csp.add_inequality("x", "y")
        domains = {"x": {1}, "y": {1}}
        result = csp._ac3(domains)
        assert result is False

    def test_ac3_chain_propagation(self):
        csp = CSPSolver()
        csp.add_variable("x", [1])
        csp.add_variable("y", [1, 2])
        csp.add_variable("z", [1, 2, 3])
        csp.add_equality("x", "y")
        csp.add_inequality("y", "z")
        domains = {"x": {1}, "y": {1, 2}, "z": {1, 2, 3}}
        csp._ac3(domains)
        # x=1 -> y=1 -> z != 1
        assert domains["y"] == {1}
        assert 1 not in domains["z"]


# ============================================================
# SAT Encoding Tests
# ============================================================

class TestSATEncoding:
    def test_simple_sat(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2])
        csp.add_variable("y", [1, 2])
        csp.add_inequality("x", "y")
        result, assignment = csp.solve_with_sat()
        assert result == CSPResult.SOLVED
        assert assignment["x"] != assignment["y"]

    def test_sat_unsatisfiable(self):
        csp = CSPSolver()
        csp.add_variable("x", [1])
        csp.add_variable("y", [1])
        csp.add_inequality("x", "y")
        result, _ = csp.solve_with_sat()
        assert result == CSPResult.UNSATISFIABLE

    def test_sat_alldiff(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_variable("z", [1, 2, 3])
        csp.add_alldiff(["x", "y", "z"])
        result, assignment = csp.solve_with_sat()
        assert result == CSPResult.SOLVED
        assert len(set(assignment.values())) == 3

    def test_sat_equality(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_equality("x", "y")
        result, assignment = csp.solve_with_sat()
        assert result == CSPResult.SOLVED
        assert assignment["x"] == assignment["y"]

    def test_sat_table_constraint(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_table(["x", "y"], [(1, 2), (2, 3), (3, 1)])
        result, assignment = csp.solve_with_sat()
        assert result == CSPResult.SOLVED
        assert (assignment["x"], assignment["y"]) in [(1, 2), (2, 3), (3, 1)]

    def test_sat_comparison(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_comparison("x", "y", '<')
        result, assignment = csp.solve_with_sat()
        assert result == CSPResult.SOLVED
        assert assignment["x"] < assignment["y"]


# ============================================================
# SMT Integration Tests
# ============================================================

class TestSMTIntegration:
    def test_simple_smt(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 6))
        csp.add_variable("y", range(1, 6))
        csp.add_inequality("x", "y")
        result, assignment = csp.solve_with_smt()
        assert result == CSPResult.SOLVED
        assert assignment["x"] != assignment["y"]

    def test_smt_arithmetic(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 6))
        csp.add_variable("y", range(1, 6))
        csp.add_arithmetic({"x": 1, "y": 1}, '==', 5)
        csp.add_comparison("x", "y", '<')
        result, assignment = csp.solve_with_smt()
        assert result == CSPResult.SOLVED
        assert assignment["x"] + assignment["y"] == 5
        assert assignment["x"] < assignment["y"]

    def test_smt_alldiff(self):
        csp = CSPSolver()
        csp.add_variable("a", range(1, 4))
        csp.add_variable("b", range(1, 4))
        csp.add_variable("c", range(1, 4))
        csp.add_alldiff(["a", "b", "c"])
        result, assignment = csp.solve_with_smt()
        assert result == CSPResult.SOLVED
        assert len(set(assignment.values())) == 3

    def test_smt_unsatisfiable(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2])
        csp.add_variable("y", [1, 2])
        csp.add_comparison("x", "y", '>')
        csp.add_comparison("x", "y", '<')
        result, _ = csp.solve_with_smt()
        assert result == CSPResult.UNSATISFIABLE

    def test_smt_sum(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 6))
        csp.add_variable("y", range(1, 6))
        csp.add_variable("z", range(1, 6))
        csp.add_sum(["x", "y", "z"], 9)
        csp.add_alldiff(["x", "y", "z"])
        result, assignment = csp.solve_with_smt()
        assert result == CSPResult.SOLVED
        assert assignment["x"] + assignment["y"] + assignment["z"] == 9

    def test_smt_domain_with_gaps(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 3, 5, 7])
        csp.add_variable("y", [2, 4, 6, 8])
        csp.add_arithmetic({"x": 1, "y": 1}, '==', 9)
        result, assignment = csp.solve_with_smt()
        assert result == CSPResult.SOLVED
        assert assignment["x"] + assignment["y"] == 9
        assert assignment["x"] in [1, 3, 5, 7]
        assert assignment["y"] in [2, 4, 6, 8]


# ============================================================
# Sudoku Tests
# ============================================================

class TestSudoku:
    def _easy_puzzle(self):
        return [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

    def test_sudoku_creation(self):
        csp, decode = sudoku(self._easy_puzzle())
        assert len(csp.variables) == 81

    def test_sudoku_solve(self):
        csp, decode = sudoku(self._easy_puzzle())
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        grid = decode(assignment)
        # Check rows
        for r in range(9):
            assert sorted(grid[r]) == list(range(1, 10))
        # Check columns
        for c in range(9):
            assert sorted(grid[r][c] for r in range(9)) == list(range(1, 10))
        # Check boxes
        for br in range(3):
            for bc in range(3):
                vals = []
                for r in range(br*3, br*3+3):
                    for c in range(bc*3, bc*3+3):
                        vals.append(grid[r][c])
                assert sorted(vals) == list(range(1, 10))

    def test_sudoku_preserves_givens(self):
        puzzle = self._easy_puzzle()
        csp, decode = sudoku(puzzle)
        result, assignment = csp.solve()
        grid = decode(assignment)
        for r in range(9):
            for c in range(9):
                if puzzle[r][c] != 0:
                    assert grid[r][c] == puzzle[r][c]


# ============================================================
# N-Queens Tests
# ============================================================

class TestNQueens:
    def test_4_queens(self):
        csp, decode = n_queens(4)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        cols = decode(assignment)
        assert len(cols) == 4
        assert len(set(cols)) == 4
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(cols[i] - cols[j]) != abs(i - j)

    def test_8_queens(self):
        csp, decode = n_queens(8)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        cols = decode(assignment)
        assert len(set(cols)) == 8

    def test_3_queens_unsatisfiable(self):
        csp, decode = n_queens(3)
        result, _ = csp.solve()
        assert result == CSPResult.UNSATISFIABLE

    def test_1_queen(self):
        csp, decode = n_queens(1)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED

    def test_5_queens_all_solutions(self):
        csp, decode = n_queens(5)
        solutions = csp.solve_all()
        assert len(solutions) == 10  # 5-queens has exactly 10 solutions

    def test_4_queens_all_solutions(self):
        csp, decode = n_queens(4)
        solutions = csp.solve_all()
        assert len(solutions) == 2  # 4-queens has exactly 2 solutions


# ============================================================
# Graph Coloring Tests
# ============================================================

class TestGraphColoring:
    def test_triangle_3_colors(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        csp, decode = graph_coloring(edges, 3, 3)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        coloring = decode(assignment)
        for u, v in edges:
            assert coloring[u] != coloring[v]

    def test_triangle_2_colors_unsat(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        csp, decode = graph_coloring(edges, 3, 2)
        result, _ = csp.solve()
        assert result == CSPResult.UNSATISFIABLE

    def test_bipartite_2_colors(self):
        # K_{2,2} is bipartite
        edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
        csp, decode = graph_coloring(edges, 4, 2)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED

    def test_petersen_3_colors(self):
        # Petersen graph needs 3 colors
        edges = [
            (0,1),(1,2),(2,3),(3,4),(4,0),  # outer
            (0,5),(1,6),(2,7),(3,8),(4,9),  # spokes
            (5,7),(7,9),(9,6),(6,8),(8,5),  # inner
        ]
        csp, decode = graph_coloring(edges, 10, 3)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        coloring = decode(assignment)
        for u, v in edges:
            assert coloring[u] != coloring[v]

    def test_empty_graph(self):
        csp, decode = graph_coloring([], 3, 1)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED


# ============================================================
# Scheduling Tests
# ============================================================

class TestScheduling:
    def test_basic_scheduling(self):
        tasks = ["A", "B", "C"]
        csp, decode = scheduling(tasks, num_slots=5)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        schedule = decode(assignment)
        # No overlaps
        for t1 in tasks:
            for t2 in tasks:
                if t1 < t2:
                    s1, e1 = schedule[t1]
                    s2, e2 = schedule[t2]
                    assert e1 <= s2 or e2 <= s1

    def test_scheduling_with_precedences(self):
        tasks = ["A", "B", "C"]
        precs = [("A", "B"), ("B", "C")]
        csp, decode = scheduling(tasks, precedences=precs, num_slots=5)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        schedule = decode(assignment)
        assert schedule["A"][1] <= schedule["B"][0]
        assert schedule["B"][1] <= schedule["C"][0]

    def test_scheduling_with_durations(self):
        tasks = ["A", "B"]
        durations = {"A": 2, "B": 3}
        csp, decode = scheduling(tasks, durations=durations, num_slots=5)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        schedule = decode(assignment)
        assert schedule["A"][1] - schedule["A"][0] == 2
        assert schedule["B"][1] - schedule["B"][0] == 3

    def test_scheduling_with_deadlines(self):
        tasks = ["A", "B"]
        deadlines = {"A": 2}
        csp, decode = scheduling(tasks, deadlines=deadlines, num_slots=4)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        schedule = decode(assignment)
        assert schedule["A"][1] <= 2

    def test_tight_schedule(self):
        tasks = ["A", "B", "C"]
        csp, decode = scheduling(tasks, num_slots=3)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED

    def test_impossible_schedule(self):
        tasks = ["A", "B", "C", "D"]
        csp, decode = scheduling(tasks, num_slots=3)
        result, _ = csp.solve()
        assert result == CSPResult.UNSATISFIABLE


# ============================================================
# Magic Square Tests
# ============================================================

class TestMagicSquare:
    def test_3x3_magic_square(self):
        csp, decode = magic_square(3)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        grid = decode(assignment)
        magic = 15  # 3*(9+1)/2
        # Check rows
        for r in range(3):
            assert sum(grid[r]) == magic
        # Check columns
        for c in range(3):
            assert sum(grid[r][c] for r in range(3)) == magic
        # Check diagonals
        assert sum(grid[i][i] for i in range(3)) == magic
        assert sum(grid[i][2-i] for i in range(3)) == magic
        # All different values 1-9
        vals = [grid[r][c] for r in range(3) for c in range(3)]
        assert sorted(vals) == list(range(1, 10))


# ============================================================
# Latin Square Tests
# ============================================================

class TestLatinSquare:
    def test_3x3_latin(self):
        csp, decode = latin_square(3)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        grid = decode(assignment)
        for r in range(3):
            assert sorted(grid[r]) == [1, 2, 3]
        for c in range(3):
            assert sorted(grid[r][c] for r in range(3)) == [1, 2, 3]

    def test_4x4_latin(self):
        csp, decode = latin_square(4)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        grid = decode(assignment)
        for r in range(4):
            assert sorted(grid[r]) == [1, 2, 3, 4]

    def test_partial_latin(self):
        partial = {(0, 0): 1, (1, 1): 2}
        csp, decode = latin_square(3, partial=partial)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        grid = decode(assignment)
        assert grid[0][0] == 1
        assert grid[1][1] == 2

    def test_all_latin_3x3(self):
        csp, decode = latin_square(3)
        solutions = csp.solve_all()
        assert len(solutions) == 12  # 3x3 has 12 reduced forms


# ============================================================
# Knapsack Tests
# ============================================================

class TestKnapsack:
    def test_basic_knapsack(self):
        items = [(2, 3), (3, 4), (4, 5)]
        csp, decode = knapsack(items, capacity=6)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        info = decode(assignment)
        assert info['weight'] <= 6

    def test_knapsack_all_fit(self):
        items = [(1, 10), (1, 20), (1, 30)]
        csp, decode = knapsack(items, capacity=10)
        solutions = csp.solve_all()
        # Should include solution with all items selected
        all_selected = None
        for sol in solutions:
            info = decode(sol)
            if len(info['selected']) == 3:
                all_selected = info
        assert all_selected is not None
        assert all_selected['weight'] == 3

    def test_knapsack_nothing_fits(self):
        items = [(10, 5), (20, 10)]
        csp, decode = knapsack(items, capacity=5)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        info = decode(assignment)
        assert info['weight'] <= 5


# ============================================================
# Edge Cases and Stress Tests
# ============================================================

class TestEdgeCases:
    def test_single_variable_single_value(self):
        csp = CSPSolver()
        csp.add_variable("x", [42])
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] == 42

    def test_no_constraints(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [4, 5, 6])
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED

    def test_large_domain_alldiff(self):
        n = 8
        csp = CSPSolver()
        for i in range(n):
            csp.add_variable(f"x{i}", range(n))
        csp.add_alldiff([f"x{i}" for i in range(n)])
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert len(set(assignment.values())) == n

    def test_negative_domains(self):
        csp = CSPSolver()
        csp.add_variable("x", [-2, -1, 0, 1, 2])
        csp.add_variable("y", [-2, -1, 0, 1, 2])
        csp.add_arithmetic({"x": 1, "y": 1}, '==', 0)
        csp.add_inequality("x", "y")
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] + assignment["y"] == 0
        assert assignment["x"] != assignment["y"]

    def test_table_with_all_combos_disallowed(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2])
        csp.add_variable("y", [1, 2])
        # Only allow (1,1) -- but also require x != y
        csp.add_table(["x", "y"], [(1, 1)])
        csp.add_inequality("x", "y")
        result, _ = csp.solve()
        assert result == CSPResult.UNSATISFIABLE

    def test_callback_complex(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 10))
        csp.add_variable("y", range(1, 10))
        csp.add_variable("z", range(1, 10))
        csp.add_callback(["x", "y", "z"], lambda a: a["x"] ** 2 + a["y"] ** 2 == a["z"] ** 2)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] ** 2 + assignment["y"] ** 2 == assignment["z"] ** 2

    def test_multiple_alldiff_overlapping(self):
        csp = CSPSolver()
        for i in range(4):
            csp.add_variable(f"v{i}", [1, 2, 3, 4])
        csp.add_alldiff(["v0", "v1", "v2"])
        csp.add_alldiff(["v1", "v2", "v3"])
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert len({assignment["v0"], assignment["v1"], assignment["v2"]}) == 3
        assert len({assignment["v1"], assignment["v2"], assignment["v3"]}) == 3


class TestConstraintInteraction:
    def test_sum_and_alldiff(self):
        csp = CSPSolver()
        for i in range(3):
            csp.add_variable(f"x{i}", range(1, 7))
        csp.add_alldiff(["x0", "x1", "x2"])
        csp.add_sum(["x0", "x1", "x2"], 6)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        vals = [assignment[f"x{i}"] for i in range(3)]
        assert sorted(vals) == [1, 2, 3]

    def test_comparison_chain_tight(self):
        csp = CSPSolver()
        for i in range(5):
            csp.add_variable(f"x{i}", range(1, 6))
        for i in range(4):
            csp.add_comparison(f"x{i}", f"x{i+1}", '<')
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        for i in range(4):
            assert assignment[f"x{i}"] < assignment[f"x{i+1}"]

    def test_equality_and_sum(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 10))
        csp.add_variable("y", range(1, 10))
        csp.add_variable("z", range(1, 10))
        csp.add_equality("x", "y")
        csp.add_sum(["x", "y", "z"], 10)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment["x"] == assignment["y"]
        assert assignment["x"] + assignment["y"] + assignment["z"] == 10


# ============================================================
# Larger Problem Tests
# ============================================================

class TestLargerProblems:
    def test_map_coloring_australia(self):
        """Color Australian states with 3 colors."""
        csp = CSPSolver()
        states = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
        for s in states:
            csp.add_variable(s, [0, 1, 2])
        borders = [
            ('WA', 'NT'), ('WA', 'SA'), ('NT', 'SA'), ('NT', 'Q'),
            ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'), ('Q', 'NSW'), ('NSW', 'V')
        ]
        for a, b in borders:
            csp.add_inequality(a, b)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        for a, b in borders:
            assert assignment[a] != assignment[b]

    def test_send_more_money(self):
        """Classic cryptarithmetic puzzle: SEND + MORE = MONEY."""
        csp = CSPSolver()
        letters = ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']
        for l in letters:
            if l in ('S', 'M'):
                csp.add_variable(l, range(1, 10))  # Leading digits != 0
            else:
                csp.add_variable(l, range(0, 10))
        csp.add_alldiff(letters)

        # SEND + MORE = MONEY
        # 1000*S + 100*E + 10*N + D + 1000*M + 100*O + 10*R + E
        # = 10000*M + 1000*O + 100*N + 10*E + Y
        csp.add_callback(
            letters,
            lambda a: (1000*a['S'] + 100*a['E'] + 10*a['N'] + a['D'] +
                       1000*a['M'] + 100*a['O'] + 10*a['R'] + a['E'] ==
                       10000*a['M'] + 1000*a['O'] + 100*a['N'] + 10*a['E'] + a['Y'])
        )
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        send = 1000*assignment['S'] + 100*assignment['E'] + 10*assignment['N'] + assignment['D']
        more = 1000*assignment['M'] + 100*assignment['O'] + 10*assignment['R'] + assignment['E']
        money = 10000*assignment['M'] + 1000*assignment['O'] + 100*assignment['N'] + 10*assignment['E'] + assignment['Y']
        assert send + more == money

    def test_zebra_puzzle_simplified(self):
        """Simplified zebra-like puzzle with 3 houses."""
        csp = CSPSolver()
        # 3 houses, 3 colors, 3 nationalities
        colors = ['red', 'blue', 'green']
        nations = ['eng', 'spa', 'nor']

        for attr in colors + nations:
            csp.add_variable(attr, [1, 2, 3])

        csp.add_alldiff(colors)
        csp.add_alldiff(nations)

        # Englishman lives in red house
        csp.add_equality('eng', 'red')
        # Norwegian in house 1
        csp.variables['nor'].domain = {1}
        # Blue next to Norwegian
        csp.add_callback(['nor', 'blue'], lambda a: abs(a['nor'] - a['blue']) == 1)

        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert assignment['eng'] == assignment['red']
        assert assignment['nor'] == 1

    def test_six_variable_puzzle(self):
        """6 variables, mixed constraints."""
        csp = CSPSolver()
        for i in range(6):
            csp.add_variable(f"v{i}", range(1, 7))
        csp.add_alldiff([f"v{i}" for i in range(6)])
        csp.add_comparison("v0", "v1", '<')
        csp.add_comparison("v2", "v3", '>')
        csp.add_sum(["v0", "v1", "v2"], 12)
        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        vals = [assignment[f"v{i}"] for i in range(6)]
        assert len(set(vals)) == 6
        assert assignment["v0"] < assignment["v1"]
        assert assignment["v2"] > assignment["v3"]
        assert assignment["v0"] + assignment["v1"] + assignment["v2"] == 12


# ============================================================
# Solver Method Comparison
# ============================================================

class TestSolverComparison:
    def _make_coloring_csp(self):
        csp = CSPSolver()
        for i in range(5):
            csp.add_variable(f"n{i}", [0, 1, 2])
        edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in edges:
            csp.add_inequality(f"n{u}", f"n{v}")
        return csp

    def test_csp_vs_sat_same_result(self):
        csp = self._make_coloring_csp()
        r1, a1 = csp.solve()

        csp2 = self._make_coloring_csp()
        r2, a2 = csp2.solve_with_sat()

        assert r1 == CSPResult.SOLVED
        assert r2 == CSPResult.SOLVED
        # Both should produce valid colorings
        edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in edges:
            assert a1[f"n{u}"] != a1[f"n{v}"]
            assert a2[f"n{u}"] != a2[f"n{v}"]

    def test_csp_vs_smt_same_result(self):
        csp = CSPSolver()
        csp.add_variable("x", range(1, 6))
        csp.add_variable("y", range(1, 6))
        csp.add_arithmetic({"x": 1, "y": 1}, '==', 7)
        csp.add_comparison("x", "y", '<')

        r1, a1 = csp.solve()
        assert r1 == CSPResult.SOLVED

        csp2 = CSPSolver()
        csp2.add_variable("x", range(1, 6))
        csp2.add_variable("y", range(1, 6))
        csp2.add_arithmetic({"x": 1, "y": 1}, '==', 7)
        csp2.add_comparison("x", "y", '<')
        r2, a2 = csp2.solve_with_smt()
        assert r2 == CSPResult.SOLVED
        assert a2["x"] + a2["y"] == 7


# ============================================================
# Propagation and Heuristic Tests
# ============================================================

class TestPropagation:
    def test_mrv_selects_smallest_domain(self):
        csp = CSPSolver()
        csp.add_variable("a", [1])
        csp.add_variable("b", [1, 2])
        csp.add_variable("c", [1, 2, 3])
        domains = {"a": {1}, "b": {1, 2}, "c": {1, 2, 3}}
        selected = csp._select_variable({}, domains)
        assert selected == "a"

    def test_mrv_with_assigned(self):
        csp = CSPSolver()
        csp.add_variable("a", [1])
        csp.add_variable("b", [1, 2])
        csp.add_variable("c", [1, 2, 3])
        domains = {"a": {1}, "b": {1, 2}, "c": {1, 2, 3}}
        selected = csp._select_variable({"a": 1}, domains)
        assert selected == "b"

    def test_forward_check_prunes(self):
        csp = CSPSolver()
        csp.add_variable("x", [1, 2])
        csp.add_variable("y", [1, 2])
        csp.add_inequality("x", "y")
        domains = {"x": {1}, "y": {1, 2}}
        ok = csp._forward_check("x", 1, domains)
        assert ok
        assert domains["y"] == {2}


# ============================================================
# Regression / Integration Tests
# ============================================================

class TestIntegration:
    def test_solve_then_verify(self):
        """Solve a problem, then verify all constraints hold."""
        csp = CSPSolver()
        for i in range(4):
            csp.add_variable(f"x{i}", range(1, 5))
        csp.add_alldiff([f"x{i}" for i in range(4)])
        csp.add_comparison("x0", "x3", '<')
        csp.add_sum(["x0", "x1"], 5)

        result, assignment = csp.solve()
        assert result == CSPResult.SOLVED
        assert csp._is_consistent(assignment)

    def test_incremental_constraints(self):
        """Add constraints incrementally and solve."""
        csp = CSPSolver()
        csp.add_variable("x", range(1, 6))
        csp.add_variable("y", range(1, 6))

        # First: just inequality
        csp.add_inequality("x", "y")
        r1, a1 = csp.solve()
        assert r1 == CSPResult.SOLVED

        # Add more constraints
        csp.add_comparison("x", "y", '<')
        r2, a2 = csp.solve()
        assert r2 == CSPResult.SOLVED
        assert a2["x"] < a2["y"]

    def test_solve_all_count(self):
        """Verify exact solution count for a well-known problem."""
        # 4x4 Latin square: 576 solutions
        # Too slow for full enumeration, just verify > 0
        csp, decode = latin_square(3)
        solutions = csp.solve_all()
        assert len(solutions) == 12

    def test_multiple_solves(self):
        """Solve the same CSP multiple times gives consistent results."""
        csp = CSPSolver()
        csp.add_variable("x", [1, 2, 3])
        csp.add_variable("y", [1, 2, 3])
        csp.add_inequality("x", "y")

        results = set()
        for _ in range(5):
            r, a = csp.solve()
            assert r == CSPResult.SOLVED
            results.add((a["x"], a["y"]))
        # Should always get the same result (deterministic)
        assert len(results) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
