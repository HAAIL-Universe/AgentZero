"""Tests for C035: DPLL SAT Solver"""
import pytest
from sat_solver import (
    Literal, Clause, Solver, SolverResult, Assignment, LBool,
    parse_dimacs, solve_dimacs, solve_clauses,
    generate_random_3sat, generate_pigeonhole, generate_queens,
    encode_sudoku, encode_graph_coloring,
)


# === Literal Tests ===

class TestLiteral:
    def test_positive_literal(self):
        l = Literal(1)
        assert l.var == 1
        assert l.neg is False
        assert int(l) == 1

    def test_negative_literal(self):
        l = Literal(3, True)
        assert l.var == 3
        assert l.neg is True
        assert int(l) == -3

    def test_negate(self):
        l = Literal(5)
        n = -l
        assert n.var == 5
        assert n.neg is True
        assert -n == l

    def test_from_int_positive(self):
        l = Literal.from_int(4)
        assert l.var == 4
        assert l.neg is False

    def test_from_int_negative(self):
        l = Literal.from_int(-7)
        assert l.var == 7
        assert l.neg is True

    def test_from_int_zero_raises(self):
        with pytest.raises(ValueError):
            Literal.from_int(0)

    def test_invalid_var_raises(self):
        with pytest.raises(ValueError):
            Literal(0)
        with pytest.raises(ValueError):
            Literal(-1)

    def test_equality(self):
        assert Literal(1) == Literal(1)
        assert Literal(1, True) == Literal(1, True)
        assert Literal(1) != Literal(1, True)
        assert Literal(1) != Literal(2)

    def test_hash(self):
        s = {Literal(1), Literal(1, True), Literal(2)}
        assert len(s) == 3
        assert Literal(1) in s

    def test_repr(self):
        assert str(Literal(3)) == "3"
        assert str(Literal(3, True)) == "-3"


class TestClause:
    def test_empty_clause(self):
        c = Clause([])
        assert c.is_empty()
        assert not c.is_unit()
        assert len(c) == 0

    def test_unit_clause(self):
        c = Clause([Literal(1)])
        assert c.is_unit()
        assert not c.is_empty()
        assert len(c) == 1

    def test_multi_literal(self):
        c = Clause([Literal(1), Literal(2, True), Literal(3)])
        assert len(c) == 3
        assert not c.is_unit()

    def test_iteration(self):
        lits = [Literal(1), Literal(2)]
        c = Clause(lits)
        assert list(c) == lits

    def test_indexing(self):
        c = Clause([Literal(1), Literal(2)])
        assert c[0] == Literal(1)
        assert c[1] == Literal(2)


class TestAssignment:
    def test_assign_and_query(self):
        a = Assignment()
        a.assign(1, True, 0)
        assert a.is_assigned(1)
        assert a.value[1] is True
        assert a.level[1] == 0

    def test_lit_value_true(self):
        a = Assignment()
        a.assign(1, True, 0)
        assert a.lit_value(Literal(1)) == LBool.TRUE
        assert a.lit_value(Literal(1, True)) == LBool.FALSE

    def test_lit_value_false(self):
        a = Assignment()
        a.assign(2, False, 0)
        assert a.lit_value(Literal(2)) == LBool.FALSE
        assert a.lit_value(Literal(2, True)) == LBool.TRUE

    def test_lit_value_undef(self):
        a = Assignment()
        assert a.lit_value(Literal(1)) == LBool.UNDEF

    def test_backtrack(self):
        a = Assignment()
        a.assign(1, True, 0)
        a.new_decision_level()
        a.assign(2, False, 1)
        a.assign(3, True, 1)
        assert a.current_level() == 1
        a.backtrack_to(0)
        assert a.current_level() == 0
        assert a.is_assigned(1)
        assert not a.is_assigned(2)
        assert not a.is_assigned(3)

    def test_trail_order(self):
        a = Assignment()
        a.assign(3, True, 0)
        a.assign(1, False, 0)
        assert a.trail == [3, 1]

    def test_multi_level_backtrack(self):
        a = Assignment()
        a.assign(1, True, 0)
        a.new_decision_level()
        a.assign(2, True, 1)
        a.new_decision_level()
        a.assign(3, True, 2)
        a.backtrack_to(0)
        assert a.is_assigned(1)
        assert not a.is_assigned(2)
        assert not a.is_assigned(3)
        assert a.current_level() == 0


# === Basic SAT Tests ===

class TestSolverBasic:
    def test_empty_formula(self):
        s = Solver()
        assert s.solve() == SolverResult.SAT

    def test_single_positive_clause(self):
        s = Solver()
        s.add_clause([1])
        result = s.solve()
        assert result == SolverResult.SAT
        assert s.model()[1] is True

    def test_single_negative_clause(self):
        s = Solver()
        s.add_clause([-1])
        result = s.solve()
        assert result == SolverResult.SAT
        assert s.model()[1] is False

    def test_two_unit_clauses_consistent(self):
        s = Solver()
        s.add_clause([1])
        s.add_clause([2])
        result = s.solve()
        assert result == SolverResult.SAT
        assert s.model()[1] is True
        assert s.model()[2] is True

    def test_contradictory_unit_clauses(self):
        s = Solver()
        s.add_clause([1])
        s.add_clause([-1])
        result = s.solve()
        assert result == SolverResult.UNSAT

    def test_simple_sat(self):
        # (x1 v x2) ^ (-x1 v x2)
        s = Solver()
        s.add_clause([1, 2])
        s.add_clause([-1, 2])
        result = s.solve()
        assert result == SolverResult.SAT
        m = s.model()
        assert s.verify(m)

    def test_simple_unsat(self):
        # (x1) ^ (-x1) ^ (x2) ^ (-x2)
        s = Solver()
        s.add_clause([1])
        s.add_clause([-1])
        result = s.solve()
        assert result == SolverResult.UNSAT

    def test_tautological_clause_ignored(self):
        s = Solver()
        # (x1 v -x1) is tautology
        idx = s.add_clause([1, -1])
        assert idx is None  # tautology returns None
        s.add_clause([2])
        result = s.solve()
        assert result == SolverResult.SAT

    def test_empty_clause_unsat(self):
        s = Solver()
        s.add_clause([])
        assert s.solve() == SolverResult.UNSAT

    def test_duplicate_literals_in_clause(self):
        s = Solver()
        s.add_clause([1, 1, 2])  # should normalize to [1, 2]
        result = s.solve()
        assert result == SolverResult.SAT

    def test_verify_correct(self):
        s = Solver()
        s.add_clause([1, 2])
        s.add_clause([-1, 2])
        s.solve()
        assert s.verify()

    def test_verify_wrong_assignment(self):
        s = Solver()
        s.add_clause([1])
        s.add_clause([2])
        s.solve()
        assert not s.verify({1: False, 2: True})


# === Unit Propagation Tests ===

class TestUnitPropagation:
    def test_chain_propagation(self):
        # x1=T, (x1 -> x2), (x2 -> x3) => x3=T
        s = Solver()
        s.add_clause([1])
        s.add_clause([-1, 2])
        s.add_clause([-2, 3])
        result = s.solve()
        assert result == SolverResult.SAT
        m = s.model()
        assert m[1] is True
        assert m[2] is True
        assert m[3] is True

    def test_propagation_detects_conflict(self):
        # x1=T, (x1 -> x2), (x1 -> -x2)
        s = Solver()
        s.add_clause([1])
        s.add_clause([-1, 2])
        s.add_clause([-1, -2])
        result = s.solve()
        assert result == SolverResult.UNSAT

    def test_long_propagation_chain(self):
        s = Solver()
        s.add_clause([1])
        for i in range(1, 10):
            s.add_clause([-(i), i + 1])
        result = s.solve()
        assert result == SolverResult.SAT
        m = s.model()
        for i in range(1, 11):
            assert m[i] is True


# === CDCL Tests ===

class TestCDCL:
    def test_learns_from_conflict(self):
        # (x1 v x2) ^ (x1 v -x2) ^ (-x1 v x3) ^ (-x1 v -x3)
        # x1=T forces x3 and -x3 (contradiction). x1=F forces x2 and -x2.
        # Actually UNSAT.
        s = Solver()
        s.add_clause([1, 2])
        s.add_clause([1, -2])
        s.add_clause([-1, 3])
        s.add_clause([-1, -3])
        result = s.solve()
        assert result == SolverResult.UNSAT

    def test_learns_unit_clause(self):
        # (x1 v x2) ^ (-x2 v x3) ^ (-x3) -- forces x2=F via chain, then x1=T
        s = Solver()
        s.add_clause([1, 2])
        s.add_clause([-2, 3])
        s.add_clause([-3])
        result = s.solve()
        assert result == SolverResult.SAT
        m = s.model()
        assert m[3] is False
        assert m[2] is False
        assert m[1] is True

    def test_conflict_at_level_zero(self):
        s = Solver()
        s.add_clause([1])
        s.add_clause([-1])
        result = s.solve()
        assert result == SolverResult.UNSAT

    def test_learned_clause_count(self):
        # A formula where conflicts generate learned clauses
        s = Solver()
        # Craft a formula that forces backtracking
        s.add_clause([1, 2, 3])
        s.add_clause([-1, -2])
        s.add_clause([-1, -3])
        s.add_clause([-2, -3])
        s.add_clause([1, 2])
        result = s.solve()
        assert result == SolverResult.SAT
        assert s.verify()

    def test_backtrack_multiple_levels(self):
        # (x1vx2) ^ (-x1vx3) ^ (-x2vx3) ^ (-x3vx4) ^ (-x3v-x4) ^ (x3vx5)
        # x3=T forces x4 and -x4 (contradiction).
        # x3=F forces x1=F, x2=F, then x1vx2 fails.
        # UNSAT.
        s = Solver()
        s.add_clause([1, 2])
        s.add_clause([-1, 3])
        s.add_clause([-2, 3])
        s.add_clause([-3, 4])
        s.add_clause([-3, -4])
        s.add_clause([3, 5])
        result = s.solve()
        assert result == SolverResult.UNSAT

    def test_actual_backtracking(self):
        # A formula that requires real backtracking but is SAT
        # (x1 v x2) ^ (-x1 v x3) ^ (-x2 v x3) ^ (-x3 v x4) ^ (x1 v -x4)
        s = Solver()
        s.add_clause([1, 2])
        s.add_clause([-1, 3])
        s.add_clause([-2, 3])
        s.add_clause([-3, 4])
        s.add_clause([1, -4])
        result = s.solve()
        assert result == SolverResult.SAT
        assert s.verify()


# === DIMACS Parser Tests ===

class TestDIMACS:
    def test_parse_simple(self):
        dimacs = """
c This is a comment
p cnf 3 2
1 2 0
-1 3 0
"""
        s = parse_dimacs(dimacs)
        assert s.num_vars == 3
        assert len(s.clauses) == 2

    def test_parse_and_solve(self):
        dimacs = """
p cnf 2 3
1 2 0
-1 2 0
1 -2 0
"""
        result, model, stats = solve_dimacs(dimacs)
        assert result == SolverResult.SAT
        assert model is not None

    def test_parse_unsat(self):
        dimacs = """
p cnf 1 2
1 0
-1 0
"""
        result, model, stats = solve_dimacs(dimacs)
        assert result == SolverResult.UNSAT
        assert model is None

    def test_parse_no_header(self):
        dimacs = "1 2 0\n-1 -2 0\n"
        s = parse_dimacs(dimacs)
        assert len(s.clauses) == 2

    def test_solve_dimacs_stats(self):
        dimacs = "p cnf 3 3\n1 2 3 0\n-1 -2 0\n-2 -3 0\n"
        result, model, stats = solve_dimacs(dimacs)
        assert result == SolverResult.SAT
        assert stats.propagations >= 0


# === solve_clauses Tests ===

class TestSolveClauses:
    def test_basic(self):
        result, model, stats = solve_clauses([[1, 2], [-1, 2]])
        assert result == SolverResult.SAT
        assert model[2] is True

    def test_with_num_vars(self):
        result, model, stats = solve_clauses([[1]], num_vars=5)
        assert result == SolverResult.SAT
        assert model[1] is True

    def test_unsat(self):
        result, model, stats = solve_clauses([[1], [-1]])
        assert result == SolverResult.UNSAT


# === Random 3-SAT Tests ===

class TestRandom3SAT:
    def test_generate_structure(self):
        clauses = generate_random_3sat(10, 20, seed=42)
        assert len(clauses) == 20
        for c in clauses:
            assert len(c) == 3
            for lit in c:
                assert 1 <= abs(lit) <= 10

    def test_deterministic_with_seed(self):
        c1 = generate_random_3sat(5, 10, seed=123)
        c2 = generate_random_3sat(5, 10, seed=123)
        assert c1 == c2

    def test_solvable_underconstrained(self):
        # ratio ~2 is usually SAT
        clauses = generate_random_3sat(20, 40, seed=1)
        result, model, _ = solve_clauses(clauses, num_vars=20)
        assert result == SolverResult.SAT
        # Verify solution
        for clause in clauses:
            satisfied = False
            for lit in clause:
                var = abs(lit)
                val = model.get(var, False)
                if lit < 0:
                    val = not val
                if val:
                    satisfied = True
                    break
            assert satisfied

    def test_solve_random_batch(self):
        # Solve several random instances
        sat_count = 0
        for seed in range(10):
            clauses = generate_random_3sat(8, 20, seed=seed)
            result, model, _ = solve_clauses(clauses, num_vars=8)
            if result == SolverResult.SAT:
                sat_count += 1
                # Verify each SAT result
                s = Solver()
                s._ensure_var(8)
                for c in clauses:
                    s.add_clause(c)
                assert s.verify(model)
        # At this ratio, most should be SAT
        assert sat_count > 0


# === Pigeonhole Tests ===

class TestPigeonhole:
    def test_php_2_unsat(self):
        # 3 pigeons, 2 holes
        clauses = generate_pigeonhole(2)
        result, model, stats = solve_clauses(clauses)
        assert result == SolverResult.UNSAT

    def test_php_3_unsat(self):
        # 4 pigeons, 3 holes
        clauses = generate_pigeonhole(3)
        result, model, stats = solve_clauses(clauses)
        assert result == SolverResult.UNSAT

    def test_php_structure(self):
        clauses = generate_pigeonhole(2)
        # 3 pigeons, 2 holes
        # 3 "at least one hole" clauses + 3*2 "no sharing" clauses = 9
        pigeon_clauses = [c for c in clauses if all(l > 0 for l in c)]
        assert len(pigeon_clauses) == 3  # 3 pigeons


# === N-Queens Tests ===

class TestNQueens:
    def test_4queens_sat(self):
        clauses = generate_queens(4)
        result, model, _ = solve_clauses(clauses, num_vars=16)
        assert result == SolverResult.SAT
        # Verify: exactly one queen per row
        for row in range(1, 5):
            queens_in_row = sum(
                1 for col in range(1, 5)
                if model.get((row - 1) * 4 + col, False)
            )
            assert queens_in_row == 1

    def test_5queens_sat(self):
        clauses = generate_queens(5)
        result, model, _ = solve_clauses(clauses, num_vars=25)
        assert result == SolverResult.SAT

    def test_queens_no_conflicts(self):
        """Verify no two queens attack each other."""
        n = 5
        clauses = generate_queens(n)
        result, model, _ = solve_clauses(clauses, num_vars=n * n)
        assert result == SolverResult.SAT

        # Extract queen positions
        queens = []
        for r in range(1, n + 1):
            for c in range(1, n + 1):
                if model.get((r - 1) * n + c, False):
                    queens.append((r, c))

        # Check no attacks
        for i, (r1, c1) in enumerate(queens):
            for j, (r2, c2) in enumerate(queens):
                if i < j:
                    assert r1 != r2, f"Same row: {(r1,c1)} and {(r2,c2)}"
                    assert c1 != c2, f"Same col: {(r1,c1)} and {(r2,c2)}"
                    assert abs(r1 - r2) != abs(c1 - c2), \
                        f"Same diag: {(r1,c1)} and {(r2,c2)}"


# === Sudoku Tests ===

class TestSudoku:
    def test_trivial_sudoku(self):
        """Almost complete sudoku -- just one cell empty."""
        grid = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 0],  # last cell empty
        ]
        clauses, decode = encode_sudoku(grid)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        solution = decode(model)
        assert solution[8][8] == 9

    def test_sudoku_two_empty(self):
        """Sudoku with only 2 empty cells -- should solve by propagation."""
        grid = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 0],  # missing 5
            [3, 4, 5, 2, 8, 6, 1, 0, 9],  # missing 7
        ]
        clauses, decode = encode_sudoku(grid)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        solution = decode(model)
        assert solution[7][8] == 5
        assert solution[8][7] == 7

    def test_sudoku_encoding_structure(self):
        """Verify the encoding generates the right number of clauses."""
        grid = [[0]*9 for _ in range(9)]
        clauses, decode = encode_sudoku(grid)
        # Each cell: at-least-one (81) + at-most-one (81*36) + row (81) + col (81) + box (81)
        # Total > 3000
        assert len(clauses) > 3000


# === Graph Coloring Tests ===

class TestGraphColoring:
    def test_triangle_3color(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        clauses, decode = encode_graph_coloring(edges, 3, 3)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        coloring = decode(model)
        for u, v in edges:
            assert coloring[u] != coloring[v]

    def test_triangle_2color_unsat(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        clauses, decode = encode_graph_coloring(edges, 3, 2)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.UNSAT

    def test_bipartite_2color(self):
        # K_{2,2} is bipartite
        edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
        clauses, decode = encode_graph_coloring(edges, 4, 2)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        coloring = decode(model)
        for u, v in edges:
            assert coloring[u] != coloring[v]

    def test_k4_3color_unsat(self):
        # K4 needs 4 colors
        edges = [(i, j) for i in range(4) for j in range(i+1, 4)]
        clauses, decode = encode_graph_coloring(edges, 4, 3)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.UNSAT

    def test_k4_4color_sat(self):
        edges = [(i, j) for i in range(4) for j in range(i+1, 4)]
        clauses, decode = encode_graph_coloring(edges, 4, 4)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        coloring = decode(model)
        for u, v in edges:
            assert coloring[u] != coloring[v]

    def test_petersen_3color(self):
        """Petersen graph is 3-colorable."""
        edges = [
            (0,1),(1,2),(2,3),(3,4),(4,0),  # outer cycle
            (0,5),(1,6),(2,7),(3,8),(4,9),  # spokes
            (5,7),(7,9),(9,6),(6,8),(8,5),  # inner pentagram
        ]
        clauses, decode = encode_graph_coloring(edges, 10, 3)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        coloring = decode(model)
        for u, v in edges:
            assert coloring[u] != coloring[v]


# === Stats Tests ===

class TestStats:
    def test_stats_tracked(self):
        s = Solver()
        s.add_clause([1, 2, 3])
        s.add_clause([-1, -2])
        s.add_clause([-2, -3])
        s.add_clause([-1, -3])
        s.solve()
        assert s.stats.propagations >= 0
        assert s.stats.decisions >= 0

    def test_no_decisions_for_unit(self):
        s = Solver()
        s.add_clause([1])
        s.add_clause([-1, 2])
        s.solve()
        # Should be solved purely by propagation
        assert s.stats.decisions == 0


# === Edge Cases ===

class TestEdgeCases:
    def test_single_var_sat(self):
        result, model, _ = solve_clauses([[1]])
        assert result == SolverResult.SAT

    def test_single_var_neg(self):
        result, model, _ = solve_clauses([[-1]])
        assert result == SolverResult.SAT
        assert model[1] is False

    def test_many_vars_chain(self):
        """x1 -> x2 -> ... -> x20, x1=T"""
        clauses = [[1]]
        for i in range(1, 20):
            clauses.append([-i, i + 1])
        result, model, _ = solve_clauses(clauses, num_vars=20)
        assert result == SolverResult.SAT
        for i in range(1, 21):
            assert model[i] is True

    def test_many_vars_contradiction(self):
        """x1 -> x2 -> ... -> x10, x1=T, x10=F"""
        clauses = [[1], [-10]]
        for i in range(1, 10):
            clauses.append([-i, i + 1])
        result, model, _ = solve_clauses(clauses, num_vars=10)
        assert result == SolverResult.UNSAT

    def test_all_positive_sat(self):
        """Every clause has only positive literals."""
        clauses = [[1, 2], [2, 3], [3, 4]]
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT

    def test_all_negative_sat(self):
        """Every clause has only negative literals."""
        clauses = [[-1, -2], [-2, -3], [-3, -4]]
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT

    def test_xor_encoding(self):
        """x1 XOR x2 = (x1 v x2) ^ (-x1 v -x2)"""
        result, model, _ = solve_clauses([[1, 2], [-1, -2]])
        assert result == SolverResult.SAT
        assert model[1] != model[2]

    def test_at_most_one(self):
        """At most one of x1, x2, x3 is true, and at least one."""
        clauses = [
            [1, 2, 3],     # at least one
            [-1, -2],      # not both 1,2
            [-1, -3],      # not both 1,3
            [-2, -3],      # not both 2,3
        ]
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        true_count = sum(1 for v in [1, 2, 3] if model.get(v, False))
        assert true_count == 1

    def test_large_clause(self):
        """A single clause with 50 literals."""
        clause = list(range(1, 51))
        result, model, _ = solve_clauses([clause])
        assert result == SolverResult.SAT

    def test_many_small_clauses(self):
        """100 binary clauses."""
        clauses = []
        for i in range(1, 51):
            clauses.append([i, i + 50])
        result, model, _ = solve_clauses(clauses, num_vars=100)
        assert result == SolverResult.SAT
        assert len(model) == 100

    def test_verify_rejects_partial(self):
        """Verify rejects assignment that doesn't cover all vars."""
        s = Solver()
        s.add_clause([1, 2])
        s.add_clause([3])
        s.solve()
        # Partial assignment missing var 3
        assert not s.verify({1: True, 2: True})


# === Harder Combinatorial Tests ===

class TestHarder:
    def test_latin_square_3x3(self):
        """3x3 Latin square as SAT."""
        n = 3
        clauses = []

        def var(r, c, v):
            return r * n * n + c * n + v + 1

        # Each cell has at least one value
        for r in range(n):
            for c in range(n):
                clauses.append([var(r, c, v) for v in range(n)])

        # Each cell at most one value
        for r in range(n):
            for c in range(n):
                for v1 in range(n):
                    for v2 in range(v1 + 1, n):
                        clauses.append([-var(r, c, v1), -var(r, c, v2)])

        # Each row has each value
        for r in range(n):
            for v in range(n):
                clauses.append([var(r, c, v) for c in range(n)])

        # Each col has each value
        for c in range(n):
            for v in range(n):
                clauses.append([var(r, c, v) for r in range(n)])

        result, model, _ = solve_clauses(clauses, num_vars=n**3)
        assert result == SolverResult.SAT

    def test_php_4_unsat(self):
        """5 pigeons, 4 holes -- harder UNSAT."""
        clauses = generate_pigeonhole(4)
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.UNSAT

    def test_5queens_harder(self):
        """5-queens with verification of all constraints."""
        n = 5
        clauses = generate_queens(n)
        result, model, _ = solve_clauses(clauses, num_vars=n*n)
        assert result == SolverResult.SAT
        queens = []
        for r in range(1, n+1):
            for c in range(1, n+1):
                if model.get((r-1)*n + c, False):
                    queens.append((r, c))
        assert len(queens) == n

    def test_random_3sat_at_threshold(self):
        """Near the phase transition (ratio ~4.27)."""
        clauses = generate_random_3sat(15, 64, seed=7)
        result, model, stats = solve_clauses(clauses, num_vars=15)
        # May be SAT or UNSAT, but should terminate
        assert result in (SolverResult.SAT, SolverResult.UNSAT)
        if result == SolverResult.SAT:
            # Verify
            for clause in clauses:
                sat = any(
                    (model.get(abs(l), False) if l > 0 else not model.get(abs(l), False))
                    for l in clause
                )
                assert sat


# === Integration Tests ===

class TestIntegration:
    def test_solve_then_add_negation(self):
        """Find one solution, block it, find another."""
        clauses = [[1, 2], [-1, 2], [1, -2]]
        result, model, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT

        # Block this solution
        blocking = []
        for v in sorted(model.keys()):
            if model[v]:
                blocking.append(-v)
            else:
                blocking.append(v)
        clauses.append(blocking)

        result2, model2, _ = solve_clauses(clauses)
        if result2 == SolverResult.SAT:
            assert model2 != model

    def test_incremental_unsat(self):
        """Start SAT, add clauses until UNSAT."""
        clauses = [[1, 2]]
        result, _, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT

        clauses.append([-1])
        result, m, _ = solve_clauses(clauses)
        assert result == SolverResult.SAT
        assert m[2] is True

        clauses.append([-2])
        result, _, _ = solve_clauses(clauses)
        assert result == SolverResult.UNSAT

    def test_all_solutions_small(self):
        """Enumerate all solutions of a small formula."""
        base_clauses = [[1, 2], [-1, -2]]  # XOR: exactly 2 solutions
        solutions = []
        clauses = list(base_clauses)

        for _ in range(10):  # safety limit
            result, model, _ = solve_clauses(clauses)
            if result == SolverResult.UNSAT:
                break
            solutions.append(model)
            blocking = []
            for v in sorted(model.keys()):
                blocking.append(-v if model[v] else v)
            clauses.append(blocking)

        assert len(solutions) == 2

    def test_dimacs_round_trip(self):
        """Generate clauses, format as DIMACS, parse back, solve."""
        original = [[1, 2, 3], [-1, -2], [-2, -3], [1, 3]]
        dimacs_lines = [f"p cnf 3 {len(original)}"]
        for c in original:
            dimacs_lines.append(" ".join(str(l) for l in c) + " 0")
        dimacs = "\n".join(dimacs_lines)

        result, model, _ = solve_dimacs(dimacs)
        assert result == SolverResult.SAT
        assert model is not None
