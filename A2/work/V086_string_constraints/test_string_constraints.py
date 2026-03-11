"""Tests for V086: String Constraint Solver."""

import pytest
from string_constraints import (
    StringConstraintSolver, StringConstraint, ConstraintKind, SolveResult,
    StringSolution, StringVar,
    # Constraint constructors
    str_var, regex_constraint, equals_const, not_equals_const,
    equals_var, not_equals_var, concat_eq, length_eq, length_le, length_ge,
    length_range, contains, prefix, suffix, char_at, in_set, not_empty,
    # High-level API
    solve_constraints, check_regex_membership, check_word_equation,
    find_string_matching, check_string_disjointness, enumerate_solutions,
    check_implication, string_solver_stats,
)

ALPHA = "abcdefghijklmnopqrstuvwxyz"
ALPHA_NUM = "abcdefghijklmnopqrstuvwxyz0123456789"


# ===== Basic constraint creation =====

class TestConstraintCreation:
    def test_str_var(self):
        v = str_var("x")
        assert v.name == "x"
        assert repr(v) == "x"

    def test_regex_constraint(self):
        c = regex_constraint("x", "a*b")
        assert c.kind == ConstraintKind.REGEX
        assert c.var == "x"
        assert c.pattern == "a*b"

    def test_equals_const(self):
        c = equals_const("x", "hello")
        assert c.kind == ConstraintKind.EQUALS_CONST
        assert c.pattern == "hello"

    def test_concat_eq(self):
        c = concat_eq("x", "y", "z")
        assert c.kind == ConstraintKind.CONCAT
        assert c.var == "x"
        assert c.var2 == "y"
        assert c.var3 == "z"

    def test_length_eq(self):
        c = length_eq("x", 5)
        assert c.kind == ConstraintKind.LENGTH_EQ
        assert c.value == 5

    def test_length_range(self):
        c = length_range("x", 2, 5)
        assert c.kind == ConstraintKind.LENGTH_RANGE
        assert c.value == 2
        assert c.value2 == 5

    def test_in_set(self):
        c = in_set("x", ["a", "b", "c"])
        assert c.kind == ConstraintKind.IN_SET
        assert c.strings == ("a", "b", "c")


# ===== Simple regex constraints =====

class TestRegexConstraints:
    def test_simple_regex(self):
        result = solve_constraints([regex_constraint("x", "ab")], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "ab"

    def test_star_regex(self):
        result = solve_constraints([regex_constraint("x", "a*")], ALPHA)
        assert result.result == SolveResult.SAT
        # a* includes empty string
        assert all(c == 'a' for c in result.assignment["x"])

    def test_alternation_regex(self):
        result = solve_constraints([regex_constraint("x", "a|b")], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] in ("a", "b")

    def test_char_class_regex(self):
        result = solve_constraints([regex_constraint("x", "[abc]")], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] in ("a", "b", "c")

    def test_regex_with_length(self):
        result = solve_constraints([
            regex_constraint("x", "a*"),
            length_eq("x", 3),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "aaa"

    def test_regex_with_length_unsat(self):
        result = solve_constraints([
            regex_constraint("x", "ab"),
            length_eq("x", 5),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_two_regex_intersection(self):
        result = solve_constraints([
            regex_constraint("x", "a*b"),
            regex_constraint("x", "(a|b)*"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        x = result.assignment["x"]
        assert x.endswith("b")
        assert all(c in "ab" for c in x)

    def test_empty_regex(self):
        # Pattern that matches nothing via intersection
        result = solve_constraints([
            regex_constraint("x", "a"),
            regex_constraint("x", "b"),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_complex_regex(self):
        result = solve_constraints([
            regex_constraint("x", "(ab)+"),
            length_le("x", 4),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] in ("ab", "abab")

    def test_dot_regex(self):
        result = solve_constraints([
            regex_constraint("x", "."),
            length_eq("x", 1),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert len(result.assignment["x"]) == 1


# ===== Equality constraints =====

class TestEqualityConstraints:
    def test_equals_const_sat(self):
        result = solve_constraints([equals_const("x", "hello")], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "hello"

    def test_equals_const_with_regex(self):
        result = solve_constraints([
            equals_const("x", "abc"),
            regex_constraint("x", "a.*"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "abc"

    def test_equals_const_conflict(self):
        result = solve_constraints([
            equals_const("x", "abc"),
            equals_const("x", "def"),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_not_equals_const(self):
        result = solve_constraints([
            regex_constraint("x", "a|b|c"),
            not_equals_const("x", "a"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] in ("b", "c")

    def test_not_equals_const_all_excluded(self):
        result = solve_constraints([
            in_set("x", ["a", "b"]),
            not_equals_const("x", "a"),
            not_equals_const("x", "b"),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT


# ===== Variable equality =====

class TestVarEquality:
    def test_equals_var(self):
        result = solve_constraints([
            regex_constraint("x", "a|b"),
            regex_constraint("y", "b|c"),
            equals_var("x", "y"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == result.assignment["y"]
        assert result.assignment["x"] == "b"

    def test_not_equals_var(self):
        result = solve_constraints([
            in_set("x", ["a", "b"]),
            in_set("y", ["a", "b"]),
            not_equals_var("x", "y"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] != result.assignment["y"]

    def test_equals_var_unsat(self):
        result = solve_constraints([
            equals_const("x", "abc"),
            equals_const("y", "def"),
            equals_var("x", "y"),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT


# ===== Length constraints =====

class TestLengthConstraints:
    def test_length_eq(self):
        result = solve_constraints([length_eq("x", 3)], ALPHA)
        assert result.result == SolveResult.SAT
        assert len(result.assignment["x"]) == 3

    def test_length_eq_zero(self):
        result = solve_constraints([length_eq("x", 0)], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == ""

    def test_length_le(self):
        result = solve_constraints([length_le("x", 2)], ALPHA)
        assert result.result == SolveResult.SAT
        assert len(result.assignment["x"]) <= 2

    def test_length_ge(self):
        result = solve_constraints([length_ge("x", 2)], ALPHA)
        assert result.result == SolveResult.SAT
        assert len(result.assignment["x"]) >= 2

    def test_length_range(self):
        result = solve_constraints([length_range("x", 2, 4)], ALPHA)
        assert result.result == SolveResult.SAT
        assert 2 <= len(result.assignment["x"]) <= 4

    def test_length_conflict(self):
        result = solve_constraints([
            length_ge("x", 5),
            length_le("x", 3),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_length_with_regex(self):
        result = solve_constraints([
            regex_constraint("x", "[ab]*"),
            length_eq("x", 4),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert len(result.assignment["x"]) == 4
        assert all(c in "ab" for c in result.assignment["x"])

    def test_negative_length_unsat(self):
        result = solve_constraints([length_eq("x", -1)], ALPHA)
        assert result.result == SolveResult.UNSAT


# ===== Contains/Prefix/Suffix =====

class TestSubstringConstraints:
    def test_contains(self):
        result = solve_constraints([contains("x", "abc")], ALPHA)
        assert result.result == SolveResult.SAT
        assert "abc" in result.assignment["x"]

    def test_contains_with_length(self):
        result = solve_constraints([
            contains("x", "ab"),
            length_eq("x", 4),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        x = result.assignment["x"]
        assert len(x) == 4
        assert "ab" in x

    def test_contains_unsat(self):
        result = solve_constraints([
            contains("x", "abc"),
            length_le("x", 2),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_prefix(self):
        result = solve_constraints([prefix("x", "abc")], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"].startswith("abc")

    def test_prefix_with_length(self):
        result = solve_constraints([
            prefix("x", "ab"),
            length_eq("x", 4),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        x = result.assignment["x"]
        assert len(x) == 4
        assert x.startswith("ab")

    def test_suffix(self):
        result = solve_constraints([suffix("x", "xyz")], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"].endswith("xyz")

    def test_suffix_with_length(self):
        result = solve_constraints([
            suffix("x", "cd"),
            length_eq("x", 4),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        x = result.assignment["x"]
        assert len(x) == 4
        assert x.endswith("cd")

    def test_prefix_suffix_combined(self):
        result = solve_constraints([
            prefix("x", "ab"),
            suffix("x", "cd"),
            length_eq("x", 4),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "abcd"

    def test_prefix_suffix_conflict(self):
        result = solve_constraints([
            prefix("x", "abc"),
            suffix("x", "xyz"),
            length_eq("x", 4),
        ], ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_empty_contains(self):
        result = solve_constraints([contains("x", "")], ALPHA)
        assert result.result == SolveResult.SAT


# ===== char_at constraints =====

class TestCharAtConstraints:
    def test_char_at_simple(self):
        result = solve_constraints([
            char_at("x", 0, "a"),
            length_eq("x", 3),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"][0] == "a"
        assert len(result.assignment["x"]) == 3

    def test_char_at_multiple(self):
        result = solve_constraints([
            char_at("x", 0, "a"),
            char_at("x", 1, "b"),
            char_at("x", 2, "c"),
            length_eq("x", 3),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "abc"

    def test_char_at_with_regex(self):
        result = solve_constraints([
            regex_constraint("x", "[a-z]*"),
            char_at("x", 0, "z"),
            length_eq("x", 2),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"][0] == "z"

    def test_char_at_negative_index(self):
        result = solve_constraints([char_at("x", -1, "a")], ALPHA)
        assert result.result == SolveResult.UNSAT


# ===== in_set constraints =====

class TestInSetConstraints:
    def test_in_set_simple(self):
        result = solve_constraints([
            in_set("x", ["hello", "world", "foo"]),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] in ("hello", "world", "foo")

    def test_in_set_with_regex(self):
        result = solve_constraints([
            in_set("x", ["ab", "cd", "ef"]),
            regex_constraint("x", "[c-f]*"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] in ("cd", "ef")

    def test_in_set_empty(self):
        result = solve_constraints([in_set("x", [])], ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_in_set_with_length(self):
        result = solve_constraints([
            in_set("x", ["a", "bb", "ccc"]),
            length_eq("x", 2),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "bb"


# ===== not_empty constraint =====

class TestNotEmptyConstraint:
    def test_not_empty(self):
        result = solve_constraints([not_empty("x")], ALPHA)
        assert result.result == SolveResult.SAT
        assert len(result.assignment["x"]) >= 1

    def test_not_empty_with_regex(self):
        result = solve_constraints([
            not_empty("x"),
            regex_constraint("x", "a*"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert len(result.assignment["x"]) >= 1
        assert all(c == "a" for c in result.assignment["x"])


# ===== Concatenation (word equations) =====

class TestConcatConstraints:
    def test_concat_simple(self):
        result = solve_constraints([
            equals_const("x", "ab"),
            equals_const("y", "cd"),
            concat_eq("x", "y", "z"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["z"] == "abcd"

    def test_concat_with_regex(self):
        result = solve_constraints([
            regex_constraint("x", "a+"),
            regex_constraint("y", "b+"),
            concat_eq("x", "y", "z"),
            length_eq("x", 2),
            length_eq("y", 2),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "aa"
        assert result.assignment["y"] == "bb"
        assert result.assignment["z"] == "aabb"

    def test_concat_z_known(self):
        result = solve_constraints([
            equals_const("z", "abcd"),
            concat_eq("x", "y", "z"),
            length_eq("x", 2),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] + result.assignment["y"] == "abcd"
        assert len(result.assignment["x"]) == 2

    def test_concat_empty(self):
        result = solve_constraints([
            equals_const("x", ""),
            equals_const("y", "abc"),
            concat_eq("x", "y", "z"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["z"] == "abc"


# ===== Multiple variables =====

class TestMultiVariable:
    def test_two_independent_vars(self):
        result = solve_constraints([
            regex_constraint("x", "a+"),
            regex_constraint("y", "b+"),
            length_eq("x", 2),
            length_eq("y", 3),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "aa"
        assert result.assignment["y"] == "bbb"

    def test_three_vars_with_equality(self):
        result = solve_constraints([
            in_set("x", ["a", "b"]),
            in_set("y", ["b", "c"]),
            in_set("z", ["c", "a"]),
            equals_var("x", "y"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == result.assignment["y"] == "b"

    def test_chain_equality(self):
        result = solve_constraints([
            equals_const("x", "test"),
            equals_var("x", "y"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "test"
        assert result.assignment["y"] == "test"


# ===== High-level API =====

class TestHighLevelAPI:
    def test_check_regex_membership(self):
        result = check_regex_membership("x", "a+b", [length_eq("x", 3)], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "aab"

    def test_check_word_equation(self):
        result = check_word_equation(
            "x", "y", "z",
            x_constraints=[regex_constraint("x", "a+")],
            y_constraints=[regex_constraint("y", "b+")],
            z_constraints=[length_eq("z", 4)],
            alphabet=ALPHA,
        )
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] + result.assignment["y"] == result.assignment["z"]

    def test_find_string_matching(self):
        s = find_string_matching("a+b", length=3, alphabet=ALPHA)
        assert s is not None
        assert len(s) == 3
        assert s.endswith("b")

    def test_find_string_matching_not_found(self):
        s = find_string_matching("a", length=5, alphabet=ALPHA)
        assert s is None

    def test_find_with_prefix(self):
        s = find_string_matching("[a-z]+", length=4, starting_with="ab", alphabet=ALPHA)
        assert s is not None
        assert len(s) == 4
        assert s.startswith("ab")

    def test_find_with_suffix(self):
        s = find_string_matching("[a-z]+", length=4, ending_with="yz", alphabet=ALPHA)
        assert s is not None
        assert len(s) == 4
        assert s.endswith("yz")

    def test_find_with_containing(self):
        s = find_string_matching("[a-z]+", length=5, containing="bcd", alphabet=ALPHA)
        assert s is not None
        assert len(s) == 5
        assert "bcd" in s

    def test_check_disjointness_disjoint(self):
        result = check_string_disjointness("a+", "b+", ALPHA)
        assert result.result == SolveResult.UNSAT

    def test_check_disjointness_overlapping(self):
        result = check_string_disjointness("(a|b)+", "(b|c)+", ALPHA)
        assert result.result == SolveResult.SAT
        assert all(c in "bc" for c in result.assignment["x"])


# ===== Enumeration =====

class TestEnumeration:
    def test_enumerate_finite(self):
        solutions = enumerate_solutions(
            [in_set("x", ["a", "b", "c"])],
            "x", max_count=10, alphabet=ALPHA,
        )
        assert set(solutions) == {"a", "b", "c"}

    def test_enumerate_regex(self):
        solutions = enumerate_solutions(
            [regex_constraint("x", "[ab]"), length_eq("x", 1)],
            "x", max_count=5, alphabet=ALPHA,
        )
        assert set(solutions) == {"a", "b"}

    def test_enumerate_bounded(self):
        solutions = enumerate_solutions(
            [regex_constraint("x", "a*"), length_le("x", 3)],
            "x", max_count=4, alphabet=ALPHA,
        )
        assert len(solutions) == 4
        assert "" in solutions
        assert "a" in solutions
        assert "aa" in solutions
        assert "aaa" in solutions


# ===== Implication checking =====

class TestImplication:
    def test_regex_implies_regex(self):
        # a+ implies (a|b)+
        result = check_implication(
            [regex_constraint("x", "a+")],
            regex_constraint("x", "(a|b)+"),
            ALPHA,
        )
        assert result is True

    def test_regex_not_implies(self):
        # (a|b)+ does NOT imply a+
        result = check_implication(
            [regex_constraint("x", "(a|b)+")],
            regex_constraint("x", "a+"),
            ALPHA,
        )
        assert result is False

    def test_const_implies_length(self):
        result = check_implication(
            [equals_const("x", "abc")],
            length_eq("x", 3),
            ALPHA,
        )
        assert result is True


# ===== Solver state / stats =====

class TestSolverState:
    def test_solver_stats(self):
        solver = StringConstraintSolver(ALPHA)
        solver.add(regex_constraint("x", "a*"))
        solver.add(length_le("x", 5))
        solver.check()
        stats = string_solver_stats(solver)
        assert stats['variables'] == 1
        assert stats['constraints'] == 2
        assert 'x_states' in stats

    def test_get_var_sfa(self):
        solver = StringConstraintSolver(ALPHA)
        solver.add(regex_constraint("x", "ab"))
        solver.check()
        sfa = solver.get_var_sfa("x")
        assert sfa is not None
        assert sfa.accepts("ab")

    def test_get_var_language_size(self):
        solver = StringConstraintSolver(ALPHA)
        solver.add(in_set("x", ["a", "b", "c"]))
        solver.check()
        size = solver.get_var_language_size("x", max_length=5)
        assert size == 3


# ===== Edge cases =====

class TestEdgeCases:
    def test_empty_string_var(self):
        result = solve_constraints([equals_const("x", "")], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == ""

    def test_no_constraints(self):
        solver = StringConstraintSolver(ALPHA)
        solver._ensure_var("x")
        result = solver.check()
        # Should succeed since sigma* accepts everything
        assert result.result == SolveResult.SAT

    def test_single_char_alphabet(self):
        result = solve_constraints([
            regex_constraint("x", "a*"),
            length_eq("x", 3),
        ], "a")
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "aaa"

    def test_multiple_same_var_constraints(self):
        result = solve_constraints([
            regex_constraint("x", "[a-z]+"),
            prefix("x", "ab"),
            suffix("x", "yz"),
            length_eq("x", 6),
            contains("x", "cd"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        x = result.assignment["x"]
        assert len(x) == 6
        assert x.startswith("ab")
        assert x.endswith("yz")
        assert "cd" in x


# ===== Combined constraint scenarios =====

class TestCombinedScenarios:
    def test_email_like_pattern(self):
        """Constrained email-like string."""
        result = solve_constraints([
            contains("x", "@"),
            prefix("x", "user"),
            suffix("x", "com"),
            length_range("x", 10, 20),
        ], "abcdefghijklmnopqrstuvwxyz@.")
        assert result.result == SolveResult.SAT
        x = result.assignment["x"]
        assert "@" in x
        assert x.startswith("user")
        assert x.endswith("com")

    def test_password_constraints(self):
        """Password must contain specific patterns."""
        result = solve_constraints([
            contains("x", "a"),
            contains("x", "1"),
            length_range("x", 4, 8),
        ], "abcdefghijklmnopqrstuvwxyz0123456789")
        assert result.result == SolveResult.SAT
        x = result.assignment["x"]
        assert "a" in x
        assert "1" in x
        assert 4 <= len(x) <= 8

    def test_three_var_concat_chain(self):
        """x . y = xy, xy . z = xyz."""
        result = solve_constraints([
            equals_const("x", "ab"),
            equals_const("y", "cd"),
            equals_const("z", "ef"),
            concat_eq("x", "y", "xy"),
            concat_eq("xy", "z", "xyz"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["xy"] == "abcd"
        assert result.assignment["xyz"] == "abcdef"

    def test_not_equals_with_finite_domain(self):
        """All different in small domain."""
        result = solve_constraints([
            in_set("x", ["a", "b", "c"]),
            in_set("y", ["a", "b", "c"]),
            not_equals_var("x", "y"),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] != result.assignment["y"]
        assert result.assignment["x"] in ("a", "b", "c")
        assert result.assignment["y"] in ("a", "b", "c")

    def test_regex_intersection_nonempty(self):
        """Two regexes that overlap on exactly one string."""
        result = solve_constraints([
            regex_constraint("x", "abc"),
            regex_constraint("x", "[a-c]+"),
            length_eq("x", 3),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "abc"

    def test_suffix_prefix_overlap(self):
        result = solve_constraints([
            prefix("x", "ab"),
            suffix("x", "bc"),
            length_eq("x", 3),
        ], ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "abc"


# ===== Solver reuse =====

class TestSolverReuse:
    def test_sequential_solves(self):
        """Solver can be rechecked after adding constraints."""
        solver = StringConstraintSolver(ALPHA)
        solver.add(regex_constraint("x", "[abc]+"))
        r1 = solver.check()
        assert r1.result == SolveResult.SAT

        solver.add(length_eq("x", 2))
        r2 = solver.check()
        assert r2.result == SolveResult.SAT
        assert len(r2.assignment["x"]) == 2

    def test_incremental_narrowing(self):
        solver = StringConstraintSolver(ALPHA)
        solver.add(regex_constraint("x", "[a-z]+"))
        solver.add(prefix("x", "ab"))
        solver.add(suffix("x", "yz"))
        solver.add(length_eq("x", 4))
        result = solver.check()
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "abyz"


# ===== Performance / stress =====

class TestPerformance:
    def test_many_constraints_same_var(self):
        """Many constraints on one variable."""
        constraints = [regex_constraint("x", "[a-z]*")]
        for i in range(5):
            constraints.append(char_at("x", i, chr(ord('a') + i)))
        constraints.append(length_eq("x", 5))
        result = solve_constraints(constraints, ALPHA)
        assert result.result == SolveResult.SAT
        assert result.assignment["x"] == "abcde"

    def test_multiple_independent_vars(self):
        """5 independent variables."""
        constraints = []
        for i in range(5):
            c = chr(ord('a') + i)
            var = f"v{i}"
            constraints.append(equals_const(var, c * 3))
        result = solve_constraints(constraints, ALPHA)
        assert result.result == SolveResult.SAT
        for i in range(5):
            c = chr(ord('a') + i)
            assert result.assignment[f"v{i}"] == c * 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
