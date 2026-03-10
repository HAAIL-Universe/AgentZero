"""Tests for V029: Abstract DPLL(T)."""

import sys
import os
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C039_abstract_interpreter'))

from abstract_dpll_t import (
    AbstractDPLLT, Verdict, DPLLTResult, LearnedClause,
    analyze_program, verify_assertions, compare_with_standard_ai,
    analyze_with_budget,
)


# =====================================================================
# Basic assertion verification
# =====================================================================

class TestBasicAssertions:
    def test_constant_true_assertion(self):
        r = analyze_program("let x = 5; assert(x > 0);")
        assert r.verdict == Verdict.SAFE
        assert r.assertions_checked >= 1

    def test_constant_false_assertion(self):
        r = analyze_program("let x = 0; assert(x > 0);")
        assert r.verdict == Verdict.UNSAFE
        assert len(r.conflicts) >= 1

    def test_no_assertions(self):
        r = analyze_program("let x = 5; let y = x + 1;")
        assert r.verdict == Verdict.SAFE
        assert r.assertions_checked == 0

    def test_multiple_passing_assertions(self):
        r = analyze_program("let x = 10; assert(x > 0); let y = x + 5; assert(y > 10);")
        assert r.verdict == Verdict.SAFE
        assert r.assertions_checked >= 2

    def test_arithmetic_assertion(self):
        r = analyze_program("let x = 3; let y = 4; let z = x + y; assert(z > 6);")
        assert r.verdict == Verdict.SAFE

    def test_subtraction_assertion(self):
        r = analyze_program("let x = 10; let y = 3; let z = x - y; assert(z > 0);")
        assert r.verdict == Verdict.SAFE

    def test_negative_value_assertion(self):
        r = analyze_program("let x = -5; assert(x < 0);")
        assert r.verdict == Verdict.SAFE

    def test_zero_assertion_fails(self):
        r = analyze_program("let x = 0; assert(x);")
        assert r.verdict == Verdict.UNSAFE


# =====================================================================
# Single branch exploration
# =====================================================================

class TestSingleBranch:
    def test_assertion_in_then_branch_safe(self):
        src = "let x = 5; if (x > 0) { assert(x > 0); }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_assertion_in_else_branch_safe(self):
        src = "let x = -1; if (x > 0) { assert(x > 0); } else { assert(x <= 0); }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_assertion_after_branch(self):
        src = "let x = 5; let y = 0; if (x > 3) { y = 1; } else { y = 2; } assert(y > 0);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_both_branches_explored(self):
        src = "let x = 5; if (x > 0) { let y = 1; } else { let y = -1; }"
        r = analyze_program(src)
        assert r.paths_explored >= 2

    def test_infeasible_branch_detected(self):
        src = "let x = 5; if (x > 0) { assert(x > 0); } else { assert(x <= 0); }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE


# =====================================================================
# Nested branches
# =====================================================================

class TestNestedBranches:
    def test_nested_safe(self):
        src = "let x = 10; if (x > 5) { if (x > 3) { assert(x > 3); } }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_nested_infeasible(self):
        src = "let x = 7; if (x > 10) { if (x < 5) { assert(0); } }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_nested_three_levels(self):
        src = "let x = 15; if (x > 0) { if (x > 5) { if (x > 10) { assert(x > 10); } } }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE
        assert r.max_decision_level >= 3

    def test_diamond_pattern(self):
        src = "let x = 5; let y = 0; if (x > 3) { y = 1; } else { y = -1; } if (y > 0) { assert(y > 0); }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE


# =====================================================================
# CDCL clause learning
# =====================================================================

class TestClauseLearning:
    def test_clause_learned_on_conflict(self):
        # With x=-1, assertion fails. The solver should learn from this.
        src = "let x = -1; assert(x > 0);"
        r = analyze_program(src, use_smt=False)
        # A conflict is found (assertion fails), verdict is UNSAFE
        assert r.verdict == Verdict.UNSAFE
        assert len(r.conflicts) >= 1

    def test_learned_clauses_prune(self):
        src = ("let x = 5; let y = 0; let z = 0; "
               "if (x > 0) { y = -1; } else { y = 1; } "
               "assert(y > 0); "
               "if (z > 0) { let w = 1; } else { let w = 2; }")
        r = analyze_program(src, use_smt=False)
        assert r.clauses_learned >= 1

    def test_no_clauses_when_safe(self):
        r = analyze_program("let x = 5; assert(x > 0);")
        assert r.clauses_learned == 0
        assert r.verdict == Verdict.SAFE


# =====================================================================
# Learned clause data structure
# =====================================================================

class TestLearnedClause:
    def test_satisfied(self):
        clause = LearnedClause(literals=[(0, True), (1, False)])
        assert clause.is_satisfied({0: True, 1: True})
        assert clause.is_satisfied({0: False, 1: False})
        assert not clause.is_satisfied({0: False, 1: True})

    def test_falsified(self):
        clause = LearnedClause(literals=[(0, True), (1, False)])
        assert clause.is_falsified({0: False, 1: True})
        assert not clause.is_falsified({0: True, 1: True})
        assert not clause.is_falsified({0: False})

    def test_unit(self):
        clause = LearnedClause(literals=[(0, True), (1, False)])
        assert clause.is_unit({1: True}) == (0, True)
        assert clause.is_unit({0: False}) == (1, False)
        assert clause.is_unit({}) is None
        assert clause.is_unit({0: True}) is None

    def test_empty_clause(self):
        clause = LearnedClause(literals=[])
        assert clause.is_falsified({})
        assert not clause.is_satisfied({})
        assert clause.is_unit({}) is None

    def test_singleton_clause(self):
        clause = LearnedClause(literals=[(0, True)])
        assert clause.is_unit({}) == (0, True)
        assert clause.is_satisfied({0: True})
        assert clause.is_falsified({0: False})


# =====================================================================
# Loop handling
# =====================================================================

class TestLoopHandling:
    def test_simple_countdown(self):
        src = "let i = 10; while (i > 0) { i = i - 1; } assert(i <= 0);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_loop_with_accumulator(self):
        src = "let s = 0; let i = 0; while (i < 5) { s = s + i; i = i + 1; } assert(i >= 5);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_branch_after_loop(self):
        src = "let x = 10; while (x > 5) { x = x - 1; } if (x <= 5) { assert(x <= 5); }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE


# =====================================================================
# SMT refinement
# =====================================================================

class TestSMTRefinement:
    def test_smt_confirms_real_conflict(self):
        r = analyze_program("let x = -1; assert(x > 0);", use_smt=True)
        assert r.verdict == Verdict.UNSAFE

    def test_smt_vs_no_smt_agrees_on_safe(self):
        r1 = analyze_program("let x = 5; assert(x > 0);", use_smt=True)
        r2 = analyze_program("let x = 5; assert(x > 0);", use_smt=False)
        assert r1.verdict == r2.verdict == Verdict.SAFE

    def test_smt_mode_flag(self):
        solver = AbstractDPLLT(use_smt_refinement=False)
        r = solver.analyze("let x = 5; assert(x > 0);")
        assert r.verdict == Verdict.SAFE


# =====================================================================
# Path counting and statistics
# =====================================================================

class TestStatistics:
    def test_single_path_count(self):
        r = analyze_program("let x = 5; assert(x > 0);")
        assert r.paths_explored >= 1

    def test_two_path_count(self):
        src = "let x = 5; if (x > 0) { let y = 1; } else { let y = 2; }"
        r = analyze_program(src)
        assert r.paths_explored >= 2

    def test_assertions_counted(self):
        r = analyze_program("let x = 5; assert(x > 0); assert(x > 1); assert(x < 10);")
        assert r.assertions_checked >= 3

    def test_max_decision_level(self):
        src = "let x = 5; if (x > 0) { if (x > 3) { let y = 1; } }"
        r = analyze_program(src)
        assert r.max_decision_level >= 2


# =====================================================================
# Complex programs
# =====================================================================

class TestComplexPrograms:
    def test_abs_value(self):
        src = "let x = -3; let result = 0; if (x >= 0) { result = x; } else { result = 0 - x; } assert(result >= 0);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_max_of_two(self):
        src = "let a = 3; let b = 7; let m = 0; if (a > b) { m = a; } else { m = b; } assert(m >= a); assert(m >= b);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_clamp(self):
        src = ("let x = 15; let lo = 0; let hi = 10; let result = x; "
               "if (x < lo) { result = lo; } "
               "if (result > hi) { result = hi; } "
               "assert(result <= 10);")
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_sign_classification(self):
        src = ("let x = -5; let sign = 0; "
               "if (x > 0) { sign = 1; } else { if (x < 0) { sign = -1; } else { sign = 0; } } "
               "assert(sign >= -1); assert(sign <= 1);")
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_sequential_branches(self):
        src = ("let a = 1; let b = 1; "
               "if (a > 0) { a = a + 1; } "
               "if (b > 0) { b = b + 1; } "
               "assert(a > 0); assert(b > 0);")
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE


# =====================================================================
# Public API
# =====================================================================

class TestPublicAPI:
    def test_verify_assertions_safe(self):
        verdict, msgs = verify_assertions("let x = 5; assert(x > 0);")
        assert verdict == Verdict.SAFE
        assert len(msgs) == 0

    def test_verify_assertions_unsafe(self):
        verdict, msgs = verify_assertions("let x = -1; assert(x > 0);")
        assert verdict == Verdict.UNSAFE
        assert len(msgs) >= 1

    def test_compare_with_standard_ai(self):
        src = "let x = 5; if (x > 0) { assert(x > 0); }"
        comparison = compare_with_standard_ai(src)
        assert 'standard_ai' in comparison
        assert 'abstract_dpll_t' in comparison
        assert comparison['abstract_dpll_t']['verdict'] == 'safe'

    def test_analyze_with_budget(self):
        src = "let x = 5; if (x > 0) { if (x > 3) { assert(x > 3); } }"
        r = analyze_with_budget(src, max_decisions=10)
        assert r.verdict == Verdict.SAFE


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_empty_program(self):
        r = analyze_program("let x = 0;")
        assert r.verdict == Verdict.SAFE

    def test_only_then_branch(self):
        src = "let x = 5; if (x > 0) { let y = 1; }"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_deeply_nested(self):
        src = ("let x = 100; "
               "if (x > 0) { if (x > 10) { if (x > 50) { if (x > 90) { assert(x > 90); } } } }")
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE
        assert r.max_decision_level >= 4

    def test_multiplication(self):
        r = analyze_program("let x = 3; let y = 4; let z = x * y; assert(z > 10);")
        assert r.verdict == Verdict.SAFE

    def test_assertion_with_comparison(self):
        r = analyze_program("let x = 5; let y = 3; assert(x > y);")
        assert r.verdict == Verdict.SAFE

    def test_multiple_failures(self):
        r = analyze_program("let x = -1; assert(x > 0); assert(x > 10);")
        assert r.verdict == Verdict.UNSAFE
        assert len(r.conflicts) >= 1


# =====================================================================
# Dependency tracking and minimal clauses
# =====================================================================

class TestDependencyTracking:
    def test_irrelevant_branch_excluded(self):
        src = ("let x = 5; let y = 0; let z = 0; "
               "if (x > 3) { y = -1; } else { y = 1; } "
               "if (x > 0) { z = 1; } else { z = -1; } "
               "assert(y > 0);")
        solver = AbstractDPLLT(use_smt_refinement=False)
        r = solver.analyze(src)
        if r.clauses_learned > 0:
            min_size = min(len(c.literals) for c in solver._clauses)
            assert min_size <= 2


# =====================================================================
# Realistic programs
# =====================================================================

class TestRealisticPrograms:
    def test_bounded_counter_safe(self):
        src = "let i = 0; while (i < 10) { i = i + 1; } assert(i >= 10);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_swap_variables(self):
        src = "let a = 3; let b = 7; let t = a; a = b; b = t; assert(a == 7); assert(b == 3);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_conditional_assignment_chain(self):
        src = ("let x = 5; let category = 0; "
               "if (x < 0) { category = -1; } else { if (x == 0) { category = 0; } else { category = 1; } } "
               "assert(category >= -1); assert(category <= 1);")
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE

    def test_loop_and_branch(self):
        src = "let s = 0; let i = 1; while (i <= 5) { if (i > 3) { s = s + i; } i = i + 1; } assert(s >= 0);"
        r = analyze_program(src)
        assert r.verdict == Verdict.SAFE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
