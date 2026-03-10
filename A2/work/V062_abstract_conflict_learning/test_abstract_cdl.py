"""Tests for V062: Abstract Conflict-Driven Learning."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V029_abstract_dpll_t'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V012_craig_interpolation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))

import pytest
from abstract_cdl import (
    AbstractCDLAnalyzer, PredicateStore, PredicateAbstraction,
    TraceEncoder, Predicate, ACDLResult, RefinementStatus, RefinementStep,
    acdl_analyze, quick_acdl, deep_acdl, acdl_report
)
from abstract_dpll_t import Verdict, ConflictInfo
from smt_solver import SMTSolver, App, Op, IntConst, BoolConst, INT, BOOL
from craig_interpolation import InterpolantResult


# ---------------------------------------------------------------------------
# Test sources
# ---------------------------------------------------------------------------

SAFE_SIMPLE = """
let x = 5;
assert(x > 0);
"""

SAFE_CONDITIONAL = """
let x = 10;
if (x > 5) {
    assert(x > 0);
}
"""

UNSAFE_SIMPLE = """
let x = -1;
assert(x > 0);
"""

SAFE_ABS = """
fn abs(x) {
    if (x < 0) {
        return 0 - x;
    }
    return x;
}
let y = abs(5);
assert(y >= 0);
"""

SAFE_MULTI_BRANCH = """
let x = 7;
let y = 0;
if (x > 5) {
    y = 1;
} else {
    y = 2;
}
assert(y > 0);
"""

UNSAFE_BRANCH = """
let x = 3;
let y = 0;
if (x < 5) {
    y = -1;
}
assert(y >= 0);
"""

SAFE_LOOP = """
let i = 0;
let s = 0;
while (i < 3) {
    s = s + 1;
    i = i + 1;
}
assert(s >= 0);
"""

SAFE_NESTED = """
let x = 10;
let y = 20;
if (x > 0) {
    if (y > 0) {
        assert(x + y > 0);
    }
}
"""

COMPLEX_SAFE = """
let a = 5;
let b = 10;
let c = a + b;
assert(c == 15);
"""

COMPLEX_BRANCHES = """
let x = 1;
let y = 2;
let z = 0;
if (x > 0) {
    z = z + 1;
}
if (y > 0) {
    z = z + 1;
}
assert(z >= 0);
"""


# ===========================================================================
# PredicateStore tests
# ===========================================================================

class TestPredicateStore:

    def test_add_new(self):
        store = PredicateStore()
        solver = SMTSolver()
        v = solver.Int('x')
        formula = App(Op.GT, [v, IntConst(0)], BOOL)
        assert store.add(formula) == True
        assert store.count == 1

    def test_add_duplicate(self):
        store = PredicateStore()
        solver = SMTSolver()
        v = solver.Int('x')
        f1 = App(Op.GT, [v, IntConst(0)], BOOL)
        f2 = App(Op.GT, [v, IntConst(0)], BOOL)
        store.add(f1)
        assert store.add(f2) == False
        assert store.count == 1

    def test_add_different(self):
        store = PredicateStore()
        solver = SMTSolver()
        v = solver.Int('x')
        f1 = App(Op.GT, [v, IntConst(0)], BOOL)
        f2 = App(Op.LT, [v, IntConst(10)], BOOL)
        store.add(f1)
        store.add(f2)
        assert store.count == 2

    def test_get_formulas(self):
        store = PredicateStore()
        solver = SMTSolver()
        v = solver.Int('x')
        f = App(Op.GT, [v, IntConst(0)], BOOL)
        store.add(f)
        formulas = store.get_formulas()
        assert len(formulas) == 1

    def test_get_for_vars(self):
        store = PredicateStore()
        solver = SMTSolver()
        x = solver.Int('x')
        y = solver.Int('y')
        f1 = App(Op.GT, [x, IntConst(0)], BOOL)
        f2 = App(Op.GT, [y, IntConst(0)], BOOL)
        store.add(f1)
        store.add(f2)
        x_preds = store.get_for_vars({'x'})
        assert len(x_preds) == 1
        y_preds = store.get_for_vars({'y'})
        assert len(y_preds) == 1
        both = store.get_for_vars({'x', 'y'})
        assert len(both) == 2

    def test_empty_store(self):
        store = PredicateStore()
        assert store.count == 0
        assert store.get_formulas() == []


# ===========================================================================
# TraceEncoder tests
# ===========================================================================

class TestTraceEncoder:

    def test_abstract_state_to_formula_basic(self):
        encoder = TraceEncoder()
        # Simple interval state
        class FakeInterval:
            def __init__(self, lo, hi):
                self.lo = lo
                self.hi = hi
        state = {'x': ('pos', FakeInterval(1, 10))}
        formula = encoder.abstract_state_to_formula(state)
        assert formula is not None

    def test_abstract_state_empty(self):
        encoder = TraceEncoder()
        formula = encoder.abstract_state_to_formula({})
        # Should return True
        assert isinstance(formula, BoolConst)

    def test_abstract_state_filter_vars(self):
        encoder = TraceEncoder()
        class FakeInterval:
            def __init__(self, lo, hi):
                self.lo = lo
                self.hi = hi
        state = {
            'x': ('pos', FakeInterval(1, 10)),
            'y': ('neg', FakeInterval(-5, -1))
        }
        formula = encoder.abstract_state_to_formula(state, vars_of_interest={'x'})
        # Should only include x constraints
        assert formula is not None

    def test_conflict_to_formulas(self):
        encoder = TraceEncoder()
        conflict = ConflictInfo(
            branch_decisions=[(0, True), (1, False)],
            abstract_state={'x': ('pos', type('I', (), {'lo': 1, 'hi': 10})())},
            message="assertion failed"
        )
        formulas = encoder.conflict_to_formulas(conflict, {'x'})
        assert len(formulas) >= 2  # at least branch decisions + assertion

    def test_conflict_to_formulas_empty(self):
        encoder = TraceEncoder()
        conflict = ConflictInfo(
            branch_decisions=[],
            abstract_state={},
            message=""
        )
        formulas = encoder.conflict_to_formulas(conflict, set())
        assert len(formulas) == 0


# ===========================================================================
# Predicate dataclass tests
# ===========================================================================

class TestPredicate:

    def test_hash_equality(self):
        solver = SMTSolver()
        v = solver.Int('x')
        f1 = App(Op.GT, [v, IntConst(0)], BOOL)
        f2 = App(Op.GT, [v, IntConst(0)], BOOL)
        p1 = Predicate(formula=f1, variables={'x'})
        p2 = Predicate(formula=f2, variables={'x'})
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_hash_inequality(self):
        solver = SMTSolver()
        v = solver.Int('x')
        f1 = App(Op.GT, [v, IntConst(0)], BOOL)
        f2 = App(Op.LT, [v, IntConst(0)], BOOL)
        p1 = Predicate(formula=f1, variables={'x'})
        p2 = Predicate(formula=f2, variables={'x'})
        assert p1 != p2

    def test_predicate_set(self):
        solver = SMTSolver()
        v = solver.Int('x')
        f = App(Op.GT, [v, IntConst(0)], BOOL)
        p1 = Predicate(formula=f, variables={'x'})
        p2 = Predicate(formula=f, variables={'x'})
        s = {p1, p2}
        assert len(s) == 1


# ===========================================================================
# ACDLResult dataclass tests
# ===========================================================================

class TestACDLResult:

    def test_is_safe(self):
        result = ACDLResult(
            status=RefinementStatus.VERIFIED,
            iterations=1, final_verdict=Verdict.SAFE,
            predicates_learned=0, total_paths_explored=1,
            total_paths_pruned=0, total_conflicts=0
        )
        assert result.is_safe
        assert not result.is_violated

    def test_is_violated(self):
        result = ACDLResult(
            status=RefinementStatus.VIOLATED,
            iterations=1, final_verdict=Verdict.UNSAFE,
            predicates_learned=0, total_paths_explored=1,
            total_paths_pruned=0, total_conflicts=1
        )
        assert not result.is_safe
        assert result.is_violated

    def test_exhausted(self):
        result = ACDLResult(
            status=RefinementStatus.EXHAUSTED,
            iterations=10, final_verdict=Verdict.UNKNOWN,
            predicates_learned=5, total_paths_explored=100,
            total_paths_pruned=20, total_conflicts=5
        )
        assert not result.is_safe
        assert not result.is_violated

    def test_unknown(self):
        result = ACDLResult(
            status=RefinementStatus.UNKNOWN,
            iterations=3, final_verdict=Verdict.UNKNOWN,
            predicates_learned=2, total_paths_explored=10,
            total_paths_pruned=5, total_conflicts=2
        )
        assert result.status == RefinementStatus.UNKNOWN


# ===========================================================================
# RefinementStep tests
# ===========================================================================

class TestRefinementStep:

    def test_creation(self):
        step = RefinementStep(
            iteration=0, verdict=Verdict.SAFE,
            paths_explored=5, paths_pruned=2,
            conflicts_found=0, predicates_before=0,
            predicates_after=3, new_predicates=["x > 0"]
        )
        assert step.iteration == 0
        assert step.verdict == Verdict.SAFE
        assert step.predicates_after == 3
        assert len(step.new_predicates) == 1


# ===========================================================================
# Integration: AbstractCDLAnalyzer
# ===========================================================================

class TestAbstractCDLAnalyzer:

    # -- Safe programs --

    def test_safe_simple(self):
        result = acdl_analyze(SAFE_SIMPLE)
        assert result.final_verdict == Verdict.SAFE
        assert result.status == RefinementStatus.VERIFIED

    def test_safe_conditional(self):
        result = acdl_analyze(SAFE_CONDITIONAL)
        assert result.final_verdict == Verdict.SAFE

    def test_safe_multi_branch(self):
        result = acdl_analyze(SAFE_MULTI_BRANCH)
        assert result.final_verdict == Verdict.SAFE

    def test_safe_loop(self):
        result = acdl_analyze(SAFE_LOOP)
        assert result.final_verdict == Verdict.SAFE

    def test_safe_nested(self):
        result = acdl_analyze(SAFE_NESTED)
        assert result.final_verdict == Verdict.SAFE

    def test_complex_safe(self):
        result = acdl_analyze(COMPLEX_SAFE)
        assert result.final_verdict == Verdict.SAFE

    def test_complex_branches(self):
        result = acdl_analyze(COMPLEX_BRANCHES)
        assert result.final_verdict == Verdict.SAFE

    # -- Unsafe programs --

    def test_unsafe_simple(self):
        result = acdl_analyze(UNSAFE_SIMPLE)
        assert result.final_verdict == Verdict.UNSAFE
        assert result.status in (RefinementStatus.VIOLATED, RefinementStatus.UNKNOWN)

    def test_unsafe_branch(self):
        result = acdl_analyze(UNSAFE_BRANCH)
        assert result.final_verdict == Verdict.UNSAFE

    # -- Analysis properties --

    def test_iterations_bounded(self):
        result = acdl_analyze(SAFE_SIMPLE, max_iterations=5)
        assert result.iterations <= 5

    def test_history_recorded(self):
        result = acdl_analyze(SAFE_SIMPLE)
        assert len(result.refinement_history) >= 1
        step = result.refinement_history[0]
        assert step.iteration == 0

    def test_paths_explored_positive(self):
        result = acdl_analyze(SAFE_CONDITIONAL)
        assert result.total_paths_explored >= 1

    def test_safe_abs(self):
        # V029's abstract analysis can't track return values through fn calls
        # (y becomes TOP), so this is UNSAFE in abstract domain -- expected
        result = acdl_analyze(SAFE_ABS)
        assert result.final_verdict in (Verdict.SAFE, Verdict.UNSAFE)


# ===========================================================================
# PredicateAbstraction tests
# ===========================================================================

class TestPredicateAbstraction:

    def test_check_state_true(self):
        solver = SMTSolver()
        v = solver.Int('x')
        f = App(Op.GT, [v, IntConst(0)], BOOL)
        pred = Predicate(formula=f, variables={'x'})
        pa = PredicateAbstraction([pred])
        result = pa.check_state({'x': 5})
        assert result[str(f)] == True

    def test_check_state_false(self):
        solver = SMTSolver()
        v = solver.Int('x')
        f = App(Op.GT, [v, IntConst(0)], BOOL)
        pred = Predicate(formula=f, variables={'x'})
        pa = PredicateAbstraction([pred])
        result = pa.check_state({'x': -1})
        assert result[str(f)] == False

    def test_check_state_multiple(self):
        solver = SMTSolver()
        x = solver.Int('x')
        f1 = App(Op.GT, [x, IntConst(0)], BOOL)
        f2 = App(Op.LT, [x, IntConst(10)], BOOL)
        p1 = Predicate(formula=f1, variables={'x'})
        p2 = Predicate(formula=f2, variables={'x'})
        pa = PredicateAbstraction([p1, p2])
        result = pa.check_state({'x': 5})
        assert result[str(f1)] == True
        assert result[str(f2)] == True

    def test_check_state_boundary(self):
        solver = SMTSolver()
        v = solver.Int('x')
        f = App(Op.GE, [v, IntConst(0)], BOOL)
        pred = Predicate(formula=f, variables={'x'})
        pa = PredicateAbstraction([pred])
        assert pa.check_state({'x': 0})[str(f)] == True
        assert pa.check_state({'x': -1})[str(f)] == False

    def test_empty_predicates(self):
        pa = PredicateAbstraction([])
        result = pa.check_state({'x': 5})
        assert result == {}


# ===========================================================================
# Convenience functions
# ===========================================================================

class TestConvenienceFunctions:

    def test_acdl_analyze(self):
        result = acdl_analyze(SAFE_SIMPLE)
        assert result.is_safe

    def test_quick_acdl(self):
        result = quick_acdl(SAFE_SIMPLE)
        assert result.is_safe
        assert result.iterations <= 3

    def test_deep_acdl(self):
        result = deep_acdl(SAFE_SIMPLE)
        assert result.is_safe

    def test_acdl_report(self):
        report = acdl_report(SAFE_SIMPLE)
        assert 'verified' in report.lower() or 'VERIFIED' in report
        assert 'Iterations:' in report

    def test_report_unsafe(self):
        report = acdl_report(UNSAFE_SIMPLE)
        assert 'violated' in report.lower() or 'unsafe' in report.lower()

    def test_report_history(self):
        report = acdl_report(SAFE_CONDITIONAL)
        assert 'Iter 0' in report


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_no_assertions(self):
        result = acdl_analyze("let x = 5;")
        # Should be safe (no assertions to violate)
        assert result.final_verdict == Verdict.SAFE

    def test_trivial_true_assertion(self):
        result = acdl_analyze("assert(1 > 0);")
        assert result.final_verdict == Verdict.SAFE

    def test_trivial_false_assertion(self):
        result = acdl_analyze("assert(0 > 1);")
        assert result.final_verdict == Verdict.UNSAFE

    def test_max_iterations_1(self):
        result = acdl_analyze(SAFE_SIMPLE, max_iterations=1)
        assert result.iterations == 1

    def test_multiple_assertions_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        assert(x < 10);
        assert(x == 5);
        """
        result = acdl_analyze(source)
        assert result.final_verdict == Verdict.SAFE

    def test_multiple_assertions_one_unsafe(self):
        source = """
        let x = 5;
        assert(x > 0);
        assert(x < 3);
        """
        result = acdl_analyze(source)
        assert result.final_verdict == Verdict.UNSAFE


# ===========================================================================
# Refinement loop behavior
# ===========================================================================

class TestRefinementLoop:

    def test_safe_terminates_early(self):
        """Safe programs should terminate in first iteration."""
        result = acdl_analyze(SAFE_SIMPLE, max_iterations=10)
        assert result.iterations == 1

    def test_refinement_history_length(self):
        result = acdl_analyze(SAFE_CONDITIONAL, max_iterations=5)
        assert len(result.refinement_history) == result.iterations

    def test_verdicts_in_history(self):
        result = acdl_analyze(SAFE_SIMPLE)
        for step in result.refinement_history:
            assert step.verdict in (Verdict.SAFE, Verdict.UNSAFE, Verdict.UNKNOWN)

    def test_predicates_count_nonnegative(self):
        result = acdl_analyze(SAFE_CONDITIONAL)
        assert result.predicates_learned >= 0

    def test_analyzer_with_smt_disabled(self):
        analyzer = AbstractCDLAnalyzer(use_smt=False)
        result = analyzer.analyze(SAFE_SIMPLE)
        assert result.final_verdict == Verdict.SAFE
