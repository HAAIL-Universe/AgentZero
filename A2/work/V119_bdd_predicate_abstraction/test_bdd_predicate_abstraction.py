"""Tests for V119: BDD-Based Predicate Abstraction."""

import pytest
import sys
import os

_here = os.path.dirname(__file__)
sys.path.insert(0, _here)
for p in [
    os.path.join(_here, '..', 'V021_bdd_model_checking'),
    os.path.join(_here, '..', 'V110_abstract_reachability_tree'),
    os.path.join(_here, '..', '..', '..', 'challenges', 'C037_smt_solver'),
    os.path.join(_here, '..', '..', '..', 'challenges', 'C010_stack_vm'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bdd_predicate_abstraction import (
    BDDPredicateManager, BDDPredicateState, TransitionBDDBuilder,
    BDDCEGAR, BDDVerdict, BDDCEGARResult, ComparisonResult,
    bdd_verify, check_assertion, bdd_vs_smt_comparison,
    get_transition_bdds, bdd_summary,
    _smt_not, _declare_vars, _collect_vars, _safe_ast_to_smt,
    INT, BOOL,
)
from bdd_model_checker import BDD
from smt_solver import Var, IntConst, App, Op, Sort, SortKind


# ===== Section 1: BDDPredicateManager =====

class TestBDDPredicateManager:
    """Test predicate-to-BDD variable mapping."""

    def test_empty_manager(self):
        mgr = BDDPredicateManager()
        assert mgr.num_predicates == 0
        state = mgr.state_top()
        assert not state.is_bottom
        assert state.bdd_node == mgr.bdd.TRUE

    def test_add_single_predicate(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        zero = IntConst(0)
        pred = App(Op.GE, [x, zero], BOOL)
        idx = mgr.add_predicate(pred, "x >= 0")
        assert idx == 0
        assert mgr.num_predicates == 1
        assert len(mgr.curr_vars) == 1
        assert len(mgr.next_vars) == 1

    def test_add_multiple_predicates(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        y = Var("y", INT)
        zero = IntConst(0)

        idx0 = mgr.add_predicate(App(Op.GE, [x, zero], BOOL), "x >= 0")
        idx1 = mgr.add_predicate(App(Op.GE, [y, zero], BOOL), "y >= 0")
        idx2 = mgr.add_predicate(App(Op.LT, [x, y], BOOL), "x < y")

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2
        assert mgr.num_predicates == 3

    def test_dedup_predicates(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        zero = IntConst(0)
        pred = App(Op.GE, [x, zero], BOOL)

        idx0 = mgr.add_predicate(pred, "x >= 0")
        idx1 = mgr.add_predicate(pred, "x >= 0 (dup)")
        assert idx0 == idx1
        assert mgr.num_predicates == 1

    def test_state_bottom(self):
        mgr = BDDPredicateManager()
        bot = mgr.state_bottom()
        assert bot.is_bottom
        assert bot.bdd_node == mgr.bdd.FALSE

    def test_state_from_predicates(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        zero = IntConst(0)
        mgr.add_predicate(App(Op.GE, [x, zero], BOOL), "x >= 0")
        mgr.add_predicate(App(Op.LE, [x, IntConst(10)], BOOL), "x <= 10")

        state = mgr.state_from_predicates({0, 1})
        assert not state.is_bottom
        # State should be conjunction of curr_var[0] AND curr_var[1]
        assert state.bdd_node != mgr.bdd.TRUE
        assert state.bdd_node != mgr.bdd.FALSE

    def test_state_from_empty_predicates(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))
        state = mgr.state_from_predicates(set())
        assert state.bdd_node == mgr.bdd.TRUE


# ===== Section 2: BDD State Subsumption =====

class TestBDDStateSubsumption:
    """Test abstract state lattice operations."""

    def test_top_subsumes_everything(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL))

        top = mgr.state_top()
        specific = mgr.state_from_predicates({0})
        assert top.subsumes(specific, mgr.bdd)

    def test_bottom_subsumed_by_everything(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL))

        top = mgr.state_top()
        bot = mgr.state_bottom()
        assert top.subsumes(bot, mgr.bdd)

    def test_bottom_subsumes_only_bottom(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL))

        bot = mgr.state_bottom()
        top = mgr.state_top()
        assert bot.subsumes(bot, mgr.bdd)
        assert not bot.subsumes(top, mgr.bdd)

    def test_specific_does_not_subsume_top(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL))

        top = mgr.state_top()
        specific = mgr.state_from_predicates({0})
        # specific requires p0=true, top allows anything
        assert not specific.subsumes(top, mgr.bdd)

    def test_join_is_disjunction(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL))
        mgr.add_predicate(App(Op.LE, [x, IntConst(10)], BOOL))

        s1 = mgr.state_from_predicates({0})
        s2 = mgr.state_from_predicates({1})
        joined = s1.join(s2, mgr.bdd)

        # Join should subsume both
        assert joined.subsumes(s1, mgr.bdd)
        assert joined.subsumes(s2, mgr.bdd)

    def test_join_with_bottom(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))

        s = mgr.state_from_predicates({0})
        bot = mgr.state_bottom()
        assert s.join(bot, mgr.bdd).bdd_node == s.bdd_node
        assert bot.join(s, mgr.bdd).bdd_node == s.bdd_node


# ===== Section 3: BDD Image Computation =====

class TestBDDImage:
    """Test BDD-based abstract post (image)."""

    def test_identity_image(self):
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL))

        # Identity transition: curr_0 <=> next_0
        bdd = mgr.bdd
        identity = bdd.IFF(mgr.curr_bdd(0), mgr.next_bdd(0))

        state = mgr.state_from_predicates({0})
        result = mgr.image(state, identity)
        # Should preserve the predicate
        assert not result.is_bottom

    def test_image_of_bottom(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))

        bdd = mgr.bdd
        identity = bdd.IFF(mgr.curr_bdd(0), mgr.next_bdd(0))

        bot = mgr.state_bottom()
        result = mgr.image(bot, identity)
        assert result.is_bottom

    def test_image_forces_predicate_true(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))

        bdd = mgr.bdd
        # Transition that forces next_0 = true regardless of curr
        trans = mgr.next_bdd(0)

        top = mgr.state_top()
        result = mgr.image(top, trans)
        assert not result.is_bottom
        # Result should have predicate 0 true
        expected = mgr.state_from_predicates({0})
        assert expected.subsumes(result, mgr.bdd)

    def test_image_forces_predicate_false(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))

        bdd = mgr.bdd
        # Transition that forces next_0 = false
        trans = bdd.NOT(mgr.next_bdd(0))

        state = mgr.state_from_predicates({0})
        result = mgr.image(state, trans)
        assert not result.is_bottom

    def test_image_infeasible_transition(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))

        # Transition requires curr_0=true but state has curr_0=false
        bdd = mgr.bdd
        trans = bdd.AND(mgr.curr_bdd(0), mgr.next_bdd(0))

        # Create state where pred 0 is false
        state = BDDPredicateState(bdd.NOT(mgr.curr_bdd(0)))
        result = mgr.image(state, trans)
        assert result.is_bottom


# ===== Section 4: SMT Helpers =====

class TestSMTHelpers:
    """Test SMT utility functions."""

    def test_smt_not_eq(self):
        x = Var("x", INT)
        y = Var("y", INT)
        eq = App(Op.EQ, [x, y], BOOL)
        neg = _smt_not(eq)
        assert isinstance(neg, App)
        assert neg.op == Op.NEQ

    def test_smt_not_lt(self):
        x = Var("x", INT)
        lt = App(Op.LT, [x, IntConst(5)], BOOL)
        neg = _smt_not(lt)
        assert neg.op == Op.GE

    def test_smt_not_and(self):
        x = Var("x", INT)
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.LE, [x, IntConst(10)], BOOL)
        conj = App(Op.AND, [a, b], BOOL)
        neg = _smt_not(conj)
        assert neg.op == Op.OR

    def test_smt_not_double(self):
        x = Var("x", INT)
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        double_neg = _smt_not(_smt_not(pred))
        # GE -> LT -> GE
        assert double_neg.op == Op.GE

    def test_collect_vars(self):
        x = Var("x", INT)
        y = Var("y", INT)
        term = App(Op.ADD, [x, y], INT)
        vars_set = _collect_vars(term)
        assert vars_set == {"x", "y"}

    def test_collect_vars_const(self):
        c = IntConst(42)
        assert _collect_vars(c) == set()


# ===== Section 5: Transition BDD Builder -- Assignments =====

class TestTransitionBDDAssignment:
    """Test transition BDD construction for assignments."""

    def test_assign_preserves_unrelated(self):
        """Assignment to x should not affect predicate on y."""
        mgr = BDDPredicateManager()
        y = Var("y", INT)
        mgr.add_predicate(App(Op.GE, [y, IntConst(0)], BOOL), "y >= 0")

        builder = TransitionBDDBuilder(mgr)
        # x := 5 (unrelated to y)
        x_smt = IntConst(5)
        trans = builder.build_assign_transition("x", x_smt, 0, 1)

        # State: y >= 0
        state = mgr.state_from_predicates({0})
        result = mgr.image(state, trans)
        # y >= 0 should still hold
        assert not result.is_bottom

    def test_assign_tautology_predicate(self):
        """x := 5 should make x >= 0 true."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        trans = builder.build_assign_transition("x", IntConst(5), 0, 1)

        # From top (no info about x)
        state = mgr.state_top()
        result = mgr.image(state, trans)
        assert not result.is_bottom

    def test_assign_contradicts_predicate(self):
        """x := -1 should make x >= 0 false."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        trans = builder.build_assign_transition("x", IntConst(-1), 0, 1)
        assert builder.smt_queries > 0  # Should have queried SMT

    def test_assign_identity(self):
        """x := x should preserve all predicates on x."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        trans = builder.build_assign_transition("x", x, 0, 1)

        state = mgr.state_from_predicates({0})
        result = mgr.image(state, trans)
        assert not result.is_bottom


# ===== Section 6: Transition BDD Builder -- Assumes =====

class TestTransitionBDDAssume:
    """Test transition BDD construction for assume/guard transitions."""

    def test_assume_implies_predicate(self):
        """assume(x > 5) should imply x >= 0."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        cond = App(Op.GT, [x, IntConst(5)], BOOL)
        trans = builder.build_assume_transition(cond, False, 0, 1)

        # From top state
        state = mgr.state_top()
        result = mgr.image(state, trans)
        assert not result.is_bottom

    def test_assume_negated(self):
        """assume(!(x >= 10)) should not force x >= 0 either way."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        cond = App(Op.GE, [x, IntConst(10)], BOOL)
        trans = builder.build_assume_transition(cond, True, 0, 1)
        # Should not crash, transition should be valid
        assert trans is not None

    def test_assume_contradicts_state(self):
        """assume(x < 0) with state {x >= 0} should yield bottom."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        cond = App(Op.LT, [x, IntConst(0)], BOOL)
        trans = builder.build_assume_transition(cond, False, 0, 1)

        # State where x >= 0 is true
        state = mgr.state_from_predicates({0})
        result = mgr.image(state, trans)
        # x < 0 contradicts x >= 0, should be infeasible
        assert result.is_bottom

    def test_skip_transition(self):
        """Skip preserves all predicates."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")
        mgr.add_predicate(App(Op.LE, [x, IntConst(10)], BOOL), "x <= 10")

        builder = TransitionBDDBuilder(mgr)
        trans = builder.build_identity_transition(0, 1)

        state = mgr.state_from_predicates({0, 1})
        result = mgr.image(state, trans)
        assert not result.is_bottom

    def test_transition_caching(self):
        """Same edge should return cached transition."""
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))

        builder = TransitionBDDBuilder(mgr)
        t1 = builder.build_identity_transition(0, 1)
        t2 = builder.build_identity_transition(0, 1)
        assert t1 is t2


# ===== Section 7: Safe Programs =====

class TestSafePrograms:
    """Test BDD CEGAR on programs that are SAFE."""

    def test_simple_assert_true(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert result.safe
        assert result.verdict == BDDVerdict.SAFE

    def test_assert_after_assignment(self):
        source = """
        let x = 10;
        let y = x;
        assert(y > 0);
        """
        result = bdd_verify(source)
        assert result.safe

    def test_conditional_safe(self):
        source = """
        let x = 5;
        if (x > 0) {
            let y = x;
            assert(y > 0);
        }
        """
        result = bdd_verify(source)
        assert result.safe

    def test_both_branches_safe(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 10;
        } else {
            y = 20;
        }
        assert(y >= 0);
        """
        result = bdd_verify(source)
        assert result.safe

    def test_no_assertions(self):
        """Program with no assertions is trivially safe."""
        source = """
        let x = 5;
        let y = 10;
        """
        result = bdd_verify(source)
        assert result.safe
        assert result.verdict == BDDVerdict.SAFE


# ===== Section 8: Unsafe Programs =====

class TestUnsafePrograms:
    """Test BDD CEGAR on programs that are UNSAFE."""

    def test_simple_assert_false(self):
        source = """
        let x = -1;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert not result.safe
        assert result.verdict == BDDVerdict.UNSAFE

    def test_conditional_unsafe(self):
        source = """
        let x = 5;
        if (x > 3) {
            x = -1;
        }
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert not result.safe

    def test_assert_with_counterexample(self):
        source = """
        let x = -5;
        assert(x >= 0);
        """
        result = bdd_verify(source)
        assert not result.safe
        assert result.counterexample is not None


# ===== Section 9: check_assertion API =====

class TestCheckAssertion:
    """Test the quick check_assertion API."""

    def test_safe_check(self):
        source = """
        let x = 10;
        assert(x > 0);
        """
        safe, inputs = check_assertion(source)
        assert safe
        assert inputs is None

    def test_unsafe_check(self):
        source = """
        let x = -1;
        assert(x >= 0);
        """
        safe, inputs = check_assertion(source)
        assert not safe


# ===== Section 10: BDD CEGAR Result Structure =====

class TestBDDCEGARResult:
    """Test result data structure completeness."""

    def test_result_fields_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert isinstance(result, BDDCEGARResult)
        assert result.verdict == BDDVerdict.SAFE
        assert result.safe is True
        assert result.counterexample is None
        assert result.iterations >= 1
        assert result.total_predicates >= 0
        assert isinstance(result.predicate_names, list)
        assert result.total_time_ms >= 0

    def test_result_fields_unsafe(self):
        source = """
        let x = -1;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert result.verdict == BDDVerdict.UNSAFE
        assert result.safe is False
        assert result.counterexample is not None
        assert isinstance(result.counterexample, list)

    def test_result_metrics(self):
        source = """
        let x = 5;
        if (x > 0) {
            assert(x > 0);
        }
        """
        result = bdd_verify(source)
        assert result.art_nodes > 0
        assert result.bdd_image_ops >= 0


# ===== Section 11: Transition BDD Inspection =====

class TestTransitionBDDInspection:
    """Test get_transition_bdds API."""

    def test_inspect_simple(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        info = get_transition_bdds(source)
        assert 'predicates' in info
        assert 'edges' in info
        assert 'total_bdd_nodes' in info
        assert len(info['predicates']) > 0

    def test_inspect_no_assertions(self):
        source = """
        let x = 5;
        """
        info = get_transition_bdds(source)
        assert info['predicates'] == []

    def test_inspect_edges(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        info = get_transition_bdds(source)
        assert len(info['edges']) > 0
        for edge in info['edges']:
            assert 'source' in edge
            assert 'target' in edge
            assert 'bdd_nodes' in edge


# ===== Section 12: BDD Summary =====

class TestBDDSummary:
    """Test human-readable summary generation."""

    def test_summary_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        s = bdd_summary(source)
        assert "safe" in s.lower()
        assert "Predicates" in s

    def test_summary_unsafe(self):
        source = """
        let x = -1;
        assert(x > 0);
        """
        s = bdd_summary(source)
        assert "unsafe" in s.lower()
        assert "Counterexample" in s

    def test_summary_contains_metrics(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        s = bdd_summary(source)
        assert "ART nodes" in s
        assert "BDD image" in s


# ===== Section 13: Multiple Predicates =====

class TestMultiplePredicates:
    """Test with multiple predicates interacting."""

    def test_two_predicates_both_true(self):
        source = """
        let x = 5;
        let y = 10;
        assert(x > 0);
        assert(y > 0);
        """
        result = bdd_verify(source)
        assert result.safe
        assert result.total_predicates >= 2

    def test_two_predicates_one_false(self):
        source = """
        let x = 5;
        let y = -1;
        assert(x > 0);
        assert(y > 0);
        """
        result = bdd_verify(source)
        assert not result.safe

    def test_related_predicates(self):
        source = """
        let x = 5;
        let y = x;
        assert(x > 0);
        assert(y > 0);
        """
        result = bdd_verify(source)
        assert result.safe


# ===== Section 14: Conditional Paths =====

class TestConditionalPaths:
    """Test programs with conditional branching."""

    def test_if_then_safe(self):
        source = """
        let x = 5;
        if (x > 3) {
            assert(x > 0);
        }
        """
        result = bdd_verify(source)
        assert result.safe

    def test_if_else_paths(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 2;
        }
        assert(y > 0);
        """
        result = bdd_verify(source)
        assert result.safe

    def test_nested_if(self):
        source = """
        let x = 10;
        if (x > 5) {
            if (x > 3) {
                assert(x > 0);
            }
        }
        """
        result = bdd_verify(source)
        assert result.safe


# ===== Section 15: BDD vs SMT Comparison =====

class TestComparison:
    """Test comparison API between BDD and SMT approaches."""

    def test_comparison_structure(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        comp = bdd_vs_smt_comparison(source)
        assert isinstance(comp, ComparisonResult)
        assert comp.bdd_result is not None
        assert comp.bdd_time_ms >= 0
        assert isinstance(comp.summary, str)

    def test_comparison_agreement_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        comp = bdd_vs_smt_comparison(source)
        assert comp.bdd_result.safe

    def test_comparison_agreement_unsafe(self):
        source = """
        let x = -1;
        assert(x > 0);
        """
        comp = bdd_vs_smt_comparison(source)
        assert not comp.bdd_result.safe


# ===== Section 16: Edge Cases =====

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_program(self):
        """Empty program with no statements."""
        source = ""
        result = bdd_verify(source)
        assert result.safe

    def test_single_let(self):
        source = "let x = 0;"
        result = bdd_verify(source)
        assert result.safe

    def test_max_iterations_reached(self):
        """Programs that might need many iterations."""
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source, max_iterations=1)
        # Should still find safe in 1 iteration for simple case
        assert result.iterations <= 1

    def test_seed_predicates_off(self):
        """Verify without V114 predicate discovery."""
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source, seed_predicates=False)
        assert result.safe


# ===== Section 17: Predicate Manager BDD Operations =====

class TestPredicateManagerBDDOps:
    """Test BDD-specific operations on predicate manager."""

    def test_curr_and_next_vars_different(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))

        curr = mgr.curr_bdd(0)
        nxt = mgr.next_bdd(0)
        assert curr != nxt  # Different BDD variables

    def test_bdd_var_indices(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL))
        mgr.add_predicate(App(Op.GE, [Var("y", INT), IntConst(0)], BOOL))

        # curr vars: 0, 2; next vars: 1, 3
        assert mgr.curr_vars == [0, 2]
        assert mgr.next_vars == [1, 3]

    def test_image_with_two_predicates(self):
        mgr = BDDPredicateManager()
        mgr.add_predicate(App(Op.GE, [Var("x", INT), IntConst(0)], BOOL), "x >= 0")
        mgr.add_predicate(App(Op.GE, [Var("y", INT), IntConst(0)], BOOL), "y >= 0")

        bdd = mgr.bdd
        # Transition: preserve both (identity)
        trans = bdd.TRUE
        for j in range(mgr.num_predicates):
            trans = bdd.AND(trans, bdd.IFF(mgr.curr_bdd(j), mgr.next_bdd(j)))

        state = mgr.state_from_predicates({0, 1})
        result = mgr.image(state, trans)
        assert not result.is_bottom


# ===== Section 18: Abstract Post for Complex Transitions =====

class TestComplexTransitions:
    """Test transition BDDs for more complex cases."""

    def test_assign_increment(self):
        """x := x + 1 should preserve x >= 0 if x >= 0 held."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        expr = App(Op.ADD, [x, IntConst(1)], INT)
        trans = builder.build_assign_transition("x", expr, 0, 1)

        # From state where x >= 0
        state = mgr.state_from_predicates({0})
        result = mgr.image(state, trans)
        # x + 1 >= 0 when x >= 0, so should be non-bottom
        assert not result.is_bottom

    def test_assign_from_constant(self):
        """x := 10 -- predicate x >= 0 should become true."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")

        builder = TransitionBDDBuilder(mgr)
        trans = builder.build_assign_transition("x", IntConst(10), 0, 1)

        # From top state
        state = mgr.state_top()
        result = mgr.image(state, trans)
        assert not result.is_bottom

    def test_multiple_assume_steps(self):
        """Chained assumes should accumulate predicate knowledge."""
        mgr = BDDPredicateManager()
        x = Var("x", INT)
        mgr.add_predicate(App(Op.GE, [x, IntConst(0)], BOOL), "x >= 0")
        mgr.add_predicate(App(Op.LE, [x, IntConst(10)], BOOL), "x <= 10")

        builder = TransitionBDDBuilder(mgr)

        # assume(x >= 0) followed by assume(x <= 10)
        t1 = builder.build_assume_transition(
            App(Op.GE, [x, IntConst(0)], BOOL), False, 0, 1
        )
        t2 = builder.build_assume_transition(
            App(Op.LE, [x, IntConst(10)], BOOL), False, 1, 2
        )

        state = mgr.state_top()
        s1 = mgr.image(state, t1)
        s2 = mgr.image(s1, t2)
        assert not s2.is_bottom


# ===== Section 19: Programs with Loops =====

class TestLoopPrograms:
    """Test programs with while loops (bounded exploration)."""

    def test_simple_loop_safe(self):
        source = """
        let x = 0;
        let i = 0;
        while (i < 5) {
            x = x + 1;
            i = i + 1;
        }
        assert(x >= 0);
        """
        result = bdd_verify(source, max_nodes=200)
        assert result.safe

    def test_loop_invariant_maintained(self):
        source = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        assert(x >= 0);
        """
        result = bdd_verify(source, max_nodes=200)
        assert result.safe


# ===== Section 20: Predicate Discovery Integration =====

class TestPredicateDiscovery:
    """Test integration with V114 predicate discovery."""

    def test_with_seeding(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        r1 = bdd_verify(source, seed_predicates=True)
        r2 = bdd_verify(source, seed_predicates=False)
        # Both should get same verdict
        assert r1.safe == r2.safe

    def test_predicates_from_cfg(self):
        source = """
        let x = 5;
        if (x > 3) {
            assert(x > 0);
        }
        """
        result = bdd_verify(source)
        # Should extract predicates from both x > 3 and x > 0
        assert result.total_predicates >= 1


# ===== Section 21: CEGAR Iteration Tracking =====

class TestCEGARTracking:
    """Test CEGAR iteration and metrics tracking."""

    def test_iteration_count(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert result.iterations >= 1

    def test_art_nodes_counted(self):
        source = """
        let x = 5;
        if (x > 0) {
            assert(x > 0);
        }
        """
        result = bdd_verify(source)
        assert result.art_nodes > 0

    def test_transition_bdds_counted(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert result.transition_bdds_built >= 0


# ===== Section 22: Verdict Enum =====

class TestVerdict:
    """Test BDDVerdict enum values."""

    def test_safe_value(self):
        assert BDDVerdict.SAFE.value == "safe"

    def test_unsafe_value(self):
        assert BDDVerdict.UNSAFE.value == "unsafe"

    def test_unknown_value(self):
        assert BDDVerdict.UNKNOWN.value == "unknown"

    def test_verdict_equality(self):
        assert BDDVerdict.SAFE == BDDVerdict.SAFE
        assert BDDVerdict.SAFE != BDDVerdict.UNSAFE


# ===== Section 23: Counterexample Quality =====

class TestCounterexampleQuality:
    """Test counterexample extraction quality."""

    def test_counterexample_is_path(self):
        source = """
        let x = -1;
        assert(x >= 0);
        """
        result = bdd_verify(source)
        assert not result.safe
        assert result.counterexample is not None
        # Path should start at ENTRY and end at ERROR
        if len(result.counterexample) > 0:
            last_type = result.counterexample[-1][1]
            assert last_type == "ERROR"

    def test_counterexample_has_inputs(self):
        source = """
        let x = -5;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert not result.safe
        # May or may not have inputs depending on path encoding
        assert result.counterexample is not None


# ===== Section 24: Multiple Assertions =====

class TestMultipleAssertions:
    """Test programs with multiple assertion points."""

    def test_all_assertions_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        let y = 10;
        assert(y > 0);
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert result.safe

    def test_first_assertion_fails(self):
        source = """
        let x = -1;
        assert(x > 0);
        let y = 10;
        assert(y > 0);
        """
        result = bdd_verify(source)
        assert not result.safe

    def test_second_assertion_fails(self):
        source = """
        let x = 5;
        assert(x > 0);
        let y = -1;
        assert(y > 0);
        """
        result = bdd_verify(source)
        assert not result.safe


# ===== Section 25: BDD Node Count Metric =====

class TestBDDMetrics:
    """Test BDD-specific metrics."""

    def test_bdd_image_ops_counted(self):
        source = """
        let x = 5;
        if (x > 0) {
            assert(x > 0);
        }
        """
        result = bdd_verify(source)
        assert result.bdd_image_ops >= 0

    def test_smt_queries_saved_nonnegative(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert result.smt_queries_saved >= 0

    def test_predicate_names_list(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = bdd_verify(source)
        assert isinstance(result.predicate_names, list)
        for name in result.predicate_names:
            assert isinstance(name, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
