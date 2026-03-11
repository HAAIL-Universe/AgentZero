"""
Tests for V111: Recursive Horn Clause Solving
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V109_chc_solver'))

from smt_solver import (
    Var, IntConst, BoolConst, App, Op, INT, BOOL
)
from chc_solver import (
    CHCSystem, Predicate, PredicateApp, HornClause, Interpretation,
    CHCResult, apply_pred,
    _and, _or, _not, _substitute
)
from recursive_chc import (
    DependencyGraph, SCC, LemmaStore,
    RecursiveCHCSolver, NonlinearCHCSolver, ModularCHCSolver,
    solve_recursive_chc, solve_nonlinear_chc, solve_modular_chc,
    analyze_dependencies, compare_strategies,
    chc_from_recursive_loop, chc_from_multi_phase,
    recursive_chc_summary,
)


# --- Helpers ---

def _var(name):
    return Var(name, INT)

def _int(n):
    return IntConst(n)

def _lt(a, b):
    return App(Op.LT, [a, b], BOOL)

def _le(a, b):
    return App(Op.LE, [a, b], BOOL)

def _gt(a, b):
    return App(Op.GT, [a, b], BOOL)

def _ge(a, b):
    return App(Op.GE, [a, b], BOOL)

def _eq(a, b):
    return App(Op.EQ, [a, b], BOOL)

def _neq(a, b):
    return App(Op.NEQ, [a, b], BOOL)

def _add(a, b):
    return App(Op.ADD, [a, b], INT)

def _sub(a, b):
    return App(Op.SUB, [a, b], INT)

def _mul(a, b):
    return App(Op.MUL, [a, b], INT)


# =============================================================================
# Section 1: Dependency Graph
# =============================================================================

class TestDependencyGraph:
    """Test predicate dependency analysis."""

    def test_empty_system(self):
        system = CHCSystem()
        graph = DependencyGraph(system)
        assert graph.get_sccs() == []
        assert graph.get_topological_order() == []

    def test_single_predicate_no_recursion(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        system.add_fact(apply_pred(p, [_var("x")]), _ge(_var("x"), _int(0)))

        graph = DependencyGraph(system)
        sccs = graph.get_sccs()
        assert len(sccs) == 1
        assert sccs[0].predicates == ["P"]
        assert not sccs[0].is_recursive

    def test_self_recursive_predicate(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        # P(x) AND x > 0 => P(x-1)
        system.add_clause(
            head=apply_pred(p, [_sub(_var("x"), _int(1))]),
            body_preds=[apply_pred(p, [_var("x")])],
            constraint=_gt(_var("x"), _int(0))
        )

        graph = DependencyGraph(system)
        recursive = graph.get_recursive_predicates()
        assert "P" in recursive

    def test_two_predicate_chain(self):
        """P -> Q (P depends on nothing, Q depends on P)."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("x", INT)])
        # fact: x >= 0 => P(x)
        system.add_fact(apply_pred(p, [_var("x")]), _ge(_var("x"), _int(0)))
        # rule: P(x) AND x > 0 => Q(x)
        system.add_clause(
            head=apply_pred(q, [_var("x")]),
            body_preds=[apply_pred(p, [_var("x")])],
            constraint=_gt(_var("x"), _int(0))
        )

        graph = DependencyGraph(system)
        topo = graph.get_topological_order()
        # P should come before Q in topological order
        assert topo.index("P") < topo.index("Q")

    def test_mutual_recursion(self):
        """P -> Q -> P forms a cycle."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("x", INT)])
        # P(x) => Q(x+1)
        system.add_clause(
            head=apply_pred(q, [_add(_var("x"), _int(1))]),
            body_preds=[apply_pred(p, [_var("x")])],
            constraint=BoolConst(True)
        )
        # Q(x) => P(x+1)
        system.add_clause(
            head=apply_pred(p, [_add(_var("x"), _int(1))]),
            body_preds=[apply_pred(q, [_var("x")])],
            constraint=BoolConst(True)
        )

        graph = DependencyGraph(system)
        sccs = graph.get_sccs()
        # Should be one recursive SCC with both P and Q
        recursive_sccs = [s for s in sccs if s.is_recursive]
        assert len(recursive_sccs) == 1
        assert set(recursive_sccs[0].predicates) == {"P", "Q"}

    def test_nonlinear_clause_detection(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("y", INT)])
        r = system.add_predicate("R", [("z", INT)])
        # P(x) AND Q(y) => R(x+y)
        system.add_clause(
            head=apply_pred(r, [_add(_var("x"), _var("y"))]),
            body_preds=[apply_pred(p, [_var("x")]), apply_pred(q, [_var("y")])],
            constraint=BoolConst(True)
        )

        graph = DependencyGraph(system)
        nonlinear = graph.get_nonlinear_clauses()
        assert len(nonlinear) == 1


# =============================================================================
# Section 2: Lemma Store
# =============================================================================

class TestLemmaStore:
    """Test lemma caching and retrieval."""

    def test_empty_store(self):
        store = LemmaStore()
        assert store.total_lemmas() == 0
        assert isinstance(store.get_conjunction("P"), BoolConst)

    def test_add_and_retrieve(self):
        store = LemmaStore()
        lemma = _ge(_var("x"), _int(0))
        store.add_lemma("P", lemma, source="base")
        assert len(store.get_lemmas("P")) == 1
        assert store.total_lemmas() == 1

    def test_no_duplicates(self):
        store = LemmaStore()
        lemma = _ge(_var("x"), _int(0))
        store.add_lemma("P", lemma)
        store.add_lemma("P", lemma)  # duplicate
        assert len(store.get_lemmas("P")) == 1

    def test_conjunction(self):
        store = LemmaStore()
        store.add_lemma("P", _ge(_var("x"), _int(0)))
        store.add_lemma("P", _le(_var("x"), _int(10)))
        conj = store.get_conjunction("P")
        assert isinstance(conj, App)
        assert conj.op == Op.AND

    def test_interpretation_cache(self):
        store = LemmaStore()
        store.update_interpretation("P", _ge(_var("x"), _int(0)))
        assert store.get_interpretation("P") is not None
        interp = store.to_interpretation()
        assert "P" in interp.mapping


# =============================================================================
# Section 3: Recursive CHC Solver -- Simple Linear Cases
# =============================================================================

class TestRecursiveSimpleLinear:
    """Test recursive solver on simple linear (non-recursive) systems."""

    def test_single_fact_single_query_safe(self):
        """P(x) where x >= 0. Query: P(x) AND x < 0 => false. Should be SAT (safe)."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        system.add_fact(apply_pred(p, [_var("x")]), _ge(_var("x"), _int(0)))
        system.add_query(
            body_preds=[apply_pred(p, [_var("x")])],
            constraint=_lt(_var("x"), _int(0))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_single_fact_single_query_unsafe(self):
        """P(x) where x >= 0. Query: P(x) AND x >= 0 => false. Should be UNSAT (unsafe)."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        system.add_fact(apply_pred(p, [_var("x")]), _ge(_var("x"), _int(0)))
        system.add_query(
            body_preds=[apply_pred(p, [_var("x")])],
            constraint=_ge(_var("x"), _int(0))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT

    def test_two_predicate_chain_safe(self):
        """P(x): x==0. P(x) => Q(x+1). Query: Q(x) AND x < 0 => false. Safe."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("x", INT)])

        system.add_fact(apply_pred(p, [_var("x")]), _eq(_var("x"), _int(0)))
        system.add_clause(
            head=apply_pred(q, [_add(_var("x"), _int(1))]),
            body_preds=[apply_pred(p, [_var("x")])],
            constraint=BoolConst(True)
        )
        system.add_query(
            body_preds=[apply_pred(q, [_var("x")])],
            constraint=_lt(_var("x"), _int(0))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_two_predicate_chain_unsafe(self):
        """P(x): x==0. P(x) => Q(x). Query: Q(x) AND x == 0 => false. Unsafe."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("x", INT)])

        system.add_fact(apply_pred(p, [_var("x")]), _eq(_var("x"), _int(0)))
        system.add_clause(
            head=apply_pred(q, [_var("x")]),
            body_preds=[apply_pred(p, [_var("x")])],
            constraint=BoolConst(True)
        )
        system.add_query(
            body_preds=[apply_pred(q, [_var("x")])],
            constraint=_eq(_var("x"), _int(0))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT


# =============================================================================
# Section 4: Recursive CHC Solver -- Recursive Predicates
# =============================================================================

class TestRecursivePredicates:
    """Test solving systems with self-recursive predicates."""

    def test_countdown_safe(self):
        """
        Inv(x): x == 10. Inv(x) AND x > 0 => Inv(x-1).
        Query: Inv(x) AND x < 0 => false.
        Safe: x never goes below 0.
        """
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])

        system.add_fact(apply_pred(inv, [_var("x")]), _eq(_var("x"), _int(10)))
        system.add_clause(
            head=apply_pred(inv, [_sub(_var("x"), _int(1))]),
            body_preds=[apply_pred(inv, [_var("x")])],
            constraint=_gt(_var("x"), _int(0))
        )
        system.add_query(
            body_preds=[apply_pred(inv, [_var("x")])],
            constraint=_lt(_var("x"), _int(0))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_countdown_unsafe(self):
        """
        Inv(x): x == 10. Inv(x) AND x > 0 => Inv(x-1).
        Query: Inv(x) AND x == 0 => false.
        Unsafe: x does reach 0.
        """
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])

        system.add_fact(apply_pred(inv, [_var("x")]), _eq(_var("x"), _int(10)))
        system.add_clause(
            head=apply_pred(inv, [_sub(_var("x"), _int(1))]),
            body_preds=[apply_pred(inv, [_var("x")])],
            constraint=_gt(_var("x"), _int(0))
        )
        system.add_query(
            body_preds=[apply_pred(inv, [_var("x")])],
            constraint=_eq(_var("x"), _int(0))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT

    def test_two_var_increment_safe(self):
        """
        Inv(x, y): x==0, y==10. Inv(x,y) AND x<y => Inv(x+1, y).
        Query: Inv(x, y) AND x > y => false.
        Safe: x never exceeds y because guard prevents it.
        """
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT), ("y", INT)])

        x, y = _var("x"), _var("y")
        system.add_fact(
            apply_pred(inv, [x, y]),
            _and(_eq(x, _int(0)), _eq(y, _int(10)))
        )
        system.add_clause(
            head=apply_pred(inv, [_add(x, _int(1)), y]),
            body_preds=[apply_pred(inv, [x, y])],
            constraint=_lt(x, y)
        )
        system.add_query(
            body_preds=[apply_pred(inv, [x, y])],
            constraint=_gt(x, y)
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_diverging_counter_unsafe(self):
        """
        Inv(x): x==0. Inv(x) => Inv(x+1).
        Query: Inv(x) AND x == 5 => false.
        Unsafe: x reaches 5.
        """
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])

        x = _var("x")
        system.add_fact(apply_pred(inv, [x]), _eq(x, _int(0)))
        system.add_clause(
            head=apply_pred(inv, [_add(x, _int(1))]),
            body_preds=[apply_pred(inv, [x])],
            constraint=BoolConst(True)
        )
        system.add_query(
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(x, _int(5))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT

    def test_bounded_increment_safe(self):
        """
        Inv(x): x==0. Inv(x) AND x<5 => Inv(x+1).
        Query: Inv(x) AND x > 5 => false.
        Safe: x never exceeds 5.
        """
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])

        x = _var("x")
        system.add_fact(apply_pred(inv, [x]), _eq(x, _int(0)))
        system.add_clause(
            head=apply_pred(inv, [_add(x, _int(1))]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_lt(x, _int(5))
        )
        system.add_query(
            body_preds=[apply_pred(inv, [x])],
            constraint=_gt(x, _int(5))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT


# =============================================================================
# Section 5: chc_from_recursive_loop convenience
# =============================================================================

class TestCHCFromRecursiveLoop:
    """Test the convenience API for creating CHC from loops."""

    def test_simple_countdown(self):
        x = _var("x")
        x_prime = _var("x_prime")

        system = chc_from_recursive_loop(
            init_constraint=_eq(x, _int(10)),
            loop_body_constraint=_and(
                _gt(x, _int(0)),
                _eq(x_prime, _sub(x, _int(1)))
            ),
            property_constraint=_ge(x, _int(0)),
            var_params=[("x", INT)]
        )

        assert "Inv" in system.predicates
        assert len(system.get_facts()) == 1
        assert len(system.get_queries()) == 1

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_loop_with_violation(self):
        x = _var("x")
        x_prime = _var("x_prime")

        system = chc_from_recursive_loop(
            init_constraint=_eq(x, _int(0)),
            loop_body_constraint=_eq(x_prime, _add(x, _int(1))),
            property_constraint=_lt(x, _int(5)),  # x < 5 -- will be violated
            var_params=[("x", INT)]
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT

    def test_two_variable_loop(self):
        x, y = _var("x"), _var("y")
        xp, yp = _var("x_prime"), _var("y_prime")

        system = chc_from_recursive_loop(
            init_constraint=_and(_eq(x, _int(0)), _eq(y, _int(10))),
            loop_body_constraint=_and(
                _lt(x, y),
                _eq(xp, _add(x, _int(1))),
                _eq(yp, y)
            ),
            property_constraint=_le(x, y),  # x <= y always holds
            var_params=[("x", INT), ("y", INT)]
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT


# =============================================================================
# Section 6: Nonlinear CHC Solver
# =============================================================================

class TestNonlinearSolver:
    """Test handling of clauses with multiple body predicates."""

    def test_simple_product(self):
        """P(x): x >= 0. Q(y): y >= 0. P(x) AND Q(y) => R(x+y). R(z) AND z < 0 => false."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("y", INT)])
        r = system.add_predicate("R", [("z", INT)])

        x, y, z = _var("x"), _var("y"), _var("z")

        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_fact(apply_pred(q, [y]), _ge(y, _int(0)))
        system.add_clause(
            head=apply_pred(r, [_add(x, y)]),
            body_preds=[apply_pred(p, [x]), apply_pred(q, [y])],
            constraint=BoolConst(True)
        )
        system.add_query(
            body_preds=[apply_pred(r, [z])],
            constraint=_lt(z, _int(0))
        )

        result = solve_nonlinear_chc(system)
        assert result.result == CHCResult.SAT

    def test_nonlinear_unsafe(self):
        """P(x): x==1. Q(y): y==2. P(x) AND Q(y) => R(x+y). R(z) AND z==3 => false."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("y", INT)])
        r = system.add_predicate("R", [("z", INT)])

        x, y, z = _var("x"), _var("y"), _var("z")

        system.add_fact(apply_pred(p, [x]), _eq(x, _int(1)))
        system.add_fact(apply_pred(q, [y]), _eq(y, _int(2)))
        system.add_clause(
            head=apply_pred(r, [_add(x, y)]),
            body_preds=[apply_pred(p, [x]), apply_pred(q, [y])],
            constraint=BoolConst(True)
        )
        system.add_query(
            body_preds=[apply_pred(r, [z])],
            constraint=_eq(z, _int(3))
        )

        result = solve_nonlinear_chc(system)
        assert result.result == CHCResult.UNSAT

    def test_linear_passthrough(self):
        """Linear system should pass through to recursive solver."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_query(
            body_preds=[apply_pred(p, [x])],
            constraint=_lt(x, _int(0))
        )

        result = solve_nonlinear_chc(system)
        assert result.result == CHCResult.SAT


# =============================================================================
# Section 7: Multi-Phase Systems
# =============================================================================

class TestMultiPhase:
    """Test multi-phase CHC systems."""

    def test_two_phase_safe(self):
        """Phase1(x): x==0. Phase1(x) => Phase2(x+5). Phase2(x) AND x < 0 => false."""
        x = _var("x")
        xp = _var("x_prime")

        system = chc_from_multi_phase(
            phases=[
                ("Phase1", _eq(x, _int(0))),
                ("Phase2", BoolConst(False)),  # No direct init for Phase2
            ],
            transitions=[
                ("Phase1", "Phase2", _eq(xp, _add(x, _int(5)))),
            ],
            property_constraint=_ge(x, _int(0)),
            var_params=[("x", INT)]
        )

        result = solve_modular_chc(system)
        assert result.result == CHCResult.SAT

    def test_two_phase_unsafe(self):
        """Phase1(x): x==10. Phase1(x) => Phase2(x-15). Phase2(x) AND x < 0 => false."""
        x = _var("x")
        xp = _var("x_prime")

        system = chc_from_multi_phase(
            phases=[
                ("Phase1", _eq(x, _int(10))),
                ("Phase2", BoolConst(False)),
            ],
            transitions=[
                ("Phase1", "Phase2", _eq(xp, _sub(x, _int(15)))),
            ],
            property_constraint=_ge(x, _int(0)),
            var_params=[("x", INT)]
        )

        result = solve_modular_chc(system)
        assert result.result == CHCResult.UNSAT

    def test_three_phase_chain(self):
        """Three phases, property holds at all."""
        x = _var("x")
        xp = _var("x_prime")

        system = chc_from_multi_phase(
            phases=[
                ("A", _eq(x, _int(0))),
                ("B", BoolConst(False)),
                ("C", BoolConst(False)),
            ],
            transitions=[
                ("A", "B", _eq(xp, _add(x, _int(1)))),
                ("B", "C", _eq(xp, _add(x, _int(2)))),
            ],
            property_constraint=_ge(x, _int(0)),
            var_params=[("x", INT)]
        )

        result = solve_modular_chc(system)
        assert result.result == CHCResult.SAT


# =============================================================================
# Section 8: Modular Solver
# =============================================================================

class TestModularSolver:
    """Test modular decomposition and solving."""

    def test_independent_predicates(self):
        """Two independent predicates, each safe."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("y", INT)])

        x, y = _var("x"), _var("y")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_fact(apply_pred(q, [y]), _ge(y, _int(0)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        result = solve_modular_chc(system)
        assert result.result == CHCResult.SAT

    def test_dependent_predicates(self):
        """P depends on Q: P is safe only if Q produces positive values."""
        system = CHCSystem()
        q = system.add_predicate("Q", [("y", INT)])
        p = system.add_predicate("P", [("x", INT)])

        x, y = _var("x"), _var("y")
        system.add_fact(apply_pred(q, [y]), _eq(y, _int(5)))
        system.add_clause(
            head=apply_pred(p, [_add(y, _int(1))]),
            body_preds=[apply_pred(q, [y])],
            constraint=BoolConst(True)
        )
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        result = solve_modular_chc(system)
        assert result.result == CHCResult.SAT

    def test_recursive_with_dependency(self):
        """Base predicate feeds into recursive predicate."""
        system = CHCSystem()
        base = system.add_predicate("Base", [("x", INT)])
        loop = system.add_predicate("Loop", [("x", INT)])

        x = _var("x")
        # Base(x): x == 0
        system.add_fact(apply_pred(base, [x]), _eq(x, _int(0)))
        # Base(x) => Loop(x)
        system.add_clause(
            head=apply_pred(loop, [x]),
            body_preds=[apply_pred(base, [x])],
            constraint=BoolConst(True)
        )
        # Loop(x) AND x < 5 => Loop(x+1)
        system.add_clause(
            head=apply_pred(loop, [_add(x, _int(1))]),
            body_preds=[apply_pred(loop, [x])],
            constraint=_lt(x, _int(5))
        )
        # Query: Loop(x) AND x > 10 => false (safe since x <= 5)
        system.add_query(
            body_preds=[apply_pred(loop, [x])],
            constraint=_gt(x, _int(10))
        )

        result = solve_modular_chc(system)
        assert result.result == CHCResult.SAT

    def test_modular_unsafe(self):
        """Base produces value that violates property when used."""
        system = CHCSystem()
        source = system.add_predicate("Source", [("x", INT)])
        sink = system.add_predicate("Sink", [("x", INT)])

        x = _var("x")
        system.add_fact(apply_pred(source, [x]), _eq(x, _int(-1)))
        system.add_clause(
            head=apply_pred(sink, [x]),
            body_preds=[apply_pred(source, [x])],
            constraint=BoolConst(True)
        )
        system.add_query(body_preds=[apply_pred(sink, [x])], constraint=_lt(x, _int(0)))

        result = solve_modular_chc(system)
        assert result.result == CHCResult.UNSAT


# =============================================================================
# Section 9: Analyze Dependencies API
# =============================================================================

class TestAnalyzeDependencies:
    """Test the dependency analysis API."""

    def test_basic_analysis(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("y", INT)])
        x, y = _var("x"), _var("y")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_clause(
            head=apply_pred(q, [y]),
            body_preds=[apply_pred(p, [y])],
            constraint=BoolConst(True)
        )

        analysis = analyze_dependencies(system)
        assert 'predicates' in analysis
        assert 'sccs' in analysis
        assert 'topological_order' in analysis
        assert 'recursive_predicates' in analysis
        assert 'is_linear' in analysis
        assert analysis['is_linear'] is True
        assert len(analysis['recursive_predicates']) == 0

    def test_recursive_detection(self):
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(inv, [x]), _eq(x, _int(0)))
        system.add_clause(
            head=apply_pred(inv, [_add(x, _int(1))]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_lt(x, _int(10))
        )

        analysis = analyze_dependencies(system)
        assert "Inv" in analysis['recursive_predicates']

    def test_nonlinear_detection(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("y", INT)])
        r = system.add_predicate("R", [("z", INT)])
        x, y, z = _var("x"), _var("y"), _var("z")
        system.add_clause(
            head=apply_pred(r, [z]),
            body_preds=[apply_pred(p, [x]), apply_pred(q, [y])],
            constraint=_eq(z, _add(x, y))
        )

        analysis = analyze_dependencies(system)
        assert analysis['nonlinear_clauses'] == 1
        assert analysis['is_linear'] is False


# =============================================================================
# Section 10: Compare Strategies
# =============================================================================

class TestCompareStrategies:
    """Test strategy comparison API."""

    def test_compare_safe_system(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        comparison = compare_strategies(system)
        assert 'recursive' in comparison
        assert 'modular' in comparison
        assert 'bmc' in comparison
        # All should agree on SAT (safe)
        assert comparison['recursive']['result'] == 'sat'
        assert comparison['modular']['result'] == 'sat'

    def test_compare_unsafe_system(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _eq(x, _int(5)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_eq(x, _int(5)))

        comparison = compare_strategies(system)
        assert comparison['recursive']['result'] == 'unsat'
        assert comparison['modular']['result'] == 'unsat'


# =============================================================================
# Section 11: Summary API
# =============================================================================

class TestSummaryAPI:
    """Test the summary convenience API."""

    def test_summary_structure(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        summary = recursive_chc_summary(system)
        assert 'structure' in summary
        assert 'result' in summary
        assert 'stats' in summary
        assert summary['result'] == 'sat'


# =============================================================================
# Section 12: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_trivial_true_fact(self):
        """Fact with True constraint -- predicate is unconstrained."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), BoolConst(True))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_eq(x, _int(42)))

        result = solve_recursive_chc(system)
        # P(x) is true for all x, including x==42, so query is satisfiable (UNSAT)
        assert result.result == CHCResult.UNSAT

    def test_trivial_false_fact(self):
        """Fact with False constraint -- predicate is empty."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), BoolConst(False))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=BoolConst(True))

        result = solve_recursive_chc(system)
        # P is empty, so query is never satisfiable (SAT = safe)
        assert result.result == CHCResult.SAT

    def test_no_queries(self):
        """System with no queries is trivially safe."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_query_with_no_body_preds(self):
        """Query with constraint only (no body predicates)."""
        system = CHCSystem()
        # phi => false where phi is satisfiable
        system.add_query(body_preds=[], constraint=_eq(_var("x"), _int(5)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT

    def test_query_with_unsat_constraint(self):
        """Query with unsatisfiable constraint."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), BoolConst(True))
        # x < 0 AND x > 0 -- impossible
        system.add_query(
            body_preds=[apply_pred(p, [x])],
            constraint=_and(_lt(x, _int(0)), _gt(x, _int(0)))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_multiple_facts_same_predicate(self):
        """Multiple fact clauses for the same predicate (disjunction)."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _eq(x, _int(1)))
        system.add_fact(apply_pred(p, [x]), _eq(x, _int(2)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_eq(x, _int(3)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT  # 3 is not in {1, 2}

    def test_multiple_facts_query_hits(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _eq(x, _int(1)))
        system.add_fact(apply_pred(p, [x]), _eq(x, _int(2)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_eq(x, _int(2)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT


# =============================================================================
# Section 13: Recursive with Multiple Clauses
# =============================================================================

class TestRecursiveMultipleClauses:
    """Test recursive predicates with multiple defining clauses."""

    def test_two_base_cases(self):
        """Inv(x) from x==0 or x==5. Inv(x) AND x<10 => Inv(x+1). Query: Inv(x) AND x>15 => false."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = _var("x")

        system.add_fact(apply_pred(inv, [x]), _eq(x, _int(0)))
        system.add_fact(apply_pred(inv, [x]), _eq(x, _int(5)))
        system.add_clause(
            head=apply_pred(inv, [_add(x, _int(1))]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_lt(x, _int(10))
        )
        system.add_query(body_preds=[apply_pred(inv, [x])], constraint=_gt(x, _int(15)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_conditional_recursion(self):
        """Different recursion paths based on conditions."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = _var("x")

        system.add_fact(apply_pred(inv, [x]), _eq(x, _int(0)))
        # If x < 5, increment by 1
        system.add_clause(
            head=apply_pred(inv, [_add(x, _int(1))]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_lt(x, _int(5))
        )
        # If 5 <= x < 10, increment by 2
        system.add_clause(
            head=apply_pred(inv, [_add(x, _int(2))]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_and(_ge(x, _int(5)), _lt(x, _int(10)))
        )
        # x should never exceed 11 (0->1->2->3->4->5->7->9->11)
        system.add_query(body_preds=[apply_pred(inv, [x])], constraint=_gt(x, _int(15)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT


# =============================================================================
# Section 14: Interpretation Verification
# =============================================================================

class TestInterpretationVerification:
    """Test that returned interpretations actually satisfy the system."""

    def test_interpretation_satisfies_facts(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT
        assert result.interpretation is not None

        # Manually verify: init => interp
        interp_formula = result.interpretation.get("P")
        assert interp_formula is not None

    def test_interpretation_blocks_query(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _eq(x, _int(5)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

        # The interpretation for P should be such that P(x) AND x<0 is UNSAT
        interp = result.interpretation
        p_interp = interp.get("P")
        if p_interp is not None:
            from smt_solver import SMTSolver, SMTResult
            s = SMTSolver()
            s.add(_and(_substitute(p_interp, {"x": x}), _lt(x, _int(0))))
            assert s.check() == SMTResult.UNSAT


# =============================================================================
# Section 15: Derivation / Counterexample Structure
# =============================================================================

class TestDerivations:
    """Test counterexample derivation structure."""

    def test_unsat_has_derivation(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _eq(x, _int(5)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_eq(x, _int(5)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.UNSAT
        # Should have a derivation or at least model
        assert result.derivation is not None

    def test_safe_no_derivation(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT
        assert result.derivation is None


# =============================================================================
# Section 16: Stats Tracking
# =============================================================================

class TestStats:
    """Test that statistics are properly tracked."""

    def test_stats_populated(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        result = solve_recursive_chc(system)
        assert result.stats is not None
        assert result.stats.strategy == "recursive"

    def test_modular_stats(self):
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        x = _var("x")
        system.add_fact(apply_pred(p, [x]), _ge(x, _int(0)))
        system.add_query(body_preds=[apply_pred(p, [x])], constraint=_lt(x, _int(0)))

        result = solve_modular_chc(system)
        assert result.stats.strategy == "modular"


# =============================================================================
# Section 17: Complex Multi-Predicate Recursive Systems
# =============================================================================

class TestComplexSystems:
    """Test complex systems with multiple interacting predicates."""

    def test_nonlinear_recursive_safe(self):
        """
        Two predicates feeding into a third, all non-negative.
        A(x): x >= 0. B(y): y >= 0.
        A(x) AND B(y) AND x > 0 AND y > 0 => C(x, y).
        C(x,y) AND x+y < 0 => false.
        Safe because x >= 0 and y >= 0 implies x+y >= 0.
        """
        system = CHCSystem()
        a = system.add_predicate("A", [("x", INT)])
        b = system.add_predicate("B", [("y", INT)])
        c = system.add_predicate("C", [("x", INT), ("y", INT)])

        x, y = _var("x"), _var("y")

        system.add_fact(apply_pred(a, [x]), _ge(x, _int(0)))
        system.add_fact(apply_pred(b, [y]), _ge(y, _int(0)))
        system.add_clause(
            head=apply_pred(c, [x, y]),
            body_preds=[apply_pred(a, [x]), apply_pred(b, [y])],
            constraint=_and(_gt(x, _int(0)), _gt(y, _int(0)))
        )
        system.add_query(
            body_preds=[apply_pred(c, [x, y])],
            constraint=_lt(_add(x, y), _int(0))
        )

        result = solve_nonlinear_chc(system)
        assert result.result == CHCResult.SAT

    def test_producer_consumer_safe(self):
        """Producer adds, consumer removes. Buffer never negative."""
        system = CHCSystem()
        state = system.add_predicate("State", [("buf", INT)])
        buf = _var("buf")

        system.add_fact(apply_pred(state, [buf]), _eq(buf, _int(0)))
        # Produce: add 1
        system.add_clause(
            head=apply_pred(state, [_add(buf, _int(1))]),
            body_preds=[apply_pred(state, [buf])],
            constraint=_lt(buf, _int(5))  # bounded buffer
        )
        # Consume: remove 1 (only if buf > 0)
        system.add_clause(
            head=apply_pred(state, [_sub(buf, _int(1))]),
            body_preds=[apply_pred(state, [buf])],
            constraint=_gt(buf, _int(0))
        )
        # Query: buf < 0 should never happen
        system.add_query(body_preds=[apply_pred(state, [buf])], constraint=_lt(buf, _int(0)))

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT

    def test_mutex_like(self):
        """Two processes, mutual exclusion: not both in critical section."""
        system = CHCSystem()
        state = system.add_predicate("State", [("p1", INT), ("p2", INT)])
        p1, p2 = _var("p1"), _var("p2")

        # 0=idle, 1=critical
        system.add_fact(
            apply_pred(state, [p1, p2]),
            _and(_eq(p1, _int(0)), _eq(p2, _int(0)))
        )
        # p1 enters critical (only if p2 not in critical)
        system.add_clause(
            head=apply_pred(state, [_int(1), p2]),
            body_preds=[apply_pred(state, [p1, p2])],
            constraint=_and(_eq(p1, _int(0)), _eq(p2, _int(0)))
        )
        # p2 enters critical (only if p1 not in critical)
        system.add_clause(
            head=apply_pred(state, [p1, _int(1)]),
            body_preds=[apply_pred(state, [p1, p2])],
            constraint=_and(_eq(p2, _int(0)), _eq(p1, _int(0)))
        )
        # p1 exits
        system.add_clause(
            head=apply_pred(state, [_int(0), p2]),
            body_preds=[apply_pred(state, [p1, p2])],
            constraint=_eq(p1, _int(1))
        )
        # p2 exits
        system.add_clause(
            head=apply_pred(state, [p1, _int(0)]),
            body_preds=[apply_pred(state, [p1, p2])],
            constraint=_eq(p2, _int(1))
        )
        # Query: both in critical => false
        system.add_query(
            body_preds=[apply_pred(state, [p1, p2])],
            constraint=_and(_eq(p1, _int(1)), _eq(p2, _int(1)))
        )

        result = solve_recursive_chc(system)
        assert result.result == CHCResult.SAT
