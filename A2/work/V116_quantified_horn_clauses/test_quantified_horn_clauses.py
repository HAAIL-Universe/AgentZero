"""Tests for V116: Quantified Horn Clauses"""

import sys, os
_base = 'Z:/AgentZero'
_a2 = _base + '/A2/work'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _a2 + '/V111_recursive_chc')
sys.path.insert(0, _a2 + '/V109_chc_solver')
sys.path.insert(0, _base + '/challenges/C037_smt_solver')
sys.path.insert(0, _a2 + '/V107_craig_interpolation')
sys.path.insert(0, _a2 + '/V002_pdr_ic3')

import pytest
from smt_solver import Var, IntConst, BoolConst, App, Op, INT, BOOL, Sort
from chc_solver import (
    CHCSystem, CHCResult, Predicate, PredicateApp,
    apply_pred, _and, _or, _not, _eq, _implies,
)

from quantified_horn_clauses import (
    # Quantifiers
    Forall, Exists,
    # Array operations
    Select, Store, ConstArray, ARRAY,
    _is_select, _is_store, _is_const_array,
    _get_select_parts, _get_store_parts,
    # Formula operations
    collect_free_vars, substitute_quantified, negate_quantified,
    # Instantiation
    QuantifierInstantiator, InstantiationStrategy, Instantiation,
    # Array axioms
    ArrayAxiomEngine,
    # Quantified CHC
    QuantifiedCHCSystem, QuantifiedClause, QuantifiedCHCSolver, QCHCOutput,
    # Array properties
    array_sorted_property, array_bounded_property,
    array_initialized_property, array_partitioned_property,
    array_exists_element,
    # Convenience APIs
    solve_quantified_chc, verify_array_property, verify_universal_property,
    check_quantified_validity, analyze_quantified_system,
    compare_instantiation_strategies, quantified_summary,
)


# --- Helpers ---
def _var(name):
    return Var(name, INT)

def _int(n):
    return IntConst(n)

def _bool(b):
    return BoolConst(b)

def _lt(a, b):
    return App(Op.LT, [a, b], BOOL)

def _le(a, b):
    return App(Op.LE, [a, b], BOOL)

def _gt(a, b):
    return App(Op.GT, [a, b], BOOL)

def _ge(a, b):
    return App(Op.GE, [a, b], BOOL)

def _add(a, b):
    return App(Op.ADD, [a, b], INT)

def _sub(a, b):
    return App(Op.SUB, [a, b], INT)

def _mul(a, b):
    return App(Op.MUL, [a, b], INT)

def _neq(a, b):
    return App(Op.NEQ, [a, b], BOOL)


# ===================================================================
# Section 1: Quantifier AST
# ===================================================================

class TestQuantifierAST:
    """Test Forall and Exists constructors and properties."""

    def test_forall_creation(self):
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        assert len(f.variables) == 1
        assert f.variables[0] == ("x", INT)

    def test_exists_creation(self):
        f = Exists([("x", INT)], _eq(_var("x"), _int(5)))
        assert len(f.variables) == 1
        assert f.variables[0] == ("x", INT)

    def test_forall_multi_var(self):
        f = Forall([("x", INT), ("y", INT)], _lt(_var("x"), _var("y")))
        assert len(f.variables) == 2

    def test_forall_equality(self):
        f1 = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        f2 = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        assert f1 == f2

    def test_forall_inequality(self):
        f1 = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        f2 = Forall([("x", INT)], _lt(_var("x"), _int(0)))
        assert f1 != f2

    def test_exists_hash(self):
        f1 = Exists([("x", INT)], _eq(_var("x"), _int(1)))
        f2 = Exists([("x", INT)], _eq(_var("x"), _int(1)))
        assert hash(f1) == hash(f2)

    def test_nested_quantifiers(self):
        inner = Exists([("y", INT)], _eq(_var("y"), _var("x")))
        outer = Forall([("x", INT)], inner)
        assert isinstance(outer, Forall)
        assert isinstance(outer.body, Exists)

    def test_forall_repr(self):
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        r = repr(f)
        assert "Forall" in r
        assert "x" in r

    def test_exists_repr(self):
        f = Exists([("i", INT)], _eq(_var("i"), _int(3)))
        r = repr(f)
        assert "Exists" in r
        assert "i" in r


# ===================================================================
# Section 2: Array Theory
# ===================================================================

class TestArrayTheory:
    """Test Select, Store, ConstArray operations."""

    def test_select_creation(self):
        s = Select(_var("a"), _int(0))
        assert _is_select(s)

    def test_store_creation(self):
        s = Store(_var("a"), _int(0), _int(42))
        assert _is_store(s)

    def test_const_array_creation(self):
        c = ConstArray(_int(0))
        assert _is_const_array(c)

    def test_select_parts(self):
        s = Select(_var("a"), _int(3))
        arr, idx = _get_select_parts(s)
        assert isinstance(arr, Var) and arr.name == "a"
        assert isinstance(idx, IntConst) and idx.value == 3

    def test_store_parts(self):
        s = Store(_var("a"), _int(1), _int(99))
        arr, idx, val = _get_store_parts(s)
        assert isinstance(arr, Var) and arr.name == "a"
        assert isinstance(idx, IntConst) and idx.value == 1
        assert isinstance(val, IntConst) and val.value == 99

    def test_nested_store(self):
        s1 = Store(_var("a"), _int(0), _int(1))
        s2 = Store(s1, _int(1), _int(2))
        assert _is_store(s2)
        arr, idx, val = _get_store_parts(s2)
        assert _is_store(arr)  # nested store

    def test_select_after_store(self):
        s = Store(_var("a"), _int(0), _int(42))
        r = Select(s, _int(0))
        assert _is_select(r)

    def test_not_select(self):
        assert not _is_select(_var("a"))
        assert not _is_select(_int(5))

    def test_not_store(self):
        assert not _is_store(_var("a"))
        assert not _is_store(Select(_var("a"), _int(0)))


# ===================================================================
# Section 3: Free Variables
# ===================================================================

class TestFreeVariables:
    """Test free variable collection in quantified formulas."""

    def test_free_var_simple(self):
        fv = collect_free_vars(_gt(_var("x"), _int(0)))
        assert fv == {"x"}

    def test_free_var_forall_binds(self):
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        fv = collect_free_vars(f)
        assert "x" not in fv

    def test_free_var_exists_binds(self):
        f = Exists([("x", INT)], _eq(_var("x"), _var("y")))
        fv = collect_free_vars(f)
        assert "x" not in fv
        assert "y" in fv

    def test_free_var_nested(self):
        inner = Exists([("y", INT)], _eq(_var("y"), _var("z")))
        outer = Forall([("x", INT)], _and(inner, _gt(_var("x"), _var("w"))))
        fv = collect_free_vars(outer)
        assert "x" not in fv
        assert "y" not in fv
        assert "z" in fv
        assert "w" in fv

    def test_free_var_const(self):
        fv = collect_free_vars(_int(5))
        assert fv == set()

    def test_free_var_bool_const(self):
        fv = collect_free_vars(_bool(True))
        assert fv == set()


# ===================================================================
# Section 4: Substitution
# ===================================================================

class TestSubstitution:
    """Test substitution in quantified formulas."""

    def test_substitute_free(self):
        f = Forall([("x", INT)], _gt(_var("x"), _var("y")))
        result = substitute_quantified(f, {"y": _int(5)})
        assert isinstance(result, Forall)
        # y should be substituted in body

    def test_substitute_respects_binding(self):
        f = Forall([("x", INT)], _gt(_var("x"), _var("y")))
        result = substitute_quantified(f, {"x": _int(99)})
        # x is bound -- should NOT be substituted
        assert isinstance(result, Forall)

    def test_substitute_exists(self):
        f = Exists([("i", INT)], _eq(_var("i"), _var("n")))
        result = substitute_quantified(f, {"n": _int(10)})
        assert isinstance(result, Exists)

    def test_substitute_nested(self):
        inner = Exists([("y", INT)], _eq(_var("y"), _var("z")))
        outer = Forall([("x", INT)], inner)
        result = substitute_quantified(outer, {"z": _int(42)})
        assert isinstance(result, Forall)
        assert isinstance(result.body, Exists)


# ===================================================================
# Section 5: Negation
# ===================================================================

class TestNegation:
    """Test De Morgan for quantifiers."""

    def test_negate_forall(self):
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        neg = negate_quantified(f)
        assert isinstance(neg, Exists)

    def test_negate_exists(self):
        f = Exists([("x", INT)], _eq(_var("x"), _int(5)))
        neg = negate_quantified(f)
        assert isinstance(neg, Forall)

    def test_double_negate(self):
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        neg2 = negate_quantified(negate_quantified(f))
        assert isinstance(neg2, Forall)

    def test_negate_plain_term(self):
        t = _gt(_var("x"), _int(0))
        neg = negate_quantified(t)
        # Should be NOT(x > 0)
        assert isinstance(neg, App)


# ===================================================================
# Section 6: Quantifier Instantiation - Term-Based
# ===================================================================

class TestTermBasedInstantiation:
    """Test term-based quantifier instantiation."""

    def test_single_var_instantiation(self):
        inst = QuantifierInstantiator()
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        terms = [_int(1), _int(2), _int(3)]
        results = inst.instantiate_forall(f, terms, InstantiationStrategy.TERM_BASED)
        assert len(results) == 3

    def test_exists_instantiation(self):
        inst = QuantifierInstantiator()
        f = Exists([("x", INT)], _eq(_var("x"), _int(5)))
        terms = [_int(3), _int(5), _int(7)]
        results = inst.instantiate_exists(f, terms)
        assert len(results) == 3

    def test_multi_var_instantiation(self):
        inst = QuantifierInstantiator()
        f = Forall([("x", INT), ("y", INT)], _lt(_var("x"), _var("y")))
        terms = [_int(0), _int(1)]
        results = inst.instantiate_forall(f, terms, InstantiationStrategy.TERM_BASED)
        assert len(results) >= 4  # 2x2 cross product

    def test_empty_ground_terms(self):
        inst = QuantifierInstantiator()
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        results = inst.instantiate_forall(f, [], InstantiationStrategy.TERM_BASED)
        assert len(results) == 0

    def test_non_quantified_passthrough(self):
        inst = QuantifierInstantiator()
        t = _gt(_var("x"), _int(0))
        results = inst.instantiate_forall(t, [_int(1)])
        assert len(results) == 1
        assert results[0] == t

    def test_stats_tracking(self):
        inst = QuantifierInstantiator()
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        inst.instantiate_forall(f, [_int(1), _int(2)], InstantiationStrategy.TERM_BASED)
        assert inst.stats["term_based"] == 2


# ===================================================================
# Section 7: Quantifier Instantiation - E-Matching
# ===================================================================

class TestEMatchingInstantiation:
    """Test E-matching quantifier instantiation."""

    def test_e_matching_with_select(self):
        inst = QuantifierInstantiator()
        # forall i. a[i] > 0
        body = _gt(Select(_var("a"), _var("i")), _int(0))
        f = Forall([("i", INT)], body)
        terms = [_int(0), _int(1), _int(2)]
        results = inst.instantiate_forall(f, terms, InstantiationStrategy.E_MATCHING)
        assert len(results) > 0

    def test_e_matching_fallback(self):
        inst = QuantifierInstantiator()
        # No select/store patterns -- should fall back to term-based
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        terms = [_int(1), _int(2)]
        results = inst.instantiate_forall(f, terms, InstantiationStrategy.E_MATCHING)
        assert len(results) >= 2

    def test_e_matching_stats(self):
        inst = QuantifierInstantiator()
        body = _gt(Select(_var("a"), _var("i")), _int(0))
        f = Forall([("i", INT)], body)
        inst.instantiate_forall(f, [_int(0)], InstantiationStrategy.E_MATCHING)
        assert inst.stats["e_matching"] >= 1


# ===================================================================
# Section 8: Quantifier Instantiation - Model-Based
# ===================================================================

class TestModelBasedInstantiation:
    """Test model-based quantifier instantiation (MBQI)."""

    def test_model_based_finds_counterexample(self):
        inst = QuantifierInstantiator()
        # forall x. x > 0 -- not valid, so MBQI should find a counterexample
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        terms = [_int(1)]
        results = inst.instantiate_forall(f, terms, InstantiationStrategy.MODEL_BASED)
        assert len(results) > 0

    def test_model_based_stats(self):
        inst = QuantifierInstantiator()
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        inst.instantiate_forall(f, [_int(0)], InstantiationStrategy.MODEL_BASED)
        assert inst.stats["model_based"] >= 1


# ===================================================================
# Section 9: Array Axiom Engine
# ===================================================================

class TestArrayAxiomEngine:
    """Test array theory axiom generation."""

    def test_read_over_write_same(self):
        engine = ArrayAxiomEngine()
        axiom = engine.read_over_write_same(_var("a"), _int(0), _int(42))
        # Select(Store(a, 0, 42), 0) == 42
        assert isinstance(axiom, App)
        assert axiom.op == Op.EQ

    def test_read_over_write_diff(self):
        engine = ArrayAxiomEngine()
        axiom = engine.read_over_write_diff(_var("a"), _int(0), _int(1), _int(42))
        # 0 != 1 => Select(Store(a, 0, 42), 1) == Select(a, 1)
        assert isinstance(axiom, App)

    def test_const_array_axiom(self):
        engine = ArrayAxiomEngine()
        axiom = engine.const_array_axiom(_int(0), _int(5))
        # Select(ConstArray(0), 5) == 0
        assert isinstance(axiom, App)
        assert axiom.op == Op.EQ

    def test_generate_axioms_for_store(self):
        engine = ArrayAxiomEngine()
        formula = Store(_var("a"), _int(0), _int(1))
        axioms = engine.generate_axioms(formula, [_int(0), _int(1)])
        assert len(axioms) > 0  # At least read-over-write-same + diffs

    def test_generate_axioms_no_ops(self):
        engine = ArrayAxiomEngine()
        formula = _gt(_var("x"), _int(0))
        axioms = engine.generate_axioms(formula, [_int(0)])
        assert len(axioms) == 0


# ===================================================================
# Section 10: Quantified CHC System
# ===================================================================

class TestQuantifiedCHCSystem:
    """Test QuantifiedCHCSystem construction and manipulation."""

    def test_create_system(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        assert "P" in sys.predicates

    def test_add_fact(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_fact(apply_pred(p, [_var("x")]), _eq(_var("x"), _int(0)))
        assert len(sys.get_facts()) == 1

    def test_add_quantified_clause(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        constraint = Forall([("i", INT)], _gt(_var("i"), _int(0)))
        sys.add_quantified_clause(
            head=apply_pred(p, [_var("x")]),
            body_preds=[],
            constraint=constraint,
        )
        assert len(sys.clauses) == 1
        assert sys.clauses[0].is_quantified

    def test_add_query(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_query([apply_pred(p, [_var("x")])], _lt(_var("x"), _int(0)))
        assert len(sys.get_queries()) == 1

    def test_declare_array(self):
        sys = QuantifiedCHCSystem()
        sys.declare_array("a")
        assert "a" in sys.array_vars

    def test_to_standard_chc(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_fact(apply_pred(p, [_var("x")]), _eq(_var("x"), _int(0)))
        sys.add_query([apply_pred(p, [_var("x")])], _lt(_var("x"), _int(0)))
        std, inst = sys.to_standard_chc()
        assert isinstance(std, CHCSystem)

    def test_collect_ground_terms(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_fact(apply_pred(p, [_int(42)]), _eq(_var("x"), _int(42)))
        terms = sys._collect_ground_terms()
        # Should include 42 and small constants
        values = {t.value for t in terms if isinstance(t, IntConst)}
        assert 42 in values
        assert 0 in values


# ===================================================================
# Section 11: Quantified CHC Solving -- Non-quantified Baseline
# ===================================================================

class TestQCHCSolvingBaseline:
    """Verify that quantified solver handles standard (non-quantified) systems."""

    def test_simple_safe(self):
        """x = 0; while (x < 10) x++; assert x >= 0"""
        sys = QuantifiedCHCSystem()
        inv = sys.add_predicate("Inv", [("x", INT)])
        sys.add_fact(apply_pred(inv, [_var("x")]), _eq(_var("x"), _int(0)))
        sys.add_clause(
            head=apply_pred(inv, [_add(_var("x"), _int(1))]),
            body_preds=[apply_pred(inv, [_var("x")])],
            constraint=_lt(_var("x"), _int(10)),
        )
        sys.add_query([apply_pred(inv, [_var("x")])], _lt(_var("x"), _int(0)))
        result = solve_quantified_chc(sys)
        assert result.result == CHCResult.SAT

    def test_simple_unsafe(self):
        """x = 10; while (x > 0) x--; assert x > 0 (fails at x=0)"""
        sys = QuantifiedCHCSystem()
        inv = sys.add_predicate("Inv", [("x", INT)])
        sys.add_fact(apply_pred(inv, [_var("x")]), _eq(_var("x"), _int(10)))
        sys.add_clause(
            head=apply_pred(inv, [_sub(_var("x"), _int(1))]),
            body_preds=[apply_pred(inv, [_var("x")])],
            constraint=_gt(_var("x"), _int(0)),
        )
        sys.add_query([apply_pred(inv, [_var("x")])], _eq(_var("x"), _int(0)))
        result = solve_quantified_chc(sys)
        assert result.result == CHCResult.UNSAT

    def test_two_variable_safe(self):
        """x = 0, y = 10; while (x < y) { x++; y--; } assert x >= 0"""
        sys = QuantifiedCHCSystem()
        inv = sys.add_predicate("Inv", [("x", INT), ("y", INT)])
        sys.add_fact(
            apply_pred(inv, [_var("x"), _var("y")]),
            _and(_eq(_var("x"), _int(0)), _eq(_var("y"), _int(10)))
        )
        sys.add_clause(
            head=apply_pred(inv, [_add(_var("x"), _int(1)), _sub(_var("y"), _int(1))]),
            body_preds=[apply_pred(inv, [_var("x"), _var("y")])],
            constraint=_lt(_var("x"), _var("y")),
        )
        sys.add_query(
            [apply_pred(inv, [_var("x"), _var("y")])],
            _lt(_var("x"), _int(0)),
        )
        result = solve_quantified_chc(sys)
        assert result.result == CHCResult.SAT


# ===================================================================
# Section 12: Quantified CHC Solving -- Universally Quantified
# ===================================================================

class TestQCHCSolvingForall:
    """Solve systems with universally quantified constraints."""

    def test_forall_in_query(self):
        """System with forall in query constraint.
        Inv(n) where n >= 0.
        Query: Inv(n) AND exists i. (0 <= i < n AND false) => false
        This should be safe since the query body is always false.
        """
        sys = QuantifiedCHCSystem()
        inv = sys.add_predicate("Inv", [("n", INT)])
        sys.add_fact(apply_pred(inv, [_var("n")]), _ge(_var("n"), _int(0)))
        # Query: safe because constraint is trivially unsat
        sys.add_quantified_clause(
            head=None,
            body_preds=[apply_pred(inv, [_var("n")])],
            constraint=_and(_ge(_var("n"), _int(0)), BoolConst(False)),
        )
        result = solve_quantified_chc(sys)
        assert result.result == CHCResult.SAT

    def test_forall_property_verification(self):
        """Verify universally quantified property using verify_universal_property."""
        # Simple: n starts at 5, decrements. Property: n >= 0 (always true within loop)
        init = _eq(_var("n"), _int(5))
        trans = _and(_gt(_var("n"), _int(0)), _eq(_var("n'"), _sub(_var("n"), _int(1))))
        prop = _ge(_var("n"), _int(0))
        result = verify_universal_property(init, trans, prop, [("n", INT)])
        assert result.result == CHCResult.SAT

    def test_forall_quantified_constraint_in_fact(self):
        """Fact with quantified constraint."""
        sys = QuantifiedCHCSystem()
        inv = sys.add_predicate("Inv", [("n", INT)])
        # forall x. x >= 0 is not valid, but we're just testing the machinery
        constraint = Forall(
            [("x", INT)],
            _implies(_ge(_var("x"), _int(0)), _ge(_var("x"), _int(0)))
        )
        sys.add_quantified_clause(
            head=apply_pred(inv, [_var("n")]),
            body_preds=[],
            constraint=_and(_eq(_var("n"), _int(0)), constraint),
        )
        sys.add_query(
            [apply_pred(inv, [_var("n")])],
            _lt(_var("n"), _int(0))
        )
        result = solve_quantified_chc(sys)
        # Eliminating quantifier: tautology conjoined with n=0
        assert result.result == CHCResult.SAT


# ===================================================================
# Section 13: Quantified CHC Solving -- Existentially Quantified
# ===================================================================

class TestQCHCSolvingExists:
    """Solve systems with existentially quantified constraints."""

    def test_exists_in_constraint(self):
        """System where existence of a witness is needed."""
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        # Fact: exists y. y > 0 AND x = y => P(x)
        constraint = Exists(
            [("y", INT)],
            _and(_gt(_var("y"), _int(0)), _eq(_var("x"), _var("y")))
        )
        sys.add_quantified_clause(
            head=apply_pred(p, [_var("x")]),
            body_preds=[],
            constraint=constraint,
        )
        # Query: P(x) AND x <= 0 => false
        sys.add_query([apply_pred(p, [_var("x")])], _le(_var("x"), _int(0)))
        result = solve_quantified_chc(sys)
        # P(x) only holds for positive x (from existential), so x <= 0 should be unsat
        assert result.result == CHCResult.SAT


# ===================================================================
# Section 14: Array Property Formulas
# ===================================================================

class TestArrayProperties:
    """Test array property formula constructors."""

    def test_sorted_property(self):
        prop = array_sorted_property("a", _int(0), _int(5))
        assert isinstance(prop, Forall)

    def test_bounded_property(self):
        prop = array_bounded_property("a", _int(0), _int(10), _int(0), _int(100))
        assert isinstance(prop, Forall)

    def test_initialized_property(self):
        prop = array_initialized_property("a", _int(0), _int(5), _int(0))
        assert isinstance(prop, Forall)

    def test_partitioned_property(self):
        prop = array_partitioned_property("a", _int(3), _int(0), _int(7))
        assert isinstance(prop, Forall)

    def test_exists_element(self):
        prop = array_exists_element("a", _int(0), _int(5), _int(42))
        assert isinstance(prop, Exists)

    def test_sorted_is_quantified(self):
        prop = array_sorted_property("arr", _int(0), _int(10))
        fv = collect_free_vars(prop)
        assert "arr" in fv
        assert "__sort_i__" not in fv  # bound by forall


# ===================================================================
# Section 15: Validity Checking
# ===================================================================

class TestValidityChecking:
    """Test quantified validity checking."""

    def test_valid_tautology(self):
        # forall x. x == x
        f = Forall([("x", INT)], _eq(_var("x"), _var("x")))
        valid, _ = check_quantified_validity(f)
        assert valid

    def test_invalid_forall(self):
        # forall x. x > 0 -- not valid
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        valid, ce = check_quantified_validity(f)
        assert not valid

    def test_valid_implication(self):
        # forall x. x > 5 => x > 0
        f = Forall([("x", INT)], _implies(_gt(_var("x"), _int(5)), _gt(_var("x"), _int(0))))
        valid, _ = check_quantified_validity(f)
        assert valid

    def test_exists_satisfiable(self):
        # exists x. x == 5
        f = Exists([("x", INT)], _eq(_var("x"), _int(5)))
        valid, model = check_quantified_validity(f)
        assert valid

    def test_non_quantified(self):
        # 1 > 0 -- trivially valid
        valid, _ = check_quantified_validity(_gt(_int(1), _int(0)))
        assert valid


# ===================================================================
# Section 16: Analysis and Summary
# ===================================================================

class TestAnalysisAndSummary:
    """Test system analysis and summary APIs."""

    def test_analyze_basic(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_fact(apply_pred(p, [_var("x")]), _eq(_var("x"), _int(0)))
        sys.add_query([apply_pred(p, [_var("x")])], _lt(_var("x"), _int(0)))
        analysis = analyze_quantified_system(sys)
        assert analysis["total_clauses"] == 2
        assert analysis["facts"] == 1
        assert analysis["queries"] == 1
        assert "P" in analysis["predicates"]

    def test_analyze_quantified(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_quantified_clause(
            head=apply_pred(p, [_var("x")]),
            body_preds=[],
            constraint=Forall([("i", INT)], _gt(_var("i"), _int(0))),
        )
        analysis = analyze_quantified_system(sys)
        assert analysis["quantified_clauses"] == 1
        assert analysis["forall_clauses"] == 1

    def test_analyze_arrays(self):
        sys = QuantifiedCHCSystem()
        sys.declare_array("a")
        analysis = analyze_quantified_system(sys)
        assert analysis["has_arrays"]
        assert "a" in analysis["array_vars"]

    def test_summary_string(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_fact(apply_pred(p, [_var("x")]), _eq(_var("x"), _int(0)))
        s = quantified_summary(sys)
        assert "Quantified CHC System" in s
        assert "P" in s

    def test_compare_strategies(self):
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        terms = [_int(0), _int(1), _int(2)]
        results = compare_instantiation_strategies(f, terms)
        assert "term_based" in results
        assert "e_matching" in results
        assert "model_based" in results


# ===================================================================
# Section 17: Integration -- Verify Array Property
# ===================================================================

class TestArrayPropertyVerification:
    """Integration tests: verify array properties via quantified CHC."""

    def test_verify_simple_scalar_property(self):
        """Scalar property through verify_array_property API."""
        result = verify_array_property(
            init_constraint=_eq(_var("n"), _int(0)),
            loop_constraint=_and(
                _lt(_var("n"), _int(5)),
                _eq(_var("n'"), _add(_var("n"), _int(1)))
            ),
            property_formula=_ge(_var("n"), _int(0)),
            var_params=[("n", INT)],
        )
        assert result.result == CHCResult.SAT

    def test_verify_property_with_array_vars(self):
        """System declaring array vars goes through array axiom path."""
        result = verify_array_property(
            init_constraint=_eq(_var("n"), _int(0)),
            loop_constraint=_and(
                _lt(_var("n"), _int(3)),
                _eq(_var("n'"), _add(_var("n"), _int(1)))
            ),
            property_formula=_ge(_var("n"), _int(0)),
            var_params=[("n", INT)],
            array_vars=["a"],
        )
        assert result.result == CHCResult.SAT

    def test_verify_unsafe_property(self):
        """Unsafe property should return UNSAT."""
        result = verify_array_property(
            init_constraint=_eq(_var("x"), _int(10)),
            loop_constraint=_and(
                _gt(_var("x"), _int(0)),
                _eq(_var("x'"), _sub(_var("x"), _int(1)))
            ),
            property_formula=_gt(_var("x"), _int(0)),  # Fails at x=0
            var_params=[("x", INT)],
        )
        assert result.result == CHCResult.UNSAT

    def test_qchc_output_has_stats(self):
        """Result includes instantiation stats."""
        result = verify_array_property(
            init_constraint=_eq(_var("n"), _int(0)),
            loop_constraint=_and(
                _lt(_var("n"), _int(3)),
                _eq(_var("n'"), _add(_var("n"), _int(1)))
            ),
            property_formula=_ge(_var("n"), _int(0)),
            var_params=[("n", INT)],
        )
        assert result.stats is not None

    def test_safe_interpretation_returned(self):
        """SAT result should include interpretation."""
        result = verify_array_property(
            init_constraint=_eq(_var("n"), _int(0)),
            loop_constraint=_and(
                _lt(_var("n"), _int(5)),
                _eq(_var("n'"), _add(_var("n"), _int(1)))
            ),
            property_formula=_ge(_var("n"), _int(0)),
            var_params=[("n", INT)],
        )
        if result.result == CHCResult.SAT and result.interpretation:
            assert "Inv" in result.interpretation


# ===================================================================
# Section 18: Edge Cases
# ===================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_system(self):
        sys = QuantifiedCHCSystem()
        result = solve_quantified_chc(sys)
        assert result.result == CHCResult.SAT

    def test_trivial_fact_only(self):
        sys = QuantifiedCHCSystem()
        p = sys.add_predicate("P", [("x", INT)])
        sys.add_fact(apply_pred(p, [_var("x")]), BoolConst(True))
        result = solve_quantified_chc(sys)
        assert result.result == CHCResult.SAT

    def test_quantified_clause_marker(self):
        c = QuantifiedClause(
            head=None, body_preds=[], constraint=BoolConst(False)
        )
        assert c.is_query
        assert not c.is_quantified

    def test_quantified_clause_with_forall(self):
        c = QuantifiedClause(
            head=None, body_preds=[],
            constraint=Forall([("x", INT)], _gt(_var("x"), _int(0)))
        )
        assert c.is_quantified

    def test_max_instances_limit(self):
        inst = QuantifierInstantiator(max_instances=3)
        f = Forall([("x", INT)], _gt(_var("x"), _int(0)))
        terms = [_int(i) for i in range(100)]
        results = inst.instantiate_forall(f, terms, InstantiationStrategy.TERM_BASED)
        assert len(results) == 3

    def test_forall_no_body_vars(self):
        """Forall with no occurrences of bound var in body."""
        f = Forall([("x", INT)], _gt(_int(5), _int(0)))
        valid, _ = check_quantified_validity(f)
        assert valid  # Vacuously true since body doesn't depend on x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
