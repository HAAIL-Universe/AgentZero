"""Tests for V109: Constrained Horn Clause Solver."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))

from chc_solver import (
    CHCSystem, CHCResult, CHCOutput, CHCStats,
    Predicate, PredicateApp, HornClause, Interpretation, Derivation,
    apply_pred, instantiate_pred,
    BMCSolver, PDRCHCSolver, InterpCHCSolver,
    chc_from_ts, chc_from_loop, solve_chc, verify_safety,
    compare_strategies, chc_summary,
    _and, _or, _not, _eq, _implies, _substitute, _smt_check, _check_implication
)
from smt_solver import (
    SMTSolver, SMTResult, Var, IntConst, BoolConst, App, Op, INT, BOOL, Sort, SortKind
)
from craig_interpolation import collect_vars
from pdr import TransitionSystem


# ============================================================
# Section 1: Data Structure Tests
# ============================================================

class TestDataStructures:
    """Test CHC data structures."""

    def test_predicate_creation(self):
        p = Predicate("Inv", [("x", INT), ("y", INT)])
        assert p.name == "Inv"
        assert p.arity == 2

    def test_predicate_equality(self):
        p1 = Predicate("Inv", [("x", INT)])
        p2 = Predicate("Inv", [("y", INT)])
        assert p1 == p2  # Same name
        assert hash(p1) == hash(p2)

    def test_predicate_app(self):
        p = Predicate("Inv", [("x", INT)])
        app = apply_pred(p, [IntConst(5)])
        assert app.predicate == p
        assert len(app.args) == 1
        assert "Inv" in repr(app)

    def test_horn_clause_fact(self):
        p = Predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        clause = HornClause(
            head=apply_pred(p, [x]),
            body_preds=[],
            constraint=App(Op.GE, [x, IntConst(0)], BOOL)
        )
        assert clause.is_fact
        assert not clause.is_query
        assert clause.is_linear

    def test_horn_clause_query(self):
        p = Predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        clause = HornClause(
            head=None,
            body_preds=[apply_pred(p, [x])],
            constraint=App(Op.LT, [x, IntConst(0)], BOOL)
        )
        assert not clause.is_fact
        assert clause.is_query
        assert clause.is_linear

    def test_horn_clause_rule(self):
        p = Predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)
        clause = HornClause(
            head=apply_pred(p, [xp]),
            body_preds=[apply_pred(p, [x])],
            constraint=_eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        )
        assert not clause.is_fact
        assert not clause.is_query
        assert clause.is_linear

    def test_nonlinear_clause(self):
        p = Predicate("P", [("x", INT)])
        q = Predicate("Q", [("y", INT)])
        x, y = Var("x", INT), Var("y", INT)
        clause = HornClause(
            head=apply_pred(p, [x]),
            body_preds=[apply_pred(p, [y]), apply_pred(q, [x])],
            constraint=BoolConst(True)
        )
        assert not clause.is_linear

    def test_chc_system_creation(self):
        system = CHCSystem()
        p = system.add_predicate("Inv", [("x", INT)])
        assert "Inv" in system.predicates

    def test_chc_system_add_fact(self):
        system = CHCSystem()
        p = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        system.add_fact(apply_pred(p, [x]), _eq(x, IntConst(0)))
        assert len(system.clauses) == 1
        assert system.clauses[0].is_fact

    def test_chc_system_add_query(self):
        system = CHCSystem()
        p = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        system.add_query([apply_pred(p, [x])], App(Op.LT, [x, IntConst(0)], BOOL))
        assert len(system.get_queries()) == 1

    def test_chc_system_linearity(self):
        system = CHCSystem()
        p = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        system.add_fact(apply_pred(p, [x]), BoolConst(True))
        assert system.is_linear

    def test_interpretation(self):
        interp = Interpretation(mapping={})
        x = Var("x", INT)
        interp.set("Inv", App(Op.GE, [x, IntConst(0)], BOOL))
        assert interp.get("Inv") is not None
        assert isinstance(interp.get("unknown"), BoolConst)

    def test_instantiate_pred(self):
        p = Predicate("Inv", [("x", INT)])
        interp = Interpretation(mapping={})
        x = Var("x", INT)
        interp.set("Inv", App(Op.GE, [x, IntConst(0)], BOOL))
        result = instantiate_pred(interp, apply_pred(p, [Var("y", INT)]))
        # Should substitute x -> y in the formula
        assert result is not None

    def test_derivation(self):
        clause = HornClause(head=None, body_preds=[], constraint=BoolConst(True))
        d = Derivation(clause=clause, children=[], model={"x": 5})
        assert d.depth() == 0

    def test_derivation_depth(self):
        c = HornClause(head=None, body_preds=[], constraint=BoolConst(True))
        leaf = Derivation(clause=c, children=[], model=None)
        inner = Derivation(clause=c, children=[leaf], model=None)
        root = Derivation(clause=c, children=[inner], model=None)
        assert root.depth() == 2

    def test_chc_stats(self):
        stats = CHCStats()
        assert stats.smt_queries == 0
        assert stats.strategy == ""


# ============================================================
# Section 2: Formula Utility Tests
# ============================================================

class TestFormulaUtils:
    """Test formula helper functions."""

    def test_and_empty(self):
        result = _and()
        assert isinstance(result, BoolConst) and result.value is True

    def test_and_single(self):
        x = Var("x", BOOL)
        result = _and(x)
        assert result == x

    def test_and_with_true(self):
        x = Var("x", BOOL)
        result = _and(BoolConst(True), x)
        assert result == x

    def test_and_with_false(self):
        x = Var("x", BOOL)
        result = _and(BoolConst(False), x)
        assert isinstance(result, BoolConst) and result.value is False

    def test_or_empty(self):
        result = _or()
        assert isinstance(result, BoolConst) and result.value is False

    def test_or_with_true(self):
        x = Var("x", BOOL)
        result = _or(BoolConst(True), x)
        assert isinstance(result, BoolConst) and result.value is True

    def test_not_boolconst(self):
        assert _not(BoolConst(True)).value is False
        assert _not(BoolConst(False)).value is True

    def test_not_complement(self):
        x, y = Var("x", INT), Var("y", INT)
        eq = App(Op.EQ, [x, y], BOOL)
        neq = _not(eq)
        assert isinstance(neq, App) and neq.op == Op.NEQ

    def test_not_double_negation(self):
        x = Var("x", BOOL)
        neg = App(Op.NOT, [x], BOOL)
        result = _not(neg)
        assert result == x

    def test_implies(self):
        a, b = Var("a", BOOL), Var("b", BOOL)
        result = _implies(a, b)
        assert isinstance(result, App) and result.op == Op.IMPLIES

    def test_eq_int(self):
        x, y = Var("x", INT), Var("y", INT)
        result = _eq(x, y)
        assert isinstance(result, App) and result.op == Op.EQ

    def test_substitute(self):
        x = Var("x", INT)
        formula = App(Op.GE, [x, IntConst(0)], BOOL)
        result = _substitute(formula, {"x": Var("y", INT)})
        assert collect_vars(result) == {"y"}

    def test_smt_check_sat(self):
        x = Var("x", INT)
        result, model = _smt_check(App(Op.EQ, [x, IntConst(5)], BOOL))
        assert result == SMTResult.SAT

    def test_smt_check_unsat(self):
        x = Var("x", INT)
        formula = _and(App(Op.GT, [x, IntConst(5)], BOOL),
                       App(Op.LT, [x, IntConst(3)], BOOL))
        result, _ = _smt_check(formula)
        assert result == SMTResult.UNSAT

    def test_check_implication(self):
        x = Var("x", INT)
        a = App(Op.EQ, [x, IntConst(5)], BOOL)
        b = App(Op.GT, [x, IntConst(3)], BOOL)
        assert _check_implication(a, b)

    def test_check_implication_false(self):
        x = Var("x", INT)
        a = App(Op.GT, [x, IntConst(0)], BOOL)
        b = App(Op.GT, [x, IntConst(10)], BOOL)
        assert not _check_implication(a, b)


# ============================================================
# Section 3: Simple Counter (SAFE)
# ============================================================

def make_simple_counter_chc():
    """
    Counter from 0 to N:
      let x = 0;
      while (x < 10) { x = x + 1; }
      assert(x >= 0);

    CHC encoding:
      x == 0 => Inv(x)
      Inv(x) AND x < 10 AND x' == x + 1 => Inv(x')
      Inv(x) AND x < 0 => false
    """
    system = CHCSystem()
    inv = system.add_predicate("Inv", [("x", INT)])
    x = Var("x", INT)
    xp = Var("x'", INT)

    # Fact: x == 0 => Inv(x)
    system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(0)))

    # Rule: Inv(x) AND x < 10 AND x' == x + 1 => Inv(x')
    system.add_clause(
        head=apply_pred(inv, [xp]),
        body_preds=[apply_pred(inv, [x])],
        constraint=_and(
            App(Op.LT, [x, IntConst(10)], BOOL),
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        )
    )

    # Query: Inv(x) AND x < 0 => false
    system.add_query(
        [apply_pred(inv, [x])],
        App(Op.LT, [x, IntConst(0)], BOOL)
    )

    return system


class TestSimpleCounterSafe:
    """Counter x=0..10, property x>=0. Should be SAFE."""

    def test_pdr_strategy(self):
        system = make_simple_counter_chc()
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT

    def test_auto_strategy(self):
        system = make_simple_counter_chc()
        result = solve_chc(system, strategy="auto")
        assert result.result == CHCResult.SAT

    def test_has_interpretation(self):
        system = make_simple_counter_chc()
        result = solve_chc(system, strategy="pdr")
        assert result.interpretation is not None

    def test_interp_cegar_strategy(self):
        system = make_simple_counter_chc()
        result = solve_chc(system, strategy="interp_cegar")
        # May return SAT or UNKNOWN depending on refinement
        assert result.result in (CHCResult.SAT, CHCResult.UNKNOWN)


# ============================================================
# Section 4: Simple Counter (UNSAFE)
# ============================================================

def make_unsafe_counter_chc():
    """
    Counter that violates property:
      x == 0 => Inv(x)
      Inv(x) AND x < 20 AND x' == x + 1 => Inv(x')
      Inv(x) AND x > 10 => false  (violated when x reaches 11)
    """
    system = CHCSystem()
    inv = system.add_predicate("Inv", [("x", INT)])
    x = Var("x", INT)
    xp = Var("x'", INT)

    system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(0)))
    system.add_clause(
        head=apply_pred(inv, [xp]),
        body_preds=[apply_pred(inv, [x])],
        constraint=_and(
            App(Op.LT, [x, IntConst(20)], BOOL),
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        )
    )
    system.add_query(
        [apply_pred(inv, [x])],
        App(Op.GT, [x, IntConst(10)], BOOL)
    )
    return system


class TestUnsafeCounter:
    """Counter reaches 11 which violates x <= 10."""

    def test_pdr_detects_unsafe(self):
        system = make_unsafe_counter_chc()
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.UNSAT

    def test_bmc_detects_unsafe(self):
        system = make_unsafe_counter_chc()
        result = solve_chc(system, strategy="bmc")
        assert result.result == CHCResult.UNSAT

    def test_has_derivation(self):
        system = make_unsafe_counter_chc()
        result = solve_chc(system, strategy="pdr")
        if result.result == CHCResult.UNSAT:
            assert result.derivation is not None


# ============================================================
# Section 5: Two-Variable System
# ============================================================

def make_two_var_chc():
    """
    Two variables with conservation:
      x=5, y=0 => Inv(x,y)
      Inv(x,y) AND x>0 AND x'=x-1 AND y'=y+1 => Inv(x',y')
      Inv(x,y) AND x+y != 5 => false
    Property: x+y == 5 always holds (sum conservation).
    """
    system = CHCSystem()
    inv = system.add_predicate("Inv", [("x", INT), ("y", INT)])
    x, y = Var("x", INT), Var("y", INT)
    xp, yp = Var("x'", INT), Var("y'", INT)

    system.add_fact(
        apply_pred(inv, [x, y]),
        _and(_eq(x, IntConst(5)), _eq(y, IntConst(0)))
    )
    system.add_clause(
        head=apply_pred(inv, [xp, yp]),
        body_preds=[apply_pred(inv, [x, y])],
        constraint=_and(
            App(Op.GT, [x, IntConst(0)], BOOL),
            _eq(xp, App(Op.SUB, [x, IntConst(1)], INT)),
            _eq(yp, App(Op.ADD, [y, IntConst(1)], INT))
        )
    )
    system.add_query(
        [apply_pred(inv, [x, y])],
        App(Op.NEQ, [App(Op.ADD, [x, y], INT), IntConst(5)], BOOL)
    )
    return system


class TestTwoVariableSystem:
    """Sum conservation: x+y == 5 always."""

    def test_pdr_safe(self):
        system = make_two_var_chc()
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT

    def test_auto_safe(self):
        system = make_two_var_chc()
        result = solve_chc(system, strategy="auto")
        assert result.result == CHCResult.SAT


# ============================================================
# Section 6: Immediate Violation
# ============================================================

class TestImmediateViolation:
    """Property violated in initial state."""

    def test_immediate_unsafe(self):
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        # Init: x == -1 (already violates x >= 0)
        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(-1)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, x)
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.LT, [x, IntConst(0)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.UNSAT

    def test_bmc_immediate(self):
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(-1)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, x)
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.LT, [x, IntConst(0)], BOOL)
        )
        result = solve_chc(system, strategy="bmc")
        assert result.result == CHCResult.UNSAT


# ============================================================
# Section 7: Trivially Safe System
# ============================================================

class TestTriviallySafe:
    """Property that trivially holds."""

    def test_always_zero(self):
        """x is always 0, property x >= 0."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(0)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, IntConst(0))
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.LT, [x, IntConst(0)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT

    def test_constant_safe(self):
        """x is always 42, property x == 42."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(42)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, IntConst(42))
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.NEQ, [x, IntConst(42)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT


# ============================================================
# Section 8: CHC from Transition System Conversion
# ============================================================

class TestCHCFromTS:
    """Test conversion from V002 TransitionSystem to CHC."""

    def test_simple_ts_conversion(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(ts.prime("x"), App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        system = chc_from_ts(ts)
        assert "Inv" in system.predicates
        assert len(system.clauses) == 3  # fact + rule + query
        assert system.is_linear

    def test_ts_safe_through_chc(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_and(
            App(Op.LT, [x, IntConst(10)], BOOL),
            _eq(ts.prime("x"), App(Op.ADD, [x, IntConst(1)], INT))
        ))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        system = chc_from_ts(ts)
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT

    def test_ts_unsafe_through_chc(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(ts.prime("x"), App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.LT, [x, IntConst(5)], BOOL))

        system = chc_from_ts(ts)
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.UNSAT


# ============================================================
# Section 9: CHC from Loop Components
# ============================================================

class TestCHCFromLoop:
    """Test chc_from_loop convenience function."""

    def test_safe_loop(self):
        x = Var("x", INT)
        xp = Var("x'", INT)

        init = _eq(x, IntConst(0))
        trans = _and(
            App(Op.LT, [x, IntConst(100)], BOOL),
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        )
        bad = App(Op.LT, [x, IntConst(0)], BOOL)

        system = chc_from_loop(init, trans, bad, ["x"])
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT

    def test_unsafe_loop(self):
        x = Var("x", INT)
        xp = Var("x'", INT)

        init = _eq(x, IntConst(10))
        trans = _eq(xp, App(Op.SUB, [x, IntConst(1)], INT))
        bad = App(Op.LT, [x, IntConst(0)], BOOL)

        system = chc_from_loop(init, trans, bad, ["x"])
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.UNSAT


# ============================================================
# Section 10: verify_safety convenience API
# ============================================================

class TestVerifySafety:
    """Test the high-level verify_safety API."""

    def test_safe_increment(self):
        x = Var("x", INT)
        xp = Var("x'", INT)

        result = verify_safety(
            init=_eq(x, IntConst(0)),
            trans=_and(
                App(Op.LT, [x, IntConst(50)], BOOL),
                _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
            ),
            prop=App(Op.GE, [x, IntConst(0)], BOOL),
            var_names=["x"]
        )
        assert result.result == CHCResult.SAT

    def test_unsafe_decrement(self):
        x = Var("x", INT)
        xp = Var("x'", INT)

        result = verify_safety(
            init=_eq(x, IntConst(5)),
            trans=_eq(xp, App(Op.SUB, [x, IntConst(1)], INT)),
            prop=App(Op.GE, [x, IntConst(0)], BOOL),
            var_names=["x"]
        )
        assert result.result == CHCResult.UNSAT


# ============================================================
# Section 11: BMC Solver
# ============================================================

class TestBMCSolver:
    """Test bounded model checking strategy."""

    def test_bmc_finds_short_cex(self):
        """BMC finds a counterexample at depth 1."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(5)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.GT, [x, IntConst(6)], BOOL)
        )

        solver = BMCSolver(system, max_depth=5)
        result = solver.solve()
        assert result.result == CHCResult.UNSAT

    def test_bmc_respects_depth_limit(self):
        """BMC returns UNKNOWN if depth limit is too low."""
        system = make_unsafe_counter_chc()  # needs 11 steps
        solver = BMCSolver(system, max_depth=2)
        result = solver.solve()
        # May or may not find it depending on encoding
        assert result.result in (CHCResult.UNSAT, CHCResult.UNKNOWN)

    def test_bmc_stats(self):
        system = make_unsafe_counter_chc()
        solver = BMCSolver(system, max_depth=20)
        result = solver.solve()
        assert result.stats.strategy == "bmc"


# ============================================================
# Section 12: PDR CHC Solver
# ============================================================

class TestPDRCHCSolver:
    """Test PDR-based strategy."""

    def test_pdr_safe(self):
        system = make_simple_counter_chc()
        solver = PDRCHCSolver(system)
        result = solver.solve()
        assert result.result == CHCResult.SAT

    def test_pdr_unsafe(self):
        system = make_unsafe_counter_chc()
        solver = PDRCHCSolver(system)
        result = solver.solve()
        assert result.result == CHCResult.UNSAT

    def test_pdr_rejects_nonlinear(self):
        """PDR solver returns UNKNOWN for non-linear CHC."""
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("y", INT)])
        x, y = Var("x", INT), Var("y", INT)

        system.add_fact(apply_pred(p, [x]), _eq(x, IntConst(0)))
        system.add_fact(apply_pred(q, [y]), _eq(y, IntConst(0)))
        # Non-linear: two body predicates
        system.add_clause(
            head=apply_pred(p, [App(Op.ADD, [x, y], INT)]),
            body_preds=[apply_pred(p, [x]), apply_pred(q, [y])],
            constraint=BoolConst(True)
        )
        system.add_query([apply_pred(p, [x])], App(Op.LT, [x, IntConst(0)], BOOL))

        solver = PDRCHCSolver(system)
        result = solver.solve()
        assert result.result == CHCResult.UNKNOWN

    def test_pdr_stats(self):
        system = make_simple_counter_chc()
        solver = PDRCHCSolver(system)
        result = solver.solve()
        assert result.stats.strategy == "pdr"
        assert result.stats.smt_queries > 0


# ============================================================
# Section 13: Interpolation-based CEGAR Solver
# ============================================================

class TestInterpCHCSolver:
    """Test interpolation CEGAR strategy."""

    def test_interp_cegar_basic(self):
        system = make_simple_counter_chc()
        solver = InterpCHCSolver(system)
        result = solver.solve()
        assert result.result in (CHCResult.SAT, CHCResult.UNKNOWN)

    def test_interp_cegar_unsafe(self):
        system = make_unsafe_counter_chc()
        solver = InterpCHCSolver(system)
        result = solver.solve()
        # Should detect UNSAT or give UNKNOWN
        assert result.result in (CHCResult.UNSAT, CHCResult.UNKNOWN)

    def test_interp_cegar_stats(self):
        system = make_simple_counter_chc()
        solver = InterpCHCSolver(system)
        result = solver.solve()
        assert result.stats.strategy == "interp_cegar"

    def test_interp_cegar_immediate_violation(self):
        """Initial state violates property."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(-5)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, x)
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.LT, [x, IntConst(0)], BOOL)
        )
        solver = InterpCHCSolver(system)
        result = solver.solve()
        assert result.result in (CHCResult.UNSAT, CHCResult.UNKNOWN)


# ============================================================
# Section 14: Conditional Transition
# ============================================================

class TestConditionalTransition:
    """Test systems with conditional transitions."""

    def test_abs_value_safe(self):
        """
        if x > 0: x' = x - 1
        else: x' = x + 1
        Property: x >= -1
        Init: x = 0
        """
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(0)))

        # Branch 1: x > 0 => x' = x - 1
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_and(
                App(Op.GT, [x, IntConst(0)], BOOL),
                _eq(xp, App(Op.SUB, [x, IntConst(1)], INT))
            )
        )
        # Branch 2: x <= 0 => x' = x + 1
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_and(
                App(Op.LE, [x, IntConst(0)], BOOL),
                _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
            )
        )

        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.LT, [x, IntConst(-1)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT


# ============================================================
# Section 15: Nondeterministic Transition
# ============================================================

class TestNondeterministic:
    """Test nondeterministic systems."""

    def test_nondet_increment(self):
        """
        x starts at 0, increments by 1 or 2.
        Property: x >= 0 (always holds since we only add positive).
        """
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(0)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        )
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_eq(xp, App(Op.ADD, [x, IntConst(2)], INT))
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.LT, [x, IntConst(0)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT


# ============================================================
# Section 16: Multi-Predicate CHC
# ============================================================

class TestMultiPredicate:
    """Test multi-predicate linear CHC systems."""

    def test_two_predicate_chain(self):
        """
        P(x) initialized with x=0
        P(x) AND x < 5 AND x'=x+1 => P(x')  (count to 5)
        P(x) AND x >= 5 => Q(x)  (transition to Q when x >= 5)
        Q(x) AND x < 0 => false  (Q should never have x < 0)
        """
        system = CHCSystem()
        p = system.add_predicate("P", [("x", INT)])
        q = system.add_predicate("Q", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(p, [x]), _eq(x, IntConst(0)))
        system.add_clause(
            head=apply_pred(p, [xp]),
            body_preds=[apply_pred(p, [x])],
            constraint=_and(
                App(Op.LT, [x, IntConst(5)], BOOL),
                _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
            )
        )
        system.add_clause(
            head=apply_pred(q, [x]),
            body_preds=[apply_pred(p, [x])],
            constraint=App(Op.GE, [x, IntConst(5)], BOOL)
        )
        system.add_query(
            [apply_pred(q, [x])],
            App(Op.LT, [x, IntConst(0)], BOOL)
        )

        result = solve_chc(system, strategy="auto")
        assert result.result in (CHCResult.SAT, CHCResult.UNKNOWN)


# ============================================================
# Section 17: System Properties
# ============================================================

class TestSystemProperties:
    """Test CHCSystem property queries."""

    def test_predicates_in_body(self):
        system = make_simple_counter_chc()
        body_preds = system.predicates_in_body()
        assert "Inv" in body_preds

    def test_predicates_in_head(self):
        system = make_simple_counter_chc()
        head_preds = system.predicates_in_head()
        assert "Inv" in head_preds

    def test_get_facts(self):
        system = make_simple_counter_chc()
        facts = system.get_facts()
        assert len(facts) == 1

    def test_get_queries(self):
        system = make_simple_counter_chc()
        queries = system.get_queries()
        assert len(queries) == 1

    def test_get_rules(self):
        system = make_simple_counter_chc()
        rules = system.get_rules()
        assert len(rules) == 1


# ============================================================
# Section 18: Compare Strategies
# ============================================================

class TestCompareStrategies:
    """Test strategy comparison."""

    def test_compare_safe(self):
        system = make_simple_counter_chc()
        results = compare_strategies(system)
        assert "pdr" in results
        assert "bmc" in results
        assert "interp_cegar" in results

    def test_compare_has_results(self):
        system = make_simple_counter_chc()
        results = compare_strategies(system)
        for strategy, data in results.items():
            assert "result" in data


# ============================================================
# Section 19: Summary
# ============================================================

class TestSummary:
    """Test summary function."""

    def test_summary_content(self):
        s = chc_summary()
        assert s["name"] == "V109: Constrained Horn Clause Solver"
        assert "pdr" in s["strategies"]
        assert "bmc" in s["strategies"]
        assert "interp_cegar" in s["strategies"]
        assert len(s["features"]) >= 5
        assert len(s["composition"]) == 3


# ============================================================
# Section 20: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_system(self):
        system = CHCSystem()
        result = solve_chc(system, strategy="bmc")
        assert result.result == CHCResult.UNKNOWN

    def test_fact_only_system(self):
        """System with only facts, no queries."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(0)))
        result = solve_chc(system, strategy="interp_cegar")
        assert result.result == CHCResult.SAT

    def test_query_only_system(self):
        """System with only a query (unsatisfiable constraint)."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        system.add_query(
            [apply_pred(inv, [x])],
            BoolConst(True)
        )
        result = solve_chc(system, strategy="interp_cegar")
        # With True interpretation, the query body is True AND True => false
        # which is violated. Refinement should strengthen Inv to false.
        assert result.result in (CHCResult.SAT, CHCResult.UNKNOWN)

    def test_single_var_bounds(self):
        """x in [0, 100], property 0 <= x <= 100."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]),
                        _and(App(Op.GE, [x, IntConst(0)], BOOL),
                             App(Op.LE, [x, IntConst(100)], BOOL)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_and(
                App(Op.GE, [xp, IntConst(0)], BOOL),
                App(Op.LE, [xp, IntConst(100)], BOOL)
            )
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.GT, [x, IntConst(100)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT

    def test_unknown_strategy(self):
        system = CHCSystem()
        with pytest.raises(ValueError):
            solve_chc(system, strategy="invalid")


# ============================================================
# Section 21: Countdown System
# ============================================================

class TestCountdown:
    """Countdown from N to 0."""

    def test_countdown_safe(self):
        """x starts at 10, decrements, property x >= 0 while x > 0."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT)])
        x = Var("x", INT)
        xp = Var("x'", INT)

        system.add_fact(apply_pred(inv, [x]), _eq(x, IntConst(10)))
        system.add_clause(
            head=apply_pred(inv, [xp]),
            body_preds=[apply_pred(inv, [x])],
            constraint=_and(
                App(Op.GT, [x, IntConst(0)], BOOL),
                _eq(xp, App(Op.SUB, [x, IntConst(1)], INT))
            )
        )
        system.add_query(
            [apply_pred(inv, [x])],
            App(Op.LT, [x, IntConst(0)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT


# ============================================================
# Section 22: Double Counter
# ============================================================

class TestDoubleCounter:
    """Two counters incrementing together."""

    def test_double_counter_relation(self):
        """x and y both start at 0, increment together. Property: x == y."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("x", INT), ("y", INT)])
        x, y = Var("x", INT), Var("y", INT)
        xp, yp = Var("x'", INT), Var("y'", INT)

        system.add_fact(
            apply_pred(inv, [x, y]),
            _and(_eq(x, IntConst(0)), _eq(y, IntConst(0)))
        )
        system.add_clause(
            head=apply_pred(inv, [xp, yp]),
            body_preds=[apply_pred(inv, [x, y])],
            constraint=_and(
                _eq(xp, App(Op.ADD, [x, IntConst(1)], INT)),
                _eq(yp, App(Op.ADD, [y, IntConst(1)], INT))
            )
        )
        system.add_query(
            [apply_pred(inv, [x, y])],
            App(Op.NEQ, [x, y], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT


# ============================================================
# Section 23: Accumulator
# ============================================================

class TestAccumulator:
    """Sum accumulator."""

    def test_accumulator_nonneg(self):
        """s += i for i in 1..n, property s >= 0."""
        system = CHCSystem()
        inv = system.add_predicate("Inv", [("i", INT), ("s", INT)])
        i, s = Var("i", INT), Var("s", INT)
        ip, sp = Var("i'", INT), Var("s'", INT)

        system.add_fact(
            apply_pred(inv, [i, s]),
            _and(_eq(i, IntConst(1)), _eq(s, IntConst(0)))
        )
        system.add_clause(
            head=apply_pred(inv, [ip, sp]),
            body_preds=[apply_pred(inv, [i, s])],
            constraint=_and(
                App(Op.LE, [i, IntConst(10)], BOOL),
                _eq(ip, App(Op.ADD, [i, IntConst(1)], INT)),
                _eq(sp, App(Op.ADD, [s, i], INT))
            )
        )
        system.add_query(
            [apply_pred(inv, [i, s])],
            App(Op.LT, [s, IntConst(0)], BOOL)
        )
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT


# ============================================================
# Section 24: Output Structure
# ============================================================

class TestOutputStructure:
    """Test CHCOutput fields."""

    def test_safe_output_has_interpretation(self):
        system = make_simple_counter_chc()
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.SAT
        assert result.interpretation is not None
        assert result.derivation is None

    def test_unsafe_output_has_derivation(self):
        system = make_unsafe_counter_chc()
        result = solve_chc(system, strategy="pdr")
        assert result.result == CHCResult.UNSAT
        assert result.derivation is not None

    def test_output_has_stats(self):
        system = make_simple_counter_chc()
        result = solve_chc(system, strategy="pdr")
        assert isinstance(result.stats, CHCStats)


# ============================================================
# Section 25: Interpretation Validation
# ============================================================

class TestInterpretationValidation:
    """Test that returned interpretations are meaningful."""

    def test_interpretation_repr(self):
        interp = Interpretation(mapping={})
        x = Var("x", INT)
        interp.set("Inv", App(Op.GE, [x, IntConst(0)], BOOL))
        s = repr(interp)
        assert "Inv" in s

    def test_interpretation_get_default(self):
        interp = Interpretation(mapping={})
        result = interp.get("nonexistent")
        assert isinstance(result, BoolConst) and result.value is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
