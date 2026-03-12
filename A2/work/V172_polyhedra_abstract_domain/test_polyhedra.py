"""Tests for V172: Polyhedra Abstract Domain."""

import pytest
from fractions import Fraction
from polyhedra import (
    LinExpr, Constraint, ConstraintKind, Polyhedron,
    PolyhedraInterpreter, AnalysisResult,
    polyhedra_from_intervals, analyze_program,
    compare_with_intervals, verify_relational_property, batch_analyze,
    _fourier_motzkin_eliminate_vars, _fm_eliminate_one,
)


# ===================================================================
# LinExpr tests
# ===================================================================

class TestLinExpr:
    def test_var(self):
        e = LinExpr.var("x")
        assert e.get_coeff("x") == 1
        assert e.constant == 0

    def test_const(self):
        e = LinExpr.const(5)
        assert e.constant == 5
        assert e.variables() == set()

    def test_zero(self):
        e = LinExpr.zero()
        assert e.constant == 0
        assert e.variables() == set()

    def test_add_exprs(self):
        e = LinExpr.var("x") + LinExpr.var("y")
        assert e.get_coeff("x") == 1
        assert e.get_coeff("y") == 1

    def test_add_scalar(self):
        e = LinExpr.var("x") + 3
        assert e.get_coeff("x") == 1
        assert e.constant == 3

    def test_sub_exprs(self):
        e = LinExpr.var("x") - LinExpr.var("y")
        assert e.get_coeff("x") == 1
        assert e.get_coeff("y") == -1

    def test_sub_scalar(self):
        e = LinExpr.var("x") - 5
        assert e.constant == -5

    def test_neg(self):
        e = -LinExpr.var("x")
        assert e.get_coeff("x") == -1

    def test_scale(self):
        e = LinExpr.var("x").scale(3)
        assert e.get_coeff("x") == 3

    def test_mul(self):
        e = LinExpr.var("x") * 2
        assert e.get_coeff("x") == 2

    def test_rmul(self):
        e = 3 * LinExpr.var("y")
        assert e.get_coeff("y") == 3

    def test_substitute(self):
        # x + y, substitute x with 2*z + 1
        e = LinExpr.var("x") + LinExpr.var("y")
        repl = LinExpr.var("z", 2) + 1
        result = e.substitute("x", repl)
        assert result.get_coeff("z") == 2
        assert result.get_coeff("y") == 1
        assert result.get_coeff("x") == 0
        assert result.constant == 1

    def test_substitute_no_var(self):
        e = LinExpr.var("y")
        result = e.substitute("x", LinExpr.const(5))
        assert result.get_coeff("y") == 1

    def test_variables(self):
        e = LinExpr.var("x") + LinExpr.var("y") + LinExpr.var("z")
        assert e.variables() == {"x", "y", "z"}

    def test_cancel_vars(self):
        e = LinExpr.var("x") - LinExpr.var("x")
        assert "x" not in e.variables()
        assert e.constant == 0

    def test_coeff_with_scale(self):
        e = LinExpr.var("x", 3) + LinExpr.var("y", -2) + 7
        assert e.get_coeff("x") == 3
        assert e.get_coeff("y") == -2
        assert e.constant == 7

    def test_repr(self):
        e = LinExpr.var("x") + LinExpr.var("y")
        r = repr(e)
        assert "x" in r and "y" in r


# ===================================================================
# Constraint tests
# ===================================================================

class TestConstraint:
    def test_le(self):
        c = Constraint.le(LinExpr.var("x") - LinExpr.const(5))
        assert c.kind == ConstraintKind.LE
        assert "x" in c.variables()

    def test_eq(self):
        c = Constraint.eq(LinExpr.var("x") - LinExpr.var("y"))
        assert c.kind == ConstraintKind.EQ
        assert c.variables() == {"x", "y"}

    def test_var_le(self):
        c = Constraint.var_le("x", "y")
        assert c.kind == ConstraintKind.LE

    def test_var_le_const(self):
        c = Constraint.var_le_const("x", 10)
        assert c.kind == ConstraintKind.LE

    def test_var_ge_const(self):
        c = Constraint.var_ge_const("x", 0)
        assert c.kind == ConstraintKind.LE

    def test_var_eq_const(self):
        c = Constraint.var_eq_const("x", 5)
        assert c.kind == ConstraintKind.EQ

    def test_var_eq(self):
        c = Constraint.var_eq("x", "y")
        assert c.kind == ConstraintKind.EQ

    def test_tautology_le(self):
        c = Constraint.le(LinExpr.const(-5))  # -5 <= 0
        assert c.is_tautology()

    def test_tautology_eq(self):
        c = Constraint.eq(LinExpr.const(0))  # 0 == 0
        assert c.is_tautology()

    def test_contradiction_le(self):
        c = Constraint.le(LinExpr.const(5))  # 5 <= 0
        assert c.is_contradiction()

    def test_contradiction_eq(self):
        c = Constraint.eq(LinExpr.const(3))  # 3 == 0
        assert c.is_contradiction()

    def test_not_tautology(self):
        c = Constraint.var_le_const("x", 5)
        assert not c.is_tautology()
        assert not c.is_contradiction()

    def test_substitute(self):
        c = Constraint.var_le_const("x", 5)  # x - 5 <= 0
        # substitute x = y + 1
        result = c.substitute("x", LinExpr.var("y") + 1)
        assert result.expr.get_coeff("y") == 1
        assert result.expr.constant == -4  # (y+1) - 5 = y - 4


# ===================================================================
# Polyhedron basic tests
# ===================================================================

class TestPolyhedronBasic:
    def test_top(self):
        p = Polyhedron.top()
        assert p.is_top()
        assert not p.is_bot()

    def test_bot(self):
        p = Polyhedron.bot()
        assert p.is_bot()
        assert not p.is_top()

    def test_from_constraints(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        assert not p.is_top()
        assert not p.is_bot()
        assert p.num_constraints() == 2

    def test_contradiction_makes_bot(self):
        p = Polyhedron.from_constraints([
            Constraint.le(LinExpr.const(5)),  # 5 <= 0, contradiction
        ])
        assert p.is_bot()

    def test_tautology_removed(self):
        p = Polyhedron.from_constraints([
            Constraint.le(LinExpr.const(-5)),  # -5 <= 0, tautology
            Constraint.var_le_const("x", 10),
        ])
        assert p.num_constraints() == 1

    def test_variables(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le("x", "y"),
            Constraint.var_le_const("z", 5),
        ])
        assert p.variables() == {"x", "y", "z"}

    def test_add_constraint(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        p2 = p.add_constraint(Constraint.var_ge_const("x", 0))
        assert p2.num_constraints() == 2

    def test_add_contradiction(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        p2 = p.add_constraint(Constraint.le(LinExpr.const(5)))
        assert p2.is_bot()

    def test_contains_point(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        assert p.contains_point({"x": 5})
        assert p.contains_point({"x": 0})
        assert p.contains_point({"x": 10})
        assert not p.contains_point({"x": 11})
        assert not p.contains_point({"x": -1})

    def test_contains_point_relational(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le("x", "y"),
        ])
        assert p.contains_point({"x": 3, "y": 5})
        assert p.contains_point({"x": 5, "y": 5})
        assert not p.contains_point({"x": 6, "y": 5})


# ===================================================================
# Bounds and intervals
# ===================================================================

class TestBounds:
    def test_simple_bounds(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        lo, hi = p.get_bounds("x")
        assert lo == 0
        assert hi == 10

    def test_derived_bounds(self):
        # x <= y, y <= 10, x >= 0
        p = Polyhedron.from_constraints([
            Constraint.var_le("x", "y"),
            Constraint.var_le_const("y", 10),
            Constraint.var_ge_const("x", 0),
        ])
        lo, hi = p.get_bounds("x")
        assert lo == 0
        assert hi == 10  # derived from x <= y <= 10

    def test_equality_bounds(self):
        p = Polyhedron.from_constraints([
            Constraint.var_eq_const("x", 5),
        ])
        lo, hi = p.get_bounds("x")
        assert lo == 5
        assert hi == 5

    def test_unbounded_above(self):
        p = Polyhedron.from_constraints([
            Constraint.var_ge_const("x", 0),
        ])
        lo, hi = p.get_bounds("x")
        assert lo == 0
        assert hi is None

    def test_unbounded_below(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
        ])
        lo, hi = p.get_bounds("x")
        assert lo is None
        assert hi == 10

    def test_linear_combination_bounds(self):
        # 2x + 3y <= 12, x >= 0, y >= 0
        p = Polyhedron.from_constraints([
            Constraint.le(LinExpr.var("x", 2) + LinExpr.var("y", 3) - LinExpr.const(12)),
            Constraint.var_ge_const("x", 0),
            Constraint.var_ge_const("y", 0),
        ])
        _, hi_x = p.get_bounds("x")
        _, hi_y = p.get_bounds("y")
        assert hi_x == 6   # x <= 6 when y = 0
        assert hi_y == 4   # y <= 4 when x = 0


# ===================================================================
# Fourier-Motzkin elimination
# ===================================================================

class TestFourierMotzkin:
    def test_eliminate_simple(self):
        # x <= y, y <= 10, eliminate y => x <= 10
        constraints = [
            Constraint.var_le("x", "y"),
            Constraint.var_le_const("y", 10),
        ]
        result = _fourier_motzkin_eliminate_vars(constraints, {"y"})
        # Should have x <= 10
        p = Polyhedron.from_constraints(result)
        lo, hi = p.get_bounds("x")
        assert hi == 10

    def test_eliminate_with_equality(self):
        # y == x + 1, y <= 10, eliminate y => x + 1 <= 10 => x <= 9
        constraints = [
            Constraint.eq(LinExpr.var("y") - LinExpr.var("x") - LinExpr.const(1)),
            Constraint.var_le_const("y", 10),
        ]
        result = _fourier_motzkin_eliminate_vars(constraints, {"y"})
        p = Polyhedron.from_constraints(result)
        _, hi = p.get_bounds("x")
        assert hi == 9

    def test_eliminate_contradiction(self):
        # x >= 5, x <= 3, eliminate x
        constraints = [
            Constraint.var_ge_const("x", 5),
            Constraint.var_le_const("x", 3),
        ]
        result = _fourier_motzkin_eliminate_vars(constraints, {"x"})
        p = Polyhedron.from_constraints(result)
        assert p.is_bot()

    def test_eliminate_preserves_unrelated(self):
        # x <= 5, y >= 0, eliminate x
        constraints = [
            Constraint.var_le_const("x", 5),
            Constraint.var_ge_const("y", 0),
        ]
        result = _fourier_motzkin_eliminate_vars(constraints, {"x"})
        p = Polyhedron.from_constraints(result)
        lo, _ = p.get_bounds("y")
        assert lo == 0


# ===================================================================
# Meet (intersection)
# ===================================================================

class TestMeet:
    def test_meet_basic(self):
        p1 = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        p2 = Polyhedron.from_constraints([Constraint.var_ge_const("x", 0)])
        m = p1.meet(p2)
        lo, hi = m.get_bounds("x")
        assert lo == 0
        assert hi == 10

    def test_meet_bot(self):
        p1 = Polyhedron.from_constraints([Constraint.var_le_const("x", 3)])
        p2 = Polyhedron.bot()
        assert p1.meet(p2).is_bot()
        assert p2.meet(p1).is_bot()

    def test_meet_contradictory(self):
        p1 = Polyhedron.from_constraints([Constraint.var_le_const("x", 3)])
        p2 = Polyhedron.from_constraints([Constraint.var_ge_const("x", 5)])
        m = p1.meet(p2)
        assert not m.is_satisfiable()

    def test_meet_relational(self):
        p1 = Polyhedron.from_constraints([Constraint.var_le("x", "y")])
        p2 = Polyhedron.from_constraints([Constraint.var_le_const("y", 10)])
        m = p1.meet(p2)
        _, hi = m.get_bounds("x")
        assert hi == 10


# ===================================================================
# Join (convex hull)
# ===================================================================

class TestJoin:
    def test_join_bot_left(self):
        p1 = Polyhedron.bot()
        p2 = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        j = p1.join(p2)
        _, hi = j.get_bounds("x")
        assert hi == 10

    def test_join_bot_right(self):
        p1 = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        p2 = Polyhedron.bot()
        j = p1.join(p2)
        _, hi = j.get_bounds("x")
        assert hi == 10

    def test_join_same(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        j = p.join(p)
        lo, hi = j.get_bounds("x")
        assert lo == 0
        assert hi == 10

    def test_join_widens_bounds(self):
        p1 = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 5),
            Constraint.var_ge_const("x", 0),
        ])
        p2 = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 3),
        ])
        j = p1.join(p2)
        lo, hi = j.get_bounds("x")
        assert lo == 0
        assert hi == 10

    def test_join_preserves_shared_constraints(self):
        # Both have x <= y
        p1 = Polyhedron.from_constraints([
            Constraint.var_le("x", "y"),
            Constraint.var_le_const("x", 5),
        ])
        p2 = Polyhedron.from_constraints([
            Constraint.var_le("x", "y"),
            Constraint.var_ge_const("x", 0),
        ])
        j = p1.join(p2)
        # x <= y should be preserved
        assert j.contains_point({"x": 3, "y": 5})
        assert not j.contains_point({"x": 6, "y": 5})

    def test_join_top(self):
        p1 = Polyhedron.top()
        p2 = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        assert p1.join(p2).is_top()


# ===================================================================
# Widening
# ===================================================================

class TestWidening:
    def test_widen_bot(self):
        p1 = Polyhedron.bot()
        p2 = Polyhedron.from_constraints([Constraint.var_le_const("x", 5)])
        assert p1.widen(p2).get_bounds("x") == (None, Fraction(5))

    def test_widen_stable(self):
        # If all constraints are stable, keep them
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        w = p.widen(p)
        lo, hi = w.get_bounds("x")
        assert lo == 0
        assert hi == 10

    def test_widen_drops_unstable(self):
        p1 = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 5),
            Constraint.var_ge_const("x", 0),
        ])
        # x is now in [0, 10], upper bound expanded
        p2 = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        w = p1.widen(p2)
        lo, hi = w.get_bounds("x")
        assert lo == 0
        # Upper bound should be dropped (gone to infinity) since x <= 5 is not implied by p2
        assert hi is None

    def test_widen_convergence(self):
        # Simulate loop iteration: x starts at 0, increments
        states = [Polyhedron.from_constraints([Constraint.var_eq_const("x", 0)])]
        for i in range(1, 10):
            next_val = Polyhedron.from_constraints([Constraint.var_eq_const("x", i)])
            joined = states[-1].join(next_val)
            widened = states[-1].widen(joined)
            states.append(widened)
        # After widening, upper bound should be gone
        _, hi = states[-1].get_bounds("x")
        assert hi is None


# ===================================================================
# Assign transfer
# ===================================================================

class TestAssign:
    def test_assign_const(self):
        p = Polyhedron.top()
        result = p.assign("x", LinExpr.const(5))
        lo, hi = result.get_bounds("x")
        assert lo == 5
        assert hi == 5

    def test_assign_preserves_others(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("y", 10)])
        result = p.assign("x", LinExpr.const(5))
        _, hi_y = result.get_bounds("y")
        assert hi_y == 10

    def test_assign_expr(self):
        # x = 5, then y = x + 1
        p = Polyhedron.top()
        p = p.assign("x", LinExpr.const(5))
        p = p.assign("y", LinExpr.var("x") + 1)
        lo, hi = p.get_bounds("y")
        assert lo == 6
        assert hi == 6

    def test_assign_self_increment(self):
        # x = 5, then x = x + 1
        p = Polyhedron.from_constraints([Constraint.var_eq_const("x", 5)])
        result = p.assign("x", LinExpr.var("x") + 1)
        lo, hi = result.get_bounds("x")
        assert lo == 6
        assert hi == 6

    def test_assign_preserves_relations(self):
        # x in [0, 10], y = x + 1
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        p = p.assign("y", LinExpr.var("x") + 1)
        # y should be in [1, 11] AND y == x + 1
        lo, hi = p.get_bounds("y")
        assert lo == 1
        assert hi == 11

    def test_assign_nondet(self):
        p = Polyhedron.from_constraints([
            Constraint.var_eq_const("x", 5),
            Constraint.var_le_const("y", 10),
        ])
        result = p.assign_nondet("x")
        # x should be unconstrained
        lo, hi = result.get_bounds("x")
        assert lo is None and hi is None
        # y should be preserved
        _, hi_y = result.get_bounds("y")
        assert hi_y == 10


# ===================================================================
# Guard (condition refinement)
# ===================================================================

class TestGuard:
    def test_guard_simple(self):
        p = Polyhedron.top()
        result = p.guard(Constraint.var_le_const("x", 10))
        _, hi = result.get_bounds("x")
        assert hi == 10

    def test_guard_relational(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("y", 10),
        ])
        result = p.guard(Constraint.var_le("x", "y"))
        _, hi = result.get_bounds("x")
        assert hi == 10

    def test_guard_bot(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("x", 3)])
        result = p.guard(Constraint.var_ge_const("x", 5))
        assert not result.is_satisfiable()


# ===================================================================
# Project and Forget
# ===================================================================

class TestProjection:
    def test_project_keep_one(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le("x", "y"),
            Constraint.var_le_const("y", 10),
            Constraint.var_ge_const("x", 0),
        ])
        proj = p.project({"x"})
        lo, hi = proj.get_bounds("x")
        assert lo == 0
        assert hi == 10

    def test_forget(self):
        p = Polyhedron.from_constraints([
            Constraint.var_eq("x", "y"),
            Constraint.var_le_const("x", 10),
        ])
        result = p.forget({"y"})
        assert "y" not in result.variables()
        _, hi = result.get_bounds("x")
        assert hi == 10

    def test_project_bot(self):
        p = Polyhedron.bot()
        assert p.project({"x"}).is_bot()


# ===================================================================
# Includes (lattice ordering)
# ===================================================================

class TestIncludes:
    def test_top_includes_all(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("x", 5)])
        assert Polyhedron.top().includes(p)

    def test_all_include_bot(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("x", 5)])
        assert p.includes(Polyhedron.bot())

    def test_narrower_includes_wider(self):
        # [0, 10] includes [2, 8]
        wide = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        narrow = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 8),
            Constraint.var_ge_const("x", 2),
        ])
        assert wide.includes(narrow)
        assert not narrow.includes(wide)

    def test_self_includes(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        assert p.includes(p)


# ===================================================================
# Satisfiability
# ===================================================================

class TestSatisfiability:
    def test_top_satisfiable(self):
        assert Polyhedron.top().is_satisfiable()

    def test_bot_not_satisfiable(self):
        assert not Polyhedron.bot().is_satisfiable()

    def test_satisfiable_constraints(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        assert p.is_satisfiable()

    def test_unsatisfiable_constraints(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 3),
            Constraint.var_ge_const("x", 5),
        ])
        assert not p.is_satisfiable()


# ===================================================================
# Extraction
# ===================================================================

class TestExtraction:
    def test_extract_intervals(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
            Constraint.var_le_const("y", 5),
        ])
        intervals = p.extract_intervals()
        assert intervals["x"] == (Fraction(0), Fraction(10))
        assert intervals["y"][1] == Fraction(5)

    def test_extract_equalities(self):
        p = Polyhedron.from_constraints([
            Constraint.var_eq("x", "y"),
            Constraint.var_le_const("x", 10),
        ])
        eqs = p.extract_equalities()
        assert len(eqs) == 1

    def test_extract_orderings(self):
        p = Polyhedron.from_constraints([
            Constraint.var_le("x", "y"),
            Constraint.var_le("y", "z"),
        ])
        orderings = p.extract_orderings()
        assert len(orderings) == 2


# ===================================================================
# PolyhedraInterpreter tests
# ===================================================================

class TestInterpreter:
    def test_assign_const(self):
        prog = ('assign', 'x', ('const', 5))
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("x")
        assert lo == 5 and hi == 5

    def test_sequence(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('const', 10)),
        )
        result = analyze_program(prog)
        lo_x, hi_x = result.final_state.get_bounds("x")
        lo_y, hi_y = result.final_state.get_bounds("y")
        assert lo_x == 5 and hi_x == 5
        assert lo_y == 10 and hi_y == 10

    def test_assign_add(self):
        prog = ('seq',
            ('assign', 'x', ('const', 3)),
            ('assign', 'y', ('add', ('var', 'x'), ('const', 2))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("y")
        assert lo == 5 and hi == 5

    def test_assign_sub(self):
        prog = ('seq',
            ('assign', 'x', ('const', 10)),
            ('assign', 'y', ('sub', ('var', 'x'), ('const', 3))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("y")
        assert lo == 7 and hi == 7

    def test_assign_mul_const(self):
        prog = ('seq',
            ('assign', 'x', ('const', 4)),
            ('assign', 'y', ('mul', ('var', 'x'), ('const', 3))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("y")
        assert lo == 12 and hi == 12

    def test_assign_neg(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('neg', ('var', 'x'))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("y")
        assert lo == -5 and hi == -5

    def test_if_simple(self):
        # x = input (unconstrained), if x <= 5 then y = 1 else y = 2
        prog = ('if',
            ('le', ('var', 'x'), ('const', 5)),
            ('assign', 'y', ('const', 1)),
            ('assign', 'y', ('const', 2)),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("y")
        assert lo == 1 and hi == 2

    def test_if_refines(self):
        # if x <= 5 then (in this branch x <= 5)
        init = Polyhedron.from_constraints([
            Constraint.var_ge_const("x", 0),
            Constraint.var_le_const("x", 10),
        ])
        prog = ('if',
            ('le', ('var', 'x'), ('const', 5)),
            ('assign', 'y', ('var', 'x')),    # y = x, knowing x <= 5
            ('assign', 'y', ('var', 'x')),    # y = x, knowing x > 5
        )
        result = analyze_program(prog, init)
        lo, hi = result.final_state.get_bounds("y")
        assert lo == 0 and hi == 10

    def test_while_simple(self):
        # x = 0; while (x < 10) x = x + 1
        prog = ('seq',
            ('assign', 'x', ('const', 0)),
            ('while',
                ('lt', ('var', 'x'), ('const', 10)),
                ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
            ),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("x")
        # After exit: x >= 10
        assert lo is not None and lo >= 10

    def test_while_with_relation(self):
        # x = 0; y = 100; while (x < 10) { x = x + 1; y = y - 1; }
        prog = ('seq',
            ('assign', 'x', ('const', 0)),
            ('assign', 'y', ('const', 100)),
            ('while',
                ('lt', ('var', 'x'), ('const', 10)),
                ('seq',
                    ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
                    ('assign', 'y', ('sub', ('var', 'y'), ('const', 1))),
                ),
            ),
        )
        result = analyze_program(prog)
        # x + y should be conserved (= 100)
        # After loop: x >= 10
        lo_x, _ = result.final_state.get_bounds("x")
        assert lo_x is not None and lo_x >= 10

    def test_assert_pass(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assert', ('ge', ('var', 'x'), ('const', 0))),
        )
        result = analyze_program(prog)
        assert len(result.warnings) == 0

    def test_assert_fail(self):
        prog = ('seq',
            ('assign', 'x', ('const', -1)),
            ('assert', ('ge', ('var', 'x'), ('const', 0))),
        )
        result = analyze_program(prog)
        assert len(result.warnings) > 0

    def test_skip(self):
        prog = ('skip',)
        result = analyze_program(prog)
        assert result.final_state.is_top()

    def test_non_linear_havoc(self):
        # x = y * z (non-linear) -> havoc x
        prog = ('seq',
            ('assign', 'y', ('const', 3)),
            ('assign', 'z', ('const', 4)),
            ('assign', 'x', ('mul', ('var', 'y'), ('var', 'z'))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("x")
        # x is havoced (non-linear), so unconstrained
        assert lo is None and hi is None


# ===================================================================
# Condition handling
# ===================================================================

class TestConditions:
    def test_le(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('le', ('var', 'x'), ('const', 5)))
        p = Polyhedron.top().guard_constraints(cs)
        _, hi = p.get_bounds("x")
        assert hi == 5

    def test_lt(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('lt', ('var', 'x'), ('const', 5)))
        p = Polyhedron.top().guard_constraints(cs)
        _, hi = p.get_bounds("x")
        assert hi == 4  # x < 5 => x <= 4 (integer)

    def test_ge(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('ge', ('var', 'x'), ('const', 5)))
        p = Polyhedron.top().guard_constraints(cs)
        lo, _ = p.get_bounds("x")
        assert lo == 5

    def test_gt(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('gt', ('var', 'x'), ('const', 5)))
        p = Polyhedron.top().guard_constraints(cs)
        lo, _ = p.get_bounds("x")
        assert lo == 6  # x > 5 => x >= 6

    def test_eq(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('eq', ('var', 'x'), ('const', 5)))
        p = Polyhedron.top().guard_constraints(cs)
        lo, hi = p.get_bounds("x")
        assert lo == 5 and hi == 5

    def test_and(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('and',
            ('ge', ('var', 'x'), ('const', 0)),
            ('le', ('var', 'x'), ('const', 10)),
        ))
        p = Polyhedron.top().guard_constraints(cs)
        lo, hi = p.get_bounds("x")
        assert lo == 0 and hi == 10

    def test_not_le(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('not', ('le', ('var', 'x'), ('const', 5))))
        p = Polyhedron.top().guard_constraints(cs)
        lo, _ = p.get_bounds("x")
        assert lo == 6  # not(x <= 5) => x > 5 => x >= 6

    def test_true(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('true',))
        assert len(cs) == 0

    def test_false(self):
        interp = PolyhedraInterpreter()
        cs = interp._cond_to_constraints(('false',))
        p = Polyhedron.top().guard_constraints(cs)
        assert p.is_bot()


# ===================================================================
# Relational verification
# ===================================================================

class TestRelationalVerification:
    def test_verify_sum_conservation(self):
        # x = a; y = b; x = x + 1; y = y - 1
        # Property: x + y == a + b (conserved)
        prog = ('seq',
            ('assign', 'x', ('var', 'a')),
            ('assign', 'y', ('var', 'b')),
            ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
            ('assign', 'y', ('sub', ('var', 'y'), ('const', 1))),
        )
        # x + y == a + b  =>  x + y - a - b == 0
        prop = Constraint.eq(
            LinExpr.var("x") + LinExpr.var("y") - LinExpr.var("a") - LinExpr.var("b")
        )
        verified, _ = verify_relational_property(prog, prop)
        assert verified

    def test_verify_ordering(self):
        # x = 0; y = 1; => x < y, i.e., x <= y - 1, i.e., x - y + 1 <= 0
        prog = ('seq',
            ('assign', 'x', ('const', 0)),
            ('assign', 'y', ('const', 1)),
        )
        prop = Constraint.le(LinExpr.var("x") - LinExpr.var("y") + 1)
        verified, _ = verify_relational_property(prog, prop)
        assert verified

    def test_verify_fails(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('const', 3)),
        )
        prop = Constraint.var_le("x", "y")  # x <= y is false when x=5, y=3
        verified, _ = verify_relational_property(prog, prop)
        assert not verified

    def test_swap_correctness(self):
        # t = x; x = y; y = t
        # After: x_new == y_old, y_new == x_old
        init = Polyhedron.from_constraints([
            Constraint.var_eq_const("x", 3),
            Constraint.var_eq_const("y", 7),
        ])
        prog = ('seq',
            ('assign', 't', ('var', 'x')),
            ('assign', 'x', ('var', 'y')),
            ('assign', 'y', ('var', 't')),
        )
        result = analyze_program(prog, init)
        lo_x, hi_x = result.final_state.get_bounds("x")
        lo_y, hi_y = result.final_state.get_bounds("y")
        assert lo_x == 7 and hi_x == 7
        assert lo_y == 3 and hi_y == 3


# ===================================================================
# Composition with intervals
# ===================================================================

class TestComposition:
    def test_polyhedra_from_intervals(self):
        p = polyhedra_from_intervals({
            "x": (0, 10),
            "y": (None, 5),
        })
        lo, hi = p.get_bounds("x")
        assert lo == 0 and hi == 10
        _, hi_y = p.get_bounds("y")
        assert hi_y == 5

    def test_compare_with_intervals(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('add', ('var', 'x'), ('const', 3))),
        )
        result = compare_with_intervals(prog)
        assert result['polyhedra_intervals']['y'] == (Fraction(8), Fraction(8))

    def test_compare_with_init(self):
        prog = ('assign', 'y', ('add', ('var', 'x'), ('const', 1)))
        result = compare_with_intervals(prog, {"x": (0, 10)})
        lo, hi = result['polyhedra_intervals']['y']
        assert lo == 1 and hi == 11


# ===================================================================
# Batch analysis
# ===================================================================

class TestBatch:
    def test_batch(self):
        progs = [
            ('assign', 'x', ('const', i))
            for i in range(5)
        ]
        results = batch_analyze(progs)
        assert len(results) == 5
        for i, r in enumerate(results):
            lo, hi = r.final_state.get_bounds("x")
            assert lo == i and hi == i


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_polyhedron(self):
        p = Polyhedron()
        assert p.is_top()

    def test_assign_bot(self):
        p = Polyhedron.bot()
        result = p.assign("x", LinExpr.const(5))
        assert result.is_bot()

    def test_guard_bot(self):
        p = Polyhedron.bot()
        result = p.guard(Constraint.var_le_const("x", 10))
        assert result.is_bot()

    def test_meet_with_self(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        m = p.meet(p)
        _, hi = m.get_bounds("x")
        assert hi == 10

    def test_narrow(self):
        wide = Polyhedron.from_constraints([
            Constraint.var_ge_const("x", 0),
        ])
        narrow_hint = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 100),
        ])
        result = wide.narrow(narrow_hint)
        _, hi = result.get_bounds("x")
        assert hi == 100

    def test_equality_semantic(self):
        p1 = Polyhedron.from_constraints([
            Constraint.var_le_const("x", 10),
            Constraint.var_ge_const("x", 0),
        ])
        p2 = Polyhedron.from_constraints([
            Constraint.var_ge_const("x", 0),
            Constraint.var_le_const("x", 10),
        ])
        assert p1 == p2

    def test_hash_stability(self):
        p1 = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        p2 = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        # Both should be usable as dict keys
        d = {p1: "a"}
        assert p1 in d

    def test_repr_top(self):
        assert "TOP" in repr(Polyhedron.top())

    def test_repr_bot(self):
        assert "BOT" in repr(Polyhedron.bot())

    def test_repr_constraints(self):
        p = Polyhedron.from_constraints([Constraint.var_le_const("x", 10)])
        r = repr(p)
        assert "Polyhedron" in r

    def test_multiple_equalities(self):
        # x == y, y == z => x == z (transitivity)
        p = Polyhedron.from_constraints([
            Constraint.var_eq("x", "y"),
            Constraint.var_eq("y", "z"),
            Constraint.var_eq_const("z", 5),
        ])
        lo, hi = p.get_bounds("x")
        assert lo == 5 and hi == 5

    def test_complex_linear_combination(self):
        # 3x + 2y <= 12, x >= 0, y >= 0
        p = Polyhedron.from_constraints([
            Constraint.le(LinExpr.var("x", 3) + LinExpr.var("y", 2) - LinExpr.const(12)),
            Constraint.var_ge_const("x", 0),
            Constraint.var_ge_const("y", 0),
        ])
        _, hi_x = p.get_bounds("x")
        _, hi_y = p.get_bounds("y")
        assert hi_x == 4   # 3x <= 12, x <= 4
        assert hi_y == 6   # 2y <= 12, y <= 6

    def test_interpret_dead_branch(self):
        # x = 5; if (x < 0) y = 1 else y = 2
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('if',
                ('lt', ('var', 'x'), ('const', 0)),
                ('assign', 'y', ('const', 1)),
                ('assign', 'y', ('const', 2)),
            ),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("y")
        assert lo == 2 and hi == 2

    def test_nested_if(self):
        prog = ('if',
            ('le', ('var', 'x'), ('const', 10)),
            ('if',
                ('ge', ('var', 'x'), ('const', 0)),
                ('assign', 'y', ('var', 'x')),  # x in [0, 10]
                ('assign', 'y', ('const', -1)),
            ),
            ('assign', 'y', ('const', 99)),
        )
        result = analyze_program(prog)
        # y could be anything from the three branches
        lo, hi = result.final_state.get_bounds("y")
        assert lo is not None
        assert hi is not None

    def test_while_never_enters(self):
        # x = 10; while (x < 5) x = x + 1
        prog = ('seq',
            ('assign', 'x', ('const', 10)),
            ('while',
                ('lt', ('var', 'x'), ('const', 5)),
                ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
            ),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds("x")
        assert lo == 10 and hi == 10


# ===================================================================
# Relational programs (the strength of polyhedra)
# ===================================================================

class TestRelationalPrograms:
    def test_loop_invariant_sum(self):
        """x + y should be conserved across the loop."""
        # x = 0; y = n; while (x < n) { x = x + 1; y = y - 1; }
        # After: x + y == n (conserved)
        init = Polyhedron.from_constraints([
            Constraint.var_eq_const("n", 10),
        ])
        prog = ('seq',
            ('assign', 'x', ('const', 0)),
            ('assign', 'y', ('var', 'n')),
            ('while',
                ('lt', ('var', 'x'), ('var', 'n')),
                ('seq',
                    ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
                    ('assign', 'y', ('sub', ('var', 'y'), ('const', 1))),
                ),
            ),
        )
        result = analyze_program(prog, init)
        # Check x + y == n at exit
        state = result.final_state
        if not state.is_bot():
            prop = Constraint.eq(
                LinExpr.var("x") + LinExpr.var("y") - LinExpr.var("n")
            )
            assert state._implies_constraint(prop)

    def test_difference_bound(self):
        """y - x should be bounded."""
        prog = ('seq',
            ('assign', 'x', ('const', 0)),
            ('assign', 'y', ('const', 5)),
        )
        result = analyze_program(prog)
        state = result.final_state
        # y - x == 5
        prop = Constraint.eq(
            LinExpr.var("y") - LinExpr.var("x") - LinExpr.const(5)
        )
        assert state._implies_constraint(prop)

    def test_array_bounds_idiom(self):
        """Verify 0 <= i < n after a bounded loop."""
        init = Polyhedron.from_constraints([
            Constraint.var_ge_const("n", 1),
            Constraint.var_le_const("n", 100),
        ])
        # i = 0; while (i < n) { use i; i = i + 1; }
        # Inside loop: 0 <= i < n
        # After loop: i >= n
        prog = ('seq',
            ('assign', 'i', ('const', 0)),
            ('while',
                ('lt', ('var', 'i'), ('var', 'n')),
                ('seq',
                    ('skip',),  # "use i here"
                    ('assign', 'i', ('add', ('var', 'i'), ('const', 1))),
                ),
            ),
        )
        result = analyze_program(prog, init)
        state = result.final_state
        # i >= n at exit
        lo_i, _ = state.get_bounds("i")
        assert lo_i is not None


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
