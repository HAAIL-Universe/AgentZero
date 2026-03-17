"""
Tests for V178: Zone Abstract Domain
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fractions import Fraction
from zone import (
    Zone, ZoneConstraint, ZoneInterpreter,
    upper_bound, lower_bound, diff_bound, eq_constraint, var_eq_const,
    zone_from_intervals, verify_zone_property, compare_with_octagon,
    INF,
)


# ===========================================================================
# Basic Construction
# ===========================================================================

class TestConstruction:
    def test_top(self):
        z = Zone.top()
        assert z.is_top()
        assert not z.is_bot()

    def test_bot(self):
        z = Zone.bot()
        assert z.is_bot()
        assert not z.is_top()

    def test_from_empty_constraints(self):
        z = Zone.from_constraints([])
        assert z.is_top()

    def test_single_upper_bound(self):
        z = Zone.from_constraints([upper_bound("x", 10)])
        assert z.get_upper_bound("x") == Fraction(10)
        assert z.get_lower_bound("x") == -INF

    def test_single_lower_bound(self):
        z = Zone.from_constraints([lower_bound("x", 3)])
        assert z.get_lower_bound("x") == Fraction(3)
        assert z.get_upper_bound("x") == INF

    def test_both_bounds(self):
        z = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        assert z.get_lower_bound("x") == Fraction(0)
        assert z.get_upper_bound("x") == Fraction(10)
        assert z.get_interval("x") == (Fraction(0), Fraction(10))

    def test_diff_constraint(self):
        z = Zone.from_constraints([
            diff_bound("x", "y", 5),  # x - y <= 5
            lower_bound("y", 0),
            upper_bound("y", 10),
        ])
        assert z.get_diff_bound("x", "y") == Fraction(5)
        # Transitive: x <= y + 5, y <= 10 => x <= 15
        assert z.get_upper_bound("x") == Fraction(15)

    def test_equality_constraint(self):
        cs = eq_constraint("x", "y")
        z = Zone.from_constraints(cs + [lower_bound("x", 0), upper_bound("x", 10)])
        assert z.get_diff_bound("x", "y") == Fraction(0)
        assert z.get_diff_bound("y", "x") == Fraction(0)
        # y inherits x's bounds through transitivity
        assert z.get_lower_bound("y") == Fraction(0)
        assert z.get_upper_bound("y") == Fraction(10)

    def test_var_eq_const(self):
        cs = var_eq_const("x", 42)
        z = Zone.from_constraints(cs)
        assert z.get_lower_bound("x") == Fraction(42)
        assert z.get_upper_bound("x") == Fraction(42)

    def test_unsatisfiable_constraints(self):
        z = Zone.from_constraints([
            lower_bound("x", 10),
            upper_bound("x", 5),
        ])
        assert z.is_bot()

    def test_unsatisfiable_diff(self):
        z = Zone.from_constraints([
            diff_bound("x", "y", -1),   # x - y <= -1 => x < y
            diff_bound("y", "x", -1),   # y - x <= -1 => y < x
        ])
        assert z.is_bot()  # x < y AND y < x is impossible

    def test_multiple_variables(self):
        z = Zone.from_constraints([
            upper_bound("x", 5),
            upper_bound("y", 10),
            upper_bound("z", 15),
            diff_bound("x", "y", 0),  # x <= y
            diff_bound("y", "z", 0),  # y <= z
        ])
        assert z.get_diff_bound("x", "z") == Fraction(0)  # Transitive: x <= z


# ===========================================================================
# Transitive Closure (Floyd-Warshall)
# ===========================================================================

class TestClosure:
    def test_transitive_diff_chain(self):
        """x - y <= 3, y - z <= 4 => x - z <= 7."""
        z = Zone.from_constraints([
            diff_bound("x", "y", 3),
            diff_bound("y", "z", 4),
        ])
        assert z.get_diff_bound("x", "z") == Fraction(7)

    def test_transitive_bound_derivation(self):
        """y >= 5, x - y <= 2 => x <= 7 (derived via zero var)."""
        z = Zone.from_constraints([
            lower_bound("y", 5),   # z0 - y <= -5
            diff_bound("x", "y", 2),  # x - y <= 2
        ])
        # x - z0 = (x - y) + (y - z0) <= 2 + (-(-5)) ... wait
        # x - z0 <= (x - y) + (y - z0): x - y <= 2, y - z0 not constrained
        # Actually: x - z0 = (x - y) + (y - z0), but y - z0 is the upper bound of y
        # y has lower bound 5, so z0 - y <= -5, not y - z0
        # x = (x - y) + y: upper bound of x = diff(x,y) + upper(y)
        # But upper(y) is INF, so upper(x) is INF
        assert z.get_upper_bound("x") == INF
        # However: x >= y - 2 >= 5 - 2 = 3? NO:
        # diff_bound(x, y, 2) means x - y <= 2, not x >= y - 2 ... well actually it does
        # z0 - x <= (z0 - y) + (y - x). z0 - y <= -5, y - x <= -(x-y) well...
        # DBM: y - x is stored at [var_x][var_y] which is not directly set.
        # After closure: DBM[x_idx][0] = DBM[x_idx][y_idx] + DBM[y_idx][0]
        # DBM[x_idx][y_idx] = ??? That's y - x, not constrained. INF.
        # So lower bound of x is also not constrained. That's correct.
        # The constraint x - y <= 2 + y >= 5 doesn't give us x >= 3 directly in zone.
        # Actually wait: z0 - y <= -5. y - x <= ? (not constrained).
        # So z0 - x <= z0 - y + y - x = -5 + INF = INF. So x >= -INF. Correct.

    def test_long_chain(self):
        """Chain of 5 vars: a-b<=1, b-c<=2, c-d<=3, d-e<=4 => a-e<=10."""
        z = Zone.from_constraints([
            diff_bound("a", "b", 1),
            diff_bound("b", "c", 2),
            diff_bound("c", "d", 3),
            diff_bound("d", "e", 4),
        ])
        assert z.get_diff_bound("a", "e") == Fraction(10)

    def test_bound_tightening(self):
        """Two paths give different bounds; closure takes the tighter one."""
        z = Zone.from_constraints([
            diff_bound("x", "z", 10),   # direct: x - z <= 10
            diff_bound("x", "y", 3),    # via y: x - z <= 3 + 4 = 7
            diff_bound("y", "z", 4),
        ])
        assert z.get_diff_bound("x", "z") == Fraction(7)  # Tighter via y

    def test_self_loop_detection(self):
        """x - y <= 3, y - x <= -5 => negative cycle (x < y AND y - 5 <= x)."""
        z = Zone.from_constraints([
            diff_bound("x", "y", 3),
            diff_bound("y", "x", -5),
        ])
        # x - y <= 3 AND y - x <= -5 => 0 <= x - x <= 3 + (-5) = -2 => 0 <= -2 contradiction
        assert z.is_bot()

    def test_20_var_chain(self):
        """20 variables with x_i - x_{i+1} <= 5 => x_0 - x_19 <= 95."""
        cs = []
        for i in range(19):
            cs.append(diff_bound(f"x{i}", f"x{i+1}", 5))
        z = Zone.from_constraints(cs)
        assert z.get_diff_bound("x0", "x19") == Fraction(95)


# ===========================================================================
# Lattice Operations
# ===========================================================================

class TestLattice:
    def test_join_with_bot(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        assert z.join(Zone.bot()).get_upper_bound("x") == Fraction(5)
        assert Zone.bot().join(z).get_upper_bound("x") == Fraction(5)

    def test_join_widens_bounds(self):
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 5)])
        z2 = Zone.from_constraints([lower_bound("x", 3), upper_bound("x", 10)])
        j = z1.join(z2)
        assert j.get_lower_bound("x") == Fraction(0)
        assert j.get_upper_bound("x") == Fraction(10)

    def test_join_preserves_common_diff(self):
        """Both zones have x - y <= 3, join should keep it."""
        z1 = Zone.from_constraints([diff_bound("x", "y", 3), lower_bound("x", 0)])
        z2 = Zone.from_constraints([diff_bound("x", "y", 3), lower_bound("x", 5)])
        j = z1.join(z2)
        assert j.get_diff_bound("x", "y") == Fraction(3)

    def test_join_different_vars(self):
        z1 = Zone.from_constraints([upper_bound("x", 5)])
        z2 = Zone.from_constraints([upper_bound("y", 10)])
        j = z1.join(z2)
        # x in z2 is unconstrained, y in z1 is unconstrained
        assert j.get_upper_bound("x") == INF
        assert j.get_upper_bound("y") == INF

    def test_meet_tightens_bounds(self):
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        z2 = Zone.from_constraints([lower_bound("x", 3), upper_bound("x", 7)])
        m = z1.meet(z2)
        assert m.get_lower_bound("x") == Fraction(3)
        assert m.get_upper_bound("x") == Fraction(7)

    def test_meet_unsatisfiable(self):
        z1 = Zone.from_constraints([upper_bound("x", 3)])
        z2 = Zone.from_constraints([lower_bound("x", 5)])
        m = z1.meet(z2)
        assert m.is_bot()

    def test_meet_with_bot(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        assert z.meet(Zone.bot()).is_bot()
        assert Zone.bot().meet(z).is_bot()

    def test_includes(self):
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        z2 = Zone.from_constraints([lower_bound("x", 2), upper_bound("x", 8)])
        assert z1.includes(z2)
        assert not z2.includes(z1)

    def test_includes_bot(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        assert z.includes(Zone.bot())
        assert not Zone.bot().includes(z)

    def test_equals(self):
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        z2 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        assert z1.equals(z2)

    def test_widen(self):
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 5)])
        z2 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        w = z1.widen(z2)
        # Upper bound grew from 5 to 10, should widen to INF
        assert w.get_upper_bound("x") == INF
        # Lower bound stable at 0
        assert w.get_lower_bound("x") == Fraction(0)

    def test_narrow(self):
        z1 = Zone.from_constraints([lower_bound("x", 0)])  # x >= 0, x <= INF
        z2 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 100)])
        n = z1.narrow(z2)
        # INF narrowed to 100
        assert n.get_upper_bound("x") == Fraction(100)
        assert n.get_lower_bound("x") == Fraction(0)


# ===========================================================================
# Transfer Functions
# ===========================================================================

class TestTransfer:
    def test_assign_const(self):
        z = Zone.top().assign_const("x", 5)
        assert z.get_lower_bound("x") == Fraction(5)
        assert z.get_upper_bound("x") == Fraction(5)

    def test_assign_const_overwrites(self):
        z = Zone.from_constraints([upper_bound("x", 10)])
        z2 = z.assign_const("x", 3)
        assert z2.get_upper_bound("x") == Fraction(3)
        assert z2.get_lower_bound("x") == Fraction(3)

    def test_assign_var(self):
        z = Zone.from_constraints([lower_bound("y", 2), upper_bound("y", 8)])
        z2 = z.assign_var("x", "y")
        assert z2.get_lower_bound("x") == Fraction(2)
        assert z2.get_upper_bound("x") == Fraction(8)
        # x == y after assignment
        assert z2.get_diff_bound("x", "y") == Fraction(0)
        assert z2.get_diff_bound("y", "x") == Fraction(0)

    def test_assign_var_preserves_diffs(self):
        """x := y should preserve z's relationship to y."""
        z = Zone.from_constraints([
            lower_bound("y", 5), upper_bound("y", 10),
            diff_bound("z", "y", 3),  # z - y <= 3
        ])
        z2 = z.assign_var("x", "y")
        # x = y, so x - z <= y - z <= ... z - y <= 3 => z - x <= 3
        assert z2.get_diff_bound("z", "x") == Fraction(3)

    def test_assign_var_plus_const(self):
        z = Zone.from_constraints([lower_bound("y", 5), upper_bound("y", 10)])
        z2 = z.assign_var_plus_const("x", "y", 3)
        assert z2.get_lower_bound("x") == Fraction(8)
        assert z2.get_upper_bound("x") == Fraction(13)
        # x - y = 3
        assert z2.get_diff_bound("x", "y") == Fraction(3)
        assert z2.get_diff_bound("y", "x") == Fraction(-3)

    def test_increment(self):
        z = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 5)])
        z2 = z.increment("x", 2)
        assert z2.get_lower_bound("x") == Fraction(2)
        assert z2.get_upper_bound("x") == Fraction(7)

    def test_decrement(self):
        z = Zone.from_constraints([lower_bound("x", 5), upper_bound("x", 10)])
        z2 = z.increment("x", -3)
        assert z2.get_lower_bound("x") == Fraction(2)
        assert z2.get_upper_bound("x") == Fraction(7)

    def test_forget(self):
        z = Zone.from_constraints([
            lower_bound("x", 0), upper_bound("x", 10),
            diff_bound("x", "y", 5),
        ])
        z2 = z.forget("x")
        assert z2.get_upper_bound("x") == INF
        assert z2.get_lower_bound("x") == -INF
        assert z2.get_diff_bound("x", "y") == INF

    def test_guard(self):
        z = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        z2 = z.guard(upper_bound("x", 5))
        assert z2.get_upper_bound("x") == Fraction(5)

    def test_guard_makes_bot(self):
        z = Zone.from_constraints([lower_bound("x", 10)])
        z2 = z.guard(upper_bound("x", 5))
        assert z2.is_bot()

    def test_assign_on_bot(self):
        z = Zone.bot().assign_const("x", 5)
        assert z.is_bot()


# ===========================================================================
# Queries
# ===========================================================================

class TestQueries:
    def test_satisfies_upper(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        assert z.satisfies(upper_bound("x", 5))
        assert z.satisfies(upper_bound("x", 10))
        assert not z.satisfies(upper_bound("x", 3))

    def test_satisfies_diff(self):
        z = Zone.from_constraints([diff_bound("x", "y", 3)])
        assert z.satisfies(diff_bound("x", "y", 3))
        assert z.satisfies(diff_bound("x", "y", 5))
        assert not z.satisfies(diff_bound("x", "y", 2))

    def test_extract_constraints(self):
        z = Zone.from_constraints([
            lower_bound("x", 0),
            upper_bound("x", 10),
        ])
        cs = z.extract_constraints()
        # Should include x <= 10 and x >= 0 (and the self-diff bound x - x <= 0 is excluded)
        assert len(cs) >= 2

    def test_extract_equalities(self):
        z = Zone.from_constraints(eq_constraint("x", "y") + [lower_bound("x", 0)])
        eqs = z.extract_equalities()
        # Should find x - y == 0
        found = any(v1 == "x" and v2 == "y" and c == 0 or
                     v1 == "y" and v2 == "x" and c == 0
                     for v1, v2, c in eqs)
        assert found

    def test_bot_queries(self):
        z = Zone.bot()
        assert z.get_upper_bound("x") is None
        assert z.get_lower_bound("x") is None
        assert z.get_diff_bound("x", "y") is None

    def test_unknown_var(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        assert z.get_upper_bound("unknown") == INF
        assert z.get_lower_bound("unknown") == -INF
        assert z.get_diff_bound("x", "unknown") == INF

    def test_variables(self):
        z = Zone.from_constraints([upper_bound("b", 5), lower_bound("a", 0)])
        assert z.variables() == ["a", "b"]


# ===========================================================================
# Interpreter (with C010 AST)
# ===========================================================================

# Minimal AST stubs for testing without C010 import
class NumberLit:
    def __init__(self, value):
        self.value = value

class Identifier:
    def __init__(self, name):
        self.name = name

class BinOp:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class LetDecl:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Assign:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Block:
    def __init__(self, stmts):
        self.stmts = stmts

class IfStmt:
    def __init__(self, cond, then_body, else_body=None):
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body

class WhileStmt:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body


class TestInterpreter:
    def test_const_assignment(self):
        interp = ZoneInterpreter()
        stmts = [LetDecl("x", NumberLit(5))]
        z = interp.analyze(stmts)
        assert z.get_lower_bound("x") == Fraction(5)
        assert z.get_upper_bound("x") == Fraction(5)

    def test_var_assignment(self):
        interp = ZoneInterpreter()
        stmts = [
            LetDecl("x", NumberLit(10)),
            LetDecl("y", Identifier("x")),
        ]
        z = interp.analyze(stmts)
        assert z.get_lower_bound("y") == Fraction(10)
        assert z.get_upper_bound("y") == Fraction(10)

    def test_increment_assignment(self):
        interp = ZoneInterpreter()
        stmts = [
            LetDecl("x", NumberLit(5)),
            Assign("x", BinOp("+", Identifier("x"), NumberLit(3))),
        ]
        z = interp.analyze(stmts)
        assert z.get_lower_bound("x") == Fraction(8)
        assert z.get_upper_bound("x") == Fraction(8)

    def test_if_then_else(self):
        interp = ZoneInterpreter()
        # x = 5; if (x <= 3) { y = 1 } else { y = 2 }
        stmts = [
            LetDecl("x", NumberLit(5)),
            IfStmt(
                BinOp("<=", Identifier("x"), NumberLit(3)),
                Block([LetDecl("y", NumberLit(1))]),
                Block([LetDecl("y", NumberLit(2))]),
            ),
        ]
        z = interp.analyze(stmts)
        # x=5, so x<=3 is infeasible, only else branch
        assert z.get_lower_bound("y") == Fraction(2)
        assert z.get_upper_bound("y") == Fraction(2)

    def test_if_both_branches(self):
        interp = ZoneInterpreter()
        # x = 5; if (x <= 10) { y = 1 } else { y = 2 }
        stmts = [
            LetDecl("x", NumberLit(5)),
            IfStmt(
                BinOp("<=", Identifier("x"), NumberLit(10)),
                Block([LetDecl("y", NumberLit(1))]),
                Block([LetDecl("y", NumberLit(2))]),
            ),
        ]
        z = interp.analyze(stmts)
        # x=5 <= 10, so then branch is feasible; x=5 > 10 is infeasible
        assert z.get_lower_bound("y") == Fraction(1)
        assert z.get_upper_bound("y") == Fraction(1)

    def test_simple_while(self):
        interp = ZoneInterpreter()
        # x = 0; while (x < 10) { x = x + 1; }
        stmts = [
            LetDecl("x", NumberLit(0)),
            WhileStmt(
                BinOp("<", Identifier("x"), NumberLit(10)),
                Block([Assign("x", BinOp("+", Identifier("x"), NumberLit(1)))]),
            ),
        ]
        z = interp.analyze(stmts)
        # After loop: x >= 10 (exit condition)
        assert z.get_lower_bound("x") >= Fraction(10)

    def test_countdown_while(self):
        interp = ZoneInterpreter()
        # x = 10; while (x > 0) { x = x - 1; }
        stmts = [
            LetDecl("x", NumberLit(10)),
            WhileStmt(
                BinOp(">", Identifier("x"), NumberLit(0)),
                Block([Assign("x", BinOp("-", Identifier("x"), NumberLit(1)))]),
            ),
        ]
        z = interp.analyze(stmts)
        # After loop: x <= 0
        assert z.get_upper_bound("x") <= Fraction(0)

    def test_two_var_while(self):
        interp = ZoneInterpreter()
        # x = 0; y = 10; while (x < y) { x = x + 1; }
        stmts = [
            LetDecl("x", NumberLit(0)),
            LetDecl("y", NumberLit(10)),
            WhileStmt(
                BinOp("<", Identifier("x"), Identifier("y")),
                Block([Assign("x", BinOp("+", Identifier("x"), NumberLit(1)))]),
            ),
        ]
        z = interp.analyze(stmts)
        # After loop: x >= y (exit: NOT x < y => x >= y)
        assert z.get_diff_bound("y", "x") <= Fraction(0)  # y - x <= 0

    def test_block(self):
        interp = ZoneInterpreter()
        stmts = Block([
            LetDecl("a", NumberLit(1)),
            LetDecl("b", NumberLit(2)),
            LetDecl("c", BinOp("+", Identifier("a"), NumberLit(3))),
        ])
        z = interp.analyze(stmts)
        assert z.get_lower_bound("c") == Fraction(4)
        assert z.get_upper_bound("c") == Fraction(4)

    def test_const_plus_var(self):
        """var := c + y should work."""
        interp = ZoneInterpreter()
        stmts = [
            LetDecl("y", NumberLit(5)),
            LetDecl("x", BinOp("+", NumberLit(3), Identifier("y"))),
        ]
        z = interp.analyze(stmts)
        assert z.get_lower_bound("x") == Fraction(8)
        assert z.get_upper_bound("x") == Fraction(8)

    def test_nested_if(self):
        interp = ZoneInterpreter()
        # x = 5; if (x > 0) { if (x < 3) { y = 1 } else { y = 2 } } else { y = 3 }
        stmts = [
            LetDecl("x", NumberLit(5)),
            IfStmt(
                BinOp(">", Identifier("x"), NumberLit(0)),
                Block([IfStmt(
                    BinOp("<", Identifier("x"), NumberLit(3)),
                    Block([LetDecl("y", NumberLit(1))]),
                    Block([LetDecl("y", NumberLit(2))]),
                )]),
                Block([LetDecl("y", NumberLit(3))]),
            ),
        ]
        z = interp.analyze(stmts)
        # x=5 > 0, so outer then; x=5 not < 3, so inner else: y = 2
        assert z.get_lower_bound("y") == Fraction(2)
        assert z.get_upper_bound("y") == Fraction(2)


# ===========================================================================
# Composition APIs
# ===========================================================================

class TestComposition:
    def test_zone_from_intervals(self):
        z = zone_from_intervals({"x": (0, 10), "y": (5, 15)})
        assert z.get_interval("x") == (Fraction(0), Fraction(10))
        assert z.get_interval("y") == (Fraction(5), Fraction(15))

    def test_zone_from_intervals_none_bounds(self):
        z = zone_from_intervals({"x": (None, 10), "y": (5, None)})
        assert z.get_upper_bound("x") == Fraction(10)
        assert z.get_lower_bound("x") == -INF
        assert z.get_lower_bound("y") == Fraction(5)
        assert z.get_upper_bound("y") == INF

    def test_verify_zone_property_upper(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        assert verify_zone_property(z, "x <= 5")
        assert verify_zone_property(z, "x <= 10")
        assert not verify_zone_property(z, "x <= 3")

    def test_verify_zone_property_lower(self):
        z = Zone.from_constraints([lower_bound("x", 3)])
        assert verify_zone_property(z, "x >= 3")
        assert verify_zone_property(z, "x >= 0")
        assert not verify_zone_property(z, "x >= 5")

    def test_verify_zone_property_diff(self):
        z = Zone.from_constraints([diff_bound("x", "y", 3)])
        assert verify_zone_property(z, "x - y <= 3")
        assert verify_zone_property(z, "x - y <= 5")
        assert not verify_zone_property(z, "x - y <= 2")

    def test_verify_zone_property_eq(self):
        z = Zone.from_constraints(eq_constraint("x", "y"))
        assert verify_zone_property(z, "x - y == 0")

    def test_compare_with_octagon(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        result = compare_with_octagon(z, [])
        assert result["zone_constraints"] > 0
        assert result["octagon_constraints"] == 0


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_assign_same_var(self):
        """x := x should be identity."""
        z = Zone.from_constraints([lower_bound("x", 3), upper_bound("x", 7)])
        z2 = z.assign_var("x", "x")
        assert z2.get_lower_bound("x") == Fraction(3)
        assert z2.get_upper_bound("x") == Fraction(7)

    def test_zero_bound(self):
        z = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 0)])
        assert z.get_lower_bound("x") == Fraction(0)
        assert z.get_upper_bound("x") == Fraction(0)
        assert not z.is_bot()

    def test_negative_bounds(self):
        z = Zone.from_constraints([lower_bound("x", -10), upper_bound("x", -5)])
        assert z.get_lower_bound("x") == Fraction(-10)
        assert z.get_upper_bound("x") == Fraction(-5)

    def test_fractional_bounds(self):
        z = Zone.from_constraints([
            ZoneConstraint("x", None, Fraction(7, 2)),  # x <= 3.5
        ])
        assert z.get_upper_bound("x") == Fraction(7, 2)

    def test_large_system(self):
        """10 variables with chain constraints."""
        cs = []
        for i in range(10):
            cs.append(lower_bound(f"v{i}", i))
            cs.append(upper_bound(f"v{i}", i + 10))
        for i in range(9):
            cs.append(diff_bound(f"v{i}", f"v{i+1}", 2))
        z = Zone.from_constraints(cs)
        assert not z.is_bot()
        assert z.var_count() == 10

    def test_forget_unknown_var(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        z2 = z.forget("unknown")
        assert z2.get_upper_bound("x") == Fraction(5)

    def test_ensure_var_idempotent(self):
        z = Zone.from_constraints([upper_bound("x", 5)])
        z2 = z._ensure_var("x")
        assert z2.get_upper_bound("x") == Fraction(5)
        assert z2._n == z._n

    def test_join_commutative(self):
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 5)])
        z2 = Zone.from_constraints([lower_bound("x", 3), upper_bound("x", 10)])
        j1 = z1.join(z2)
        j2 = z2.join(z1)
        assert j1.get_lower_bound("x") == j2.get_lower_bound("x")
        assert j1.get_upper_bound("x") == j2.get_upper_bound("x")

    def test_meet_commutative(self):
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 10)])
        z2 = Zone.from_constraints([lower_bound("x", 3), upper_bound("x", 7)])
        m1 = z1.meet(z2)
        m2 = z2.meet(z1)
        assert m1.get_lower_bound("x") == m2.get_lower_bound("x")
        assert m1.get_upper_bound("x") == m2.get_upper_bound("x")

    def test_diff_equality_detection(self):
        z = Zone.from_constraints([
            diff_bound("x", "y", 5),  # x - y <= 5
            diff_bound("y", "x", -5), # y - x <= -5, i.e. x - y >= 5
        ])
        # x - y == 5
        eqs = z.extract_equalities()
        found = any(c == Fraction(5) or c == Fraction(-5) for _, _, c in eqs)
        assert found

    def test_zone_constraint_str(self):
        assert str(upper_bound("x", 5)) == "x <= 5"
        assert str(lower_bound("x", 3)) == "x >= 3"
        assert str(diff_bound("x", "y", 2)) == "x - y <= 2"

    def test_repr(self):
        z = Zone.top()
        assert "TOP" in repr(z)
        z = Zone.bot()
        assert "BOT" in repr(z)

    def test_empty_while(self):
        """While loop that never executes."""
        interp = ZoneInterpreter()
        stmts = [
            LetDecl("x", NumberLit(10)),
            WhileStmt(
                BinOp("<", Identifier("x"), NumberLit(5)),
                Block([Assign("x", BinOp("+", Identifier("x"), NumberLit(1)))]),
            ),
        ]
        z = interp.analyze(stmts)
        # Loop doesn't execute, x stays 10
        assert z.get_lower_bound("x") == Fraction(10)
        assert z.get_upper_bound("x") == Fraction(10)


# ===========================================================================
# Difference Constraint Reasoning
# ===========================================================================

class TestDifferenceReasoning:
    def test_shortest_path_semantics(self):
        """Zone domain gives shortest path = tightest difference bound."""
        z = Zone.from_constraints([
            diff_bound("a", "b", 10),
            diff_bound("b", "c", 5),
            diff_bound("a", "c", 20),
        ])
        # Shortest: a-c <= min(20, 10+5) = 15
        assert z.get_diff_bound("a", "c") == Fraction(15)

    def test_negative_weight_cycle_free(self):
        """Consistent constraints form a negative-cycle-free graph."""
        z = Zone.from_constraints([
            diff_bound("x", "y", 3),
            diff_bound("y", "z", 4),
            diff_bound("z", "x", -6),  # z - x <= -6, i.e. x - z >= 6
        ])
        # x-y <= 3, y-z <= 4, z-x <= -6 => x-x <= 3+4+(-6) = 1 (positive, ok)
        assert not z.is_bot()

    def test_negative_weight_cycle_detected(self):
        z = Zone.from_constraints([
            diff_bound("x", "y", 3),
            diff_bound("y", "z", 4),
            diff_bound("z", "x", -8),  # z - x <= -8, i.e. x - z >= 8
        ])
        # x-x <= 3+4+(-8) = -1 < 0 => contradiction
        assert z.is_bot()

    def test_implied_bounds_from_diffs(self):
        """Difference constraints + one variable bound => bound on other."""
        z = Zone.from_constraints([
            upper_bound("y", 10),        # y <= 10
            diff_bound("x", "y", 3),     # x - y <= 3
        ])
        # x - z0 <= (x - y) + (y - z0) = 3 + 10 = 13 ... wait
        # DBM: x-y <= 3 is DBM[y_idx][x_idx] = 3
        # y-z0 <= 10 is DBM[0][y_idx] = 10
        # x-z0 = (x-y) + (y-z0) => DBM[0][x_idx] = min(INF, DBM[0][y_idx] + DBM[y_idx][x_idx])
        # Wait, DBM[i][j] = x_j - x_i <= c
        # diff_bound(x, y, 3): x - y <= 3 => x_j=x, x_i=y => DBM[y_idx][x_idx] = 3
        # upper_bound(y, 10): y <= 10 => y - z0 <= 10 => DBM[0][y_idx] = 10
        # Closure: DBM[0][x_idx] = min(INF, DBM[0][y_idx] + DBM[y_idx][x_idx]) = 10 + 3 = 13
        assert z.get_upper_bound("x") == Fraction(13)

    def test_scheduling_example(self):
        """Scheduling: task A before B, B before C, with durations."""
        z = Zone.from_constraints([
            lower_bound("a", 0),            # A starts at or after 0
            diff_bound("a", "b", -5),       # A - B <= -5, i.e. B >= A + 5
            diff_bound("b", "c", -3),       # B - C <= -3, i.e. C >= B + 3
            upper_bound("c", 20),           # C must finish by 20
        ])
        # B >= A + 5, C >= B + 3 => C >= A + 8
        # A - B <= -5
        assert z.get_diff_bound("a", "b") <= Fraction(-5)
        # A - C <= -8 (transitive)
        assert z.get_diff_bound("a", "c") <= Fraction(-8)
        # A starts >= 0, C <= 20, C >= A + 8, so A <= 12
        assert z.get_upper_bound("a") <= Fraction(12)

    def test_temporal_distance(self):
        """Model temporal distances between events."""
        z = Zone.from_constraints([
            lower_bound("e1", 0), upper_bound("e1", 0),  # e1 at time 0
            diff_bound("e1", "e2", -2),   # e1 - e2 <= -2, i.e. e2 >= e1 + 2
            diff_bound("e2", "e1", 5),    # e2 - e1 <= 5
            diff_bound("e2", "e3", -1),   # e2 - e3 <= -1, i.e. e3 >= e2 + 1
            diff_bound("e3", "e2", 3),    # e3 - e2 <= 3
        ])
        # e2 in [2, 5], e3 in [3, 8]
        assert z.get_lower_bound("e2") == Fraction(2)
        assert z.get_upper_bound("e2") == Fraction(5)
        assert z.get_lower_bound("e3") == Fraction(3)
        assert z.get_upper_bound("e3") == Fraction(8)


# ===========================================================================
# Widening and Narrowing Convergence
# ===========================================================================

class TestConvergence:
    def test_widening_stabilizes(self):
        """Widening should stabilize even with unbounded iteration."""
        z0 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 0)])
        z1 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 1)])
        z2 = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 2)])

        w = z0.widen(z1)
        # Upper grew from 0 to 1 -> widened to INF
        assert w.get_upper_bound("x") == INF

        w2 = w.widen(z2)
        # INF is stable
        assert w2.get_upper_bound("x") == INF

    def test_narrow_recovers_precision(self):
        """Narrowing after widening should recover some precision."""
        z_widened = Zone.from_constraints([lower_bound("x", 0)])  # x >= 0, upper = INF
        z_precise = Zone.from_constraints([lower_bound("x", 0), upper_bound("x", 100)])

        n = z_widened.narrow(z_precise)
        assert n.get_upper_bound("x") == Fraction(100)
        assert n.get_lower_bound("x") == Fraction(0)

    def test_widen_diff_constraint(self):
        """Widening on difference constraints."""
        z1 = Zone.from_constraints([
            diff_bound("x", "y", 3),
            lower_bound("x", 0),
        ])
        z2 = Zone.from_constraints([
            diff_bound("x", "y", 5),  # grew from 3 to 5
            lower_bound("x", 0),
        ])
        w = z1.widen(z2)
        # x - y bound grew => widened to INF
        assert w.get_diff_bound("x", "y") == INF


# ===========================================================================
# Integration with real analysis patterns
# ===========================================================================

class TestAnalysisPatterns:
    def test_swap_detection(self):
        """After swap, difference should reverse."""
        # Before: x=3, y=7 => x-y = -4
        z = Zone.from_constraints(var_eq_const("x", 3) + var_eq_const("y", 7))
        assert z.get_diff_bound("x", "y") == Fraction(-4)
        # Swap via temp: t=x, x=y, y=t
        z = z.assign_var("t", "x")
        z = z.assign_var("x", "y")
        z = z.assign_var("y", "t")
        assert z.get_diff_bound("x", "y") == Fraction(4)
        assert z.get_lower_bound("x") == Fraction(7)
        assert z.get_lower_bound("y") == Fraction(3)

    def test_accumulator_pattern(self):
        """Accumulator: s = 0; for i=0..9: s += 1."""
        interp = ZoneInterpreter()
        stmts = [
            LetDecl("s", NumberLit(0)),
            LetDecl("i", NumberLit(0)),
            WhileStmt(
                BinOp("<", Identifier("i"), NumberLit(10)),
                Block([
                    Assign("s", BinOp("+", Identifier("s"), NumberLit(1))),
                    Assign("i", BinOp("+", Identifier("i"), NumberLit(1))),
                ]),
            ),
        ]
        z = interp.analyze(stmts)
        # After loop: i >= 10
        assert z.get_lower_bound("i") >= Fraction(10)
        # s - i should be preserved (both increment by 1 each iteration)
        # Initially s=0, i=0 => s-i=0. Each iteration: s'=s+1, i'=i+1 => s'-i'=s-i.
        # Zone should capture: s - i == 0 (or close)
        diff = z.get_diff_bound("s", "i")
        assert diff <= Fraction(0)  # s - i <= 0

    def test_bounded_buffer(self):
        """Buffer with read <= write, 0 <= read, write <= SIZE."""
        SIZE = 100
        z = Zone.from_constraints([
            lower_bound("read", 0),
            lower_bound("write", 0),
            upper_bound("read", SIZE),
            upper_bound("write", SIZE),
            diff_bound("read", "write", 0),  # read <= write
        ])
        # write - read >= 0 (items available)
        assert z.get_diff_bound("read", "write") == Fraction(0)
        # After writing: write := write + 1 (if write < SIZE)
        z2 = z.guard(ZoneConstraint("write", None, Fraction(SIZE - 1)))  # write <= 99
        z2 = z2.increment("write", 1)
        assert z2.get_diff_bound("read", "write") <= Fraction(-1)  # read < write


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
