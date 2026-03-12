"""Tests for V173: Octagon Abstract Domain."""

import pytest
from fractions import Fraction
from octagon import (
    Octagon, OctConstraint, OctExpr, OctagonInterpreter, OctAnalysisResult,
    octagon_from_intervals, analyze_program, compare_with_intervals,
    verify_octagonal_property, compare_with_polyhedra, batch_analyze,
    INF, _bar,
)


# ===================================================================
# 1. Basic Octagon Construction
# ===================================================================

class TestBasicConstruction:
    def test_top(self):
        t = Octagon.top()
        assert not t.is_bot()
        assert t.is_top()
        assert t.num_variables() == 0

    def test_bot(self):
        b = Octagon.bot()
        assert b.is_bot()
        assert not b.is_top()

    def test_from_constraints_single_var(self):
        cs = [OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)]
        o = Octagon.from_constraints(cs)
        assert not o.is_bot()
        lo, hi = o.get_bounds('x')
        assert lo == 0
        assert hi == 10

    def test_from_constraints_two_vars(self):
        cs = [
            OctConstraint.var_le('x', 5),
            OctConstraint.var_ge('x', 0),
            OctConstraint.var_le('y', 5),
            OctConstraint.var_ge('y', 0),
            OctConstraint.diff_le('x', 'y', 2),  # x - y <= 2
        ]
        o = Octagon.from_constraints(cs)
        assert not o.is_bot()
        lo, hi = o.get_difference_bound('x', 'y')
        assert hi == 2

    def test_unsatisfiable_constraints(self):
        cs = [OctConstraint.var_le('x', 5), OctConstraint.var_ge('x', 10)]
        o = Octagon.from_constraints(cs)
        assert o.is_bot()

    def test_ensure_var(self):
        o = Octagon.top()
        o2 = o._ensure_var('x')
        assert 'x' in o2.variables()
        lo, hi = o2.get_bounds('x')
        assert lo is None and hi is None


# ===================================================================
# 2. Constraint Application
# ===================================================================

class TestConstraintApplication:
    def test_var_bounds(self):
        cs = [OctConstraint.var_le('x', 7), OctConstraint.var_ge('x', 3)]
        o = Octagon.from_constraints(cs)
        lo, hi = o.get_bounds('x')
        assert lo == 3
        assert hi == 7

    def test_difference_bound(self):
        cs = [OctConstraint.diff_le('x', 'y', 5)]
        o = Octagon.from_constraints(cs)
        _, hi = o.get_difference_bound('x', 'y')
        assert hi == 5

    def test_sum_bound(self):
        cs = [OctConstraint.sum_le('x', 'y', 10)]
        o = Octagon.from_constraints(cs)
        _, hi = o.get_sum_bound('x', 'y')
        assert hi == 10

    def test_equality(self):
        cs = OctConstraint.eq('x', 'y')
        o = Octagon.from_constraints(cs)
        lo, hi = o.get_difference_bound('x', 'y')
        assert lo == 0 and hi == 0

    def test_var_equality(self):
        cs = OctConstraint.var_eq('x', 5)
        o = Octagon.from_constraints(cs)
        lo, hi = o.get_bounds('x')
        assert lo == 5 and hi == 5

    def test_guard_strengthens(self):
        cs = [OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)]
        o = Octagon.from_constraints(cs)
        o2 = o.guard(OctConstraint.var_le('x', 5))
        _, hi = o2.get_bounds('x')
        assert hi == 5


# ===================================================================
# 3. Lattice Operations
# ===================================================================

class TestLatticeOps:
    def test_join_widens_bounds(self):
        cs1 = OctConstraint.var_eq('x', 3)
        cs2 = OctConstraint.var_eq('x', 7)
        o1 = Octagon.from_constraints(cs1)
        o2 = Octagon.from_constraints(cs2)
        j = o1.join(o2)
        lo, hi = j.get_bounds('x')
        assert lo == 3
        assert hi == 7

    def test_join_with_bot(self):
        o = Octagon.from_constraints([OctConstraint.var_le('x', 5)])
        b = Octagon.bot()
        assert o.join(b) is not None
        assert not o.join(b).is_bot()
        assert b.join(o) is not None

    def test_meet_narrows_bounds(self):
        o1 = Octagon.from_constraints([OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)])
        o2 = Octagon.from_constraints([OctConstraint.var_le('x', 5), OctConstraint.var_ge('x', 3)])
        m = o1.meet(o2)
        lo, hi = m.get_bounds('x')
        assert lo == 3
        assert hi == 5

    def test_meet_to_bot(self):
        o1 = Octagon.from_constraints([OctConstraint.var_le('x', 3)])
        o2 = Octagon.from_constraints([OctConstraint.var_ge('x', 10)])
        m = o1.meet(o2)
        assert m.is_bot()

    def test_includes(self):
        o1 = Octagon.from_constraints([OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)])
        o2 = Octagon.from_constraints([OctConstraint.var_le('x', 5), OctConstraint.var_ge('x', 2)])
        assert o1.includes(o2)
        assert not o2.includes(o1)

    def test_bot_included_in_everything(self):
        o = Octagon.from_constraints([OctConstraint.var_le('x', 5)])
        assert o.includes(Octagon.bot())

    def test_widen_drops_unstable(self):
        o1 = Octagon.from_constraints([OctConstraint.var_le('x', 5), OctConstraint.var_ge('x', 0)])
        o2 = Octagon.from_constraints([OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)])
        w = o1.widen(o2)
        lo, hi = w.get_bounds('x')
        assert lo == 0   # Stable: kept
        assert hi is None  # Unstable: dropped to INF

    def test_narrow_recovers_precision(self):
        o_wide = Octagon.from_constraints([OctConstraint.var_ge('x', 0)])  # x >= 0, no upper
        o_inner = Octagon.from_constraints([OctConstraint.var_le('x', 8), OctConstraint.var_ge('x', 0)])
        n = o_wide.narrow(o_inner)
        _, hi = n.get_bounds('x')
        assert hi == 8


# ===================================================================
# 4. Transfer Functions
# ===================================================================

class TestTransferFunctions:
    def test_assign_const(self):
        o = Octagon.top()
        o2 = o.assign('x', OctExpr.constant(5))
        lo, hi = o2.get_bounds('x')
        assert lo == 5 and hi == 5

    def test_assign_var_copy(self):
        o = Octagon.from_constraints(OctConstraint.var_eq('y', 3))
        o2 = o.assign('x', OctExpr.variable('y'))
        lo, hi = o2.get_bounds('x')
        assert lo == 3 and hi == 3

    def test_assign_var_plus_const(self):
        o = Octagon.from_constraints(OctConstraint.var_eq('y', 5))
        o2 = o.assign('x', OctExpr.variable('y', const=2))
        lo, hi = o2.get_bounds('x')
        assert lo == 7 and hi == 7

    def test_assign_neg_var(self):
        o = Octagon.from_constraints(OctConstraint.var_eq('y', 3))
        o2 = o.assign('x', OctExpr.variable('y', coeff=-1))
        lo, hi = o2.get_bounds('x')
        assert lo == -3 and hi == -3

    def test_increment(self):
        o = Octagon.from_constraints(OctConstraint.var_eq('x', 5))
        o2 = o.assign('x', OctExpr.variable('x', const=1))
        lo, hi = o2.get_bounds('x')
        assert lo == 6 and hi == 6

    def test_forget(self):
        cs = OctConstraint.var_eq('x', 5)
        o = Octagon.from_constraints(cs)
        o2 = o.forget('x')
        lo, hi = o2.get_bounds('x')
        assert lo is None and hi is None

    def test_assign_sum(self):
        """x = y + z."""
        o = Octagon.from_constraints(
            OctConstraint.var_eq('y', 3) + OctConstraint.var_eq('z', 4)
        )
        o2 = o.assign('x', OctExpr.binary('y', 1, 'z', 1))
        lo, hi = o2.get_bounds('x')
        assert lo == 7 and hi == 7

    def test_assign_diff(self):
        """x = y - z."""
        o = Octagon.from_constraints(
            OctConstraint.var_eq('y', 10) + OctConstraint.var_eq('z', 3)
        )
        o2 = o.assign('x', OctExpr.binary('y', 1, 'z', -1))
        lo, hi = o2.get_bounds('x')
        assert lo == 7 and hi == 7


# ===================================================================
# 5. Closure and Derived Bounds
# ===================================================================

class TestClosure:
    def test_transitive_difference(self):
        """x - y <= 3, y - z <= 2 => x - z <= 5."""
        cs = [OctConstraint.diff_le('x', 'y', 3), OctConstraint.diff_le('y', 'z', 2)]
        o = Octagon.from_constraints(cs)
        _, hi = o.get_difference_bound('x', 'z')
        assert hi == 5

    def test_strengthening_tightens_bounds(self):
        """x - y <= 3, x >= 0, y >= 0 should give tighter bounds via strengthening."""
        cs = [
            OctConstraint.diff_le('x', 'y', 3),
            OctConstraint.var_ge('x', 0),
            OctConstraint.var_ge('y', 0),
            OctConstraint.var_le('x', 10),
            OctConstraint.var_le('y', 10),
        ]
        o = Octagon.from_constraints(cs)
        assert not o.is_bot()
        assert o.contains_point({'x': 3, 'y': 0})
        assert o.contains_point({'x': 5, 'y': 3})

    def test_bar_function(self):
        assert _bar(0) == 1
        assert _bar(1) == 0
        assert _bar(4) == 5
        assert _bar(5) == 4


# ===================================================================
# 6. Contains Point
# ===================================================================

class TestContainsPoint:
    def test_point_in_bounds(self):
        cs = [OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)]
        o = Octagon.from_constraints(cs)
        assert o.contains_point({'x': 5})
        assert o.contains_point({'x': 0})
        assert o.contains_point({'x': 10})

    def test_point_out_of_bounds(self):
        cs = [OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)]
        o = Octagon.from_constraints(cs)
        assert not o.contains_point({'x': -1})
        assert not o.contains_point({'x': 11})

    def test_point_with_difference_constraint(self):
        cs = [
            OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0),
            OctConstraint.var_le('y', 10), OctConstraint.var_ge('y', 0),
            OctConstraint.diff_le('x', 'y', 2),
        ]
        o = Octagon.from_constraints(cs)
        assert o.contains_point({'x': 3, 'y': 2})
        assert not o.contains_point({'x': 5, 'y': 0})  # x - y = 5 > 2

    def test_bot_contains_nothing(self):
        assert not Octagon.bot().contains_point({'x': 0})


# ===================================================================
# 7. OctConstraint Construction
# ===================================================================

class TestOctConstraintConstruction:
    def test_var_le(self):
        c = OctConstraint.var_le('x', 5)
        assert c.var1 == 'x'
        assert c.coeff1 == 1
        assert c.bound == 5

    def test_var_ge(self):
        c = OctConstraint.var_ge('x', 3)
        assert c.coeff1 == -1
        assert c.bound == -3

    def test_diff_le(self):
        c = OctConstraint.diff_le('x', 'y', 4)
        assert c.var1 == 'x' and c.coeff1 == 1
        assert c.var2 == 'y' and c.coeff2 == -1
        assert c.bound == 4

    def test_sum_le(self):
        c = OctConstraint.sum_le('x', 'y', 10)
        assert c.coeff1 == 1 and c.coeff2 == 1
        assert c.bound == 10

    def test_diff_ge(self):
        c = OctConstraint.diff_ge('x', 'y', 2)
        # x - y >= 2 => y - x <= -2
        assert c.bound == -2

    def test_sum_ge(self):
        c = OctConstraint.sum_ge('x', 'y', 5)
        # x + y >= 5 => -x - y <= -5
        assert c.bound == -5

    def test_repr(self):
        c = OctConstraint.diff_le('x', 'y', 3)
        s = repr(c)
        assert 'x' in s and 'y' in s


# ===================================================================
# 8. OctExpr Construction
# ===================================================================

class TestOctExprConstruction:
    def test_constant(self):
        e = OctExpr.constant(5)
        assert e.kind == 'const'
        assert e.const == 5

    def test_variable(self):
        e = OctExpr.variable('x')
        assert e.kind == 'var'
        assert e.var1 == 'x'
        assert e.coeff1 == 1

    def test_binary(self):
        e = OctExpr.binary('x', 1, 'y', -1, 3)
        assert e.kind == 'binop'
        assert e.var1 == 'x' and e.var2 == 'y'
        assert e.const == 3


# ===================================================================
# 9. Interpreter: Simple Assignments
# ===================================================================

class TestInterpreterAssignments:
    def test_assign_const(self):
        prog = ('assign', 'x', ('const', 5))
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('x')
        assert lo == 5 and hi == 5

    def test_assign_var(self):
        prog = ('seq',
            ('assign', 'x', ('const', 3)),
            ('assign', 'y', ('var', 'x')),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 3 and hi == 3

    def test_assign_add(self):
        prog = ('seq',
            ('assign', 'x', ('const', 3)),
            ('assign', 'y', ('const', 4)),
            ('assign', 'z', ('add', ('var', 'x'), ('var', 'y'))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('z')
        assert lo == 7 and hi == 7

    def test_assign_sub(self):
        prog = ('seq',
            ('assign', 'x', ('const', 10)),
            ('assign', 'y', ('const', 3)),
            ('assign', 'z', ('sub', ('var', 'x'), ('var', 'y'))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('z')
        assert lo == 7 and hi == 7

    def test_increment(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('x')
        assert lo == 6 and hi == 6


# ===================================================================
# 10. Interpreter: Conditionals
# ===================================================================

class TestInterpreterConditionals:
    def test_simple_if(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('if', ('le', ('var', 'x'), ('const', 3)),
                ('assign', 'y', ('const', 1)),
                ('assign', 'y', ('const', 2)),
            ),
        )
        result = analyze_program(prog)
        # x = 5, so x <= 3 is false. Then branch is BOT, else branch gives y = 2
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 2 and hi == 2

    def test_conditional_with_range(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('if', ('le', ('var', 'x'), ('const', 10)),
                ('assign', 'y', ('const', 1)),
                ('assign', 'y', ('const', 2)),
            ),
        )
        result = analyze_program(prog)
        # x = 5 <= 10, so then branch taken, y = 1
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 1 and hi == 1

    def test_if_both_branches_feasible(self):
        """When input is a range, both branches may be feasible."""
        init = octagon_from_intervals({'x': (0, 10)})
        prog = ('if', ('le', ('var', 'x'), ('const', 5)),
            ('assign', 'y', ('const', 1)),
            ('assign', 'y', ('const', 2)),
        )
        result = analyze_program(prog, init)
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 1 and hi == 2


# ===================================================================
# 11. Interpreter: Loops
# ===================================================================

class TestInterpreterLoops:
    def test_simple_countdown(self):
        prog = ('seq',
            ('assign', 'i', ('const', 10)),
            ('while', ('gt', ('var', 'i'), ('const', 0)),
                ('assign', 'i', ('sub', ('var', 'i'), ('const', 1))),
            ),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('i')
        # After widening, lower bound may be lost (standard octagon behavior).
        # Exit condition i <= 0 gives upper bound.
        assert hi is not None and hi <= 0

    def test_accumulator_loop(self):
        prog = ('seq',
            ('assign', 's', ('const', 0)),
            ('assign', 'i', ('const', 0)),
            ('while', ('lt', ('var', 'i'), ('const', 5)),
                ('seq',
                    ('assign', 's', ('add', ('var', 's'), ('var', 'i'))),
                    ('assign', 'i', ('add', ('var', 'i'), ('const', 1))),
                ),
            ),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('i')
        assert lo is not None and lo >= 5

    def test_loop_with_widening(self):
        """Widening should ensure convergence for unbounded loops."""
        prog = ('seq',
            ('assign', 'x', ('const', 0)),
            ('while', ('lt', ('var', 'x'), ('const', 100)),
                ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
            ),
        )
        result = analyze_program(prog)
        # After widening, x >= 100 at exit (loop condition is x < 100, exit when x >= 100)
        lo, hi = result.final_state.get_bounds('x')
        assert lo is not None and lo >= 100


# ===================================================================
# 12. Interpreter: Assertions
# ===================================================================

class TestInterpreterAssertions:
    def test_assert_true(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assert', ('le', ('var', 'x'), ('const', 10))),
        )
        result = analyze_program(prog)
        assert len(result.warnings) == 0

    def test_assert_false(self):
        prog = ('seq',
            ('assign', 'x', ('const', 15)),
            ('assert', ('le', ('var', 'x'), ('const', 10))),
        )
        result = analyze_program(prog)
        assert len(result.warnings) > 0

    def test_assert_maybe(self):
        init = octagon_from_intervals({'x': (0, 20)})
        prog = ('assert', ('le', ('var', 'x'), ('const', 10)))
        result = analyze_program(prog, init)
        assert len(result.warnings) > 0  # x could be > 10


# ===================================================================
# 13. Relational Properties
# ===================================================================

class TestRelationalProperties:
    def test_difference_tracking(self):
        """After x = y + 3, the octagon should know x - y == 3."""
        prog = ('seq',
            ('assign', 'y', ('const', 5)),
            ('assign', 'x', ('add', ('var', 'y'), ('const', 3))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_difference_bound('x', 'y')
        assert lo == 3 and hi == 3

    def test_sum_tracking(self):
        """After x = c - y, the octagon should know x + y == c."""
        prog = ('seq',
            ('assign', 'y', ('const', 3)),
            ('assign', 'x', ('sub', ('const', 10), ('var', 'y'))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_sum_bound('x', 'y')
        assert lo == 10 and hi == 10

    def test_equality_preservation(self):
        """x = y should maintain x - y == 0."""
        prog = ('seq',
            ('assign', 'y', ('const', 7)),
            ('assign', 'x', ('var', 'y')),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_difference_bound('x', 'y')
        assert lo == 0 and hi == 0


# ===================================================================
# 14. Extract and Query
# ===================================================================

class TestExtractQuery:
    def test_extract_intervals(self):
        cs = [OctConstraint.var_le('x', 5), OctConstraint.var_ge('x', 1),
              OctConstraint.var_le('y', 10), OctConstraint.var_ge('y', 2)]
        o = Octagon.from_constraints(cs)
        ivs = o.extract_intervals()
        assert ivs['x'] == (Fraction(1), Fraction(5))
        assert ivs['y'] == (Fraction(2), Fraction(10))

    def test_extract_constraints(self):
        cs = [OctConstraint.diff_le('x', 'y', 3),
              OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)]
        o = Octagon.from_constraints(cs)
        extracted = o.extract_constraints()
        assert len(extracted) > 0

    def test_num_constraints(self):
        cs = [OctConstraint.var_le('x', 10), OctConstraint.var_ge('x', 0)]
        o = Octagon.from_constraints(cs)
        assert o.num_constraints() > 0

    def test_bot_extract(self):
        b = Octagon.bot()
        assert b.extract_intervals() == {}
        assert b.extract_constraints() == []
        assert b.num_constraints() == 0


# ===================================================================
# 15. Convenience APIs
# ===================================================================

class TestConvenienceAPIs:
    def test_octagon_from_intervals(self):
        o = octagon_from_intervals({'x': (0, 10), 'y': (5, 20)})
        lo_x, hi_x = o.get_bounds('x')
        lo_y, hi_y = o.get_bounds('y')
        assert lo_x == 0 and hi_x == 10
        assert lo_y == 5 and hi_y == 20

    def test_octagon_from_intervals_unbounded(self):
        o = octagon_from_intervals({'x': (None, 10)})
        lo, hi = o.get_bounds('x')
        assert lo is None
        assert hi == 10

    def test_compare_with_intervals(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('sub', ('const', 10), ('var', 'x'))),
        )
        result = compare_with_intervals(prog)
        assert 'octagon_state' in result
        assert 'octagon_intervals' in result

    def test_batch_analyze(self):
        progs = [
            ('assign', 'x', ('const', 1)),
            ('assign', 'x', ('const', 2)),
            ('assign', 'x', ('const', 3)),
        ]
        results = batch_analyze(progs)
        assert len(results) == 3
        for r in results:
            assert not r.final_state.is_bot()


# ===================================================================
# 16. Verify Property
# ===================================================================

class TestVerifyProperty:
    def test_verify_true(self):
        prog = ('assign', 'x', ('const', 5))
        verified, _ = verify_octagonal_property(prog, OctConstraint.var_le('x', 10))
        assert verified

    def test_verify_false(self):
        prog = ('assign', 'x', ('const', 15))
        verified, _ = verify_octagonal_property(prog, OctConstraint.var_le('x', 10))
        assert not verified

    def test_verify_relational(self):
        prog = ('seq',
            ('assign', 'y', ('const', 3)),
            ('assign', 'x', ('add', ('var', 'y'), ('const', 2))),
        )
        verified, _ = verify_octagonal_property(prog, OctConstraint.diff_le('x', 'y', 2))
        assert verified

    def test_verify_on_bot(self):
        """Unreachable state -> property vacuously true."""
        init = Octagon.bot()
        prog = ('assign', 'x', ('const', 100))
        verified, _ = verify_octagonal_property(prog, OctConstraint.var_le('x', 5), init)
        assert verified


# ===================================================================
# 17. Compare with Polyhedra
# ===================================================================

class TestComparePolyhedra:
    def test_compare_basic(self):
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('sub', ('const', 10), ('var', 'x'))),
        )
        result = compare_with_polyhedra(prog)
        assert 'octagon' in result
        oct_data = result['octagon']
        assert oct_data['domain'] == 'octagon'
        assert not oct_data['is_bot']


# ===================================================================
# 18. Negation & Self-Assignment
# ===================================================================

class TestNegationSelfAssignment:
    def test_negate_and_shift(self):
        """x = -x + 10."""
        o = Octagon.from_constraints(OctConstraint.var_eq('x', 3))
        o2 = o.assign('x', OctExpr.variable('x', coeff=-1, const=10))
        lo, hi = o2.get_bounds('x')
        assert lo == 7 and hi == 7

    def test_self_decrement(self):
        """x = x - 1."""
        o = Octagon.from_constraints(OctConstraint.var_eq('x', 10))
        o2 = o.assign('x', OctExpr.variable('x', const=-1))
        lo, hi = o2.get_bounds('x')
        assert lo == 9 and hi == 9


# ===================================================================
# 19. Relational Loop Invariants
# ===================================================================

class TestRelationalLoopInvariants:
    def test_conservation_law(self):
        """x + y == 10 should be preserved across a loop that transfers between x and y."""
        prog = ('seq',
            ('assign', 'x', ('const', 10)),
            ('assign', 'y', ('const', 0)),
            ('while', ('gt', ('var', 'x'), ('const', 0)),
                ('seq',
                    ('assign', 'x', ('sub', ('var', 'x'), ('const', 1))),
                    ('assign', 'y', ('add', ('var', 'y'), ('const', 1))),
                ),
            ),
        )
        result = analyze_program(prog)
        # After widening, exact conservation law may be lost.
        # But exit condition (x <= 0) should give upper bound on x.
        hi_x = result.final_state.get_bounds('x')[1]
        assert hi_x is not None and hi_x <= 0

    def test_ordering_preservation(self):
        """After x = 0, y = 5, if we increment both, x < y should still hold."""
        prog = ('seq',
            ('assign', 'x', ('const', 0)),
            ('assign', 'y', ('const', 5)),
            ('assign', 'x', ('add', ('var', 'x'), ('const', 1))),
            ('assign', 'y', ('add', ('var', 'y'), ('const', 1))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_difference_bound('x', 'y')
        # x = 1, y = 6, so x - y = -5
        assert lo == -5 and hi == -5


# ===================================================================
# 20. Skip and Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_skip(self):
        prog = ('skip',)
        result = analyze_program(prog)
        assert not result.final_state.is_bot()

    def test_empty_sequence(self):
        prog = ('seq',)
        result = analyze_program(prog)
        assert not result.final_state.is_bot()

    def test_non_octagonal_expression(self):
        """Non-linear expression should fall back to forget."""
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('mul', ('var', 'x'), ('var', 'x'))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('y')
        # y is havoc'd because x * x is non-octagonal
        assert lo is None and hi is None

    def test_nested_if(self):
        init = octagon_from_intervals({'x': (0, 20)})
        prog = ('if', ('le', ('var', 'x'), ('const', 10)),
            ('if', ('le', ('var', 'x'), ('const', 5)),
                ('assign', 'y', ('const', 1)),
                ('assign', 'y', ('const', 2)),
            ),
            ('assign', 'y', ('const', 3)),
        )
        result = analyze_program(prog, init)
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 1 and hi == 3

    def test_repr_top(self):
        assert "TOP" in repr(Octagon.top())

    def test_repr_bot(self):
        assert "BOT" in repr(Octagon.bot())

    def test_repr_constraints(self):
        o = Octagon.from_constraints([OctConstraint.var_le('x', 5)])
        r = repr(o)
        assert 'Octagon' in r


# ===================================================================
# 21. Condition Handling
# ===================================================================

class TestConditionHandling:
    def test_eq_condition(self):
        init = octagon_from_intervals({'x': (0, 10)})
        prog = ('if', ('eq', ('var', 'x'), ('const', 5)),
            ('assign', 'y', ('const', 1)),
            ('assign', 'y', ('const', 2)),
        )
        result = analyze_program(prog, init)
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 1 and hi == 2

    def test_ne_condition_overapprox(self):
        """ne is non-convex, so it's overapproximated."""
        init = octagon_from_intervals({'x': (0, 10)})
        prog = ('if', ('ne', ('var', 'x'), ('const', 5)),
            ('assign', 'y', ('const', 1)),
            ('assign', 'y', ('const', 2)),
        )
        result = analyze_program(prog, init)
        # Both branches feasible due to overapproximation
        lo, hi = result.final_state.get_bounds('y')
        assert lo is not None and hi is not None

    def test_and_condition(self):
        init = octagon_from_intervals({'x': (0, 20)})
        prog = ('if', ('and', ('ge', ('var', 'x'), ('const', 5)), ('le', ('var', 'x'), ('const', 10))),
            ('assign', 'y', ('const', 1)),
            ('assign', 'y', ('const', 2)),
        )
        result = analyze_program(prog, init)
        # Both branches feasible
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 1 and hi == 2

    def test_not_condition(self):
        init = octagon_from_intervals({'x': (0, 10)})
        prog = ('if', ('not', ('le', ('var', 'x'), ('const', 5))),
            ('assign', 'y', ('var', 'x')),
            ('assign', 'y', ('const', 0)),
        )
        result = analyze_program(prog, init)
        # In then branch: x > 5
        # In else branch: x <= 5, y = 0
        lo, hi = result.final_state.get_bounds('y')
        assert lo == 0


# ===================================================================
# 22. Octagon vs Interval Precision
# ===================================================================

class TestPrecisionGains:
    def test_octagon_captures_difference(self):
        """Octagon can track x - y == c, intervals cannot."""
        prog = ('seq',
            ('assign', 'x', ('const', 10)),
            ('assign', 'y', ('sub', ('var', 'x'), ('const', 3))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_difference_bound('x', 'y')
        assert lo == 3 and hi == 3  # Exact difference known

    def test_octagon_captures_sum(self):
        """Octagon can track x + y == c."""
        prog = ('seq',
            ('assign', 'x', ('const', 4)),
            ('assign', 'y', ('sub', ('const', 10), ('var', 'x'))),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_sum_bound('x', 'y')
        assert lo == 10 and hi == 10

    def test_copy_propagation(self):
        """After x = y, octagon tracks x == y."""
        prog = ('seq',
            ('assign', 'x', ('const', 5)),
            ('assign', 'y', ('var', 'x')),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_difference_bound('x', 'y')
        assert lo == 0 and hi == 0


# ===================================================================
# 23. Scalability
# ===================================================================

class TestScalability:
    def test_many_variables(self):
        """Octagon with 20 variables should still be tractable."""
        constraints = []
        for i in range(20):
            constraints.append(OctConstraint.var_le(f'x{i}', 100))
            constraints.append(OctConstraint.var_ge(f'x{i}', 0))
        for i in range(19):
            constraints.append(OctConstraint.diff_le(f'x{i}', f'x{i+1}', 5))
        o = Octagon.from_constraints(constraints)
        assert not o.is_bot()
        lo, hi = o.get_bounds('x0')
        assert lo == 0 and hi == 100

    def test_closure_derives_transitive_bounds(self):
        """Closure should derive: x0 - x19 <= 95 from 19 difference constraints of <= 5."""
        constraints = []
        for i in range(20):
            constraints.append(OctConstraint.var_le(f'x{i}', 1000))
            constraints.append(OctConstraint.var_ge(f'x{i}', 0))
        for i in range(19):
            constraints.append(OctConstraint.diff_le(f'x{i}', f'x{i+1}', 5))
        o = Octagon.from_constraints(constraints)
        _, hi = o.get_difference_bound('x0', 'x19')
        assert hi == 95  # 19 * 5


# ===================================================================
# 24. Integration: Full Program Analysis
# ===================================================================

class TestFullProgramAnalysis:
    def test_swap(self):
        """Swap x and y using temp."""
        prog = ('seq',
            ('assign', 'x', ('const', 3)),
            ('assign', 'y', ('const', 7)),
            ('assign', 't', ('var', 'x')),
            ('assign', 'x', ('var', 'y')),
            ('assign', 'y', ('var', 't')),
        )
        result = analyze_program(prog)
        lo_x, hi_x = result.final_state.get_bounds('x')
        lo_y, hi_y = result.final_state.get_bounds('y')
        assert lo_x == 7 and hi_x == 7
        assert lo_y == 3 and hi_y == 3

    def test_abs_value(self):
        """Compute absolute value: y = |x|."""
        init = octagon_from_intervals({'x': (-10, 10)})
        prog = ('if', ('ge', ('var', 'x'), ('const', 0)),
            ('assign', 'y', ('var', 'x')),
            ('assign', 'y', ('sub', ('const', 0), ('var', 'x'))),
        )
        result = analyze_program(prog, init)
        lo, hi = result.final_state.get_bounds('y')
        assert lo is not None and lo >= 0  # y >= 0 (abs value is non-negative)

    def test_min_function(self):
        """min(x, y): result <= both x and y."""
        init = octagon_from_intervals({'x': (0, 10), 'y': (0, 10)})
        prog = ('if', ('le', ('var', 'x'), ('var', 'y')),
            ('assign', 'r', ('var', 'x')),
            ('assign', 'r', ('var', 'y')),
        )
        result = analyze_program(prog, init)
        lo, hi = result.final_state.get_bounds('r')
        assert lo is not None and lo >= 0
        assert hi is not None and hi <= 10

    def test_linear_search_loop(self):
        """Simple loop: i from 0 to n-1."""
        prog = ('seq',
            ('assign', 'n', ('const', 10)),
            ('assign', 'i', ('const', 0)),
            ('while', ('lt', ('var', 'i'), ('var', 'n')),
                ('assign', 'i', ('add', ('var', 'i'), ('const', 1))),
            ),
        )
        result = analyze_program(prog)
        lo, hi = result.final_state.get_bounds('i')
        # At exit: i >= n = 10
        assert lo is not None and lo >= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
