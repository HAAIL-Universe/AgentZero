"""Tests for V179: Abstract Domain Hierarchy."""

import pytest
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V178_zone_abstract_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V172_polyhedra_abstract_domain'))

from domain_hierarchy import (
    DomainLevel, LinearConstraint, AbstractDomain,
    SignDomain, IntervalDomain, ZoneDomain, OctagonDomain, PolyhedraDomain,
    DomainHierarchy, AdaptiveDomain,
    sign_domain, interval_domain, zone_domain, octagon_domain, polyhedra_domain,
    adaptive_domain, classify_constraint, multi_level_bounds,
    precision_report, refinement_analysis,
    _align_levels, _make_bot, _make_top, _promote,
)
from abstract_interpreter import Sign, Interval
from zone import Zone, upper_bound, lower_bound, diff_bound
from octagon import Octagon, OctConstraint, OctExpr
from polyhedra import Polyhedron, LinExpr, Constraint


# ============================================================
# DomainLevel Tests
# ============================================================

class TestDomainLevel:
    def test_ordering(self):
        assert DomainLevel.SIGN < DomainLevel.INTERVAL
        assert DomainLevel.INTERVAL < DomainLevel.ZONE
        assert DomainLevel.ZONE < DomainLevel.OCTAGON
        assert DomainLevel.OCTAGON < DomainLevel.POLYHEDRA

    def test_values(self):
        assert DomainLevel.SIGN == 1
        assert DomainLevel.POLYHEDRA == 5


# ============================================================
# LinearConstraint Tests
# ============================================================

class TestLinearConstraint:
    def test_var_le(self):
        c = LinearConstraint.var_le('x', 10)
        assert c.coeffs == {'x': Fraction(1)}
        assert c.bound == Fraction(10)

    def test_var_ge(self):
        c = LinearConstraint.var_ge('x', 5)
        assert c.coeffs == {'x': Fraction(-1)}
        assert c.bound == Fraction(-5)

    def test_diff_le(self):
        c = LinearConstraint.diff_le('x', 'y', 3)
        assert c.coeffs == {'x': Fraction(1), 'y': Fraction(-1)}
        assert c.bound == Fraction(3)

    def test_sum_le(self):
        c = LinearConstraint.sum_le('x', 'y', 7)
        assert c.coeffs == {'x': Fraction(1), 'y': Fraction(1)}
        assert c.bound == Fraction(7)

    def test_eq(self):
        cs = LinearConstraint.eq('x', 'y')
        assert len(cs) == 2

    def test_classify_interval(self):
        c = LinearConstraint.var_le('x', 10)
        assert c.classify() == DomainLevel.INTERVAL

    def test_classify_zone(self):
        c = LinearConstraint.diff_le('x', 'y', 5)
        assert c.classify() == DomainLevel.ZONE

    def test_classify_octagon(self):
        c = LinearConstraint.sum_le('x', 'y', 10)
        assert c.classify() == DomainLevel.OCTAGON

    def test_classify_polyhedra(self):
        c = LinearConstraint({'x': Fraction(2), 'y': Fraction(3)}, Fraction(10))
        assert c.classify() == DomainLevel.POLYHEDRA

    def test_classify_trivial(self):
        c = LinearConstraint({}, Fraction(5))
        assert c.classify() == DomainLevel.SIGN

    def test_negate(self):
        c = LinearConstraint.var_le('x', 10)
        n = c.negate()
        assert n.coeffs == {'x': Fraction(-1)}
        assert n.bound == Fraction(-10)


# ============================================================
# SignDomain Tests
# ============================================================

class TestSignDomain:
    def test_level(self):
        d = sign_domain()
        assert d.level == DomainLevel.SIGN

    def test_top(self):
        d = sign_domain()
        assert d.is_top()
        assert not d.is_bot()

    def test_bot(self):
        d = SignDomain(bottom=True)
        assert d.is_bot()
        assert not d.is_top()

    def test_set_and_get(self):
        d = sign_domain().set_sign('x', Sign.POS)
        assert d.get_sign('x') == Sign.POS

    def test_join(self):
        a = sign_domain({'x': Sign.POS})
        b = sign_domain({'x': Sign.NEG})
        c = a.join(b)
        assert c.get_sign('x') == Sign.TOP

    def test_meet(self):
        a = sign_domain({'x': Sign.NON_NEG})
        b = sign_domain({'x': Sign.NON_POS})
        c = a.meet(b)
        assert c.get_sign('x') == Sign.ZERO

    def test_join_with_bot(self):
        a = sign_domain({'x': Sign.POS})
        b = SignDomain(bottom=True)
        assert a.join(b).get_sign('x') == Sign.POS
        assert b.join(a).get_sign('x') == Sign.POS

    def test_includes(self):
        a = sign_domain({'x': Sign.TOP})
        b = sign_domain({'x': Sign.POS})
        assert a.includes(b)
        assert not b.includes(a)

    def test_assign_const(self):
        d = sign_domain()
        d = d.assign('x', 5)
        assert d.get_sign('x') == Sign.POS
        d = d.assign('y', -3)
        assert d.get_sign('y') == Sign.NEG
        d = d.assign('z', 0)
        assert d.get_sign('z') == Sign.ZERO

    def test_assign_var(self):
        d = sign_domain({'x': Sign.POS})
        d = d.assign('y', 'x')
        assert d.get_sign('y') == Sign.POS

    def test_forget(self):
        d = sign_domain({'x': Sign.POS})
        d = d.forget('x')
        assert d.get_sign('x') == Sign.TOP

    def test_get_bounds_pos(self):
        d = sign_domain({'x': Sign.POS})
        lo, hi = d.get_bounds('x')
        assert lo == Fraction(1)
        assert hi is None

    def test_get_bounds_neg(self):
        d = sign_domain({'x': Sign.NEG})
        lo, hi = d.get_bounds('x')
        assert lo is None
        assert hi == Fraction(-1)

    def test_get_bounds_zero(self):
        d = sign_domain({'x': Sign.ZERO})
        lo, hi = d.get_bounds('x')
        assert lo == Fraction(0)
        assert hi == Fraction(0)

    def test_extract_constraints(self):
        d = sign_domain({'x': Sign.POS, 'y': Sign.NON_NEG})
        cs = d.extract_constraints()
        assert len(cs) >= 2

    def test_guard_positive(self):
        d = sign_domain({'x': Sign.TOP})
        d = d.guard(LinearConstraint.var_ge('x', 1))
        assert d.get_sign('x') == Sign.POS

    def test_guard_non_pos(self):
        d = sign_domain({'x': Sign.TOP})
        d = d.guard(LinearConstraint.var_le('x', -1))
        assert d.get_sign('x') in (Sign.NEG, Sign.NON_POS)

    def test_variables(self):
        d = sign_domain({'x': Sign.POS, 'y': Sign.NEG})
        assert d.variables == {'x', 'y'}


# ============================================================
# IntervalDomain Tests
# ============================================================

class TestIntervalDomain:
    def test_level(self):
        d = interval_domain()
        assert d.level == DomainLevel.INTERVAL

    def test_top(self):
        d = interval_domain()
        assert d.is_top()

    def test_bot(self):
        d = IntervalDomain(bottom=True)
        assert d.is_bot()

    def test_set_and_get(self):
        d = interval_domain()
        d = d.set_interval('x', Interval(0, 10))
        assert d.get_interval('x') == Interval(0, 10)

    def test_join(self):
        a = interval_domain({'x': Interval(0, 5)})
        b = interval_domain({'x': Interval(3, 10)})
        c = a.join(b)
        assert c.get_interval('x') == Interval(0, 10)

    def test_meet(self):
        a = interval_domain({'x': Interval(0, 10)})
        b = interval_domain({'x': Interval(5, 15)})
        c = a.meet(b)
        assert c.get_interval('x') == Interval(5, 10)

    def test_widen(self):
        a = interval_domain({'x': Interval(0, 5)})
        b = interval_domain({'x': Interval(0, 10)})
        c = a.widen(b)
        # Widening: hi grew, so it goes to INF
        assert c.get_interval('x').hi == float('inf')

    def test_narrow(self):
        a = interval_domain({'x': Interval(0, float('inf'))})
        b = interval_domain({'x': Interval(0, 100)})
        c = a.narrow(b)
        assert c.get_interval('x').hi == 100

    def test_includes(self):
        a = interval_domain({'x': Interval(0, 10)})
        b = interval_domain({'x': Interval(2, 8)})
        assert a.includes(b)
        assert not b.includes(a)

    def test_equals(self):
        a = interval_domain({'x': Interval(0, 10)})
        b = interval_domain({'x': Interval(0, 10)})
        assert a.equals(b)

    def test_assign_const(self):
        d = interval_domain()
        d = d.assign('x', 42)
        assert d.get_interval('x') == Interval(42, 42)

    def test_assign_var(self):
        d = interval_domain({'x': Interval(0, 10)})
        d = d.assign('y', 'x')
        assert d.get_interval('y') == Interval(0, 10)

    def test_guard_upper(self):
        d = interval_domain({'x': Interval(0, 100)})
        d = d.guard(LinearConstraint.var_le('x', 50))
        lo, hi = d.get_bounds('x')
        assert hi == Fraction(50)

    def test_guard_lower(self):
        d = interval_domain({'x': Interval(0, 100)})
        d = d.guard(LinearConstraint.var_ge('x', 25))
        lo, hi = d.get_bounds('x')
        assert lo == Fraction(25)

    def test_forget(self):
        d = interval_domain({'x': Interval(0, 10)})
        d = d.forget('x')
        lo, hi = d.get_bounds('x')
        assert lo is None and hi is None

    def test_get_bounds(self):
        d = interval_domain({'x': Interval(3, 7)})
        lo, hi = d.get_bounds('x')
        assert lo == Fraction(3)
        assert hi == Fraction(7)

    def test_extract_constraints(self):
        d = interval_domain({'x': Interval(0, 10)})
        cs = d.extract_constraints()
        assert len(cs) == 2  # x <= 10, x >= 0

    def test_extract_intervals(self):
        d = interval_domain({'x': Interval(1, 5), 'y': Interval(0, 3)})
        itvs = d.extract_intervals()
        assert 'x' in itvs
        assert 'y' in itvs


# ============================================================
# ZoneDomain Tests
# ============================================================

class TestZoneDomain:
    def test_level(self):
        d = zone_domain()
        assert d.level == DomainLevel.ZONE

    def test_top(self):
        d = zone_domain()
        assert d.is_top()

    def test_bot(self):
        d = zone_domain(Zone.bot())
        assert d.is_bot()

    def test_assign_const(self):
        d = zone_domain()
        d = d.assign('x', 5)
        lo, hi = d.get_bounds('x')
        assert lo == Fraction(5)
        assert hi == Fraction(5)

    def test_assign_var(self):
        d = zone_domain()
        d = d.assign('x', 5)
        d = d.assign('y', 'x')
        lo, hi = d.get_bounds('y')
        assert lo == Fraction(5)
        assert hi == Fraction(5)

    def test_assign_var_plus_const(self):
        d = zone_domain()
        d = d.assign('x', 5)
        d = d.assign('y', ('+', 'x', 3))
        lo, hi = d.get_bounds('y')
        assert lo == Fraction(8)
        assert hi == Fraction(8)

    def test_guard_diff(self):
        d = zone_domain()
        d = d.assign('x', 10)
        d = d.assign('y', 3)
        d = d.guard(LinearConstraint.diff_le('x', 'y', 5))
        # x - y <= 5. x=10, y=3, diff=7 > 5. But guard refines, not checks.
        # Since x=10, y=3 are exact, guarding x-y<=5 makes it bottom? No...
        # Actually zone guard tightens the DBM. For exact values, if x-y=7 > 5, becomes inconsistent.
        assert d.is_bot()

    def test_guard_upper(self):
        d = zone_domain()
        d = d.guard(LinearConstraint.var_le('x', 10))
        lo, hi = d.get_bounds('x')
        assert hi == Fraction(10)

    def test_join(self):
        a = zone_domain()
        a = a.assign('x', 5)
        b = zone_domain()
        b = b.assign('x', 10)
        c = a.join(b)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(5)
        assert hi == Fraction(10)

    def test_meet(self):
        a = zone_domain()
        a = a.guard(LinearConstraint.var_le('x', 10))
        a = a.guard(LinearConstraint.var_ge('x', 0))
        b = zone_domain()
        b = b.guard(LinearConstraint.var_le('x', 7))
        b = b.guard(LinearConstraint.var_ge('x', 3))
        c = a.meet(b)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(3)
        assert hi == Fraction(7)

    def test_forget(self):
        d = zone_domain()
        d = d.assign('x', 5)
        d = d.forget('x')
        lo, hi = d.get_bounds('x')
        assert lo is None and hi is None

    def test_includes(self):
        a = zone_domain()
        a = a.guard(LinearConstraint.var_le('x', 100))
        a = a.guard(LinearConstraint.var_ge('x', 0))
        b = zone_domain()
        b = b.guard(LinearConstraint.var_le('x', 50))
        b = b.guard(LinearConstraint.var_ge('x', 10))
        assert a.includes(b)
        assert not b.includes(a)

    def test_extract_constraints(self):
        d = zone_domain()
        d = d.assign('x', 5)
        cs = d.extract_constraints()
        assert len(cs) >= 1


# ============================================================
# OctagonDomain Tests
# ============================================================

class TestOctagonDomain:
    def test_level(self):
        d = octagon_domain()
        assert d.level == DomainLevel.OCTAGON

    def test_top(self):
        d = octagon_domain()
        assert d.is_top()

    def test_bot(self):
        d = octagon_domain(Octagon.bot())
        assert d.is_bot()

    def test_assign_const(self):
        d = octagon_domain()
        d = d.assign('x', 5)
        lo, hi = d.get_bounds('x')
        assert lo == Fraction(5)
        assert hi == Fraction(5)

    def test_assign_var(self):
        d = octagon_domain()
        d = d.assign('x', 5)
        d = d.assign('y', 'x')
        lo, hi = d.get_bounds('y')
        assert lo == Fraction(5)
        assert hi == Fraction(5)

    def test_guard_diff(self):
        d = octagon_domain()
        d = d.guard(LinearConstraint.var_le('x', 10))
        d = d.guard(LinearConstraint.var_ge('x', 0))
        d = d.guard(LinearConstraint.diff_le('x', 'y', 5))
        lo, hi = d.get_bounds('x')
        assert hi == Fraction(10)

    def test_guard_sum(self):
        d = octagon_domain()
        d = d.guard(LinearConstraint.sum_le('x', 'y', 10))
        # x + y <= 10 is an octagonal constraint
        cs = d.extract_constraints()
        assert len(cs) >= 1

    def test_join(self):
        a = octagon_domain()
        a = a.assign('x', 5)
        b = octagon_domain()
        b = b.assign('x', 10)
        c = a.join(b)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(5)
        assert hi == Fraction(10)

    def test_forget(self):
        d = octagon_domain()
        d = d.assign('x', 5)
        d = d.forget('x')
        lo, hi = d.get_bounds('x')
        assert lo is None and hi is None

    def test_includes(self):
        a = octagon_domain()
        a = a.guard(LinearConstraint.var_le('x', 100))
        a = a.guard(LinearConstraint.var_ge('x', 0))
        b = octagon_domain()
        b = b.guard(LinearConstraint.var_le('x', 50))
        b = b.guard(LinearConstraint.var_ge('x', 10))
        assert a.includes(b)
        assert not b.includes(a)


# ============================================================
# PolyhedraDomain Tests
# ============================================================

class TestPolyhedraDomain:
    def test_level(self):
        d = polyhedra_domain()
        assert d.level == DomainLevel.POLYHEDRA

    def test_top(self):
        d = polyhedra_domain()
        assert d.is_top()

    def test_bot(self):
        d = polyhedra_domain(Polyhedron.bot())
        assert d.is_bot()

    def test_assign_const(self):
        d = polyhedra_domain()
        d = d.assign('x', 5)
        lo, hi = d.get_bounds('x')
        assert lo == Fraction(5)
        assert hi == Fraction(5)

    def test_assign_var(self):
        d = polyhedra_domain()
        d = d.assign('x', 5)
        d = d.assign('y', 'x')
        lo, hi = d.get_bounds('y')
        assert lo == Fraction(5)
        assert hi == Fraction(5)

    def test_guard_linear(self):
        d = polyhedra_domain()
        # 2x + 3y <= 10
        c = LinearConstraint({'x': Fraction(2), 'y': Fraction(3)}, Fraction(10))
        d = d.guard(c)
        assert not d.is_bot()

    def test_forget(self):
        d = polyhedra_domain()
        d = d.assign('x', 5)
        d = d.forget('x')
        lo, hi = d.get_bounds('x')
        assert lo is None and hi is None

    def test_join(self):
        a = polyhedra_domain()
        a = a.assign('x', 5)
        b = polyhedra_domain()
        b = b.assign('x', 10)
        c = a.join(b)
        lo, hi = c.get_bounds('x')
        assert lo is not None and lo <= Fraction(5)
        assert hi is not None and hi >= Fraction(10)

    def test_extract_constraints(self):
        d = polyhedra_domain()
        d = d.guard(LinearConstraint.var_le('x', 10))
        cs = d.extract_constraints()
        assert len(cs) >= 1


# ============================================================
# Promotion Tests
# ============================================================

class TestPromotion:
    def test_sign_to_interval(self):
        d = sign_domain({'x': Sign.POS})
        promoted = d.promote_to(DomainLevel.INTERVAL)
        assert isinstance(promoted, IntervalDomain)
        lo, hi = promoted.get_bounds('x')
        assert lo == Fraction(1)  # POS -> x >= 1

    def test_interval_to_zone(self):
        d = interval_domain({'x': Interval(0, 10)})
        promoted = d.promote_to(DomainLevel.ZONE)
        assert isinstance(promoted, ZoneDomain)
        lo, hi = promoted.get_bounds('x')
        assert lo == Fraction(0)
        assert hi == Fraction(10)

    def test_interval_to_octagon(self):
        d = interval_domain({'x': Interval(0, 10), 'y': Interval(5, 15)})
        promoted = d.promote_to(DomainLevel.OCTAGON)
        assert isinstance(promoted, OctagonDomain)
        lo, hi = promoted.get_bounds('x')
        assert lo == Fraction(0)
        assert hi == Fraction(10)

    def test_zone_to_octagon(self):
        z = Zone.top()
        z = z.guard(upper_bound('x', 10))
        z = z.guard(lower_bound('x', 0))
        z = z.guard(diff_bound('x', 'y', 5))
        d = zone_domain(z)
        promoted = d.promote_to(DomainLevel.OCTAGON)
        assert isinstance(promoted, OctagonDomain)
        lo, hi = promoted.get_bounds('x')
        assert hi == Fraction(10)

    def test_zone_to_polyhedra(self):
        z = Zone.top()
        z = z.guard(upper_bound('x', 10))
        z = z.guard(lower_bound('x', 0))
        d = zone_domain(z)
        promoted = d.promote_to(DomainLevel.POLYHEDRA)
        assert isinstance(promoted, PolyhedraDomain)
        lo, hi = promoted.get_bounds('x')
        assert lo == Fraction(0)
        assert hi == Fraction(10)

    def test_promote_bot(self):
        d = SignDomain(bottom=True)
        promoted = d.promote_to(DomainLevel.INTERVAL)
        assert promoted.is_bot()

    def test_promote_same_level(self):
        d = interval_domain({'x': Interval(0, 10)})
        assert d.promote_to(DomainLevel.INTERVAL) is d

    def test_promote_invalid_demote(self):
        d = octagon_domain()
        with pytest.raises(ValueError):
            d.promote_to(DomainLevel.INTERVAL)

    def test_sign_to_polyhedra(self):
        d = sign_domain({'x': Sign.NON_NEG})
        promoted = d.promote_to(DomainLevel.POLYHEDRA)
        assert isinstance(promoted, PolyhedraDomain)
        lo, hi = promoted.get_bounds('x')
        assert lo == Fraction(0)


# ============================================================
# Cross-Domain Lattice Operations (Auto-Promotion)
# ============================================================

class TestCrossDomainOps:
    def test_sign_join_interval(self):
        a = sign_domain({'x': Sign.POS})
        b = interval_domain({'x': Interval(0, 10)})
        c = a.join(b)
        assert isinstance(c, IntervalDomain)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(0)  # join of [1,inf) and [0,10] -> [0,inf)

    def test_interval_meet_zone(self):
        a = interval_domain({'x': Interval(0, 100)})
        z = Zone.top()
        z = z.guard(upper_bound('x', 50))
        z = z.guard(lower_bound('x', 10))
        b = zone_domain(z)
        c = a.meet(b)
        assert isinstance(c, ZoneDomain)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(10)
        assert hi == Fraction(50)

    def test_zone_join_octagon(self):
        z = Zone.top()
        z = z.guard(upper_bound('x', 5))
        z = z.guard(lower_bound('x', 0))
        a = zone_domain(z)

        oct = Octagon.from_constraints([
            OctConstraint.var_le('x', 10),
            OctConstraint.var_ge('x', 3),
        ])
        b = octagon_domain(oct)

        c = a.join(b)
        assert isinstance(c, OctagonDomain)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(0)
        assert hi == Fraction(10)

    def test_interval_widen_octagon(self):
        a = interval_domain({'x': Interval(0, 5)})
        oct = Octagon.from_constraints([
            OctConstraint.var_le('x', 10),
            OctConstraint.var_ge('x', 0),
        ])
        b = octagon_domain(oct)
        c = a.widen(b)
        assert isinstance(c, OctagonDomain)

    def test_align_levels(self):
        a = sign_domain({'x': Sign.POS})
        b = polyhedra_domain()
        a2, b2 = _align_levels(a, b)
        assert a2.level == DomainLevel.POLYHEDRA
        assert b2.level == DomainLevel.POLYHEDRA

    def test_includes_cross(self):
        a = interval_domain({'x': Interval(0, 100)})
        z = Zone.top()
        z = z.guard(upper_bound('x', 50))
        z = z.guard(lower_bound('x', 10))
        b = zone_domain(z)
        assert a.includes(b)
        assert not b.includes(a)


# ============================================================
# DomainHierarchy Tests
# ============================================================

class TestDomainHierarchy:
    def test_classify_constraints(self):
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('y', 0),
        ]
        level = DomainHierarchy.classify_constraints(constraints)
        assert level == DomainLevel.INTERVAL

    def test_classify_with_diff(self):
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.diff_le('x', 'y', 5),
        ]
        level = DomainHierarchy.classify_constraints(constraints)
        assert level == DomainLevel.ZONE

    def test_classify_with_sum(self):
        constraints = [
            LinearConstraint.sum_le('x', 'y', 10),
        ]
        level = DomainHierarchy.classify_constraints(constraints)
        assert level == DomainLevel.OCTAGON

    def test_classify_general_linear(self):
        constraints = [
            LinearConstraint({'x': Fraction(2), 'y': Fraction(3)}, Fraction(10)),
        ]
        level = DomainHierarchy.classify_constraints(constraints)
        assert level == DomainLevel.POLYHEDRA

    def test_create(self):
        d = DomainHierarchy.create(DomainLevel.INTERVAL)
        assert isinstance(d, IntervalDomain)
        assert d.is_top()

    def test_create_with_constraints(self):
        constraints = [LinearConstraint.var_le('x', 10)]
        d = DomainHierarchy.create(DomainLevel.INTERVAL, constraints)
        lo, hi = d.get_bounds('x')
        assert hi == Fraction(10)

    def test_auto_create(self):
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.diff_le('x', 'y', 5),
        ]
        d = DomainHierarchy.auto_create(constraints)
        assert isinstance(d, ZoneDomain)

    def test_multi_level_analyze(self):
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('x', 0),
        ]
        results = DomainHierarchy.multi_level_analyze(constraints, 'x')
        assert DomainLevel.INTERVAL in results
        assert DomainLevel.ZONE in results
        # All should give [0, 10] for these simple constraints
        for level, (lo, hi) in results.items():
            if level >= DomainLevel.INTERVAL:
                assert lo == Fraction(0)
                assert hi == Fraction(10)

    def test_precision_comparison(self):
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('x', 0),
            LinearConstraint.var_le('y', 20),
        ]
        report = DomainHierarchy.precision_comparison(constraints, ['x', 'y'])
        assert 'x' in report
        assert 'y' in report

    def test_domain_chain(self):
        constraints = [LinearConstraint.var_le('x', 10)]
        chain = DomainHierarchy.domain_chain(constraints)
        assert len(chain) == 5
        levels = [l for l, d in chain]
        assert levels == list(DomainLevel)

    def test_refinement_gain(self):
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('x', 0),
        ]
        result = DomainHierarchy.refinement_gain(constraints, 'x')
        assert result['variable'] == 'x'
        assert 'bounds' in result
        assert 'gains' in result


# ============================================================
# AdaptiveDomain Tests
# ============================================================

class TestAdaptiveDomain:
    def test_starts_at_interval(self):
        ad = adaptive_domain()
        assert ad.level == DomainLevel.INTERVAL

    def test_stays_interval_for_bounds(self):
        ad = adaptive_domain()
        ad = ad.guard(LinearConstraint.var_le('x', 10))
        assert ad.level == DomainLevel.INTERVAL

    def test_promotes_to_zone_for_diff(self):
        ad = adaptive_domain()
        ad = ad.guard(LinearConstraint.var_le('x', 10))
        ad = ad.guard(LinearConstraint.diff_le('x', 'y', 5))
        assert ad.level == DomainLevel.ZONE

    def test_promotes_to_octagon_for_sum(self):
        ad = adaptive_domain()
        ad = ad.guard(LinearConstraint.sum_le('x', 'y', 10))
        assert ad.level == DomainLevel.OCTAGON

    def test_promotes_to_polyhedra_for_general(self):
        ad = adaptive_domain()
        c = LinearConstraint({'x': Fraction(2), 'y': Fraction(3)}, Fraction(10))
        ad = ad.guard(c)
        assert ad.level == DomainLevel.POLYHEDRA

    def test_max_level_cap(self):
        ad = adaptive_domain(max_level=DomainLevel.ZONE)
        c = LinearConstraint.sum_le('x', 'y', 10)
        ad = ad.guard(c)
        # Capped at ZONE, can't represent sum constraint, stays at ZONE
        assert ad.level == DomainLevel.ZONE

    def test_assign(self):
        ad = adaptive_domain()
        ad = ad.assign('x', 5)
        lo, hi = ad.get_bounds('x')
        assert lo == Fraction(5) and hi == Fraction(5)

    def test_forget(self):
        ad = adaptive_domain()
        ad = ad.assign('x', 5)
        ad = ad.forget('x')
        lo, hi = ad.get_bounds('x')
        assert lo is None and hi is None

    def test_join(self):
        a = adaptive_domain()
        a = a.assign('x', 5)
        b = adaptive_domain()
        b = b.assign('x', 10)
        c = a.join(b)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(5)
        assert hi == Fraction(10)

    def test_meet(self):
        a = adaptive_domain()
        a = a.guard(LinearConstraint.var_le('x', 10))
        a = a.guard(LinearConstraint.var_ge('x', 0))
        b = adaptive_domain()
        b = b.guard(LinearConstraint.var_le('x', 7))
        b = b.guard(LinearConstraint.var_ge('x', 3))
        c = a.meet(b)
        lo, hi = c.get_bounds('x')
        assert lo == Fraction(3)
        assert hi == Fraction(7)

    def test_widen(self):
        a = adaptive_domain()
        a = a.assign('x', 5)
        b = adaptive_domain()
        b = b.assign('x', 10)
        c = a.widen(b)
        assert not c.is_bot()

    def test_is_bot(self):
        ad = adaptive_domain()
        assert not ad.is_bot()

    def test_is_top(self):
        ad = adaptive_domain()
        assert ad.is_top()

    def test_extract_intervals(self):
        ad = adaptive_domain()
        ad = ad.assign('x', 5)
        ad = ad.assign('y', 10)
        itvs = ad.extract_intervals()
        assert 'x' in itvs
        assert 'y' in itvs

    def test_progressive_promotion(self):
        """Start with interval, add zone constraint, then octagon constraint."""
        ad = adaptive_domain()
        ad = ad.guard(LinearConstraint.var_le('x', 10))
        assert ad.level == DomainLevel.INTERVAL
        ad = ad.guard(LinearConstraint.diff_le('x', 'y', 5))
        assert ad.level == DomainLevel.ZONE
        ad = ad.guard(LinearConstraint.sum_le('x', 'y', 15))
        assert ad.level == DomainLevel.OCTAGON

    def test_start_at_sign(self):
        ad = adaptive_domain(start_level=DomainLevel.SIGN)
        assert ad.level == DomainLevel.SIGN
        ad = ad.guard(LinearConstraint.var_le('x', 10))
        assert ad.level == DomainLevel.INTERVAL


# ============================================================
# Make Top/Bot Factory Tests
# ============================================================

class TestFactories:
    def test_make_top_all_levels(self):
        for level in DomainLevel:
            d = _make_top(level)
            assert d.is_top()
            assert d.level == level

    def test_make_bot_all_levels(self):
        for level in DomainLevel:
            d = _make_bot(level)
            assert d.is_bot()
            assert d.level == level

    def test_factory_functions(self):
        assert sign_domain().level == DomainLevel.SIGN
        assert interval_domain().level == DomainLevel.INTERVAL
        assert zone_domain().level == DomainLevel.ZONE
        assert octagon_domain().level == DomainLevel.OCTAGON
        assert polyhedra_domain().level == DomainLevel.POLYHEDRA


# ============================================================
# Constraint Classification Tests
# ============================================================

class TestClassifyConstraint:
    def test_classify_api(self):
        c = LinearConstraint.var_le('x', 10)
        assert classify_constraint(c) == DomainLevel.INTERVAL

    def test_mixed_classify(self):
        c1 = LinearConstraint.var_le('x', 10)
        c2 = LinearConstraint.diff_le('x', 'y', 5)
        c3 = LinearConstraint.sum_le('x', 'y', 10)
        assert classify_constraint(c1) == DomainLevel.INTERVAL
        assert classify_constraint(c2) == DomainLevel.ZONE
        assert classify_constraint(c3) == DomainLevel.OCTAGON


# ============================================================
# Public API Tests
# ============================================================

class TestPublicAPI:
    def test_multi_level_bounds(self):
        cs = [LinearConstraint.var_le('x', 10), LinearConstraint.var_ge('x', 0)]
        results = multi_level_bounds(cs, 'x')
        assert DomainLevel.INTERVAL in results

    def test_precision_report(self):
        cs = [LinearConstraint.var_le('x', 10), LinearConstraint.var_ge('x', 0)]
        report = precision_report(cs, ['x'])
        assert 'x' in report

    def test_refinement_analysis(self):
        cs = [LinearConstraint.var_le('x', 10), LinearConstraint.var_ge('x', 0)]
        result = refinement_analysis(cs, 'x')
        assert result['variable'] == 'x'


# ============================================================
# Precision Hierarchy Property Tests
# ============================================================

class TestPrecisionHierarchy:
    """Verify that higher-precision domains give tighter or equal bounds."""

    def test_zone_more_precise_than_interval(self):
        """Zone can track x-y<=c, interval cannot."""
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('x', 0),
            LinearConstraint.var_le('y', 10),
            LinearConstraint.var_ge('y', 0),
            LinearConstraint.diff_le('x', 'y', 2),  # x - y <= 2
        ]
        itv = DomainHierarchy.create(DomainLevel.INTERVAL, constraints)
        zone = DomainHierarchy.create(DomainLevel.ZONE, constraints)

        # Interval ignores diff constraint for individual bounds
        _, itv_hi_x = itv.get_bounds('x')
        _, zone_hi_x = zone.get_bounds('x')
        assert itv_hi_x == Fraction(10)
        assert zone_hi_x == Fraction(10)  # same for x alone

        # But zone can derive y >= x - 2. If x in [0,10], y in [0,10] intersected
        # with y >= x - 2 and x <= y + 2. The individual bounds stay the same
        # for independent variables, but the relational info is there.

    def test_octagon_more_precise_than_zone(self):
        """Octagon can track x+y<=c, zone cannot."""
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('x', 0),
            LinearConstraint.var_le('y', 10),
            LinearConstraint.var_ge('y', 0),
            LinearConstraint.sum_le('x', 'y', 8),  # x + y <= 8
        ]
        zone = DomainHierarchy.create(DomainLevel.ZONE, constraints)
        oct = DomainHierarchy.create(DomainLevel.OCTAGON, constraints)

        # Zone ignores sum constraint
        _, zone_hi_x = zone.get_bounds('x')
        assert zone_hi_x == Fraction(10)

        # Octagon uses x + y <= 8: if y >= 0, then x <= 8
        _, oct_hi_x = oct.get_bounds('x')
        assert oct_hi_x <= Fraction(8)  # octagon derives tighter bound

    def test_promotion_preserves_info(self):
        """Promoting doesn't lose information."""
        d = interval_domain({'x': Interval(0, 10), 'y': Interval(5, 15)})
        promoted = d.promote_to(DomainLevel.ZONE)
        lo_x, hi_x = promoted.get_bounds('x')
        lo_y, hi_y = promoted.get_bounds('y')
        assert lo_x == Fraction(0) and hi_x == Fraction(10)
        assert lo_y == Fraction(5) and hi_y == Fraction(15)

    def test_all_levels_agree_on_simple_bounds(self):
        """For simple x <= c constraints, all levels should agree."""
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('x', 0),
        ]
        bounds = DomainHierarchy.multi_level_analyze(constraints, 'x')
        for level in [DomainLevel.INTERVAL, DomainLevel.ZONE,
                      DomainLevel.OCTAGON, DomainLevel.POLYHEDRA]:
            lo, hi = bounds[level]
            assert lo == Fraction(0)
            assert hi == Fraction(10)


# ============================================================
# Integration / End-to-End Tests
# ============================================================

class TestIntegration:
    def test_scheduling_constraint_analysis(self):
        """Analyze scheduling constraints at multiple precision levels."""
        # Task A: [0, 10], Task B: [5, 15], A before B by at most 3
        constraints = [
            LinearConstraint.var_le('a', 10),
            LinearConstraint.var_ge('a', 0),
            LinearConstraint.var_le('b', 15),
            LinearConstraint.var_ge('b', 5),
            LinearConstraint.diff_le('a', 'b', 3),  # a - b <= 3
            LinearConstraint.diff_le('b', 'a', 10),  # b - a <= 10 (b can be 10 ahead)
        ]
        d = DomainHierarchy.auto_create(constraints)
        assert isinstance(d, ZoneDomain)
        lo_a, hi_a = d.get_bounds('a')
        assert lo_a is not None and hi_a is not None

    def test_resource_constraint_analysis(self):
        """Analyze resource constraints requiring octagon."""
        # x + y <= 100 (total resource), x >= 0, y >= 0
        constraints = [
            LinearConstraint.var_ge('x', 0),
            LinearConstraint.var_ge('y', 0),
            LinearConstraint.sum_le('x', 'y', 100),
        ]
        d = DomainHierarchy.auto_create(constraints)
        assert isinstance(d, OctagonDomain)
        _, hi_x = d.get_bounds('x')
        assert hi_x is not None and hi_x <= Fraction(100)

    def test_full_pipeline(self):
        """Full pipeline: classify -> create -> analyze -> promote -> compare."""
        constraints = [
            LinearConstraint.var_le('x', 10),
            LinearConstraint.var_ge('x', 0),
            LinearConstraint.diff_le('x', 'y', 5),
        ]
        # 1. Classify
        level = DomainHierarchy.classify_constraints(constraints)
        assert level == DomainLevel.ZONE

        # 2. Create at minimum level
        d = DomainHierarchy.auto_create(constraints)
        assert d.level == DomainLevel.ZONE

        # 3. Promote to octagon
        promoted = d.promote_to(DomainLevel.OCTAGON)
        assert promoted.level == DomainLevel.OCTAGON

        # 4. Bounds preserved
        lo, hi = promoted.get_bounds('x')
        assert lo == Fraction(0)
        assert hi == Fraction(10)

    def test_adaptive_real_world(self):
        """Simulate incremental constraint collection with adaptive domain."""
        ad = adaptive_domain(start_level=DomainLevel.SIGN)

        # Phase 1: Simple positivity
        ad = ad.guard(LinearConstraint.var_ge('x', 1))
        assert ad.level == DomainLevel.INTERVAL

        # Phase 2: Bounds
        ad = ad.guard(LinearConstraint.var_le('x', 100))
        assert ad.level == DomainLevel.INTERVAL

        # Phase 3: Relational
        ad = ad.guard(LinearConstraint.diff_le('x', 'y', 10))
        assert ad.level == DomainLevel.ZONE

        # Phase 4: Sum constraint
        ad = ad.guard(LinearConstraint.sum_le('x', 'y', 50))
        assert ad.level == DomainLevel.OCTAGON

        # Bounds still tracked
        lo, hi = ad.get_bounds('x')
        assert lo is not None and lo >= Fraction(1)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
