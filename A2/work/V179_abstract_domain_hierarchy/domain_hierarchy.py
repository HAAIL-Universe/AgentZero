"""
V179: Abstract Domain Hierarchy
================================
Unified lattice of abstract domains with automatic promotion.

Hierarchy (least -> most precise):
  Sign < Interval < Zone < Octagon < Polyhedra

Each domain is wrapped in a common protocol (AbstractDomain) that provides:
  - Lattice operations: join, meet, widen, narrow
  - Subsumption: includes, equals, is_bot, is_top
  - Transfer: assign, guard, forget
  - Queries: get_bounds, extract_intervals
  - Promotion: promote_to(target_level) lifts a domain to a more precise one

Composes: C039 (sign/interval), V172 (polyhedra), V173 (octagon), V178 (zone)
"""

from __future__ import annotations
import sys
import os
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Set
from fractions import Fraction

# -- Import existing domains --

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (
    Sign, Interval, AbstractEnv,
    sign_join,
    interval_join, interval_meet, interval_widen,
)


def _sign_expand(s):
    if s == Sign.NEG: return {Sign.NEG}
    if s == Sign.ZERO: return {Sign.ZERO}
    if s == Sign.POS: return {Sign.POS}
    if s == Sign.NON_NEG: return {Sign.ZERO, Sign.POS}
    if s == Sign.NON_POS: return {Sign.ZERO, Sign.NEG}
    if s == Sign.BOT: return set()
    return {Sign.NEG, Sign.ZERO, Sign.POS}


def sign_meet(a: Sign, b: Sign) -> Sign:
    """Greatest lower bound in the sign lattice."""
    if a == Sign.TOP: return b
    if b == Sign.TOP: return a
    if a == Sign.BOT or b == Sign.BOT: return Sign.BOT
    common = _sign_expand(a) & _sign_expand(b)
    if not common: return Sign.BOT
    if common == {Sign.NEG}: return Sign.NEG
    if common == {Sign.ZERO}: return Sign.ZERO
    if common == {Sign.POS}: return Sign.POS
    if common == {Sign.ZERO, Sign.POS}: return Sign.NON_NEG
    if common == {Sign.ZERO, Sign.NEG}: return Sign.NON_POS
    return Sign.TOP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V178_zone_abstract_domain'))
from zone import (
    Zone, ZoneConstraint, ZoneInterpreter,
    upper_bound, lower_bound, diff_bound, eq_constraint, var_eq_const,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))
from octagon import (
    Octagon, OctConstraint, OctExpr, OctagonInterpreter,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V172_polyhedra_abstract_domain'))
from polyhedra import (
    Polyhedron, LinExpr, Constraint, ConstraintKind, PolyhedraInterpreter,
)


# ============================================================
# Domain Level Enum
# ============================================================

class DomainLevel(IntEnum):
    """Precision hierarchy. Higher value = more precise."""
    SIGN = 1
    INTERVAL = 2
    ZONE = 3
    OCTAGON = 4
    POLYHEDRA = 5


# ============================================================
# Unified Constraint Representation
# ============================================================

@dataclass(frozen=True)
class LinearConstraint:
    """A general linear constraint: sum(coeffs[var] * var) <= bound.

    Can represent all constraint types in the hierarchy:
    - Interval: x <= c  (coeffs={x:1}, bound=c)
    - Zone: x - y <= c  (coeffs={x:1, y:-1}, bound=c)
    - Octagon: x + y <= c  (coeffs={x:1, y:1}, bound=c)
    - Polyhedra: arbitrary linear (coeffs={...}, bound=c)
    """
    coeffs: Dict[str, Fraction]  # var -> coefficient
    bound: Fraction

    @staticmethod
    def var_le(var: str, c) -> LinearConstraint:
        """x <= c"""
        return LinearConstraint({var: Fraction(1)}, Fraction(c))

    @staticmethod
    def var_ge(var: str, c) -> LinearConstraint:
        """x >= c  =>  -x <= -c"""
        return LinearConstraint({var: Fraction(-1)}, Fraction(-c))

    @staticmethod
    def diff_le(x: str, y: str, c) -> LinearConstraint:
        """x - y <= c"""
        return LinearConstraint({x: Fraction(1), y: Fraction(-1)}, Fraction(c))

    @staticmethod
    def sum_le(x: str, y: str, c) -> LinearConstraint:
        """x + y <= c"""
        return LinearConstraint({x: Fraction(1), y: Fraction(1)}, Fraction(c))

    @staticmethod
    def eq(x: str, y: str) -> List[LinearConstraint]:
        """x == y  =>  [x - y <= 0, y - x <= 0]"""
        return [
            LinearConstraint.diff_le(x, y, 0),
            LinearConstraint.diff_le(y, x, 0),
        ]

    def classify(self) -> DomainLevel:
        """What's the minimum domain level that can represent this constraint?"""
        non_zero = {v: c for v, c in self.coeffs.items() if c != 0}
        if len(non_zero) == 0:
            return DomainLevel.SIGN  # trivial
        if len(non_zero) == 1:
            v, c = next(iter(non_zero.items()))
            if c == 1 or c == -1:
                return DomainLevel.INTERVAL
            return DomainLevel.POLYHEDRA
        if len(non_zero) == 2:
            coeffs = list(non_zero.values())
            abs_coeffs = sorted(abs(c) for c in coeffs)
            if abs_coeffs == [Fraction(1), Fraction(1)]:
                # Check: are they diff or sum?
                signs = sorted(coeffs)
                if signs == [Fraction(-1), Fraction(1)]:
                    return DomainLevel.ZONE  # x - y <= c
                else:
                    return DomainLevel.OCTAGON  # x + y <= c or -x - y <= c
            return DomainLevel.POLYHEDRA
        return DomainLevel.POLYHEDRA

    def negate(self) -> LinearConstraint:
        """Negate: -(sum coeffs*var) <= -bound  (flips the constraint direction)."""
        return LinearConstraint(
            {v: -c for v, c in self.coeffs.items()},
            -self.bound
        )


# ============================================================
# Abstract Domain Protocol (Base Class)
# ============================================================

class AbstractDomain:
    """Base class for all abstract domain wrappers in the hierarchy."""

    @property
    def level(self) -> DomainLevel:
        raise NotImplementedError

    @property
    def variables(self) -> Set[str]:
        raise NotImplementedError

    # -- Lattice --

    def is_bot(self) -> bool:
        raise NotImplementedError

    def is_top(self) -> bool:
        raise NotImplementedError

    def join(self, other: AbstractDomain) -> AbstractDomain:
        """Least upper bound. Promotes to the more precise domain if needed."""
        a, b = _align_levels(self, other)
        return a._join_same(b)

    def meet(self, other: AbstractDomain) -> AbstractDomain:
        """Greatest lower bound. Promotes to the more precise domain if needed."""
        a, b = _align_levels(self, other)
        return a._meet_same(b)

    def widen(self, other: AbstractDomain) -> AbstractDomain:
        """Widening. Promotes to the more precise domain if needed."""
        a, b = _align_levels(self, other)
        return a._widen_same(b)

    def narrow(self, other: AbstractDomain) -> AbstractDomain:
        """Narrowing. Promotes to the more precise domain if needed."""
        a, b = _align_levels(self, other)
        return a._narrow_same(b)

    def includes(self, other: AbstractDomain) -> bool:
        """Does self over-approximate other? (self >= other)"""
        a, b = _align_levels(self, other)
        return a._includes_same(b)

    def equals(self, other: AbstractDomain) -> bool:
        a, b = _align_levels(self, other)
        return a._equals_same(b)

    # -- Same-level operations (implemented by subclasses) --

    def _join_same(self, other: AbstractDomain) -> AbstractDomain:
        raise NotImplementedError

    def _meet_same(self, other: AbstractDomain) -> AbstractDomain:
        raise NotImplementedError

    def _widen_same(self, other: AbstractDomain) -> AbstractDomain:
        raise NotImplementedError

    def _narrow_same(self, other: AbstractDomain) -> AbstractDomain:
        raise NotImplementedError

    def _includes_same(self, other: AbstractDomain) -> bool:
        raise NotImplementedError

    def _equals_same(self, other: AbstractDomain) -> bool:
        raise NotImplementedError

    # -- Transfer --

    def assign(self, var: str, expr) -> AbstractDomain:
        """var := expr. expr depends on domain level."""
        raise NotImplementedError

    def guard(self, constraint: LinearConstraint) -> AbstractDomain:
        """Refine with a constraint."""
        raise NotImplementedError

    def forget(self, var: str) -> AbstractDomain:
        """Project away a variable."""
        raise NotImplementedError

    # -- Queries --

    def get_bounds(self, var: str) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        """Get [lo, hi] bounds for a variable. None means unbounded."""
        raise NotImplementedError

    def extract_intervals(self) -> Dict[str, Tuple[Optional[Fraction], Optional[Fraction]]]:
        """Extract interval bounds for all variables."""
        return {v: self.get_bounds(v) for v in self.variables}

    def extract_constraints(self) -> List[LinearConstraint]:
        """Extract all constraints as LinearConstraints."""
        raise NotImplementedError

    # -- Promotion --

    def promote_to(self, target: DomainLevel) -> AbstractDomain:
        """Promote this domain to a higher (or equal) precision level."""
        if target == self.level:
            return self
        if target < self.level:
            raise ValueError(f"Cannot demote from {self.level.name} to {target.name}")
        return _promote(self, target)


# ============================================================
# Sign Domain Wrapper
# ============================================================

class SignDomain(AbstractDomain):
    """Wrapper around C039's Sign domain for individual variables."""

    def __init__(self, signs: Optional[Dict[str, Sign]] = None, bottom: bool = False):
        self._signs = dict(signs) if signs else {}
        self._bottom = bottom

    @property
    def level(self) -> DomainLevel:
        return DomainLevel.SIGN

    @property
    def variables(self) -> Set[str]:
        return set(self._signs.keys())

    def is_bot(self) -> bool:
        if self._bottom:
            return True
        return any(s == Sign.BOT for s in self._signs.values())

    def is_top(self) -> bool:
        if self._bottom:
            return False
        return all(s == Sign.TOP for s in self._signs.values())

    def get_sign(self, var: str) -> Sign:
        return self._signs.get(var, Sign.TOP)

    def set_sign(self, var: str, s: Sign) -> SignDomain:
        new = dict(self._signs)
        new[var] = s
        return SignDomain(new, self._bottom)

    def _join_same(self, other: SignDomain) -> SignDomain:
        if self._bottom:
            return other
        if other._bottom:
            return self
        all_vars = set(self._signs) | set(other._signs)
        result = {}
        for v in all_vars:
            result[v] = sign_join(self.get_sign(v), other.get_sign(v))
        return SignDomain(result)

    def _meet_same(self, other: SignDomain) -> SignDomain:
        if self._bottom or other._bottom:
            return SignDomain(bottom=True)
        all_vars = set(self._signs) | set(other._signs)
        result = {}
        for v in all_vars:
            result[v] = sign_meet(self.get_sign(v), other.get_sign(v))
        return SignDomain(result)

    def _widen_same(self, other: SignDomain) -> SignDomain:
        # Sign domain is finite, widen = join
        return self._join_same(other)

    def _narrow_same(self, other: SignDomain) -> SignDomain:
        return self._meet_same(other)

    def _includes_same(self, other: SignDomain) -> bool:
        if other._bottom:
            return True
        if self._bottom:
            return False
        for v in other._signs:
            s_self = self.get_sign(v)
            s_other = other.get_sign(v)
            # self includes other if join(self, other) == self
            if sign_join(s_self, s_other) != s_self:
                return False
        return True

    def _equals_same(self, other: SignDomain) -> bool:
        return self._includes_same(other) and other._includes_same(self)

    def assign(self, var: str, expr) -> SignDomain:
        """expr is a Sign value or (op, operands...) tuple."""
        if isinstance(expr, Sign):
            return self.set_sign(var, expr)
        if isinstance(expr, (int, float)):
            if expr > 0:
                s = Sign.POS
            elif expr < 0:
                s = Sign.NEG
            else:
                s = Sign.ZERO
            return self.set_sign(var, s)
        if isinstance(expr, str):
            # Variable reference
            return self.set_sign(var, self.get_sign(expr))
        return self.set_sign(var, Sign.TOP)

    def guard(self, constraint: LinearConstraint) -> SignDomain:
        # Sign domain can refine very coarsely
        non_zero = {v: c for v, c in constraint.coeffs.items() if c != 0}
        if len(non_zero) == 1:
            v, c = next(iter(non_zero.items()))
            bound = constraint.bound
            if c == Fraction(1) and bound < 0:
                # x <= negative => x is NEG or ZERO
                current = self.get_sign(v)
                refined = sign_meet(current, Sign.NON_POS)
                return self.set_sign(v, refined)
            if c == Fraction(-1) and bound < 0:
                # -x <= negative => x >= positive => x is POS
                current = self.get_sign(v)
                refined = sign_meet(current, Sign.POS)
                return self.set_sign(v, refined)
        return self

    def forget(self, var: str) -> SignDomain:
        new = dict(self._signs)
        new[var] = Sign.TOP
        return SignDomain(new, self._bottom)

    def get_bounds(self, var: str) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        s = self.get_sign(var)
        lo, hi = None, None
        if s == Sign.BOT:
            return (Fraction(1), Fraction(0))  # empty
        if s == Sign.ZERO:
            return (Fraction(0), Fraction(0))
        if s == Sign.POS:
            lo = Fraction(1)
        if s == Sign.NEG:
            hi = Fraction(-1)
        if s == Sign.NON_NEG:
            lo = Fraction(0)
        if s == Sign.NON_POS:
            hi = Fraction(0)
        return (lo, hi)

    def extract_constraints(self) -> List[LinearConstraint]:
        result = []
        for v, s in self._signs.items():
            if s == Sign.POS:
                result.append(LinearConstraint.var_ge(v, 1))
            elif s == Sign.NEG:
                result.append(LinearConstraint.var_le(v, -1))
            elif s == Sign.ZERO:
                result.append(LinearConstraint.var_le(v, 0))
                result.append(LinearConstraint.var_ge(v, 0))
            elif s == Sign.NON_NEG:
                result.append(LinearConstraint.var_ge(v, 0))
            elif s == Sign.NON_POS:
                result.append(LinearConstraint.var_le(v, 0))
        return result

    def __repr__(self):
        if self._bottom:
            return "SignDomain(BOT)"
        return f"SignDomain({self._signs})"


# ============================================================
# Interval Domain Wrapper
# ============================================================

INF = float('inf')

class IntervalDomain(AbstractDomain):
    """Wrapper around C039's Interval for individual variables."""

    def __init__(self, intervals: Optional[Dict[str, Interval]] = None, bottom: bool = False):
        self._intervals = dict(intervals) if intervals else {}
        self._bottom = bottom

    @property
    def level(self) -> DomainLevel:
        return DomainLevel.INTERVAL

    @property
    def variables(self) -> Set[str]:
        return set(self._intervals.keys())

    def is_bot(self) -> bool:
        if self._bottom:
            return True
        return any(i.lo > i.hi for i in self._intervals.values())

    def is_top(self) -> bool:
        if self._bottom:
            return False
        return all(i.lo == -INF and i.hi == INF for i in self._intervals.values())

    def get_interval(self, var: str) -> Interval:
        return self._intervals.get(var, Interval(-INF, INF))

    def set_interval(self, var: str, itv: Interval) -> IntervalDomain:
        new = dict(self._intervals)
        new[var] = itv
        return IntervalDomain(new, self._bottom)

    def _join_same(self, other: IntervalDomain) -> IntervalDomain:
        if self._bottom:
            return other
        if other._bottom:
            return self
        all_vars = set(self._intervals) | set(other._intervals)
        result = {}
        for v in all_vars:
            result[v] = interval_join(self.get_interval(v), other.get_interval(v))
        return IntervalDomain(result)

    def _meet_same(self, other: IntervalDomain) -> IntervalDomain:
        if self._bottom or other._bottom:
            return IntervalDomain(bottom=True)
        all_vars = set(self._intervals) | set(other._intervals)
        result = {}
        for v in all_vars:
            result[v] = interval_meet(self.get_interval(v), other.get_interval(v))
        return IntervalDomain(result)

    def _widen_same(self, other: IntervalDomain) -> IntervalDomain:
        if self._bottom:
            return other
        if other._bottom:
            return self
        all_vars = set(self._intervals) | set(other._intervals)
        result = {}
        for v in all_vars:
            result[v] = interval_widen(self.get_interval(v), other.get_interval(v))
        return IntervalDomain(result)

    def _narrow_same(self, other: IntervalDomain) -> IntervalDomain:
        if self._bottom:
            return self
        if other._bottom:
            return other
        all_vars = set(self._intervals) | set(other._intervals)
        result = {}
        for v in all_vars:
            a = self.get_interval(v)
            b = other.get_interval(v)
            lo = b.lo if a.lo == -INF else a.lo
            hi = b.hi if a.hi == INF else a.hi
            result[v] = Interval(lo, hi)
        return IntervalDomain(result)

    def _includes_same(self, other: IntervalDomain) -> bool:
        if other._bottom:
            return True
        if self._bottom:
            return False
        for v in other._intervals:
            s = self.get_interval(v)
            o = other.get_interval(v)
            if o.lo < s.lo or o.hi > s.hi:
                return False
        return True

    def _equals_same(self, other: IntervalDomain) -> bool:
        return self._includes_same(other) and other._includes_same(self)

    def assign(self, var: str, expr) -> IntervalDomain:
        if isinstance(expr, Interval):
            return self.set_interval(var, expr)
        if isinstance(expr, (int, float)):
            return self.set_interval(var, Interval(expr, expr))
        if isinstance(expr, str):
            return self.set_interval(var, self.get_interval(expr))
        return self.set_interval(var, Interval(-INF, INF))

    def guard(self, constraint: LinearConstraint) -> IntervalDomain:
        non_zero = {v: c for v, c in constraint.coeffs.items() if c != 0}
        if len(non_zero) == 1:
            v, c = next(iter(non_zero.items()))
            bound = constraint.bound
            current = self.get_interval(v)
            if c > 0:
                # c*v <= bound => v <= bound/c
                new_hi = min(current.hi, float(bound / c))
                return self.set_interval(v, Interval(current.lo, new_hi))
            else:
                # c*v <= bound => v >= bound/c (c is negative, division flips)
                new_lo = max(current.lo, float(bound / c))
                return self.set_interval(v, Interval(new_lo, current.hi))
        return self

    def forget(self, var: str) -> IntervalDomain:
        return self.set_interval(var, Interval(-INF, INF))

    def get_bounds(self, var: str) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        itv = self.get_interval(var)
        lo = None if itv.lo == -INF else Fraction(itv.lo)
        hi = None if itv.hi == INF else Fraction(itv.hi)
        return (lo, hi)

    def extract_constraints(self) -> List[LinearConstraint]:
        result = []
        for v, itv in self._intervals.items():
            if itv.hi != INF:
                result.append(LinearConstraint.var_le(v, itv.hi))
            if itv.lo != -INF:
                result.append(LinearConstraint.var_ge(v, itv.lo))
        return result

    def __repr__(self):
        if self._bottom:
            return "IntervalDomain(BOT)"
        return f"IntervalDomain({self._intervals})"


# ============================================================
# Zone Domain Wrapper
# ============================================================

class ZoneDomain(AbstractDomain):
    """Wrapper around V178's Zone."""

    def __init__(self, zone: Optional[Zone] = None):
        self._zone = zone if zone is not None else Zone.top()

    @property
    def level(self) -> DomainLevel:
        return DomainLevel.ZONE

    @property
    def variables(self) -> Set[str]:
        return set(self._zone.variables())

    def is_bot(self) -> bool:
        return self._zone.is_bot()

    def is_top(self) -> bool:
        return self._zone.is_top()

    def _join_same(self, other: ZoneDomain) -> ZoneDomain:
        return ZoneDomain(self._zone.join(other._zone))

    def _meet_same(self, other: ZoneDomain) -> ZoneDomain:
        return ZoneDomain(self._zone.meet(other._zone))

    def _widen_same(self, other: ZoneDomain) -> ZoneDomain:
        return ZoneDomain(self._zone.widen(other._zone))

    def _narrow_same(self, other: ZoneDomain) -> ZoneDomain:
        return ZoneDomain(self._zone.narrow(other._zone))

    def _includes_same(self, other: ZoneDomain) -> bool:
        return self._zone.includes(other._zone)

    def _equals_same(self, other: ZoneDomain) -> bool:
        return self._zone.equals(other._zone)

    def assign(self, var: str, expr) -> ZoneDomain:
        if isinstance(expr, (int, float)):
            return ZoneDomain(self._zone.assign_const(var, expr))
        if isinstance(expr, str):
            return ZoneDomain(self._zone.assign_var(var, expr))
        if isinstance(expr, tuple) and len(expr) == 3:
            # ('+', var, const) or ('-', var, const)
            op, src, c = expr
            if op == '+':
                return ZoneDomain(self._zone.assign_var_plus_const(var, src, c))
            elif op == '-':
                return ZoneDomain(self._zone.assign_var_plus_const(var, src, -c))
        return self.forget(var)

    def guard(self, constraint: LinearConstraint) -> ZoneDomain:
        zc = _linear_to_zone_constraint(constraint)
        if zc is not None:
            return ZoneDomain(self._zone.guard(zc))
        return self

    def forget(self, var: str) -> ZoneDomain:
        return ZoneDomain(self._zone.forget(var))

    def get_bounds(self, var: str) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        lo, hi = self._zone.get_interval(var)
        ZONE_INF = Fraction(10**9)
        lo_r = None if lo is None or lo <= -ZONE_INF else Fraction(lo)
        hi_r = None if hi is None or hi >= ZONE_INF else Fraction(hi)
        return (lo_r, hi_r)

    def extract_constraints(self) -> List[LinearConstraint]:
        result = []
        for zc in self._zone.extract_constraints():
            result.append(_zone_constraint_to_linear(zc))
        return result

    def __repr__(self):
        if self.is_bot():
            return "ZoneDomain(BOT)"
        return f"ZoneDomain({self._zone.extract_constraints()})"


# ============================================================
# Octagon Domain Wrapper
# ============================================================

class OctagonDomain(AbstractDomain):
    """Wrapper around V173's Octagon."""

    def __init__(self, octagon: Optional[Octagon] = None):
        self._octagon = octagon if octagon is not None else Octagon.top()

    @property
    def level(self) -> DomainLevel:
        return DomainLevel.OCTAGON

    @property
    def variables(self) -> Set[str]:
        return set(self._octagon.variables())

    def is_bot(self) -> bool:
        return self._octagon.is_bot()

    def is_top(self) -> bool:
        return self._octagon.is_top()

    def _join_same(self, other: OctagonDomain) -> OctagonDomain:
        return OctagonDomain(self._octagon.join(other._octagon))

    def _meet_same(self, other: OctagonDomain) -> OctagonDomain:
        return OctagonDomain(self._octagon.meet(other._octagon))

    def _widen_same(self, other: OctagonDomain) -> OctagonDomain:
        return OctagonDomain(self._octagon.widen(other._octagon))

    def _narrow_same(self, other: OctagonDomain) -> OctagonDomain:
        return OctagonDomain(self._octagon.narrow(other._octagon))

    def _includes_same(self, other: OctagonDomain) -> bool:
        return self._octagon.includes(other._octagon)

    def _equals_same(self, other: OctagonDomain) -> bool:
        return self._includes_same(other) and other._includes_same(self)

    def assign(self, var: str, expr) -> OctagonDomain:
        if isinstance(expr, OctExpr):
            return OctagonDomain(self._octagon.assign(var, expr))
        if isinstance(expr, (int, float)):
            return OctagonDomain(self._octagon.assign(var, OctExpr.constant(expr)))
        if isinstance(expr, str):
            return OctagonDomain(self._octagon.assign(var, OctExpr.variable(expr)))
        return self.forget(var)

    def guard(self, constraint: LinearConstraint) -> OctagonDomain:
        oc = _linear_to_oct_constraint(constraint)
        if oc is not None:
            return OctagonDomain(self._octagon.guard(oc))
        return self

    def forget(self, var: str) -> OctagonDomain:
        return OctagonDomain(self._octagon.forget(var))

    def get_bounds(self, var: str) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        lo, hi = self._octagon.get_bounds(var)
        ZONE_INF = 10**18
        lo_r = None if lo is None or lo <= -ZONE_INF else Fraction(lo)
        hi_r = None if hi is None or hi >= ZONE_INF else Fraction(hi)
        return (lo_r, hi_r)

    def extract_constraints(self) -> List[LinearConstraint]:
        result = []
        for oc in self._octagon.extract_constraints():
            result.append(_oct_constraint_to_linear(oc))
        return result

    def __repr__(self):
        if self.is_bot():
            return "OctagonDomain(BOT)"
        return f"OctagonDomain(vars={list(self.variables)})"


# ============================================================
# Polyhedra Domain Wrapper
# ============================================================

class PolyhedraDomain(AbstractDomain):
    """Wrapper around V172's Polyhedron."""

    def __init__(self, poly: Optional[Polyhedron] = None):
        self._poly = poly if poly is not None else Polyhedron.top()

    @property
    def level(self) -> DomainLevel:
        return DomainLevel.POLYHEDRA

    @property
    def variables(self) -> Set[str]:
        all_vars = set()
        for c in self._poly.constraints():
            for v, _ in c.expr.coeffs:
                all_vars.add(v)
        return all_vars

    def is_bot(self) -> bool:
        return self._poly.is_bot()

    def is_top(self) -> bool:
        return self._poly.is_top()

    def _join_same(self, other: PolyhedraDomain) -> PolyhedraDomain:
        return PolyhedraDomain(self._poly.join(other._poly))

    def _meet_same(self, other: PolyhedraDomain) -> PolyhedraDomain:
        return PolyhedraDomain(self._poly.meet(other._poly))

    def _widen_same(self, other: PolyhedraDomain) -> PolyhedraDomain:
        return PolyhedraDomain(self._poly.widen(other._poly))

    def _narrow_same(self, other: PolyhedraDomain) -> PolyhedraDomain:
        return PolyhedraDomain(self._poly.narrow(other._poly))

    def _includes_same(self, other: PolyhedraDomain) -> bool:
        return self._poly.includes(other._poly)

    def _equals_same(self, other: PolyhedraDomain) -> bool:
        return self._poly == other._poly

    def assign(self, var: str, expr) -> PolyhedraDomain:
        if isinstance(expr, LinExpr):
            return PolyhedraDomain(self._poly.assign(var, expr))
        if isinstance(expr, (int, float)):
            return PolyhedraDomain(self._poly.assign(var, LinExpr.const(expr)))
        if isinstance(expr, str):
            return PolyhedraDomain(self._poly.assign(var, LinExpr.var(expr)))
        return self.forget(var)

    def guard(self, constraint: LinearConstraint) -> PolyhedraDomain:
        pc = _linear_to_poly_constraint(constraint)
        return PolyhedraDomain(self._poly.guard(pc))

    def forget(self, var: str) -> PolyhedraDomain:
        return PolyhedraDomain(self._poly.forget([var]))

    def get_bounds(self, var: str) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        lo, hi = self._poly.get_bounds(var)
        return (lo, hi)

    def extract_constraints(self) -> List[LinearConstraint]:
        result = []
        for c in self._poly.constraints():
            coeffs = {v: coeff for v, coeff in c.expr.coeffs}
            bound = -c.expr.constant  # LinExpr: sum + constant <= 0 => sum <= -constant
            result.append(LinearConstraint(coeffs, bound))
        return result

    def __repr__(self):
        if self.is_bot():
            return "PolyhedraDomain(BOT)"
        return f"PolyhedraDomain(vars={list(self.variables)})"


# ============================================================
# Constraint Conversion Helpers
# ============================================================

def _linear_to_zone_constraint(lc: LinearConstraint) -> Optional[ZoneConstraint]:
    """Convert LinearConstraint to ZoneConstraint if possible."""
    non_zero = {v: c for v, c in lc.coeffs.items() if c != 0}
    if len(non_zero) == 0:
        return None
    if len(non_zero) == 1:
        v, c = next(iter(non_zero.items()))
        if c == Fraction(1):
            return upper_bound(v, lc.bound)
        elif c == Fraction(-1):
            return lower_bound(v, -lc.bound)
        return None
    if len(non_zero) == 2:
        items = list(non_zero.items())
        v1, c1 = items[0]
        v2, c2 = items[1]
        if c1 == Fraction(1) and c2 == Fraction(-1):
            return diff_bound(v1, v2, lc.bound)
        if c1 == Fraction(-1) and c2 == Fraction(1):
            return diff_bound(v2, v1, lc.bound)
        return None
    return None


def _linear_to_oct_constraint(lc: LinearConstraint) -> Optional[OctConstraint]:
    """Convert LinearConstraint to OctConstraint if possible."""
    non_zero = {v: c for v, c in lc.coeffs.items() if c != 0}
    if len(non_zero) == 0:
        return None
    if len(non_zero) == 1:
        v, c = next(iter(non_zero.items()))
        if c == Fraction(1):
            return OctConstraint.var_le(v, float(lc.bound))
        elif c == Fraction(-1):
            return OctConstraint.var_ge(v, float(-lc.bound))
        return None
    if len(non_zero) == 2:
        items = list(non_zero.items())
        v1, c1 = items[0]
        v2, c2 = items[1]
        b = float(lc.bound)
        if c1 == Fraction(1) and c2 == Fraction(-1):
            return OctConstraint.diff_le(v1, v2, b)
        if c1 == Fraction(-1) and c2 == Fraction(1):
            return OctConstraint.diff_le(v2, v1, b)
        if c1 == Fraction(1) and c2 == Fraction(1):
            return OctConstraint.sum_le(v1, v2, b)
        if c1 == Fraction(-1) and c2 == Fraction(-1):
            return OctConstraint.sum_ge(v1, v2, -b)
        return None
    return None


def _linear_to_poly_constraint(lc: LinearConstraint) -> Constraint:
    """Convert LinearConstraint to V172 Constraint. Always possible."""
    # LinExpr: sum(coeff * var) + constant, Constraint: expr <= 0
    # lc: sum(coeffs * var) <= bound
    # => sum(coeffs * var) - bound <= 0
    expr = LinExpr.const(-lc.bound)
    for v, c in lc.coeffs.items():
        if c != 0:
            expr = expr + LinExpr.var(v, c)
    return Constraint.le(expr)


def _zone_constraint_to_linear(zc: ZoneConstraint) -> LinearConstraint:
    """Convert ZoneConstraint to LinearConstraint."""
    # ZoneConstraint: var1 - var2 <= bound
    if zc.var2 is None:
        # Upper bound: var1 <= bound
        return LinearConstraint.var_le(zc.var1, zc.bound)
    if zc.var1 is None:
        # Lower bound: -var2 <= bound => var2 >= -bound
        return LinearConstraint.var_ge(zc.var2, -zc.bound)
    return LinearConstraint.diff_le(zc.var1, zc.var2, zc.bound)


def _oct_constraint_to_linear(oc: OctConstraint) -> LinearConstraint:
    """Convert OctConstraint to LinearConstraint."""
    coeffs = {}
    if oc.var1 is not None:
        coeffs[oc.var1] = Fraction(oc.coeff1)
    if oc.var2 is not None:
        key = oc.var2
        if key in coeffs:
            coeffs[key] += Fraction(oc.coeff2)
        else:
            coeffs[key] = Fraction(oc.coeff2)
    return LinearConstraint(coeffs, Fraction(oc.bound))


# ============================================================
# Promotion Logic
# ============================================================

def _promote(dom: AbstractDomain, target: DomainLevel) -> AbstractDomain:
    """Promote a domain to a higher level by extracting constraints and re-encoding."""
    if dom.is_bot():
        return _make_bot(target)

    # Extract intervals/constraints from source
    constraints = dom.extract_constraints()
    variables = dom.variables

    if target == DomainLevel.INTERVAL:
        result = IntervalDomain()
        for v in variables:
            lo, hi = dom.get_bounds(v)
            lo_f = float(lo) if lo is not None else -INF
            hi_f = float(hi) if hi is not None else INF
            result = result.set_interval(v, Interval(lo_f, hi_f))
        return result

    if target == DomainLevel.ZONE:
        zone = Zone.top()
        for lc in constraints:
            zc = _linear_to_zone_constraint(lc)
            if zc is not None:
                zone = zone.guard(zc)
        return ZoneDomain(zone)

    if target == DomainLevel.OCTAGON:
        oct_constraints = []
        for lc in constraints:
            oc = _linear_to_oct_constraint(lc)
            if oc is not None:
                oct_constraints.append(oc)
        if oct_constraints:
            octagon = Octagon.from_constraints(oct_constraints)
        else:
            octagon = Octagon.top()
        return OctagonDomain(octagon)

    if target == DomainLevel.POLYHEDRA:
        poly_constraints = []
        for lc in constraints:
            poly_constraints.append(_linear_to_poly_constraint(lc))
        if poly_constraints:
            poly = Polyhedron.from_constraints(poly_constraints)
        else:
            poly = Polyhedron.top()
        return PolyhedraDomain(poly)

    raise ValueError(f"Unknown target level: {target}")


def _make_bot(level: DomainLevel) -> AbstractDomain:
    """Create a bottom element at the given level."""
    if level == DomainLevel.SIGN:
        return SignDomain(bottom=True)
    if level == DomainLevel.INTERVAL:
        return IntervalDomain(bottom=True)
    if level == DomainLevel.ZONE:
        return ZoneDomain(Zone.bot())
    if level == DomainLevel.OCTAGON:
        return OctagonDomain(Octagon.bot())
    if level == DomainLevel.POLYHEDRA:
        return PolyhedraDomain(Polyhedron.bot())
    raise ValueError(f"Unknown level: {level}")


def _make_top(level: DomainLevel) -> AbstractDomain:
    """Create a top element at the given level."""
    if level == DomainLevel.SIGN:
        return SignDomain()
    if level == DomainLevel.INTERVAL:
        return IntervalDomain()
    if level == DomainLevel.ZONE:
        return ZoneDomain(Zone.top())
    if level == DomainLevel.OCTAGON:
        return OctagonDomain(Octagon.top())
    if level == DomainLevel.POLYHEDRA:
        return PolyhedraDomain(Polyhedron.top())
    raise ValueError(f"Unknown level: {level}")


def _align_levels(a: AbstractDomain, b: AbstractDomain) -> Tuple[AbstractDomain, AbstractDomain]:
    """Promote both domains to the higher of their two levels."""
    if a.level == b.level:
        return (a, b)
    target = max(a.level, b.level)
    if a.level < target:
        a = a.promote_to(target)
    if b.level < target:
        b = b.promote_to(target)
    return (a, b)


# ============================================================
# Hierarchy Analyzer
# ============================================================

class DomainHierarchy:
    """Analyze constraints and variables across multiple domain levels.

    Provides:
    - Automatic domain selection based on constraint types
    - Multi-level analysis (run same program at different precision levels)
    - Precision comparison between levels
    """

    @staticmethod
    def classify_constraints(constraints: List[LinearConstraint]) -> DomainLevel:
        """Find the minimum domain level needed to represent all constraints."""
        max_level = DomainLevel.SIGN
        for c in constraints:
            level = c.classify()
            if level > max_level:
                max_level = level
        return max_level

    @staticmethod
    def create(level: DomainLevel, constraints: Optional[List[LinearConstraint]] = None) -> AbstractDomain:
        """Create a domain at the specified level, optionally initialized with constraints."""
        dom = _make_top(level)
        if constraints:
            for c in constraints:
                dom = dom.guard(c)
        return dom

    @staticmethod
    def auto_create(constraints: List[LinearConstraint]) -> AbstractDomain:
        """Create a domain at the minimum level needed for the given constraints."""
        level = DomainHierarchy.classify_constraints(constraints)
        return DomainHierarchy.create(level, constraints)

    @staticmethod
    def multi_level_analyze(
        constraints: List[LinearConstraint],
        var: str,
        levels: Optional[List[DomainLevel]] = None,
    ) -> Dict[DomainLevel, Tuple[Optional[Fraction], Optional[Fraction]]]:
        """Analyze bounds for a variable at multiple domain levels.

        Returns {level: (lo, hi)} for each level.
        """
        if levels is None:
            levels = list(DomainLevel)
        results = {}
        for level in levels:
            dom = DomainHierarchy.create(level, constraints)
            results[level] = dom.get_bounds(var)
        return results

    @staticmethod
    def precision_comparison(
        constraints: List[LinearConstraint],
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Dict[DomainLevel, Tuple[Optional[Fraction], Optional[Fraction]]]]:
        """Compare precision of different domain levels for all variables.

        Returns {var: {level: (lo, hi)}} showing how each level bounds each variable.
        """
        if variables is None:
            # Collect all variables from constraints
            all_vars = set()
            for c in constraints:
                for v in c.coeffs:
                    if c.coeffs[v] != 0:
                        all_vars.add(v)
            variables = sorted(all_vars)

        result = {}
        for var in variables:
            result[var] = DomainHierarchy.multi_level_analyze(constraints, var)
        return result

    @staticmethod
    def domain_chain(
        constraints: List[LinearConstraint],
    ) -> List[Tuple[DomainLevel, AbstractDomain]]:
        """Create the full domain chain from least to most precise.

        Returns [(level, domain)] for all levels, each initialized with the constraints.
        """
        chain = []
        for level in DomainLevel:
            dom = DomainHierarchy.create(level, constraints)
            chain.append((level, dom))
        return chain

    @staticmethod
    def refinement_gain(
        constraints: List[LinearConstraint],
        var: str,
    ) -> Dict[str, Any]:
        """Measure how much precision each domain level adds for a variable.

        Returns analysis including bounds at each level and the precision gain from
        promoting between adjacent levels.
        """
        bounds = DomainHierarchy.multi_level_analyze(constraints, var)
        levels_list = sorted(bounds.keys())

        gains = []
        for i in range(1, len(levels_list)):
            prev_level = levels_list[i - 1]
            curr_level = levels_list[i]
            prev_lo, prev_hi = bounds[prev_level]
            curr_lo, curr_hi = bounds[curr_level]

            # Compute width reduction
            def width(lo, hi):
                if lo is None or hi is None:
                    return None  # unbounded
                return hi - lo

            prev_w = width(prev_lo, prev_hi)
            curr_w = width(curr_lo, curr_hi)

            if prev_w is not None and curr_w is not None and prev_w > 0:
                reduction = float((prev_w - curr_w) / prev_w)
            elif prev_w is None and curr_w is not None:
                reduction = 1.0  # went from unbounded to bounded
            elif prev_w == curr_w:
                reduction = 0.0
            else:
                reduction = None

            gains.append({
                'from': prev_level.name,
                'to': curr_level.name,
                'prev_bounds': (prev_lo, prev_hi),
                'curr_bounds': (curr_lo, curr_hi),
                'reduction': reduction,
            })

        return {
            'variable': var,
            'bounds': {l.name: b for l, b in bounds.items()},
            'gains': gains,
        }


# ============================================================
# Adaptive Domain Selector
# ============================================================

class AdaptiveDomain:
    """Starts at a low precision and automatically promotes when constraints demand it.

    Usage:
        ad = AdaptiveDomain(start_level=DomainLevel.INTERVAL)
        ad = ad.guard(LinearConstraint.var_le('x', 10))
        ad = ad.guard(LinearConstraint.diff_le('x', 'y', 5))  # auto-promotes to ZONE
    """

    def __init__(self, domain: Optional[AbstractDomain] = None,
                 start_level: DomainLevel = DomainLevel.INTERVAL,
                 max_level: DomainLevel = DomainLevel.POLYHEDRA):
        if domain is None:
            domain = _make_top(start_level)
        self._domain = domain
        self._max_level = max_level

    @property
    def level(self) -> DomainLevel:
        return self._domain.level

    @property
    def domain(self) -> AbstractDomain:
        return self._domain

    def _ensure_level(self, needed: DomainLevel) -> AbstractDomain:
        """Promote domain if needed, capped at max_level."""
        target = min(needed, self._max_level)
        if self._domain.level >= target:
            return self._domain
        return self._domain.promote_to(target)

    def guard(self, constraint: LinearConstraint) -> AdaptiveDomain:
        needed = constraint.classify()
        dom = self._ensure_level(needed)
        dom = dom.guard(constraint)
        return AdaptiveDomain(dom, max_level=self._max_level)

    def assign(self, var: str, expr) -> AdaptiveDomain:
        return AdaptiveDomain(self._domain.assign(var, expr), max_level=self._max_level)

    def forget(self, var: str) -> AdaptiveDomain:
        return AdaptiveDomain(self._domain.forget(var), max_level=self._max_level)

    def join(self, other: AdaptiveDomain) -> AdaptiveDomain:
        result = self._domain.join(other._domain)
        return AdaptiveDomain(result, max_level=self._max_level)

    def meet(self, other: AdaptiveDomain) -> AdaptiveDomain:
        result = self._domain.meet(other._domain)
        return AdaptiveDomain(result, max_level=self._max_level)

    def widen(self, other: AdaptiveDomain) -> AdaptiveDomain:
        result = self._domain.widen(other._domain)
        return AdaptiveDomain(result, max_level=self._max_level)

    def get_bounds(self, var: str) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        return self._domain.get_bounds(var)

    def is_bot(self) -> bool:
        return self._domain.is_bot()

    def is_top(self) -> bool:
        return self._domain.is_top()

    def extract_intervals(self) -> Dict[str, Tuple[Optional[Fraction], Optional[Fraction]]]:
        return self._domain.extract_intervals()

    def __repr__(self):
        return f"AdaptiveDomain(level={self.level.name}, domain={self._domain})"


# ============================================================
# Public API
# ============================================================

def sign_domain(signs: Optional[Dict[str, Sign]] = None) -> SignDomain:
    return SignDomain(signs)

def interval_domain(intervals: Optional[Dict[str, Interval]] = None) -> IntervalDomain:
    return IntervalDomain(intervals)

def zone_domain(zone: Optional[Zone] = None) -> ZoneDomain:
    return ZoneDomain(zone)

def octagon_domain(octagon: Optional[Octagon] = None) -> OctagonDomain:
    return OctagonDomain(octagon)

def polyhedra_domain(poly: Optional[Polyhedron] = None) -> PolyhedraDomain:
    return PolyhedraDomain(poly)

def adaptive_domain(start_level=DomainLevel.INTERVAL,
                    max_level=DomainLevel.POLYHEDRA) -> AdaptiveDomain:
    return AdaptiveDomain(start_level=start_level, max_level=max_level)

def hierarchy() -> DomainHierarchy:
    return DomainHierarchy()

def classify_constraint(constraint: LinearConstraint) -> DomainLevel:
    return constraint.classify()

def multi_level_bounds(constraints, var, levels=None):
    return DomainHierarchy.multi_level_analyze(constraints, var, levels)

def precision_report(constraints, variables=None):
    return DomainHierarchy.precision_comparison(constraints, variables)

def refinement_analysis(constraints, var):
    return DomainHierarchy.refinement_gain(constraints, var)
