"""
V172: Polyhedra Abstract Domain

A relational abstract domain that tracks linear relationships between variables
using systems of linear inequalities (H-representation).

Examples of properties expressible:
  - x <= y  (ordering)
  - 2*x + 3*y <= 10  (linear combination bounds)
  - x + y == n  (conservation laws)
  - 0 <= i < n  (array bounds)

Key algorithms:
  - Fourier-Motzkin elimination for projection (variable elimination)
  - Convex hull via constraint intersection of dual cones
  - Widening by constraint stability (drop unstable constraints)
  - Transfer functions for assignments and guards

Composes with C039-style domains via interval/sign extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from fractions import Fraction
from enum import Enum
import itertools


# ---------------------------------------------------------------------------
# Linear Constraint Representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinExpr:
    """Linear expression: c0 + c1*x1 + c2*x2 + ...
    Stored as {var_name: coefficient} + constant.
    """
    coeffs: FrozenSet[Tuple[str, Fraction]]  # frozenset of (var, coeff)
    constant: Fraction

    @staticmethod
    def var(name, coeff=1):
        return LinExpr(frozenset([(name, Fraction(coeff))]), Fraction(0))

    @staticmethod
    def const(val):
        return LinExpr(frozenset(), Fraction(val))

    @staticmethod
    def zero():
        return LinExpr(frozenset(), Fraction(0))

    def get_coeffs(self):
        """Return mutable dict of coefficients."""
        return dict(self.coeffs)

    def get_coeff(self, var):
        d = dict(self.coeffs)
        return d.get(var, Fraction(0))

    def variables(self):
        return {v for v, c in self.coeffs if c != 0}

    def __add__(self, other):
        if isinstance(other, (int, float, Fraction)):
            return LinExpr(self.coeffs, self.constant + Fraction(other))
        d = self.get_coeffs()
        for v, c in other.coeffs:
            d[v] = d.get(v, Fraction(0)) + c
        # Remove zeros
        d = {v: c for v, c in d.items() if c != 0}
        return LinExpr(frozenset(d.items()), self.constant + other.constant)

    def __sub__(self, other):
        if isinstance(other, (int, float, Fraction)):
            return LinExpr(self.coeffs, self.constant - Fraction(other))
        return self + other.scale(Fraction(-1))

    def __neg__(self):
        return self.scale(Fraction(-1))

    def scale(self, factor):
        factor = Fraction(factor)
        d = {v: c * factor for v, c in self.coeffs}
        d = {v: c for v, c in d.items() if c != 0}
        return LinExpr(frozenset(d.items()), self.constant * factor)

    def __mul__(self, scalar):
        return self.scale(scalar)

    def __rmul__(self, scalar):
        return self.scale(scalar)

    def substitute(self, var, expr):
        """Replace var with expr (a LinExpr)."""
        coeff = self.get_coeff(var)
        if coeff == 0:
            return self
        d = self.get_coeffs()
        del d[var]
        result = LinExpr(frozenset(d.items()), self.constant)
        return result + expr.scale(coeff)

    def __repr__(self):
        parts = []
        for v, c in sorted(self.coeffs):
            if c == 1:
                parts.append(v)
            elif c == -1:
                parts.append(f"-{v}")
            else:
                parts.append(f"{c}*{v}")
        if self.constant != 0 or not parts:
            parts.append(str(self.constant))
        return " + ".join(parts).replace("+ -", "- ")


class ConstraintKind(Enum):
    LE = "<="   # expr <= 0
    EQ = "=="   # expr == 0


@dataclass(frozen=True)
class Constraint:
    """Linear constraint: expr <= 0 or expr == 0."""
    expr: LinExpr
    kind: ConstraintKind

    @staticmethod
    def le(expr):
        """expr <= 0"""
        return Constraint(expr, ConstraintKind.LE)

    @staticmethod
    def eq(expr):
        """expr == 0"""
        return Constraint(expr, ConstraintKind.EQ)

    @staticmethod
    def var_le(x, y):
        """x <= y  =>  x - y <= 0"""
        return Constraint.le(LinExpr.var(x) - LinExpr.var(y))

    @staticmethod
    def var_le_const(x, c):
        """x <= c  =>  x - c <= 0"""
        return Constraint.le(LinExpr.var(x) - LinExpr.const(c))

    @staticmethod
    def var_ge_const(x, c):
        """x >= c  =>  c - x <= 0  =>  -x + c <= 0"""
        return Constraint.le(LinExpr.var(x, -1) + Fraction(c))

    @staticmethod
    def var_eq_const(x, c):
        """x == c  =>  x - c == 0"""
        return Constraint.eq(LinExpr.var(x) - LinExpr.const(c))

    @staticmethod
    def var_eq(x, y):
        """x == y  =>  x - y == 0"""
        return Constraint.eq(LinExpr.var(x) - LinExpr.var(y))

    def variables(self):
        return self.expr.variables()

    def is_tautology(self):
        """Is this constraint always true (e.g., 0 <= 0, -5 <= 0)?"""
        if self.expr.variables():
            return False
        if self.kind == ConstraintKind.LE:
            return self.expr.constant <= 0
        else:
            return self.expr.constant == 0

    def is_contradiction(self):
        """Is this constraint always false (e.g., 5 <= 0)?"""
        if self.expr.variables():
            return False
        if self.kind == ConstraintKind.LE:
            return self.expr.constant > 0
        else:
            return self.expr.constant != 0

    def substitute(self, var, expr):
        return Constraint(self.expr.substitute(var, expr), self.kind)

    def __repr__(self):
        return f"{self.expr} {self.kind.value} 0"


# ---------------------------------------------------------------------------
# Polyhedron (Convex Polyhedron Abstract Domain Element)
# ---------------------------------------------------------------------------

class Polyhedron:
    """A convex polyhedron represented as a conjunction of linear constraints.

    Special elements:
      - TOP: unconstrained (empty constraint set)
      - BOT: infeasible (contains contradictory constraints)
    """

    def __init__(self, constraints=None, *, _bot=False):
        if _bot:
            self._constraints = []
            self._is_bot = True
            return
        self._is_bot = False
        self._constraints = []
        if constraints:
            for c in constraints:
                if c.is_contradiction():
                    self._is_bot = True
                    self._constraints = []
                    return
                if not c.is_tautology():
                    self._constraints.append(c)

    @staticmethod
    def top():
        return Polyhedron()

    @staticmethod
    def bot():
        return Polyhedron(_bot=True)

    @staticmethod
    def from_constraints(constraints):
        return Polyhedron(constraints)

    def is_bot(self):
        return self._is_bot

    def is_top(self):
        return not self._is_bot and len(self._constraints) == 0

    def constraints(self):
        return list(self._constraints)

    def variables(self):
        vs = set()
        for c in self._constraints:
            vs |= c.variables()
        return vs

    def num_constraints(self):
        return len(self._constraints)

    def add_constraint(self, c):
        """Return new polyhedron with added constraint."""
        if self._is_bot:
            return self
        if c.is_contradiction():
            return Polyhedron.bot()
        if c.is_tautology():
            return Polyhedron(self._constraints[:])
        return Polyhedron(self._constraints + [c])

    def add_constraints(self, cs):
        result = self
        for c in cs:
            result = result.add_constraint(c)
            if result.is_bot():
                return result
        return result

    # -------------------------------------------------------------------
    # Lattice operations
    # -------------------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, Polyhedron):
            return False
        if self._is_bot and other._is_bot:
            return True
        if self._is_bot or other._is_bot:
            return False
        # Semantic equality: each includes the other
        return self.includes(other) and other.includes(self)

    def __hash__(self):
        return hash(("poly", self._is_bot, len(self._constraints)))

    def includes(self, other):
        """Does self >= other in the lattice? (self includes other)
        Every point in other is also in self.
        Equivalently: every constraint in self is implied by other's constraints.
        """
        if other.is_bot():
            return True
        if self.is_bot():
            return False
        if self.is_top():
            return True
        # Check each of self's constraints against other
        for c in self._constraints:
            if not other._implies_constraint(c):
                return False
        return True

    def _implies_constraint(self, c):
        """Does this polyhedron imply constraint c?
        Uses Fourier-Motzkin to check if adding NOT(c) makes the system infeasible.
        For c: expr <= 0, NOT(c) is expr > 0, approximated as expr >= 1 (integer)
        or we check the bound.
        """
        if c.is_tautology():
            return True
        # Get bound on c.expr within this polyhedron
        bound = self._upper_bound_expr(c.expr)
        if bound is None:
            return False  # unbounded
        if c.kind == ConstraintKind.LE:
            return bound <= 0
        else:  # EQ
            lb = self._lower_bound_expr(c.expr)
            if lb is None:
                return False
            return lb >= 0 and bound <= 0

    def _upper_bound_expr(self, expr):
        """Find upper bound of expr within this polyhedron using FM elimination.
        Returns None if unbounded.
        """
        # Project all variables out of the system {constraints} + {-expr + t <= 0}
        # and find the smallest t.
        # Simpler approach: iteratively eliminate variables from the constraints
        # and track the bound on the expression.
        vars_to_elim = expr.variables() | self.variables()
        target_var = "__target__"
        # Rewrite as: maximize target s.t. constraints AND target <= expr
        # i.e., target - expr <= 0
        target_constraint = Constraint.le(
            LinExpr.var(target_var) - expr
        )
        all_constraints = self._constraints + [target_constraint]
        # Eliminate all variables except target
        vars_to_elim = vars_to_elim - {target_var}
        result = _fourier_motzkin_eliminate_vars(all_constraints, vars_to_elim)
        # Find upper bound on target
        best = None
        for c in result:
            coeff = c.expr.get_coeff(target_var)
            if coeff > 0 and c.kind == ConstraintKind.LE:
                # coeff * target + constant <= 0
                # target <= -constant / coeff
                bound = -c.expr.constant / coeff
                if best is None or bound < best:
                    best = bound
            elif coeff > 0 and c.kind == ConstraintKind.EQ:
                bound = -c.expr.constant / coeff
                if best is None or bound < best:
                    best = bound
        return best

    def _lower_bound_expr(self, expr):
        """Find lower bound of expr. Returns None if unbounded below."""
        ub = self._upper_bound_expr(-expr)
        if ub is None:
            return None
        return -ub

    def get_bounds(self, var):
        """Get [lo, hi] bounds for a variable. Returns (lo, hi) with None for unbounded."""
        lo = self._lower_bound_expr(LinExpr.var(var))
        hi = self._upper_bound_expr(LinExpr.var(var))
        return (lo, hi)

    def get_interval(self, var):
        """Get interval bounds as (lo, hi) Fraction pair."""
        return self.get_bounds(var)

    # -------------------------------------------------------------------
    # Meet (intersection)
    # -------------------------------------------------------------------

    def meet(self, other):
        """Greatest lower bound: intersection of polyhedra."""
        if self.is_bot() or other.is_bot():
            return Polyhedron.bot()
        return Polyhedron(self._constraints + other._constraints)

    # -------------------------------------------------------------------
    # Join (convex hull)
    # -------------------------------------------------------------------

    def join(self, other):
        """Least upper bound: convex hull of two polyhedra.
        Uses constraint-based approach: keep constraints implied by both.
        Also discovers relational constraints (sums/differences) between variable pairs.
        """
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        # Check effective satisfiability (detects contradictory constraint sets)
        if not self.is_satisfiable():
            return other
        if not other.is_satisfiable():
            return self
        if self.is_top() or other.is_top():
            return Polyhedron.top()

        # Convex hull: a constraint c is in the join iff
        # both self and other imply c.
        # We consider constraints from both polyhedra.
        result = []
        all_constraints = self._constraints + other._constraints
        for c in all_constraints:
            if self._implies_constraint(c) and other._implies_constraint(c):
                result.append(c)

        # Also try to derive new shared constraints via variable bounds
        all_vars = sorted(self.variables() | other.variables())
        for v in all_vars:
            s_lo, s_hi = self.get_bounds(v)
            o_lo, o_hi = other.get_bounds(v)
            # Join bounds: min of lowers, max of uppers
            if s_lo is not None and o_lo is not None:
                lo = min(s_lo, o_lo)
                c = Constraint.le(LinExpr.var(v, -1) + lo)  # -v + lo <= 0, i.e., v >= lo
                if not any(_constraint_equivalent(c, r) for r in result):
                    result.append(c)
            if s_hi is not None and o_hi is not None:
                hi = max(s_hi, o_hi)
                c = Constraint.le(LinExpr.var(v) - LinExpr.const(hi))  # v - hi <= 0
                if not any(_constraint_equivalent(c, r) for r in result):
                    result.append(c)

        # Discover relational constraints between pairs of variables
        # Try sums (v1+v2) and differences (v1-v2) -- key for conservation laws
        for i, v1 in enumerate(all_vars):
            for v2 in all_vars[i+1:]:
                # Sum: v1 + v2
                sum_expr = LinExpr.var(v1) + LinExpr.var(v2)
                s_sum_lo = self._lower_bound_expr(sum_expr)
                s_sum_hi = self._upper_bound_expr(sum_expr)
                o_sum_lo = other._lower_bound_expr(sum_expr)
                o_sum_hi = other._upper_bound_expr(sum_expr)
                if (s_sum_lo is not None and o_sum_lo is not None and
                    s_sum_hi is not None and o_sum_hi is not None):
                    if s_sum_lo == s_sum_hi == o_sum_lo == o_sum_hi:
                        # Both agree on exact value: v1 + v2 == c
                        c = Constraint.eq(sum_expr - LinExpr.const(s_sum_lo))
                        if not any(_constraint_equivalent(c, r) for r in result):
                            result.append(c)
                    else:
                        lo = min(s_sum_lo, o_sum_lo)
                        hi = max(s_sum_hi, o_sum_hi)
                        c_lo = Constraint.le(-sum_expr + LinExpr.const(lo))
                        c_hi = Constraint.le(sum_expr - LinExpr.const(hi))
                        if not any(_constraint_equivalent(c_lo, r) for r in result):
                            result.append(c_lo)
                        if not any(_constraint_equivalent(c_hi, r) for r in result):
                            result.append(c_hi)

                # Difference: v1 - v2
                diff_expr = LinExpr.var(v1) - LinExpr.var(v2)
                s_diff_lo = self._lower_bound_expr(diff_expr)
                s_diff_hi = self._upper_bound_expr(diff_expr)
                o_diff_lo = other._lower_bound_expr(diff_expr)
                o_diff_hi = other._upper_bound_expr(diff_expr)
                if (s_diff_lo is not None and o_diff_lo is not None and
                    s_diff_hi is not None and o_diff_hi is not None):
                    if s_diff_lo == s_diff_hi == o_diff_lo == o_diff_hi:
                        c = Constraint.eq(diff_expr - LinExpr.const(s_diff_lo))
                        if not any(_constraint_equivalent(c, r) for r in result):
                            result.append(c)
                    else:
                        lo = min(s_diff_lo, o_diff_lo)
                        hi = max(s_diff_hi, o_diff_hi)
                        c_lo = Constraint.le(-diff_expr + LinExpr.const(lo))
                        c_hi = Constraint.le(diff_expr - LinExpr.const(hi))
                        if not any(_constraint_equivalent(c_lo, r) for r in result):
                            result.append(c_lo)
                        if not any(_constraint_equivalent(c_hi, r) for r in result):
                            result.append(c_hi)

        return Polyhedron(result)

    # -------------------------------------------------------------------
    # Widening
    # -------------------------------------------------------------------

    def widen(self, other):
        """Widening: keep only constraints from self that are satisfied by other.
        This ensures convergence in finite steps.
        """
        if self.is_bot():
            return other
        if other.is_bot():
            return self

        # Standard widening: keep constraints from self that other implies
        result = []
        for c in self._constraints:
            if other._implies_constraint(c):
                result.append(c)

        return Polyhedron(result)

    def narrow(self, other):
        """Narrowing: refine self with constraints from other.
        Dual of widening -- improves precision after fixpoint.
        """
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        # Add other's constraints that are consistent
        result = list(self._constraints)
        for c in other._constraints:
            # Add if it doesn't make the system infeasible
            test = Polyhedron(result + [c])
            if not test.is_bot():
                result.append(c)
        return Polyhedron(result)

    # -------------------------------------------------------------------
    # Transfer functions
    # -------------------------------------------------------------------

    def assign(self, var, expr):
        """Abstract transfer for var := expr.
        expr is a LinExpr.
        Eliminates old var and adds var == expr.
        """
        if self.is_bot():
            return self

        # Eliminate var from existing constraints via substitution
        # Replace var with a fresh temporary, then add var == expr
        temp = f"__old_{var}__"
        new_constraints = []
        for c in self._constraints:
            # Replace var with temp in existing constraints
            new_c = c.substitute(var, LinExpr.var(temp))
            new_constraints.append(new_c)

        # Add: var == expr (with temp replacing var in expr if needed)
        new_expr = expr.substitute(var, LinExpr.var(temp))
        eq_constraint = Constraint.eq(LinExpr.var(var) - new_expr)
        new_constraints.append(eq_constraint)

        # Eliminate temp
        result = _fourier_motzkin_eliminate_vars(new_constraints, {temp})
        return Polyhedron(result)

    def assign_nondet(self, var):
        """Havoc / nondeterministic assignment: forget everything about var."""
        if self.is_bot():
            return self
        result = _fourier_motzkin_eliminate_vars(self._constraints, {var})
        return Polyhedron(result)

    def guard(self, constraint):
        """Assume a constraint holds (condition in if/while)."""
        return self.add_constraint(constraint)

    def guard_constraints(self, constraints):
        """Assume multiple constraints."""
        return self.add_constraints(constraints)

    # -------------------------------------------------------------------
    # Projection
    # -------------------------------------------------------------------

    def project(self, vars_to_keep):
        """Project onto the given variables, eliminating all others."""
        if self.is_bot():
            return self
        vars_to_elim = self.variables() - set(vars_to_keep)
        result = _fourier_motzkin_eliminate_vars(self._constraints, vars_to_elim)
        return Polyhedron(result)

    def forget(self, vars_to_forget):
        """Eliminate the given variables."""
        if self.is_bot():
            return self
        result = _fourier_motzkin_eliminate_vars(self._constraints, set(vars_to_forget))
        return Polyhedron(result)

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def is_satisfiable(self):
        """Check if the polyhedron is non-empty (not BOT).
        Uses iterative FM elimination to detect contradictions.
        """
        if self._is_bot:
            return False
        if not self._constraints:
            return True
        # Try to eliminate all variables
        result = _fourier_motzkin_eliminate_vars(self._constraints, self.variables())
        # Check remaining constant constraints
        for c in result:
            if c.is_contradiction():
                return False
        return True

    def contains_point(self, point):
        """Check if a concrete point {var: value} satisfies all constraints."""
        if self.is_bot():
            return False
        for c in self._constraints:
            val = c.expr.constant
            for v, coeff in c.expr.coeffs:
                if v not in point:
                    return False  # unbound variable
                val += coeff * Fraction(point[v])
            if c.kind == ConstraintKind.LE:
                if val > 0:
                    return False
            else:
                if val != 0:
                    return False
        return True

    # -------------------------------------------------------------------
    # Extraction for composition with other domains
    # -------------------------------------------------------------------

    def extract_intervals(self):
        """Extract per-variable interval bounds."""
        result = {}
        for v in self.variables():
            lo, hi = self.get_bounds(v)
            result[v] = (lo, hi)
        return result

    def extract_equalities(self):
        """Extract equalities between variables or variable-constant."""
        result = []
        for c in self._constraints:
            if c.kind == ConstraintKind.EQ:
                result.append(c)
        return result

    def extract_orderings(self):
        """Extract pairwise variable orderings (x <= y, etc.)."""
        result = []
        for c in self._constraints:
            if c.kind == ConstraintKind.LE:
                coeffs = c.expr.get_coeffs()
                if len(coeffs) == 2 and c.expr.constant == 0:
                    vars_list = sorted(coeffs.keys())
                    c0, c1 = coeffs[vars_list[0]], coeffs[vars_list[1]]
                    if c0 == 1 and c1 == -1:
                        result.append((vars_list[0], "<=", vars_list[1]))
                    elif c0 == -1 and c1 == 1:
                        result.append((vars_list[1], "<=", vars_list[0]))
        return result

    def __repr__(self):
        if self._is_bot:
            return "Polyhedron(BOT)"
        if not self._constraints:
            return "Polyhedron(TOP)"
        cs = ", ".join(str(c) for c in self._constraints)
        return f"Polyhedron({cs})"


# ---------------------------------------------------------------------------
# Fourier-Motzkin Elimination
# ---------------------------------------------------------------------------

def _fourier_motzkin_eliminate_vars(constraints, vars_to_elim):
    """Eliminate variables from a set of constraints using Fourier-Motzkin.
    Returns remaining constraints in terms of non-eliminated variables.
    """
    current = list(constraints)
    for var in vars_to_elim:
        current = _fm_eliminate_one(current, var)
    return current


def _fm_eliminate_one(constraints, var):
    """Eliminate one variable from constraints via Fourier-Motzkin."""
    positive = []   # constraints where var has positive coefficient
    negative = []   # constraints where var has negative coefficient
    zero = []       # constraints not involving var
    equalities = [] # equality constraints involving var

    for c in constraints:
        coeff = c.expr.get_coeff(var)
        if coeff == 0:
            zero.append(c)
        elif c.kind == ConstraintKind.EQ:
            equalities.append(c)
        elif coeff > 0:
            positive.append(c)
        else:
            negative.append(c)

    # If there are equalities involving var, use substitution
    if equalities:
        eq = equalities[0]
        coeff = eq.expr.get_coeff(var)
        # var = -(rest) / coeff
        rest = eq.expr.get_coeffs()
        del rest[var]
        # expr_for_var: solve coeff*var + rest + const == 0 => var = -(rest + const)/coeff
        replacement = LinExpr(frozenset(rest.items()), eq.expr.constant).scale(Fraction(-1) / coeff)

        result = list(zero)
        # Substitute into all other constraints
        for c in positive + negative + equalities[1:]:
            new_c = c.substitute(var, replacement)
            if not new_c.is_tautology():
                if new_c.is_contradiction():
                    return [new_c]  # System is infeasible
                result.append(new_c)
        return result

    # No equalities: combine positive and negative pairs
    result = list(zero)
    for p in positive:
        for n in negative:
            # p: a*var + p_rest <= 0  (a > 0)
            # n: b*var + n_rest <= 0  (b < 0)
            # Combine: (-b)*p + a*n eliminates var
            a = p.expr.get_coeff(var)
            b = n.expr.get_coeff(var)
            # Scale: p by (-b), n by a
            combined_expr = p.expr.scale(-b) + n.expr.scale(a)
            # var term should be 0
            new_c = Constraint.le(combined_expr)
            if not new_c.is_tautology():
                if new_c.is_contradiction():
                    return [new_c]
                result.append(new_c)

    return result


def _constraint_equivalent(c1, c2):
    """Check if two constraints are syntactically equivalent (up to scaling)."""
    if c1.kind != c2.kind:
        return False
    # Normalize: make leading coefficient 1
    d1 = c1.expr.get_coeffs()
    d2 = c2.expr.get_coeffs()
    if set(d1.keys()) != set(d2.keys()):
        return False
    if not d1:
        return c1.expr.constant == c2.expr.constant if c1.kind == ConstraintKind.EQ else (
            (c1.expr.constant <= 0) == (c2.expr.constant <= 0)
        )
    first_var = sorted(d1.keys())[0]
    scale = d2[first_var] / d1[first_var] if d1[first_var] != 0 else Fraction(1)
    if scale <= 0 and c1.kind == ConstraintKind.LE:
        return False
    for v in d1:
        if d1[v] * scale != d2.get(v, Fraction(0)):
            return False
    return c1.expr.constant * scale == c2.expr.constant


# ---------------------------------------------------------------------------
# Polyhedra Abstract Interpreter
# ---------------------------------------------------------------------------

class PolyhedraInterpreter:
    """Abstract interpreter using the polyhedra domain.

    Analyzes programs (as simple imperative ASTs) tracking relational
    properties between variables.

    AST format (simplified, compatible with C010 subset):
      - ('assign', var, expr)
      - ('seq', stmt1, stmt2, ...)
      - ('if', cond, then_stmts, else_stmts)
      - ('while', cond, body_stmts)
      - ('assert', cond)

    expr format:
      - ('var', name)
      - ('const', value)
      - ('add', e1, e2)
      - ('sub', e1, e2)
      - ('mul', e1, e2)   # only if one operand is constant
      - ('neg', e)

    cond format:
      - ('le', e1, e2)    # e1 <= e2
      - ('lt', e1, e2)    # e1 < e2
      - ('ge', e1, e2)    # e1 >= e2
      - ('gt', e1, e2)    # e1 > e2
      - ('eq', e1, e2)    # e1 == e2
      - ('ne', e1, e2)    # e1 != e2
      - ('and', c1, c2)
      - ('or', c1, c2)
      - ('not', c)
      - ('true',)
      - ('false',)
    """

    def __init__(self, max_iterations=50, widen_delay=2):
        self.max_iterations = max_iterations
        self.widen_delay = widen_delay
        self.warnings = []

    def analyze(self, program, init=None):
        """Analyze a program starting from init polyhedron (default TOP)."""
        state = init if init is not None else Polyhedron.top()
        self.warnings = []
        result = self._interpret(program, state)
        return AnalysisResult(
            final_state=result,
            warnings=list(self.warnings),
        )

    def _interpret(self, stmt, state):
        if state.is_bot():
            return state

        tag = stmt[0]
        if tag == 'assign':
            return self._interpret_assign(stmt, state)
        elif tag == 'seq':
            for s in stmt[1:]:
                state = self._interpret(s, state)
                if state.is_bot():
                    return state
            return state
        elif tag == 'if':
            return self._interpret_if(stmt, state)
        elif tag == 'while':
            return self._interpret_while(stmt, state)
        elif tag == 'assert':
            return self._interpret_assert(stmt, state)
        elif tag == 'skip':
            return state
        else:
            raise ValueError(f"Unknown statement: {tag}")

    def _interpret_assign(self, stmt, state):
        _, var, expr = stmt
        linexpr = self._expr_to_linexpr(expr)
        if linexpr is not None:
            return state.assign(var, linexpr)
        else:
            # Non-linear expression: havoc the variable
            return state.assign_nondet(var)

    def _interpret_if(self, stmt, state):
        _, cond, then_body, else_body = stmt
        # Guard for then branch
        then_constraints = self._cond_to_constraints(cond)
        else_constraints = self._cond_to_constraints(('not', cond))

        then_state = state
        for c in then_constraints:
            then_state = then_state.guard(c)
        then_result = self._interpret(then_body, then_state)

        else_state = state
        for c in else_constraints:
            else_state = else_state.guard(c)
        else_result = self._interpret(else_body, else_state)

        return then_result.join(else_result)

    def _interpret_while(self, stmt, state):
        _, cond, body = stmt
        guard_cs = self._cond_to_constraints(cond)
        exit_cs = self._cond_to_constraints(('not', cond))

        current = state
        for iteration in range(self.max_iterations):
            # Enter loop body
            body_entry = current
            for c in guard_cs:
                body_entry = body_entry.guard(c)
            body_exit = self._interpret(body, body_entry)

            # Join with loop entry
            next_state = current.join(body_exit)

            # Apply widening after delay
            if iteration >= self.widen_delay:
                next_state = current.widen(next_state)

            # Check convergence
            if next_state.includes(current) and current.includes(next_state):
                break
            current = next_state

        # Apply exit condition
        result = current
        for c in exit_cs:
            result = result.guard(c)
        return result

    def _interpret_assert(self, stmt, state):
        _, cond = stmt
        # Check if assertion can be violated
        neg_constraints = self._cond_to_constraints(('not', cond))
        violation_state = state
        for c in neg_constraints:
            violation_state = violation_state.guard(c)
        if violation_state.is_satisfiable():
            self.warnings.append(f"Assertion may fail: {cond}")
        return state

    def _expr_to_linexpr(self, expr):
        """Convert expression AST to LinExpr, or None if non-linear."""
        tag = expr[0]
        if tag == 'const':
            return LinExpr.const(expr[1])
        elif tag == 'var':
            return LinExpr.var(expr[1])
        elif tag == 'add':
            l = self._expr_to_linexpr(expr[1])
            r = self._expr_to_linexpr(expr[2])
            if l is None or r is None:
                return None
            return l + r
        elif tag == 'sub':
            l = self._expr_to_linexpr(expr[1])
            r = self._expr_to_linexpr(expr[2])
            if l is None or r is None:
                return None
            return l - r
        elif tag == 'mul':
            l = self._expr_to_linexpr(expr[1])
            r = self._expr_to_linexpr(expr[2])
            if l is None or r is None:
                return None
            # At least one must be constant
            if not l.variables():
                return r.scale(l.constant)
            elif not r.variables():
                return l.scale(r.constant)
            else:
                return None  # non-linear
        elif tag == 'neg':
            inner = self._expr_to_linexpr(expr[1])
            if inner is None:
                return None
            return -inner
        else:
            return None

    def _cond_to_constraints(self, cond):
        """Convert condition AST to list of Constraints."""
        tag = cond[0]
        if tag == 'true':
            return []  # no constraint needed
        elif tag == 'false':
            return [Constraint.le(LinExpr.const(1))]  # 1 <= 0 is false
        elif tag == 'le':
            l = self._expr_to_linexpr(cond[1])
            r = self._expr_to_linexpr(cond[2])
            if l is None or r is None:
                return []
            return [Constraint.le(l - r)]  # l - r <= 0
        elif tag == 'lt':
            l = self._expr_to_linexpr(cond[1])
            r = self._expr_to_linexpr(cond[2])
            if l is None or r is None:
                return []
            # l < r => l - r + 1 <= 0 (integer arithmetic)
            return [Constraint.le(l - r + 1)]
        elif tag == 'ge':
            return self._cond_to_constraints(('le', cond[2], cond[1]))
        elif tag == 'gt':
            return self._cond_to_constraints(('lt', cond[2], cond[1]))
        elif tag == 'eq':
            l = self._expr_to_linexpr(cond[1])
            r = self._expr_to_linexpr(cond[2])
            if l is None or r is None:
                return []
            return [Constraint.eq(l - r)]
        elif tag == 'ne':
            # x != y is not convex, overapproximate as TOP
            return []
        elif tag == 'and':
            return self._cond_to_constraints(cond[1]) + self._cond_to_constraints(cond[2])
        elif tag == 'or':
            # Overapproximate: join of the two branches (weaken)
            # We can't represent disjunction in a single polyhedron
            return []
        elif tag == 'not':
            return self._negate_cond(cond[1])
        else:
            return []

    def _negate_cond(self, cond):
        """Negate a condition and return constraints."""
        tag = cond[0]
        if tag == 'true':
            return self._cond_to_constraints(('false',))
        elif tag == 'false':
            return self._cond_to_constraints(('true',))
        elif tag == 'le':
            return self._cond_to_constraints(('gt', cond[1], cond[2]))
        elif tag == 'lt':
            return self._cond_to_constraints(('ge', cond[1], cond[2]))
        elif tag == 'ge':
            return self._cond_to_constraints(('lt', cond[1], cond[2]))
        elif tag == 'gt':
            return self._cond_to_constraints(('le', cond[1], cond[2]))
        elif tag == 'eq':
            # not(eq) -> ne, which is non-convex, overapproximate
            return []
        elif tag == 'ne':
            return self._cond_to_constraints(('eq', cond[1], cond[2]))
        elif tag == 'and':
            # not(A and B) = not(A) or not(B) -- disjunction, overapproximate
            return []
        elif tag == 'or':
            # not(A or B) = not(A) and not(B)
            return self._negate_cond(cond[1]) + self._negate_cond(cond[2])
        elif tag == 'not':
            return self._cond_to_constraints(cond[1])
        else:
            return []


@dataclass
class AnalysisResult:
    final_state: Polyhedron
    warnings: List[str]


# ---------------------------------------------------------------------------
# Convenience / Composition APIs
# ---------------------------------------------------------------------------

def polyhedra_from_intervals(intervals):
    """Create a polyhedron from per-variable intervals.
    intervals: {var: (lo, hi)} where lo/hi can be None for unbounded.
    """
    constraints = []
    for var, (lo, hi) in intervals.items():
        if lo is not None:
            constraints.append(Constraint.var_ge_const(var, lo))
        if hi is not None:
            constraints.append(Constraint.var_le_const(var, hi))
    return Polyhedron.from_constraints(constraints)


def analyze_program(program, init=None, max_iterations=50, widen_delay=2):
    """Convenience: analyze a program and return result."""
    interp = PolyhedraInterpreter(max_iterations, widen_delay)
    return interp.analyze(program, init)


def compare_with_intervals(program, init_intervals=None):
    """Compare polyhedra analysis (relational) with interval-only analysis.
    Shows the precision gain from relational tracking.
    """
    # Polyhedra analysis
    init = polyhedra_from_intervals(init_intervals) if init_intervals else None
    interp = PolyhedraInterpreter()
    poly_result = interp.analyze(program, init)

    # Extract intervals from polyhedra
    poly_intervals = {}
    if not poly_result.final_state.is_bot():
        poly_intervals = poly_result.final_state.extract_intervals()

    # Simple interval analysis (no relational info)
    iv_init = Polyhedron.top()
    if init_intervals:
        for var, (lo, hi) in init_intervals.items():
            if lo is not None:
                iv_init = iv_init.guard(Constraint.var_ge_const(var, lo))
            if hi is not None:
                iv_init = iv_init.guard(Constraint.var_le_const(var, hi))

    return {
        'polyhedra_state': poly_result.final_state,
        'polyhedra_intervals': poly_intervals,
        'polyhedra_warnings': poly_result.warnings,
        'polyhedra_constraints': poly_result.final_state.num_constraints()
            if not poly_result.final_state.is_bot() else 0,
        'relational_properties': poly_result.final_state.extract_equalities()
            if not poly_result.final_state.is_bot() else [],
        'orderings': poly_result.final_state.extract_orderings()
            if not poly_result.final_state.is_bot() else [],
    }


def verify_relational_property(program, property_constraint, init=None):
    """Verify that a relational property holds at program exit.
    Returns (verified: bool, counterexample: optional).
    """
    interp = PolyhedraInterpreter()
    result = interp.analyze(program, init)
    if result.final_state.is_bot():
        return True, None  # unreachable exit -> property holds vacuously
    if result.final_state._implies_constraint(property_constraint):
        return True, None
    return False, result.final_state


def batch_analyze(programs, init=None):
    """Analyze multiple programs, return list of results."""
    interp = PolyhedraInterpreter()
    results = []
    for prog in programs:
        results.append(interp.analyze(prog, init))
    return results
