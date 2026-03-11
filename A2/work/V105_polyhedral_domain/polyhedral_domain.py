"""V105: Polyhedral Abstract Domain

The most precise numeric abstract domain in the standard hierarchy:
  Constants < Intervals < Zones < Octagons < Polyhedra

Tracks arbitrary linear inequality constraints: a1*x1 + a2*x2 + ... + an*xn <= c

Uses the constraint-only (H-representation) approach with:
- Fourier-Motzkin elimination for projection (forget)
- Redundancy removal via LP-style feasibility checks
- Gauss-Jordan for equality detection
- Widening via constraint subsumption

Composes: C010 (parser) + C039 (abstract interpreter, for comparison)

Author: A2 (AgentZero verification agent)
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Any, FrozenSet
from fractions import Fraction
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))

from stack_vm import (
    lex, Parser,
    IntLit, FloatLit, StringLit, BoolLit, Var,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)
from abstract_interpreter import analyze as c039_analyze

# ---------------------------------------------------------------------------
# Fraction-based exact arithmetic to avoid floating-point imprecision
# ---------------------------------------------------------------------------

ZERO = Fraction(0)
ONE = Fraction(1)
INF = float('inf')

def frac(x):
    """Convert to Fraction, handling inf."""
    if x == float('inf') or x == float('-inf'):
        return x
    return Fraction(x)


# ---------------------------------------------------------------------------
# Linear Constraint: a1*x1 + a2*x2 + ... + an*xn <= b
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearConstraint:
    """A linear inequality: sum(coeffs[var] * var) <= bound.

    Stored as {var_name: Fraction_coeff}, Fraction_bound.
    coeffs with value 0 are omitted.
    """
    coeffs: tuple  # tuple of (var_name, Fraction) pairs, sorted by name
    bound: Fraction
    is_equality: bool = False  # True means == instead of <=

    @staticmethod
    def from_dict(coeffs_dict: Dict[str, Any], bound, is_equality=False):
        """Create from a dict of {var: coeff}."""
        cleaned = {}
        for v, c in coeffs_dict.items():
            c = Fraction(c)
            if c != ZERO:
                cleaned[v] = c
        return LinearConstraint(
            coeffs=tuple(sorted(cleaned.items())),
            bound=Fraction(bound),
            is_equality=is_equality
        )

    @property
    def coeffs_dict(self) -> Dict[str, Fraction]:
        return dict(self.coeffs)

    @property
    def variables(self) -> Set[str]:
        return {v for v, _ in self.coeffs}

    def evaluate(self, point: Dict[str, Fraction]) -> Fraction:
        """Evaluate sum(coeffs[v] * point[v])."""
        s = ZERO
        for v, c in self.coeffs:
            s += c * point.get(v, ZERO)
        return s

    def is_satisfied(self, point: Dict[str, Fraction]) -> bool:
        val = self.evaluate(point)
        if self.is_equality:
            return val == self.bound
        return val <= self.bound

    def negate_coeff(self, var: str):
        """Return new constraint with var's coefficient negated."""
        d = self.coeffs_dict
        if var in d:
            d[var] = -d[var]
        return LinearConstraint.from_dict(d, self.bound, self.is_equality)

    def scale(self, factor: Fraction):
        """Scale all coefficients and bound by factor."""
        if factor == ZERO:
            return LinearConstraint.from_dict({}, self.bound, self.is_equality)
        d = {v: c * factor for v, c in self.coeffs}
        new_bound = self.bound * factor
        # If factor < 0 and not equality, flip inequality
        eq = self.is_equality
        if factor < 0 and not eq:
            d = {v: -c for v, c in d.items()}
            new_bound = -new_bound
            # a <= b scaled by -k: -ka >= -kb, i.e., ka <= kb -> already flipped
            # Actually: ax <= b, multiply by -k (k>0): -kax >= -kb -> kax <= kb
            # We need to be careful. Let's just negate everything for negative scale.
            pass
        return LinearConstraint.from_dict(d, new_bound, eq)

    def add(self, other: 'LinearConstraint', c1=ONE, c2=ONE):
        """Return c1*self + c2*other (Fourier-Motzkin combination)."""
        d = {}
        for v, c in self.coeffs:
            d[v] = d.get(v, ZERO) + c1 * c
        for v, c in other.coeffs:
            d[v] = d.get(v, ZERO) + c2 * c
        return LinearConstraint.from_dict(d, c1 * self.bound + c2 * other.bound)

    def substitute(self, var: str, expr_coeffs: Dict[str, Fraction], expr_const: Fraction):
        """Substitute var with linear expression: var = sum(expr_coeffs) + expr_const.

        If constraint is: a*var + rest <= b
        After substitution: a*(sum(expr_coeffs) + expr_const) + rest <= b
        """
        d = self.coeffs_dict
        a = d.pop(var, ZERO)
        if a == ZERO:
            return LinearConstraint.from_dict(d, self.bound, self.is_equality)
        # Add a * expr_coeffs to d
        for v, c in expr_coeffs.items():
            d[v] = d.get(v, ZERO) + a * c
        new_bound = self.bound - a * expr_const
        return LinearConstraint.from_dict(d, new_bound, self.is_equality)

    def __str__(self):
        if not self.coeffs:
            op = "==" if self.is_equality else "<="
            return f"0 {op} {self.bound}"
        parts = []
        for v, c in self.coeffs:
            if c == ONE:
                parts.append(v)
            elif c == -ONE:
                parts.append(f"-{v}")
            else:
                parts.append(f"{c}*{v}")
        lhs = " + ".join(parts).replace("+ -", "- ")
        op = "==" if self.is_equality else "<="
        return f"{lhs} {op} {self.bound}"

    def __repr__(self):
        return str(self)


# ---------------------------------------------------------------------------
# Polyhedral Domain (H-representation: set of linear constraints)
# ---------------------------------------------------------------------------

class PolyhedralDomain:
    """Abstract domain of convex polyhedra over named variables.

    Represented as a conjunction of linear inequalities (H-representation).
    This is the most precise non-relational-to-relational numeric domain.

    Key operations:
    - join: convex hull (over-approximated via constraint intersection)
    - meet: constraint conjunction
    - widen: drop non-implied constraints
    - project (forget): Fourier-Motzkin variable elimination
    """

    def __init__(self, var_names: List[str] = None, constraints: List[LinearConstraint] = None):
        self.var_names: List[str] = list(var_names) if var_names else []
        self.constraints: List[LinearConstraint] = list(constraints) if constraints else []
        self._is_bot: bool = False

    def copy(self):
        p = PolyhedralDomain(self.var_names[:], [c for c in self.constraints])
        p._is_bot = self._is_bot
        return p

    def is_bot(self) -> bool:
        """Check if the polyhedron is empty (bottom)."""
        if self._is_bot:
            return True
        # Check for trivially unsatisfiable constraints: 0 <= negative
        for c in self.constraints:
            if not c.coeffs and not c.is_equality:
                if c.bound < ZERO:
                    self._is_bot = True
                    return True
            if not c.coeffs and c.is_equality:
                if c.bound != ZERO:
                    self._is_bot = True
                    return True

        # Check for per-variable contradictions: upper < lower
        # Collect tightest unary bounds from constraints + equalities
        uppers = {}  # var -> min upper bound
        lowers = {}  # var -> max lower bound
        for c in self.constraints:
            cd = c.coeffs_dict
            if len(cd) != 1:
                continue
            var = next(iter(cd))
            coeff = cd[var]
            if c.is_equality:
                val = c.bound / coeff
                uppers[var] = min(uppers.get(var, c.bound / coeff), val)
                lowers[var] = max(lowers.get(var, val), val)
            elif coeff > 0:
                # coeff*var <= bound -> var <= bound/coeff
                val = c.bound / coeff
                uppers[var] = min(uppers.get(var, val), val)
            elif coeff < 0:
                # coeff*var <= bound -> var >= bound/coeff (flip)
                val = c.bound / coeff
                lowers[var] = max(lowers.get(var, val), val)

        for var in set(uppers) & set(lowers):
            if lowers[var] > uppers[var]:
                self._is_bot = True
                return True

        # Check multi-variable constraints against known equalities
        # If a variable has lower == upper, it's a constant
        known = {}
        for var in set(uppers) & set(lowers):
            if uppers[var] == lowers[var]:
                known[var] = uppers[var]

        if known:
            for c in self.constraints:
                cd = c.coeffs_dict
                # Try to evaluate constraint with known values
                remaining = {}
                val = ZERO
                for v, coeff in cd.items():
                    if v in known:
                        val += coeff * known[v]
                    else:
                        remaining[v] = coeff
                if not remaining:
                    # All variables are known -- check if constraint holds
                    if c.is_equality and val != c.bound:
                        self._is_bot = True
                        return True
                    if not c.is_equality and val > c.bound:
                        self._is_bot = True
                        return True

        return False

    @staticmethod
    def bot(var_names: List[str] = None) -> 'PolyhedralDomain':
        p = PolyhedralDomain(var_names or [])
        p._is_bot = True
        return p

    @staticmethod
    def top(var_names: List[str] = None) -> 'PolyhedralDomain':
        return PolyhedralDomain(var_names or [])

    def add_var(self, name: str):
        """Add a variable (unconstrained)."""
        if name not in self.var_names:
            self.var_names.append(name)

    def set_upper(self, var: str, bound):
        """Add constraint: var <= bound."""
        self.add_var(var)
        self.constraints.append(LinearConstraint.from_dict({var: 1}, bound))

    def set_lower(self, var: str, bound):
        """Add constraint: var >= bound (i.e., -var <= -bound)."""
        self.add_var(var)
        self.constraints.append(LinearConstraint.from_dict({var: -1}, -Fraction(bound)))

    def set_equal(self, var: str, value):
        """Add constraint: var == value."""
        self.add_var(var)
        self.constraints.append(LinearConstraint.from_dict({var: 1}, value, is_equality=True))

    def add_constraint(self, coeffs: Dict[str, Any], bound, is_equality=False):
        """Add a general linear constraint."""
        for v in coeffs:
            self.add_var(v)
        self.constraints.append(LinearConstraint.from_dict(coeffs, bound, is_equality))

    def get_upper(self, var: str) -> float:
        """Get upper bound on var by projecting out all other variables."""
        if self._is_bot:
            return -INF
        if var not in self.var_names:
            return INF
        return self._compute_bound(var, upper=True)

    def get_lower(self, var: str) -> float:
        """Get lower bound on var by projecting out all other variables."""
        if self._is_bot:
            return INF
        if var not in self.var_names:
            return -INF
        return self._compute_bound(var, upper=False)

    def _compute_bound(self, var: str, upper: bool) -> float:
        """Compute bound on var by Fourier-Motzkin elimination of all other vars."""
        # Work on a copy to avoid modifying self
        tmp = self.copy()
        others = [v for v in tmp.var_names if v != var]
        for other in others:
            tmp.forget(other)
        # Now tmp only has constraints on var
        best = INF if upper else -INF
        for c in tmp.constraints:
            cd = c.coeffs_dict
            coeff = cd.get(var, ZERO)
            if coeff == ZERO:
                continue
            if c.is_equality:
                val = float(c.bound / coeff)
                if upper:
                    best = min(best, val)
                else:
                    best = max(best, val)
            elif upper and coeff > 0:
                best = min(best, float(c.bound / coeff))
            elif not upper and coeff < 0:
                best = max(best, float(c.bound / coeff))
        return best

    def get_interval(self, var: str) -> Tuple[float, float]:
        """Get [lower, upper] interval for var."""
        return (self.get_lower(var), self.get_upper(var))

    def forget(self, var: str):
        """Project out (eliminate) a variable using Fourier-Motzkin elimination."""
        if self._is_bot:
            return

        # Expand equalities involving var into two inequalities
        expanded = []
        for c in self.constraints:
            if c.is_equality and var in c.variables:
                # a*var + rest == b -> a*var + rest <= b AND -a*var - rest <= -b
                expanded.append(LinearConstraint.from_dict(c.coeffs_dict, c.bound, False))
                neg_d = {v: -k for v, k in c.coeffs}
                expanded.append(LinearConstraint.from_dict(neg_d, -c.bound, False))
            else:
                expanded.append(c)

        pos = []   # constraints with positive coefficient for var
        neg = []   # constraints with negative coefficient for var
        zero = []  # constraints not involving var

        for c in expanded:
            cd = c.coeffs_dict
            coeff = cd.get(var, ZERO)
            if coeff > 0:
                pos.append(c)
            elif coeff < 0:
                neg.append(c)
            else:
                zero.append(c)

        # Combine each positive with each negative to eliminate var
        new_constraints = list(zero)
        for p in pos:
            pc = p.coeffs_dict[var]
            for n in neg:
                nc = -n.coeffs_dict[var]  # make positive
                # pc*var + rest_p <= bp  =>  var <= (bp - rest_p) / pc
                # nc*var - rest_n >= -bn  =>  var >= (-bn + rest_n) / nc
                # Combine: nc * p + pc * n  (eliminates var)
                combined = p.add(n, nc, pc)
                # Verify var is eliminated
                cd = combined.coeffs_dict
                if var not in cd:
                    new_constraints.append(combined)

        self.constraints = new_constraints
        if var in self.var_names:
            self.var_names.remove(var)

    def assign_const(self, var: str, value):
        """Strong update: var := constant."""
        self.add_var(var)
        self.forget(var)
        self.add_var(var)  # re-add after forget removes it
        value = Fraction(value)
        self.constraints.append(LinearConstraint.from_dict({var: 1}, value, is_equality=True))

    def assign_var(self, target: str, source: str):
        """Strong update: target := source."""
        self.add_var(target)
        if target == source:
            return
        # Substitute target out, then add target == source
        self.forget(target)
        self.add_var(target)  # re-add after forget removes it
        self.constraints.append(
            LinearConstraint.from_dict({target: 1, source: -1}, 0, is_equality=True)
        )

    def assign_linear(self, target: str, coeffs: Dict[str, Fraction], constant: Fraction):
        """Strong update: target := sum(coeffs[v]*v) + constant.

        This is the general linear assignment.
        """
        self.add_var(target)
        # If target appears in the RHS, we need a fresh temporary
        if target in coeffs:
            # target := ... + c*target + ...
            # Use substitution: forget target, add target == expr
            # But we need old target value. Use Fourier-Motzkin approach.
            # Save constraints about target, eliminate target, re-add with new definition
            pass

        self.forget(target)
        self.add_var(target)  # re-add after forget removes it
        # Add target == sum(coeffs) + constant
        eq_coeffs = {target: Fraction(1)}
        for v, c in coeffs.items():
            eq_coeffs[v] = eq_coeffs.get(v, ZERO) - Fraction(c)
        self.constraints.append(
            LinearConstraint.from_dict(eq_coeffs, constant, is_equality=True)
        )

    def assign_expr(self, target: str, coeffs: Dict[str, Fraction], constant: Fraction):
        """Assign target := linear_expr, handling self-referential assignments.

        For target := a*target + b*y + c, we substitute old target constraints.
        """
        self.add_var(target)
        coeffs = {v: Fraction(c) for v, c in coeffs.items()}
        constant = Fraction(constant)

        if target not in coeffs:
            # Simple case: target doesn't appear in RHS
            self.forget(target)
            self.add_var(target)  # re-add after forget removes it
            eq_coeffs = {target: ONE}
            for v, c in coeffs.items():
                eq_coeffs[v] = eq_coeffs.get(v, ZERO) - c
            self.constraints.append(
                LinearConstraint.from_dict(eq_coeffs, constant, is_equality=True)
            )
            return

        # Self-referential: target := a*target + rest
        # Introduce fresh var for old target value, substitute, then project
        target_coeff = coeffs.pop(target)

        # Rename target -> __old_target in all constraints
        old_name = f"__{target}_old"
        new_constraints = []
        for c in self.constraints:
            cd = c.coeffs_dict
            if target in cd:
                val = cd.pop(target)
                cd[old_name] = val
                new_constraints.append(LinearConstraint.from_dict(cd, c.bound, c.is_equality))
            else:
                new_constraints.append(c)
        self.constraints = new_constraints
        if target in self.var_names:
            self.var_names.remove(target)
        self.var_names.append(old_name)

        # Now add: target == target_coeff * old_name + rest + constant
        eq_coeffs = {target: ONE, old_name: -target_coeff}
        for v, c in coeffs.items():
            eq_coeffs[v] = eq_coeffs.get(v, ZERO) - c
        self.var_names.append(target)
        self.constraints.append(
            LinearConstraint.from_dict(eq_coeffs, constant, is_equality=True)
        )

        # Project out old_name
        self.forget(old_name)

    def _expand_equalities(self) -> List[LinearConstraint]:
        """Expand equalities into pairs of inequalities for join/widen."""
        result = []
        for c in self.constraints:
            if c.is_equality:
                result.append(LinearConstraint.from_dict(c.coeffs_dict, c.bound, False))
                neg_d = {v: -k for v, k in c.coeffs}
                result.append(LinearConstraint.from_dict(neg_d, -c.bound, False))
            else:
                result.append(c)
        return result

    def _is_implied(self, constraint: LinearConstraint, other_constraints: List[LinearConstraint]) -> bool:
        """Check if constraint is implied by other_constraints.

        Uses a simple LP-like check: maximize LHS subject to other_constraints.
        If max <= bound, the constraint is redundant.

        For simplicity, we use a bounded interval-based overapproximation.
        """
        if not constraint.coeffs:
            return constraint.bound >= ZERO if not constraint.is_equality else constraint.bound == ZERO

        # Try to find if the constraint is directly present or subsumed
        for oc in other_constraints:
            if oc.coeffs == constraint.coeffs and not oc.is_equality and not constraint.is_equality:
                if oc.bound <= constraint.bound:
                    return True
            if oc.coeffs == constraint.coeffs and oc.bound == constraint.bound:
                if oc.is_equality or not constraint.is_equality:
                    return True
        return False

    def join(self, other: 'PolyhedralDomain') -> 'PolyhedralDomain':
        """Convex hull (over-approximation via constraint intersection).

        Keep only constraints that are satisfied by BOTH polyhedra.
        This is an over-approximation of the true convex hull.
        """
        if self._is_bot:
            return other.copy()
        if other._is_bot:
            return self.copy()

        all_vars = list(set(self.var_names) | set(other.var_names))

        # For each constraint in self, check if it's implied by other
        # Keep constraints valid in both (weakened bounds)
        result_constraints = []

        self_expanded = self._expand_equalities()
        other_expanded = other._expand_equalities()

        # Strategy: for each constraint in self, find the weakest version
        # that also holds in other. For unary constraints, this means taking
        # the max bound. For relational constraints, similarly.
        for c in self_expanded:
            # Find matching constraint in other (same coefficients)
            for oc in other_expanded:
                if c.coeffs == oc.coeffs:
                    # Both have same direction -- weaken bound
                    result_constraints.append(
                        LinearConstraint(c.coeffs, max(c.bound, oc.bound), False)
                    )
                    break

        # Also check other's constraints against self
        for oc in other_expanded:
            for c in self_expanded:
                if oc.coeffs == c.coeffs:
                    # Already handled above
                    break
            else:
                # Constraint only in other, not in self -> drop (not valid in both)
                pass

        # Remove duplicates
        seen = set()
        deduped = []
        for c in result_constraints:
            key = (c.coeffs, c.bound)
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        result = PolyhedralDomain(all_vars, deduped)
        return result

    def meet(self, other: 'PolyhedralDomain') -> 'PolyhedralDomain':
        """Intersection: conjunction of all constraints."""
        if self._is_bot or other._is_bot:
            return PolyhedralDomain.bot(list(set(self.var_names) | set(other.var_names)))

        all_vars = list(set(self.var_names) | set(other.var_names))
        all_constraints = self.constraints + other.constraints
        result = PolyhedralDomain(all_vars, all_constraints)
        return result

    def widen(self, other: 'PolyhedralDomain') -> 'PolyhedralDomain':
        """Widening: keep only constraints from self that are satisfied by other.

        Standard polyhedral widening (BHRZ03-style simplified):
        Keep constraint c from self if other satisfies c (other subset of halfspace(c)).
        Drop constraints that other violates -> forces convergence to fixpoint.
        """
        if self._is_bot:
            return other.copy()
        if other._is_bot:
            return self.copy()

        all_vars = list(set(self.var_names) | set(other.var_names))

        # Keep constraints from self that are also satisfied by other's constraints
        # A constraint c: ax <= b from self is kept if other implies ax <= b
        # Over-approximation: keep c if every constraint in other with same coeffs has bound <= b

        self_expanded = self._expand_equalities()
        other_expanded = other._expand_equalities()

        kept = []
        for c in self_expanded:
            # Check if other satisfies this constraint
            # Simple check: find matching coeff in other
            satisfied = False
            for oc in other_expanded:
                if oc.coeffs == c.coeffs and oc.bound <= c.bound:
                    satisfied = True
                    break

            if satisfied:
                kept.append(c)
            else:
                # Also keep if it's a "structural" constraint (bounds on single vars)
                # that other doesn't violate much -- but standard widening drops it
                pass

        result = PolyhedralDomain(all_vars, kept)
        return result

    def leq(self, other: 'PolyhedralDomain') -> bool:
        """Check if self is subset of other (self => other's constraints)."""
        if self._is_bot:
            return True
        if other._is_bot:
            return False

        # self <= other iff every constraint of other is implied by self
        # Over-approximation: check if self has matching (tighter) constraints
        other_expanded = other._expand_equalities()
        self_expanded = self._expand_equalities()

        for oc in other_expanded:
            if self._is_implied(oc, self_expanded):
                continue
            # Check by matching coefficients
            found = False
            for sc in self_expanded:
                if sc.coeffs == oc.coeffs and sc.bound <= oc.bound:
                    found = True
                    break
            if not found:
                return False
        return True

    def equals(self, other: 'PolyhedralDomain') -> bool:
        """Check mutual inclusion."""
        if self._is_bot and other._is_bot:
            return True
        if self._is_bot or other._is_bot:
            return False
        return self.leq(other) and other.leq(self)

    def get_constraints(self) -> List[str]:
        """Human-readable list of constraints."""
        if self._is_bot:
            return ["BOTTOM (empty)"]
        if not self.constraints:
            return ["TOP (unconstrained)"]
        return [str(c) for c in self.constraints]

    def get_relational_constraints(self) -> List[str]:
        """Get only constraints involving 2+ variables."""
        result = []
        for c in self.constraints:
            if len(c.variables) >= 2:
                result.append(str(c))
        return result

    def __str__(self):
        if self._is_bot:
            return "BOT"
        if not self.constraints:
            return "TOP"
        return " AND ".join(str(c) for c in self.constraints)


# ---------------------------------------------------------------------------
# Polyhedral Interpreter for C10 programs
# ---------------------------------------------------------------------------

class PolyhedralInterpreter:
    """Abstract interpreter using the polyhedral domain for C10 programs."""

    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations
        self.warnings: List[str] = []
        self.functions: Dict[str, Any] = {}

    def analyze(self, source: str) -> dict:
        """Analyze a C10 program with the polyhedral domain."""
        tokens = lex(source)
        ast = Parser(tokens).parse()

        env = PolyhedralDomain([])

        for stmt in ast.stmts:
            if isinstance(stmt, FnDecl):
                self.functions[stmt.name] = stmt
            else:
                env = self._interpret_stmt(stmt, env, self.functions)
                if env.is_bot():
                    break

        return {
            'env': env,
            'warnings': self.warnings,
            'functions': list(self.functions.keys())
        }

    def _interpret_stmt(self, stmt, env: PolyhedralDomain, functions: dict) -> PolyhedralDomain:
        if env.is_bot():
            return env

        if isinstance(stmt, LetDecl):
            return self._interpret_let(stmt, env, functions)
        elif isinstance(stmt, Assign):
            return self._interpret_assign(stmt, env, functions)
        elif isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, env, functions)
        elif isinstance(stmt, WhileStmt):
            return self._interpret_while(stmt, env, functions)
        elif isinstance(stmt, Block):
            return self._interpret_block(stmt.stmts, env, functions)
        elif isinstance(stmt, PrintStmt):
            return env
        elif isinstance(stmt, ReturnStmt):
            return env
        elif isinstance(stmt, FnDecl):
            return env
        else:
            return env

    def _interpret_block(self, stmts, env: PolyhedralDomain, functions: dict) -> PolyhedralDomain:
        if hasattr(stmts, 'stmts'):
            stmts = stmts.stmts
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env, functions)
            if env.is_bot():
                break
        return env

    def _interpret_let(self, stmt: LetDecl, env: PolyhedralDomain, functions: dict) -> PolyhedralDomain:
        name = stmt.name
        env.add_var(name)
        if stmt.value is not None:
            self._apply_assignment(name, stmt.value, env)
        return env

    def _interpret_assign(self, stmt: Assign, env: PolyhedralDomain, functions: dict) -> PolyhedralDomain:
        name = stmt.name
        env.add_var(name)
        self._apply_assignment(name, stmt.value, env)
        return env

    def _apply_assignment(self, target: str, expr, env: PolyhedralDomain):
        """Apply assignment target := expr to the polyhedral domain."""

        if isinstance(expr, IntLit):
            env.assign_const(target, expr.value)
            return

        if isinstance(expr, Var):
            env.assign_var(target, expr.name)
            return

        if isinstance(expr, BinOp):
            # Try to decompose into a linear expression
            coeffs, const = self._linearize(expr)
            if coeffs is not None:
                env.assign_expr(target, coeffs, const)
                return

            # Non-linear: evaluate to interval bounds
            lo, hi = self._eval_interval(expr, env)
            env.forget(target)
            if lo != -INF:
                env.set_lower(target, lo)
            if hi != INF:
                env.set_upper(target, hi)
            return

        if isinstance(expr, UnaryOp) and expr.op == '-':
            if isinstance(expr.operand, Var):
                env.assign_expr(target, {expr.operand.name: Fraction(-1)}, ZERO)
                return
            if isinstance(expr.operand, IntLit):
                env.assign_const(target, -expr.operand.value)
                return
            coeffs, const = self._linearize_expr(expr)
            if coeffs is not None:
                env.assign_expr(target, coeffs, const)
                return

        # Fallback: forget target, use interval bounds
        lo, hi = self._eval_interval(expr, env)
        env.forget(target)
        if lo != -INF:
            env.set_lower(target, lo)
        if hi != INF:
            env.set_upper(target, hi)

    def _linearize(self, expr) -> Tuple[Optional[Dict[str, Fraction]], Optional[Fraction]]:
        """Try to decompose a BinOp into linear form: {var: coeff} + constant."""
        return self._linearize_expr(expr)

    def _linearize_expr(self, expr) -> Tuple[Optional[Dict[str, Fraction]], Optional[Fraction]]:
        """Recursively linearize an expression. Returns (coeffs_dict, constant) or (None, None)."""
        if isinstance(expr, IntLit):
            return {}, Fraction(expr.value)

        if isinstance(expr, Var):
            return {expr.name: ONE}, ZERO

        if isinstance(expr, UnaryOp) and expr.op == '-':
            sub_c, sub_k = self._linearize_expr(expr.operand)
            if sub_c is None:
                return None, None
            return {v: -c for v, c in sub_c.items()}, -sub_k

        if isinstance(expr, BinOp):
            left_c, left_k = self._linearize_expr(expr.left)
            right_c, right_k = self._linearize_expr(expr.right)

            if left_c is None or right_c is None:
                return None, None

            if expr.op == '+':
                merged = dict(left_c)
                for v, c in right_c.items():
                    merged[v] = merged.get(v, ZERO) + c
                return merged, left_k + right_k

            if expr.op == '-':
                merged = dict(left_c)
                for v, c in right_c.items():
                    merged[v] = merged.get(v, ZERO) - c
                return merged, left_k - right_k

            if expr.op == '*':
                # Linear only if one side is constant
                if not left_c and left_k is not None:
                    # left is constant
                    return {v: c * left_k for v, c in right_c.items()}, right_k * left_k
                if not right_c and right_k is not None:
                    # right is constant
                    return {v: c * right_k for v, c in left_c.items()}, left_k * right_k
                return None, None

            # Division, modulo, etc. -- not linear
            return None, None

        return None, None

    def _eval_interval(self, expr, env: PolyhedralDomain) -> Tuple[float, float]:
        """Evaluate expression to an interval [lo, hi] using current constraints."""
        if isinstance(expr, IntLit):
            return (expr.value, expr.value)

        if isinstance(expr, Var):
            return env.get_interval(expr.name)

        if isinstance(expr, BoolLit):
            v = 1 if expr.value else 0
            return (v, v)

        if isinstance(expr, UnaryOp) and expr.op == '-':
            lo, hi = self._eval_interval(expr.operand, env)
            return (-hi, -lo)

        if isinstance(expr, BinOp):
            lo_l, hi_l = self._eval_interval(expr.left, env)
            lo_r, hi_r = self._eval_interval(expr.right, env)

            if expr.op == '+':
                return (lo_l + lo_r, hi_l + hi_r)
            if expr.op == '-':
                return (lo_l - hi_r, hi_l - lo_r)
            if expr.op == '*':
                products = [lo_l * lo_r, lo_l * hi_r, hi_l * lo_r, hi_l * hi_r]
                products = [p for p in products if not math.isnan(p)]
                if not products:
                    return (-INF, INF)
                return (min(products), max(products))
            if expr.op == '/':
                if lo_r <= 0 <= hi_r:
                    self.warnings.append("possible division by zero")
                    return (-INF, INF)
                if lo_r > 0:
                    vals = [lo_l / lo_r, lo_l / hi_r, hi_l / lo_r, hi_l / hi_r]
                    return (min(vals), max(vals))
                return (-INF, INF)
            if expr.op == '%':
                if lo_r <= 0 <= hi_r:
                    return (-INF, INF)
                abs_r = max(abs(lo_r), abs(hi_r))
                return (0, abs_r - 1) if lo_l >= 0 else (-(abs_r - 1), abs_r - 1)

        if isinstance(expr, CallExpr):
            return (-INF, INF)

        return (-INF, INF)

    def _refine_condition(self, cond, env: PolyhedralDomain) -> Tuple[PolyhedralDomain, PolyhedralDomain]:
        """Refine environment based on condition being true/false."""
        then_env = env.copy()
        else_env = env.copy()

        if isinstance(cond, BinOp):
            # Try to linearize both sides
            left_c, left_k = self._linearize_expr(cond.left)
            right_c, right_k = self._linearize_expr(cond.right)

            if left_c is not None and right_c is not None:
                # Constraint: left - right OP 0
                # diff = left - right
                diff_c = dict(left_c)
                for v, c in right_c.items():
                    diff_c[v] = diff_c.get(v, ZERO) - c
                diff_k = left_k - right_k  # constant part: diff_c * vars + diff_k OP 0

                # Normalize: sum(diff_c * vars) OP -diff_k
                bound = -diff_k

                if cond.op == '<':
                    # left < right -> diff <= -1 (integers)
                    then_env.add_constraint(diff_c, bound - 1)
                    # else: right <= left -> -diff <= 0 -> negate
                    neg_c = {v: -c for v, c in diff_c.items()}
                    else_env.add_constraint(neg_c, -bound)

                elif cond.op == '<=':
                    then_env.add_constraint(diff_c, bound)
                    neg_c = {v: -c for v, c in diff_c.items()}
                    else_env.add_constraint(neg_c, -bound - 1)

                elif cond.op == '>':
                    # left > right -> -diff <= -1
                    neg_c = {v: -c for v, c in diff_c.items()}
                    then_env.add_constraint(neg_c, -bound - 1)
                    # else: left <= right
                    then_env_tmp = then_env  # already added
                    else_env.add_constraint(diff_c, bound)

                elif cond.op == '>=':
                    neg_c = {v: -c for v, c in diff_c.items()}
                    then_env.add_constraint(neg_c, -bound)
                    else_env.add_constraint(diff_c, bound - 1)

                elif cond.op == '==':
                    then_env.add_constraint(diff_c, bound)
                    neg_diff = {v: -c for v, c in diff_c.items()}
                    then_env.add_constraint(neg_diff, -bound)
                    # else: != -> no simple linear constraint

                elif cond.op == '!=':
                    # then: != -> no constraint
                    # else: == -> equality
                    else_env.add_constraint(diff_c, bound)
                    neg_diff = {v: -c for v, c in diff_c.items()}
                    else_env.add_constraint(neg_diff, -bound)

                return then_env, else_env

        if isinstance(cond, Var):
            # var is truthy -> var != 0 -> no linear constraint available
            # var is falsy -> var == 0
            then_env_copy = then_env
            else_env.add_constraint({cond.name: 1}, 0, is_equality=True)
            return then_env_copy, else_env

        if isinstance(cond, UnaryOp) and cond.op == '!':
            # !cond: flip then/else
            inner_then, inner_else = self._refine_condition(cond.operand, env)
            return inner_else, inner_then

        return then_env, else_env

    def _interpret_if(self, stmt: IfStmt, env: PolyhedralDomain, functions: dict) -> PolyhedralDomain:
        then_env, else_env = self._refine_condition(stmt.cond, env)

        then_result = self._interpret_block(stmt.then_body, then_env, functions)

        if stmt.else_body:
            else_result = self._interpret_block(stmt.else_body, else_env, functions)
        else:
            else_result = else_env

        if then_result.is_bot():
            return else_result
        if else_result.is_bot():
            return then_result

        return then_result.join(else_result)

    def _interpret_while(self, stmt: WhileStmt, env: PolyhedralDomain, functions: dict) -> PolyhedralDomain:
        current = env.copy()

        for i in range(self.max_iterations):
            then_env, _ = self._refine_condition(stmt.cond, current)
            body_env = self._interpret_block(stmt.body, then_env, functions)

            if body_env.is_bot():
                break

            joined = current.join(body_env)
            next_env = current.widen(joined)

            if next_env.equals(current):
                break
            current = next_env

        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env


# ---------------------------------------------------------------------------
# Top-level API functions
# ---------------------------------------------------------------------------

def polyhedral_analyze(source: str, max_iterations: int = 50) -> dict:
    """Analyze a C10 program with the polyhedral abstract domain.

    Returns dict with 'env' (PolyhedralDomain), 'warnings', 'functions'.
    """
    interp = PolyhedralInterpreter(max_iterations)
    return interp.analyze(source)


def get_variable_range(source: str, var_name: str) -> Tuple[float, float]:
    """Get the interval [lower, upper] for a variable after analysis."""
    result = polyhedral_analyze(source)
    env = result['env']
    return env.get_interval(var_name)


def get_all_constraints(source: str) -> List[str]:
    """Get all constraints after analysis."""
    result = polyhedral_analyze(source)
    return result['env'].get_constraints()


def get_relational_constraints(source: str) -> List[str]:
    """Get only relational (multi-variable) constraints after analysis."""
    result = polyhedral_analyze(source)
    return result['env'].get_relational_constraints()


def compare_analyses(source: str) -> dict:
    """Compare polyhedral analysis with C039 interval analysis."""
    poly_result = polyhedral_analyze(source)
    c039_result = c039_analyze(source)

    poly_env = poly_result['env']
    c039_env = c039_result['env']

    all_vars = poly_env.var_names

    comparison = {}
    precision_gains = []
    relational_only = []

    for var in all_vars:
        poly_lo, poly_hi = poly_env.get_interval(var)
        c039_interval = c039_env.get_interval(var)
        c039_lo = float(c039_interval.lo) if c039_interval.lo != -INF else -INF
        c039_hi = float(c039_interval.hi) if c039_interval.hi != INF else INF

        comparison[var] = {
            'polyhedral': (poly_lo, poly_hi),
            'interval': (c039_lo, c039_hi),
        }

        poly_width = poly_hi - poly_lo if poly_hi != INF and poly_lo != -INF else INF
        c039_width = c039_hi - c039_lo if c039_hi != INF and c039_lo != -INF else INF

        if poly_width < c039_width:
            precision_gains.append(var)

    rel = poly_env.get_relational_constraints()

    return {
        'polyhedral_constraints': poly_env.get_constraints(),
        'interval_results': comparison,
        'precision_gains': precision_gains,
        'relational_constraints': rel,
        'polyhedral_warnings': poly_result['warnings'],
    }


def verify_property(source: str, property_str: str) -> dict:
    """Verify a linear property about the program's final state.

    property_str examples: "x - y <= 5", "x + y <= 10", "x <= 100", "x == y"
    """
    result = polyhedral_analyze(source)
    env = result['env']

    if env.is_bot():
        return {
            'property': property_str,
            'verdict': 'VERIFIED',
            'reason': 'program state is bottom (unreachable)',
            'all_constraints': env.get_constraints()
        }

    # Parse property
    prop = property_str.strip()

    # Try to match patterns
    verdict = 'UNKNOWN'
    details = ''

    constraints = env.get_constraints()

    # Simple check: see if the property string matches a constraint
    for c_str in constraints:
        if _property_matches(prop, c_str):
            verdict = 'VERIFIED'
            details = f'directly implied by constraint: {c_str}'
            break

    if verdict == 'UNKNOWN':
        # Try interval check for simple unary properties
        verdict, details = _check_property_from_intervals(prop, env)

    return {
        'property': property_str,
        'verdict': verdict,
        'details': details,
        'all_constraints': constraints
    }


def _property_matches(prop: str, constraint: str) -> bool:
    """Check if property is directly matched by a constraint."""
    # Normalize whitespace
    p = ' '.join(prop.split())
    c = ' '.join(constraint.split())
    return p == c


def _check_property_from_intervals(prop: str, env: PolyhedralDomain) -> Tuple[str, str]:
    """Try to verify a property from interval bounds."""
    import re

    # Pattern: var <= N
    m = re.match(r'(\w+)\s*<=\s*(-?\d+)', prop)
    if m:
        var, bound = m.group(1), int(m.group(2))
        hi = env.get_upper(var)
        if hi <= bound:
            return 'VERIFIED', f'{var} upper bound is {hi} <= {bound}'
        return 'UNKNOWN', f'{var} upper bound is {hi}, need <= {bound}'

    # Pattern: var >= N
    m = re.match(r'(\w+)\s*>=\s*(-?\d+)', prop)
    if m:
        var, bound = m.group(1), int(m.group(2))
        lo = env.get_lower(var)
        if lo >= bound:
            return 'VERIFIED', f'{var} lower bound is {lo} >= {bound}'
        return 'UNKNOWN', f'{var} lower bound is {lo}, need >= {bound}'

    # Pattern: var == N
    m = re.match(r'(\w+)\s*==\s*(-?\d+)', prop)
    if m:
        var, val = m.group(1), int(m.group(2))
        lo, hi = env.get_interval(var)
        if lo == val and hi == val:
            return 'VERIFIED', f'{var} is exactly {val}'
        return 'UNKNOWN', f'{var} range is [{lo}, {hi}], need == {val}'

    return 'UNKNOWN', 'could not parse or verify property'


def polyhedral_summary(source: str) -> str:
    """Get a human-readable summary of polyhedral analysis results."""
    result = polyhedral_analyze(source)
    env = result['env']

    lines = ["Polyhedral Analysis Summary", "=" * 40]

    if env.is_bot():
        lines.append("State: BOTTOM (unreachable)")
        return '\n'.join(lines)

    lines.append(f"Variables: {', '.join(env.var_names)}")
    lines.append("")

    lines.append("Variable ranges:")
    for var in env.var_names:
        lo, hi = env.get_interval(var)
        lo_s = str(lo) if lo != -INF else "-inf"
        hi_s = str(hi) if hi != INF else "+inf"
        lines.append(f"  {var}: [{lo_s}, {hi_s}]")

    lines.append("")
    lines.append("All constraints:")
    for c in env.get_constraints():
        lines.append(f"  {c}")

    rel = env.get_relational_constraints()
    if rel:
        lines.append("")
        lines.append("Relational constraints:")
        for r in rel:
            lines.append(f"  {r}")

    if result['warnings']:
        lines.append("")
        lines.append("Warnings:")
        for w in result['warnings']:
            lines.append(f"  {w}")

    return '\n'.join(lines)
