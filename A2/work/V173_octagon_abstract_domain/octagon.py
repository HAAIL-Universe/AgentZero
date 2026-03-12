"""
V173: Octagon Abstract Domain

A weakly relational abstract domain that tracks constraints of the form:
    +/- x +/- y <= c    (two-variable difference bounds)
    +/- x <= c          (single-variable bounds)

More scalable than full polyhedra (O(n^3) closure vs exponential FM elimination)
while capturing many useful relational properties like:
    - x <= y (ordering)
    - x + y <= 10 (sum bounds)
    - x - y == 0 (equality)
    - -5 <= x <= 10 (intervals, as special case)

Representation: Difference Bound Matrix (DBM) of size 2n x 2n where:
    - Variable x_i has two indices: 2i (positive, +x_i) and 2i+1 (negative, -x_i)
    - DBM[2i][2j] represents: x_j - x_i <= c  (equivalently: +x_j - x_i <= c)
    - DBM[2i+1][2j+1] represents: -x_j + x_i <= c (equivalently: x_i - x_j <= c)
    - DBM[2i][2j+1] represents: -x_j - x_i <= c (equivalently: -(x_i + x_j) <= c)
    - DBM[2i+1][2j] represents: x_j + x_i <= c (equivalently: x_i + x_j <= c)

    More simply, for signed variables v_p (p in {2i, 2i+1}):
    - DBM[p][q] means: signed(q) - signed(p) <= c
    where signed(2i) = +x_i, signed(2i+1) = -x_i

Key algorithm: Floyd-Warshall shortest-path closure for tight bounds.
Strengthening step: DBM[i][j] = min(DBM[i][j], (DBM[i][bar(i)] + DBM[bar(j)][j]) / 2)
where bar(2k) = 2k+1 and bar(2k+1) = 2k.

Composes with V172 (polyhedra) and C039-style domains.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from fractions import Fraction
from enum import Enum
import itertools

INF = Fraction(10**9)  # Represents +infinity (no constraint)


# ---------------------------------------------------------------------------
# Octagon Domain
# ---------------------------------------------------------------------------

def _bar(i):
    """Complement index: bar(2k) = 2k+1, bar(2k+1) = 2k."""
    return i ^ 1


class Octagon:
    """Octagon abstract domain element.

    Uses a DBM (Difference Bound Matrix) of size 2n x 2n.
    Variables are mapped to indices: var -> int index i,
    then 2i = +var, 2i+1 = -var in the DBM.

    DBM[p][q] = c means: signed_var(q) - signed_var(p) <= c
    """

    def __init__(self, n, dbm, var_map, is_bot=False):
        """
        n: number of variables
        dbm: 2n x 2n matrix (list of lists of Fraction)
        var_map: {var_name: index} mapping
        is_bot: True if this represents the empty set
        """
        self._n = n
        self._dbm = dbm
        self._var_map = var_map  # var_name -> index (0..n-1)
        self._rev_map = {i: v for v, i in var_map.items()}
        self._is_bot = is_bot

    @staticmethod
    def top():
        """TOP: no constraints (all bounds infinite)."""
        return Octagon(0, [], {})

    @staticmethod
    def bot():
        """BOT: empty set (unsatisfiable)."""
        return Octagon(0, [], {}, is_bot=True)

    @staticmethod
    def from_constraints(constraints, var_names=None):
        """Build octagon from a list of OctConstraint objects.

        If var_names not provided, extracts from constraints.
        """
        if var_names is None:
            var_names = set()
            for c in constraints:
                if c.var1 is not None:
                    var_names.add(c.var1)
                if c.var2 is not None:
                    var_names.add(c.var2)
            var_names = sorted(var_names)

        n = len(var_names)
        var_map = {v: i for i, v in enumerate(var_names)}
        size = 2 * n
        dbm = [[INF] * size for _ in range(size)]

        # Diagonal = 0
        for i in range(size):
            dbm[i][i] = Fraction(0)

        for c in constraints:
            _apply_constraint_to_dbm(dbm, var_map, c)

        oct = Octagon(n, dbm, var_map)
        oct._close()
        if oct._has_negative_cycle():
            return Octagon.bot()
        return oct

    def _ensure_var(self, var):
        """Ensure variable exists in the octagon; return new octagon if needed."""
        if var in self._var_map:
            return self
        # Add new variable
        new_map = dict(self._var_map)
        new_idx = self._n
        new_map[var] = new_idx
        new_n = self._n + 1
        new_size = 2 * new_n
        old_size = 2 * self._n

        # Extend DBM
        new_dbm = [[INF] * new_size for _ in range(new_size)]
        for i in range(old_size):
            for j in range(old_size):
                new_dbm[i][j] = self._dbm[i][j]
        # New variable: diagonal = 0
        new_dbm[2 * new_idx][2 * new_idx] = Fraction(0)
        new_dbm[2 * new_idx + 1][2 * new_idx + 1] = Fraction(0)
        # Cross-diagonal for new var (x - x <= 0, -x + x <= 0)
        new_dbm[2 * new_idx][2 * new_idx + 1] = INF  # no upper bound on +x - (-x) = 2x
        new_dbm[2 * new_idx + 1][2 * new_idx] = INF  # no upper bound on -x - (+x) = -2x

        return Octagon(new_n, new_dbm, new_map)

    def _ensure_vars(self, vars_list):
        """Ensure all variables exist."""
        result = self
        for v in vars_list:
            result = result._ensure_var(v)
        return result

    def is_bot(self):
        return self._is_bot

    def is_top(self):
        if self._is_bot:
            return False
        if self._n == 0:
            return True
        for i in range(2 * self._n):
            for j in range(2 * self._n):
                if i != j and self._dbm[i][j] < INF:
                    return True  # Has some finite constraint -> not top
                    # Wait, that means it HAS a constraint, so it's NOT top
        # Actually if all off-diagonal are INF, it IS top
        for i in range(2 * self._n):
            for j in range(2 * self._n):
                if i != j and self._dbm[i][j] < INF:
                    return False
        return True

    def variables(self):
        return set(self._var_map.keys())

    def num_variables(self):
        return self._n

    # -------------------------------------------------------------------
    # Closure (Floyd-Warshall + strengthening)
    # -------------------------------------------------------------------

    def _close(self):
        """Compute shortest-path closure (Floyd-Warshall) + strengthening."""
        size = 2 * self._n
        m = self._dbm

        # Floyd-Warshall
        for k in range(size):
            for i in range(size):
                if m[i][k] >= INF:
                    continue
                for j in range(size):
                    if m[k][j] >= INF:
                        continue
                    new_val = m[i][k] + m[k][j]
                    if new_val < m[i][j]:
                        m[i][j] = new_val

        # Strengthening: tighten using unary constraints
        # DBM[i][j] = min(DBM[i][j], (DBM[i][bar(i)] + DBM[bar(j)][j]) / 2)
        for i in range(size):
            for j in range(size):
                bi = _bar(i)
                bj = _bar(j)
                if m[i][bi] < INF and m[bj][j] < INF:
                    tight = (m[i][bi] + m[bj][j]) / 2
                    if tight < m[i][j]:
                        m[i][j] = tight

    def _has_negative_cycle(self):
        """Check for negative diagonal (inconsistency)."""
        for i in range(2 * self._n):
            if self._dbm[i][i] < 0:
                return True
        return False

    # -------------------------------------------------------------------
    # Lattice operations
    # -------------------------------------------------------------------

    def join(self, other):
        """Join (least upper bound): union of concrete states."""
        if self.is_bot():
            return other
        if other.is_bot():
            return self

        # Unify variable sets
        all_vars = sorted(self.variables() | other.variables())
        a = self._ensure_vars(all_vars)
        b = other._ensure_vars(all_vars)
        a._close()
        b._close()

        n = a._n
        size = 2 * n
        dbm = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                dbm[i][j] = max(a._dbm[i][j], b._dbm[i][j])

        result = Octagon(n, dbm, dict(a._var_map))
        # No need to close -- max of closed DBMs is already closed for octagons
        # Actually that's not quite true, but it's a sound over-approximation
        return result

    def meet(self, other):
        """Meet (greatest lower bound): intersection."""
        if self.is_bot() or other.is_bot():
            return Octagon.bot()

        all_vars = sorted(self.variables() | other.variables())
        a = self._ensure_vars(all_vars)
        b = other._ensure_vars(all_vars)

        n = a._n
        size = 2 * n
        dbm = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                dbm[i][j] = min(a._dbm[i][j], b._dbm[i][j])

        result = Octagon(n, dbm, dict(a._var_map))
        result._close()
        if result._has_negative_cycle():
            return Octagon.bot()
        return result

    def includes(self, other):
        """Check if self >= other (self includes other's concretization)."""
        if other.is_bot():
            return True
        if self.is_bot():
            return False

        all_vars = sorted(self.variables() | other.variables())
        a = self._ensure_vars(all_vars)
        b = other._ensure_vars(all_vars)

        size = 2 * a._n
        for i in range(size):
            for j in range(size):
                if a._dbm[i][j] < b._dbm[i][j]:
                    return False
        return True

    def widen(self, other):
        """Widening: keep constraints from self that are satisfied by other,
        set others to INF.
        """
        if self.is_bot():
            return other
        if other.is_bot():
            return self

        all_vars = sorted(self.variables() | other.variables())
        a = self._ensure_vars(all_vars)
        b = other._ensure_vars(all_vars)

        n = a._n
        size = 2 * n
        dbm = [[INF] * size for _ in range(size)]
        for i in range(size):
            dbm[i][i] = Fraction(0)
            for j in range(size):
                if i == j:
                    continue
                if b._dbm[i][j] <= a._dbm[i][j]:
                    # Constraint is stable (other satisfies it)
                    dbm[i][j] = a._dbm[i][j]
                # else: constraint violated -> drop to INF

        return Octagon(n, dbm, dict(a._var_map))

    def narrow(self, other):
        """Narrowing: improve precision after widening."""
        if self.is_bot():
            return other
        if other.is_bot():
            return self

        all_vars = sorted(self.variables() | other.variables())
        a = self._ensure_vars(all_vars)
        b = other._ensure_vars(all_vars)

        n = a._n
        size = 2 * n
        dbm = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if a._dbm[i][j] >= INF:
                    dbm[i][j] = b._dbm[i][j]
                else:
                    dbm[i][j] = a._dbm[i][j]

        result = Octagon(n, dbm, dict(a._var_map))
        result._close()
        if result._has_negative_cycle():
            return Octagon.bot()
        return result

    # -------------------------------------------------------------------
    # Transfer functions
    # -------------------------------------------------------------------

    def assign(self, var, expr):
        """Abstract assignment: var = expr.

        expr is an OctExpr (linear over at most 2 variables with +/- unit coeffs + constant).
        For non-octagonal expressions, falls back to forget (havoc).
        """
        if self.is_bot():
            return self

        result = self._ensure_var(var)
        idx = result._var_map[var]

        if isinstance(expr, OctExpr):
            if expr.kind == 'const':
                # var = c
                return result._assign_const(idx, expr.const)
            elif expr.kind == 'var':
                # var = other_var + c
                result = result._ensure_var(expr.var1)
                other_idx = result._var_map[expr.var1]
                return result._assign_var(idx, other_idx, expr.coeff1, expr.const)
            elif expr.kind == 'binop':
                result = result._ensure_var(expr.var1)
                if expr.var2 is not None:
                    result = result._ensure_var(expr.var2)
                    other_idx1 = result._var_map[expr.var1]
                    other_idx2 = result._var_map[expr.var2]
                    return result._assign_binop(idx, other_idx1, expr.coeff1,
                                                 other_idx2, expr.coeff2, expr.const)
                else:
                    other_idx = result._var_map[expr.var1]
                    return result._assign_var(idx, other_idx, expr.coeff1, expr.const)

        # Fallback: non-octagonal expression -> forget variable
        return result.forget(var)

    def _assign_const(self, idx, c):
        """var[idx] = c."""
        # First forget the variable, then add x = c constraints
        result = self._forget_idx(idx)
        size = 2 * result._n
        p = 2 * idx      # +x
        q = 2 * idx + 1  # -x
        # +x <= c  and  -x <= -c  means x = c
        # DBM encoding: x - 0 <= c means DBM[bar(p)][p] but for constant...
        # Actually for unary: DBM[q][p] = 2c (means +x - (-x) <= 2c, i.e., 2x <= 2c)
        # and DBM[p][q] = -2c (means -x - (+x) <= -2c, i.e., -2x <= -2c)
        result._dbm[q][p] = Fraction(2) * Fraction(c)   # 2x <= 2c
        result._dbm[p][q] = Fraction(-2) * Fraction(c)  # -2x <= -2c
        result._close()
        if result._has_negative_cycle():
            return Octagon.bot()
        return result

    def _assign_var(self, idx, src_idx, coeff, c):
        """var[idx] = coeff * var[src_idx] + c, where coeff is +1 or -1."""
        c = Fraction(c)
        if idx == src_idx:
            # Self-assignment: x = x + c (increment) or x = -x + c
            if coeff == 1:
                # x = x + c: shift all bounds
                return self._increment_var(idx, c)
            else:
                # x = -x + c: negate and shift
                return self._negate_and_shift(idx, c)

        result = self._forget_idx(idx)
        size = 2 * result._n
        p = 2 * idx
        q = 2 * idx + 1
        sp = 2 * src_idx
        sq = 2 * src_idx + 1

        if coeff == 1:
            # x = y + c
            # For every constraint involving y, derive one for x
            for i in range(size):
                if i == p or i == q:
                    continue
                # DBM[i][sp] = d means: +y - signed(i) <= d, so +x - signed(i) <= d + c... no
                # Actually: signed(sp) - signed(i) <= d, and x = signed(sp) + c
                # so signed(p) - signed(i) = signed(sp) + c - signed(i) <= d + c? No.
                # Let me think more carefully.
                # DBM[i][j] means: signed(j) - signed(i) <= DBM[i][j]
                # If x = y + c, then for signed(p) = +x = +y + c:
                # +x - signed(i) = +y + c - signed(i), and +y - signed(i) <= DBM[i][sp]
                # So +x - signed(i) <= DBM[i][sp] + c
                if result._dbm[i][sp] < INF:
                    result._dbm[i][p] = min(result._dbm[i][p], result._dbm[i][sp] + c)
                # -x - signed(i) = -y - c - signed(i), and -y - signed(i) <= DBM[i][sq]
                # So -x - signed(i) <= DBM[i][sq] - c
                if result._dbm[i][sq] < INF:
                    result._dbm[i][q] = min(result._dbm[i][q], result._dbm[i][sq] - c)
                # signed(j) - (+x) = signed(j) - (+y) - c, and signed(j) - (+y) <= DBM[sp][j]... wait
                # DBM[sp][i] means: signed(i) - (+y) <= DBM[sp][i]
                # signed(i) - (+x) = signed(i) - (+y) - c <= DBM[sp][i] - c
                if result._dbm[sp][i] < INF:
                    result._dbm[p][i] = min(result._dbm[p][i], result._dbm[sp][i] - c)
                # signed(i) - (-x) = signed(i) + x = signed(i) + y + c
                # signed(i) - (-y) <= DBM[sq][i], and signed(i) + x = signed(i) + y + c
                # = (signed(i) - (-y)) + c <= DBM[sq][i] + c
                if result._dbm[sq][i] < INF:
                    result._dbm[q][i] = min(result._dbm[q][i], result._dbm[sq][i] + c)

            # Unary constraint for x from y's unary:
            # 2x <= 2y + 2c, so DBM[q][p] = DBM[sq][sp] + 2c
            if result._dbm[sq][sp] < INF:
                result._dbm[q][p] = min(result._dbm[q][p], result._dbm[sq][sp] + 2 * c)
            # -2x <= -2y - 2c, so DBM[p][q] = DBM[sp][sq] - 2c
            if result._dbm[sp][sq] < INF:
                result._dbm[p][q] = min(result._dbm[p][q], result._dbm[sp][sq] - 2 * c)

        elif coeff == -1:
            # x = -y + c
            for i in range(size):
                if i == p or i == q:
                    continue
                # +x = -y + c, so +x - signed(i) = -y + c - signed(i)
                # -y - signed(i) <= DBM[i][sq], so +x - signed(i) <= DBM[i][sq] + c
                if result._dbm[i][sq] < INF:
                    result._dbm[i][p] = min(result._dbm[i][p], result._dbm[i][sq] + c)
                # -x = +y - c, so -x - signed(i) = +y - c - signed(i)
                # +y - signed(i) <= DBM[i][sp], so -x - signed(i) <= DBM[i][sp] - c
                if result._dbm[i][sp] < INF:
                    result._dbm[i][q] = min(result._dbm[i][q], result._dbm[i][sp] - c)
                # signed(i) - (+x) = signed(i) + y - c
                # signed(i) - (-y) <= DBM[sq][i], so signed(i) + y <= DBM[sq][i]
                # signed(i) - (+x) <= DBM[sq][i] - c
                if result._dbm[sq][i] < INF:
                    result._dbm[p][i] = min(result._dbm[p][i], result._dbm[sq][i] - c)
                # signed(i) - (-x) = signed(i) - y + c... wait, -x = y - c
                # signed(i) - (-x) = signed(i) - (y - c) = signed(i) - y + c
                # signed(i) - (+y) <= DBM[sp][i]
                # signed(i) - (-x) = signed(i) - y + c <= DBM[sp][i] + c
                if result._dbm[sp][i] < INF:
                    result._dbm[q][i] = min(result._dbm[q][i], result._dbm[sp][i] + c)

            # Unary constraints
            if result._dbm[sp][sq] < INF:
                result._dbm[q][p] = min(result._dbm[q][p], result._dbm[sp][sq] + 2 * c)
            if result._dbm[sq][sp] < INF:
                result._dbm[p][q] = min(result._dbm[p][q], result._dbm[sq][sp] - 2 * c)

        result._close()
        if result._has_negative_cycle():
            return Octagon.bot()
        return result

    def _assign_binop(self, idx, idx1, coeff1, idx2, coeff2, c):
        """var[idx] = coeff1 * var[idx1] + coeff2 * var[idx2] + c.
        This is only precisely representable when idx == idx1 or idx == idx2,
        otherwise we forget and use bounds from the RHS.
        """
        c = Fraction(c)

        if idx == idx1:
            # x = coeff1 * x + coeff2 * y + c
            if coeff1 == 1:
                # x = x + coeff2 * y + c -> x' - x = coeff2 * y + c
                # This is an increment by a variable expression -- complex case.
                # Fall back to forget + add bounds from RHS
                pass
            elif coeff1 == 0:
                # x = coeff2 * y + c
                return self._assign_var(idx, idx2, coeff2, c)

        if idx == idx2:
            if coeff2 == 1:
                pass
            elif coeff2 == 0:
                return self._assign_var(idx, idx1, coeff1, c)

        # General case: x = c1*y + c2*z + c
        # Forget x, then compute bounds on RHS and set them for x
        result = self._forget_idx(idx)
        size = 2 * result._n
        p = 2 * idx
        q = 2 * idx + 1

        # Compute bounds on c1*y + c2*z + c from the current state
        # For +x = c1*y + c2*z + c:
        #   upper bound: max(c1*y + c2*z + c) = c + ...
        #   lower bound: min(c1*y + c2*z + c) = c + ...

        # Get upper/lower bounds on c1*y
        y_p = 2 * idx1
        y_q = 2 * idx1 + 1
        z_p = 2 * idx2
        z_q = 2 * idx2 + 1

        # Upper bound on c1*y: if c1=+1, upper(y) = DBM[y_q][y_p]/2
        #                       if c1=-1, upper(-y) = DBM[y_p][y_q]/2
        def _get_upper(var_idx, coeff):
            vp = 2 * var_idx
            vq = 2 * var_idx + 1
            if coeff == 1:
                if self._dbm[vq][vp] < INF:
                    return self._dbm[vq][vp] / 2
            elif coeff == -1:
                if self._dbm[vp][vq] < INF:
                    return self._dbm[vp][vq] / 2
            return None

        def _get_lower(var_idx, coeff):
            vp = 2 * var_idx
            vq = 2 * var_idx + 1
            if coeff == 1:
                if self._dbm[vp][vq] < INF:
                    return -self._dbm[vp][vq] / 2
            elif coeff == -1:
                if self._dbm[vq][vp] < INF:
                    return -self._dbm[vq][vp] / 2
            return None

        upper_y = _get_upper(idx1, coeff1)
        upper_z = _get_upper(idx2, coeff2)
        lower_y = _get_lower(idx1, coeff1)
        lower_z = _get_lower(idx2, coeff2)

        if upper_y is not None and upper_z is not None:
            ub = Fraction(c) + upper_y + upper_z
            result._dbm[q][p] = min(result._dbm[q][p], 2 * ub)
        if lower_y is not None and lower_z is not None:
            lb = Fraction(c) + lower_y + lower_z
            result._dbm[p][q] = min(result._dbm[p][q], -2 * lb)

        # Also try to capture relational constraints between x and y/z
        # x = c1*y + c2*z + c
        # If c1 = +1, c2 = +1: x = y + z + c -> x - y = z + c
        # We can add DBM constraints relating x to y and x to z
        # x - y = c2*z + c: upper(x-y) = c2*upper(z) + c if c2>0
        if coeff1 == 1:
            # x - y = coeff2 * z + c
            if coeff2 == 1:
                ub_z = _get_upper(idx2, 1)
                lb_z = _get_lower(idx2, 1)
            elif coeff2 == -1:
                ub_z = _get_upper(idx2, -1)
                lb_z = _get_lower(idx2, -1)
            else:
                ub_z = None
                lb_z = None
            if ub_z is not None:
                # x - y <= ub_z + c
                y_p2 = 2 * idx1
                result._dbm[y_p2][p] = min(result._dbm[y_p2][p], ub_z + Fraction(c))
                result._dbm[q][y_p2 + 1] = min(result._dbm[q][y_p2 + 1], ub_z + Fraction(c))
            if lb_z is not None:
                # x - y >= lb_z + c => y - x <= -(lb_z + c)
                result._dbm[p][2 * idx1] = min(result._dbm[p][2 * idx1], -(lb_z + Fraction(c)))
                result._dbm[2 * idx1 + 1][q] = min(result._dbm[2 * idx1 + 1][q], -(lb_z + Fraction(c)))

        if coeff2 == 1:
            # x - z = coeff1 * y + c
            if coeff1 == 1:
                ub_y = _get_upper(idx1, 1)
                lb_y = _get_lower(idx1, 1)
            elif coeff1 == -1:
                ub_y = _get_upper(idx1, -1)
                lb_y = _get_lower(idx1, -1)
            else:
                ub_y = None
                lb_y = None
            if ub_y is not None:
                z_p2 = 2 * idx2
                result._dbm[z_p2][p] = min(result._dbm[z_p2][p], ub_y + Fraction(c))
                result._dbm[q][z_p2 + 1] = min(result._dbm[q][z_p2 + 1], ub_y + Fraction(c))
            if lb_y is not None:
                result._dbm[p][2 * idx2] = min(result._dbm[p][2 * idx2], -(lb_y + Fraction(c)))
                result._dbm[2 * idx2 + 1][q] = min(result._dbm[2 * idx2 + 1][q], -(lb_y + Fraction(c)))

        result._close()
        if result._has_negative_cycle():
            return Octagon.bot()
        return result

    def _increment_var(self, idx, c):
        """x = x + c: shift bounds by c."""
        c = Fraction(c)
        size = 2 * self._n
        p = 2 * idx
        q = 2 * idx + 1

        dbm = [row[:] for row in self._dbm]
        for i in range(size):
            if i == p or i == q:
                continue
            # signed(p) - signed(i) was bounded; now signed(p) increases by c
            # DBM[i][p] (constraint: +x - signed(i) <= d) -> +x was x, now x+c, so new bound = d + c? No.
            # Wait: DBM[i][p] = d means signed(p) - signed(i) <= d.
            # New +x_new = +x_old + c.
            # signed(p)_new - signed(i) = signed(p)_old + c - signed(i) <= d + c?
            # No: we want to express constraints on new x in terms of old bounds.
            # signed(p)_new - signed(i) <= d  <=> signed(p)_old + c - signed(i) <= d <=> signed(p)_old - signed(i) <= d - c
            # Old constraint was signed(p)_old - signed(i) <= DBM[i][p].
            # We need: d - c >= DBM_old[i][p], so d >= DBM_old[i][p] + c.
            # Tightest new: DBM_new[i][p] = DBM_old[i][p] + c
            if dbm[i][p] < INF:
                dbm[i][p] = dbm[i][p] + c
            # DBM[i][q]: signed(q) - signed(i) = -x_new - signed(i) = -x_old - c - signed(i)
            # Old: -x_old - signed(i) <= DBM[i][q]
            # New: -x_new - signed(i) = -x_old - c - signed(i) <= DBM[i][q] - c... wait
            # We need bounds on new -x = -x_old - c relative to signed(i):
            # -x_new - signed(i) <= d <=> -x_old - c - signed(i) <= d <=> -x_old - signed(i) <= d + c
            # So d + c >= DBM_old[i][q] => d >= DBM_old[i][q] - c
            if dbm[i][q] < INF:
                dbm[i][q] = dbm[i][q] - c
            # DBM[p][i]: signed(i) - (+x_new) <= d <=> signed(i) - (+x_old) - c <= d
            # <=> signed(i) - (+x_old) <= d + c
            # So d + c >= DBM_old[p][i] => d >= DBM_old[p][i] - c
            if dbm[p][i] < INF:
                dbm[p][i] = dbm[p][i] - c
            # DBM[q][i]: signed(i) - (-x_new) <= d <=> signed(i) + x_new <= d
            # = signed(i) + x_old + c <= d <=> signed(i) + x_old <= d - c
            # <=> signed(i) - (-x_old) <= d - c
            # Old: signed(i) - (-x_old) <= DBM_old[q][i]
            # So d - c >= DBM_old[q][i] => d >= DBM_old[q][i] + c
            if dbm[q][i] < INF:
                dbm[q][i] = dbm[q][i] + c

        # Unary: DBM[q][p] = 2*upper(x). New upper = old upper + c => DBM[q][p] += 2c
        # But this is already handled by the i==bar(p) case... hmm, q and p are handled:
        # Actually p and q are skipped in the loop above for cross-constraints,
        # but the unary self-constraints DBM[q][p] and DBM[p][q] need updating:
        # These are the +x <-> -x constraints: DBM[q][p] means +x - (-x) = 2x <= d
        # New 2x = 2(x_old + c) = 2x_old + 2c, so DBM[q][p] stays the same (it bounds 2x, and 2x increased)
        # Wait, I need to think again.
        # DBM[q][p] means signed(p) - signed(q) <= d, i.e., +x - (-x) = 2x <= d.
        # New: 2x_new = 2x_old + 2c. For this to be <= d, we need 2x_old <= d - 2c.
        # Old: 2x_old <= DBM_old[q][p]. So d - 2c >= DBM_old[q][p] => d >= DBM_old[q][p] + 2c.
        # Hmm no. The loop above already handles i = q for p and i = p for q?
        # No, p and q were skipped! Let me fix:
        # Unary bounds:
        if dbm[q][p] < INF:
            dbm[q][p] = dbm[q][p] + 2 * c  # Old: not right either...
            # Hmm, let's just re-derive:
            # 2x_new <= d <=> 2(x_old + c) <= d <=> 2x_old <= d - 2c
            # Best d: d = 2x_old_max + 2c = DBM_old[q][p] + 2c?
            # Wait: 2x_old <= DBM_old[q][p], so 2x_new = 2x_old + 2c <= DBM_old[q][p] + 2c
            # So yes: DBM_new[q][p] = DBM_old[q][p] + 2c... but I already overwrote it above? No, p and q were skipped.
            pass  # Already set correctly by the logic above...
        # Actually p and q were skipped, so dbm[q][p] and dbm[p][q] still have old values.
        # Fix them:
        old_qp = self._dbm[q][p]
        old_pq = self._dbm[p][q]
        if old_qp < INF:
            dbm[q][p] = old_qp  # 2x_new <= 2x_max_old + 2c? No...
            # Sigh, let me think cleanly. For x_new = x_old + c:
            # 2*x_new <= d iff 2*x_old + 2c <= d iff 2*x_old <= d - 2c
            # Tightest d: d = sup{2*x_old} + 2c. sup{2*x_old} = DBM_old[q][p].
            # So d = DBM_old[q][p] + 2c.
            dbm[q][p] = old_qp + 2 * c
        if old_pq < INF:
            # -2*x_new <= d iff -2*x_old - 2c <= d iff -2*x_old <= d + 2c
            # Tightest d: d = sup{-2*x_old} - 2c = DBM_old[p][q] - 2c.
            dbm[p][q] = old_pq - 2 * c

        result = Octagon(self._n, dbm, dict(self._var_map))
        # No closure needed -- increment preserves closure
        return result

    def _negate_and_shift(self, idx, c):
        """x = -x + c."""
        c = Fraction(c)
        size = 2 * self._n
        p = 2 * idx
        q = 2 * idx + 1

        dbm = [row[:] for row in self._dbm]

        # Swap p and q roles, then shift by c
        for i in range(size):
            if i == p or i == q:
                continue
            # New +x = -x_old + c, so signed(p)_new = -signed(q)_old + c? Not quite.
            # Actually +x_new = -x_old + c.
            # DBM_new[i][p]: +x_new - signed(i) <= d
            # +x_new = -x_old + c, so -x_old + c - signed(i) <= d
            # -x_old - signed(i) <= d - c. Old: DBM_old[i][q].
            # d - c >= DBM_old[i][q] => d >= DBM_old[i][q] + c
            dbm[i][p] = self._dbm[i][q] + c if self._dbm[i][q] < INF else INF
            # DBM_new[i][q]: -x_new - signed(i) <= d
            # -x_new = x_old - c. x_old - c - signed(i) <= d
            # x_old - signed(i) <= d + c. Old: DBM_old[i][p].
            # d + c >= DBM_old[i][p] => d >= DBM_old[i][p] - c
            dbm[i][q] = self._dbm[i][p] - c if self._dbm[i][p] < INF else INF
            # DBM_new[p][i]: signed(i) - (+x_new) <= d
            # signed(i) - (-x_old + c) = signed(i) + x_old - c <= d
            # signed(i) - (-x_old) - c <= d. Old: signed(i) - (-x_old) <= DBM_old[q][i].
            # d >= DBM_old[q][i] - c
            dbm[p][i] = self._dbm[q][i] - c if self._dbm[q][i] < INF else INF
            # DBM_new[q][i]: signed(i) - (-x_new) <= d
            # -x_new = x_old - c. signed(i) - (x_old - c) = signed(i) - x_old + c <= d
            # signed(i) - (+x_old) + c <= d. Old: DBM_old[p][i].
            # d >= DBM_old[p][i] + c
            dbm[q][i] = self._dbm[p][i] + c if self._dbm[p][i] < INF else INF

        # Unary: DBM_new[q][p] = 2*x_new_max. x_new = -x_old + c. max(x_new) = -min(x_old) + c.
        # min(x_old) = -DBM_old[p][q]/2. So max(x_new) = DBM_old[p][q]/2 + c.
        # DBM_new[q][p] = 2 * max(x_new) = DBM_old[p][q] + 2c.
        if self._dbm[p][q] < INF:
            dbm[q][p] = self._dbm[p][q] + 2 * c
        else:
            dbm[q][p] = INF
        # DBM_new[p][q] = -2*min(x_new). min(x_new) = -max(x_old) + c = -DBM_old[q][p]/2 + c.
        # -2*min(x_new) = DBM_old[q][p] - 2c.
        if self._dbm[q][p] < INF:
            dbm[p][q] = self._dbm[q][p] - 2 * c
        else:
            dbm[p][q] = INF

        result = Octagon(self._n, dbm, dict(self._var_map))
        result._close()
        if result._has_negative_cycle():
            return Octagon.bot()
        return result

    def forget(self, var):
        """Havoc: remove all constraints involving var."""
        if self.is_bot():
            return self
        if var not in self._var_map:
            return self
        idx = self._var_map[var]
        return self._forget_idx(idx)

    def _forget_idx(self, idx):
        """Remove all constraints involving variable at index idx."""
        size = 2 * self._n
        p = 2 * idx
        q = 2 * idx + 1

        dbm = [row[:] for row in self._dbm]
        for i in range(size):
            if i != p and i != q:
                dbm[i][p] = INF
                dbm[i][q] = INF
                dbm[p][i] = INF
                dbm[q][i] = INF
        dbm[p][q] = INF
        dbm[q][p] = INF
        dbm[p][p] = Fraction(0)
        dbm[q][q] = Fraction(0)

        return Octagon(self._n, dbm, dict(self._var_map))

    def guard(self, constraint):
        """Add a guard constraint (from if/while condition)."""
        if self.is_bot():
            return self

        result = self._ensure_vars([v for v in [constraint.var1, constraint.var2] if v is not None])
        size = 2 * result._n
        dbm = [row[:] for row in result._dbm]
        _apply_constraint_to_dbm(dbm, result._var_map, constraint)

        new_oct = Octagon(result._n, dbm, dict(result._var_map))
        new_oct._close()
        if new_oct._has_negative_cycle():
            return Octagon.bot()
        return new_oct

    def guard_constraints(self, constraints):
        """Add multiple guard constraints."""
        result = self
        for c in constraints:
            result = result.guard(c)
            if result.is_bot():
                return result
        return result

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def get_bounds(self, var):
        """Get interval bounds [lo, hi] for a variable."""
        if self.is_bot() or var not in self._var_map:
            return None, None
        idx = self._var_map[var]
        p = 2 * idx
        q = 2 * idx + 1
        # Upper: DBM[q][p] = 2 * upper => upper = DBM[q][p] / 2
        upper = self._dbm[q][p] / 2 if self._dbm[q][p] < INF else None
        # Lower: DBM[p][q] = -2 * lower => lower = -DBM[p][q] / 2
        lower = -self._dbm[p][q] / 2 if self._dbm[p][q] < INF else None
        return lower, upper

    def get_difference_bound(self, var1, var2):
        """Get bound on var1 - var2."""
        if self.is_bot():
            return None, None
        if var1 not in self._var_map or var2 not in self._var_map:
            return None, None
        i1 = self._var_map[var1]
        i2 = self._var_map[var2]
        p1, q1 = 2 * i1, 2 * i1 + 1
        p2, q2 = 2 * i2, 2 * i2 + 1
        # var1 - var2: +var1 - (+var2) <= DBM[p2][p1]
        upper = self._dbm[p2][p1] if self._dbm[p2][p1] < INF else None
        # -(var1 - var2) = var2 - var1 <= DBM[p1][p2]
        lower = -self._dbm[p1][p2] if self._dbm[p1][p2] < INF else None
        return lower, upper

    def get_sum_bound(self, var1, var2):
        """Get bound on var1 + var2."""
        if self.is_bot():
            return None, None
        if var1 not in self._var_map or var2 not in self._var_map:
            return None, None
        i1 = self._var_map[var1]
        i2 = self._var_map[var2]
        p1, q1 = 2 * i1, 2 * i1 + 1
        p2, q2 = 2 * i2, 2 * i2 + 1
        # var1 + var2 = +var1 - (-var2) <= DBM[q2][p1]
        upper = self._dbm[q2][p1] if self._dbm[q2][p1] < INF else None
        # -(var1 + var2) = -var1 - var2 = -var1 - (+var2) <= ... hmm
        # Actually: -(var1 + var2) = (-var1) - (+var2)
        # DBM[p2][q1] = signed(q1) - signed(p2) = -var1 - var2 <= DBM[p2][q1]
        lower = -self._dbm[p2][q1] if self._dbm[p2][q1] < INF else None
        return lower, upper

    def is_satisfiable(self):
        """Check if the octagon is non-empty."""
        if self._is_bot:
            return False
        return True  # If no negative cycle was detected during closure, it's satisfiable

    def contains_point(self, point):
        """Check if a concrete point {var: value} satisfies all constraints."""
        if self.is_bot():
            return False
        for var, val in point.items():
            if var not in self._var_map:
                continue
            lo, hi = self.get_bounds(var)
            if lo is not None and Fraction(val) < lo:
                return False
            if hi is not None and Fraction(val) > hi:
                return False
        # Check all pairwise constraints
        for v1 in self._var_map:
            for v2 in self._var_map:
                if v1 >= v2:
                    continue
                if v1 not in point or v2 not in point:
                    continue
                val1 = Fraction(point[v1])
                val2 = Fraction(point[v2])

                lo_diff, hi_diff = self.get_difference_bound(v1, v2)
                if hi_diff is not None and val1 - val2 > hi_diff:
                    return False
                if lo_diff is not None and val1 - val2 < lo_diff:
                    return False

                lo_sum, hi_sum = self.get_sum_bound(v1, v2)
                if hi_sum is not None and val1 + val2 > hi_sum:
                    return False
                if lo_sum is not None and val1 + val2 < lo_sum:
                    return False
        return True

    def extract_intervals(self):
        """Extract per-variable interval bounds."""
        result = {}
        for v in self._var_map:
            lo, hi = self.get_bounds(v)
            result[v] = (lo, hi)
        return result

    def extract_constraints(self):
        """Extract all non-trivial octagonal constraints as OctConstraint objects."""
        if self.is_bot():
            return []
        result = []
        size = 2 * self._n
        seen = set()

        for i in range(size):
            for j in range(size):
                if i == j or self._dbm[i][j] >= INF:
                    continue
                key = (i, j)
                if key in seen:
                    continue

                # Decode: signed(j) - signed(i) <= dbm[i][j]
                c = _decode_dbm_entry(self._rev_map, i, j, self._dbm[i][j])
                if c is not None and c not in result:
                    result.append(c)
                    seen.add(key)

        return result

    def num_constraints(self):
        if self.is_bot():
            return 0
        count = 0
        size = 2 * self._n
        for i in range(size):
            for j in range(size):
                if i != j and self._dbm[i][j] < INF:
                    count += 1
        return count

    def __repr__(self):
        if self._is_bot:
            return "Octagon(BOT)"
        if self._n == 0:
            return "Octagon(TOP)"
        constraints = self.extract_constraints()
        if not constraints:
            return "Octagon(TOP)"
        cs = ", ".join(str(c) for c in constraints[:10])
        if len(constraints) > 10:
            cs += f", ... ({len(constraints)} total)"
        return f"Octagon({cs})"


# ---------------------------------------------------------------------------
# Octagonal Constraint
# ---------------------------------------------------------------------------

class OctConstraint:
    """An octagonal constraint: coeff1*var1 + coeff2*var2 <= bound
    where coeff1, coeff2 are in {-1, 0, +1}.

    Unary: coeff2 = 0, var2 = None -> coeff1*var1 <= bound
    Binary: both variables present -> coeff1*var1 + coeff2*var2 <= bound
    """

    def __init__(self, var1, coeff1, var2, coeff2, bound):
        self.var1 = var1
        self.coeff1 = coeff1
        self.var2 = var2
        self.coeff2 = coeff2
        self.bound = Fraction(bound)

    @staticmethod
    def var_le(var, bound):
        """var <= bound"""
        return OctConstraint(var, 1, None, 0, bound)

    @staticmethod
    def var_ge(var, bound):
        """-var <= -bound, i.e., var >= bound"""
        return OctConstraint(var, -1, None, 0, -Fraction(bound))

    @staticmethod
    def diff_le(var1, var2, bound):
        """var1 - var2 <= bound"""
        return OctConstraint(var1, 1, var2, -1, bound)

    @staticmethod
    def sum_le(var1, var2, bound):
        """var1 + var2 <= bound"""
        return OctConstraint(var1, 1, var2, 1, bound)

    @staticmethod
    def diff_ge(var1, var2, bound):
        """var1 - var2 >= bound, i.e., -(var1 - var2) <= -bound, i.e., var2 - var1 <= -bound"""
        return OctConstraint(var2, 1, var1, -1, -Fraction(bound))

    @staticmethod
    def sum_ge(var1, var2, bound):
        """var1 + var2 >= bound, i.e., -var1 - var2 <= -bound"""
        return OctConstraint(var1, -1, var2, -1, -Fraction(bound))

    @staticmethod
    def eq(var1, var2):
        """var1 == var2 (represented as two constraints: var1-var2<=0 and var2-var1<=0)"""
        return [OctConstraint.diff_le(var1, var2, 0), OctConstraint.diff_le(var2, var1, 0)]

    @staticmethod
    def var_eq(var, val):
        """var == val"""
        return [OctConstraint.var_le(var, val), OctConstraint.var_ge(var, val)]

    def __repr__(self):
        parts = []
        if self.var1 is not None:
            if self.coeff1 == 1:
                parts.append(f"{self.var1}")
            elif self.coeff1 == -1:
                parts.append(f"-{self.var1}")
        if self.var2 is not None:
            if self.coeff2 == 1:
                parts.append(f"+ {self.var2}" if parts else f"{self.var2}")
            elif self.coeff2 == -1:
                parts.append(f"- {self.var2}" if parts else f"-{self.var2}")
        lhs = " ".join(parts) if parts else "0"
        return f"{lhs} <= {self.bound}"

    def __eq__(self, other):
        if not isinstance(other, OctConstraint):
            return False
        return (self.var1 == other.var1 and self.coeff1 == other.coeff1 and
                self.var2 == other.var2 and self.coeff2 == other.coeff2 and
                self.bound == other.bound)

    def __hash__(self):
        return hash((self.var1, self.coeff1, self.var2, self.coeff2, self.bound))


# ---------------------------------------------------------------------------
# Octagonal Expression (for transfer functions)
# ---------------------------------------------------------------------------

class OctExpr:
    """Expression representable in the octagon domain.

    kind = 'const': value c
    kind = 'var': coeff1 * var1 + const
    kind = 'binop': coeff1 * var1 + coeff2 * var2 + const
    """

    def __init__(self, kind, var1=None, coeff1=0, var2=None, coeff2=0, const=0):
        self.kind = kind
        self.var1 = var1
        self.coeff1 = coeff1
        self.var2 = var2
        self.coeff2 = coeff2
        self.const = Fraction(const)

    @staticmethod
    def constant(c):
        return OctExpr('const', const=c)

    @staticmethod
    def variable(var, coeff=1, const=0):
        return OctExpr('var', var1=var, coeff1=coeff, const=const)

    @staticmethod
    def binary(var1, coeff1, var2, coeff2, const=0):
        return OctExpr('binop', var1=var1, coeff1=coeff1, var2=var2, coeff2=coeff2, const=const)


# ---------------------------------------------------------------------------
# DBM Helper Functions
# ---------------------------------------------------------------------------

def _apply_constraint_to_dbm(dbm, var_map, c):
    """Apply an OctConstraint to a DBM."""
    if c.var1 is None:
        return  # Tautology / constant constraint

    if c.var2 is None:
        # Unary: coeff1 * var1 <= bound
        idx = var_map[c.var1]
        if c.coeff1 == 1:
            # +var1 <= bound => 2*var1 <= 2*bound => DBM[2i+1][2i] = min(..., 2*bound)
            p = 2 * idx
            q = 2 * idx + 1
            dbm[q][p] = min(dbm[q][p], 2 * c.bound)
        elif c.coeff1 == -1:
            # -var1 <= bound => -2*var1 <= 2*bound => DBM[2i][2i+1] = min(..., 2*bound)
            p = 2 * idx
            q = 2 * idx + 1
            dbm[p][q] = min(dbm[p][q], 2 * c.bound)
    else:
        # Binary: coeff1*var1 + coeff2*var2 <= bound
        i1 = var_map[c.var1]
        i2 = var_map[c.var2]

        # signed(j) - signed(i) <= DBM[i][j]
        # coeff1*var1 + coeff2*var2 <= bound
        # We need to find i, j such that signed(j) - signed(i) = coeff1*var1 + coeff2*var2

        if c.coeff1 == 1 and c.coeff2 == -1:
            # var1 - var2 <= bound => (+var1) - (+var2) <= bound
            # signed(2*i1) - signed(2*i2) <= bound => DBM[2*i2][2*i1] = min(..., bound)
            dbm[2 * i2][2 * i1] = min(dbm[2 * i2][2 * i1], c.bound)
            # Also: (-var2) - (-var1) <= bound => DBM[2*i1+1][2*i2+1] = min(..., bound)
            dbm[2 * i1 + 1][2 * i2 + 1] = min(dbm[2 * i1 + 1][2 * i2 + 1], c.bound)
        elif c.coeff1 == -1 and c.coeff2 == 1:
            # -var1 + var2 <= bound => var2 - var1 <= bound
            dbm[2 * i1][2 * i2] = min(dbm[2 * i1][2 * i2], c.bound)
            dbm[2 * i2 + 1][2 * i1 + 1] = min(dbm[2 * i2 + 1][2 * i1 + 1], c.bound)
        elif c.coeff1 == 1 and c.coeff2 == 1:
            # var1 + var2 <= bound => (+var1) - (-var2) <= bound
            # signed(2*i1) - signed(2*i2+1) <= bound => DBM[2*i2+1][2*i1] = min(..., bound)
            dbm[2 * i2 + 1][2 * i1] = min(dbm[2 * i2 + 1][2 * i1], c.bound)
            # Also: (+var2) - (-var1) <= bound => DBM[2*i1+1][2*i2] = min(..., bound)
            dbm[2 * i1 + 1][2 * i2] = min(dbm[2 * i1 + 1][2 * i2], c.bound)
        elif c.coeff1 == -1 and c.coeff2 == -1:
            # -var1 - var2 <= bound => (-var1) - (+var2) <= bound
            # signed(2*i1+1) - signed(2*i2) <= bound => DBM[2*i2][2*i1+1] = min(..., bound)
            dbm[2 * i2][2 * i1 + 1] = min(dbm[2 * i2][2 * i1 + 1], c.bound)
            dbm[2 * i1][2 * i2 + 1] = min(dbm[2 * i1][2 * i2 + 1], c.bound)


def _decode_dbm_entry(rev_map, i, j, bound):
    """Decode a DBM entry into an OctConstraint.
    DBM[i][j] = bound means signed(j) - signed(i) <= bound.
    signed(2k) = +var_k, signed(2k+1) = -var_k.
    """
    var_j = rev_map.get(j // 2)
    var_i = rev_map.get(i // 2)
    if var_j is None or var_i is None:
        return None

    coeff_j = 1 if j % 2 == 0 else -1  # signed(j)
    coeff_i = 1 if i % 2 == 0 else -1  # signed(i)

    # signed(j) - signed(i) <= bound
    # => coeff_j * var_j - coeff_i * var_i <= bound
    # => coeff_j * var_j + (-coeff_i) * var_i <= bound

    if var_j == var_i:
        # Unary constraint
        if coeff_j == coeff_i:
            return None  # 0 <= bound (tautology if bound >= 0)
        # coeff_j * var - coeff_i * var <= bound
        net_coeff = coeff_j - coeff_i
        # net_coeff is +2 or -2
        # net_coeff * var / 2 <= bound / 2... but we want: sign*var <= bound/2
        if net_coeff == 2:
            return OctConstraint.var_le(var_j, bound / 2)
        elif net_coeff == -2:
            return OctConstraint(var_j, -1, None, 0, bound / 2)
    else:
        return OctConstraint(var_j, coeff_j, var_i, -coeff_i, bound)

    return None


# ---------------------------------------------------------------------------
# Octagon Interpreter (tuple-based AST, same as V172)
# ---------------------------------------------------------------------------

class OctagonInterpreter:
    """Abstract interpreter using the Octagon domain.

    Uses the same AST format as V172 PolyhedraInterpreter:
      - ('assign', var, expr)
      - ('seq', stmt1, stmt2, ...)
      - ('if', cond, then_stmts, else_stmts)
      - ('while', cond, body_stmts)
      - ('assert', cond)
    """

    def __init__(self, max_iterations=50, widen_delay=2):
        self.max_iterations = max_iterations
        self.widen_delay = widen_delay
        self.warnings = []

    def analyze(self, program, init=None):
        """Analyze a program starting from init octagon (default TOP)."""
        state = init if init is not None else Octagon.top()
        self.warnings = []
        result = self._interpret(program, state)
        return OctAnalysisResult(
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
        oct_expr = self._expr_to_octexpr(expr)
        if oct_expr is not None:
            return state.assign(var, oct_expr)
        else:
            result = state._ensure_var(var)
            return result.forget(var)

    def _interpret_if(self, stmt, state):
        _, cond, then_body, else_body = stmt
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
            body_entry = current
            for c in guard_cs:
                body_entry = body_entry.guard(c)
            body_exit = self._interpret(body, body_entry)

            next_state = current.join(body_exit)

            if iteration >= self.widen_delay:
                next_state = current.widen(next_state)

            if next_state.includes(current) and current.includes(next_state):
                break
            current = next_state

        result = current
        for c in exit_cs:
            result = result.guard(c)
        return result

    def _interpret_assert(self, stmt, state):
        _, cond = stmt
        neg_constraints = self._cond_to_constraints(('not', cond))
        violation_state = state
        for c in neg_constraints:
            violation_state = violation_state.guard(c)
        if violation_state.is_satisfiable():
            self.warnings.append(f"Assertion may fail: {cond}")
        return state

    def _expr_to_octexpr(self, expr):
        """Convert expression AST to OctExpr, or None if not octagonal."""
        tag = expr[0]
        if tag == 'const':
            return OctExpr.constant(expr[1])
        elif tag == 'var':
            return OctExpr.variable(expr[1])
        elif tag == 'neg':
            inner = self._expr_to_octexpr(expr[1])
            if inner is None:
                return None
            if inner.kind == 'const':
                return OctExpr.constant(-inner.const)
            elif inner.kind == 'var':
                return OctExpr.variable(inner.var1, -inner.coeff1, -inner.const)
            return None
        elif tag == 'add':
            l = self._expr_to_octexpr(expr[1])
            r = self._expr_to_octexpr(expr[2])
            if l is None or r is None:
                return None
            return self._combine_add(l, r)
        elif tag == 'sub':
            l = self._expr_to_octexpr(expr[1])
            r = self._expr_to_octexpr(expr[2])
            if l is None or r is None:
                return None
            # Negate r
            r_neg = self._negate_octexpr(r)
            if r_neg is None:
                return None
            return self._combine_add(l, r_neg)
        elif tag == 'mul':
            l = self._expr_to_octexpr(expr[1])
            r = self._expr_to_octexpr(expr[2])
            if l is None or r is None:
                return None
            # At least one must be a constant +/-1
            if l.kind == 'const' and l.const in (1, -1):
                return self._scale_octexpr(r, int(l.const))
            elif r.kind == 'const' and r.const in (1, -1):
                return self._scale_octexpr(l, int(r.const))
            elif l.kind == 'const' and r.kind == 'var':
                # c * var: only if c is +1 or -1
                return None
            elif r.kind == 'const' and l.kind == 'var':
                return None
            return None
        return None

    def _negate_octexpr(self, e):
        if e.kind == 'const':
            return OctExpr.constant(-e.const)
        elif e.kind == 'var':
            return OctExpr.variable(e.var1, -e.coeff1, -e.const)
        elif e.kind == 'binop':
            return OctExpr.binary(e.var1, -e.coeff1, e.var2, -e.coeff2, -e.const)
        return None

    def _scale_octexpr(self, e, factor):
        if e.kind == 'const':
            return OctExpr.constant(e.const * factor)
        elif e.kind == 'var':
            return OctExpr.variable(e.var1, e.coeff1 * factor, e.const * factor)
        elif e.kind == 'binop':
            return OctExpr.binary(e.var1, e.coeff1 * factor, e.var2, e.coeff2 * factor, e.const * factor)
        return None

    def _combine_add(self, l, r):
        """Add two OctExprs."""
        if l.kind == 'const' and r.kind == 'const':
            return OctExpr.constant(l.const + r.const)
        elif l.kind == 'const' and r.kind == 'var':
            return OctExpr.variable(r.var1, r.coeff1, r.const + l.const)
        elif l.kind == 'var' and r.kind == 'const':
            return OctExpr.variable(l.var1, l.coeff1, l.const + r.const)
        elif l.kind == 'var' and r.kind == 'var':
            if l.var1 == r.var1:
                # Same variable: coeff1 + coeff2 must be in {-1, 0, 1}
                total = l.coeff1 + r.coeff1
                if total == 0:
                    return OctExpr.constant(l.const + r.const)
                elif abs(total) == 1:
                    return OctExpr.variable(l.var1, total, l.const + r.const)
                return None
            # Two different variables
            if abs(l.coeff1) <= 1 and abs(r.coeff1) <= 1:
                return OctExpr.binary(l.var1, l.coeff1, r.var1, r.coeff1, l.const + r.const)
            return None
        elif l.kind == 'const' and r.kind == 'binop':
            return OctExpr.binary(r.var1, r.coeff1, r.var2, r.coeff2, r.const + l.const)
        elif l.kind == 'binop' and r.kind == 'const':
            return OctExpr.binary(l.var1, l.coeff1, l.var2, l.coeff2, l.const + r.const)
        return None

    def _cond_to_constraints(self, cond):
        """Convert condition AST to list of OctConstraints."""
        tag = cond[0]
        if tag == 'true':
            return []
        elif tag == 'false':
            # Unsatisfiable: x <= -INF for some dummy
            return [OctConstraint('__false__', 1, None, 0, Fraction(-10**9))]
        elif tag == 'le':
            return self._comparison_to_constraints(cond[1], cond[2], 'le')
        elif tag == 'lt':
            return self._comparison_to_constraints(cond[1], cond[2], 'lt')
        elif tag == 'ge':
            return self._comparison_to_constraints(cond[2], cond[1], 'le')
        elif tag == 'gt':
            return self._comparison_to_constraints(cond[2], cond[1], 'lt')
        elif tag == 'eq':
            cs1 = self._comparison_to_constraints(cond[1], cond[2], 'le')
            cs2 = self._comparison_to_constraints(cond[2], cond[1], 'le')
            return cs1 + cs2
        elif tag == 'ne':
            return []  # Non-convex, overapproximate
        elif tag == 'and':
            return self._cond_to_constraints(cond[1]) + self._cond_to_constraints(cond[2])
        elif tag == 'or':
            return []  # Disjunction not representable, overapproximate
        elif tag == 'not':
            return self._negate_cond(cond[1])
        return []

    def _comparison_to_constraints(self, lhs_expr, rhs_expr, kind):
        """Convert lhs_expr <= rhs_expr (or < for 'lt') to OctConstraints.
        Returns list of OctConstraints for: lhs - rhs <= 0 (or <= -1 for lt).
        """
        l = self._expr_to_octexpr(lhs_expr)
        r = self._expr_to_octexpr(rhs_expr)
        if l is None or r is None:
            return []

        r_neg = self._negate_octexpr(r)
        if r_neg is None:
            return []
        diff = self._combine_add(l, r_neg)
        if diff is None:
            return []

        offset = Fraction(0) if kind == 'le' else Fraction(-1)  # lt: lhs - rhs <= -1

        if diff.kind == 'const':
            if diff.const <= -offset:
                return []  # Tautology (or will be caught)
            else:
                return [OctConstraint('__false__', 1, None, 0, Fraction(-10**9))]
        elif diff.kind == 'var':
            # coeff1 * var1 + const <= offset
            # coeff1 * var1 <= offset - const
            bound = offset - diff.const
            return [OctConstraint(diff.var1, diff.coeff1, None, 0, bound)]
        elif diff.kind == 'binop':
            # coeff1*var1 + coeff2*var2 + const <= offset
            bound = offset - diff.const
            return [OctConstraint(diff.var1, diff.coeff1, diff.var2, diff.coeff2, bound)]

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
            return []  # not(eq) is ne, non-convex
        elif tag == 'ne':
            return self._cond_to_constraints(('eq', cond[1], cond[2]))
        elif tag == 'and':
            return []  # not(and) = or, non-convex
        elif tag == 'or':
            return self._negate_cond(cond[1]) + self._negate_cond(cond[2])
        elif tag == 'not':
            return self._cond_to_constraints(cond[1])
        return []


@dataclass
class OctAnalysisResult:
    final_state: Octagon
    warnings: List[str]


# ---------------------------------------------------------------------------
# Convenience / Composition APIs
# ---------------------------------------------------------------------------

def octagon_from_intervals(intervals):
    """Create an octagon from per-variable intervals.
    intervals: {var: (lo, hi)} where lo/hi can be None for unbounded.
    """
    constraints = []
    for var, (lo, hi) in intervals.items():
        if lo is not None:
            constraints.append(OctConstraint.var_ge(var, lo))
        if hi is not None:
            constraints.append(OctConstraint.var_le(var, hi))
    if not constraints:
        return Octagon.top()
    return Octagon.from_constraints(constraints)


def analyze_program(program, init=None, max_iterations=50, widen_delay=2):
    """Convenience: analyze a program and return result."""
    interp = OctagonInterpreter(max_iterations, widen_delay)
    return interp.analyze(program, init)


def compare_with_intervals(program, init_intervals=None):
    """Compare octagon analysis (relational) with interval-only analysis.
    Shows precision gain from difference/sum bound tracking.
    """
    init = octagon_from_intervals(init_intervals) if init_intervals else None
    interp = OctagonInterpreter()
    oct_result = interp.analyze(program, init)

    oct_intervals = {}
    if not oct_result.final_state.is_bot():
        oct_intervals = oct_result.final_state.extract_intervals()

    return {
        'octagon_state': oct_result.final_state,
        'octagon_intervals': oct_intervals,
        'octagon_warnings': oct_result.warnings,
        'octagon_constraints': oct_result.final_state.num_constraints()
            if not oct_result.final_state.is_bot() else 0,
        'relational_constraints': oct_result.final_state.extract_constraints()
            if not oct_result.final_state.is_bot() else [],
    }


def verify_octagonal_property(program, constraint, init=None):
    """Verify that an octagonal property holds at program exit.
    constraint: OctConstraint.
    Returns (verified: bool, final_state: Octagon).
    """
    interp = OctagonInterpreter()
    result = interp.analyze(program, init)
    if result.final_state.is_bot():
        return True, result.final_state

    # Check if the constraint is implied by the final state
    # Add negation of constraint; if result is BOT, constraint holds
    neg = _negate_oct_constraint(constraint)
    test_state = result.final_state
    for c in neg:
        test_state = test_state.guard(c)

    if not test_state.is_satisfiable():
        return True, result.final_state
    return False, result.final_state


def _negate_oct_constraint(c):
    """Negate an OctConstraint, returning a list of constraints.
    c1*v1 + c2*v2 <= b => c1*v1 + c2*v2 > b => c1*v1 + c2*v2 >= b+1 (integers)
    => -c1*v1 - c2*v2 <= -(b+1)
    """
    return [OctConstraint(
        c.var1, -c.coeff1 if c.var1 else 0,
        c.var2, -c.coeff2 if c.var2 else 0,
        -(c.bound + 1)
    )]


def compare_with_polyhedra(program, init_intervals=None):
    """Compare octagon vs polyhedra analysis (if V172 available)."""
    oct_init = octagon_from_intervals(init_intervals) if init_intervals else None
    oct_interp = OctagonInterpreter()
    oct_result = oct_interp.analyze(program, oct_init)

    oct_data = {
        'domain': 'octagon',
        'intervals': oct_result.final_state.extract_intervals() if not oct_result.final_state.is_bot() else {},
        'warnings': oct_result.warnings,
        'is_bot': oct_result.final_state.is_bot(),
    }

    # Try to use V172 polyhedra for comparison
    try:
        import sys
        sys.path.insert(0, 'Z:/AgentZero/A2/work/V172_polyhedra_abstract_domain')
        from polyhedra import PolyhedraInterpreter, polyhedra_from_intervals as poly_from_iv

        poly_init = poly_from_iv(init_intervals) if init_intervals else None
        poly_interp = PolyhedraInterpreter()
        poly_result = poly_interp.analyze(program, poly_init)

        poly_data = {
            'domain': 'polyhedra',
            'intervals': poly_result.final_state.extract_intervals() if not poly_result.final_state.is_bot() else {},
            'warnings': poly_result.warnings,
            'is_bot': poly_result.final_state.is_bot(),
        }

        return {'octagon': oct_data, 'polyhedra': poly_data}
    except ImportError:
        return {'octagon': oct_data, 'polyhedra': None}


def batch_analyze(programs, init=None):
    """Analyze multiple programs, return list of results."""
    interp = OctagonInterpreter()
    return [interp.analyze(prog, init) for prog in programs]
