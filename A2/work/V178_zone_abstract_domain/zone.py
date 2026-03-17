"""
V178: Zone Abstract Domain

A relational abstract domain that tracks difference constraints of the form:
    x - y <= c    (variable differences)
    x <= c        (upper bounds, via x - x0 <= c where x0 is a special zero variable)
    x >= c        (lower bounds, via x0 - x <= -c)

Simpler and faster than octagon (V173): uses n x n DBM instead of 2n x 2n.
Does NOT track sum constraints (x + y <= c) -- only differences.

Representation: Difference Bound Matrix (DBM) of size (n+1) x (n+1) where:
    - Index 0 is the "zero" variable (always 0)
    - DBM[i][j] = c means: x_j - x_i <= c
    - DBM[0][j] = c means: x_j <= c (upper bound)
    - DBM[i][0] = c means: -x_i <= c, i.e. x_i >= -c (lower bound)

Key algorithm: Floyd-Warshall shortest-path closure for tight bounds.
No strengthening step needed (unlike octagon).

Composes with V173 (octagon) and V172 (polyhedra) for domain comparison.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from fractions import Fraction
import copy

INF = Fraction(10**9)  # Represents +infinity (no constraint)
ZERO_VAR = "__z0"  # Internal name for the zero variable (index 0)


# ---------------------------------------------------------------------------
# Constraint Representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZoneConstraint:
    """A constraint of the form: var1 - var2 <= bound.

    Special cases:
    - var2 is None: var1 <= bound  (upper bound, encoded as var1 - z0 <= bound)
    - var1 is None: var2 >= -bound (lower bound, encoded as z0 - var2 <= bound, i.e. var2 >= -bound)
    - Both set: var1 - var2 <= bound (difference constraint)
    """
    var1: Optional[str]  # None means zero variable
    var2: Optional[str]  # None means zero variable
    bound: Fraction

    def __str__(self):
        b = self.bound
        if self.var1 is None and self.var2 is None:
            return f"0 <= {b}"
        elif self.var2 is None:
            return f"{self.var1} <= {b}"
        elif self.var1 is None:
            return f"{self.var2} >= {-b}"
        else:
            return f"{self.var1} - {self.var2} <= {b}"


def upper_bound(var, bound):
    """var <= bound."""
    return ZoneConstraint(var, None, Fraction(bound))


def lower_bound(var, bound):
    """var >= bound (encoded as z0 - var <= -bound)."""
    return ZoneConstraint(None, var, Fraction(-bound))


def diff_bound(var1, var2, bound):
    """var1 - var2 <= bound."""
    return ZoneConstraint(var1, var2, Fraction(bound))


def eq_constraint(var1, var2):
    """var1 == var2 (var1 - var2 <= 0 AND var2 - var1 <= 0)."""
    return [diff_bound(var1, var2, 0), diff_bound(var2, var1, 0)]


def var_eq_const(var, c):
    """var == c (var <= c AND var >= c)."""
    return [upper_bound(var, c), lower_bound(var, c)]


# ---------------------------------------------------------------------------
# Zone Domain
# ---------------------------------------------------------------------------

class Zone:
    """Zone abstract domain element.

    Uses a DBM of size (n+1) x (n+1) where index 0 is the zero variable.
    DBM[i][j] = c means x_j - x_i <= c.
    """

    def __init__(self, n, dbm, var_map, is_bot=False):
        """
        n: number of real variables (not counting zero var)
        dbm: (n+1) x (n+1) matrix (list of lists of Fraction)
        var_map: {var_name: index (1..n)} mapping
        is_bot: True if this represents the empty set
        """
        self._n = n
        self._dbm = dbm
        self._var_map = var_map  # var_name -> index (1..n)
        self._rev_map = {i: v for v, i in var_map.items()}
        self._is_bot = is_bot

    @staticmethod
    def top():
        """TOP: no constraints (all bounds infinite)."""
        dbm = [[Fraction(0) if i == j else INF for j in range(1)] for i in range(1)]
        return Zone(0, dbm, {})

    @staticmethod
    def bot():
        """BOT: empty set (unsatisfiable)."""
        return Zone(0, [[Fraction(0)]], {}, is_bot=True)

    @staticmethod
    def from_constraints(constraints, var_names=None):
        """Build zone from a list of ZoneConstraint objects."""
        if var_names is None:
            var_names = set()
            for c in constraints:
                if c.var1 is not None:
                    var_names.add(c.var1)
                if c.var2 is not None:
                    var_names.add(c.var2)
            var_names = sorted(var_names)

        n = len(var_names)
        var_map = {v: i + 1 for i, v in enumerate(var_names)}
        size = n + 1
        dbm = [[INF] * size for _ in range(size)]
        for i in range(size):
            dbm[i][i] = Fraction(0)

        for c in constraints:
            _apply_constraint(dbm, var_map, c)

        zone = Zone(n, dbm, var_map)
        zone._close()
        if zone._has_negative_cycle():
            return Zone.bot()
        return zone

    def _ensure_var(self, var):
        """Ensure variable exists; return (possibly new) zone."""
        if var in self._var_map:
            return self
        new_map = dict(self._var_map)
        new_idx = self._n + 1
        new_map[var] = new_idx
        new_n = self._n + 1
        new_size = new_n + 1
        old_size = self._n + 1

        new_dbm = [[INF] * new_size for _ in range(new_size)]
        for i in range(old_size):
            for j in range(old_size):
                new_dbm[i][j] = self._dbm[i][j]
        new_dbm[new_idx][new_idx] = Fraction(0)

        return Zone(new_n, new_dbm, new_map)

    def _ensure_vars(self, vars_list):
        result = self
        for v in vars_list:
            result = result._ensure_var(v)
        return result

    def _reindex(self, target_var_map):
        """Rebuild zone with a specific var_map ordering.

        Used to align two zones before componentwise operations.
        """
        if self._is_bot:
            return Zone.bot()
        n = len(target_var_map)
        size = n + 1
        new_dbm = [[INF] * size for _ in range(size)]
        for i in range(size):
            new_dbm[i][i] = Fraction(0)

        # Map old indices to new indices
        for i_old in range(self._n + 1):
            for j_old in range(self._n + 1):
                if self._dbm[i_old][j_old] >= INF:
                    continue
                # What variable is at old index i_old?
                var_i = self._rev_map.get(i_old)  # None = zero var
                var_j = self._rev_map.get(j_old)  # None = zero var
                # New index
                i_new = target_var_map[var_i] if var_i is not None else 0
                j_new = target_var_map[var_j] if var_j is not None else 0
                new_dbm[i_new][j_new] = min(new_dbm[i_new][j_new], self._dbm[i_old][j_old])

        return Zone(n, new_dbm, dict(target_var_map))

    def is_bot(self):
        """Check if this zone represents the empty set."""
        return self._is_bot

    def is_top(self):
        """Check if this zone has no constraints."""
        if self._is_bot:
            return False
        size = self._n + 1
        for i in range(size):
            for j in range(size):
                if i == j:
                    if self._dbm[i][j] != Fraction(0):
                        return False
                else:
                    if self._dbm[i][j] < INF:
                        return False
        return True

    def _close(self):
        """Floyd-Warshall shortest-path closure."""
        if self._is_bot:
            return
        size = self._n + 1
        for k in range(size):
            for i in range(size):
                if self._dbm[i][k] >= INF:
                    continue  # No edge i->k, skip
                for j in range(size):
                    if self._dbm[k][j] >= INF:
                        continue  # No edge k->j, skip
                    via_k = self._dbm[i][k] + self._dbm[k][j]
                    if via_k < self._dbm[i][j]:
                        self._dbm[i][j] = via_k

    def _has_negative_cycle(self):
        """Check diagonal for negative entries (inconsistency)."""
        if self._is_bot:
            return True
        size = self._n + 1
        for i in range(size):
            if self._dbm[i][i] < Fraction(0):
                return True
        return False

    # ---------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------

    def get_upper_bound(self, var):
        """Get upper bound of variable, or INF if unbounded."""
        if self._is_bot or var not in self._var_map:
            return None if self._is_bot else INF
        idx = self._var_map[var]
        return self._dbm[0][idx]

    def get_lower_bound(self, var):
        """Get lower bound of variable, or -INF if unbounded."""
        if self._is_bot or var not in self._var_map:
            return None if self._is_bot else -INF
        idx = self._var_map[var]
        return -self._dbm[idx][0]

    def get_interval(self, var):
        """Get [lower, upper] interval for variable."""
        lo = self.get_lower_bound(var)
        hi = self.get_upper_bound(var)
        return (lo, hi)

    def get_diff_bound(self, var1, var2):
        """Get bound on var1 - var2, or INF if unbounded."""
        if self._is_bot:
            return None
        if var1 not in self._var_map or var2 not in self._var_map:
            return INF
        i = self._var_map[var2]  # DBM[i][j] = x_j - x_i <= c
        j = self._var_map[var1]
        return self._dbm[i][j]

    def satisfies(self, constraint):
        """Check if this zone implies the constraint."""
        if self._is_bot:
            return True  # bot implies everything
        v1 = constraint.var1
        v2 = constraint.var2
        i = self._var_map.get(v2, 0) if v2 is not None else 0
        j = self._var_map.get(v1, 0) if v1 is not None else 0
        if (v1 is not None and v1 not in self._var_map) or \
           (v2 is not None and v2 not in self._var_map):
            return False
        return self._dbm[i][j] <= constraint.bound

    def extract_constraints(self):
        """Extract all non-trivial constraints as ZoneConstraint objects."""
        if self._is_bot:
            return []
        constraints = []
        size = self._n + 1
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                c = self._dbm[i][j]
                if c >= INF:
                    continue
                # DBM[i][j] = x_j - x_i <= c
                v_i = self._rev_map.get(i)  # None for zero var
                v_j = self._rev_map.get(j)  # None for zero var
                constraints.append(ZoneConstraint(v_j, v_i, c))
        return constraints

    def extract_equalities(self):
        """Find variable pairs where var1 - var2 == c (both bounds tight)."""
        if self._is_bot:
            return []
        equalities = []
        seen = set()
        size = self._n + 1
        for i in range(size):
            for j in range(i + 1, size):
                c_ij = self._dbm[i][j]  # x_j - x_i <= c_ij
                c_ji = self._dbm[j][i]  # x_i - x_j <= c_ji
                if c_ij < INF and c_ji < INF and c_ij + c_ji == Fraction(0):
                    v_i = self._rev_map.get(i)
                    v_j = self._rev_map.get(j)
                    key = (min(str(v_i), str(v_j)), max(str(v_i), str(v_j)))
                    if key not in seen:
                        seen.add(key)
                        equalities.append((v_j, v_i, c_ij))  # x_j - x_i == c_ij
        return equalities

    # ---------------------------------------------------------------------------
    # Lattice Operations
    # ---------------------------------------------------------------------------

    @staticmethod
    def _align(z1, z2):
        """Align two zones to use the same var_map. Returns (z1', z2') with same indexing."""
        all_vars = sorted(set(list(z1._var_map.keys()) + list(z2._var_map.keys())))
        unified_map = {v: i + 1 for i, v in enumerate(all_vars)}
        return z1._reindex(unified_map), z2._reindex(unified_map)

    def join(self, other):
        """Join (union over-approximation): componentwise max."""
        if self._is_bot:
            return other._copy()
        if other._is_bot:
            return self._copy()

        z1, z2 = Zone._align(self, other)

        size = z1._n + 1
        new_dbm = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                new_dbm[i][j] = max(z1._dbm[i][j], z2._dbm[i][j])

        result = Zone(z1._n, new_dbm, dict(z1._var_map))
        result._close()
        return result

    def meet(self, other):
        """Meet (intersection): componentwise min."""
        if self._is_bot or other._is_bot:
            return Zone.bot()

        z1, z2 = Zone._align(self, other)

        size = z1._n + 1
        new_dbm = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                new_dbm[i][j] = min(z1._dbm[i][j], z2._dbm[i][j])

        result = Zone(z1._n, new_dbm, dict(z1._var_map))
        result._close()
        if result._has_negative_cycle():
            return Zone.bot()
        return result

    def widen(self, other):
        """Widening: keep bounds from self that are stable; drop growing bounds to INF."""
        if self._is_bot:
            return other._copy()
        if other._is_bot:
            return self._copy()

        z1, z2 = Zone._align(self, other)

        size = z1._n + 1
        new_dbm = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if z2._dbm[i][j] <= z1._dbm[i][j]:
                    new_dbm[i][j] = z1._dbm[i][j]
                else:
                    new_dbm[i][j] = INF  # Bound increased -> widen to infinity

        result = Zone(z1._n, new_dbm, dict(z1._var_map))
        # Don't close after widening (preserves widened structure)
        return result

    def narrow(self, other):
        """Narrowing: refine INF bounds from self using finite bounds in other."""
        if self._is_bot:
            return Zone.bot()
        if other._is_bot:
            return self._copy()

        z1, z2 = Zone._align(self, other)

        size = z1._n + 1
        new_dbm = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if z1._dbm[i][j] >= INF:
                    new_dbm[i][j] = z2._dbm[i][j]
                else:
                    new_dbm[i][j] = z1._dbm[i][j]

        result = Zone(z1._n, new_dbm, dict(z1._var_map))
        result._close()
        if result._has_negative_cycle():
            return Zone.bot()
        return result

    def includes(self, other):
        """Check if self >= other (other is included in self)."""
        if other._is_bot:
            return True
        if self._is_bot:
            return False

        z1, z2 = Zone._align(self, other)

        size = z1._n + 1
        for i in range(size):
            for j in range(size):
                if z1._dbm[i][j] < z2._dbm[i][j]:
                    return False
        return True

    def equals(self, other):
        """Semantic equality."""
        return self.includes(other) and other.includes(self)

    # ---------------------------------------------------------------------------
    # Transfer Functions
    # ---------------------------------------------------------------------------

    def assign_const(self, var, c):
        """var := c."""
        if self._is_bot:
            return Zone.bot()
        z = self._ensure_var(var)
        return z._assign_value(var, c)

    def _assign_value(self, var, c):
        """Internal: set var = c, forgetting old constraints on var."""
        c = Fraction(c)
        idx = self._var_map[var]
        size = self._n + 1
        new_dbm = [row[:] for row in self._dbm]

        # Reset all constraints involving var
        for k in range(size):
            if k != idx:
                new_dbm[idx][k] = INF
                new_dbm[k][idx] = INF

        # Set var = c: var - z0 <= c AND z0 - var <= -c
        new_dbm[0][idx] = c
        new_dbm[idx][0] = -c

        result = Zone(self._n, new_dbm, dict(self._var_map))
        result._close()
        return result

    def assign_var(self, var, src):
        """var := src."""
        if self._is_bot:
            return Zone.bot()
        z = self._ensure_vars([var, src])
        return z._assign_from_var(var, src)

    def _assign_from_var(self, var, src):
        """Internal: var := src, preserving difference constraints through src."""
        idx_var = self._var_map[var]
        idx_src = self._var_map[src]
        size = self._n + 1
        new_dbm = [row[:] for row in self._dbm]

        # var := src means var - src = 0, var - k = src - k, k - var = k - src
        for k in range(size):
            if k != idx_var:
                new_dbm[idx_var][k] = self._dbm[idx_src][k]  # var - k <= src - k
                new_dbm[k][idx_var] = self._dbm[k][idx_src]  # k - var <= k - src

        # var - var = 0
        new_dbm[idx_var][idx_var] = Fraction(0)

        result = Zone(self._n, new_dbm, dict(self._var_map))
        result._close()
        return result

    def assign_var_plus_const(self, var, src, c):
        """var := src + c."""
        if self._is_bot:
            return Zone.bot()
        c = Fraction(c)
        z = self._ensure_vars([var, src])
        return z._assign_from_var_plus_c(var, src, c)

    def _assign_from_var_plus_c(self, var, src, c):
        """Internal: var := src + c."""
        idx_var = self._var_map[var]
        idx_src = self._var_map[src]
        size = self._n + 1
        new_dbm = [row[:] for row in self._dbm]

        # var = src + c means:
        # DBM[k][var] = var - k bound. var = src + c => var - k = (src - k) + c
        #   so DBM[k][var] = DBM[k][src] + c
        # DBM[var][k] = k - var bound. k - var = (k - src) - c
        #   so DBM[var][k] = DBM[src][k] - c
        for k in range(size):
            if k != idx_var:
                if self._dbm[k][idx_src] < INF:
                    new_dbm[k][idx_var] = self._dbm[k][idx_src] + c
                else:
                    new_dbm[k][idx_var] = INF
                if self._dbm[idx_src][k] < INF:
                    new_dbm[idx_var][k] = self._dbm[idx_src][k] - c
                else:
                    new_dbm[idx_var][k] = INF

        new_dbm[idx_var][idx_var] = Fraction(0)

        result = Zone(self._n, new_dbm, dict(self._var_map))
        result._close()
        return result

    def increment(self, var, c):
        """var := var + c."""
        return self.assign_var_plus_const(var, var, c)

    def forget(self, var):
        """Remove all constraints on var (project out)."""
        if self._is_bot:
            return Zone.bot()
        if var not in self._var_map:
            return self._copy()
        idx = self._var_map[var]
        size = self._n + 1
        new_dbm = [row[:] for row in self._dbm]

        for k in range(size):
            if k != idx:
                new_dbm[idx][k] = INF
                new_dbm[k][idx] = INF

        return Zone(self._n, new_dbm, dict(self._var_map))

    def guard(self, constraint):
        """Refine zone by adding a constraint (e.g., from a branch condition)."""
        if self._is_bot:
            return Zone.bot()
        z = self
        if constraint.var1 is not None:
            z = z._ensure_var(constraint.var1)
        if constraint.var2 is not None:
            z = z._ensure_var(constraint.var2)

        new_dbm = [row[:] for row in z._dbm]
        _apply_constraint(new_dbm, z._var_map, constraint)

        result = Zone(z._n, new_dbm, dict(z._var_map))
        result._close()
        if result._has_negative_cycle():
            return Zone.bot()
        return result

    def add_constraint(self, constraint):
        """Alias for guard."""
        return self.guard(constraint)

    # ---------------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------------

    def _copy(self):
        if self._is_bot:
            return Zone.bot()
        new_dbm = [row[:] for row in self._dbm]
        return Zone(self._n, new_dbm, dict(self._var_map))

    def var_count(self):
        return self._n

    def variables(self):
        return sorted(self._var_map.keys())

    def __repr__(self):
        if self._is_bot:
            return "Zone(BOT)"
        constraints = self.extract_constraints()
        if not constraints:
            return "Zone(TOP)"
        return f"Zone({', '.join(str(c) for c in constraints)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_constraint(dbm, var_map, c):
    """Apply a ZoneConstraint to a DBM."""
    i = var_map.get(c.var2, 0) if c.var2 is not None else 0  # source index
    j = var_map.get(c.var1, 0) if c.var1 is not None else 0  # target index
    b = Fraction(c.bound)
    if b < dbm[i][j]:
        dbm[i][j] = b


# ---------------------------------------------------------------------------
# Zone Interpreter (for C010-style programs)
# ---------------------------------------------------------------------------

class ZoneInterpreter:
    """Abstract interpreter using Zone domain over simple imperative AST.

    Supports: assignment (const, var, var+const), sequencing, if/else, while (with widening).
    Uses C010-style AST nodes (LetDecl, Assign, Block, IfStmt, WhileStmt, BinOp).
    """

    def __init__(self, widening_delay=3):
        self._widening_delay = widening_delay

    def analyze(self, stmts, init_zone=None):
        """Analyze a list of statements (or a Block) starting from init_zone."""
        if init_zone is None:
            init_zone = Zone.top()
        if hasattr(stmts, 'stmts'):
            stmts = stmts.stmts
        env = init_zone
        for stmt in stmts:
            env = self._interpret_stmt(env, stmt)
        return env

    def _interpret_stmt(self, env, stmt):
        if env.is_bot():
            return Zone.bot()
        cls = type(stmt).__name__

        if cls == 'LetDecl':
            return self._interpret_let(env, stmt)
        elif cls == 'Assign':
            return self._interpret_assign(env, stmt)
        elif cls == 'Block':
            return self._interpret_block(env, stmt)
        elif cls == 'IfStmt':
            return self._interpret_if(env, stmt)
        elif cls == 'WhileStmt':
            return self._interpret_while(env, stmt)
        else:
            return env  # Skip unknown statements

    def _interpret_let(self, env, stmt):
        var = stmt.name
        val = stmt.value
        return self._interpret_assignment(env, var, val)

    def _interpret_assign(self, env, stmt):
        var = stmt.name if hasattr(stmt, 'name') else str(stmt.target)
        val = stmt.value
        return self._interpret_assignment(env, var, val)

    def _interpret_assignment(self, env, var, val):
        cls = type(val).__name__

        if cls == 'NumberLit':
            return env.assign_const(var, val.value)
        elif cls == 'Identifier':
            return env.assign_var(var, val.name)
        elif cls == 'BinOp':
            return self._interpret_binop_assign(env, var, val)
        else:
            # Unknown RHS -> forget variable
            return env._ensure_var(var).forget(var)

    def _interpret_binop_assign(self, env, var, binop):
        left_cls = type(binop.left).__name__
        right_cls = type(binop.right).__name__
        op = binop.op

        # var := x + c or var := x - c
        if left_cls == 'Identifier' and right_cls == 'NumberLit':
            src = binop.left.name
            c = binop.right.value
            if op == '+':
                return env.assign_var_plus_const(var, src, c)
            elif op == '-':
                return env.assign_var_plus_const(var, src, -c)

        # var := c + x
        if left_cls == 'NumberLit' and right_cls == 'Identifier':
            if op == '+':
                src = binop.right.name
                c = binop.left.value
                return env.assign_var_plus_const(var, src, c)

        # var := x + y or x - y (non-constant -> forget)
        if left_cls == 'Identifier' and right_cls == 'Identifier':
            if op == '-':
                # var = x - y: we can encode this if var is fresh or if we expand
                # For simplicity, use forget + constrain
                src1 = binop.left.name
                src2 = binop.right.name
                env2 = env._ensure_var(var).forget(var)
                # var = x - y => var - x <= -y, var + y <= x
                # Actually: var - x = -y => var - x <= -y AND x - var <= y
                # But y is a variable not a constant. We need:
                # var = x - y => var - x + y = 0 => var + y - x = 0
                # In zone domain: var - x <= diff(x,y) where diff is dynamic
                # This exceeds zone's expressiveness for 3-variable relations.
                # Best effort: use known bounds on x and y.
                ub_x = env.get_upper_bound(src1)
                lb_x = env.get_lower_bound(src1)
                ub_y = env.get_upper_bound(src2)
                lb_y = env.get_lower_bound(src2)
                # var = x - y, so var in [lb_x - ub_y, ub_x - lb_y]
                if ub_x < INF and lb_y is not None and lb_y > -INF:
                    env2 = env2.guard(upper_bound(var, ub_x - lb_y))
                if lb_x is not None and lb_x > -INF and ub_y < INF:
                    env2 = env2.guard(lower_bound(var, lb_x - ub_y))
                return env2

        # Fallback: forget var
        return env._ensure_var(var).forget(var)

    def _interpret_block(self, env, block):
        stmts = block.stmts if hasattr(block, 'stmts') else block
        for s in stmts:
            env = self._interpret_stmt(env, s)
        return env

    def _interpret_if(self, env, stmt):
        cond = stmt.cond
        then_body = stmt.then_body
        else_body = getattr(stmt, 'else_body', None)

        then_guard = self._condition_to_constraint(cond, negate=False)
        else_guard = self._condition_to_constraint(cond, negate=True)

        then_env = env
        for g in then_guard:
            then_env = then_env.guard(g)
        then_result = self._interpret_body(then_env, then_body)

        else_env = env
        for g in else_guard:
            else_env = else_env.guard(g)
        if else_body:
            else_result = self._interpret_body(else_env, else_body)
        else:
            else_result = else_env

        return then_result.join(else_result)

    def _interpret_while(self, env, stmt):
        cond = stmt.cond
        body = stmt.body
        guards = self._condition_to_constraint(cond, negate=False)
        exit_guards = self._condition_to_constraint(cond, negate=True)

        current = env
        for iteration in range(100):
            # Enter loop body
            body_entry = current
            for g in guards:
                body_entry = body_entry.guard(g)
            body_exit = self._interpret_body(body_entry, body)

            # Join with pre-loop state
            next_state = current.join(body_exit)

            # Apply widening after delay
            if iteration >= self._widening_delay:
                next_state = current.widen(next_state)

            if current.includes(next_state):
                break
            current = next_state

        # Apply exit condition
        result = current
        for g in exit_guards:
            result = result.guard(g)
        return result

    def _interpret_body(self, env, body):
        if hasattr(body, 'stmts'):
            return self._interpret_block(env, body)
        elif isinstance(body, list):
            for s in body:
                env = self._interpret_stmt(env, s)
            return env
        else:
            return self._interpret_stmt(env, body)

    def _condition_to_constraint(self, cond, negate=False):
        """Convert AST condition to zone constraints.

        Returns a list of ZoneConstraint (conjunction).
        """
        cls = type(cond).__name__

        if cls == 'BinOp':
            op = cond.op
            left = cond.left
            right = cond.right
            left_cls = type(left).__name__
            right_cls = type(right).__name__

            constraints = self._binop_to_constraints(op, left, left_cls, right, right_cls, negate)
            if constraints is not None:
                return constraints

        return []  # Cannot express this condition

    def _binop_to_constraints(self, op, left, left_cls, right, right_cls, negate):
        """Convert a comparison BinOp to zone constraints."""
        # Negate the operator if needed
        if negate:
            neg_map = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '=='}
            op = neg_map.get(op, op)

        # x < c  =>  x - z0 <= c - 1  (integer domain)
        # x <= c =>  x - z0 <= c
        # x > c  =>  z0 - x <= -(c+1) (i.e., x >= c+1)
        # x >= c =>  z0 - x <= -c     (i.e., x >= c)
        # x == c =>  x <= c AND x >= c
        # x < y  =>  x - y <= -1
        # x <= y =>  x - y <= 0
        # x > y  =>  y - x <= -1
        # x >= y =>  y - x <= 0
        # x == y =>  x - y <= 0 AND y - x <= 0

        if left_cls == 'Identifier' and right_cls == 'NumberLit':
            var = left.name
            c = Fraction(right.value)
            return self._var_cmp_const(op, var, c)

        if left_cls == 'NumberLit' and right_cls == 'Identifier':
            var = right.name
            c = Fraction(left.value)
            # c op var => var reverse_op c
            rev = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==', '!=': '!='}
            return self._var_cmp_const(rev.get(op, op), var, c)

        if left_cls == 'Identifier' and right_cls == 'Identifier':
            v1 = left.name
            v2 = right.name
            return self._var_cmp_var(op, v1, v2)

        # expr - expr patterns: BinOp(-, Identifier, Identifier) cmp NumberLit
        if left_cls == 'BinOp' and right_cls == 'NumberLit':
            inner = left
            if hasattr(inner, 'op') and inner.op == '-':
                inner_l = type(inner.left).__name__
                inner_r = type(inner.right).__name__
                if inner_l == 'Identifier' and inner_r == 'Identifier':
                    v1 = inner.left.name
                    v2 = inner.right.name
                    c = Fraction(right.value)
                    # (v1 - v2) op c
                    return self._diff_cmp_const(op, v1, v2, c)

        return None

    def _var_cmp_const(self, op, var, c):
        if op == '<':
            return [upper_bound(var, c - 1)]
        elif op == '<=':
            return [upper_bound(var, c)]
        elif op == '>':
            return [lower_bound(var, c + 1)]
        elif op == '>=':
            return [lower_bound(var, c)]
        elif op == '==':
            return [upper_bound(var, c), lower_bound(var, c)]
        elif op == '!=':
            return []  # Can't express disequality in zones
        return None

    def _var_cmp_var(self, op, v1, v2):
        if op == '<':
            return [diff_bound(v1, v2, -1)]
        elif op == '<=':
            return [diff_bound(v1, v2, 0)]
        elif op == '>':
            return [diff_bound(v2, v1, -1)]
        elif op == '>=':
            return [diff_bound(v2, v1, 0)]
        elif op == '==':
            return [diff_bound(v1, v2, 0), diff_bound(v2, v1, 0)]
        elif op == '!=':
            return []
        return None

    def _diff_cmp_const(self, op, v1, v2, c):
        """(v1 - v2) op c."""
        if op == '<':
            return [diff_bound(v1, v2, c - 1)]
        elif op == '<=':
            return [diff_bound(v1, v2, c)]
        elif op == '>':
            return [diff_bound(v2, v1, -(c + 1))]
        elif op == '>=':
            return [diff_bound(v2, v1, -c)]
        elif op == '==':
            return [diff_bound(v1, v2, c), diff_bound(v2, v1, -c)]
        return None


# ---------------------------------------------------------------------------
# Composition APIs
# ---------------------------------------------------------------------------

def zone_from_intervals(intervals):
    """Create a Zone from a dict of {var: (lower, upper)} intervals.

    This is how Zone compares to plain interval analysis:
    intervals give per-variable bounds, zone can also derive difference bounds.
    """
    constraints = []
    for var, (lo, hi) in intervals.items():
        if lo is not None:
            constraints.append(lower_bound(var, lo))
        if hi is not None:
            constraints.append(upper_bound(var, hi))
    return Zone.from_constraints(constraints)


def compare_with_octagon(zone, octagon_constraints):
    """Compare zone precision with octagon on the same program.

    Returns dict with comparison results.
    """
    zone_cs = zone.extract_constraints()
    zone_set = set()
    for c in zone_cs:
        zone_set.add((c.var1, c.var2, c.bound))

    oct_set = set()
    for c in octagon_constraints:
        oct_set.add((c.var1, c.var2, c.bound))

    return {
        "zone_constraints": len(zone_set),
        "octagon_constraints": len(oct_set),
        "zone_only": zone_set - oct_set,
        "octagon_only": oct_set - zone_set,
        "shared": zone_set & oct_set,
    }


def verify_zone_property(zone, prop_str):
    """Verify a simple property string against a zone.

    Supported: "x <= 5", "x >= 0", "x - y <= 3", "x == y", "x - y == 0".
    Returns True if zone implies the property.
    """
    prop_str = prop_str.strip()

    # Parse: var - var op const
    import re
    m = re.match(r'(\w+)\s*-\s*(\w+)\s*(<=|>=|==|<|>)\s*(-?\d+)', prop_str)
    if m:
        v1, v2, op, c = m.group(1), m.group(2), m.group(3), Fraction(m.group(4))
        if op == '==' or op == '<=':
            return zone.satisfies(diff_bound(v1, v2, c))
        elif op == '>=':
            return zone.satisfies(diff_bound(v2, v1, -c))
        elif op == '<':
            return zone.satisfies(diff_bound(v1, v2, c - 1))
        elif op == '>':
            return zone.satisfies(diff_bound(v2, v1, -(c + 1)))

    # Parse: var op const
    m = re.match(r'(\w+)\s*(<=|>=|==|<|>)\s*(-?\d+)', prop_str)
    if m:
        var, op, c = m.group(1), m.group(2), Fraction(m.group(3))
        if op == '<=':
            return zone.satisfies(upper_bound(var, c))
        elif op == '>=':
            return zone.satisfies(lower_bound(var, c))
        elif op == '==':
            return zone.satisfies(upper_bound(var, c)) and zone.satisfies(lower_bound(var, c))
        elif op == '<':
            return zone.satisfies(upper_bound(var, c - 1))
        elif op == '>':
            return zone.satisfies(lower_bound(var, c + 1))

    # Parse: var == var
    m = re.match(r'(\w+)\s*==\s*(\w+)', prop_str)
    if m:
        v1, v2 = m.group(1), m.group(2)
        return zone.satisfies(diff_bound(v1, v2, 0)) and zone.satisfies(diff_bound(v2, v1, 0))

    return False
