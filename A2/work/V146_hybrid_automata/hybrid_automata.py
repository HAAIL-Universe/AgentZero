"""
V146: Hybrid Automata Verification

Extends V118 (timed automata) with continuous dynamics -- variables evolve
according to differential equations within locations (modes).

Components:
1. Continuous variables with per-location flow dynamics
2. Hybrid automaton definition (modes, flows, guards, resets, invariants)
3. Rectangular automata (rate intervals per variable per mode)
4. Zone-based reachability for rectangular automata
5. Safety and reachability checking with counterexample traces
6. Simulation (concrete trajectory computation)
7. Example systems (thermostat, water tank, railroad crossing)
8. Composition (parallel product of hybrid automata)

Key insight: For rectangular automata (flow rate in [a,b] per variable per mode),
the zone (DBM) approach from V118 extends naturally. We use a "time-scaling" trick:
variable x with flow dx/dt in [a,b] is equivalent to clock-like constraints where
x advances at rate a to b instead of exactly 1.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from enum import Enum
from collections import deque
import math

INF = float('inf')


# --- Linear Constraints ---

class CompOp(Enum):
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    EQ = '=='
    NE = '!='


@dataclass(frozen=True)
class LinearExpr:
    """Linear expression: c0 + c1*x1 + c2*x2 + ..."""
    coeffs: Tuple[Tuple[str, float], ...]  # (var_name, coefficient) pairs
    constant: float

    def evaluate(self, valuation: Dict[str, float]) -> float:
        result = self.constant
        for var, coeff in self.coeffs:
            result += coeff * valuation.get(var, 0.0)
        return result

    def variables(self) -> Set[str]:
        return {v for v, _ in self.coeffs if _ != 0}

    def __str__(self):
        parts = []
        if self.constant != 0 or not self.coeffs:
            parts.append(str(self.constant))
        for var, coeff in self.coeffs:
            if coeff == 0:
                continue
            if coeff == 1:
                parts.append(var)
            elif coeff == -1:
                parts.append(f"-{var}")
            else:
                parts.append(f"{coeff}*{var}")
        return " + ".join(parts) if parts else "0"


def var_expr(name: str) -> LinearExpr:
    return LinearExpr(((name, 1.0),), 0.0)


def const_expr(val: float) -> LinearExpr:
    return LinearExpr((), val)


def linear_expr(coeffs: Dict[str, float], constant: float = 0.0) -> LinearExpr:
    return LinearExpr(tuple(sorted(coeffs.items())), constant)


@dataclass(frozen=True)
class LinearConstraint:
    """lhs op rhs where lhs and rhs are linear expressions.
    Normalized to: sum(coeffs * vars) op constant."""
    coeffs: Tuple[Tuple[str, float], ...]
    op: CompOp
    constant: float

    def evaluate(self, valuation: Dict[str, float]) -> bool:
        lhs = sum(c * valuation.get(v, 0.0) for v, c in self.coeffs)
        if self.op == CompOp.LT:
            return lhs < self.constant
        elif self.op == CompOp.LE:
            return lhs <= self.constant
        elif self.op == CompOp.GT:
            return lhs > self.constant
        elif self.op == CompOp.GE:
            return lhs >= self.constant
        elif self.op == CompOp.EQ:
            return abs(lhs - self.constant) < 1e-9
        elif self.op == CompOp.NE:
            return abs(lhs - self.constant) >= 1e-9
        return False

    def variables(self) -> Set[str]:
        return {v for v, c in self.coeffs if c != 0}

    def __str__(self):
        parts = []
        for var, coeff in self.coeffs:
            if coeff == 0:
                continue
            if coeff == 1:
                parts.append(var)
            elif coeff == -1:
                parts.append(f"-{var}")
            else:
                parts.append(f"{coeff}*{var}")
        lhs = " + ".join(parts) if parts else "0"
        return f"{lhs} {self.op.value} {self.constant}"


def var_leq(var: str, val: float) -> LinearConstraint:
    return LinearConstraint(((var, 1.0),), CompOp.LE, val)

def var_geq(var: str, val: float) -> LinearConstraint:
    return LinearConstraint(((var, 1.0),), CompOp.GE, val)

def var_lt(var: str, val: float) -> LinearConstraint:
    return LinearConstraint(((var, 1.0),), CompOp.LT, val)

def var_gt(var: str, val: float) -> LinearConstraint:
    return LinearConstraint(((var, 1.0),), CompOp.GT, val)

def var_eq(var: str, val: float) -> LinearConstraint:
    return LinearConstraint(((var, 1.0),), CompOp.EQ, val)

def diff_leq(v1: str, v2: str, val: float) -> LinearConstraint:
    return LinearConstraint(((v1, 1.0), (v2, -1.0)), CompOp.LE, val)

def diff_geq(v1: str, v2: str, val: float) -> LinearConstraint:
    return LinearConstraint(((v1, 1.0), (v2, -1.0)), CompOp.GE, val)


@dataclass(frozen=True)
class Predicate:
    """Conjunction of linear constraints."""
    constraints: Tuple[LinearConstraint, ...]

    def evaluate(self, valuation: Dict[str, float]) -> bool:
        return all(c.evaluate(valuation) for c in self.constraints)

    def is_true(self) -> bool:
        return len(self.constraints) == 0

    def variables(self) -> Set[str]:
        s = set()
        for c in self.constraints:
            s.update(c.variables())
        return s

    def __str__(self):
        if not self.constraints:
            return "true"
        return " && ".join(str(c) for c in self.constraints)


def true_pred() -> Predicate:
    return Predicate(())

def pred_and(p1: Predicate, p2: Predicate) -> Predicate:
    return Predicate(p1.constraints + p2.constraints)

def pred_from(*constraints: LinearConstraint) -> Predicate:
    return Predicate(tuple(constraints))


# --- Flow (Continuous Dynamics) ---

@dataclass(frozen=True)
class FlowInterval:
    """Rate interval for a variable: dx/dt in [lo, hi]."""
    lo: float
    hi: float

    def is_clock(self) -> bool:
        """Rate exactly 1 (standard clock)."""
        return self.lo == 1.0 and self.hi == 1.0

    def is_stopped(self) -> bool:
        """Rate exactly 0."""
        return self.lo == 0.0 and self.hi == 0.0

    def is_exact(self) -> bool:
        return self.lo == self.hi

    def __str__(self):
        if self.is_clock():
            return "1"
        if self.is_stopped():
            return "0"
        if self.is_exact():
            return str(self.lo)
        return f"[{self.lo}, {self.hi}]"


CLOCK_RATE = FlowInterval(1.0, 1.0)
STOPPED = FlowInterval(0.0, 0.0)


@dataclass(frozen=True)
class Flow:
    """Continuous dynamics for a mode: maps variable -> rate interval."""
    rates: Tuple[Tuple[str, FlowInterval], ...]

    def get_rate(self, var: str) -> FlowInterval:
        for v, r in self.rates:
            if v == var:
                return r
        return STOPPED  # Default: variable doesn't change

    def __str__(self):
        if not self.rates:
            return "{}"
        parts = [f"d{v}/dt in {r}" for v, r in self.rates]
        return "{" + ", ".join(parts) + "}"


def flow_from(**rates: Tuple[float, float]) -> Flow:
    """Create flow from keyword args: flow_from(x=(1,2), y=(0,0))."""
    items = []
    for var, rate in sorted(rates.items()):
        if isinstance(rate, (int, float)):
            items.append((var, FlowInterval(float(rate), float(rate))))
        else:
            items.append((var, FlowInterval(float(rate[0]), float(rate[1]))))
    return Flow(tuple(items))


def clock_flow(*vars: str) -> Flow:
    """All variables are clocks (rate 1)."""
    return Flow(tuple((v, CLOCK_RATE) for v in sorted(vars)))


def stopped_flow() -> Flow:
    return Flow(())


# --- Reset (Discrete Update) ---

@dataclass(frozen=True)
class Reset:
    """Discrete update: var := value or var := var2 + offset."""
    assignments: Tuple[Tuple[str, Optional[str], float], ...]
    # (target_var, source_var_or_None, offset)
    # var := offset (if source is None)
    # var := source + offset (if source is given)

    def apply(self, valuation: Dict[str, float]) -> Dict[str, float]:
        result = dict(valuation)
        for target, source, offset in self.assignments:
            if source is None:
                result[target] = offset
            else:
                result[target] = valuation.get(source, 0.0) + offset
        return result

    def __str__(self):
        parts = []
        for target, source, offset in self.assignments:
            if source is None:
                parts.append(f"{target} := {offset}")
            elif offset == 0:
                parts.append(f"{target} := {source}")
            else:
                parts.append(f"{target} := {source} + {offset}")
        return "{" + ", ".join(parts) + "}" if parts else "{}"


def reset_to(var: str, val: float) -> Reset:
    return Reset(((var, None, val),))

def reset_vars(**assignments: float) -> Reset:
    return Reset(tuple((k, None, v) for k, v in sorted(assignments.items())))

def no_reset() -> Reset:
    return Reset(())


# --- Hybrid Automaton ---

@dataclass(frozen=True)
class HybridEdge:
    """Discrete transition in a hybrid automaton."""
    source: str       # source mode
    target: str       # target mode
    label: str        # action label
    guard: Predicate  # enabling condition
    reset: Reset      # variable updates

    def __str__(self):
        return f"{self.source} --[{self.label}, {self.guard}, {self.reset}]--> {self.target}"


@dataclass
class HybridAutomaton:
    """Hybrid automaton: finite control + continuous dynamics."""
    modes: Set[str]                    # locations/modes
    initial_mode: str
    variables: List[str]               # continuous variables
    flows: Dict[str, Flow]            # mode -> flow dynamics
    invariants: Dict[str, Predicate]  # mode -> staying condition
    edges: List[HybridEdge]
    initial_condition: Predicate       # constraint on initial variable values

    def get_edges_from(self, mode: str) -> List[HybridEdge]:
        return [e for e in self.edges if e.source == mode]

    def get_flow(self, mode: str) -> Flow:
        return self.flows.get(mode, stopped_flow())

    def get_invariant(self, mode: str) -> Predicate:
        return self.invariants.get(mode, true_pred())

    def is_rectangular(self) -> bool:
        """Check if this is a rectangular hybrid automaton.
        Rectangular: all flows are intervals, guards/invariants/resets
        are rectangular (involve single variables only)."""
        for mode in self.modes:
            flow = self.get_flow(mode)
            for var in self.variables:
                rate = flow.get_rate(var)
                # Rate must be an interval (always true by construction)
        # Check guards and invariants are rectangular (single-variable constraints)
        for edge in self.edges:
            for c in edge.guard.constraints:
                if len([v for v, coeff in c.coeffs if coeff != 0]) > 1:
                    return False
        for inv in self.invariants.values():
            for c in inv.constraints:
                if len([v for v, coeff in c.coeffs if coeff != 0]) > 1:
                    return False
        return True

    def max_constant(self) -> float:
        max_c = 0.0
        for edge in self.edges:
            for c in edge.guard.constraints:
                max_c = max(max_c, abs(c.constant))
        for inv in self.invariants.values():
            for c in inv.constraints:
                max_c = max(max_c, abs(c.constant))
        for c in self.initial_condition.constraints:
            max_c = max(max_c, abs(c.constant))
        return max_c


# --- Rectangular Zone (Extended DBM) ---

class RectZone:
    """
    Zone for rectangular hybrid automata.

    Represents a conjunction of constraints:
    - var_i in [lo_i, hi_i] (variable bounds)
    - var_i - var_j <= c_ij (difference constraints)

    Uses a DBM-like structure where index 0 is the constant reference.
    DBM[i][j] = (bound, strict) means var_i - var_j <(=) bound.
    """

    def __init__(self, variables: List[str]):
        self.variables = variables
        self.n = len(variables) + 1  # +1 for reference (index 0)
        self.var_index = {name: i + 1 for i, name in enumerate(variables)}
        # Initialize unconstrained
        self.dbm = [[(INF, False)] * self.n for _ in range(self.n)]
        for i in range(self.n):
            self.dbm[i][i] = (0, False)

    def copy(self) -> 'RectZone':
        z = RectZone.__new__(RectZone)
        z.variables = self.variables
        z.n = self.n
        z.var_index = self.var_index
        z.dbm = [row[:] for row in self.dbm]
        return z

    def _idx(self, var: str) -> int:
        return self.var_index[var]

    def is_empty(self) -> bool:
        """Check if zone is empty (inconsistent constraints)."""
        for i in range(self.n):
            b, s = self.dbm[i][i]
            if b < 0 or (b == 0 and s):
                return True
        return False

    def _add_bounds(self, b1: Tuple[float, bool], b2: Tuple[float, bool]) -> Tuple[float, bool]:
        """Add two DBM bounds: (a, s1) + (b, s2)."""
        if b1[0] == INF or b2[0] == INF:
            return (INF, False)
        return (b1[0] + b2[0], b1[1] or b2[1])

    def _min_bound(self, b1: Tuple[float, bool], b2: Tuple[float, bool]) -> Tuple[float, bool]:
        """Minimum of two DBM bounds."""
        if b1[0] < b2[0]:
            return b1
        if b2[0] < b1[0]:
            return b2
        # Equal values: strict is tighter
        return (b1[0], b1[1] or b2[1])

    def _leq_bound(self, b1: Tuple[float, bool], b2: Tuple[float, bool]) -> bool:
        """Check b1 <= b2 in the DBM ordering."""
        if b2[0] == INF:
            return True
        if b1[0] == INF:
            return False
        if b1[0] < b2[0]:
            return True
        if b1[0] > b2[0]:
            return False
        # Equal: strict <= non-strict, strict <= strict, non-strict <= non-strict
        return (not b1[1]) or b2[1]

    def canonicalize(self):
        """Floyd-Warshall shortest path closure."""
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    ij_via_k = self._add_bounds(self.dbm[i][k], self.dbm[k][j])
                    self.dbm[i][j] = self._min_bound(self.dbm[i][j], ij_via_k)
        return self

    def constrain(self, constraint: LinearConstraint) -> 'RectZone':
        """Apply a linear constraint to the zone."""
        z = self.copy()
        # Handle single-variable constraints: var op const
        non_zero = [(v, c) for v, c in constraint.coeffs if c != 0]
        if len(non_zero) == 1:
            var, coeff = non_zero[0]
            idx = z._idx(var)
            val = constraint.constant / coeff
            if coeff > 0:
                if constraint.op in (CompOp.LE, CompOp.EQ):
                    # var <= val => dbm[idx][0] = min(current, val)
                    z.dbm[idx][0] = z._min_bound(z.dbm[idx][0], (val, False))
                if constraint.op == CompOp.LT:
                    z.dbm[idx][0] = z._min_bound(z.dbm[idx][0], (val, True))
                if constraint.op in (CompOp.GE, CompOp.EQ):
                    # var >= val => -var <= -val => dbm[0][idx] = min(current, -val)
                    z.dbm[0][idx] = z._min_bound(z.dbm[0][idx], (-val, False))
                if constraint.op == CompOp.GT:
                    z.dbm[0][idx] = z._min_bound(z.dbm[0][idx], (-val, True))
            else:
                # coeff < 0: -var op/coeff val => var (flip_op) -val/coeff
                actual_val = -val  # val / coeff with coeff = -1
                if constraint.op in (CompOp.LE, CompOp.EQ):
                    # -var <= val => var >= -val
                    z.dbm[0][idx] = z._min_bound(z.dbm[0][idx], (val / (-coeff), False))
                if constraint.op in (CompOp.GE, CompOp.EQ):
                    z.dbm[idx][0] = z._min_bound(z.dbm[idx][0], (-constraint.constant / coeff, False))
                if constraint.op == CompOp.LT:
                    z.dbm[0][idx] = z._min_bound(z.dbm[0][idx], (val / (-coeff), True))
                if constraint.op == CompOp.GT:
                    z.dbm[idx][0] = z._min_bound(z.dbm[idx][0], (-constraint.constant / coeff, True))

        elif len(non_zero) == 2:
            # Difference constraint: a*x + b*y op c
            (v1, c1), (v2, c2) = non_zero
            if c1 == 1 and c2 == -1:
                # x - y op c
                idx1, idx2 = z._idx(v1), z._idx(v2)
                val = constraint.constant
                if constraint.op in (CompOp.LE, CompOp.EQ):
                    z.dbm[idx1][idx2] = z._min_bound(z.dbm[idx1][idx2], (val, False))
                if constraint.op == CompOp.LT:
                    z.dbm[idx1][idx2] = z._min_bound(z.dbm[idx1][idx2], (val, True))
                if constraint.op in (CompOp.GE, CompOp.EQ):
                    z.dbm[idx2][idx1] = z._min_bound(z.dbm[idx2][idx1], (-val, False))
                if constraint.op == CompOp.GT:
                    z.dbm[idx2][idx1] = z._min_bound(z.dbm[idx2][idx1], (-val, True))
            elif c1 == -1 and c2 == 1:
                # -x + y op c => y - x op c
                idx1, idx2 = z._idx(v2), z._idx(v1)
                val = constraint.constant
                if constraint.op in (CompOp.LE, CompOp.EQ):
                    z.dbm[idx1][idx2] = z._min_bound(z.dbm[idx1][idx2], (val, False))
                if constraint.op == CompOp.LT:
                    z.dbm[idx1][idx2] = z._min_bound(z.dbm[idx1][idx2], (val, True))
                if constraint.op in (CompOp.GE, CompOp.EQ):
                    z.dbm[idx2][idx1] = z._min_bound(z.dbm[idx2][idx1], (-val, False))
                if constraint.op == CompOp.GT:
                    z.dbm[idx2][idx1] = z._min_bound(z.dbm[idx2][idx1], (-val, True))
        z.canonicalize()
        return z

    def constrain_pred(self, pred: Predicate) -> 'RectZone':
        """Apply all constraints in a predicate."""
        z = self
        for c in pred.constraints:
            z = z.constrain(c)
            if z.is_empty():
                return z
        return z

    def time_elapse(self, flow: Flow) -> 'RectZone':
        """
        Apply time elapse with given flow dynamics.

        For rectangular automata, variable x with rate dx/dt in [a, b]:
        - After time delta >= 0: x' = x + delta * rate, rate in [a, b]
        - So x' - x in [0, INF] if a >= 0 (growing)
        - Upper bound on x grows by delta * b (if b > 0)
        - Lower bound on x grows by delta * a (if a > 0)

        For exact rate (a = b = r):
        - All variables with same rate maintain difference constraints
        - x' - y' = (x + r*t) - (y + r*t) = x - y (preserved)

        For general rectangular flows:
        - Different rates break difference constraints between variables
        - Same-rate variables preserve differences
        """
        z = self.copy()

        # Group variables by rate
        rate_groups = {}  # flow_interval -> [var_indices]
        for var in self.variables:
            rate = flow.get_rate(var)
            key = (rate.lo, rate.hi)
            if key not in rate_groups:
                rate_groups[key] = []
            rate_groups[key].append(self._idx(var))

        # For stopped variables (rate 0), everything is preserved
        # For variables with different rates, relax difference constraints between groups

        # Step 1: Remove upper bounds on variables with positive rate
        # (they can grow indefinitely with time)
        for var in self.variables:
            rate = flow.get_rate(var)
            idx = self._idx(var)
            if rate.hi > 0:
                # Upper bound goes to infinity (time can pass arbitrarily)
                z.dbm[idx][0] = (INF, False)
            if rate.lo < 0:
                # Lower bound goes to -infinity
                z.dbm[0][idx] = (INF, False)

        # Step 2: Relax difference constraints between variables with different rates
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 == var2:
                    continue
                rate1 = flow.get_rate(var1)
                rate2 = flow.get_rate(var2)
                idx1 = self._idx(var1)
                idx2 = self._idx(var2)
                # x1 - x2 difference: after time t, becomes (x1 + r1*t) - (x2 + r2*t) = (x1-x2) + (r1-r2)*t
                # If r1 - r2 can be positive, upper bound on x1-x2 goes to INF
                # If r1 - r2 can be negative, lower bound on x1-x2 goes to -INF
                r_diff_max = rate1.hi - rate2.lo
                r_diff_min = rate1.lo - rate2.hi
                if r_diff_max > 0:
                    z.dbm[idx1][idx2] = (INF, False)
                if r_diff_min < 0:
                    z.dbm[idx2][idx1] = (INF, False)
                # If rates are identical (r1 == r2), difference is preserved (no relaxation)

        z.canonicalize()
        return z

    def reset(self, reset: Reset) -> 'RectZone':
        """Apply discrete reset to the zone."""
        z = self.copy()
        for target, source, offset in reset.assignments:
            idx = z._idx(target)
            if source is None:
                # target := offset (constant assignment)
                # Remove all constraints involving target
                for j in range(z.n):
                    z.dbm[idx][j] = (INF, False)
                    z.dbm[j][idx] = (INF, False)
                z.dbm[idx][idx] = (0, False)
                # Set target = offset: target - 0 <= offset AND 0 - target <= -offset
                z.dbm[idx][0] = (offset, False)
                z.dbm[0][idx] = (-offset, False)
                # Derive difference constraints with other variables
                # target - var_j <= offset - lower(var_j)
                # var_j - target <= upper(var_j) - offset
                # These will be derived by canonicalization
            else:
                # target := source + offset
                src_idx = z._idx(source)
                # Copy all constraints from source, shifted by offset
                for j in range(z.n):
                    if j == idx:
                        continue
                    # target - j <= source - j + offset? No...
                    # target = source + offset
                    # target - j = source + offset - j = (source - j) + offset
                    b, s = z.dbm[src_idx][j]
                    if b != INF:
                        z.dbm[idx][j] = (b + offset, s)
                    else:
                        z.dbm[idx][j] = (INF, False)
                    # j - target = j - source - offset = (j - source) - offset
                    b, s = z.dbm[j][src_idx]
                    if b != INF:
                        z.dbm[j][idx] = (b - offset, s)
                    else:
                        z.dbm[j][idx] = (INF, False)
                z.dbm[idx][idx] = (0, False)
        z.canonicalize()
        return z

    def includes(self, other: 'RectZone') -> bool:
        """Check if self includes other (other is subset of self)."""
        if other.is_empty():
            return True
        for i in range(self.n):
            for j in range(self.n):
                if not self._leq_bound(other.dbm[i][j], self.dbm[i][j]):
                    return False
        return True

    def intersect(self, other: 'RectZone') -> 'RectZone':
        """Intersection of two zones."""
        z = self.copy()
        for i in range(self.n):
            for j in range(self.n):
                z.dbm[i][j] = self._min_bound(self.dbm[i][j], other.dbm[i][j])
        z.canonicalize()
        return z

    def get_bounds(self, var: str) -> Tuple[float, float]:
        """Get [lower, upper] bounds for a variable."""
        idx = self._idx(var)
        upper = self.dbm[idx][0][0]
        lower_neg = self.dbm[0][idx][0]
        lower = -lower_neg if lower_neg != INF else -INF
        return (lower, upper)

    def sample(self) -> Optional[Dict[str, float]]:
        """Sample a concrete valuation from the zone."""
        if self.is_empty():
            return None
        vals = {}
        for var in self.variables:
            lo, hi = self.get_bounds(var)
            if lo == -INF:
                lo = 0.0
            if hi == INF:
                hi = lo + 1.0
            vals[var] = (lo + hi) / 2.0
        return vals

    def __eq__(self, other):
        if not isinstance(other, RectZone):
            return False
        return self.dbm == other.dbm

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.dbm))

    def __str__(self):
        parts = []
        for var in self.variables:
            lo, hi = self.get_bounds(var)
            parts.append(f"{var} in [{lo}, {hi}]")
        return "{" + ", ".join(parts) + "}"


def initial_zone(variables: List[str], init_cond: Predicate) -> RectZone:
    """Create initial zone from initial condition predicate."""
    z = RectZone(variables)
    z = z.constrain_pred(init_cond)
    return z


# --- Symbolic State ---

@dataclass(frozen=True)
class SymState:
    """Symbolic state: (mode, zone_hash) for visited tracking."""
    mode: str
    zone_id: int  # hash of zone DBM

    def __hash__(self):
        return hash((self.mode, self.zone_id))


# --- Simulation ---

@dataclass
class TraceStep:
    """One step in a concrete trace."""
    mode: str
    entry_values: Dict[str, float]
    time_spent: float
    exit_values: Dict[str, float]
    edge_label: Optional[str] = None  # None for last step


@dataclass
class SimTrace:
    """Concrete simulation trace."""
    steps: List[TraceStep]
    reaches_target: bool


def simulate_step(ha: HybridAutomaton, mode: str, values: Dict[str, float],
                  time_delta: float) -> Dict[str, float]:
    """Simulate continuous evolution for time_delta time units."""
    flow = ha.get_flow(mode)
    result = dict(values)
    for var in ha.variables:
        rate = flow.get_rate(var)
        # Use midpoint of rate interval for simulation
        mid_rate = (rate.lo + rate.hi) / 2.0
        result[var] = values[var] + mid_rate * time_delta
    return result


def simulate(ha: HybridAutomaton, initial_values: Dict[str, float],
             steps: List[Tuple[float, str]], time_step: float = 0.01) -> SimTrace:
    """
    Simulate a hybrid automaton trajectory.

    steps: list of (time_to_wait, edge_label) pairs
    """
    trace = []
    mode = ha.initial_mode
    values = dict(initial_values)

    for time_wait, edge_label in steps:
        entry_values = dict(values)
        # Continuous evolution
        values = simulate_step(ha, mode, values, time_wait)
        exit_values = dict(values)

        trace.append(TraceStep(
            mode=mode,
            entry_values=entry_values,
            time_spent=time_wait,
            exit_values=exit_values,
            edge_label=edge_label
        ))

        # Find matching edge
        taken = None
        for edge in ha.get_edges_from(mode):
            if edge.label == edge_label and edge.guard.evaluate(values):
                taken = edge
                break
        if taken is None:
            return SimTrace(steps=trace, reaches_target=False)

        # Apply reset
        values = taken.reset.apply(values)
        mode = taken.target

    # Final state
    trace.append(TraceStep(
        mode=mode,
        entry_values=dict(values),
        time_spent=0,
        exit_values=dict(values),
        edge_label=None
    ))

    return SimTrace(steps=trace, reaches_target=True)


# --- Reachability Analysis ---

@dataclass
class ReachResult:
    """Result of reachability analysis."""
    reachable: bool
    target_modes: Set[str]
    explored_states: int
    trace: Optional[List[Tuple[str, str]]] = None  # [(mode, edge_label), ...]
    zone_at_target: Optional[RectZone] = None


def _zone_key(mode: str, zone: RectZone) -> Tuple:
    """Create a hashable key for (mode, zone) pair."""
    return (mode, tuple(tuple(row) for row in zone.dbm))


def check_reachability(ha: HybridAutomaton, targets: Set[str],
                       max_iterations: int = 1000) -> ReachResult:
    """
    Check if any target mode is reachable using zone-based BFS.

    For rectangular automata: uses RectZone with time_elapse + guard + reset + invariant.
    """
    variables = ha.variables

    # Initial zone
    z0 = initial_zone(variables, ha.initial_condition)
    inv = ha.get_invariant(ha.initial_mode)
    z0 = z0.constrain_pred(inv)
    if z0.is_empty():
        return ReachResult(reachable=False, target_modes=targets, explored_states=0)

    # Check if initial mode is target
    if ha.initial_mode in targets:
        return ReachResult(reachable=True, target_modes=targets, explored_states=1,
                           trace=[(ha.initial_mode, "")], zone_at_target=z0)

    # BFS
    queue = deque()
    queue.append((ha.initial_mode, z0, [(ha.initial_mode, "")]))
    visited = set()
    visited.add(_zone_key(ha.initial_mode, z0))
    explored = 0

    while queue and explored < max_iterations:
        mode, zone, trace = queue.popleft()
        explored += 1

        # Time elapse
        flow = ha.get_flow(mode)
        z_elapsed = zone.time_elapse(flow)
        # Apply invariant to constrain elapsed zone
        z_elapsed = z_elapsed.constrain_pred(ha.get_invariant(mode))
        if z_elapsed.is_empty():
            continue

        # Try each outgoing edge
        for edge in ha.get_edges_from(mode):
            # Apply guard
            z_guarded = z_elapsed.constrain_pred(edge.guard)
            if z_guarded.is_empty():
                continue

            # Apply reset
            z_reset = z_guarded.reset(edge.reset)

            # Apply target invariant
            z_target = z_reset.constrain_pred(ha.get_invariant(edge.target))
            if z_target.is_empty():
                continue

            new_trace = trace + [(edge.target, edge.label)]

            # Check if target reached
            if edge.target in targets:
                return ReachResult(
                    reachable=True,
                    target_modes=targets,
                    explored_states=explored,
                    trace=new_trace,
                    zone_at_target=z_target
                )

            # Check if already visited (subsumption)
            key = _zone_key(edge.target, z_target)
            if key not in visited:
                # Also check zone inclusion
                subsumed = False
                for v_key in list(visited):
                    if v_key[0] == edge.target:
                        v_zone = RectZone(variables)
                        v_zone.dbm = [list(row) for row in v_key[1]]
                        if v_zone.includes(z_target):
                            subsumed = True
                            break
                if not subsumed:
                    visited.add(key)
                    queue.append((edge.target, z_target, new_trace))

    return ReachResult(reachable=False, target_modes=targets, explored_states=explored)


def check_safety(ha: HybridAutomaton, unsafe_modes: Set[str],
                 max_iterations: int = 1000) -> ReachResult:
    """Check if unsafe modes are unreachable (safety)."""
    result = check_reachability(ha, unsafe_modes, max_iterations)
    return result


# --- Product Construction ---

def product(ha1: HybridAutomaton, ha2: HybridAutomaton,
            sync_labels: Optional[Set[str]] = None) -> HybridAutomaton:
    """
    Synchronous product of two hybrid automata.
    Edges with labels in sync_labels must synchronize.
    Edges with other labels can fire independently.
    """
    if sync_labels is None:
        # Synchronize on shared labels
        labels1 = {e.label for e in ha1.edges}
        labels2 = {e.label for e in ha2.edges}
        sync_labels = labels1 & labels2

    # Create product modes
    modes = set()
    for m1 in ha1.modes:
        for m2 in ha2.modes:
            modes.add(f"{m1}_{m2}")

    initial_mode = f"{ha1.initial_mode}_{ha2.initial_mode}"

    # Variables: union (must be disjoint)
    variables = ha1.variables + ha2.variables

    # Flows: combine per mode pair
    flows = {}
    for m1 in ha1.modes:
        for m2 in ha2.modes:
            prod_mode = f"{m1}_{m2}"
            f1 = ha1.get_flow(m1)
            f2 = ha2.get_flow(m2)
            combined_rates = list(f1.rates) + list(f2.rates)
            flows[prod_mode] = Flow(tuple(combined_rates))

    # Invariants: conjunction
    invariants = {}
    for m1 in ha1.modes:
        for m2 in ha2.modes:
            prod_mode = f"{m1}_{m2}"
            inv1 = ha1.get_invariant(m1)
            inv2 = ha2.get_invariant(m2)
            invariants[prod_mode] = pred_and(inv1, inv2)

    # Edges
    edges = []
    # Synchronized edges
    for e1 in ha1.edges:
        for e2 in ha2.edges:
            if e1.label in sync_labels and e1.label == e2.label:
                src = f"{e1.source}_{e2.source}"
                tgt = f"{e1.target}_{e2.target}"
                guard = pred_and(e1.guard, e2.guard)
                reset = Reset(e1.reset.assignments + e2.reset.assignments)
                edges.append(HybridEdge(src, tgt, e1.label, guard, reset))

    # Independent edges
    for e1 in ha1.edges:
        if e1.label not in sync_labels:
            for m2 in ha2.modes:
                src = f"{e1.source}_{m2}"
                tgt = f"{e1.target}_{m2}"
                edges.append(HybridEdge(src, tgt, e1.label, e1.guard, e1.reset))

    for e2 in ha2.edges:
        if e2.label not in sync_labels:
            for m1 in ha1.modes:
                src = f"{m1}_{e2.source}"
                tgt = f"{m1}_{e2.target}"
                edges.append(HybridEdge(src, tgt, e2.label, e2.guard, e2.reset))

    # Initial condition: conjunction
    init_cond = pred_and(ha1.initial_condition, ha2.initial_condition)

    return HybridAutomaton(
        modes=modes,
        initial_mode=initial_mode,
        variables=variables,
        flows=flows,
        invariants=invariants,
        edges=edges,
        initial_condition=init_cond
    )


# --- Example Systems ---

def thermostat(temp_low: float = 18.0, temp_high: float = 22.0,
               heat_rate: float = 1.0, cool_rate: float = -0.5) -> HybridAutomaton:
    """
    Thermostat controller:
    - Mode 'heat': temperature rises at heat_rate
    - Mode 'cool': temperature falls at cool_rate
    - Switch heat->cool when temp >= temp_high
    - Switch cool->heat when temp <= temp_low
    - Invariant: temp in [temp_low - 2, temp_high + 2]
    """
    return HybridAutomaton(
        modes={'heat', 'cool'},
        initial_mode='heat',
        variables=['temp'],
        flows={
            'heat': flow_from(temp=heat_rate),
            'cool': flow_from(temp=cool_rate),
        },
        invariants={
            'heat': pred_from(var_leq('temp', temp_high)),
            'cool': pred_from(var_geq('temp', temp_low)),
        },
        edges=[
            HybridEdge('heat', 'cool', 'off',
                        pred_from(var_geq('temp', temp_high)), no_reset()),
            HybridEdge('cool', 'heat', 'on',
                        pred_from(var_leq('temp', temp_low)), no_reset()),
        ],
        initial_condition=pred_from(var_geq('temp', temp_low), var_leq('temp', temp_high))
    )


def water_tank(inflow_rate: float = 2.0, outflow_rate: float = -1.0,
               low_level: float = 1.0, high_level: float = 10.0) -> HybridAutomaton:
    """
    Water tank with pump controller:
    - Mode 'filling': pump on, water level rises
    - Mode 'draining': pump off, water level falls
    - Switch filling->draining when level >= high_level
    - Switch draining->filling when level <= low_level
    """
    return HybridAutomaton(
        modes={'filling', 'draining'},
        initial_mode='filling',
        variables=['level'],
        flows={
            'filling': flow_from(level=inflow_rate),
            'draining': flow_from(level=outflow_rate),
        },
        invariants={
            'filling': pred_from(var_leq('level', high_level)),
            'draining': pred_from(var_geq('level', low_level)),
        },
        edges=[
            HybridEdge('filling', 'draining', 'pump_off',
                        pred_from(var_geq('level', high_level)), no_reset()),
            HybridEdge('draining', 'filling', 'pump_on',
                        pred_from(var_leq('level', low_level)), no_reset()),
        ],
        initial_condition=pred_from(var_geq('level', low_level), var_leq('level', high_level))
    )


def railroad_crossing(approach_speed: float = 1.0,
                      gate_speed: float = 1.0) -> HybridAutomaton:
    """
    Railroad crossing controller:
    - Variables: train_pos (train position), gate (gate angle 0=open, 90=closed)
    - Train approaches from far away, crosses at position 0
    - Gate must be closed before train reaches crossing

    Modes: far, near, crossing, past
    Gate: opening, closing, open, closed
    """
    return HybridAutomaton(
        modes={'far_open', 'near_closing', 'near_closed', 'cross_closed', 'past_opening', 'past_open'},
        initial_mode='far_open',
        variables=['train', 'gate'],
        flows={
            'far_open': flow_from(train=(-approach_speed, -approach_speed), gate=0),
            'near_closing': flow_from(train=(-approach_speed, -approach_speed), gate=gate_speed),
            'near_closed': flow_from(train=(-approach_speed, -approach_speed), gate=0),
            'cross_closed': flow_from(train=(-approach_speed, -approach_speed), gate=0),
            'past_opening': flow_from(train=(-approach_speed, -approach_speed), gate=(-gate_speed, -gate_speed)),
            'past_open': flow_from(train=(-approach_speed, -approach_speed), gate=0),
        },
        invariants={
            'far_open': pred_from(var_geq('train', 100), var_eq('gate', 0)),
            'near_closing': pred_from(var_geq('train', 0), var_leq('gate', 90)),
            'near_closed': pred_from(var_geq('train', 0), var_eq('gate', 90)),
            'cross_closed': pred_from(var_leq('train', 0), var_eq('gate', 90)),
            'past_opening': pred_from(var_leq('train', -10), var_geq('gate', 0)),
            'past_open': pred_from(var_leq('train', -10), var_eq('gate', 0)),
        },
        edges=[
            HybridEdge('far_open', 'near_closing', 'approach',
                        pred_from(var_leq('train', 100)), no_reset()),
            HybridEdge('near_closing', 'near_closed', 'gate_closed',
                        pred_from(var_geq('gate', 90)), no_reset()),
            HybridEdge('near_closed', 'cross_closed', 'enter',
                        pred_from(var_leq('train', 0)), no_reset()),
            HybridEdge('cross_closed', 'past_opening', 'exit',
                        pred_from(var_leq('train', -10)), no_reset()),
            HybridEdge('past_opening', 'past_open', 'gate_opened',
                        pred_from(var_leq('gate', 0)), no_reset()),
        ],
        initial_condition=pred_from(var_geq('train', 200), var_eq('gate', 0))
    )


def bouncing_ball(gravity: float = 9.81, coeff_restitution: float = 0.9,
                  initial_height: float = 10.0) -> HybridAutomaton:
    """
    Bouncing ball:
    - Variables: h (height), v (velocity)
    - Mode 'fly': dh/dt = v, dv/dt = -gravity (approximated as rectangular)
    - When h <= 0: bounce (v := -coeff * v)

    Note: This is an approximation -- true bouncing ball has dv/dt = -g which is
    not rectangular. We approximate with v in [-v_max, v_max] per timestep.
    For more precise analysis, we use a piecewise rectangular over-approximation.
    """
    # Approximate with two modes: rising and falling
    v_max = math.sqrt(2 * gravity * initial_height)
    return HybridAutomaton(
        modes={'fly'},
        initial_mode='fly',
        variables=['h', 'v'],
        flows={
            # Approximation: v changes at rate [-gravity, -gravity]
            # h changes at rate [v_lo, v_hi] -- but this depends on v!
            # For rectangular approximation: h rate = [-v_max, v_max], v rate = [-gravity, -gravity]
            'fly': flow_from(h=(-v_max, v_max), v=(-gravity, -gravity)),
        },
        invariants={
            'fly': pred_from(var_geq('h', 0)),
        },
        edges=[
            HybridEdge('fly', 'fly', 'bounce',
                        pred_from(var_leq('h', 0), var_lt('v', 0)),
                        # v := -coeff * v (approximated as reset to range)
                        Reset((('h', None, 0), ('v', None, coeff_restitution * v_max)))),
        ],
        initial_condition=pred_from(var_eq('h', initial_height), var_eq('v', 0))
    )


def two_tank(capacity: float = 10.0) -> HybridAutomaton:
    """
    Two-tank system: fluid flows between two tanks with a shared pump.
    - Variables: h1 (tank 1 level), h2 (tank 2 level)
    - Mode 'fill1': pump fills tank 1, tank 2 drains
    - Mode 'fill2': pump fills tank 2, tank 1 drains
    - Switch when target tank reaches capacity or source tank empties
    """
    return HybridAutomaton(
        modes={'fill1', 'fill2'},
        initial_mode='fill1',
        variables=['h1', 'h2'],
        flows={
            'fill1': flow_from(h1=2, h2=-1),
            'fill2': flow_from(h1=-1, h2=2),
        },
        invariants={
            'fill1': pred_from(var_leq('h1', capacity), var_geq('h2', 0)),
            'fill2': pred_from(var_geq('h1', 0), var_leq('h2', capacity)),
        },
        edges=[
            HybridEdge('fill1', 'fill2', 'switch',
                        pred_from(var_geq('h1', capacity)),
                        no_reset()),
            HybridEdge('fill2', 'fill1', 'switch',
                        pred_from(var_geq('h2', capacity)),
                        no_reset()),
            # Also switch if source drains
            HybridEdge('fill1', 'fill2', 'drain_switch',
                        pred_from(var_leq('h2', 0)),
                        no_reset()),
            HybridEdge('fill2', 'fill1', 'drain_switch',
                        pred_from(var_leq('h1', 0)),
                        no_reset()),
        ],
        initial_condition=pred_from(
            var_eq('h1', 2), var_eq('h2', 5)
        )
    )


# --- High-Level Verification API ---

@dataclass
class VerifyResult:
    """Result of hybrid automata verification."""
    verdict: str  # 'SAFE', 'UNSAFE', 'UNKNOWN'
    property_name: str
    explored_states: int
    trace: Optional[List[Tuple[str, str]]] = None
    zone_at_violation: Optional[RectZone] = None
    details: str = ""


def verify_safety(ha: HybridAutomaton, unsafe_modes: Set[str],
                  property_name: str = "safety",
                  max_iterations: int = 1000) -> VerifyResult:
    """Verify that unsafe modes are never reached."""
    result = check_reachability(ha, unsafe_modes, max_iterations)
    if result.reachable:
        return VerifyResult(
            verdict='UNSAFE',
            property_name=property_name,
            explored_states=result.explored_states,
            trace=result.trace,
            zone_at_violation=result.zone_at_target,
            details=f"Unsafe mode reachable: {result.trace[-1][0] if result.trace else '?'}"
        )
    return VerifyResult(
        verdict='SAFE',
        property_name=property_name,
        explored_states=result.explored_states,
        details=f"No unsafe mode reachable after {result.explored_states} states"
    )


def verify_invariant(ha: HybridAutomaton, invariant: Predicate,
                     property_name: str = "invariant",
                     max_iterations: int = 1000) -> VerifyResult:
    """
    Verify that a predicate holds in all reachable states.

    Strategy: add 'violation' mode reachable when invariant is violated.
    """
    # Create augmented automaton with violation mode
    aug_modes = ha.modes | {'__violation__'}
    aug_edges = list(ha.edges)
    # From every mode, add edge to violation when invariant is negated
    for mode in ha.modes:
        # Negate invariant: if invariant is conjunction of c1 AND c2 AND ...
        # negation is disjunction: NOT c1 OR NOT c2 OR ...
        for c in invariant.constraints:
            neg_op = _negate_op(c.op)
            if neg_op is not None:
                neg_constraint = LinearConstraint(c.coeffs, neg_op, c.constant)
                aug_edges.append(HybridEdge(
                    mode, '__violation__', 'violate',
                    pred_from(neg_constraint), no_reset()
                ))

    aug_ha = HybridAutomaton(
        modes=aug_modes,
        initial_mode=ha.initial_mode,
        variables=ha.variables,
        flows={**ha.flows, '__violation__': stopped_flow()},
        invariants=dict(ha.invariants),
        edges=aug_edges,
        initial_condition=ha.initial_condition
    )

    return verify_safety(aug_ha, {'__violation__'}, property_name, max_iterations)


def _negate_op(op: CompOp) -> Optional[CompOp]:
    """Negate a comparison operator."""
    return {
        CompOp.LT: CompOp.GE,
        CompOp.LE: CompOp.GT,
        CompOp.GT: CompOp.LE,
        CompOp.GE: CompOp.LT,
        CompOp.EQ: CompOp.NE,
        CompOp.NE: CompOp.EQ,
    }.get(op)


def verify_bounded_liveness(ha: HybridAutomaton, target_modes: Set[str],
                            property_name: str = "liveness",
                            max_iterations: int = 1000) -> VerifyResult:
    """Check if a target mode is eventually reachable (bounded check)."""
    result = check_reachability(ha, target_modes, max_iterations)
    if result.reachable:
        return VerifyResult(
            verdict='SATISFIED',
            property_name=property_name,
            explored_states=result.explored_states,
            trace=result.trace,
            details=f"Target mode {result.trace[-1][0] if result.trace else '?'} reachable"
        )
    return VerifyResult(
        verdict='UNKNOWN',
        property_name=property_name,
        explored_states=result.explored_states,
        details=f"Target not reached in {result.explored_states} states (may still be reachable)"
    )


def analyze_modes(ha: HybridAutomaton, max_iterations: int = 1000) -> Dict[str, bool]:
    """Check reachability of each mode."""
    reachable = {ha.initial_mode: True}
    for mode in ha.modes:
        if mode != ha.initial_mode:
            result = check_reachability(ha, {mode}, max_iterations)
            reachable[mode] = result.reachable
    return reachable


def zone_graph_summary(ha: HybridAutomaton, max_iterations: int = 500) -> Dict:
    """Explore zone graph and return summary statistics."""
    variables = ha.variables
    z0 = initial_zone(variables, ha.initial_condition)
    inv = ha.get_invariant(ha.initial_mode)
    z0 = z0.constrain_pred(inv)

    if z0.is_empty():
        return {'explored': 0, 'modes_reached': set(), 'edges_taken': 0}

    queue = deque()
    queue.append((ha.initial_mode, z0))
    visited = set()
    visited.add(_zone_key(ha.initial_mode, z0))
    modes_reached = {ha.initial_mode}
    edges_taken = 0
    explored = 0

    while queue and explored < max_iterations:
        mode, zone = queue.popleft()
        explored += 1

        flow = ha.get_flow(mode)
        z_elapsed = zone.time_elapse(flow)
        z_elapsed = z_elapsed.constrain_pred(ha.get_invariant(mode))
        if z_elapsed.is_empty():
            continue

        for edge in ha.get_edges_from(mode):
            z_guarded = z_elapsed.constrain_pred(edge.guard)
            if z_guarded.is_empty():
                continue
            z_reset = z_guarded.reset(edge.reset)
            z_target = z_reset.constrain_pred(ha.get_invariant(edge.target))
            if z_target.is_empty():
                continue

            edges_taken += 1
            modes_reached.add(edge.target)

            key = _zone_key(edge.target, z_target)
            if key not in visited:
                subsumed = False
                for v_key in list(visited):
                    if v_key[0] == edge.target:
                        v_zone = RectZone(variables)
                        v_zone.dbm = [list(row) for row in v_key[1]]
                        if v_zone.includes(z_target):
                            subsumed = True
                            break
                if not subsumed:
                    visited.add(key)
                    queue.append((edge.target, z_target))

    return {
        'explored': explored,
        'modes_reached': modes_reached,
        'edges_taken': edges_taken,
        'zone_count': len(visited),
    }


def compare_hybrid_vs_timed(ha: HybridAutomaton, property_name: str = "comparison") -> Dict:
    """
    Compare the hybrid automaton analysis results with a note about
    what timed automata cannot capture.
    """
    is_rect = ha.is_rectangular()
    summary = zone_graph_summary(ha)
    non_clock_vars = []
    for mode in ha.modes:
        flow = ha.get_flow(mode)
        for var in ha.variables:
            rate = flow.get_rate(var)
            if not rate.is_clock() and not rate.is_stopped():
                if var not in non_clock_vars:
                    non_clock_vars.append(var)

    return {
        'is_rectangular': is_rect,
        'num_variables': len(ha.variables),
        'num_modes': len(ha.modes),
        'non_clock_variables': non_clock_vars,
        'zone_graph': summary,
        'timed_automata_expressible': len(non_clock_vars) == 0,
        'note': 'Hybrid automata extend timed automata with non-unit-rate dynamics'
                if non_clock_vars else 'This system could be modeled as a timed automaton'
    }


def batch_verify(ha: HybridAutomaton,
                 properties: List[Tuple[str, Set[str]]],
                 max_iterations: int = 1000) -> List[VerifyResult]:
    """Verify multiple safety properties (name, unsafe_modes) against a hybrid automaton."""
    results = []
    for name, unsafe in properties:
        results.append(verify_safety(ha, unsafe, name, max_iterations))
    return results
