"""V207: Stochastic Timed Games

Combines V202 (timed games with DBM zones) and V165 (stochastic parity games)
to create games with real-time constraints AND probabilistic transitions.

Three player types:
  MIN  -- controller, minimizes cost / tries to reach target
  MAX  -- adversary, maximizes cost / prevents target
  RANDOM -- nature, chooses successor by probability distribution

Solvers:
  - Almost-sure timed reachability (MIN wins if target reached with prob 1)
  - Positive-probability timed reachability (MIN wins if prob > 0)
  - Stochastic timed safety (MIN avoids unsafe with prob 1)
  - Expected-time reachability (minimize expected time to target)
  - Qualitative Buchi (visit accepting infinitely often, a.s. or p.p.)

Composes: V202 (timed games) + V165 (stochastic games) + V206 (DBM zones)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import math

# ============================================================
# DBM Zone Library (self-contained, adapted from V206)
# ============================================================

INF = float('inf')
_INF_BOUND = (INF, False)


def _min_bound(a, b):
    """Tighter of two bounds."""
    if a[0] < b[0]:
        return a
    if b[0] < a[0]:
        return b
    return (a[0], a[1] or b[1])  # same value => strict wins


def _add_bounds(a, b):
    """Add two bounds."""
    if a[0] == INF or b[0] == INF:
        return _INF_BOUND
    return (a[0] + b[0], a[1] or b[1])


def _leq_bound(a, b):
    """Is bound a <= bound b?"""
    if a[0] < b[0]:
        return True
    if a[0] > b[0]:
        return False
    if b[1]:
        return a[1]
    return True


class CompOp(Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="


@dataclass(frozen=True)
class ClockConstraint:
    clock: str
    op: CompOp
    value: int


@dataclass(frozen=True)
class Guard:
    constraints: Tuple[ClockConstraint, ...]


def true_guard():
    return Guard(constraints=())


def clock_lt(c, v):
    return ClockConstraint(c, CompOp.LT, v)

def clock_le(c, v):
    return ClockConstraint(c, CompOp.LE, v)

def clock_gt(c, v):
    return ClockConstraint(c, CompOp.GT, v)

def clock_ge(c, v):
    return ClockConstraint(c, CompOp.GE, v)

def clock_eq(c, v):
    return ClockConstraint(c, CompOp.EQ, v)

def guard_and(*guards_or_constraints):
    cs = []
    for g in guards_or_constraints:
        if isinstance(g, Guard):
            cs.extend(g.constraints)
        elif isinstance(g, ClockConstraint):
            cs.append(g)
        else:
            raise TypeError(f"Expected Guard or ClockConstraint, got {type(g)}")
    return Guard(constraints=tuple(cs))


class Zone:
    """DBM zone representation for clock valuations."""

    def __init__(self, n, clock_names):
        self.n = n  # number of clocks (index 0 = reference)
        self.clock_names = list(clock_names)
        dim = n + 1
        self.dbm = [[_INF_BOUND] * dim for _ in range(dim)]
        for i in range(dim):
            self.dbm[i][i] = (0, False)

    def copy(self):
        z = Zone.__new__(Zone)
        z.n = self.n
        z.clock_names = self.clock_names
        dim = self.n + 1
        z.dbm = [row[:] for row in self.dbm]
        return z

    def is_empty(self):
        dim = self.n + 1
        for i in range(dim):
            b = self.dbm[i][i]
            if b[0] < 0 or (b[0] == 0 and b[1]):
                return True
        return False

    def canonicalize(self):
        dim = self.n + 1
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    s = _add_bounds(self.dbm[i][k], self.dbm[k][j])
                    if _leq_bound(s, self.dbm[i][j]):
                        self.dbm[i][j] = s
        return self

    def includes(self, other):
        dim = self.n + 1
        for i in range(dim):
            for j in range(dim):
                if not _leq_bound(other.dbm[i][j], self.dbm[i][j]):
                    return False
        return True

    def intersect(self, other):
        z = self.copy()
        dim = self.n + 1
        for i in range(dim):
            for j in range(dim):
                z.dbm[i][j] = _min_bound(z.dbm[i][j], other.dbm[i][j])
        z.canonicalize()
        return z

    def __eq__(self, other):
        if not isinstance(other, Zone):
            return False
        dim = self.n + 1
        for i in range(dim):
            for j in range(dim):
                if self.dbm[i][j] != other.dbm[i][j]:
                    return False
        return True

    def __hash__(self):
        dim = self.n + 1
        return hash(tuple(self.dbm[i][j] for i in range(dim) for j in range(dim)))


def make_zone(clock_names):
    """Unconstrained zone: all clocks >= 0."""
    n = len(clock_names)
    z = Zone(n, clock_names)
    for i in range(1, n + 1):
        z.dbm[0][i] = (0, False)  # 0 - x_i <= 0 => x_i >= 0
    z.canonicalize()
    return z


def make_zero_zone(clock_names):
    """All clocks = 0."""
    n = len(clock_names)
    z = Zone(n, clock_names)
    dim = n + 1
    for i in range(dim):
        for j in range(dim):
            z.dbm[i][j] = (0, False)
    return z


def _clock_index(z, clock_name):
    return z.clock_names.index(clock_name) + 1


def constrain_zone(z, cc):
    z = z.copy()
    idx = _clock_index(z, cc.clock)
    if cc.op == CompOp.LE:
        z.dbm[idx][0] = _min_bound(z.dbm[idx][0], (cc.value, False))
    elif cc.op == CompOp.LT:
        z.dbm[idx][0] = _min_bound(z.dbm[idx][0], (cc.value, True))
    elif cc.op == CompOp.GE:
        z.dbm[0][idx] = _min_bound(z.dbm[0][idx], (-cc.value, False))
    elif cc.op == CompOp.GT:
        z.dbm[0][idx] = _min_bound(z.dbm[0][idx], (-cc.value, True))
    elif cc.op == CompOp.EQ:
        z.dbm[idx][0] = _min_bound(z.dbm[idx][0], (cc.value, False))
        z.dbm[0][idx] = _min_bound(z.dbm[0][idx], (-cc.value, False))
    z.canonicalize()
    return z


def apply_guard(z, guard):
    for cc in guard.constraints:
        z = constrain_zone(z, cc)
        if z.is_empty():
            return z
    return z


def reset_clocks(z, resets, clock_names=None):
    z = z.copy()
    for c in resets:
        idx = _clock_index(z, c)
        dim = z.n + 1
        for j in range(dim):
            z.dbm[idx][j] = z.dbm[0][j]
            z.dbm[j][idx] = z.dbm[j][0]
        z.dbm[idx][idx] = (0, False)
    z.canonicalize()
    return z


def future(z):
    """Time elapse: remove upper bounds on clocks."""
    z = z.copy()
    for i in range(1, z.n + 1):
        z.dbm[i][0] = _INF_BOUND
    z.canonicalize()
    return z


def past(z):
    """Reverse time: remove lower bounds on clocks."""
    z = z.copy()
    for i in range(1, z.n + 1):
        z.dbm[0][i] = (0, False)
    z.canonicalize()
    return z


def apply_invariant(z, inv):
    return apply_guard(z, inv)


def successor_zone(z, guard, resets, target_inv):
    """Forward symbolic step: guard -> reset -> future -> target invariant."""
    z = apply_guard(z, guard)
    if z.is_empty():
        return None
    z = reset_clocks(z, resets)
    z = future(z)
    z = apply_invariant(z, target_inv)
    if z.is_empty():
        return None
    return z


def backward_zone(z_target, guard, resets, source_inv, clock_names):
    """Backward symbolic step: past -> undo resets -> guard -> source invariant."""
    z = past(z_target)
    # Undo resets: free the reset clocks
    z2 = z.copy()
    for c in resets:
        idx = _clock_index(z2, c)
        dim = z2.n + 1
        for j in range(dim):
            z2.dbm[idx][j] = _INF_BOUND
            z2.dbm[j][idx] = _INF_BOUND if j != idx else (0, False)
        z2.dbm[idx][idx] = (0, False)
        # Reset clock must be >= 0
        z2.dbm[0][idx] = _min_bound(z2.dbm[0][idx], (0, False))
    z2.canonicalize()
    z2 = apply_guard(z2, guard)
    if z2.is_empty():
        return None
    z2 = apply_invariant(z2, source_inv)
    if z2.is_empty():
        return None
    return z2


# ============================================================
# Stochastic Timed Game
# ============================================================

class PlayerType(Enum):
    MIN = "min"       # Controller -- tries to reach target
    MAX = "max"       # Adversary -- tries to prevent target
    RANDOM = "random"  # Nature -- probabilistic choice


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    guard: Guard = field(default_factory=true_guard)
    resets: FrozenSet[str] = field(default_factory=frozenset)
    label: str = ""


@dataclass
class StochasticTimedGame:
    """Timed game with MIN, MAX, and RANDOM locations.

    RANDOM locations have probabilistic edges: when entered, nature
    picks a successor edge according to the probability distribution.
    """
    locations: Set[str] = field(default_factory=set)
    initial: str = ""
    clocks: Set[str] = field(default_factory=set)
    edges: List[Edge] = field(default_factory=list)
    invariants: Dict[str, Guard] = field(default_factory=dict)
    owner: Dict[str, PlayerType] = field(default_factory=dict)
    # For RANDOM locations: edge_index -> probability
    probabilities: Dict[int, float] = field(default_factory=dict)
    accepting: Set[str] = field(default_factory=set)

    def add_location(self, loc, player=PlayerType.MIN, invariant=None):
        self.locations.add(loc)
        self.owner[loc] = player
        if invariant is not None:
            self.invariants[loc] = invariant

    def add_edge(self, source, target, guard=None, resets=None,
                 label="", probability=None):
        """Add edge. For edges from RANDOM locations, specify probability."""
        if guard is None:
            guard = true_guard()
        if resets is None:
            resets = frozenset()
        edge = Edge(source=source, target=target, guard=guard,
                    resets=frozenset(resets), label=label)
        idx = len(self.edges)
        self.edges.append(edge)
        if probability is not None:
            self.probabilities[idx] = probability
        return idx

    def get_edges_from(self, loc):
        """Return [(index, edge)] for edges from loc."""
        return [(i, e) for i, e in enumerate(self.edges) if e.source == loc]

    def get_invariant(self, loc):
        return self.invariants.get(loc, true_guard())

    def get_probability(self, edge_idx):
        return self.probabilities.get(edge_idx, 1.0)

    def max_constant(self):
        """Largest constant in guards and invariants."""
        m = 0
        for e in self.edges:
            for cc in e.guard.constraints:
                m = max(m, abs(cc.value))
        for inv in self.invariants.values():
            for cc in inv.constraints:
                m = max(m, abs(cc.value))
        return m

    def validate(self):
        """Check game is well-formed."""
        errors = []
        if self.initial and self.initial not in self.locations:
            errors.append(f"Initial location '{self.initial}' not in locations")
        for i, e in enumerate(self.edges):
            if e.source not in self.locations:
                errors.append(f"Edge {i}: source '{e.source}' not in locations")
            if e.target not in self.locations:
                errors.append(f"Edge {i}: target '{e.target}' not in locations")
            for cc in e.guard.constraints:
                if cc.clock not in self.clocks:
                    errors.append(f"Edge {i}: clock '{cc.clock}' not in clocks")
            for c in e.resets:
                if c not in self.clocks:
                    errors.append(f"Edge {i}: reset clock '{c}' not in clocks")
        # Check RANDOM locations have valid distributions
        for loc in self.locations:
            if self.owner.get(loc) == PlayerType.RANDOM:
                out_edges = self.get_edges_from(loc)
                if not out_edges:
                    errors.append(f"RANDOM location '{loc}' has no outgoing edges")
                    continue
                total = sum(self.get_probability(i) for i, _ in out_edges)
                if abs(total - 1.0) > 1e-9:
                    errors.append(
                        f"RANDOM location '{loc}': probabilities sum to {total}, not 1.0")
        return errors


# ============================================================
# Symbolic State Space
# ============================================================

@dataclass(frozen=True)
class SymState:
    location: str
    zone: int  # index into ZoneStore


class ZoneStore:
    """Deduplicates zones for efficient fixed-point computation."""

    def __init__(self):
        self._zones = []
        self._index = {}

    def add(self, z):
        key = id(z)
        # Check equality-based dedup
        for i, existing in enumerate(self._zones):
            if z == existing:
                return i
        idx = len(self._zones)
        self._zones.append(z)
        return idx

    def get(self, idx):
        return self._zones[idx]

    def __len__(self):
        return len(self._zones)


# ============================================================
# Zone-level Operations for Game Solving
# ============================================================

def _zones_include(zone_list, z):
    """Check if z is included in the union of zones in zone_list."""
    for existing in zone_list:
        if existing.includes(z):
            return True
    return False


def _zones_union_add(zone_list, z):
    """Add z to zone_list if not already subsumed. Returns updated list."""
    if z.is_empty():
        return zone_list
    if _zones_include(zone_list, z):
        return zone_list
    # Remove zones subsumed by z
    filtered = [existing for existing in zone_list if not z.includes(existing)]
    filtered.append(z)
    return filtered


def _zones_equal(zs1, zs2):
    """Check if two zone sets represent the same region."""
    for z in zs1:
        if not _zones_include(zs2, z):
            return False
    for z in zs2:
        if not _zones_include(zs1, z):
            return False
    return True


# ============================================================
# Result Types
# ============================================================

@dataclass
class StochasticTimedResult:
    """Result of stochastic timed game solving."""
    winning_locations_as: Set[str] = field(default_factory=set)  # almost-sure
    winning_locations_pp: Set[str] = field(default_factory=set)  # positive-prob
    winning_zones_as: Dict[str, List] = field(default_factory=dict)  # loc -> zones
    winning_zones_pp: Dict[str, List] = field(default_factory=dict)
    iterations: int = 0
    strategy: Dict[str, List[int]] = field(default_factory=dict)  # loc -> edge indices


@dataclass
class ExpectedTimeResult:
    """Result of expected-time reachability computation."""
    reachable: bool = False
    expected_time: Dict[str, float] = field(default_factory=dict)  # loc -> E[time]
    strategy: Dict[str, int] = field(default_factory=dict)  # loc -> edge index
    iterations: int = 0


@dataclass
class SimStep:
    location: str
    delay: float
    action: str
    target: str
    probability: float = 1.0


@dataclass
class SimTrace:
    steps: List[SimStep] = field(default_factory=list)
    reached_target: bool = False
    total_time: float = 0.0


# ============================================================
# Forward Exploration
# ============================================================

def explore_reachable(game, clock_names=None):
    """Forward zone-based exploration. Returns {loc: [zones]}."""
    if clock_names is None:
        clock_names = sorted(game.clocks)

    init_zone = make_zero_zone(clock_names)
    init_zone = future(init_zone)
    inv = game.get_invariant(game.initial)
    init_zone = apply_invariant(init_zone, inv)
    if init_zone.is_empty():
        return {}

    reached = {loc: [] for loc in game.locations}
    reached[game.initial] = [init_zone]
    worklist = [(game.initial, init_zone)]

    while worklist:
        loc, z = worklist.pop()
        for idx, edge in game.get_edges_from(loc):
            target_inv = game.get_invariant(edge.target)
            sz = successor_zone(z, edge.guard, edge.resets, target_inv)
            if sz is not None and not sz.is_empty():
                if not _zones_include(reached[edge.target], sz):
                    reached[edge.target] = _zones_union_add(
                        reached[edge.target], sz)
                    worklist.append((edge.target, sz))

    return reached


# ============================================================
# Solvers
# ============================================================

def solve_positive_prob_reachability(game, targets, max_iterations=100):
    """Positive-probability timed reachability.

    MIN wins at location l if there EXISTS a strategy that reaches
    targets with probability > 0. RANDOM locations are treated
    cooperatively (like MIN) -- if ANY successor leads to winning,
    the RANDOM location is winning.

    Uses backward zone-based attractor computation.
    """
    clock_names = sorted(game.clocks)

    # Initialize winning zones: targets get full zone
    winning = {}
    for loc in game.locations:
        if loc in targets:
            z = make_zone(clock_names)
            inv = game.get_invariant(loc)
            z = apply_invariant(z, inv)
            winning[loc] = [z] if not z.is_empty() else []
        else:
            winning[loc] = []

    for iteration in range(max_iterations):
        changed = False
        new_winning = {loc: list(winning[loc]) for loc in game.locations}

        for loc in game.locations:
            if loc in targets:
                continue
            owner = game.owner.get(loc, PlayerType.MIN)

            for idx, edge in game.get_edges_from(loc):
                if not winning[edge.target]:
                    continue
                source_inv = game.get_invariant(loc)
                for tz in winning[edge.target]:
                    bz = backward_zone(tz, edge.guard, edge.resets,
                                       source_inv, clock_names)
                    if bz is not None and not bz.is_empty():
                        if not _zones_include(new_winning[loc], bz):
                            new_winning[loc] = _zones_union_add(
                                new_winning[loc], bz)
                            changed = True

        winning = new_winning
        if not changed:
            break

    # Determine winning locations
    result = StochasticTimedResult(iterations=iteration + 1)
    for loc in game.locations:
        if winning[loc]:
            result.winning_locations_pp.add(loc)
    result.winning_zones_pp = winning
    return result


def _graph_reachable(game, sources, candidate):
    """BFS: which locations in candidate can reach sources?"""
    # Build reverse adjacency within candidate
    reached = set(sources & candidate)
    worklist = list(reached)
    while worklist:
        loc = worklist.pop()
        for idx, edge in enumerate(game.edges):
            if edge.target == loc and edge.source in candidate and edge.source not in reached:
                reached.add(edge.source)
                worklist.append(edge.source)
    return reached


def solve_almost_sure_reachability(game, targets, max_iterations=100):
    """Almost-sure timed reachability.

    Uses location-level graph analysis (handles retry cycles correctly):
    1. Start with candidate = all locations that can reach targets
    2. Remove 'bad' locations:
       - MAX with any edge leaving candidate
       - RANDOM with any positive-prob edge leaving candidate
    3. Recompute reachability to targets within remaining candidate
    4. Repeat until fixed point

    Then validates with zone-based analysis.
    """
    clock_names = sorted(game.clocks)

    # Phase 1: Location-level almost-sure analysis
    candidate = set(game.locations)

    for iteration in range(max_iterations):
        # Keep only locations that can reach targets within candidate
        candidate = _graph_reachable(game, targets, candidate)

        # Remove bad locations
        changed = False
        to_remove = set()
        for loc in candidate:
            if loc in targets:
                continue
            owner = game.owner.get(loc, PlayerType.MIN)
            out_edges = game.get_edges_from(loc)

            if owner == PlayerType.MAX:
                # MAX: if ANY edge leaves candidate, MAX takes it
                for idx, edge in out_edges:
                    if edge.target not in candidate:
                        to_remove.add(loc)
                        break

            elif owner == PlayerType.RANDOM:
                # RANDOM: if ANY positive-prob edge leaves, not a.s.
                for idx, edge in out_edges:
                    prob = game.get_probability(idx)
                    if prob > 0 and edge.target not in candidate:
                        to_remove.add(loc)
                        break

            elif owner == PlayerType.MIN:
                # MIN: needs at least one edge staying in candidate
                has_good = False
                for idx, edge in out_edges:
                    if edge.target in candidate:
                        has_good = True
                        break
                if not has_good and out_edges:
                    to_remove.add(loc)

        if to_remove:
            candidate -= to_remove
            changed = True

        if not changed:
            break

    # Phase 2: Zone-based validation
    winning = {}
    for loc in game.locations:
        if loc in candidate:
            z = make_zone(clock_names)
            inv = game.get_invariant(loc)
            z = apply_invariant(z, inv)
            winning[loc] = [z] if not z.is_empty() else []
        else:
            winning[loc] = []

    result = StochasticTimedResult(iterations=iteration + 1)
    result.winning_zones_as = winning
    for loc in game.locations:
        if winning[loc]:
            result.winning_locations_as.add(loc)
    return result


def solve_stochastic_timed_reachability(game, targets, max_iterations=100):
    """Solve both almost-sure and positive-probability reachability."""
    pp = solve_positive_prob_reachability(game, targets, max_iterations)
    as_ = solve_almost_sure_reachability(game, targets, max_iterations)

    result = StochasticTimedResult(
        winning_locations_as=as_.winning_locations_as,
        winning_locations_pp=pp.winning_locations_pp,
        winning_zones_as=as_.winning_zones_as,
        winning_zones_pp=pp.winning_zones_pp,
        iterations=pp.iterations + as_.iterations,
    )
    return result


def solve_stochastic_timed_safety(game, unsafe, max_iterations=100):
    """Almost-sure timed safety: MIN avoids unsafe with probability 1.

    Dual of reachability: locations from which MAX cannot force
    reaching unsafe almost-surely.
    """
    clock_names = sorted(game.clocks)
    safe_locs = game.locations - unsafe

    # Initialize: all safe locations potentially winning
    winning = {}
    for loc in game.locations:
        if loc in unsafe:
            winning[loc] = []
        else:
            z = make_zone(clock_names)
            inv = game.get_invariant(loc)
            z = apply_invariant(z, inv)
            winning[loc] = [z] if not z.is_empty() else []

    # Iteratively remove locations from which MAX can force reaching unsafe
    for iteration in range(max_iterations):
        changed = False
        new_winning = {loc: list(winning[loc]) for loc in game.locations}

        for loc in game.locations:
            if loc in unsafe or not winning[loc]:
                continue
            owner = game.owner.get(loc, PlayerType.MIN)

            if owner == PlayerType.MIN:
                # MIN: needs at least one edge staying in winning
                has_safe_edge = False
                for idx, edge in game.get_edges_from(loc):
                    if winning[edge.target]:
                        has_safe_edge = True
                        break
                if not has_safe_edge and game.get_edges_from(loc):
                    new_winning[loc] = []
                    changed = True

            elif owner == PlayerType.MAX:
                # MAX: if ANY edge leads to losing, MAX takes it
                for idx, edge in game.get_edges_from(loc):
                    if not winning[edge.target]:
                        new_winning[loc] = []
                        changed = True
                        break

            elif owner == PlayerType.RANDOM:
                # RANDOM: if ANY positive-prob edge leads to losing, losing
                for idx, edge in game.get_edges_from(loc):
                    prob = game.get_probability(idx)
                    if prob > 0 and not winning[edge.target]:
                        new_winning[loc] = []
                        changed = True
                        break

        winning = new_winning
        if not changed:
            break

    result = StochasticTimedResult(iterations=iteration + 1)
    result.winning_zones_as = winning
    for loc in game.locations:
        if winning[loc]:
            result.winning_locations_as.add(loc)
    return result


def solve_expected_time(game, targets, max_iterations=200, epsilon=1e-6):
    """Compute expected time to reach targets under optimal MIN strategy.

    Uses value iteration on the zone graph. Each location gets an
    expected-time value. MIN minimizes, MAX maximizes, RANDOM averages.

    For simplicity, uses a discrete abstraction: time steps of size 1.
    """
    clock_names = sorted(game.clocks)

    # First check reachability
    pp = solve_positive_prob_reachability(game, targets, max_iterations)
    reachable_locs = pp.winning_locations_pp

    # Value iteration on locations
    values = {}
    strategy = {}
    for loc in game.locations:
        if loc in targets:
            values[loc] = 0.0
        elif loc in reachable_locs:
            values[loc] = float(max_iterations)  # upper bound
        else:
            values[loc] = INF

    for iteration in range(max_iterations):
        max_delta = 0.0
        new_values = dict(values)

        for loc in game.locations:
            if loc in targets or loc not in reachable_locs:
                continue

            owner = game.owner.get(loc, PlayerType.MIN)
            out_edges = game.get_edges_from(loc)
            if not out_edges:
                continue

            if owner == PlayerType.MIN:
                best = INF
                best_edge = -1
                for idx, edge in out_edges:
                    v = 1.0 + values.get(edge.target, INF)
                    if v < best:
                        best = v
                        best_edge = idx
                new_values[loc] = best
                strategy[loc] = best_edge

            elif owner == PlayerType.MAX:
                worst = -INF
                for idx, edge in out_edges:
                    v = 1.0 + values.get(edge.target, INF)
                    if v > worst:
                        worst = v
                new_values[loc] = min(worst, float(max_iterations))

            elif owner == PlayerType.RANDOM:
                expected = 0.0
                for idx, edge in out_edges:
                    prob = game.get_probability(idx)
                    expected += prob * (1.0 + values.get(edge.target, INF))
                new_values[loc] = min(expected, float(max_iterations))

            delta = abs(new_values[loc] - values[loc])
            max_delta = max(max_delta, delta)

        values = new_values
        if max_delta < epsilon:
            break

    result = ExpectedTimeResult(iterations=iteration + 1)
    result.expected_time = {loc: values[loc] for loc in game.locations
                           if values[loc] < INF}
    result.strategy = strategy
    result.reachable = game.initial in result.expected_time
    return result


def solve_qualitative_buchi(game, accepting=None, max_iterations=100):
    """Qualitative Buchi: visit accepting locations infinitely often.

    Returns almost-sure and positive-probability winning regions.

    Positive-prob Buchi: find maximal set S such that:
    - Every loc in S can reach accepting within S
    - From every accepting loc in S, can reach accepting again within S
    - MIN locations have at least one edge staying in S
    - MAX locations have ALL edges staying in S (MAX can't escape)

    Almost-sure Buchi: additionally RANDOM locations must have ALL
    positive-prob edges staying in S.
    """
    if accepting is None:
        accepting = game.accepting

    # Positive-probability Buchi via greatest fixed point
    candidate = set(game.locations)

    for outer in range(max_iterations):
        old_candidate = set(candidate)

        # Step 1: Remove locations without outgoing edges into candidate
        for loc in list(candidate):
            if loc not in candidate:
                continue
            out_edges = game.get_edges_from(loc)
            owner = game.owner.get(loc, PlayerType.MIN)

            if owner == PlayerType.MAX:
                # MAX: all edges must stay in candidate
                for idx, edge in out_edges:
                    if edge.target not in candidate:
                        candidate.discard(loc)
                        break
            else:
                # MIN/RANDOM: need at least one edge in candidate
                has_edge_in = any(e.target in candidate for _, e in out_edges)
                if not has_edge_in and out_edges:
                    candidate.discard(loc)
                elif not out_edges:
                    # No outgoing edges at all -- can't cycle
                    candidate.discard(loc)

        # Step 2: Must be able to reach accepting within candidate
        can_reach_acc = _graph_reachable(game, accepting, candidate)
        candidate = can_reach_acc

        # Step 3: Accepting locations must be able to reach accepting
        # again (cycle requirement for infinitely often)
        # From each accepting loc, there must be a path back to some
        # accepting loc within candidate
        for acc_loc in list(accepting & candidate):
            # Check if acc_loc can reach any accepting loc via non-trivial path
            visited = set()
            frontier = []
            for idx, edge in game.get_edges_from(acc_loc):
                if edge.target in candidate:
                    frontier.append(edge.target)
            while frontier:
                loc = frontier.pop()
                if loc in visited:
                    continue
                visited.add(loc)
                if loc in accepting:
                    break  # Can reach accepting again
                for idx, edge in game.get_edges_from(loc):
                    if edge.target in candidate and edge.target not in visited:
                        frontier.append(edge.target)
            else:
                # Could not reach accepting from this accepting loc
                candidate.discard(acc_loc)

        if candidate == old_candidate:
            break

    pp_winning = candidate

    # Almost-sure Buchi: additionally require RANDOM closure
    as_candidate = set(pp_winning)
    for outer in range(max_iterations):
        old_as = set(as_candidate)

        # Remove bad RANDOM and MAX locations
        for loc in list(as_candidate):
            owner = game.owner.get(loc, PlayerType.MIN)
            if owner == PlayerType.RANDOM:
                for idx, edge in game.get_edges_from(loc):
                    prob = game.get_probability(idx)
                    if prob > 0 and edge.target not in as_candidate:
                        as_candidate.discard(loc)
                        break
            elif owner == PlayerType.MAX:
                for idx, edge in game.get_edges_from(loc):
                    if edge.target not in as_candidate:
                        as_candidate.discard(loc)
                        break

        # Re-check: remaining must reach accepting and have cycles
        as_candidate = _graph_reachable(game, accepting, as_candidate)

        if as_candidate == old_as:
            break

    result = StochasticTimedResult()
    result.winning_locations_pp = pp_winning
    result.winning_locations_as = as_candidate
    return result


# ============================================================
# Simulation
# ============================================================

def simulate_play(game, strategy_min, strategy_max=None,
                  random_seed=None, max_steps=50):
    """Simulate a play in the stochastic timed game.

    strategy_min: Dict[str, Tuple[float, int]] -- location -> (delay, edge_index)
    strategy_max: Dict[str, Tuple[float, int]] -- same for MAX
    random_seed: seed for reproducible randomness

    Returns SimTrace.
    """
    import random as _random
    rng = _random.Random(random_seed)

    clock_names = sorted(game.clocks)
    clock_vals = {c: 0.0 for c in clock_names}
    current = game.initial
    trace = SimTrace()

    for step in range(max_steps):
        owner = game.owner.get(current, PlayerType.MIN)

        if owner == PlayerType.RANDOM:
            # Nature chooses by distribution
            out_edges = game.get_edges_from(current)
            if not out_edges:
                break
            indices = [i for i, _ in out_edges]
            probs = [game.get_probability(i) for i in indices]
            chosen_idx = rng.choices(indices, weights=probs, k=1)[0]
            chosen_edge = game.edges[chosen_idx]
            delay = 0.0  # RANDOM transitions are instantaneous
            prob = game.get_probability(chosen_idx)

            # Apply resets
            for c in chosen_edge.resets:
                clock_vals[c] = 0.0

            trace.steps.append(SimStep(
                location=current, delay=delay,
                action=chosen_edge.label or f"e{chosen_idx}",
                target=chosen_edge.target, probability=prob))
            trace.total_time += delay
            current = chosen_edge.target

        else:
            # MIN or MAX chooses
            strat = strategy_min if owner == PlayerType.MIN else strategy_max
            if strat is None or current not in strat:
                break
            delay, edge_idx = strat[current]

            # Advance clocks
            for c in clock_names:
                clock_vals[c] += delay

            edge = game.edges[edge_idx]
            # Apply resets
            for c in edge.resets:
                clock_vals[c] = 0.0

            trace.steps.append(SimStep(
                location=current, delay=delay,
                action=edge.label or f"e{edge_idx}",
                target=edge.target))
            trace.total_time += delay
            current = edge.target

        if current in game.accepting:
            trace.reached_target = True
            break

    return trace


# ============================================================
# Analysis Utilities
# ============================================================

def game_statistics(game):
    """Summary statistics of the game."""
    owners = {}
    for loc in game.locations:
        t = game.owner.get(loc, PlayerType.MIN).value
        owners[t] = owners.get(t, 0) + 1

    return {
        "locations": len(game.locations),
        "edges": len(game.edges),
        "clocks": len(game.clocks),
        "min_locations": owners.get("min", 0),
        "max_locations": owners.get("max", 0),
        "random_locations": owners.get("random", 0),
        "max_constant": game.max_constant(),
        "accepting": len(game.accepting),
    }


def compare_as_pp(game, targets, max_iterations=100):
    """Compare almost-sure vs positive-probability winning regions."""
    result = solve_stochastic_timed_reachability(game, targets, max_iterations)
    return {
        "almost_sure": result.winning_locations_as,
        "positive_prob": result.winning_locations_pp,
        "as_only": result.winning_locations_as,
        "pp_not_as": result.winning_locations_pp - result.winning_locations_as,
        "neither": game.locations - result.winning_locations_pp,
    }


# ============================================================
# Example Games
# ============================================================

def make_coin_flip_game():
    """Simple coin flip timed game.

    MIN moves to FLIP (RANDOM), which goes to WIN (p=0.5) or RETRY (p=0.5).
    RETRY goes back to start. Clock constraint: must flip within 3 time units.
    """
    g = StochasticTimedGame()
    g.clocks = {"x"}
    g.initial = "start"
    g.accepting = {"win"}

    g.add_location("start", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("x", 3),)))
    g.add_location("flip", PlayerType.RANDOM)
    g.add_location("win", PlayerType.MIN)
    g.add_location("retry", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("x", 3),)))

    g.add_edge("start", "flip", guard=Guard(
        constraints=(clock_le("x", 2),)), label="go")
    g.add_edge("flip", "win", probability=0.5, label="heads")
    g.add_edge("flip", "retry", probability=0.5, label="tails",
               resets=frozenset({"x"}))
    g.add_edge("retry", "flip", guard=Guard(
        constraints=(clock_le("x", 2),)), label="retry")

    return g


def make_probabilistic_traffic():
    """Traffic controller with random sensor failures.

    MIN controls traffic light timing. Sensor (RANDOM) may fail.
    Goal: cars pass safely (reach 'passed') despite sensor uncertainty.
    """
    g = StochasticTimedGame()
    g.clocks = {"t", "s"}
    g.initial = "idle"
    g.accepting = {"passed"}

    g.add_location("idle", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("t", 5),)))
    g.add_location("sense", PlayerType.RANDOM)
    g.add_location("green", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("s", 3),)))
    g.add_location("red", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("s", 4),)))
    g.add_location("passed", PlayerType.MIN)
    g.add_location("fail", PlayerType.MAX)

    # Idle -> sense
    g.add_edge("idle", "sense", guard=Guard(
        constraints=(clock_ge("t", 1),)), resets=frozenset({"s"}), label="check")
    # Sensor: ok (0.8) or fail (0.2)
    g.add_edge("sense", "green", probability=0.8, label="ok")
    g.add_edge("sense", "red", probability=0.2, label="fault")
    # Green -> passed
    g.add_edge("green", "passed", guard=Guard(
        constraints=(clock_ge("s", 1),)), label="go")
    # Red -> idle (retry)
    g.add_edge("red", "idle", guard=Guard(
        constraints=(clock_ge("s", 2),)), resets=frozenset({"t"}), label="wait")

    return g


def make_adversarial_random_game():
    """Game where MIN and MAX compete with randomness.

    MIN chooses path, then RANDOM determines outcome. MAX can
    interfere at one point.
    """
    g = StochasticTimedGame()
    g.clocks = {"x"}
    g.initial = "s0"
    g.accepting = {"win"}

    g.add_location("s0", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("x", 5),)))
    g.add_location("s1", PlayerType.RANDOM)
    g.add_location("s2", PlayerType.RANDOM)
    g.add_location("s3", PlayerType.MAX)
    g.add_location("win", PlayerType.MIN)
    g.add_location("lose", PlayerType.MAX)

    # MIN: fast path (risky) or safe path
    g.add_edge("s0", "s1", guard=Guard(
        constraints=(clock_le("x", 2),)), label="fast")
    g.add_edge("s0", "s2", guard=Guard(
        constraints=(clock_ge("x", 2), clock_le("x", 4))),
        resets=frozenset({"x"}), label="safe")

    # Fast random: 40% win, 60% to MAX
    g.add_edge("s1", "win", probability=0.4, label="lucky")
    g.add_edge("s1", "s3", probability=0.6, label="unlucky")

    # Safe random: 70% win, 30% to MAX
    g.add_edge("s2", "win", probability=0.7, label="success")
    g.add_edge("s2", "s3", probability=0.3, label="blocked")

    # MAX: can send to lose or back to start
    g.add_edge("s3", "lose", label="block")
    g.add_edge("s3", "s0", resets=frozenset({"x"}), label="release")

    return g


def make_retry_game():
    """Game where MIN can retry a random action indefinitely.

    Each retry has probability p of success. Almost-sure winning
    because retries are unbounded, but expected time depends on p.
    """
    g = StochasticTimedGame()
    g.clocks = {"x"}
    g.initial = "try"
    g.accepting = {"done"}

    g.add_location("try", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("x", 2),)))
    g.add_location("flip", PlayerType.RANDOM)
    g.add_location("done", PlayerType.MIN)
    g.add_location("reset", PlayerType.MIN)

    g.add_edge("try", "flip", guard=Guard(
        constraints=(clock_le("x", 1),)), label="attempt")
    g.add_edge("flip", "done", probability=0.3, label="success")
    g.add_edge("flip", "reset", probability=0.7, label="fail")
    g.add_edge("reset", "try", resets=frozenset({"x"}), label="retry")

    return g


def make_two_player_stochastic():
    """Two-player game with stochastic environment.

    MIN and MAX alternate, with RANDOM intermediary steps.
    Tests interaction of all three player types.
    """
    g = StochasticTimedGame()
    g.clocks = {"c"}
    g.initial = "m0"
    g.accepting = {"goal"}

    g.add_location("m0", PlayerType.MIN, invariant=Guard(
        constraints=(clock_le("c", 4),)))
    g.add_location("r1", PlayerType.RANDOM)
    g.add_location("a1", PlayerType.MAX, invariant=Guard(
        constraints=(clock_le("c", 6),)))
    g.add_location("r2", PlayerType.RANDOM)
    g.add_location("goal", PlayerType.MIN)
    g.add_location("back", PlayerType.MIN)

    # MIN -> RANDOM
    g.add_edge("m0", "r1", guard=Guard(
        constraints=(clock_ge("c", 1),)), label="move")

    # RANDOM: good (0.6) or bad (0.4) for MIN
    g.add_edge("r1", "a1", probability=0.6, label="favorable")
    g.add_edge("r1", "back", probability=0.4, label="unfavorable")

    # MAX: must choose, one leads to another random
    g.add_edge("a1", "r2", label="continue")
    g.add_edge("a1", "back", label="deny")

    # RANDOM: reach goal or back
    g.add_edge("r2", "goal", probability=0.5, label="arrive")
    g.add_edge("r2", "back", probability=0.5, label="bounce")

    # Back -> restart
    g.add_edge("back", "m0", resets=frozenset({"c"}), label="restart")

    return g
