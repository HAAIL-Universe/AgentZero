"""
V206: Weighted Timed Games -- Min-Cost Reachability over Timed Games

Composes:
  - V202 (timed games): zone-based state space, clock guards, resets
  - V160 (energy games): weighted edges, cost optimization

Key concepts:
  - Priced timed games: edges have discrete costs, locations have rate costs
  - Priced zones: zones augmented with linear cost functions over clock valuations
  - Min-cost reachability: find strategy minimizing cost to reach target
  - Cost-bounded reachability: can player reach target within budget?
  - Pareto-optimal strategies: time vs cost tradeoffs

Theory: Alur, La Torre, Pappas (2004) "Optimal Paths in Weighted Timed Automata"
        Bouyer, Cassez, Fleury, Larsen (2004) "Optimal Strategies in Priced Timed Game Automata"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Set, List, Tuple, FrozenSet
import math


# ============================================================
# Core data structures
# ============================================================

class Player(Enum):
    MIN = auto()   # Minimizer (wants low cost to reach target)
    MAX = auto()   # Maximizer (wants to prevent or raise cost)


class CompOp(Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="


@dataclass(frozen=True)
class ClockConstraint:
    """Single clock constraint: clock op value, or clock1 - clock2 op value."""
    clock1: str
    clock2: Optional[str]  # None for simple constraint
    op: CompOp
    value: int

    def satisfied(self, valuation: Dict[str, float]) -> bool:
        v1 = valuation.get(self.clock1, 0.0)
        v2 = valuation.get(self.clock2, 0.0) if self.clock2 else 0.0
        diff = v1 - v2
        if self.op == CompOp.LT: return diff < self.value
        if self.op == CompOp.LE: return diff <= self.value
        if self.op == CompOp.GT: return diff > self.value
        if self.op == CompOp.GE: return diff >= self.value
        if self.op == CompOp.EQ: return abs(diff - self.value) < 1e-9
        return False


@dataclass(frozen=True)
class Guard:
    """Conjunction of clock constraints."""
    constraints: Tuple[ClockConstraint, ...]

    def satisfied(self, valuation: Dict[str, float]) -> bool:
        return all(c.satisfied(valuation) for c in self.constraints)

    @staticmethod
    def true_guard() -> Guard:
        return Guard(())


@dataclass(frozen=True)
class Edge:
    """Weighted timed game edge."""
    source: str
    target: str
    label: str
    guard: Guard
    resets: FrozenSet[str]
    cost: int  # discrete transition cost (>= 0)

    def __repr__(self):
        return f"Edge({self.source}->{self.target}, '{self.label}', cost={self.cost})"


@dataclass
class WeightedTimedGame:
    """
    Priced timed game automaton.

    Locations have:
      - owner (MIN or MAX player)
      - clock invariants
      - rate costs (cost per time unit spent in location)

    Edges have:
      - clock guards
      - clock resets
      - discrete transition costs
    """
    locations: Set[str] = field(default_factory=set)
    initial: str = ""
    clocks: Set[str] = field(default_factory=set)
    edges: List[Edge] = field(default_factory=list)
    invariants: Dict[str, Guard] = field(default_factory=dict)
    owner: Dict[str, Player] = field(default_factory=dict)
    accepting: Set[str] = field(default_factory=set)  # target locations
    rate_cost: Dict[str, int] = field(default_factory=dict)  # cost per time unit

    def add_location(self, loc: str, player: Player = Player.MIN,
                     invariant: Optional[Guard] = None,
                     rate: int = 0):
        self.locations.add(loc)
        self.owner[loc] = player
        if invariant:
            self.invariants[loc] = invariant
        self.rate_cost[loc] = rate

    def add_edge(self, source: str, target: str, label: str = "",
                 guard: Optional[Guard] = None,
                 resets: Optional[FrozenSet[str]] = None,
                 cost: int = 0) -> int:
        e = Edge(source, target, label,
                 guard or Guard.true_guard(),
                 resets or frozenset(), cost)
        self.edges.append(e)
        return len(self.edges) - 1

    def get_edges_from(self, loc: str) -> List[Tuple[int, Edge]]:
        return [(i, e) for i, e in enumerate(self.edges) if e.source == loc]

    def max_constant(self) -> int:
        """Largest constant in guards/invariants."""
        m = 0
        for e in self.edges:
            for c in e.guard.constraints:
                m = max(m, abs(c.value))
        for inv in self.invariants.values():
            for c in inv.constraints:
                m = max(m, abs(c.value))
        return max(m, 1)


# ============================================================
# DBM (Difference Bound Matrix) Zone representation
# ============================================================

INF = float('inf')


@dataclass
class Zone:
    """
    DBM-based zone. Clocks indexed 0..n where 0 is the reference clock (=0).
    dbm[i][j] = (bound, strict) means clock_i - clock_j <= bound (or < bound if strict).
    """
    n: int  # number of clocks (not counting reference)
    clock_names: List[str]
    dbm: List[List[Tuple[float, bool]]]  # (bound, strict): True = strict (<)

    def copy(self) -> Zone:
        new_dbm = [row[:] for row in self.dbm]
        return Zone(self.n, self.clock_names[:], new_dbm)

    def is_empty(self) -> bool:
        for i in range(self.n + 1):
            b, s = self.dbm[i][i]
            if b < 0 or (b == 0 and s):
                return True
        return False

    def clock_index(self, name: str) -> int:
        return self.clock_names.index(name) + 1  # 0 is reference

    def _add_bounds(self, b1: Tuple[float, bool],
                    b2: Tuple[float, bool]) -> Tuple[float, bool]:
        v1, s1 = b1
        v2, s2 = b2
        if v1 == INF or v2 == INF:
            return (INF, True)
        return (v1 + v2, s1 or s2)

    def _min_bound(self, b1: Tuple[float, bool],
                   b2: Tuple[float, bool]) -> Tuple[float, bool]:
        v1, s1 = b1
        v2, s2 = b2
        if v1 < v2:
            return b1
        if v2 < v1:
            return b2
        return (v1, s1 or s2)  # same value: stricter wins

    def canonicalize(self):
        """Floyd-Warshall shortest path closure."""
        dim = self.n + 1
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    via_k = self._add_bounds(self.dbm[i][k], self.dbm[k][j])
                    self.dbm[i][j] = self._min_bound(self.dbm[i][j], via_k)

    def includes(self, other: Zone) -> bool:
        """Does self include other? (other subset of self)"""
        dim = self.n + 1
        for i in range(dim):
            for j in range(dim):
                sv, ss = self.dbm[i][j]
                ov, os = other.dbm[i][j]
                if ov < sv:
                    continue
                if ov > sv:
                    return False
                if os and not ss:  # other is strict, self is not
                    continue
                if ss and not os:  # self is strict, other is not
                    return False
        return True

    def intersect(self, other: Zone) -> Zone:
        dim = self.n + 1
        z = self.copy()
        for i in range(dim):
            for j in range(dim):
                z.dbm[i][j] = self._min_bound(self.dbm[i][j], other.dbm[i][j])
        z.canonicalize()
        return z

    def __eq__(self, other):
        if not isinstance(other, Zone):
            return False
        return self.n == other.n and self.dbm == other.dbm

    def __hash__(self):
        return hash((self.n, tuple(tuple(row) for row in self.dbm)))


def make_zone(clock_names: List[str]) -> Zone:
    """Create unconstrained zone (all clocks >= 0)."""
    n = len(clock_names)
    dim = n + 1
    dbm = [[(INF, True)] * dim for _ in range(dim)]
    for i in range(dim):
        dbm[i][i] = (0, False)
    # All clocks >= 0: clock_i - ref >= 0, i.e., ref - clock_i <= 0
    for i in range(1, dim):
        dbm[0][i] = (0, False)
    return Zone(n, clock_names[:], dbm)


def make_zero_zone(clock_names: List[str]) -> Zone:
    """Zone where all clocks = 0."""
    n = len(clock_names)
    dim = n + 1
    dbm = [[(0, False)] * dim for _ in range(dim)]
    return Zone(n, clock_names[:], dbm)


def constrain_zone(z: Zone, constraint: ClockConstraint) -> Zone:
    """Apply a constraint to a zone."""
    zc = z.copy()
    i = z.clock_index(constraint.clock1) if constraint.clock1 else 0
    j = z.clock_index(constraint.clock2) if constraint.clock2 else 0
    strict = constraint.op in (CompOp.LT, CompOp.GT)
    val = constraint.value

    if constraint.op in (CompOp.LT, CompOp.LE):
        # clock1 - clock2 <= val (or < val)
        zc.dbm[i][j] = zc._min_bound(zc.dbm[i][j], (val, strict))
    elif constraint.op in (CompOp.GT, CompOp.GE):
        # clock2 - clock1 <= -val (or < -val)
        zc.dbm[j][i] = zc._min_bound(zc.dbm[j][i], (-val, strict))
    elif constraint.op == CompOp.EQ:
        zc.dbm[i][j] = zc._min_bound(zc.dbm[i][j], (val, False))
        zc.dbm[j][i] = zc._min_bound(zc.dbm[j][i], (-val, False))

    zc.canonicalize()
    return zc


def apply_guard(z: Zone, guard: Guard) -> Zone:
    """Apply all constraints in guard to zone."""
    result = z
    for c in guard.constraints:
        result = constrain_zone(result, c)
    return result


def reset_clocks(z: Zone, resets: FrozenSet[str]) -> Zone:
    """Reset specified clocks to 0."""
    zc = z.copy()
    dim = z.n + 1
    for clock_name in resets:
        if clock_name not in z.clock_names:
            continue
        idx = z.clock_index(clock_name)
        for k in range(dim):
            zc.dbm[idx][k] = zc.dbm[0][k]
            zc.dbm[k][idx] = zc.dbm[k][0]
        zc.dbm[idx][idx] = (0, False)
    zc.canonicalize()
    return zc


def future(z: Zone) -> Zone:
    """Time successor: remove upper bounds on clocks (allow time to pass)."""
    zc = z.copy()
    dim = z.n + 1
    for i in range(1, dim):
        zc.dbm[i][0] = (INF, True)
    zc.canonicalize()
    return zc


def past(z: Zone) -> Zone:
    """Time predecessor: remove lower bounds on clocks."""
    zc = z.copy()
    dim = z.n + 1
    for i in range(1, dim):
        zc.dbm[0][i] = (0, False)
    zc.canonicalize()
    return zc


def apply_invariant(z: Zone, inv: Guard) -> Zone:
    """Apply location invariant to zone."""
    return apply_guard(z, inv)


def successor_zone(z: Zone, edge: Edge, target_inv: Guard,
                   clock_names: List[str]) -> Optional[Zone]:
    """Forward zone successor: guard -> reset -> future -> invariant."""
    z1 = apply_guard(z, edge.guard)
    if z1.is_empty():
        return None
    z2 = reset_clocks(z1, edge.resets)
    z3 = future(z2)
    z4 = apply_invariant(z3, target_inv)
    if z4.is_empty():
        return None
    return z4


def backward_zone(z_target: Zone, edge: Edge, source_inv: Guard,
                  clock_names: List[str]) -> Optional[Zone]:
    """Backward zone predecessor: past -> undo resets -> guard -> invariant."""
    # Start from target zone
    z = z_target.copy()

    # Undo resets: free the reset clocks
    dim = z.n + 1
    for clock_name in edge.resets:
        if clock_name not in z.clock_names:
            continue
        idx = z.clock_index(clock_name)
        # After reset, clock was 0, so undo means clock was anything
        for k in range(dim):
            z.dbm[idx][k] = (INF, True)
            z.dbm[k][idx] = (INF, True) if k != idx else (0, False)
        z.dbm[idx][idx] = (0, False)
        # But non-reset clocks keep their relations to reference
        z.dbm[0][idx] = (0, False)  # clock >= 0

    z.canonicalize()
    if z.is_empty():
        return None

    # Apply guard
    z = apply_guard(z, edge.guard)
    if z.is_empty():
        return None

    # Past operator (allow earlier times)
    z = past(z)

    # Apply source invariant
    z = apply_invariant(z, source_inv)
    if z.is_empty():
        return None

    return z


# ============================================================
# Priced Zones
# ============================================================

@dataclass
class PricedZone:
    """
    A zone augmented with a linear cost function.

    cost_offset: base cost at the zone's reference point
    cost_rates: Dict[clock_name -> rate] -- cost increases linearly with clock value
    location_rate: rate cost from the location (cost per time unit)

    The cost at a valuation v is:
      cost_offset + sum(cost_rates[c] * v[c] for c in cost_rates) + location_rate * delay

    For min-cost reachability, we track the minimum cost to reach the target
    from any point in this zone.
    """
    zone: Zone
    cost_offset: float  # minimum cost achievable from this zone
    cost_rates: Dict[str, float] = field(default_factory=dict)
    location_rate: float = 0.0

    def is_empty(self) -> bool:
        return self.zone.is_empty()

    def copy(self) -> PricedZone:
        return PricedZone(
            self.zone.copy(),
            self.cost_offset,
            dict(self.cost_rates),
            self.location_rate
        )

    def min_cost(self) -> float:
        """Minimum cost achievable from any point in this zone."""
        return self.cost_offset

    def cost_at_valuation(self, valuation: Dict[str, float], delay: float = 0.0) -> float:
        """Cost at a specific clock valuation with given delay."""
        c = self.cost_offset
        for clock, rate in self.cost_rates.items():
            c += rate * valuation.get(clock, 0.0)
        c += self.location_rate * delay
        return c


def priced_zone_union(pzones: List[PricedZone]) -> List[PricedZone]:
    """
    Merge priced zones with same underlying zone, keeping minimum cost.
    Returns non-dominated priced zones.
    """
    if not pzones:
        return []

    result: List[PricedZone] = []
    for pz in pzones:
        if pz.is_empty():
            continue
        dominated = False
        new_result = []
        for existing in result:
            if existing.zone.includes(pz.zone) and existing.cost_offset <= pz.cost_offset:
                dominated = True
                new_result.append(existing)
            elif pz.zone.includes(existing.zone) and pz.cost_offset <= existing.cost_offset:
                # pz dominates existing, skip existing
                continue
            else:
                new_result.append(existing)
        if not dominated:
            new_result.append(pz)
        result = new_result
    return result


# ============================================================
# Min-Cost Reachability Solver
# ============================================================

@dataclass
class CostResult:
    """Result of min-cost reachability analysis."""
    reachable: bool
    min_cost: float  # INF if unreachable
    strategy: Dict[str, Tuple[float, int]]  # location -> (delay, edge_index)
    winning_zones: Dict[str, List[PricedZone]]  # location -> priced zones
    iterations: int
    cost_bound_used: float


def _get_invariant(game: WeightedTimedGame, loc: str) -> Guard:
    return game.invariants.get(loc, Guard.true_guard())


def solve_min_cost_reachability(
    game: WeightedTimedGame,
    targets: Optional[Set[str]] = None,
    max_iterations: int = 100,
    cost_bound: float = INF
) -> CostResult:
    """
    Backward fixed-point computation for min-cost reachability.

    MIN player tries to reach targets with minimum cost.
    MAX player tries to prevent reaching or maximize cost.

    Algorithm:
      1. Initialize: target locations have cost 0
      2. Backward: for each edge reaching a priced zone, compute
         predecessor priced zone (adding edge cost + rate * delay)
      3. For MIN locations: take minimum over successor costs
      4. For MAX locations: take maximum over successor costs
      5. Iterate until fixed point
    """
    if targets is None:
        targets = game.accepting

    clock_names = sorted(game.clocks)
    if not clock_names:
        clock_names = ["_dummy"]
        game.clocks.add("_dummy")

    # Initialize winning zones: targets have cost 0
    winning: Dict[str, List[PricedZone]] = {loc: [] for loc in game.locations}
    for t in targets:
        inv = _get_invariant(game, t)
        z = make_zone(clock_names)
        z = apply_invariant(z, inv)
        if not z.is_empty():
            winning[t].append(PricedZone(z, 0.0, {}, 0.0))

    strategy: Dict[str, Tuple[float, int]] = {}

    for iteration in range(max_iterations):
        changed = False

        for loc in game.locations:
            if loc in targets:
                continue

            loc_inv = _get_invariant(game, loc)
            is_min = game.owner.get(loc, Player.MIN) == Player.MIN
            loc_rate = game.rate_cost.get(loc, 0)

            # Collect all priced predecessor zones via edges from this location
            candidate_pzones: List[Tuple[PricedZone, int]] = []  # (pzone, edge_idx)

            for edge_idx, edge in game.get_edges_from(loc):
                target_pzones = winning.get(edge.target, [])
                if not target_pzones:
                    continue

                for tpz in target_pzones:
                    # Backward: compute predecessor zone
                    pred_z = backward_zone(tpz.zone, edge, loc_inv, clock_names)
                    if pred_z is None or pred_z.is_empty():
                        continue

                    # Cost = edge_cost + target_cost + rate * (max delay in zone)
                    # For MIN player, we want the cheapest path
                    # Delay cost is bounded by invariant
                    max_delay = _max_delay_in_zone(pred_z, loc_inv, clock_names)
                    delay_cost = loc_rate * max_delay if max_delay < INF else 0

                    total_cost = edge.cost + tpz.cost_offset + delay_cost

                    if total_cost <= cost_bound:
                        new_rates = dict(tpz.cost_rates)
                        new_pz = PricedZone(pred_z, total_cost, new_rates, loc_rate)
                        candidate_pzones.append((new_pz, edge_idx))

            if not candidate_pzones:
                continue

            if is_min:
                # MIN picks the cheapest option
                best_pzones = []
                best_cost = INF
                best_edge = -1
                for pz, eidx in candidate_pzones:
                    if pz.cost_offset < best_cost:
                        best_cost = pz.cost_offset
                        best_edge = eidx
                    best_pzones.append(pz)
                merged = priced_zone_union(best_pzones)

                if best_edge >= 0:
                    strategy[loc] = (0.0, best_edge)
            else:
                # MAX picks the most expensive (worst for MIN)
                # But MAX must also allow reaching target (game semantics)
                # In timed games: MAX wants to maximize cost, not prevent reachability
                # We take the max-cost option that still reaches
                best_pzones = []
                worst_cost = -INF
                worst_edge = -1
                for pz, eidx in candidate_pzones:
                    if pz.cost_offset > worst_cost:
                        worst_cost = pz.cost_offset
                        worst_edge = eidx
                    best_pzones.append(pz)
                merged = priced_zone_union(best_pzones)

            # Check if winning zones changed
            old_zones = winning[loc]
            if len(merged) != len(old_zones):
                changed = True
            elif any(not _pzone_equal(m, o) for m, o in zip(merged, old_zones)):
                changed = True

            if changed or not old_zones:
                winning[loc] = merged
                changed = True

        if not changed:
            break

    # Check if initial location is winning
    initial_loc = game.initial
    initial_pzones = winning.get(initial_loc, [])

    # Check if zero valuation is in any winning zone
    zero_zone = make_zero_zone(clock_names)
    reachable = False
    min_cost = INF

    for pz in initial_pzones:
        test = pz.zone.intersect(zero_zone)
        if not test.is_empty():
            reachable = True
            if pz.cost_offset < min_cost:
                min_cost = pz.cost_offset

    return CostResult(
        reachable=reachable,
        min_cost=min_cost if reachable else INF,
        strategy=strategy,
        winning_zones=winning,
        iterations=iteration + 1 if 'iteration' in dir() else 0,
        cost_bound_used=cost_bound
    )


def _max_delay_in_zone(z: Zone, inv: Guard, clock_names: List[str]) -> float:
    """Estimate maximum delay possible in a zone under invariant."""
    # The max delay is bounded by invariant upper bounds minus current clock values
    # From DBM: upper bound on clock_i is dbm[i][0] (clock_i - ref <= bound)
    max_delay = INF
    for c in inv.constraints:
        if c.op in (CompOp.LE, CompOp.LT):
            # clock <= value => max time is value
            max_delay = min(max_delay, c.value)
    if max_delay == INF:
        max_delay = 0  # no invariant => can't estimate, use 0
    return max(max_delay, 0)


def _pzone_equal(a: PricedZone, b: PricedZone) -> bool:
    return (a.zone == b.zone and
            abs(a.cost_offset - b.cost_offset) < 1e-9 and
            a.cost_rates == b.cost_rates)


# ============================================================
# Cost-Bounded Reachability
# ============================================================

def solve_cost_bounded_reachability(
    game: WeightedTimedGame,
    targets: Optional[Set[str]] = None,
    budget: float = INF,
    max_iterations: int = 100
) -> CostResult:
    """Can MIN player reach target within the given cost budget?"""
    return solve_min_cost_reachability(game, targets, max_iterations, budget)


# ============================================================
# Pareto-Optimal Analysis (Time vs Cost)
# ============================================================

@dataclass
class ParetoPoint:
    """A point on the Pareto frontier: (time, cost)."""
    time: float
    cost: float
    strategy: Dict[str, Tuple[float, int]]

    def dominates(self, other: ParetoPoint) -> bool:
        """Does self dominate other? (better or equal in both dimensions)"""
        return (self.time <= other.time and self.cost <= other.cost and
                (self.time < other.time or self.cost < other.cost))


@dataclass
class ParetoResult:
    """Result of Pareto-optimal analysis."""
    frontier: List[ParetoPoint]
    all_points: List[ParetoPoint]


def compute_pareto_frontier(
    game: WeightedTimedGame,
    targets: Optional[Set[str]] = None,
    time_budgets: Optional[List[float]] = None,
    max_iterations: int = 100
) -> ParetoResult:
    """
    Compute Pareto frontier of (time, cost) tradeoffs.

    Strategy: solve min-cost reachability under different time budgets.
    Time budget is enforced via a virtual clock with invariant.
    """
    if targets is None:
        targets = game.accepting
    if time_budgets is None:
        M = game.max_constant()
        time_budgets = [float(t) for t in range(1, M * 3 + 1)]

    all_points: List[ParetoPoint] = []

    for tb in time_budgets:
        # Create time-bounded variant: add a global clock with invariant
        bounded_game = _add_time_bound(game, tb)
        result = solve_min_cost_reachability(bounded_game, targets, max_iterations)
        if result.reachable:
            all_points.append(ParetoPoint(tb, result.min_cost, result.strategy))

    # Extract Pareto frontier (non-dominated points)
    frontier = []
    for p in sorted(all_points, key=lambda x: x.time):
        dominated = False
        for q in frontier:
            if q.dominates(p):
                dominated = True
                break
        if not dominated:
            # Remove any existing points dominated by p
            frontier = [q for q in frontier if not p.dominates(q)]
            frontier.append(p)

    return ParetoResult(frontier=frontier, all_points=all_points)


def _add_time_bound(game: WeightedTimedGame, time_bound: float) -> WeightedTimedGame:
    """Add a global time-bounding clock to the game."""
    g = WeightedTimedGame()
    g.locations = set(game.locations)
    g.initial = game.initial
    g.clocks = set(game.clocks) | {"__time"}
    g.edges = list(game.edges)
    g.invariants = dict(game.invariants)
    g.owner = dict(game.owner)
    g.accepting = set(game.accepting)
    g.rate_cost = dict(game.rate_cost)

    # Add time bound invariant to all non-target locations
    time_constraint = ClockConstraint("__time", None, CompOp.LE, int(time_bound))
    for loc in g.locations:
        existing = g.invariants.get(loc, Guard.true_guard())
        new_constraints = list(existing.constraints) + [time_constraint]
        g.invariants[loc] = Guard(tuple(new_constraints))

    return g


# ============================================================
# Simulation
# ============================================================

@dataclass
class SimStep:
    """One step of a game simulation."""
    location: str
    delay: float
    edge_label: str
    edge_cost: int
    delay_cost: float
    total_cost_so_far: float
    valuation: Dict[str, float]


def simulate_play(
    game: WeightedTimedGame,
    strategy_min: Dict[str, Tuple[float, int]],
    strategy_max: Optional[Dict[str, Tuple[float, int]]] = None,
    max_steps: int = 50,
    initial_valuation: Optional[Dict[str, float]] = None
) -> List[SimStep]:
    """
    Simulate a play in the weighted timed game.

    Strategies map locations to (delay, edge_index).
    """
    valuation = dict(initial_valuation or {c: 0.0 for c in game.clocks})
    loc = game.initial
    total_cost = 0.0
    trace: List[SimStep] = []

    for _ in range(max_steps):
        if loc in game.accepting:
            break

        is_min = game.owner.get(loc, Player.MIN) == Player.MIN
        strat = strategy_min if is_min else (strategy_max or strategy_min)

        if loc not in strat:
            break

        delay, edge_idx = strat[loc]
        if edge_idx >= len(game.edges):
            break

        edge = game.edges[edge_idx]
        if edge.source != loc:
            break

        # Apply delay
        for c in valuation:
            valuation[c] += delay

        # Check invariant
        inv = _get_invariant(game, loc)
        if not inv.satisfied(valuation):
            break

        # Check guard
        if not edge.guard.satisfied(valuation):
            break

        # Compute costs
        loc_rate = game.rate_cost.get(loc, 0)
        delay_cost = loc_rate * delay
        total_cost += delay_cost + edge.cost

        trace.append(SimStep(
            location=loc,
            delay=delay,
            edge_label=edge.label,
            edge_cost=edge.cost,
            delay_cost=delay_cost,
            total_cost_so_far=total_cost,
            valuation=dict(valuation)
        ))

        # Apply resets
        for c in edge.resets:
            if c in valuation:
                valuation[c] = 0.0

        loc = edge.target

    return trace


# ============================================================
# Region-based min-cost (discrete exploration)
# ============================================================

@dataclass
class RegionState:
    """Discrete state: location + clock region (integer part + fractional ordering)."""
    location: str
    int_vals: Tuple[int, ...]  # integer parts of each clock
    frac_order: Tuple[int, ...]  # ordering of fractional parts (indices)
    at_integer: Tuple[bool, ...]  # whether each clock is at integer value

    def __hash__(self):
        return hash((self.location, self.int_vals, self.frac_order, self.at_integer))

    def __eq__(self, other):
        return (isinstance(other, RegionState) and
                self.location == other.location and
                self.int_vals == other.int_vals and
                self.frac_order == other.frac_order and
                self.at_integer == other.at_integer)


def solve_min_cost_region(
    game: WeightedTimedGame,
    targets: Optional[Set[str]] = None,
    max_states: int = 10000
) -> CostResult:
    """
    Min-cost reachability via region graph exploration.

    Enumerates clock regions explicitly, computing exact min costs.
    Works for small games with small clock constants.
    """
    if targets is None:
        targets = game.accepting
    clock_names = sorted(game.clocks)
    n_clocks = len(clock_names)
    M = game.max_constant()

    if n_clocks == 0:
        # No clocks: reduce to weighted graph
        return _solve_untimed_min_cost(game, targets)

    # BFS/Dijkstra on region graph
    import heapq

    # Initial state: all clocks at 0
    init_ints = tuple(0 for _ in range(n_clocks))
    init_frac = tuple(range(n_clocks))  # all equal fractional (0)
    init_at_int = tuple(True for _ in range(n_clocks))
    init_state = RegionState(game.initial, init_ints, init_frac, init_at_int)

    # Dijkstra
    dist: Dict[RegionState, float] = {init_state: 0.0}
    prev: Dict[RegionState, Tuple[RegionState, int]] = {}  # state -> (prev, edge_idx)
    heap = [(0.0, id(init_state), init_state)]
    visited: Set[RegionState] = set()
    best_target_cost = INF
    best_target_state = None

    while heap and len(visited) < max_states:
        cost, _, state = heapq.heappop(heap)
        if state in visited:
            continue
        visited.add(state)

        if state.location in targets:
            if cost < best_target_cost:
                best_target_cost = cost
                best_target_state = state
            continue

        # Time successors (delay within region)
        time_succs = _region_time_successors(state, M, clock_names,
                                             game.invariants.get(state.location))
        loc_rate = game.rate_cost.get(state.location, 0)

        for ts in time_succs:
            # Cost of time step: rate * 1 unit (approximate)
            time_cost = loc_rate * 1  # unit time step through region
            new_cost = cost + time_cost
            if new_cost < dist.get(ts, INF):
                dist[ts] = new_cost
                heapq.heappush(heap, (new_cost, id(ts), ts))
                prev[ts] = (state, -1)

        # Edge successors
        for edge_idx, edge in game.get_edges_from(state.location):
            if _region_satisfies_guard(state, edge.guard, clock_names, M):
                succ = _region_edge_successor(state, edge, clock_names, M,
                                              game.invariants.get(edge.target))
                if succ is not None:
                    new_cost = cost + edge.cost
                    if new_cost < dist.get(succ, INF):
                        dist[succ] = new_cost
                        heapq.heappush(heap, (new_cost, id(succ), succ))
                        prev[succ] = (state, edge_idx)

    # Extract strategy
    strategy: Dict[str, Tuple[float, int]] = {}
    if best_target_state is not None:
        state = best_target_state
        while state in prev:
            pred_state, edge_idx = prev[state]
            if edge_idx >= 0:
                strategy[pred_state.location] = (0.0, edge_idx)
            state = pred_state

    reachable = best_target_cost < INF
    return CostResult(
        reachable=reachable,
        min_cost=best_target_cost if reachable else INF,
        strategy=strategy,
        winning_zones={},
        iterations=len(visited),
        cost_bound_used=INF
    )


def _region_time_successors(state: RegionState, M: int,
                            clock_names: List[str],
                            invariant: Optional[Guard]) -> List[RegionState]:
    """Compute time-successor regions."""
    n = len(clock_names)
    results = []

    # Case 1: some clock is at integer -> advance to non-integer
    if any(state.at_integer):
        new_at_int = tuple(False for _ in range(n))
        # Fractional ordering: clocks that were at integer get smallest fractional part
        int_clocks = [i for i in range(n) if state.at_integer[i]]
        non_int_clocks = [i for i in range(n) if not state.at_integer[i]]
        # New ordering: non-int clocks keep order, int clocks are at bottom
        new_frac = tuple(non_int_clocks + int_clocks)
        succ = RegionState(state.location, state.int_vals, new_frac, new_at_int)
        if _region_satisfies_invariant(succ, invariant, clock_names, M):
            results.append(succ)

    # Case 2: advance fractional part -- the clock with largest fractional hits next integer
    if not all(state.at_integer):
        # The last clock in frac_order has largest fractional part
        # When it reaches integer, its int part increments
        if state.frac_order:
            top_clock = state.frac_order[-1]
            new_ints = list(state.int_vals)
            new_ints[top_clock] += 1
            if new_ints[top_clock] <= M + 1:
                new_at_int = list(state.at_integer)
                new_at_int[top_clock] = True
                # Reorder: remove top_clock from frac_order
                new_frac = tuple(i for i in state.frac_order if i != top_clock)
                succ = RegionState(state.location, tuple(new_ints),
                                   new_frac, tuple(new_at_int))
                if _region_satisfies_invariant(succ, invariant, clock_names, M):
                    results.append(succ)

    return results


def _region_satisfies_guard(state: RegionState, guard: Guard,
                            clock_names: List[str], M: int) -> bool:
    """Check if region can satisfy guard (conservative)."""
    for c in guard.constraints:
        idx = clock_names.index(c.clock1) if c.clock1 in clock_names else -1
        if idx < 0:
            continue
        val = state.int_vals[idx]
        at_int = state.at_integer[idx]

        if c.op == CompOp.LE:
            if val > c.value:
                return False
        elif c.op == CompOp.LT:
            if val > c.value or (val == c.value and at_int):
                return False
        elif c.op == CompOp.GE:
            if val < c.value and not (val == c.value - 1 and not at_int):
                return False
            if at_int and val < c.value:
                return False
        elif c.op == CompOp.GT:
            if val < c.value:
                return False
        elif c.op == CompOp.EQ:
            if not at_int or val != c.value:
                return False
    return True


def _region_satisfies_invariant(state: RegionState, invariant: Optional[Guard],
                                clock_names: List[str], M: int) -> bool:
    if invariant is None:
        return True
    return _region_satisfies_guard(state, invariant, clock_names, M)


def _region_edge_successor(state: RegionState, edge: Edge,
                           clock_names: List[str], M: int,
                           target_inv: Optional[Guard]) -> Optional[RegionState]:
    """Compute edge successor region."""
    n = len(clock_names)
    new_ints = list(state.int_vals)
    new_at_int = list(state.at_integer)
    new_frac = list(state.frac_order)

    for clock_name in edge.resets:
        if clock_name in clock_names:
            idx = clock_names.index(clock_name)
            new_ints[idx] = 0
            new_at_int[idx] = True
            if idx in new_frac:
                new_frac.remove(idx)

    succ = RegionState(edge.target, tuple(new_ints),
                       tuple(new_frac), tuple(new_at_int))
    if not _region_satisfies_invariant(succ, target_inv, clock_names, M):
        return None
    return succ


def _solve_untimed_min_cost(game: WeightedTimedGame,
                            targets: Set[str]) -> CostResult:
    """Min-cost reachability on untimed game (just weighted graph)."""
    import heapq

    dist: Dict[str, float] = {t: 0.0 for t in targets}
    heap = [(0.0, t) for t in targets]
    prev: Dict[str, int] = {}

    # Backward Dijkstra from targets
    while heap:
        cost, loc = heapq.heappop(heap)
        if cost > dist.get(loc, INF):
            continue

        # Find edges targeting this location
        for edge_idx, edge in enumerate(game.edges):
            if edge.target == loc:
                new_cost = cost + edge.cost
                if new_cost < dist.get(edge.source, INF):
                    dist[edge.source] = new_cost
                    prev[edge.source] = edge_idx
                    heapq.heappush(heap, (new_cost, edge.source))

    strategy = {loc: (0.0, eidx) for loc, eidx in prev.items()}
    reachable = game.initial in dist
    return CostResult(
        reachable=reachable,
        min_cost=dist.get(game.initial, INF),
        strategy=strategy,
        winning_zones={},
        iterations=len(dist),
        cost_bound_used=INF
    )


# ============================================================
# Verification
# ============================================================

def verify_strategy_cost(
    game: WeightedTimedGame,
    strategy: Dict[str, Tuple[float, int]],
    targets: Optional[Set[str]] = None,
    max_steps: int = 100
) -> Tuple[bool, float]:
    """
    Verify a strategy reaches targets and compute its cost.
    Returns (reaches_target, total_cost).
    """
    if targets is None:
        targets = game.accepting

    trace = simulate_play(game, strategy, max_steps=max_steps)
    if not trace:
        # Check if initial is already target
        return game.initial in targets, 0.0

    last = trace[-1]
    final_edge = game.edges[strategy[last.location][1]] if last.location in strategy else None
    final_loc = final_edge.target if final_edge else last.location
    reaches = final_loc in targets
    return reaches, last.total_cost_so_far


# ============================================================
# Example games
# ============================================================

def make_simple_weighted_game() -> WeightedTimedGame:
    """
    Simple 3-location game:
    s0 --(x<=2, cost=3)--> s1 --(y<=1, cost=1)--> target
    s0 --(x<=5, cost=8)--> target  (direct but expensive)

    MIN player owns all. Optimal: go via s1 (cost=4) vs direct (cost=8).
    """
    g = WeightedTimedGame()
    g.initial = "s0"
    g.clocks = {"x", "y"}
    g.accepting = {"target"}

    g.add_location("s0", Player.MIN, rate=0)
    g.add_location("s1", Player.MIN, rate=1)  # rate cost in s1
    g.add_location("target", Player.MIN, rate=0)

    # s0 -> s1: guard x <= 2, reset y, cost 3
    g.add_edge("s0", "s1", "fast",
               Guard((ClockConstraint("x", None, CompOp.LE, 2),)),
               frozenset({"y"}), cost=3)

    # s1 -> target: guard y <= 1, cost 1
    g.add_edge("s1", "target", "finish",
               Guard((ClockConstraint("y", None, CompOp.LE, 1),)),
               frozenset(), cost=1)

    # s0 -> target: guard x <= 5, cost 8
    g.add_edge("s0", "target", "direct",
               Guard((ClockConstraint("x", None, CompOp.LE, 5),)),
               frozenset(), cost=8)

    return g


def make_two_player_cost_game() -> WeightedTimedGame:
    """
    Two-player weighted timed game.
    MIN at s0 chooses: cheap path (s1, cost 2) or expensive path (s2, cost 5).
    MAX at s1 can delay (rate=3) or proceed quickly.
    MAX at s2 must proceed (rate=0, cost 0).
    """
    g = WeightedTimedGame()
    g.initial = "s0"
    g.clocks = {"x"}
    g.accepting = {"goal"}

    g.add_location("s0", Player.MIN, rate=0)
    g.add_location("s1", Player.MAX,
                   Guard((ClockConstraint("x", None, CompOp.LE, 3),)),
                   rate=3)
    g.add_location("s2", Player.MIN,
                   Guard((ClockConstraint("x", None, CompOp.LE, 5),)),
                   rate=0)
    g.add_location("goal", Player.MIN, rate=0)

    # s0 -> s1 (cheap edge, but MAX controls s1)
    g.add_edge("s0", "s1", "cheap", cost=2)

    # s0 -> s2 (expensive edge, but MIN controls s2)
    g.add_edge("s0", "s2", "expensive", cost=5)

    # s1 -> goal
    g.add_edge("s1", "goal", "exit1", cost=0)

    # s2 -> goal
    g.add_edge("s2", "goal", "exit2", cost=0)

    return g


def make_rate_cost_game() -> WeightedTimedGame:
    """
    Game where location rate costs matter.
    s0 (rate=0) -> s1 (rate=2, inv: x<=3) -> target
    s0 (rate=0) -> s2 (rate=1, inv: x<=5) -> target

    Via s1: edge cost 1 + up to 6 rate cost = 7
    Via s2: edge cost 4 + up to 5 rate cost = 9
    But s1 has a tight invariant, so actual rate cost is bounded.
    """
    g = WeightedTimedGame()
    g.initial = "s0"
    g.clocks = {"x"}
    g.accepting = {"target"}

    g.add_location("s0", Player.MIN, rate=0)
    g.add_location("s1", Player.MIN,
                   Guard((ClockConstraint("x", None, CompOp.LE, 3),)),
                   rate=2)
    g.add_location("s2", Player.MIN,
                   Guard((ClockConstraint("x", None, CompOp.LE, 5),)),
                   rate=1)
    g.add_location("target", Player.MIN, rate=0)

    g.add_edge("s0", "s1", "path1",
               Guard((ClockConstraint("x", None, CompOp.LE, 1),)),
               frozenset({"x"}), cost=1)
    g.add_edge("s0", "s2", "path2",
               Guard((ClockConstraint("x", None, CompOp.LE, 1),)),
               frozenset({"x"}), cost=4)

    g.add_edge("s1", "target", "arrive1", cost=0)
    g.add_edge("s2", "target", "arrive2", cost=0)

    return g


def make_scheduling_game() -> WeightedTimedGame:
    """
    Job scheduling as a weighted timed game.
    Two jobs: fast (cost 5, deadline x<=2) and slow (cost 2, deadline x<=5).
    Must complete both. MIN chooses order.
    """
    g = WeightedTimedGame()
    g.initial = "start"
    g.clocks = {"x", "y"}
    g.accepting = {"done"}

    g.add_location("start", Player.MIN, rate=0)
    g.add_location("job_a", Player.MIN,
                   Guard((ClockConstraint("x", None, CompOp.LE, 2),)),
                   rate=1)
    g.add_location("job_b", Player.MIN,
                   Guard((ClockConstraint("y", None, CompOp.LE, 5),)),
                   rate=1)
    g.add_location("after_a", Player.MIN, rate=0)
    g.add_location("after_b", Player.MIN, rate=0)
    g.add_location("done", Player.MIN, rate=0)

    # Start -> job_a first
    g.add_edge("start", "job_a", "do_a_first",
               Guard.true_guard(), frozenset({"x"}), cost=5)
    # Start -> job_b first
    g.add_edge("start", "job_b", "do_b_first",
               Guard.true_guard(), frozenset({"y"}), cost=2)

    # job_a -> after_a
    g.add_edge("job_a", "after_a", "a_done",
               Guard((ClockConstraint("x", None, CompOp.GE, 1),)),
               frozenset(), cost=0)
    # after_a -> job_b
    g.add_edge("after_a", "job_b", "then_b",
               Guard.true_guard(), frozenset({"y"}), cost=2)

    # job_b -> after_b
    g.add_edge("job_b", "after_b", "b_done",
               Guard((ClockConstraint("y", None, CompOp.GE, 1),)),
               frozenset(), cost=0)
    # after_b -> job_a
    g.add_edge("after_b", "job_a", "then_a",
               Guard.true_guard(), frozenset({"x"}), cost=5)

    # after_a -> done (if already did b)
    # after_b -> done (if already did a)
    # Simplified: both paths lead to done
    g.add_edge("after_a", "done", "finish_ab",
               Guard.true_guard(), frozenset(), cost=0)
    g.add_edge("after_b", "done", "finish_ba",
               Guard.true_guard(), frozenset(), cost=0)

    return g


def make_energy_timed_game() -> WeightedTimedGame:
    """
    Energy-like timed game where MIN must accumulate enough 'energy' (negative cost)
    before reaching target. Some edges give energy (negative cost modeled as 0),
    others consume it.
    """
    g = WeightedTimedGame()
    g.initial = "idle"
    g.clocks = {"x"}
    g.accepting = {"goal"}

    g.add_location("idle", Player.MIN,
                   Guard((ClockConstraint("x", None, CompOp.LE, 10),)),
                   rate=0)
    g.add_location("charge", Player.MIN,
                   Guard((ClockConstraint("x", None, CompOp.LE, 3),)),
                   rate=0)  # free to be here
    g.add_location("expensive", Player.MIN,
                   Guard((ClockConstraint("x", None, CompOp.LE, 2),)),
                   rate=5)
    g.add_location("goal", Player.MIN, rate=0)

    # idle -> charge (free)
    g.add_edge("idle", "charge", "start_charge",
               Guard.true_guard(), frozenset({"x"}), cost=0)
    # charge -> idle (free, resets)
    g.add_edge("charge", "idle", "charged",
               Guard((ClockConstraint("x", None, CompOp.GE, 1),)),
               frozenset({"x"}), cost=0)
    # idle -> expensive
    g.add_edge("idle", "expensive", "go_expensive",
               Guard.true_guard(), frozenset({"x"}), cost=3)
    # expensive -> goal
    g.add_edge("expensive", "goal", "reach_goal",
               Guard((ClockConstraint("x", None, CompOp.GE, 1),)),
               frozenset(), cost=2)
    # idle -> goal (direct, very expensive)
    g.add_edge("idle", "goal", "direct",
               Guard((ClockConstraint("x", None, CompOp.GE, 2),)),
               frozenset(), cost=20)

    return g
