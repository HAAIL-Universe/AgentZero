"""
V118: Timed Automata Verification

Real-time systems verification using Alur-Dill timed automata with zone-based
symbolic state space exploration.

Components:
1. Clock constraints (guards, invariants)
2. Timed automaton definition (locations, edges, clocks)
3. Zone (DBM) representation for efficient clock constraint manipulation
4. Zone-based successor computation (guard, reset, invariant, time elapse)
5. Zone graph exploration (BFS/DFS reachability)
6. Timed word acceptance
7. Safety and reachability checking
8. Timed automaton operations (product, complement region-based)
9. Diagnostic traces (counterexample paths with concrete timing)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from enum import Enum

INF = float('inf')


# --- Clock Constraints ---

class CompOp(Enum):
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    EQ = '=='


@dataclass(frozen=True)
class ClockConstraint:
    """Atomic clock constraint: x op c or x - y op c."""
    clock1: str
    clock2: Optional[str]  # None for x op c, string for x - y op c
    op: CompOp
    value: int  # Integer constants (Alur-Dill: rational, we use int for simplicity)

    def evaluate(self, valuation: Dict[str, float]) -> bool:
        """Check constraint against concrete clock values."""
        v1 = valuation.get(self.clock1, 0.0)
        v2 = valuation.get(self.clock2, 0.0) if self.clock2 else 0.0
        diff = v1 - v2
        if self.op == CompOp.LT:
            return diff < self.value
        elif self.op == CompOp.LE:
            return diff <= self.value
        elif self.op == CompOp.GT:
            return diff > self.value
        elif self.op == CompOp.GE:
            return diff >= self.value
        elif self.op == CompOp.EQ:
            return diff == self.value
        return False

    def __str__(self):
        lhs = self.clock1 if not self.clock2 else f"{self.clock1}-{self.clock2}"
        return f"{lhs} {self.op.value} {self.value}"


@dataclass(frozen=True)
class Guard:
    """Conjunction of clock constraints."""
    constraints: Tuple[ClockConstraint, ...]

    def evaluate(self, valuation: Dict[str, float]) -> bool:
        return all(c.evaluate(valuation) for c in self.constraints)

    def is_true(self) -> bool:
        return len(self.constraints) == 0

    def __str__(self):
        if not self.constraints:
            return "true"
        return " && ".join(str(c) for c in self.constraints)


def true_guard() -> Guard:
    return Guard(())


def clock_leq(clock: str, value: int) -> Guard:
    return Guard((ClockConstraint(clock, None, CompOp.LE, value),))


def clock_lt(clock: str, value: int) -> Guard:
    return Guard((ClockConstraint(clock, None, CompOp.LT, value),))


def clock_geq(clock: str, value: int) -> Guard:
    return Guard((ClockConstraint(clock, None, CompOp.GE, value),))


def clock_gt(clock: str, value: int) -> Guard:
    return Guard((ClockConstraint(clock, None, CompOp.GT, value),))


def clock_eq(clock: str, value: int) -> Guard:
    return Guard((ClockConstraint(clock, None, CompOp.EQ, value),))


def clock_diff_leq(c1: str, c2: str, value: int) -> Guard:
    return Guard((ClockConstraint(c1, c2, CompOp.LE, value),))


def clock_diff_geq(c1: str, c2: str, value: int) -> Guard:
    return Guard((ClockConstraint(c1, c2, CompOp.GE, value),))


def guard_and(g1: Guard, g2: Guard) -> Guard:
    return Guard(g1.constraints + g2.constraints)


# --- Timed Automaton ---

@dataclass(frozen=True)
class Edge:
    """Edge in a timed automaton."""
    source: str
    target: str
    label: str  # action label
    guard: Guard
    resets: FrozenSet[str]  # clocks to reset

    def __str__(self):
        r = "{" + ",".join(sorted(self.resets)) + "}" if self.resets else "{}"
        return f"{self.source} --[{self.label}, {self.guard}, {r}]--> {self.target}"


@dataclass
class TimedAutomaton:
    """Alur-Dill timed automaton."""
    locations: Set[str]
    initial: str
    clocks: Set[str]
    edges: List[Edge]
    invariants: Dict[str, Guard]  # location -> invariant (must hold while in location)
    accepting: Set[str] = field(default_factory=set)  # for Buchi acceptance
    alphabet: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.alphabet:
            self.alphabet = {e.label for e in self.edges}

    def get_edges_from(self, location: str) -> List[Edge]:
        return [e for e in self.edges if e.source == location]

    def get_invariant(self, location: str) -> Guard:
        return self.invariants.get(location, true_guard())

    def max_constant(self) -> int:
        """Largest constant appearing in any constraint."""
        max_c = 0
        for e in self.edges:
            for c in e.guard.constraints:
                max_c = max(max_c, abs(c.value))
        for inv in self.invariants.values():
            for c in inv.constraints:
                max_c = max(max_c, abs(c.value))
        return max_c


# --- Zone (DBM) ---

class Zone:
    """
    Difference Bound Matrix representation of clock zones.

    Clocks are indexed 0..n where 0 is the reference clock (always 0).
    DBM[i][j] = (bound, strict) means clock_i - clock_j <= bound (if not strict)
    or clock_i - clock_j < bound (if strict).

    (INF, False) means unconstrained.
    """

    def __init__(self, n_clocks: int, clock_names: List[str]):
        self.n = n_clocks + 1  # +1 for reference clock 0
        self.clock_names = clock_names
        self.clock_index = {name: i + 1 for i, name in enumerate(clock_names)}
        # Initialize: all differences unconstrained, diagonal = (0, False)
        self.dbm = [[(INF, False)] * self.n for _ in range(self.n)]
        for i in range(self.n):
            self.dbm[i][i] = (0, False)

    def copy(self) -> 'Zone':
        z = Zone.__new__(Zone)
        z.n = self.n
        z.clock_names = self.clock_names
        z.clock_index = self.clock_index
        z.dbm = [row[:] for row in self.dbm]
        return z

    def _idx(self, clock: str) -> int:
        return self.clock_index[clock]

    def is_empty(self) -> bool:
        """Check if the zone is empty (inconsistent)."""
        for i in range(self.n):
            b, s = self.dbm[i][i]
            if b < 0 or (b == 0 and s):
                return True
        return False

    def _add_bounds(self, b1: Tuple[float, bool], b2: Tuple[float, bool]) -> Tuple[float, bool]:
        """Add two DBM bounds: (v1,s1) + (v2,s2)."""
        v1, s1 = b1
        v2, s2 = b2
        if v1 == INF or v2 == INF:
            return (INF, False)
        return (v1 + v2, s1 or s2)

    def _min_bound(self, b1: Tuple[float, bool], b2: Tuple[float, bool]) -> Tuple[float, bool]:
        """Minimum of two DBM bounds."""
        v1, s1 = b1
        v2, s2 = b2
        if v1 < v2:
            return b1
        elif v2 < v1:
            return b2
        else:
            # Same value: strict is tighter
            return (v1, s1 or s2)

    def _leq_bound(self, b1: Tuple[float, bool], b2: Tuple[float, bool]) -> bool:
        """Check b1 <= b2 in bound ordering."""
        v1, s1 = b1
        v2, s2 = b2
        if v1 < v2:
            return True
        if v1 > v2:
            return False
        # Same value: (v, True) < (v, False)
        if s1 and not s2:
            return True
        return s1 == s2  # equal

    def canonicalize(self):
        """Floyd-Warshall shortest path closure."""
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    via_k = self._add_bounds(self.dbm[i][k], self.dbm[k][j])
                    self.dbm[i][j] = self._min_bound(self.dbm[i][j], via_k)

    def constrain(self, constraint: ClockConstraint):
        """Intersect zone with a clock constraint."""
        i = self._idx(constraint.clock1) if constraint.clock1 else 0
        j = self._idx(constraint.clock2) if constraint.clock2 else 0

        op = constraint.op
        val = constraint.value

        if op == CompOp.LE:
            # clock1 - clock2 <= val  =>  dbm[i][j] = min(current, (val, False))
            self.dbm[i][j] = self._min_bound(self.dbm[i][j], (val, False))
        elif op == CompOp.LT:
            self.dbm[i][j] = self._min_bound(self.dbm[i][j], (val, True))
        elif op == CompOp.GE:
            # clock1 - clock2 >= val  =>  clock2 - clock1 <= -val
            self.dbm[j][i] = self._min_bound(self.dbm[j][i], (-val, False))
        elif op == CompOp.GT:
            self.dbm[j][i] = self._min_bound(self.dbm[j][i], (-val, True))
        elif op == CompOp.EQ:
            self.dbm[i][j] = self._min_bound(self.dbm[i][j], (val, False))
            self.dbm[j][i] = self._min_bound(self.dbm[j][i], (-val, False))

    def apply_guard(self, guard: Guard):
        """Intersect zone with a guard (conjunction of constraints)."""
        for c in guard.constraints:
            self.constrain(c)
        self.canonicalize()

    def reset(self, clock: str):
        """Reset a clock to 0."""
        x = self._idx(clock)
        for i in range(self.n):
            # x - i <= 0 - i_lower  =>  use dbm[0][i]
            self.dbm[x][i] = self.dbm[0][i]
            self.dbm[i][x] = self.dbm[i][0]
        self.dbm[x][x] = (0, False)

    def apply_resets(self, resets: FrozenSet[str]):
        """Apply multiple clock resets."""
        for c in resets:
            if c in self.clock_index:
                self.reset(c)

    def future(self):
        """Time elapse: remove upper bounds on clocks (let time pass)."""
        for i in range(1, self.n):
            self.dbm[i][0] = (INF, False)  # clock_i - 0 unconstrained (clock can grow)
        # Note: differences between clocks are preserved
        # (time passes uniformly for all clocks)

    def intersect(self, other: 'Zone') -> 'Zone':
        """Intersect two zones."""
        assert self.n == other.n
        z = self.copy()
        for i in range(self.n):
            for j in range(self.n):
                z.dbm[i][j] = self._min_bound(self.dbm[i][j], other.dbm[i][j])
        z.canonicalize()
        return z

    def includes(self, other: 'Zone') -> bool:
        """Check if self includes other (other is a subset of self)."""
        for i in range(self.n):
            for j in range(self.n):
                if not self._leq_bound(other.dbm[i][j], self.dbm[i][j]):
                    return False
        return True

    def equals(self, other: 'Zone') -> bool:
        """Check zone equality."""
        return self.dbm == other.dbm

    def get_sample(self) -> Optional[Dict[str, float]]:
        """Get a concrete valuation from the zone, or None if empty."""
        if self.is_empty():
            return None
        # Assign each clock its lower bound (negation of dbm[0][i])
        result = {}
        for name, idx in self.clock_index.items():
            lower_bound, strict = self.dbm[0][idx]
            # lower bound of clock_i is -dbm[0][i]
            lower = -lower_bound if lower_bound != INF else 0
            if strict:
                lower += 0.01  # nudge past strict bound
            result[name] = max(0, lower)
        # Verify against upper bounds
        for name, idx in self.clock_index.items():
            upper_bound, strict = self.dbm[idx][0]
            if upper_bound != INF:
                result[name] = min(result[name], upper_bound - (0.01 if strict else 0))
            result[name] = max(0, result[name])
        return result

    def __str__(self):
        parts = []
        for name, idx in self.clock_index.items():
            # Upper bound: clock - 0 <= ub
            ub, us = self.dbm[idx][0]
            # Lower bound: 0 - clock <= -lb  =>  clock >= lb
            lb_neg, ls = self.dbm[0][idx]
            lb = -lb_neg
            if ub == INF and lb <= 0:
                continue  # unconstrained
            ub_str = f"{'<' if us else '<='} {ub}" if ub != INF else ""
            lb_str = f"{lb} {'<' if ls else '<='}" if lb > 0 else ""
            if lb_str and ub_str:
                parts.append(f"{lb_str} {name} {ub_str}")
            elif ub_str:
                parts.append(f"{name} {ub_str}")
            elif lb_str:
                parts.append(f"{lb_str} {name}")
        if not parts:
            return "true"
        return " && ".join(parts)

    def __repr__(self):
        return f"Zone({self})"


def initial_zone(clocks: List[str]) -> Zone:
    """Create initial zone: all clocks = 0."""
    z = Zone(len(clocks), clocks)
    for name in clocks:
        idx = z._idx(name)
        z.dbm[idx][0] = (0, False)  # clock - 0 <= 0
        z.dbm[0][idx] = (0, False)  # 0 - clock <= 0
    return z


# --- Symbolic State ---

@dataclass(frozen=True)
class SymbolicState:
    """Location + zone pair for zone graph exploration."""
    location: str
    zone_id: int  # index into zone storage (zones aren't hashable)


@dataclass
class ZoneStore:
    """Storage for zones with deduplication."""
    zones: List[Zone] = field(default_factory=list)

    def add(self, zone: Zone) -> int:
        self.zones.append(zone)
        return len(self.zones) - 1

    def get(self, idx: int) -> Zone:
        return self.zones[idx]


# --- Zone Graph Exploration ---

@dataclass
class Trace:
    """Counterexample/witness trace."""
    steps: List[Tuple[str, str, Optional[Dict[str, float]]]]
    # Each step: (location, action_label, clock_valuation)

    def __str__(self):
        lines = []
        for loc, action, vals in self.steps:
            v_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(vals.items())) if vals else ""
            lines.append(f"  {loc} [{v_str}] --{action}-->")
        return "\n".join(lines)


@dataclass
class ReachabilityResult:
    """Result of reachability analysis."""
    reachable: bool
    target_location: Optional[str] = None
    trace: Optional[Trace] = None
    states_explored: int = 0
    zones_created: int = 0


@dataclass
class SafetyResult:
    """Result of safety analysis."""
    safe: bool
    violated_location: Optional[str] = None
    trace: Optional[Trace] = None
    states_explored: int = 0


@dataclass
class ZoneGraphStats:
    """Statistics from zone graph exploration."""
    n_locations: int = 0
    n_symbolic_states: int = 0
    n_edges: int = 0
    n_zones: int = 0


def _successor_zone(zone: Zone, edge: Edge, target_inv: Guard) -> Optional[Zone]:
    """Compute successor zone for an edge."""
    z = zone.copy()
    # 1. Apply guard
    z.apply_guard(edge.guard)
    if z.is_empty():
        return None
    # 2. Apply resets
    z.apply_resets(edge.resets)
    # 3. Apply target invariant
    z.apply_guard(target_inv)
    if z.is_empty():
        return None
    # 4. Time elapse (future)
    z.future()
    # 5. Apply target invariant again (must hold after time passes)
    z.apply_guard(target_inv)
    z.canonicalize()
    if z.is_empty():
        return None
    return z


def explore_zone_graph(ta: TimedAutomaton) -> Tuple[Dict[str, List[Zone]], ZoneGraphStats]:
    """Explore the full zone graph via BFS. Returns reachable zones per location."""
    clocks = sorted(ta.clocks)
    # Initial zone: all clocks = 0, time can elapse under initial invariant
    z0 = initial_zone(clocks)
    init_inv = ta.get_invariant(ta.initial)
    z0.apply_guard(init_inv)
    z0.future()
    z0.apply_guard(init_inv)
    z0.canonicalize()

    if z0.is_empty():
        return {}, ZoneGraphStats()

    # BFS
    reached: Dict[str, List[Zone]] = {loc: [] for loc in ta.locations}
    reached[ta.initial].append(z0)
    queue = [(ta.initial, z0)]
    stats = ZoneGraphStats(n_locations=len(ta.locations), n_zones=1, n_symbolic_states=1)

    while queue:
        loc, zone = queue.pop(0)
        for edge in ta.get_edges_from(loc):
            target_inv = ta.get_invariant(edge.target)
            succ = _successor_zone(zone, edge, target_inv)
            if succ is None:
                continue
            # Check if this zone is already subsumed by an existing zone
            subsumed = False
            for existing in reached[edge.target]:
                if existing.includes(succ):
                    subsumed = True
                    break
            if not subsumed:
                # Remove zones subsumed by the new one
                reached[edge.target] = [
                    ez for ez in reached[edge.target] if not succ.includes(ez)
                ]
                reached[edge.target].append(succ)
                queue.append((edge.target, succ))
                stats.n_zones += 1
                stats.n_symbolic_states += 1
                stats.n_edges += 1

    return reached, stats


def check_reachability(ta: TimedAutomaton,
                       target_locations: Set[str]) -> ReachabilityResult:
    """Check if any target location is reachable. Returns trace if yes."""
    clocks = sorted(ta.clocks)
    z0 = initial_zone(clocks)
    init_inv = ta.get_invariant(ta.initial)
    z0.apply_guard(init_inv)
    z0.future()
    z0.apply_guard(init_inv)
    z0.canonicalize()

    if z0.is_empty():
        return ReachabilityResult(reachable=False, states_explored=0)

    if ta.initial in target_locations:
        sample = z0.get_sample()
        trace = Trace([(ta.initial, "init", sample)])
        return ReachabilityResult(reachable=True, target_location=ta.initial,
                                  trace=trace, states_explored=1, zones_created=1)

    # BFS with parent tracking
    reached: Dict[str, List[Tuple[Zone, Optional[Tuple[str, str, int]]]]] = {
        loc: [] for loc in ta.locations
    }
    reached[ta.initial].append((z0, None))
    queue = [(ta.initial, 0)]  # (location, zone_index in reached[location])
    explored = 0

    while queue:
        loc, z_idx = queue.pop(0)
        zone, _ = reached[loc][z_idx]
        explored += 1

        for edge in ta.get_edges_from(loc):
            target_inv = ta.get_invariant(edge.target)
            succ = _successor_zone(zone, edge, target_inv)
            if succ is None:
                continue

            parent_info = (loc, edge.label, z_idx)

            # Check if target reached
            if edge.target in target_locations:
                # Reconstruct trace
                trace_steps = []
                sample = succ.get_sample()
                trace_steps.append((edge.target, edge.label, sample))
                cur_loc, cur_label, cur_idx = parent_info
                while True:
                    cur_zone, cur_parent = reached[cur_loc][cur_idx]
                    sample = cur_zone.get_sample()
                    if cur_parent is None:
                        trace_steps.append((cur_loc, "init", sample))
                        break
                    trace_steps.append((cur_loc, cur_label, sample))
                    cur_loc, cur_label, cur_idx = cur_parent
                trace_steps.reverse()
                new_idx = len(reached[edge.target])
                reached[edge.target].append((succ, parent_info))
                return ReachabilityResult(
                    reachable=True, target_location=edge.target,
                    trace=Trace(trace_steps),
                    states_explored=explored,
                    zones_created=sum(len(v) for v in reached.values())
                )

            # Subsumption check
            subsumed = False
            for existing_zone, _ in reached[edge.target]:
                if existing_zone.includes(succ):
                    subsumed = True
                    break
            if not subsumed:
                idx = len(reached[edge.target])
                reached[edge.target].append((succ, parent_info))
                queue.append((edge.target, idx))

    return ReachabilityResult(
        reachable=False, states_explored=explored,
        zones_created=sum(len(v) for v in reached.values())
    )


def check_safety(ta: TimedAutomaton,
                  unsafe_locations: Set[str]) -> SafetyResult:
    """Check if all unsafe locations are unreachable."""
    result = check_reachability(ta, unsafe_locations)
    if result.reachable:
        return SafetyResult(
            safe=False, violated_location=result.target_location,
            trace=result.trace, states_explored=result.states_explored
        )
    return SafetyResult(safe=True, states_explored=result.states_explored)


# --- Timed Word Acceptance ---

@dataclass(frozen=True)
class TimedAction:
    """An action with a timestamp."""
    action: str
    time: float


def check_timed_word(ta: TimedAutomaton, word: List[TimedAction]) -> bool:
    """Check if a timed automaton accepts a timed word."""
    clocks = sorted(ta.clocks)
    valuation = {c: 0.0 for c in clocks}
    current_time = 0.0
    location = ta.initial

    for timed_action in word:
        # Time must be non-decreasing
        if timed_action.time < current_time:
            return False
        delay = timed_action.time - current_time

        # Let time elapse
        for c in clocks:
            valuation[c] += delay
        current_time = timed_action.time

        # Check location invariant holds after time elapse
        inv = ta.get_invariant(location)
        if not inv.evaluate(valuation):
            return False

        # Find matching edge
        found = False
        for edge in ta.get_edges_from(location):
            if edge.label != timed_action.action:
                continue
            if not edge.guard.evaluate(valuation):
                continue
            # Take this edge
            for c in edge.resets:
                valuation[c] = 0.0
            location = edge.target
            # Check target invariant
            if not ta.get_invariant(location).evaluate(valuation):
                continue
            found = True
            break
        if not found:
            return False

    return location in ta.accepting if ta.accepting else True


# --- Product Construction ---

def product(ta1: TimedAutomaton, ta2: TimedAutomaton) -> TimedAutomaton:
    """Synchronous product of two timed automata over shared alphabet."""
    shared = ta1.alphabet & ta2.alphabet
    ta1_only = ta1.alphabet - shared
    ta2_only = ta2.alphabet - shared

    locations = set()
    edges = []
    invariants = {}
    new_clocks = set()

    # Rename clocks to avoid collision
    c1_map = {c: f"1_{c}" for c in ta1.clocks}
    c2_map = {c: f"2_{c}" for c in ta2.clocks}
    new_clocks = {f"1_{c}" for c in ta1.clocks} | {f"2_{c}" for c in ta2.clocks}

    def rename_guard(g: Guard, cmap: Dict[str, str]) -> Guard:
        new_constraints = []
        for c in g.constraints:
            c1 = cmap.get(c.clock1, c.clock1)
            c2 = cmap.get(c.clock2, c.clock2) if c.clock2 else None
            new_constraints.append(ClockConstraint(c1, c2, c.op, c.value))
        return Guard(tuple(new_constraints))

    def rename_resets(resets: FrozenSet[str], cmap: Dict[str, str]) -> FrozenSet[str]:
        return frozenset(cmap.get(c, c) for c in resets)

    for l1 in ta1.locations:
        for l2 in ta2.locations:
            loc = f"{l1},{l2}"
            locations.add(loc)
            # Combine invariants
            inv1 = rename_guard(ta1.get_invariant(l1), c1_map)
            inv2 = rename_guard(ta2.get_invariant(l2), c2_map)
            combined = guard_and(inv1, inv2)
            if not combined.is_true():
                invariants[loc] = combined

    # Shared actions: both must synchronize
    for l1 in ta1.locations:
        for l2 in ta2.locations:
            for e1 in ta1.get_edges_from(l1):
                if e1.label in shared:
                    for e2 in ta2.get_edges_from(l2):
                        if e2.label == e1.label:
                            src = f"{l1},{l2}"
                            tgt = f"{e1.target},{e2.target}"
                            g = guard_and(
                                rename_guard(e1.guard, c1_map),
                                rename_guard(e2.guard, c2_map)
                            )
                            r = rename_resets(e1.resets, c1_map) | rename_resets(e2.resets, c2_map)
                            edges.append(Edge(src, tgt, e1.label, g, r))

    # ta1-only actions: ta1 moves, ta2 stays
    for l1 in ta1.locations:
        for l2 in ta2.locations:
            for e1 in ta1.get_edges_from(l1):
                if e1.label in ta1_only:
                    src = f"{l1},{l2}"
                    tgt = f"{e1.target},{l2}"
                    g = rename_guard(e1.guard, c1_map)
                    r = rename_resets(e1.resets, c1_map)
                    edges.append(Edge(src, tgt, e1.label, g, r))

    # ta2-only actions: ta2 moves, ta1 stays
    for l1 in ta1.locations:
        for l2 in ta2.locations:
            for e2 in ta2.get_edges_from(l2):
                if e2.label in ta2_only:
                    src = f"{l1},{l2}"
                    tgt = f"{l1},{e2.target}"
                    g = rename_guard(e2.guard, c2_map)
                    r = rename_resets(e2.resets, c2_map)
                    edges.append(Edge(src, tgt, e2.label, g, r))

    initial = f"{ta1.initial},{ta2.initial}"
    accepting = set()
    if ta1.accepting and ta2.accepting:
        for a1 in ta1.accepting:
            for a2 in ta2.accepting:
                accepting.add(f"{a1},{a2}")
    elif ta1.accepting:
        for a1 in ta1.accepting:
            for l2 in ta2.locations:
                accepting.add(f"{a1},{l2}")
    elif ta2.accepting:
        for l1 in ta1.locations:
            for a2 in ta2.accepting:
                accepting.add(f"{l1},{a2}")

    return TimedAutomaton(
        locations=locations, initial=initial, clocks=new_clocks,
        edges=edges, invariants=invariants, accepting=accepting,
        alphabet=ta1.alphabet | ta2.alphabet
    )


# --- Diagnostic Utilities ---

def zone_graph_summary(ta: TimedAutomaton) -> str:
    """Human-readable summary of zone graph exploration."""
    reached, stats = explore_zone_graph(ta)
    lines = [f"Timed Automaton: {len(ta.locations)} locations, {len(ta.clocks)} clocks, {len(ta.edges)} edges"]
    lines.append(f"Zone graph: {stats.n_symbolic_states} symbolic states, {stats.n_zones} zones")
    for loc in sorted(ta.locations):
        zones = reached.get(loc, [])
        if zones:
            lines.append(f"  {loc}: {len(zones)} zone(s)")
            for z in zones:
                lines.append(f"    {z}")
    return "\n".join(lines)


def check_empty_language(ta: TimedAutomaton) -> bool:
    """Check if the language (set of accepted timed words) is empty."""
    if not ta.accepting:
        return True
    result = check_reachability(ta, ta.accepting)
    return not result.reachable


def check_language_inclusion(ta1: TimedAutomaton, ta2: TimedAutomaton) -> Tuple[bool, Optional[Trace]]:
    """
    Approximate language inclusion check: L(ta1) subset L(ta2)?
    Uses product with complement check (exact for deterministic ta2).
    Returns (included, counterexample_trace).
    Note: True result is approximate for nondeterministic ta2.
    """
    # For safety properties expressed as automata, this is useful
    # Full complement construction requires region automata (expensive)
    # We do: product of ta1 with ta2's unsafe states
    if not ta2.accepting:
        return (True, None)

    # Check if ta1 can reach a state where ta2 is NOT accepting
    non_accepting = ta2.locations - ta2.accepting
    if not non_accepting:
        return (True, None)

    prod = product(ta1, ta2)
    # Target: ta1 in accepting, ta2 in non-accepting
    targets = set()
    for l1 in (ta1.accepting if ta1.accepting else ta1.locations):
        for l2 in non_accepting:
            targets.add(f"{l1},{l2}")

    result = check_reachability(prod, targets)
    return (not result.reachable, result.trace)


# --- Convenience Builders ---

def simple_ta(locations: List[str], initial: str, clocks: List[str],
              edges: List[Tuple], invariants: Dict[str, Guard] = None,
              accepting: Set[str] = None) -> TimedAutomaton:
    """
    Build a TA from simplified edge tuples:
    (src, tgt, label, guard, resets) where resets is a list of clock names.
    """
    edge_list = []
    for e in edges:
        src, tgt, label, guard, resets = e
        edge_list.append(Edge(src, tgt, label, guard, frozenset(resets)))
    return TimedAutomaton(
        locations=set(locations),
        initial=initial,
        clocks=set(clocks),
        edges=edge_list,
        invariants=invariants or {},
        accepting=accepting or set()
    )


# --- Example Systems ---

def fischer_mutex(n: int = 2) -> TimedAutomaton:
    """
    Fischer's mutual exclusion protocol for n processes (simplified 2-process).

    Each process i has 3 locations: idle, req, wait.
    - idle: not competing. Transition to req when id==0, reset x_i.
    - req: write id=i. Invariant: x_i <= delta (must write quickly). Go to wait.
    - wait: read id. If id==i after x_i >= Delta, enter CS. Otherwise back to idle.

    Timing: delta < Delta ensures mutual exclusion (last writer wins).
    Shared variable 'id' is encoded in location names.
    """
    if n != 2:
        raise ValueError("Only n=2 supported in this simplified version")

    delta = 1   # upper bound: must write within delta
    big_delta = 2  # lower bound: must wait at least Delta before entering CS

    locs = set()
    edges_list = []
    invariants = {}

    # Process states: idle, req, wait, cs
    proc_states = ['idle', 'req', 'wait', 'cs']
    # Shared variable id: 0 (free), 1, 2
    for s1 in proc_states:
        for s2 in proc_states:
            for vid in range(3):
                loc = f"{s1}_{s2}_{vid}"
                locs.add(loc)
                # Invariant: req state has x_i <= delta
                inv_cs = []
                if s1 == 'req':
                    inv_cs.append(ClockConstraint('x1', None, CompOp.LE, delta))
                if s2 == 'req':
                    inv_cs.append(ClockConstraint('x2', None, CompOp.LE, delta))
                if inv_cs:
                    invariants[loc] = Guard(tuple(inv_cs))

    # Process 1 transitions
    for s2 in proc_states:
        for vid in range(3):
            # idle -> req: guard id==0, reset x1, set id=1
            if vid == 0:
                src = f"idle_{s2}_{vid}"
                tgt = f"req_{s2}_1"  # id becomes 1
                edges_list.append(Edge(src, tgt, "try1", true_guard(), frozenset({'x1'})))

            # req -> wait: invariant forces leaving within delta, go to wait
            if vid == 1:  # only when we own it
                src = f"req_{s2}_{vid}"
                tgt = f"wait_{s2}_{vid}"
                edges_list.append(Edge(src, tgt, "write1", true_guard(), frozenset()))
                # Note: don't reset x1 -- we need to measure total time since try1

            # wait -> cs: guard id==1 AND x1 >= Delta
            if vid == 1:
                src = f"wait_{s2}_{vid}"
                tgt = f"cs_{s2}_{vid}"
                edges_list.append(Edge(src, tgt, "enter1",
                                       clock_geq('x1', big_delta), frozenset()))

            # wait -> idle: guard id != 1 (someone else took it)
            if vid != 1:
                src = f"wait_{s2}_{vid}"
                tgt = f"idle_{s2}_{vid}"
                edges_list.append(Edge(src, tgt, "abort1", true_guard(), frozenset()))

            # cs -> idle: release, id=0
            src = f"cs_{s2}_{vid}"
            tgt = f"idle_{s2}_0"
            edges_list.append(Edge(src, tgt, "exit1", true_guard(), frozenset()))

    # Process 2 transitions (symmetric)
    for s1 in proc_states:
        for vid in range(3):
            # idle -> req: guard id==0, reset x2, set id=2
            if vid == 0:
                src = f"{s1}_idle_{vid}"
                tgt = f"{s1}_req_2"
                edges_list.append(Edge(src, tgt, "try2", true_guard(), frozenset({'x2'})))

            if vid == 2:
                src = f"{s1}_req_{vid}"
                tgt = f"{s1}_wait_{vid}"
                edges_list.append(Edge(src, tgt, "write2", true_guard(), frozenset()))

            if vid == 2:
                src = f"{s1}_wait_{vid}"
                tgt = f"{s1}_cs_{vid}"
                edges_list.append(Edge(src, tgt, "enter2",
                                       clock_geq('x2', big_delta), frozenset()))

            if vid != 2:
                src = f"{s1}_wait_{vid}"
                tgt = f"{s1}_idle_{vid}"
                edges_list.append(Edge(src, tgt, "abort2", true_guard(), frozenset()))

            src = f"{s1}_cs_{vid}"
            tgt = f"{s1}_idle_0"
            edges_list.append(Edge(src, tgt, "exit2", true_guard(), frozenset()))

    initial = "idle_idle_0"
    unsafe = set()
    for vid in range(3):
        unsafe.add(f"cs_cs_{vid}")

    return TimedAutomaton(
        locations=locs, initial=initial, clocks={'x1', 'x2'},
        edges=edges_list, invariants=invariants
    ), unsafe


def train_gate_controller() -> TimedAutomaton:
    """
    Classic train-gate controller example.
    Train approaches, controller lowers gate, train passes, gate raises.
    Timing constraints ensure gate is down before train arrives.
    """
    locs = {'idle', 'approaching', 'lowering', 'down', 'crossing', 'raising'}
    edges = [
        Edge('idle', 'approaching', 'approach', true_guard(), frozenset({'x'})),
        Edge('approaching', 'lowering', 'lower',
             clock_leq('x', 5), frozenset({'x'})),
        Edge('lowering', 'down', 'lowered',
             guard_and(clock_geq('x', 1), clock_leq('x', 2)), frozenset({'x'})),
        Edge('down', 'crossing', 'enter',
             clock_geq('x', 0), frozenset({'x'})),
        Edge('crossing', 'raising', 'exit',
             clock_leq('x', 5), frozenset({'x'})),
        Edge('raising', 'idle', 'raised',
             guard_and(clock_geq('x', 1), clock_leq('x', 2)), frozenset({'x'})),
    ]
    invariants = {
        'approaching': clock_leq('x', 5),
        'lowering': clock_leq('x', 2),
        'crossing': clock_leq('x', 5),
        'raising': clock_leq('x', 2),
    }
    return TimedAutomaton(
        locations=locs, initial='idle', clocks={'x'},
        edges=edges, invariants=invariants, accepting={'idle'}
    )


def simple_light_timer() -> TimedAutomaton:
    """Simple light with timer: on for exactly 3 time units, then off."""
    return simple_ta(
        locations=['off', 'on'],
        initial='off',
        clocks=['t'],
        edges=[
            ('off', 'on', 'press', true_guard(), ['t']),
            ('on', 'off', 'timeout', clock_eq('t', 3), []),
        ],
        invariants={'on': clock_leq('t', 3)},
        accepting={'off'}
    )
