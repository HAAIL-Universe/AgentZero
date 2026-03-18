"""
V202: Timed Games
Composes V118 (timed automata) + V156 (parity games) + V160 (energy games)

Two-player games on timed automata where players choose delays and transitions.
Zone-based symbolic algorithms for reachability, safety, Buchi, and timed energy games.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V118_timed_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, FrozenSet
from enum import Enum
from copy import deepcopy

from timed_automata import (
    TimedAutomaton, Edge, Guard, Zone, ClockConstraint, CompOp,
    true_guard, clock_leq, clock_lt, clock_geq, clock_gt, clock_eq,
    clock_diff_leq, clock_diff_geq, guard_and, initial_zone
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Player(Enum):
    EVEN = 0  # Controller / Reachability player
    ODD = 1   # Environment / Safety player

    @property
    def opponent(self):
        return Player.ODD if self == Player.EVEN else Player.EVEN


@dataclass
class TimedGame:
    """Two-player game on a timed automaton."""
    locations: Set[str] = field(default_factory=set)
    initial: str = ""
    clocks: Set[str] = field(default_factory=set)
    edges: List[Edge] = field(default_factory=list)
    invariants: Dict[str, Guard] = field(default_factory=dict)
    owner: Dict[str, Player] = field(default_factory=dict)  # location -> player
    # Optional attributes for game conditions
    accepting: Set[str] = field(default_factory=set)        # for Buchi
    priorities: Dict[str, int] = field(default_factory=dict) # for parity
    weights: Dict[int, int] = field(default_factory=dict)    # edge_index -> weight (energy)

    def add_location(self, loc: str, player: Player, invariant: Optional[Guard] = None,
                     priority: int = 0):
        self.locations.add(loc)
        self.owner[loc] = player
        if invariant is not None:
            self.invariants[loc] = invariant
        if priority != 0:
            self.priorities[loc] = priority

    def add_edge(self, source: str, target: str, label: str,
                 guard: Optional[Guard] = None, resets: Optional[FrozenSet[str]] = None,
                 weight: int = 0) -> int:
        g = guard if guard is not None else true_guard()
        r = resets if resets is not None else frozenset()
        idx = len(self.edges)
        self.edges.append(Edge(source, target, label, g, r))
        if weight != 0:
            self.weights[idx] = weight
        return idx

    def get_edges_from(self, loc: str) -> List[Tuple[int, Edge]]:
        return [(i, e) for i, e in enumerate(self.edges) if e.source == loc]

    def get_invariant(self, loc: str) -> Guard:
        return self.invariants.get(loc, true_guard())

    def to_timed_automaton(self) -> TimedAutomaton:
        return TimedAutomaton(
            locations=set(self.locations),
            initial=self.initial,
            clocks=set(self.clocks),
            edges=list(self.edges),
            invariants=dict(self.invariants),
            accepting=set(self.accepting)
        )

    def max_constant(self) -> int:
        """Largest constant appearing in guards/invariants."""
        m = 0
        for e in self.edges:
            for cc in e.guard.constraints:
                m = max(m, abs(cc.value))
        for g in self.invariants.values():
            for cc in g.constraints:
                m = max(m, abs(cc.value))
        return m


# ---------------------------------------------------------------------------
# Symbolic state: (location, zone)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymState:
    location: str
    zone_id: int


class ZoneStore:
    """Deduplicated zone storage."""
    def __init__(self):
        self._zones: List[Zone] = []

    def add(self, z: Zone) -> int:
        idx = len(self._zones)
        self._zones.append(z)
        return idx

    def get(self, idx: int) -> Zone:
        return self._zones[idx]

    def __len__(self):
        return len(self._zones)


# ---------------------------------------------------------------------------
# Zone operations for games
# ---------------------------------------------------------------------------

def _successor_zone(zone: Zone, edge: Edge, target_inv: Guard) -> Optional[Zone]:
    """Compute successor zone: guard -> reset -> future -> invariant."""
    z = zone.copy()
    z.apply_guard(edge.guard)
    if z.is_empty():
        return None
    z.apply_resets(edge.resets)
    z.apply_guard(target_inv)
    if z.is_empty():
        return None
    z.future()
    z.apply_guard(target_inv)
    if z.is_empty():
        return None
    z.canonicalize()
    return z


def _timed_predecessor(game: TimedGame, target_locs: Dict[str, List[Zone]],
                       source_loc: str, source_zone: Zone) -> Optional[Zone]:
    """
    Controllable predecessor for EVEN: can the controller choose a delay + edge
    from source_loc/source_zone to reach some zone in target_locs?

    Returns the sub-zone of source_zone from which this is possible (or None).
    """
    clock_names = sorted(game.clocks)
    result_zone = Zone(len(clock_names), clock_names)
    # Start empty -- union of all predecessor zones
    # Make an explicitly empty zone
    result_zone.constrain(ClockConstraint(clock_names[0], None, CompOp.LT, 0))
    result_zone.canonicalize()

    source_inv = game.get_invariant(source_loc)

    for idx, edge in enumerate(game.edges):
        if edge.source != source_loc:
            continue
        target_zones = target_locs.get(edge.target, [])
        for tz in target_zones:
            # Backward: from target zone, undo the edge
            pre = _backward_edge(game, edge, source_inv, tz, clock_names)
            if pre is None or pre.is_empty():
                continue
            # Intersect with source zone
            inter = pre.intersect(source_zone)
            if inter.is_empty():
                continue
            # Union into result (over-approx: just take the widest)
            result_zone = _zone_union_overapprox(result_zone, inter, clock_names)

    if result_zone.is_empty():
        return None
    return result_zone


def _backward_edge(game: TimedGame, edge: Edge, source_inv: Guard,
                   target_zone: Zone, clock_names: List[str]) -> Optional[Zone]:
    """
    Compute the set of clock valuations in source_loc from which
    taking edge reaches target_zone (after delay + edge + delay).

    Backward: undo future, undo reset, apply guard, apply source invariant.
    """
    z = target_zone.copy()

    # Undo time elapse at target: past operator
    # Past of Z = {v - d | v in Z, d >= 0} -- remove lower bounds
    z = _zone_past(z, clock_names)
    # Apply target invariant (must hold during time passage)
    z.apply_guard(game.get_invariant(edge.target))
    if z.is_empty():
        return None

    # Undo resets: for each reset clock, it was 0 after edge, so constrain to 0
    # then free it (unreset = the clock had any value before)
    z = _undo_resets(z, edge.resets, clock_names)
    if z is None or z.is_empty():
        return None

    # Apply edge guard
    z.apply_guard(edge.guard)
    if z.is_empty():
        return None

    # Undo time elapse at source
    z = _zone_past(z, clock_names)
    z.apply_guard(source_inv)
    if z.is_empty():
        return None

    z.canonicalize()
    return z


def _zone_past(z: Zone, clock_names: List[str]) -> Zone:
    """Past operator: {v - d | v in z, d >= 0}. Remove lower bounds on clocks."""
    result = z.copy()
    n = len(clock_names)
    # In DBM: dbm[0][i+1] encodes 0 - clock_i <= bound, i.e., clock_i >= -bound
    # Past removes lower bounds: set dbm[0][i+1] = (0, False) meaning clock_i >= 0
    for i in range(n):
        result.dbm[0][i + 1] = (0, False)  # 0 - clock_i <= 0, i.e., clock_i >= 0
    result.canonicalize()
    return result


def _undo_resets(z: Zone, resets: FrozenSet[str], clock_names: List[str]) -> Optional[Zone]:
    """Undo clock resets: after reset, clock was 0. Before reset, clock was anything."""
    if not resets:
        return z
    result = z.copy()
    for clock in resets:
        if clock not in clock_names:
            continue
        idx = clock_names.index(clock) + 1  # +1 for reference clock 0
        # After reset, clock = 0. In the target zone, clock is constrained.
        # Before the reset, the clock value is unknown (free).
        # So: constrain clock = 0 in the zone (that's the post-reset state),
        # then free the clock dimension.

        # First, constrain to clock = 0 (the reset value)
        result.constrain(ClockConstraint(clock, None, CompOp.EQ, 0))
        if result.is_empty():
            return None

        # Then free the clock: remove all constraints involving this clock
        # This means the PRE-reset value was unconstrained
        n = len(clock_names) + 1
        for j in range(n):
            if j != idx:
                result.dbm[idx][j] = (10**9, False)  # large bound
                result.dbm[j][idx] = (10**9, False)
        # Keep clock >= 0
        result.dbm[idx][0] = (0, True)
    result.canonicalize()
    return result


def _zone_union_overapprox(z1: Zone, z2: Zone, clock_names: List[str]) -> Zone:
    """Over-approximate union of two zones (convex hull)."""
    if z1.is_empty():
        return z2.copy()
    if z2.is_empty():
        return z1.copy()
    # Convex hull: take max of each DBM entry
    n = len(clock_names) + 1
    result = z1.copy()
    for i in range(n):
        for j in range(n):
            v1, s1 = z1.dbm[i][j]
            v2, s2 = z2.dbm[i][j]
            if v1 > v2 or (v1 == v2 and not s1 and s2):
                result.dbm[i][j] = (v1, s1)
            else:
                result.dbm[i][j] = (v2, s2)
    result.canonicalize()
    return result


# ---------------------------------------------------------------------------
# Forward reachability with game structure
# ---------------------------------------------------------------------------

def _explore_game(game: TimedGame) -> Dict[str, List[Zone]]:
    """Forward zone-based exploration of the timed game."""
    clock_names = sorted(game.clocks)
    n_clocks = len(clock_names)

    init_zone = initial_zone(clock_names)
    init_inv = game.get_invariant(game.initial)
    init_zone.apply_guard(init_inv)
    init_zone.future()
    init_zone.apply_guard(init_inv)
    init_zone.canonicalize()

    reachable: Dict[str, List[Zone]] = {loc: [] for loc in game.locations}
    reachable[game.initial].append(init_zone)

    queue = [(game.initial, init_zone)]

    while queue:
        loc, zone = queue.pop(0)
        for idx, edge in game.get_edges_from(loc):
            target_inv = game.get_invariant(edge.target)
            succ = _successor_zone(zone, edge, target_inv)
            if succ is None or succ.is_empty():
                continue
            # Check subsumption
            subsumed = False
            for existing in reachable[edge.target]:
                if existing.includes(succ):
                    subsumed = True
                    break
            if not subsumed:
                # Remove zones subsumed by new one
                reachable[edge.target] = [
                    z for z in reachable[edge.target] if not succ.includes(z)
                ]
                reachable[edge.target].append(succ)
                queue.append((edge.target, succ))

    return reachable


# ---------------------------------------------------------------------------
# Reachability timed game solving (backward fixed point)
# ---------------------------------------------------------------------------

@dataclass
class TimedGameResult:
    """Result of solving a timed game."""
    winner: Dict[str, Player]  # location -> winner (approximate per-location)
    winning_zones_even: Dict[str, List[Zone]]  # zones where Even wins
    winning_zones_odd: Dict[str, List[Zone]]   # zones where Odd wins
    iterations: int = 0
    strategy_edges: Dict[str, List[int]] = field(default_factory=dict)  # loc -> edge indices


def _can_reach_winning(game: TimedGame, loc: str, zone: Zone,
                       win_locs: Set[str]) -> Tuple[bool, List[int]]:
    """
    Check if from (loc, zone), there exists an edge whose successor zone
    overlaps with a reachable zone at a winning location.
    Returns (can_reach, list_of_good_edge_indices).
    """
    good_edges = []
    for idx, edge in game.get_edges_from(loc):
        if edge.target not in win_locs:
            continue
        target_inv = game.get_invariant(edge.target)
        succ = _successor_zone(zone, edge, target_inv)
        if succ is not None and not succ.is_empty():
            good_edges.append(idx)
    return (len(good_edges) > 0, good_edges)


def _all_edges_reach_winning(game: TimedGame, loc: str, zone: Zone,
                             win_locs: Set[str]) -> bool:
    """
    Check if ALL edges from (loc, zone) lead to winning locations.
    Dead end: Even only wins if loc is already a target (in win_locs).
    """
    edges = game.get_edges_from(loc)
    if not edges:
        return loc in win_locs  # dead end -> Even wins only if already at target

    any_takeable = False
    for idx, edge in edges:
        target_inv = game.get_invariant(edge.target)
        succ = _successor_zone(zone, edge, target_inv)
        if succ is not None and not succ.is_empty():
            any_takeable = True
            if edge.target not in win_locs:
                return False
        # If successor is empty/None, this edge is not actually takeable -> skip
    if not any_takeable:
        return loc in win_locs  # effectively a dead end
    # All takeable edges lead to winning locations
    return True


def solve_reachability(game: TimedGame, targets: Set[str],
                       max_iterations: int = 100) -> TimedGameResult:
    """
    Solve timed reachability game: Even tries to reach targets, Odd tries to prevent.

    Forward fixed-point: propagate winning status from targets backward using
    forward successor checks.
    """
    # Forward exploration to find reachable zones
    reachable = _explore_game(game)

    # Winning locations for Even (location-level approximation)
    win_locs: Set[str] = set()
    # Initialize: target locations that are reachable
    for t in targets:
        if reachable.get(t):
            win_locs.add(t)

    # Fixed point: expand winning set
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        changed = False

        for loc in game.locations:
            if loc in win_locs:
                continue
            if not reachable.get(loc):
                continue

            owner = game.owner.get(loc, Player.ODD)

            if owner == Player.EVEN:
                # Even wins if there EXISTS an edge to a winning location
                for rz in reachable[loc]:
                    can, _ = _can_reach_winning(game, loc, rz, win_locs)
                    if can:
                        win_locs.add(loc)
                        changed = True
                        break
            else:
                # Odd location: Even wins only if ALL takeable edges lead to winning
                all_win = True
                for rz in reachable[loc]:
                    if not _all_edges_reach_winning(game, loc, rz, win_locs):
                        all_win = False
                        break
                if all_win:
                    win_locs.add(loc)
                    changed = True

        if not changed:
            break

    # Build result
    win_even: Dict[str, List[Zone]] = {loc: [] for loc in game.locations}
    win_odd: Dict[str, List[Zone]] = {loc: [] for loc in game.locations}
    winner: Dict[str, Player] = {}
    strategy_edges: Dict[str, List[int]] = {}

    for loc in game.locations:
        if loc in win_locs:
            winner[loc] = Player.EVEN
            win_even[loc] = [z.copy() for z in reachable.get(loc, [])]
            if game.owner.get(loc) == Player.EVEN:
                for rz in reachable.get(loc, []):
                    _, edges = _can_reach_winning(game, loc, rz, win_locs)
                    if edges:
                        strategy_edges[loc] = edges
                        break
        else:
            winner[loc] = Player.ODD
            if reachable.get(loc):
                win_odd[loc] = [z.copy() for z in reachable[loc]]

    return TimedGameResult(
        winner=winner,
        winning_zones_even=win_even,
        winning_zones_odd=win_odd,
        iterations=iterations,
        strategy_edges=strategy_edges
    )


def solve_safety(game: TimedGame, unsafe: Set[str],
                 max_iterations: int = 100) -> TimedGameResult:
    """
    Solve timed safety game: Even tries to avoid unsafe locations forever.
    Odd tries to force Even into unsafe locations.

    Backward attractor from unsafe set, respecting ownership.
    """
    reachable = _explore_game(game)

    # Losing locations for Even (unsafe attractor)
    lose_locs: Set[str] = set()
    for u in unsafe:
        if reachable.get(u):
            lose_locs.add(u)

    # Fixed point: expand losing set (attractor for Odd toward unsafe)
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        changed = False

        for loc in game.locations:
            if loc in lose_locs:
                continue
            if not reachable.get(loc):
                continue

            owner = game.owner.get(loc, Player.ODD)

            if owner == Player.ODD:
                # Odd wins if there EXISTS an edge to a losing location
                for rz in reachable[loc]:
                    can, _ = _can_reach_winning(game, loc, rz, lose_locs)
                    if can:
                        lose_locs.add(loc)
                        changed = True
                        break
            else:
                # Even location: Even loses only if ALL takeable edges lead to losing
                all_lose = True
                has_any_edge = False
                for rz in reachable[loc]:
                    edges = game.get_edges_from(loc)
                    for idx, edge in edges:
                        target_inv = game.get_invariant(edge.target)
                        succ = _successor_zone(rz, edge, target_inv)
                        if succ is not None and not succ.is_empty():
                            has_any_edge = True
                            if edge.target not in lose_locs:
                                all_lose = False
                                break
                    if not all_lose:
                        break
                if has_any_edge and all_lose:
                    lose_locs.add(loc)
                    changed = True

        if not changed:
            break

    # Build result
    win_even: Dict[str, List[Zone]] = {loc: [] for loc in game.locations}
    win_odd: Dict[str, List[Zone]] = {loc: [] for loc in game.locations}
    winner: Dict[str, Player] = {}

    for loc in game.locations:
        if loc not in lose_locs:
            winner[loc] = Player.EVEN
            if reachable.get(loc):
                win_even[loc] = [z.copy() for z in reachable[loc]]
        else:
            winner[loc] = Player.ODD
            if reachable.get(loc):
                win_odd[loc] = [z.copy() for z in reachable[loc]]

    return TimedGameResult(
        winner=winner,
        winning_zones_even=win_even,
        winning_zones_odd=win_odd,
        iterations=iterations
    )


# ---------------------------------------------------------------------------
# Buchi timed game (infinitely often visit accepting)
# ---------------------------------------------------------------------------

def solve_buchi(game: TimedGame, accepting: Optional[Set[str]] = None,
                max_iterations: int = 50) -> TimedGameResult:
    """
    Solve timed Buchi game: Even must visit accepting locations infinitely often.

    Nested fixed point at location level:
    1. Start with all reachable locations as candidate set
    2. Inner: compute which locations in candidate can reach accepting (within candidate)
    3. Remove non-reaching locations from candidate
    4. Repeat until stable
    """
    acc = accepting if accepting is not None else game.accepting
    if not acc:
        return TimedGameResult(
            winner={loc: Player.ODD for loc in game.locations},
            winning_zones_even={loc: [] for loc in game.locations},
            winning_zones_odd=_explore_game(game),
            iterations=0
        )

    reachable = _explore_game(game)

    # Candidate winning set: all reachable locations
    candidate: Set[str] = {loc for loc in game.locations if reachable.get(loc)}

    for outer in range(max_iterations):
        # First: remove dead-end locations from candidate
        # (dead ends can't participate in infinite plays)
        dead_ends_removed = True
        while dead_ends_removed:
            dead_ends_removed = False
            to_remove = set()
            for loc in candidate:
                has_edge_in_candidate = False
                for rz in reachable.get(loc, []):
                    for idx, edge in game.get_edges_from(loc):
                        if edge.target not in candidate:
                            continue
                        target_inv = game.get_invariant(edge.target)
                        succ = _successor_zone(rz, edge, target_inv)
                        if succ is not None and not succ.is_empty():
                            has_edge_in_candidate = True
                            break
                    if has_edge_in_candidate:
                        break
                if not has_edge_in_candidate:
                    to_remove.add(loc)
            if to_remove:
                candidate -= to_remove
                dead_ends_removed = True

        # Inner: which locations in candidate can reach accepting?
        # Start from accepting locations that are still in candidate
        can_reach_acc: Set[str] = set(acc) & candidate

        for _ in range(max_iterations):
            inner_changed = False
            for loc in candidate:
                if loc in can_reach_acc:
                    continue
                owner = game.owner.get(loc, Player.ODD)
                if owner == Player.EVEN:
                    # Even: exists edge to can_reach_acc within candidate
                    for rz in reachable.get(loc, []):
                        for idx, edge in game.get_edges_from(loc):
                            if edge.target not in can_reach_acc:
                                continue
                            if edge.target not in candidate:
                                continue
                            target_inv = game.get_invariant(edge.target)
                            succ = _successor_zone(rz, edge, target_inv)
                            if succ is not None and not succ.is_empty():
                                can_reach_acc.add(loc)
                                inner_changed = True
                                break
                        if loc in can_reach_acc:
                            break
                else:
                    # Odd: all takeable edges within candidate reach acc
                    all_ok = True
                    has_edge = False
                    for rz in reachable.get(loc, []):
                        for idx, edge in game.get_edges_from(loc):
                            if edge.target not in candidate:
                                continue
                            target_inv = game.get_invariant(edge.target)
                            succ = _successor_zone(rz, edge, target_inv)
                            if succ is not None and not succ.is_empty():
                                has_edge = True
                                if edge.target not in can_reach_acc:
                                    all_ok = False
                                    break
                        if not all_ok:
                            break
                    if has_edge and all_ok:
                        can_reach_acc.add(loc)
                        inner_changed = True

            if not inner_changed:
                break

        # Remove locations that can't reach accepting
        new_candidate = candidate & can_reach_acc
        if new_candidate == candidate:
            break
        candidate = new_candidate

    # Build result
    win_even: Dict[str, List[Zone]] = {loc: [] for loc in game.locations}
    win_odd: Dict[str, List[Zone]] = {loc: [] for loc in game.locations}
    winner = {}

    for loc in game.locations:
        if loc in candidate:
            winner[loc] = Player.EVEN
            win_even[loc] = [z.copy() for z in reachable.get(loc, [])]
        else:
            winner[loc] = Player.ODD
            if reachable.get(loc):
                win_odd[loc] = [z.copy() for z in reachable[loc]]

    return TimedGameResult(
        winner=winner,
        winning_zones_even=win_even,
        winning_zones_odd=win_odd,
        iterations=outer + 1
    )


# ---------------------------------------------------------------------------
# Timed energy game
# ---------------------------------------------------------------------------

@dataclass
class TimedEnergyResult:
    """Result of a timed energy game."""
    winning_locations: Set[str]  # locations where Even can win
    min_energy: Dict[str, Optional[int]]  # minimum initial energy per location
    iterations: int = 0
    strategy_edges: Dict[str, List[int]] = field(default_factory=dict)


def solve_timed_energy(game: TimedGame, max_iterations: int = 100) -> TimedEnergyResult:
    """
    Solve timed energy game: Even must keep energy >= 0 forever.
    Edges have integer weights (game.weights). Time passage has zero weight.

    Reduction: build finite abstraction (zone graph), then solve energy game on it.
    """
    from energy_games import (
        EnergyGame as EG, Player as EPlayer, solve_energy
    )

    clock_names = sorted(game.clocks)

    # Build zone graph
    reachable = _explore_game(game)

    # Assign IDs to symbolic states
    state_ids: Dict[Tuple[str, int], int] = {}
    zone_map: Dict[str, List[Tuple[int, Zone]]] = {}  # loc -> [(zone_idx, zone)]
    next_id = 0

    for loc in game.locations:
        zone_map[loc] = []
        for i, z in enumerate(reachable.get(loc, [])):
            sid = next_id
            state_ids[(loc, i)] = sid
            zone_map[loc].append((sid, z))
            next_id += 1

    if next_id == 0:
        return TimedEnergyResult(
            winning_locations=set(),
            min_energy={loc: None for loc in game.locations},
            iterations=0
        )

    # Build energy game on zone graph
    eg = EG()
    for loc in game.locations:
        p = EPlayer.EVEN if game.owner.get(loc) == Player.EVEN else EPlayer.ODD
        for sid, _ in zone_map.get(loc, []):
            eg.add_vertex(sid, p)

    # Add edges based on zone successors
    for loc in game.locations:
        for zi, (sid, sz) in enumerate(zone_map.get(loc, [])):
            for eidx, edge in game.get_edges_from(loc):
                target_inv = game.get_invariant(edge.target)
                succ = _successor_zone(sz, edge, target_inv)
                if succ is None or succ.is_empty():
                    continue
                w = game.weights.get(eidx, 0)
                # Find matching target zone
                for tj, (tid, tz) in enumerate(zone_map.get(edge.target, [])):
                    if tz.includes(succ) or succ.includes(tz):
                        eg.add_edge(sid, tid, w)

    # Ensure all vertices have at least a self-loop check
    # (dead ends handled by energy game solver)

    # Solve
    result = solve_energy(eg)

    # Map back to locations
    winning_locs = set()
    min_energy: Dict[str, Optional[int]] = {}

    for loc in game.locations:
        loc_min = None
        for sid, _ in zone_map.get(loc, []):
            e = result.min_energy.get(sid)
            if e is not None:
                if loc_min is None or e < loc_min:
                    loc_min = e
        min_energy[loc] = loc_min
        if loc_min is not None:
            winning_locs.add(loc)

    # Strategy edges
    strategy_edges = {}
    for loc in game.locations:
        if game.owner.get(loc) == Player.EVEN and loc in winning_locs:
            edges = []
            for eidx, edge in game.get_edges_from(loc):
                if edge.target in winning_locs:
                    edges.append(eidx)
            if edges:
                strategy_edges[loc] = edges

    return TimedEnergyResult(
        winning_locations=winning_locs,
        min_energy=min_energy,
        iterations=0,
        strategy_edges=strategy_edges
    )


# ---------------------------------------------------------------------------
# Simulation and verification
# ---------------------------------------------------------------------------

@dataclass
class TimedGameTrace:
    """Trace of a timed game play."""
    steps: List[Tuple[str, float, str, str]]  # (location, delay, action, target)
    winner: Optional[Player] = None


def simulate_play(game: TimedGame, strategy_even: Dict[str, Tuple[float, int]],
                  strategy_odd: Dict[str, Tuple[float, int]],
                  max_steps: int = 20) -> TimedGameTrace:
    """
    Simulate a play. Strategies map location -> (delay, edge_index).
    Returns trace of (location, delay, action, target_location).
    """
    clock_names = sorted(game.clocks)
    valuation = {c: 0.0 for c in clock_names}
    loc = game.initial
    steps = []

    for _ in range(max_steps):
        owner = game.owner.get(loc, Player.ODD)
        strategy = strategy_even if owner == Player.EVEN else strategy_odd

        if loc not in strategy:
            break

        delay, edge_idx = strategy[loc]

        # Advance clocks by delay
        for c in clock_names:
            valuation[c] += delay

        # Check invariant
        inv = game.get_invariant(loc)
        if not inv.evaluate(valuation):
            break

        # Check edge guard
        if edge_idx >= len(game.edges):
            break
        edge = game.edges[edge_idx]
        if edge.source != loc:
            break
        if not edge.guard.evaluate(valuation):
            break

        steps.append((loc, delay, edge.label, edge.target))

        # Apply resets
        for c in edge.resets:
            valuation[c] = 0.0

        loc = edge.target

    return TimedGameTrace(steps=steps)


def check_timed_strategy(game: TimedGame, targets: Set[str],
                         strategy: Dict[str, Tuple[float, int]],
                         max_steps: int = 50) -> bool:
    """Check if a concrete strategy for Even reaches targets."""
    clock_names = sorted(game.clocks)
    valuation = {c: 0.0 for c in clock_names}
    loc = game.initial

    for _ in range(max_steps):
        if loc in targets:
            return True
        if loc not in strategy:
            return False

        delay, edge_idx = strategy[loc]
        for c in clock_names:
            valuation[c] += delay

        inv = game.get_invariant(loc)
        if not inv.evaluate(valuation):
            return False

        edge = game.edges[edge_idx]
        if edge.source != loc or not edge.guard.evaluate(valuation):
            return False

        for c in edge.resets:
            valuation[c] = 0.0
        loc = edge.target

    return loc in targets


# ---------------------------------------------------------------------------
# Analysis and comparison
# ---------------------------------------------------------------------------

def game_statistics(game: TimedGame) -> Dict:
    """Statistics about a timed game."""
    even_locs = sum(1 for p in game.owner.values() if p == Player.EVEN)
    odd_locs = sum(1 for p in game.owner.values() if p == Player.ODD)
    return {
        'locations': len(game.locations),
        'even_locations': even_locs,
        'odd_locations': odd_locs,
        'edges': len(game.edges),
        'clocks': len(game.clocks),
        'max_constant': game.max_constant(),
        'has_weights': bool(game.weights),
        'has_priorities': bool(game.priorities),
        'accepting': len(game.accepting),
    }


def compare_reachability_safety(game: TimedGame, targets: Set[str],
                                unsafe: Optional[Set[str]] = None) -> Dict:
    """Compare reachability and safety game results."""
    reach = solve_reachability(game, targets)

    if unsafe is None:
        unsafe = game.locations - targets
    safety = solve_safety(game, unsafe)

    return {
        'reachability': {
            'even_wins': [l for l, w in reach.winner.items() if w == Player.EVEN],
            'odd_wins': [l for l, w in reach.winner.items() if w == Player.ODD],
            'iterations': reach.iterations,
        },
        'safety': {
            'even_wins': [l for l, w in safety.winner.items() if w == Player.EVEN],
            'odd_wins': [l for l, w in safety.winner.items() if w == Player.ODD],
            'iterations': safety.iterations,
        }
    }


def game_summary(game: TimedGame) -> str:
    """Human-readable summary of a timed game."""
    stats = game_statistics(game)
    lines = [
        f"Timed Game: {stats['locations']} locations "
        f"({stats['even_locations']} Even, {stats['odd_locations']} Odd)",
        f"  Edges: {stats['edges']}, Clocks: {stats['clocks']}",
        f"  Max constant: {stats['max_constant']}",
    ]
    if stats['has_weights']:
        lines.append(f"  Energy game (weighted edges)")
    if stats['has_priorities']:
        lines.append(f"  Parity game (priorities on locations)")
    if stats['accepting']:
        lines.append(f"  Buchi game ({stats['accepting']} accepting locations)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example timed games
# ---------------------------------------------------------------------------

def cat_mouse_game() -> TimedGame:
    """
    Classic cat-and-mouse timed game.
    Mouse (Even) tries to reach the hole. Cat (Odd) tries to catch it.
    Mouse must reach hole within 5 time units. Cat intercepts if possible.
    """
    game = TimedGame(
        locations={'start', 'path1', 'path2', 'hole', 'caught'},
        initial='start',
        clocks={'x'},
        owner={
            'start': Player.EVEN,   # Mouse chooses path
            'path1': Player.ODD,    # Cat intercepts on path1
            'path2': Player.EVEN,   # Mouse controls timing on path2
            'hole': Player.EVEN,    # Mouse wins
            'caught': Player.ODD,   # Cat wins
        },
        invariants={
            'start': clock_leq('x', 1),
            'path1': clock_leq('x', 5),
            'path2': clock_leq('x', 5),
        },
        accepting={'hole'},
    )
    # Mouse chooses path
    game.add_edge('start', 'path1', 'go_path1', clock_leq('x', 1))
    game.add_edge('start', 'path2', 'go_path2', clock_leq('x', 1))
    # Cat controls path1: can catch or let through
    game.add_edge('path1', 'caught', 'catch1', clock_leq('x', 3))
    game.add_edge('path1', 'hole', 'escape1', clock_geq('x', 3))
    # Mouse controls path2: can escape (timing advantage)
    game.add_edge('path2', 'caught', 'catch2', clock_geq('x', 4))
    game.add_edge('path2', 'hole', 'escape2', clock_geq('x', 2))

    return game


def resource_game() -> TimedGame:
    """
    Timed energy game: controller manages a resource that drains over time.
    Must periodically recharge. Environment controls drain rate.
    """
    game = TimedGame(
        locations={'idle', 'working', 'charging', 'depleted'},
        initial='idle',
        clocks={'x'},
        owner={
            'idle': Player.EVEN,
            'working': Player.ODD,
            'charging': Player.EVEN,
            'depleted': Player.ODD,
        },
        invariants={
            'idle': clock_leq('x', 2),
            'working': clock_leq('x', 5),
            'charging': clock_leq('x', 3),
        }
    )
    # Start working
    e0 = game.add_edge('idle', 'working', 'start', clock_leq('x', 2),
                       frozenset({'x'}), weight=-1)
    # Environment: work drains or finishes
    e1 = game.add_edge('working', 'depleted', 'drain', clock_geq('x', 3),
                       weight=-5)
    e2 = game.add_edge('working', 'idle', 'finish', clock_geq('x', 1),
                       frozenset({'x'}), weight=-2)
    # Controller charges
    e3 = game.add_edge('idle', 'charging', 'charge', clock_leq('x', 1),
                       frozenset({'x'}), weight=0)
    e4 = game.add_edge('charging', 'idle', 'charged', clock_geq('x', 2),
                       frozenset({'x'}), weight=5)

    return game


def traffic_light_game() -> TimedGame:
    """
    Timed game: controller sets traffic light timing. Environment is traffic.
    Controller (Even) wants safe + efficient. Environment (Odd) tries to cause crashes.
    """
    game = TimedGame(
        locations={'green_ns', 'yellow_ns', 'red_ns', 'green_ew', 'yellow_ew', 'red_ew', 'crash'},
        initial='green_ns',
        clocks={'x'},
        owner={
            'green_ns': Player.EVEN,
            'yellow_ns': Player.EVEN,
            'red_ns': Player.EVEN,
            'green_ew': Player.EVEN,
            'yellow_ew': Player.EVEN,
            'red_ew': Player.EVEN,
            'crash': Player.ODD,
        },
        invariants={
            'green_ns': clock_leq('x', 10),
            'yellow_ns': clock_leq('x', 3),
            'red_ns': clock_leq('x', 2),
            'green_ew': clock_leq('x', 10),
            'yellow_ew': clock_leq('x', 3),
            'red_ew': clock_leq('x', 2),
        }
    )
    # NS cycle
    game.add_edge('green_ns', 'yellow_ns', 'ns_yellow',
                  clock_geq('x', 3), frozenset({'x'}))
    game.add_edge('yellow_ns', 'red_ns', 'ns_red',
                  clock_geq('x', 2), frozenset({'x'}))
    game.add_edge('red_ns', 'green_ew', 'switch_ew',
                  clock_geq('x', 1), frozenset({'x'}))
    # EW cycle
    game.add_edge('green_ew', 'yellow_ew', 'ew_yellow',
                  clock_geq('x', 3), frozenset({'x'}))
    game.add_edge('yellow_ew', 'red_ew', 'ew_red',
                  clock_geq('x', 2), frozenset({'x'}))
    game.add_edge('red_ew', 'green_ns', 'switch_ns',
                  clock_geq('x', 1), frozenset({'x'}))

    return game


def fischer_game(n: int = 2) -> TimedGame:
    """
    Fischer's mutual exclusion as a timed game.
    Process 1 (Even) tries to enter critical section.
    Environment (Odd) may cause conflict at the interference point.
    Key timing: process must wait >= delta2 (3) before entering CS.
    Environment can only interfere during the wait (Odd controls interference point).
    """
    game = TimedGame(
        locations={'idle', 'set', 'wait', 'cs', 'interfere_point', 'conflict'},
        initial='idle',
        clocks={'x'},
        owner={
            'idle': Player.EVEN,
            'set': Player.EVEN,
            'wait': Player.EVEN,            # Process controls waiting
            'cs': Player.EVEN,
            'interfere_point': Player.ODD,  # Environment decides to interfere or not
            'conflict': Player.ODD,
        },
        invariants={
            'set': clock_leq('x', 2),       # Must set id within delta1=2
            'interfere_point': clock_leq('x', 3),  # Must decide within window
        }
    )
    # Process sets id
    game.add_edge('idle', 'set', 'try', true_guard(), frozenset({'x'}))
    # After setting, go to interference check point
    game.add_edge('set', 'interfere_point', 'set_id', clock_leq('x', 2), frozenset({'x'}))
    # Environment: interfere only in early window, or let through after delay
    game.add_edge('interfere_point', 'conflict', 'interfere', clock_leq('x', 1))
    game.add_edge('interfere_point', 'wait', 'no_interfere', clock_geq('x', 2))
    # Process waits and enters CS
    game.add_edge('wait', 'cs', 'enter', clock_geq('x', 3))
    # Retry from conflict
    game.add_edge('conflict', 'idle', 'retry', true_guard(), frozenset({'x'}))
    # Exit CS
    game.add_edge('cs', 'idle', 'exit', true_guard(), frozenset({'x'}))

    return game


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def make_timed_game(locations: List[Tuple[str, str, Optional[str]]],
                    edges: List[Tuple[str, str, str, Optional[str], Optional[str]]],
                    initial: str, clocks: List[str]) -> TimedGame:
    """
    Build a timed game from tuples.
    locations: [(name, 'even'|'odd', invariant_str|None), ...]
    edges: [(src, dst, label, guard_str|None, resets_str|None), ...]

    Guard strings: 'x<=5', 'x>=3', 'x<2', 'x>1', 'x==4', 'x-y<=3'
    Invariant strings: same format
    Resets: 'x,y' or 'x'
    """
    game = TimedGame(initial=initial, clocks=set(clocks))

    for loc_tuple in locations:
        name = loc_tuple[0]
        player = Player.EVEN if loc_tuple[1].lower() == 'even' else Player.ODD
        inv = None
        if len(loc_tuple) > 2 and loc_tuple[2]:
            inv = _parse_guard_str(loc_tuple[2])
        game.add_location(name, player, inv)

    for edge_tuple in edges:
        src, dst, label = edge_tuple[0], edge_tuple[1], edge_tuple[2]
        guard = None
        if len(edge_tuple) > 3 and edge_tuple[3]:
            guard = _parse_guard_str(edge_tuple[3])
        resets = frozenset()
        if len(edge_tuple) > 4 and edge_tuple[4]:
            resets = frozenset(edge_tuple[4].split(','))
        game.add_edge(src, dst, label, guard, resets)

    return game


def _parse_guard_str(s: str) -> Guard:
    """Parse simple guard string like 'x<=5' or 'x>=3 && y<2'."""
    parts = s.split('&&')
    constraints = []
    for part in parts:
        part = part.strip()
        constraints.append(_parse_constraint_str(part))
    return Guard(constraints=tuple(constraints))


def _parse_constraint_str(s: str) -> ClockConstraint:
    """Parse 'x<=5', 'x-y>=3', etc."""
    s = s.strip()
    # Check for difference constraint
    if '-' in s.split('<=')[0].split('>=')[0].split('<')[0].split('>')[0].split('==')[0]:
        for op_str, op in [('<=', CompOp.LE), ('>=', CompOp.GE), ('<', CompOp.LT),
                           ('>', CompOp.GT), ('==', CompOp.EQ)]:
            if op_str in s:
                lhs, rhs = s.split(op_str)
                c1, c2 = lhs.strip().split('-')
                return ClockConstraint(c1.strip(), c2.strip(), op, int(rhs.strip()))

    for op_str, op in [('<=', CompOp.LE), ('>=', CompOp.GE), ('<', CompOp.LT),
                       ('>', CompOp.GT), ('==', CompOp.EQ)]:
        if op_str in s:
            lhs, rhs = s.split(op_str)
            return ClockConstraint(lhs.strip(), None, op, int(rhs.strip()))

    raise ValueError(f"Cannot parse constraint: {s}")
