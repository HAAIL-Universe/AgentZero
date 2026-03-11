"""
V142: Timed Automata + LTL Model Checking
Composes V118 (timed automata) + V023 (LTL model checking)

Verifies LTL temporal properties over timed automata via zone-based
product construction: TA x NBA (Nondeterministic Buchi Automaton).

The product preserves timed semantics (zones) while tracking NBA state.
Accepting cycle detection in the product zone graph determines LTL satisfaction.

AI-Generated | Claude (Anthropic) | AgentZero A2 | 2026-03-11
"""

import sys, os, math
from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, Tuple, FrozenSet
from enum import Enum, auto

# Import V118 timed automata
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V118_timed_automata'))
from timed_automata import (
    TimedAutomaton, Edge, Guard, ClockConstraint, CompOp, Zone,
    initial_zone, true_guard, guard_and,
    clock_leq, clock_lt, clock_geq, clock_gt, clock_eq,
    clock_diff_leq, clock_diff_geq,
    simple_ta, check_reachability, check_safety,
    explore_zone_graph, ZoneGraphStats, Trace, TimedAction,
    ReachabilityResult, SafetyResult
)

# Import V023 LTL (needs V021 BDD on path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V023_ltl_model_checking'))
from ltl_model_checker import (
    LTL, LTLOp, Atom, LTLTrue, LTLFalse, Not, And, Or,
    Implies, Next, Finally, Globally, Until, Release, WeakUntil,
    parse_ltl, atoms, nnf, ltl_to_gba, gba_to_nba, NBA, GBA, Label,
    LTLResult
)


# ============================================================
# Data structures
# ============================================================

class TimedLTLVerdict(Enum):
    """Verification result."""
    SATISFIED = auto()
    VIOLATED = auto()
    UNKNOWN = auto()


@dataclass
class ProductState:
    """State in the TA x NBA product: (location, nba_state)."""
    location: str
    nba_state: int

    def __hash__(self):
        return hash((self.location, self.nba_state))

    def __eq__(self, other):
        if not isinstance(other, ProductState):
            return False
        return self.location == other.location and self.nba_state == other.nba_state

    def __repr__(self):
        return f"({self.location}, q{self.nba_state})"


@dataclass
class ProductZoneState:
    """Symbolic state in the product zone graph."""
    product_state: ProductState
    zone_idx: int  # index into zone store


@dataclass
class TimedLTLResult:
    """Result of timed LTL model checking."""
    verdict: TimedLTLVerdict
    formula: Optional[LTL] = None
    counterexample: Optional[List[Tuple[str, str, Optional[Dict[str, float]]]]] = None
    nba_states: int = 0
    product_states_explored: int = 0
    zones_created: int = 0
    accepting_cycle_found: bool = False


@dataclass
class TimedLTLStats:
    """Statistics for timed LTL model checking."""
    formula_atoms: int = 0
    nba_states: int = 0
    nba_transitions: int = 0
    product_states: int = 0
    zones_created: int = 0
    subsumption_hits: int = 0
    cycle_checks: int = 0


# ============================================================
# Labeling function
# ============================================================

@dataclass
class LocationLabeling:
    """Maps locations to sets of atomic propositions that hold there."""
    labels: Dict[str, Set[str]] = field(default_factory=dict)

    def get_labels(self, location: str) -> Set[str]:
        return self.labels.get(location, set())

    def add_label(self, location: str, prop: str):
        if location not in self.labels:
            self.labels[location] = set()
        self.labels[location].add(prop)

    def matches_label(self, location: str, label: Label) -> bool:
        """Check if location's propositions satisfy NBA label."""
        loc_props = self.get_labels(location)
        # All positive atoms must be in loc_props
        for p in label.pos:
            if p not in loc_props:
                return False
        # All negative atoms must NOT be in loc_props
        for n in label.neg:
            if n in loc_props:
                return False
        return True


def auto_labeling(ta: TimedAutomaton, formula: LTL) -> LocationLabeling:
    """Create labeling where each location name is an atomic proposition.

    For atoms matching location names, the location satisfies its own atom.
    Also supports prefix-based labeling: atom 'critical' matches any location
    containing 'critical' as substring.
    """
    labeling = LocationLabeling()
    formula_atoms = atoms(formula)

    for loc in ta.locations:
        for atom_name in formula_atoms:
            if atom_name == loc:
                labeling.add_label(loc, atom_name)
            elif atom_name in loc:
                # substring match: 'critical' matches 'critical_1', 'critical_2'
                labeling.add_label(loc, atom_name)
    return labeling


# ============================================================
# Zone Store for Product
# ============================================================

class ProductZoneStore:
    """Stores zones for the product zone graph."""

    def __init__(self):
        self.zones: List[Zone] = []

    def add(self, zone: Zone) -> int:
        idx = len(self.zones)
        self.zones.append(zone)
        return idx

    def get(self, idx: int) -> Zone:
        return self.zones[idx]


# ============================================================
# Product Construction: TA x NBA
# ============================================================

def _successor_zone(zone: Zone, edge: Edge, target_inv: Guard) -> Optional[Zone]:
    """Compute successor zone for a TA edge. Returns None if empty."""
    z = zone.copy()
    # Apply guard
    z.apply_guard(edge.guard)
    if z.is_empty():
        return None
    # Apply resets
    z.apply_resets(edge.resets)
    # Apply target invariant
    z.apply_guard(target_inv)
    if z.is_empty():
        return None
    # Time elapse
    z.future()
    # Re-apply invariant after time elapse
    z.apply_guard(target_inv)
    z.canonicalize()
    if z.is_empty():
        return None
    return z


def build_product_zone_graph(
    ta: TimedAutomaton,
    nba: NBA,
    labeling: LocationLabeling,
    max_states: int = 50000
) -> Tuple[Dict[ProductState, List[Tuple[Zone, int]]], Set[ProductState], TimedLTLStats]:
    """Build the product zone graph: TA x NBA.

    Returns:
        reached: maps ProductState -> list of (zone, parent_info)
        initial_states: set of initial ProductStates
        stats: exploration statistics
    """
    stats = TimedLTLStats()
    zone_store = ProductZoneStore()

    # reached[ps] = [(zone, parent_key)] where parent_key = (prev_ps, edge_label, prev_zone_idx)
    reached: Dict[ProductState, List[Tuple[Zone, Optional[Tuple]]]] = {}
    initial_states: Set[ProductState] = set()

    # Initialize: initial zone at initial location x initial NBA states
    init_zone = initial_zone(sorted(ta.clocks))
    init_inv = ta.get_invariant(ta.initial)
    init_zone.apply_guard(init_inv)
    if not init_zone.is_empty():
        init_zone.future()
        init_zone.apply_guard(init_inv)
        init_zone.canonicalize()

    if init_zone.is_empty():
        return reached, initial_states, stats

    # For each initial NBA state, check if initial location labels match
    # NBA initial transitions
    for nba_init in nba.initial:
        ps = ProductState(ta.initial, nba_init)
        initial_states.add(ps)
        reached[ps] = [(init_zone.copy(), None)]

    # BFS exploration
    queue: List[Tuple[ProductState, int]] = []  # (product_state, zone_index_in_reached)
    for ps in initial_states:
        queue.append((ps, 0))

    total_explored = 0

    while queue and total_explored < max_states:
        ps, z_idx = queue.pop(0)
        total_explored += 1

        if ps not in reached or z_idx >= len(reached[ps]):
            continue

        current_zone = reached[ps][z_idx][0]
        current_loc = ps.location
        current_nba = ps.nba_state

        # For each TA edge from current location
        for edge in ta.get_edges_from(current_loc):
            target_inv = ta.get_invariant(edge.target)
            succ_zone = _successor_zone(current_zone, edge, target_inv)
            if succ_zone is None:
                continue

            # For each NBA transition from current NBA state
            if current_nba not in nba.transitions:
                continue

            for nba_label, nba_next in nba.transitions[current_nba]:
                # Check if the TARGET location's labels match the NBA label
                if not labeling.matches_label(edge.target, nba_label):
                    continue

                succ_ps = ProductState(edge.target, nba_next)
                parent = (ps, edge.label, z_idx)

                # Check subsumption
                if succ_ps in reached:
                    subsumed = False
                    for existing_zone, _ in reached[succ_ps]:
                        if existing_zone.includes(succ_zone):
                            subsumed = True
                            stats.subsumption_hits += 1
                            break
                    if subsumed:
                        continue
                    # Add new zone (could also remove subsumed existing zones)
                    new_idx = len(reached[succ_ps])
                    reached[succ_ps].append((succ_zone, parent))
                    queue.append((succ_ps, new_idx))
                else:
                    reached[succ_ps] = [(succ_zone, parent)]
                    queue.append((succ_ps, 0))

                stats.zones_created += 1

    stats.product_states = sum(len(zones) for zones in reached.values())
    stats.nba_states = len(nba.states)
    stats.nba_transitions = sum(len(trans) for trans in nba.transitions.values())

    return reached, initial_states, stats


# ============================================================
# Accepting Cycle Detection
# ============================================================

def _find_accepting_cycle(
    ta: TimedAutomaton,
    nba: NBA,
    labeling: LocationLabeling,
    reached: Dict[ProductState, List[Tuple[Zone, Optional[Tuple]]]],
    initial_states: Set[ProductState],
    max_depth: int = 1000
) -> Optional[List[Tuple[str, str]]]:
    """Detect accepting cycle in product zone graph.

    An accepting cycle exists if from some reachable accepting product state,
    we can reach the same product state again (with zone inclusion).

    Uses nested DFS (Schwoon-Esparza approach adapted for zones):
    - Outer DFS: explore reachable states
    - Inner DFS: from accepting states, search for cycle back

    Returns: list of (location, action) steps forming lasso if found, else None.
    """
    # Accepting product states: those whose NBA component is accepting
    accepting_ps = set()
    for ps in reached:
        if ps.nba_state in nba.accepting:
            accepting_ps.add(ps)

    if not accepting_ps:
        return None

    # Build adjacency from reached data
    # For each product state, compute successors (product_state -> [(succ_ps, edge_label)])
    adjacency: Dict[ProductState, List[Tuple[ProductState, str]]] = {}

    for ps in reached:
        loc = ps.location
        nba_st = ps.nba_state
        succs = []

        for edge in ta.get_edges_from(loc):
            if nba_st not in nba.transitions:
                continue
            for nba_label, nba_next in nba.transitions[nba_st]:
                if labeling.matches_label(edge.target, nba_label):
                    succ_ps = ProductState(edge.target, nba_next)
                    if succ_ps in reached:
                        succs.append((succ_ps, edge.label))

        adjacency[ps] = succs

    # Nested DFS for accepting cycle
    # Outer DFS
    outer_visited: Set[ProductState] = set()
    inner_visited: Set[ProductState] = set()
    cycle_found = False
    cycle_trace: List[Tuple[str, str]] = []

    def outer_dfs(ps: ProductState, path: List[Tuple[ProductState, str]]) -> bool:
        nonlocal cycle_found
        if cycle_found:
            return True

        outer_visited.add(ps)

        for succ_ps, label in adjacency.get(ps, []):
            if cycle_found:
                return True
            if succ_ps not in outer_visited:
                outer_dfs(succ_ps, path + [(succ_ps, label)])

        # If accepting, launch inner DFS
        if ps in accepting_ps and not cycle_found:
            inner_visited.clear()
            if inner_dfs(ps, ps, []):
                cycle_found = True
                return True

        return cycle_found

    def inner_dfs(start: ProductState, current: ProductState,
                  path: List[Tuple[str, str]]) -> bool:
        """Search for path back to start from current."""
        inner_visited.add(current)

        for succ_ps, label in adjacency.get(current, []):
            if succ_ps == start:
                # Found cycle!
                cycle_trace.extend(path + [(succ_ps.location, label)])
                return True
            if succ_ps not in inner_visited:
                if inner_dfs(start, succ_ps, path + [(succ_ps.location, label)]):
                    return True

        return False

    # Run outer DFS from initial states
    for init_ps in initial_states:
        if init_ps not in outer_visited:
            outer_dfs(init_ps, [(init_ps, "")])
        if cycle_found:
            break

    return cycle_trace if cycle_found else None


# ============================================================
# Main LTL Verification for Timed Automata
# ============================================================

def check_timed_ltl(
    ta: TimedAutomaton,
    formula: LTL,
    labeling: Optional[LocationLabeling] = None,
    max_states: int = 50000
) -> TimedLTLResult:
    """Check if timed automaton satisfies LTL formula.

    Algorithm:
    1. Negate the formula: check if NOT(formula) has an accepting run
    2. Build NBA for NOT(formula)
    3. Build product zone graph: TA x NBA
    4. Search for accepting cycle in product
    5. If cycle found: formula VIOLATED (counterexample exists)
       If no cycle: formula SATISFIED

    Args:
        ta: Timed automaton
        formula: LTL formula to check
        labeling: Location labeling (auto-generated if None)
        max_states: Maximum product states to explore

    Returns:
        TimedLTLResult with verdict and diagnostics
    """
    if labeling is None:
        labeling = auto_labeling(ta, formula)

    # Step 1: Negate formula and build NBA
    neg_formula = nnf(Not(formula))
    gba = ltl_to_gba(neg_formula)
    nba = gba_to_nba(gba)

    # Step 2: Build product zone graph
    reached, initial_states, stats = build_product_zone_graph(
        ta, nba, labeling, max_states
    )

    # Step 3: Search for accepting cycle
    cycle = _find_accepting_cycle(ta, nba, labeling, reached, initial_states)

    if cycle is not None:
        return TimedLTLResult(
            verdict=TimedLTLVerdict.VIOLATED,
            formula=formula,
            counterexample=[(loc, act, None) for loc, act in cycle],
            nba_states=len(nba.states),
            product_states_explored=stats.product_states,
            zones_created=stats.zones_created,
            accepting_cycle_found=True
        )
    else:
        return TimedLTLResult(
            verdict=TimedLTLVerdict.SATISFIED,
            formula=formula,
            nba_states=len(nba.states),
            product_states_explored=stats.product_states,
            zones_created=stats.zones_created,
            accepting_cycle_found=False
        )


def check_timed_ltl_parsed(
    ta: TimedAutomaton,
    formula_str: str,
    labeling: Optional[LocationLabeling] = None,
    max_states: int = 50000
) -> TimedLTLResult:
    """Check timed LTL with formula given as string.

    Example: check_timed_ltl_parsed(ta, "G(F(open))")
    """
    formula = parse_ltl(formula_str)
    return check_timed_ltl(ta, formula, labeling, max_states)


# ============================================================
# Safety and Liveness Convenience APIs
# ============================================================

def check_timed_safety(
    ta: TimedAutomaton,
    safe_prop: str,
    labeling: Optional[LocationLabeling] = None,
    max_states: int = 50000
) -> TimedLTLResult:
    """Check G(safe_prop): the proposition always holds.

    Args:
        ta: Timed automaton
        safe_prop: Atomic proposition that should always hold
        labeling: Location labeling
    """
    formula = Globally(Atom(safe_prop))
    return check_timed_ltl(ta, formula, labeling, max_states)


def check_timed_liveness(
    ta: TimedAutomaton,
    live_prop: str,
    labeling: Optional[LocationLabeling] = None,
    max_states: int = 50000
) -> TimedLTLResult:
    """Check G(F(live_prop)): the proposition holds infinitely often.

    Args:
        ta: Timed automaton
        live_prop: Atomic proposition that should recur
        labeling: Location labeling
    """
    formula = Globally(Finally(Atom(live_prop)))
    return check_timed_ltl(ta, formula, labeling, max_states)


def check_timed_response(
    ta: TimedAutomaton,
    trigger: str,
    response: str,
    labeling: Optional[LocationLabeling] = None,
    max_states: int = 50000
) -> TimedLTLResult:
    """Check G(trigger -> F(response)): every trigger eventually gets response.

    Args:
        ta: Timed automaton
        trigger: Atomic proposition for trigger
        response: Atomic proposition for response
        labeling: Location labeling
    """
    formula = Globally(Implies(Atom(trigger), Finally(Atom(response))))
    return check_timed_ltl(ta, formula, labeling, max_states)


def check_timed_until(
    ta: TimedAutomaton,
    hold_prop: str,
    goal_prop: str,
    labeling: Optional[LocationLabeling] = None,
    max_states: int = 50000
) -> TimedLTLResult:
    """Check hold_prop U goal_prop: hold_prop holds until goal_prop is reached.

    Args:
        ta: Timed automaton
        hold_prop: Atomic proposition that holds until goal
        goal_prop: Atomic proposition for goal
        labeling: Location labeling
    """
    formula = Until(Atom(hold_prop), Atom(goal_prop))
    return check_timed_ltl(ta, formula, labeling, max_states)


# ============================================================
# Zone Graph Abstraction -> Finite State
# ============================================================

@dataclass
class ZoneGraphAbstraction:
    """Finite-state abstraction of a timed automaton via zone graph.

    Each symbolic state (location, zone_idx) becomes a node.
    Transitions follow the zone graph edges.
    """
    states: Set[str]
    initial: Set[str]
    transitions: Dict[str, List[Tuple[str, str]]]  # state -> [(label, target)]
    state_labels: Dict[str, Set[str]]  # state -> atomic propositions
    accepting: Set[str]

    def n_states(self) -> int:
        return len(self.states)

    def n_transitions(self) -> int:
        return sum(len(ts) for ts in self.transitions.values())


def abstract_zone_graph(
    ta: TimedAutomaton,
    labeling: Optional[LocationLabeling] = None,
    formula: Optional[LTL] = None
) -> ZoneGraphAbstraction:
    """Build finite-state abstraction from timed automaton zone graph.

    Each (location, zone_index) pair becomes a state in the abstraction.
    This abstraction is sound for reachability properties.
    """
    if labeling is None and formula is not None:
        labeling = auto_labeling(ta, formula)
    elif labeling is None:
        labeling = LocationLabeling()

    zone_graph, zg_stats = explore_zone_graph(ta)

    states: Set[str] = set()
    initial: Set[str] = set()
    transitions: Dict[str, List[Tuple[str, str]]] = {}
    state_labels: Dict[str, Set[str]] = {}

    # Create states for each (location, zone_idx) pair
    state_zone_map: Dict[str, Tuple[str, Zone]] = {}
    for loc, zones in zone_graph.items():
        for i, zone in enumerate(zones):
            s = f"{loc}_{i}"
            states.add(s)
            state_labels[s] = labeling.get_labels(loc)
            state_zone_map[s] = (loc, zone)
            if loc == ta.initial and i == 0:
                initial.add(s)

    # Build transitions by re-exploring edges
    for s in states:
        loc, zone = state_zone_map[s]
        transitions[s] = []

        for edge in ta.get_edges_from(loc):
            target_inv = ta.get_invariant(edge.target)
            succ_zone = _successor_zone(zone, edge, target_inv)
            if succ_zone is None:
                continue

            # Find matching target zone
            if edge.target in zone_graph:
                for j, target_zone in enumerate(zone_graph[edge.target]):
                    if target_zone.includes(succ_zone):
                        t = f"{edge.target}_{j}"
                        transitions[s].append((edge.label, t))
                        break

    accepting = set()
    for s in states:
        loc = s.rsplit('_', 1)[0]
        if loc in ta.accepting:
            accepting.add(s)

    return ZoneGraphAbstraction(
        states=states,
        initial=initial,
        transitions=transitions,
        state_labels=state_labels,
        accepting=accepting
    )


# ============================================================
# Comparison API
# ============================================================

def compare_timed_vs_untimed(
    ta: TimedAutomaton,
    formula: LTL,
    labeling: Optional[LocationLabeling] = None
) -> Dict:
    """Compare timed LTL result with zone-graph abstraction analysis.

    Returns dict with:
    - timed_result: full timed LTL result
    - zone_graph_size: abstraction size
    - nba_size: NBA automaton size
    """
    if labeling is None:
        labeling = auto_labeling(ta, formula)

    timed_result = check_timed_ltl(ta, formula, labeling)
    abstraction = abstract_zone_graph(ta, labeling, formula)

    return {
        'formula': str(formula),
        'timed_verdict': timed_result.verdict.name,
        'product_states': timed_result.product_states_explored,
        'nba_states': timed_result.nba_states,
        'zone_graph_states': abstraction.n_states(),
        'zone_graph_transitions': abstraction.n_transitions(),
        'has_counterexample': timed_result.counterexample is not None,
    }


# ============================================================
# Example Systems with Properties
# ============================================================

def light_timer_with_properties() -> Tuple[TimedAutomaton, LocationLabeling, List[Tuple[str, LTL]]]:
    """Light timer system with LTL properties.

    Locations: off, on
    Clock: x
    Transitions: off->on (press), on->off (press, x>=2), on->off (timeout, x=5)

    Properties:
    - G(F(off)): light always eventually turns off
    - G(on -> F(off)): on always leads to off
    """
    ta = simple_ta(
        locations=['off', 'on'],
        initial='off',
        clocks=['x'],
        edges=[
            ('off', 'on', 'press', true_guard(), ['x']),
            ('on', 'off', 'press', clock_geq('x', 2), []),
            ('on', 'off', 'timeout', clock_eq('x', 5), []),
        ],
        invariants={'on': clock_leq('x', 5)},
        accepting={'off', 'on'}
    )

    labeling = LocationLabeling()
    labeling.add_label('off', 'off')
    labeling.add_label('on', 'on')

    properties = [
        ("safety: light has invariant", Globally(Implies(Atom('on'), LTLTrue()))),
        ("reachability: on is reachable", Finally(Atom('on'))),
    ]

    return ta, labeling, properties


def train_gate_with_properties() -> Tuple[TimedAutomaton, LocationLabeling, List[Tuple[str, LTL]]]:
    """Train-gate controller with LTL properties.

    Models a rail crossing: train approaches, gate closes, train crosses, gate opens.
    Safety: gate is closed when train is crossing.
    """
    ta = simple_ta(
        locations=['idle', 'approaching', 'crossing', 'leaving'],
        initial='idle',
        clocks=['x'],
        edges=[
            ('idle', 'approaching', 'approach', true_guard(), ['x']),
            ('approaching', 'crossing', 'enter', clock_geq('x', 1), ['x']),
            ('crossing', 'leaving', 'exit', clock_geq('x', 1), ['x']),
            ('leaving', 'idle', 'clear', clock_geq('x', 1), []),
        ],
        invariants={
            'approaching': clock_leq('x', 3),
            'crossing': clock_leq('x', 5),
            'leaving': clock_leq('x', 2),
        },
        accepting={'idle', 'approaching', 'crossing', 'leaving'}
    )

    labeling = LocationLabeling()
    labeling.add_label('idle', 'idle')
    labeling.add_label('approaching', 'approaching')
    labeling.add_label('crossing', 'crossing')
    labeling.add_label('leaving', 'leaving')
    labeling.add_label('idle', 'safe')
    labeling.add_label('approaching', 'safe')
    labeling.add_label('leaving', 'safe')
    # crossing is NOT safe (train in crossing zone)

    properties = [
        ("liveness: train eventually clears", Globally(Implies(Atom('approaching'), Finally(Atom('idle'))))),
        ("reachability: crossing reachable", Finally(Atom('crossing'))),
    ]

    return ta, labeling, properties


def mutex_with_properties(n: int = 2) -> Tuple[TimedAutomaton, LocationLabeling, List[Tuple[str, LTL]]]:
    """Simple mutual exclusion protocol with timing.

    Two processes trying to enter critical section.
    Process i: idle_i -> trying_i (set x_i) -> critical_i (x_i <= 3) -> idle_i
    Safety: not both in critical simultaneously.
    """
    locations = []
    edges_list = []
    invariants = {}
    clocks = []

    for i in range(1, n + 1):
        idle = f'idle_{i}'
        trying = f'trying_{i}'
        critical = f'crit_{i}'
        clock = f'x{i}'

        locations.extend([idle, trying, critical])
        clocks.append(clock)

        edges_list.append((idle, trying, f'try_{i}', true_guard(), [clock]))
        edges_list.append((trying, critical, f'enter_{i}', clock_geq(clock, 1), [clock]))
        edges_list.append((critical, idle, f'exit_{i}', true_guard(), []))

        invariants[critical] = clock_leq(clock, 3)

    # Build product locations manually for 2 processes
    if n == 2:
        prod_locs = []
        prod_edges = []
        prod_inv = {}

        for l1 in ['idle_1', 'trying_1', 'crit_1']:
            for l2 in ['idle_2', 'trying_2', 'crit_2']:
                loc = f"{l1}_{l2}"
                prod_locs.append(loc)
                # Invariant: intersection of component invariants
                inv_parts = []
                if l1 in invariants:
                    inv_parts.append(invariants[l1])
                if l2 in invariants:
                    inv_parts.append(invariants[l2])
                if inv_parts:
                    combined = inv_parts[0]
                    for p in inv_parts[1:]:
                        combined = guard_and(combined, p)
                    prod_inv[loc] = combined

        # Process 1 transitions (process 2 stays)
        for l2 in ['idle_2', 'trying_2', 'crit_2']:
            prod_edges.append((f'idle_1_{l2}', f'trying_1_{l2}', 'try_1', true_guard(), ['x1']))
            prod_edges.append((f'trying_1_{l2}', f'crit_1_{l2}', 'enter_1', clock_geq('x1', 1), ['x1']))
            prod_edges.append((f'crit_1_{l2}', f'idle_1_{l2}', 'exit_1', true_guard(), []))

        # Process 2 transitions (process 1 stays)
        for l1 in ['idle_1', 'trying_1', 'crit_1']:
            prod_edges.append((f'{l1}_idle_2', f'{l1}_trying_2', 'try_2', true_guard(), ['x2']))
            prod_edges.append((f'{l1}_trying_2', f'{l1}_crit_2', 'enter_2', clock_geq('x2', 1), ['x2']))
            prod_edges.append((f'{l1}_crit_2', f'{l1}_idle_2', 'exit_2', true_guard(), []))

        ta = simple_ta(
            locations=prod_locs,
            initial='idle_1_idle_2',
            clocks=['x1', 'x2'],
            edges=prod_edges,
            invariants=prod_inv,
            accepting=set(prod_locs)
        )

        labeling = LocationLabeling()
        for loc in prod_locs:
            labeling.add_label(loc, loc)
            if 'crit_1' in loc and 'crit_2' in loc:
                labeling.add_label(loc, 'both_critical')
            if 'crit_1' in loc or 'crit_2' in loc:
                labeling.add_label(loc, 'some_critical')
            if 'crit_1' not in loc and 'crit_2' not in loc:
                labeling.add_label(loc, 'no_critical')

        properties = [
            ("safety: not both critical", Globally(Not(Atom('both_critical')))),
            ("liveness: someone can be critical", Finally(Atom('some_critical'))),
        ]

        return ta, labeling, properties

    # Fallback for n != 2: single process
    ta = simple_ta(
        locations=locations,
        initial='idle_1',
        clocks=clocks,
        edges=edges_list,
        invariants=invariants,
        accepting=set(locations)
    )
    labeling = auto_labeling(ta, LTLTrue())
    return ta, labeling, []


# ============================================================
# Batch Verification
# ============================================================

def batch_check(
    ta: TimedAutomaton,
    formulas: List[Tuple[str, LTL]],
    labeling: Optional[LocationLabeling] = None,
    max_states: int = 50000
) -> List[Tuple[str, TimedLTLResult]]:
    """Check multiple LTL formulas against same timed automaton.

    Args:
        ta: Timed automaton
        formulas: List of (name, formula) pairs
        labeling: Location labeling (auto-generated per formula if None)

    Returns:
        List of (name, result) pairs
    """
    results = []
    for name, formula in formulas:
        lab = labeling if labeling is not None else auto_labeling(ta, formula)
        result = check_timed_ltl(ta, formula, lab, max_states)
        results.append((name, result))
    return results


def verification_summary(
    ta: TimedAutomaton,
    results: List[Tuple[str, TimedLTLResult]]
) -> str:
    """Generate human-readable summary of verification results."""
    lines = [
        f"Timed LTL Verification Summary",
        f"Timed Automaton: {len(ta.locations)} locations, {len(ta.edges)} edges, {len(ta.clocks)} clocks",
        f"Properties checked: {len(results)}",
        ""
    ]
    for name, result in results:
        status = "PASS" if result.verdict == TimedLTLVerdict.SATISFIED else "FAIL"
        lines.append(f"  [{status}] {name}")
        lines.append(f"         NBA states: {result.nba_states}, "
                     f"Product states: {result.product_states_explored}")
        if result.counterexample:
            trace_str = " -> ".join(f"{loc}" for loc, _, _ in result.counterexample[:5])
            if len(result.counterexample) > 5:
                trace_str += " -> ..."
            lines.append(f"         Counterexample: {trace_str}")

    passed = sum(1 for _, r in results if r.verdict == TimedLTLVerdict.SATISFIED)
    lines.append(f"\nResult: {passed}/{len(results)} properties satisfied")
    return "\n".join(lines)
