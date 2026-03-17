"""
V183: TCTL Model Checking

Timed Computation Tree Logic model checking over timed automata.
Composes V118 (timed automata, zone-based exploration) with temporal logic.

TCTL extends CTL with quantitative timing bounds:
  - E F_{<=k} phi     -- exists path reaching phi within k time units
  - A F_{<=k} phi     -- all paths reach phi within k time units
  - E G_{<=k} phi     -- exists path where phi holds for k time units
  - A G_{<=k} phi     -- all paths satisfy phi for k time units
  - E [phi U_{<=k} psi] -- exists path where phi until psi within k
  - A [phi U_{<=k} psi] -- all paths: phi until psi within k
  - Boolean combinations: and, or, not

Algorithm: zone-based symbolic model checking with on-the-fly exploration.
For bounded properties, we track an extra "formula clock" that measures elapsed time.

Key insight: TCTL_{<=k} formulas are checked by adding a fresh clock z,
resetting z at the start, and checking the untimed CTL formula with
an extra constraint z <= k in the zone.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet, Union
from enum import Enum
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V118_timed_automata'))
from timed_automata import (
    TimedAutomaton, Edge, Guard, ClockConstraint, CompOp, Zone,
    initial_zone, SymbolicState, ZoneStore, Trace,
    true_guard, clock_leq, clock_lt, clock_geq, clock_gt,
    clock_eq, guard_and, _successor_zone, simple_ta
)

INF = float('inf')


# ============================================================
# TCTL Formula AST
# ============================================================

class TCTLFormula:
    """Base class for TCTL formulas."""
    pass


@dataclass(frozen=True)
class Atomic(TCTLFormula):
    """Atomic proposition: location label."""
    label: str

    def __str__(self):
        return self.label


@dataclass(frozen=True)
class TrueF(TCTLFormula):
    """Tautology."""
    def __str__(self):
        return "true"


@dataclass(frozen=True)
class FalseF(TCTLFormula):
    """Contradiction."""
    def __str__(self):
        return "false"


@dataclass(frozen=True)
class Not(TCTLFormula):
    """Negation."""
    sub: TCTLFormula

    def __str__(self):
        return f"!({self.sub})"


@dataclass(frozen=True)
class And(TCTLFormula):
    """Conjunction."""
    left: TCTLFormula
    right: TCTLFormula

    def __str__(self):
        return f"({self.left} && {self.right})"


@dataclass(frozen=True)
class Or(TCTLFormula):
    """Disjunction."""
    left: TCTLFormula
    right: TCTLFormula

    def __str__(self):
        return f"({self.left} || {self.right})"


@dataclass(frozen=True)
class Implies(TCTLFormula):
    """Implication."""
    left: TCTLFormula
    right: TCTLFormula

    def __str__(self):
        return f"({self.left} => {self.right})"


class BoundType(Enum):
    """Type of timing bound."""
    UNBOUNDED = 'unbounded'
    LEQ = '<='
    LT = '<'
    GEQ = '>='
    GT = '>'
    EQ = '=='


@dataclass(frozen=True)
class TimeBound:
    """Timing bound for TCTL operators."""
    bound_type: BoundType
    value: int = 0

    def __str__(self):
        if self.bound_type == BoundType.UNBOUNDED:
            return ""
        return f"_{{{self.bound_type.value}{self.value}}}"

    @staticmethod
    def unbounded():
        return TimeBound(BoundType.UNBOUNDED)

    @staticmethod
    def leq(k):
        return TimeBound(BoundType.LEQ, k)

    @staticmethod
    def lt(k):
        return TimeBound(BoundType.LT, k)

    @staticmethod
    def geq(k):
        return TimeBound(BoundType.GEQ, k)

    @staticmethod
    def gt(k):
        return TimeBound(BoundType.GT, k)

    @staticmethod
    def eq(k):
        return TimeBound(BoundType.EQ, k)


@dataclass(frozen=True)
class EF(TCTLFormula):
    """Exists Finally: E F_{bound} phi."""
    sub: TCTLFormula
    bound: TimeBound = field(default_factory=TimeBound.unbounded)

    def __str__(self):
        return f"E F{self.bound} {self.sub}"


@dataclass(frozen=True)
class AF(TCTLFormula):
    """All Finally: A F_{bound} phi."""
    sub: TCTLFormula
    bound: TimeBound = field(default_factory=TimeBound.unbounded)

    def __str__(self):
        return f"A F{self.bound} {self.sub}"


@dataclass(frozen=True)
class EG(TCTLFormula):
    """Exists Globally: E G_{bound} phi."""
    sub: TCTLFormula
    bound: TimeBound = field(default_factory=TimeBound.unbounded)

    def __str__(self):
        return f"E G{self.bound} {self.sub}"


@dataclass(frozen=True)
class AG(TCTLFormula):
    """All Globally: A G_{bound} phi."""
    sub: TCTLFormula
    bound: TimeBound = field(default_factory=TimeBound.unbounded)

    def __str__(self):
        return f"A G{self.bound} {self.sub}"


@dataclass(frozen=True)
class EU(TCTLFormula):
    """Exists Until: E [phi U_{bound} psi]."""
    left: TCTLFormula
    right: TCTLFormula
    bound: TimeBound = field(default_factory=TimeBound.unbounded)

    def __str__(self):
        return f"E [{self.left} U{self.bound} {self.right}]"


@dataclass(frozen=True)
class AU(TCTLFormula):
    """All Until: A [phi U_{bound} psi]."""
    left: TCTLFormula
    right: TCTLFormula
    bound: TimeBound = field(default_factory=TimeBound.unbounded)

    def __str__(self):
        return f"A [{self.left} U{self.bound} {self.right}]"


# ============================================================
# TCTL Model Checking Result
# ============================================================

@dataclass
class TCTLResult:
    """Result of TCTL model checking."""
    satisfied: bool
    formula: TCTLFormula
    satisfying_locations: Set[str] = field(default_factory=set)
    witness_trace: Optional[Trace] = None
    counterexample_trace: Optional[Trace] = None
    states_explored: int = 0
    zones_created: int = 0

    def __str__(self):
        status = "SATISFIED" if self.satisfied else "VIOLATED"
        return f"TCTL: {self.formula} -- {status} (explored {self.states_explored} states)"


# ============================================================
# Labeling Function
# ============================================================

def _get_labels(ta: TimedAutomaton, location: str) -> Set[str]:
    """Get labels for a location. Uses location name as default label."""
    labels = {location}
    if hasattr(ta, 'labels') and location in ta.labels:
        labels.update(ta.labels[location])
    return labels


def _location_satisfies_atomic(ta: TimedAutomaton, location: str, label: str) -> bool:
    """Check if a location satisfies an atomic proposition."""
    return label in _get_labels(ta, location)


# ============================================================
# Zone Graph with Formula Clock
# ============================================================

@dataclass
class TimedState:
    """State in the timed model checking exploration.
    Pairs a location with a zone that includes a formula clock.
    """
    location: str
    zone: Zone
    parent: Optional['TimedState'] = None
    edge_label: str = ""

    def __hash__(self):
        return hash(self.location)

    def __eq__(self, other):
        return isinstance(other, TimedState) and self.location == other.location


def _add_formula_clock(ta: TimedAutomaton, clock_name: str = "__tctl_z") -> Tuple[TimedAutomaton, str]:
    """Create a copy of the TA with an additional formula clock.
    The formula clock is never reset by any edge, so it measures total elapsed time.
    """
    new_clocks = set(ta.clocks) | {clock_name}
    new_ta = TimedAutomaton(
        locations=ta.locations,
        initial=ta.initial,
        clocks=new_clocks,
        edges=ta.edges,
        invariants=dict(ta.invariants),
        accepting=ta.accepting,
        alphabet=ta.alphabet,
    )
    if hasattr(ta, 'labels'):
        new_ta.labels = ta.labels
    return new_ta, clock_name


def _make_bound_guard(clock_name: str, bound: TimeBound) -> Guard:
    """Create a guard from a time bound on the formula clock."""
    if bound.bound_type == BoundType.UNBOUNDED:
        return true_guard()
    elif bound.bound_type == BoundType.LEQ:
        return clock_leq(clock_name, bound.value)
    elif bound.bound_type == BoundType.LT:
        return clock_lt(clock_name, bound.value)
    elif bound.bound_type == BoundType.GEQ:
        return clock_geq(clock_name, bound.value)
    elif bound.bound_type == BoundType.GT:
        return clock_gt(clock_name, bound.value)
    elif bound.bound_type == BoundType.EQ:
        return clock_eq(clock_name, bound.value)
    return true_guard()


def _extract_trace(state: TimedState) -> Trace:
    """Extract a trace from a TimedState by following parent pointers."""
    steps = []
    current = state
    while current is not None:
        vals = current.zone.get_sample()
        steps.append((current.location, current.edge_label, vals if vals else {}))
        current = current.parent
    steps.reverse()
    return Trace(steps=steps)


# ============================================================
# Successor computation with formula clock
# ============================================================

def _compute_successors(ta: TimedAutomaton, state: TimedState) -> List[TimedState]:
    """Compute successor states for a TimedState."""
    successors = []
    for edge in ta.edges:
        if edge.source != state.location:
            continue
        target_inv = ta.invariants.get(edge.target, true_guard())
        succ_zone = _successor_zone(state.zone, edge, target_inv)
        if succ_zone is not None and not succ_zone.is_empty():
            successors.append(TimedState(
                location=edge.target,
                zone=succ_zone,
                parent=state,
                edge_label=edge.label,
            ))
    return successors


def _initial_state(ta: TimedAutomaton) -> TimedState:
    """Create initial timed state with all clocks at 0, then apply invariant + future."""
    z = initial_zone(list(ta.clocks))
    inv = ta.invariants.get(ta.initial, true_guard())
    z.apply_guard(inv)
    z.future()
    z.apply_guard(inv)
    z.canonicalize()
    return TimedState(location=ta.initial, zone=z)


# ============================================================
# Core TCTL Model Checking
# ============================================================

def _check_atomic_locations(ta: TimedAutomaton, formula: TCTLFormula) -> Set[str]:
    """Get set of locations satisfying a state formula (non-temporal)."""
    if isinstance(formula, TrueF):
        return set(ta.locations)
    elif isinstance(formula, FalseF):
        return set()
    elif isinstance(formula, Atomic):
        return {loc for loc in ta.locations if _location_satisfies_atomic(ta, loc, formula.label)}
    elif isinstance(formula, Not):
        sub = _check_atomic_locations(ta, formula.sub)
        return set(ta.locations) - sub
    elif isinstance(formula, And):
        return _check_atomic_locations(ta, formula.left) & _check_atomic_locations(ta, formula.right)
    elif isinstance(formula, Or):
        return _check_atomic_locations(ta, formula.left) | _check_atomic_locations(ta, formula.right)
    elif isinstance(formula, Implies):
        left = _check_atomic_locations(ta, formula.left)
        right = _check_atomic_locations(ta, formula.right)
        return (set(ta.locations) - left) | right
    return set()


def _is_state_formula(formula: TCTLFormula) -> bool:
    """Check if a formula is a pure state formula (no temporal operators)."""
    if isinstance(formula, (TrueF, FalseF, Atomic)):
        return True
    if isinstance(formula, Not):
        return _is_state_formula(formula.sub)
    if isinstance(formula, (And, Or, Implies)):
        return _is_state_formula(formula.left) and _is_state_formula(formula.right)
    return False


def _check_ef_bounded(ta: TimedAutomaton, phi: TCTLFormula, bound: TimeBound,
                      max_states: int = 10000) -> TCTLResult:
    """Check E F_{<=k} phi: exists a path reaching phi within the time bound.

    Algorithm: BFS zone exploration with formula clock.
    Add formula clock z, explore zone graph, check if any state satisfies
    phi AND z satisfies the bound.
    """
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None

    # Add formula clock
    aug_ta, fclock = _add_formula_clock(ta)
    bound_guard = _make_bound_guard(fclock, bound)

    init = _initial_state(aug_ta)
    queue = deque([init])
    visited = {}  # location -> list of zones (for subsumption)
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        # Check if this state satisfies phi (within time bound)
        loc_ok = phi_locs is None or state.location in phi_locs
        if loc_ok:
            # Check time bound: intersect zone with bound guard
            test_zone = state.zone.copy()
            test_zone.apply_guard(bound_guard)
            if not test_zone.is_empty():
                if phi_locs is not None:
                    return TCTLResult(
                        satisfied=True,
                        formula=EF(phi, bound),
                        satisfying_locations={state.location},
                        witness_trace=_extract_trace(state),
                        states_explored=states_explored,
                        zones_created=zones_created,
                    )

        # Check subsumption
        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        # Expand successors
        for succ in _compute_successors(aug_ta, state):
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=False,
        formula=EF(phi, bound),
        states_explored=states_explored,
        zones_created=zones_created,
    )


def _check_ag_bounded(ta: TimedAutomaton, phi: TCTLFormula, bound: TimeBound,
                      max_states: int = 10000) -> TCTLResult:
    """Check A G_{<=k} phi: on all paths, phi holds for at least k time units.

    Dual of E F_{<=k} !phi. If we can find a path reaching !phi within k,
    then A G_{<=k} phi is violated.
    """
    neg_result = _check_ef_bounded(ta, Not(phi) if not isinstance(phi, Not) else phi.sub,
                                   bound, max_states)
    satisfied = not neg_result.satisfied
    return TCTLResult(
        satisfied=satisfied,
        formula=AG(phi, bound),
        satisfying_locations=_check_atomic_locations(ta, phi) if satisfied else set(),
        counterexample_trace=neg_result.witness_trace if not satisfied else None,
        states_explored=neg_result.states_explored,
        zones_created=neg_result.zones_created,
    )


def _check_af_bounded(ta: TimedAutomaton, phi: TCTLFormula, bound: TimeBound,
                      max_states: int = 10000) -> TCTLResult:
    """Check A F_{<=k} phi: all paths reach phi within k time units.

    Algorithm: explore zone graph. phi must be reachable from every state
    within the time bound. If any state can avoid phi (by staying forever
    in a non-phi location or by exceeding the bound), AF is violated.
    """
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None

    aug_ta, fclock = _add_formula_clock(ta)
    bound_guard = _make_bound_guard(fclock, bound)

    init = _initial_state(aug_ta)
    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        # Check if phi holds here within bound
        loc_ok = phi_locs is None or state.location in phi_locs
        if loc_ok:
            test_zone = state.zone.copy()
            test_zone.apply_guard(bound_guard)
            if not test_zone.is_empty():
                continue  # This path satisfied phi

        # phi not satisfied. If this location can stay forever, AF fails.
        if _can_stay_forever(aug_ta, state.location):
            return TCTLResult(
                satisfied=False,
                formula=AF(phi, bound),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )

        # Check if past bound
        if bound.bound_type != BoundType.UNBOUNDED:
            beyond_guard = _make_beyond_guard(fclock, bound)
            beyond_zone = state.zone.copy()
            beyond_zone.apply_guard(beyond_guard)
            if beyond_zone.equals(state.zone) and not state.zone.is_empty():
                return TCTLResult(
                    satisfied=False,
                    formula=AF(phi, bound),
                    counterexample_trace=_extract_trace(state),
                    states_explored=states_explored,
                    zones_created=zones_created,
                )

        # Subsumption
        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        succs = _compute_successors(aug_ta, state)
        if not succs:
            return TCTLResult(
                satisfied=False,
                formula=AF(phi, bound),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )
        for succ in succs:
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=True,
        formula=AF(phi, bound),
        satisfying_locations=phi_locs if phi_locs else set(),
        states_explored=states_explored,
        zones_created=zones_created,
    )


def _make_beyond_guard(clock_name: str, bound: TimeBound) -> Guard:
    """Create guard for 'past the bound' (negation of bound constraint)."""
    if bound.bound_type == BoundType.LEQ:
        return clock_gt(clock_name, bound.value)
    elif bound.bound_type == BoundType.LT:
        return clock_geq(clock_name, bound.value)
    elif bound.bound_type == BoundType.GEQ:
        return clock_lt(clock_name, bound.value)
    elif bound.bound_type == BoundType.GT:
        return clock_leq(clock_name, bound.value)
    return true_guard()


def _check_eg_bounded(ta: TimedAutomaton, phi: TCTLFormula, bound: TimeBound,
                      max_states: int = 10000) -> TCTLResult:
    """Check E G_{<=k} phi: exists a path where phi holds for k time units.

    Dual of A F_{<=k} !phi.
    """
    neg_result = _check_af_bounded(ta, Not(phi) if not isinstance(phi, Not) else phi.sub,
                                   bound, max_states)
    satisfied = not neg_result.satisfied
    return TCTLResult(
        satisfied=satisfied,
        formula=EG(phi, bound),
        satisfying_locations=_check_atomic_locations(ta, phi) if satisfied else set(),
        witness_trace=neg_result.counterexample_trace if satisfied else None,
        states_explored=neg_result.states_explored,
        zones_created=neg_result.zones_created,
    )


def _check_eu_bounded(ta: TimedAutomaton, phi: TCTLFormula, psi: TCTLFormula,
                      bound: TimeBound, max_states: int = 10000) -> TCTLResult:
    """Check E [phi U_{<=k} psi]: exists path where phi holds until psi, within bound.

    Algorithm: BFS. Find a path where every state satisfies phi,
    ending in a state satisfying psi, all within the time bound.
    """
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None
    psi_locs = _check_atomic_locations(ta, psi) if _is_state_formula(psi) else None

    aug_ta, fclock = _add_formula_clock(ta)
    bound_guard = _make_bound_guard(fclock, bound)

    init = _initial_state(aug_ta)
    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        # Check if psi holds within bound
        psi_ok = psi_locs is None or state.location in psi_locs
        if psi_ok:
            test_zone = state.zone.copy()
            test_zone.apply_guard(bound_guard)
            if not test_zone.is_empty():
                return TCTLResult(
                    satisfied=True,
                    formula=EU(phi, psi, bound),
                    satisfying_locations={state.location},
                    witness_trace=_extract_trace(state),
                    states_explored=states_explored,
                    zones_created=zones_created,
                )

        # phi must hold on the path (if not psi)
        phi_ok = phi_locs is None or state.location in phi_locs
        if not phi_ok:
            continue  # phi doesn't hold, can't extend this path

        # Subsumption
        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        for succ in _compute_successors(aug_ta, state):
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=False,
        formula=EU(phi, psi, bound),
        states_explored=states_explored,
        zones_created=zones_created,
    )


def _check_au_bounded(ta: TimedAutomaton, phi: TCTLFormula, psi: TCTLFormula,
                      bound: TimeBound, max_states: int = 10000) -> TCTLResult:
    """Check A [phi U_{<=k} psi]: all paths satisfy phi-until-psi within bound.

    All paths must eventually reach psi (within bound), and phi must hold
    on every state before psi is reached.
    """
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None
    psi_locs = _check_atomic_locations(ta, psi) if _is_state_formula(psi) else None

    aug_ta, fclock = _add_formula_clock(ta)
    bound_guard = _make_bound_guard(fclock, bound)

    init = _initial_state(aug_ta)
    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        # Check if psi holds within bound (path completed successfully)
        psi_ok = psi_locs is None or state.location in psi_locs
        if psi_ok:
            test_zone = state.zone.copy()
            test_zone.apply_guard(bound_guard)
            if not test_zone.is_empty():
                continue  # This path is satisfied

        # phi must hold (if psi doesn't)
        phi_ok = phi_locs is None or state.location in phi_locs
        if not phi_ok:
            # phi doesn't hold and psi doesn't hold -- violation
            return TCTLResult(
                satisfied=False,
                formula=AU(phi, psi, bound),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )

        # Check if past bound
        if bound.bound_type != BoundType.UNBOUNDED:
            beyond_guard = _make_beyond_guard(fclock, bound)
            beyond_zone = state.zone.copy()
            beyond_zone.apply_guard(beyond_guard)
            if beyond_zone.equals(state.zone) and not state.zone.is_empty():
                return TCTLResult(
                    satisfied=False,
                    formula=AU(phi, psi, bound),
                    counterexample_trace=_extract_trace(state),
                    states_explored=states_explored,
                    zones_created=zones_created,
                )

        # Subsumption
        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        succs = _compute_successors(aug_ta, state)
        if not succs:
            # Deadlock without reaching psi
            return TCTLResult(
                satisfied=False,
                formula=AU(phi, psi, bound),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )
        for succ in succs:
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=True,
        formula=AU(phi, psi, bound),
        satisfying_locations=psi_locs if psi_locs else set(),
        states_explored=states_explored,
        zones_created=zones_created,
    )


# ============================================================
# Unbounded TCTL (standard CTL over timed automata)
# ============================================================

def _check_loc_satisfies_formula(ta: TimedAutomaton, location: str,
                                  formula: TCTLFormula, max_states: int) -> bool:
    """Check if a formula holds when evaluated from a specific location as initial."""
    if _is_state_formula(formula):
        return location in _check_atomic_locations(ta, formula)
    # For temporal formulas, create a modified TA starting from this location
    modified_ta = TimedAutomaton(
        locations=ta.locations,
        initial=location,
        clocks=ta.clocks,
        edges=ta.edges,
        invariants=ta.invariants,
        accepting=ta.accepting,
        alphabet=ta.alphabet,
    )
    if hasattr(ta, 'labels'):
        modified_ta.labels = ta.labels
    result = check_tctl(modified_ta, formula, max_states)
    return result.satisfied


def _check_ef_unbounded(ta: TimedAutomaton, phi: TCTLFormula,
                        max_states: int = 10000) -> TCTLResult:
    """Check E F phi (unbounded): exists path reaching phi."""
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None
    is_temporal = not _is_state_formula(phi)

    init = _initial_state(ta)
    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        if is_temporal:
            # For temporal sub-formulas, check from this location
            if _check_loc_satisfies_formula(ta, state.location, phi, max_states // 2):
                return TCTLResult(
                    satisfied=True,
                    formula=EF(phi),
                    satisfying_locations={state.location},
                    witness_trace=_extract_trace(state),
                    states_explored=states_explored,
                    zones_created=zones_created,
                )
        elif phi_locs is not None and state.location in phi_locs:
            return TCTLResult(
                satisfied=True,
                formula=EF(phi),
                satisfying_locations={state.location},
                witness_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )

        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        for succ in _compute_successors(ta, state):
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=False,
        formula=EF(phi),
        states_explored=states_explored,
        zones_created=zones_created,
    )


def _check_ag_unbounded(ta: TimedAutomaton, phi: TCTLFormula,
                        max_states: int = 10000) -> TCTLResult:
    """Check A G phi: phi holds on all reachable states."""
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None

    init = _initial_state(ta)
    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        loc_ok = phi_locs is None or state.location in phi_locs
        if not loc_ok:
            return TCTLResult(
                satisfied=False,
                formula=AG(phi),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )

        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        for succ in _compute_successors(ta, state):
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=True,
        formula=AG(phi),
        satisfying_locations=phi_locs if phi_locs else set(ta.locations),
        states_explored=states_explored,
        zones_created=zones_created,
    )


# ============================================================
# Main TCTL Model Checker
# ============================================================

def check_tctl(ta: TimedAutomaton, formula: TCTLFormula,
               max_states: int = 10000) -> TCTLResult:
    """Check a TCTL formula against a timed automaton.

    Supports:
    - Atomic propositions (location labels)
    - Boolean combinations (and, or, not, implies)
    - E F_{bound} phi, A F_{bound} phi
    - E G_{bound} phi, A G_{bound} phi
    - E [phi U_{bound} psi], A [phi U_{bound} psi]
    - Both bounded and unbounded variants
    """
    if isinstance(formula, (TrueF, FalseF, Atomic)):
        locs = _check_atomic_locations(ta, formula)
        sat = ta.initial in locs
        return TCTLResult(satisfied=sat, formula=formula, satisfying_locations=locs)

    if isinstance(formula, Not):
        sub_result = check_tctl(ta, formula.sub, max_states)
        return TCTLResult(
            satisfied=not sub_result.satisfied,
            formula=formula,
            satisfying_locations=set(ta.locations) - sub_result.satisfying_locations,
            states_explored=sub_result.states_explored,
            zones_created=sub_result.zones_created,
        )

    if isinstance(formula, And):
        left = check_tctl(ta, formula.left, max_states)
        if not left.satisfied:
            return TCTLResult(satisfied=False, formula=formula,
                              states_explored=left.states_explored,
                              zones_created=left.zones_created)
        right = check_tctl(ta, formula.right, max_states)
        return TCTLResult(
            satisfied=right.satisfied,
            formula=formula,
            satisfying_locations=left.satisfying_locations & right.satisfying_locations,
            states_explored=left.states_explored + right.states_explored,
            zones_created=left.zones_created + right.zones_created,
        )

    if isinstance(formula, Or):
        left = check_tctl(ta, formula.left, max_states)
        if left.satisfied:
            return TCTLResult(satisfied=True, formula=formula,
                              satisfying_locations=left.satisfying_locations,
                              states_explored=left.states_explored,
                              zones_created=left.zones_created)
        right = check_tctl(ta, formula.right, max_states)
        return TCTLResult(
            satisfied=right.satisfied,
            formula=formula,
            satisfying_locations=left.satisfying_locations | right.satisfying_locations,
            states_explored=left.states_explored + right.states_explored,
            zones_created=left.zones_created + right.zones_created,
        )

    if isinstance(formula, Implies):
        return check_tctl(ta, Or(Not(formula.left), formula.right), max_states)

    # Temporal operators
    if isinstance(formula, EF):
        if formula.bound.bound_type == BoundType.UNBOUNDED:
            return _check_ef_unbounded(ta, formula.sub, max_states)
        return _check_ef_bounded(ta, formula.sub, formula.bound, max_states)

    if isinstance(formula, AF):
        if formula.bound.bound_type == BoundType.UNBOUNDED:
            return _check_af_unbounded(ta, formula.sub, max_states)
        return _check_af_bounded(ta, formula.sub, formula.bound, max_states)

    if isinstance(formula, EG):
        if formula.bound.bound_type == BoundType.UNBOUNDED:
            return _check_eg_unbounded(ta, formula.sub, max_states)
        return _check_eg_bounded(ta, formula.sub, formula.bound, max_states)

    if isinstance(formula, AG):
        if formula.bound.bound_type == BoundType.UNBOUNDED:
            return _check_ag_unbounded(ta, formula.sub, max_states)
        return _check_ag_bounded(ta, formula.sub, formula.bound, max_states)

    if isinstance(formula, EU):
        if formula.bound.bound_type == BoundType.UNBOUNDED:
            return _check_eu_unbounded(ta, formula.left, formula.right, max_states)
        return _check_eu_bounded(ta, formula.left, formula.right, formula.bound, max_states)

    if isinstance(formula, AU):
        if formula.bound.bound_type == BoundType.UNBOUNDED:
            return _check_au_unbounded(ta, formula.left, formula.right, max_states)
        return _check_au_bounded(ta, formula.left, formula.right, formula.bound, max_states)

    raise ValueError(f"Unknown formula type: {type(formula)}")


def _can_stay_forever(ta: TimedAutomaton, location: str) -> bool:
    """Check if a location can let time pass indefinitely.
    A location can stay forever if it has no invariant that bounds time,
    or if it has a self-loop that resets clocks and the invariant allows re-entry.
    """
    inv = ta.invariants.get(location)
    if inv is None or inv.is_true():
        return True
    # Check if invariant has upper bounds on clocks (which force exit)
    for c in inv.constraints:
        if c.op in (CompOp.LE, CompOp.LT, CompOp.EQ):
            # Upper bound on a clock -- forces exit eventually
            return False
    return True


def _check_eg_unbounded(ta: TimedAutomaton, phi: TCTLFormula,
                        max_states: int = 10000) -> TCTLResult:
    """Check E G phi: exists infinite path where phi always holds.

    Three ways to satisfy E G phi:
    1. A cycle in phi-satisfying states (zone subsumption)
    2. A phi-satisfying location with no invariant (can stay forever)
    3. A phi-satisfying self-loop that allows infinite repetition
    """
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None

    init = _initial_state(ta)
    if phi_locs is not None and init.location not in phi_locs:
        return TCTLResult(satisfied=False, formula=EG(phi),
                          states_explored=1, zones_created=1)

    queue = deque([init])
    visited = {}  # location -> list of zones
    states_explored = 0
    zones_created = 1
    found_cycle = False
    cycle_state = None

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        loc = state.location
        loc_ok = phi_locs is None or loc in phi_locs
        if not loc_ok:
            continue

        # Check if this location can stay forever (no invariant forcing exit)
        if _can_stay_forever(ta, loc):
            return TCTLResult(
                satisfied=True,
                formula=EG(phi),
                satisfying_locations=phi_locs if phi_locs else set(ta.locations),
                witness_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )

        if loc not in visited:
            visited[loc] = []

        # Check subsumption -- if a previous zone includes this one, we have a cycle
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                found_cycle = True
                cycle_state = state
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        # Only expand through phi-satisfying successors
        for succ in _compute_successors(ta, state):
            zones_created += 1
            succ_ok = phi_locs is None or succ.location in phi_locs
            if succ_ok:
                queue.append(succ)

    if found_cycle:
        return TCTLResult(
            satisfied=True,
            formula=EG(phi),
            satisfying_locations=phi_locs if phi_locs else set(ta.locations),
            witness_trace=_extract_trace(cycle_state) if cycle_state else None,
            states_explored=states_explored,
            zones_created=zones_created,
        )

    return TCTLResult(
        satisfied=False,
        formula=EG(phi),
        states_explored=states_explored,
        zones_created=zones_created,
    )


def _check_af_unbounded(ta: TimedAutomaton, phi: TCTLFormula,
                        max_states: int = 10000) -> TCTLResult:
    """Check A F phi (unbounded): all paths must eventually reach phi.

    Fails if any reachable non-phi state can stay forever (no invariant forcing exit)
    or if there's a cycle through non-phi states.
    """
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None

    init = _initial_state(ta)
    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        # Check if phi holds
        loc_ok = phi_locs is None or state.location in phi_locs
        if loc_ok:
            continue  # This path satisfied phi

        # phi not satisfied here. Can this state stay forever?
        if _can_stay_forever(ta, state.location):
            return TCTLResult(
                satisfied=False,
                formula=AF(phi),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )

        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        succs = _compute_successors(ta, state)
        if not succs:
            # Deadlock without phi
            return TCTLResult(
                satisfied=False,
                formula=AF(phi),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )
        for succ in succs:
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=True,
        formula=AF(phi),
        satisfying_locations=phi_locs if phi_locs else set(),
        states_explored=states_explored,
        zones_created=zones_created,
    )


def _check_eu_unbounded(ta: TimedAutomaton, phi: TCTLFormula, psi: TCTLFormula,
                        max_states: int = 10000) -> TCTLResult:
    """Check E [phi U psi]: exists path where phi holds until psi."""
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None
    psi_locs = _check_atomic_locations(ta, psi) if _is_state_formula(psi) else None

    init = _initial_state(ta)

    # Check if initial state satisfies psi
    if psi_locs is not None and init.location in psi_locs:
        return TCTLResult(
            satisfied=True,
            formula=EU(phi, psi),
            satisfying_locations={init.location},
            witness_trace=_extract_trace(init),
            states_explored=1,
            zones_created=1,
        )

    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        # phi must hold on path (before psi)
        phi_ok = phi_locs is None or state.location in phi_locs
        if not phi_ok:
            continue

        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        for succ in _compute_successors(ta, state):
            zones_created += 1
            psi_ok = psi_locs is None or succ.location in psi_locs
            if psi_ok and psi_locs is not None:
                return TCTLResult(
                    satisfied=True,
                    formula=EU(phi, psi),
                    satisfying_locations={succ.location},
                    witness_trace=_extract_trace(succ),
                    states_explored=states_explored,
                    zones_created=zones_created,
                )
            queue.append(succ)

    return TCTLResult(
        satisfied=False,
        formula=EU(phi, psi),
        states_explored=states_explored,
        zones_created=zones_created,
    )


def _check_au_unbounded(ta: TimedAutomaton, phi: TCTLFormula, psi: TCTLFormula,
                        max_states: int = 10000) -> TCTLResult:
    """Check A [phi U psi]: all paths satisfy phi until psi."""
    phi_locs = _check_atomic_locations(ta, phi) if _is_state_formula(phi) else None
    psi_locs = _check_atomic_locations(ta, psi) if _is_state_formula(psi) else None

    init = _initial_state(ta)
    queue = deque([init])
    visited = {}
    states_explored = 0
    zones_created = 1

    while queue and states_explored < max_states:
        state = queue.popleft()
        states_explored += 1

        # Check psi
        psi_ok = psi_locs is None or state.location in psi_locs
        if psi_ok:
            continue  # This path satisfied

        # phi must hold
        phi_ok = phi_locs is None or state.location in phi_locs
        if not phi_ok:
            return TCTLResult(
                satisfied=False,
                formula=AU(phi, psi),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )

        loc = state.location
        if loc not in visited:
            visited[loc] = []
        subsumed = False
        for prev_zone in visited[loc]:
            if prev_zone.includes(state.zone):
                subsumed = True
                break
        if subsumed:
            continue
        visited[loc].append(state.zone)

        succs = _compute_successors(ta, state)
        if not succs:
            # Deadlock without psi
            return TCTLResult(
                satisfied=False,
                formula=AU(phi, psi),
                counterexample_trace=_extract_trace(state),
                states_explored=states_explored,
                zones_created=zones_created,
            )
        for succ in succs:
            zones_created += 1
            queue.append(succ)

    return TCTLResult(
        satisfied=True,
        formula=AU(phi, psi),
        satisfying_locations=psi_locs if psi_locs else set(),
        states_explored=states_explored,
        zones_created=zones_created,
    )


# ============================================================
# Convenience API
# ============================================================

def check_tctl_batch(ta: TimedAutomaton, formulas: List[TCTLFormula],
                     max_states: int = 10000) -> List[TCTLResult]:
    """Check multiple TCTL formulas against a timed automaton."""
    return [check_tctl(ta, f, max_states) for f in formulas]


def tctl_summary(ta: TimedAutomaton, formulas: List[TCTLFormula],
                 max_states: int = 10000) -> str:
    """Generate a summary of TCTL verification results."""
    results = check_tctl_batch(ta, formulas, max_states)
    lines = [f"TCTL Verification Summary ({len(formulas)} properties)"]
    lines.append("=" * 60)
    for r in results:
        status = "PASS" if r.satisfied else "FAIL"
        lines.append(f"  [{status}] {r.formula} ({r.states_explored} states)")
    passed = sum(1 for r in results if r.satisfied)
    lines.append(f"\n{passed}/{len(formulas)} properties satisfied")
    return "\n".join(lines)


# ============================================================
# Example Timed Automata (with labels)
# ============================================================

def labeled_ta(locations, initial, clocks, edges, invariants=None,
               labels=None, accepting=None):
    """Create a timed automaton with location labels."""
    inv = invariants or {}
    ta = simple_ta(
        locations=locations,
        initial=initial,
        clocks=clocks,
        edges=edges,
        invariants=inv,
    )
    if accepting:
        ta.accepting = set(accepting)
    if labels:
        ta.labels = labels
    return ta


def example_light_controller():
    """Simple light controller: off -> on -> off with timing constraints.
    - Light turns on within 2 time units
    - Light stays on for at most 5 time units
    """
    return labeled_ta(
        locations=["off", "on"],
        initial="off",
        clocks=["x"],
        edges=[
            ("off", "on", "turn_on", clock_leq("x", 2), frozenset(["x"])),
            ("on", "off", "turn_off", clock_geq("x", 3), frozenset(["x"])),
        ],
        invariants={"on": clock_leq("x", 5)},
        labels={"off": {"idle"}, "on": {"active"}},
    )


def example_request_response():
    """Request-response system with deadline.
    - Request must be processed within 10 time units
    - Processing takes at least 2 time units
    """
    return labeled_ta(
        locations=["idle", "processing", "done", "timeout"],
        initial="idle",
        clocks=["x"],
        edges=[
            ("idle", "processing", "request", true_guard(), frozenset(["x"])),
            ("processing", "done", "respond", clock_geq("x", 2), frozenset()),
            ("processing", "timeout", "expire", clock_gt("x", 10), frozenset()),
        ],
        invariants={"processing": clock_leq("x", 11)},
        labels={
            "idle": {"waiting"},
            "processing": {"busy"},
            "done": {"success"},
            "timeout": {"failure"},
        },
    )


def example_mutex_protocol():
    """Simple mutex protocol with timing.
    Process: idle -> trying -> critical -> idle
    Must enter critical within 5 time units of trying.
    Must leave critical within 3 time units.
    """
    return labeled_ta(
        locations=["idle", "trying", "critical"],
        initial="idle",
        clocks=["x"],
        edges=[
            ("idle", "trying", "try", true_guard(), frozenset(["x"])),
            ("trying", "critical", "enter", clock_leq("x", 5), frozenset(["x"])),
            ("critical", "idle", "exit", clock_leq("x", 3), frozenset(["x"])),
        ],
        invariants={
            "trying": clock_leq("x", 5),
            "critical": clock_leq("x", 3),
        },
        labels={
            "idle": {"free"},
            "trying": {"waiting"},
            "critical": {"locked"},
        },
    )


def example_train_crossing():
    """Train crossing controller.
    Train: far -> near -> crossing -> far
    Gate: open -> lowering -> closed -> raising -> open
    Timing: gate must close before train arrives at crossing.
    """
    return labeled_ta(
        locations=["approach", "near", "crossing", "passed"],
        initial="approach",
        clocks=["x"],
        edges=[
            ("approach", "near", "approach", clock_geq("x", 2), frozenset(["x"])),
            ("near", "crossing", "enter", clock_geq("x", 3), frozenset(["x"])),
            ("crossing", "passed", "leave", clock_leq("x", 5), frozenset(["x"])),
            ("passed", "approach", "reset", true_guard(), frozenset(["x"])),
        ],
        invariants={
            "near": clock_leq("x", 5),
            "crossing": clock_leq("x", 5),
        },
        labels={
            "approach": {"safe"},
            "near": {"warning"},
            "crossing": {"danger"},
            "passed": {"safe"},
        },
    )
