"""
V078: Partial Order Reduction for Model Checking

Reduces state explosion in concurrent system verification by exploiting
commutativity of independent transitions. Only a representative subset of
interleavings is explored while preserving verification correctness.

Implements:
1. Explicit-state concurrent system model (processes, shared variables, transitions)
2. Independence relation computation (static + dynamic)
3. Stubborn set method (Valmari)
4. Ample set method (Clarke/Grumberg/Peled)
5. Sleep set method (Godefroid)
6. Combined POR + sleep sets
7. Full-state BFS model checker (baseline)
8. POR-reduced model checker
9. Deadlock detection, safety property checking, reachability
10. Statistics and comparison APIs

Standalone implementation -- no composition with existing V-challenges needed,
as this operates on an explicit concurrent system model (not BDDs or SMT).
"""

from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Tuple, Optional, Callable, FrozenSet, Any, NamedTuple
)
from collections import deque
from enum import Enum
import itertools


# ---------------------------------------------------------------------------
# Concurrent System Model
# ---------------------------------------------------------------------------

class VarType(Enum):
    INT = "int"
    BOOL = "bool"


@dataclass(frozen=True)
class Location:
    """A control location in a process."""
    process: str
    label: str

    def __repr__(self):
        return f"{self.process}.{self.label}"


@dataclass(frozen=True)
class Transition:
    """A transition in a concurrent system."""
    process: str
    source: str        # source location label
    target: str        # target location label
    guard: Optional[Callable] = None   # state -> bool
    action: Optional[Callable] = None  # state -> state (returns new var assignment)
    reads: FrozenSet[str] = frozenset()   # variables read
    writes: FrozenSet[str] = frozenset()  # variables written
    label: str = ""

    def __repr__(self):
        lbl = self.label or f"{self.process}:{self.source}->{self.target}"
        return lbl

    def __hash__(self):
        return hash((self.process, self.source, self.target, self.label))

    def __eq__(self, other):
        if not isinstance(other, Transition):
            return NotImplemented
        return (self.process == other.process and
                self.source == other.source and
                self.target == other.target and
                self.label == other.label)


@dataclass
class Process:
    """A process in the concurrent system."""
    name: str
    locations: List[str]
    initial: str
    transitions: List[Transition] = field(default_factory=list)

    def add_transition(self, source, target, guard=None, action=None,
                       reads=frozenset(), writes=frozenset(), label=""):
        t = Transition(
            process=self.name, source=source, target=target,
            guard=guard, action=action,
            reads=frozenset(reads) if not isinstance(reads, frozenset) else reads,
            writes=frozenset(writes) if not isinstance(writes, frozenset) else writes,
            label=label or f"{self.name}:{source}->{target}"
        )
        self.transitions.append(t)
        return t


@dataclass(frozen=True)
class GlobalState:
    """A global state: process locations + shared variable values."""
    locations: Tuple  # ((proc_name, loc_label), ...)
    variables: Tuple  # ((var_name, value), ...)

    def get_loc(self, process):
        for p, l in self.locations:
            if p == process:
                return l
        return None

    def get_var(self, name):
        for n, v in self.variables:
            if n == name:
                return v
        return None

    def set_loc(self, process, new_loc):
        new_locs = tuple(
            (p, new_loc) if p == process else (p, l)
            for p, l in self.locations
        )
        return GlobalState(locations=new_locs, variables=self.variables)

    def set_var(self, name, value):
        new_vars = tuple(
            (n, value) if n == name else (n, v)
            for n, v in self.variables
        )
        return GlobalState(locations=self.locations, variables=new_vars)

    def set_vars(self, updates):
        """Apply multiple variable updates."""
        var_dict = dict(self.variables)
        var_dict.update(updates)
        new_vars = tuple(sorted(var_dict.items()))
        return GlobalState(locations=self.locations, variables=new_vars)

    def to_dict(self):
        """Convert to mutable dict for guard/action evaluation."""
        d = dict(self.variables)
        for p, l in self.locations:
            d[f"_loc_{p}"] = l
        return d

    def __repr__(self):
        locs = ",".join(f"{p}={l}" for p, l in self.locations)
        vs = ",".join(f"{n}={v}" for n, v in self.variables)
        return f"({locs}|{vs})" if vs else f"({locs})"


@dataclass
class ConcurrentSystem:
    """A concurrent system with processes and shared variables."""
    processes: Dict[str, Process] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)  # name -> initial value
    _transitions_cache: Optional[List[Transition]] = field(default=None, repr=False)

    def add_process(self, name, locations, initial):
        p = Process(name=name, locations=locations, initial=initial)
        self.processes[name] = p
        self._transitions_cache = None
        return p

    def add_variable(self, name, initial_value=0):
        self.variables[name] = initial_value

    def initial_state(self):
        locs = tuple(sorted((p.name, p.initial) for p in self.processes.values()))
        vs = tuple(sorted(self.variables.items()))
        return GlobalState(locations=locs, variables=vs)

    def all_transitions(self):
        if self._transitions_cache is None:
            self._transitions_cache = []
            for p in self.processes.values():
                self._transitions_cache.extend(p.transitions)
        return self._transitions_cache

    def enabled(self, state):
        """Return transitions enabled in the given state."""
        result = []
        for t in self.all_transitions():
            if state.get_loc(t.process) != t.source:
                continue
            if t.guard is not None:
                try:
                    if not t.guard(state.to_dict()):
                        continue
                except Exception:
                    continue
            result.append(t)
        return result

    def execute(self, state, transition):
        """Execute a transition, returning the successor state."""
        new_state = state.set_loc(transition.process, transition.target)
        if transition.action is not None:
            updates = transition.action(state.to_dict())
            if isinstance(updates, dict):
                new_state = new_state.set_vars(updates)
        return new_state


# ---------------------------------------------------------------------------
# Independence Relation
# ---------------------------------------------------------------------------

def are_independent_static(t1: Transition, t2: Transition) -> bool:
    """
    Static independence: two transitions are independent if they belong to
    different processes and their read/write sets don't conflict.
    """
    if t1.process == t2.process:
        return False
    # Check read-write and write-write conflicts
    if t1.writes & t2.writes:
        return False
    if t1.writes & t2.reads:
        return False
    if t1.reads & t2.writes:
        return False
    return True


def compute_independence_relation(transitions: List[Transition]) -> Set[Tuple[int, int]]:
    """Compute the static independence relation as index pairs."""
    independent = set()
    for i in range(len(transitions)):
        for j in range(i + 1, len(transitions)):
            if are_independent_static(transitions[i], transitions[j]):
                independent.add((i, j))
                independent.add((j, i))
    return independent


def are_independent_dynamic(system: ConcurrentSystem, state: GlobalState,
                            t1: Transition, t2: Transition) -> bool:
    """
    Dynamic independence: check independence in a specific state by
    actually executing both orderings and comparing results.
    """
    if t1.process == t2.process:
        return False
    # First check static independence (fast path)
    if are_independent_static(t1, t2):
        return True
    # Dynamic check: execute t1;t2 and t2;t1, compare results
    try:
        s12 = system.execute(system.execute(state, t1), t2)
        s21 = system.execute(system.execute(state, t2), t1)
        return s12 == s21
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stubborn Set Method (Valmari)
# ---------------------------------------------------------------------------

def compute_stubborn_set(system: ConcurrentSystem, state: GlobalState,
                         enabled_set: List[Transition]) -> List[Transition]:
    """
    Compute a stubborn set for the given state.

    A stubborn set T_s at state s satisfies:
    1. If enabled(s) is non-empty, T_s contains at least one enabled transition
    2. For every enabled t in T_s, all transitions dependent on t are also in T_s
    3. For every disabled t in T_s, there's a "necessary enabling" transition in T_s

    We use a simplified approach: pick one enabled transition, close under dependence.
    """
    if not enabled_set:
        return []

    all_trans = system.all_transitions()
    enabled_ids = set(id(t) for t in enabled_set)

    # Start with the first enabled transition (heuristic: pick one with fewest dependencies)
    seed = min(enabled_set, key=lambda t: len(t.reads | t.writes))

    stubborn = {id(seed): seed}
    worklist = [seed]

    while worklist:
        current = worklist.pop()
        for t in all_trans:
            if id(t) in stubborn:
                continue
            if not are_independent_static(current, t):
                stubborn[id(t)] = t
                worklist.append(t)

    # Return only enabled transitions from the stubborn set
    result = [t for t in stubborn.values() if id(t) in enabled_ids]
    # Ensure at least one enabled transition
    if not result and enabled_set:
        result = [enabled_set[0]]
    return result


# ---------------------------------------------------------------------------
# Ample Set Method (Clarke/Grumberg/Peled)
# ---------------------------------------------------------------------------

def compute_ample_set(system: ConcurrentSystem, state: GlobalState,
                      enabled_set: List[Transition],
                      on_stack: Optional[Set[GlobalState]] = None,
                      check_visibility: Optional[Callable] = None) -> List[Transition]:
    """
    Compute an ample set for the given state.

    Ample set conditions (C0-C3):
    C0: ample(s) = {} iff enabled(s) = {}
    C1: Along every path in the full state graph, a transition dependent on
        a transition in ample(s) cannot occur before some transition in ample(s)
        (approximated: all enabled transitions of one process, all independent of other enabled)
    C2: If ample(s) != enabled(s), no transition in ample(s) is "visible"
        (touches a variable in the property) -- for safety properties
    C3: A cycle in the reduced graph cannot be formed entirely from states
        where ample(s) is a strict subset of enabled(s)

    Simplified: try each process's enabled transitions as a candidate ample set.
    """
    if not enabled_set:
        return []

    if len(enabled_set) == 1:
        return enabled_set

    # Group enabled transitions by process
    by_process = {}
    for t in enabled_set:
        by_process.setdefault(t.process, []).append(t)

    # Try each process as candidate ample set
    for proc, proc_trans in by_process.items():
        # C1 check: all transitions of other processes that are enabled
        # must be independent of all transitions in proc_trans
        other_enabled = [t for t in enabled_set if t.process != proc]
        all_independent = True
        for pt in proc_trans:
            for ot in other_enabled:
                if not are_independent_static(pt, ot):
                    all_independent = False
                    break
            if not all_independent:
                break

        if not all_independent:
            continue

        # C2 check: if using check_visibility, ensure no visible transition
        if check_visibility and proc_trans != enabled_set:
            if any(check_visibility(t) for t in proc_trans):
                continue

        # C3 check: if a successor is on the DFS stack, don't reduce
        if on_stack is not None and proc_trans != enabled_set:
            all_on_stack = True
            for t in proc_trans:
                succ = system.execute(state, t)
                if succ not in on_stack:
                    all_on_stack = False
                    break
            if all_on_stack:
                continue

        return proc_trans

    # No reduction possible -- return full enabled set
    return enabled_set


# ---------------------------------------------------------------------------
# Sleep Set Method (Godefroid)
# ---------------------------------------------------------------------------

@dataclass
class SleepEntry:
    """Entry in a sleep set: a transition to skip."""
    transition: Transition


def compute_sleep_successors(system: ConcurrentSystem, state: GlobalState,
                             transition: Transition,
                             current_sleep: Set[Transition]) -> Set[Transition]:
    """
    Compute the sleep set for the successor state after executing `transition`.

    Sleep set propagation: if t is executed from s with sleep set Z,
    the sleep set at succ(s,t) is:
      {t' in Z | t' independent of t} union {t'' | t'' was explored before t at s and independent of t}
    """
    new_sleep = set()
    for st in current_sleep:
        if are_independent_static(st, transition):
            new_sleep.add(st)
    return new_sleep


# ---------------------------------------------------------------------------
# Model Checking Results
# ---------------------------------------------------------------------------

class MCResult(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    DEADLOCK = "deadlock"


@dataclass
class ModelCheckOutput:
    """Result of model checking."""
    result: MCResult
    states_explored: int = 0
    transitions_explored: int = 0
    counterexample: Optional[List[Tuple[GlobalState, Optional[Transition]]]] = None
    deadlock_state: Optional[GlobalState] = None
    reduction_ratio: float = 1.0  # states_reduced / states_full
    method: str = "full"

    def __repr__(self):
        return (f"MCOutput({self.result.value}, states={self.states_explored}, "
                f"trans={self.transitions_explored}, method={self.method})")


# ---------------------------------------------------------------------------
# Full State Space BFS (Baseline)
# ---------------------------------------------------------------------------

def full_state_bfs(system: ConcurrentSystem,
                   property_fn: Optional[Callable] = None,
                   check_deadlock: bool = False,
                   max_states: int = 100000) -> ModelCheckOutput:
    """
    Exhaustive BFS exploration of the full state space.
    property_fn: state -> bool (should be True for safe states)
    """
    init = system.initial_state()
    visited = {init}
    queue = deque([(init, [(init, None)])])  # (state, trace)
    states_explored = 0
    trans_explored = 0

    while queue:
        if states_explored >= max_states:
            break
        state, trace = queue.popleft()
        states_explored += 1

        # Check property
        if property_fn is not None and not property_fn(state):
            return ModelCheckOutput(
                result=MCResult.UNSAFE,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                method="full_bfs"
            )

        enabled = system.enabled(state)

        # Check deadlock
        if check_deadlock and not enabled:
            return ModelCheckOutput(
                result=MCResult.DEADLOCK,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                deadlock_state=state,
                method="full_bfs"
            )

        for t in enabled:
            succ = system.execute(state, t)
            trans_explored += 1
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, trace + [(succ, t)]))

    return ModelCheckOutput(
        result=MCResult.SAFE,
        states_explored=states_explored,
        transitions_explored=trans_explored,
        method="full_bfs"
    )


# ---------------------------------------------------------------------------
# POR-Reduced Model Checker (Stubborn Sets)
# ---------------------------------------------------------------------------

def stubborn_set_bfs(system: ConcurrentSystem,
                     property_fn: Optional[Callable] = None,
                     check_deadlock: bool = False,
                     max_states: int = 100000) -> ModelCheckOutput:
    """BFS model checking with stubborn set reduction."""
    init = system.initial_state()
    visited = {init}
    queue = deque([(init, [(init, None)])])
    states_explored = 0
    trans_explored = 0

    while queue:
        if states_explored >= max_states:
            break
        state, trace = queue.popleft()
        states_explored += 1

        if property_fn is not None and not property_fn(state):
            return ModelCheckOutput(
                result=MCResult.UNSAFE,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                method="stubborn_bfs"
            )

        enabled = system.enabled(state)

        if check_deadlock and not enabled:
            return ModelCheckOutput(
                result=MCResult.DEADLOCK,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                deadlock_state=state,
                method="stubborn_bfs"
            )

        # Compute stubborn set
        reduced = compute_stubborn_set(system, state, enabled)

        for t in reduced:
            succ = system.execute(state, t)
            trans_explored += 1
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, trace + [(succ, t)]))

    return ModelCheckOutput(
        result=MCResult.SAFE,
        states_explored=states_explored,
        transitions_explored=trans_explored,
        method="stubborn_bfs"
    )


# ---------------------------------------------------------------------------
# POR-Reduced Model Checker (Ample Sets) with DFS
# ---------------------------------------------------------------------------

def ample_set_dfs(system: ConcurrentSystem,
                  property_fn: Optional[Callable] = None,
                  check_deadlock: bool = False,
                  check_visibility: Optional[Callable] = None,
                  max_states: int = 100000) -> ModelCheckOutput:
    """DFS model checking with ample set reduction (supports C3 via stack check)."""
    init = system.initial_state()
    visited = set()
    on_stack = set()
    states_explored = 0
    trans_explored = 0
    result = [None]  # mutable container for nested function

    def dfs(state, trace):
        nonlocal states_explored, trans_explored

        if states_explored >= max_states:
            return
        if result[0] is not None:
            return

        visited.add(state)
        on_stack.add(state)
        states_explored += 1

        if property_fn is not None and not property_fn(state):
            result[0] = ModelCheckOutput(
                result=MCResult.UNSAFE,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                method="ample_dfs"
            )
            on_stack.discard(state)
            return

        enabled = system.enabled(state)

        if check_deadlock and not enabled:
            result[0] = ModelCheckOutput(
                result=MCResult.DEADLOCK,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                deadlock_state=state,
                method="ample_dfs"
            )
            on_stack.discard(state)
            return

        reduced = compute_ample_set(system, state, enabled, on_stack, check_visibility)

        for t in reduced:
            succ = system.execute(state, t)
            trans_explored += 1
            if succ not in visited:
                dfs(succ, trace + [(succ, t)])
                if result[0] is not None:
                    on_stack.discard(state)
                    return

        on_stack.discard(state)

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, max_states + 100))
    try:
        dfs(init, [(init, None)])
    finally:
        sys.setrecursionlimit(old_limit)

    if result[0] is not None:
        return result[0]

    return ModelCheckOutput(
        result=MCResult.SAFE,
        states_explored=states_explored,
        transitions_explored=trans_explored,
        method="ample_dfs"
    )


# ---------------------------------------------------------------------------
# Sleep Set BFS
# ---------------------------------------------------------------------------

def sleep_set_bfs(system: ConcurrentSystem,
                  property_fn: Optional[Callable] = None,
                  check_deadlock: bool = False,
                  max_states: int = 100000) -> ModelCheckOutput:
    """BFS model checking with sleep set reduction."""
    init = system.initial_state()
    # Track (state, sleep_set) pairs -- same state with different sleep sets are different
    visited = {}  # state -> set of explored sleep-set fingerprints
    visited[init] = {frozenset()}
    queue = deque([(init, frozenset(), [(init, None)])])
    states_explored = 0
    trans_explored = 0

    while queue:
        if states_explored >= max_states:
            break
        state, sleep, trace = queue.popleft()
        states_explored += 1

        if property_fn is not None and not property_fn(state):
            return ModelCheckOutput(
                result=MCResult.UNSAFE,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                method="sleep_bfs"
            )

        enabled = system.enabled(state)
        awake = [t for t in enabled if t not in sleep]

        if check_deadlock and not enabled:
            return ModelCheckOutput(
                result=MCResult.DEADLOCK,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                deadlock_state=state,
                method="sleep_bfs"
            )

        explored_at_state = []
        for t in awake:
            succ = system.execute(state, t)
            trans_explored += 1

            # Propagate sleep set
            new_sleep = set()
            for st in sleep:
                if are_independent_static(st, t):
                    new_sleep.add(st)
            # Add previously explored siblings that are independent
            for prev_t in explored_at_state:
                if are_independent_static(prev_t, t):
                    new_sleep.add(prev_t)

            sleep_fp = frozenset(id(s) for s in new_sleep)
            if succ not in visited:
                visited[succ] = set()

            if sleep_fp not in visited[succ]:
                visited[succ].add(sleep_fp)
                queue.append((succ, frozenset(new_sleep), trace + [(succ, t)]))

            explored_at_state.append(t)

    return ModelCheckOutput(
        result=MCResult.SAFE,
        states_explored=states_explored,
        transitions_explored=trans_explored,
        method="sleep_bfs"
    )


# ---------------------------------------------------------------------------
# Combined: Stubborn Sets + Sleep Sets
# ---------------------------------------------------------------------------

def combined_por_bfs(system: ConcurrentSystem,
                     property_fn: Optional[Callable] = None,
                     check_deadlock: bool = False,
                     max_states: int = 100000) -> ModelCheckOutput:
    """BFS with both stubborn sets and sleep sets for maximum reduction."""
    init = system.initial_state()
    visited = {}
    visited[init] = {frozenset()}
    queue = deque([(init, frozenset(), [(init, None)])])
    states_explored = 0
    trans_explored = 0

    while queue:
        if states_explored >= max_states:
            break
        state, sleep, trace = queue.popleft()
        states_explored += 1

        if property_fn is not None and not property_fn(state):
            return ModelCheckOutput(
                result=MCResult.UNSAFE,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                method="combined_por"
            )

        enabled = system.enabled(state)

        if check_deadlock and not enabled:
            return ModelCheckOutput(
                result=MCResult.DEADLOCK,
                states_explored=states_explored,
                transitions_explored=trans_explored,
                counterexample=trace,
                deadlock_state=state,
                method="combined_por"
            )

        # First apply stubborn set reduction
        stubborn = compute_stubborn_set(system, state, enabled)
        # Then filter by sleep set
        awake = [t for t in stubborn if t not in sleep]
        if not awake and stubborn:
            awake = stubborn[:1]  # ensure progress

        explored_at_state = []
        for t in awake:
            succ = system.execute(state, t)
            trans_explored += 1

            new_sleep = set()
            for st in sleep:
                if are_independent_static(st, t):
                    new_sleep.add(st)
            for prev_t in explored_at_state:
                if are_independent_static(prev_t, t):
                    new_sleep.add(prev_t)

            sleep_fp = frozenset(id(s) for s in new_sleep)
            if succ not in visited:
                visited[succ] = set()

            if sleep_fp not in visited[succ]:
                visited[succ].add(sleep_fp)
                queue.append((succ, frozenset(new_sleep), trace + [(succ, t)]))

            explored_at_state.append(t)

    return ModelCheckOutput(
        result=MCResult.SAFE,
        states_explored=states_explored,
        transitions_explored=trans_explored,
        method="combined_por"
    )


# ---------------------------------------------------------------------------
# Convenience: Model Check with Comparison
# ---------------------------------------------------------------------------

def model_check(system: ConcurrentSystem,
                property_fn: Optional[Callable] = None,
                check_deadlock: bool = False,
                method: str = "stubborn",
                max_states: int = 100000) -> ModelCheckOutput:
    """Main API: model check with specified POR method."""
    methods = {
        "full": full_state_bfs,
        "stubborn": stubborn_set_bfs,
        "sleep": sleep_set_bfs,
        "combined": combined_por_bfs,
    }
    if method == "ample":
        return ample_set_dfs(system, property_fn, check_deadlock,
                             max_states=max_states)
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Use: {list(methods.keys()) + ['ample']}")
    return methods[method](system, property_fn, check_deadlock, max_states)


def compare_methods(system: ConcurrentSystem,
                    property_fn: Optional[Callable] = None,
                    check_deadlock: bool = False,
                    max_states: int = 100000) -> Dict[str, ModelCheckOutput]:
    """Run all methods and compare state/transition counts."""
    results = {}
    for method in ["full", "stubborn", "ample", "sleep", "combined"]:
        results[method] = model_check(system, property_fn, check_deadlock,
                                      method=method, max_states=max_states)
    # Compute reduction ratios relative to full
    full_states = results["full"].states_explored
    if full_states > 0:
        for name, r in results.items():
            r.reduction_ratio = r.states_explored / full_states
    return results


# ---------------------------------------------------------------------------
# State Space Statistics
# ---------------------------------------------------------------------------

@dataclass
class StateSpaceStats:
    """Statistics about the state space."""
    total_states: int = 0
    total_transitions: int = 0
    deadlock_states: int = 0
    max_enabled: int = 0
    avg_enabled: float = 0.0
    independence_pairs: int = 0
    total_pairs: int = 0
    independence_ratio: float = 0.0


def compute_state_space_stats(system: ConcurrentSystem,
                              max_states: int = 100000) -> StateSpaceStats:
    """Compute statistics about the full state space."""
    init = system.initial_state()
    visited = {init}
    queue = deque([init])
    stats = StateSpaceStats()
    total_enabled = 0
    all_trans = system.all_transitions()

    # Compute independence relation
    n = len(all_trans)
    stats.total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            if are_independent_static(all_trans[i], all_trans[j]):
                stats.independence_pairs += 1
    if stats.total_pairs > 0:
        stats.independence_ratio = stats.independence_pairs / stats.total_pairs

    while queue and len(visited) < max_states:
        state = queue.popleft()
        stats.total_states += 1
        enabled = system.enabled(state)
        n_enabled = len(enabled)
        total_enabled += n_enabled
        stats.max_enabled = max(stats.max_enabled, n_enabled)

        if not enabled:
            stats.deadlock_states += 1

        for t in enabled:
            stats.total_transitions += 1
            succ = system.execute(state, t)
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    if stats.total_states > 0:
        stats.avg_enabled = total_enabled / stats.total_states

    return stats


# ---------------------------------------------------------------------------
# Reachability Analysis
# ---------------------------------------------------------------------------

def reachable_states(system: ConcurrentSystem,
                     method: str = "full",
                     max_states: int = 100000) -> Set[GlobalState]:
    """Compute all reachable states (using specified method)."""
    init = system.initial_state()
    visited = {init}
    queue = deque([init])

    while queue and len(visited) < max_states:
        state = queue.popleft()
        enabled = system.enabled(state)

        if method == "stubborn":
            explore = compute_stubborn_set(system, state, enabled)
        else:
            explore = enabled

        for t in explore:
            succ = system.execute(state, t)
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    return visited


def find_deadlocks(system: ConcurrentSystem,
                   max_states: int = 100000) -> List[GlobalState]:
    """Find all deadlock states."""
    init = system.initial_state()
    visited = {init}
    queue = deque([init])
    deadlocks = []

    while queue and len(visited) < max_states:
        state = queue.popleft()
        enabled = system.enabled(state)

        if not enabled:
            deadlocks.append(state)
            continue

        for t in enabled:
            succ = system.execute(state, t)
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    return deadlocks


# ---------------------------------------------------------------------------
# Example Systems
# ---------------------------------------------------------------------------

def make_mutex_system(n_processes: int = 2) -> ConcurrentSystem:
    """
    Peterson's mutual exclusion protocol for 2 processes,
    or a ticket-based lock for n > 2.
    """
    sys = ConcurrentSystem()

    if n_processes == 2:
        # Peterson's algorithm
        sys.add_variable("flag0", 0)
        sys.add_variable("flag1", 0)
        sys.add_variable("turn", 0)

        p0 = sys.add_process("P0", ["idle", "want", "wait", "crit"], "idle")
        p0.add_transition("idle", "want",
                          action=lambda s: {"flag0": 1},
                          writes=frozenset(["flag0"]),
                          label="P0:set_flag")
        p0.add_transition("want", "wait",
                          action=lambda s: {"turn": 1},
                          writes=frozenset(["turn"]),
                          label="P0:set_turn")
        p0.add_transition("wait", "crit",
                          guard=lambda s: s["flag1"] == 0 or s["turn"] == 0,
                          reads=frozenset(["flag1", "turn"]),
                          label="P0:enter_crit")
        p0.add_transition("crit", "idle",
                          action=lambda s: {"flag0": 0},
                          writes=frozenset(["flag0"]),
                          label="P0:exit_crit")

        p1 = sys.add_process("P1", ["idle", "want", "wait", "crit"], "idle")
        p1.add_transition("idle", "want",
                          action=lambda s: {"flag1": 1},
                          writes=frozenset(["flag1"]),
                          label="P1:set_flag")
        p1.add_transition("want", "wait",
                          action=lambda s: {"turn": 0},
                          writes=frozenset(["turn"]),
                          label="P1:set_turn")
        p1.add_transition("wait", "crit",
                          guard=lambda s: s["flag0"] == 0 or s["turn"] == 1,
                          reads=frozenset(["flag0", "turn"]),
                          label="P1:enter_crit")
        p1.add_transition("crit", "idle",
                          action=lambda s: {"flag1": 0},
                          writes=frozenset(["flag1"]),
                          label="P1:exit_crit")
    else:
        # Ticket lock for n processes
        sys.add_variable("ticket", 0)
        sys.add_variable("serving", 0)
        for i in range(n_processes):
            sys.add_variable(f"my_ticket_{i}", 0)

        for i in range(n_processes):
            p = sys.add_process(f"P{i}", ["idle", "wait", "crit"], "idle")
            vi = f"my_ticket_{i}"
            p.add_transition("idle", "wait",
                             action=lambda s, _i=i: {f"my_ticket_{_i}": s["ticket"],
                                                     "ticket": s["ticket"] + 1},
                             reads=frozenset(["ticket"]),
                             writes=frozenset(["ticket", vi]),
                             label=f"P{i}:take_ticket")
            p.add_transition("wait", "crit",
                             guard=lambda s, _i=i: s[f"my_ticket_{_i}"] == s["serving"],
                             reads=frozenset([vi, "serving"]),
                             label=f"P{i}:enter_crit")
            p.add_transition("crit", "idle",
                             action=lambda s: {"serving": s["serving"] + 1},
                             reads=frozenset(["serving"]),
                             writes=frozenset(["serving"]),
                             label=f"P{i}:exit_crit")

    return sys


def make_producer_consumer(buffer_size: int = 2) -> ConcurrentSystem:
    """Producer-consumer with bounded buffer."""
    sys = ConcurrentSystem()
    sys.add_variable("count", 0)

    prod = sys.add_process("producer", ["idle", "producing"], "idle")
    prod.add_transition("idle", "producing",
                        guard=lambda s: s["count"] < buffer_size,
                        action=lambda s: {"count": s["count"] + 1},
                        reads=frozenset(["count"]),
                        writes=frozenset(["count"]),
                        label="produce")
    prod.add_transition("producing", "idle",
                        label="prod_done")

    cons = sys.add_process("consumer", ["idle", "consuming"], "idle")
    cons.add_transition("idle", "consuming",
                        guard=lambda s: s["count"] > 0,
                        action=lambda s: {"count": s["count"] - 1},
                        reads=frozenset(["count"]),
                        writes=frozenset(["count"]),
                        label="consume")
    cons.add_transition("consuming", "idle",
                        label="cons_done")

    return sys


def make_dining_philosophers(n: int = 3) -> ConcurrentSystem:
    """Dining philosophers (deadlock-prone version)."""
    sys = ConcurrentSystem()
    for i in range(n):
        sys.add_variable(f"fork_{i}", 0)  # 0 = free, 1 = taken

    for i in range(n):
        left = f"fork_{i}"
        right = f"fork_{(i + 1) % n}"
        p = sys.add_process(f"Phil{i}", ["think", "hungry", "eat"], "think")

        p.add_transition("think", "hungry",
                         guard=lambda s, _l=left: s[_l] == 0,
                         action=lambda s, _l=left: {_l: 1},
                         reads=frozenset([left]),
                         writes=frozenset([left]),
                         label=f"Phil{i}:take_left")
        p.add_transition("hungry", "eat",
                         guard=lambda s, _r=right: s[_r] == 0,
                         action=lambda s, _r=right: {_r: 1},
                         reads=frozenset([right]),
                         writes=frozenset([right]),
                         label=f"Phil{i}:take_right")
        p.add_transition("eat", "think",
                         action=lambda s, _l=left, _r=right: {_l: 0, _r: 0},
                         writes=frozenset([left, right]),
                         label=f"Phil{i}:release")

    return sys


def make_counter_system(n_processes: int = 2, max_val: int = 3) -> ConcurrentSystem:
    """N processes incrementing a shared counter (race condition demo)."""
    sys = ConcurrentSystem()
    sys.add_variable("counter", 0)

    for i in range(n_processes):
        p = sys.add_process(f"P{i}", ["read", "write", "done"], "read")
        p.add_transition("read", "write",
                         reads=frozenset(["counter"]),
                         label=f"P{i}:read")
        p.add_transition("write", "done",
                         guard=lambda s: s["counter"] < max_val,
                         action=lambda s: {"counter": s["counter"] + 1},
                         reads=frozenset(["counter"]),
                         writes=frozenset(["counter"]),
                         label=f"P{i}:write")
        p.add_transition("done", "read",
                         label=f"P{i}:restart")

    return sys


def make_independent_system(n_processes: int = 3) -> ConcurrentSystem:
    """N fully independent processes (maximum POR benefit)."""
    sys = ConcurrentSystem()
    for i in range(n_processes):
        sys.add_variable(f"x_{i}", 0)

    for i in range(n_processes):
        vi = f"x_{i}"
        p = sys.add_process(f"P{i}", ["a", "b", "c"], "a")
        p.add_transition("a", "b",
                         action=lambda s, _v=vi: {_v: s[_v] + 1},
                         reads=frozenset([vi]),
                         writes=frozenset([vi]),
                         label=f"P{i}:a_to_b")
        p.add_transition("b", "c",
                         action=lambda s, _v=vi: {_v: s[_v] + 1},
                         reads=frozenset([vi]),
                         writes=frozenset([vi]),
                         label=f"P{i}:b_to_c")

    return sys
