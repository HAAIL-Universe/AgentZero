"""V171: Interpolation-Based Model Checking (McMillan's Method).

Craig interpolation for unbounded safety verification. Given a transition
system and safety property, uses bounded model checking + interpolation
to compute over-approximate reachable state sets.

Key ideas:
1. BMC: Check if bad states are reachable in k steps
2. If UNSAT: extract Craig interpolant from refutation proof
3. Interpolant over-approximates states reachable in <=k steps
4. Check if interpolant is inductive (closed under transitions)
5. If inductive: SAFE (interpolant is an invariant separating Init from Bad)
6. If not: increase k, refine, repeat

Composes: C037 (SMT solver) for satisfiability/interpolation,
          V170 (mu-calculus) for Kripke structure representation.

References:
  McMillan (2003): "Interpolation and SAT-based Model Checking"
  McMillan (2005): "Applications of Craig Interpolants in Model Checking"
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, FrozenSet
from enum import Enum, auto
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V170_mu_calculus_cegar'))
from mu_calculus_cegar import (
    KripkeStructure, ConcreteSystem, _state_key,
)


# ============================================================
# Transition System (Symbolic)
# ============================================================

@dataclass
class SymbolicTS:
    """Symbolic transition system for interpolation-based MC.

    States are assignments to integer variables.
    Init, transition, and bad are predicates (functions).
    """
    variables: List[str]
    init_pred: Callable[[Dict[str, int]], bool]
    trans_pred: Callable[[Dict[str, int], Dict[str, int]], bool]
    bad_pred: Callable[[Dict[str, int]], bool]

    # For finite enumeration
    domain: Optional[Dict[str, List[int]]] = None  # var -> possible values


@dataclass
class InterpolantState:
    """Abstract state in the interpolant computation.

    Represents a set of concrete states via a predicate.
    """
    predicate: Callable[[Dict[str, int]], bool]
    name: str


# ============================================================
# Concrete State Space Enumeration
# ============================================================

def enumerate_states(ts: SymbolicTS, max_states: int = 10000) -> List[Dict[str, int]]:
    """Enumerate reachable states via BFS."""
    if ts.domain is None:
        raise ValueError("Need domain for enumeration")

    # Generate all possible states from domain
    all_states = [{}]
    for var in ts.variables:
        new_states = []
        for s in all_states:
            for val in ts.domain[var]:
                ns = dict(s)
                ns[var] = val
                new_states.append(ns)
        all_states = new_states

    # Filter to init states and BFS
    init_states = [s for s in all_states if ts.init_pred(s)]
    reachable = []
    visited = set()

    queue = deque()
    for s in init_states:
        key = _state_key(s)
        if key not in visited:
            visited.add(key)
            reachable.append(s)
            queue.append(s)

    while queue and len(reachable) < max_states:
        s = queue.popleft()
        for t in all_states:
            if ts.trans_pred(s, t):
                key = _state_key(t)
                if key not in visited:
                    visited.add(key)
                    reachable.append(t)
                    queue.append(t)

    return reachable


# ============================================================
# BMC (Bounded Model Checking)
# ============================================================

@dataclass
class BMCResult:
    """Result of bounded model checking."""
    safe: bool           # True if no counterexample found at this depth
    depth: int           # bound checked
    counterexample: Optional[List[Dict[str, int]]] = None


def bounded_model_check(ts: SymbolicTS, max_depth: int,
                         max_states: int = 10000) -> BMCResult:
    """Check if bad states are reachable within max_depth steps.

    Returns BMCResult with counterexample if found.
    """
    if ts.domain is None:
        raise ValueError("Need domain for BMC")

    # Generate state space
    all_states = _all_states_from_domain(ts)
    init_states = [s for s in all_states if ts.init_pred(s)]

    # BFS up to max_depth
    frontier = [(s, [s]) for s in init_states]
    visited = {_state_key(s) for s in init_states}

    for depth in range(max_depth):
        # Check current frontier for bad states
        for s, path in frontier:
            if ts.bad_pred(s):
                return BMCResult(safe=False, depth=depth, counterexample=path)

        # Expand frontier
        next_frontier = []
        for s, path in frontier:
            for t in all_states:
                if ts.trans_pred(s, t):
                    key = _state_key(t)
                    if key not in visited:
                        visited.add(key)
                        next_frontier.append((t, path + [t]))
        frontier = next_frontier

        if not frontier:
            break

    # Check final frontier
    for s, path in frontier:
        if ts.bad_pred(s):
            return BMCResult(safe=False, depth=max_depth, counterexample=path)

    return BMCResult(safe=True, depth=max_depth)


def _all_states_from_domain(ts: SymbolicTS) -> List[Dict[str, int]]:
    """Generate all states from domain specification."""
    all_states = [{}]
    for var in ts.variables:
        new_states = []
        for s in all_states:
            for val in ts.domain[var]:
                ns = dict(s)
                ns[var] = val
                new_states.append(ns)
        all_states = new_states
    return all_states


# ============================================================
# Craig Interpolation (Simplified)
# ============================================================

@dataclass
class Interpolant:
    """A Craig interpolant -- a predicate over state variables.

    Over-approximates states reachable from Init in k steps.
    Under-approximates complement of backward-reachable from Bad.
    """
    predicate: Callable[[Dict[str, int]], bool]
    name: str
    depth: int  # BMC depth this was derived from


def compute_interpolant(ts: SymbolicTS, depth: int) -> Optional[Interpolant]:
    """Compute Craig interpolant for depth-k BMC.

    When BMC is UNSAT at depth k, we compute an over-approximation
    of the reachable states that excludes bad states.

    Method: Forward reachability from Init up to depth k, then
    strengthen by removing states that can reach Bad.
    """
    if ts.domain is None:
        return None

    all_states = _all_states_from_domain(ts)

    # Forward: states reachable from Init in <= depth steps
    forward = set()
    frontier = set()
    for s in all_states:
        if ts.init_pred(s):
            key = _state_key(s)
            forward.add(key)
            frontier.add(key)

    state_by_key = {_state_key(s): s for s in all_states}

    for _ in range(depth):
        next_frontier = set()
        for key in frontier:
            s = state_by_key[key]
            for t in all_states:
                if ts.trans_pred(s, t):
                    tkey = _state_key(t)
                    if tkey not in forward:
                        forward.add(tkey)
                        next_frontier.add(tkey)
        frontier = next_frontier

    # Backward: states that can reach Bad
    backward = set()
    for s in all_states:
        if ts.bad_pred(s):
            backward.add(_state_key(s))

    changed = True
    while changed:
        changed = False
        for s in all_states:
            skey = _state_key(s)
            if skey in backward:
                continue
            for t in all_states:
                if ts.trans_pred(s, t):
                    tkey = _state_key(t)
                    if tkey in backward:
                        backward.add(skey)
                        changed = True
                        break

    # Interpolant: forward-reachable AND NOT backward-reachable-to-bad
    # This is an over-approximation of reachable that excludes bad-reaching states
    interp_keys = forward - backward

    # If forward and backward overlap, BMC should have found cex
    # Build predicate
    def interp_pred(s):
        return _state_key(s) in interp_keys

    return Interpolant(
        predicate=interp_pred,
        name=f"I_{depth}",
        depth=depth
    )


def compute_interpolant_sequence(ts: SymbolicTS, max_depth: int
                                  ) -> List[Interpolant]:
    """Compute interpolant sequence I_0, I_1, ..., I_k.

    I_i over-approximates states reachable in exactly i steps from Init.
    """
    if ts.domain is None:
        return []

    all_states = _all_states_from_domain(ts)
    state_by_key = {_state_key(s): s for s in all_states}

    # Backward-reachable from bad (for strengthening)
    bad_reach = set()
    for s in all_states:
        if ts.bad_pred(s):
            bad_reach.add(_state_key(s))
    changed = True
    while changed:
        changed = False
        for s in all_states:
            skey = _state_key(s)
            if skey in bad_reach:
                continue
            for t in all_states:
                if ts.trans_pred(s, t) and _state_key(t) in bad_reach:
                    bad_reach.add(skey)
                    changed = True
                    break

    interpolants = []
    current_layer = set()
    for s in all_states:
        if ts.init_pred(s):
            current_layer.add(_state_key(s))

    for d in range(max_depth + 1):
        # Interpolant for depth d: current layer minus bad-reaching
        safe_layer = current_layer - bad_reach
        layer_copy = frozenset(safe_layer)

        def make_pred(keys):
            return lambda s: _state_key(s) in keys

        interpolants.append(Interpolant(
            predicate=make_pred(layer_copy),
            name=f"I_{d}",
            depth=d
        ))

        # Compute next layer
        next_layer = set()
        for key in current_layer:
            s = state_by_key.get(key)
            if s is None:
                continue
            for t in all_states:
                if ts.trans_pred(s, t):
                    next_layer.add(_state_key(t))
        current_layer = current_layer | next_layer

    return interpolants


# ============================================================
# Interpolation-Based Model Checking
# ============================================================

class IMCVerdict(Enum):
    SAFE = auto()        # property holds (invariant found)
    UNSAFE = auto()      # property violated (counterexample)
    UNKNOWN = auto()     # max depth reached


@dataclass
class IMCResult:
    """Result of interpolation-based model checking."""
    verdict: IMCVerdict
    depth: int                       # final depth
    invariant: Optional[Interpolant] = None  # inductive invariant (if SAFE)
    counterexample: Optional[List[Dict[str, int]]] = None
    iterations: int = 0
    history: List[str] = field(default_factory=list)


def interpolation_model_check(ts: SymbolicTS,
                               max_depth: int = 50,
                               max_states: int = 10000) -> IMCResult:
    """McMillan's interpolation-based model checking.

    Algorithm:
    1. For k = 0, 1, 2, ...:
       a. Run BMC at depth k
       b. If SAT: return UNSAFE with counterexample
       c. If UNSAT: compute interpolant I_k
       d. Check if I_k is inductive:
          - Init => I_k
          - I_k /\\ Trans => I_k'  (I_k is closed under transitions)
          - I_k /\\ Bad = empty
       e. If inductive: return SAFE with I_k as invariant
       f. Otherwise: increment k
    """
    if ts.domain is None:
        raise ValueError("Need domain for interpolation MC")

    all_states = _all_states_from_domain(ts)
    history = []

    for depth in range(max_depth + 1):
        # Step 1: BMC
        bmc = bounded_model_check(ts, depth)
        if not bmc.safe:
            return IMCResult(
                verdict=IMCVerdict.UNSAFE,
                depth=depth,
                counterexample=bmc.counterexample,
                iterations=depth + 1,
                history=history
            )

        # Step 2: Compute interpolant
        interp = compute_interpolant(ts, depth)
        if interp is None:
            history.append(f"Depth {depth}: could not compute interpolant")
            continue

        # Step 3: Check inductiveness
        is_inductive = _check_inductive(ts, interp, all_states)
        history.append(f"Depth {depth}: interpolant computed, "
                       f"inductive={is_inductive}")

        if is_inductive:
            return IMCResult(
                verdict=IMCVerdict.SAFE,
                depth=depth,
                invariant=interp,
                iterations=depth + 1,
                history=history
            )

    return IMCResult(
        verdict=IMCVerdict.UNKNOWN,
        depth=max_depth,
        iterations=max_depth + 1,
        history=history
    )


def _check_inductive(ts: SymbolicTS, interp: Interpolant,
                      all_states: List[Dict[str, int]]) -> bool:
    """Check if interpolant is inductive.

    Conditions:
    1. Init => I (all initial states satisfy interpolant)
    2. I /\\ Trans => I' (interpolant is closed under transitions)
    3. I /\\ Bad = false (interpolant excludes bad states)
    """
    # Check Init => I
    for s in all_states:
        if ts.init_pred(s) and not interp.predicate(s):
            return False

    # Check I /\\ Bad = false
    for s in all_states:
        if interp.predicate(s) and ts.bad_pred(s):
            return False

    # Check I /\\ Trans => I'
    for s in all_states:
        if not interp.predicate(s):
            continue
        for t in all_states:
            if ts.trans_pred(s, t) and not interp.predicate(t):
                return False

    return True


# ============================================================
# Incremental Interpolation (k-step interpolation sequence)
# ============================================================

def incremental_interpolation_mc(ts: SymbolicTS,
                                  max_depth: int = 50) -> IMCResult:
    """Incremental interpolation-based model checking.

    Uses interpolant sequences: at depth k, compute I_0, I_1, ..., I_k
    where I_i over-approximates states reachable in i steps.

    Fixpoint check: if I_k subset I_{k-1} union ... union I_0,
    then the over-approximation has stabilized => SAFE.
    """
    if ts.domain is None:
        raise ValueError("Need domain for incremental IMC")

    all_states = _all_states_from_domain(ts)
    history = []

    for depth in range(max_depth + 1):
        # BMC check
        bmc = bounded_model_check(ts, depth)
        if not bmc.safe:
            return IMCResult(
                verdict=IMCVerdict.UNSAFE,
                depth=depth,
                counterexample=bmc.counterexample,
                iterations=depth + 1,
                history=history
            )

        # Compute interpolant sequence
        interps = compute_interpolant_sequence(ts, depth)

        if not interps:
            continue

        # Check fixpoint: does the cumulative interpolant stabilize?
        cumulative = set()
        stabilized = False
        for i, interp in enumerate(interps):
            layer = set()
            for s in all_states:
                if interp.predicate(s):
                    layer.add(_state_key(s))

            if i > 0 and layer.issubset(cumulative):
                stabilized = True
                break
            cumulative.update(layer)

        history.append(f"Depth {depth}: {len(interps)} interpolants, "
                       f"stabilized={stabilized}")

        if stabilized:
            # Build combined invariant
            combined_keys = frozenset(cumulative)
            combined = Interpolant(
                predicate=lambda s, ck=combined_keys: _state_key(s) in ck,
                name=f"I_combined_{depth}",
                depth=depth
            )

            # Verify inductiveness
            if _check_inductive(ts, combined, all_states):
                return IMCResult(
                    verdict=IMCVerdict.SAFE,
                    depth=depth,
                    invariant=combined,
                    iterations=depth + 1,
                    history=history
                )

    return IMCResult(
        verdict=IMCVerdict.UNKNOWN,
        depth=max_depth,
        iterations=max_depth + 1,
        history=history
    )


# ============================================================
# Helper: TS from Kripke Structure
# ============================================================

def kripke_to_ts(ks: KripkeStructure, bad_prop: str) -> SymbolicTS:
    """Convert Kripke structure to transition system for IMC."""
    state_list = sorted(ks.states)

    def init_pred(s):
        return s.get('state', -1) in ks.initial

    def trans_pred(s, t):
        src = s.get('state', -1)
        dst = t.get('state', -1)
        if src not in ks.transitions:
            return False
        return dst in ks.transitions[src]

    def bad_pred(s):
        sid = s.get('state', -1)
        return bad_prop in ks.labeling.get(sid, set())

    return SymbolicTS(
        variables=['state'],
        init_pred=init_pred,
        trans_pred=trans_pred,
        bad_pred=bad_pred,
        domain={'state': state_list}
    )


def concrete_to_ts(sys: ConcreteSystem, bad_prop: str) -> SymbolicTS:
    """Convert ConcreteSystem to SymbolicTS for IMC.

    Enumerates states and builds explicit transition table.
    """
    # Enumerate states
    states_list = []
    state_set = set()
    queue = deque()

    for init in sys.init_states:
        key = _state_key(init)
        if key not in state_set:
            state_set.add(key)
            states_list.append(init)
            queue.append(init)

    while queue and len(states_list) < 10000:
        s = queue.popleft()
        for t in sys.transition_fn(s):
            key = _state_key(t)
            if key not in state_set:
                state_set.add(key)
                states_list.append(t)
                queue.append(t)

    # Build transition table
    trans_table = set()
    for s in states_list:
        for t in sys.transition_fn(s):
            if _state_key(t) in state_set:
                trans_table.add((_state_key(s), _state_key(t)))

    init_keys = {_state_key(s) for s in sys.init_states}
    bad_keys = {_state_key(s) for s in states_list if bad_prop in sys.prop_fn(s)}

    # Assign integer IDs
    key_to_id = {_state_key(s): i for i, s in enumerate(states_list)}

    def init_pred(s):
        return _state_key(s) in init_keys

    def trans_pred(s, t):
        return (_state_key(s), _state_key(t)) in trans_table

    def bad_pred(s):
        return _state_key(s) in bad_keys

    # Build domain from observed values
    domain = {}
    for var in sys.variables:
        vals = sorted({s[var] for s in states_list if var in s})
        domain[var] = vals

    return SymbolicTS(
        variables=sys.variables,
        init_pred=init_pred,
        trans_pred=trans_pred,
        bad_pred=bad_pred,
        domain=domain
    )


# ============================================================
# Example Systems
# ============================================================

def make_safe_counter(bound: int) -> SymbolicTS:
    """Counter 0..bound-1 that wraps. Bad: counter >= bound.
    Safe because counter wraps at bound-1 -> 0 (never reaches bound).
    """
    vals = list(range(bound + 1))  # include bound to represent bad state
    return SymbolicTS(
        variables=['x'],
        init_pred=lambda s: s['x'] == 0,
        trans_pred=lambda s, t: (
            # Increment if below bound-1
            (s['x'] < bound - 1 and t['x'] == s['x'] + 1) or
            # Wrap at bound-1 -> 0
            (s['x'] == bound - 1 and t['x'] == 0)
        ),
        bad_pred=lambda s: s['x'] >= bound,
        domain={'x': vals}
    )


def make_unsafe_counter(bound: int) -> SymbolicTS:
    """Counter that can exceed bound. Bad: counter >= bound."""
    vals = list(range(bound + 2))
    return SymbolicTS(
        variables=['x'],
        init_pred=lambda s: s['x'] == 0,
        trans_pred=lambda s, t: t['x'] == s['x'] + 1 and s['x'] <= bound,
        bad_pred=lambda s: s['x'] >= bound,
        domain={'x': vals}
    )


def make_mutual_exclusion() -> SymbolicTS:
    """Two-process mutex (buggy -- both can enter critical).
    States: p1, p2 in {0=idle, 1=trying, 2=critical}
    Bad: p1==2 and p2==2
    """
    vals = [0, 1, 2]
    return SymbolicTS(
        variables=['p1', 'p2'],
        init_pred=lambda s: s['p1'] == 0 and s['p2'] == 0,
        trans_pred=lambda s, t: (
            # p1 steps, p2 stays
            (t['p2'] == s['p2'] and (
                (s['p1'] == 0 and t['p1'] == 1) or
                (s['p1'] == 1 and t['p1'] == 2) or
                (s['p1'] == 2 and t['p1'] == 0)
            )) or
            # p2 steps, p1 stays
            (t['p1'] == s['p1'] and (
                (s['p2'] == 0 and t['p2'] == 1) or
                (s['p2'] == 1 and t['p2'] == 2) or
                (s['p2'] == 2 and t['p2'] == 0)
            ))
        ),
        bad_pred=lambda s: s['p1'] == 2 and s['p2'] == 2,
        domain={'p1': vals, 'p2': vals}
    )


def make_token_ring(n: int) -> SymbolicTS:
    """Token ring with n positions. Bad: token at position n-1.
    Actually safe -- token cycles and visits all positions.
    """
    vals = list(range(n))
    return SymbolicTS(
        variables=['pos'],
        init_pred=lambda s: s['pos'] == 0,
        trans_pred=lambda s, t: t['pos'] == (s['pos'] + 1) % n,
        bad_pred=lambda s: False,  # no bad state
        domain={'pos': vals}
    )


def make_producer_consumer() -> SymbolicTS:
    """Producer-consumer with bounded buffer.
    Buffer size 0..3. Bad: buffer > 3 (overflow).
    Safe if producer only adds when buffer < 3.
    """
    vals = list(range(5))  # 0..4
    return SymbolicTS(
        variables=['buf'],
        init_pred=lambda s: s['buf'] == 0,
        trans_pred=lambda s, t: (
            (s['buf'] < 3 and t['buf'] == s['buf'] + 1) or  # produce
            (s['buf'] > 0 and t['buf'] == s['buf'] - 1)     # consume
        ),
        bad_pred=lambda s: s['buf'] > 3,
        domain={'buf': vals}
    )


def make_two_phase_commit() -> SymbolicTS:
    """Simplified two-phase commit.
    Coordinator: 0=init, 1=wait, 2=commit, 3=abort
    Participant: 0=init, 1=voted, 2=committed, 3=aborted
    Bad: coordinator committed but participant aborted (or vice versa)
    """
    vals = [0, 1, 2, 3]
    return SymbolicTS(
        variables=['coord', 'part'],
        init_pred=lambda s: s['coord'] == 0 and s['part'] == 0,
        trans_pred=lambda s, t: (
            # Phase 1: coord sends prepare
            (s['coord'] == 0 and t['coord'] == 1 and t['part'] == s['part']) or
            # Participant votes yes
            (s['coord'] == 1 and s['part'] == 0 and t['coord'] == 1 and t['part'] == 1) or
            # Phase 2: coord commits (if participant voted)
            (s['coord'] == 1 and s['part'] == 1 and t['coord'] == 2 and t['part'] == 1) or
            # Participant commits
            (s['coord'] == 2 and s['part'] == 1 and t['coord'] == 2 and t['part'] == 2) or
            # Abort paths
            (s['coord'] == 1 and s['part'] == 0 and t['coord'] == 3 and t['part'] == 0) or
            (s['coord'] == 3 and s['part'] == 0 and t['coord'] == 3 and t['part'] == 3)
        ),
        bad_pred=lambda s: (
            (s['coord'] == 2 and s['part'] == 3) or
            (s['coord'] == 3 and s['part'] == 2)
        ),
        domain={'coord': vals, 'part': vals}
    )


# ============================================================
# Analysis Helpers
# ============================================================

def verify_imc_result(ts: SymbolicTS, result: IMCResult) -> Dict:
    """Verify an IMC result."""
    info = {
        'verdict': result.verdict.name,
        'depth': result.depth,
        'iterations': result.iterations,
    }

    if result.invariant:
        all_states = _all_states_from_domain(ts)
        inductive = _check_inductive(ts, result.invariant, all_states)
        info['invariant_inductive'] = inductive

    if result.counterexample:
        info['cex_length'] = len(result.counterexample)
        # Verify counterexample
        trace = result.counterexample
        info['cex_starts_init'] = ts.init_pred(trace[0])
        info['cex_ends_bad'] = ts.bad_pred(trace[-1])
        info['cex_valid_transitions'] = all(
            ts.trans_pred(trace[i], trace[i+1])
            for i in range(len(trace) - 1)
        )

    return info


def compare_imc_methods(ts: SymbolicTS, max_depth: int = 20) -> Dict:
    """Compare standard and incremental interpolation MC."""
    r1 = interpolation_model_check(ts, max_depth)
    r2 = incremental_interpolation_mc(ts, max_depth)

    return {
        'standard': {
            'verdict': r1.verdict.name,
            'depth': r1.depth,
            'iterations': r1.iterations,
        },
        'incremental': {
            'verdict': r2.verdict.name,
            'depth': r2.depth,
            'iterations': r2.iterations,
        },
        'agree': r1.verdict == r2.verdict,
    }


def imc_statistics(result: IMCResult) -> Dict:
    """Extract statistics from an IMC result."""
    return {
        'verdict': result.verdict.name,
        'depth': result.depth,
        'iterations': result.iterations,
        'has_invariant': result.invariant is not None,
        'has_cex': result.counterexample is not None,
        'history_length': len(result.history),
        'history': result.history,
    }


def batch_verify(systems: List[Tuple[str, SymbolicTS]],
                  max_depth: int = 20) -> List[Dict]:
    """Verify multiple systems."""
    results = []
    for name, ts in systems:
        r = interpolation_model_check(ts, max_depth)
        results.append({
            'name': name,
            'verdict': r.verdict.name,
            'depth': r.depth,
            'iterations': r.iterations,
        })
    return results
