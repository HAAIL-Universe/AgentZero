"""V196: Strategy Simplification -- reduce controller size via simulation relations.

Composes V186 (reactive synthesis / MealyMachine) + V192 (strategy composition / minimize_mealy)
      + V195 (distributed synthesis / DistributedController).

Given a synthesized Mealy machine controller, simplify it:
1. Forward simulation: find simulation relation (preorder on states)
2. Backward simulation: dual notion, coarser quotient possible
3. Simulation quotient: merge simulation-equivalent states
4. Don't-care optimization: exploit undefined transitions to merge more states
5. Input reduction: identify and remove irrelevant input variables
6. Output canonicalization: normalize output encoding
7. Distributed simplification: simplify each local controller in a distributed system
8. Simplification pipeline: chain multiple simplifications for maximum reduction
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, FrozenSet
from collections import deque
from itertools import product as cartesian_product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from V186_reactive_synthesis.reactive_synthesis import (
    MealyMachine, SynthesisResult, SynthesisVerdict,
    synthesize, verify_controller, controller_statistics,
)
from V192_strategy_composition.strategy_composition import (
    minimize_mealy, mealy_equivalence, CompositionResult,
    _all_valuations, _s,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SimplificationResult:
    """Result of a simplification operation."""
    original: MealyMachine
    simplified: MealyMachine
    method: str
    original_states: int
    simplified_states: int
    original_transitions: int
    simplified_transitions: int
    equivalent: bool  # whether verified equivalent
    details: Dict = field(default_factory=dict)

    @property
    def state_reduction(self) -> float:
        if self.original_states == 0:
            return 0.0
        return 1.0 - self.simplified_states / self.original_states

    @property
    def transition_reduction(self) -> float:
        if self.original_transitions == 0:
            return 0.0
        return 1.0 - self.simplified_transitions / self.original_transitions


@dataclass
class SimulationRelation:
    """A simulation relation between two Mealy machines (or same machine).

    pairs: set of (s1, s2) where s1 simulates s2 (s1 >= s2).
    s1 simulates s2 means: for every input, s2's output matches s1's output,
    and s1's successor simulates s2's successor.
    """
    pairs: Set[Tuple[int, int]]
    machine_a: MealyMachine
    machine_b: MealyMachine  # may be same as machine_a

    def simulates(self, s1: int, s2: int) -> bool:
        """Does s1 simulate s2?"""
        return (s1, s2) in self.pairs

    def equivalent_pairs(self) -> Set[Tuple[int, int]]:
        """Return pairs where s1 simulates s2 AND s2 simulates s1."""
        return {(a, b) for (a, b) in self.pairs if (b, a) in self.pairs}

    def equivalence_classes(self) -> List[Set[int]]:
        """Compute equivalence classes from mutual simulation."""
        eq_pairs = self.equivalent_pairs()
        if not eq_pairs:
            return [{s} for s in self.machine_a.states]

        # Union-find
        parent = {}
        states = set()
        for a, b in eq_pairs:
            states.add(a)
            states.add(b)
        for s in self.machine_a.states:
            states.add(s)
        for s in states:
            parent[s] = s

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for a, b in eq_pairs:
            union(a, b)

        classes = {}
        for s in states:
            r = find(s)
            if r not in classes:
                classes[r] = set()
            classes[r].add(s)
        return list(classes.values())


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def compute_forward_simulation(machine: MealyMachine) -> SimulationRelation:
    """Compute the largest forward simulation on a Mealy machine.

    s1 forward-simulates s2 if for all inputs i:
    - output(s1, i) == output(s2, i) (or s2 has no transition)
    - next(s1, i) forward-simulates next(s2, i)

    Uses greatest fixpoint iteration: start with all pairs, remove violators.
    """
    states = list(machine.states)
    all_inputs = list(_all_valuations(_s(machine.inputs)))

    # Initial: all pairs (s1, s2) where output signatures match
    pairs = set()
    for s1 in states:
        for s2 in states:
            compatible = True
            for inp in all_inputs:
                inp_fs = frozenset(inp)
                t1 = machine.transitions.get((s1, inp_fs))
                t2 = machine.transitions.get((s2, inp_fs))
                if t2 is None:
                    continue  # s2 undefined -> any s1 output ok
                if t1 is None:
                    compatible = False
                    break
                if _s(t1[1]) != _s(t2[1]):
                    compatible = False
                    break
            if compatible:
                pairs.add((s1, s2))

    # Greatest fixpoint: remove pairs that violate successor condition
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for (s1, s2) in pairs:
            for inp in all_inputs:
                inp_fs = frozenset(inp)
                t2 = machine.transitions.get((s2, inp_fs))
                if t2 is None:
                    continue
                t1 = machine.transitions.get((s1, inp_fs))
                if t1 is None:
                    to_remove.add((s1, s2))
                    break
                ns1, _ = t1
                ns2, _ = t2
                if (ns1, ns2) not in pairs:
                    to_remove.add((s1, s2))
                    break
        if to_remove:
            pairs -= to_remove
            changed = True

    return SimulationRelation(pairs=pairs, machine_a=machine, machine_b=machine)


def compute_backward_simulation(machine: MealyMachine) -> SimulationRelation:
    """Compute the largest backward simulation on a Mealy machine.

    s1 backward-simulates s2 if for all inputs i and for every predecessor p2
    of s2 under input i, there exists a predecessor p1 of s1 under input i
    with same output and p1 backward-simulates p2.

    Uses greatest fixpoint on the reverse transition relation.
    """
    states = list(machine.states)
    all_inputs = list(_all_valuations(_s(machine.inputs)))

    # Build reverse transitions: (next_state, input) -> list of (prev_state, output)
    reverse = {}
    for (s, inp_fs), (ns, out) in machine.transitions.items():
        key = (ns, inp_fs)
        if key not in reverse:
            reverse[key] = []
        reverse[key].append((s, out))

    # Initial: all pairs with compatible output signatures
    pairs = set()
    for s1 in states:
        for s2 in states:
            compatible = True
            for inp in all_inputs:
                inp_fs = frozenset(inp)
                t1 = machine.transitions.get((s1, inp_fs))
                t2 = machine.transitions.get((s2, inp_fs))
                if t2 is None:
                    continue
                if t1 is None:
                    compatible = False
                    break
                if _s(t1[1]) != _s(t2[1]):
                    compatible = False
                    break
            if compatible:
                pairs.add((s1, s2))

    # Greatest fixpoint: check backward condition
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for (s1, s2) in pairs:
            for inp in all_inputs:
                inp_fs = frozenset(inp)
                preds2 = reverse.get((s2, inp_fs), [])
                for (p2, out2) in preds2:
                    # Need some predecessor p1 of s1 with same output and (p1,p2) in pairs
                    preds1 = reverse.get((s1, inp_fs), [])
                    found = False
                    for (p1, out1) in preds1:
                        if _s(out1) == _s(out2) and (p1, p2) in pairs:
                            found = True
                            break
                    if not found:
                        to_remove.add((s1, s2))
                        break
                if (s1, s2) in to_remove:
                    break
        if to_remove:
            pairs -= to_remove
            changed = True

    return SimulationRelation(pairs=pairs, machine_a=machine, machine_b=machine)


# ---------------------------------------------------------------------------
# Simulation quotient
# ---------------------------------------------------------------------------

def simulation_quotient(machine: MealyMachine, sim: SimulationRelation) -> MealyMachine:
    """Build quotient machine by merging simulation-equivalent states.

    Two states are simulation-equivalent if they mutually simulate each other.
    The quotient picks one representative per equivalence class.
    """
    eq_classes = sim.equivalence_classes()

    # Map each state to its representative (min element in class)
    state_map = {}
    for cls in eq_classes:
        rep = min(cls)
        for s in cls:
            state_map[s] = rep

    # For states not in any eq_class pair, map to themselves
    for s in machine.states:
        if s not in state_map:
            state_map[s] = s

    new_states = set(state_map.values())
    new_initial = state_map[machine.initial]
    new_trans = {}

    for (s, inp_fs), (ns, out) in machine.transitions.items():
        new_s = state_map[s]
        new_ns = state_map[ns]
        key = (new_s, inp_fs)
        if key not in new_trans:
            new_trans[key] = (new_ns, out)

    # Renumber states to 0..n-1
    state_list = sorted(new_states)
    renumber = {s: i for i, s in enumerate(state_list)}

    final_trans = {}
    for (s, inp_fs), (ns, out) in new_trans.items():
        final_trans[(renumber[s], inp_fs)] = (renumber[ns], out)

    return MealyMachine(
        states=set(range(len(state_list))),
        initial=renumber[new_initial],
        inputs=machine.inputs,
        outputs=machine.outputs,
        transitions=final_trans,
    )


# ---------------------------------------------------------------------------
# Don't-care optimization
# ---------------------------------------------------------------------------

def dont_care_merge(machine: MealyMachine) -> MealyMachine:
    """Exploit undefined transitions (don't-cares) to merge more states.

    If two states differ only on inputs where one or both are undefined,
    they can be merged by filling in the don't-care with the defined value.
    Then apply standard minimization.
    """
    states = list(machine.states)
    all_inputs = list(_all_valuations(_s(machine.inputs)))

    # Build extended transition table filling don't-cares
    # For each pair, check if they're compatible (agree on all defined inputs)
    def compatible(s1, s2):
        for inp in all_inputs:
            inp_fs = frozenset(inp)
            t1 = machine.transitions.get((s1, inp_fs))
            t2 = machine.transitions.get((s2, inp_fs))
            if t1 is not None and t2 is not None:
                if _s(t1[1]) != _s(t2[1]):
                    return False
            # Check successor compatibility would require fixpoint,
            # but for merging we just check output agreement
        return True

    # Find compatible groups using greedy union-find
    parent = {s: s for s in states}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Try to merge compatible pairs
    for i, s1 in enumerate(states):
        for s2 in states[i + 1:]:
            if find(s1) != find(s2) and compatible(s1, s2):
                # Also check successor compatibility in current groups
                can_merge = True
                for inp in all_inputs:
                    inp_fs = frozenset(inp)
                    t1 = machine.transitions.get((s1, inp_fs))
                    t2 = machine.transitions.get((s2, inp_fs))
                    if t1 is not None and t2 is not None:
                        ns1, _ = t1
                        ns2, _ = t2
                        if find(ns1) != find(ns2):
                            # Would need to also merge successors -- skip for now
                            # unless they're already in the same group
                            can_merge = False
                            break
                if can_merge:
                    union(s1, s2)

    # Build merged machine
    groups = {}
    for s in states:
        r = find(s)
        if r not in groups:
            groups[r] = []
        groups[r].append(s)

    # Renumber
    reps = sorted(groups.keys())
    renumber = {r: i for i, r in enumerate(reps)}
    state_to_new = {s: renumber[find(s)] for s in states}

    new_trans = {}
    for (s, inp_fs), (ns, out) in machine.transitions.items():
        key = (state_to_new[s], inp_fs)
        if key not in new_trans:
            new_trans[key] = (state_to_new[ns], out)

    result = MealyMachine(
        states=set(range(len(reps))),
        initial=state_to_new[machine.initial],
        inputs=machine.inputs,
        outputs=machine.outputs,
        transitions=new_trans,
    )

    # Apply standard minimization on the filled machine
    return minimize_mealy(result)


# ---------------------------------------------------------------------------
# Input reduction
# ---------------------------------------------------------------------------

def find_irrelevant_inputs(machine: MealyMachine) -> Set[str]:
    """Find input variables that don't affect behavior.

    An input variable v is irrelevant if for every state s and every
    input valuation i, flipping v in i doesn't change the output or
    the (equivalence class of) successor state.
    """
    inputs = sorted(_s(machine.inputs))
    if len(inputs) <= 1:
        return set()

    # First minimize to canonical form
    min_machine = minimize_mealy(machine)
    all_inp_vals = list(_all_valuations(_s(min_machine.inputs)))

    irrelevant = set()
    for var in inputs:
        is_irrelevant = True
        for s in min_machine.states:
            for inp in all_inp_vals:
                inp_fs = frozenset(inp)
                # Flip var
                if var in inp:
                    flipped = frozenset(inp - {var})
                else:
                    flipped = frozenset(inp | {var})

                t1 = min_machine.transitions.get((s, inp_fs))
                t2 = min_machine.transitions.get((s, flipped))

                if t1 is None and t2 is None:
                    continue
                if t1 is None or t2 is None:
                    is_irrelevant = False
                    break
                if _s(t1[1]) != _s(t2[1]) or t1[0] != t2[0]:
                    is_irrelevant = False
                    break
            if not is_irrelevant:
                break
        if is_irrelevant:
            irrelevant.add(var)

    return irrelevant


def reduce_inputs(machine: MealyMachine, remove_vars: Set[str] = None) -> MealyMachine:
    """Remove irrelevant input variables from a Mealy machine.

    If remove_vars is None, automatically detects irrelevant inputs.
    Projects transitions onto the reduced input alphabet.
    """
    if remove_vars is None:
        remove_vars = find_irrelevant_inputs(machine)

    if not remove_vars:
        return machine

    new_inputs = _s(machine.inputs) - remove_vars
    new_trans = {}

    for (s, inp_fs), (ns, out) in machine.transitions.items():
        # Project input to remaining variables
        projected = frozenset(_s(inp_fs) - remove_vars)
        key = (s, projected)
        if key not in new_trans:
            new_trans[key] = (ns, out)

    return MealyMachine(
        states=set(machine.states),
        initial=machine.initial,
        inputs=new_inputs,
        outputs=machine.outputs,
        transitions=new_trans,
    )


# ---------------------------------------------------------------------------
# Output canonicalization
# ---------------------------------------------------------------------------

def canonicalize_outputs(machine: MealyMachine) -> MealyMachine:
    """Remove output variables that are always the same value.

    If an output variable is always true (or always false) across all
    transitions, remove it from the output set and transitions.
    Returns a machine with only non-constant outputs.
    """
    if not machine.transitions:
        return machine

    always_true = set(_s(machine.outputs))
    always_false = set(_s(machine.outputs))

    for (_, _), (_, out) in machine.transitions.items():
        out_s = _s(out)
        always_true &= out_s
        always_false -= out_s

    constant_vars = always_true | always_false
    if not constant_vars:
        return machine

    remaining_outputs = _s(machine.outputs) - constant_vars
    new_trans = {}
    for (s, inp_fs), (ns, out) in machine.transitions.items():
        new_out = frozenset(_s(out) - constant_vars)
        new_trans[(s, inp_fs)] = (ns, new_out)

    return MealyMachine(
        states=set(machine.states),
        initial=machine.initial,
        inputs=machine.inputs,
        outputs=remaining_outputs,
        transitions=new_trans,
    )


# ---------------------------------------------------------------------------
# Unreachable state removal
# ---------------------------------------------------------------------------

def remove_unreachable(machine: MealyMachine) -> MealyMachine:
    """Remove states not reachable from the initial state."""
    reachable = set()
    queue = deque([machine.initial])
    while queue:
        s = queue.popleft()
        if s in reachable:
            continue
        reachable.add(s)
        for (src, inp_fs), (ns, _) in machine.transitions.items():
            if src == s and ns not in reachable:
                queue.append(ns)

    if reachable == machine.states:
        return machine

    new_trans = {}
    for (s, inp_fs), (ns, out) in machine.transitions.items():
        if s in reachable:
            new_trans[(s, inp_fs)] = (ns, out)

    # Renumber
    state_list = sorted(reachable)
    renumber = {s: i for i, s in enumerate(state_list)}

    final_trans = {}
    for (s, inp_fs), (ns, out) in new_trans.items():
        final_trans[(renumber[s], inp_fs)] = (renumber[ns], out)

    return MealyMachine(
        states=set(range(len(state_list))),
        initial=renumber[machine.initial],
        inputs=machine.inputs,
        outputs=machine.outputs,
        transitions=final_trans,
    )


# ---------------------------------------------------------------------------
# Signature-based state merging
# ---------------------------------------------------------------------------

def signature_merge(machine: MealyMachine, depth: int = 2) -> MealyMachine:
    """Merge states with identical k-step output signatures.

    Two states are k-equivalent if they produce identical outputs for all
    input sequences of length <= k. This is a coarser relation than bisimulation
    for small k, yielding more aggressive (but approximate) merging.

    For k = infinity this converges to bisimulation (same as minimize_mealy).
    For finite k, it's an over-approximation that may lose information.

    We use this as a fast pre-pass before full minimization.
    """
    all_inputs = sorted(_all_valuations(_s(machine.inputs)), key=lambda x: tuple(sorted(x)))
    states = list(machine.states)

    # Level 0: group by immediate output signature
    sig = {}
    for s in states:
        s_sig = []
        for inp in all_inputs:
            t = machine.transitions.get((s, frozenset(inp)))
            s_sig.append(frozenset(t[1]) if t is not None else None)
        sig[s] = tuple(s_sig)

    groups = {}
    for s, s_sig in sig.items():
        if s_sig not in groups:
            groups[s_sig] = []
        groups[s_sig].append(s)
    partitions = list(groups.values())

    # Refine for 'depth' levels
    for _ in range(depth):
        state_to_part = {}
        for pi, part in enumerate(partitions):
            for s in part:
                state_to_part[s] = pi

        new_partitions = []
        for part in partitions:
            if len(part) <= 1:
                new_partitions.append(part)
                continue

            subgroups = {}
            for s in part:
                s_sig = []
                for inp in all_inputs:
                    t = machine.transitions.get((s, frozenset(inp)))
                    s_sig.append(state_to_part.get(t[0], -1) if t is not None else -2)
                key = tuple(s_sig)
                if key not in subgroups:
                    subgroups[key] = []
                subgroups[key].append(s)
            new_partitions.extend(subgroups.values())
        partitions = new_partitions

    # Build quotient
    state_to_part = {}
    for pi, part in enumerate(partitions):
        for s in part:
            state_to_part[s] = pi

    new_initial = state_to_part[machine.initial]
    new_trans = {}
    for pi, part in enumerate(partitions):
        rep = part[0]
        for inp in all_inputs:
            t = machine.transitions.get((rep, frozenset(inp)))
            if t is not None:
                ns, out = t
                new_trans[(pi, frozenset(inp))] = (state_to_part[ns], out)

    return MealyMachine(
        states=set(range(len(partitions))),
        initial=new_initial,
        inputs=machine.inputs,
        outputs=machine.outputs,
        transitions=new_trans,
    )


# ---------------------------------------------------------------------------
# Full simplification pipeline
# ---------------------------------------------------------------------------

def simplify(machine: MealyMachine, method: str = "full") -> SimplificationResult:
    """Simplify a Mealy machine controller.

    Methods:
    - "minimize": standard partition-refinement bisimulation minimization
    - "simulation": forward simulation quotient (can be coarser than bisimulation)
    - "dont_care": exploit undefined transitions then minimize
    - "input_reduce": remove irrelevant input variables
    - "full": chain all simplifications for maximum reduction
    """
    orig_states = len(machine.states)
    orig_trans = len(machine.transitions)

    if method == "minimize":
        result = minimize_mealy(machine)
    elif method == "simulation":
        sim = compute_forward_simulation(machine)
        result = simulation_quotient(machine, sim)
        result = minimize_mealy(result)  # clean up
    elif method == "dont_care":
        result = dont_care_merge(machine)
    elif method == "input_reduce":
        result = reduce_inputs(machine)
        result = minimize_mealy(result)
    elif method == "full":
        result = full_simplification_pipeline(machine)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Verify equivalence
    eq, _ = mealy_equivalence(machine, result, max_depth=50)

    return SimplificationResult(
        original=machine,
        simplified=result,
        method=method,
        original_states=orig_states,
        simplified_states=len(result.states),
        original_transitions=orig_trans,
        simplified_transitions=len(result.transitions),
        equivalent=eq,
        details={"method": method},
    )


def full_simplification_pipeline(machine: MealyMachine) -> MealyMachine:
    """Apply all simplification techniques in optimal order.

    Order: unreachable removal -> input reduction -> don't-care merge
         -> simulation quotient -> minimize -> output canonicalize
    """
    # 1. Remove unreachable states
    m = remove_unreachable(machine)

    # 2. Input reduction (removes irrelevant variables)
    m = reduce_inputs(m)

    # 3. Don't-care merge (fill undefined transitions to enable merging)
    m = dont_care_merge(m)

    # 4. Forward simulation quotient
    sim = compute_forward_simulation(m)
    m = simulation_quotient(m, sim)

    # 5. Standard bisimulation minimization (cleanup)
    m = minimize_mealy(m)

    return m


# ---------------------------------------------------------------------------
# Strategy comparison and analysis
# ---------------------------------------------------------------------------

def compare_simplification_methods(machine: MealyMachine) -> Dict:
    """Compare all simplification methods on the same machine.

    Returns dict with method names -> SimplificationResult.
    """
    methods = ["minimize", "simulation", "dont_care", "input_reduce", "full"]
    results = {}
    for method in methods:
        results[method] = simplify(machine, method)
    return results


def simplification_statistics(result: SimplificationResult) -> Dict:
    """Compute statistics about a simplification result."""
    return {
        "method": result.method,
        "original_states": result.original_states,
        "simplified_states": result.simplified_states,
        "state_reduction": f"{result.state_reduction:.1%}",
        "original_transitions": result.original_transitions,
        "simplified_transitions": result.simplified_transitions,
        "transition_reduction": f"{result.transition_reduction:.1%}",
        "equivalent": result.equivalent,
    }


def simplification_summary(result: SimplificationResult) -> str:
    """Human-readable summary of simplification."""
    stats = simplification_statistics(result)
    lines = [
        f"Strategy Simplification ({stats['method']})",
        f"  States: {stats['original_states']} -> {stats['simplified_states']} ({stats['state_reduction']} reduction)",
        f"  Transitions: {stats['original_transitions']} -> {stats['simplified_transitions']} ({stats['transition_reduction']} reduction)",
        f"  Equivalence verified: {stats['equivalent']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distributed controller simplification
# ---------------------------------------------------------------------------

def simplify_distributed(controllers: Dict[str, MealyMachine],
                         method: str = "full") -> Dict[str, SimplificationResult]:
    """Simplify each local controller in a distributed system.

    Args:
        controllers: mapping of process name -> local MealyMachine
        method: simplification method to use

    Returns:
        mapping of process name -> SimplificationResult
    """
    results = {}
    for name, machine in controllers.items():
        results[name] = simplify(machine, method)
    return results


def distributed_simplification_summary(results: Dict[str, SimplificationResult]) -> str:
    """Summary of distributed controller simplification."""
    lines = ["Distributed Controller Simplification"]
    total_orig = 0
    total_simp = 0
    for name, result in sorted(results.items()):
        total_orig += result.original_states
        total_simp += result.simplified_states
        lines.append(
            f"  {name}: {result.original_states} -> {result.simplified_states} states "
            f"({result.state_reduction:.0%} reduction)"
        )
    if total_orig > 0:
        overall = 1.0 - total_simp / total_orig
        lines.append(f"  Total: {total_orig} -> {total_simp} states ({overall:.0%} reduction)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-machine simulation
# ---------------------------------------------------------------------------

def compute_cross_simulation(machine_a: MealyMachine,
                              machine_b: MealyMachine) -> SimulationRelation:
    """Compute forward simulation from states of machine_a to states of machine_b.

    Requires same I/O signature. s_a simulates s_b if for all inputs:
    - Same outputs
    - Successors maintain simulation
    """
    if _s(machine_a.inputs) != _s(machine_b.inputs):
        return SimulationRelation(set(), machine_a, machine_b)
    if _s(machine_a.outputs) != _s(machine_b.outputs):
        return SimulationRelation(set(), machine_a, machine_b)

    all_inputs = list(_all_valuations(_s(machine_a.inputs)))

    # Initialize: pairs with compatible outputs
    pairs = set()
    for sa in machine_a.states:
        for sb in machine_b.states:
            compat = True
            for inp in all_inputs:
                inp_fs = frozenset(inp)
                ta = machine_a.transitions.get((sa, inp_fs))
                tb = machine_b.transitions.get((sb, inp_fs))
                if tb is None:
                    continue
                if ta is None:
                    compat = False
                    break
                if _s(ta[1]) != _s(tb[1]):
                    compat = False
                    break
            if compat:
                pairs.add((sa, sb))

    # Greatest fixpoint
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for (sa, sb) in pairs:
            for inp in all_inputs:
                inp_fs = frozenset(inp)
                tb = machine_b.transitions.get((sb, inp_fs))
                if tb is None:
                    continue
                ta = machine_a.transitions.get((sa, inp_fs))
                if ta is None:
                    to_remove.add((sa, sb))
                    break
                if (ta[0], tb[0]) not in pairs:
                    to_remove.add((sa, sb))
                    break
        if to_remove:
            pairs -= to_remove
            changed = True

    return SimulationRelation(pairs=pairs, machine_a=machine_a, machine_b=machine_b)


def is_simulated_by(machine_a: MealyMachine, machine_b: MealyMachine) -> bool:
    """Check if machine_b simulates machine_a (machine_b >= machine_a).

    True if initial state of machine_b simulates initial state of machine_a.
    """
    sim = compute_cross_simulation(machine_b, machine_a)
    return sim.simulates(machine_b.initial, machine_a.initial)


# ---------------------------------------------------------------------------
# Mealy machine builders (helpers for testing)
# ---------------------------------------------------------------------------

def make_mealy(states, initial, inputs, outputs, transitions) -> MealyMachine:
    """Convenience builder for MealyMachine.

    transitions: list of (state, input_set, next_state, output_set)
    """
    trans = {}
    for s, inp, ns, out in transitions:
        trans[(s, frozenset(inp))] = (ns, frozenset(out))
    return MealyMachine(
        states=set(states),
        initial=initial,
        inputs=set(inputs),
        outputs=set(outputs),
        transitions=trans,
    )
