"""V192: Strategy Composition -- compose controllers from sub-specifications.

Composes V186 (reactive synthesis) + V187 (GR(1) synthesis).

Operations:
- Parallel composition: product of Mealy machines on disjoint outputs
- Sequential composition: chain outputs -> inputs
- Conjunctive synthesis: And(spec1, spec2)
- Assume-guarantee composition: circular AG reasoning
- Priority composition: resolve output conflicts
- Compositional verification against original spec
- Spec decomposition heuristics
"""

import sys
sys.path.insert(0, 'Z:/AgentZero/A2/work/V186_reactive_synthesis')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V187_gr1_synthesis')

from reactive_synthesis import (
    MealyMachine, SynthesisResult, SynthesisVerdict,
    synthesize, synthesize_assume_guarantee, synthesize_safety,
    verify_controller, controller_statistics,
)
from gr1_synthesis import (
    GR1Game, GR1Strategy, GR1Result, GR1Verdict,
    BoolGR1Spec, gr1_synthesize, gr1_solve,
    make_game, make_safety_game, make_reachability_game,
    strategy_to_mealy, verify_strategy, build_bool_game,
)

sys.path.insert(0, 'Z:/AgentZero/A2/work/V023_ltl_model_checking')
from ltl_model_checker import (
    Atom, LTLTrue, LTLFalse, And as LTLAnd, Or as LTLOr,
    Not as LTLNot, Implies as LTLImplies,
    Next, Finally, Globally, Until, LTLOp,
)


# ---------------------------------------------------------------------------
# Helpers (defined first -- used by everything below)
# ---------------------------------------------------------------------------

def _s(x):
    """Convert to set (handles lists from MealyMachine)."""
    if isinstance(x, set):
        return x
    if isinstance(x, frozenset):
        return set(x)
    return set(x)


def _all_valuations(variables):
    """Generate all 2^|variables| Boolean valuations."""
    vars_list = sorted(variables)
    n = len(vars_list)
    for i in range(1 << n):
        yield frozenset(vars_list[j] for j in range(n) if i & (1 << j))


def _flatten_and(spec):
    """Flatten nested And into a list of conjuncts."""
    if hasattr(spec, 'op') and spec.op == LTLOp.AND:
        result = []
        if spec.left is not None:
            result.extend(_flatten_and(spec.left))
        if spec.right is not None:
            result.extend(_flatten_and(spec.right))
        return result
    return [spec]


def _collect_atoms(spec):
    """Collect all atomic proposition names from an LTL formula."""
    atoms = set()
    if hasattr(spec, 'op') and spec.op == LTLOp.ATOM:
        if spec.name:
            atoms.add(spec.name)
        return atoms
    if hasattr(spec, 'left') and spec.left is not None:
        atoms |= _collect_atoms(spec.left)
    if hasattr(spec, 'right') and spec.right is not None:
        atoms |= _collect_atoms(spec.right)
    return atoms


def _group_by_shared_vars(components):
    """Group components by connected sys variables using union-find."""
    n = len(components)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            _, _, sys_i = components[i]
            _, _, sys_j = components[j]
            if sys_i & sys_j:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    result = []
    for indices in groups.values():
        if len(indices) == 1:
            spec, env_v, sys_v = components[indices[0]]
            result.append((spec, env_v, sys_v))
        else:
            specs = [components[i][0] for i in indices]
            env_v = set()
            sys_v = set()
            for i in indices:
                env_v |= components[i][1]
                sys_v |= components[i][2]
            combined = specs[0]
            for s in specs[1:]:
                combined = LTLAnd(combined, s)
            result.append((combined, env_v, sys_v))

    return result


def _ensure_mealy(obj):
    """Ensure object is a MealyMachine (convert if needed)."""
    return obj


# ---------------------------------------------------------------------------
# Composition Result
# ---------------------------------------------------------------------------

class CompositionResult:
    """Result of a strategy composition operation."""
    __slots__ = ('controller', 'method', 'components', 'conflicts',
                 'n_states', 'verified', 'details')

    def __init__(self, controller, method, components=None, conflicts=None,
                 n_states=0, verified=None, details=None):
        self.controller = controller
        self.method = method
        self.components = components or []
        self.conflicts = conflicts or []
        self.n_states = n_states
        self.verified = verified
        self.details = details or {}


# ---------------------------------------------------------------------------
# Parallel Composition (disjoint outputs)
# ---------------------------------------------------------------------------

def parallel_compose(c1, c2):
    """Compose two Mealy machines with disjoint outputs running in parallel.

    Both controllers observe the same inputs and produce independent outputs.
    The composed machine state is the product (s1, s2).

    Requires: c1.outputs and c2.outputs are disjoint.
    Inputs: union of both input sets.
    """
    c1_out = _s(c1.outputs)
    c2_out = _s(c2.outputs)
    c1_inp = _s(c1.inputs)
    c2_inp = _s(c2.inputs)

    overlap = c1_out & c2_out
    if overlap:
        raise ValueError(f"Outputs not disjoint: {overlap}. Use priority_compose instead.")

    all_inputs = c1_inp | c2_inp
    all_outputs = c1_out | c2_out

    initial = 0
    state_map = {(c1.initial, c2.initial): 0}
    next_id = 1
    transitions = {}
    queue = [(c1.initial, c2.initial)]
    visited = set()

    while queue:
        pair = queue.pop(0)
        if pair in visited:
            continue
        visited.add(pair)
        s1, s2 = pair
        sid = state_map[pair]

        for inp in _all_valuations(all_inputs):
            inp1 = inp & c1_inp
            inp2 = inp & c2_inp

            t1 = c1.transitions.get((s1, frozenset(inp1)))
            t2 = c2.transitions.get((s2, frozenset(inp2)))
            if t1 is None or t2 is None:
                continue

            ns1, out1 = t1
            ns2, out2 = t2

            next_pair = (ns1, ns2)
            if next_pair not in state_map:
                state_map[next_pair] = next_id
                next_id += 1
            nid = state_map[next_pair]

            combined_out = frozenset(_s(out1) | _s(out2))
            transitions[(sid, frozenset(inp))] = (nid, combined_out)

            if next_pair not in visited:
                queue.append(next_pair)

    states = set(range(next_id))
    machine = MealyMachine(
        states=states,
        initial=initial,
        inputs=all_inputs,
        outputs=all_outputs,
        transitions=transitions,
    )

    return CompositionResult(
        controller=machine,
        method='parallel',
        components=[c1, c2],
        n_states=len(states),
    )


# ---------------------------------------------------------------------------
# Sequential Composition (c1 outputs feed c2 inputs)
# ---------------------------------------------------------------------------

def sequential_compose(c1, c2, shared_vars=None):
    """Compose two Mealy machines sequentially: c1's outputs feed c2's inputs.

    shared_vars: variables from c1.outputs that become c2.inputs.
                 Defaults to c1.outputs & c2.inputs.
    External inputs: c1.inputs + (c2.inputs - shared_vars).
    External outputs: c2.outputs + (c1.outputs - shared_vars).
    """
    c1_out = _s(c1.outputs)
    c2_out = _s(c2.outputs)
    c1_inp = _s(c1.inputs)
    c2_inp = _s(c2.inputs)

    if shared_vars is None:
        shared_vars = c1_out & c2_inp
    else:
        shared_vars = _s(shared_vars)
    if not shared_vars:
        raise ValueError("No shared variables between c1 outputs and c2 inputs.")

    ext_inputs = c1_inp | (c2_inp - shared_vars)
    ext_outputs = c2_out | (c1_out - shared_vars)

    initial = 0
    state_map = {(c1.initial, c2.initial): 0}
    next_id = 1
    transitions = {}
    queue = [(c1.initial, c2.initial)]
    visited = set()

    while queue:
        pair = queue.pop(0)
        if pair in visited:
            continue
        visited.add(pair)
        s1, s2 = pair
        sid = state_map[pair]

        for inp in _all_valuations(ext_inputs):
            inp1 = inp & c1_inp
            t1 = c1.transitions.get((s1, frozenset(inp1)))
            if t1 is None:
                continue
            ns1, out1 = t1
            out1_s = _s(out1)

            # Feed c1's output into c2's input
            c2_internal = out1_s & shared_vars
            c2_external = inp & c2_inp
            c2_full_inp = frozenset(c2_internal | c2_external)

            t2 = c2.transitions.get((s2, frozenset(c2_full_inp & c2_inp)))
            if t2 is None:
                continue
            ns2, out2 = t2

            next_pair = (ns1, ns2)
            if next_pair not in state_map:
                state_map[next_pair] = next_id
                next_id += 1
            nid = state_map[next_pair]

            combined_out = frozenset(_s(out2) | (out1_s - shared_vars))
            transitions[(sid, frozenset(inp))] = (nid, combined_out)

            if next_pair not in visited:
                queue.append(next_pair)

    states = set(range(next_id))
    machine = MealyMachine(
        states=states,
        initial=initial,
        inputs=ext_inputs,
        outputs=ext_outputs,
        transitions=transitions,
    )

    return CompositionResult(
        controller=machine,
        method='sequential',
        components=[c1, c2],
        n_states=len(states),
        details={'shared_vars': shared_vars},
    )


# ---------------------------------------------------------------------------
# Priority Composition (overlapping outputs with priority)
# ---------------------------------------------------------------------------

def priority_compose(c1, c2, priority='c1'):
    """Compose two Mealy machines where outputs may overlap.

    When both produce conflicting values for an output variable,
    the priority controller's value wins.

    priority: 'c1' or 'c2' (which controller wins conflicts)
    """
    c1_out = _s(c1.outputs)
    c2_out = _s(c2.outputs)
    c1_inp = _s(c1.inputs)
    c2_inp = _s(c2.inputs)

    all_inputs = c1_inp | c2_inp
    all_outputs = c1_out | c2_out
    overlap = c1_out & c2_out

    initial = 0
    state_map = {(c1.initial, c2.initial): 0}
    next_id = 1
    transitions = {}
    conflicts = []
    queue = [(c1.initial, c2.initial)]
    visited = set()

    while queue:
        pair = queue.pop(0)
        if pair in visited:
            continue
        visited.add(pair)
        s1, s2 = pair
        sid = state_map[pair]

        for inp in _all_valuations(all_inputs):
            inp1 = inp & c1_inp
            inp2 = inp & c2_inp

            t1 = c1.transitions.get((s1, frozenset(inp1)))
            t2 = c2.transitions.get((s2, frozenset(inp2)))
            if t1 is None or t2 is None:
                continue

            ns1, out1 = t1
            ns2, out2 = t2
            out1_s = _s(out1)
            out2_s = _s(out2)

            # Detect conflicts
            for v in overlap:
                if (v in out1_s) != (v in out2_s):
                    conflicts.append((sid, inp, v, v in out1_s, v in out2_s))

            # Merge with priority
            if priority == 'c1':
                combined = set(out2_s)
                for v in overlap:
                    if v in out1_s:
                        combined.add(v)
                    else:
                        combined.discard(v)
                for v in (c1_out - overlap):
                    if v in out1_s:
                        combined.add(v)
            else:
                combined = set(out1_s)
                for v in overlap:
                    if v in out2_s:
                        combined.add(v)
                    else:
                        combined.discard(v)
                for v in (c2_out - overlap):
                    if v in out2_s:
                        combined.add(v)

            next_pair = (ns1, ns2)
            if next_pair not in state_map:
                state_map[next_pair] = next_id
                next_id += 1
            nid = state_map[next_pair]

            transitions[(sid, frozenset(inp))] = (nid, frozenset(combined))
            if next_pair not in visited:
                queue.append(next_pair)

    states = set(range(next_id))
    machine = MealyMachine(
        states=states,
        initial=initial,
        inputs=all_inputs,
        outputs=all_outputs,
        transitions=transitions,
    )

    return CompositionResult(
        controller=machine,
        method='priority',
        components=[c1, c2],
        conflicts=conflicts,
        n_states=len(states),
        details={'priority': priority, 'overlap': overlap},
    )


# ---------------------------------------------------------------------------
# Conjunctive Synthesis (synthesize And(spec1, spec2) monolithically)
# ---------------------------------------------------------------------------

def conjunctive_synthesize(spec1, spec2, env_vars, sys_vars):
    """Synthesize a controller satisfying both specs simultaneously.

    Uses V186 to synthesize And(spec1, spec2).
    """
    combined = LTLAnd(spec1, spec2)
    result = synthesize(combined, env_vars, sys_vars)

    return CompositionResult(
        controller=result.controller,
        method='conjunctive',
        n_states=len(result.controller.states) if result.controller else 0,
        verified=(result.verdict == SynthesisVerdict.REALIZABLE),
        details={
            'verdict': result.verdict.name,
            'spec1': str(spec1),
            'spec2': str(spec2),
            'combined': str(combined),
        },
    )


# ---------------------------------------------------------------------------
# Assume-Guarantee Composition
# ---------------------------------------------------------------------------

def assume_guarantee_compose(specs, env_vars, sys_vars):
    """Assume-guarantee composition of multiple specifications.

    Each spec is a tuple (assumption, guarantee):
      - Component i assumes all other guarantees hold
      - Component i guarantees its own guarantee

    Circular reasoning: if each component is realizable under its assumptions,
    the composition satisfies all guarantees.
    """
    controllers = []
    details = {'components': []}

    for i, (assumption, guarantee) in enumerate(specs):
        ag_spec = LTLImplies(assumption, guarantee)
        result = synthesize(ag_spec, env_vars, sys_vars)

        comp_detail = {
            'index': i,
            'assumption': str(assumption),
            'guarantee': str(guarantee),
            'verdict': result.verdict.name,
        }
        details['components'].append(comp_detail)

        if result.verdict != SynthesisVerdict.REALIZABLE:
            return CompositionResult(
                controller=None,
                method='assume_guarantee',
                details=details,
            )

        controllers.append(result.controller)

    if len(controllers) == 1:
        machine = controllers[0]
    else:
        machine = controllers[0]
        for c in controllers[1:]:
            res = priority_compose(machine, c, priority='c1')
            machine = res.controller

    return CompositionResult(
        controller=machine,
        method='assume_guarantee',
        components=controllers,
        n_states=len(machine.states) if machine else 0,
        verified=True,
        details=details,
    )


# ---------------------------------------------------------------------------
# GR(1) Assume-Guarantee Composition
# ---------------------------------------------------------------------------

def gr1_assume_guarantee(specs):
    """Assume-guarantee composition using GR(1) synthesis.

    Each spec is a BoolGR1Spec. Synthesizes each independently, then composes.
    """
    results = []

    for i, spec in enumerate(specs):
        result = gr1_synthesize(spec)
        results.append(result)

        if result.verdict != GR1Verdict.REALIZABLE:
            return CompositionResult(
                controller=None,
                method='gr1_assume_guarantee',
                details={'failed_component': i, 'verdict': result.verdict.name},
            )

    return CompositionResult(
        controller=results[0].strategy if len(results) == 1 else results,
        method='gr1_assume_guarantee',
        components=results,
        n_states=sum(len(r.winning_region) for r in results),
        verified=True,
        details={'n_components': len(specs)},
    )


# ---------------------------------------------------------------------------
# Spec Decomposition
# ---------------------------------------------------------------------------

def decompose_spec(spec, env_vars, sys_vars):
    """Heuristically decompose an LTL spec into independent sub-specs.

    Splits conjunctions and identifies variable dependencies.
    Returns list of (sub_spec, relevant_env_vars, relevant_sys_vars).
    """
    conjuncts = _flatten_and(spec)

    if len(conjuncts) <= 1:
        return [(spec, set(env_vars), set(sys_vars))]

    components = []
    for c in conjuncts:
        atoms = _collect_atoms(c)
        c_env = set(a for a in atoms if a in env_vars)
        c_sys = set(a for a in atoms if a in sys_vars)
        components.append((c, c_env, c_sys))

    return _group_by_shared_vars(components)


def compose_from_decomposition(spec, env_vars, sys_vars):
    """Decompose spec, synthesize sub-specs independently, compose results.

    If decomposition yields >1 group, synthesizes each independently
    and parallel-composes. Falls back to monolithic if any sub-spec fails.
    """
    groups = decompose_spec(spec, env_vars, sys_vars)

    if len(groups) <= 1:
        result = synthesize(spec, env_vars, sys_vars)
        return CompositionResult(
            controller=result.controller,
            method='monolithic',
            n_states=len(result.controller.states) if result.controller else 0,
            verified=(result.verdict == SynthesisVerdict.REALIZABLE),
        )

    controllers = []
    for sub_spec, s_env, s_sys in groups:
        result = synthesize(sub_spec, list(s_env), list(s_sys))
        if result.verdict != SynthesisVerdict.REALIZABLE:
            full_result = synthesize(spec, env_vars, sys_vars)
            return CompositionResult(
                controller=full_result.controller,
                method='monolithic_fallback',
                n_states=len(full_result.controller.states) if full_result.controller else 0,
                verified=(full_result.verdict == SynthesisVerdict.REALIZABLE),
                details={'failed_sub': str(sub_spec)},
            )
        controllers.append(result.controller)

    composed = controllers[0]
    for c in controllers[1:]:
        res = parallel_compose(composed, c)
        composed = res.controller

    return CompositionResult(
        controller=composed,
        method='decomposed',
        components=controllers,
        n_states=len(composed.states),
        verified=True,
        details={'n_groups': len(groups)},
    )


# ---------------------------------------------------------------------------
# Compositional Verification
# ---------------------------------------------------------------------------

def verify_composition(composed, spec, env_vars, sys_vars, max_depth=50):
    """Verify a composed controller against a specification."""
    if composed is None:
        return False, "No controller to verify"
    return verify_controller(composed, spec, env_vars, sys_vars, max_depth)


def compare_composition_methods(spec, env_vars, sys_vars):
    """Compare monolithic vs compositional synthesis."""
    import time

    t0 = time.time()
    mono_result = synthesize(spec, env_vars, sys_vars)
    mono_time = time.time() - t0

    t0 = time.time()
    comp_result = compose_from_decomposition(spec, env_vars, sys_vars)
    comp_time = time.time() - t0

    mono_states = len(mono_result.controller.states) if mono_result.controller else 0

    return {
        'monolithic': {
            'verdict': mono_result.verdict.name,
            'states': mono_states,
            'time': mono_time,
        },
        'compositional': {
            'method': comp_result.method,
            'states': comp_result.n_states,
            'time': comp_time,
        },
        'reduction': (mono_states - comp_result.n_states) if comp_result.n_states > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Mealy Machine Operations
# ---------------------------------------------------------------------------

def product_mealy(c1, c2):
    """Raw product of two Mealy machines (shared I/O, union outputs)."""
    all_inputs = _s(c1.inputs) | _s(c2.inputs)
    all_outputs = _s(c1.outputs) | _s(c2.outputs)
    c1_inp = _s(c1.inputs)
    c2_inp = _s(c2.inputs)

    initial = 0
    state_map = {(c1.initial, c2.initial): 0}
    next_id = 1
    transitions = {}
    queue = [(c1.initial, c2.initial)]
    visited = set()

    while queue:
        pair = queue.pop(0)
        if pair in visited:
            continue
        visited.add(pair)
        s1, s2 = pair
        sid = state_map[pair]

        for inp in _all_valuations(all_inputs):
            t1 = c1.transitions.get((s1, frozenset(inp & c1_inp)))
            t2 = c2.transitions.get((s2, frozenset(inp & c2_inp)))
            if t1 is None or t2 is None:
                continue

            ns1, out1 = t1
            ns2, out2 = t2

            next_pair = (ns1, ns2)
            if next_pair not in state_map:
                state_map[next_pair] = next_id
                next_id += 1
            nid = state_map[next_pair]

            transitions[(sid, frozenset(inp))] = (nid, frozenset(_s(out1) | _s(out2)))
            if next_pair not in visited:
                queue.append(next_pair)

    return MealyMachine(
        states=set(range(next_id)),
        initial=initial,
        inputs=all_inputs,
        outputs=all_outputs,
        transitions=transitions,
    )


def restrict_mealy(machine, keep_outputs):
    """Project a Mealy machine to a subset of outputs."""
    keep = set(keep_outputs)
    new_trans = {}
    for key, (ns, out) in machine.transitions.items():
        new_trans[key] = (ns, frozenset(_s(out) & keep))
    return MealyMachine(
        states=machine.states,
        initial=machine.initial,
        inputs=machine.inputs,
        outputs=keep & _s(machine.outputs),
        transitions=new_trans,
    )


def rename_mealy(machine, input_map=None, output_map=None):
    """Rename inputs/outputs of a Mealy machine."""
    imap = input_map or {}
    omap = output_map or {}

    new_inputs = set(imap.get(v, v) for v in machine.inputs)
    new_outputs = set(omap.get(v, v) for v in machine.outputs)

    new_trans = {}
    for (s, inp), (ns, out) in machine.transitions.items():
        new_inp = frozenset(imap.get(v, v) for v in inp)
        new_out = frozenset(omap.get(v, v) for v in out)
        new_trans[(s, new_inp)] = (ns, new_out)

    return MealyMachine(
        states=machine.states,
        initial=machine.initial,
        inputs=new_inputs,
        outputs=new_outputs,
        transitions=new_trans,
    )


def minimize_mealy(machine):
    """Minimize a Mealy machine by merging bisimilar states.

    Uses partition refinement (Hopcroft-style).
    """
    m_inputs = _s(machine.inputs)
    all_inp_vals = sorted(_all_valuations(m_inputs), key=lambda x: tuple(sorted(x)))

    output_sig = {}
    for s in machine.states:
        sig = []
        for inp in all_inp_vals:
            t = machine.transitions.get((s, frozenset(inp)))
            sig.append(frozenset(t[1]) if t is not None else None)
        output_sig[s] = tuple(sig)

    partition_map = {}
    for s, sig in output_sig.items():
        if sig not in partition_map:
            partition_map[sig] = []
        partition_map[sig].append(s)
    partitions = list(partition_map.values())

    changed = True
    while changed:
        changed = False
        state_to_part = {}
        for pi, p in enumerate(partitions):
            for s in p:
                state_to_part[s] = pi

        new_partitions = []
        for part in partitions:
            if len(part) <= 1:
                new_partitions.append(part)
                continue

            subgroups = {}
            for s in part:
                sig = []
                for inp in all_inp_vals:
                    t = machine.transitions.get((s, frozenset(inp)))
                    sig.append(state_to_part.get(t[0], -1) if t is not None else -2)
                sig = tuple(sig)
                if sig not in subgroups:
                    subgroups[sig] = []
                subgroups[sig].append(s)

            if len(subgroups) > 1:
                changed = True
            new_partitions.extend(subgroups.values())
        partitions = new_partitions

    state_to_part = {}
    for pi, part in enumerate(partitions):
        for s in part:
            state_to_part[s] = pi

    new_initial = state_to_part[machine.initial]
    new_states = set(range(len(partitions)))
    new_trans = {}

    for pi, part in enumerate(partitions):
        rep = part[0]
        for inp in _all_valuations(m_inputs):
            t = machine.transitions.get((rep, frozenset(inp)))
            if t is not None:
                ns, out = t
                new_trans[(pi, frozenset(inp))] = (state_to_part[ns], out)

    return MealyMachine(
        states=new_states,
        initial=new_initial,
        inputs=machine.inputs,
        outputs=machine.outputs,
        transitions=new_trans,
    )


def mealy_equivalence(c1, c2, max_depth=100):
    """Check if two Mealy machines are input-output equivalent.

    BFS over product states; any output disagreement => not equivalent.
    """
    if _s(c1.inputs) != _s(c2.inputs) or _s(c1.outputs) != _s(c2.outputs):
        return False, "I/O variable mismatch"

    queue = [(c1.initial, c2.initial, 0)]
    visited = set()

    while queue:
        s1, s2, depth = queue.pop(0)
        if (s1, s2) in visited or depth > max_depth:
            continue
        visited.add((s1, s2))

        for inp in _all_valuations(_s(c1.inputs)):
            t1 = c1.transitions.get((s1, frozenset(inp)))
            t2 = c2.transitions.get((s2, frozenset(inp)))

            if t1 is None and t2 is None:
                continue
            if t1 is None or t2 is None:
                return False, f"Transition defined in one but not other at depth {depth}"

            ns1, out1 = t1
            ns2, out2 = t2
            if _s(out1) != _s(out2):
                return False, f"Output mismatch at depth {depth}: {out1} vs {out2}"

            if (ns1, ns2) not in visited:
                queue.append((ns1, ns2, depth + 1))

    return True, "Equivalent up to depth"


# ---------------------------------------------------------------------------
# Simulation & Trace Analysis
# ---------------------------------------------------------------------------

def simulate_composition(controller, input_sequence):
    """Simulate a composed controller on an input sequence.

    Returns list of (state, input, output) triples.
    """
    if controller is None:
        return []

    trace = []
    state = controller.initial
    for inp in input_sequence:
        inp_fs = frozenset(inp) if not isinstance(inp, frozenset) else inp
        t = controller.transitions.get((state, inp_fs))
        if t is None:
            break
        ns, out = t
        trace.append((state, inp_fs, out))
        state = ns
    return trace


def composition_statistics(result):
    """Statistics about a composition result."""
    stats = {
        'method': result.method,
        'n_states': result.n_states,
        'verified': result.verified,
        'n_components': len(result.components),
        'n_conflicts': len(result.conflicts),
    }
    if result.controller is not None:
        stats['n_transitions'] = len(result.controller.transitions)
        stats['inputs'] = sorted(result.controller.inputs)
        stats['outputs'] = sorted(result.controller.outputs)
    return stats
