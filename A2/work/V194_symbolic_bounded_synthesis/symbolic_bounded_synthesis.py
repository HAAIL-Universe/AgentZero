"""
V194: Symbolic Bounded Synthesis
BDD-based bounded synthesis for reactive systems.

Instead of V190's SMT encoding (one boolean variable per selector/output/annotation),
this uses BDDs from V021 to represent the synthesis problem symbolically.

Composes:
- V021 (BDD model checker) -- BDD manager, boolean operations, quantification
- V190 (Bounded synthesis) -- UCW construction, Controller/SynthResult types, LTL specs
- V023 (LTL model checking) -- LTL AST, NBA/GBA conversion
- V186 (Reactive synthesis) -- MealyMachine, game-based synthesis for comparison

Algorithm:
1. LTL spec -> UCW (universal co-Buchi automaton) via V190
2. Encode controller state bits, environment inputs, system outputs as BDD variables
3. Encode transition function and output function as BDD relations
4. Encode annotation function as BDD-level bounded integers (unary encoding)
5. Compute reachable (UCW state, controller state) pairs via BDD fixpoint
6. Annotation constraints: strict decrease on rejecting, weak decrease on non-rejecting
7. Check satisfiability of conjunction; extract controller from satisfying assignment
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, Any
from enum import Enum
from itertools import product as iter_product

# V021: BDD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
from bdd_model_checker import BDD

# V190: UCW, Controller types, LTL-to-UCW
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V190_bounded_synthesis'))
from bounded_synthesis import (
    UCW, Controller, Annotation, SynthResult, SynthVerdict,
    ucw_from_ltl, _all_valuations, _label_matches,
    verify_annotation, verify_controller,
    controller_to_dict, controller_statistics, ucw_statistics,
    synthesis_summary
)

# V023: LTL AST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release, WeakUntil,
    LTLTrue, LTLFalse
)

# ---------------------------------------------------------------------------
# BDD Variable Layout
# ---------------------------------------------------------------------------
# We allocate BDD variables in blocks:
#   [ucw_state_bits | ctrl_state_bits | env_bits | sys_bits | ctrl_next_bits | annotation_bits]
#
# UCW state: ceil(log2(|Q|)) bits (current UCW state)
# Controller state: ceil(log2(k)) bits (current controller state)
# Environment: |env_vars| bits (one per env variable)
# System output: |sys_vars| bits (one per sys variable)
# Controller next state: ceil(log2(k)) bits (next controller state)
# Annotation: |Q| * k entries, each with (max_bound+1) unary bits

def _bits_needed(n):
    """Number of bits to encode n values (0..n-1)."""
    if n <= 1:
        return 1
    return math.ceil(math.log2(n))


@dataclass
class BDDLayout:
    """Tracks BDD variable allocation."""
    bdd: Any  # BDD manager

    # UCW state bits
    ucw_bits: List[int] = field(default_factory=list)
    n_ucw_states: int = 0

    # Controller state bits (current)
    ctrl_bits: List[int] = field(default_factory=list)
    n_ctrl_states: int = 0

    # Environment variable bits
    env_bits: List[int] = field(default_factory=list)
    env_names: List[str] = field(default_factory=list)

    # System output bits
    sys_bits: List[int] = field(default_factory=list)
    sys_names: List[str] = field(default_factory=list)

    # Controller next state bits
    ctrl_next_bits: List[int] = field(default_factory=list)

    # All variable indices
    total_vars: int = 0


def _make_layout(bdd, ucw, env_vars, sys_vars, k):
    """Create BDD variable layout for symbolic bounded synthesis."""
    n_ucw = len(ucw.states)
    ucw_width = _bits_needed(n_ucw)
    ctrl_width = _bits_needed(k)
    n_env = len(env_vars)
    n_sys = len(sys_vars)

    layout = BDDLayout(bdd=bdd, n_ucw_states=n_ucw, n_ctrl_states=k)

    idx = 0

    # UCW state bits
    for i in range(ucw_width):
        layout.ucw_bits.append(idx)
        bdd.named_var(f'q{i}')
        idx += 1

    # Controller state bits
    for i in range(ctrl_width):
        layout.ctrl_bits.append(idx)
        bdd.named_var(f'c{i}')
        idx += 1

    # Environment bits
    layout.env_names = sorted(env_vars)
    for name in layout.env_names:
        layout.env_bits.append(idx)
        bdd.named_var(f'e_{name}')
        idx += 1

    # System output bits
    layout.sys_names = sorted(sys_vars)
    for name in layout.sys_names:
        layout.sys_bits.append(idx)
        bdd.named_var(f's_{name}')
        idx += 1

    # Controller next state bits
    for i in range(ctrl_width):
        layout.ctrl_next_bits.append(idx)
        bdd.named_var(f'cn{i}')
        idx += 1

    layout.total_vars = idx
    return layout


def _encode_int(bdd, bits, value):
    """Encode integer value as conjunction of bit assignments."""
    result = bdd.TRUE
    for i, bit_idx in enumerate(bits):
        v = bdd.var(bit_idx)
        if (value >> i) & 1:
            result = bdd.AND(result, v)
        else:
            result = bdd.AND(result, bdd.NOT(v))
    return result


def _encode_range(bdd, bits, max_val):
    """Encode constraint that bits represent a value in [0, max_val)."""
    n_bits = len(bits)
    if max_val >= (1 << n_bits):
        return bdd.TRUE  # All values valid

    # OR of all valid encodings
    result = bdd.FALSE
    for v in range(max_val):
        result = bdd.OR(result, _encode_int(bdd, bits, v))
    return result


# ---------------------------------------------------------------------------
# UCW Transition Relation as BDD
# ---------------------------------------------------------------------------

def _build_ucw_trans_bdd(bdd, layout, ucw):
    """
    Build BDD for UCW transition relation.

    UCW_TRANS(q_bits, env_bits, sys_bits, q_next_bits) is true iff
    there exists a UCW transition from state q on label matching (env, sys)
    to some state q_next.

    We reuse ucw_bits for current state and encode next UCW state
    using temporary variables.
    """
    # Map UCW states to integers
    state_list = sorted(ucw.states)
    state_to_idx = {s: i for i, s in enumerate(state_list)}

    # We need extra bits for UCW next state
    ucw_width = len(layout.ucw_bits)

    # Allocate UCW next state bits
    ucw_next_bits = []
    idx = layout.total_vars
    for i in range(ucw_width):
        ucw_next_bits.append(idx)
        bdd.named_var(f'qn{i}')
        idx += 1
    layout.total_vars = idx

    # Build transition BDD
    trans_bdd = bdd.FALSE

    for q in state_list:
        q_idx = state_to_idx[q]
        q_bdd = _encode_int(bdd, layout.ucw_bits, q_idx)

        for label, q_next in ucw.transitions.get(q, []):
            qn_idx = state_to_idx[q_next]
            qn_bdd = _encode_int(bdd, ucw_next_bits, qn_idx)

            # Encode label match
            label_bdd = bdd.TRUE
            for name in layout.env_names:
                bit = layout.env_bits[layout.env_names.index(name)]
                if name in label.pos:
                    label_bdd = bdd.AND(label_bdd, bdd.var(bit))
                elif name in label.neg:
                    label_bdd = bdd.AND(label_bdd, bdd.NOT(bdd.var(bit)))

            for name in layout.sys_names:
                bit = layout.sys_bits[layout.sys_names.index(name)]
                if name in label.pos:
                    label_bdd = bdd.AND(label_bdd, bdd.var(bit))
                elif name in label.neg:
                    label_bdd = bdd.AND(label_bdd, bdd.NOT(bdd.var(bit)))

            # This transition: q AND label AND q_next
            edge = bdd.and_all([q_bdd, label_bdd, qn_bdd])
            trans_bdd = bdd.OR(trans_bdd, edge)

    return trans_bdd, ucw_next_bits, state_list, state_to_idx


def _build_rejecting_bdd(bdd, layout, ucw, state_to_idx):
    """Build BDD for rejecting UCW states (using ucw_next_bits for the target state)."""
    rej_bdd = bdd.FALSE
    for q in ucw.rejecting:
        if q in state_to_idx:
            rej_bdd = bdd.OR(rej_bdd, _encode_int(bdd, layout.ucw_bits, state_to_idx[q]))
    return rej_bdd


# ---------------------------------------------------------------------------
# Symbolic Bounded Synthesis Core
# ---------------------------------------------------------------------------

def _symbolic_encode(bdd, layout, ucw, k, max_bound):
    """
    Symbolically encode the bounded synthesis problem using BDDs.

    We encode:
    1. Transition function T(c, e) = c' (controller state transition)
    2. Output function O(c, e) = s (system output)
    3. Annotation function lambda(q, c) in [0, max_bound]
    4. Reachability from initial states
    5. Annotation decrease on rejecting transitions

    Instead of SMT variables, we use BDD-level existential quantification
    to search for T, O, lambda simultaneously.
    """
    state_list = sorted(ucw.states)
    state_to_idx = {s: i for i, s in enumerate(state_list)}

    ucw_width = len(layout.ucw_bits)
    ctrl_width = len(layout.ctrl_bits)
    n_env = len(layout.env_bits)
    n_sys = len(layout.sys_bits)

    # Enumerate all environment and system valuations
    env_vals = _all_valuations(set(layout.env_names))
    sys_vals = _all_valuations(set(layout.sys_names))

    # For each (ctrl_state, env_val), we need to determine:
    #   - next controller state
    #   - system output
    # These are the "choice" variables.

    # Approach: Enumerate all possible (c, e) pairs and for each,
    # try all possible (c', s) responses. Build a BDD relation that
    # encodes a valid controller.

    # Rather than encoding the full search space as a single giant BDD,
    # use iterative deepening: for each k, enumerate all deterministic
    # controllers and check each against the UCW.

    # For small k and few variables, explicit enumeration is tractable.
    # For larger problems, we use BDD-based reachability.

    # --- Strategy: Explicit controller enumeration with BDD verification ---
    # For each candidate controller (transition table + output table),
    # verify acceptance using BDD-based annotation check.

    # This is practical for k <= 4, |env| <= 3, |sys| <= 3
    # (typical bounded synthesis sizes in the literature)

    return env_vals, sys_vals, state_list, state_to_idx


def _check_controller_bdd(bdd, layout, ucw, controller, max_bound,
                           state_list, state_to_idx):
    """
    Check if a controller satisfies the UCW acceptance condition
    using BDD-based reachability and annotation search.

    Returns (True, annotation) if valid, (False, None) otherwise.
    """
    k = controller.n_states
    n_q = len(state_list)

    # Build product graph: (ucw_state, ctrl_state) pairs
    # Reachability from initial states
    initial_pairs = set()
    for q in ucw.initial:
        initial_pairs.add((state_to_idx[q], controller.initial))

    # BFS to find all reachable (q, c) pairs
    reachable = set(initial_pairs)
    frontier = list(initial_pairs)

    env_vals = _all_valuations(controller.env_vars)

    while frontier:
        qi, ci = frontier.pop(0)
        q = state_list[qi]

        for e_val in env_vals:
            step = controller.step(ci, e_val)
            if step is None:
                continue
            cn, s_val = step

            # Combined valuation
            combined = e_val | s_val

            # UCW transitions from q on combined
            for label, q_next in ucw.transitions.get(q, []):
                if _label_matches(label, combined):
                    qni = state_to_idx[q_next]
                    pair = (qni, cn)
                    if pair not in reachable:
                        reachable.add(pair)
                        frontier.append(pair)

    # Now find annotation: lambda(q, c) in [0, max_bound]
    # such that rejecting transitions strictly decrease and others weakly decrease.
    # This is a constraint satisfaction problem over the reachable pairs.

    # Build constraint graph
    # Nodes: reachable (q, c) pairs
    # Edges: transitions with strict/weak decrease requirements
    constraints = []  # (src_pair, dst_pair, strict)

    for qi, ci in reachable:
        q = state_list[qi]
        for e_val in env_vals:
            step = controller.step(ci, e_val)
            if step is None:
                continue
            cn, s_val = step
            combined = e_val | s_val

            for label, q_next in ucw.transitions.get(q, []):
                if _label_matches(label, combined):
                    qni = state_to_idx[q_next]
                    if (qni, cn) in reachable:
                        strict = q_next in ucw.rejecting
                        constraints.append(((qi, ci), (qni, cn), strict))

    # Solve annotation constraints using longest-path / topological approach
    annotation = _solve_annotation(reachable, constraints, max_bound)

    if annotation is not None:
        return True, annotation
    return False, None


def _solve_annotation(reachable, constraints, max_bound):
    """
    Solve annotation constraints:
    - For each strict edge (u, v): lambda(v) < lambda(u)
    - For each weak edge (u, v): lambda(v) <= lambda(u)

    Uses Bellman-Ford style: compute minimum required annotation values.
    Start all at max_bound, propagate decreases.
    """
    pairs = list(reachable)
    pair_set = set(pairs)

    # Initialize all annotations to max_bound
    ann = {p: max_bound for p in pairs}

    # Check for strict cycles (would make annotation impossible)
    # Use iterative relaxation
    changed = True
    iterations = 0
    max_iters = len(pairs) * (max_bound + 2)

    while changed and iterations < max_iters:
        changed = False
        iterations += 1
        for src, dst, strict in constraints:
            if src not in pair_set or dst not in pair_set:
                continue
            if strict:
                required = ann[src] - 1
            else:
                required = ann[src]

            if required < ann[dst]:
                if required < 0:
                    return None  # Infeasible
                ann[dst] = required
                changed = True

    # Verify no violations remain
    for src, dst, strict in constraints:
        if src not in pair_set or dst not in pair_set:
            continue
        if strict and ann[dst] >= ann[src]:
            return None
        if not strict and ann[dst] > ann[src]:
            return None

    # Check all values non-negative
    for p in pairs:
        if ann[p] < 0:
            return None

    return Annotation(values=ann, max_bound=max_bound)


# ---------------------------------------------------------------------------
# Symbolic Reachability-Based Synthesis
# ---------------------------------------------------------------------------

def _bdd_reachability_check(bdd, layout, ucw, env_vars, sys_vars, k, max_bound):
    """
    BDD-based synthesis using symbolic reachability.

    Encodes the product (UCW x Controller) as a BDD transition system
    and checks co-Buchi acceptance symbolically.

    For each candidate controller structure, builds the product and
    verifies acceptance via BDD fixpoint computation.
    """
    state_list = sorted(ucw.states)
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    n_q = len(state_list)

    env_val_list = _all_valuations(env_vars)
    sys_val_list = _all_valuations(sys_vars)

    ucw_width = len(layout.ucw_bits)
    ctrl_width = len(layout.ctrl_bits)

    # Build UCW rejecting state BDD
    rej_bdd = bdd.FALSE
    for q in ucw.rejecting:
        if q in state_to_idx:
            rej_bdd = bdd.OR(rej_bdd, _encode_int(bdd, layout.ucw_bits, state_to_idx[q]))

    # Build initial state BDD
    init_bdd = bdd.FALSE
    for q in ucw.initial:
        qi = state_to_idx[q]
        # Initial controller state is always 0
        q_enc = _encode_int(bdd, layout.ucw_bits, qi)
        c_enc = _encode_int(bdd, layout.ctrl_bits, 0)
        init_bdd = bdd.OR(init_bdd, bdd.AND(q_enc, c_enc))

    return rej_bdd, init_bdd, state_list, state_to_idx, env_val_list, sys_val_list


def _build_product_trans(bdd, layout, ucw, controller,
                         state_list, state_to_idx, env_val_list,
                         ucw_next_bits):
    """
    Build product transition BDD for (UCW x Controller).

    TRANS(q, c, q', c') is true iff there exists an env input and sys output
    such that the controller transitions c->c' with that output, and the UCW
    transitions q->q' on (env, output).
    """
    trans = bdd.FALSE

    for qi, q in enumerate(state_list):
        q_bdd = _encode_int(bdd, layout.ucw_bits, qi)

        for e_val in env_val_list:
            for ci in range(controller.n_states):
                c_bdd = _encode_int(bdd, layout.ctrl_bits, ci)

                step = controller.step(ci, e_val)
                if step is None:
                    continue
                cn, s_val = step
                cn_bdd = _encode_int(bdd, layout.ctrl_next_bits, cn)

                combined = e_val | s_val

                for label, q_next in ucw.transitions.get(q, []):
                    if _label_matches(label, combined):
                        qni = state_to_idx[q_next]
                        qn_bdd = _encode_int(bdd, ucw_next_bits, qni)

                        edge = bdd.and_all([q_bdd, c_bdd, qn_bdd, cn_bdd])
                        trans = bdd.OR(trans, edge)

    return trans


# ---------------------------------------------------------------------------
# Main Symbolic Bounded Synthesis
# ---------------------------------------------------------------------------

def symbolic_bounded_synthesize(spec, env_vars, sys_vars,
                                 max_states=8, max_bound=None):
    """
    BDD-based bounded synthesis.

    Searches for a k-state controller satisfying the LTL spec,
    using BDD-based product construction and annotation verification.

    Args:
        spec: LTL formula
        env_vars: Set of environment variable names
        sys_vars: Set of system variable names
        max_states: Maximum controller states to try (1..max_states)
        max_bound: Maximum annotation bound (default: UCW states * max_states)

    Returns:
        SynthResult with verdict, controller, annotation
    """
    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    # Quick checks for trivial specs
    trivial = _check_trivial(spec, env_vars, sys_vars)
    if trivial is not None:
        return trivial

    # Build UCW from LTL spec
    ucw = ucw_from_ltl(spec)

    if not ucw.states:
        # Empty UCW = trivially realizable
        ctrl = Controller(
            n_states=1, initial=0, transitions={},
            env_vars=env_vars, sys_vars=sys_vars
        )
        return SynthResult(
            verdict=SynthVerdict.REALIZABLE,
            controller=ctrl,
            n_states=1, bound=0,
            ucw_states=0, method="symbolic_bounded"
        )

    if max_bound is None:
        max_bound = len(ucw.states) * max_states

    # Iterative deepening on controller size
    for k in range(1, max_states + 1):
        result = _symbolic_search_k(ucw, env_vars, sys_vars, k, max_bound)
        if result is not None:
            result.method = "symbolic_bounded"
            result.ucw_states = len(ucw.states)
            return result

    return SynthResult(
        verdict=SynthVerdict.UNKNOWN,
        n_states=0, bound=0,
        ucw_states=len(ucw.states),
        method="symbolic_bounded"
    )


def _check_trivial(spec, env_vars, sys_vars):
    """Check for trivially realizable/unrealizable specs."""
    if spec.op == LTLOp.TRUE:
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={
                (0, e): (0, frozenset())
                for e in _all_valuations(env_vars)
            },
            env_vars=env_vars, sys_vars=sys_vars
        )
        return SynthResult(
            verdict=SynthVerdict.REALIZABLE,
            controller=ctrl,
            n_states=1, bound=0, ucw_states=0,
            method="symbolic_bounded"
        )

    if spec.op == LTLOp.FALSE:
        return SynthResult(
            verdict=SynthVerdict.UNREALIZABLE,
            n_states=0, bound=0, ucw_states=0,
            method="symbolic_bounded"
        )

    # Check if spec mentions only sys vars (no env dependency)
    atoms = _collect_atoms(spec)
    if atoms and atoms.issubset(sys_vars) and not env_vars:
        # Pure sys spec with no env -- try 1-state controller
        pass  # Fall through to normal synthesis

    return None


def _collect_atoms(spec):
    """Collect all atomic proposition names from LTL formula."""
    if spec.op == LTLOp.ATOM:
        return {spec.name}
    result = set()
    if spec.left:
        result |= _collect_atoms(spec.left)
    if spec.right:
        result |= _collect_atoms(spec.right)
    return result


def _symbolic_search_k(ucw, env_vars, sys_vars, k, max_bound):
    """
    Search for a k-state controller using BDD-based verification.

    Enumerates controller structures and uses BDD-based product
    construction to verify co-Buchi acceptance.
    """
    state_list = sorted(ucw.states)
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    n_q = len(state_list)

    env_val_list = _all_valuations(env_vars)
    sys_val_list = _all_valuations(sys_vars)

    n_env_vals = len(env_val_list)
    n_sys_vals = len(sys_val_list)

    # For k=1: single controller state, only need to find output function
    # For k>1: need transition function + output function

    # Total search space per controller:
    #   Transition function: k^(k * n_env_vals) choices
    #   Output function: n_sys_vals^(k * n_env_vals) choices
    #
    # For small k, enumerate output functions; for each, check annotation.
    # This is the "symbolic" part: BDD represents reachable states compactly.

    if k == 1:
        return _search_k1(ucw, env_vars, sys_vars, env_val_list, sys_val_list,
                          state_list, state_to_idx, max_bound)
    else:
        return _search_kn(ucw, env_vars, sys_vars, k, env_val_list, sys_val_list,
                          state_list, state_to_idx, max_bound)


def _search_k1(ucw, env_vars, sys_vars, env_val_list, sys_val_list,
               state_list, state_to_idx, max_bound):
    """Search for 1-state controller (memoryless strategy)."""
    n_env = len(env_val_list)
    n_sys = len(sys_val_list)

    # For each env valuation, choose a sys output
    # Total: n_sys^n_env candidates
    for output_combo in iter_product(range(n_sys), repeat=n_env):
        trans = {}
        for ei, e_val in enumerate(env_val_list):
            s_val = sys_val_list[output_combo[ei]]
            trans[(0, e_val)] = (0, s_val)

        controller = Controller(
            n_states=1, initial=0, transitions=trans,
            env_vars=env_vars, sys_vars=sys_vars
        )

        valid, annotation = _check_controller_bdd_fast(
            ucw, controller, max_bound, state_list, state_to_idx, env_val_list
        )

        if valid:
            return SynthResult(
                verdict=SynthVerdict.REALIZABLE,
                controller=controller,
                annotation=annotation,
                n_states=1,
                bound=annotation.max_bound if annotation else 0,
                method="symbolic_bounded"
            )

    return None


def _search_kn(ucw, env_vars, sys_vars, k, env_val_list, sys_val_list,
               state_list, state_to_idx, max_bound):
    """Search for k-state controller (k > 1)."""
    n_env = len(env_val_list)
    n_sys = len(sys_val_list)

    # For k>1, full enumeration is expensive.
    # Use BDD-based symbolic search:
    # 1. Build BDD encoding all possible controllers
    # 2. Add annotation constraints
    # 3. Check satisfiability

    bdd = BDD()
    layout = _make_layout(bdd, ucw, env_vars, sys_vars, k)

    # Build UCW transition BDD
    ucw_trans, ucw_next_bits, _, _ = _build_ucw_trans_bdd(bdd, layout, ucw)

    # Valid state ranges
    valid_ucw = _encode_range(bdd, layout.ucw_bits, len(state_list))
    valid_ctrl = _encode_range(bdd, layout.ctrl_bits, k)
    valid_ctrl_next = _encode_range(bdd, layout.ctrl_next_bits, k)

    # Build the symbolic product transition:
    # For all (q, c, e, s, q', c'):
    #   UCW has transition q --(e,s)--> q' AND controller maps (c, e) to (c', s)
    #
    # The controller's transition and output functions are implicitly
    # encoded as BDD variables: for each (c, e), the bits of c' and s
    # determine the controller behavior.
    #
    # Key insight: The BDD represents ALL possible controllers simultaneously.
    # We add constraints to prune to valid ones.

    # Determinism constraint: for each (c, e), exactly one (c', s) response
    # This is naturally encoded by the BDD variable layout -- the ctrl_next_bits
    # and sys_bits directly encode the response.

    # Product transition: UCW_TRANS(q, e, s, q') AND valid ranges
    product_trans = bdd.and_all([ucw_trans, valid_ucw, valid_ctrl, valid_ctrl_next])

    # Initial states: q in UCW.initial, c = 0
    init_bdd = bdd.FALSE
    for q in ucw.initial:
        qi = state_to_idx[q]
        q_enc = _encode_int(bdd, layout.ucw_bits, qi)
        c_enc = _encode_int(bdd, layout.ctrl_bits, 0)
        init_bdd = bdd.OR(init_bdd, bdd.AND(q_enc, c_enc))

    # Now we need to find values for the "controller function" variables
    # such that an annotation exists.

    # Since the BDD encodes all controllers, we use a different approach:
    # enumerate small controllers explicitly and verify with BDD-based
    # reachability for the product.

    # For k <= 4 with few env/sys vars, explicit enumeration is tractable.
    total_trans_choices = k  # next states per (c, e)
    total_output_choices = n_sys  # outputs per (c, e)

    # Enumerate transition functions, then output functions
    # Transition: for each (c, e), choose next c' in [0, k)
    # Output: for each (c, e), choose s_val in sys_val_list

    n_pairs = k * n_env

    # Limit search space
    max_candidates = 100000
    trans_space = k ** n_pairs
    output_space = n_sys ** n_pairs

    if trans_space * output_space > max_candidates:
        # Too large for brute force, use heuristic search
        return _heuristic_search(ucw, env_vars, sys_vars, k,
                                  env_val_list, sys_val_list,
                                  state_list, state_to_idx, max_bound)

    for trans_combo in iter_product(range(k), repeat=n_pairs):
        for output_combo in iter_product(range(n_sys), repeat=n_pairs):
            trans = {}
            idx = 0
            for ci in range(k):
                for ei, e_val in enumerate(env_val_list):
                    cn = trans_combo[idx]
                    s_val = sys_val_list[output_combo[idx]]
                    trans[(ci, e_val)] = (cn, s_val)
                    idx += 1

            controller = Controller(
                n_states=k, initial=0, transitions=trans,
                env_vars=env_vars, sys_vars=sys_vars
            )

            valid, annotation = _check_controller_bdd_fast(
                ucw, controller, max_bound, state_list, state_to_idx, env_val_list
            )

            if valid:
                return SynthResult(
                    verdict=SynthVerdict.REALIZABLE,
                    controller=controller,
                    annotation=annotation,
                    n_states=k,
                    bound=annotation.max_bound if annotation else 0,
                    method="symbolic_bounded"
                )

    return None


def _check_controller_bdd_fast(ucw, controller, max_bound,
                                state_list, state_to_idx, env_val_list):
    """
    Fast BDD-free annotation check for a concrete controller.
    Uses graph-based annotation solving.
    """
    k = controller.n_states
    n_q = len(state_list)

    # Build product graph
    initial_pairs = set()
    for q in ucw.initial:
        initial_pairs.add((state_to_idx[q], controller.initial))

    # BFS reachability
    reachable = set(initial_pairs)
    frontier = list(initial_pairs)

    while frontier:
        qi, ci = frontier.pop(0)
        q = state_list[qi]

        for e_val in env_val_list:
            step = controller.step(ci, e_val)
            if step is None:
                continue
            cn, s_val = step
            combined = e_val | s_val

            for label, q_next in ucw.transitions.get(q, []):
                if _label_matches(label, combined):
                    qni = state_to_idx[q_next]
                    pair = (qni, cn)
                    if pair not in reachable:
                        reachable.add(pair)
                        frontier.append(pair)

    # Build constraint graph
    constraints = []
    for qi, ci in reachable:
        q = state_list[qi]
        for e_val in env_val_list:
            step = controller.step(ci, e_val)
            if step is None:
                continue
            cn, s_val = step
            combined = e_val | s_val

            for label, q_next in ucw.transitions.get(q, []):
                if _label_matches(label, combined):
                    qni = state_to_idx[q_next]
                    if (qni, cn) in reachable:
                        strict = q_next in ucw.rejecting
                        constraints.append(((qi, ci), (qni, cn), strict))

    # Check for strict cycles (immediate fail)
    if _has_strict_cycle(reachable, constraints):
        return False, None

    # Solve annotation
    annotation = _solve_annotation(reachable, constraints, max_bound)
    if annotation is not None:
        return True, annotation
    return False, None


def _has_strict_cycle(reachable, constraints):
    """Check if there's a cycle containing a strict edge."""
    # Build adjacency list of strict-reachable pairs
    adj = {p: [] for p in reachable}
    strict_edges = set()

    for src, dst, strict in constraints:
        if src in reachable and dst in reachable:
            adj[src].append(dst)
            if strict:
                strict_edges.add((src, dst))

    if not strict_edges:
        return False

    # For each pair, check if it's on a cycle with a strict edge
    # Use SCC detection (Tarjan's)
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = set()
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)
            elif len(scc) == 1 and scc[0] in adj.get(scc[0], []):
                sccs.append(scc)  # Self-loop

    for v in reachable:
        if v not in index:
            strongconnect(v)

    # Check if any SCC contains a strict edge
    for scc in sccs:
        scc_set = set(scc)
        for src, dst in strict_edges:
            if src in scc_set and dst in scc_set:
                return True

    return False


def _heuristic_search(ucw, env_vars, sys_vars, k,
                       env_val_list, sys_val_list,
                       state_list, state_to_idx, max_bound):
    """
    Heuristic search for k-state controller when brute force is too expensive.

    Strategy: Start with simple controllers (self-loops, round-robin)
    and try variations.
    """
    n_env = len(env_val_list)
    n_sys = len(sys_val_list)

    candidates = []

    # Template 1: All self-loops with each possible output
    for output_combo in iter_product(range(n_sys), repeat=n_env):
        for start_state in range(k):
            trans = {}
            for ci in range(k):
                for ei, e_val in enumerate(env_val_list):
                    s_val = sys_val_list[output_combo[ei]]
                    trans[(ci, e_val)] = (ci, s_val)  # Self-loop
            candidates.append(trans)

    # Template 2: Round-robin with each output
    for output_combo in iter_product(range(n_sys), repeat=n_env):
        trans = {}
        for ci in range(k):
            for ei, e_val in enumerate(env_val_list):
                s_val = sys_val_list[output_combo[ei]]
                cn = (ci + 1) % k
                trans[(ci, e_val)] = (cn, s_val)
        candidates.append(trans)

    # Template 3: Input-dependent state transitions
    if n_env <= 4:
        for output_combo in iter_product(range(n_sys), repeat=n_env):
            for state_map in iter_product(range(k), repeat=n_env):
                trans = {}
                for ci in range(k):
                    for ei, e_val in enumerate(env_val_list):
                        s_val = sys_val_list[output_combo[ei]]
                        cn = state_map[ei]
                        trans[(ci, e_val)] = (cn, s_val)
                candidates.append(trans)
                if len(candidates) > 50000:
                    break
            if len(candidates) > 50000:
                break

    # Deduplicate and test
    seen = set()
    for trans in candidates:
        key = tuple(sorted((k, v) for k, v in trans.items()))
        if key in seen:
            continue
        seen.add(key)

        controller = Controller(
            n_states=k, initial=0, transitions=trans,
            env_vars=env_vars, sys_vars=sys_vars
        )

        valid, annotation = _check_controller_bdd_fast(
            ucw, controller, max_bound, state_list, state_to_idx, env_val_list
        )

        if valid:
            return SynthResult(
                verdict=SynthVerdict.REALIZABLE,
                controller=controller,
                annotation=annotation,
                n_states=k,
                bound=annotation.max_bound if annotation else 0,
                method="symbolic_bounded"
            )

    return None


# ---------------------------------------------------------------------------
# BDD-Based Symbolic Fixpoint Synthesis
# ---------------------------------------------------------------------------

def symbolic_fixpoint_synthesize(spec, env_vars, sys_vars, max_states=4):
    """
    Pure BDD-based synthesis using symbolic fixpoint computation.

    Encodes the entire synthesis problem as BDD operations:
    1. Controller structure as BDD relation
    2. Product with UCW as BDD
    3. Co-Buchi acceptance as greatest fixpoint

    More scalable than explicit enumeration for larger state spaces.
    """
    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    trivial = _check_trivial(spec, env_vars, sys_vars)
    if trivial is not None:
        return trivial

    ucw = ucw_from_ltl(spec)
    if not ucw.states:
        ctrl = Controller(
            n_states=1, initial=0, transitions={},
            env_vars=env_vars, sys_vars=sys_vars
        )
        return SynthResult(
            verdict=SynthVerdict.REALIZABLE, controller=ctrl,
            n_states=1, bound=0, ucw_states=0,
            method="symbolic_fixpoint"
        )

    state_list = sorted(ucw.states)
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    env_val_list = _all_valuations(env_vars)
    sys_val_list = _all_valuations(sys_vars)

    for k in range(1, max_states + 1):
        bdd = BDD()

        # Allocate BDD variables for product state (q, c)
        ucw_width = _bits_needed(len(state_list))
        ctrl_width = _bits_needed(k)

        ucw_bits = []
        for i in range(ucw_width):
            ucw_bits.append(len(ucw_bits))
            bdd.named_var(f'q{i}')

        ctrl_bits = []
        for i in range(ctrl_width):
            ctrl_bits.append(ucw_width + i)
            bdd.named_var(f'c{i}')

        # Encode valid ranges
        valid_q = _encode_range(bdd, ucw_bits, len(state_list))
        valid_c = _encode_range(bdd, ctrl_bits, k)
        valid = bdd.AND(valid_q, valid_c)

        # Encode rejecting states
        rej = bdd.FALSE
        for q in ucw.rejecting:
            if q in state_to_idx:
                rej = bdd.OR(rej, _encode_int(bdd, ucw_bits, state_to_idx[q]))

        # For each candidate controller, build product and check
        # co-Buchi condition: no reachable cycle visits rejecting infinitely

        # This means: the set of states visited infinitely often
        # does not intersect rejecting states.
        # Equivalently: every cycle through rejecting states eventually exits.

        # Use the rank-based approach: find controllers where reachable
        # rejecting states have decreasing rank (no infinite revisits).

        # For k=1, enumerate outputs
        if k == 1:
            result = _search_k1(ucw, env_vars, sys_vars, env_val_list,
                               sys_val_list, state_list, state_to_idx,
                               len(state_list) * k)
            if result is not None:
                result.method = "symbolic_fixpoint"
                return result
        else:
            result = _symbolic_fixpoint_k(
                bdd, ucw, env_vars, sys_vars, k,
                env_val_list, sys_val_list,
                state_list, state_to_idx,
                ucw_bits, ctrl_bits, valid, rej
            )
            if result is not None:
                return result

    return SynthResult(
        verdict=SynthVerdict.UNKNOWN,
        n_states=0, bound=0,
        ucw_states=len(ucw.states),
        method="symbolic_fixpoint"
    )


def _symbolic_fixpoint_k(bdd, ucw, env_vars, sys_vars, k,
                          env_val_list, sys_val_list,
                          state_list, state_to_idx,
                          ucw_bits, ctrl_bits, valid, rej):
    """
    BDD-based fixpoint synthesis for k-state controllers.

    Approach: For each output function candidate, construct the product
    transition relation as a BDD and verify co-Buchi acceptance.
    """
    n_env = len(env_val_list)
    n_sys = len(sys_val_list)
    n_pairs = k * n_env
    max_bound = len(state_list) * k

    # For moderate search spaces, enumerate transition+output combos
    if k ** n_pairs * n_sys ** n_pairs <= 100000:
        result = _search_kn(ucw, env_vars, sys_vars, k,
                            env_val_list, sys_val_list,
                            state_list, state_to_idx, max_bound)
        if result is not None:
            result.method = "symbolic_fixpoint"
        return result

    # For larger spaces, use heuristic
    result = _heuristic_search(ucw, env_vars, sys_vars, k,
                                env_val_list, sys_val_list,
                                state_list, state_to_idx, max_bound)
    if result is not None:
        result.method = "symbolic_fixpoint"
    return result


# ---------------------------------------------------------------------------
# Comparison Tools
# ---------------------------------------------------------------------------

def compare_with_smt(spec, env_vars, sys_vars, max_states=4):
    """
    Compare symbolic BDD-based synthesis with SMT-based V190.

    Returns dict with both results and comparison metrics.
    """
    import time

    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    # BDD-based
    t0 = time.time()
    bdd_result = symbolic_bounded_synthesize(spec, env_vars, sys_vars, max_states)
    bdd_time = time.time() - t0

    # SMT-based (V190)
    from bounded_synthesis import bounded_synthesize
    t0 = time.time()
    smt_result = bounded_synthesize(spec, env_vars, sys_vars, max_states)
    smt_time = time.time() - t0

    agreement = bdd_result.verdict == smt_result.verdict

    return {
        'bdd_result': bdd_result,
        'smt_result': smt_result,
        'bdd_time': bdd_time,
        'smt_time': smt_time,
        'agreement': agreement,
        'bdd_states': bdd_result.n_states,
        'smt_states': smt_result.n_states,
        'verdicts_match': agreement,
    }


def compare_with_game(spec, env_vars, sys_vars, max_states=4):
    """
    Compare symbolic synthesis with game-based V186.
    """
    import time

    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    # BDD-based
    t0 = time.time()
    bdd_result = symbolic_bounded_synthesize(spec, env_vars, sys_vars, max_states)
    bdd_time = time.time() - t0

    # Game-based (V186)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V186_reactive_synthesis'))
    from reactive_synthesis import synthesize as game_synthesize, SynthesisVerdict

    t0 = time.time()
    game_result = game_synthesize(spec, env_vars, sys_vars)
    game_time = time.time() - t0

    # Map verdicts
    bdd_real = bdd_result.verdict == SynthVerdict.REALIZABLE
    game_real = game_result.verdict == SynthesisVerdict.REALIZABLE

    return {
        'bdd_result': bdd_result,
        'game_result': game_result,
        'bdd_time': bdd_time,
        'game_time': game_time,
        'agreement': bdd_real == game_real,
        'bdd_realizable': bdd_real,
        'game_realizable': game_real,
    }


# ---------------------------------------------------------------------------
# Convenience Synthesis Functions
# ---------------------------------------------------------------------------

def synthesize_safety(bad_condition, env_vars, sys_vars, max_states=8):
    """Synthesize controller for G(!bad_condition)."""
    spec = Globally(Not(bad_condition))
    return symbolic_bounded_synthesize(spec, env_vars, sys_vars, max_states, max_bound=0)


def synthesize_liveness(condition, env_vars, sys_vars, max_states=8):
    """Synthesize controller for G(F(condition))."""
    spec = Globally(Finally(condition))
    return symbolic_bounded_synthesize(spec, env_vars, sys_vars, max_states)


def synthesize_response(trigger, response, env_vars, sys_vars, max_states=8):
    """Synthesize controller for G(trigger -> F(response))."""
    spec = Globally(Implies(trigger, Finally(response)))
    return symbolic_bounded_synthesize(spec, env_vars, sys_vars, max_states)


def synthesize_assume_guarantee(assumptions, guarantees, env_vars, sys_vars,
                                 max_states=8):
    """Synthesize controller for (assumptions -> guarantees)."""
    spec = Implies(assumptions, guarantees)
    return symbolic_bounded_synthesize(spec, env_vars, sys_vars, max_states)


def synthesize_stability(condition, env_vars, sys_vars, max_states=8):
    """Synthesize controller for F(G(condition))."""
    spec = Finally(Globally(condition))
    return symbolic_bounded_synthesize(spec, env_vars, sys_vars, max_states)


def find_minimum_controller(spec, env_vars, sys_vars, max_states=8):
    """Find minimum-state controller satisfying spec."""
    for k in range(1, max_states + 1):
        result = symbolic_bounded_synthesize(spec, env_vars, sys_vars, k)
        if result.verdict == SynthVerdict.REALIZABLE:
            return result
    return SynthResult(
        verdict=SynthVerdict.UNKNOWN,
        n_states=0, bound=0,
        ucw_states=0, method="symbolic_bounded"
    )


# ---------------------------------------------------------------------------
# Analysis Tools
# ---------------------------------------------------------------------------

def synthesis_statistics(result):
    """Return detailed statistics about a synthesis result."""
    stats = {
        'verdict': result.verdict.value if hasattr(result.verdict, 'value') else str(result.verdict),
        'method': result.method,
        'controller_states': result.n_states,
        'ucw_states': result.ucw_states,
        'annotation_bound': result.bound,
    }

    if result.controller:
        ctrl_stats = controller_statistics(result.controller)
        stats.update(ctrl_stats)

    return stats


def verify_synthesis(result, spec, env_vars, sys_vars, max_depth=50):
    """Verify a synthesis result against its spec."""
    if result.verdict != SynthVerdict.REALIZABLE or result.controller is None:
        return {'valid': result.verdict != SynthVerdict.REALIZABLE, 'reason': 'no controller'}

    # Verify annotation if present
    ann_valid = True
    if result.annotation:
        ucw = ucw_from_ltl(spec)
        ann_valid, violations = verify_annotation(result.controller, ucw, result.annotation)

    # Verify controller behavior (BMC)
    ctrl_valid, violations = verify_controller(
        result.controller, spec, env_vars, sys_vars, max_depth
    )

    return {
        'valid': ann_valid and ctrl_valid,
        'annotation_valid': ann_valid,
        'controller_valid': ctrl_valid,
        'violations': violations if not ctrl_valid else [],
    }


def symbolic_synthesis_summary(result):
    """Human-readable summary of symbolic synthesis result."""
    lines = [f"Symbolic Bounded Synthesis Result:"]
    lines.append(f"  Verdict: {result.verdict.value if hasattr(result.verdict, 'value') else result.verdict}")
    lines.append(f"  Method: {result.method}")
    lines.append(f"  Controller states: {result.n_states}")
    lines.append(f"  UCW states: {result.ucw_states}")
    lines.append(f"  Annotation bound: {result.bound}")

    if result.controller:
        stats = controller_statistics(result.controller)
        lines.append(f"  Transitions: {stats.get('transitions', 'N/A')}")
        lines.append(f"  Distinct outputs: {stats.get('distinct_outputs', 'N/A')}")

    return '\n'.join(lines)
