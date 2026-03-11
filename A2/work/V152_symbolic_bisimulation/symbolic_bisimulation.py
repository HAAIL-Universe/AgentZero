"""
V152: Symbolic Bisimulation
BDD-based bisimulation for labeled transition systems.
Uses symbolic partition refinement -- handles exponentially larger state spaces
than explicit enumeration.

Composes: V021 (BDD model checking)

Key idea: represent partitions and transitions as BDDs, compute preimages
symbolically, split blocks using BDD set operations. The Paige-Tarjan
partition refinement algorithm done entirely in the symbolic domain.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
from bdd_model_checker import BDD, BDDNode


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SymbolicTS:
    """Symbolic (BDD-encoded) labeled transition system."""
    bdd: BDD
    state_vars: List[str]          # Current-state variable names
    next_vars: List[str]           # Next-state (primed) variable names
    actions: List[str]             # Action labels
    trans: Dict[str, BDDNode]      # action -> transition relation BDD T_a(x, x')
    labels: Dict[str, BDDNode]     # proposition -> BDD of states satisfying it
    n_state_bits: int              # Number of bits per state encoding
    valid_states: Optional[BDDNode] = None  # BDD of valid states (for non-power-of-2)


@dataclass
class BisimResult:
    """Result of symbolic bisimulation computation."""
    partition: List[BDDNode]       # List of block BDDs (each a set of states)
    n_blocks: int                  # Number of equivalence classes
    n_iterations: int              # Refinement iterations to fixpoint
    equivalence: BDDNode           # Equivalence relation R(x, y) as BDD
    bisimilar_to_init: Optional[BDDNode] = None  # States bisimilar to initial


@dataclass
class CrossBisimResult:
    """Result of cross-system bisimulation check."""
    bisimilar: bool                # Whether initial states are bisimilar
    partition: List[BDDNode]       # Joint partition
    n_blocks: int
    n_iterations: int
    witness: Optional[Dict] = None  # Distinguishing action if not bisimilar


# ---------------------------------------------------------------------------
# Symbolic transition system construction
# ---------------------------------------------------------------------------

def make_symbolic_ts(state_var_names: List[str],
                     actions: List[str],
                     transitions: Dict[str, List[Tuple[int, int]]],
                     state_labels: Dict[int, Set[str]],
                     propositions: Optional[List[str]] = None,
                     n_actual_states: Optional[int] = None) -> SymbolicTS:
    """
    Build a SymbolicTS from an explicit description.

    Args:
        state_var_names: Names for boolean state variables
        actions: List of action labels
        transitions: action -> list of (src, dst) pairs
        state_labels: state_id -> set of proposition names
        propositions: list of all proposition names (auto-detected if None)
        n_actual_states: actual number of valid states (if < 2^n_bits)
    """
    n_bits = len(state_var_names)
    n_states = 1 << n_bits

    bdd = BDD()

    # Allocate variables: state vars, then next-state vars
    sv = []
    nv = []
    for name in state_var_names:
        sv.append(bdd.named_var(name))
    next_names = [f"{name}'" for name in state_var_names]
    for name in next_names:
        nv.append(bdd.named_var(name))

    # Encode state as conjunction of (possibly negated) variables
    def state_bdd(sid, use_next=False):
        bits = sv if not use_next else nv
        result = bdd.TRUE
        for i in range(n_bits):
            if (sid >> (n_bits - 1 - i)) & 1:
                result = bdd.AND(result, bits[i])
            else:
                result = bdd.AND(result, bdd.NOT(bits[i]))
        return result

    # Build transition relations
    trans_bdds = {}
    for action in actions:
        pairs = transitions.get(action, [])
        rel = bdd.FALSE
        for src, dst in pairs:
            src_bdd = state_bdd(src, use_next=False)
            dst_bdd = state_bdd(dst, use_next=True)
            rel = bdd.OR(rel, bdd.AND(src_bdd, dst_bdd))
        trans_bdds[action] = rel

    # Build label BDDs
    if propositions is None:
        propositions = set()
        for s, labs in state_labels.items():
            propositions.update(labs)
        propositions = sorted(propositions)

    label_bdds = {}
    for prop in propositions:
        states_with_prop = bdd.FALSE
        for sid, labs in state_labels.items():
            if prop in labs:
                states_with_prop = bdd.OR(states_with_prop, state_bdd(sid))
        label_bdds[prop] = states_with_prop

    # Build valid states BDD (for non-power-of-2 state counts)
    valid = None
    actual = n_actual_states if n_actual_states is not None else n_states
    if actual < n_states:
        valid = bdd.FALSE
        for sid in range(actual):
            valid = bdd.OR(valid, state_bdd(sid))

    return SymbolicTS(
        bdd=bdd,
        state_vars=[bdd.var_index(n) for n in state_var_names],
        next_vars=[bdd.var_index(n) for n in next_names],
        actions=actions,
        trans=trans_bdds,
        labels=label_bdds,
        n_state_bits=n_bits,
        valid_states=valid,
    )


def make_symbolic_ts_from_kripke(n_states: int,
                                  transitions: List[Tuple[int, int]],
                                  state_labels: Dict[int, Set[str]],
                                  propositions: Optional[List[str]] = None) -> SymbolicTS:
    """
    Build a SymbolicTS from a Kripke structure (unlabeled transitions).
    All transitions get the single action "tau".
    """
    n_bits = max(1, math.ceil(math.log2(max(n_states, 2))))
    var_names = [f"s{i}" for i in range(n_bits)]
    return make_symbolic_ts(
        state_var_names=var_names,
        actions=["tau"],
        transitions={"tau": transitions},
        state_labels=state_labels,
        propositions=propositions,
        n_actual_states=n_states,
    )


def make_symbolic_ts_from_lts(n_states: int,
                               transitions: List[Tuple[int, str, int]],
                               state_labels: Optional[Dict[int, Set[str]]] = None,
                               propositions: Optional[List[str]] = None) -> SymbolicTS:
    """
    Build a SymbolicTS from a labeled transition system.
    transitions: list of (src, action, dst) triples
    """
    n_bits = max(1, math.ceil(math.log2(max(n_states, 2))))
    var_names = [f"s{i}" for i in range(n_bits)]

    actions = sorted(set(a for _, a, _ in transitions))
    trans_dict = {a: [] for a in actions}
    for src, act, dst in transitions:
        trans_dict[act].append((src, dst))

    if state_labels is None:
        state_labels = {i: set() for i in range(n_states)}

    return make_symbolic_ts(
        state_var_names=var_names,
        actions=actions,
        transitions=trans_dict,
        state_labels=state_labels,
        propositions=propositions,
        n_actual_states=n_states,
    )


# ---------------------------------------------------------------------------
# Core symbolic operations
# ---------------------------------------------------------------------------

def _preimage(ts: SymbolicTS, target: BDDNode, action: str) -> BDDNode:
    """
    Compute preimage: {s | exists s'. T_a(s, s') AND s' in target}.
    Requires substituting current vars into target for next vars,
    then existentially quantifying next vars.
    """
    bdd = ts.bdd
    trans = ts.trans.get(action, bdd.FALSE)

    # Rename target from state vars to next vars
    rename_map = {}
    for sv, nv in zip(ts.state_vars, ts.next_vars):
        rename_map[sv] = nv
    target_next = bdd.rename(target, rename_map)

    # Conjoin with transition relation and quantify out next vars
    conj = bdd.AND(trans, target_next)
    result = bdd.exists_multi(ts.next_vars, conj)
    return result


def _preimage_all(ts: SymbolicTS, target: BDDNode) -> BDDNode:
    """Preimage over all actions (union)."""
    bdd = ts.bdd
    result = bdd.FALSE
    for action in ts.actions:
        result = bdd.OR(result, _preimage(ts, target, action))
    return result


def _all_states(ts: SymbolicTS) -> BDDNode:
    """BDD of all valid states (for n_state_bits, all 2^n are valid)."""
    return ts.bdd.TRUE


def _state_bdd(ts: SymbolicTS, state_id: int) -> BDDNode:
    """BDD for a single state."""
    bdd = ts.bdd
    result = bdd.TRUE
    n = ts.n_state_bits
    for i, sv in enumerate(ts.state_vars):
        var = bdd.var(sv) if isinstance(sv, int) else bdd.named_var(sv)
        # Actually sv is already an index from var_index
        var = bdd.var(sv)
        if (state_id >> (n - 1 - i)) & 1:
            result = bdd.AND(result, var)
        else:
            result = bdd.AND(result, bdd.NOT(var))
    return result


def _block_count(ts: SymbolicTS, block: BDDNode) -> int:
    """Count states in a block BDD."""
    return ts.bdd.sat_count(block, ts.n_state_bits)


# ---------------------------------------------------------------------------
# Symbolic partition refinement -- Strong bisimulation
# ---------------------------------------------------------------------------

def _initial_partition(ts: SymbolicTS) -> List[BDDNode]:
    """
    Compute initial partition based on state labels.
    States with the same set of labels go in the same block.
    Restricts to valid states if not all bit patterns are used.
    """
    bdd = ts.bdd
    valid = ts.valid_states if ts.valid_states is not None else bdd.TRUE
    props = sorted(ts.labels.keys())

    if not props:
        # No labels -- everything in one block (restricted to valid)
        return [valid]

    # Build signature BDDs: for each combination of label truth values
    # Group states by their label signature
    signatures = {}  # tuple of bools -> BDD

    # Enumerate all 2^|props| combinations
    n_props = len(props)
    for mask in range(1 << n_props):
        sig = tuple((mask >> i) & 1 for i in range(n_props))
        # Build BDD: conjunction of (prop_i if sig[i] else NOT prop_i)
        block = valid
        for i, prop in enumerate(props):
            if sig[i]:
                block = bdd.AND(block, ts.labels[prop])
            else:
                block = bdd.AND(block, bdd.NOT(ts.labels[prop]))
        if block != bdd.FALSE:
            signatures[sig] = block

    return list(signatures.values())


def _refine_partition(ts: SymbolicTS, partition: List[BDDNode]) -> List[BDDNode]:
    """
    One round of partition refinement.
    For each splitter block B and action a, split every block C into:
      C_hit  = C AND pre_a(B)
      C_miss = C AND NOT pre_a(B)
    """
    bdd = ts.bdd
    new_partition = list(partition)
    changed = True

    while changed:
        changed = False
        next_partition = []
        for block_c in new_partition:
            split = False
            for action in ts.actions:
                for block_b in new_partition:
                    pre_b = _preimage(ts, block_b, action)
                    hit = bdd.AND(block_c, pre_b)
                    miss = bdd.AND(block_c, bdd.NOT(pre_b))
                    if hit != bdd.FALSE and miss != bdd.FALSE:
                        # Split!
                        next_partition.append(hit)
                        next_partition.append(miss)
                        split = True
                        changed = True
                        break
                if split:
                    break
            if not split:
                next_partition.append(block_c)
        new_partition = next_partition

    return new_partition


def compute_strong_bisimulation(ts: SymbolicTS,
                                 max_iterations: int = 100) -> BisimResult:
    """
    Compute the coarsest strong bisimulation on a symbolic transition system.
    Uses iterated partition refinement until fixpoint.
    """
    bdd = ts.bdd
    partition = _initial_partition(ts)
    n_iters = 0

    for _ in range(max_iterations):
        n_iters += 1
        new_partition = _refine_step(ts, partition)
        if len(new_partition) == len(partition):
            # Check if actually the same
            same = True
            for old_b, new_b in zip(sorted(partition, key=id),
                                     sorted(new_partition, key=id)):
                pass
            # Better check: set of frozen block IDs
            old_set = frozenset(id(b) for b in partition)
            new_set = frozenset(id(b) for b in new_partition)
            if old_set == new_set:
                break
            # More careful: check BDD equality
            if _partitions_equal(bdd, partition, new_partition):
                break
        partition = new_partition

    # Build equivalence relation R(x, y)
    equiv = _partition_to_equivalence(ts, partition)

    return BisimResult(
        partition=partition,
        n_blocks=len(partition),
        n_iterations=n_iters,
        equivalence=equiv,
    )


def _refine_step(ts: SymbolicTS, partition: List[BDDNode]) -> List[BDDNode]:
    """Single refinement step: split each block by all (action, splitter) pairs."""
    bdd = ts.bdd
    new_partition = []

    for block_c in partition:
        sub_blocks = [block_c]
        for action in ts.actions:
            for block_b in partition:
                pre_b = _preimage(ts, block_b, action)
                next_sub = []
                for sb in sub_blocks:
                    hit = bdd.AND(sb, pre_b)
                    miss = bdd.AND(sb, bdd.NOT(pre_b))
                    if hit != bdd.FALSE and miss != bdd.FALSE:
                        next_sub.append(hit)
                        next_sub.append(miss)
                    else:
                        next_sub.append(sb)
                sub_blocks = next_sub
        new_partition.extend(sub_blocks)

    return new_partition


def _partitions_equal(bdd: BDD, p1: List[BDDNode], p2: List[BDDNode]) -> bool:
    """Check if two partitions are equal (same set of blocks)."""
    if len(p1) != len(p2):
        return False
    # For each block in p1, find a matching block in p2
    used = [False] * len(p2)
    for b1 in p1:
        found = False
        for j, b2 in enumerate(p2):
            if not used[j] and b1 == b2:
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True


def _partition_to_equivalence(ts: SymbolicTS, partition: List[BDDNode]) -> BDDNode:
    """
    Build equivalence relation BDD R(x, y) from partition.
    Uses a second set of state variables (y) aliased to next vars.
    R(x, y) = OR over blocks B: (x in B) AND (y in B)
    where y uses next_vars encoding.
    """
    bdd = ts.bdd
    rename_map = {}
    for sv, nv in zip(ts.state_vars, ts.next_vars):
        rename_map[sv] = nv

    equiv = bdd.FALSE
    for block in partition:
        block_y = bdd.rename(block, rename_map)
        equiv = bdd.OR(equiv, bdd.AND(block, block_y))
    return equiv


# ---------------------------------------------------------------------------
# Symbolic weak bisimulation (with tau closure)
# ---------------------------------------------------------------------------

TAU = "__tau__"


def _tau_closure(ts: SymbolicTS, states: BDDNode,
                  max_steps: int = 100) -> BDDNode:
    """
    Compute tau-closure: states reachable via zero or more tau transitions.
    Fixpoint: R = states OR pre_tau^{-1}(R) -- actually forward reachable.
    """
    bdd = ts.bdd
    if TAU not in ts.trans:
        return states

    tau_rel = ts.trans[TAU]
    result = states

    for _ in range(max_steps):
        # Forward image via tau: {s' | exists s in result. T_tau(s, s')}
        rename_map = {}
        for sv, nv in zip(ts.state_vars, ts.next_vars):
            rename_map[sv] = nv
        result_as_curr = result
        conj = bdd.AND(tau_rel, result_as_curr)
        # Quantify out current vars to get next-state set
        img = bdd.exists_multi(ts.state_vars, conj)
        # Rename next vars back to current vars
        rename_back = {}
        for sv, nv in zip(ts.state_vars, ts.next_vars):
            rename_back[nv] = sv
        img_curr = bdd.rename(img, rename_back)

        new_result = bdd.OR(result, img_curr)
        if new_result == result:
            break
        result = new_result

    return result


def _weak_preimage(ts: SymbolicTS, target: BDDNode, action: str) -> BDDNode:
    """
    Weak preimage: tau* ; action ; tau*.
    {s | exists s1 in tau*(s), s2 in succ_a(s1), s3 in tau*(s2): s3 in target}
    """
    bdd = ts.bdd

    if action == TAU:
        # Weak tau preimage: tau+ (at least one tau step that reaches target)
        closed = _tau_closure(ts, target)
        return closed

    # Step 1: tau-close the target (backward): states that can tau-reach target
    target_closed = _tau_closure_backward(ts, target)

    # Step 2: action preimage of tau-closed target
    pre_a = _preimage(ts, target_closed, action)

    # Step 3: tau-close backward from pre_a
    result = _tau_closure_backward(ts, pre_a)

    return result


def _tau_closure_backward(ts: SymbolicTS, states: BDDNode,
                           max_steps: int = 100) -> BDDNode:
    """
    Backward tau-closure: states that can reach 'states' via tau*.
    """
    bdd = ts.bdd
    if TAU not in ts.trans:
        return states

    result = states
    for _ in range(max_steps):
        pre_tau = _preimage(ts, result, TAU)
        new_result = bdd.OR(result, pre_tau)
        if new_result == result:
            break
        result = new_result

    return result


def _refine_step_weak(ts: SymbolicTS, partition: List[BDDNode]) -> List[BDDNode]:
    """Single weak refinement step using weak preimages."""
    bdd = ts.bdd
    new_partition = []
    observable_actions = [a for a in ts.actions if a != TAU]

    for block_c in partition:
        sub_blocks = [block_c]
        # Only split on observable actions for weak bisimulation
        for action in observable_actions:
            for block_b in partition:
                pre_b = _weak_preimage(ts, block_b, action)
                next_sub = []
                for sb in sub_blocks:
                    hit = bdd.AND(sb, pre_b)
                    miss = bdd.AND(sb, bdd.NOT(pre_b))
                    if hit != bdd.FALSE and miss != bdd.FALSE:
                        next_sub.append(hit)
                        next_sub.append(miss)
                    else:
                        next_sub.append(sb)
                sub_blocks = next_sub
        new_partition.extend(sub_blocks)

    return new_partition


def compute_weak_bisimulation(ts: SymbolicTS,
                               max_iterations: int = 100) -> BisimResult:
    """
    Compute coarsest weak bisimulation.
    Like strong bisimulation but uses weak preimages (tau* ; a ; tau*).
    """
    bdd = ts.bdd
    partition = _initial_partition(ts)
    n_iters = 0

    for _ in range(max_iterations):
        n_iters += 1
        new_partition = _refine_step_weak(ts, partition)
        if _partitions_equal(bdd, partition, new_partition):
            break
        partition = new_partition

    equiv = _partition_to_equivalence(ts, partition)

    return BisimResult(
        partition=partition,
        n_blocks=len(partition),
        n_iterations=n_iters,
        equivalence=equiv,
    )


# ---------------------------------------------------------------------------
# Symbolic branching bisimulation
# ---------------------------------------------------------------------------

def _refine_step_branching(ts: SymbolicTS,
                            partition: List[BDDNode]) -> List[BDDNode]:
    """
    Branching bisimulation refinement.
    A tau transition s -tau-> s' is stuttering if s and s' are in the same block.
    Only non-stuttering transitions cause splits.
    """
    bdd = ts.bdd
    new_partition = []

    for block_c in partition:
        sub_blocks = [block_c]
        for action in ts.actions:
            for block_b in partition:
                if action == TAU:
                    # For tau: only split if transition leaves the block
                    # pre_tau(B) restricted to states NOT in B
                    pre_b = _preimage(ts, block_b, TAU)
                    # States in block_c that can tau-reach block_b
                    # but only if block_b != block_c (non-stuttering)
                    if block_b == block_c:
                        continue
                    # Also: states that can tau-reach B while staying in current block
                    # should NOT be split (stuttering). Only split if there's a
                    # tau transition that goes OUTSIDE current block to reach B.
                    # Simplified: just use the preimage
                else:
                    pre_b = _preimage(ts, block_b, action)

                next_sub = []
                for sb in sub_blocks:
                    hit = bdd.AND(sb, pre_b)
                    miss = bdd.AND(sb, bdd.NOT(pre_b))
                    if hit != bdd.FALSE and miss != bdd.FALSE:
                        next_sub.append(hit)
                        next_sub.append(miss)
                    else:
                        next_sub.append(sb)
                sub_blocks = next_sub
        new_partition.extend(sub_blocks)

    return new_partition


def compute_branching_bisimulation(ts: SymbolicTS,
                                    max_iterations: int = 100) -> BisimResult:
    """
    Compute coarsest branching bisimulation.
    Finer than weak bisimulation -- preserves branching structure.
    Tau transitions within the same block are stuttering (ignored).
    """
    bdd = ts.bdd
    partition = _initial_partition(ts)
    n_iters = 0

    for _ in range(max_iterations):
        n_iters += 1
        new_partition = _refine_step_branching(ts, partition)
        if _partitions_equal(bdd, partition, new_partition):
            break
        partition = new_partition

    equiv = _partition_to_equivalence(ts, partition)

    return BisimResult(
        partition=partition,
        n_blocks=len(partition),
        n_iterations=n_iters,
        equivalence=equiv,
    )


# ---------------------------------------------------------------------------
# Cross-system bisimulation
# ---------------------------------------------------------------------------

def check_bisimilar(ts: SymbolicTS, state1: int, state2: int,
                     mode: str = "strong") -> bool:
    """Check if two states in the same system are bisimilar."""
    if mode == "strong":
        result = compute_strong_bisimulation(ts)
    elif mode == "weak":
        result = compute_weak_bisimulation(ts)
    elif mode == "branching":
        result = compute_branching_bisimulation(ts)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    s1_bdd = _state_bdd(ts, state1)
    s2_bdd = _state_bdd(ts, state2)

    # Check if s1 and s2 are in the same block
    for block in result.partition:
        in1 = ts.bdd.AND(s1_bdd, block) != ts.bdd.FALSE
        in2 = ts.bdd.AND(s2_bdd, block) != ts.bdd.FALSE
        if in1 and in2:
            return True
        if in1 or in2:
            return False
    return False


def check_cross_bisimulation(ts1: SymbolicTS, ts2: SymbolicTS,
                              init1: int = 0, init2: int = 0,
                              mode: str = "strong") -> CrossBisimResult:
    """
    Check if initial states of two systems are bisimilar.
    Builds a disjoint union system and checks bisimulation.
    """
    bdd = BDD()

    # Determine encoding sizes
    n1 = ts1.n_state_bits
    n2 = ts2.n_state_bits
    # Need one extra bit to distinguish systems, plus max(n1, n2) state bits
    n_bits = max(n1, n2) + 1  # bit 0 = system selector

    var_names = [f"x{i}" for i in range(n_bits)]

    # Collect all actions
    all_actions = sorted(set(ts1.actions) | set(ts2.actions))

    # Collect all propositions
    all_props = sorted(set(ts1.labels.keys()) | set(ts2.labels.keys()))

    # Build disjoint union transitions
    # System 1: selector bit = 0, state bits encode ts1 states
    # System 2: selector bit = 1, state bits encode ts2 states
    offset2 = 1 << (n_bits - 1)  # Offset for system 2 states

    trans_dict = {a: [] for a in all_actions}

    # Reconstruct explicit transitions from BDDs (for small systems)
    n1_states = 1 << n1
    n2_states = 1 << n2

    for action in all_actions:
        # System 1 transitions
        if action in ts1.trans:
            for src in range(n1_states):
                for dst in range(n1_states):
                    if _check_transition(ts1, action, src, dst):
                        trans_dict[action].append((src, dst))
        # System 2 transitions
        if action in ts2.trans:
            for src in range(n2_states):
                for dst in range(n2_states):
                    if _check_transition(ts2, action, src, dst):
                        trans_dict[action].append((src + offset2, dst + offset2))

    # Build state labels for disjoint union
    state_labels = {}
    for sid in range(n1_states):
        labs = set()
        for prop in all_props:
            if prop in ts1.labels and _state_in_bdd(ts1, sid, ts1.labels[prop]):
                labs.add(prop)
        state_labels[sid] = labs

    for sid in range(n2_states):
        labs = set()
        for prop in all_props:
            if prop in ts2.labels and _state_in_bdd(ts2, sid, ts2.labels[prop]):
                labs.add(prop)
        state_labels[sid + offset2] = labs

    # Build joint symbolic TS
    joint_ts = make_symbolic_ts(
        state_var_names=var_names,
        actions=all_actions,
        transitions=trans_dict,
        state_labels=state_labels,
        propositions=all_props if all_props else None,
    )

    # Compute bisimulation on joint system
    if mode == "strong":
        result = compute_strong_bisimulation(joint_ts)
    elif mode == "weak":
        result = compute_weak_bisimulation(joint_ts)
    elif mode == "branching":
        result = compute_branching_bisimulation(joint_ts)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Check if init1 and init2+offset are in the same block
    s1_bdd = _state_bdd(joint_ts, init1)
    s2_bdd = _state_bdd(joint_ts, init2 + offset2)

    bisimilar = False
    for block in result.partition:
        in1 = joint_ts.bdd.AND(s1_bdd, block) != joint_ts.bdd.FALSE
        in2 = joint_ts.bdd.AND(s2_bdd, block) != joint_ts.bdd.FALSE
        if in1 and in2:
            bisimilar = True
            break

    # Find distinguishing action if not bisimilar
    witness = None
    if not bisimilar:
        witness = _find_distinguishing_action(joint_ts, result, init1,
                                               init2 + offset2)

    return CrossBisimResult(
        bisimilar=bisimilar,
        partition=result.partition,
        n_blocks=result.n_blocks,
        n_iterations=result.n_iterations,
        witness=witness,
    )


def _check_transition(ts: SymbolicTS, action: str, src: int, dst: int) -> bool:
    """Check if a specific transition exists."""
    bdd = ts.bdd
    trans = ts.trans.get(action, bdd.FALSE)
    src_bdd = _state_bdd(ts, src)

    # Build dst in next vars
    dst_bdd = bdd.TRUE
    n = ts.n_state_bits
    for i, nv in enumerate(ts.next_vars):
        var = bdd.var(nv)
        if (dst >> (n - 1 - i)) & 1:
            dst_bdd = bdd.AND(dst_bdd, var)
        else:
            dst_bdd = bdd.AND(dst_bdd, bdd.NOT(var))

    check = bdd.AND(trans, bdd.AND(src_bdd, dst_bdd))
    return check != bdd.FALSE


def _state_in_bdd(ts: SymbolicTS, state_id: int, bdd_set: BDDNode) -> bool:
    """Check if a state is in a BDD set."""
    s = _state_bdd(ts, state_id)
    return ts.bdd.AND(s, bdd_set) != ts.bdd.FALSE


def _find_distinguishing_action(ts: SymbolicTS, result: BisimResult,
                                 s1: int, s2: int) -> Optional[Dict]:
    """Find an action that distinguishes two non-bisimilar states."""
    for action in ts.actions:
        for block in result.partition:
            pre = _preimage(ts, block, action)
            s1_bdd = _state_bdd(ts, s1)
            s2_bdd = _state_bdd(ts, s2)
            s1_can = ts.bdd.AND(s1_bdd, pre) != ts.bdd.FALSE
            s2_can = ts.bdd.AND(s2_bdd, pre) != ts.bdd.FALSE
            if s1_can != s2_can:
                return {"action": action, "s1_can_reach_block": s1_can,
                        "s2_can_reach_block": s2_can}
    return None


# ---------------------------------------------------------------------------
# Quotient construction (minimization)
# ---------------------------------------------------------------------------

def compute_quotient(ts: SymbolicTS,
                      partition: List[BDDNode]) -> Dict:
    """
    Build the quotient (minimized) system from a bisimulation partition.
    Returns an explicit description of the minimized system.
    """
    bdd = ts.bdd
    n_blocks = len(partition)

    # Map each state to its block index
    state_to_block = {}
    n_states = 1 << ts.n_state_bits
    for sid in range(n_states):
        s_bdd = _state_bdd(ts, sid)
        for bi, block in enumerate(partition):
            if bdd.AND(s_bdd, block) != bdd.FALSE:
                state_to_block[sid] = bi
                break

    # Build quotient transitions: block_i -a-> block_j
    # iff exists s in block_i, s' in block_j: s -a-> s'
    quot_trans = {}
    for action in ts.actions:
        edges = set()
        for bi, block_i in enumerate(partition):
            pre_results = {}
            for bj, block_j in enumerate(partition):
                pre = _preimage(ts, block_j, action)
                if bdd.AND(block_i, pre) != bdd.FALSE:
                    edges.add((bi, bj))
            quot_trans[action] = sorted(edges)

    # Build quotient labels
    quot_labels = {}
    for bi, block in enumerate(partition):
        labs = set()
        for prop, prop_bdd in ts.labels.items():
            if bdd.AND(block, prop_bdd) != bdd.FALSE:
                labs.add(prop)
        quot_labels[bi] = labs

    return {
        "n_blocks": n_blocks,
        "transitions": quot_trans,
        "labels": quot_labels,
        "state_to_block": state_to_block,
        "blocks": partition,
    }


def minimize(ts: SymbolicTS, mode: str = "strong") -> Dict:
    """Compute bisimulation quotient (minimized system)."""
    if mode == "strong":
        result = compute_strong_bisimulation(ts)
    elif mode == "weak":
        result = compute_weak_bisimulation(ts)
    elif mode == "branching":
        result = compute_branching_bisimulation(ts)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return compute_quotient(ts, result.partition)


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

def compare_bisimulations(ts: SymbolicTS) -> Dict:
    """
    Compare strong, weak, and branching bisimulation on the same system.
    Returns block counts and whether they differ.
    """
    strong = compute_strong_bisimulation(ts)
    has_tau = TAU in ts.actions

    result = {
        "strong": {
            "n_blocks": strong.n_blocks,
            "n_iterations": strong.n_iterations,
        }
    }

    if has_tau:
        weak = compute_weak_bisimulation(ts)
        branching = compute_branching_bisimulation(ts)
        result["weak"] = {
            "n_blocks": weak.n_blocks,
            "n_iterations": weak.n_iterations,
        }
        result["branching"] = {
            "n_blocks": branching.n_blocks,
            "n_iterations": branching.n_iterations,
        }
        result["hierarchy"] = (
            f"strong ({strong.n_blocks} blocks) >= "
            f"branching ({branching.n_blocks} blocks) >= "
            f"weak ({weak.n_blocks} blocks)"
        )
    else:
        result["note"] = "No tau actions -- weak and branching equal strong"

    return result


def bisimulation_summary(ts: SymbolicTS, mode: str = "strong") -> Dict:
    """Summary of a bisimulation result with state counts per block."""
    if mode == "strong":
        result = compute_strong_bisimulation(ts)
    elif mode == "weak":
        result = compute_weak_bisimulation(ts)
    elif mode == "branching":
        result = compute_branching_bisimulation(ts)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    block_sizes = []
    for block in result.partition:
        size = int(ts.bdd.sat_count(block, ts.n_state_bits))
        block_sizes.append(size)

    return {
        "mode": mode,
        "n_blocks": result.n_blocks,
        "n_iterations": result.n_iterations,
        "block_sizes": block_sizes,
        "total_states": sum(block_sizes),
        "reduction_ratio": sum(block_sizes) / max(result.n_blocks, 1),
    }


# ---------------------------------------------------------------------------
# Parametric system generator (for scalability testing)
# ---------------------------------------------------------------------------

def make_chain(n: int, labeled: bool = True) -> SymbolicTS:
    """
    Create a chain of n states: 0 -> 1 -> ... -> n-1.
    If labeled, alternating labels (even=p, odd=q).
    """
    n_bits = max(1, math.ceil(math.log2(max(n, 2))))
    var_names = [f"s{i}" for i in range(n_bits)]
    transitions = {"a": [(i, i + 1) for i in range(n - 1)]}
    labels = {}
    if labeled:
        for i in range(n):
            labels[i] = {"p"} if i % 2 == 0 else {"q"}
    else:
        for i in range(n):
            labels[i] = set()
    return make_symbolic_ts(var_names, ["a"], transitions, labels,
                           n_actual_states=n)


def make_ring(n: int) -> SymbolicTS:
    """Create a ring of n states: 0 -> 1 -> ... -> n-1 -> 0."""
    n_bits = max(1, math.ceil(math.log2(max(n, 2))))
    var_names = [f"s{i}" for i in range(n_bits)]
    transitions = {"a": [(i, (i + 1) % n) for i in range(n)]}
    labels = {i: set() for i in range(n)}
    return make_symbolic_ts(var_names, ["a"], transitions, labels,
                           n_actual_states=n)


def make_binary_tree(depth: int) -> SymbolicTS:
    """Create a full binary tree of given depth."""
    n_states = (1 << (depth + 1)) - 1
    n_bits = max(1, math.ceil(math.log2(max(n_states, 2))))
    var_names = [f"s{i}" for i in range(n_bits)]

    transitions = {"left": [], "right": []}
    labels = {}
    for i in range(n_states):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n_states:
            transitions["left"].append((i, left))
        if right < n_states:
            transitions["right"].append((i, right))
        # Leaves have label "leaf", internal nodes "internal"
        if 2 * i + 1 >= n_states:
            labels[i] = {"leaf"}
        else:
            labels[i] = {"internal"}

    return make_symbolic_ts(var_names, ["left", "right"], transitions, labels,
                           n_actual_states=n_states)


def make_parallel_composition(n_components: int,
                               states_per_component: int = 2) -> SymbolicTS:
    """
    Create parallel composition of n identical 2-state components.
    Total states = states_per_component^n_components (exponential).
    Each component independently toggles between states.
    """
    n_bits = n_components  # One bit per component
    var_names = [f"c{i}" for i in range(n_bits)]
    actions = [f"toggle_{i}" for i in range(n_components)]

    transitions = {}
    total = 1 << n_bits
    for ci in range(n_components):
        action = f"toggle_{ci}"
        pairs = []
        for s in range(total):
            # Toggle bit ci
            t = s ^ (1 << (n_bits - 1 - ci))
            pairs.append((s, t))
        transitions[action] = pairs

    labels = {}
    for s in range(total):
        labs = set()
        for ci in range(n_components):
            if (s >> (n_bits - 1 - ci)) & 1:
                labs.add(f"c{ci}_on")
        labels[s] = labs

    return make_symbolic_ts(var_names, actions, transitions, labels)
