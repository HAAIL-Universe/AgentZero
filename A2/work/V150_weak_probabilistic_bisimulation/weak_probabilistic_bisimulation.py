"""
V150: Weak Probabilistic Bisimulation

Behavioral equivalence for probabilistic systems that abstracts away internal
(tau/silent) transitions. Two states are weakly bisimilar if they produce the
same observable behavior regardless of internal steps.

Key concepts:
- Internal (tau) transitions: unobservable, may be abstracted away
- Weak transition: zero or more tau steps, one observable step, zero or more tau steps
- Weak bisimulation: partition refinement on weak transition distributions
- Branching bisimulation: preserves the branching structure (stronger than weak)

Composes: V148 (probabilistic bisimulation) + V065 (Markov chains) + V067 (labeled MCs)
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, FrozenSet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V148_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from probabilistic_bisimulation import (
    BisimVerdict, BisimResult, DistanceResult,
    compute_bisimulation, check_bisimilar, bisimulation_quotient,
    compute_bisimulation_distance, verify_bisimulation_smt,
)
from pctl_model_check import LabeledMC, make_labeled_mc
from markov_chain import MarkovChain, make_chain


# ---- Data Structures ----

TAU = "__tau__"  # sentinel label for internal transitions


class WeakBisimVerdict(Enum):
    WEAKLY_BISIMILAR = "weakly_bisimilar"
    NOT_WEAKLY_BISIMILAR = "not_weakly_bisimilar"
    BRANCHING_BISIMILAR = "branching_bisimilar"
    NOT_BRANCHING_BISIMILAR = "not_branching_bisimilar"


@dataclass
class WeakBisimResult:
    verdict: WeakBisimVerdict
    partition: Optional[List[Set[int]]] = None
    relation: Optional[Set[FrozenSet[int]]] = None
    witness: Optional[str] = None
    quotient: Optional['LabeledProbTS'] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class LabeledProbTS:
    """Labeled Probabilistic Transition System with internal transitions.

    Unlike a plain LabeledMC where transitions are unlabeled (just probabilities),
    here each state has a set of available *actions* (including tau), and each
    action leads to a probability distribution over successor states.

    Attributes:
        n_states: number of states
        actions: Dict[int, Dict[str, List[Tuple[int, float]]]]
            actions[s][a] = [(t, p), ...] -- action a at state s leads to t with prob p
        state_labels: Dict[int, Set[str]] -- atomic propositions true at each state
        state_names: Optional[List[str]] -- human-readable names
    """
    n_states: int
    actions: Dict[int, Dict[str, List[Tuple[int, float]]]]
    state_labels: Dict[int, Set[str]]
    state_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.state_names is None:
            self.state_names = [f"s{i}" for i in range(self.n_states)]

    def has_tau(self, state: int) -> bool:
        """Check if state has any tau transitions."""
        return TAU in self.actions.get(state, {})

    def tau_successors(self, state: int) -> List[Tuple[int, float]]:
        """Get tau-transition successors of a state."""
        return self.actions.get(state, {}).get(TAU, [])

    def observable_actions(self, state: int) -> Set[str]:
        """Get non-tau actions available at a state."""
        return {a for a in self.actions.get(state, {}) if a != TAU}

    def action_distribution(self, state: int, action: str) -> List[Tuple[int, float]]:
        """Get probability distribution for an action at a state."""
        return self.actions.get(state, {}).get(action, [])


def make_labeled_prob_ts(
    n_states: int,
    actions: Dict[int, Dict[str, List[Tuple[int, float]]]],
    state_labels: Dict[int, Set[str]],
    state_names: Optional[List[str]] = None,
) -> LabeledProbTS:
    """Create a labeled probabilistic transition system."""
    # Validate
    for s in range(n_states):
        if s not in state_labels:
            state_labels[s] = set()
        for a, dist in actions.get(s, {}).items():
            total = sum(p for _, p in dist)
            if abs(total - 1.0) > 1e-6 and len(dist) > 0:
                raise ValueError(
                    f"Action '{a}' at state {s}: probabilities sum to {total}, not 1.0"
                )
    return LabeledProbTS(n_states, actions, state_labels, state_names)


# ---- Tau Closure (Weak Transition Computation) ----

def _tau_closure(ts: LabeledProbTS, state: int, max_iter: int = 1000) -> Dict[int, float]:
    """Compute tau* reachability distribution from a state.

    Returns distribution over states reachable by zero or more tau transitions.
    Uses iterative fixpoint computation (matrix power series for tau transitions).

    For deterministic tau transitions, this gives probability 1.0 to the
    tau-reachable stable state. For probabilistic tau, it distributes mass.

    NOTE: This assumes tau transitions eventually terminate (no tau-loops with
    probability 1). If there's a tau-cycle with total probability 1, the
    computation may not converge.
    """
    # Build tau transition matrix (sparse)
    tau_trans = {}  # state -> [(next, prob)]
    for s in range(ts.n_states):
        tau_trans[s] = ts.tau_successors(s)

    # Start with point distribution at state
    dist = {state: 1.0}

    for _ in range(max_iter):
        new_dist = {}
        changed = False

        for s, p_s in dist.items():
            tau_succs = tau_trans.get(s, [])
            if not tau_succs:
                # s is tau-stable: keep mass at s
                new_dist[s] = new_dist.get(s, 0.0) + p_s
            else:
                # s has tau transitions: distribute mass
                # But we also need to account for self-staying probability
                # In a DTMC with tau, each tau step replaces the state
                for t, p_t in tau_succs:
                    new_dist[t] = new_dist.get(t, 0.0) + p_s * p_t

        # Check convergence
        max_diff = 0.0
        all_keys = set(dist.keys()) | set(new_dist.keys())
        for k in all_keys:
            max_diff = max(max_diff, abs(dist.get(k, 0.0) - new_dist.get(k, 0.0)))

        dist = new_dist
        if max_diff < 1e-10:
            break

    return dist


def _weak_transition(
    ts: LabeledProbTS, state: int, action: str
) -> Dict[int, float]:
    """Compute weak transition distribution: tau* ; action ; tau*

    For observable action a:
        s =a=> mu  iff  s --tau*--> s' --a--> mu' --tau*--> mu

    Returns combined distribution over final states.
    """
    if action == TAU:
        # Weak tau is just tau*
        return _tau_closure(ts, state)

    result = {}

    # Step 1: tau* from state
    pre_dist = _tau_closure(ts, state)

    # Step 2: for each reachable intermediate state, take action a
    for s_mid, p_mid in pre_dist.items():
        a_dist = ts.action_distribution(s_mid, action)
        if not a_dist:
            continue  # action not available at this intermediate state

        # Step 3: for each post-action state, apply tau*
        for s_post, p_action in a_dist:
            post_dist = _tau_closure(ts, s_post)
            for s_final, p_final in post_dist.items():
                prob = p_mid * p_action * p_final
                if prob > 1e-15:
                    result[s_final] = result.get(s_final, 0.0) + prob

    return result


def _stable_states(ts: LabeledProbTS) -> Set[int]:
    """Find states with no tau transitions (tau-stable)."""
    stable = set()
    for s in range(ts.n_states):
        if not ts.has_tau(s):
            stable.add(s)
    return stable


# ---- Weak Probabilistic Bisimulation ----

def _block_prob(dist: Dict[int, float], block: Set[int]) -> float:
    """Sum probability mass landing in a given block."""
    return sum(dist.get(s, 0.0) for s in block)


def compute_weak_bisimulation(
    ts: LabeledProbTS, max_iter: int = 1000
) -> WeakBisimResult:
    """Compute weak probabilistic bisimulation via partition refinement.

    Two states s, t are weakly bisimilar if:
    1. They have the same labels (atomic propositions)
    2. For every observable action a and every weak bisimulation class C:
       sum_{c in C} s =a=> c  ==  sum_{c in C} t =a=> c
       (weak transition probabilities to each equivalence class match)
    3. The relation is also closed under weak tau transitions

    Uses Larsen-Skou style partition refinement on weak transitions.
    """
    # Initial partition by labels
    label_groups = {}
    for s in range(ts.n_states):
        key = frozenset(ts.state_labels.get(s, set()))
        if key not in label_groups:
            label_groups[key] = set()
        label_groups[key].add(s)

    partition = list(label_groups.values())

    # Precompute all weak transitions
    # weak_trans[s][a] = distribution dict
    weak_trans = {}
    all_obs_actions = set()
    for s in range(ts.n_states):
        weak_trans[s] = {}
        obs_acts = set()
        # Collect observable actions reachable via tau*
        pre_dist = _tau_closure(ts, s)
        for s_mid in pre_dist:
            obs_acts |= ts.observable_actions(s_mid)
        all_obs_actions |= obs_acts
        for a in obs_acts:
            weak_trans[s][a] = _weak_transition(ts, s, a)

    # Partition refinement
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        new_partition = []
        refined = False

        for block in partition:
            if len(block) <= 1:
                new_partition.append(block)
                continue

            # Try to split this block
            sub_blocks = _split_block_weak(block, partition, weak_trans, all_obs_actions)
            if len(sub_blocks) > 1:
                refined = True
            new_partition.extend(sub_blocks)

        partition = new_partition
        if not refined:
            break

    # Build relation
    relation = set()
    for block in partition:
        block_list = sorted(block)
        for i in range(len(block_list)):
            for j in range(i + 1, len(block_list)):
                relation.add(frozenset({block_list[i], block_list[j]}))

    # Check if trivial (each state in own block)
    is_trivial = all(len(b) == 1 for b in partition)

    return WeakBisimResult(
        verdict=WeakBisimVerdict.WEAKLY_BISIMILAR if not is_trivial
                else WeakBisimVerdict.NOT_WEAKLY_BISIMILAR,
        partition=partition,
        relation=relation if relation else None,
        statistics={
            "iterations": iterations,
            "num_blocks": len(partition),
            "reduction_ratio": 1.0 - len(partition) / ts.n_states if ts.n_states > 0 else 0.0,
        },
    )


def _split_block_weak(
    block: Set[int],
    partition: List[Set[int]],
    weak_trans: Dict,
    all_actions: Set[str],
) -> List[Set[int]]:
    """Split a block based on weak transition probabilities to partition classes."""
    # Compute signature for each state
    sigs = {}
    for s in block:
        sig = []
        for bi, target_block in enumerate(partition):
            for a in sorted(all_actions):
                dist = weak_trans.get(s, {}).get(a, {})
                prob = _block_prob(dist, target_block)
                sig.append((bi, a, round(prob, 10)))
        sigs[s] = tuple(sig)

    # Group by signature
    groups = {}
    for s, sig in sigs.items():
        if sig not in groups:
            groups[sig] = set()
        groups[sig].add(s)

    return list(groups.values())


def check_weakly_bisimilar(
    ts: LabeledProbTS, s1: int, s2: int
) -> WeakBisimResult:
    """Check if two states are weakly bisimilar."""
    result = compute_weak_bisimulation(ts)

    for block in result.partition:
        if s1 in block and s2 in block:
            return WeakBisimResult(
                verdict=WeakBisimVerdict.WEAKLY_BISIMILAR,
                partition=result.partition,
                relation=result.relation,
                statistics=result.statistics,
            )

    # Find distinguishing evidence
    witness = f"States {s1} and {s2} are in different equivalence classes"
    labels1 = ts.state_labels.get(s1, set())
    labels2 = ts.state_labels.get(s2, set())
    if labels1 != labels2:
        witness = f"Label mismatch: s{s1}={labels1}, s{s2}={labels2}"

    return WeakBisimResult(
        verdict=WeakBisimVerdict.NOT_WEAKLY_BISIMILAR,
        partition=result.partition,
        witness=witness,
        statistics=result.statistics,
    )


# ---- Branching Bisimulation ----

def compute_branching_bisimulation(
    ts: LabeledProbTS, max_iter: int = 1000
) -> WeakBisimResult:
    """Compute branching probabilistic bisimulation.

    Branching bisimulation is stronger than weak bisimulation.
    It additionally requires that intermediate states along tau paths
    are bisimilar to the source state (preserves branching structure).

    s ~_b t iff:
    1. labels(s) == labels(t)
    2. For every action a and block C:
       - If a != tau: s -a-> mu implies t =a=>_b mu' with mu(C) == mu'(C)
       - If a == tau: s -tau-> mu implies either:
         (a) t -tau-> mu' with mu(C) == mu'(C) for all C, OR
         (b) mu is in the same block as s (stuttering)
    """
    # Initial partition by labels
    label_groups = {}
    for s in range(ts.n_states):
        key = frozenset(ts.state_labels.get(s, set()))
        if key not in label_groups:
            label_groups[key] = set()
        label_groups[key].add(s)

    partition = list(label_groups.values())

    # Build block membership lookup
    def block_of(state, part):
        for i, b in enumerate(part):
            if state in b:
                return i
        return -1

    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        new_partition = []
        refined = False

        for block in partition:
            if len(block) <= 1:
                new_partition.append(block)
                continue

            sub_blocks = _split_block_branching(block, partition, ts)
            if len(sub_blocks) > 1:
                refined = True
            new_partition.extend(sub_blocks)

        partition = new_partition
        if not refined:
            break

    # Build relation
    relation = set()
    for block in partition:
        block_list = sorted(block)
        for i in range(len(block_list)):
            for j in range(i + 1, len(block_list)):
                relation.add(frozenset({block_list[i], block_list[j]}))

    is_trivial = all(len(b) == 1 for b in partition)

    return WeakBisimResult(
        verdict=WeakBisimVerdict.BRANCHING_BISIMILAR if not is_trivial
                else WeakBisimVerdict.NOT_BRANCHING_BISIMILAR,
        partition=partition,
        relation=relation if relation else None,
        statistics={
            "iterations": iterations,
            "num_blocks": len(partition),
            "reduction_ratio": 1.0 - len(partition) / ts.n_states if ts.n_states > 0 else 0.0,
        },
    )


def _split_block_branching(
    block: Set[int],
    partition: List[Set[int]],
    ts: LabeledProbTS,
) -> List[Set[int]]:
    """Split a block for branching bisimulation.

    For branching bisimulation, tau transitions that stay within the same
    block are treated as stuttering (invisible). Only transitions that
    leave the block or observable actions distinguish states.
    """
    # Signature: for each state, compute its distinguishing signature
    sigs = {}
    for s in block:
        sig_parts = []
        all_acts = set(ts.actions.get(s, {}).keys())

        for a in sorted(all_acts):
            dist = ts.action_distribution(s, a)
            if not dist:
                continue

            if a == TAU:
                # For tau: compute block probabilities
                # A tau transition that goes entirely within the same block is stuttering
                block_probs = []
                for bi, target_block in enumerate(partition):
                    prob = sum(p for t, p in dist if t in target_block)
                    block_probs.append((bi, round(prob, 10)))
                # Check if all mass stays in current block
                current_block_idx = None
                for bi, b in enumerate(partition):
                    if s in b:
                        current_block_idx = bi
                        break
                in_block_mass = sum(p for t, p in dist if t in block)
                if abs(in_block_mass - 1.0) < 1e-9:
                    # All tau mass stays in block -- stuttering, ignore
                    continue
                sig_parts.append(("tau", tuple(block_probs)))
            else:
                # Observable action: compute block probabilities
                block_probs = []
                for bi, target_block in enumerate(partition):
                    prob = sum(p for t, p in dist if t in target_block)
                    block_probs.append((bi, round(prob, 10)))
                sig_parts.append((a, tuple(block_probs)))

        sigs[s] = tuple(sorted(sig_parts))

    # Group by signature
    groups = {}
    for s, sig in sigs.items():
        if sig not in groups:
            groups[sig] = set()
        groups[sig].add(s)

    return list(groups.values())


def check_branching_bisimilar(
    ts: LabeledProbTS, s1: int, s2: int
) -> WeakBisimResult:
    """Check if two states are branching bisimilar."""
    result = compute_branching_bisimulation(ts)

    for block in result.partition:
        if s1 in block and s2 in block:
            return WeakBisimResult(
                verdict=WeakBisimVerdict.BRANCHING_BISIMILAR,
                partition=result.partition,
                relation=result.relation,
                statistics=result.statistics,
            )

    witness = f"States {s1} and {s2} are in different branching equivalence classes"
    return WeakBisimResult(
        verdict=WeakBisimVerdict.NOT_BRANCHING_BISIMILAR,
        partition=result.partition,
        witness=witness,
        statistics=result.statistics,
    )


# ---- Quotient Construction ----

def weak_bisimulation_quotient(ts: LabeledProbTS) -> LabeledProbTS:
    """Compute the weak bisimulation quotient of a transition system.

    Collapses weakly bisimilar states into representative states.
    The quotient preserves observable behavior (weak traces).
    """
    result = compute_weak_bisimulation(ts)
    return _build_quotient(ts, result.partition)


def branching_bisimulation_quotient(ts: LabeledProbTS) -> LabeledProbTS:
    """Compute the branching bisimulation quotient."""
    result = compute_branching_bisimulation(ts)
    return _build_quotient(ts, result.partition)


def _build_quotient(ts: LabeledProbTS, partition: List[Set[int]]) -> LabeledProbTS:
    """Build quotient transition system from partition."""
    n_blocks = len(partition)

    # Map states to blocks
    state_to_block = {}
    for bi, block in enumerate(partition):
        for s in block:
            state_to_block[s] = bi

    # Build quotient actions
    # For each block, pick a representative and aggregate its actions
    quotient_actions = {}
    for bi, block in enumerate(partition):
        rep = min(block)  # representative state
        quotient_actions[bi] = {}

        for a, dist in ts.actions.get(rep, {}).items():
            # Map target states to blocks and aggregate probabilities
            block_dist = {}
            for t, p in dist:
                tb = state_to_block[t]
                block_dist[tb] = block_dist.get(tb, 0.0) + p

            quotient_actions[bi][a] = [(tb, p) for tb, p in sorted(block_dist.items())]

    # Build quotient labels
    quotient_labels = {}
    for bi, block in enumerate(partition):
        rep = min(block)
        quotient_labels[bi] = ts.state_labels.get(rep, set()).copy()

    # Build names
    names = []
    for bi, block in enumerate(partition):
        if len(block) == 1:
            s = min(block)
            names.append(ts.state_names[s] if ts.state_names else f"s{s}")
        else:
            names.append(f"[{','.join(ts.state_names[s] if ts.state_names else f's{s}' for s in sorted(block))}]")

    return LabeledProbTS(
        n_states=n_blocks,
        actions=quotient_actions,
        state_labels=quotient_labels,
        state_names=names,
    )


# ---- Cross-System Weak Bisimulation ----

def check_cross_weak_bisimulation(
    ts1: LabeledProbTS, ts2: LabeledProbTS
) -> WeakBisimResult:
    """Check weak bisimulation between two transition systems via disjoint union."""
    n1 = ts1.n_states
    n2 = ts2.n_states
    n_total = n1 + n2

    # Build disjoint union
    combined_actions = {}
    combined_labels = {}

    # ts1 states: 0..n1-1
    for s in range(n1):
        combined_actions[s] = {}
        for a, dist in ts1.actions.get(s, {}).items():
            combined_actions[s][a] = [(t, p) for t, p in dist]
        combined_labels[s] = ts1.state_labels.get(s, set()).copy()

    # ts2 states: n1..n1+n2-1
    for s in range(n2):
        combined_actions[n1 + s] = {}
        for a, dist in ts2.actions.get(s, {}).items():
            combined_actions[n1 + s][a] = [(n1 + t, p) for t, p in dist]
        combined_labels[n1 + s] = ts2.state_labels.get(s, set()).copy()

    names = []
    for s in range(n1):
        names.append(f"A.{ts1.state_names[s] if ts1.state_names else f's{s}'}")
    for s in range(n2):
        names.append(f"B.{ts2.state_names[s] if ts2.state_names else f's{s}'}")

    combined = LabeledProbTS(n_total, combined_actions, combined_labels, names)
    result = compute_weak_bisimulation(combined)

    # Check for cross-system pairs
    cross_pairs = set()
    for block in result.partition:
        sys1_states = {s for s in block if s < n1}
        sys2_states = {s - n1 for s in block if s >= n1}
        if sys1_states and sys2_states:
            for s1 in sys1_states:
                for s2 in sys2_states:
                    cross_pairs.add(frozenset({s1, s2 + n1}))

    verdict = WeakBisimVerdict.WEAKLY_BISIMILAR if cross_pairs else WeakBisimVerdict.NOT_WEAKLY_BISIMILAR

    return WeakBisimResult(
        verdict=verdict,
        partition=result.partition,
        relation=cross_pairs if cross_pairs else None,
        witness=None if cross_pairs else "No cross-system weakly bisimilar pairs found",
        statistics={
            **result.statistics,
            "sys1_states": n1,
            "sys2_states": n2,
            "cross_pairs": len(cross_pairs),
        },
    )


# ---- Weak Simulation ----

def compute_weak_simulation(
    ts: LabeledProbTS, max_iter: int = 1000
) -> WeakBisimResult:
    """Compute weak simulation preorder.

    s weakly simulates t if:
    1. labels(s) == labels(t) (or labels(s) superset)
    2. For every action a at t and every equivalence class C:
       t =a=> mu implies s =a=> mu' with mu'(C) >= mu(C) for all C

    Returns directed relation: (s, t) means s weakly simulates t.
    """
    # Start with all label-compatible pairs
    sim = set()
    for s in range(ts.n_states):
        for t in range(ts.n_states):
            ls = ts.state_labels.get(s, set())
            lt = ts.state_labels.get(t, set())
            if ls == lt:
                sim.add((s, t))

    # Precompute weak transitions
    weak_trans = {}
    for s in range(ts.n_states):
        weak_trans[s] = {}
        pre_dist = _tau_closure(ts, s)
        obs_acts = set()
        for s_mid in pre_dist:
            obs_acts |= ts.observable_actions(s_mid)
        for a in obs_acts:
            weak_trans[s][a] = _weak_transition(ts, s, a)

    # Refinement: remove pairs that violate simulation condition
    for _ in range(max_iter):
        removed = set()

        for (s, t) in sim:
            if (s, t) in removed:
                continue

            # For every action at t, s must be able to match
            t_acts = set(weak_trans.get(t, {}).keys())

            for a in t_acts:
                t_dist = weak_trans[t][a]
                s_dist = weak_trans.get(s, {}).get(a, {})

                if not s_dist and t_dist:
                    removed.add((s, t))
                    break

                # Check block probabilities (using current sim relation)
                # Simple check: s must reach at least where t reaches
                for target, p_t in t_dist.items():
                    p_s = s_dist.get(target, 0.0)
                    if p_s < p_t - 1e-9:
                        # s reaches target with less probability
                        removed.add((s, t))
                        break
                if (s, t) in removed:
                    break

        if not removed:
            break
        sim -= removed

    return WeakBisimResult(
        verdict=WeakBisimVerdict.WEAKLY_BISIMILAR if len(sim) > ts.n_states
                else WeakBisimVerdict.NOT_WEAKLY_BISIMILAR,
        relation={frozenset({s, t}) for s, t in sim if s != t},
        statistics={
            "relation_size": len(sim),
            "non_trivial_pairs": len({(s, t) for s, t in sim if s != t}),
        },
    )


# ---- Divergence Sensitivity ----

def detect_divergence(ts: LabeledProbTS) -> Dict[int, bool]:
    """Detect divergent states (states that can perform infinite tau sequences).

    A state is divergent if it can reach a tau-cycle with positive probability.
    Divergence-sensitive weak bisimulation treats divergent states differently.
    """
    divergent = {}

    for s in range(ts.n_states):
        # BFS/DFS to check for tau-reachable cycles
        visited = set()
        stack = [s]
        is_divergent = False

        while stack:
            current = stack.pop()
            if current in visited:
                if current == s or True:  # cycle found
                    is_divergent = True
                    break
                continue
            visited.add(current)
            for t, p in ts.tau_successors(current):
                if p > 0 and t != current:  # ignore self-loops for now
                    if t in visited or t == s:
                        is_divergent = True
                        break
                    stack.append(t)
                elif p > 0 and t == current:
                    # Self-loop tau
                    is_divergent = True
            if is_divergent:
                break

        divergent[s] = is_divergent

    return divergent


def compute_divergence_sensitive_bisimulation(
    ts: LabeledProbTS, max_iter: int = 1000
) -> WeakBisimResult:
    """Compute divergence-sensitive weak bisimulation.

    Like weak bisimulation, but divergent states can only be bisimilar
    to other divergent states. This prevents equating a state that can
    loop forever with one that must eventually produce output.
    """
    divergent = detect_divergence(ts)

    # Initial partition: by labels AND divergence status
    label_groups = {}
    for s in range(ts.n_states):
        key = (frozenset(ts.state_labels.get(s, set())), divergent.get(s, False))
        if key not in label_groups:
            label_groups[key] = set()
        label_groups[key].add(s)

    partition = list(label_groups.values())

    # Precompute weak transitions
    weak_trans = {}
    all_obs_actions = set()
    for s in range(ts.n_states):
        weak_trans[s] = {}
        pre_dist = _tau_closure(ts, s)
        obs_acts = set()
        for s_mid in pre_dist:
            obs_acts |= ts.observable_actions(s_mid)
        all_obs_actions |= obs_acts
        for a in obs_acts:
            weak_trans[s][a] = _weak_transition(ts, s, a)

    # Partition refinement (same as weak bisim)
    iterations = 0
    for iteration in range(max_iter):
        iterations = iteration + 1
        new_partition = []
        refined = False

        for block in partition:
            if len(block) <= 1:
                new_partition.append(block)
                continue

            sub_blocks = _split_block_weak(block, partition, weak_trans, all_obs_actions)
            if len(sub_blocks) > 1:
                refined = True
            new_partition.extend(sub_blocks)

        partition = new_partition
        if not refined:
            break

    relation = set()
    for block in partition:
        block_list = sorted(block)
        for i in range(len(block_list)):
            for j in range(i + 1, len(block_list)):
                relation.add(frozenset({block_list[i], block_list[j]}))

    is_trivial = all(len(b) == 1 for b in partition)

    return WeakBisimResult(
        verdict=WeakBisimVerdict.WEAKLY_BISIMILAR if not is_trivial
                else WeakBisimVerdict.NOT_WEAKLY_BISIMILAR,
        partition=partition,
        relation=relation if relation else None,
        statistics={
            "iterations": iterations,
            "num_blocks": len(partition),
            "reduction_ratio": 1.0 - len(partition) / ts.n_states if ts.n_states > 0 else 0.0,
            "divergent_states": [s for s, d in divergent.items() if d],
            "convergent_states": [s for s, d in divergent.items() if not d],
        },
    )


# ---- Conversion Utilities ----

def lmc_to_prob_ts(
    lmc: LabeledMC, tau_label: Optional[str] = None
) -> LabeledProbTS:
    """Convert a LabeledMC to a LabeledProbTS.

    If tau_label is given, transitions between states with that label
    become tau transitions. Otherwise, all transitions are treated as
    a single observable action 'step'.
    """
    actions = {}
    for s in range(lmc.mc.n_states):
        actions[s] = {}
        succs = lmc.mc.successors(s)
        if not succs:
            continue

        if tau_label is not None:
            # Separate tau transitions from observable ones
            tau_dist = []
            obs_dist = []
            for t, p in succs:
                if tau_label in lmc.labels.get(s, set()):
                    tau_dist.append((t, p))
                else:
                    obs_dist.append((t, p))

            if tau_dist:
                actions[s][TAU] = tau_dist
            if obs_dist:
                actions[s]["step"] = obs_dist

            # If all transitions are tau, no observable action
            if not tau_dist and not obs_dist:
                actions[s]["step"] = list(succs)
        else:
            actions[s]["step"] = list(succs)

    return LabeledProbTS(
        n_states=lmc.mc.n_states,
        actions=actions,
        state_labels=dict(lmc.labels),
        state_names=lmc.mc.state_labels,
    )


def prob_ts_to_lmc(ts: LabeledProbTS) -> LabeledMC:
    """Convert a LabeledProbTS to a LabeledMC by merging all actions.

    All actions (including tau) are combined into a single transition matrix.
    This loses the action structure but allows using V148/V067 tools.
    """
    matrix = [[0.0] * ts.n_states for _ in range(ts.n_states)]

    for s in range(ts.n_states):
        total_actions = len(ts.actions.get(s, {}))
        if total_actions == 0:
            # Self-loop for deadlock states
            matrix[s][s] = 1.0
            continue

        # Average over all actions (uniform action choice)
        for a, dist in ts.actions.get(s, {}).items():
            for t, p in dist:
                matrix[s][t] += p / total_actions

    return make_labeled_mc(matrix, ts.state_labels, ts.state_names)


# ---- Weak Bisimulation Distance ----

def compute_weak_bisimulation_distance(
    ts: LabeledProbTS,
    discount: float = 0.9,
    max_iter: int = 500,
    tol: float = 1e-8,
    threshold: float = 0.1,
) -> DistanceResult:
    """Compute weak bisimulation distance between all pairs of states.

    Uses discounted Kantorovich metric on weak transitions.
    d(s,t) = 1.0 if labels differ
    d(s,t) = discount * max_a { W(s =a=> mu, t =a=> mu') } if labels match

    where W is the Wasserstein/earth mover's distance.
    """
    n = ts.n_states

    # Initialize distances
    dist = [[0.0] * n for _ in range(n)]
    for s in range(n):
        for t in range(n):
            ls = ts.state_labels.get(s, set())
            lt = ts.state_labels.get(t, set())
            if ls != lt:
                dist[s][t] = 1.0

    # Precompute weak transitions
    weak_trans = {}
    all_obs_actions = set()
    for s in range(n):
        weak_trans[s] = {}
        pre = _tau_closure(ts, s)
        obs_acts = set()
        for s_mid in pre:
            obs_acts |= ts.observable_actions(s_mid)
        all_obs_actions |= obs_acts
        for a in obs_acts:
            weak_trans[s][a] = _weak_transition(ts, s, a)

    # Fixpoint iteration
    converged = False
    for iteration in range(max_iter):
        new_dist = [[0.0] * n for _ in range(n)]
        max_change = 0.0

        for s in range(n):
            for t in range(s + 1, n):
                ls = ts.state_labels.get(s, set())
                lt = ts.state_labels.get(t, set())
                if ls != lt:
                    new_dist[s][t] = 1.0
                    new_dist[t][s] = 1.0
                    continue

                # Max over all actions of Wasserstein distance
                max_w = 0.0
                for a in all_obs_actions:
                    mu_s = weak_trans.get(s, {}).get(a, {})
                    mu_t = weak_trans.get(t, {}).get(a, {})

                    if not mu_s and not mu_t:
                        continue
                    if bool(mu_s) != bool(mu_t):
                        max_w = max(max_w, 1.0)
                        continue

                    # Greedy earth mover
                    w = _greedy_earth_mover(mu_s, mu_t, dist, n)
                    max_w = max(max_w, w)

                new_dist[s][t] = discount * max_w
                new_dist[t][s] = new_dist[s][t]

                max_change = max(max_change, abs(new_dist[s][t] - dist[s][t]))

        dist = new_dist
        if max_change < tol:
            converged = True
            break

    # Extract results
    bisimilar = []
    near_bisimilar = []
    for s in range(n):
        for t in range(s + 1, n):
            if dist[s][t] < tol:
                bisimilar.append((s, t))
            elif dist[s][t] < threshold:
                near_bisimilar.append((s, t, dist[s][t]))

    return DistanceResult(
        distances=dist,
        max_distance=max(max(row) for row in dist) if n > 0 else 0.0,
        bisimilar_pairs=bisimilar,
        near_bisimilar_pairs=near_bisimilar,
        statistics={
            "iterations": iteration + 1 if n > 0 else 0,
            "converged": converged,
            "discount": discount,
        },
    )


def _greedy_earth_mover(
    mu1: Dict[int, float], mu2: Dict[int, float],
    dist: List[List[float]], n: int,
) -> float:
    """Compute earth mover's distance between two distributions using greedy algorithm."""
    # Collect all states with mass
    all_states = set(mu1.keys()) | set(mu2.keys())
    if not all_states:
        return 0.0

    # Create supply/demand
    supply = [(s, mu1.get(s, 0.0)) for s in sorted(all_states) if mu1.get(s, 0.0) > 1e-15]
    demand = [(s, mu2.get(s, 0.0)) for s in sorted(all_states) if mu2.get(s, 0.0) > 1e-15]

    if not supply or not demand:
        return 1.0 if supply or demand else 0.0

    # Sort pairs by distance
    pairs = []
    for i, (s1, _) in enumerate(supply):
        for j, (s2, _) in enumerate(demand):
            pairs.append((dist[s1][s2] if s1 < n and s2 < n else 1.0, i, j))
    pairs.sort()

    # Greedy flow
    remaining_supply = [s for _, s in supply]
    remaining_demand = [d for _, d in demand]
    total_cost = 0.0

    for cost, i, j in pairs:
        flow = min(remaining_supply[i], remaining_demand[j])
        if flow > 1e-15:
            total_cost += cost * flow
            remaining_supply[i] -= flow
            remaining_demand[j] -= flow

    return total_cost


# ---- Comparison with Strong Bisimulation ----

def compare_strong_vs_weak(ts: LabeledProbTS) -> Dict:
    """Compare strong bisimulation (V148) with weak bisimulation (V150).

    Weak bisimulation is always coarser (fewer blocks) than strong
    because it abstracts away tau transitions.
    """
    # Convert to LabeledMC for strong bisimulation
    lmc = prob_ts_to_lmc(ts)
    strong_result = compute_bisimulation(lmc)

    weak_result = compute_weak_bisimulation(ts)
    branching_result = compute_branching_bisimulation(ts)

    return {
        "strong": {
            "num_blocks": len(strong_result.partition),
            "partition": strong_result.partition,
        },
        "branching": {
            "num_blocks": len(branching_result.partition),
            "partition": branching_result.partition,
        },
        "weak": {
            "num_blocks": len(weak_result.partition),
            "partition": weak_result.partition,
        },
        "hierarchy": {
            "strong_leq_branching": len(strong_result.partition) >= len(branching_result.partition),
            "branching_leq_weak": len(branching_result.partition) >= len(weak_result.partition),
        },
    }


# ---- High-Level APIs ----

def minimize_weak(ts: LabeledProbTS) -> Tuple[LabeledProbTS, WeakBisimResult]:
    """Compute weak bisimulation quotient (minimization)."""
    result = compute_weak_bisimulation(ts)
    quotient = _build_quotient(ts, result.partition)
    result.quotient = quotient
    return quotient, result


def minimize_branching(ts: LabeledProbTS) -> Tuple[LabeledProbTS, WeakBisimResult]:
    """Compute branching bisimulation quotient (minimization)."""
    result = compute_branching_bisimulation(ts)
    quotient = _build_quotient(ts, result.partition)
    result.quotient = quotient
    return quotient, result


def weak_bisimulation_summary(ts: LabeledProbTS) -> str:
    """Generate human-readable weak bisimulation analysis."""
    weak = compute_weak_bisimulation(ts)
    branching = compute_branching_bisimulation(ts)
    divergent = detect_divergence(ts)

    lines = [
        f"Weak Probabilistic Bisimulation Analysis ({ts.n_states} states)",
        f"=" * 60,
        f"",
        f"Weak bisimulation:     {weak.statistics['num_blocks']} blocks "
        f"(reduction: {weak.statistics['reduction_ratio']:.1%})",
        f"Branching bisimulation: {branching.statistics['num_blocks']} blocks "
        f"(reduction: {branching.statistics['reduction_ratio']:.1%})",
        f"",
        f"Divergent states: {[s for s, d in divergent.items() if d]}",
        f"Convergent states: {[s for s, d in divergent.items() if not d]}",
        f"",
        f"Weak partition:",
    ]
    for i, block in enumerate(weak.partition):
        names = [ts.state_names[s] if ts.state_names else f"s{s}" for s in sorted(block)]
        lines.append(f"  Block {i}: {{{', '.join(names)}}}")

    lines.append("")
    lines.append("Branching partition:")
    for i, block in enumerate(branching.partition):
        names = [ts.state_names[s] if ts.state_names else f"s{s}" for s in sorted(block)]
        lines.append(f"  Block {i}: {{{', '.join(names)}}}")

    return "\n".join(lines)
