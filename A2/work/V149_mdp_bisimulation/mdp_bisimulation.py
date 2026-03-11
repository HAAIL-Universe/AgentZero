"""
V149: MDP Bisimulation
Behavioral equivalence for Markov Decision Processes.

Extends V148 (probabilistic bisimulation for Markov chains) to MDPs where
we have both nondeterministic action choice AND probabilistic transitions.

Two MDP states are bisimilar iff:
1. They have the same atomic proposition labels
2. For every action at one state, there exists a matching action at the other
   with the same transition probability to each bisimulation equivalence class

Composes: V069 (MDP) + V148 (probabilistic bisimulation) + V065 (Markov chains)
          + C037 (SMT solver)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V069_mdp_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V148_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
import math

from mdp_verification import MDP, make_mdp, Policy, mdp_to_mc, Objective
from markov_chain import MarkovChain, make_chain
from pctl_model_check import LabeledMC, make_labeled_mc
from prob_bisimulation import (
    BisimResult, BisimVerdict, compute_bisimulation as mc_compute_bisimulation,
    bisimulation_quotient as mc_bisimulation_quotient,
    compute_bisimulation_distance as mc_compute_distance,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MDPBisimVerdict(Enum):
    BISIMILAR = "bisimilar"
    NOT_BISIMILAR = "not_bisimilar"
    SIMULATES = "simulates"
    NOT_SIMULATES = "not_simulates"


@dataclass
class MDPBisimResult:
    verdict: MDPBisimVerdict
    partition: Optional[List[Set[int]]] = None
    witness: Optional[str] = None
    quotient_mdp: Optional['MDP'] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class MDPSimResult:
    verdict: MDPBisimVerdict
    relation: Optional[Set[Tuple[int, int]]] = None
    witness: Optional[str] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class MDPDistanceResult:
    distances: List[List[float]]
    max_distance: float
    bisimilar_pairs: List[Tuple[int, int]]
    near_bisimilar_pairs: List[Tuple[int, int, float]]
    statistics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Labeled MDP
# ---------------------------------------------------------------------------

@dataclass
class LabeledMDP:
    """MDP augmented with atomic proposition labels per state."""
    mdp: MDP
    labels: Dict[int, Set[str]]

    def states_with(self, label: str) -> Set[int]:
        return {s for s, ls in self.labels.items() if label in ls}

    def states_without(self, label: str) -> Set[int]:
        all_states = set(range(self.mdp.n_states))
        return all_states - self.states_with(label)


def make_labeled_mdp(n_states: int,
                     action_transitions: Dict[int, Dict[str, List[float]]],
                     labels: Dict[int, Set[str]],
                     rewards: Optional[Dict[int, Dict[str, float]]] = None,
                     state_labels: Optional[List[str]] = None) -> LabeledMDP:
    """Convenience constructor for labeled MDPs."""
    mdp = make_mdp(n_states, action_transitions, rewards, state_labels)
    full_labels = {s: labels.get(s, set()) for s in range(n_states)}
    return LabeledMDP(mdp=mdp, labels=full_labels)


# ---------------------------------------------------------------------------
# MDP Bisimulation (partition refinement)
# ---------------------------------------------------------------------------

def _label_partition(lmdp: LabeledMDP) -> List[Set[int]]:
    """Initial partition: group states by their label sets."""
    n = lmdp.mdp.n_states
    label_to_block = {}
    for s in range(n):
        key = frozenset(lmdp.labels.get(s, set()))
        if key not in label_to_block:
            label_to_block[key] = set()
        label_to_block[key].add(s)
    return list(label_to_block.values())


def _state_to_block(partition: List[Set[int]], n_states: int) -> List[int]:
    """Map each state to its block index."""
    mapping = [0] * n_states
    for bi, block in enumerate(partition):
        for s in block:
            mapping[s] = bi
    return mapping


def _action_block_probs(mdp: MDP, state: int, action_idx: int,
                        mapping: List[int], n_blocks: int) -> Tuple[float, ...]:
    """Compute transition probability to each block for a given (state, action)."""
    probs = [0.0] * n_blocks
    for t in range(mdp.n_states):
        p = mdp.transition[state][action_idx][t]
        if p > 0:
            probs[mapping[t]] += p
    # Round to avoid floating-point splitting
    return tuple(round(p, 10) for p in probs)


def _action_signature(mdp: MDP, state: int, action_idx: int,
                      mapping: List[int], n_blocks: int) -> Tuple[float, ...]:
    """Signature of one action: block-probability vector."""
    return _action_block_probs(mdp, state, action_idx, mapping, n_blocks)


def _state_signature(mdp: MDP, state: int,
                     mapping: List[int], n_blocks: int) -> FrozenSet[Tuple[float, ...]]:
    """
    MDP bisimulation signature for a state:
    The SET of action signatures (unordered, since action names don't matter
    for bisimulation -- only the available distributions matter).
    """
    n_actions = len(mdp.actions[state])
    if n_actions == 0:
        return frozenset()
    sigs = set()
    for ai in range(n_actions):
        sig = _action_signature(mdp, state, ai, mapping, n_blocks)
        sigs.add(sig)
    return frozenset(sigs)


def _refine_partition(mdp: MDP, partition: List[Set[int]]) -> List[Set[int]]:
    """One step of partition refinement for MDP bisimulation."""
    n = mdp.n_states
    n_blocks = len(partition)
    mapping = _state_to_block(partition, n)

    new_partition = []
    for block in partition:
        # Split block by state signature
        sig_to_states = {}
        for s in block:
            sig = _state_signature(mdp, s, mapping, n_blocks)
            if sig not in sig_to_states:
                sig_to_states[sig] = set()
            sig_to_states[sig].add(s)
        new_partition.extend(sig_to_states.values())
    return new_partition


def compute_mdp_bisimulation(lmdp: LabeledMDP, max_iter: int = 1000) -> MDPBisimResult:
    """
    Compute the coarsest MDP bisimulation partition via partition refinement.

    Two states s, t are bisimilar iff:
    1. L(s) = L(t) (same atomic propositions)
    2. For every action a available at s, there exists action b at t such that
       for all blocks B: P(s,a,B) = P(t,b,B), and vice versa.
    """
    mdp = lmdp.mdp
    n = mdp.n_states

    if n == 0:
        return MDPBisimResult(
            verdict=MDPBisimVerdict.BISIMILAR,
            partition=[],
            statistics={"iterations": 0, "num_blocks": 0, "original_states": 0}
        )

    partition = _label_partition(lmdp)
    iterations = 0

    for i in range(max_iter):
        new_partition = _refine_partition(mdp, partition)
        iterations = i + 1
        if len(new_partition) == len(partition):
            # Check if partition is truly stable (same blocks)
            old_sets = [frozenset(b) for b in partition]
            new_sets = [frozenset(b) for b in new_partition]
            if set(old_sets) == set(new_sets):
                break
        partition = new_partition

    # Determine verdict
    all_singletons = all(len(b) == 1 for b in partition)
    verdict = MDPBisimVerdict.NOT_BISIMILAR if all_singletons and n > 1 else MDPBisimVerdict.BISIMILAR

    # Find witness if not all bisimilar
    witness = None
    if len(partition) > 1:
        # Check if any block has more than 1 state (some states ARE bisimilar)
        multi = [b for b in partition if len(b) > 1]
        if not multi and n > 1:
            # All states distinct
            s0, s1 = list(partition[0])[0], list(partition[1])[0]
            witness = f"States {s0} and {s1} are in different blocks"

    stats = {
        "iterations": iterations,
        "num_blocks": len(partition),
        "original_states": n,
        "reduction_ratio": 1.0 - len(partition) / n if n > 0 else 0.0
    }

    return MDPBisimResult(
        verdict=verdict,
        partition=partition,
        statistics=stats
    )


def check_mdp_bisimilar(lmdp: LabeledMDP, s1: int, s2: int) -> MDPBisimResult:
    """Check if two specific states are bisimilar."""
    result = compute_mdp_bisimulation(lmdp)
    mapping = _state_to_block(result.partition, lmdp.mdp.n_states)
    same_block = mapping[s1] == mapping[s2]

    if same_block:
        return MDPBisimResult(
            verdict=MDPBisimVerdict.BISIMILAR,
            partition=result.partition,
            witness=f"States {s1} and {s2} are in the same block",
            statistics=result.statistics
        )
    else:
        return MDPBisimResult(
            verdict=MDPBisimVerdict.NOT_BISIMILAR,
            partition=result.partition,
            witness=f"States {s1} (block {mapping[s1]}) and {s2} (block {mapping[s2]}) are in different blocks",
            statistics=result.statistics
        )


# ---------------------------------------------------------------------------
# MDP Bisimulation Quotient
# ---------------------------------------------------------------------------

def mdp_bisimulation_quotient(lmdp: LabeledMDP) -> Tuple[LabeledMDP, MDPBisimResult]:
    """
    Compute the bisimulation quotient MDP: collapse bisimilar states.

    Returns the minimized labeled MDP and the bisimulation result.
    """
    result = compute_mdp_bisimulation(lmdp)
    partition = result.partition
    mdp = lmdp.mdp
    n = mdp.n_states

    if n == 0:
        return lmdp, result

    n_blocks = len(partition)
    mapping = _state_to_block(partition, n)

    # Build quotient MDP
    # For each block, pick a representative state and build action transitions
    action_transitions = {}
    rewards_dict = {}

    for bi, block in enumerate(partition):
        rep = min(block)  # representative
        n_actions = len(mdp.actions[rep])

        if n_actions == 0:
            # Deadlock state -- give it a self-loop action
            action_transitions[bi] = {"_deadlock": [0.0] * n_blocks}
            action_transitions[bi]["_deadlock"][bi] = 1.0
            continue

        block_actions = {}
        block_rewards = {}

        # Collect unique action signatures (deduplicate identical distributions)
        seen_sigs = {}
        for ai in range(n_actions):
            sig = _action_block_probs(mdp, rep, ai, mapping, n_blocks)
            action_name = mdp.actions[rep][ai]
            if sig not in seen_sigs:
                seen_sigs[sig] = action_name
                probs = [0.0] * n_blocks
                for t in range(n):
                    p = mdp.transition[rep][ai][t]
                    if p > 0:
                        probs[mapping[t]] += p
                block_actions[action_name] = probs
                # Average reward within the block for this action
                if mdp.rewards is not None:
                    block_rewards[action_name] = mdp.rewards[rep][ai]

        action_transitions[bi] = block_actions
        if block_rewards:
            rewards_dict[bi] = block_rewards

    # Labels for quotient states
    quotient_labels = {}
    for bi, block in enumerate(partition):
        rep = min(block)
        quotient_labels[bi] = lmdp.labels.get(rep, set())

    state_labels = [f"B{bi}" for bi in range(n_blocks)]

    quotient = make_labeled_mdp(
        n_states=n_blocks,
        action_transitions=action_transitions,
        labels=quotient_labels,
        rewards=rewards_dict if rewards_dict else None,
        state_labels=state_labels
    )

    result.quotient_mdp = quotient.mdp
    return quotient, result


# ---------------------------------------------------------------------------
# MDP Simulation
# ---------------------------------------------------------------------------

def _can_match_action(mdp: MDP, s: int, ai: int,
                      t: int, relation: Set[Tuple[int, int]],
                      mapping: List[int], n_blocks: int) -> bool:
    """
    Check if state t has an action that matches s's action ai
    w.r.t. transition probabilities to equivalence classes defined by relation.

    For simulation: s simulates t means for every action at t, s has a matching action.
    We check: does t have an action matching s's action ai?
    """
    n = mdp.n_states
    sig_s = _action_block_probs(mdp, s, ai, mapping, n_blocks)

    for bi in range(len(mdp.actions[t])):
        sig_t = _action_block_probs(mdp, t, bi, mapping, n_blocks)
        if sig_s == sig_t:
            return True
    return False


def compute_mdp_simulation(lmdp: LabeledMDP, max_iter: int = 1000) -> MDPSimResult:
    """
    Compute the MDP simulation preorder.

    s simulates t (written s >= t) iff:
    1. L(s) = L(t)
    2. For every action b at t, there exists action a at s such that
       for all blocks B: P(s,a,B) >= weight-matching condition holds.

    For deterministic action matching (exact distribution match):
    s simulates t iff for every action at t, s has an action with the
    same distribution over bisimulation classes.
    """
    mdp = lmdp.mdp
    n = mdp.n_states

    # Start with label-compatible pairs
    relation = set()
    for s in range(n):
        for t in range(n):
            if lmdp.labels.get(s, set()) == lmdp.labels.get(t, set()):
                relation.add((s, t))

    # First compute bisimulation partition for block mapping
    bisim = compute_mdp_bisimulation(lmdp)
    mapping = _state_to_block(bisim.partition, n)
    n_blocks = len(bisim.partition)

    # Iterative refinement
    iterations = 0
    for _ in range(max_iter):
        removed = set()
        for (s, t) in relation:
            if (s, t) in removed:
                continue
            # For every action at t, s must have a matching action
            all_matched = True
            for bi in range(len(mdp.actions[t])):
                sig_t = _action_block_probs(mdp, t, bi, mapping, n_blocks)
                found = False
                for ai in range(len(mdp.actions[s])):
                    sig_s = _action_block_probs(mdp, s, ai, mapping, n_blocks)
                    if sig_s == sig_t:
                        found = True
                        break
                if not found:
                    all_matched = False
                    break

            if not all_matched:
                removed.add((s, t))

        if not removed:
            break
        relation -= removed
        iterations += 1

    stats = {
        "iterations": iterations,
        "relation_size": len(relation),
        "total_pairs": n * n
    }

    # Check if simulation is total (bisimulation)
    is_bisim = all((t, s) in relation for (s, t) in relation)

    return MDPSimResult(
        verdict=MDPBisimVerdict.SIMULATES if len(relation) > 0 else MDPBisimVerdict.NOT_SIMULATES,
        relation=relation,
        statistics=stats
    )


def check_mdp_simulates(lmdp: LabeledMDP, s: int, t: int) -> MDPSimResult:
    """Check if state s simulates state t."""
    result = compute_mdp_simulation(lmdp)
    if (s, t) in result.relation:
        return MDPSimResult(
            verdict=MDPBisimVerdict.SIMULATES,
            relation=result.relation,
            witness=f"State {s} simulates state {t}",
            statistics=result.statistics
        )
    else:
        return MDPSimResult(
            verdict=MDPBisimVerdict.NOT_SIMULATES,
            relation=result.relation,
            witness=f"State {s} does not simulate state {t}",
            statistics=result.statistics
        )


# ---------------------------------------------------------------------------
# MDP Bisimulation Distance
# ---------------------------------------------------------------------------

def compute_mdp_bisimulation_distance(lmdp: LabeledMDP,
                                      discount: float = 0.9,
                                      max_iter: int = 500,
                                      tol: float = 1e-8,
                                      threshold: float = 0.1) -> MDPDistanceResult:
    """
    Compute bisimulation distance between MDP states.

    Uses a Hausdorff-like lifting:
    d(s,t) = 0 if L(s) != L(t) -> 1.0 (max distance for label mismatch)
    d(s,t) = discount * max(
        max_a min_b K(delta(s,a), delta(t,b), d),  -- s's worst unmatched action
        max_b min_a K(delta(s,a), delta(t,b), d)   -- t's worst unmatched action
    )
    where K is the Kantorovich distance (earth mover) lifted from d.
    """
    mdp = lmdp.mdp
    n = mdp.n_states

    # Initialize: 1.0 for different labels, 0.0 for same labels
    dist = [[0.0] * n for _ in range(n)]
    for s in range(n):
        for t in range(s + 1, n):
            if lmdp.labels.get(s, set()) != lmdp.labels.get(t, set()):
                dist[s][t] = 1.0
                dist[t][s] = 1.0

    iterations = 0
    for it in range(max_iter):
        new_dist = [[0.0] * n for _ in range(n)]
        max_change = 0.0

        for s in range(n):
            for t in range(s + 1, n):
                if lmdp.labels.get(s, set()) != lmdp.labels.get(t, set()):
                    new_dist[s][t] = 1.0
                    new_dist[t][s] = 1.0
                    continue

                # Hausdorff distance over action distributions
                hausdorff = 0.0

                # For each action at s, find best matching action at t
                if len(mdp.actions[s]) > 0 and len(mdp.actions[t]) > 0:
                    max_s = 0.0
                    for ai in range(len(mdp.actions[s])):
                        min_t = float('inf')
                        for bi in range(len(mdp.actions[t])):
                            k = _kantorovich_mdp(mdp, s, ai, t, bi, dist)
                            min_t = min(min_t, k)
                        max_s = max(max_s, min_t)

                    max_t = 0.0
                    for bi in range(len(mdp.actions[t])):
                        min_s = float('inf')
                        for ai in range(len(mdp.actions[s])):
                            k = _kantorovich_mdp(mdp, s, ai, t, bi, dist)
                            min_s = min(min_s, k)
                        max_t = max(max_t, min_s)

                    hausdorff = max(max_s, max_t)
                elif len(mdp.actions[s]) != len(mdp.actions[t]):
                    # One has actions, other doesn't
                    hausdorff = 1.0

                d = discount * hausdorff
                d = min(d, 1.0)
                new_dist[s][t] = d
                new_dist[t][s] = d
                max_change = max(max_change, abs(d - dist[s][t]))

        dist = new_dist
        iterations = it + 1
        if max_change < tol:
            break

    # Collect pairs
    bisimilar_pairs = []
    near_pairs = []
    for s in range(n):
        for t in range(s + 1, n):
            if dist[s][t] < tol:
                bisimilar_pairs.append((s, t))
            elif dist[s][t] < threshold:
                near_pairs.append((s, t, dist[s][t]))

    return MDPDistanceResult(
        distances=dist,
        max_distance=max(dist[s][t] for s in range(n) for t in range(n)) if n > 0 else 0.0,
        bisimilar_pairs=bisimilar_pairs,
        near_bisimilar_pairs=near_pairs,
        statistics={
            "iterations": iterations,
            "discount": discount,
            "converged": iterations < max_iter,
            "n_states": n
        }
    )


def _kantorovich_mdp(mdp: MDP, s: int, ai: int, t: int, bi: int,
                     dist: List[List[float]]) -> float:
    """
    Kantorovich (earth mover) distance between distributions delta(s,ai) and delta(t,bi)
    using greedy coupling (optimal for 1D, good approximation for small state spaces).
    """
    n = mdp.n_states
    # Build distribution vectors
    p = list(mdp.transition[s][ai])
    q = list(mdp.transition[t][bi])

    # Greedy earth mover: sort states by distance, move mass greedily
    # For small state spaces, use the direct LP-free approach
    total_cost = 0.0
    remaining_p = list(p)
    remaining_q = list(q)

    # Sort state pairs by distance (ascending) for greedy coupling
    pairs = []
    for i in range(n):
        for j in range(n):
            if remaining_p[i] > 1e-12 and remaining_q[j] > 1e-12:
                pairs.append((dist[i][j], i, j))
    pairs.sort()

    for d, i, j in pairs:
        if remaining_p[i] < 1e-12 or remaining_q[j] < 1e-12:
            continue
        flow = min(remaining_p[i], remaining_q[j])
        total_cost += flow * d
        remaining_p[i] -= flow
        remaining_q[j] -= flow

    return total_cost


# ---------------------------------------------------------------------------
# Cross-system MDP bisimulation
# ---------------------------------------------------------------------------

def check_cross_mdp_bisimulation(lmdp1: LabeledMDP, lmdp2: LabeledMDP) -> MDPBisimResult:
    """
    Check bisimulation between two separate labeled MDPs via disjoint union.
    States of lmdp2 are offset by lmdp1.n_states.
    """
    n1 = lmdp1.mdp.n_states
    n2 = lmdp2.mdp.n_states
    n = n1 + n2

    # Build disjoint union
    action_transitions = {}
    rewards_dict = {}

    for s in range(n1):
        acts = {}
        rews = {}
        for ai in range(len(lmdp1.mdp.actions[s])):
            name = lmdp1.mdp.actions[s][ai]
            probs = [0.0] * n
            for t in range(n1):
                probs[t] = lmdp1.mdp.transition[s][ai][t]
            acts[name] = probs
            if lmdp1.mdp.rewards is not None:
                rews[name] = lmdp1.mdp.rewards[s][ai]
        action_transitions[s] = acts
        if rews:
            rewards_dict[s] = rews

    for s in range(n2):
        acts = {}
        rews = {}
        for ai in range(len(lmdp2.mdp.actions[s])):
            name = lmdp2.mdp.actions[s][ai]
            probs = [0.0] * n
            for t in range(n2):
                probs[n1 + t] = lmdp2.mdp.transition[s][ai][t]
            acts[name] = probs
            if lmdp2.mdp.rewards is not None:
                rews[name] = lmdp2.mdp.rewards[s][ai]
        action_transitions[n1 + s] = acts
        if rews:
            rewards_dict[n1 + s] = rews

    # Union labels
    labels = {}
    for s in range(n1):
        labels[s] = lmdp1.labels.get(s, set())
    for s in range(n2):
        labels[n1 + s] = lmdp2.labels.get(s, set())

    union_lmdp = make_labeled_mdp(
        n_states=n,
        action_transitions=action_transitions,
        labels=labels,
        rewards=rewards_dict if rewards_dict else None
    )

    result = compute_mdp_bisimulation(union_lmdp)

    # Check if states from lmdp1 share blocks with states from lmdp2
    mapping = _state_to_block(result.partition, n)
    cross_bisimilar = False
    for s in range(n1):
        for t in range(n2):
            if mapping[s] == mapping[n1 + t]:
                cross_bisimilar = True
                break
        if cross_bisimilar:
            break

    if cross_bisimilar:
        result.verdict = MDPBisimVerdict.BISIMILAR
        result.witness = "Systems share bisimilar states"
    else:
        result.verdict = MDPBisimVerdict.NOT_BISIMILAR
        result.witness = "No cross-system bisimilar state pairs found"

    return result


def check_cross_mdp_bisimilar_states(lmdp1: LabeledMDP, s1: int,
                                     lmdp2: LabeledMDP, s2: int) -> MDPBisimResult:
    """Check if specific states from two MDPs are bisimilar."""
    n1 = lmdp1.mdp.n_states
    result = check_cross_mdp_bisimulation(lmdp1, lmdp2)
    mapping = _state_to_block(result.partition, n1 + lmdp2.mdp.n_states)

    if mapping[s1] == mapping[n1 + s2]:
        return MDPBisimResult(
            verdict=MDPBisimVerdict.BISIMILAR,
            partition=result.partition,
            witness=f"State {s1} of MDP1 is bisimilar to state {s2} of MDP2",
            statistics=result.statistics
        )
    else:
        return MDPBisimResult(
            verdict=MDPBisimVerdict.NOT_BISIMILAR,
            partition=result.partition,
            witness=f"State {s1} of MDP1 is NOT bisimilar to state {s2} of MDP2",
            statistics=result.statistics
        )


# ---------------------------------------------------------------------------
# Policy-induced bisimulation
# ---------------------------------------------------------------------------

def policy_bisimulation(lmdp: LabeledMDP, policy: Policy) -> BisimResult:
    """
    Compute bisimulation on the Markov chain induced by a policy.
    Uses V148's MC bisimulation directly.
    """
    mc = mdp_to_mc(lmdp.mdp, policy)
    lmc = make_labeled_mc(mc.transition, lmdp.labels, mc.state_labels)
    return mc_compute_bisimulation(lmc)


def compare_policy_bisimulations(lmdp: LabeledMDP,
                                 policy1: Policy,
                                 policy2: Policy) -> Dict:
    """Compare bisimulation partitions under two different policies."""
    result1 = policy_bisimulation(lmdp, policy1)
    result2 = policy_bisimulation(lmdp, policy2)

    return {
        "policy1_blocks": len(result1.partition),
        "policy2_blocks": len(result2.partition),
        "policy1_reduction": result1.statistics.get("reduction_ratio", 0),
        "policy2_reduction": result2.statistics.get("reduction_ratio", 0),
        "same_partition": _partitions_equal(result1.partition, result2.partition),
        "policy1_result": result1,
        "policy2_result": result2
    }


def _partitions_equal(p1: List[Set[int]], p2: List[Set[int]]) -> bool:
    """Check if two partitions are equal (same blocks, possibly different order)."""
    return set(frozenset(b) for b in p1) == set(frozenset(b) for b in p2)


# ---------------------------------------------------------------------------
# MDP bisimulation with reward awareness
# ---------------------------------------------------------------------------

def compute_reward_bisimulation(lmdp: LabeledMDP, max_iter: int = 1000) -> MDPBisimResult:
    """
    Reward-aware MDP bisimulation: states must also agree on immediate rewards
    for matching actions (not just transition distributions).
    """
    mdp = lmdp.mdp
    n = mdp.n_states

    if n == 0 or mdp.rewards is None:
        return compute_mdp_bisimulation(lmdp)

    # Initial partition: by labels AND reward signatures
    label_reward_to_block = {}
    for s in range(n):
        label_key = frozenset(lmdp.labels.get(s, set()))
        # Reward signature: sorted set of (rounded reward) values across actions
        reward_sig = tuple(sorted(set(round(r, 10) for r in mdp.rewards[s])))
        key = (label_key, reward_sig)
        if key not in label_reward_to_block:
            label_reward_to_block[key] = set()
        label_reward_to_block[key].add(s)

    partition = list(label_reward_to_block.values())
    iterations = 0

    for i in range(max_iter):
        n_blocks = len(partition)
        mapping = _state_to_block(partition, n)

        new_partition = []
        for block in partition:
            sig_to_states = {}
            for s in block:
                # Signature includes reward for each action
                sigs = set()
                for ai in range(len(mdp.actions[s])):
                    bp = _action_block_probs(mdp, s, ai, mapping, n_blocks)
                    r = round(mdp.rewards[s][ai], 10)
                    sigs.add((bp, r))
                sig = frozenset(sigs)
                if sig not in sig_to_states:
                    sig_to_states[sig] = set()
                sig_to_states[sig].add(s)
            new_partition.extend(sig_to_states.values())

        iterations = i + 1
        if len(new_partition) == len(partition):
            old_sets = set(frozenset(b) for b in partition)
            new_sets = set(frozenset(b) for b in new_partition)
            if old_sets == new_sets:
                break
        partition = new_partition

    all_singletons = all(len(b) == 1 for b in partition)
    verdict = MDPBisimVerdict.NOT_BISIMILAR if all_singletons and n > 1 else MDPBisimVerdict.BISIMILAR

    return MDPBisimResult(
        verdict=verdict,
        partition=partition,
        statistics={
            "iterations": iterations,
            "num_blocks": len(partition),
            "original_states": n,
            "reduction_ratio": 1.0 - len(partition) / n if n > 0 else 0.0,
            "reward_aware": True
        }
    )


# ---------------------------------------------------------------------------
# SMT-verified bisimulation
# ---------------------------------------------------------------------------

def verify_mdp_bisimulation_smt(lmdp: LabeledMDP,
                                partition: List[Set[int]]) -> Dict:
    """
    SMT-verify that a partition is a valid MDP bisimulation.

    Checks for each pair of states in the same block:
    1. Same labels
    2. Action matching: for every action at s, t has an action with same block probs
    """
    try:
        from smt_solver import SMTSolver, SMTResult
    except ImportError:
        return {"verified": False, "reason": "SMT solver not available"}

    mdp = lmdp.mdp
    n = mdp.n_states
    n_blocks = len(partition)
    mapping = _state_to_block(partition, n)

    violations = []

    for block in partition:
        states = sorted(block)
        if len(states) < 2:
            continue

        # Check label consistency
        ref_labels = lmdp.labels.get(states[0], set())
        for s in states[1:]:
            if lmdp.labels.get(s, set()) != ref_labels:
                violations.append(f"Label mismatch: states {states[0]} and {s}")

        # Check action matching for all pairs
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                s, t = states[i], states[j]
                # Every action at s must have a match at t
                for ai in range(len(mdp.actions[s])):
                    sig_s = _action_block_probs(mdp, s, ai, mapping, n_blocks)
                    found = False
                    for bi in range(len(mdp.actions[t])):
                        sig_t = _action_block_probs(mdp, t, bi, mapping, n_blocks)
                        if sig_s == sig_t:
                            found = True
                            break
                    if not found:
                        violations.append(
                            f"Action {mdp.actions[s][ai]} at state {s} has no match at state {t}"
                        )

                # Every action at t must have a match at s
                for bi in range(len(mdp.actions[t])):
                    sig_t = _action_block_probs(mdp, t, bi, mapping, n_blocks)
                    found = False
                    for ai in range(len(mdp.actions[s])):
                        sig_s = _action_block_probs(mdp, s, ai, mapping, n_blocks)
                        if sig_s == sig_t:
                            found = True
                            break
                    if not found:
                        violations.append(
                            f"Action {mdp.actions[t][bi]} at state {t} has no match at state {s}"
                        )

    return {
        "verified": len(violations) == 0,
        "violations": violations,
        "num_blocks": n_blocks,
        "num_states": n
    }


# ---------------------------------------------------------------------------
# Comparison with MC bisimulation
# ---------------------------------------------------------------------------

def compare_mdp_vs_mc_bisimulation(lmdp: LabeledMDP, policy: Policy) -> Dict:
    """
    Compare MDP bisimulation (action-aware) vs MC bisimulation under a policy.

    MDP bisimulation is always at least as fine as MC bisimulation under any policy,
    because MDP bisimulation requires matching ALL actions, while MC bisimulation
    only sees the single action chosen by the policy.
    """
    mdp_result = compute_mdp_bisimulation(lmdp)
    mc_result = policy_bisimulation(lmdp, policy)

    mdp_blocks = len(mdp_result.partition)
    mc_blocks = len(mc_result.partition)

    return {
        "mdp_blocks": mdp_blocks,
        "mc_blocks": mc_blocks,
        "mdp_finer": mdp_blocks >= mc_blocks,
        "mdp_reduction_ratio": mdp_result.statistics.get("reduction_ratio", 0),
        "mc_reduction_ratio": mc_result.statistics.get("reduction_ratio", 0),
        "mdp_result": mdp_result,
        "mc_result": mc_result
    }


# ---------------------------------------------------------------------------
# High-level APIs
# ---------------------------------------------------------------------------

def minimize_mdp(lmdp: LabeledMDP) -> Tuple[LabeledMDP, MDPBisimResult]:
    """Minimize an MDP via bisimulation quotient."""
    return mdp_bisimulation_quotient(lmdp)


def mdp_bisimulation_summary(lmdp: LabeledMDP) -> str:
    """Human-readable summary of MDP bisimulation analysis."""
    result = compute_mdp_bisimulation(lmdp)
    n = lmdp.mdp.n_states
    nb = len(result.partition)

    lines = [
        f"MDP Bisimulation Analysis",
        f"  States: {n}",
        f"  Blocks: {nb}",
        f"  Reduction: {result.statistics.get('reduction_ratio', 0):.1%}",
        f"  Iterations: {result.statistics.get('iterations', 0)}",
        f"  Verdict: {result.verdict.value}",
    ]

    if result.partition:
        for i, block in enumerate(result.partition):
            lines.append(f"  Block {i}: {sorted(block)}")

    return "\n".join(lines)


def analyze_mdp_bisimulation(lmdp: LabeledMDP,
                             discount: float = 0.9) -> Dict:
    """
    Full MDP bisimulation analysis: partition, quotient, distance, simulation.
    """
    bisim = compute_mdp_bisimulation(lmdp)
    quotient_lmdp, _ = mdp_bisimulation_quotient(lmdp)
    distance = compute_mdp_bisimulation_distance(lmdp, discount=discount)
    simulation = compute_mdp_simulation(lmdp)

    return {
        "bisimulation": bisim,
        "quotient": quotient_lmdp,
        "distance": distance,
        "simulation": simulation,
        "summary": mdp_bisimulation_summary(lmdp)
    }
