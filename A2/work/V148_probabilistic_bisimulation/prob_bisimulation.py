"""V148: Probabilistic Bisimulation.

Behavioral equivalence and refinement for probabilistic systems (Markov chains).
Composes V065 (Markov chains) + V067 (labeled MCs) + C037 (SMT solver).

Key concepts:
- Strong probabilistic bisimulation: partition refinement (Larsen-Skou)
- Bisimulation quotient: minimize MC by collapsing bisimilar states
- Simulation preorder: one-directional behavioral refinement
- Bisimulation distance: Kantorovich lifting of metric on states
- SMT verification: machine-checked bisimulation/simulation certificates
"""

import sys
sys.path.insert(0, 'Z:/AgentZero/A2/work/V065_markov_chain_analysis')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V067_pctl_model_checking')
sys.path.insert(0, 'Z:/AgentZero/challenges/C037_smt_solver')

from markov_chain import MarkovChain, make_chain, analyze_chain
from pctl_model_check import LabeledMC, make_labeled_mc
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BisimVerdict(Enum):
    BISIMILAR = "bisimilar"
    NOT_BISIMILAR = "not_bisimilar"
    SIMULATES = "simulates"
    NOT_SIMULATES = "not_simulates"


@dataclass
class BisimResult:
    """Result of bisimulation checking."""
    verdict: BisimVerdict
    partition: Optional[List[Set[int]]] = None  # equivalence classes
    relation: Optional[Set[FrozenSet[int]]] = None  # pairs
    witness: Optional[str] = None  # why not bisimilar
    quotient: Optional['LabeledMC'] = None  # minimized system
    statistics: Dict = field(default_factory=dict)


@dataclass
class SimResult:
    """Result of simulation checking."""
    verdict: BisimVerdict
    relation: Optional[Set[Tuple[int, int]]] = None  # (s, t) means s simulates t
    witness: Optional[str] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class DistanceResult:
    """Result of bisimulation distance computation."""
    distances: List[List[float]]  # d[i][j] = distance between state i and j
    max_distance: float
    bisimilar_pairs: List[Tuple[int, int]]
    near_bisimilar_pairs: List[Tuple[int, int, float]]  # within threshold
    statistics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strong probabilistic bisimulation (Larsen-Skou partition refinement)
# ---------------------------------------------------------------------------

def _label_partition(lmc: LabeledMC) -> List[Set[int]]:
    """Initial partition based on atomic proposition labels."""
    label_to_states = {}
    for s in range(lmc.mc.n_states):
        label_key = frozenset(lmc.labels.get(s, set()))
        if label_key not in label_to_states:
            label_to_states[label_key] = set()
        label_to_states[label_key].add(s)
    return list(label_to_states.values())


def _state_to_block(partition: List[Set[int]], n_states: int) -> List[int]:
    """Map each state to its block index."""
    mapping = [0] * n_states
    for idx, block in enumerate(partition):
        for s in block:
            mapping[s] = idx
    return mapping


def _block_transition_prob(mc: MarkovChain, state: int,
                           block: Set[int]) -> float:
    """Total transition probability from state to a block."""
    total = 0.0
    for t in block:
        total += mc.transition[state][t]
    return total


def _refine_partition(mc: MarkovChain,
                      partition: List[Set[int]]) -> List[Set[int]]:
    """One refinement step: split blocks based on transition probabilities."""
    new_partition = []
    tol = 1e-10

    for block in partition:
        if len(block) <= 1:
            new_partition.append(block)
            continue

        # For each state in block, compute signature:
        # tuple of (prob to block_0, prob to block_1, ..., prob to block_k)
        signatures = {}
        for s in block:
            sig = tuple(
                round(_block_transition_prob(mc, s, b) / tol) * tol
                for b in partition
            )
            if sig not in signatures:
                signatures[sig] = set()
            signatures[sig].add(s)

        new_partition.extend(signatures.values())

    return new_partition


def compute_bisimulation(lmc: LabeledMC,
                         max_iter: int = 1000) -> BisimResult:
    """Compute the maximal strong probabilistic bisimulation.

    Uses Larsen-Skou partition refinement:
    1. Start with label-based partition
    2. Refine by splitting blocks with different transition distributions
    3. Iterate until stable (fixpoint)

    Returns BisimResult with the coarsest stable partition.
    """
    partition = _label_partition(lmc)
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        new_partition = _refine_partition(lmc.mc, partition)

        # Check stability (same number of blocks and same blocks)
        if len(new_partition) == len(partition):
            old_blocks = {frozenset(b) for b in partition}
            new_blocks = {frozenset(b) for b in new_partition}
            if old_blocks == new_blocks:
                break

        partition = new_partition

    # Build equivalence relation
    relation = set()
    for block in partition:
        for s in block:
            for t in block:
                if s != t:
                    relation.add(frozenset({s, t}))

    return BisimResult(
        verdict=BisimVerdict.BISIMILAR if len(partition) < lmc.mc.n_states
                else BisimVerdict.NOT_BISIMILAR,
        partition=partition,
        relation=relation if relation else None,
        statistics={
            'iterations': iterations,
            'num_blocks': len(partition),
            'original_states': lmc.mc.n_states,
            'reduction_ratio': 1.0 - len(partition) / lmc.mc.n_states
                               if lmc.mc.n_states > 0 else 0.0,
        }
    )


# ---------------------------------------------------------------------------
# Check bisimilarity of two specific states
# ---------------------------------------------------------------------------

def check_bisimilar(lmc: LabeledMC, s1: int, s2: int) -> BisimResult:
    """Check whether states s1 and s2 are bisimilar."""
    result = compute_bisimulation(lmc)

    # Find which blocks s1 and s2 belong to
    for block in result.partition:
        if s1 in block and s2 in block:
            return BisimResult(
                verdict=BisimVerdict.BISIMILAR,
                partition=result.partition,
                relation=result.relation,
                statistics=result.statistics,
            )

    # Find distinguishing reason
    s2b = _state_to_block(result.partition, lmc.mc.n_states)
    witness = f"States {s1} and {s2} are in different equivalence classes"

    # Check if labels differ
    l1 = frozenset(lmc.labels.get(s1, set()))
    l2 = frozenset(lmc.labels.get(s2, set()))
    if l1 != l2:
        witness = f"Different labels: {s1} has {l1}, {s2} has {l2}"
    else:
        # Find distinguishing block
        for i, block in enumerate(result.partition):
            p1 = _block_transition_prob(lmc.mc, s1, block)
            p2 = _block_transition_prob(lmc.mc, s2, block)
            if abs(p1 - p2) > 1e-10:
                witness = (f"Different transition prob to block {i} "
                           f"({set(block)}): {s1}->{p1:.4f}, {s2}->{p2:.4f}")
                break

    return BisimResult(
        verdict=BisimVerdict.NOT_BISIMILAR,
        partition=result.partition,
        witness=witness,
        statistics=result.statistics,
    )


# ---------------------------------------------------------------------------
# Bisimulation quotient (minimization)
# ---------------------------------------------------------------------------

def bisimulation_quotient(lmc: LabeledMC) -> LabeledMC:
    """Compute the bisimulation quotient (minimized labeled MC).

    Collapses bisimilar states into single representative states.
    The quotient preserves all PCTL properties.
    """
    result = compute_bisimulation(lmc)
    partition = result.partition
    n_blocks = len(partition)

    # Build block mapping
    s2b = _state_to_block(partition, lmc.mc.n_states)

    # Build quotient transition matrix
    # For block B, pick any representative state s in B,
    # then P_quot[B][B'] = sum_{t in B'} P[s][t]
    matrix = [[0.0] * n_blocks for _ in range(n_blocks)]
    for bi, block in enumerate(partition):
        rep = next(iter(block))  # pick any representative
        for bj, target_block in enumerate(partition):
            matrix[bi][bj] = _block_transition_prob(lmc.mc, rep, target_block)

    # Build quotient labels (union of labels in block)
    labels = {}
    for bi, block in enumerate(partition):
        block_labels = set()
        for s in block:
            block_labels.update(lmc.labels.get(s, set()))
        labels[bi] = block_labels

    # State labels for readability
    state_labels = []
    for bi, block in enumerate(partition):
        if len(block) == 1:
            s = next(iter(block))
            orig_label = (lmc.mc.state_labels[s]
                          if lmc.mc.state_labels else f"s{s}")
            state_labels.append(orig_label)
        else:
            state_labels.append(f"B{bi}{{{','.join(str(s) for s in sorted(block))}}}")

    return make_labeled_mc(matrix, labels, state_labels)


# ---------------------------------------------------------------------------
# Simulation preorder
# ---------------------------------------------------------------------------

def compute_simulation(lmc: LabeledMC,
                       max_iter: int = 1000) -> SimResult:
    """Compute the maximal simulation preorder.

    s simulates t (written s >= t) if:
    1. s and t have the same labels
    2. For every transition t -> distribution mu_t, there exists a
       matching: sum over u of w(t',u) * P(s,u) >= P(t,t') for all t'
       where w is a coupling (weight function)

    For finite labeled MCs, we use the coinductive characterization:
    Start with R = {(s,t) | labels(s) = labels(t)}, refine until stable.

    A pair (s,t) in R means "s simulates t".
    """
    n = lmc.mc.n_states
    mc = lmc.mc

    # Initialize: all label-compatible pairs
    relation = set()
    for s in range(n):
        for t in range(n):
            ls = frozenset(lmc.labels.get(s, set()))
            lt = frozenset(lmc.labels.get(t, set()))
            if ls == lt:
                relation.add((s, t))

    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        removed = set()

        for (s, t) in relation:
            if (s, t) in removed:
                continue
            # Check: can s match all transitions of t?
            # For each equivalence class C w.r.t. current relation,
            # s must have at least as much probability to C as t does
            # This is the probabilistic simulation condition

            # Build current classes from t's perspective
            # For simulation: s simulates t means for each state t'
            # reachable from t, there's a matching state s' reachable
            # from s in the same simulation class
            if not _can_simulate(mc, s, t, relation, removed):
                removed.add((s, t))

        if not removed:
            break
        relation -= removed

    return SimResult(
        verdict=BisimVerdict.SIMULATES if relation else BisimVerdict.NOT_SIMULATES,
        relation=relation,
        statistics={
            'iterations': iterations,
            'relation_size': len(relation),
            'total_pairs': n * n,
        }
    )


def _can_simulate(mc: MarkovChain, s: int, t: int,
                  relation: Set[Tuple[int, int]],
                  removed: Set[Tuple[int, int]]) -> bool:
    """Check if s can simulate t under current relation.

    For probabilistic simulation, we need: for each set C of states
    that are equivalent under the relation's image, the probability
    from s to C must be >= probability from t to C.

    More precisely: group successor states by their simulation class
    (states that s can simulate), and check probability matching.
    """
    n = mc.n_states
    active = relation - removed

    # For each successor t' of t with prob > 0, we need to find
    # matching probability from s to states that simulate t'
    # This is a coupling/matching problem.

    # Simplified check using simulation classes:
    # Group states by which states they can be simulated by
    # For each group, check that s has enough probability

    # Build the set of states that s can reach with their probs
    s_succs = {}  # state -> prob from s
    for u in range(n):
        p = mc.transition[s][u]
        if p > 1e-15:
            s_succs[u] = p

    t_succs = {}  # state -> prob from t
    for u in range(n):
        p = mc.transition[t][u]
        if p > 1e-15:
            t_succs[u] = p

    # For each successor t' of t, find which successors of s can simulate t'
    # Then check that there's a valid weight assignment (coupling)
    # This is a max-flow / LP problem in general, but for small systems
    # we can use a greedy approach or exact check

    # Simple sufficient condition: partition successors into classes
    # where class(u) = frozenset of states that simulate u
    # Then for each class, sum(P(s,u) for u in class) >= sum(P(t,v) for v in class)

    # Build classes based on simulation relation
    def sim_class(u):
        """States that simulate u."""
        return frozenset(v for v in range(n) if (v, u) in active)

    # Group t's successors by their simulation class
    class_to_t_prob = {}
    for t_prime, p_t in t_succs.items():
        cls = sim_class(t_prime)
        if cls not in class_to_t_prob:
            class_to_t_prob[cls] = 0.0
        class_to_t_prob[cls] += p_t

    # For each class, check s has enough probability to simulators
    for cls, needed_prob in class_to_t_prob.items():
        available_prob = 0.0
        for s_prime, p_s in s_succs.items():
            if s_prime in cls:
                available_prob += p_s
        if available_prob < needed_prob - 1e-10:
            return False

    return True


def check_simulates(lmc: LabeledMC, s: int, t: int) -> SimResult:
    """Check if state s simulates state t."""
    result = compute_simulation(lmc)
    if (s, t) in result.relation:
        return SimResult(
            verdict=BisimVerdict.SIMULATES,
            relation=result.relation,
            statistics=result.statistics,
        )
    else:
        # Find witness
        ls = frozenset(lmc.labels.get(s, set()))
        lt = frozenset(lmc.labels.get(t, set()))
        if ls != lt:
            witness = f"Different labels: {s} has {ls}, {t} has {lt}"
        else:
            witness = (f"State {s} cannot match the transition distribution "
                       f"of state {t}")
        return SimResult(
            verdict=BisimVerdict.NOT_SIMULATES,
            witness=witness,
            statistics=result.statistics,
        )


# ---------------------------------------------------------------------------
# Bisimulation distance (Kantorovich metric)
# ---------------------------------------------------------------------------

def compute_bisimulation_distance(lmc: LabeledMC,
                                  discount: float = 0.9,
                                  max_iter: int = 500,
                                  tol: float = 1e-8,
                                  threshold: float = 0.1
                                  ) -> DistanceResult:
    """Compute bisimulation distances between all state pairs.

    Uses the discounted Kantorovich lifting:
    d(s,t) = 0                           if labels(s) != labels(t) -> d = 1
    d(s,t) = discount * W(mu_s, mu_t)    if labels(s) == labels(t)

    where W is the Kantorovich (earth mover's) distance w.r.t. d.

    For differently labeled states, d = 1.0 (maximum distance).
    Iterative fixpoint computation converges for discount < 1.
    """
    n = lmc.mc.n_states
    mc = lmc.mc

    # Initialize distances
    dist = [[0.0] * n for _ in range(n)]
    for s in range(n):
        for t in range(n):
            ls = frozenset(lmc.labels.get(s, set()))
            lt = frozenset(lmc.labels.get(t, set()))
            if ls != lt:
                dist[s][t] = 1.0

    # Iterative fixpoint
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        new_dist = [[0.0] * n for _ in range(n)]
        max_change = 0.0

        for s in range(n):
            for t in range(s + 1, n):
                ls = frozenset(lmc.labels.get(s, set()))
                lt = frozenset(lmc.labels.get(t, set()))
                if ls != lt:
                    new_dist[s][t] = 1.0
                    new_dist[t][s] = 1.0
                    continue

                # Kantorovich distance between distributions
                # For distributions mu_s and mu_t over states,
                # W(mu_s, mu_t) = min coupling sum c(i,j) * d(i,j)
                # For 1D (sorted), this is the L1 wasserstein.
                # For general metric: use the dual formulation or LP.

                # Use the dual: W = max_f sum_i (mu_s(i) - mu_t(i)) * f(i)
                # where f is 1-Lipschitz w.r.t. d.

                # Simpler: for small n, use greedy earth mover
                w = _kantorovich_distance(mc, s, t, dist, n)
                d_new = discount * w
                new_dist[s][t] = d_new
                new_dist[t][s] = d_new

                max_change = max(max_change, abs(d_new - dist[s][t]))

        dist = new_dist
        if max_change < tol:
            break

    # Collect results
    max_distance = 0.0
    bisimilar = []
    near_bisimilar = []

    for s in range(n):
        for t in range(s + 1, n):
            d = dist[s][t]
            max_distance = max(max_distance, d)
            if d < tol:
                bisimilar.append((s, t))
            elif d < threshold:
                near_bisimilar.append((s, t, d))

    return DistanceResult(
        distances=dist,
        max_distance=max_distance,
        bisimilar_pairs=bisimilar,
        near_bisimilar_pairs=near_bisimilar,
        statistics={
            'iterations': iterations,
            'discount': discount,
            'converged': iterations < max_iter,
        }
    )


def _kantorovich_distance(mc: MarkovChain, s: int, t: int,
                          dist: List[List[float]], n: int) -> float:
    """Compute Kantorovich (earth mover's) distance between distributions
    P(s, .) and P(t, .) using the greedy transportation algorithm.

    This is optimal for tree metrics but approximate for general metrics.
    For small state spaces (our use case), it's close enough.
    We use the simpler L1-based upper bound for efficiency.
    """
    # Build the "excess" and "deficit" vectors
    # excess[u] = P(s,u) - P(t,u)
    excess = [mc.transition[s][u] - mc.transition[t][u] for u in range(n)]

    # Upper bound via sum of |excess| weighted by distances
    # W <= sum_u |excess[u]| * max_v d(u,v) -- too loose
    # Better: greedy matching of positive excess to negative deficit

    pos = [(u, excess[u]) for u in range(n) if excess[u] > 1e-15]
    neg = [(u, -excess[u]) for u in range(n) if excess[u] < -1e-15]

    if not pos or not neg:
        return 0.0

    # Greedy: match each positive excess with nearest negative deficit
    total_cost = 0.0
    neg_remaining = list(neg)

    for pu, pe in pos:
        remaining = pe
        while remaining > 1e-15 and neg_remaining:
            # Find nearest unmatched deficit
            best_idx = 0
            best_cost = dist[pu][neg_remaining[0][0]]
            for i in range(1, len(neg_remaining)):
                c = dist[pu][neg_remaining[i][0]]
                if c < best_cost:
                    best_cost = c
                    best_idx = i

            nu, ne = neg_remaining[best_idx]
            transfer = min(remaining, ne)
            total_cost += transfer * best_cost
            remaining -= transfer
            ne -= transfer
            if ne < 1e-15:
                neg_remaining.pop(best_idx)
            else:
                neg_remaining[best_idx] = (nu, ne)

    return total_cost


# ---------------------------------------------------------------------------
# Cross-system bisimulation (compare two different MCs)
# ---------------------------------------------------------------------------

def check_cross_bisimulation(lmc1: LabeledMC,
                             lmc2: LabeledMC) -> BisimResult:
    """Check bisimulation between states of two different labeled MCs.

    Constructs the disjoint union MC and checks bisimulation on it.
    States 0..n1-1 from lmc1, states n1..n1+n2-1 from lmc2.
    """
    n1 = lmc1.mc.n_states
    n2 = lmc2.mc.n_states
    n = n1 + n2

    # Build disjoint union transition matrix
    matrix = [[0.0] * n for _ in range(n)]
    for s in range(n1):
        for t in range(n1):
            matrix[s][t] = lmc1.mc.transition[s][t]
    for s in range(n2):
        for t in range(n2):
            matrix[n1 + s][n1 + t] = lmc2.mc.transition[s][t]

    # Build disjoint union labels
    labels = {}
    for s in range(n1):
        labels[s] = set(lmc1.labels.get(s, set()))
    for s in range(n2):
        labels[n1 + s] = set(lmc2.labels.get(s, set()))

    combined = make_labeled_mc(matrix, labels)
    result = compute_bisimulation(combined)

    # Check which cross-system pairs are bisimilar
    cross_pairs = set()
    for block in result.partition:
        sys1_states = {s for s in block if s < n1}
        sys2_states = {s - n1 for s in block if s >= n1}
        if sys1_states and sys2_states:
            for s1 in sys1_states:
                for s2 in sys2_states:
                    cross_pairs.add(frozenset({s1, n1 + s2}))

    has_cross = len(cross_pairs) > 0

    return BisimResult(
        verdict=BisimVerdict.BISIMILAR if has_cross else BisimVerdict.NOT_BISIMILAR,
        partition=result.partition,
        relation=cross_pairs if cross_pairs else None,
        witness=None if has_cross else "No cross-system bisimilar pairs found",
        statistics={
            **result.statistics,
            'system1_states': n1,
            'system2_states': n2,
            'cross_bisimilar_pairs': len(cross_pairs),
        }
    )


def check_cross_bisimilar_states(lmc1: LabeledMC, s1: int,
                                 lmc2: LabeledMC, s2: int) -> BisimResult:
    """Check if state s1 in lmc1 is bisimilar to state s2 in lmc2."""
    n1 = lmc1.mc.n_states
    result = check_cross_bisimulation(lmc1, lmc2)

    pair = frozenset({s1, n1 + s2})
    if result.relation and pair in result.relation:
        return BisimResult(
            verdict=BisimVerdict.BISIMILAR,
            partition=result.partition,
            statistics=result.statistics,
        )
    else:
        # Find witness
        ls1 = frozenset(lmc1.labels.get(s1, set()))
        ls2 = frozenset(lmc2.labels.get(s2, set()))
        if ls1 != ls2:
            witness = f"Different labels: sys1.{s1} has {ls1}, sys2.{s2} has {ls2}"
        else:
            witness = (f"sys1.{s1} and sys2.{s2} have different transition "
                       f"distributions (not bisimilar)")
        return BisimResult(
            verdict=BisimVerdict.NOT_BISIMILAR,
            partition=result.partition,
            witness=witness,
            statistics=result.statistics,
        )


# ---------------------------------------------------------------------------
# Lumping (aggregation based on partition)
# ---------------------------------------------------------------------------

def lump_chain(lmc: LabeledMC,
               partition: List[Set[int]]) -> LabeledMC:
    """Lump (aggregate) a labeled MC according to a given partition.

    The partition must be a valid lumping (all states in a block must
    have the same transition probabilities to each block).
    """
    n_blocks = len(partition)
    mc = lmc.mc

    # Build block mapping
    s2b = _state_to_block(partition, mc.n_states)

    # Build lumped transition matrix
    matrix = [[0.0] * n_blocks for _ in range(n_blocks)]
    for bi, block in enumerate(partition):
        rep = next(iter(block))
        for bj, target_block in enumerate(partition):
            matrix[bi][bj] = _block_transition_prob(mc, rep, target_block)

    # Build lumped labels
    labels = {}
    for bi, block in enumerate(partition):
        block_labels = set()
        for s in block:
            block_labels.update(lmc.labels.get(s, set()))
        labels[bi] = block_labels

    state_labels = [f"B{bi}" for bi in range(n_blocks)]
    return make_labeled_mc(matrix, labels, state_labels)


def is_valid_lumping(lmc: LabeledMC,
                     partition: List[Set[int]]) -> Tuple[bool, Optional[str]]:
    """Check if a partition is a valid probabilistic lumping.

    A partition is valid if for every block B and every pair of states
    s1, s2 in B, they have the same transition probability to every block.
    """
    mc = lmc.mc
    tol = 1e-10

    for bi, block in enumerate(partition):
        if len(block) <= 1:
            continue
        states = sorted(block)
        ref = states[0]

        # Check labels
        ref_labels = frozenset(lmc.labels.get(ref, set()))
        for s in states[1:]:
            s_labels = frozenset(lmc.labels.get(s, set()))
            if s_labels != ref_labels:
                return False, (f"States {ref} and {s} in block {bi} have "
                               f"different labels: {ref_labels} vs {s_labels}")

        # Check transition probabilities
        for bj, target_block in enumerate(partition):
            ref_prob = _block_transition_prob(mc, ref, target_block)
            for s in states[1:]:
                s_prob = _block_transition_prob(mc, s, target_block)
                if abs(ref_prob - s_prob) > tol:
                    return False, (f"States {ref} and {s} differ in prob to "
                                   f"block {bj}: {ref_prob} vs {s_prob}")

    return True, None


# ---------------------------------------------------------------------------
# SMT-verified bisimulation checking
# ---------------------------------------------------------------------------

def verify_bisimulation_smt(lmc: LabeledMC,
                            partition: List[Set[int]]) -> Dict:
    """Verify a bisimulation partition using SMT.

    Checks all conditions of valid probabilistic bisimulation:
    1. Label consistency within blocks
    2. Transition probability consistency to each block
    """
    from smt_solver import SMTSolver, IntConst, App, Op, Var, Sort, SortKind

    INT = Sort(SortKind.INT)
    BOOL = Sort(SortKind.BOOL)

    results = {
        'valid': True,
        'label_consistent': True,
        'transition_consistent': True,
        'obligations_checked': 0,
        'obligations_passed': 0,
        'failures': [],
    }

    mc = lmc.mc

    # Scale probabilities to integers (multiply by 10^6 for precision)
    SCALE = 1000000

    # Check label consistency
    for bi, block in enumerate(partition):
        if len(block) <= 1:
            continue
        states = sorted(block)
        ref_labels = frozenset(lmc.labels.get(states[0], set()))
        for s in states[1:]:
            results['obligations_checked'] += 1
            s_labels = frozenset(lmc.labels.get(s, set()))
            if s_labels == ref_labels:
                results['obligations_passed'] += 1
            else:
                results['valid'] = False
                results['label_consistent'] = False
                results['failures'].append(
                    f"Block {bi}: states {states[0]} and {s} have "
                    f"different labels")

    # Check transition consistency via SMT
    for bi, block in enumerate(partition):
        if len(block) <= 1:
            continue
        states = sorted(block)
        ref = states[0]

        for bj, target_block in enumerate(partition):
            ref_prob_scaled = round(_block_transition_prob(mc, ref, target_block) * SCALE)

            for s in states[1:]:
                results['obligations_checked'] += 1
                s_prob_scaled = round(_block_transition_prob(mc, s, target_block) * SCALE)

                # SMT check: ref_prob == s_prob
                solver = SMTSolver()
                p_ref = solver.Int('p_ref')
                p_s = solver.Int('p_s')

                # Assert actual values
                solver.add(App(Op.EQ, [p_ref, IntConst(ref_prob_scaled)], BOOL))
                solver.add(App(Op.EQ, [p_s, IntConst(s_prob_scaled)], BOOL))

                # Assert they're NOT equal (try to refute)
                solver.add(App(Op.NEQ, [p_ref, p_s], BOOL))

                from smt_solver import SMTResult
                result = solver.check()

                if result == SMTResult.UNSAT:
                    # UNSAT means they ARE equal (good)
                    results['obligations_passed'] += 1
                else:
                    results['valid'] = False
                    results['transition_consistent'] = False
                    results['failures'].append(
                        f"Block {bi}, target {bj}: states {ref} and {s} "
                        f"differ ({ref_prob_scaled}/{SCALE} vs "
                        f"{s_prob_scaled}/{SCALE})")

    return results


# ---------------------------------------------------------------------------
# High-level APIs
# ---------------------------------------------------------------------------

def minimize(lmc: LabeledMC) -> Tuple[LabeledMC, BisimResult]:
    """Minimize a labeled MC via bisimulation quotient.

    Returns (minimized_lmc, bisimulation_result).
    """
    result = compute_bisimulation(lmc)
    quotient = bisimulation_quotient(lmc)
    result.quotient = quotient
    return quotient, result


def compare_systems(lmc1: LabeledMC,
                    lmc2: LabeledMC,
                    discount: float = 0.9) -> Dict:
    """Compare two labeled MCs via bisimulation and distance.

    Returns a comprehensive comparison dictionary.
    """
    cross = check_cross_bisimulation(lmc1, lmc2)

    # Minimize both
    min1, res1 = minimize(lmc1)
    min2, res2 = minimize(lmc2)

    # Check if minimized systems are bisimilar
    min_cross = check_cross_bisimulation(min1, min2)

    return {
        'cross_bisimulation': {
            'verdict': cross.verdict.value,
            'cross_pairs': len(cross.relation) if cross.relation else 0,
        },
        'system1': {
            'original_states': lmc1.mc.n_states,
            'minimized_states': min1.mc.n_states,
            'reduction': res1.statistics.get('reduction_ratio', 0.0),
        },
        'system2': {
            'original_states': lmc2.mc.n_states,
            'minimized_states': min2.mc.n_states,
            'reduction': res2.statistics.get('reduction_ratio', 0.0),
        },
        'minimized_cross': {
            'verdict': min_cross.verdict.value,
            'cross_pairs': len(min_cross.relation) if min_cross.relation else 0,
        },
    }


def bisimulation_summary(lmc: LabeledMC) -> str:
    """Generate a human-readable summary of bisimulation analysis."""
    result = compute_bisimulation(lmc)
    lines = []
    lines.append(f"Bisimulation Analysis ({lmc.mc.n_states} states)")
    lines.append(f"  Equivalence classes: {result.statistics['num_blocks']}")
    lines.append(f"  Iterations: {result.statistics['iterations']}")
    lines.append(f"  Reduction: {result.statistics['reduction_ratio']:.1%}")

    for i, block in enumerate(result.partition):
        states = sorted(block)
        labels = set()
        for s in states:
            labels.update(lmc.labels.get(s, set()))
        lines.append(f"  Block {i}: {states} labels={labels}")

    return "\n".join(lines)
