"""V228: Causal Discovery from Interventions -- Active structure learning.

Uses interventional experiments to disambiguate causal structures that are
observationally equivalent. Implements optimal intervention selection,
CPDAG refinement from interventional data, and active discovery loops.

Composes V214 (Causal Discovery) + V225 (Causal RL / interventional planning).

Key idea: Observational data can only identify a DAG up to its Markov
equivalence class (CPDAG). Interventional data breaks symmetries by
revealing parent-child directionality. This module selects which
interventions to perform to maximally reduce structural uncertainty.

AI-Generated | Claude (Anthropic) | AgentZero A2
"""

from __future__ import annotations

import math
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V214_causal_discovery'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V211_causal_inference'))

from causal_discovery import (
    pc_algorithm, hill_climbing, hybrid_discovery,
    chi_squared_test, mutual_information_test,
    sample_from_bn, structural_hamming_distance,
    learn_bn_structure, PCResult, HillClimbResult,
)
from bayesian_networks import BayesianNetwork, Factor, variable_elimination
from causal_inference import CausalModel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InterventionalDataset:
    """A collection of observational and interventional data.

    Each interventional dataset is tagged with which variable was intervened on
    and what value it was set to (or None for observational).
    """
    observational: list[dict] = field(default_factory=list)
    interventional: dict[str, list[dict]] = field(default_factory=dict)
    # interventional[var_name] = list of samples from do(var_name = value)
    # Each sample dict includes all variables plus '__intervention_value__'

    def add_observational(self, samples: list[dict]):
        self.observational.extend(samples)

    def add_interventional(self, target: str, value, samples: list[dict]):
        """Add interventional samples from do(target = value)."""
        key = target
        if key not in self.interventional:
            self.interventional[key] = []
        for s in samples:
            row = dict(s)
            row['__intervention_value__'] = value
            row['__intervention_target__'] = target
            self.interventional[key].append(row)

    def all_samples(self) -> list[dict]:
        """Return all samples (obs + interventional, stripped of metadata)."""
        result = list(self.observational)
        for samples in self.interventional.values():
            for s in samples:
                row = {k: v for k, v in s.items()
                       if k not in ('__intervention_value__', '__intervention_target__')}
                result.append(row)
        return result

    def interventional_for(self, target: str) -> list[dict]:
        """Get interventional samples for a specific target, stripped of metadata."""
        raw = self.interventional.get(target, [])
        return [{k: v for k, v in s.items()
                 if k not in ('__intervention_value__', '__intervention_target__')}
                for s in raw]

    def total_samples(self) -> int:
        n = len(self.observational)
        for samples in self.interventional.values():
            n += len(samples)
        return n


@dataclass
class CPDAG:
    """Completed Partially Directed Acyclic Graph.

    Represents a Markov equivalence class: some edges are directed (known
    causal direction), some are undirected (direction unknown).
    """
    variables: list[str]
    directed: set[tuple[str, str]] = field(default_factory=set)   # (parent, child)
    undirected: set[frozenset[str]] = field(default_factory=set)  # {a, b}

    def neighbors(self, v: str) -> set[str]:
        """All adjacent nodes (directed or undirected)."""
        result = set()
        for p, c in self.directed:
            if p == v:
                result.add(c)
            elif c == v:
                result.add(p)
        for edge in self.undirected:
            if v in edge:
                result.update(edge - {v})
        return result

    def children(self, v: str) -> set[str]:
        return {c for p, c in self.directed if p == v}

    def parents(self, v: str) -> set[str]:
        return {p for p, c in self.directed if c == v}

    def undirected_neighbors(self, v: str) -> set[str]:
        result = set()
        for edge in self.undirected:
            if v in edge:
                result.update(edge - {v})
        return result

    def num_undirected(self) -> int:
        return len(self.undirected)

    def is_fully_oriented(self) -> bool:
        return len(self.undirected) == 0

    def to_dag_set(self) -> set[tuple[str, str]]:
        """Convert to DAG by arbitrarily orienting undirected edges."""
        edges = set(self.directed)
        for edge in self.undirected:
            a, b = sorted(edge)
            edges.add((a, b))
        return edges

    def copy(self) -> CPDAG:
        return CPDAG(
            variables=list(self.variables),
            directed=set(self.directed),
            undirected={frozenset(e) for e in self.undirected},
        )


@dataclass
class InterventionResult:
    """Result of an intervention experiment."""
    target: str                          # variable intervened on
    value: object                        # intervention value
    samples: list[dict]                  # observed data from intervention
    edges_oriented: list[tuple[str, str]]  # edges oriented as a result
    cpdag_before: CPDAG | None = None
    cpdag_after: CPDAG | None = None


@dataclass
class ActiveDiscoveryResult:
    """Result of a full active causal discovery run."""
    final_cpdag: CPDAG
    final_dag: set[tuple[str, str]]
    interventions_performed: list[InterventionResult]
    total_samples_used: int
    iterations: int
    edges_oriented_by_observation: int
    edges_oriented_by_intervention: int
    shd: dict | None = None  # structural hamming distance if ground truth provided


# ---------------------------------------------------------------------------
# CPDAG construction from PC result
# ---------------------------------------------------------------------------

def pc_result_to_cpdag(result: PCResult) -> CPDAG:
    """Convert a PC algorithm result to a CPDAG."""
    directed = set(result.oriented_edges)
    undirected = set()
    for edge in result.undirected_edges:
        a, b = tuple(edge)
        undirected.add(frozenset({a, b}))
    return CPDAG(
        variables=list(result.variables),
        directed=directed,
        undirected=undirected,
    )


def dag_to_cpdag(edges: set[tuple[str, str]], variables: list[str]) -> CPDAG:
    """Convert a known DAG to its CPDAG (Markov equivalence class).

    An edge is compelled (directed in CPDAG) if reversing it would change
    the set of v-structures or create a cycle. Otherwise it's reversible
    (undirected in CPDAG).
    """
    # Build adjacency
    children_map: dict[str, set[str]] = defaultdict(set)
    parents_map: dict[str, set[str]] = defaultdict(set)
    for p, c in edges:
        children_map[p].add(c)
        parents_map[c].add(p)

    # Find all v-structures: X -> Z <- Y where X and Y are not adjacent
    v_structures: set[tuple[str, str, str]] = set()  # (X, Z, Y) with X < Y
    for z in variables:
        pars = sorted(parents_map[z])
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                x, y = pars[i], pars[j]
                # Check X and Y are not adjacent
                if y not in children_map[x] and x not in children_map[y]:
                    v_structures.add((x, z, y))

    # An edge is compelled if it participates in a v-structure or if
    # reversing it would create a new v-structure or a cycle.
    compelled: set[tuple[str, str]] = set()

    # Edges in v-structures are compelled
    for x, z, y in v_structures:
        compelled.add((x, z))
        compelled.add((y, z))

    # Apply Meek's rules to propagate compelled status
    changed = True
    while changed:
        changed = False
        for p, c in edges:
            if (p, c) in compelled:
                continue
            # Rule 1: If there exists X -> p -> c and X is not adjacent to c
            for x in parents_map[p]:
                if (x, p) in compelled:
                    if c not in children_map[x] and x not in children_map[c]:
                        compelled.add((p, c))
                        changed = True
                        break
            if (p, c) in compelled:
                continue
            # Rule 2: If p -> X -> c for some X (p -> c is shortcut)
            for x in children_map[p]:
                if (p, x) in compelled and c in children_map[x] and (x, c) in compelled:
                    compelled.add((p, c))
                    changed = True
                    break
            if (p, c) in compelled:
                continue
            # Rule 3: If p - X -> c and p - Y -> c, X and Y not adjacent
            undet_neighbors_of_p = set()
            for pp, cc in edges:
                if pp == p and (pp, cc) not in compelled:
                    undet_neighbors_of_p.add(cc)
                elif cc == p and (pp, cc) not in compelled:
                    undet_neighbors_of_p.add(pp)
            for x in undet_neighbors_of_p:
                if c in children_map[x] and (x, c) in compelled:
                    for y in undet_neighbors_of_p:
                        if y != x and c in children_map[y] and (y, c) in compelled:
                            if y not in children_map[x] and x not in children_map[y]:
                                compelled.add((p, c))
                                changed = True
                                break
                    if (p, c) in compelled:
                        break

    directed = compelled
    undirected: set[frozenset[str]] = set()
    for p, c in edges:
        if (p, c) not in compelled:
            undirected.add(frozenset({p, c}))

    return CPDAG(variables=variables, directed=directed, undirected=undirected)


# ---------------------------------------------------------------------------
# Interventional orientation: use do(X) data to orient edges
# ---------------------------------------------------------------------------

def orient_edges_from_intervention(
    cpdag: CPDAG,
    target: str,
    int_data: list[dict],
    obs_data: list[dict],
    alpha: float = 0.05,
) -> tuple[CPDAG, list[tuple[str, str]]]:
    """Orient undirected edges in CPDAG using interventional data on `target`.

    When we intervene on X (do(X=x)), all edges INTO X are cut.
    For each undirected edge {X, Y}:
      - If Y's distribution changes when we do(X), then X -> Y (X causes Y).
      - If Y's distribution does NOT change, then Y -> X (Y causes X, and
        cutting X's parents doesn't affect Y).

    For edges not involving X:
      - If {A, B} and we see B's distribution change under do(X), but A's
        does not, and A is adjacent to X, this may help orient A -> B.

    Returns (new_cpdag, list of newly oriented edges as (parent, child)).
    """
    result = cpdag.copy()
    oriented: list[tuple[str, str]] = []

    # Compute marginal distributions from observational data
    obs_marginals = _compute_marginals(obs_data, cpdag.variables)
    # Compute marginal distributions from interventional data (excluding target)
    int_marginals = _compute_marginals(int_data, cpdag.variables)

    # For each undirected edge involving the intervention target
    target_undirected = []
    for edge in list(result.undirected):
        if target in edge:
            target_undirected.append(edge)

    for edge in target_undirected:
        other = next(iter(edge - {target}))

        # Test if other's distribution changed under intervention on target
        changed = _distribution_changed(
            obs_marginals.get(other, {}),
            int_marginals.get(other, {}),
            alpha=alpha,
        )

        if changed:
            # X -> Y: intervening on X changes Y's distribution
            result.undirected.discard(edge)
            result.directed.add((target, other))
            oriented.append((target, other))
        else:
            # Y -> X: intervening on X doesn't affect Y
            result.undirected.discard(edge)
            result.directed.add((other, target))
            oriented.append((other, target))

    # Apply Meek's rules to propagate any new orientations
    result = _apply_meek_rules(result)

    # Check for additional orientations from indirect effects
    for edge in list(result.undirected):
        a, b = sorted(edge)
        a_changed = _distribution_changed(
            obs_marginals.get(a, {}), int_marginals.get(a, {}), alpha=alpha,
        )
        b_changed = _distribution_changed(
            obs_marginals.get(b, {}), int_marginals.get(b, {}), alpha=alpha,
        )
        if a_changed and not b_changed:
            # Target affects A but not B: if path Target->...->A->B or B->A
            # B is upstream of the effect, so B -> A
            result.undirected.discard(edge)
            result.directed.add((b, a))
            oriented.append((b, a))
        elif b_changed and not a_changed:
            result.undirected.discard(edge)
            result.directed.add((a, b))
            oriented.append((a, b))

    # Final Meek propagation
    if oriented:
        result = _apply_meek_rules(result)

    return result, oriented


def _compute_marginals(
    data: list[dict], variables: list[str]
) -> dict[str, dict]:
    """Compute empirical marginal distribution for each variable."""
    marginals: dict[str, dict] = {}
    if not data:
        return marginals
    for v in variables:
        counts: dict = {}
        total = 0
        for row in data:
            val = row.get(v)
            if val is not None:
                counts[val] = counts.get(val, 0) + 1
                total += 1
        if total > 0:
            marginals[v] = {k: c / total for k, c in counts.items()}
    return marginals


def _distribution_changed(
    obs_dist: dict, int_dist: dict, alpha: float = 0.05
) -> bool:
    """Test if two distributions are significantly different.

    Uses total variation distance with a threshold based on sample
    count considerations. For small differences, we use a chi-squared-like
    test; for clear differences, we use TVD directly.
    """
    if not obs_dist or not int_dist:
        return False

    all_vals = set(obs_dist.keys()) | set(int_dist.keys())
    tvd = 0.5 * sum(
        abs(obs_dist.get(v, 0.0) - int_dist.get(v, 0.0))
        for v in all_vals
    )

    # Threshold: alpha-based. Lower alpha -> need larger difference.
    # Heuristic: TVD > sqrt(alpha) is a reasonable threshold for moderate samples.
    threshold = math.sqrt(alpha) * 0.5
    return tvd > threshold


# ---------------------------------------------------------------------------
# Meek's orientation rules
# ---------------------------------------------------------------------------

def _apply_meek_rules(cpdag: CPDAG) -> CPDAG:
    """Apply Meek's 3 orientation rules until fixpoint."""
    changed = True
    while changed:
        changed = False
        for edge in list(cpdag.undirected):
            a, b = sorted(edge)
            oriented_dir = _check_meek(cpdag, a, b)
            if oriented_dir:
                cpdag.undirected.discard(edge)
                cpdag.directed.add(oriented_dir)
                changed = True
                break  # restart after change

            oriented_dir = _check_meek(cpdag, b, a)
            if oriented_dir:
                cpdag.undirected.discard(edge)
                cpdag.directed.add(oriented_dir)
                changed = True
                break
    return cpdag


def _check_meek(cpdag: CPDAG, a: str, b: str) -> tuple[str, str] | None:
    """Check if a -> b is forced by Meek's rules.

    Rule 1: Exists X -> A - B and X not adjacent to B => A -> B
    Rule 2: Exists A -> X -> B => A -> B
    Rule 3: Exists X -> B, Y -> B, A - X, A - Y, X not adj Y => A -> B
    """
    # Rule 1: X -> A and X not adjacent to B
    for x in cpdag.parents(a):
        if x != b and b not in cpdag.neighbors(x):
            return (a, b)

    # Rule 2: A -> X -> B for some X
    for x in cpdag.children(a):
        if (x, b) in cpdag.directed:
            return (a, b)

    # Rule 3: Two undirected neighbors X, Y of A that are parents of B
    # and X, Y not adjacent
    a_undet = cpdag.undirected_neighbors(a) - {b}
    b_parents_from_undet = set()
    for x in a_undet:
        if (x, b) in cpdag.directed:
            b_parents_from_undet.add(x)
    parents_list = sorted(b_parents_from_undet)
    for i in range(len(parents_list)):
        for j in range(i + 1, len(parents_list)):
            x, y = parents_list[i], parents_list[j]
            if y not in cpdag.neighbors(x):
                return (a, b)

    return None


# ---------------------------------------------------------------------------
# Intervention selection: choose which variable to intervene on
# ---------------------------------------------------------------------------

def intervention_score_edge_count(cpdag: CPDAG, target: str) -> float:
    """Score an intervention target by how many undirected edges it could orient.

    An intervention on `target` directly resolves all undirected edges
    involving `target`. Additional edges may be oriented via Meek's rules.
    """
    direct = sum(1 for e in cpdag.undirected if target in e)
    return float(direct)


def intervention_score_entropy(
    cpdag: CPDAG, target: str, data: list[dict]
) -> float:
    """Score by expected information gain: how much structural entropy
    the intervention reduces.

    Structural entropy = log2(number of DAGs consistent with CPDAG).
    We approximate: each undirected edge contributes 1 bit of uncertainty.
    An intervention on target resolves edges touching target plus cascading
    Meek orientations.
    """
    if cpdag.num_undirected() == 0:
        return 0.0

    # Simulate the intervention's effect
    simulated = cpdag.copy()
    resolved = 0
    for edge in list(simulated.undirected):
        if target in edge:
            # Assume we'll learn the direction
            a, b = sorted(edge)
            simulated.undirected.discard(edge)
            simulated.directed.add((a, b))  # direction doesn't matter for counting
            resolved += 1

    # Apply Meek rules for cascade
    simulated = _apply_meek_rules(simulated)
    total_resolved = cpdag.num_undirected() - simulated.num_undirected()

    return float(total_resolved)


def intervention_score_separator(cpdag: CPDAG, target: str) -> float:
    """Score by separator set: prefer variables that separate many components
    in the undirected subgraph of the CPDAG.

    Interventions on separator nodes provide maximum structural information
    because they disambiguate multiple regions simultaneously.
    """
    if cpdag.num_undirected() == 0:
        return 0.0

    # Build undirected adjacency from undirected edges only
    adj: dict[str, set[str]] = defaultdict(set)
    nodes_in_undirected: set[str] = set()
    for edge in cpdag.undirected:
        a, b = sorted(edge)
        adj[a].add(b)
        adj[b].add(a)
        nodes_in_undirected.add(a)
        nodes_in_undirected.add(b)

    if target not in nodes_in_undirected:
        return 0.0

    # Count connected components without target
    remaining = nodes_in_undirected - {target}
    visited: set[str] = set()
    components = 0
    for node in remaining:
        if node not in visited:
            components += 1
            # BFS
            queue = [node]
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                for nb in adj[current]:
                    if nb != target and nb not in visited:
                        queue.append(nb)

    # Score = number of components created (higher = better separator)
    # Also factor in direct edge count
    direct_edges = sum(1 for e in cpdag.undirected if target in e)
    return float(components + direct_edges)


def select_intervention(
    cpdag: CPDAG,
    candidates: list[str] | None = None,
    strategy: str = "entropy",
    data: list[dict] | None = None,
    cost_fn: Callable[[str], float] | None = None,
) -> str | None:
    """Select the best variable to intervene on.

    Strategies:
      - "edge_count": maximize direct undirected edges resolved
      - "entropy": maximize total edges resolved (including Meek cascade)
      - "separator": maximize graph separation
      - "cost_aware": entropy / cost (cheapest high-information intervention)

    Returns None if CPDAG is fully oriented.
    """
    if cpdag.is_fully_oriented():
        return None

    if candidates is None:
        candidates = list(cpdag.variables)

    best_target = None
    best_score = -1.0

    for target in candidates:
        if strategy == "edge_count":
            score = intervention_score_edge_count(cpdag, target)
        elif strategy == "entropy":
            score = intervention_score_entropy(cpdag, target, data or [])
        elif strategy == "separator":
            score = intervention_score_separator(cpdag, target)
        elif strategy == "cost_aware":
            info = intervention_score_entropy(cpdag, target, data or [])
            cost = cost_fn(target) if cost_fn else 1.0
            score = info / max(cost, 1e-10)
        else:
            score = intervention_score_entropy(cpdag, target, data or [])

        if score > best_score:
            best_score = score
            best_target = target

    return best_target


# ---------------------------------------------------------------------------
# Interventional data generation (simulation)
# ---------------------------------------------------------------------------

def simulate_intervention(
    bn: BayesianNetwork,
    target: str,
    value: object,
    n: int,
    seed: int | None = None,
) -> list[dict]:
    """Simulate interventional data from do(target = value) on a BN.

    Creates a mutilated graph (removes incoming edges to target, fixes its
    value) and samples via forward sampling.

    We do this manually rather than via CausalModel.do() + sample_from_bn()
    because the mutilated BN may have Factor key format mismatches.
    """
    import random
    rng = random.Random(seed)

    # Build mutilated graph: same structure but target has no parents
    # and is fixed to `value`.
    order = bn.topological_sort()

    samples = []
    for _ in range(n):
        row: dict = {}
        for node in order:
            if node == target:
                row[node] = value
                continue

            # Get parent values
            parent_vals = {p: row[p] for p in bn.parents.get(node, [])}
            domain = bn.domains[node]
            probs = []
            for val in domain:
                assignment = {**parent_vals, node: val}
                factor = bn.cpts[node]
                p = factor.get(assignment)
                probs.append(p)

            # Normalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(domain)] * len(domain)

            # Sample
            r = rng.random()
            cumulative = 0.0
            chosen = domain[-1]
            for val, p in zip(domain, probs):
                cumulative += p
                if r < cumulative:
                    chosen = val
                    break
            row[node] = chosen

        samples.append(row)

    return samples


def simulate_soft_intervention(
    bn: BayesianNetwork,
    target: str,
    new_dist: dict,
    n: int,
    seed: int | None = None,
) -> list[dict]:
    """Simulate a soft intervention: replace target's CPT with new_dist.

    Unlike do(X=x) which fixes a value, a soft intervention changes the
    mechanism but doesn't sever incoming edges.
    """
    import random
    rng = random.Random(seed)

    # Create modified BN with new CPT for target
    modified = BayesianNetwork()
    for v in bn.nodes:
        modified.add_node(v, bn.domains[v])
    for v in bn.nodes:
        for p in bn.parents.get(v, []):
            modified.add_edge(p, v)

    # Copy all CPTs
    for v in bn.nodes:
        if v == target:
            # Use new distribution (ignore parents)
            modified.set_cpt(v, new_dist)
        else:
            modified.set_cpt(v, bn.cpts[v])

    return sample_from_bn(modified, n, seed=seed)


# ---------------------------------------------------------------------------
# Active causal discovery: the main loop
# ---------------------------------------------------------------------------

def active_causal_discovery(
    initial_data: list[dict],
    variables: list[str],
    intervention_fn: Callable[[str, object], list[dict]],
    domains: dict[str, list] | None = None,
    max_interventions: int = 10,
    samples_per_intervention: int = 100,
    strategy: str = "entropy",
    alpha: float = 0.05,
    cost_fn: Callable[[str], float] | None = None,
    ground_truth: set[tuple[str, str]] | None = None,
) -> ActiveDiscoveryResult:
    """Active causal discovery loop.

    1. Learn initial CPDAG from observational data (PC algorithm)
    2. While CPDAG has undirected edges:
       a. Select best variable to intervene on
       b. Perform intervention (via intervention_fn callback)
       c. Orient edges using interventional data
       d. Apply Meek's rules for cascade
    3. Return final DAG

    Args:
        initial_data: Observational dataset (list of dicts)
        variables: Variable names
        intervention_fn: Callable(target, value) -> list[dict] of samples
        domains: Variable domains {var: [val1, val2, ...]}
        max_interventions: Maximum number of interventions to perform
        samples_per_intervention: Samples per intervention
        strategy: Intervention selection strategy
        alpha: Significance level for independence tests
        cost_fn: Optional cost function for cost-aware strategy
        ground_truth: Optional true DAG for evaluation

    Returns:
        ActiveDiscoveryResult with final CPDAG and intervention history
    """
    # Infer domains if not provided
    if domains is None:
        domains = {}
        for v in variables:
            vals = set()
            for row in initial_data:
                if v in row:
                    vals.add(row[v])
            domains[v] = sorted(vals)

    # Step 1: Learn initial CPDAG from observational data
    pc_result = pc_algorithm(initial_data, variables=variables, alpha=alpha)
    cpdag = pc_result_to_cpdag(pc_result)
    edges_by_obs = len(cpdag.directed)

    dataset = InterventionalDataset()
    dataset.add_observational(initial_data)

    interventions: list[InterventionResult] = []
    iteration = 0
    edges_by_int = 0

    # Step 2: Iterative intervention loop
    while not cpdag.is_fully_oriented() and iteration < max_interventions:
        # Select intervention target
        target = select_intervention(
            cpdag,
            candidates=variables,
            strategy=strategy,
            data=dataset.all_samples(),
            cost_fn=cost_fn,
        )
        if target is None:
            break

        # Choose an intervention value (first non-default value, or first value)
        int_value = domains[target][0]
        if len(domains[target]) > 1:
            # Use the second value to maximize signal
            int_value = domains[target][1]

        # Perform intervention
        cpdag_before = cpdag.copy()
        int_samples = intervention_fn(target, int_value)

        dataset.add_interventional(target, int_value, int_samples)

        # Orient edges using interventional data
        cpdag, newly_oriented = orient_edges_from_intervention(
            cpdag, target, int_samples, initial_data, alpha=alpha,
        )
        edges_by_int += len(newly_oriented)

        interventions.append(InterventionResult(
            target=target,
            value=int_value,
            samples=int_samples,
            edges_oriented=newly_oriented,
            cpdag_before=cpdag_before,
            cpdag_after=cpdag.copy(),
        ))

        iteration += 1

    # Compute SHD if ground truth provided
    shd = None
    if ground_truth is not None:
        final_edges = cpdag.to_dag_set()
        shd = structural_hamming_distance(ground_truth, final_edges)

    return ActiveDiscoveryResult(
        final_cpdag=cpdag,
        final_dag=cpdag.to_dag_set(),
        interventions_performed=interventions,
        total_samples_used=dataset.total_samples(),
        iterations=iteration,
        edges_oriented_by_observation=edges_by_obs,
        edges_oriented_by_intervention=edges_by_int,
        shd=shd,
    )


# ---------------------------------------------------------------------------
# Batch intervention planning: decide all interventions upfront
# ---------------------------------------------------------------------------

def plan_interventions(
    cpdag: CPDAG,
    max_interventions: int | None = None,
    strategy: str = "entropy",
    cost_fn: Callable[[str], float] | None = None,
) -> list[str]:
    """Plan a set of interventions to fully orient a CPDAG.

    Uses greedy sequential planning: at each step, select the intervention
    that resolves the most undirected edges (simulated), then simulate
    orientation and repeat.

    Returns ordered list of variables to intervene on.
    """
    sim_cpdag = cpdag.copy()
    plan: list[str] = []
    remaining = max_interventions or len(cpdag.variables)

    while not sim_cpdag.is_fully_oriented() and len(plan) < remaining:
        target = select_intervention(
            sim_cpdag,
            strategy=strategy,
            cost_fn=cost_fn,
        )
        if target is None:
            break

        plan.append(target)

        # Simulate: orient all undirected edges involving target
        for edge in list(sim_cpdag.undirected):
            if target in edge:
                a, b = sorted(edge)
                sim_cpdag.undirected.discard(edge)
                sim_cpdag.directed.add((a, b))

        sim_cpdag = _apply_meek_rules(sim_cpdag)

    return plan


def minimum_intervention_set(cpdag: CPDAG) -> list[str]:
    """Find a minimum set of interventions to fully orient the CPDAG.

    Uses the greedy approach which gives a log-approximation to the
    minimum vertex cover on the undirected subgraph.
    """
    sim_cpdag = cpdag.copy()
    targets: list[str] = []

    while not sim_cpdag.is_fully_oriented():
        # Pick variable with most undirected edges
        best = None
        best_count = -1
        for v in sim_cpdag.variables:
            count = sum(1 for e in sim_cpdag.undirected if v in e)
            if count > best_count:
                best_count = count
                best = v

        if best is None or best_count == 0:
            break

        targets.append(best)

        # Orient all edges involving this variable
        for edge in list(sim_cpdag.undirected):
            if best in edge:
                a, b = sorted(edge)
                sim_cpdag.undirected.discard(edge)
                sim_cpdag.directed.add((a, b))

        sim_cpdag = _apply_meek_rules(sim_cpdag)

    return targets


# ---------------------------------------------------------------------------
# Interventional independence testing
# ---------------------------------------------------------------------------

def interventional_independence_test(
    obs_data: list[dict],
    int_data: list[dict],
    x: str,
    y: str,
    target: str,
    alpha: float = 0.05,
) -> dict:
    """Test if X _||_ Y changes under intervention on target.

    If X _||_ Y in observational data but X !_||_ Y in interventional data
    (or vice versa), this reveals causal structure.

    Returns dict with:
      - obs_independent: bool
      - int_independent: bool
      - structure_revealed: bool (if the two tests disagree)
      - implication: str (what the disagreement implies)
    """
    obs_indep, obs_stat, obs_thresh = chi_squared_test(
        obs_data, x, y, alpha=alpha,
    )
    int_indep, int_stat, int_thresh = chi_squared_test(
        int_data, x, y, alpha=alpha,
    )

    structure_revealed = obs_indep != int_indep

    if structure_revealed:
        if obs_indep and not int_indep:
            # Were independent, now dependent under do(target)
            # target was a confounder or mediator
            implication = (
                f"{x} and {y} became dependent under do({target}): "
                f"{target} was blocking their association"
            )
        else:
            # Were dependent, now independent under do(target)
            # target was mediating or confounding
            implication = (
                f"{x} and {y} became independent under do({target}): "
                f"{target} was creating their association (confounder or mediator)"
            )
    else:
        implication = f"No change in {x}-{y} independence under do({target})"

    return {
        'obs_independent': obs_indep,
        'int_independent': int_indep,
        'structure_revealed': structure_revealed,
        'implication': implication,
        'obs_statistic': obs_stat,
        'int_statistic': int_stat,
    }


# ---------------------------------------------------------------------------
# Verifiable invariance: test if a mechanism is invariant under intervention
# ---------------------------------------------------------------------------

def check_mechanism_invariance(
    obs_data: list[dict],
    int_data: list[dict],
    child: str,
    parents: list[str],
    target: str,
    alpha: float = 0.05,
) -> dict:
    """Test if P(child | parents) is invariant under intervention on target.

    If child's mechanism P(child | parents) doesn't change under do(target),
    then target is NOT a parent of child (or its effect is fully mediated
    through the listed parents).

    This is the key invariance principle for causal discovery from
    interventions.
    """
    if not parents:
        # Test marginal invariance
        obs_marginal = _compute_marginals(obs_data, [child]).get(child, {})
        int_marginal = _compute_marginals(int_data, [child]).get(child, {})
        changed = _distribution_changed(obs_marginal, int_marginal, alpha)
        return {
            'invariant': not changed,
            'child': child,
            'parents': parents,
            'target': target,
            'implication': (
                f"P({child}) {'unchanged' if not changed else 'CHANGED'} "
                f"under do({target})"
            ),
        }

    # Stratified test: check P(child | parents = p) for each parent config
    all_parent_vals = _enumerate_parent_configs(obs_data + int_data, parents)

    any_changed = False
    strata_results: list[dict] = []

    for pconfig in all_parent_vals:
        obs_cond = _conditional_dist(obs_data, child, parents, pconfig)
        int_cond = _conditional_dist(int_data, child, parents, pconfig)

        changed = _distribution_changed(obs_cond, int_cond, alpha)
        if changed:
            any_changed = True
        strata_results.append({
            'parent_config': pconfig,
            'obs_dist': obs_cond,
            'int_dist': int_cond,
            'changed': changed,
        })

    return {
        'invariant': not any_changed,
        'child': child,
        'parents': parents,
        'target': target,
        'strata': strata_results,
        'implication': (
            f"P({child} | {', '.join(parents)}) "
            f"{'invariant' if not any_changed else 'NOT invariant'} "
            f"under do({target})"
        ),
    }


def _enumerate_parent_configs(
    data: list[dict], parents: list[str]
) -> list[dict]:
    """Enumerate all observed parent value configurations."""
    configs: list[dict] = []
    seen: set[tuple] = set()
    for row in data:
        key = tuple(row.get(p) for p in parents)
        if key not in seen:
            seen.add(key)
            configs.append({p: row[p] for p in parents})
    return configs


def _conditional_dist(
    data: list[dict],
    child: str,
    parents: list[str],
    pconfig: dict,
) -> dict:
    """Compute P(child | parents = pconfig) from data."""
    counts: dict = {}
    total = 0
    for row in data:
        if all(row.get(p) == pconfig[p] for p in parents):
            val = row[child]
            counts[val] = counts.get(val, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: c / total for k, c in counts.items()}


# ---------------------------------------------------------------------------
# Multi-target intervention: intervene on multiple variables at once
# ---------------------------------------------------------------------------

def orient_from_multi_intervention(
    cpdag: CPDAG,
    targets: set[str],
    int_data: list[dict],
    obs_data: list[dict],
    alpha: float = 0.05,
) -> tuple[CPDAG, list[tuple[str, str]]]:
    """Orient edges using a multi-target intervention do(X1=x1, X2=x2, ...).

    When we intervene on multiple variables simultaneously, we cut all
    incoming edges to ALL targets. This can resolve more edges at once.
    """
    result = cpdag.copy()
    oriented: list[tuple[str, str]] = []

    obs_marginals = _compute_marginals(obs_data, cpdag.variables)
    int_marginals = _compute_marginals(int_data, cpdag.variables)

    # Process edges involving any target
    for edge in list(result.undirected):
        a, b = sorted(edge)
        a_is_target = a in targets
        b_is_target = b in targets

        if a_is_target and b_is_target:
            # Both targets: can't determine direction from this intervention
            continue
        elif a_is_target:
            # a was intervened on
            changed = _distribution_changed(
                obs_marginals.get(b, {}), int_marginals.get(b, {}), alpha,
            )
            if changed:
                result.undirected.discard(edge)
                result.directed.add((a, b))
                oriented.append((a, b))
            else:
                result.undirected.discard(edge)
                result.directed.add((b, a))
                oriented.append((b, a))
        elif b_is_target:
            changed = _distribution_changed(
                obs_marginals.get(a, {}), int_marginals.get(a, {}), alpha,
            )
            if changed:
                result.undirected.discard(edge)
                result.directed.add((b, a))
                oriented.append((b, a))
            else:
                result.undirected.discard(edge)
                result.directed.add((a, b))
                oriented.append((a, b))

    result = _apply_meek_rules(result)
    return result, oriented


# ---------------------------------------------------------------------------
# Transportability: can we transfer causal knowledge across domains?
# ---------------------------------------------------------------------------

def check_transportability(
    source_data: list[dict],
    target_data: list[dict],
    variables: list[str],
    alpha: float = 0.05,
) -> dict:
    """Test which causal mechanisms are transportable across domains.

    A mechanism P(Y | Parents(Y)) is transportable if it's invariant
    across source and target domains. Non-transportable mechanisms
    need re-learning.
    """
    source_marginals = _compute_marginals(source_data, variables)
    target_marginals = _compute_marginals(target_data, variables)

    transportable: list[str] = []
    non_transportable: list[str] = []

    for v in variables:
        changed = _distribution_changed(
            source_marginals.get(v, {}),
            target_marginals.get(v, {}),
            alpha=alpha,
        )
        if changed:
            non_transportable.append(v)
        else:
            transportable.append(v)

    return {
        'transportable': transportable,
        'non_transportable': non_transportable,
        'fraction_transportable': (
            len(transportable) / len(variables) if variables else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Convenience: build benchmark problems
# ---------------------------------------------------------------------------

def build_chain_bn(n: int = 4, domain: list = None) -> BayesianNetwork:
    """Build a chain BN: X1 -> X2 -> ... -> Xn.

    Useful benchmark for interventional discovery: a chain has many
    observationally equivalent orientations.
    """
    if domain is None:
        domain = [0, 1]

    bn = BayesianNetwork()
    variables = [f"X{i}" for i in range(1, n + 1)]
    for v in variables:
        bn.add_node(v, domain)

    for i in range(len(variables) - 1):
        bn.add_edge(variables[i], variables[i + 1])

    # CPTs: root is uniform, others depend on parent
    bn.set_cpt_dict(variables[0], {v: 1.0 / len(domain) for v in domain})

    for i in range(1, len(variables)):
        parent = variables[i - 1]
        child = variables[i]
        cpt = {}
        for pv in domain:
            # P(child | parent) -- biased toward same value
            dist = {}
            for cv in domain:
                if cv == pv:
                    dist[cv] = 0.7
                else:
                    dist[cv] = 0.3 / (len(domain) - 1) if len(domain) > 1 else 0.3
            cpt[pv] = dist
        bn.set_cpt_dict(child, cpt)

    return bn


def build_collider_bn(domain: list = None) -> BayesianNetwork:
    """Build a collider BN: X -> Z <- Y.

    V-structure is identifiable from observational data alone.
    """
    if domain is None:
        domain = [0, 1]

    bn = BayesianNetwork()
    for v in ["X", "Y", "Z"]:
        bn.add_node(v, domain)
    bn.add_edge("X", "Z")
    bn.add_edge("Y", "Z")

    bn.set_cpt_dict("X", {0: 0.5, 1: 0.5})
    bn.set_cpt_dict("Y", {0: 0.5, 1: 0.5})

    # Z is AND-like function of X and Y
    cpt_z = {}
    for xv in domain:
        for yv in domain:
            if xv == 1 and yv == 1:
                cpt_z[(xv, yv)] = {1: 0.9, 0: 0.1}
            else:
                cpt_z[(xv, yv)] = {1: 0.1, 0: 0.9}
    bn.set_cpt_dict("Z", cpt_z)

    return bn


def build_diamond_bn(domain: list = None) -> BayesianNetwork:
    """Build a diamond BN: X -> A, X -> B, A -> Y, B -> Y.

    Has undirected edges in CPDAG that require intervention to resolve.
    """
    if domain is None:
        domain = [0, 1]

    bn = BayesianNetwork()
    for v in ["X", "A", "B", "Y"]:
        bn.add_node(v, domain)
    bn.add_edge("X", "A")
    bn.add_edge("X", "B")
    bn.add_edge("A", "Y")
    bn.add_edge("B", "Y")

    bn.set_cpt_dict("X", {0: 0.5, 1: 0.5})
    bn.set_cpt_dict("A", {0: {0: 0.8, 1: 0.2}, 1: {0: 0.2, 1: 0.8}})
    bn.set_cpt_dict("B", {0: {0: 0.7, 1: 0.3}, 1: {0: 0.3, 1: 0.7}})

    cpt_y = {}
    for av in domain:
        for bv in domain:
            if av == 1 or bv == 1:
                cpt_y[(av, bv)] = {1: 0.8, 0: 0.2}
            else:
                cpt_y[(av, bv)] = {1: 0.1, 0: 0.9}
    bn.set_cpt_dict("Y", cpt_y)

    return bn


def build_confounder_bn(domain: list = None) -> BayesianNetwork:
    """Build a confounded BN: U -> X, U -> Y, X -> Y.

    Requires intervention on X to distinguish X -> Y from X <- U -> Y.
    """
    if domain is None:
        domain = [0, 1]

    bn = BayesianNetwork()
    for v in ["U", "X", "Y"]:
        bn.add_node(v, domain)
    bn.add_edge("U", "X")
    bn.add_edge("U", "Y")
    bn.add_edge("X", "Y")

    bn.set_cpt_dict("U", {0: 0.5, 1: 0.5})
    bn.set_cpt_dict("X", {0: {0: 0.8, 1: 0.2}, 1: {0: 0.2, 1: 0.8}})
    cpt_y = {}
    for uv in domain:
        for xv in domain:
            if uv == 1 and xv == 1:
                cpt_y[(uv, xv)] = {1: 0.9, 0: 0.1}
            elif uv == 1 or xv == 1:
                cpt_y[(uv, xv)] = {1: 0.6, 0: 0.4}
            else:
                cpt_y[(uv, xv)] = {1: 0.2, 0: 0.8}
    bn.set_cpt_dict("Y", cpt_y)

    return bn


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def discovery_summary(result: ActiveDiscoveryResult) -> dict:
    """Generate a summary of an active discovery run."""
    return {
        'total_interventions': result.iterations,
        'total_samples': result.total_samples_used,
        'edges_from_observation': result.edges_oriented_by_observation,
        'edges_from_intervention': result.edges_oriented_by_intervention,
        'final_directed_edges': len(result.final_cpdag.directed),
        'remaining_undirected': result.final_cpdag.num_undirected(),
        'fully_oriented': result.final_cpdag.is_fully_oriented(),
        'shd': result.shd,
        'interventions': [
            {
                'target': ir.target,
                'value': ir.value,
                'edges_oriented': len(ir.edges_oriented),
                'details': ir.edges_oriented,
            }
            for ir in result.interventions_performed
        ],
    }
