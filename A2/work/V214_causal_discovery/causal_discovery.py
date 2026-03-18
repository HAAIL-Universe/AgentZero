"""V214: Causal Discovery -- Structure learning from data.

Implements constraint-based (PC algorithm), score-based (hill climbing with BIC),
and hybrid approaches for learning causal DAG structure from observational data.

Composes V209 (Bayesian Networks) + V211 (Causal Inference).

AI-Generated | Claude (Anthropic) | AgentZero A2
"""

from __future__ import annotations

import math
import itertools
from collections import defaultdict
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V211_causal_inference'))

from bayesian_networks import BayesianNetwork, variable_elimination, Factor
from causal_inference import CausalModel


# ---------------------------------------------------------------------------
# Statistical independence testing from data
# ---------------------------------------------------------------------------

def _contingency_table(
    data: list[dict],
    x: str,
    y: str,
    z: set[str] | None = None,
) -> dict:
    """Build contingency counts from data.

    Returns nested dict: z_val -> x_val -> y_val -> count
    If z is None or empty, z_val is always ().
    """
    z = z or set()
    z_sorted = sorted(z)
    counts: dict = {}
    for row in data:
        z_key = tuple(row[v] for v in z_sorted)
        x_val = row[x]
        y_val = row[y]
        if z_key not in counts:
            counts[z_key] = {}
        if x_val not in counts[z_key]:
            counts[z_key][x_val] = {}
        counts[z_key][x_val][y_val] = counts[z_key][x_val].get(y_val, 0) + 1
    return counts


def chi_squared_test(
    data: list[dict],
    x: str,
    y: str,
    z: set[str] | None = None,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """Test conditional independence X _||_ Y | Z using chi-squared test.

    Returns (independent, statistic, threshold).
    Uses asymptotic chi-squared distribution approximation.
    """
    z = z or set()
    table = _contingency_table(data, x, y, z)

    total_stat = 0.0
    total_df = 0

    for z_key, xy_counts in table.items():
        # Get all x and y values in this stratum
        x_vals = sorted(xy_counts.keys())
        y_vals_set: set = set()
        for xv in x_vals:
            y_vals_set.update(xy_counts[xv].keys())
        y_vals = sorted(y_vals_set)

        if len(x_vals) < 2 or len(y_vals) < 2:
            continue

        # Row and column totals
        n = 0
        row_totals: dict = {}
        col_totals: dict = {yv: 0 for yv in y_vals}
        for xv in x_vals:
            row_totals[xv] = 0
            for yv in y_vals:
                c = xy_counts[xv].get(yv, 0)
                row_totals[xv] += c
                col_totals[yv] += c
                n += c

        if n == 0:
            continue

        # Chi-squared statistic
        stat = 0.0
        for xv in x_vals:
            for yv in y_vals:
                observed = xy_counts[xv].get(yv, 0)
                expected = row_totals[xv] * col_totals[yv] / n
                if expected > 0:
                    stat += (observed - expected) ** 2 / expected

        df = (len(x_vals) - 1) * (len(y_vals) - 1)
        total_stat += stat
        total_df += df

    if total_df == 0:
        return True, 0.0, 0.0

    # Chi-squared critical value approximation (Wilson-Hilferty)
    threshold = _chi2_critical(total_df, alpha)
    independent = total_stat < threshold
    return independent, total_stat, threshold


def _chi2_critical(df: int, alpha: float) -> float:
    """Approximate chi-squared critical value using Wilson-Hilferty transform."""
    if df <= 0:
        return 0.0
    # z_alpha for common alpha values
    z = _normal_quantile(1 - alpha)
    # Wilson-Hilferty approximation
    term = 1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))
    return df * max(0, term) ** 3


def _normal_quantile(p: float) -> float:
    """Approximate normal quantile (inverse CDF) using rational approximation."""
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    # Rational approximation (Abramowitz & Stegun 26.2.23)
    if p > 0.5:
        return -_normal_quantile(1 - p)

    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


def mutual_information_test(
    data: list[dict],
    x: str,
    y: str,
    z: set[str] | None = None,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """Test conditional independence using mutual information (G-test).

    Uses 2*N*MI as test statistic (asymptotically chi-squared).
    Returns (independent, statistic, threshold).
    """
    z = z or set()
    table = _contingency_table(data, x, y, z)

    total_stat = 0.0
    total_df = 0

    for z_key, xy_counts in table.items():
        x_vals = sorted(xy_counts.keys())
        y_vals_set: set = set()
        for xv in x_vals:
            y_vals_set.update(xy_counts[xv].keys())
        y_vals = sorted(y_vals_set)

        if len(x_vals) < 2 or len(y_vals) < 2:
            continue

        n = 0
        row_totals: dict = {}
        col_totals: dict = {yv: 0 for yv in y_vals}
        for xv in x_vals:
            row_totals[xv] = 0
            for yv in y_vals:
                c = xy_counts[xv].get(yv, 0)
                row_totals[xv] += c
                col_totals[yv] += c
                n += c

        if n == 0:
            continue

        # G-statistic = 2 * sum(observed * ln(observed / expected))
        g = 0.0
        for xv in x_vals:
            for yv in y_vals:
                observed = xy_counts[xv].get(yv, 0)
                expected = row_totals[xv] * col_totals[yv] / n
                if observed > 0 and expected > 0:
                    g += observed * math.log(observed / expected)
        g *= 2

        df = (len(x_vals) - 1) * (len(y_vals) - 1)
        total_stat += g
        total_df += df

    if total_df == 0:
        return True, 0.0, 0.0

    threshold = _chi2_critical(total_df, alpha)
    independent = total_stat < threshold
    return independent, total_stat, threshold


# ---------------------------------------------------------------------------
# PC Algorithm (constraint-based causal discovery)
# ---------------------------------------------------------------------------

class PCResult:
    """Result of PC algorithm."""

    def __init__(
        self,
        skeleton: dict[str, set[str]],
        separating_sets: dict[tuple[str, str], set[str]],
        oriented_edges: set[tuple[str, str]],
        undirected_edges: set[tuple[str, str]],
        variables: list[str],
    ):
        self.skeleton = skeleton          # adjacency (undirected)
        self.separating_sets = separating_sets  # (x,y) -> conditioning set
        self.oriented_edges = oriented_edges    # directed edges
        self.undirected_edges = undirected_edges  # remaining undirected
        self.variables = variables

    def to_dag(self) -> set[tuple[str, str]]:
        """Return all edges as directed. Undirected edges are arbitrarily oriented."""
        edges = set(self.oriented_edges)
        for u, v in self.undirected_edges:
            if (v, u) not in edges:
                edges.add((u, v))
        return edges

    def to_cpdag(self) -> tuple[set[tuple[str, str]], set[frozenset[str]]]:
        """Return CPDAG representation: (directed_edges, undirected_edges)."""
        directed = set(self.oriented_edges)
        undirected = set()
        for u, v in self.undirected_edges:
            undirected.add(frozenset({u, v}))
        return directed, undirected


def pc_algorithm(
    data: list[dict],
    variables: list[str] | None = None,
    alpha: float = 0.05,
    test: str = "chi2",
    max_cond_size: int | None = None,
) -> PCResult:
    """PC algorithm for causal structure learning.

    Phase 1: Learn skeleton by testing conditional independences.
    Phase 2: Orient v-structures (colliders).
    Phase 3: Apply Meek's rules for additional orientations.

    Args:
        data: List of observation dicts.
        variables: Variable names (inferred from data if None).
        alpha: Significance level for independence tests.
        test: "chi2" or "mi" (mutual information).
        max_cond_size: Maximum conditioning set size (None = no limit).
    """
    if not data:
        raise ValueError("Data cannot be empty")

    if variables is None:
        variables = sorted(data[0].keys())

    test_fn = chi_squared_test if test == "chi2" else mutual_information_test

    # Phase 1: Skeleton discovery
    skeleton, sep_sets = _pc_skeleton(data, variables, alpha, test_fn, max_cond_size)

    # Phase 2: Orient v-structures
    oriented, undirected = _orient_v_structures(variables, skeleton, sep_sets)

    # Phase 3: Meek's rules
    oriented, undirected = _meek_rules(variables, skeleton, oriented, undirected)

    return PCResult(skeleton, sep_sets, oriented, undirected, variables)


def _pc_skeleton(
    data: list[dict],
    variables: list[str],
    alpha: float,
    test_fn,
    max_cond_size: int | None,
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]]]:
    """Phase 1: Learn undirected skeleton."""
    # Start with complete undirected graph
    adj: dict[str, set[str]] = {v: set(variables) - {v} for v in variables}
    sep_sets: dict[tuple[str, str], set[str]] = {}

    d = 0  # conditioning set size
    while True:
        if max_cond_size is not None and d > max_cond_size:
            break

        found_edge_to_test = False

        for x in list(variables):
            for y in list(adj.get(x, set())):
                if y not in adj.get(x, set()):
                    continue

                # Neighbors of x excluding y
                neighbors = adj[x] - {y}
                if len(neighbors) < d:
                    continue

                found_edge_to_test = True

                # Test all conditioning sets of size d from neighbors
                for z_tuple in itertools.combinations(sorted(neighbors), d):
                    z_set = set(z_tuple)
                    independent, _, _ = test_fn(data, x, y, z_set, alpha)

                    if independent:
                        # Remove edge
                        adj[x].discard(y)
                        adj[y].discard(x)
                        # Store separating set (canonical order)
                        key = (min(x, y), max(x, y))
                        sep_sets[key] = z_set
                        break

        if not found_edge_to_test:
            break
        d += 1

    return adj, sep_sets


def _orient_v_structures(
    variables: list[str],
    skeleton: dict[str, set[str]],
    sep_sets: dict[tuple[str, str], set[str]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Phase 2: Orient v-structures (colliders) X -> Z <- Y."""
    oriented: set[tuple[str, str]] = set()
    undirected: set[tuple[str, str]] = set()

    # Collect all undirected edges first
    seen: set[frozenset[str]] = set()
    for x in variables:
        for y in skeleton.get(x, set()):
            edge = frozenset({x, y})
            if edge not in seen:
                seen.add(edge)
                undirected.add((min(x, y), max(x, y)))

    # For each unshielded triple X - Z - Y (where X and Y not adjacent)
    for z in variables:
        neighbors = sorted(skeleton.get(z, set()))
        for i, x in enumerate(neighbors):
            for y in neighbors[i + 1:]:
                # Check X and Y are not adjacent (unshielded)
                if y in skeleton.get(x, set()):
                    continue

                # Check if Z is in the separating set of X and Y
                key = (min(x, y), max(x, y))
                sep = sep_sets.get(key, set())

                if z not in sep:
                    # V-structure: X -> Z <- Y
                    oriented.add((x, z))
                    oriented.add((y, z))
                    undirected.discard((min(x, z), max(x, z)))
                    undirected.discard((min(y, z), max(y, z)))

    return oriented, undirected


def _meek_rules(
    variables: list[str],
    skeleton: dict[str, set[str]],
    oriented: set[tuple[str, str]],
    undirected: set[tuple[str, str]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Phase 3: Apply Meek's orientation rules until fixpoint."""
    changed = True
    while changed:
        changed = False

        for u_edge in list(undirected):
            a, b = u_edge

            for x, y in [(a, b), (b, a)]:
                orient_as = None

                # Rule 1: If Z -> X - Y and Z not adjacent to Y, orient X -> Y
                for z in variables:
                    if (z, x) in oriented and z not in skeleton.get(y, set()):
                        orient_as = (x, y)
                        break

                # Rule 2: If X -> Z -> Y (directed path), orient X -> Y
                if orient_as is None:
                    for z in variables:
                        if (x, z) in oriented and (z, y) in oriented:
                            orient_as = (x, y)
                            break

                # Rule 3: If X - Z1 -> Y and X - Z2 -> Y (two undirected to Y
                # via different Z), orient X -> Y
                if orient_as is None:
                    z_candidates = []
                    for z in skeleton.get(x, set()):
                        if z == y:
                            continue
                        x_z_undir = (min(x, z), max(x, z)) in undirected
                        if x_z_undir and (z, y) in oriented:
                            z_candidates.append(z)
                    if len(z_candidates) >= 2:
                        # Check z candidates are not adjacent to each other
                        for i, z1 in enumerate(z_candidates):
                            for z2 in z_candidates[i + 1:]:
                                if z2 not in skeleton.get(z1, set()):
                                    orient_as = (x, y)
                                    break
                            if orient_as:
                                break

                if orient_as:
                    oriented.add(orient_as)
                    undirected.discard(u_edge)
                    changed = True
                    break

    return oriented, undirected


# ---------------------------------------------------------------------------
# Score-based structure learning (Hill Climbing with BIC)
# ---------------------------------------------------------------------------

def bic_score(
    data: list[dict],
    node: str,
    parents: list[str],
    domains: dict[str, list],
) -> float:
    """Compute BIC score for a node given parents.

    BIC = log-likelihood - (k/2) * log(N)
    where k = number of free parameters.
    """
    n = len(data)
    if n == 0:
        return 0.0

    # Count (parent_config, node_value) occurrences
    parent_counts: dict = {}
    joint_counts: dict = {}

    for row in data:
        p_key = tuple(row[p] for p in parents)
        v = row[node]
        parent_counts[p_key] = parent_counts.get(p_key, 0) + 1
        jk = (p_key, v)
        joint_counts[jk] = joint_counts.get(jk, 0) + 1

    # Log-likelihood
    ll = 0.0
    for (p_key, v), count in joint_counts.items():
        p_count = parent_counts.get(p_key, 0)
        if p_count > 0 and count > 0:
            ll += count * math.log(count / p_count)

    # Number of free parameters
    node_card = len(domains.get(node, []))
    parent_configs = 1
    for p in parents:
        parent_configs *= len(domains.get(p, []))
    k = parent_configs * (node_card - 1)

    return ll - (k / 2) * math.log(n)


def _has_cycle(adj: dict[str, set[str]], source: str, target: str) -> bool:
    """Check if adding edge source -> target would create a cycle.

    Returns True if target can already reach source (cycle would form).
    """
    visited = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, set()))
    return False


class HillClimbResult:
    """Result of hill climbing structure search."""

    def __init__(
        self,
        edges: set[tuple[str, str]],
        variables: list[str],
        domains: dict[str, list],
        score: float,
        iterations: int,
    ):
        self.edges = edges
        self.variables = variables
        self.domains = domains
        self.score = score
        self.iterations = iterations

    def to_dag(self) -> set[tuple[str, str]]:
        return set(self.edges)


def hill_climbing(
    data: list[dict],
    variables: list[str] | None = None,
    domains: dict[str, list] | None = None,
    max_parents: int = 4,
    max_iterations: int = 500,
    restarts: int = 0,
    initial_edges: set[tuple[str, str]] | None = None,
) -> HillClimbResult:
    """Greedy hill climbing with BIC score for DAG structure learning.

    Operations: add edge, remove edge, reverse edge.
    """
    if not data:
        raise ValueError("Data cannot be empty")

    if variables is None:
        variables = sorted(data[0].keys())

    if domains is None:
        domains = {}
        for v in variables:
            vals = sorted(set(row[v] for row in data))
            domains[v] = vals

    best_result = None

    for restart in range(restarts + 1):
        # Current graph as adjacency (parent -> children for cycle check)
        children: dict[str, set[str]] = {v: set() for v in variables}
        parents: dict[str, list[str]] = {v: [] for v in variables}
        edges: set[tuple[str, str]] = set()

        if restart == 0 and initial_edges:
            for u, v in initial_edges:
                edges.add((u, v))
                children[u].add(v)
                parents[v].append(u)

        # Cache node scores
        node_scores: dict[str, float] = {}
        for v in variables:
            node_scores[v] = bic_score(data, v, parents[v], domains)

        total_score = sum(node_scores.values())
        iteration = 0

        while iteration < max_iterations:
            best_op = None
            best_delta = 0.0
            iteration += 1

            # Try all add/remove/reverse operations
            for u in variables:
                for v in variables:
                    if u == v:
                        continue

                    if (u, v) in edges:
                        # Try remove u -> v
                        new_parents = [p for p in parents[v] if p != u]
                        new_score = bic_score(data, v, new_parents, domains)
                        delta = new_score - node_scores[v]
                        if delta > best_delta:
                            best_delta = delta
                            best_op = ("remove", u, v)

                        # Try reverse u -> v to v -> u
                        if len(parents[u]) < max_parents:
                            # Check no cycle after reverse
                            # Temporarily remove u->v, then check v->u
                            children[u].discard(v)
                            would_cycle = _has_cycle(children, v, u)
                            children[u].add(v)

                            if not would_cycle:
                                new_parents_v = [p for p in parents[v] if p != u]
                                new_parents_u = parents[u] + [v]
                                score_v = bic_score(data, v, new_parents_v, domains)
                                score_u = bic_score(data, u, new_parents_u, domains)
                                delta = (score_v + score_u) - (node_scores[v] + node_scores[u])
                                if delta > best_delta:
                                    best_delta = delta
                                    best_op = ("reverse", u, v)
                    else:
                        # Try add u -> v
                        if (v, u) in edges:
                            continue
                        if len(parents[v]) >= max_parents:
                            continue
                        if _has_cycle(children, u, v):
                            continue

                        new_parents = parents[v] + [u]
                        new_score = bic_score(data, v, new_parents, domains)
                        delta = new_score - node_scores[v]
                        if delta > best_delta:
                            best_delta = delta
                            best_op = ("add", u, v)

            if best_op is None:
                break

            # Apply operation
            op, u, v = best_op
            if op == "add":
                edges.add((u, v))
                children[u].add(v)
                parents[v].append(u)
                node_scores[v] = bic_score(data, v, parents[v], domains)
            elif op == "remove":
                edges.discard((u, v))
                children[u].discard(v)
                parents[v] = [p for p in parents[v] if p != u]
                node_scores[v] = bic_score(data, v, parents[v], domains)
            elif op == "reverse":
                edges.discard((u, v))
                children[u].discard(v)
                parents[v] = [p for p in parents[v] if p != u]
                edges.add((v, u))
                children[v].add(u)
                parents[u].append(v)
                node_scores[v] = bic_score(data, v, parents[v], domains)
                node_scores[u] = bic_score(data, u, parents[u], domains)

            total_score = sum(node_scores.values())

        result = HillClimbResult(edges, variables, domains, total_score, iteration)
        if best_result is None or result.score > best_result.score:
            best_result = result

    return best_result


# ---------------------------------------------------------------------------
# Hybrid: PC + Hill Climbing (MMHC-style)
# ---------------------------------------------------------------------------

def hybrid_discovery(
    data: list[dict],
    variables: list[str] | None = None,
    alpha: float = 0.05,
    test: str = "chi2",
    max_parents: int = 4,
    max_iterations: int = 500,
) -> HillClimbResult:
    """Hybrid structure learning: PC skeleton restricts HC search space.

    1. Run PC to find skeleton (undirected edges).
    2. Use skeleton as candidate set for HC (only allow edges in skeleton).
    """
    if variables is None:
        variables = sorted(data[0].keys())

    # Phase 1: PC skeleton
    pc_result = pc_algorithm(data, variables, alpha, test)
    skeleton = pc_result.skeleton

    # Phase 2: HC restricted to skeleton edges
    domains: dict[str, list] = {}
    for v in variables:
        vals = sorted(set(row[v] for row in data))
        domains[v] = vals

    # Build initial edges from PC orientations
    initial_edges = set(pc_result.oriented_edges)

    # Restricted HC: only add edges that exist in skeleton
    result = _restricted_hill_climbing(
        data, variables, domains, skeleton, initial_edges,
        max_parents, max_iterations,
    )
    return result


def _restricted_hill_climbing(
    data: list[dict],
    variables: list[str],
    domains: dict[str, list],
    allowed_adj: dict[str, set[str]],
    initial_edges: set[tuple[str, str]],
    max_parents: int,
    max_iterations: int,
) -> HillClimbResult:
    """Hill climbing restricted to edges in allowed adjacency."""
    children: dict[str, set[str]] = {v: set() for v in variables}
    parents: dict[str, list[str]] = {v: [] for v in variables}
    edges: set[tuple[str, str]] = set()

    for u, v in initial_edges:
        if v in allowed_adj.get(u, set()):
            edges.add((u, v))
            children[u].add(v)
            parents[v].append(u)

    node_scores: dict[str, float] = {}
    for v in variables:
        node_scores[v] = bic_score(data, v, parents[v], domains)

    iteration = 0
    while iteration < max_iterations:
        best_op = None
        best_delta = 0.0
        iteration += 1

        for u in variables:
            for v in allowed_adj.get(u, set()):
                if u == v:
                    continue

                if (u, v) in edges:
                    # Remove
                    new_parents = [p for p in parents[v] if p != u]
                    delta = bic_score(data, v, new_parents, domains) - node_scores[v]
                    if delta > best_delta:
                        best_delta = delta
                        best_op = ("remove", u, v)

                    # Reverse
                    if v in allowed_adj.get(u, set()) and len(parents[u]) < max_parents:
                        children[u].discard(v)
                        would_cycle = _has_cycle(children, v, u)
                        children[u].add(v)
                        if not would_cycle:
                            np_v = [p for p in parents[v] if p != u]
                            np_u = parents[u] + [v]
                            sv = bic_score(data, v, np_v, domains)
                            su = bic_score(data, u, np_u, domains)
                            delta = (sv + su) - (node_scores[v] + node_scores[u])
                            if delta > best_delta:
                                best_delta = delta
                                best_op = ("reverse", u, v)
                elif (v, u) not in edges:
                    # Add
                    if len(parents[v]) >= max_parents:
                        continue
                    if _has_cycle(children, u, v):
                        continue
                    new_parents = parents[v] + [u]
                    delta = bic_score(data, v, new_parents, domains) - node_scores[v]
                    if delta > best_delta:
                        best_delta = delta
                        best_op = ("add", u, v)

        if best_op is None:
            break

        op, u, v = best_op
        if op == "add":
            edges.add((u, v))
            children[u].add(v)
            parents[v].append(u)
            node_scores[v] = bic_score(data, v, parents[v], domains)
        elif op == "remove":
            edges.discard((u, v))
            children[u].discard(v)
            parents[v] = [p for p in parents[v] if p != u]
            node_scores[v] = bic_score(data, v, parents[v], domains)
        elif op == "reverse":
            edges.discard((u, v))
            children[u].discard(v)
            parents[v] = [p for p in parents[v] if p != u]
            edges.add((v, u))
            children[v].add(u)
            parents[u].append(v)
            node_scores[v] = bic_score(data, v, parents[v], domains)
            node_scores[u] = bic_score(data, u, parents[u], domains)

    total_score = sum(node_scores.values())
    return HillClimbResult(edges, variables, domains, total_score, iteration)


# ---------------------------------------------------------------------------
# Structure evaluation and comparison
# ---------------------------------------------------------------------------

def structural_hamming_distance(
    true_edges: set[tuple[str, str]],
    learned_edges: set[tuple[str, str]],
) -> dict:
    """Compute Structural Hamming Distance (SHD) and related metrics.

    SHD counts: missing edges + extra edges + wrongly oriented edges.
    """
    true_undir = {frozenset(e) for e in true_edges}
    learned_undir = {frozenset(e) for e in learned_edges}

    # Skeleton metrics
    skeleton_tp = len(true_undir & learned_undir)
    skeleton_fp = len(learned_undir - true_undir)
    skeleton_fn = len(true_undir - learned_undir)

    # Direction metrics (among correctly identified edges)
    correct_dir = len(true_edges & learned_edges)
    reversed_dir = 0
    for u, v in true_edges:
        if (v, u) in learned_edges and (u, v) not in learned_edges:
            reversed_dir += 1

    # SHD = missing + extra + reversed
    shd = skeleton_fn + skeleton_fp + reversed_dir

    precision = skeleton_tp / (skeleton_tp + skeleton_fp) if (skeleton_tp + skeleton_fp) > 0 else 0.0
    recall = skeleton_tp / (skeleton_tp + skeleton_fn) if (skeleton_tp + skeleton_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "shd": shd,
        "skeleton_tp": skeleton_tp,
        "skeleton_fp": skeleton_fp,
        "skeleton_fn": skeleton_fn,
        "correct_direction": correct_dir,
        "reversed_direction": reversed_dir,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Data generation from known BN (for testing)
# ---------------------------------------------------------------------------

def sample_from_bn(
    bn: BayesianNetwork,
    n: int,
    seed: int | None = None,
) -> list[dict]:
    """Generate samples from a Bayesian network using forward sampling."""
    import random
    if seed is not None:
        random.seed(seed)

    order = bn.topological_sort()
    samples = []

    for _ in range(n):
        row: dict = {}
        for node in order:
            parent_vals = {p: row[p] for p in bn.parents.get(node, [])}
            # Get conditional distribution
            domain = bn.domains[node]
            probs = []
            for val in domain:
                assignment = {**parent_vals, node: val}
                factor = bn.cpts[node]
                p = factor.get(assignment)
                probs.append(p)

            # Normalize (in case of floating point)
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            # Sample
            r = random.random()
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


# ---------------------------------------------------------------------------
# Convenience: learn and build BN
# ---------------------------------------------------------------------------

def learn_bn_structure(
    data: list[dict],
    method: str = "hc",
    **kwargs,
) -> BayesianNetwork:
    """Learn BN structure from data and estimate parameters.

    Args:
        data: Observational data.
        method: "pc", "hc" (hill climbing), or "hybrid".
        **kwargs: Passed to the learning algorithm.

    Returns:
        BayesianNetwork with learned structure and MLE parameters.
    """
    if not data:
        raise ValueError("Data cannot be empty")

    variables = kwargs.pop("variables", None) or sorted(data[0].keys())

    if method == "pc":
        result = pc_algorithm(data, variables, **kwargs)
        edges = result.to_dag()
    elif method == "hc":
        result = hill_climbing(data, variables, **kwargs)
        edges = result.to_dag()
    elif method == "hybrid":
        result = hybrid_discovery(data, variables, **kwargs)
        edges = result.to_dag()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build BN
    domains: dict[str, list] = {}
    for v in variables:
        domains[v] = sorted(set(row[v] for row in data))

    bn = BayesianNetwork()
    for v in variables:
        bn.add_node(v, domains[v])
    for u, v in edges:
        bn.add_edge(u, v)

    # MLE parameter estimation
    _estimate_parameters_mle(bn, data)

    return bn


def _estimate_parameters_mle(bn: BayesianNetwork, data: list[dict]) -> None:
    """Estimate CPT parameters using Maximum Likelihood Estimation."""
    for node in bn.nodes:
        parents = bn.parents.get(node, [])
        domain = bn.domains[node]

        # Count occurrences
        counts: dict = {}
        parent_totals: dict = {}

        for row in data:
            p_key = tuple(row[p] for p in parents)
            v = row[node]
            jk = (p_key, v)
            counts[jk] = counts.get(jk, 0) + 1
            parent_totals[p_key] = parent_totals.get(p_key, 0) + 1

        # Build CPT with Laplace smoothing
        if not parents:
            cpt = {}
            total = sum(counts.get(((), v), 0) for v in domain) + len(domain)
            for v in domain:
                cpt[v] = (counts.get(((), v), 0) + 1) / total
            bn.set_cpt_dict(node, cpt)
        else:
            cpt = {}
            # Get all parent configurations
            parent_domains = [bn.domains[p] for p in parents]
            for p_vals in itertools.product(*parent_domains):
                p_key = tuple(p_vals)
                total = parent_totals.get(p_key, 0) + len(domain)
                dist = {}
                for v in domain:
                    dist[v] = (counts.get((p_key, v), 0) + 1) / total
                cpt[p_key] = dist
            bn.set_cpt_dict(node, cpt)


def learn_causal_model(
    data: list[dict],
    method: str = "hc",
    **kwargs,
) -> CausalModel:
    """Learn causal model from data. Returns CausalModel wrapping learned BN."""
    bn = learn_bn_structure(data, method, **kwargs)
    return CausalModel(bn)
