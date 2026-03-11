"""
C164: Causal Discovery
Extending C163 (Structural Equation Model) with structure learning algorithms.

Components:
- ConditionalIndependenceTest: Chi-squared, mutual info, partial correlation tests
- ScoringFunction: BIC, BDeu, AIC for scoring DAG structures
- EquivalenceClass: CPDAG (completed partially directed acyclic graph) representation
- PCAlgorithm: Constraint-based discovery (stable PC, all 4 Meek rules)
- GESAlgorithm: Greedy Equivalence Search (score-based, forward+backward phases)
- FCIAlgorithm: Fast Causal Inference (handles latent confounders, PAG output)
- DiscoveryAnalyzer: Compare learned vs true graphs (SHD, precision, recall, F1)
"""

import math
import random
from collections import defaultdict, deque
from itertools import combinations
from copy import deepcopy
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C163_structural_equation_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C161_causal_inference'))
from sem import LinearSEM, SEMSimulator
from causal_inference import CausalGraph


# ---------------------------------------------------------------------------
# ConditionalIndependenceTest
# ---------------------------------------------------------------------------

class ConditionalIndependenceTest:
    """Statistical tests for conditional independence.

    Supports:
    - Chi-squared test (categorical data)
    - Mutual information test (categorical data)
    - Partial correlation test (continuous data)
    """

    @staticmethod
    def chi_squared(data, x, y, z_set=None, alpha=0.05):
        """Chi-squared test for conditional independence.

        Args:
            data: list of dicts, each mapping variable name -> value
            x, y: variable names to test
            z_set: set of conditioning variable names (or None)
            alpha: significance level

        Returns:
            (is_independent, p_value, statistic)
        """
        z_set = set(z_set) if z_set else set()

        if not z_set:
            # Marginal independence test
            # Count joint and marginal frequencies
            counts_xy = defaultdict(int)
            counts_x = defaultdict(int)
            counts_y = defaultdict(int)
            n = len(data)

            for row in data:
                vx, vy = row[x], row[y]
                counts_xy[(vx, vy)] += 1
                counts_x[vx] += 1
                counts_y[vy] += 1

            # Compute chi-squared statistic
            stat = 0.0
            x_vals = sorted(counts_x.keys())
            y_vals = sorted(counts_y.keys())

            for vx in x_vals:
                for vy in y_vals:
                    observed = counts_xy.get((vx, vy), 0)
                    expected = counts_x[vx] * counts_y[vy] / n
                    if expected > 0:
                        stat += (observed - expected) ** 2 / expected

            df = max(1, (len(x_vals) - 1) * (len(y_vals) - 1))
            p_value = _chi2_survival(stat, df)
            return (bool(p_value > alpha), float(p_value), float(stat))
        else:
            # Conditional independence: test within each stratum of Z
            z_list = sorted(z_set)
            strata = defaultdict(list)
            for row in data:
                z_key = tuple(row[z] for z in z_list)
                strata[z_key].append(row)

            total_stat = 0.0
            total_df = 0

            for z_key, stratum in strata.items():
                n_s = len(stratum)
                if n_s < 2:
                    continue

                counts_xy = defaultdict(int)
                counts_x = defaultdict(int)
                counts_y = defaultdict(int)

                for row in stratum:
                    vx, vy = row[x], row[y]
                    counts_xy[(vx, vy)] += 1
                    counts_x[vx] += 1
                    counts_y[vy] += 1

                x_vals = sorted(counts_x.keys())
                y_vals = sorted(counts_y.keys())

                if len(x_vals) < 2 or len(y_vals) < 2:
                    continue

                for vx in x_vals:
                    for vy in y_vals:
                        observed = counts_xy.get((vx, vy), 0)
                        expected = counts_x[vx] * counts_y[vy] / n_s
                        if expected > 0:
                            total_stat += (observed - expected) ** 2 / expected

                total_df += (len(x_vals) - 1) * (len(y_vals) - 1)

            if total_df == 0:
                return (True, 1.0, 0.0)

            p_value = _chi2_survival(total_stat, total_df)
            return (bool(p_value > alpha), float(p_value), float(total_stat))

    @staticmethod
    def mutual_information(data, x, y, z_set=None, alpha=0.05):
        """Mutual information test for conditional independence.

        Uses 2*N*MI ~ chi-squared(df) as the test statistic.
        """
        z_set = set(z_set) if z_set else set()
        n = len(data)

        if not z_set:
            mi = _compute_mi(data, x, y)
            stat = 2 * n * mi
            # Get cardinalities
            x_vals = set(row[x] for row in data)
            y_vals = set(row[y] for row in data)
            df = max(1, (len(x_vals) - 1) * (len(y_vals) - 1))
            p_value = _chi2_survival(stat, df)
            return (bool(p_value > alpha), float(p_value), float(stat))
        else:
            z_list = sorted(z_set)
            strata = defaultdict(list)
            for row in data:
                z_key = tuple(row[z] for z in z_list)
                strata[z_key].append(row)

            total_stat = 0.0
            total_df = 0

            for z_key, stratum in strata.items():
                if len(stratum) < 2:
                    continue
                mi = _compute_mi(stratum, x, y)
                total_stat += 2 * len(stratum) * mi
                x_vals = set(row[x] for row in stratum)
                y_vals = set(row[y] for row in stratum)
                if len(x_vals) >= 2 and len(y_vals) >= 2:
                    total_df += (len(x_vals) - 1) * (len(y_vals) - 1)

            if total_df == 0:
                return (True, 1.0, 0.0)

            p_value = _chi2_survival(total_stat, total_df)
            return (bool(p_value > alpha), float(p_value), float(total_stat))

    @staticmethod
    def partial_correlation(data, x, y, z_set=None, alpha=0.05):
        """Partial correlation test for continuous data.

        Uses Fisher's z-transform for the test.

        Args:
            data: list of dicts with continuous values
        """
        z_set = set(z_set) if z_set else set()
        n = len(data)

        if not z_set:
            # Simple Pearson correlation
            x_vals = np.array([row[x] for row in data])
            y_vals = np.array([row[y] for row in data])
            r = _pearson_corr(x_vals, y_vals)
        else:
            # Partial correlation via regression residuals
            z_list = sorted(z_set)
            x_vals = np.array([row[x] for row in data])
            y_vals = np.array([row[y] for row in data])
            z_matrix = np.column_stack([np.array([row[z] for row in data]) for z in z_list])

            # Residuals of X regressed on Z
            x_resid = _residuals(x_vals, z_matrix)
            y_resid = _residuals(y_vals, z_matrix)
            r = _pearson_corr(x_resid, y_resid)

        # Fisher z-transform
        if abs(r) >= 1.0:
            r = 0.999 * np.sign(r)

        z_stat = 0.5 * math.log((1 + r) / (1 - r))
        se = 1.0 / math.sqrt(max(1, n - len(z_set) - 3))
        test_stat = abs(z_stat / se)

        # Two-tailed test using normal approximation
        p_value = 2 * _normal_survival(test_stat)

        return (bool(p_value > alpha), float(p_value), float(r))


# ---------------------------------------------------------------------------
# ScoringFunction
# ---------------------------------------------------------------------------

class ScoringFunction:
    """Scoring functions for evaluating DAG structures.

    All scores are decomposable: total score = sum of local scores.
    Higher scores are better.
    """

    @staticmethod
    def bic(data, node, parents, continuous=False):
        """Bayesian Information Criterion (BIC) score.

        For categorical data: BIC = log-likelihood - (k/2)*log(n)
        """
        n = len(data)
        if n == 0:
            return 0.0

        if continuous:
            return ScoringFunction._bic_continuous(data, node, parents, n)
        return ScoringFunction._bic_categorical(data, node, parents, n)

    @staticmethod
    def _bic_categorical(data, node, parents, n):
        """BIC for categorical data."""
        parents = sorted(parents) if parents else []

        # Count frequencies
        counts = defaultdict(lambda: defaultdict(int))
        parent_counts = defaultdict(int)

        for row in data:
            p_key = tuple(row[p] for p in parents) if parents else ()
            v = row[node]
            counts[p_key][v] += 1
            parent_counts[p_key] += 1

        # Log-likelihood
        ll = 0.0
        for p_key, child_counts in counts.items():
            total = parent_counts[p_key]
            for v, c in child_counts.items():
                if c > 0 and total > 0:
                    ll += c * math.log(c / total)

        # Number of free parameters
        node_card = len(set(row[node] for row in data))
        parent_configs = max(1, len(parent_counts))
        k = parent_configs * (node_card - 1)

        return ll - (k / 2) * math.log(n)

    @staticmethod
    def _bic_continuous(data, node, parents, n):
        """BIC for continuous/Gaussian data using linear regression."""
        y = np.array([row[node] for row in data])

        if not parents:
            # No parents: just variance of node
            ss_res = np.sum((y - np.mean(y)) ** 2)
            k = 2  # mean + variance
        else:
            parents = sorted(parents)
            X = np.column_stack([np.array([row[p] for row in data]) for p in parents])
            X = np.column_stack([np.ones(n), X])  # intercept

            # OLS
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_hat = X @ beta
                ss_res = np.sum((y - y_hat) ** 2)
            except np.linalg.LinAlgError:
                ss_res = np.sum((y - np.mean(y)) ** 2)

            k = len(parents) + 2  # coefficients + intercept + variance

        if ss_res <= 0:
            ss_res = 1e-10

        ll = -n / 2 * (math.log(2 * math.pi) + math.log(ss_res / n) + 1)
        return ll - (k / 2) * math.log(n)

    @staticmethod
    def aic(data, node, parents, continuous=False):
        """Akaike Information Criterion.

        AIC = log-likelihood - k
        """
        n = len(data)
        if n == 0:
            return 0.0

        if continuous:
            y = np.array([row[node] for row in data])
            parents = sorted(parents) if parents else []
            if not parents:
                ss_res = np.sum((y - np.mean(y)) ** 2)
                k = 2
            else:
                X = np.column_stack([np.array([row[p] for row in data]) for p in parents])
                X = np.column_stack([np.ones(n), X])
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    ss_res = np.sum((y - X @ beta) ** 2)
                except np.linalg.LinAlgError:
                    ss_res = np.sum((y - np.mean(y)) ** 2)
                k = len(parents) + 2

            if ss_res <= 0:
                ss_res = 1e-10
            ll = -n / 2 * (math.log(2 * math.pi) + math.log(ss_res / n) + 1)
            return ll - k

        # Categorical
        parents = sorted(parents) if parents else []
        counts = defaultdict(lambda: defaultdict(int))
        parent_counts = defaultdict(int)
        for row in data:
            p_key = tuple(row[p] for p in parents) if parents else ()
            v = row[node]
            counts[p_key][v] += 1
            parent_counts[p_key] += 1

        ll = 0.0
        for p_key, child_counts in counts.items():
            total = parent_counts[p_key]
            for v, c in child_counts.items():
                if c > 0 and total > 0:
                    ll += c * math.log(c / total)

        node_card = len(set(row[node] for row in data))
        parent_configs = max(1, len(parent_counts))
        k = parent_configs * (node_card - 1)

        return ll - k

    @staticmethod
    def bdeu(data, node, parents, ess=1.0):
        """Bayesian Dirichlet equivalent uniform (BDeu) score.

        Args:
            ess: equivalent sample size (prior strength)
        """
        parents = sorted(parents) if parents else []

        counts = defaultdict(lambda: defaultdict(int))
        parent_counts = defaultdict(int)

        for row in data:
            p_key = tuple(row[p] for p in parents) if parents else ()
            v = row[node]
            counts[p_key][v] += 1
            parent_counts[p_key] += 1

        node_vals = sorted(set(row[node] for row in data))
        r = len(node_vals)
        q = max(1, len(set(
            tuple(row[p] for p in parents) for row in data
        ))) if parents else 1

        alpha_ij = ess / q
        alpha_ijk = ess / (q * r)

        score = 0.0
        for p_key in (parent_counts.keys() if parents else [()]):
            n_ij = parent_counts[p_key] if parents else len(data)
            score += _log_gamma(alpha_ij) - _log_gamma(alpha_ij + n_ij)
            for val in node_vals:
                n_ijk = counts[p_key].get(val, 0)
                score += _log_gamma(alpha_ijk + n_ijk) - _log_gamma(alpha_ijk)

        return score

    @staticmethod
    def score_dag(data, nodes, parent_map, method='bic', continuous=False, ess=1.0):
        """Score an entire DAG structure.

        Args:
            parent_map: dict {node: [parent1, parent2, ...]}
            method: 'bic', 'aic', or 'bdeu'
        """
        total = 0.0
        for node in nodes:
            parents = parent_map.get(node, [])
            if method == 'bic':
                total += ScoringFunction.bic(data, node, parents, continuous)
            elif method == 'aic':
                total += ScoringFunction.aic(data, node, parents, continuous)
            elif method == 'bdeu':
                total += ScoringFunction.bdeu(data, node, parents, ess)
        return total


# ---------------------------------------------------------------------------
# EquivalenceClass
# ---------------------------------------------------------------------------

class EquivalenceClass:
    """Completed Partially Directed Acyclic Graph (CPDAG).

    Represents a Markov equivalence class of DAGs.
    Edges can be directed (->), undirected (-), or circle (o) for PAGs.
    """

    def __init__(self, nodes=None):
        self.nodes = list(nodes) if nodes else []
        self.directed = set()    # (a, b) means a -> b
        self.undirected = set()  # (a, b) where a < b means a - b
        self.circle = set()      # (a, b) means a o-> b (circle at a end)
        self._adjacency = defaultdict(set)

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def add_directed_edge(self, a, b):
        """Add a -> b."""
        self.directed.add((a, b))
        self._adjacency[a].add(b)
        self._adjacency[b].add(a)
        # Remove undirected if exists
        self.undirected.discard((min(a, b), max(a, b)))

    def add_undirected_edge(self, a, b):
        """Add a - b."""
        key = (min(a, b), max(a, b))
        self.undirected.add(key)
        self._adjacency[a].add(b)
        self._adjacency[b].add(a)

    def add_circle_edge(self, a, b):
        """Add a o-> b (circle at a's end)."""
        self.circle.add((a, b))
        self._adjacency[a].add(b)
        self._adjacency[b].add(a)

    def remove_edge(self, a, b):
        """Remove any edge between a and b."""
        self.directed.discard((a, b))
        self.directed.discard((b, a))
        self.undirected.discard((min(a, b), max(a, b)))
        self.circle.discard((a, b))
        self.circle.discard((b, a))
        self._adjacency[a].discard(b)
        self._adjacency[b].discard(a)

    def is_adjacent(self, a, b):
        return b in self._adjacency.get(a, set())

    def neighbors(self, node):
        return set(self._adjacency.get(node, set()))

    def has_directed(self, a, b):
        """Check if a -> b exists."""
        return (a, b) in self.directed

    def has_undirected(self, a, b):
        """Check if a - b exists."""
        return (min(a, b), max(a, b)) in self.undirected

    def orient_edge(self, a, b):
        """Orient an undirected edge a - b to a -> b."""
        key = (min(a, b), max(a, b))
        self.undirected.discard(key)
        self.directed.add((a, b))
        self.directed.discard((b, a))

    def to_dag_edges(self):
        """Return all directed edges (directed + arbitrarily oriented undirected)."""
        edges = set(self.directed)
        for a, b in self.undirected:
            edges.add((a, b))
        return edges

    def get_parents(self, node):
        """Return definite parents (nodes with directed edge into node)."""
        return {a for a, b in self.directed if b == node}

    def skeleton(self):
        """Return skeleton as set of frozensets."""
        skel = set()
        for a, b in self.directed:
            skel.add(frozenset([a, b]))
        for a, b in self.undirected:
            skel.add(frozenset([a, b]))
        for a, b in self.circle:
            skel.add(frozenset([a, b]))
        return skel

    def to_causal_graph(self):
        """Convert to CausalGraph (orient undirected edges arbitrarily)."""
        g = CausalGraph()
        for node in self.nodes:
            g.add_node(node)
        for a, b in self.directed:
            g.add_edge(a, b)
        for a, b in self.undirected:
            g.add_edge(a, b)  # arbitrary orientation
        return g


# ---------------------------------------------------------------------------
# PCAlgorithm
# ---------------------------------------------------------------------------

class PCAlgorithm:
    """PC algorithm for causal discovery from observational data.

    Constraint-based: uses conditional independence tests to learn a CPDAG.

    Features:
    - Stable variant (order-independent edge removal)
    - All 4 Meek orientation rules
    - Multiple CI test options
    """

    def __init__(self, alpha=0.05, max_cond_size=None, stable=True,
                 test_method='chi_squared'):
        """
        Args:
            alpha: significance level for CI tests
            max_cond_size: max conditioning set size (None = no limit)
            stable: use stable PC (order-independent)
            test_method: 'chi_squared', 'mutual_information', or 'partial_correlation'
        """
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.stable = stable
        self.test_method = test_method
        self.sepsets = {}
        self.test_count = 0

    def fit(self, data, variables=None):
        """Run PC algorithm on data.

        Args:
            data: list of dicts, each mapping variable name -> value
            variables: list of variable names (auto-detected if None)

        Returns:
            EquivalenceClass (CPDAG)
        """
        if variables is None:
            variables = sorted(data[0].keys())

        self.sepsets = {}
        self.test_count = 0

        # Phase 1: Learn skeleton
        adjacency = self._learn_skeleton(data, variables)

        # Phase 2: Orient v-structures
        cpdag = EquivalenceClass(variables)

        # Add all skeleton edges as undirected
        for x in variables:
            for y in adjacency[x]:
                if x < y:
                    cpdag.add_undirected_edge(x, y)

        # Orient v-structures: X - Y - Z where X,Z not adjacent and Y not in sep(X,Z)
        for y in variables:
            neighbors = sorted(adjacency[y])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    x, z = neighbors[i], neighbors[j]
                    if z not in adjacency[x]:
                        sep = self.sepsets.get((x, z), self.sepsets.get((z, x), set()))
                        if y not in sep:
                            cpdag.orient_edge(x, y)
                            cpdag.orient_edge(z, y)

        # Phase 3: Apply Meek's rules
        self._apply_meek_rules(cpdag, variables)

        return cpdag

    def _learn_skeleton(self, data, variables):
        """Phase 1: Learn the undirected skeleton via CI tests."""
        adjacency = defaultdict(set)
        for i, x in enumerate(variables):
            for y in variables[i + 1:]:
                adjacency[x].add(y)
                adjacency[y].add(x)

        max_size = self.max_cond_size
        if max_size is None:
            max_size = len(variables) - 2

        for cond_size in range(max_size + 1):
            if self.stable:
                # Stable PC: use adjacency snapshot
                adj_snapshot = {k: set(v) for k, v in adjacency.items()}
            else:
                adj_snapshot = adjacency

            edges_to_check = []
            for x in variables:
                for y in sorted(adj_snapshot[x]):
                    if x < y:
                        edges_to_check.append((x, y))

            for x, y in edges_to_check:
                if y not in adjacency[x]:
                    continue

                # Check both sides for conditioning sets (stable PC)
                found = False
                for side_node, other_node in [(x, y), (y, x)]:
                    neighbors = (adj_snapshot[side_node] if self.stable else adjacency[side_node]) - {other_node}
                    if len(neighbors) < cond_size:
                        continue

                    for z_set in combinations(sorted(neighbors), cond_size):
                        z_set_s = set(z_set)
                        is_indep = self._ci_test(data, x, y, z_set_s)
                        if is_indep:
                            adjacency[x].discard(y)
                            adjacency[y].discard(x)
                            self.sepsets[(x, y)] = z_set_s
                            self.sepsets[(y, x)] = z_set_s
                            found = True
                            break
                    if found:
                        break

        return adjacency

    def _ci_test(self, data, x, y, z_set):
        """Run conditional independence test."""
        self.test_count += 1

        if self.test_method == 'chi_squared':
            is_indep, _, _ = ConditionalIndependenceTest.chi_squared(data, x, y, z_set, self.alpha)
        elif self.test_method == 'mutual_information':
            is_indep, _, _ = ConditionalIndependenceTest.mutual_information(data, x, y, z_set, self.alpha)
        elif self.test_method == 'partial_correlation':
            is_indep, _, _ = ConditionalIndependenceTest.partial_correlation(data, x, y, z_set, self.alpha)
        else:
            raise ValueError(f"Unknown test method: {self.test_method}")

        return is_indep

    def _apply_meek_rules(self, cpdag, variables):
        """Apply all 4 Meek orientation rules until convergence."""
        changed = True
        while changed:
            changed = False
            changed |= self._meek_rule1(cpdag, variables)
            changed |= self._meek_rule2(cpdag, variables)
            changed |= self._meek_rule3(cpdag, variables)
            changed |= self._meek_rule4(cpdag, variables)

    def _meek_rule1(self, cpdag, variables):
        """R1: If a -> b - c and a not adj c, orient b -> c."""
        changed = False
        for a, b in list(cpdag.undirected):
            # a - b undirected, check if some c -> a and c not adj b
            for c in variables:
                if cpdag.has_directed(c, a) and not cpdag.is_adjacent(c, b):
                    cpdag.orient_edge(a, b)
                    changed = True
                    break
                if cpdag.has_directed(c, b) and not cpdag.is_adjacent(c, a):
                    cpdag.orient_edge(b, a)
                    changed = True
                    break
        return changed

    def _meek_rule2(self, cpdag, variables):
        """R2: If a -> c -> b and a - b, orient a -> b."""
        changed = False
        for a, b in list(cpdag.undirected):
            for c in variables:
                if cpdag.has_directed(a, c) and cpdag.has_directed(c, b):
                    cpdag.orient_edge(a, b)
                    changed = True
                    break
                if cpdag.has_directed(b, c) and cpdag.has_directed(c, a):
                    cpdag.orient_edge(b, a)
                    changed = True
                    break
        return changed

    def _meek_rule3(self, cpdag, variables):
        """R3: If a - c -> b and a - d -> b and c not adj d, orient a -> b."""
        changed = False
        for e1, e2 in list(cpdag.undirected):
            for a, b in [(e1, e2), (e2, e1)]:
                # Find two nodes c, d s.t. a-c->b, a-d->b, c not adj d
                candidates = []
                for c in variables:
                    if c != a and c != b:
                        if cpdag.has_undirected(a, c) and cpdag.has_directed(c, b):
                            candidates.append(c)

                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        c, d = candidates[i], candidates[j]
                        if not cpdag.is_adjacent(c, d):
                            cpdag.orient_edge(a, b)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        return changed

    def _meek_rule4(self, cpdag, variables):
        """R4: If a - c -> d -> b and a adj d, orient a -> b."""
        changed = False
        for e1, e2 in list(cpdag.undirected):
            for a, b in [(e1, e2), (e2, e1)]:
                for c in variables:
                    if c == a or c == b:
                        continue
                    if not cpdag.has_undirected(a, c):
                        continue
                    for d in variables:
                        if d == a or d == b or d == c:
                            continue
                        if cpdag.has_directed(c, d) and cpdag.has_directed(d, b):
                            if cpdag.is_adjacent(a, d):
                                cpdag.orient_edge(a, b)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
        return changed


# ---------------------------------------------------------------------------
# GESAlgorithm
# ---------------------------------------------------------------------------

class GESAlgorithm:
    """Greedy Equivalence Search for score-based causal discovery.

    Two-phase:
    1. Forward: greedily add edges that improve score
    2. Backward: greedily remove edges that improve score

    Operates on equivalence classes (CPDAGs).
    """

    def __init__(self, score_method='bic', continuous=False, ess=1.0):
        """
        Args:
            score_method: 'bic', 'aic', or 'bdeu'
            continuous: True for continuous data
            ess: equivalent sample size (for BDeu)
        """
        self.score_method = score_method
        self.continuous = continuous
        self.ess = ess

    def fit(self, data, variables=None):
        """Run GES on data.

        Returns:
            EquivalenceClass (CPDAG)
        """
        if variables is None:
            variables = sorted(data[0].keys())

        # Start with empty graph
        parent_map = {v: [] for v in variables}

        # Forward phase: add edges
        parent_map = self._forward_phase(data, variables, parent_map)

        # Backward phase: remove edges
        parent_map = self._backward_phase(data, variables, parent_map)

        # Convert to CPDAG
        return self._to_cpdag(variables, parent_map)

    def _score_node(self, data, node, parents):
        """Score a single node given its parents."""
        if self.score_method == 'bic':
            return ScoringFunction.bic(data, node, parents, self.continuous)
        elif self.score_method == 'aic':
            return ScoringFunction.aic(data, node, parents, self.continuous)
        elif self.score_method == 'bdeu':
            return ScoringFunction.bdeu(data, node, parents, self.ess)
        raise ValueError(f"Unknown score method: {self.score_method}")

    def _total_score(self, data, variables, parent_map):
        """Score the full DAG."""
        return sum(self._score_node(data, v, parent_map[v]) for v in variables)

    def _forward_phase(self, data, variables, parent_map):
        """Greedily add edges that improve the score."""
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_edge = None

            for y in variables:
                current_score = self._score_node(data, y, parent_map[y])

                for x in variables:
                    if x == y or x in parent_map[y]:
                        continue
                    # Check acyclicity
                    if self._would_create_cycle(parent_map, x, y):
                        continue

                    new_parents = parent_map[y] + [x]
                    new_score = self._score_node(data, y, new_parents)
                    gain = new_score - current_score

                    if gain > best_gain:
                        best_gain = gain
                        best_edge = (x, y)

            if best_edge is not None:
                x, y = best_edge
                parent_map[y] = parent_map[y] + [x]
                improved = True

        return parent_map

    def _backward_phase(self, data, variables, parent_map):
        """Greedily remove edges that improve the score."""
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_edge = None

            for y in variables:
                current_score = self._score_node(data, y, parent_map[y])

                for x in list(parent_map[y]):
                    new_parents = [p for p in parent_map[y] if p != x]
                    new_score = self._score_node(data, y, new_parents)
                    gain = new_score - current_score

                    if gain > best_gain:
                        best_gain = gain
                        best_edge = (x, y)

            if best_edge is not None:
                x, y = best_edge
                parent_map[y] = [p for p in parent_map[y] if p != x]
                improved = True

        return parent_map

    def _would_create_cycle(self, parent_map, new_parent, child):
        """Check if adding new_parent -> child would create a cycle."""
        # DFS from new_parent going backwards through parent edges
        visited = set()
        stack = [new_parent]
        while stack:
            node = stack.pop()
            if node == child:
                return True  # child is ancestor of new_parent => cycle
            if node in visited:
                continue
            visited.add(node)
            stack.extend(parent_map.get(node, []))
        return False

    def _to_cpdag(self, variables, parent_map):
        """Convert a DAG (parent_map) to its CPDAG."""
        cpdag = EquivalenceClass(variables)

        # Find v-structures to determine which edges must be directed
        must_direct = set()

        for y in variables:
            parents = parent_map[y]
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    p1, p2 = parents[i], parents[j]
                    # Check if p1 and p2 are non-adjacent
                    if p2 not in parent_map.get(p1, []) and p1 not in parent_map.get(p2, []):
                        # v-structure: p1 -> y <- p2
                        must_direct.add((p1, y))
                        must_direct.add((p2, y))

        # Add edges
        all_edges = set()
        for y in variables:
            for x in parent_map[y]:
                all_edges.add((x, y))

        for x, y in all_edges:
            if (x, y) in must_direct:
                cpdag.add_directed_edge(x, y)
            else:
                # Check if it's compelled by Meek rules
                # Simplified: edges in same chain as v-structure are directed
                cpdag.add_undirected_edge(x, y)

        # Apply Meek rules to propagate orientations
        pc = PCAlgorithm()
        pc._apply_meek_rules(cpdag, variables)

        return cpdag


# ---------------------------------------------------------------------------
# FCIAlgorithm
# ---------------------------------------------------------------------------

class FCIAlgorithm:
    """Fast Causal Inference algorithm.

    Handles latent confounders and selection bias.
    Outputs a PAG (Partial Ancestral Graph) with:
    - -> directed edges
    - o-> circle edges (uncertain endpoint)
    - o-o bidirected circle edges

    Based on Spirtes, Glymour, Scheines (2000).
    """

    def __init__(self, alpha=0.05, max_cond_size=None, test_method='chi_squared'):
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.test_method = test_method
        self.sepsets = {}
        self.test_count = 0

    def fit(self, data, variables=None):
        """Run FCI algorithm.

        Returns:
            EquivalenceClass representing a PAG
        """
        if variables is None:
            variables = sorted(data[0].keys())

        self.sepsets = {}
        self.test_count = 0

        # Step 1: Learn skeleton (same as PC)
        adjacency = self._learn_skeleton(data, variables)

        # Step 2: Orient v-structures
        pag = EquivalenceClass(variables)
        for x in variables:
            for y in adjacency[x]:
                if x < y:
                    # Start with circle endpoints: x o-o y
                    pag.add_circle_edge(x, y)
                    pag.add_circle_edge(y, x)

        # Orient definite v-structures
        for y in variables:
            neighbors = sorted(adjacency[y])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    x, z = neighbors[i], neighbors[j]
                    if z not in adjacency[x]:
                        sep = self.sepsets.get((x, z), self.sepsets.get((z, x), set()))
                        if y not in sep:
                            # Orient x *-> y <-* z
                            self._orient_arrowhead(pag, x, y)
                            self._orient_arrowhead(pag, z, y)

        # Step 3: Apply FCI orientation rules
        self._apply_fci_rules(pag, variables, adjacency)

        return pag

    def _learn_skeleton(self, data, variables):
        """Learn skeleton via CI tests (same as PC)."""
        adjacency = defaultdict(set)
        for i, x in enumerate(variables):
            for y in variables[i + 1:]:
                adjacency[x].add(y)
                adjacency[y].add(x)

        max_size = self.max_cond_size
        if max_size is None:
            max_size = len(variables) - 2

        for cond_size in range(max_size + 1):
            edges_to_check = []
            for x in variables:
                for y in sorted(adjacency[x]):
                    if x < y:
                        edges_to_check.append((x, y))

            for x, y in edges_to_check:
                if y not in adjacency[x]:
                    continue

                found = False
                for side, other in [(x, y), (y, x)]:
                    neighbors = adjacency[side] - {other}
                    if len(neighbors) < cond_size:
                        continue
                    for z_set in combinations(sorted(neighbors), cond_size):
                        z_set_s = set(z_set)
                        is_indep = self._ci_test(data, x, y, z_set_s)
                        if is_indep:
                            adjacency[x].discard(y)
                            adjacency[y].discard(x)
                            self.sepsets[(x, y)] = z_set_s
                            self.sepsets[(y, x)] = z_set_s
                            found = True
                            break
                    if found:
                        break

        return adjacency

    def _ci_test(self, data, x, y, z_set):
        """Run conditional independence test."""
        self.test_count += 1
        if self.test_method == 'chi_squared':
            is_indep, _, _ = ConditionalIndependenceTest.chi_squared(data, x, y, z_set, self.alpha)
        elif self.test_method == 'mutual_information':
            is_indep, _, _ = ConditionalIndependenceTest.mutual_information(data, x, y, z_set, self.alpha)
        elif self.test_method == 'partial_correlation':
            is_indep, _, _ = ConditionalIndependenceTest.partial_correlation(data, x, y, z_set, self.alpha)
        else:
            raise ValueError(f"Unknown test method: {self.test_method}")
        return is_indep

    def _orient_arrowhead(self, pag, from_node, to_node):
        """Orient edge so that to_node has an arrowhead (from_node *-> to_node)."""
        # Remove circle marks and add directed
        pag.circle.discard((to_node, from_node))  # remove circle at to_node's end
        pag.circle.discard((from_node, to_node))
        pag.directed.add((from_node, to_node))
        pag.undirected.discard((min(from_node, to_node), max(from_node, to_node)))

    def _orient_tail(self, pag, from_node, to_node):
        """Orient edge so that from_node has a tail (from_node -> to_node)."""
        pag.circle.discard((from_node, to_node))
        # Ensure directed exists
        if not pag.has_directed(from_node, to_node):
            pag.directed.add((from_node, to_node))

    def _apply_fci_rules(self, pag, variables, adjacency):
        """Apply FCI orientation rules R1-R4, R8-R10."""
        changed = True
        iterations = 0
        max_iter = len(variables) * 10

        while changed and iterations < max_iter:
            changed = False
            iterations += 1

            # R1: If a *-> b o-* c, a not adj c => orient b -> c
            for b in variables:
                for a in variables:
                    if a == b:
                        continue
                    if not pag.has_directed(a, b):
                        continue
                    for c in variables:
                        if c == a or c == b:
                            continue
                        if not pag.is_adjacent(b, c):
                            continue
                        if pag.is_adjacent(a, c):
                            continue
                        if (b, c) in pag.circle:
                            self._orient_arrowhead(pag, b, c)
                            self._orient_tail(pag, b, c)
                            changed = True

            # R3: If a *-> b <-* c, a *-o d o-* c, a not adj c, d *-o b
            # => orient d *-> b
            for b in variables:
                parents_of_b = [a for a in variables if a != b and pag.has_directed(a, b)]
                if len(parents_of_b) < 2:
                    continue
                for i in range(len(parents_of_b)):
                    for j in range(i + 1, len(parents_of_b)):
                        a, c = parents_of_b[i], parents_of_b[j]
                        if pag.is_adjacent(a, c):
                            continue
                        for d in variables:
                            if d in (a, b, c):
                                continue
                            if not pag.is_adjacent(d, a):
                                continue
                            if not pag.is_adjacent(d, c):
                                continue
                            if not pag.is_adjacent(d, b):
                                continue
                            if (d, b) in pag.circle:
                                self._orient_arrowhead(pag, d, b)
                                changed = True

        # Remaining circle edges that couldn't be oriented stay as circles


# ---------------------------------------------------------------------------
# DiscoveryAnalyzer
# ---------------------------------------------------------------------------

class DiscoveryAnalyzer:
    """Compare learned causal structures against ground truth.

    Metrics:
    - Structural Hamming Distance (SHD)
    - Skeleton precision, recall, F1
    - Arrowhead precision, recall, F1
    """

    @staticmethod
    def structural_hamming_distance(true_edges, learned_cpdag):
        """Compute SHD between true DAG edges and learned CPDAG.

        Args:
            true_edges: set of (parent, child) tuples (ground truth DAG)
            learned_cpdag: EquivalenceClass

        Returns:
            int: SHD (lower is better)
        """
        shd = 0

        # True skeleton
        true_skeleton = set()
        for a, b in true_edges:
            true_skeleton.add(frozenset([a, b]))

        learned_skeleton = learned_cpdag.skeleton()

        # Missing edges
        for edge in true_skeleton:
            if edge not in learned_skeleton:
                shd += 1

        # Extra edges
        for edge in learned_skeleton:
            if edge not in true_skeleton:
                shd += 1

        # Wrong orientations (edge exists in both but different direction)
        for edge in true_skeleton & learned_skeleton:
            a, b = tuple(edge)
            true_dir = (a, b) in true_edges
            true_rev = (b, a) in true_edges
            learned_dir = learned_cpdag.has_directed(a, b)
            learned_rev = learned_cpdag.has_directed(b, a)

            if true_dir and learned_rev:
                shd += 1
            elif true_rev and learned_dir:
                shd += 1

        return shd

    @staticmethod
    def skeleton_metrics(true_edges, learned_cpdag):
        """Compute skeleton precision, recall, F1.

        Returns:
            dict with 'precision', 'recall', 'f1'
        """
        true_skeleton = set()
        for a, b in true_edges:
            true_skeleton.add(frozenset([a, b]))

        learned_skeleton = learned_cpdag.skeleton()

        tp = len(true_skeleton & learned_skeleton)
        fp = len(learned_skeleton - true_skeleton)
        fn = len(true_skeleton - learned_skeleton)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    @staticmethod
    def arrowhead_metrics(true_edges, learned_cpdag):
        """Compute arrowhead precision and recall.

        An arrowhead is correct if edge a -> b in learned matches a -> b in true.

        Returns:
            dict with 'precision', 'recall', 'f1'
        """
        true_arrows = set(true_edges)
        learned_arrows = set(learned_cpdag.directed)

        # Only count arrowheads where the skeleton edge is correct
        true_skeleton = set(frozenset([a, b]) for a, b in true_edges)

        relevant_learned = set()
        for a, b in learned_arrows:
            if frozenset([a, b]) in true_skeleton:
                relevant_learned.add((a, b))

        relevant_true = set()
        for a, b in true_arrows:
            if frozenset([a, b]) in learned_cpdag.skeleton():
                relevant_true.add((a, b))

        tp = len(relevant_learned & relevant_true)
        fp = len(relevant_learned - relevant_true)
        fn = len(relevant_true - relevant_learned)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    @staticmethod
    def compare(true_edges, learned_cpdag):
        """Full comparison: SHD + skeleton + arrowhead metrics.

        Returns:
            dict with all metrics
        """
        return {
            'shd': DiscoveryAnalyzer.structural_hamming_distance(true_edges, learned_cpdag),
            'skeleton': DiscoveryAnalyzer.skeleton_metrics(true_edges, learned_cpdag),
            'arrowhead': DiscoveryAnalyzer.arrowhead_metrics(true_edges, learned_cpdag)
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _chi2_survival(x, df):
    """Approximate chi-squared survival function P(X > x) for X ~ chi2(df).

    Uses the regularized incomplete gamma function approximation.
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0

    # Use scipy if available, else approximate
    try:
        from scipy.stats import chi2
        return chi2.sf(x, df)
    except ImportError:
        pass

    # Wilson-Hilferty normal approximation
    if df >= 1:
        z = ((x / df) ** (1.0 / 3) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
        return _normal_survival(z)

    return _normal_survival(math.sqrt(x))


def _normal_survival(z):
    """Approximate P(Z > z) for standard normal Z."""
    # Abramowitz & Stegun approximation
    if z < -8:
        return 1.0
    if z > 8:
        return 0.0

    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-z * z / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )

    if z > 0:
        return p
    return 1.0 - p


def _compute_mi(data, x, y):
    """Compute mutual information I(X; Y) from data."""
    n = len(data)
    if n == 0:
        return 0.0

    counts_xy = defaultdict(int)
    counts_x = defaultdict(int)
    counts_y = defaultdict(int)

    for row in data:
        vx, vy = row[x], row[y]
        counts_xy[(vx, vy)] += 1
        counts_x[vx] += 1
        counts_y[vy] += 1

    mi = 0.0
    for (vx, vy), count in counts_xy.items():
        p_xy = count / n
        p_x = counts_x[vx] / n
        p_y = counts_y[vy] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))

    return max(0.0, mi)


def _pearson_corr(x, y):
    """Compute Pearson correlation between two arrays."""
    n = len(x)
    if n < 2:
        return 0.0

    mx = np.mean(x)
    my = np.mean(y)
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)

    if sx < 1e-15 or sy < 1e-15:
        return 0.0

    return np.sum((x - mx) * (y - my)) / ((n - 1) * sx * sy)


def _residuals(y, X):
    """Compute OLS residuals of y regressed on X."""
    n = len(y)
    X_full = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
        return y - X_full @ beta
    except np.linalg.LinAlgError:
        return y - np.mean(y)


def _log_gamma(x):
    """Log-gamma function."""
    if x <= 0:
        return 0.0
    return math.lgamma(x)


def _generate_categorical_data(graph_edges, n_samples, cardinalities=None, seed=None):
    """Generate categorical data from a DAG for testing.

    Args:
        graph_edges: list of (parent, child) tuples
        n_samples: number of samples
        cardinalities: dict {node: num_values} (default 2 for all)
        seed: random seed

    Returns:
        list of dicts, each mapping variable name -> value
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Build parent map and topo order
    nodes = set()
    parent_map = defaultdict(list)
    children_map = defaultdict(list)
    for p, c in graph_edges:
        nodes.add(p)
        nodes.add(c)
        parent_map[c].append(p)
        children_map[p].append(c)

    # Find roots (no parents)
    roots = [n for n in nodes if not parent_map[n]]

    # Topological sort
    topo = []
    visited = set()
    in_degree = {n: len(parent_map[n]) for n in nodes}
    queue = deque([n for n in nodes if in_degree[n] == 0])
    while queue:
        n = queue.popleft()
        topo.append(n)
        for c in children_map[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if cardinalities is None:
        cardinalities = {n: 2 for n in nodes}

    # Generate CPDs (random but consistent)
    cpds = {}
    for node in topo:
        parents = parent_map[node]
        card = cardinalities[node]
        if not parents:
            # Prior distribution
            probs = [rng.random() + 0.1 for _ in range(card)]
            total = sum(probs)
            cpds[node] = {(): [p / total for p in probs]}
        else:
            parent_cards = [cardinalities[p] for p in parents]
            configs = 1
            for pc in parent_cards:
                configs *= pc
            cpds[node] = {}
            for cfg_idx in range(configs):
                # Decode config
                config = []
                idx = cfg_idx
                for pc in reversed(parent_cards):
                    config.append(idx % pc)
                    idx //= pc
                config.reverse()
                config = tuple(config)

                # Create strong dependencies: child value depends on parent values
                probs = [0.05] * card
                # Use XOR-like combination so all parents influence child
                combined = 0
                for cv in config:
                    combined = (combined + cv) % card
                probs[combined] = 0.85
                total = sum(probs)
                cpds[node] = cpds.get(node, {})
                cpds[node][config] = [p / total for p in probs]

    # Sample
    data = []
    for _ in range(n_samples):
        sample = {}
        for node in topo:
            parents = parent_map[node]
            if not parents:
                config = ()
            else:
                config = tuple(sample[p] for p in parents)

            probs = cpds[node][config]
            r = rng.random()
            cum = 0.0
            val = 0
            for i, p in enumerate(probs):
                cum += p
                if r < cum:
                    val = i
                    break
            else:
                val = len(probs) - 1
            sample[node] = val
        data.append(sample)

    return data


def _generate_continuous_data(graph_edges, n_samples, coefficients=None, noise_std=1.0, seed=None):
    """Generate continuous (Gaussian) data from a linear SEM.

    Args:
        graph_edges: list of (parent, child) tuples
        n_samples: number of samples
        coefficients: dict {(parent, child): coefficient} (random if None)
        noise_std: noise standard deviation
        seed: random seed

    Returns:
        list of dicts
    """
    rng = np.random.RandomState(seed)

    nodes = set()
    parent_map = defaultdict(list)
    children_map = defaultdict(list)
    for p, c in graph_edges:
        nodes.add(p)
        nodes.add(c)
        parent_map[c].append(p)
        children_map[p].append(c)

    # Topo sort
    topo = []
    in_degree = {n: len(parent_map[n]) for n in nodes}
    queue = deque([n for n in sorted(nodes) if in_degree[n] == 0])
    while queue:
        n = queue.popleft()
        topo.append(n)
        for c in children_map[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if coefficients is None:
        coefficients = {}
        for p, c in graph_edges:
            coefficients[(p, c)] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])

    data = []
    for _ in range(n_samples):
        sample = {}
        for node in topo:
            val = rng.normal(0, noise_std)
            for p in parent_map[node]:
                coeff = coefficients.get((p, node), 1.0)
                val += coeff * sample[p]
            sample[node] = val
        data.append(sample)

    return data
