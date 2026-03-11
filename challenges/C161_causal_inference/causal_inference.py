"""
C161: Causal Inference
Composing C160 (Probabilistic Graphical Models) for causal reasoning.

Components:
- CausalGraph: DAG with causal semantics, graph surgery, d-separation
- Intervention: do(X=x) operator (mutilated graph construction)
- BackdoorCriterion: identify and compute backdoor adjustment sets
- FrontdoorCriterion: identify and compute frontdoor adjustment
- DoCalculus: the 3 rules of do-calculus with identification
- CounterfactualEngine: twin-network counterfactuals (abduction-action-prediction)
- InstrumentalVariable: IV estimation (Wald estimator)
- CausalDiscovery: constraint-based structure learning (PC, FCI skeleton)
- MediationAnalysis: direct/indirect effect decomposition
- CausalUtils: ATE, CATE, bounds, sensitivity
"""

import math
import random
from collections import defaultdict, deque
from itertools import product as iter_product
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C160_probabilistic_graphical_models'))
from pgm import Factor, BayesianNetwork, PGMUtils


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------

class CausalGraph:
    """A directed acyclic graph with causal semantics.

    Wraps a BayesianNetwork but adds causal operations:
    graph surgery (do-operator), d-separation queries,
    ancestor/descendant lookups, and path analysis.
    """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.cardinalities = {}
        self.cpds = {}  # node -> Factor
        self._children = defaultdict(list)
        self._parents = defaultdict(list)
        self._bidirected = []  # latent confounders as bidirected edges

    def add_node(self, name, cardinality=2):
        if name not in self.nodes:
            self.nodes.append(name)
            self.cardinalities[name] = cardinality

    def add_edge(self, parent, child):
        self.edges.append((parent, child))
        self._children[parent].append(child)
        self._parents[child].append(parent)

    def add_bidirected(self, a, b):
        """Add a bidirected edge (latent common cause)."""
        self._bidirected.append((a, b))

    def set_cpd(self, node, cpd_factor):
        self.cpds[node] = cpd_factor

    def get_parents(self, node):
        return list(self._parents[node])

    def get_children(self, node):
        return list(self._children[node])

    def ancestors(self, node):
        """Return all ancestors of node (not including node itself)."""
        visited = set()
        queue = deque(self._parents[node])
        while queue:
            n = queue.popleft()
            if n not in visited:
                visited.add(n)
                queue.extend(self._parents[n])
        return visited

    def descendants(self, node):
        """Return all descendants of node (not including node itself)."""
        visited = set()
        queue = deque(self._children[node])
        while queue:
            n = queue.popleft()
            if n not in visited:
                visited.add(n)
                queue.extend(self._children[n])
        return visited

    def topological_sort(self):
        in_degree = defaultdict(int)
        for node in self.nodes:
            in_degree[node] = 0
        for p, c in self.edges:
            in_degree[c] += 1
        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        result = []
        while queue:
            n = queue.popleft()
            result.append(n)
            for c in self._children[n]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        return result

    def is_d_separated(self, x, y, z):
        """Test d-separation: X _||_ Y | Z using Bayes-Ball algorithm."""
        if isinstance(x, str):
            x = {x}
        if isinstance(y, str):
            y = {y}
        if isinstance(z, str):
            z = {z}
        x, y, z = set(x), set(y), set(z)

        reachable = self._bayes_ball_reachable(x, z)
        return len(reachable & y) == 0

    def _bayes_ball_reachable(self, source, observed):
        """Bayes-Ball algorithm to find reachable nodes."""
        visited = set()
        # (node, direction): direction is 'up' (from child) or 'down' (from parent)
        queue = deque()
        for s in source:
            queue.append((s, 'up'))
            queue.append((s, 'down'))

        reachable = set()
        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            reachable.add(node)

            if direction == 'up' and node not in observed:
                # Pass through non-observed node upward to parents
                for p in self._parents[node]:
                    queue.append((p, 'up'))
                # Pass down to children
                for c in self._children[node]:
                    queue.append((c, 'down'))
            elif direction == 'down':
                if node not in observed:
                    # Non-observed: pass down to children
                    for c in self._children[node]:
                        queue.append((c, 'down'))
                # Observed or not: v-structure activation (pass up from collider)
                if node in observed:
                    for p in self._parents[node]:
                        queue.append((p, 'up'))

        return reachable - source

    def mutilate(self, interventions):
        """Return a new CausalGraph with incoming edges to intervention targets removed.

        This implements the graph surgery for do(X=x): remove all edges into X.
        interventions: dict mapping node -> fixed value
        """
        new_graph = CausalGraph()
        for n in self.nodes:
            new_graph.add_node(n, self.cardinalities[n])

        intervened = set(interventions.keys())
        for p, c in self.edges:
            if c not in intervened:
                new_graph.add_edge(p, c)

        # Copy CPDs for non-intervened nodes
        for n in self.nodes:
            if n in self.cpds and n not in intervened:
                new_graph.set_cpd(n, self.cpds[n])

        # Set delta CPDs for intervened nodes
        for node, value in interventions.items():
            card = self.cardinalities[node]
            vals = np.zeros(card)
            vals[value] = 1.0
            new_graph.set_cpd(node, Factor([node], {node: card}, vals))

        # Copy bidirected edges (latent confounders unaffected by intervention)
        for a, b in self._bidirected:
            new_graph.add_bidirected(a, b)

        return new_graph

    def to_bayesian_network(self):
        """Convert to a BayesianNetwork for inference."""
        bn = BayesianNetwork()
        for n in self.nodes:
            bn.add_node(n, self.cardinalities[n])
        for p, c in self.edges:
            bn.add_edge(p, c)
        for n, cpd in self.cpds.items():
            bn.set_cpd(n, cpd)
        return bn

    def interventional_query(self, query_var, interventions, evidence=None):
        """Compute P(query_var | do(interventions), evidence).

        1. Mutilate graph (remove edges into intervened nodes)
        2. Set intervened nodes to delta distributions
        3. Run variable elimination on mutilated graph
        """
        mutilated = self.mutilate(interventions)
        bn = mutilated.to_bayesian_network()

        # Merge intervention values into evidence
        full_evidence = dict(interventions)
        if evidence:
            full_evidence.update(evidence)

        result = bn.variable_elimination(query_var, full_evidence)
        result.normalize()
        return result

    def sample_interventional(self, interventions, n_samples=1000):
        """Sample from the interventional distribution P(V | do(X=x))."""
        mutilated = self.mutilate(interventions)
        bn = mutilated.to_bayesian_network()
        return bn.sample(n_samples, evidence=interventions)

    def all_paths(self, source, target):
        """Find all directed paths from source to target."""
        paths = []
        def dfs(current, path):
            if current == target:
                paths.append(list(path))
                return
            for c in self._children[current]:
                if c not in path:
                    path.append(c)
                    dfs(c, path)
                    path.pop()
        dfs(source, [source])
        return paths

    def has_directed_path(self, source, target):
        """Check if there's a directed path from source to target."""
        visited = set()
        queue = deque([source])
        while queue:
            n = queue.popleft()
            if n == target:
                return True
            if n in visited:
                continue
            visited.add(n)
            queue.extend(self._children[n])
        return False


# ---------------------------------------------------------------------------
# Intervention (do-operator)
# ---------------------------------------------------------------------------

class Intervention:
    """Represents a do() intervention on variables."""

    def __init__(self, assignments):
        """assignments: dict mapping variable -> value."""
        self.assignments = dict(assignments)

    def apply(self, causal_graph):
        """Apply intervention to a CausalGraph, returning mutilated graph."""
        return causal_graph.mutilate(self.assignments)

    def __repr__(self):
        parts = [f"do({v}={val})" for v, val in self.assignments.items()]
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# BackdoorCriterion
# ---------------------------------------------------------------------------

class BackdoorCriterion:
    """Identifies valid backdoor adjustment sets and computes adjusted estimates."""

    @staticmethod
    def find_adjustment_set(graph, treatment, outcome):
        """Find a valid backdoor adjustment set (if one exists).

        Z satisfies the backdoor criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X
        2. Z blocks every path between X and Y that contains an arrow into X
        """
        desc_x = graph.descendants(treatment)
        # Candidates: all nodes except treatment, outcome, and descendants of treatment
        candidates = [n for n in graph.nodes
                      if n != treatment and n != outcome and n not in desc_x]

        # Try minimal sets first (empty, singletons, pairs, ...)
        for size in range(len(candidates) + 1):
            for subset in _combinations(candidates, size):
                z_set = set(subset)
                if BackdoorCriterion.is_valid(graph, treatment, outcome, z_set):
                    return z_set
        return None

    @staticmethod
    def is_valid(graph, treatment, outcome, z_set):
        """Check if z_set satisfies the backdoor criterion for (treatment, outcome)."""
        desc_x = graph.descendants(treatment)
        # Condition 1: no descendant of X in Z
        if z_set & desc_x:
            return False
        # Condition 2: Z blocks all backdoor paths
        # Create a graph without edges out of treatment for testing
        # Actually: test if X and Y are d-separated given Z in the graph
        # where we remove all edges OUT of X
        test_graph = CausalGraph()
        for n in graph.nodes:
            test_graph.add_node(n, graph.cardinalities[n])
        for p, c in graph.edges:
            if p != treatment:  # Remove edges out of treatment
                test_graph.add_edge(p, c)
        # Also include bidirected edges as pairs of directed edges through a latent
        for a, b in graph._bidirected:
            latent = f"__L_{a}_{b}"
            test_graph.add_node(latent, 2)
            test_graph.add_edge(latent, a)
            test_graph.add_edge(latent, b)

        return test_graph.is_d_separated(treatment, outcome, z_set)

    @staticmethod
    def adjust(graph, treatment, outcome, z_set, treatment_value):
        """Compute P(Y | do(X=x)) using backdoor adjustment.

        P(Y=y | do(X=x)) = sum_z P(Y=y | X=x, Z=z) P(Z=z)
        """
        bn = graph.to_bayesian_network()
        card_y = graph.cardinalities[outcome]
        result = np.zeros(card_y)

        if not z_set:
            # No adjustment needed -- just condition
            evidence = {treatment: treatment_value}
            factor = bn.variable_elimination(outcome, evidence)
            factor.normalize()
            return factor

        z_list = sorted(z_set)
        z_cards = [graph.cardinalities[z] for z in z_list]

        for z_vals in iter_product(*[range(c) for c in z_cards]):
            z_assignment = dict(zip(z_list, z_vals))

            # P(Y | X=x, Z=z)
            evidence = {treatment: treatment_value}
            evidence.update(z_assignment)
            py_given_xz = bn.variable_elimination(outcome, evidence)
            py_given_xz.normalize()

            # P(Z=z)
            if len(z_list) == 1:
                pz = bn.variable_elimination(z_list[0], {})
                pz.normalize()
                pz_val = pz.values[z_vals[0]]
            else:
                # Joint P(Z=z) by chain rule
                pz_val = 1.0
                remaining = dict(z_assignment)
                for z_var in z_list:
                    others = {k: v for k, v in remaining.items() if k != z_var}
                    pz_i = bn.variable_elimination(z_var, others)
                    pz_i.normalize()
                    pz_val *= pz_i.values[remaining[z_var]]

                    # This is approximate; for exact, we'd need joint factor
                    # Actually let's do it properly via sequential elimination
                pz_val = _joint_prob(bn, z_assignment)

            for y_val in range(card_y):
                result[y_val] += float(py_given_xz.values[y_val]) * pz_val

        # Normalize
        total = result.sum()
        if total > 0:
            result /= total

        return Factor([outcome], {outcome: card_y}, result)


def _joint_prob(bn, assignment):
    """Compute P(assignment) from a Bayesian network."""
    # Use chain rule: P(X1=x1, ..., Xn=xn) = product of P(Xi | parents(Xi))
    prob = 1.0
    for node in bn.nodes:
        if node in assignment:
            parents = bn.get_parents(node)
            if node in bn.cpds:
                cpd = bn.cpds[node]
                parent_assignment = {p: assignment[p] for p in parents if p in assignment}
                reduced = cpd.reduce(parent_assignment)
                if node in reduced.variables:
                    idx = reduced.variables.index(node)
                    # Get the probability of the assigned value
                    full_assign = dict(parent_assignment)
                    full_assign[node] = assignment[node]
                    prob *= cpd.get_value(full_assign)
    return prob


# ---------------------------------------------------------------------------
# FrontdoorCriterion
# ---------------------------------------------------------------------------

class FrontdoorCriterion:
    """Identifies and applies frontdoor adjustment."""

    @staticmethod
    def find_mediator_set(graph, treatment, outcome):
        """Find a set M satisfying the frontdoor criterion:
        1. X blocks all directed paths from X to M (trivially, X intercepts all paths)
           Actually: M intercepts all directed paths from X to Y
        2. No unblocked backdoor path from X to M
        3. All backdoor paths from M to Y are blocked by X
        """
        desc_x = graph.descendants(treatment)
        anc_y = graph.ancestors(outcome)

        # Candidates: nodes that are descendants of X and ancestors of Y (mediators)
        candidates = [n for n in graph.nodes
                      if n != treatment and n != outcome
                      and n in desc_x and n in anc_y]

        for size in range(1, len(candidates) + 1):
            for subset in _combinations(candidates, size):
                m_set = set(subset)
                if FrontdoorCriterion.is_valid(graph, treatment, outcome, m_set):
                    return m_set
        return None

    @staticmethod
    def is_valid(graph, treatment, outcome, m_set):
        """Check if m_set satisfies the frontdoor criterion."""
        # 1. X intercepts all directed paths from X to Y through M
        # (all directed paths from X to Y go through M)
        all_paths = graph.all_paths(treatment, outcome)
        for path in all_paths:
            intermediates = set(path[1:-1])  # exclude X and Y
            if not (intermediates & m_set):
                return False  # Path not intercepted by M

        # 2. No unblocked backdoor from X to M (blocked by empty set)
        # Actually: no backdoor path from X to M, meaning X d-separates M from
        # non-descendants in the manipulated graph. Simpler check:
        # no non-causal path from X to any m in M
        for m in m_set:
            # In graph with edges out of X removed, X and M should be d-separated
            test_graph = CausalGraph()
            for n in graph.nodes:
                test_graph.add_node(n, graph.cardinalities[n])
            for p, c in graph.edges:
                if p != treatment:
                    test_graph.add_edge(p, c)
            for a, b in graph._bidirected:
                latent = f"__L_{a}_{b}"
                test_graph.add_node(latent, 2)
                test_graph.add_edge(latent, a)
                test_graph.add_edge(latent, b)
            if not test_graph.is_d_separated(treatment, m, set()):
                return False

        # 3. All backdoor paths from M to Y are blocked by X
        for m in m_set:
            # In graph with edges out of M removed, M and Y d-separated by {X}
            test_graph2 = CausalGraph()
            for n in graph.nodes:
                test_graph2.add_node(n, graph.cardinalities[n])
            for p, c in graph.edges:
                if p not in m_set:
                    test_graph2.add_edge(p, c)
            for a, b in graph._bidirected:
                latent = f"__L_{a}_{b}"
                test_graph2.add_node(latent, 2)
                test_graph2.add_edge(latent, a)
                test_graph2.add_edge(latent, b)
            if not test_graph2.is_d_separated(m, outcome, {treatment}):
                return False

        return True

    @staticmethod
    def adjust(graph, treatment, outcome, m_set, treatment_value):
        """Compute P(Y | do(X=x)) via frontdoor adjustment.

        P(Y=y | do(X=x)) = sum_m P(M=m|X=x) sum_x' P(Y=y|X=x',M=m) P(X=x')
        """
        bn = graph.to_bayesian_network()
        card_y = graph.cardinalities[outcome]
        card_x = graph.cardinalities[treatment]
        result = np.zeros(card_y)

        m_list = sorted(m_set)
        m_cards = [graph.cardinalities[m] for m in m_list]

        for m_vals in iter_product(*[range(c) for c in m_cards]):
            m_assign = dict(zip(m_list, m_vals))

            # P(M=m | X=x)
            evidence_xm = {treatment: treatment_value}
            evidence_xm.update(m_assign)
            # Get P(M | X=x) for each m variable
            pm_given_x = 1.0
            for i, m_var in enumerate(m_list):
                f = bn.variable_elimination(m_var, {treatment: treatment_value})
                f.normalize()
                pm_given_x *= float(f.values[m_vals[i]])

            # sum_x' P(Y|X=x', M=m) P(X=x')
            inner_sum = np.zeros(card_y)
            for x_prime in range(card_x):
                evidence_inner = {treatment: x_prime}
                evidence_inner.update(m_assign)
                py = bn.variable_elimination(outcome, evidence_inner)
                py.normalize()

                px = bn.variable_elimination(treatment, {})
                px.normalize()
                px_val = float(px.values[x_prime])

                for y_val in range(card_y):
                    inner_sum[y_val] += float(py.values[y_val]) * px_val

            result += pm_given_x * inner_sum

        total = result.sum()
        if total > 0:
            result /= total
        return Factor([outcome], {outcome: card_y}, result)


# ---------------------------------------------------------------------------
# DoCalculus -- The Three Rules
# ---------------------------------------------------------------------------

class DoCalculus:
    """Implements the three rules of do-calculus for causal effect identification.

    Rule 1 (Insertion/deletion of observations):
        P(y | do(x), z, w) = P(y | do(x), w) if (Y _||_ Z | X, W) in G_bar(X)

    Rule 2 (Action/observation exchange):
        P(y | do(x), do(z), w) = P(y | do(x), z, w) if (Y _||_ Z | X, W) in G_bar(X)_under(Z)

    Rule 3 (Insertion/deletion of actions):
        P(y | do(x), do(z), w) = P(y | do(x), w) if (Y _||_ Z | X, W) in G_bar(X)_bar(Z(S))
        where Z(S) = Z - An(W) in G_bar(X)
    """

    @staticmethod
    def rule1_applicable(graph, y, z, x_do, w=None):
        """Check if Rule 1 applies: can we ignore observation Z?
        Test: (Y _||_ Z | X, W) in G_overline{X} (graph with incoming edges to X removed).
        """
        if w is None:
            w = set()
        x_do = _to_set(x_do)
        y = _to_set(y)
        z = _to_set(z)
        w = _to_set(w)

        # G_overline{X}: remove incoming edges to X (same as mutilate)
        g_overline_x = _remove_incoming(graph, x_do)
        conditioning = x_do | w
        return g_overline_x.is_d_separated(y, z, conditioning)

    @staticmethod
    def rule2_applicable(graph, y, z, x_do, w=None):
        """Check if Rule 2 applies: can we replace do(Z) with observation Z?
        Test: (Y _||_ Z | X, W) in G_overline{X}, underline{Z}.
        """
        if w is None:
            w = set()
        x_do = _to_set(x_do)
        y = _to_set(y)
        z = _to_set(z)
        w = _to_set(w)

        # G_overline{X}, underline{Z}: remove incoming to X AND outgoing from Z
        g = _remove_incoming(graph, x_do)
        g = _remove_outgoing(g, z)
        conditioning = x_do | w
        return g.is_d_separated(y, z, conditioning)

    @staticmethod
    def rule3_applicable(graph, y, z, x_do, w=None):
        """Check if Rule 3 applies: can we remove do(Z)?
        Test: (Y _||_ Z | X, W) in G_bar(X)_bar(Z(S))
        where Z(S) = Z - An(W) in G_bar(X).
        """
        if w is None:
            w = set()
        x_do = _to_set(x_do)
        y = _to_set(y)
        z = _to_set(z)
        w = _to_set(w)

        # Compute Z(S) = Z \ ancestors of W in G_overline{X}
        g_overline_x = _remove_incoming(graph, x_do)
        anc_w = set()
        for wn in w:
            anc_w |= g_overline_x.ancestors(wn)
        z_s = z - anc_w

        # G_overline{X}, overline{Z(S)}
        g = _remove_incoming(graph, x_do | z_s)
        conditioning = x_do | w
        return g.is_d_separated(y, z, conditioning)

    @staticmethod
    def is_identifiable(graph, y, x_do):
        """Check if P(y | do(x)) is identifiable from observational data.

        Uses the ID algorithm (Tian & Pearl, 2002) simplified version:
        - If no backdoor path exists, it's trivially identifiable
        - If a valid adjustment set exists, it's identifiable
        - If frontdoor criterion applies, it's identifiable
        """
        y = _to_set(y)
        x_do = _to_set(x_do)

        # Check if d-separated in mutilated graph (no confounding)
        g_mut = _remove_incoming(graph, x_do)
        if g_mut.is_d_separated(x_do, y, set()):
            return True

        # Check backdoor criterion
        for y_var in y:
            for x_var in x_do:
                adj = BackdoorCriterion.find_adjustment_set(graph, x_var, y_var)
                if adj is not None:
                    return True

        # Check frontdoor criterion
        for y_var in y:
            for x_var in x_do:
                m = FrontdoorCriterion.find_mediator_set(graph, x_var, y_var)
                if m is not None:
                    return True

        return False


# ---------------------------------------------------------------------------
# CounterfactualEngine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """Computes counterfactual queries using the twin-network method.

    Three steps:
    1. Abduction: Update P(U | evidence) given factual observations
    2. Action: Modify the model with the counterfactual intervention
    3. Prediction: Compute the query in the modified model
    """

    def __init__(self, graph):
        self.graph = graph

    def query(self, outcome, intervention, evidence):
        """Compute P(Y_x | evidence) -- the counterfactual probability.

        outcome: variable to query
        intervention: dict of do() assignments (counterfactual world)
        evidence: dict of observed values (factual world)

        Returns: Factor over outcome variable
        """
        # Step 1: Abduction -- compute posterior over exogenous variables
        # In a discrete BN, this means computing P(parents | evidence)
        bn = self.graph.to_bayesian_network()

        # Step 2: Action -- mutilate graph with intervention
        mutilated = self.graph.mutilate(intervention)
        bn_mut = mutilated.to_bayesian_network()

        # Step 3: Prediction -- compute outcome in mutilated model
        # but using the posterior from step 1 as prior

        # For discrete BNs, we use the twin-network approach:
        # Create a combined model with factual and counterfactual copies
        # sharing exogenous (root) variables

        card_y = self.graph.cardinalities[outcome]
        result = np.zeros(card_y)

        # Identify root nodes (no parents) -- these are the "exogenous" variables
        roots = [n for n in self.graph.nodes if not self.graph.get_parents(n)]
        non_roots = [n for n in self.graph.nodes if self.graph.get_parents(n)]
        intervened_vars = set(intervention.keys())

        # Enumerate all possible root configurations
        root_cards = [self.graph.cardinalities[r] for r in roots]

        for root_vals in iter_product(*[range(c) for c in root_cards]):
            root_assign = dict(zip(roots, root_vals))

            # P(roots | evidence): abduction step
            # Compute P(roots) and check consistency with evidence
            p_roots = 1.0
            for r in roots:
                if r in self.graph.cpds:
                    cpd = self.graph.cpds[r]
                    p_roots *= float(cpd.values[root_assign[r]])
                else:
                    p_roots *= 1.0 / self.graph.cardinalities[r]

            # Forward simulate factual world to check evidence consistency
            factual_vals = dict(root_assign)
            consistent = True
            topo = self.graph.topological_sort()
            for node in topo:
                if node in roots:
                    continue
                parents = self.graph.get_parents(node)
                if node in self.graph.cpds:
                    cpd = self.graph.cpds[node]
                    parent_assign = {p: factual_vals[p] for p in parents}
                    reduced = cpd.reduce(parent_assign)
                    reduced.normalize()
                    # If this node is in evidence, check/weight by its probability
                    if node in evidence:
                        p_roots *= float(reduced.values[evidence[node]])
                        factual_vals[node] = evidence[node]
                    else:
                        # Marginalize (sum over possible values)
                        # For counterfactual, we need a specific value
                        # Use most likely for non-evidence non-intervened
                        factual_vals[node] = int(np.argmax(reduced.values))
                else:
                    factual_vals[node] = 0

            if p_roots < 1e-15:
                continue

            # Forward simulate counterfactual world with same exogenous
            cf_vals = dict(root_assign)
            # Apply interventions
            cf_vals.update(intervention)
            topo_mut = mutilated.topological_sort()
            for node in topo_mut:
                if node in roots or node in intervened_vars:
                    continue
                parents_mut = mutilated.get_parents(node)
                if node in mutilated.cpds:
                    cpd = mutilated.cpds[node]
                    parent_assign = {p: cf_vals[p] for p in parents_mut}
                    reduced = cpd.reduce(parent_assign)
                    reduced.normalize()
                    cf_vals[node] = int(np.argmax(reduced.values))
                else:
                    cf_vals[node] = 0

            if outcome in cf_vals:
                result[cf_vals[outcome]] += p_roots

        total = result.sum()
        if total > 0:
            result /= total
        return Factor([outcome], {outcome: card_y}, result)

    def expected_counterfactual(self, outcome, intervention, evidence):
        """Compute E[Y_x | evidence] for binary outcome."""
        factor = self.query(outcome, intervention, evidence)
        # E[Y] = sum_y y * P(Y=y)
        card = self.graph.cardinalities[outcome]
        return sum(i * float(factor.values[i]) for i in range(card))

    def probability_of_necessity(self, treatment, outcome, treatment_val=1, outcome_val=1):
        """P(Y_{x=0}=0 | X=1, Y=1) -- probability that X was necessary for Y.

        PN = P(Y_0 = 0 | X=1, Y=1)
        """
        no_treatment = 1 - treatment_val
        no_outcome = 1 - outcome_val
        evidence = {treatment: treatment_val, outcome: outcome_val}
        intervention = {treatment: no_treatment}
        factor = self.query(outcome, intervention, evidence)
        return float(factor.values[no_outcome])

    def probability_of_sufficiency(self, treatment, outcome, treatment_val=1, outcome_val=1):
        """P(Y_{x=1}=1 | X=0, Y=0) -- probability that X would have been sufficient.

        PS = P(Y_1 = 1 | X=0, Y=0)
        """
        no_treatment = 1 - treatment_val
        no_outcome = 1 - outcome_val
        evidence = {treatment: no_treatment, outcome: no_outcome}
        intervention = {treatment: treatment_val}
        factor = self.query(outcome, intervention, evidence)
        return float(factor.values[outcome_val])


# ---------------------------------------------------------------------------
# InstrumentalVariable
# ---------------------------------------------------------------------------

class InstrumentalVariable:
    """Instrumental variable estimation for causal effects.

    An instrument Z for the effect of X on Y must satisfy:
    1. Z is associated with X (relevance)
    2. Z affects Y only through X (exclusion)
    3. Z and Y share no common causes (independence)
    """

    @staticmethod
    def is_valid_instrument(graph, instrument, treatment, outcome):
        """Check if Z is a valid instrument for X -> Y."""
        z, x, y = instrument, treatment, outcome

        # Instrument must be different from treatment
        if z == x:
            return False

        # 1. Z must be associated with X (Z is ancestor of X or connected to X)
        if not graph.has_directed_path(z, x):
            # Check if Z influences X at all
            desc_z = graph.descendants(z)
            if x not in desc_z:
                return False

        # 2. Exclusion: no direct path from Z to Y except through X
        # In the graph with X removed, Z and Y should be d-separated
        test_graph = CausalGraph()
        for n in graph.nodes:
            if n != x:
                test_graph.add_node(n, graph.cardinalities[n])
        for p, c in graph.edges:
            if p != x and c != x:
                test_graph.add_edge(p, c)
        for a, b in graph._bidirected:
            if a != x and b != x:
                test_graph.add_bidirected(a, b)
                latent = f"__L_{a}_{b}"
                test_graph.add_node(latent, 2)
                test_graph.add_edge(latent, a)
                test_graph.add_edge(latent, b)

        if z not in test_graph.nodes or y not in test_graph.nodes:
            return True  # Trivially, if removing X disconnects Z from graph

        if not test_graph.is_d_separated(z, y, set()):
            return False

        # 3. Independence: no bidirected edge (latent confounder) between Z and Y
        for a, b in graph._bidirected:
            if (a == z and b == y) or (a == y and b == z):
                return False
            # Also check if Z shares a latent cause with Y through any path
            # (simplified: just check direct bidirected)

        return True

    @staticmethod
    def wald_estimator(data, instrument, treatment, outcome):
        """Compute the Wald estimator: E[Y|Z=1]-E[Y|Z=0] / E[X|Z=1]-E[X|Z=0].

        data: list of dicts with variable assignments
        """
        z1_x, z1_y, z0_x, z0_y = [], [], [], []
        for d in data:
            if d[instrument] == 1:
                z1_x.append(d[treatment])
                z1_y.append(d[outcome])
            else:
                z0_x.append(d[treatment])
                z0_y.append(d[outcome])

        if not z1_x or not z0_x:
            return None

        ey_z1 = np.mean(z1_y)
        ey_z0 = np.mean(z0_y)
        ex_z1 = np.mean(z1_x)
        ex_z0 = np.mean(z0_x)

        denom = ex_z1 - ex_z0
        if abs(denom) < 1e-10:
            return None  # Weak instrument

        return (ey_z1 - ey_z0) / denom


# ---------------------------------------------------------------------------
# CausalDiscovery
# ---------------------------------------------------------------------------

class CausalDiscovery:
    """Constraint-based causal discovery algorithms."""

    @staticmethod
    def pc_algorithm(data, cardinalities, alpha=0.05, max_cond_size=3):
        """PC algorithm for causal structure learning.

        1. Start with complete undirected graph
        2. Remove edges using conditional independence tests
        3. Orient v-structures
        4. Apply orientation rules (Meek's rules)

        Returns: dict with 'directed', 'undirected', 'adjacency', 'sepsets'
        """
        variables = sorted(cardinalities.keys())
        n = len(variables)

        # Start with complete undirected graph
        adjacency = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                adjacency[variables[i]].add(variables[j])
                adjacency[variables[j]].add(variables[i])

        sepsets = {}

        # Phase 1: Edge removal via CI tests
        for cond_size in range(max_cond_size + 1):
            edges_to_check = []
            for x in variables:
                for y in list(adjacency[x]):
                    if x < y:
                        edges_to_check.append((x, y))

            for x, y in edges_to_check:
                if y not in adjacency[x]:
                    continue

                # Neighbors of X excluding Y (potential conditioning sets)
                neighbors = adjacency[x] - {y}
                if len(neighbors) < cond_size:
                    continue

                found_independent = False
                for z_set in _combinations(sorted(neighbors), cond_size):
                    z_set = set(z_set)
                    is_indep, _ = PGMUtils.independence_test(
                        data, x, y, z_set, cardinalities, alpha
                    )
                    if is_indep:
                        adjacency[x].discard(y)
                        adjacency[y].discard(x)
                        sepsets[(x, y)] = z_set
                        sepsets[(y, x)] = z_set
                        found_independent = True
                        break

        # Phase 2: Orient v-structures
        directed = set()
        undirected = set()

        for x in variables:
            for y in adjacency[x]:
                undirected.add((min(x, y), max(x, y)))

        for y in variables:
            neighbors = sorted(adjacency[y])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    x, z = neighbors[i], neighbors[j]
                    # If x and z are not adjacent and y not in sepset(x,z)
                    if z not in adjacency[x]:
                        sep = sepsets.get((x, z), sepsets.get((z, x), set()))
                        if y not in sep:
                            # Orient as x -> y <- z (v-structure)
                            directed.add((x, y))
                            directed.add((z, y))
                            undirected.discard((min(x, y), max(x, y)))
                            undirected.discard((min(y, z), max(y, z)))

        # Phase 3: Meek's rules (simplified)
        changed = True
        while changed:
            changed = False
            for a, b in list(undirected):
                # Rule 1: If c -> a - b and c not adjacent to b, orient a -> b
                for c in variables:
                    if (c, a) in directed and c not in adjacency[b]:
                        directed.add((a, b))
                        undirected.discard((a, b))
                        undirected.discard((b, a))
                        changed = True
                        break
                    if (c, b) in directed and c not in adjacency[a]:
                        directed.add((b, a))
                        undirected.discard((a, b))
                        undirected.discard((b, a))
                        changed = True
                        break

        return {
            'directed': directed,
            'undirected': undirected,
            'adjacency': dict(adjacency),
            'sepsets': sepsets
        }

    @staticmethod
    def find_latent_confounders(graph, data, cardinalities, alpha=0.05):
        """Detect potential latent confounders via residual dependencies.

        If X and Y are not adjacent in the learned structure but show
        dependence conditioned on their common effects, suspect a latent confounder.
        """
        result = graph.to_bayesian_network()
        confounders = []

        for i, x in enumerate(graph.nodes):
            for y in graph.nodes[i+1:]:
                if y in graph._children[x] or x in graph._children[y]:
                    continue  # Already connected
                # Check if they have common children
                children_x = set(graph.get_children(x))
                children_y = set(graph.get_children(y))
                common = children_x & children_y
                if common:
                    # Test if X and Y are dependent given common children
                    is_indep, stat = PGMUtils.independence_test(
                        data, x, y, common, cardinalities, alpha
                    )
                    if not is_indep:
                        confounders.append((x, y, stat))

        return confounders


# ---------------------------------------------------------------------------
# MediationAnalysis
# ---------------------------------------------------------------------------

class MediationAnalysis:
    """Causal mediation analysis: decompose total effect into direct and indirect."""

    @staticmethod
    def total_effect(graph, treatment, outcome, treatment_val=1, control_val=0):
        """TE = E[Y | do(X=1)] - E[Y | do(X=0)]."""
        f1 = graph.interventional_query(outcome, {treatment: treatment_val})
        f0 = graph.interventional_query(outcome, {treatment: control_val})
        # E[Y] for binary outcome
        card = graph.cardinalities[outcome]
        e1 = sum(i * float(f1.values[i]) for i in range(card))
        e0 = sum(i * float(f0.values[i]) for i in range(card))
        return e1 - e0

    @staticmethod
    def controlled_direct_effect(graph, treatment, outcome, mediator,
                                  treatment_val=1, control_val=0, mediator_val=0):
        """CDE(m) = E[Y | do(X=1, M=m)] - E[Y | do(X=0, M=m)]."""
        f1 = graph.interventional_query(outcome, {treatment: treatment_val, mediator: mediator_val})
        f0 = graph.interventional_query(outcome, {treatment: control_val, mediator: mediator_val})
        card = graph.cardinalities[outcome]
        e1 = sum(i * float(f1.values[i]) for i in range(card))
        e0 = sum(i * float(f0.values[i]) for i in range(card))
        return e1 - e0

    @staticmethod
    def natural_direct_effect(graph, treatment, outcome, mediator,
                               treatment_val=1, control_val=0):
        """NDE = E[Y_{x=1, M_{x=0}}] - E[Y_{x=0, M_{x=0}}].

        Uses counterfactual computation via twin network.
        """
        engine = CounterfactualEngine(graph)

        # We need E[Y] under do(X=1) with M at its "natural" level under X=0
        # This requires a nested counterfactual which we approximate:
        # NDE ~ E[Y | do(X=1, M=m0)] averaged over m0 from P(M | do(X=0))

        # Get distribution of M under do(X=0)
        fm0 = graph.interventional_query(mediator, {treatment: control_val})
        fm0.normalize()

        card_y = graph.cardinalities[outcome]
        card_m = graph.cardinalities[mediator]
        nde_result = np.zeros(card_y)

        for m_val in range(card_m):
            pm = float(fm0.values[m_val])
            f1 = graph.interventional_query(outcome, {treatment: treatment_val, mediator: m_val})
            f0 = graph.interventional_query(outcome, {treatment: control_val, mediator: m_val})
            for y_val in range(card_y):
                nde_result[y_val] += pm * (float(f1.values[y_val]) - float(f0.values[y_val]))

        return sum(i * nde_result[i] for i in range(card_y))

    @staticmethod
    def natural_indirect_effect(graph, treatment, outcome, mediator,
                                 treatment_val=1, control_val=0):
        """NIE = TE - NDE."""
        te = MediationAnalysis.total_effect(graph, treatment, outcome, treatment_val, control_val)
        nde = MediationAnalysis.natural_direct_effect(graph, treatment, outcome, mediator,
                                                       treatment_val, control_val)
        return te - nde

    @staticmethod
    def decompose(graph, treatment, outcome, mediator,
                   treatment_val=1, control_val=0):
        """Full decomposition: TE = NDE + NIE."""
        te = MediationAnalysis.total_effect(graph, treatment, outcome, treatment_val, control_val)
        nde = MediationAnalysis.natural_direct_effect(graph, treatment, outcome, mediator,
                                                       treatment_val, control_val)
        nie = te - nde
        return {
            'total_effect': te,
            'natural_direct_effect': nde,
            'natural_indirect_effect': nie,
            'proportion_mediated': abs(nie) / abs(te) if abs(te) > 1e-10 else 0.0
        }


# ---------------------------------------------------------------------------
# CausalUtils
# ---------------------------------------------------------------------------

class CausalUtils:
    """Utility functions for causal inference."""

    @staticmethod
    def average_treatment_effect(graph, treatment, outcome, treatment_val=1, control_val=0):
        """ATE = E[Y | do(X=1)] - E[Y | do(X=0)]."""
        return MediationAnalysis.total_effect(graph, treatment, outcome, treatment_val, control_val)

    @staticmethod
    def ate_from_data(data, treatment, outcome, adjustment_set=None):
        """Estimate ATE from observational data with optional adjustment.

        If adjustment_set is provided, uses stratified estimation.
        """
        if adjustment_set is None or len(adjustment_set) == 0:
            # Simple difference in means
            treated = [d[outcome] for d in data if d[treatment] == 1]
            control = [d[outcome] for d in data if d[treatment] == 0]
            if not treated or not control:
                return 0.0
            return np.mean(treated) - np.mean(control)

        # Stratified estimation
        z_list = sorted(adjustment_set)

        # Group by adjustment strata
        strata = defaultdict(lambda: {'treated': [], 'control': []})
        for d in data:
            key = tuple(d[z] for z in z_list)
            if d[treatment] == 1:
                strata[key]['treated'].append(d[outcome])
            else:
                strata[key]['control'].append(d[outcome])

        total_weight = 0
        weighted_effect = 0
        for key, groups in strata.items():
            if groups['treated'] and groups['control']:
                n = len(groups['treated']) + len(groups['control'])
                effect = np.mean(groups['treated']) - np.mean(groups['control'])
                weighted_effect += effect * n
                total_weight += n

        if total_weight == 0:
            return 0.0
        return weighted_effect / total_weight

    @staticmethod
    def conditional_ate(data, treatment, outcome, condition_var, condition_val,
                        adjustment_set=None):
        """CATE: ATE conditioned on a subgroup."""
        filtered = [d for d in data if d[condition_var] == condition_val]
        return CausalUtils.ate_from_data(filtered, treatment, outcome, adjustment_set)

    @staticmethod
    def inverse_probability_weighting(data, treatment, outcome, propensity_scores=None):
        """IPW estimator for ATE.

        ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
        where e(X) = P(T=1 | X)
        """
        if propensity_scores is None:
            # Simple estimate: P(T=1)
            p_treat = np.mean([d[treatment] for d in data])
            propensity_scores = [p_treat] * len(data)

        n = len(data)
        sum_treated = 0.0
        sum_control = 0.0

        for i, d in enumerate(data):
            e = propensity_scores[i]
            e = max(0.01, min(0.99, e))  # Clip for stability
            y = d[outcome]
            t = d[treatment]
            sum_treated += y * t / e
            sum_control += y * (1 - t) / (1 - e)

        return sum_treated / n - sum_control / n

    @staticmethod
    def bounds_no_assumptions(data, treatment, outcome):
        """Natural bounds on ATE without any assumptions (Manski bounds).

        Lower: E[Y|X=1] * P(X=1) + 0 * P(X=0) - (E[Y|X=0] * P(X=0) + 1 * P(X=1))
        For binary outcomes.
        """
        treated = [d[outcome] for d in data if d[treatment] == 1]
        control = [d[outcome] for d in data if d[treatment] == 0]
        n = len(data)
        n1 = len(treated)
        n0 = len(control)

        if n1 == 0 or n0 == 0:
            return (-1.0, 1.0)

        p1 = n1 / n
        p0 = n0 / n
        ey1 = np.mean(treated)
        ey0 = np.mean(control)

        # Manski bounds for binary outcome
        lower = (ey1 * p1 + 0 * p0) - (ey0 * p0 + 1 * p1)
        upper = (ey1 * p1 + 1 * p0) - (ey0 * p0 + 0 * p1)

        # Simplified: lower = ey1 - ey0 - min range, upper = ey1 - ey0 + max range
        # Actually the tighter standard bounds:
        lower = ey1 - 1  # Worst case: all controls would have Y=1 if treated
        upper = ey1 - 0  # Best case: all controls would have Y=0 if treated

        # Standard Manski bounds
        lower = max(-1, ey1 - ey0 - 1 + p1)
        upper = min(1, ey1 - ey0 + 1 - p1)

        return (lower, upper)

    @staticmethod
    def sensitivity_analysis(data, treatment, outcome, gamma_range=None):
        """Rosenbaum-style sensitivity analysis.

        For each gamma (odds ratio of unmeasured confounding),
        compute the range of possible ATEs.

        Returns: list of (gamma, lower_ate, upper_ate)
        """
        if gamma_range is None:
            gamma_range = [1.0, 1.5, 2.0, 3.0, 5.0]

        treated = [d[outcome] for d in data if d[treatment] == 1]
        control = [d[outcome] for d in data if d[treatment] == 0]
        naive_ate = np.mean(treated) - np.mean(control) if treated and control else 0.0

        results = []
        for gamma in gamma_range:
            # Bias range: the confounding bias is bounded by log(gamma) * outcome_variance
            if gamma == 1.0:
                results.append((gamma, naive_ate, naive_ate))
            else:
                # Approximate sensitivity: shift ATE by +/- bias factor
                all_y = [d[outcome] for d in data]
                outcome_range = max(all_y) - min(all_y) if all_y else 1.0
                bias = math.log(gamma) * outcome_range * 0.5
                results.append((gamma, naive_ate - bias, naive_ate + bias))

        return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_set(x):
    if isinstance(x, str):
        return {x}
    return set(x)


def _combinations(lst, size):
    """Generate all combinations of given size from lst."""
    if size == 0:
        yield ()
        return
    for i in range(len(lst)):
        for rest in _combinations(lst[i+1:], size - 1):
            yield (lst[i],) + rest


def _remove_incoming(graph, nodes):
    """Return new CausalGraph with incoming edges to nodes removed."""
    nodes = _to_set(nodes)
    new_graph = CausalGraph()
    for n in graph.nodes:
        new_graph.add_node(n, graph.cardinalities[n])
    for p, c in graph.edges:
        if c not in nodes:
            new_graph.add_edge(p, c)
    for n, cpd in graph.cpds.items():
        new_graph.set_cpd(n, cpd)
    for a, b in graph._bidirected:
        new_graph.add_bidirected(a, b)
    return new_graph


def _remove_outgoing(graph, nodes):
    """Return new CausalGraph with outgoing edges from nodes removed."""
    nodes = _to_set(nodes)
    new_graph = CausalGraph()
    for n in graph.nodes:
        new_graph.add_node(n, graph.cardinalities[n])
    for p, c in graph.edges:
        if p not in nodes:
            new_graph.add_edge(p, c)
    for n, cpd in graph.cpds.items():
        new_graph.set_cpd(n, cpd)
    for a, b in graph._bidirected:
        new_graph.add_bidirected(a, b)
    return new_graph
