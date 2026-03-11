"""
C160: Probabilistic Graphical Models
Composing C156 (distributions) concepts with graph structure.

Components:
- Factor: potential function over variables
- BayesianNetwork: DAG + CPTs, variable elimination
- MarkovNetwork: undirected + factors, partition function
- FactorGraph: bipartite, belief propagation (sum-product, max-product)
- JunctionTree: exact inference via clique tree
- StructureLearning: learn BN structure from data (BIC, K2, PC)
- DynamicBayesNet: time-unrolled Bayesian network
- PGMUtils: d-separation, Markov blanket, topological ops
"""

import math
import random
from collections import defaultdict, deque
from itertools import product as iter_product
import numpy as np


# ---------------------------------------------------------------------------
# Factor
# ---------------------------------------------------------------------------

class Factor:
    """A factor (potential function) over a set of discrete variables."""

    def __init__(self, variables, cardinalities, values=None):
        """
        variables: list of variable names
        cardinalities: dict mapping variable -> number of states
        values: flat numpy array of factor values (row-major over variables)
        """
        self.variables = list(variables)
        self.cardinalities = {v: cardinalities[v] for v in self.variables}
        shape = tuple(self.cardinalities[v] for v in self.variables)
        if values is not None:
            self.values = np.array(values, dtype=float).reshape(shape)
        else:
            self.values = np.ones(shape, dtype=float)

    def get_value(self, assignment):
        """Get factor value for a given assignment dict."""
        idx = tuple(assignment[v] for v in self.variables)
        return float(self.values[idx])

    def set_value(self, assignment, value):
        """Set factor value for a given assignment dict."""
        idx = tuple(assignment[v] for v in self.variables)
        self.values[idx] = value

    def multiply(self, other):
        """Multiply two factors, returning a new factor."""
        new_vars = list(self.variables)
        for v in other.variables:
            if v not in new_vars:
                new_vars.append(v)

        new_card = {}
        new_card.update(self.cardinalities)
        new_card.update(other.cardinalities)

        new_factor = Factor(new_vars, new_card)
        # Iterate over all assignments
        ranges = [range(new_card[v]) for v in new_vars]
        for assignment_vals in iter_product(*ranges):
            assignment = dict(zip(new_vars, assignment_vals))
            val = self.get_value(assignment) * other.get_value(assignment)
            new_factor.set_value(assignment, val)
        return new_factor

    def marginalize(self, variable):
        """Sum out a variable, returning a new factor."""
        if variable not in self.variables:
            return self  # nothing to do
        new_vars = [v for v in self.variables if v != variable]
        new_card = {v: self.cardinalities[v] for v in new_vars}
        new_factor = Factor(new_vars, new_card)

        if not new_vars:
            # Summing out the last variable
            new_factor.values = np.array(float(np.sum(self.values))).reshape(())
            return new_factor

        # Sum along the axis of the variable to remove
        axis = self.variables.index(variable)
        new_factor.values = np.sum(self.values, axis=axis)
        return new_factor

    def reduce(self, evidence):
        """Reduce factor by fixing variables to observed values."""
        if not evidence:
            return self
        remaining_evidence = {v: val for v, val in evidence.items() if v in self.variables}
        if not remaining_evidence:
            return self

        new_vars = [v for v in self.variables if v not in remaining_evidence]
        new_card = {v: self.cardinalities[v] for v in new_vars}
        new_factor = Factor(new_vars, new_card)

        if not new_vars:
            # All variables are evidence
            idx = tuple(remaining_evidence[v] for v in self.variables)
            new_factor.values = np.array(float(self.values[idx])).reshape(())
            return new_factor

        ranges = [range(new_card[v]) for v in new_vars]
        for assignment_vals in iter_product(*ranges):
            assignment = dict(zip(new_vars, assignment_vals))
            full_assignment = dict(assignment)
            full_assignment.update(remaining_evidence)
            val = self.get_value(full_assignment)
            new_factor.set_value(assignment, val)
        return new_factor

    def normalize(self):
        """Normalize factor values to sum to 1."""
        total = np.sum(self.values)
        if total > 0:
            self.values = self.values / total
        return self

    def __repr__(self):
        return f"Factor({self.variables})"


# ---------------------------------------------------------------------------
# BayesianNetwork
# ---------------------------------------------------------------------------

class BayesianNetwork:
    """Directed acyclic graphical model with conditional probability tables."""

    def __init__(self):
        self.nodes = []
        self.edges = []  # (parent, child)
        self.cpds = {}   # node -> Factor (CPT)
        self._parents = defaultdict(list)
        self._children = defaultdict(list)
        self.cardinalities = {}

    def add_node(self, name, cardinality=2):
        """Add a node with given cardinality."""
        if name not in self.nodes:
            self.nodes.append(name)
            self.cardinalities[name] = cardinality

    def add_edge(self, parent, child):
        """Add a directed edge from parent to child."""
        self.edges.append((parent, child))
        self._parents[child].append(parent)
        self._children[parent].append(child)

    def set_cpd(self, node, cpd_factor):
        """Set the conditional probability distribution for a node."""
        self.cpds[node] = cpd_factor

    def get_parents(self, node):
        """Get parents of a node."""
        return self._parents[node]

    def get_children(self, node):
        """Get children of a node."""
        return self._children[node]

    def topological_sort(self):
        """Return nodes in topological order."""
        in_degree = defaultdict(int)
        for node in self.nodes:
            in_degree[node] = 0
        for p, c in self.edges:
            in_degree[c] += 1

        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for child in self._children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result

    def sample(self, n_samples=1, evidence=None):
        """Forward sampling (with rejection for evidence)."""
        order = self.topological_sort()
        samples = []
        max_attempts = n_samples * 1000

        for _ in range(max_attempts):
            if len(samples) >= n_samples:
                break
            sample = {}
            for node in order:
                parents = self.get_parents(node)
                cpd = self.cpds[node]
                # Build assignment for parents
                parent_assignment = {p: sample[p] for p in parents}
                # Get conditional distribution
                reduced = cpd.reduce(parent_assignment)
                probs = reduced.values.flatten()
                total = np.sum(probs)
                if total > 0:
                    probs = probs / total
                val = np.random.choice(len(probs), p=probs)
                sample[node] = int(val)

            # Check evidence
            if evidence:
                if all(sample.get(v) == val for v, val in evidence.items()):
                    samples.append(sample)
            else:
                samples.append(sample)

        return samples

    def variable_elimination(self, query, evidence=None):
        """Exact inference via variable elimination."""
        if evidence is None:
            evidence = {}

        # Collect all factors
        factors = []
        for node in self.nodes:
            f = self.cpds[node]
            if evidence:
                f = f.reduce(evidence)
            factors.append(f)

        # Determine elimination order (simple: all non-query, non-evidence)
        to_eliminate = [n for n in self.nodes
                        if n not in query and n not in evidence]

        # Eliminate variables one by one
        for var in to_eliminate:
            # Find factors containing this variable
            relevant = [f for f in factors if var in f.variables]
            remaining = [f for f in factors if var not in f.variables]

            if not relevant:
                continue

            # Multiply all relevant factors
            product = relevant[0]
            for f in relevant[1:]:
                product = product.multiply(f)

            # Sum out the variable
            new_factor = product.marginalize(var)
            remaining.append(new_factor)
            factors = remaining

        # Multiply remaining factors
        if not factors:
            return Factor(query, {q: self.cardinalities[q] for q in query})

        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)

        result.normalize()
        return result

    def markov_blanket(self, node):
        """Get the Markov blanket of a node (parents + children + co-parents)."""
        blanket = set()
        blanket.update(self._parents[node])
        blanket.update(self._children[node])
        for child in self._children[node]:
            blanket.update(self._parents[child])
        blanket.discard(node)
        return blanket

    def is_d_separated(self, x, y, z):
        """Check if x and y are d-separated given z using Bayes-Ball algorithm."""
        # x, y are sets of nodes, z is the conditioning set
        x_set = set(x) if isinstance(x, (list, set)) else {x}
        y_set = set(y) if isinstance(y, (list, set)) else {y}
        z_set = set(z) if isinstance(z, (list, set)) else {z}

        # Bayes-Ball: find reachable nodes from x given z
        reachable = self._bayes_ball_reachable(x_set, z_set)
        return len(reachable & y_set) == 0

    def _bayes_ball_reachable(self, source, observed):
        """Bayes-Ball algorithm to find reachable nodes."""
        # Track visited (node, direction) pairs
        visited = set()
        reachable = set()
        # Queue: (node, from_child) -- from_child=True means ball came from child
        queue = deque()
        for s in source:
            queue.append((s, True))   # as if from child
            queue.append((s, False))  # as if from parent

        while queue:
            node, from_child = queue.popleft()
            if (node, from_child) in visited:
                continue
            visited.add((node, from_child))

            if node not in observed:
                reachable.add(node)

            if from_child and node not in observed:
                # Pass to parents
                for parent in self._parents[node]:
                    if (parent, True) not in visited:
                        queue.append((parent, True))
                # Pass to children
                for child in self._children[node]:
                    if (child, False) not in visited:
                        queue.append((child, False))
            elif from_child and node in observed:
                # Blocked through non-collider, but pass to parents (v-structure)
                for parent in self._parents[node]:
                    if (parent, True) not in visited:
                        queue.append((parent, True))
            elif not from_child and node not in observed:
                # Pass to children only
                for child in self._children[node]:
                    if (child, False) not in visited:
                        queue.append((child, False))
            elif not from_child and node in observed:
                # V-structure activation: observed collider passes to parents
                for parent in self._parents[node]:
                    if (parent, True) not in visited:
                        queue.append((parent, True))

        return reachable - source

    def to_markov_network(self):
        """Moralize to create an equivalent MarkovNetwork."""
        mn = MarkovNetwork()
        for node in self.nodes:
            mn.add_node(node, self.cardinalities[node])

        # Add edges: original edges (undirected) + marry parents
        added = set()
        for p, c in self.edges:
            key = tuple(sorted([p, c]))
            if key not in added:
                mn.add_edge(p, c)
                added.add(key)

        # Marry parents (moralization)
        for node in self.nodes:
            parents = self._parents[node]
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    key = tuple(sorted([parents[i], parents[j]]))
                    if key not in added:
                        mn.add_edge(parents[i], parents[j])
                        added.add(key)

        # Convert CPDs to factors
        for node in self.nodes:
            mn.add_factor(self.cpds[node])

        return mn

    def log_likelihood(self, data):
        """Compute log-likelihood of data given the network."""
        ll = 0.0
        for sample in data:
            for node in self.nodes:
                cpd = self.cpds[node]
                val = cpd.get_value(sample)
                if val > 0:
                    ll += math.log(val)
                else:
                    ll += float('-inf')
        return ll


# ---------------------------------------------------------------------------
# MarkovNetwork
# ---------------------------------------------------------------------------

class MarkovNetwork:
    """Undirected graphical model with potential functions (factors)."""

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.factors = []
        self.cardinalities = {}
        self._neighbors = defaultdict(set)

    def add_node(self, name, cardinality=2):
        if name not in self.nodes:
            self.nodes.append(name)
            self.cardinalities[name] = cardinality

    def add_edge(self, a, b):
        self.edges.append((a, b))
        self._neighbors[a].add(b)
        self._neighbors[b].add(a)

    def add_factor(self, factor):
        self.factors.append(factor)

    def get_neighbors(self, node):
        return self._neighbors[node]

    def partition_function(self, evidence=None):
        """Compute the partition function Z by brute force."""
        if evidence is None:
            evidence = {}

        free_vars = [n for n in self.nodes if n not in evidence]
        ranges = [range(self.cardinalities[v]) for v in free_vars]

        Z = 0.0
        for vals in iter_product(*ranges):
            assignment = dict(zip(free_vars, vals))
            assignment.update(evidence)
            val = 1.0
            for factor in self.factors:
                val *= factor.get_value(assignment)
            Z += val
        return Z

    def query(self, query_vars, evidence=None):
        """Exact inference by enumeration."""
        if evidence is None:
            evidence = {}

        free_vars = [n for n in self.nodes if n not in evidence]
        query_card = {v: self.cardinalities[v] for v in query_vars}
        result = Factor(query_vars, query_card)

        non_query_free = [v for v in free_vars if v not in query_vars]
        ranges_nq = [range(self.cardinalities[v]) for v in non_query_free]
        ranges_q = [range(self.cardinalities[v]) for v in query_vars]

        for q_vals in iter_product(*ranges_q):
            q_assignment = dict(zip(query_vars, q_vals))
            total = 0.0
            for nq_vals in iter_product(*ranges_nq):
                assignment = dict(zip(non_query_free, nq_vals))
                assignment.update(q_assignment)
                assignment.update(evidence)
                val = 1.0
                for factor in self.factors:
                    val *= factor.get_value(assignment)
                total += val
            result.set_value(q_assignment, total)

        result.normalize()
        return result

    def gibbs_sample(self, n_samples=100, burn_in=50, evidence=None):
        """Gibbs sampling for approximate inference."""
        if evidence is None:
            evidence = {}

        # Initialize randomly
        state = {}
        for node in self.nodes:
            if node in evidence:
                state[node] = evidence[node]
            else:
                state[node] = random.randint(0, self.cardinalities[node] - 1)

        free_vars = [n for n in self.nodes if n not in evidence]
        samples = []

        for i in range(n_samples + burn_in):
            for var in free_vars:
                # Sample var conditioned on everything else
                probs = []
                for val in range(self.cardinalities[var]):
                    state[var] = val
                    p = 1.0
                    for factor in self.factors:
                        if var in factor.variables:
                            p *= factor.get_value(state)
                    probs.append(p)
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                else:
                    probs = [1.0 / len(probs)] * len(probs)
                state[var] = random.choices(range(self.cardinalities[var]),
                                            weights=probs)[0]
            if i >= burn_in:
                samples.append(dict(state))

        return samples


# ---------------------------------------------------------------------------
# FactorGraph
# ---------------------------------------------------------------------------

class FactorGraph:
    """Bipartite factor graph for belief propagation."""

    def __init__(self):
        self.variables = []
        self.factors = []
        self.cardinalities = {}
        self._var_to_factors = defaultdict(list)
        self._factor_to_vars = {}

    def add_variable(self, name, cardinality=2):
        if name not in self.variables:
            self.variables.append(name)
            self.cardinalities[name] = cardinality

    def add_factor(self, factor, name=None):
        if name is None:
            name = f"f{len(self.factors)}"
        self.factors.append((name, factor))
        self._factor_to_vars[name] = factor.variables
        for v in factor.variables:
            self._var_to_factors[v].append(name)

    def belief_propagation(self, max_iter=50, tol=1e-6):
        """Loopy belief propagation (sum-product algorithm)."""
        # Initialize messages
        # var_to_factor[var][factor_name] = message (array of cardinality)
        # factor_to_var[factor_name][var] = message (array of cardinality)
        var_to_factor = {}
        factor_to_var = {}

        for var in self.variables:
            var_to_factor[var] = {}
            card = self.cardinalities[var]
            for fname in self._var_to_factors[var]:
                var_to_factor[var][fname] = np.ones(card) / card

        for fname, factor in self.factors:
            factor_to_var[fname] = {}
            for var in factor.variables:
                card = self.cardinalities[var]
                factor_to_var[fname][var] = np.ones(card) / card

        for iteration in range(max_iter):
            old_f2v = {fn: {v: msg.copy() for v, msg in msgs.items()}
                       for fn, msgs in factor_to_var.items()}

            # Update factor -> variable messages
            for fname, factor in self.factors:
                fvars = factor.variables
                for target_var in fvars:
                    other_vars = [v for v in fvars if v != target_var]
                    # Compute message by summing over other variables
                    target_card = self.cardinalities[target_var]
                    msg = np.zeros(target_card)

                    if not other_vars:
                        # Unary factor
                        for val in range(target_card):
                            assignment = {target_var: val}
                            msg[val] = factor.get_value(assignment)
                    else:
                        other_ranges = [range(self.cardinalities[v]) for v in other_vars]
                        for target_val in range(target_card):
                            s = 0.0
                            for other_vals in iter_product(*other_ranges):
                                assignment = {target_var: target_val}
                                assignment.update(dict(zip(other_vars, other_vals)))
                                fval = factor.get_value(assignment)
                                # Multiply by incoming messages from other vars
                                prod = fval
                                for ov, oval in zip(other_vars, other_vals):
                                    prod *= var_to_factor[ov][fname][oval]
                                s += prod
                            msg[target_val] = s

                    total = np.sum(msg)
                    if total > 0:
                        msg /= total
                    factor_to_var[fname][target_var] = msg

            # Update variable -> factor messages
            for var in self.variables:
                for target_fname in self._var_to_factors[var]:
                    card = self.cardinalities[var]
                    msg = np.ones(card)
                    for fname in self._var_to_factors[var]:
                        if fname != target_fname:
                            msg *= factor_to_var[fname][var]
                    total = np.sum(msg)
                    if total > 0:
                        msg /= total
                    var_to_factor[var][target_fname] = msg

            # Check convergence
            max_diff = 0.0
            for fname in factor_to_var:
                for var in factor_to_var[fname]:
                    diff = np.max(np.abs(factor_to_var[fname][var] - old_f2v[fname][var]))
                    max_diff = max(max_diff, diff)

            if max_diff < tol:
                break

        # Compute beliefs
        beliefs = {}
        for var in self.variables:
            card = self.cardinalities[var]
            belief = np.ones(card)
            for fname in self._var_to_factors[var]:
                belief *= factor_to_var[fname][var]
            total = np.sum(belief)
            if total > 0:
                belief /= total
            beliefs[var] = belief

        return beliefs

    def max_product(self, max_iter=50, tol=1e-6):
        """Max-product belief propagation for MAP inference."""
        var_to_factor = {}
        factor_to_var = {}

        for var in self.variables:
            var_to_factor[var] = {}
            card = self.cardinalities[var]
            for fname in self._var_to_factors[var]:
                var_to_factor[var][fname] = np.ones(card) / card

        for fname, factor in self.factors:
            factor_to_var[fname] = {}
            for var in factor.variables:
                card = self.cardinalities[var]
                factor_to_var[fname][var] = np.ones(card) / card

        for iteration in range(max_iter):
            old_f2v = {fn: {v: msg.copy() for v, msg in msgs.items()}
                       for fn, msgs in factor_to_var.items()}

            # Factor -> variable: use max instead of sum
            for fname, factor in self.factors:
                fvars = factor.variables
                for target_var in fvars:
                    other_vars = [v for v in fvars if v != target_var]
                    target_card = self.cardinalities[target_var]
                    msg = np.zeros(target_card)

                    if not other_vars:
                        for val in range(target_card):
                            assignment = {target_var: val}
                            msg[val] = factor.get_value(assignment)
                    else:
                        other_ranges = [range(self.cardinalities[v]) for v in other_vars]
                        for target_val in range(target_card):
                            max_val = 0.0
                            for other_vals in iter_product(*other_ranges):
                                assignment = {target_var: target_val}
                                assignment.update(dict(zip(other_vars, other_vals)))
                                fval = factor.get_value(assignment)
                                prod = fval
                                for ov, oval in zip(other_vars, other_vals):
                                    prod *= var_to_factor[ov][fname][oval]
                                max_val = max(max_val, prod)
                            msg[target_val] = max_val

                    total = np.sum(msg)
                    if total > 0:
                        msg /= total
                    factor_to_var[fname][target_var] = msg

            # Variable -> factor
            for var in self.variables:
                for target_fname in self._var_to_factors[var]:
                    card = self.cardinalities[var]
                    msg = np.ones(card)
                    for fname in self._var_to_factors[var]:
                        if fname != target_fname:
                            msg *= factor_to_var[fname][var]
                    total = np.sum(msg)
                    if total > 0:
                        msg /= total
                    var_to_factor[var][target_fname] = msg

            max_diff = 0.0
            for fname in factor_to_var:
                for var in factor_to_var[fname]:
                    diff = np.max(np.abs(factor_to_var[fname][var] - old_f2v[fname][var]))
                    max_diff = max(max_diff, diff)
            if max_diff < tol:
                break

        # MAP assignment
        beliefs = {}
        for var in self.variables:
            card = self.cardinalities[var]
            belief = np.ones(card)
            for fname in self._var_to_factors[var]:
                belief *= factor_to_var[fname][var]
            beliefs[var] = belief

        map_assignment = {}
        for var in self.variables:
            map_assignment[var] = int(np.argmax(beliefs[var]))

        return map_assignment, beliefs


# ---------------------------------------------------------------------------
# JunctionTree
# ---------------------------------------------------------------------------

class JunctionTree:
    """Exact inference via junction tree (clique tree) algorithm."""

    def __init__(self, bn):
        """Build junction tree from a BayesianNetwork."""
        self.bn = bn
        self.cliques = []
        self.sepsets = []
        self.tree_edges = []
        self.clique_factors = {}
        self._build(bn)

    def _build(self, bn):
        """Build junction tree: moralize, triangulate, find cliques, build tree."""
        # 1. Moralize
        moral_edges = set()
        for node in bn.nodes:
            parents = bn.get_parents(node)
            for p in parents:
                moral_edges.add(tuple(sorted([p, node])))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moral_edges.add(tuple(sorted([parents[i], parents[j]])))

        # Build adjacency for moral graph
        adj = defaultdict(set)
        for a, b in moral_edges:
            adj[a].add(b)
            adj[b].add(a)

        # 2. Triangulate (greedy: eliminate node with min fill-in)
        remaining = set(bn.nodes)
        elim_order = []
        elim_cliques = []

        while remaining:
            # Pick node with minimum fill edges needed
            best_node = None
            best_fill = float('inf')
            for node in remaining:
                neighbors_in = adj[node] & remaining
                fill = 0
                nbrs = list(neighbors_in)
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        if nbrs[j] not in adj[nbrs[i]]:
                            fill += 1
                if fill < best_fill:
                    best_fill = fill
                    best_node = node

            neighbors_in = adj[best_node] & remaining
            clique = frozenset(neighbors_in | {best_node})
            elim_cliques.append(clique)

            # Add fill edges
            nbrs = list(neighbors_in)
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    adj[nbrs[i]].add(nbrs[j])
                    adj[nbrs[j]].add(nbrs[i])

            elim_order.append(best_node)
            remaining.remove(best_node)

        # 3. Find maximal cliques (remove subsets)
        maximal = []
        for c in elim_cliques:
            is_subset = False
            for other in elim_cliques:
                if c != other and c < other:
                    is_subset = True
                    break
            if not is_subset and c not in maximal:
                maximal.append(c)

        self.cliques = [set(c) for c in maximal]

        # 4. Build clique tree using maximum spanning tree on sepset sizes
        if len(self.cliques) <= 1:
            self.tree_edges = []
        else:
            # Compute all pairwise sepsets
            edges = []
            for i in range(len(self.cliques)):
                for j in range(i + 1, len(self.cliques)):
                    sepset = self.cliques[i] & self.cliques[j]
                    if sepset:
                        edges.append((len(sepset), i, j, sepset))

            # Maximum spanning tree (Kruskal's, sort descending by sepset size)
            edges.sort(reverse=True, key=lambda x: x[0])
            parent_uf = list(range(len(self.cliques)))

            def find(x):
                while parent_uf[x] != x:
                    parent_uf[x] = parent_uf[parent_uf[x]]
                    x = parent_uf[x]
                return x

            for size, i, j, sepset in edges:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent_uf[ri] = rj
                    self.tree_edges.append((i, j))
                    self.sepsets.append(sepset)

        # 5. Assign CPD factors to cliques
        self.clique_factors = {i: [] for i in range(len(self.cliques))}
        for node in bn.nodes:
            cpd = bn.cpds[node]
            cpd_vars = set(cpd.variables)
            # Find smallest clique containing all CPD variables
            best_idx = None
            best_size = float('inf')
            for idx, clique in enumerate(self.cliques):
                if cpd_vars <= clique and len(clique) < best_size:
                    best_idx = idx
                    best_size = len(clique)
            if best_idx is not None:
                self.clique_factors[best_idx].append(cpd)

    def query(self, query_vars, evidence=None):
        """Perform inference using message passing on the junction tree."""
        if evidence is None:
            evidence = {}

        # Initialize clique potentials
        potentials = {}
        for idx in range(len(self.cliques)):
            clique_vars = sorted(self.cliques[idx])
            card = {v: self.bn.cardinalities[v] for v in clique_vars}
            pot = Factor(clique_vars, card)
            # Multiply in assigned factors
            for f in self.clique_factors[idx]:
                f_reduced = f.reduce(evidence)
                pot = pot.multiply(f_reduced)
            # Apply evidence reduction
            pot = pot.reduce(evidence)
            potentials[idx] = pot

        if len(self.cliques) == 1:
            # Only one clique, just marginalize
            result = potentials[0]
            for var in list(result.variables):
                if var not in query_vars:
                    result = result.marginalize(var)
            result.normalize()
            return result

        # Build adjacency for tree
        tree_adj = defaultdict(list)
        for i, j in self.tree_edges:
            tree_adj[i].append(j)
            tree_adj[j].append(i)

        # Message passing: collect to root, then distribute
        root = 0
        messages = {}

        # Collect phase (leaves to root)
        visited = set()
        order = []

        def dfs_order(node, parent):
            visited.add(node)
            for nbr in tree_adj[node]:
                if nbr not in visited:
                    dfs_order(nbr, node)
            if parent is not None:
                order.append((node, parent))

        dfs_order(root, None)

        for src, dst in order:
            msg = potentials[src]
            # Multiply incoming messages (except from dst)
            for nbr in tree_adj[src]:
                if nbr != dst and (nbr, src) in messages:
                    msg = msg.multiply(messages[(nbr, src)])
            # Marginalize to sepset
            sepset = self.cliques[src] & self.cliques[dst]
            # Remove evidence vars from consideration
            sepset_free = sepset - set(evidence.keys())
            msg_vars = set(msg.variables)
            for var in list(msg_vars):
                if var not in sepset_free:
                    msg = msg.marginalize(var)
            messages[(src, dst)] = msg

        # Distribute phase (root to leaves)
        dist_order = list(reversed(order))
        for dst, src in dist_order:
            msg = potentials[src]
            for nbr in tree_adj[src]:
                if nbr != dst and (nbr, src) in messages:
                    msg = msg.multiply(messages[(nbr, src)])
            sepset = self.cliques[src] & self.cliques[dst]
            sepset_free = sepset - set(evidence.keys())
            msg_vars = set(msg.variables)
            for var in list(msg_vars):
                if var not in sepset_free:
                    msg = msg.marginalize(var)
            messages[(src, dst)] = msg

        # Find clique containing query variables
        query_set = set(query_vars)
        best_clique = None
        for idx, clique in enumerate(self.cliques):
            free_clique = clique - set(evidence.keys())
            if query_set <= free_clique:
                best_clique = idx
                break

        if best_clique is None:
            # Fall back to variable elimination
            return self.bn.variable_elimination(query_vars, evidence)

        # Compute belief at best clique
        belief = potentials[best_clique]
        for nbr in tree_adj[best_clique]:
            if (nbr, best_clique) in messages:
                belief = belief.multiply(messages[(nbr, best_clique)])

        # Marginalize to query variables
        for var in list(belief.variables):
            if var not in query_vars:
                belief = belief.marginalize(var)

        belief.normalize()
        return belief


# ---------------------------------------------------------------------------
# StructureLearning
# ---------------------------------------------------------------------------

class StructureLearning:
    """Learn Bayesian Network structure from data."""

    def __init__(self, data, cardinalities):
        """
        data: list of dicts (each dict is a sample)
        cardinalities: dict of variable -> number of states
        """
        self.data = data
        self.cardinalities = cardinalities
        self.variables = list(cardinalities.keys())
        self.n = len(data)

    def _count(self, variables, data=None):
        """Count joint occurrences."""
        if data is None:
            data = self.data
        counts = defaultdict(int)
        for sample in data:
            key = tuple(sample[v] for v in variables)
            counts[key] += 1
        return counts

    def _estimate_cpd(self, node, parents):
        """Estimate CPD from data using MLE with Laplace smoothing."""
        all_vars = list(parents) + [node]
        card = {v: self.cardinalities[v] for v in all_vars}

        # For Factor, variable order is parents first, then node
        factor_vars = list(parents) + [node]
        factor = Factor(factor_vars, card)

        alpha = 1.0  # Laplace smoothing

        if not parents:
            counts = self._count([node])
            total = sum(counts.values()) + alpha * self.cardinalities[node]
            for val in range(self.cardinalities[node]):
                prob = (counts.get((val,), 0) + alpha) / total
                factor.set_value({node: val}, prob)
        else:
            parent_configs = list(iter_product(*[range(self.cardinalities[p]) for p in parents]))
            for pconfig in parent_configs:
                parent_assignment = dict(zip(parents, pconfig))
                matching = [s for s in self.data
                            if all(s[p] == v for p, v in parent_assignment.items())]
                node_counts = defaultdict(int)
                for s in matching:
                    node_counts[s[node]] += 1
                total = len(matching) + alpha * self.cardinalities[node]
                for val in range(self.cardinalities[node]):
                    prob = (node_counts[val] + alpha) / total
                    assignment = dict(parent_assignment)
                    assignment[node] = val
                    factor.set_value(assignment, prob)

        return factor

    def bic_score(self, node, parents):
        """Compute BIC score for node given parents."""
        cpd = self._estimate_cpd(node, parents)
        ll = 0.0
        for sample in self.data:
            val = cpd.get_value(sample)
            if val > 0:
                ll += math.log(val)
            else:
                ll += -100  # penalty for zero probability

        # Number of free parameters
        parent_configs = 1
        for p in parents:
            parent_configs *= self.cardinalities[p]
        k = parent_configs * (self.cardinalities[node] - 1)

        return ll - 0.5 * k * math.log(self.n)

    def k2_search(self, ordering=None, max_parents=3):
        """K2 greedy structure learning algorithm."""
        if ordering is None:
            ordering = list(self.variables)

        bn = BayesianNetwork()
        for v in ordering:
            bn.add_node(v, self.cardinalities[v])

        for i, node in enumerate(ordering):
            parents = []
            current_score = self.bic_score(node, parents)

            improved = True
            while improved and len(parents) < max_parents:
                improved = False
                best_parent = None
                best_score = current_score

                for j in range(i):
                    candidate = ordering[j]
                    if candidate not in parents:
                        new_parents = parents + [candidate]
                        score = self.bic_score(node, new_parents)
                        if score > best_score:
                            best_score = score
                            best_parent = candidate

                if best_parent is not None:
                    parents.append(best_parent)
                    current_score = best_score
                    improved = True

            for p in parents:
                bn.add_edge(p, node)

            cpd = self._estimate_cpd(node, parents)
            bn.set_cpd(node, cpd)

        return bn

    def hill_climb(self, max_parents=3, max_iter=100):
        """Hill-climbing structure learning."""
        bn = BayesianNetwork()
        for v in self.variables:
            bn.add_node(v, self.cardinalities[v])

        # Start with no edges; compute initial score
        edges = set()
        parents_map = defaultdict(list)

        def total_score():
            s = 0.0
            for node in self.variables:
                s += self.bic_score(node, parents_map[node])
            return s

        # Initialize CPDs
        for node in self.variables:
            cpd = self._estimate_cpd(node, [])
            bn.set_cpd(node, cpd)

        current_score = total_score()

        for iteration in range(max_iter):
            best_op = None
            best_score = current_score

            # Try adding edges
            for i, a in enumerate(self.variables):
                for j, b in enumerate(self.variables):
                    if a == b:
                        continue
                    if (a, b) in edges:
                        continue
                    if len(parents_map[b]) >= max_parents:
                        continue
                    # Check no cycle
                    if self._would_create_cycle(a, b, parents_map):
                        continue
                    new_parents = parents_map[b] + [a]
                    new_score = current_score - self.bic_score(b, parents_map[b]) + self.bic_score(b, new_parents)
                    if new_score > best_score:
                        best_score = new_score
                        best_op = ('add', a, b)

            # Try removing edges
            for a, b in list(edges):
                new_parents = [p for p in parents_map[b] if p != a]
                new_score = current_score - self.bic_score(b, parents_map[b]) + self.bic_score(b, new_parents)
                if new_score > best_score:
                    best_score = new_score
                    best_op = ('remove', a, b)

            # Try reversing edges
            for a, b in list(edges):
                # Remove a->b, add b->a
                if (b, a) in edges:
                    continue
                if len(parents_map[a]) >= max_parents:
                    continue
                new_parents_b = [p for p in parents_map[b] if p != a]
                temp_parents = dict(parents_map)
                temp_parents[b] = new_parents_b
                if self._would_create_cycle(b, a, temp_parents):
                    continue
                new_parents_a = parents_map[a] + [b]
                score_diff = (self.bic_score(b, new_parents_b) - self.bic_score(b, parents_map[b]) +
                              self.bic_score(a, new_parents_a) - self.bic_score(a, parents_map[a]))
                new_score = current_score + score_diff
                if new_score > best_score:
                    best_score = new_score
                    best_op = ('reverse', a, b)

            if best_op is None:
                break

            op, a, b = best_op
            if op == 'add':
                edges.add((a, b))
                parents_map[b].append(a)
            elif op == 'remove':
                edges.discard((a, b))
                parents_map[b] = [p for p in parents_map[b] if p != a]
            elif op == 'reverse':
                edges.discard((a, b))
                parents_map[b] = [p for p in parents_map[b] if p != a]
                edges.add((b, a))
                parents_map[a].append(b)

            current_score = best_score

        # Build final BN
        result = BayesianNetwork()
        for v in self.variables:
            result.add_node(v, self.cardinalities[v])
        for a, b in edges:
            result.add_edge(a, b)
        for node in self.variables:
            cpd = self._estimate_cpd(node, parents_map[node])
            result.set_cpd(node, cpd)

        return result

    def _would_create_cycle(self, parent, child, parents_map):
        """Check if adding parent->child creates a cycle."""
        # BFS from child using existing edges
        visited = set()
        queue = deque([parent])
        while queue:
            node = queue.popleft()
            if node == child:
                return True
            if node in visited:
                continue
            visited.add(node)
            for p in parents_map.get(node, []):
                queue.append(p)
        return False


# ---------------------------------------------------------------------------
# DynamicBayesNet
# ---------------------------------------------------------------------------

class DynamicBayesNet:
    """Dynamic Bayesian Network (2-time-slice model)."""

    def __init__(self):
        self.intra_edges = []  # edges within a time slice
        self.inter_edges = []  # edges between consecutive time slices
        self.nodes = []
        self.cardinalities = {}
        self.initial_cpds = {}  # CPDs for t=0
        self.transition_cpds = {}  # CPDs for t>0
        self._intra_parents = defaultdict(list)
        self._inter_parents = defaultdict(list)

    def add_node(self, name, cardinality=2):
        if name not in self.nodes:
            self.nodes.append(name)
            self.cardinalities[name] = cardinality

    def add_intra_edge(self, parent, child):
        """Edge within the same time slice."""
        self.intra_edges.append((parent, child))
        self._intra_parents[child].append(parent)

    def add_inter_edge(self, parent, child):
        """Edge from time t to time t+1."""
        self.inter_edges.append((parent, child))
        self._inter_parents[child].append(parent)

    def set_initial_cpd(self, node, cpd):
        """CPD for t=0."""
        self.initial_cpds[node] = cpd

    def set_transition_cpd(self, node, cpd):
        """CPD for t>0 (conditioned on t and t-1 parents)."""
        self.transition_cpds[node] = cpd

    def unroll(self, T):
        """Unroll for T time steps into a static BayesianNetwork."""
        bn = BayesianNetwork()

        # Create nodes for each time step
        for t in range(T):
            for node in self.nodes:
                tnode = f"{node}_t{t}"
                bn.add_node(tnode, self.cardinalities[node])

        # Add intra-slice edges for each time step
        for t in range(T):
            for parent, child in self.intra_edges:
                bn.add_edge(f"{parent}_t{t}", f"{child}_t{t}")

        # Add inter-slice edges
        for t in range(1, T):
            for parent, child in self.inter_edges:
                bn.add_edge(f"{parent}_t{t-1}", f"{child}_t{t}")

        # Set CPDs
        for node in self.nodes:
            # t=0: use initial CPD
            if node in self.initial_cpds:
                cpd = self.initial_cpds[node]
                # Rename variables
                new_vars = [f"{v}_t0" for v in cpd.variables]
                new_card = {f"{v}_t0": cpd.cardinalities[v] for v in cpd.variables}
                new_cpd = Factor(new_vars, new_card, cpd.values.flatten())
                bn.set_cpd(f"{node}_t0", new_cpd)

            # t>0: use transition CPD
            if node in self.transition_cpds:
                cpd = self.transition_cpds[node]
                for t in range(1, T):
                    new_vars = []
                    new_card = {}
                    for v in cpd.variables:
                        if v.endswith("_prev"):
                            base = v[:-5]
                            tv = f"{base}_t{t-1}"
                        else:
                            tv = f"{v}_t{t}"
                        new_vars.append(tv)
                        new_card[tv] = cpd.cardinalities[v]
                    new_cpd = Factor(new_vars, new_card, cpd.values.flatten())
                    bn.set_cpd(f"{node}_t{t}", new_cpd)

        return bn

    def filter(self, observations, T=None):
        """Forward filtering: P(X_t | y_1:t) for each t."""
        if T is None:
            T = len(observations)

        bn = self.unroll(T)
        results = []

        for t in range(T):
            # Build evidence from observations up to time t
            evidence = {}
            for tt in range(t + 1):
                if tt < len(observations):
                    for var, val in observations[tt].items():
                        evidence[f"{var}_t{tt}"] = val

            # Query hidden variables at time t
            hidden = [n for n in self.nodes if n not in observations[0]]
            query_vars = [f"{h}_t{t}" for h in hidden]
            if query_vars:
                result = bn.variable_elimination(query_vars, evidence)
                results.append(result)
            else:
                results.append(None)

        return results


# ---------------------------------------------------------------------------
# PGMUtils
# ---------------------------------------------------------------------------

class PGMUtils:
    """Utility functions for probabilistic graphical models."""

    @staticmethod
    def mutual_information(data, x, y, cardinalities):
        """Compute mutual information I(X;Y) from data."""
        n = len(data)
        if n == 0:
            return 0.0

        # Count marginals and joint
        px = defaultdict(int)
        py = defaultdict(int)
        pxy = defaultdict(int)

        for sample in data:
            xv = sample[x]
            yv = sample[y]
            px[xv] += 1
            py[yv] += 1
            pxy[(xv, yv)] += 1

        mi = 0.0
        for xv in range(cardinalities[x]):
            for yv in range(cardinalities[y]):
                joint = pxy.get((xv, yv), 0) / n
                mx = px.get(xv, 0) / n
                my = py.get(yv, 0) / n
                if joint > 0 and mx > 0 and my > 0:
                    mi += joint * math.log(joint / (mx * my))
        return mi

    @staticmethod
    def conditional_mutual_information(data, x, y, z, cardinalities):
        """Compute conditional mutual information I(X;Y|Z)."""
        n = len(data)
        if n == 0:
            return 0.0

        cmi = 0.0
        for zv in range(cardinalities[z]):
            z_data = [s for s in data if s[z] == zv]
            nz = len(z_data)
            if nz == 0:
                continue
            pz = nz / n

            px_z = defaultdict(int)
            py_z = defaultdict(int)
            pxy_z = defaultdict(int)

            for s in z_data:
                px_z[s[x]] += 1
                py_z[s[y]] += 1
                pxy_z[(s[x], s[y])] += 1

            for xv in range(cardinalities[x]):
                for yv in range(cardinalities[y]):
                    joint = pxy_z.get((xv, yv), 0) / nz
                    mx = px_z.get(xv, 0) / nz
                    my = py_z.get(yv, 0) / nz
                    if joint > 0 and mx > 0 and my > 0:
                        cmi += pz * joint * math.log(joint / (mx * my))
        return cmi

    @staticmethod
    def independence_test(data, x, y, z_set, cardinalities, alpha=0.05):
        """Chi-squared independence test: X _||_ Y | Z."""
        n = len(data)

        if not z_set:
            # Unconditional test
            observed = defaultdict(int)
            px = defaultdict(int)
            py = defaultdict(int)
            for s in data:
                observed[(s[x], s[y])] += 1
                px[s[x]] += 1
                py[s[y]] += 1

            chi2 = 0.0
            for xv in range(cardinalities[x]):
                for yv in range(cardinalities[y]):
                    expected = px.get(xv, 0) * py.get(yv, 0) / n if n > 0 else 0
                    obs = observed.get((xv, yv), 0)
                    if expected > 0:
                        chi2 += (obs - expected) ** 2 / expected

            df = (cardinalities[x] - 1) * (cardinalities[y] - 1)
            # Approximate p-value using chi2 > threshold
            # For simplicity, use critical values for common alpha=0.05
            critical = _chi2_critical(df, alpha)
            return chi2 < critical, chi2
        else:
            # Conditional test: sum chi2 across Z configurations
            z_list = list(z_set)
            z_ranges = [range(cardinalities[z]) for z in z_list]
            total_chi2 = 0.0
            total_df = 0

            for z_vals in iter_product(*z_ranges):
                z_assignment = dict(zip(z_list, z_vals))
                z_data = [s for s in data
                          if all(s[z] == v for z, v in z_assignment.items())]
                nz = len(z_data)
                if nz == 0:
                    continue

                observed = defaultdict(int)
                px = defaultdict(int)
                py = defaultdict(int)
                for s in z_data:
                    observed[(s[x], s[y])] += 1
                    px[s[x]] += 1
                    py[s[y]] += 1

                for xv in range(cardinalities[x]):
                    for yv in range(cardinalities[y]):
                        expected = px.get(xv, 0) * py.get(yv, 0) / nz if nz > 0 else 0
                        obs = observed.get((xv, yv), 0)
                        if expected > 0:
                            total_chi2 += (obs - expected) ** 2 / expected

                total_df += (cardinalities[x] - 1) * (cardinalities[y] - 1)

            if total_df == 0:
                return True, 0.0
            critical = _chi2_critical(total_df, alpha)
            return total_chi2 < critical, total_chi2

    @staticmethod
    def pc_algorithm(data, cardinalities, alpha=0.05, max_cond_size=3):
        """PC algorithm for constraint-based structure learning.
        Returns a PDAG (partially directed acyclic graph) as a set of edges."""
        variables = list(cardinalities.keys())
        n = len(variables)

        # Start with complete undirected graph
        adj = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                adj[variables[i]].add(variables[j])
                adj[variables[j]].add(variables[i])

        sepsets = {}

        # Remove edges based on conditional independence
        for size in range(max_cond_size + 1):
            for x in variables:
                for y in list(adj[x]):
                    if y not in adj[x]:
                        continue
                    # Find conditioning sets of given size from neighbors of x (excluding y)
                    neighbors = [n for n in adj[x] if n != y]
                    if len(neighbors) < size:
                        continue

                    from itertools import combinations
                    for z_set in combinations(neighbors, size):
                        z_set = set(z_set)
                        independent, _ = PGMUtils.independence_test(
                            data, x, y, z_set, cardinalities, alpha)
                        if independent:
                            adj[x].discard(y)
                            adj[y].discard(x)
                            sepsets[(x, y)] = z_set
                            sepsets[(y, x)] = z_set
                            break

        # Orient v-structures: X - Z - Y where X and Y not adjacent,
        # Z not in sepset(X, Y) => X -> Z <- Y
        directed = set()
        undirected = set()
        for x in variables:
            for y in adj[x]:
                undirected.add(tuple(sorted([x, y])))

        for x in variables:
            for y in variables:
                if x == y or y in adj[x]:
                    continue
                # Find common neighbors
                common = adj[x] & adj[y]
                for z in common:
                    sep = sepsets.get((x, y), set())
                    if z not in sep:
                        directed.add((x, z))
                        directed.add((y, z))
                        undirected.discard(tuple(sorted([x, z])))
                        undirected.discard(tuple(sorted([y, z])))

        return {
            'directed': directed,
            'undirected': undirected,
            'adjacency': dict(adj),
            'sepsets': sepsets
        }

    @staticmethod
    def kl_divergence(p_values, q_values):
        """KL divergence D_KL(P || Q)."""
        kl = 0.0
        for pv, qv in zip(p_values, q_values):
            if pv > 0 and qv > 0:
                kl += pv * math.log(pv / qv)
            elif pv > 0:
                kl = float('inf')
        return kl

    @staticmethod
    def entropy(values):
        """Shannon entropy H(X)."""
        h = 0.0
        for v in values:
            if v > 0:
                h -= v * math.log(v)
        return h


def _chi2_critical(df, alpha=0.05):
    """Approximate chi-squared critical value."""
    # Wilson-Hilferty approximation for chi2 inverse CDF
    if df <= 0:
        return 0.0
    # z values for common alphas
    z_map = {0.01: 2.326, 0.05: 1.645, 0.1: 1.282}
    z = z_map.get(alpha, 1.645)
    # Wilson-Hilferty approximation
    term = 1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))
    return df * (term ** 3)
