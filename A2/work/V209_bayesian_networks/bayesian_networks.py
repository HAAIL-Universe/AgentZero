"""V209: Bayesian Network Inference.

Exact inference over directed acyclic graphical models using:
1. Variable Elimination (VE) -- exact, optimal ordering heuristics
2. Belief Propagation (BP) -- message passing on junction trees
3. Factor operations -- multiply, marginalize, reduce, normalize

Supports:
- DAG construction with CPT (conditional probability table) specification
- Evidence observation (hard evidence)
- Prior/posterior marginal queries
- MAP (Maximum A Posteriori) inference
- D-separation testing
- Topological operations on the DAG
- Junction tree construction for efficient repeated queries
"""

from __future__ import annotations
from collections import defaultdict
from itertools import product as iter_product
from functools import reduce
from math import log, exp


# ---------------------------------------------------------------------------
# Factor: core data structure for probability tables
# ---------------------------------------------------------------------------

class Factor:
    """A discrete probability factor over a set of variables.

    Each variable has a named domain (list of possible values).
    The table maps variable assignments to probabilities.
    """

    def __init__(self, variables: list[str], domains: dict[str, list],
                 table: dict[tuple, float] | None = None):
        self.variables = list(variables)
        self.domains = {v: list(domains[v]) for v in self.variables}
        self.table: dict[tuple, float] = {}
        if table:
            for assignment, prob in table.items():
                self.table[assignment] = prob
        else:
            # Initialize uniform
            for combo in self._all_assignments():
                self.table[combo] = 1.0

    def _all_assignments(self) -> list[tuple]:
        """Generate all possible value assignments for this factor's variables."""
        if not self.variables:
            return [()]
        domain_lists = [self.domains[v] for v in self.variables]
        return list(iter_product(*domain_lists))

    def get(self, assignment: dict[str, object]) -> float:
        """Look up probability for a named assignment dict."""
        key = tuple(assignment[v] for v in self.variables)
        return self.table.get(key, 0.0)

    def set(self, assignment: dict[str, object], value: float):
        """Set probability for a named assignment dict."""
        key = tuple(assignment[v] for v in self.variables)
        self.table[key] = value

    def multiply(self, other: Factor) -> Factor:
        """Multiply two factors (factor product)."""
        new_vars = list(dict.fromkeys(self.variables + other.variables))
        new_domains = {}
        for v in new_vars:
            if v in self.domains:
                new_domains[v] = self.domains[v]
            else:
                new_domains[v] = other.domains[v]

        result = Factor(new_vars, new_domains)
        for assignment in result._all_assignments():
            adict = dict(zip(new_vars, assignment))
            val = self.get(adict) * other.get(adict)
            result.table[assignment] = val
        return result

    def marginalize(self, variable: str) -> Factor:
        """Sum out a variable from this factor."""
        if variable not in self.variables:
            return self
        new_vars = [v for v in self.variables if v != variable]
        new_domains = {v: self.domains[v] for v in new_vars}
        result = Factor(new_vars, new_domains)

        # Zero out
        for key in result.table:
            result.table[key] = 0.0

        var_idx = self.variables.index(variable)
        for assignment, prob in self.table.items():
            reduced = tuple(v for i, v in enumerate(assignment) if i != var_idx)
            result.table[reduced] = result.table.get(reduced, 0.0) + prob
        return result

    def reduce(self, variable: str, value) -> Factor:
        """Reduce factor by observing variable=value (hard evidence)."""
        if variable not in self.variables:
            return self
        var_idx = self.variables.index(variable)
        new_vars = [v for v in self.variables if v != variable]
        new_domains = {v: self.domains[v] for v in new_vars}
        result = Factor(new_vars, new_domains)

        for assignment, prob in self.table.items():
            if assignment[var_idx] == value:
                reduced = tuple(v for i, v in enumerate(assignment) if i != var_idx)
                result.table[reduced] = prob
        return result

    def normalize(self) -> Factor:
        """Normalize so all entries sum to 1."""
        total = sum(self.table.values())
        if total == 0:
            return self
        result = Factor(self.variables, self.domains)
        for key, val in self.table.items():
            result.table[key] = val / total
        return result

    def max_assignment(self) -> tuple[dict[str, object], float]:
        """Return the assignment with maximum probability."""
        best_key = max(self.table, key=self.table.get)
        best_val = self.table[best_key]
        return dict(zip(self.variables, best_key)), best_val

    def entropy(self) -> float:
        """Shannon entropy of the factor (assumes normalized)."""
        h = 0.0
        for p in self.table.values():
            if p > 0:
                h -= p * log(p)
        return h

    def kl_divergence(self, other: Factor) -> float:
        """KL divergence D(self || other). Both must be over same variables."""
        assert self.variables == other.variables
        kl = 0.0
        for key, p in self.table.items():
            q = other.table.get(key, 0.0)
            if p > 0 and q > 0:
                kl += p * log(p / q)
            elif p > 0:
                return float('inf')
        return kl

    def __repr__(self):
        return f"Factor({self.variables})"


# ---------------------------------------------------------------------------
# Bayesian Network: DAG + CPTs
# ---------------------------------------------------------------------------

class BayesianNetwork:
    """A Bayesian network: directed acyclic graph with CPTs.

    Each node has a domain (list of possible values) and a CPT
    (conditional probability table given parents).
    """

    def __init__(self):
        self.nodes: list[str] = []
        self.parents: dict[str, list[str]] = {}
        self.children: dict[str, list[str]] = defaultdict(list)
        self.domains: dict[str, list] = {}
        self.cpts: dict[str, Factor] = {}

    def add_node(self, name: str, domain: list):
        """Add a node with its domain of possible values."""
        if name in self.domains:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes.append(name)
        self.domains[name] = list(domain)
        self.parents[name] = []

    def add_edge(self, parent: str, child: str):
        """Add a directed edge parent -> child."""
        if parent not in self.domains:
            raise ValueError(f"Node '{parent}' not found")
        if child not in self.domains:
            raise ValueError(f"Node '{child}' not found")
        if parent not in self.parents[child]:
            self.parents[child].append(parent)
            self.children[parent].append(child)

    def set_cpt(self, node: str, table: dict[tuple, float]):
        """Set conditional probability table for a node.

        For a node X with parents P1, P2:
        table keys are (p1_val, p2_val, x_val) tuples.
        """
        parent_vars = self.parents[node]
        all_vars = parent_vars + [node]
        domains = {v: self.domains[v] for v in all_vars}
        self.cpts[node] = Factor(all_vars, domains, table)

    def set_cpt_dict(self, node: str, cpt: dict):
        """Set CPT using a more readable dict format.

        For a root node: {val: prob, ...}
        For a node with parents: {(parent_vals...): {val: prob, ...}, ...}
        """
        parent_vars = self.parents[node]
        all_vars = parent_vars + [node]
        domains = {v: self.domains[v] for v in all_vars}
        table = {}

        if not parent_vars:
            # Root node: cpt is {val: prob}
            for val, prob in cpt.items():
                table[(val,)] = prob
        else:
            for parent_vals, dist in cpt.items():
                if not isinstance(parent_vals, tuple):
                    parent_vals = (parent_vals,)
                for val, prob in dist.items():
                    table[parent_vals + (val,)] = prob

        self.cpts[node] = Factor(all_vars, domains, table)

    def get_factors(self) -> list[Factor]:
        """Get all CPT factors."""
        return [self.cpts[n] for n in self.nodes if n in self.cpts]

    def topological_sort(self) -> list[str]:
        """Topological sort of nodes (parents before children)."""
        in_degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = [n for n in self.nodes if in_degree[n] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self.children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result

    def ancestors(self, node: str) -> set[str]:
        """All ancestors of a node."""
        result = set()
        stack = list(self.parents[node])
        while stack:
            n = stack.pop()
            if n not in result:
                result.add(n)
                stack.extend(self.parents[n])
        return result

    def descendants(self, node: str) -> set[str]:
        """All descendants of a node."""
        result = set()
        stack = list(self.children[node])
        while stack:
            n = stack.pop()
            if n not in result:
                result.add(n)
                stack.extend(self.children[n])
        return result

    def markov_blanket(self, node: str) -> set[str]:
        """Markov blanket: parents + children + co-parents."""
        blanket = set(self.parents[node])
        for child in self.children[node]:
            blanket.add(child)
            blanket.update(self.parents[child])
        blanket.discard(node)
        return blanket

    def is_d_separated(self, x: set[str], y: set[str], z: set[str]) -> bool:
        """Test d-separation: are X and Y d-separated given Z?

        Uses the Bayes-Ball algorithm.
        """
        # Bayes-Ball: determine reachable nodes from X given evidence Z
        visited_top = set()    # visited from top (parent side)
        visited_bottom = set() # visited from bottom (child side)
        queue = []

        # Start: schedule X nodes as coming from top
        for node in x:
            queue.append((node, "bottom"))  # entering from bottom = via child

        while queue:
            node, direction = queue.pop(0)

            if direction == "bottom":
                # Arrived from child side
                if node not in z:
                    # Not observed: pass through to parents and children
                    if node not in visited_top:
                        visited_top.add(node)
                        for parent in self.parents[node]:
                            queue.append((parent, "bottom"))
                    if node not in visited_bottom:
                        visited_bottom.add(node)
                        for child in self.children[node]:
                            queue.append((child, "top"))
            else:
                # direction == "top": arrived from parent side
                if node not in z:
                    # Not observed: pass to children
                    if node not in visited_bottom:
                        visited_bottom.add(node)
                        for child in self.children[node]:
                            queue.append((child, "top"))
                else:
                    # Observed: pass to parents (v-structure activation)
                    if node not in visited_top:
                        visited_top.add(node)
                        for parent in self.parents[node]:
                            queue.append((parent, "bottom"))

        # X and Y are d-separated if no Y node was reached
        reachable = visited_top | visited_bottom
        return len(y & reachable) == 0

    def sample(self, evidence: dict[str, object] | None = None,
               n_samples: int = 1) -> list[dict[str, object]]:
        """Forward sampling (rejection sampling with evidence)."""
        import random
        order = self.topological_sort()
        samples = []
        attempts = 0
        max_attempts = n_samples * 1000

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            sample = {}
            valid = True

            for node in order:
                parent_vals = {p: sample[p] for p in self.parents[node]}
                # Get conditional distribution
                dist = {}
                for val in self.domains[node]:
                    assignment = {**parent_vals, node: val}
                    dist[val] = self.cpts[node].get(assignment)

                # Sample from distribution
                total = sum(dist.values())
                if total == 0:
                    valid = False
                    break
                r = random.random() * total
                cumulative = 0.0
                chosen = self.domains[node][0]
                for val, prob in dist.items():
                    cumulative += prob
                    if r <= cumulative:
                        chosen = val
                        break
                sample[node] = chosen

            if valid:
                if evidence is None or all(sample.get(k) == v for k, v in evidence.items()):
                    samples.append(sample)

        return samples


# ---------------------------------------------------------------------------
# Variable Elimination
# ---------------------------------------------------------------------------

def variable_elimination(bn: BayesianNetwork, query: list[str],
                         evidence: dict[str, object] | None = None,
                         elimination_order: list[str] | None = None) -> Factor:
    """Exact inference via variable elimination.

    Args:
        bn: The Bayesian network.
        query: List of query variable names.
        evidence: Dict of observed variable assignments.
        elimination_order: Order to eliminate hidden variables.
            If None, uses min-degree heuristic.

    Returns:
        Normalized factor over query variables.
    """
    evidence = evidence or {}
    factors = list(bn.get_factors())

    # Apply evidence: reduce factors
    for var, val in evidence.items():
        factors = [f.reduce(var, val) for f in factors]

    # Determine elimination order
    hidden = [n for n in bn.nodes if n not in query and n not in evidence]
    if elimination_order is None:
        elimination_order = _min_degree_order(factors, hidden)
    else:
        elimination_order = [v for v in elimination_order if v in hidden]

    # Eliminate each hidden variable
    for var in elimination_order:
        # Collect factors that mention this variable
        relevant = [f for f in factors if var in f.variables]
        remaining = [f for f in factors if var not in f.variables]

        if relevant:
            # Multiply all relevant factors
            product = relevant[0]
            for f in relevant[1:]:
                product = product.multiply(f)
            # Marginalize out the variable
            marginalized = product.marginalize(var)
            remaining.append(marginalized)

        factors = remaining

    # Multiply remaining factors and normalize
    if not factors:
        return Factor(query, {v: bn.domains[v] for v in query})

    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)

    return result.normalize()


def _min_degree_order(factors: list[Factor], hidden: list[str]) -> list[str]:
    """Min-degree elimination ordering heuristic.

    At each step, eliminate the variable that appears in the fewest factors.
    """
    remaining = list(hidden)
    order = []
    current_factors = list(factors)

    while remaining:
        # Count how many factors each remaining variable appears in
        best_var = None
        best_count = float('inf')
        for var in remaining:
            count = sum(1 for f in current_factors if var in f.variables)
            if count < best_count:
                best_count = count
                best_var = var

        order.append(best_var)
        remaining.remove(best_var)

        # Simulate elimination to update factor structure
        relevant = [f for f in current_factors if best_var in f.variables]
        rest = [f for f in current_factors if best_var not in f.variables]
        if relevant:
            combined_vars = set()
            combined_domains = {}
            for f in relevant:
                for v in f.variables:
                    combined_vars.add(v)
                    combined_domains[v] = f.domains[v]
            combined_vars.discard(best_var)
            if combined_vars:
                placeholder = Factor(list(combined_vars), combined_domains)
                rest.append(placeholder)
        current_factors = rest

    return order


# ---------------------------------------------------------------------------
# MAP Inference
# ---------------------------------------------------------------------------

def map_inference(bn: BayesianNetwork, evidence: dict[str, object] | None = None,
                  map_vars: list[str] | None = None) -> tuple[dict[str, object], float]:
    """Maximum A Posteriori inference.

    Finds the most probable assignment to map_vars given evidence.
    Uses max-elimination (replace sum with max in variable elimination).

    Args:
        bn: Bayesian network.
        evidence: Observed evidence.
        map_vars: Variables to find MAP assignment for. If None, all non-evidence.

    Returns:
        (best_assignment, probability)
    """
    evidence = evidence or {}
    if map_vars is None:
        map_vars = [n for n in bn.nodes if n not in evidence]

    factors = list(bn.get_factors())

    # Apply evidence
    for var, val in evidence.items():
        factors = [f.reduce(var, val) for f in factors]

    # Eliminate non-MAP, non-evidence variables by summing
    hidden = [n for n in bn.nodes if n not in map_vars and n not in evidence]
    elim_order = _min_degree_order(factors, hidden)

    for var in elim_order:
        relevant = [f for f in factors if var in f.variables]
        remaining = [f for f in factors if var not in f.variables]
        if relevant:
            product = relevant[0]
            for f in relevant[1:]:
                product = product.multiply(f)
            marginalized = product.marginalize(var)
            remaining.append(marginalized)
        factors = remaining

    # Now maximize over MAP variables
    if not factors:
        return {}, 0.0

    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)

    return result.max_assignment()


# ---------------------------------------------------------------------------
# Junction Tree (for efficient repeated queries)
# ---------------------------------------------------------------------------

class JunctionTree:
    """Junction tree for efficient belief propagation.

    Construction: moralize -> triangulate -> identify cliques -> build tree.
    Inference: message passing on the tree.
    """

    def __init__(self, bn: BayesianNetwork):
        self.bn = bn
        self.cliques: list[frozenset[str]] = []
        self.separators: dict[tuple[int, int], frozenset[str]] = {}
        self.tree_edges: list[tuple[int, int]] = []
        self.potentials: dict[int, Factor] = {}
        self._calibrated = False

        self._build(bn)

    def _build(self, bn: BayesianNetwork):
        """Build junction tree from BN."""
        # Step 1: Moralize -- marry parents, drop directions
        moral_edges = set()
        for node in bn.nodes:
            parents = bn.parents[node]
            for p in parents:
                moral_edges.add((min(p, node), max(p, node)))
            # Marry parents
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    edge = (min(parents[i], parents[j]),
                            max(parents[i], parents[j]))
                    moral_edges.add(edge)

        # Build adjacency
        adj = defaultdict(set)
        for a, b in moral_edges:
            adj[a].add(b)
            adj[b].add(a)

        # Step 2: Triangulate using min-fill heuristic
        triangulated_adj = defaultdict(set)
        for a, b in moral_edges:
            triangulated_adj[a].add(b)
            triangulated_adj[b].add(a)

        remaining = set(bn.nodes)
        elimination_order = []

        while remaining:
            # Min-fill: pick node that adds fewest fill edges
            best_node = None
            best_fill = float('inf')
            for node in remaining:
                neighbors_in = triangulated_adj[node] & remaining
                fill = 0
                neighbors_list = list(neighbors_in)
                for i in range(len(neighbors_list)):
                    for j in range(i + 1, len(neighbors_list)):
                        if neighbors_list[j] not in triangulated_adj[neighbors_list[i]]:
                            fill += 1
                if fill < best_fill:
                    best_fill = fill
                    best_node = node

            # Add fill edges
            neighbors_in = triangulated_adj[best_node] & remaining
            neighbors_list = list(neighbors_in)
            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    a, b = neighbors_list[i], neighbors_list[j]
                    triangulated_adj[a].add(b)
                    triangulated_adj[b].add(a)

            elimination_order.append(best_node)
            remaining.remove(best_node)

        # Step 3: Find maximal cliques from elimination ordering
        remaining2 = set(bn.nodes)
        cliques = []
        for node in elimination_order:
            neighbors_in = triangulated_adj[node] & remaining2
            clique = frozenset({node} | neighbors_in)
            # Check if subsumed by existing clique
            subsumed = False
            for c in cliques:
                if clique <= c:
                    subsumed = True
                    break
            if not subsumed:
                # Remove cliques subsumed by this one
                cliques = [c for c in cliques if not c <= clique]
                cliques.append(clique)
            remaining2.remove(node)

        self.cliques = cliques

        # Step 4: Build max-weight spanning tree over cliques
        if len(cliques) <= 1:
            self.tree_edges = []
        else:
            # Compute separator weights (intersection sizes)
            edges = []
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    sep = cliques[i] & cliques[j]
                    if sep:
                        edges.append((len(sep), i, j, sep))

            # Kruskal's for max spanning tree
            edges.sort(reverse=True)
            parent_uf = list(range(len(cliques)))

            def find(x):
                while parent_uf[x] != x:
                    parent_uf[x] = parent_uf[parent_uf[x]]
                    x = parent_uf[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent_uf[ra] = rb
                    return True
                return False

            for _, i, j, sep in edges:
                if union(i, j):
                    self.tree_edges.append((i, j))
                    self.separators[(i, j)] = sep
                    self.separators[(j, i)] = sep

        # Step 5: Assign CPT factors to cliques
        for clique_idx in range(len(cliques)):
            clique = cliques[clique_idx]
            domains = {v: bn.domains[v] for v in clique}
            self.potentials[clique_idx] = Factor(list(clique), domains)
            # Initialize to uniform 1.0
            for key in self.potentials[clique_idx].table:
                self.potentials[clique_idx].table[key] = 1.0

        for node in bn.nodes:
            if node not in bn.cpts:
                continue
            factor = bn.cpts[node]
            factor_vars = set(factor.variables)
            # Find smallest clique containing all factor variables
            best_idx = None
            best_size = float('inf')
            for idx, clique in enumerate(cliques):
                if factor_vars <= clique and len(clique) < best_size:
                    best_idx = idx
                    best_size = len(clique)
            if best_idx is not None:
                self.potentials[best_idx] = self.potentials[best_idx].multiply(factor)

    def calibrate(self, evidence: dict[str, object] | None = None):
        """Run belief propagation to calibrate the junction tree.

        Two-pass message passing: collect (leaves to root) then distribute.
        """
        evidence = evidence or {}

        # Apply evidence
        for var, val in evidence.items():
            for idx in range(len(self.cliques)):
                if var in self.cliques[idx]:
                    self.potentials[idx] = self.potentials[idx].reduce(var, val)

        if not self.tree_edges:
            self._calibrated = True
            return

        # Build adjacency list
        adj = defaultdict(list)
        for i, j in self.tree_edges:
            adj[i].append(j)
            adj[j].append(i)

        # Root at clique 0
        root = 0
        # BFS to get message schedule
        visited = {root}
        queue = [root]
        bfs_order = []
        parent_map = {}
        while queue:
            node = queue.pop(0)
            bfs_order.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    parent_map[neighbor] = node

        # Collect phase (leaves to root)
        messages = {}
        for node in reversed(bfs_order[1:]):
            parent = parent_map[node]
            sep = self.separators.get((node, parent),
                                      self.cliques[node] & self.cliques[parent])

            # Message = product of potential and incoming messages, marginalize to separator
            msg_factor = self.potentials[node]
            for child in adj[node]:
                if child != parent and (child, node) in messages:
                    msg_factor = msg_factor.multiply(messages[(child, node)])

            # Marginalize to separator variables
            vars_to_eliminate = [v for v in msg_factor.variables if v not in sep]
            for v in vars_to_eliminate:
                msg_factor = msg_factor.marginalize(v)

            messages[(node, parent)] = msg_factor

        # Update root potential
        for child in adj[root]:
            if (child, root) in messages:
                self.potentials[root] = self.potentials[root].multiply(
                    messages[(child, root)])

        # Distribute phase (root to leaves)
        for node in bfs_order[1:]:
            parent = parent_map[node]
            sep = self.separators.get((parent, node),
                                      self.cliques[parent] & self.cliques[node])

            # Parent's belief marginalized to separator
            parent_belief = self.potentials[parent]
            vars_to_eliminate = [v for v in parent_belief.variables if v not in sep]
            for v in vars_to_eliminate:
                parent_belief = parent_belief.marginalize(v)

            # Divide by old separator message (if exists) and multiply into child
            old_msg = messages.get((node, parent))
            if old_msg:
                # lambda message: parent_sep / old_sep_msg
                # Multiply child by parent_sep, then we need to handle division
                # Simple approach: just multiply the child by the ratio
                lambda_msg = Factor(parent_belief.variables, parent_belief.domains)
                for key, val in parent_belief.table.items():
                    old_val = old_msg.table.get(key, 0.0)
                    if old_val > 0:
                        lambda_msg.table[key] = val / old_val
                    else:
                        lambda_msg.table[key] = 0.0
                self.potentials[node] = self.potentials[node].multiply(lambda_msg)
            else:
                self.potentials[node] = self.potentials[node].multiply(parent_belief)

        self._calibrated = True

    def query(self, variables: list[str]) -> Factor:
        """Query marginal distribution over variables after calibration."""
        if not self._calibrated:
            self.calibrate()

        var_set = set(variables)
        # Find clique containing all query variables
        best_idx = None
        best_size = float('inf')
        for idx, clique in enumerate(self.cliques):
            if var_set <= clique and len(clique) < best_size:
                best_idx = idx
                best_size = len(clique)

        if best_idx is None:
            raise ValueError(
                f"No single clique contains all query variables {variables}. "
                "Use variable elimination for multi-clique queries."
            )

        potential = self.potentials[best_idx]
        # Marginalize out non-query variables
        to_eliminate = [v for v in potential.variables if v not in var_set]
        for v in to_eliminate:
            potential = potential.marginalize(v)

        return potential.normalize()


# ---------------------------------------------------------------------------
# Conditional Independence Testing
# ---------------------------------------------------------------------------

def conditional_independence(bn: BayesianNetwork, x: str, y: str,
                             z: set[str] | None = None) -> bool:
    """Test if X is conditionally independent of Y given Z using d-separation."""
    z = z or set()
    return bn.is_d_separated({x}, {y}, z)


# ---------------------------------------------------------------------------
# Network construction helpers
# ---------------------------------------------------------------------------

def build_chain(names: list[str], domains: list[list],
                cpts: list[dict]) -> BayesianNetwork:
    """Build a simple chain network: X1 -> X2 -> ... -> Xn."""
    bn = BayesianNetwork()
    for name, domain in zip(names, domains):
        bn.add_node(name, domain)
    for i in range(len(names) - 1):
        bn.add_edge(names[i], names[i + 1])
    for name, cpt in zip(names, cpts):
        bn.set_cpt_dict(name, cpt)
    return bn


def build_naive_bayes(class_var: str, class_domain: list,
                       class_prior: dict,
                       features: list[str], feature_domains: list[list],
                       likelihoods: list[dict]) -> BayesianNetwork:
    """Build a Naive Bayes classifier as a Bayesian network.

    Structure: class -> feature1, class -> feature2, ...
    """
    bn = BayesianNetwork()
    bn.add_node(class_var, class_domain)
    bn.set_cpt_dict(class_var, class_prior)

    for feat, domain, lik in zip(features, feature_domains, likelihoods):
        bn.add_node(feat, domain)
        bn.add_edge(class_var, feat)
        bn.set_cpt_dict(feat, lik)

    return bn


# ---------------------------------------------------------------------------
# Diagnostic queries
# ---------------------------------------------------------------------------

def most_probable_explanation(bn: BayesianNetwork,
                              evidence: dict[str, object]) -> tuple[dict, float]:
    """Find the most probable explanation (MPE) for all variables given evidence."""
    return map_inference(bn, evidence)


def mutual_information(bn: BayesianNetwork, x: str, y: str,
                        evidence: dict[str, object] | None = None) -> float:
    """Compute mutual information I(X; Y | evidence) from the network."""
    evidence = evidence or {}
    joint = variable_elimination(bn, [x, y], evidence)
    marginal_x = variable_elimination(bn, [x], evidence)
    marginal_y = variable_elimination(bn, [y], evidence)

    mi = 0.0
    for x_val in bn.domains[x]:
        for y_val in bn.domains[y]:
            p_xy = joint.get({x: x_val, y: y_val})
            p_x = marginal_x.get({x: x_val})
            p_y = marginal_y.get({y: y_val})
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * log(p_xy / (p_x * p_y))
    return mi


def sensitivity_analysis(bn: BayesianNetwork, query_var: str, query_val,
                          evidence: dict[str, object] | None = None) -> dict[str, float]:
    """How much does each unobserved variable's observation change P(query_var=query_val)?

    Returns dict mapping each non-evidence, non-query variable to max absolute
    change in P(query_var=query_val) across its domain values.
    """
    evidence = evidence or {}
    base = variable_elimination(bn, [query_var], evidence)
    base_prob = base.get({query_var: query_val})

    sensitivity = {}
    for node in bn.nodes:
        if node == query_var or node in evidence:
            continue
        max_change = 0.0
        for val in bn.domains[node]:
            new_evidence = {**evidence, node: val}
            result = variable_elimination(bn, [query_var], new_evidence)
            new_prob = result.get({query_var: query_val})
            change = abs(new_prob - base_prob)
            max_change = max(max_change, change)
        sensitivity[node] = max_change

    return sensitivity
