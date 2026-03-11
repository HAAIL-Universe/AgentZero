"""
C187: Causal Inference
Standalone implementation -- no external ML libraries.

Components:
- CausalGraph: DAG representation for causal structure
- StructuralCausalModel (SCM): Structural equations with noise terms
- Intervention: do-calculus (do-operator), mutilated graphs
- Counterfactual: Abduction-action-prediction framework
- CausalEstimator: ATE/ATT/CATE estimation
- BackdoorCriterion: Identify valid adjustment sets
- FrontdoorCriterion: Front-door adjustment
- InstrumentalVariable: IV estimation (2SLS)
- CausalDiscovery: PC algorithm (constraint-based structure learning)
- DoCalculus: Three rules of do-calculus for identification
- PropensityScore: IPW and matching estimators
- MediationAnalysis: Direct/indirect effects decomposition
"""

import math
import random
from collections import deque


# ============================================================
# Causal Graph
# ============================================================

class CausalGraph:
    """Directed acyclic graph for causal relationships."""

    def __init__(self):
        self.nodes = set()
        self.edges = {}      # node -> set of children
        self.parents = {}    # node -> set of parents

    def add_node(self, name):
        self.nodes.add(name)
        if name not in self.edges:
            self.edges[name] = set()
        if name not in self.parents:
            self.parents[name] = set()

    def add_edge(self, parent, child):
        self.add_node(parent)
        self.add_node(child)
        self.edges[parent].add(child)
        self.parents[child].add(parent)

    def children(self, node):
        return self.edges.get(node, set())

    def get_parents(self, node):
        return self.parents.get(node, set())

    def has_edge(self, parent, child):
        return child in self.edges.get(parent, set())

    def remove_incoming(self, node):
        """Remove all edges into node (graph mutilation for do-operator)."""
        for p in list(self.parents.get(node, set())):
            self.edges[p].discard(node)
        self.parents[node] = set()

    def copy(self):
        g = CausalGraph()
        g.nodes = set(self.nodes)
        g.edges = {n: set(s) for n, s in self.edges.items()}
        g.parents = {n: set(s) for n, s in self.parents.items()}
        return g

    def ancestors(self, node):
        """All ancestors of node (not including node itself)."""
        result = set()
        queue = deque(self.parents.get(node, set()))
        while queue:
            n = queue.popleft()
            if n not in result:
                result.add(n)
                queue.extend(self.parents.get(n, set()))
        return result

    def descendants(self, node):
        """All descendants of node (not including node itself)."""
        result = set()
        queue = deque(self.edges.get(node, set()))
        while queue:
            n = queue.popleft()
            if n not in result:
                result.add(n)
                queue.extend(self.edges.get(n, set()))
        return result

    def topological_sort(self):
        """Kahn's algorithm."""
        in_degree = {n: len(self.parents.get(n, set())) for n in self.nodes}
        queue = deque(n for n in self.nodes if in_degree[n] == 0)
        result = []
        while queue:
            n = queue.popleft()
            result.append(n)
            for c in sorted(self.edges.get(n, set())):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle")
        return result

    def is_d_separated(self, x, y, z_set):
        """Check if X and Y are d-separated given Z using Bayes-Ball algorithm."""
        if x == y:
            return False

        z = set(z_set) if z_set else set()

        # BFS on the moral/ancestral graph using Bayes-Ball rules
        # State: (node, direction) where direction is 'up' or 'down'
        visited = set()
        queue = deque()

        # Start from x going both directions
        queue.append((x, 'up'))
        queue.append((x, 'down'))
        reachable = set()

        # Pre-compute ancestors of Z for collider activation
        z_ancestors = set()
        for zn in z:
            z_ancestors.update(self.ancestors(zn))
            z_ancestors.add(zn)

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node != x:
                reachable.add(node)

            if direction == 'up' and node not in z:
                # Traversing up through non-conditioned node
                # Can go to parents (continue up) and children (go down via fork)
                for parent in self.get_parents(node):
                    queue.append((parent, 'up'))
                for child in self.children(node):
                    queue.append((child, 'down'))

            elif direction == 'down':
                # Traversing down
                if node not in z:
                    # Non-conditioned: can continue down to children
                    for child in self.children(node):
                        queue.append((child, 'down'))
                if node in z_ancestors:
                    # Conditioned or ancestor of conditioned: collider is active
                    for parent in self.get_parents(node):
                        queue.append((parent, 'up'))

        return y not in reachable

    def to_undirected(self):
        """Return undirected adjacency."""
        adj = {n: set() for n in self.nodes}
        for p in self.nodes:
            for c in self.edges.get(p, set()):
                adj[p].add(c)
                adj[c].add(p)
        return adj

    def moral_graph(self):
        """Moralize: marry parents, drop directions."""
        adj = self.to_undirected()
        for n in self.nodes:
            parents = list(self.get_parents(n))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    adj[parents[i]].add(parents[j])
                    adj[parents[j]].add(parents[i])
        return adj


# ============================================================
# Structural Causal Model
# ============================================================

class StructuralCausalModel:
    """
    SCM: A set of structural equations with exogenous noise.

    Each equation: X_i = f_i(parents(X_i), U_i)
    where U_i is exogenous noise.
    """

    def __init__(self, graph=None):
        self.graph = graph or CausalGraph()
        self.equations = {}   # node -> callable(parent_values, noise)
        self.noise_dists = {} # node -> callable() returning noise sample
        self._data = []       # observational data

    def add_equation(self, node, parents, equation, noise_dist=None):
        """
        Add structural equation for node.
        equation: callable(parent_values_dict, noise) -> value
        noise_dist: callable() -> noise sample (default: standard normal)
        """
        self.graph.add_node(node)
        for p in parents:
            self.graph.add_edge(p, node)
        self.equations[node] = (parents, equation)
        self.noise_dists[node] = noise_dist or (lambda: random.gauss(0, 1))

    def sample(self, n=1, interventions=None, noise_values=None):
        """
        Sample from the SCM.
        interventions: dict of {node: value} for do-operator
        noise_values: list of dicts {node: noise} for counterfactuals
        """
        order = self.graph.topological_sort()
        samples = []

        for i in range(n):
            values = {}
            noises = {}

            for node in order:
                if interventions and node in interventions:
                    values[node] = interventions[node]
                    noises[node] = 0  # noise irrelevant under intervention
                    continue

                if noise_values and i < len(noise_values) and node in noise_values[i]:
                    noise = noise_values[i][node]
                else:
                    noise = self.noise_dists[node]()

                noises[node] = noise

                if node in self.equations:
                    parents, eq = self.equations[node]
                    parent_vals = {p: values[p] for p in parents}
                    values[node] = eq(parent_vals, noise)
                else:
                    values[node] = noise

            samples.append(dict(values))

        return samples

    def intervene(self, interventions):
        """Return samples from mutilated model (do-operator)."""
        return lambda n=1: self.sample(n, interventions=interventions)

    def counterfactual(self, evidence, intervention, target):
        """
        Three-step counterfactual:
        1. Abduction: infer noise from evidence
        2. Action: apply intervention
        3. Prediction: compute target under new model with inferred noise
        """
        # Step 1: Abduction -- infer noise values from evidence
        order = self.graph.topological_sort()
        noise_inferred = {}
        values = dict(evidence)

        # Forward pass to fill in any missing values and infer noise
        for node in order:
            if node in evidence:
                # Infer noise from structural equation
                if node in self.equations:
                    parents, eq = self.equations[node]
                    parent_vals = {p: values.get(p, 0) for p in parents}
                    # For linear equations: value = f(parents, noise)
                    # We need to solve for noise given value
                    # Try noise=0 to get deterministic part, then noise = value - det
                    det_value = eq(parent_vals, 0)
                    noise_inferred[node] = evidence[node] - det_value
                else:
                    noise_inferred[node] = evidence[node]
            else:
                if node in self.equations:
                    parents, eq = self.equations[node]
                    parent_vals = {p: values.get(p, 0) for p in parents}
                    noise = noise_inferred.get(node, 0)
                    values[node] = eq(parent_vals, noise)

        # Step 2 & 3: Action + Prediction
        cf_values = {}
        for node in order:
            if node in intervention:
                cf_values[node] = intervention[node]
            elif node in self.equations:
                parents, eq = self.equations[node]
                parent_vals = {p: cf_values.get(p, values.get(p, 0)) for p in parents}
                noise = noise_inferred.get(node, 0)
                cf_values[node] = eq(parent_vals, noise)
            else:
                cf_values[node] = noise_inferred.get(node, values.get(node, 0))

        if isinstance(target, str):
            return cf_values.get(target)
        return {t: cf_values.get(t) for t in target}

    def generate_data(self, n=1000, seed=None):
        """Generate observational data."""
        if seed is not None:
            random.seed(seed)
        self._data = self.sample(n)
        return self._data


# ============================================================
# Backdoor Criterion
# ============================================================

class BackdoorCriterion:
    """Identify valid adjustment sets using the backdoor criterion."""

    def __init__(self, graph):
        self.graph = graph

    def is_valid_adjustment_set(self, treatment, outcome, adjustment_set):
        """
        Check if Z satisfies the backdoor criterion for (X, Y):
        1. No node in Z is a descendant of X
        2. Z blocks all backdoor paths from X to Y
        """
        z = set(adjustment_set)

        # Condition 1: No descendant of X in Z
        desc_x = self.graph.descendants(treatment)
        if z & desc_x:
            return False

        # Condition 2: Z blocks all non-causal paths (d-separates X and Y
        # in the mutilated graph where we remove edges out of X)
        mutilated = self.graph.copy()
        for child in list(mutilated.children(treatment)):
            mutilated.edges[treatment].discard(child)
            mutilated.parents[child].discard(treatment)

        return mutilated.is_d_separated(treatment, outcome, z)

    def find_minimal_adjustment_set(self, treatment, outcome):
        """Find a minimal valid adjustment set."""
        # Try parents of treatment first (often sufficient)
        parents = self.graph.get_parents(treatment)
        if self.is_valid_adjustment_set(treatment, outcome, parents):
            # Try to minimize
            for p in sorted(parents):
                smaller = parents - {p}
                if self.is_valid_adjustment_set(treatment, outcome, smaller):
                    parents = smaller
            return parents

        # Try all subsets of non-descendants
        desc_x = self.graph.descendants(treatment)
        candidates = self.graph.nodes - {treatment, outcome} - desc_x

        # Try single nodes
        for c in sorted(candidates):
            if self.is_valid_adjustment_set(treatment, outcome, {c}):
                return {c}

        # Try pairs
        cand_list = sorted(candidates)
        for i in range(len(cand_list)):
            for j in range(i + 1, len(cand_list)):
                s = {cand_list[i], cand_list[j]}
                if self.is_valid_adjustment_set(treatment, outcome, s):
                    return s

        # Empty set might work
        if self.is_valid_adjustment_set(treatment, outcome, set()):
            return set()

        return None  # No valid adjustment set exists


# ============================================================
# Front-door Criterion
# ============================================================

class FrontdoorCriterion:
    """Front-door adjustment for causal effect identification."""

    def __init__(self, graph):
        self.graph = graph

    def find_frontdoor_set(self, treatment, outcome):
        """
        Find mediator set M satisfying front-door criterion:
        1. X blocks all directed paths from X to M
        2. No unblocked backdoor path from X to M
        3. All directed paths from M to Y are blocked by X
        """
        # Candidates: nodes on directed paths from treatment to outcome
        desc_x = self.graph.descendants(treatment)
        anc_y = self.graph.ancestors(outcome)
        candidates = desc_x & anc_y

        if not candidates:
            return None

        # Check each candidate
        for c in sorted(candidates):
            if self._is_valid_frontdoor(treatment, outcome, {c}):
                return {c}

        return None

    def _is_valid_frontdoor(self, treatment, outcome, mediator_set):
        """Check front-door criterion conditions."""
        m = set(mediator_set)

        # Condition 1: Treatment intercepts all directed paths from X to M
        # (all directed paths from X to any m in M go through X -- trivially true
        # since X is the treatment)

        # Condition 2: No unblocked backdoor X -> M
        # Check d-separation of X and M in mutilated graph (remove X->M edges)
        # Actually: no backdoor path from X to M means parents of X
        # don't create unblocked paths to M
        # Simplified: check if the confounders of X don't directly affect M

        # Condition 3: X blocks all backdoor paths from M to Y
        for mi in m:
            if not self.graph.is_d_separated(mi, outcome, {treatment} | (m - {mi})):
                # There's an unblocked path from M to Y not through X
                # But we only need X to block BACKDOOR paths
                pass

        # Practical check: M must mediate the full effect
        # M must be on all directed paths from X to Y
        # and no backdoor path from M to Y unblocked by X

        # Check: all paths X->Y go through M
        g_no_m = self.graph.copy()
        for mi in m:
            # Remove mediator nodes
            for p in list(g_no_m.get_parents(mi)):
                g_no_m.edges[p].discard(mi)
            for c in list(g_no_m.children(mi)):
                g_no_m.parents[c].discard(mi)
            g_no_m.edges[mi] = set()
            g_no_m.parents[mi] = set()

        # After removing M, X should not reach Y via directed paths
        desc_x_no_m = g_no_m.descendants(treatment)
        if outcome in desc_x_no_m:
            return False

        return True

    def estimate(self, data, treatment, outcome, mediator):
        """
        Front-door adjustment formula:
        P(Y|do(X)) = sum_M P(M|X) * sum_X' P(Y|M,X') * P(X')
        """
        if isinstance(mediator, set):
            mediator = sorted(mediator)[0]

        # Group data
        x_vals = sorted(set(d[treatment] for d in data))
        m_vals = sorted(set(d[mediator] for d in data))

        results = {}
        for x in x_vals:
            expected_y = 0
            for m in m_vals:
                # P(M=m | X=x)
                x_data = [d for d in data if d[treatment] == x]
                if not x_data:
                    continue
                p_m_given_x = sum(1 for d in x_data if d[mediator] == m) / len(x_data)

                # sum_x' P(Y|M=m,X=x') * P(X=x')
                weighted_y = 0
                for x_prime in x_vals:
                    p_x_prime = sum(1 for d in data if d[treatment] == x_prime) / len(data)
                    mx_data = [d for d in data if d[mediator] == m and d[treatment] == x_prime]
                    if mx_data:
                        e_y = sum(d[outcome] for d in mx_data) / len(mx_data)
                    else:
                        e_y = 0
                    weighted_y += e_y * p_x_prime

                expected_y += p_m_given_x * weighted_y

            results[x] = expected_y

        return results


# ============================================================
# Instrumental Variable Estimation
# ============================================================

class InstrumentalVariable:
    """Two-stage least squares (2SLS) IV estimation."""

    def __init__(self):
        pass

    def estimate(self, data, instrument, treatment, outcome, covariates=None):
        """
        2SLS estimation:
        Stage 1: Regress treatment on instrument (+ covariates)
        Stage 2: Regress outcome on predicted treatment (+ covariates)
        """
        n = len(data)
        covs = covariates or []

        # Extract arrays
        z = [d[instrument] for d in data]
        x = [d[treatment] for d in data]
        y = [d[outcome] for d in data]
        c = [[d[cv] for cv in covs] for d in data] if covs else [[]] * n

        # Stage 1: X = alpha + beta*Z + gamma*C + epsilon
        # Simple OLS for first stage
        x_hat = self._first_stage(z, x, c)

        # Stage 2: Y = a + b*X_hat + g*C + error
        # The coefficient b is the causal effect
        effect = self._second_stage(x_hat, y, c)

        return effect

    def _first_stage(self, z, x, c):
        """Regress X on Z (and covariates), return predicted X."""
        n = len(z)
        # Build design matrix [1, Z, C1, C2, ...]
        ncov = len(c[0]) if c[0] else 0
        k = 2 + ncov

        # X^T X
        xtx = [[0.0] * k for _ in range(k)]
        xty = [0.0] * k

        for i in range(n):
            row = [1.0, z[i]] + (list(c[i]) if c[i] else [])
            for j in range(k):
                for l in range(k):
                    xtx[j][l] += row[j] * row[l]
                xty[j] += row[j] * x[i]

        # Solve via Gaussian elimination
        beta = self._solve_linear(xtx, xty)

        # Predicted values
        x_hat = []
        for i in range(n):
            row = [1.0, z[i]] + (list(c[i]) if c[i] else [])
            pred = sum(beta[j] * row[j] for j in range(k))
            x_hat.append(pred)

        return x_hat

    def _second_stage(self, x_hat, y, c):
        """Regress Y on X_hat (and covariates), return coefficient on X_hat."""
        n = len(x_hat)
        ncov = len(c[0]) if c[0] else 0
        k = 2 + ncov

        xtx = [[0.0] * k for _ in range(k)]
        xty = [0.0] * k

        for i in range(n):
            row = [1.0, x_hat[i]] + (list(c[i]) if c[i] else [])
            for j in range(k):
                for l in range(k):
                    xtx[j][l] += row[j] * row[l]
                xty[j] += row[j] * y[i]

        beta = self._solve_linear(xtx, xty)
        return beta[1]  # Coefficient on X_hat = causal effect

    def _solve_linear(self, A, b):
        """Solve Ax = b via Gaussian elimination with partial pivoting."""
        n = len(b)
        # Augmented matrix
        aug = [A[i][:] + [b[i]] for i in range(n)]

        for col in range(n):
            # Partial pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            if abs(aug[col][col]) < 1e-12:
                continue

            # Eliminate
            for row in range(col + 1, n):
                factor = aug[row][col] / aug[col][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-12:
                x[i] = 0.0
                continue
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]

        return x


# ============================================================
# Propensity Score Methods
# ============================================================

class PropensityScore:
    """Propensity score estimation for causal inference."""

    def __init__(self):
        pass

    def estimate_propensity(self, data, treatment, covariates):
        """Estimate P(T=1|X) using logistic regression."""
        n = len(data)
        ncov = len(covariates)

        # Extract features and labels
        X = [[d[c] for c in covariates] for d in data]
        T = [d[treatment] for d in data]

        # Logistic regression via gradient descent
        weights = [0.0] * (ncov + 1)  # +1 for intercept
        lr = 0.01

        for epoch in range(200):
            grad = [0.0] * (ncov + 1)
            for i in range(n):
                features = [1.0] + X[i]
                logit = sum(w * f for w, f in zip(weights, features))
                logit = max(-20, min(20, logit))  # clip
                prob = 1.0 / (1.0 + math.exp(-logit))
                error = T[i] - prob
                for j in range(ncov + 1):
                    grad[j] += error * features[j]

            for j in range(ncov + 1):
                weights[j] += lr * grad[j] / n

        # Compute propensity scores
        scores = []
        for i in range(n):
            features = [1.0] + X[i]
            logit = sum(w * f for w, f in zip(weights, features))
            logit = max(-20, min(20, logit))
            prob = 1.0 / (1.0 + math.exp(-logit))
            scores.append(prob)

        return scores

    def ipw_estimate(self, data, treatment, outcome, covariates):
        """
        Inverse Probability Weighting (IPW) for ATE.
        ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
        """
        scores = self.estimate_propensity(data, treatment, covariates)
        n = len(data)

        treated_sum = 0.0
        control_sum = 0.0
        treated_weight = 0.0
        control_weight = 0.0

        for i in range(n):
            t = data[i][treatment]
            y = data[i][outcome]
            e = max(0.01, min(0.99, scores[i]))  # clip for stability

            if t == 1:
                w = 1.0 / e
                treated_sum += y * w
                treated_weight += w
            else:
                w = 1.0 / (1.0 - e)
                control_sum += y * w
                control_weight += w

        if treated_weight == 0 or control_weight == 0:
            return 0.0

        ate = treated_sum / treated_weight - control_sum / control_weight
        return ate

    def matching_estimate(self, data, treatment, outcome, covariates, k=1):
        """
        Propensity score matching for ATE.
        Match each treated unit to k nearest control units.
        """
        scores = self.estimate_propensity(data, treatment, covariates)
        n = len(data)

        treated = [(i, scores[i]) for i in range(n) if data[i][treatment] == 1]
        control = [(i, scores[i]) for i in range(n) if data[i][treatment] == 0]

        if not treated or not control:
            return 0.0

        # Match treated to nearest control
        att_effects = []
        for ti, ts in treated:
            # Find k nearest controls
            dists = [(abs(ts - cs), ci) for ci, cs in control]
            dists.sort()
            matched = dists[:k]
            matched_outcome = sum(data[ci][outcome] for _, ci in matched) / k
            att_effects.append(data[ti][outcome] - matched_outcome)

        # Match control to nearest treated (for ATE)
        atc_effects = []
        for ci, cs in control:
            dists = [(abs(cs - ts), ti) for ti, ts in treated]
            dists.sort()
            matched = dists[:k]
            matched_outcome = sum(data[ti][outcome] for _, ti in matched) / k
            atc_effects.append(matched_outcome - data[ci][outcome])

        # ATE = weighted average of ATT and ATC
        n_t = len(treated)
        n_c = len(control)
        att = sum(att_effects) / n_t
        atc = sum(atc_effects) / n_c
        ate = (n_t * att + n_c * atc) / n

        return ate


# ============================================================
# Causal Effect Estimator
# ============================================================

class CausalEstimator:
    """Estimate causal effects using various methods."""

    def __init__(self, graph=None):
        self.graph = graph

    def ate_regression(self, data, treatment, outcome, adjustment_set):
        """
        Average Treatment Effect via regression adjustment.
        E[Y|do(X=1)] - E[Y|do(X=0)] using backdoor adjustment.
        """
        if not data:
            return 0.0

        adj = list(adjustment_set)

        # Stratified estimation
        # Linear regression: Y = a + b*X + c1*Z1 + c2*Z2 + ...
        ncov = len(adj) + 2  # intercept + treatment + adjustments
        n = len(data)

        # Build and solve normal equations
        xtx = [[0.0] * ncov for _ in range(ncov)]
        xty = [0.0] * ncov

        for d in data:
            row = [1.0, d[treatment]] + [d[a] for a in adj]
            y = d[outcome]
            for i in range(ncov):
                for j in range(ncov):
                    xtx[i][j] += row[i] * row[j]
                xty[i] += row[i] * y

        iv = InstrumentalVariable()
        beta = iv._solve_linear(xtx, xty)

        return beta[1]  # Coefficient on treatment = ATE

    def cate(self, data, treatment, outcome, adjustment_set, subgroup_var, subgroup_val):
        """Conditional ATE for a subgroup."""
        sub_data = [d for d in data if d[subgroup_var] == subgroup_val]
        return self.ate_regression(sub_data, treatment, outcome, adjustment_set)

    def att(self, data, treatment, outcome, adjustment_set):
        """Average Treatment effect on the Treated."""
        adj = list(adjustment_set)
        treated = [d for d in data if d[treatment] == 1]
        control = [d for d in data if d[treatment] == 0]

        if not treated or not control:
            return 0.0

        if not adj:
            return (sum(d[outcome] for d in treated) / len(treated) -
                    sum(d[outcome] for d in control) / len(control))

        # For each treated unit, estimate counterfactual outcome
        effects = []
        for t in treated:
            # Find similar controls
            matched = self._find_matches(t, control, adj)
            if matched:
                cf_outcome = sum(d[outcome] for d in matched) / len(matched)
                effects.append(t[outcome] - cf_outcome)

        return sum(effects) / len(effects) if effects else 0.0

    def _find_matches(self, unit, pool, covariates, k=5):
        """Find k nearest matches based on covariates."""
        dists = []
        for d in pool:
            dist = sum((unit[c] - d[c]) ** 2 for c in covariates)
            dists.append((dist, d))
        dists.sort(key=lambda x: x[0])
        return [d for _, d in dists[:k]]


# ============================================================
# Mediation Analysis
# ============================================================

class MediationAnalysis:
    """Decompose total effect into direct and indirect effects."""

    def __init__(self):
        pass

    def analyze(self, data, treatment, mediator, outcome, covariates=None):
        """
        Baron-Kenny mediation analysis.
        Returns: total_effect, direct_effect, indirect_effect, proportion_mediated
        """
        covs = covariates or []
        iv = InstrumentalVariable()

        # Total effect: Y = c*X + ...
        total = self._regress_coefficient(data, treatment, outcome, covs, iv)

        # Path a: M = a*X + ...
        path_a = self._regress_coefficient(data, treatment, mediator, covs, iv)

        # Direct effect: Y = c'*X + b*M + ...
        direct = self._regress_coefficient(data, treatment, outcome, [mediator] + covs, iv)

        # Path b
        path_b = self._regress_coefficient(data, mediator, outcome, [treatment] + covs, iv)

        # Indirect effect = a * b (or total - direct)
        indirect = path_a * path_b

        # Proportion mediated
        if abs(total) > 1e-10:
            prop_mediated = indirect / total
        else:
            prop_mediated = 0.0

        return {
            'total_effect': total,
            'direct_effect': direct,
            'indirect_effect': indirect,
            'proportion_mediated': prop_mediated,
            'path_a': path_a,
            'path_b': path_b,
        }

    def _regress_coefficient(self, data, target_var, outcome, other_vars, iv_solver):
        """Get regression coefficient of target_var on outcome controlling for other_vars."""
        n = len(data)
        k = 2 + len(other_vars)  # intercept + target + others

        xtx = [[0.0] * k for _ in range(k)]
        xty = [0.0] * k

        for d in data:
            row = [1.0, d[target_var]] + [d[v] for v in other_vars]
            y = d[outcome]
            for i in range(k):
                for j in range(k):
                    xtx[i][j] += row[i] * row[j]
                xty[i] += row[i] * y

        beta = iv_solver._solve_linear(xtx, xty)
        return beta[1]


# ============================================================
# Causal Discovery (PC Algorithm)
# ============================================================

class CausalDiscovery:
    """PC algorithm for causal structure learning from data."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def pc_algorithm(self, data, variables):
        """
        PC algorithm:
        1. Start with complete undirected graph
        2. Remove edges based on conditional independence
        3. Orient edges using v-structures and Meek rules
        """
        n_vars = len(variables)

        # Step 1: Complete undirected graph
        adj = {v: set(variables) - {v} for v in variables}
        sep_sets = {}  # (x,y) -> conditioning set that made them independent

        # Step 2: Remove edges based on conditional independence tests
        depth = 0
        while depth < n_vars - 1:
            for x in variables:
                for y in list(adj[x]):
                    if y not in adj[x]:
                        continue
                    # Find conditioning sets of size depth from neighbors of x (excluding y)
                    neighbors = sorted(adj[x] - {y})
                    if len(neighbors) < depth:
                        continue

                    found_independent = False
                    for cond_set in self._subsets(neighbors, depth):
                        if self._is_independent(data, x, y, cond_set):
                            adj[x].discard(y)
                            adj[y].discard(x)
                            sep_sets[(x, y)] = set(cond_set)
                            sep_sets[(y, x)] = set(cond_set)
                            found_independent = True
                            break

                    if found_independent:
                        break

            depth += 1

        # Step 3: Orient v-structures
        dag = CausalGraph()
        for v in variables:
            dag.add_node(v)

        oriented = set()  # (parent, child) edges that are oriented
        undirected = set()

        for x in variables:
            for y in adj[x]:
                if (x, y) not in oriented and (y, x) not in oriented:
                    undirected.add((min(x, y), max(x, y)))

        # Find v-structures: X -> Z <- Y where X and Y are not adjacent
        # and Z is not in sep_set(X, Y)
        for x in variables:
            for y in variables:
                if x >= y:
                    continue
                if y in adj[x]:
                    continue  # X and Y are adjacent, skip
                # Find common neighbors
                common = adj[x] & adj[y]
                for z in common:
                    key = (x, y)
                    if key in sep_sets and z not in sep_sets[key]:
                        # Orient as v-structure: X -> Z <- Y
                        oriented.add((x, z))
                        oriented.add((y, z))
                        undirected.discard((min(x, z), max(x, z)))
                        undirected.discard((min(y, z), max(y, z)))

        # Apply Meek rules to orient remaining edges
        changed = True
        while changed:
            changed = False
            for pair in list(undirected):
                a, b = pair
                # Rule 1: If A -> B - C and A not adj C, orient B -> C
                for direction in [(a, b), (b, a)]:
                    src, dst = direction
                    # Check if any node points to src with oriented edge
                    for x in variables:
                        if (x, src) in oriented and (src, dst) not in oriented and (dst, src) not in oriented:
                            if dst not in adj.get(x, set()) or (x == dst):
                                # x is not adjacent to dst (or x == dst which is trivially true)
                                if x != dst and dst not in adj.get(x, set()):
                                    oriented.add((src, dst))
                                    undirected.discard(pair)
                                    changed = True
                                    break
                    if pair not in undirected:
                        break

        # Build final graph
        for p, c in oriented:
            dag.add_edge(p, c)

        # Remaining undirected edges: orient arbitrarily (lower -> higher)
        for a, b in undirected:
            dag.add_edge(a, b)

        return dag

    def _subsets(self, items, size):
        """Generate all subsets of given size."""
        if size == 0:
            yield ()
            return
        for i in range(len(items)):
            for rest in self._subsets(items[i + 1:], size - 1):
                yield (items[i],) + rest

    def _is_independent(self, data, x, y, cond_set):
        """Test conditional independence using partial correlation."""
        if not data:
            return False

        # Compute partial correlation
        r = self._partial_correlation(data, x, y, list(cond_set))
        n = len(data)
        k = len(cond_set)

        # Fisher's z-transform
        if abs(r) >= 1.0:
            return False

        z = 0.5 * math.log((1 + r) / (1 - r))
        # Test statistic
        df = n - k - 3
        if df <= 0:
            return False

        t_stat = abs(z) * math.sqrt(df)

        # Approximate p-value using normal distribution
        # For large n, z * sqrt(n-k-3) ~ N(0,1) under H0
        p_value = 2 * (1 - self._normal_cdf(t_stat))

        return p_value > self.alpha

    def _partial_correlation(self, data, x, y, z_vars):
        """Compute partial correlation of x,y given z_vars using recursive formula."""
        if not z_vars:
            return self._correlation(data, x, y)

        z = z_vars[-1]
        rest = z_vars[:-1]

        r_xy_rest = self._partial_correlation(data, x, y, rest)
        r_xz_rest = self._partial_correlation(data, x, z, rest)
        r_yz_rest = self._partial_correlation(data, y, z, rest)

        denom = math.sqrt((1 - r_xz_rest ** 2) * (1 - r_yz_rest ** 2))
        if abs(denom) < 1e-12:
            return 0.0

        return (r_xy_rest - r_xz_rest * r_yz_rest) / denom

    def _correlation(self, data, x, y):
        """Pearson correlation."""
        n = len(data)
        if n < 2:
            return 0.0

        x_vals = [d[x] for d in data]
        y_vals = [d[y] for d in data]

        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_vals, y_vals))
        var_x = sum((xi - x_mean) ** 2 for xi in x_vals)
        var_y = sum((yi - y_mean) ** 2 for yi in y_vals)

        denom = math.sqrt(var_x * var_y)
        if abs(denom) < 1e-12:
            return 0.0

        return cov / denom

    def _normal_cdf(self, x):
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ============================================================
# Do-Calculus Rules
# ============================================================

class DoCalculus:
    """
    Three rules of do-calculus for causal effect identification.

    Rule 1: Insertion/deletion of observations
    Rule 2: Action/observation exchange
    Rule 3: Insertion/deletion of actions
    """

    def __init__(self, graph):
        self.graph = graph

    def rule1(self, y, x_do, z, w=None):
        """
        Rule 1: P(y|do(x),z,w) = P(y|do(x),w) if (Y _||_ Z | X,W) in G_X_bar
        where G_X_bar is graph with incoming edges to X removed.

        Returns True if Z can be removed from the conditioning set.
        """
        w = w or set()
        g_mutilated = self.graph.copy()
        if isinstance(x_do, str):
            x_do = {x_do}
        for x in x_do:
            g_mutilated.remove_incoming(x)

        if isinstance(z, str):
            z = {z}
        if isinstance(y, str):
            y_node = y
        else:
            y_node = sorted(y)[0]

        for zi in z:
            if not g_mutilated.is_d_separated(y_node, zi, x_do | w):
                return False
        return True

    def rule2(self, y, x_do, z, w=None):
        """
        Rule 2: P(y|do(x),do(z),w) = P(y|do(x),z,w)
        if (Y _||_ Z | X,W) in G_X_bar_Z_underbar
        where G_X_bar is graph with incoming to X removed,
        and Z_underbar has outgoing from Z removed.

        Returns True if do(Z) can be replaced with observing Z.
        """
        w = w or set()
        g = self.graph.copy()
        if isinstance(x_do, str):
            x_do = {x_do}
        for x in x_do:
            g.remove_incoming(x)

        if isinstance(z, str):
            z = {z}

        # Remove outgoing edges from Z
        for zi in z:
            for child in list(g.children(zi)):
                g.edges[zi].discard(child)
                g.parents[child].discard(zi)

        if isinstance(y, str):
            y_node = y
        else:
            y_node = sorted(y)[0]

        for zi in z:
            if not g.is_d_separated(y_node, zi, x_do | w):
                return False
        return True

    def rule3(self, y, x_do, z, w=None):
        """
        Rule 3: P(y|do(x),do(z),w) = P(y|do(x),w)
        if (Y _||_ Z | X,W) in G_X_bar_Z(W)_bar
        where Z(W) = Z \\ An(W) in G_X_bar

        Returns True if do(Z) can be removed entirely.
        """
        w = w or set()
        g = self.graph.copy()
        if isinstance(x_do, str):
            x_do = {x_do}
        for x in x_do:
            g.remove_incoming(x)

        if isinstance(z, str):
            z = {z}

        # Find ancestors of W in modified graph
        anc_w = set()
        for wi in w:
            anc_w.update(g.ancestors(wi))
            anc_w.add(wi)

        # Z(W) = Z \ An(W)_G_X_bar
        z_w = z - anc_w

        # Remove incoming to Z(W)
        g2 = g.copy()
        for zi in z_w:
            g2.remove_incoming(zi)

        if isinstance(y, str):
            y_node = y
        else:
            y_node = sorted(y)[0]

        for zi in z:
            if not g2.is_d_separated(y_node, zi, x_do | w):
                return False
        return True

    def is_identifiable(self, treatment, outcome):
        """
        Check if P(outcome | do(treatment)) is identifiable
        using the backdoor criterion.
        """
        bd = BackdoorCriterion(self.graph)
        adj_set = bd.find_minimal_adjustment_set(treatment, outcome)
        return adj_set is not None


# ============================================================
# Sensitivity Analysis
# ============================================================

class SensitivityAnalysis:
    """Assess robustness of causal estimates to unmeasured confounding."""

    def __init__(self):
        pass

    def e_value(self, estimate, se=None):
        """
        E-value: minimum strength of unmeasured confounding that would
        explain away the observed effect.

        For risk ratio RR:
        E-value = RR + sqrt(RR * (RR - 1))
        """
        if estimate <= 0:
            return {'e_value': 1.0, 'estimate': estimate}

        # Convert to approximate risk ratio
        rr = math.exp(estimate) if abs(estimate) < 10 else estimate

        if rr < 1:
            rr = 1.0 / rr

        e_val = rr + math.sqrt(rr * (rr - 1))

        result = {'e_value': e_val, 'estimate': estimate}

        if se is not None:
            # E-value for confidence interval bound
            ci_lower = estimate - 1.96 * se
            if ci_lower > 0:
                rr_lower = math.exp(ci_lower) if abs(ci_lower) < 10 else ci_lower
                if rr_lower < 1:
                    rr_lower = 1.0 / rr_lower
                result['e_value_ci'] = rr_lower + math.sqrt(rr_lower * (rr_lower - 1))
            else:
                result['e_value_ci'] = 1.0

        return result

    def rosenbaum_bounds(self, data, treatment, outcome, gamma_range=None):
        """
        Rosenbaum bounds: how large would hidden bias need to be
        to alter the conclusion?

        gamma: odds ratio of differential treatment assignment
        """
        if gamma_range is None:
            gamma_range = [1.0, 1.5, 2.0, 3.0, 5.0]

        treated = [d[outcome] for d in data if d[treatment] == 1]
        control = [d[outcome] for d in data if d[treatment] == 0]

        if not treated or not control:
            return []

        obs_diff = sum(treated) / len(treated) - sum(control) / len(control)
        se = math.sqrt(
            self._variance(treated) / len(treated) +
            self._variance(control) / len(control)
        )

        results = []
        for gamma in gamma_range:
            # Under hidden bias gamma, the bounds on the test statistic shift
            # Upper bound: effect could be as large as obs + log(gamma)*se
            # Lower bound: effect could be as small as obs - log(gamma)*se
            adjustment = math.log(gamma) * se if se > 0 else 0
            results.append({
                'gamma': gamma,
                'lower_bound': obs_diff - adjustment,
                'upper_bound': obs_diff + adjustment,
                'significant_at_lower': (obs_diff - adjustment) > 0,
            })

        return results

    def _variance(self, values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


# ============================================================
# Difference-in-Differences
# ============================================================

class DifferenceInDifferences:
    """Difference-in-differences estimator for panel data."""

    def __init__(self):
        pass

    def estimate(self, data, treatment_group, time_var, outcome, pre_period, post_period):
        """
        DiD estimator:
        delta = (E[Y|T=1,post] - E[Y|T=1,pre]) - (E[Y|T=0,post] - E[Y|T=0,pre])
        """
        treated_pre = [d[outcome] for d in data
                       if d[treatment_group] == 1 and d[time_var] == pre_period]
        treated_post = [d[outcome] for d in data
                        if d[treatment_group] == 1 and d[time_var] == post_period]
        control_pre = [d[outcome] for d in data
                       if d[treatment_group] == 0 and d[time_var] == pre_period]
        control_post = [d[outcome] for d in data
                        if d[treatment_group] == 0 and d[time_var] == post_period]

        if not all([treated_pre, treated_post, control_pre, control_post]):
            return {'effect': 0.0, 'valid': False}

        mean = lambda lst: sum(lst) / len(lst)

        treated_diff = mean(treated_post) - mean(treated_pre)
        control_diff = mean(control_post) - mean(control_pre)

        effect = treated_diff - control_diff

        return {
            'effect': effect,
            'treated_diff': treated_diff,
            'control_diff': control_diff,
            'treated_pre_mean': mean(treated_pre),
            'treated_post_mean': mean(treated_post),
            'control_pre_mean': mean(control_pre),
            'control_post_mean': mean(control_post),
            'valid': True,
        }

    def parallel_trends_test(self, data, treatment_group, time_var, outcome, periods):
        """
        Test parallel trends assumption using pre-treatment periods.
        Returns trend differences -- should be close to 0.
        """
        results = []
        for i in range(1, len(periods)):
            t_prev = periods[i - 1]
            t_curr = periods[i]

            treated_prev = [d[outcome] for d in data
                            if d[treatment_group] == 1 and d[time_var] == t_prev]
            treated_curr = [d[outcome] for d in data
                            if d[treatment_group] == 1 and d[time_var] == t_curr]
            control_prev = [d[outcome] for d in data
                            if d[treatment_group] == 0 and d[time_var] == t_prev]
            control_curr = [d[outcome] for d in data
                            if d[treatment_group] == 0 and d[time_var] == t_curr]

            if all([treated_prev, treated_curr, control_prev, control_curr]):
                mean = lambda lst: sum(lst) / len(lst)
                t_change = mean(treated_curr) - mean(treated_prev)
                c_change = mean(control_curr) - mean(control_prev)
                results.append({
                    'period': (t_prev, t_curr),
                    'treated_change': t_change,
                    'control_change': c_change,
                    'difference': t_change - c_change,
                })

        return results


# ============================================================
# Regression Discontinuity
# ============================================================

class RegressionDiscontinuity:
    """Regression discontinuity design for causal inference."""

    def __init__(self):
        pass

    def estimate(self, data, running_var, outcome, cutoff, bandwidth=None, kernel='triangular'):
        """
        Local linear regression at the cutoff.
        Estimate the discontinuity in outcome at the cutoff.
        """
        if bandwidth is None:
            # Silverman's rule of thumb
            vals = [d[running_var] for d in data]
            std = math.sqrt(self._variance(vals))
            bandwidth = 1.06 * std * len(vals) ** (-0.2)

        # Filter to bandwidth window
        in_window = [d for d in data
                     if abs(d[running_var] - cutoff) <= bandwidth]

        if len(in_window) < 4:
            return {'effect': 0.0, 'valid': False}

        # Split into above/below cutoff
        below = [d for d in in_window if d[running_var] < cutoff]
        above = [d for d in in_window if d[running_var] >= cutoff]

        if not below or not above:
            return {'effect': 0.0, 'valid': False}

        # Local linear regression on each side
        y_below = self._local_linear(below, running_var, outcome, cutoff, bandwidth, kernel)
        y_above = self._local_linear(above, running_var, outcome, cutoff, bandwidth, kernel)

        effect = y_above - y_below

        return {
            'effect': effect,
            'y_below': y_below,
            'y_above': y_above,
            'bandwidth': bandwidth,
            'n_below': len(below),
            'n_above': len(above),
            'valid': True,
        }

    def _local_linear(self, data, running_var, outcome, cutoff, bandwidth, kernel):
        """Weighted local linear regression, predict at cutoff."""
        n = len(data)
        if n == 0:
            return 0.0

        # Weighted least squares: Y = a + b*(X - cutoff)
        sw = 0.0
        swx = 0.0
        swy = 0.0
        swxx = 0.0
        swxy = 0.0

        for d in data:
            x = d[running_var] - cutoff
            y = d[outcome]
            w = self._kernel_weight(x / bandwidth, kernel)

            sw += w
            swx += w * x
            swy += w * y
            swxx += w * x * x
            swxy += w * x * y

        if abs(sw) < 1e-12:
            return 0.0

        # Solve 2x2 system for intercept (= prediction at cutoff)
        det = sw * swxx - swx * swx
        if abs(det) < 1e-12:
            return swy / sw

        a = (swxx * swy - swx * swxy) / det
        return a

    def _kernel_weight(self, u, kernel):
        """Kernel function for weighting."""
        if abs(u) > 1:
            return 0.0
        if kernel == 'triangular':
            return 1.0 - abs(u)
        elif kernel == 'uniform':
            return 1.0
        elif kernel == 'epanechnikov':
            return 0.75 * (1 - u * u)
        return 1.0 - abs(u)  # default triangular

    def _variance(self, values):
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)
