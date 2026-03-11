"""
C163: Structural Equation Model
Extending C161 (Causal Inference) with explicit structural equations.

Components:
- StructuralEquation: f(parents, noise) mapping for a single variable
- LinearSEM: Linear structural equation model (B matrix + noise)
- NonlinearSEM: Arbitrary function-based structural equations
- SEMIdentification: Instrumental variable identification, Wright's path tracing
- SEMIntervention: do(X=x) via equation replacement (not just graph surgery)
- SEMCounterfactual: Abduction-action-prediction with structural equations
- SEMEstimation: OLS, 2SLS, path coefficients from data
- SEMFitMetrics: Model fit assessment (R-squared per equation, overall)
- SEMSimulator: Monte Carlo simulation from SEM specifications
- SEMAnalyzer: Total/direct/indirect effects, mediation decomposition
"""

import math
import random
from collections import defaultdict, deque
from copy import deepcopy
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C161_causal_inference'))
from causal_inference import CausalGraph


# ---------------------------------------------------------------------------
# StructuralEquation
# ---------------------------------------------------------------------------

class StructuralEquation:
    """A single structural equation: X_i = f(parents, noise).

    Can be linear (coefficients + intercept + noise) or arbitrary function.
    """

    def __init__(self, variable, parents=None, func=None,
                 coefficients=None, intercept=0.0, noise_std=1.0):
        """
        Args:
            variable: Name of the variable this equation defines.
            parents: List of parent variable names.
            func: Optional callable f(parent_values_dict, noise) -> value.
                  If None, uses linear form: intercept + sum(coeff_i * parent_i) + noise.
            coefficients: Dict {parent_name: coefficient} for linear form.
            intercept: Intercept term for linear form.
            noise_std: Standard deviation of Gaussian noise term.
        """
        self.variable = variable
        self.parents = list(parents) if parents else []
        self.func = func
        self.coefficients = dict(coefficients) if coefficients else {}
        self.intercept = float(intercept)
        self.noise_std = float(noise_std)
        self.is_linear = func is None

    def evaluate(self, parent_values, noise=None):
        """Evaluate the structural equation.

        Args:
            parent_values: Dict {parent_name: value}.
            noise: Noise value. If None, sampled from N(0, noise_std).
        Returns:
            Computed value for this variable.
        """
        if noise is None:
            noise = random.gauss(0, self.noise_std) if self.noise_std > 0 else 0.0

        if self.func is not None:
            return self.func(parent_values, noise)

        # Linear form
        value = self.intercept
        for p in self.parents:
            coeff = self.coefficients.get(p, 0.0)
            value += coeff * parent_values.get(p, 0.0)
        value += noise
        return value

    def __repr__(self):
        if self.is_linear:
            terms = []
            if self.intercept != 0:
                terms.append(f"{self.intercept:.2f}")
            for p in self.parents:
                c = self.coefficients.get(p, 0.0)
                terms.append(f"{c:.2f}*{p}")
            terms.append(f"N(0,{self.noise_std:.2f})")
            return f"{self.variable} = {' + '.join(terms)}"
        return f"{self.variable} = f({', '.join(self.parents)}) + noise"


# ---------------------------------------------------------------------------
# LinearSEM
# ---------------------------------------------------------------------------

class LinearSEM:
    """Linear Structural Equation Model.

    X = BX + e, where B is the path coefficient matrix and e ~ N(0, Sigma).
    Supports:
    - Adding variables and equations
    - Intervention via equation replacement
    - Sampling / simulation
    - Covariance computation (implied covariance)
    - Total, direct, and indirect effects
    """

    def __init__(self):
        self.variables = []
        self.equations = {}  # var -> StructuralEquation
        self._var_index = {}

    def add_variable(self, name, noise_std=1.0):
        """Add an exogenous variable (no parents)."""
        if name not in self._var_index:
            self._var_index[name] = len(self.variables)
            self.variables.append(name)
            self.equations[name] = StructuralEquation(
                variable=name, parents=[], coefficients={},
                intercept=0.0, noise_std=noise_std
            )

    def add_equation(self, variable, parents, coefficients, intercept=0.0, noise_std=1.0):
        """Add/replace a structural equation for a variable.

        Args:
            variable: Variable name.
            parents: List of parent names.
            coefficients: Dict or list of coefficients for parents.
            intercept: Intercept term.
            noise_std: Noise standard deviation.
        """
        if variable not in self._var_index:
            self._var_index[variable] = len(self.variables)
            self.variables.append(variable)

        if isinstance(coefficients, (list, tuple)):
            coefficients = {p: c for p, c in zip(parents, coefficients)}

        self.equations[variable] = StructuralEquation(
            variable=variable, parents=parents, coefficients=coefficients,
            intercept=intercept, noise_std=noise_std
        )

    def get_path_matrix(self):
        """Return the B matrix (path coefficients).

        B[i,j] = coefficient of variable j in equation for variable i.
        So X = BX + e.
        """
        n = len(self.variables)
        B = np.zeros((n, n))
        for var in self.variables:
            eq = self.equations[var]
            i = self._var_index[var]
            for p, c in eq.coefficients.items():
                if p in self._var_index:
                    j = self._var_index[p]
                    B[i, j] = c
        return B

    def get_noise_covariance(self):
        """Return the noise covariance matrix (diagonal)."""
        n = len(self.variables)
        Sigma = np.zeros((n, n))
        for var in self.variables:
            eq = self.equations[var]
            i = self._var_index[var]
            Sigma[i, i] = eq.noise_std ** 2
        return Sigma

    def implied_covariance(self):
        """Compute the implied covariance matrix: Sigma = (I-B)^{-1} Omega ((I-B)^{-1})^T.

        Where B is the path matrix and Omega is the noise covariance.
        """
        n = len(self.variables)
        B = self.get_path_matrix()
        Omega = self.get_noise_covariance()
        I = np.eye(n)
        IB_inv = np.linalg.inv(I - B)
        return IB_inv @ Omega @ IB_inv.T

    def topological_order(self):
        """Return variables in topological order."""
        in_degree = defaultdict(int)
        children = defaultdict(list)
        for var in self.variables:
            in_degree[var] = 0
        for var in self.variables:
            for p in self.equations[var].parents:
                if p in self._var_index:
                    children[p].append(var)
                    in_degree[var] += 1

        queue = deque([v for v in self.variables if in_degree[v] == 0])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        return order

    def sample(self, n_samples=1, interventions=None, noise_values=None):
        """Sample from the SEM.

        Args:
            n_samples: Number of samples to generate.
            interventions: Dict {var: value} for do(var=value).
            noise_values: Optional dict {var: array of noise values}.
        Returns:
            Dict {var: np.array of values}.
        """
        interventions = interventions or {}
        order = self.topological_order()
        data = {v: np.zeros(n_samples) for v in self.variables}

        for var in order:
            if var in interventions:
                data[var] = np.full(n_samples, interventions[var])
                continue

            eq = self.equations[var]
            if noise_values and var in noise_values:
                noise = np.array(noise_values[var])
            else:
                noise = np.random.normal(0, eq.noise_std, n_samples) if eq.noise_std > 0 else np.zeros(n_samples)

            values = np.full(n_samples, eq.intercept)
            for p in eq.parents:
                if p in self._var_index:
                    c = eq.coefficients.get(p, 0.0)
                    values = values + c * data[p]
            values = values + noise
            data[var] = values

        return data

    def total_effect(self, cause, effect):
        """Compute total causal effect of cause on effect.

        Uses (I-B)^{-1} matrix: total effect is element [effect_idx, cause_idx].
        """
        B = self.get_path_matrix()
        I = np.eye(len(self.variables))
        total = np.linalg.inv(I - B)
        i = self._var_index[effect]
        j = self._var_index[cause]
        return total[i, j]

    def direct_effect(self, cause, effect):
        """Return the direct causal effect (path coefficient)."""
        eq = self.equations.get(effect)
        if eq is None:
            return 0.0
        return eq.coefficients.get(cause, 0.0)

    def indirect_effect(self, cause, effect):
        """Compute indirect effect = total - direct."""
        return self.total_effect(cause, effect) - self.direct_effect(cause, effect)

    def to_causal_graph(self):
        """Convert to a CausalGraph."""
        g = CausalGraph()
        for var in self.variables:
            g.add_node(var)
        for var in self.variables:
            for p in self.equations[var].parents:
                if p in self._var_index:
                    g.add_edge(p, var)
        return g


# ---------------------------------------------------------------------------
# NonlinearSEM
# ---------------------------------------------------------------------------

class NonlinearSEM:
    """Nonlinear Structural Equation Model.

    Each variable has an arbitrary function f(parents, noise).
    """

    def __init__(self):
        self.variables = []
        self.equations = {}
        self._var_set = set()

    def add_equation(self, variable, parents=None, func=None, noise_std=1.0):
        """Add a structural equation.

        Args:
            variable: Variable name.
            parents: List of parent variable names.
            func: Callable f(parent_values_dict, noise) -> value.
                  If None, variable is exogenous (just noise).
            noise_std: Noise standard deviation.
        """
        parents = list(parents) if parents else []
        if variable not in self._var_set:
            self.variables.append(variable)
            self._var_set.add(variable)

        if func is None:
            func = lambda pv, n: n

        self.equations[variable] = StructuralEquation(
            variable=variable, parents=parents, func=func, noise_std=noise_std
        )

    def topological_order(self):
        """Return variables in topological order."""
        in_degree = defaultdict(int)
        children = defaultdict(list)
        for var in self.variables:
            in_degree[var] = 0
        for var in self.variables:
            eq = self.equations.get(var)
            if eq:
                for p in eq.parents:
                    if p in self._var_set:
                        children[p].append(var)
                        in_degree[var] += 1

        queue = deque([v for v in self.variables if in_degree[v] == 0])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for c in children.get(v, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        return order

    def sample(self, n_samples=1, interventions=None, noise_values=None):
        """Sample from the nonlinear SEM."""
        interventions = interventions or {}
        order = self.topological_order()
        data = {v: np.zeros(n_samples) for v in self.variables}

        for var in order:
            if var in interventions:
                val = interventions[var]
                if callable(val):
                    data[var] = np.array([val() for _ in range(n_samples)])
                else:
                    data[var] = np.full(n_samples, val)
                continue

            eq = self.equations[var]
            for i in range(n_samples):
                parent_vals = {p: data[p][i] for p in eq.parents}
                if noise_values and var in noise_values:
                    noise = noise_values[var][i] if hasattr(noise_values[var], '__getitem__') else noise_values[var]
                else:
                    noise = random.gauss(0, eq.noise_std) if eq.noise_std > 0 else 0.0
                data[var][i] = eq.evaluate(parent_vals, noise)

        return data

    def counterfactual(self, evidence, intervention, target, noise_values=None):
        """Compute counterfactual: given evidence, what would target be under intervention?

        Steps (abduction-action-prediction):
        1. Abduction: infer noise values consistent with evidence.
        2. Action: replace equations for intervened variables.
        3. Prediction: compute target under modified model with inferred noise.
        """
        # Step 1: Abduction -- for exogenous vars and vars with evidence,
        # back-compute noise. For linear equations this is exact; for nonlinear
        # we use the provided noise_values or assume zero noise for unobserved.
        order = self.topological_order()
        inferred_noise = {}
        values = dict(evidence)

        # Forward pass to fill in values using evidence
        for var in order:
            if var in evidence:
                # Back-compute noise
                eq = self.equations[var]
                parent_vals = {p: values.get(p, 0.0) for p in eq.parents}
                if eq.is_linear:
                    predicted = eq.intercept
                    for p in eq.parents:
                        predicted += eq.coefficients.get(p, 0.0) * parent_vals.get(p, 0.0)
                    inferred_noise[var] = evidence[var] - predicted
                else:
                    # For nonlinear, use provided noise or assume 0
                    if noise_values and var in noise_values:
                        inferred_noise[var] = noise_values[var]
                    else:
                        inferred_noise[var] = 0.0
            else:
                # Forward compute
                eq = self.equations[var]
                parent_vals = {p: values.get(p, 0.0) for p in eq.parents}
                n = inferred_noise.get(var, 0.0)
                if noise_values and var in noise_values:
                    n = noise_values[var]
                values[var] = eq.evaluate(parent_vals, n)
                inferred_noise[var] = n

        # Step 2 & 3: Action + Prediction -- re-evaluate with interventions and inferred noise
        cf_values = {}
        for var in order:
            if var in intervention:
                cf_values[var] = intervention[var]
            else:
                eq = self.equations[var]
                parent_vals = {p: cf_values.get(p, 0.0) for p in eq.parents}
                n = inferred_noise.get(var, 0.0)
                cf_values[var] = eq.evaluate(parent_vals, n)

        if target in cf_values:
            return cf_values[target]
        return cf_values


# ---------------------------------------------------------------------------
# SEMIdentification
# ---------------------------------------------------------------------------

class SEMIdentification:
    """Identification analysis for structural equation models.

    Methods:
    - is_identified: Check if a causal effect is identified.
    - instrumental_variables: Find valid instruments.
    - wright_path_tracing: Compute effects via Wright's path tracing rules.
    - all_directed_paths: Enumerate all directed paths between two nodes.
    """

    def __init__(self, sem):
        """
        Args:
            sem: A LinearSEM instance.
        """
        self.sem = sem

    def _build_adjacency(self):
        """Build adjacency from SEM equations."""
        adj = defaultdict(list)
        for var in self.sem.variables:
            eq = self.sem.equations[var]
            for p in eq.parents:
                if p in self.sem._var_index:
                    adj[p].append(var)
        return adj

    def all_directed_paths(self, source, target, max_length=None):
        """Find all directed paths from source to target."""
        adj = self._build_adjacency()
        paths = []
        stack = [(source, [source])]
        while stack:
            node, path = stack.pop()
            if max_length and len(path) > max_length + 1:
                continue
            if node == target and len(path) > 1:
                paths.append(list(path))
                continue
            for child in adj.get(node, []):
                if child not in path:  # no cycles
                    stack.append((child, path + [child]))
        return paths

    def wright_path_tracing(self, cause, effect):
        """Compute total effect using Wright's path tracing rules.

        For linear SEMs, the total effect is the sum over all directed paths
        of the product of edge coefficients along each path.
        """
        paths = self.all_directed_paths(cause, effect)
        total = 0.0
        for path in paths:
            product = 1.0
            for i in range(len(path) - 1):
                parent = path[i]
                child = path[i + 1]
                eq = self.sem.equations[child]
                coeff = eq.coefficients.get(parent, 0.0)
                product *= coeff
            total += product
        return total

    def _ancestors(self, node):
        """Get ancestors of a node in the SEM."""
        visited = set()
        queue = deque()
        eq = self.sem.equations.get(node)
        if eq:
            queue.extend(eq.parents)
        while queue:
            n = queue.popleft()
            if n not in visited and n in self.sem._var_index:
                visited.add(n)
                eq2 = self.sem.equations.get(n)
                if eq2:
                    queue.extend(eq2.parents)
        return visited

    def _descendants(self, node):
        """Get descendants of a node in the SEM."""
        adj = self._build_adjacency()
        visited = set()
        queue = deque(adj.get(node, []))
        while queue:
            n = queue.popleft()
            if n not in visited:
                visited.add(n)
                queue.extend(adj.get(n, []))
        return visited

    def is_identified(self, cause, effect, observed=None):
        """Check if the causal effect of cause on effect is identified.

        Uses backdoor criterion: effect is identified if there exists a set
        of observed variables that blocks all backdoor paths.

        Args:
            cause: Treatment variable.
            effect: Outcome variable.
            observed: Set of observed variables. If None, all variables are observed.
        Returns:
            (bool, set or None): (is_identified, adjustment_set or None).
        """
        if observed is None:
            observed = set(self.sem.variables)
        else:
            observed = set(observed)

        # Try all subsets of observed \ {cause, effect} as adjustment sets
        candidates = list(observed - {cause, effect})
        # Try empty set first, then single variables, then pairs...
        for size in range(len(candidates) + 1):
            for subset in self._combinations(candidates, size):
                s = set(subset)
                if self._is_valid_backdoor(cause, effect, s):
                    return True, s
        return False, None

    def _is_valid_backdoor(self, cause, effect, adjustment_set):
        """Check if adjustment_set satisfies the backdoor criterion."""
        # 1. No node in Z is a descendant of cause
        desc_cause = self._descendants(cause)
        for z in adjustment_set:
            if z in desc_cause:
                return False

        # 2. Z blocks all backdoor paths from cause to effect
        # A backdoor path is a path that starts with an arrow INTO cause
        # Check d-separation of cause and effect given Z in the mutilated graph
        # (remove all arrows out of cause)
        return self._blocks_backdoor_paths(cause, effect, adjustment_set)

    def _blocks_backdoor_paths(self, cause, effect, z_set):
        """Check if z_set blocks all non-causal paths between cause and effect."""
        # Build the graph structure
        adj = self._build_adjacency()
        parents_of = defaultdict(list)
        for var in self.sem.variables:
            eq = self.sem.equations[var]
            for p in eq.parents:
                if p in self.sem._var_index:
                    parents_of[var].append(p)

        # Use d-separation check: BFS over active paths
        # A path is active if it's not blocked by z_set
        # We check in the "moral" sense using Bayes Ball algorithm
        visited = set()
        queue = deque()

        # Start from cause, going "up" (through parents)
        for p in parents_of.get(cause, []):
            queue.append((p, 'up'))

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node == effect:
                return False  # Found an active path

            if direction == 'up':
                if node not in z_set:
                    # Continue up through parents
                    for p in parents_of.get(node, []):
                        queue.append((p, 'up'))
                    # Continue down through children
                    for c in adj.get(node, []):
                        if c != cause:
                            queue.append((c, 'down'))
                # If node in z_set and going up, path is blocked (chain/fork)
            else:  # direction == 'down'
                if node not in z_set:
                    # Continue down
                    for c in adj.get(node, []):
                        queue.append((c, 'down'))
                # Collider: if node or descendant is in z_set, path becomes active
                # (handled by going up when conditioned)
                if node in z_set:
                    for p in parents_of.get(node, []):
                        queue.append((p, 'up'))

        return True  # No active path found

    def instrumental_variables(self, cause, effect, observed=None):
        """Find valid instrumental variables for the cause->effect relationship.

        An instrument Z is valid if:
        1. Z is associated with cause (relevance).
        2. Z affects effect only through cause (exclusion).
        3. Z is not confounded with effect (independence).

        Returns list of valid instrument names.
        """
        if observed is None:
            observed = set(self.sem.variables)

        adj = self._build_adjacency()
        instruments = []

        for z in self.sem.variables:
            if z == cause or z == effect:
                continue
            if z not in observed:
                continue

            # Check relevance: z must have a directed path to cause
            paths_to_cause = self.all_directed_paths(z, cause)
            if not paths_to_cause:
                continue

            # Check exclusion: z must not have a direct path to effect
            # that doesn't go through cause
            paths_to_effect = self.all_directed_paths(z, effect)
            all_through_cause = True
            for path in paths_to_effect:
                if cause not in path[1:-1] and cause != path[-1]:
                    # Path doesn't go through cause
                    # Unless the path IS z -> ... -> cause -> ... -> effect
                    if cause not in path:
                        all_through_cause = False
                        break
            if not all_through_cause:
                continue

            # Check: z should not be a descendant of cause
            desc_cause = self._descendants(cause)
            if z in desc_cause:
                continue

            instruments.append(z)

        return instruments

    @staticmethod
    def _combinations(lst, size):
        """Generate all combinations of given size."""
        if size == 0:
            yield []
            return
        for i in range(len(lst)):
            for rest in SEMIdentification._combinations(lst[i+1:], size - 1):
                yield [lst[i]] + rest


# ---------------------------------------------------------------------------
# SEMIntervention
# ---------------------------------------------------------------------------

class SEMIntervention:
    """Intervention operations on structural equation models.

    Implements do(X=x) by replacing structural equations,
    not just graph surgery.
    """

    @staticmethod
    def do(sem, interventions):
        """Create a new SEM with interventions applied.

        Args:
            sem: A LinearSEM or NonlinearSEM.
            interventions: Dict {var: value} for do(var=value).
        Returns:
            A new SEM with intervened equations replaced by constants.
        """
        if isinstance(sem, LinearSEM):
            return SEMIntervention._do_linear(sem, interventions)
        else:
            return SEMIntervention._do_nonlinear(sem, interventions)

    @staticmethod
    def _do_linear(sem, interventions):
        """Apply interventions to a LinearSEM."""
        new_sem = LinearSEM()
        for var in sem.variables:
            if var in interventions:
                new_sem.add_variable(var, noise_std=0.0)
                new_sem.add_equation(var, [], {}, intercept=interventions[var], noise_std=0.0)
            else:
                eq = sem.equations[var]
                # Filter out intervened parents (they're now constants)
                new_sem.add_equation(
                    var, eq.parents, dict(eq.coefficients),
                    eq.intercept, eq.noise_std
                )
        return new_sem

    @staticmethod
    def _do_nonlinear(sem, interventions):
        """Apply interventions to a NonlinearSEM."""
        new_sem = NonlinearSEM()
        for var in sem.variables:
            if var in interventions:
                val = interventions[var]
                new_sem.add_equation(var, [], func=lambda pv, n, v=val: v, noise_std=0.0)
            else:
                eq = sem.equations[var]
                new_sem.add_equation(var, eq.parents, eq.func, eq.noise_std)
        return new_sem

    @staticmethod
    def conditional_intervention(sem, interventions, conditions, n_samples=1000):
        """Interventional distribution conditioned on observed variables.

        Compute E[Y | do(X=x), Z=z] by sampling from the intervened model
        and filtering/weighting by conditions.

        Args:
            sem: A LinearSEM or NonlinearSEM.
            interventions: Dict {var: value}.
            conditions: Dict {var: value} -- observed conditions.
            n_samples: Number of samples for estimation.
        Returns:
            Dict {var: mean_value} for non-intervened, non-conditioned variables.
        """
        intervened = SEMIntervention.do(sem, interventions)
        data = intervened.sample(n_samples=n_samples)

        # Filter samples consistent with conditions (within tolerance)
        tolerance = 0.5  # for continuous variables
        mask = np.ones(n_samples, dtype=bool)
        for var, val in conditions.items():
            if var in data:
                mask &= np.abs(data[var] - val) < tolerance

        if not np.any(mask):
            # Relax tolerance
            tolerance = 2.0
            mask = np.ones(n_samples, dtype=bool)
            for var, val in conditions.items():
                if var in data:
                    mask &= np.abs(data[var] - val) < tolerance

        result = {}
        for var in data:
            if var not in interventions and var not in conditions:
                if np.any(mask):
                    result[var] = float(np.mean(data[var][mask]))
                else:
                    result[var] = float(np.mean(data[var]))
        return result


# ---------------------------------------------------------------------------
# SEMCounterfactual
# ---------------------------------------------------------------------------

class SEMCounterfactual:
    """Counterfactual reasoning with structural equation models.

    Implements the three-step procedure:
    1. Abduction: Infer exogenous noise from evidence.
    2. Action: Modify equations (intervention).
    3. Prediction: Compute outcome under modified model with inferred noise.
    """

    @staticmethod
    def query(sem, evidence, intervention, target):
        """Compute a counterfactual query.

        "Given that we observed evidence, what would target be
        if we had set intervention?"

        Args:
            sem: A LinearSEM or NonlinearSEM.
            evidence: Dict {var: observed_value}.
            intervention: Dict {var: intervened_value}.
            target: Variable name to query.
        Returns:
            Counterfactual value of target.
        """
        if isinstance(sem, LinearSEM):
            return SEMCounterfactual._linear_counterfactual(sem, evidence, intervention, target)
        else:
            return sem.counterfactual(evidence, intervention, target)

    @staticmethod
    def _linear_counterfactual(sem, evidence, intervention, target):
        """Exact counterfactual for linear SEMs.

        Step 1: Abduction -- solve for noise terms given evidence.
        Step 2: Action -- replace equations for interventions.
        Step 3: Prediction -- forward-compute with inferred noise.
        """
        order = sem.topological_order()

        # Step 1: Abduction -- compute all values and infer noise
        values = {}
        inferred_noise = {}

        for var in order:
            eq = sem.equations[var]
            predicted = eq.intercept
            for p in eq.parents:
                if p in sem._var_index:
                    predicted += eq.coefficients.get(p, 0.0) * values.get(p, 0.0)

            if var in evidence:
                values[var] = evidence[var]
                inferred_noise[var] = evidence[var] - predicted
            else:
                # Use zero noise for unobserved variables during abduction
                values[var] = predicted
                inferred_noise[var] = 0.0

        # Step 2 & 3: Action + Prediction
        cf_values = {}
        for var in order:
            if var in intervention:
                cf_values[var] = intervention[var]
            else:
                eq = sem.equations[var]
                predicted = eq.intercept
                for p in eq.parents:
                    if p in sem._var_index:
                        predicted += eq.coefficients.get(p, 0.0) * cf_values.get(p, 0.0)
                cf_values[var] = predicted + inferred_noise.get(var, 0.0)

        return cf_values[target]

    @staticmethod
    def effect_of_treatment_on_treated(sem, evidence, cause, effect, treatment_value, control_value):
        """ETT: E[Y(treatment) - Y(control) | X=treatment, evidence].

        Args:
            sem: A LinearSEM.
            evidence: Dict of observed values.
            cause: Treatment variable name.
            effect: Outcome variable name.
            treatment_value: Value under treatment.
            control_value: Value under control.
        Returns:
            ETT estimate.
        """
        # Factual: evidence already contains cause=treatment_value
        full_evidence = dict(evidence)
        full_evidence[cause] = treatment_value

        # Counterfactual: what would effect be if cause=control_value?
        y_control = SEMCounterfactual.query(sem, full_evidence, {cause: control_value}, effect)
        y_treatment = full_evidence.get(effect)
        if y_treatment is None:
            y_treatment = SEMCounterfactual.query(sem, full_evidence, {cause: treatment_value}, effect)

        return y_treatment - y_control

    @staticmethod
    def probability_of_necessity(sem, evidence, cause, effect, n_samples=1000):
        """PN: P(Y(0)=0 | X=1, Y=1) -- Would Y not have occurred without X?

        Approximated via sampling for continuous variables.
        Uses the counterfactual framework.
        """
        # For linear SEMs, compute exactly
        if isinstance(sem, LinearSEM):
            # The counterfactual Y under do(X=0) given evidence
            cf_value = SEMCounterfactual.query(sem, evidence, {cause: 0}, effect)
            # PN is about whether Y would be below threshold
            # For continuous: return the difference
            return evidence.get(effect, 0.0) - cf_value
        return 0.0

    @staticmethod
    def probability_of_sufficiency(sem, evidence, cause, effect, n_samples=1000):
        """PS: P(Y(1)=1 | X=0, Y=0) -- Would Y have occurred if X had?

        For linear SEMs, compute the counterfactual change.
        """
        if isinstance(sem, LinearSEM):
            cf_value = SEMCounterfactual.query(sem, evidence, {cause: 1}, effect)
            return cf_value - evidence.get(effect, 0.0)
        return 0.0


# ---------------------------------------------------------------------------
# SEMEstimation
# ---------------------------------------------------------------------------

class SEMEstimation:
    """Estimate SEM parameters from data.

    Methods:
    - ols: Ordinary Least Squares estimation
    - two_stage_ls: Two-Stage Least Squares (with instruments)
    - path_coefficients: Estimate all path coefficients from data
    """

    @staticmethod
    def ols(data, target, predictors):
        """Ordinary Least Squares regression.

        Args:
            data: Dict {var: np.array}.
            target: Target variable name.
            predictors: List of predictor variable names.
        Returns:
            Dict with 'coefficients' {var: coeff}, 'intercept', 'r_squared'.
        """
        y = np.array(data[target])
        n = len(y)

        if not predictors:
            return {
                'coefficients': {},
                'intercept': float(np.mean(y)),
                'r_squared': 0.0
            }

        X = np.column_stack([np.array(data[p]) for p in predictors])
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n), X])

        # beta = (X'X)^{-1} X'y
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(len(predictors) + 1)

        intercept = beta[0]
        coeffs = {p: float(beta[i+1]) for i, p in enumerate(predictors)}

        # R-squared
        y_pred = X_with_intercept @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'coefficients': coeffs,
            'intercept': float(intercept),
            'r_squared': float(r_squared)
        }

    @staticmethod
    def two_stage_ls(data, target, endogenous, instruments, exogenous=None):
        """Two-Stage Least Squares estimation.

        Args:
            data: Dict {var: np.array}.
            target: Outcome variable.
            endogenous: List of endogenous regressors.
            instruments: List of instrument variable names.
            exogenous: Optional list of exogenous controls.
        Returns:
            Dict with 'coefficients', 'intercept'.
        """
        exogenous = exogenous or []
        y = np.array(data[target])
        n = len(y)

        # Stage 1: Regress endogenous on instruments + exogenous
        first_stage_fitted = {}
        for endo in endogenous:
            Z_cols = [np.array(data[z]) for z in instruments]
            Z_cols += [np.array(data[x]) for x in exogenous]
            if not Z_cols:
                first_stage_fitted[endo] = np.array(data[endo])
                continue
            Z = np.column_stack([np.ones(n)] + Z_cols)
            endo_y = np.array(data[endo])
            try:
                gamma = np.linalg.lstsq(Z, endo_y, rcond=None)[0]
            except np.linalg.LinAlgError:
                gamma = np.zeros(Z.shape[1])
            first_stage_fitted[endo] = Z @ gamma

        # Stage 2: Regress target on fitted endogenous + exogenous
        X_cols = [first_stage_fitted[e] for e in endogenous]
        X_cols += [np.array(data[x]) for x in exogenous]
        X = np.column_stack([np.ones(n)] + X_cols)

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(X.shape[1])

        all_vars = endogenous + exogenous
        coeffs = {v: float(beta[i+1]) for i, v in enumerate(all_vars)}

        return {
            'coefficients': coeffs,
            'intercept': float(beta[0])
        }

    @staticmethod
    def path_coefficients(data, sem):
        """Estimate all path coefficients in a LinearSEM from data.

        Args:
            data: Dict {var: np.array}.
            sem: LinearSEM specifying the graph structure.
        Returns:
            Dict {(parent, child): estimated_coefficient}.
        """
        estimated = {}
        for var in sem.variables:
            eq = sem.equations[var]
            if eq.parents:
                result = SEMEstimation.ols(data, var, eq.parents)
                for p in eq.parents:
                    estimated[(p, var)] = result['coefficients'].get(p, 0.0)
        return estimated


# ---------------------------------------------------------------------------
# SEMFitMetrics
# ---------------------------------------------------------------------------

class SEMFitMetrics:
    """Model fit assessment for structural equation models."""

    @staticmethod
    def equation_r_squared(data, sem):
        """Compute R-squared for each structural equation.

        Args:
            data: Dict {var: np.array}.
            sem: LinearSEM.
        Returns:
            Dict {var: r_squared}.
        """
        results = {}
        for var in sem.variables:
            eq = sem.equations[var]
            if eq.parents:
                result = SEMEstimation.ols(data, var, eq.parents)
                results[var] = result['r_squared']
            else:
                results[var] = 0.0  # Exogenous variable
        return results

    @staticmethod
    def residuals(data, sem):
        """Compute residuals for each equation.

        Args:
            data: Dict {var: np.array}.
            sem: LinearSEM.
        Returns:
            Dict {var: np.array of residuals}.
        """
        residuals = {}
        for var in sem.variables:
            eq = sem.equations[var]
            y = np.array(data[var])
            predicted = np.full(len(y), eq.intercept)
            for p in eq.parents:
                if p in data:
                    predicted = predicted + eq.coefficients.get(p, 0.0) * np.array(data[p])
            residuals[var] = y - predicted
        return residuals

    @staticmethod
    def implied_vs_observed_covariance(data, sem):
        """Compare implied and observed covariance matrices.

        Args:
            data: Dict {var: np.array}.
            sem: LinearSEM.
        Returns:
            Dict with 'implied', 'observed', 'difference', 'rmse'.
        """
        implied = sem.implied_covariance()

        # Observed covariance
        vars_list = sem.variables
        n = len(data[vars_list[0]])
        data_matrix = np.column_stack([np.array(data[v]) for v in vars_list])
        observed = np.cov(data_matrix, rowvar=False, ddof=1)

        diff = implied - observed
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        return {
            'implied': implied,
            'observed': observed,
            'difference': diff,
            'rmse': rmse
        }

    @staticmethod
    def aic_bic(data, sem):
        """Compute AIC and BIC for the SEM.

        Args:
            data: Dict {var: np.array}.
            sem: LinearSEM.
        Returns:
            Dict with 'aic', 'bic', 'log_likelihood', 'n_params'.
        """
        n_obs = len(data[sem.variables[0]])
        resid = SEMFitMetrics.residuals(data, sem)

        # Log-likelihood assuming Gaussian residuals
        log_lik = 0.0
        n_params = 0
        for var in sem.variables:
            eq = sem.equations[var]
            r = resid[var]
            var_resid = np.var(r, ddof=1) if len(r) > 1 else 1.0
            if var_resid <= 0:
                var_resid = 1e-10
            log_lik += -0.5 * n_obs * (np.log(2 * np.pi * var_resid) + 1)
            n_params += len(eq.parents) + 1 + 1  # coeffs + intercept + variance

        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n_obs)

        return {
            'aic': float(aic),
            'bic': float(bic),
            'log_likelihood': float(log_lik),
            'n_params': n_params
        }


# ---------------------------------------------------------------------------
# SEMSimulator
# ---------------------------------------------------------------------------

class SEMSimulator:
    """Monte Carlo simulation from SEM specifications."""

    @staticmethod
    def simulate(sem, n_samples=1000, interventions=None, seed=None):
        """Run simulation from a SEM.

        Args:
            sem: LinearSEM or NonlinearSEM.
            interventions: Optional dict {var: value}.
            n_samples: Number of samples.
            seed: Random seed.
        Returns:
            Dict {var: np.array}.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return sem.sample(n_samples=n_samples, interventions=interventions)

    @staticmethod
    def interventional_distribution(sem, cause, effect, values, n_samples=5000, seed=None):
        """Compute E[effect | do(cause=v)] for multiple values of cause.

        Args:
            sem: LinearSEM or NonlinearSEM.
            cause: Intervention variable.
            effect: Outcome variable.
            values: List/array of intervention values.
            n_samples: Samples per intervention.
            seed: Random seed.
        Returns:
            Dict with 'values' and 'expectations'.
        """
        expectations = []
        for v in values:
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
            data = sem.sample(n_samples=n_samples, interventions={cause: v})
            expectations.append(float(np.mean(data[effect])))
        return {
            'values': list(values),
            'expectations': expectations
        }

    @staticmethod
    def sensitivity_analysis(sem, cause, effect, noise_multipliers, n_samples=5000, seed=None):
        """Assess sensitivity of causal effect to noise levels.

        Args:
            sem: LinearSEM.
            cause, effect: Variable names.
            noise_multipliers: List of multipliers for noise std.
            n_samples: Samples per setting.
        Returns:
            Dict with 'multipliers', 'effects', 'variances'.
        """
        effects = []
        variances = []
        original_stds = {v: sem.equations[v].noise_std for v in sem.variables}

        for mult in noise_multipliers:
            # Modify noise levels
            modified_sem = LinearSEM()
            for var in sem.variables:
                eq = sem.equations[var]
                modified_sem.add_equation(
                    var, eq.parents, dict(eq.coefficients),
                    eq.intercept, eq.noise_std * mult
                )

            # Compute effect
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)

            # Sample with and without intervention
            data_do1 = modified_sem.sample(n_samples, interventions={cause: 1})
            data_do0 = modified_sem.sample(n_samples, interventions={cause: 0})
            effect_est = float(np.mean(data_do1[effect]) - np.mean(data_do0[effect]))
            var_est = float(np.var(data_do1[effect]) + np.var(data_do0[effect]))
            effects.append(effect_est)
            variances.append(var_est)

        return {
            'multipliers': list(noise_multipliers),
            'effects': effects,
            'variances': variances
        }


# ---------------------------------------------------------------------------
# SEMAnalyzer
# ---------------------------------------------------------------------------

class SEMAnalyzer:
    """High-level analysis combining SEM components."""

    def __init__(self, sem):
        self.sem = sem
        self.identification = SEMIdentification(sem) if isinstance(sem, LinearSEM) else None

    def decompose_effect(self, cause, effect):
        """Decompose total effect into direct + indirect.

        Args:
            cause, effect: Variable names.
        Returns:
            Dict with 'total', 'direct', 'indirect', 'paths'.
        """
        if not isinstance(self.sem, LinearSEM):
            raise ValueError("Effect decomposition requires LinearSEM")

        total = self.sem.total_effect(cause, effect)
        direct = self.sem.direct_effect(cause, effect)
        indirect = total - direct

        # Enumerate paths
        ident = SEMIdentification(self.sem)
        paths = ident.all_directed_paths(cause, effect)
        path_effects = []
        for path in paths:
            product = 1.0
            for i in range(len(path) - 1):
                eq = self.sem.equations[path[i+1]]
                product *= eq.coefficients.get(path[i], 0.0)
            path_effects.append({
                'path': path,
                'effect': product,
                'is_direct': len(path) == 2
            })

        return {
            'total': total,
            'direct': direct,
            'indirect': indirect,
            'paths': path_effects
        }

    def mediation_analysis(self, cause, mediator, effect):
        """Analyze mediation: cause -> mediator -> effect.

        Returns:
            Dict with 'total', 'direct', 'indirect_via_mediator', 'proportion_mediated'.
        """
        if not isinstance(self.sem, LinearSEM):
            raise ValueError("Mediation analysis requires LinearSEM")

        total = self.sem.total_effect(cause, effect)
        direct = self.sem.direct_effect(cause, effect)

        # Indirect through mediator specifically
        a = self.sem.direct_effect(cause, mediator)
        b = self.sem.direct_effect(mediator, effect)
        indirect_med = a * b

        proportion = indirect_med / total if abs(total) > 1e-10 else 0.0

        return {
            'total': total,
            'direct': direct,
            'indirect_via_mediator': indirect_med,
            'a_path': a,
            'b_path': b,
            'proportion_mediated': proportion
        }

    def full_report(self, data=None, n_samples=5000, seed=42):
        """Generate a full analysis report.

        Args:
            data: Optional observed data dict. If None, generates synthetic.
            n_samples: Samples for synthetic data.
            seed: Random seed.
        Returns:
            Dict with comprehensive analysis results.
        """
        if not isinstance(self.sem, LinearSEM):
            raise ValueError("Full report requires LinearSEM")

        report = {
            'variables': list(self.sem.variables),
            'n_equations': len(self.sem.equations),
        }

        # Path matrix
        B = self.sem.get_path_matrix()
        report['path_matrix'] = B.tolist()

        # Generate or use data
        if data is None:
            np.random.seed(seed)
            random.seed(seed)
            data = self.sem.sample(n_samples)

        # Fit metrics
        report['r_squared'] = SEMFitMetrics.equation_r_squared(data, self.sem)
        report['aic_bic'] = SEMFitMetrics.aic_bic(data, self.sem)

        cov_analysis = SEMFitMetrics.implied_vs_observed_covariance(data, self.sem)
        report['covariance_rmse'] = cov_analysis['rmse']

        # Estimation accuracy
        estimated = SEMEstimation.path_coefficients(data, self.sem)
        report['estimated_coefficients'] = {f"{k[0]}->{k[1]}": v for k, v in estimated.items()}

        # True coefficients for comparison
        true_coeffs = {}
        for var in self.sem.variables:
            eq = self.sem.equations[var]
            for p, c in eq.coefficients.items():
                true_coeffs[f"{p}->{var}"] = c
        report['true_coefficients'] = true_coeffs

        return report

    def causal_query(self, query_type, **kwargs):
        """Unified interface for causal queries.

        Args:
            query_type: One of 'ate', 'intervention', 'counterfactual', 'mediation'.
            **kwargs: Query-specific parameters.
        Returns:
            Query result.
        """
        if query_type == 'ate':
            cause = kwargs['cause']
            effect = kwargs['effect']
            return self.sem.total_effect(cause, effect)

        elif query_type == 'intervention':
            interventions = kwargs['interventions']
            target = kwargs['target']
            n_samples = kwargs.get('n_samples', 5000)
            data = self.sem.sample(n_samples, interventions=interventions)
            return float(np.mean(data[target]))

        elif query_type == 'counterfactual':
            evidence = kwargs['evidence']
            intervention = kwargs['intervention']
            target = kwargs['target']
            return SEMCounterfactual.query(self.sem, evidence, intervention, target)

        elif query_type == 'mediation':
            return self.mediation_analysis(kwargs['cause'], kwargs['mediator'], kwargs['effect'])

        else:
            raise ValueError(f"Unknown query type: {query_type}")
