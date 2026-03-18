"""V211: Causal Inference.

Causal reasoning over Bayesian networks using Pearl's do-calculus framework.

Composes V209 (Bayesian Networks) for probabilistic inference.

Supports:
1. Interventions (do-operator): P(Y | do(X=x))
   - Graph surgery: remove incoming edges to intervened node, fix value
2. Backdoor criterion: identify adjustment sets for causal effect estimation
3. Frontdoor criterion: estimate causal effects through mediators
4. Counterfactuals: P(Y_x | evidence) via twin-network construction
5. Instrumental variables: identify and use instruments for causal estimation
6. Causal effect estimation: ATE, CATE, direct/indirect effects
7. do-calculus rules (3 rules of Pearl) for identifiability
"""

from __future__ import annotations
from collections import defaultdict
from itertools import product as iter_product
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))
from bayesian_networks import Factor, BayesianNetwork, variable_elimination


# ---------------------------------------------------------------------------
# CausalModel: BN + interventions + counterfactuals
# ---------------------------------------------------------------------------

class CausalModel:
    """A causal model wrapping a Bayesian network with do-calculus support.

    The causal model adds:
    - Interventions (do-operator) via graph surgery
    - Backdoor/frontdoor criterion checking
    - Counterfactual reasoning via twin networks
    - Causal effect estimation
    """

    def __init__(self, bn: BayesianNetwork):
        self.bn = bn

    # -- Graph utilities ------------------------------------------------

    def _ancestors(self, nodes: set[str], graph_parents: dict[str, list[str]]) -> set[str]:
        """All ancestors of a set of nodes in the given graph."""
        result = set()
        stack = []
        for n in nodes:
            stack.extend(graph_parents.get(n, []))
        while stack:
            node = stack.pop()
            if node not in result:
                result.add(node)
                stack.extend(graph_parents.get(node, []))
        return result

    def _descendants(self, nodes: set[str]) -> set[str]:
        """All descendants of a set of nodes."""
        result = set()
        stack = []
        for n in nodes:
            stack.extend(self.bn.children.get(n, []))
        while stack:
            node = stack.pop()
            if node not in result:
                result.add(node)
                stack.extend(self.bn.children.get(node, []))
        return result

    def _reachable_undirected(self, start: str, removed: set[str],
                               graph_parents: dict[str, list[str]],
                               graph_children: dict[str, list[str]]) -> set[str]:
        """Nodes reachable from start via undirected edges, ignoring removed nodes."""
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited or node in removed:
                continue
            visited.add(node)
            for p in graph_parents.get(node, []):
                if p not in visited and p not in removed:
                    stack.append(p)
            for c in graph_children.get(node, []):
                if c not in visited and c not in removed:
                    stack.append(c)
        return visited

    # -- D-separation (Bayes Ball) --------------------------------------

    def d_separated(self, x: set[str], y: set[str], z: set[str]) -> bool:
        """Test if X _||_ Y | Z using the Bayes Ball algorithm.

        Returns True if X and Y are d-separated given Z.
        """
        # Bayes Ball: find nodes reachable from X considering Z as observed
        reachable = set()
        # Ancestor set of Z (needed for v-structure activation)
        z_ancestors = self._ancestors(z, self.bn.parents)
        z_and_ancestors = z | z_ancestors

        # Visit queue: (node, direction) where direction is 'up' or 'down'
        visited = set()
        queue = []
        for x_node in x:
            queue.append((x_node, 'up'))
            queue.append((x_node, 'down'))

        while queue:
            node, direction = queue.pop(0)
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            reachable.add(node)

            # Rules for Bayes Ball
            if direction == 'up' and node not in z:
                # Passing through non-observed node upward
                for parent in self.bn.parents.get(node, []):
                    queue.append((parent, 'up'))
                for child in self.bn.children.get(node, []):
                    queue.append((child, 'down'))
            elif direction == 'down':
                if node not in z:
                    # Non-observed: pass down to children
                    for child in self.bn.children.get(node, []):
                        queue.append((child, 'down'))
                if node in z_and_ancestors:
                    # Observed or ancestor of observed: v-structure active
                    for parent in self.bn.parents.get(node, []):
                        queue.append((parent, 'up'))

        return len(y & reachable) == 0

    # -- Intervention (do-operator) -------------------------------------

    def do(self, interventions: dict[str, object]) -> BayesianNetwork:
        """Apply do-operator: returns mutilated graph with interventions.

        do(X=x) means:
        1. Remove all edges into X
        2. Set X to value x with probability 1

        Args:
            interventions: dict mapping variable name to forced value

        Returns:
            A new BayesianNetwork with the intervention applied.
        """
        mutilated = BayesianNetwork()

        # Add all nodes
        for node in self.bn.nodes:
            mutilated.add_node(node, self.bn.domains[node])

        # Add edges, skipping incoming edges to intervened nodes
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if node not in interventions:
                    mutilated.add_edge(parent, node)

        # Set CPTs
        for node in self.bn.nodes:
            if node in interventions:
                # Deterministic CPT: P(X=x) = 1, P(X=other) = 0
                val = interventions[node]
                table = {}
                for d_val in self.bn.domains[node]:
                    table[(d_val,)] = 1.0 if d_val == val else 0.0
                mutilated.set_cpt(node, table)
            else:
                # Copy original CPT (but parents might have changed)
                orig_cpt = self.bn.cpts[node]
                mutilated.cpts[node] = Factor(
                    list(orig_cpt.variables),
                    {v: list(orig_cpt.domains[v]) for v in orig_cpt.variables},
                    dict(orig_cpt.table)
                )

        return mutilated

    def interventional_query(self, query: list[str],
                              interventions: dict[str, object],
                              evidence: dict[str, object] | None = None) -> Factor:
        """Compute P(query | do(interventions), evidence).

        Combines do-operator (graph surgery) with standard conditioning.
        """
        mutilated = self.do(interventions)
        return variable_elimination(mutilated, query, evidence)

    # -- Backdoor criterion ---------------------------------------------

    def backdoor_criterion(self, x: str, y: str, z: set[str]) -> bool:
        """Check if Z satisfies the backdoor criterion for (X, Y).

        Z satisfies backdoor criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X
        2. Z blocks every path between X and Y that contains an arrow into X
           (i.e., Z d-separates X and Y in the graph where edges out of X are removed)
        """
        # Condition 1: No descendant of X in Z
        x_descendants = self._descendants({x})
        if z & x_descendants:
            return False

        # Condition 2: Z d-separates X and Y in G_underbar_X
        # (graph with outgoing edges from X removed)
        # Build a temporary model with edges from X removed
        temp_bn = BayesianNetwork()
        for node in self.bn.nodes:
            temp_bn.add_node(node, self.bn.domains[node])
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if parent != x:  # Remove outgoing edges from X
                    temp_bn.add_edge(parent, node)

        temp_model = CausalModel(temp_bn)
        return temp_model.d_separated({x}, {y}, z)

    def find_backdoor_set(self, x: str, y: str) -> set[str] | None:
        """Find a minimal set satisfying the backdoor criterion, or None.

        Tries parent set of X first (common valid adjustment set),
        then searches subsets of non-descendants.
        """
        x_descendants = self._descendants({x})
        candidates = set(self.bn.nodes) - {x, y} - x_descendants

        # Try parents of X first (often sufficient)
        parents_of_x = set(self.bn.parents[x])
        if parents_of_x <= candidates and self.backdoor_criterion(x, y, parents_of_x):
            return parents_of_x

        # Try empty set
        if self.backdoor_criterion(x, y, set()):
            return set()

        # Try single variables
        for c in sorted(candidates):
            if self.backdoor_criterion(x, y, {c}):
                return {c}

        # Try pairs
        cand_list = sorted(candidates)
        for i in range(len(cand_list)):
            for j in range(i + 1, len(cand_list)):
                z = {cand_list[i], cand_list[j]}
                if self.backdoor_criterion(x, y, z):
                    return z

        # Full set of candidates
        if self.backdoor_criterion(x, y, candidates):
            return candidates

        return None

    # -- Frontdoor criterion --------------------------------------------

    def frontdoor_criterion(self, x: str, y: str, m: set[str]) -> bool:
        """Check if M satisfies the frontdoor criterion for (X, Y).

        M satisfies frontdoor criterion relative to (X, Y) if:
        1. X intercepts all directed paths from X to Y (M captures all X->Y paths)
        2. There is no unblocked back-door path from X to M
        3. All back-door paths from M to Y are blocked by X

        Formally:
        1. All directed paths X -> ... -> Y go through M
        2. X d-separates M from X's non-descendants in the mutilated graph
        3. X blocks all backdoor paths from M to Y
        """
        # Condition 1: M intercepts all directed X->Y paths
        if not self._m_intercepts_all_paths(x, y, m):
            return False

        # Condition 2: No backdoor path from X to any node in M
        # (no unblocked path from X to M in G_underbar_X)
        temp_bn = BayesianNetwork()
        for node in self.bn.nodes:
            temp_bn.add_node(node, self.bn.domains[node])
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if parent != x:
                    temp_bn.add_edge(parent, node)
        temp_model = CausalModel(temp_bn)
        if not temp_model.d_separated({x}, m, set()):
            return False

        # Condition 3: All backdoor paths from M to Y are blocked by {X}
        for mi in m:
            if not self.backdoor_criterion(mi, y, {x}):
                return False

        return True

    def _m_intercepts_all_paths(self, x: str, y: str, m: set[str]) -> bool:
        """Check if all directed paths from x to y pass through m."""
        # DFS from X to Y, not passing through M -- if Y reachable, M doesn't intercept
        visited = set()
        stack = [x]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == y:
                return False  # Found path that bypasses M
            if node != x and node in m:
                continue  # M blocks this path
            for child in self.bn.children.get(node, []):
                stack.append(child)
        return True

    # -- Backdoor adjustment formula ------------------------------------

    def backdoor_adjustment(self, x: str, y: str, x_val: object,
                            z: set[str] | None = None) -> dict:
        """Compute P(Y | do(X=x_val)) using backdoor adjustment.

        P(Y=y | do(X=x)) = sum_z P(Y=y | X=x, Z=z) P(Z=z)

        If z is None, automatically finds a valid adjustment set.
        Returns dict mapping y values to probabilities.
        """
        if z is None:
            z = self.find_backdoor_set(x, y)
            if z is None:
                raise ValueError(f"No valid backdoor adjustment set for ({x}, {y})")

        z_list = sorted(z)
        result = {y_val: 0.0 for y_val in self.bn.domains[y]}

        if not z_list:
            # No confounders: P(Y | do(X=x)) = P(Y | X=x)
            evidence = {x: x_val}
            factor = variable_elimination(self.bn, [y], evidence)
            factor = factor.normalize()
            for y_val in self.bn.domains[y]:
                result[y_val] = factor.get({y: y_val})
            return result

        # Enumerate all assignments to Z
        z_domains = [self.bn.domains[v] for v in z_list]
        for z_assignment in iter_product(*z_domains):
            z_dict = dict(zip(z_list, z_assignment))

            # P(Y | X=x, Z=z)
            evidence = {x: x_val}
            evidence.update(z_dict)
            cond_factor = variable_elimination(self.bn, [y], evidence)
            cond_factor = cond_factor.normalize()

            # P(Z=z) -- marginal probability of this Z assignment
            z_factor = variable_elimination(self.bn, z_list)
            z_factor = z_factor.normalize()
            p_z = z_factor.get(z_dict)

            for y_val in self.bn.domains[y]:
                result[y_val] += cond_factor.get({y: y_val}) * p_z

        return result

    # -- Frontdoor adjustment formula -----------------------------------

    def frontdoor_adjustment(self, x: str, y: str, m: str,
                              x_val: object) -> dict:
        """Compute P(Y | do(X=x_val)) using frontdoor adjustment.

        P(Y=y | do(X=x)) = sum_m P(M=m | X=x) sum_x' P(Y=y | X=x', M=m) P(X=x')

        Args:
            x: treatment variable
            y: outcome variable
            m: mediator variable (single)
            x_val: intervention value
        """
        result = {y_val: 0.0 for y_val in self.bn.domains[y]}

        for m_val in self.bn.domains[m]:
            # P(M=m | X=x)
            m_factor = variable_elimination(self.bn, [m], {x: x_val})
            m_factor = m_factor.normalize()
            p_m_given_x = m_factor.get({m: m_val})

            inner_sum = {y_val: 0.0 for y_val in self.bn.domains[y]}
            for x_prime in self.bn.domains[x]:
                # P(Y | X=x', M=m)
                y_factor = variable_elimination(self.bn, [y], {x: x_prime, m: m_val})
                y_factor = y_factor.normalize()
                # P(X=x')
                x_factor = variable_elimination(self.bn, [x])
                x_factor = x_factor.normalize()
                p_x_prime = x_factor.get({x: x_prime})

                for y_val in self.bn.domains[y]:
                    inner_sum[y_val] += y_factor.get({y: y_val}) * p_x_prime

            for y_val in self.bn.domains[y]:
                result[y_val] += p_m_given_x * inner_sum[y_val]

        return result

    # -- Causal effect estimation ---------------------------------------

    def average_treatment_effect(self, x: str, y: str,
                                  x_treat: object, x_control: object) -> float:
        """Compute ATE = E[Y | do(X=x_treat)] - E[Y | do(X=x_control)].

        Assumes Y domain is numeric.
        """
        p_treat = self.interventional_query([y], {x: x_treat}).normalize()
        p_control = self.interventional_query([y], {x: x_control}).normalize()

        e_treat = sum(y_val * p_treat.get({y: y_val}) for y_val in self.bn.domains[y])
        e_control = sum(y_val * p_control.get({y: y_val}) for y_val in self.bn.domains[y])

        return e_treat - e_control

    def controlled_direct_effect(self, x: str, y: str, mediators: set[str],
                                  x_treat: object, x_control: object,
                                  mediator_vals: dict[str, object]) -> float:
        """Compute CDE: E[Y | do(X=x_treat, M=m)] - E[Y | do(X=x_control, M=m)].

        Fixes mediators to specific values to isolate the direct effect.
        """
        interventions_treat = {x: x_treat}
        interventions_treat.update(mediator_vals)
        interventions_control = {x: x_control}
        interventions_control.update(mediator_vals)

        p_treat = self.interventional_query([y], interventions_treat).normalize()
        p_control = self.interventional_query([y], interventions_control).normalize()

        e_treat = sum(y_val * p_treat.get({y: y_val}) for y_val in self.bn.domains[y])
        e_control = sum(y_val * p_control.get({y: y_val}) for y_val in self.bn.domains[y])

        return e_treat - e_control

    def natural_direct_effect(self, x: str, y: str, m: str,
                               x_treat: object, x_control: object) -> float:
        """Compute NDE: E[Y_{x_treat, M_{x_control}}] - E[Y_{x_control}].

        The effect of X on Y when mediator M is held at its natural value
        under control conditions.
        """
        # NDE = sum_m [E[Y|do(X=x1,M=m)] - E[Y|do(X=x0,M=m)]] P(M=m|do(X=x0))
        p_m_control = self.interventional_query([m], {x: x_control}).normalize()

        nde = 0.0
        for m_val in self.bn.domains[m]:
            p_m = p_m_control.get({m: m_val})
            cde_m = self.controlled_direct_effect(x, y, {m}, x_treat, x_control,
                                                   {m: m_val})
            nde += cde_m * p_m

        return nde

    def natural_indirect_effect(self, x: str, y: str, m: str,
                                 x_treat: object, x_control: object) -> float:
        """Compute NIE = ATE - NDE.

        The indirect (mediated) effect of X on Y through M.
        """
        ate = self.average_treatment_effect(x, y, x_treat, x_control)
        nde = self.natural_direct_effect(x, y, m, x_treat, x_control)
        return ate - nde

    # -- Counterfactual reasoning ---------------------------------------

    def counterfactual_query(self, query_var: str, intervention: dict[str, object],
                              evidence: dict[str, object]) -> dict:
        """Compute counterfactual: P(Y_x | evidence).

        "Given that we observed evidence, what would Y have been if X were x?"

        Uses the twin network method:
        1. Abduction: Compute posterior P(U | evidence) over exogenous variables
        2. Action: Apply intervention do(X=x)
        3. Prediction: Compute P(Y | U_posterior, do(X=x))

        For discrete BNs, we use a simpler approach:
        - Create a twin network with factual (observed) and counterfactual (intervened) copies
        - Exogenous variables (roots) are shared between twins
        - Condition on evidence in factual world, query counterfactual world
        """
        twin = self._build_twin_network(intervention)
        # Evidence applies to factual world variables
        twin_evidence = {k: v for k, v in evidence.items()}
        # Query the counterfactual version of the variable
        cf_var = f"{query_var}_cf"
        result_factor = variable_elimination(twin, [cf_var], twin_evidence)
        result_factor = result_factor.normalize()

        result = {}
        for val in self.bn.domains[query_var]:
            result[val] = result_factor.get({cf_var: val})
        return result

    def _build_twin_network(self, intervention: dict[str, object]) -> BayesianNetwork:
        """Build a twin network for counterfactual reasoning.

        The twin network has:
        - Factual world: original variables (same names)
        - Counterfactual world: variables with _cf suffix
        - Exogenous (root) variables shared between worlds
        """
        twin = BayesianNetwork()
        roots = [n for n in self.bn.nodes if not self.bn.parents[n]]
        non_roots = [n for n in self.bn.nodes if self.bn.parents[n]]

        # Add root (exogenous) nodes -- shared between worlds
        for r in roots:
            twin.add_node(r, self.bn.domains[r])

        # Add factual world non-root nodes
        for n in non_roots:
            twin.add_node(n, self.bn.domains[n])

        # Add counterfactual world non-root nodes
        for n in non_roots:
            twin.add_node(f"{n}_cf", self.bn.domains[n])

        # Also add counterfactual copies of roots that are intervened
        # (they keep the same value as factual, but we need cf aliases)
        for r in roots:
            twin.add_node(f"{r}_cf", self.bn.domains[r])

        # Add factual edges and CPTs
        for n in non_roots:
            for p in self.bn.parents[n]:
                twin.add_edge(p, n)
            twin.cpts[n] = Factor(
                list(self.bn.cpts[n].variables),
                {v: list(self.bn.cpts[n].domains[v]) for v in self.bn.cpts[n].variables},
                dict(self.bn.cpts[n].table)
            )

        # Add counterfactual edges and CPTs
        for n in non_roots:
            cf_name = f"{n}_cf"
            if n in intervention:
                # Intervened: deterministic CPT, no parents
                val = intervention[n]
                table = {}
                for d_val in self.bn.domains[n]:
                    table[(d_val,)] = 1.0 if d_val == val else 0.0
                twin.set_cpt(cf_name, table)
            else:
                # Not intervened: same CPT structure but with cf parent names
                orig_parents = self.bn.parents[n]
                cf_parents = []
                for p in orig_parents:
                    cf_parents.append(f"{p}_cf")  # Always use cf version
                for cp in cf_parents:
                    twin.add_edge(cp, cf_name)

                # Build CPT with cf variable names
                orig_cpt = self.bn.cpts[n]
                cf_vars = cf_parents + [cf_name]
                cf_domains = {}
                for i, op in enumerate(orig_parents):
                    cf_domains[cf_parents[i]] = list(self.bn.domains[op])
                cf_domains[cf_name] = list(self.bn.domains[n])

                cf_table = {}
                for assignment, prob in orig_cpt.table.items():
                    cf_table[assignment] = prob

                twin.cpts[cf_name] = Factor(cf_vars, cf_domains, cf_table)

        # Root CPTs (factual world)
        for r in roots:
            twin.cpts[r] = Factor(
                list(self.bn.cpts[r].variables),
                {v: list(self.bn.cpts[r].domains[v]) for v in self.bn.cpts[r].variables},
                dict(self.bn.cpts[r].table)
            )

        # Root counterfactual copies
        for r in roots:
            cf_name = f"{r}_cf"
            if r in intervention:
                # Intervened root: fix to intervention value
                val = intervention[r]
                table = {}
                for d_val in self.bn.domains[r]:
                    table[(d_val,)] = 1.0 if d_val == val else 0.0
                twin.set_cpt(cf_name, table)
            else:
                # Non-intervened root: deterministic copy of factual
                twin.add_edge(r, cf_name)
                table = {}
                for r_val in self.bn.domains[r]:
                    for cf_val in self.bn.domains[r]:
                        table[(r_val, cf_val)] = 1.0 if r_val == cf_val else 0.0
                twin.set_cpt(cf_name, table)

        return twin

    # -- Instrumental variables -----------------------------------------

    def is_instrument(self, z: str, x: str, y: str) -> bool:
        """Check if Z is a valid instrument for estimating X -> Y.

        Z is a valid instrument if:
        1. Z is associated with X (not d-separated from X given empty set)
        2. Z does not directly affect Y (no directed path Z -> Y not through X)
        3. Z and Y share no common causes (Z _||_ Y | empty in G_underbar_X)
        """
        # Condition 1: Z associated with X
        if self.d_separated({z}, {x}, set()):
            return False

        # Condition 2: No directed path Z -> Y bypassing X
        visited = set()
        stack = [z]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == y:
                return False  # Direct path found
            if node == x:
                continue  # Path through X is OK
            for child in self.bn.children.get(node, []):
                stack.append(child)

        # Condition 3: Z _||_ Y in G_underbar_X (edges from X removed)
        temp_bn = BayesianNetwork()
        for node in self.bn.nodes:
            temp_bn.add_node(node, self.bn.domains[node])
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if parent != x:
                    temp_bn.add_edge(parent, node)
        temp_model = CausalModel(temp_bn)
        return temp_model.d_separated({z}, {y}, set())

    # -- do-calculus rules (identifiability) ----------------------------

    def rule1_holds(self, y: set[str], x: set[str], z: set[str],
                     w: set[str]) -> bool:
        """Rule 1 of do-calculus: Insertion/deletion of observations.

        P(y | do(x), z, w) = P(y | do(x), w)
        if (Y _||_ Z | X, W) in G_overbar_X (remove incoming edges to X)

        This rule allows us to add or remove observations from a causal query.
        """
        # Build G_overbar_X: remove incoming edges to all nodes in X
        g_overbar = BayesianNetwork()
        for node in self.bn.nodes:
            g_overbar.add_node(node, self.bn.domains[node])
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if node not in x:
                    g_overbar.add_edge(parent, node)
        model = CausalModel(g_overbar)
        return model.d_separated(y, z, x | w)

    def rule2_holds(self, y: set[str], x: set[str], z: set[str],
                     w: set[str]) -> bool:
        """Rule 2 of do-calculus: Action/observation exchange.

        P(y | do(x), do(z), w) = P(y | do(x), z, w)
        if (Y _||_ Z | X, W) in G_overbar_X_underbar_Z

        This rule allows exchanging do(z) with observation z.
        """
        # Build G_overbar_X_underbar_Z: remove incoming to X, remove outgoing from Z
        g = BayesianNetwork()
        for node in self.bn.nodes:
            g.add_node(node, self.bn.domains[node])
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if node in x:
                    continue  # Remove incoming to X
                if parent in z:
                    continue  # Remove outgoing from Z
                g.add_edge(parent, node)
        model = CausalModel(g)
        return model.d_separated(y, z, x | w)

    def rule3_holds(self, y: set[str], x: set[str], z: set[str],
                     w: set[str]) -> bool:
        """Rule 3 of do-calculus: Insertion/deletion of actions.

        P(y | do(x), do(z), w) = P(y | do(x), w)
        if (Y _||_ Z | X, W) in G_overbar_X_overbar_Z(W)

        where Z(W) = Z \\ ancestors(W) in G_overbar_X

        This rule allows removing do(z) entirely.
        """
        # First find Z(W): non-ancestors of W in G_overbar_X
        g_overbar_x = BayesianNetwork()
        for node in self.bn.nodes:
            g_overbar_x.add_node(node, self.bn.domains[node])
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if node not in x:
                    g_overbar_x.add_edge(parent, node)
        model_ox = CausalModel(g_overbar_x)
        w_ancestors = model_ox._ancestors(w, g_overbar_x.parents)
        z_w = z - w_ancestors

        # Build G_overbar_X_overbar_Z(W): remove incoming to X and to Z(W)
        g = BayesianNetwork()
        for node in self.bn.nodes:
            g.add_node(node, self.bn.domains[node])
        remove_incoming = x | z_w
        for node in self.bn.nodes:
            for parent in self.bn.parents[node]:
                if node not in remove_incoming:
                    g.add_edge(parent, node)
        model = CausalModel(g)
        return model.d_separated(y, z, x | w)

    # -- Convenience builders -------------------------------------------

    def causal_effect(self, x: str, y: str, x_val: object) -> dict:
        """Compute P(Y | do(X=x_val)) using the best available method.

        Tries in order:
        1. Direct graph surgery (always works for interventional queries)
        """
        result_factor = self.interventional_query([y], {x: x_val})
        result_factor = result_factor.normalize()
        return {y_val: result_factor.get({y: y_val}) for y_val in self.bn.domains[y]}

    def causal_effect_all(self, x: str, y: str) -> dict[object, dict]:
        """Compute P(Y | do(X=x)) for all values of X.

        Returns dict mapping x_val -> {y_val: prob}.
        """
        return {x_val: self.causal_effect(x, y, x_val)
                for x_val in self.bn.domains[x]}


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_smoking_cancer_model() -> CausalModel:
    """Classic smoking-cancer example with confounding.

    Structure: U -> Smoking, U -> Cancer, Smoking -> Cancer
    U is an unobserved confounder.
    Without adjustment, P(Cancer|Smoking) != P(Cancer|do(Smoking)).
    """
    bn = BayesianNetwork()
    bn.add_node('U', [0, 1])      # Unobserved confounder (genetic factor)
    bn.add_node('Smoking', [0, 1])  # 0=no, 1=yes
    bn.add_node('Cancer', [0, 1])   # 0=no, 1=yes

    bn.add_edge('U', 'Smoking')
    bn.add_edge('U', 'Cancer')
    bn.add_edge('Smoking', 'Cancer')

    # P(U)
    bn.set_cpt('U', {(0,): 0.5, (1,): 0.5})
    # P(Smoking | U)
    bn.set_cpt('Smoking', {(0, 0): 0.9, (0, 1): 0.1,
                            (1, 0): 0.2, (1, 1): 0.8})
    # P(Cancer | U, Smoking)
    bn.set_cpt('Cancer', {(0, 0, 0): 0.95, (0, 0, 1): 0.05,
                           (0, 1, 0): 0.80, (0, 1, 1): 0.20,
                           (1, 0, 0): 0.70, (1, 0, 1): 0.30,
                           (1, 1, 0): 0.50, (1, 1, 1): 0.50})

    return CausalModel(bn)


def build_frontdoor_model() -> CausalModel:
    """Classic frontdoor criterion example.

    Structure: U -> X, U -> Y, X -> M -> Y
    U is unobserved. M is the mediator (frontdoor variable).
    X -> Y effect is identifiable via frontdoor adjustment through M.
    """
    bn = BayesianNetwork()
    bn.add_node('U', [0, 1])
    bn.add_node('X', [0, 1])
    bn.add_node('M', [0, 1])
    bn.add_node('Y', [0, 1])

    bn.add_edge('U', 'X')
    bn.add_edge('U', 'Y')
    bn.add_edge('X', 'M')
    bn.add_edge('M', 'Y')

    bn.set_cpt('U', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('X', {(0, 0): 0.8, (0, 1): 0.2,
                      (1, 0): 0.3, (1, 1): 0.7})
    bn.set_cpt('M', {(0, 0): 0.9, (0, 1): 0.1,
                      (1, 0): 0.3, (1, 1): 0.7})
    bn.set_cpt('Y', {(0, 0, 0): 0.95, (0, 0, 1): 0.05,
                      (0, 1, 0): 0.40, (0, 1, 1): 0.60,
                      (1, 0, 0): 0.80, (1, 0, 1): 0.20,
                      (1, 1, 0): 0.30, (1, 1, 1): 0.70})

    return CausalModel(bn)


def build_instrument_model() -> CausalModel:
    """Instrumental variable example.

    Structure: Z -> X -> Y, U -> X, U -> Y
    Z is an instrument for X -> Y (e.g., random encouragement).
    """
    bn = BayesianNetwork()
    bn.add_node('U', [0, 1])
    bn.add_node('Z', [0, 1])
    bn.add_node('X', [0, 1])
    bn.add_node('Y', [0, 1])

    bn.add_edge('Z', 'X')
    bn.add_edge('U', 'X')
    bn.add_edge('U', 'Y')
    bn.add_edge('X', 'Y')

    bn.set_cpt('U', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('Z', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('X', {(0, 0, 0): 0.9, (0, 0, 1): 0.1,
                      (0, 1, 0): 0.4, (0, 1, 1): 0.6,
                      (1, 0, 0): 0.7, (1, 0, 1): 0.3,
                      (1, 1, 0): 0.2, (1, 1, 1): 0.8})
    bn.set_cpt('Y', {(0, 0, 0): 0.95, (0, 0, 1): 0.05,
                      (0, 1, 0): 0.70, (0, 1, 1): 0.30,
                      (1, 0, 0): 0.60, (1, 0, 1): 0.40,
                      (1, 1, 0): 0.30, (1, 1, 1): 0.70})

    return CausalModel(bn)
