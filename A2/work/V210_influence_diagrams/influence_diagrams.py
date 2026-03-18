"""V210: Influence Diagrams -- Decision-theoretic extension of Bayesian Networks.

Extends V209 (Bayesian Networks) with:
1. Decision nodes -- variables under the agent's control
2. Utility nodes -- numeric payoff functions
3. Information links -- what is observed before each decision
4. Policy optimization -- find optimal decision rules via EU maximization
5. Value of information -- how much observing a variable is worth

References:
- Howard & Matheson (1984) "Influence Diagrams"
- Jensen & Nielsen (2007) "Bayesian Networks and Decision Graphs"
- Shachter (1986) "Evaluating Influence Diagrams"
"""

from __future__ import annotations
from collections import defaultdict
from itertools import product as iter_product
from math import log

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))

from bayesian_networks import (
    Factor, BayesianNetwork, variable_elimination, _min_degree_order,
)


# ---------------------------------------------------------------------------
# Node types for influence diagrams
# ---------------------------------------------------------------------------

class NodeType:
    CHANCE = "chance"      # random variable (circle in diagram)
    DECISION = "decision"  # controlled by agent (rectangle)
    UTILITY = "utility"    # payoff function (diamond)


# ---------------------------------------------------------------------------
# Utility Factor: maps assignments to real-valued utilities
# ---------------------------------------------------------------------------

class UtilityFactor:
    """A utility function over parent variables.

    Like a Factor, but values are utilities (can be negative),
    and there is no normalization.
    """

    def __init__(self, variables: list[str], domains: dict[str, list],
                 table: dict[tuple, float] | None = None):
        self.variables = list(variables)
        self.domains = {v: list(domains[v]) for v in self.variables}
        self.table: dict[tuple, float] = {}
        if table:
            self.table = dict(table)
        else:
            for combo in self._all_assignments():
                self.table[combo] = 0.0

    def _all_assignments(self) -> list[tuple]:
        if not self.variables:
            return [()]
        domain_lists = [self.domains[v] for v in self.variables]
        return list(iter_product(*domain_lists))

    def get(self, assignment: dict[str, object]) -> float:
        key = tuple(assignment[v] for v in self.variables)
        return self.table.get(key, 0.0)

    def set(self, assignment: dict[str, object], value: float):
        key = tuple(assignment[v] for v in self.variables)
        self.table[key] = value

    def __repr__(self):
        return f"UtilityFactor({self.variables})"


# ---------------------------------------------------------------------------
# Decision Policy: mapping from observations to actions
# ---------------------------------------------------------------------------

class Policy:
    """A decision policy: maps observed states to decisions.

    For a decision D with information set {I1, I2, ...}:
    policy[(i1_val, i2_val, ...)] = d_val
    """

    def __init__(self, decision: str, info_vars: list[str],
                 mapping: dict[tuple, object] | None = None):
        self.decision = decision
        self.info_vars = list(info_vars)
        self.mapping: dict[tuple, object] = mapping or {}

    def decide(self, observation: dict[str, object]) -> object:
        """Given observed values, return the decision."""
        key = tuple(observation[v] for v in self.info_vars)
        return self.mapping.get(key)

    def __repr__(self):
        return f"Policy({self.decision}, info={self.info_vars}, rules={len(self.mapping)})"


# ---------------------------------------------------------------------------
# Influence Diagram
# ---------------------------------------------------------------------------

class InfluenceDiagram:
    """An influence diagram: BN + decision nodes + utility nodes.

    Extends BayesianNetwork with decision-theoretic concepts.
    """

    def __init__(self):
        self.bn = BayesianNetwork()
        self.node_types: dict[str, str] = {}
        self.utility_factors: dict[str, UtilityFactor] = {}
        self.decision_order: list[str] = []  # temporal ordering of decisions
        self.info_sets: dict[str, list[str]] = {}  # what each decision observes

    # -- Node construction --

    def add_chance_node(self, name: str, domain: list):
        """Add a chance (random) node."""
        self.bn.add_node(name, domain)
        self.node_types[name] = NodeType.CHANCE

    def add_decision_node(self, name: str, domain: list, info_vars: list[str] | None = None):
        """Add a decision node.

        Args:
            name: Decision variable name.
            domain: Possible actions/choices.
            info_vars: Variables observed before this decision.
                       If None, determined from incoming arcs.
        """
        self.bn.add_node(name, domain)
        self.node_types[name] = NodeType.DECISION
        self.decision_order.append(name)
        self.info_sets[name] = list(info_vars) if info_vars is not None else []

    def add_utility_node(self, name: str, parents: list[str],
                          table: dict[tuple, float]):
        """Add a utility node with its utility function.

        Args:
            name: Utility node name.
            parents: Variables the utility depends on.
            table: Mapping from parent value tuples to utility values.
        """
        parent_domains = {p: self.bn.domains[p] for p in parents}
        self.utility_factors[name] = UtilityFactor(parents, parent_domains, table)
        self.node_types[name] = NodeType.UTILITY

    def add_edge(self, parent: str, child: str):
        """Add a directed edge (information or causal arc)."""
        if child in self.node_types and self.node_types[child] == NodeType.UTILITY:
            return  # utility parents tracked in utility_factors
        self.bn.add_edge(parent, child)
        # Auto-add to info set if child is a decision node
        if self.node_types.get(child) == NodeType.DECISION:
            if parent not in self.info_sets[child]:
                self.info_sets[child].append(parent)

    def set_cpt(self, node: str, table: dict[tuple, float]):
        """Set CPT for a chance node."""
        self.bn.set_cpt(node, table)

    def set_cpt_dict(self, node: str, cpt: dict):
        """Set CPT using dict format for a chance node."""
        self.bn.set_cpt_dict(node, cpt)

    # -- Query helpers --

    def chance_nodes(self) -> list[str]:
        return [n for n, t in self.node_types.items() if t == NodeType.CHANCE]

    def decision_nodes(self) -> list[str]:
        return list(self.decision_order)

    def utility_nodes(self) -> list[str]:
        return [n for n, t in self.node_types.items() if t == NodeType.UTILITY]

    def get_info_set(self, decision: str) -> list[str]:
        """What variables are observed before this decision?"""
        return list(self.info_sets.get(decision, []))

    # -- Expected utility computation --

    def _enumerate_eu(self, evidence: dict[str, object],
                       policies: dict[str, Policy]) -> float:
        """Core EU computation: enumerate over chance + unassigned decision nodes.

        Uses joint probability P(all chance | assignment) and normalizes by
        P(evidence) to get the correct conditional expectation.

        - Chance nodes not in evidence: enumerate, weight by CPT
        - Chance nodes in evidence: fixed, still included in CPT product
          (needed for correct conditioning via P(evidence) normalization)
        - Decision nodes with policy: assigned via policy
        - Decision nodes without policy: enumerate uniformly (marginalize)
        """
        chance = [n for n in self.bn.nodes if self.node_types.get(n) == NodeType.CHANCE]
        chance_to_enum = [c for c in chance if c not in evidence]

        # Decision nodes without a policy and not in evidence need enumeration
        decisions_to_enum = [
            d for d in self.decision_order
            if d not in evidence and d not in policies
        ]

        # Variables to enumerate: unobserved chance + unassigned decisions
        enum_vars = chance_to_enum + decisions_to_enum
        enum_domains = [self.bn.domains[v] for v in enum_vars]

        weighted_eu = 0.0
        total_prob = 0.0

        for vals in iter_product(*enum_domains) if enum_vars else [()]:
            assignment = dict(evidence)
            for var, val in zip(enum_vars, vals):
                assignment[var] = val

            # Apply policies for decision nodes that have them
            for dec in self.decision_order:
                if dec in assignment:
                    continue
                policy = policies.get(dec)
                if policy:
                    decision_val = policy.decide(assignment)
                    if decision_val is not None:
                        assignment[dec] = decision_val

            # Compute joint probability using ALL chance node CPTs
            prob = 1.0
            for node in chance:
                if node not in self.bn.cpts:
                    continue
                prob *= self.bn.cpts[node].get(assignment)
                if prob == 0.0:
                    break
            if prob == 0.0:
                continue

            utility = sum(uf.get(assignment) for uf in self.utility_factors.values())
            weighted_eu += prob * utility
            total_prob += prob

        # Normalize: EU = sum(P(x,e)*U) / sum(P(x,e)) = E[U | evidence]
        # This correctly handles both evidence conditioning and decision
        # marginalization: the ratio naturally adjusts for both.
        if total_prob > 0:
            return weighted_eu / total_prob
        return 0.0

    def expected_utility(self, policies: dict[str, Policy],
                          evidence: dict[str, object] | None = None) -> float:
        """Compute expected utility under given policies and evidence."""
        return self._enumerate_eu(evidence or {}, policies)

    # -- Policy optimization --

    def optimize_single_decision(self, decision: str,
                                  evidence: dict[str, object] | None = None,
                                  other_policies: dict[str, Policy] | None = None,
                                  ) -> tuple[Policy, float]:
        """Find optimal policy for a single decision node.

        Enumerates all possible policies (mapping from info set to action)
        and picks the one maximizing expected utility.

        Returns (optimal_policy, expected_utility).
        """
        evidence = evidence or {}
        other_policies = other_policies or {}
        info_vars = self.info_sets.get(decision, [])
        domain = self.bn.domains[decision]

        # Enumerate info set configurations
        if info_vars:
            info_domains = [self.bn.domains[v] for v in info_vars]
            info_configs = list(iter_product(*info_domains))
        else:
            info_configs = [()]

        # For each info config, find the best action
        best_mapping = {}
        for config in info_configs:
            best_action = None
            best_eu = float('-inf')

            for action in domain:
                # Create a policy that maps this config to this action
                test_mapping = dict(best_mapping)
                test_mapping[config] = action
                test_policy = Policy(decision, info_vars, test_mapping)

                # Build evidence + fixed configs
                test_evidence = dict(evidence)
                for var, val in zip(info_vars, config):
                    test_evidence[var] = val
                test_evidence[decision] = action

                # Compute EU for this specific (config, action) pair
                all_policies = dict(other_policies)
                all_policies[decision] = test_policy

                eu = self._conditional_eu(decision, action, info_vars, config,
                                           evidence, other_policies)

                if eu > best_eu:
                    best_eu = eu
                    best_action = action

            best_mapping[config] = best_action

        optimal_policy = Policy(decision, info_vars, best_mapping)
        all_policies = dict(other_policies)
        all_policies[decision] = optimal_policy
        total_eu = self.expected_utility(all_policies, evidence)

        return optimal_policy, total_eu

    def _conditional_eu(self, decision: str, action: object,
                         info_vars: list[str], info_config: tuple,
                         evidence: dict[str, object],
                         other_policies: dict[str, Policy]) -> float:
        """Expected utility conditioned on a specific info config and action."""
        cond_evidence = dict(evidence)
        for var, val in zip(info_vars, info_config):
            if self.node_types.get(var) == NodeType.CHANCE:
                cond_evidence[var] = val

        # Fix the decision + info decision vars via evidence
        cond_evidence[decision] = action
        for var, val in zip(info_vars, info_config):
            if self.node_types.get(var) == NodeType.DECISION:
                cond_evidence[var] = val

        all_policies = dict(other_policies)
        return self._enumerate_eu(cond_evidence, all_policies)

    def optimize_all_decisions(self, evidence: dict[str, object] | None = None,
                                ) -> tuple[dict[str, Policy], float]:
        """Find optimal policies for all decisions (backward induction).

        Optimizes decisions in reverse temporal order, fixing
        later decisions' optimal policies when optimizing earlier ones.
        """
        evidence = evidence or {}
        policies = {}

        # Backward induction: optimize from last decision to first
        for decision in reversed(self.decision_order):
            policy, _ = self.optimize_single_decision(
                decision, evidence, other_policies=policies
            )
            policies[decision] = policy

        total_eu = self.expected_utility(policies, evidence)
        return policies, total_eu

    # -- Value of information --

    def value_of_information(self, decision: str, info_var: str,
                              evidence: dict[str, object] | None = None,
                              ) -> float:
        """Compute the value of perfect information about info_var for a decision.

        VOI = EU(with info_var observed) - EU(without info_var observed)

        This is the maximum amount the agent would pay to observe info_var
        before making the decision.
        """
        evidence = evidence or {}

        # EU without observing info_var
        _, eu_without = self.optimize_all_decisions(evidence)

        # EU with observing info_var: weight by P(info_var = val)
        marginal = variable_elimination(self.bn, [info_var], evidence)
        eu_with = 0.0
        for val in self.bn.domains[info_var]:
            p_val = marginal.get({info_var: val})
            if p_val > 0:
                # Add info_var to decision's info set temporarily
                original_info = list(self.info_sets.get(decision, []))
                if info_var not in self.info_sets[decision]:
                    self.info_sets[decision].append(info_var)

                new_evidence = dict(evidence)
                new_evidence[info_var] = val
                _, eu_given_val = self.optimize_all_decisions(new_evidence)
                eu_with += p_val * eu_given_val

                # Restore
                self.info_sets[decision] = original_info

        return eu_with - eu_without

    def value_of_perfect_information(self, evidence: dict[str, object] | None = None,
                                      ) -> float:
        """Compute EVPI: value of observing ALL chance nodes before ALL decisions.

        EVPI = EU(perfect information) - EU(current information)
        """
        evidence = evidence or {}

        # Current EU
        _, eu_current = self.optimize_all_decisions(evidence)

        # Perfect information: enumerate all chance configs, optimize for each
        chance = [c for c in self.chance_nodes() if c not in evidence]
        if not chance:
            return 0.0

        chance_domains = [self.bn.domains[c] for c in chance]
        eu_perfect = 0.0

        for vals in iter_product(*chance_domains):
            # Full evidence
            full_evidence = dict(evidence)
            for var, val in zip(chance, vals):
                full_evidence[var] = val

            # Probability of this configuration
            prob = 1.0
            for node in self.bn.nodes:
                if self.node_types.get(node) != NodeType.CHANCE:
                    continue
                if node not in self.bn.cpts:
                    continue
                prob *= self.bn.cpts[node].get(full_evidence)
                if prob == 0.0:
                    break
            if prob == 0.0:
                continue

            # Optimal EU given perfect knowledge
            _, eu_given = self.optimize_all_decisions(full_evidence)
            eu_perfect += prob * eu_given

        return eu_perfect - eu_current

    # -- Analysis --

    def decision_table(self, decision: str,
                        evidence: dict[str, object] | None = None,
                        other_policies: dict[str, Policy] | None = None,
                        ) -> list[dict]:
        """Build a decision table showing EU for each (info_config, action) pair.

        Returns list of dicts with keys: info_config, action, expected_utility.
        """
        evidence = evidence or {}
        other_policies = other_policies or {}
        info_vars = self.info_sets.get(decision, [])
        domain = self.bn.domains[decision]

        if info_vars:
            info_domains = [self.bn.domains[v] for v in info_vars]
            info_configs = list(iter_product(*info_domains))
        else:
            info_configs = [()]

        rows = []
        for config in info_configs:
            for action in domain:
                eu = self._conditional_eu(
                    decision, action, info_vars, config,
                    evidence, other_policies
                )
                rows.append({
                    "info_config": dict(zip(info_vars, config)) if info_vars else {},
                    "action": action,
                    "expected_utility": round(eu, 6),
                })
        return rows

    def strategy_summary(self, evidence: dict[str, object] | None = None,
                          ) -> dict:
        """Summarize optimal strategy: policies, EU, and VOI for each variable."""
        evidence = evidence or {}
        policies, eu = self.optimize_all_decisions(evidence)

        voi = {}
        for decision in self.decision_order:
            for node in self.chance_nodes():
                if node not in evidence and node not in self.info_sets.get(decision, []):
                    voi_val = self.value_of_information(decision, node, evidence)
                    if voi_val > 1e-9:
                        voi[f"{decision}:{node}"] = round(voi_val, 6)

        return {
            "expected_utility": round(eu, 6),
            "policies": {d: {str(k): str(v) for k, v in p.mapping.items()}
                         for d, p in policies.items()},
            "value_of_information": voi,
        }


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def build_simple_decision(chance_name: str, chance_domain: list,
                           chance_prior: dict,
                           decision_name: str, decision_domain: list,
                           utility_table: dict[tuple, float],
                           observe_chance: bool = False,
                           ) -> InfluenceDiagram:
    """Build a simple influence diagram: one chance node, one decision, one utility.

    Structure: Chance -> Utility <- Decision
    Optionally: Chance -> Decision (information link)
    """
    diag = InfluenceDiagram()
    diag.add_chance_node(chance_name, chance_domain)
    diag.set_cpt_dict(chance_name, chance_prior)

    info = [chance_name] if observe_chance else []
    diag.add_decision_node(decision_name, decision_domain, info_vars=info)

    diag.add_utility_node("U", [chance_name, decision_name], utility_table)
    return diag


def build_sequential_decisions(stages: list[dict]) -> InfluenceDiagram:
    """Build a multi-stage influence diagram.

    Each stage dict has:
        chance: (name, domain, cpt_dict)  -- optional chance node
        decision: (name, domain, info_vars)
        utility: (name, parents, table)  -- optional utility
    """
    diag = InfluenceDiagram()

    for stage in stages:
        if "chance" in stage:
            name, domain, cpt = stage["chance"]
            diag.add_chance_node(name, domain)
            diag.set_cpt_dict(name, cpt)

        if "decision" in stage:
            name, domain, info = stage["decision"]
            diag.add_decision_node(name, domain, info_vars=info)

        if "utility" in stage:
            name, parents, table = stage["utility"]
            diag.add_utility_node(name, parents, table)

    return diag


def build_medical_diagnosis() -> InfluenceDiagram:
    """Classic medical diagnosis influence diagram.

    Disease (chance) -> Test Result (chance)
    Test Result -> Treatment Decision
    Disease, Treatment -> Outcome Utility
    """
    diag = InfluenceDiagram()

    # Disease: healthy or sick
    diag.add_chance_node("Disease", ["healthy", "sick"])
    diag.set_cpt_dict("Disease", {"healthy": 0.9, "sick": 0.1})

    # Test: positive or negative, depends on disease
    diag.add_chance_node("Test", ["pos", "neg"])
    diag.add_edge("Disease", "Test")
    diag.set_cpt_dict("Test", {
        ("healthy",): {"pos": 0.05, "neg": 0.95},  # 5% false positive
        ("sick",):    {"pos": 0.95, "neg": 0.05},   # 5% false negative
    })

    # Treatment decision: treat or not, observes test result
    diag.add_decision_node("Treatment", ["treat", "no_treat"], info_vars=["Test"])

    # Utility: depends on disease and treatment
    diag.add_utility_node("U", ["Disease", "Treatment"], {
        ("healthy", "treat"):    70,   # unnecessary treatment cost
        ("healthy", "no_treat"): 100,  # best outcome
        ("sick", "treat"):       80,   # treated disease
        ("sick", "no_treat"):    0,    # untreated disease (worst)
    })

    return diag


def build_oil_wildcatter() -> InfluenceDiagram:
    """Classic oil wildcatter problem (Raiffa, 1968).

    Oil (chance) -> Seismic Test (chance)
    Seismic Test -> Drill Decision
    Oil, Drill -> Payoff
    """
    diag = InfluenceDiagram()

    # Oil: dry, wet, or soaking
    diag.add_chance_node("Oil", ["dry", "wet", "soaking"])
    diag.set_cpt_dict("Oil", {"dry": 0.5, "wet": 0.3, "soaking": 0.2})

    # Seismic test result
    diag.add_chance_node("Seismic", ["closed", "open", "diffuse"])
    diag.add_edge("Oil", "Seismic")
    diag.set_cpt_dict("Seismic", {
        ("dry",):     {"closed": 0.1, "open": 0.3, "diffuse": 0.6},
        ("wet",):     {"closed": 0.3, "open": 0.4, "diffuse": 0.3},
        ("soaking",): {"closed": 0.5, "open": 0.4, "diffuse": 0.1},
    })

    # Drill decision: observes seismic result
    diag.add_decision_node("Drill", ["drill", "no_drill"], info_vars=["Seismic"])

    # Payoff
    diag.add_utility_node("U", ["Oil", "Drill"], {
        ("dry", "drill"):      -70,
        ("dry", "no_drill"):     0,
        ("wet", "drill"):       50,
        ("wet", "no_drill"):     0,
        ("soaking", "drill"):  200,
        ("soaking", "no_drill"): 0,
    })

    return diag
