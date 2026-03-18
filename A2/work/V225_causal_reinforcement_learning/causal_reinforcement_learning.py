"""V225: Causal Reinforcement Learning.

Composes V213 (MDP) + V211 (Causal Inference) + V221 (Contextual Causal Bandits)
for sequential decision-making under confounding.

Standard RL assumes E[R|a] = E[R|do(a)], i.e., observed rewards equal interventional
rewards. When unobserved confounders affect both action selection and outcomes, this
breaks. Causal RL separates observational from interventional distributions.

Key components:
  - CausalMDP: MDP with causal graph structure over state variables
  - ConfoundedMDP: MDP where logged data has confounding bias
  - CausalQLearning: Q-learning with backdoor/frontdoor adjustment
  - OffPolicyCausalEvaluator: counterfactual off-policy evaluation
  - CausalRewardDecomposition: decompose reward into direct/indirect effects
  - InterventionalPlanner: plan using do-calculus instead of conditional probs
  - CausalTransferRL: transfer causal knowledge across environments
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import sys
import os

# Add parent paths for dependencies
_base = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_base, "V213_markov_decision_processes"))
sys.path.insert(0, os.path.join(_base, "V211_causal_inference"))
sys.path.insert(0, os.path.join(_base, "V221_contextual_causal_bandits"))
sys.path.insert(0, os.path.join(_base, "V209_bayesian_networks"))
sys.path.insert(0, os.path.join(_base, "V214_causal_discovery"))

from markov_decision_processes import (
    MDP,
    MDPResult,
    value_iteration,
    policy_iteration,
    simulate,
    expected_total_reward,
    occupancy_measure,
)
from causal_inference import CausalModel, variable_elimination
from bayesian_networks import BayesianNetwork, Factor


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CausalState:
    """State represented as assignment to causal variables."""
    variables: dict[str, object]

    def key(self) -> str:
        """Deterministic string key for MDP state."""
        return "|".join(f"{k}={v}" for k, v in sorted(self.variables.items()))

    @staticmethod
    def from_key(key: str) -> "CausalState":
        """Parse state key back to CausalState."""
        variables = {}
        for part in key.split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                # Try numeric conversion
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                variables[k] = v
        return CausalState(variables=variables)


@dataclass
class CausalTransition:
    """A transition with causal provenance."""
    state: str
    action: str
    next_state: str
    probability: float
    reward: float
    causal_mechanism: str = ""  # Which structural equation generated this


@dataclass
class ConfoundedObservation:
    """An observation from a confounded behavioral policy."""
    state: str
    action: str
    next_state: str
    reward: float
    context: dict[str, object] = field(default_factory=dict)


@dataclass
class CausalRLResult:
    """Result from causal RL algorithms."""
    policy: dict[str, str]  # state -> action
    values: dict[str, float]  # state -> value
    q_values: dict[str, dict[str, float]]  # state -> action -> Q
    iterations: int = 0
    converged: bool = False
    causal_adjustments: int = 0  # Number of causal corrections applied
    confounding_detected: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CausalMDP: MDP with causal structure
# ---------------------------------------------------------------------------

class CausalMDP:
    """MDP where transitions are governed by a structural causal model.

    State variables form a causal DAG. Actions are interventions on
    action variables. Transitions follow structural equations.
    """

    def __init__(self, name: str = "CausalMDP"):
        self.name = name
        self.state_vars: list[str] = []  # Causal variables forming state
        self.state_domains: dict[str, list] = {}  # Variable -> possible values
        self.action_var: str = "action"  # Which variable is the action
        self.action_domain: list[str] = []  # Possible actions
        self.reward_var: str = "reward"  # Which variable is reward

        # Causal graph: parent -> children
        self.causal_parents: dict[str, list[str]] = defaultdict(list)
        self.causal_children: dict[str, list[str]] = defaultdict(list)

        # Structural equations: var -> function(parent_values) -> distribution
        # Each function returns dict[value, probability]
        self._structural_eqs: dict[str, Callable] = {}

        # Reward function: (state_vars, action) -> float
        self._reward_fn: Optional[Callable] = None

        # Confounders (unobserved variables affecting multiple observed vars)
        self.confounders: dict[str, list[str]] = {}  # confounder -> affected vars
        self.confounder_domains: dict[str, list] = {}
        self.confounder_priors: dict[str, dict] = {}  # confounder -> {val: prob}

        # Underlying MDP (built from causal structure)
        self._mdp: Optional[MDP] = None

    def add_state_var(self, name: str, domain: list):
        """Add a causal variable to the state space."""
        self.state_vars.append(name)
        self.state_domains[name] = list(domain)
        return self

    def set_action_var(self, name: str, domain: list[str]):
        """Set which variable represents the action."""
        self.action_var = name
        self.action_domain = list(domain)
        return self

    def add_edge(self, parent: str, child: str):
        """Add causal edge parent -> child."""
        if parent not in self.causal_parents[child]:
            self.causal_parents[child].append(parent)
        if child not in self.causal_children[parent]:
            self.causal_children[parent].append(child)
        return self

    def set_structural_eq(self, var: str, fn: Callable):
        """Set structural equation for variable.

        fn(parent_values: dict[str, object]) -> dict[value, probability]
        where parent_values maps parent variable names to their values.
        """
        self._structural_eqs[var] = fn
        return self

    def set_reward_fn(self, fn: Callable):
        """Set reward function: fn(state_vars: dict, action: str) -> float."""
        self._reward_fn = fn
        return self

    def add_confounder(self, name: str, domain: list, prior: dict,
                       affects: list[str]):
        """Add an unobserved confounder variable.

        Args:
            name: Confounder variable name
            domain: Possible values
            prior: Prior distribution {value: probability}
            affects: List of observed variables this confounder influences
        """
        self.confounders[name] = affects
        self.confounder_domains[name] = list(domain)
        self.confounder_priors[name] = dict(prior)
        for v in affects:
            self.add_edge(name, v)
        return self

    def _enumerate_states(self) -> list[str]:
        """Enumerate all possible state keys."""
        vars_list = sorted(self.state_vars)
        domains = [self.state_domains[v] for v in vars_list]

        def cartesian(domains):
            if not domains:
                return [{}]
            first = domains[0]
            rest = cartesian(domains[1:])
            return [{**r, vars_list[len(domains) - len(domains)]: v}
                    for v in first for r in rest]

        # Simpler approach
        from itertools import product as iterproduct
        combos = list(iterproduct(*domains))
        states = []
        for combo in combos:
            assignment = {vars_list[i]: combo[i] for i in range(len(vars_list))}
            cs = CausalState(variables=assignment)
            states.append(cs.key())
        return states

    def _compute_transition(self, state_key: str, action: str) -> list[tuple[str, float, float]]:
        """Compute P(s'|s, do(a)) using structural equations.

        Returns list of (next_state_key, probability, reward).
        """
        state = CausalState.from_key(state_key)
        sv = dict(state.variables)

        # Set action variable
        sv[self.action_var] = action

        # For each next-state variable, compute distribution given parents
        # Need topological order of state variables
        topo_order = self._topological_sort()

        # Compute joint distribution over next-state variables
        # Using chain rule over causal DAG
        next_state_dists = {}  # var -> {value: prob}

        for var in topo_order:
            if var in self.confounders:
                continue  # Handle confounders separately
            if var == self.action_var:
                continue  # Action is intervened

            if var in self._structural_eqs:
                # Get parent values.
                # State variable parents represent CURRENT (t) values,
                # so always use sv. Only non-state, non-confounder parents
                # that were computed this timestep use next_state_dists.
                parent_vals = {}
                for p in self.causal_parents.get(var, []):
                    if p == self.action_var:
                        parent_vals[p] = action
                    elif p in self.confounders:
                        continue  # Marginalize out
                    elif p in self.state_domains:
                        # State variable parent = current value (temporal edge)
                        if p in sv:
                            parent_vals[p] = sv[p]
                    elif p in next_state_dists:
                        parent_vals[p] = next_state_dists[p]
                    elif p in sv:
                        parent_vals[p] = sv[p]

                # If any parent has a distribution, we need to expand
                dist_parents = {k: v for k, v in parent_vals.items()
                                if isinstance(v, dict)}
                fixed_parents = {k: v for k, v in parent_vals.items()
                                 if not isinstance(v, dict)}

                if not dist_parents:
                    # All parents fixed
                    next_state_dists[var] = self._structural_eqs[var](parent_vals)
                else:
                    # Marginalize over distributional parents
                    merged = self._marginalize_structural_eq(
                        var, fixed_parents, dist_parents)
                    next_state_dists[var] = merged
            else:
                # No structural equation -- variable stays the same
                if var in sv:
                    next_state_dists[var] = {sv[var]: 1.0}

        # Handle confounders: marginalize them out
        if self.confounders:
            next_state_dists = self._marginalize_confounders(
                sv, action, next_state_dists)

        # Build joint distribution over next states
        vars_list = sorted(self.state_vars)
        transitions = []

        # Enumerate combinations weighted by probabilities
        from itertools import product as iterproduct
        var_values = []
        var_probs = []
        for v in vars_list:
            if v in next_state_dists:
                dist = next_state_dists[v]
                var_values.append(list(dist.keys()))
                var_probs.append(dist)
            else:
                # Default: stays the same
                var_values.append([sv.get(v, 0)])
                var_probs.append({sv.get(v, 0): 1.0})

        for combo in iterproduct(*var_values):
            assignment = {vars_list[i]: combo[i] for i in range(len(vars_list))}
            prob = 1.0
            for i, v in enumerate(vars_list):
                prob *= var_probs[i].get(combo[i], 0.0)
            if prob > 1e-12:
                ns = CausalState(variables=assignment)
                reward = 0.0
                if self._reward_fn:
                    reward = self._reward_fn(assignment, action)
                transitions.append((ns.key(), prob, reward))

        return transitions

    def _marginalize_structural_eq(self, var: str,
                                    fixed_parents: dict,
                                    dist_parents: dict) -> dict:
        """Marginalize structural equation over distributional parents."""
        result = defaultdict(float)

        # Enumerate all combinations of distributional parent values
        from itertools import product as iterproduct
        dist_names = list(dist_parents.keys())
        dist_vals = [list(dist_parents[n].keys()) for n in dist_names]

        for combo in iterproduct(*dist_vals):
            # Probability of this parent combination
            p_combo = 1.0
            parent_vals = dict(fixed_parents)
            for i, name in enumerate(dist_names):
                parent_vals[name] = combo[i]
                p_combo *= dist_parents[name].get(combo[i], 0.0)

            if p_combo < 1e-15:
                continue

            # Get structural equation output for these parent values
            child_dist = self._structural_eqs[var](parent_vals)
            for val, p_val in child_dist.items():
                result[val] += p_combo * p_val

        return dict(result)

    def _marginalize_confounders(self, sv: dict, action: str,
                                  next_state_dists: dict) -> dict:
        """Marginalize unobserved confounders from transition computation.

        This implements the do-calculus adjustment for confounders.
        """
        if not self.confounders:
            return next_state_dists

        # For each confounder, sum over its values weighted by prior
        from itertools import product as iterproduct

        conf_names = list(self.confounders.keys())
        conf_vals = [self.confounder_domains[c] for c in conf_names]

        vars_list = sorted(self.state_vars)
        merged_dists = {v: defaultdict(float) for v in vars_list}

        for combo in iterproduct(*conf_vals):
            conf_assignment = {conf_names[i]: combo[i]
                               for i in range(len(conf_names))}

            # Prior probability of this confounder combination
            p_conf = 1.0
            for i, c in enumerate(conf_names):
                p_conf *= self.confounder_priors[c].get(combo[i], 0.0)

            if p_conf < 1e-15:
                continue

            # Recompute structural equations with confounder values known
            for var in vars_list:
                if var in self._structural_eqs:
                    parent_vals = {}
                    for p in self.causal_parents.get(var, []):
                        if p == self.action_var:
                            parent_vals[p] = action
                        elif p in conf_assignment:
                            parent_vals[p] = conf_assignment[p]
                        elif p in sv:
                            parent_vals[p] = sv[p]

                    child_dist = self._structural_eqs[var](parent_vals)
                    for val, p_val in child_dist.items():
                        merged_dists[var][val] += p_conf * p_val

        return {v: dict(d) for v, d in merged_dists.items() if d}

    def _topological_sort(self) -> list[str]:
        """Topological sort of all variables (state + confounders).

        Self-edges (x -> x) are temporal (current -> next) and don't
        create within-timestep dependencies, so they're excluded.
        """
        all_vars = set(self.state_vars) | set(self.confounders.keys())
        in_degree = defaultdict(int)
        for v in all_vars:
            for p in self.causal_parents.get(v, []):
                if p in all_vars and p != v:  # Exclude self-loops
                    in_degree[v] += 1

        queue = [v for v in all_vars if in_degree[v] == 0]
        result = []
        while queue:
            queue.sort()  # Deterministic
            v = queue.pop(0)
            result.append(v)
            for c in self.causal_children.get(v, []):
                if c in all_vars and c != v:  # Exclude self-loops
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)

        # Add any remaining vars not reached (shouldn't happen in DAGs)
        for v in sorted(all_vars):
            if v not in result:
                result.append(v)

        return result

    def to_mdp(self) -> MDP:
        """Convert CausalMDP to standard MDP by computing all transitions."""
        mdp = MDP(self.name)
        states = self._enumerate_states()

        for s in states:
            mdp.add_state(s)
        for a in self.action_domain:
            mdp.add_action(a)

        # Set first state as initial
        if states:
            mdp.set_initial(states[0])

        # Compute all transitions
        for s in states:
            for a in self.action_domain:
                transitions = self._compute_transition(s, a)
                for ns, prob, reward in transitions:
                    if ns in states:
                        mdp.add_transition(s, a, ns, prob, reward)

        self._mdp = mdp
        return mdp

    def interventional_transition(self, state_key: str, action: str) -> list[tuple[str, float, float]]:
        """Compute P(s' | s, do(action)) -- interventional transition.

        Unlike observational P(s'|s,a), this removes confounding bias
        by applying do-calculus (graph surgery on action variable).
        """
        # In a CausalMDP, the structural equations already define
        # interventional semantics. The do() operation severs incoming
        # edges to the action variable (which has none in our formulation).
        return self._compute_transition(state_key, action)

    def observational_transition(self, state_key: str, action: str,
                                  behavior_policy: dict[str, dict[str, float]]) -> list[tuple[str, float, float]]:
        """Compute P(s'|s, A=a) -- observational transition under behavior policy.

        When there's confounding, P(s'|s,A=a) != P(s'|s,do(a)).
        The behavioral policy's action selection may be correlated with
        unobserved confounders.

        Args:
            behavior_policy: state -> {action: probability}
        """
        if not self.confounders:
            return self._compute_transition(state_key, action)

        state = CausalState.from_key(state_key)
        sv = dict(state.variables)

        # Under observational distribution, we condition on A=a
        # instead of intervening. This means confounders that affect
        # both action selection and outcomes create bias.

        # Use Bayes' rule: P(s'|s,A=a) = sum_u P(s'|s,a,u) P(u|s,A=a)
        # where P(u|s,A=a) = P(A=a|s,u) P(u) / P(A=a|s)

        from itertools import product as iterproduct

        conf_names = list(self.confounders.keys())
        conf_vals = [self.confounder_domains[c] for c in conf_names]

        # P(u|s, A=a) via Bayes
        posterior_u = {}
        total = 0.0
        for combo in iterproduct(*conf_vals):
            conf_assignment = {conf_names[i]: combo[i]
                               for i in range(len(conf_names))}
            # Prior P(u)
            p_u = 1.0
            for i, c in enumerate(conf_names):
                p_u *= self.confounder_priors[c].get(combo[i], 0.0)

            # P(A=a|s,u) from behavior policy (which may depend on confounder)
            p_a_given_su = behavior_policy.get(state_key, {}).get(action, 0.0)
            # If behavior policy depends on confounder, adjust
            # For simplicity, use uniform if no specific info
            if p_a_given_su == 0.0:
                p_a_given_su = 1.0 / max(len(self.action_domain), 1)

            posterior_u[tuple(combo)] = p_u * p_a_given_su
            total += p_u * p_a_given_su

        if total < 1e-15:
            return self._compute_transition(state_key, action)

        for k in posterior_u:
            posterior_u[k] /= total

        # Now compute P(s'|s,a) = sum_u P(s'|s,a,u) P(u|s,A=a)
        transitions = defaultdict(lambda: [0.0, 0.0])  # ns -> [prob, reward]

        for combo, p_u in posterior_u.items():
            if p_u < 1e-15:
                continue
            conf_assignment = {conf_names[i]: combo[i]
                               for i in range(len(conf_names))}

            # Compute transition with known confounders
            parent_vals_base = dict(sv)
            parent_vals_base[self.action_var] = action
            parent_vals_base.update(conf_assignment)

            # Recompute with confounders fixed
            vars_list = sorted(self.state_vars)
            var_dists = {}
            for var in vars_list:
                if var in self._structural_eqs:
                    pv = {}
                    for p in self.causal_parents.get(var, []):
                        if p in parent_vals_base:
                            pv[p] = parent_vals_base[p]
                    var_dists[var] = self._structural_eqs[var](pv)
                elif var in sv:
                    var_dists[var] = {sv[var]: 1.0}

            var_values = [list(var_dists.get(v, {sv.get(v, 0): 1.0}).keys())
                          for v in vars_list]
            for val_combo in iterproduct(*var_values):
                assignment = {vars_list[i]: val_combo[i]
                              for i in range(len(vars_list))}
                p_transition = 1.0
                for i, v in enumerate(vars_list):
                    dist = var_dists.get(v, {sv.get(v, 0): 1.0})
                    p_transition *= dist.get(val_combo[i], 0.0)

                if p_transition > 1e-12:
                    ns = CausalState(variables=assignment)
                    ns_key = ns.key()
                    reward = 0.0
                    if self._reward_fn:
                        reward = self._reward_fn(assignment, action)
                    transitions[ns_key][0] += p_u * p_transition
                    transitions[ns_key][1] += p_u * p_transition * reward

        result = []
        for ns_key, (prob, weighted_reward) in transitions.items():
            reward = weighted_reward / prob if prob > 1e-15 else 0.0
            result.append((ns_key, prob, reward))
        return result


# ---------------------------------------------------------------------------
# ConfoundedMDP: MDP with observational data from confounded behavior policy
# ---------------------------------------------------------------------------

class ConfoundedMDP:
    """Wrapper around MDP where logged data comes from a confounded policy.

    The key problem: in logged data, actions are chosen by a behavior policy
    that may observe confounders we don't have access to. Standard Q-learning
    on such data gives biased estimates.

    This class provides methods to detect and correct for confounding.
    """

    def __init__(self, mdp: MDP, name: str = "ConfoundedMDP"):
        self.mdp = mdp
        self.name = name
        self.observations: list[ConfoundedObservation] = []
        self._adjustment_sets: dict[str, set[str]] = {}  # action -> adjustment vars

    def add_observation(self, state: str, action: str, next_state: str,
                         reward: float, context: dict = None):
        """Add an observation from the logged (potentially confounded) data."""
        self.observations.append(ConfoundedObservation(
            state=state, action=action, next_state=next_state,
            reward=reward, context=context or {}
        ))

    def add_observations_batch(self, obs: list[tuple]):
        """Add batch of (state, action, next_state, reward, context?) tuples."""
        for o in obs:
            ctx = o[4] if len(o) > 4 else {}
            self.add_observation(o[0], o[1], o[2], o[3], ctx)

    def empirical_reward(self, state: str, action: str) -> float:
        """Naive empirical reward: E[R | S=s, A=a] (potentially biased)."""
        rewards = [o.reward for o in self.observations
                   if o.state == state and o.action == action]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def empirical_transition(self, state: str, action: str) -> dict[str, float]:
        """Naive empirical transition: P(S'|S=s, A=a) (potentially biased)."""
        next_states = [o.next_state for o in self.observations
                       if o.state == state and o.action == action]
        if not next_states:
            return {}
        counts = defaultdict(int)
        for ns in next_states:
            counts[ns] += 1
        total = len(next_states)
        return {ns: c / total for ns, c in counts.items()}

    def detect_confounding(self, state: str, action: str,
                            context_var: str) -> float:
        """Detect confounding by checking if context variable is associated
        with both action selection and reward.

        Returns confounding strength (0 = no confounding, higher = more).
        Uses the difference between conditional and marginal reward estimates.
        """
        # Group observations by context value
        context_groups = defaultdict(list)
        for o in self.observations:
            if o.state == state:
                cv = o.context.get(context_var)
                if cv is not None:
                    context_groups[cv].append(o)

        if len(context_groups) < 2:
            return 0.0

        # Check if action frequency varies with context (action-context association)
        action_freq_by_context = {}
        for cv, obs in context_groups.items():
            total = len(obs)
            action_count = sum(1 for o in obs if o.action == action)
            action_freq_by_context[cv] = action_count / total if total > 0 else 0.0

        # Check if reward varies with context for same action
        reward_by_context = {}
        for cv, obs in context_groups.items():
            action_obs = [o for o in obs if o.action == action]
            if action_obs:
                reward_by_context[cv] = sum(o.reward for o in action_obs) / len(action_obs)

        if not reward_by_context or not action_freq_by_context:
            return 0.0

        # Confounding strength: variation in action freq * variation in reward
        freq_vals = list(action_freq_by_context.values())
        rew_vals = list(reward_by_context.values())

        freq_range = max(freq_vals) - min(freq_vals)
        rew_range = max(rew_vals) - min(rew_vals) if len(rew_vals) > 1 else 0.0

        return freq_range * rew_range

    def backdoor_adjusted_reward(self, state: str, action: str,
                                  adjustment_var: str) -> float:
        """Compute E[R | do(A=a), S=s] using backdoor adjustment.

        Formula: E[R|do(a),s] = sum_z E[R|s,a,z] P(z|s)

        Args:
            adjustment_var: Context variable to adjust for (backdoor set)
        """
        # Group by adjustment variable values
        groups = defaultdict(list)
        state_obs = [o for o in self.observations if o.state == state]

        for o in state_obs:
            z = o.context.get(adjustment_var)
            if z is not None:
                groups[z].append(o)

        if not groups:
            return self.empirical_reward(state, action)

        # P(z|s) marginal distribution
        total_state_obs = sum(len(v) for v in groups.values())
        p_z = {z: len(obs) / total_state_obs for z, obs in groups.items()}

        # E[R|s,a,z] conditional reward
        adjusted_reward = 0.0
        for z, z_obs in groups.items():
            action_obs = [o for o in z_obs if o.action == action]
            if action_obs:
                e_r_saz = sum(o.reward for o in action_obs) / len(action_obs)
                adjusted_reward += e_r_saz * p_z[z]

        return adjusted_reward

    def ipw_reward(self, state: str, action: str,
                   propensity_scores: dict[str, dict[str, float]]) -> float:
        """Inverse Propensity Weighting estimate of E[R|do(a),s].

        Formula: E[R|do(a),s] = (1/N) sum_{i: a_i=a} R_i / pi(a|s_i)

        Args:
            propensity_scores: state -> {action: P(action|state)} behavior policy
        """
        relevant = [o for o in self.observations
                    if o.state == state and o.action == action]
        if not relevant:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0
        for o in relevant:
            pi_a = propensity_scores.get(o.state, {}).get(o.action, 0.1)
            pi_a = max(pi_a, 0.01)  # Clip for stability
            w = 1.0 / pi_a
            weighted_sum += w * o.reward
            weight_sum += w

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def doubly_robust_reward(self, state: str, action: str,
                              propensity_scores: dict[str, dict[str, float]],
                              baseline_values: dict[str, dict[str, float]]) -> float:
        """Doubly robust estimate of E[R|do(a),s].

        Combines IPW with a direct model (baseline) for lower variance.
        DR = E_hat[R|s,a] + (1/N) sum w_i (R_i - E_hat[R|s_i,a_i])

        Args:
            propensity_scores: state -> {action: P(action|state)}
            baseline_values: state -> {action: E_hat[R|s,a]} direct model
        """
        baseline = baseline_values.get(state, {}).get(action, 0.0)

        relevant = [o for o in self.observations
                    if o.state == state and o.action == action]
        if not relevant:
            return baseline

        correction = 0.0
        weight_sum = 0.0
        for o in relevant:
            pi_a = propensity_scores.get(o.state, {}).get(o.action, 0.1)
            pi_a = max(pi_a, 0.01)
            w = 1.0 / pi_a
            residual = o.reward - baseline_values.get(o.state, {}).get(o.action, 0.0)
            correction += w * residual
            weight_sum += w

        if weight_sum > 0:
            correction /= weight_sum

        return baseline + correction


# ---------------------------------------------------------------------------
# CausalQLearning: Q-learning with causal adjustments
# ---------------------------------------------------------------------------

class CausalQLearning:
    """Q-learning that corrects for confounding in observational data.

    Standard Q-learning assumes: Q(s,a) = E[R + gamma * max Q(s',a') | s, a]
    Under confounding: E[R|s,a] != E[R|s,do(a)].

    This algorithm uses backdoor adjustment or IPW to debias the Q-update.
    """

    def __init__(self, mdp: MDP, gamma: float = 0.99,
                 alpha: float = 0.1, epsilon: float = 0.1,
                 adjustment_method: str = "backdoor",
                 seed: int = None):
        self.mdp = mdp
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.adjustment_method = adjustment_method
        self.rng = random.Random(seed)

        # Q-values
        self.q_values: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))

        # Causal adjustment data
        self.adjustment_var: Optional[str] = None
        self.propensity_scores: dict[str, dict[str, float]] = {}
        self.observation_buffer: list[ConfoundedObservation] = []
        self.causal_adjustments = 0

    def set_adjustment_var(self, var: str):
        """Set the variable to use for backdoor adjustment."""
        self.adjustment_var = var

    def set_propensity_scores(self, scores: dict[str, dict[str, float]]):
        """Set propensity scores for IPW adjustment."""
        self.propensity_scores = scores

    def update(self, state: str, action: str, reward: float,
               next_state: str, context: dict = None, done: bool = False):
        """Causal Q-learning update with adjustment."""
        context = context or {}

        # Store observation for adjustment computation
        self.observation_buffer.append(ConfoundedObservation(
            state=state, action=action, next_state=next_state,
            reward=reward, context=context
        ))

        # Compute adjusted reward
        adjusted_reward = reward
        if self.adjustment_method == "backdoor" and self.adjustment_var:
            adjusted_reward = self._backdoor_adjust(
                state, action, reward, context)
            self.causal_adjustments += 1
        elif self.adjustment_method == "ipw" and self.propensity_scores:
            adjusted_reward = self._ipw_adjust(state, action, reward)
            self.causal_adjustments += 1

        # Q-learning update with adjusted reward
        if done:
            td_target = adjusted_reward
        else:
            actions = self.mdp.get_actions(next_state)
            if actions:
                max_q = max(self.q_values[next_state][a] for a in actions)
            else:
                max_q = 0.0
            td_target = adjusted_reward + self.gamma * max_q

        old_q = self.q_values[state][action]
        self.q_values[state][action] = old_q + self.alpha * (td_target - old_q)

    def _backdoor_adjust(self, state: str, action: str,
                          reward: float, context: dict) -> float:
        """Apply backdoor adjustment to reward."""
        if self.adjustment_var not in context:
            return reward

        # Get observations for this state with different context values
        relevant = [o for o in self.observation_buffer
                    if o.state == state and o.action == action
                    and self.adjustment_var in o.context]

        if len(relevant) < 5:  # Not enough data for adjustment
            return reward

        # Group by adjustment variable
        groups = defaultdict(list)
        for o in relevant:
            groups[o.context[self.adjustment_var]].append(o.reward)

        # P(z|s)
        total = sum(len(v) for v in groups.values())
        adjusted = 0.0
        for z, rewards in groups.items():
            p_z = len(rewards) / total
            e_r = sum(rewards) / len(rewards)
            adjusted += e_r * p_z

        return adjusted

    def _ipw_adjust(self, state: str, action: str, reward: float) -> float:
        """Apply IPW adjustment to reward."""
        pi_a = self.propensity_scores.get(state, {}).get(action, 0.5)
        pi_a = max(pi_a, 0.01)
        # Self-normalized IPW: reweight this reward
        return reward / pi_a * (1.0 / len(self.mdp.actions))

    def select_action(self, state: str) -> str:
        """Epsilon-greedy action selection."""
        actions = self.mdp.get_actions(state)
        if not actions:
            return self.mdp.actions[0] if self.mdp.actions else ""

        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        # Greedy
        best_a = actions[0]
        best_q = self.q_values[state][actions[0]]
        for a in actions[1:]:
            q = self.q_values[state][a]
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def get_policy(self) -> dict[str, str]:
        """Extract greedy policy from Q-values."""
        policy = {}
        for s in set(o.state for o in self.observation_buffer) | set(self.q_values.keys()):
            actions = self.mdp.get_actions(s)
            if actions:
                best_a = max(actions, key=lambda a: self.q_values[s][a])
                policy[s] = best_a
        return policy

    def get_values(self) -> dict[str, float]:
        """Extract value function from Q-values."""
        values = {}
        for s in self.q_values:
            actions = self.mdp.get_actions(s)
            if actions:
                values[s] = max(self.q_values[s][a] for a in actions)
            else:
                values[s] = 0.0
        return values

    def result(self) -> CausalRLResult:
        """Get current learning result."""
        return CausalRLResult(
            policy=self.get_policy(),
            values=self.get_values(),
            q_values={s: dict(aq) for s, aq in self.q_values.items()},
            iterations=len(self.observation_buffer),
            causal_adjustments=self.causal_adjustments,
        )


# ---------------------------------------------------------------------------
# OffPolicyCausalEvaluator: evaluate target policy from logged data
# ---------------------------------------------------------------------------

class OffPolicyCausalEvaluator:
    """Evaluate a target policy using logged data from a different behavior policy.

    Standard importance sampling can have high variance. Causal off-policy
    evaluation uses structural knowledge to reduce variance.

    Methods:
      - IS: standard importance sampling
      - WIS: weighted (self-normalized) importance sampling
      - DR: doubly robust estimation
      - Causal IS: importance sampling with causal adjustment
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def importance_sampling(self, trajectories: list[list[ConfoundedObservation]],
                             target_policy: dict[str, dict[str, float]],
                             behavior_policy: dict[str, dict[str, float]]) -> float:
        """Standard importance sampling estimator.

        V_target = (1/N) sum_i prod_t (pi_target(a_t|s_t) / pi_behavior(a_t|s_t)) * G_i
        """
        if not trajectories:
            return 0.0

        estimates = []
        for traj in trajectories:
            rho = 1.0  # importance weight
            G = 0.0    # discounted return
            for t, o in enumerate(traj):
                pi_t = target_policy.get(o.state, {}).get(o.action, 0.0)
                pi_b = behavior_policy.get(o.state, {}).get(o.action, 0.1)
                pi_b = max(pi_b, 0.01)
                rho *= pi_t / pi_b
                G += (self.gamma ** t) * o.reward

            estimates.append(rho * G)

        return sum(estimates) / len(estimates)

    def weighted_importance_sampling(self,
                                      trajectories: list[list[ConfoundedObservation]],
                                      target_policy: dict[str, dict[str, float]],
                                      behavior_policy: dict[str, dict[str, float]]) -> float:
        """Weighted (self-normalized) importance sampling. Lower variance than IS."""
        if not trajectories:
            return 0.0

        weighted_returns = []
        weights = []
        for traj in trajectories:
            rho = 1.0
            G = 0.0
            for t, o in enumerate(traj):
                pi_t = target_policy.get(o.state, {}).get(o.action, 0.0)
                pi_b = behavior_policy.get(o.state, {}).get(o.action, 0.1)
                pi_b = max(pi_b, 0.01)
                rho *= pi_t / pi_b
                G += (self.gamma ** t) * o.reward

            weighted_returns.append(rho * G)
            weights.append(rho)

        total_weight = sum(weights)
        if total_weight < 1e-15:
            return 0.0
        return sum(weighted_returns) / total_weight

    def doubly_robust(self, trajectories: list[list[ConfoundedObservation]],
                       target_policy: dict[str, dict[str, float]],
                       behavior_policy: dict[str, dict[str, float]],
                       q_model: dict[str, dict[str, float]],
                       v_model: dict[str, float]) -> float:
        """Doubly robust off-policy evaluation.

        Combines direct method (q_model) with IS correction.
        Consistent if either model or IS weights are correct.
        """
        if not trajectories:
            return 0.0

        estimates = []
        for traj in trajectories:
            dr_value = v_model.get(traj[0].state, 0.0) if traj else 0.0
            rho_prod = 1.0

            for t, o in enumerate(traj):
                pi_t = target_policy.get(o.state, {}).get(o.action, 0.0)
                pi_b = behavior_policy.get(o.state, {}).get(o.action, 0.1)
                pi_b = max(pi_b, 0.01)
                rho = pi_t / pi_b

                q_sa = q_model.get(o.state, {}).get(o.action, 0.0)
                v_s = v_model.get(o.state, 0.0)

                # DR correction at step t
                discount = self.gamma ** t
                dr_value += discount * rho_prod * (
                    rho * (o.reward + self.gamma * v_model.get(o.next_state, 0.0) - q_sa)
                )
                rho_prod *= rho

            estimates.append(dr_value)

        return sum(estimates) / len(estimates)

    def causal_importance_sampling(self,
                                    trajectories: list[list[ConfoundedObservation]],
                                    target_policy: dict[str, dict[str, float]],
                                    behavior_policy: dict[str, dict[str, float]],
                                    adjustment_var: str) -> float:
        """Causal IS: importance sampling with backdoor adjustment.

        Adjusts for confounding by stratifying on the adjustment variable.
        Lower variance than standard IS when confounding is present.
        """
        if not trajectories:
            return 0.0

        # Group trajectories by adjustment variable value
        strata = defaultdict(list)
        for traj in trajectories:
            if traj and adjustment_var in traj[0].context:
                z = traj[0].context[adjustment_var]
                strata[z].append(traj)
            else:
                strata["__none__"].append(traj)

        # IS within each stratum, then average weighted by P(z)
        total_trajs = len(trajectories)
        adjusted_value = 0.0

        for z, z_trajs in strata.items():
            p_z = len(z_trajs) / total_trajs
            z_estimate = self.weighted_importance_sampling(
                z_trajs, target_policy, behavior_policy)
            adjusted_value += p_z * z_estimate

        return adjusted_value


# ---------------------------------------------------------------------------
# CausalRewardDecomposition: decompose reward into direct/indirect effects
# ---------------------------------------------------------------------------

class CausalRewardDecomposition:
    """Decompose reward into direct and indirect causal effects.

    Given a causal graph over state variables, decompose the effect of an
    action on reward into:
      - Direct effect: action -> reward (not through any mediator)
      - Indirect effect: action -> mediator -> reward
      - Spurious effect: confounder -> action, confounder -> reward

    This helps understand WHY a policy works and enables better transfer.
    """

    def __init__(self):
        self.state_vars: list[str] = []
        self.mediators: list[str] = []
        self.confounders: list[str] = []

        # Stored estimates
        self._total_effects: dict[str, dict[str, float]] = {}  # state -> action -> TE
        self._direct_effects: dict[str, dict[str, float]] = {}
        self._indirect_effects: dict[str, dict[str, float]] = {}
        self._spurious_effects: dict[str, dict[str, float]] = {}

    def set_mediators(self, mediators: list[str]):
        """Set which state variables are mediators between action and reward."""
        self.mediators = list(mediators)

    def set_confounders(self, confounders: list[str]):
        """Set which context variables are confounders."""
        self.confounders = list(confounders)

    def estimate_effects(self, observations: list[ConfoundedObservation],
                          action: str, baseline_action: str):
        """Estimate total, direct, indirect, and spurious effects.

        Uses observational data to estimate:
          TE = E[R | do(a)] - E[R | do(a0)]
          NDE = E[R_{a, M_{a0}}] - E[R_{a0}]  (natural direct effect)
          NIE = E[R_{a, M_a}] - E[R_{a, M_{a0}}]  (natural indirect effect)
          SE = E[R | a] - E[R | do(a)] (spurious/confounding effect)
        """
        # Group observations
        a_obs = [o for o in observations if o.action == action]
        a0_obs = [o for o in observations if o.action == baseline_action]

        if not a_obs or not a0_obs:
            return

        # Total effect (observational -- may include confounding)
        e_r_a = sum(o.reward for o in a_obs) / len(a_obs)
        e_r_a0 = sum(o.reward for o in a0_obs) / len(a0_obs)
        te_obs = e_r_a - e_r_a0

        # If we have confounders, estimate interventional effect
        if self.confounders:
            # Backdoor adjustment for total effect
            e_r_do_a = self._backdoor_estimate(observations, action)
            e_r_do_a0 = self._backdoor_estimate(observations, baseline_action)
            te_int = e_r_do_a - e_r_do_a0 if (e_r_do_a is not None and e_r_do_a0 is not None) else te_obs
            spurious = te_obs - te_int
        else:
            te_int = te_obs
            spurious = 0.0

        # Direct/indirect decomposition via mediators
        if self.mediators:
            nde, nie = self._mediation_analysis(
                observations, action, baseline_action)
        else:
            nde = te_int
            nie = 0.0

        # Store results keyed by action
        for s in set(o.state for o in observations):
            if s not in self._total_effects:
                self._total_effects[s] = {}
                self._direct_effects[s] = {}
                self._indirect_effects[s] = {}
                self._spurious_effects[s] = {}
            self._total_effects[s][action] = te_int
            self._direct_effects[s][action] = nde
            self._indirect_effects[s][action] = nie
            self._spurious_effects[s][action] = spurious

    def _backdoor_estimate(self, observations: list[ConfoundedObservation],
                            action: str) -> Optional[float]:
        """Estimate E[R|do(a)] using backdoor adjustment over confounders."""
        # Group by confounder values
        groups = defaultdict(list)
        for o in observations:
            z_key = tuple(o.context.get(c) for c in sorted(self.confounders))
            groups[z_key].append(o)

        total = len(observations)
        adjusted = 0.0
        covered = 0

        for z_key, z_obs in groups.items():
            p_z = len(z_obs) / total
            a_obs = [o for o in z_obs if o.action == action]
            if a_obs:
                e_r = sum(o.reward for o in a_obs) / len(a_obs)
                adjusted += e_r * p_z
                covered += 1

        return adjusted if covered > 0 else None

    def _mediation_analysis(self, observations: list[ConfoundedObservation],
                             action: str, baseline_action: str) -> tuple[float, float]:
        """Natural direct and indirect effect estimation via mediation."""
        # NDE: effect of action when mediator held at baseline distribution
        # NIE: effect through mediator

        a_obs = [o for o in observations if o.action == action]
        a0_obs = [o for o in observations if o.action == baseline_action]

        if not a_obs or not a0_obs:
            return 0.0, 0.0

        # Get mediator distributions under each action
        med_dist_a = defaultdict(lambda: defaultdict(int))
        med_dist_a0 = defaultdict(lambda: defaultdict(int))

        for o in a_obs:
            for m in self.mediators:
                if m in o.context:
                    med_dist_a[m][o.context[m]] += 1

        for o in a0_obs:
            for m in self.mediators:
                if m in o.context:
                    med_dist_a0[m][o.context[m]] += 1

        # Simple NDE/NIE via difference of conditional expectations
        # NDE = E[R|a, M=m0] - E[R|a0, M=m0] where m0 is baseline mediator dist
        # NIE = E[R|a, M=m1] - E[R|a, M=m0]

        e_r_a = sum(o.reward for o in a_obs) / len(a_obs)
        e_r_a0 = sum(o.reward for o in a0_obs) / len(a0_obs)
        te = e_r_a - e_r_a0

        # Approximate: NIE = proportion of effect through mediators
        # Use correlation between mediator change and reward change
        if self.mediators and any(med_dist_a[m] for m in self.mediators):
            # Check if mediator distribution shifts with action
            shifts = []
            for m in self.mediators:
                dist_a = med_dist_a[m]
                dist_a0 = med_dist_a0[m]
                if dist_a and dist_a0:
                    # Total variation distance
                    all_vals = set(dist_a.keys()) | set(dist_a0.keys())
                    total_a = sum(dist_a.values())
                    total_a0 = sum(dist_a0.values())
                    if total_a > 0 and total_a0 > 0:
                        tvd = 0.5 * sum(
                            abs(dist_a.get(v, 0) / total_a - dist_a0.get(v, 0) / total_a0)
                            for v in all_vals
                        )
                        shifts.append(tvd)

            if shifts:
                avg_shift = sum(shifts) / len(shifts)
                nie = te * min(avg_shift, 1.0)  # Proportion mediated
                nde = te - nie
            else:
                nde = te
                nie = 0.0
        else:
            nde = te
            nie = 0.0

        return nde, nie

    def get_decomposition(self, state: str, action: str) -> dict[str, float]:
        """Get effect decomposition for (state, action)."""
        return {
            "total": self._total_effects.get(state, {}).get(action, 0.0),
            "direct": self._direct_effects.get(state, {}).get(action, 0.0),
            "indirect": self._indirect_effects.get(state, {}).get(action, 0.0),
            "spurious": self._spurious_effects.get(state, {}).get(action, 0.0),
        }


# ---------------------------------------------------------------------------
# InterventionalPlanner: plan using do-calculus
# ---------------------------------------------------------------------------

class InterventionalPlanner:
    """Plan in MDPs using interventional (do-calculus) distributions.

    Standard planning uses P(s'|s,a). When the transition model is learned
    from confounded data, this gives biased plans. InterventionalPlanner
    uses causal adjustment to compute P(s'|s, do(a)) for planning.
    """

    def __init__(self, mdp: MDP, gamma: float = 0.99):
        self.mdp = mdp
        self.gamma = gamma
        self.confounded_mdp: Optional[ConfoundedMDP] = None
        self._adjusted_transitions: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
        self._adjusted_rewards: dict[tuple[str, str], float] = {}

    def set_confounded_data(self, confounded_mdp: ConfoundedMDP):
        """Set the confounded observational data source."""
        self.confounded_mdp = confounded_mdp

    def compute_adjusted_model(self, adjustment_var: str):
        """Compute the adjusted (deconfounded) MDP transition model.

        Uses backdoor adjustment on the confounded data to estimate
        P(s'|s, do(a)) and E[R|s, do(a)].
        """
        if not self.confounded_mdp:
            return

        # Get all observed states and actions
        states = set()
        actions = set()
        for o in self.confounded_mdp.observations:
            states.add(o.state)
            actions.add(o.action)

        for s in states:
            for a in actions:
                # Adjusted transition probabilities
                s_obs = [o for o in self.confounded_mdp.observations
                         if o.state == s and o.action == a]
                if not s_obs:
                    continue

                # Group by adjustment variable
                groups = defaultdict(list)
                all_s_obs = [o for o in self.confounded_mdp.observations
                             if o.state == s]
                for o in all_s_obs:
                    z = o.context.get(adjustment_var)
                    if z is not None:
                        groups[z].append(o)

                total_s = sum(len(v) for v in groups.values())

                # Adjusted transition: P(s'|s,do(a)) = sum_z P(s'|s,a,z) P(z|s)
                adj_trans = defaultdict(float)
                adj_reward = 0.0

                for z, z_obs in groups.items():
                    p_z = len(z_obs) / total_s if total_s > 0 else 0.0
                    z_a_obs = [o for o in z_obs if o.action == a]
                    if not z_a_obs:
                        continue
                    n_za = len(z_a_obs)
                    for o in z_a_obs:
                        adj_trans[o.next_state] += p_z / n_za
                        adj_reward += p_z * o.reward / n_za

                if adj_trans:
                    self._adjusted_transitions[(s, a)] = [
                        (ns, p, 0.0) for ns, p in adj_trans.items()
                    ]
                    self._adjusted_rewards[(s, a)] = adj_reward

    def plan(self, epsilon: float = 1e-8, max_iter: int = 10000) -> CausalRLResult:
        """Plan using adjusted transition model via value iteration.

        Uses P(s'|s, do(a)) instead of P(s'|s, a) for Bellman updates.
        """
        # Build adjusted MDP
        adj_mdp = MDP(self.mdp.name + "_adjusted")

        # Add all states and actions from original MDP
        for s in self.mdp.states:
            adj_mdp.add_state(s, s in self.mdp.terminal_states)
        for a in self.mdp.actions:
            adj_mdp.add_action(a)
        if self.mdp.initial_state:
            adj_mdp.set_initial(self.mdp.initial_state)

        # Use adjusted transitions where available, original otherwise
        for s in self.mdp.states:
            if s in self.mdp.terminal_states:
                continue
            for a in self.mdp.actions:
                if (s, a) in self._adjusted_transitions:
                    for ns, p, _ in self._adjusted_transitions[(s, a)]:
                        reward = self._adjusted_rewards.get((s, a), 0.0)
                        adj_mdp.add_transition(s, a, ns, p, reward)
                else:
                    for ns, p, r in self.mdp.get_transitions(s, a):
                        adj_mdp.add_transition(s, a, ns, p, r)

        # Solve adjusted MDP
        result = value_iteration(adj_mdp, self.gamma, epsilon, max_iter)

        return CausalRLResult(
            policy=result.policy,
            values=result.values,
            q_values=result.q_values or {},
            iterations=result.iterations,
            converged=result.converged,
            causal_adjustments=len(self._adjusted_transitions),
        )


# ---------------------------------------------------------------------------
# CausalTransferRL: transfer causal knowledge across environments
# ---------------------------------------------------------------------------

class CausalTransferRL:
    """Transfer causal knowledge between source and target MDPs.

    When the causal structure (graph) is shared but parameters differ,
    we can transfer:
      - Causal graph structure (which variables affect which)
      - Intervention effectiveness (which actions have causal effects)
      - Invariant causal mechanisms (structural equations that don't change)

    This enables faster learning in the target environment.
    """

    def __init__(self):
        self.source_graph: dict[str, list[str]] = {}  # var -> parents
        self.source_effects: dict[str, dict[str, float]] = {}  # action -> var -> effect
        self.invariant_mechanisms: set[str] = set()  # Vars with stable structural eqs
        self._transferred_q_init: dict[str, dict[str, float]] = {}

    def learn_source(self, causal_mdp: CausalMDP, source_result: MDPResult):
        """Learn transferable causal knowledge from source environment."""
        # Extract causal graph
        self.source_graph = dict(causal_mdp.causal_parents)

        # Extract which actions causally affect which variables
        # by comparing transitions under different actions
        states = causal_mdp._enumerate_states()
        for a in causal_mdp.action_domain:
            self.source_effects[a] = {}
            for s in states[:5]:  # Sample a few states
                trans = causal_mdp._compute_transition(s, a)
                if trans:
                    # Check which state variables change
                    s_vars = CausalState.from_key(s).variables
                    for ns, p, _ in trans:
                        if p > 0.1:
                            ns_vars = CausalState.from_key(ns).variables
                            for v in s_vars:
                                if v in ns_vars and ns_vars[v] != s_vars[v]:
                                    self.source_effects[a][v] = \
                                        self.source_effects[a].get(v, 0) + 1

    def identify_invariant_mechanisms(self, source_obs: list[ConfoundedObservation],
                                       target_obs: list[ConfoundedObservation],
                                       threshold: float = 0.1):
        """Identify which causal mechanisms are invariant across environments.

        A mechanism is invariant if P(Y|parents(Y)) is similar in source and target.
        """
        # Compare transition distributions for each variable
        source_trans = self._compute_var_transitions(source_obs)
        target_trans = self._compute_var_transitions(target_obs)

        for var in source_trans:
            if var in target_trans:
                # Compare distributions
                tvd = self._total_variation_distance(
                    source_trans[var], target_trans[var])
                if tvd < threshold:
                    self.invariant_mechanisms.add(var)

    def _compute_var_transitions(self, obs: list[ConfoundedObservation]) -> dict[str, dict]:
        """Compute empirical variable transition distributions."""
        trans = defaultdict(lambda: defaultdict(int))
        for o in obs:
            # Parse state keys to get variable changes
            s_vars = CausalState.from_key(o.state).variables
            ns_vars = CausalState.from_key(o.next_state).variables
            for v in s_vars:
                if v in ns_vars:
                    key = (s_vars[v], o.action, ns_vars[v])
                    trans[v][key] += 1
        return dict(trans)

    def _total_variation_distance(self, dist1: dict, dist2: dict) -> float:
        """Total variation distance between two empirical distributions."""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        t1 = sum(dist1.values())
        t2 = sum(dist2.values())
        if t1 == 0 or t2 == 0:
            return 1.0
        return 0.5 * sum(
            abs(dist1.get(k, 0) / t1 - dist2.get(k, 0) / t2)
            for k in all_keys
        )

    def transfer_q_init(self, source_q: dict[str, dict[str, float]],
                         state_mapping: dict[str, str] = None) -> dict[str, dict[str, float]]:
        """Transfer Q-values from source to target as initialization.

        Uses causal structure to determine which Q-values are transferable.
        Only transfers Q-values for (state, action) pairs where the
        causal mechanism is identified as invariant.

        Args:
            source_q: Q-values from source environment
            state_mapping: optional source_state -> target_state mapping
        """
        transferred = {}
        for s, aq in source_q.items():
            target_s = state_mapping.get(s, s) if state_mapping else s
            transferred[target_s] = {}
            for a, q in aq.items():
                # Check if this action's causal effects are invariant
                affected_vars = self.source_effects.get(a, {})
                if not affected_vars or all(v in self.invariant_mechanisms
                                             for v in affected_vars):
                    transferred[target_s][a] = q
                else:
                    transferred[target_s][a] = 0.0  # Don't transfer

        self._transferred_q_init = transferred
        return transferred


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compare_policies(mdp: MDP, causal_policy: dict[str, str],
                      naive_policy: dict[str, str],
                      gamma: float = 0.99) -> dict[str, float]:
    """Compare causal and naive policies by advantage per state.

    Returns advantage: V_causal(s) - V_naive(s) for each state.
    """
    from markov_decision_processes import _evaluate_policy

    v_causal = _evaluate_policy(mdp, causal_policy, gamma)
    v_naive = _evaluate_policy(mdp, naive_policy, gamma)

    advantage = {}
    for s in set(list(v_causal.keys()) + list(v_naive.keys())):
        advantage[s] = v_causal.get(s, 0.0) - v_naive.get(s, 0.0)

    return advantage


def build_confounded_treatment_mdp() -> tuple[MDP, ConfoundedMDP]:
    """Example: Medical treatment MDP with confounding.

    States: healthy, mild, severe
    Actions: treat, wait
    Confounder: patient type (strong, weak) affects both treatment choice
    and outcome.

    Strong patients are more likely to be treated AND more likely to recover,
    creating confounding: treatment appears more effective than it is.
    """
    mdp = MDP("ConfoundedTreatment")
    mdp.add_state("healthy", terminal=True)
    mdp.add_state("mild")
    mdp.add_state("severe")
    mdp.set_initial("mild")

    mdp.add_action("treat")
    mdp.add_action("wait")

    # True (interventional) transitions
    mdp.add_transition("mild", "treat", "healthy", 0.6, 1.0)
    mdp.add_transition("mild", "treat", "mild", 0.3, 0.0)
    mdp.add_transition("mild", "treat", "severe", 0.1, -1.0)

    mdp.add_transition("mild", "wait", "healthy", 0.3, 1.0)
    mdp.add_transition("mild", "wait", "mild", 0.4, 0.0)
    mdp.add_transition("mild", "wait", "severe", 0.3, -1.0)

    mdp.add_transition("severe", "treat", "healthy", 0.3, 1.0)
    mdp.add_transition("severe", "treat", "mild", 0.4, 0.0)
    mdp.add_transition("severe", "treat", "severe", 0.3, -1.0)

    mdp.add_transition("severe", "wait", "healthy", 0.1, 1.0)
    mdp.add_transition("severe", "wait", "mild", 0.3, 0.0)
    mdp.add_transition("severe", "wait", "severe", 0.6, -1.0)

    # Create confounded observational data
    conf = ConfoundedMDP(mdp, "ConfoundedTreatment")

    # Strong patients: treated more often, better outcomes regardless
    rng = random.Random(42)
    for _ in range(200):
        patient_type = "strong"
        state = rng.choice(["mild", "severe"])
        # Strong patients treated 80% of the time (confounding!)
        action = "treat" if rng.random() < 0.8 else "wait"
        # Strong patients have better outcomes
        if action == "treat":
            if state == "mild":
                r = rng.choice([1.0, 1.0, 1.0, 0.0, -1.0])  # 60% good
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
            else:
                r = rng.choice([1.0, 1.0, 0.0, 0.0, -1.0])  # 40% good
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
        else:
            if state == "mild":
                r = rng.choice([1.0, 1.0, 0.0, 0.0, -1.0])  # 40% good naturally
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
            else:
                r = rng.choice([1.0, 0.0, 0.0, -1.0, -1.0])  # 20% good
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
        conf.add_observation(state, action, ns, r,
                              {"patient_type": patient_type})

    # Weak patients: treated less often, worse outcomes regardless
    for _ in range(200):
        patient_type = "weak"
        state = rng.choice(["mild", "severe"])
        # Weak patients treated only 30% of the time
        action = "treat" if rng.random() < 0.3 else "wait"
        # Weak patients have worse outcomes
        if action == "treat":
            if state == "mild":
                r = rng.choice([1.0, 0.0, 0.0, -1.0, -1.0])  # 20% good
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
            else:
                r = rng.choice([0.0, 0.0, -1.0, -1.0, -1.0])  # 0% good
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
        else:
            if state == "mild":
                r = rng.choice([0.0, 0.0, -1.0, -1.0, -1.0])  # 0% good
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
            else:
                r = rng.choice([0.0, -1.0, -1.0, -1.0, -1.0])  # 0% good
                ns = "healthy" if r > 0 else ("mild" if r == 0 else "severe")
        conf.add_observation(state, action, ns, r,
                              {"patient_type": patient_type})

    return mdp, conf


def build_causal_gridworld() -> CausalMDP:
    """Example: Causal gridworld where position variables have causal structure.

    State: (x, y) position
    Actions: up, down, left, right
    Causal structure: action -> x' (or y'), terrain -> reward
    """
    cmdp = CausalMDP("CausalGridworld")

    cmdp.add_state_var("x", [0, 1, 2])
    cmdp.add_state_var("y", [0, 1, 2])
    cmdp.set_action_var("action", ["up", "down", "left", "right"])

    # Causal edges: action affects next position
    cmdp.add_edge("action", "x")
    cmdp.add_edge("action", "y")
    # Current position affects next position
    cmdp.add_edge("x", "x")
    cmdp.add_edge("y", "y")

    # Structural equations
    def x_transition(parents):
        x = parents.get("x", 0)
        action = parents.get("action", "")
        if action == "right":
            new_x = min(x + 1, 2)
        elif action == "left":
            new_x = max(x - 1, 0)
        else:
            new_x = x
        return {new_x: 1.0}

    def y_transition(parents):
        y = parents.get("y", 0)
        action = parents.get("action", "")
        if action == "up":
            new_y = min(y + 1, 2)
        elif action == "down":
            new_y = max(y - 1, 0)
        else:
            new_y = y
        return {new_y: 1.0}

    cmdp.set_structural_eq("x", x_transition)
    cmdp.set_structural_eq("y", y_transition)

    # Reward: reaching (2,2) is goal
    def reward_fn(state_vars, action):
        if state_vars.get("x") == 2 and state_vars.get("y") == 2:
            return 1.0
        return -0.1

    cmdp.set_reward_fn(reward_fn)

    return cmdp


def build_confounded_bandit_mdp() -> tuple[CausalMDP, dict]:
    """Example: Single-state MDP (bandit) with confounding.

    Two actions: drug_A, drug_B
    Confounder: severity (hidden from agent) affects both
    doctor's choice and patient outcome.
    """
    cmdp = CausalMDP("ConfoundedBandit")
    cmdp.add_state_var("health", [0, 1, 2])  # 0=sick, 1=recovering, 2=healthy
    cmdp.set_action_var("treatment", ["drug_A", "drug_B"])

    cmdp.add_edge("treatment", "health")
    cmdp.add_edge("health", "health")

    # Confounder: severity (unobserved)
    cmdp.add_confounder("severity", [0, 1], {0: 0.5, 1: 0.5},
                         affects=["health"])

    def health_transition(parents):
        h = parents.get("health", 0)
        treatment = parents.get("treatment", "drug_A")
        severity = parents.get("severity", 0)

        # Use defaultdict to avoid duplicate-key overwrites when
        # min(h+1,2) == h or max(h-1,0) == h at boundaries.
        from collections import defaultdict as _dd
        dist = _dd(float)

        if treatment == "drug_A":
            if severity == 0:  # Low severity
                dist[min(h + 1, 2)] += 0.7
                dist[h] += 0.3
            else:  # High severity
                dist[min(h + 1, 2)] += 0.3
                dist[h] += 0.5
                dist[max(h - 1, 0)] += 0.2
        else:  # drug_B
            if severity == 0:
                dist[min(h + 1, 2)] += 0.5
                dist[h] += 0.5
            else:
                dist[min(h + 1, 2)] += 0.6
                dist[h] += 0.3
                dist[max(h - 1, 0)] += 0.1

        return dict(dist)

    cmdp.set_structural_eq("health", health_transition)

    def reward_fn(state_vars, action):
        h = state_vars.get("health", 0)
        return {0: -1.0, 1: 0.0, 2: 1.0}.get(h, 0.0)

    cmdp.set_reward_fn(reward_fn)

    # Behavior policy (confounded): doctors give drug_A to low-severity patients
    behavior = {
        "severity_0": {"drug_A": 0.8, "drug_B": 0.2},
        "severity_1": {"drug_A": 0.3, "drug_B": 0.7},
    }

    return cmdp, behavior
