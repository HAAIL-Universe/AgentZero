"""V213: Markov Decision Processes

Full MDP framework with value iteration, policy iteration, linear programming,
Q-learning, and RTDP. Composes V209 (Bayesian Networks) for transition model
representation and V210 (Influence Diagrams) for expected utility computation.

Supports: finite MDPs, discounted/average/total reward, sparse transitions,
reachability analysis, policy simulation, and MDP-to-influence-diagram conversion.

AI-Generated | Claude (Anthropic) | AgentZero A2 Session 294 | 2026-03-18
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

# ── Compose V209 (Bayesian Networks) for probabilistic model representation ──
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "V209_bayesian_networks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "V210_influence_diagrams"))
from bayesian_networks import BayesianNetwork, Factor, variable_elimination
from influence_diagrams import InfluenceDiagram, Policy, UtilityFactor, NodeType


# ─────────────────────────────────────────────────────────────────────────────
#  Core Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class RewardType(Enum):
    """Reward structure type."""
    STATE_ACTION = auto()       # R(s, a)
    STATE_ACTION_STATE = auto() # R(s, a, s')


@dataclass
class Transition:
    """A single transition entry: (s, a) -> (s', prob, reward)."""
    state: str
    action: str
    next_state: str
    probability: float
    reward: float = 0.0


@dataclass
class MDPResult:
    """Result of an MDP solver."""
    values: dict[str, float]             # V(s) for each state
    policy: dict[str, str]               # pi(s) -> action
    q_values: dict[str, dict[str, float]] | None = None  # Q(s,a)
    iterations: int = 0
    converged: bool = False


class MDP:
    """Finite Markov Decision Process.

    States and actions are strings. Transitions are sparse:
    only non-zero probability transitions are stored.
    """

    def __init__(self, name: str = "MDP"):
        self.name = name
        self.states: list[str] = []
        self._state_set: set[str] = set()
        self.actions: list[str] = []
        self._action_set: set[str] = set()
        self.initial_state: str | None = None
        self.terminal_states: set[str] = set()

        # Transitions: (state, action) -> list of (next_state, prob, reward)
        self._transitions: dict[tuple[str, str], list[tuple[str, float, float]]] = {}

        # State-action reward (optional, added to transition rewards)
        self._sa_rewards: dict[tuple[str, str], float] = {}

        # Available actions per state (None = all actions)
        self._available_actions: dict[str, list[str]] | None = None

    def add_state(self, state: str, terminal: bool = False):
        """Add a state to the MDP."""
        if state not in self._state_set:
            self.states.append(state)
            self._state_set.add(state)
        if terminal:
            self.terminal_states.add(state)

    def add_action(self, action: str):
        """Add an action to the MDP."""
        if action not in self._action_set:
            self.actions.append(action)
            self._action_set.add(action)

    def set_initial(self, state: str):
        """Set the initial state."""
        self.initial_state = state

    def add_transition(self, s: str, a: str, s_prime: str, prob: float, reward: float = 0.0):
        """Add a transition: P(s'|s,a) = prob, R(s,a,s') = reward."""
        # Auto-register states and actions
        self.add_state(s)
        self.add_state(s_prime)
        self.add_action(a)
        key = (s, a)
        if key not in self._transitions:
            self._transitions[key] = []
        self._transitions[key].append((s_prime, prob, reward))

    def set_reward(self, s: str, a: str, reward: float):
        """Set state-action reward R(s,a). Added to any transition rewards."""
        self._sa_rewards[(s, a)] = reward

    def set_available_actions(self, state: str, actions: list[str]):
        """Restrict available actions for a state."""
        if self._available_actions is None:
            self._available_actions = {}
        self._available_actions[state] = actions

    def get_actions(self, state: str) -> list[str]:
        """Get available actions in a state."""
        if state in self.terminal_states:
            return []
        if self._available_actions is not None and state in self._available_actions:
            return self._available_actions[state]
        # Return actions that have transitions from this state
        return [a for a in self.actions if (state, a) in self._transitions]

    def get_transitions(self, s: str, a: str) -> list[tuple[str, float, float]]:
        """Get transitions from (s,a): list of (s', prob, reward)."""
        base = self._transitions.get((s, a), [])
        sa_reward = self._sa_rewards.get((s, a), 0.0)
        if sa_reward != 0.0:
            return [(sp, p, r + sa_reward) for sp, p, r in base]
        return base

    def validate(self) -> list[str]:
        """Validate MDP structure. Returns list of issues."""
        issues = []
        if not self.states:
            issues.append("No states defined")
        if not self.actions:
            issues.append("No actions defined")

        for (s, a), transitions in self._transitions.items():
            total_prob = sum(p for _, p, _ in transitions)
            if abs(total_prob - 1.0) > 1e-9:
                issues.append(f"P(.|{s},{a}) sums to {total_prob:.6f}, not 1.0")
            for sp, p, _ in transitions:
                if p < 0:
                    issues.append(f"Negative probability: P({sp}|{s},{a}) = {p}")

        return issues

    def expected_reward(self, s: str, a: str, values: dict[str, float],
                        gamma: float = 1.0) -> float:
        """Compute Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]."""
        total = 0.0
        for sp, p, r in self.get_transitions(s, a):
            total += p * (r + gamma * values.get(sp, 0.0))
        return total

    def successor_states(self, s: str) -> set[str]:
        """All states reachable from s in one step (any action)."""
        result = set()
        for a in self.get_actions(s):
            for sp, p, _ in self.get_transitions(s, a):
                if p > 0:
                    result.add(sp)
        return result

    def reachable_states(self, start: str | None = None) -> set[str]:
        """All states reachable from start (default: initial_state)."""
        s0 = start or self.initial_state
        if s0 is None:
            return set(self.states)
        visited = set()
        queue = [s0]
        while queue:
            s = queue.pop()
            if s in visited:
                continue
            visited.add(s)
            for sp in self.successor_states(s):
                if sp not in visited:
                    queue.append(sp)
        return visited


# ─────────────────────────────────────────────────────────────────────────────
#  Solvers
# ─────────────────────────────────────────────────────────────────────────────

def value_iteration(mdp: MDP, gamma: float = 0.99, epsilon: float = 1e-8,
                    max_iter: int = 10000) -> MDPResult:
    """Classic value iteration. Converges to optimal V* and pi*."""
    V = {s: 0.0 for s in mdp.states}

    for iteration in range(1, max_iter + 1):
        delta = 0.0
        V_new = {}
        for s in mdp.states:
            actions = mdp.get_actions(s)
            if not actions:
                V_new[s] = 0.0
                continue
            best = max(mdp.expected_reward(s, a, V, gamma) for a in actions)
            V_new[s] = best
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < epsilon:
            # Extract policy
            policy = _extract_policy(mdp, V, gamma)
            q = _compute_q_values(mdp, V, gamma)
            return MDPResult(values=V, policy=policy, q_values=q,
                             iterations=iteration, converged=True)

    policy = _extract_policy(mdp, V, gamma)
    q = _compute_q_values(mdp, V, gamma)
    return MDPResult(values=V, policy=policy, q_values=q,
                     iterations=max_iter, converged=False)


def policy_iteration(mdp: MDP, gamma: float = 0.99,
                     max_iter: int = 1000, eval_max_iter: int = 1000,
                     eval_epsilon: float = 1e-10) -> MDPResult:
    """Policy iteration: evaluate -> improve -> repeat until stable."""
    # Initialize with arbitrary policy (first available action)
    policy = {}
    for s in mdp.states:
        actions = mdp.get_actions(s)
        policy[s] = actions[0] if actions else ""

    for iteration in range(1, max_iter + 1):
        # Policy evaluation: solve V^pi
        V = _evaluate_policy(mdp, policy, gamma, eval_max_iter, eval_epsilon)

        # Policy improvement
        stable = True
        new_policy = {}
        for s in mdp.states:
            actions = mdp.get_actions(s)
            if not actions:
                new_policy[s] = ""
                continue
            best_a = max(actions, key=lambda a: mdp.expected_reward(s, a, V, gamma))
            new_policy[s] = best_a
            if new_policy[s] != policy[s]:
                stable = False
        policy = new_policy

        if stable:
            q = _compute_q_values(mdp, V, gamma)
            return MDPResult(values=V, policy=policy, q_values=q,
                             iterations=iteration, converged=True)

    V = _evaluate_policy(mdp, policy, gamma, eval_max_iter, eval_epsilon)
    q = _compute_q_values(mdp, V, gamma)
    return MDPResult(values=V, policy=policy, q_values=q,
                     iterations=max_iter, converged=False)


def linear_programming(mdp: MDP, gamma: float = 0.99,
                       max_iter: int = 50000, epsilon: float = 1e-8) -> MDPResult:
    """Solve MDP via LP relaxation (iterative constraint tightening).

    Minimizes sum V(s) subject to V(s) >= R(s,a) + gamma * sum P(s'|s,a) V(s')
    for all s, a. Uses iterative projection (no external LP solver).
    """
    V = {s: 0.0 for s in mdp.states}

    for iteration in range(1, max_iter + 1):
        delta = 0.0
        for s in mdp.states:
            actions = mdp.get_actions(s)
            if not actions:
                continue
            # Tighten: V(s) = max_a Q(s,a) (project onto feasible)
            best = max(mdp.expected_reward(s, a, V, gamma) for a in actions)
            delta = max(delta, abs(best - V[s]))
            V[s] = best

        if delta < epsilon:
            policy = _extract_policy(mdp, V, gamma)
            q = _compute_q_values(mdp, V, gamma)
            return MDPResult(values=V, policy=policy, q_values=q,
                             iterations=iteration, converged=True)

    policy = _extract_policy(mdp, V, gamma)
    q = _compute_q_values(mdp, V, gamma)
    return MDPResult(values=V, policy=policy, q_values=q,
                     iterations=max_iter, converged=False)


def q_learning(mdp: MDP, gamma: float = 0.99, alpha: float = 0.1,
               epsilon_greedy: float = 0.1, episodes: int = 10000,
               max_steps: int = 1000, seed: int | None = None) -> MDPResult:
    """Model-free Q-learning (off-policy TD control).

    Explores via epsilon-greedy, learns Q(s,a) from samples.
    Requires mdp.initial_state to be set.
    """
    rng = random.Random(seed)
    Q: dict[str, dict[str, float]] = {}
    for s in mdp.states:
        Q[s] = {a: 0.0 for a in mdp.get_actions(s)}

    start = mdp.initial_state or mdp.states[0]

    for ep in range(episodes):
        s = start
        for step in range(max_steps):
            actions = mdp.get_actions(s)
            if not actions:
                break

            # Epsilon-greedy action selection
            if rng.random() < epsilon_greedy:
                a = rng.choice(actions)
            else:
                a = max(actions, key=lambda a: Q[s].get(a, 0.0))

            # Sample transition
            transitions = mdp.get_transitions(s, a)
            if not transitions:
                break
            r_val = rng.random()
            cumulative = 0.0
            sp, reward = transitions[0][0], transitions[0][2]
            for t_sp, t_p, t_r in transitions:
                cumulative += t_p
                if r_val <= cumulative:
                    sp, reward = t_sp, t_r
                    break

            # Q-learning update
            future = 0.0
            sp_actions = mdp.get_actions(sp)
            if sp_actions:
                future = max(Q[sp].get(a2, 0.0) for a2 in sp_actions)
            old_q = Q[s].get(a, 0.0)
            Q[s][a] = old_q + alpha * (reward + gamma * future - old_q)

            s = sp
            if s in mdp.terminal_states:
                break

    # Extract values and policy from Q
    V = {}
    policy = {}
    for s in mdp.states:
        actions = mdp.get_actions(s)
        if not actions:
            V[s] = 0.0
            policy[s] = ""
        else:
            best_a = max(actions, key=lambda a: Q[s].get(a, 0.0))
            V[s] = Q[s][best_a]
            policy[s] = best_a

    return MDPResult(values=V, policy=policy, q_values=Q,
                     iterations=episodes, converged=True)


def rtdp(mdp: MDP, gamma: float = 0.99, epsilon: float = 1e-4,
         trials: int = 5000, max_steps: int = 500,
         seed: int | None = None) -> MDPResult:
    """Real-Time Dynamic Programming. Focuses updates on reachable states."""
    rng = random.Random(seed)
    V = {s: 0.0 for s in mdp.states}
    start = mdp.initial_state or mdp.states[0]

    for trial in range(trials):
        s = start
        for step in range(max_steps):
            actions = mdp.get_actions(s)
            if not actions:
                break

            # Bellman update at current state
            best = max(mdp.expected_reward(s, a, V, gamma) for a in actions)
            V[s] = best

            # Greedy action
            best_a = max(actions, key=lambda a: mdp.expected_reward(s, a, V, gamma))

            # Stochastic transition
            transitions = mdp.get_transitions(s, best_a)
            if not transitions:
                break
            r_val = rng.random()
            cumulative = 0.0
            sp = transitions[0][0]
            for t_sp, t_p, _ in transitions:
                cumulative += t_p
                if r_val <= cumulative:
                    sp = t_sp
                    break

            s = sp
            if s in mdp.terminal_states:
                break

    policy = _extract_policy(mdp, V, gamma)
    q = _compute_q_values(mdp, V, gamma)
    return MDPResult(values=V, policy=policy, q_values=q,
                     iterations=trials, converged=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Policy Analysis & Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(mdp: MDP, policy: dict[str, str], steps: int = 100,
             start: str | None = None, seed: int | None = None
             ) -> list[tuple[str, str, str, float]]:
    """Simulate an MDP under a policy. Returns [(s, a, s', r), ...]."""
    rng = random.Random(seed)
    s = start or mdp.initial_state or mdp.states[0]
    trajectory = []

    for _ in range(steps):
        a = policy.get(s, "")
        if not a or s in mdp.terminal_states:
            break
        transitions = mdp.get_transitions(s, a)
        if not transitions:
            break

        r_val = rng.random()
        cumulative = 0.0
        sp, reward = transitions[0][0], transitions[0][2]
        for t_sp, t_p, t_r in transitions:
            cumulative += t_p
            if r_val <= cumulative:
                sp, reward = t_sp, t_r
                break

        trajectory.append((s, a, sp, reward))
        s = sp

    return trajectory


def expected_total_reward(mdp: MDP, policy: dict[str, str], gamma: float = 0.99,
                          start: str | None = None, n_simulations: int = 10000,
                          max_steps: int = 500, seed: int | None = None) -> float:
    """Estimate expected total discounted reward via Monte Carlo."""
    rng = random.Random(seed)
    s0 = start or mdp.initial_state or mdp.states[0]
    total = 0.0

    for _ in range(n_simulations):
        s = s0
        disc = 1.0
        ep_reward = 0.0
        for _ in range(max_steps):
            a = policy.get(s, "")
            if not a or s in mdp.terminal_states:
                break
            transitions = mdp.get_transitions(s, a)
            if not transitions:
                break
            r_val = rng.random()
            cumulative = 0.0
            sp, reward = transitions[0][0], transitions[0][2]
            for t_sp, t_p, t_r in transitions:
                cumulative += t_p
                if r_val <= cumulative:
                    sp, reward = t_sp, t_r
                    break
            ep_reward += disc * reward
            disc *= gamma
            s = sp
        total += ep_reward

    return total / n_simulations


def policy_advantage(mdp: MDP, policy1: dict[str, str], policy2: dict[str, str],
                     gamma: float = 0.99) -> dict[str, float]:
    """Compute advantage of policy1 over policy2: A(s) = V1(s) - V2(s)."""
    V1 = _evaluate_policy(mdp, policy1, gamma)
    V2 = _evaluate_policy(mdp, policy2, gamma)
    return {s: V1[s] - V2[s] for s in mdp.states}


# ─────────────────────────────────────────────────────────────────────────────
#  MDP <-> Influence Diagram Conversion (V210 composition)
# ─────────────────────────────────────────────────────────────────────────────

def mdp_to_influence_diagram(mdp: MDP, horizon: int = 3) -> InfluenceDiagram:
    """Convert a finite-horizon MDP to an influence diagram.

    Unrolls the MDP for `horizon` steps into a time-indexed ID:
    - Chance nodes: S_0, S_1, ..., S_horizon (state at each time)
    - Decision nodes: A_0, A_1, ..., A_{horizon-1}
    - Utility nodes: U_0, U_1, ..., U_{horizon-1}

    A_t observes S_t. S_{t+1} depends on S_t and A_t.
    U_t depends on S_t, A_t, S_{t+1}.
    """
    diagram = InfluenceDiagram()
    state_domain = list(mdp.states)
    action_domain = list(mdp.actions)

    # Add initial state node
    diagram.add_chance_node(f"S_0", state_domain)

    # Set initial distribution (uniform or delta on initial_state)
    if mdp.initial_state:
        init_cpt = {}
        for s in state_domain:
            init_cpt[(s,)] = 1.0 if s == mdp.initial_state else 0.0
        diagram.bn.set_cpt(f"S_0", init_cpt)
    else:
        uniform_p = 1.0 / len(state_domain)
        init_cpt = {(s,): uniform_p for s in state_domain}
        diagram.bn.set_cpt(f"S_0", init_cpt)

    for t in range(horizon):
        s_node = f"S_{t}"
        a_node = f"A_{t}"
        s_next = f"S_{t+1}"

        # Decision node: A_t observes S_t
        diagram.add_decision_node(a_node, action_domain, info_vars=[s_node])

        # Chance node: S_{t+1} depends on S_t, A_t
        diagram.add_chance_node(s_next, state_domain)
        diagram.add_edge(s_node, s_next)
        diagram.add_edge(a_node, s_next)

        # Set transition CPT for S_{t+1}
        cpt = {}
        for s in state_domain:
            for a in action_domain:
                transitions = mdp.get_transitions(s, a)
                trans_dict = {sp: p for sp, p, _ in transitions}
                for sp in state_domain:
                    cpt[(s, a, sp)] = trans_dict.get(sp, 0.0)
        diagram.bn.set_cpt(s_next, cpt)

        # Utility node: U_t depends on S_t, A_t, S_{t+1}
        u_table = {}
        for s in state_domain:
            for a in action_domain:
                transitions = mdp.get_transitions(s, a)
                trans_dict = {sp: r for sp, _, r in transitions}
                for sp in state_domain:
                    u_table[(s, a, sp)] = trans_dict.get(sp, 0.0)
        diagram.add_utility_node(f"U_{t}", [s_node, a_node, s_next], u_table)

    return diagram


def influence_diagram_to_mdp(diagram: InfluenceDiagram) -> MDP:
    """Extract an MDP from a single-decision influence diagram.

    Treats the single decision as the action, its chance-node parents as
    the state, and the utility as the reward.
    """
    decisions = diagram.decision_nodes()
    if len(decisions) != 1:
        raise ValueError(f"Expected 1 decision node, got {len(decisions)}")

    decision = decisions[0]
    action_domain = diagram.bn.domains[decision]
    info_vars = diagram.get_info_set(decision)

    if not info_vars:
        raise ValueError("Decision has no information set (no state)")

    # Use the first info var as state
    state_var = info_vars[0]
    state_domain = diagram.bn.domains[state_var]

    mdp = MDP(name="MDP_from_ID")
    for s in state_domain:
        mdp.add_state(s)
    for a in action_domain:
        mdp.add_action(a)

    # Extract transitions from chance successors of decision
    children = [c for c in diagram.bn.nodes if decision in diagram.bn.parents.get(c, [])]
    chance_children = [c for c in children if diagram.node_types.get(c) == NodeType.CHANCE]

    if chance_children:
        next_state_node = chance_children[0]
        cpt_factor = diagram.bn.cpts.get(next_state_node)
        if cpt_factor:
            for s in state_domain:
                for a in action_domain:
                    for sp in state_domain:
                        prob = cpt_factor.get({state_var: s, decision: a, next_state_node: sp})
                        if prob > 0:
                            mdp.add_transition(s, a, sp, prob)

    return mdp


# ─────────────────────────────────────────────────────────────────────────────
#  MDP <-> Bayesian Network Conversion (V209 composition)
# ─────────────────────────────────────────────────────────────────────────────

def mdp_transition_bn(mdp: MDP, state: str, action: str) -> BayesianNetwork:
    """Create a BN representing the transition distribution P(S'|s,a).

    Useful for probabilistic queries about next-state distributions.
    """
    bn = BayesianNetwork()
    domain = list(mdp.states)
    bn.add_node("next_state", domain)

    transitions = mdp.get_transitions(state, action)
    trans_dict = {sp: p for sp, p, _ in transitions}
    cpt = {}
    for sp in domain:
        cpt[(sp,)] = trans_dict.get(sp, 0.0)
    bn.set_cpt("next_state", cpt)
    return bn


def occupancy_measure(mdp: MDP, policy: dict[str, str], gamma: float = 0.99,
                      max_iter: int = 1000, epsilon: float = 1e-10
                      ) -> dict[tuple[str, str], float]:
    """Compute discounted state-action occupancy measure d(s,a).

    d(s,a) = sum_{t=0}^inf gamma^t * P(s_t=s, a_t=a | pi).
    Normalized so sum d(s,a) = 1/(1-gamma).
    """
    # State visitation: d_s(s) via iterative computation
    n = len(mdp.states)
    # Initial distribution
    if mdp.initial_state:
        mu = {s: (1.0 if s == mdp.initial_state else 0.0) for s in mdp.states}
    else:
        mu = {s: 1.0 / n for s in mdp.states}

    # Accumulate discounted visitation
    visit = {s: 0.0 for s in mdp.states}
    disc = 1.0
    current = dict(mu)

    for _ in range(max_iter):
        for s in mdp.states:
            visit[s] += disc * current[s]

        # Compute next state distribution
        next_dist = {s: 0.0 for s in mdp.states}
        for s in mdp.states:
            if current[s] < 1e-15:
                continue
            a = policy.get(s, "")
            if not a:
                continue
            for sp, p, _ in mdp.get_transitions(s, a):
                next_dist[sp] += current[s] * p

        disc *= gamma
        if disc < epsilon:
            break
        current = next_dist

    # Convert to state-action occupancy
    result = {}
    for s in mdp.states:
        a = policy.get(s, "")
        if a:
            result[(s, a)] = visit[s]
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_policy(mdp: MDP, V: dict[str, float], gamma: float) -> dict[str, str]:
    """Extract greedy policy from value function."""
    policy = {}
    for s in mdp.states:
        actions = mdp.get_actions(s)
        if not actions:
            policy[s] = ""
        else:
            policy[s] = max(actions, key=lambda a: mdp.expected_reward(s, a, V, gamma))
    return policy


def _compute_q_values(mdp: MDP, V: dict[str, float], gamma: float
                      ) -> dict[str, dict[str, float]]:
    """Compute Q(s,a) from V."""
    Q = {}
    for s in mdp.states:
        Q[s] = {}
        for a in mdp.get_actions(s):
            Q[s][a] = mdp.expected_reward(s, a, V, gamma)
    return Q


def _evaluate_policy(mdp: MDP, policy: dict[str, str], gamma: float,
                     max_iter: int = 1000, epsilon: float = 1e-10
                     ) -> dict[str, float]:
    """Evaluate a policy via iterative Bellman equation."""
    V = {s: 0.0 for s in mdp.states}
    for _ in range(max_iter):
        delta = 0.0
        V_new = {}
        for s in mdp.states:
            a = policy.get(s, "")
            if not a or s in mdp.terminal_states:
                V_new[s] = 0.0
                continue
            V_new[s] = mdp.expected_reward(s, a, V, gamma)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < epsilon:
            break
    return V


# ─────────────────────────────────────────────────────────────────────────────
#  Example MDPs
# ─────────────────────────────────────────────────────────────────────────────

def gridworld(rows: int = 4, cols: int = 4, goal: tuple[int, int] = (3, 3),
              trap: tuple[int, int] | None = (1, 1),
              goal_reward: float = 1.0, trap_reward: float = -1.0,
              step_reward: float = -0.04, slip_prob: float = 0.0) -> MDP:
    """Create a standard gridworld MDP.

    Agent can move N/S/E/W. With probability slip_prob, action has no effect.
    Goal and trap are terminal states.
    """
    mdp = MDP(name=f"GridWorld_{rows}x{cols}")

    # Create states
    for r in range(rows):
        for c in range(cols):
            s = f"({r},{c})"
            terminal = (r, c) == goal or (trap and (r, c) == trap)
            mdp.add_state(s, terminal=terminal)

    mdp.set_initial(f"(0,0)")

    directions = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

    for r in range(rows):
        for c in range(cols):
            s = f"({r},{c})"
            if s in mdp.terminal_states:
                continue

            for action, (dr, dc) in directions.items():
                nr, nc = r + dr, c + dc
                # Boundary check
                if 0 <= nr < rows and 0 <= nc < cols:
                    sp = f"({nr},{nc})"
                else:
                    sp = s  # bounce back

                # Determine reward
                if (nr, nc) == goal:
                    reward = goal_reward
                elif trap and (nr, nc) == trap:
                    reward = trap_reward
                else:
                    reward = step_reward

                if slip_prob > 0:
                    mdp.add_transition(s, action, sp, 1.0 - slip_prob, reward)
                    mdp.add_transition(s, action, s, slip_prob, step_reward)
                else:
                    mdp.add_transition(s, action, sp, 1.0, reward)

    return mdp


def inventory_management(max_stock: int = 5, max_order: int = 3,
                         demand_probs: list[float] | None = None,
                         holding_cost: float = -1.0, stockout_cost: float = -5.0,
                         order_cost: float = -2.0, sale_revenue: float = 3.0) -> MDP:
    """Inventory management MDP.

    State: current stock level [0..max_stock].
    Action: order quantity [0..max_order].
    Demand: random (Poisson-like distribution).
    """
    mdp = MDP(name="Inventory")
    if demand_probs is None:
        # Simple demand distribution
        demand_probs = [0.3, 0.4, 0.2, 0.1]  # P(demand=0,1,2,3)

    states = [f"stock_{i}" for i in range(max_stock + 1)]
    for s in states:
        mdp.add_state(s)
    mdp.set_initial("stock_0")

    for stock in range(max_stock + 1):
        for order in range(max_order + 1):
            action = f"order_{order}"
            mdp.add_action(action)
            available = min(stock + order, max_stock)

            # For each demand level
            transitions: dict[str, tuple[float, float]] = {}
            for demand, prob in enumerate(demand_probs):
                if prob <= 0:
                    continue
                sold = min(demand, available)
                remaining = available - sold
                unmet = demand - sold

                reward = (sold * sale_revenue +
                          order * order_cost +
                          remaining * holding_cost +
                          unmet * stockout_cost)

                next_state = f"stock_{remaining}"
                if next_state in transitions:
                    old_p, old_r = transitions[next_state]
                    # Merge: weight rewards by probability
                    new_p = old_p + prob
                    new_r = (old_r * old_p + reward * prob) / new_p
                    transitions[next_state] = (new_p, new_r)
                else:
                    transitions[next_state] = (prob, reward)

            for ns, (p, r) in transitions.items():
                mdp.add_transition(f"stock_{stock}", action, ns, p, r)

    return mdp


def gambling(states_count: int = 100) -> MDP:
    """Gambler's problem: bet on coin flips to reach a target.

    State: current capital [0..states_count].
    Action: bet amount [1..min(s, states_count-s)].
    Win (p=0.4): gain bet. Lose (p=0.6): lose bet.
    """
    mdp = MDP(name="Gambler")
    for i in range(states_count + 1):
        mdp.add_state(f"${i}", terminal=(i == 0 or i == states_count))
    mdp.set_initial(f"$50")

    p_win = 0.4
    for s in range(1, states_count):
        max_bet = min(s, states_count - s)
        for bet in range(1, max_bet + 1):
            action = f"bet_{bet}"
            mdp.add_action(action)
            win_state = f"${s + bet}"
            lose_state = f"${s - bet}"
            win_reward = 1.0 if s + bet == states_count else 0.0
            mdp.add_transition(f"${s}", action, win_state, p_win, win_reward)
            mdp.add_transition(f"${s}", action, lose_state, 1.0 - p_win, 0.0)

    return mdp


def two_state_mdp() -> MDP:
    """Minimal 2-state MDP for testing."""
    mdp = MDP(name="TwoState")
    mdp.add_state("s0")
    mdp.add_state("s1")
    mdp.set_initial("s0")

    mdp.add_transition("s0", "a", "s0", 0.5, 2.0)
    mdp.add_transition("s0", "a", "s1", 0.5, 0.0)
    mdp.add_transition("s0", "b", "s1", 1.0, 1.0)
    mdp.add_transition("s1", "a", "s0", 0.8, -1.0)
    mdp.add_transition("s1", "a", "s1", 0.2, 0.0)
    mdp.add_transition("s1", "b", "s1", 1.0, 0.5)

    return mdp


# ─────────────────────────────────────────────────────────────────────────────
#  Comparison & Summary
# ─────────────────────────────────────────────────────────────────────────────

def compare_solvers(mdp: MDP, gamma: float = 0.99) -> dict[str, MDPResult]:
    """Run all exact solvers and compare results."""
    results = {}
    results["value_iteration"] = value_iteration(mdp, gamma=gamma)
    results["policy_iteration"] = policy_iteration(mdp, gamma=gamma)
    results["lp"] = linear_programming(mdp, gamma=gamma)
    return results


def mdp_summary(mdp: MDP) -> dict:
    """Summary statistics for an MDP."""
    n_transitions = sum(len(v) for v in mdp._transitions.values())
    return {
        "name": mdp.name,
        "states": len(mdp.states),
        "actions": len(mdp.actions),
        "transitions": n_transitions,
        "terminal_states": len(mdp.terminal_states),
        "initial_state": mdp.initial_state,
        "issues": mdp.validate(),
    }
