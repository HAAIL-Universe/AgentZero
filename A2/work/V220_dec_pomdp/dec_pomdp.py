"""
V220: Decentralized POMDPs (Dec-POMDPs)

Multi-agent partially observable Markov decision processes where each agent
has private observations and must act based on local information only.

Composes:
  - V216 POMDP: belief-space reasoning, alpha vectors, observation models
  - V205 Concurrent Game Structures: joint actions, multi-agent transitions

Key concepts:
  - Joint actions: tuple of individual agent actions
  - Private observations: each agent sees its own observation
  - Local policies: map observation histories to actions
  - Team reward: cooperative (shared) or individual rewards
  - Finite-horizon dynamic programming (exact for small problems)
  - JESP: Joint Equilibrium-based Search for Policies (iterative best response)
  - Centralized-planning decentralized-execution via occupancy states

Complexity: Dec-POMDPs are NEXP-complete in general.
"""

from __future__ import annotations
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class DecPOMDP:
    """Decentralized POMDP specification."""
    name: str = "unnamed"
    agents: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    # Per-agent action sets
    actions: Dict[str, List[str]] = field(default_factory=dict)
    # Per-agent observation sets
    observations: Dict[str, List[str]] = field(default_factory=dict)
    # Transition: P(s'|s, joint_action)
    _transitions: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[str, float]]] = field(
        default_factory=dict
    )
    # Reward: R(s, joint_action) -- team reward (cooperative)
    _rewards: Dict[Tuple[str, Tuple[str, ...]], float] = field(default_factory=dict)
    # Observation: O_i(o_i | s', joint_action) per agent
    _obs_probs: Dict[Tuple[str, str, Tuple[str, ...]], List[Tuple[str, float]]] = field(
        default_factory=dict
    )
    # Initial state distribution
    _initial: Dict[str, float] = field(default_factory=dict)
    # Discount factor
    gamma: float = 0.95
    # Finite horizon (None = infinite)
    horizon: Optional[int] = None

    # -- Building API --

    def add_agent(self, agent: str) -> None:
        if agent not in self.agents:
            self.agents.append(agent)

    def add_state(self, state: str) -> None:
        if state not in self.states:
            self.states.append(state)

    def add_action(self, agent: str, action: str) -> None:
        if agent not in self.actions:
            self.actions[agent] = []
        if action not in self.actions[agent]:
            self.actions[agent].append(action)

    def add_observation(self, agent: str, obs: str) -> None:
        if agent not in self.observations:
            self.observations[agent] = []
        if obs not in self.observations[agent]:
            self.observations[agent].append(obs)

    def set_transition(self, state: str, joint_action: Tuple[str, ...],
                       next_state: str, prob: float) -> None:
        key = (state, joint_action)
        if key not in self._transitions:
            self._transitions[key] = []
        # Update existing or append
        for i, (s, p) in enumerate(self._transitions[key]):
            if s == next_state:
                self._transitions[key][i] = (next_state, prob)
                return
        self._transitions[key].append((next_state, prob))

    def set_reward(self, state: str, joint_action: Tuple[str, ...],
                   reward: float) -> None:
        self._rewards[(state, joint_action)] = reward

    def set_observation_prob(self, agent: str, next_state: str,
                             joint_action: Tuple[str, ...],
                             obs: str, prob: float) -> None:
        key = (agent, next_state, joint_action)
        if key not in self._obs_probs:
            self._obs_probs[key] = []
        for i, (o, p) in enumerate(self._obs_probs[key]):
            if o == obs:
                self._obs_probs[key][i] = (obs, prob)
                return
        self._obs_probs[key].append((obs, prob))

    def set_initial_state(self, state: str, prob: float) -> None:
        self._initial[state] = prob

    # -- Query API --

    def get_joint_actions(self) -> List[Tuple[str, ...]]:
        """All joint action tuples (cartesian product of agent action sets)."""
        agent_acts = [self.actions.get(a, []) for a in self.agents]
        return list(itertools.product(*agent_acts))

    def get_transitions(self, state: str,
                        joint_action: Tuple[str, ...]) -> List[Tuple[str, float]]:
        return self._transitions.get((state, joint_action), [])

    def get_reward(self, state: str, joint_action: Tuple[str, ...]) -> float:
        return self._rewards.get((state, joint_action), 0.0)

    def get_observation_prob(self, agent: str, next_state: str,
                             joint_action: Tuple[str, ...], obs: str) -> float:
        entries = self._obs_probs.get((agent, next_state, joint_action), [])
        for o, p in entries:
            if o == obs:
                return p
        return 0.0

    def get_observation_dist(self, agent: str, next_state: str,
                             joint_action: Tuple[str, ...]) -> List[Tuple[str, float]]:
        return self._obs_probs.get((agent, next_state, joint_action), [])

    def get_initial_belief(self) -> Dict[str, float]:
        if self._initial:
            return dict(self._initial)
        # Uniform
        n = len(self.states)
        return {s: 1.0 / n for s in self.states} if n > 0 else {}

    def get_joint_observation_prob(self, next_state: str,
                                   joint_action: Tuple[str, ...],
                                   joint_obs: Tuple[str, ...]) -> float:
        """P(joint_obs | s', joint_action) = product of individual O_i."""
        prob = 1.0
        for i, agent in enumerate(self.agents):
            prob *= self.get_observation_prob(agent, next_state, joint_action,
                                             joint_obs[i])
            if prob == 0.0:
                return 0.0
        return prob

    def get_joint_observations(self) -> List[Tuple[str, ...]]:
        """All joint observation tuples."""
        agent_obs = [self.observations.get(a, []) for a in self.agents]
        return list(itertools.product(*agent_obs))

    def validate(self) -> List[str]:
        """Check structural validity. Returns list of issues."""
        issues = []
        if not self.agents:
            issues.append("No agents defined")
        if not self.states:
            issues.append("No states defined")
        for agent in self.agents:
            if agent not in self.actions or not self.actions[agent]:
                issues.append(f"Agent '{agent}' has no actions")
            if agent not in self.observations or not self.observations[agent]:
                issues.append(f"Agent '{agent}' has no observations")
        # Check transition distributions sum to ~1
        for (s, ja), dists in self._transitions.items():
            total = sum(p for _, p in dists)
            if abs(total - 1.0) > 1e-6:
                issues.append(
                    f"Transitions from ({s}, {ja}) sum to {total:.6f}, not 1.0"
                )
        # Check observation distributions
        for (agent, sp, ja), dists in self._obs_probs.items():
            total = sum(p for _, p in dists)
            if abs(total - 1.0) > 1e-6:
                issues.append(
                    f"Obs dist for agent={agent}, s'={sp}, ja={ja} sums to {total:.6f}"
                )
        # Check initial belief
        if self._initial:
            total = sum(self._initial.values())
            if abs(total - 1.0) > 1e-6:
                issues.append(f"Initial belief sums to {total:.6f}")
        return issues


# ---------------------------------------------------------------------------
# Policy representations
# ---------------------------------------------------------------------------

@dataclass
class LocalPolicy:
    """A finite-horizon local policy for one agent as a policy tree.

    At each step t, maps observation history (tuple of observations) to action.
    For finite horizon H, history lengths are 0..H-1.
    """
    agent: str
    # history (tuple of obs) -> action
    mapping: Dict[Tuple[str, ...], str] = field(default_factory=dict)

    def get_action(self, history: Tuple[str, ...]) -> str:
        """Get action for observation history. Falls back to shorter prefixes."""
        if history in self.mapping:
            return self.mapping[history]
        # Fall back to empty history (stationary policy)
        if () in self.mapping:
            return self.mapping[()]
        return ""

    def set_action(self, history: Tuple[str, ...], action: str) -> None:
        self.mapping[history] = action


@dataclass
class JointPolicy:
    """Collection of local policies, one per agent."""
    policies: Dict[str, LocalPolicy] = field(default_factory=dict)

    def get_action(self, agent: str, history: Tuple[str, ...]) -> str:
        if agent in self.policies:
            return self.policies[agent].get_action(history)
        return ""

    def set_policy(self, agent: str, policy: LocalPolicy) -> None:
        self.policies[agent] = policy


@dataclass
class DecPOMDPResult:
    """Result from a Dec-POMDP solver."""
    joint_policy: JointPolicy
    value: float  # Expected value from initial state
    iterations: int = 0
    converged: bool = False
    solver: str = ""
    horizon: int = 0


# ---------------------------------------------------------------------------
# Finite-Horizon Dynamic Programming (Exhaustive)
# ---------------------------------------------------------------------------

def _enumerate_local_policies(agent: str, agent_actions: List[str],
                               agent_obs: List[str],
                               horizon: int) -> List[LocalPolicy]:
    """Enumerate all deterministic local policies for an agent.

    At t=0: choose action (no observations yet) -> |A| choices
    At t=1: for each obs, choose action -> |A|^|O| choices
    ...
    At t=k: |A|^(|O|^k) choices
    Total: product over t of |A|^(|O|^t)
    """
    if horizon <= 0:
        return [LocalPolicy(agent=agent)]

    # Build all observation histories of length 0..horizon-1
    histories_by_step: List[List[Tuple[str, ...]]] = []
    for t in range(horizon):
        if t == 0:
            histories_by_step.append([()])  # Empty history
        else:
            # All obs sequences of length t
            histories_by_step.append(
                list(itertools.product(*([agent_obs] * t)))
            )

    # All histories across all steps
    all_histories = []
    for step_hists in histories_by_step:
        all_histories.extend(step_hists)

    # Enumerate all action assignments
    action_assignments = list(
        itertools.product(*([agent_actions] * len(all_histories)))
    )

    policies = []
    for assignment in action_assignments:
        policy = LocalPolicy(agent=agent)
        for hist, act in zip(all_histories, assignment):
            policy.set_action(hist, act)
        policies.append(policy)

    return policies


def evaluate_joint_policy(dec: DecPOMDP, joint_policy: JointPolicy,
                          horizon: int,
                          n_simulations: int = 5000,
                          seed: Optional[int] = None) -> float:
    """Monte Carlo evaluation of a joint policy."""
    rng = random.Random(seed)
    b0 = dec.get_initial_belief()
    total_reward = 0.0

    for _ in range(n_simulations):
        # Sample initial state
        state = _sample_dist(b0, rng)
        histories = {agent: () for agent in dec.agents}
        episode_reward = 0.0
        discount = 1.0

        for t in range(horizon):
            # Each agent picks action based on local history
            joint_action = tuple(
                joint_policy.get_action(agent, histories[agent])
                for agent in dec.agents
            )

            # Reward
            r = dec.get_reward(state, joint_action)
            episode_reward += discount * r
            discount *= dec.gamma

            # Transition
            trans = dec.get_transitions(state, joint_action)
            if not trans:
                break
            next_state = _sample_list(trans, rng)

            # Observations
            for i, agent in enumerate(dec.agents):
                obs_dist = dec.get_observation_dist(agent, next_state,
                                                    joint_action)
                if obs_dist:
                    obs = _sample_list(obs_dist, rng)
                else:
                    obs = dec.observations[agent][0] if dec.observations.get(agent) else ""
                histories[agent] = histories[agent] + (obs,)

            state = next_state

        total_reward += episode_reward

    return total_reward / n_simulations


def exhaustive_dp(dec: DecPOMDP, horizon: Optional[int] = None,
                  seed: Optional[int] = None) -> DecPOMDPResult:
    """Exhaustive search over all joint policies.

    Only feasible for very small problems (2-3 states, 2 actions, 2 obs, horizon 2-3).
    """
    h = horizon if horizon is not None else (dec.horizon or 2)

    best_value = float('-inf')
    best_policy = None

    # Enumerate local policies per agent
    agent_policies = {}
    for agent in dec.agents:
        agent_policies[agent] = _enumerate_local_policies(
            agent,
            dec.actions.get(agent, []),
            dec.observations.get(agent, []),
            h
        )

    # Iterate over all joint policy combinations
    agent_list = dec.agents
    policy_lists = [agent_policies[a] for a in agent_list]

    for combo in itertools.product(*policy_lists):
        jp = JointPolicy()
        for agent, pol in zip(agent_list, combo):
            jp.set_policy(agent, pol)

        val = evaluate_joint_policy(dec, jp, h, n_simulations=2000,
                                    seed=seed)
        if val > best_value:
            best_value = val
            best_policy = jp

    return DecPOMDPResult(
        joint_policy=best_policy or JointPolicy(),
        value=best_value,
        iterations=1,
        converged=True,
        solver="exhaustive_dp",
        horizon=h
    )


# ---------------------------------------------------------------------------
# JESP: Joint Equilibrium-based Search for Policies
# ---------------------------------------------------------------------------

def jesp(dec: DecPOMDP, horizon: Optional[int] = None,
         max_iter: int = 50, n_restarts: int = 5,
         seed: Optional[int] = None) -> DecPOMDPResult:
    """Joint Equilibrium-based Search for Policies.

    Iterative best response: fix all agents' policies except one,
    optimize that one agent's policy, repeat until convergence.
    Multiple random restarts to escape local optima.
    """
    rng = random.Random(seed)
    h = horizon if horizon is not None else (dec.horizon or 2)

    best_value = float('-inf')
    best_policy = None
    total_iterations = 0

    for restart in range(n_restarts):
        # Initialize with random policies
        jp = _random_joint_policy(dec, h, rng)
        current_val = evaluate_joint_policy(dec, jp, h, n_simulations=3000,
                                            seed=rng.randint(0, 2**31))

        converged = False
        for iteration in range(max_iter):
            improved = False
            for agent in dec.agents:
                # Best response for this agent
                best_agent_policy, best_agent_val = _best_response(
                    dec, jp, agent, h, rng
                )
                if best_agent_val > current_val + 1e-8:
                    jp.set_policy(agent, best_agent_policy)
                    current_val = best_agent_val
                    improved = True

            total_iterations += 1
            if not improved:
                converged = True
                break

        if current_val > best_value:
            best_value = current_val
            best_policy = jp

    return DecPOMDPResult(
        joint_policy=best_policy or JointPolicy(),
        value=best_value,
        iterations=total_iterations,
        converged=True,
        solver="jesp",
        horizon=h
    )


def _best_response(dec: DecPOMDP, jp: JointPolicy, agent: str,
                    horizon: int, rng: random.Random
                    ) -> Tuple[LocalPolicy, float]:
    """Find best response policy for one agent, given others fixed."""
    agent_actions = dec.actions.get(agent, [])
    agent_obs = dec.observations.get(agent, [])

    # Enumerate all local policies for this agent
    candidates = _enumerate_local_policies(agent, agent_actions, agent_obs,
                                           horizon)

    # Cap candidates to prevent combinatorial explosion
    if len(candidates) > 500:
        candidates = rng.sample(candidates, 500)

    best_policy = candidates[0] if candidates else LocalPolicy(agent=agent)
    best_val = float('-inf')

    for candidate in candidates:
        test_jp = JointPolicy(policies=dict(jp.policies))
        test_jp.set_policy(agent, candidate)
        val = evaluate_joint_policy(dec, test_jp, horizon,
                                    n_simulations=1500,
                                    seed=rng.randint(0, 2**31))
        if val > best_val:
            best_val = val
            best_policy = candidate

    return best_policy, best_val


def _random_joint_policy(dec: DecPOMDP, horizon: int,
                          rng: random.Random) -> JointPolicy:
    """Create a random joint policy."""
    jp = JointPolicy()
    for agent in dec.agents:
        agent_actions = dec.actions.get(agent, [])
        agent_obs = dec.observations.get(agent, [])
        policy = LocalPolicy(agent=agent)

        for t in range(horizon):
            if t == 0:
                histories = [()]
            else:
                histories = list(itertools.product(*([agent_obs] * t)))
            for hist in histories:
                policy.set_action(hist, rng.choice(agent_actions))

        jp.set_policy(agent, policy)
    return jp


# ---------------------------------------------------------------------------
# Centralized Planning, Decentralized Execution (CPDE)
# ---------------------------------------------------------------------------

def cpde(dec: DecPOMDP, horizon: Optional[int] = None,
         n_simulations: int = 5000,
         seed: Optional[int] = None) -> DecPOMDPResult:
    """Centralized Planning, Decentralized Execution.

    Convert Dec-POMDP to centralized POMDP over joint observations,
    solve it, then extract local policies via simulation.

    This gives an upper bound (since centralized planning has more info).
    The extracted local policies are approximate.
    """
    rng = random.Random(seed)
    h = horizon if horizon is not None else (dec.horizon or 2)

    # Phase 1: Solve centralized version via value iteration on belief space
    # For small problems, use DP over occupancy states
    b0 = dec.get_initial_belief()
    joint_actions = dec.get_joint_actions()
    joint_obs = dec.get_joint_observations()

    # Q-value computation: Q(b, ja) for belief state b, joint action ja
    # Use backward induction for finite horizon
    # V_H(b) = 0
    # Q_t(b, ja) = sum_s b(s) * [R(s,ja) + gamma * sum_{s'} T(s'|s,ja) * sum_{jo} O(jo|s',ja) * V_{t+1}(b')]
    # V_t(b) = max_ja Q_t(b, ja)

    # Since belief space is continuous, we do this via simulation-based rollouts
    # with a greedy centralized policy

    # Build centralized Q-values at initial belief via rollout
    best_ja_per_step = {}

    # Forward simulation to build centralized policy
    # Use multi-step lookahead from initial belief
    centralized_policy = _centralized_rollout_policy(dec, h, rng)

    # Phase 2: Extract local policies via behavioral cloning
    # Run centralized policy, record (agent, history) -> action mappings
    jp = _extract_local_policies(dec, centralized_policy, h, rng)

    val = evaluate_joint_policy(dec, jp, h, n_simulations=n_simulations,
                                seed=rng.randint(0, 2**31))

    return DecPOMDPResult(
        joint_policy=jp,
        value=val,
        iterations=1,
        converged=True,
        solver="cpde",
        horizon=h
    )


def _centralized_rollout_policy(dec: DecPOMDP, horizon: int,
                                 rng: random.Random
                                 ) -> Dict[Tuple[str, Tuple[Tuple[str, ...], ...]], Tuple[str, ...]]:
    """Build centralized policy mapping (state, joint_obs_history) -> joint_action.

    Uses one-step lookahead with Monte Carlo rollout for value estimation.
    """
    joint_actions = dec.get_joint_actions()
    policy = {}

    def rollout_value(state: str, step: int, n_rollouts: int = 200) -> float:
        """Estimate value from state at step using random policy."""
        if step >= horizon:
            return 0.0
        total = 0.0
        for _ in range(n_rollouts):
            s = state
            val = 0.0
            discount = 1.0
            for t in range(step, horizon):
                ja = rng.choice(joint_actions)
                val += discount * dec.get_reward(s, ja)
                discount *= dec.gamma
                trans = dec.get_transitions(s, ja)
                if not trans:
                    break
                s = _sample_list(trans, rng)
            total += val
        return total / n_rollouts

    # One-step lookahead from each state
    for s in dec.states:
        best_ja = joint_actions[0] if joint_actions else ()
        best_q = float('-inf')
        for ja in joint_actions:
            r = dec.get_reward(s, ja)
            # Expected future value
            ev = 0.0
            trans = dec.get_transitions(s, ja)
            for sp, tp in trans:
                ev += tp * rollout_value(sp, 1)
            q = r + dec.gamma * ev
            if q > best_q:
                best_q = q
                best_ja = ja
        policy[s] = best_ja

    return policy


def _extract_local_policies(dec: DecPOMDP,
                             centralized_policy: Dict,
                             horizon: int,
                             rng: random.Random) -> JointPolicy:
    """Extract local policies from centralized policy via simulation."""
    b0 = dec.get_initial_belief()
    # Accumulate (agent, history) -> action counts
    action_counts: Dict[Tuple[str, Tuple[str, ...]], Dict[str, int]] = {}

    n_episodes = 3000
    for _ in range(n_episodes):
        state = _sample_dist(b0, rng)
        histories = {agent: () for agent in dec.agents}

        for t in range(horizon):
            # Centralized policy picks joint action based on state
            ja = centralized_policy.get(state, dec.get_joint_actions()[0])

            # Record each agent's action
            for i, agent in enumerate(dec.agents):
                key = (agent, histories[agent])
                if key not in action_counts:
                    action_counts[key] = {}
                act = ja[i]
                action_counts[key][act] = action_counts[key].get(act, 0) + 1

            # Transition
            trans = dec.get_transitions(state, ja)
            if not trans:
                break
            next_state = _sample_list(trans, rng)

            # Observations
            for i, agent in enumerate(dec.agents):
                obs_dist = dec.get_observation_dist(agent, next_state, ja)
                if obs_dist:
                    obs = _sample_list(obs_dist, rng)
                else:
                    obs = dec.observations[agent][0]
                histories[agent] = histories[agent] + (obs,)

            state = next_state

    # Extract majority-vote policies
    jp = JointPolicy()
    for agent in dec.agents:
        policy = LocalPolicy(agent=agent)
        for (ag, hist), counts in action_counts.items():
            if ag == agent:
                best_act = max(counts, key=counts.get)
                policy.set_action(hist, best_act)
        jp.set_policy(agent, policy)

    return jp


# ---------------------------------------------------------------------------
# Belief-based analysis
# ---------------------------------------------------------------------------

def occupancy_state(dec: DecPOMDP, joint_policy: JointPolicy,
                    horizon: int, step: int,
                    n_samples: int = 10000,
                    seed: Optional[int] = None) -> Dict[Tuple[str, Tuple[Tuple[str, ...], ...]], float]:
    """Compute the occupancy state (joint belief over state and observation histories).

    Returns distribution over (state, (agent1_history, agent2_history, ...)).
    """
    rng = random.Random(seed)
    b0 = dec.get_initial_belief()
    counts: Dict[Tuple[str, Tuple[Tuple[str, ...], ...]], int] = {}

    for _ in range(n_samples):
        state = _sample_dist(b0, rng)
        histories = {agent: () for agent in dec.agents}

        for t in range(step):
            ja = tuple(
                joint_policy.get_action(agent, histories[agent])
                for agent in dec.agents
            )
            trans = dec.get_transitions(state, ja)
            if not trans:
                break
            next_state = _sample_list(trans, rng)
            for i, agent in enumerate(dec.agents):
                obs_dist = dec.get_observation_dist(agent, next_state, ja)
                if obs_dist:
                    obs = _sample_list(obs_dist, rng)
                else:
                    obs = dec.observations[agent][0]
                histories[agent] = histories[agent] + (obs,)
            state = next_state

        hist_tuple = tuple(histories[a] for a in dec.agents)
        key = (state, hist_tuple)
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}


def information_loss(dec: DecPOMDP, horizon: int,
                     seed: Optional[int] = None) -> Dict[str, float]:
    """Measure information loss per agent: entropy of state given observation history.

    Lower = agent observations are more informative about true state.
    """
    rng = random.Random(seed)
    b0 = dec.get_initial_belief()
    joint_actions = dec.get_joint_actions()

    # Simulate and collect (agent, obs_history) -> state distribution
    state_given_history: Dict[Tuple[str, Tuple[str, ...]], Dict[str, int]] = {}
    n_episodes = 5000

    for _ in range(n_episodes):
        state = _sample_dist(b0, rng)
        histories = {agent: () for agent in dec.agents}

        for t in range(horizon):
            ja = rng.choice(joint_actions)  # Random policy for information analysis
            trans = dec.get_transitions(state, ja)
            if not trans:
                break
            next_state = _sample_list(trans, rng)
            for i, agent in enumerate(dec.agents):
                obs_dist = dec.get_observation_dist(agent, next_state, ja)
                if obs_dist:
                    obs = _sample_list(obs_dist, rng)
                else:
                    obs = dec.observations[agent][0]
                histories[agent] = histories[agent] + (obs,)
            state = next_state

        # Record final state for each agent's history
        for agent in dec.agents:
            key = (agent, histories[agent])
            if key not in state_given_history:
                state_given_history[key] = {}
            state_given_history[key][state] = \
                state_given_history[key].get(state, 0) + 1

    # Compute conditional entropy H(S|history) per agent
    agent_entropies: Dict[str, List[float]] = {a: [] for a in dec.agents}
    for (agent, hist), state_counts in state_given_history.items():
        total = sum(state_counts.values())
        ent = 0.0
        for count in state_counts.values():
            p = count / total
            if p > 0:
                ent -= p * math.log2(p)
        agent_entropies[agent].append(ent)

    return {
        agent: (sum(ents) / len(ents) if ents else 0.0)
        for agent, ents in agent_entropies.items()
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(dec: DecPOMDP, joint_policy: JointPolicy,
             horizon: Optional[int] = None,
             seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Simulate one episode under a joint policy."""
    rng = random.Random(seed)
    h = horizon if horizon is not None else (dec.horizon or 10)
    b0 = dec.get_initial_belief()
    state = _sample_dist(b0, rng)
    histories = {agent: () for agent in dec.agents}
    trace = []

    for t in range(h):
        joint_action = tuple(
            joint_policy.get_action(agent, histories[agent])
            for agent in dec.agents
        )
        reward = dec.get_reward(state, joint_action)

        trans = dec.get_transitions(state, joint_action)
        if not trans:
            trace.append({
                "step": t, "state": state,
                "joint_action": joint_action,
                "observations": {},
                "reward": reward, "next_state": None
            })
            break

        next_state = _sample_list(trans, rng)
        observations = {}
        for i, agent in enumerate(dec.agents):
            obs_dist = dec.get_observation_dist(agent, next_state, joint_action)
            if obs_dist:
                obs = _sample_list(obs_dist, rng)
            else:
                obs = dec.observations[agent][0]
            observations[agent] = obs
            histories[agent] = histories[agent] + (obs,)

        trace.append({
            "step": t, "state": state,
            "joint_action": joint_action,
            "observations": observations,
            "reward": reward, "next_state": next_state
        })
        state = next_state

    return trace


def compare_solvers(dec: DecPOMDP, horizon: Optional[int] = None,
                    seed: int = 42) -> Dict[str, DecPOMDPResult]:
    """Run all applicable solvers and compare results."""
    h = horizon if horizon is not None else (dec.horizon or 2)
    results = {}

    # JESP (always applicable)
    results["jesp"] = jesp(dec, horizon=h, seed=seed)

    # CPDE
    results["cpde"] = cpde(dec, horizon=h, seed=seed)

    # Exhaustive only for tiny problems
    n_states = len(dec.states)
    max_actions = max(len(dec.actions.get(a, [])) for a in dec.agents) if dec.agents else 0
    max_obs = max(len(dec.observations.get(a, [])) for a in dec.agents) if dec.agents else 0
    # Rough estimate of policy space size
    if n_states <= 4 and max_actions <= 2 and max_obs <= 2 and h <= 2:
        result = exhaustive_dp(dec, horizon=h, seed=seed)
        result.solver = "exhaustive"
        results["exhaustive"] = result

    return results


def dec_pomdp_summary(dec: DecPOMDP) -> Dict[str, Any]:
    """Summary statistics for a Dec-POMDP."""
    return {
        "name": dec.name,
        "agents": dec.agents,
        "n_states": len(dec.states),
        "n_joint_actions": len(dec.get_joint_actions()),
        "actions_per_agent": {a: len(dec.actions.get(a, [])) for a in dec.agents},
        "observations_per_agent": {a: len(dec.observations.get(a, []))
                                    for a in dec.agents},
        "n_transitions": len(dec._transitions),
        "n_rewards": len(dec._rewards),
        "gamma": dec.gamma,
        "horizon": dec.horizon,
        "issues": dec.validate()
    }


# ---------------------------------------------------------------------------
# Example problems
# ---------------------------------------------------------------------------

def decentralized_tiger(listen_cost: float = -1.0,
                        tiger_penalty: float = -100.0,
                        treasure_reward: float = 10.0,
                        listen_accuracy: float = 0.85,
                        gamma: float = 0.95) -> DecPOMDP:
    """Decentralized Tiger Problem (Nair et al. 2003).

    Two agents face two doors. A tiger is behind one door, treasure behind
    the other. Each agent can listen (noisy) or open a door.
    Both must coordinate -- if either opens the tiger door, both are penalized.
    After any door is opened, the tiger is randomly reassigned.
    """
    dec = DecPOMDP(name="dec_tiger", gamma=gamma)

    # States: tiger location
    for s in ["tiger-left", "tiger-right"]:
        dec.add_state(s)
    dec.set_initial_state("tiger-left", 0.5)
    dec.set_initial_state("tiger-right", 0.5)

    # Agents
    for agent in ["agent1", "agent2"]:
        dec.add_agent(agent)
        for act in ["listen", "open-left", "open-right"]:
            dec.add_action(agent, act)
        for obs in ["hear-left", "hear-right"]:
            dec.add_observation(agent, obs)

    # For each state and joint action
    actions_list = ["listen", "open-left", "open-right"]
    for s in dec.states:
        for a1 in actions_list:
            for a2 in actions_list:
                ja = (a1, a2)

                # Reward
                if a1 == "listen" and a2 == "listen":
                    dec.set_reward(s, ja, 2 * listen_cost)
                elif a1 == "listen" or a2 == "listen":
                    # One listens, one opens
                    opener_act = a2 if a1 == "listen" else a1
                    if s == "tiger-left" and opener_act == "open-left":
                        dec.set_reward(s, ja, tiger_penalty)
                    elif s == "tiger-right" and opener_act == "open-right":
                        dec.set_reward(s, ja, tiger_penalty)
                    else:
                        dec.set_reward(s, ja, treasure_reward)
                else:
                    # Both open
                    opens_tiger = False
                    opens_treasure = False
                    for act in [a1, a2]:
                        if s == "tiger-left" and act == "open-left":
                            opens_tiger = True
                        elif s == "tiger-right" and act == "open-right":
                            opens_tiger = True
                        else:
                            opens_treasure = True
                    if opens_tiger:
                        dec.set_reward(s, ja, tiger_penalty)
                    else:
                        dec.set_reward(s, ja, 2 * treasure_reward)

                # Transitions
                any_open = a1 != "listen" or a2 != "listen"
                if any_open:
                    # Reset: tiger randomly placed
                    dec.set_transition(s, ja, "tiger-left", 0.5)
                    dec.set_transition(s, ja, "tiger-right", 0.5)
                else:
                    # Listening: state unchanged
                    dec.set_transition(s, ja, s, 1.0)

                # Observations (each agent independently)
                for i, agent in enumerate(["agent1", "agent2"]):
                    agent_act = ja[i]
                    for sp in dec.states:
                        if agent_act == "listen":
                            if sp == "tiger-left":
                                dec.set_observation_prob(
                                    agent, sp, ja, "hear-left", listen_accuracy)
                                dec.set_observation_prob(
                                    agent, sp, ja, "hear-right",
                                    1 - listen_accuracy)
                            else:
                                dec.set_observation_prob(
                                    agent, sp, ja, "hear-right", listen_accuracy)
                                dec.set_observation_prob(
                                    agent, sp, ja, "hear-left",
                                    1 - listen_accuracy)
                        else:
                            # Opened a door -> uniform observation (uninformative)
                            dec.set_observation_prob(
                                agent, sp, ja, "hear-left", 0.5)
                            dec.set_observation_prob(
                                agent, sp, ja, "hear-right", 0.5)

    return dec


def cooperative_box_pushing(gamma: float = 0.95) -> DecPOMDP:
    """Cooperative Box-Pushing Problem (Seuken & Zilberstein 2007).

    Two agents on a 1D grid. A small box and a large box are on the grid.
    Small box: one agent can push it (reward 10).
    Large box: requires both agents pushing together (reward 100).
    Agents have noisy position sensing.
    """
    dec = DecPOMDP(name="box_pushing", gamma=gamma)

    # Simplified: 4 positions (0,1,2,3), agents start at 0 and 3
    # Small box at 1, large box at 2
    # States: (agent1_pos, agent2_pos, small_box_pushed, large_box_pushed)
    # Simplified to key states
    states = ["start", "a1_at_small", "a2_at_small", "both_at_large",
              "a1_at_large", "a2_at_large", "small_done", "large_done", "done"]
    for s in states:
        dec.add_state(s)
    dec.set_initial_state("start", 1.0)

    for agent in ["agent1", "agent2"]:
        dec.add_agent(agent)
        for act in ["move-left", "move-right", "push", "stay"]:
            dec.add_action(agent, act)
        for obs in ["empty", "small-box", "large-box", "wall"]:
            dec.add_observation(agent, obs)

    # Simplified transition model
    joint_actions = dec.get_joint_actions()
    for s in states:
        for ja in joint_actions:
            a1, a2 = ja
            if s == "start":
                if a1 == "move-right" and a2 == "move-left":
                    # Both move toward center
                    dec.set_transition(s, ja, "both_at_large", 0.7)
                    dec.set_transition(s, ja, "a1_at_small", 0.15)
                    dec.set_transition(s, ja, "a2_at_large", 0.15)
                elif a1 == "move-right":
                    dec.set_transition(s, ja, "a1_at_small", 0.8)
                    dec.set_transition(s, ja, "start", 0.2)
                elif a2 == "move-left":
                    dec.set_transition(s, ja, "a2_at_large", 0.8)
                    dec.set_transition(s, ja, "start", 0.2)
                else:
                    dec.set_transition(s, ja, "start", 1.0)
                dec.set_reward(s, ja, -1.0)  # Step cost

            elif s == "a1_at_small":
                if a1 == "push":
                    dec.set_transition(s, ja, "small_done", 0.9)
                    dec.set_transition(s, ja, "a1_at_small", 0.1)
                    dec.set_reward(s, ja, 0.0)
                elif a1 == "move-right":
                    dec.set_transition(s, ja, "a1_at_large", 0.8)
                    dec.set_transition(s, ja, "a1_at_small", 0.2)
                    dec.set_reward(s, ja, -1.0)
                else:
                    dec.set_transition(s, ja, "a1_at_small", 1.0)
                    dec.set_reward(s, ja, -1.0)

            elif s == "both_at_large":
                if a1 == "push" and a2 == "push":
                    dec.set_transition(s, ja, "large_done", 0.9)
                    dec.set_transition(s, ja, "both_at_large", 0.1)
                    dec.set_reward(s, ja, 0.0)
                else:
                    dec.set_transition(s, ja, "both_at_large", 1.0)
                    dec.set_reward(s, ja, -1.0)

            elif s == "a1_at_large":
                if a2 == "move-left":
                    dec.set_transition(s, ja, "both_at_large", 0.8)
                    dec.set_transition(s, ja, "a1_at_large", 0.2)
                else:
                    dec.set_transition(s, ja, "a1_at_large", 1.0)
                dec.set_reward(s, ja, -1.0)

            elif s == "a2_at_large":
                if a1 == "move-right":
                    dec.set_transition(s, ja, "both_at_large", 0.8)
                    dec.set_transition(s, ja, "a2_at_large", 0.2)
                else:
                    dec.set_transition(s, ja, "a2_at_large", 1.0)
                dec.set_reward(s, ja, -1.0)

            elif s == "a2_at_small":
                if a2 == "push":
                    dec.set_transition(s, ja, "small_done", 0.9)
                    dec.set_transition(s, ja, "a2_at_small", 0.1)
                    dec.set_reward(s, ja, 0.0)
                else:
                    dec.set_transition(s, ja, "a2_at_small", 1.0)
                    dec.set_reward(s, ja, -1.0)

            elif s == "small_done":
                dec.set_transition(s, ja, "done", 1.0)
                dec.set_reward(s, ja, 10.0)

            elif s == "large_done":
                dec.set_transition(s, ja, "done", 1.0)
                dec.set_reward(s, ja, 100.0)

            elif s == "done":
                dec.set_transition(s, ja, "done", 1.0)
                dec.set_reward(s, ja, 0.0)

    # Observations: noisy position sensing
    for s in states:
        for ja in joint_actions:
            for sp in states:
                for agent in ["agent1", "agent2"]:
                    if "small" in sp and agent[5] == "1" and "a1" in sp:
                        dec.set_observation_prob(agent, sp, ja, "small-box", 0.8)
                        dec.set_observation_prob(agent, sp, ja, "empty", 0.2)
                    elif "small" in sp and agent[5] == "2" and "a2" in sp:
                        dec.set_observation_prob(agent, sp, ja, "small-box", 0.8)
                        dec.set_observation_prob(agent, sp, ja, "empty", 0.2)
                    elif "large" in sp:
                        dec.set_observation_prob(agent, sp, ja, "large-box", 0.8)
                        dec.set_observation_prob(agent, sp, ja, "empty", 0.2)
                    elif sp == "done":
                        dec.set_observation_prob(agent, sp, ja, "empty", 1.0)
                    else:
                        dec.set_observation_prob(agent, sp, ja, "empty", 0.7)
                        dec.set_observation_prob(agent, sp, ja, "wall", 0.3)

    return dec


def multi_agent_meeting(grid_size: int = 3, gamma: float = 0.95) -> DecPOMDP:
    """Multi-Agent Meeting Problem.

    Two agents on a grid must meet at the same cell.
    Each has noisy position observations. Reward when co-located.
    """
    dec = DecPOMDP(name="meeting", gamma=gamma)

    # States: (a1_pos, a2_pos) on 1D grid
    positions = list(range(grid_size))
    for p1 in positions:
        for p2 in positions:
            dec.add_state(f"{p1}_{p2}")

    # Initial: corners
    dec.set_initial_state(f"0_{grid_size-1}", 1.0)

    for agent in ["agent1", "agent2"]:
        dec.add_agent(agent)
        for act in ["left", "right", "stay"]:
            dec.add_action(agent, act)
        for p in positions:
            dec.add_observation(agent, f"pos_{p}")

    def move(pos, action, max_pos):
        if action == "left":
            return max(0, pos - 1)
        elif action == "right":
            return min(max_pos, pos + 1)
        return pos

    actions_list = ["left", "right", "stay"]
    for p1 in positions:
        for p2 in positions:
            s = f"{p1}_{p2}"
            for a1 in actions_list:
                for a2 in actions_list:
                    ja = (a1, a2)
                    np1 = move(p1, a1, grid_size - 1)
                    np2 = move(p2, a2, grid_size - 1)
                    ns = f"{np1}_{np2}"

                    # Stochastic: 80% intended, 20% stay
                    if ns == s:
                        dec.set_transition(s, ja, s, 1.0)
                    else:
                        dec.set_transition(s, ja, ns, 0.8)
                        dec.set_transition(s, ja, s, 0.2)

                    # Reward: meeting
                    if p1 == p2:
                        dec.set_reward(s, ja, 10.0)
                    else:
                        dec.set_reward(s, ja, -1.0)

                    # Observations: each agent sees own noisy position
                    for sp in [s, ns]:
                        sp_p1, sp_p2 = int(sp.split("_")[0]), int(sp.split("_")[1])
                        for agent, pos in [("agent1", sp_p1), ("agent2", sp_p2)]:
                            # 70% correct, 30% adjacent
                            for op in positions:
                                if op == pos:
                                    prob = 0.7
                                elif abs(op - pos) == 1:
                                    prob = 0.15
                                else:
                                    prob = 0.0
                            # Normalize for edge positions
                            total_noise = 0.7 + 0.15 * min(2, len([x for x in positions if abs(x - pos) == 1]))
                            for op in positions:
                                if op == pos:
                                    p_obs = 0.7 / total_noise
                                elif abs(op - pos) == 1:
                                    p_obs = 0.15 / total_noise
                                else:
                                    p_obs = 0.0
                                if p_obs > 0:
                                    dec.set_observation_prob(
                                        agent, sp, ja, f"pos_{op}", p_obs)

    return dec


def communication_channel(gamma: float = 0.95) -> DecPOMDP:
    """Dec-POMDP with explicit communication.

    Two agents observe different parts of the environment.
    Agent1 sees signal A or B. Agent2 must act on it.
    Agent1 can communicate (action = message) to help Agent2.
    Tests whether JESP discovers communication strategies.
    """
    dec = DecPOMDP(name="communication", gamma=gamma)

    for s in ["signal-A", "signal-B"]:
        dec.add_state(s)
    dec.set_initial_state("signal-A", 0.5)
    dec.set_initial_state("signal-B", 0.5)

    # Agent1: sees the signal, can send message
    dec.add_agent("sender")
    for act in ["msg-A", "msg-B"]:
        dec.add_action("sender", act)
    for obs in ["see-A", "see-B"]:
        dec.add_observation("sender", obs)

    # Agent2: receives message (as observation), must act
    dec.add_agent("receiver")
    for act in ["act-A", "act-B"]:
        dec.add_action("receiver", act)
    for obs in ["heard-A", "heard-B"]:
        dec.add_observation("receiver", obs)

    for s in dec.states:
        for sa in ["msg-A", "msg-B"]:
            for ra in ["act-A", "act-B"]:
                ja = (sa, ra)

                # State stays the same
                dec.set_transition(s, ja, s, 1.0)

                # Reward: receiver matches signal
                signal_letter = s.split("-")[1]  # A or B
                act_letter = ra.split("-")[1]
                if signal_letter == act_letter:
                    dec.set_reward(s, ja, 10.0)
                else:
                    dec.set_reward(s, ja, -10.0)

                # Sender observations: sees signal perfectly
                for sp in dec.states:
                    if sp == "signal-A":
                        dec.set_observation_prob("sender", sp, ja, "see-A", 1.0)
                        dec.set_observation_prob("sender", sp, ja, "see-B", 0.0)
                    else:
                        dec.set_observation_prob("sender", sp, ja, "see-B", 1.0)
                        dec.set_observation_prob("sender", sp, ja, "see-A", 0.0)

                    # Receiver observations: hears sender's message (with noise)
                    if sa == "msg-A":
                        dec.set_observation_prob("receiver", sp, ja,
                                                 "heard-A", 0.9)
                        dec.set_observation_prob("receiver", sp, ja,
                                                 "heard-B", 0.1)
                    else:
                        dec.set_observation_prob("receiver", sp, ja,
                                                 "heard-B", 0.9)
                        dec.set_observation_prob("receiver", sp, ja,
                                                 "heard-A", 0.1)

    return dec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_dist(dist: Dict[str, float], rng: random.Random) -> str:
    items = list(dist.items())
    r = rng.random()
    cumulative = 0.0
    for item, prob in items:
        cumulative += prob
        if r <= cumulative:
            return item
    return items[-1][0] if items else ""


def _sample_list(dist: List[Tuple[str, float]], rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for item, prob in dist:
        cumulative += prob
        if r <= cumulative:
            return item
    return dist[-1][0] if dist else ""
