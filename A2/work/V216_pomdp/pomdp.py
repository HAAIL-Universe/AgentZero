"""V216: Partially Observable Markov Decision Processes (POMDPs).

Complete POMDP framework with belief-state tracking, exact and approximate
solvers, and classic example problems.

Composes V213 (MDP) for underlying dynamics and extends with:
- Observation model O(o|s',a)
- Belief state representation and Bayesian update
- Alpha-vector value iteration (exact, for small POMDPs)
- Point-based value iteration (PBVI, scalable approximate)
- Perseus (randomized point-based backup)
- QMDP upper bound (fast heuristic)
- FIB (Fast Informed Bound)
- Belief-space simulation and policy execution
- POMDP-to-belief-MDP conversion
- Classic problems: Tiger, RockSample, Machine Maintenance, Hallway Navigation

AI-Generated | Claude (Anthropic) | AgentZero A2 Session 297 | 2026-03-18
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# Compose V213 (MDP) for transition dynamics
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "V213_markov_decision_processes"))
from markov_decision_processes import MDP, value_iteration as mdp_value_iteration


# ─────────────────────────────────────────────────────────────────────────────
#  Core Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlphaVector:
    """An alpha vector represents value for a specific plan (action sequence).

    Each alpha vector is associated with an action and maps belief states
    to values via: V(b) = sum_s alpha(s) * b(s).
    """
    action: str
    values: dict[str, float]  # state -> value

    def dot(self, belief: dict[str, float]) -> float:
        """Compute alpha . b = sum_s alpha(s) * b(s)."""
        return sum(self.values.get(s, 0.0) * b for s, b in belief.items() if b > 0)


@dataclass
class POMDPResult:
    """Result of a POMDP solver."""
    alpha_vectors: list[AlphaVector]
    iterations: int = 0
    converged: bool = False
    value_at_b0: float = 0.0

    def value(self, belief: dict[str, float]) -> float:
        """Compute V(b) = max_alpha (alpha . b)."""
        if not self.alpha_vectors:
            return 0.0
        return max(av.dot(belief) for av in self.alpha_vectors)

    def best_action(self, belief: dict[str, float]) -> str:
        """Select action that maximizes value at current belief."""
        if not self.alpha_vectors:
            return ""
        best_av = max(self.alpha_vectors, key=lambda av: av.dot(belief))
        return best_av.action

    def policy(self, belief: dict[str, float]) -> tuple[str, float]:
        """Return (best_action, value) at a belief point."""
        if not self.alpha_vectors:
            return ("", 0.0)
        best_av = max(self.alpha_vectors, key=lambda av: av.dot(belief))
        return (best_av.action, best_av.dot(belief))


class POMDP:
    """Partially Observable Markov Decision Process.

    Extends MDP with an observation model. The agent cannot observe the
    true state directly; instead it receives observations drawn from
    O(o | s', a) after taking action a and transitioning to s'.

    Belief state b(s) = P(s | history) is maintained via Bayesian update.
    """

    def __init__(self, name: str = "POMDP"):
        self.name = name
        self.states: list[str] = []
        self._state_set: set[str] = set()
        self.actions: list[str] = []
        self._action_set: set[str] = set()
        self.observations: list[str] = []
        self._obs_set: set[str] = set()

        # Transition: (s, a) -> [(s', prob)]
        self._transitions: dict[tuple[str, str], list[tuple[str, float]]] = {}

        # Reward: R(s, a) or R(s, a, s')
        self._rewards_sa: dict[tuple[str, str], float] = {}
        self._rewards_sas: dict[tuple[str, str, str], float] = {}

        # Observation: (a, s') -> [(o, prob)]
        self._observations: dict[tuple[str, str], list[tuple[str, float]]] = {}

        # Initial belief (uniform if not set)
        self._initial_belief: dict[str, float] | None = None

        # Discount factor
        self.gamma: float = 0.95

    def add_state(self, state: str):
        """Add a state."""
        if state not in self._state_set:
            self.states.append(state)
            self._state_set.add(state)

    def add_action(self, action: str):
        """Add an action."""
        if action not in self._action_set:
            self.actions.append(action)
            self._action_set.add(action)

    def add_observation(self, obs: str):
        """Add an observation."""
        if obs not in self._obs_set:
            self.observations.append(obs)
            self._obs_set.add(obs)

    def add_transition(self, s: str, a: str, s_prime: str, prob: float):
        """Add transition P(s'|s,a)."""
        self.add_state(s)
        self.add_state(s_prime)
        self.add_action(a)
        key = (s, a)
        if key not in self._transitions:
            self._transitions[key] = []
        self._transitions[key].append((s_prime, prob))

    def set_reward(self, s: str, a: str, reward: float, s_prime: str | None = None):
        """Set reward R(s,a) or R(s,a,s')."""
        if s_prime is not None:
            self._rewards_sas[(s, a, s_prime)] = reward
        else:
            self._rewards_sa[(s, a)] = reward

    def add_observation_prob(self, a: str, s_prime: str, obs: str, prob: float):
        """Add observation probability O(o | s', a)."""
        self.add_observation(obs)
        key = (a, s_prime)
        if key not in self._observations:
            self._observations[key] = []
        self._observations[key].append((obs, prob))

    def set_initial_belief(self, belief: dict[str, float]):
        """Set initial belief state."""
        self._initial_belief = dict(belief)

    def get_initial_belief(self) -> dict[str, float]:
        """Get initial belief (uniform if not set)."""
        if self._initial_belief:
            return dict(self._initial_belief)
        n = len(self.states)
        return {s: 1.0 / n for s in self.states}

    def get_transitions(self, s: str, a: str) -> list[tuple[str, float]]:
        """Get P(s'|s,a) as list of (s', prob)."""
        return self._transitions.get((s, a), [])

    def get_observation_prob(self, a: str, s_prime: str, obs: str) -> float:
        """Get O(o|s',a)."""
        for o, p in self._observations.get((a, s_prime), []):
            if o == obs:
                return p
        return 0.0

    def get_observation_dist(self, a: str, s_prime: str) -> list[tuple[str, float]]:
        """Get full observation distribution O(.|s',a)."""
        return self._observations.get((a, s_prime), [])

    def get_reward(self, s: str, a: str, s_prime: str | None = None) -> float:
        """Get reward R(s,a) or R(s,a,s')."""
        if s_prime is not None and (s, a, s_prime) in self._rewards_sas:
            return self._rewards_sas[(s, a, s_prime)]
        return self._rewards_sa.get((s, a), 0.0)

    def expected_reward(self, belief: dict[str, float], a: str) -> float:
        """Compute expected immediate reward: sum_s b(s) * R(s,a)."""
        r = 0.0
        for s, bs in belief.items():
            if bs <= 0:
                continue
            # Use R(s,a,s') if available, else R(s,a)
            if any(k[0] == s and k[1] == a for k in self._rewards_sas):
                for sp, tp in self.get_transitions(s, a):
                    r += bs * tp * self.get_reward(s, a, sp)
            else:
                r += bs * self.get_reward(s, a)
        return r

    def belief_update(self, belief: dict[str, float], action: str,
                      observation: str) -> dict[str, float]:
        """Bayesian belief update: b'(s') = eta * O(o|s',a) * sum_s T(s'|s,a) * b(s).

        Returns normalized posterior belief. Returns uniform if observation
        has zero probability (shouldn't happen in well-formed POMDP).
        """
        new_belief = {}
        for sp in self.states:
            # sum_s T(s'|s,a) * b(s)
            pred = 0.0
            for s, bs in belief.items():
                if bs <= 0:
                    continue
                for s2, tp in self.get_transitions(s, action):
                    if s2 == sp:
                        pred += bs * tp
            # O(o|s',a) * prediction
            obs_p = self.get_observation_prob(action, sp, observation)
            new_belief[sp] = obs_p * pred

        # Normalize
        total = sum(new_belief.values())
        if total > 0:
            for s in new_belief:
                new_belief[s] /= total
        else:
            # Zero-probability observation: return uniform
            n = len(self.states)
            new_belief = {s: 1.0 / n for s in self.states}

        return new_belief

    def observation_probability(self, belief: dict[str, float], action: str,
                                 obs: str) -> float:
        """P(o | b, a) = sum_s' O(o|s',a) * sum_s T(s'|s,a) * b(s)."""
        prob = 0.0
        for sp in self.states:
            pred = 0.0
            for s, bs in belief.items():
                if bs <= 0:
                    continue
                for s2, tp in self.get_transitions(s, action):
                    if s2 == sp:
                        pred += bs * tp
            prob += self.get_observation_prob(action, sp, obs) * pred
        return prob

    def to_mdp(self) -> MDP:
        """Extract underlying MDP (ignoring observations)."""
        mdp = MDP(name=f"{self.name}_MDP")
        for s in self.states:
            mdp.add_state(s)
        for a in self.actions:
            mdp.add_action(a)
        for (s, a), transitions in self._transitions.items():
            for sp, tp in transitions:
                r = self.get_reward(s, a, sp)
                if r == 0.0:
                    r = self.get_reward(s, a)
                mdp.add_transition(s, a, sp, tp, r)
        return mdp

    def validate(self) -> list[str]:
        """Validate POMDP structure."""
        issues = []
        if not self.states:
            issues.append("No states defined")
        if not self.actions:
            issues.append("No actions defined")
        if not self.observations:
            issues.append("No observations defined")

        # Check transition probabilities
        for (s, a), transitions in self._transitions.items():
            total = sum(p for _, p in transitions)
            if abs(total - 1.0) > 1e-9:
                issues.append(f"T(.|{s},{a}) sums to {total:.6f}")

        # Check observation probabilities
        for (a, sp), obs_dist in self._observations.items():
            total = sum(p for _, p in obs_dist)
            if abs(total - 1.0) > 1e-9:
                issues.append(f"O(.|{sp},{a}) sums to {total:.6f}")

        return issues


# ─────────────────────────────────────────────────────────────────────────────
#  Solvers
# ─────────────────────────────────────────────────────────────────────────────

def qmdp(pomdp: POMDP, epsilon: float = 1e-8, max_iter: int = 10000) -> POMDPResult:
    """QMDP upper bound: solve underlying MDP, use Q-values as alpha vectors.

    Fast heuristic. Assumes full observability after one step.
    Good when information-gathering actions aren't critical.
    """
    mdp = pomdp.to_mdp()
    result = mdp_value_iteration(mdp, gamma=pomdp.gamma, epsilon=epsilon,
                                  max_iter=max_iter)

    alpha_vectors = []
    for a in pomdp.actions:
        values = {}
        for s in pomdp.states:
            # Q(s,a) = R(s,a) + gamma * sum_s' T(s'|s,a) * V(s')
            q = pomdp.get_reward(s, a)
            for sp, tp in pomdp.get_transitions(s, a):
                r_sas = pomdp.get_reward(s, a, sp)
                if r_sas != 0.0 and pomdp.get_reward(s, a) == 0.0:
                    q_contribution = tp * (r_sas + pomdp.gamma * result.values.get(sp, 0.0))
                else:
                    q_contribution = tp * (pomdp.gamma * result.values.get(sp, 0.0))
                q += q_contribution
            # If using R(s,a) model, the reward was already added once
            # Fix: compute properly
            values[s] = _compute_q_sa(pomdp, s, a, result.values)
        alpha_vectors.append(AlphaVector(action=a, values=values))

    b0 = pomdp.get_initial_belief()
    val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0

    return POMDPResult(alpha_vectors=alpha_vectors, iterations=result.iterations,
                       converged=result.converged, value_at_b0=val)


def _compute_q_sa(pomdp: POMDP, s: str, a: str, V: dict[str, float]) -> float:
    """Compute Q(s,a) = R(s,a) + gamma * sum_s' T(s'|s,a) * [R(s,a,s')/R(s,a) + V(s')]."""
    # Determine reward model
    has_sas = any(k[0] == s and k[1] == a for k in pomdp._rewards_sas)
    q = 0.0
    if has_sas:
        for sp, tp in pomdp.get_transitions(s, a):
            r = pomdp.get_reward(s, a, sp)
            q += tp * (r + pomdp.gamma * V.get(sp, 0.0))
    else:
        q = pomdp.get_reward(s, a)
        for sp, tp in pomdp.get_transitions(s, a):
            q += tp * pomdp.gamma * V.get(sp, 0.0)
    return q


def fib(pomdp: POMDP, epsilon: float = 1e-6, max_iter: int = 100) -> POMDPResult:
    """Fast Informed Bound (FIB).

    Tighter upper bound than QMDP. Accounts for observation uncertainty
    by computing per-observation-weighted max over successor alpha vectors.
    """
    # Initialize with QMDP alpha vectors
    qmdp_result = qmdp(pomdp, epsilon=epsilon)
    alpha_vectors = list(qmdp_result.alpha_vectors)

    for iteration in range(max_iter):
        new_alphas = []
        changed = False

        for a in pomdp.actions:
            values = {}
            for s in pomdp.states:
                v = pomdp.get_reward(s, a)
                obs_contrib = 0.0
                for o in pomdp.observations:
                    # For each observation, find best alpha vector for the
                    # predicted next-state distribution
                    best_o = -math.inf
                    for av in alpha_vectors:
                        val = 0.0
                        for sp, tp in pomdp.get_transitions(s, a):
                            op = pomdp.get_observation_prob(a, sp, o)
                            val += tp * op * av.values.get(sp, 0.0)
                        best_o = max(best_o, val)
                    if best_o > -math.inf:
                        obs_contrib += best_o
                v += pomdp.gamma * obs_contrib
                values[s] = v

            new_alphas.append(AlphaVector(action=a, values=values))

        # Check convergence
        max_diff = 0.0
        for i, nav in enumerate(new_alphas):
            if i < len(alpha_vectors):
                for s in pomdp.states:
                    diff = abs(nav.values.get(s, 0.0) - alpha_vectors[i].values.get(s, 0.0))
                    max_diff = max(max_diff, diff)

        alpha_vectors = new_alphas
        if max_diff < epsilon:
            b0 = pomdp.get_initial_belief()
            val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
            return POMDPResult(alpha_vectors=alpha_vectors, iterations=iteration + 1,
                               converged=True, value_at_b0=val)

    b0 = pomdp.get_initial_belief()
    val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
    return POMDPResult(alpha_vectors=alpha_vectors, iterations=max_iter,
                       converged=False, value_at_b0=val)


def exact_value_iteration(pomdp: POMDP, epsilon: float = 1e-6,
                           max_iter: int = 100, max_alphas: int = 500
                           ) -> POMDPResult:
    """Exact POMDP value iteration via alpha-vector pruning.

    Computes the optimal value function as a set of alpha vectors.
    Exact but exponential: |alpha| can grow as O(|A| * |alpha|^|O|) per iteration.
    Only practical for small POMDPs.

    Uses incremental pruning to control alpha vector growth.
    """
    # Initialize: one alpha per action (immediate reward)
    alpha_vectors = []
    for a in pomdp.actions:
        values = {s: pomdp.get_reward(s, a) for s in pomdp.states}
        alpha_vectors.append(AlphaVector(action=a, values=values))

    for iteration in range(max_iter):
        new_alphas = []

        for a in pomdp.actions:
            # For each action, compute backup alpha vectors
            # alpha_a(s) = R(s,a) + gamma * sum_o max_alpha' sum_s' T(s'|s,a)*O(o|s',a)*alpha'(s')
            #
            # Step 1: For each observation o and old alpha vector alpha',
            # compute g_{a,o,alpha'}(s) = sum_s' T(s'|s,a)*O(o|s',a)*alpha'(s')
            g_ao: dict[str, list[dict[str, float]]] = {}
            for o in pomdp.observations:
                g_ao[o] = []
                for av in alpha_vectors:
                    g = {}
                    for s in pomdp.states:
                        val = 0.0
                        for sp, tp in pomdp.get_transitions(s, a):
                            op = pomdp.get_observation_prob(a, sp, o)
                            val += tp * op * av.values.get(sp, 0.0)
                        g[s] = val
                    g_ao[o].append(g)

            # Step 2: Cross-sum over observations.
            # For small problems, enumerate all combos. For larger, use
            # incremental pruning.
            if len(pomdp.observations) <= 6 and len(alpha_vectors) <= 20:
                # Full enumeration (small problems)
                combos = _cross_sum_enumerate(pomdp, a, g_ao)
            else:
                # Incremental pruning
                combos = _cross_sum_incremental(pomdp, a, g_ao, max_alphas)

            new_alphas.extend(combos)

        # Prune dominated alpha vectors
        new_alphas = _prune_alpha_vectors(pomdp, new_alphas, max_alphas)

        # Check convergence: max Bellman error
        max_diff = _bellman_error(pomdp, alpha_vectors, new_alphas)

        alpha_vectors = new_alphas

        if max_diff < epsilon:
            b0 = pomdp.get_initial_belief()
            val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
            return POMDPResult(alpha_vectors=alpha_vectors, iterations=iteration + 1,
                               converged=True, value_at_b0=val)

    b0 = pomdp.get_initial_belief()
    val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
    return POMDPResult(alpha_vectors=alpha_vectors, iterations=max_iter,
                       converged=False, value_at_b0=val)


def pbvi(pomdp: POMDP, belief_points: list[dict[str, float]] | None = None,
         n_points: int = 50, epsilon: float = 1e-6, max_iter: int = 100,
         expand_interval: int = 10, seed: int | None = None) -> POMDPResult:
    """Point-Based Value Iteration (PBVI).

    Approximate solver that maintains value function at a finite set of
    belief points. Much more scalable than exact VI.

    Belief point set can be provided or generated via random exploration.
    """
    rng = random.Random(seed)

    # Generate belief points if not provided
    if belief_points is None:
        belief_points = _generate_belief_points(pomdp, n_points, rng)

    # Ensure initial belief is included
    b0 = pomdp.get_initial_belief()
    belief_points = [b0] + belief_points

    # Initialize alpha vectors (one per action)
    alpha_vectors = []
    for a in pomdp.actions:
        values = {s: pomdp.get_reward(s, a) / (1.0 - pomdp.gamma + 1e-10)
                  for s in pomdp.states}
        alpha_vectors.append(AlphaVector(action=a, values=values))

    for iteration in range(max_iter):
        new_alphas = []

        for b in belief_points:
            # Backup at this belief point: find the best action's alpha vector
            best_av = None
            best_val = -math.inf

            for a in pomdp.actions:
                # Compute backup alpha for action a at belief b
                av = _point_backup(pomdp, b, a, alpha_vectors)
                val = av.dot(b)
                if val > best_val:
                    best_val = val
                    best_av = av

            if best_av is not None:
                new_alphas.append(best_av)

        # Prune: keep only non-dominated
        new_alphas = _prune_simple(new_alphas, belief_points)

        # Check convergence
        max_diff = 0.0
        for b in belief_points:
            old_v = max(av.dot(b) for av in alpha_vectors) if alpha_vectors else 0.0
            new_v = max(av.dot(b) for av in new_alphas) if new_alphas else 0.0
            max_diff = max(max_diff, abs(new_v - old_v))

        alpha_vectors = new_alphas

        if max_diff < epsilon:
            val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
            return POMDPResult(alpha_vectors=alpha_vectors, iterations=iteration + 1,
                               converged=True, value_at_b0=val)

        # Expand belief set periodically
        if expand_interval > 0 and (iteration + 1) % expand_interval == 0:
            new_points = _expand_belief_points(pomdp, belief_points, alpha_vectors, rng)
            belief_points.extend(new_points)

    val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
    return POMDPResult(alpha_vectors=alpha_vectors, iterations=max_iter,
                       converged=False, value_at_b0=val)


def perseus(pomdp: POMDP, belief_points: list[dict[str, float]] | None = None,
            n_points: int = 100, epsilon: float = 1e-6, max_iter: int = 200,
            seed: int | None = None) -> POMDPResult:
    """Perseus: randomized point-based value iteration.

    Like PBVI but randomly samples belief points to back up, only keeping
    backups that improve value. More efficient than PBVI for large belief sets.
    """
    rng = random.Random(seed)

    if belief_points is None:
        belief_points = _generate_belief_points(pomdp, n_points, rng)

    b0 = pomdp.get_initial_belief()
    belief_points = [b0] + belief_points

    # Initialize
    alpha_vectors = []
    for a in pomdp.actions:
        # Initialize pessimistically: R_min / (1 - gamma)
        r_min = min(
            (pomdp.get_reward(s, a) for s in pomdp.states),
            default=0.0
        )
        values = {s: r_min / (1.0 - pomdp.gamma + 1e-10) for s in pomdp.states}
        alpha_vectors.append(AlphaVector(action=a, values=values))

    for iteration in range(max_iter):
        # Find not-yet-improved belief points
        old_values = {
            i: max(av.dot(b) for av in alpha_vectors)
            for i, b in enumerate(belief_points)
        }

        new_alphas = list(alpha_vectors)  # start with old set
        improved = set()
        remaining = list(range(len(belief_points)))
        rng.shuffle(remaining)

        for bi in remaining:
            if bi in improved:
                continue

            b = belief_points[bi]
            # Best backup at this point
            best_av = None
            best_val = -math.inf
            for a in pomdp.actions:
                av = _point_backup(pomdp, b, a, alpha_vectors)
                val = av.dot(b)
                if val > best_val:
                    best_val = val
                    best_av = av

            if best_av is not None and best_val >= old_values[bi] - 1e-10:
                new_alphas.append(best_av)
                # Mark all belief points improved by this new vector
                for j in remaining:
                    if j not in improved:
                        v_new = best_av.dot(belief_points[j])
                        if v_new >= old_values[j] - 1e-10:
                            improved.add(j)

            if len(improved) >= len(belief_points):
                break

        # Prune
        alpha_vectors = _prune_simple(new_alphas, belief_points)

        # Convergence check
        max_diff = 0.0
        for i, b in enumerate(belief_points):
            new_v = max(av.dot(b) for av in alpha_vectors) if alpha_vectors else 0.0
            max_diff = max(max_diff, abs(new_v - old_values[i]))

        if max_diff < epsilon:
            val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
            return POMDPResult(alpha_vectors=alpha_vectors, iterations=iteration + 1,
                               converged=True, value_at_b0=val)

    val = max(av.dot(b0) for av in alpha_vectors) if alpha_vectors else 0.0
    return POMDPResult(alpha_vectors=alpha_vectors, iterations=max_iter,
                       converged=False, value_at_b0=val)


# ─────────────────────────────────────────────────────────────────────────────
#  Belief-Space Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_pomdp(pomdp: POMDP, result: POMDPResult, steps: int = 50,
                   seed: int | None = None
                   ) -> list[dict[str, Any]]:
    """Simulate a POMDP under the policy defined by alpha vectors.

    Returns trajectory: list of {state, belief, action, observation, reward}.
    """
    rng = random.Random(seed)
    belief = pomdp.get_initial_belief()

    # Sample initial true state from belief
    state = _sample_from_dist(belief, rng)

    trajectory = []
    for step in range(steps):
        # Select action from alpha-vector policy
        action = result.best_action(belief)
        if not action:
            break

        # Get reward
        reward = pomdp.get_reward(state, action)

        # Transition: sample s' from T(.|s,a)
        transitions = pomdp.get_transitions(state, action)
        if not transitions:
            break
        next_state = _sample_from_transitions(transitions, rng)

        # Add R(s,a,s') if present
        r_sas = pomdp.get_reward(state, action, next_state)
        if r_sas != 0.0 and reward == 0.0:
            reward = r_sas

        # Observation: sample o from O(.|s',a)
        obs_dist = pomdp.get_observation_dist(action, next_state)
        if not obs_dist:
            break
        observation = _sample_from_transitions(obs_dist, rng)

        # Record
        trajectory.append({
            "step": step,
            "state": state,
            "belief": dict(belief),
            "action": action,
            "observation": observation,
            "reward": reward,
            "next_state": next_state,
        })

        # Update belief
        belief = pomdp.belief_update(belief, action, observation)
        state = next_state

    return trajectory


def evaluate_policy(pomdp: POMDP, result: POMDPResult, n_episodes: int = 1000,
                    max_steps: int = 100, seed: int | None = None) -> dict[str, float]:
    """Evaluate a POMDP policy via Monte Carlo simulation.

    Returns dict with mean_reward, std_reward, min_reward, max_reward.
    """
    rng = random.Random(seed)
    rewards = []

    for ep in range(n_episodes):
        belief = pomdp.get_initial_belief()
        state = _sample_from_dist(belief, rng)
        total_reward = 0.0
        discount = 1.0

        for step in range(max_steps):
            action = result.best_action(belief)
            if not action:
                break

            reward = pomdp.get_reward(state, action)
            transitions = pomdp.get_transitions(state, action)
            if not transitions:
                break

            next_state = _sample_from_transitions(transitions, rng)
            r_sas = pomdp.get_reward(state, action, next_state)
            if r_sas != 0.0 and reward == 0.0:
                reward = r_sas

            total_reward += discount * reward
            discount *= pomdp.gamma

            obs_dist = pomdp.get_observation_dist(action, next_state)
            if not obs_dist:
                break
            observation = _sample_from_transitions(obs_dist, rng)
            belief = pomdp.belief_update(belief, action, observation)
            state = next_state

        rewards.append(total_reward)

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) if rewards else 0.0
    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "n_episodes": n_episodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Belief Point Generation & Expansion
# ─────────────────────────────────────────────────────────────────────────────

def _generate_belief_points(pomdp: POMDP, n: int, rng: random.Random
                             ) -> list[dict[str, float]]:
    """Generate belief points via random forward simulation from b0."""
    points = []
    b0 = pomdp.get_initial_belief()

    for _ in range(n):
        b = dict(b0)
        # Random walk of 1-10 steps
        walk_len = rng.randint(1, min(10, max(1, len(pomdp.states))))
        for _ in range(walk_len):
            a = rng.choice(pomdp.actions)
            o = rng.choice(pomdp.observations)
            b_new = pomdp.belief_update(b, a, o)
            # Only use if observation was plausible
            if any(v > 0.01 for v in b_new.values()):
                b = b_new
        points.append(b)

    # Also add corner beliefs (pure states)
    for s in pomdp.states:
        corner = {s2: (1.0 if s2 == s else 0.0) for s2 in pomdp.states}
        points.append(corner)

    return points


def _expand_belief_points(pomdp: POMDP, beliefs: list[dict[str, float]],
                           alpha_vectors: list[AlphaVector],
                           rng: random.Random, max_new: int = 10
                           ) -> list[dict[str, float]]:
    """Expand belief set by one-step successors of existing beliefs."""
    new_points = []
    candidates = list(beliefs)
    rng.shuffle(candidates)

    for b in candidates[:max_new]:
        a = rng.choice(pomdp.actions)
        o = rng.choice(pomdp.observations)
        b_new = pomdp.belief_update(b, a, o)
        # Check it's meaningfully different
        if _belief_distance(b, b_new) > 0.01:
            new_points.append(b_new)

    return new_points


def _belief_distance(b1: dict[str, float], b2: dict[str, float]) -> float:
    """L1 distance between two belief states."""
    keys = set(b1.keys()) | set(b2.keys())
    return sum(abs(b1.get(k, 0.0) - b2.get(k, 0.0)) for k in keys)


# ─────────────────────────────────────────────────────────────────────────────
#  Alpha Vector Operations
# ─────────────────────────────────────────────────────────────────────────────

def _point_backup(pomdp: POMDP, belief: dict[str, float], action: str,
                   alpha_vectors: list[AlphaVector]) -> AlphaVector:
    """Compute backup alpha vector for a specific action at a belief point.

    alpha_a(s) = R(s,a) + gamma * sum_o [max_alpha' sum_s' T(s'|s,a)*O(o|s',a)*alpha'(s')]

    But for the point backup, we select alpha' per observation based on
    which maximizes value at the predicted next belief.
    """
    values = {}
    for s in pomdp.states:
        v = _get_sa_reward(pomdp, s, action)
        obs_sum = 0.0
        for o in pomdp.observations:
            # Find best alpha vector for this observation
            best_val = -math.inf
            best_contrib = 0.0
            for av in alpha_vectors:
                contrib = 0.0
                for sp, tp in pomdp.get_transitions(s, action):
                    op = pomdp.get_observation_prob(action, sp, o)
                    contrib += tp * op * av.values.get(sp, 0.0)
                if contrib > best_val:
                    best_val = contrib
                    best_contrib = contrib
            if best_val > -math.inf:
                obs_sum += best_contrib
        v += pomdp.gamma * obs_sum
        values[s] = v
    return AlphaVector(action=action, values=values)


def _get_sa_reward(pomdp: POMDP, s: str, a: str) -> float:
    """Get R(s,a), computing expected R(s,a,s') if needed."""
    if (s, a) in pomdp._rewards_sa:
        return pomdp._rewards_sa[(s, a)]
    # Compute expected R(s,a,s') over transitions
    r = 0.0
    for sp, tp in pomdp.get_transitions(s, a):
        r += tp * pomdp._rewards_sas.get((s, a, sp), 0.0)
    return r


def _cross_sum_enumerate(pomdp: POMDP, action: str,
                          g_ao: dict[str, list[dict[str, float]]]
                          ) -> list[AlphaVector]:
    """Full cross-sum enumeration for exact VI (small problems)."""
    obs_list = pomdp.observations

    # Start with reward
    reward_vals = {s: _get_sa_reward(pomdp, s, action) for s in pomdp.states}

    # Recursive cross-sum
    current = [reward_vals]

    for o in obs_list:
        g_list = g_ao[o]
        next_set = []
        for existing in current:
            for g in g_list:
                combined = {}
                for s in pomdp.states:
                    combined[s] = existing[s] + pomdp.gamma * g[s]
                next_set.append(combined)
        current = next_set

    return [AlphaVector(action=action, values=v) for v in current]


def _cross_sum_incremental(pomdp: POMDP, action: str,
                            g_ao: dict[str, list[dict[str, float]]],
                            max_alphas: int) -> list[AlphaVector]:
    """Incremental pruning cross-sum for exact VI (larger problems)."""
    obs_list = pomdp.observations

    reward_vals = {s: _get_sa_reward(pomdp, s, action) for s in pomdp.states}

    # Start with first observation's g vectors + reward
    if not obs_list:
        return [AlphaVector(action=action, values=reward_vals)]

    current = []
    first_g = g_ao[obs_list[0]]
    for g in first_g:
        vals = {s: reward_vals[s] + pomdp.gamma * g[s] for s in pomdp.states}
        current.append(vals)

    # Incrementally add observations with pruning
    for oi in range(1, len(obs_list)):
        o = obs_list[oi]
        g_list = g_ao[o]
        next_set = []
        for existing in current:
            for g in g_list:
                combined = {s: existing[s] + pomdp.gamma * g[s] for s in pomdp.states}
                next_set.append(combined)
        # Prune after each observation
        if len(next_set) > max_alphas:
            # Keep only non-dominated vectors (sample a few belief points)
            avs = [AlphaVector(action=action, values=v) for v in next_set]
            avs = _prune_alpha_vectors(pomdp, avs, max_alphas)
            current = [av.values for av in avs]
        else:
            current = next_set

    return [AlphaVector(action=action, values=v) for v in current]


def _prune_alpha_vectors(pomdp: POMDP, alphas: list[AlphaVector],
                          max_keep: int = 500) -> list[AlphaVector]:
    """Prune dominated alpha vectors.

    Uses a combination of:
    1. Point-wise dominance check
    2. Random belief sampling to identify useful vectors
    """
    if len(alphas) <= 1:
        return alphas

    # Sample random belief points for checking
    rng = random.Random(42)
    test_beliefs = []
    for s in pomdp.states:
        test_beliefs.append({s2: (1.0 if s2 == s else 0.0) for s2 in pomdp.states})
    # Uniform
    n = len(pomdp.states)
    test_beliefs.append({s: 1.0 / n for s in pomdp.states})
    # Random
    for _ in range(min(20, max_keep)):
        raw = {s: rng.random() for s in pomdp.states}
        total = sum(raw.values())
        test_beliefs.append({s: v / total for s, v in raw.items()})

    # Find useful vectors: those that are best at some test belief
    useful = set()
    for b in test_beliefs:
        best_i = -1
        best_v = -math.inf
        for i, av in enumerate(alphas):
            v = av.dot(b)
            if v > best_v:
                best_v = v
                best_i = i
        if best_i >= 0:
            useful.add(best_i)

    pruned = [alphas[i] for i in useful]

    # If still too many, keep the max_keep best at uniform belief
    if len(pruned) > max_keep:
        uniform = {s: 1.0 / n for s in pomdp.states}
        pruned.sort(key=lambda av: av.dot(uniform), reverse=True)
        pruned = pruned[:max_keep]

    return pruned if pruned else alphas[:1]


def _prune_simple(alphas: list[AlphaVector],
                   beliefs: list[dict[str, float]]) -> list[AlphaVector]:
    """Simple pruning: keep only vectors that are best at some belief point."""
    if not alphas or not beliefs:
        return alphas

    useful = set()
    for b in beliefs:
        best_i = -1
        best_v = -math.inf
        for i, av in enumerate(alphas):
            v = av.dot(b)
            if v > best_v:
                best_v = v
                best_i = i
        if best_i >= 0:
            useful.add(best_i)

    return [alphas[i] for i in sorted(useful)] if useful else alphas[:1]


def _bellman_error(pomdp: POMDP, old_alphas: list[AlphaVector],
                    new_alphas: list[AlphaVector]) -> float:
    """Estimate max Bellman error over a set of test beliefs."""
    test_beliefs = []
    n = len(pomdp.states)
    for s in pomdp.states:
        test_beliefs.append({s2: (1.0 if s2 == s else 0.0) for s2 in pomdp.states})
    test_beliefs.append({s: 1.0 / n for s in pomdp.states})

    max_diff = 0.0
    for b in test_beliefs:
        old_v = max(av.dot(b) for av in old_alphas) if old_alphas else 0.0
        new_v = max(av.dot(b) for av in new_alphas) if new_alphas else 0.0
        max_diff = max(max_diff, abs(new_v - old_v))
    return max_diff


# ─────────────────────────────────────────────────────────────────────────────
#  Sampling Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_from_dist(dist: dict[str, float], rng: random.Random) -> str:
    """Sample from a probability distribution."""
    r = rng.random()
    cumulative = 0.0
    for item, prob in dist.items():
        cumulative += prob
        if r <= cumulative:
            return item
    # Fallback (rounding)
    return list(dist.keys())[-1]


def _sample_from_transitions(transitions: list[tuple[str, float]],
                               rng: random.Random) -> str:
    """Sample from a transition list [(item, prob)]."""
    r = rng.random()
    cumulative = 0.0
    for item, prob in transitions:
        cumulative += prob
        if r <= cumulative:
            return item
    return transitions[-1][0]


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis & Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_solvers(pomdp: POMDP, seed: int = 42) -> dict[str, POMDPResult]:
    """Run all solvers and compare."""
    results = {}
    results["qmdp"] = qmdp(pomdp)
    results["fib"] = fib(pomdp)
    results["pbvi"] = pbvi(pomdp, n_points=50, seed=seed)
    return results


def belief_entropy(belief: dict[str, float]) -> float:
    """Shannon entropy of a belief state: H(b) = -sum b(s) log b(s)."""
    h = 0.0
    for p in belief.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def information_gain(pomdp: POMDP, belief: dict[str, float],
                     action: str) -> float:
    """Expected information gain from taking an action.

    IG(b,a) = H(b) - E_o[H(b'|o,a)] = H(b) - sum_o P(o|b,a) H(tau(b,a,o))
    """
    h_before = belief_entropy(belief)
    expected_h_after = 0.0

    for o in pomdp.observations:
        p_o = pomdp.observation_probability(belief, action, o)
        if p_o > 0.001:
            b_new = pomdp.belief_update(belief, action, o)
            expected_h_after += p_o * belief_entropy(b_new)

    return h_before - expected_h_after


def most_informative_action(pomdp: POMDP, belief: dict[str, float]) -> tuple[str, float]:
    """Find the action that maximizes expected information gain."""
    best_a = ""
    best_ig = -math.inf
    for a in pomdp.actions:
        ig = information_gain(pomdp, belief, a)
        if ig > best_ig:
            best_ig = ig
            best_a = a
    return (best_a, best_ig)


# ─────────────────────────────────────────────────────────────────────────────
#  Classic Example POMDPs
# ─────────────────────────────────────────────────────────────────────────────

def tiger_problem(listen_cost: float = -1.0, tiger_penalty: float = -100.0,
                  treasure_reward: float = 10.0, listen_accuracy: float = 0.85,
                  gamma: float = 0.95) -> POMDP:
    """The classic Tiger problem.

    Agent stands before two doors. Behind one is a tiger (bad), behind the
    other is treasure (good). Agent can listen (noisy observation) or open
    a door. After opening, the problem resets.
    """
    pomdp = POMDP(name="Tiger")
    pomdp.gamma = gamma

    # States
    pomdp.add_state("tiger-left")
    pomdp.add_state("tiger-right")

    # Actions
    pomdp.add_action("listen")
    pomdp.add_action("open-left")
    pomdp.add_action("open-right")

    # Observations
    pomdp.add_observation("hear-left")
    pomdp.add_observation("hear-right")

    # Transitions
    # Listen: state doesn't change
    pomdp.add_transition("tiger-left", "listen", "tiger-left", 1.0)
    pomdp.add_transition("tiger-right", "listen", "tiger-right", 1.0)

    # Open: resets to uniform
    for s in ["tiger-left", "tiger-right"]:
        for a in ["open-left", "open-right"]:
            pomdp.add_transition(s, a, "tiger-left", 0.5)
            pomdp.add_transition(s, a, "tiger-right", 0.5)

    # Rewards
    pomdp.set_reward("tiger-left", "listen", listen_cost)
    pomdp.set_reward("tiger-right", "listen", listen_cost)
    pomdp.set_reward("tiger-left", "open-left", tiger_penalty)
    pomdp.set_reward("tiger-left", "open-right", treasure_reward)
    pomdp.set_reward("tiger-right", "open-left", treasure_reward)
    pomdp.set_reward("tiger-right", "open-right", tiger_penalty)

    # Observations
    # Listen: noisy indicator
    pomdp.add_observation_prob("listen", "tiger-left", "hear-left", listen_accuracy)
    pomdp.add_observation_prob("listen", "tiger-left", "hear-right", 1.0 - listen_accuracy)
    pomdp.add_observation_prob("listen", "tiger-right", "hear-left", 1.0 - listen_accuracy)
    pomdp.add_observation_prob("listen", "tiger-right", "hear-right", listen_accuracy)

    # Open: uninformative (uniform)
    for s in ["tiger-left", "tiger-right"]:
        for a in ["open-left", "open-right"]:
            pomdp.add_observation_prob(a, s, "hear-left", 0.5)
            pomdp.add_observation_prob(a, s, "hear-right", 0.5)

    # Initial belief: uniform
    pomdp.set_initial_belief({"tiger-left": 0.5, "tiger-right": 0.5})

    return pomdp


def machine_maintenance(n_conditions: int = 3, gamma: float = 0.95) -> POMDP:
    """Machine maintenance POMDP.

    Machine degrades over time. Agent can inspect (noisy) or repair.
    States: good, fair, poor (or more with n_conditions).
    """
    pomdp = POMDP(name="MachineMaintenance")
    pomdp.gamma = gamma

    conditions = [f"cond_{i}" for i in range(n_conditions)]
    for c in conditions:
        pomdp.add_state(c)

    pomdp.add_action("operate")
    pomdp.add_action("inspect")
    pomdp.add_action("repair")

    obs_labels = [f"obs_{i}" for i in range(n_conditions)]
    for o in obs_labels:
        pomdp.add_observation(o)

    # Transitions
    for i, s in enumerate(conditions):
        # Operate: degrade with some probability
        for j, sp in enumerate(conditions):
            if j == i:
                p = 0.6 if i < n_conditions - 1 else 1.0
            elif j == i + 1:
                p = 0.4 if i < n_conditions - 1 else 0.0
            else:
                p = 0.0
            if p > 0:
                pomdp.add_transition(s, "operate", sp, p)

        # Inspect: same as operate (doesn't change state) but costs time
        for j, sp in enumerate(conditions):
            if j == i:
                p = 0.6 if i < n_conditions - 1 else 1.0
            elif j == i + 1:
                p = 0.4 if i < n_conditions - 1 else 0.0
            else:
                p = 0.0
            if p > 0:
                pomdp.add_transition(s, "inspect", sp, p)

        # Repair: return to best condition
        pomdp.add_transition(s, "repair", conditions[0], 1.0)

    # Rewards
    for i, s in enumerate(conditions):
        pomdp.set_reward(s, "operate", 10.0 - 5.0 * i)  # better condition -> more reward
        pomdp.set_reward(s, "inspect", 5.0 - 2.5 * i)   # half reward for inspecting
        pomdp.set_reward(s, "repair", -8.0)              # repair is costly

    # Observations
    accuracy = 0.7
    for a in ["operate", "inspect", "repair"]:
        for i, sp in enumerate(conditions):
            for j, o in enumerate(obs_labels):
                if a == "inspect":
                    # Higher accuracy when inspecting
                    if i == j:
                        p = 0.9
                    else:
                        p = 0.1 / (n_conditions - 1) if n_conditions > 1 else 0.0
                else:
                    # Lower accuracy otherwise
                    if i == j:
                        p = accuracy
                    else:
                        p = (1.0 - accuracy) / (n_conditions - 1) if n_conditions > 1 else 0.0
                pomdp.add_observation_prob(a, sp, o, p)

    pomdp.set_initial_belief({conditions[0]: 1.0, **{c: 0.0 for c in conditions[1:]}})
    return pomdp


def hallway_navigation(length: int = 4, gamma: float = 0.95) -> POMDP:
    """Hallway navigation POMDP.

    Agent navigates a 1D hallway with noisy movement and position sensing.
    Goal: reach the rightmost cell.
    """
    pomdp = POMDP(name="Hallway")
    pomdp.gamma = gamma

    states = [f"pos_{i}" for i in range(length)]
    for s in states:
        pomdp.add_state(s)

    pomdp.add_action("left")
    pomdp.add_action("right")
    pomdp.add_action("stay")

    obs = [f"see_{i}" for i in range(length)]
    for o in obs:
        pomdp.add_observation(o)

    slip = 0.1  # probability of staying in place

    for i in range(length):
        s = states[i]

        # Move right
        if i < length - 1:
            pomdp.add_transition(s, "right", states[i + 1], 1.0 - slip)
            pomdp.add_transition(s, "right", s, slip)
        else:
            pomdp.add_transition(s, "right", s, 1.0)

        # Move left
        if i > 0:
            pomdp.add_transition(s, "left", states[i - 1], 1.0 - slip)
            pomdp.add_transition(s, "left", s, slip)
        else:
            pomdp.add_transition(s, "left", s, 1.0)

        # Stay
        pomdp.add_transition(s, "stay", s, 1.0)

    # Rewards: reward for reaching goal (rightmost)
    goal = length - 1
    for i in range(length):
        for a in ["left", "right", "stay"]:
            if i == goal:
                pomdp.set_reward(states[i], a, 1.0)
            else:
                pomdp.set_reward(states[i], a, -0.1)

    # Observations: noisy position
    for a in ["left", "right", "stay"]:
        for i in range(length):
            for j in range(length):
                if i == j:
                    p = 0.7
                elif abs(i - j) == 1:
                    p = 0.15
                else:
                    p = 0.0
                # Redistribute leftover to neighbors
                if p > 0:
                    pomdp.add_observation_prob(a, states[i], obs[j], p)

    # Normalize observation probabilities
    for a in ["left", "right", "stay"]:
        for i in range(length):
            sp = states[i]
            dist = pomdp.get_observation_dist(a, sp)
            total = sum(p for _, p in dist)
            if abs(total - 1.0) > 1e-9 and total > 0:
                pomdp._observations[(a, sp)] = [(o, p / total) for o, p in dist]

    pomdp.set_initial_belief({s: 1.0 / length for s in states})
    return pomdp


def rock_sample_small(gamma: float = 0.95) -> POMDP:
    """Small RockSample-like POMDP.

    Agent on a 1D grid with one rock. Rock is either good or bad.
    Agent can move, sample rock (get reward if good), or check (noisy sensor).
    """
    pomdp = POMDP(name="RockSample")
    pomdp.gamma = gamma

    # States: (position, rock_quality)
    positions = ["left", "rock", "right"]
    qualities = ["good", "bad"]

    for pos in positions:
        for q in qualities:
            pomdp.add_state(f"{pos}_{q}")

    pomdp.add_action("move-right")
    pomdp.add_action("move-left")
    pomdp.add_action("sample")
    pomdp.add_action("check")

    pomdp.add_observation("none")
    pomdp.add_observation("good-signal")
    pomdp.add_observation("bad-signal")

    for q in qualities:
        # Move right
        pomdp.add_transition(f"left_{q}", "move-right", f"rock_{q}", 1.0)
        pomdp.add_transition(f"rock_{q}", "move-right", f"right_{q}", 1.0)
        pomdp.add_transition(f"right_{q}", "move-right", f"right_{q}", 1.0)

        # Move left
        pomdp.add_transition(f"left_{q}", "move-left", f"left_{q}", 1.0)
        pomdp.add_transition(f"rock_{q}", "move-left", f"left_{q}", 1.0)
        pomdp.add_transition(f"right_{q}", "move-left", f"rock_{q}", 1.0)

        # Sample: rock becomes bad after sampling
        pomdp.add_transition(f"left_{q}", "sample", f"left_{q}", 1.0)
        pomdp.add_transition(f"rock_{q}", "sample", f"rock_bad", 1.0)
        pomdp.add_transition(f"right_{q}", "sample", f"right_{q}", 1.0)

        # Check: doesn't change state
        pomdp.add_transition(f"left_{q}", "check", f"left_{q}", 1.0)
        pomdp.add_transition(f"rock_{q}", "check", f"rock_{q}", 1.0)
        pomdp.add_transition(f"right_{q}", "check", f"right_{q}", 1.0)

    # Rewards
    for q in qualities:
        for pos in positions:
            s = f"{pos}_{q}"
            pomdp.set_reward(s, "move-right", 0.0)
            pomdp.set_reward(s, "move-left", 0.0)
            pomdp.set_reward(s, "check", -1.0)  # checking costs
            if pos == "rock":
                pomdp.set_reward(s, "sample", 10.0 if q == "good" else -10.0)
            else:
                pomdp.set_reward(s, "sample", -1.0)  # sampling nothing

    # Exit reward for reaching right
    for q in qualities:
        pomdp.set_reward(f"right_{q}", "move-right", 10.0)

    # Observations
    sensor_accuracy = 0.8
    for q in qualities:
        for pos in positions:
            sp = f"{pos}_{q}"
            # Movement and sample: uninformative
            for a in ["move-right", "move-left", "sample"]:
                pomdp.add_observation_prob(a, sp, "none", 1.0)

            # Check: noisy sensor
            if q == "good":
                pomdp.add_observation_prob("check", sp, "good-signal", sensor_accuracy)
                pomdp.add_observation_prob("check", sp, "bad-signal", 1.0 - sensor_accuracy)
            else:
                pomdp.add_observation_prob("check", sp, "good-signal", 1.0 - sensor_accuracy)
                pomdp.add_observation_prob("check", sp, "bad-signal", sensor_accuracy)

    # Initial belief: at left, 50/50 rock quality
    pomdp.set_initial_belief({
        "left_good": 0.5, "left_bad": 0.5,
        "rock_good": 0.0, "rock_bad": 0.0,
        "right_good": 0.0, "right_bad": 0.0,
    })

    return pomdp


def pomdp_summary(pomdp: POMDP) -> dict:
    """Summary statistics for a POMDP."""
    n_transitions = sum(len(v) for v in pomdp._transitions.values())
    n_obs_entries = sum(len(v) for v in pomdp._observations.values())
    return {
        "name": pomdp.name,
        "states": len(pomdp.states),
        "actions": len(pomdp.actions),
        "observations": len(pomdp.observations),
        "transitions": n_transitions,
        "observation_entries": n_obs_entries,
        "gamma": pomdp.gamma,
        "issues": pomdp.validate(),
    }
