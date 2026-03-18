"""V224: Interactive POMDPs (I-POMDPs).

Multi-agent partially observable decision-making where agents model each
other's beliefs, preferences, and decision-making processes. Each agent
maintains a recursive belief hierarchy: "I believe you believe I believe..."

Composes V216 (POMDP) for single-agent belief update and planning,
extending it with:
- Interactive state space (physical state x model of other agents)
- Intentional models: belief, frame (transition, observation, reward, policy)
- Recursive nesting: level-0 (POMDP), level-1 (models opponents as level-0), etc.
- Belief update over interactive states (physical + model space)
- Level-k planning: solve own POMDP given opponent model predictions
- Nash equilibrium detection in belief space
- Theory of Mind reasoning: predict, explain, deceive
- Classic examples: multi-agent Tiger, pursuit-evasion, signaling games

AI-Generated | Claude (Anthropic) | AgentZero A2 Session 307 | 2026-03-18
"""

from __future__ import annotations

import math
import random
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from copy import deepcopy

# Compose V216 (POMDP) for belief updates and planning
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "V216_pomdp"))
from pomdp import POMDP, AlphaVector, POMDPResult, belief_update, qmdp, pbvi


# ---------------------------------------------------------------------------
#  Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """An agent's decision frame: transitions, observations, rewards.

    A frame captures the agent's local view of the environment dynamics.
    Different agents may have different frames (asymmetric information).
    """
    agent_id: str
    states: list[str]
    actions: list[str]
    observations: list[str]
    # Transition: T[s][joint_action] -> {s': prob}
    transitions: dict[str, dict[tuple, dict[str, float]]]
    # Observation: O[s'][joint_action] -> {o: prob}
    obs_function: dict[str, dict[tuple, dict[str, float]]]
    # Reward: R[s][joint_action] -> float
    rewards: dict[str, dict[tuple, float]]
    discount: float = 0.9

    def get_actions(self) -> list[str]:
        return self.actions

    def get_observations(self) -> list[str]:
        return self.observations


@dataclass
class IntentionalModel:
    """A model of another agent: their frame + current belief + policy level.

    This is what agent i thinks agent j is: j's decision-making apparatus.
    At level 0, the model includes a fixed policy.
    At level k > 0, the model includes j's own models of other agents.
    """
    agent_id: str
    frame: Frame
    level: int  # nesting level (0 = fixed policy, k = models others at k-1)
    belief: dict[str, float]  # belief over physical states
    # For level 0: fixed policy mapping states to action distributions
    policy: Optional[dict[str, dict[str, float]]] = None
    # For level k > 0: models of other agents (recursive)
    sub_models: Optional[dict[str, 'IntentionalModel']] = None

    def predict_action(self, state: Optional[str] = None) -> dict[str, float]:
        """Predict action distribution given current belief/state."""
        if self.level == 0:
            if self.policy is None:
                # Uniform random if no policy specified
                n = len(self.frame.actions)
                return {a: 1.0 / n for a in self.frame.actions}
            if state and state in self.policy:
                return self.policy[state]
            # Belief-weighted policy
            action_probs: dict[str, float] = {}
            for a in self.frame.actions:
                action_probs[a] = 0.0
            for s, b in self.belief.items():
                if b > 0 and s in self.policy:
                    for a, p in self.policy[s].items():
                        action_probs[a] = action_probs.get(a, 0.0) + b * p
            total = sum(action_probs.values())
            if total > 0:
                return {a: p / total for a, p in action_probs.items()}
            n = len(self.frame.actions)
            return {a: 1.0 / n for a in self.frame.actions}
        else:
            # Level k > 0: solve POMDP given sub-models
            return self._solve_level_k()

    def _solve_level_k(self) -> dict[str, float]:
        """Solve level-k model by converting to POMDP with opponent predictions."""
        # Get opponent action predictions from sub-models
        opponent_preds = {}
        if self.sub_models:
            for oid, model in self.sub_models.items():
                opponent_preds[oid] = model.predict_action()

        # Convert to single-agent POMDP by marginalizing over opponent actions
        pomdp = _frame_to_pomdp(self.frame, opponent_preds)
        result = qmdp(pomdp, horizon=5, discount=self.frame.discount)

        # Extract action from alpha vectors at current belief
        if not result.alpha_vectors:
            n = len(self.frame.actions)
            return {a: 1.0 / n for a in self.frame.actions}

        best_val = float('-inf')
        best_action = self.frame.actions[0]
        for av in result.alpha_vectors:
            val = av.dot(self.belief)
            if val > best_val:
                best_val = val
                best_action = av.action
        return {best_action: 1.0}


@dataclass
class InteractiveState:
    """An interactive state: physical state + models of other agents.

    IS_i = S x M_j for agent i with one opponent j.
    More generally IS_i = S x M_{-i} (product of all opponent model spaces).
    """
    physical_state: str
    agent_models: dict[str, IntentionalModel]  # opponent_id -> model

    def __hash__(self):
        # Hash by physical state + model fingerprints
        model_key = tuple(
            (k, v.level, tuple(sorted(v.belief.items())))
            for k, v in sorted(self.agent_models.items())
        )
        return hash((self.physical_state, model_key))

    def __eq__(self, other):
        if not isinstance(other, InteractiveState):
            return False
        return (self.physical_state == other.physical_state and
                self._models_equal(other))

    def _models_equal(self, other: InteractiveState) -> bool:
        if set(self.agent_models.keys()) != set(other.agent_models.keys()):
            return False
        for k in self.agent_models:
            m1, m2 = self.agent_models[k], other.agent_models[k]
            if m1.level != m2.level:
                return False
            if abs(sum(abs(m1.belief.get(s, 0) - m2.belief.get(s, 0))
                       for s in set(m1.belief) | set(m2.belief))) > 1e-9:
                return False
        return True


@dataclass
class IPOMDPResult:
    """Result of an I-POMDP solver."""
    policy: dict[str, dict[str, float]]  # belief_key -> action_dist
    value: float = 0.0
    iterations: int = 0
    opponent_predictions: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  I-POMDP Class
# ---------------------------------------------------------------------------

class IPOMDP:
    """Interactive POMDP for a single agent reasoning about others.

    The agent maintains beliefs over interactive states (physical state +
    opponent models). Planning involves predicting opponent behavior via
    their models, then solving a POMDP over physical states.
    """

    def __init__(
        self,
        agent_id: str,
        frame: Frame,
        opponent_models: dict[str, list[IntentionalModel]],
        initial_belief: dict[str, float],
        model_priors: Optional[dict[str, dict[int, float]]] = None,
        level: int = 1,
    ):
        """
        Args:
            agent_id: This agent's identifier
            frame: This agent's decision frame
            opponent_models: For each opponent, a list of possible models
            initial_belief: Initial belief over physical states
            model_priors: Prior probability over each opponent's models
            level: This agent's reasoning level
        """
        self.agent_id = agent_id
        self.frame = frame
        self.opponent_models = opponent_models  # opp_id -> [models]
        self.belief = dict(initial_belief)
        self.level = level

        # Model priors: probability over each opponent's possible models
        if model_priors is None:
            self.model_priors: dict[str, list[float]] = {}
            for oid, models in opponent_models.items():
                n = len(models)
                self.model_priors[oid] = [1.0 / n] * n
        else:
            self.model_priors = {}
            for oid, mp in model_priors.items():
                models = opponent_models[oid]
                self.model_priors[oid] = [mp.get(i, 1.0/len(models))
                                          for i in range(len(models))]

    def predict_opponent(self, opponent_id: str) -> dict[str, float]:
        """Predict opponent's action distribution by averaging over models."""
        models = self.opponent_models[opponent_id]
        priors = self.model_priors[opponent_id]

        action_dist: dict[str, float] = {}
        for model, prior in zip(models, priors):
            if prior < 1e-12:
                continue
            pred = model.predict_action()
            for a, p in pred.items():
                action_dist[a] = action_dist.get(a, 0.0) + prior * p

        total = sum(action_dist.values())
        if total > 0:
            return {a: p / total for a, p in action_dist.items()}
        n = len(models[0].frame.actions) if models else 1
        actions = models[0].frame.actions if models else ["noop"]
        return {a: 1.0 / n for a in actions}

    def predict_all_opponents(self) -> dict[str, dict[str, float]]:
        """Predict action distributions for all opponents."""
        return {oid: self.predict_opponent(oid)
                for oid in self.opponent_models}

    def update_belief(
        self,
        own_action: str,
        observation: str,
        opponent_actions: Optional[dict[str, str]] = None,
    ) -> dict[str, float]:
        """Update belief over physical states given action and observation.

        If opponent actions are observed, use them directly.
        Otherwise, marginalize over predicted opponent actions.
        """
        new_belief: dict[str, float] = {}

        if opponent_actions:
            # Joint action is known
            joint_action = self._make_joint_action(own_action, opponent_actions)
            for s_new in self.frame.states:
                prob = 0.0
                for s, b in self.belief.items():
                    if b < 1e-12:
                        continue
                    t = self.frame.transitions.get(s, {}).get(joint_action, {})
                    t_prob = t.get(s_new, 0.0)
                    o = self.frame.obs_function.get(s_new, {}).get(joint_action, {})
                    o_prob = o.get(observation, 0.0)
                    prob += b * t_prob * o_prob
                if prob > 1e-12:
                    new_belief[s_new] = prob
        else:
            # Marginalize over opponent actions
            opp_preds = self.predict_all_opponents()
            for s_new in self.frame.states:
                prob = 0.0
                for s, b in self.belief.items():
                    if b < 1e-12:
                        continue
                    for opp_combo, combo_prob in self._opponent_action_combos(opp_preds):
                        joint = self._make_joint_action(own_action, opp_combo)
                        t = self.frame.transitions.get(s, {}).get(joint, {})
                        t_prob = t.get(s_new, 0.0)
                        o = self.frame.obs_function.get(s_new, {}).get(joint, {})
                        o_prob = o.get(observation, 0.0)
                        prob += b * t_prob * o_prob * combo_prob
                if prob > 1e-12:
                    new_belief[s_new] = prob

        # Normalize
        total = sum(new_belief.values())
        if total > 0:
            self.belief = {s: p / total for s, p in new_belief.items()}
        # else: keep old belief (zero-probability observation)

        return dict(self.belief)

    def update_model_beliefs(
        self,
        own_action: str,
        observation: str,
        opponent_obs: Optional[dict[str, str]] = None,
    ) -> None:
        """Update beliefs about which opponent model is correct.

        Uses Bayesian update: P(model|obs) proportional to P(obs|model) * P(model).
        """
        for oid, models in self.opponent_models.items():
            priors = self.model_priors[oid]
            posteriors = []

            for i, (model, prior) in enumerate(zip(models, priors)):
                if prior < 1e-12:
                    posteriors.append(0.0)
                    continue
                # Likelihood: how well does this model predict what we saw?
                pred = model.predict_action()
                # We don't directly observe opponent action, but we can
                # compute consistency with our observation
                likelihood = self._model_observation_likelihood(
                    model, own_action, observation
                )
                posteriors.append(prior * likelihood)

            total = sum(posteriors)
            if total > 0:
                self.model_priors[oid] = [p / total for p in posteriors]

    def _model_observation_likelihood(
        self,
        model: IntentionalModel,
        own_action: str,
        observation: str,
    ) -> float:
        """Compute P(observation | model) by marginalizing over model's actions."""
        pred = model.predict_action()
        likelihood = 0.0
        for opp_action, opp_prob in pred.items():
            if opp_prob < 1e-12:
                continue
            opp_dict = {model.agent_id: opp_action}
            joint = self._make_joint_action(own_action, opp_dict)
            for s, b in self.belief.items():
                if b < 1e-12:
                    continue
                for s_new in self.frame.states:
                    t = self.frame.transitions.get(s, {}).get(joint, {})
                    o = self.frame.obs_function.get(s_new, {}).get(joint, {})
                    likelihood += b * t.get(s_new, 0.0) * o.get(observation, 0.0) * opp_prob
        return likelihood

    def solve(self, horizon: int = 10, method: str = "qmdp") -> IPOMDPResult:
        """Solve the I-POMDP by converting to POMDP with opponent predictions.

        Steps:
        1. Predict opponent actions from models
        2. Convert to single-agent POMDP by marginalizing opponents
        3. Solve POMDP
        4. Return policy and opponent predictions
        """
        opp_preds = self.predict_all_opponents()
        pomdp = _frame_to_pomdp(self.frame, opp_preds)

        if method == "qmdp":
            result = qmdp(pomdp, horizon=horizon, discount=self.frame.discount)
        elif method == "pbvi":
            result = pbvi(pomdp, n_points=50, horizon=horizon,
                         discount=self.frame.discount)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract policy from alpha vectors
        policy: dict[str, dict[str, float]] = {}
        belief_key = _belief_key(self.belief)
        if result.alpha_vectors:
            best_val = float('-inf')
            best_action = self.frame.actions[0]
            for av in result.alpha_vectors:
                val = av.dot(self.belief)
                if val > best_val:
                    best_val = val
                    best_action = av.action
            policy[belief_key] = {best_action: 1.0}
            value = best_val
        else:
            policy[belief_key] = {self.frame.actions[0]: 1.0}
            value = 0.0

        return IPOMDPResult(
            policy=policy,
            value=value,
            iterations=result.iterations,
            opponent_predictions=opp_preds,
        )

    def best_action(self, horizon: int = 10, method: str = "qmdp") -> str:
        """Get the best action for the current belief state."""
        result = self.solve(horizon=horizon, method=method)
        belief_key = _belief_key(self.belief)
        action_dist = result.policy.get(belief_key, {})
        if not action_dist:
            return self.frame.actions[0]
        return max(action_dist, key=action_dist.get)

    def simulate(
        self,
        true_state: str,
        opponent_policies: dict[str, dict[str, dict[str, float]]],
        steps: int = 10,
        horizon: int = 10,
    ) -> list[dict[str, Any]]:
        """Simulate the I-POMDP agent interacting with opponents.

        Args:
            true_state: True initial physical state
            opponent_policies: opp_id -> {state -> {action: prob}}
            steps: Number of simulation steps
            horizon: Planning horizon for each step
        """
        trajectory: list[dict[str, Any]] = []
        state = true_state
        total_reward = 0.0

        for step in range(steps):
            # Agent chooses action
            action = self.best_action(horizon=horizon)

            # Opponents choose actions
            opp_actions = {}
            for oid, pol in opponent_policies.items():
                if state in pol:
                    dist = pol[state]
                    actions_list = list(dist.keys())
                    probs = list(dist.values())
                    opp_actions[oid] = random.choices(actions_list, weights=probs, k=1)[0]
                else:
                    opp_actions[oid] = random.choice(
                        self.opponent_models[oid][0].frame.actions
                    )

            # Joint action
            joint = self._make_joint_action(action, opp_actions)

            # Reward
            reward = self.frame.rewards.get(state, {}).get(joint, 0.0)
            total_reward += reward * (self.frame.discount ** step)

            # Transition
            t_dist = self.frame.transitions.get(state, {}).get(joint, {})
            if t_dist:
                states_list = list(t_dist.keys())
                probs = list(t_dist.values())
                new_state = random.choices(states_list, weights=probs, k=1)[0]
            else:
                new_state = state

            # Observation
            o_dist = self.frame.obs_function.get(new_state, {}).get(joint, {})
            if o_dist:
                obs_list = list(o_dist.keys())
                obs_probs = list(o_dist.values())
                obs = random.choices(obs_list, weights=obs_probs, k=1)[0]
            else:
                obs = self.frame.observations[0] if self.frame.observations else "none"

            trajectory.append({
                "step": step,
                "state": state,
                "action": action,
                "opponent_actions": dict(opp_actions),
                "joint_action": joint,
                "reward": reward,
                "observation": obs,
                "belief": dict(self.belief),
                "new_state": new_state,
            })

            # Update beliefs
            self.update_belief(action, obs, opp_actions)
            self.update_model_beliefs(action, obs)

            state = new_state

        return trajectory

    def _make_joint_action(
        self, own_action: str, opp_actions: dict[str, str]
    ) -> tuple:
        """Create joint action tuple in canonical order."""
        # Sort by agent id, own agent first
        parts = [(self.agent_id, own_action)]
        for oid in sorted(opp_actions.keys()):
            parts.append((oid, opp_actions[oid]))
        return tuple(a for _, a in parts)

    def _opponent_action_combos(
        self, opp_preds: dict[str, dict[str, float]]
    ):
        """Enumerate all opponent action combinations with probabilities."""
        if not opp_preds:
            yield {}, 1.0
            return

        opp_ids = sorted(opp_preds.keys())
        opp_action_lists = []
        opp_prob_lists = []
        for oid in opp_ids:
            pred = opp_preds[oid]
            actions = list(pred.keys())
            probs = list(pred.values())
            opp_action_lists.append(actions)
            opp_prob_lists.append(probs)

        for combo_indices in itertools.product(
            *[range(len(al)) for al in opp_action_lists]
        ):
            combo = {}
            prob = 1.0
            for k, idx in enumerate(combo_indices):
                combo[opp_ids[k]] = opp_action_lists[k][idx]
                prob *= opp_prob_lists[k][idx]
            if prob > 1e-12:
                yield combo, prob


# ---------------------------------------------------------------------------
#  Theory of Mind
# ---------------------------------------------------------------------------

class TheoryOfMind:
    """Theory of Mind reasoning over I-POMDP models.

    Provides capabilities for:
    - Prediction: What will the other agent do?
    - Explanation: Why did the other agent do that?
    - Deception: Can I manipulate the other agent's beliefs?
    - Perspective taking: What does the other agent believe?
    """

    def __init__(self, ipomdp: IPOMDP):
        self.ipomdp = ipomdp

    def predict(self, opponent_id: str) -> dict[str, float]:
        """Predict what opponent will do next."""
        return self.ipomdp.predict_opponent(opponent_id)

    def explain_action(
        self,
        opponent_id: str,
        observed_action: str,
    ) -> dict[str, float]:
        """Explain why opponent took an action by computing model posteriors.

        Returns updated model probabilities after observing the action.
        Bayesian: P(model | action) ~ P(action | model) * P(model)
        """
        models = self.ipomdp.opponent_models[opponent_id]
        priors = self.ipomdp.model_priors[opponent_id]

        posteriors = []
        for model, prior in zip(models, priors):
            pred = model.predict_action()
            likelihood = pred.get(observed_action, 0.0)
            posteriors.append(prior * likelihood)

        total = sum(posteriors)
        if total > 0:
            posteriors = [p / total for p in posteriors]
        else:
            posteriors = list(priors)  # No update if impossible

        return {f"model_{i}": p for i, p in enumerate(posteriors)}

    def perspective_take(self, opponent_id: str) -> dict[str, float]:
        """What does the opponent believe about the physical state?

        Weighted average of beliefs across opponent models.
        """
        models = self.ipomdp.opponent_models[opponent_id]
        priors = self.ipomdp.model_priors[opponent_id]

        combined_belief: dict[str, float] = {}
        for model, prior in zip(models, priors):
            if prior < 1e-12:
                continue
            for s, b in model.belief.items():
                combined_belief[s] = combined_belief.get(s, 0.0) + prior * b

        total = sum(combined_belief.values())
        if total > 0:
            return {s: p / total for s, p in combined_belief.items()}
        return dict(self.ipomdp.belief)

    def information_advantage(self, opponent_id: str) -> float:
        """Measure information advantage: how much more does agent know?

        Positive = agent has more precise beliefs (lower entropy).
        """
        own_entropy = _entropy(self.ipomdp.belief)
        opp_belief = self.perspective_take(opponent_id)
        opp_entropy = _entropy(opp_belief)
        return opp_entropy - own_entropy

    def deception_value(
        self,
        opponent_id: str,
        target_action: str,
    ) -> dict[str, float]:
        """Compute how much each of our actions induces a target opponent action.

        For each of our actions, simulate one step and predict opponent's
        subsequent action. Return our action -> probability of target action.
        """
        results: dict[str, float] = {}
        models = self.ipomdp.opponent_models[opponent_id]
        priors = self.ipomdp.model_priors[opponent_id]

        for own_action in self.ipomdp.frame.actions:
            target_prob = 0.0
            for model, prior in zip(models, priors):
                if prior < 1e-12:
                    continue
                # If we take own_action, how does it affect opponent model?
                # The opponent updates their belief based on their observation
                # For simplicity, estimate probability opponent takes target_action
                pred = model.predict_action()
                target_prob += prior * pred.get(target_action, 0.0)
            results[own_action] = target_prob

        return results

    def belief_divergence(self, opponent_id: str) -> float:
        """KL divergence between own and opponent beliefs.

        Measures how differently the agents see the world.
        """
        own = self.ipomdp.belief
        opp = self.perspective_take(opponent_id)
        return _kl_divergence(own, opp)


# ---------------------------------------------------------------------------
#  Level-k Analysis
# ---------------------------------------------------------------------------

def level_k_analysis(
    frames: dict[str, Frame],
    initial_beliefs: dict[str, dict[str, float]],
    max_level: int = 3,
    horizon: int = 10,
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute level-k strategies for all agents.

    Level 0: Random uniform policy
    Level 1: Best response to level 0
    Level k: Best response to level k-1

    Returns: agent_id -> {level -> action_distribution}
    """
    agent_ids = sorted(frames.keys())
    # Level 0: uniform random
    level_policies: dict[int, dict[str, dict[str, dict[str, float]]]] = {}

    # Level 0: uniform policies
    level_policies[0] = {}
    for aid in agent_ids:
        frame = frames[aid]
        n = len(frame.actions)
        uniform = {a: 1.0 / n for a in frame.actions}
        level_policies[0][aid] = {s: dict(uniform) for s in frame.states}

    # Build level k from level k-1
    for k in range(1, max_level + 1):
        level_policies[k] = {}
        for aid in agent_ids:
            frame = frames[aid]
            # Build opponent models at level k-1
            opp_preds = {}
            for oid in agent_ids:
                if oid == aid:
                    continue
                # Opponent's predicted action from their level k-1 policy
                opp_frame = frames[oid]
                opp_belief = initial_beliefs[oid]
                action_dist: dict[str, float] = {}
                for a in opp_frame.actions:
                    action_dist[a] = 0.0
                for s, b in opp_belief.items():
                    if b > 0 and s in level_policies[k - 1][oid]:
                        for a, p in level_policies[k - 1][oid][s].items():
                            action_dist[a] = action_dist.get(a, 0.0) + b * p
                total = sum(action_dist.values())
                if total > 0:
                    opp_preds[oid] = {a: p / total for a, p in action_dist.items()}
                else:
                    n = len(opp_frame.actions)
                    opp_preds[oid] = {a: 1.0 / n for a in opp_frame.actions}

            # Solve POMDP for this agent given opponent predictions
            pomdp = _frame_to_pomdp(frame, opp_preds)
            result = qmdp(pomdp, horizon=horizon, discount=frame.discount)

            # Extract policy
            policy = {}
            for s in frame.states:
                point_belief = {st: (1.0 if st == s else 0.0) for st in frame.states}
                if result.alpha_vectors:
                    best_val = float('-inf')
                    best_action = frame.actions[0]
                    for av in result.alpha_vectors:
                        val = av.dot(point_belief)
                        if val > best_val:
                            best_val = val
                            best_action = av.action
                    policy[s] = {best_action: 1.0}
                else:
                    n = len(frame.actions)
                    policy[s] = {a: 1.0 / n for a in frame.actions}
            level_policies[k][aid] = policy

    # Summarize: for each agent, what does each level prescribe?
    result_dict: dict[str, dict[int, dict[str, float]]] = {}
    for aid in agent_ids:
        result_dict[aid] = {}
        for k in range(max_level + 1):
            belief = initial_beliefs[aid]
            action_dist: dict[str, float] = {}
            for s, b in belief.items():
                if b > 0 and s in level_policies[k][aid]:
                    for a, p in level_policies[k][aid][s].items():
                        action_dist[a] = action_dist.get(a, 0.0) + b * p
            total = sum(action_dist.values())
            if total > 0:
                result_dict[aid][k] = {a: p / total for a, p in action_dist.items()}
            else:
                n = len(frames[aid].actions)
                result_dict[aid][k] = {a: 1.0 / n for a in frames[aid].actions}

    return result_dict


def find_nash_belief(
    frames: dict[str, Frame],
    initial_beliefs: dict[str, dict[str, float]],
    max_iterations: int = 20,
    horizon: int = 10,
    tolerance: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Find Nash equilibrium in belief space via iterative best response.

    Each agent alternates solving their POMDP given the other's current policy.
    Converges when no agent changes their action distribution.
    """
    agent_ids = sorted(frames.keys())

    # Initialize with uniform policies
    policies: dict[str, dict[str, dict[str, float]]] = {}
    for aid in agent_ids:
        frame = frames[aid]
        n = len(frame.actions)
        uniform = {a: 1.0 / n for a in frame.actions}
        policies[aid] = {s: dict(uniform) for s in frame.states}

    for iteration in range(max_iterations):
        changed = False
        for aid in agent_ids:
            frame = frames[aid]
            # Opponent predictions from current policies
            opp_preds = {}
            for oid in agent_ids:
                if oid == aid:
                    continue
                opp_belief = initial_beliefs[oid]
                action_dist: dict[str, float] = {}
                for s, b in opp_belief.items():
                    if b > 0 and s in policies[oid]:
                        for a, p in policies[oid][s].items():
                            action_dist[a] = action_dist.get(a, 0.0) + b * p
                total = sum(action_dist.values())
                if total > 0:
                    opp_preds[oid] = {a: p / total for a, p in action_dist.items()}
                else:
                    n = len(frames[oid].actions)
                    opp_preds[oid] = {a: 1.0 / n for a in frames[oid].actions}

            # Best response
            pomdp = _frame_to_pomdp(frame, opp_preds)
            result = qmdp(pomdp, horizon=horizon, discount=frame.discount)

            new_policy = {}
            for s in frame.states:
                point_belief = {st: (1.0 if st == s else 0.0) for st in frame.states}
                if result.alpha_vectors:
                    best_val = float('-inf')
                    best_action = frame.actions[0]
                    for av in result.alpha_vectors:
                        val = av.dot(point_belief)
                        if val > best_val:
                            best_val = val
                            best_action = av.action
                    new_policy[s] = {best_action: 1.0}
                else:
                    n = len(frame.actions)
                    new_policy[s] = {a: 1.0 / n for a in frame.actions}

            # Check convergence
            for s in frame.states:
                old_dist = policies[aid].get(s, {})
                new_dist = new_policy.get(s, {})
                for a in set(old_dist.keys()) | set(new_dist.keys()):
                    if abs(old_dist.get(a, 0) - new_dist.get(a, 0)) > tolerance:
                        changed = True
                        break

            policies[aid] = new_policy

        if not changed:
            break

    # Return equilibrium action distributions
    eq: dict[str, dict[str, float]] = {}
    for aid in agent_ids:
        belief = initial_beliefs[aid]
        action_dist: dict[str, float] = {}
        for s, b in belief.items():
            if b > 0 and s in policies[aid]:
                for a, p in policies[aid][s].items():
                    action_dist[a] = action_dist.get(a, 0.0) + b * p
        total = sum(action_dist.values())
        if total > 0:
            eq[aid] = {a: p / total for a, p in action_dist.items()}
        else:
            n = len(frames[aid].actions)
            eq[aid] = {a: 1.0 / n for a in frames[aid].actions}

    return eq


# ---------------------------------------------------------------------------
#  Helper Functions
# ---------------------------------------------------------------------------

def _frame_to_pomdp(
    frame: Frame,
    opponent_predictions: dict[str, dict[str, float]],
) -> POMDP:
    """Convert a Frame + opponent predictions to a single-agent POMDP.

    Marginalizes over opponent actions weighted by their predicted distributions.
    """
    states = frame.states
    actions = frame.actions
    observations = frame.observations

    # Build marginalized transition and observation functions
    transitions: dict[str, dict[str, dict[str, float]]] = {}
    obs_fn: dict[str, dict[str, dict[str, float]]] = {}
    rewards: dict[str, dict[str, float]] = {}

    for s in states:
        transitions[s] = {}
        rewards[s] = {}
        for a in actions:
            t_marginal: dict[str, float] = {}
            r_marginal = 0.0
            total_weight = 0.0

            # Enumerate opponent action combinations
            for opp_combo, combo_prob in _enumerate_opp_combos(
                frame.agent_id, opponent_predictions
            ):
                joint = _build_joint(frame.agent_id, a, opp_combo)
                t = frame.transitions.get(s, {}).get(joint, {})
                for s_new, tp in t.items():
                    t_marginal[s_new] = t_marginal.get(s_new, 0.0) + combo_prob * tp
                r = frame.rewards.get(s, {}).get(joint, 0.0)
                r_marginal += combo_prob * r
                total_weight += combo_prob

            transitions[s][a] = t_marginal
            rewards[s][a] = r_marginal

    for s_new in states:
        obs_fn[s_new] = {}
        for a in actions:
            o_marginal: dict[str, float] = {}
            for opp_combo, combo_prob in _enumerate_opp_combos(
                frame.agent_id, opponent_predictions
            ):
                joint = _build_joint(frame.agent_id, a, opp_combo)
                o = frame.obs_function.get(s_new, {}).get(joint, {})
                for obs, op in o.items():
                    o_marginal[obs] = o_marginal.get(obs, 0.0) + combo_prob * op
            obs_fn[s_new][a] = o_marginal

    return POMDP(
        states=states,
        actions=actions,
        observations=observations,
        transitions=transitions,
        obs_function=obs_fn,
        rewards=rewards,
        discount=frame.discount,
    )


def _enumerate_opp_combos(
    own_id: str,
    opp_predictions: dict[str, dict[str, float]],
):
    """Yield (opp_combo_dict, probability) for all opponent action combos."""
    if not opp_predictions:
        yield {}, 1.0
        return

    opp_ids = sorted(opp_predictions.keys())
    opp_actions = [list(opp_predictions[oid].keys()) for oid in opp_ids]
    opp_probs = [list(opp_predictions[oid].values()) for oid in opp_ids]

    for combo in itertools.product(*[range(len(al)) for al in opp_actions]):
        d = {}
        p = 1.0
        for k, idx in enumerate(combo):
            d[opp_ids[k]] = opp_actions[k][idx]
            p *= opp_probs[k][idx]
        if p > 1e-12:
            yield d, p


def _build_joint(own_id: str, own_action: str, opp_combo: dict[str, str]) -> tuple:
    """Build joint action tuple in canonical order (own agent first)."""
    parts = [own_action]
    for oid in sorted(opp_combo.keys()):
        parts.append(opp_combo[oid])
    return tuple(parts)


def _belief_key(belief: dict[str, float]) -> str:
    """Create a hashable key from a belief dict."""
    items = sorted(belief.items())
    return "|".join(f"{s}:{p:.4f}" for s, p in items if p > 1e-6)


def _entropy(dist: dict[str, float]) -> float:
    """Shannon entropy of a distribution."""
    h = 0.0
    for p in dist.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL divergence D(p || q)."""
    kl = 0.0
    for s in p:
        ps = p.get(s, 0.0)
        qs = q.get(s, 1e-12)
        if ps > 1e-12:
            kl += ps * math.log2(ps / max(qs, 1e-12))
    return kl


# ---------------------------------------------------------------------------
#  Example Problems
# ---------------------------------------------------------------------------

def build_multi_agent_tiger() -> dict[str, Any]:
    """Multi-agent Tiger problem.

    Two agents face two doors. A tiger is behind one door.
    Each agent can: listen, open-left, open-right.
    Listening gives a noisy observation about tiger location.
    Opening the tiger door gives -100, treasure door gives +10.
    If both open same correct door: +20 (cooperation bonus).
    """
    states = ["tiger-left", "tiger-right"]
    actions = ["listen", "open-left", "open-right"]
    observations = ["hear-left", "hear-right"]

    def make_frame(agent_id: str, other_id: str) -> Frame:
        transitions = {}
        obs_function = {}
        rewards_dict = {}

        for s in states:
            transitions[s] = {}
            rewards_dict[s] = {}
            for a1 in actions:
                for a2 in actions:
                    joint = (a1, a2) if agent_id < other_id else (a2, a1)

                    # If anyone opens a door, tiger resets uniformly
                    if a1 != "listen" or a2 != "listen":
                        transitions[s][joint] = {
                            "tiger-left": 0.5, "tiger-right": 0.5
                        }
                    else:
                        transitions[s][joint] = {s: 1.0}

                    # Rewards for this agent
                    if a1 == "listen":
                        r = -1.0  # Listening cost
                    elif a1 == "open-left":
                        if s == "tiger-left":
                            r = -100.0  # Opened tiger door
                        else:
                            r = 10.0
                            if a2 == "open-left":
                                r = 20.0  # Cooperation bonus
                    else:  # open-right
                        if s == "tiger-right":
                            r = -100.0
                        else:
                            r = 10.0
                            if a2 == "open-right":
                                r = 20.0
                    rewards_dict[s][joint] = r

        for s_new in states:
            obs_function[s_new] = {}
            for a1 in actions:
                for a2 in actions:
                    joint = (a1, a2) if agent_id < other_id else (a2, a1)
                    if a1 == "listen":
                        if s_new == "tiger-left":
                            obs_function[s_new][joint] = {
                                "hear-left": 0.85, "hear-right": 0.15
                            }
                        else:
                            obs_function[s_new][joint] = {
                                "hear-left": 0.15, "hear-right": 0.85
                            }
                    else:
                        obs_function[s_new][joint] = {
                            "hear-left": 0.5, "hear-right": 0.5
                        }

        return Frame(
            agent_id=agent_id,
            states=states,
            actions=actions,
            observations=observations,
            transitions=transitions,
            obs_function=obs_function,
            rewards=rewards_dict,
            discount=0.9,
        )

    frame_a = make_frame("A", "B")
    frame_b = make_frame("B", "A")

    return {
        "frames": {"A": frame_a, "B": frame_b},
        "initial_beliefs": {
            "A": {"tiger-left": 0.5, "tiger-right": 0.5},
            "B": {"tiger-left": 0.5, "tiger-right": 0.5},
        },
        "description": "Multi-agent Tiger: cooperative door-opening under uncertainty",
    }


def build_pursuit_evasion() -> dict[str, Any]:
    """Simple pursuit-evasion on a 1D grid.

    Pursuer (P) tries to catch Evader (E) on a 5-cell grid.
    P can move left, right, or stay. E can move left, right, or stay.
    P gets +100 for catching E, -1 per step.
    E gets +1 per step not caught, -100 for being caught.
    P observes noisy distance to E.
    """
    n_cells = 5
    states = [f"P{p}E{e}" for p in range(n_cells) for e in range(n_cells)]
    p_actions = ["left", "stay", "right"]
    e_actions = ["left", "stay", "right"]
    p_observations = ["near", "medium", "far"]

    def make_pursuer_frame() -> Frame:
        transitions = {}
        obs_function = {}
        rewards_dict = {}

        for s in states:
            pp = int(s[1])
            ep = int(s[3])
            transitions[s] = {}
            rewards_dict[s] = {}

            for pa in p_actions:
                for ea in e_actions:
                    joint = (pa, ea)
                    # Move pursuer
                    new_pp = pp + (-1 if pa == "left" else 1 if pa == "right" else 0)
                    new_pp = max(0, min(n_cells - 1, new_pp))
                    # Move evader
                    new_ep = ep + (-1 if ea == "left" else 1 if ea == "right" else 0)
                    new_ep = max(0, min(n_cells - 1, new_ep))

                    new_state = f"P{new_pp}E{new_ep}"
                    transitions[s][joint] = {new_state: 1.0}

                    caught = (new_pp == new_ep)
                    rewards_dict[s][joint] = 100.0 if caught else -1.0

        for s_new in states:
            pp = int(s_new[1])
            ep = int(s_new[3])
            dist = abs(pp - ep)
            obs_function[s_new] = {}
            for pa in p_actions:
                for ea in e_actions:
                    joint = (pa, ea)
                    if dist <= 1:
                        obs_function[s_new][joint] = {"near": 0.8, "medium": 0.15, "far": 0.05}
                    elif dist <= 2:
                        obs_function[s_new][joint] = {"near": 0.15, "medium": 0.7, "far": 0.15}
                    else:
                        obs_function[s_new][joint] = {"near": 0.05, "medium": 0.15, "far": 0.8}

        return Frame(
            agent_id="pursuer",
            states=states,
            actions=p_actions,
            observations=p_observations,
            transitions=transitions,
            obs_function=obs_function,
            rewards=rewards_dict,
            discount=0.95,
        )

    def make_evader_frame() -> Frame:
        transitions = {}
        obs_function = {}
        rewards_dict = {}
        e_observations = ["near", "medium", "far"]

        for s in states:
            pp = int(s[1])
            ep = int(s[3])
            transitions[s] = {}
            rewards_dict[s] = {}

            for pa in p_actions:
                for ea in e_actions:
                    joint = (pa, ea)
                    new_pp = pp + (-1 if pa == "left" else 1 if pa == "right" else 0)
                    new_pp = max(0, min(n_cells - 1, new_pp))
                    new_ep = ep + (-1 if ea == "left" else 1 if ea == "right" else 0)
                    new_ep = max(0, min(n_cells - 1, new_ep))

                    new_state = f"P{new_pp}E{new_ep}"
                    transitions[s][joint] = {new_state: 1.0}

                    caught = (new_pp == new_ep)
                    rewards_dict[s][joint] = -100.0 if caught else 1.0

        for s_new in states:
            pp = int(s_new[1])
            ep = int(s_new[3])
            dist = abs(pp - ep)
            obs_function[s_new] = {}
            for pa in p_actions:
                for ea in e_actions:
                    joint = (pa, ea)
                    if dist <= 1:
                        obs_function[s_new][joint] = {"near": 0.8, "medium": 0.15, "far": 0.05}
                    elif dist <= 2:
                        obs_function[s_new][joint] = {"near": 0.15, "medium": 0.7, "far": 0.15}
                    else:
                        obs_function[s_new][joint] = {"near": 0.05, "medium": 0.15, "far": 0.8}

        return Frame(
            agent_id="evader",
            states=states,
            actions=e_actions,
            observations=e_observations,
            transitions=transitions,
            obs_function=obs_function,
            rewards=rewards_dict,
            discount=0.95,
        )

    p_frame = make_pursuer_frame()
    e_frame = make_evader_frame()

    # Initial belief: pursuer at 0, evader uniformly distributed
    p_belief = {}
    for s in states:
        pp = int(s[1])
        if pp == 0:
            p_belief[s] = 1.0 / n_cells
        else:
            p_belief[s] = 0.0
    e_belief = {}
    for s in states:
        ep = int(s[3])
        if ep == n_cells - 1:
            e_belief[s] = 1.0 / n_cells
        else:
            e_belief[s] = 0.0

    return {
        "frames": {"pursuer": p_frame, "evader": e_frame},
        "initial_beliefs": {"pursuer": p_belief, "evader": e_belief},
        "description": "Pursuit-evasion on 1D grid with noisy distance observations",
    }


def build_signaling_game() -> dict[str, Any]:
    """Signaling game: sender-receiver with information asymmetry.

    Nature chooses a state (high/low quality).
    Sender knows the state and sends a signal (cheap talk or costly).
    Receiver observes signal and chooses action (invest/pass).

    High quality + invest -> both benefit
    Low quality + invest -> receiver loses, sender gains
    Costly signal -> sender pays cost
    """
    states = ["high", "low"]
    sender_actions = ["signal-high", "signal-low", "signal-costly"]
    receiver_actions = ["invest", "pass"]
    sender_obs = ["none"]  # Sender knows state perfectly
    receiver_obs = ["see-high", "see-low", "see-costly"]

    def make_sender_frame() -> Frame:
        transitions = {}
        obs_function = {}
        rewards_dict = {}

        for s in states:
            transitions[s] = {}
            rewards_dict[s] = {}
            for sa in sender_actions:
                for ra in receiver_actions:
                    joint = (sa, ra)
                    # State doesn't change
                    transitions[s][joint] = {s: 1.0}

                    # Sender reward
                    if ra == "invest":
                        r = 10.0  # Sender always benefits from investment
                    else:
                        r = 0.0
                    if sa == "signal-costly":
                        r -= 5.0  # Costly signal
                    rewards_dict[s][joint] = r

        for s_new in states:
            obs_function[s_new] = {}
            for sa in sender_actions:
                for ra in receiver_actions:
                    joint = (sa, ra)
                    obs_function[s_new][joint] = {"none": 1.0}

        return Frame(
            agent_id="sender",
            states=states,
            actions=sender_actions,
            observations=sender_obs,
            transitions=transitions,
            obs_function=obs_function,
            rewards=rewards_dict,
            discount=0.9,
        )

    def make_receiver_frame() -> Frame:
        transitions = {}
        obs_function = {}
        rewards_dict = {}

        for s in states:
            transitions[s] = {}
            rewards_dict[s] = {}
            for sa in sender_actions:
                for ra in receiver_actions:
                    joint = (sa, ra)
                    transitions[s][joint] = {s: 1.0}

                    if ra == "invest":
                        if s == "high":
                            r = 10.0  # Good investment
                        else:
                            r = -10.0  # Bad investment
                    else:
                        r = 0.0
                    rewards_dict[s][joint] = r

        for s_new in states:
            obs_function[s_new] = {}
            for sa in sender_actions:
                for ra in receiver_actions:
                    joint = (sa, ra)
                    # Receiver sees the signal
                    if sa == "signal-high":
                        obs_function[s_new][joint] = {"see-high": 1.0}
                    elif sa == "signal-low":
                        obs_function[s_new][joint] = {"see-low": 1.0}
                    else:
                        obs_function[s_new][joint] = {"see-costly": 1.0}

        return Frame(
            agent_id="receiver",
            states=states,
            actions=receiver_actions,
            observations=receiver_obs,
            transitions=transitions,
            obs_function=obs_function,
            rewards=rewards_dict,
            discount=0.9,
        )

    return {
        "frames": {"sender": make_sender_frame(), "receiver": make_receiver_frame()},
        "initial_beliefs": {
            "sender": {"high": 0.5, "low": 0.5},
            "receiver": {"high": 0.5, "low": 0.5},
        },
        "description": "Signaling game: sender communicates quality, receiver decides to invest",
    }


def build_coordination_game() -> dict[str, Any]:
    """Coordination game with partial observability.

    Two agents must coordinate on meeting at location A or B.
    Each prefers one location but gets bonus for coordinating.
    Noisy observation of other agent's location choice.
    """
    states = ["state"]  # Single physical state (pure coordination)
    actions = ["go-A", "go-B"]
    observations = ["hint-A", "hint-B"]

    def make_frame(agent_id: str, other_id: str, preferred: str) -> Frame:
        transitions = {}
        obs_function = {}
        rewards_dict = {}

        for s in states:
            transitions[s] = {}
            rewards_dict[s] = {}
            for a1 in actions:
                for a2 in actions:
                    joint = (a1, a2) if agent_id < other_id else (a2, a1)
                    transitions[s][joint] = {"state": 1.0}

                    # Coordination bonus
                    if a1 == a2:
                        r = 10.0
                        if a1 == preferred:
                            r = 15.0  # Extra for preferred location
                    else:
                        r = 0.0
                    rewards_dict[s][joint] = r

        for s_new in states:
            obs_function[s_new] = {}
            for a1 in actions:
                for a2 in actions:
                    joint = (a1, a2) if agent_id < other_id else (a2, a1)
                    # Noisy hint about other's choice
                    if a2 == "go-A":
                        obs_function[s_new][joint] = {"hint-A": 0.7, "hint-B": 0.3}
                    else:
                        obs_function[s_new][joint] = {"hint-A": 0.3, "hint-B": 0.7}

        return Frame(
            agent_id=agent_id,
            states=states,
            actions=actions,
            observations=observations,
            transitions=transitions,
            obs_function=obs_function,
            rewards=rewards_dict,
            discount=0.9,
        )

    return {
        "frames": {
            "alice": make_frame("alice", "bob", "go-A"),
            "bob": make_frame("bob", "alice", "go-B"),
        },
        "initial_beliefs": {
            "alice": {"state": 1.0},
            "bob": {"state": 1.0},
        },
        "description": "Coordination game: agents must meet at same location with different preferences",
    }
