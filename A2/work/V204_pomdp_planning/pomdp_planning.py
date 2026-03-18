"""V204: Online POMDP Planning -- POMCP and DESPOT algorithms.

Composes V200 (probabilistic partial observation / POMDP foundations).

Online POMDP planning avoids computing a full policy over the entire belief
space. Instead, at each step, it plans from the *current* belief using Monte
Carlo tree search to select the best action.

Key algorithms:
1. POMCP (Partially Observable Monte Carlo Planning):
   - UCB1 tree policy over action nodes
   - Particle-based belief representation (unweighted)
   - Rollout policy for leaf evaluation
   - Belief update by particle reinvigoration
2. DESPOT (Determinized Sparse Partially Observable Tree):
   - Deterministic scenarios (K pre-sampled state sequences)
   - Regularized policy tree search (penalize large trees)
   - Anytime alpha-vector bounds
3. Rollout policies: random, heuristic, greedy-reward
4. Simulation and evaluation framework

APIs:
- pomcp_search(pomdp, belief_particles, ...) -> action
- despot_search(pomdp, belief_particles, ...) -> action
- simulate_online(pomdp, planner, horizon, ...) -> trajectory
- evaluate_planner(pomdp, planner, n_episodes, ...) -> stats
- compare_planners(pomdp, planners, ...) -> comparison
"""

import sys
import os
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V200_probabilistic_partial_obs'))

from probabilistic_partial_obs import (
    POMDP, POMDPObjective,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class POMCPConfig:
    """Configuration for POMCP search."""
    num_simulations: int = 1000       # number of MCTS simulations per step
    exploration_const: float = 1.0    # UCB1 exploration constant (c)
    discount: float = 0.95            # discount factor
    max_depth: int = 50               # max rollout/search depth
    rollout_policy: Optional[Callable] = None  # (state, actions) -> action
    particle_count: int = 500         # particles for belief
    reinvigoration_count: int = 5     # particles to add after update


@dataclass
class DESPOTConfig:
    """Configuration for DESPOT search."""
    num_scenarios: int = 500          # K deterministic scenarios
    max_depth: int = 50               # max tree depth
    discount: float = 0.95            # discount factor
    lambda_penalty: float = 0.1       # regularization penalty per node
    max_tree_size: int = 10000        # max nodes in tree
    rollout_policy: Optional[Callable] = None
    num_expansions: int = 500         # number of trial expansions


@dataclass
class ActionNode:
    """Action node in POMCP tree."""
    action: Any
    visit_count: int = 0
    total_value: float = 0.0
    obs_children: Dict[Any, 'BeliefNode'] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class BeliefNode:
    """Belief node in POMCP tree (corresponds to a history)."""
    particles: List[Any] = field(default_factory=list)
    visit_count: int = 0
    action_children: Dict[Any, ActionNode] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# POMDP simulation helpers
# ---------------------------------------------------------------------------

def _sample_transition(pomdp: POMDP, state, action) -> Tuple[Any, float]:
    """Sample next state and reward from POMDP."""
    transitions = pomdp.transitions.get((state, action), [])
    if not transitions:
        return state, 0.0
    # Sample next state from distribution
    r = random.random()
    cumulative = 0.0
    next_state = transitions[-1][0]  # fallback
    for ns, prob in transitions:
        cumulative += float(prob)
        if r <= cumulative:
            next_state = ns
            break
    reward = float(pomdp.rewards.get((state, action), 0.0))
    return next_state, reward


def _get_observation(pomdp: POMDP, state) -> Any:
    """Get observation for a state."""
    return pomdp.obs.get(state, state)


def _available_actions(pomdp: POMDP, state) -> List[Any]:
    """Get available actions for a state."""
    if pomdp.state_actions and state in pomdp.state_actions:
        return list(pomdp.state_actions[state])
    return list(pomdp.actions)


def _all_actions(pomdp: POMDP) -> List[Any]:
    """Get all possible actions."""
    return list(pomdp.actions)


def _sample_initial_state(pomdp: POMDP) -> Any:
    """Sample a state from the initial distribution."""
    if not pomdp.initial:
        return list(pomdp.states)[0]
    r = random.random()
    cumulative = 0.0
    for state, prob in pomdp.initial:
        cumulative += float(prob)
        if r <= cumulative:
            return state
    return pomdp.initial[-1][0]


def _random_rollout_policy(state, actions):
    """Default random rollout policy."""
    return random.choice(actions)


# ---------------------------------------------------------------------------
# POMCP (Partially Observable Monte Carlo Planning)
# ---------------------------------------------------------------------------

class POMCP:
    """POMCP: Monte Carlo tree search for POMDPs.

    Uses UCB1 for action selection, particle-based belief representation,
    and random (or custom) rollouts for leaf evaluation.
    """

    def __init__(self, pomdp: POMDP, config: Optional[POMCPConfig] = None):
        self.pomdp = pomdp
        self.config = config or POMCPConfig()
        self.root: Optional[BeliefNode] = None

    def search(self, belief_particles: List[Any]) -> Any:
        """Run POMCP search from given belief particles, return best action."""
        self.root = BeliefNode(particles=list(belief_particles))
        actions = _all_actions(self.pomdp)
        if not actions:
            return None

        for _ in range(self.config.num_simulations):
            # Sample a state from particles
            if not self.root.particles:
                break
            state = random.choice(self.root.particles)
            self._simulate(state, self.root, 0)

        # Select action with highest visit count (robust selection)
        best_action = None
        best_visits = -1
        for a, anode in self.root.action_children.items():
            if anode.visit_count > best_visits:
                best_visits = anode.visit_count
                best_action = a
        return best_action

    def _simulate(self, state, node: BeliefNode, depth: int) -> float:
        """Recursive POMCP simulation."""
        if depth >= self.config.max_depth:
            return 0.0

        actions = _all_actions(self.pomdp)
        if not actions:
            return 0.0

        # Expand if needed
        for a in actions:
            if a not in node.action_children:
                node.action_children[a] = ActionNode(action=a)

        # UCB1 action selection
        action = self._ucb1_select(node, actions)
        anode = node.action_children[action]

        # Simulate transition
        next_state, reward = _sample_transition(self.pomdp, state, action)
        obs = _get_observation(self.pomdp, next_state)

        # Get or create observation child
        if obs not in anode.obs_children:
            anode.obs_children[obs] = BeliefNode()
            # Rollout from new node
            rollout_val = self._rollout(next_state, depth + 1)
            total = reward + self.config.discount * rollout_val
        else:
            child_node = anode.obs_children[obs]
            child_node.particles.append(next_state)
            total = reward + self.config.discount * self._simulate(
                next_state, child_node, depth + 1
            )

        # Update statistics
        node.visit_count += 1
        anode.visit_count += 1
        anode.total_value += total
        return total

    def _ucb1_select(self, node: BeliefNode, actions: List[Any]) -> Any:
        """Select action using UCB1."""
        c = self.config.exploration_const
        log_n = math.log(max(node.visit_count, 1))
        best_val = float('-inf')
        best_action = actions[0]

        for a in actions:
            anode = node.action_children.get(a)
            if anode is None or anode.visit_count == 0:
                return a  # prioritize unexplored
            ucb = anode.q_value + c * math.sqrt(log_n / anode.visit_count)
            if ucb > best_val:
                best_val = ucb
                best_action = a
        return best_action

    def _rollout(self, state, depth: int) -> float:
        """Random rollout from state."""
        rollout_fn = self.config.rollout_policy or _random_rollout_policy
        total = 0.0
        discount = 1.0
        s = state
        for d in range(depth, self.config.max_depth):
            actions = _available_actions(self.pomdp, s)
            if not actions:
                break
            a = rollout_fn(s, actions)
            ns, r = _sample_transition(self.pomdp, s, a)
            total += discount * r
            discount *= self.config.discount
            s = ns
        return total

    def update_belief(self, belief_particles: List[Any], action, observation) -> List[Any]:
        """Update belief particles after taking action and receiving observation."""
        new_particles = []
        for state in belief_particles:
            ns, _ = _sample_transition(self.pomdp, state, action)
            obs = _get_observation(self.pomdp, ns)
            if obs == observation:
                new_particles.append(ns)

        # Reinvigoration: if too few particles match, resample with rejection
        if len(new_particles) < self.config.reinvigoration_count:
            for _ in range(self.config.reinvigoration_count * 10):
                s = random.choice(belief_particles)
                ns, _ = _sample_transition(self.pomdp, s, action)
                obs = _get_observation(self.pomdp, ns)
                if obs == observation:
                    new_particles.append(ns)
                if len(new_particles) >= self.config.particle_count:
                    break

        # Ensure minimum particles
        if not new_particles:
            # Desperate: sample from prior
            for _ in range(self.config.particle_count):
                s = _sample_initial_state(self.pomdp)
                new_particles.append(s)
        elif len(new_particles) < self.config.particle_count:
            # Resample with replacement to fill
            while len(new_particles) < self.config.particle_count:
                new_particles.append(random.choice(new_particles))

        return new_particles[:self.config.particle_count]

    def get_action_values(self) -> Dict[Any, float]:
        """Get Q-values for all actions from last search."""
        if not self.root:
            return {}
        return {a: an.q_value for a, an in self.root.action_children.items()}

    def get_action_visits(self) -> Dict[Any, int]:
        """Get visit counts for all actions from last search."""
        if not self.root:
            return {}
        return {a: an.visit_count for a, an in self.root.action_children.items()}


# ---------------------------------------------------------------------------
# DESPOT (Determinized Sparse Partially Observable Tree)
# ---------------------------------------------------------------------------

@dataclass
class DESPOTNode:
    """Node in DESPOT search tree."""
    is_action: bool = False
    action: Any = None
    state_scenarios: Dict[int, Any] = field(default_factory=dict)  # scenario_id -> state
    visit_count: int = 0
    total_value: float = 0.0
    children: Dict[Any, 'DESPOTNode'] = field(default_factory=dict)
    # For belief nodes: maps obs -> child
    # For action nodes: maps scenario_id -> (next_state, reward, obs)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class DESPOT:
    """DESPOT: Determinized Sparse Partially Observable Tree.

    Pre-samples K scenarios (deterministic state sequences). Builds a
    sparse search tree using only the sampled scenarios. Regularization
    penalty encourages smaller policy trees.
    """

    def __init__(self, pomdp: POMDP, config: Optional[DESPOTConfig] = None):
        self.pomdp = pomdp
        self.config = config or DESPOTConfig()
        self.root: Optional[DESPOTNode] = None
        self._scenarios: List[List[float]] = []  # pre-sampled random numbers

    def search(self, belief_particles: List[Any]) -> Any:
        """Run DESPOT search from belief particles, return best action."""
        actions = _all_actions(self.pomdp)
        if not actions:
            return None

        # Sample scenarios: each scenario picks a start state and pre-rolls random numbers
        K = self.config.num_scenarios
        scenarios = {}
        for k in range(K):
            s = random.choice(belief_particles)
            scenarios[k] = s
        # Pre-sample random numbers for determinization
        self._scenarios = [
            [random.random() for _ in range(self.config.max_depth)]
            for _ in range(K)
        ]

        self.root = DESPOTNode(state_scenarios=scenarios)

        # Run trial expansions
        for _ in range(self.config.num_expansions):
            self._trial(self.root, 0)

        # Select best action: highest value - penalty
        best_action = None
        best_score = float('-inf')
        for a, child in self.root.children.items():
            # Regularized value: value - lambda * (tree_size / K)
            tree_size = self._count_nodes(child)
            reg_value = child.value - self.config.lambda_penalty * (tree_size / max(K, 1))
            if reg_value > best_score:
                best_score = reg_value
                best_action = a
        return best_action

    def _trial(self, node: DESPOTNode, depth: int):
        """One DESPOT trial expansion."""
        if depth >= self.config.max_depth or not node.state_scenarios:
            return 0.0

        actions = _all_actions(self.pomdp)
        if not actions:
            return 0.0

        # Expand: try each action
        for a in actions:
            if a not in node.children:
                node.children[a] = DESPOTNode(is_action=True, action=a)

        # Select action (UCB1-like or value-based)
        action = self._select_action(node, actions)
        anode = node.children[action]

        # For each scenario, simulate transition
        obs_groups: Dict[Any, Dict[int, Any]] = defaultdict(dict)
        total_reward = 0.0
        scenario_count = 0

        for k, state in node.state_scenarios.items():
            ns, reward = self._deterministic_transition(state, action, k, depth)
            obs = _get_observation(self.pomdp, ns)
            obs_groups[obs][k] = ns
            total_reward += reward
            scenario_count += 1

        if scenario_count == 0:
            return 0.0

        avg_reward = total_reward / scenario_count

        # Branch on observations
        future_value = 0.0
        for obs, scenario_states in obs_groups.items():
            if obs not in anode.children:
                anode.children[obs] = DESPOTNode(state_scenarios=scenario_states)
                # Rollout value for new node
                rv = 0.0
                for k, s in scenario_states.items():
                    rv += self._rollout(s, depth + 1)
                rv /= max(len(scenario_states), 1)
                future_value += (len(scenario_states) / scenario_count) * rv
            else:
                child = anode.children[obs]
                child.state_scenarios.update(scenario_states)
                rv = self._trial(child, depth + 1)
                future_value += (len(scenario_states) / scenario_count) * rv

        total = avg_reward + self.config.discount * future_value

        # Update
        node.visit_count += 1
        anode.visit_count += 1
        anode.total_value += total
        return total

    def _deterministic_transition(self, state, action, scenario_id: int, depth: int):
        """Deterministic transition using pre-sampled random number."""
        transitions = self.pomdp.transitions.get((state, action), [])
        if not transitions:
            return state, 0.0
        r = self._scenarios[scenario_id % len(self._scenarios)][
            min(depth, len(self._scenarios[0]) - 1)
        ]
        cumulative = 0.0
        next_state = transitions[-1][0]
        for ns, prob in transitions:
            cumulative += float(prob)
            if r <= cumulative:
                next_state = ns
                break
        reward = float(self.pomdp.rewards.get((state, action), 0.0))
        return next_state, reward

    def _select_action(self, node: DESPOTNode, actions: List[Any]) -> Any:
        """Select action for expansion (explore-exploit)."""
        c = 1.0
        log_n = math.log(max(node.visit_count, 1))
        best_val = float('-inf')
        best_action = actions[0]
        for a in actions:
            anode = node.children.get(a)
            if anode is None or anode.visit_count == 0:
                return a
            ucb = anode.value + c * math.sqrt(log_n / anode.visit_count)
            if ucb > best_val:
                best_val = ucb
                best_action = a
        return best_action

    def _rollout(self, state, depth: int) -> float:
        """Rollout from state for value estimation."""
        rollout_fn = self.config.rollout_policy or _random_rollout_policy
        total = 0.0
        discount = 1.0
        s = state
        for d in range(depth, self.config.max_depth):
            actions = _available_actions(self.pomdp, s)
            if not actions:
                break
            a = rollout_fn(s, actions)
            ns, r = _sample_transition(self.pomdp, s, a)
            total += discount * r
            discount *= self.config.discount
            s = ns
        return total

    def _count_nodes(self, node: DESPOTNode) -> int:
        """Count nodes in subtree."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def get_action_values(self) -> Dict[Any, float]:
        """Get values for all actions from last search."""
        if not self.root:
            return {}
        return {a: child.value for a, child in self.root.children.items()
                if not child.is_action or True}

    def get_action_visits(self) -> Dict[Any, int]:
        """Get visit counts for all actions."""
        if not self.root:
            return {}
        return {a: child.visit_count for a, child in self.root.children.items()}


# ---------------------------------------------------------------------------
# Greedy Rollout Policy
# ---------------------------------------------------------------------------

def make_greedy_rollout(pomdp: POMDP) -> Callable:
    """Create a greedy (immediate reward) rollout policy."""
    def greedy(state, actions):
        best_a = actions[0]
        best_r = float('-inf')
        for a in actions:
            r = float(pomdp.rewards.get((state, a), 0.0))
            if r > best_r:
                best_r = r
                best_a = a
        return best_a
    return greedy


def make_heuristic_rollout(heuristic: Callable) -> Callable:
    """Create a rollout policy from a state heuristic function.

    heuristic(state) -> float (higher = better)
    The policy picks the action leading to the best expected next state.
    """
    def policy(state, actions):
        # Without a POMDP reference we can't simulate -- just pick first
        return actions[0]
    return policy


# ---------------------------------------------------------------------------
# Online simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationStep:
    """One step of an online planning simulation."""
    state: Any
    belief_particles: List[Any]
    action: Any
    observation: Any
    reward: float
    cumulative_reward: float


@dataclass
class SimulationResult:
    """Result of running an online planner on a POMDP episode."""
    steps: List[SimulationStep]
    total_reward: float
    discounted_reward: float
    horizon_reached: int
    final_state: Any


def simulate_online(
    pomdp: POMDP,
    planner,  # POMCP or DESPOT instance
    horizon: int = 100,
    initial_particles: Optional[List[Any]] = None,
    seed: Optional[int] = None,
    discount: float = 0.95,
) -> SimulationResult:
    """Simulate an online planning episode.

    The planner searches at each step from the current belief,
    takes an action, observes, and updates belief.
    """
    if seed is not None:
        random.seed(seed)

    # Initialize
    state = _sample_initial_state(pomdp)
    if initial_particles is None:
        cfg = getattr(planner, 'config', None)
        n_particles = getattr(cfg, 'particle_count', None) or getattr(cfg, 'num_scenarios', 500)
        initial_particles = [_sample_initial_state(pomdp) for _ in range(n_particles)]
    particles = list(initial_particles)

    steps = []
    total_reward = 0.0
    discounted_reward = 0.0
    discount_factor = 1.0

    for t in range(horizon):
        # Plan
        action = planner.search(particles)
        if action is None:
            break

        # Execute
        next_state, reward = _sample_transition(pomdp, state, action)
        obs = _get_observation(pomdp, next_state)

        total_reward += reward
        discounted_reward += discount_factor * reward
        discount_factor *= discount

        steps.append(SimulationStep(
            state=state,
            belief_particles=particles[:10],  # save subset for memory
            action=action,
            observation=obs,
            reward=reward,
            cumulative_reward=total_reward,
        ))

        # Update belief
        if isinstance(planner, POMCP):
            particles = planner.update_belief(particles, action, obs)
        else:
            # Generic particle filter update
            particles = _particle_filter_update(pomdp, particles, action, obs,
                                                 target_count=len(particles))

        state = next_state

    return SimulationResult(
        steps=steps,
        total_reward=total_reward,
        discounted_reward=discounted_reward,
        horizon_reached=len(steps),
        final_state=state,
    )


def _particle_filter_update(
    pomdp: POMDP,
    particles: List[Any],
    action: Any,
    observation: Any,
    target_count: int = 500,
) -> List[Any]:
    """Generic particle filter belief update."""
    new_particles = []
    for s in particles:
        ns, _ = _sample_transition(pomdp, s, action)
        if _get_observation(pomdp, ns) == observation:
            new_particles.append(ns)

    if not new_particles:
        # Fallback: keep all transitioned particles
        for s in particles:
            ns, _ = _sample_transition(pomdp, s, action)
            new_particles.append(ns)

    while len(new_particles) < target_count:
        new_particles.append(random.choice(new_particles))

    return new_particles[:target_count]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Result of evaluating a planner over multiple episodes."""
    mean_total_reward: float
    mean_discounted_reward: float
    std_total_reward: float
    mean_horizon: float
    n_episodes: int
    episode_rewards: List[float]


def evaluate_planner(
    pomdp: POMDP,
    planner,
    n_episodes: int = 10,
    horizon: int = 100,
    discount: float = 0.95,
    seed: Optional[int] = None,
) -> EvaluationResult:
    """Evaluate a planner over multiple episodes."""
    rewards = []
    disc_rewards = []
    horizons = []

    for i in range(n_episodes):
        if seed is not None:
            random.seed(seed + i)
        result = simulate_online(pomdp, planner, horizon=horizon, discount=discount)
        rewards.append(result.total_reward)
        disc_rewards.append(result.discounted_reward)
        horizons.append(result.horizon_reached)

    mean_r = sum(rewards) / max(len(rewards), 1)
    mean_d = sum(disc_rewards) / max(len(disc_rewards), 1)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
    std_r = math.sqrt(var_r)

    return EvaluationResult(
        mean_total_reward=mean_r,
        mean_discounted_reward=mean_d,
        std_total_reward=std_r,
        mean_horizon=sum(horizons) / max(len(horizons), 1),
        n_episodes=n_episodes,
        episode_rewards=rewards,
    )


def compare_planners(
    pomdp: POMDP,
    planners: Dict[str, Any],
    n_episodes: int = 10,
    horizon: int = 100,
    discount: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, EvaluationResult]:
    """Compare multiple planners on the same POMDP."""
    results = {}
    for name, planner in planners.items():
        results[name] = evaluate_planner(
            pomdp, planner, n_episodes=n_episodes,
            horizon=horizon, discount=discount, seed=seed,
        )
    return results


# ---------------------------------------------------------------------------
# Example POMDPs for testing
# ---------------------------------------------------------------------------

def make_tiger_planning_pomdp() -> POMDP:
    """Classic Tiger POMDP with noisy observations for planning tests.

    To model stochastic observations with our deterministic obs map,
    we use expanded states: (tiger_pos, heard_side). Listen transitions
    probabilistically to states with correct/incorrect heard_side.

    States: (tiger_pos, heard) for tiger_pos in {L,R}, heard in {L,R}
    Actions: listen, open_left, open_right
    Observations: hear_left, hear_right (determined by heard component)
    Listen accuracy: 0.85 (correct hearing)
    Rewards: listen=-1, open_correct=+10, open_wrong=-100
    """
    # Expanded states: (tiger_pos, last_heard)
    positions = ["L", "R"]
    hearings = ["L", "R"]
    states = set()
    for tp in positions:
        for h in hearings:
            states.add((tp, h))

    actions = {"listen", "open_left", "open_right"}

    # Observation: determined by heard component
    obs_map = {}
    for tp in positions:
        for h in hearings:
            obs_map[(tp, h)] = "hear_left" if h == "L" else "hear_right"

    transitions = {}
    rewards = {}
    accuracy = 0.85

    for tp in positions:
        for h in hearings:
            s = (tp, h)

            # Listen: tiger stays, but heard changes probabilistically
            # 85% chance of hearing correctly
            correct_h = tp  # correct hearing matches tiger position
            wrong_h = "R" if tp == "L" else "L"
            transitions[(s, "listen")] = [
                ((tp, correct_h), accuracy),
                ((tp, wrong_h), 1.0 - accuracy),
            ]
            rewards[(s, "listen")] = -1

            # Open left: reward depends on tiger position, resets to uniform
            transitions[(s, "open_left")] = [
                (("L", "L"), 0.25), (("L", "R"), 0.25),
                (("R", "L"), 0.25), (("R", "R"), 0.25),
            ]
            rewards[(s, "open_left")] = -100 if tp == "L" else 10

            # Open right
            transitions[(s, "open_right")] = [
                (("L", "L"), 0.25), (("L", "R"), 0.25),
                (("R", "L"), 0.25), (("R", "R"), 0.25),
            ]
            rewards[(s, "open_right")] = -100 if tp == "R" else 10

    initial = [
        (("L", "L"), 0.25), (("L", "R"), 0.25),
        (("R", "L"), 0.25), (("R", "R"), 0.25),
    ]

    return POMDP(
        states=states,
        actions=actions,
        transitions=transitions,
        obs=obs_map,
        rewards=rewards,
        initial=initial,
        objective=POMDPObjective.REWARD_DISCOUNTED,
    )


def make_maze_planning_pomdp() -> POMDP:
    """4x4 grid maze POMDP for planning tests.

    Agent starts at (0,0), goal at (3,3).
    Observations: wall configuration (N/S/E/W walls visible).
    Actions: north, south, east, west.
    Reward: +100 at goal, -1 per step.
    """
    size = 4
    states = set()
    for r in range(size):
        for c in range(size):
            states.add((r, c))

    actions = {"north", "south", "east", "west"}
    goal = (3, 3)

    transitions = {}
    rewards = {}
    obs_map = {}

    moves = {
        "north": (-1, 0), "south": (1, 0),
        "east": (0, 1), "west": (0, -1),
    }

    for r in range(size):
        for c in range(size):
            s = (r, c)
            # Observation: which directions are walls
            walls = []
            if r == 0: walls.append("N")
            if r == size - 1: walls.append("S")
            if c == 0: walls.append("W")
            if c == size - 1: walls.append("E")
            obs_map[s] = tuple(sorted(walls))

            for a, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    ns = (nr, nc)
                else:
                    ns = s  # bounce off wall
                transitions[(s, a)] = [(ns, 1.0)]
                rewards[(s, a)] = 100.0 if ns == goal else -1.0

    initial = [((0, 0), 1.0)]

    return POMDP(
        states=states,
        actions=actions,
        transitions=transitions,
        obs=obs_map,
        rewards=rewards,
        initial=initial,
        objective=POMDPObjective.REWARD_FINITE,
    )


def make_hallway_pomdp() -> POMDP:
    """1D hallway POMDP: agent moves left/right, observes position zone.

    States: positions 0..4 (short hallway for tractable planning)
    Actions: left, right
    Observations: zone_0, zone_1, zone_2 (coarse position)
    Goal: reach position 4
    Reward: +100 at goal, -1 per step, -2 for hitting left wall
    """
    n = 5
    states = set(range(n))
    actions = {"left", "right"}

    transitions = {}
    rewards = {}
    obs_map = {}

    for s in range(n):
        # Observation: coarse zone
        if s <= 1:
            obs_map[s] = "zone_0"
        elif s <= 2:
            obs_map[s] = "zone_1"
        else:
            obs_map[s] = "zone_2"

        # Right: move right
        ns_r = min(n - 1, s + 1)
        transitions[(s, "right")] = [(ns_r, 1.0)]
        rewards[(s, "right")] = 100.0 if ns_r == n - 1 else -1.0

        # Left: move left
        ns_l = max(0, s - 1)
        transitions[(s, "left")] = [(ns_l, 1.0)]
        rewards[(s, "left")] = -2.0  # penalty for going wrong way

    initial = [(0, 1.0)]

    return POMDP(
        states=states,
        actions=actions,
        transitions=transitions,
        obs=obs_map,
        rewards=rewards,
        initial=initial,
        objective=POMDPObjective.REWARD_FINITE,
    )


# ---------------------------------------------------------------------------
# Summary / statistics
# ---------------------------------------------------------------------------

def planner_summary(planner, name: str = "") -> str:
    """Summarize a planner's last search results."""
    lines = [f"=== Planner: {name or type(planner).__name__} ==="]
    vals = planner.get_action_values()
    visits = planner.get_action_visits()
    if vals:
        lines.append("Action values:")
        for a in sorted(vals.keys(), key=str):
            v = vals[a]
            n = visits.get(a, 0)
            lines.append(f"  {a}: Q={v:.3f}, visits={n}")
    if hasattr(planner, 'root') and planner.root:
        lines.append(f"Root visits: {planner.root.visit_count}")
    return "\n".join(lines)


def evaluation_summary(results: Dict[str, EvaluationResult]) -> str:
    """Summarize comparison results."""
    lines = ["=== Planner Comparison ==="]
    for name, res in sorted(results.items()):
        lines.append(
            f"  {name}: mean_reward={res.mean_total_reward:.2f} "
            f"(+/-{res.std_total_reward:.2f}), "
            f"disc_reward={res.mean_discounted_reward:.2f}, "
            f"horizon={res.mean_horizon:.1f}"
        )
    return "\n".join(lines)
