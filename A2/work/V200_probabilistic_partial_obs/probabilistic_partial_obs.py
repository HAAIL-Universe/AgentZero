"""V200: Probabilistic Partial Observation -- POMDPs and belief-based strategies.

Composes V198 (partial observation games) + V160 (energy games).

A POMDP (Partially Observable Markov Decision Process) extends partial
observation games with probabilistic transitions. The agent cannot see the
full state, receives observations, and must act to maximize reward or
satisfy temporal objectives under uncertainty.

Key concepts:
- Belief state: probability distribution over states (not just a set)
- Belief update: Bayesian conditioning on observation after action
- Value iteration over belief space
- PBVI (Point-Based Value Iteration): sample beliefs, compute alpha-vectors
- Qualitative POMDP: almost-sure/positive-probability winning
- Stochastic partial observation games: 2-player probabilistic + PO

Algorithms:
1. Exact belief update (Bayes rule)
2. Value iteration for finite-horizon POMDPs
3. PBVI for infinite-horizon (approximate)
4. Qualitative almost-sure reachability under PO
5. Stochastic PO game solving (Player 1 vs Nature + Player 2)
"""

import sys, os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum
from fractions import Fraction
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V198_partial_observation_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))

from partial_observation_games import (
    PartialObsGame, KnowledgeState, KnowledgeGame, ObsStrategy,
    POGameResult, Objective,
    _initial_belief, _observation_split, build_knowledge_game,
    solve_safety, solve_reachability, solve as solve_po,
    analyze_observability, game_statistics as po_game_statistics,
)
from parity_games import Player


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class POMDPObjective(Enum):
    """POMDP objective types."""
    REWARD_FINITE = "reward_finite"       # maximize total reward (finite horizon)
    REWARD_DISCOUNTED = "reward_disc"     # maximize discounted reward (infinite)
    REACHABILITY = "reach"                # reach target with max probability
    SAFETY = "safety"                     # stay safe with max probability
    BUCHI = "buchi"                       # visit target infinitely often (almost-sure)
    ALMOST_SURE_REACH = "as_reach"        # reach target with probability 1


@dataclass
class POMDP:
    """A Partially Observable Markov Decision Process.

    States: finite set of state IDs.
    Actions: finite set per state.
    Transitions: state x action -> distribution over next states.
    Observations: state -> observation (deterministic observation function).
    Rewards: state x action -> real reward.
    Initial: distribution over initial states.

    Attributes:
        states: set of state IDs
        actions: set of action IDs (global action set)
        state_actions: state -> set of available actions (default: all actions)
        transitions: (state, action) -> list of (next_state, probability)
        obs: state -> observation ID
        rewards: (state, action) -> reward value
        initial: list of (state, probability) for initial distribution
        objective: POMDP objective type
        target: set of target states (for reachability/safety/Buchi)
        discount: discount factor for infinite-horizon (0 < gamma <= 1)
        horizon: finite horizon length (for finite-horizon problems)
    """
    states: Set[int] = field(default_factory=set)
    actions: Set[int] = field(default_factory=set)
    state_actions: Dict[int, Set[int]] = field(default_factory=dict)
    transitions: Dict[Tuple[int, int], List[Tuple[int, Fraction]]] = field(default_factory=dict)
    obs: Dict[int, int] = field(default_factory=dict)
    rewards: Dict[Tuple[int, int], Fraction] = field(default_factory=dict)
    initial: List[Tuple[int, Fraction]] = field(default_factory=list)
    objective: POMDPObjective = POMDPObjective.REWARD_FINITE
    target: Set[int] = field(default_factory=set)
    discount: Fraction = Fraction(1)
    horizon: int = 1

    def add_state(self, s: int, observation: int):
        """Add a state with its observation."""
        self.states.add(s)
        self.obs[s] = observation

    def add_action(self, a: int):
        """Add an action to the global action set."""
        self.actions.add(a)

    def set_state_actions(self, s: int, acts: Set[int]):
        """Set available actions for a state."""
        self.state_actions[s] = acts

    def get_actions(self, s: int) -> Set[int]:
        """Get available actions for a state."""
        return self.state_actions.get(s, self.actions)

    def add_transition(self, s: int, a: int, s_next: int, prob: Fraction):
        """Add a transition: from s under action a, go to s_next with prob."""
        key = (s, a)
        if key not in self.transitions:
            self.transitions[key] = []
        self.transitions[key].append((s_next, Fraction(prob)))

    def set_reward(self, s: int, a: int, r):
        """Set reward for (state, action) pair."""
        self.rewards[(s, a)] = Fraction(r)

    def set_initial(self, dist: List[Tuple[int, Fraction]]):
        """Set initial state distribution."""
        self.initial = [(s, Fraction(p)) for s, p in dist]

    def get_transitions(self, s: int, a: int) -> List[Tuple[int, Fraction]]:
        """Get transition distribution for (state, action)."""
        return self.transitions.get((s, a), [])

    def get_reward(self, s: int, a: int) -> Fraction:
        """Get reward for (state, action)."""
        return self.rewards.get((s, a), Fraction(0))

    def all_observations(self) -> Set[int]:
        """Get all unique observation IDs."""
        return set(self.obs.values())

    def obs_class(self, o: int) -> Set[int]:
        """Get all states with observation o."""
        return {s for s in self.states if self.obs.get(s) == o}

    def is_valid(self) -> bool:
        """Check if POMDP is well-formed (transitions sum to 1)."""
        for key, dist in self.transitions.items():
            total = sum(p for _, p in dist)
            if total != Fraction(1):
                return False
        # Check initial distribution
        if self.initial:
            total = sum(p for _, p in self.initial)
            if total != Fraction(1):
                return False
        return True


# ---------------------------------------------------------------------------
# Belief state representation
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    """A belief state: probability distribution over POMDP states.

    Represented as a dict mapping state -> probability.
    Only states with non-zero probability are stored.
    """
    dist: Dict[int, Fraction] = field(default_factory=dict)

    def __hash__(self):
        return hash(tuple(sorted(self.dist.items())))

    def __eq__(self, other):
        return isinstance(other, Belief) and self.dist == other.dist

    def support(self) -> Set[int]:
        """States with non-zero probability."""
        return {s for s, p in self.dist.items() if p > 0}

    def prob(self, s: int) -> Fraction:
        """Probability of state s."""
        return self.dist.get(s, Fraction(0))

    def observation(self, pomdp: 'POMDP') -> Optional[int]:
        """Get the observation for this belief (all support states must share obs)."""
        obs_set = {pomdp.obs[s] for s in self.support()}
        if len(obs_set) == 1:
            return obs_set.pop()
        return None  # mixed observation (shouldn't happen in valid belief update)

    def is_valid(self) -> bool:
        """Check if probabilities sum to 1."""
        total = sum(self.dist.values())
        return total == Fraction(1) and all(p >= 0 for p in self.dist.values())

    def entropy(self) -> float:
        """Shannon entropy of the belief (in bits)."""
        import math
        h = 0.0
        for p in self.dist.values():
            pf = float(p)
            if pf > 0:
                h -= pf * math.log2(pf)
        return h

    @staticmethod
    def uniform(states: Set[int]) -> 'Belief':
        """Create a uniform belief over the given states."""
        n = len(states)
        if n == 0:
            return Belief()
        p = Fraction(1, n)
        return Belief({s: p for s in states})

    @staticmethod
    def point(s: int) -> 'Belief':
        """Create a point belief (certainty about state s)."""
        return Belief({s: Fraction(1)})

    @staticmethod
    def from_initial(pomdp: 'POMDP') -> 'Belief':
        """Create belief from POMDP initial distribution."""
        return Belief({s: p for s, p in pomdp.initial if p > 0})


# ---------------------------------------------------------------------------
# Belief update (Bayes rule)
# ---------------------------------------------------------------------------

def belief_update(pomdp: POMDP, b: Belief, action: int, observation: int) -> Optional[Belief]:
    """Bayesian belief update: b' = update(b, action, observation).

    Given current belief b, action taken, and observation received,
    compute the posterior belief b'.

    Returns None if the observation has zero probability (impossible).
    """
    # Step 1: predict -- apply action to get predicted belief
    predicted = defaultdict(Fraction)
    for s, p_s in b.dist.items():
        if p_s <= 0:
            continue
        for s_next, p_t in pomdp.get_transitions(s, action):
            predicted[s_next] += p_s * p_t

    # Step 2: condition on observation
    # Filter to states matching the observation
    conditioned = {}
    for s_next, p in predicted.items():
        if pomdp.obs.get(s_next) == observation and p > 0:
            conditioned[s_next] = p

    # Step 3: normalize
    total = sum(conditioned.values())
    if total == 0:
        return None  # impossible observation

    return Belief({s: p / total for s, p in conditioned.items()})


def observation_probability(pomdp: POMDP, b: Belief, action: int, observation: int) -> Fraction:
    """Probability of receiving an observation after taking action from belief b."""
    total = Fraction(0)
    for s, p_s in b.dist.items():
        if p_s <= 0:
            continue
        for s_next, p_t in pomdp.get_transitions(s, action):
            if pomdp.obs.get(s_next) == observation:
                total += p_s * p_t
    return total


def possible_observations(pomdp: POMDP, b: Belief, action: int) -> Set[int]:
    """Get all observations that have positive probability after action from belief."""
    obs_set = set()
    for s in b.support():
        for s_next, p_t in pomdp.get_transitions(s, action):
            if p_t > 0:
                obs_set.add(pomdp.obs[s_next])
    return obs_set


def belief_expected_reward(pomdp: POMDP, b: Belief, action: int) -> Fraction:
    """Expected immediate reward for taking action in belief b."""
    total = Fraction(0)
    for s, p_s in b.dist.items():
        if p_s > 0:
            total += p_s * pomdp.get_reward(s, action)
    return total


# ---------------------------------------------------------------------------
# Alpha-vector representation for value function
# ---------------------------------------------------------------------------

@dataclass
class AlphaVector:
    """An alpha-vector: a hyperplane in belief space.

    The value at belief b is: sum_s alpha[s] * b(s).
    Each alpha-vector is associated with an action.
    """
    values: Dict[int, Fraction]   # state -> value
    action: int                    # action this alpha-vector is for

    def evaluate(self, b: Belief) -> Fraction:
        """Evaluate this alpha-vector at belief b."""
        total = Fraction(0)
        for s, p in b.dist.items():
            total += p * self.values.get(s, Fraction(0))
        return total


def value_at_belief(alphas: List[AlphaVector], b: Belief) -> Tuple[Fraction, int]:
    """Compute value and best action at a belief using alpha-vectors.

    Returns (value, best_action).
    """
    best_val = None
    best_act = -1
    for alpha in alphas:
        val = alpha.evaluate(b)
        if best_val is None or val > best_val:
            best_val = val
            best_act = alpha.action
    if best_val is None:
        return (Fraction(0), -1)
    return (best_val, best_act)


# ---------------------------------------------------------------------------
# Finite-horizon value iteration
# ---------------------------------------------------------------------------

def finite_horizon_vi(pomdp: POMDP, max_alphas: int = 200) -> List[List[AlphaVector]]:
    """Point-based finite-horizon value iteration for POMDP.

    Uses corner beliefs (point beliefs at each state) plus initial belief
    as the belief set for point-based backup. This avoids the exponential
    blowup of exact enumeration while being exact for small state spaces.

    Returns a list of alpha-vector sets, one per time step.
    alphas[t] gives the value function at time-to-go t.
    """
    states = sorted(pomdp.states)
    actions = sorted(pomdp.actions)
    observations = sorted(pomdp.all_observations())
    horizon = pomdp.horizon

    # Belief points: corner beliefs + initial belief
    belief_points = [Belief.point(s) for s in states]
    if pomdp.initial:
        belief_points.append(Belief.from_initial(pomdp))

    # Terminal alpha-vectors: zero value
    alphas = [AlphaVector({s: Fraction(0) for s in states}, a) for a in actions]
    all_alphas = [alphas]

    for t in range(horizon):
        new_alphas = []

        for b in belief_points:
            best_alpha = None
            best_val = None

            for a in actions:
                # Compute backup alpha-vector for action a at belief b
                # alpha_a(s) = r(s,a) + gamma * sum_o sum_s' T(s,a,s') * [obs(s')==o] * alpha_o^*(s')
                # where alpha_o^* is the best alpha from previous set at updated belief b_a_o

                # For each observation, find the best previous alpha at the updated belief
                best_prev_per_obs = {}
                for o in observations:
                    b_next = belief_update(pomdp, b, a, o)
                    if b_next is None:
                        continue
                    # Find best alpha from previous set at b_next
                    best_prev_val = None
                    best_prev = None
                    for prev_alpha in alphas:
                        v = prev_alpha.evaluate(b_next)
                        if best_prev_val is None or v > best_prev_val:
                            best_prev_val = v
                            best_prev = prev_alpha
                    if best_prev is not None:
                        best_prev_per_obs[o] = best_prev

                # Build the alpha-vector
                alpha_vals = {}
                for s in states:
                    val = pomdp.get_reward(s, a)
                    for o, prev_alpha in best_prev_per_obs.items():
                        for s_next, p_t in pomdp.get_transitions(s, a):
                            if pomdp.obs.get(s_next) == o:
                                val += pomdp.discount * p_t * prev_alpha.values.get(s_next, Fraction(0))
                    alpha_vals[s] = val

                candidate = AlphaVector(alpha_vals, a)
                v = candidate.evaluate(b)
                if best_val is None or v > best_val:
                    best_val = v
                    best_alpha = candidate

            if best_alpha is not None:
                new_alphas.append(best_alpha)

        # Prune dominated alpha-vectors
        alphas = _prune_alphas(new_alphas, states)
        if len(alphas) > max_alphas:
            alphas = alphas[:max_alphas]
        all_alphas.append(alphas)

    return list(reversed(all_alphas))


def _prune_alphas(alphas: List[AlphaVector], states: List[int]) -> List[AlphaVector]:
    """Remove dominated alpha-vectors (simple pairwise dominance check)."""
    if len(alphas) <= 1:
        return alphas
    pruned = []
    for i, a in enumerate(alphas):
        dominated = False
        for j, b in enumerate(alphas):
            if i == j:
                continue
            # Check if b dominates a (b[s] >= a[s] for all s, strict for some)
            all_geq = True
            some_strict = False
            for s in states:
                va = a.values.get(s, Fraction(0))
                vb = b.values.get(s, Fraction(0))
                if vb < va:
                    all_geq = False
                    break
                if vb > va:
                    some_strict = True
            if all_geq and some_strict:
                dominated = True
                break
        if not dominated:
            pruned.append(a)
    return pruned if pruned else alphas[:1]  # keep at least one


# ---------------------------------------------------------------------------
# Point-Based Value Iteration (PBVI)
# ---------------------------------------------------------------------------

def pbvi(pomdp: POMDP, belief_points: List[Belief],
         iterations: int = 50, tolerance: Fraction = Fraction(1, 1000)) -> List[AlphaVector]:
    """Point-Based Value Iteration for infinite-horizon discounted POMDPs.

    Approximates the value function using alpha-vectors computed at
    a finite set of belief points.

    Args:
        pomdp: the POMDP
        belief_points: set of belief points to use
        iterations: max number of iterations
        tolerance: convergence threshold

    Returns:
        List of alpha-vectors representing the value function.
    """
    states = sorted(pomdp.states)
    actions = sorted(pomdp.actions)
    observations = sorted(pomdp.all_observations())

    # Initialize: one alpha-vector per action with immediate rewards
    alphas = []
    for a in actions:
        # Pessimistic init: r(s,a) / (1 - gamma) if gamma < 1
        vals = {}
        for s in states:
            r = pomdp.get_reward(s, a)
            if pomdp.discount < 1:
                vals[s] = r / (1 - pomdp.discount)
            else:
                vals[s] = r
        alphas.append(AlphaVector(vals, a))

    for iteration in range(iterations):
        new_alphas = []
        max_change = Fraction(0)

        for b in belief_points:
            best_alpha = None
            best_val = None

            for a in actions:
                # Compute backup alpha-vector for action a at belief b
                alpha_vals = {}
                for s in states:
                    val = pomdp.get_reward(s, a)
                    for o in observations:
                        # Find best prev alpha for this observation
                        b_next = belief_update(pomdp, b, a, o)
                        if b_next is None:
                            continue
                        _, _ = value_at_belief(alphas, b_next)
                        # Compute contribution of this observation
                        for s_next, p_t in pomdp.get_transitions(s, a):
                            if pomdp.obs.get(s_next) == o:
                                # Find best alpha at b_next
                                best_next_val = None
                                best_next_alpha = None
                                for prev_a in alphas:
                                    nv = prev_a.evaluate(b_next)
                                    if best_next_val is None or nv > best_next_val:
                                        best_next_val = nv
                                        best_next_alpha = prev_a
                                if best_next_alpha is not None:
                                    val += pomdp.discount * p_t * best_next_alpha.values.get(s_next, Fraction(0))
                    alpha_vals[s] = val

                candidate = AlphaVector(alpha_vals, a)
                v = candidate.evaluate(b)
                if best_val is None or v > best_val:
                    best_val = v
                    best_alpha = candidate

            if best_alpha is not None:
                # Check change
                old_val, _ = value_at_belief(alphas, b)
                change = abs(best_val - old_val)
                if change > max_change:
                    max_change = change
                new_alphas.append(best_alpha)

        if new_alphas:
            alphas = _prune_alphas(new_alphas, states)

        if max_change <= tolerance:
            break

    return alphas


# ---------------------------------------------------------------------------
# Qualitative reachability under partial observation
# ---------------------------------------------------------------------------

def almost_sure_reachability(pomdp: POMDP) -> Tuple[bool, Optional[Dict[int, int]]]:
    """Check if target is reachable with probability 1 from initial belief.

    Uses belief-space fixpoint: iteratively remove beliefs from which
    no action guarantees progress toward the target.

    Returns:
        (is_almost_sure, strategy) where strategy maps observation -> action.
    """
    if not pomdp.target:
        return (False, None)

    b0 = Belief.from_initial(pomdp)
    if not b0.support():
        return (False, None)

    # Check if initial belief is entirely in target
    if b0.support() <= pomdp.target:
        return (True, {})

    # BFS over belief space
    visited = {}  # belief -> (can_reach, action)
    queue = [b0]
    belief_key = lambda b: frozenset(b.dist.items())
    visited_keys = {belief_key(b0)}
    all_beliefs = [b0]

    while queue:
        b = queue.pop(0)
        for a in sorted(pomdp.actions):
            for o in possible_observations(pomdp, b, a):
                b_next = belief_update(pomdp, b, a, o)
                if b_next is None:
                    continue
                key = belief_key(b_next)
                if key not in visited_keys:
                    visited_keys.add(key)
                    all_beliefs.append(b_next)
                    queue.append(b_next)

    # Now do backward fixpoint from target beliefs
    # A belief is "winning" if support subset of target, or
    # there exists an action where all successor beliefs are winning

    target = pomdp.target
    max_iters = len(all_beliefs) + 1

    winning = set()  # indices into all_beliefs
    for i, b in enumerate(all_beliefs):
        if b.support() <= target:
            winning.add(i)

    changed = True
    strategy = {}  # observation -> action
    for _ in range(max_iters):
        if not changed:
            break
        changed = False
        for i, b in enumerate(all_beliefs):
            if i in winning:
                continue
            # Try each action
            for a in sorted(pomdp.actions):
                all_succ_win = True
                obs_set = possible_observations(pomdp, b, a)
                if not obs_set:
                    all_succ_win = False
                    continue
                for o in obs_set:
                    b_next = belief_update(pomdp, b, a, o)
                    if b_next is None:
                        continue
                    # Find this belief in all_beliefs
                    key = belief_key(b_next)
                    found = False
                    for j, bj in enumerate(all_beliefs):
                        if belief_key(bj) == key:
                            if j in winning:
                                found = True
                            break
                    if not found:
                        all_succ_win = False
                        break
                if all_succ_win:
                    winning.add(i)
                    changed = True
                    # Record strategy
                    bo = b.observation(pomdp)
                    if bo is not None:
                        strategy[bo] = a
                    break

    # Check if initial belief is winning
    is_winning = 0 in winning
    return (is_winning, strategy if is_winning else None)


def positive_reachability(pomdp: POMDP) -> Tuple[bool, Optional[Dict[int, int]]]:
    """Check if target is reachable with positive probability from initial belief.

    Easier than almost-sure: just need one path with non-zero probability.

    Returns:
        (is_positive, strategy) where strategy maps observation -> action.
    """
    if not pomdp.target:
        return (False, None)

    b0 = Belief.from_initial(pomdp)
    if not b0.support():
        return (False, None)

    if b0.support() & pomdp.target:
        return (True, {})

    # BFS: find any belief whose support intersects target
    belief_key = lambda b: frozenset(b.dist.items())
    visited = {belief_key(b0)}
    queue = [(b0, {})]  # (belief, strategy_so_far)

    while queue:
        b, strat = queue.pop(0)
        for a in sorted(pomdp.actions):
            for o in possible_observations(pomdp, b, a):
                b_next = belief_update(pomdp, b, a, o)
                if b_next is None:
                    continue
                key = belief_key(b_next)

                new_strat = dict(strat)
                bo = b.observation(pomdp)
                if bo is not None:
                    new_strat[bo] = a

                if b_next.support() & pomdp.target:
                    return (True, new_strat)

                if key not in visited:
                    visited.add(key)
                    queue.append((b_next, new_strat))

    return (False, None)


# ---------------------------------------------------------------------------
# Safety under partial observation
# ---------------------------------------------------------------------------

def safety_probability(pomdp: POMDP, steps: int) -> Tuple[Fraction, Optional[Dict[int, int]]]:
    """Compute max probability of staying safe for `steps` steps.

    Safe means avoiding states NOT in pomdp.target (target = safe states).

    Returns:
        (max_probability, strategy) where strategy maps observation -> action.
    """
    safe = pomdp.target
    b0 = Belief.from_initial(pomdp)
    if not b0.support():
        return (Fraction(0), None)

    # Check initial safety
    if not (b0.support() <= safe):
        return (Fraction(0), None)

    # Dynamic programming over belief space
    # V(b, t) = max_a sum_o P(o|b,a) * V(update(b,a,o), t-1)
    # with V(b, 0) = 1 if support(b) subset safe, else 0

    belief_key = lambda b: frozenset(b.dist.items())

    # Enumerate reachable beliefs
    all_beliefs = [b0]
    key_to_idx = {belief_key(b0): 0}
    queue = [b0]

    while queue:
        b = queue.pop(0)
        for a in sorted(pomdp.actions):
            for o in possible_observations(pomdp, b, a):
                b_next = belief_update(pomdp, b, a, o)
                if b_next is None:
                    continue
                key = belief_key(b_next)
                if key not in key_to_idx:
                    key_to_idx[key] = len(all_beliefs)
                    all_beliefs.append(b_next)
                    queue.append(b_next)

    n = len(all_beliefs)
    # V[i] = value at belief i
    V = [Fraction(0)] * n
    best_action = [None] * n  # per step strategy

    # Base case: V(b, 0) = 1 if safe
    for i, b in enumerate(all_beliefs):
        if b.support() <= safe:
            V[i] = Fraction(1)

    strategy = {}  # obs -> action (greedy at each step)

    for t in range(steps):
        V_new = [Fraction(0)] * n
        for i, b in enumerate(all_beliefs):
            if not (b.support() <= safe):
                V_new[i] = Fraction(0)
                continue
            best_val = Fraction(0)
            best_a = None
            for a in sorted(pomdp.actions):
                val = Fraction(0)
                obs_set = possible_observations(pomdp, b, a)
                for o in obs_set:
                    p_o = observation_probability(pomdp, b, a, o)
                    b_next = belief_update(pomdp, b, a, o)
                    if b_next is not None:
                        j = key_to_idx.get(belief_key(b_next))
                        if j is not None:
                            val += p_o * V[j]
                if val >= best_val:
                    best_val = val
                    best_a = a
            V_new[i] = best_val
            best_action[i] = best_a
        V = V_new

    # Extract strategy from initial belief
    # Record obs -> action from best_action at the final step
    for i, b in enumerate(all_beliefs):
        bo = b.observation(pomdp)
        if bo is not None and best_action[i] is not None:
            strategy[bo] = best_action[i]

    return (V[0], strategy)


# ---------------------------------------------------------------------------
# Stochastic partial observation game (2-player + probabilistic)
# ---------------------------------------------------------------------------

@dataclass
class StochasticPOGame:
    """A 2-player stochastic game with partial observation.

    Three types of vertices:
    - Player 1 (Even): protagonist, partial observation
    - Player 2 (Odd): antagonist, full observation
    - Nature: probabilistic, chooses successor according to distribution

    Player 1 tries to maximize probability of reaching target.
    Player 2 tries to minimize it.
    Nature is neutral (probabilistic).
    """
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, Set[int]] = field(default_factory=dict)  # for P1/P2 vertices
    prob_edges: Dict[int, List[Tuple[int, Fraction]]] = field(default_factory=dict)  # for Nature vertices
    owner: Dict[int, str] = field(default_factory=dict)  # "P1", "P2", "Nature"
    obs: Dict[int, int] = field(default_factory=dict)  # P1's observation
    initial: Set[int] = field(default_factory=set)
    target: Set[int] = field(default_factory=set)

    def add_vertex(self, v: int, owner: str, observation: int = 0):
        """Add a vertex. Owner is 'P1', 'P2', or 'Nature'."""
        self.vertices.add(v)
        self.owner[v] = owner
        self.obs[v] = observation

    def add_edge(self, u: int, v: int):
        """Add edge for P1/P2 vertex."""
        if u not in self.edges:
            self.edges[u] = set()
        self.edges[u].add(v)

    def add_prob_edge(self, u: int, v: int, prob: Fraction):
        """Add probabilistic edge for Nature vertex."""
        if u not in self.prob_edges:
            self.prob_edges[u] = []
        self.prob_edges[u].append((v, Fraction(prob)))

    def successors(self, v: int) -> Set[int]:
        """Get successors (for P1/P2 vertices)."""
        return self.edges.get(v, set())

    def prob_successors(self, v: int) -> List[Tuple[int, Fraction]]:
        """Get probabilistic successors (for Nature vertices)."""
        return self.prob_edges.get(v, [])


def solve_stochastic_po_game(game: StochasticPOGame,
                              iterations: int = 100) -> Tuple[Dict[int, Fraction], Optional[Dict[int, int]]]:
    """Solve a stochastic PO game for reachability probability.

    Computes the value (max reachability probability under P1's optimal
    observation-based strategy, against P2's optimal counter-strategy).

    Uses belief-state value iteration.

    Returns:
        (values, strategy) where values maps initial vertices to their value,
        and strategy maps observation -> target observation.
    """
    # Convert to belief space
    # Build initial beliefs
    target = game.target

    # Enumerate beliefs via BFS from initial
    initial_states = game.initial if game.initial else game.vertices
    obs_groups = defaultdict(set)
    for v in initial_states:
        obs_groups[game.obs[v]].add(v)

    # Initial beliefs: one per observation of initial states
    initial_beliefs = []
    for o, states in obs_groups.items():
        b = Belief.uniform(states)
        initial_beliefs.append(b)

    belief_key = lambda b: frozenset(b.dist.items())
    all_beliefs = list(initial_beliefs)
    key_to_idx = {belief_key(b): i for i, b in enumerate(initial_beliefs)}

    # Expand one step from a belief state
    def expand_belief(b):
        """Expand a belief by one step, return list of (action/nature, obs) -> next belief."""
        successors = []
        support = b.support()
        if not support:
            return successors

        # Determine owner of this belief (all states should have same owner in well-formed game)
        owners = {game.owner[s] for s in support}

        if "Nature" in owners:
            # Nature vertex: apply probabilistic transition
            predicted = defaultdict(Fraction)
            for s, p_s in b.dist.items():
                for s_next, p_t in game.prob_successors(s):
                    predicted[s_next] += p_s * p_t
            # Group by observation
            obs_groups_next = defaultdict(dict)
            for s_next, p in predicted.items():
                o = game.obs[s_next]
                obs_groups_next[o][s_next] = obs_groups_next[o].get(s_next, Fraction(0)) + p
            for o, dist in obs_groups_next.items():
                total = sum(dist.values())
                if total > 0:
                    b_next = Belief({s: p / total for s, p in dist.items()})
                    successors.append(b_next)
        else:
            # P1 or P2: deterministic choice
            # Group available moves by target observation
            for s in support:
                for s_next in game.successors(s):
                    pass  # enumerate all possible target observations

            # Get all possible target observations from any state in belief
            target_obs = set()
            for s in support:
                for s_next in game.successors(s):
                    target_obs.add(game.obs[s_next])

            for to in target_obs:
                # If choosing target observation to: update belief
                new_dist = {}
                for s, p_s in b.dist.items():
                    reachable = {s_next for s_next in game.successors(s)
                                 if game.obs[s_next] == to}
                    if reachable:
                        # Uniform distribution among reachable states with obs to
                        per = p_s / len(reachable)
                        for s_next in reachable:
                            new_dist[s_next] = new_dist.get(s_next, Fraction(0)) + per
                if new_dist:
                    total = sum(new_dist.values())
                    if total > 0:
                        b_next = Belief({s: p / total for s, p in new_dist.items()})
                        successors.append(b_next)

        return successors

    # BFS to enumerate beliefs
    queue = list(range(len(all_beliefs)))
    while queue:
        i = queue.pop(0)
        b = all_beliefs[i]
        for b_next in expand_belief(b):
            key = belief_key(b_next)
            if key not in key_to_idx:
                key_to_idx[key] = len(all_beliefs)
                all_beliefs.append(b_next)
                queue.append(len(all_beliefs) - 1)

    n = len(all_beliefs)

    # Value iteration
    V = [Fraction(0)] * n
    for i, b in enumerate(all_beliefs):
        if b.support() <= target:
            V[i] = Fraction(1)

    for _ in range(iterations):
        V_new = list(V)
        for i, b in enumerate(all_beliefs):
            if b.support() <= target:
                V_new[i] = Fraction(1)
                continue
            if not b.support():
                continue

            owners = {game.owner[s] for s in b.support()}

            succ_beliefs = expand_belief(b)
            if not succ_beliefs:
                V_new[i] = Fraction(1) if b.support() <= target else Fraction(0)
                continue

            succ_vals = []
            for b_next in succ_beliefs:
                key = belief_key(b_next)
                j = key_to_idx.get(key)
                if j is not None:
                    succ_vals.append(V[j])

            if not succ_vals:
                continue

            if "P1" in owners:
                V_new[i] = max(succ_vals)
            elif "P2" in owners:
                V_new[i] = min(succ_vals)
            else:
                # Nature: weighted average
                total = Fraction(0)
                count = len(succ_vals)
                for sv in succ_vals:
                    total += sv
                V_new[i] = total / count if count > 0 else Fraction(0)

        V = V_new

    # Extract values and strategy for initial beliefs
    values = {}
    strategy = {}
    for i, b in enumerate(initial_beliefs):
        for s in b.support():
            values[s] = V[i]
        # Strategy: for P1 beliefs, pick successor with max value
        owners = {game.owner[s] for s in b.support()}
        if "P1" in owners:
            succ_beliefs = expand_belief(b)
            best_val = Fraction(-1)
            best_obs = None
            for b_next in succ_beliefs:
                key = belief_key(b_next)
                j = key_to_idx.get(key)
                if j is not None and V[j] > best_val:
                    best_val = V[j]
                    bo_next = b_next.observation(game)
                    if bo_next is not None:
                        best_obs = bo_next
            bo = b.observation(game)
            if bo is not None and best_obs is not None:
                strategy[bo] = best_obs

    return (values, strategy)


# ---------------------------------------------------------------------------
# POMDP simulation
# ---------------------------------------------------------------------------

def simulate_pomdp(pomdp: POMDP, strategy: Dict[int, int],
                   steps: int, initial_state: Optional[int] = None) -> List[dict]:
    """Simulate a POMDP run using an observation-based strategy.

    Args:
        pomdp: the POMDP
        strategy: observation -> action mapping
        steps: number of steps to simulate
        initial_state: specific initial state (or sample from initial dist)

    Returns:
        List of step records with state, observation, action, reward, belief.
    """
    import random

    # Pick initial state
    if initial_state is not None:
        state = initial_state
    else:
        # Sample from initial distribution
        r = Fraction(random.random())
        cumulative = Fraction(0)
        state = pomdp.initial[0][0] if pomdp.initial else min(pomdp.states)
        for s, p in pomdp.initial:
            cumulative += p
            if r <= cumulative:
                state = s
                break

    # Initialize belief from initial distribution
    belief = Belief.from_initial(pomdp)
    trace = []

    for t in range(steps):
        obs_id = pomdp.obs[state]
        action = strategy.get(obs_id)
        if action is None:
            # No strategy for this observation -- pick first available
            action = min(pomdp.get_actions(state)) if pomdp.get_actions(state) else 0

        reward = pomdp.get_reward(state, action)

        trace.append({
            'step': t,
            'state': state,
            'observation': obs_id,
            'action': action,
            'reward': float(reward),
            'belief_size': len(belief.support()),
            'belief_entropy': belief.entropy(),
        })

        # Transition
        transitions = pomdp.get_transitions(state, action)
        if not transitions:
            break

        r = Fraction(random.random())
        cumulative = Fraction(0)
        next_state = transitions[0][0]
        for s_next, p in transitions:
            cumulative += p
            if r <= cumulative:
                next_state = s_next
                break

        # Update belief
        next_obs = pomdp.obs[next_state]
        new_belief = belief_update(pomdp, belief, action, next_obs)
        if new_belief is not None:
            belief = new_belief

        state = next_state

    return trace


# ---------------------------------------------------------------------------
# Analysis and comparison tools
# ---------------------------------------------------------------------------

def pomdp_statistics(pomdp: POMDP) -> dict:
    """Compute statistics about a POMDP."""
    n_states = len(pomdp.states)
    n_actions = len(pomdp.actions)
    n_obs = len(pomdp.all_observations())
    n_transitions = sum(len(v) for v in pomdp.transitions.values())
    n_target = len(pomdp.target)

    # Information ratio: observations / states
    info_ratio = n_obs / n_states if n_states > 0 else 0

    return {
        'states': n_states,
        'actions': n_actions,
        'observations': n_obs,
        'transitions': n_transitions,
        'target_states': n_target,
        'info_ratio': info_ratio,
        'objective': pomdp.objective.value,
        'horizon': pomdp.horizon,
        'discount': float(pomdp.discount),
    }


def compare_mdp_vs_pomdp(pomdp: POMDP, steps: int = 10) -> dict:
    """Compare full-observation (MDP) value vs partial-observation (POMDP) value.

    The MDP is derived by making all states fully observable.
    The difference represents the "price of partial information".
    """
    # MDP: each state has unique observation
    mdp = POMDP(
        states=set(pomdp.states),
        actions=set(pomdp.actions),
        state_actions=dict(pomdp.state_actions),
        transitions=dict(pomdp.transitions),
        obs={s: s for s in pomdp.states},  # full observation
        rewards=dict(pomdp.rewards),
        initial=list(pomdp.initial),
        objective=pomdp.objective,
        target=set(pomdp.target),
        discount=pomdp.discount,
        horizon=pomdp.horizon,
    )

    # Solve both
    if pomdp.objective == POMDPObjective.SAFETY:
        po_val, po_strat = safety_probability(pomdp, steps)
        mdp_val, mdp_strat = safety_probability(mdp, steps)
    elif pomdp.objective == POMDPObjective.REWARD_FINITE:
        po_alphas = finite_horizon_vi(pomdp)
        mdp_alphas = finite_horizon_vi(mdp)
        b0_po = Belief.from_initial(pomdp)
        b0_mdp = Belief.from_initial(mdp)
        po_val, _ = value_at_belief(po_alphas[0] if po_alphas else [], b0_po)
        mdp_val, _ = value_at_belief(mdp_alphas[0] if mdp_alphas else [], b0_mdp)
    else:
        # For reachability, use positive reachability
        po_reach, _ = positive_reachability(pomdp)
        mdp_reach, _ = positive_reachability(mdp)
        po_val = Fraction(1) if po_reach else Fraction(0)
        mdp_val = Fraction(1) if mdp_reach else Fraction(0)

    info_cost = mdp_val - po_val

    return {
        'mdp_value': float(mdp_val),
        'pomdp_value': float(po_val),
        'information_cost': float(info_cost),
        'mdp_observations': len(mdp.all_observations()),
        'pomdp_observations': len(pomdp.all_observations()),
    }


def belief_space_size(pomdp: POMDP, max_beliefs: int = 1000) -> dict:
    """Estimate the reachable belief space size.

    BFS from initial belief, enumerating all reachable beliefs.
    Capped at max_beliefs to avoid explosion.
    """
    b0 = Belief.from_initial(pomdp)
    belief_key = lambda b: frozenset(b.dist.items())

    visited = {belief_key(b0)}
    queue = [b0]
    capped = False

    while queue and len(visited) < max_beliefs:
        b = queue.pop(0)
        for a in sorted(pomdp.actions):
            for o in possible_observations(pomdp, b, a):
                b_next = belief_update(pomdp, b, a, o)
                if b_next is None:
                    continue
                key = belief_key(b_next)
                if key not in visited:
                    visited.add(key)
                    queue.append(b_next)
                    if len(visited) >= max_beliefs:
                        capped = True
                        break
            if capped:
                break

    return {
        'reachable_beliefs': len(visited),
        'capped': capped,
        'cap': max_beliefs,
    }
