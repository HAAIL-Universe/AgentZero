"""V203: Symbolic Quantitative Partial Observation.

BDD-encoded belief-energy games. Composes:
  V200 (POMDPs/beliefs) + V160 (energy games) + V021 (BDD model checking)

Key innovation: represent belief sets as BDD nodes over state bits,
avoiding explicit enumeration of the exponentially large belief space.
Energy/cost objectives are solved over the symbolic belief space.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V160_energy_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V200_probabilistic_partial_obs'))

from bdd_model_checker import BDD, BDDNode
from energy_games import (
    EnergyGame, EnergyResult, Player, solve_energy, solve_mean_payoff,
    MeanPayoffResult,
)
from probabilistic_partial_obs import (
    POMDP, Belief, AlphaVector, belief_update,
    observation_probability, possible_observations,
    belief_expected_reward, finite_horizon_vi, pbvi,
)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class SymbolicPOGame:
    """Two-player partially-observable game with energy/cost objectives.

    Player Even (P0) has partial observation (sees obs, not state).
    Player Odd (P1) has full observation (adversary).
    Edges carry integer weights for energy objectives.
    """
    states: Set[int] = field(default_factory=set)
    actions_even: Set[int] = field(default_factory=set)  # P0 actions
    actions_odd: Set[int] = field(default_factory=set)   # P1 actions
    owner: Dict[int, str] = field(default_factory=dict)  # state -> "P0"/"P1"/"Nature"
    transitions: Dict[Tuple[int, int], List[Tuple[int, int, Fraction]]] = field(
        default_factory=dict)  # (state, action) -> [(next, weight, prob)]
    obs: Dict[int, int] = field(default_factory=dict)    # state -> observation
    initial: Set[int] = field(default_factory=set)
    target: Set[int] = field(default_factory=set)         # reachability target
    unsafe: Set[int] = field(default_factory=set)          # safety: avoid these

    def add_state(self, s: int, owner: str, observation: int):
        self.states.add(s)
        self.owner[s] = owner
        self.obs[s] = observation

    def add_transition(self, s: int, a: int, s_next: int,
                       weight: int = 0, prob: Fraction = Fraction(1)):
        key = (s, a)
        if key not in self.transitions:
            self.transitions[key] = []
        self.transitions[key].append((s_next, weight, prob))

    def all_observations(self) -> Set[int]:
        return set(self.obs.values())

    def obs_class(self, o: int) -> Set[int]:
        return {s for s, ob in self.obs.items() if ob == o}

    def get_actions(self, s: int) -> Set[int]:
        return {a for (st, a) in self.transitions if st == s}

    def is_valid(self) -> bool:
        for (s, a), succs in self.transitions.items():
            if s not in self.states:
                return False
            probs = [p for _, _, p in succs]
            if probs and abs(sum(probs) - 1) > Fraction(1, 1000):
                return False
        return True


@dataclass
class SymbolicBelief:
    """A belief state represented both symbolically (BDD) and explicitly.

    The BDD encodes the support set (which states are possible).
    The explicit dict tracks probabilities for quantitative reasoning.
    """
    bdd_support: BDDNode       # BDD over state bits: which states are possible
    distribution: Dict[int, Fraction]  # state -> probability

    @property
    def support(self) -> Set[int]:
        return {s for s, p in self.distribution.items() if p > 0}

    def prob(self, s: int) -> Fraction:
        return self.distribution.get(s, Fraction(0))

    def entropy(self) -> float:
        import math
        h = 0.0
        for p in self.distribution.values():
            if p > 0:
                pf = float(p)
                h -= pf * math.log2(pf)
        return h


@dataclass
class BeliefEnergyResult:
    """Result of solving a belief-energy game."""
    min_energy: Dict[int, Optional[int]]   # obs -> min initial energy for P0 to win
    winning_beliefs: List[SymbolicBelief]  # beliefs where P0 wins
    strategy: Dict[int, int]               # obs -> action for P0
    value_function: Dict[int, float]       # obs -> value (mean-payoff or safety prob)
    iterations: int = 0
    belief_states_explored: int = 0


# ---------------------------------------------------------------------------
# BDD encoding of belief support sets
# ---------------------------------------------------------------------------

class BeliefBDDEncoder:
    """Encodes belief support sets as BDDs over state-indicator variables.

    Each state s gets a BDD variable x_s. A belief support {s1, s3}
    is encoded as x_s1 AND NOT x_s2 AND x_s3 AND NOT x_s4 (for 4 states).

    For sets of beliefs (e.g., "all beliefs where s1 or s3 possible"),
    we use OR of supports.
    """

    def __init__(self, states: Set[int]):
        self.bdd = BDD()
        self.states = sorted(states)
        self.state_to_var: Dict[int, str] = {}
        self.var_to_state: Dict[str, int] = {}

        for s in self.states:
            name = f"s{s}"
            self.state_to_var[s] = name
            self.var_to_state[name] = s
            self.bdd.named_var(name)

    def encode_support(self, support: Set[int]) -> BDDNode:
        """Encode a specific support set as a BDD cube."""
        result = self.bdd.TRUE
        for s in self.states:
            v = self.bdd.named_var(self.state_to_var[s])
            if s in support:
                result = self.bdd.AND(result, v)
            else:
                result = self.bdd.AND(result, self.bdd.NOT(v))
        return result

    def encode_support_superset(self, must_include: Set[int]) -> BDDNode:
        """Encode all supports that include the given states.

        Returns BDD: AND_{s in must_include} x_s
        (other variables are free -- any value OK)
        """
        result = self.bdd.TRUE
        for s in must_include:
            v = self.bdd.named_var(self.state_to_var[s])
            result = self.bdd.AND(result, v)
        return result

    def encode_observation_class(self, game: SymbolicPOGame, o: int) -> BDDNode:
        """Encode all supports consistent with observation o.

        States not in obs_class(o) must be absent.
        At least one state in obs_class(o) must be present.
        """
        obs_states = game.obs_class(o)
        non_obs = self.states_set - obs_states if hasattr(self, 'states_set') else set(self.states) - obs_states

        # Non-observation states must be absent
        result = self.bdd.TRUE
        for s in non_obs:
            v = self.bdd.named_var(self.state_to_var[s])
            result = self.bdd.AND(result, self.bdd.NOT(v))

        # At least one observation state must be present
        at_least_one = self.bdd.FALSE
        for s in obs_states:
            v = self.bdd.named_var(self.state_to_var[s])
            at_least_one = self.bdd.OR(at_least_one, v)

        return self.bdd.AND(result, at_least_one)

    def decode_support(self, bdd_node: BDDNode) -> Optional[Set[int]]:
        """Decode a BDD cube back to a support set.

        Returns None if the BDD represents multiple support sets.
        """
        assignment = self.bdd.any_sat(bdd_node)
        if assignment is None:
            return None
        support = set()
        for s in self.states:
            var_name = self.state_to_var[s]
            idx = self.bdd.var_index(var_name)
            if assignment.get(idx, False):
                support.add(s)
        return support

    def enumerate_supports(self, bdd_node: BDDNode) -> List[Set[int]]:
        """Enumerate all support sets represented by a BDD."""
        num_vars = len(self.states)
        assignments = self.bdd.all_sat(bdd_node, num_vars)
        supports = []
        for assignment in assignments:
            support = set()
            for s in self.states:
                var_name = self.state_to_var[s]
                idx = self.bdd.var_index(var_name)
                if assignment.get(idx, False):
                    support.add(s)
            supports.append(support)
        return supports

    def support_count(self, bdd_node: BDDNode) -> int:
        """Count the number of support sets represented by a BDD."""
        return self.bdd.sat_count(bdd_node, len(self.states))

    def union(self, a: BDDNode, b: BDDNode) -> BDDNode:
        """Union of two belief support sets (OR)."""
        return self.bdd.OR(a, b)

    def intersect(self, a: BDDNode, b: BDDNode) -> BDDNode:
        """Intersection of belief support sets (AND)."""
        return self.bdd.AND(a, b)

    def complement(self, a: BDDNode) -> BDDNode:
        """Complement of a belief support set."""
        return self.bdd.NOT(a)

    def is_empty(self, a: BDDNode) -> bool:
        """Check if a BDD represents the empty set."""
        return a is self.bdd.FALSE or self.bdd.any_sat(a) is None

    def contains(self, bdd_set: BDDNode, support: Set[int]) -> bool:
        """Check if a specific support is in the BDD set."""
        cube = self.encode_support(support)
        check = self.bdd.AND(bdd_set, cube)
        return not self.is_empty(check)


# ---------------------------------------------------------------------------
# Symbolic belief update
# ---------------------------------------------------------------------------

def symbolic_belief_update(
    game: SymbolicPOGame,
    belief: SymbolicBelief,
    action: int,
    observation: int,
    encoder: BeliefBDDEncoder,
) -> Optional[SymbolicBelief]:
    """Update belief after action and observation, maintaining BDD support.

    Computes Bayesian update: b'(s') = sum_s b(s) * T(s,a,s') * [obs(s')=o] / Z
    Also updates the BDD support to reflect new possible states.
    """
    new_dist: Dict[int, Fraction] = {}
    obs_class = game.obs_class(observation)

    for s_next in obs_class:
        prob_sum = Fraction(0)
        for s in belief.support:
            key = (s, action)
            if key not in game.transitions:
                continue
            for (target, _weight, prob) in game.transitions[key]:
                if target == s_next:
                    prob_sum += belief.prob(s) * prob
        if prob_sum > 0:
            new_dist[s_next] = prob_sum

    if not new_dist:
        return None

    # Normalize
    total = sum(new_dist.values())
    if total == 0:
        return None
    new_dist = {s: p / total for s, p in new_dist.items()}

    # Build BDD support
    new_support = set(new_dist.keys())
    bdd_support = encoder.encode_support(new_support)

    return SymbolicBelief(
        bdd_support=bdd_support,
        distribution=new_dist,
    )


def symbolic_possible_observations(
    game: SymbolicPOGame,
    belief: SymbolicBelief,
    action: int,
) -> Set[int]:
    """Observations possible from belief under action."""
    obs_set = set()
    for s in belief.support:
        key = (s, action)
        if key not in game.transitions:
            continue
        for (s_next, _w, p) in game.transitions[key]:
            if p > 0:
                obs_set.add(game.obs[s_next])
    return obs_set


# ---------------------------------------------------------------------------
# Belief-energy game construction
# ---------------------------------------------------------------------------

def build_belief_energy_game(
    game: SymbolicPOGame,
    encoder: BeliefBDDEncoder,
    max_beliefs: int = 200,
) -> Tuple[EnergyGame, Dict[int, SymbolicBelief], Dict[int, int]]:
    """Build a finite energy game over explored belief states.

    Explores beliefs breadth-first from initial states, stopping at max_beliefs.
    Each belief state becomes a vertex in the energy game.
    Edge weights are expected costs/rewards from the original game.

    Returns:
        energy_game: the constructed energy game
        belief_map: vertex_id -> SymbolicBelief
        obs_map: vertex_id -> observation
    """
    # Start from initial belief
    initial_dist: Dict[int, Fraction] = {}
    for s in game.initial:
        initial_dist[s] = Fraction(1, len(game.initial))

    initial_support = set(game.initial)
    initial_bdd = encoder.encode_support(initial_support)
    initial_belief = SymbolicBelief(
        bdd_support=initial_bdd,
        distribution=initial_dist,
    )

    eg = EnergyGame()
    belief_map: Dict[int, SymbolicBelief] = {}
    obs_map: Dict[int, int] = {}

    # Map support sets to vertex IDs for dedup
    support_to_id: Dict[frozenset, int] = {}
    next_id = 0
    queue: List[int] = []

    def get_or_create_belief(b: SymbolicBelief) -> int:
        nonlocal next_id
        key = frozenset(b.support)
        if key in support_to_id:
            return support_to_id[key]
        if len(belief_map) >= max_beliefs:
            return -1
        vid = next_id
        next_id += 1
        support_to_id[key] = vid

        # Determine owner: P0 if all states in support are P0-owned,
        # otherwise P1 (adversary controls resolution)
        owners = {game.owner.get(s, "P0") for s in b.support}
        if len(owners) == 1 and "P0" in owners:
            player = Player.EVEN
        else:
            player = Player.ODD

        eg.add_vertex(vid, player)
        belief_map[vid] = b

        # Observation from belief (should be uniform within support)
        observations = {game.obs[s] for s in b.support}
        obs_map[vid] = min(observations) if observations else 0

        queue.append(vid)
        return vid

    get_or_create_belief(initial_belief)

    while queue:
        vid = queue.pop(0)
        belief = belief_map[vid]

        # Get available actions from any state in support
        actions = set()
        for s in belief.support:
            actions |= game.get_actions(s)

        for action in actions:
            # Compute expected weight
            expected_weight = Fraction(0)
            for s in belief.support:
                key = (s, action)
                if key not in game.transitions:
                    continue
                for (_next, weight, prob) in game.transitions[key]:
                    expected_weight += belief.prob(s) * prob * weight

            # For each possible observation, create successor belief
            obs_set = symbolic_possible_observations(game, belief, action)
            for obs_val in obs_set:
                new_belief = symbolic_belief_update(
                    game, belief, action, obs_val, encoder,
                )
                if new_belief is None:
                    continue

                succ_id = get_or_create_belief(new_belief)
                if succ_id < 0:
                    continue  # max beliefs reached

                # Weight is the integer part of expected weight
                w = int(expected_weight) if expected_weight >= 0 else -int(-expected_weight)
                eg.add_edge(vid, succ_id, w)

    return eg, belief_map, obs_map


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_belief_energy(
    game: SymbolicPOGame,
    max_beliefs: int = 200,
) -> BeliefEnergyResult:
    """Solve a partially-observable energy game.

    Builds a belief-space energy game, solves it, and projects
    the solution back to observation-based strategies.
    """
    encoder = BeliefBDDEncoder(game.states)
    eg, belief_map, obs_map = build_belief_energy_game(
        game, encoder, max_beliefs,
    )

    if not game.states or not eg.vertices:
        return BeliefEnergyResult(
            min_energy={}, winning_beliefs=[], strategy={},
            value_function={}, belief_states_explored=0,
        )

    result = solve_energy(eg)

    # Project back to observations
    strategy: Dict[int, int] = {}
    min_energy_by_obs: Dict[int, Optional[int]] = {}
    value_by_obs: Dict[int, float] = {}

    for vid, belief in belief_map.items():
        o = obs_map[vid]
        e = result.min_energy.get(vid)
        if o not in min_energy_by_obs or (
            e is not None and (
                min_energy_by_obs[o] is None or e < min_energy_by_obs[o]
            )
        ):
            min_energy_by_obs[o] = e

        # Extract strategy
        if vid in result.strategy_energy:
            succ = result.strategy_energy[vid]
            # Find the action that leads to this successor
            for s in belief.support:
                for a in game.get_actions(s):
                    key = (s, a)
                    if key in game.transitions:
                        for (tgt, _w, _p) in game.transitions[key]:
                            if tgt in belief_map.get(succ, SymbolicBelief(
                                encoder.bdd.FALSE, {})).support:
                                strategy[o] = a
                                break

    # Compute value function from energy result
    for vid, belief in belief_map.items():
        o = obs_map[vid]
        e = result.min_energy.get(vid)
        if e is not None and e != float('inf'):
            value_by_obs[o] = float(e)
        elif o not in value_by_obs:
            value_by_obs[o] = float('inf')

    winning = [
        belief_map[vid] for vid in result.win_energy
        if vid in belief_map
    ]

    return BeliefEnergyResult(
        min_energy=min_energy_by_obs,
        winning_beliefs=winning,
        strategy=strategy,
        value_function=value_by_obs,
        iterations=0,
        belief_states_explored=len(belief_map),
    )


def solve_belief_mean_payoff(
    game: SymbolicPOGame,
    max_beliefs: int = 200,
) -> BeliefEnergyResult:
    """Solve a PO game for mean-payoff objective.

    Builds belief-space game and solves for mean-payoff values.
    """
    encoder = BeliefBDDEncoder(game.states)
    eg, belief_map, obs_map = build_belief_energy_game(
        game, encoder, max_beliefs,
    )

    if not game.states or not eg.vertices:
        return BeliefEnergyResult(
            min_energy={}, winning_beliefs=[], strategy={},
            value_function={}, belief_states_explored=0,
        )

    mp_result = solve_mean_payoff(eg)

    # Project to observations
    value_by_obs: Dict[int, float] = {}
    strategy: Dict[int, int] = {}

    for vid, belief in belief_map.items():
        o = obs_map[vid]
        val = mp_result.values.get(vid, 0.0)
        if o not in value_by_obs or val > value_by_obs[o]:
            value_by_obs[o] = val

        if vid in mp_result.strategy_p0:
            succ = mp_result.strategy_p0[vid]
            for s in belief.support:
                for a in game.get_actions(s):
                    key = (s, a)
                    if key in game.transitions:
                        for (tgt, _w, _p) in game.transitions[key]:
                            if tgt in belief_map.get(succ, SymbolicBelief(
                                encoder.bdd.FALSE, {})).support:
                                strategy[o] = a
                                break

    winning = [
        belief_map[vid] for vid in mp_result.win_nonneg
        if vid in belief_map
    ]

    return BeliefEnergyResult(
        min_energy={},
        winning_beliefs=winning,
        strategy=strategy,
        value_function=value_by_obs,
        belief_states_explored=len(belief_map),
    )


# ---------------------------------------------------------------------------
# Symbolic safety: BDD-based backward reachability over beliefs
# ---------------------------------------------------------------------------

def symbolic_safety_analysis(
    game: SymbolicPOGame,
    encoder: BeliefBDDEncoder,
    max_steps: int = 100,
) -> Tuple[BDDNode, int]:
    """Compute safe belief region via backward BDD fixed-point.

    Starts from unsafe belief supports and computes backward reachable
    beliefs (those that can be forced to unsafe by Odd).
    Returns BDD of SAFE belief supports.
    """
    # Encode unsafe supports: any support containing an unsafe state
    unsafe_bdd = encoder.bdd.FALSE
    for s in game.unsafe:
        if s in encoder.state_to_var:
            v = encoder.bdd.named_var(encoder.state_to_var[s])
            unsafe_bdd = encoder.bdd.OR(unsafe_bdd, v)

    # Fixed-point: expand unsafe backwards
    current = unsafe_bdd
    for step in range(max_steps):
        # Pre-image: beliefs from which Odd can force into current
        pre = _symbolic_preimage(game, encoder, current)
        expanded = encoder.bdd.OR(current, pre)
        if expanded is current or encoder.bdd.node_count(expanded) == encoder.bdd.node_count(current):
            # Check actual equivalence
            diff = encoder.bdd.AND(expanded, encoder.bdd.NOT(current))
            if encoder.is_empty(diff):
                break
        current = expanded

    # Safe = NOT unsafe_reachable
    safe = encoder.bdd.NOT(current)
    return safe, step + 1


def _symbolic_preimage(
    game: SymbolicPOGame,
    encoder: BeliefBDDEncoder,
    target_bdd: BDDNode,
) -> BDDNode:
    """Compute pre-image of a BDD belief set at state level.

    A state s is in the pre-image if it has a transition to some state
    s_next that is in the target set. Target membership is checked by
    restricting the target BDD: set s_next=True and all other state vars
    to False, then check if the result is TRUE.
    """
    # First, decode which individual states are in the target set.
    # A state s is "in the target" if the support {s} alone satisfies target_bdd.
    target_states: Set[int] = set()
    for s in game.states:
        if s not in encoder.state_to_var:
            continue
        cube = encoder.encode_support({s})
        check = encoder.bdd.AND(target_bdd, cube)
        if not encoder.is_empty(check):
            target_states.add(s)

    pre = encoder.bdd.FALSE
    for s in game.states:
        for a in game.get_actions(s):
            key = (s, a)
            if key not in game.transitions:
                continue
            for (s_next, _w, _p) in game.transitions[key]:
                if s_next in target_states:
                    v_s = encoder.bdd.named_var(encoder.state_to_var[s])
                    pre = encoder.bdd.OR(pre, v_s)
    return pre


# ---------------------------------------------------------------------------
# Belief reachability analysis (symbolic)
# ---------------------------------------------------------------------------

def symbolic_belief_reachability(
    game: SymbolicPOGame,
    encoder: BeliefBDDEncoder,
    max_steps: int = 50,
) -> Tuple[BDDNode, int]:
    """Forward reachability of belief supports from initial states.

    Returns BDD representing all reachable belief supports and
    the number of iterations until fixpoint.
    """
    initial_support = encoder.encode_support(game.initial)
    reachable = initial_support

    for step in range(max_steps):
        # Image: beliefs reachable in one step from current
        img = _symbolic_image(game, encoder, reachable)
        expanded = encoder.bdd.OR(reachable, img)
        diff = encoder.bdd.AND(expanded, encoder.bdd.NOT(reachable))
        if encoder.is_empty(diff):
            break
        reachable = expanded

    return reachable, step + 1


def _symbolic_image(
    game: SymbolicPOGame,
    encoder: BeliefBDDEncoder,
    source_bdd: BDDNode,
) -> BDDNode:
    """Compute image (successors) of a BDD belief set at state level."""
    img = encoder.bdd.FALSE
    for s in game.states:
        if s not in encoder.state_to_var:
            continue
        v_s = encoder.bdd.named_var(encoder.state_to_var[s])
        check = encoder.bdd.AND(source_bdd, v_s)
        if encoder.is_empty(check):
            continue
        for a in game.get_actions(s):
            key = (s, a)
            if key not in game.transitions:
                continue
            for (s_next, _w, _p) in game.transitions[key]:
                if s_next in encoder.state_to_var:
                    v_next = encoder.bdd.named_var(encoder.state_to_var[s_next])
                    img = encoder.bdd.OR(img, v_next)
    return img


# ---------------------------------------------------------------------------
# POMDP to SymbolicPOGame conversion
# ---------------------------------------------------------------------------

def pomdp_to_symbolic_game(
    pomdp: POMDP,
    default_weight: int = 0,
) -> SymbolicPOGame:
    """Convert a POMDP to a SymbolicPOGame.

    All states become P0-owned (single-player PO game).
    Rewards become edge weights.
    """
    game = SymbolicPOGame()

    for s in pomdp.states:
        game.add_state(s, "P0", pomdp.obs.get(s, s))
    game.initial = set(s for s, _p in pomdp.initial) if pomdp.initial else set()
    game.target = set(pomdp.target) if pomdp.target else set()

    for s in pomdp.states:
        for a in pomdp.get_actions(s):
            game.actions_even.add(a)
            transitions = pomdp.get_transitions(s, a)
            weight = int(pomdp.get_reward(s, a)) if pomdp.get_reward(s, a) else default_weight
            for (s_next, prob) in transitions:
                game.add_transition(s, a, s_next, weight, prob)

    return game


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_belief_energy_game(
    game: SymbolicPOGame,
    strategy: Dict[int, int],
    steps: int,
    initial_state: Optional[int] = None,
    initial_energy: int = 0,
) -> List[dict]:
    """Simulate a play of the belief-energy game.

    P0 plays according to observation-based strategy.
    Nature resolves probabilistic transitions (picks first successor).
    """
    import random

    if initial_state is not None:
        state = initial_state
    elif game.initial:
        state = min(game.initial)
    else:
        state = min(game.states)

    energy = initial_energy
    encoder = BeliefBDDEncoder(game.states)
    belief_dist = {s: Fraction(1, len(game.initial)) for s in game.initial} if game.initial else {state: Fraction(1)}
    belief = SymbolicBelief(
        bdd_support=encoder.encode_support(set(belief_dist.keys())),
        distribution=belief_dist,
    )

    trace = []
    for step in range(steps):
        obs_val = game.obs.get(state, 0)
        action = strategy.get(obs_val)
        if action is None:
            # No strategy for this obs -- pick first available
            actions = game.get_actions(state)
            action = min(actions) if actions else 0

        key = (state, action)
        succs = game.transitions.get(key, [])
        if not succs:
            trace.append({
                'step': step, 'state': state, 'observation': obs_val,
                'action': action, 'weight': 0, 'energy': energy,
                'belief_size': len(belief.support),
                'belief_entropy': belief.entropy(),
                'terminated': True,
            })
            break

        # Resolve probabilistic transition
        r = random.random()
        cumulative = 0.0
        chosen = succs[0]
        for (s_next, w, p) in succs:
            cumulative += float(p)
            if r <= cumulative:
                chosen = (s_next, w, p)
                break

        next_state, weight, _prob = chosen
        energy += weight

        # Update belief
        new_obs = game.obs.get(next_state, 0)
        new_belief = symbolic_belief_update(game, belief, action, new_obs, encoder)
        if new_belief is not None:
            belief = new_belief

        trace.append({
            'step': step, 'state': state, 'observation': obs_val,
            'action': action, 'weight': weight, 'energy': energy,
            'next_state': next_state,
            'belief_size': len(belief.support),
            'belief_entropy': belief.entropy(),
            'terminated': False,
        })

        state = next_state

    return trace


# ---------------------------------------------------------------------------
# Analysis and comparison
# ---------------------------------------------------------------------------

def game_statistics(game: SymbolicPOGame) -> dict:
    """Compute statistics about a symbolic PO game."""
    num_transitions = sum(len(v) for v in game.transitions.values())
    owners = {}
    for s in game.states:
        o = game.owner.get(s, "P0")
        owners[o] = owners.get(o, 0) + 1

    return {
        'num_states': len(game.states),
        'num_observations': len(game.all_observations()),
        'num_actions_even': len(game.actions_even),
        'num_actions_odd': len(game.actions_odd),
        'num_transitions': num_transitions,
        'owners': owners,
        'num_initial': len(game.initial),
        'num_target': len(game.target),
        'num_unsafe': len(game.unsafe),
    }


def compare_energy_vs_mean_payoff(
    game: SymbolicPOGame,
    max_beliefs: int = 200,
) -> dict:
    """Compare energy and mean-payoff solutions for the same PO game."""
    energy_result = solve_belief_energy(game, max_beliefs)
    mp_result = solve_belief_mean_payoff(game, max_beliefs)

    return {
        'energy': {
            'min_energy': energy_result.min_energy,
            'winning_count': len(energy_result.winning_beliefs),
            'belief_states': energy_result.belief_states_explored,
        },
        'mean_payoff': {
            'values': mp_result.value_function,
            'winning_count': len(mp_result.winning_beliefs),
            'belief_states': mp_result.belief_states_explored,
        },
    }


def analyze_belief_space(
    game: SymbolicPOGame,
    encoder: BeliefBDDEncoder,
    max_steps: int = 50,
) -> dict:
    """Analyze the belief space structure of a PO game."""
    reachable_bdd, reach_iters = symbolic_belief_reachability(
        game, encoder, max_steps,
    )
    reachable_count = encoder.support_count(reachable_bdd)

    safe_bdd, safe_iters = symbolic_safety_analysis(
        game, encoder, max_steps,
    )
    safe_count = encoder.support_count(safe_bdd)

    # Check if initial is safe
    initial_bdd = encoder.encode_support(game.initial) if game.initial else encoder.bdd.FALSE
    initial_safe = not encoder.is_empty(encoder.bdd.AND(initial_bdd, safe_bdd))

    return {
        'reachable_supports': reachable_count,
        'reachable_iterations': reach_iters,
        'safe_supports': safe_count,
        'safety_iterations': safe_iters,
        'initial_is_safe': initial_safe,
        'total_possible_supports': 2 ** len(game.states),
    }


# ---------------------------------------------------------------------------
# Example game builders
# ---------------------------------------------------------------------------

def make_tiger_game() -> SymbolicPOGame:
    """Classic Tiger POMDP as a symbolic PO game.

    2 states (tiger-left=0, tiger-right=1), 3 actions (listen=0, open-left=1, open-right=2).
    Listening gives noisy observation. Opening the right door gives reward.
    """
    game = SymbolicPOGame()
    # States: 0=tiger-left, 1=tiger-right
    # Observations: 0=hear-left, 1=hear-right
    game.add_state(0, "P0", 0)  # tiger-left, obs: hear-left (mostly)
    game.add_state(1, "P0", 1)  # tiger-right, obs: hear-right (mostly)
    game.initial = {0, 1}
    game.actions_even = {0, 1, 2}

    # Listen action (0): stays in same state, noisy observation
    # Weight 0 (small cost = -1 in energy)
    game.add_transition(0, 0, 0, -1, Fraction(85, 100))  # correct obs
    game.add_transition(0, 0, 1, -1, Fraction(15, 100))  # wrong obs (state change models noise)
    game.add_transition(1, 0, 1, -1, Fraction(85, 100))
    game.add_transition(1, 0, 0, -1, Fraction(15, 100))

    # Open-left (1): big penalty if tiger is there, reward otherwise
    game.add_transition(0, 1, 0, -100, Fraction(1))  # tiger attacks!
    game.add_transition(1, 1, 1, 10, Fraction(1))     # safe, reward

    # Open-right (2): symmetric
    game.add_transition(0, 2, 0, 10, Fraction(1))     # safe, reward
    game.add_transition(1, 2, 1, -100, Fraction(1))   # tiger attacks!

    return game


def make_maze_game(size: int = 3) -> SymbolicPOGame:
    """Grid maze with partial observation.

    P0 navigates a grid. Can only see wall/no-wall in current cell.
    Goal is bottom-right corner. P1 controls obstacle placement.
    """
    game = SymbolicPOGame()
    n = size

    for r in range(n):
        for c in range(n):
            s = r * n + c
            # Observation: position mod 2 (limited info)
            obs = (r + c) % 2
            owner = "P0"
            game.add_state(s, owner, obs)

    game.initial = {0}
    game.target = {n * n - 1}

    # Actions: 0=up, 1=right, 2=down, 3=left
    game.actions_even = {0, 1, 2, 3}

    for r in range(n):
        for c in range(n):
            s = r * n + c
            moves = []
            if r > 0: moves.append((0, (r-1)*n + c))
            if c < n-1: moves.append((1, r*n + c + 1))
            if r < n-1: moves.append((2, (r+1)*n + c))
            if c > 0: moves.append((3, r*n + c - 1))

            for action, target in moves:
                # Weight: -1 per step (minimize steps)
                game.add_transition(s, action, target, -1, Fraction(1))

            # Self-loop for invalid moves (stay in place)
            valid_actions = {a for a, _ in moves}
            for a in game.actions_even - valid_actions:
                game.add_transition(s, a, s, -1, Fraction(1))

    return game


def make_surveillance_game() -> SymbolicPOGame:
    """Surveillance game: P0 patrols, P1 (adversary) tries to intrude.

    4 locations in a ring. P0 has noisy sensor (sees adjacent only).
    P1 tries to reach location 0 (asset). Energy = time P1 spends outside.
    """
    game = SymbolicPOGame()

    # States: (patrol_pos, intruder_pos) encoded as patrol*4 + intruder
    for pp in range(4):
        for ip in range(4):
            s = pp * 4 + ip
            # P0 observes own position + whether intruder is adjacent
            adj = abs(pp - ip) <= 1 or abs(pp - ip) == 3  # ring adjacency
            obs = pp * 2 + (1 if adj else 0)
            # P0 controls patrol, P1 controls intruder
            # Alternate: even states = P0 turn, odd = P1 turn
            owner = "P0" if (pp + ip) % 2 == 0 else "P1"
            game.add_state(s, owner, obs)

    game.initial = {0 * 4 + 2}  # patrol at 0, intruder at 2
    game.unsafe = {pp * 4 + 0 for pp in range(1, 4)}  # intruder at asset (0) when patrol away

    # P0 actions: move clockwise (0) or counter-clockwise (1) or stay (2)
    game.actions_even = {0, 1, 2}
    # P1 actions: same set
    game.actions_odd = {0, 1, 2}

    for pp in range(4):
        for ip in range(4):
            s = pp * 4 + ip
            if game.owner[s] == "P0":
                for a in game.actions_even:
                    new_pp = (pp + (1 if a == 0 else (-1 if a == 1 else 0))) % 4
                    s_next = new_pp * 4 + ip
                    # Weight: +1 if intruder not at asset (good for P0)
                    w = 1 if ip != 0 else -10
                    game.add_transition(s, a, s_next, w, Fraction(1))
            else:
                for a in game.actions_odd:
                    new_ip = (ip + (1 if a == 0 else (-1 if a == 1 else 0))) % 4
                    s_next = pp * 4 + new_ip
                    w = 1 if new_ip != 0 else -10
                    game.add_transition(s, a, s_next, w, Fraction(1))

    return game
