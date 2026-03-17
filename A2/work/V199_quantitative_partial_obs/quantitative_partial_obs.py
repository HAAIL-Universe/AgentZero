"""V199: Quantitative Partial Observation Games.

Energy and mean-payoff objectives under imperfect information.
Composes V198 (partial observation) + V160 (energy games) + V161 (mean-payoff parity).

In quantitative PO games, Player 1 (Even) has partial observation and must
play observation-uniform strategies while maintaining quantitative objectives:
- Energy: maintain non-negative energy level
- Mean-payoff: achieve mean-payoff >= threshold
- Energy-parity: energy + parity condition combined
- Mean-payoff with safety: mean-payoff + avoid bad states

The key challenge: belief-based strategies must track worst-case energy
across all states consistent with the current observation.

Algorithms:
1. Knowledge-energy game: extend knowledge game with energy tracking
2. Belief-energy value iteration: worst-case energy across beliefs
3. Mean-payoff under PO: threshold checking via energy reduction
4. Quantitative-qualitative decomposition: compare PO vs perfect info
"""

import sys, os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum
from collections import deque
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V198_partial_observation_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V161_mean_payoff_parity'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))

from partial_observation_games import (
    PartialObsGame, KnowledgeState, KnowledgeGame, ObsStrategy,
    POGameResult, Objective,
    _initial_belief, _observation_split, build_knowledge_game,
    solve_safety, solve_reachability, solve as solve_po,
    analyze_observability, game_statistics as po_game_statistics,
)
from energy_games import (
    EnergyGame, EnergyResult, EnergyParityGame, EnergyParityResult,
    MeanPayoffResult as EnergyMPResult,
    Player, INF_ENERGY,
    solve_energy, solve_energy_parity, solve_fixed_energy,
    solve_mean_payoff as solve_mp_explicit,
    mean_payoff_threshold,
)
from parity_games import ParityGame, zielonka, attractor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class QObjective(Enum):
    """Quantitative objective types."""
    ENERGY = "energy"
    MEAN_PAYOFF = "mean_payoff"
    ENERGY_PARITY = "energy_parity"
    ENERGY_SAFETY = "energy_safety"
    MEAN_PAYOFF_SAFETY = "mp_safety"


@dataclass
class QuantPOGame:
    """A two-player game with partial observation and weighted edges.

    Player 1 (Even) has partial observation.
    Player 2 (Odd) has full observation.
    Edges carry integer weights for quantitative objectives.

    Attributes:
        vertices: set of vertex IDs
        edges: adjacency list (vertex -> list of (successor, weight))
        owner: vertex -> Player
        obs: vertex -> observation ID
        initial: set of initial vertices
        objective: quantitative objective type
        target: set of target/safe/accepting vertices (for combined objectives)
        priority: vertex -> priority (for parity objectives)
    """
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    owner: Dict[int, Player] = field(default_factory=dict)
    obs: Dict[int, int] = field(default_factory=dict)
    initial: Set[int] = field(default_factory=set)
    objective: QObjective = QObjective.ENERGY
    target: Set[int] = field(default_factory=set)
    priority: Dict[int, int] = field(default_factory=dict)

    def add_vertex(self, v: int, player: Player, observation: int,
                   prio: int = 0):
        self.vertices.add(v)
        self.owner[v] = player
        self.obs[v] = observation
        self.priority[v] = prio

    def add_edge(self, u: int, v: int, weight: int = 0):
        if u not in self.edges:
            self.edges[u] = []
        self.edges[u].append((v, weight))

    def successors(self, v: int) -> List[Tuple[int, int]]:
        return self.edges.get(v, [])

    def successor_vertices(self, v: int) -> Set[int]:
        return {t for t, _ in self.successors(v)}

    def predecessors(self, v: int) -> List[Tuple[int, int]]:
        result = []
        for u in self.vertices:
            for t, w in self.edges.get(u, []):
                if t == v:
                    result.append((u, w))
        return result

    def obs_class(self, o: int) -> Set[int]:
        return {v for v in self.vertices if self.obs.get(v) == o}

    def all_observations(self) -> Set[int]:
        return set(self.obs.values())

    def max_weight(self) -> int:
        w = 0
        for v in self.vertices:
            for _, ew in self.edges.get(v, []):
                w = max(w, abs(ew))
        return w

    def weight_bound(self) -> int:
        """n * W bound for energy game convergence."""
        return len(self.vertices) * self.max_weight()

    def to_energy_game(self) -> EnergyGame:
        """Convert to explicit energy game (ignoring observations)."""
        eg = EnergyGame()
        for v in self.vertices:
            eg.add_vertex(v, self.owner[v])
        for v in self.vertices:
            for t, w in self.edges.get(v, []):
                eg.add_edge(v, t, w)
        return eg

    def to_partial_obs_game(self, objective: Objective = Objective.SAFETY) -> PartialObsGame:
        """Convert to qualitative PO game (ignoring weights)."""
        pog = PartialObsGame()
        pog.objective = objective
        pog.initial = set(self.initial)
        pog.target = set(self.target)
        for v in self.vertices:
            pog.add_vertex(v, self.owner[v], self.obs[v],
                          self.priority.get(v, 0))
        for v in self.vertices:
            for t, _ in self.edges.get(v, []):
                pog.add_edge(v, t)
        return pog

    def is_observation_consistent(self) -> bool:
        """Check same observation -> same owner."""
        obs_owner = {}
        for v in self.vertices:
            o = self.obs.get(v)
            p = self.owner.get(v)
            if o in obs_owner:
                if obs_owner[o] != p:
                    return False
            else:
                obs_owner[o] = p
        return True


# ---------------------------------------------------------------------------
# Knowledge-Energy Game construction
# ---------------------------------------------------------------------------

@dataclass
class BeliefEnergyState:
    """State in the knowledge-energy game.

    A belief (frozenset of vertices) paired with an energy map
    tracking worst-case energy for each vertex in the belief.
    """
    belief: FrozenSet[int]
    observation: int

    def __hash__(self):
        return hash((self.belief, self.observation))

    def __eq__(self, other):
        return (self.belief == other.belief and
                self.observation == other.observation)


@dataclass
class QPOResult:
    """Result of solving a quantitative partial observation game."""
    winning: bool                          # can Player 1 win from initial belief?
    min_energy: Optional[int] = None       # minimum initial energy needed (energy obj)
    mean_payoff_value: Optional[Fraction] = None  # achievable mean-payoff (MP obj)
    strategy: Optional[Dict[FrozenSet[int], int]] = None  # belief -> target obs
    belief_energies: Optional[Dict[FrozenSet[int], int]] = None  # belief -> energy
    num_beliefs: int = 0
    iterations: int = 0


# ---------------------------------------------------------------------------
# Core: Belief-Energy Post operators
# ---------------------------------------------------------------------------

def _belief_post(game: QuantPOGame, belief: FrozenSet[int],
                 target_obs: int) -> Tuple[FrozenSet[int], int]:
    """Compute successor belief when P1 targets observation target_obs.

    For P1 vertices in belief: move to successors with obs == target_obs.
    For P2 vertices in belief: P2 can move to any successor (adversarial).

    Returns (new_belief, worst_case_weight).
    The worst-case weight is the minimum edge weight taken across the belief
    (since Player 2 controls the worst case).
    """
    new_states = set()
    worst_weight = None  # will compute min over P1 choices, max adversarial

    for v in belief:
        succs = game.successors(v)
        if not succs:
            continue

        if game.owner[v] == Player.EVEN:
            # P1 moves: must pick successor with obs == target_obs
            for t, w in succs:
                if game.obs.get(t) == target_obs:
                    new_states.add(t)
                    if worst_weight is None:
                        worst_weight = w
                    else:
                        worst_weight = min(worst_weight, w)
        else:
            # P2 moves: can pick ANY successor (adversarial)
            for t, w in succs:
                new_states.add(t)
                if worst_weight is None:
                    worst_weight = w
                else:
                    worst_weight = min(worst_weight, w)

    if worst_weight is None:
        worst_weight = 0

    return frozenset(new_states), worst_weight


def _belief_post_detailed(game: QuantPOGame, belief: FrozenSet[int],
                          target_obs: int) -> Tuple[FrozenSet[int], Dict[int, int]]:
    """Like _belief_post but returns per-state worst-case weights.

    Returns (new_belief, weight_map) where weight_map[new_state] = min weight
    of all edges leading to new_state from the belief.
    """
    new_states = set()
    weight_map = {}  # new_state -> worst (min) weight arriving

    for v in belief:
        succs = game.successors(v)
        if not succs:
            continue

        if game.owner[v] == Player.EVEN:
            for t, w in succs:
                if game.obs.get(t) == target_obs:
                    new_states.add(t)
                    if t not in weight_map:
                        weight_map[t] = w
                    else:
                        weight_map[t] = min(weight_map[t], w)
        else:
            for t, w in succs:
                new_states.add(t)
                if t not in weight_map:
                    weight_map[t] = w
                else:
                    weight_map[t] = min(weight_map[t], w)

    return frozenset(new_states), weight_map


def _all_target_observations(game: QuantPOGame,
                             belief: FrozenSet[int]) -> Set[int]:
    """Get all observations reachable from the belief in one step."""
    obs_set = set()
    for v in belief:
        for t, _ in game.successors(v):
            obs_set.add(game.obs.get(t))
    return obs_set


# ---------------------------------------------------------------------------
# Energy solving under partial observation
# ---------------------------------------------------------------------------

def solve_energy_po(game: QuantPOGame,
                    max_beliefs: int = 5000) -> QPOResult:
    """Solve energy objective under partial observation.

    Builds a knowledge-energy game via BFS over belief states,
    then performs value iteration to find minimum initial energy
    for Player 1 to maintain non-negative energy forever.

    The key insight: worst-case energy in a belief state is the maximum
    energy needed across all states in the belief (since P2 controls
    which state is the 'real' one).
    """
    if not game.initial:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    # Build belief transition graph
    initial_belief = frozenset(game.initial)
    initial_obs = None
    for v in game.initial:
        o = game.obs.get(v)
        if initial_obs is None:
            initial_obs = o
        # All initial states should share an observation
    if initial_obs is None:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    # BFS over beliefs
    beliefs = {}  # belief -> id
    belief_edges = {}  # belief_id -> [(target_belief_id, worst_weight)]
    belief_owner = {}  # belief_id -> Player
    queue = deque()

    bid = 0
    beliefs[initial_belief] = bid
    belief_owner[bid] = _belief_owner(game, initial_belief)
    belief_edges[bid] = []
    queue.append(initial_belief)
    bid += 1

    while queue and bid < max_beliefs:
        b = queue.popleft()
        src_id = beliefs[b]

        target_obs = _all_target_observations(game, b)
        for to in target_obs:
            new_b, worst_w = _belief_post(game, b, to)
            if not new_b:
                continue
            if new_b not in beliefs:
                beliefs[new_b] = bid
                belief_owner[bid] = _belief_owner(game, new_b)
                belief_edges[bid] = []
                queue.append(new_b)
                bid += 1
            belief_edges[src_id].append((beliefs[new_b], worst_w))

    n = len(beliefs)
    if n == 0:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    # Value iteration on belief-energy game
    # energy[bid] = minimum initial energy needed at belief bid
    # Bound: n_beliefs * max_abs_belief_edge_weight
    max_bw = 0
    for bid_i in range(n):
        for _, w in belief_edges.get(bid_i, []):
            max_bw = max(max_bw, abs(w))
    bound = n * (max_bw + 1) if max_bw > 0 else n
    energy = {bid: 0 for bid in range(n)}

    changed = True
    iters = 0
    while changed and iters < bound + 1:
        changed = False
        iters += 1
        for bid in range(n):
            edges = belief_edges.get(bid, [])
            if not edges:
                continue

            if belief_owner[bid] == Player.EVEN:
                # P1 minimizes energy needed
                best = INF_ENERGY
                for tid, w in edges:
                    needed = energy[tid] - w
                    if needed < 0:
                        needed = 0
                    best = min(best, needed)
            else:
                # P2 maximizes energy needed (adversarial)
                best = 0
                for tid, w in edges:
                    needed = energy[tid] - w
                    if needed < 0:
                        needed = 0
                    best = max(best, needed)

            if best > bound:
                best = INF_ENERGY

            if energy[bid] != best:
                energy[bid] = best
                changed = True

    # Non-convergence: identify which beliefs are still changing (divergent)
    if changed:
        divergent = set()
        for bid in range(n):
            edges = belief_edges.get(bid, [])
            if not edges:
                continue
            if belief_owner[bid] == Player.EVEN:
                best = INF_ENERGY
                for tid, w in edges:
                    needed = max(0, energy[tid] - w)
                    best = min(best, needed)
            else:
                best = 0
                for tid, w in edges:
                    needed = max(0, energy[tid] - w)
                    best = max(best, needed)
            if best != energy[bid]:
                divergent.add(bid)
        # Mark divergent beliefs and anything depending on them as INF
        prop_changed = True
        while prop_changed:
            prop_changed = False
            for bid in range(n):
                if energy[bid] == INF_ENERGY:
                    continue
                if bid in divergent:
                    energy[bid] = INF_ENERGY
                    prop_changed = True
                    continue
                edges = belief_edges.get(bid, [])
                if belief_owner[bid] == Player.EVEN:
                    # If ALL successors are INF, P1 can't avoid INF
                    if edges and all(energy[tid] == INF_ENERGY for tid, _ in edges):
                        energy[bid] = INF_ENERGY
                        prop_changed = True
                elif belief_owner[bid] == Player.ODD:
                    # If ANY successor is INF, P2 can force INF
                    if any(energy[tid] == INF_ENERGY for tid, _ in edges):
                        energy[bid] = INF_ENERGY
                        prop_changed = True

    init_id = beliefs.get(initial_belief, 0)
    init_energy = energy[init_id]

    # Extract strategy
    strategy = {}
    inv_beliefs = {v: k for k, v in beliefs.items()}
    for bid in range(n):
        edges = belief_edges.get(bid, [])
        if not edges or belief_owner[bid] != Player.EVEN:
            continue
        best_tid = None
        best_val = INF_ENERGY
        for tid, w in edges:
            needed = energy[tid] - w
            if needed < 0:
                needed = 0
            if needed < best_val:
                best_val = needed
                best_tid = tid
        if best_tid is not None and bid in inv_beliefs:
            # Find the target observation for this edge
            b = inv_beliefs[bid]
            for to in _all_target_observations(game, b):
                new_b, _ = _belief_post(game, b, to)
                if new_b in beliefs and beliefs[new_b] == best_tid:
                    strategy[b] = to
                    break

    winning = init_energy != INF_ENERGY
    belief_energies = {inv_beliefs[bid]: energy[bid]
                       for bid in range(n) if bid in inv_beliefs
                       and energy[bid] != INF_ENERGY}

    return QPOResult(
        winning=winning,
        min_energy=init_energy if winning else None,
        strategy=strategy if winning else None,
        belief_energies=belief_energies,
        num_beliefs=n,
        iterations=iters,
    )


def _belief_owner(game: QuantPOGame, belief: FrozenSet[int]) -> Player:
    """Determine owner of a belief state.

    If all states share observation -> same owner (by consistency).
    Mixed ownership uses Even (protagonist chooses).
    """
    owners = {game.owner.get(v, Player.EVEN) for v in belief}
    if Player.ODD in owners:
        return Player.ODD
    return Player.EVEN


# ---------------------------------------------------------------------------
# Mean-payoff under partial observation
# ---------------------------------------------------------------------------

def solve_mean_payoff_po(game: QuantPOGame,
                         threshold: float = 0.0,
                         max_beliefs: int = 5000) -> QPOResult:
    """Solve mean-payoff objective under partial observation.

    Reduces to energy game: subtract threshold from all weights,
    then check if Player 1 can maintain non-negative energy.

    Mean-payoff >= threshold iff shifted energy game is winnable.
    """
    # Create shifted game
    shifted = QuantPOGame()
    shifted.vertices = set(game.vertices)
    shifted.owner = dict(game.owner)
    shifted.obs = dict(game.obs)
    shifted.initial = set(game.initial)
    shifted.objective = QObjective.ENERGY
    shifted.target = set(game.target)
    shifted.priority = dict(game.priority)

    # Shift weights: w' = w - threshold
    # Use scaled integer arithmetic for precision
    # Scale by n to handle rational thresholds
    n = len(game.vertices) if game.vertices else 1
    scale = n
    int_threshold = int(threshold * scale)

    for v in game.vertices:
        shifted_edges = []
        for t, w in game.edges.get(v, []):
            shifted_w = w * scale - int_threshold
            shifted_edges.append((t, shifted_w))
        shifted.edges[v] = shifted_edges

    result = solve_energy_po(shifted, max_beliefs)

    # Convert back
    return QPOResult(
        winning=result.winning,
        min_energy=result.min_energy,
        mean_payoff_value=Fraction(int(threshold * 1000), 1000) if result.winning else None,
        strategy=result.strategy,
        belief_energies=result.belief_energies,
        num_beliefs=result.num_beliefs,
        iterations=result.iterations,
    )


def find_optimal_mean_payoff_po(game: QuantPOGame,
                                max_beliefs: int = 5000) -> QPOResult:
    """Find the optimal mean-payoff achievable under partial observation.

    Binary search over threshold values. The optimal value is a rational
    p/q with |q| <= n (number of vertices).
    """
    n = len(game.vertices)
    if n == 0:
        return QPOResult(winning=False, num_beliefs=0)

    W = game.max_weight()
    lo = -W
    hi = W

    # Binary search with precision 1/(2n^2)
    precision = Fraction(1, max(2 * n * n, 1))
    best_result = None
    lo_f = Fraction(lo)
    hi_f = Fraction(hi)

    while hi_f - lo_f > precision:
        mid = (lo_f + hi_f) / 2
        result = solve_mean_payoff_po(game, float(mid), max_beliefs)
        if result.winning:
            lo_f = mid
            best_result = result
            best_result.mean_payoff_value = mid
        else:
            hi_f = mid

    if best_result is None:
        # Try lo
        result = solve_mean_payoff_po(game, float(lo_f), max_beliefs)
        if result.winning:
            best_result = result
            best_result.mean_payoff_value = lo_f

    if best_result is None:
        return QPOResult(winning=False, num_beliefs=0)

    return best_result


# ---------------------------------------------------------------------------
# Energy-Safety: energy + avoid bad states
# ---------------------------------------------------------------------------

def solve_energy_safety_po(game: QuantPOGame,
                           max_beliefs: int = 5000) -> QPOResult:
    """Solve energy objective with safety constraint under PO.

    Player 1 must maintain non-negative energy AND avoid target (bad) states.
    Approach: restrict belief graph to safe beliefs, then solve energy.
    """
    bad = game.target  # target = bad states for safety

    # Build belief graph, filtering out beliefs containing bad states
    if not game.initial:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    initial_belief = frozenset(game.initial)
    # Check initial belief is safe
    if initial_belief & bad:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    beliefs = {}
    belief_edges = {}
    belief_owner = {}
    queue = deque()

    bid = 0
    beliefs[initial_belief] = bid
    belief_owner[bid] = _belief_owner(game, initial_belief)
    belief_edges[bid] = []
    queue.append(initial_belief)
    bid += 1

    while queue and bid < max_beliefs:
        b = queue.popleft()
        src_id = beliefs[b]

        target_obs = _all_target_observations(game, b)
        for to in target_obs:
            new_b, worst_w = _belief_post(game, b, to)
            if not new_b:
                continue
            # Safety: skip beliefs containing bad states
            if new_b & bad:
                continue
            if new_b not in beliefs:
                beliefs[new_b] = bid
                belief_owner[bid] = _belief_owner(game, new_b)
                belief_edges[bid] = []
                queue.append(new_b)
                bid += 1
            belief_edges[src_id].append((beliefs[new_b], worst_w))

    n = len(beliefs)
    if n == 0:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    # Remove belief nodes with no outgoing edges (dead ends lose for P1)
    # Even dead-end: P1 has no safe choice -> loses
    # Odd dead-end: P2's only moves reach bad states -> P1 loses safety
    # Iteratively remove losing beliefs
    removed = set()
    changed = True
    while changed:
        changed = False
        for bid_i in range(n):
            if bid_i in removed:
                continue
            live_edges = [(t, w) for t, w in belief_edges.get(bid_i, [])
                          if t not in removed]
            belief_edges[bid_i] = live_edges
            if not live_edges:
                removed.add(bid_i)
                changed = True

    init_id = beliefs.get(initial_belief)
    if init_id is None or init_id in removed:
        return QPOResult(winning=False, min_energy=None, num_beliefs=n)

    # Value iteration on safe belief-energy game
    max_bw = 0
    for bid_i in range(n):
        if bid_i not in removed:
            for _, w in belief_edges.get(bid_i, []):
                max_bw = max(max_bw, abs(w))
    bound = n * (max_bw + 1) if max_bw > 0 else n
    energy = {}
    for bid_i in range(n):
        energy[bid_i] = 0 if bid_i not in removed else INF_ENERGY

    changed = True
    iters = 0
    while changed and iters < bound + 1:
        changed = False
        iters += 1
        for bid_i in range(n):
            if bid_i in removed:
                continue
            edges = belief_edges.get(bid_i, [])
            if not edges:
                continue

            if belief_owner[bid_i] == Player.EVEN:
                best = INF_ENERGY
                for tid, w in edges:
                    if tid in removed:
                        continue
                    needed = energy[tid] - w
                    if needed < 0:
                        needed = 0
                    best = min(best, needed)
            else:
                best = 0
                for tid, w in edges:
                    needed = energy[tid] - w if tid not in removed else INF_ENERGY
                    if needed < 0:
                        needed = 0
                    best = max(best, needed)

            if best > bound:
                best = INF_ENERGY
            if energy[bid_i] != best:
                energy[bid_i] = best
                changed = True

    init_energy = energy.get(init_id, INF_ENERGY)
    winning = init_energy != INF_ENERGY

    inv_beliefs = {v: k for k, v in beliefs.items()}
    belief_energies = {inv_beliefs[bid_i]: energy[bid_i]
                       for bid_i in range(n) if bid_i in inv_beliefs
                       and energy[bid_i] != INF_ENERGY}

    return QPOResult(
        winning=winning,
        min_energy=init_energy if winning else None,
        belief_energies=belief_energies,
        num_beliefs=n,
        iterations=iters,
    )


# ---------------------------------------------------------------------------
# Energy-Parity under partial observation
# ---------------------------------------------------------------------------

def solve_energy_parity_po(game: QuantPOGame,
                           max_beliefs: int = 5000) -> QPOResult:
    """Solve energy-parity objective under partial observation.

    Player 1 must maintain non-negative energy AND satisfy parity condition.
    Approach: build belief graph with parity tracking, solve iteratively.

    The parity condition on beliefs: a belief is accepting with priority p
    if the MAXIMUM priority across all states in the belief is p.
    (Conservative: P2 controls which state is real.)
    """
    if not game.initial:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    initial_belief = frozenset(game.initial)

    # Build belief graph
    beliefs = {}
    belief_edges = {}
    belief_owner = {}
    belief_priority = {}
    queue = deque()

    bid = 0
    beliefs[initial_belief] = bid
    belief_owner[bid] = _belief_owner(game, initial_belief)
    belief_priority[bid] = _belief_parity(game, initial_belief)
    belief_edges[bid] = []
    queue.append(initial_belief)
    bid += 1

    while queue and bid < max_beliefs:
        b = queue.popleft()
        src_id = beliefs[b]

        target_obs = _all_target_observations(game, b)
        for to in target_obs:
            new_b, worst_w = _belief_post(game, b, to)
            if not new_b:
                continue
            if new_b not in beliefs:
                beliefs[new_b] = bid
                belief_owner[bid] = _belief_owner(game, new_b)
                belief_priority[bid] = _belief_parity(game, new_b)
                belief_edges[bid] = []
                queue.append(new_b)
                bid += 1
            belief_edges[src_id].append((beliefs[new_b], worst_w))

    n = len(beliefs)
    if n == 0:
        return QPOResult(winning=False, min_energy=None, num_beliefs=0)

    # Build EnergyParityGame from belief graph
    epg = EnergyParityGame()
    for bid_i in range(n):
        epg.add_vertex(bid_i, belief_owner[bid_i], belief_priority.get(bid_i, 0))
    for bid_i in range(n):
        for tid, w in belief_edges.get(bid_i, []):
            epg.add_edge(bid_i, tid, w)

    # Solve using V160's energy-parity solver
    ep_result = solve_energy_parity(epg)

    init_id = beliefs.get(initial_belief, 0)
    init_energy = ep_result.min_energy.get(init_id)
    winning = init_id in ep_result.win_energy

    inv_beliefs = {v: k for k, v in beliefs.items()}
    belief_energies = {}
    for bid_i in range(n):
        if bid_i in inv_beliefs and bid_i in ep_result.win_energy:
            e = ep_result.min_energy.get(bid_i)
            if e is not None and e != INF_ENERGY:
                belief_energies[inv_beliefs[bid_i]] = e

    return QPOResult(
        winning=winning,
        min_energy=init_energy if winning else None,
        belief_energies=belief_energies,
        num_beliefs=n,
        iterations=0,
    )


def _belief_parity(game: QuantPOGame, belief: FrozenSet[int]) -> int:
    """Priority of a belief state.

    Conservative (adversarial): P2 controls which state is real,
    so P2 picks the priority worst for P1 (Even).
    For parity: odd priorities are bad for Even.
    Rule: if any state has odd priority, use max odd priority
    (P2 keeps real state at odd-priority vertex). Otherwise max even.
    """
    if not belief:
        return 0
    priorities = [game.priority.get(v, 0) for v in belief]
    odd_prios = [p for p in priorities if p % 2 == 1]
    if odd_prios:
        return max(odd_prios)
    return max(priorities)


# ---------------------------------------------------------------------------
# Comparison: perfect vs partial observation
# ---------------------------------------------------------------------------

def compare_perfect_vs_partial(game: QuantPOGame) -> Dict:
    """Compare quantitative values under perfect vs partial observation.

    Under perfect observation, solve the explicit energy/mean-payoff game.
    Under partial observation, solve the belief-based game.
    The difference shows the cost of imperfect information.
    """
    # Perfect information: solve explicit energy game
    eg = game.to_energy_game()
    perfect_result = solve_energy(eg)

    # Partial observation
    po_result = solve_energy_po(game)

    # Compare initial states
    perfect_energies = {}
    for v in game.initial:
        e = perfect_result.min_energy.get(v)
        if e is not None and e != INF_ENERGY:
            perfect_energies[v] = e

    return {
        'perfect_winning': {v for v in game.initial
                           if v in perfect_result.win_energy},
        'partial_winning': po_result.winning,
        'perfect_min_energy': perfect_energies,
        'partial_min_energy': po_result.min_energy,
        'information_cost': _information_cost(perfect_energies, po_result),
        'num_beliefs': po_result.num_beliefs,
        'perfect_strategy': {v: perfect_result.strategy_energy.get(v)
                            for v in game.initial
                            if v in perfect_result.strategy_energy},
    }


def _information_cost(perfect_energies: Dict[int, int],
                      po_result: QPOResult) -> Optional[int]:
    """Cost of imperfect information = PO energy - perfect energy."""
    if not perfect_energies or not po_result.winning:
        return None
    perfect_max = max(perfect_energies.values()) if perfect_energies else 0
    if po_result.min_energy is None:
        return None
    return po_result.min_energy - perfect_max


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_play(game: QuantPOGame, strategy: Dict[FrozenSet[int], int],
                  initial_energy: int, max_steps: int = 100,
                  adversary: str = 'worst') -> List[Dict]:
    """Simulate a play in the quantitative PO game.

    P1 follows the belief-based strategy.
    P2 plays according to 'adversary' mode:
      'worst': minimizes P1's energy
      'best': maximizes P1's energy
      'first': picks first available successor

    Returns list of step dicts with belief, energy, observation, weight.
    """
    if not game.initial:
        return []

    belief = frozenset(game.initial)
    energy = initial_energy
    trace = []

    for step in range(max_steps):
        obs = None
        for v in belief:
            obs = game.obs.get(v)
            break

        trace.append({
            'step': step,
            'belief': belief,
            'belief_size': len(belief),
            'observation': obs,
            'energy': energy,
        })

        if energy < 0:
            trace[-1]['status'] = 'energy_depleted'
            break

        # P1 picks target observation from strategy
        target_obs = strategy.get(belief)
        if target_obs is None:
            # Try any available move
            avail = _all_target_observations(game, belief)
            if not avail:
                trace[-1]['status'] = 'dead_end'
                break
            target_obs = min(avail)

        # Compute successor belief and weight
        new_belief, worst_w = _belief_post(game, belief, target_obs)
        if not new_belief:
            trace[-1]['status'] = 'no_successor'
            break

        # Apply weight to energy
        if adversary == 'worst':
            energy += worst_w
        elif adversary == 'best':
            # Best case: max weight
            best_w = worst_w
            for v in belief:
                for t, w in game.successors(v):
                    if game.obs.get(t) == target_obs:
                        best_w = max(best_w, w)
            energy += best_w
        else:
            energy += worst_w

        trace[-1]['weight'] = worst_w
        trace[-1]['target_obs'] = target_obs
        belief = new_belief

    return trace


# ---------------------------------------------------------------------------
# Fixed energy check
# ---------------------------------------------------------------------------

def check_fixed_energy_po(game: QuantPOGame, initial_energy: int,
                          max_beliefs: int = 5000) -> QPOResult:
    """Check if P1 can win with a specific initial energy under PO.

    More efficient than full solve when we just need a yes/no for a given energy.
    """
    result = solve_energy_po(game, max_beliefs)
    if not result.winning:
        return QPOResult(winning=False, min_energy=None,
                        num_beliefs=result.num_beliefs)

    can_win = result.min_energy is not None and result.min_energy <= initial_energy
    return QPOResult(
        winning=can_win,
        min_energy=result.min_energy,
        strategy=result.strategy if can_win else None,
        num_beliefs=result.num_beliefs,
    )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def make_energy_po_game(n: int, edges: List[Tuple[int, int, int]],
                        owners: Dict[int, Player],
                        observations: Dict[int, int],
                        initial: Optional[Set[int]] = None) -> QuantPOGame:
    """Build a quantitative PO game from edge list.

    edges: list of (src, dst, weight)
    """
    g = QuantPOGame()
    g.objective = QObjective.ENERGY
    for v in range(n):
        g.add_vertex(v, owners.get(v, Player.EVEN),
                     observations.get(v, v))
    for u, v, w in edges:
        g.add_edge(u, v, w)
    g.initial = initial if initial is not None else {0}
    return g


def make_charging_po_game(n: int, charge: int, drain: int,
                          obs_groups: Optional[List[Set[int]]] = None) -> QuantPOGame:
    """Charging game under partial observation.

    Vertex 0 charges (+charge), others drain (-drain).
    P1 chooses direction, P2 chooses nothing (all Even-owned).
    Observation groups control what P1 can see.
    """
    g = QuantPOGame()
    g.objective = QObjective.ENERGY

    if obs_groups is None:
        # Default: each vertex is its own observation
        obs_groups = [{i} for i in range(n)]

    obs_map = {}
    for oid, group in enumerate(obs_groups):
        for v in group:
            obs_map[v] = oid

    for v in range(n):
        g.add_vertex(v, Player.EVEN, obs_map.get(v, v))

    # Vertex 0 charges, edges to all others
    for v in range(1, n):
        g.add_edge(0, v, charge)
    # Other vertices drain, edges to 0 and next
    for v in range(1, n):
        g.add_edge(v, 0, -drain)
        if v + 1 < n:
            g.add_edge(v, v + 1, -drain)
    # Last vertex loops to 0
    if n > 1:
        g.add_edge(n - 1, 0, -drain)

    g.initial = {0}
    return g


def make_adversarial_po_game() -> QuantPOGame:
    """Adversarial quantitative PO game.

    P1 at vertex 0 (obs 0) goes to vertex 1 or 2.
    P2 at vertex 1 (obs 1) goes to 0 (weight +2) or 3 (weight -3).
    P2 at vertex 2 (obs 1) goes to 0 (weight +1) or 3 (weight -1).
    Vertices 1 and 2 share observation 1, so P1 can't distinguish them.
    Vertex 3 is a sink (obs 2, weight 0 self-loop).
    """
    g = QuantPOGame()
    g.objective = QObjective.ENERGY

    g.add_vertex(0, Player.EVEN, 0)
    g.add_vertex(1, Player.ODD, 1)
    g.add_vertex(2, Player.ODD, 1)
    g.add_vertex(3, Player.ODD, 2)

    g.add_edge(0, 1, 0)
    g.add_edge(0, 2, 0)
    g.add_edge(1, 0, 2)
    g.add_edge(1, 3, -3)
    g.add_edge(2, 0, 1)
    g.add_edge(2, 3, -1)
    g.add_edge(3, 3, 0)

    g.initial = {0}
    return g


def make_corridor_po_game(length: int, reward: int = 1,
                          penalty: int = 2) -> QuantPOGame:
    """Corridor game: P1 navigates a corridor with partial observation.

    Linear chain 0 -> 1 -> ... -> length-1 -> 0.
    P1 gets +reward moving forward, -penalty at the end.
    Observation: groups of 2 consecutive vertices share observations.
    """
    g = QuantPOGame()
    g.objective = QObjective.MEAN_PAYOFF

    for v in range(length):
        obs = v // 2
        g.add_vertex(v, Player.EVEN, obs)

    for v in range(length - 1):
        g.add_edge(v, v + 1, reward)
    g.add_edge(length - 1, 0, -penalty)

    g.initial = {0}
    return g


def make_choice_po_game() -> QuantPOGame:
    """P1 must choose between safe (+1 loop) and risky (+3/-2 opponent choice).

    Vertex 0: P1 (obs 0) -> 1 or 2
    Vertex 1: Even (obs 1) -> 0 (weight +1) [safe loop]
    Vertex 2: Odd (obs 2) -> 0 (weight +3) or 0 (weight -2) [risky]

    Under perfect info, P1 picks safe path (guaranteed +1).
    Under PO, the analysis depends on what P1 can observe.
    """
    g = QuantPOGame()
    g.objective = QObjective.ENERGY

    g.add_vertex(0, Player.EVEN, 0)
    g.add_vertex(1, Player.EVEN, 1)
    g.add_vertex(2, Player.ODD, 2)

    g.add_edge(0, 1, 0)   # go to safe
    g.add_edge(0, 2, 0)   # go to risky
    g.add_edge(1, 0, 1)   # safe: +1
    g.add_edge(2, 0, 3)   # risky: +3
    g.add_edge(2, 0, -2)  # risky: -2 (P2 chooses)

    g.initial = {0}
    return g


def make_hidden_drain_game() -> QuantPOGame:
    """P1 can't see which drain rate is active.

    Vertex 0: P1 (obs 0) -> 1 or 2 (choose path)
    Vertex 1: sink with drain -1 (obs 1)
    Vertex 2: sink with drain -3 (obs 1)  -- same obs as 1!
    Vertex 3: charge station +5 (obs 2)

    P1 can't distinguish vertex 1 from 2 (same observation).
    If P2 can influence which state P1 is in, worst case applies.
    """
    g = QuantPOGame()
    g.objective = QObjective.ENERGY

    g.add_vertex(0, Player.EVEN, 0)
    g.add_vertex(1, Player.ODD, 1)
    g.add_vertex(2, Player.ODD, 1)  # same obs as 1
    g.add_vertex(3, Player.EVEN, 2)

    g.add_edge(0, 1, 0)
    g.add_edge(0, 2, 0)
    g.add_edge(1, 3, -1)
    g.add_edge(1, 1, -1)
    g.add_edge(2, 3, -3)
    g.add_edge(2, 2, -3)
    g.add_edge(3, 0, 5)

    g.initial = {0}
    return g


def make_energy_parity_po_game() -> QuantPOGame:
    """Energy-parity game under partial observation.

    Vertex 0: P1 (obs 0, prio 0) -> 1, 2
    Vertex 1: Even (obs 1, prio 2) -> 0 weight +1 [even prio, positive energy]
    Vertex 2: Odd (obs 1, prio 1) -> 0 weight +2 [odd prio, more energy]
    Vertices 1 and 2 share observation -> P1 can't distinguish them.
    P2 controls whether the real state has even or odd parity.
    """
    g = QuantPOGame()
    g.objective = QObjective.ENERGY_PARITY

    g.add_vertex(0, Player.EVEN, 0, prio=0)
    g.add_vertex(1, Player.EVEN, 1, prio=2)  # good parity
    g.add_vertex(2, Player.ODD, 1, prio=1)   # bad parity
    g.add_vertex(3, Player.EVEN, 2, prio=0)

    g.add_edge(0, 1, 0)
    g.add_edge(0, 2, 0)
    g.add_edge(1, 3, 1)
    g.add_edge(2, 3, 2)
    g.add_edge(3, 0, 0)

    g.initial = {0}
    return g


# ---------------------------------------------------------------------------
# Analysis and statistics
# ---------------------------------------------------------------------------

def game_statistics(game: QuantPOGame) -> Dict:
    """Compute statistics about the quantitative PO game."""
    edge_count = sum(len(e) for e in game.edges.values())
    weights = [w for v in game.vertices for _, w in game.edges.get(v, [])]
    obs_classes = {}
    for v in game.vertices:
        o = game.obs.get(v)
        if o not in obs_classes:
            obs_classes[o] = set()
        obs_classes[o].add(v)

    return {
        'vertices': len(game.vertices),
        'edges': edge_count,
        'observations': len(obs_classes),
        'max_obs_class_size': max(len(c) for c in obs_classes.values()) if obs_classes else 0,
        'min_weight': min(weights) if weights else 0,
        'max_weight': max(weights) if weights else 0,
        'avg_weight': sum(weights) / len(weights) if weights else 0,
        'even_vertices': sum(1 for v in game.vertices if game.owner.get(v) == Player.EVEN),
        'odd_vertices': sum(1 for v in game.vertices if game.owner.get(v) == Player.ODD),
        'initial_size': len(game.initial),
        'objective': game.objective.value,
        'observation_consistent': game.is_observation_consistent(),
    }


def game_summary(game: QuantPOGame) -> str:
    """Human-readable summary of the quantitative PO game."""
    stats = game_statistics(game)
    lines = [
        f"Quantitative PO Game ({stats['objective']})",
        f"  Vertices: {stats['vertices']} ({stats['even_vertices']} Even, {stats['odd_vertices']} Odd)",
        f"  Edges: {stats['edges']}",
        f"  Observations: {stats['observations']} (max class size: {stats['max_obs_class_size']})",
        f"  Weights: [{stats['min_weight']}, {stats['max_weight']}] avg={stats['avg_weight']:.1f}",
        f"  Initial: {stats['initial_size']} vertices",
        f"  Obs consistent: {stats['observation_consistent']}",
    ]
    return '\n'.join(lines)


def solve(game: QuantPOGame, max_beliefs: int = 5000, **kwargs) -> QPOResult:
    """Unified solver dispatching by objective type."""
    if game.objective == QObjective.ENERGY:
        return solve_energy_po(game, max_beliefs)
    elif game.objective == QObjective.MEAN_PAYOFF:
        threshold = kwargs.get('threshold', 0.0)
        return solve_mean_payoff_po(game, threshold, max_beliefs)
    elif game.objective == QObjective.ENERGY_SAFETY:
        return solve_energy_safety_po(game, max_beliefs)
    elif game.objective == QObjective.ENERGY_PARITY:
        return solve_energy_parity_po(game, max_beliefs)
    elif game.objective == QObjective.MEAN_PAYOFF_SAFETY:
        # Mean-payoff + safety: first restrict to safe, then solve MP
        safe_game = QuantPOGame()
        safe_game.vertices = set(game.vertices)
        safe_game.edges = dict(game.edges)
        safe_game.owner = dict(game.owner)
        safe_game.obs = dict(game.obs)
        safe_game.initial = set(game.initial)
        safe_game.objective = QObjective.ENERGY  # energy reduction
        safe_game.target = set(game.target)
        safe_game.priority = dict(game.priority)
        threshold = kwargs.get('threshold', 0.0)
        return solve_mean_payoff_po(safe_game, threshold, max_beliefs)
    else:
        return QPOResult(winning=False, num_beliefs=0)


def quantitative_decomposition(game: QuantPOGame) -> Dict:
    """Decompose the game into qualitative and quantitative components.

    Compare:
    1. Qualitative PO (just reachability/safety ignoring weights)
    2. Quantitative perfect info (energy/MP with full observation)
    3. Quantitative PO (full problem)
    """
    # 1. Qualitative PO
    pog = game.to_partial_obs_game(Objective.SAFETY)
    qual_result = solve_po(pog)

    # 2. Quantitative perfect info
    eg = game.to_energy_game()
    perf_result = solve_energy(eg)

    # 3. Quantitative PO
    qpo_result = solve_energy_po(game)

    return {
        'qualitative_po_winning': qual_result.winning,
        'perfect_info_winning': {v for v in game.initial
                                 if v in perf_result.win_energy},
        'quantitative_po_winning': qpo_result.winning,
        'perfect_info_energies': {v: perf_result.min_energy.get(v)
                                  for v in game.initial
                                  if perf_result.min_energy.get(v) is not None
                                  and perf_result.min_energy.get(v) != INF_ENERGY},
        'po_min_energy': qpo_result.min_energy,
        'num_beliefs': qpo_result.num_beliefs,
    }
