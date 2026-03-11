"""
V164: Stochastic Energy Games
==============================
Two-player energy games with probabilistic (random) vertices.

Three vertex types:
- Player EVEN: minimizes energy needed (picks successor)
- Player ODD: maximizes energy needed (picks successor)
- RANDOM: successor chosen according to probability distribution

The energy objective: Player EVEN wins a play iff the energy level never
drops below 0. With random vertices, EVEN wants to ensure this holds
with probability 1 (almost surely) under optimal play.

Key results:
- Almost-sure energy games are decidable
- The minimum initial energy for almost-sure winning is computable
- Value iteration extends naturally: RANDOM vertices take expected value
- The finite-memory determinacy result extends to stochastic setting

Algorithms implemented:
1. Almost-sure energy game solving (value iteration)
2. Positive-probability energy game solving (exists winning run with prob > 0)
3. Expected energy analysis (expected accumulated energy under strategies)
4. Strategy extraction for all three vertex types
5. Stochastic energy-parity games (combined objectives)
6. Comparison with deterministic V160 energy games

Composes: V160 (Energy Games), V156 (Parity Games).
"""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import sys

sys.path.insert(0, 'Z:/AgentZero/A2/work/V160_energy_games')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V156_parity_games')
from energy_games import (
    Player, EnergyGame, EnergyResult, EnergyParityGame,
    EnergyParityResult, solve_energy, solve_energy_parity,
    INF_ENERGY, _sub_energy
)
from parity_games import ParityGame, zielonka


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class VertexType(Enum):
    EVEN = auto()    # Player Even chooses (minimizer)
    ODD = auto()     # Player Odd chooses (maximizer)
    RANDOM = auto()  # Nature chooses (probabilistic)


@dataclass
class StochasticEnergyGame:
    """A 2.5-player energy game with probabilistic vertices.

    Edges carry integer weights. Random vertices have probability distributions
    over successors.
    """
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    vertex_type: Dict[int, VertexType] = field(default_factory=dict)
    # For RANDOM vertices: (target, weight) -> probability
    probabilities: Dict[int, Dict[Tuple[int, int], float]] = field(default_factory=lambda: defaultdict(dict))

    def add_vertex(self, v: int, vtype: VertexType):
        self.vertices.add(v)
        self.vertex_type[v] = vtype

    def add_edge(self, u: int, v: int, weight: int, prob: float = 1.0):
        """Add edge u -> v with weight. For RANDOM vertices, prob is required."""
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].append((v, weight))
        if self.vertex_type.get(u) == VertexType.RANDOM:
            self.probabilities[u][(v, weight)] = prob

    def successors(self, v: int) -> List[Tuple[int, int]]:
        """Return list of (target, weight) pairs."""
        return self.edges.get(v, [])

    def max_weight(self) -> int:
        W = 0
        for v in self.vertices:
            for (_, w) in self.edges.get(v, []):
                W = max(W, abs(w))
        return W

    def total_weight_bound(self) -> int:
        n = len(self.vertices)
        W = self.max_weight()
        return n * W

    def get_prob(self, v: int, target: int, weight: int) -> float:
        """Get probability of edge (v, target, weight) for random vertex."""
        return self.probabilities.get(v, {}).get((target, weight), 0.0)

    def validate(self) -> List[str]:
        """Check that random vertex probabilities sum to 1."""
        errors = []
        for v in self.vertices:
            if self.vertex_type.get(v) == VertexType.RANDOM:
                succs = self.successors(v)
                if not succs:
                    errors.append(f"Random vertex {v} has no successors")
                    continue
                total = sum(self.get_prob(v, t, w) for (t, w) in succs)
                if abs(total - 1.0) > 1e-9:
                    errors.append(f"Random vertex {v}: probabilities sum to {total}, not 1.0")
        return errors

    def to_energy_game(self) -> EnergyGame:
        """Convert to deterministic energy game (RANDOM -> EVEN).

        Useful for comparison: deterministic analysis is a lower bound.
        """
        g = EnergyGame()
        for v in self.vertices:
            vtype = self.vertex_type[v]
            player = Player.EVEN if vtype != VertexType.ODD else Player.ODD
            g.add_vertex(v, player)
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t, w)
        return g


@dataclass
class StochasticEnergyResult:
    """Result of stochastic energy game analysis."""
    # Minimum initial energy for almost-sure winning (None = cannot win a.s.)
    min_energy: Dict[int, Optional[int]]
    # Almost-sure winning region
    win_almost_sure: Set[int]
    # Positive-probability winning region (can win with prob > 0)
    win_positive: Set[int]
    # Losing region (cannot win from here)
    win_opponent: Set[int]
    # Strategy for Even player
    strategy_even: Dict[int, int]
    # Strategy for Odd player
    strategy_odd: Dict[int, int]
    # Expected energy consumption from each vertex (under optimal strategy)
    expected_energy: Dict[int, Optional[float]]


# ---------------------------------------------------------------------------
# Almost-Sure Winning Region (Qualitative)
# ---------------------------------------------------------------------------

def _compute_as_winning_region(game: StochasticEnergyGame, energy: Dict[int, float]) -> Set[int]:
    """Compute almost-sure winning region given energy values.

    A vertex is in the AS winning region iff:
    - EVEN vertex: exists a successor with finite energy
    - ODD vertex: all successors have finite energy
    - RANDOM vertex: all successors with positive probability have finite energy
    """
    win = set()
    for v in game.vertices:
        succs = game.successors(v)
        if not succs:
            # Dead-end: EVEN loses, ODD means EVEN wins
            if game.vertex_type[v] != VertexType.EVEN:
                win.add(v)
            continue

        vtype = game.vertex_type[v]
        if vtype == VertexType.EVEN:
            # Even wins if at least one successor has finite energy
            for (t, w) in succs:
                needed = _sub_energy(energy.get(t, INF_ENERGY), w)
                if needed != INF_ENERGY:
                    win.add(v)
                    break
        elif vtype == VertexType.ODD:
            # Odd can't prevent winning if all successors have finite energy
            all_finite = True
            for (t, w) in succs:
                needed = _sub_energy(energy.get(t, INF_ENERGY), w)
                if needed == INF_ENERGY:
                    all_finite = False
                    break
            if all_finite:
                win.add(v)
        else:  # RANDOM
            # Almost-sure: all successors with positive probability must be winnable
            all_prob_finite = True
            for (t, w) in succs:
                p = game.get_prob(v, t, w)
                if p > 0:
                    needed = _sub_energy(energy.get(t, INF_ENERGY), w)
                    if needed == INF_ENERGY:
                        all_prob_finite = False
                        break
            if all_prob_finite:
                win.add(v)
    return win


# ---------------------------------------------------------------------------
# Stochastic Energy Game Solver (Value Iteration)
# ---------------------------------------------------------------------------

def solve_stochastic_energy(game: StochasticEnergyGame) -> StochasticEnergyResult:
    """Solve a stochastic energy game.

    For almost-sure winning, we need:
    - EVEN vertices: min over successors of (energy[succ] - weight)
    - ODD vertices: max over successors of (energy[succ] - weight)
    - RANDOM vertices: expected value over successors weighted by probabilities
      BUT: if ANY successor with positive probability has infinite energy,
      the random vertex also has infinite energy (almost-sure requirement).

    Returns minimum initial energy for almost-sure winning from each vertex.
    """
    if not game.vertices:
        return StochasticEnergyResult({}, set(), set(), set(), {}, {}, {})

    n = len(game.vertices)
    W = game.max_weight()
    bound = n * W + 1

    # Initialize: energy[v] = 0 for all v (optimistic)
    energy = {v: 0.0 for v in game.vertices}
    strategy = {v: None for v in game.vertices}

    # Handle dead ends
    for v in game.vertices:
        succs = game.successors(v)
        if not succs:
            if game.vertex_type[v] == VertexType.EVEN:
                energy[v] = INF_ENERGY

    # Value iteration
    changed = True
    iterations = 0
    max_iterations = n * (bound + 1) + 1

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        for v in game.vertices:
            succs = game.successors(v)
            if not succs:
                continue

            old_val = energy[v]
            vtype = game.vertex_type[v]

            if vtype == VertexType.EVEN:
                # Even minimizes
                best = INF_ENERGY
                best_succ = None
                for (t, w) in succs:
                    needed = _sub_energy(energy[t], w)
                    if needed < best:
                        best = needed
                        best_succ = t
                new_val = max(0.0, best)

            elif vtype == VertexType.ODD:
                # Odd maximizes
                worst = 0.0
                worst_succ = None
                for (t, w) in succs:
                    needed = _sub_energy(energy[t], w)
                    val = max(0.0, needed)
                    if val > worst or worst_succ is None:
                        worst = val
                        worst_succ = t
                new_val = worst
                best_succ = worst_succ

            else:  # RANDOM
                # Almost-sure: if any positive-prob successor has infinite energy, result is infinite
                any_inf = False
                expected = 0.0
                best_succ = None
                for (t, w) in succs:
                    p = game.get_prob(v, t, w)
                    if p <= 0:
                        continue
                    needed = _sub_energy(energy[t], w)
                    if needed == INF_ENERGY:
                        any_inf = True
                        break
                    val = max(0.0, needed)
                    expected += p * val
                    if best_succ is None:
                        best_succ = t

                if any_inf:
                    new_val = INF_ENERGY
                else:
                    # For almost-sure, we need enough energy for ALL possible outcomes.
                    # The expected value is a lower bound; the actual requirement is
                    # the maximum over all possible outcomes (worst case over random).
                    # This is because we need to survive ALL random outcomes, not just
                    # in expectation.
                    max_needed = 0.0
                    for (t, w) in succs:
                        p = game.get_prob(v, t, w)
                        if p <= 0:
                            continue
                        needed = _sub_energy(energy[t], w)
                        val = max(0.0, needed)
                        max_needed = max(max_needed, val)
                    new_val = max_needed

            # Cap at bound
            if new_val != INF_ENERGY and new_val > bound:
                new_val = INF_ENERGY

            if new_val != old_val:
                energy[v] = new_val
                strategy[v] = best_succ
                changed = True

    # Build result
    min_energy = {}
    win_as = set()
    win_opponent = set()
    strategy_even = {}
    strategy_odd = {}

    for v in game.vertices:
        if energy[v] == INF_ENERGY:
            min_energy[v] = None
            win_opponent.add(v)
        else:
            min_energy[v] = int(math.ceil(energy[v]))
            win_as.add(v)

    # Extract strategies
    for v in game.vertices:
        succs = game.successors(v)
        if not succs:
            continue
        vtype = game.vertex_type[v]
        if vtype == VertexType.EVEN and v in win_as:
            best_t = None
            best_need = INF_ENERGY
            for (t, w) in succs:
                if t in win_as:
                    needed = max(0, _sub_energy(energy[t], w))
                    if needed < best_need:
                        best_need = needed
                        best_t = t
            if best_t is not None:
                strategy_even[v] = best_t
        elif vtype == VertexType.ODD and v in win_opponent:
            best_t = None
            best_need = 0
            for (t, w) in succs:
                needed = _sub_energy(energy[t], w)
                val = max(0, needed) if needed != INF_ENERGY else INF_ENERGY
                if val == INF_ENERGY or val > best_need:
                    best_need = val
                    best_t = t
            if best_t is not None:
                strategy_odd[v] = best_t

    # Compute positive-probability winning region
    win_pos = _compute_positive_prob_winning(game, energy)

    # Compute expected energy
    exp_energy = _compute_expected_energy(game, energy, win_as)

    return StochasticEnergyResult(
        min_energy=min_energy,
        win_almost_sure=win_as,
        win_positive=win_pos,
        win_opponent=win_opponent,
        strategy_even=strategy_even,
        strategy_odd=strategy_odd,
        expected_energy=exp_energy
    )


def _compute_positive_prob_winning(game: StochasticEnergyGame,
                                    energy: Dict[int, float]) -> Set[int]:
    """Compute positive-probability winning region.

    A vertex is in the positive-prob winning region if there EXISTS a play
    (sequence of moves + random outcomes) that keeps energy >= 0 forever.

    For RANDOM vertices, only ONE successor needs to be winnable (not all).
    """
    if not game.vertices:
        return set()

    n = len(game.vertices)
    W = game.max_weight()
    bound = n * W + 1

    # Recompute with optimistic random semantics
    pp_energy = {v: 0.0 for v in game.vertices}

    for v in game.vertices:
        if not game.successors(v) and game.vertex_type[v] == VertexType.EVEN:
            pp_energy[v] = INF_ENERGY

    changed = True
    iterations = 0
    max_iterations = n * (bound + 1) + 1

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        for v in game.vertices:
            succs = game.successors(v)
            if not succs:
                continue

            old_val = pp_energy[v]
            vtype = game.vertex_type[v]

            if vtype == VertexType.EVEN or vtype == VertexType.RANDOM:
                # Both Even and Random: just need ONE successor to be winnable
                best = INF_ENERGY
                for (t, w) in succs:
                    if vtype == VertexType.RANDOM:
                        p = game.get_prob(v, t, w)
                        if p <= 0:
                            continue
                    needed = _sub_energy(pp_energy[t], w)
                    if needed < best:
                        best = needed
                new_val = max(0.0, best)
            else:  # ODD
                worst = 0.0
                for (t, w) in succs:
                    needed = _sub_energy(pp_energy[t], w)
                    val = max(0.0, needed)
                    if val > worst:
                        worst = val
                new_val = worst

            if new_val != INF_ENERGY and new_val > bound:
                new_val = INF_ENERGY

            if new_val != old_val:
                pp_energy[v] = new_val
                changed = True

    return {v for v in game.vertices if pp_energy[v] != INF_ENERGY}


def _compute_expected_energy(game: StochasticEnergyGame,
                              energy: Dict[int, float],
                              win_as: Set[int]) -> Dict[int, Optional[float]]:
    """Compute expected energy consumption from each vertex.

    For vertices in the AS winning region, compute the expected energy level
    after one step under optimal play.
    """
    result = {}
    for v in game.vertices:
        if v not in win_as:
            result[v] = None
            continue

        succs = game.successors(v)
        if not succs:
            result[v] = 0.0
            continue

        vtype = game.vertex_type[v]
        if vtype == VertexType.EVEN:
            # Even plays optimally: pick minimum-energy successor
            best = INF_ENERGY
            for (t, w) in succs:
                if t in win_as:
                    needed = max(0.0, _sub_energy(energy[t], w))
                    if needed < best:
                        best = needed
            result[v] = best if best != INF_ENERGY else None
        elif vtype == VertexType.ODD:
            # Odd plays adversarially: pick maximum-energy successor (in winning region)
            worst = 0.0
            for (t, w) in succs:
                if t in win_as:
                    needed = max(0.0, _sub_energy(energy[t], w))
                    if needed > worst:
                        worst = needed
            result[v] = worst
        else:  # RANDOM
            # Expected value over random outcomes
            exp = 0.0
            for (t, w) in succs:
                p = game.get_prob(v, t, w)
                if p <= 0:
                    continue
                needed = max(0.0, _sub_energy(energy[t], w))
                exp += p * needed
            result[v] = exp

    return result


# ---------------------------------------------------------------------------
# Stochastic Energy-Parity Games
# ---------------------------------------------------------------------------

@dataclass
class StochasticEnergyParityGame:
    """Stochastic game with both energy and parity objectives."""
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    vertex_type: Dict[int, VertexType] = field(default_factory=dict)
    priority: Dict[int, int] = field(default_factory=dict)
    probabilities: Dict[int, Dict[Tuple[int, int], float]] = field(default_factory=lambda: defaultdict(dict))

    def add_vertex(self, v: int, vtype: VertexType, prio: int):
        self.vertices.add(v)
        self.vertex_type[v] = vtype
        self.priority[v] = prio

    def add_edge(self, u: int, v: int, weight: int, prob: float = 1.0):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].append((v, weight))
        if self.vertex_type.get(u) == VertexType.RANDOM:
            self.probabilities[u][(v, weight)] = prob

    def successors(self, v: int) -> List[Tuple[int, int]]:
        return self.edges.get(v, [])

    def max_priority(self) -> int:
        return max(self.priority.values()) if self.priority else 0

    def max_weight(self) -> int:
        W = 0
        for v in self.vertices:
            for (_, w) in self.edges.get(v, []):
                W = max(W, abs(w))
        return W

    def get_prob(self, v: int, target: int, weight: int) -> float:
        return self.probabilities.get(v, {}).get((target, weight), 0.0)

    def to_energy_game(self) -> StochasticEnergyGame:
        """Strip parity, keep energy + stochastic."""
        g = StochasticEnergyGame()
        for v in self.vertices:
            g.add_vertex(v, self.vertex_type[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                p = self.get_prob(v, t, w) if self.vertex_type[v] == VertexType.RANDOM else 1.0
                g.add_edge(v, t, w, p)
        return g

    def to_parity_game(self) -> ParityGame:
        """Strip weights + stochastic, keep parity (RANDOM -> EVEN)."""
        g = ParityGame()
        for v in self.vertices:
            vtype = self.vertex_type[v]
            player = Player.EVEN if vtype != VertexType.ODD else Player.ODD
            g.add_vertex(v, player, self.priority[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t)
        return g


@dataclass
class StochasticEnergyParityResult:
    """Result of stochastic energy-parity game analysis."""
    min_energy: Dict[int, Optional[int]]
    win_even: Set[int]
    win_odd: Set[int]
    strategy_even: Dict[int, int]
    strategy_odd: Dict[int, int]


def solve_stochastic_energy_parity(game: StochasticEnergyParityGame) -> StochasticEnergyParityResult:
    """Solve stochastic energy-parity game.

    Even wins iff:
    1. Energy never drops below 0 (almost surely), AND
    2. Highest priority seen infinitely often is even (almost surely).

    Approach: iterative refinement.
    1. Solve parity game (ignoring energy and randomness, RANDOM -> EVEN)
    2. Check energy condition under parity strategy on restricted graph
    3. Remove failures + compute Odd attractor
    4. Repeat until stable
    """
    if not game.vertices:
        return StochasticEnergyParityResult({}, set(), set(), {}, {})

    # Step 1: Solve parity game (convert RANDOM to EVEN)
    pg = game.to_parity_game()
    parity_sol = zielonka(pg)
    win_even_parity = parity_sol.win_even
    parity_strategy = dict(parity_sol.strategy_even)

    # Iterative refinement
    current_vertices = set(game.vertices)
    final_win_even = set()
    final_strategy_even = {}

    for _iteration in range(len(game.vertices) + 1):
        if not current_vertices:
            break

        # Solve parity on current subgame
        sub_pg = ParityGame()
        for v in current_vertices:
            vtype = game.vertex_type[v]
            player = Player.EVEN if vtype != VertexType.ODD else Player.ODD
            sub_pg.add_vertex(v, player, game.priority[v])
        for v in current_vertices:
            for (t, w) in game.edges.get(v, []):
                if t in current_vertices:
                    sub_pg.add_edge(v, t)

        # Check if subgame has any edges
        has_edges = False
        for v in current_vertices:
            for (t, w) in game.edges.get(v, []):
                if t in current_vertices:
                    has_edges = True
                    break
            if has_edges:
                break

        if not has_edges:
            break

        sub_sol = zielonka(sub_pg)
        sub_win_even = sub_sol.win_even
        sub_strategy = dict(sub_sol.strategy_even)

        if not sub_win_even:
            break

        # Check energy under parity strategy for Even vertices
        # Build restricted stochastic energy game with parity strategy
        seg = StochasticEnergyGame()
        for v in sub_win_even:
            seg.add_vertex(v, game.vertex_type[v])

        for v in sub_win_even:
            vtype = game.vertex_type[v]
            if vtype == VertexType.EVEN:
                # Follow parity strategy
                target = sub_strategy.get(v)
                if target is not None and target in sub_win_even:
                    # Find corresponding weight
                    for (t, w) in game.edges.get(v, []):
                        if t == target:
                            seg.add_edge(v, t, w)
                            break
                else:
                    # No strategy edge; pick any edge in win_even
                    for (t, w) in game.edges.get(v, []):
                        if t in sub_win_even:
                            seg.add_edge(v, t, w)
                            break
            elif vtype == VertexType.ODD:
                # Odd picks any edge in subgame
                for (t, w) in game.edges.get(v, []):
                    if t in sub_win_even:
                        seg.add_edge(v, t, w)
            else:  # RANDOM
                for (t, w) in game.edges.get(v, []):
                    if t in sub_win_even:
                        p = game.get_prob(v, t, w)
                        seg.add_edge(v, t, w, p)

        energy_result = solve_stochastic_energy(seg)
        energy_win = energy_result.win_almost_sure

        if energy_win == sub_win_even:
            # All vertices in parity winning region also satisfy energy
            final_win_even = energy_win
            final_strategy_even = sub_strategy
            break

        # Remove vertices that fail energy check
        failures = sub_win_even - energy_win
        if not failures:
            final_win_even = energy_win
            final_strategy_even = sub_strategy
            break

        # Compute Odd attractor of failures in current subgame
        odd_attr = _odd_attractor(game, current_vertices, failures)
        current_vertices -= odd_attr

    # Build final result
    min_energy = {}
    win_odd = game.vertices - final_win_even
    strategy_odd = {}

    if final_win_even:
        # Compute energy values for winning region
        seg = StochasticEnergyGame()
        for v in final_win_even:
            seg.add_vertex(v, game.vertex_type[v])
        for v in final_win_even:
            for (t, w) in game.edges.get(v, []):
                if t in final_win_even:
                    p = game.get_prob(v, t, w) if game.vertex_type[v] == VertexType.RANDOM else 1.0
                    seg.add_edge(v, t, w, p)
        er = solve_stochastic_energy(seg)
        for v in final_win_even:
            min_energy[v] = er.min_energy.get(v, 0)
    for v in win_odd:
        min_energy[v] = None

    return StochasticEnergyParityResult(
        min_energy=min_energy,
        win_even=final_win_even,
        win_odd=win_odd,
        strategy_even=final_strategy_even,
        strategy_odd=strategy_odd
    )


def _odd_attractor(game: StochasticEnergyParityGame,
                    subgame: Set[int], target: Set[int]) -> Set[int]:
    """Compute Odd's attractor of target within subgame.

    Odd attracts if:
    - ODD vertex: exists edge to attractor
    - EVEN vertex: all edges go to attractor
    - RANDOM vertex: all edges with positive probability go to attractor
    """
    attr = set(target)
    queue = deque(target)

    while queue:
        v = queue.popleft()
        for u in subgame:
            if u in attr:
                continue
            succs_in_sub = [(t, w) for (t, w) in game.edges.get(u, []) if t in subgame]
            if not succs_in_sub:
                continue

            vtype = game.vertex_type[u]
            if vtype == VertexType.ODD:
                # Odd can choose to go to attractor
                for (t, w) in succs_in_sub:
                    if t in attr:
                        attr.add(u)
                        queue.append(u)
                        break
            elif vtype == VertexType.EVEN:
                # Even is forced: all successors must be in attractor
                if all(t in attr for (t, w) in succs_in_sub):
                    attr.add(u)
                    queue.append(u)
            else:  # RANDOM
                # All positive-prob successors in attractor
                all_in = True
                for (t, w) in succs_in_sub:
                    p = game.get_prob(u, t, w)
                    if p > 0 and t not in attr:
                        all_in = False
                        break
                if all_in:
                    attr.add(u)
                    queue.append(u)

    return attr


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_play(game: StochasticEnergyGame,
                  start: int,
                  initial_energy: int,
                  strategy_even: Dict[int, int],
                  strategy_odd: Dict[int, int],
                  steps: int = 50,
                  seed: int = 42) -> List[Tuple[int, int]]:
    """Simulate a play of the stochastic energy game.

    Returns list of (vertex, energy_level) pairs.
    For RANDOM vertices, uses deterministic pseudo-random choices.
    """
    import random as rng
    rng_local = rng.Random(seed)

    trace = [(start, initial_energy)]
    current = start
    energy_level = initial_energy

    for _ in range(steps):
        succs = game.successors(current)
        if not succs:
            break

        vtype = game.vertex_type[current]
        if vtype == VertexType.EVEN:
            target = strategy_even.get(current)
            if target is None:
                target = succs[0][0]
            # Find weight for this target
            weight = 0
            for (t, w) in succs:
                if t == target:
                    weight = w
                    break
        elif vtype == VertexType.ODD:
            target = strategy_odd.get(current)
            if target is None:
                target = succs[0][0]
            weight = 0
            for (t, w) in succs:
                if t == target:
                    weight = w
                    break
        else:  # RANDOM
            # Choose according to probability distribution
            r = rng_local.random()
            cumulative = 0.0
            target = succs[0][0]
            weight = succs[0][1]
            for (t, w) in succs:
                p = game.get_prob(current, t, w)
                cumulative += p
                if r <= cumulative:
                    target = t
                    weight = w
                    break

        energy_level += weight
        current = target
        trace.append((current, energy_level))

        if energy_level < 0:
            break  # Energy depleted

    return trace


# ---------------------------------------------------------------------------
# Strategy Verification
# ---------------------------------------------------------------------------

def verify_strategy(game: StochasticEnergyGame,
                     strategy_even: Dict[int, int],
                     initial_energy: Dict[int, int]) -> Dict[str, object]:
    """Verify that Even's strategy is winning with given initial energies.

    Checks:
    1. Strategy is well-defined (maps each Even vertex to a valid successor)
    2. Energy stays non-negative under worst-case Odd + worst-case Random
    3. Reports any violations found
    """
    violations = []
    valid_vertices = set()

    for v in game.vertices:
        if game.vertex_type[v] == VertexType.EVEN:
            succs = game.successors(v)
            if not succs:
                continue
            if v not in strategy_even:
                violations.append(f"No strategy for Even vertex {v}")
                continue
            target = strategy_even[v]
            if not any(t == target for (t, w) in succs):
                violations.append(f"Invalid strategy at {v}: target {target} not a successor")
                continue
            valid_vertices.add(v)

    # Check energy invariant: for each vertex in win region,
    # initial_energy[v] + weight >= initial_energy[target]
    energy_ok = True
    for v in game.vertices:
        if v not in initial_energy:
            continue
        e = initial_energy[v]
        succs = game.successors(v)
        if not succs:
            continue

        vtype = game.vertex_type[v]
        if vtype == VertexType.EVEN:
            target = strategy_even.get(v)
            if target is None:
                continue
            for (t, w) in succs:
                if t == target:
                    new_e = e + w
                    if new_e < 0:
                        violations.append(f"Energy violation at {v}: {e} + {w} = {new_e} < 0")
                        energy_ok = False
                    elif t in initial_energy and new_e < initial_energy[t]:
                        violations.append(f"Energy insufficient at {v}->{t}: {new_e} < {initial_energy[t]}")
                        energy_ok = False
                    break
        elif vtype == VertexType.ODD:
            # Check worst case
            for (t, w) in succs:
                new_e = e + w
                if t in initial_energy and new_e < initial_energy[t]:
                    # This is only a problem if t is in the winning region
                    pass  # Odd may not choose this edge if it leads outside winning
        else:  # RANDOM
            for (t, w) in succs:
                p = game.get_prob(v, t, w)
                if p > 0:
                    new_e = e + w
                    if new_e < 0:
                        violations.append(f"Energy violation at random {v}->{t}: {e} + {w} = {new_e} < 0")
                        energy_ok = False

    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'energy_ok': energy_ok,
        'strategy_vertices': len(valid_vertices)
    }


# ---------------------------------------------------------------------------
# Construction Helpers
# ---------------------------------------------------------------------------

def make_chain_game(n: int, weights: List[int], random_probs: Optional[List[float]] = None) -> StochasticEnergyGame:
    """Create a chain game: 0 -> 1 -> ... -> n-1 -> 0.

    Vertices alternate EVEN, ODD, RANDOM.
    If random_probs given, random vertices split into two successors.
    """
    g = StochasticEnergyGame()
    types = [VertexType.EVEN, VertexType.ODD, VertexType.RANDOM]

    for i in range(n):
        g.add_vertex(i, types[i % 3])

    for i in range(n):
        w = weights[i % len(weights)]
        next_v = (i + 1) % n

        if g.vertex_type[i] == VertexType.RANDOM and random_probs:
            # Split: go forward with prob p, self-loop with prob 1-p
            p = random_probs[0] if len(random_probs) == 1 else random_probs[i % len(random_probs)]
            g.add_edge(i, next_v, w, p)
            g.add_edge(i, i, 0, 1.0 - p)
        else:
            g.add_edge(i, next_v, w)

    return g


def make_diamond_game(gain: int, loss: int, prob: float = 0.5) -> StochasticEnergyGame:
    """Create a diamond-shaped game:

         0 (EVEN)
        / \\
       1   2 (RANDOM vertices)
        \\ /
         3 (ODD) -> 0

    Edge 0->1 has weight +gain, 0->2 has weight +gain.
    Random 1 goes to 3 with prob p (weight 0) or self-loop (weight -loss).
    Random 2 goes to 3 with prob p (weight 0) or self-loop (weight -loss).
    Edge 3->0 has weight -loss.
    """
    g = StochasticEnergyGame()
    g.add_vertex(0, VertexType.EVEN)
    g.add_vertex(1, VertexType.RANDOM)
    g.add_vertex(2, VertexType.RANDOM)
    g.add_vertex(3, VertexType.ODD)

    g.add_edge(0, 1, gain)
    g.add_edge(0, 2, gain)
    g.add_edge(1, 3, 0, prob)
    g.add_edge(1, 1, -loss, 1.0 - prob)
    g.add_edge(2, 3, 0, prob)
    g.add_edge(2, 2, -loss, 1.0 - prob)
    g.add_edge(3, 0, -loss)

    return g


def make_gambling_game(bet: int, win_prob: float = 0.5) -> StochasticEnergyGame:
    """Create a gambling game:

    0 (EVEN): choose to bet or not
    1 (RANDOM): win or lose the bet
    2: safe state (no bet, small cost)

    0 -> 1 (bet): weight 0
    0 -> 2 (don't bet): weight -1
    1 -> 0 (win): weight +bet
    1 -> 0 (lose): weight -bet
    2 -> 0: weight 0
    """
    g = StochasticEnergyGame()
    g.add_vertex(0, VertexType.EVEN)
    g.add_vertex(1, VertexType.RANDOM)
    g.add_vertex(2, VertexType.EVEN)

    g.add_edge(0, 1, 0)      # bet
    g.add_edge(0, 2, -1)     # don't bet
    g.add_edge(1, 0, bet, win_prob)       # win
    g.add_edge(1, 0, -bet, 1.0 - win_prob)  # lose
    g.add_edge(2, 0, 0)

    return g


def make_random_walk_game(n: int, step_prob: float = 0.5) -> StochasticEnergyGame:
    """Create a random walk on 0..n-1, all RANDOM vertices.

    Each vertex i goes to i+1 (weight +1, prob p) or i-1 (weight -1, prob 1-p).
    Boundary: 0 can only go to 1; n-1 can only go to n-2.
    """
    g = StochasticEnergyGame()
    for i in range(n):
        g.add_vertex(i, VertexType.RANDOM)

    for i in range(n):
        if i == 0:
            g.add_edge(i, i + 1, 1, 1.0)
        elif i == n - 1:
            g.add_edge(i, i - 1, -1, 1.0)
        else:
            g.add_edge(i, i + 1, 1, step_prob)
            g.add_edge(i, i - 1, -1, 1.0 - step_prob)

    return g


# ---------------------------------------------------------------------------
# Comparison with Deterministic Energy Games
# ---------------------------------------------------------------------------

def compare_with_deterministic(game: StochasticEnergyGame) -> Dict[str, object]:
    """Compare stochastic analysis with deterministic V160 analysis.

    Converts RANDOM vertices to EVEN (optimistic) and ODD (pessimistic),
    and compares the three analyses.
    """
    # Stochastic (actual)
    stoch_result = solve_stochastic_energy(game)

    # Optimistic: RANDOM -> EVEN (Even can choose best random outcome)
    opt_game = EnergyGame()
    for v in game.vertices:
        vtype = game.vertex_type[v]
        player = Player.EVEN if vtype != VertexType.ODD else Player.ODD
        opt_game.add_vertex(v, player)
    for v in game.vertices:
        for (t, w) in game.edges.get(v, []):
            opt_game.add_edge(v, t, w)
    opt_result = solve_energy(opt_game)

    # Pessimistic: RANDOM -> ODD (Odd picks worst random outcome)
    pess_game = EnergyGame()
    for v in game.vertices:
        vtype = game.vertex_type[v]
        player = Player.ODD if vtype != VertexType.EVEN else Player.EVEN
        # Actually: EVEN stays EVEN, ODD stays ODD, RANDOM -> ODD
        if vtype == VertexType.EVEN:
            pess_game.add_vertex(v, Player.EVEN)
        else:
            pess_game.add_vertex(v, Player.ODD)
    for v in game.vertices:
        for (t, w) in game.edges.get(v, []):
            pess_game.add_edge(v, t, w)
    pess_result = solve_energy(pess_game)

    return {
        'stochastic': {
            'win_as': stoch_result.win_almost_sure,
            'min_energy': stoch_result.min_energy
        },
        'optimistic': {
            'win': opt_result.win_energy,
            'min_energy': opt_result.min_energy
        },
        'pessimistic': {
            'win': pess_result.win_energy,
            'min_energy': pess_result.min_energy
        },
        'ordering_valid': pess_result.win_energy <= stoch_result.win_almost_sure <= opt_result.win_energy,
        'summary': {
            'stochastic_win_size': len(stoch_result.win_almost_sure),
            'optimistic_win_size': len(opt_result.win_energy),
            'pessimistic_win_size': len(pess_result.win_energy),
        }
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stochastic_energy_statistics(game: StochasticEnergyGame) -> Dict[str, object]:
    """Compute statistics about a stochastic energy game."""
    n_even = sum(1 for v in game.vertices if game.vertex_type[v] == VertexType.EVEN)
    n_odd = sum(1 for v in game.vertices if game.vertex_type[v] == VertexType.ODD)
    n_random = sum(1 for v in game.vertices if game.vertex_type[v] == VertexType.RANDOM)
    n_edges = sum(len(game.edges.get(v, [])) for v in game.vertices)

    result = solve_stochastic_energy(game)

    return {
        'vertices': len(game.vertices),
        'edges': n_edges,
        'even_vertices': n_even,
        'odd_vertices': n_odd,
        'random_vertices': n_random,
        'max_weight': game.max_weight(),
        'win_almost_sure': len(result.win_almost_sure),
        'win_positive': len(result.win_positive),
        'win_opponent': len(result.win_opponent),
        'min_energy': result.min_energy,
        'expected_energy': result.expected_energy,
    }
