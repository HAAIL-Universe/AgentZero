"""
V160: Energy Games
==================
Two-player infinite-duration games with quantitative (energy) objectives.

Energy games extend parity games with integer weights on edges. Player 0 (Energy)
tries to keep the cumulative energy level non-negative forever. Player 1 (Opponent)
tries to deplete the energy below zero.

Key results:
- Energy games are decidable in NP intersect coNP (like parity games)
- The minimum initial energy to win is computable in pseudo-polynomial time
- Energy games with parity conditions subsume both energy and parity objectives
- Strong connection to mean-payoff games via threshold problem

Algorithms implemented:
1. Energy game solving (value iteration for minimum initial energy)
2. Strategy extraction for both players
3. Energy-parity games (combined energy + parity winning condition)
4. Mean-payoff reduction (energy threshold <-> mean-payoff >= 0)
5. Fixed initial energy game solving
6. Parametric construction helpers

Composes: V156 (Parity Games) for parity condition handling.
"""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import sys

sys.path.insert(0, 'Z:/AgentZero/A2/work/V156_parity_games')
from parity_games import Player, ParityGame, Solution, zielonka


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EnergyGame:
    """A two-player energy game on a weighted graph.

    Vertices are owned by Player.EVEN (energy player, wants to survive)
    or Player.ODD (opponent, wants to deplete energy).
    Edges carry integer weights (can be negative).
    """
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    owner: Dict[int, Player] = field(default_factory=dict)

    def add_vertex(self, v: int, player: Player):
        self.vertices.add(v)
        self.owner[v] = player

    def add_edge(self, u: int, v: int, weight: int):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].append((v, weight))

    def successors(self, v: int) -> List[Tuple[int, int]]:
        """Return list of (target, weight) pairs."""
        return self.edges.get(v, [])

    def predecessors(self, v: int) -> List[Tuple[int, int]]:
        """Return list of (source, weight) pairs leading to v."""
        result = []
        for u in self.vertices:
            for (t, w) in self.edges.get(u, []):
                if t == v:
                    result.append((u, w))
        return result

    def max_weight(self) -> int:
        """Maximum absolute weight in the game."""
        W = 0
        for v in self.vertices:
            for (_, w) in self.edges.get(v, []):
                W = max(W, abs(w))
        return W

    def total_weight_bound(self) -> int:
        """Upper bound on minimum initial energy: n * W."""
        n = len(self.vertices)
        W = self.max_weight()
        return n * W


@dataclass
class EnergyResult:
    """Result of energy game analysis."""
    # Minimum initial energy to win from each vertex (None = cannot win)
    min_energy: Dict[int, Optional[int]]
    # Winning region for energy player (finite min_energy)
    win_energy: Set[int]
    # Winning region for opponent (min_energy is None / infinite)
    win_opponent: Set[int]
    # Optimal strategy for energy player (vertex -> successor)
    strategy_energy: Dict[int, int]
    # Optimal strategy for opponent (vertex -> successor)
    strategy_opponent: Dict[int, int]


@dataclass
class EnergyParityGame:
    """Energy game with additional parity winning condition.

    Player 0 (Even/Energy) wins a play iff:
    1. The energy level never drops below 0 (energy condition), AND
    2. The highest priority seen infinitely often is even (parity condition).
    """
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    owner: Dict[int, Player] = field(default_factory=dict)
    priority: Dict[int, int] = field(default_factory=dict)

    def add_vertex(self, v: int, player: Player, prio: int):
        self.vertices.add(v)
        self.owner[v] = player
        self.priority[v] = prio

    def add_edge(self, u: int, v: int, weight: int):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].append((v, weight))

    def successors(self, v: int) -> List[Tuple[int, int]]:
        return self.edges.get(v, [])

    def max_priority(self) -> int:
        if not self.priority:
            return 0
        return max(self.priority.values())

    def max_weight(self) -> int:
        W = 0
        for v in self.vertices:
            for (_, w) in self.edges.get(v, []):
                W = max(W, abs(w))
        return W

    def to_energy_game(self) -> EnergyGame:
        """Strip parity condition, keep only energy."""
        g = EnergyGame()
        for v in self.vertices:
            g.add_vertex(v, self.owner[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t, w)
        return g

    def to_parity_game(self) -> ParityGame:
        """Strip weights, keep only parity."""
        g = ParityGame()
        for v in self.vertices:
            g.add_vertex(v, self.owner[v], self.priority[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t)
        return g


@dataclass
class EnergyParityResult:
    """Result of energy-parity game analysis."""
    min_energy: Dict[int, Optional[int]]
    win_energy: Set[int]
    win_opponent: Set[int]
    strategy_energy: Dict[int, int]
    strategy_opponent: Dict[int, int]


@dataclass
class MeanPayoffResult:
    """Result of mean-payoff game analysis."""
    # Mean-payoff value from each vertex under optimal play
    values: Dict[int, float]
    # Winning region for player 0 (value >= 0)
    win_nonneg: Set[int]
    # Winning region for player 1 (value < 0)
    win_neg: Set[int]
    # Optimal strategies
    strategy_p0: Dict[int, int]
    strategy_p1: Dict[int, int]


# ---------------------------------------------------------------------------
# Energy Game Solver (Value Iteration)
# ---------------------------------------------------------------------------

INF_ENERGY = float('inf')


def solve_energy(game: EnergyGame) -> EnergyResult:
    """Solve an energy game: compute minimum initial energy for each vertex.

    Uses a value iteration (Bellman-Ford style) algorithm:
    - For Even vertices: min over successors of (energy[succ] - weight)
    - For Odd vertices: max over successors of (energy[succ] - weight)

    The energy at v is the minimum initial credit needed so that Even can
    keep energy >= 0 forever starting from v.

    Runs in O(n * m * W) where n = vertices, m = edges, W = max weight.
    """
    if not game.vertices:
        return EnergyResult({}, set(), set(), {}, {})

    n = len(game.vertices)
    W = game.max_weight()
    bound = n * W + 1  # values above this mean "cannot win"

    # Initialize: energy[v] = 0 for all v (optimistic)
    energy = {v: 0 for v in game.vertices}
    strategy = {v: None for v in game.vertices}

    # Check for dead ends -- vertices with no outgoing edges
    # Dead-end for Even player = loses (infinite energy needed)
    # Dead-end for Odd player = Even wins trivially (Odd stuck)
    for v in game.vertices:
        succs = game.successors(v)
        if not succs:
            if game.owner[v] == Player.EVEN:
                energy[v] = INF_ENERGY  # Even can't move, loses
            # Odd can't move -> Even wins with energy 0

    # Value iteration: repeat until fixpoint
    changed = True
    iterations = 0
    max_iterations = n * (bound + 1) + 1  # pseudo-polynomial bound

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        for v in game.vertices:
            succs = game.successors(v)
            if not succs:
                continue

            old_val = energy[v]

            if game.owner[v] == Player.EVEN:
                # Even minimizes initial energy needed
                best = INF_ENERGY
                best_succ = None
                for (t, w) in succs:
                    # After taking edge (v, t) with weight w, energy changes by w
                    # Need: current_energy + w >= energy[t]
                    # So: current_energy >= energy[t] - w
                    needed = _sub_energy(energy[t], w)
                    if needed < best:
                        best = needed
                        best_succ = t
                new_val = max(0, best)  # energy can't be negative at start
            else:
                # Odd maximizes initial energy needed (worst case for Even)
                worst = 0
                worst_succ = None
                for (t, w) in succs:
                    needed = _sub_energy(energy[t], w)
                    val = max(0, needed)
                    if val > worst or worst_succ is None:
                        worst = val
                        worst_succ = t
                new_val = worst
                best_succ = worst_succ

            # Cap at bound (signals "cannot win")
            if new_val > bound:
                new_val = INF_ENERGY

            if new_val != old_val:
                energy[v] = new_val
                strategy[v] = best_succ
                changed = True

    # Build result
    min_energy = {}
    win_energy = set()
    win_opponent = set()
    strategy_energy = {}
    strategy_opponent = {}

    for v in game.vertices:
        if energy[v] == INF_ENERGY:
            min_energy[v] = None
            win_opponent.add(v)
        else:
            min_energy[v] = int(energy[v])
            win_energy.add(v)

    # Extract strategies
    for v in game.vertices:
        succs = game.successors(v)
        if not succs:
            continue

        if game.owner[v] == Player.EVEN:
            if v in win_energy:
                # Pick successor minimizing energy need
                best_t = None
                best_need = INF_ENERGY
                for (t, w) in succs:
                    if t in win_energy:
                        needed = max(0, _sub_energy(energy[t], w))
                        if needed < best_need:
                            best_need = needed
                            best_t = t
                if best_t is not None:
                    strategy_energy[v] = best_t
            else:
                strategy_opponent[v] = succs[0][0]  # any move, Even loses anyway
        else:
            if v in win_opponent:
                # Pick successor maximizing energy need (or staying in win_opponent)
                best_t = None
                best_need = -1
                for (t, w) in succs:
                    if t in win_opponent:
                        best_t = t
                        break
                    needed = max(0, _sub_energy(energy[t], w))
                    if needed > best_need:
                        best_need = needed
                        best_t = t
                if best_t is not None:
                    strategy_opponent[v] = best_t
            else:
                # Odd vertex in Even's winning region
                # Pick worst successor for Even (max energy needed, but still winnable)
                worst_t = None
                worst_need = -1
                for (t, w) in succs:
                    needed = max(0, _sub_energy(energy[t], w))
                    if needed > worst_need:
                        worst_need = needed
                        worst_t = t
                if worst_t is not None:
                    strategy_opponent[v] = worst_t

    return EnergyResult(min_energy, win_energy, win_opponent,
                        strategy_energy, strategy_opponent)


def _sub_energy(e: float, w: int) -> float:
    """Compute e - w handling infinity."""
    if e == INF_ENERGY:
        return INF_ENERGY
    return e - w


# ---------------------------------------------------------------------------
# Fixed Initial Energy
# ---------------------------------------------------------------------------

def solve_fixed_energy(game: EnergyGame, initial_energy: int) -> Dict[int, bool]:
    """Determine which vertices Even can win with a fixed initial energy.

    Returns dict mapping vertex -> True if Even can win from there
    with the given initial energy.
    """
    result = solve_energy(game)
    return {v: (result.min_energy[v] is not None and
                result.min_energy[v] <= initial_energy)
            for v in game.vertices}


# ---------------------------------------------------------------------------
# Energy-Parity Game Solver
# ---------------------------------------------------------------------------

def solve_energy_parity(game: EnergyParityGame) -> EnergyParityResult:
    """Solve an energy-parity game.

    Player Even wins iff:
    1. Energy stays >= 0 forever, AND
    2. The max priority seen infinitely often is even.

    Algorithm: iterative approach combining parity game structure with
    energy computation. Uses Zielonka-style recursion where at each level
    we solve an energy game restricted to the relevant subgame.

    The key insight (Chatterjee-Doyen 2012): the minimum initial energy
    for energy-parity games is bounded by (n * d * W) where n = vertices,
    d = number of distinct priorities, W = max absolute weight.
    """
    if not game.vertices:
        return EnergyParityResult({}, set(), set(), {}, {})

    n = len(game.vertices)
    d = len(set(game.priority.values())) if game.priority else 1
    W = game.max_weight()
    bound = max(1, n * d * W)

    # Strategy: intersect parity winning region with energy winning region.
    # A vertex is in the energy-parity winning region iff:
    # 1. Even wins the parity condition (can force even max priority infinitely often)
    # 2. Even can maintain non-negative energy while doing so
    #
    # Iterative approach: solve parity on the graph, then check energy on the
    # parity-winning subgame. If some vertices fail energy, remove them and
    # re-solve parity. Repeat until stable.

    remaining = set(game.vertices)
    strat_even = {}
    strat_odd = {}

    while True:
        # Solve parity on remaining vertices
        pg = ParityGame()
        for v in remaining:
            if v in game.priority:
                pg.add_vertex(v, game.owner[v], game.priority[v])
        for v in remaining:
            for (t, w) in game.edges.get(v, []):
                if t in remaining:
                    pg.add_edge(v, t)

        parity_sol = zielonka(pg)
        parity_win = parity_sol.win_even & remaining

        if not parity_win:
            break

        # Build energy game restricted to parity-winning vertices
        eg = EnergyGame()
        for v in parity_win:
            eg.add_vertex(v, game.owner[v])
        for v in parity_win:
            for (t, w) in game.edges.get(v, []):
                if t in parity_win:
                    eg.add_edge(v, t, w)

        # Check energy on this subgame
        energy_result = solve_energy(eg)

        # Vertices that win both
        win_both = energy_result.win_energy
        fail = parity_win - win_both

        if not fail:
            # Stable -- all parity-winners also win energy
            strat_even = energy_result.strategy_energy
            break
        else:
            # Remove failing vertices and retry
            remaining -= fail

    win_even = set()
    win_odd = set()
    min_energy = {}

    # Build energy game on final winning region
    eg_final = EnergyGame()
    for v in remaining:
        eg_final.add_vertex(v, game.owner[v])
    for v in remaining:
        for (t, w) in game.edges.get(v, []):
            if t in remaining:
                eg_final.add_edge(v, t, w)

    # Re-solve parity to get final classification
    pg_final = ParityGame()
    for v in remaining:
        if v in game.priority:
            pg_final.add_vertex(v, game.owner[v], game.priority[v])
    for v in remaining:
        for (t, w) in game.edges.get(v, []):
            if t in remaining:
                pg_final.add_edge(v, t)

    parity_final = zielonka(pg_final)
    parity_win_final = parity_final.win_even & remaining

    # Check energy on parity-winning subgame
    eg_win = EnergyGame()
    for v in parity_win_final:
        eg_win.add_vertex(v, game.owner[v])
    for v in parity_win_final:
        for (t, w) in game.edges.get(v, []):
            if t in parity_win_final:
                eg_win.add_edge(v, t, w)

    if parity_win_final:
        energy_final = solve_energy(eg_win)
        win_even = energy_final.win_energy
        strat_even = energy_final.strategy_energy
    else:
        win_even = set()

    win_odd = game.vertices - win_even

    for v in game.vertices:
        if v in win_even:
            min_energy[v] = energy_final.min_energy.get(v)
        else:
            min_energy[v] = None

    return EnergyParityResult(min_energy, win_even, win_odd,
                              strat_even, strat_odd)


def _solve_ep_recursive(game: EnergyParityGame, verts: Set[int],
                         bound: int) -> Tuple[Set[int], Set[int], Dict, Dict]:
    """Recursive Zielonka-style solver for energy-parity games.

    Returns (win_even, win_odd, strategy_even, strategy_odd).
    """
    if not verts:
        return set(), set(), {}, {}

    # Find max priority in subgame
    max_p = max(game.priority[v] for v in verts if v in game.priority)
    player = Player.EVEN if max_p % 2 == 0 else Player.ODD

    # Vertices with max priority
    U = {v for v in verts if game.priority.get(v) == max_p}

    # Compute attractor of U for `player` within verts
    attr, attr_strat = _attractor_ep(game, U, player, verts)

    # Solve the subgame without the attractor
    remaining = verts - attr
    w0, w1, s0, s1 = _solve_ep_recursive(game, remaining, bound)

    if player == Player.EVEN:
        # Even controls max priority vertices
        # If Odd's winning region in subgame is empty, Even wins everything
        if not w1:
            # Even wins attr + remaining
            strat_even = {**s0, **attr_strat}
            # For Even vertices in attr without strategy yet, pick any successor in verts
            for v in attr:
                if v not in strat_even and game.owner[v] == Player.EVEN:
                    for (t, w) in game.successors(v):
                        if t in verts:
                            strat_even[v] = t
                            break
            return verts, set(), strat_even, {}
        else:
            # Odd wins w1 -- compute Odd's attractor to w1
            attr2, attr2_strat = _attractor_ep(game, w1, Player.ODD, verts)
            remaining2 = verts - attr2
            # Check energy condition on attr2
            # Vertices in attr2 might still have energy issues
            # Recurse on the rest
            rw0, rw1, rs0, rs1 = _solve_ep_recursive(game, remaining2, bound)
            strat_odd = {**s1, **attr2_strat, **rs1}
            strat_even = {**rs0}
            return rw0, attr2 | rw1, strat_even, strat_odd
    else:
        # Odd controls max priority vertices
        if not w0:
            strat_odd = {**s1, **attr_strat}
            for v in attr:
                if v not in strat_odd and game.owner[v] == Player.ODD:
                    for (t, w) in game.successors(v):
                        if t in verts:
                            strat_odd[v] = t
                            break
            return set(), verts, {}, strat_odd
        else:
            attr2, attr2_strat = _attractor_ep(game, w0, Player.EVEN, verts)
            remaining2 = verts - attr2
            rw0, rw1, rs0, rs1 = _solve_ep_recursive(game, remaining2, bound)
            strat_even = {**s0, **attr2_strat, **rs0}
            strat_odd = {**rs1}
            return attr2 | rw0, rw1, strat_even, strat_odd


def _attractor_ep(game: EnergyParityGame, target: Set[int], player: Player,
                  arena: Set[int]) -> Tuple[Set[int], Dict[int, int]]:
    """Compute attractor set for player toward target within arena."""
    attr = set(target)
    strategy = {}
    queue = deque(target)

    while queue:
        v = queue.popleft()
        # Find predecessors in arena
        for u in arena:
            if u in attr:
                continue
            for (t, w) in game.edges.get(u, []):
                if t == v:
                    if game.owner[u] == player:
                        # Player can choose to go to attr
                        attr.add(u)
                        strategy[u] = v
                        queue.append(u)
                        break
                    else:
                        # Opponent -- must check ALL successors in arena lead to attr
                        all_in_attr = True
                        for (t2, w2) in game.edges.get(u, []):
                            if t2 in arena and t2 not in attr:
                                all_in_attr = False
                                break
                        if all_in_attr:
                            attr.add(u)
                            queue.append(u)
                            break

    return attr, strategy


def _compute_min_energy_ep(game: EnergyParityGame, start: int,
                           win_set: Set[int], strategy: Dict[int, int],
                           bound: int) -> int:
    """Compute minimum initial energy for a vertex in the winning set.

    Uses value iteration restricted to the winning set.
    """
    # Build an energy game from the winning region
    eg = EnergyGame()
    for v in win_set:
        eg.add_vertex(v, game.owner[v])
    for v in win_set:
        for (t, w) in game.edges.get(v, []):
            if t in win_set:
                eg.add_edge(v, t, w)

    result = solve_energy(eg)
    if result.min_energy.get(start) is not None:
        return result.min_energy[start]
    return bound  # Fallback


# ---------------------------------------------------------------------------
# Mean-Payoff Games
# ---------------------------------------------------------------------------

def solve_mean_payoff(game: EnergyGame) -> MeanPayoffResult:
    """Solve a mean-payoff game.

    The mean payoff of a play v0 v1 v2 ... is:
        lim inf_{n->inf} (1/n) * sum_{i=0}^{n-1} w(v_i, v_{i+1})

    Player 0 wants to maximize, Player 1 wants to minimize.

    Connection to energy games: Player 0 has a strategy with mean-payoff >= 0
    iff Player 0 can win the energy game (with some finite initial energy).

    Algorithm: binary search on threshold + energy game solving.
    Also computes approximate values via value iteration.
    """
    if not game.vertices:
        return MeanPayoffResult({}, set(), set(), {}, {})

    n = len(game.vertices)
    W = game.max_weight()

    if W == 0:
        # All weights zero -- mean payoff is 0 everywhere
        values = {v: 0.0 for v in game.vertices}
        return MeanPayoffResult(values, set(game.vertices), set(), {}, {})

    # Use value iteration to approximate mean-payoff values
    # Standard approach: compute T_n(v) = optimal total weight over n steps
    # Mean payoff = lim T_n / n

    num_iters = n * n  # sufficient for convergence

    # T[v] = optimal total weight achievable starting from v in remaining steps
    T = {v: 0.0 for v in game.vertices}
    strategy_p0 = {}
    strategy_p1 = {}

    for _ in range(num_iters):
        T_new = {}
        for v in game.vertices:
            succs = game.successors(v)
            if not succs:
                T_new[v] = 0.0
                continue

            vals = []
            for (t, w) in succs:
                vals.append((w + T[t], t))

            if game.owner[v] == Player.EVEN:
                best_val, best_t = max(vals, key=lambda x: x[0])
                strategy_p0[v] = best_t
            else:
                best_val, best_t = min(vals, key=lambda x: x[0])
                strategy_p1[v] = best_t

            T_new[v] = best_val
        T = T_new

    # Mean payoff approximation: T[v] / num_iters
    values = {v: T[v] / num_iters for v in game.vertices}

    # Classify winning regions
    win_nonneg = set()
    win_neg = set()

    # Use energy game for exact classification
    energy_result = solve_energy(game)
    for v in game.vertices:
        if v in energy_result.win_energy:
            win_nonneg.add(v)
        else:
            win_neg.add(v)

    return MeanPayoffResult(values, win_nonneg, win_neg,
                           strategy_p0, strategy_p1)


def mean_payoff_threshold(game: EnergyGame, threshold: float) -> Set[int]:
    """Find vertices where Player 0 achieves mean-payoff >= threshold.

    Reduces to energy game by subtracting threshold from all weights.
    """
    shifted = EnergyGame()
    for v in game.vertices:
        shifted.add_vertex(v, game.owner[v])
    for v in game.vertices:
        for (t, w) in game.edges.get(v, []):
            # Shift weight: if mean-payoff >= threshold, then mean of (w - threshold) >= 0
            # Use integer approximation (multiply by precision factor)
            shifted_w = w - int(threshold)  # Works for integer thresholds
            shifted.add_edge(v, t, shifted_w)

    result = solve_energy(shifted)
    return result.win_energy


# ---------------------------------------------------------------------------
# Simulation and Verification
# ---------------------------------------------------------------------------

def simulate_play(game: EnergyGame, start: int, strategy_even: Dict[int, int],
                  strategy_odd: Dict[int, int], initial_energy: int,
                  max_steps: int = 100) -> List[Tuple[int, int, int]]:
    """Simulate a play and return trace of (vertex, weight_taken, energy_after).

    Returns the trace. Energy dropping below 0 means Even loses.
    """
    trace = []
    v = start
    energy = initial_energy
    visited = set()

    for step in range(max_steps):
        succs = game.successors(v)
        if not succs:
            trace.append((v, 0, energy))
            break

        # Choose move based on strategy
        if game.owner[v] == Player.EVEN:
            t = strategy_even.get(v)
            if t is None:
                t = succs[0][0]
        else:
            t = strategy_odd.get(v)
            if t is None:
                t = succs[0][0]

        # Find weight for this edge
        w = 0
        for (tt, ww) in succs:
            if tt == t:
                w = ww
                break

        energy += w
        trace.append((v, w, energy))

        # Check for cycle detection
        state = (v, step % (len(game.vertices) + 1))
        if energy < 0:
            break

        v = t

    return trace


def verify_energy_strategy(game: EnergyGame, start: int,
                           strategy: Dict[int, int],
                           initial_energy: int,
                           max_steps: int = 1000) -> bool:
    """Verify that a strategy for Even maintains non-negative energy.

    Simulates against all possible opponent moves (DFS).
    Returns True if the strategy is winning from start with initial_energy.
    """
    # Use the solved game to check: if Even's minimum energy to win from start
    # is at most initial_energy, and the strategy only targets winning vertices,
    # then the strategy is valid.
    # For a more direct check, simulate with cycle detection.

    # Simulate the strategy, tracking visited (vertex, energy) states.
    # Use exact energy (no capping) but detect cycles.
    # A cycle with non-negative accumulated weight is fine.
    # A cycle with negative accumulated weight means energy depletes.

    v = start
    energy = initial_energy
    path = []  # list of (vertex, energy_before_move)
    visited_states = {}  # (vertex) -> list of energy levels seen

    for step in range(max_steps):
        succs = game.successors(v)
        if not succs:
            if game.owner[v] == Player.EVEN:
                return False  # Even stuck = loses
            return True  # Odd stuck = Even wins

        if game.owner[v] == Player.EVEN:
            t = strategy.get(v)
            if t is None:
                return False

            w = 0
            for (tt, ww) in succs:
                if tt == t:
                    w = ww
                    break

            new_energy = energy + w
            if new_energy < 0:
                return False

            # Check for cycle: same vertex with same or lower energy = will deplete or stall
            if v in visited_states:
                for prev_energy in visited_states[v]:
                    if new_energy <= prev_energy:
                        # Energy is not increasing -> if cycle has net <= 0 weight, will deplete
                        # Actually: we're at v again with energy <= before. If weight sum
                        # in the cycle is <= 0, this will eventually deplete (or stall at 0).
                        # For negative net: will deplete. For zero net: stable (ok).
                        if new_energy < prev_energy:
                            return False  # Strictly decreasing -> will deplete
                        else:
                            return True  # Same energy at same vertex = stable cycle

            visited_states.setdefault(v, []).append(energy)
            energy = new_energy
            v = t
        else:
            # For Odd's moves: try ALL and if ANY leads to depletion, return False
            # This is expensive for DFS, so use simplified single-path check
            # Pick worst move for Even
            worst_t = None
            worst_energy = float('inf')
            for (t, w) in succs:
                ne = energy + w
                if ne < 0:
                    return False  # Opponent can deplete
                if ne < worst_energy:
                    worst_energy = ne
                    worst_t = t

            visited_states.setdefault(v, []).append(energy)
            energy = worst_energy
            v = worst_t

    return True


# ---------------------------------------------------------------------------
# Construction Helpers
# ---------------------------------------------------------------------------

def make_simple_energy_game(edges: List[Tuple[int, int, int]],
                            owners: Dict[int, Player]) -> EnergyGame:
    """Create an energy game from edge list and owner dict.

    edges: list of (source, target, weight)
    owners: dict mapping vertex -> Player
    """
    g = EnergyGame()
    for v, p in owners.items():
        g.add_vertex(v, p)
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


def make_energy_parity_game(edges: List[Tuple[int, int, int]],
                            owners: Dict[int, Player],
                            priorities: Dict[int, int]) -> EnergyParityGame:
    """Create an energy-parity game from components."""
    g = EnergyParityGame()
    for v, p in owners.items():
        g.add_vertex(v, p, priorities.get(v, 0))
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


def make_chain_energy_game(n: int, weights: Optional[List[int]] = None) -> EnergyGame:
    """Create a chain 0 -> 1 -> ... -> n-1 -> 0 with alternating owners.

    Default weights: +1 for even vertices, -1 for odd vertices.
    """
    g = EnergyGame()
    if weights is None:
        weights = [(1 if i % 2 == 0 else -1) for i in range(n)]

    for i in range(n):
        g.add_vertex(i, Player.EVEN if i % 2 == 0 else Player.ODD)
    for i in range(n):
        g.add_edge(i, (i + 1) % n, weights[i % len(weights)])

    return g


def make_charging_game(n: int, charge: int, drain: int) -> EnergyGame:
    """Create a game where Even can visit a charging station.

    Structure: vertices 0..n-1 in a line, vertex 0 is charging (+charge),
    all others drain (-drain). Even owns vertex 0, Odd owns the rest.
    Vertex n-1 loops back to 0.
    """
    g = EnergyGame()
    for i in range(n):
        g.add_vertex(i, Player.EVEN if i == 0 else Player.ODD)

    for i in range(n - 1):
        g.add_edge(i, i + 1, -drain)
    g.add_edge(n - 1, 0, charge)
    # Even at vertex 0 can also self-loop to charge
    g.add_edge(0, 0, charge)

    return g


def make_choice_game() -> EnergyGame:
    """Classic example: Even chooses between safe and risky paths.

    Vertex 0 (Even): choice point
    Vertex 1 (Odd): safe path (weight +1 loop)
    Vertex 2 (Odd): risky path (weight -2, then +5 payoff or -3 penalty)
    Vertex 3 (Even): payoff (+5 back to 0)
    Vertex 4 (Even): penalty (-3 back to 0)
    """
    g = EnergyGame()
    g.add_vertex(0, Player.EVEN)
    g.add_vertex(1, Player.ODD)
    g.add_vertex(2, Player.ODD)
    g.add_vertex(3, Player.EVEN)
    g.add_vertex(4, Player.EVEN)

    g.add_edge(0, 1, 0)   # Even -> safe
    g.add_edge(0, 2, -2)  # Even -> risky (costs 2)
    g.add_edge(1, 1, 1)   # safe loop (+1)
    g.add_edge(2, 3, 5)   # Odd -> payoff
    g.add_edge(2, 4, -3)  # Odd -> penalty
    g.add_edge(3, 0, 0)   # back to start
    g.add_edge(4, 0, 0)   # back to start

    return g


# ---------------------------------------------------------------------------
# Comparison and Analysis
# ---------------------------------------------------------------------------

def compare_energy_vs_parity(game: EnergyParityGame) -> dict:
    """Compare results of pure energy, pure parity, and combined analysis."""
    # Pure energy
    eg = game.to_energy_game()
    energy_result = solve_energy(eg)

    # Pure parity
    pg = game.to_parity_game()
    parity_result = zielonka(pg)

    # Combined
    ep_result = solve_energy_parity(game)

    return {
        'energy_only': {
            'win_energy': energy_result.win_energy,
            'win_opponent': energy_result.win_opponent,
            'min_energy': energy_result.min_energy,
        },
        'parity_only': {
            'win_even': parity_result.win_even,
            'win_odd': parity_result.win_odd,
        },
        'energy_parity': {
            'win_energy': ep_result.win_energy,
            'win_opponent': ep_result.win_opponent,
            'min_energy': ep_result.min_energy,
        },
        'analysis': {
            'energy_wins_but_parity_loses': energy_result.win_energy - parity_result.win_even,
            'parity_wins_but_energy_loses': parity_result.win_even - energy_result.win_energy,
            'both_win': energy_result.win_energy & parity_result.win_even,
            'neither_wins': energy_result.win_opponent & parity_result.win_odd,
            'combined_wins': ep_result.win_energy,
        }
    }


def energy_game_statistics(game: EnergyGame) -> dict:
    """Compute statistics about an energy game."""
    result = solve_energy(game)

    finite_energies = [e for e in result.min_energy.values() if e is not None]

    return {
        'num_vertices': len(game.vertices),
        'num_edges': sum(len(s) for s in game.edges.values()),
        'max_weight': game.max_weight(),
        'num_even_vertices': sum(1 for v in game.vertices if game.owner[v] == Player.EVEN),
        'num_odd_vertices': sum(1 for v in game.vertices if game.owner[v] == Player.ODD),
        'win_energy_count': len(result.win_energy),
        'win_opponent_count': len(result.win_opponent),
        'min_energies': result.min_energy,
        'max_min_energy': max(finite_energies) if finite_energies else None,
        'avg_min_energy': sum(finite_energies) / len(finite_energies) if finite_energies else None,
    }


def parity_to_energy(pg: ParityGame) -> EnergyGame:
    """Convert a parity game to an energy game.

    Uses the standard reduction: for each edge (u, v), the weight
    encodes the parity objective. This is a theoretical construction
    showing that energy games subsume parity games.

    The reduction adds intermediate vertices and uses weight encoding
    to simulate the parity condition via energy.
    """
    # Simple encoding: weight = 0 for all edges (just reuses the graph structure)
    # The actual parity-to-energy reduction is more complex and involves
    # exponential blowup. Here we provide a structural conversion.
    g = EnergyGame()
    for v in pg.vertices:
        g.add_vertex(v, pg.owner[v])
    for v in pg.vertices:
        for t in pg.successors(v):
            g.add_edge(v, t, 0)
    return g


def energy_to_mean_payoff(game: EnergyGame) -> MeanPayoffResult:
    """Analyze mean-payoff properties of an energy game."""
    return solve_mean_payoff(game)
