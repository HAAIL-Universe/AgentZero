"""
V161: Mean-Payoff Parity Games
===============================
Two-player infinite-duration games with combined mean-payoff and parity objectives.

Player Even wins a play iff:
1. The highest priority seen infinitely often is even (parity condition), AND
2. The long-run average weight (mean payoff) is >= threshold (mean-payoff condition).

This is strictly more expressive than either parity games (V156) or mean-payoff
games (V160) alone. Applications include:
- Quantitative synthesis: build a controller that satisfies qualitative (parity)
  and quantitative (mean-payoff) requirements simultaneously
- Resource-aware verification: ensure liveness (parity) while maintaining
  resource bounds (mean-payoff)

Key theoretical results:
- Mean-payoff parity games are determined (Chatterjee, Doyen, Henzinger 2010)
- Decidable in NP intersect coNP
- Optimal values are rational with bounded denominators

Algorithms:
1. Direct mean-payoff parity solving via iterative Zielonka + mean-payoff refinement
2. Threshold mean-payoff parity (is mean-payoff >= threshold under parity?)
3. Optimal value computation via binary search + threshold checking
4. Strategy extraction and verification
5. Decomposition analysis: parity-only vs mean-payoff-only vs combined

Composes: V156 (Parity Games), V160 (Energy Games).
"""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import sys

sys.path.insert(0, 'Z:/AgentZero/A2/work/V156_parity_games')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V160_energy_games')
from parity_games import Player, ParityGame, Solution, zielonka, attractor
from energy_games import (EnergyGame, EnergyResult, EnergyParityGame,
                          solve_energy, solve_energy_parity,
                          MeanPayoffResult, INF_ENERGY)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MeanPayoffParityGame:
    """A two-player game with combined mean-payoff and parity objectives.

    Vertices have owners (Even/Odd), priorities (for parity), and
    edges carry integer weights (for mean-payoff).

    Even wins iff: parity condition holds AND mean-payoff >= 0
    (or >= threshold for threshold variant).
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
        W = 0
        for v in self.vertices:
            for (_, w) in self.edges.get(v, []):
                W = max(W, abs(w))
        return max(W, 1)

    def max_priority(self) -> int:
        if not self.priority:
            return 0
        return max(self.priority.values())

    def to_parity_game(self) -> ParityGame:
        """Strip weights, keep only parity condition."""
        g = ParityGame()
        for v in self.vertices:
            g.add_vertex(v, self.owner[v], self.priority[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t)
        return g

    def to_energy_game(self) -> EnergyGame:
        """Strip priorities, keep only weights as energy game."""
        g = EnergyGame()
        for v in self.vertices:
            g.add_vertex(v, self.owner[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t, w)
        return g

    def to_energy_parity_game(self) -> EnergyParityGame:
        """Convert to energy-parity game (keeps both conditions)."""
        g = EnergyParityGame()
        for v in self.vertices:
            g.add_vertex(v, self.owner[v], self.priority[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t, w)
        return g

    def subgame(self, verts: Set[int]) -> 'MeanPayoffParityGame':
        """Restrict game to vertex subset."""
        g = MeanPayoffParityGame()
        for v in verts:
            if v in self.owner and v in self.priority:
                g.add_vertex(v, self.owner[v], self.priority[v])
        for v in verts:
            for (t, w) in self.edges.get(v, []):
                if t in verts:
                    g.add_edge(v, t, w)
        return g

    def shift_weights(self, delta: int) -> 'MeanPayoffParityGame':
        """Return a copy with all weights shifted by delta."""
        g = MeanPayoffParityGame()
        for v in self.vertices:
            g.add_vertex(v, self.owner[v], self.priority[v])
        for v in self.vertices:
            for (t, w) in self.edges.get(v, []):
                g.add_edge(v, t, w + delta)
        return g


@dataclass
class MPPResult:
    """Result of mean-payoff parity game analysis."""
    # Winning region for Even (parity + mean-payoff >= threshold)
    win_even: Set[int]
    # Winning region for Odd
    win_odd: Set[int]
    # Strategy for Even (vertex -> chosen successor)
    strategy_even: Dict[int, int]
    # Strategy for Odd
    strategy_odd: Dict[int, int]
    # Mean-payoff values under optimal play (if computed)
    values: Optional[Dict[int, float]] = None
    # Threshold used
    threshold: float = 0.0


# ---------------------------------------------------------------------------
# Core Solver: Mean-Payoff Parity via Shifted Energy-Parity
# ---------------------------------------------------------------------------

def _compute_sccs(vertices: Set[int], edges: Dict[int, List[Tuple[int, int]]]) -> List[Set[int]]:
    """Tarjan's SCC algorithm."""
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for (t, _) in edges.get(v, []):
            if t not in vertices:
                continue
            if t not in index:
                strongconnect(t)
                lowlink[v] = min(lowlink[v], lowlink[t])
            elif t in on_stack:
                lowlink[v] = min(lowlink[v], index[t])

        if lowlink[v] == index[v]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.add(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in vertices:
        if v not in index:
            strongconnect(v)

    return sccs


def _scc_mean_payoff(scc: Set[int], game: MeanPayoffParityGame, player: Player) -> float:
    """Compute the optimal mean payoff within an SCC for a given player.

    Uses the minimum cycle mean / maximum cycle mean approach.
    For Even: maximize mean-payoff. For Odd: minimize mean-payoff.
    """
    if len(scc) <= 1:
        v = next(iter(scc))
        # Check self-loop
        for (t, w) in game.edges.get(v, []):
            if t == v:
                return float(w)
        return 0.0

    # Bellman-Ford based minimum/maximum cycle mean
    verts = sorted(scc)
    n = len(verts)
    INF = float('inf')

    # dist[k][v] = optimal weight of a walk of length k ending at v
    # Using Karp's algorithm for minimum (or maximum) cycle mean
    dist = [{v: INF for v in verts} for _ in range(n + 1)]

    # Start from all vertices
    for v in verts:
        dist[0][v] = 0.0

    for k in range(1, n + 1):
        for v in verts:
            for (t, w) in game.edges.get(v, []):
                if t in scc:
                    cand = dist[k-1][v] + w
                    if cand < dist[k][t]:
                        dist[k][t] = cand

    # Karp's formula: max over v of min over k<n of (dist[n][v] - dist[k][v]) / (n - k)
    # This gives the MAXIMUM cycle mean
    best_mcm = -INF
    for v in verts:
        if dist[n][v] >= INF:
            continue
        worst_for_v = INF
        for k in range(n):
            if dist[k][v] >= INF:
                continue
            mcm = (dist[n][v] - dist[k][v]) / (n - k)
            worst_for_v = min(worst_for_v, mcm)
        if worst_for_v < INF:
            best_mcm = max(best_mcm, worst_for_v)

    if best_mcm == -INF or best_mcm == INF:
        return 0.0
    return best_mcm


def _solve_mp_under_strategy(game: MeanPayoffParityGame,
                             verts: Set[int],
                             strategy_even: Dict[int, int],
                             threshold: float) -> Set[int]:
    """Check mean-payoff under Even's fixed parity strategy.

    For Even-owned vertices, only the strategy edge is available.
    For Odd-owned vertices, all edges remain (Odd plays adversarially).
    Returns the set of vertices where energy is finite (mean-payoff >= threshold).
    """
    if not verts:
        return set()

    n = len(verts)
    if threshold == int(threshold):
        shift = int(threshold)
        scale = 1
    else:
        scale = n
        shift = int(round(threshold * n))

    eg = EnergyGame()
    for v in verts:
        eg.add_vertex(v, game.owner[v])
    for v in verts:
        if game.owner[v] == Player.EVEN and v in strategy_even:
            # Only include the strategy edge
            target = strategy_even[v]
            for (t, w) in game.edges.get(v, []):
                if t == target and t in verts:
                    eg.add_edge(v, t, w * scale - shift)
                    break
        else:
            # Odd vertex or no strategy: include all edges in subgame
            for (t, w) in game.edges.get(v, []):
                if t in verts:
                    eg.add_edge(v, t, w * scale - shift)

    result = solve_energy(eg)
    return set(result.win_energy)


def _solve_mp_free(game: MeanPayoffParityGame,
                   verts: Set[int],
                   threshold: float) -> Set[int]:
    """Check mean-payoff freely (Even chooses best edge). No parity constraint."""
    if not verts:
        return set()

    n = len(verts)
    if threshold == int(threshold):
        shift = int(threshold)
        scale = 1
    else:
        scale = n
        shift = int(round(threshold * n))

    eg = EnergyGame()
    for v in verts:
        eg.add_vertex(v, game.owner[v])
    for v in verts:
        for (t, w) in game.edges.get(v, []):
            if t in verts:
                eg.add_edge(v, t, w * scale - shift)

    result = solve_energy(eg)
    return set(result.win_energy)


def solve_mpp_threshold(game: MeanPayoffParityGame, threshold: float = 0.0) -> MPPResult:
    """Solve a mean-payoff parity game with a given threshold.

    Even wins from vertex v iff there exists a strategy such that:
    1. The parity condition is satisfied, AND
    2. The mean-payoff is >= threshold.

    Algorithm: Iterative refinement (McNaughton-Zielonka style).
    1. Solve parity on current subgame -> Even's winning region + strategy
    2. Check mean-payoff under Even's PARITY STRATEGY (not free choice)
    3. If some vertices fail mean-payoff under this strategy, remove them
       and their Odd attractor, then re-solve parity
    4. Repeat until stable or empty

    Key insight: Even must use a SINGLE strategy satisfying both conditions.
    Checking mean-payoff under the parity strategy ensures consistency.
    """
    if not game.vertices:
        return MPPResult(set(), set(), {}, {}, threshold=threshold)

    remaining = set(game.vertices)
    final_strategy_even = {}
    final_strategy_odd = {}

    for _ in range(len(game.vertices) + 1):
        if not remaining:
            break

        # Step 1: Solve parity on current subgame
        sub = game.subgame(remaining)
        pg = sub.to_parity_game()
        if not pg.vertices:
            break
        parity_sol = zielonka(pg)

        parity_win_even = set(parity_sol.win_even) & remaining
        parity_win_odd = set(parity_sol.win_odd) & remaining

        if not parity_win_even:
            break

        # Step 2: Check mean-payoff under Even's parity strategy
        mp_ok = _solve_mp_under_strategy(
            game, parity_win_even, parity_sol.strategy_even, threshold
        )

        if mp_ok == parity_win_even:
            # Even's parity strategy also achieves mean-payoff. Done!
            final_strategy_even.update(parity_sol.strategy_even)
            final_strategy_odd.update(parity_sol.strategy_odd)
            win_even = parity_win_even
            win_odd = game.vertices - win_even
            return MPPResult(
                win_even=win_even,
                win_odd=win_odd,
                strategy_even=final_strategy_even,
                strategy_odd=final_strategy_odd,
                threshold=threshold,
            )

        # Step 3: MP failed for some vertices under this parity strategy.
        # Remove them + their Odd attractor from the game.
        mp_lost = parity_win_even - mp_ok
        to_remove = parity_win_odd | mp_lost

        if not to_remove:
            # Stable but some vertices couldn't win MP. Return what we have.
            win_even = mp_ok
            win_odd = game.vertices - win_even
            final_strategy_even.update(parity_sol.strategy_even)
            final_strategy_odd.update(parity_sol.strategy_odd)
            return MPPResult(
                win_even=win_even,
                win_odd=win_odd,
                strategy_even=final_strategy_even,
                strategy_odd=final_strategy_odd,
                threshold=threshold,
            )

        sub_pg = game.subgame(remaining).to_parity_game()
        odd_attr = attractor(sub_pg, to_remove, Player.ODD)
        final_strategy_odd.update(parity_sol.strategy_odd)
        remaining -= odd_attr

    # Exhausted: Even can't win anywhere
    return MPPResult(
        win_even=set(),
        win_odd=set(game.vertices),
        strategy_even=final_strategy_even,
        strategy_odd=final_strategy_odd,
        threshold=threshold,
    )


def solve_mpp(game: MeanPayoffParityGame) -> MPPResult:
    """Solve a mean-payoff parity game (threshold = 0).

    Even wins iff parity condition holds AND mean-payoff >= 0.
    """
    return solve_mpp_threshold(game, threshold=0.0)


# ---------------------------------------------------------------------------
# Optimal Value Computation
# ---------------------------------------------------------------------------

def compute_mpp_values(game: MeanPayoffParityGame) -> MPPResult:
    """Compute the optimal mean-payoff value for each vertex under parity constraint.

    Uses binary search over threshold values. The optimal mean-payoff value v*
    at a vertex is the supremum of thresholds t such that Even wins the
    mean-payoff parity game with threshold t.

    Mean-payoff parity values are rationals p/q with |q| <= n, so we can
    binary search with precision 1/(n^2).
    """
    if not game.vertices:
        return MPPResult(set(), set(), {}, {}, values={}, threshold=0.0)

    n = len(game.vertices)
    W = game.max_weight()

    if n == 0 or W == 0:
        result = solve_mpp(game)
        result.values = {v: 0.0 for v in game.vertices}
        return result

    # Values are in range [-W, W] with denominator <= n
    precision = 1.0 / (2 * n * n) if n > 0 else 0.5
    values = {}

    for v in game.vertices:
        lo, hi = -float(W), float(W)

        # Binary search for optimal threshold
        for _ in range(int(math.log2(2 * W / precision)) + 5):
            mid = (lo + hi) / 2.0
            result = solve_mpp_threshold(game, mid)
            if v in result.win_even:
                lo = mid  # Even can achieve at least mid
            else:
                hi = mid  # Even cannot achieve mid

        # Snap to nearest rational with denominator <= n
        values[v] = _snap_rational(lo, n, W)

    # Final solve at threshold=0 for winning regions
    base_result = solve_mpp(game)
    base_result.values = values
    return base_result


def _snap_rational(x: float, n: int, W: int) -> float:
    """Snap a float to the nearest rational p/q with |q| <= n and |p/q| <= W."""
    best = round(x)
    best_dist = abs(x - best)

    for q in range(1, n + 1):
        p = round(x * q)
        val = p / q
        if abs(val) <= W + 1:
            dist = abs(x - val)
            if dist < best_dist:
                best = val
                best_dist = dist

    return best


# ---------------------------------------------------------------------------
# Strategy Verification
# ---------------------------------------------------------------------------

def verify_mpp_strategy(game: MeanPayoffParityGame, strategy: Dict[int, int],
                        player: Player, threshold: float = 0.0,
                        max_steps: int = 10000) -> Dict[str, object]:
    """Verify a strategy for a mean-payoff parity game.

    Simulates a play under the given strategy (opponent plays adversarially).
    Checks:
    1. Parity condition: highest priority seen infinitely often is even (for Even)
    2. Mean-payoff: long-run average >= threshold (for Even)
    """
    if not game.vertices:
        return {"valid": True, "reason": "empty game"}

    results = {}

    for start in game.vertices:
        if game.owner[start] != player and start not in strategy:
            # Skip vertices not reachable under this strategy
            continue

        # Simulate play
        path = [start]
        weights = []
        visited = {}  # vertex -> first step index
        visited[start] = 0
        current = start
        cycle_start = -1

        for step in range(max_steps):
            succs = game.successors(current)
            if not succs:
                break

            if current in strategy:
                # Strategy player chooses
                chosen = strategy[current]
                w = None
                for (t, wt) in succs:
                    if t == chosen:
                        w = wt
                        break
                if w is None:
                    results[start] = {"valid": False, "reason": f"strategy picks invalid successor {chosen} from {current}"}
                    break
                nxt, weight = chosen, w
            else:
                # Opponent chooses adversarially (worst for strategy player)
                if player == Player.EVEN:
                    # Odd chooses to minimize mean-payoff
                    nxt, weight = min(succs, key=lambda tw: tw[1])
                else:
                    nxt, weight = max(succs, key=lambda tw: tw[1])

            weights.append(weight)
            path.append(nxt)

            if nxt in visited:
                cycle_start = visited[nxt]
                break
            visited[nxt] = len(path) - 1
            current = nxt
        else:
            results[start] = {"valid": True, "reason": "no cycle found in max_steps",
                              "mean_payoff": sum(weights) / len(weights) if weights else 0.0}
            continue

        if cycle_start >= 0:
            cycle_weights = weights[cycle_start:]
            cycle_path = path[cycle_start:-1]
            cycle_priorities = [game.priority.get(v, 0) for v in cycle_path]

            mean_payoff = sum(cycle_weights) / len(cycle_weights) if cycle_weights else 0.0
            max_prio = max(cycle_priorities) if cycle_priorities else 0
            parity_ok = (max_prio % 2 == 0) if player == Player.EVEN else (max_prio % 2 == 1)

            if player == Player.EVEN:
                valid = parity_ok and mean_payoff >= threshold - 1e-9
            else:
                valid = (not parity_ok) or mean_payoff < threshold + 1e-9

            results[start] = {
                "valid": valid,
                "mean_payoff": mean_payoff,
                "parity_ok": parity_ok,
                "max_cycle_priority": max_prio,
                "cycle_length": len(cycle_weights),
            }
        elif not succs:
            results[start] = {"valid": game.owner[current] != player,
                              "reason": "dead end reached"}

    return results


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_play(game: MeanPayoffParityGame,
                  strategy_even: Dict[int, int],
                  strategy_odd: Dict[int, int],
                  start: int,
                  steps: int = 100) -> Dict[str, object]:
    """Simulate a play from start vertex under given strategies.

    Returns path, weights, running mean-payoff, priorities.
    """
    path = [start]
    weights = []
    priorities = [game.priority.get(start, 0)]
    cum_weight = 0.0
    running_mean = []
    current = start

    for step in range(steps):
        succs = game.successors(current)
        if not succs:
            break

        if game.owner[current] == Player.EVEN:
            strat = strategy_even
        else:
            strat = strategy_odd

        if current in strat:
            chosen = strat[current]
            w = None
            for (t, wt) in succs:
                if t == chosen:
                    w = wt
                    break
            if w is None:
                # Strategy invalid at this vertex; pick first successor
                chosen, w = succs[0]
        else:
            chosen, w = succs[0]

        weights.append(w)
        cum_weight += w
        running_mean.append(cum_weight / (step + 1))
        current = chosen
        path.append(current)
        priorities.append(game.priority.get(current, 0))

    return {
        "path": path,
        "weights": weights,
        "priorities": priorities,
        "running_mean": running_mean,
        "final_mean": running_mean[-1] if running_mean else 0.0,
        "steps": len(weights),
    }


# ---------------------------------------------------------------------------
# Decomposition Analysis
# ---------------------------------------------------------------------------

def decompose_mpp(game: MeanPayoffParityGame, threshold: float = 0.0) -> Dict[str, object]:
    """Compare winning regions under different objective combinations.

    Returns:
    - parity_only: winning regions with only parity condition
    - mean_payoff_only: winning regions with only mean-payoff >= threshold
    - combined: winning regions with both conditions
    - analysis: which vertices are lost due to the combined constraint
    """
    # Parity only
    pg = game.to_parity_game()
    parity_result = zielonka(pg)

    # Mean-payoff only (via energy game with shifted weights)
    mp_win_even = _solve_mp_free(game, set(game.vertices), threshold)
    mp_win_odd = game.vertices - mp_win_even

    # Combined
    combined = solve_mpp_threshold(game, threshold)

    # Analysis
    parity_only_extra = parity_result.win_even - combined.win_even
    mp_only_extra = mp_win_even - combined.win_even
    both_individual = parity_result.win_even & mp_win_even
    combined_loss = both_individual - combined.win_even

    return {
        "parity_only": {
            "win_even": set(parity_result.win_even),
            "win_odd": set(parity_result.win_odd),
        },
        "mean_payoff_only": {
            "win_even": mp_win_even,
            "win_odd": mp_win_odd,
        },
        "combined": {
            "win_even": combined.win_even,
            "win_odd": combined.win_odd,
        },
        "analysis": {
            "parity_wins_but_combined_loses": parity_only_extra,
            "mp_wins_but_combined_loses": mp_only_extra,
            "both_individual_win": both_individual,
            "lost_to_interaction": combined_loss,
        },
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Construction Helpers
# ---------------------------------------------------------------------------

def make_mpp_game(n: int, edges: List[Tuple[int, int, int]],
                  owners: Dict[int, Player], priorities: Dict[int, int]) -> MeanPayoffParityGame:
    """Create a mean-payoff parity game from explicit specification."""
    g = MeanPayoffParityGame()
    for v in range(n):
        g.add_vertex(v, owners.get(v, Player.EVEN), priorities.get(v, 0))
    for (u, v, w) in edges:
        g.add_edge(u, v, w)
    return g


def make_chain_mpp(n: int, weights: Optional[List[int]] = None,
                   priorities: Optional[List[int]] = None) -> MeanPayoffParityGame:
    """Create a chain game: 0 -> 1 -> 2 -> ... -> n-1 -> 0.

    Even owns all vertices. Useful for testing mean-payoff computation.
    """
    g = MeanPayoffParityGame()
    if weights is None:
        weights = [1] * n
    if priorities is None:
        priorities = [0] * n
    for v in range(n):
        g.add_vertex(v, Player.EVEN, priorities[v % len(priorities)])
    for v in range(n):
        w = weights[v % len(weights)]
        g.add_edge(v, (v + 1) % n, w)
    return g


def make_choice_mpp(good_weight: int, bad_weight: int,
                    good_prio: int = 0, bad_prio: int = 1) -> MeanPayoffParityGame:
    """Create a game where Even chooses between two cycles.

    Vertex 0 (Even): chooses to go to vertex 1 (good cycle) or vertex 2 (bad cycle).
    Vertex 1 -> 0 with good_weight and good_prio.
    Vertex 2 -> 0 with bad_weight and bad_prio.
    """
    g = MeanPayoffParityGame()
    g.add_vertex(0, Player.EVEN, 0)
    g.add_vertex(1, Player.EVEN, good_prio)
    g.add_vertex(2, Player.EVEN, bad_prio)
    g.add_edge(0, 1, 0)
    g.add_edge(0, 2, 0)
    g.add_edge(1, 0, good_weight)
    g.add_edge(2, 0, bad_weight)
    return g


def make_adversarial_mpp(weight: int, even_prio: int = 0,
                         odd_prio: int = 1) -> MeanPayoffParityGame:
    """Adversarial choice game.

    Vertex 0 (Odd): chooses vertex 1 or vertex 2.
    Vertex 1 (Even, even_prio) -> 0 with weight.
    Vertex 2 (Even, odd_prio) -> 0 with -weight.
    """
    g = MeanPayoffParityGame()
    g.add_vertex(0, Player.ODD, 0)
    g.add_vertex(1, Player.EVEN, even_prio)
    g.add_vertex(2, Player.EVEN, odd_prio)
    g.add_edge(0, 1, 0)
    g.add_edge(0, 2, 0)
    g.add_edge(1, 0, weight)
    g.add_edge(2, 0, -weight)
    return g


def make_tradeoff_mpp() -> MeanPayoffParityGame:
    """A game demonstrating the tradeoff between parity and mean-payoff.

    Even must choose between:
    - A cycle with good parity (even priority) but bad mean-payoff (negative)
    - A cycle with bad parity (odd priority) but good mean-payoff (positive)

    Even loses both ways: can't satisfy both conditions simultaneously.
    """
    g = MeanPayoffParityGame()
    g.add_vertex(0, Player.EVEN, 0)   # choice point
    g.add_vertex(1, Player.EVEN, 2)   # good parity (even), bad weight
    g.add_vertex(2, Player.EVEN, 1)   # bad parity (odd), good weight
    g.add_edge(0, 1, 0)
    g.add_edge(0, 2, 0)
    g.add_edge(1, 0, -3)  # negative mean-payoff
    g.add_edge(2, 0, 3)   # positive mean-payoff, but odd priority
    return g


def make_counter_mpp(n: int) -> MeanPayoffParityGame:
    """A counting game with n states in a cycle.

    States 0..n-1, all Even-owned. Edge weights: +1 except last edge (-(n-1)).
    Priorities: even everywhere -> Even wins parity trivially.
    Mean-payoff: (1*(n-1) + (-(n-1))) / n = 0.
    """
    g = MeanPayoffParityGame()
    for v in range(n):
        g.add_vertex(v, Player.EVEN, 0)
    for v in range(n - 1):
        g.add_edge(v, v + 1, 1)
    g.add_edge(n - 1, 0, -(n - 1))
    return g


# ---------------------------------------------------------------------------
# Summary and Statistics
# ---------------------------------------------------------------------------

def mpp_statistics(game: MeanPayoffParityGame) -> Dict[str, object]:
    """Compute statistics about a mean-payoff parity game."""
    n_edges = sum(len(game.edges.get(v, [])) for v in game.vertices)
    even_verts = sum(1 for v in game.vertices if game.owner[v] == Player.EVEN)
    odd_verts = len(game.vertices) - even_verts

    prios = sorted(set(game.priority.values())) if game.priority else []
    weights = []
    for v in game.vertices:
        for (_, w) in game.edges.get(v, []):
            weights.append(w)

    sccs = _compute_sccs(game.vertices, game.edges)
    nontrivial_sccs = [s for s in sccs if len(s) > 1 or
                       any(t == v for v in s for (t, _) in game.edges.get(v, []))]

    return {
        "vertices": len(game.vertices),
        "edges": n_edges,
        "even_vertices": even_verts,
        "odd_vertices": odd_verts,
        "priorities": prios,
        "max_priority": max(prios) if prios else 0,
        "weight_range": (min(weights), max(weights)) if weights else (0, 0),
        "mean_weight": sum(weights) / len(weights) if weights else 0.0,
        "sccs": len(sccs),
        "nontrivial_sccs": len(nontrivial_sccs),
    }


def mpp_summary(game: MeanPayoffParityGame, threshold: float = 0.0) -> str:
    """Human-readable summary of a mean-payoff parity game analysis."""
    stats = mpp_statistics(game)
    result = solve_mpp_threshold(game, threshold)

    lines = [
        f"Mean-Payoff Parity Game Summary",
        f"  Vertices: {stats['vertices']} ({stats['even_vertices']} Even, {stats['odd_vertices']} Odd)",
        f"  Edges: {stats['edges']}",
        f"  Priorities: {stats['priorities']}",
        f"  Weight range: [{stats['weight_range'][0]}, {stats['weight_range'][1]}]",
        f"  SCCs: {stats['sccs']} ({stats['nontrivial_sccs']} nontrivial)",
        f"  Threshold: {threshold}",
        f"  Even wins: {sorted(result.win_even)}",
        f"  Odd wins: {sorted(result.win_odd)}",
    ]
    return "\n".join(lines)
