"""V082: Energy Games and Mean-Payoff Games

Quantitative two-player games on finite weighted graphs where Player 0 (Even)
tries to maximize and Player 1 (Odd) tries to minimize.

Energy Games: Each edge has an integer weight. Even wins if the accumulated
energy level never drops below 0 (from some initial credit). The key question:
what is the minimum initial credit needed to win?

Mean-Payoff Games: Each edge has an integer weight. The payoff is the
long-run average weight. Even wins if the mean payoff is >= 0 (or a threshold).

Key results:
- Mean-payoff and energy objectives are determined (one player always wins).
- Positional (memoryless) strategies suffice for both players.
- Mean-payoff value = 0 iff energy game is winnable with finite credit.
- Energy-parity: combined energy + parity objectives (compose with V076).

Composes:
- V076 (parity games): ParityGame, Player, attractor, zielonka for energy-parity
- Standalone: energy games and mean-payoff games need no external dependencies

Algorithms:
1. Energy game: value iteration (Brim, Chaloupka, Zwick 2011)
2. Mean-payoff game: Karp's cycle mean + binary lifting
3. Energy-parity: Chatterjee-Doyen reduction to energy games
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
import sys
import os
from math import inf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V076_parity_games'))
from parity_games import ParityGame, Player, attractor, zielonka, ParityResult


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class WeightedGame:
    """A two-player weighted game arena.

    Attributes:
        nodes: set of node identifiers (ints)
        owner: maps node -> Player (EVEN maximizes, ODD minimizes)
        successors: maps node -> set of successor nodes
        predecessors: maps node -> set of predecessor nodes (auto-computed)
        weight: maps (src, dst) -> int weight on the edge
    """
    nodes: Set[int] = field(default_factory=set)
    owner: Dict[int, Player] = field(default_factory=dict)
    successors: Dict[int, Set[int]] = field(default_factory=dict)
    predecessors: Dict[int, Set[int]] = field(default_factory=dict)
    weight: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def add_node(self, node: int, owner: Player) -> None:
        self.nodes.add(node)
        self.owner[node] = owner
        self.successors.setdefault(node, set())
        self.predecessors.setdefault(node, set())

    def add_edge(self, src: int, dst: int, w: int) -> None:
        self.successors.setdefault(src, set()).add(dst)
        self.predecessors.setdefault(dst, set()).add(src)
        self.weight[(src, dst)] = w

    def max_weight(self) -> int:
        if not self.weight:
            return 0
        return max(self.weight.values())

    def min_weight(self) -> int:
        if not self.weight:
            return 0
        return min(self.weight.values())

    def n_edges(self) -> int:
        return len(self.weight)

    def validate(self) -> List[str]:
        issues = []
        for n in self.nodes:
            if n not in self.owner:
                issues.append(f"node {n} has no owner")
            if not self.successors.get(n):
                issues.append(f"node {n} has no successors (deadlock)")
        for (s, d) in self.weight:
            if s not in self.nodes:
                issues.append(f"edge ({s},{d}) has unknown source")
            if d not in self.nodes:
                issues.append(f"edge ({s},{d}) has unknown dest")
        return issues


@dataclass
class EnergyResult:
    """Result of energy game analysis.

    min_credit: maps node -> minimum initial credit for Even to win (inf if losing)
    winner: maps node -> Player who wins from that node (with sufficient credit)
    strategy_even: maps node -> chosen successor for Even nodes (memoryless)
    strategy_odd: maps node -> chosen successor for Odd nodes (memoryless)
    """
    min_credit: Dict[int, float]
    winner: Dict[int, Player]
    strategy_even: Dict[int, int] = field(default_factory=dict)
    strategy_odd: Dict[int, int] = field(default_factory=dict)


@dataclass
class MeanPayoffResult:
    """Result of mean-payoff game analysis.

    value: maps node -> mean-payoff value (for both players playing optimally)
    winner: maps node -> Player who wins (Even if value >= threshold)
    strategy_even: maps node -> chosen successor for Even nodes
    strategy_odd: maps node -> chosen successor for Odd nodes
    """
    value: Dict[int, float]
    winner: Dict[int, Player]
    strategy_even: Dict[int, int] = field(default_factory=dict)
    strategy_odd: Dict[int, int] = field(default_factory=dict)
    threshold: float = 0.0


@dataclass
class EnergyParityResult:
    """Result of combined energy + parity game analysis.

    min_credit: maps node -> minimum initial credit for Even to win both objectives
    winner: maps node -> Player who wins (Even must keep energy >= 0 AND win parity)
    strategy_even: maps node -> chosen successor for Even nodes
    strategy_odd: maps node -> chosen successor for Odd nodes
    """
    min_credit: Dict[int, float]
    winner: Dict[int, Player]
    strategy_even: Dict[int, int] = field(default_factory=dict)
    strategy_odd: Dict[int, int] = field(default_factory=dict)


# ===========================================================================
# Energy Games
# ===========================================================================

def solve_energy(game: WeightedGame, bound: Optional[int] = None) -> EnergyResult:
    """Solve an energy game via value iteration.

    Computes the minimum initial credit needed at each node for Even to win.
    Even wins if energy never drops below 0 along an infinite play.

    The algorithm iterates: for Even nodes, take min over successors (best choice);
    for Odd nodes, take max over successors (worst case). The credit at node v is:
        credit(v) = opt_{(v,u)} { max(0, credit(u) - weight(v,u)) }
    where opt = min for Even, max for Odd.

    Args:
        game: the weighted game arena
        bound: upper bound on credits (default: n * W where W = max |weight|).
               Nodes exceeding this are declared losing for Even.

    Returns:
        EnergyResult with min_credit, winner, and strategies.
    """
    if not game.nodes:
        return EnergyResult({}, {})

    n = len(game.nodes)
    W = max(abs(w) for w in game.weight.values()) if game.weight else 1
    if bound is None:
        bound = n * W

    # Initialize credits: 0 for all nodes
    credit = {v: 0 for v in game.nodes}
    strategy = {v: None for v in game.nodes}

    # Value iteration until fixpoint
    changed = True
    iterations = 0
    max_iterations = n * bound + n  # guaranteed convergence bound
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for v in game.nodes:
            succs = game.successors.get(v, set())
            if not succs:
                continue

            # Compute required credit for each successor
            options = []
            for u in succs:
                w = game.weight.get((v, u), 0)
                # To transition v->u with weight w, we need:
                # energy_at_v + w >= 0 AND energy_at_u >= credit(u)
                # So energy_at_v >= max(0, credit(u) - w)
                needed = max(0, credit[u] - w)
                options.append((needed, u))

            if game.owner[v] == Player.EVEN:
                # Even chooses the best (minimum credit needed)
                best_val, best_succ = min(options, key=lambda x: x[0])
            else:
                # Odd chooses the worst (maximum credit needed)
                best_val, best_succ = max(options, key=lambda x: x[0])

            if best_val > bound:
                best_val = bound + 1  # sentinel for "losing"

            if best_val != credit[v]:
                credit[v] = best_val
                strategy[v] = best_succ
                changed = True
            elif strategy[v] is None:
                strategy[v] = best_succ

    # Determine winners
    winner = {}
    strategy_even = {}
    strategy_odd = {}
    for v in game.nodes:
        if credit[v] <= bound:
            winner[v] = Player.EVEN
        else:
            winner[v] = Player.ODD
            credit[v] = inf

        # Record strategy
        succs = game.successors.get(v, set())
        if not succs:
            continue
        if strategy[v] is not None:
            if game.owner[v] == Player.EVEN:
                strategy_even[v] = strategy[v]
            else:
                strategy_odd[v] = strategy[v]
        else:
            # Fallback: pick arbitrary successor
            s = next(iter(succs))
            if game.owner[v] == Player.EVEN:
                strategy_even[v] = s
            else:
                strategy_odd[v] = s

    return EnergyResult(credit, winner, strategy_even, strategy_odd)


# ===========================================================================
# Mean-Payoff Games
# ===========================================================================

def _compute_cycle_values(game: WeightedGame) -> Dict[int, float]:
    """Compute optimal cycle values using Karp's algorithm adapted for games.

    For each node, find the value of the optimal cycle reachable from it.
    Uses a modified Bellman-Ford approach with game semantics.
    """
    n = len(game.nodes)
    if n == 0:
        return {}

    node_list = sorted(game.nodes)
    node_idx = {v: i for i, v in enumerate(node_list)}

    # Use the energy-mean-payoff connection:
    # Mean payoff value = lim_{k->inf} val_k / k
    # where val_k is the energy value for k-step games
    #
    # We use binary search on the threshold: mean payoff >= t iff
    # the energy game with weights w(e) - t is winnable.
    return {}  # placeholder, actual mean-payoff computed differently


def solve_mean_payoff(game: WeightedGame, threshold: float = 0.0,
                      epsilon: float = 1e-9) -> MeanPayoffResult:
    """Solve a mean-payoff game.

    The mean payoff of an infinite play v0, v1, v2, ... is:
        lim inf_{n->inf} (1/n) * sum_{i=0}^{n-1} weight(v_i, v_{i+1})

    Even wins if mean payoff >= threshold, Odd wins otherwise.
    Both players have optimal positional strategies.

    Algorithm: Binary search on threshold using energy game reduction.
    Mean payoff >= t iff the energy game with weights w(e) - t has finite
    winning credit for Even.

    We use a rational approach: scale weights to integers and binary search.

    Args:
        game: the weighted game arena
        threshold: Even wins if mean payoff >= threshold
        epsilon: precision for mean-payoff value computation

    Returns:
        MeanPayoffResult with values, winners, and strategies.
    """
    if not game.nodes:
        return MeanPayoffResult({}, {}, threshold=threshold)

    n = len(game.nodes)
    W = max(abs(w) for w in game.weight.values()) if game.weight else 1

    # Mean-payoff values are rational with denominator <= n
    # So values are multiples of 1/n in [-W, W]
    # We compute the value per strongly connected component via binary search

    # First, find SCCs
    sccs = _tarjan_scc(game)

    # For each SCC, compute the mean-payoff value
    scc_values = {}  # maps node -> mean-payoff value

    for scc in sccs:
        if len(scc) == 1:
            v = next(iter(scc))
            # Single node: check for self-loop
            if v in game.successors.get(v, set()):
                scc_values[v] = float(game.weight.get((v, v), 0))
            else:
                scc_values[v] = None  # transient node
        else:
            # Binary search for mean-payoff value of this SCC
            val = _scc_mean_payoff(game, scc, W, n, epsilon)
            for v in scc:
                scc_values[v] = val

    # For transient nodes, compute value based on reachable SCCs
    value = {}
    strategy_even = {}
    strategy_odd = {}

    # Process in reverse topological order (SCCs already in reverse topo from Tarjan)
    for scc in sccs:
        for v in scc:
            if scc_values.get(v) is not None and len(scc) > 1:
                value[v] = scc_values[v]
            elif scc_values.get(v) is not None and len(scc) == 1 and v in game.successors.get(v, set()):
                value[v] = scc_values[v]
            else:
                # Transient: value depends on successors
                succs = game.successors.get(v, set())
                if not succs:
                    value[v] = -inf if game.owner[v] == Player.EVEN else inf
                    continue
                succ_vals = []
                for u in succs:
                    if u in value:
                        succ_vals.append((value[u], u))
                if not succ_vals:
                    value[v] = 0.0
                    continue
                if game.owner[v] == Player.EVEN:
                    best_val, best_succ = max(succ_vals, key=lambda x: x[0])
                    strategy_even[v] = best_succ
                else:
                    best_val, best_succ = min(succ_vals, key=lambda x: x[0])
                    strategy_odd[v] = best_succ
                value[v] = best_val

    # Determine winners and extract strategies for SCC nodes
    winner = {}
    for v in game.nodes:
        if v not in value:
            value[v] = 0.0
        if value[v] >= threshold - epsilon:
            winner[v] = Player.EVEN
        else:
            winner[v] = Player.ODD

    # Extract strategies within SCCs
    for scc in sccs:
        if len(scc) <= 1:
            v = list(scc)[0]
            succs = game.successors.get(v, set())
            if succs:
                if game.owner[v] == Player.EVEN:
                    if v not in strategy_even:
                        # Pick successor in winning region or with best value
                        best = max(succs, key=lambda u: value.get(u, -inf))
                        strategy_even[v] = best
                else:
                    if v not in strategy_odd:
                        best = min(succs, key=lambda u: value.get(u, inf))
                        strategy_odd[v] = best
            continue

        scc_val = scc_values.get(next(iter(scc)), 0.0)
        for v in scc:
            succs = game.successors.get(v, set())
            if not succs:
                continue
            if game.owner[v] == Player.EVEN:
                if v not in strategy_even:
                    # Stay in SCC if value is good
                    scc_succs = [u for u in succs if u in scc]
                    if scc_succs:
                        strategy_even[v] = scc_succs[0]
                    else:
                        strategy_even[v] = next(iter(succs))
            else:
                if v not in strategy_odd:
                    scc_succs = [u for u in succs if u in scc]
                    if scc_succs:
                        strategy_odd[v] = scc_succs[0]
                    else:
                        strategy_odd[v] = next(iter(succs))

    return MeanPayoffResult(value, winner, strategy_even, strategy_odd, threshold)


def _tarjan_scc(game: WeightedGame) -> List[Set[int]]:
    """Tarjan's SCC algorithm. Returns SCCs in reverse topological order."""
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    result = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in game.successors.get(v, set()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.add(w)
                if w == v:
                    break
            result.append(scc)

    for v in sorted(game.nodes):
        if v not in index:
            strongconnect(v)

    return result


def _scc_mean_payoff(game: WeightedGame, scc: Set[int],
                     W: int, n: int, epsilon: float) -> float:
    """Compute mean-payoff value for a single SCC using binary search + energy.

    The mean-payoff value v* satisfies:
    - Energy game with weights w(e) - v* is won by Even (mean payoff >= v*)
    - Energy game with weights w(e) - (v* + eps) is lost by Even

    Since values are multiples of 1/n, we search over rational values.
    """
    # Build sub-game restricted to SCC
    sub = WeightedGame()
    for v in scc:
        sub.add_node(v, game.owner[v])
    for v in scc:
        for u in game.successors.get(v, set()):
            if u in scc:
                sub.add_edge(v, u, game.weight.get((v, u), 0))

    # Binary search for the mean-payoff value
    lo = -W
    hi = W

    # Use n * denominator precision
    for _ in range(60):  # enough iterations for any practical precision
        mid = (lo + hi) / 2.0

        # Shift weights: w'(e) = w(e) - mid, scaled to integers
        # We test: can Even win the energy game with shifted weights?
        shifted = WeightedGame()
        for v in sub.nodes:
            shifted.add_node(v, sub.owner[v])
        for (s, d), w in sub.weight.items():
            # Use scaled integer weights for precision
            shifted.add_edge(s, d, 0)  # placeholder
            shifted.weight[(s, d)] = w  # we'll adjust in the energy test

        # Direct energy test with fractional threshold
        even_wins = _energy_test_shifted(sub, mid)

        if even_wins:
            lo = mid  # Even can achieve at least mid
        else:
            hi = mid

        if hi - lo < epsilon / 2:
            break

    return (lo + hi) / 2.0


def _energy_test_shifted(game: WeightedGame, shift: float) -> bool:
    """Test if Even wins the energy game with weights w(e) - shift.

    Returns True if Even can win from ALL nodes in the game (since this is
    an SCC, if Even can win from any node, it can eventually reach any other).
    """
    n = len(game.nodes)
    if n == 0:
        return True

    W = max(abs(game.weight.get(e, 0) - shift) for e in game.weight) if game.weight else 1
    bound = int(n * (W + 1)) + 1

    credit = {v: 0 for v in game.nodes}

    changed = True
    iterations = 0
    max_iter = n * bound + n
    while changed and iterations < max_iter:
        changed = False
        iterations += 1
        for v in game.nodes:
            succs = game.successors.get(v, set())
            if not succs:
                continue

            options = []
            for u in succs:
                w = game.weight.get((v, u), 0) - shift
                needed = max(0, credit[u] - w)
                options.append(needed)

            if game.owner[v] == Player.EVEN:
                best = min(options)
            else:
                best = max(options)

            # Cap at bound
            if best > bound:
                best = bound + 1

            if abs(best - credit[v]) > 1e-12:
                credit[v] = best
                changed = True

    # Even wins if all credits are finite (within bound)
    return all(credit[v] <= bound for v in game.nodes)


# ===========================================================================
# Energy-Parity Games (Composition with V076)
# ===========================================================================

@dataclass
class WeightedParityGame:
    """A weighted parity game: nodes have both priorities and edge weights.

    Even wins if: energy stays >= 0 AND highest infinitely-visited priority is even.
    """
    nodes: Set[int] = field(default_factory=set)
    owner: Dict[int, Player] = field(default_factory=dict)
    priority: Dict[int, int] = field(default_factory=dict)
    successors: Dict[int, Set[int]] = field(default_factory=dict)
    predecessors: Dict[int, Set[int]] = field(default_factory=dict)
    weight: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def add_node(self, node: int, owner: Player, priority: int) -> None:
        self.nodes.add(node)
        self.owner[node] = owner
        self.priority[node] = priority
        self.successors.setdefault(node, set())
        self.predecessors.setdefault(node, set())

    def add_edge(self, src: int, dst: int, w: int) -> None:
        self.successors.setdefault(src, set()).add(dst)
        self.predecessors.setdefault(dst, set()).add(src)
        self.weight[(src, dst)] = w

    def to_parity_game(self) -> ParityGame:
        """Extract the parity game (ignoring weights)."""
        pg = ParityGame()
        for n in self.nodes:
            pg.add_node(n, self.owner[n], self.priority[n])
        for n in self.nodes:
            for s in self.successors.get(n, set()):
                pg.add_edge(n, s)
        return pg

    def to_weighted_game(self) -> WeightedGame:
        """Extract the weighted game (ignoring priorities)."""
        wg = WeightedGame()
        for n in self.nodes:
            wg.add_node(n, self.owner[n])
        for (s, d), w in self.weight.items():
            wg.add_edge(s, d, w)
        return wg

    def max_priority(self) -> int:
        if not self.nodes:
            return -1
        return max(self.priority[n] for n in self.nodes)


def solve_energy_parity(game: WeightedParityGame,
                        bound: Optional[int] = None) -> EnergyParityResult:
    """Solve a combined energy-parity game.

    Even wins if BOTH:
    1. The energy level never drops below 0 (energy objective)
    2. The maximum priority seen infinitely often is even (parity objective)

    Algorithm: Iterative approach based on Chatterjee-Doyen (2012).
    We solve energy games restricted to parity-winning regions.

    The approach:
    1. Solve the parity game (V076 Zielonka) to find parity-winning regions
    2. Restrict to Even's parity-winning region
    3. Solve energy game on the restricted arena
    4. Nodes where Even wins both = Even's energy-parity winning region

    This is sound but not complete for the full energy-parity problem.
    For completeness, we iterate: remove losing nodes and re-solve parity.

    Args:
        game: WeightedParityGame
        bound: energy upper bound (default: n * W)

    Returns:
        EnergyParityResult
    """
    if not game.nodes:
        return EnergyParityResult({}, {})

    n = len(game.nodes)
    W = max(abs(w) for w in game.weight.values()) if game.weight else 1
    if bound is None:
        bound = n * W

    remaining = set(game.nodes)
    even_winning = set()
    strategy_even = {}
    strategy_odd = {}

    # Iterative refinement
    for _ in range(n + 1):
        if not remaining:
            break

        # Step 1: Solve parity game on remaining nodes
        pg = ParityGame()
        for v in remaining:
            pg.add_node(v, game.owner[v], game.priority[v])
        for v in remaining:
            for u in game.successors.get(v, set()):
                if u in remaining:
                    pg.add_edge(v, u)

        # Check all nodes have successors
        dead = {v for v in remaining if not pg.successors.get(v)}
        if dead:
            remaining -= dead
            continue

        parity_result = zielonka(pg)
        parity_even = parity_result.win_even

        if not parity_even:
            break  # No parity-winning nodes left

        # Step 2: Solve energy game restricted to parity-winning region
        wg = WeightedGame()
        for v in parity_even:
            wg.add_node(v, game.owner[v])
        for v in parity_even:
            for u in game.successors.get(v, set()):
                if u in parity_even:
                    wg.add_edge(v, u, game.weight.get((v, u), 0))

        # Check all nodes have successors in restricted game
        dead = {v for v in parity_even if not wg.successors.get(v)}
        if dead:
            remaining -= dead
            continue

        energy_result = solve_energy(wg, bound)
        energy_even = {v for v in parity_even if energy_result.winner.get(v) == Player.EVEN}

        if energy_even == parity_even:
            # All parity-winning nodes also win energy
            even_winning |= energy_even
            strategy_even.update(energy_result.strategy_even)
            strategy_odd.update(energy_result.strategy_odd)
            break
        elif not energy_even:
            # No energy-winning nodes in parity region -> remove and retry
            remaining -= parity_even
        else:
            # Some nodes win both, some don't -> add winners, remove losers, retry
            even_winning |= energy_even
            strategy_even.update({v: s for v, s in energy_result.strategy_even.items()
                                  if v in energy_even})
            strategy_odd.update({v: s for v, s in energy_result.strategy_odd.items()
                                 if v in energy_even})
            remaining -= parity_even

    # Build final result
    min_credit = {}
    winner = {}
    for v in game.nodes:
        if v in even_winning:
            winner[v] = Player.EVEN
            # Re-solve energy for exact credits
            min_credit[v] = 0  # Will be updated below
        else:
            winner[v] = Player.ODD
            min_credit[v] = inf

    # Get exact credits for winning region
    if even_winning:
        wg = WeightedGame()
        for v in even_winning:
            wg.add_node(v, game.owner[v])
        for v in even_winning:
            for u in game.successors.get(v, set()):
                if u in even_winning:
                    wg.add_edge(v, u, game.weight.get((v, u), 0))

        dead = {v for v in even_winning if not wg.successors.get(v)}
        live = even_winning - dead
        if live:
            wg2 = WeightedGame()
            for v in live:
                wg2.add_node(v, game.owner[v])
            for v in live:
                for u in game.successors.get(v, set()):
                    if u in live:
                        wg2.add_edge(v, u, game.weight.get((v, u), 0))
            if all(wg2.successors.get(v) for v in live):
                er = solve_energy(wg2, bound)
                for v in live:
                    min_credit[v] = er.min_credit.get(v, inf)
                    if er.min_credit.get(v, inf) > bound:
                        min_credit[v] = inf
                        winner[v] = Player.ODD

        for v in dead:
            winner[v] = Player.ODD
            min_credit[v] = inf

    return EnergyParityResult(min_credit, winner, strategy_even, strategy_odd)


# ===========================================================================
# Convenience constructors and utilities
# ===========================================================================

def make_weighted_game(nodes: List[Tuple[int, int]],
                       edges: List[Tuple[int, int, int]]) -> WeightedGame:
    """Create a weighted game from node list and edge list.

    Args:
        nodes: list of (node_id, owner) where owner: 0=EVEN, 1=ODD
        edges: list of (src, dst, weight)
    """
    game = WeightedGame()
    for node_id, owner_int in nodes:
        game.add_node(node_id, Player.EVEN if owner_int == 0 else Player.ODD)
    for src, dst, w in edges:
        game.add_edge(src, dst, w)
    return game


def make_weighted_parity_game(nodes: List[Tuple[int, int, int]],
                              edges: List[Tuple[int, int, int]]) -> WeightedParityGame:
    """Create a weighted parity game.

    Args:
        nodes: list of (node_id, owner, priority) where owner: 0=EVEN, 1=ODD
        edges: list of (src, dst, weight)
    """
    game = WeightedParityGame()
    for node_id, owner_int, prio in nodes:
        game.add_node(node_id, Player.EVEN if owner_int == 0 else Player.ODD, prio)
    for src, dst, w in edges:
        game.add_edge(src, dst, w)
    return game


def energy_to_mean_payoff(game: WeightedGame) -> MeanPayoffResult:
    """Solve the mean-payoff game using the energy-mean-payoff connection.

    The mean payoff value v* >= 0 iff Even wins the energy game.
    This is a convenience wrapper around solve_mean_payoff.
    """
    return solve_mean_payoff(game)


def mean_payoff_to_energy(game: WeightedGame, shift: float = 0.0) -> EnergyResult:
    """Convert mean-payoff problem to energy game by shifting weights.

    If mean payoff >= shift, then energy game with w'(e) = w(e) - shift is winnable.
    """
    shifted = WeightedGame()
    for v in game.nodes:
        shifted.add_node(v, game.owner[v])
    for (s, d), w in game.weight.items():
        shifted.add_edge(s, d, int(w - shift))
    return solve_energy(shifted)


def verify_energy_strategy(game: WeightedGame, result: EnergyResult,
                           max_steps: int = 1000) -> Dict:
    """Verify an energy game strategy by simulation.

    Returns dict with 'valid', 'min_energy_seen', 'steps_simulated'.
    """
    report = {'valid': True, 'node_reports': {}}

    for start in game.nodes:
        if result.winner.get(start) != Player.EVEN:
            continue

        credit = result.min_credit.get(start, 0)
        if credit == inf:
            continue

        energy = credit
        min_energy = energy
        v = start
        visited = []

        for step in range(max_steps):
            visited.append(v)
            if game.owner[v] == Player.EVEN:
                nxt = result.strategy_even.get(v)
            else:
                nxt = result.strategy_odd.get(v)

            if nxt is None:
                succs = game.successors.get(v, set())
                if succs:
                    nxt = next(iter(succs))
                else:
                    break

            w = game.weight.get((v, nxt), 0)
            energy += w
            min_energy = min(min_energy, energy)

            if energy < 0:
                report['valid'] = False
                report['node_reports'][start] = {
                    'failed_at_step': step,
                    'energy': energy,
                    'node': v,
                    'next': nxt,
                    'weight': w
                }
                break

            v = nxt

            # Check for cycle (we've returned to a visited state)
            if v in visited[:-1]:
                break

        report['node_reports'][start] = {
            'min_energy': min_energy,
            'steps': len(visited),
            'final_energy': energy
        }

    return report


def verify_mean_payoff_strategy(game: WeightedGame, result: MeanPayoffResult,
                                max_steps: int = 10000) -> Dict:
    """Verify a mean-payoff strategy by simulation.

    Returns dict with per-node average weight along strategy.
    """
    report = {'node_reports': {}}

    for start in game.nodes:
        v = start
        total_weight = 0
        steps = 0
        visited_at = {}

        for step in range(max_steps):
            if v in visited_at:
                # Found a cycle: compute exact cycle mean
                cycle_start = visited_at[v]
                cycle_len = step - cycle_start
                if cycle_len > 0:
                    report['node_reports'][start] = {
                        'cycle_mean': total_weight / step if step > 0 else 0,
                        'cycle_len': cycle_len,
                        'steps': step
                    }
                    break
            visited_at[v] = step

            if game.owner[v] == Player.EVEN:
                nxt = result.strategy_even.get(v)
            else:
                nxt = result.strategy_odd.get(v)

            if nxt is None:
                succs = game.successors.get(v, set())
                if succs:
                    nxt = next(iter(succs))
                else:
                    break

            w = game.weight.get((v, nxt), 0)
            total_weight += w
            steps += 1
            v = nxt

        if start not in report['node_reports']:
            report['node_reports'][start] = {
                'avg_weight': total_weight / max(steps, 1),
                'steps': steps
            }

    return report


def compare_energy_mean_payoff(game: WeightedGame) -> Dict:
    """Compare energy and mean-payoff results on the same game.

    Shows the connection: mean payoff >= 0 iff energy game winnable.
    """
    energy = solve_energy(game)
    mean_payoff = solve_mean_payoff(game)

    agreement = 0
    disagreement = 0
    for v in game.nodes:
        e_winner = energy.winner.get(v)
        m_winner = mean_payoff.winner.get(v)
        if e_winner == m_winner:
            agreement += 1
        else:
            disagreement += 1

    return {
        'energy_result': energy,
        'mean_payoff_result': mean_payoff,
        'agreement': agreement,
        'disagreement': disagreement,
        'n_nodes': len(game.nodes),
        'connection_holds': disagreement == 0
    }


def parity_game_to_weighted(pg: ParityGame, weight_fn=None) -> WeightedGame:
    """Convert a V076 ParityGame to a WeightedGame.

    Args:
        pg: ParityGame from V076
        weight_fn: optional function (src, dst, pg) -> weight.
                   Default: all weights = 0.
    """
    wg = WeightedGame()
    for n in pg.nodes:
        wg.add_node(n, pg.owner[n])
    for n in pg.nodes:
        for s in pg.successors.get(n, set()):
            w = weight_fn(n, s, pg) if weight_fn else 0
            wg.add_edge(n, s, w)
    return wg


def weighted_game_summary(game: WeightedGame) -> str:
    """Human-readable summary of a weighted game."""
    lines = [f"WeightedGame: {len(game.nodes)} nodes, {game.n_edges()} edges"]
    lines.append(f"  Weight range: [{game.min_weight()}, {game.max_weight()}]")
    even_count = sum(1 for v in game.nodes if game.owner[v] == Player.EVEN)
    lines.append(f"  Even nodes: {even_count}, Odd nodes: {len(game.nodes) - even_count}")
    return "\n".join(lines)
