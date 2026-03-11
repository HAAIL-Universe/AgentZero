"""V167: Concurrent Stochastic Games

Two-player simultaneous-move stochastic games with parity winning conditions.

Unlike turn-based games (V156, V165) where players alternate moves,
concurrent games have BOTH players choosing actions simultaneously at each state.
The next state is determined by a probability distribution that depends on
BOTH players' action choices.

Key differences from turn-based:
- No vertex ownership -- both players act at every vertex
- Strategies are MIXED (probability distributions over actions), not pure
- Solving requires linear programming (minimax), not just attractor computation
- Almost-sure winning is strictly harder than in turn-based setting

Composes: V165 (stochastic parity), V156 (parity games)

Uses numpy/scipy for LP solving (permitted by CLAUDE.md for numerical computation).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Tuple, Optional, FrozenSet
import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class ConcurrentVertex:
    """A vertex in a concurrent stochastic game."""
    id: int
    priority: int
    actions_even: List[str]  # actions available to Even
    actions_odd: List[str]   # actions available to Odd
    # transition: (action_even, action_odd) -> distribution over successors
    # distribution is Dict[int, float] mapping successor vertex -> probability
    transitions: Dict[Tuple[str, str], Dict[int, float]] = field(default_factory=dict)


class CSGResult:
    """Result of solving a concurrent stochastic game."""
    def __init__(self):
        self.win_even_as: Set[int] = set()   # almost-sure winning for Even
        self.win_odd_as: Set[int] = set()    # almost-sure winning for Odd
        self.win_even_pp: Set[int] = set()   # positive-prob winning for Even
        self.win_odd_pp: Set[int] = set()    # positive-prob winning for Odd
        # Mixed strategies: vertex -> probability distribution over actions
        self.strategy_even_as: Dict[int, Dict[str, float]] = {}
        self.strategy_odd_as: Dict[int, Dict[str, float]] = {}
        self.strategy_even_pp: Dict[int, Dict[str, float]] = {}
        self.strategy_odd_pp: Dict[int, Dict[str, float]] = {}


class ConcurrentStochasticGame:
    """A two-player concurrent stochastic game with parity condition.

    At each vertex, BOTH players simultaneously choose an action.
    The pair of actions determines a probability distribution over successors.
    Even wins if the highest priority visited infinitely often is even.
    """

    def __init__(self):
        self.vertices: Dict[int, ConcurrentVertex] = {}

    def add_vertex(self, v: int, priority: int,
                   actions_even: List[str], actions_odd: List[str]) -> None:
        self.vertices[v] = ConcurrentVertex(
            id=v, priority=priority,
            actions_even=list(actions_even),
            actions_odd=list(actions_odd)
        )

    def add_transition(self, v: int, act_even: str, act_odd: str,
                       distribution: Dict[int, float]) -> None:
        """Add transition from vertex v under action pair (act_even, act_odd)."""
        vertex = self.vertices[v]
        assert act_even in vertex.actions_even, f"Invalid Even action {act_even} at {v}"
        assert act_odd in vertex.actions_odd, f"Invalid Odd action {act_odd} at {v}"
        total = sum(distribution.values())
        assert abs(total - 1.0) < 1e-9, f"Probabilities sum to {total}, not 1.0"
        for succ in distribution:
            assert succ in self.vertices, f"Successor {succ} not in game"
        vertex.transitions[(act_even, act_odd)] = dict(distribution)

    def successors(self, v: int) -> Set[int]:
        """All possible successors from vertex v (union over all action pairs)."""
        result = set()
        for dist in self.vertices[v].transitions.values():
            result.update(dist.keys())
        return result

    def validate(self) -> List[str]:
        """Validate game structure."""
        errors = []
        for v, vertex in self.vertices.items():
            for ae in vertex.actions_even:
                for ao in vertex.actions_odd:
                    if (ae, ao) not in vertex.transitions:
                        errors.append(f"Missing transition at {v} for ({ae}, {ao})")
            for (ae, ao), dist in vertex.transitions.items():
                total = sum(dist.values())
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"Probabilities sum to {total} at {v}, ({ae},{ao})")
                for succ in dist:
                    if succ not in self.vertices:
                        errors.append(f"Unknown successor {succ} at {v}")
        return errors

    def vertex_set(self) -> Set[int]:
        return set(self.vertices.keys())

    def subgame(self, verts: Set[int]) -> ConcurrentStochasticGame:
        """Restrict to a subset of vertices (normalizes probabilities)."""
        g = ConcurrentStochasticGame()
        for v in verts:
            if v in self.vertices:
                vert = self.vertices[v]
                g.add_vertex(v, vert.priority, vert.actions_even, vert.actions_odd)
                for (ae, ao), dist in vert.transitions.items():
                    restricted = {s: p for s, p in dist.items() if s in verts}
                    if restricted:
                        total = sum(restricted.values())
                        if total > 0:
                            normalized = {s: p / total for s, p in restricted.items()}
                            g.vertices[v].transitions[(ae, ao)] = normalized
        return g


# ---------------------------------------------------------------------------
# Matrix game solving (core LP for concurrent games)
# ---------------------------------------------------------------------------

def solve_matrix_game(matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve a zero-sum matrix game using LP (minimax theorem).

    matrix[i][j] = payoff to the row player (Even) when Even plays i, Odd plays j.

    Returns: (value, even_strategy, odd_strategy)
    - value: game value (expected payoff under optimal play)
    - even_strategy: probability distribution over Even's actions (maximizer)
    - odd_strategy: probability distribution over Odd's actions (minimizer)
    """
    m, n = matrix.shape  # m = Even actions, n = Odd actions

    if m == 0 or n == 0:
        return 0.0, np.array([]), np.array([])

    # Handle 1x1 trivially
    if m == 1 and n == 1:
        return float(matrix[0, 0]), np.array([1.0]), np.array([1.0])

    # Shift matrix to ensure all values positive (LP requirement)
    shift = abs(matrix.min()) + 1.0

    # Even (row player) maximizes: max v s.t. A^T x >= v*1, x >= 0, sum(x) = 1
    # Equivalent LP for Even: max v
    # Rewrite as minimization for linprog: min -v
    # Variables: [x_0, ..., x_{m-1}, v]
    # Constraints: for each j: sum_i (a_ij + shift) * x_i >= v
    #   => sum_i (a_ij + shift) * x_i - v >= 0
    #   => -sum_i (a_ij + shift) * x_i + v <= 0
    # sum(x_i) = 1, x_i >= 0

    shifted = matrix + shift

    # linprog minimizes c^T x
    c = np.zeros(m + 1)
    c[m] = -1.0  # minimize -v = maximize v

    # Inequality: A_ub @ x <= b_ub
    # For each Odd action j: -sum_i shifted[i,j]*x_i + v <= 0
    A_ub = np.zeros((n, m + 1))
    for j in range(n):
        for i in range(m):
            A_ub[j, i] = -shifted[i, j]
        A_ub[j, m] = 1.0
    b_ub = np.zeros(n)

    # Equality: sum(x_i) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: x_i >= 0, v unbounded
    bounds = [(0.0, None)] * m + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if not result.success:
        # Fallback: uniform strategy
        even_strat = np.ones(m) / m
        odd_strat = np.ones(n) / n
        val = float(even_strat @ matrix @ odd_strat)
        return val, even_strat, odd_strat

    even_strat = result.x[:m]
    val = result.x[m] - shift  # undo shift

    # Now solve for Odd (column player): min w s.t. A y <= w*1, y >= 0, sum(y) = 1
    # linprog: min w
    c2 = np.zeros(n + 1)
    c2[n] = 1.0  # minimize w

    A_ub2 = np.zeros((m, n + 1))
    for i in range(m):
        for j in range(n):
            A_ub2[i, j] = shifted[i, j]
        A_ub2[i, n] = -1.0
    b_ub2 = np.zeros(m)

    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1.0
    b_eq2 = np.array([1.0])

    bounds2 = [(0.0, None)] * n + [(None, None)]

    result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                      bounds=bounds2, method='highs')

    if result2.success:
        odd_strat = result2.x[:n]
    else:
        odd_strat = np.ones(n) / n

    # Clean up small negatives from numerical noise
    even_strat = np.maximum(even_strat, 0.0)
    odd_strat = np.maximum(odd_strat, 0.0)
    if even_strat.sum() > 0:
        even_strat /= even_strat.sum()
    if odd_strat.sum() > 0:
        odd_strat /= odd_strat.sum()

    return val, even_strat, odd_strat


# ---------------------------------------------------------------------------
# Concurrent reachability / safety (base for parity reduction)
# ---------------------------------------------------------------------------

def concurrent_attractor_value(
    game: ConcurrentStochasticGame,
    target: Set[int],
    player_even_maximizes: bool,
    restrict: Optional[Set[int]] = None,
) -> Dict[int, float]:
    """Compute value of reaching target in a concurrent reachability game.

    Returns dict mapping vertex -> probability of reaching target under optimal play.
    player_even_maximizes: if True, Even tries to maximize reach probability;
                           if False, Even tries to minimize (Odd maximizes).

    Uses value iteration on the concurrent game.
    """
    verts = restrict if restrict is not None else game.vertex_set()
    # Initialize: target vertices have value 1, others 0
    values: Dict[int, float] = {}
    for v in verts:
        values[v] = 1.0 if v in target else 0.0

    for _iteration in range(200):
        new_values = dict(values)
        max_change = 0.0
        for v in verts:
            if v in target:
                continue
            vertex = game.vertices[v]
            if not vertex.transitions:
                continue
            # Build payoff matrix: entry[i][j] = expected value under (ae_i, ao_j)
            ae_list = vertex.actions_even
            ao_list = vertex.actions_odd
            m, n = len(ae_list), len(ao_list)
            if m == 0 or n == 0:
                continue
            matrix = np.zeros((m, n))
            for i, ae in enumerate(ae_list):
                for j, ao in enumerate(ao_list):
                    dist = vertex.transitions.get((ae, ao), {})
                    expected = sum(
                        prob * values.get(succ, 0.0)
                        for succ, prob in dist.items()
                    )
                    matrix[i, j] = expected

            if player_even_maximizes:
                val, _, _ = solve_matrix_game(matrix)
            else:
                # Even minimizes = Odd maximizes = negate and solve
                val_neg, _, _ = solve_matrix_game(-matrix)
                val = -val_neg

            new_values[v] = val
            max_change = max(max_change, abs(val - values[v]))

        values = new_values
        if max_change < 1e-10:
            break

    return values


def concurrent_almost_sure_reach(
    game: ConcurrentStochasticGame,
    target: Set[int],
    avoid: Optional[Set[int]] = None,
) -> Set[int]:
    """Vertices from which Even can reach target almost-surely.

    Even maximizes probability; if value == 1.0 it's almost-sure.
    """
    restrict = game.vertex_set()
    if avoid:
        restrict = restrict - avoid
    values = concurrent_attractor_value(game, target & restrict, True, restrict)
    return {v for v, val in values.items() if val > 1.0 - 1e-8}


def concurrent_positive_prob_reach(
    game: ConcurrentStochasticGame,
    target: Set[int],
    avoid: Optional[Set[int]] = None,
) -> Set[int]:
    """Vertices from which Even can reach target with positive probability."""
    restrict = game.vertex_set()
    if avoid:
        restrict = restrict - avoid
    values = concurrent_attractor_value(game, target & restrict, True, restrict)
    return {v for v, val in values.items() if val > 1e-8}


# ---------------------------------------------------------------------------
# Concurrent parity game solver
# ---------------------------------------------------------------------------

def solve_concurrent_parity_almost_sure(
    game: ConcurrentStochasticGame,
) -> Tuple[Set[int], Set[int], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """Solve concurrent stochastic parity game for almost-sure winning.

    Uses the McNaughton-Zielonka style recursive decomposition adapted
    for concurrent games. At each step, uses LP-based concurrent attractor
    instead of the standard deterministic attractor.

    Returns: (win_even, win_odd, strategy_even, strategy_odd)
    """
    verts = game.vertex_set()
    if not verts:
        return set(), set(), {}, {}

    # Find max priority
    max_prio = max(game.vertices[v].priority for v in verts)
    prio_verts = {v for v in verts if game.vertices[v].priority == max_prio}
    player_even = (max_prio % 2 == 0)  # Even wins if max prio is even

    if player_even:
        # Max priority is even -- good for Even
        # Even tries to visit prio_verts infinitely often (almost-surely)
        # Compute vertices from which Even can force reaching prio_verts almost-surely
        reach_as = concurrent_almost_sure_reach(game, prio_verts)

        # Vertices NOT in reach_as: Odd wins (Even can't reach max-prio AS)
        complement = verts - reach_as
        if complement:
            sub = game.subgame(complement)
            w0_sub, w1_sub, s0_sub, s1_sub = solve_concurrent_parity_almost_sure(sub)
            # Odd wins complement's Odd region + rest where Even can't reach
            odd_region = w1_sub | (complement - w0_sub)
        else:
            w0_sub, w1_sub = set(), set()
            s0_sub, s1_sub = {}, {}
            odd_region = set()

        if odd_region:
            # Remove Odd's region and recurse
            remaining = verts - odd_region
            if remaining:
                sub2 = game.subgame(remaining)
                w0_r, w1_r, s0_r, s1_r = solve_concurrent_parity_almost_sure(sub2)
                even_win = w0_r
                odd_win = odd_region | w1_r
                strat_even = {**s0_r}
                strat_odd = {**s1_sub, **s1_r}
            else:
                even_win = set()
                odd_win = odd_region
                strat_even = {}
                strat_odd = dict(s1_sub)
        else:
            even_win = verts
            odd_win = set()
            strat_even = {}
            strat_odd = {}

    else:
        # Max priority is odd -- good for Odd
        # Odd wants to visit prio_verts infinitely; Even wants to avoid them
        # Even can avoid AS = Even can force NOT reaching prio_verts
        # Complement: vertices from which Even CANNOT avoid prio_verts
        reach_as_odd = concurrent_almost_sure_reach(
            _swap_players(game), prio_verts
        )
        # reach_as_odd = vertices where "swapped Even" can reach prio_verts AS
        # In original game, this is where Odd can force reaching prio_verts AS

        complement = verts - reach_as_odd
        if complement:
            sub = game.subgame(complement)
            w0_sub, w1_sub, s0_sub, s1_sub = solve_concurrent_parity_almost_sure(sub)
            even_region = w0_sub | (complement - w1_sub)
        else:
            w0_sub, w1_sub = set(), set()
            s0_sub, s1_sub = {}, {}
            even_region = set()

        if even_region:
            remaining = verts - even_region
            if remaining:
                sub2 = game.subgame(remaining)
                w0_r, w1_r, s0_r, s1_r = solve_concurrent_parity_almost_sure(sub2)
                odd_win = w1_r
                even_win = even_region | w0_r
                strat_odd = {**s1_r}
                strat_even = {**s0_sub, **s0_r}
            else:
                odd_win = set()
                even_win = even_region
                strat_odd = {}
                strat_even = dict(s0_sub)
        else:
            odd_win = verts
            even_win = set()
            strat_even = {}
            strat_odd = {}

    # Compute mixed strategies for vertices where we don't have them yet
    _compute_mixed_strategies(game, even_win, strat_even, maximize=True)
    _compute_mixed_strategies(game, odd_win, strat_odd, maximize=False)

    return even_win, odd_win, strat_even, strat_odd


def _compute_parity_value(
    game: ConcurrentStochasticGame,
    max_iterations: int = 100,
) -> Dict[int, float]:
    """Compute Even's probability of winning the parity game from each vertex.

    Uses the Zielonka-style recursive value computation:
    The value of a parity game decomposes by max priority.

    Returns dict mapping vertex -> probability Even wins under optimal play.
    """
    verts = game.vertex_set()
    if not verts:
        return {}

    max_prio = max(game.vertices[v].priority for v in verts)
    prio_verts = {v for v in verts if game.vertices[v].priority == max_prio}
    player_even = (max_prio % 2 == 0)

    if len(set(game.vertices[v].priority for v in verts)) == 1:
        # All same priority: if even, Even wins everywhere; if odd, Odd wins
        return {v: (1.0 if player_even else 0.0) for v in verts}

    # Compute game value via value iteration on parity condition
    # For a parity game with max priority p:
    # If p is even: Even wants to visit priority-p vertices; value = prob of infinitely often
    # If p is odd: Odd wants to visit priority-p vertices; Even wants to avoid

    # Strategy: compute the value of the game where visiting max-priority
    # vertices leads to "winning/losing" depending on parity,
    # and non-max vertices recurse to the subgame without max-priority vertices.

    # First compute values in the subgame without max-priority vertices
    non_max_verts = verts - prio_verts
    if non_max_verts:
        sub = game.subgame(non_max_verts)
        sub_values = _compute_parity_value(sub)
    else:
        sub_values = {}

    # Now compute full game values using value iteration
    # At max-priority vertices: if even, value starts at 1; if odd, value starts at 0
    # At other vertices: start with sub_values
    values = {}
    for v in verts:
        if v in prio_verts:
            values[v] = 1.0 if player_even else 0.0
        elif v in sub_values:
            values[v] = sub_values[v]
        else:
            values[v] = 0.5  # initial guess

    # Value iteration: each vertex plays optimally in the concurrent game
    for _iter in range(max_iterations):
        new_values = dict(values)
        max_change = 0.0
        for v in verts:
            vertex = game.vertices[v]
            if not vertex.transitions:
                continue
            ae_list = vertex.actions_even
            ao_list = vertex.actions_odd
            m, n = len(ae_list), len(ao_list)
            if m == 0 or n == 0:
                continue

            matrix = np.zeros((m, n))
            for i, ae in enumerate(ae_list):
                for j, ao in enumerate(ao_list):
                    dist = vertex.transitions.get((ae, ao), {})
                    expected = sum(
                        prob * values.get(succ, 0.0)
                        for succ, prob in dist.items()
                    )
                    matrix[i, j] = expected

            # Even maximizes, so game value for Even
            val, _, _ = solve_matrix_game(matrix)
            new_values[v] = max(0.0, min(1.0, val))
            max_change = max(max_change, abs(new_values[v] - values[v]))

        values = new_values
        if max_change < 1e-10:
            break

    return values


def solve_concurrent_parity_positive_prob(
    game: ConcurrentStochasticGame,
) -> Tuple[Set[int], Set[int], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """Solve concurrent stochastic parity game for positive-probability winning.

    In concurrent games, PP winning regions may OVERLAP (unlike turn-based).
    A vertex can be in both Even's and Odd's PP region.

    Computes game values (Even's winning probability) and thresholds:
    - Even wins PP from v iff value(v) > 0
    - Odd wins PP from v iff value(v) < 1

    Returns: (win_even, win_odd, strategy_even, strategy_odd)
    """
    values = _compute_parity_value(game)

    win_even = {v for v, val in values.items() if val > 1e-8}
    win_odd = {v for v, val in values.items() if val < 1.0 - 1e-8}

    strat_even: Dict[int, Dict[str, float]] = {}
    strat_odd: Dict[int, Dict[str, float]] = {}
    _compute_mixed_strategies(game, win_even, strat_even, maximize=True)
    _compute_mixed_strategies(game, win_odd, strat_odd, maximize=False)

    return win_even, win_odd, strat_even, strat_odd


def _swap_players(game: ConcurrentStochasticGame) -> ConcurrentStochasticGame:
    """Swap Even and Odd actions (for computing Odd's reachability as Even's)."""
    swapped = ConcurrentStochasticGame()
    for v, vertex in game.vertices.items():
        # Swap: Even gets Odd's actions and vice versa
        swapped.add_vertex(v, vertex.priority, vertex.actions_odd, vertex.actions_even)
        for (ae, ao), dist in vertex.transitions.items():
            # In swapped game, new Even action = old Odd, new Odd action = old Even
            swapped.vertices[v].transitions[(ao, ae)] = dict(dist)
    return swapped


def _compute_mixed_strategies(
    game: ConcurrentStochasticGame,
    region: Set[int],
    strategies: Dict[int, Dict[str, float]],
    maximize: bool,
) -> None:
    """Compute mixed strategies for vertices in region that don't have them yet."""
    for v in region:
        if v in strategies:
            continue
        if v not in game.vertices:
            continue
        vertex = game.vertices[v]
        ae_list = vertex.actions_even
        ao_list = vertex.actions_odd
        if not ae_list or not ao_list:
            continue
        # Build value matrix (doesn't matter what values -- just need a valid strategy)
        m, n = len(ae_list), len(ao_list)
        matrix = np.zeros((m, n))
        for i, ae in enumerate(ae_list):
            for j, ao in enumerate(ao_list):
                dist = vertex.transitions.get((ae, ao), {})
                # Use "target in region" as value proxy
                val = sum(p for s, p in dist.items() if s in region)
                matrix[i, j] = val

        if maximize:
            _, even_strat, _ = solve_matrix_game(matrix)
            strategies[v] = {ae_list[i]: float(even_strat[i]) for i in range(m)
                             if even_strat[i] > 1e-10}
        else:
            _, _, odd_strat = solve_matrix_game(matrix)
            strategies[v] = {ao_list[j]: float(odd_strat[j]) for j in range(n)
                             if odd_strat[j] > 1e-10}
        if not strategies[v]:
            # Fallback: uniform
            actions = ae_list if maximize else ao_list
            strategies[v] = {a: 1.0 / len(actions) for a in actions}


# ---------------------------------------------------------------------------
# Main solver API
# ---------------------------------------------------------------------------

def solve_concurrent_stochastic(game: ConcurrentStochasticGame) -> CSGResult:
    """Solve a concurrent stochastic parity game.

    Computes both almost-sure and positive-probability winning regions
    with mixed (randomized) strategies for both players.
    """
    result = CSGResult()

    w0_as, w1_as, s0_as, s1_as = solve_concurrent_parity_almost_sure(game)
    result.win_even_as = w0_as
    result.win_odd_as = w1_as
    result.strategy_even_as = s0_as
    result.strategy_odd_as = s1_as

    w0_pp, w1_pp, s0_pp, s1_pp = solve_concurrent_parity_positive_prob(game)
    result.win_even_pp = w0_pp
    result.win_odd_pp = w1_pp
    result.strategy_even_pp = s0_pp
    result.strategy_odd_pp = s1_pp

    return result


# ---------------------------------------------------------------------------
# Game construction helpers
# ---------------------------------------------------------------------------

def make_concurrent_game(
    vertices: List[Tuple[int, int, List[str], List[str]]],
    transitions: List[Tuple[int, str, str, Dict[int, float]]],
) -> ConcurrentStochasticGame:
    """Build a concurrent game from lists.

    vertices: [(id, priority, even_actions, odd_actions), ...]
    transitions: [(vertex, act_even, act_odd, {succ: prob, ...}), ...]
    """
    game = ConcurrentStochasticGame()
    for v, prio, ae, ao in vertices:
        game.add_vertex(v, prio, ae, ao)
    for v, ae, ao, dist in transitions:
        game.add_transition(v, ae, ao, dist)
    return game


def make_matching_pennies(n_states: int = 2) -> ConcurrentStochasticGame:
    """Classic matching pennies game extended to parity.

    Two states, two actions each. Even wants heads-heads (stay at even-prio),
    Odd wants mismatch (go to odd-prio).
    """
    game = ConcurrentStochasticGame()
    # State 0: priority 0 (even) -- Even's "safe" state
    # State 1: priority 1 (odd) -- Odd's "safe" state
    game.add_vertex(0, 0, ["H", "T"], ["H", "T"])
    game.add_vertex(1, 1, ["H", "T"], ["H", "T"])

    for src in [0, 1]:
        # Match -> go to state 0 (even prio), mismatch -> go to state 1 (odd prio)
        game.add_transition(src, "H", "H", {0: 1.0})  # match
        game.add_transition(src, "T", "T", {0: 1.0})  # match
        game.add_transition(src, "H", "T", {1: 1.0})  # mismatch
        game.add_transition(src, "T", "H", {1: 1.0})  # mismatch

    return game


def make_rock_paper_scissors_game() -> ConcurrentStochasticGame:
    """Rock-Paper-Scissors with parity objectives.

    3 outcome states with different priorities.
    """
    game = ConcurrentStochasticGame()
    actions = ["R", "P", "S"]

    # Arena state where game is played
    game.add_vertex(0, 0, actions, actions)
    # Even wins (prio 2)
    game.add_vertex(1, 2, ["stay"], ["stay"])
    # Odd wins (prio 1)
    game.add_vertex(2, 1, ["stay"], ["stay"])
    # Draw - return to arena (prio 0)
    game.add_vertex(3, 0, ["stay"], ["stay"])

    # Outcome state self-loops
    for state in [1, 2, 3]:
        game.add_transition(state, "stay", "stay", {0: 1.0})  # back to arena

    # RPS outcomes
    wins = {("R", "S"), ("P", "R"), ("S", "P")}
    draws = {("R", "R"), ("P", "P"), ("S", "S")}
    for ae in actions:
        for ao in actions:
            if (ae, ao) in wins:
                game.add_transition(0, ae, ao, {1: 1.0})  # Even wins
            elif (ae, ao) in draws:
                game.add_transition(0, ae, ao, {3: 1.0})  # Draw
            else:
                game.add_transition(0, ae, ao, {2: 1.0})  # Odd wins

    return game


def make_concurrent_reachability(
    n_states: int,
    target: int,
    trap: int,
) -> ConcurrentStochasticGame:
    """Simple concurrent reachability game.

    Chain of states. Even chooses "go" or "stay", Odd chooses "block" or "allow".
    - (go, allow): advance to next state
    - (go, block): random -- 50% advance, 50% stay
    - (stay, *): stay at current state
    Target is absorbing with even prio; trap is absorbing with odd prio.
    """
    game = ConcurrentStochasticGame()

    # First pass: add all vertices
    for i in range(n_states):
        if i == target:
            game.add_vertex(i, 2, ["stay"], ["stay"])
        elif i == trap:
            game.add_vertex(i, 1, ["stay"], ["stay"])
        else:
            game.add_vertex(i, 0, ["go", "stay"], ["allow", "block"])

    # Second pass: add transitions
    for i in range(n_states):
        if i == target or i == trap:
            game.add_transition(i, "stay", "stay", {i: 1.0})
        else:
            nxt = min(i + 1, n_states - 1)
            game.add_transition(i, "go", "allow", {nxt: 1.0})
            game.add_transition(i, "go", "block", {nxt: 0.5, i: 0.5})
            game.add_transition(i, "stay", "allow", {i: 1.0})
            game.add_transition(i, "stay", "block", {i: 1.0})

    return game


def make_concurrent_safety(
    n_states: int,
    safe_region: Set[int],
) -> ConcurrentStochasticGame:
    """Concurrent safety game. Even wants to stay in safe_region forever."""
    game = ConcurrentStochasticGame()
    for i in range(n_states):
        prio = 0 if i in safe_region else 1
        game.add_vertex(i, prio, ["L", "R"], ["L", "R"])

    for i in range(n_states):
        left = (i - 1) % n_states
        right = (i + 1) % n_states
        # Match directions: go that way. Mismatch: random.
        game.add_transition(i, "L", "L", {left: 1.0})
        game.add_transition(i, "R", "R", {right: 1.0})
        game.add_transition(i, "L", "R", {left: 0.5, right: 0.5})
        game.add_transition(i, "R", "L", {left: 0.5, right: 0.5})

    return game


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_play(
    game: ConcurrentStochasticGame,
    start: int,
    strategy_even: Dict[int, Dict[str, float]],
    strategy_odd: Dict[int, Dict[str, float]],
    steps: int = 20,
    seed: Optional[int] = None,
) -> List[Tuple[int, int, str, str]]:
    """Simulate a play under given mixed strategies.

    Returns list of (vertex, priority, even_action, odd_action).
    """
    rng = np.random.default_rng(seed)
    trace = []
    current = start

    for _ in range(steps):
        if current not in game.vertices:
            break
        vertex = game.vertices[current]
        prio = vertex.priority

        # Sample Even's action
        even_dist = strategy_even.get(current, {})
        if not even_dist:
            # Uniform fallback
            even_dist = {a: 1.0 / len(vertex.actions_even) for a in vertex.actions_even}
        ae_actions = list(even_dist.keys())
        ae_probs = np.array([even_dist[a] for a in ae_actions])
        ae_probs /= ae_probs.sum()
        ae = ae_actions[rng.choice(len(ae_actions), p=ae_probs)]

        # Sample Odd's action
        odd_dist = strategy_odd.get(current, {})
        if not odd_dist:
            odd_dist = {a: 1.0 / len(vertex.actions_odd) for a in vertex.actions_odd}
        ao_actions = list(odd_dist.keys())
        ao_probs = np.array([odd_dist[a] for a in ao_actions])
        ao_probs /= ao_probs.sum()
        ao = ao_actions[rng.choice(len(ao_actions), p=ao_probs)]

        trace.append((current, prio, ae, ao))

        # Transition
        dist = vertex.transitions.get((ae, ao), {})
        if not dist:
            break
        succs = list(dist.keys())
        probs = np.array([dist[s] for s in succs])
        probs /= probs.sum()
        current = succs[rng.choice(len(succs), p=probs)]

    return trace


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_strategy(
    game: ConcurrentStochasticGame,
    region: Set[int],
    strategy: Dict[int, Dict[str, float]],
    is_even: bool,
) -> Dict:
    """Verify a mixed strategy is valid (well-formed distributions, actions exist)."""
    errors = []
    checked = 0

    for v in region:
        if v not in game.vertices:
            continue
        if v not in strategy:
            continue
        checked += 1
        vertex = game.vertices[v]
        dist = strategy[v]
        actions = vertex.actions_even if is_even else vertex.actions_odd

        # Check actions are valid
        for a in dist:
            if a not in actions:
                errors.append(f"Invalid action '{a}' at vertex {v}")

        # Check distribution sums to ~1
        total = sum(dist.values())
        if abs(total - 1.0) > 1e-6:
            errors.append(f"Strategy at {v} sums to {total}")

        # Check no negative probabilities
        for a, p in dist.items():
            if p < -1e-10:
                errors.append(f"Negative probability {p} for action '{a}' at {v}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "checked_vertices": checked,
    }


# ---------------------------------------------------------------------------
# Analysis and comparison
# ---------------------------------------------------------------------------

def to_turn_based(game: ConcurrentStochasticGame):
    """Convert concurrent game to turn-based stochastic game (V165 format).

    Creates intermediate vertices for action choices.
    Vertex layout: original vertices are Even-owned, intermediate vertices
    encode Even's choice (Odd-owned), and random vertices resolve.
    """
    # Import V165
    import sys
    sys.path.insert(0, "Z:/AgentZero/A2/work/V165_stochastic_parity_games")
    from stochastic_parity import StochasticParityGame, VertexType

    tb = StochasticParityGame()
    next_id = max(game.vertices.keys()) + 1 if game.vertices else 0

    for v, vertex in game.vertices.items():
        # Original vertex: Even chooses action
        tb.add_vertex(v, VertexType.EVEN, vertex.priority)

        for i, ae in enumerate(vertex.actions_even):
            # Intermediate vertex: Odd chooses action (after Even chose ae)
            ae_id = next_id
            next_id += 1
            tb.add_vertex(ae_id, VertexType.ODD, vertex.priority)
            tb.add_edge(v, ae_id)

            for j, ao in enumerate(vertex.actions_odd):
                dist = vertex.transitions.get((ae, ao), {})
                if not dist:
                    continue

                if len(dist) == 1:
                    # Deterministic: direct edge
                    succ = list(dist.keys())[0]
                    tb.add_edge(ae_id, succ)
                else:
                    # Probabilistic: random vertex
                    rand_id = next_id
                    next_id += 1
                    tb.add_vertex(rand_id, VertexType.RANDOM, vertex.priority)
                    tb.add_edge(ae_id, rand_id)
                    for succ, prob in dist.items():
                        tb.add_edge(rand_id, succ, prob)

    return tb


def compare_with_turn_based(game: ConcurrentStochasticGame) -> Dict:
    """Compare concurrent game solution with turn-based approximation."""
    # Solve concurrent
    result = solve_concurrent_stochastic(game)

    # For comparison, we can note the sizes
    return {
        "concurrent_vertices": len(game.vertices),
        "win_even_as": sorted(result.win_even_as),
        "win_odd_as": sorted(result.win_odd_as),
        "win_even_pp": sorted(result.win_even_pp),
        "win_odd_pp": sorted(result.win_odd_pp),
        "even_strategy_vertices_as": len(result.strategy_even_as),
        "odd_strategy_vertices_as": len(result.strategy_odd_as),
    }


def concurrent_game_statistics(game: ConcurrentStochasticGame) -> Dict:
    """Compute statistics about a concurrent stochastic game."""
    if not game.vertices:
        return {"vertices": 0}

    priorities = [v.priority for v in game.vertices.values()]
    even_actions = [len(v.actions_even) for v in game.vertices.values()]
    odd_actions = [len(v.actions_odd) for v in game.vertices.values()]

    return {
        "vertices": len(game.vertices),
        "priorities": sorted(set(priorities)),
        "max_priority": max(priorities),
        "min_priority": min(priorities),
        "max_even_actions": max(even_actions),
        "max_odd_actions": max(odd_actions),
        "total_action_pairs": sum(
            len(v.actions_even) * len(v.actions_odd)
            for v in game.vertices.values()
        ),
        "total_transitions": sum(
            len(v.transitions) for v in game.vertices.values()
        ),
    }


def batch_solve(
    games: List[Tuple[str, ConcurrentStochasticGame]],
) -> Dict[str, CSGResult]:
    """Solve multiple concurrent games."""
    results = {}
    for name, game in games:
        results[name] = solve_concurrent_stochastic(game)
    return results
