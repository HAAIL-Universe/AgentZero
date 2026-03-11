"""V165: Stochastic Parity Games -- 2.5-player parity games with probabilistic vertices.

Composes V156 (Parity Games) + V164 (Stochastic Energy Games).

A stochastic parity game has three vertex types:
  - EVEN: controlled by player Even (who wants the highest priority visited
    infinitely often to be even)
  - ODD: controlled by player Odd (who wants the highest priority to be odd)
  - RANDOM: nature selects a successor according to a probability distribution

Winning conditions:
  - Almost-sure: Even wins if, under optimal play, the parity condition holds
    with probability 1 (against any Odd strategy).
  - Positive-probability: Even wins if there EXISTS an Even strategy such that
    the parity condition holds with probability > 0 (against any Odd strategy).

Key insight: almost-sure parity in stochastic games requires a qualitative
analysis. Unlike deterministic parity (Zielonka), RANDOM vertices introduce
the need for iterative attractor refinement where:
  - EVEN vertices: need at least one successor in winning region
  - ODD vertices: need all successors in winning region
  - RANDOM vertices (almost-sure): need all positive-prob successors in winning region
  - RANDOM vertices (positive-prob): need at least one positive-prob successor in winning region
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Set, List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
from parity_games import ParityGame, Player, Solution as ParitySolution, zielonka, attractor


# ---- Data Structures ----

class VertexType(Enum):
    EVEN = auto()
    ODD = auto()
    RANDOM = auto()


@dataclass
class StochasticParityGame:
    """A 2.5-player parity game with probabilistic vertices."""
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, Set[int]] = field(default_factory=dict)
    vertex_type: Dict[int, VertexType] = field(default_factory=dict)
    priority: Dict[int, int] = field(default_factory=dict)
    probabilities: Dict[int, Dict[int, float]] = field(default_factory=dict)
    # probabilities: for RANDOM vertices, maps {vertex: {successor: prob}}

    def add_vertex(self, v: int, vtype: VertexType, prio: int):
        self.vertices.add(v)
        self.vertex_type[v] = vtype
        self.priority[v] = prio
        if v not in self.edges:
            self.edges[v] = set()
        if vtype == VertexType.RANDOM:
            if v not in self.probabilities:
                self.probabilities[v] = {}

    def add_edge(self, u: int, v: int, prob: float = 1.0):
        if u not in self.edges:
            self.edges[u] = set()
        self.edges[u].add(v)
        if self.vertex_type.get(u) == VertexType.RANDOM:
            if u not in self.probabilities:
                self.probabilities[u] = {}
            self.probabilities[u][v] = prob

    def successors(self, v: int) -> Set[int]:
        return self.edges.get(v, set())

    def predecessors(self, v: int) -> Set[int]:
        preds = set()
        for u in self.vertices:
            if v in self.edges.get(u, set()):
                preds.add(u)
        return preds

    def max_priority(self) -> int:
        if not self.priority:
            return 0
        return max(self.priority.values())

    def vertices_with_priority(self, p: int) -> Set[int]:
        return {v for v in self.vertices if self.priority.get(v) == p}

    def get_prob(self, v: int, succ: int) -> float:
        if self.vertex_type.get(v) != VertexType.RANDOM:
            return 1.0
        return self.probabilities.get(v, {}).get(succ, 0.0)

    def validate(self) -> List[str]:
        errors = []
        for v in self.vertices:
            if self.vertex_type.get(v) == VertexType.RANDOM:
                probs = self.probabilities.get(v, {})
                total = sum(probs.values())
                if abs(total - 1.0) > 1e-9 and probs:
                    errors.append(f"RANDOM vertex {v}: probs sum to {total}")
                succs = self.edges.get(v, set())
                for s in succs:
                    if s not in probs:
                        errors.append(f"RANDOM vertex {v}: successor {s} has no probability")
        return errors

    def to_parity_game(self) -> ParityGame:
        """Convert to deterministic parity game (RANDOM -> EVEN)."""
        pg = ParityGame(
            vertices=set(self.vertices),
            edges={v: set(s) for v, s in self.edges.items()},
            owner={v: Player.EVEN if self.vertex_type[v] != VertexType.ODD
                   else Player.ODD for v in self.vertices},
            priority=dict(self.priority)
        )
        return pg

    def subgame(self, verts: Set[int]) -> 'StochasticParityGame':
        """Restrict game to a subset of vertices."""
        g = StochasticParityGame()
        for v in verts:
            g.add_vertex(v, self.vertex_type[v], self.priority[v])
        for v in verts:
            for s in self.edges.get(v, set()):
                if s in verts:
                    prob = self.get_prob(v, s) if self.vertex_type[v] == VertexType.RANDOM else 1.0
                    g.add_edge(v, s, prob)
        return g


@dataclass
class StochasticParityResult:
    """Result of solving a stochastic parity game."""
    win_even_as: Set[int] = field(default_factory=set)    # Almost-sure winning for Even
    win_odd_as: Set[int] = field(default_factory=set)     # Almost-sure winning for Odd
    win_even_pp: Set[int] = field(default_factory=set)    # Positive-prob winning for Even
    win_odd_pp: Set[int] = field(default_factory=set)     # Positive-prob winning for Odd
    strategy_even_as: Dict[int, int] = field(default_factory=dict)
    strategy_odd_as: Dict[int, int] = field(default_factory=dict)
    strategy_even_pp: Dict[int, int] = field(default_factory=dict)
    strategy_odd_pp: Dict[int, int] = field(default_factory=dict)


# ---- Attractor Computation for Stochastic Games ----

def stochastic_attractor(game: StochasticParityGame, target: Set[int],
                         player: VertexType, subgame: Set[int],
                         mode: str = 'almost_sure') -> Set[int]:
    """Compute attractor for a player in a stochastic game.

    Args:
        game: The stochastic parity game
        target: Target vertex set
        player: Which player's attractor (EVEN or ODD)
        subgame: Restrict to these vertices
        mode: 'almost_sure' or 'positive_prob'
            - almost_sure: RANDOM vertices attracted if ALL positive-prob succs in attractor
            - positive_prob: RANDOM vertices attracted if ANY positive-prob succ in attractor

    Returns:
        The attractor set (includes target)
    """
    opponent = VertexType.ODD if player == VertexType.EVEN else VertexType.EVEN
    attr = target & subgame
    frontier = set(attr)

    while frontier:
        new_frontier = set()
        for v in subgame - attr:
            succs_in_sub = game.successors(v) & subgame
            if not succs_in_sub:
                continue

            vtype = game.vertex_type[v]

            if vtype == player:
                # Player's vertex: attracted if ANY successor in attractor
                if succs_in_sub & attr:
                    new_frontier.add(v)
            elif vtype == opponent:
                # Opponent's vertex: attracted if ALL successors in attractor
                if succs_in_sub <= attr:
                    new_frontier.add(v)
            elif vtype == VertexType.RANDOM:
                if mode == 'almost_sure':
                    # RANDOM attracted if ALL positive-prob successors in attractor
                    all_in = True
                    for s in succs_in_sub:
                        p = game.get_prob(v, s)
                        if p > 0 and s not in attr:
                            all_in = False
                            break
                    if all_in:
                        new_frontier.add(v)
                else:
                    # positive_prob: RANDOM attracted if ANY positive-prob succ in attractor
                    for s in succs_in_sub:
                        p = game.get_prob(v, s)
                        if p > 0 and s in attr:
                            new_frontier.add(v)
                            break

        if not new_frontier:
            break
        attr |= new_frontier
        frontier = new_frontier

    return attr


# ---- Core Solver: Qualitative Stochastic Parity ----

def _solve_almost_sure(game: StochasticParityGame) -> Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    """Solve for almost-sure winning regions via iterative refinement.

    Returns (win_even, win_odd, strategy_even, strategy_odd).

    Algorithm:
    1. Solve deterministic parity on current subgame (RANDOM treated as EVEN)
    2. Check if Even's winning region is "closed" for RANDOM:
       all positive-prob successors (in ORIGINAL game) must be in win_even
    3. If not closed: remove bad RANDOM vertices, compute Odd attractor, shrink game
    4. Repeat until stable

    Key insight: Zielonka's subgame restriction removes edges leaving the subgame,
    which is invalid for RANDOM vertices -- a RANDOM vertex in a cycle will
    almost-surely follow every positive-prob edge infinitely often (Borel-Cantelli).
    """
    current = set(game.vertices)
    win_even = set()
    win_odd = set(game.vertices)

    for _iteration in range(len(game.vertices) + 1):
        if not current:
            break

        # Solve deterministic parity on current subgame
        sub = game.subgame(current)
        pg = sub.to_parity_game()
        sol = zielonka(pg)

        # Check RANDOM closure: for each RANDOM v in win_even,
        # ALL positive-prob successors (in ORIGINAL game) must also be in win_even
        bad_random = set()
        for v in sol.win_even:
            if game.vertex_type[v] == VertexType.RANDOM:
                for s in game.successors(v):  # ORIGINAL game edges
                    p = game.get_prob(v, s)
                    if p > 0 and s not in sol.win_even:
                        bad_random.add(v)
                        break

        if not bad_random:
            # Stable: Even wins sol.win_even, Odd wins the rest
            win_even = sol.win_even
            win_odd = game.vertices - win_even
            break
        else:
            # Remove bad RANDOM vertices + Odd attractor
            odd_attr = _compute_odd_attractor_as(game, current, bad_random)
            new_current = current - odd_attr
            if new_current == current:
                # No progress -- shouldn't happen, but safety valve
                win_even = sol.win_even - bad_random
                win_odd = game.vertices - win_even
                break
            current = new_current
    else:
        # Exhausted iterations
        win_even = set()
        win_odd = set(game.vertices)

    # Extract strategies
    strategy_even = {}
    strategy_odd = {}
    for v in win_even:
        if game.vertex_type[v] == VertexType.EVEN:
            for s in game.successors(v):
                if s in win_even:
                    strategy_even[v] = s
                    break
    for v in win_odd:
        if game.vertex_type[v] == VertexType.ODD:
            for s in game.successors(v):
                if s in win_odd:
                    strategy_odd[v] = s
                    break

    return win_even, win_odd, strategy_even, strategy_odd


def _compute_odd_attractor_as(game: StochasticParityGame, subgame: Set[int],
                              target: Set[int]) -> Set[int]:
    """Compute Odd's attractor in the almost-sure sense.

    Odd attracts if:
    - ODD vertex: ANY successor in attractor
    - EVEN vertex: ALL successors (in subgame) in attractor
    - RANDOM vertex: ANY positive-prob successor in attractor
      (for Odd, having ONE bad outcome is enough to attract almost-surely)
    """
    attr = target & subgame
    changed = True
    while changed:
        changed = False
        for v in subgame - attr:
            succs = game.successors(v) & subgame
            if not succs:
                continue

            vtype = game.vertex_type[v]
            attracted = False

            if vtype == VertexType.ODD:
                # Odd chooses: ANY successor in attractor
                if succs & attr:
                    attracted = True
            elif vtype == VertexType.EVEN:
                # Even is forced: ALL successors in attractor
                if succs <= attr:
                    attracted = True
            elif vtype == VertexType.RANDOM:
                # Almost-sure: if ANY positive-prob successor is in attractor,
                # Odd benefits (play will reach attractor a.s.)
                for s in succs:
                    p = game.get_prob(v, s)
                    if p > 0 and s in attr:
                        attracted = True
                        break

            if attracted:
                attr.add(v)
                changed = True

    return attr


def _solve_positive_prob(game: StochasticParityGame) -> Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    """Solve for positive-probability winning regions.

    For positive-prob, RANDOM vertices behave exactly like EVEN vertices
    (Even only needs ONE random outcome with positive probability to go well).
    So we convert RANDOM -> EVEN and use deterministic Zielonka.
    """
    # Convert to deterministic parity game (RANDOM -> EVEN)
    pg = game.to_parity_game()
    det_sol = zielonka(pg)

    # Extract strategies for Even-owned and RANDOM vertices
    strategy_even = {}
    for v in det_sol.win_even:
        vtype = game.vertex_type[v]
        if vtype == VertexType.EVEN or vtype == VertexType.RANDOM:
            if v in det_sol.strategy_even:
                strategy_even[v] = det_sol.strategy_even[v]

    strategy_odd = {}
    for v in det_sol.win_odd:
        if game.vertex_type[v] == VertexType.ODD:
            if v in det_sol.strategy_odd:
                strategy_odd[v] = det_sol.strategy_odd[v]

    return det_sol.win_even, det_sol.win_odd, strategy_even, strategy_odd


# ---- Main API ----

def solve_stochastic_parity(game: StochasticParityGame) -> StochasticParityResult:
    """Solve a stochastic parity game for both almost-sure and positive-probability winning.

    Returns a StochasticParityResult with winning regions and strategies for both modes.
    """
    we_as, wo_as, se_as, so_as = _solve_almost_sure(game)
    we_pp, wo_pp, se_pp, so_pp = _solve_positive_prob(game)

    return StochasticParityResult(
        win_even_as=we_as,
        win_odd_as=wo_as,
        win_even_pp=we_pp,
        win_odd_pp=wo_pp,
        strategy_even_as=se_as,
        strategy_odd_as=so_as,
        strategy_even_pp=se_pp,
        strategy_odd_pp=so_pp,
    )


def solve_almost_sure(game: StochasticParityGame) -> StochasticParityResult:
    """Solve only for almost-sure winning regions."""
    we, wo, se, so = _solve_almost_sure(game)
    return StochasticParityResult(
        win_even_as=we, win_odd_as=wo,
        strategy_even_as=se, strategy_odd_as=so,
    )


def solve_positive_prob(game: StochasticParityGame) -> StochasticParityResult:
    """Solve only for positive-probability winning regions."""
    we, wo, se, so = _solve_positive_prob(game)
    return StochasticParityResult(
        win_even_pp=we, win_odd_pp=wo,
        strategy_even_pp=se, strategy_odd_pp=so,
    )


# ---- Simulation ----

def simulate_play(game: StochasticParityGame, start: int,
                  strategy_even: Dict[int, int], strategy_odd: Dict[int, int],
                  steps: int = 50, seed: int = 42) -> List[Tuple[int, int]]:
    """Simulate a play through the game.

    Returns list of (vertex, priority) pairs representing the play.
    """
    import random
    rng = random.Random(seed)

    trace = []
    v = start
    for _ in range(steps):
        if v not in game.vertices:
            break
        trace.append((v, game.priority[v]))

        succs = game.successors(v)
        if not succs:
            break

        vtype = game.vertex_type[v]
        if vtype == VertexType.EVEN:
            if v in strategy_even:
                v = strategy_even[v]
            else:
                v = min(succs)  # default: pick smallest
        elif vtype == VertexType.ODD:
            if v in strategy_odd:
                v = strategy_odd[v]
            else:
                v = min(succs)
        else:
            # RANDOM: choose according to probabilities
            succs_list = sorted(succs)
            probs = [game.get_prob(v, s) for s in succs_list]
            total = sum(probs)
            if total <= 0:
                v = succs_list[0]
            else:
                r = rng.random() * total
                cumul = 0.0
                chosen = succs_list[-1]
                for s, p in zip(succs_list, probs):
                    cumul += p
                    if r <= cumul:
                        chosen = s
                        break
                v = chosen

    return trace


def verify_strategy(game: StochasticParityGame,
                    strategy_even: Dict[int, int],
                    win_region: Set[int],
                    mode: str = 'almost_sure') -> Dict:
    """Verify that a strategy is valid and closed within the winning region.

    Checks:
    - Strategy assigns a valid successor for each Even vertex in win_region
    - All successors (including random) stay within win_region (for almost-sure)
    - At least one random successor in win_region (for positive-prob)
    """
    errors = []
    for v in win_region:
        vtype = game.vertex_type[v]
        succs = game.successors(v) & win_region

        if vtype == VertexType.EVEN:
            if v not in strategy_even:
                # Check if there's any successor in win region
                if succs:
                    errors.append(f"Even vertex {v}: no strategy but has successors in win region")
            else:
                target = strategy_even[v]
                if target not in win_region:
                    errors.append(f"Even vertex {v}: strategy goes to {target} outside win region")

        elif vtype == VertexType.ODD:
            # All successors should be in win region (opponent is forced to stay)
            all_succs = game.successors(v)
            for s in all_succs:
                if s not in win_region:
                    if mode == 'almost_sure':
                        errors.append(f"Odd vertex {v}: successor {s} outside win region")

        elif vtype == VertexType.RANDOM:
            if mode == 'almost_sure':
                # All positive-prob successors must be in win region
                for s in game.successors(v):
                    p = game.get_prob(v, s)
                    if p > 0 and s not in win_region:
                        errors.append(f"Random vertex {v}: positive-prob successor {s} outside win region")
            else:
                # At least one positive-prob successor in win region
                has_good = False
                for s in game.successors(v):
                    p = game.get_prob(v, s)
                    if p > 0 and s in win_region:
                        has_good = True
                        break
                if not has_good and game.successors(v):
                    errors.append(f"Random vertex {v}: no positive-prob successor in win region")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'checked_vertices': len(win_region),
    }


# ---- Game Construction Helpers ----

def make_game(vertices: List[Tuple[int, str, int]],
              edges: List[Tuple[int, int]],
              probs: Optional[Dict[int, Dict[int, float]]] = None) -> StochasticParityGame:
    """Construct a stochastic parity game from lists.

    Args:
        vertices: List of (id, type_str, priority) where type_str is 'even'/'odd'/'random'
        edges: List of (from, to)
        probs: Optional {random_vertex: {successor: probability}}
    """
    type_map = {'even': VertexType.EVEN, 'odd': VertexType.ODD, 'random': VertexType.RANDOM}
    g = StochasticParityGame()
    for vid, vtype_str, prio in vertices:
        g.add_vertex(vid, type_map[vtype_str], prio)
    for u, v in edges:
        prob = 1.0
        if probs and u in probs and v in probs[u]:
            prob = probs[u][v]
        g.add_edge(u, v, prob)
    return g


def make_simple_stochastic(n: int, random_vertex: int,
                           prob_even: float = 0.5) -> StochasticParityGame:
    """Create a simple stochastic parity game for testing.

    Linear chain: 0 -> 1 -> ... -> n-1 -> 0
    Vertex `random_vertex` is RANDOM with prob_even to go forward, (1-prob_even) to loop.
    Even-numbered vertices owned by Even, odd by Odd.
    Priority = vertex id mod 3.
    """
    g = StochasticParityGame()
    for i in range(n):
        if i == random_vertex:
            vtype = VertexType.RANDOM
        elif i % 2 == 0:
            vtype = VertexType.EVEN
        else:
            vtype = VertexType.ODD
        g.add_vertex(i, vtype, i % 3)

    for i in range(n):
        nxt = (i + 1) % n
        if i == random_vertex:
            g.add_edge(i, nxt, prob_even)
            g.add_edge(i, i, 1.0 - prob_even)
        else:
            g.add_edge(i, nxt)

    return g


def make_buchi_stochastic(states: int, accepting: Set[int],
                          even_states: Set[int], random_states: Set[int],
                          transitions: List[Tuple[int, int]],
                          probs: Optional[Dict[int, Dict[int, float]]] = None) -> StochasticParityGame:
    """Create a stochastic Buchi game (Even wins iff accepting states visited infinitely often).

    Priority: 2 for accepting (even -- good for Even), 1 for non-accepting.
    """
    g = StochasticParityGame()
    for s in range(states):
        if s in random_states:
            vtype = VertexType.RANDOM
        elif s in even_states:
            vtype = VertexType.EVEN
        else:
            vtype = VertexType.ODD
        prio = 2 if s in accepting else 1
        g.add_vertex(s, vtype, prio)

    for u, v in transitions:
        prob = 1.0
        if probs and u in probs and v in probs[u]:
            prob = probs[u][v]
        g.add_edge(u, v, prob)

    return g


def make_reachability_stochastic(states: int, target: Set[int],
                                 even_states: Set[int], random_states: Set[int],
                                 transitions: List[Tuple[int, int]],
                                 probs: Optional[Dict[int, Dict[int, float]]] = None) -> StochasticParityGame:
    """Create a stochastic reachability game (Even wins iff target is reached).

    Priority: 2 for target (add self-loops), 1 for non-target.
    """
    g = StochasticParityGame()
    for s in range(states):
        if s in random_states:
            vtype = VertexType.RANDOM
        elif s in even_states:
            vtype = VertexType.EVEN
        else:
            vtype = VertexType.ODD
        prio = 2 if s in target else 1
        g.add_vertex(s, vtype, prio)

    for u, v in transitions:
        prob = 1.0
        if probs and u in probs and v in probs[u]:
            prob = probs[u][v]
        g.add_edge(u, v, prob)

    # Self-loops on target
    for t in target:
        if t not in g.edges or t not in g.edges[t]:
            g.add_edge(t, t)

    return g


def make_safety_stochastic(states: int, bad: Set[int],
                           even_states: Set[int], random_states: Set[int],
                           transitions: List[Tuple[int, int]],
                           probs: Optional[Dict[int, Dict[int, float]]] = None) -> StochasticParityGame:
    """Create a stochastic safety game (Even wins iff bad states are avoided forever).

    Priority: 0 for safe states (even -- good for Even), 1 for bad states.
    """
    g = StochasticParityGame()
    for s in range(states):
        if s in random_states:
            vtype = VertexType.RANDOM
        elif s in even_states:
            vtype = VertexType.EVEN
        else:
            vtype = VertexType.ODD
        prio = 1 if s in bad else 0
        g.add_vertex(s, vtype, prio)

    for u, v in transitions:
        prob = 1.0
        if probs and u in probs and v in probs[u]:
            prob = probs[u][v]
        g.add_edge(u, v, prob)

    return g


# ---- Analysis & Comparison ----

def compare_with_deterministic(game: StochasticParityGame) -> Dict:
    """Compare stochastic results with deterministic Zielonka on same structure.

    The deterministic version treats RANDOM as EVEN.
    """
    # Stochastic solve
    result = solve_stochastic_parity(game)

    # Deterministic solve (RANDOM -> EVEN)
    pg = game.to_parity_game()
    det_sol = zielonka(pg)

    return {
        'deterministic': {
            'win_even': det_sol.win_even,
            'win_odd': det_sol.win_odd,
        },
        'stochastic_as': {
            'win_even': result.win_even_as,
            'win_odd': result.win_odd_as,
        },
        'stochastic_pp': {
            'win_even': result.win_even_pp,
            'win_odd': result.win_odd_pp,
        },
        # Ordering: as_even <= pp_even (almost-sure is harder to win)
        'as_subset_pp': result.win_even_as <= result.win_even_pp,
        # Deterministic should equal positive-prob (RANDOM=EVEN is same)
        'pp_matches_det': result.win_even_pp == det_sol.win_even,
    }


def stochastic_parity_statistics(game: StochasticParityGame) -> Dict:
    """Compute statistics about a stochastic parity game."""
    n_even = sum(1 for v in game.vertices if game.vertex_type[v] == VertexType.EVEN)
    n_odd = sum(1 for v in game.vertices if game.vertex_type[v] == VertexType.ODD)
    n_random = sum(1 for v in game.vertices if game.vertex_type[v] == VertexType.RANDOM)
    n_edges = sum(len(s) for s in game.edges.values())
    priorities = set(game.priority.values()) if game.priority else set()

    return {
        'vertices': len(game.vertices),
        'edges': n_edges,
        'even_vertices': n_even,
        'odd_vertices': n_odd,
        'random_vertices': n_random,
        'max_priority': game.max_priority(),
        'distinct_priorities': len(priorities),
        'priorities': sorted(priorities),
    }


def batch_solve(games: List[Tuple[str, StochasticParityGame]]) -> Dict[str, StochasticParityResult]:
    """Solve multiple games and return results keyed by name."""
    results = {}
    for name, game in games:
        results[name] = solve_stochastic_parity(game)
    return results
