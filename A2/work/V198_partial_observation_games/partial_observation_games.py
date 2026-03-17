"""V198: Partial Observation Games -- games with imperfect information.

Composes V156 (parity games) + V159 (symbolic parity) + V021 (BDD).

In partial observation games, one or both players cannot see the full game
state. Instead, each player observes an equivalence class (observation) and
must play a uniform strategy: the same move from all states sharing an
observation.

Key concepts:
- Observation function: maps states to observations (equivalence classes)
- Observation-based (uniform) strategy: same move for all states in same obs class
- Knowledge set: player's belief about current state given observation history
- Belief-based solving: subset construction over observations
- Antichains: efficient representation of downward-closed belief sets

Algorithms:
1. Subset construction: build knowledge game from partial observation game
2. Safety solving under partial observation (exponential in general)
3. Reachability under partial observation
4. Buchi under partial observation
5. Antichain optimization for safety/reachability
"""

import sys, os, math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V159_symbolic_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

from parity_games import ParityGame, Player, Solution, verify_solution, zielonka, make_game
from symbolic_parity_games import (
    SymbolicParityGame, SymbolicSolution, explicit_to_symbolic,
    solve_symbolic, extract_winning_sets, verify_symbolic_solution,
    symbolic_game_stats
)
from bdd_model_checker import BDD, BDDNode


class Objective(Enum):
    """Game objective types."""
    SAFETY = "safety"           # avoid bad states forever
    REACHABILITY = "reach"      # reach target states
    BUCHI = "buchi"             # visit accepting states infinitely often
    CO_BUCHI = "co_buchi"       # visit rejecting states finitely often
    PARITY = "parity"           # general parity condition


@dataclass
class PartialObsGame:
    """A two-player game with partial observation.

    Player 1 (the protagonist, Even) has partial observation.
    Player 2 (the antagonist, Odd) has full observation.

    Attributes:
        vertices: set of vertex IDs
        edges: adjacency list (vertex -> set of successors)
        owner: vertex -> Player (who controls this vertex)
        obs: vertex -> observation ID (Player 1's observation function)
        initial: set of initial vertices
        objective: game objective type
        target: set of target/accepting/bad vertices (meaning depends on objective)
        priority: vertex -> priority (for parity objectives)
    """
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, Set[int]] = field(default_factory=dict)
    owner: Dict[int, Player] = field(default_factory=dict)
    obs: Dict[int, int] = field(default_factory=dict)
    initial: Set[int] = field(default_factory=set)
    objective: Objective = Objective.SAFETY
    target: Set[int] = field(default_factory=set)
    priority: Dict[int, int] = field(default_factory=dict)

    def add_vertex(self, v: int, player: Player, observation: int,
                   prio: int = 0):
        """Add a vertex with owner, observation, and priority."""
        self.vertices.add(v)
        self.owner[v] = player
        self.obs[v] = observation
        self.priority[v] = prio

    def add_edge(self, u: int, v: int):
        """Add a directed edge."""
        if u not in self.edges:
            self.edges[u] = set()
        self.edges[u].add(v)

    def successors(self, v: int) -> Set[int]:
        """Get successors of a vertex."""
        return self.edges.get(v, set())

    def predecessors(self, v: int) -> Set[int]:
        """Get predecessors of a vertex."""
        return {u for u in self.vertices if v in self.edges.get(u, set())}

    def obs_class(self, o: int) -> Set[int]:
        """Get all vertices with observation o."""
        return {v for v in self.vertices if self.obs.get(v) == o}

    def all_observations(self) -> Set[int]:
        """Get set of all observation IDs."""
        return set(self.obs.values())

    def is_observation_consistent(self) -> bool:
        """Check if observation respects ownership (same obs -> same owner)."""
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

    def is_observation_action_consistent(self) -> bool:
        """Check if vertices with same observation have same action set.

        This is required for well-defined partial observation games:
        states in the same observation class must have the same
        available actions (successor observations).
        """
        obs_actions = {}
        for v in self.vertices:
            o = self.obs[v]
            action_obs = frozenset(self.obs.get(w) for w in self.successors(v))
            if o in obs_actions:
                if obs_actions[o] != action_obs:
                    return False
            else:
                obs_actions[o] = action_obs
        return True


@dataclass
class KnowledgeState:
    """A state in the knowledge game (belief set).

    Represents Player 1's belief about the current game state:
    the set of states that are consistent with the observation history.
    """
    belief: FrozenSet[int]     # set of possible current states
    observation: int            # current observation

    def __hash__(self):
        return hash((self.belief, self.observation))

    def __eq__(self, other):
        return (isinstance(other, KnowledgeState) and
                self.belief == other.belief and
                self.observation == other.observation)


@dataclass
class KnowledgeGame:
    """The knowledge game (subset construction).

    Each vertex is a KnowledgeState (belief set + observation).
    This converts a partial observation game into a perfect information game.
    """
    states: Dict[KnowledgeState, int]      # KnowledgeState -> numeric ID
    id_to_state: Dict[int, KnowledgeState]  # numeric ID -> KnowledgeState
    game: ParityGame                         # V156 parity game on knowledge states
    initial_id: int                          # initial state ID


@dataclass
class ObsStrategy:
    """An observation-based (uniform) strategy.

    Maps observations to actions (successor observations).
    """
    moves: Dict[int, int]  # observation -> target observation

    def action(self, obs: int) -> Optional[int]:
        """Get the action for a given observation."""
        return self.moves.get(obs)


@dataclass
class POGameResult:
    """Result of solving a partial observation game."""
    winning: bool                               # True if Player 1 wins
    winning_beliefs: Set[FrozenSet[int]]        # winning belief sets
    strategy: Optional[ObsStrategy] = None      # winning strategy
    knowledge_game_size: int = 0                # size of knowledge game
    explored_beliefs: int = 0                    # beliefs explored during solving


# --- Core: Knowledge Game Construction ---

def _initial_belief(game: PartialObsGame) -> FrozenSet[int]:
    """Compute initial belief set from initial vertices."""
    if game.initial:
        return frozenset(game.initial)
    return frozenset(game.vertices)


def _observation_split(game: PartialObsGame,
                       belief: FrozenSet[int]) -> Dict[int, FrozenSet[int]]:
    """Split a belief set by observations."""
    obs_groups = {}
    for v in belief:
        o = game.obs[v]
        if o not in obs_groups:
            obs_groups[o] = set()
        obs_groups[o].add(v)
    return {o: frozenset(s) for o, s in obs_groups.items()}


def _post(game: PartialObsGame, belief: FrozenSet[int],
          target_obs: int) -> FrozenSet[int]:
    """Compute successors of belief set filtered to target observation.

    For Player 1 vertices: move to successors with target_obs.
    For Player 2 vertices: all successors (opponent resolves nondeterminism).
    """
    result = set()
    for v in belief:
        for w in game.successors(v):
            if game.obs[w] == target_obs:
                result.add(w)
    return frozenset(result)


def _post_all(game: PartialObsGame,
              belief: FrozenSet[int]) -> Dict[int, FrozenSet[int]]:
    """Compute all successor beliefs grouped by observation."""
    all_succs = set()
    for v in belief:
        all_succs.update(game.successors(v))

    obs_groups = {}
    for w in all_succs:
        o = game.obs[w]
        if o not in obs_groups:
            obs_groups[o] = set()
        obs_groups[o].add(w)

    return {o: frozenset(s) for o, s in obs_groups.items()}


def _uncontrolled_post(game: PartialObsGame,
                       belief: FrozenSet[int]) -> Set[FrozenSet[int]]:
    """Compute successor beliefs for Player 2 (uncontrolled) moves.

    From a belief where Player 2 owns vertices, all possible successor
    beliefs arise from Player 2's choices.
    """
    # Group by Player 2's possible moves
    # For each P2 vertex, successors are P2's choice
    # Result is the set of possible successor belief sets
    all_succs = set()
    for v in belief:
        all_succs.update(game.successors(v))

    if not all_succs:
        return set()

    # Group successors by observation
    obs_groups = {}
    for w in all_succs:
        o = game.obs[w]
        if o not in obs_groups:
            obs_groups[o] = set()
        obs_groups[o].add(w)

    return {frozenset(s) for s in obs_groups.values()}


def build_knowledge_game(game: PartialObsGame,
                         max_beliefs: int = 10000) -> KnowledgeGame:
    """Build the knowledge game via subset construction.

    Each state in the knowledge game is a belief set (set of possible
    game states). The knowledge game has perfect information.

    Args:
        game: partial observation game
        max_beliefs: maximum number of belief states to explore

    Returns:
        KnowledgeGame with the expanded game
    """
    states = {}
    id_to_state = {}
    next_id = 0

    init_belief = _initial_belief(game)
    init_obs_groups = _observation_split(game, init_belief)

    # Queue for BFS
    queue = deque()

    # Create initial knowledge states (one per observation in initial belief)
    kg = ParityGame()
    initial_id = -1

    for obs_val, belief_set in init_obs_groups.items():
        ks = KnowledgeState(belief=belief_set, observation=obs_val)
        if ks not in states:
            states[ks] = next_id
            id_to_state[next_id] = ks
            if initial_id < 0:
                initial_id = next_id

            # Determine owner from observation
            sample_v = next(iter(belief_set))
            p = game.owner[sample_v]
            # Priority: for safety, 0 if all safe, 1 if any bad
            prio = _knowledge_state_priority(game, belief_set)
            kg.add_vertex(next_id, p, prio)
            queue.append(next_id)
            next_id += 1

    # If multiple initial obs groups, create a virtual initial vertex
    if len(init_obs_groups) > 1:
        virtual_init = next_id
        ks_init = KnowledgeState(belief=init_belief, observation=-1)
        states[ks_init] = virtual_init
        id_to_state[virtual_init] = ks_init
        prio = _knowledge_state_priority(game, init_belief)
        kg.add_vertex(virtual_init, Player.EVEN, prio)
        for obs_val, belief_set in init_obs_groups.items():
            ks = KnowledgeState(belief=belief_set, observation=obs_val)
            kg.add_edge(virtual_init, states[ks])
        initial_id = virtual_init
        next_id += 1

    # BFS exploration
    while queue and next_id < max_beliefs:
        sid = queue.popleft()
        ks = id_to_state[sid]
        belief = ks.belief

        if not belief:
            continue

        # Compute successor beliefs
        succ_beliefs = _post_all(game, belief)

        for obs_val, succ_belief in succ_beliefs.items():
            if not succ_belief:
                continue

            succ_ks = KnowledgeState(belief=succ_belief, observation=obs_val)
            if succ_ks not in states:
                states[succ_ks] = next_id
                id_to_state[next_id] = succ_ks

                sample_v = next(iter(succ_belief))
                p = game.owner[sample_v]
                prio = _knowledge_state_priority(game, succ_belief)
                kg.add_vertex(next_id, p, prio)
                queue.append(next_id)
                next_id += 1

            kg.add_edge(sid, states[succ_ks])

    # Add self-loops to dead-end knowledge states for safety/co-Buchi
    # (dead ends are safe for these objectives -- play just stops)
    if game.objective in (Objective.SAFETY, Objective.CO_BUCHI):
        for sid in list(kg.vertices):
            if not kg.edges.get(sid):
                kg.add_edge(sid, sid)

    return KnowledgeGame(
        states=states,
        id_to_state=id_to_state,
        game=kg,
        initial_id=initial_id
    )


def _knowledge_state_priority(game: PartialObsGame,
                               belief: FrozenSet[int]) -> int:
    """Compute priority for a knowledge state based on game objective.

    For safety: 0 (safe) if no state in belief is in target (bad), else 1
    For reachability: 2 (accepting) if any state in belief is in target, else 1
    For Buchi: 2 (accepting) if ALL states in belief are accepting, else 1
    For co-Buchi: 0 (safe) if no rejecting state, else 1
    For parity: max priority in belief set
    """
    if game.objective == Objective.SAFETY:
        if belief & game.target:
            return 1  # bad -- Odd likes
        return 0  # safe -- Even likes

    elif game.objective == Objective.REACHABILITY:
        if belief & game.target:
            return 2  # target reached -- Even likes
        return 1  # not yet -- Odd likes

    elif game.objective == Objective.BUCHI:
        # For Buchi under PO: a knowledge state is accepting only if
        # ALL states in the belief are accepting (conservative)
        if belief and belief <= game.target:
            return 2  # all accepting
        return 1  # some non-accepting

    elif game.objective == Objective.CO_BUCHI:
        if belief & game.target:
            return 1  # some rejecting
        return 0  # all safe

    elif game.objective == Objective.PARITY:
        if not belief:
            return 0
        return max(game.priority.get(v, 0) for v in belief)

    return 0


# --- Solving: Direct Belief-Based Algorithms ---

def solve_safety(game: PartialObsGame,
                 max_beliefs: int = 10000) -> POGameResult:
    """Solve a partial observation safety game.

    Player 1 wins if the play never visits a state in game.target (bad).
    Uses backward fixed-point on belief sets.

    Antichain optimization: only track maximal safe belief sets.
    """
    assert game.objective == Objective.SAFETY

    bad = game.target
    all_obs = game.all_observations()

    # Compute the losing beliefs for Player 1 via backward propagation
    # A belief B is losing if:
    #   - B intersects bad (immediate loss), OR
    #   - For P1 vertices in B: ALL actions lead to a losing belief
    #   - For P2 vertices in B: SOME action leads to a losing belief

    # Forward approach: compute the knowledge game and solve as parity
    kg = build_knowledge_game(game, max_beliefs)
    if not kg.game.vertices:
        return POGameResult(
            winning=True, winning_beliefs=set(),
            knowledge_game_size=0, explored_beliefs=0
        )

    # Solve the knowledge game
    sol = zielonka(kg.game)

    # Check if initial state is winning for Player 1 (Even)
    winning = kg.initial_id in sol.win_even

    # Extract winning beliefs
    winning_beliefs = set()
    for ks, sid in kg.states.items():
        if sid in sol.win_even and ks.observation >= 0:
            winning_beliefs.add(ks.belief)

    # Extract strategy
    strategy = _extract_obs_strategy(game, kg, sol)

    return POGameResult(
        winning=winning,
        winning_beliefs=winning_beliefs,
        strategy=strategy,
        knowledge_game_size=len(kg.game.vertices),
        explored_beliefs=len(kg.states)
    )


def solve_reachability(game: PartialObsGame,
                       max_beliefs: int = 10000) -> POGameResult:
    """Solve a partial observation reachability game.

    Player 1 wins if the play reaches a state in game.target.
    """
    assert game.objective == Objective.REACHABILITY

    kg = build_knowledge_game(game, max_beliefs)
    if not kg.game.vertices:
        return POGameResult(
            winning=False, winning_beliefs=set(),
            knowledge_game_size=0, explored_beliefs=0
        )

    # Add self-loops to target knowledge states (for parity winning condition)
    for ks, sid in kg.states.items():
        if ks.belief and ks.belief & game.target:
            kg.game.add_edge(sid, sid)

    sol = zielonka(kg.game)
    winning = kg.initial_id in sol.win_even

    winning_beliefs = set()
    for ks, sid in kg.states.items():
        if sid in sol.win_even and ks.observation >= 0:
            winning_beliefs.add(ks.belief)

    strategy = _extract_obs_strategy(game, kg, sol)

    return POGameResult(
        winning=winning,
        winning_beliefs=winning_beliefs,
        strategy=strategy,
        knowledge_game_size=len(kg.game.vertices),
        explored_beliefs=len(kg.states)
    )


def solve_buchi(game: PartialObsGame,
                max_beliefs: int = 10000) -> POGameResult:
    """Solve a partial observation Buchi game.

    Player 1 wins if accepting states are visited infinitely often.
    """
    assert game.objective == Objective.BUCHI

    kg = build_knowledge_game(game, max_beliefs)
    if not kg.game.vertices:
        return POGameResult(
            winning=False, winning_beliefs=set(),
            knowledge_game_size=0, explored_beliefs=0
        )

    sol = zielonka(kg.game)
    winning = kg.initial_id in sol.win_even

    winning_beliefs = set()
    for ks, sid in kg.states.items():
        if sid in sol.win_even and ks.observation >= 0:
            winning_beliefs.add(ks.belief)

    strategy = _extract_obs_strategy(game, kg, sol)

    return POGameResult(
        winning=winning,
        winning_beliefs=winning_beliefs,
        strategy=strategy,
        knowledge_game_size=len(kg.game.vertices),
        explored_beliefs=len(kg.states)
    )


def solve_parity(game: PartialObsGame,
                 max_beliefs: int = 10000) -> POGameResult:
    """Solve a partial observation parity game (general case).

    Uses knowledge game construction + V156 Zielonka on expanded game.
    """
    kg = build_knowledge_game(game, max_beliefs)
    if not kg.game.vertices:
        return POGameResult(
            winning=False, winning_beliefs=set(),
            knowledge_game_size=0, explored_beliefs=0
        )

    sol = zielonka(kg.game)
    winning = kg.initial_id in sol.win_even

    winning_beliefs = set()
    for ks, sid in kg.states.items():
        if sid in sol.win_even and ks.observation >= 0:
            winning_beliefs.add(ks.belief)

    strategy = _extract_obs_strategy(game, kg, sol)

    return POGameResult(
        winning=winning,
        winning_beliefs=winning_beliefs,
        strategy=strategy,
        knowledge_game_size=len(kg.game.vertices),
        explored_beliefs=len(kg.states)
    )


def _extract_obs_strategy(game: PartialObsGame, kg: KnowledgeGame,
                           sol: Solution) -> Optional[ObsStrategy]:
    """Extract an observation-based strategy from knowledge game solution."""
    moves = {}

    for ks, sid in kg.states.items():
        if sid not in sol.win_even:
            continue
        if ks.observation < 0:
            continue
        # Check if this is a Player 1 vertex
        sample_v = next(iter(ks.belief), None)
        if sample_v is None:
            continue
        if game.owner.get(sample_v) != Player.EVEN:
            continue

        # Find the strategy move
        if sid in sol.strategy_even:
            succ_id = sol.strategy_even[sid]
            if succ_id in kg.id_to_state:
                succ_ks = kg.id_to_state[succ_id]
                if succ_ks.observation >= 0:
                    obs = ks.observation
                    if obs not in moves:
                        moves[obs] = succ_ks.observation

    return ObsStrategy(moves=moves) if moves else None


# --- Antichain-Based Optimization ---

def _is_subset(a: FrozenSet[int], b: FrozenSet[int]) -> bool:
    """Check if a is a subset of b."""
    return a <= b


def _is_superset(a: FrozenSet[int], b: FrozenSet[int]) -> bool:
    """Check if a is a superset of b."""
    return a >= b


def antichain_insert(antichain: List[FrozenSet[int]],
                     belief: FrozenSet[int],
                     direction: str = "down") -> List[FrozenSet[int]]:
    """Insert a belief into an antichain, maintaining the antichain property.

    direction="down": downward-closed (keep maximal elements)
    direction="up": upward-closed (keep minimal elements)
    """
    if direction == "down":
        # Keep maximal: remove elements that are subsets of belief
        if any(_is_superset(existing, belief) for existing in antichain):
            return antichain  # already dominated
        new_chain = [e for e in antichain if not _is_subset(e, belief)]
        new_chain.append(belief)
        return new_chain
    else:
        # Keep minimal: remove elements that are supersets of belief
        if any(_is_subset(existing, belief) for existing in antichain):
            return antichain  # already dominated
        new_chain = [e for e in antichain if not _is_superset(e, belief)]
        new_chain.append(belief)
        return new_chain


def antichain_contains(antichain: List[FrozenSet[int]],
                       belief: FrozenSet[int],
                       direction: str = "down") -> bool:
    """Check if belief is in the downward/upward closure of the antichain."""
    if direction == "down":
        return any(_is_superset(existing, belief) for existing in antichain)
    else:
        return any(_is_subset(existing, belief) for existing in antichain)


def solve_safety_antichain(game: PartialObsGame) -> POGameResult:
    """Solve safety game using antichain optimization.

    Computes maximal safe belief sets using backward fixed-point
    with antichain representation (avoids enumerating all beliefs).

    The losing region is computed as a least fixed-point:
    Lose = mu X. Bad | CPre_2(X)
    where Bad = beliefs intersecting target,
    CPre_2(X) = beliefs where Player 2 can force into X.

    Safe = complement of Lose (maximal safe beliefs).
    """
    bad = game.target
    all_obs = game.all_observations()

    # Group vertices by observation
    obs_to_verts = {}
    for v in game.vertices:
        o = game.obs[v]
        if o not in obs_to_verts:
            obs_to_verts[o] = set()
        obs_to_verts[o].add(v)

    # Direct fixed-point on per-observation safe sets
    # Safe[o] = maximal subsets of obs_class(o) that are safe
    # Initialize: for each obs, the full obs class is safe if no bad state
    safe_beliefs = {}  # obs -> list of maximal safe belief sets
    for o, verts in obs_to_verts.items():
        fv = frozenset(verts)
        if fv & bad:
            # Some states bad -- safe subset is those without bad
            safe_only = frozenset(verts - bad)
            if safe_only:
                safe_beliefs[o] = [safe_only]
            else:
                safe_beliefs[o] = []
        else:
            safe_beliefs[o] = [fv]

    # Fixed-point: remove beliefs where Player 2 can force into unsafe
    changed = True
    iterations = 0
    while changed and iterations < 100:
        changed = False
        iterations += 1

        for o, verts in obs_to_verts.items():
            if not safe_beliefs.get(o):
                continue

            new_safe = []
            for belief in safe_beliefs[o]:
                # Check: from this belief, can Player 1 guarantee staying safe?
                # For P1 vertices: there exists an action leading to a safe belief
                # For P2 vertices: all actions lead to safe beliefs
                still_safe = True

                # Compute successor observations
                succ_obs_map = _post_all(game, belief)

                if not succ_obs_map:
                    # Dead end -- safe for safety (no bad reached)
                    new_safe = antichain_insert(new_safe, belief, "down")
                    continue

                # Check if any P1 action keeps us safe
                p1_verts = {v for v in belief if game.owner.get(v) == Player.EVEN}
                p2_verts = {v for v in belief if game.owner.get(v) == Player.ODD}

                if p1_verts and not p2_verts:
                    # Pure P1 belief -- exists an action to safe successor
                    has_safe_action = False
                    for target_o, succ_belief in succ_obs_map.items():
                        if _belief_is_safe(succ_belief, safe_beliefs.get(target_o, [])):
                            has_safe_action = True
                            break
                    still_safe = has_safe_action

                elif p2_verts and not p1_verts:
                    # Pure P2 belief -- all actions must lead to safe
                    for target_o, succ_belief in succ_obs_map.items():
                        if not _belief_is_safe(succ_belief, safe_beliefs.get(target_o, [])):
                            still_safe = False
                            break

                else:
                    # Mixed -- check all successors are safe
                    for target_o, succ_belief in succ_obs_map.items():
                        if not _belief_is_safe(succ_belief, safe_beliefs.get(target_o, [])):
                            still_safe = False
                            break

                if still_safe:
                    new_safe = antichain_insert(new_safe, belief, "down")
                else:
                    changed = True

            safe_beliefs[o] = new_safe

    # Check initial belief
    init_belief = _initial_belief(game)
    init_obs_groups = _observation_split(game, init_belief)

    winning = True
    for o, belief in init_obs_groups.items():
        if not _belief_is_safe(belief, safe_beliefs.get(o, [])):
            winning = False
            break

    winning_belief_set = set()
    for o, beliefs in safe_beliefs.items():
        for b in beliefs:
            winning_belief_set.add(b)

    return POGameResult(
        winning=winning,
        winning_beliefs=winning_belief_set,
        knowledge_game_size=0,
        explored_beliefs=sum(len(v) for v in safe_beliefs.values())
    )


def _belief_is_safe(belief: FrozenSet[int],
                    safe_beliefs: List[FrozenSet[int]]) -> bool:
    """Check if belief is contained in some safe belief (downward-closed)."""
    return any(belief <= sb for sb in safe_beliefs)


# --- Game Construction Helpers ---

def make_safety_po_game(n_states: int, bad: Set[int],
                        even_states: Set[int],
                        transitions: List[Tuple[int, int]],
                        observations: Dict[int, int]) -> PartialObsGame:
    """Create a partial observation safety game.

    Even wins iff the play never visits bad states.
    """
    game = PartialObsGame(objective=Objective.SAFETY, target=bad)
    for s in range(n_states):
        player = Player.EVEN if s in even_states else Player.ODD
        obs = observations.get(s, s)  # default: perfect observation
        game.add_vertex(s, player, obs)
    for u, v in transitions:
        game.add_edge(u, v)
    game.initial = {0}
    return game


def make_reachability_po_game(n_states: int, target: Set[int],
                               even_states: Set[int],
                               transitions: List[Tuple[int, int]],
                               observations: Dict[int, int]) -> PartialObsGame:
    """Create a partial observation reachability game.

    Even wins iff the play reaches a target state.
    """
    game = PartialObsGame(objective=Objective.REACHABILITY, target=target)
    for s in range(n_states):
        player = Player.EVEN if s in even_states else Player.ODD
        obs = observations.get(s, s)
        game.add_vertex(s, player, obs)
    for u, v in transitions:
        game.add_edge(u, v)
    for t in target:
        game.add_edge(t, t)  # self-loops on target
    game.initial = {0}
    return game


def make_buchi_po_game(n_states: int, accepting: Set[int],
                        even_states: Set[int],
                        transitions: List[Tuple[int, int]],
                        observations: Dict[int, int]) -> PartialObsGame:
    """Create a partial observation Buchi game.

    Even wins iff accepting states are visited infinitely often.
    """
    game = PartialObsGame(objective=Objective.BUCHI, target=accepting)
    for s in range(n_states):
        player = Player.EVEN if s in even_states else Player.ODD
        obs = observations.get(s, s)
        game.add_vertex(s, player, obs)
    for u, v in transitions:
        game.add_edge(u, v)
    game.initial = {0}
    return game


def make_co_buchi_po_game(n_states: int, rejecting: Set[int],
                           even_states: Set[int],
                           transitions: List[Tuple[int, int]],
                           observations: Dict[int, int]) -> PartialObsGame:
    """Create a partial observation co-Buchi game."""
    game = PartialObsGame(objective=Objective.CO_BUCHI, target=rejecting)
    for s in range(n_states):
        player = Player.EVEN if s in even_states else Player.ODD
        obs = observations.get(s, s)
        game.add_vertex(s, player, obs)
    for u, v in transitions:
        game.add_edge(u, v)
    game.initial = {0}
    return game


# --- Analysis ---

def analyze_observability(game: PartialObsGame) -> Dict:
    """Analyze the observation structure of a game."""
    all_obs = game.all_observations()
    obs_classes = {o: game.obs_class(o) for o in all_obs}
    max_class_size = max((len(c) for c in obs_classes.values()), default=0)
    min_class_size = min((len(c) for c in obs_classes.values()), default=0)

    # Information ratio: |observations| / |vertices|
    # 1.0 = perfect info, lower = less info
    info_ratio = len(all_obs) / len(game.vertices) if game.vertices else 0

    # Check if observations are trivial (all same observation)
    trivial = len(all_obs) <= 1

    # Check if perfect information (each vertex has unique observation)
    perfect = len(all_obs) == len(game.vertices)

    return {
        'num_observations': len(all_obs),
        'num_vertices': len(game.vertices),
        'info_ratio': info_ratio,
        'max_class_size': max_class_size,
        'min_class_size': min_class_size,
        'is_perfect_info': perfect,
        'is_trivial': trivial,
        'obs_consistent': game.is_observation_consistent(),
        'obs_sizes': {o: len(c) for o, c in obs_classes.items()},
    }


def solve(game: PartialObsGame, max_beliefs: int = 10000) -> POGameResult:
    """Solve a partial observation game (dispatches by objective)."""
    if game.objective == Objective.SAFETY:
        return solve_safety(game, max_beliefs)
    elif game.objective == Objective.REACHABILITY:
        return solve_reachability(game, max_beliefs)
    elif game.objective == Objective.BUCHI:
        return solve_buchi(game, max_beliefs)
    elif game.objective == Objective.CO_BUCHI:
        return solve_parity(game, max_beliefs)  # general solver
    elif game.objective == Objective.PARITY:
        return solve_parity(game, max_beliefs)
    else:
        raise ValueError(f"Unknown objective: {game.objective}")


def compare_perfect_vs_partial(game: PartialObsGame) -> Dict:
    """Compare solving with perfect vs partial observation.

    Creates a perfect-info version (unique obs per vertex) and solves both.
    """
    # Solve partial observation version
    po_result = solve(game)

    # Create perfect-info version
    perfect_game = PartialObsGame(
        objective=game.objective,
        target=game.target.copy(),
        initial=game.initial.copy()
    )
    for v in game.vertices:
        perfect_game.add_vertex(v, game.owner[v], v, game.priority.get(v, 0))
    for u in game.vertices:
        for w in game.successors(u):
            perfect_game.add_edge(u, w)

    pi_result = solve(perfect_game)

    return {
        'partial_observation': {
            'winning': po_result.winning,
            'beliefs_explored': po_result.explored_beliefs,
            'knowledge_game_size': po_result.knowledge_game_size,
        },
        'perfect_information': {
            'winning': pi_result.winning,
            'beliefs_explored': pi_result.explored_beliefs,
            'knowledge_game_size': pi_result.knowledge_game_size,
        },
        'info_loss_matters': po_result.winning != pi_result.winning,
        'observability': analyze_observability(game),
    }


def game_statistics(game: PartialObsGame) -> Dict:
    """Compute statistics about a partial observation game."""
    n_edges = sum(len(s) for s in game.edges.values())
    even_verts = sum(1 for v in game.vertices if game.owner.get(v) == Player.EVEN)
    obs_info = analyze_observability(game)

    return {
        'vertices': len(game.vertices),
        'edges': n_edges,
        'even_vertices': even_verts,
        'odd_vertices': len(game.vertices) - even_verts,
        'objective': game.objective.value,
        'target_size': len(game.target),
        'initial_size': len(game.initial),
        **obs_info,
    }


def game_summary(game: PartialObsGame) -> str:
    """Human-readable summary of a partial observation game."""
    stats = game_statistics(game)
    result = solve(game)

    lines = [
        f"Partial Observation Game ({stats['objective']})",
        f"  Vertices: {stats['vertices']} ({stats['even_vertices']} Even, {stats['odd_vertices']} Odd)",
        f"  Edges: {stats['edges']}",
        f"  Observations: {stats['num_observations']} (info ratio: {stats['info_ratio']:.2f})",
        f"  Perfect info: {stats['is_perfect_info']}",
        f"  Target: {stats['target_size']} states",
        f"  Winner: {'Player 1 (Even)' if result.winning else 'Player 2 (Odd)'}",
        f"  Knowledge game: {result.knowledge_game_size} states",
        f"  Beliefs explored: {result.explored_beliefs}",
    ]
    if result.strategy:
        lines.append(f"  Strategy: {result.strategy.moves}")
    return "\n".join(lines)
