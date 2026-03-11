"""
V153: Game-based Bisimulation

Bisimulation characterized as a two-player game between Attacker (Spoiler)
and Defender (Duplicator). The Attacker tries to show two states are NOT
bisimilar; the Defender tries to maintain the bisimulation relation.

Game structure:
- Positions are pairs (s1, s2) of states from two LTSs (or same LTS)
- Attacker picks an action and a successor from one side
- Defender must match from the other side
- If Defender can't match -> Attacker wins (not bisimilar)
- If play goes on forever -> Defender wins (bisimilar)

This infinite-duration game is solved as a parity game (V076):
- Attacker-owned nodes have priority 1 (odd -> Attacker wins if visited infinitely)
- Defender-owned nodes have priority 0 (even -> Defender wins)
- Deadlock nodes for Defender have priority 1 (Attacker wins)

Composes: V076 (parity games) for game solving
"""

import sys
sys.path.insert(0, 'Z:/AgentZero/A2/work/V076_parity_games')

from parity_games import (
    ParityGame, ParityResult, Player, attractor, zielonka, solve,
    verify_strategy
)

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from enum import Enum


# --- LTS representation ---

@dataclass
class LTS:
    """Labeled Transition System for bisimulation games."""
    n_states: int
    actions: Set[str]
    transitions: Dict[int, List[Tuple[str, int]]]  # state -> [(action, target)]
    labels: Dict[int, Set[str]] = field(default_factory=dict)  # state -> atomic props

    def successors(self, state: int, action: str) -> Set[int]:
        """Get all successors of state under action."""
        return {t for a, t in self.transitions.get(state, []) if a == action}

    def enabled_actions(self, state: int) -> Set[str]:
        """Get all actions enabled at state."""
        return {a for a, _ in self.transitions.get(state, [])}

    def state_label(self, state: int) -> FrozenSet[str]:
        """Get label set for state (frozen for hashing)."""
        return frozenset(self.labels.get(state, set()))


def make_lts(n_states, transitions, labels=None):
    """Build LTS from transition list [(src, action, dst), ...]."""
    actions = set()
    trans_dict = {}
    for src, act, dst in transitions:
        actions.add(act)
        trans_dict.setdefault(src, []).append((act, dst))
    lab = {}
    if labels:
        lab = {s: set(props) for s, props in labels.items()}
    return LTS(n_states=n_states, actions=actions,
               transitions=trans_dict, labels=lab)


# --- Game positions ---

class NodeKind(Enum):
    ATTACKER = "attacker"     # Attacker picks action + side + successor
    DEFENDER = "defender"     # Defender must match the move
    DEADLOCK = "deadlock"     # Defender has no matching move (Attacker wins)


@dataclass
class GameNode:
    """A position in the bisimulation game."""
    kind: NodeKind
    s1: int  # state from LTS1
    s2: int  # state from LTS2
    # For defender nodes: which action was played and which successor was chosen
    action: Optional[str] = None
    chosen: Optional[int] = None  # the attacker's chosen successor
    side: Optional[int] = None    # 1 or 2: which side attacker moved on


@dataclass
class BisimGameResult:
    """Result of a bisimulation game."""
    bisimilar: bool
    attacker_wins: Set[Tuple[int, int]]   # pairs where attacker wins
    defender_wins: Set[Tuple[int, int]]   # pairs where defender wins
    strategy: Optional[Dict] = None       # winning strategy (for attacker or defender)
    game_size: int = 0                    # number of game positions
    parity_result: Optional[ParityResult] = None
    distinguishing_play: Optional[List] = None  # witness play for non-bisimilar


@dataclass
class FullGameResult:
    """Full result with game details."""
    bisimilar: bool
    game: Optional[ParityGame] = None
    parity_result: Optional[ParityResult] = None
    node_map: Optional[Dict] = None       # game node id -> GameNode
    attacker_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    defender_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    distinguishing_sequence: Optional[List[str]] = None


# --- Bisimulation Game Construction ---

def build_bisimulation_game(lts1: LTS, lts2: LTS,
                            check_labels: bool = True) -> Tuple[ParityGame, Dict[int, GameNode], Dict]:
    """
    Build a parity game encoding the bisimulation game between two LTSs.

    Attacker-owned nodes: (s1, s2) pairs -- Attacker picks action + side + successor
    Defender-owned nodes: (s1, s2, action, chosen, side) -- Defender must match

    Returns: (parity_game, node_id_to_GameNode, reverse_map)
    """
    game = ParityGame()
    node_map = {}       # int -> GameNode
    reverse_map = {}    # GameNode key -> int
    next_id = [0]

    def get_or_create(kind, s1, s2, action=None, chosen=None, side=None):
        key = (kind, s1, s2, action, chosen, side)
        if key in reverse_map:
            return reverse_map[key]
        nid = next_id[0]
        next_id[0] += 1
        node = GameNode(kind=kind, s1=s1, s2=s2, action=action,
                        chosen=chosen, side=side)
        node_map[nid] = node
        reverse_map[key] = nid

        if kind == NodeKind.ATTACKER:
            # Attacker (Spoiler) owns this node. Priority 0 = Defender wins if play stays here.
            game.add_node(nid, Player.ODD, 0)
        elif kind == NodeKind.DEFENDER:
            # Defender (Duplicator) owns this node. Priority 0 = Defender wins.
            game.add_node(nid, Player.EVEN, 0)
        elif kind == NodeKind.DEADLOCK:
            # Deadlock for defender = attacker wins. Priority 1 (odd).
            game.add_node(nid, Player.EVEN, 1)
        return nid

    # BFS to explore reachable game positions
    queue = []
    visited = set()

    # Start: create attacker node for each pair we want to check
    # For full game: all pairs. For specific check: just the pair of interest.
    for s1 in range(lts1.n_states):
        for s2 in range(lts2.n_states):
            # If labels differ, attacker wins immediately (label mismatch)
            if check_labels and lts1.state_label(s1) != lts2.state_label(s2):
                nid = get_or_create(NodeKind.DEADLOCK, s1, s2)
                # Self-loop to make parity game valid (every node needs a successor)
                game.add_edge(nid, nid)
                continue

            nid = get_or_create(NodeKind.ATTACKER, s1, s2)
            queue.append(nid)

    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = node_map[nid]

        if node.kind == NodeKind.ATTACKER:
            s1, s2 = node.s1, node.s2
            has_move = False

            # Attacker's moves: pick an action, a side, and a successor
            all_actions = lts1.enabled_actions(s1) | lts2.enabled_actions(s2)
            for action in all_actions:
                # Side 1: attacker moves in LTS1
                for t1 in lts1.successors(s1, action):
                    has_move = True
                    # Create defender node: defender must match from LTS2
                    def_id = get_or_create(NodeKind.DEFENDER, s1, s2,
                                           action=action, chosen=t1, side=1)
                    game.add_edge(nid, def_id)

                    # Defender's responses: pick a matching successor in LTS2
                    s2_succs = lts2.successors(s2, action)
                    if not s2_succs:
                        # No match -> deadlock for defender
                        _ensure_deadlock(game, def_id, node_map)
                    else:
                        for t2 in s2_succs:
                            # Next attacker node
                            next_att = get_or_create(NodeKind.ATTACKER, t1, t2)
                            game.add_edge(def_id, next_att)
                            if next_att not in visited:
                                queue.append(next_att)

                # Side 2: attacker moves in LTS2
                for t2 in lts2.successors(s2, action):
                    has_move = True
                    def_id = get_or_create(NodeKind.DEFENDER, s1, s2,
                                           action=action, chosen=t2, side=2)
                    game.add_edge(nid, def_id)

                    # Defender's responses: pick a matching successor in LTS1
                    s1_succs = lts1.successors(s1, action)
                    if not s1_succs:
                        _ensure_deadlock(game, def_id, node_map)
                    else:
                        for t1 in s1_succs:
                            next_att = get_or_create(NodeKind.ATTACKER, t1, t2)
                            game.add_edge(def_id, next_att)
                            if next_att not in visited:
                                queue.append(next_att)

            if not has_move:
                # Both states are deadlocked (no enabled actions) -> defender wins
                # (they're bisimilar if both stuck with same labels)
                game.add_edge(nid, nid)  # self-loop, priority 0 = defender wins

        elif node.kind == NodeKind.DEFENDER:
            # Already handled above during attacker processing
            if nid not in game.successors or not game.successors[nid]:
                _ensure_deadlock(game, nid, node_map)

    return game, node_map, reverse_map


def _ensure_deadlock(game, def_id, node_map):
    """Ensure a defender node with no moves leads to attacker winning."""
    # Defender can't move -> attacker wins. Set priority to 1 (odd).
    game.priority[def_id] = 1
    # Must have a successor for valid parity game
    game.add_edge(def_id, def_id)


# --- Solving ---

def check_bisimulation_game(lts1: LTS, lts2: LTS, s1: int = 0, s2: int = 0,
                            check_labels: bool = True,
                            algorithm: str = "zielonka") -> BisimGameResult:
    """
    Check if s1 in lts1 is bisimilar to s2 in lts2 using a game-based approach.

    Returns BisimGameResult with bisimilarity verdict, winning regions, and strategy.
    """
    pg, node_map, rev_map = build_bisimulation_game(lts1, lts2, check_labels)

    if not pg.nodes:
        return BisimGameResult(bisimilar=True, attacker_wins=set(),
                               defender_wins=set(), game_size=0)

    result = solve(pg, algorithm=algorithm)

    # Extract bisimilar/non-bisimilar pairs from attacker nodes
    attacker_wins_pairs = set()
    defender_wins_pairs = set()
    for nid, node in node_map.items():
        if node.kind == NodeKind.ATTACKER:
            pair = (node.s1, node.s2)
            if nid in result.win1:  # Odd player = Attacker
                attacker_wins_pairs.add(pair)
            else:
                defender_wins_pairs.add(pair)

    # Check the specific pair
    att_key = (NodeKind.ATTACKER, s1, s2, None, None, None)
    dl_key = (NodeKind.DEADLOCK, s1, s2, None, None, None)

    if att_key in rev_map:
        init_nid = rev_map[att_key]
        bisimilar = init_nid in result.win0
    elif dl_key in rev_map:
        bisimilar = False
    else:
        # Pair not in game (shouldn't happen for valid states)
        bisimilar = True

    # Extract distinguishing play if not bisimilar
    dist_play = None
    if not bisimilar:
        dist_play = _extract_distinguishing_play(pg, result, node_map, rev_map, s1, s2)

    return BisimGameResult(
        bisimilar=bisimilar,
        attacker_wins=attacker_wins_pairs,
        defender_wins=defender_wins_pairs,
        game_size=len(pg.nodes),
        parity_result=result,
        distinguishing_play=dist_play
    )


def _extract_distinguishing_play(pg, result, node_map, rev_map, s1, s2):
    """Extract a finite distinguishing play (attacker's winning strategy)."""
    play = []
    att_key = (NodeKind.ATTACKER, s1, s2, None, None, None)
    if att_key not in rev_map:
        return None

    current = rev_map[att_key]
    visited_positions = set()
    max_steps = 100

    for _ in range(max_steps):
        if current in visited_positions:
            break
        visited_positions.add(current)
        node = node_map.get(current)
        if node is None:
            break

        if node.kind == NodeKind.ATTACKER:
            play.append(('attacker', node.s1, node.s2))
            # Follow attacker's winning strategy (ODD = player 1)
            next_move = result.strategy1.get(current)
            if next_move is None:
                # Attacker-owned but no strategy -> pick any successor
                succs = pg.successors.get(current, set())
                if not succs:
                    break
                # Prefer moves leading to win1
                for s in succs:
                    if s in result.win1:
                        next_move = s
                        break
                if next_move is None:
                    next_move = next(iter(succs))
            current = next_move

        elif node.kind == NodeKind.DEFENDER:
            play.append(('defender_challenge', node.s1, node.s2,
                          node.action, node.chosen, node.side))
            # Defender's best response (EVEN = player 0)
            next_move = result.strategy0.get(current)
            if next_move is None:
                succs = pg.successors.get(current, set())
                if not succs:
                    break
                next_move = next(iter(succs))
            current = next_move

        elif node.kind == NodeKind.DEADLOCK:
            play.append(('deadlock', node.s1, node.s2))
            break

    return play


# --- Full game analysis ---

def full_bisimulation_game(lts1: LTS, lts2: LTS, s1: int = 0, s2: int = 0,
                           check_labels: bool = True) -> FullGameResult:
    """Full game construction and analysis with detailed results."""
    pg, node_map, rev_map = build_bisimulation_game(lts1, lts2, check_labels)

    if not pg.nodes:
        return FullGameResult(bisimilar=True)

    result = solve(pg)

    attacker_pairs = set()
    defender_pairs = set()
    for nid, node in node_map.items():
        if node.kind == NodeKind.ATTACKER:
            pair = (node.s1, node.s2)
            if nid in result.win1:
                attacker_pairs.add(pair)
            else:
                defender_pairs.add(pair)

    att_key = (NodeKind.ATTACKER, s1, s2, None, None, None)
    dl_key = (NodeKind.DEADLOCK, s1, s2, None, None, None)
    if att_key in rev_map:
        bisimilar = rev_map[att_key] in result.win0
    elif dl_key in rev_map:
        bisimilar = False
    else:
        bisimilar = True

    # Extract distinguishing sequence
    dist_seq = None
    if not bisimilar:
        dist_seq = _extract_action_sequence(pg, result, node_map, rev_map, s1, s2)

    return FullGameResult(
        bisimilar=bisimilar,
        game=pg,
        parity_result=result,
        node_map=node_map,
        attacker_pairs=attacker_pairs,
        defender_pairs=defender_pairs,
        distinguishing_sequence=dist_seq
    )


def _extract_action_sequence(pg, result, node_map, rev_map, s1, s2):
    """Extract the sequence of actions that distinguishes s1 from s2."""
    att_key = (NodeKind.ATTACKER, s1, s2, None, None, None)
    if att_key not in rev_map:
        return None

    actions = []
    current = rev_map[att_key]
    visited = set()

    for _ in range(50):
        if current in visited:
            break
        visited.add(current)
        node = node_map.get(current)
        if not node:
            break

        if node.kind == NodeKind.DEADLOCK:
            break
        elif node.kind == NodeKind.DEFENDER:
            actions.append(node.action)
            # Follow defender's move
            succs = pg.successors.get(current, set())
            if not succs:
                break
            current = next(iter(succs))
        elif node.kind == NodeKind.ATTACKER:
            # Follow attacker strategy
            next_move = result.strategy1.get(current)
            if next_move is None:
                succs = pg.successors.get(current, set())
                winning_succs = [s for s in succs if s in result.win1]
                if winning_succs:
                    next_move = winning_succs[0]
                elif succs:
                    next_move = next(iter(succs))
                else:
                    break
            current = next_move

    return actions if actions else None


# --- Weak bisimulation game ---

def build_weak_bisimulation_game(lts: LTS, s1: int, s2: int,
                                  tau_action: str = "tau") -> Tuple[ParityGame, Dict, Dict]:
    """
    Build a weak bisimulation game.

    In weak bisimulation:
    - Attacker plays action a from one side
    - Defender must match with tau* . a . tau* from the other side
      (or tau* if a is tau)

    We precompute weak successors and build the game over those.
    """
    # Precompute tau closures
    tau_closure = {}
    for s in range(lts.n_states):
        tau_closure[s] = _compute_tau_closure(lts, s, tau_action)

    # Compute weak successors: s =a=> T means exists s' in tau*(s), s' -a-> s'', s'' in tau*(s'')
    # For tau: s =tau=> T means tau*(s) (zero or more taus)
    def weak_successors(state, action):
        result = set()
        if action == tau_action:
            # Weak tau: just tau closure
            return tau_closure[state]
        else:
            # Weak a: tau* . a . tau*
            for s_mid in tau_closure[state]:
                for t in lts.successors(s_mid, action):
                    result.update(tau_closure[t])
        return result

    # Build game using weak successors
    game = ParityGame()
    node_map = {}
    reverse_map = {}
    next_id = [0]

    def get_or_create(kind, sa, sb, action=None, chosen=None, side=None):
        key = (kind, sa, sb, action, chosen, side)
        if key in reverse_map:
            return reverse_map[key]
        nid = next_id[0]
        next_id[0] += 1
        node = GameNode(kind=kind, s1=sa, s2=sb, action=action,
                        chosen=chosen, side=side)
        node_map[nid] = node
        reverse_map[key] = nid
        if kind == NodeKind.ATTACKER:
            game.add_node(nid, Player.ODD, 0)
        elif kind == NodeKind.DEFENDER:
            game.add_node(nid, Player.EVEN, 0)
        elif kind == NodeKind.DEADLOCK:
            game.add_node(nid, Player.EVEN, 1)
        return nid

    queue = []
    visited_nodes = set()

    init = get_or_create(NodeKind.ATTACKER, s1, s2)
    queue.append(init)

    while queue:
        nid = queue.pop(0)
        if nid in visited_nodes:
            continue
        visited_nodes.add(nid)
        node = node_map[nid]

        if node.kind != NodeKind.ATTACKER:
            continue

        sa, sb = node.s1, node.s2
        has_move = False

        # Check label compatibility
        if lts.state_label(sa) != lts.state_label(sb):
            game.priority[nid] = 1  # Attacker wins (label mismatch)
            game.add_edge(nid, nid)
            continue

        all_actions = lts.enabled_actions(sa) | lts.enabled_actions(sb)
        for action in all_actions:
            # Strong successors for attacker's concrete move
            # Side 1: attacker picks concrete a-successor from sa
            for ta in lts.successors(sa, action):
                has_move = True
                def_id = get_or_create(NodeKind.DEFENDER, sa, sb,
                                       action=action, chosen=ta, side=1)
                game.add_edge(nid, def_id)

                # Defender matches with WEAK successor from sb
                weak_tb = weak_successors(sb, action)
                if not weak_tb:
                    _ensure_deadlock(game, def_id, node_map)
                else:
                    for tb in weak_tb:
                        next_att = get_or_create(NodeKind.ATTACKER, ta, tb)
                        game.add_edge(def_id, next_att)
                        if next_att not in visited_nodes:
                            queue.append(next_att)

            # Side 2: attacker picks concrete a-successor from sb
            for tb in lts.successors(sb, action):
                has_move = True
                def_id = get_or_create(NodeKind.DEFENDER, sa, sb,
                                       action=action, chosen=tb, side=2)
                game.add_edge(nid, def_id)

                weak_ta = weak_successors(sa, action)
                if not weak_ta:
                    _ensure_deadlock(game, def_id, node_map)
                else:
                    for ta in weak_ta:
                        next_att = get_or_create(NodeKind.ATTACKER, ta, tb)
                        game.add_edge(def_id, next_att)
                        if next_att not in visited_nodes:
                            queue.append(next_att)

            # Side 1: attacker plays tau (internal) from sa
            if action == tau_action:
                for ta in lts.successors(sa, tau_action):
                    has_move = True
                    def_id = get_or_create(NodeKind.DEFENDER, sa, sb,
                                           action=tau_action, chosen=ta, side=1)
                    game.add_edge(nid, def_id)
                    # Defender matches tau with zero or more taus
                    for tb in tau_closure[sb]:
                        next_att = get_or_create(NodeKind.ATTACKER, ta, tb)
                        game.add_edge(def_id, next_att)
                        if next_att not in visited_nodes:
                            queue.append(next_att)

        if not has_move:
            game.add_edge(nid, nid)  # Both deadlocked, defender wins

    return game, node_map, reverse_map


def _compute_tau_closure(lts: LTS, state: int, tau_action: str) -> Set[int]:
    """Compute transitive closure under tau transitions."""
    closure = {state}
    worklist = [state]
    while worklist:
        s = worklist.pop()
        for t in lts.successors(s, tau_action):
            if t not in closure:
                closure.add(t)
                worklist.append(t)
    return closure


def check_weak_bisimulation_game(lts: LTS, s1: int = 0, s2: int = 0,
                                  tau_action: str = "tau") -> BisimGameResult:
    """Check weak bisimulation between s1 and s2 using game-based approach."""
    pg, node_map, rev_map = build_weak_bisimulation_game(lts, s1, s2, tau_action)

    if not pg.nodes:
        return BisimGameResult(bisimilar=True, attacker_wins=set(),
                               defender_wins=set(), game_size=0)

    result = solve(pg)

    attacker_wins = set()
    defender_wins = set()
    for nid, node in node_map.items():
        if node.kind == NodeKind.ATTACKER:
            pair = (node.s1, node.s2)
            if nid in result.win1:
                attacker_wins.add(pair)
            else:
                defender_wins.add(pair)

    att_key = (NodeKind.ATTACKER, s1, s2, None, None, None)
    bisimilar = att_key in rev_map and rev_map[att_key] in result.win0

    return BisimGameResult(
        bisimilar=bisimilar,
        attacker_wins=attacker_wins,
        defender_wins=defender_wins,
        game_size=len(pg.nodes),
        parity_result=result
    )


# --- Simulation game (asymmetric) ---

def check_simulation_game(lts1: LTS, lts2: LTS, s1: int = 0, s2: int = 0,
                          check_labels: bool = True) -> BisimGameResult:
    """
    Check if lts1.s1 is SIMULATED BY lts2.s2 using a game.

    Simulation is one-directional: attacker only moves on lts1's side,
    defender must match on lts2's side. If defender can always match,
    lts2.s2 simulates lts1.s1.
    """
    game = ParityGame()
    node_map = {}
    reverse_map = {}
    next_id = [0]

    def get_or_create(kind, sa, sb, action=None, chosen=None):
        key = (kind, sa, sb, action, chosen)
        if key in reverse_map:
            return reverse_map[key]
        nid = next_id[0]
        next_id[0] += 1
        node = GameNode(kind=kind, s1=sa, s2=sb, action=action, chosen=chosen, side=1)
        node_map[nid] = node
        reverse_map[key] = nid
        if kind == NodeKind.ATTACKER:
            game.add_node(nid, Player.ODD, 0)
        elif kind == NodeKind.DEFENDER:
            game.add_node(nid, Player.EVEN, 0)
        elif kind == NodeKind.DEADLOCK:
            game.add_node(nid, Player.EVEN, 1)
        return nid

    queue = []
    visited = set()

    init = get_or_create(NodeKind.ATTACKER, s1, s2)
    queue.append(init)

    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = node_map[nid]

        if node.kind != NodeKind.ATTACKER:
            continue

        sa, sb = node.s1, node.s2

        if check_labels and lts1.state_label(sa) != lts2.state_label(sb):
            game.priority[nid] = 1
            game.add_edge(nid, nid)
            continue

        has_move = False
        for action in lts1.enabled_actions(sa):
            for ta in lts1.successors(sa, action):
                has_move = True
                def_id = get_or_create(NodeKind.DEFENDER, sa, sb,
                                       action=action, chosen=ta)
                game.add_edge(nid, def_id)

                tb_set = lts2.successors(sb, action)
                if not tb_set:
                    _ensure_deadlock(game, def_id, node_map)
                else:
                    for tb in tb_set:
                        next_att = get_or_create(NodeKind.ATTACKER, ta, tb)
                        game.add_edge(def_id, next_att)
                        if next_att not in visited:
                            queue.append(next_att)

        if not has_move:
            game.add_edge(nid, nid)

    if not game.nodes:
        return BisimGameResult(bisimilar=True, attacker_wins=set(),
                               defender_wins=set(), game_size=0)

    result = solve(game)

    attacker_wins = set()
    defender_wins = set()
    for nid, node in node_map.items():
        if node.kind == NodeKind.ATTACKER:
            pair = (node.s1, node.s2)
            if nid in result.win1:
                attacker_wins.add(pair)
            else:
                defender_wins.add(pair)

    att_key = (NodeKind.ATTACKER, s1, s2, None, None)
    simulated = att_key in reverse_map and reverse_map[att_key] in result.win0

    return BisimGameResult(
        bisimilar=simulated,  # "bisimilar" field means "simulated" here
        attacker_wins=attacker_wins,
        defender_wins=defender_wins,
        game_size=len(game.nodes),
        parity_result=result
    )


# --- Comparison with partition refinement ---

def partition_bisimulation(lts: LTS) -> List[Set[int]]:
    """
    Standard partition refinement bisimulation (non-game-based).
    Used for comparison/validation.
    """
    # Initial partition: group by labels
    label_groups = {}
    for s in range(lts.n_states):
        lab = lts.state_label(s)
        label_groups.setdefault(lab, set()).add(s)
    partition = list(label_groups.values())

    changed = True
    while changed:
        changed = False
        new_partition = []
        for block in partition:
            splits = _split_block(lts, block, partition)
            if len(splits) > 1:
                changed = True
            new_partition.extend(splits)
        partition = new_partition

    return partition


def _split_block(lts: LTS, block: Set[int], partition: List[Set[int]]) -> List[Set[int]]:
    """Split a block based on transition signatures w.r.t. current partition."""
    block_index = {}
    for i, b in enumerate(partition):
        for s in b:
            block_index[s] = i

    signatures = {}
    for s in block:
        sig = []
        for action in sorted(lts.actions):
            targets = set()
            for t in lts.successors(s, action):
                targets.add(block_index.get(t, -1))
            sig.append((action, frozenset(targets)))
        signatures.setdefault(tuple(sig), set()).add(s)

    return list(signatures.values())


def compare_game_vs_partition(lts1: LTS, lts2: LTS = None,
                              s1: int = 0, s2: int = 0) -> Dict:
    """Compare game-based bisimulation with partition refinement."""
    if lts2 is None:
        lts2 = lts1

    game_result = check_bisimulation_game(lts1, lts2, s1, s2)

    # For same-LTS comparison, run partition refinement
    partition_result = None
    partition_bisim = None
    if lts1 is lts2:
        partition = partition_bisimulation(lts1)
        # Check if s1 and s2 are in the same block
        partition_bisim = any(s1 in block and s2 in block for block in partition)
        partition_result = {
            'n_blocks': len(partition),
            'partition': [sorted(b) for b in partition],
            'bisimilar': partition_bisim
        }

    return {
        'game_bisimilar': game_result.bisimilar,
        'partition_bisimilar': partition_bisim,
        'agree': game_result.bisimilar == partition_bisim if partition_bisim is not None else None,
        'game_size': game_result.game_size,
        'attacker_wins': len(game_result.attacker_wins),
        'defender_wins': len(game_result.defender_wins),
        'partition': partition_result
    }


# --- Convenience builders ---

def make_vending_machine_lts():
    """Classic example: two vending machines (bisimilar or not)."""
    # VM1: coin -> {coffee, tea}
    lts1 = make_lts(3, [
        (0, 'coin', 1),
        (1, 'coffee', 0),
        (1, 'tea', 0),
    ])
    # VM2: coin -> coffee OR coin -> tea (separate choices)
    lts2 = make_lts(4, [
        (0, 'coin', 1),
        (0, 'coin', 2),
        (1, 'coffee', 0),
        (2, 'tea', 0),
    ])
    return lts1, lts2


def make_bisimilar_pair():
    """Two LTSs that ARE bisimilar (different structure, same behavior)."""
    lts1 = make_lts(2, [
        (0, 'a', 1),
        (1, 'b', 0),
    ])
    lts2 = make_lts(3, [
        (0, 'a', 1),
        (0, 'a', 2),
        (1, 'b', 0),
        (2, 'b', 0),
    ])
    return lts1, lts2


def bisimulation_game_summary(lts1: LTS, lts2: LTS = None,
                               s1: int = 0, s2: int = 0) -> Dict:
    """High-level summary of the bisimulation game."""
    if lts2 is None:
        lts2 = lts1
    result = check_bisimulation_game(lts1, lts2, s1, s2)
    return {
        'bisimilar': result.bisimilar,
        'game_positions': result.game_size,
        'attacker_winning_pairs': len(result.attacker_wins),
        'defender_winning_pairs': len(result.defender_wins),
        'has_distinguishing_play': result.distinguishing_play is not None,
        'distinguishing_play_length': len(result.distinguishing_play) if result.distinguishing_play else 0
    }
