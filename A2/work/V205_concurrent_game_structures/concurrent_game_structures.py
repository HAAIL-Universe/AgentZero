"""
V205: Concurrent Game Structures -- ATL/ATL* Model Checking

Concurrent game structures model multi-agent systems where agents
simultaneously choose actions and the combined choice determines the
next state. ATL (Alternating-time Temporal Logic) reasons about what
coalitions of agents can enforce.

Composes:
- V156 (parity games) -- for ATL* strategy synthesis via game reduction
- V023 (LTL model checking) -- for LTL formula/automaton construction

Key concepts:
- CGS: states, agents, action functions, transition function
- ATL: <<A>>X phi, <<A>>G phi, <<A>>F phi, <<A>> phi U psi
- ATL*: <<A>> psi where psi is arbitrary LTL over path (more expressive)
- ATL model checking: fixed-point computation (polynomial in |states|)
- ATL* model checking: reduction to parity games (EXPTIME-complete)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, FrozenSet, List, Optional, Set, Tuple, Callable, Any
)
from itertools import product as cartesian_product

# Compose V156 parity games
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V156_parity_games'))
from parity_games import (
    ParityGame, Player, Solution as ParitySolution,
    zielonka, verify_solution as verify_parity,
    make_game as make_parity_game
)

# Compose V023 LTL model checking
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V023_ltl_model_checking'))
from ltl_model_checker import (
    LTL, LTLOp, Atom, LTLTrue, LTLFalse,
    Not as LTLNot, And as LTLAnd, Or as LTLOr,
    Next as LTLNext, Finally as LTLFinally, Globally as LTLGlobally,
    Until as LTLUntil, Release as LTLRelease,
    nnf, atoms as ltl_atoms, subformulas,
    ltl_to_gba, gba_to_nba, NBA
)


# ---------------------------------------------------------------------------
# ATL Formula Representation
# ---------------------------------------------------------------------------

class ATLOp(Enum):
    """ATL formula operators."""
    ATOM = auto()       # Atomic proposition
    TRUE = auto()
    FALSE = auto()
    NOT = auto()        # Boolean negation
    AND = auto()        # Boolean conjunction
    OR = auto()         # Boolean disjunction
    IMPLIES = auto()    # Boolean implication
    # Coalition modalities: <<A>> temporal
    COAL_NEXT = auto()     # <<A>> X phi
    COAL_GLOBALLY = auto() # <<A>> G phi
    COAL_FINALLY = auto()  # <<A>> F phi
    COAL_UNTIL = auto()    # <<A>> phi U psi
    # ATL* extension: <<A>> psi where psi is LTL
    COAL_PATH = auto()     # <<A>> psi (psi is an LTL path formula)


@dataclass(frozen=True)
class ATL:
    """ATL/ATL* formula."""
    op: ATLOp
    name: str = ""                          # For ATOM
    coalition: FrozenSet[str] = frozenset()  # Agent names in <<A>>
    left: Optional[ATL] = None              # Sub-formula
    right: Optional[ATL] = None             # Second sub-formula (Until)
    path_formula: Optional[LTL] = None      # For COAL_PATH (ATL*)

    def __repr__(self):
        if self.op == ATLOp.ATOM:
            return self.name
        if self.op == ATLOp.TRUE:
            return "true"
        if self.op == ATLOp.FALSE:
            return "false"
        if self.op == ATLOp.NOT:
            return f"!{self.left}"
        if self.op == ATLOp.AND:
            return f"({self.left} & {self.right})"
        if self.op == ATLOp.OR:
            return f"({self.left} | {self.right})"
        if self.op == ATLOp.IMPLIES:
            return f"({self.left} -> {self.right})"
        c = "{" + ",".join(sorted(self.coalition)) + "}"
        if self.op == ATLOp.COAL_NEXT:
            return f"<<{c}>>X {self.left}"
        if self.op == ATLOp.COAL_GLOBALLY:
            return f"<<{c}>>G {self.left}"
        if self.op == ATLOp.COAL_FINALLY:
            return f"<<{c}>>F {self.left}"
        if self.op == ATLOp.COAL_UNTIL:
            return f"<<{c}>>({self.left} U {self.right})"
        if self.op == ATLOp.COAL_PATH:
            return f"<<{c}>> {self.path_formula}"
        return f"ATL({self.op})"


# ATL formula constructors
def ATLAtom(name: str) -> ATL:
    return ATL(op=ATLOp.ATOM, name=name)

def ATLTrue_() -> ATL:
    return ATL(op=ATLOp.TRUE)

def ATLFalse_() -> ATL:
    return ATL(op=ATLOp.FALSE)

def ATLNot(f: ATL) -> ATL:
    return ATL(op=ATLOp.NOT, left=f)

def ATLAnd(a: ATL, b: ATL) -> ATL:
    return ATL(op=ATLOp.AND, left=a, right=b)

def ATLOr(a: ATL, b: ATL) -> ATL:
    return ATL(op=ATLOp.OR, left=a, right=b)

def ATLImplies(a: ATL, b: ATL) -> ATL:
    return ATL(op=ATLOp.IMPLIES, left=a, right=b)

def CoalNext(coalition: Set[str], phi: ATL) -> ATL:
    return ATL(op=ATLOp.COAL_NEXT, coalition=frozenset(coalition), left=phi)

def CoalGlobally(coalition: Set[str], phi: ATL) -> ATL:
    return ATL(op=ATLOp.COAL_GLOBALLY, coalition=frozenset(coalition), left=phi)

def CoalFinally(coalition: Set[str], phi: ATL) -> ATL:
    return ATL(op=ATLOp.COAL_FINALLY, coalition=frozenset(coalition), left=phi)

def CoalUntil(coalition: Set[str], phi: ATL, psi: ATL) -> ATL:
    return ATL(op=ATLOp.COAL_UNTIL, coalition=frozenset(coalition),
               left=phi, right=psi)

def CoalPath(coalition: Set[str], psi: LTL) -> ATL:
    """ATL* coalition path quantifier: <<A>> psi where psi is LTL."""
    return ATL(op=ATLOp.COAL_PATH, coalition=frozenset(coalition),
               path_formula=psi)


# ---------------------------------------------------------------------------
# Concurrent Game Structure
# ---------------------------------------------------------------------------

@dataclass
class ConcurrentGameStructure:
    """
    Multi-agent concurrent game structure.

    At each state, every agent simultaneously chooses an action.
    The joint action (one per agent) determines the successor state.

    Attributes:
        states: Set of state identifiers
        agents: List of agent names
        actions: For each agent, for each state: set of available actions
                 actions[agent][state] -> Set[action_id]
        transition: Maps (state, joint_action_tuple) -> next_state
        labeling: Maps state -> set of atomic propositions true there
        initial: Set of initial states
    """
    states: Set[int] = field(default_factory=set)
    agents: List[str] = field(default_factory=list)
    actions: Dict[str, Dict[int, Set[int]]] = field(default_factory=dict)
    transition: Dict[Tuple[int, Tuple[int, ...]], int] = field(default_factory=dict)
    labeling: Dict[int, Set[str]] = field(default_factory=dict)
    initial: Set[int] = field(default_factory=set)

    def add_state(self, s: int, labels: Optional[Set[str]] = None,
                  initial: bool = False):
        """Add a state with optional labeling."""
        self.states.add(s)
        if labels:
            self.labeling[s] = labels
        else:
            self.labeling.setdefault(s, set())
        if initial:
            self.initial.add(s)

    def add_agent(self, name: str):
        """Register an agent."""
        if name not in self.agents:
            self.agents.append(name)
            self.actions[name] = {}

    def set_actions(self, agent: str, state: int, acts: Set[int]):
        """Set available actions for agent at state."""
        self.actions[agent][state] = acts

    def set_transition(self, state: int, joint_action: Tuple[int, ...],
                       next_state: int):
        """Set transition for a state and joint action vector."""
        self.transition[(state, joint_action)] = next_state

    def get_actions(self, agent: str, state: int) -> Set[int]:
        """Get available actions for agent at state."""
        return self.actions.get(agent, {}).get(state, set())

    def joint_actions(self, state: int) -> List[Tuple[int, ...]]:
        """Get all joint action tuples at a state."""
        per_agent = []
        for ag in self.agents:
            acts = sorted(self.get_actions(ag, state))
            if not acts:
                return []
            per_agent.append(acts)
        return list(cartesian_product(*per_agent))

    def successors(self, state: int) -> Set[int]:
        """Get all possible successor states from a state."""
        result = set()
        for ja in self.joint_actions(state):
            ns = self.transition.get((state, ja))
            if ns is not None:
                result.add(ns)
        return result

    def coalition_effectiveness(self, state: int, coalition: FrozenSet[str],
                                target: Set[int]) -> bool:
        """
        Can the coalition force the next state into target,
        regardless of what opponents do?

        Coalition picks their actions; opponents try all combinations.
        Returns True iff there exists a coalition action profile such that
        ALL opponent completions lead to a state in target.
        """
        coal_indices = [i for i, ag in enumerate(self.agents)
                        if ag in coalition]
        opp_indices = [i for i, ag in enumerate(self.agents)
                       if ag not in coalition]

        # Coalition action profiles
        coal_acts = []
        for i in coal_indices:
            acts = sorted(self.get_actions(self.agents[i], state))
            if not acts:
                return False
            coal_acts.append(acts)

        # Opponent action profiles
        opp_acts = []
        for i in opp_indices:
            acts = sorted(self.get_actions(self.agents[i], state))
            if not acts:
                acts = [0]  # dummy if no opponents
            opp_acts.append(acts)

        coal_profiles = list(cartesian_product(*coal_acts)) if coal_acts else [()]
        opp_profiles = list(cartesian_product(*opp_acts)) if opp_acts else [()]

        for cp in coal_profiles:
            all_good = True
            for op in opp_profiles:
                # Build joint action
                ja = [0] * len(self.agents)
                for idx, ci in enumerate(coal_indices):
                    ja[ci] = cp[idx]
                for idx, oi in enumerate(opp_indices):
                    ja[oi] = op[idx]
                ja_tuple = tuple(ja)
                ns = self.transition.get((state, ja_tuple))
                if ns is None or ns not in target:
                    all_good = False
                    break
            if all_good:
                return True
        return False

    def validate(self) -> List[str]:
        """Check structural validity."""
        errors = []
        for s in self.states:
            for ja in self.joint_actions(s):
                if (s, ja) not in self.transition:
                    errors.append(
                        f"Missing transition at state {s}, action {ja}")
        for s in self.initial:
            if s not in self.states:
                errors.append(f"Initial state {s} not in states")
        return errors


# ---------------------------------------------------------------------------
# ATL Model Checking (fixed-point computation)
# ---------------------------------------------------------------------------

@dataclass
class ATLResult:
    """Result of ATL model checking."""
    formula: ATL
    satisfaction_set: Set[int]  # States where formula holds
    holds_in_initial: bool      # Whether all initial states satisfy
    strategy: Optional[Dict[int, Dict[str, int]]] = None
    # strategy[state][agent] -> action chosen by coalition


def _pre_coalition(cgs: ConcurrentGameStructure,
                   coalition: FrozenSet[str],
                   target: Set[int]) -> Set[int]:
    """
    Compute Pre_A(target): states where coalition A can force
    the next state into target, regardless of opponents.
    """
    result = set()
    for s in cgs.states:
        if cgs.coalition_effectiveness(s, coalition, target):
            result.add(s)
    return result


def check_atl(cgs: ConcurrentGameStructure, formula: ATL) -> ATLResult:
    """
    Model check an ATL formula on a concurrent game structure.

    Uses fixed-point computation:
    - <<A>>X phi: Pre_A([[phi]])
    - <<A>>G phi: nu Z. [[phi]] & Pre_A(Z)  (greatest fixpoint)
    - <<A>>F phi: mu Z. [[phi]] | Pre_A(Z)  (least fixpoint)
    - <<A>>(phi U psi): mu Z. [[psi]] | ([[phi]] & Pre_A(Z))

    Returns ATLResult with satisfaction set and initial-state check.
    """
    sat = _check_atl_recursive(cgs, formula)
    holds = cgs.initial.issubset(sat)
    return ATLResult(
        formula=formula,
        satisfaction_set=sat,
        holds_in_initial=holds,
    )


def _check_atl_recursive(cgs: ConcurrentGameStructure,
                          formula: ATL) -> Set[int]:
    """Recursively compute satisfaction set for ATL formula."""
    op = formula.op

    if op == ATLOp.TRUE:
        return set(cgs.states)
    if op == ATLOp.FALSE:
        return set()
    if op == ATLOp.ATOM:
        return {s for s in cgs.states
                if formula.name in cgs.labeling.get(s, set())}
    if op == ATLOp.NOT:
        inner = _check_atl_recursive(cgs, formula.left)
        return cgs.states - inner
    if op == ATLOp.AND:
        left = _check_atl_recursive(cgs, formula.left)
        right = _check_atl_recursive(cgs, formula.right)
        return left & right
    if op == ATLOp.OR:
        left = _check_atl_recursive(cgs, formula.left)
        right = _check_atl_recursive(cgs, formula.right)
        return left | right
    if op == ATLOp.IMPLIES:
        left = _check_atl_recursive(cgs, formula.left)
        right = _check_atl_recursive(cgs, formula.right)
        return (cgs.states - left) | right

    coal = formula.coalition

    if op == ATLOp.COAL_NEXT:
        phi_set = _check_atl_recursive(cgs, formula.left)
        return _pre_coalition(cgs, coal, phi_set)

    if op == ATLOp.COAL_GLOBALLY:
        # nu Z. [[phi]] & Pre_A(Z)  -- greatest fixpoint
        phi_set = _check_atl_recursive(cgs, formula.left)
        z = set(cgs.states)  # Start with all states
        while True:
            z_new = phi_set & _pre_coalition(cgs, coal, z)
            if z_new == z:
                return z
            z = z_new

    if op == ATLOp.COAL_FINALLY:
        # mu Z. [[phi]] | Pre_A(Z)  -- least fixpoint
        phi_set = _check_atl_recursive(cgs, formula.left)
        z = set()  # Start with empty set
        while True:
            z_new = phi_set | _pre_coalition(cgs, coal, z)
            if z_new == z:
                return z
            z = z_new

    if op == ATLOp.COAL_UNTIL:
        # mu Z. [[psi]] | ([[phi]] & Pre_A(Z))
        phi_set = _check_atl_recursive(cgs, formula.left)
        psi_set = _check_atl_recursive(cgs, formula.right)
        z = set()
        while True:
            z_new = psi_set | (phi_set & _pre_coalition(cgs, coal, z))
            if z_new == z:
                return z
            z = z_new

    if op == ATLOp.COAL_PATH:
        # ATL*: reduce to parity game
        return _check_atl_star(cgs, coal, formula.path_formula)

    raise ValueError(f"Unknown ATL operator: {op}")


# ---------------------------------------------------------------------------
# ATL* Model Checking (parity game reduction)
# ---------------------------------------------------------------------------

def _check_atl_star(cgs: ConcurrentGameStructure,
                    coalition: FrozenSet[str],
                    path_formula: LTL) -> Set[int]:
    """
    ATL* model checking via parity game reduction.

    1. Convert negated LTL path formula to Buchi automaton
    2. Build product game: CGS x automaton
    3. Coalition = Odd player, Opponents = Even player
    4. Solve parity game; Odd's winning region projected to CGS states
       gives satisfaction set (coalition can avoid negated formula)
    """
    # Build automaton for the negated path formula
    neg_formula = nnf(LTL(op=LTLOp.NOT, left=path_formula))
    gba = ltl_to_gba(neg_formula)
    nba = gba_to_nba(gba)

    if not nba.states or not nba.initial:
        # Negation has no accepting runs -> formula holds everywhere
        return set(cgs.states)

    # Check if automaton has any transitions at all
    has_transitions = any(
        len(nba.transitions.get(q, [])) > 0 for q in nba.states
    )
    if not has_transitions:
        # Automaton stuck at initial state with no transitions
        # -> negated formula unsatisfiable -> original holds everywhere
        return set(cgs.states)

    # Build product parity game: CGS x NBA
    pg, state_map = _build_product_game(cgs, coalition, nba)

    if not pg.vertices:
        return set(cgs.states)

    # Solve the parity game
    sol = zielonka(pg)

    # Encoding: Odd = coalition, Even = opponents
    # Odd wins => coalition can avoid negated formula's acceptance
    # => original formula holds
    # Project Odd's winning region to CGS states (phase-0 vertices only)
    sat = set()
    for pv in sol.win_odd:
        if pv in state_map:
            sat.add(state_map[pv])

    # States not in product: automaton can't start there -> formula holds
    states_in_product = set(state_map.values())
    for s in cgs.states:
        if s not in states_in_product:
            sat.add(s)

    return sat


def _build_product_game(
    cgs: ConcurrentGameStructure,
    coalition: FrozenSet[str],
    nba: NBA
) -> Tuple[ParityGame, Dict[int, int]]:
    """
    Build product parity game from CGS and NBA.

    Encoding:
    - Odd player = coalition (wants to AVOID negated formula acceptance)
    - Even player = opponents (wants to SATISFY negated formula)
    - Buchi-to-parity: accepting aut states -> priority 2, non-accepting -> 1
      Even wins if max inf priority is even (=2, accepting visited inf often)
    - Sink vertex: priority 1 (odd), Odd wins. Used when automaton dies
      (negated formula cannot be satisfied from that point).

    For multi-agent: coalition chooses first (Odd), opponents resolve (Even).
    """
    pg = ParityGame()
    state_map = {}  # product_vertex -> cgs_state (phase-0 only)
    vid = 0
    vertex_ids = {}  # (cgs_state, aut_state) -> vertex_id for phase 0

    coal_indices = [i for i, ag in enumerate(cgs.agents) if ag in coalition]
    opp_indices = [i for i, ag in enumerate(cgs.agents) if ag not in coalition]

    from collections import deque

    # BFS to find reachable (cgs_state, aut_state) pairs
    initial_pairs = set()
    for s in cgs.states:
        for q in nba.initial:
            initial_pairs.add((s, q))

    visited = set()
    queue = deque(initial_pairs)
    reachable = set()

    while queue:
        pair = queue.popleft()
        if pair in visited:
            continue
        visited.add(pair)
        reachable.add(pair)
        s, q = pair

        labels = cgs.labeling.get(s, set())
        for label, q_next in nba.transitions.get(q, []):
            if _label_matches(label, labels):
                for s_next in cgs.successors(s):
                    if (s_next, q_next) not in visited:
                        queue.append((s_next, q_next))

    if not reachable:
        return pg, state_map

    # Create SINK vertex: Odd-owned, priority 1, self-loop
    # Represents "automaton died" = negated formula violated = coalition wins
    sink = vid
    vid += 1
    pg.add_vertex(sink, Player.ODD, 1)
    pg.add_edge(sink, sink)

    # Create phase-0 vertices for reachable pairs
    for (s, q) in reachable:
        v0 = vid
        vid += 1
        # Buchi parity: accepting -> 2, non-accepting -> 1
        prio = 2 if q in nba.accepting else 1
        pg.add_vertex(v0, Player.ODD, prio)
        vertex_ids[(s, q)] = v0
        state_map[v0] = s

    # Create edges
    for (s, q) in reachable:
        v0 = vertex_ids[(s, q)]
        labels = cgs.labeling.get(s, set())

        # Find matching automaton transitions at this state
        aut_succs = []
        for label, q_next in nba.transitions.get(q, []):
            if _label_matches(label, labels):
                aut_succs.append(q_next)

        if not aut_succs:
            # Automaton dead end -> negated formula can't continue -> sink
            pg.add_edge(v0, sink)
            continue

        if not opp_indices:
            # No opponents -- coalition controls everything
            for ja in cgs.joint_actions(s):
                ns = cgs.transition.get((s, ja))
                if ns is None:
                    continue
                added = False
                for q_next in aut_succs:
                    if (ns, q_next) in vertex_ids:
                        pg.add_edge(v0, vertex_ids[(ns, q_next)])
                        added = True
                if not added:
                    # CGS successor exists but no matching aut state -> sink
                    pg.add_edge(v0, sink)

        elif not coal_indices:
            # No coalition -- opponents control everything
            pg.owner[v0] = Player.EVEN
            for ja in cgs.joint_actions(s):
                ns = cgs.transition.get((s, ja))
                if ns is None:
                    continue
                added = False
                for q_next in aut_succs:
                    if (ns, q_next) in vertex_ids:
                        pg.add_edge(v0, vertex_ids[(ns, q_next)])
                        added = True
                if not added:
                    pg.add_edge(v0, sink)
        else:
            # Both coalition and opponents have choices
            coal_acts = []
            for ci in coal_indices:
                acts = sorted(cgs.get_actions(cgs.agents[ci], s))
                if not acts:
                    acts = [0]
                coal_acts.append(acts)
            coal_profiles = list(cartesian_product(*coal_acts))

            opp_acts = []
            for oi in opp_indices:
                acts = sorted(cgs.get_actions(cgs.agents[oi], s))
                if not acts:
                    acts = [0]
                opp_acts.append(acts)
            opp_profiles = list(cartesian_product(*opp_acts))

            for cp in coal_profiles:
                # Intermediate: Even (opponent) chooses among opponent profiles
                v_mid = vid
                vid += 1
                pg.add_vertex(v_mid, Player.EVEN, pg.priority[v0])
                pg.add_edge(v0, v_mid)

                for op in opp_profiles:
                    ja = [0] * len(cgs.agents)
                    for idx, ci in enumerate(coal_indices):
                        ja[ci] = cp[idx]
                    for idx, oi in enumerate(opp_indices):
                        ja[oi] = op[idx]
                    ns = cgs.transition.get((s, tuple(ja)))
                    if ns is None:
                        continue
                    added = False
                    for q_next in aut_succs:
                        if (ns, q_next) in vertex_ids:
                            pg.add_edge(v_mid, vertex_ids[(ns, q_next)])
                            added = True
                    if not added:
                        pg.add_edge(v_mid, sink)

    # Ensure no dead ends (self-loop at priority preserving semantics)
    for v in list(pg.vertices):
        if v != sink and not pg.successors(v):
            pg.add_edge(v, sink)

    return pg, state_map


def _label_matches(label, state_labels: Set[str]) -> bool:
    """Check if a Buchi automaton label matches state propositions."""
    for p in label.pos:
        if p not in state_labels:
            return False
    for n in label.neg:
        if n in state_labels:
            return False
    return True


# ---------------------------------------------------------------------------
# Strategy Extraction
# ---------------------------------------------------------------------------

def extract_coalition_strategy(
    cgs: ConcurrentGameStructure,
    coalition: FrozenSet[str],
    formula: ATL
) -> Dict[int, Dict[str, int]]:
    """
    Extract a witness strategy for the coalition.

    For each state in the satisfaction set, returns an action for each
    coalition member that enforces the temporal property.

    Returns: {state: {agent: action}} for states where formula holds.
    """
    result = check_atl(cgs, formula)
    sat = result.satisfaction_set
    strategy = {}

    if formula.op == ATLOp.COAL_NEXT:
        target = _check_atl_recursive(cgs, formula.left)
        for s in sat:
            action = _find_coalition_action(cgs, coalition, s, target)
            if action:
                strategy[s] = action

    elif formula.op == ATLOp.COAL_GLOBALLY:
        # Strategy: always stay in sat (which is a fixpoint)
        for s in sat:
            action = _find_coalition_action(cgs, coalition, s, sat)
            if action:
                strategy[s] = action

    elif formula.op == ATLOp.COAL_FINALLY:
        target = _check_atl_recursive(cgs, formula.left)
        # Strategy: if in target, done; otherwise move toward target
        for s in sat:
            if s in target:
                # Any action works from target
                action = _find_any_coalition_action(cgs, coalition, s, sat)
                if action:
                    strategy[s] = action
            else:
                action = _find_coalition_action(cgs, coalition, s, sat)
                if action:
                    strategy[s] = action

    elif formula.op == ATLOp.COAL_UNTIL:
        phi_set = _check_atl_recursive(cgs, formula.left)
        psi_set = _check_atl_recursive(cgs, formula.right)
        for s in sat:
            if s in psi_set:
                action = _find_any_coalition_action(cgs, coalition, s, sat)
                if action:
                    strategy[s] = action
            else:
                action = _find_coalition_action(cgs, coalition, s, sat)
                if action:
                    strategy[s] = action

    return strategy


def _find_coalition_action(
    cgs: ConcurrentGameStructure,
    coalition: FrozenSet[str],
    state: int,
    target: Set[int]
) -> Optional[Dict[str, int]]:
    """Find a coalition action profile that forces next state into target."""
    coal_indices = [i for i, ag in enumerate(cgs.agents) if ag in coalition]
    opp_indices = [i for i, ag in enumerate(cgs.agents) if ag not in coalition]

    coal_acts = []
    for ci in coal_indices:
        acts = sorted(cgs.get_actions(cgs.agents[ci], state))
        if not acts:
            return None
        coal_acts.append(acts)

    opp_acts = []
    for oi in opp_indices:
        acts = sorted(cgs.get_actions(cgs.agents[oi], state))
        if not acts:
            acts = [0]
        opp_acts.append(acts)

    coal_profiles = list(cartesian_product(*coal_acts)) if coal_acts else [()]
    opp_profiles = list(cartesian_product(*opp_acts)) if opp_acts else [()]

    for cp in coal_profiles:
        all_good = True
        for op in opp_profiles:
            ja = [0] * len(cgs.agents)
            for idx, ci in enumerate(coal_indices):
                ja[ci] = cp[idx]
            for idx, oi in enumerate(opp_indices):
                ja[oi] = op[idx]
            ns = cgs.transition.get((state, tuple(ja)))
            if ns is None or ns not in target:
                all_good = False
                break
        if all_good:
            result = {}
            for idx, ci in enumerate(coal_indices):
                result[cgs.agents[ci]] = cp[idx]
            return result
    return None


def _find_any_coalition_action(
    cgs: ConcurrentGameStructure,
    coalition: FrozenSet[str],
    state: int,
    safe: Set[int]
) -> Optional[Dict[str, int]]:
    """Find any coalition action that keeps state in safe set if possible."""
    action = _find_coalition_action(cgs, coalition, state, safe)
    if action:
        return action
    # Fallback: pick first available actions
    result = {}
    for ag in cgs.agents:
        if ag in coalition:
            acts = sorted(cgs.get_actions(ag, state))
            if acts:
                result[ag] = acts[0]
    return result if result else None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_play(
    cgs: ConcurrentGameStructure,
    strategy: Dict[int, Dict[str, int]],
    opponent_strategy: Optional[Dict[int, Dict[str, int]]] = None,
    start: Optional[int] = None,
    coalition: Optional[FrozenSet[str]] = None,
    max_steps: int = 50
) -> List[Tuple[int, Tuple[int, ...], Set[str]]]:
    """
    Simulate a play in the CGS using given strategies.

    Returns list of (state, joint_action, labels) tuples.
    Opponents use provided strategy or default to action 0.
    """
    if start is None:
        start = min(cgs.initial) if cgs.initial else min(cgs.states)
    if coalition is None:
        coalition = frozenset()

    trace = []
    state = start
    visited_states = []

    for _ in range(max_steps):
        labels = cgs.labeling.get(state, set())

        # Build joint action
        ja = []
        for ag in cgs.agents:
            if ag in coalition and state in strategy:
                act = strategy[state].get(ag, 0)
            elif opponent_strategy and state in opponent_strategy:
                act = opponent_strategy[state].get(ag, 0)
            else:
                acts = sorted(cgs.get_actions(ag, state))
                act = acts[0] if acts else 0
            ja.append(act)

        ja_tuple = tuple(ja)
        trace.append((state, ja_tuple, labels))

        # Check for cycle
        if state in visited_states:
            break
        visited_states.append(state)

        # Transition
        next_state = cgs.transition.get((state, ja_tuple))
        if next_state is None:
            break
        state = next_state

    return trace


# ---------------------------------------------------------------------------
# Analysis and Comparison
# ---------------------------------------------------------------------------

def coalition_power(cgs: ConcurrentGameStructure) -> Dict[FrozenSet[str], Set[int]]:
    """
    For every possible coalition, compute the set of states from which
    the coalition can force reaching any target in one step.
    This gives a "power index" for coalitions.
    """
    from itertools import combinations
    result = {}
    all_agents = set(cgs.agents)

    for size in range(len(cgs.agents) + 1):
        for combo in combinations(cgs.agents, size):
            coal = frozenset(combo)
            # States where coalition can force ANY successor
            power_states = set()
            for s in cgs.states:
                for s2 in cgs.successors(s):
                    if cgs.coalition_effectiveness(s, coal, {s2}):
                        power_states.add(s)
                        break
            result[coal] = power_states

    return result


def game_statistics(cgs: ConcurrentGameStructure) -> Dict[str, Any]:
    """Compute statistics about the game structure."""
    total_transitions = len(cgs.transition)
    total_actions = sum(
        len(acts) for agent_acts in cgs.actions.values()
        for acts in agent_acts.values()
    )
    all_labels = set()
    for labels in cgs.labeling.values():
        all_labels.update(labels)

    return {
        "states": len(cgs.states),
        "agents": len(cgs.agents),
        "agent_names": list(cgs.agents),
        "transitions": total_transitions,
        "total_action_slots": total_actions,
        "propositions": sorted(all_labels),
        "initial_states": len(cgs.initial),
    }


def compare_coalitions(
    cgs: ConcurrentGameStructure,
    formula: ATL,
    coalitions: List[Set[str]]
) -> Dict[str, Dict[str, Any]]:
    """Compare what different coalitions can achieve for a formula template."""
    results = {}
    for coal in coalitions:
        coal_frozen = frozenset(coal)
        # Reconstruct formula with new coalition
        new_formula = _replace_coalition(formula, coal_frozen)
        result = check_atl(cgs, new_formula)
        results[str(sorted(coal))] = {
            "satisfaction_set": sorted(result.satisfaction_set),
            "holds_initial": result.holds_in_initial,
            "fraction": (len(result.satisfaction_set) / len(cgs.states)
                        if cgs.states else 0),
        }
    return results


def _replace_coalition(formula: ATL, new_coal: FrozenSet[str]) -> ATL:
    """Replace the coalition in a formula's top-level modality."""
    if formula.op in (ATLOp.COAL_NEXT, ATLOp.COAL_GLOBALLY,
                      ATLOp.COAL_FINALLY):
        return ATL(op=formula.op, coalition=new_coal,
                   left=formula.left, right=formula.right)
    if formula.op == ATLOp.COAL_UNTIL:
        return ATL(op=formula.op, coalition=new_coal,
                   left=formula.left, right=formula.right)
    if formula.op == ATLOp.COAL_PATH:
        return ATL(op=formula.op, coalition=new_coal,
                   path_formula=formula.path_formula)
    return formula


# ---------------------------------------------------------------------------
# Example Game Constructors
# ---------------------------------------------------------------------------

def make_simple_voting_game(n_voters: int = 3) -> ConcurrentGameStructure:
    """
    Simple majority voting game.
    Each voter simultaneously votes 0 or 1.
    If majority votes 1, next state is "pass"; otherwise "fail".
    From pass/fail, loops back to voting state.
    """
    cgs = ConcurrentGameStructure()
    # States: 0 = voting, 1 = pass, 2 = fail
    cgs.add_state(0, {"voting"}, initial=True)
    cgs.add_state(1, {"pass"})
    cgs.add_state(2, {"fail"})

    for i in range(n_voters):
        name = f"v{i}"
        cgs.add_agent(name)
        cgs.set_actions(name, 0, {0, 1})  # vote 0 or 1
        cgs.set_actions(name, 1, {0})      # only action at pass
        cgs.set_actions(name, 2, {0})      # only action at fail

    threshold = n_voters // 2 + 1

    # Transitions from voting state
    for ja in cgs.joint_actions(0):
        votes_for = sum(ja)
        if votes_for >= threshold:
            cgs.set_transition(0, ja, 1)  # pass
        else:
            cgs.set_transition(0, ja, 2)  # fail

    # Transitions from pass/fail back to voting
    for ja in cgs.joint_actions(1):
        cgs.set_transition(1, ja, 0)
    for ja in cgs.joint_actions(2):
        cgs.set_transition(2, ja, 0)

    return cgs


def make_train_gate_game() -> ConcurrentGameStructure:
    """
    Train-gate controller game (classic ATL example).
    Two agents: train (requests entry) and controller (opens/closes gate).

    States: idle, request, enter, crash
    Train: 0=idle, 1=request
    Controller: 0=close, 1=open
    """
    cgs = ConcurrentGameStructure()
    cgs.add_state(0, {"idle"}, initial=True)
    cgs.add_state(1, {"request"})
    cgs.add_state(2, {"in_tunnel"})
    cgs.add_state(3, {"crash"})

    cgs.add_agent("train")
    cgs.add_agent("controller")

    # Train actions
    cgs.set_actions("train", 0, {0, 1})       # idle: stay or request
    cgs.set_actions("train", 1, {0, 1})       # request: wait or enter
    cgs.set_actions("train", 2, {0})           # in tunnel: exit
    cgs.set_actions("train", 3, {0})           # crash: stuck

    # Controller actions
    cgs.set_actions("controller", 0, {0})      # idle: nothing
    cgs.set_actions("controller", 1, {0, 1})   # request: deny or open
    cgs.set_actions("controller", 2, {0})      # in tunnel: nothing
    cgs.set_actions("controller", 3, {0})      # crash: nothing

    # idle: train stays idle -> idle; train requests -> request
    cgs.set_transition(0, (0, 0), 0)
    cgs.set_transition(0, (1, 0), 1)

    # request: train waits (0) + controller denies (0) -> request
    # train waits (0) + controller opens (1) -> request (train didn't move)
    # train enters (1) + controller denies (0) -> crash!
    # train enters (1) + controller opens (1) -> enter
    cgs.set_transition(1, (0, 0), 1)
    cgs.set_transition(1, (0, 1), 1)
    cgs.set_transition(1, (1, 0), 3)
    cgs.set_transition(1, (1, 1), 2)

    # in tunnel: always goes back to idle
    cgs.set_transition(2, (0, 0), 0)

    # crash: self-loop
    cgs.set_transition(3, (0, 0), 3)

    return cgs


def make_resource_allocation_game(
    n_processes: int = 2,
    n_resources: int = 1
) -> ConcurrentGameStructure:
    """
    Resource allocation game.
    Multiple processes compete for limited resources.
    Each process: 0=idle, 1=request. Allocator: grants to one process.
    """
    cgs = ConcurrentGameStructure()

    # State encoding: tuple of process states + who holds resource
    # Simplify to 2 processes, 1 resource
    # States: (p0_state, p1_state, holder)
    # p_state: 0=idle, 1=requesting, 2=holding
    # holder: -1=free, 0=p0, 1=p1

    state_id = 0
    state_map = {}
    reverse_map = {}

    for p0 in range(3):
        for p1 in range(3):
            for h in [-1, 0, 1]:
                # Validity: if holder=0, p0 must be 2; if holder=1, p1 must be 2
                # Both can't hold simultaneously (1 resource)
                if p0 == 2 and p1 == 2:
                    continue
                if h == 0 and p0 != 2:
                    continue
                if h == 1 and p1 != 2:
                    continue
                if h == -1 and (p0 == 2 or p1 == 2):
                    continue

                labels = set()
                if p0 == 0:
                    labels.add("p0_idle")
                elif p0 == 1:
                    labels.add("p0_req")
                elif p0 == 2:
                    labels.add("p0_has")
                if p1 == 0:
                    labels.add("p1_idle")
                elif p1 == 1:
                    labels.add("p1_req")
                elif p1 == 2:
                    labels.add("p1_has")
                if h == -1:
                    labels.add("free")

                is_init = (p0 == 0 and p1 == 0 and h == -1)
                cgs.add_state(state_id, labels, initial=is_init)
                state_map[(p0, p1, h)] = state_id
                reverse_map[state_id] = (p0, p1, h)
                state_id += 1

    # Agents: p0, p1, allocator
    cgs.add_agent("p0")
    cgs.add_agent("p1")
    cgs.add_agent("allocator")

    for sid in cgs.states:
        p0, p1, h = reverse_map[sid]

        # p0 actions: 0=nop, 1=request (if idle), 2=release (if holding)
        p0_acts = {0}
        if p0 == 0:
            p0_acts.add(1)
        if p0 == 2:
            p0_acts.add(2)

        # p1 actions
        p1_acts = {0}
        if p1 == 0:
            p1_acts.add(1)
        if p1 == 2:
            p1_acts.add(2)

        # allocator actions: 0=nop, 1=grant_p0, 2=grant_p1
        alloc_acts = {0}
        if h == -1:
            if p0 == 1:
                alloc_acts.add(1)
            if p1 == 1:
                alloc_acts.add(2)

        cgs.set_actions("p0", sid, p0_acts)
        cgs.set_actions("p1", sid, p1_acts)
        cgs.set_actions("allocator", sid, alloc_acts)

        for ja in cgs.joint_actions(sid):
            a0, a1, alloc = ja
            np0, np1, nh = p0, p1, h

            # Process actions
            if a0 == 1 and p0 == 0:
                np0 = 1
            if a0 == 2 and p0 == 2:
                np0 = 0
                if nh == 0:
                    nh = -1
            if a1 == 1 and p1 == 0:
                np1 = 1
            if a1 == 2 and p1 == 2:
                np1 = 0
                if nh == 1:
                    nh = -1

            # Allocator grants (after release)
            if alloc == 1 and nh == -1 and np0 == 1:
                np0 = 2
                nh = 0
            elif alloc == 2 and nh == -1 and np1 == 1:
                np1 = 2
                nh = 1

            target = (np0, np1, nh)
            if target in state_map:
                cgs.set_transition(sid, ja, state_map[target])

    return cgs


def make_pursuit_evasion_game(grid_size: int = 3) -> ConcurrentGameStructure:
    """
    Simple pursuit-evasion on a grid.
    Two agents: pursuer and evader.
    Each can move N/S/E/W or stay.
    Pursuer wins if they reach the same cell as evader.
    """
    cgs = ConcurrentGameStructure()

    # State: (pursuer_pos, evader_pos) where pos = (row, col) encoded as int
    def pos_to_id(r, c):
        return r * grid_size + c

    def id_to_pos(pid):
        return pid // grid_size, pid % grid_size

    n = grid_size * grid_size
    state_id = 0
    state_map = {}
    reverse_map = {}

    for pp in range(n):
        for ep in range(n):
            labels = set()
            pr, pc = id_to_pos(pp)
            er, ec = id_to_pos(ep)
            if pp == ep:
                labels.add("caught")
            labels.add(f"p_{pr}_{pc}")
            labels.add(f"e_{er}_{ec}")
            is_init = (pp == 0 and ep == n - 1)  # corners
            cgs.add_state(state_id, labels, initial=is_init)
            state_map[(pp, ep)] = state_id
            reverse_map[state_id] = (pp, ep)
            state_id += 1

    cgs.add_agent("pursuer")
    cgs.add_agent("evader")

    def neighbors(pos):
        r, c = id_to_pos(pos)
        result = [pos]  # stay
        if r > 0:
            result.append(pos_to_id(r - 1, c))
        if r < grid_size - 1:
            result.append(pos_to_id(r + 1, c))
        if c > 0:
            result.append(pos_to_id(r, c - 1))
        if c < grid_size - 1:
            result.append(pos_to_id(r, c + 1))
        return result

    for sid in cgs.states:
        pp, ep = reverse_map[sid]
        p_moves = neighbors(pp)
        e_moves = neighbors(ep)

        # Encode moves as action ids (index into neighbor list)
        p_act_map = {i: m for i, m in enumerate(p_moves)}
        e_act_map = {i: m for i, m in enumerate(e_moves)}

        cgs.set_actions("pursuer", sid, set(range(len(p_moves))))
        cgs.set_actions("evader", sid, set(range(len(e_moves))))

        for pa in range(len(p_moves)):
            for ea in range(len(e_moves)):
                np = p_act_map[pa]
                ne = e_act_map[ea]
                # If already caught, stay caught
                if pp == ep:
                    cgs.set_transition(sid, (pa, ea), sid)
                else:
                    target = (np, ne)
                    if target in state_map:
                        cgs.set_transition(sid, (pa, ea), state_map[target])

    return cgs
