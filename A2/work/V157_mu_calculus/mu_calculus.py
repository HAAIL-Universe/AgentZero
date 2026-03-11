"""V157: Mu-Calculus Model Checking

Modal mu-calculus: propositional logic + modal operators (Diamond/Box) +
fixpoint operators (mu = least, nu = greatest).

Two model checking approaches:
1. Direct fixpoint iteration (Emerson-Lei) -- evaluate formulas as state sets
2. Parity game reduction -- convert to parity game, solve with V156

Composes: V156 (parity games), V021 (BDD model checking, optional)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict, FrozenSet, List, Optional, Set, Tuple, Union
)
import sys
import os

# ---------------------------------------------------------------------------
# Mu-calculus formula AST
# ---------------------------------------------------------------------------

class Formula:
    """Base class for mu-calculus formulas."""
    pass


@dataclass(frozen=True)
class Prop(Formula):
    """Atomic proposition: holds in states labeled with `name`."""
    name: str
    def __repr__(self): return f"Prop({self.name})"


@dataclass(frozen=True)
class Var(Formula):
    """Fixpoint variable reference."""
    name: str
    def __repr__(self): return f"Var({self.name})"


@dataclass(frozen=True)
class TT(Formula):
    """True -- holds in all states."""
    def __repr__(self): return "TT"


@dataclass(frozen=True)
class FF(Formula):
    """False -- holds in no states."""
    def __repr__(self): return "FF"


@dataclass(frozen=True)
class Not(Formula):
    """Negation (for closed formulas or at proposition level)."""
    sub: Formula
    def __repr__(self): return f"Not({self.sub})"


@dataclass(frozen=True)
class And(Formula):
    """Conjunction."""
    left: Formula
    right: Formula
    def __repr__(self): return f"And({self.left}, {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    """Disjunction."""
    left: Formula
    right: Formula
    def __repr__(self): return f"Or({self.left}, {self.right})"


@dataclass(frozen=True)
class Diamond(Formula):
    """Existential modality: <action> phi.
    There exists a successor via `action` satisfying `phi`.
    action=None means any action."""
    action: Optional[str]
    sub: Formula
    def __repr__(self):
        a = self.action or "*"
        return f"<{a}>{self.sub}"


@dataclass(frozen=True)
class Box(Formula):
    """Universal modality: [action] phi.
    All successors via `action` satisfy `phi`.
    action=None means any action."""
    action: Optional[str]
    sub: Formula
    def __repr__(self):
        a = self.action or "*"
        return f"[{a}]{self.sub}"


@dataclass(frozen=True)
class Mu(Formula):
    """Least fixpoint: mu X. phi(X)."""
    var: str
    body: Formula
    def __repr__(self): return f"mu {self.var}. {self.body}"


@dataclass(frozen=True)
class Nu(Formula):
    """Greatest fixpoint: nu X. phi(X)."""
    var: str
    body: Formula
    def __repr__(self): return f"nu {self.var}. {self.body}"


# ---------------------------------------------------------------------------
# Labeled Transition System (LTS)
# ---------------------------------------------------------------------------

@dataclass
class LTS:
    """Labeled Transition System.

    states: set of state identifiers (ints)
    transitions: dict mapping state -> list of (action, target_state)
    labels: dict mapping state -> set of atomic proposition names
    """
    states: Set[int]
    transitions: Dict[int, List[Tuple[str, int]]]
    labels: Dict[int, Set[str]]

    def successors(self, state: int, action: Optional[str] = None) -> Set[int]:
        """Get successor states, optionally filtered by action."""
        result = set()
        for a, t in self.transitions.get(state, []):
            if action is None or a == action:
                result.add(t)
        return result

    def predecessors(self, state: int, action: Optional[str] = None) -> Set[int]:
        """Get predecessor states, optionally filtered by action."""
        result = set()
        for s in self.states:
            for a, t in self.transitions.get(s, []):
                if t == state and (action is None or a == action):
                    result.add(s)
        return result

    def actions(self) -> Set[str]:
        """Get all action labels."""
        acts = set()
        for trans_list in self.transitions.values():
            for a, _ in trans_list:
                acts.add(a)
        return acts


def make_lts(
    num_states: int,
    transitions: List[Tuple[int, str, int]],
    labels: Dict[int, Set[str]]
) -> LTS:
    """Create an LTS from a compact description.

    transitions: list of (source, action, target) triples
    labels: dict mapping state -> set of proposition names
    """
    states = set(range(num_states))
    trans_dict: Dict[int, List[Tuple[str, int]]] = {s: [] for s in states}
    for src, act, tgt in transitions:
        trans_dict[src].append((act, tgt))
    lab = {s: labels.get(s, set()) for s in states}
    return LTS(states=states, transitions=trans_dict, labels=lab)


# ---------------------------------------------------------------------------
# Positive Normal Form (PNF) conversion
# ---------------------------------------------------------------------------

def to_pnf(formula: Formula) -> Formula:
    """Convert formula to Positive Normal Form (negation only at propositions).

    Uses De Morgan, dual modalities, and fixpoint duality:
      ~(phi & psi) = ~phi | ~psi
      ~(phi | psi) = ~phi & ~psi
      ~<a>phi = [a]~phi
      ~[a]phi = <a>~phi
      ~(mu X. phi) = nu X. ~phi[X/~X]  (with variable negation)
      ~(nu X. phi) = mu X. ~phi[X/~X]
    """
    return _push_negation(formula, negated=False)


def _push_negation(f: Formula, negated: bool) -> Formula:
    if isinstance(f, TT):
        return FF() if negated else TT()
    if isinstance(f, FF):
        return TT() if negated else FF()
    if isinstance(f, Prop):
        return Not(f) if negated else f
    if isinstance(f, Var):
        # Variables under negation handled by fixpoint duality
        return Not(f) if negated else f
    if isinstance(f, Not):
        return _push_negation(f.sub, not negated)
    if isinstance(f, And):
        if negated:
            return Or(_push_negation(f.left, True), _push_negation(f.right, True))
        return And(_push_negation(f.left, False), _push_negation(f.right, False))
    if isinstance(f, Or):
        if negated:
            return And(_push_negation(f.left, True), _push_negation(f.right, True))
        return Or(_push_negation(f.left, False), _push_negation(f.right, False))
    if isinstance(f, Diamond):
        if negated:
            return Box(f.action, _push_negation(f.sub, True))
        return Diamond(f.action, _push_negation(f.sub, False))
    if isinstance(f, Box):
        if negated:
            return Diamond(f.action, _push_negation(f.sub, True))
        return Box(f.action, _push_negation(f.sub, False))
    if isinstance(f, Mu):
        if negated:
            # ~(mu X. phi(X)) = nu X. ~phi(~X) -- but we handle var neg via Not(Var)
            return Nu(f.var, _push_negation(f.body, True))
        return Mu(f.var, _push_negation(f.body, False))
    if isinstance(f, Nu):
        if negated:
            return Mu(f.var, _push_negation(f.body, True))
        return Nu(f.var, _push_negation(f.body, False))
    raise ValueError(f"Unknown formula type: {type(f)}")


# ---------------------------------------------------------------------------
# Subformula enumeration and alternation depth
# ---------------------------------------------------------------------------

def subformulas(f: Formula) -> List[Formula]:
    """Get all subformulas (post-order)."""
    result = []
    visited = set()

    def visit(g: Formula):
        key = repr(g)
        if key in visited:
            return
        visited.add(key)
        if isinstance(g, (TT, FF, Prop, Var)):
            pass
        elif isinstance(g, Not):
            visit(g.sub)
        elif isinstance(g, (And, Or)):
            visit(g.left)
            visit(g.right)
        elif isinstance(g, (Diamond, Box)):
            visit(g.sub)
        elif isinstance(g, (Mu, Nu)):
            visit(g.body)
        result.append(g)

    visit(f)
    return result


def free_vars(f: Formula) -> Set[str]:
    """Get free fixpoint variables in formula."""
    if isinstance(f, (TT, FF, Prop)):
        return set()
    if isinstance(f, Var):
        return {f.name}
    if isinstance(f, Not):
        return free_vars(f.sub)
    if isinstance(f, (And, Or)):
        return free_vars(f.left) | free_vars(f.right)
    if isinstance(f, (Diamond, Box)):
        return free_vars(f.sub)
    if isinstance(f, (Mu, Nu)):
        return free_vars(f.body) - {f.var}
    return set()


def is_closed(f: Formula) -> bool:
    """Check if formula has no free variables."""
    return len(free_vars(f)) == 0


def alternation_depth(f: Formula) -> int:
    """Compute the alternation depth of a mu-calculus formula.

    Alternation depth counts the maximum nesting of mu inside nu (or vice versa)
    where the outer variable is free in the inner body.
    """
    return _alt_depth(f, {})


def _alt_depth(f: Formula, env: Dict[str, str]) -> int:
    """env maps variable name -> 'mu' or 'nu' (its binder type)."""
    if isinstance(f, (TT, FF, Prop)):
        return 0
    if isinstance(f, Var):
        return 0
    if isinstance(f, Not):
        return _alt_depth(f.sub, env)
    if isinstance(f, (And, Or)):
        return max(_alt_depth(f.left, env), _alt_depth(f.right, env))
    if isinstance(f, (Diamond, Box)):
        return _alt_depth(f.sub, env)
    if isinstance(f, Mu):
        new_env = dict(env)
        new_env[f.var] = 'mu'
        body_depth = _alt_depth(f.body, new_env)
        # Check if any nu-bound variable free in body causes alternation
        extra = 0
        for v in free_vars(f.body):
            if v != f.var and env.get(v) == 'nu':
                extra = 1
        return body_depth + extra
    if isinstance(f, Nu):
        new_env = dict(env)
        new_env[f.var] = 'nu'
        body_depth = _alt_depth(f.body, new_env)
        extra = 0
        for v in free_vars(f.body):
            if v != f.var and env.get(v) == 'mu':
                extra = 1
        return body_depth + extra
    return 0


def fixpoint_nesting_depth(f: Formula) -> int:
    """Compute the maximum nesting depth of fixpoint operators."""
    if isinstance(f, (TT, FF, Prop, Var)):
        return 0
    if isinstance(f, Not):
        return fixpoint_nesting_depth(f.sub)
    if isinstance(f, (And, Or)):
        return max(fixpoint_nesting_depth(f.left), fixpoint_nesting_depth(f.right))
    if isinstance(f, (Diamond, Box)):
        return fixpoint_nesting_depth(f.sub)
    if isinstance(f, (Mu, Nu)):
        return 1 + fixpoint_nesting_depth(f.body)
    return 0


# ---------------------------------------------------------------------------
# Direct fixpoint model checking (Emerson-Lei / Naive)
# ---------------------------------------------------------------------------

def eval_formula(lts: LTS, formula: Formula, env: Optional[Dict[str, Set[int]]] = None) -> Set[int]:
    """Evaluate a mu-calculus formula on an LTS.

    Returns the set of states satisfying the formula.

    Uses direct fixpoint iteration:
    - mu X. phi: start from empty set, iterate phi until fixpoint
    - nu X. phi: start from all states, iterate phi until fixpoint
    """
    if env is None:
        env = {}
    return _eval(lts, formula, env)


def _eval(lts: LTS, f: Formula, env: Dict[str, Set[int]]) -> Set[int]:
    if isinstance(f, TT):
        return set(lts.states)

    if isinstance(f, FF):
        return set()

    if isinstance(f, Prop):
        return {s for s in lts.states if f.name in lts.labels.get(s, set())}

    if isinstance(f, Var):
        if f.name in env:
            return set(env[f.name])
        raise ValueError(f"Unbound variable: {f.name}")

    if isinstance(f, Not):
        if isinstance(f.sub, Prop):
            prop_states = _eval(lts, f.sub, env)
            return lts.states - prop_states
        if isinstance(f.sub, Var):
            if f.sub.name in env:
                return lts.states - env[f.sub.name]
            raise ValueError(f"Unbound variable: {f.sub.name}")
        # General negation on closed subformula
        sub_states = _eval(lts, f.sub, env)
        return lts.states - sub_states

    if isinstance(f, And):
        left = _eval(lts, f.left, env)
        right = _eval(lts, f.right, env)
        return left & right

    if isinstance(f, Or):
        left = _eval(lts, f.left, env)
        right = _eval(lts, f.right, env)
        return left | right

    if isinstance(f, Diamond):
        sub = _eval(lts, f.sub, env)
        # States that have a successor (via action) in sub
        result = set()
        for s in lts.states:
            succs = lts.successors(s, f.action)
            if succs & sub:
                result.add(s)
        return result

    if isinstance(f, Box):
        sub = _eval(lts, f.sub, env)
        # States where ALL successors (via action) are in sub
        result = set()
        for s in lts.states:
            succs = lts.successors(s, f.action)
            if succs <= sub:  # empty set <= any set is True (vacuous truth)
                result.add(s)
        return result

    if isinstance(f, Mu):
        # Least fixpoint: start from empty, iterate
        current = set()
        for _ in range(len(lts.states) + 1):
            new_env = dict(env)
            new_env[f.var] = current
            next_val = _eval(lts, f.body, new_env)
            if next_val == current:
                return current
            current = next_val
        return current  # should have converged

    if isinstance(f, Nu):
        # Greatest fixpoint: start from all states, iterate
        current = set(lts.states)
        for _ in range(len(lts.states) + 1):
            new_env = dict(env)
            new_env[f.var] = current
            next_val = _eval(lts, f.body, new_env)
            if next_val == current:
                return current
            current = next_val
        return current

    raise ValueError(f"Unknown formula type: {type(f)}")


# ---------------------------------------------------------------------------
# Parity game reduction
# ---------------------------------------------------------------------------

def _register_subformulas(f: Formula, get_index):
    """Register all subformulas depth-first."""
    if isinstance(f, (TT, FF, Prop, Var)):
        get_index(f)
        return
    if isinstance(f, Not):
        _register_subformulas(f.sub, get_index)
        get_index(f)
        return
    if isinstance(f, (And, Or)):
        _register_subformulas(f.left, get_index)
        _register_subformulas(f.right, get_index)
        get_index(f)
        return
    if isinstance(f, (Diamond, Box)):
        _register_subformulas(f.sub, get_index)
        get_index(f)
        return
    if isinstance(f, (Mu, Nu)):
        _register_subformulas(f.body, get_index)
        get_index(f)
        return


def _collect_fixpoints(f: Formula, info: Dict, depth: int):
    """Collect fixpoint variable info: name -> (type, depth)."""
    if isinstance(f, (TT, FF, Prop, Var)):
        return
    if isinstance(f, Not):
        _collect_fixpoints(f.sub, info, depth)
    elif isinstance(f, (And, Or)):
        _collect_fixpoints(f.left, info, depth)
        _collect_fixpoints(f.right, info, depth)
    elif isinstance(f, (Diamond, Box)):
        _collect_fixpoints(f.sub, info, depth)
    elif isinstance(f, Mu):
        info[f.var] = ('mu', depth)
        _collect_fixpoints(f.body, info, depth + 1)
    elif isinstance(f, Nu):
        info[f.var] = ('nu', depth)
        _collect_fixpoints(f.body, info, depth + 1)


def _game_vertex_info(
    f: Formula, fi: int, s: int, lts: LTS,
    sub_list: List[Formula], get_index, fp_info, var_priority
) -> Tuple:
    """Determine owner, priority, and successors for a game vertex.

    Returns (Player, priority: int, successors: List[int]).

    Terminal nodes use self-loops with even priority (true) or odd priority (false).
    At a self-loop, the owner is irrelevant (only one edge choice).

    Convention for non-terminal:
    - Or, Diamond, Nu nodes: owned by Even (existential player, prover)
    - And, Box, Mu nodes: owned by Odd (universal player, refuter)
    """
    v156_dir = os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games')
    if v156_dir not in sys.path:
        sys.path.insert(0, v156_dir)
    from parity_games import Player

    me = _v_id(fi, s)  # self-loop target for terminal nodes

    if isinstance(f, TT):
        return (Player.EVEN, 0, [me])  # self-loop, even prio -> Even wins

    if isinstance(f, FF):
        return (Player.ODD, 1, [me])  # self-loop, odd prio -> Odd wins

    if isinstance(f, Prop):
        if f.name in lts.labels.get(s, set()):
            return (Player.EVEN, 0, [me])  # true
        else:
            return (Player.ODD, 1, [me])  # false

    if isinstance(f, Not):
        if isinstance(f.sub, Prop):
            if f.sub.name not in lts.labels.get(s, set()):
                return (Player.EVEN, 0, [me])  # negated prop true
            else:
                return (Player.ODD, 1, [me])  # negated prop false
        if isinstance(f.sub, Var):
            var_idx = get_index(f.sub)
            target = _v_id(var_idx, s)
            return (Player.ODD, 0, [target])
        # General negation
        sub_idx = get_index(f.sub)
        target = _v_id(sub_idx, s)
        return (Player.ODD, 0, [target])

    if isinstance(f, Var):
        var_name = f.name
        if var_name in var_priority:
            prio = var_priority[var_name]
        else:
            prio = 0
        binding_fi = _find_binding(sub_list, var_name)
        if binding_fi is not None:
            binding_f = sub_list[binding_fi]
            body_idx = get_index(binding_f.body)
            target = _v_id(body_idx, s)
            owner = Player.EVEN if isinstance(binding_f, Nu) else Player.ODD
            return (owner, prio, [target])
        return (Player.ODD, 1, [me])  # unbound -> false

    if isinstance(f, And):
        left_idx = get_index(f.left)
        right_idx = get_index(f.right)
        return (Player.ODD, 0, [_v_id(left_idx, s), _v_id(right_idx, s)])

    if isinstance(f, Or):
        left_idx = get_index(f.left)
        right_idx = get_index(f.right)
        return (Player.EVEN, 0, [_v_id(left_idx, s), _v_id(right_idx, s)])

    if isinstance(f, Diamond):
        succ_states = lts.successors(s, f.action)
        sub_idx = get_index(f.sub)
        targets = [_v_id(sub_idx, t) for t in succ_states]
        if not targets:
            return (Player.EVEN, 1, [me])  # no successors -> false (odd prio)
        return (Player.EVEN, 0, targets)

    if isinstance(f, Box):
        succ_states = lts.successors(s, f.action)
        sub_idx = get_index(f.sub)
        targets = [_v_id(sub_idx, t) for t in succ_states]
        if not targets:
            return (Player.ODD, 0, [me])  # vacuously true (even prio)
        return (Player.ODD, 0, targets)

    if isinstance(f, Mu):
        body_idx = get_index(f.body)
        return (Player.ODD, 0, [_v_id(body_idx, s)])

    if isinstance(f, Nu):
        body_idx = get_index(f.body)
        return (Player.EVEN, 0, [_v_id(body_idx, s)])

    return (Player.EVEN, 0, [me])


# Global vertex map reference -- set during game construction
_current_v_map = None

def _v_id(fi: int, s: int) -> int:
    """Get vertex id for (formula_index, state) pair."""
    return _current_v_map[(fi, s)]


def _find_binding(sub_list: List[Formula], var_name: str) -> Optional[int]:
    """Find the index of the fixpoint formula binding var_name."""
    for i, f in enumerate(sub_list):
        if isinstance(f, (Mu, Nu)) and f.var == var_name:
            return i
    return None


def formula_to_parity_game(
    lts: LTS,
    formula: Formula
) -> Tuple:
    """Reduce mu-calculus model checking to a parity game.

    Returns (ParityGame, v_map, sub_list, top_idx).
    """
    global _current_v_map

    v156_dir = os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games')
    if v156_dir not in sys.path:
        sys.path.insert(0, v156_dir)
    from parity_games import ParityGame, Player

    # Enumerate subformulas
    sub_list = []
    index_map = {}

    def get_index(f: Formula) -> int:
        key = repr(f)
        if key not in index_map:
            idx = len(sub_list)
            index_map[key] = idx
            sub_list.append(f)
        return index_map[key]

    _register_subformulas(formula, get_index)
    top_idx = get_index(formula)

    # Collect fixpoint info
    fp_info = {}
    _collect_fixpoints(formula, fp_info, 0)

    max_depth = max((d for _, d in fp_info.values()), default=-1)
    var_priority = {}
    for vname, (ftype, depth) in fp_info.items():
        if ftype == 'mu':
            var_priority[vname] = 2 * depth + 1  # odd
        else:
            var_priority[vname] = 2 * depth + 2  # even

    # Build vertex map
    v_map = {}
    vid = 0
    for fi in range(len(sub_list)):
        for s in lts.states:
            v_map[(fi, s)] = vid
            vid += 1

    _current_v_map = v_map

    # Build game
    game = ParityGame()

    for fi, f in enumerate(sub_list):
        for s in lts.states:
            vertex = v_map[(fi, s)]
            owner, prio, successors = _game_vertex_info(
                f, fi, s, lts, sub_list, get_index, fp_info, var_priority
            )
            game.add_vertex(vertex, owner, prio)
            for succ in successors:
                game.add_edge(vertex, succ)

    _current_v_map = None
    return game, v_map, sub_list, top_idx


# ---------------------------------------------------------------------------
# Model checking via parity game
# ---------------------------------------------------------------------------

def check_via_parity_game(lts: LTS, formula: Formula) -> Set[int]:
    """Model check using parity game reduction + V156 Zielonka solver.

    Returns set of states satisfying the formula.
    """
    v156_dir = os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games')
    if v156_dir not in sys.path:
        sys.path.insert(0, v156_dir)
    from parity_games import zielonka

    game, v_map, sub_list, top_idx = formula_to_parity_game(lts, formula)
    solution = zielonka(game)

    # States where formula holds = states where (top_idx, s) is in Even's winning region
    result = set()
    for s in lts.states:
        vid = v_map[(top_idx, s)]
        if vid in solution.win_even:
            result.add(s)
    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def model_check(
    lts: LTS,
    formula: Formula,
    method: str = "direct"
) -> Set[int]:
    """Model check a mu-calculus formula on an LTS.

    method: "direct" (fixpoint iteration) or "game" (parity game reduction)
    Returns: set of states satisfying the formula.
    """
    if method == "direct":
        return eval_formula(lts, formula)
    elif method == "game":
        return check_via_parity_game(lts, formula)
    else:
        raise ValueError(f"Unknown method: {method}")


def check_state(lts: LTS, formula: Formula, state: int, method: str = "direct") -> bool:
    """Check if a specific state satisfies the formula."""
    sat_states = model_check(lts, formula, method)
    return state in sat_states


def compare_methods(lts: LTS, formula: Formula) -> Dict:
    """Compare direct fixpoint and parity game methods."""
    direct_result = model_check(lts, formula, "direct")
    game_result = model_check(lts, formula, "game")
    return {
        "direct_states": direct_result,
        "game_states": game_result,
        "agree": direct_result == game_result,
        "formula": repr(formula),
        "lts_states": len(lts.states),
    }


# ---------------------------------------------------------------------------
# CTL encoding in mu-calculus
# ---------------------------------------------------------------------------

def ctl_EF(phi: Formula) -> Formula:
    """EF phi = mu X. (phi | <*>X)"""
    return Mu("__ef", Or(phi, Diamond(None, Var("__ef"))))


def ctl_AG(phi: Formula) -> Formula:
    """AG phi = nu X. (phi & [*]X)"""
    return Nu("__ag", And(phi, Box(None, Var("__ag"))))


def ctl_AF(phi: Formula) -> Formula:
    """AF phi = mu X. (phi | [*]X)
    Wait -- AF is: on all paths, eventually phi.
    AF phi = mu X. (phi | (NOT deadlock & [*]X))
    Simpler: AF phi = mu X. (phi | [*]X) works if no deadlocks.
    Actually: AF phi = mu X. (phi | ([*]X & <*>TT))
    The <*>TT ensures state is not a deadlock.
    """
    return Mu("__af", Or(phi, And(Box(None, Var("__af")), Diamond(None, TT()))))


def ctl_EG(phi: Formula) -> Formula:
    """EG phi = nu X. (phi & <*>X)"""
    return Nu("__eg", And(phi, Diamond(None, Var("__eg"))))


def ctl_EU(phi: Formula, psi: Formula) -> Formula:
    """E[phi U psi] = mu X. (psi | (phi & <*>X))"""
    return Mu("__eu", Or(psi, And(phi, Diamond(None, Var("__eu")))))


def ctl_AU(phi: Formula, psi: Formula) -> Formula:
    """A[phi U psi] = mu X. (psi | (phi & [*]X & <*>TT))"""
    return Mu("__au", Or(psi, And(And(phi, Box(None, Var("__au"))), Diamond(None, TT()))))


def ctl_EX(phi: Formula) -> Formula:
    """EX phi = <*>phi"""
    return Diamond(None, phi)


def ctl_AX(phi: Formula) -> Formula:
    """AX phi = [*]phi"""
    return Box(None, phi)


# ---------------------------------------------------------------------------
# Formula parser
# ---------------------------------------------------------------------------

def parse_mu(text: str) -> Formula:
    """Parse a mu-calculus formula from text.

    Syntax:
      tt, ff                        -- constants
      p, q, r, ...                  -- propositions (lowercase identifiers)
      X, Y, Z, ...                  -- variables (uppercase identifiers)
      ~phi, !phi                    -- negation
      phi & psi, phi /\\ psi        -- conjunction
      phi | psi, phi \\/ psi        -- disjunction
      <a>phi                        -- diamond (action a)
      <>phi                         -- diamond (any action)
      [a]phi                        -- box (action a)
      []phi                         -- box (any action)
      mu X. phi                     -- least fixpoint
      nu X. phi                     -- greatest fixpoint
      (phi)                         -- parentheses
    """
    tokens = _tokenize(text)
    pos = [0]
    result = _parse_formula(tokens, pos)
    return result


def _tokenize(text: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in '()&|~!.':
            tokens.append(c)
            i += 1
        elif c == '<':
            j = i + 1
            if j < len(text) and text[j] == '>':
                tokens.append('<>')
                i = j + 1
            else:
                # Find matching >
                end = text.index('>', j)
                tokens.append('<' + text[j:end] + '>')
                i = end + 1
        elif c == '[':
            j = i + 1
            if j < len(text) and text[j] == ']':
                tokens.append('[]')
                i = j + 1
            else:
                end = text.index(']', j)
                tokens.append('[' + text[j:end] + ']')
                i = end + 1
        elif c == '/' and i + 1 < len(text) and text[i + 1] == '\\':
            tokens.append('&')
            i += 2
        elif c == '\\' and i + 1 < len(text) and text[i + 1] == '/':
            tokens.append('|')
            i += 2
        elif c.isalpha() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            i += 1  # skip unknown
    return tokens


def _parse_formula(tokens, pos) -> Formula:
    return _parse_or(tokens, pos)


def _parse_or(tokens, pos) -> Formula:
    left = _parse_and(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]] == '|':
        pos[0] += 1
        right = _parse_and(tokens, pos)
        left = Or(left, right)
    return left


def _parse_and(tokens, pos) -> Formula:
    left = _parse_unary(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]] == '&':
        pos[0] += 1
        right = _parse_unary(tokens, pos)
        left = And(left, right)
    return left


def _parse_unary(tokens, pos) -> Formula:
    if pos[0] >= len(tokens):
        return FF()
    tok = tokens[pos[0]]

    if tok in ('~', '!'):
        pos[0] += 1
        sub = _parse_unary(tokens, pos)
        return Not(sub)

    if tok == 'mu' or tok == 'nu':
        pos[0] += 1
        var_name = tokens[pos[0]]
        pos[0] += 1
        if pos[0] < len(tokens) and tokens[pos[0]] == '.':
            pos[0] += 1
        body = _parse_formula(tokens, pos)
        if tok == 'mu':
            return Mu(var_name, body)
        else:
            return Nu(var_name, body)

    return _parse_atom(tokens, pos)


def _parse_atom(tokens, pos) -> Formula:
    if pos[0] >= len(tokens):
        return FF()
    tok = tokens[pos[0]]

    if tok == '(':
        pos[0] += 1
        result = _parse_formula(tokens, pos)
        if pos[0] < len(tokens) and tokens[pos[0]] == ')':
            pos[0] += 1
        return result

    if tok == 'tt':
        pos[0] += 1
        return TT()

    if tok == 'ff':
        pos[0] += 1
        return FF()

    # Diamond: <a>phi or <>phi
    if tok.startswith('<') and tok.endswith('>'):
        action = tok[1:-1] if len(tok) > 2 else None
        pos[0] += 1
        sub = _parse_unary(tokens, pos)
        return Diamond(action, sub)

    # Box: [a]phi or []phi
    if tok.startswith('[') and tok.endswith(']'):
        action = tok[1:-1] if len(tok) > 2 else None
        pos[0] += 1
        sub = _parse_unary(tokens, pos)
        return Box(action, sub)

    # Identifier
    if tok[0].isupper():
        pos[0] += 1
        return Var(tok)

    pos[0] += 1
    return Prop(tok)


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def formula_info(f: Formula) -> Dict:
    """Get information about a formula."""
    subs = subformulas(f)
    return {
        "repr": repr(f),
        "subformula_count": len(subs),
        "alternation_depth": alternation_depth(f),
        "fixpoint_nesting": fixpoint_nesting_depth(f),
        "free_vars": free_vars(f),
        "is_closed": is_closed(f),
    }


def check_result(lts: LTS, formula: Formula, method: str = "direct") -> Dict:
    """Full model checking result with metadata."""
    sat_states = model_check(lts, formula, method)
    return {
        "sat_states": sat_states,
        "sat_count": len(sat_states),
        "total_states": len(lts.states),
        "formula": repr(formula),
        "method": method,
        "formula_info": formula_info(formula),
    }


def batch_check(lts: LTS, formulas: List[Formula], method: str = "direct") -> List[Dict]:
    """Check multiple formulas on the same LTS."""
    return [check_result(lts, f, method) for f in formulas]


def mu_calculus_summary(lts: LTS, formula: Formula) -> str:
    """Human-readable summary of model checking result."""
    info = formula_info(formula)
    direct = model_check(lts, formula, "direct")
    lines = [
        f"Formula: {info['repr']}",
        f"  Alternation depth: {info['alternation_depth']}",
        f"  Fixpoint nesting: {info['fixpoint_nesting']}",
        f"  Subformulas: {info['subformula_count']}",
        f"  Closed: {info['is_closed']}",
        f"LTS: {len(lts.states)} states",
        f"Satisfying states (direct): {sorted(direct)}",
    ]
    try:
        game = model_check(lts, formula, "game")
        lines.append(f"Satisfying states (game):   {sorted(game)}")
        lines.append(f"Methods agree: {direct == game}")
    except Exception as e:
        lines.append(f"Game method error: {e}")
    return "\n".join(lines)
