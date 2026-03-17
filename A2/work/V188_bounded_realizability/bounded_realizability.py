"""
V188: Bounded Realizability -- LTL realizability with bounded-state controllers.

Multiple realizability checking methods:
1. Bounded realizability -- check if k-state controller exists via product game
2. Incremental search -- find minimum controller size
3. Safety realizability -- direct game solving for safety specs
4. Environment counterstrategies -- when unrealizable, extract winning env strategy
5. Quick checks -- syntactic/semantic pre-screening before full analysis

Composes:
- V023 (LTL -> GBA -> NBA pipeline)
- V186 (reactive synthesis, game arena, parity solving)
- V187 (GR(1) fixpoint concepts)
"""

import sys
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Dict, Set, List, Tuple, Optional, FrozenSet, Any, Callable
)
from collections import deque
from fractions import Fraction

# --- Imports from V023 (LTL model checking) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V023_ltl_model_checking'))
from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies, Iff,
    Next, Finally, Globally, Until, Release, WeakUntil,
    LTLTrue, LTLFalse, nnf, atoms, subformulas,
    ltl_to_gba, gba_to_nba, Label, GBA, NBA,
    parse_ltl
)

# --- Imports from V186 (reactive synthesis) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V186_reactive_synthesis'))
from reactive_synthesis import (
    MealyMachine as V186Mealy,
    SynthesisResult as V186Result,
    SynthesisVerdict,
    synthesize as v186_synthesize,
)

# --- Imports from V187 (GR(1) synthesis) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V187_gr1_synthesis'))
from gr1_synthesis import (
    GR1Game, GR1Result, GR1Verdict,
    gr1_solve, make_game as make_gr1_game,
)


# ============================================================
# Data structures
# ============================================================

class RealVerdict(Enum):
    """Realizability verdict."""
    REALIZABLE = "REALIZABLE"
    UNREALIZABLE = "UNREALIZABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class BoundedController:
    """A controller with bounded state count."""
    n_states: int
    initial: int  # initial controller state
    # transitions: (ctrl_state, nba_state, env_input) -> (next_ctrl, output)
    transitions: Dict[Tuple[int, int, FrozenSet[str]],
                       Tuple[int, FrozenSet[str]]]
    env_vars: Set[str]
    sys_vars: Set[str]

    def step(self, ctrl_state: int, nba_state: int,
             env_input: FrozenSet[str]) -> Optional[Tuple[int, FrozenSet[str]]]:
        """Execute one step. Returns (next_ctrl_state, sys_output) or None."""
        key = (ctrl_state, nba_state, env_input)
        return self.transitions.get(key)

    def to_mealy(self) -> 'MealyMachine':
        """Convert to flat Mealy machine (loses NBA state tracking)."""
        states = set(range(self.n_states))
        transitions = {}
        for (cs, ns, env_in), (ncs, sys_out) in self.transitions.items():
            # Flatten: use (ctrl_state, nba_state) as Mealy state
            flat_src = cs * 1000 + ns
            flat_dst = ncs * 1000 + 0  # nba state unknown at this point
            transitions[(flat_src, env_in)] = (flat_dst, sys_out)
        return MealyMachine(
            states=states,
            initial=self.initial,
            transitions=transitions,
            env_vars=self.env_vars,
            sys_vars=self.sys_vars,
        )


@dataclass
class MealyMachine:
    """Simple Mealy machine."""
    states: Set[int]
    initial: int
    transitions: Dict[Tuple[int, FrozenSet[str]], Tuple[int, FrozenSet[str]]]
    env_vars: Set[str]
    sys_vars: Set[str]

    def step(self, state: int,
             env_input: FrozenSet[str]) -> Optional[Tuple[int, FrozenSet[str]]]:
        """One step."""
        return self.transitions.get((state, env_input))

    def simulate(self, inputs: List[FrozenSet[str]],
                 max_steps: int = 100) -> List[Tuple[int, FrozenSet[str], FrozenSet[str]]]:
        """Simulate. Returns [(state, input, output), ...]."""
        trace = []
        state = self.initial
        for i, inp in enumerate(inputs):
            if i >= max_steps:
                break
            result = self.step(state, inp)
            if result is None:
                break
            next_state, output = result
            trace.append((state, inp, output))
            state = next_state
        return trace


@dataclass
class EnvCounterstrategy:
    """Environment's winning strategy (proves unrealizability)."""
    states: Set[int]
    initial: int
    # transitions: (env_state, sys_output) -> (next_env_state, env_input)
    transitions: Dict[Tuple[int, FrozenSet[str]], Tuple[int, FrozenSet[str]]]
    env_vars: Set[str]
    sys_vars: Set[str]
    description: str = ""

    def step(self, state: int,
             sys_output: FrozenSet[str]) -> Optional[Tuple[int, FrozenSet[str]]]:
        """One step."""
        return self.transitions.get((state, sys_output))


@dataclass
class RealResult:
    """Full realizability checking result."""
    verdict: RealVerdict
    controller: Optional[MealyMachine] = None
    bounded_controller: Optional[BoundedController] = None
    counterstrategy: Optional[EnvCounterstrategy] = None
    bound: Optional[int] = None  # k if bounded check
    min_states: Optional[int] = None  # minimum controller size found
    automaton_states: int = 0
    game_states: int = 0
    method: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Helper: all valuations
# ============================================================

def _all_valuations(variables: Set[str]) -> List[FrozenSet[str]]:
    """Generate all 2^|variables| boolean valuations."""
    var_list = sorted(variables)
    n = len(var_list)
    result = []
    for i in range(1 << n):
        val = frozenset(var_list[j] for j in range(n) if (i >> j) & 1)
        result.append(val)
    return result


def _label_matches(label: Label, valuation: FrozenSet[str]) -> bool:
    """Check if a valuation satisfies a label."""
    for p in label.pos:
        if p not in valuation:
            return False
    for n in label.neg:
        if n in valuation:
            return False
    return True


# ============================================================
# 1. Bounded Realizability (product game approach)
# ============================================================

def _build_bounded_game(nba: NBA, env_vars: Set[str], sys_vars: Set[str],
                        k: int) -> Tuple[Dict, Set, Set, Set, Dict]:
    """
    Build a bounded product game for k-state controller search.

    Game states: (nba_state, ctrl_state, turn)
    - turn='env': environment picks input
    - turn='sys': system picks output + next ctrl state
    - turn='nba': nba transitions (deterministic given full valuation)

    Returns: (transitions, initial, env_vertices, sys_vertices, accepting)
    """
    env_vals = _all_valuations(env_vars)
    sys_vals = _all_valuations(sys_vars)

    transitions = {}  # vertex -> [successors]
    env_vertices = set()  # environment choice vertices
    sys_vertices = set()  # system choice vertices
    accepting = set()  # accepting vertices (for Buchi)
    all_vertices = set()

    # Initial states: (q0, 0, 'env') for each initial NBA state q0
    initial = set()
    for q0 in nba.initial:
        v = (q0, 0, 'env')
        initial.add(v)

    # Build game arena via BFS
    queue = deque(initial)
    visited = set(initial)

    while queue:
        v = queue.popleft()
        all_vertices.add(v)

        if v[2] == 'env':
            # Environment vertex: picks input
            q, c, _ = v
            env_vertices.add(v)
            succs = []
            for env_in in env_vals:
                mid = (q, c, 'sys', env_in)
                succs.append(mid)
                if mid not in visited:
                    visited.add(mid)
                    queue.append(mid)
            transitions[v] = succs

            # Mark accepting
            if q in nba.accepting:
                accepting.add(v)

        elif v[2] == 'sys':
            # System vertex: picks output + next controller state
            q, c, _, env_in = v
            sys_vertices.add(v)
            all_vertices.add(v)
            succs = []
            for sys_out in sys_vals:
                for next_c in range(k):
                    # Compute NBA transitions
                    full_val = env_in | sys_out
                    # Find all NBA successors
                    if q in nba.transitions:
                        for label, q_next in nba.transitions[q]:
                            if _label_matches(label, full_val):
                                nxt = (q_next, next_c, 'env')
                                succs.append(nxt)
                                if nxt not in visited:
                                    visited.add(nxt)
                                    queue.append(nxt)
            transitions[v] = succs

            if q in nba.accepting:
                accepting.add(v)

    return transitions, initial, env_vertices, sys_vertices, accepting


def _solve_buchi_game(transitions: Dict, initial: Set, env_vertices: Set,
                      sys_vertices: Set, accepting: Set) -> Tuple[Set, Dict]:
    """
    Solve a Buchi game. System wins if it can visit accepting states
    infinitely often.

    Uses the standard attractor-based algorithm:
    Compute greatest fixpoint of: accepting AND sys can force revisit.

    Returns: (winning_region, strategy)
    """
    all_vertices = set(transitions.keys())
    for v, succs in transitions.items():
        for s in succs:
            all_vertices.add(s)

    # Dead-end vertices (no successors) cannot produce infinite plays.
    # System loses from dead-end sys vertices; env loses from dead-end env vertices.
    # Iteratively remove dead ends and their attractors.
    dead_ends = set()
    for v in all_vertices:
        succs = transitions.get(v, [])
        if not succs:
            dead_ends.add(v)

    # Buchi game solving: iterative removal of losing vertices
    # System wins from states where it can force infinitely many visits
    # to accepting states.
    #
    # Algorithm: McNaughton-Zielonka style for Buchi
    # Simplified: repeatedly compute env attractor of non-accepting trap

    def cpre_sys(target: Set) -> Set:
        """States where system can force next state into target."""
        result = set()
        for v in all_vertices:
            succs = transitions.get(v, [])
            if not succs:
                continue
            if v in sys_vertices:
                # System chooses: exists successor in target
                if any(s in target for s in succs):
                    result.add(v)
            elif v in env_vertices:
                # Environment chooses: all successors in target
                if all(s in target for s in succs):
                    result.add(v)
            else:
                # Mid vertex (sys type)
                if any(s in target for s in succs):
                    result.add(v)
        return result

    def env_attractor(target: Set) -> Set:
        """States from which environment can force reaching target."""
        attr = set(target)
        changed = True
        while changed:
            changed = False
            for v in all_vertices:
                if v in attr:
                    continue
                succs = transitions.get(v, [])
                if not succs:
                    continue
                if v in env_vertices:
                    # Env chooses: exists successor in attr
                    if any(s in attr for s in succs):
                        attr.add(v)
                        changed = True
                elif v in sys_vertices:
                    # Sys chooses: all successors in attr
                    if all(s in attr for s in succs):
                        attr.add(v)
                        changed = True
                else:
                    # Mid vertex (sys type)
                    if all(s in attr for s in succs):
                        attr.add(v)
                        changed = True
        return attr

    # Pre-processing: remove dead-end sys/mid vertices (no successors = can't
    # produce infinite play = system loses). Then remove env attractor of those.
    win_sys = set(all_vertices)
    sys_dead = set()
    for v in dead_ends:
        if v in sys_vertices or v not in env_vertices:
            sys_dead.add(v)
    if sys_dead:
        sys_dead_attr = env_attractor(sys_dead)
        win_sys -= sys_dead_attr

    changed = True
    while changed:
        changed = False
        # Non-accepting states in current winning region
        non_acc = win_sys - accepting
        # States where env can trap system in non-accepting states forever
        # = states from which env can force staying in non_acc
        # Actually: env attractor of states with no escape to accepting

        # Compute trap: states from which system CANNOT reach accepting
        # within current win_sys
        reachable_to_acc = set()
        frontier = accepting & win_sys
        q = deque(frontier)
        visited_r = set(frontier)
        while q:
            v = q.popleft()
            reachable_to_acc.add(v)
            # Find predecessors
            for u in win_sys:
                if u in visited_r:
                    continue
                succs = transitions.get(u, [])
                if v in succs:
                    # u -> v exists, so u can reach accepting (potentially)
                    if u in sys_vertices or (u not in env_vertices):
                        # sys can choose this edge
                        visited_r.add(u)
                        q.append(u)
                    elif u in env_vertices:
                        # env vertex: only if ALL successors in win_sys
                        # lead to reachable_to_acc... too complex for backward
                        pass

        # Simpler approach: standard Buchi game algorithm
        # Remove env attractor of (win_sys \ accepting) trap components
        # Actually, use the classic algorithm:
        # 1. Find states that can reach accepting (within win_sys)
        # 2. Remove those that can't
        # 3. Iterate

        can_reach_acc = set()
        frontier = deque(accepting & win_sys)
        can_reach_acc = set(accepting & win_sys)
        while frontier:
            v = frontier.popleft()
            for u in win_sys:
                if u in can_reach_acc:
                    continue
                succs = transitions.get(u, [])
                if not succs:
                    continue
                in_win = [s for s in succs if s in win_sys]
                if not in_win:
                    continue
                if u in sys_vertices or (u not in env_vertices):
                    if v in succs:
                        can_reach_acc.add(u)
                        frontier.append(u)
                elif u in env_vertices:
                    if all(s in can_reach_acc for s in in_win):
                        can_reach_acc.add(u)
                        frontier.append(u)

        # States that can't reach accepting -> env can trap there
        trap = win_sys - can_reach_acc
        if trap:
            # Remove env attractor of trap
            attr = env_attractor(trap)
            new_win = win_sys - attr
            if new_win != win_sys:
                win_sys = new_win
                changed = True

    # Extract strategy: for sys vertices, pick successor in win_sys
    # that leads toward accepting
    strategy = {}
    for v in win_sys:
        succs = transitions.get(v, [])
        if v in sys_vertices or (v not in env_vertices):
            # System picks best successor
            for s in succs:
                if s in win_sys:
                    strategy[v] = s
                    break

    return win_sys, strategy


def check_bounded(spec: LTL, env_vars: Set[str], sys_vars: Set[str],
                  bound: int) -> RealResult:
    """
    Check if spec is realizable by a controller with at most `bound` states.

    Builds a product game NBA x {0..k-1} and solves the Buchi game.
    """
    # Convert spec to NBA
    neg_spec = Not(spec)
    gba = ltl_to_gba(neg_spec)
    nba_neg = gba_to_nba(gba)

    # Also get the direct NBA for the spec
    gba_spec = ltl_to_gba(spec)
    nba_spec = gba_to_nba(gba_spec)

    # Build bounded product game
    transitions, initial, env_verts, sys_verts, accepting = \
        _build_bounded_game(nba_spec, env_vars, sys_vars, bound)

    game_states = len(set(transitions.keys()))

    # Solve Buchi game
    winning, strategy = _solve_buchi_game(
        transitions, initial, env_verts, sys_verts, accepting
    )

    # Check if all initial states are winning for system
    all_init_winning = all(v in winning for v in initial)

    if all_init_winning and initial:
        # Extract bounded controller from strategy
        ctrl = _extract_bounded_controller(
            nba_spec, strategy, env_vars, sys_vars, bound, initial
        )
        return RealResult(
            verdict=RealVerdict.REALIZABLE,
            bounded_controller=ctrl,
            bound=bound,
            automaton_states=len(nba_spec.states),
            game_states=game_states,
            method="bounded_realizability",
            details={"winning_region_size": len(winning)},
        )
    else:
        return RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            bound=bound,
            automaton_states=len(nba_spec.states),
            game_states=game_states,
            method="bounded_realizability",
            details={"winning_region_size": len(winning)},
        )


def _extract_bounded_controller(nba: NBA, strategy: Dict,
                                env_vars: Set[str], sys_vars: Set[str],
                                k: int, initial: Set) -> BoundedController:
    """Extract a bounded controller from the game strategy."""
    transitions = {}
    env_vals = _all_valuations(env_vars)
    sys_vals = _all_valuations(sys_vars)

    # Walk strategy to find controller transitions
    visited = set()
    queue = deque(initial)

    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)

        if len(v) == 3 and v[2] == 'env':
            q, c, _ = v
            # For each env input, find what system does
            for env_in in env_vals:
                mid = (q, c, 'sys', env_in)
                if mid in strategy:
                    nxt = strategy[mid]
                    if len(nxt) == 3 and nxt[2] == 'env':
                        q_next, c_next, _ = nxt
                        # Determine sys output: what valuation led to this transition
                        for sys_out in sys_vals:
                            full_val = env_in | sys_out
                            if q in nba.transitions:
                                for label, q_dst in nba.transitions[q]:
                                    if q_dst == q_next and _label_matches(label, full_val):
                                        transitions[(c, q, env_in)] = (c_next, sys_out)
                                        break
                            if (c, q, env_in) in transitions:
                                break

                    if nxt not in visited:
                        queue.append(nxt)
        elif len(v) == 4 and v[2] == 'sys':
            if v in strategy:
                nxt = strategy[v]
                if nxt not in visited:
                    queue.append(nxt)

    init_ctrl = 0
    for v in initial:
        if len(v) == 3:
            init_ctrl = v[1]
            break

    return BoundedController(
        n_states=k,
        initial=init_ctrl,
        transitions=transitions,
        env_vars=env_vars,
        sys_vars=sys_vars,
    )


# ============================================================
# 2. Incremental Realizability Search
# ============================================================

def find_minimum_controller(spec: LTL, env_vars: Set[str],
                            sys_vars: Set[str],
                            max_bound: int = 8) -> RealResult:
    """
    Find the minimum number of controller states needed.
    Incrementally checks k=1,2,...,max_bound.
    """
    for k in range(1, max_bound + 1):
        result = check_bounded(spec, env_vars, sys_vars, k)
        if result.verdict == RealVerdict.REALIZABLE:
            result.min_states = k
            result.method = "incremental_search"
            return result

    return RealResult(
        verdict=RealVerdict.UNKNOWN,
        bound=max_bound,
        method="incremental_search",
        details={"searched_up_to": max_bound},
    )


# ============================================================
# 3. Safety Realizability
# ============================================================

def check_safety(bad_condition: LTL, env_vars: Set[str],
                 sys_vars: Set[str]) -> RealResult:
    """
    Check realizability of safety spec G(!bad_condition).
    Uses direct safety game (no Buchi needed).
    """
    spec = Globally(Not(bad_condition))
    env_vals = _all_valuations(env_vars)
    sys_vals = _all_valuations(sys_vars)
    all_props = env_vars | sys_vars

    # Safety game: system wins if it can ALWAYS avoid bad states.
    # State = current valuation. System picks sys_vars, env picks env_vars.
    # Bad state = valuation where bad_condition holds.

    def eval_ltl_prop(formula: LTL, valuation: FrozenSet[str]) -> bool:
        """Evaluate a propositional (non-temporal) LTL formula."""
        if formula.op == LTLOp.ATOM:
            return formula.name in valuation
        elif formula.op == LTLOp.TRUE:
            return True
        elif formula.op == LTLOp.FALSE:
            return False
        elif formula.op == LTLOp.NOT:
            return not eval_ltl_prop(formula.left, valuation)
        elif formula.op == LTLOp.AND:
            return (eval_ltl_prop(formula.left, valuation) and
                    eval_ltl_prop(formula.right, valuation))
        elif formula.op == LTLOp.OR:
            return (eval_ltl_prop(formula.left, valuation) or
                    eval_ltl_prop(formula.right, valuation))
        elif formula.op == LTLOp.IMPLIES:
            return (not eval_ltl_prop(formula.left, valuation) or
                    eval_ltl_prop(formula.right, valuation))
        elif formula.op == LTLOp.IFF:
            l = eval_ltl_prop(formula.left, valuation)
            r = eval_ltl_prop(formula.right, valuation)
            return l == r
        else:
            # Temporal operator in propositional eval -> default True (safe)
            return True

    # All possible states (valuations of all props)
    all_states = _all_valuations(all_props)

    # Identify bad states
    bad_states = set()
    for val in all_states:
        if eval_ltl_prop(bad_condition, val):
            bad_states.add(val)

    safe_states = set(all_states) - bad_states

    # Safety game: iterative removal
    # System wins from states where it can keep the game in safe states forever
    win = set(safe_states)
    changed = True
    while changed:
        changed = False
        new_win = set()
        for val in win:
            # For this state, check if system has a response to every env move
            env_part = frozenset(v for v in val if v in env_vars)
            sys_part = frozenset(v for v in val if v in sys_vars)

            # System must be able to pick sys_vars such that for ALL env_vars,
            # the resulting state is in win.
            # But this is a one-shot game (memoryless safety)
            # Actually for propositional safety, we check:
            # for every env choice, does system have a winning response?
            can_win = False
            for sv in sys_vals:
                # Check: for ALL env choices, is sv + ev in win?
                all_ok = True
                for ev in env_vals:
                    next_val = ev | sv
                    if next_val not in win:
                        all_ok = False
                        break
                if all_ok:
                    can_win = True
                    break
            if can_win:
                new_win.add(val)

        if new_win != win:
            win = new_win
            changed = True

    # Check if system can pick initial sys_vars to be in win for all env
    realizable = False
    winning_sys = None
    for sv in sys_vals:
        all_ok = True
        for ev in env_vals:
            if (ev | sv) not in win:
                all_ok = False
                break
        if all_ok:
            realizable = True
            winning_sys = sv
            break

    if realizable:
        # Build simple memoryless controller
        ctrl_trans = {}
        for ev in env_vals:
            # Find best sys response
            for sv in sys_vals:
                if (ev | sv) in win:
                    ctrl_trans[(0, ev)] = (0, sv)
                    break
        ctrl = MealyMachine(
            states={0},
            initial=0,
            transitions=ctrl_trans,
            env_vars=env_vars,
            sys_vars=sys_vars,
        )
        return RealResult(
            verdict=RealVerdict.REALIZABLE,
            controller=ctrl,
            method="safety_realizability",
            details={"safe_states": len(safe_states),
                     "winning_states": len(win)},
        )
    else:
        return RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            method="safety_realizability",
            details={"safe_states": len(safe_states),
                     "winning_states": len(win)},
        )


# ============================================================
# 4. Environment Counterstrategy Extraction
# ============================================================

def extract_counterstrategy(spec: LTL, env_vars: Set[str],
                            sys_vars: Set[str]) -> RealResult:
    """
    When spec is unrealizable, extract an environment counterstrategy
    that defeats ANY system controller.
    """
    # First check realizability via V186
    v186_result = v186_synthesize(spec, env_vars, sys_vars)

    if v186_result.verdict == SynthesisVerdict.REALIZABLE:
        return RealResult(
            verdict=RealVerdict.REALIZABLE,
            method="counterstrategy_extraction",
            details={"note": "Spec is realizable, no counterstrategy exists"},
        )

    # Unrealizable: synthesize for negated spec (environment's game)
    # Environment wins the game for NOT(spec)
    neg_spec = Not(spec)
    env_result = v186_synthesize(neg_spec, sys_vars, env_vars)

    if env_result.verdict == SynthesisVerdict.REALIZABLE and env_result.controller:
        # Convert V186 controller to our counterstrategy format
        v186_ctrl = env_result.controller
        cs_trans = {}
        for (state, inp), (nxt, out) in v186_ctrl.transitions.items():
            cs_trans[(state, inp)] = (nxt, out)

        cs = EnvCounterstrategy(
            states=v186_ctrl.states,
            initial=v186_ctrl.initial,
            transitions=cs_trans,
            env_vars=env_vars,
            sys_vars=sys_vars,
            description="Environment strategy defeating all system controllers",
        )
        return RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            counterstrategy=cs,
            method="counterstrategy_extraction",
            details={"env_controller_states": len(v186_ctrl.states)},
        )
    else:
        return RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            method="counterstrategy_extraction",
            details={"note": "Could not extract explicit counterstrategy"},
        )


# ============================================================
# 5. Quick Realizability Checks (syntactic/semantic pre-screening)
# ============================================================

def _is_propositional(formula: LTL) -> bool:
    """Check if formula has no temporal operators."""
    if formula.op in (LTLOp.ATOM, LTLOp.TRUE, LTLOp.FALSE):
        return True
    if formula.op in (LTLOp.X, LTLOp.F, LTLOp.G, LTLOp.U, LTLOp.R, LTLOp.W):
        return False
    if formula.left and not _is_propositional(formula.left):
        return False
    if formula.right and not _is_propositional(formula.right):
        return False
    return True


def _is_safety_spec(formula: LTL) -> bool:
    """Check if formula is of the form G(propositional)."""
    if formula.op == LTLOp.G and formula.left:
        return _is_propositional(formula.left)
    return False


def quick_check(spec: LTL, env_vars: Set[str],
                sys_vars: Set[str]) -> RealResult:
    """
    Fast realizability pre-screening using syntactic checks.

    Handles:
    - TRUE/FALSE specs
    - Pure propositional specs
    - Safety specs G(prop) via direct game
    - Falls back to bounded check
    """
    # Trivial cases
    if spec.op == LTLOp.TRUE:
        return RealResult(
            verdict=RealVerdict.REALIZABLE,
            method="quick_check",
            details={"reason": "spec is TRUE"},
        )

    if spec.op == LTLOp.FALSE:
        return RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            method="quick_check",
            details={"reason": "spec is FALSE"},
        )

    # G(false) is unrealizable
    if spec.op == LTLOp.G and spec.left and spec.left.op == LTLOp.FALSE:
        return RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            method="quick_check",
            details={"reason": "G(false) is unsatisfiable"},
        )

    # G(true) is trivially realizable
    if spec.op == LTLOp.G and spec.left and spec.left.op == LTLOp.TRUE:
        return RealResult(
            verdict=RealVerdict.REALIZABLE,
            method="quick_check",
            details={"reason": "G(true) is trivially realizable"},
        )

    # Safety spec: use direct game
    if _is_safety_spec(spec):
        # G(prop) <=> G(!bad) where bad = !prop
        inner = spec.left
        bad = Not(inner)
        return check_safety(bad, env_vars, sys_vars)

    # Pure propositional (no temporal): check if system can satisfy once
    if _is_propositional(spec):
        env_vals = _all_valuations(env_vars)
        sys_vals = _all_valuations(sys_vars)

        def eval_prop(f: LTL, val: FrozenSet[str]) -> bool:
            if f.op == LTLOp.ATOM:
                return f.name in val
            if f.op == LTLOp.TRUE:
                return True
            if f.op == LTLOp.FALSE:
                return False
            if f.op == LTLOp.NOT:
                return not eval_prop(f.left, val)
            if f.op == LTLOp.AND:
                return eval_prop(f.left, val) and eval_prop(f.right, val)
            if f.op == LTLOp.OR:
                return eval_prop(f.left, val) or eval_prop(f.right, val)
            if f.op == LTLOp.IMPLIES:
                return not eval_prop(f.left, val) or eval_prop(f.right, val)
            if f.op == LTLOp.IFF:
                return eval_prop(f.left, val) == eval_prop(f.right, val)
            return True

        # System must find output that satisfies spec for ALL env inputs
        for sv in sys_vals:
            all_ok = True
            for ev in env_vals:
                if not eval_prop(spec, ev | sv):
                    all_ok = False
                    break
            if all_ok:
                ctrl_trans = {}
                for ev in env_vals:
                    ctrl_trans[(0, ev)] = (0, sv)
                ctrl = MealyMachine(
                    states={0}, initial=0,
                    transitions=ctrl_trans,
                    env_vars=env_vars, sys_vars=sys_vars,
                )
                return RealResult(
                    verdict=RealVerdict.REALIZABLE,
                    controller=ctrl,
                    method="quick_check",
                    details={"reason": "propositional spec satisfiable"},
                )
        return RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            method="quick_check",
            details={"reason": "propositional spec unsatisfiable for all sys choices"},
        )

    # Default: use bounded check with small bound
    return check_bounded(spec, env_vars, sys_vars, bound=4)


# ============================================================
# 6. Comparison and Analysis APIs
# ============================================================

def compare_methods(spec: LTL, env_vars: Set[str],
                    sys_vars: Set[str],
                    max_bound: int = 4) -> Dict[str, Any]:
    """
    Compare different realizability checking methods on same spec.
    """
    results = {}

    # Quick check
    qr = quick_check(spec, env_vars, sys_vars)
    results["quick_check"] = {
        "verdict": qr.verdict.value,
        "method": qr.method,
        "details": qr.details,
    }

    # Bounded checks
    for k in range(1, max_bound + 1):
        br = check_bounded(spec, env_vars, sys_vars, k)
        results[f"bounded_k{k}"] = {
            "verdict": br.verdict.value,
            "game_states": br.game_states,
            "automaton_states": br.automaton_states,
        }
        if br.verdict == RealVerdict.REALIZABLE:
            results["min_bound"] = k
            break

    return results


def realizability_summary(result: RealResult) -> str:
    """Human-readable summary of realizability result."""
    lines = [f"Verdict: {result.verdict.value}"]
    lines.append(f"Method: {result.method}")
    if result.bound is not None:
        lines.append(f"Bound: {result.bound}")
    if result.min_states is not None:
        lines.append(f"Minimum controller states: {result.min_states}")
    if result.automaton_states:
        lines.append(f"Automaton states: {result.automaton_states}")
    if result.game_states:
        lines.append(f"Game states: {result.game_states}")
    if result.controller:
        lines.append(f"Controller states: {len(result.controller.states)}")
    if result.bounded_controller:
        lines.append(f"Bounded controller: {result.bounded_controller.n_states} states")
    if result.counterstrategy:
        lines.append(f"Counterstrategy: {len(result.counterstrategy.states)} states")
    for k, v in result.details.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def check_realizable(spec: LTL, env_vars: Set[str],
                     sys_vars: Set[str]) -> RealResult:
    """
    Main API: check LTL realizability using best available method.

    Tries quick checks first, then bounded search.
    """
    # Try quick check
    qr = quick_check(spec, env_vars, sys_vars)
    if qr.verdict != RealVerdict.UNKNOWN:
        return qr

    # Quick check returned from bounded -- already have result
    return qr


def check_and_explain(spec: LTL, env_vars: Set[str],
                      sys_vars: Set[str]) -> RealResult:
    """
    Check realizability and provide explanation.
    If unrealizable, attempts to extract counterstrategy.
    If realizable, provides controller.
    """
    result = check_realizable(spec, env_vars, sys_vars)

    if result.verdict == RealVerdict.UNREALIZABLE:
        # Try to get counterstrategy
        cs_result = extract_counterstrategy(spec, env_vars, sys_vars)
        if cs_result.counterstrategy:
            result.counterstrategy = cs_result.counterstrategy
            result.details["counterstrategy_extracted"] = True

    return result
