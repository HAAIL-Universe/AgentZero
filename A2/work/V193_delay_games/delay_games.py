"""
V193: Delay Games -- Synthesis with Bounded Lookahead

Delay games extend reactive synthesis by allowing the system player to see
a bounded number of future environment moves before committing to its output.
This strictly increases realizability: specs unrealizable without delay may
become realizable with sufficient lookahead.

Key concepts:
- Constant delay k: system sees next k environment moves before responding
- Delay game arena: expanded state space with environment move buffers
- Minimum delay: smallest k making a spec realizable
- Reduction: delay-k game -> standard game over buffered arena

Composes:
- V186 (reactive synthesis): MealyMachine, SynthesisResult, game solving
- V187 (GR(1) synthesis): GR1Game, gr1_solve, efficient fixpoint
- V023 (LTL model checker): LTL AST, NBA construction

Theory: Klein & Zimmermann (2015), Hosch & Landweber (1972)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import (
    Set, Dict, List, Tuple, Optional, FrozenSet, Callable, Any
)
from enum import Enum
from collections import deque
from itertools import product as cartesian_product

# -- Imports from existing tools --

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies, Iff,
    Next, Finally, Globally, Until, Release, WeakUntil,
    LTLTrue, LTLFalse, atoms, nnf, parse_ltl,
    ltl_to_gba, gba_to_nba, Label, NBA
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V186_reactive_synthesis'))
from reactive_synthesis import (
    MealyMachine, SynthesisResult, SynthesisVerdict,
    synthesize as v186_synthesize,
    verify_controller,
    controller_statistics,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V187_gr1_synthesis'))
from gr1_synthesis import (
    GR1Game, GR1Result, GR1Verdict, GR1Strategy,
    gr1_solve, build_bool_game, BoolGR1Spec,
    make_game, game_statistics, gr1_summary,
    strategy_to_mealy,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
from parity_games import ParityGame, Player, Solution, zielonka


# ============================================================
# Data Structures
# ============================================================

@dataclass
class DelayGameResult:
    """Result of delay game synthesis."""
    realizable: bool
    delay: int  # the delay value used
    controller: Optional[MealyMachine] = None
    buffered_states: int = 0
    original_states: int = 0
    method: str = "delay_game"
    iterations: int = 0


@dataclass
class MinDelayResult:
    """Result of minimum delay search."""
    realizable: bool
    min_delay: int  # minimum delay for realizability (-1 if unrealizable at max)
    results: Dict[int, DelayGameResult] = field(default_factory=dict)
    searched_delays: List[int] = field(default_factory=list)


@dataclass
class DelayStrategy:
    """Strategy for a delay game -- maps (state, buffer) to output."""
    delay: int
    transitions: Dict[Tuple[Any, Tuple], Tuple[Any, FrozenSet[str]]] = field(default_factory=dict)
    initial_state: Any = None

    def step(self, state, buffer):
        """Given state and buffer of future env moves, return (next_state, output)."""
        key = (state, tuple(buffer))
        if key in self.transitions:
            return self.transitions[key]
        return None


# ============================================================
# Valuation helpers
# ============================================================

def _all_valuations(variables: Set[str]) -> List[FrozenSet[str]]:
    """Generate all 2^|vars| truth assignments as frozensets."""
    var_list = sorted(variables)
    result = []
    for i in range(1 << len(var_list)):
        val = frozenset(v for j, v in enumerate(var_list) if i & (1 << j))
        result.append(val)
    return result


def _label_matches(label, valuation: FrozenSet[str]) -> bool:
    """Check if a valuation satisfies an NBA label."""
    for p in label.pos:
        if p not in valuation:
            return False
    for n in label.neg:
        if n in valuation:
            return False
    return True


# ============================================================
# Core: Delay Game Arena Construction
# ============================================================

def build_delay_arena(
    nba: NBA,
    env_vars: Set[str],
    sys_vars: Set[str],
    delay: int,
) -> Tuple[ParityGame, Dict, Dict, Any]:
    """
    Build a parity game arena for a delay-k game.

    In a delay-k game, the system sees the next k environment moves
    before deciding its current output. The arena encodes this by
    maintaining a buffer of pending environment moves.

    State space: (nba_state, buffer) where buffer is a tuple of k env valuations.

    Turn structure:
    - Environment vertex (q, buf, 'env'): env adds one move to buffer end
    - System vertex (q, buf, 'sys'): sys sees full buffer, picks output,
      consumes first buffer element, NBA transitions on (buf[0], sys_out)

    Returns: (game, vertex_to_info, info_to_vertex, initial_vertices)
    """
    all_env = _all_valuations(env_vars)
    all_sys = _all_valuations(sys_vars)

    game = ParityGame(set(), {}, {}, {})
    v2info = {}
    info2v = {}
    vid = [0]

    def get_vertex(info, owner, priority):
        if info in info2v:
            return info2v[info]
        v = vid[0]
        vid[0] += 1
        info2v[info] = v
        v2info[v] = info
        game.vertices.add(v)
        game.edges[v] = set()
        game.owner[v] = owner
        game.priority[v] = priority
        return v

    initial_vertices = set()

    # Phase 1: build initial buffer
    # For delay k, we need k env moves buffered before system starts deciding.
    # Initial states: (q0, empty_buffer, 'env') -- env fills buffer first.
    # After k env moves, buffer is full and system can start responding.

    # For k=0, no buffer -- standard game (system sees current env move only)
    # For k=1, system sees 1 future env move (buffer of size 1)
    # etc.

    # We use a phased approach:
    # - Fill phase: env fills buffer from empty to size k
    # - Play phase: alternating env-append / sys-consume-and-respond

    # States in play phase: (q, buffer_of_size_k, 'env') or (q, buffer_of_size_k+1, 'sys')
    # When env moves in play phase: buffer grows to k+1
    # When sys responds: consumes buf[0], transitions NBA, buffer shrinks to k

    # BFS to build reachable arena
    queue = deque()

    for q0 in nba.initial:
        if delay == 0:
            # No lookahead: standard turn-based game
            # env vertex: env picks input
            is_acc = q0 in nba.accepting
            prio = 2 if is_acc else 1
            v = get_vertex((q0, (), 'env'), Player.ODD, prio)
            initial_vertices.add(v)
            queue.append((q0, (), 'env'))
        else:
            # Fill phase starts: env fills buffer
            v = get_vertex((q0, (), 'fill'), Player.ODD, 0)
            initial_vertices.add(v)
            queue.append((q0, (), 'fill'))

    visited = set()

    while queue:
        state_info = queue.popleft()
        if state_info in visited:
            continue
        visited.add(state_info)

        q, buf, phase = state_info
        v = info2v[state_info]

        if phase == 'fill':
            # Environment fills buffer -- env chooses next input
            for ev in all_env:
                new_buf = buf + (ev,)
                if len(new_buf) < delay:
                    # Still filling
                    next_info = (q, new_buf, 'fill')
                    next_v = get_vertex(next_info, Player.ODD, 0)
                    game.edges[v].add(next_v)
                    queue.append(next_info)
                else:
                    # Buffer full -- transition to play phase (env turn)
                    is_acc = q in nba.accepting
                    prio = 2 if is_acc else 1
                    next_info = (q, new_buf, 'env')
                    next_v = get_vertex(next_info, Player.ODD, prio)
                    game.edges[v].add(next_v)
                    queue.append(next_info)

        elif phase == 'env':
            # Play phase, env turn: env appends one more move to buffer
            for ev in all_env:
                new_buf = buf + (ev,)
                # Now system has buffer of size delay+1 (or 1 if delay=0)
                next_info = (q, new_buf, 'sys')
                next_v = get_vertex(next_info, Player.EVEN, 0)
                game.edges[v].add(next_v)
                queue.append(next_info)

        elif phase == 'sys':
            # Play phase, sys turn: system sees buffer, picks output,
            # NBA transitions on (buf[0], sys_output), buf[0] consumed
            consumed_env = buf[0]
            remaining_buf = buf[1:]

            for sv in all_sys:
                combined = consumed_env | sv
                # Check NBA transitions
                for label, q_next in nba.transitions.get(q, []):
                    if _label_matches(label, combined):
                        is_acc = q_next in nba.accepting
                        prio = 2 if is_acc else 1
                        next_info = (q_next, remaining_buf, 'env')
                        next_v = get_vertex(next_info, Player.ODD, prio)
                        game.edges[v].add(next_v)
                        queue.append(next_info)

    # Add sink for dead-end vertices
    has_dead = False
    for v in list(game.vertices):
        if not game.edges.get(v):
            if not has_dead:
                sink = vid[0]
                vid[0] += 1
                game.vertices.add(sink)
                game.edges[sink] = {sink}
                game.owner[sink] = Player.EVEN
                game.priority[sink] = 1  # losing for system
                has_dead = True
            game.edges[v].add(sink)

    return game, v2info, info2v, initial_vertices


# ============================================================
# Delay Game Synthesis
# ============================================================

def synthesize_with_delay(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    delay: int,
) -> DelayGameResult:
    """
    Synthesize a controller with bounded lookahead (delay).

    The system sees `delay` future environment moves before deciding
    its current output. delay=0 is equivalent to standard synthesis.

    Args:
        spec: LTL specification
        env_vars: environment-controlled variables
        sys_vars: system-controlled variables
        delay: number of future env moves visible to system

    Returns: DelayGameResult with controller if realizable
    """
    if delay < 0:
        raise ValueError("Delay must be non-negative")

    # For delay=0, delegate to standard synthesis
    if delay == 0:
        result = v186_synthesize(spec, env_vars, sys_vars)
        return DelayGameResult(
            realizable=(result.verdict == SynthesisVerdict.REALIZABLE),
            delay=0,
            controller=result.controller,
            buffered_states=result.game_vertices,
            original_states=result.automaton_states,
            method="standard_synthesis",
            iterations=0,
        )

    # Build NBA from spec
    gba = ltl_to_gba(spec)
    nba = gba_to_nba(gba)

    # Build delay game arena
    game, v2info, info2v, initial_verts = build_delay_arena(
        nba, env_vars, sys_vars, delay
    )

    # Solve parity game
    solution = zielonka(game)

    # Check if all initial vertices are winning for system (Even)
    all_initial_winning = all(v in solution.win_even for v in initial_verts)

    if not all_initial_winning:
        return DelayGameResult(
            realizable=False,
            delay=delay,
            buffered_states=len(game.vertices),
            original_states=len(nba.states),
            method="delay_game",
        )

    # Extract controller
    controller = _extract_delay_controller(
        nba, game, solution, v2info, info2v, initial_verts,
        env_vars, sys_vars, delay
    )

    return DelayGameResult(
        realizable=True,
        delay=delay,
        controller=controller,
        buffered_states=len(game.vertices),
        original_states=len(nba.states),
        method="delay_game",
    )


def _extract_delay_controller(
    nba, game, solution, v2info, info2v, initial_verts,
    env_vars, sys_vars, delay
):
    """Extract a MealyMachine controller from the delay game solution."""
    all_env = _all_valuations(env_vars)
    all_sys = _all_valuations(sys_vars)

    # Controller states are (nba_state, buffer) pairs in the winning region
    # We use integer IDs for the Mealy machine
    state_map = {}
    state_id = [0]
    transitions = {}

    def get_state_id(key):
        if key not in state_map:
            state_map[key] = state_id[0]
            state_id[0] += 1
        return state_map[key]

    # Find initial state
    init_key = None
    for iv in initial_verts:
        if iv in solution.win_even:
            info = v2info[iv]
            q, buf, phase = info
            if phase == 'fill' and len(buf) == 0:
                init_key = (q, buf)
                break
            elif phase == 'env' and delay == 0:
                init_key = (q, buf)
                break

    if init_key is None:
        # Fallback: use first winning initial
        for iv in initial_verts:
            if iv in solution.win_even:
                info = v2info[iv]
                init_key = (info[0], info[1])
                break

    if init_key is None:
        return None

    init_id = get_state_id(init_key)

    # BFS through winning strategy to build transitions
    # For the Mealy machine, we abstract away the buffer:
    # input = current env valuation, output = sys response
    # Internal state tracks (nba_state, buffer_contents)

    queue = deque()
    visited_states = set()

    # We need to trace through fill phase + play phase
    # The Mealy machine input is one env valuation per step
    # During fill phase, sys doesn't output anything -- we skip these
    # During play phase, each step: env appends, sys responds

    # For the controller, we model it as:
    # State = (q, buffer_of_size_delay)
    # On input ev: buffer becomes buffer + (ev,), sys picks output,
    #   NBA transitions on (buffer[0], output), new buffer = buffer[1:]
    # This is the play-phase behavior

    # First trace fill phase to get initial play state
    def trace_fill(q, buf):
        """Follow winning strategy through fill phase to reach play state."""
        if len(buf) >= delay:
            return (q, buf)
        # Env chooses -- we need to handle all env choices
        # Return the set of reachable play states
        play_states = set()
        fill_info = (q, buf, 'fill')
        if fill_info not in info2v:
            return None
        v = info2v[fill_info]
        if v not in solution.win_even:
            return None
        for ev in all_env:
            new_buf = buf + (ev,)
            if len(new_buf) < delay:
                result = trace_fill(q, new_buf)
                if result:
                    play_states.add(result)
            else:
                play_states.add((q, new_buf))
        return play_states

    # For the Mealy machine, we need deterministic transitions
    # State = (q, buffer), Input = env_val, Output = sys_val
    # The buffer is part of the controller state

    # BFS from initial play states
    if delay > 0:
        # Get all initial play states (after fill)
        for q0 in nba.initial:
            for buf in _all_buffers(env_vars, delay):
                play_key = (q0, buf)
                env_info = (q0, buf, 'env')
                if env_info in info2v and info2v[env_info] in solution.win_even:
                    queue.append(play_key)
    else:
        queue.append(init_key)

    while queue:
        play_key = queue.popleft()
        if play_key in visited_states:
            continue
        visited_states.add(play_key)
        q, buf = play_key
        sid = get_state_id(play_key)

        for ev in all_env:
            # Env appends ev to buffer
            extended_buf = buf + (ev,)
            sys_info = (q, extended_buf, 'sys')

            if sys_info not in info2v:
                continue
            sys_v = info2v[sys_info]
            if sys_v not in solution.win_even:
                continue

            # System's strategy choice
            if sys_v in solution.strategy_even:
                next_v = solution.strategy_even[sys_v]
            else:
                # Pick any winning successor
                next_v = None
                for succ in game.edges.get(sys_v, set()):
                    if succ in solution.win_even:
                        next_v = succ
                        break
                if next_v is None:
                    continue

            next_info = v2info.get(next_v)
            if next_info is None:
                continue

            q_next, next_buf, _ = next_info

            # Determine system output: which sys_val caused this transition?
            consumed = buf[0] if buf else ev
            for sv in all_sys:
                combined = consumed | sv
                found = False
                for label, qt in nba.transitions.get(q, []):
                    if qt == q_next and _label_matches(label, combined):
                        found = True
                        break
                if found:
                    next_play_key = (q_next, next_buf)
                    next_sid = get_state_id(next_play_key)
                    transitions[(sid, ev)] = (next_sid, sv)
                    queue.append(next_play_key)
                    break

    states = set(range(state_id[0]))
    return MealyMachine(
        states=states,
        initial=init_id,
        inputs=set(env_vars),
        outputs=set(sys_vars),
        transitions=transitions,
    )


def _all_buffers(env_vars: Set[str], size: int) -> List[Tuple]:
    """Generate all possible buffers of given size."""
    all_env = _all_valuations(env_vars)
    if size == 0:
        return [()]
    result = []
    for combo in cartesian_product(all_env, repeat=size):
        result.append(tuple(combo))
    return result


# ============================================================
# Minimum Delay Search
# ============================================================

def find_minimum_delay(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    max_delay: int = 5,
) -> MinDelayResult:
    """
    Find the minimum delay k such that spec is realizable with delay k.

    Searches delay values 0, 1, 2, ..., max_delay.

    Returns MinDelayResult with minimum delay found.
    """
    results = {}
    searched = []

    for k in range(max_delay + 1):
        searched.append(k)
        result = synthesize_with_delay(spec, env_vars, sys_vars, k)
        results[k] = result

        if result.realizable:
            return MinDelayResult(
                realizable=True,
                min_delay=k,
                results=results,
                searched_delays=searched,
            )

    return MinDelayResult(
        realizable=False,
        min_delay=-1,
        results=results,
        searched_delays=searched,
    )


# ============================================================
# GR(1) Delay Games
# ============================================================

def gr1_delay_synthesize(
    spec: BoolGR1Spec,
    delay: int,
) -> DelayGameResult:
    """
    GR(1) synthesis with delay using buffered state space.

    Expands the state space to include environment move buffers,
    then solves the resulting GR(1) game.
    """
    if delay < 0:
        raise ValueError("Delay must be non-negative")

    if delay == 0:
        game = build_bool_game(spec)
        result = gr1_solve(game)
        return DelayGameResult(
            realizable=(result.verdict == GR1Verdict.REALIZABLE),
            delay=0,
            buffered_states=result.n_states,
            original_states=result.n_states,
            method="gr1_standard",
            iterations=result.iterations,
        )

    # Build buffered GR(1) game
    buffered_game = _build_buffered_gr1_game(spec, delay)
    result = gr1_solve(buffered_game, extract_strategy=True)

    controller = None
    if result.verdict == GR1Verdict.REALIZABLE and result.strategy:
        controller = _gr1_delay_to_mealy(
            buffered_game, result.strategy, spec, delay
        )

    return DelayGameResult(
        realizable=(result.verdict == GR1Verdict.REALIZABLE),
        delay=delay,
        controller=controller,
        buffered_states=result.n_states,
        original_states=2 ** (len(spec.env_vars) + len(spec.sys_vars)),
        method="gr1_delay",
        iterations=result.iterations,
    )


def _build_buffered_gr1_game(spec: BoolGR1Spec, delay: int) -> GR1Game:
    """
    Build a GR(1) game with buffered environment moves.

    States: (original_state, buffer) where buffer has `delay` env valuations.
    Transitions: env adds to buffer + sys consumes from buffer front.
    """
    env_vars = list(sorted(spec.env_vars))
    sys_vars = list(sorted(spec.sys_vars))
    all_env = _all_valuations(set(env_vars))
    all_sys = _all_valuations(set(sys_vars))
    all_vals = _all_valuations(set(env_vars) | set(sys_vars))

    # Original states
    orig_states = list(all_vals)

    # Buffered states: (orig_state, buffer_tuple)
    all_bufs = _all_buffers(set(env_vars), delay)

    states = set()
    initial = set()
    transitions = {}
    env_justice_sets = [set() for _ in spec.env_justice]
    sys_justice_sets = [set() for _ in spec.sys_justice]

    for orig in orig_states:
        for buf in all_bufs:
            bstate = (orig, buf)
            states.add(bstate)

            # Check if this is an initial state
            if spec.env_init(orig) and spec.sys_init(orig):
                initial.add(bstate)

            # Justice: based on original state
            for i, jf in enumerate(spec.env_justice):
                if jf(orig):
                    env_justice_sets[i].add(bstate)
            for i, jf in enumerate(spec.sys_justice):
                if jf(orig):
                    sys_justice_sets[i].add(bstate)

            # Transitions: env picks new env val (appended to buffer),
            # sys picks sys val, transition uses buf[0] as current env input
            env_choices = []
            for new_env in all_env:
                # Environment's choice: what to append to buffer
                sys_successors = set()
                consumed = buf[0]  # first element consumed
                remaining = buf[1:] + (new_env,)  # shift buffer

                for sv in all_sys:
                    # Next original state: consumed env vals + sys vals
                    next_orig = consumed | sv
                    # Check transition validity
                    next_env_vals = consumed  # env vars in next state
                    if spec.env_trans(orig, next_env_vals) and spec.sys_trans(orig, next_orig):
                        next_bstate = (next_orig, remaining)
                        if next_bstate in states:
                            sys_successors.add(next_bstate)

                if sys_successors:
                    env_choices.append(sys_successors)

            if env_choices:
                transitions[bstate] = env_choices

    # Filter to reachable states
    reachable = set()
    frontier = deque(initial)
    while frontier:
        s = frontier.popleft()
        if s in reachable:
            continue
        reachable.add(s)
        for env_choice in transitions.get(s, []):
            for succ in env_choice:
                if succ not in reachable:
                    frontier.append(succ)

    # Restrict everything to reachable
    states = reachable
    initial = initial & reachable
    transitions = {s: t for s, t in transitions.items() if s in reachable}
    for i in range(len(env_justice_sets)):
        env_justice_sets[i] = env_justice_sets[i] & reachable
    for i in range(len(sys_justice_sets)):
        sys_justice_sets[i] = sys_justice_sets[i] & reachable

    return GR1Game(
        states=states,
        initial=initial,
        transitions=transitions,
        env_justice=env_justice_sets if env_justice_sets else [],
        sys_justice=sys_justice_sets if sys_justice_sets else [states],
    )


def _gr1_delay_to_mealy(game, strategy, spec, delay):
    """Convert GR(1) delay strategy to MealyMachine."""
    # Simplified: return None for now, strategy is in GR1Strategy form
    # Full extraction requires mapping buffered states back to original
    return None


# ============================================================
# Specialized Delay Synthesis
# ============================================================

def synthesize_safety_with_delay(
    bad_condition: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    delay: int,
) -> DelayGameResult:
    """Synthesize to avoid bad_condition forever, with delay-k lookahead."""
    spec = Globally(Not(bad_condition))
    return synthesize_with_delay(spec, env_vars, sys_vars, delay)


def synthesize_reachability_with_delay(
    target: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    delay: int,
) -> DelayGameResult:
    """Synthesize to eventually reach target, with delay-k lookahead."""
    spec = Finally(target)
    return synthesize_with_delay(spec, env_vars, sys_vars, delay)


def synthesize_response_with_delay(
    trigger: LTL,
    response: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    delay: int,
) -> DelayGameResult:
    """Synthesize G(trigger -> F response) with delay-k lookahead."""
    spec = Globally(Implies(trigger, Finally(response)))
    return synthesize_with_delay(spec, env_vars, sys_vars, delay)


def synthesize_liveness_with_delay(
    condition: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    delay: int,
) -> DelayGameResult:
    """Synthesize GF(condition) with delay-k lookahead."""
    spec = Globally(Finally(condition))
    return synthesize_with_delay(spec, env_vars, sys_vars, delay)


# ============================================================
# Delay Comparison
# ============================================================

def compare_delays(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    delays: List[int],
) -> Dict:
    """
    Compare synthesis results across different delay values.

    Returns dict with per-delay metrics and comparison summary.
    """
    results = {}
    for k in delays:
        r = synthesize_with_delay(spec, env_vars, sys_vars, k)
        results[k] = {
            'realizable': r.realizable,
            'buffered_states': r.buffered_states,
            'controller_states': len(r.controller.states) if r.controller else 0,
            'method': r.method,
        }

    # Find minimum realizable delay
    min_real = None
    for k in sorted(delays):
        if results[k]['realizable']:
            min_real = k
            break

    return {
        'per_delay': results,
        'min_realizable': min_real,
        'delays_tested': sorted(delays),
    }


def delay_benefit_analysis(
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    max_delay: int = 3,
) -> Dict:
    """
    Analyze the benefit of delay for a given specification.

    Reports whether delay helps, the minimum delay needed,
    and the state space growth factor.
    """
    results = {}
    for k in range(max_delay + 1):
        r = synthesize_with_delay(spec, env_vars, sys_vars, k)
        results[k] = r

    standard = results[0]
    benefit = False
    min_k = -1

    for k in range(1, max_delay + 1):
        if results[k].realizable and not standard.realizable:
            benefit = True
            if min_k == -1:
                min_k = k

    growth = {}
    for k in range(1, max_delay + 1):
        if results[0].buffered_states > 0:
            growth[k] = results[k].buffered_states / max(1, results[0].buffered_states)
        else:
            growth[k] = 0

    return {
        'standard_realizable': standard.realizable,
        'delay_helps': benefit,
        'min_delay': min_k,
        'state_growth': growth,
        'results': {k: r.realizable for k, r in results.items()},
    }


# ============================================================
# Verification
# ============================================================

def verify_delay_controller(
    controller: MealyMachine,
    spec: LTL,
    env_vars: Set[str],
    sys_vars: Set[str],
    max_depth: int = 50,
) -> Tuple[bool, List[str]]:
    """Verify a delay controller against its spec using bounded checking."""
    return verify_controller(controller, spec, env_vars, sys_vars, max_depth)


def delay_game_summary(result: DelayGameResult) -> str:
    """Human-readable summary of delay game result."""
    lines = [
        f"Delay Game Synthesis (k={result.delay})",
        f"  Realizable: {result.realizable}",
        f"  Method: {result.method}",
        f"  Buffered states: {result.buffered_states}",
        f"  Original states: {result.original_states}",
    ]
    if result.controller:
        lines.append(f"  Controller states: {len(result.controller.states)}")
        lines.append(f"  Controller transitions: {len(result.controller.transitions)}")
    return "\n".join(lines)


def delay_statistics(result: DelayGameResult) -> Dict:
    """Statistics for a delay game result."""
    stats = {
        'delay': result.delay,
        'realizable': result.realizable,
        'buffered_states': result.buffered_states,
        'original_states': result.original_states,
        'method': result.method,
    }
    if result.controller:
        stats['controller_states'] = len(result.controller.states)
        stats['controller_transitions'] = len(result.controller.transitions)
    return stats
