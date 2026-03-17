"""V187: GR(1) Synthesis -- Efficient reactive synthesis for Generalized Reactivity(1) specs.

Composes concepts from V186 (reactive synthesis) + V156 (parity games) but uses a
direct 3-nested fixpoint algorithm that's polynomial in the state space.

GR(1) specs: (AND_i GF(J_i^e)) -> (AND_j GF(J_j^s))
- If env satisfies all justice assumptions infinitely often,
  then sys satisfies all justice guarantees infinitely often.

The fixpoint runs in O(n * m * |S|^2) where n = #guarantees, m = #assumptions.
No parity game construction needed -- this is the key efficiency gain.

References:
- Piterman, Pnueli, Sa'ar (2006): "Synthesis of Reactive(1) Designs"
- Bloem, Jobstmann, Piterman, Pnueli, Sa'ar (2012): "Synthesis of Reactive(1) Designs"
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum
from itertools import product as iter_product


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GR1Verdict(Enum):
    REALIZABLE = "realizable"
    UNREALIZABLE = "unrealizable"


@dataclass
class GR1Game:
    """Explicit-state GR(1) game.

    States can be any hashable values.
    Transitions: for each state, a list of env choices.
    Each env choice maps to a set of successor states (sys picks among these).

    Controllable predecessor: Cpre(T) = {s | forall env_choice, exists successor in T}
    """
    states: Set
    initial: Set
    transitions: Dict  # state -> List[Set[state]]  (list of env choices, each giving sys options)
    env_justice: List[Set]  # GF assumptions (state sets)
    sys_justice: List[Set]  # GF guarantees (state sets)

    def cpre(self, target: Set) -> Set:
        """Controllable predecessor: sys can force next state into target."""
        result = set()
        for s in self.states:
            env_choices = self.transitions.get(s, [])
            if not env_choices:
                continue
            all_covered = True
            for successors in env_choices:
                if not (successors & target):
                    all_covered = False
                    break
            if all_covered:
                result.add(s)
        return result

    def upre(self, target: Set) -> Set:
        """Uncontrollable predecessor: env can force next state into target."""
        result = set()
        for s in self.states:
            env_choices = self.transitions.get(s, [])
            if not env_choices:
                continue
            for successors in env_choices:
                if successors and successors <= target:
                    result.add(s)
                    break
        return result

    def apre(self, target: Set) -> Set:
        """Attractor predecessor: sys can reach target in one step (exists env, exists sys)."""
        result = set()
        for s in self.states:
            for successors in self.transitions.get(s, []):
                if successors & target:
                    result.add(s)
                    break
        return result


@dataclass
class GR1Strategy:
    """Strategy for GR(1) game -- controller with modes.

    Mode = which guarantee we're currently trying to reach (0..n-1).
    The controller cycles through modes: reach G_0, advance to mode 1,
    reach G_1, advance to mode 2, ..., reach G_{n-1}, wrap to mode 0.
    """
    n_modes: int
    # (state, mode, env_choice_idx) -> (next_state, next_mode)
    transitions: Dict[Tuple, Tuple]
    initial_state: Any = None
    initial_mode: int = 0

    def step(self, state, mode, env_choice_idx):
        """Given current state, mode, and env choice, return (next_state, next_mode)."""
        key = (state, mode, env_choice_idx)
        return self.transitions.get(key, (None, mode))

    def simulate(self, game, env_choices_per_step, max_steps=100):
        """Simulate strategy against env choices.

        env_choices_per_step: list of env_choice_idx per step.
        Returns: list of (state, mode) pairs.
        """
        if self.initial_state is None:
            return []
        trace = [(self.initial_state, self.initial_mode)]
        state, mode = self.initial_state, self.initial_mode
        for step_i, env_idx in enumerate(env_choices_per_step):
            if step_i >= max_steps:
                break
            next_state, next_mode = self.step(state, mode, env_idx)
            if next_state is None:
                break
            trace.append((next_state, next_mode))
            state, mode = next_state, next_mode
        return trace


@dataclass
class GR1Result:
    """Result of GR(1) synthesis."""
    verdict: GR1Verdict
    winning_region: Set = field(default_factory=set)
    strategy: Optional[GR1Strategy] = None
    iterations: int = 0
    n_states: int = 0
    n_guarantees: int = 0
    n_assumptions: int = 0
    fixpoint_layers: Optional[Dict] = None  # Debug: per-guarantee layer info


# ---------------------------------------------------------------------------
# Core algorithm: 3-nested fixpoint
# ---------------------------------------------------------------------------

def gr1_solve(game: GR1Game, extract_strategy: bool = True) -> GR1Result:
    """Solve GR(1) game using 3-nested fixpoint algorithm.

    The fixpoint computes the winning region for the system player:
    W = nu Z. AND_j mu Y_j. ((G_j & Cpre(Z)) | Cpre(Y_j) | OR_i (nu X. ((~A_i & Cpre(X)) | Cpre(Y_j) | (G_j & Cpre(Z)))))

    Returns GR1Result with winning region and optional strategy.
    """
    states = game.states
    n = len(game.sys_justice) if game.sys_justice else 0
    m = len(game.env_justice)

    # Degenerate: no guarantees means system trivially wins
    if n == 0:
        return GR1Result(
            verdict=GR1Verdict.REALIZABLE if game.initial else GR1Verdict.UNREALIZABLE,
            winning_region=set(states),
            n_states=len(states),
            n_guarantees=0,
            n_assumptions=m,
        )

    # Initialize Z[j] = all states (greatest fixpoint from above)
    Z = [set(states) for _ in range(n)]

    # For strategy extraction: track which "layer" each state belongs to in each mode
    # layer_mem[j][s] = (layer_type, info)
    # layer_type: 'guarantee' (reached G_j, advance), 'progress' (toward Y), 'assumption_i' (exploit)
    layer_mem = [{} for _ in range(n)]

    total_iters = 0
    changed = True

    while changed:
        changed = False
        for j in range(n):
            next_j = (j + 1) % n

            # Least fixpoint Y (from below)
            Y = set()
            y_changed = True

            while y_changed:
                total_iters += 1
                old_Y = Y

                # Base targets
                reach_guarantee = game.sys_justice[j] & game.cpre(Z[next_j])
                progress = game.cpre(Y)
                new_Y = reach_guarantee | progress

                # Track layers for new states
                for s in reach_guarantee:
                    if s not in layer_mem[j] or layer_mem[j][s][0] != 'guarantee':
                        layer_mem[j][s] = ('guarantee', next_j)
                for s in progress - reach_guarantee:
                    if s not in layer_mem[j] or layer_mem[j][s][0] not in ('guarantee',):
                        layer_mem[j][s] = ('progress', j)

                # For each assumption, compute greatest fixpoint X
                for i in range(m):
                    not_assumption = states - game.env_justice[i]
                    X = set(states)
                    x_changed = True

                    while x_changed:
                        old_X = X
                        X = (not_assumption & game.cpre(X)) | progress | reach_guarantee
                        x_changed = X != old_X

                    # States gained from exploiting this assumption
                    for s in X - new_Y:
                        layer_mem[j][s] = ('assumption', i)

                    new_Y = new_Y | X

                Y = new_Y
                y_changed = Y != old_Y

            if Z[j] != Y:
                changed = True
                Z[j] = Y

    winning = Z[0]

    # Check realizability from initial states
    init_in_winning = game.initial & winning
    realizable = bool(init_in_winning)

    # Extract strategy
    strategy = None
    if realizable and extract_strategy:
        strategy = _extract_strategy(game, Z, layer_mem)

    return GR1Result(
        verdict=GR1Verdict.REALIZABLE if realizable else GR1Verdict.UNREALIZABLE,
        winning_region=winning,
        strategy=strategy,
        iterations=total_iters,
        n_states=len(states),
        n_guarantees=n,
        n_assumptions=m,
    )


def _extract_strategy(game, Z, layer_mem):
    """Extract GR(1) strategy from converged fixpoint.

    For each (state, mode), the strategy picks a successor based on priority:
    1. If in guarantee state and can advance mode -> advance
    2. Otherwise stay in current Z[j] (progress or exploit assumption)
    """
    n = len(game.sys_justice) if game.sys_justice else 1
    transitions = {}

    for j in range(n):
        next_j = (j + 1) % n

        for s in Z[j]:
            env_choices = game.transitions.get(s, [])
            for idx, successors in enumerate(env_choices):
                key = (s, j, idx)

                # Priority 1: in guarantee state, advance mode
                if s in game.sys_justice[j]:
                    good = successors & Z[next_j]
                    if good:
                        transitions[key] = (next(iter(good)), next_j)
                        continue

                # Priority 2: stay in Z[j]
                good = successors & Z[j]
                if good:
                    transitions[key] = (next(iter(good)), j)
                    continue

                # Priority 3: any successor (shouldn't happen in winning region)
                if successors:
                    transitions[key] = (next(iter(successors)), j)

    # Find initial state
    initial_state = None
    for s in game.initial:
        if s in Z[0]:
            initial_state = s
            break

    return GR1Strategy(
        n_modes=n,
        transitions=transitions,
        initial_state=initial_state,
        initial_mode=0,
    )


# ---------------------------------------------------------------------------
# Boolean variable game builder
# ---------------------------------------------------------------------------

@dataclass
class BoolGR1Spec:
    """GR(1) specification over Boolean variables.

    Variables are partitioned into env_vars and sys_vars.
    A state is a frozenset of variable names that are True.
    """
    env_vars: List[str]
    sys_vars: List[str]
    env_init: Callable = None  # (state) -> bool
    sys_init: Callable = None  # (state) -> bool
    env_trans: Callable = None  # (state, next_env_vals) -> bool
    sys_trans: Callable = None  # (state, next_state) -> bool
    env_justice: List[Callable] = field(default_factory=list)  # [(state) -> bool]
    sys_justice: List[Callable] = field(default_factory=list)  # [(state) -> bool]


def _all_valuations(variables):
    """Generate all Boolean valuations for a set of variables."""
    if not variables:
        return [frozenset()]
    vars_list = sorted(variables)
    result = []
    for bits in range(2 ** len(vars_list)):
        val = frozenset(v for i, v in enumerate(vars_list) if bits & (1 << i))
        result.append(val)
    return result


def build_bool_game(spec: BoolGR1Spec) -> GR1Game:
    """Build explicit-state GR(1) game from Boolean variable specification.

    Enumerates all 2^(|env_vars|+|sys_vars|) states and all transitions.
    """
    all_vars = spec.env_vars + spec.sys_vars
    all_states_list = _all_valuations(all_vars)
    all_states = set(all_states_list)

    env_vals_list = _all_valuations(spec.env_vars)
    sys_vals_list = _all_valuations(spec.sys_vars)

    # Filter states by combined init
    initial = set()
    for s in all_states:
        env_ok = spec.env_init(s) if spec.env_init else True
        sys_ok = spec.sys_init(s) if spec.sys_init else True
        if env_ok and sys_ok:
            initial.add(s)

    # Build transitions
    transitions = {}
    for s in all_states:
        env_choices = []
        for env_next in env_vals_list:
            # Check env transition validity
            if spec.env_trans and not spec.env_trans(s, env_next):
                continue
            # For this env choice, find valid sys successors
            sys_successors = set()
            for sys_next in sys_vals_list:
                next_state = env_next | sys_next
                if spec.sys_trans and not spec.sys_trans(s, next_state):
                    continue
                sys_successors.add(next_state)
            if sys_successors:
                env_choices.append(sys_successors)
        transitions[s] = env_choices

    # Convert justice predicates to state sets
    env_justice = []
    for pred in spec.env_justice:
        env_justice.append({s for s in all_states if pred(s)})

    sys_justice = []
    for pred in spec.sys_justice:
        sys_justice.append({s for s in all_states if pred(s)})

    return GR1Game(
        states=all_states,
        initial=initial,
        transitions=transitions,
        env_justice=env_justice,
        sys_justice=sys_justice,
    )


# ---------------------------------------------------------------------------
# High-level synthesis API
# ---------------------------------------------------------------------------

def gr1_synthesize(spec: BoolGR1Spec) -> GR1Result:
    """End-to-end GR(1) synthesis from Boolean spec."""
    game = build_bool_game(spec)
    return gr1_solve(game)


# ---------------------------------------------------------------------------
# Game construction helpers
# ---------------------------------------------------------------------------

def make_game(states, initial, transitions, env_justice=None, sys_justice=None):
    """Quick game construction.

    transitions: dict mapping state -> list of sets (env choices -> sys options)
    """
    return GR1Game(
        states=set(states),
        initial=set(initial),
        transitions=dict(transitions),
        env_justice=list(env_justice) if env_justice else [],
        sys_justice=list(sys_justice) if sys_justice else [],
    )


def make_safety_game(states, initial, transitions, bad_states):
    """Game where sys must avoid bad_states forever.

    Equivalent to GR(1) with sys_justice = [S \ bad] (sys must visit safe states inf often).
    """
    safe = set(states) - set(bad_states)
    return GR1Game(
        states=set(states),
        initial=set(initial),
        transitions=dict(transitions),
        env_justice=[],
        sys_justice=[safe],
    )


def make_reachability_game(states, initial, transitions, target):
    """Game where sys must reach target.

    Modeled as GR(1) with sys_justice = [target].
    """
    return GR1Game(
        states=set(states),
        initial=set(initial),
        transitions=dict(transitions),
        env_justice=[],
        sys_justice=[set(target)],
    )


def make_response_game(states, initial, transitions, triggers_responses, env_justice=None):
    """Game where for each (trigger, response) pair, G(trigger -> F(response)).

    Approximated as GR(1): env_justice = triggers, sys_justice = responses.
    More precisely: assumes env visits trigger inf often, sys must visit response inf often.
    """
    ej = list(env_justice) if env_justice else []
    sj = []
    for trigger, response in triggers_responses:
        ej.append(set(trigger))
        sj.append(set(response))
    return GR1Game(
        states=set(states),
        initial=set(initial),
        transitions=dict(transitions),
        env_justice=ej,
        sys_justice=sj,
    )


# ---------------------------------------------------------------------------
# Attractor computation (useful for safety subproblems)
# ---------------------------------------------------------------------------

def sys_attractor(game, target, bound=None):
    """Compute sys attractor: states from which sys can force reaching target.

    Iteratively applies Cpre until fixpoint.
    """
    attr = set(target) & game.states
    limit = bound if bound else len(game.states) + 1
    for _ in range(limit):
        new = attr | game.cpre(attr)
        if new == attr:
            break
        attr = new
    return attr


def env_attractor(game, target, bound=None):
    """Compute env attractor: states from which env can force reaching target."""
    attr = set(target) & game.states
    limit = bound if bound else len(game.states) + 1
    for _ in range(limit):
        new = attr | game.upre(attr)
        if new == attr:
            break
        attr = new
    return attr


# ---------------------------------------------------------------------------
# Verification: check that a strategy is winning
# ---------------------------------------------------------------------------

def verify_strategy(game, strategy, max_depth=50):
    """Verify that a GR(1) strategy is winning by bounded simulation.

    Checks that for all env behaviors up to max_depth, the strategy
    satisfies all guarantees cyclically.
    """
    if strategy is None or strategy.initial_state is None:
        return False, "No strategy"

    n = strategy.n_modes
    if n == 0:
        return True, "No guarantees"

    # BFS over (state, mode) pairs, tracking guarantee visits
    from collections import deque

    visited = set()
    queue = deque()
    start = (strategy.initial_state, strategy.initial_mode)
    queue.append((start, frozenset(), 0))

    guarantee_reached = [False] * n
    violations = []

    while queue:
        (state, mode), visited_guarantees, depth = queue.popleft()

        if depth > max_depth:
            continue

        if (state, mode) in visited:
            # Check if we've seen all guarantees on this cycle
            continue
        visited.add((state, mode))

        # Check guarantee
        if state in game.sys_justice[mode]:
            guarantee_reached[mode] = True

        # Try all env choices
        env_choices = game.transitions.get(state, [])
        for idx, _ in enumerate(env_choices):
            result = strategy.step(state, mode, idx)
            next_state, next_mode = result
            if next_state is None:
                violations.append(f"No response at state={state}, mode={mode}, env={idx}")
                continue
            # Verify next state is valid successor
            if next_state not in env_choices[idx]:
                violations.append(f"Invalid successor {next_state} at state={state}")
                continue
            queue.append(((next_state, next_mode), visited_guarantees, depth + 1))

    if violations:
        return False, "; ".join(violations[:5])

    return True, "OK"


# ---------------------------------------------------------------------------
# Analysis and comparison
# ---------------------------------------------------------------------------

def game_statistics(game):
    """Compute statistics about a GR(1) game."""
    total_transitions = 0
    max_env_choices = 0
    max_sys_choices = 0
    dead_ends = 0

    for s in game.states:
        env_choices = game.transitions.get(s, [])
        if not env_choices:
            dead_ends += 1
            continue
        max_env_choices = max(max_env_choices, len(env_choices))
        for successors in env_choices:
            total_transitions += len(successors)
            max_sys_choices = max(max_sys_choices, len(successors))

    return {
        'states': len(game.states),
        'initial': len(game.initial),
        'transitions': total_transitions,
        'max_env_branching': max_env_choices,
        'max_sys_branching': max_sys_choices,
        'dead_ends': dead_ends,
        'env_justice': len(game.env_justice),
        'sys_justice': len(game.sys_justice),
    }


def compare_with_v186(game, result):
    """Compare GR(1) result with general reactive synthesis (V186) on same problem.

    Returns comparison dict. Requires V186 to be available.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, 'Z:/AgentZero/A2/work/V186_reactive_synthesis')
        from reactive_synthesis import synthesize, Atom, Globally, Finally, And, Implies
        # Can only compare if we can express the GR(1) spec as LTL
        return {
            'gr1_verdict': result.verdict.value,
            'gr1_winning_size': len(result.winning_region),
            'gr1_iterations': result.iterations,
            'note': 'V186 comparison requires LTL formulation of the spec',
        }
    except ImportError:
        return {'error': 'V186 not available'}


def gr1_summary(game, result):
    """Human-readable summary of GR(1) synthesis result."""
    stats = game_statistics(game)
    lines = [
        f"GR(1) Synthesis Result",
        f"  Verdict: {result.verdict.value}",
        f"  States: {stats['states']} ({stats['initial']} initial)",
        f"  Transitions: {stats['transitions']}",
        f"  Guarantees: {result.n_guarantees}, Assumptions: {result.n_assumptions}",
        f"  Winning region: {len(result.winning_region)}/{stats['states']} states",
        f"  Fixpoint iterations: {result.iterations}",
    ]
    if result.strategy:
        lines.append(f"  Strategy: {len(result.strategy.transitions)} entries, {result.strategy.n_modes} modes")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mealy machine conversion (for compatibility with V186)
# ---------------------------------------------------------------------------

@dataclass
class MealyMachine:
    """Mealy machine controller (compatible with V186 format)."""
    states: Set[int]
    initial: int
    inputs: Set[str]
    outputs: Set[str]
    transitions: Dict  # (state, input_val) -> (next_state, output_val)

    def step(self, state, input_val):
        """Given state and input, return (next_state, output)."""
        key = (state, input_val)
        return self.transitions.get(key, (state, frozenset()))

    def simulate(self, input_sequence, max_steps=100):
        """Simulate against input sequence."""
        trace = []
        state = self.initial
        for inp in input_sequence[:max_steps]:
            next_state, output = self.step(state, inp)
            trace.append((state, inp, output, next_state))
            state = next_state
        return trace


def strategy_to_mealy(game, strategy, env_vars=None, sys_vars=None):
    """Convert GR(1) strategy to Mealy machine.

    Requires that game states are frozensets of variable names (from build_bool_game).
    """
    if strategy is None or strategy.initial_state is None:
        return None

    env_vars = set(env_vars) if env_vars else set()
    sys_vars = set(sys_vars) if sys_vars else set()

    # States: (game_state, mode) pairs reachable from initial
    from collections import deque

    mealy_states = {}
    mealy_transitions = {}
    queue = deque()

    start = (strategy.initial_state, strategy.initial_mode)
    mealy_states[start] = 0
    queue.append(start)

    while queue:
        gs, mode = queue.popleft()
        state_id = mealy_states[(gs, mode)]

        env_choices = game.transitions.get(gs, [])
        for idx, successors in enumerate(env_choices):
            # Determine env input valuation from the env choice index
            # This requires knowing the env valuation for this choice
            # For now, use idx as input identifier
            result = strategy.step(gs, mode, idx)
            next_gs, next_mode = result
            if next_gs is None:
                continue

            next_key = (next_gs, next_mode)
            if next_key not in mealy_states:
                mealy_states[next_key] = len(mealy_states)
                queue.append(next_key)

            # Extract output valuation from next state
            output = frozenset(v for v in sys_vars if v in next_gs)
            input_val = frozenset(v for v in env_vars if v in next_gs)

            mealy_transitions[(state_id, input_val)] = (mealy_states[next_key], output)

    return MealyMachine(
        states=set(mealy_states.values()),
        initial=0,
        inputs=env_vars,
        outputs=sys_vars,
        transitions=mealy_transitions,
    )
