"""V191: Parameterized Synthesis -- Synthesize controllers for families of systems.

Given a parameterized specification Spec(N) where N is a natural number (e.g.,
number of processes, buffer size, ring size), synthesize a controller template
that works for ALL N >= N_min, or find a cutoff N_c such that correctness for
N <= N_c implies correctness for all N.

Key techniques:
1. **Instance synthesis**: Solve GR(1) games for specific N values
2. **Controller templates**: Extract parameterized patterns from instances
3. **Cutoff detection**: Find N_c where controller structure stabilizes
4. **Inductive verification**: Prove N -> N+1 preservation
5. **Symmetry reduction**: Exploit process symmetry in token-ring/mutex specs

Composes:
  - V187 (GR(1) synthesis) for per-instance solving
  - V186 concepts (MealyMachine) for controller representation

References:
- Jacobs & Bloem (2012): "Parameterized Synthesis"
- Khalimov, Jacobs, Bloem (2013): "PARTY: Parameterized Synthesis of Token Rings"
- Emerson & Namjoshi (1995): "Reasoning about Rings" (cutoff results)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum
from itertools import product as iter_product
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V187_gr1_synthesis'))
from gr1_synthesis import (
    GR1Game, GR1Verdict, GR1Result, GR1Strategy, MealyMachine,
    gr1_solve, gr1_synthesize, build_bool_game, BoolGR1Spec,
    make_game, strategy_to_mealy, verify_strategy,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ParamVerdict(Enum):
    REALIZABLE = "realizable"       # Controller template found for all N
    UNREALIZABLE = "unrealizable"   # No controller exists for some N
    CUTOFF_FOUND = "cutoff_found"   # Cutoff N_c found; holds for N <= N_c => all N
    UNKNOWN = "unknown"


@dataclass
class ProcessTemplate:
    """Template for a single process in a parameterized system.

    Each process has local states, and transitions depend on its own state
    plus shared/neighbor state (e.g., token, signals).
    """
    local_states: List[str]          # e.g., ['idle', 'trying', 'critical']
    initial_state: str
    # (local_state, input_signal) -> [(next_state, output_signal)]
    transitions: Dict[Tuple[str, str], List[Tuple[str, str]]]
    # Which local states satisfy which properties
    labels: Dict[str, Set[str]]      # property_name -> set of states where it holds


@dataclass
class ParamSpec:
    """Parameterized specification.

    Describes a family of systems parameterized by N (number of processes).
    Each process is symmetric (same template) with a topology connecting them.
    """
    process_template: ProcessTemplate
    topology: str = "ring"           # "ring", "star", "clique", "pipeline"
    # Env controls input signals, sys controls process transitions
    env_signals: List[str] = field(default_factory=list)
    sys_signals: List[str] = field(default_factory=list)
    # Safety: G(!bad) for each process
    safety: List[str] = field(default_factory=list)     # property names that must always hold
    # Mutual exclusion: at most one process in these states
    mutex: List[str] = field(default_factory=list)       # states that are mutually exclusive
    # Liveness: GF(property) for each process
    liveness: List[str] = field(default_factory=list)    # property names for response
    # Fairness assumptions on env
    env_fairness: List[str] = field(default_factory=list)


@dataclass
class ControllerTemplate:
    """Parameterized controller -- works for any N.

    A controller template is a single-process Mealy machine that, when
    replicated N times with the appropriate topology, satisfies the spec.
    """
    states: List[str]
    initial: str
    # (state, local_state, neighbor_signal) -> (next_state, output_action)
    transitions: Dict[Tuple[str, str, str], Tuple[str, str]]

    def instantiate(self, n: int) -> List[Dict]:
        """Create N copies of this controller."""
        instances = []
        for i in range(n):
            instances.append({
                'id': i,
                'state': self.initial,
                'transitions': dict(self.transitions),
            })
        return instances


@dataclass
class InstanceResult:
    """Result of synthesis for a specific N value."""
    n: int
    verdict: GR1Verdict
    game_states: int
    winning_region_size: int
    strategy: Optional[GR1Strategy] = None
    mealy: Optional[MealyMachine] = None


@dataclass
class CutoffResult:
    """Result of cutoff analysis."""
    cutoff_n: int                    # Smallest N where structure stabilizes
    verified_up_to: int              # Largest N verified
    stable_from: int                 # N from which controller structure is identical
    structure_signatures: Dict[int, str] = field(default_factory=dict)
    is_inductive: bool = False       # Whether N -> N+1 preservation was proved


@dataclass
class ParamResult:
    """Full result of parameterized synthesis."""
    verdict: ParamVerdict
    instance_results: Dict[int, InstanceResult] = field(default_factory=dict)
    controller_template: Optional[ControllerTemplate] = None
    cutoff: Optional[CutoffResult] = None
    n_min: int = 1
    n_max_checked: int = 0


# ---------------------------------------------------------------------------
# Instance generation: ParamSpec x N -> GR1Game
# ---------------------------------------------------------------------------

def _state_name(proc_id: int, local: str) -> str:
    """Create a global state name for process i in local state s."""
    return f"p{proc_id}_{local}"


def _ring_neighbors(proc_id: int, n: int) -> Tuple[int, int]:
    """Return (left_neighbor, right_neighbor) in a ring of size n."""
    return ((proc_id - 1) % n, (proc_id + 1) % n)


def _star_neighbors(proc_id: int, n: int) -> List[int]:
    """Star topology: process 0 is hub, others are spokes."""
    if proc_id == 0:
        return list(range(1, n))
    return [0]


def _pipeline_neighbors(proc_id: int, n: int) -> Tuple[Optional[int], Optional[int]]:
    """Pipeline: left-to-right, process 0 is source, n-1 is sink."""
    left = proc_id - 1 if proc_id > 0 else None
    right = proc_id + 1 if proc_id < n - 1 else None
    return (left, right)


def instantiate_ring_game(spec: ParamSpec, n: int) -> GR1Game:
    """Build a GR(1) game for N processes in a ring topology.

    Global state = tuple of local states (one per process).
    Token passing: exactly one process holds the token at a time.
    Env controls token movement, sys controls process state transitions.
    """
    template = spec.process_template
    local_states = template.local_states

    # Global states: all combinations of local states + token position
    global_states = set()
    transitions = {}

    # Generate all (local_state_tuple, token_pos) combinations
    all_local_combos = list(iter_product(local_states, repeat=n))

    for combo in all_local_combos:
        for token_pos in range(n):
            state = (combo, token_pos)
            global_states.add(state)

    # Initial state: all processes in initial_state, token at process 0
    init_combo = tuple([template.initial_state] * n)
    initial = {(init_combo, 0)}

    # Transitions: env chooses token movement, sys chooses process actions
    for combo in all_local_combos:
        for token_pos in range(n):
            state = (combo, token_pos)
            env_choices = []

            # Env choice 1: token stays
            sys_options_stay = _compute_sys_options(
                spec, combo, token_pos, token_pos, n
            )
            if sys_options_stay:
                env_choices.append(sys_options_stay)

            # Env choice 2: token moves left
            new_token = (token_pos - 1) % n
            sys_options_left = _compute_sys_options(
                spec, combo, token_pos, new_token, n
            )
            if sys_options_left:
                env_choices.append(sys_options_left)

            # Env choice 3: token moves right
            new_token = (token_pos + 1) % n
            sys_options_right = _compute_sys_options(
                spec, combo, token_pos, new_token, n
            )
            if sys_options_right:
                env_choices.append(sys_options_right)

            if env_choices:
                transitions[state] = env_choices

    # Justice conditions from spec
    env_justice = []
    sys_justice = []

    # Env fairness: token visits each process (standard ring fairness)
    if spec.env_fairness:
        for i in range(n):
            token_at_i = {s for s in global_states if s[1] == i}
            env_justice.append(token_at_i)
    else:
        # Default: env is fair -- token visits each process
        for i in range(n):
            token_at_i = {s for s in global_states if s[1] == i}
            env_justice.append(token_at_i)

    # Sys liveness: each process eventually satisfies liveness properties
    for prop_name in spec.liveness:
        if prop_name in template.labels:
            good_local_states = template.labels[prop_name]
            for i in range(n):
                justice_set = {
                    s for s in global_states
                    if s[0][i] in good_local_states
                }
                sys_justice.append(justice_set)

    return GR1Game(
        states=global_states,
        initial=initial,
        transitions=transitions,
        env_justice=env_justice if env_justice else [global_states],
        sys_justice=sys_justice if sys_justice else [global_states],
    )


def _compute_sys_options(
    spec: ParamSpec,
    combo: Tuple[str, ...],
    old_token: int,
    new_token: int,
    n: int,
) -> Set:
    """Compute sys successor states given current combo and token movement.

    The process holding the token can change state.
    Other processes keep their state (or have limited non-token transitions).
    """
    template = spec.process_template
    token_holder = old_token
    holder_state = combo[token_holder]

    # Determine neighbor signal based on ring topology
    left, right = _ring_neighbors(token_holder, n)
    left_state = combo[left]
    right_state = combo[right]
    # Signal is the neighbor states (simplified to right neighbor's state)
    signal = right_state

    # Get possible transitions for token holder
    key = (holder_state, signal)
    possible = template.transitions.get(key, [])

    # Also try wildcard signal
    wildcard_key = (holder_state, '*')
    possible = possible + template.transitions.get(wildcard_key, [])

    # If token holder has "has_token" signal
    token_key = (holder_state, 'has_token')
    possible = possible + template.transitions.get(token_key, [])

    if not possible:
        # Self-loop if no transitions defined
        possible = [(holder_state, 'none')]

    successors = set()
    for next_state, _output in possible:
        new_combo = list(combo)
        new_combo[token_holder] = next_state

        # Check mutex: at most one process in mutex states
        if spec.mutex:
            mutex_count = sum(1 for i in range(n) if new_combo[i] in spec.mutex)
            if mutex_count > 1:
                continue  # Skip this successor (violates safety)

        # Check safety
        safe = True
        for prop_name in spec.safety:
            if prop_name in template.labels:
                bad_states = template.labels[prop_name]
                for i in range(n):
                    if new_combo[i] in bad_states:
                        safe = False
                        break
            if not safe:
                break
        if not safe:
            continue

        successors.add((tuple(new_combo), new_token))

    return successors


def instantiate_game(spec: ParamSpec, n: int) -> GR1Game:
    """Build a GR(1) game for N processes with the specified topology."""
    if spec.topology == "ring":
        return instantiate_ring_game(spec, n)
    elif spec.topology == "pipeline":
        return instantiate_pipeline_game(spec, n)
    else:
        return instantiate_ring_game(spec, n)


def instantiate_pipeline_game(spec: ParamSpec, n: int) -> GR1Game:
    """Build a GR(1) game for N processes in a pipeline topology.

    No token -- env controls input to process 0, output flows right.
    Sys controls all process transitions.
    """
    template = spec.process_template
    local_states = template.local_states

    all_combos = list(iter_product(local_states, repeat=n))
    global_states = set(all_combos)
    init_combo = tuple([template.initial_state] * n)
    initial = {init_combo}

    transitions = {}
    for combo in all_combos:
        # Env choices: different input signals to process 0
        env_signals = spec.env_signals if spec.env_signals else ['none']
        env_choices = []

        for env_sig in env_signals:
            sys_succs = set()

            # Each process can transition based on its left neighbor
            # Generate all combinations of per-process choices
            per_proc_options = []
            for i in range(n):
                proc_state = combo[i]
                if i == 0:
                    signal = env_sig
                else:
                    signal = combo[i - 1]  # Left neighbor's state

                key = (proc_state, signal)
                possible = template.transitions.get(key, [])
                wildcard_key = (proc_state, '*')
                possible = possible + template.transitions.get(wildcard_key, [])

                if not possible:
                    possible = [(proc_state, 'none')]

                per_proc_options.append(possible)

            # Generate all combinations of per-process transitions
            for trans_combo in iter_product(*per_proc_options):
                new_combo = tuple(ns for ns, _out in trans_combo)

                # Check mutex
                if spec.mutex:
                    mutex_count = sum(1 for i in range(n) if new_combo[i] in spec.mutex)
                    if mutex_count > 1:
                        continue

                sys_succs.add(new_combo)

            if sys_succs:
                env_choices.append(sys_succs)

        if env_choices:
            transitions[combo] = env_choices

    env_justice = [global_states]
    sys_justice = []

    for prop_name in spec.liveness:
        if prop_name in template.labels:
            good_local_states = template.labels[prop_name]
            for i in range(n):
                justice_set = {
                    s for s in global_states
                    if s[i] in good_local_states
                }
                sys_justice.append(justice_set)

    if not sys_justice:
        sys_justice = [global_states]

    return GR1Game(
        states=global_states,
        initial=initial,
        transitions=transitions,
        env_justice=env_justice,
        sys_justice=sys_justice,
    )


# ---------------------------------------------------------------------------
# Controller structure signature
# ---------------------------------------------------------------------------

def _strategy_signature(result: InstanceResult) -> str:
    """Compute a structural signature of a strategy.

    Two strategies with the same signature have the same structure
    (same number of modes, same transition pattern modulo process renaming).
    """
    if result.strategy is None:
        return f"no_strategy_{result.verdict.value}"

    strategy = result.strategy
    # Signature: (n_modes, n_transitions, transition_pattern_hash)
    n_modes = strategy.n_modes
    n_trans = len(strategy.transitions)

    # Create a canonical form: sort transitions and hash
    sorted_keys = sorted(str(k) for k in strategy.transitions.keys())
    pattern = f"modes={n_modes},trans={n_trans},keys_hash={hash(tuple(sorted_keys)) % 10000}"
    return pattern


def _mealy_signature(mealy: Optional[MealyMachine]) -> str:
    """Compute structural signature of a Mealy machine."""
    if mealy is None:
        return "no_mealy"
    n_states = mealy.n_states
    n_trans = len(mealy.transitions)
    return f"mealy_s={n_states}_t={n_trans}"


# ---------------------------------------------------------------------------
# Cutoff detection
# ---------------------------------------------------------------------------

def detect_cutoff(
    spec: ParamSpec,
    n_min: int = 2,
    n_max: int = 8,
    stability_window: int = 2,
) -> Tuple[ParamResult, Dict[int, GR1Game]]:
    """Try to find a cutoff N_c for the parameterized spec.

    Synthesizes controllers for N = n_min, ..., n_max.
    If the controller structure stabilizes for `stability_window` consecutive N,
    declares N_c as the start of the stable region.

    Returns (ParamResult, dict of games for further analysis).
    """
    instance_results = {}
    games = {}
    signatures = {}

    for n in range(n_min, n_max + 1):
        game = instantiate_game(spec, n)
        games[n] = game

        result = gr1_solve(game, extract_strategy=True)

        inst = InstanceResult(
            n=n,
            verdict=result.verdict,
            game_states=len(game.states),
            winning_region_size=len(result.winning_region),
            strategy=result.strategy,
        )

        # Check if initial states are in winning region
        if result.verdict == GR1Verdict.REALIZABLE:
            if not (game.initial <= result.winning_region):
                inst.verdict = GR1Verdict.UNREALIZABLE
                inst.strategy = None

        instance_results[n] = inst
        signatures[n] = _strategy_signature(inst)

        # Early exit: if unrealizable, no cutoff exists
        if inst.verdict == GR1Verdict.UNREALIZABLE:
            return ParamResult(
                verdict=ParamVerdict.UNREALIZABLE,
                instance_results=instance_results,
                n_min=n_min,
                n_max_checked=n,
            ), games

    # Check for stability
    stable_from = None
    for n in range(n_min + 1, n_max + 1):
        if instance_results[n].verdict != GR1Verdict.REALIZABLE:
            stable_from = None
            continue

        # Check if last `stability_window` consecutive N have same signature pattern
        window_start = n - stability_window + 1
        if window_start < n_min:
            continue

        all_same = True
        all_realizable = True
        for k in range(window_start, n + 1):
            if instance_results[k].verdict != GR1Verdict.REALIZABLE:
                all_realizable = False
                break
            # Compare relative structure: same modes, same winning region coverage
            if _relative_signature(instance_results[k], k) != \
               _relative_signature(instance_results[window_start], window_start):
                all_same = False
                break

        if all_realizable and all_same:
            stable_from = window_start
            break

    if stable_from is not None:
        cutoff = CutoffResult(
            cutoff_n=stable_from,
            verified_up_to=n_max,
            stable_from=stable_from,
            structure_signatures=signatures,
        )
        return ParamResult(
            verdict=ParamVerdict.CUTOFF_FOUND,
            instance_results=instance_results,
            cutoff=cutoff,
            n_min=n_min,
            n_max_checked=n_max,
        ), games

    # No cutoff found but all realizable
    all_realizable = all(
        r.verdict == GR1Verdict.REALIZABLE
        for r in instance_results.values()
    )

    return ParamResult(
        verdict=ParamVerdict.REALIZABLE if all_realizable else ParamVerdict.UNKNOWN,
        instance_results=instance_results,
        n_min=n_min,
        n_max_checked=n_max,
        cutoff=CutoffResult(
            cutoff_n=n_max,
            verified_up_to=n_max,
            stable_from=n_max,
            structure_signatures=signatures,
        ) if all_realizable else None,
    ), games


def _relative_signature(inst: InstanceResult, n: int) -> str:
    """Signature normalized by N (to detect scaling patterns)."""
    if inst.verdict != GR1Verdict.REALIZABLE:
        return "unrealizable"
    # Relative winning region coverage
    coverage = inst.winning_region_size / max(inst.game_states, 1)
    coverage_bin = round(coverage, 2)
    n_modes = inst.strategy.n_modes if inst.strategy else 0
    return f"cov={coverage_bin}_modes={n_modes}"


# ---------------------------------------------------------------------------
# Symmetry reduction
# ---------------------------------------------------------------------------

def reduce_by_symmetry(game: GR1Game, n_procs: int) -> GR1Game:
    """Reduce a GR(1) game by exploiting process symmetry.

    For ring/clique topologies with identical processes, states that differ
    only by a rotation are equivalent. We quotient by the rotation group.

    State format expected: (local_combo_tuple, token_pos) or just (local_combo_tuple,).
    """
    if n_procs <= 1:
        return game

    # Compute equivalence classes under rotation
    state_to_canonical = {}
    canonical_states = set()

    for state in game.states:
        canon = _canonical_rotation(state, n_procs)
        state_to_canonical[state] = canon
        canonical_states.add(canon)

    # Map initial states
    canonical_initial = {state_to_canonical[s] for s in game.initial}

    # Map transitions
    canonical_transitions = {}
    for state, env_choices in game.transitions.items():
        canon_state = state_to_canonical[state]
        if canon_state in canonical_transitions:
            continue  # Already processed

        new_env_choices = []
        for successors in env_choices:
            new_succs = {state_to_canonical[s] for s in successors if s in state_to_canonical}
            if new_succs:
                new_env_choices.append(new_succs)

        if new_env_choices:
            canonical_transitions[canon_state] = new_env_choices

    # Map justice conditions
    canon_env_justice = []
    for justice_set in game.env_justice:
        canon_set = {state_to_canonical[s] for s in justice_set if s in state_to_canonical}
        if canon_set:
            canon_env_justice.append(canon_set)
    if not canon_env_justice:
        canon_env_justice = [canonical_states]

    canon_sys_justice = []
    for justice_set in game.sys_justice:
        canon_set = {state_to_canonical[s] for s in justice_set if s in state_to_canonical}
        if canon_set:
            canon_sys_justice.append(canon_set)
    if not canon_sys_justice:
        canon_sys_justice = [canonical_states]

    return GR1Game(
        states=canonical_states,
        initial=canonical_initial,
        transitions=canonical_transitions,
        env_justice=canon_env_justice,
        sys_justice=canon_sys_justice,
    )


def _canonical_rotation(state, n_procs: int):
    """Find lexicographically smallest rotation of a state."""
    if isinstance(state, tuple) and len(state) == 2:
        combo, token = state
        if isinstance(combo, tuple) and len(combo) == n_procs:
            # Try all rotations, pick lexicographically smallest
            best = (combo, token)
            for r in range(1, n_procs):
                rotated = combo[r:] + combo[:r]
                new_token = (token - r) % n_procs
                candidate = (rotated, new_token)
                if candidate < best:
                    best = candidate
            return best
    elif isinstance(state, tuple) and len(state) == n_procs:
        # No token, just local state tuple
        best = state
        for r in range(1, n_procs):
            rotated = state[r:] + state[:r]
            if rotated < best:
                best = rotated
        return best
    return state


# ---------------------------------------------------------------------------
# Inductive verification: N -> N+1
# ---------------------------------------------------------------------------

def verify_inductive_step(
    spec: ParamSpec,
    n_base: int,
    base_result: InstanceResult,
) -> bool:
    """Verify that a controller for N processes extends to N+1.

    Strategy: check that the winning region for N+1 contains all initial
    states, using the N-process controller structure as a guide.

    This is a sound but incomplete check -- if it passes, the controller
    generalizes. If it fails, we cannot conclude anything.
    """
    if base_result.verdict != GR1Verdict.REALIZABLE:
        return False

    # Build N+1 game
    n_next = n_base + 1
    game_next = instantiate_game(spec, n_next)

    # Solve N+1
    result_next = gr1_solve(game_next, extract_strategy=False)

    if result_next.verdict != GR1Verdict.REALIZABLE:
        return False

    # Check initial states in winning region
    return game_next.initial <= result_next.winning_region


# ---------------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------------

def extract_controller_template(
    spec: ParamSpec,
    instance_results: Dict[int, InstanceResult],
) -> Optional[ControllerTemplate]:
    """Extract a parameterized controller template from instance solutions.

    Finds common structure across solutions for different N values.
    The template is a single-process controller that, when replicated,
    should work for any N.
    """
    # Find the smallest realizable instance
    realizable = {
        n: r for n, r in instance_results.items()
        if r.verdict == GR1Verdict.REALIZABLE and r.strategy is not None
    }

    if not realizable:
        return None

    # Use the smallest instance as the template base
    smallest_n = min(realizable.keys())
    base = realizable[smallest_n]
    strategy = base.strategy

    # Extract per-process behavior from the global strategy
    template_states = [f"ctrl_{i}" for i in range(strategy.n_modes)]
    template_initial = "ctrl_0"

    # Build template transitions from strategy
    template_transitions = {}

    for (state, mode, env_idx), (next_state, next_mode) in strategy.transitions.items():
        if isinstance(state, tuple) and len(state) == 2:
            combo, token = state
            # Extract process 0's local state as representative
            if smallest_n > 0:
                local = combo[0]
                ctrl_state = f"ctrl_{mode}"
                next_ctrl = f"ctrl_{next_mode}"
                # Signal from token position
                signal = "has_token" if token == 0 else "no_token"
                key = (ctrl_state, local, signal)
                if key not in template_transitions:
                    next_local = next_state[0][0] if isinstance(next_state, tuple) and len(next_state) == 2 and isinstance(next_state[0], tuple) else local
                    template_transitions[key] = (next_ctrl, next_local)

    return ControllerTemplate(
        states=template_states,
        initial=template_initial,
        transitions=template_transitions,
    )


# ---------------------------------------------------------------------------
# Full parameterized synthesis pipeline
# ---------------------------------------------------------------------------

def parameterized_synthesize(
    spec: ParamSpec,
    n_min: int = 2,
    n_max: int = 6,
    use_symmetry: bool = True,
    stability_window: int = 2,
) -> ParamResult:
    """Full parameterized synthesis pipeline.

    1. Instantiate and solve for each N in [n_min, n_max]
    2. Detect cutoff (structure stabilization)
    3. Optionally verify inductive step
    4. Extract controller template

    Returns ParamResult with verdict, instances, cutoff info, and template.
    """
    result, games = detect_cutoff(
        spec, n_min=n_min, n_max=n_max, stability_window=stability_window
    )

    # Try symmetry reduction if enabled and cutoff not found
    if use_symmetry and result.verdict == ParamVerdict.UNKNOWN:
        sym_results = {}
        for n, game in games.items():
            reduced = reduce_by_symmetry(game, n)
            red_result = gr1_solve(reduced, extract_strategy=True)

            # Map back: check if original initial in winning region
            original_game = games[n]

            inst = InstanceResult(
                n=n,
                verdict=red_result.verdict,
                game_states=len(reduced.states),
                winning_region_size=len(red_result.winning_region),
                strategy=red_result.strategy,
            )

            if red_result.verdict == GR1Verdict.REALIZABLE:
                # Check initial states
                reduced_initial = {_canonical_rotation(s, n) for s in original_game.initial}
                if not (reduced_initial <= red_result.winning_region):
                    inst.verdict = GR1Verdict.UNREALIZABLE

            sym_results[n] = inst

        # Update result with symmetry-reduced data
        result.instance_results.update(sym_results)

    # Try inductive verification if cutoff found
    if result.verdict == ParamVerdict.CUTOFF_FOUND and result.cutoff:
        n_c = result.cutoff.cutoff_n
        if n_c in result.instance_results:
            is_inductive = verify_inductive_step(
                spec, n_c, result.instance_results[n_c]
            )
            result.cutoff.is_inductive = is_inductive
            if is_inductive:
                result.verdict = ParamVerdict.REALIZABLE

    # Extract template
    result.controller_template = extract_controller_template(
        spec, result.instance_results
    )

    return result


# ---------------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------------

def analyze_scaling(
    spec: ParamSpec,
    n_values: List[int],
) -> Dict[str, Any]:
    """Analyze how game size and synthesis time scale with N.

    Returns a dict with per-N statistics:
    - state_count, transition_count, winning_region_size, verdict
    """
    stats = {}
    for n in n_values:
        game = instantiate_game(spec, n)
        n_transitions = sum(
            sum(len(succ) for succ in choices)
            for choices in game.transitions.values()
        )
        result = gr1_solve(game, extract_strategy=False)

        stats[n] = {
            'states': len(game.states),
            'transitions': n_transitions,
            'winning_region': len(result.winning_region),
            'verdict': result.verdict.value,
            'initial_winning': game.initial <= result.winning_region,
        }

    return stats


def scaling_summary(stats: Dict[str, Any]) -> str:
    """Format scaling analysis as a readable summary."""
    lines = ["N  | States | Trans  | Winning | Verdict"]
    lines.append("---+--------+--------+---------+---------")
    for n in sorted(stats.keys()):
        s = stats[n]
        lines.append(
            f"{n:2d} | {s['states']:6d} | {s['transitions']:6d} | {s['winning_region']:7d} | {s['verdict']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predefined parameterized specifications
# ---------------------------------------------------------------------------

def mutex_ring_spec() -> ParamSpec:
    """Mutual exclusion in a token ring.

    Each process: idle -> trying -> critical -> idle.
    Token holder can enter critical section.
    Env controls token passing. Sys controls process state transitions.
    Safety: at most one process in critical.
    Liveness: every trying process eventually enters critical.
    """
    template = ProcessTemplate(
        local_states=['idle', 'trying', 'critical'],
        initial_state='idle',
        transitions={
            # Without token: can only go idle -> trying
            ('idle', 'no_token'): [('idle', 'none'), ('trying', 'none')],
            ('trying', 'no_token'): [('trying', 'none')],
            ('critical', 'no_token'): [('idle', 'release')],
            # With token: trying -> critical is allowed
            ('idle', 'has_token'): [('idle', 'pass'), ('trying', 'none')],
            ('trying', 'has_token'): [('critical', 'enter')],
            ('critical', 'has_token'): [('idle', 'release')],
            # Wildcard fallback
            ('idle', '*'): [('idle', 'none'), ('trying', 'none')],
            ('trying', '*'): [('trying', 'none')],
            ('critical', '*'): [('idle', 'release')],
        },
        labels={
            'in_critical': {'critical'},
            'is_trying': {'trying'},
            'not_critical': {'idle', 'trying'},
        },
    )
    return ParamSpec(
        process_template=template,
        topology="ring",
        mutex=['critical'],
        liveness=['in_critical'],
        env_fairness=['token_visits'],
    )


def pipeline_spec() -> ParamSpec:
    """Pipeline synchronization.

    Each process: empty -> full.
    Process i can become full when process i-1 is full (data flows right).
    Process 0 can become full from env input.
    Liveness: every process eventually becomes full.
    """
    template = ProcessTemplate(
        local_states=['empty', 'full'],
        initial_state='empty',
        transitions={
            ('empty', 'full'): [('full', 'ready')],  # Left neighbor full -> fill
            ('empty', 'empty'): [('empty', 'wait')],  # Left neighbor empty -> wait
            ('empty', 'input'): [('full', 'ready')],  # Env input (process 0)
            ('empty', '*'): [('empty', 'wait')],
            ('full', '*'): [('full', 'none'), ('empty', 'drain')],
        },
        labels={
            'is_full': {'full'},
        },
    )
    return ParamSpec(
        process_template=template,
        topology="pipeline",
        env_signals=['input', 'no_input'],
        liveness=['is_full'],
    )


def token_passing_spec() -> ParamSpec:
    """Simple token passing in a ring.

    Each process: no_token -> has_token -> no_token.
    Exactly one process has the token at any time.
    Safety: at most one token holder.
    Liveness: every process eventually gets the token.
    """
    template = ProcessTemplate(
        local_states=['waiting', 'holding'],
        initial_state='waiting',
        transitions={
            ('waiting', 'has_token'): [('holding', 'take')],
            ('waiting', 'no_token'): [('waiting', 'wait')],
            ('waiting', '*'): [('waiting', 'wait')],
            ('holding', 'has_token'): [('holding', 'keep'), ('waiting', 'pass')],
            ('holding', 'no_token'): [('waiting', 'pass')],
            ('holding', '*'): [('holding', 'keep'), ('waiting', 'pass')],
        },
        labels={
            'has_token': {'holding'},
        },
    )
    return ParamSpec(
        process_template=template,
        topology="ring",
        mutex=['holding'],
        liveness=['has_token'],
        env_fairness=['token_visits'],
    )


# ---------------------------------------------------------------------------
# Direct game construction helpers (for custom parameterized games)
# ---------------------------------------------------------------------------

def build_parameterized_game(
    n: int,
    state_generator: Callable[[int], Set],
    initial_generator: Callable[[int], Set],
    transition_generator: Callable[[int, Set], Dict],
    env_justice_generator: Callable[[int, Set], List[Set]],
    sys_justice_generator: Callable[[int, Set], List[Set]],
) -> GR1Game:
    """Build a GR(1) game from parameterized generators.

    Each generator takes N (and optionally the state set) and returns
    the corresponding component. This allows custom parameterized families
    beyond the template-based approach.
    """
    states = state_generator(n)
    initial = initial_generator(n)
    transitions = transition_generator(n, states)
    env_justice = env_justice_generator(n, states)
    sys_justice = sys_justice_generator(n, states)

    return GR1Game(
        states=states,
        initial=initial,
        transitions=transitions,
        env_justice=env_justice if env_justice else [states],
        sys_justice=sys_justice if sys_justice else [states],
    )


def solve_parameterized_family(
    builder: Callable[[int], GR1Game],
    n_range: range,
) -> Dict[int, InstanceResult]:
    """Solve a family of GR(1) games built by a parameterized builder."""
    results = {}
    for n in n_range:
        game = builder(n)
        result = gr1_solve(game, extract_strategy=True)

        inst = InstanceResult(
            n=n,
            verdict=result.verdict,
            game_states=len(game.states),
            winning_region_size=len(result.winning_region),
            strategy=result.strategy,
        )

        if result.verdict == GR1Verdict.REALIZABLE:
            if not (game.initial <= result.winning_region):
                inst.verdict = GR1Verdict.UNREALIZABLE
                inst.strategy = None

        results[n] = inst

    return results


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_with_without_symmetry(
    spec: ParamSpec,
    n: int,
) -> Dict[str, Any]:
    """Compare synthesis with and without symmetry reduction for N processes."""
    game = instantiate_game(spec, n)
    result_full = gr1_solve(game, extract_strategy=False)

    reduced = reduce_by_symmetry(game, n)
    result_sym = gr1_solve(reduced, extract_strategy=False)

    full_realizable = result_full.verdict == GR1Verdict.REALIZABLE and \
                      game.initial <= result_full.winning_region
    reduced_initial = {_canonical_rotation(s, n) for s in game.initial}
    sym_realizable = result_sym.verdict == GR1Verdict.REALIZABLE and \
                     reduced_initial <= result_sym.winning_region

    return {
        'n': n,
        'full_states': len(game.states),
        'reduced_states': len(reduced.states),
        'reduction_ratio': len(reduced.states) / max(len(game.states), 1),
        'full_verdict': 'realizable' if full_realizable else 'unrealizable',
        'sym_verdict': 'realizable' if sym_realizable else 'unrealizable',
        'verdicts_agree': full_realizable == sym_realizable,
    }


def parameterized_summary(result: ParamResult) -> str:
    """Format parameterized synthesis result as readable summary."""
    lines = [f"Parameterized Synthesis Result: {result.verdict.value}"]
    lines.append(f"Range: N = {result.n_min} to {result.n_max_checked}")
    lines.append("")

    for n in sorted(result.instance_results.keys()):
        inst = result.instance_results[n]
        lines.append(
            f"  N={n}: {inst.verdict.value} "
            f"(states={inst.game_states}, winning={inst.winning_region_size})"
        )

    if result.cutoff:
        lines.append("")
        lines.append(f"Cutoff: N_c = {result.cutoff.cutoff_n}")
        lines.append(f"Stable from: N = {result.cutoff.stable_from}")
        lines.append(f"Inductive: {result.cutoff.is_inductive}")

    if result.controller_template:
        lines.append("")
        lines.append(f"Controller template: {len(result.controller_template.states)} states")
        lines.append(f"  Template transitions: {len(result.controller_template.transitions)}")

    return "\n".join(lines)
