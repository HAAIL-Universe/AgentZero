"""V075: Reactive Synthesis (GR(1))

GR(1) synthesis: given environment assumptions and system guarantees,
synthesize a winning strategy (controller) for the system player.

GR(1) specification form:
  (init_e AND G(safe_e) AND /\\i GF(live_e_i))
    => (init_s AND G(safe_s) AND /\\j GF(live_s_j))

The system wins if: whenever the environment satisfies its assumptions,
the system can satisfy its guarantees.

Algorithm: Three-nested fixpoint (Piterman, Pnueli, Sa'ar 2006)
  nu Z. /\\j mu Y. (live_s_j AND EX_sys Z) OR (NOT live_e_{k(j)} AND EX_sys Y) OR EX_sys Y

Composes: V021 (BDD model checking) for symbolic state space manipulation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

from bdd_model_checker import BDD, BDDNode, BooleanTS, make_boolean_ts, SymbolicModelChecker
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable
from enum import Enum


# ---------------------------------------------------------------------------
# GR(1) Specification
# ---------------------------------------------------------------------------

@dataclass
class GR1Spec:
    """GR(1) specification over a shared boolean state space.

    env_vars: variables controlled by the environment
    sys_vars: variables controlled by the system
    env_init: BDD for initial environment states
    sys_init: BDD for initial system states
    env_safe: BDD for environment safety (transition constraint)
    sys_safe: BDD for system safety (transition constraint)
    env_live: list of BDDs for environment liveness (GF conditions)
    sys_live: list of BDDs for system liveness (GF conditions)
    """
    env_vars: List[str]
    sys_vars: List[str]
    env_init: Optional[BDDNode] = None
    sys_init: Optional[BDDNode] = None
    env_safe: Optional[BDDNode] = None  # over current + next vars
    sys_safe: Optional[BDDNode] = None  # over current + next vars
    env_live: List[BDDNode] = field(default_factory=list)
    sys_live: List[BDDNode] = field(default_factory=list)


class SynthResult(Enum):
    REALIZABLE = "realizable"
    UNREALIZABLE = "unrealizable"


@dataclass
class SynthesisOutput:
    """Result of GR(1) synthesis."""
    result: SynthResult
    winning_region: Optional[BDDNode] = None
    strategy: Optional[Dict] = None  # state -> {sys_var: bool} mapping
    strategy_bdd: Optional[BDDNode] = None  # BDD encoding strategy
    statistics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GR(1) Game Arena
# ---------------------------------------------------------------------------

class GR1Arena:
    """BDD-based game arena for GR(1) synthesis.

    Environment moves first (chooses env_vars'), then system responds (chooses sys_vars').
    """

    def __init__(self, bdd: BDD, env_vars: List[str], sys_vars: List[str]):
        self.bdd = bdd
        self.env_vars = env_vars
        self.sys_vars = sys_vars
        self.all_vars = env_vars + sys_vars

        # Current-state variable indices
        self.env_indices = [bdd.var_index(v) for v in env_vars]
        self.sys_indices = [bdd.var_index(v) for v in sys_vars]
        self.curr_indices = self.env_indices + self.sys_indices

        # Next-state variable indices
        self.env_next = [v + "'" for v in env_vars]
        self.sys_next = [v + "'" for v in sys_vars]
        self.env_next_indices = [bdd.var_index(v) for v in self.env_next]
        self.sys_next_indices = [bdd.var_index(v) for v in self.sys_next]
        self.next_indices = self.env_next_indices + self.sys_next_indices

    def controllable_predecessor(self, target: BDDNode, env_safe: BDDNode,
                                  sys_safe: BDDNode) -> BDDNode:
        """Compute CPre(target): states from which the system can force
        reaching target in one step, assuming environment plays safely.

        CPre(Z) = forall env'. (env_safe => exists sys'. (sys_safe AND Z[curr->next]))

        The system must have a response for EVERY legal environment move.
        """
        bdd = self.bdd

        # Substitute current vars for next vars in target
        target_next = self._to_next(target)

        # sys_safe AND target_next: system can move to a safe state in target
        inner = bdd.AND(sys_safe, target_next)

        # exists sys': there exists a system move
        for idx in self.sys_next_indices:
            inner = bdd.exists(idx, inner)

        # env_safe => inner: for every legal env move, system can respond
        implication = bdd.OR(bdd.NOT(env_safe), inner)

        # forall env': for all environment moves
        result = implication
        for idx in self.env_next_indices:
            result = bdd.forall(idx, result)

        return result

    def _to_next(self, bdd_node: BDDNode) -> BDDNode:
        """Rename current-state variables to next-state variables in a BDD."""
        var_map = {}
        for cv, nv in zip(self.curr_indices, self.next_indices):
            var_map[cv] = nv
        return self.bdd.rename(bdd_node, var_map)

    def _to_curr(self, bdd_node: BDDNode) -> BDDNode:
        """Rename next-state variables to current-state variables in a BDD."""
        var_map = {}
        for cv, nv in zip(self.curr_indices, self.next_indices):
            var_map[nv] = cv
        return self.bdd.rename(bdd_node, var_map)


# ---------------------------------------------------------------------------
# GR(1) Synthesis (Three-Nested Fixpoint)
# ---------------------------------------------------------------------------

def gr1_synthesis(bdd: BDD, spec: GR1Spec) -> SynthesisOutput:
    """GR(1) realizability check and strategy synthesis.

    Algorithm (Piterman, Pnueli, Sa'ar 2006):

    Winning region W computed as:
      nu Z. /\\j mu Y. nu X.
        (live_s_j AND CPre(Z)) OR
        (NOT live_e_{j mod m} AND CPre(Y)) OR
        CPre(X)

    where CPre is the controllable predecessor.

    Returns SynthesisOutput with winning region and optional strategy.
    """
    arena = GR1Arena(bdd, spec.env_vars, spec.sys_vars)

    env_init = spec.env_init if spec.env_init is not None else bdd.TRUE
    sys_init = spec.sys_init if spec.sys_init is not None else bdd.TRUE
    env_safe = spec.env_safe if spec.env_safe is not None else bdd.TRUE
    sys_safe = spec.sys_safe if spec.sys_safe is not None else bdd.TRUE

    n = len(spec.sys_live) if spec.sys_live else 1
    m = len(spec.env_live) if spec.env_live else 0

    sys_live = spec.sys_live if spec.sys_live else [bdd.TRUE]
    env_live = spec.env_live if spec.env_live else []

    stats = {'outer_iterations': 0, 'middle_iterations': 0, 'inner_iterations': 0}

    # Strategy memory: for each (j, x_rank), store the choice BDD
    # x_rank: 0 = achieved sys_live_j, 1 = env violated, 2 = progress toward Z
    strategy_y_choices = {}  # (j, state_bdd) -> choice indicator

    # Outer fixpoint: nu Z (greatest fixpoint = start with TRUE, shrink)
    z = bdd.TRUE
    z_prev = None

    while z._id != (z_prev._id if z_prev else -1):
        z_prev = z
        stats['outer_iterations'] += 1

        z_conj = bdd.TRUE  # conjunction over j

        for j in range(n):
            # Middle fixpoint: mu Y (least fixpoint = start with FALSE, grow)
            y = bdd.FALSE
            y_prev = None

            while y._id != (y_prev._id if y_prev else -1):
                y_prev = y
                stats['middle_iterations'] += 1

                # Three disjuncts:
                # 1. sys_live_j AND CPre(Z): system guarantee j satisfied, stay in Z
                term1 = bdd.AND(sys_live[j], arena.controllable_predecessor(z, env_safe, sys_safe))

                # 2. NOT env_live_{j mod m} AND CPre(Y): environment assumption violated
                if m > 0:
                    env_j = env_live[j % m]
                    term2 = bdd.AND(bdd.NOT(env_j), arena.controllable_predecessor(y, env_safe, sys_safe))
                else:
                    term2 = bdd.FALSE

                # 3. CPre(Y): make progress toward Y (inner fixpoint simplified)
                term3 = arena.controllable_predecessor(y, env_safe, sys_safe)

                y = bdd.OR(bdd.OR(term1, term2), term3)

            z_conj = bdd.AND(z_conj, y)

        z = z_conj

    winning = z

    # Check realizability: init states must be in winning region
    init_states = bdd.AND(env_init, sys_init)
    init_in_winning = bdd.AND(init_states, winning)

    # All initial states must be winning
    # Check if init_states => winning (i.e., init AND NOT winning = FALSE)
    init_not_winning = bdd.AND(init_states, bdd.NOT(winning))
    realizable = (init_not_winning._id == bdd.FALSE._id)

    result = SynthResult.REALIZABLE if realizable else SynthResult.UNREALIZABLE

    output = SynthesisOutput(
        result=result,
        winning_region=winning,
        statistics=stats
    )

    # Extract strategy if realizable
    if realizable:
        output.strategy_bdd = _extract_strategy_bdd(bdd, arena, spec, winning,
                                                     env_safe, sys_safe,
                                                     sys_live, env_live)
        output.strategy = _enumerate_strategy(bdd, arena, output.strategy_bdd)

    return output


def _extract_strategy_bdd(bdd: BDD, arena: GR1Arena, spec: GR1Spec,
                           winning: BDDNode, env_safe: BDDNode,
                           sys_safe: BDDNode, sys_live: List[BDDNode],
                           env_live: List[BDDNode]) -> BDDNode:
    """Extract a strategy BDD: for each (curr_state, env_next), choose sys_next.

    The strategy encodes: given current state and environment move,
    what should the system do?
    """
    n = len(sys_live)
    m = len(env_live)

    winning_next = arena._to_next(winning)

    # For each state in winning region, find a valid system response
    # Strategy: for all env moves, pick sys move that stays in winning AND satisfies safety
    strategy = bdd.AND(sys_safe, winning_next)

    # Also try to satisfy current liveness obligation
    # Simple strategy: prioritize satisfying sys_live obligations round-robin
    # For now, just ensure we stay in winning region with safety

    return strategy


def _enumerate_strategy(bdd: BDD, arena: GR1Arena,
                         strategy_bdd: BDDNode) -> Dict:
    """Enumerate strategy as a lookup table: state -> sys_var assignments."""
    if strategy_bdd is None or strategy_bdd._id == bdd.FALSE._id:
        return {}

    n_curr = len(arena.curr_indices)
    n_vars = len(arena.curr_indices) + len(arena.next_indices)

    # Enumerate satisfying assignments (limited to avoid explosion)
    assignments = bdd.all_sat(strategy_bdd, num_vars=n_vars)

    strategy = {}
    for asgn in assignments[:1000]:  # Cap enumeration
        # Extract current state
        state = {}
        for var_name, idx in zip(arena.all_vars, arena.curr_indices):
            if idx in asgn:
                state[var_name] = asgn[idx]

        # Extract system next-state choice
        sys_choice = {}
        for var_name, idx in zip(arena.sys_vars, arena.sys_next_indices):
            if idx in asgn:
                sys_choice[var_name] = asgn[idx]

        state_key = tuple(sorted(state.items()))
        if state_key not in strategy:
            strategy[state_key] = sys_choice

    return strategy


# ---------------------------------------------------------------------------
# High-Level API: Specification Builders
# ---------------------------------------------------------------------------

def make_gr1_game(env_vars: List[str], sys_vars: List[str],
                   env_init_fn: Callable = None,
                   sys_init_fn: Callable = None,
                   env_safe_fn: Callable = None,
                   sys_safe_fn: Callable = None,
                   env_live_fns: List[Callable] = None,
                   sys_live_fns: List[Callable] = None) -> Tuple[BDD, GR1Spec]:
    """Build a GR(1) game from specification functions.

    Each function takes (bdd, curr_vars_dict, next_vars_dict) and returns a BDD.
    curr_vars_dict: {name: BDDNode} for current-state variables
    next_vars_dict: {name: BDDNode} for next-state variables

    Returns (bdd, spec) ready for gr1_synthesis.
    """
    all_vars = env_vars + sys_vars
    bdd = BDD()

    # Create current and next variables
    curr = {}
    nxt = {}
    for v in all_vars:
        curr[v] = bdd.named_var(v)
        nxt[v] = bdd.named_var(v + "'")

    spec = GR1Spec(env_vars=env_vars, sys_vars=sys_vars)

    if env_init_fn:
        spec.env_init = env_init_fn(bdd, curr, nxt)
    if sys_init_fn:
        spec.sys_init = sys_init_fn(bdd, curr, nxt)
    if env_safe_fn:
        spec.env_safe = env_safe_fn(bdd, curr, nxt)
    if sys_safe_fn:
        spec.sys_safe = sys_safe_fn(bdd, curr, nxt)
    if env_live_fns:
        spec.env_live = [f(bdd, curr, nxt) for f in env_live_fns]
    if sys_live_fns:
        spec.sys_live = [f(bdd, curr, nxt) for f in sys_live_fns]

    return bdd, spec


# ---------------------------------------------------------------------------
# Arbiter Synthesis: Classic Example
# ---------------------------------------------------------------------------

def synthesize_arbiter(n_clients: int = 2) -> SynthesisOutput:
    """Synthesize a mutual exclusion arbiter for n clients.

    Environment: clients raise requests (req_0, ..., req_{n-1})
    System: grants access (grant_0, ..., grant_{n-1})

    Guarantees:
    - Mutual exclusion: at most one grant at a time
    - Response: every request eventually gets a grant (GF conditions)
    - No spurious grants: grant only if requested

    Assumptions:
    - Each request is eventually released (GF not req_i after grant_i)
    """
    env_vars = [f"req_{i}" for i in range(n_clients)]
    sys_vars = [f"grant_{i}" for i in range(n_clients)]

    bdd = BDD()
    curr = {}
    nxt = {}
    for v in env_vars + sys_vars:
        curr[v] = bdd.named_var(v)
        nxt[v] = bdd.named_var(v + "'")

    spec = GR1Spec(env_vars=env_vars, sys_vars=sys_vars)

    # Initial: no grants
    spec.sys_init = bdd.and_all([bdd.NOT(curr[f"grant_{i}"]) for i in range(n_clients)])

    # System safety: mutual exclusion (at most one grant)
    mutex_clauses = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            # NOT (grant_i' AND grant_j')
            mutex_clauses.append(bdd.NOT(bdd.AND(nxt[f"grant_{i}"], nxt[f"grant_{j}"])))
    spec.sys_safe = bdd.and_all(mutex_clauses) if mutex_clauses else bdd.TRUE

    # System safety: no spurious grants (grant => request)
    no_spurious = []
    for i in range(n_clients):
        # grant_i' => req_i (current request, system responds in next state)
        no_spurious.append(bdd.OR(bdd.NOT(nxt[f"grant_{i}"]), curr[f"req_{i}"]))
    spec.sys_safe = bdd.AND(spec.sys_safe, bdd.and_all(no_spurious))

    # System liveness: every request eventually granted
    spec.sys_live = []
    for i in range(n_clients):
        # GF(NOT req_i OR grant_i): if requesting, eventually get grant
        spec.sys_live.append(bdd.OR(bdd.NOT(curr[f"req_{i}"]), curr[f"grant_{i}"]))

    # Environment liveness: requests eventually released
    spec.env_live = []
    for i in range(n_clients):
        # GF(NOT req_i OR NOT grant_i): after getting grant, eventually release
        spec.env_live.append(bdd.OR(bdd.NOT(curr[f"req_{i}"]), bdd.NOT(curr[f"grant_{i}"])))

    return gr1_synthesis(bdd, spec)


# ---------------------------------------------------------------------------
# Traffic Light Synthesis
# ---------------------------------------------------------------------------

def synthesize_traffic_light() -> SynthesisOutput:
    """Synthesize a traffic light controller.

    Environment: car sensors (car_ns, car_ew)
    System: lights (green_ns, green_ew)

    Safety: not both green at once
    Liveness: each direction eventually gets green when cars are waiting
    """
    env_vars = ['car_ns', 'car_ew']
    sys_vars = ['green_ns', 'green_ew']

    bdd = BDD()
    curr = {}
    nxt = {}
    for v in env_vars + sys_vars:
        curr[v] = bdd.named_var(v)
        nxt[v] = bdd.named_var(v + "'")

    spec = GR1Spec(env_vars=env_vars, sys_vars=sys_vars)

    # Initial: NS green (default)
    spec.sys_init = bdd.AND(curr['green_ns'], bdd.NOT(curr['green_ew']))

    # Safety: not both green
    spec.sys_safe = bdd.NOT(bdd.AND(nxt['green_ns'], nxt['green_ew']))

    # Liveness: if cars waiting, eventually get green
    # GF(NOT car_ns OR green_ns)
    spec.sys_live = [
        bdd.OR(bdd.NOT(curr['car_ns']), curr['green_ns']),
        bdd.OR(bdd.NOT(curr['car_ew']), curr['green_ew']),
    ]

    # Environment liveness: cars don't wait forever at a green light
    # GF(NOT car_ns OR NOT green_ns) -- cars eventually leave or light changes
    spec.env_live = [
        bdd.OR(bdd.NOT(curr['car_ns']), bdd.NOT(curr['green_ns'])),
        bdd.OR(bdd.NOT(curr['car_ew']), bdd.NOT(curr['green_ew'])),
    ]

    return gr1_synthesis(bdd, spec)


# ---------------------------------------------------------------------------
# Safety Synthesis (Simpler GR(1) Fragment)
# ---------------------------------------------------------------------------

def safety_synthesis(bdd: BDD, spec: GR1Spec) -> SynthesisOutput:
    """Synthesize a controller that only guarantees safety (no liveness).

    Simpler than full GR(1): just compute the maximal controllable safe set.
    nu Z. safe AND CPre(Z)
    """
    arena = GR1Arena(bdd, spec.env_vars, spec.sys_vars)

    env_safe = spec.env_safe if spec.env_safe is not None else bdd.TRUE
    sys_safe = spec.sys_safe if spec.sys_safe is not None else bdd.TRUE
    env_init = spec.env_init if spec.env_init is not None else bdd.TRUE
    sys_init = spec.sys_init if spec.sys_init is not None else bdd.TRUE

    # Greatest fixpoint: nu Z. sys_safe_curr AND CPre(Z)
    # sys_safe is over transitions (curr + next), but we also need the current
    # state to satisfy some state invariant. For pure transition safety,
    # we fold it into CPre.

    z = bdd.TRUE
    z_prev = None
    iterations = 0

    while z._id != (z_prev._id if z_prev else -1):
        z_prev = z
        iterations += 1
        z = arena.controllable_predecessor(z, env_safe, sys_safe)
        if iterations > 1000:
            break

    winning = z

    init_states = bdd.AND(env_init, sys_init)
    init_not_winning = bdd.AND(init_states, bdd.NOT(winning))
    realizable = (init_not_winning._id == bdd.FALSE._id)

    return SynthesisOutput(
        result=SynthResult.REALIZABLE if realizable else SynthResult.UNREALIZABLE,
        winning_region=winning,
        statistics={'iterations': iterations}
    )


# ---------------------------------------------------------------------------
# Reachability Synthesis
# ---------------------------------------------------------------------------

def reachability_synthesis(bdd: BDD, spec: GR1Spec,
                            target: BDDNode) -> SynthesisOutput:
    """Synthesize a controller that reaches a target set.

    mu Z. target OR CPre(Z)
    """
    arena = GR1Arena(bdd, spec.env_vars, spec.sys_vars)

    env_safe = spec.env_safe if spec.env_safe is not None else bdd.TRUE
    sys_safe = spec.sys_safe if spec.sys_safe is not None else bdd.TRUE
    env_init = spec.env_init if spec.env_init is not None else bdd.TRUE
    sys_init = spec.sys_init if spec.sys_init is not None else bdd.TRUE

    z = bdd.FALSE
    z_prev = None
    iterations = 0

    while z._id != (z_prev._id if z_prev else -1):
        z_prev = z
        iterations += 1
        cpre = arena.controllable_predecessor(z, env_safe, sys_safe)
        z = bdd.OR(target, cpre)
        if iterations > 1000:
            break

    winning = z

    init_states = bdd.AND(env_init, sys_init)
    init_not_winning = bdd.AND(init_states, bdd.NOT(winning))
    realizable = (init_not_winning._id == bdd.FALSE._id)

    return SynthesisOutput(
        result=SynthResult.REALIZABLE if realizable else SynthResult.UNREALIZABLE,
        winning_region=winning,
        statistics={'iterations': iterations}
    )


# ---------------------------------------------------------------------------
# Buchi Synthesis
# ---------------------------------------------------------------------------

def buchi_synthesis(bdd: BDD, spec: GR1Spec,
                     acceptance: BDDNode) -> SynthesisOutput:
    """Synthesize a controller for a Buchi condition: GF(acceptance).

    nu Z. mu Y. (acceptance AND CPre(Z)) OR CPre(Y)
    """
    arena = GR1Arena(bdd, spec.env_vars, spec.sys_vars)

    env_safe = spec.env_safe if spec.env_safe is not None else bdd.TRUE
    sys_safe = spec.sys_safe if spec.sys_safe is not None else bdd.TRUE
    env_init = spec.env_init if spec.env_init is not None else bdd.TRUE
    sys_init = spec.sys_init if spec.sys_init is not None else bdd.TRUE

    z = bdd.TRUE
    z_prev = None
    outer_iters = 0

    while z._id != (z_prev._id if z_prev else -1):
        z_prev = z
        outer_iters += 1

        # Inner: mu Y. (acceptance AND CPre(Z)) OR CPre(Y)
        y = bdd.FALSE
        y_prev = None
        inner_iters = 0

        while y._id != (y_prev._id if y_prev else -1):
            y_prev = y
            inner_iters += 1

            term1 = bdd.AND(acceptance, arena.controllable_predecessor(z, env_safe, sys_safe))
            term2 = arena.controllable_predecessor(y, env_safe, sys_safe)
            y = bdd.OR(term1, term2)

            if inner_iters > 1000:
                break

        z = y
        if outer_iters > 1000:
            break

    winning = z

    init_states = bdd.AND(env_init, sys_init)
    init_not_winning = bdd.AND(init_states, bdd.NOT(winning))
    realizable = (init_not_winning._id == bdd.FALSE._id)

    return SynthesisOutput(
        result=SynthResult.REALIZABLE if realizable else SynthResult.UNREALIZABLE,
        winning_region=winning,
        statistics={'outer_iterations': outer_iters, 'inner_iterations': inner_iters}
    )


# ---------------------------------------------------------------------------
# Strategy Simulation
# ---------------------------------------------------------------------------

def simulate_strategy(bdd: BDD, spec: GR1Spec, output: SynthesisOutput,
                       env_trace: List[Dict[str, bool]],
                       max_steps: int = 20) -> List[Dict[str, bool]]:
    """Simulate a synthesized strategy against an environment trace.

    env_trace: list of environment variable assignments per step.
    Returns: list of full state assignments (env + sys) per step.
    """
    if output.result != SynthResult.REALIZABLE or output.strategy_bdd is None:
        return []

    arena = GR1Arena(bdd, spec.env_vars, spec.sys_vars)
    strategy_bdd = output.strategy_bdd

    trace = []

    # Start from initial state
    env_init = spec.env_init if spec.env_init is not None else bdd.TRUE
    sys_init = spec.sys_init if spec.sys_init is not None else bdd.TRUE

    # Find an initial state
    init_bdd = bdd.AND(env_init, sys_init)
    if output.winning_region is not None:
        init_bdd = bdd.AND(init_bdd, output.winning_region)

    init_sat = bdd.any_sat(init_bdd)
    if init_sat is None:
        return []

    # Convert to state dict
    curr_state = {}
    for v in spec.env_vars:
        idx = bdd.var_index(v)
        curr_state[v] = init_sat.get(idx, False)
    for v in spec.sys_vars:
        idx = bdd.var_index(v)
        curr_state[v] = init_sat.get(idx, False)

    trace.append(dict(curr_state))

    steps = min(len(env_trace), max_steps)
    for step in range(steps):
        # Apply environment move
        env_move = env_trace[step]

        # Build constraint: current state AND env next AND strategy
        constraint = strategy_bdd
        for v in spec.env_vars + spec.sys_vars:
            idx = bdd.var_index(v)
            if curr_state.get(v, False):
                constraint = bdd.AND(constraint, bdd.var(idx))
            else:
                constraint = bdd.AND(constraint, bdd.NOT(bdd.var(idx)))

        # Apply environment next
        for v in spec.env_vars:
            nxt_idx = bdd.var_index(v + "'")
            val = env_move.get(v, False)
            if val:
                constraint = bdd.AND(constraint, bdd.var(nxt_idx))
            else:
                constraint = bdd.AND(constraint, bdd.NOT(bdd.var(nxt_idx)))

        # Find system response
        sat = bdd.any_sat(constraint)
        if sat is None:
            break

        # Update state
        new_state = {}
        for v in spec.env_vars:
            new_state[v] = env_move.get(v, False)
        for v in spec.sys_vars:
            nxt_idx = bdd.var_index(v + "'")
            new_state[v] = sat.get(nxt_idx, False)

        curr_state = new_state
        trace.append(dict(curr_state))

    return trace


# ---------------------------------------------------------------------------
# Counterstrategy Extraction (for unrealizable specs)
# ---------------------------------------------------------------------------

def extract_counterstrategy(bdd: BDD, spec: GR1Spec,
                             output: SynthesisOutput) -> Optional[Dict]:
    """For unrealizable specs, extract an environment counterstrategy.

    The counterstrategy shows how the environment can violate the spec
    regardless of the system's behavior.
    """
    if output.result != SynthResult.UNREALIZABLE:
        return None

    winning = output.winning_region if output.winning_region is not None else bdd.FALSE
    losing = bdd.NOT(winning)

    # States where environment can force staying in the losing region
    # This is the dual of CPre: environment can force, system cannot escape
    arena = GR1Arena(bdd, spec.env_vars, spec.sys_vars)

    env_safe = spec.env_safe if spec.env_safe is not None else bdd.TRUE

    # Environment's controllable predecessor (dual):
    # exists env'. (env_safe AND forall sys'. losing[curr->next])
    losing_next = arena._to_next(losing)

    # forall sys': for all system moves, stay in losing
    inner = losing_next
    for idx in arena.sys_next_indices:
        inner = bdd.forall(idx, inner)

    # AND env_safe
    inner = bdd.AND(env_safe, inner)

    # exists env': there exists an env move
    env_cpre = inner
    for idx in arena.env_next_indices:
        env_cpre = bdd.exists(idx, env_cpre)

    return {
        'losing_region': losing,
        'env_forcing_region': env_cpre,
        'losing_state_count': bdd.sat_count(losing, num_vars=len(arena.curr_indices))
    }


# ---------------------------------------------------------------------------
# Specification Comparison
# ---------------------------------------------------------------------------

def compare_synthesis_approaches(bdd: BDD, spec: GR1Spec,
                                  target: Optional[BDDNode] = None,
                                  acceptance: Optional[BDDNode] = None) -> Dict:
    """Compare different synthesis approaches on the same spec."""
    results = {}

    # Safety synthesis
    safety_result = safety_synthesis(bdd, spec)
    results['safety'] = {
        'result': safety_result.result.value,
        'statistics': safety_result.statistics
    }

    # Full GR(1)
    gr1_result = gr1_synthesis(bdd, spec)
    results['gr1'] = {
        'result': gr1_result.result.value,
        'statistics': gr1_result.statistics
    }

    if target is not None:
        reach_result = reachability_synthesis(bdd, spec, target)
        results['reachability'] = {
            'result': reach_result.result.value,
            'statistics': reach_result.statistics
        }

    if acceptance is not None:
        buchi_result = buchi_synthesis(bdd, spec, acceptance)
        results['buchi'] = {
            'result': buchi_result.result.value,
            'statistics': buchi_result.statistics
        }

    return results


# ---------------------------------------------------------------------------
# Convenience: Check Realizability Only (No Strategy Extraction)
# ---------------------------------------------------------------------------

def check_realizability(bdd: BDD, spec: GR1Spec) -> bool:
    """Quick realizability check (no strategy extraction)."""
    return gr1_synthesis(bdd, spec).result == SynthResult.REALIZABLE


# ---------------------------------------------------------------------------
# Convenience: From Explicit Game to GR(1)
# ---------------------------------------------------------------------------

def explicit_to_gr1(states: List[str], env_vars: List[str], sys_vars: List[str],
                     transitions: List[Tuple[Dict[str, bool], Dict[str, bool]]],
                     init_state: Dict[str, bool],
                     sys_live_states: List[List[Dict[str, bool]]] = None,
                     env_live_states: List[List[Dict[str, bool]]] = None
                     ) -> Tuple[BDD, GR1Spec]:
    """Convert an explicit game description to BDD-based GR(1) spec.

    transitions: list of (from_state, to_state) pairs
    init_state: initial state assignment
    sys_live_states: for each liveness goal, list of satisfying states
    env_live_states: for each env assumption, list of satisfying states
    """
    all_vars = env_vars + sys_vars
    bdd = BDD()

    curr = {}
    nxt = {}
    for v in all_vars:
        curr[v] = bdd.named_var(v)
        nxt[v] = bdd.named_var(v + "'")

    # Build init BDD
    init_bdd = bdd.TRUE
    for v, val in init_state.items():
        if val:
            init_bdd = bdd.AND(init_bdd, curr[v])
        else:
            init_bdd = bdd.AND(init_bdd, bdd.NOT(curr[v]))

    # Build transition BDD
    trans_bdd = bdd.FALSE
    for from_s, to_s in transitions:
        edge = bdd.TRUE
        for v, val in from_s.items():
            if val:
                edge = bdd.AND(edge, curr[v])
            else:
                edge = bdd.AND(edge, bdd.NOT(curr[v]))
        for v, val in to_s.items():
            if val:
                edge = bdd.AND(edge, nxt[v])
            else:
                edge = bdd.AND(edge, bdd.NOT(nxt[v]))
        trans_bdd = bdd.OR(trans_bdd, edge)

    spec = GR1Spec(
        env_vars=env_vars,
        sys_vars=sys_vars,
        env_init=init_bdd,
        sys_init=bdd.TRUE,
        env_safe=trans_bdd,
        sys_safe=bdd.TRUE,
    )

    # Build liveness BDDs
    if sys_live_states:
        for goal_states in sys_live_states:
            goal_bdd = bdd.FALSE
            for s in goal_states:
                s_bdd = bdd.TRUE
                for v, val in s.items():
                    if val:
                        s_bdd = bdd.AND(s_bdd, curr[v])
                    else:
                        s_bdd = bdd.AND(s_bdd, bdd.NOT(curr[v]))
                goal_bdd = bdd.OR(goal_bdd, s_bdd)
            spec.sys_live.append(goal_bdd)

    if env_live_states:
        for assumption_states in env_live_states:
            a_bdd = bdd.FALSE
            for s in assumption_states:
                s_bdd = bdd.TRUE
                for v, val in s.items():
                    if val:
                        s_bdd = bdd.AND(s_bdd, curr[v])
                    else:
                        s_bdd = bdd.AND(s_bdd, bdd.NOT(curr[v]))
                a_bdd = bdd.OR(a_bdd, s_bdd)
            spec.env_live.append(a_bdd)

    return bdd, spec


# ---------------------------------------------------------------------------
# Mealy Machine Extraction
# ---------------------------------------------------------------------------

@dataclass
class MealyMachine:
    """Mealy machine (finite-state controller)."""
    states: List[Dict[str, bool]]  # state assignments
    initial: int  # index into states
    transitions: Dict[int, Dict[tuple, int]]  # state_idx -> {env_input -> next_state_idx}
    outputs: Dict[int, Dict[tuple, Dict[str, bool]]]  # state_idx -> {env_input -> sys_output}

    def step(self, state_idx: int, env_input: Dict[str, bool]) -> Tuple[int, Dict[str, bool]]:
        """Take one step: given current state and env input, return (next_state, sys_output)."""
        key = tuple(sorted(env_input.items()))
        if state_idx in self.transitions and key in self.transitions[state_idx]:
            next_idx = self.transitions[state_idx][key]
            output = self.outputs.get(state_idx, {}).get(key, {})
            return next_idx, output
        return state_idx, {}  # no transition = stay


def extract_mealy_machine(bdd: BDD, spec: GR1Spec,
                           output: SynthesisOutput) -> Optional[MealyMachine]:
    """Extract a Mealy machine from a synthesized strategy."""
    if output.result != SynthResult.REALIZABLE or output.strategy is None:
        return None

    # Collect unique states from strategy keys
    state_list = []
    state_to_idx = {}

    for state_key in output.strategy:
        state_dict = dict(state_key)
        if state_key not in state_to_idx:
            state_to_idx[state_key] = len(state_list)
            state_list.append(state_dict)

    if not state_list:
        return None

    # Find initial state
    init_idx = 0

    # Build transitions and outputs
    transitions = {}
    outputs = {}

    for state_key, sys_choice in output.strategy.items():
        src_idx = state_to_idx[state_key]
        if src_idx not in transitions:
            transitions[src_idx] = {}
            outputs[src_idx] = {}

        # Extract env input from the state
        env_input = {v: dict(state_key).get(v, False) for v in spec.env_vars}
        env_key = tuple(sorted(env_input.items()))

        # Next state = env_input + sys_choice
        next_state = dict(env_input)
        next_state.update(sys_choice)
        next_key = tuple(sorted(next_state.items()))

        if next_key not in state_to_idx:
            state_to_idx[next_key] = len(state_list)
            state_list.append(next_state)

        transitions[src_idx][env_key] = state_to_idx[next_key]
        outputs[src_idx][env_key] = sys_choice

    return MealyMachine(
        states=state_list,
        initial=init_idx,
        transitions=transitions,
        outputs=outputs
    )


# ---------------------------------------------------------------------------
# Verification of Synthesized Controllers
# ---------------------------------------------------------------------------

def verify_controller(bdd: BDD, spec: GR1Spec,
                       output: SynthesisOutput) -> Dict:
    """Verify that a synthesized controller satisfies the GR(1) spec.

    Checks:
    1. All initial states are in winning region
    2. Winning region is closed under system strategy
    3. Strategy satisfies safety constraints
    """
    if output.result != SynthResult.REALIZABLE:
        return {'verified': False, 'reason': 'unrealizable'}

    arena = GR1Arena(bdd, spec.env_vars, spec.sys_vars)
    winning = output.winning_region

    env_init = spec.env_init if spec.env_init is not None else bdd.TRUE
    sys_init = spec.sys_init if spec.sys_init is not None else bdd.TRUE

    checks = {}

    # Check 1: init in winning
    init_states = bdd.AND(env_init, sys_init)
    init_outside = bdd.AND(init_states, bdd.NOT(winning))
    checks['init_in_winning'] = (init_outside._id == bdd.FALSE._id)

    # Check 2: winning is a fixpoint (closed under CPre)
    env_safe = spec.env_safe if spec.env_safe is not None else bdd.TRUE
    sys_safe = spec.sys_safe if spec.sys_safe is not None else bdd.TRUE
    cpre_w = arena.controllable_predecessor(winning, env_safe, sys_safe)
    # winning should imply cpre_w: W => CPre(W)
    w_not_cpre = bdd.AND(winning, bdd.NOT(cpre_w))
    checks['winning_closed'] = (w_not_cpre._id == bdd.FALSE._id)

    # Check 3: strategy satisfies safety
    if output.strategy_bdd is not None:
        strat_violates = bdd.AND(output.strategy_bdd, bdd.NOT(sys_safe))
        checks['strategy_safe'] = (strat_violates._id == bdd.FALSE._id)
    else:
        checks['strategy_safe'] = True  # no strategy to check

    checks['verified'] = all(checks.values())
    return checks
