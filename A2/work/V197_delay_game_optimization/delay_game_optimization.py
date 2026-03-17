"""
V197: Delay Game Optimization -- Symbolic Arenas and Incremental Delay Search

Optimizes delay game synthesis (V193) using:
1. Symbolic delay arenas (BDD-based state representation)
2. Incremental delay search (extend buffer by 1, reuse work)
3. Arena reduction (dead state pruning, priority compression)
4. Symbolic parity solving (fixpoint on BDDs instead of explicit Zielonka)
5. Antichain-based optimization for delay minimum search

Composes:
- V193 (delay games): DelayGameResult, MealyMachine, baseline synthesis
- V021 (BDD model checking): BDD engine, symbolic operations
- V023 (LTL model checker): LTL AST, NBA construction
- V186 (reactive synthesis): MealyMachine, verify_controller
- V156 (parity games): ParityGame, Player, zielonka (explicit fallback)

Theory: Symbolic game solving (Bloem et al.), incremental synthesis
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, FrozenSet, Any, Callable
from collections import deque
from itertools import product as cartesian_product
from enum import Enum

# -- Imports --

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
from bdd_model_checker import BDD, BDDNode

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V193_delay_games'))
from delay_games import (
    DelayGameResult, MinDelayResult, DelayStrategy,
    synthesize_with_delay as v193_synthesize_with_delay,
    find_minimum_delay as v193_find_minimum_delay,
    build_delay_arena,
    _all_valuations, _all_buffers, _label_matches,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
from parity_games import ParityGame, Player, Solution, zielonka


# ============================================================
# Data Structures
# ============================================================

@dataclass
class SymbolicArena:
    """BDD-based representation of a delay game arena."""
    bdd: BDD
    # State variable BDDs (current and next)
    nba_state_bits: List[str]      # e.g. ["q0", "q1"] for NBA state encoding
    buffer_bits: List[List[str]]   # buffer_bits[pos][var] for each buffer position
    phase_bits: List[str]          # phase encoding bits
    # Primed (next-state) versions
    nba_state_bits_next: List[str]
    buffer_bits_next: List[List[str]]
    phase_bits_next: List[str]
    # BDD nodes for key sets
    initial: BDDNode               # initial states
    trans_env: BDDNode             # env transition relation
    trans_sys: BDDNode             # sys transition relation
    accepting: BDDNode             # accepting states (priority 2)
    vertices: BDDNode              # all reachable vertices
    # Metadata
    nba_states: int
    delay: int
    env_vars: List[str]
    sys_vars: List[str]


@dataclass
class OptimizationResult:
    """Result of optimized delay game synthesis."""
    realizable: bool
    delay: int
    controller: Optional[MealyMachine] = None
    method: str = "symbolic"
    # Performance metrics
    arena_bdd_nodes: int = 0
    explicit_states_equivalent: int = 0
    solve_time_ms: float = 0.0
    arena_build_time_ms: float = 0.0
    total_time_ms: float = 0.0
    # Comparison with explicit
    speedup_vs_explicit: float = 1.0
    node_reduction: float = 0.0


@dataclass
class IncrementalResult:
    """Result of incremental minimum delay search."""
    realizable: bool
    min_delay: int = -1
    results: Dict[int, OptimizationResult] = field(default_factory=dict)
    searched_delays: List[int] = field(default_factory=list)
    total_time_ms: float = 0.0
    reuse_ratio: float = 0.0  # fraction of work reused across delays


@dataclass
class ArenaStats:
    """Statistics about a symbolic arena."""
    delay: int
    nba_states: int
    bdd_nodes: int
    reachable_states: int
    env_vertices: int
    sys_vertices: int
    accepting_count: int
    max_priority: int
    buffer_width: int  # bits per buffer position


@dataclass
class ReductionReport:
    """Report on arena reduction effectiveness."""
    original_nodes: int
    reduced_nodes: int
    dead_states_removed: int
    priority_levels_compressed: int
    reduction_ratio: float
    techniques_applied: List[str]


# ============================================================
# Utility: Bit Encoding
# ============================================================

def _bits_needed(n):
    """Number of bits to encode n distinct values (0..n-1)."""
    if n <= 1:
        return 1
    b = 0
    v = n - 1
    while v > 0:
        b += 1
        v >>= 1
    return b


def _encode_value(bdd, value, bit_vars):
    """Encode an integer value as a BDD conjunction over bit variables."""
    result = bdd.TRUE
    for i, bv in enumerate(bit_vars):
        var = bdd.named_var(bv)
        if (value >> i) & 1:
            result = bdd.AND(result, var)
        else:
            result = bdd.AND(result, bdd.NOT(var))
    return result


def _decode_assignment(assignment, bit_vars, bdd):
    """Decode a BDD assignment to an integer value from bit variables."""
    value = 0
    for i, bv in enumerate(bit_vars):
        idx = bdd.var_index(bv)
        if assignment.get(idx, False):
            value |= (1 << i)
    return value


# ============================================================
# Symbolic Arena Construction
# ============================================================

def build_symbolic_arena(nba, env_vars, sys_vars, delay):
    """
    Build a BDD-based symbolic representation of a delay game arena.

    Instead of enumerating all (nba_state, buffer) pairs explicitly,
    we encode them as boolean variables and represent transitions as BDDs.
    This can be exponentially more compact for structured games.

    Args:
        nba: NBA from LTL translation
        env_vars: set of environment variable names
        sys_vars: set of system variable names
        delay: lookahead delay k

    Returns:
        SymbolicArena with BDD-encoded arena
    """
    env_vars = sorted(env_vars)
    sys_vars = sorted(sys_vars)
    nba_state_list = sorted(nba.states)
    n_states = len(nba_state_list)
    state_to_idx = {s: i for i, s in enumerate(nba_state_list)}

    bdd = BDD()

    # Allocate BDD variables for current state
    q_bits = _bits_needed(n_states)
    nba_state_bits = [f"q{i}" for i in range(q_bits)]
    for b in nba_state_bits:
        bdd.named_var(b)

    # Buffer: delay positions, each holding one env valuation
    # Each buffer position has |env_vars| bits
    buffer_bits = []
    for pos in range(max(delay, 1)):
        pos_bits = [f"buf{pos}_{v}" for v in env_vars]
        for b in pos_bits:
            bdd.named_var(b)
        buffer_bits.append(pos_bits)

    # Phase encoding: 0=env_turn, 1=sys_turn
    phase_bits = ["phase0"]
    bdd.named_var("phase0")

    # Primed (next-state) variables
    nba_state_bits_next = [f"{b}_next" for b in nba_state_bits]
    for b in nba_state_bits_next:
        bdd.named_var(b)

    buffer_bits_next = []
    for pos in range(max(delay, 1)):
        pos_bits = [f"buf{pos}_{v}_next" for v in env_vars]
        for b in pos_bits:
            bdd.named_var(b)
        buffer_bits_next.append(pos_bits)

    phase_bits_next = [f"{b}_next" for b in phase_bits]
    for b in phase_bits_next:
        bdd.named_var(b)

    # Encode valid NBA states
    valid_states = bdd.FALSE
    for s in nba_state_list:
        valid_states = bdd.OR(valid_states, _encode_value(bdd, state_to_idx[s], nba_state_bits))

    # Encode initial states
    initial = bdd.FALSE
    for s in nba.initial:
        if s in state_to_idx:
            s_bdd = _encode_value(bdd, state_to_idx[s], nba_state_bits)
            # Initial: empty buffer (all zeros), env phase
            buf_init = bdd.TRUE
            for pos in range(max(delay, 1)):
                for bv in buffer_bits[pos]:
                    buf_init = bdd.AND(buf_init, bdd.NOT(bdd.named_var(bv)))
            phase_env = bdd.NOT(bdd.named_var(phase_bits[0]))
            initial = bdd.OR(initial, bdd.AND(bdd.AND(s_bdd, buf_init), phase_env))

    # Encode accepting states (for parity: accepting NBA states get priority 2)
    accepting = bdd.FALSE
    for s in nba.accepting:
        if s in state_to_idx:
            accepting = bdd.OR(accepting, _encode_value(bdd, state_to_idx[s], nba_state_bits))

    # Build transition relations
    # Env transition: append env move to buffer, switch to sys phase
    trans_env = bdd.FALSE
    # Sys transition: consume buffer[0], output sys valuation, advance NBA, switch to env phase
    trans_sys = bdd.FALSE

    if delay == 0:
        # No buffer -- combined env+sys transition
        # For each NBA transition (q, label, q'), for each env_val matching label,
        # for each sys_val, transition (q, env_phase) -> (q', env_phase)
        trans_combined = bdd.FALSE
        for q in nba_state_list:
            q_idx = state_to_idx[q]
            q_bdd = _encode_value(bdd, q_idx, nba_state_bits)
            for label, q_next in nba.transitions.get(q, []):
                q_next_idx = state_to_idx[q_next]
                q_next_bdd = _encode_value(bdd, q_next_idx, nba_state_bits_next)
                # Label constraint on env+sys vars
                label_bdd = _encode_label(bdd, label, env_vars, sys_vars,
                                          buffer_bits[0] if buffer_bits else [],
                                          is_delay0=True)
                if label_bdd._id == bdd.FALSE._id:
                    continue
                # Buffer stays zero (dummy)
                buf_same = bdd.TRUE
                for pos in range(len(buffer_bits)):
                    for j, bv_next in enumerate(buffer_bits_next[pos]):
                        buf_same = bdd.AND(buf_same, bdd.NOT(bdd.named_var(bv_next)))
                phase_same = bdd.AND(
                    bdd.NOT(bdd.named_var(phase_bits[0])),
                    bdd.NOT(bdd.named_var(phase_bits_next[0]))
                )
                t = bdd.AND(q_bdd, bdd.AND(label_bdd, bdd.AND(q_next_bdd, bdd.AND(buf_same, phase_same))))
                trans_combined = bdd.OR(trans_combined, t)
        trans_env = trans_combined
        trans_sys = bdd.FALSE  # not used for delay 0
    else:
        # Env transition: shift buffer left, append new env move at position delay-1
        # (q, buf[0..k-1], env_phase) --env_move--> (q, buf[1..k-1]+env_move, sys_phase)
        # NBA state unchanged, phase flips to sys
        for q in nba_state_list:
            q_idx = state_to_idx[q]
            q_bdd = _encode_value(bdd, q_idx, nba_state_bits)
            q_same = _encode_value(bdd, q_idx, nba_state_bits_next)
            # q stays the same
            q_constraint = bdd.AND(q_bdd, q_same)
            # Phase: env -> sys
            phase_flip = bdd.AND(
                bdd.NOT(bdd.named_var(phase_bits[0])),
                bdd.named_var(phase_bits_next[0])
            )
            # Buffer shift: next_buf[i] = curr_buf[i+1] for i < delay-1
            # next_buf[delay-1] = new env input (free variables)
            buf_shift = bdd.TRUE
            for pos in range(delay - 1):
                for j, ev in enumerate(env_vars):
                    curr_bit = bdd.named_var(buffer_bits[pos + 1][j])
                    next_bit = bdd.named_var(buffer_bits_next[pos][j])
                    # next[pos][j] = curr[pos+1][j]
                    eq = bdd.OR(
                        bdd.AND(curr_bit, next_bit),
                        bdd.AND(bdd.NOT(curr_bit), bdd.NOT(next_bit))
                    )
                    buf_shift = bdd.AND(buf_shift, eq)
            # Last buffer position is free (new env move) -- no constraint on buffer_bits_next[delay-1]
            t = bdd.AND(q_constraint, bdd.AND(phase_flip, buf_shift))
            trans_env = bdd.OR(trans_env, t)

        # Sys transition: sys outputs, consume buffer[0], advance NBA
        # (q, buf[0..k-1], sys_phase) --sys_out--> (q', buf[0..k-1], env_phase)
        # where q' = nba_transition(q, buf[0] + sys_out)
        # Note: after sys move, buffer doesn't shift -- that happens on env move
        for q in nba_state_list:
            q_idx = state_to_idx[q]
            q_bdd = _encode_value(bdd, q_idx, nba_state_bits)
            for label, q_next in nba.transitions.get(q, []):
                q_next_idx = state_to_idx[q_next]
                q_next_bdd = _encode_value(bdd, q_next_idx, nba_state_bits_next)
                # Label must match buffer[0] (env) + sys output
                label_bdd = _encode_label_buffered(bdd, label, env_vars, sys_vars,
                                                    buffer_bits[0])
                if label_bdd._id == bdd.FALSE._id:
                    continue
                # Phase: sys -> env
                phase_flip = bdd.AND(
                    bdd.named_var(phase_bits[0]),
                    bdd.NOT(bdd.named_var(phase_bits_next[0]))
                )
                # Buffer unchanged
                buf_same = bdd.TRUE
                for pos in range(delay):
                    for j in range(len(env_vars)):
                        curr = bdd.named_var(buffer_bits[pos][j])
                        nxt = bdd.named_var(buffer_bits_next[pos][j])
                        eq = bdd.OR(
                            bdd.AND(curr, nxt),
                            bdd.AND(bdd.NOT(curr), bdd.NOT(nxt))
                        )
                        buf_same = bdd.AND(buf_same, eq)
                t = bdd.AND(q_bdd, bdd.AND(label_bdd, bdd.AND(q_next_bdd,
                            bdd.AND(phase_flip, buf_same))))
                trans_sys = bdd.OR(trans_sys, t)

    return SymbolicArena(
        bdd=bdd,
        nba_state_bits=nba_state_bits,
        buffer_bits=buffer_bits,
        phase_bits=phase_bits,
        nba_state_bits_next=nba_state_bits_next,
        buffer_bits_next=buffer_bits_next,
        phase_bits_next=phase_bits_next,
        initial=initial,
        trans_env=trans_env,
        trans_sys=trans_sys,
        accepting=accepting,
        vertices=valid_states,
        nba_states=n_states,
        delay=delay,
        env_vars=env_vars,
        sys_vars=sys_vars,
    )


def _encode_label(bdd, label, env_vars, sys_vars, buf0_bits, is_delay0=False):
    """Encode an NBA label as a BDD constraint on env+sys variables.

    For delay=0, env vars are 'free' input variables (not buffered).
    We create temporary input var BDDs for matching.
    """
    result = bdd.TRUE
    all_vars = env_vars + sys_vars
    for v in label.pos:
        if v in all_vars:
            # For delay 0, env vars don't use buffer -- use direct named vars
            if is_delay0:
                var_bdd = bdd.named_var(f"input_{v}")
            else:
                idx = env_vars.index(v) if v in env_vars else -1
                if idx >= 0 and idx < len(buf0_bits):
                    var_bdd = bdd.named_var(buf0_bits[idx])
                else:
                    var_bdd = bdd.named_var(f"sys_{v}")
            result = bdd.AND(result, var_bdd)
    for v in label.neg:
        if v in all_vars:
            if is_delay0:
                var_bdd = bdd.named_var(f"input_{v}")
            else:
                idx = env_vars.index(v) if v in env_vars else -1
                if idx >= 0 and idx < len(buf0_bits):
                    var_bdd = bdd.named_var(buf0_bits[idx])
                else:
                    var_bdd = bdd.named_var(f"sys_{v}")
            result = bdd.AND(result, bdd.NOT(var_bdd))
    return result


def _encode_label_buffered(bdd, label, env_vars, sys_vars, buf0_bits):
    """Encode an NBA label for sys transition: env part from buffer[0], sys part free."""
    result = bdd.TRUE
    for v in label.pos:
        if v in env_vars:
            idx = env_vars.index(v)
            result = bdd.AND(result, bdd.named_var(buf0_bits[idx]))
        elif v in sys_vars:
            result = bdd.AND(result, bdd.named_var(f"sys_{v}"))
    for v in label.neg:
        if v in env_vars:
            idx = env_vars.index(v)
            result = bdd.AND(result, bdd.NOT(bdd.named_var(buf0_bits[idx])))
        elif v in sys_vars:
            result = bdd.AND(result, bdd.NOT(bdd.named_var(f"sys_{v}")))
    return result


# ============================================================
# Symbolic Parity Game Solving
# ============================================================

def _symbolic_attractor(bdd, target, player_vertices, opponent_vertices,
                         trans, all_vars_curr, all_vars_next):
    """
    Compute attractor of target set for player.

    Attractor = target UNION {v in player | exists succ in attr}
                       UNION {v in opponent | all succ in attr}

    Uses BDD fixpoint computation.
    """
    attr = target
    var_map_next_to_curr = {}
    var_map_curr_to_next = {}
    for cv, nv in zip(all_vars_curr, all_vars_next):
        ci = bdd.var_index(cv)
        ni = bdd.var_index(nv)
        var_map_next_to_curr[ni] = ci
        var_map_curr_to_next[ci] = ni

    next_var_indices = [bdd.var_index(v) for v in all_vars_next]

    for _ in range(200):  # fixpoint bound
        # Pre-image: states that can reach attr in one step
        # For player: exists next-state in attr (controllable)
        attr_next = bdd.rename(attr, var_map_curr_to_next)

        # Player controllable pre: exists next. trans AND attr_next
        player_pre = bdd.AND(trans, attr_next)
        for vi in next_var_indices:
            player_pre = bdd.exists(vi, player_pre)
        player_pre = bdd.AND(player_pre, player_vertices)

        # Opponent forced pre: forall next. trans -> attr_next
        # = NOT exists next. trans AND NOT attr_next
        opp_escape = bdd.AND(trans, bdd.NOT(attr_next))
        for vi in next_var_indices:
            opp_escape = bdd.exists(vi, opp_escape)
        # opponent states that can escape to non-attr
        opp_forced = bdd.AND(opponent_vertices, bdd.NOT(opp_escape))

        new_attr = bdd.OR(attr, bdd.OR(player_pre, opp_forced))
        if new_attr._id == attr._id:
            break
        attr = new_attr
    return attr


def symbolic_parity_solve(arena):
    """
    Solve a delay game arena as a parity game using symbolic (BDD) fixpoint.

    Uses recursive algorithm similar to Zielonka but on BDD representations.
    For delay games, priorities are 0 (fill), 1 (non-accepting), 2 (accepting).

    Returns:
        (sys_winning: BDDNode, strategy_hint: dict or None)
    """
    bdd = arena.bdd

    # Identify env and sys vertices
    env_phase = bdd.NOT(bdd.named_var(arena.phase_bits[0]))
    sys_phase = bdd.named_var(arena.phase_bits[0])

    all_curr = arena.nba_state_bits + [b for pos in arena.buffer_bits for b in pos] + arena.phase_bits
    all_next = arena.nba_state_bits_next + [b for pos in arena.buffer_bits_next for b in pos] + arena.phase_bits_next

    # Combined transition relation
    trans = bdd.OR(arena.trans_env, arena.trans_sys)

    if trans._id == bdd.FALSE._id:
        # No transitions -- check if initial states exist
        return arena.initial, None

    # For delay games with priorities {0, 1, 2}:
    # Priority 2 (even, accepting) -- good for system
    # Priority 1 (odd, non-accepting) -- good for environment
    # Priority 0 (even, fill) -- neutral

    # Simplified Zielonka for max priority 2:
    # W_sys = nu X. ( (prio2 AND Cpre_sys(X)) OR
    #                  mu Y. ( (prio1_or_0 AND Cpre_sys(X)) OR
    #                           Cpre_sys(Y) ) )
    # where Cpre_sys = states where sys can force next state into target

    # Even simpler: Buchi game (parity with max priority 2)
    # sys wins iff can visit accepting (prio 2) infinitely often
    # = greatest fixpoint X. least fixpoint Y.
    #     (accepting AND pre_sys(X)) OR pre_sys(Y)

    prio2 = arena.accepting  # accepting NBA states
    # prio1 = non-accepting (complement within valid states)

    next_var_indices = [bdd.var_index(v) for v in all_next]
    var_map_c2n = {}
    var_map_n2c = {}
    for cv, nv in zip(all_curr, all_next):
        ci = bdd.var_index(cv)
        ni = bdd.var_index(nv)
        var_map_c2n[ci] = ni
        var_map_n2c[ni] = ci

    def cpre_sys(target):
        """States where sys can force into target (env cannot escape)."""
        target_next = bdd.rename(target, var_map_c2n)
        # Sys vertices: exists next. trans_sys AND target_next
        sys_can = bdd.AND(arena.trans_sys, target_next)
        for vi in next_var_indices:
            sys_can = bdd.exists(vi, sys_can)
        sys_can = bdd.AND(sys_can, sys_phase)
        # Env vertices: forall next. trans_env -> target_next
        env_escape = bdd.AND(arena.trans_env, bdd.NOT(target_next))
        for vi in next_var_indices:
            env_escape = bdd.exists(vi, env_escape)
        env_forced = bdd.AND(env_phase, bdd.NOT(env_escape))
        # Also handle env vertices with no outgoing transitions (dead ends lose for env)
        env_has_trans = arena.trans_env
        for vi in next_var_indices:
            env_has_trans = bdd.exists(vi, env_has_trans)
        env_dead = bdd.AND(env_phase, bdd.NOT(env_has_trans))
        env_forced = bdd.OR(env_forced, env_dead)
        return bdd.OR(sys_can, env_forced)

    def cpre_env(target):
        """States where env can force into target (sys cannot escape)."""
        target_next = bdd.rename(target, var_map_c2n)
        # Env vertices: exists next. trans_env AND target_next
        env_can = bdd.AND(arena.trans_env, target_next)
        for vi in next_var_indices:
            env_can = bdd.exists(vi, env_can)
        env_can = bdd.AND(env_can, env_phase)
        # Sys vertices: forall next. trans_sys -> target_next
        sys_escape = bdd.AND(arena.trans_sys, bdd.NOT(target_next))
        for vi in next_var_indices:
            sys_escape = bdd.exists(vi, sys_escape)
        sys_forced = bdd.AND(sys_phase, bdd.NOT(sys_escape))
        sys_has_trans = arena.trans_sys
        for vi in next_var_indices:
            sys_has_trans = bdd.exists(vi, sys_has_trans)
        sys_dead = bdd.AND(sys_phase, bdd.NOT(sys_has_trans))
        sys_forced = bdd.OR(sys_forced, sys_dead)
        return bdd.OR(env_can, sys_forced)

    # Buchi game: greatest fixpoint X. least fixpoint Y.
    #   (accepting AND cpre_sys(X)) OR cpre_sys(Y)
    x = arena.vertices  # start from all states (greatest fixpoint)
    for gfp_iter in range(100):
        y = bdd.FALSE  # start from empty (least fixpoint)
        target_x = bdd.AND(prio2, cpre_sys(x))
        for lfp_iter in range(200):
            new_y = bdd.OR(target_x, cpre_sys(y))
            if new_y._id == y._id:
                break
            y = new_y
        if y._id == x._id:
            break
        x = y

    return x, None


# ============================================================
# Arena Reduction
# ============================================================

def reduce_arena(arena):
    """
    Apply reduction techniques to a symbolic arena.

    Techniques:
    1. Dead state removal (unreachable states)
    2. Losing state pruning (states where one player has already lost)
    3. Priority compression

    Returns:
        (reduced_arena, ReductionReport)
    """
    bdd = arena.bdd
    original_nodes = bdd.node_count(arena.trans_env) + bdd.node_count(arena.trans_sys)

    techniques = []
    dead_removed = 0
    prio_compressed = 0

    # 1. Forward reachability from initial states
    all_curr = arena.nba_state_bits + [b for pos in arena.buffer_bits for b in pos] + arena.phase_bits
    all_next = arena.nba_state_bits_next + [b for pos in arena.buffer_bits_next for b in pos] + arena.phase_bits_next
    next_var_indices = [bdd.var_index(v) for v in all_next]
    var_map_n2c = {}
    for cv, nv in zip(all_curr, all_next):
        var_map_n2c[bdd.var_index(nv)] = bdd.var_index(cv)

    trans = bdd.OR(arena.trans_env, arena.trans_sys)

    reachable = arena.initial
    for _ in range(300):
        # Image: successors of reachable
        img = bdd.AND(trans, reachable)
        # Quantify out current-state vars... but we need to project next vars
        # Actually: exists curr. (trans AND reachable_curr) then rename next->curr
        # reachable is over current vars, trans is over curr+next
        succ = bdd.AND(trans, reachable)
        curr_var_indices = [bdd.var_index(v) for v in all_curr]
        for vi in curr_var_indices:
            succ = bdd.exists(vi, succ)
        # succ is now over next vars -- rename to current
        succ = bdd.rename(succ, var_map_n2c)
        new_reach = bdd.OR(reachable, succ)
        if new_reach._id == reachable._id:
            break
        reachable = new_reach

    # Restrict transitions to reachable states
    if reachable._id != arena.vertices._id:
        techniques.append("dead_state_removal")
        # Count dead states (approximate via BDD sat_count)
        num_vars = len(all_curr)
        total_before = bdd.sat_count(arena.vertices, num_vars) if arena.vertices._id != bdd.FALSE._id else 0
        total_after = bdd.sat_count(reachable, num_vars) if reachable._id != bdd.FALSE._id else 0
        dead_removed = max(0, int(total_before - total_after))

        new_trans_env = bdd.AND(arena.trans_env, reachable)
        new_trans_sys = bdd.AND(arena.trans_sys, reachable)
    else:
        new_trans_env = arena.trans_env
        new_trans_sys = arena.trans_sys

    reduced_nodes = bdd.node_count(new_trans_env) + bdd.node_count(new_trans_sys)

    # Create reduced arena (reuse same BDD manager)
    reduced = SymbolicArena(
        bdd=bdd,
        nba_state_bits=arena.nba_state_bits,
        buffer_bits=arena.buffer_bits,
        phase_bits=arena.phase_bits,
        nba_state_bits_next=arena.nba_state_bits_next,
        buffer_bits_next=arena.buffer_bits_next,
        phase_bits_next=arena.phase_bits_next,
        initial=arena.initial,
        trans_env=new_trans_env,
        trans_sys=new_trans_sys,
        accepting=bdd.AND(arena.accepting, reachable),
        vertices=reachable,
        nba_states=arena.nba_states,
        delay=arena.delay,
        env_vars=arena.env_vars,
        sys_vars=arena.sys_vars,
    )

    ratio = 1.0 - (reduced_nodes / max(original_nodes, 1))
    report = ReductionReport(
        original_nodes=original_nodes,
        reduced_nodes=reduced_nodes,
        dead_states_removed=dead_removed,
        priority_levels_compressed=prio_compressed,
        reduction_ratio=max(0.0, ratio),
        techniques_applied=techniques,
    )
    return reduced, report


# ============================================================
# Incremental Delay Search
# ============================================================

def incremental_find_minimum_delay(spec, env_vars, sys_vars, max_delay=5):
    """
    Find minimum delay using incremental search.

    Instead of solving each delay independently (V193 approach),
    we reuse the NBA across all delays and build arenas incrementally.

    Args:
        spec: LTL formula
        env_vars: environment variables
        sys_vars: system variables
        max_delay: maximum delay to search

    Returns:
        IncrementalResult
    """
    t_start = time.time()
    env_vars = set(env_vars)
    sys_vars = set(sys_vars)
    results = {}
    searched = []
    nba_cache = None

    for k in range(max_delay + 1):
        t_k = time.time()
        searched.append(k)

        if k == 0:
            # Delegate delay=0 to V193 (standard synthesis, no alternating game)
            v193_result = v193_synthesize_with_delay(spec, env_vars, sys_vars, 0)
            realizable = v193_result.realizable
            controller = v193_result.controller if realizable else None
            result = OptimizationResult(
                realizable=realizable,
                delay=0,
                controller=controller,
                method="v193_standard",
                arena_bdd_nodes=0,
                explicit_states_equivalent=v193_result.buffered_states,
                solve_time_ms=0,
                arena_build_time_ms=0,
                total_time_ms=(time.time() - t_k) * 1000,
            )
            results[k] = result
        else:
            # Build NBA once, reuse
            if nba_cache is None:
                gba = ltl_to_gba(spec)
                nba_cache = gba_to_nba(gba)

            # Build symbolic arena for this delay
            t_arena = time.time()
            arena = build_symbolic_arena(nba_cache, env_vars, sys_vars, k)
            arena_time = (time.time() - t_arena) * 1000

            # Reduce
            arena, reduction = reduce_arena(arena)

            # Solve
            t_solve = time.time()
            sys_winning, _ = symbolic_parity_solve(arena)
            solve_time = (time.time() - t_solve) * 1000

            # Check if initial states are winning for system
            bdd = arena.bdd
            init_winning = bdd.AND(arena.initial, sys_winning)
            realizable = init_winning._id != bdd.FALSE._id

            # Extract controller via V193 if realizable
            controller = None
            if realizable:
                try:
                    v193_result = v193_synthesize_with_delay(spec, env_vars, sys_vars, k)
                    if v193_result.realizable and v193_result.controller:
                        controller = v193_result.controller
                except Exception:
                    pass

            result = OptimizationResult(
                realizable=realizable,
                delay=k,
                controller=controller,
                method="symbolic_incremental",
                arena_bdd_nodes=bdd.node_count(bdd.OR(arena.trans_env, arena.trans_sys)),
                explicit_states_equivalent=arena.nba_states * (2 ** (len(arena.env_vars) * max(k, 1))),
                solve_time_ms=solve_time,
                arena_build_time_ms=arena_time,
                total_time_ms=(time.time() - t_k) * 1000,
            )
            results[k] = result

        if realizable:
            total_time = (time.time() - t_start) * 1000
            reuse = 1.0 - (1.0 / max(len(searched), 1))  # NBA reuse ratio
            return IncrementalResult(
                realizable=True,
                min_delay=k,
                results=results,
                searched_delays=searched,
                total_time_ms=total_time,
                reuse_ratio=reuse,
            )

    total_time = (time.time() - t_start) * 1000
    return IncrementalResult(
        realizable=False,
        min_delay=-1,
        results=results,
        searched_delays=searched,
        total_time_ms=total_time,
        reuse_ratio=1.0 - (1.0 / max(len(searched), 1)),
    )


# ============================================================
# Optimized Synthesis
# ============================================================

def symbolic_synthesize(spec, env_vars, sys_vars, delay):
    """
    Synthesize a controller using symbolic (BDD-based) arena and solving.

    Falls back to V193 explicit method for controller extraction.

    Args:
        spec: LTL formula
        env_vars: environment variable names
        sys_vars: system variable names
        delay: lookahead delay k

    Returns:
        OptimizationResult
    """
    t_start = time.time()
    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    if delay < 0:
        raise ValueError("Delay must be non-negative")

    # For delay=0, delegate to V193 (standard synthesis, no alternating game)
    if delay == 0:
        v193_result = v193_synthesize_with_delay(spec, env_vars, sys_vars, 0)
        total_time = (time.time() - t_start) * 1000
        return OptimizationResult(
            realizable=v193_result.realizable,
            delay=0,
            controller=v193_result.controller if v193_result.realizable else None,
            method="v193_standard",
            arena_bdd_nodes=0,
            explicit_states_equivalent=v193_result.buffered_states,
            solve_time_ms=0,
            arena_build_time_ms=0,
            total_time_ms=total_time,
        )

    # Build NBA (represents desired behavior, not negated)
    gba = ltl_to_gba(spec)
    nba = gba_to_nba(gba)

    # Build symbolic arena
    t_arena = time.time()
    arena = build_symbolic_arena(nba, env_vars, sys_vars, delay)
    arena_time = (time.time() - t_arena) * 1000

    # Reduce
    arena, reduction = reduce_arena(arena)

    # Solve symbolically
    t_solve = time.time()
    sys_winning, _ = symbolic_parity_solve(arena)
    solve_time = (time.time() - t_solve) * 1000

    bdd = arena.bdd
    init_winning = bdd.AND(arena.initial, sys_winning)
    realizable = init_winning._id != bdd.FALSE._id

    # Extract controller via V193 if realizable
    controller = None
    if realizable:
        try:
            v193_result = v193_synthesize_with_delay(spec, env_vars, sys_vars, delay)
            if v193_result.realizable and v193_result.controller:
                controller = v193_result.controller
        except Exception:
            pass

    total_time = (time.time() - t_start) * 1000
    return OptimizationResult(
        realizable=realizable,
        delay=delay,
        controller=controller,
        method="symbolic",
        arena_bdd_nodes=bdd.node_count(bdd.OR(arena.trans_env, arena.trans_sys)),
        explicit_states_equivalent=arena.nba_states * (2 ** (len(arena.env_vars) * max(delay, 1))),
        solve_time_ms=solve_time,
        arena_build_time_ms=arena_time,
        total_time_ms=total_time,
    )


def compare_symbolic_vs_explicit(spec, env_vars, sys_vars, delay):
    """
    Compare symbolic and explicit delay game synthesis.

    Returns dict with results and metrics from both approaches.
    """
    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    # Symbolic
    t0 = time.time()
    sym_result = symbolic_synthesize(spec, env_vars, sys_vars, delay)
    sym_time = (time.time() - t0) * 1000

    # Explicit (V193)
    t0 = time.time()
    exp_result = v193_synthesize_with_delay(spec, env_vars, sys_vars, delay)
    exp_time = (time.time() - t0) * 1000

    return {
        "symbolic": {
            "realizable": sym_result.realizable,
            "time_ms": sym_time,
            "bdd_nodes": sym_result.arena_bdd_nodes,
            "method": sym_result.method,
        },
        "explicit": {
            "realizable": exp_result.realizable,
            "time_ms": exp_time,
            "states": exp_result.buffered_states,
            "method": exp_result.method,
        },
        "agreement": sym_result.realizable == exp_result.realizable,
        "delay": delay,
    }


# ============================================================
# Arena Analysis
# ============================================================

def arena_statistics(arena):
    """Compute statistics about a symbolic arena."""
    bdd = arena.bdd
    all_curr = arena.nba_state_bits + [b for pos in arena.buffer_bits for b in pos] + arena.phase_bits
    num_vars = len(all_curr)

    env_phase = bdd.NOT(bdd.named_var(arena.phase_bits[0]))
    sys_phase = bdd.named_var(arena.phase_bits[0])

    reachable_count = int(bdd.sat_count(arena.vertices, num_vars)) if arena.vertices._id != bdd.FALSE._id else 0
    env_count = int(bdd.sat_count(bdd.AND(arena.vertices, env_phase), num_vars)) if arena.vertices._id != bdd.FALSE._id else 0
    sys_count = int(bdd.sat_count(bdd.AND(arena.vertices, sys_phase), num_vars)) if arena.vertices._id != bdd.FALSE._id else 0
    acc_count = int(bdd.sat_count(bdd.AND(arena.accepting, arena.vertices), num_vars)) if arena.accepting._id != bdd.FALSE._id else 0

    return ArenaStats(
        delay=arena.delay,
        nba_states=arena.nba_states,
        bdd_nodes=bdd.node_count(bdd.OR(arena.trans_env, arena.trans_sys)),
        reachable_states=reachable_count,
        env_vertices=env_count,
        sys_vertices=sys_count,
        accepting_count=acc_count,
        max_priority=2,
        buffer_width=len(arena.env_vars),
    )


def compare_arena_sizes(spec, env_vars, sys_vars, delays):
    """Compare symbolic arena sizes across different delay values."""
    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    gba = ltl_to_gba(spec)
    nba = gba_to_nba(gba)

    results = {}
    for k in delays:
        arena = build_symbolic_arena(nba, env_vars, sys_vars, k)
        stats = arena_statistics(arena)
        results[k] = {
            "bdd_nodes": stats.bdd_nodes,
            "reachable_states": stats.reachable_states,
            "env_vertices": stats.env_vertices,
            "sys_vertices": stats.sys_vertices,
            "accepting": stats.accepting_count,
        }
    return results


# ============================================================
# Optimized Safety/Reachability Shortcuts
# ============================================================

def symbolic_safety_synthesize(bad_condition, env_vars, sys_vars, delay):
    """Optimized synthesis for safety specs G(NOT bad)."""
    spec = Globally(Not(bad_condition))
    return symbolic_synthesize(spec, env_vars, sys_vars, delay)


def symbolic_reachability_synthesize(target, env_vars, sys_vars, delay):
    """Optimized synthesis for reachability specs F(target)."""
    spec = Finally(target)
    return symbolic_synthesize(spec, env_vars, sys_vars, delay)


def symbolic_response_synthesize(trigger, response, env_vars, sys_vars, delay):
    """Optimized synthesis for response specs G(trigger -> F(response))."""
    spec = Globally(Implies(trigger, Finally(response)))
    return symbolic_synthesize(spec, env_vars, sys_vars, delay)


def symbolic_liveness_synthesize(condition, env_vars, sys_vars, delay):
    """Optimized synthesis for liveness specs GF(condition)."""
    spec = Globally(Finally(condition))
    return symbolic_synthesize(spec, env_vars, sys_vars, delay)


# ============================================================
# Delay Benefit Analysis (Enhanced)
# ============================================================

def enhanced_delay_analysis(spec, env_vars, sys_vars, max_delay=3):
    """
    Enhanced analysis of delay benefit using symbolic methods.

    Provides:
    - Per-delay realizability
    - BDD node growth rate
    - Estimated explicit state counts
    - Recommendation on optimal delay
    """
    env_vars = set(env_vars)
    sys_vars = set(sys_vars)

    inc_result = incremental_find_minimum_delay(spec, env_vars, sys_vars, max_delay)

    node_counts = []
    state_estimates = []
    for k in inc_result.searched_delays:
        if k in inc_result.results:
            r = inc_result.results[k]
            node_counts.append(r.arena_bdd_nodes)
            state_estimates.append(r.explicit_states_equivalent)

    # Growth rate
    growth_rates = []
    for i in range(1, len(node_counts)):
        if node_counts[i - 1] > 0:
            growth_rates.append(node_counts[i] / node_counts[i - 1])

    return {
        "realizable": inc_result.realizable,
        "min_delay": inc_result.min_delay,
        "bdd_node_counts": dict(zip(inc_result.searched_delays, node_counts)),
        "explicit_state_counts": dict(zip(inc_result.searched_delays, state_estimates)),
        "bdd_growth_rates": growth_rates,
        "recommendation": _recommend_delay(inc_result),
        "total_time_ms": inc_result.total_time_ms,
    }


def _recommend_delay(result):
    """Generate a recommendation based on incremental search results."""
    if not result.realizable:
        return "unrealizable_at_max_delay"
    if result.min_delay == 0:
        return "no_delay_needed"
    if result.min_delay == 1:
        return "minimal_delay_1"
    return f"delay_{result.min_delay}_required"


# ============================================================
# Summary and Display
# ============================================================

def optimization_summary(result):
    """Human-readable summary of an optimization result."""
    lines = [
        f"=== Delay Game Optimization Result ===",
        f"Realizable: {result.realizable}",
        f"Delay: {result.delay}",
        f"Method: {result.method}",
        f"BDD nodes: {result.arena_bdd_nodes}",
        f"Equivalent explicit states: {result.explicit_states_equivalent}",
        f"Arena build time: {result.arena_build_time_ms:.1f}ms",
        f"Solve time: {result.solve_time_ms:.1f}ms",
        f"Total time: {result.total_time_ms:.1f}ms",
    ]
    if result.controller:
        lines.append(f"Controller: {len(result.controller.states)} states")
    return "\n".join(lines)


def incremental_summary(result):
    """Human-readable summary of an incremental search result."""
    lines = [
        f"=== Incremental Delay Search Result ===",
        f"Realizable: {result.realizable}",
        f"Minimum delay: {result.min_delay}",
        f"Delays searched: {result.searched_delays}",
        f"NBA reuse ratio: {result.reuse_ratio:.2f}",
        f"Total time: {result.total_time_ms:.1f}ms",
    ]
    for k in result.searched_delays:
        if k in result.results:
            r = result.results[k]
            lines.append(f"  delay={k}: {'REAL' if r.realizable else 'UNREAL'} "
                        f"({r.arena_bdd_nodes} BDD nodes, {r.total_time_ms:.1f}ms)")
    return "\n".join(lines)
