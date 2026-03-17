"""V190: Bounded Synthesis (SMT-based).

Synthesizes finite-state reactive controllers from LTL specifications using
SMT-based bounded synthesis (Finkbeiner & Schewe 2007).

Key idea: encode the existence of a k-state Mealy machine controller as an
SMT formula. Uses annotation functions to witness bounded co-Buchi acceptance:
the annotation must strictly decrease on rejecting transitions, ensuring every
run visits accepting states infinitely often.

Pipeline:
  1. LTL phi -> negate -> NBA(not phi) -> UCW(phi)
  2. For k = 1, 2, ...: encode SMT formula Phi(k, b) and check satisfiability
  3. If SAT: extract controller from SMT model

Composes:
  - V023 (LTL AST, ltl_to_gba, gba_to_nba)
  - C037 (SMT solver)
  - V186/V188 (MealyMachine for output compatibility)
"""

import sys
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List, Tuple, FrozenSet, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from ltl_model_checker import (
    LTL, LTLOp, Atom, Not as LTLNot, And as LTLAnd, Or as LTLOr,
    Implies as LTLImplies, Next, Finally, Globally, Until, Release,
    LTLTrue, LTLFalse,
    ltl_to_gba, gba_to_nba, NBA, Label, GBA
)
from smt_solver import SMTSolver, SMTResult, INT, BOOL


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SynthVerdict(Enum):
    REALIZABLE = "realizable"
    UNREALIZABLE = "unrealizable"
    UNKNOWN = "unknown"


@dataclass
class UCW:
    """Universal Co-Buchi Automaton.

    Derived from NBA of negated spec. ALL runs must visit rejecting states
    only finitely often.
    """
    states: Set[int]
    initial: Set[int]
    transitions: Dict[int, List[Tuple[Label, int]]]  # state -> [(label, next)]
    rejecting: Set[int]  # states that must be visited finitely often
    ap: Set[str]


@dataclass
class Annotation:
    """Annotation function witnessing bounded co-Buchi acceptance.

    Maps (ucw_state, controller_state) -> integer bound value.
    Must decrease on rejecting transitions, stay or decrease on non-rejecting.
    """
    values: Dict[Tuple[int, int], int]  # (q, c) -> bound value
    max_bound: int


@dataclass
class Controller:
    """Finite-state Mealy machine controller."""
    n_states: int
    initial: int
    transitions: Dict[Tuple[int, FrozenSet[str]], Tuple[int, FrozenSet[str]]]
    # (ctrl_state, env_input) -> (next_ctrl_state, sys_output)
    env_vars: Set[str]
    sys_vars: Set[str]

    def step(self, state, env_input):
        """Execute one step."""
        key = (state, frozenset(env_input) if not isinstance(env_input, frozenset) else env_input)
        return self.transitions.get(key)

    def simulate(self, inputs, max_steps=100):
        """Simulate controller on input sequence."""
        trace = []
        state = self.initial
        for i, inp in enumerate(inputs[:max_steps]):
            inp_fs = frozenset(inp) if not isinstance(inp, frozenset) else inp
            result = self.step(state, inp_fs)
            if result is None:
                break
            next_state, output = result
            trace.append((state, inp_fs, output, next_state))
            state = next_state
        return trace


@dataclass
class SynthResult:
    """Result of bounded synthesis."""
    verdict: SynthVerdict
    controller: Optional[Controller] = None
    annotation: Optional[Annotation] = None
    n_states: int = 0
    bound: int = 0
    ucw_states: int = 0
    smt_vars: int = 0
    smt_clauses: int = 0
    method: str = "bounded_synthesis"
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# UCW construction
# ---------------------------------------------------------------------------

def ucw_from_ltl(spec):
    """Convert LTL spec to Universal Co-Buchi Automaton.

    Pipeline: negate spec -> NBA(not spec) -> UCW(spec).
    The accepting states of NBA(not spec) become rejecting states of UCW(spec).
    """
    neg_spec = LTLNot(spec)
    gba = ltl_to_gba(neg_spec)
    nba = gba_to_nba(gba)

    return UCW(
        states=set(nba.states),
        initial=set(nba.initial),
        transitions=dict(nba.transitions),
        rejecting=set(nba.accepting),  # accepting in NBA(not phi) = rejecting in UCW(phi)
        ap=set(nba.ap),
    )


def ucw_from_nba(nba):
    """Wrap an existing NBA as a UCW (accepting -> rejecting)."""
    return UCW(
        states=set(nba.states),
        initial=set(nba.initial),
        transitions=dict(nba.transitions),
        rejecting=set(nba.accepting),
        ap=set(nba.ap),
    )


# ---------------------------------------------------------------------------
# Helper: enumerate valuations
# ---------------------------------------------------------------------------

def _all_valuations(variables):
    """Generate all 2^|variables| valuations as frozensets of true variables."""
    var_list = sorted(variables)
    n = len(var_list)
    result = []
    for i in range(1 << n):
        valuation = frozenset(var_list[j] for j in range(n) if (i >> j) & 1)
        result.append(valuation)
    return result


def _label_matches(label, valuation):
    """Check if a valuation satisfies a label."""
    for p in label.pos:
        if p not in valuation:
            return False
    for n in label.neg:
        if n in valuation:
            return False
    return True


# ---------------------------------------------------------------------------
# SMT encoding
# ---------------------------------------------------------------------------

def encode_bounded(ucw, env_vars, sys_vars, k, b):
    """Encode bounded synthesis as SMT formula.

    Uses fully-boolean encoding for the transition function (sel variables)
    to avoid integer EQ in implication premises (which causes UNKNOWN in C037).
    Reachability-guarded annotation constraints handle absorbing rejecting sinks.

    Args:
        ucw: Universal Co-Buchi Automaton
        env_vars: set of environment variable names
        sys_vars: set of system variable names
        k: number of controller states (1..k)
        b: annotation bound (0..b)

    Returns:
        (solver, var_info, n_vars)
    """
    solver = SMTSolver()
    env_vals = _all_valuations(env_vars)
    sys_vals = _all_valuations(sys_vars)

    sel = {}   # (c, e_idx, c_next) -> SMT bool var (transition selector)
    out = {}   # (c, e_idx, v) -> SMT bool var
    lam = {}   # (q, c) -> SMT int var
    reach = {} # (q, c) -> SMT bool var

    ucw_states = sorted(ucw.states)

    # Create controller transition variables (boolean selectors)
    for c in range(k):
        for e_idx in range(len(env_vals)):
            if k == 1:
                # Only one choice: always go to state 0
                sel[(c, e_idx, 0)] = solver.BoolVal(True)
            else:
                # Create boolean selector for each possible next state
                sel_vars = []
                for c_next in range(k):
                    sv = solver.Bool(f'sel_{c}_{e_idx}_{c_next}')
                    sel[(c, e_idx, c_next)] = sv
                    sel_vars.append(sv)
                # Exactly-one constraint: at least one, and pairwise exclusion
                solver.add(solver.Or(*sel_vars))
                for i in range(len(sel_vars)):
                    for j in range(i + 1, len(sel_vars)):
                        solver.add(solver.Or(
                            solver.Not(sel_vars[i]),
                            solver.Not(sel_vars[j])
                        ))

            # Output variables
            for v in sorted(sys_vars):
                o_var = solver.Bool(f'out_{c}_{e_idx}_{v}')
                out[(c, e_idx, v)] = o_var

    # Create annotation and reachability variables
    for q in ucw_states:
        for c in range(k):
            l_var = solver.Int(f'lam_{q}_{c}')
            lam[(q, c)] = l_var
            solver.add(l_var >= solver.IntVal(0))
            solver.add(l_var <= solver.IntVal(b))

            r_var = solver.Bool(f'reach_{q}_{c}')
            reach[(q, c)] = r_var

    # Initial reachability
    for q in ucw_states:
        if q in ucw.initial:
            solver.add(reach[(q, 0)])

    # Reachability propagation and annotation constraints
    for q in ucw_states:
        trans_q = ucw.transitions.get(q, [])
        if not trans_q:
            continue

        for c in range(k):
            for e_idx, e_val in enumerate(env_vals):
                for s_idx, s_val in enumerate(sys_vals):
                    # Build output-match condition
                    out_conds = []
                    for v in sorted(sys_vars):
                        if v in s_val:
                            out_conds.append(out[(c, e_idx, v)])
                        else:
                            out_conds.append(solver.Not(out[(c, e_idx, v)]))

                    if len(out_conds) == 0:
                        out_match = solver.BoolVal(True)
                    elif len(out_conds) == 1:
                        out_match = out_conds[0]
                    else:
                        out_match = solver.And(*out_conds)

                    combined = e_val | s_val

                    for label, q_next in trans_q:
                        if not _label_matches(label, combined):
                            continue

                        is_rejecting = q_next in ucw.rejecting

                        for c_next in range(k):
                            # Premise: reachable AND sel matches AND output matches
                            premise = solver.And(
                                reach[(q, c)],
                                sel[(c, e_idx, c_next)],
                                out_match
                            )

                            # Propagate reachability
                            solver.add(solver.Implies(premise, reach[(q_next, c_next)]))

                            # Annotation constraint
                            if is_rejecting:
                                constraint = lam[(q_next, c_next)] < lam[(q, c)]
                            else:
                                constraint = lam[(q_next, c_next)] <= lam[(q, c)]

                            solver.add(solver.Implies(premise, constraint))

    var_info = {
        'sel': sel,
        'out': out,
        'lam': lam,
        'reach': reach,
        'env_vals': env_vals,
        'sys_vals': sys_vals,
        'k': k,
        'b': b,
        'ucw_states': ucw_states,
    }

    n_vars = len(sel) + len(out) + len(lam) + len(reach)

    return solver, var_info, n_vars


def extract_controller(solver, var_info, env_vars, sys_vars):
    """Extract controller from SAT SMT model."""
    model = solver.model()
    if model is None:
        return None, None

    out = var_info['out']
    lam = var_info['lam']
    env_vals = var_info['env_vals']
    k = var_info['k']
    b = var_info['b']
    ucw_states = var_info['ucw_states']

    # Extract transition function from sel variables
    transitions = {}
    for c in range(k):
        for e_idx, e_val in enumerate(env_vals):
            # Find which next state is selected
            next_c = 0
            for c_next in range(k):
                sel_name = f'sel_{c}_{e_idx}_{c_next}'
                val = model.get(sel_name, False)
                if val:
                    next_c = c_next
                    break

            # Get output
            sys_output = set()
            for v in sorted(sys_vars):
                out_name = f'out_{c}_{e_idx}_{v}'
                val = model.get(out_name, False)
                if val:
                    sys_output.add(v)

            transitions[(c, e_val)] = (next_c, frozenset(sys_output))

    controller = Controller(
        n_states=k,
        initial=0,
        transitions=transitions,
        env_vars=set(env_vars),
        sys_vars=set(sys_vars),
    )

    # Extract annotation
    ann_values = {}
    for q in ucw_states:
        for c in range(k):
            lam_name = f'lam_{q}_{c}'
            val = model.get(lam_name, 0)
            if isinstance(val, float):
                val = int(val)
            ann_values[(q, c)] = val

    annotation = Annotation(values=ann_values, max_bound=b)

    return controller, annotation


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def bounded_synthesize(spec, env_vars, sys_vars, max_states=8, max_bound=None):
    """SMT-based bounded synthesis.

    Searches for a k-state controller satisfying the LTL spec,
    trying k = 1, 2, ..., max_states.

    Args:
        spec: LTL formula
        env_vars: set of environment variable names
        sys_vars: set of system variable names
        max_states: maximum controller states to try
        max_bound: maximum annotation bound (default: |UCW states| * max_states)

    Returns:
        SynthResult
    """
    # Quick check for trivial specs
    quick = _quick_check(spec, env_vars, sys_vars)
    if quick is not None:
        return quick

    # Build UCW
    ucw = ucw_from_ltl(spec)

    if not ucw.states:
        # Empty automaton = trivially realizable
        return SynthResult(
            verdict=SynthVerdict.REALIZABLE,
            controller=_trivial_controller(env_vars, sys_vars),
            n_states=1,
            ucw_states=0,
            method="bounded_synthesis_trivial",
        )

    if max_bound is None:
        max_bound = len(ucw.states) * max_states + 1

    for k in range(1, max_states + 1):
        for b in range(0, min(max_bound + 1, len(ucw.states) * k + 2)):
            solver, var_info, n_vars = encode_bounded(ucw, env_vars, sys_vars, k, b)

            result = solver.check()

            if result == SMTResult.SAT:
                controller, annotation = extract_controller(solver, var_info, env_vars, sys_vars)
                return SynthResult(
                    verdict=SynthVerdict.REALIZABLE,
                    controller=controller,
                    annotation=annotation,
                    n_states=k,
                    bound=b,
                    ucw_states=len(ucw.states),
                    smt_vars=n_vars,
                    method="bounded_synthesis",
                    details={'max_tried_k': k, 'max_tried_b': b},
                )
        # All bounds failed for this k, try next k

    return SynthResult(
        verdict=SynthVerdict.UNKNOWN,
        ucw_states=len(ucw.states),
        method="bounded_synthesis",
        details={'max_tried_k': max_states, 'max_tried_b': max_bound},
    )


def synthesize_safety(bad_condition, env_vars, sys_vars, max_states=8):
    """Synthesize controller for safety spec G(!bad_condition).

    For safety specs, the annotation bound is always 0 (no rejecting visits allowed).
    """
    spec = Globally(LTLNot(bad_condition))
    return bounded_synthesize(spec, env_vars, sys_vars, max_states=max_states, max_bound=0)


def synthesize_liveness(condition, env_vars, sys_vars, max_states=8):
    """Synthesize controller for liveness spec GF(condition)."""
    spec = Globally(Finally(condition))
    return bounded_synthesize(spec, env_vars, sys_vars, max_states=max_states)


def synthesize_response(trigger, response, env_vars, sys_vars, max_states=8):
    """Synthesize controller for response spec G(trigger -> F(response))."""
    spec = Globally(LTLImplies(trigger, Finally(response)))
    return bounded_synthesize(spec, env_vars, sys_vars, max_states=max_states)


def synthesize_assume_guarantee(assumptions, guarantees, env_vars, sys_vars, max_states=8):
    """Synthesize controller under assume-guarantee form."""
    spec = LTLImplies(assumptions, guarantees)
    return bounded_synthesize(spec, env_vars, sys_vars, max_states=max_states)


# ---------------------------------------------------------------------------
# Minimum controller search
# ---------------------------------------------------------------------------

def find_minimum_controller(spec, env_vars, sys_vars, max_states=8, max_bound=None):
    """Find the minimum-state controller for the spec.

    Tries k = 1, 2, ... and returns as soon as a controller is found.
    The first k that works is the minimum.
    """
    ucw = ucw_from_ltl(spec)

    if not ucw.states:
        ctrl = _trivial_controller(env_vars, sys_vars)
        return SynthResult(
            verdict=SynthVerdict.REALIZABLE,
            controller=ctrl,
            n_states=1,
            ucw_states=0,
            method="minimum_controller",
        )

    if max_bound is None:
        max_bound = len(ucw.states) * max_states + 1

    for k in range(1, max_states + 1):
        for b in range(0, min(max_bound + 1, len(ucw.states) * k + 2)):
            solver, var_info, n_vars = encode_bounded(ucw, env_vars, sys_vars, k, b)
            result = solver.check()

            if result == SMTResult.SAT:
                controller, annotation = extract_controller(solver, var_info, env_vars, sys_vars)
                return SynthResult(
                    verdict=SynthVerdict.REALIZABLE,
                    controller=controller,
                    annotation=annotation,
                    n_states=k,
                    bound=b,
                    ucw_states=len(ucw.states),
                    smt_vars=n_vars,
                    method="minimum_controller",
                    details={'min_states': k},
                )

    return SynthResult(
        verdict=SynthVerdict.UNKNOWN,
        ucw_states=len(ucw.states),
        method="minimum_controller",
        details={'max_tried': max_states},
    )


# ---------------------------------------------------------------------------
# Synthesis with extra constraints
# ---------------------------------------------------------------------------

def synthesize_with_constraints(spec, env_vars, sys_vars, constraint_fn,
                                 max_states=8, max_bound=None):
    """Synthesize with additional constraints on the controller.

    Args:
        constraint_fn: function(solver, var_info) -> None
            Adds extra SMT constraints to the encoding.
            Can access tau, out, lam variables through var_info.
    """
    ucw = ucw_from_ltl(spec)

    if max_bound is None:
        max_bound = len(ucw.states) * max_states + 1

    for k in range(1, max_states + 1):
        for b in range(0, min(max_bound + 1, len(ucw.states) * k + 2)):
            solver, var_info, n_vars = encode_bounded(ucw, env_vars, sys_vars, k, b)

            # Apply extra constraints
            constraint_fn(solver, var_info)

            result = solver.check()

            if result == SMTResult.SAT:
                controller, annotation = extract_controller(solver, var_info, env_vars, sys_vars)
                return SynthResult(
                    verdict=SynthVerdict.REALIZABLE,
                    controller=controller,
                    annotation=annotation,
                    n_states=k,
                    bound=b,
                    ucw_states=len(ucw.states),
                    smt_vars=n_vars,
                    method="constrained_synthesis",
                )

    return SynthResult(
        verdict=SynthVerdict.UNKNOWN,
        ucw_states=len(ucw.states),
        method="constrained_synthesis",
    )


# ---------------------------------------------------------------------------
# Annotation verification
# ---------------------------------------------------------------------------

def verify_annotation(controller, ucw, annotation):
    """Verify that an annotation witnesses bounded co-Buchi acceptance.

    Only checks reachable product states -- unreachable states may have
    unsatisfiable annotation constraints (e.g., absorbing rejecting sinks).

    Returns (valid, violations) where violations is a list of problematic transitions.
    """
    # Compute reachable (q, c) pairs
    reachable = set()
    env_vals = _all_valuations(controller.env_vars)

    # Initial: (q0, 0) for each initial UCW state
    frontier = []
    for q in ucw.initial:
        reachable.add((q, 0))
        frontier.append((q, 0))

    while frontier:
        q, c = frontier.pop()
        trans_q = ucw.transitions.get(q, [])
        for e_val in env_vals:
            result = controller.step(c, e_val)
            if result is None:
                continue
            next_c, sys_output = result
            combined = e_val | sys_output
            for label, q_next in trans_q:
                if _label_matches(label, combined):
                    if (q_next, next_c) not in reachable:
                        reachable.add((q_next, next_c))
                        frontier.append((q_next, next_c))

    # Check annotation only for reachable states
    violations = []
    for q, c in sorted(reachable):
        trans_q = ucw.transitions.get(q, [])
        for e_val in env_vals:
            result = controller.step(c, e_val)
            if result is None:
                continue
            next_c, sys_output = result
            combined = e_val | sys_output

            for label, q_next in trans_q:
                if not _label_matches(label, combined):
                    continue
                if (q_next, next_c) not in reachable:
                    continue

                lam_cur = annotation.values.get((q, c), 0)
                lam_next = annotation.values.get((q_next, next_c), 0)

                if q_next in ucw.rejecting:
                    if lam_next >= lam_cur:
                        violations.append({
                            'type': 'rejecting_not_decreasing',
                            'q': q, 'c': c, 'q_next': q_next, 'c_next': next_c,
                            'lam_cur': lam_cur, 'lam_next': lam_next,
                            'env': e_val, 'sys': sys_output,
                        })
                else:
                    if lam_next > lam_cur:
                        violations.append({
                            'type': 'non_rejecting_increasing',
                            'q': q, 'c': c, 'q_next': q_next, 'c_next': next_c,
                            'lam_cur': lam_cur, 'lam_next': lam_next,
                            'env': e_val, 'sys': sys_output,
                        })

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Controller verification (bounded model checking)
# ---------------------------------------------------------------------------

def verify_controller(controller, spec, env_vars, sys_vars, max_depth=50):
    """Bounded verification of controller against LTL spec.

    Simulates all possible environment input sequences up to max_depth
    and checks the spec on resulting traces.
    """
    env_vals = _all_valuations(env_vars)

    # BFS over (depth, ctrl_state, trace) -- check spec prefix on traces
    from collections import deque

    violations = []
    queue = deque()
    queue.append((controller.initial, []))

    checked = 0
    max_checks = 10000  # limit to avoid explosion

    while queue and checked < max_checks:
        state, trace = queue.popleft()
        checked += 1

        if len(trace) >= max_depth:
            continue

        for e_val in env_vals:
            result = controller.step(state, e_val)
            if result is None:
                continue
            next_state, sys_output = result
            step_val = e_val | sys_output
            new_trace = trace + [step_val]

            # Check safety properties on each step
            violation = _check_trace_prefix(spec, new_trace)
            if violation:
                violations.append({
                    'trace': new_trace,
                    'violation': violation,
                })
                if len(violations) >= 5:
                    return False, violations

            if len(new_trace) < max_depth:
                queue.append((next_state, new_trace))

    return len(violations) == 0, violations


def _check_trace_prefix(spec, trace):
    """Check if a trace prefix violates a safety property.

    Only checks G(p) style properties -- returns violation description
    if the invariant is broken at any step.
    """
    if spec.op == LTLOp.G and spec.left is not None:
        inner = spec.left
        for i, step in enumerate(trace):
            if not _eval_propositional(inner, step):
                return f"G-property violated at step {i}"
    return None


def _eval_propositional(formula, valuation):
    """Evaluate a propositional LTL formula on a valuation."""
    if formula.op == LTLOp.TRUE:
        return True
    if formula.op == LTLOp.FALSE:
        return False
    if formula.op == LTLOp.ATOM:
        return formula.name in valuation
    if formula.op == LTLOp.NOT:
        return not _eval_propositional(formula.left, valuation)
    if formula.op == LTLOp.AND:
        return (_eval_propositional(formula.left, valuation) and
                _eval_propositional(formula.right, valuation))
    if formula.op == LTLOp.OR:
        return (_eval_propositional(formula.left, valuation) or
                _eval_propositional(formula.right, valuation))
    if formula.op == LTLOp.IMPLIES:
        return (not _eval_propositional(formula.left, valuation) or
                _eval_propositional(formula.right, valuation))
    if formula.op == LTLOp.IFF:
        a = _eval_propositional(formula.left, valuation)
        b = _eval_propositional(formula.right, valuation)
        return a == b
    # Temporal operators can't be evaluated propositionally
    return True


# ---------------------------------------------------------------------------
# Comparison with game-based approach
# ---------------------------------------------------------------------------

def compare_with_game(spec, env_vars, sys_vars, max_states=4):
    """Compare SMT-based synthesis with V188 game-based approach.

    Returns dict with both results and comparison.
    """
    # SMT-based
    smt_result = bounded_synthesize(spec, env_vars, sys_vars, max_states=max_states)

    # Game-based (V188)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V188_bounded_realizability'))
        from bounded_realizability import check_realizable
        game_result = check_realizable(spec, env_vars, sys_vars)
        game_verdict = game_result.verdict.value
    except Exception as e:
        game_result = None
        game_verdict = f"error: {e}"

    agreement = False
    if game_result is not None:
        smt_real = smt_result.verdict == SynthVerdict.REALIZABLE
        game_real = game_verdict == "REALIZABLE"
        agreement = (smt_real == game_real)

    return {
        'smt_verdict': smt_result.verdict.value,
        'smt_states': smt_result.n_states,
        'smt_bound': smt_result.bound,
        'game_verdict': game_verdict,
        'agreement': agreement,
        'smt_result': smt_result,
        'game_result': game_result,
    }


# ---------------------------------------------------------------------------
# UCW analysis
# ---------------------------------------------------------------------------

def ucw_statistics(ucw):
    """Compute statistics about a UCW."""
    n_transitions = sum(len(ts) for ts in ucw.transitions.values())
    return {
        'states': len(ucw.states),
        'initial': len(ucw.initial),
        'transitions': n_transitions,
        'rejecting': len(ucw.rejecting),
        'non_rejecting': len(ucw.states) - len(ucw.rejecting),
        'ap': len(ucw.ap),
    }


def synthesis_summary(result):
    """Human-readable summary of synthesis result."""
    lines = [f"Bounded Synthesis: {result.verdict.value}"]
    if result.verdict == SynthVerdict.REALIZABLE:
        lines.append(f"  Controller states: {result.n_states}")
        lines.append(f"  Annotation bound: {result.bound}")
        lines.append(f"  UCW states: {result.ucw_states}")
        if result.controller:
            lines.append(f"  Transitions: {len(result.controller.transitions)}")
    elif result.verdict == SynthVerdict.UNKNOWN:
        lines.append(f"  UCW states: {result.ucw_states}")
        lines.append(f"  Exhausted search space")
    lines.append(f"  Method: {result.method}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_check(spec, env_vars, sys_vars):
    """Quick check for trivially realizable/unrealizable specs."""
    if spec.op == LTLOp.TRUE:
        return SynthResult(
            verdict=SynthVerdict.REALIZABLE,
            controller=_trivial_controller(env_vars, sys_vars),
            n_states=1,
            method="quick_check_true",
        )
    if spec.op == LTLOp.FALSE:
        return SynthResult(
            verdict=SynthVerdict.UNREALIZABLE,
            method="quick_check_false",
        )
    # G(true) is trivially realizable
    if spec.op == LTLOp.G and spec.left is not None and spec.left.op == LTLOp.TRUE:
        return SynthResult(
            verdict=SynthVerdict.REALIZABLE,
            controller=_trivial_controller(env_vars, sys_vars),
            n_states=1,
            method="quick_check_g_true",
        )
    # G(false) is unrealizable
    if spec.op == LTLOp.G and spec.left is not None and spec.left.op == LTLOp.FALSE:
        return SynthResult(
            verdict=SynthVerdict.UNREALIZABLE,
            method="quick_check_g_false",
        )
    return None


def _trivial_controller(env_vars, sys_vars):
    """Create a 1-state controller that outputs empty set."""
    env_vals = _all_valuations(env_vars)
    transitions = {}
    for e_val in env_vals:
        transitions[(0, e_val)] = (0, frozenset())
    return Controller(
        n_states=1,
        initial=0,
        transitions=transitions,
        env_vars=set(env_vars),
        sys_vars=set(sys_vars),
    )


# ---------------------------------------------------------------------------
# Convenience: controller to compatible formats
# ---------------------------------------------------------------------------

def controller_to_dict(controller):
    """Convert controller to serializable dict."""
    trans = {}
    for (c, e_val), (nc, s_val) in controller.transitions.items():
        key = f"{c}:{','.join(sorted(e_val)) if e_val else '_'}"
        trans[key] = {
            'next': nc,
            'output': sorted(s_val),
        }
    return {
        'n_states': controller.n_states,
        'initial': controller.initial,
        'env_vars': sorted(controller.env_vars),
        'sys_vars': sorted(controller.sys_vars),
        'transitions': trans,
    }


def controller_statistics(controller):
    """Compute statistics about a controller."""
    n_trans = len(controller.transitions)
    outputs = set()
    for _, (_, s_val) in controller.transitions.items():
        outputs.add(s_val)
    states_used = {controller.initial}
    for (c, _), (nc, _) in controller.transitions.items():
        states_used.add(c)
        states_used.add(nc)
    return {
        'n_states': controller.n_states,
        'states_used': len(states_used),
        'transitions': n_trans,
        'distinct_outputs': len(outputs),
        'env_vars': len(controller.env_vars),
        'sys_vars': len(controller.sys_vars),
    }
